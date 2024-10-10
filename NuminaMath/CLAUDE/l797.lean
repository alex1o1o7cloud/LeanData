import Mathlib

namespace selection_with_both_genders_l797_79728

/-- The number of ways to select 3 people from a group of 4 male students and 6 female students, 
    such that both male and female students are included. -/
theorem selection_with_both_genders (male_count : Nat) (female_count : Nat) : 
  male_count = 4 → female_count = 6 → 
  (Nat.choose (male_count + female_count) 3 - 
   Nat.choose male_count 3 - 
   Nat.choose female_count 3) = 96 := by
  sorry

end selection_with_both_genders_l797_79728


namespace ceiling_product_equation_l797_79792

theorem ceiling_product_equation : ∃! x : ℝ, ⌈x⌉ * x = 168 ∧ x = 168 / 13 := by
  sorry

end ceiling_product_equation_l797_79792


namespace russian_players_pairing_probability_l797_79715

/-- The probability of all Russian players being paired with each other in a tennis tournament -/
theorem russian_players_pairing_probability
  (total_players : ℕ)
  (russian_players : ℕ)
  (h1 : total_players = 10)
  (h2 : russian_players = 4)
  (h3 : russian_players ≤ total_players) :
  (russian_players.choose 2 : ℚ) / (total_players.choose 2) = 1 / 21 :=
sorry

end russian_players_pairing_probability_l797_79715


namespace regression_line_not_exact_l797_79757

-- Define the regression line equation
def regression_line (x : ℝ) : ℝ := 0.5 * x - 85

-- Define the specific x value
def x_value : ℝ := 200

-- Theorem statement
theorem regression_line_not_exact (ε : ℝ) (h : ε > 0) :
  ∃ y : ℝ, y ≠ 15 ∧ |y - regression_line x_value| < ε :=
sorry

end regression_line_not_exact_l797_79757


namespace parabola_vertex_l797_79772

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = -2 * (x - 3)^2 + 4

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (3, 4)

/-- Theorem: The vertex of the parabola y = -2(x-3)^2 + 4 is (3, 4) -/
theorem parabola_vertex : 
  ∀ x y : ℝ, parabola x y → (x, y) = vertex :=
sorry

end parabola_vertex_l797_79772


namespace pop_spending_proof_l797_79783

/-- The amount of money Pop spent on cereal -/
def pop_spending : ℝ := 15

/-- The amount of money Crackle spent on cereal -/
def crackle_spending : ℝ := 3 * pop_spending

/-- The amount of money Snap spent on cereal -/
def snap_spending : ℝ := 2 * crackle_spending

/-- The total amount spent on cereal -/
def total_spending : ℝ := 150

theorem pop_spending_proof :
  pop_spending + crackle_spending + snap_spending = total_spending ∧
  pop_spending = 15 := by
  sorry

end pop_spending_proof_l797_79783


namespace inequality_proof_l797_79700

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*a*c)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 :=
by sorry

end inequality_proof_l797_79700


namespace solution_is_axes_l797_79729

def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - p.2)^2 = p.1^2 + p.2^2}

def x_axis : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 0}

def y_axis : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0}

theorem solution_is_axes : solution_set = x_axis ∪ y_axis := by
  sorry

end solution_is_axes_l797_79729


namespace trigonometric_product_equals_one_l797_79709

theorem trigonometric_product_equals_one :
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1 := by
sorry

end trigonometric_product_equals_one_l797_79709


namespace congruence_problem_l797_79726

theorem congruence_problem : ∃ n : ℤ, 0 ≤ n ∧ n < 9 ∧ -1234 ≡ n [ZMOD 9] ∧ n = 8 := by
  sorry

end congruence_problem_l797_79726


namespace cubic_function_root_condition_l797_79769

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 6 * x

theorem cubic_function_root_condition (a : ℝ) :
  (∃! x₀ : ℝ, f a x₀ = 0 ∧ x₀ < 0) → a > 2 := by
  sorry


end cubic_function_root_condition_l797_79769


namespace impossibleEggDivision_l797_79790

/-- Represents the number of eggs of each type -/
structure EggCounts where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- Represents the ratio of eggs in each group -/
structure EggRatio where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- Function to check if it's possible to divide eggs into groups with a given ratio -/
def canDivideEggs (counts : EggCounts) (ratio : EggRatio) (numGroups : ℕ) : Prop :=
  ∃ (groupSize : ℕ),
    counts.typeA = numGroups * groupSize * ratio.typeA ∧
    counts.typeB = numGroups * groupSize * ratio.typeB ∧
    counts.typeC = numGroups * groupSize * ratio.typeC

/-- Theorem stating that it's impossible to divide the given eggs into 5 groups with the specified ratio -/
theorem impossibleEggDivision : 
  let counts : EggCounts := ⟨15, 12, 8⟩
  let ratio : EggRatio := ⟨2, 3, 1⟩
  let numGroups : ℕ := 5
  ¬(canDivideEggs counts ratio numGroups) := by
  sorry


end impossibleEggDivision_l797_79790


namespace probability_total_gt_seven_is_five_twelfths_l797_79733

/-- The number of faces on each die -/
def faces : ℕ := 6

/-- The total number of possible outcomes when throwing two dice -/
def total_outcomes : ℕ := faces * faces

/-- The number of outcomes that result in a total greater than 7 -/
def favorable_outcomes : ℕ := 15

/-- The probability of getting a total more than 7 when throwing two 6-sided dice -/
def probability_total_gt_seven : ℚ := favorable_outcomes / total_outcomes

theorem probability_total_gt_seven_is_five_twelfths :
  probability_total_gt_seven = 5 / 12 := by
  sorry

end probability_total_gt_seven_is_five_twelfths_l797_79733


namespace f_max_at_seven_l797_79707

/-- The quadratic function we're analyzing -/
def f (y : ℝ) : ℝ := y^2 - 14*y + 24

/-- The theorem stating that f achieves its maximum at y = 7 -/
theorem f_max_at_seven :
  ∀ y : ℝ, f y ≤ f 7 := by
  sorry

end f_max_at_seven_l797_79707


namespace police_emergency_number_has_large_prime_divisor_l797_79754

/-- A police emergency number is a positive integer that ends with 133 in decimal representation. -/
def is_police_emergency_number (n : ℕ) : Prop :=
  n > 0 ∧ n % 1000 = 133

/-- Every police emergency number has a prime divisor greater than 7. -/
theorem police_emergency_number_has_large_prime_divisor (n : ℕ) :
  is_police_emergency_number n → ∃ p : ℕ, p.Prime ∧ p > 7 ∧ p ∣ n :=
by sorry

end police_emergency_number_has_large_prime_divisor_l797_79754


namespace polynomial_divisibility_l797_79719

theorem polynomial_divisibility (P : ℤ → ℤ) (n : ℤ) 
  (h1 : ∃ k1 : ℤ, P n = 3 * k1)
  (h2 : ∃ k2 : ℤ, P (n + 1) = 3 * k2)
  (h3 : ∃ k3 : ℤ, P (n + 2) = 3 * k3)
  (h_poly : ∀ x y : ℤ, ∃ a b c : ℤ, P (x + y) = P x + a * y + b * y^2 + c * y^3) :
  ∀ m : ℤ, ∃ k : ℤ, P m = 3 * k :=
by sorry

end polynomial_divisibility_l797_79719


namespace function_difference_l797_79735

theorem function_difference (f : ℕ+ → ℕ+) 
  (h_mono : ∀ m n : ℕ+, m < n → f m < f n)
  (h_comp : ∀ n : ℕ+, f (f n) = 3 * n) :
  f 2202 - f 2022 = 510 := by
sorry

end function_difference_l797_79735


namespace omega_squared_plus_omega_plus_one_eq_zero_l797_79731

theorem omega_squared_plus_omega_plus_one_eq_zero :
  let ω : ℂ := -1/2 + (Real.sqrt 3 / 2) * Complex.I
  ω^2 + ω + 1 = 0 := by
  sorry

end omega_squared_plus_omega_plus_one_eq_zero_l797_79731


namespace jason_tom_blue_difference_l797_79758

/-- Represents the number of marbles a person has -/
structure MarbleCount where
  blue : ℕ
  red : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the difference in blue marbles between two MarbleCounts -/
def blueDifference (a b : MarbleCount) : ℕ :=
  if a.blue ≥ b.blue then a.blue - b.blue else b.blue - a.blue

theorem jason_tom_blue_difference :
  let jason : MarbleCount := { blue := 44, red := 16, green := 8, yellow := 0 }
  let tom : MarbleCount := { blue := 24, red := 0, green := 7, yellow := 10 }
  blueDifference jason tom = 20 := by
  sorry

end jason_tom_blue_difference_l797_79758


namespace max_candy_leftover_l797_79716

theorem max_candy_leftover (x : ℕ) (h : x > 11) : 
  ∃ (q r : ℕ), x = 11 * q + r ∧ r > 0 ∧ r ≤ 10 :=
by sorry

end max_candy_leftover_l797_79716


namespace shiela_colors_l797_79703

theorem shiela_colors (total_blocks : ℕ) (blocks_per_color : ℕ) (h1 : total_blocks = 49) (h2 : blocks_per_color = 7) :
  total_blocks / blocks_per_color = 7 :=
by sorry

end shiela_colors_l797_79703


namespace trapezoid_height_is_four_l797_79740

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  -- The length of the midline
  midline : ℝ
  -- The lengths of the bases
  base1 : ℝ
  base2 : ℝ
  -- The height of the trapezoid
  height : ℝ
  -- Condition: The trapezoid has an inscribed circle
  has_inscribed_circle : Prop
  -- Condition: The midline is the average of the bases
  midline_avg : midline = (base1 + base2) / 2
  -- Condition: The area ratio of the parts divided by the midline
  area_ratio : (base1 - midline) / (base2 - midline) = 7 / 13

/-- The main theorem about the height of the trapezoid -/
theorem trapezoid_height_is_four (t : IsoscelesTrapezoid) 
  (h_midline : t.midline = 5) : t.height = 4 := by
  sorry


end trapezoid_height_is_four_l797_79740


namespace total_necklaces_l797_79708

def necklaces_problem (boudreaux rhonda latch cecilia : ℕ) : Prop :=
  boudreaux = 12 ∧
  rhonda = boudreaux / 2 ∧
  latch = 3 * rhonda - 4 ∧
  cecilia = latch + 3 ∧
  boudreaux + rhonda + latch + cecilia = 49

theorem total_necklaces : ∃ (boudreaux rhonda latch cecilia : ℕ), 
  necklaces_problem boudreaux rhonda latch cecilia :=
by
  sorry

end total_necklaces_l797_79708


namespace garden_perimeter_l797_79739

/-- 
A rectangular garden has a diagonal of 34 meters and an area of 240 square meters.
This theorem proves that the perimeter of such a garden is 80 meters.
-/
theorem garden_perimeter : 
  ∀ (a b : ℝ), 
  a > 0 → b > 0 →  -- Ensure positive dimensions
  a * b = 240 →    -- Area condition
  a^2 + b^2 = 34^2 →  -- Diagonal condition
  2 * (a + b) = 80 :=  -- Perimeter calculation
by
  sorry

#check garden_perimeter

end garden_perimeter_l797_79739


namespace cousins_ages_sum_l797_79742

theorem cousins_ages_sum (a b c d : ℕ) : 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 →  -- single-digit ages
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →  -- distinct ages
  (a * b = 20 ∧ c * d = 36) ∨ (a * c = 20 ∧ b * d = 36) ∨ 
  (a * d = 20 ∧ b * c = 36) →  -- product conditions
  a + b + c + d = 21 := by
sorry

end cousins_ages_sum_l797_79742


namespace divisibility_problem_l797_79789

theorem divisibility_problem (a : ℤ) : 
  0 ≤ a ∧ a < 13 → (12^20 + a) % 13 = 0 → a = 12 := by
  sorry

end divisibility_problem_l797_79789


namespace calculate_expression_l797_79712

theorem calculate_expression : -1^4 - 1/4 * (2 - (-3)^2) = 3/4 := by
  sorry

end calculate_expression_l797_79712


namespace estimate_pi_l797_79705

theorem estimate_pi (total_points : ℕ) (circle_points : ℕ) 
  (h1 : total_points = 1000) 
  (h2 : circle_points = 780) : 
  (circle_points : ℚ) / total_points * 4 = 78 / 25 := by
  sorry

end estimate_pi_l797_79705


namespace strawberry_sales_l797_79737

/-- The number of pints of strawberries sold by a supermarket -/
def pints_sold : ℕ := 54

/-- The revenue from selling strawberries on sale -/
def sale_revenue : ℕ := 216

/-- The revenue that would have been made without the sale -/
def non_sale_revenue : ℕ := 324

/-- The price difference between non-sale and sale price per pint -/
def price_difference : ℕ := 2

theorem strawberry_sales :
  ∃ (sale_price : ℚ),
    sale_price > 0 ∧
    sale_price * pints_sold = sale_revenue ∧
    (sale_price + price_difference) * pints_sold = non_sale_revenue :=
by sorry

end strawberry_sales_l797_79737


namespace impossibleColoring_l797_79717

def Color := Bool

def isRed (c : Color) : Prop := c = true
def isBlue (c : Color) : Prop := c = false

theorem impossibleColoring :
  ¬∃(f : ℕ → Color),
    (∀ n : ℕ, n > 1000 → (isRed (f n) ∨ isBlue (f n))) ∧
    (∀ m n : ℕ, m > 1000 → n > 1000 → m ≠ n → isRed (f m) → isRed (f n) → isBlue (f (m * n))) ∧
    (∀ m n : ℕ, m > 1000 → n > 1000 → m = n + 1 → ¬(isBlue (f m) ∧ isBlue (f n))) :=
by
  sorry

end impossibleColoring_l797_79717


namespace selection_schemes_correct_l797_79794

/-- The number of ways to select 4 students from 4 boys and 2 girls, with at least 1 girl in the group -/
def selection_schemes (num_boys num_girls group_size : ℕ) : ℕ :=
  Nat.choose (num_boys + num_girls) group_size - Nat.choose num_boys group_size

theorem selection_schemes_correct :
  selection_schemes 4 2 4 = 14 := by
  sorry

#eval selection_schemes 4 2 4

end selection_schemes_correct_l797_79794


namespace original_profit_margin_is_15_percent_l797_79752

/-- Represents the profit margin as a real number between 0 and 1 -/
def ProfitMargin : Type := { x : ℝ // 0 ≤ x ∧ x ≤ 1 }

/-- The decrease in purchase price -/
def price_decrease : ℝ := 0.08

/-- The increase in profit margin -/
def margin_increase : ℝ := 0.10

/-- The original profit margin -/
def original_margin : ProfitMargin := ⟨0.15, by sorry⟩

theorem original_profit_margin_is_15_percent :
  ∀ (initial_price : ℝ),
  initial_price > 0 →
  let new_price := initial_price * (1 - price_decrease)
  let new_margin := original_margin.val + margin_increase
  let original_profit := initial_price * original_margin.val
  let new_profit := new_price * new_margin
  original_profit = new_profit := by sorry

end original_profit_margin_is_15_percent_l797_79752


namespace log_one_fifth_25_l797_79767

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_one_fifth_25 : log (1/5) 25 = -2 := by
  sorry

end log_one_fifth_25_l797_79767


namespace uncle_jerry_tomatoes_l797_79738

def tomatoes_problem (yesterday today total : ℕ) : Prop :=
  (yesterday = 120) ∧
  (today = yesterday + 50) ∧
  (total = yesterday + today)

theorem uncle_jerry_tomatoes : ∃ yesterday today total : ℕ,
  tomatoes_problem yesterday today total ∧ total = 290 := by sorry

end uncle_jerry_tomatoes_l797_79738


namespace circle_properties_l797_79723

-- Define the circle C
def circle_C (D E F : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + D * p.1 + E * p.2 + F = 0}

-- Define the line l
def line_l (D E F : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | D * p.1 + E * p.2 + F = 0}

-- Define the circle M
def circle_M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

theorem circle_properties (D E F : ℝ) 
    (h1 : D^2 + E^2 = F^2) 
    (h2 : F > 0) : 
  F > 4 ∧ 
  (let d := |F - 2| / 2
   let r := Real.sqrt (F^2 - 4*F) / 2
   d^2 - r^2 = 1) ∧
  (∃ M : Set (ℝ × ℝ), M = circle_M ∧ 
    (∀ p ∈ M, p ∈ line_l D E F → False) ∧
    (∀ p ∈ M, p ∈ circle_C D E F → False)) :=
by sorry

end circle_properties_l797_79723


namespace right_triangle_trig_l797_79721

theorem right_triangle_trig (A B C : Real) (h1 : A + B + C = Real.pi) 
  (h2 : C = Real.pi / 2) (h3 : Real.sin A = 2 / 3) : Real.cos B = 2 / 3 := by
  sorry

end right_triangle_trig_l797_79721


namespace divisible_by_91_l797_79770

theorem divisible_by_91 (n : ℕ) : ∃ k : ℤ, 9^(n+2) + 10^(2*n+1) = 91 * k := by
  sorry

end divisible_by_91_l797_79770


namespace unique_prime_sum_and_difference_l797_79710

theorem unique_prime_sum_and_difference : 
  ∃! p : ℕ, 
    Prime p ∧ 
    (∃ q₁ q₂ : ℕ, Prime q₁ ∧ Prime q₂ ∧ p = q₁ + q₂) ∧
    (∃ q₃ q₄ : ℕ, Prime q₃ ∧ Prime q₄ ∧ q₃ > q₄ ∧ p = q₃ - q₄) ∧
    p = 5 :=
by sorry

end unique_prime_sum_and_difference_l797_79710


namespace ellipse_quadrant_area_diff_zero_l797_79775

/-- Definition of an ellipse with center (h, k) and parameters a, b, c -/
def Ellipse (h k a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - h)^2 / a + (p.2 - k)^2 / b = c}

/-- Areas of the ellipse in each quadrant -/
def QuadrantAreas (e : Set (ℝ × ℝ)) : ℝ × ℝ × ℝ × ℝ :=
  sorry

/-- Theorem: The difference of areas in alternating quadrants is zero -/
theorem ellipse_quadrant_area_diff_zero
  (h k a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  let e := Ellipse h k a b c
  let (R1, R2, R3, R4) := QuadrantAreas e
  R1 - R2 + R3 - R4 = 0 := by sorry


end ellipse_quadrant_area_diff_zero_l797_79775


namespace sophies_bakery_purchase_l797_79773

/-- Sophie's bakery purchase problem -/
theorem sophies_bakery_purchase
  (cupcake_price : ℚ)
  (cupcake_quantity : ℕ)
  (doughnut_price : ℚ)
  (doughnut_quantity : ℕ)
  (cookie_price : ℚ)
  (cookie_quantity : ℕ)
  (pie_slice_price : ℚ)
  (total_spent : ℚ)
  (h1 : cupcake_price = 2)
  (h2 : cupcake_quantity = 5)
  (h3 : doughnut_price = 1)
  (h4 : doughnut_quantity = 6)
  (h5 : cookie_price = 0.6)
  (h6 : cookie_quantity = 15)
  (h7 : pie_slice_price = 2)
  (h8 : total_spent = 33)
  : (total_spent - (cupcake_price * cupcake_quantity + doughnut_price * doughnut_quantity + cookie_price * cookie_quantity)) / pie_slice_price = 4 := by
  sorry

end sophies_bakery_purchase_l797_79773


namespace inverse_function_problem_l797_79755

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x + 2

-- Define the function f
def f (c d x : ℝ) : ℝ := c * x + d

-- State the theorem
theorem inverse_function_problem (c d : ℝ) :
  (∀ x, g x = (Function.invFun (f c d) x) - 5) →
  (Function.invFun (f c d) = Function.invFun (f c d)) →
  7 * c + 3 * d = -14/3 := by
  sorry

end inverse_function_problem_l797_79755


namespace pens_taken_after_first_month_l797_79795

theorem pens_taken_after_first_month 
  (total_pens : ℕ) 
  (pens_taken_second_month : ℕ) 
  (remaining_pens : ℕ) : 
  total_pens = 315 → 
  pens_taken_second_month = 41 → 
  remaining_pens = 237 → 
  total_pens - (total_pens - remaining_pens - pens_taken_second_month) - pens_taken_second_month = remaining_pens → 
  total_pens - remaining_pens - pens_taken_second_month = 37 := by
  sorry

end pens_taken_after_first_month_l797_79795


namespace function_inequality_implies_positive_c_l797_79701

/-- Given a function f(x) = x^2 + x + c, if f(f(x)) > x for all real x, then c > 0 -/
theorem function_inequality_implies_positive_c (c : ℝ) : 
  (∀ x : ℝ, (x^2 + x + c)^2 + (x^2 + x + c) + c > x) → c > 0 := by
  sorry

end function_inequality_implies_positive_c_l797_79701


namespace asian_games_mascot_sales_l797_79741

/-- Represents the sales situation of Asian Games mascots -/
theorem asian_games_mascot_sales 
  (initial_sales : ℕ) 
  (total_sales_next_two_days : ℕ) 
  (growth_rate : ℝ) :
  initial_sales = 5000 →
  total_sales_next_two_days = 30000 →
  (initial_sales : ℝ) * (1 + growth_rate) + (initial_sales : ℝ) * (1 + growth_rate)^2 = total_sales_next_two_days :=
by sorry

end asian_games_mascot_sales_l797_79741


namespace diane_poker_debt_l797_79760

/-- Calculates the amount owed in a poker game scenario -/
def amount_owed (initial_amount winnings total_loss : ℕ) : ℕ :=
  total_loss - (initial_amount + winnings)

/-- Theorem: In Diane's poker game scenario, she owes $50 to her friends -/
theorem diane_poker_debt : amount_owed 100 65 215 = 50 := by
  sorry

end diane_poker_debt_l797_79760


namespace cube_sphere_volume_ratio_cube_inscribed_in_sphere_volume_ratio_l797_79778

/-- The ratio of the volume of a cube inscribed in a sphere to the volume of the sphere. -/
theorem cube_sphere_volume_ratio : ℝ :=
  2 * Real.sqrt 3 / Real.pi

/-- Theorem: For a cube inscribed in a sphere, the ratio of the volume of the cube
    to the volume of the sphere is 2√3/π. -/
theorem cube_inscribed_in_sphere_volume_ratio :
  let s : ℝ := cube_side_length -- side length of the cube
  let r : ℝ := sphere_radius -- radius of the sphere
  let cube_volume : ℝ := s^3
  let sphere_volume : ℝ := (4/3) * Real.pi * r^3
  r = (Real.sqrt 3 / 2) * s → -- condition that the cube is inscribed in the sphere
  cube_volume / sphere_volume = cube_sphere_volume_ratio :=
by
  sorry

end cube_sphere_volume_ratio_cube_inscribed_in_sphere_volume_ratio_l797_79778


namespace difference_of_squares_factorization_l797_79785

-- Define the expression
def expression (a b : ℝ) : ℝ := -4 * a^2 + b^2

-- Theorem: The expression can be factored using the difference of squares formula
theorem difference_of_squares_factorization (a b : ℝ) :
  ∃ (x y : ℝ), expression a b = (x + y) * (x - y) :=
sorry

end difference_of_squares_factorization_l797_79785


namespace max_snack_bars_l797_79702

/-- Represents the number of snack bars in a pack -/
inductive PackSize
  | single : PackSize
  | twin : PackSize
  | four : PackSize

/-- Represents the price of a pack of snack bars -/
def price (p : PackSize) : ℚ :=
  match p with
  | PackSize.single => 1
  | PackSize.twin => 5/2
  | PackSize.four => 4

/-- Represents the number of snack bars in a pack -/
def bars_in_pack (p : PackSize) : ℕ :=
  match p with
  | PackSize.single => 1
  | PackSize.twin => 2
  | PackSize.four => 4

/-- The budget available for purchasing snack bars -/
def budget : ℚ := 10

/-- A purchase combination is represented as a function from PackSize to ℕ -/
def PurchaseCombination := PackSize → ℕ

/-- The total cost of a purchase combination -/
def total_cost (c : PurchaseCombination) : ℚ :=
  (c PackSize.single) * (price PackSize.single) +
  (c PackSize.twin) * (price PackSize.twin) +
  (c PackSize.four) * (price PackSize.four)

/-- The total number of snack bars in a purchase combination -/
def total_bars (c : PurchaseCombination) : ℕ :=
  (c PackSize.single) * (bars_in_pack PackSize.single) +
  (c PackSize.twin) * (bars_in_pack PackSize.twin) +
  (c PackSize.four) * (bars_in_pack PackSize.four)

/-- A purchase combination is valid if its total cost is within the budget -/
def is_valid_combination (c : PurchaseCombination) : Prop :=
  total_cost c ≤ budget

theorem max_snack_bars :
  ∃ (max : ℕ), 
    (∃ (c : PurchaseCombination), is_valid_combination c ∧ total_bars c = max) ∧
    (∀ (c : PurchaseCombination), is_valid_combination c → total_bars c ≤ max) ∧
    max = 10 :=
  sorry

end max_snack_bars_l797_79702


namespace candidate_c_wins_l797_79704

/-- Represents a candidate in the election --/
inductive Candidate
  | A
  | B
  | C
  | D
  | E

/-- Returns the vote count for a given candidate --/
def votes (c : Candidate) : Float :=
  match c with
  | Candidate.A => 4237.5
  | Candidate.B => 7298.25
  | Candidate.C => 12498.75
  | Candidate.D => 8157.5
  | Candidate.E => 3748.3

/-- Calculates the total number of votes --/
def totalVotes : Float :=
  votes Candidate.A + votes Candidate.B + votes Candidate.C + votes Candidate.D + votes Candidate.E

/-- Calculates the percentage of votes for a given candidate --/
def votePercentage (c : Candidate) : Float :=
  (votes c / totalVotes) * 100

/-- Theorem stating that Candidate C has the highest percentage of votes --/
theorem candidate_c_wins :
  ∀ c : Candidate, c ≠ Candidate.C → votePercentage Candidate.C > votePercentage c :=
by sorry

end candidate_c_wins_l797_79704


namespace sam_initial_puppies_l797_79799

/-- The number of puppies Sam gave away -/
def puppies_given : ℝ := 2.0

/-- The number of puppies Sam has now -/
def puppies_remaining : ℕ := 4

/-- The initial number of puppies Sam had -/
def initial_puppies : ℝ := puppies_given + puppies_remaining

theorem sam_initial_puppies : initial_puppies = 6.0 := by
  sorry

end sam_initial_puppies_l797_79799


namespace bug_path_tiles_l797_79797

/-- The number of tiles a bug visits when walking diagonally across a rectangular floor -/
def tiles_visited (width length : ℕ) : ℕ :=
  width + length - Nat.gcd width length

theorem bug_path_tiles : tiles_visited 12 18 = 24 := by
  sorry

end bug_path_tiles_l797_79797


namespace max_participants_l797_79768

/-- Represents the outcome of a chess game -/
inductive GameResult
| Win
| Draw
| Loss

/-- Represents a chess tournament -/
structure ChessTournament where
  participants : Nat
  results : Fin participants → Fin participants → GameResult

/-- Calculates the score of a player against two other players -/
def score (t : ChessTournament) (p1 p2 p3 : Fin t.participants) : Rat :=
  let s1 := match t.results p1 p2 with
    | GameResult.Win => 1
    | GameResult.Draw => 1/2
    | GameResult.Loss => 0
  let s2 := match t.results p1 p3 with
    | GameResult.Win => 1
    | GameResult.Draw => 1/2
    | GameResult.Loss => 0
  s1 + s2

/-- The tournament satisfies the given conditions -/
def validTournament (t : ChessTournament) : Prop :=
  (∀ p1 p2 : Fin t.participants, p1 ≠ p2 → t.results p1 p2 ≠ t.results p2 p1) ∧
  (∀ p1 p2 p3 : Fin t.participants, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 →
    (score t p1 p2 p3 = 3/2 ∨ score t p2 p1 p3 = 3/2 ∨ score t p3 p1 p2 = 3/2))

/-- The maximum number of participants in a valid tournament is 5 -/
theorem max_participants : ∀ t : ChessTournament, validTournament t → t.participants ≤ 5 := by
  sorry

end max_participants_l797_79768


namespace trajectory_of_midpoint_l797_79762

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

/-- The fixed point through which the line passes -/
def fixedPoint : ℝ × ℝ := (0, 1)

/-- The equation of the trajectory of the midpoint of the chord -/
def trajectoryEquation (x y : ℝ) : Prop := 4*x^2 - y^2 + y = 0

/-- Theorem stating that the trajectory equation is correct for the given conditions -/
theorem trajectory_of_midpoint (x y : ℝ) :
  (∃ (k : ℝ), ∃ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧
    y₁ = k*x₁ + fixedPoint.2 ∧ y₂ = k*x₂ + fixedPoint.2 ∧
    x = (x₁ + x₂)/2 ∧ y = (y₁ + y₂)/2) →
  trajectoryEquation x y :=
sorry

end trajectory_of_midpoint_l797_79762


namespace cube_sum_of_equal_ratios_l797_79725

theorem cube_sum_of_equal_ratios (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (h : (a^3 + 9) / a = (b^3 + 9) / b ∧ (b^3 + 9) / b = (c^3 + 9) / c) :
  a^3 + b^3 + c^3 = -27 := by
sorry

end cube_sum_of_equal_ratios_l797_79725


namespace race_track_circumference_difference_l797_79748

/-- The difference in circumferences of two concentric circles, where the outer circle's radius is 8 feet more than the inner circle's radius of 15 feet, is equal to 16π feet. -/
theorem race_track_circumference_difference : 
  let inner_radius : ℝ := 15
  let outer_radius : ℝ := inner_radius + 8
  let inner_circumference : ℝ := 2 * Real.pi * inner_radius
  let outer_circumference : ℝ := 2 * Real.pi * outer_radius
  outer_circumference - inner_circumference = 16 * Real.pi := by
  sorry

end race_track_circumference_difference_l797_79748


namespace linear_system_solution_l797_79753

theorem linear_system_solution :
  ∃! (x y : ℝ), (x - y = 1) ∧ (3 * x + 2 * y = 8) :=
by
  -- Proof goes here
  sorry

end linear_system_solution_l797_79753


namespace train_speed_l797_79720

/-- The speed of a train given its length and time to cross a pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 1600) (h2 : time = 40) :
  length / time = 40 := by
sorry

end train_speed_l797_79720


namespace survey_result_l797_79744

theorem survey_result (total : ℕ) (radio_dislike_percent : ℚ) (music_dislike_percent : ℚ)
  (h_total : total = 1500)
  (h_radio : radio_dislike_percent = 40 / 100)
  (h_music : music_dislike_percent = 15 / 100) :
  (total : ℚ) * radio_dislike_percent * music_dislike_percent = 90 :=
by sorry

end survey_result_l797_79744


namespace five_digit_numbers_count_correct_l797_79713

/-- Counts five-digit numbers with specific digit conditions -/
def count_five_digit_numbers : ℕ × ℕ × ℕ × ℕ × ℕ :=
  let all_identical := 9
  let two_different := 1215
  let three_different := 16200
  let four_different := 45360
  let five_different := 27216
  (all_identical, two_different, three_different, four_different, five_different)

/-- The first digit of a five-digit number cannot be zero -/
axiom first_digit_nonzero : ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 → n / 10000 ≠ 0

/-- The sum of all cases equals the total number of five-digit numbers -/
theorem five_digit_numbers_count_correct :
  let (a, b, c, d, e) := count_five_digit_numbers
  a + b + c + d + e = 90000 :=
sorry

end five_digit_numbers_count_correct_l797_79713


namespace circle_condition_l797_79746

theorem circle_condition (m : ℝ) : 
  (∃ (a b r : ℝ), ∀ (x y : ℝ), (x^2 + y^2 - x + y + m = 0) ↔ ((x - a)^2 + (y - b)^2 = r^2)) → 
  m < 1/2 := by
sorry

end circle_condition_l797_79746


namespace ellipse_point_distance_to_y_axis_l797_79777

/-- Given an ellipse with equation x²/4 + y² = 1 and foci at (-√3, 0) and (√3, 0),
    if a point M(x,y) on the ellipse satisfies the condition that the vectors from
    the foci to M are perpendicular, then the absolute value of x is 2√6/3. -/
theorem ellipse_point_distance_to_y_axis 
  (x y : ℝ) 
  (h_ellipse : x^2/4 + y^2 = 1) 
  (h_perpendicular : (x + Real.sqrt 3) * (x - Real.sqrt 3) + y * y = 0) : 
  |x| = 2 * Real.sqrt 6 / 3 := by
  sorry

end ellipse_point_distance_to_y_axis_l797_79777


namespace walking_time_calculation_l797_79780

/-- A person walks at a constant rate. They cover 36 yards in 18 minutes and have 120 feet left to walk. -/
theorem walking_time_calculation (distance_covered : ℝ) (time_taken : ℝ) (distance_left : ℝ) :
  distance_covered = 36 * 3 →
  time_taken = 18 →
  distance_left = 120 →
  distance_left / (distance_covered / time_taken) = 20 := by
  sorry

end walking_time_calculation_l797_79780


namespace scientific_notation_equality_l797_79751

theorem scientific_notation_equality : ∃ (a : ℝ) (n : ℤ), 
  0.0000023 = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 2.3 ∧ n = -6 := by
  sorry

end scientific_notation_equality_l797_79751


namespace inequality_proof_l797_79784

theorem inequality_proof (x y : ℝ) (h : 3 * x^2 + 2 * y^2 ≤ 6) : 2 * x + y ≤ Real.sqrt 11 := by
  sorry

end inequality_proof_l797_79784


namespace inverse_f_at_negative_eight_l797_79781

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - Real.log x / Real.log 3

theorem inverse_f_at_negative_eight (a : ℝ) :
  f a 1 = 1 → f a (3^9) = -8 := by sorry

end inverse_f_at_negative_eight_l797_79781


namespace sphere_in_cone_surface_area_ratio_l797_79747

theorem sphere_in_cone_surface_area_ratio (r : ℝ) (h : r > 0) :
  let cone_height : ℝ := 3 * r
  let triangle_side : ℝ := 2 * Real.sqrt 3 * r
  let sphere_surface_area : ℝ := 4 * Real.pi * r^2
  let cone_base_radius : ℝ := Real.sqrt 3 * r
  let cone_lateral_area : ℝ := Real.pi * cone_base_radius * triangle_side
  let cone_base_area : ℝ := Real.pi * cone_base_radius^2
  let cone_total_surface_area : ℝ := cone_lateral_area + cone_base_area
  cone_total_surface_area / sphere_surface_area = 9 / 4 :=
by sorry

end sphere_in_cone_surface_area_ratio_l797_79747


namespace quadratic_equation_solution_l797_79727

theorem quadratic_equation_solution (x : ℝ) :
  x^2 + 4*x - 2 = 0 ↔ x = -2 + Real.sqrt 6 ∨ x = -2 - Real.sqrt 6 := by
  sorry

end quadratic_equation_solution_l797_79727


namespace square_sum_zero_implies_both_zero_l797_79706

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end square_sum_zero_implies_both_zero_l797_79706


namespace log_z_m_value_l797_79759

theorem log_z_m_value (x y z m : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1) (hm : m > 0)
  (hlogx : Real.log m / Real.log x = 24)
  (hlogy : Real.log m / Real.log y = 40)
  (hlogxyz : Real.log m / (Real.log x + Real.log y + Real.log z) = 12) :
  Real.log m / Real.log z = 60 := by
  sorry

end log_z_m_value_l797_79759


namespace no_prime_roots_for_quadratic_l797_79761

theorem no_prime_roots_for_quadratic : 
  ¬ ∃ (k : ℤ), ∃ (p q : ℕ), 
    Prime p ∧ Prime q ∧ 
    (p : ℤ) + q = 59 ∧
    (p : ℤ) * q = k ∧
    ∀ (x : ℤ), x^2 - 59*x + k = 0 ↔ x = p ∨ x = q :=
by sorry

end no_prime_roots_for_quadratic_l797_79761


namespace butter_mixture_profit_percentage_l797_79787

/-- Calculates the profit percentage for a butter mixture sale --/
theorem butter_mixture_profit_percentage 
  (butter1_weight : ℝ) 
  (butter1_price : ℝ) 
  (butter2_weight : ℝ) 
  (butter2_price : ℝ) 
  (selling_price : ℝ) :
  butter1_weight = 54 →
  butter1_price = 150 →
  butter2_weight = 36 →
  butter2_price = 125 →
  selling_price = 196 →
  let total_cost := butter1_weight * butter1_price + butter2_weight * butter2_price
  let total_weight := butter1_weight + butter2_weight
  let selling_amount := selling_price * total_weight
  let profit := selling_amount - total_cost
  let profit_percentage := (profit / total_cost) * 100
  profit_percentage = 40 := by
sorry

end butter_mixture_profit_percentage_l797_79787


namespace complete_square_l797_79766

theorem complete_square (x : ℝ) : x^2 - 6*x + 10 = (x - 3)^2 + 1 := by
  sorry

end complete_square_l797_79766


namespace no_positive_integer_solutions_l797_79774

theorem no_positive_integer_solutions :
  ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ 3 * x^2 + 2 * x + 2 = y^2 := by
  sorry

end no_positive_integer_solutions_l797_79774


namespace xyz_value_l797_79786

theorem xyz_value (x y z : ℂ) 
  (eq1 : x * y + 5 * y = -20)
  (eq2 : y * z + 5 * z = -20)
  (eq3 : z * x + 5 * x = -20) :
  x * y * z = 100 := by
sorry

end xyz_value_l797_79786


namespace space_diagonals_count_l797_79771

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- Definition of a space diagonal in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  Nat.choose Q.vertices 2 - Q.edges - 2 * Q.quadrilateral_faces

/-- Theorem stating the number of space diagonals in the given polyhedron -/
theorem space_diagonals_count (Q : ConvexPolyhedron) 
  (h1 : Q.vertices = 30)
  (h2 : Q.edges = 58)
  (h3 : Q.faces = 36)
  (h4 : Q.triangular_faces = 26)
  (h5 : Q.quadrilateral_faces = 10)
  (h6 : Q.triangular_faces + Q.quadrilateral_faces = Q.faces) :
  space_diagonals Q = 357 := by
  sorry


end space_diagonals_count_l797_79771


namespace john_volunteer_frequency_l797_79730

/-- The number of hours John volunteers per year -/
def annual_hours : ℕ := 72

/-- The number of hours per volunteering session -/
def hours_per_session : ℕ := 3

/-- The number of months in a year -/
def months_per_year : ℕ := 12

/-- The number of times John volunteers per month -/
def volunteer_times_per_month : ℚ :=
  (annual_hours / hours_per_session : ℚ) / months_per_year

theorem john_volunteer_frequency :
  volunteer_times_per_month = 2 := by
  sorry

end john_volunteer_frequency_l797_79730


namespace caitlins_number_l797_79765

def is_two_digit_prime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ Nat.Prime n

theorem caitlins_number (a b c : ℕ) 
  (h1 : is_two_digit_prime a)
  (h2 : is_two_digit_prime b)
  (h3 : is_two_digit_prime c)
  (h4 : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h5 : 1 ≤ a + b ∧ a + b ≤ 31)
  (h6 : a + c < a + b)
  (h7 : b + c > a + b) :
  c = 11 := by
sorry

end caitlins_number_l797_79765


namespace water_break_frequency_l797_79796

theorem water_break_frequency
  (total_work_time : ℕ)
  (sitting_break_interval : ℕ)
  (water_break_excess : ℕ)
  (h1 : total_work_time = 240)
  (h2 : sitting_break_interval = 120)
  (h3 : water_break_excess = 10)
  : ℕ :=
  by
  -- Proof goes here
  sorry

#check water_break_frequency

end water_break_frequency_l797_79796


namespace new_teacher_student_ratio_l797_79749

/-- Proves that given the initial conditions, the new ratio of teachers to students is 1:25 -/
theorem new_teacher_student_ratio
  (initial_ratio : ℚ)
  (initial_teachers : ℕ)
  (student_increase : ℕ)
  (teacher_increase : ℕ)
  (new_student_ratio : ℚ)
  (h1 : initial_ratio = 50 / 1)
  (h2 : initial_teachers = 3)
  (h3 : student_increase = 50)
  (h4 : teacher_increase = 5)
  (h5 : new_student_ratio = 25 / 1) :
  (initial_teachers + teacher_increase) / (initial_ratio * initial_teachers + student_increase) = 1 / 25 := by
  sorry


end new_teacher_student_ratio_l797_79749


namespace function_range_l797_79718

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem function_range (a : ℝ) :
  (∀ x ∈ Set.Icc 0 a, f x ≤ 3) ∧
  (∃ x ∈ Set.Icc 0 a, f x = 3) ∧
  (∀ x ∈ Set.Icc 0 a, f x ≥ 2) ∧
  (∃ x ∈ Set.Icc 0 a, f x = 2) →
  a ∈ Set.Icc 1 2 :=
by sorry

end function_range_l797_79718


namespace inequality_proof_l797_79764

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y^2016 ≥ 1) :
  x^2016 + y > 1 - 1/100 := by
  sorry

end inequality_proof_l797_79764


namespace factorization_equality_l797_79782

theorem factorization_equality (x : ℝ) : x * (x - 3) + (3 - x) = (x - 3) * (x - 1) := by
  sorry

end factorization_equality_l797_79782


namespace max_square_triangle_area_ratio_l797_79734

/-- A triangle with vertices A, B, and C. -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A square with vertices X, Y, Z, and V. -/
structure Square where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  V : ℝ × ℝ

/-- The area of a triangle. -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- The area of a square. -/
def squareArea (s : Square) : ℝ := sorry

/-- Predicate to check if a point is on a line segment. -/
def isOnSegment (p : ℝ × ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if two line segments are parallel. -/
def areParallel (a1 : ℝ × ℝ) (b1 : ℝ × ℝ) (a2 : ℝ × ℝ) (b2 : ℝ × ℝ) : Prop := sorry

/-- The main theorem stating the maximum area ratio. -/
theorem max_square_triangle_area_ratio 
  (t : Triangle) 
  (s : Square) 
  (h1 : isOnSegment s.X t.A t.B)
  (h2 : isOnSegment s.Y t.B t.C)
  (h3 : isOnSegment s.Z t.C t.A)
  (h4 : isOnSegment s.V t.A t.C)
  (h5 : areParallel s.V s.Z t.A t.B) :
  squareArea s / triangleArea t ≤ 1/2 := by sorry

end max_square_triangle_area_ratio_l797_79734


namespace nigella_base_salary_l797_79788

def house_sale_income (base_salary : ℝ) (commission_rate : ℝ) (house_prices : List ℝ) : ℝ :=
  base_salary + (commission_rate * (house_prices.sum))

theorem nigella_base_salary :
  let commission_rate : ℝ := 0.02
  let house_a_price : ℝ := 60000
  let house_b_price : ℝ := 3 * house_a_price
  let house_c_price : ℝ := 2 * house_a_price - 110000
  let house_prices : List ℝ := [house_a_price, house_b_price, house_c_price]
  let total_income : ℝ := 8000
  ∃ (base_salary : ℝ), 
    house_sale_income base_salary commission_rate house_prices = total_income ∧
    base_salary = 3000 :=
by sorry

end nigella_base_salary_l797_79788


namespace domain_of_sqrt_tan_plus_sqrt_neg_cos_l797_79791

theorem domain_of_sqrt_tan_plus_sqrt_neg_cos (x : ℝ) :
  x ∈ Set.Icc 0 (2 * Real.pi) →
  (∃ y, y = Real.sqrt (Real.tan x) + Real.sqrt (-Real.cos x)) ↔
  x ∈ Set.Ico Real.pi (3 * Real.pi / 2) :=
sorry

end domain_of_sqrt_tan_plus_sqrt_neg_cos_l797_79791


namespace max_carry_weight_is_1001_l797_79736

/-- Represents the loader with a waggon and a cart -/
structure Loader :=
  (waggon_capacity : ℕ)
  (cart_capacity : ℕ)

/-- Represents the sand sacks in the storehouse -/
structure Storehouse :=
  (total_weight : ℕ)
  (max_sack_weight : ℕ)

/-- The maximum weight of sand the loader can carry -/
def max_carry_weight (l : Loader) (s : Storehouse) : ℕ :=
  l.waggon_capacity + l.cart_capacity

/-- Theorem stating the maximum weight the loader can carry -/
theorem max_carry_weight_is_1001 (l : Loader) (s : Storehouse) :
  l.waggon_capacity = 1000 →
  l.cart_capacity = 1 →
  s.total_weight > 1001 →
  s.max_sack_weight ≤ 1 →
  max_carry_weight l s = 1001 :=
by sorry

end max_carry_weight_is_1001_l797_79736


namespace min_cos_sum_sin_triangle_angles_l797_79711

theorem min_cos_sum_sin_triangle_angles (A B C : Real) : 
  A + B + C = π → 
  A > 0 → B > 0 → C > 0 →
  ∃ (m : Real), m = -2 * Real.sqrt 6 / 9 ∧ 
    ∀ (X Y Z : Real), X + Y + Z = π → X > 0 → Y > 0 → Z > 0 → 
      m ≤ Real.cos X * (Real.sin Y + Real.sin Z) :=
by sorry

end min_cos_sum_sin_triangle_angles_l797_79711


namespace red_cars_count_l797_79714

theorem red_cars_count (black_cars : ℕ) (ratio_red : ℕ) (ratio_black : ℕ) : 
  black_cars = 75 → ratio_red = 3 → ratio_black = 8 → 
  ∃ (red_cars : ℕ), red_cars * ratio_black = black_cars * ratio_red ∧ red_cars = 28 :=
by
  sorry

end red_cars_count_l797_79714


namespace point_not_on_graph_l797_79745

theorem point_not_on_graph : ¬(2 / (2 + 2) = 2 / 3) := by sorry

end point_not_on_graph_l797_79745


namespace dot_product_AB_AC_l797_79793

def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, -1)
def C : ℝ × ℝ := (3, 4)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def vector_AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem dot_product_AB_AC : dot_product vector_AB vector_AC = -2 := by sorry

end dot_product_AB_AC_l797_79793


namespace total_defective_rate_is_correct_l797_79724

/-- The defective rate of worker x -/
def worker_x_rate : ℝ := 0.005

/-- The defective rate of worker y -/
def worker_y_rate : ℝ := 0.008

/-- The fraction of products checked by worker y -/
def worker_y_fraction : ℝ := 0.8

/-- The fraction of products checked by worker x -/
def worker_x_fraction : ℝ := 1 - worker_y_fraction

/-- The total defective rate of all products -/
def total_defective_rate : ℝ := worker_x_rate * worker_x_fraction + worker_y_rate * worker_y_fraction

theorem total_defective_rate_is_correct :
  total_defective_rate = 0.0074 := by sorry

end total_defective_rate_is_correct_l797_79724


namespace basketball_team_selection_l797_79763

theorem basketball_team_selection (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) :
  total_players = 16 →
  quadruplets = 4 →
  starters = 6 →
  (Nat.choose quadruplets 3) * (Nat.choose (total_players - quadruplets) (starters - 3)) = 880 :=
by sorry

end basketball_team_selection_l797_79763


namespace tenth_term_of_sequence_l797_79779

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r ^ (n - 1)

theorem tenth_term_of_sequence (a : ℚ) (r : ℚ) :
  a = 5 → r = 3/2 → geometric_sequence a r 10 = 98415/512 := by
  sorry

end tenth_term_of_sequence_l797_79779


namespace percentage_problem_l797_79732

theorem percentage_problem (P : ℝ) : 
  0.15 * 0.30 * (P / 100) * 4000 = 90 → P = 50 := by
  sorry

end percentage_problem_l797_79732


namespace least_multiple_945_l797_79798

-- Define a function to check if a number is a multiple of 45
def isMultipleOf45 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 45 * k

-- Define a function to get the digits of a number
def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

-- Define a function to calculate the product of a list of numbers
def productOfList (l : List ℕ) : ℕ :=
  l.foldl (· * ·) 1

-- Define the main theorem
theorem least_multiple_945 :
  (isMultipleOf45 945) ∧
  (isMultipleOf45 (productOfList (digits 945))) ∧
  (∀ n : ℕ, n > 0 ∧ n < 945 →
    ¬(isMultipleOf45 n ∧ isMultipleOf45 (productOfList (digits n)))) :=
sorry

end least_multiple_945_l797_79798


namespace tea_cost_price_l797_79743

/-- The cost price per kg of the 80 kg of tea -/
def C : ℝ := sorry

/-- The theorem stating the conditions and the result to be proved -/
theorem tea_cost_price :
  -- 80 kg of tea at cost price C
  -- 20 kg of tea at $20 per kg
  -- Total selling price for 100 kg at $20.8 per kg
  -- 30% profit margin
  80 * C + 20 * 20 = (100 * 20.8) / 1.3 →
  C = 15 := by sorry

end tea_cost_price_l797_79743


namespace part_one_part_two_l797_79722

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - 2 * a) / (x - (a^2 + 1)) < 0}

-- Part 1
theorem part_one : 
  A 2 ∩ (Set.univ \ B 2) = {x | 2 < x ∧ x ≤ 4 ∨ 5 ≤ x ∧ x < 7} :=
by sorry

-- Part 2
theorem part_two :
  {a : ℝ | a ≠ 1 ∧ A a ∪ B a = A a} = {a | 1 < a ∧ a ≤ 3 ∨ a = -1} :=
by sorry

end part_one_part_two_l797_79722


namespace max_arithmetic_mean_of_special_pairs_l797_79750

theorem max_arithmetic_mean_of_special_pairs : ∃ (a b : ℕ), 
  10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a > b ∧
  (a + b) / 2 = (25 / 24) * Real.sqrt (a * b) ∧
  ∀ (c d : ℕ), 10 ≤ c ∧ c < 100 ∧ 10 ≤ d ∧ d < 100 ∧ c > d ∧
    (c + d) / 2 = (25 / 24) * Real.sqrt (c * d) →
    (a + b) / 2 ≥ (c + d) / 2 ∧
  (a + b) / 2 = 75 :=
by sorry

end max_arithmetic_mean_of_special_pairs_l797_79750


namespace triangle_area_l797_79776

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  a = 1 →
  b = Real.sqrt 3 →
  C = π / 6 →
  (1/2) * a * b * Real.sin C = Real.sqrt 3 / 4 := by
  sorry

end triangle_area_l797_79776


namespace f_composition_equals_9184_l797_79756

/-- The function f(x) = 3x^2 + 2x - 1 -/
def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 1

/-- Theorem: f(f(f(1))) = 9184 -/
theorem f_composition_equals_9184 : f (f (f 1)) = 9184 := by
  sorry

end f_composition_equals_9184_l797_79756
