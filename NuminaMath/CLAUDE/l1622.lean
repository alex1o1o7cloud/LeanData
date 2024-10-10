import Mathlib

namespace gumball_pigeonhole_min_gumballs_for_five_same_color_l1622_162282

theorem gumball_pigeonhole : ∀ (draw : ℕ),
  (∃ (color : Fin 4), (λ i => [10, 8, 9, 7].get i) color ≥ 5) →
  draw ≥ 17 :=
by
  sorry

theorem min_gumballs_for_five_same_color :
  ∃ (draw : ℕ), draw = 17 ∧
  (∀ (smaller : ℕ), smaller < draw →
    ¬∃ (color : Fin 4), (λ i => [10, 8, 9, 7].get i) color ≥ 5) ∧
  (∃ (color : Fin 4), (λ i => [10, 8, 9, 7].get i) color ≥ 5) :=
by
  sorry

end gumball_pigeonhole_min_gumballs_for_five_same_color_l1622_162282


namespace parabola_through_fixed_point_l1622_162289

-- Define the line equation
def line_equation (a x y : ℝ) : Prop := (a - 1) * x - y + 2 * a + 1 = 0

-- Define the fixed point P
def fixed_point : ℝ × ℝ := (-2, 3)

-- Theorem statement
theorem parabola_through_fixed_point :
  (∀ a : ℝ, line_equation a (fixed_point.1) (fixed_point.2)) ∧
  (∃ p : ℝ, p > 0 ∧ 
    ((∀ x y : ℝ, (x, y) = fixed_point → y^2 = -2*p*x) ∨
     (∀ x y : ℝ, (x, y) = fixed_point → x^2 = 2*p*y))) :=
sorry

end parabola_through_fixed_point_l1622_162289


namespace exists_negative_monomial_degree_5_l1622_162204

/-- A monomial in x and y -/
structure Monomial where
  coeff : ℤ
  x_power : ℕ
  y_power : ℕ

/-- The degree of a monomial -/
def Monomial.degree (m : Monomial) : ℕ := m.x_power + m.y_power

/-- A monomial is negative if its coefficient is negative -/
def Monomial.isNegative (m : Monomial) : Prop := m.coeff < 0

theorem exists_negative_monomial_degree_5 :
  ∃ m : Monomial, m.isNegative ∧ m.degree = 5 :=
sorry

end exists_negative_monomial_degree_5_l1622_162204


namespace periodic_function_l1622_162239

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def satisfies_condition (f : ℝ → ℝ) : Prop := ∀ x, f x + f (10 - x) = 4

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem periodic_function (f : ℝ → ℝ) 
  (h1 : is_even f) 
  (h2 : satisfies_condition f) : 
  is_periodic f 20 := by
  sorry

end periodic_function_l1622_162239


namespace trigonometric_equation_solution_l1622_162271

theorem trigonometric_equation_solution (x : ℝ) :
  (∃ k : ℤ, x = -π/28 + π*k/7 ∨ x = π/12 + 2*π*k/3 ∨ x = 5*π/44 + 2*π*k/11) ↔
  (Real.cos (11*x) - Real.cos (3*x) - Real.sin (11*x) + Real.sin (3*x) = Real.sqrt 2 * Real.cos (14*x)) :=
by sorry

end trigonometric_equation_solution_l1622_162271


namespace x_times_x_minus_one_eq_six_is_quadratic_l1622_162278

/-- A quadratic equation is an equation of the form ax² + bx + c = 0, where a ≠ 0 -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x(x-1) = 6 -/
def f (x : ℝ) : ℝ := x * (x - 1) - 6

theorem x_times_x_minus_one_eq_six_is_quadratic : is_quadratic_equation f := by
  sorry

end x_times_x_minus_one_eq_six_is_quadratic_l1622_162278


namespace organic_egg_tray_price_l1622_162220

/-- The price of a tray of organic eggs -/
def tray_price (individual_price : ℚ) (tray_size : ℕ) (savings_per_egg : ℚ) : ℚ :=
  (individual_price - savings_per_egg) * tray_size / 100

/-- Proof that the price of a tray of 30 organic eggs is $12 -/
theorem organic_egg_tray_price :
  let individual_price : ℚ := 50
  let tray_size : ℕ := 30
  let savings_per_egg : ℚ := 10
  tray_price individual_price tray_size savings_per_egg = 12 := by sorry

end organic_egg_tray_price_l1622_162220


namespace smallest_factorial_divisible_by_23m_and_33n_l1622_162268

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem smallest_factorial_divisible_by_23m_and_33n :
  (∀ k < 24, ¬(factorial k % (23 * k) = 0)) ∧
  (factorial 24 % (23 * 24) = 0) ∧
  (∀ k < 12, ¬(factorial k % (33 * k) = 0)) ∧
  (factorial 12 % (33 * 12) = 0) := by
  sorry

#check smallest_factorial_divisible_by_23m_and_33n

end smallest_factorial_divisible_by_23m_and_33n_l1622_162268


namespace system_solution_l1622_162202

theorem system_solution : 
  ∀ x y : ℝ, 
    (x + y = (7 - x) + (7 - y) ∧ 
     x^2 - y = (x - 2) + (y - 2)) ↔ 
    ((x = -5 ∧ y = 12) ∨ (x = 2 ∧ y = 5)) :=
by sorry

end system_solution_l1622_162202


namespace book_arrangement_theorem_l1622_162254

theorem book_arrangement_theorem :
  let n : ℕ := 7  -- number of books
  let k : ℕ := 3  -- number of shelves
  let arrangements := (n - 1).choose (k - 1) * n.factorial
  arrangements = 75600 := by
  sorry

end book_arrangement_theorem_l1622_162254


namespace green_corner_plants_l1622_162258

theorem green_corner_plants (total_pots : ℕ) (green_lily_cost spider_plant_cost : ℕ) (total_budget : ℕ)
  (h1 : total_pots = 46)
  (h2 : green_lily_cost = 9)
  (h3 : spider_plant_cost = 6)
  (h4 : total_budget = 390) :
  ∃ (green_lily_pots spider_plant_pots : ℕ),
    green_lily_pots + spider_plant_pots = total_pots ∧
    green_lily_cost * green_lily_pots + spider_plant_cost * spider_plant_pots = total_budget ∧
    green_lily_pots = 38 ∧
    spider_plant_pots = 8 :=
by sorry

end green_corner_plants_l1622_162258


namespace cauchy_schwarz_on_unit_circle_l1622_162293

theorem cauchy_schwarz_on_unit_circle (a b x y : ℝ) 
  (h1 : a^2 + b^2 = 1) (h2 : x^2 + y^2 = 1) : a*x + b*y ≤ 1 := by
  sorry

end cauchy_schwarz_on_unit_circle_l1622_162293


namespace algebra_textbooks_count_l1622_162263

theorem algebra_textbooks_count : ∃ (x y n : ℕ), 
  x * n + y = 2015 ∧ 
  y * n + x = 1580 ∧ 
  n > 0 ∧ 
  y = 287 := by
  sorry

end algebra_textbooks_count_l1622_162263


namespace unemployment_rate_after_changes_l1622_162225

theorem unemployment_rate_after_changes (initial_unemployment : ℝ) : 
  initial_unemployment ≥ 0 ∧ initial_unemployment ≤ 100 →
  1.1 * initial_unemployment + 0.85 * (100 - initial_unemployment) = 100 →
  1.1 * initial_unemployment = 66 :=
by sorry

end unemployment_rate_after_changes_l1622_162225


namespace anna_lettuce_plants_l1622_162273

/-- The number of large salads Anna wants --/
def desired_salads : ℕ := 12

/-- The fraction of lettuce that will be lost --/
def loss_fraction : ℚ := 1/2

/-- The number of large salads each lettuce plant provides --/
def salads_per_plant : ℕ := 3

/-- The number of lettuce plants Anna should grow --/
def plants_to_grow : ℕ := 8

theorem anna_lettuce_plants : 
  (plants_to_grow : ℚ) * (1 - loss_fraction) * salads_per_plant ≥ desired_salads := by
  sorry

end anna_lettuce_plants_l1622_162273


namespace happiness_difference_test_l1622_162264

-- Define the data from the problem
def total_observations : ℕ := 1184
def boys_happy : ℕ := 638
def boys_unhappy : ℕ := 128
def girls_happy : ℕ := 372
def girls_unhappy : ℕ := 46
def total_happy : ℕ := 1010
def total_unhappy : ℕ := 174
def total_boys : ℕ := 766
def total_girls : ℕ := 418

-- Define the χ² calculation function
def chi_square : ℚ :=
  (total_observations : ℚ) * (boys_happy * girls_unhappy - boys_unhappy * girls_happy)^2 /
  (total_happy * total_unhappy * total_boys * total_girls)

-- Define the critical values
def critical_value_001 : ℚ := 6635 / 1000
def critical_value_0005 : ℚ := 7879 / 1000

-- Theorem statement
theorem happiness_difference_test :
  (chi_square > critical_value_001) ∧ (chi_square < critical_value_0005) :=
by sorry

end happiness_difference_test_l1622_162264


namespace sum_first_eight_super_nice_l1622_162249

def is_prime (n : ℕ) : Prop := sorry

def is_super_nice (n : ℕ) : Prop :=
  (∃ p q r : ℕ, is_prime p ∧ is_prime q ∧ is_prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ n = p * q * r) ∨
  (∃ p : ℕ, is_prime p ∧ n = p^4)

def first_eight_super_nice : List ℕ :=
  [16, 30, 42, 66, 70, 81, 105, 110]

theorem sum_first_eight_super_nice :
  (∀ n ∈ first_eight_super_nice, is_super_nice n) ∧
  (∀ m : ℕ, m < 16 → ¬is_super_nice m) ∧
  (∀ m : ℕ, m > 110 ∧ is_super_nice m → ∃ n ∈ first_eight_super_nice, m > n) ∧
  (List.sum first_eight_super_nice = 520) :=
by sorry

end sum_first_eight_super_nice_l1622_162249


namespace maryGarbageBillIs102_l1622_162284

/-- Calculates Mary's monthly garbage bill --/
def maryGarbageBill : ℚ :=
  let trashBinCharge : ℚ := 10
  let recyclingBinCharge : ℚ := 5
  let trashBinCount : ℕ := 2
  let recyclingBinCount : ℕ := 1
  let weeksInMonth : ℕ := 4
  let elderlyDiscountPercentage : ℚ := 18 / 100
  let inappropriateItemsFine : ℚ := 20

  let weeklyCharge := trashBinCharge * trashBinCount + recyclingBinCharge * recyclingBinCount
  let monthlyCharge := weeklyCharge * weeksInMonth
  let discountAmount := monthlyCharge * elderlyDiscountPercentage
  let discountedMonthlyCharge := monthlyCharge - discountAmount
  discountedMonthlyCharge + inappropriateItemsFine

theorem maryGarbageBillIs102 : maryGarbageBill = 102 := by
  sorry

end maryGarbageBillIs102_l1622_162284


namespace price_change_calculation_l1622_162218

theorem price_change_calculation :
  let original_price := 100
  let price_after_day1 := original_price * (1 - 0.12)
  let price_after_day2 := price_after_day1 * (1 - 0.10)
  let price_after_day3 := price_after_day2 * (1 - 0.08)
  let final_price := price_after_day3 * (1 + 0.05)
  (final_price / original_price) * 100 = 76.5072 := by
sorry

end price_change_calculation_l1622_162218


namespace fourth_month_sales_l1622_162227

def sales_month1 : ℕ := 3435
def sales_month2 : ℕ := 3920
def sales_month3 : ℕ := 3855
def sales_month5 : ℕ := 3560
def sales_month6 : ℕ := 2000
def average_sale : ℕ := 3500
def num_months : ℕ := 6

theorem fourth_month_sales :
  ∃ (sales_month4 : ℕ),
    sales_month4 = 4230 ∧
    (sales_month1 + sales_month2 + sales_month3 + sales_month4 + sales_month5 + sales_month6) / num_months = average_sale :=
by sorry

end fourth_month_sales_l1622_162227


namespace hyperbola_center_is_two_two_l1622_162270

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  (4 * y - 8)^2 / 8^2 - (5 * x - 10)^2 / 7^2 = 1

/-- The center of the hyperbola -/
def hyperbola_center : ℝ × ℝ := (2, 2)

/-- Theorem: The center of the given hyperbola is (2, 2) -/
theorem hyperbola_center_is_two_two :
  ∀ x y : ℝ, hyperbola_equation x y → hyperbola_center = (x, y) := by
  sorry

end hyperbola_center_is_two_two_l1622_162270


namespace triangle_tangent_product_range_l1622_162214

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    satisfying a^2 + b^2 + √2ab = c^2, prove that 0 < tan A * tan (2*B) < 1/2 -/
theorem triangle_tangent_product_range (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a^2 + b^2 + Real.sqrt 2 * a * b = c^2 →
  0 < Real.tan A * Real.tan (2 * B) ∧ Real.tan A * Real.tan (2 * B) < 1 / 2 := by
  sorry

end triangle_tangent_product_range_l1622_162214


namespace triangle_prime_sides_area_not_integer_l1622_162201

theorem triangle_prime_sides_area_not_integer 
  (a b c : ℕ) 
  (ha : Prime a) 
  (hb : Prime b) 
  (hc : Prime c) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  ¬(∃ (S : ℕ), S^2 * 16 = (a + b + c) * ((a + b + c) - 2*a) * ((a + b + c) - 2*b) * ((a + b + c) - 2*c)) :=
sorry

end triangle_prime_sides_area_not_integer_l1622_162201


namespace magician_min_earnings_l1622_162224

/-- Represents the earnings of a magician selling card decks --/
def magician_earnings (initial_decks : ℕ) (remaining_decks : ℕ) (full_price : ℕ) (discounted_price : ℕ) : ℕ :=
  (initial_decks - remaining_decks) * discounted_price

/-- Theorem stating the minimum earnings of the magician --/
theorem magician_min_earnings :
  let initial_decks : ℕ := 15
  let remaining_decks : ℕ := 3
  let full_price : ℕ := 3
  let discounted_price : ℕ := 2
  magician_earnings initial_decks remaining_decks full_price discounted_price ≥ 24 := by
  sorry

#check magician_min_earnings

end magician_min_earnings_l1622_162224


namespace sales_balance_l1622_162242

/-- Represents the sales increase of product C as a percentage -/
def sales_increase_C : ℝ := 0.3

/-- Represents the proportion of total sales from product C last year -/
def last_year_C_proportion : ℝ := 0.4

/-- Represents the decrease in sales for products A and B -/
def sales_decrease_AB : ℝ := 0.2

/-- Represents the proportion of total sales from products A and B last year -/
def last_year_AB_proportion : ℝ := 1 - last_year_C_proportion

theorem sales_balance :
  last_year_C_proportion * (1 + sales_increase_C) + 
  last_year_AB_proportion * (1 - sales_decrease_AB) = 1 := by
  sorry

end sales_balance_l1622_162242


namespace bar_charts_cannot_show_change_line_charts_can_show_change_bar_charts_show_change_is_false_l1622_162237

-- Define the types of charts
inductive Chart
| Bar
| Line

-- Define the capability of showing increase or decrease
def can_show_change (c : Chart) : Prop :=
  match c with
  | Chart.Line => true
  | Chart.Bar => false

-- Theorem stating that bar charts cannot show change
theorem bar_charts_cannot_show_change :
  ¬(can_show_change Chart.Bar) :=
by
  sorry

-- Theorem stating that line charts can show change
theorem line_charts_can_show_change :
  can_show_change Chart.Line :=
by
  sorry

-- Main theorem proving the original statement is false
theorem bar_charts_show_change_is_false :
  ¬(∀ (c : Chart), can_show_change c) :=
by
  sorry

end bar_charts_cannot_show_change_line_charts_can_show_change_bar_charts_show_change_is_false_l1622_162237


namespace quadratic_inequality_equivalence_l1622_162209

theorem quadratic_inequality_equivalence (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x - 4*a ≥ 0) ↔ (-16 ≤ a ∧ a ≤ 0) :=
by sorry

end quadratic_inequality_equivalence_l1622_162209


namespace count_valid_integers_valid_integers_formula_correct_l1622_162267

/-- The number of n-digit decimal integers using only digits 1, 2, and 3,
    and containing each of these digits at least once. -/
def validIntegers (n : ℕ+) : ℕ :=
  3^n.val - 3 * 2^n.val + 3

/-- Theorem stating that validIntegers gives the correct count. -/
theorem count_valid_integers (n : ℕ+) :
  validIntegers n = (3^n.val - 3 * 2^n.val + 3) := by
  sorry

/-- Proof that the formula is correct for all positive integers n. -/
theorem valid_integers_formula_correct :
  ∀ n : ℕ+, validIntegers n = (3^n.val - 3 * 2^n.val + 3) := by
  sorry

end count_valid_integers_valid_integers_formula_correct_l1622_162267


namespace hyperbola_eccentricity_l1622_162248

/-- Given a hyperbola with the following properties:
    1. A line is drawn through the left focus F₁ at a 30° angle
    2. This line intersects the right branch of the hyperbola at point P
    3. A circle with diameter PF₁ passes through the right focus F₂
    Then the eccentricity of the hyperbola is √3 -/
theorem hyperbola_eccentricity (F₁ F₂ P : ℝ × ℝ) (a b c : ℝ) :
  let e := c / a
  (P.1 = c ∧ P.2 = b^2 / a) →  -- P is on the right branch
  (P.2 / (2 * c) = Real.tan (30 * π / 180)) →  -- Line through F₁ is at 30°
  (Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) = Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2)) →  -- Circle condition
  e = Real.sqrt 3 :=
by sorry

end hyperbola_eccentricity_l1622_162248


namespace least_stamps_stamps_23_robert_stamps_l1622_162279

theorem least_stamps (n : ℕ) : n > 0 ∧ n % 7 = 2 ∧ n % 4 = 3 → n ≥ 23 := by
  sorry

theorem stamps_23 : 23 % 7 = 2 ∧ 23 % 4 = 3 := by
  sorry

theorem robert_stamps : ∃ n : ℕ, n > 0 ∧ n % 7 = 2 ∧ n % 4 = 3 ∧ 
  ∀ m : ℕ, (m > 0 ∧ m % 7 = 2 ∧ m % 4 = 3) → n ≤ m := by
  sorry

end least_stamps_stamps_23_robert_stamps_l1622_162279


namespace number_of_one_point_two_stamps_l1622_162283

/-- Represents the number of stamps of each denomination -/
structure StampCounts where
  half : ℕ
  eightyPercent : ℕ
  onePointTwo : ℕ

/-- The total value of all stamps in cents -/
def totalValue (s : StampCounts) : ℕ :=
  50 * s.half + 80 * s.eightyPercent + 120 * s.onePointTwo

/-- The theorem stating the number of 1.2 yuan stamps given the conditions -/
theorem number_of_one_point_two_stamps :
  ∃ (s : StampCounts),
    totalValue s = 6000 ∧
    s.eightyPercent = 4 * s.half ∧
    s.onePointTwo = 13 :=
by sorry

end number_of_one_point_two_stamps_l1622_162283


namespace min_value_theorem_l1622_162217

theorem min_value_theorem (x y : ℝ) :
  3 * |x - y| + |2 * x - 5| = x + 1 →
  2 * x + y ≥ 4 := by
sorry

end min_value_theorem_l1622_162217


namespace firm_partners_count_l1622_162203

theorem firm_partners_count :
  ∀ (partners associates : ℕ),
  (partners : ℚ) / associates = 2 / 63 →
  partners / (associates + 50) = 1 / 34 →
  partners = 20 := by
sorry

end firm_partners_count_l1622_162203


namespace book_sharing_probability_l1622_162295

/-- The number of students sharing books -/
def num_students : ℕ := 2

/-- The number of books being shared -/
def num_books : ℕ := 3

/-- The total number of possible book distribution scenarios -/
def total_scenarios : ℕ := 8

/-- The number of scenarios where one student gets all books and the other gets none -/
def favorable_scenarios : ℕ := 2

/-- The probability of one student getting all books and the other getting none -/
def probability : ℚ := favorable_scenarios / total_scenarios

theorem book_sharing_probability :
  probability = 1/4 := by sorry

end book_sharing_probability_l1622_162295


namespace olympiad_team_formation_l1622_162210

theorem olympiad_team_formation (n : ℕ) (k : ℕ) (roles : ℕ) 
  (h1 : n = 20) 
  (h2 : k = 3) 
  (h3 : roles = 3) :
  (n.factorial / ((n - k).factorial * k.factorial)) * (k.factorial / (roles.factorial * (k - roles).factorial)) = 6840 :=
sorry

end olympiad_team_formation_l1622_162210


namespace completing_square_equivalence_l1622_162233

theorem completing_square_equivalence (x : ℝ) : 
  (x^2 - 2*x - 1 = 0) ↔ ((x - 1)^2 = 2) := by
sorry

end completing_square_equivalence_l1622_162233


namespace camera_filter_savings_percentage_l1622_162207

theorem camera_filter_savings_percentage : 
  let kit_price : ℚ := 144.20
  let filter_prices : List ℚ := [21.75, 21.75, 18.60, 18.60, 23.80, 29.35, 29.35]
  let total_individual_price : ℚ := filter_prices.sum
  let savings : ℚ := total_individual_price - kit_price
  let savings_percentage : ℚ := (savings / total_individual_price) * 100
  savings_percentage = 11.64 := by sorry

end camera_filter_savings_percentage_l1622_162207


namespace circle_polar_to_cartesian_and_area_l1622_162256

/-- Given a circle C with polar equation p = 2cosθ, this theorem proves that
    its Cartesian equation is x² - 2x + y² = 0 and its area is π. -/
theorem circle_polar_to_cartesian_and_area :
  ∀ (p θ x y : ℝ),
  (p = 2 * Real.cos θ) →                  -- Polar equation
  (x = p * Real.cos θ ∧ y = p * Real.sin θ) →  -- Polar to Cartesian conversion
  (x^2 - 2*x + y^2 = 0) ∧                 -- Cartesian equation
  (Real.pi = (Real.pi : ℝ)) :=            -- Area (π)
by sorry

end circle_polar_to_cartesian_and_area_l1622_162256


namespace davids_original_portion_l1622_162297

/-- Given a total initial amount of $1500 shared among David, Elisa, and Frank,
    where the total final amount is $2700, Elisa and Frank both triple their initial investments,
    and David loses $200, prove that David's original portion is $800. -/
theorem davids_original_portion (d e f : ℝ) : 
  d + e + f = 1500 →
  d - 200 + 3 * e + 3 * f = 2700 →
  d = 800 := by
sorry

end davids_original_portion_l1622_162297


namespace f_of_5_equals_0_l1622_162240

theorem f_of_5_equals_0 (f : ℝ → ℝ) (h : ∀ x, f (2 * x + 1) = x^2 - 2*x) : f 5 = 0 := by
  sorry

end f_of_5_equals_0_l1622_162240


namespace euler_family_mean_age_l1622_162250

def euler_family_ages : List ℕ := [6, 6, 6, 6, 8, 8, 16]

theorem euler_family_mean_age :
  (euler_family_ages.sum / euler_family_ages.length : ℚ) = 8 := by
  sorry

end euler_family_mean_age_l1622_162250


namespace symmetry_axis_of_quadratic_l1622_162235

/-- A quadratic function of the form y = (x + h)^2 has a symmetry axis of x = -h -/
theorem symmetry_axis_of_quadratic (h : ℝ) : 
  let f : ℝ → ℝ := λ x => (x + h)^2
  ∀ x : ℝ, f ((-h) - (x - (-h))) = f x := by
  sorry

end symmetry_axis_of_quadratic_l1622_162235


namespace friends_to_movies_l1622_162244

theorem friends_to_movies (total_friends : ℕ) (cant_go : ℕ) (can_go : ℕ) 
  (h1 : total_friends = 15)
  (h2 : cant_go = 7)
  (h3 : can_go = total_friends - cant_go) :
  can_go = 8 := by
  sorry

end friends_to_movies_l1622_162244


namespace investment_result_l1622_162246

/-- Calculates the future value of an investment with compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proves that an investment of $4000 at 10% annual compound interest for 2 years results in $4840 -/
theorem investment_result : compound_interest 4000 0.1 2 = 4840 := by
  sorry

end investment_result_l1622_162246


namespace complement_A_union_B_subset_l1622_162228

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x < 2}

-- State the theorem
theorem complement_A_union_B_subset :
  (Set.compl A ∪ B) ⊆ {x : ℝ | x < 2} := by sorry

end complement_A_union_B_subset_l1622_162228


namespace expected_voters_for_candidate_A_prove_expected_voters_for_candidate_A_l1622_162238

theorem expected_voters_for_candidate_A : ℝ → Prop :=
  fun x => 
    -- Define the percentage of Democrats
    let percent_democrats : ℝ := 0.60
    -- Define the percentage of Republicans
    let percent_republicans : ℝ := 1 - percent_democrats
    -- Define the percentage of Democrats voting for A
    let percent_democrats_for_A : ℝ := 0.85
    -- Define the percentage of Republicans voting for A
    let percent_republicans_for_A : ℝ := 0.20
    -- Calculate the total percentage of voters for A
    let total_percent_for_A : ℝ := 
      percent_democrats * percent_democrats_for_A + 
      percent_republicans * percent_republicans_for_A
    -- The theorem statement
    x = total_percent_for_A * 100

-- The proof of the theorem
theorem prove_expected_voters_for_candidate_A : 
  expected_voters_for_candidate_A 59 := by
  sorry

end expected_voters_for_candidate_A_prove_expected_voters_for_candidate_A_l1622_162238


namespace unique_solution_power_sum_l1622_162275

theorem unique_solution_power_sum (a b c : ℕ) :
  (∀ n : ℕ, a^n + b^n = c^(n+1)) → (a = 2 ∧ b = 2 ∧ c = 2) :=
by sorry

end unique_solution_power_sum_l1622_162275


namespace dice_sum_divisibility_probability_l1622_162255

theorem dice_sum_divisibility_probability (n : ℕ) (a b c : ℕ) 
  (h1 : a + b + c = n) 
  (h2 : 0 ≤ a ∧ a ≤ n) 
  (h3 : 0 ≤ b ∧ b ≤ n) 
  (h4 : 0 ≤ c ∧ c ≤ n) :
  (a^3 + b^3 + c^3 + 6*a*b*c : ℚ) / (n^3 : ℚ) ≥ 1/4 := by
  sorry

end dice_sum_divisibility_probability_l1622_162255


namespace truck_travel_distance_l1622_162252

/-- Given a truck that travels 150 miles on 5 gallons of diesel,
    prove that it can travel 210 miles on 7 gallons of diesel,
    assuming a constant rate of travel. -/
theorem truck_travel_distance 
  (initial_distance : ℝ) 
  (initial_fuel : ℝ) 
  (new_fuel : ℝ) 
  (h1 : initial_distance = 150) 
  (h2 : initial_fuel = 5) 
  (h3 : new_fuel = 7) :
  (initial_distance / initial_fuel) * new_fuel = 210 :=
by sorry

end truck_travel_distance_l1622_162252


namespace specific_ellipse_semi_minor_axis_l1622_162296

/-- Represents an ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  focus : ℝ × ℝ
  semi_major_endpoint : ℝ × ℝ

/-- Calculates the semi-minor axis of an ellipse -/
def semi_minor_axis (e : Ellipse) : ℝ :=
  sorry

/-- Theorem stating the semi-minor axis of the specific ellipse is √21 -/
theorem specific_ellipse_semi_minor_axis :
  let e : Ellipse := {
    center := (0, 0),
    focus := (0, -2),
    semi_major_endpoint := (0, 5)
  }
  semi_minor_axis e = Real.sqrt 21 := by
  sorry

end specific_ellipse_semi_minor_axis_l1622_162296


namespace perpendicular_transitivity_l1622_162281

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the theorem
theorem perpendicular_transitivity
  (m n : Line) (α β : Plane)
  (h1 : perpendicular m β)
  (h2 : perpendicular n β)
  (h3 : perpendicular n α) :
  perpendicular m α :=
sorry

end perpendicular_transitivity_l1622_162281


namespace max_value_when_a_zero_one_zero_iff_a_positive_l1622_162280

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 / x - (a + 1) * Real.log x

-- Part 1: Maximum value when a = 0
theorem max_value_when_a_zero :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f 0 y ≤ f 0 x ∧ f 0 x = -1 :=
sorry

-- Part 2: Range of a for exactly one zero
theorem one_zero_iff_a_positive :
  ∀ (a : ℝ), (∃! (x : ℝ), x > 0 ∧ f a x = 0) ↔ a > 0 :=
sorry

end

end max_value_when_a_zero_one_zero_iff_a_positive_l1622_162280


namespace binomial_expansion_theorem_l1622_162234

/-- The number of terms with integer exponents in the expansion of (√x + 1/(2∛x))^n -/
def integer_exponent_terms (n : ℕ) : ℕ :=
  (Finset.filter (fun r => (2 * n - 3 * r) % 3 = 0) (Finset.range (n + 1))).card

/-- The coefficients of the first three terms in the expansion -/
def first_three_coeffs (n : ℕ) : Fin 3 → ℚ
  | 0 => 1
  | 1 => n / 2
  | 2 => n * (n - 1) / 8

/-- The condition that the first three coefficients form an arithmetic sequence -/
def arithmetic_sequence_condition (n : ℕ) : Prop :=
  2 * (first_three_coeffs n 1) = (first_three_coeffs n 0) + (first_three_coeffs n 2)

theorem binomial_expansion_theorem (n : ℕ) :
  arithmetic_sequence_condition n → integer_exponent_terms n = 3 := by
  sorry

end binomial_expansion_theorem_l1622_162234


namespace vasya_has_winning_strategy_l1622_162200

/-- Represents the state of the game board --/
structure GameState where
  piles : List Nat
  deriving Repr

/-- Represents a player in the game --/
inductive Player
  | Petya
  | Vasya
  deriving Repr

/-- Represents a move in the game --/
structure Move where
  pileIndices : List Nat
  stonesToRemove : Nat
  deriving Repr

/-- Checks if a move is valid for a given player and game state --/
def isValidMove (player : Player) (state : GameState) (move : Move) : Bool :=
  match player with
  | Player.Petya => move.pileIndices.length == 1 && move.stonesToRemove ≤ 3
  | Player.Vasya => move.pileIndices.length == move.stonesToRemove && move.stonesToRemove ≤ 3

/-- Applies a move to the game state --/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if the game is over --/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- The main theorem stating that Vasya has a winning strategy --/
theorem vasya_has_winning_strategy :
  ∃ (strategy : GameState → Move),
    ∀ (initialState : GameState),
      initialState.piles.length == 11 →
      (∀ pile ∈ initialState.piles, pile == 10) →
      ∀ (petyaStrategy : GameState → Move),
        isValidMove Player.Petya initialState (petyaStrategy initialState) →
        (∀ state : GameState, isValidMove Player.Petya state (petyaStrategy state)) →
        isValidMove Player.Vasya (applyMove initialState (petyaStrategy initialState)) (strategy (applyMove initialState (petyaStrategy initialState))) →
        (∀ state : GameState, isValidMove Player.Vasya state (strategy state)) →
        ∃ (finalState : GameState),
          isGameOver finalState ∧
          (finalState.piles.all (· == 0) ∨ ¬isValidMove Player.Petya finalState (petyaStrategy finalState)) :=
  sorry

end vasya_has_winning_strategy_l1622_162200


namespace remainder_theorem_l1622_162226

theorem remainder_theorem (P D Q R Q' R' : ℕ) (hD : D > 1) 
  (h1 : P = Q * D + R) (h2 : Q = (D - 1) * Q' + R') :
  P % (D * (D - 1)) = D * R' + R :=
by sorry

end remainder_theorem_l1622_162226


namespace basketball_preference_theorem_l1622_162287

/-- Represents the school population and basketball preferences -/
structure School where
  total_students : ℕ
  male_ratio : ℚ
  female_ratio : ℚ
  male_basketball_ratio : ℚ
  female_basketball_ratio : ℚ

/-- Calculate the percentage of students who do not like basketball -/
def percentage_not_liking_basketball (s : School) : ℚ :=
  let male_students := s.total_students * s.male_ratio / (s.male_ratio + s.female_ratio)
  let female_students := s.total_students * s.female_ratio / (s.male_ratio + s.female_ratio)
  let male_liking_basketball := male_students * s.male_basketball_ratio
  let female_liking_basketball := female_students * s.female_basketball_ratio
  let total_not_liking := s.total_students - (male_liking_basketball + female_liking_basketball)
  total_not_liking / s.total_students * 100

/-- The main theorem to prove -/
theorem basketball_preference_theorem (s : School) 
  (h1 : s.total_students = 1000)
  (h2 : s.male_ratio = 3)
  (h3 : s.female_ratio = 2)
  (h4 : s.male_basketball_ratio = 2/3)
  (h5 : s.female_basketball_ratio = 1/5) :
  percentage_not_liking_basketball s = 52 := by
  sorry


end basketball_preference_theorem_l1622_162287


namespace arithmetic_problem_l1622_162230

theorem arithmetic_problem : 4 * (8 - 3)^2 - 2 * 7 = 86 := by
  sorry

end arithmetic_problem_l1622_162230


namespace berry_pie_theorem_l1622_162262

/-- Represents the amount of berries picked by each person -/
structure BerryPicker where
  strawberries : ℕ
  blueberries : ℕ
  raspberries : ℕ

/-- Represents the requirements for each type of pie -/
structure PieRequirements where
  strawberry : ℕ
  blueberry : ℕ
  raspberry : ℕ

/-- Calculates the maximum number of complete pies that can be made -/
def max_pies (christine : BerryPicker) (rachel : BerryPicker) (req : PieRequirements) : ℕ × ℕ × ℕ :=
  let total_strawberries := christine.strawberries + rachel.strawberries
  let total_blueberries := christine.blueberries + rachel.blueberries
  let total_raspberries := christine.raspberries + rachel.raspberries
  (total_strawberries / req.strawberry,
   total_blueberries / req.blueberry,
   total_raspberries / req.raspberry)

theorem berry_pie_theorem (christine : BerryPicker) (rachel : BerryPicker) (req : PieRequirements) :
  christine.strawberries = 10 ∧
  christine.blueberries = 8 ∧
  christine.raspberries = 20 ∧
  rachel.strawberries = 2 * christine.strawberries ∧
  rachel.blueberries = 2 * christine.blueberries ∧
  rachel.raspberries = christine.raspberries / 2 ∧
  req.strawberry = 3 ∧
  req.blueberry = 2 ∧
  req.raspberry = 4 →
  max_pies christine rachel req = (10, 12, 7) := by
  sorry

end berry_pie_theorem_l1622_162262


namespace best_approx_sqrt3_l1622_162266

def best_rational_approx (n : ℕ) (x : ℝ) : ℚ :=
  sorry

theorem best_approx_sqrt3 :
  best_rational_approx 15 (Real.sqrt 3) = 26 / 15 := by
  sorry

end best_approx_sqrt3_l1622_162266


namespace binomial_coefficient_congruence_l1622_162241

theorem binomial_coefficient_congruence 
  (p : Nat) 
  (hp : p.Prime ∧ p > 3 ∧ Odd p) 
  (a b : Nat) 
  (hab : a > b ∧ b > 1) : 
  Nat.choose (a * p) (a * p) ≡ Nat.choose a b [MOD p^3] := by
  sorry

end binomial_coefficient_congruence_l1622_162241


namespace quadratic_functions_problem_l1622_162247

/-- Given two quadratic functions y₁ and y₂ satisfying certain conditions, 
    prove that α = 1, y₁ = -2x² + 4x + 3, and y₂ = 3x² + 12x + 10 -/
theorem quadratic_functions_problem 
  (y₁ y₂ : ℝ → ℝ) 
  (α : ℝ) 
  (h_α_pos : α > 0)
  (h_y₁_max : ∀ x, y₁ x ≤ y₁ α)
  (h_y₁_max_val : y₁ α = 5)
  (h_y₂_α : y₂ α = 25)
  (h_y₂_min : ∀ x, y₂ x ≥ -2)
  (h_sum : ∀ x, y₁ x + y₂ x = x^2 + 16*x + 13) :
  α = 1 ∧ 
  (∀ x, y₁ x = -2*x^2 + 4*x + 3) ∧ 
  (∀ x, y₂ x = 3*x^2 + 12*x + 10) := by
  sorry

end quadratic_functions_problem_l1622_162247


namespace acid_solution_mixture_l1622_162219

/-- Given:
  n : ℝ, amount of initial solution in ounces
  y : ℝ, amount of added solution in ounces
  n > 30
  initial solution concentration is n%
  added solution concentration is 20%
  final solution concentration is (n-15)%
Prove: y = 15n / (n+35) -/
theorem acid_solution_mixture (n : ℝ) (y : ℝ) (h1 : n > 30) :
  (n * (n / 100) + y * (20 / 100)) / (n + y) = (n - 15) / 100 →
  y = 15 * n / (n + 35) := by
sorry

end acid_solution_mixture_l1622_162219


namespace monotone_decreasing_implies_a_ge_one_l1622_162211

/-- A function f(x) = ln x - ax is monotonically decreasing on (1, +∞) -/
def is_monotone_decreasing (a : ℝ) : Prop :=
  ∀ x y, 1 < x → x < y → Real.log y - a * y < Real.log x - a * x

/-- If f(x) = ln x - ax is monotonically decreasing on (1, +∞), then a ≥ 1 -/
theorem monotone_decreasing_implies_a_ge_one (a : ℝ) :
  is_monotone_decreasing a → a ≥ 1 := by sorry

end monotone_decreasing_implies_a_ge_one_l1622_162211


namespace yearly_reading_pages_l1622_162291

/-- The number of pages read in a year, given the number of novels read per month,
    pages per novel, and months in a year. -/
def pages_read_in_year (novels_per_month : ℕ) (pages_per_novel : ℕ) (months_in_year : ℕ) : ℕ :=
  novels_per_month * pages_per_novel * months_in_year

/-- Theorem stating that reading 4 novels of 200 pages each month for 12 months
    results in reading 9600 pages in a year. -/
theorem yearly_reading_pages :
  pages_read_in_year 4 200 12 = 9600 := by
  sorry

end yearly_reading_pages_l1622_162291


namespace quadratic_solutions_l1622_162269

theorem quadratic_solutions : 
  (∃ (x : ℝ), x^2 - 8*x + 12 = 0) ∧ 
  (∃ (x : ℝ), x^2 - 2*x - 8 = 0) ∧ 
  ({x : ℝ | x^2 - 8*x + 12 = 0} = {2, 6}) ∧
  ({x : ℝ | x^2 - 2*x - 8 = 0} = {-2, 4}) :=
by sorry

end quadratic_solutions_l1622_162269


namespace all_natural_numbers_reachable_l1622_162221

-- Define the operations
def f (n : ℕ) : ℕ := 10 * n

def g (n : ℕ) : ℕ := 10 * n + 4

def h (n : ℕ) : ℕ := n / 2

-- Define the set of reachable numbers
inductive Reachable : ℕ → Prop where
  | start : Reachable 4
  | apply_f {n : ℕ} : Reachable n → Reachable (f n)
  | apply_g {n : ℕ} : Reachable n → Reachable (g n)
  | apply_h {n : ℕ} : Even n → Reachable n → Reachable (h n)

-- Theorem statement
theorem all_natural_numbers_reachable : ∀ m : ℕ, Reachable m := by
  sorry

end all_natural_numbers_reachable_l1622_162221


namespace intersection_of_A_and_B_l1622_162223

def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x^2 - 2*x - 3 < 0}

theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 1 3 := by sorry

end intersection_of_A_and_B_l1622_162223


namespace correct_balloons_left_l1622_162245

/-- Given the number of balloons of each color and the number of friends,
    calculate the number of balloons left after even distribution. -/
def balloons_left (yellow blue pink violet friends : ℕ) : ℕ :=
  let total := yellow + blue + pink + violet
  total % friends

theorem correct_balloons_left :
  balloons_left 20 24 50 102 9 = 7 := by
  sorry

end correct_balloons_left_l1622_162245


namespace probability_prime_8_sided_die_l1622_162274

-- Define a fair 8-sided die
def fair_8_sided_die : Finset ℕ := Finset.range 8

-- Define the set of prime numbers from 1 to 8
def primes_1_to_8 : Finset ℕ := {2, 3, 5, 7}

-- Theorem: The probability of rolling a prime number on a fair 8-sided die is 1/2
theorem probability_prime_8_sided_die :
  (Finset.card primes_1_to_8 : ℚ) / (Finset.card fair_8_sided_die : ℚ) = 1 / 2 := by
  sorry


end probability_prime_8_sided_die_l1622_162274


namespace no_single_non_divisible_l1622_162260

/-- Represents a 5x5 table of non-zero digits -/
def Table := Fin 5 → Fin 5 → Fin 9

/-- Checks if a number is divisible by 3 -/
def isDivisibleBy3 (n : ℕ) : Prop := n % 3 = 0

/-- Sums the digits in a row or column -/
def sumDigits (digits : Fin 5 → Fin 9) : ℕ :=
  (Finset.univ.sum fun i => (digits i).val) + 5

/-- Theorem stating the impossibility of having exactly one number not divisible by 3 -/
theorem no_single_non_divisible (t : Table) : 
  ¬ (∃! n : Fin 10, ¬ isDivisibleBy3 (sumDigits (fun i => 
    if n.val < 5 then t i n.val else t n.val (i - 5)))) := by
  sorry

end no_single_non_divisible_l1622_162260


namespace probability_of_matching_pair_l1622_162206

def num_blue_socks : ℕ := 12
def num_green_socks : ℕ := 10

def total_socks : ℕ := num_blue_socks + num_green_socks

def ways_to_pick_two (n : ℕ) : ℕ := n * (n - 1) / 2

def matching_pairs : ℕ := ways_to_pick_two num_blue_socks + ways_to_pick_two num_green_socks

def total_ways : ℕ := ways_to_pick_two total_socks

theorem probability_of_matching_pair :
  (matching_pairs : ℚ) / total_ways = 111 / 231 := by sorry

end probability_of_matching_pair_l1622_162206


namespace min_value_theorem_l1622_162294

theorem min_value_theorem (p q r s t u v w : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) 
  (ht : t > 0) (hu : u > 0) (hv : v > 0) (hw : w > 0)
  (h1 : p * q * r * s = 16) (h2 : t * u * v * w = 25) :
  (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 40 ∧
  ∃ (p' q' r' s' t' u' v' w' : ℝ),
    p' > 0 ∧ q' > 0 ∧ r' > 0 ∧ s' > 0 ∧ 
    t' > 0 ∧ u' > 0 ∧ v' > 0 ∧ w' > 0 ∧
    p' * q' * r' * s' = 16 ∧ t' * u' * v' * w' = 25 ∧
    (p' * t')^2 + (q' * u')^2 + (r' * v')^2 + (s' * w')^2 = 40 :=
by sorry

end min_value_theorem_l1622_162294


namespace sqrt_x_minus_one_real_l1622_162265

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by
  sorry

end sqrt_x_minus_one_real_l1622_162265


namespace partner_a_share_l1622_162286

/-- Calculates the share of a partner in a partnership based on investments and known share of another partner. -/
def calculate_share (investment_a investment_b investment_c : ℚ) (share_b : ℚ) : ℚ :=
  let total_investment := investment_a + investment_b + investment_c
  let ratio_a := investment_a / total_investment
  let ratio_b := investment_b / total_investment
  let total_profit := share_b / ratio_b
  ratio_a * total_profit

/-- Theorem stating that given the investments and b's share, a's share is approximately $560. -/
theorem partner_a_share (investment_a investment_b investment_c share_b : ℚ) 
  (h1 : investment_a = 7000)
  (h2 : investment_b = 11000)
  (h3 : investment_c = 18000)
  (h4 : share_b = 880) :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |calculate_share investment_a investment_b investment_c share_b - 560| < ε :=
sorry

end partner_a_share_l1622_162286


namespace point_transformation_l1622_162236

/-- Rotate a point (x, y) counterclockwise by 90° around (h, k) -/
def rotate90 (x y h k : ℝ) : ℝ × ℝ :=
  (h - (y - k), k + (x - h))

/-- Reflect a point (x, y) about the line y = -x -/
def reflectAboutNegX (x y : ℝ) : ℝ × ℝ :=
  (-y, -x)

theorem point_transformation (c d : ℝ) : 
  let (x₁, y₁) := rotate90 c d 2 3
  let (x₂, y₂) := reflectAboutNegX x₁ y₁
  (x₂ = 7 ∧ y₂ = -10) → d - c = -7 := by
  sorry

end point_transformation_l1622_162236


namespace max_consecutive_sum_l1622_162229

theorem max_consecutive_sum (n : ℕ) : (n * (n + 1)) / 2 ≤ 1000 ↔ n ≤ 44 := by sorry

end max_consecutive_sum_l1622_162229


namespace share_calculation_l1622_162251

theorem share_calculation (total : ℝ) (a b c : ℝ) 
  (h_total : total = 700)
  (h_a_b : a = (1/2) * b)
  (h_b_c : b = (1/2) * c)
  (h_sum : a + b + c = total) :
  c = 400 := by
sorry

end share_calculation_l1622_162251


namespace camel_cost_l1622_162253

/-- The cost of animals in a market --/
structure AnimalCosts where
  camel : ℕ
  horse : ℕ
  ox : ℕ
  elephant : ℕ

/-- The conditions of the animal costs problem --/
def animal_costs_conditions (costs : AnimalCosts) : Prop :=
  10 * costs.camel = 24 * costs.horse ∧
  16 * costs.horse = 4 * costs.ox ∧
  6 * costs.ox = 4 * costs.elephant ∧
  10 * costs.elephant = 150000

/-- The theorem stating that under the given conditions, a camel costs 6000 --/
theorem camel_cost (costs : AnimalCosts) : 
  animal_costs_conditions costs → costs.camel = 6000 := by
  sorry

end camel_cost_l1622_162253


namespace ellipse_slope_at_pi_third_l1622_162272

/-- Given an ellipse with parametric equations x = 2cos(t) and y = 4sin(t),
    prove that the slope of the line OM, where M is the point on the ellipse
    corresponding to t = π/3 and O is the origin, is 2√3. -/
theorem ellipse_slope_at_pi_third :
  let x : ℝ → ℝ := λ t ↦ 2 * Real.cos t
  let y : ℝ → ℝ := λ t ↦ 4 * Real.sin t
  let M : ℝ × ℝ := (x (π/3), y (π/3))
  let O : ℝ × ℝ := (0, 0)
  let slope := (M.2 - O.2) / (M.1 - O.1)
  slope = 2 * Real.sqrt 3 := by
sorry

end ellipse_slope_at_pi_third_l1622_162272


namespace am_gm_inequality_l1622_162292

theorem am_gm_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hab : a < b) (hbc : b < c) :
  ((a + c) / 2) - Real.sqrt (a * c) < (c - a)^2 / (8 * a) := by
sorry

end am_gm_inequality_l1622_162292


namespace students_not_enrolled_in_languages_l1622_162231

/-- Given a class with the following properties:
  * There are 150 students in total
  * 61 students are taking French
  * 32 students are taking German
  * 45 students are taking Spanish
  * 15 students are taking both French and German
  * 12 students are taking both French and Spanish
  * 10 students are taking both German and Spanish
  * 5 students are taking all three languages
  This theorem proves that the number of students not enrolled in any
  of these language courses is 44. -/
theorem students_not_enrolled_in_languages (total : ℕ) (french : ℕ) (german : ℕ) (spanish : ℕ)
  (french_and_german : ℕ) (french_and_spanish : ℕ) (german_and_spanish : ℕ) (all_three : ℕ)
  (h_total : total = 150)
  (h_french : french = 61)
  (h_german : german = 32)
  (h_spanish : spanish = 45)
  (h_french_and_german : french_and_german = 15)
  (h_french_and_spanish : french_and_spanish = 12)
  (h_german_and_spanish : german_and_spanish = 10)
  (h_all_three : all_three = 5) :
  total - (french + german + spanish - french_and_german - french_and_spanish - german_and_spanish + all_three) = 44 := by
  sorry

end students_not_enrolled_in_languages_l1622_162231


namespace greatest_integer_inequality_l1622_162205

theorem greatest_integer_inequality : ∃ (y : ℤ), (5 : ℚ) / 8 > (y : ℚ) / 17 ∧ 
  ∀ (z : ℤ), (5 : ℚ) / 8 > (z : ℚ) / 17 → z ≤ y :=
by
  -- The proof goes here
  sorry

end greatest_integer_inequality_l1622_162205


namespace meal_preparation_assignments_l1622_162243

theorem meal_preparation_assignments (n : ℕ) (h : n = 6) :
  (n.choose 3) * ((n - 3).choose 1) * ((n - 4).choose 2) = 60 := by
  sorry

end meal_preparation_assignments_l1622_162243


namespace calculation_proofs_l1622_162232

theorem calculation_proofs :
  (1.4 + (-0.2) + 0.6 + (-1.8) = 0) ∧
  ((-1/6 + 3/2 - 5/12) * (-48) = -44) ∧
  ((-1/3)^3 * (-3)^2 * (-1)^2011 = 1/3) ∧
  (-1^3 * (-5) / ((-3)^2 + 2 * (-5)) = -5) := by
  sorry

end calculation_proofs_l1622_162232


namespace probability_under_20_l1622_162277

theorem probability_under_20 (total : ℕ) (over_30 : ℕ) (h1 : total = 130) (h2 : over_30 = 90) :
  (total - over_30 : ℚ) / total = 4 / 13 := by
sorry

end probability_under_20_l1622_162277


namespace pears_for_apples_l1622_162276

/-- The cost of fruits in a common unit -/
structure FruitCost where
  apple : ℕ
  orange : ℕ
  pear : ℕ

/-- The relationship between apple and orange costs -/
def apple_orange_relation (fc : FruitCost) : Prop :=
  10 * fc.apple = 5 * fc.orange

/-- The relationship between orange and pear costs -/
def orange_pear_relation (fc : FruitCost) : Prop :=
  4 * fc.orange = 6 * fc.pear

/-- The main theorem: Nancy can buy 15 pears for the price of 20 apples -/
theorem pears_for_apples (fc : FruitCost) 
  (h1 : apple_orange_relation fc) 
  (h2 : orange_pear_relation fc) : 
  20 * fc.apple = 15 * fc.pear :=
by sorry

end pears_for_apples_l1622_162276


namespace remainder_problem_l1622_162285

theorem remainder_problem (x : ℕ) :
  x < 100 →
  x % 3 = 2 →
  x % 4 = 2 →
  x % 5 = 2 →
  x = 2 ∨ x = 62 :=
by sorry

end remainder_problem_l1622_162285


namespace stating_cubic_factorization_condition_l1622_162290

/-- Represents a cubic equation of the form x^3 + ax^2 + bx + c = 0 -/
structure CubicEquation (α : Type) [Field α] where
  a : α
  b : α
  c : α

/-- Represents the factored form (x^2 + m)(x + n) = 0 -/
structure FactoredForm (α : Type) [Field α] where
  m : α
  n : α

/-- 
Theorem stating the necessary and sufficient condition for a cubic equation 
to be factored into the given form
-/
theorem cubic_factorization_condition {α : Type} [Field α] (eq : CubicEquation α) :
  (∃ (ff : FactoredForm α), 
    ∀ (x : α), x^3 + eq.a * x^2 + eq.b * x + eq.c = 0 ↔ 
    (x^2 + ff.m) * (x + ff.n) = 0) ↔ 
  eq.c = eq.a * eq.b :=
sorry

end stating_cubic_factorization_condition_l1622_162290


namespace power_product_equals_78125_l1622_162298

theorem power_product_equals_78125 (a : ℕ) (h : a = 5) : a^3 * a^4 = 78125 := by
  sorry

end power_product_equals_78125_l1622_162298


namespace seed_mixture_problem_l1622_162208

/-- Proves that in a mixture of seed mixtures X and Y, where X is 40% ryegrass
    and Y is 25% ryegrass, if the final mixture contains 35% ryegrass,
    then the percentage of X in the final mixture is 200/3. -/
theorem seed_mixture_problem (x y : ℝ) :
  x + y = 100 →  -- x and y represent percentages of X and Y in the final mixture
  0.40 * x + 0.25 * y = 35 →  -- The final mixture contains 35% ryegrass
  x = 200 / 3 := by
  sorry

end seed_mixture_problem_l1622_162208


namespace normalized_coordinates_sum_of_squares_is_one_l1622_162259

/-- The sum of squares of normalized coordinates is 1 -/
theorem normalized_coordinates_sum_of_squares_is_one
  (a b : ℝ) -- Coordinates of point Q
  (d : ℝ) -- Distance from origin to Q
  (h_d : d = Real.sqrt (a^2 + b^2)) -- Definition of distance
  (u : ℝ) (h_u : u = b / d) -- Definition of u
  (v : ℝ) (h_v : v = a / d) -- Definition of v
  : u^2 + v^2 = 1 := by
  sorry

end normalized_coordinates_sum_of_squares_is_one_l1622_162259


namespace intersection_implies_sum_l1622_162222

def f (x a b : ℝ) : ℝ := -|x - a| + b
def g (x c d : ℝ) : ℝ := |x - c| - d

theorem intersection_implies_sum (a b c d : ℝ) :
  f 1 a b = 4 ∧ g 1 c d = 4 ∧ f 7 a b = 2 ∧ g 7 c d = 2 → a + c = 8 := by
  sorry

end intersection_implies_sum_l1622_162222


namespace ivan_walking_time_l1622_162261

/-- Represents the problem of determining how long Ivan Ivanovich walked. -/
theorem ivan_walking_time 
  (s : ℝ) -- Total distance from home to work
  (t : ℝ) -- Usual time taken by car
  (v : ℝ) -- Car's speed
  (u : ℝ) -- Ivan's walking speed
  (h1 : s = v * t) -- Total distance equals car speed times usual time
  (h2 : s = u * T + v * (t - T + 1/6)) -- Distance covered by walking and car
  (h3 : v * (1/12) = s - u * T) -- Car meets Ivan halfway through its usual journey
  (h4 : v > 0) -- Car speed is positive
  (h5 : u > 0) -- Walking speed is positive
  (h6 : v > u) -- Car is faster than walking
  : T = 55 := by
  sorry

#check ivan_walking_time

end ivan_walking_time_l1622_162261


namespace hexagon_ratio_theorem_l1622_162288

/-- Represents a hexagon with specific properties -/
structure Hexagon where
  /-- Total area of the hexagon in square units -/
  total_area : ℝ
  /-- Width of the hexagon -/
  width : ℝ
  /-- Height of the rectangle below PQ -/
  rect_height : ℝ
  /-- Area below PQ -/
  area_below_pq : ℝ
  /-- Ensures the hexagon consists of 7 unit squares -/
  area_constraint : total_area = 7
  /-- Ensures PQ bisects the hexagon area -/
  bisect_constraint : area_below_pq = total_area / 2
  /-- Ensures the triangle base is half the hexagon width -/
  triangle_base_constraint : width / 2 = width - (width / 2)

/-- The main theorem to prove -/
theorem hexagon_ratio_theorem (h : Hexagon) : 
  let xq := (h.area_below_pq - h.width * h.rect_height) / (h.width / 4)
  let qy := h.width - xq
  xq / qy = 3 := by
  sorry

end hexagon_ratio_theorem_l1622_162288


namespace total_owls_on_fence_l1622_162215

def initial_owls : ℕ := 3
def joining_owls : ℕ := 2

theorem total_owls_on_fence : initial_owls + joining_owls = 5 := by
  sorry

end total_owls_on_fence_l1622_162215


namespace infinite_geometric_series_sum_l1622_162213

/-- The sum of an infinite geometric series with first term 5/3 and common ratio 1/3 is 5/2. -/
theorem infinite_geometric_series_sum :
  let a : ℚ := 5/3  -- First term
  let r : ℚ := 1/3  -- Common ratio
  let S : ℚ := a / (1 - r)  -- Sum formula for infinite geometric series
  S = 5/2 := by
  sorry

end infinite_geometric_series_sum_l1622_162213


namespace irene_age_is_46_l1622_162257

/-- Given Eddie's age, calculate Irene's age based on the relationships between Eddie, Becky, and Irene. -/
def calculate_irene_age (eddie_age : ℕ) : ℕ :=
  let becky_age := eddie_age / 4
  2 * becky_age

/-- Theorem stating that given the conditions, Irene's age is 46. -/
theorem irene_age_is_46 :
  let eddie_age : ℕ := 92
  calculate_irene_age eddie_age = 46 := by
  sorry

#eval calculate_irene_age 92

end irene_age_is_46_l1622_162257


namespace cryptarithmetic_problem_l1622_162212

theorem cryptarithmetic_problem (A B C : ℕ) : 
  A < 10 → B < 10 → C < 10 →  -- Single-digit integers
  A ≠ B → A ≠ C → B ≠ C →     -- Unique digits
  A * B = 24 →                -- First equation
  A - C = 4 →                 -- Second equation
  C = 0 :=                    -- Conclusion
by
  sorry

end cryptarithmetic_problem_l1622_162212


namespace average_book_cost_l1622_162299

def initial_amount : ℕ := 236
def books_bought : ℕ := 6
def remaining_amount : ℕ := 14

theorem average_book_cost :
  (initial_amount - remaining_amount) / books_bought = 37 := by
  sorry

end average_book_cost_l1622_162299


namespace alberto_bjorn_difference_l1622_162216

/-- Represents a biker's travel distance over time --/
structure BikerTravel where
  miles : ℝ
  hours : ℝ

/-- Alberto's travel after 4 hours --/
def alberto : BikerTravel :=
  { miles := 60
  , hours := 4 }

/-- Bjorn's travel after 4 hours --/
def bjorn : BikerTravel :=
  { miles := 45
  , hours := 4 }

/-- The difference in miles traveled between two bikers --/
def mileDifference (a b : BikerTravel) : ℝ :=
  a.miles - b.miles

/-- Theorem stating the difference in miles traveled between Alberto and Bjorn --/
theorem alberto_bjorn_difference :
  mileDifference alberto bjorn = 15 := by
  sorry

end alberto_bjorn_difference_l1622_162216
