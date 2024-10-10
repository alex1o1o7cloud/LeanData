import Mathlib

namespace misread_number_calculation_l2438_243828

theorem misread_number_calculation (n : ℕ) (initial_avg correct_avg wrong_num : ℚ) 
  (h1 : n = 10)
  (h2 : initial_avg = 23)
  (h3 : correct_avg = 24)
  (h4 : wrong_num = 26) : 
  ∃ (actual_num : ℚ), 
    (n : ℚ) * correct_avg - (n : ℚ) * initial_avg = actual_num - wrong_num ∧ 
    actual_num = 36 := by
  sorry

end misread_number_calculation_l2438_243828


namespace odd_function_minimum_value_l2438_243867

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- F is defined as a linear combination of f and x, plus a constant -/
def F (f : ℝ → ℝ) (a b : ℝ) (x : ℝ) : ℝ :=
  a * f x + b * x + 1

theorem odd_function_minimum_value
    (f : ℝ → ℝ) (a b : ℝ)
    (h_odd : IsOdd f)
    (h_max : ∀ x > 0, F f a b x ≤ 2) :
    ∀ x < 0, F f a b x ≥ 0 :=
  sorry

end odd_function_minimum_value_l2438_243867


namespace root_sum_squares_l2438_243898

theorem root_sum_squares (p q r : ℝ) : 
  (p^3 - 15*p^2 + 25*p - 10 = 0) →
  (q^3 - 15*q^2 + 25*q - 10 = 0) →
  (r^3 - 15*r^2 + 25*r - 10 = 0) →
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 400 := by
sorry

end root_sum_squares_l2438_243898


namespace unique_intersection_l2438_243855

/-- The curve C in the xy-plane -/
def curve (x y : ℝ) : Prop :=
  y = x^2 ∧ x ∈ Set.Icc (-Real.sqrt 2) (Real.sqrt 2)

/-- The line l in the xy-plane -/
def line (x y : ℝ) : Prop :=
  y - x = 2

/-- The intersection point of the curve and the line -/
def intersection_point : ℝ × ℝ := (-1, 1)

theorem unique_intersection :
  ∃! p : ℝ × ℝ, curve p.1 p.2 ∧ line p.1 p.2 ∧ p = intersection_point := by
  sorry

end unique_intersection_l2438_243855


namespace m_range_theorem_l2438_243858

/-- The equation x^2 + mx + 1 = 0 has two real roots -/
def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

/-- ∀x ∈ ℝ, 4x^2 + 4(m-2)x + 1 ≠ 0 -/
def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

/-- The range of values for m is (1, 2) -/
def range_m : Set ℝ := { m | 1 < m ∧ m < 2 }

theorem m_range_theorem (m : ℝ) :
  (¬(p m ∧ q m)) ∧ (¬¬(q m)) → m ∈ range_m :=
by sorry

end m_range_theorem_l2438_243858


namespace constant_theta_and_z_forms_line_l2438_243891

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- The set of points satisfying θ = c and z = d -/
def ConstantThetaAndZ (c d : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.θ = c ∧ p.z = d}

/-- Definition of a line in cylindrical coordinates -/
def IsLine (S : Set CylindricalPoint) : Prop :=
  ∃ (a b : ℝ), ∀ p ∈ S, p.r = a * p.θ + b

theorem constant_theta_and_z_forms_line (c d : ℝ) :
  IsLine (ConstantThetaAndZ c d) := by
  sorry


end constant_theta_and_z_forms_line_l2438_243891


namespace cookie_radius_l2438_243877

theorem cookie_radius (x y : ℝ) :
  x^2 + y^2 + 26 = 6*x + 12*y →
  ∃ (center_x center_y : ℝ), (x - center_x)^2 + (y - center_y)^2 = 19 :=
by sorry

end cookie_radius_l2438_243877


namespace a_range_when_p_and_q_false_l2438_243884

/-- Proposition p: y = a^x is monotonically decreasing on ℝ -/
def p (a : ℝ) : Prop := a > 0 ∧ a ≠ 1 ∧ ∀ x y : ℝ, x < y → a^x > a^y

/-- Proposition q: y = log(ax^2 - x + a) has range ℝ -/
def q (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, a * x^2 - x + a > 0 ∧ Real.log (a * x^2 - x + a) = y

/-- If "p and q" is false, then a is in (0, 1/2] ∪ (1, ∞) -/
theorem a_range_when_p_and_q_false (a : ℝ) : ¬(p a ∧ q a) → (0 < a ∧ a ≤ 1/2) ∨ a > 1 := by
  sorry

end a_range_when_p_and_q_false_l2438_243884


namespace count_odd_coefficients_l2438_243844

/-- The number of odd coefficients in (x^2 + x + 1)^n -/
def odd_coefficients (n : ℕ+) : ℕ :=
  (2^n.val - 1) / 3 * 4 + 1

/-- Theorem stating the number of odd coefficients in (x^2 + x + 1)^n -/
theorem count_odd_coefficients (n : ℕ+) :
  odd_coefficients n = (2^n.val - 1) / 3 * 4 + 1 :=
by sorry

end count_odd_coefficients_l2438_243844


namespace coefficient_of_monomial_l2438_243810

theorem coefficient_of_monomial (a b : ℝ) :
  let expression := (4 * Real.pi * a^2 * b) / 5
  let coefficient := -(4 / 5) * Real.pi
  coefficient = expression / (a^2 * b) := by sorry

end coefficient_of_monomial_l2438_243810


namespace amount_of_b_l2438_243805

theorem amount_of_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 100) (h4 : (3/10) * a = (1/5) * b) : b = 60 := by
  sorry

end amount_of_b_l2438_243805


namespace arithmetic_sequence_term_count_l2438_243845

theorem arithmetic_sequence_term_count 
  (a₁ aₙ d : ℤ) 
  (h₁ : a₁ = -25)
  (h₂ : aₙ = 96)
  (h₃ : d = 7)
  (h₄ : aₙ = a₁ + (n - 1) * d)
  : n = 18 := by
  sorry

end arithmetic_sequence_term_count_l2438_243845


namespace buttons_solution_l2438_243880

def buttons_problem (mari kendra sue will lea : ℕ) : Prop :=
  mari = 8 ∧
  kendra = 5 * mari + 4 ∧
  sue = kendra / 2 ∧
  will = (5 * (kendra + sue)) / 2 ∧
  lea = will - will / 5

theorem buttons_solution :
  ∃ (mari kendra sue will lea : ℕ),
    buttons_problem mari kendra sue will lea ∧ lea = 132 := by
  sorry

end buttons_solution_l2438_243880


namespace line_through_point_with_triangle_area_l2438_243842

theorem line_through_point_with_triangle_area (x y : ℝ) :
  let P : ℝ × ℝ := (4/3, 2)
  let l : ℝ → ℝ → Prop := λ x y ↦ 6*x + 3*y - 14 = 0
  let A : ℝ × ℝ := (7/3, 0)
  let B : ℝ × ℝ := (0, 14/3)
  let O : ℝ × ℝ := (0, 0)
  l P.1 P.2 ∧
  l A.1 A.2 ∧
  l B.1 B.2 ∧
  A.1 > 0 ∧
  B.2 > 0 ∧
  (1/2 * A.1 * B.2 = 6) →
  l x y :=
by sorry

end line_through_point_with_triangle_area_l2438_243842


namespace ians_money_left_l2438_243800

/-- Calculates Ian's remaining money after expenses and taxes -/
def ians_remaining_money (total_hours : ℕ) (first_rate second_rate : ℚ) 
  (spending_ratio tax_rate : ℚ) (monthly_expense : ℚ) : ℚ :=
  let total_earnings := (first_rate * (total_hours / 2 : ℚ)) + (second_rate * (total_hours / 2 : ℚ))
  let spending := total_earnings * spending_ratio
  let taxes := total_earnings * tax_rate
  let total_deductions := spending + taxes + monthly_expense
  total_earnings - total_deductions

theorem ians_money_left :
  ians_remaining_money 8 18 22 (1/2) (1/10) 50 = 14 := by
  sorry

end ians_money_left_l2438_243800


namespace semicircle_radius_in_trapezoid_l2438_243802

/-- A trapezoid with specific measurements and an inscribed semicircle. -/
structure TrapezoidWithSemicircle where
  -- Define the trapezoid
  AB : ℝ
  CD : ℝ
  side1 : ℝ
  side2 : ℝ
  -- Conditions
  AB_eq : AB = 27
  CD_eq : CD = 45
  side1_eq : side1 = 13
  side2_eq : side2 = 15
  -- Semicircle properties
  semicircle_diameter : ℝ
  semicircle_diameter_eq : semicircle_diameter = AB
  tangential_to_CD : Bool -- represents that the semicircle is tangential to CD

/-- The radius of the semicircle in the trapezoid is 13.5. -/
theorem semicircle_radius_in_trapezoid (t : TrapezoidWithSemicircle) :
  t.semicircle_diameter / 2 = 13.5 := by
  sorry

end semicircle_radius_in_trapezoid_l2438_243802


namespace new_average_production_l2438_243825

def past_days : ℕ := 14
def past_average : ℝ := 60
def today_production : ℝ := 90

theorem new_average_production :
  let total_past_production : ℝ := past_days * past_average
  let total_production : ℝ := total_past_production + today_production
  let new_average : ℝ := total_production / (past_days + 1)
  new_average = 62 := by sorry

end new_average_production_l2438_243825


namespace percentage_problem_l2438_243875

theorem percentage_problem (x : ℝ) (h : 0.25 * x = 70) : x = 280 := by
  sorry

end percentage_problem_l2438_243875


namespace luncheon_cost_theorem_l2438_243886

/-- The cost of a luncheon item combination -/
structure LuncheonCost where
  sandwiches : ℕ
  coffees : ℕ
  pies : ℕ
  total : ℚ

/-- The given luncheon costs -/
def givenLuncheons : List LuncheonCost := [
  ⟨5, 9, 2, 595/100⟩,
  ⟨7, 12, 2, 790/100⟩,
  ⟨3, 5, 1, 350/100⟩
]

/-- The theorem to prove -/
theorem luncheon_cost_theorem (s c p : ℚ) 
  (h1 : 5*s + 9*c + 2*p = 595/100)
  (h2 : 7*s + 12*c + 2*p = 790/100)
  (h3 : 3*s + 5*c + p = 350/100) :
  s + c + p = 105/100 := by
  sorry

end luncheon_cost_theorem_l2438_243886


namespace smallest_perfect_square_factor_l2438_243859

def y : ℕ := 2^3 * 3^2 * 4^6 * 5^5 * 7^8 * 8^3 * 9^10 * 11^11

theorem smallest_perfect_square_factor (k : ℕ) : 
  (k > 0 ∧ ∃ m : ℕ, k * y = m^2 ∧ ∀ n : ℕ, 0 < n ∧ n < k → ¬∃ m : ℕ, n * y = m^2) ↔ k = 110 :=
sorry

end smallest_perfect_square_factor_l2438_243859


namespace mn_product_is_66_l2438_243888

/-- A parabola shifted from y = x^2 --/
structure ShiftedParabola where
  m : ℝ
  n : ℝ
  h_shift : ℝ := 3  -- left shift
  v_shift : ℝ := 2  -- upward shift

/-- The product of m and n for a parabola y = x^2 + mx + n
    obtained by shifting y = x^2 up by 2 units and left by 3 units --/
def mn_product (p : ShiftedParabola) : ℝ := p.m * p.n

/-- Theorem: The product mn equals 66 for the specified shifted parabola --/
theorem mn_product_is_66 (p : ShiftedParabola) : mn_product p = 66 := by
  sorry

end mn_product_is_66_l2438_243888


namespace fraction_simplification_l2438_243853

theorem fraction_simplification (x : ℝ) : (x + 2) / 4 + (3 - 4 * x) / 3 = (-13 * x + 18) / 12 := by
  sorry

end fraction_simplification_l2438_243853


namespace tree_height_after_two_years_l2438_243808

/-- The height of a tree that triples its height every year -/
def tree_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (3 ^ years)

/-- Theorem: A tree that triples its height every year and reaches 81 feet after 4 years
    will be 9 feet tall after 2 years -/
theorem tree_height_after_two_years
  (h : ∃ (initial_height : ℝ), tree_height initial_height 4 = 81) :
  ∃ (initial_height : ℝ), tree_height initial_height 2 = 9 :=
by
  sorry

end tree_height_after_two_years_l2438_243808


namespace quadratic_inequality_solution_l2438_243897

theorem quadratic_inequality_solution (x : ℝ) : 
  3 * x^2 + 4 * x - 9 < 0 ∧ x ≥ -2 → -2 ≤ x ∧ x < 1 := by
sorry

end quadratic_inequality_solution_l2438_243897


namespace profit_is_three_l2438_243866

/-- Calculates the profit from selling apples and oranges -/
def calculate_profit (apple_buy_price : ℚ) (apple_sell_price : ℚ) 
                     (orange_buy_price : ℚ) (orange_sell_price : ℚ)
                     (apples_sold : ℕ) (oranges_sold : ℕ) : ℚ :=
  let apple_profit := (apple_sell_price - apple_buy_price) * apples_sold
  let orange_profit := (orange_sell_price - orange_buy_price) * oranges_sold
  apple_profit + orange_profit

/-- Proves that the profit from selling 5 apples and 5 oranges is $3 -/
theorem profit_is_three :
  let apple_buy_price : ℚ := 3 / 2  -- $3 for 2 apples
  let apple_sell_price : ℚ := 2     -- $10 for 5 apples, so $2 each
  let orange_buy_price : ℚ := 9 / 10  -- $2.70 for 3 oranges
  let orange_sell_price : ℚ := 1
  let apples_sold : ℕ := 5
  let oranges_sold : ℕ := 5
  calculate_profit apple_buy_price apple_sell_price orange_buy_price orange_sell_price apples_sold oranges_sold = 3 := by
  sorry

end profit_is_three_l2438_243866


namespace total_discount_calculation_l2438_243838

theorem total_discount_calculation (cost_price_A cost_price_B cost_price_C : ℝ)
  (markup_percentage : ℝ) (loss_percentage_A loss_percentage_B loss_percentage_C : ℝ)
  (h1 : cost_price_A = 200)
  (h2 : cost_price_B = 150)
  (h3 : cost_price_C = 100)
  (h4 : markup_percentage = 0.5)
  (h5 : loss_percentage_A = 0.01)
  (h6 : loss_percentage_B = 0.03)
  (h7 : loss_percentage_C = 0.02) :
  let marked_price (cp : ℝ) := cp * (1 + markup_percentage)
  let selling_price (cp : ℝ) (loss : ℝ) := cp * (1 - loss)
  let discount (cp : ℝ) (loss : ℝ) := marked_price cp - selling_price cp loss
  discount cost_price_A loss_percentage_A +
  discount cost_price_B loss_percentage_B +
  discount cost_price_C loss_percentage_C = 233.5 :=
by sorry


end total_discount_calculation_l2438_243838


namespace expression_value_l2438_243874

theorem expression_value (a b c d x : ℝ) 
  (h1 : a + b = 0) 
  (h2 : c * d = 1) 
  (h3 : |x| = 3) : 
  10 * a + 10 * b + c * d * x = 3 ∨ 10 * a + 10 * b + c * d * x = -3 := by
  sorry

end expression_value_l2438_243874


namespace function_is_linear_l2438_243861

/-- Given a real number k, we define a function f that satisfies two conditions -/
def satisfies_conditions (k : ℝ) (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x^2 + 2*x*y + y^2) = (x + y) * (f x + f y)) ∧ 
  (∀ x : ℝ, |f x - k*x| ≤ |x^2 - x|)

/-- Theorem stating that if f satisfies the conditions, then f(x) = kx for all x ∈ ℝ -/
theorem function_is_linear (k : ℝ) (f : ℝ → ℝ) 
  (h : satisfies_conditions k f) : 
  ∀ x : ℝ, f x = k * x :=
sorry

end function_is_linear_l2438_243861


namespace actual_distance_traveled_l2438_243887

/-- Given a person walking at 4 km/hr, if increasing their speed to 5 km/hr
    would result in walking 6 km more in the same time, then the actual
    distance traveled is 24 km. -/
theorem actual_distance_traveled (actual_speed actual_distance : ℝ) 
    (h1 : actual_speed = 4)
    (h2 : actual_distance / actual_speed = (actual_distance + 6) / 5) :
  actual_distance = 24 := by
  sorry

end actual_distance_traveled_l2438_243887


namespace min_value_quadratic_roots_l2438_243819

theorem min_value_quadratic_roots (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ x : ℝ, x^2 + a*x + b = 0) →
  (∃ x : ℝ, x^2 + b*x + a = 0) →
  3*a + 2*b ≥ 20 := by
sorry

end min_value_quadratic_roots_l2438_243819


namespace no_real_solutions_implies_a_less_than_one_l2438_243820

theorem no_real_solutions_implies_a_less_than_one :
  (∀ x : ℝ, ¬∃ (y : ℝ), y^2 = x + 4 ∧ y = a - 1) → a < 1 :=
by sorry

end no_real_solutions_implies_a_less_than_one_l2438_243820


namespace bologna_sandwich_count_l2438_243863

/-- Represents the number of sandwiches of each type -/
structure SandwichCount where
  cheese : ℕ
  bologna : ℕ
  peanutButter : ℕ

/-- The ratio of sandwiches -/
def sandwichRatio : ℕ → SandwichCount
  | x => { cheese := 1, bologna := x, peanutButter := 8 }

/-- The total number of sandwiches -/
def totalSandwiches : ℕ := 80

theorem bologna_sandwich_count :
  ∃ x : ℕ, 
    let ratio := sandwichRatio x
    (ratio.cheese + ratio.bologna + ratio.peanutButter) * y = totalSandwiches →
    ratio.bologna * y = 8 := by
  sorry

end bologna_sandwich_count_l2438_243863


namespace distance_after_translation_l2438_243878

/-- Given two points A and B in a 2D plane, and a translation vector,
    prove that the distance between A and the translated B is √153. -/
theorem distance_after_translation :
  let A : ℝ × ℝ := (2, -2)
  let B : ℝ × ℝ := (8, 6)
  let translation : ℝ × ℝ := (-3, 4)
  let C : ℝ × ℝ := (B.1 + translation.1, B.2 + translation.2)
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 153 := by
  sorry

end distance_after_translation_l2438_243878


namespace bicycle_price_after_discounts_l2438_243851

def original_price : ℝ := 200
def tuesday_discount : ℝ := 0.40
def thursday_discount : ℝ := 0.25

theorem bicycle_price_after_discounts :
  original_price * (1 - tuesday_discount) * (1 - thursday_discount) = 90 := by
  sorry

end bicycle_price_after_discounts_l2438_243851


namespace amc8_participants_l2438_243809

/-- The number of mathematics students at Euclid Middle School taking the AMC 8 contest -/
def total_students (germain newton young gauss : ℕ) : ℕ :=
  germain + newton + young + gauss

/-- Theorem stating that the total number of students taking the AMC 8 contest is 38 -/
theorem amc8_participants : total_students 12 10 9 7 = 38 := by
  sorry

end amc8_participants_l2438_243809


namespace motorcyclist_hiker_meeting_time_l2438_243857

/-- Calculates the waiting time for a motorcyclist and hiker to meet given their speeds and initial separation time. -/
theorem motorcyclist_hiker_meeting_time 
  (hiker_speed : ℝ) 
  (motorcyclist_speed : ℝ) 
  (separation_time : ℝ) 
  (hᵢ : hiker_speed = 6) 
  (mᵢ : motorcyclist_speed = 30) 
  (tᵢ : separation_time = 12 / 60) : 
  (motorcyclist_speed * separation_time) / hiker_speed = 1 := by
  sorry

#eval (60 : ℕ)  -- Expected result in minutes

end motorcyclist_hiker_meeting_time_l2438_243857


namespace hall_mat_expenditure_l2438_243892

/-- The total expenditure to cover the floor of a rectangular hall with a mat -/
def total_expenditure (length width height cost_per_sqm : ℝ) : ℝ :=
  length * width * cost_per_sqm

/-- Theorem: The total expenditure to cover the floor of a rectangular hall
    with dimensions 20 m × 15 m × 5 m using a mat that costs Rs. 40 per square meter
    is equal to Rs. 12,000. -/
theorem hall_mat_expenditure :
  total_expenditure 20 15 5 40 = 12000 := by
  sorry

end hall_mat_expenditure_l2438_243892


namespace equilateral_triangle_perimeter_l2438_243830

/-- Given an equilateral triangle where the area is numerically twice the length of one of its sides,
    the perimeter of the triangle is 8√3 units. -/
theorem equilateral_triangle_perimeter : ∀ s : ℝ,
  s > 0 →  -- side length is positive
  (s^2 * Real.sqrt 3) / 4 = 2 * s →  -- area is twice the side length
  3 * s = 8 * Real.sqrt 3 :=
by sorry

end equilateral_triangle_perimeter_l2438_243830


namespace iced_coffee_consumption_ratio_l2438_243883

/-- Proves that the ratio of daily servings consumed to servings per container is 1:2 -/
theorem iced_coffee_consumption_ratio 
  (servings_per_bottle : ℕ) 
  (cost_per_bottle : ℚ) 
  (total_cost : ℚ) 
  (duration_weeks : ℕ) 
  (h1 : servings_per_bottle = 6)
  (h2 : cost_per_bottle = 3)
  (h3 : total_cost = 21)
  (h4 : duration_weeks = 2) :
  (total_cost / cost_per_bottle * servings_per_bottle) / (duration_weeks * 7) / servings_per_bottle = 1 / 2 := by
  sorry

end iced_coffee_consumption_ratio_l2438_243883


namespace arithmetic_mean_sqrt3_sqrt2_l2438_243823

theorem arithmetic_mean_sqrt3_sqrt2 :
  let a := Real.sqrt 3 + Real.sqrt 2
  let b := Real.sqrt 3 - Real.sqrt 2
  (a + b) / 2 = Real.sqrt 3 := by
sorry

end arithmetic_mean_sqrt3_sqrt2_l2438_243823


namespace abs_negative_two_l2438_243890

theorem abs_negative_two : |(-2 : ℝ)| = 2 := by
  sorry

end abs_negative_two_l2438_243890


namespace theater_seats_l2438_243846

theorem theater_seats (people_watching : ℕ) (empty_seats : ℕ) : 
  people_watching = 532 → empty_seats = 218 → people_watching + empty_seats = 750 := by
  sorry

end theater_seats_l2438_243846


namespace chairs_moved_by_alex_l2438_243868

/-- Given that Carey moves x chairs, Pat moves y chairs, and Alex moves z chairs,
    with a total of 74 chairs to be moved, prove that the number of chairs
    Alex moves is equal to 74 minus the sum of chairs moved by Carey and Pat. -/
theorem chairs_moved_by_alex (x y z : ℕ) (h : x + y + z = 74) :
  z = 74 - x - y := by sorry

end chairs_moved_by_alex_l2438_243868


namespace arithmetic_sequence_middle_term_l2438_243893

/-- An arithmetic sequence with 5 terms -/
structure ArithmeticSequence5 where
  a : ℝ  -- first term
  e : ℝ  -- last term
  y : ℝ  -- middle term

/-- Theorem: In an arithmetic sequence with 5 terms, where 12 is the first term,
    56 is the last term, and y is the middle term, y equals 34. -/
theorem arithmetic_sequence_middle_term 
  (seq : ArithmeticSequence5) 
  (h1 : seq.a = 12) 
  (h2 : seq.e = 56) : 
  seq.y = 34 := by
  sorry

end arithmetic_sequence_middle_term_l2438_243893


namespace consecutive_integers_sum_l2438_243850

theorem consecutive_integers_sum (a b c : ℤ) : 
  b = 19 ∧ c = b + 1 ∧ a = b - 1 → a + b + c = 57 := by
  sorry

end consecutive_integers_sum_l2438_243850


namespace division_problem_l2438_243856

theorem division_problem (a b c : ℚ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2/5) : 
  c / a = 5/6 := by
  sorry

end division_problem_l2438_243856


namespace parabola_directrix_l2438_243804

/-- The equation of the directrix of the parabola x² = 4y is y = -1 -/
theorem parabola_directrix (x y : ℝ) : 
  (∀ x y, x^2 = 4*y → ∃ k, y = -k ∧ k = 1) :=
sorry

end parabola_directrix_l2438_243804


namespace contrapositive_equivalence_l2438_243899

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a - 8 > b - 8) → ¬(a > b)) ↔ (a - 8 ≤ b - 8 → a ≤ b) := by sorry

end contrapositive_equivalence_l2438_243899


namespace sqrt_equation_solution_l2438_243821

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (10 + Real.sqrt x) = 4 → x = 36 := by
  sorry

end sqrt_equation_solution_l2438_243821


namespace largest_integer_satisfying_inequality_five_satisfies_inequality_five_is_largest_integer_l2438_243865

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, (x - 1 : ℚ) / 4 - 3 / 7 < 2 / 3 → x ≤ 5 :=
by sorry

theorem five_satisfies_inequality :
  (5 - 1 : ℚ) / 4 - 3 / 7 < 2 / 3 :=
by sorry

theorem five_is_largest_integer :
  ∃ x : ℤ, x = 5 ∧
    ((x - 1 : ℚ) / 4 - 3 / 7 < 2 / 3) ∧
    (∀ y : ℤ, y > x → (y - 1 : ℚ) / 4 - 3 / 7 ≥ 2 / 3) :=
by sorry

end largest_integer_satisfying_inequality_five_satisfies_inequality_five_is_largest_integer_l2438_243865


namespace continued_fraction_evaluation_l2438_243879

theorem continued_fraction_evaluation :
  let x : ℚ := 1 + (3 / (4 + (5 / (6 + (7/8)))))
  x = 85/52 := by
sorry

end continued_fraction_evaluation_l2438_243879


namespace product_with_seven_zeros_is_odd_l2438_243896

def binary_num (n : ℕ) : Prop := ∀ d : ℕ, d ∈ n.digits 2 → d = 0 ∨ d = 1

def count_zeros (n : ℕ) : ℕ := (n.digits 2).filter (· = 0) |>.length

theorem product_with_seven_zeros_is_odd (m : ℕ) :
  binary_num m →
  count_zeros (17 * m) = 7 →
  Odd (17 * m) :=
by sorry

end product_with_seven_zeros_is_odd_l2438_243896


namespace four_folds_result_l2438_243843

/-- Represents a square piece of paper. -/
structure Square :=
  (side : ℝ)
  (side_positive : side > 0)

/-- Represents a fold on the paper. -/
inductive Fold
  | Diagonal
  | Perpendicular

/-- Represents the pattern of creases on the unfolded paper. -/
structure CreasePattern :=
  (folds : List Fold)
  (is_symmetrical : Bool)
  (center_at_mean : Bool)

/-- Function to perform a single fold. -/
def fold (s : Square) : CreasePattern :=
  sorry

/-- Function to perform four folds. -/
def four_folds (s : Square) : CreasePattern :=
  sorry

/-- Theorem stating the result of folding a square paper four times. -/
theorem four_folds_result (s : Square) :
  let pattern := four_folds s
  pattern.is_symmetrical ∧ 
  pattern.center_at_mean ∧ 
  (∃ (d p : Fold), d = Fold.Diagonal ∧ p = Fold.Perpendicular ∧ d ∈ pattern.folds ∧ p ∈ pattern.folds) :=
by sorry

end four_folds_result_l2438_243843


namespace answer_determines_sanity_not_species_l2438_243812

-- Define the species of the interlocutor
inductive Species
| Human
| Ghoul

-- Define the mental state of the interlocutor
inductive MentalState
| Sane
| Insane

-- Define the possible answers
inductive Answer
| Yes
| No

-- Define a function that determines the answer based on species and mental state
def getAnswer (s : Species) (m : MentalState) : Answer :=
  match m with
  | MentalState.Sane => Answer.Yes
  | MentalState.Insane => Answer.No

-- Theorem stating that the answer determines sanity but not species
theorem answer_determines_sanity_not_species :
  ∀ (s1 s2 : Species) (m1 m2 : MentalState),
    getAnswer s1 m1 = getAnswer s2 m2 →
    m1 = m2 ∧ (s1 = s2 ∨ s1 ≠ s2) :=
by sorry

end answer_determines_sanity_not_species_l2438_243812


namespace line_through_origin_and_intersection_l2438_243895

/-- The equation of the line passing through the origin and the intersection point of two given lines -/
theorem line_through_origin_and_intersection (x y : ℝ) : 
  (x - 3*y + 4 = 0 ∧ 2*x + y + 5 = 0) → 
  (3*x + 19*y = 0) := by sorry

end line_through_origin_and_intersection_l2438_243895


namespace probability_at_least_seven_three_times_l2438_243826

/-- The probability of rolling at least a seven on a single roll of an 8-sided die -/
def p : ℚ := 1/4

/-- The number of rolls -/
def n : ℕ := 4

/-- The probability of rolling at least a seven at least three times in four rolls of an 8-sided die -/
theorem probability_at_least_seven_three_times : 
  (Finset.sum (Finset.range 2) (λ k => (n.choose (n - k)) * p^(n - k) * (1 - p)^k)) = 13/256 := by
  sorry

end probability_at_least_seven_three_times_l2438_243826


namespace inscribed_quadrilateral_side_lengths_l2438_243841

/-- A quadrilateral inscribed in a circle with given properties has specific side lengths -/
theorem inscribed_quadrilateral_side_lengths (R : ℝ) (d₁ d₂ : ℝ) :
  R = 25 →
  d₁ = 48 →
  d₂ = 40 →
  ∃ (a b c d : ℝ),
    a = 5 * Real.sqrt 10 ∧
    b = 9 * Real.sqrt 10 ∧
    c = 13 * Real.sqrt 10 ∧
    d = 15 * Real.sqrt 10 ∧
    a^2 + c^2 = d₁^2 ∧
    b^2 + d^2 = d₂^2 ∧
    a * c + b * d = d₁ * d₂ :=
by sorry

end inscribed_quadrilateral_side_lengths_l2438_243841


namespace division_equality_l2438_243873

theorem division_equality : (786^2 * 74) / 23592 = 1938.8 := by
  sorry

end division_equality_l2438_243873


namespace prob_snow_at_least_one_day_l2438_243840

-- Define the probabilities
def prob_snow_friday : ℝ := 0.30
def prob_snow_monday : ℝ := 0.45

-- Theorem statement
theorem prob_snow_at_least_one_day : 
  let prob_no_snow_friday := 1 - prob_snow_friday
  let prob_no_snow_monday := 1 - prob_snow_monday
  let prob_no_snow_both := prob_no_snow_friday * prob_no_snow_monday
  1 - prob_no_snow_both = 0.615 := by
  sorry

end prob_snow_at_least_one_day_l2438_243840


namespace sqrt_fraction_sum_l2438_243871

theorem sqrt_fraction_sum : Real.sqrt (25 / 36 + 16 / 9) = Real.sqrt 89 / 6 := by
  sorry

end sqrt_fraction_sum_l2438_243871


namespace seating_position_indeterminable_l2438_243831

/-- Represents a seat number as a pair of integers -/
def SeatNumber := ℤ × ℤ

/-- Represents a seating position as a row and column -/
structure SeatingPosition where
  row : ℤ
  column : ℤ

/-- Function that attempts to determine the seating position from a seat number -/
noncomputable def determineSeatingPosition (seatNumber : SeatNumber) : Option SeatingPosition :=
  sorry

/-- Theorem stating that it's not possible to determine the seating position
    from the seat number (2, 4) without additional information -/
theorem seating_position_indeterminable :
  ∀ (f : SeatNumber → Option SeatingPosition),
    ∃ (p1 p2 : SeatingPosition), p1 ≠ p2 ∧
      (f (2, 4) = some p1 ∨ f (2, 4) = some p2 ∨ f (2, 4) = none) :=
by
  sorry

end seating_position_indeterminable_l2438_243831


namespace only_event1_is_random_l2438_243818

-- Define the possible types of events
inductive EventType
  | Random
  | Certain
  | Impossible

-- Define the events
def event1 : EventType := EventType.Random
def event2 : EventType := EventType.Certain
def event3 : EventType := EventType.Impossible

-- Define a function to check if an event is random
def isRandomEvent (e : EventType) : Prop :=
  e = EventType.Random

-- Theorem statement
theorem only_event1_is_random :
  isRandomEvent event1 ∧ ¬isRandomEvent event2 ∧ ¬isRandomEvent event3 :=
by
  sorry


end only_event1_is_random_l2438_243818


namespace solution_of_equation_l2438_243837

theorem solution_of_equation (x : ℝ) : 
  (Real.sqrt (3 * x + 1) + Real.sqrt (3 * x + 6) = Real.sqrt (4 * x - 2) + Real.sqrt (4 * x + 3)) → x = 3 :=
by sorry

end solution_of_equation_l2438_243837


namespace factor_implies_p_value_l2438_243836

theorem factor_implies_p_value (m p : ℤ) : 
  (∃ k : ℤ, m^2 - p*m - 24 = (m - 8) * k) → p = 5 := by
  sorry

end factor_implies_p_value_l2438_243836


namespace scientific_notation_of_12000_l2438_243847

theorem scientific_notation_of_12000 :
  (12000 : ℝ) = 1.2 * (10 ^ 4) := by sorry

end scientific_notation_of_12000_l2438_243847


namespace greater_number_problem_l2438_243881

theorem greater_number_problem (x y : ℝ) : 
  y = 2 * x ∧ x + y = 96 → y = 64 := by
  sorry

end greater_number_problem_l2438_243881


namespace ryegrass_percentage_in_x_l2438_243860

/-- Represents the composition of a seed mixture -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The percentage of mixture X in the final blend -/
def x_percentage : ℝ := 13.333333333333332

/-- The percentage of ryegrass in the final blend -/
def final_ryegrass_percentage : ℝ := 27

/-- Seed mixture X -/
def mixture_x : SeedMixture where
  ryegrass := 40  -- This is what we want to prove
  bluegrass := 60
  fescue := 0

/-- Seed mixture Y -/
def mixture_y : SeedMixture where
  ryegrass := 25
  bluegrass := 0
  fescue := 75

theorem ryegrass_percentage_in_x : 
  (mixture_x.ryegrass * x_percentage + mixture_y.ryegrass * (100 - x_percentage)) / 100 = final_ryegrass_percentage := by
  sorry

#check ryegrass_percentage_in_x

end ryegrass_percentage_in_x_l2438_243860


namespace prob_ace_king_queen_same_suit_l2438_243811

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards dealt -/
def CardsDealt : ℕ := 3

/-- Represents the probability of drawing a specific Ace from a standard deck -/
def ProbFirstAce : ℚ := 1 / StandardDeck

/-- Represents the probability of drawing a specific King after an Ace is drawn -/
def ProbSecondKing : ℚ := 1 / (StandardDeck - 1)

/-- Represents the probability of drawing a specific Queen after an Ace and a King are drawn -/
def ProbThirdQueen : ℚ := 1 / (StandardDeck - 2)

/-- The probability of dealing an Ace, King, and Queen of the same suit in that order -/
def ProbAceKingQueen : ℚ := ProbFirstAce * ProbSecondKing * ProbThirdQueen

theorem prob_ace_king_queen_same_suit :
  ProbAceKingQueen = 1 / 132600 := by
  sorry

end prob_ace_king_queen_same_suit_l2438_243811


namespace three_pieces_per_box_l2438_243894

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the tape needed for a single box -/
def tapeForBox (box : BoxDimensions) : ℕ :=
  box.length + 2 * box.width

/-- The number of 15x30 boxes -/
def numSmallBoxes : ℕ := 5

/-- The number of 40x40 boxes -/
def numLargeBoxes : ℕ := 2

/-- The dimensions of the small boxes -/
def smallBox : BoxDimensions :=
  { length := 30, width := 15 }

/-- The dimensions of the large boxes -/
def largeBox : BoxDimensions :=
  { length := 40, width := 40 }

/-- The total amount of tape needed -/
def totalTape : ℕ := 540

/-- Theorem: Each box needs 3 pieces of tape -/
theorem three_pieces_per_box :
  (∃ (n : ℕ), n > 0 ∧
    n * (numSmallBoxes * tapeForBox smallBox + numLargeBoxes * tapeForBox largeBox) = totalTape * n ∧
    n * (numSmallBoxes + numLargeBoxes) = 3 * n * (numSmallBoxes + numLargeBoxes)) := by
  sorry


end three_pieces_per_box_l2438_243894


namespace train_crossing_time_l2438_243801

/-- Given a train crossing two platforms, calculate the time to cross the second platform -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (platform1_length : ℝ) 
  (platform2_length : ℝ) 
  (time1 : ℝ) 
  (h1 : train_length = 270) 
  (h2 : platform1_length = 120) 
  (h3 : platform2_length = 250) 
  (h4 : time1 = 15) : 
  (train_length + platform2_length) / ((train_length + platform1_length) / time1) = 20 := by
sorry

end train_crossing_time_l2438_243801


namespace february_to_january_sales_ratio_l2438_243870

/-- The ratio of window screens sold in February to January is 2:3 -/
theorem february_to_january_sales_ratio :
  ∀ (january february march : ℕ),
  february = march / 4 →
  march = 8800 →
  january + february + march = 12100 →
  (february : ℚ) / january = 2 / 3 := by
sorry

end february_to_january_sales_ratio_l2438_243870


namespace hyperbola_focus_product_l2438_243876

-- Define the hyperbola C
def hyperbola (x y : ℝ) (m : ℝ) : Prop :=
  x^2 / 9 - y^2 / m = 1

-- Define the foci F₁ and F₂
def is_focus (F : ℝ × ℝ) (m : ℝ) : Prop :=
  ∃ (x y : ℝ), hyperbola x y m ∧ 
  ((F.1 - x)^2 + (F.2 - y)^2 = 16 ∨ (F.1 - x)^2 + (F.2 - y)^2 = 16)

-- Define point P on the hyperbola
def point_on_hyperbola (P : ℝ × ℝ) (m : ℝ) : Prop :=
  hyperbola P.1 P.2 m

-- Define the dot product condition
def perpendicular_vectors (P F₁ F₂ : ℝ × ℝ) : Prop :=
  (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0

-- Define the directrix condition
def directrix_through_focus (F : ℝ × ℝ) : Prop :=
  F.1 = -4

-- Main theorem
theorem hyperbola_focus_product (m : ℝ) (P F₁ F₂ : ℝ × ℝ) :
  is_focus F₁ m →
  is_focus F₂ m →
  point_on_hyperbola P m →
  perpendicular_vectors P F₁ F₂ →
  (directrix_through_focus F₁ ∨ directrix_through_focus F₂) →
  ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 14^2 :=
sorry

end hyperbola_focus_product_l2438_243876


namespace max_value_sum_of_roots_l2438_243815

theorem max_value_sum_of_roots (a b c : ℝ) 
  (sum_eq : a + b + c = 3)
  (a_ge : a ≥ -1)
  (b_ge : b ≥ -2)
  (c_ge : c ≥ -3) :
  ∃ (max : ℝ), max = 3 * Real.sqrt 34 ∧
    Real.sqrt (2 * a + 2) + Real.sqrt (4 * b + 8) + Real.sqrt (6 * c + 18) ≤ max ∧
    ∃ (a' b' c' : ℝ), a' + b' + c' = 3 ∧ a' ≥ -1 ∧ b' ≥ -2 ∧ c' ≥ -3 ∧
      Real.sqrt (2 * a' + 2) + Real.sqrt (4 * b' + 8) + Real.sqrt (6 * c' + 18) = max :=
sorry

end max_value_sum_of_roots_l2438_243815


namespace total_roses_l2438_243807

/-- The total number of roses in a special n-gon garden -/
def roseCount (n : ℕ) : ℕ :=
  Nat.choose n 4 + Nat.choose (n - 1) 2

/-- Properties of the rose garden -/
structure RoseGarden (n : ℕ) where
  convex : n ≥ 4
  redRoses : Fin n → Unit  -- One red rose at each vertex
  paths : Fin n → Fin n → Unit  -- Path between each pair of vertices
  noTripleIntersection : Unit  -- No three paths intersect at a single point
  whiteRoses : Unit  -- One white/black rose in each region

/-- Theorem: The total number of roses in the garden is given by roseCount -/
theorem total_roses (n : ℕ) (garden : RoseGarden n) : 
  (Fin n → Unit) × Unit → ℕ :=
by sorry

end total_roses_l2438_243807


namespace arithmetic_progression_sine_squared_l2438_243869

theorem arithmetic_progression_sine_squared (x y z α : Real) : 
  (y = (x + z) / 2) →  -- x, y, z form an arithmetic progression
  (α = Real.arcsin (Real.sqrt 7 / 4)) →  -- α is defined as arcsin(√7/4)
  (8 / Real.sin y = 1 / Real.sin x + 1 / Real.sin z) →  -- 1/sin(x), 4/sin(y), 1/sin(z) form an arithmetic progression
  Real.sin y ^ 2 = 7 / 13 := by
sorry

end arithmetic_progression_sine_squared_l2438_243869


namespace smallest_angle_in_triangle_l2438_243834

theorem smallest_angle_in_triangle (a b c : ℝ) (h1 : a = 4) (h2 : b = 3) (h3 : c = 2) :
  let C := Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))
  C = Real.arccos (7/8) ∧ C ≤ Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)) ∧ C ≤ Real.arccos ((a^2 + c^2 - b^2) / (2*a*c)) :=
by sorry

end smallest_angle_in_triangle_l2438_243834


namespace exist_two_players_with_eight_points_no_undefeated_pair_from_sixth_round_l2438_243885

/-- Represents a chess tournament with the given rules --/
structure ChessTournament where
  players : ℕ
  rounds : ℕ
  win_points : ℚ
  draw_points : ℚ
  loss_points : ℚ
  bye_points : ℚ
  max_byes : ℕ

/-- Defines the specific tournament in the problem --/
def problem_tournament : ChessTournament :=
  { players := 29
  , rounds := 9
  , win_points := 1
  , draw_points := 1/2
  , loss_points := 0
  , bye_points := 1
  , max_byes := 1 }

/-- Represents the state of a player after a certain number of rounds --/
structure PlayerState where
  wins : ℕ
  losses : ℕ
  byes : ℕ

/-- Calculates the total points for a player --/
def total_points (t : ChessTournament) (p : PlayerState) : ℚ :=
  p.wins * t.win_points + p.losses * t.loss_points + min p.byes t.max_byes * t.bye_points

/-- Theorem stating that two players can have 8 points each before the final round --/
theorem exist_two_players_with_eight_points (t : ChessTournament) :
  t = problem_tournament →
  ∃ (p1 p2 : PlayerState),
    total_points t p1 = 8 ∧
    total_points t p2 = 8 ∧
    p1.wins + p1.losses + p1.byes < t.rounds ∧
    p2.wins + p2.losses + p2.byes < t.rounds :=
  sorry

/-- Theorem stating that from the 6th round, no two undefeated players can meet --/
theorem no_undefeated_pair_from_sixth_round (t : ChessTournament) :
  t = problem_tournament →
  ∀ (r : ℕ), r ≥ 6 →
  ¬∃ (p1 p2 : PlayerState),
    p1.wins = r - 1 ∧
    p2.wins = r - 1 ∧
    p1.losses = 0 ∧
    p2.losses = 0 :=
  sorry

end exist_two_players_with_eight_points_no_undefeated_pair_from_sixth_round_l2438_243885


namespace new_person_weight_l2438_243824

/-- Proves that the weight of a new person is 77 kg when they replace a 65 kg person in a group of 8,
    causing the average weight to increase by 1.5 kg -/
theorem new_person_weight (n : ℕ) (old_weight new_weight avg_increase : ℝ) 
  (h1 : n = 8)
  (h2 : old_weight = 65)
  (h3 : avg_increase = 1.5) :
  new_weight = old_weight + n * avg_increase :=
by sorry

end new_person_weight_l2438_243824


namespace base5_division_theorem_l2438_243827

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a number from base 10 to base 5 -/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 5) ((m % 5) :: acc)
  aux n []

theorem base5_division_theorem :
  let dividend := [4, 3, 0, 2]  -- 2034₅ in reverse order
  let divisor := [3, 2]         -- 23₅ in reverse order
  let quotient := [0, 4]        -- 40₅ in reverse order
  (base5ToBase10 dividend) / (base5ToBase10 divisor) = base5ToBase10 quotient :=
by sorry

end base5_division_theorem_l2438_243827


namespace smallest_divisible_by_20_and_63_l2438_243862

theorem smallest_divisible_by_20_and_63 : ∀ n : ℕ, n > 0 ∧ 20 ∣ n ∧ 63 ∣ n → n ≥ 1260 := by
  sorry

#check smallest_divisible_by_20_and_63

end smallest_divisible_by_20_and_63_l2438_243862


namespace fourth_root_simplification_l2438_243822

theorem fourth_root_simplification :
  (3^7 * 5^3 : ℝ)^(1/4) = 3 * (135 : ℝ)^(1/4) := by
  sorry

end fourth_root_simplification_l2438_243822


namespace mans_downstream_speed_l2438_243882

/-- Proves that given a man's upstream speed of 30 kmph and still water speed of 35 kmph, his downstream speed is 40 kmph. -/
theorem mans_downstream_speed 
  (upstream_speed : ℝ) 
  (still_water_speed : ℝ) 
  (h1 : upstream_speed = 30) 
  (h2 : still_water_speed = 35) : 
  still_water_speed + (still_water_speed - upstream_speed) = 40 := by
  sorry

#check mans_downstream_speed

end mans_downstream_speed_l2438_243882


namespace opposite_numbers_iff_different_sign_l2438_243872

/-- Two real numbers are opposite if and only if they differ only in sign -/
theorem opposite_numbers_iff_different_sign (a b : ℝ) : 
  (a = -b) ↔ (abs a = abs b) :=
sorry

end opposite_numbers_iff_different_sign_l2438_243872


namespace inequality_proof_l2438_243849

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  1 ≤ ((x + y) * (x^3 + y^3)) / ((x^2 + y^2)^2) ∧ 
  ((x + y) * (x^3 + y^3)) / ((x^2 + y^2)^2) ≤ 9/8 := by
  sorry

end inequality_proof_l2438_243849


namespace salary_change_percentage_l2438_243817

theorem salary_change_percentage (x : ℝ) : 
  (1 - x / 100) * (1 + x / 100) = 84 / 100 → x = 40 := by
  sorry

end salary_change_percentage_l2438_243817


namespace no_solution_for_f_l2438_243829

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Function f as defined in the problem -/
def f (n : ℕ+) : ℕ := (sumOfDigits n.val) * n.val

/-- 3-adic valuation of a natural number -/
def threeAdicVal (n : ℕ) : ℕ := sorry

/-- Main theorem: There is no positive integer n such that f(n) = 19091997 -/
theorem no_solution_for_f :
  ∀ n : ℕ+, f n ≠ 19091997 := by sorry

end no_solution_for_f_l2438_243829


namespace cube_difference_l2438_243816

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 40) : 
  a^3 - b^3 = 248.5 := by sorry

end cube_difference_l2438_243816


namespace parallel_transitive_perpendicular_lines_parallel_l2438_243889

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (lineParallel : Line → Line → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β γ : Plane)

-- Axioms for different objects
axiom different_lines : m ≠ n
axiom different_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ

-- Theorem 1: Transitivity of parallel planes
theorem parallel_transitive :
  parallel α β → parallel β γ → parallel α γ :=
sorry

-- Theorem 2: Lines perpendicular to the same plane are parallel
theorem perpendicular_lines_parallel :
  perpendicular m α → perpendicular n α → lineParallel m n :=
sorry

end parallel_transitive_perpendicular_lines_parallel_l2438_243889


namespace geometric_series_sum_l2438_243852

/-- The sum of the geometric series with a specific pattern -/
theorem geometric_series_sum : 
  ∑' k : ℕ, (3 ^ (2 ^ k)) / ((9 ^ (2 ^ k)) - 1) = 1 / 2 := by
  sorry

end geometric_series_sum_l2438_243852


namespace factor_81_minus_27x_cubed_l2438_243814

theorem factor_81_minus_27x_cubed (x : ℝ) : 81 - 27 * x^3 = 27 * (3 - x) * (9 + 3*x + x^2) := by
  sorry

end factor_81_minus_27x_cubed_l2438_243814


namespace figure_to_square_l2438_243854

/-- Represents a figure on a grid --/
structure GridFigure where
  area : ℕ

/-- Represents a cut of the figure --/
inductive Cut
  | Part : Cut

/-- Represents the result of cutting and arranging --/
inductive Arrangement
  | Square : Arrangement

/-- Theorem: If a grid figure's area is a perfect square, 
    it can be cut into three parts and arranged into a square --/
theorem figure_to_square (f : GridFigure) 
  (h : ∃ n : ℕ, f.area = n * n) : 
  ∃ (c1 c2 c3 : Cut) (arr : Arrangement), 
    arr = Arrangement.Square := by
  sorry

end figure_to_square_l2438_243854


namespace max_handshakes_l2438_243803

theorem max_handshakes (n : ℕ) (h : n = 60) : n * (n - 1) / 2 = 1770 := by
  sorry

end max_handshakes_l2438_243803


namespace fraction_simplification_l2438_243813

theorem fraction_simplification (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hsum : a + b + c ≠ 0) :
  (a^2 + a*b - b^2 + a*c) / (b^2 + b*c - c^2 + b*a) = (a - b) / (b - c) := by
  sorry

end fraction_simplification_l2438_243813


namespace jqk_base14_to_binary_digits_l2438_243839

def base14_to_decimal (j k q : ℕ) : ℕ := j * 14^2 + q * 14 + k

def count_binary_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log2 n + 1

theorem jqk_base14_to_binary_digits : 
  count_binary_digits (base14_to_decimal 11 13 12) = 11 := by
  sorry

end jqk_base14_to_binary_digits_l2438_243839


namespace unique_solution_l2438_243832

/-- Represents a single-digit integer (0 to 9) -/
def SingleDigit : Type := {n : ℕ // n ≤ 9}

/-- The equation 38A - B1 = 364 holds for single-digit integers A and B -/
def EquationHolds (A B : SingleDigit) : Prop :=
  380 + A.val - 10 * B.val - 1 = 364

theorem unique_solution :
  ∃! (A B : SingleDigit), EquationHolds A B ∧ A.val = 5 ∧ B.val = 2 := by
  sorry

end unique_solution_l2438_243832


namespace juan_peter_speed_difference_l2438_243833

/-- The speed difference between Juan and Peter -/
def speed_difference (juan_speed peter_speed : ℝ) : ℝ :=
  juan_speed - peter_speed

/-- The total distance traveled by Juan and Peter -/
def total_distance (juan_speed peter_speed : ℝ) (time : ℝ) : ℝ :=
  (juan_speed + peter_speed) * time

theorem juan_peter_speed_difference :
  ∃ (juan_speed : ℝ),
    speed_difference juan_speed 5.0 = 3 ∧
    total_distance juan_speed 5.0 1.5 = 19.5 := by
  sorry

end juan_peter_speed_difference_l2438_243833


namespace count_numeric_hex_500_l2438_243835

/-- Converts a positive integer to its hexadecimal representation --/
def to_hex (n : ℕ+) : List (Fin 16) :=
  sorry

/-- Checks if a hexadecimal digit is numeric (0-9) --/
def is_numeric_hex_digit (d : Fin 16) : Bool :=
  d.val < 10

/-- Checks if a hexadecimal number contains only numeric digits --/
def has_only_numeric_digits (h : List (Fin 16)) : Bool :=
  h.all is_numeric_hex_digit

/-- Counts numbers with only numeric hexadecimal digits up to n --/
def count_numeric_hex (n : ℕ+) : ℕ :=
  (List.range n).filter (fun i => has_only_numeric_digits (to_hex ⟨i + 1, by sorry⟩)) |>.length

theorem count_numeric_hex_500 : count_numeric_hex 500 = 199 :=
  sorry

end count_numeric_hex_500_l2438_243835


namespace bouncing_ball_distance_l2438_243864

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (bounceRatio : ℝ) (numBounces : ℕ) : ℝ :=
  let rec bounceSum (height : ℝ) (n : ℕ) : ℝ :=
    if n = 0 then 0
    else height + height * bounceRatio + bounceSum (height * bounceRatio) (n - 1)
  initialHeight + 2 * bounceSum initialHeight numBounces

/-- The bouncing ball problem -/
theorem bouncing_ball_distance :
  ∃ (d : ℝ), abs (d - totalDistance 20 (2/3) 4) < 0.5 ∧ Int.floor d = 80 := by
  sorry


end bouncing_ball_distance_l2438_243864


namespace bread_cost_l2438_243806

/-- The cost of each loaf of bread given the total number of loaves, 
    cost of peanut butter, initial amount of money, and amount left over. -/
theorem bread_cost (num_loaves : ℕ) (peanut_butter_cost initial_money leftover : ℚ) :
  num_loaves = 3 ∧ 
  peanut_butter_cost = 2 ∧ 
  initial_money = 14 ∧ 
  leftover = 5.25 →
  (initial_money - leftover - peanut_butter_cost) / num_loaves = 2.25 := by
  sorry

end bread_cost_l2438_243806


namespace rectangles_in_4x4_grid_l2438_243848

def grid_size : Nat := 4

-- Define the number of ways to choose 2 items from n items
def choose_two (n : Nat) : Nat :=
  n * (n - 1) / 2

-- Define the number of rectangles in a square grid
def num_rectangles (n : Nat) : Nat :=
  (choose_two n) * (choose_two n)

-- Theorem statement
theorem rectangles_in_4x4_grid :
  num_rectangles grid_size = 36 := by
  sorry

end rectangles_in_4x4_grid_l2438_243848
