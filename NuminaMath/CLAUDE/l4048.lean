import Mathlib

namespace lcm_225_624_l4048_404896

theorem lcm_225_624 : Nat.lcm 225 624 = 46800 := by
  sorry

end lcm_225_624_l4048_404896


namespace inverse_proportion_problem_l4048_404848

/-- Given that x and y are inversely proportional, prove that if x = 3y when x + y = 60, then y = 45 when x = 15. -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) 
  (h2 : ∃ x₀ y₀ : ℝ, x₀ = 3 * y₀ ∧ x₀ + y₀ = 60 ∧ x₀ * y₀ = k) :
  x = 15 → y = 45 := by
  sorry

end inverse_proportion_problem_l4048_404848


namespace train_length_calculation_l4048_404813

/-- The length of a train given its speed, the speed of a man it's passing, and the time it takes to cross the man completely. -/
theorem train_length_calculation (train_speed : ℝ) (man_speed : ℝ) (crossing_time : ℝ) :
  train_speed = 63 →
  man_speed = 3 →
  crossing_time = 53.99568034557235 →
  ∃ (train_length : ℝ), abs (train_length - 900) < 0.1 :=
by sorry

end train_length_calculation_l4048_404813


namespace cone_areas_l4048_404879

/-- Represents a cone with given slant height and height -/
structure Cone where
  slantHeight : ℝ
  height : ℝ

/-- Calculates the lateral area of a cone -/
def lateralArea (c : Cone) : ℝ := sorry

/-- Calculates the area of the sector when the cone's lateral surface is unfolded -/
def sectorArea (c : Cone) : ℝ := sorry

theorem cone_areas (c : Cone) (h1 : c.slantHeight = 1) (h2 : c.height = 0.8) : 
  lateralArea c = 3/5 * Real.pi ∧ sectorArea c = 3/5 * Real.pi := by sorry

end cone_areas_l4048_404879


namespace investment_time_period_l4048_404822

/-- 
Given:
- principal: The sum invested (in rupees)
- rate_difference: The difference in interest rates (as a decimal)
- interest_difference: The additional interest earned due to the higher rate (in rupees)

Proves that the time period for which the sum is invested is 2 years.
-/
theorem investment_time_period 
  (principal : ℝ) 
  (rate_difference : ℝ) 
  (interest_difference : ℝ) 
  (h1 : principal = 14000)
  (h2 : rate_difference = 0.03)
  (h3 : interest_difference = 840) :
  principal * rate_difference * 2 = interest_difference := by
  sorry

end investment_time_period_l4048_404822


namespace age_multiplier_problem_l4048_404807

theorem age_multiplier_problem (A : ℕ) (N : ℚ) : 
  A = 50 → (A + 5) * N - 5 * (A - 5) = A → N = 5 := by
sorry

end age_multiplier_problem_l4048_404807


namespace dress_pocket_ratio_l4048_404831

/-- Proves that the ratio of dresses with pockets to the total number of dresses is 1:2 --/
theorem dress_pocket_ratio :
  ∀ (total_dresses : ℕ) (dresses_with_pockets : ℕ) (total_pockets : ℕ),
    total_dresses = 24 →
    total_pockets = 32 →
    dresses_with_pockets * 2 = total_dresses * 1 →
    dresses_with_pockets * 2 + dresses_with_pockets * 3 = total_pockets * 3 →
    (dresses_with_pockets : ℚ) / total_dresses = 1 / 2 := by
  sorry

end dress_pocket_ratio_l4048_404831


namespace quadratic_rewrite_l4048_404860

theorem quadratic_rewrite :
  ∃ (a b c : ℤ), 
    (∀ x : ℝ, 4 * x^2 - 40 * x + 48 = (a * x + b)^2 + c) ∧
    a * b = -20 ∧
    c = -52 := by
  sorry

end quadratic_rewrite_l4048_404860


namespace remainder_of_a_l4048_404811

theorem remainder_of_a (a : ℤ) :
  (a^100 % 73 = 2) → (a^101 % 73 = 69) → (a % 73 = 71) := by
  sorry

end remainder_of_a_l4048_404811


namespace banana_orange_equivalence_l4048_404838

theorem banana_orange_equivalence (banana_value orange_value : ℚ) : 
  (3 / 4 : ℚ) * 12 * banana_value = 6 * orange_value →
  (2 / 3 : ℚ) * 9 * banana_value = 4 * orange_value := by
sorry

end banana_orange_equivalence_l4048_404838


namespace prove_weekly_earnings_l4048_404830

def total_earnings : ℕ := 133
def num_weeks : ℕ := 19
def weekly_earnings : ℚ := total_earnings / num_weeks

theorem prove_weekly_earnings : weekly_earnings = 7 := by
  sorry

end prove_weekly_earnings_l4048_404830


namespace first_digit_must_be_odd_l4048_404858

/-- Represents a permutation of digits 0 to 9 -/
def Permutation := Fin 10 → Fin 10

/-- Checks if a permutation contains each digit exactly once -/
def is_valid_permutation (p : Permutation) : Prop :=
  ∀ i j : Fin 10, p i = p j → i = j

/-- Calculates the sum A as described in the problem -/
def sum_A (p : Permutation) : ℕ :=
  (10 * p 0 + p 1) + (10 * p 2 + p 3) + (10 * p 4 + p 5) + (10 * p 6 + p 7) + (10 * p 8 + p 9)

/-- Calculates the sum B as described in the problem -/
def sum_B (p : Permutation) : ℕ :=
  (10 * p 1 + p 2) + (10 * p 3 + p 4) + (10 * p 5 + p 6) + (10 * p 7 + p 8)

theorem first_digit_must_be_odd (p : Permutation) 
  (h_valid : is_valid_permutation p) 
  (h_equal : sum_A p = sum_B p) : 
  ¬ Even (p 0) :=
by sorry

end first_digit_must_be_odd_l4048_404858


namespace negative_inequality_l4048_404823

theorem negative_inequality (a b : ℝ) (h : a > b) : -a < -b := by
  sorry

end negative_inequality_l4048_404823


namespace balloon_purchase_theorem_l4048_404845

/-- The price of a regular balloon -/
def regular_price : ℚ := 1

/-- The price of a discounted balloon -/
def discounted_price : ℚ := 1/2

/-- The total budget available -/
def budget : ℚ := 30

/-- The cost of a set of three balloons -/
def set_cost : ℚ := 2 * regular_price + discounted_price

/-- The number of balloons in a set -/
def balloons_per_set : ℕ := 3

/-- The maximum number of balloons that can be purchased -/
def max_balloons : ℕ := 36

theorem balloon_purchase_theorem : 
  (budget / set_cost : ℚ).floor * balloons_per_set = max_balloons :=
sorry

end balloon_purchase_theorem_l4048_404845


namespace tan_alpha_value_l4048_404889

theorem tan_alpha_value (α : Real) 
  (h : (Real.sin α - 2 * Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = -5) : 
  Real.tan α = -23/16 := by
  sorry

end tan_alpha_value_l4048_404889


namespace f_extrema_l4048_404834

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x ^ 2 + Real.sqrt 3 * Real.sin (2 * x) + 1

theorem f_extrema :
  let I : Set ℝ := Set.Icc 0 (Real.pi / 2)
  (∀ x ∈ I, f x ≥ 1) ∧
  (∀ x ∈ I, f x ≤ 3 + Real.sqrt 3) ∧
  (∃ x ∈ I, f x = 1) ∧
  (∃ x ∈ I, f x = 3 + Real.sqrt 3) :=
by sorry

end f_extrema_l4048_404834


namespace convex_quadrilateral_area_is_120_l4048_404829

def convex_quadrilateral_area (a b c d : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧  -- areas are positive
  a < d ∧ b < d ∧ c < d ∧          -- fourth triangle has largest area
  a = 10 ∧ b = 20 ∧ c = 30 →       -- given areas
  a + b + c + d = 120              -- total area

theorem convex_quadrilateral_area_is_120 :
  ∀ a b c d : ℝ, convex_quadrilateral_area a b c d :=
by
  sorry

end convex_quadrilateral_area_is_120_l4048_404829


namespace trajectory_equation_MN_range_l4048_404821

-- Define the circle P
structure CircleP where
  center : ℝ × ℝ
  passes_through_F : center.1^2 + center.2^2 = (center.1 - 1)^2 + center.2^2
  tangent_to_l : center.1 + 1 = abs center.2

-- Define the circle F
def circleF (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the trajectory C
def trajectoryC (x y : ℝ) : Prop := y^2 = 4*x

-- Define the intersection points M and N
structure Intersection (p : CircleP) where
  M : ℝ × ℝ
  N : ℝ × ℝ
  on_circle_P : (M.1 - p.center.1)^2 + (M.2 - p.center.2)^2 = (p.center.1 - 1)^2 + p.center.2^2
              ∧ (N.1 - p.center.1)^2 + (N.2 - p.center.2)^2 = (p.center.1 - 1)^2 + p.center.2^2
  on_circle_F : circleF M.1 M.2 ∧ circleF N.1 N.2

-- Theorem statements
theorem trajectory_equation (p : CircleP) : trajectoryC p.center.1 p.center.2 := by sorry

theorem MN_range (p : CircleP) (i : Intersection p) : 
  Real.sqrt 3 ≤ Real.sqrt ((i.M.1 - i.N.1)^2 + (i.M.2 - i.N.2)^2) ∧ 
  Real.sqrt ((i.M.1 - i.N.1)^2 + (i.M.2 - i.N.2)^2) < 2 := by sorry

end trajectory_equation_MN_range_l4048_404821


namespace unique_solution_equation_l4048_404847

theorem unique_solution_equation :
  ∃! x : ℝ, (x^18 + 1) * (x^16 + x^14 + x^12 + x^10 + x^8 + x^6 + x^4 + x^2 + 1) = 18 * x^9 :=
by sorry

end unique_solution_equation_l4048_404847


namespace bills_theorem_l4048_404810

/-- Represents the water and electricity bills for DingDing's family -/
structure Bills where
  may_water : ℝ
  may_total : ℝ
  june_water_increase : ℝ
  june_electricity_increase : ℝ

/-- Calculates the total bill for June -/
def june_total (b : Bills) : ℝ :=
  b.may_water * (1 + b.june_water_increase) + 
  (b.may_total - b.may_water) * (1 + b.june_electricity_increase)

/-- Calculates the total bill for May and June -/
def may_june_total (b : Bills) : ℝ :=
  b.may_total + june_total b

/-- Theorem stating the properties of the bills -/
theorem bills_theorem (b : Bills) 
  (h1 : b.may_total = 140)
  (h2 : b.june_water_increase = 0.1)
  (h3 : b.june_electricity_increase = 0.2) :
  june_total b = -0.1 * b.may_water + 168 ∧
  may_june_total b = 304 ↔ b.may_water = 40 := by
  sorry

end bills_theorem_l4048_404810


namespace crossnumber_puzzle_l4048_404846

/-- A number is a two-digit number if it's between 10 and 99 inclusive. -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The tens digit of a natural number -/
def tensDigit (n : ℕ) : ℕ := n / 10

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The statement of the crossnumber puzzle -/
theorem crossnumber_puzzle :
  ∃! (a b c d : ℕ),
    isTwoDigit a ∧ isTwoDigit b ∧ isTwoDigit c ∧ isTwoDigit d ∧
    Nat.Prime a ∧
    ∃ (m n p : ℕ), b = m^2 ∧ c = n^2 ∧ d = p^2 ∧
    tensDigit a = unitsDigit b ∧
    unitsDigit a = tensDigit d ∧
    c = d :=
sorry

end crossnumber_puzzle_l4048_404846


namespace monomial_sum_implies_m_pow_n_eq_nine_l4048_404898

/-- If the sum of a^(m-1)b^2 and (1/2)a^2b^n is a monomial, then m^n = 9 -/
theorem monomial_sum_implies_m_pow_n_eq_nine 
  (a b : ℝ) (m n : ℕ) 
  (h : ∃ (k : ℝ) (p q : ℕ), a^(m-1) * b^2 + (1/2) * a^2 * b^n = k * a^p * b^q) :
  m^n = 9 := by
sorry

end monomial_sum_implies_m_pow_n_eq_nine_l4048_404898


namespace total_revenue_over_three_days_l4048_404839

-- Define pie types
inductive PieType
  | Apple
  | Blueberry
  | Cherry

-- Define a structure for daily sales data
structure DailySales where
  apple_price : ℕ
  blueberry_price : ℕ
  cherry_price : ℕ
  apple_sold : ℕ
  blueberry_sold : ℕ
  cherry_sold : ℕ

def slices_per_pie : ℕ := 6

def day1_sales : DailySales := {
  apple_price := 5,
  blueberry_price := 6,
  cherry_price := 7,
  apple_sold := 12,
  blueberry_sold := 8,
  cherry_sold := 10
}

def day2_sales : DailySales := {
  apple_price := 6,
  blueberry_price := 7,
  cherry_price := 8,
  apple_sold := 15,
  blueberry_sold := 10,
  cherry_sold := 14
}

def day3_sales : DailySales := {
  apple_price := 4,
  blueberry_price := 7,
  cherry_price := 9,
  apple_sold := 18,
  blueberry_sold := 7,
  cherry_sold := 13
}

def calculate_daily_revenue (sales : DailySales) : ℕ :=
  sales.apple_price * slices_per_pie * sales.apple_sold +
  sales.blueberry_price * slices_per_pie * sales.blueberry_sold +
  sales.cherry_price * slices_per_pie * sales.cherry_sold

theorem total_revenue_over_three_days :
  calculate_daily_revenue day1_sales +
  calculate_daily_revenue day2_sales +
  calculate_daily_revenue day3_sales = 4128 := by
  sorry


end total_revenue_over_three_days_l4048_404839


namespace seahawks_touchdowns_l4048_404852

theorem seahawks_touchdowns 
  (total_points : ℕ)
  (field_goals : ℕ)
  (touchdown_points : ℕ)
  (field_goal_points : ℕ)
  (h1 : total_points = 37)
  (h2 : field_goals = 3)
  (h3 : touchdown_points = 7)
  (h4 : field_goal_points = 3) :
  (total_points - field_goals * field_goal_points) / touchdown_points = 4 := by
  sorry

end seahawks_touchdowns_l4048_404852


namespace central_projection_items_correct_l4048_404895

-- Define the set of all items
inductive Item : Type
  | Searchlight
  | CarLight
  | Sun
  | Moon
  | DeskLamp

-- Define a predicate for items that form central projections
def FormsCentralProjection (item : Item) : Prop :=
  match item with
  | Item.Searchlight => True
  | Item.CarLight => True
  | Item.Sun => False
  | Item.Moon => False
  | Item.DeskLamp => True

-- Define the set of items that form central projections
def CentralProjectionItems : Set Item :=
  {item : Item | FormsCentralProjection item}

-- Theorem statement
theorem central_projection_items_correct :
  CentralProjectionItems = {Item.Searchlight, Item.CarLight, Item.DeskLamp} := by
  sorry


end central_projection_items_correct_l4048_404895


namespace kristy_baked_69_cookies_l4048_404869

def cookie_problem (C : ℕ) : Prop :=
  let remaining_after_kristy := C - 3
  let remaining_after_brother := remaining_after_kristy / 2
  let remaining_after_friend1 := remaining_after_brother - 4
  let friend2_took := 2 * 4
  let friend2_returned := friend2_took / 4
  let remaining_after_friend2 := remaining_after_friend1 - (friend2_took - friend2_returned)
  let remaining_after_friend3 := remaining_after_friend2 - 8
  let remaining_after_friend4 := remaining_after_friend3 - 3
  let final_remaining := remaining_after_friend4 - 7
  2 * final_remaining = 10

theorem kristy_baked_69_cookies : ∃ C : ℕ, cookie_problem C ∧ C = 69 := by
  sorry

end kristy_baked_69_cookies_l4048_404869


namespace cone_surface_area_l4048_404837

/-- A cone with base radius 1 and lateral surface that unfolds into a semicircle has a total surface area of 3π. -/
theorem cone_surface_area (cone : Real → Real → Real) 
  (h1 : cone 1 2 = 2 * Real.pi) -- Lateral surface area
  (h2 : cone 0 1 = Real.pi) -- Base area
  : cone 0 1 + cone 1 2 = 3 * Real.pi := by
  sorry

end cone_surface_area_l4048_404837


namespace equation_identity_l4048_404826

theorem equation_identity (a b c x : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  a^2 * ((x - b) / (a - b)) * ((x - c) / (a - c)) +
  b^2 * ((x - a) / (b - a)) * ((x - c) / (b - c)) +
  c^2 * ((x - a) / (c - a)) * ((x - b) / (c - b)) = x^2 := by
  sorry

end equation_identity_l4048_404826


namespace equivalence_of_divisibility_conditions_l4048_404856

theorem equivalence_of_divisibility_conditions (f : ℕ → ℕ) :
  (∀ m n : ℕ+, m ≤ n → (f m + n : ℕ) ∣ (f n + m : ℕ)) ↔
  (∀ m n : ℕ+, m ≥ n → (f m + n : ℕ) ∣ (f n + m : ℕ)) :=
by sorry

end equivalence_of_divisibility_conditions_l4048_404856


namespace intersection_slope_l4048_404888

/-- The slope of the line passing through the intersection points of two circles -/
theorem intersection_slope (x y : ℝ) : 
  (x^2 + y^2 - 6*x + 4*y - 20 = 0) ∧ 
  (x^2 + y^2 - 16*x + 8*y + 40 = 0) → 
  (∃ m : ℝ, m = -5/2 ∧ 
    ∀ (x₁ y₁ x₂ y₂ : ℝ), 
      (x₁^2 + y₁^2 - 6*x₁ + 4*y₁ - 20 = 0) ∧ 
      (x₁^2 + y₁^2 - 16*x₁ + 8*y₁ + 40 = 0) ∧
      (x₂^2 + y₂^2 - 6*x₂ + 4*y₂ - 20 = 0) ∧ 
      (x₂^2 + y₂^2 - 16*x₂ + 8*y₂ + 40 = 0) ∧
      (x₁ ≠ x₂) →
      m = (y₂ - y₁) / (x₂ - x₁)) :=
sorry

end intersection_slope_l4048_404888


namespace school_council_composition_l4048_404805

theorem school_council_composition :
  -- Total number of classes
  ∀ (total_classes : ℕ),
  -- Number of students per council
  ∀ (students_per_council : ℕ),
  -- Number of classes with more girls than boys
  ∀ (classes_more_girls : ℕ),
  -- Number of boys and girls in Petya's class
  ∀ (petyas_class_boys petyas_class_girls : ℕ),
  -- Total number of boys and girls across all councils
  ∀ (total_boys total_girls : ℕ),

  total_classes = 20 →
  students_per_council = 5 →
  classes_more_girls = 15 →
  petyas_class_boys = 1 →
  petyas_class_girls = 4 →
  total_boys = total_girls →
  total_boys + total_girls = total_classes * students_per_council →

  -- Conclusion: In the remaining 4 classes, there are 19 boys and 1 girl
  ∃ (remaining_boys remaining_girls : ℕ),
    remaining_boys = 19 ∧
    remaining_girls = 1 ∧
    remaining_boys + remaining_girls = (total_classes - classes_more_girls - 1) * students_per_council :=
by sorry

end school_council_composition_l4048_404805


namespace four_distinct_three_digit_numbers_with_sum_divisibility_l4048_404861

theorem four_distinct_three_digit_numbers_with_sum_divisibility :
  ∃ (a b c d : ℕ),
    100 ≤ a ∧ a < 1000 ∧
    100 ≤ b ∧ b < 1000 ∧
    100 ≤ c ∧ c < 1000 ∧
    100 ≤ d ∧ d < 1000 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (b + c + d) % a = 0 ∧
    (a + c + d) % b = 0 ∧
    (a + b + d) % c = 0 ∧
    (a + b + c) % d = 0 := by
  sorry

end four_distinct_three_digit_numbers_with_sum_divisibility_l4048_404861


namespace fraction_simplification_l4048_404891

theorem fraction_simplification : (1 - 1/4) / (1 - 1/3) = 9/8 := by
  sorry

end fraction_simplification_l4048_404891


namespace unique_intersection_l4048_404892

/-- The quadratic function f(x) = bx^2 + bx + 2 -/
def f (b : ℝ) (x : ℝ) : ℝ := b * x^2 + b * x + 2

/-- The linear function g(x) = 2x + 4 -/
def g (x : ℝ) : ℝ := 2 * x + 4

/-- The discriminant of the quadratic equation resulting from equating f and g -/
def discriminant (b : ℝ) : ℝ := (b - 2)^2 + 8 * b

theorem unique_intersection (b : ℝ) : 
  (∃! x, f b x = g x) ↔ b = -2 := by sorry

end unique_intersection_l4048_404892


namespace probability_four_ones_in_five_rolls_prob_four_ones_in_five_rolls_l4048_404853

/-- The probability of rolling exactly 4 ones in 5 rolls of a fair six-sided die -/
theorem probability_four_ones_in_five_rolls : ℚ :=
  25 / 7776

/-- A fair six-sided die -/
def fair_six_sided_die : Finset ℕ := Finset.range 6

/-- The number of rolls -/
def num_rolls : ℕ := 5

/-- The number of desired ones -/
def desired_ones : ℕ := 4

/-- The probability of rolling a specific number on a fair six-sided die -/
def prob_single_roll (n : ℕ) : ℚ :=
  if n ∈ fair_six_sided_die then 1 / 6 else 0

/-- The main theorem: The probability of rolling exactly 4 ones in 5 rolls of a fair six-sided die is 25/7776 -/
theorem prob_four_ones_in_five_rolls :
  (Nat.choose num_rolls desired_ones) *
  (prob_single_roll 1) ^ desired_ones *
  (1 - prob_single_roll 1) ^ (num_rolls - desired_ones) =
  probability_four_ones_in_five_rolls := by sorry

end probability_four_ones_in_five_rolls_prob_four_ones_in_five_rolls_l4048_404853


namespace b_finishing_time_l4048_404894

/-- The number of days it takes B to finish the remaining work after A leaves -/
def days_for_B_to_finish (a_days b_days collab_days : ℚ) : ℚ :=
  let total_work := 1
  let a_rate := 1 / a_days
  let b_rate := 1 / b_days
  let combined_rate := a_rate + b_rate
  let work_done_together := combined_rate * collab_days
  let remaining_work := total_work - work_done_together
  remaining_work / b_rate

/-- Theorem stating that B will take 76/5 days to finish the remaining work -/
theorem b_finishing_time :
  days_for_B_to_finish 5 16 2 = 76 / 5 := by
  sorry

end b_finishing_time_l4048_404894


namespace fifteen_percent_less_than_80_l4048_404886

theorem fifteen_percent_less_than_80 : ∃ x : ℝ, x + (1/4) * x = 80 - 0.15 * 80 ∧ x = 54 := by
  sorry

end fifteen_percent_less_than_80_l4048_404886


namespace planes_parallel_if_perpendicular_to_same_line_l4048_404864

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_same_line 
  (m : Line) (α β : Plane) (hα : α ≠ β) :
  perpendicular m α → perpendicular m β → parallel α β :=
sorry

end planes_parallel_if_perpendicular_to_same_line_l4048_404864


namespace hexagon_area_sum_l4048_404825

/-- A regular hexagon with side length 3 -/
structure RegularHexagon where
  side_length : ℝ
  side_length_eq : side_length = 3

/-- The area of a regular hexagon can be expressed as √p + √q where p and q are positive integers -/
def hexagon_area (h : RegularHexagon) : ∃ (p q : ℕ+), Real.sqrt p.val + Real.sqrt q.val = (3 * Real.sqrt 3 / 2) * h.side_length ^ 2 :=
  sorry

/-- The sum of p and q is 297 -/
theorem hexagon_area_sum (h : RegularHexagon) : 
  ∃ (p q : ℕ+), (Real.sqrt p.val + Real.sqrt q.val = (3 * Real.sqrt 3 / 2) * h.side_length ^ 2) ∧ (p.val + q.val = 297) :=
  sorry

end hexagon_area_sum_l4048_404825


namespace at_least_one_even_difference_l4048_404816

theorem at_least_one_even_difference (n : ℕ) (a b : Fin (2*n+1) → ℤ) 
  (h : ∃ σ : Equiv.Perm (Fin (2*n+1)), ∀ i, b i = a (σ i)) :
  ∃ k : Fin (2*n+1), Even (a k - b k) := by
sorry

end at_least_one_even_difference_l4048_404816


namespace log_sqrt12_1728sqrt12_eq_7_l4048_404887

theorem log_sqrt12_1728sqrt12_eq_7 : Real.log (1728 * Real.sqrt 12) / Real.log (Real.sqrt 12) = 7 := by
  sorry

end log_sqrt12_1728sqrt12_eq_7_l4048_404887


namespace min_value_quadratic_l4048_404873

theorem min_value_quadratic (a b : ℝ) : 
  let f := fun x : ℝ ↦ x^2 - a*x + b
  (∃ r₁ ∈ Set.Icc (-1) 1, f r₁ = 0) →
  (∃ r₂ ∈ Set.Icc 1 2, f r₂ = 0) →
  ∃ m : ℝ, m = -1 ∧ ∀ x : ℝ, a - 2*b ≥ m :=
by sorry

end min_value_quadratic_l4048_404873


namespace lunch_cost_theorem_l4048_404843

/-- The cost of the Taco Grande Plate -/
def taco_grande_cost : ℝ := 8

/-- The cost of Mike's additional items -/
def mike_additional_cost : ℝ := 2 + 4 + 2

/-- Mike's total bill -/
def mike_bill : ℝ := taco_grande_cost + mike_additional_cost

/-- John's total bill -/
def john_bill : ℝ := taco_grande_cost

/-- The combined total cost of Mike and John's lunch -/
def combined_total_cost : ℝ := mike_bill + john_bill

theorem lunch_cost_theorem :
  (mike_bill = 2 * john_bill) → combined_total_cost = 24 := by
  sorry

#eval combined_total_cost

end lunch_cost_theorem_l4048_404843


namespace toy_difference_l4048_404841

/-- The number of toys each person has -/
structure ToyCount where
  mandy : ℕ
  anna : ℕ
  amanda : ℕ

/-- The conditions of the problem -/
def ProblemConditions (tc : ToyCount) : Prop :=
  tc.mandy = 20 ∧
  tc.anna = 3 * tc.mandy ∧
  tc.mandy + tc.anna + tc.amanda = 142 ∧
  tc.amanda > tc.anna

/-- The theorem to be proved -/
theorem toy_difference (tc : ToyCount) (h : ProblemConditions tc) : 
  tc.amanda - tc.anna = 2 := by
  sorry

end toy_difference_l4048_404841


namespace bike_ride_distance_l4048_404840

/-- Calculates the total distance ridden given a constant riding rate and time, including breaks -/
def total_distance (rate : ℚ) (total_time : ℚ) (break_time : ℚ) (num_breaks : ℕ) : ℚ :=
  rate * (total_time - (break_time * num_breaks))

/-- The theorem to be proved -/
theorem bike_ride_distance :
  let rate : ℚ := 2 / 10  -- 2 miles per 10 minutes
  let total_time : ℚ := 40  -- 40 minutes total time
  let break_time : ℚ := 5  -- 5 minutes per break
  let num_breaks : ℕ := 2  -- 2 breaks
  total_distance rate total_time break_time num_breaks = 6 := by
  sorry


end bike_ride_distance_l4048_404840


namespace claire_age_l4048_404820

def age_problem (gabriel fiona ethan claire : ℕ) : Prop :=
  (gabriel = fiona - 2) ∧
  (fiona = ethan + 5) ∧
  (ethan = claire + 6) ∧
  (gabriel = 21)

theorem claire_age :
  ∀ gabriel fiona ethan claire : ℕ,
  age_problem gabriel fiona ethan claire →
  claire = 12 := by
sorry

end claire_age_l4048_404820


namespace second_visit_date_l4048_404800

/-- Represents the bill amount for a single person --/
structure Bill :=
  (base : ℕ)
  (date : ℕ)

/-- The restaurant scenario --/
structure RestaurantScenario :=
  (first_visit : Bill)
  (second_visit : Bill)
  (num_friends : ℕ)
  (days_between : ℕ)

/-- The conditions of the problem --/
def problem_conditions (scenario : RestaurantScenario) : Prop :=
  scenario.num_friends = 3 ∧
  scenario.days_between = 4 ∧
  scenario.first_visit.base + scenario.first_visit.date = 168 ∧
  scenario.num_friends * scenario.second_visit.base + scenario.second_visit.date = 486 ∧
  scenario.first_visit.base = scenario.second_visit.base ∧
  scenario.second_visit.date = scenario.first_visit.date + scenario.days_between

/-- The theorem to prove --/
theorem second_visit_date (scenario : RestaurantScenario) :
  problem_conditions scenario → scenario.second_visit.date = 15 :=
by
  sorry

end second_visit_date_l4048_404800


namespace tournament_matches_l4048_404844

/-- Represents a single-elimination tournament -/
structure Tournament where
  initial_teams : ℕ
  matches_played : ℕ

/-- The number of teams remaining after playing a certain number of matches -/
def teams_remaining (t : Tournament) : ℕ :=
  t.initial_teams - t.matches_played

theorem tournament_matches (t : Tournament) 
  (h1 : t.initial_teams = 128)
  (h2 : teams_remaining t = 1) : 
  t.matches_played = 127 := by
sorry

end tournament_matches_l4048_404844


namespace sally_box_sales_l4048_404872

theorem sally_box_sales (saturday_sales : ℕ) : 
  (saturday_sales + (3 / 2 : ℚ) * saturday_sales = 150) → 
  saturday_sales = 60 := by
sorry

end sally_box_sales_l4048_404872


namespace wheelbarrow_sale_ratio_l4048_404867

def duck_price : ℕ := 10
def chicken_price : ℕ := 8
def ducks_sold : ℕ := 2
def chickens_sold : ℕ := 5
def additional_earnings : ℕ := 60

def total_earnings : ℕ := duck_price * ducks_sold + chicken_price * chickens_sold

def wheelbarrow_cost : ℕ := total_earnings / 2

theorem wheelbarrow_sale_ratio :
  (wheelbarrow_cost + additional_earnings) / wheelbarrow_cost = 3 := by
  sorry

end wheelbarrow_sale_ratio_l4048_404867


namespace complex_power_sum_l4048_404824

/-- Given a complex number z such that z + 1/z = 2cos(5°), 
    prove that z^1000 + 1/z^1000 = -2cos(40°) -/
theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^1000 + 1/z^1000 = -2 * Real.cos (40 * π / 180) := by
  sorry

end complex_power_sum_l4048_404824


namespace hyperbola_eccentricity_range_l4048_404859

theorem hyperbola_eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_intersect : ∃ (x y : ℝ), y = 2*x ∧ x^2/a^2 - y^2/b^2 = 1) :
  let e := Real.sqrt (1 + (b/a)^2)
  e > Real.sqrt 5 := by sorry

end hyperbola_eccentricity_range_l4048_404859


namespace cube_sum_reciprocal_l4048_404803

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = 3) : x^3 + 1/x^3 = 18 := by
  sorry

end cube_sum_reciprocal_l4048_404803


namespace count_initials_sets_l4048_404842

/-- The number of letters available for initials -/
def num_letters : ℕ := 10

/-- The number of letters in each set of initials -/
def initials_length : ℕ := 4

/-- The number of different four-letter sets of initials possible using letters A through J -/
theorem count_initials_sets : (num_letters ^ initials_length : ℕ) = 10000 := by
  sorry

end count_initials_sets_l4048_404842


namespace two_valid_numbers_l4048_404875

def digits (n : ℕ) : Finset ℕ :=
  (n.digits 10).toFinset

def valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ (digits n ∪ digits (n * n) = Finset.range 9)

theorem two_valid_numbers :
  {n : ℕ | valid_number n} = {567, 854} := by sorry

end two_valid_numbers_l4048_404875


namespace pablo_share_fraction_l4048_404836

/-- Represents the number of eggs each person has -/
structure EggDistribution :=
  (mia : ℕ)
  (sofia : ℕ)
  (pablo : ℕ)
  (juan : ℕ)

/-- The initial distribution of eggs -/
def initial_distribution (m : ℕ) : EggDistribution :=
  { mia := m
  , sofia := 3 * m
  , pablo := 12 * m
  , juan := 5 }

/-- The fraction of eggs Pablo gives to Sofia -/
def pablo_to_sofia_fraction (m : ℕ) : ℚ :=
  (4 * m + 5 : ℚ) / (48 * m : ℚ)

theorem pablo_share_fraction (m : ℕ) :
  let init := initial_distribution m
  let total := init.mia + init.sofia + init.pablo + init.juan
  let equal_share := total / 4
  let sofia_needs := equal_share - init.sofia
  sofia_needs / init.pablo = pablo_to_sofia_fraction m := by
  sorry

end pablo_share_fraction_l4048_404836


namespace password_20_combinations_l4048_404833

def password_combinations (n : ℕ) (k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem password_20_combinations :
  ∃ (k : ℕ), k ≤ 5 ∧ password_combinations 5 k = 20 ↔ k = 3 :=
sorry

end password_20_combinations_l4048_404833


namespace smaller_square_side_length_l4048_404876

/-- A square with side length 2 -/
structure Square :=
  (side : ℝ)
  (is_two : side = 2)

/-- An equilateral triangle with vertices P, T, U where T is on RS and U is on SQ of square PQRS -/
structure EquilateralTriangle (sq : Square) :=
  (P T U : ℝ × ℝ)
  (is_equilateral : sorry)
  (T_on_RS : sorry)
  (U_on_SQ : sorry)

/-- A smaller square with vertex R and a vertex on PT -/
structure SmallerSquare (sq : Square) (tri : EquilateralTriangle sq) :=
  (side : ℝ)
  (vertex_on_PT : sorry)
  (sides_parallel : sorry)

/-- The theorem stating the properties of the smaller square's side length -/
theorem smaller_square_side_length 
  (sq : Square) 
  (tri : EquilateralTriangle sq) 
  (small_sq : SmallerSquare sq tri) :
  ∃ (d e f : ℕ), 
    d > 0 ∧ e > 0 ∧ f > 0 ∧
    ¬ (∃ (p : ℕ), Prime p ∧ p^2 ∣ e) ∧
    small_sq.side = (d - Real.sqrt e) / f ∧
    d = 4 ∧ e = 10 ∧ f = 3 ∧
    d + e + f = 17 := by
  sorry

end smaller_square_side_length_l4048_404876


namespace line_contains_diameter_l4048_404812

/-- Given a circle with equation x^2 + y^2 - 2x + 6y + 8 = 0, 
    prove that the line 2x + y + 1 = 0 contains a diameter of the circle -/
theorem line_contains_diameter (x y : ℝ) :
  (x^2 + y^2 - 2*x + 6*y + 8 = 0) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 + y₁^2 - 2*x₁ + 6*y₁ + 8 = 0) ∧
    (x₂^2 + y₂^2 - 2*x₂ + 6*y₂ + 8 = 0) ∧
    (2*x₁ + y₁ + 1 = 0) ∧
    (2*x₂ + y₂ + 1 = 0) ∧
    ((x₁ - x₂)^2 + (y₁ - y₂)^2 = (2*1)^2 + (2*(-3))^2)) :=
by sorry

end line_contains_diameter_l4048_404812


namespace handshake_problem_l4048_404808

theorem handshake_problem (n : ℕ) (s : ℕ) : 
  n * (n - 1) / 2 + s = 159 → n = 18 ∧ s = 6 := by
  sorry

end handshake_problem_l4048_404808


namespace best_estimate_and_error_prob_l4048_404818

/-- Represents a measurement with an error margin and probability --/
structure Measurement where
  value : ℝ
  error_margin : ℝ
  error_prob : ℝ

/-- The problem setup --/
def river_length_problem (gsa awra : Measurement) : Prop :=
  gsa.value = 402 ∧
  gsa.error_margin = 0.5 ∧
  gsa.error_prob = 0.04 ∧
  awra.value = 403 ∧
  awra.error_margin = 0.5 ∧
  awra.error_prob = 0.04

/-- The theorem to prove --/
theorem best_estimate_and_error_prob
  (gsa awra : Measurement)
  (h : river_length_problem gsa awra) :
  ∃ (estimate error_prob : ℝ),
    estimate = 402.5 ∧
    error_prob = 0.04 :=
  sorry

end best_estimate_and_error_prob_l4048_404818


namespace proposition_p_and_q_true_l4048_404855

open Real

theorem proposition_p_and_q_true : 
  (∃ φ : ℝ, (φ = π / 2 ∧ 
    (∀ x : ℝ, sin (x + φ) = sin (-x - φ)) ∧
    (∃ ψ : ℝ, ψ ≠ π / 2 ∧ ∀ x : ℝ, sin (x + ψ) = sin (-x - ψ)))) ∧
  (∃ x₀ : ℝ, 0 < x₀ ∧ x₀ < π / 2 ∧ sin x₀ ≠ 1 / 2) :=
by sorry

end proposition_p_and_q_true_l4048_404855


namespace pattern_3_7_verify_other_pairs_l4048_404809

/-- The pattern function that transforms two numbers according to the given rule -/
def pattern (a b : ℕ) : ℕ := (a + b) * a - a

/-- The theorem stating that the pattern applied to (3, 7) results in 27 -/
theorem pattern_3_7 : pattern 3 7 = 27 := by
  sorry

/-- Verification of other given pairs -/
theorem verify_other_pairs :
  pattern 2 3 = 8 ∧
  pattern 4 5 = 32 ∧
  pattern 5 8 = 60 ∧
  pattern 6 7 = 72 ∧
  pattern 7 8 = 98 := by
  sorry

end pattern_3_7_verify_other_pairs_l4048_404809


namespace max_lessons_is_216_l4048_404862

/-- Represents the teacher's wardrobe and lesson capacity. -/
structure TeacherWardrobe where
  shirts : ℕ
  pants : ℕ
  shoes : ℕ
  jackets : ℕ
  lesson_count : ℕ

/-- Calculates the number of lessons possible with the given wardrobe. -/
def calculate_lessons (w : TeacherWardrobe) : ℕ :=
  2 * w.shirts * w.pants * w.shoes

/-- Checks if the wardrobe satisfies the given conditions. -/
def satisfies_conditions (w : TeacherWardrobe) : Prop :=
  w.jackets = 2 ∧
  calculate_lessons { w with shirts := w.shirts + 1 } = w.lesson_count + 36 ∧
  calculate_lessons { w with pants := w.pants + 1 } = w.lesson_count + 72 ∧
  calculate_lessons { w with shoes := w.shoes + 1 } = w.lesson_count + 54

/-- The theorem stating the maximum number of lessons. -/
theorem max_lessons_is_216 :
  ∃ (w : TeacherWardrobe), satisfies_conditions w ∧ w.lesson_count = 216 ∧
  ∀ (w' : TeacherWardrobe), satisfies_conditions w' → w'.lesson_count ≤ 216 :=
sorry

end max_lessons_is_216_l4048_404862


namespace right_triangles_common_hypotenuse_l4048_404893

-- Define the triangles and their properties
def triangle_ABC (a : ℝ) := {BC : ℝ // BC = 2 ∧ ∃ (AC : ℝ), AC = a}
def triangle_ABD := {AD : ℝ // AD = 3}

-- Define the theorem
theorem right_triangles_common_hypotenuse (a : ℝ) 
  (h : a ≥ Real.sqrt 5) -- Ensure BD is real
  (ABC : triangle_ABC a) (ABD : triangle_ABD) :
  ∃ (BD : ℝ), BD = Real.sqrt (a^2 - 5) :=
sorry

end right_triangles_common_hypotenuse_l4048_404893


namespace n_divided_by_six_l4048_404815

theorem n_divided_by_six (n : ℕ) (h : n = 6^2024) : n / 6 = 6^2023 := by
  sorry

end n_divided_by_six_l4048_404815


namespace fair_entrance_fee_l4048_404801

/-- Represents the entrance fee structure and ride costs at a fair --/
structure FairPrices where
  under18Fee : ℝ
  over18Fee : ℝ
  rideCost : ℝ
  under18Fee_pos : 0 < under18Fee
  over18Fee_eq : over18Fee = 1.2 * under18Fee
  rideCost_eq : rideCost = 0.5

/-- Calculates the total cost for a group at the fair --/
def totalCost (prices : FairPrices) (numUnder18 : ℕ) (numOver18 : ℕ) (totalRides : ℕ) : ℝ :=
  numUnder18 * prices.under18Fee + numOver18 * prices.over18Fee + totalRides * prices.rideCost

/-- The main theorem stating the entrance fee for persons under 18 --/
theorem fair_entrance_fee :
  ∃ (prices : FairPrices), totalCost prices 2 1 9 = 20.5 ∧ prices.under18Fee = 5 := by
  sorry

end fair_entrance_fee_l4048_404801


namespace inverse_variation_cube_l4048_404814

theorem inverse_variation_cube (y : ℝ → ℝ) (k : ℝ) :
  (∀ x, x ≠ 0 → 3 * (y x) = k / (x^3)) →  -- 3y varies inversely as the cube of x
  y 3 = 27 →                              -- y = 27 when x = 3
  y 9 = 1 :=                              -- y = 1 when x = 9
by
  sorry

end inverse_variation_cube_l4048_404814


namespace smallest_n_divisibility_l4048_404806

theorem smallest_n_divisibility : ∃ (n : ℕ), n = 4058209 ∧
  (∃ (m : ℤ), n + 2015 = 2016 * m) ∧
  (∃ (k : ℤ), n + 2016 = 2015 * k) ∧
  (∀ (n' : ℕ), n' < n →
    (∃ (m : ℤ), n' + 2015 = 2016 * m) →
    (∃ (k : ℤ), n' + 2016 = 2015 * k) → False) :=
by sorry

end smallest_n_divisibility_l4048_404806


namespace cookie_jar_spending_ratio_l4048_404851

theorem cookie_jar_spending_ratio (initial_amount : ℕ) (doris_spent : ℕ) (final_amount : ℕ) : 
  initial_amount = 21 →
  doris_spent = 6 →
  final_amount = 12 →
  (initial_amount - doris_spent - final_amount) / doris_spent = 1 / 2 :=
by
  sorry

end cookie_jar_spending_ratio_l4048_404851


namespace area_of_non_intersecting_graphs_l4048_404881

/-- The area of the set A of points (a, b) such that the graphs of 
    f(x) = x^2 - 2ax + 1 and g(x) = 2b(a-x) do not intersect is π. -/
theorem area_of_non_intersecting_graphs (a b x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 - 2*a*x + 1
  let g : ℝ → ℝ := λ x => 2*b*(a-x)
  let A : Set (ℝ × ℝ) := {(a, b) | ∀ x, f x ≠ g x}
  MeasureTheory.volume A = π := by
sorry

end area_of_non_intersecting_graphs_l4048_404881


namespace chinese_barrel_stack_l4048_404863

/-- Calculates the total number of barrels in a terraced stack --/
def totalBarrels (a b n : ℕ) : ℕ :=
  let c := a + n - 1
  let d := b + n - 1
  (n * ((2 * a + c) * b + (2 * c + a) * d + (d - b))) / 6

/-- The problem statement --/
theorem chinese_barrel_stack : totalBarrels 2 1 15 = 1360 := by
  sorry

end chinese_barrel_stack_l4048_404863


namespace erased_number_l4048_404802

theorem erased_number (a : ℤ) (b : ℤ) (h1 : -4 ≤ b ∧ b ≤ 4) 
  (h2 : 8 * a - b = 1703) : a + b = 214 := by
  sorry

end erased_number_l4048_404802


namespace ellipse_equation_l4048_404883

/-- Given an ellipse with foci at (-2,0) and (2,0) passing through the point (2√3, √3),
    its standard equation is x²/16 + y²/12 = 1 -/
theorem ellipse_equation (x y : ℝ) :
  let f1 : ℝ × ℝ := (-2, 0)
  let f2 : ℝ × ℝ := (2, 0)
  let p : ℝ × ℝ := (2 * Real.sqrt 3, Real.sqrt 3)
  let d1 := Real.sqrt ((x - f1.1)^2 + (y - f1.2)^2)
  let d2 := Real.sqrt ((x - f2.1)^2 + (y - f2.2)^2)
  let passing_point := d1 + d2 = Real.sqrt ((p.1 - f1.1)^2 + (p.2 - f1.2)^2) +
                                 Real.sqrt ((p.1 - f2.1)^2 + (p.2 - f2.2)^2)
  passing_point → x^2 / 16 + y^2 / 12 = 1 :=
by
  sorry

end ellipse_equation_l4048_404883


namespace solve_equation_l4048_404897

theorem solve_equation : ∃ x : ℝ, 
  ((0.66^3 - 0.1^3) / 0.66^2) + x + 0.1^2 = 0.5599999999999999 ∧ 
  x = -0.107504 := by
sorry

end solve_equation_l4048_404897


namespace no_prime_roots_for_quadratic_l4048_404827

/-- A quadratic equation x^2 - 101x + k = 0 where k is an integer -/
def quadratic_equation (k : ℤ) (x : ℝ) : Prop :=
  x^2 - 101*x + k = 0

/-- Definition of a prime number -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- The main theorem stating that no integer k exists such that both roots of the quadratic equation are prime -/
theorem no_prime_roots_for_quadratic :
  ¬∃ (k : ℤ), ∃ (p q : ℕ), 
    is_prime p ∧ is_prime q ∧
    quadratic_equation k (p : ℝ) ∧ quadratic_equation k (q : ℝ) ∧
    p ≠ q :=
sorry

end no_prime_roots_for_quadratic_l4048_404827


namespace cylindrical_tin_height_l4048_404868

/-- The height of a cylindrical tin given its diameter and volume -/
theorem cylindrical_tin_height (diameter : ℝ) (volume : ℝ) (h_diameter : diameter = 8) (h_volume : volume = 80) :
  (volume / (π * (diameter / 2)^2)) = 80 / (π * 4^2) :=
by sorry

end cylindrical_tin_height_l4048_404868


namespace b_current_age_l4048_404865

/-- Given two people A and B, proves B's current age is 39 years -/
theorem b_current_age (a b : ℕ) : 
  (a + 10 = 2 * (b - 10)) →  -- A's age in 10 years equals twice B's age 10 years ago
  (a = b + 9) →              -- A is currently 9 years older than B
  b = 39 := by               -- B's current age is 39 years
sorry


end b_current_age_l4048_404865


namespace ivans_initial_money_l4048_404877

theorem ivans_initial_money (initial_money : ℝ) : 
  (4/5 * initial_money - 5 = 3) → initial_money = 10 := by
  sorry

end ivans_initial_money_l4048_404877


namespace infinite_series_convergence_l4048_404890

theorem infinite_series_convergence : 
  let f (n : ℕ) := (n^3 + 2*n^2 + 5*n + 2) / (3^n * (n^3 + 3))
  ∑' (n : ℕ), f n = 1/2 := by sorry

end infinite_series_convergence_l4048_404890


namespace intersection_point_l4048_404835

-- Define the system of equations
def system (x y m n : ℝ) : Prop :=
  2 * x + y = m ∧ x - y = n

-- Define the solution to the system
def solution : ℝ × ℝ := (-1, 3)

-- Define the lines
def line1 (x y m : ℝ) : Prop := y = -2 * x + m
def line2 (x y n : ℝ) : Prop := y = x - n

-- Theorem statement
theorem intersection_point :
  ∀ (m n : ℝ),
  system (solution.1) (solution.2) m n →
  ∃ (x y : ℝ), 
    line1 x y m ∧ 
    line2 x y n ∧ 
    x = solution.1 ∧ 
    y = solution.2 := by
  sorry

end intersection_point_l4048_404835


namespace tangent_line_sin_plus_one_l4048_404804

/-- The equation of the tangent line to y = sin x + 1 at (0, 1) is x - y + 1 = 0 -/
theorem tangent_line_sin_plus_one (x y : ℝ) : 
  let f : ℝ → ℝ := λ t => Real.sin t + 1
  let df : ℝ → ℝ := λ t => Real.cos t
  let tangent_point : ℝ × ℝ := (0, 1)
  let tangent_slope : ℝ := df tangent_point.1
  x - y + 1 = 0 ↔ y = tangent_slope * (x - tangent_point.1) + tangent_point.2 :=
by
  sorry

#check tangent_line_sin_plus_one

end tangent_line_sin_plus_one_l4048_404804


namespace triangle_inequality_l4048_404849

theorem triangle_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_sum : a + b + c = 1) : 
  5 * (a^2 + b^2 + c^2) + 18 * a * b * c ≥ 7/3 := by
  sorry

end triangle_inequality_l4048_404849


namespace unique_a_value_l4048_404882

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 4

-- Define the condition for the function to be positive outside [2, 8]
def positive_outside (a : ℝ) : Prop :=
  ∀ x, (x < 2 ∨ x > 8) → f a x > 0

-- Theorem statement
theorem unique_a_value : ∃! a : ℝ, positive_outside a :=
  sorry

end unique_a_value_l4048_404882


namespace pencil_distribution_l4048_404854

theorem pencil_distribution (boxes : Real) (pencils_per_box : Real) (students : Nat) :
  boxes = 4.0 →
  pencils_per_box = 648.0 →
  students = 36 →
  (boxes * pencils_per_box) / students = 72 := by
  sorry

end pencil_distribution_l4048_404854


namespace scout_troop_profit_l4048_404899

-- Define the parameters
def num_bars : ℕ := 1500
def purchase_rate : ℚ := 1 / 3
def selling_rate : ℚ := 3 / 4
def fixed_cost : ℚ := 50

-- Define the profit calculation
def profit : ℚ :=
  num_bars * selling_rate - (num_bars * purchase_rate + fixed_cost)

-- Theorem statement
theorem scout_troop_profit : profit = 575 := by
  sorry

end scout_troop_profit_l4048_404899


namespace complex_root_magnitude_l4048_404817

theorem complex_root_magnitude (n : ℕ) (a : ℝ) (z : ℂ) 
  (h1 : n ≥ 2) 
  (h2 : 0 < a) 
  (h3 : a < (n + 1 : ℝ) / (n - 1 : ℝ)) 
  (h4 : z^(n+1) - a * z^n + a * z - 1 = 0) : 
  Complex.abs z = 1 := by
sorry

end complex_root_magnitude_l4048_404817


namespace existence_of_k_values_l4048_404832

/-- Represents a triple of numbers -/
structure Triple :=
  (a b c : ℤ)

/-- Checks if the sums of powers with exponents 1, 2, and 3 are equal for two triples -/
def sumPowersEqual (t1 t2 : Triple) : Prop :=
  ∀ m : ℕ, m ≤ 3 → t1.a^m + t1.b^m + t1.c^m = t2.a^m + t2.b^m + t2.c^m

/-- Represents the 6-member group formed from two triples -/
def sixMemberGroup (t1 t2 : Triple) (k : ℤ) : Finset ℤ :=
  {t1.a, t1.b, t1.c, t2.a + k, t2.b + k, t2.c + k}

/-- Checks if a 6-member group can be simplified to a 4-member group -/
def simplifiesToFour (t1 t2 : Triple) (k : ℤ) : Prop :=
  (sixMemberGroup t1 t2 k).card = 4

/-- Checks if a 6-member group can be simplified to a 5-member group but not further -/
def simplifiesToFiveOnly (t1 t2 : Triple) (k : ℤ) : Prop :=
  (sixMemberGroup t1 t2 k).card = 5

/-- The main theorem to be proved -/
theorem existence_of_k_values 
  (I II III IV : Triple)
  (h1 : sumPowersEqual I II)
  (h2 : sumPowersEqual III IV) :
  ∃ k : ℤ, 
    (simplifiesToFour I II k ∨ simplifiesToFour II I k) ∧
    (simplifiesToFiveOnly III IV k ∨ simplifiesToFiveOnly IV III k) :=
sorry

end existence_of_k_values_l4048_404832


namespace line_through_A_and_B_line_through_C_and_D_line_through_E_and_K_line_through_M_and_P_l4048_404850

-- Define points
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (3, 5)
def C : ℝ × ℝ := (2, 2)
def D : ℝ × ℝ := (-1, -1)
def E : ℝ × ℝ := (0, 4)
def K : ℝ × ℝ := (2, 0)
def M : ℝ × ℝ := (3, 2)
def P : ℝ × ℝ := (6, 3)

-- Define line equations
def line1 (x : ℝ) : Prop := x = 3
def line2 (x y : ℝ) : Prop := y = x
def line3 (x y : ℝ) : Prop := y = -2 * x + 4
def line4 (x y : ℝ) : Prop := y = (1/3) * x + 1

-- Theorem statements
theorem line_through_A_and_B : 
  ∀ x y : ℝ, (x, y) = A ∨ (x, y) = B → line1 x := by sorry

theorem line_through_C_and_D : 
  ∀ x y : ℝ, (x, y) = C ∨ (x, y) = D → line2 x y := by sorry

theorem line_through_E_and_K : 
  ∀ x y : ℝ, (x, y) = E ∨ (x, y) = K → line3 x y := by sorry

theorem line_through_M_and_P : 
  ∀ x y : ℝ, (x, y) = M ∨ (x, y) = P → line4 x y := by sorry

end line_through_A_and_B_line_through_C_and_D_line_through_E_and_K_line_through_M_and_P_l4048_404850


namespace second_car_distance_rate_l4048_404870

/-- Represents the race scenario with two cars and a motorcycle --/
structure RaceScenario where
  l : ℝ  -- Length of the race distance
  v1 : ℝ  -- Speed of the first car
  v2 : ℝ  -- Speed of the second car
  vM : ℝ  -- Speed of the motorcycle

/-- Conditions of the race --/
def race_conditions (r : RaceScenario) : Prop :=
  r.l > 0 ∧  -- The race distance is positive
  r.v1 > 0 ∧ r.v2 > 0 ∧ r.vM > 0 ∧  -- All speeds are positive
  r.l / r.v2 - r.l / r.v1 = 1/60 ∧  -- Second car takes 1 minute longer than the first car
  r.v1 = 4 * r.vM ∧  -- First car is 4 times faster than the motorcycle
  r.v2 / 60 - r.vM / 60 = r.l / 6 ∧  -- Second car covers 1/6 more distance per minute than the motorcycle
  r.l / r.vM < 10  -- Motorcycle covers the distance in less than 10 minutes

/-- The theorem to be proved --/
theorem second_car_distance_rate (r : RaceScenario) :
  race_conditions r → r.v2 / 60 = 2/3 := by
  sorry

end second_car_distance_rate_l4048_404870


namespace polly_total_tweets_l4048_404880

/-- Represents an emotional state or activity of Polly the parakeet -/
structure State where
  name : String
  tweets_per_minute : ℕ
  duration : ℕ

/-- Calculates the total number of tweets for a given state -/
def tweets_for_state (s : State) : ℕ := s.tweets_per_minute * s.duration

/-- The list of Polly's states during the day -/
def polly_states : List State := [
  { name := "Happy", tweets_per_minute := 18, duration := 50 },
  { name := "Hungry", tweets_per_minute := 4, duration := 35 },
  { name := "Watching reflection", tweets_per_minute := 45, duration := 30 },
  { name := "Sad", tweets_per_minute := 6, duration := 20 },
  { name := "Playing with toys", tweets_per_minute := 25, duration := 75 }
]

/-- Calculates the total number of tweets for all states -/
def total_tweets (states : List State) : ℕ :=
  states.map tweets_for_state |>.sum

/-- Theorem: The total number of tweets Polly makes during the day is 4385 -/
theorem polly_total_tweets : total_tweets polly_states = 4385 := by
  sorry

end polly_total_tweets_l4048_404880


namespace regular_octahedron_parallel_edges_l4048_404878

structure RegularOctahedron where
  vertices : Finset (Fin 6)
  edges : Finset (Fin 6 × Fin 6)
  faces : Finset (Fin 3 → Fin 6)
  vertex_count : vertices.card = 6
  edge_count : edges.card = 12
  face_count : faces.card = 8

def parallel_edges (o : RegularOctahedron) : Finset (Fin 6 × Fin 6 × Fin 6 × Fin 6) :=
  sorry

theorem regular_octahedron_parallel_edges (o : RegularOctahedron) :
  (parallel_edges o).card = 6 := by
  sorry

end regular_octahedron_parallel_edges_l4048_404878


namespace symmetry_origin_symmetry_y_eq_x_vertex_2_neg_2_l4048_404884

-- Define the curve E
def E (x y : ℝ) : Prop := x^2 + x*y + y^2 = 4

-- Symmetry with respect to the origin
theorem symmetry_origin : ∀ x y : ℝ, E x y ↔ E (-x) (-y) := by sorry

-- Symmetry with respect to the line y = x
theorem symmetry_y_eq_x : ∀ x y : ℝ, E x y ↔ E y x := by sorry

-- (2, -2) is a vertex of E
theorem vertex_2_neg_2 : E 2 (-2) ∧ (∃ ε > 0, ∀ x y : ℝ, 
  (x - 2)^2 + (y + 2)^2 < ε^2 → E x y → x^2 + y^2 ≥ 2^2 + (-2)^2) := by sorry

end symmetry_origin_symmetry_y_eq_x_vertex_2_neg_2_l4048_404884


namespace robin_cupcakes_l4048_404828

/-- Calculates the total number of cupcakes Robin has after baking and selling. -/
def total_cupcakes (initial : ℕ) (sold : ℕ) (additional : ℕ) : ℕ :=
  initial - sold + additional

/-- Theorem stating that Robin has 59 cupcakes in total. -/
theorem robin_cupcakes : total_cupcakes 42 22 39 = 59 := by
  sorry

end robin_cupcakes_l4048_404828


namespace complex_roots_problem_l4048_404871

theorem complex_roots_problem (p q r : ℂ) : 
  p + q + r = 2 ∧ 
  p * q * r = 2 ∧ 
  p * q + p * r + q * r = 0 → 
  (p = 2 ∧ q = Complex.I * Real.sqrt 2 ∧ r = -Complex.I * Real.sqrt 2) ∨
  (p = 2 ∧ q = -Complex.I * Real.sqrt 2 ∧ r = Complex.I * Real.sqrt 2) ∨
  (p = Complex.I * Real.sqrt 2 ∧ q = 2 ∧ r = -Complex.I * Real.sqrt 2) ∨
  (p = Complex.I * Real.sqrt 2 ∧ q = -Complex.I * Real.sqrt 2 ∧ r = 2) ∨
  (p = -Complex.I * Real.sqrt 2 ∧ q = 2 ∧ r = Complex.I * Real.sqrt 2) ∨
  (p = -Complex.I * Real.sqrt 2 ∧ q = Complex.I * Real.sqrt 2 ∧ r = 2) :=
by sorry

end complex_roots_problem_l4048_404871


namespace coffee_price_coffee_price_is_12_l4048_404857

/-- The regular price for a half-pound of coffee, given a 25% discount and 
    quarter-pound bags sold for $4.50 after the discount. -/
theorem coffee_price : ℝ :=
  let discount_rate : ℝ := 0.25
  let discounted_price_quarter_pound : ℝ := 4.50
  let regular_price_half_pound : ℝ := 12

  have h1 : discounted_price_quarter_pound = 
    regular_price_half_pound / 2 * (1 - discount_rate) := by sorry

  regular_price_half_pound

/-- Proof that the regular price for a half-pound of coffee is $12 -/
theorem coffee_price_is_12 : coffee_price = 12 := by sorry

end coffee_price_coffee_price_is_12_l4048_404857


namespace floor_of_negative_two_point_seven_l4048_404866

theorem floor_of_negative_two_point_seven : ⌊(-2.7 : ℝ)⌋ = -3 := by
  sorry

end floor_of_negative_two_point_seven_l4048_404866


namespace max_value_polynomial_l4048_404885

theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  (∃ (max : ℝ), ∀ (a b : ℝ), a + b = 5 → 
    a^4*b + a^3*b + a^2*b + a*b + a*b^2 + a*b^3 + a*b^4 ≤ max) ∧
  (x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 ≤ 6084/17) ∧
  (∃ (x₀ y₀ : ℝ), x₀ + y₀ = 5 ∧ 
    x₀^4*y₀ + x₀^3*y₀ + x₀^2*y₀ + x₀*y₀ + x₀*y₀^2 + x₀*y₀^3 + x₀*y₀^4 = 6084/17) :=
by sorry

end max_value_polynomial_l4048_404885


namespace machine_value_theorem_l4048_404874

/-- Calculates the machine's value after two years and a major overhaul -/
def machine_value_after_two_years_and_overhaul (initial_value : ℝ) : ℝ :=
  let year1_depreciation_rate := 0.10
  let year2_depreciation_rate := 0.12
  let repair_rate := 0.03
  let overhaul_rate := 0.15
  
  let value_after_year1 := initial_value * (1 - year1_depreciation_rate) * (1 + repair_rate)
  let value_after_year2 := value_after_year1 * (1 - year2_depreciation_rate) * (1 + repair_rate)
  let final_value := value_after_year2 * (1 - overhaul_rate)
  
  final_value

/-- Theorem stating that the machine's value after two years and a major overhaul 
    is approximately $863.23, given an initial value of $1200 -/
theorem machine_value_theorem :
  ∃ ε > 0, abs (machine_value_after_two_years_and_overhaul 1200 - 863.23) < ε :=
by
  sorry

end machine_value_theorem_l4048_404874


namespace tangent_and_cosine_identities_l4048_404819

theorem tangent_and_cosine_identities 
  (α β : Real) 
  (h1 : 0 < α ∧ α < π) 
  (h2 : 0 < β ∧ β < π) 
  (h3 : (Real.tan α)^2 - 5*(Real.tan α) + 6 = 0) 
  (h4 : (Real.tan β)^2 - 5*(Real.tan β) + 6 = 0) : 
  Real.tan (α + β) = -1 ∧ Real.cos (α - β) = 7*Real.sqrt 2/10 := by
  sorry

end tangent_and_cosine_identities_l4048_404819
