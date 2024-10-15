import Mathlib

namespace NUMINAMATH_CALUDE_transformation_of_curve_l3138_313812

-- Define the transformation φ
def φ (p : ℝ × ℝ) : ℝ × ℝ := (3 * p.1, 4 * p.2)

-- Define the initial curve
def initial_curve (p : ℝ × ℝ) : Prop := p.1^2 + p.2^2 = 1

-- Define the final curve
def final_curve (p : ℝ × ℝ) : Prop := p.1^2 / 9 + p.2^2 / 16 = 1

-- Theorem statement
theorem transformation_of_curve :
  ∀ p : ℝ × ℝ, initial_curve p ↔ final_curve (φ p) := by sorry

end NUMINAMATH_CALUDE_transformation_of_curve_l3138_313812


namespace NUMINAMATH_CALUDE_P_in_quadrant_III_l3138_313882

-- Define the point P
def P : ℝ × ℝ := (-3, -4)

-- Define the quadrants
def in_quadrant_I (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0
def in_quadrant_II (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 > 0
def in_quadrant_III (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0
def in_quadrant_IV (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

-- Theorem: P lies in Quadrant III
theorem P_in_quadrant_III : in_quadrant_III P := by sorry

end NUMINAMATH_CALUDE_P_in_quadrant_III_l3138_313882


namespace NUMINAMATH_CALUDE_unequal_gender_probability_l3138_313831

/-- The number of grandchildren Mrs. Lee has -/
def n : ℕ := 12

/-- The probability of having a grandson (or granddaughter) -/
def p : ℚ := 1/2

/-- The probability of having an unequal number of grandsons and granddaughters -/
def unequal_probability : ℚ := 793/1024

theorem unequal_gender_probability :
  (1 - (n.choose (n/2)) * p^n) = unequal_probability :=
sorry

end NUMINAMATH_CALUDE_unequal_gender_probability_l3138_313831


namespace NUMINAMATH_CALUDE_choose_two_from_three_l3138_313878

theorem choose_two_from_three (n : ℕ) (h : n = 3) :
  Nat.choose n 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_choose_two_from_three_l3138_313878


namespace NUMINAMATH_CALUDE_class_average_score_class_average_is_85_l3138_313891

theorem class_average_score : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ
  | total_students, students_score_92, students_score_80, students_score_70, score_70 =>
    let total_score := students_score_92 * 92 + students_score_80 * 80 + students_score_70 * score_70
    total_score / total_students

theorem class_average_is_85 :
  class_average_score 10 5 4 1 70 = 85 := by
  sorry

end NUMINAMATH_CALUDE_class_average_score_class_average_is_85_l3138_313891


namespace NUMINAMATH_CALUDE_h_zero_iff_b_eq_seven_fifths_l3138_313843

/-- The function h(x) = 5x - 7 -/
def h (x : ℝ) : ℝ := 5 * x - 7

/-- Theorem stating that h(b) = 0 if and only if b = 7/5 -/
theorem h_zero_iff_b_eq_seven_fifths :
  ∀ b : ℝ, h b = 0 ↔ b = 7 / 5 := by sorry

end NUMINAMATH_CALUDE_h_zero_iff_b_eq_seven_fifths_l3138_313843


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l3138_313821

theorem smallest_three_digit_multiple_of_17 : ∀ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) → (n % 17 = 0) → n ≥ 102 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l3138_313821


namespace NUMINAMATH_CALUDE_alpha_plus_beta_equals_112_l3138_313855

theorem alpha_plus_beta_equals_112 
  (α β : ℝ) 
  (h : ∀ x : ℝ, (x - α) / (x + β) = (x^2 - 96*x + 2210) / (x^2 + 65*x - 3510)) : 
  α + β = 112 := by
sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_equals_112_l3138_313855


namespace NUMINAMATH_CALUDE_curve_in_second_quadrant_l3138_313832

-- Define the curve C
def C (a x y : ℝ) : Prop := x^2 + y^2 + 2*a*x - 4*a*y + 5*a^2 - 4 = 0

-- Define the second quadrant
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Theorem statement
theorem curve_in_second_quadrant :
  ∀ a : ℝ, (∀ x y : ℝ, C a x y → second_quadrant x y) ↔ a > 2 :=
sorry

end NUMINAMATH_CALUDE_curve_in_second_quadrant_l3138_313832


namespace NUMINAMATH_CALUDE_f_min_implies_a_range_l3138_313837

/-- A function f with a real parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := |3*x - 1| + a*x + 2

/-- The theorem stating that if f has a minimum value, then a is in [-3, 3] -/
theorem f_min_implies_a_range (a : ℝ) :
  (∃ (m : ℝ), ∀ (x : ℝ), f a x ≥ m) → a ∈ Set.Icc (-3) 3 := by
  sorry

end NUMINAMATH_CALUDE_f_min_implies_a_range_l3138_313837


namespace NUMINAMATH_CALUDE_largest_five_digit_congruent_to_17_mod_34_l3138_313848

theorem largest_five_digit_congruent_to_17_mod_34 :
  ∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 34 = 17 → n ≤ 99994 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_five_digit_congruent_to_17_mod_34_l3138_313848


namespace NUMINAMATH_CALUDE_perfect_fit_implies_r_squared_one_l3138_313880

/-- Represents a sample point in a scatter plot -/
structure SamplePoint where
  x : ℝ
  y : ℝ

/-- Represents a linear regression model -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Checks if a point lies on a given line -/
def pointOnLine (p : SamplePoint) (model : LinearRegression) : Prop :=
  p.y = model.slope * p.x + model.intercept

/-- The coefficient of determination (R²) for a regression model -/
def R_squared (data : List SamplePoint) (model : LinearRegression) : ℝ :=
  sorry -- Definition of R² calculation

theorem perfect_fit_implies_r_squared_one
  (data : List SamplePoint)
  (model : LinearRegression)
  (h_non_zero_slope : model.slope ≠ 0)
  (h_all_points_on_line : ∀ p ∈ data, pointOnLine p model) :
  R_squared data model = 1 :=
sorry

end NUMINAMATH_CALUDE_perfect_fit_implies_r_squared_one_l3138_313880


namespace NUMINAMATH_CALUDE_diet_soda_count_l3138_313841

/-- Given a grocery store inventory, calculate the number of diet soda bottles. -/
theorem diet_soda_count (regular_soda : ℕ) (difference : ℕ) : 
  regular_soda = 79 → difference = 26 → regular_soda - difference = 53 := by
  sorry

#check diet_soda_count

end NUMINAMATH_CALUDE_diet_soda_count_l3138_313841


namespace NUMINAMATH_CALUDE_division_theorem_l3138_313857

theorem division_theorem (dividend divisor remainder quotient : ℕ) : 
  dividend = divisor * quotient + remainder →
  dividend = 167 →
  divisor = 18 →
  remainder = 5 →
  quotient = 9 := by
sorry

end NUMINAMATH_CALUDE_division_theorem_l3138_313857


namespace NUMINAMATH_CALUDE_seed_germination_percentage_l3138_313872

theorem seed_germination_percentage (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate1 germination_rate2 : ℚ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate1 = 30 / 100 →
  germination_rate2 = 35 / 100 →
  (((seeds_plot1 : ℚ) * germination_rate1 + (seeds_plot2 : ℚ) * germination_rate2) / 
   ((seeds_plot1 : ℚ) + (seeds_plot2 : ℚ))) * 100 = 32 := by
  sorry

end NUMINAMATH_CALUDE_seed_germination_percentage_l3138_313872


namespace NUMINAMATH_CALUDE_negative_result_operations_l3138_313827

theorem negative_result_operations : 
  (-(-4) > 0) ∧ 
  (abs (-4) > 0) ∧ 
  (-4^2 < 0) ∧ 
  ((-4)^2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negative_result_operations_l3138_313827


namespace NUMINAMATH_CALUDE_converse_of_negative_square_positive_l3138_313844

theorem converse_of_negative_square_positive :
  (∀ x : ℝ, x < 0 → x^2 > 0) →
  (∀ x : ℝ, x^2 > 0 → x < 0) :=
sorry

end NUMINAMATH_CALUDE_converse_of_negative_square_positive_l3138_313844


namespace NUMINAMATH_CALUDE_custom_mult_equation_solution_l3138_313887

-- Define the custom operation
def customMult (a b : ℝ) : ℝ := 4 * a * b

-- State the theorem
theorem custom_mult_equation_solution :
  ∀ x : ℝ, (customMult x x) + (customMult 2 x) - (customMult 2 4) = 0 → x = 2 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_equation_solution_l3138_313887


namespace NUMINAMATH_CALUDE_charity_box_distribution_l3138_313836

/-- The charity organization's box distribution problem -/
theorem charity_box_distribution
  (box_cost : ℕ)
  (donation_multiplier : ℕ)
  (total_boxes : ℕ)
  (h1 : box_cost = 245)
  (h2 : donation_multiplier = 4)
  (h3 : total_boxes = 2000) :
  ∃ (initial_boxes : ℕ),
    initial_boxes * box_cost * (1 + donation_multiplier) = total_boxes * box_cost ∧
    initial_boxes = 400 := by
  sorry

end NUMINAMATH_CALUDE_charity_box_distribution_l3138_313836


namespace NUMINAMATH_CALUDE_probability_of_selection_X_l3138_313885

theorem probability_of_selection_X 
  (prob_Y : ℝ) 
  (prob_X_and_Y : ℝ) 
  (h1 : prob_Y = 2/5) 
  (h2 : prob_X_and_Y = 0.05714285714285714) : 
  ∃ (prob_X : ℝ), prob_X = 0.14285714285714285 ∧ prob_X_and_Y = prob_X * prob_Y :=
by
  sorry

end NUMINAMATH_CALUDE_probability_of_selection_X_l3138_313885


namespace NUMINAMATH_CALUDE_carltons_outfits_l3138_313810

/-- Represents a person's wardrobe and outfit combinations -/
structure Wardrobe where
  buttonUpShirts : ℕ
  sweaterVests : ℕ
  outfits : ℕ

/-- Calculates the number of outfits for a given wardrobe -/
def calculateOutfits (w : Wardrobe) : Prop :=
  w.sweaterVests = 2 * w.buttonUpShirts ∧
  w.outfits = w.sweaterVests * w.buttonUpShirts

/-- Theorem: Carlton's wardrobe has 18 outfits -/
theorem carltons_outfits :
  ∃ (w : Wardrobe), w.buttonUpShirts = 3 ∧ calculateOutfits w ∧ w.outfits = 18 := by
  sorry


end NUMINAMATH_CALUDE_carltons_outfits_l3138_313810


namespace NUMINAMATH_CALUDE_candy_bar_cost_l3138_313890

/-- Proves that the cost of each candy bar is $2, given the total spent and number of candy bars. -/
theorem candy_bar_cost (total_spent : ℚ) (num_candy_bars : ℕ) (h1 : total_spent = 4) (h2 : num_candy_bars = 2) :
  total_spent / num_candy_bars = 2 := by
  sorry

#check candy_bar_cost

end NUMINAMATH_CALUDE_candy_bar_cost_l3138_313890


namespace NUMINAMATH_CALUDE_regular_hourly_wage_l3138_313823

theorem regular_hourly_wage (
  working_days_per_week : ℕ)
  (working_hours_per_day : ℕ)
  (overtime_rate : ℚ)
  (total_earnings : ℚ)
  (total_hours_worked : ℕ)
  (weeks : ℕ)
  (h1 : working_days_per_week = 6)
  (h2 : working_hours_per_day = 10)
  (h3 : overtime_rate = 21/5)
  (h4 : total_earnings = 525)
  (h5 : total_hours_worked = 245)
  (h6 : weeks = 4) :
  let regular_hours := working_days_per_week * working_hours_per_day * weeks
  let overtime_hours := total_hours_worked - regular_hours
  let overtime_pay := overtime_rate * overtime_hours
  let regular_pay := total_earnings - overtime_pay
  let regular_hourly_wage := regular_pay / regular_hours
  regular_hourly_wage = 21/10 := by
sorry

end NUMINAMATH_CALUDE_regular_hourly_wage_l3138_313823


namespace NUMINAMATH_CALUDE_largest_number_of_circles_l3138_313883

/-- Given a convex quadrilateral BCDE in the plane where lines EB and DC intersect at A,
    this theorem proves that the largest number of nonoverlapping circles that can lie in
    BCDE and are tangent to both BE and CD is 5, given the specified conditions. -/
theorem largest_number_of_circles
  (AB : ℝ) (AC : ℝ) (AD : ℝ) (AE : ℝ) (cos_BAC : ℝ)
  (h_AB : AB = 2)
  (h_AC : AC = 5)
  (h_AD : AD = 200)
  (h_AE : AE = 500)
  (h_cos_BAC : cos_BAC = 7/9)
  : ℕ :=
5

#check largest_number_of_circles

end NUMINAMATH_CALUDE_largest_number_of_circles_l3138_313883


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3138_313889

theorem solve_exponential_equation :
  ∃ n : ℕ, 16^n * 16^n * 16^n * 16^n = 256^4 ∧ n = 2 := by
sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3138_313889


namespace NUMINAMATH_CALUDE_average_visitors_theorem_l3138_313867

/-- Calculates the average number of visitors per day in a 30-day month starting on Sunday -/
def averageVisitorsPerDay (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let numSundays := 4
  let numOtherDays := 30 - numSundays
  let totalVisitors := numSundays * sundayVisitors + numOtherDays * otherDayVisitors
  totalVisitors / 30

/-- Proves that the average number of visitors per day is 188 given the specified conditions -/
theorem average_visitors_theorem :
  averageVisitorsPerDay 500 140 = 188 := by
  sorry

end NUMINAMATH_CALUDE_average_visitors_theorem_l3138_313867


namespace NUMINAMATH_CALUDE_sin_equality_proof_l3138_313860

theorem sin_equality_proof (n : ℤ) : 
  -90 ≤ n ∧ n ≤ 90 ∧ Real.sin (n * π / 180) = Real.sin (782 * π / 180) → 
  n = 62 ∨ n = -62 := by
  sorry

end NUMINAMATH_CALUDE_sin_equality_proof_l3138_313860


namespace NUMINAMATH_CALUDE_cookie_jar_spending_l3138_313845

theorem cookie_jar_spending (initial_amount : ℝ) (final_amount : ℝ) (doris_spent : ℝ) :
  initial_amount = 24 →
  final_amount = 15 →
  initial_amount - (doris_spent + doris_spent / 2) = final_amount →
  doris_spent = 6 := by
sorry

end NUMINAMATH_CALUDE_cookie_jar_spending_l3138_313845


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l3138_313899

theorem no_positive_integer_solutions :
  ¬ ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ 21 * x * y = 7 - 3 * x - 4 * y := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l3138_313899


namespace NUMINAMATH_CALUDE_f_composition_negative_one_l3138_313884

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem f_composition_negative_one : f (f (-1)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_one_l3138_313884


namespace NUMINAMATH_CALUDE_range_of_fraction_l3138_313871

theorem range_of_fraction (a b : ℝ) (ha : 1 < a ∧ a < 4) (hb : 2 < b ∧ b < 8) :
  1/8 < a/b ∧ a/b < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_fraction_l3138_313871


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l3138_313834

theorem ratio_x_to_y (x y : ℝ) (h : 0.1 * x = 0.2 * y) : x / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l3138_313834


namespace NUMINAMATH_CALUDE_sugar_substitute_usage_l3138_313824

/-- Proves that Christopher uses 1 packet of sugar substitute per coffee --/
theorem sugar_substitute_usage
  (coffees_per_day : ℕ)
  (packets_per_box : ℕ)
  (cost_per_box : ℚ)
  (total_cost : ℚ)
  (total_days : ℕ)
  (h1 : coffees_per_day = 2)
  (h2 : packets_per_box = 30)
  (h3 : cost_per_box = 4)
  (h4 : total_cost = 24)
  (h5 : total_days = 90) :
  (total_cost / cost_per_box * packets_per_box) / (total_days * coffees_per_day) = 1 := by
  sorry


end NUMINAMATH_CALUDE_sugar_substitute_usage_l3138_313824


namespace NUMINAMATH_CALUDE_workshop_percentage_l3138_313817

-- Define the workday duration in minutes
def workday_minutes : ℕ := 8 * 60

-- Define the duration of the first workshop in minutes
def first_workshop_minutes : ℕ := 60

-- Define the duration of the second workshop in minutes
def second_workshop_minutes : ℕ := 2 * first_workshop_minutes

-- Define the total time spent in workshops
def total_workshop_minutes : ℕ := first_workshop_minutes + second_workshop_minutes

-- Theorem statement
theorem workshop_percentage :
  (total_workshop_minutes : ℚ) / (workday_minutes : ℚ) * 100 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_workshop_percentage_l3138_313817


namespace NUMINAMATH_CALUDE_ellipse_equation_l3138_313876

/-- Given an ellipse C with equation x²/a² + y²/b² = 1 where a > b > 0,
    foci at (-2, 0) and (2, 0), and the product of slopes of lines from
    the left vertex to the intersection points of the ellipse with the
    circle having diameter F₁F₂ being 1/3, prove that the standard
    equation of the ellipse C is x²/6 + y²/2 = 1. -/
theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (∃ (x y : ℝ → ℝ), ∀ t, (x t)^2 / a^2 + (y t)^2 / b^2 = 1) →
  (∃ (F₁ F₂ : ℝ × ℝ), F₁ = (-2, 0) ∧ F₂ = (2, 0)) →
  (∃ (M N : ℝ × ℝ), M.1 > 0 ∧ M.2 > 0 ∧ N.1 < 0 ∧ N.2 > 0) →
  (∃ (A : ℝ × ℝ), A = (-a, 0)) →
  (∃ (m₁ m₂ : ℝ), m₁ * m₂ = 1/3) →
  (x^2 / 6 + y^2 / 2 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3138_313876


namespace NUMINAMATH_CALUDE_louis_suit_cost_is_141_l3138_313892

/-- The cost of Louis's velvet suit materials -/
def louis_suit_cost (fabric_price_per_yard : ℝ) (pattern_price : ℝ) (thread_price_per_spool : ℝ) 
  (fabric_yards : ℝ) (thread_spools : ℕ) : ℝ :=
  fabric_price_per_yard * fabric_yards + pattern_price + thread_price_per_spool * thread_spools

/-- Theorem: The total cost of Louis's suit materials is $141 -/
theorem louis_suit_cost_is_141 : 
  louis_suit_cost 24 15 3 5 2 = 141 := by
  sorry

end NUMINAMATH_CALUDE_louis_suit_cost_is_141_l3138_313892


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_segments_l3138_313858

/-- Given a right triangle with legs in ratio 3:7 and altitude to hypotenuse of 42,
    prove that the altitude divides the hypotenuse into segments of length 18 and 98 -/
theorem right_triangle_hypotenuse_segments
  (a b c h : ℝ)
  (right_angle : a^2 + b^2 = c^2)
  (leg_ratio : b = 7/3 * a)
  (altitude : h = 42)
  (geo_mean : a * b = h^2) :
  ∃ (x y : ℝ), x + y = c ∧ x * y = h^2 ∧ x = 18 ∧ y = 98 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_segments_l3138_313858


namespace NUMINAMATH_CALUDE_march_book_sales_l3138_313896

theorem march_book_sales (january_sales february_sales : ℕ) 
  (h1 : january_sales = 15)
  (h2 : february_sales = 16)
  (h3 : (january_sales + february_sales + march_sales) / 3 = 16) :
  march_sales = 17 := by
  sorry

end NUMINAMATH_CALUDE_march_book_sales_l3138_313896


namespace NUMINAMATH_CALUDE_meeting_probability_in_our_tournament_l3138_313854

/-- Represents a knockout tournament --/
structure KnockoutTournament where
  total_players : Nat
  num_rounds : Nat
  random_pairing : Bool
  equal_win_chance : Bool

/-- The probability of two specific players meeting in a tournament --/
def meeting_probability (t : KnockoutTournament) : Rat :=
  sorry

/-- Our specific tournament --/
def our_tournament : KnockoutTournament :=
  { total_players := 32
  , num_rounds := 5
  , random_pairing := true
  , equal_win_chance := true }

theorem meeting_probability_in_our_tournament :
  meeting_probability our_tournament = 11097 / 167040 := by
  sorry

end NUMINAMATH_CALUDE_meeting_probability_in_our_tournament_l3138_313854


namespace NUMINAMATH_CALUDE_linear_function_decreasing_iff_k_lt_neg_two_l3138_313853

/-- A linear function y = mx + b where m = k + 2 and b = -1 -/
def linear_function (k : ℝ) (x : ℝ) : ℝ := (k + 2) * x - 1

/-- The property that y decreases as x increases -/
def decreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

theorem linear_function_decreasing_iff_k_lt_neg_two (k : ℝ) :
  decreasing_function (linear_function k) ↔ k < -2 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_decreasing_iff_k_lt_neg_two_l3138_313853


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3138_313869

theorem repeating_decimal_sum : 
  (1 : ℚ) / 9 + (2 : ℚ) / 99 + (2 : ℚ) / 333 = (503 : ℚ) / 3663 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3138_313869


namespace NUMINAMATH_CALUDE_x_squared_plus_nine_y_squared_l3138_313829

theorem x_squared_plus_nine_y_squared (x y : ℝ) 
  (h1 : x + 3*y = 6) (h2 : x*y = -9) : x^2 + 9*y^2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_nine_y_squared_l3138_313829


namespace NUMINAMATH_CALUDE_smallest_digit_change_l3138_313849

def original_sum : ℕ := 753 + 946 + 821
def incorrect_result : ℕ := 2420
def correct_result : ℕ := 2520

def change_digit (n : ℕ) (place : ℕ) (new_digit : ℕ) : ℕ := 
  n - (n / 10^place % 10) * 10^place + new_digit * 10^place

theorem smallest_digit_change :
  ∃ (d : ℕ), d < 10 ∧ 
    change_digit 821 2 (d + 1) + 753 + 946 = correct_result ∧
    ∀ (n : ℕ) (p : ℕ) (digit : ℕ), 
      digit < d → 
      change_digit 753 p digit + 946 + 821 ≠ correct_result ∧
      753 + change_digit 946 p digit + 821 ≠ correct_result ∧
      753 + 946 + change_digit 821 p digit ≠ correct_result :=
sorry

#check smallest_digit_change

end NUMINAMATH_CALUDE_smallest_digit_change_l3138_313849


namespace NUMINAMATH_CALUDE_min_discount_rate_l3138_313852

/-- The minimum discount rate for a product with given cost and marked prices, ensuring a minimum profit percentage. -/
theorem min_discount_rate (cost : ℝ) (marked : ℝ) (min_profit_percent : ℝ) :
  cost = 1000 →
  marked = 1500 →
  min_profit_percent = 5 →
  ∃ x : ℝ, x = 0.7 ∧
    ∀ y : ℝ, (marked * y - cost ≥ cost * (min_profit_percent / 100) → y ≥ x) :=
by sorry

end NUMINAMATH_CALUDE_min_discount_rate_l3138_313852


namespace NUMINAMATH_CALUDE_basketball_free_throws_l3138_313826

/-- Proves that DeShawn made 12 free-throws given the conditions of the basketball practice problem. -/
theorem basketball_free_throws 
  (deshawn : ℕ) -- DeShawn's free-throws
  (kayla : ℕ) -- Kayla's free-throws
  (annieka : ℕ) -- Annieka's free-throws
  (h1 : kayla = deshawn + deshawn / 2) -- Kayla made 50% more than DeShawn
  (h2 : annieka = kayla - 4) -- Annieka made 4 fewer than Kayla
  (h3 : annieka = 14) -- Annieka made 14 free-throws
  : deshawn = 12 := by
  sorry

end NUMINAMATH_CALUDE_basketball_free_throws_l3138_313826


namespace NUMINAMATH_CALUDE_magnified_diameter_is_five_l3138_313842

/-- The magnification factor of an electron microscope. -/
def magnification : ℝ := 1000

/-- The actual diameter of the tissue in centimeters. -/
def actual_diameter : ℝ := 0.005

/-- The diameter of the magnified image in centimeters. -/
def magnified_diameter : ℝ := actual_diameter * magnification

/-- Theorem stating that the diameter of the magnified image is 5 centimeters. -/
theorem magnified_diameter_is_five :
  magnified_diameter = 5 := by sorry

end NUMINAMATH_CALUDE_magnified_diameter_is_five_l3138_313842


namespace NUMINAMATH_CALUDE_percentage_increase_is_30_percent_l3138_313895

-- Define the initial weight James can lift for 20 meters
def initial_weight : ℝ := 300

-- Define the weight increase for 20 meters
def weight_increase : ℝ := 50

-- Define the weight with straps for 10 meters
def weight_with_straps : ℝ := 546

-- Define the strap increase percentage
def strap_increase : ℝ := 0.20

-- Define the function to calculate the weight for 10 meters with a given percentage increase
def weight_for_10m (p : ℝ) : ℝ := (initial_weight + weight_increase) * (1 + p)

-- Define the function to calculate the weight for 10 meters with straps
def weight_for_10m_with_straps (p : ℝ) : ℝ := weight_for_10m p * (1 + strap_increase)

-- Theorem to prove
theorem percentage_increase_is_30_percent :
  ∃ p : ℝ, p = 0.3 ∧ weight_for_10m_with_straps p = weight_with_straps :=
sorry

end NUMINAMATH_CALUDE_percentage_increase_is_30_percent_l3138_313895


namespace NUMINAMATH_CALUDE_fescue_percentage_in_y_l3138_313811

/-- Represents the composition of a seed mixture -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The final mixture of X and Y -/
def final_mixture (x y : SeedMixture) (x_proportion : ℝ) : SeedMixture :=
  { ryegrass := x_proportion * x.ryegrass + (1 - x_proportion) * y.ryegrass,
    bluegrass := x_proportion * x.bluegrass + (1 - x_proportion) * y.bluegrass,
    fescue := x_proportion * x.fescue + (1 - x_proportion) * y.fescue }

theorem fescue_percentage_in_y
  (x : SeedMixture)
  (y : SeedMixture)
  (h_x_ryegrass : x.ryegrass = 0.4)
  (h_x_bluegrass : x.bluegrass = 0.6)
  (h_x_fescue : x.fescue = 0)
  (h_y_ryegrass : y.ryegrass = 0.25)
  (h_final_ryegrass : (final_mixture x y 0.6667).ryegrass = 0.35)
  : y.fescue = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_fescue_percentage_in_y_l3138_313811


namespace NUMINAMATH_CALUDE_vectors_perpendicular_l3138_313818

theorem vectors_perpendicular : ∀ (a b : ℝ × ℝ), 
  a = (2, -3) → b = (3, 2) → a.1 * b.1 + a.2 * b.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_vectors_perpendicular_l3138_313818


namespace NUMINAMATH_CALUDE_cylinder_height_l3138_313807

/-- Proves that a cylinder with a circular base perimeter of 6 feet and a side surface
    formed by a rectangular plate with a diagonal of 10 feet has a height of 8 feet. -/
theorem cylinder_height (base_perimeter : ℝ) (diagonal : ℝ) (height : ℝ) : 
  base_perimeter = 6 → diagonal = 10 → height = 8 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_l3138_313807


namespace NUMINAMATH_CALUDE_digit_sum_theorem_l3138_313881

/-- Concatenate two digits to form a two-digit number -/
def concatenate (a b : Nat) : Nat := 10 * a + b

/-- Check if a number is prime -/
def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

theorem digit_sum_theorem (p q r : Nat) : 
  p < 10 → q < 10 → r < 10 →
  p ≠ q → p ≠ r → q ≠ r →
  isPrime (concatenate p q) →
  isPrime (concatenate p r) →
  isPrime (concatenate q r) →
  concatenate p q ≠ concatenate p r →
  concatenate p q ≠ concatenate q r →
  concatenate p r ≠ concatenate q r →
  (concatenate p q) * (concatenate p r) = 221 →
  p + q + r = 11 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_theorem_l3138_313881


namespace NUMINAMATH_CALUDE_at_least_one_quadratic_has_solution_l3138_313897

theorem at_least_one_quadratic_has_solution (a b c : ℝ) : 
  ∃ x : ℝ, (x^2 + (a - b)*x + (b - c) = 0) ∨ 
            (x^2 + (b - c)*x + (c - a) = 0) ∨ 
            (x^2 + (c - a)*x + (a - b) = 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_quadratic_has_solution_l3138_313897


namespace NUMINAMATH_CALUDE_fraction_simplification_and_evaluation_l3138_313808

theorem fraction_simplification_and_evaluation (x : ℝ) (h : x ≠ 2) :
  (x^6 - 16*x^3 + 64) / (x^3 - 8) = x^3 - 8 ∧ 
  (6^6 - 16*6^3 + 64) / (6^3 - 8) = 208 :=
sorry

end NUMINAMATH_CALUDE_fraction_simplification_and_evaluation_l3138_313808


namespace NUMINAMATH_CALUDE_power_function_increasing_m_l3138_313851

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ x, f x = a * x^b

-- Define an increasing function on (0, +∞)
def isIncreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y → f x < f y

-- The main theorem
theorem power_function_increasing_m (m : ℝ) :
  let f := fun x : ℝ => (m^2 - m - 1) * x^m
  isPowerFunction f ∧ isIncreasingOn f → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_increasing_m_l3138_313851


namespace NUMINAMATH_CALUDE_rogers_app_ratio_l3138_313859

/-- Proof that Roger's app ratio is 2 given the problem conditions -/
theorem rogers_app_ratio : 
  let max_apps : ℕ := 50
  let recommended_apps : ℕ := 35
  let delete_apps : ℕ := 20
  let rogers_apps : ℕ := max_apps + delete_apps
  rogers_apps / recommended_apps = 2 := by sorry

end NUMINAMATH_CALUDE_rogers_app_ratio_l3138_313859


namespace NUMINAMATH_CALUDE_total_days_is_210_l3138_313863

/-- Calculates the total number of days spent on two islands given the durations of expeditions. -/
def total_days_on_islands (island_a_first : ℕ) (island_b_first : ℕ) : ℕ :=
  let island_a_second := island_a_first + 2
  let island_a_third := island_a_second * 2
  let island_b_second := island_b_first - 3
  let island_b_third := island_b_first
  let total_weeks := (island_a_first + island_a_second + island_a_third) +
                     (island_b_first + island_b_second + island_b_third)
  total_weeks * 7

/-- Theorem stating that the total number of days spent on both islands is 210. -/
theorem total_days_is_210 : total_days_on_islands 3 5 = 210 := by
  sorry

end NUMINAMATH_CALUDE_total_days_is_210_l3138_313863


namespace NUMINAMATH_CALUDE_sum_of_x_values_l3138_313870

theorem sum_of_x_values (x : ℝ) : 
  (50 < x ∧ x < 150) →
  (Real.cos (2 * x * π / 180))^3 + (Real.cos (6 * x * π / 180))^3 = 
    8 * (Real.cos (4 * x * π / 180))^3 * (Real.cos (x * π / 180))^3 →
  ∃ (s : Finset ℝ), (∀ y ∈ s, 
    (50 < y ∧ y < 150) ∧
    (Real.cos (2 * y * π / 180))^3 + (Real.cos (6 * y * π / 180))^3 = 
      8 * (Real.cos (4 * y * π / 180))^3 * (Real.cos (y * π / 180))^3) ∧
  (s.sum id = 270) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_values_l3138_313870


namespace NUMINAMATH_CALUDE_prob_aces_or_kings_correct_l3138_313819

/-- The number of cards in the deck -/
def deck_size : ℕ := 52

/-- The number of aces in the deck -/
def num_aces : ℕ := 5

/-- The number of kings in the deck -/
def num_kings : ℕ := 4

/-- The probability of drawing either two aces or at least one king -/
def prob_aces_or_kings : ℚ := 104 / 663

theorem prob_aces_or_kings_correct :
  let prob_two_aces := (num_aces * (num_aces - 1)) / (deck_size * (deck_size - 1))
  let prob_one_king := 2 * (num_kings * (deck_size - num_kings)) / (deck_size * (deck_size - 1))
  let prob_two_kings := (num_kings * (num_kings - 1)) / (deck_size * (deck_size - 1))
  prob_two_aces + prob_one_king + prob_two_kings = prob_aces_or_kings := by
  sorry

end NUMINAMATH_CALUDE_prob_aces_or_kings_correct_l3138_313819


namespace NUMINAMATH_CALUDE_inequality_proof_l3138_313825

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  2 * Real.sqrt (b * c + c * a + a * b) ≤ Real.sqrt 3 * (((b + c) * (c + a) * (a + b)) ^ (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3138_313825


namespace NUMINAMATH_CALUDE_multiplication_equation_solution_l3138_313840

theorem multiplication_equation_solution : ∃ x : ℚ, 9 * x = 36 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_equation_solution_l3138_313840


namespace NUMINAMATH_CALUDE_odd_multiple_of_three_l3138_313846

theorem odd_multiple_of_three (a : ℕ) : 
  Odd (88 * a) → (88 * a) % 3 = 0 → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_odd_multiple_of_three_l3138_313846


namespace NUMINAMATH_CALUDE_cubic_function_extremum_l3138_313850

/-- Given a cubic function f with a local extremum at x = -1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + b*x + a^2

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + b

theorem cubic_function_extremum (a b : ℝ) (h1 : a > 1) 
  (h2 : f a b (-1) = 0) (h3 : f' a b (-1) = 0) :
  a = 2 ∧ b = 9 ∧ 
  (∀ x ∈ Set.Icc (-4 : ℝ) 0, 0 ≤ f a b x ∧ f a b x ≤ 4) ∧
  (∃ x ∈ Set.Icc (-4 : ℝ) 0, f a b x = 0) ∧
  (∃ x ∈ Set.Icc (-4 : ℝ) 0, f a b x = 4) := by
  sorry

#check cubic_function_extremum

end NUMINAMATH_CALUDE_cubic_function_extremum_l3138_313850


namespace NUMINAMATH_CALUDE_percentage_of_muslim_boys_l3138_313835

theorem percentage_of_muslim_boys (total_boys : ℕ) (hindu_percentage : ℚ) (sikh_percentage : ℚ) (other_boys : ℕ) :
  total_boys = 850 →
  hindu_percentage = 28 / 100 →
  sikh_percentage = 10 / 100 →
  other_boys = 187 →
  (total_boys - (hindu_percentage * total_boys + sikh_percentage * total_boys + other_boys)) / total_boys = 40 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_muslim_boys_l3138_313835


namespace NUMINAMATH_CALUDE_range_of_a_l3138_313864

-- Define the equation and its roots
def equation (m : ℝ) (x : ℝ) : Prop := x^2 - m*x - 2 = 0

-- Define the inequality condition for a
def inequality_condition (a m : ℝ) (x₁ x₂ : ℝ) : Prop :=
  a^2 - 5*a - 3 ≥ |x₁ - x₂|

-- Define the range for m
def m_range (m : ℝ) : Prop := -1 ≤ m ∧ m ≤ 1

-- Define the condition for the quadratic inequality having no solutions
def no_solutions (a : ℝ) : Prop :=
  ∀ x : ℝ, a*x^2 + 2*x - 1 ≤ 0

theorem range_of_a :
  ∀ m : ℝ, m_range m →
  ∀ x₁ x₂ : ℝ, equation m x₁ ∧ equation m x₂ ∧ x₁ ≠ x₂ →
  ∀ a : ℝ, (∀ m : ℝ, m_range m → inequality_condition a m x₁ x₂) ∧ no_solutions a →
  a ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3138_313864


namespace NUMINAMATH_CALUDE_negative_half_power_times_two_power_l3138_313816

theorem negative_half_power_times_two_power : (-0.5)^2016 * 2^2017 = 2 := by
  sorry

end NUMINAMATH_CALUDE_negative_half_power_times_two_power_l3138_313816


namespace NUMINAMATH_CALUDE_seastar_arms_l3138_313809

theorem seastar_arms (num_starfish : ℕ) (arms_per_starfish : ℕ) (total_arms : ℕ) : 
  num_starfish = 7 → arms_per_starfish = 5 → total_arms = 49 → 
  total_arms - (num_starfish * arms_per_starfish) = 14 := by
sorry

end NUMINAMATH_CALUDE_seastar_arms_l3138_313809


namespace NUMINAMATH_CALUDE_part_1_part_2_l3138_313866

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  t.A + t.B + t.C = 180 ∧ t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0

def satisfies_law_of_sines (t : Triangle) : Prop :=
  t.a / Real.sin t.A = t.b / Real.sin t.B ∧
  t.b / Real.sin t.B = t.c / Real.sin t.C

def is_geometric_sequence (a b c : Real) : Prop :=
  b * b = a * c

-- Theorem statements
theorem part_1 (t : Triangle) 
  (h1 : is_valid_triangle t)
  (h2 : satisfies_law_of_sines t)
  (h3 : t.B = 60)
  (h4 : t.b = Real.sqrt 3)
  (h5 : t.A = 45) :
  t.a = Real.sqrt 2 := by sorry

theorem part_2 (t : Triangle)
  (h1 : is_valid_triangle t)
  (h2 : satisfies_law_of_sines t)
  (h3 : t.B = 60)
  (h4 : is_geometric_sequence t.a t.b t.c) :
  t.A = 60 ∧ t.C = 60 := by sorry

end NUMINAMATH_CALUDE_part_1_part_2_l3138_313866


namespace NUMINAMATH_CALUDE_function_domain_constraint_l3138_313868

theorem function_domain_constraint (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x : ℝ, f x = (x - 7)^(1/3) / (a * x^2 + 4 * a * x + 3)) →
  (∀ x : ℝ, f x ≠ 0) →
  (0 < a ∧ a < 3/4) :=
sorry

end NUMINAMATH_CALUDE_function_domain_constraint_l3138_313868


namespace NUMINAMATH_CALUDE_gcd_of_abcd_plus_dcba_l3138_313814

def abcd_plus_dcba (a : ℕ) : ℕ := 4201 * a + 12606

def number_set : Set ℕ := {n | ∃ a : ℕ, 1 ≤ a ∧ a ≤ 4 ∧ n = abcd_plus_dcba a}

theorem gcd_of_abcd_plus_dcba :
  ∃ g : ℕ, g > 0 ∧ (∀ n ∈ number_set, g ∣ n) ∧
  (∀ d : ℕ, d > 0 → (∀ n ∈ number_set, d ∣ n) → d ≤ g) ∧
  g = 4201 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_abcd_plus_dcba_l3138_313814


namespace NUMINAMATH_CALUDE_fraction_simplification_l3138_313875

theorem fraction_simplification (x : ℝ) (h : 2 * x ≠ 2) :
  (6 * x^3 + 13 * x^2 + 15 * x - 25) / (2 * x^3 + 4 * x^2 + 4 * x - 10) = (6 * x - 5) / (2 * x - 2) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3138_313875


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_l3138_313803

theorem chocolate_bar_cost (total_bars : ℕ) (unsold_bars : ℕ) (total_revenue : ℕ) : 
  total_bars = 11 → 
  unsold_bars = 7 → 
  total_revenue = 16 → 
  (total_revenue : ℚ) / ((total_bars - unsold_bars) : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_cost_l3138_313803


namespace NUMINAMATH_CALUDE_big_eighteen_games_l3138_313888

/-- Represents a basketball conference with the given structure -/
structure BasketballConference where
  num_divisions : Nat
  teams_per_division : Nat
  intra_division_games : Nat
  inter_division_games : Nat

/-- Calculates the total number of conference games scheduled -/
def total_games (conf : BasketballConference) : Nat :=
  let total_teams := conf.num_divisions * conf.teams_per_division
  let games_per_team := (conf.teams_per_division - 1) * conf.intra_division_games + 
                        (total_teams - conf.teams_per_division) * conf.inter_division_games
  total_teams * games_per_team / 2

/-- The Big Eighteen Basketball Conference -/
def big_eighteen : BasketballConference :=
  { num_divisions := 3
  , teams_per_division := 6
  , intra_division_games := 3
  , inter_division_games := 1 }

theorem big_eighteen_games : total_games big_eighteen = 243 := by
  sorry

end NUMINAMATH_CALUDE_big_eighteen_games_l3138_313888


namespace NUMINAMATH_CALUDE_diagonal_increase_l3138_313886

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := sorry

theorem diagonal_increase (n : ℕ) :
  num_diagonals (n + 1) = num_diagonals n + n - 1 :=
by sorry

end NUMINAMATH_CALUDE_diagonal_increase_l3138_313886


namespace NUMINAMATH_CALUDE_inequality_solution_l3138_313813

theorem inequality_solution (x : ℝ) : 
  (Real.sqrt (x^3 - 18*x - 5) + 2) * abs (x^3 - 4*x^2 - 5*x + 18) ≤ 0 ↔ x = 1 - Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3138_313813


namespace NUMINAMATH_CALUDE_parabola_and_bisector_intercept_l3138_313879

-- Define the line l
def line_l (x y : ℝ) : Prop := y = (1/2) * (x + 4)

-- Define the parabola G
def parabola_G (p : ℝ) (x y : ℝ) : Prop := x^2 = 2 * p * y

-- Define the intersection points B and C
def intersection_points (xB yB xC yC : ℝ) : Prop :=
  line_l xB yB ∧ line_l xC yC ∧ 
  parabola_G 2 xB yB ∧ parabola_G 2 xC yC

-- Define the midpoint of BC
def midpoint_BC (x y : ℝ) : Prop :=
  ∃ xB yB xC yC, intersection_points xB yB xC yC ∧
  x = (xB + xC) / 2 ∧ y = (yB + yC) / 2

-- Define the perpendicular bisector of BC
def perp_bisector (x y : ℝ) : Prop :=
  ∃ x0 y0, midpoint_BC x0 y0 ∧ y - y0 = -2 * (x - x0)

-- Theorem statement
theorem parabola_and_bisector_intercept :
  (∃ p : ℝ, p > 0 ∧ ∀ x y, parabola_G p x y ↔ x^2 = 4 * y) ∧
  (∃ b : ℝ, b = 9/2 ∧ perp_bisector 0 b) ∧
  (∃ x, x = 1 ∧ midpoint_BC x ((1/2) * (x + 4))) :=
sorry

end NUMINAMATH_CALUDE_parabola_and_bisector_intercept_l3138_313879


namespace NUMINAMATH_CALUDE_number_relations_l3138_313861

theorem number_relations : 
  (∃ n : ℤ, 28 = 4 * n) ∧ 
  (∃ n : ℤ, 361 = 19 * n) ∧ 
  (∀ n : ℤ, 63 ≠ 19 * n) ∧ 
  (∃ n : ℤ, 45 = 15 * n) ∧ 
  (∃ n : ℤ, 30 = 15 * n) ∧ 
  (∃ n : ℤ, 144 = 12 * n) := by
sorry

end NUMINAMATH_CALUDE_number_relations_l3138_313861


namespace NUMINAMATH_CALUDE_arithmetic_seq_sum_l3138_313894

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- Given an arithmetic sequence with S_5 = 20, prove that a_2 + a_3 + a_4 = 12 -/
theorem arithmetic_seq_sum (seq : ArithmeticSequence) (h : seq.S 5 = 20) :
  seq.a 2 + seq.a 3 + seq.a 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_sum_l3138_313894


namespace NUMINAMATH_CALUDE_complex_fraction_theorem_l3138_313822

theorem complex_fraction_theorem (z₁ z₂ z₃ : ℂ) 
  (h1 : Complex.abs z₁ = Real.sqrt 5)
  (h2 : Complex.abs z₂ = Real.sqrt 5)
  (h3 : z₁ + z₃ = z₂) : 
  z₁ * z₂ / z₃^2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_theorem_l3138_313822


namespace NUMINAMATH_CALUDE_interest_difference_theorem_l3138_313805

theorem interest_difference_theorem (P : ℝ) : 
  P * ((1 + 0.1)^2 - 1) - P * 0.1 * 2 = 36 → P = 3600 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_theorem_l3138_313805


namespace NUMINAMATH_CALUDE_line_hyperbola_intersection_l3138_313833

theorem line_hyperbola_intersection :
  ∃ (k : ℝ), k > 0 ∧
  ∃ (x y : ℝ), y = Real.sqrt 3 * x ∧ y = k / x :=
by sorry

end NUMINAMATH_CALUDE_line_hyperbola_intersection_l3138_313833


namespace NUMINAMATH_CALUDE_cricket_players_count_l3138_313830

theorem cricket_players_count (total players : ℕ) (hockey football softball : ℕ) :
  total = 55 →
  hockey = 12 →
  football = 13 →
  softball = 15 →
  players = total - (hockey + football + softball) →
  players = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_cricket_players_count_l3138_313830


namespace NUMINAMATH_CALUDE_max_value_theorem_l3138_313862

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 1) :
  2*a*b + 2*b*c*Real.sqrt 2 ≤ Real.sqrt 3 ∧ 
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ 
    a'^2 + b'^2 + c'^2 = 1 ∧ 
    2*a'*b' + 2*b'*c'*Real.sqrt 2 = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3138_313862


namespace NUMINAMATH_CALUDE_modular_inverse_of_7_mod_10000_l3138_313801

-- Define the modulus
def m : ℕ := 10000

-- Define the number we're finding the inverse for
def a : ℕ := 7

-- Define the claimed inverse
def claimed_inverse : ℕ := 8571

-- Theorem statement
theorem modular_inverse_of_7_mod_10000 :
  (a * claimed_inverse) % m = 1 ∧ 0 ≤ claimed_inverse ∧ claimed_inverse < m :=
by sorry

end NUMINAMATH_CALUDE_modular_inverse_of_7_mod_10000_l3138_313801


namespace NUMINAMATH_CALUDE_trigonometric_propositions_l3138_313804

theorem trigonometric_propositions :
  (∃ α : ℝ, Real.sin α + Real.cos α = Real.sqrt 2) ∧
  (∀ x : ℝ, Real.sin (3 * Real.pi / 2 + x) = Real.sin (3 * Real.pi / 2 + (-x))) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_propositions_l3138_313804


namespace NUMINAMATH_CALUDE_rent_equation_l3138_313838

/-- The monthly rent of Janet's apartment -/
def monthly_rent : ℝ := 1250

/-- Janet's savings -/
def savings : ℝ := 2225

/-- Additional amount Janet needs -/
def additional_amount : ℝ := 775

/-- Deposit required by the landlord -/
def deposit : ℝ := 500

/-- Number of months' rent required in advance -/
def months_in_advance : ℕ := 2

theorem rent_equation :
  2 * monthly_rent + deposit = savings + additional_amount :=
by sorry

end NUMINAMATH_CALUDE_rent_equation_l3138_313838


namespace NUMINAMATH_CALUDE_expression_value_l3138_313877

theorem expression_value (x y z : ℤ) (hx : x = 3) (hy : y = 2) (hz : z = 1) :
  3 * x - 4 * y + 2 * z = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3138_313877


namespace NUMINAMATH_CALUDE_combined_contingency_funds_l3138_313828

/-- Calculates the combined contingency funds from two donations given specific conditions. -/
theorem combined_contingency_funds 
  (donation1 : ℝ) 
  (donation2 : ℝ) 
  (community_pantry_rate : ℝ) 
  (crisis_fund_rate : ℝ) 
  (livelihood_rate : ℝ) 
  (disaster_relief_rate : ℝ) 
  (international_aid_rate : ℝ) 
  (education_rate : ℝ) 
  (healthcare_rate : ℝ) 
  (conversion_rate : ℝ) :
  donation1 = 360 →
  donation2 = 180 →
  community_pantry_rate = 0.35 →
  crisis_fund_rate = 0.40 →
  livelihood_rate = 0.10 →
  disaster_relief_rate = 0.05 →
  international_aid_rate = 0.30 →
  education_rate = 0.25 →
  healthcare_rate = 0.25 →
  conversion_rate = 1.20 →
  (donation1 - (community_pantry_rate + crisis_fund_rate + livelihood_rate + disaster_relief_rate) * donation1) +
  (conversion_rate * donation2 - (international_aid_rate + education_rate + healthcare_rate) * conversion_rate * donation2) = 79.20 := by
  sorry


end NUMINAMATH_CALUDE_combined_contingency_funds_l3138_313828


namespace NUMINAMATH_CALUDE_all_stones_equal_weight_l3138_313873

/-- A type representing a stone with an integer weight -/
structure Stone where
  weight : ℤ

/-- A function that checks if a list of 12 stones can be split into two groups of 6 with equal weight -/
def canBalanceAny12 (stones : List Stone) : Prop :=
  stones.length = 13 ∧
  ∀ (subset : List Stone), subset.length = 12 ∧ subset.Sublist stones →
    ∃ (group1 group2 : List Stone),
      group1.length = 6 ∧ group2.length = 6 ∧
      group1.Sublist subset ∧ group2.Sublist subset ∧
      (group1.map Stone.weight).sum = (group2.map Stone.weight).sum

/-- The main theorem -/
theorem all_stones_equal_weight (stones : List Stone) :
  canBalanceAny12 stones →
  ∀ (s1 s2 : Stone), s1 ∈ stones → s2 ∈ stones → s1.weight = s2.weight :=
by sorry

end NUMINAMATH_CALUDE_all_stones_equal_weight_l3138_313873


namespace NUMINAMATH_CALUDE_lizard_eyes_count_l3138_313847

theorem lizard_eyes_count :
  ∀ (E W S : ℕ),
  W = 3 * E →
  S = 7 * W →
  E = S + W - 69 →
  E = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_lizard_eyes_count_l3138_313847


namespace NUMINAMATH_CALUDE_first_month_sale_l3138_313806

/-- Proves that the sale in the first month was 6235, given the sales for months 2-6
    and the desired average sale for 6 months. -/
theorem first_month_sale
  (sale_2 : ℕ) (sale_3 : ℕ) (sale_4 : ℕ) (sale_5 : ℕ) (sale_6 : ℕ) (average : ℕ)
  (h1 : sale_2 = 6927)
  (h2 : sale_3 = 6855)
  (h3 : sale_4 = 7230)
  (h4 : sale_5 = 6562)
  (h5 : sale_6 = 5191)
  (h6 : average = 6500) :
  6235 = 6 * average - (sale_2 + sale_3 + sale_4 + sale_5 + sale_6) :=
by sorry

end NUMINAMATH_CALUDE_first_month_sale_l3138_313806


namespace NUMINAMATH_CALUDE_distance_from_origin_to_point_l3138_313874

theorem distance_from_origin_to_point (x y : ℝ) :
  x = 8 ∧ y = -15 →
  Real.sqrt (x^2 + y^2) = 17 :=
by sorry

end NUMINAMATH_CALUDE_distance_from_origin_to_point_l3138_313874


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_l3138_313856

theorem perpendicular_lines_slope (a : ℝ) : 
  (∃ (x y : ℝ), y = a * x - 2) ∧ 
  (∃ (x y : ℝ), y = (a + 2) * x + 1) ∧ 
  (a * (a + 2) = -1) → 
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_l3138_313856


namespace NUMINAMATH_CALUDE_sets_operations_l3138_313898

-- Define the sets A and B
def A : Set ℝ := {x | -3 < x ∧ x < 2}
def B : Set ℝ := {x | Real.exp (x - 1) ≥ 1}

-- Define the theorem
theorem sets_operations :
  (A ∪ B = {x | x > -3}) ∧
  ((Aᶜ) ∩ B = {x | x ≥ 2}) := by
  sorry

end NUMINAMATH_CALUDE_sets_operations_l3138_313898


namespace NUMINAMATH_CALUDE_two_pump_filling_time_l3138_313820

/-- Given two pumps with different filling rates, calculate the time taken to fill a tank when both pumps work together. -/
theorem two_pump_filling_time 
  (small_pump_rate : ℝ) 
  (large_pump_rate : ℝ) 
  (h1 : small_pump_rate = 1 / 4) 
  (h2 : large_pump_rate = 2) : 
  1 / (small_pump_rate + large_pump_rate) = 4 / 9 := by
  sorry

#check two_pump_filling_time

end NUMINAMATH_CALUDE_two_pump_filling_time_l3138_313820


namespace NUMINAMATH_CALUDE_complete_graph_inequality_l3138_313815

theorem complete_graph_inequality (n k : ℕ) (N_k N_k_plus_1 : ℕ) 
  (h1 : 2 ≤ k) (h2 : k < n) (h3 : N_k > 0) (h4 : N_k_plus_1 > 0) :
  (N_k_plus_1 : ℚ) / N_k ≥ (1 : ℚ) / (k^2 - 1) * (k^2 * N_k / N_k_plus_1 - n) := by
  sorry

end NUMINAMATH_CALUDE_complete_graph_inequality_l3138_313815


namespace NUMINAMATH_CALUDE_expired_yogurt_percentage_l3138_313802

theorem expired_yogurt_percentage (total_packs : ℕ) (cost_per_pack : ℚ) (refund_amount : ℚ) :
  total_packs = 80 →
  cost_per_pack = 12 →
  refund_amount = 384 →
  (refund_amount / cost_per_pack) / total_packs * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_expired_yogurt_percentage_l3138_313802


namespace NUMINAMATH_CALUDE_second_caterer_cheaper_at_50_l3138_313893

/-- Represents the cost function for a caterer -/
structure Caterer where
  base_fee : ℕ
  per_person : ℕ
  discount : ℕ → ℕ

/-- Calculate the total cost for a caterer given the number of people -/
def total_cost (c : Caterer) (people : ℕ) : ℕ :=
  c.base_fee + c.per_person * people - c.discount people

/-- First caterer's pricing model -/
def caterer1 : Caterer :=
  { base_fee := 120
  , per_person := 18
  , discount := λ _ => 0 }

/-- Second caterer's pricing model -/
def caterer2 : Caterer :=
  { base_fee := 250
  , per_person := 14
  , discount := λ n => if n ≥ 50 then 50 else 0 }

/-- Theorem stating that 50 is the least number of people for which the second caterer is cheaper -/
theorem second_caterer_cheaper_at_50 :
  (total_cost caterer2 50 < total_cost caterer1 50) ∧
  (∀ n : ℕ, n < 50 → total_cost caterer1 n ≤ total_cost caterer2 n) :=
sorry

end NUMINAMATH_CALUDE_second_caterer_cheaper_at_50_l3138_313893


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3138_313800

theorem regular_polygon_sides (exterior_angle : ℝ) :
  exterior_angle = 18 →
  (360 / exterior_angle : ℝ) = 20 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3138_313800


namespace NUMINAMATH_CALUDE_practice_paper_percentage_l3138_313865

theorem practice_paper_percentage (total_students : ℕ) 
  (passed_all : ℝ) (passed_none : ℝ) (passed_one : ℝ) (passed_four : ℝ) (passed_three : ℕ)
  (h1 : total_students = 2500)
  (h2 : passed_all = 0.1)
  (h3 : passed_none = 0.1)
  (h4 : passed_one = 0.2 * (1 - passed_all - passed_none))
  (h5 : passed_four = 0.24)
  (h6 : passed_three = 500) :
  let remaining := 1 - passed_all - passed_none - passed_one - passed_four - (passed_three : ℝ) / total_students
  let passed_two := (1 - passed_all - passed_none - passed_one - passed_four - (passed_three : ℝ) / total_students) * remaining
  ∃ (ε : ℝ), abs (passed_two - 0.5002) < ε ∧ ε > 0 ∧ ε < 0.0001 :=
by sorry

end NUMINAMATH_CALUDE_practice_paper_percentage_l3138_313865


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_l3138_313839

theorem sin_cos_pi_12 : Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_l3138_313839
