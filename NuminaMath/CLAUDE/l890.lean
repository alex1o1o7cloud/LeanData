import Mathlib

namespace four_point_circle_theorem_l890_89046

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A type representing a circle in a plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Check if a point is on or inside a circle -/
def Point.onOrInside (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 ≤ c.radius^2

/-- Check if a point is on the circumference of a circle -/
def Point.onCircumference (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - p.x) = (r.y - p.y) * (q.x - p.x)

/-- The main theorem -/
theorem four_point_circle_theorem (a b c d : Point) 
  (h : ¬collinear a b c ∧ ¬collinear a b d ∧ ¬collinear a c d ∧ ¬collinear b c d) :
  ∃ (circ : Circle), 
    (Point.onCircumference a circ ∧ Point.onCircumference b circ ∧ Point.onCircumference c circ ∧ Point.onOrInside d circ) ∨
    (Point.onCircumference a circ ∧ Point.onCircumference b circ ∧ Point.onCircumference d circ ∧ Point.onOrInside c circ) ∨
    (Point.onCircumference a circ ∧ Point.onCircumference c circ ∧ Point.onCircumference d circ ∧ Point.onOrInside b circ) ∨
    (Point.onCircumference b circ ∧ Point.onCircumference c circ ∧ Point.onCircumference d circ ∧ Point.onOrInside a circ) :=
sorry

end four_point_circle_theorem_l890_89046


namespace parallelogram_area_from_boards_l890_89094

/-- The area of a parallelogram formed by two boards crossing at a 45-degree angle -/
theorem parallelogram_area_from_boards (board1_width board2_width : ℝ) 
  (h1 : board1_width = 5)
  (h2 : board2_width = 8)
  (h3 : Real.pi / 4 = 45 * Real.pi / 180) :
  board2_width * (board1_width * Real.sin (Real.pi / 4)) = 20 * Real.sqrt 2 := by
  sorry

end parallelogram_area_from_boards_l890_89094


namespace luke_stickers_l890_89029

/-- Calculates the number of stickers Luke has left after a series of events. -/
def stickers_left (initial : ℕ) (bought : ℕ) (birthday : ℕ) (given : ℕ) (used : ℕ) : ℕ :=
  initial + bought + birthday - given - used

/-- Proves that Luke has 39 stickers left after the given events. -/
theorem luke_stickers : stickers_left 20 12 20 5 8 = 39 := by
  sorry

end luke_stickers_l890_89029


namespace restriction_surjective_l890_89012

theorem restriction_surjective
  (f : Set.Ioc 0 1 → Set.Ioo 0 1)
  (hf_continuous : Continuous f)
  (hf_surjective : Function.Surjective f) :
  ∀ a ∈ Set.Ioo 0 1,
    Function.Surjective (fun x => f ⟨x, by sorry⟩ : Set.Ioo a 1 → Set.Ioo 0 1) :=
by sorry

end restriction_surjective_l890_89012


namespace min_value_of_expression_min_value_achievable_l890_89086

theorem min_value_of_expression (x : ℝ) : 
  (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 5 * Real.sqrt 6 / 3 :=
by sorry

theorem min_value_achievable : 
  ∃ x : ℝ, (x^2 + 9) / Real.sqrt (x^2 + 5) = 5 * Real.sqrt 6 / 3 :=
by sorry

end min_value_of_expression_min_value_achievable_l890_89086


namespace decimal_division_equivalence_l890_89049

theorem decimal_division_equivalence : 
  (∃ n : ℕ, (0.1 : ℚ) = n * (0.001 : ℚ) ∧ n = 100) ∧ 
  (∃ m : ℕ, (1 : ℚ) = m * (0.01 : ℚ) ∧ m = 100) := by
  sorry

#check decimal_division_equivalence

end decimal_division_equivalence_l890_89049


namespace symmetric_point_wrt_y_eq_x_l890_89081

def is_symmetric_point (p1 p2 : ℝ × ℝ) : Prop :=
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint.1 = midpoint.2 ∧ (p2.2 - p1.2) / (p2.1 - p1.1) = -1

theorem symmetric_point_wrt_y_eq_x : 
  is_symmetric_point (3, 1) (1, 3) :=
by sorry

end symmetric_point_wrt_y_eq_x_l890_89081


namespace sum_inequality_l890_89050

theorem sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/(2*b) + 1/(3*c) = 1) : a + 2*b + 3*c ≥ 9 := by
  sorry

end sum_inequality_l890_89050


namespace expected_outcome_is_correct_l890_89093

/-- Represents the possible outcomes of rolling a die -/
inductive DieOutcome
| One
| Two
| Three
| Four
| Five
| Six

/-- The probability of rolling a specific outcome -/
def probability (outcome : DieOutcome) : ℚ :=
  match outcome with
  | DieOutcome.One | DieOutcome.Two | DieOutcome.Three => 1/3
  | DieOutcome.Four | DieOutcome.Five | DieOutcome.Six => 1/6

/-- The monetary value associated with each outcome -/
def monetaryValue (outcome : DieOutcome) : ℚ :=
  match outcome with
  | DieOutcome.One | DieOutcome.Two | DieOutcome.Three => 4
  | DieOutcome.Four => -2
  | DieOutcome.Five => -5
  | DieOutcome.Six => -7

/-- The expected monetary outcome of a roll -/
def expectedMonetaryOutcome : ℚ :=
  (probability DieOutcome.One * monetaryValue DieOutcome.One) +
  (probability DieOutcome.Two * monetaryValue DieOutcome.Two) +
  (probability DieOutcome.Three * monetaryValue DieOutcome.Three) +
  (probability DieOutcome.Four * monetaryValue DieOutcome.Four) +
  (probability DieOutcome.Five * monetaryValue DieOutcome.Five) +
  (probability DieOutcome.Six * monetaryValue DieOutcome.Six)

theorem expected_outcome_is_correct :
  expectedMonetaryOutcome = 167/100 := by sorry

end expected_outcome_is_correct_l890_89093


namespace digit_difference_in_base_d_l890_89013

/-- Given digits A and B in base d > 8, if AB̄_d + AĀ_d = 194_d, then A_d - B_d = 5_d -/
theorem digit_difference_in_base_d (d A B : ℕ) (h1 : d > 8) 
  (h2 : A < d ∧ B < d) 
  (h3 : A * d + B + A * d + A = 1 * d * d + 9 * d + 4) : 
  A - B = 5 := by
  sorry

end digit_difference_in_base_d_l890_89013


namespace complex_number_quadrant_l890_89098

theorem complex_number_quadrant : ∃ (z : ℂ), z = (3 + 4*I)*I ∧ (z.re < 0 ∧ z.im > 0) :=
  sorry

end complex_number_quadrant_l890_89098


namespace toothbrush_ratio_l890_89066

theorem toothbrush_ratio (total brushes_jan brushes_feb brushes_mar : ℕ)
  (busiest_slowest_diff : ℕ) :
  total = 330 →
  brushes_jan = 53 →
  brushes_feb = 67 →
  brushes_mar = 46 →
  busiest_slowest_diff = 36 →
  ∃ (brushes_apr brushes_may : ℕ),
    brushes_apr + brushes_may = total - (brushes_jan + brushes_feb + brushes_mar) ∧
    brushes_apr - brushes_may = busiest_slowest_diff ∧
    brushes_apr * 16 = brushes_may * 25 :=
by sorry

end toothbrush_ratio_l890_89066


namespace student_contribution_l890_89020

theorem student_contribution
  (total_contribution : ℕ)
  (class_funds : ℕ)
  (num_students : ℕ)
  (h1 : total_contribution = 90)
  (h2 : class_funds = 14)
  (h3 : num_students = 19)
  : (total_contribution - class_funds) / num_students = 4 := by
  sorry

end student_contribution_l890_89020


namespace problem_solution_l890_89033

theorem problem_solution (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 2) : 
  (a + b)^2 = 17 ∧ a^2 - 6*a*b + b^2 = 1 := by
  sorry

end problem_solution_l890_89033


namespace problem_statement_l890_89080

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a * (6 - a) ≤ 9) ∧ 
  (a * b = a + b + 3 → a * b ≥ 9) ∧
  (a + b = 2 → 1 / a + 2 / b ≥ 3 / 2 + Real.sqrt 2) := by
  sorry

end problem_statement_l890_89080


namespace pancakes_remaining_l890_89014

theorem pancakes_remaining (total : ℕ) (bobby_ate : ℕ) (dog_ate : ℕ) : 
  total = 21 → bobby_ate = 5 → dog_ate = 7 → total - (bobby_ate + dog_ate) = 9 := by
  sorry

end pancakes_remaining_l890_89014


namespace multiply_divide_example_l890_89005

theorem multiply_divide_example : (3.242 * 10) / 100 = 0.3242 := by
  sorry

end multiply_divide_example_l890_89005


namespace equilateral_triangle_area_l890_89003

/-- For an equilateral triangle where the square of each side's length
    is equal to the perimeter, the area of the triangle is 9√3/4 square units. -/
theorem equilateral_triangle_area (s : ℝ) (h1 : s > 0) (h2 : s^2 = 3*s) :
  (s^2 * Real.sqrt 3) / 4 = 9 * Real.sqrt 3 / 4 := by
  sorry

end equilateral_triangle_area_l890_89003


namespace wendy_pictures_l890_89044

theorem wendy_pictures (total : ℕ) (num_albums : ℕ) (pics_per_album : ℕ) (first_album : ℕ) : 
  total = 79 →
  num_albums = 5 →
  pics_per_album = 7 →
  first_album + num_albums * pics_per_album = total →
  first_album = 44 := by
sorry

end wendy_pictures_l890_89044


namespace geometric_sequence_sum_l890_89076

/-- Given a geometric sequence {a_n} where a₁ = 3 and a₄ = 24, prove that a₃ + a₄ + a₅ = 84 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Condition for geometric sequence
  a 1 = 3 →
  a 4 = 24 →
  a 3 + a 4 + a 5 = 84 := by
sorry

end geometric_sequence_sum_l890_89076


namespace cubic_polynomial_remainder_l890_89074

/-- A cubic polynomial of the form ax³ - 6x² + bx - 5 -/
def f (a b x : ℝ) : ℝ := a * x^3 - 6 * x^2 + b * x - 5

theorem cubic_polynomial_remainder (a b : ℝ) :
  (f a b 1 = -5) ∧ (f a b (-2) = -53) → a = 7 ∧ b = -7 := by
  sorry

end cubic_polynomial_remainder_l890_89074


namespace percentage_problem_l890_89002

/-- The percentage that, when applied to 12356, results in 6.178 is 0.05% -/
theorem percentage_problem : ∃ p : ℝ, p * 12356 = 6.178 ∧ p = 0.0005 := by sorry

end percentage_problem_l890_89002


namespace A_and_D_independent_l890_89009

def num_balls : ℕ := 5

def prob_A : ℚ := 1 / num_balls
def prob_B : ℚ := 1 / num_balls
def prob_C : ℚ := 3 / (num_balls * num_balls)
def prob_D : ℚ := 1 / num_balls

def prob_AD : ℚ := 1 / (num_balls * num_balls)

theorem A_and_D_independent : prob_AD = prob_A * prob_D := by sorry

end A_and_D_independent_l890_89009


namespace investment_income_percentage_l890_89041

/-- Proves that the total annual income from two investments is 6% of the total invested amount -/
theorem investment_income_percentage : ∀ (investment1 investment2 rate1 rate2 : ℝ),
  investment1 = 2400 →
  investment2 = 2399.9999999999995 →
  rate1 = 0.04 →
  rate2 = 0.08 →
  let total_investment := investment1 + investment2
  let total_income := investment1 * rate1 + investment2 * rate2
  (total_income / total_investment) * 100 = 6 := by sorry

end investment_income_percentage_l890_89041


namespace min_value_of_x_l890_89015

theorem min_value_of_x (x : ℝ) (h1 : x > 0) (h2 : Real.log x ≥ Real.log 2 + (1/2) * Real.log x) : x ≥ 4 := by
  sorry

end min_value_of_x_l890_89015


namespace decimal_to_fraction_l890_89073

theorem decimal_to_fraction (x : ℚ) : x = 3.675 → x = 147 / 40 := by
  sorry

end decimal_to_fraction_l890_89073


namespace amount_received_by_B_l890_89083

/-- Theorem: Given a total amount of 1440, if A receives 1/3 as much as B, and B receives 1/4 as much as C, then B receives 202.5. -/
theorem amount_received_by_B (total : ℝ) (a b c : ℝ) : 
  total = 1440 →
  a = (1/3) * b →
  b = (1/4) * c →
  a + b + c = total →
  b = 202.5 := by
  sorry

end amount_received_by_B_l890_89083


namespace loan_years_approx_eight_l890_89099

/-- Calculates the number of years for which the first part of a loan is lent, given the total sum,
    the second part, and interest rates for both parts. -/
def calculate_years (total : ℚ) (second_part : ℚ) (rate1 : ℚ) (rate2 : ℚ) : ℚ :=
  let first_part := total - second_part
  let n := (second_part * rate2 * 3) / (first_part * rate1)
  n

/-- Proves that given the specified conditions, the number of years for which
    the first part is lent is approximately 8. -/
theorem loan_years_approx_eight :
  let total := 2691
  let second_part := 1656
  let rate1 := 3 / 100
  let rate2 := 5 / 100
  let years := calculate_years total second_part rate1 rate2
  ∃ ε > 0, abs (years - 8) < ε := by
  sorry


end loan_years_approx_eight_l890_89099


namespace partial_fraction_decomposition_l890_89024

-- Define the polynomial
def p (x : ℝ) : ℝ := x^3 - 18*x^2 + 91*x - 170

-- State the theorem
theorem partial_fraction_decomposition 
  (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_roots : p a = 0 ∧ p b = 0 ∧ p c = 0) 
  (D E F : ℝ) 
  (h_decomp : ∀ (s : ℝ), s ≠ a → s ≠ b → s ≠ c → 
    1 / (s^3 - 18*s^2 + 91*s - 170) = D / (s - a) + E / (s - b) + F / (s - c)) :
  D + E + F = 0 := by sorry

end partial_fraction_decomposition_l890_89024


namespace four_x_plus_t_is_odd_l890_89070

theorem four_x_plus_t_is_odd (x t : ℤ) (h : 2 * x - t = 11) : 
  ∃ k : ℤ, 4 * x + t = 2 * k + 1 := by
sorry

end four_x_plus_t_is_odd_l890_89070


namespace round_37_396_to_nearest_tenth_l890_89031

/-- Represents a repeating decimal with an integer part and a repeating fractional part -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Rounds a RepeatingDecimal to the nearest tenth -/
def roundToNearestTenth (x : RepeatingDecimal) : ℚ :=
  sorry

/-- The repeating decimal 37.396396... -/
def x : RepeatingDecimal :=
  { integerPart := 37, repeatingPart := 396 }

theorem round_37_396_to_nearest_tenth :
  roundToNearestTenth x = 37.4 := by
  sorry

end round_37_396_to_nearest_tenth_l890_89031


namespace sin_45_degrees_l890_89038

theorem sin_45_degrees : Real.sin (π / 4) = Real.sqrt 2 / 2 := by
  -- Define the properties of the unit circle and 45° angle
  have unit_circle : ∀ θ, Real.sin θ ^ 2 + Real.cos θ ^ 2 = 1 := by sorry
  have symmetry_45 : Real.sin (π / 4) = Real.cos (π / 4) := by sorry

  -- Proof goes here
  sorry

end sin_45_degrees_l890_89038


namespace no_dual_integer_root_quadratics_l890_89072

theorem no_dual_integer_root_quadratics : 
  ¬ ∃ (a b c : ℤ), 
    (∃ (x₁ x₂ : ℤ), x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) ∧
    (∃ (y₁ y₂ : ℤ), y₁ ≠ y₂ ∧ (a + 1) * y₁^2 + (b + 1) * y₁ + (c + 1) = 0 ∧ (a + 1) * y₂^2 + (b + 1) * y₂ + (c + 1) = 0) :=
by sorry

end no_dual_integer_root_quadratics_l890_89072


namespace angle_measure_l890_89057

theorem angle_measure (x : ℝ) : 
  (180 - x = 4 * (90 - x)) → x = 60 := by
  sorry

end angle_measure_l890_89057


namespace sqrt_product_sqrt_l890_89004

theorem sqrt_product_sqrt : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by sorry

end sqrt_product_sqrt_l890_89004


namespace f_5_eq_neg_f_3_l890_89030

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def periodic_negative (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = -f x

def quadratic_on_interval (f : ℝ → ℝ) : Prop := ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2

theorem f_5_eq_neg_f_3 (f : ℝ → ℝ) 
  (h1 : is_odd f) 
  (h2 : periodic_negative f) 
  (h3 : quadratic_on_interval f) : 
  f 5 = -f 3 := by
  sorry

end f_5_eq_neg_f_3_l890_89030


namespace jeff_fills_ten_boxes_l890_89048

/-- The number of boxes Jeff can fill with his donuts -/
def boxes_filled (donuts_per_day : ℕ) (days : ℕ) (jeff_eats_per_day : ℕ) (chris_eats : ℕ) (donuts_per_box : ℕ) : ℕ :=
  ((donuts_per_day * days) - (jeff_eats_per_day * days) - chris_eats) / donuts_per_box

/-- Proof that Jeff can fill 10 boxes with his donuts -/
theorem jeff_fills_ten_boxes :
  boxes_filled 10 12 1 8 10 = 10 := by
  sorry

end jeff_fills_ten_boxes_l890_89048


namespace kotelmel_triangle_area_error_l890_89023

/-- The margin of error between Kotelmel's formula and the correct formula for the area of an equilateral triangle --/
theorem kotelmel_triangle_area_error :
  let a : ℝ := 1  -- We can use any positive real number for a
  let kotelmel_area := (1/3 + 1/10) * a^2
  let correct_area := (a^2 / 4) * Real.sqrt 3
  let error_percentage := |correct_area - kotelmel_area| / correct_area * 100
  ∃ ε > 0, error_percentage < 0.075 + ε ∧ error_percentage > 0.075 - ε :=
by sorry


end kotelmel_triangle_area_error_l890_89023


namespace parallel_line_through_point_l890_89077

/-- Given a line with slope m and a point P(x1, y1), prove that the line
    y = mx + (y1 - mx1) passes through P and is parallel to the original line. -/
theorem parallel_line_through_point (m x1 y1 : ℝ) :
  let L2 : ℝ → ℝ := λ x => m * x + (y1 - m * x1)
  (L2 x1 = y1) ∧ (∀ x y, y = L2 x ↔ y - y1 = m * (x - x1)) := by sorry

end parallel_line_through_point_l890_89077


namespace leila_earnings_proof_l890_89040

-- Define the given conditions
def voltaire_daily_viewers : ℕ := 50
def leila_daily_viewers : ℕ := 2 * voltaire_daily_viewers
def earnings_per_view : ℚ := 1/2

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define Leila's weekly earnings
def leila_weekly_earnings : ℚ := leila_daily_viewers * earnings_per_view * days_in_week

-- Theorem statement
theorem leila_earnings_proof : leila_weekly_earnings = 350 := by
  sorry

end leila_earnings_proof_l890_89040


namespace difference_of_squares_l890_89016

theorem difference_of_squares (m : ℝ) : m^2 - 1 = (m + 1) * (m - 1) := by sorry

end difference_of_squares_l890_89016


namespace xy_plus_one_is_square_l890_89039

theorem xy_plus_one_is_square (x y : ℕ) 
  (h : (1 : ℚ) / x + (1 : ℚ) / y = 1 / (x + 2) + 1 / (y - 2)) : 
  ∃ (n : ℤ), (x * y + 1 : ℤ) = n ^ 2 := by
sorry

end xy_plus_one_is_square_l890_89039


namespace complex_equation_proof_l890_89087

theorem complex_equation_proof (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + x*y + y^2 = 0) : 
  (x / (x + y))^2005 + (y / (x + y))^2005 = 1 := by
  sorry

end complex_equation_proof_l890_89087


namespace g_12_equals_155_l890_89043

/-- The function g defined for all integers n -/
def g (n : ℤ) : ℤ := n^2 - n + 23

/-- Theorem stating that g(12) equals 155 -/
theorem g_12_equals_155 : g 12 = 155 := by
  sorry

end g_12_equals_155_l890_89043


namespace molecular_weight_aluminum_iodide_l890_89089

/-- Given that the molecular weight of 7 moles of aluminum iodide is 2856 grams,
    prove that the molecular weight of one mole of aluminum iodide is 408 grams/mole. -/
theorem molecular_weight_aluminum_iodide :
  let total_weight : ℝ := 2856
  let num_moles : ℝ := 7
  total_weight / num_moles = 408 := by
  sorry

end molecular_weight_aluminum_iodide_l890_89089


namespace five_travelers_three_rooms_l890_89019

/-- The number of ways to arrange travelers into guest rooms -/
def arrange_travelers (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- There are at least 1 traveler in each room -/
axiom at_least_one (n k : ℕ) : arrange_travelers n k > 0 → k ≤ n

theorem five_travelers_three_rooms :
  arrange_travelers 5 3 = 150 :=
sorry

end five_travelers_three_rooms_l890_89019


namespace equal_increment_implies_linear_l890_89011

/-- A function with the property that equal increments in input correspond to equal increments in output -/
def EqualIncrementFunction (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ x₃ x₄ : ℝ, x₂ - x₁ = x₄ - x₃ → f x₂ - f x₁ = f x₄ - f x₃

/-- The main theorem: if a function has the equal increment property, then it is linear -/
theorem equal_increment_implies_linear (f : ℝ → ℝ) (h : EqualIncrementFunction f) :
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b :=
sorry

end equal_increment_implies_linear_l890_89011


namespace hyperbola_condition_l890_89055

/-- The equation (x^2)/(k-2) + (y^2)/(5-k) = 1 represents a hyperbola -/
def is_hyperbola (k : ℝ) : Prop :=
  k < 2 ∨ k > 5

/-- The general form of the equation -/
def equation (x y k : ℝ) : Prop :=
  x^2 / (k - 2) + y^2 / (5 - k) = 1

theorem hyperbola_condition (k : ℝ) :
  (∀ x y, equation x y k → is_hyperbola k) ∧
  (is_hyperbola k → ∃ x y, equation x y k) :=
sorry

end hyperbola_condition_l890_89055


namespace work_increase_with_absence_l890_89078

/-- Given a total work W distributed among p persons, if 1/5 of the members are absent,
    the increase in work for each remaining person is W/(4p). -/
theorem work_increase_with_absence (W p : ℝ) (h : p > 0) :
  let original_work_per_person := W / p
  let remaining_persons := (4 / 5) * p
  let new_work_per_person := W / remaining_persons
  new_work_per_person - original_work_per_person = W / (4 * p) :=
by sorry

end work_increase_with_absence_l890_89078


namespace playground_area_l890_89092

/-- A rectangular playground with perimeter 72 feet and length three times the width has an area of 243 square feet. -/
theorem playground_area : ∀ w l : ℝ,
  w > 0 →
  l > 0 →
  2 * (w + l) = 72 →
  l = 3 * w →
  w * l = 243 := by
sorry

end playground_area_l890_89092


namespace min_value_quadratic_min_value_achieved_l890_89007

theorem min_value_quadratic (x y : ℝ) : 
  y = 5 * x^2 - 8 * x + 20 → x ≥ 1 → y ≥ 13 := by
  sorry

theorem min_value_achieved (x : ℝ) : 
  x ≥ 1 → ∃ y : ℝ, y = 5 * x^2 - 8 * x + 20 ∧ y = 13 := by
  sorry

end min_value_quadratic_min_value_achieved_l890_89007


namespace second_category_amount_is_720_l890_89053

/-- Represents a budget with three categories -/
structure Budget where
  total : ℕ
  ratio1 : ℕ
  ratio2 : ℕ
  ratio3 : ℕ

/-- Calculates the amount allocated to the second category in a budget -/
def amount_second_category (b : Budget) : ℕ :=
  b.total * b.ratio2 / (b.ratio1 + b.ratio2 + b.ratio3)

/-- Theorem stating that for a budget with ratio 5:4:1 and total $1800, 
    the amount allocated to the second category is $720 -/
theorem second_category_amount_is_720 :
  ∀ (b : Budget), b.total = 1800 ∧ b.ratio1 = 5 ∧ b.ratio2 = 4 ∧ b.ratio3 = 1 →
  amount_second_category b = 720 := by
  sorry

end second_category_amount_is_720_l890_89053


namespace crayons_remaining_l890_89035

theorem crayons_remaining (initial : ℕ) (given_away : ℕ) (lost : ℕ) : 
  initial = 440 → given_away = 111 → lost = 106 → initial - given_away - lost = 223 := by
  sorry

end crayons_remaining_l890_89035


namespace x_intercept_of_l_equation_of_l_l890_89025

-- Define the lines l₁, l₂, and l₃
def l₁ (x y : ℝ) : Prop := x - y + 1 = 0
def l₂ (x y : ℝ) : Prop := 2*x + y - 4 = 0
def l₃ (x y : ℝ) : Prop := 4*x + 5*y - 12 = 0

-- Define the intersection point of l₁ and l₂
def intersection (x y : ℝ) : Prop := l₁ x y ∧ l₂ x y

-- Define line l
def l (x y : ℝ) : Prop := ∃ (ix iy : ℝ), intersection ix iy ∧ (y - iy) = ((x - ix) * (3 - iy)) / (3 - ix)

-- Theorem for part 1
theorem x_intercept_of_l : 
  (∃ (x y : ℝ), intersection x y) → 
  l 3 3 → 
  (∃ (x : ℝ), l x 0 ∧ x = -3) :=
sorry

-- Theorem for part 2
theorem equation_of_l :
  (∃ (x y : ℝ), intersection x y) →
  (∀ (x y : ℝ), l x y ↔ l₃ (x + a) (y + b)) →
  (∀ (x y : ℝ), l x y ↔ 4*x + 5*y - 14 = 0) :=
sorry

end x_intercept_of_l_equation_of_l_l890_89025


namespace range_of_x_plus_y_l890_89032

theorem range_of_x_plus_y (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) :
  ∃ (z : ℝ), z = x + y ∧ -Real.sqrt 6 ≤ z ∧ z ≤ Real.sqrt 6 ∧
  ∀ (w : ℝ), (∃ (a b : ℝ), w = a + b ∧ a^2 + 2*a*b + 4*b^2 = 6) →
    -Real.sqrt 6 ≤ w ∧ w ≤ Real.sqrt 6 :=
sorry

end range_of_x_plus_y_l890_89032


namespace tree_shadow_length_l890_89028

/-- Given a tree and a flag pole, proves the length of the tree's shadow. -/
theorem tree_shadow_length 
  (tree_height : ℝ) 
  (flagpole_height : ℝ) 
  (flagpole_shadow : ℝ) 
  (h1 : tree_height = 12)
  (h2 : flagpole_height = 150)
  (h3 : flagpole_shadow = 100) : 
  (tree_height * flagpole_shadow) / flagpole_height = 8 :=
by sorry

end tree_shadow_length_l890_89028


namespace highway_distance_l890_89027

/-- Proves the distance a car can travel on the highway given its city fuel efficiency and efficiency increase -/
theorem highway_distance (city_efficiency : ℝ) (efficiency_increase : ℝ) (highway_gas : ℝ) :
  city_efficiency = 30 →
  efficiency_increase = 0.2 →
  highway_gas = 7 →
  (city_efficiency * (1 + efficiency_increase)) * highway_gas = 252 := by
  sorry

end highway_distance_l890_89027


namespace evaluate_expression_l890_89052

theorem evaluate_expression : 
  (7 ^ (1/4 : ℝ)) / (3 ^ (1/3 : ℝ)) / ((7 ^ (1/2 : ℝ)) / (3 ^ (1/6 : ℝ))) = 
  (1/7 : ℝ) ^ (1/4 : ℝ) * (1/3 : ℝ) ^ (1/6 : ℝ) := by
  sorry

end evaluate_expression_l890_89052


namespace negation_of_universal_statement_l890_89054

theorem negation_of_universal_statement :
  (¬ (∀ x : ℝ, 2^x + x^2 > 0)) ↔ (∃ x₀ : ℝ, 2^x₀ + x₀^2 ≤ 0) := by
  sorry

end negation_of_universal_statement_l890_89054


namespace sum_of_powers_of_i_l890_89097

def i : ℂ := Complex.I

theorem sum_of_powers_of_i : i^14760 + i^14761 + i^14762 + i^14763 = 0 := by sorry

end sum_of_powers_of_i_l890_89097


namespace emma_henry_weight_l890_89017

theorem emma_henry_weight (e f g h : ℝ) 
  (ef_sum : e + f = 310)
  (fg_sum : f + g = 265)
  (gh_sum : g + h = 280) :
  e + h = 325 := by
sorry

end emma_henry_weight_l890_89017


namespace simplify_product_of_square_roots_l890_89065

theorem simplify_product_of_square_roots (y : ℝ) (h : y ≥ 0) :
  Real.sqrt (48 * y) * Real.sqrt (18 * y) * Real.sqrt (50 * y) = 120 * y * Real.sqrt (3 * y) := by
  sorry

end simplify_product_of_square_roots_l890_89065


namespace complex_equation_solution_l890_89006

theorem complex_equation_solution (z : ℂ) : (3 - I) * z = 1 - I → z = 2/5 - 1/5 * I := by
  sorry

end complex_equation_solution_l890_89006


namespace product_is_three_l890_89082

/-- The repeating decimal 0.333... --/
def repeating_third : ℚ := 1/3

/-- The product of the repeating decimal 0.333... and 9 --/
def product : ℚ := repeating_third * 9

/-- Theorem stating that the product of 0.333... and 9 is equal to 3 --/
theorem product_is_three : product = 3 := by sorry

end product_is_three_l890_89082


namespace no_one_blue_point_coloring_l890_89075

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a color type
inductive Color
  | Red
  | Blue

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define a circle of radius 1
def unitCircle (center : Point) : Set Point :=
  {p : Point | (p.x - center.x)^2 + (p.y - center.y)^2 = 1}

-- State the theorem
theorem no_one_blue_point_coloring :
  ¬ (∀ (center : Point),
      ∃! (p : Point), p ∈ unitCircle center ∧ coloring p = Color.Blue) ∧
    (∃ (p q : Point), coloring p ≠ coloring q) :=
  sorry

end no_one_blue_point_coloring_l890_89075


namespace marble_probability_theorem_l890_89000

/-- Represents a box of marbles -/
structure MarbleBox where
  total : ℕ
  red : ℕ
  blue : ℕ
  sum_constraint : red + blue = total

/-- Represents the probability of drawing two marbles of the same color -/
def drawProbability (box1 box2 : MarbleBox) (color : ℕ → ℕ) : ℚ :=
  (color box1.red / box1.total) * (color box2.red / box2.total)

theorem marble_probability_theorem (box1 box2 : MarbleBox) :
  box1.total + box2.total = 34 →
  drawProbability box1 box2 (fun x => x) = 19/34 →
  drawProbability box1 box2 (fun x => box1.total - x) = 64/289 := by
  sorry

end marble_probability_theorem_l890_89000


namespace average_difference_l890_89069

/-- The average of an arithmetic sequence with first term a and last term b -/
def arithmeticMean (a b : Int) : Rat := (a + b) / 2

/-- The set of even integers from a to b inclusive -/
def evenIntegers (a b : Int) : Set Int := {n : Int | a ≤ n ∧ n ≤ b ∧ n % 2 = 0}

/-- The set of odd integers from a to b inclusive -/
def oddIntegers (a b : Int) : Set Int := {n : Int | a ≤ n ∧ n ≤ b ∧ n % 2 = 1}

theorem average_difference :
  (arithmeticMean 20 60 - arithmeticMean 10 140 = -35) ∧
  (arithmeticMean 21 59 - arithmeticMean 11 139 = -35) := by
  sorry

end average_difference_l890_89069


namespace complex_division_equality_l890_89008

theorem complex_division_equality : Complex.I * 2 / (1 - Complex.I) = -1 + Complex.I := by sorry

end complex_division_equality_l890_89008


namespace pyramid_height_equals_cube_volume_l890_89090

theorem pyramid_height_equals_cube_volume (cube_edge : ℝ) (pyramid_base : ℝ) (pyramid_height : ℝ) :
  cube_edge = 5 →
  pyramid_base = 10 →
  cube_edge ^ 3 = (1 / 3) * pyramid_base ^ 2 * pyramid_height →
  pyramid_height = 3.75 := by
sorry

end pyramid_height_equals_cube_volume_l890_89090


namespace correct_operation_l890_89063

theorem correct_operation (a b : ℝ) : 3 * a^2 * b - 3 * b * a^2 = 0 := by
  sorry

end correct_operation_l890_89063


namespace lunch_break_duration_l890_89037

-- Define the painting rates and lunch break duration
structure PaintingScenario where
  joseph_rate : ℝ
  helpers_rate : ℝ
  lunch_break : ℝ

-- Define the conditions from the problem
def monday_condition (s : PaintingScenario) : Prop :=
  (8 - s.lunch_break) * (s.joseph_rate + s.helpers_rate) = 0.6

def tuesday_condition (s : PaintingScenario) : Prop :=
  (5 - s.lunch_break) * s.helpers_rate = 0.3

def wednesday_condition (s : PaintingScenario) : Prop :=
  (6 - s.lunch_break) * s.joseph_rate = 0.1

-- Theorem stating that the lunch break is 45 minutes
theorem lunch_break_duration :
  ∃ (s : PaintingScenario),
    monday_condition s ∧
    tuesday_condition s ∧
    wednesday_condition s ∧
    s.lunch_break = 0.75 := by sorry

end lunch_break_duration_l890_89037


namespace greatest_number_with_odd_factors_l890_89051

theorem greatest_number_with_odd_factors :
  ∀ n : ℕ, n < 1000 → (∃ k : ℕ, n = k^2) →
  (∀ m : ℕ, m < 1000 → (∃ l : ℕ, m = l^2) → m ≤ n) →
  n = 961 :=
by sorry

end greatest_number_with_odd_factors_l890_89051


namespace probability_endpoints_of_edge_is_four_fifths_l890_89096

/-- A regular octahedron -/
structure RegularOctahedron where
  vertices : Finset (Fin 6)
  edges : Finset (Fin 6 × Fin 6)
  vertex_count : vertices.card = 6
  edge_count_per_vertex : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 4

/-- The probability of choosing two vertices that are endpoints of an edge -/
def probability_endpoints_of_edge (o : RegularOctahedron) : ℚ :=
  (4 : ℚ) / 5

/-- Theorem: The probability of randomly choosing two vertices of a regular octahedron 
    that are endpoints of an edge is 4/5 -/
theorem probability_endpoints_of_edge_is_four_fifths (o : RegularOctahedron) :
  probability_endpoints_of_edge o = 4 / 5 := by
  sorry

end probability_endpoints_of_edge_is_four_fifths_l890_89096


namespace inequality_system_solution_set_l890_89095

theorem inequality_system_solution_set :
  {x : ℝ | 6 > 2 * (x + 1) ∧ 1 - x < 2} = {x : ℝ | -1 < x ∧ x < 2} := by
  sorry

end inequality_system_solution_set_l890_89095


namespace sum_thirteen_is_156_l890_89059

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  sum : ℕ → ℝ -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, sum n = n * (2 * a 1 + (n - 1) * d) / 2
  third_sum : sum 3 = 6
  specific_sum : a 9 + a 11 + a 13 = 60

/-- The sum of the first 13 terms of the arithmetic sequence is 156 -/
theorem sum_thirteen_is_156 (seq : ArithmeticSequence) : seq.sum 13 = 156 := by
  sorry

end sum_thirteen_is_156_l890_89059


namespace complex_subtraction_l890_89034

theorem complex_subtraction : (7 - 3*I) - (2 + 4*I) = 5 - 7*I := by sorry

end complex_subtraction_l890_89034


namespace bug_crawl_distance_l890_89088

-- Define the positions of the bug
def start_pos : ℤ := 3
def pos1 : ℤ := -5
def pos2 : ℤ := 8
def end_pos : ℤ := 0

-- Define the function to calculate distance between two points
def distance (a b : ℤ) : ℕ := (a - b).natAbs

-- Define the total distance
def total_distance : ℕ := 
  distance start_pos pos1 + distance pos1 pos2 + distance pos2 end_pos

-- Theorem to prove
theorem bug_crawl_distance : total_distance = 29 := by
  sorry

end bug_crawl_distance_l890_89088


namespace equation_solution_l890_89010

theorem equation_solution (x : ℂ) : 
  (x^2 + x + 1) / (x + 1) = x^2 + 2*x + 2 ↔ 
  (x = -1 ∨ x = (-1 + Complex.I * Real.sqrt 3) / 2 ∨ x = (-1 - Complex.I * Real.sqrt 3) / 2) :=
by sorry

end equation_solution_l890_89010


namespace expected_value_binomial_li_expected_traffic_jams_l890_89061

/-- The number of intersections Mr. Li passes through -/
def n : ℕ := 6

/-- The probability of a traffic jam at each intersection -/
def p : ℚ := 1/6

/-- The expected value of a binomial distribution is n * p -/
theorem expected_value_binomial (n : ℕ) (p : ℚ) :
  n * p = 1 → n = 6 ∧ p = 1/6 := by sorry

/-- The expected number of traffic jams Mr. Li encounters is 1 -/
theorem li_expected_traffic_jams :
  n * p = 1 := by sorry

end expected_value_binomial_li_expected_traffic_jams_l890_89061


namespace g_sum_symmetric_l890_89022

/-- Given a function g(x) = ax^8 + bx^6 - cx^4 + 5 where g(10) = 3,
    prove that g(10) + g(-10) = 6 -/
theorem g_sum_symmetric (a b c : ℝ) : 
  let g : ℝ → ℝ := λ x ↦ a * x^8 + b * x^6 - c * x^4 + 5
  g 10 = 3 → g 10 + g (-10) = 6 := by sorry

end g_sum_symmetric_l890_89022


namespace inequality_proof_l890_89062

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) * (a^2 + b^2 + c^2) ≥ 9 * a * b * c := by
  sorry

end inequality_proof_l890_89062


namespace william_washed_two_normal_cars_l890_89036

/-- The time William spends washing a normal car's windows -/
def window_time : ℕ := 4

/-- The time William spends washing a normal car's body -/
def body_time : ℕ := 7

/-- The time William spends cleaning a normal car's tires -/
def tire_time : ℕ := 4

/-- The time William spends waxing a normal car -/
def wax_time : ℕ := 9

/-- The total time William spends on one normal car -/
def normal_car_time : ℕ := window_time + body_time + tire_time + wax_time

/-- The time William spends on one big SUV -/
def suv_time : ℕ := 2 * normal_car_time

/-- The total time William spent washing all vehicles -/
def total_time : ℕ := 96

/-- The number of normal cars William washed -/
def normal_cars : ℕ := (total_time - suv_time) / normal_car_time

theorem william_washed_two_normal_cars : normal_cars = 2 := by
  sorry

end william_washed_two_normal_cars_l890_89036


namespace smallest_n_value_l890_89068

/-- The number of ordered quadruplets (a, b, c, d) satisfying the given conditions -/
def num_quadruplets : ℕ := 90000

/-- The greatest common divisor of the quadruplets -/
def quadruplet_gcd : ℕ := 90

/-- 
  The function that counts the number of ordered quadruplets (a, b, c, d) 
  satisfying gcd(a, b, c, d) = quadruplet_gcd and lcm(a, b, c, d) = n
-/
def count_quadruplets (n : ℕ) : ℕ := sorry

/-- The theorem stating the smallest possible value of n -/
theorem smallest_n_value : 
  (∃ (n : ℕ), n > 0 ∧ count_quadruplets n = num_quadruplets) ∧ 
  (∀ (m : ℕ), m > 0 ∧ count_quadruplets m = num_quadruplets → m ≥ 32400) :=
sorry

end smallest_n_value_l890_89068


namespace photo_arrangement_count_l890_89060

/-- The number of ways to arrange 5 people in a row for a photo, with one person fixed in the middle -/
def photo_arrangements : ℕ := 24

/-- The number of people in the photo -/
def total_people : ℕ := 5

/-- The number of people who can be arranged in non-middle positions -/
def non_middle_people : ℕ := total_people - 1

theorem photo_arrangement_count : photo_arrangements = non_middle_people! := by
  sorry

end photo_arrangement_count_l890_89060


namespace partial_fraction_decomposition_constant_l890_89071

theorem partial_fraction_decomposition_constant (A B C : ℝ) :
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 2 →
    1 / (x^3 + 2*x^2 - 19*x - 30) = A / (x + 3) + B / (x - 2) + C / ((x - 2)^2)) →
  A = 1/25 := by
sorry

end partial_fraction_decomposition_constant_l890_89071


namespace storage_area_wheels_l890_89056

theorem storage_area_wheels (bicycles tricycles unicycles cars : ℕ)
  (h_bicycles : bicycles = 24)
  (h_tricycles : tricycles = 14)
  (h_unicycles : unicycles = 10)
  (h_cars : cars = 18) :
  let total_wheels := bicycles * 2 + tricycles * 3 + unicycles * 1 + cars * 4
  let unicycle_wheels := unicycles * 1
  let ratio_numerator := unicycle_wheels
  let ratio_denominator := total_wheels
  (total_wheels = 172) ∧ 
  (ratio_numerator = 5 ∧ ratio_denominator = 86) :=
by sorry

end storage_area_wheels_l890_89056


namespace homer_candy_crush_ratio_l890_89026

/-- Proves that the ratio of points scored on the third try to points scored on the second try is 2:1 in Homer's Candy Crush game -/
theorem homer_candy_crush_ratio :
  ∀ (first_try second_try third_try : ℕ),
    first_try = 400 →
    second_try = first_try - 70 →
    ∃ (m : ℕ), third_try = m * second_try →
    first_try + second_try + third_try = 1390 →
    third_try = 2 * second_try :=
by
  sorry

#check homer_candy_crush_ratio

end homer_candy_crush_ratio_l890_89026


namespace distribution_combinations_l890_89021

/-- The number of ways to distribute 2 objects among 4 categories -/
def distributionCount : ℕ := 10

/-- The number of categories -/
def categoryCount : ℕ := 4

/-- The number of objects to distribute -/
def objectCount : ℕ := 2

theorem distribution_combinations :
  (categoryCount : ℕ) + (categoryCount * (categoryCount - 1) / 2) = distributionCount :=
sorry

end distribution_combinations_l890_89021


namespace swimming_speed_in_still_water_l890_89045

/-- The swimming speed of a person in still water, given their performance against a current. -/
theorem swimming_speed_in_still_water 
  (current_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : current_speed = 4) 
  (h2 : distance = 12) 
  (h3 : time = 2) : 
  ∃ (speed : ℝ), speed - current_speed = distance / time ∧ speed = 10 :=
sorry

end swimming_speed_in_still_water_l890_89045


namespace expand_product_l890_89084

theorem expand_product (x y : ℝ) : (3*x + 4*y)*(2*x - 5*y + 7) = 6*x^2 - 7*x*y + 21*x - 20*y^2 + 28*y := by
  sorry

end expand_product_l890_89084


namespace area_of_quadrilateral_OBEC_l890_89079

/-- A line with slope -3 passing through points A and B -/
def line1 (x y : ℝ) : Prop := y = -3 * x + 13

/-- A line passing through points C and D -/
def line2 (x y : ℝ) : Prop := y = -x + 7

/-- Point A on the x-axis -/
def A : ℝ × ℝ := (5, 0)

/-- Point B on the y-axis -/
def B : ℝ × ℝ := (0, 13)

/-- Point C on the x-axis -/
def C : ℝ × ℝ := (5, 0)

/-- Point D on the y-axis -/
def D : ℝ × ℝ := (0, 7)

/-- Point E where the lines intersect -/
def E : ℝ × ℝ := (3, 4)

/-- The area of quadrilateral OBEC -/
def area_OBEC : ℝ := 67.5

theorem area_of_quadrilateral_OBEC :
  line1 E.1 E.2 ∧ line2 E.1 E.2 →
  area_OBEC = (B.2 * E.1 + C.1 * E.2) / 2 := by
  sorry

end area_of_quadrilateral_OBEC_l890_89079


namespace all_cells_colored_l890_89001

/-- Represents a 6x6 grid where cells can be colored -/
structure Grid :=
  (colored : Fin 6 → Fin 6 → Bool)

/-- Returns the number of colored cells in a 2x2 square starting at (i, j) -/
def count_2x2 (g : Grid) (i j : Fin 4) : Nat :=
  (g.colored i j).toNat + (g.colored i (j + 1)).toNat +
  (g.colored (i + 1) j).toNat + (g.colored (i + 1) (j + 1)).toNat

/-- Returns the number of colored cells in a 1x3 stripe starting at (i, j) -/
def count_1x3 (g : Grid) (i : Fin 6) (j : Fin 4) : Nat :=
  (g.colored i j).toNat + (g.colored i (j + 1)).toNat + (g.colored i (j + 2)).toNat

/-- The main theorem -/
theorem all_cells_colored (g : Grid) 
  (h1 : ∀ i j : Fin 4, count_2x2 g i j = count_2x2 g 0 0)
  (h2 : ∀ i : Fin 6, ∀ j : Fin 4, count_1x3 g i j = count_1x3 g 0 0) :
  ∀ i j : Fin 6, g.colored i j = true := by
  sorry

end all_cells_colored_l890_89001


namespace sin_pi_plus_alpha_l890_89058

theorem sin_pi_plus_alpha (α : Real) :
  (∃ (x y : Real), x = Real.sqrt 5 ∧ y = -2 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.sin (Real.pi + α) = 2/3 := by
  sorry

end sin_pi_plus_alpha_l890_89058


namespace solution_satisfies_equations_l890_89067

theorem solution_satisfies_equations :
  let x : ℚ := 599 / 204
  let y : ℚ := 65 / 136
  (7 * x - 50 * y = -3) ∧ (3 * x - 2 * y = 8) := by
  sorry

end solution_satisfies_equations_l890_89067


namespace action_figure_price_l890_89091

/-- Given the cost of sneakers, initial savings, number of action figures sold, and money left after purchase, 
    prove the price per action figure. -/
theorem action_figure_price 
  (sneaker_cost : ℕ) 
  (initial_savings : ℕ) 
  (figures_sold : ℕ) 
  (money_left : ℕ) 
  (h1 : sneaker_cost = 90)
  (h2 : initial_savings = 15)
  (h3 : figures_sold = 10)
  (h4 : money_left = 25) :
  (sneaker_cost - initial_savings + money_left) / figures_sold = 10 := by
  sorry

end action_figure_price_l890_89091


namespace vector_parallel_to_a_l890_89064

/-- Given a vector a = (-5, 4), prove that (-5k, 4k) is parallel to a for any scalar k. -/
theorem vector_parallel_to_a (k : ℝ) : 
  ∃ (t : ℝ), ((-5 : ℝ), (4 : ℝ)) = t • ((-5*k : ℝ), (4*k : ℝ)) := by
  sorry

end vector_parallel_to_a_l890_89064


namespace price_adjustment_l890_89018

theorem price_adjustment (original_price : ℝ) (original_price_positive : 0 < original_price) : 
  let reduced_price := 0.8 * original_price
  let final_price := reduced_price * 1.375
  final_price = 1.1 * original_price :=
by sorry

end price_adjustment_l890_89018


namespace quadratic_equation_roots_l890_89042

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x - 3*a = 0 ∧ x = -2) →
  (∃ y : ℝ, y^2 - a*y - 3*a = 0 ∧ y = 6) := by
sorry

end quadratic_equation_roots_l890_89042


namespace xyz_value_l890_89085

theorem xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 30 * Real.rpow 4 (1/3))
  (hxz : x * z = 45 * Real.rpow 4 (1/3))
  (hyz : y * z = 18 * Real.rpow 4 (1/3)) :
  x * y * z = 540 * Real.sqrt 3 := by
sorry

end xyz_value_l890_89085


namespace cosine_sum_17th_roots_l890_89047

theorem cosine_sum_17th_roots : 
  Real.cos (2 * Real.pi / 17) + Real.cos (6 * Real.pi / 17) + Real.cos (10 * Real.pi / 17) = (Real.sqrt 13 - 1) / 4 := by
  sorry

end cosine_sum_17th_roots_l890_89047
