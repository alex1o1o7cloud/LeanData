import Mathlib

namespace NUMINAMATH_CALUDE_triangle_inequality_l2302_230281

def is_triangle (a b c : ℝ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality (a b c : ℝ) :
  is_triangle a b c → a^2 + b^2 + c^2 + a*b*c < 8 ∧
  (∀ d : ℝ, d < 8 → ∃ a' b' c' : ℝ, is_triangle a' b' c' ∧ a' + b' + c' = 4 ∧ a'^2 + b'^2 + c'^2 + a'*b'*c' ≥ d) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2302_230281


namespace NUMINAMATH_CALUDE_average_visitors_is_276_l2302_230250

/-- Calculates the average number of visitors per day in a 30-day month starting on Sunday -/
def averageVisitorsPerDay (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let totalDays : ℕ := 30
  let sundays : ℕ := 4
  let otherDays : ℕ := totalDays - sundays
  let totalVisitors : ℕ := sundays * sundayVisitors + otherDays * otherDayVisitors
  (totalVisitors : ℚ) / totalDays

/-- Theorem: The average number of visitors per day is 276 -/
theorem average_visitors_is_276 :
  averageVisitorsPerDay 510 240 = 276 := by
  sorry


end NUMINAMATH_CALUDE_average_visitors_is_276_l2302_230250


namespace NUMINAMATH_CALUDE_number_of_valid_paths_l2302_230292

-- Define the grid dimensions
def grid_width : ℕ := 8
def grid_height : ℕ := 4

-- Define the blocked segments
def blocked_segments : List (ℕ × ℕ × ℕ × ℕ) := [(6, 2, 6, 3), (8, 2, 8, 3)]

-- Define a function to calculate valid paths
def valid_paths (width : ℕ) (height : ℕ) (blocked : List (ℕ × ℕ × ℕ × ℕ)) : ℕ :=
  sorry

-- Theorem statement
theorem number_of_valid_paths :
  valid_paths grid_width grid_height blocked_segments = 271 :=
sorry

end NUMINAMATH_CALUDE_number_of_valid_paths_l2302_230292


namespace NUMINAMATH_CALUDE_productivity_wage_relation_l2302_230239

/-- Represents the initial workday length in hours -/
def initial_workday : ℝ := 8

/-- Represents the reduced workday length in hours -/
def reduced_workday : ℝ := 7

/-- Represents the wage increase percentage -/
def wage_increase : ℝ := 5

/-- Represents the required productivity increase percentage -/
def productivity_increase : ℝ := 20

/-- Proves that a 20% productivity increase results in a 5% wage increase
    when the workday is reduced from 8 to 7 hours -/
theorem productivity_wage_relation :
  (reduced_workday / initial_workday) * (1 + productivity_increase / 100) = 1 + wage_increase / 100 :=
by sorry

end NUMINAMATH_CALUDE_productivity_wage_relation_l2302_230239


namespace NUMINAMATH_CALUDE_inverse_square_relation_l2302_230256

/-- Given that x varies inversely as the square of y, prove that x = 1 when y = 2,
    given that x = 0.1111111111111111 when y = 6. -/
theorem inverse_square_relation (x y : ℝ) (k : ℝ) (h1 : x = k / y^2) 
    (h2 : 0.1111111111111111 = k / 6^2) : 
  1 = k / 2^2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_square_relation_l2302_230256


namespace NUMINAMATH_CALUDE_revenue_is_405_main_theorem_l2302_230217

/-- Represents the rental business scenario --/
structure RentalBusiness where
  canoe_cost : ℕ
  kayak_cost : ℕ
  canoe_count : ℕ
  kayak_count : ℕ

/-- Calculates the total revenue for the rental business --/
def total_revenue (rb : RentalBusiness) : ℕ :=
  rb.canoe_cost * rb.canoe_count + rb.kayak_cost * rb.kayak_count

/-- Theorem stating that under the given conditions, the total revenue is $405 --/
theorem revenue_is_405 (rb : RentalBusiness) 
  (h1 : rb.canoe_cost = 15)
  (h2 : rb.kayak_cost = 18)
  (h3 : rb.canoe_count = (3 * rb.kayak_count) / 2)
  (h4 : rb.canoe_count = rb.kayak_count + 5) :
  total_revenue rb = 405 := by
  sorry

/-- Main theorem combining all conditions and proving the result --/
theorem main_theorem : ∃ (rb : RentalBusiness), 
  rb.canoe_cost = 15 ∧ 
  rb.kayak_cost = 18 ∧ 
  rb.canoe_count = (3 * rb.kayak_count) / 2 ∧
  rb.canoe_count = rb.kayak_count + 5 ∧
  total_revenue rb = 405 := by
  sorry

end NUMINAMATH_CALUDE_revenue_is_405_main_theorem_l2302_230217


namespace NUMINAMATH_CALUDE_blue_balloons_count_l2302_230248

def total_balloons : ℕ := 200
def red_percentage : ℚ := 35 / 100
def green_percentage : ℚ := 25 / 100
def purple_percentage : ℚ := 15 / 100

theorem blue_balloons_count :
  (total_balloons : ℚ) * (1 - (red_percentage + green_percentage + purple_percentage)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_blue_balloons_count_l2302_230248


namespace NUMINAMATH_CALUDE_problem_solution_l2302_230216

theorem problem_solution (x y : ℝ) (h1 : x^(3*y) = 8) (h2 : x = 2) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2302_230216


namespace NUMINAMATH_CALUDE_tan_double_alpha_l2302_230204

theorem tan_double_alpha (α β : Real) 
  (h1 : Real.tan (α + β) = 3) 
  (h2 : Real.tan (α - β) = 2) : 
  Real.tan (2 * α) = -1 := by
sorry

end NUMINAMATH_CALUDE_tan_double_alpha_l2302_230204


namespace NUMINAMATH_CALUDE_domain_intersection_is_closed_open_interval_l2302_230269

-- Define the domains of the two functions
def domain_sqrt (x : ℝ) : Prop := 4 - x^2 ≥ 0
def domain_ln (x : ℝ) : Prop := 4 - x > 0

-- Define the intersection of the domains
def domain_intersection (x : ℝ) : Prop := domain_sqrt x ∧ domain_ln x

-- Theorem statement
theorem domain_intersection_is_closed_open_interval :
  ∀ x, domain_intersection x ↔ x ∈ Set.Ici (-2) ∩ Set.Iio 1 :=
sorry

end NUMINAMATH_CALUDE_domain_intersection_is_closed_open_interval_l2302_230269


namespace NUMINAMATH_CALUDE_no_solution_implies_a_leq_two_l2302_230282

theorem no_solution_implies_a_leq_two (a : ℝ) : 
  (∀ x : ℝ, ¬(x > 1 ∧ x < a - 1)) → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_leq_two_l2302_230282


namespace NUMINAMATH_CALUDE_lcm_problem_l2302_230229

theorem lcm_problem (m : ℕ+) 
  (h1 : Nat.lcm 40 m = 120) 
  (h2 : Nat.lcm m 45 = 180) : 
  m = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l2302_230229


namespace NUMINAMATH_CALUDE_average_income_l2302_230267

/-- Given the average monthly incomes of pairs of individuals and the income of one individual,
    prove the average monthly income of a specific pair. -/
theorem average_income (P Q R : ℕ) : 
  (P + Q) / 2 = 5050 →
  (Q + R) / 2 = 6250 →
  P = 4000 →
  (P + R) / 2 = 5200 := by
  sorry

end NUMINAMATH_CALUDE_average_income_l2302_230267


namespace NUMINAMATH_CALUDE_exponent_division_l2302_230299

theorem exponent_division (a : ℝ) (m n : ℕ) (h : a ≠ 0) : a ^ m / a ^ n = a ^ (m - n) := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2302_230299


namespace NUMINAMATH_CALUDE_linear_decreasing_negative_slope_l2302_230243

/-- A linear function f(x) = kx + b that is monotonically decreasing on ℝ has a negative slope k. -/
theorem linear_decreasing_negative_slope (k b : ℝ) : 
  (∀ x y, x < y → (k * x + b) > (k * y + b)) → k < 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_decreasing_negative_slope_l2302_230243


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l2302_230288

/-- The quartic polynomial q(x) that satisfies given conditions -/
def q (x : ℚ) : ℚ := (1/6) * x^4 - (8/3) * x^3 - (14/3) * x^2 - (8/3) * x - 16/3

/-- Theorem stating that q(x) satisfies the given conditions -/
theorem q_satisfies_conditions : 
  q 1 = -8 ∧ q 2 = -18 ∧ q 3 = -40 ∧ q 4 = -80 ∧ q 5 = -140 := by
  sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_l2302_230288


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l2302_230215

/-- An arithmetic sequence of integers -/
def ArithmeticSequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

/-- The sequence is increasing -/
def IncreasingSequence (b : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, n < m → b n < b m

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  ArithmeticSequence b →
  IncreasingSequence b →
  b 5 * b 6 = 21 →
  b 4 * b 7 = -779 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l2302_230215


namespace NUMINAMATH_CALUDE_drawer_probability_l2302_230245

theorem drawer_probability (shirts : ℕ) (shorts : ℕ) (socks : ℕ) :
  shirts = 6 →
  shorts = 7 →
  socks = 8 →
  let total := shirts + shorts + socks
  let favorable := Nat.choose shirts 2 * Nat.choose shorts 1 * Nat.choose socks 1
  let total_outcomes := Nat.choose total 4
  (favorable : ℚ) / total_outcomes = 56 / 399 := by
  sorry

end NUMINAMATH_CALUDE_drawer_probability_l2302_230245


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2302_230247

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 64 + y^2 / 28 = 1

-- Define the asymptote of the hyperbola
def asymptote (x y : ℝ) : Prop := x - Real.sqrt 3 * y = 0

-- Define the standard form of a hyperbola
def hyperbola_standard_form (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Theorem statement
theorem hyperbola_equation (Γ : Set (ℝ × ℝ)) :
  (∃ F₁ F₂ : ℝ × ℝ, (∀ x y, ellipse x y ↔ (x - F₁.1)^2 + (y - F₁.2)^2 + (x - F₂.1)^2 + (y - F₂.2)^2 = 2 * Real.sqrt ((x - F₁.1)^2 + (y - F₁.2)^2) * Real.sqrt ((x - F₂.1)^2 + (y - F₂.2)^2)) ∧
                     (∀ x y, (x, y) ∈ Γ ↔ |(x - F₁.1)^2 + (y - F₁.2)^2 - (x - F₂.1)^2 - (y - F₂.2)^2| = 2 * Real.sqrt ((x - F₁.1)^2 + (y - F₁.2)^2) * Real.sqrt ((x - F₂.1)^2 + (y - F₂.2)^2))) →
  (∃ x y, (x, y) ∈ Γ ∧ asymptote x y) →
  ∃ x y, (x, y) ∈ Γ ↔ hyperbola_standard_form 27 9 x y :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2302_230247


namespace NUMINAMATH_CALUDE_sin_18_cos_36_equals_quarter_l2302_230251

theorem sin_18_cos_36_equals_quarter : Real.sin (18 * π / 180) * Real.cos (36 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_18_cos_36_equals_quarter_l2302_230251


namespace NUMINAMATH_CALUDE_sum_of_squares_in_ratio_l2302_230213

theorem sum_of_squares_in_ratio (x y z : ℝ) : 
  x + y + z = 9 ∧ y = 2*x ∧ z = 4*x → x^2 + y^2 + z^2 = 1701 / 49 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_in_ratio_l2302_230213


namespace NUMINAMATH_CALUDE_pyramid_volume_l2302_230278

/-- The volume of a pyramid with a regular hexagonal base and specific triangle areas -/
theorem pyramid_volume (base_area : ℝ) (triangle_ABG_area : ℝ) (triangle_DEG_area : ℝ)
  (h_base : base_area = 648)
  (h_ABG : triangle_ABG_area = 180)
  (h_DEG : triangle_DEG_area = 162) :
  ∃ (volume : ℝ), volume = 432 * Real.sqrt 22 := by
  sorry

#check pyramid_volume

end NUMINAMATH_CALUDE_pyramid_volume_l2302_230278


namespace NUMINAMATH_CALUDE_lg_expression_equals_two_l2302_230254

-- Define lg as the base-10 logarithm
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem lg_expression_equals_two :
  (lg 5)^2 + lg 2 * lg 50 = 2 := by
  sorry

end NUMINAMATH_CALUDE_lg_expression_equals_two_l2302_230254


namespace NUMINAMATH_CALUDE_white_balls_count_l2302_230255

/-- Calculates the number of white balls in a bag given specific conditions -/
theorem white_balls_count (total : ℕ) (green yellow red purple : ℕ) (prob_not_red_purple : ℚ) : 
  total = 60 ∧ 
  green = 18 ∧ 
  yellow = 5 ∧ 
  red = 6 ∧ 
  purple = 9 ∧ 
  prob_not_red_purple = 3/4 → 
  total - (green + yellow + red + purple) = 22 := by
sorry

end NUMINAMATH_CALUDE_white_balls_count_l2302_230255


namespace NUMINAMATH_CALUDE_scooter_gain_percent_l2302_230202

-- Define the purchase price, repair cost, and selling price
def purchase_price : ℚ := 900
def repair_cost : ℚ := 300
def selling_price : ℚ := 1260

-- Define the total cost
def total_cost : ℚ := purchase_price + repair_cost

-- Define the gain
def gain : ℚ := selling_price - total_cost

-- Define the gain percent
def gain_percent : ℚ := (gain / total_cost) * 100

-- Theorem to prove
theorem scooter_gain_percent : gain_percent = 5 := by
  sorry

end NUMINAMATH_CALUDE_scooter_gain_percent_l2302_230202


namespace NUMINAMATH_CALUDE_kim_coffee_time_l2302_230228

/-- Represents the time Kim spends on her morning routine -/
structure MorningRoutine where
  coffee_time : ℕ
  status_update_time_per_employee : ℕ
  payroll_update_time_per_employee : ℕ
  number_of_employees : ℕ
  total_time : ℕ

/-- Theorem stating that Kim spends 5 minutes making coffee -/
theorem kim_coffee_time (routine : MorningRoutine)
  (h1 : routine.status_update_time_per_employee = 2)
  (h2 : routine.payroll_update_time_per_employee = 3)
  (h3 : routine.number_of_employees = 9)
  (h4 : routine.total_time = 50)
  (h5 : routine.total_time = routine.coffee_time +
    routine.number_of_employees * (routine.status_update_time_per_employee +
    routine.payroll_update_time_per_employee)) :
  routine.coffee_time = 5 := by
  sorry

end NUMINAMATH_CALUDE_kim_coffee_time_l2302_230228


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_min_value_equality_l2302_230200

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) :
  ∀ z w : ℝ, z > 0 → w > 0 → 1/z + 9/w = 1 → x + 2*y ≤ z + 2*w :=
by sorry

theorem min_value_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) :
  x + 2*y = 19 + 6*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_min_value_equality_l2302_230200


namespace NUMINAMATH_CALUDE_square_roots_of_four_l2302_230289

-- Define the square root property
def is_square_root (x y : ℝ) : Prop := y ^ 2 = x

-- Theorem statement
theorem square_roots_of_four :
  ∃ (a b : ℝ), a ≠ b ∧ is_square_root 4 a ∧ is_square_root 4 b ∧
  ∀ (c : ℝ), is_square_root 4 c → (c = a ∨ c = b) :=
sorry

end NUMINAMATH_CALUDE_square_roots_of_four_l2302_230289


namespace NUMINAMATH_CALUDE_remaining_bonus_l2302_230225

def bonus : ℚ := 1496
def kitchen_fraction : ℚ := 1 / 22
def holiday_fraction : ℚ := 1 / 4
def christmas_fraction : ℚ := 1 / 8

theorem remaining_bonus : 
  bonus - (bonus * kitchen_fraction + bonus * holiday_fraction + bonus * christmas_fraction) = 867 := by
  sorry

end NUMINAMATH_CALUDE_remaining_bonus_l2302_230225


namespace NUMINAMATH_CALUDE_quadratic_set_intersection_l2302_230270

theorem quadratic_set_intersection (p q : ℝ) : 
  let A := {x : ℝ | x^2 + p*x + q = 0}
  let B := {x : ℝ | x^2 - 3*x + 2 = 0}
  (A ∩ B = A) ↔ 
  ((p^2 < 4*q) ∨ (p = -2 ∧ q = 1) ∨ (p = -4 ∧ q = 4) ∨ (p = -3 ∧ q = 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_set_intersection_l2302_230270


namespace NUMINAMATH_CALUDE_triangle_angle_C_l2302_230231

theorem triangle_angle_C (A B C : ℝ) (h_triangle : A + B + C = PI) 
  (h_eq1 : 5 * Real.sin A + 3 * Real.cos B = 8)
  (h_eq2 : 3 * Real.sin B + 5 * Real.cos A = 0) :
  C = PI / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l2302_230231


namespace NUMINAMATH_CALUDE_initial_crayons_count_l2302_230230

/-- The initial number of crayons in the drawer -/
def initial_crayons : ℕ := sorry

/-- The number of pencils in the drawer -/
def pencils : ℕ := 26

/-- The number of crayons added to the drawer -/
def added_crayons : ℕ := 12

/-- The total number of crayons after adding -/
def total_crayons : ℕ := 53

theorem initial_crayons_count : initial_crayons = 41 := by
  sorry

end NUMINAMATH_CALUDE_initial_crayons_count_l2302_230230


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l2302_230233

theorem sum_of_x_and_y (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 - y^2 = 27) : x + y = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l2302_230233


namespace NUMINAMATH_CALUDE_ellipse_k_range_l2302_230264

/-- Represents an ellipse equation in the form x^2 + ky^2 = 2 --/
structure EllipseEquation where
  k : ℝ

/-- Predicate to check if the equation represents a valid ellipse with foci on the y-axis --/
def is_valid_ellipse (e : EllipseEquation) : Prop :=
  0 < e.k ∧ e.k < 1

/-- Theorem stating the range of k for a valid ellipse with foci on the y-axis --/
theorem ellipse_k_range (e : EllipseEquation) : 
  (∃ (x y : ℝ), x^2 + e.k * y^2 = 2) ∧ 
  (∃ (c : ℝ), c ≠ 0 ∧ ∀ (x y : ℝ), x^2 + e.k * y^2 = 2 → x^2 + (y - c)^2 = x^2 + (y + c)^2) 
  ↔ is_valid_ellipse e :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l2302_230264


namespace NUMINAMATH_CALUDE_integer_solutions_of_inequalities_l2302_230240

theorem integer_solutions_of_inequalities :
  let S : Set ℤ := {x | (2 + x : ℝ) > (7 - 4*x) ∧ (x : ℝ) < ((4 + x) / 2)}
  S = {2, 3} := by sorry

end NUMINAMATH_CALUDE_integer_solutions_of_inequalities_l2302_230240


namespace NUMINAMATH_CALUDE_jack_afternoon_emails_l2302_230218

/-- The number of emails Jack received in a day -/
def total_emails : ℕ := 10

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 6

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 1

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := total_emails - morning_emails - evening_emails

theorem jack_afternoon_emails :
  afternoon_emails = 4 := by sorry

end NUMINAMATH_CALUDE_jack_afternoon_emails_l2302_230218


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l2302_230252

theorem bridge_length_calculation (train_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ) :
  train_length = 250 →
  crossing_time = 20 →
  train_speed_kmh = 77.4 →
  ∃ (bridge_length : ℝ),
    bridge_length = 180 ∧
    bridge_length = (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l2302_230252


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2302_230226

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = Real.sqrt 6 / 6) :
  let c := Real.sqrt (a^2 + b^2)
  c / a = Real.sqrt 42 / 6 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2302_230226


namespace NUMINAMATH_CALUDE_exist_distinct_indices_with_difference_not_t_l2302_230241

theorem exist_distinct_indices_with_difference_not_t 
  (n : ℕ+) (t : ℝ) (ht : t ≠ 0) (a : Fin (2*n - 1) → ℝ) :
  ∃ (s : Finset (Fin (2*n - 1))), 
    s.card = n ∧ 
    ∀ (i j : Fin n), i ≠ j → 
      ∃ (x y : Fin (2*n - 1)), x ∈ s ∧ y ∈ s ∧ a x - a y ≠ t :=
by sorry

end NUMINAMATH_CALUDE_exist_distinct_indices_with_difference_not_t_l2302_230241


namespace NUMINAMATH_CALUDE_triple_solution_l2302_230274

theorem triple_solution (a b c : ℝ) : 
  a^2 + b^2 + c^2 = 1 ∧ a * (2*b - 2*a - c) ≥ 1/2 →
  ((a = 1/Real.sqrt 6 ∧ b = 2/Real.sqrt 6 ∧ c = -1/Real.sqrt 6) ∨
   (a = -1/Real.sqrt 6 ∧ b = -2/Real.sqrt 6 ∧ c = 1/Real.sqrt 6)) :=
by sorry

end NUMINAMATH_CALUDE_triple_solution_l2302_230274


namespace NUMINAMATH_CALUDE_evaluate_expression_l2302_230236

theorem evaluate_expression : 7^3 - 4 * 7^2 + 6 * 7 - 2 = 187 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2302_230236


namespace NUMINAMATH_CALUDE_car_speed_acceleration_l2302_230219

/-- Proves that given an initial speed of 45 m/s, an acceleration of 2.5 m/s² for 10 seconds,
    the final speed will be 70 m/s and 252 km/h. -/
theorem car_speed_acceleration (initial_speed : Real) (acceleration : Real) (time : Real) :
  initial_speed = 45 ∧ acceleration = 2.5 ∧ time = 10 →
  let final_speed := initial_speed + acceleration * time
  final_speed = 70 ∧ final_speed * 3.6 = 252 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_acceleration_l2302_230219


namespace NUMINAMATH_CALUDE_max_value_problem_l2302_230237

theorem max_value_problem (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
  11 * x + 3 * y + 8 * z ≤ Real.sqrt (2292.25 / 225) :=
sorry

end NUMINAMATH_CALUDE_max_value_problem_l2302_230237


namespace NUMINAMATH_CALUDE_detergent_water_ratio_change_l2302_230205

/-- Given a solution with an initial ratio of bleach to detergent to water of 4:40:100,
    which is then altered so that the ratio of bleach to detergent is tripled,
    and the altered solution contains 300 liters of water and 60 liters of detergent,
    prove that the ratio of detergent to water changed by a factor of 5. -/
theorem detergent_water_ratio_change 
  (initial_ratio : Fin 3 → ℚ)
  (altered_ratio : Fin 3 → ℚ)
  (altered_water : ℚ)
  (altered_detergent : ℚ) :
  initial_ratio 0 = 4 →
  initial_ratio 1 = 40 →
  initial_ratio 2 = 100 →
  altered_ratio 0 = 3 * initial_ratio 0 →
  altered_ratio 1 = initial_ratio 1 →
  altered_water = 300 →
  altered_detergent = 60 →
  (initial_ratio 2 / initial_ratio 1) / (altered_water / altered_detergent) = 5 := by
  sorry

end NUMINAMATH_CALUDE_detergent_water_ratio_change_l2302_230205


namespace NUMINAMATH_CALUDE_x_sixth_minus_six_x_squared_l2302_230258

theorem x_sixth_minus_six_x_squared (x : ℝ) (h : x = 3) : x^6 - 6*x^2 = 675 := by
  sorry

end NUMINAMATH_CALUDE_x_sixth_minus_six_x_squared_l2302_230258


namespace NUMINAMATH_CALUDE_toy_value_proof_l2302_230295

theorem toy_value_proof (total_toys : ℕ) (total_worth : ℕ) (special_toy_value : ℕ) :
  total_toys = 9 →
  total_worth = 52 →
  special_toy_value = 12 →
  ∃ (other_toy_value : ℕ),
    other_toy_value * (total_toys - 1) + special_toy_value = total_worth ∧
    other_toy_value = 5 := by
  sorry

end NUMINAMATH_CALUDE_toy_value_proof_l2302_230295


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2302_230275

theorem triangle_abc_properties (A B C : Real) (h : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Given conditions
  A + B = 3 * C →
  2 * Real.sin (A - C) = Real.sin B →
  -- AB = 5 (implicitly used in the height calculation)
  -- Prove sin A and height h
  Real.sin A = 3 * (10 : Real).sqrt / 10 ∧ h = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2302_230275


namespace NUMINAMATH_CALUDE_no_triangle_from_tangent_line_l2302_230221

/-- Given a line ax + by + c = 0 (where a, b, and c are positive) tangent to the circle x^2 + y^2 = 2,
    there does not exist a triangle with side lengths a, b, and c. -/
theorem no_triangle_from_tangent_line (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
    (h_tangent : ∀ x y : ℝ, a * x + b * y + c = 0 → x^2 + y^2 = 2) :
  ¬ ∃ (A B C : ℝ × ℝ), 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = c ∧
    Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = a ∧
    Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = b :=
by sorry

end NUMINAMATH_CALUDE_no_triangle_from_tangent_line_l2302_230221


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l2302_230209

theorem sqrt_sum_inequality {a b c d : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  Real.sqrt (a / (b + c + d)) + Real.sqrt (b / (a + c + d)) +
  Real.sqrt (c / (a + b + d)) + Real.sqrt (d / (a + b + c)) > 2 ∧
  ∀ m : ℝ, (∀ a b c d : ℝ, a > 0 → b > 0 → c > 0 → d > 0 →
    Real.sqrt (a / (b + c + d)) + Real.sqrt (b / (a + c + d)) +
    Real.sqrt (c / (a + b + d)) + Real.sqrt (d / (a + b + c)) > m) →
  m ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l2302_230209


namespace NUMINAMATH_CALUDE_statement_a_statement_c_l2302_230291

-- Define the line l
def line_l (a : ℝ) (x y : ℝ) : Prop :=
  (a^2 + a + 1) * x - y + 1 = 0

-- Define perpendicularity of two lines
def perpendicular (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

-- Theorem for statement A
theorem statement_a :
  perpendicular (((-1)^2 + (-1) + 1) / (-1)) (-1) :=
by sorry

-- Theorem for statement C
theorem statement_c (a : ℝ) :
  line_l a 0 1 :=
by sorry

end NUMINAMATH_CALUDE_statement_a_statement_c_l2302_230291


namespace NUMINAMATH_CALUDE_nine_b_equals_eighteen_l2302_230277

theorem nine_b_equals_eighteen (a b : ℤ) 
  (h1 : 6 * a + 3 * b = 0) 
  (h2 : b - 3 = a) : 
  9 * b = 18 := by
sorry

end NUMINAMATH_CALUDE_nine_b_equals_eighteen_l2302_230277


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2302_230286

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {2, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2302_230286


namespace NUMINAMATH_CALUDE_shoe_box_problem_l2302_230294

theorem shoe_box_problem (pairs : ℕ) (prob : ℝ) (total : ℕ) : 
  pairs = 100 →
  prob = 0.005025125628140704 →
  (pairs : ℝ) / ((total * (total - 1)) / 2) = prob →
  total = 200 :=
sorry

end NUMINAMATH_CALUDE_shoe_box_problem_l2302_230294


namespace NUMINAMATH_CALUDE_sqrt_equation_roots_l2302_230263

theorem sqrt_equation_roots (p : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ Real.sqrt (x - p) = x ∧ Real.sqrt (y - p) = y) ↔ 0 ≤ p ∧ p < (1/4 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_sqrt_equation_roots_l2302_230263


namespace NUMINAMATH_CALUDE_rhombus_line_equations_l2302_230260

-- Define the rhombus ABCD
structure Rhombus where
  A : ℝ × ℝ
  C : ℝ × ℝ
  P : ℝ × ℝ

-- Define the rhombus with given coordinates
def given_rhombus : Rhombus := {
  A := (-4, 7)
  C := (2, -3)
  P := (3, -1)
}

-- Define a line equation
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

-- Theorem statement
theorem rhombus_line_equations (ABCD : Rhombus) 
  (h1 : ABCD = given_rhombus) :
  ∃ (line_AD line_BD : LineEquation),
    (line_AD.a = 2 ∧ line_AD.b = -1 ∧ line_AD.c = 15) ∧
    (line_BD.a = 3 ∧ line_BD.b = -5 ∧ line_BD.c = 13) := by
  sorry

end NUMINAMATH_CALUDE_rhombus_line_equations_l2302_230260


namespace NUMINAMATH_CALUDE_change_received_l2302_230244

def skirt_price : ℝ := 13
def blouse_price : ℝ := 6
def shoes_price : ℝ := 25
def handbag_price : ℝ := 35
def handbag_discount_rate : ℝ := 0.1
def coupon_discount : ℝ := 5
def amount_paid : ℝ := 150

def total_cost : ℝ := 2 * skirt_price + 3 * blouse_price + shoes_price + handbag_price

def discounted_handbag_price : ℝ := handbag_price * (1 - handbag_discount_rate)

def total_cost_after_discounts : ℝ := 
  2 * skirt_price + 3 * blouse_price + shoes_price + discounted_handbag_price - coupon_discount

theorem change_received : 
  amount_paid - total_cost_after_discounts = 54.5 := by sorry

end NUMINAMATH_CALUDE_change_received_l2302_230244


namespace NUMINAMATH_CALUDE_complex_product_real_l2302_230201

theorem complex_product_real (b : ℝ) : 
  let z₁ : ℂ := 1 + I
  let z₂ : ℂ := 2 + b * I
  (z₁ * z₂).im = 0 → b = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_l2302_230201


namespace NUMINAMATH_CALUDE_line_segment_no_intersection_l2302_230223

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line segment between two points -/
structure LineSegment where
  p1 : Point
  p2 : Point

/-- Checks if a line segment intersects both x and y axes -/
def intersectsBothAxes (l : LineSegment) : Prop :=
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
    ((l.p1.x + t * (l.p2.x - l.p1.x) = 0 ∧ l.p1.y + t * (l.p2.y - l.p1.y) ≠ 0) ∨
     (l.p1.x + t * (l.p2.x - l.p1.x) ≠ 0 ∧ l.p1.y + t * (l.p2.y - l.p1.y) = 0))

theorem line_segment_no_intersection :
  let p1 : Point := ⟨-3, 4⟩
  let p2 : Point := ⟨-5, 1⟩
  let segment : LineSegment := ⟨p1, p2⟩
  ¬(intersectsBothAxes segment) :=
by
  sorry

end NUMINAMATH_CALUDE_line_segment_no_intersection_l2302_230223


namespace NUMINAMATH_CALUDE_donut_selection_problem_l2302_230211

theorem donut_selection_problem :
  let n : ℕ := 6  -- number of donuts to select
  let k : ℕ := 4  -- number of donut types
  Nat.choose (n + k - 1) (k - 1) = 84 := by
  sorry

end NUMINAMATH_CALUDE_donut_selection_problem_l2302_230211


namespace NUMINAMATH_CALUDE_payment_difference_l2302_230224

/-- The original price of the dish -/
def original_price : Float := 24.00000000000002

/-- The discount percentage -/
def discount_percent : Float := 0.10

/-- The tip percentage -/
def tip_percent : Float := 0.15

/-- The discounted price of the dish -/
def discounted_price : Float := original_price * (1 - discount_percent)

/-- John's tip amount -/
def john_tip : Float := original_price * tip_percent

/-- Jane's tip amount -/
def jane_tip : Float := discounted_price * tip_percent

/-- John's total payment -/
def john_total : Float := discounted_price + john_tip

/-- Jane's total payment -/
def jane_total : Float := discounted_price + jane_tip

/-- Theorem stating the difference between John's and Jane's payments -/
theorem payment_difference : john_total - jane_total = 0.3600000000000003 := by
  sorry

end NUMINAMATH_CALUDE_payment_difference_l2302_230224


namespace NUMINAMATH_CALUDE_intersection_range_l2302_230227

/-- The line equation y = a(x + 2) -/
def line (a x : ℝ) : ℝ := a * (x + 2)

/-- The curve equation x^2 - y|y| = 1 -/
def curve (x y : ℝ) : Prop := x^2 - y * abs y = 1

/-- The number of intersection points between the line and the curve -/
def intersection_count (a : ℝ) : ℕ := sorry

/-- The theorem stating the range of a for exactly 2 intersection points -/
theorem intersection_range :
  ∀ a : ℝ, intersection_count a = 2 ↔ a ∈ Set.Ioo (-Real.sqrt 3 / 3) 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l2302_230227


namespace NUMINAMATH_CALUDE_decreasing_f_sufficient_not_necessary_for_increasing_g_l2302_230207

open Real

theorem decreasing_f_sufficient_not_necessary_for_increasing_g
  (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → a^x > a^y) →
  (∀ x y : ℝ, x < y → (2 - a) * x^3 < (2 - a) * y^3) ∧
  ¬(∀ x y : ℝ, x < y → (2 - a) * x^3 < (2 - a) * y^3 →
    a^x > a^y) :=
by sorry

end NUMINAMATH_CALUDE_decreasing_f_sufficient_not_necessary_for_increasing_g_l2302_230207


namespace NUMINAMATH_CALUDE_tree_height_after_four_years_l2302_230273

/-- The height of a tree after n years, given its initial height and growth rate -/
def treeHeight (initialHeight : ℝ) (growthRate : ℝ) (n : ℕ) : ℝ :=
  initialHeight * growthRate^(n - 1)

/-- Theorem stating the height of the tree after 4 years -/
theorem tree_height_after_four_years
  (h1 : treeHeight 2 2 7 = 64)
  (h2 : treeHeight 2 2 1 = 2) :
  treeHeight 2 2 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_tree_height_after_four_years_l2302_230273


namespace NUMINAMATH_CALUDE_final_highway_length_l2302_230266

def highway_extension (initial_length day1_construction day2_multiplier additional_miles : ℕ) : ℕ := 
  initial_length + day1_construction + day1_construction * day2_multiplier + additional_miles

theorem final_highway_length :
  highway_extension 200 50 3 250 = 650 := by
  sorry

end NUMINAMATH_CALUDE_final_highway_length_l2302_230266


namespace NUMINAMATH_CALUDE_blueberries_for_pint_of_jam_l2302_230235

/-- The number of pints in a quart -/
def pints_per_quart : ℕ := 2

/-- The number of quarts of jam needed for one pie -/
def quarts_per_pie : ℕ := 1

/-- The total number of blueberries needed for all pies -/
def total_blueberries : ℕ := 2400

/-- The number of pies to be made -/
def number_of_pies : ℕ := 6

/-- The number of blueberries needed for one pint of jam -/
def blueberries_per_pint : ℕ := total_blueberries / (number_of_pies * quarts_per_pie * pints_per_quart)

theorem blueberries_for_pint_of_jam :
  blueberries_per_pint = 200 :=
sorry

end NUMINAMATH_CALUDE_blueberries_for_pint_of_jam_l2302_230235


namespace NUMINAMATH_CALUDE_hundredth_term_is_397_l2302_230283

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- The 100th term of the specific arithmetic sequence -/
def hundredthTerm : ℝ := arithmeticSequenceTerm 1 4 100

theorem hundredth_term_is_397 : hundredthTerm = 397 := by sorry

end NUMINAMATH_CALUDE_hundredth_term_is_397_l2302_230283


namespace NUMINAMATH_CALUDE_jennifer_spending_l2302_230268

theorem jennifer_spending (total : ℚ) (sandwich_fraction : ℚ) (ticket_fraction : ℚ) (book_fraction : ℚ) :
  total = 120 →
  sandwich_fraction = 1 / 5 →
  ticket_fraction = 1 / 6 →
  book_fraction = 1 / 2 →
  total - (sandwich_fraction * total + ticket_fraction * total + book_fraction * total) = 16 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_spending_l2302_230268


namespace NUMINAMATH_CALUDE_delores_initial_money_l2302_230259

/-- The initial amount of money Delores had --/
def initial_amount : ℕ := sorry

/-- The cost of the computer --/
def computer_cost : ℕ := 400

/-- The cost of the printer --/
def printer_cost : ℕ := 40

/-- The amount of money left after purchases --/
def remaining_money : ℕ := 10

/-- Theorem stating that Delores' initial amount of money was $450 --/
theorem delores_initial_money : 
  initial_amount = computer_cost + printer_cost + remaining_money := by sorry

end NUMINAMATH_CALUDE_delores_initial_money_l2302_230259


namespace NUMINAMATH_CALUDE_factorization_equality_l2302_230271

theorem factorization_equality (a b : ℝ) : 3 * a^2 + 6 * a * b + 3 * b^2 = 3 * (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2302_230271


namespace NUMINAMATH_CALUDE_only_zero_satisfies_equations_l2302_230287

theorem only_zero_satisfies_equations (x y a : ℝ) 
  (eq1 : x + y = a) 
  (eq2 : x^3 + y^3 = a) 
  (eq3 : x^5 + y^5 = a) : 
  a = 0 :=
sorry

end NUMINAMATH_CALUDE_only_zero_satisfies_equations_l2302_230287


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l2302_230212

theorem largest_prime_factor_of_expression : 
  ∃ p : ℕ, Prime p ∧ p ∣ (18^4 + 3 * 18^2 + 1 - 17^4) ∧ 
  ∀ q : ℕ, Prime q → q ∣ (18^4 + 3 * 18^2 + 1 - 17^4) → q ≤ p ∧ p = 307 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l2302_230212


namespace NUMINAMATH_CALUDE_locus_of_p_l2302_230280

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the point A on the hyperbola
def point_on_hyperbola (a b x y : ℝ) : Prop := hyperbola a b x y ∧ x ≠ 0 ∧ y ≠ 0

-- Define the reflection of a point about y-axis
def reflect_y (x y : ℝ) : ℝ × ℝ := (-x, y)

-- Define the reflection of a point about x-axis
def reflect_x (x y : ℝ) : ℝ × ℝ := (x, -y)

-- Define the reflection of a point about origin
def reflect_origin (x y : ℝ) : ℝ × ℝ := (-x, -y)

-- Define perpendicularity of two lines
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Main theorem
theorem locus_of_p (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  ∀ (x y : ℝ), y ≠ 0 →
  (∃ (x0 y0 x1 y1 : ℝ),
    point_on_hyperbola a b x0 y0 ∧
    point_on_hyperbola a b x1 y1 ∧
    perpendicular (x1 - x0) (y1 - y0) (-2*x0) (-2*y0) ∧
    x = ((a^2 + b^2) / (a^2 - b^2)) * x0 ∧
    y = -((a^2 + b^2) / (a^2 - b^2)) * y0) →
  x^2 / a^2 - y^2 / b^2 = (a^2 + b^2)^2 / (a^2 - b^2)^2 :=
by sorry

end NUMINAMATH_CALUDE_locus_of_p_l2302_230280


namespace NUMINAMATH_CALUDE_min_c_plus_d_l2302_230203

theorem min_c_plus_d (a b c d : ℕ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d →
  a < b ∧ b < c ∧ c < d →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  ∃ (n : ℕ), a + b + c + d = n^2 →
  11 ≤ c + d ∧ ∃ (a' b' c' d' : ℕ), 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧
    a' < b' ∧ b' < c' ∧ c' < d' ∧
    a' ≠ b' ∧ a' ≠ c' ∧ a' ≠ d' ∧ b' ≠ c' ∧ b' ≠ d' ∧ c' ≠ d' ∧
    ∃ (m : ℕ), a' + b' + c' + d' = m^2 ∧
    c' + d' = 11 :=
by sorry

end NUMINAMATH_CALUDE_min_c_plus_d_l2302_230203


namespace NUMINAMATH_CALUDE_inequality_proof_l2302_230242

def M := {x : ℝ | -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0}

theorem inequality_proof (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (|1/3 * a + 1/6 * b| < 1/4) ∧ (|1 - 4*a*b| > 2 * |a - b|) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2302_230242


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l2302_230253

theorem smallest_n_satisfying_conditions : 
  ∃ n : ℕ, n > 10 ∧ n % 4 = 3 ∧ n % 5 = 2 ∧ 
  (∀ m : ℕ, m > 10 ∧ m % 4 = 3 ∧ m % 5 = 2 → n ≤ m) ∧ 
  n = 27 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l2302_230253


namespace NUMINAMATH_CALUDE_digit_47_is_6_l2302_230222

/-- The decimal representation of 1/17 as a list of digits -/
def decimal_rep_1_17 : List Nat := [0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1, 7, 6, 4, 7]

/-- The length of the repeating cycle in the decimal representation of 1/17 -/
def cycle_length : Nat := 16

/-- The 47th digit after the decimal point in the decimal representation of 1/17 -/
def digit_47 : Nat := decimal_rep_1_17[(47 - 1) % cycle_length]

theorem digit_47_is_6 : digit_47 = 6 := by sorry

end NUMINAMATH_CALUDE_digit_47_is_6_l2302_230222


namespace NUMINAMATH_CALUDE_sugar_delivery_problem_l2302_230276

/-- Represents the sugar delivery problem -/
def SugarDelivery (total_bags : ℕ) (total_weight : ℝ) (granulated_ratio : ℝ) (sugar_mass_ratio : ℝ) : Prop :=
  ∃ (sugar_bags : ℕ) (granulated_bags : ℕ) (sugar_weight : ℝ) (granulated_weight : ℝ),
    -- Total number of bags
    sugar_bags + granulated_bags = total_bags ∧
    -- Granulated sugar bags ratio
    granulated_bags = (1 + granulated_ratio) * sugar_bags ∧
    -- Total weight
    sugar_weight + granulated_weight = total_weight ∧
    -- Mass ratio between sugar and granulated sugar bags
    sugar_weight * granulated_bags = sugar_mass_ratio * granulated_weight * sugar_bags ∧
    -- Correct weights
    sugar_weight = 3 ∧ granulated_weight = 1.8

theorem sugar_delivery_problem :
  SugarDelivery 63 4.8 0.25 0.75 :=
sorry

end NUMINAMATH_CALUDE_sugar_delivery_problem_l2302_230276


namespace NUMINAMATH_CALUDE_coefficient_of_x_l2302_230210

/-- The coefficient of x in the simplified form of 2(x - 5) + 5(8 - 3x^2 + 6x) - 9(3x - 2) is 5 -/
theorem coefficient_of_x (x : ℝ) : 
  let expression := 2*(x - 5) + 5*(8 - 3*x^2 + 6*x) - 9*(3*x - 2)
  ∃ a b c : ℝ, expression = a*x^2 + 5*x + c := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_l2302_230210


namespace NUMINAMATH_CALUDE_equal_population_after_15_years_l2302_230261

/-- The rate of population increase in Village Y that results in equal populations after 15 years -/
def rate_of_increase_village_y (
  initial_population_x : ℕ
  ) (initial_population_y : ℕ
  ) (decrease_rate_x : ℕ
  ) (years : ℕ
  ) : ℕ :=
  (initial_population_x - decrease_rate_x * years - initial_population_y) / years

theorem equal_population_after_15_years 
  (initial_population_x : ℕ)
  (initial_population_y : ℕ)
  (decrease_rate_x : ℕ)
  (years : ℕ) :
  initial_population_x = 72000 →
  initial_population_y = 42000 →
  decrease_rate_x = 1200 →
  years = 15 →
  rate_of_increase_village_y initial_population_x initial_population_y decrease_rate_x years = 800 :=
by
  sorry

#eval rate_of_increase_village_y 72000 42000 1200 15

end NUMINAMATH_CALUDE_equal_population_after_15_years_l2302_230261


namespace NUMINAMATH_CALUDE_application_methods_count_l2302_230220

def number_of_universities : ℕ := 6
def universities_to_choose : ℕ := 3
def universities_with_conflict : ℕ := 2

theorem application_methods_count :
  (number_of_universities.choose universities_to_choose) -
  (universities_with_conflict * (number_of_universities - universities_with_conflict).choose (universities_to_choose - 1)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_application_methods_count_l2302_230220


namespace NUMINAMATH_CALUDE_finite_values_l2302_230296

def recurrence (A B : ℕ) (x : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → x (n + 1) = A * Nat.gcd (x n) (x (n - 1)) + B

theorem finite_values (A B : ℕ) (x : ℕ → ℕ) (h : recurrence A B x) :
  ∃ (S : Finset ℕ), ∀ n : ℕ, x n ∈ S :=
sorry

end NUMINAMATH_CALUDE_finite_values_l2302_230296


namespace NUMINAMATH_CALUDE_triangle_shape_l2302_230257

/-- Given a triangle ABC where BC⋅cos A = AC⋅cos B, prove that the triangle is either isosceles or right-angled -/
theorem triangle_shape (A B C : Real) (BC AC : Real) 
  (h : BC * Real.cos A = AC * Real.cos B) :
  (A = B) ∨ (A + B = Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_shape_l2302_230257


namespace NUMINAMATH_CALUDE_x0_value_l2302_230293

-- Define the function f
def f (x : ℝ) : ℝ := 13 - 8*x + x^2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := -8 + 2*x

-- Theorem statement
theorem x0_value (x₀ : ℝ) (h : f' x₀ = 4) : x₀ = 6 := by
  sorry

end NUMINAMATH_CALUDE_x0_value_l2302_230293


namespace NUMINAMATH_CALUDE_positive_real_inequalities_l2302_230249

theorem positive_real_inequalities (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x^3 - y^3 ≥ 4*x → x^2 > 2*y) ∧
  (x^5 - y^3 ≥ 2*x → x^3 ≥ 2*y) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequalities_l2302_230249


namespace NUMINAMATH_CALUDE_fraction_difference_numerator_l2302_230232

theorem fraction_difference_numerator : ∃ (p q : ℕ+), 
  (2024 : ℚ) / 2023 - (2023 : ℚ) / 2024 = (p : ℚ) / q ∧ 
  Nat.gcd p q = 1 ∧ 
  p = 4047 := by
sorry

end NUMINAMATH_CALUDE_fraction_difference_numerator_l2302_230232


namespace NUMINAMATH_CALUDE_diagonal_length_l2302_230279

structure Parallelogram :=
  (A B C D : ℝ × ℝ)
  (is_parallelogram : sorry)
  (diagonal_bisects : sorry)
  (AB_eq_CD : dist A B = dist C D)
  (BC_eq_AD : dist B C = dist A D)
  (AB_length : dist A B = 5)
  (BC_length : dist B C = 3)

/-- The length of the diagonal AC in the given parallelogram is 5√2 -/
theorem diagonal_length (p : Parallelogram) : dist p.A p.C = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_length_l2302_230279


namespace NUMINAMATH_CALUDE_certain_number_plus_two_l2302_230206

theorem certain_number_plus_two (x : ℝ) (h : x - 2 = 5) : x + 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_plus_two_l2302_230206


namespace NUMINAMATH_CALUDE_initial_amount_at_racetrack_l2302_230285

/-- Represents the sequence of bets and their outcomes at the racetrack --/
def racetrack_bets (initial_amount : ℝ) : ℝ :=
  let after_first := initial_amount * 2
  let after_second := after_first - 60
  let after_third := after_second * 2
  let after_fourth := after_third - 60
  let after_fifth := after_fourth * 2
  after_fifth - 60

/-- Theorem stating that the initial amount at the racetrack was 52.5 francs --/
theorem initial_amount_at_racetrack : 
  ∃ (x : ℝ), x > 0 ∧ racetrack_bets x = 0 ∧ x = 52.5 :=
sorry

end NUMINAMATH_CALUDE_initial_amount_at_racetrack_l2302_230285


namespace NUMINAMATH_CALUDE_playset_cost_indeterminate_l2302_230298

theorem playset_cost_indeterminate 
  (lumber_inflation : ℝ) 
  (nails_inflation : ℝ) 
  (fabric_inflation : ℝ) 
  (total_increase : ℝ) 
  (h1 : lumber_inflation = 0.20)
  (h2 : nails_inflation = 0.10)
  (h3 : fabric_inflation = 0.05)
  (h4 : total_increase = 97) :
  ∃ (L N F : ℝ), 
    L * lumber_inflation + N * nails_inflation + F * fabric_inflation = total_increase ∧
    ∃ (L' N' F' : ℝ), 
      L' ≠ L ∧
      L' * lumber_inflation + N' * nails_inflation + F' * fabric_inflation = total_increase :=
by sorry

end NUMINAMATH_CALUDE_playset_cost_indeterminate_l2302_230298


namespace NUMINAMATH_CALUDE_solve_equation_l2302_230284

theorem solve_equation (n m x : ℚ) : 
  (5 / 7 : ℚ) = n / 91 ∧ 
  (5 / 7 : ℚ) = (m + n) / 105 ∧ 
  (5 / 7 : ℚ) = (x - m) / 140 → 
  x = 110 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l2302_230284


namespace NUMINAMATH_CALUDE_money_left_after_distributions_and_donations_l2302_230234

def total_income : ℝ := 1200000

def children_share : ℝ := 0.2
def wife_share : ℝ := 0.3
def donation_rate : ℝ := 0.05
def num_children : ℕ := 3

theorem money_left_after_distributions_and_donations :
  let amount_to_children := children_share * total_income * num_children
  let amount_to_wife := wife_share * total_income
  let remaining_before_donation := total_income - (amount_to_children + amount_to_wife)
  let donation_amount := donation_rate * remaining_before_donation
  total_income - (amount_to_children + amount_to_wife + donation_amount) = 114000 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_distributions_and_donations_l2302_230234


namespace NUMINAMATH_CALUDE_distance_from_y_axis_is_18_l2302_230297

def point_P (x : ℝ) : ℝ × ℝ := (x, -9)

def distance_to_x_axis (p : ℝ × ℝ) : ℝ := |p.2|

def distance_to_y_axis (p : ℝ × ℝ) : ℝ := |p.1|

theorem distance_from_y_axis_is_18 (x : ℝ) :
  let p := point_P x
  distance_to_x_axis p = (1/2) * distance_to_y_axis p →
  distance_to_y_axis p = 18 := by sorry

end NUMINAMATH_CALUDE_distance_from_y_axis_is_18_l2302_230297


namespace NUMINAMATH_CALUDE_range_of_p_l2302_230214

-- Define the sequence a_n
def a (n : ℕ+) : ℝ := (-1 : ℝ)^(n.val - 1) * (2 * n.val - 1)

-- Define the sum S_n
def S (n : ℕ+) : ℝ := (-1 : ℝ)^(n.val - 1) * n.val

-- Theorem statement
theorem range_of_p (p : ℝ) :
  (∀ n : ℕ+, (a (n + 1) - p) * (a n - p) < 0) ↔ -3 < p ∧ p < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_p_l2302_230214


namespace NUMINAMATH_CALUDE_total_crayons_l2302_230246

/-- The number of crayons initially in the drawer -/
def initial_crayons : ℕ := 9

/-- The number of crayons Benny added to the drawer -/
def added_crayons : ℕ := 3

/-- Theorem stating that the total number of crayons after adding is 12 -/
theorem total_crayons : initial_crayons + added_crayons = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_l2302_230246


namespace NUMINAMATH_CALUDE_range_of_a_l2302_230272

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 + x - 2 > 0
def q (x a : ℝ) : Prop := x > a

-- Define what it means for q to be sufficient but not necessary for p
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, q x a → p x) ∧ ¬(∀ x, p x → q x a)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  sufficient_not_necessary a → a ∈ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2302_230272


namespace NUMINAMATH_CALUDE_tunneled_cube_surface_area_l2302_230262

/-- Represents a cube with a tunnel carved through it -/
structure TunneledCube where
  side_length : ℝ
  tunnel_distance : ℝ

/-- Calculates the total surface area of a tunneled cube -/
def total_surface_area (c : TunneledCube) : ℝ :=
  sorry

/-- Theorem stating the total surface area of the specific tunneled cube -/
theorem tunneled_cube_surface_area :
  ∃ (c : TunneledCube), c.side_length = 10 ∧ c.tunnel_distance = 3 ∧
  total_surface_area c = 600 + 73.5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tunneled_cube_surface_area_l2302_230262


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2302_230238

/-- An isosceles triangle with congruent sides of 7 cm and perimeter of 20 cm has a base of 6 cm. -/
theorem isosceles_triangle_base_length :
  ∀ (base congruent_side : ℝ),
  congruent_side = 7 →
  base + 2 * congruent_side = 20 →
  base = 6 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2302_230238


namespace NUMINAMATH_CALUDE_set_a_forms_triangle_l2302_230265

/-- Triangle Inequality Theorem: A set of three line segments can form a triangle
    if and only if the sum of the lengths of any two sides is strictly greater
    than the length of the remaining side. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Prove that the set of line segments (3, 5, 7) can form a triangle. -/
theorem set_a_forms_triangle : can_form_triangle 3 5 7 := by
  sorry

end NUMINAMATH_CALUDE_set_a_forms_triangle_l2302_230265


namespace NUMINAMATH_CALUDE_massager_vibration_rate_l2302_230208

theorem massager_vibration_rate (lowest_rate : ℝ) : 
  (∃ (highest_rate : ℝ),
    highest_rate = lowest_rate * 1.6 ∧ 
    (5 * 60) * highest_rate = 768000) →
  lowest_rate = 1600 := by
sorry

end NUMINAMATH_CALUDE_massager_vibration_rate_l2302_230208


namespace NUMINAMATH_CALUDE_fraction_subtraction_equality_l2302_230290

theorem fraction_subtraction_equality : (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_equality_l2302_230290
