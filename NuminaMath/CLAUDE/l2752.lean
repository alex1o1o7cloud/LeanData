import Mathlib

namespace NUMINAMATH_CALUDE_game_probabilities_l2752_275206

/-- Represents the game between players A and B -/
structure Game where
  oddProbA : ℝ
  evenProbB : ℝ
  maxRounds : ℕ

/-- Calculates the probability of the 4th round determining the winner and A winning -/
def probAWinsFourth (g : Game) : ℝ := sorry

/-- Calculates the mathematical expectation of the total number of rounds played -/
def expectedRounds (g : Game) : ℝ := sorry

/-- The main theorem about the game -/
theorem game_probabilities (g : Game) 
  (h1 : g.oddProbA = 2/3)
  (h2 : g.evenProbB = 2/3)
  (h3 : g.maxRounds = 8) :
  probAWinsFourth g = 10/81 ∧ expectedRounds g = 2968/729 := by sorry

end NUMINAMATH_CALUDE_game_probabilities_l2752_275206


namespace NUMINAMATH_CALUDE_always_ahead_probability_l2752_275286

/-- Represents the probability that candidate A's cumulative vote count 
    always remains ahead of candidate B's during the counting process 
    in an election where A receives n votes and B receives m votes. -/
def election_probability (n m : ℕ) : ℚ :=
  (n - m : ℚ) / (n + m : ℚ)

/-- Theorem stating the probability that candidate A's cumulative vote count 
    always remains ahead of candidate B's during the counting process. -/
theorem always_ahead_probability (n m : ℕ) (h : n > m) :
  election_probability n m = (n - m : ℚ) / (n + m : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_always_ahead_probability_l2752_275286


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2752_275275

/-- Given a hyperbola with equation x²/a² - y² = 1 where a > 0,
    and the length of its real axis is 1,
    prove that the equation of its asymptotes is y = ±2x -/
theorem hyperbola_asymptotes (a : ℝ) (h1 : a > 0) (h2 : 2 * a = 1) :
  ∃ (f : ℝ → ℝ), (∀ x, f x = 2 * x ∨ f x = -2 * x) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x y, x^2/a^2 - y^2 = 1 → x > δ → |y - f x| < ε) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2752_275275


namespace NUMINAMATH_CALUDE_monomial_coefficient_and_degree_l2752_275242

/-- Represents a monomial in two variables -/
structure Monomial (α : Type*) [Ring α] where
  coeff : α
  x_exp : ℕ
  y_exp : ℕ

/-- The degree of a monomial is the sum of its exponents -/
def Monomial.degree {α : Type*} [Ring α] (m : Monomial α) : ℕ :=
  m.x_exp + m.y_exp

theorem monomial_coefficient_and_degree :
  let m : Monomial ℤ := { coeff := -2, x_exp := 1, y_exp := 3 }
  (m.coeff = -2) ∧ (m.degree = 4) := by sorry

end NUMINAMATH_CALUDE_monomial_coefficient_and_degree_l2752_275242


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2752_275266

theorem sum_of_three_numbers (p q r M : ℚ) 
  (h1 : p + q + r = 100)
  (h2 : p + 10 = M)
  (h3 : q - 5 = M)
  (h4 : r / 5 = M) :
  M = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2752_275266


namespace NUMINAMATH_CALUDE_computer_price_increase_l2752_275260

theorem computer_price_increase (original_price : ℝ) : 
  original_price + 0.2 * original_price = 351 → 
  2 * original_price = 585 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l2752_275260


namespace NUMINAMATH_CALUDE_problem_statement_l2752_275207

theorem problem_statement (a b c d : ℝ) : 
  (a * b > 0 ∧ b * c - a * d > 0 → c / a - d / b > 0) ∧
  (a * b > 0 ∧ c / a - d / b > 0 → b * c - a * d > 0) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l2752_275207


namespace NUMINAMATH_CALUDE_jenny_easter_eggs_l2752_275244

theorem jenny_easter_eggs (red_eggs : ℕ) (orange_eggs : ℕ) (eggs_per_basket : ℕ) 
  (h1 : red_eggs = 21)
  (h2 : orange_eggs = 28)
  (h3 : eggs_per_basket ≥ 5)
  (h4 : red_eggs % eggs_per_basket = 0)
  (h5 : orange_eggs % eggs_per_basket = 0) :
  eggs_per_basket = 7 := by
sorry

end NUMINAMATH_CALUDE_jenny_easter_eggs_l2752_275244


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2752_275239

theorem smallest_integer_satisfying_inequality : 
  ∃ x : ℤ, (∀ y : ℤ, (5 : ℚ) / 8 < (y + 3 : ℚ) / 15 → x ≤ y) ∧ (5 : ℚ) / 8 < (x + 3 : ℚ) / 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2752_275239


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l2752_275262

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem derivative_f_at_one : 
  deriv f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l2752_275262


namespace NUMINAMATH_CALUDE_tonys_initial_money_is_87_l2752_275290

/-- Calculates Tony's initial amount of money -/
def tonys_initial_money (cheese_cost beef_cost beef_amount cheese_amount money_left : ℕ) : ℕ :=
  cheese_cost * cheese_amount + beef_cost * beef_amount + money_left

/-- Proves that Tony's initial amount of money was $87 -/
theorem tonys_initial_money_is_87 :
  tonys_initial_money 7 5 1 3 61 = 87 := by
  sorry

end NUMINAMATH_CALUDE_tonys_initial_money_is_87_l2752_275290


namespace NUMINAMATH_CALUDE_root_implies_range_l2752_275238

-- Define the function f(x) = ax^2 - 2ax + a - 9
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + a - 9

-- Define the property that f has at least one root in (-2, 0)
def has_root_in_interval (a : ℝ) : Prop :=
  ∃ x : ℝ, -2 < x ∧ x < 0 ∧ f a x = 0

-- Define the range of a
def a_range (a : ℝ) : Prop :=
  a < -9 ∨ (1 < a ∧ a < 9) ∨ 9 < a

-- State the theorem
theorem root_implies_range :
  ∀ a : ℝ, has_root_in_interval a → a_range a :=
sorry

end NUMINAMATH_CALUDE_root_implies_range_l2752_275238


namespace NUMINAMATH_CALUDE_rogers_app_ratio_l2752_275251

/-- Proof that Roger's app ratio is 2 given the problem conditions -/
theorem rogers_app_ratio : 
  let max_apps : ℕ := 50
  let recommended_apps : ℕ := 35
  let delete_apps : ℕ := 20
  let rogers_apps : ℕ := max_apps + delete_apps
  rogers_apps / recommended_apps = 2 := by sorry

end NUMINAMATH_CALUDE_rogers_app_ratio_l2752_275251


namespace NUMINAMATH_CALUDE_exists_sum_of_digits_div_13_l2752_275256

def sumOfDigits (n : ℕ) : ℕ := sorry

theorem exists_sum_of_digits_div_13 (n : ℕ) : 
  ∃ k ∈ Finset.range 79, (sumOfDigits (n + k)) % 13 = 0 := by sorry

end NUMINAMATH_CALUDE_exists_sum_of_digits_div_13_l2752_275256


namespace NUMINAMATH_CALUDE_marching_band_ratio_l2752_275234

theorem marching_band_ratio (total_students : ℕ) (marching_band_fraction : ℚ) 
  (brass_to_saxophone : ℚ) (saxophone_to_alto : ℚ) (alto_players : ℕ) :
  total_students = 600 →
  marching_band_fraction = 1 / 5 →
  brass_to_saxophone = 1 / 5 →
  saxophone_to_alto = 1 / 3 →
  alto_players = 4 →
  (↑alto_players / (marching_band_fraction * saxophone_to_alto * brass_to_saxophone)) / 
  (marching_band_fraction * ↑total_students) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_marching_band_ratio_l2752_275234


namespace NUMINAMATH_CALUDE_walk_distance_l2752_275294

theorem walk_distance (x y : ℝ) : 
  x > 0 → y > 0 → 
  (x^2 + y^2 - x*y = 9) → 
  x = 3 :=
by sorry

end NUMINAMATH_CALUDE_walk_distance_l2752_275294


namespace NUMINAMATH_CALUDE_negation_of_cubic_greater_than_square_l2752_275272

theorem negation_of_cubic_greater_than_square :
  (¬ ∀ x : ℕ, x^3 > x^2) ↔ (∃ x : ℕ, x^3 ≤ x^2) := by sorry

end NUMINAMATH_CALUDE_negation_of_cubic_greater_than_square_l2752_275272


namespace NUMINAMATH_CALUDE_fourth_root_16_times_cube_root_8_times_sqrt_4_eq_8_l2752_275215

theorem fourth_root_16_times_cube_root_8_times_sqrt_4_eq_8 :
  (16 : ℝ) ^ (1/4) * (8 : ℝ) ^ (1/3) * (4 : ℝ) ^ (1/2) = 8 :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_16_times_cube_root_8_times_sqrt_4_eq_8_l2752_275215


namespace NUMINAMATH_CALUDE_hyperbola_directrices_distance_l2752_275280

/-- Given a hyperbola with foci at (±√26, 0) and asymptotes y = ±(3/2)x,
    prove that the distance between its two directrices is (8√26)/13 -/
theorem hyperbola_directrices_distance (a b c : ℝ) : 
  (c = Real.sqrt 26) →                  -- focus distance
  (b / a = 3 / 2) →                     -- asymptote slope
  (a^2 + b^2 = 26) →                    -- relation between a, b, and c
  (2 * (a^2 / c)) = (8 * Real.sqrt 26) / 13 := by
  sorry

#check hyperbola_directrices_distance

end NUMINAMATH_CALUDE_hyperbola_directrices_distance_l2752_275280


namespace NUMINAMATH_CALUDE_expression_simplification_l2752_275269

theorem expression_simplification (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (sum_eq_one : a + b + c = 1) :
  1 / (b^2 + c^2 - a^2) + 1 / (a^2 + c^2 - b^2) + 1 / (a^2 + b^2 - c^2) = 
  3 / (2 * (-b - c + b * c)) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2752_275269


namespace NUMINAMATH_CALUDE_valid_assignments_count_l2752_275295

/-- Represents the set of mascots -/
inductive Mascot
| AXiang
| AHe
| ARu
| AYi
| LeYangyang

/-- Represents the set of volunteers -/
inductive Volunteer
| A
| B
| C
| D
| E

/-- A function that assigns mascots to volunteers -/
def Assignment := Volunteer → Mascot

/-- Predicate that checks if an assignment satisfies the given conditions -/
def ValidAssignment (f : Assignment) : Prop :=
  (f Volunteer.A = Mascot.AXiang ∨ f Volunteer.B = Mascot.AXiang) ∧
  f Volunteer.C ≠ Mascot.LeYangyang

/-- The number of valid assignments -/
def NumValidAssignments : ℕ := sorry

/-- Theorem stating that the number of valid assignments is 36 -/
theorem valid_assignments_count : NumValidAssignments = 36 := by sorry

end NUMINAMATH_CALUDE_valid_assignments_count_l2752_275295


namespace NUMINAMATH_CALUDE_rugby_team_size_l2752_275287

theorem rugby_team_size (initial_avg : ℝ) (new_player_weight : ℝ) (new_avg : ℝ) :
  initial_avg = 180 →
  new_player_weight = 210 →
  new_avg = 181.42857142857142 →
  ∃ n : ℕ, (n : ℝ) * initial_avg + new_player_weight = (n + 1 : ℝ) * new_avg ∧ n = 20 :=
by sorry

end NUMINAMATH_CALUDE_rugby_team_size_l2752_275287


namespace NUMINAMATH_CALUDE_root_implies_b_eq_neg_20_l2752_275205

-- Define the polynomial
def f (a b : ℚ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 16

-- State the theorem
theorem root_implies_b_eq_neg_20 (a b : ℚ) :
  f a b (Real.sqrt 5 + 3) = 0 → b = -20 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_b_eq_neg_20_l2752_275205


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2752_275240

theorem inequality_equivalence (x y : ℝ) (h : x ≠ 0) :
  (y - x)^2 < x^2 ↔ y > 0 ∧ y < 2*x := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2752_275240


namespace NUMINAMATH_CALUDE_houses_with_neither_feature_l2752_275282

theorem houses_with_neither_feature (total : ℕ) (garage : ℕ) (pool : ℕ) (both : ℕ) :
  total = 70 →
  garage = 50 →
  pool = 40 →
  both = 35 →
  total - (garage + pool - both) = 15 :=
by sorry

end NUMINAMATH_CALUDE_houses_with_neither_feature_l2752_275282


namespace NUMINAMATH_CALUDE_quadratic_function_properties_quadratic_function_max_value_l2752_275210

/-- A quadratic function f(x) = ax^2 + bx + c where the solution set of f(x) > -2x is {x | 1 < x < 3} -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_properties (a b c : ℝ) 
  (h_solution_set : ∀ x, 1 < x ∧ x < 3 ↔ QuadraticFunction a b c x > -2 * x) :
  (∃ x₀ > 0, QuadraticFunction a b c x₀ = 2 * a ∧ 
   ∀ x, QuadraticFunction a b c x = 2 * a → x = x₀) →
  QuadraticFunction a b c = fun x ↦ -x^2 + 2*x - 3 :=
sorry

theorem quadratic_function_max_value (a b c : ℝ) 
  (h_solution_set : ∀ x, 1 < x ∧ x < 3 ↔ QuadraticFunction a b c x > -2 * x) :
  (∃ x₀, ∀ x, QuadraticFunction a b c x ≤ QuadraticFunction a b c x₀ ∧ 
   QuadraticFunction a b c x₀ > 0) →
  (-2 - Real.sqrt 3 < a ∧ a < 0) ∨ (-2 + Real.sqrt 3 < a ∧ a < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_quadratic_function_max_value_l2752_275210


namespace NUMINAMATH_CALUDE_solution_set_of_inequalities_l2752_275223

theorem solution_set_of_inequalities :
  let S := {x : ℝ | |x| - 1 < 0 ∧ x^2 - 3*x < 0}
  S = {x : ℝ | 0 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequalities_l2752_275223


namespace NUMINAMATH_CALUDE_speed_train_B_is_25_l2752_275222

/-- Represents the distance between two stations in kilometers -/
def distance_between_stations : ℝ := 155

/-- Represents the speed of the train from station A in km/h -/
def speed_train_A : ℝ := 20

/-- Represents the time difference between the starts of the two trains in hours -/
def time_difference : ℝ := 1

/-- Represents the total time until the trains meet in hours -/
def total_time : ℝ := 4

/-- Represents the time the train from B travels in hours -/
def time_train_B : ℝ := 3

/-- Theorem stating that the speed of the train from station B is 25 km/h -/
theorem speed_train_B_is_25 : 
  ∃ (speed_B : ℝ), 
    speed_B * time_train_B = distance_between_stations - speed_train_A * total_time ∧ 
    speed_B = 25 := by
  sorry

end NUMINAMATH_CALUDE_speed_train_B_is_25_l2752_275222


namespace NUMINAMATH_CALUDE_opposite_of_negative_nine_l2752_275221

theorem opposite_of_negative_nine :
  ∃ (x : ℤ), (x + (-9) = 0) ∧ (x = 9) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_nine_l2752_275221


namespace NUMINAMATH_CALUDE_perfect_fit_implies_r_squared_one_l2752_275265

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

end NUMINAMATH_CALUDE_perfect_fit_implies_r_squared_one_l2752_275265


namespace NUMINAMATH_CALUDE_quadrilateral_area_rational_l2752_275248

/-- The area of a quadrilateral with integer coordinates is rational -/
theorem quadrilateral_area_rational
  (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℤ) :
  ∃ (q : ℚ), q = |x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂)| / 2 +
              |x₁ * (y₃ - y₄) + x₃ * (y₄ - y₁) + x₄ * (y₁ - y₃)| / 2 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_rational_l2752_275248


namespace NUMINAMATH_CALUDE_david_window_washing_time_l2752_275216

/-- Represents the time taken to wash windows -/
def wash_time (windows_per_unit : ℕ) (minutes_per_unit : ℕ) (total_windows : ℕ) : ℕ :=
  (total_windows / windows_per_unit) * minutes_per_unit

/-- Proves that it takes David 160 minutes to wash all windows in his house -/
theorem david_window_washing_time :
  wash_time 4 10 64 = 160 := by
  sorry

#eval wash_time 4 10 64

end NUMINAMATH_CALUDE_david_window_washing_time_l2752_275216


namespace NUMINAMATH_CALUDE_trajectory_is_line_segment_l2752_275212

/-- The trajectory of a point P satisfying |PF₁| + |PF₂| = 8, where F₁ and F₂ are fixed points -/
theorem trajectory_is_line_segment (F₁ F₂ P : ℝ × ℝ) : 
  F₁ = (-4, 0) → 
  F₂ = (4, 0) → 
  dist P F₁ + dist P F₂ = 8 →
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • F₁ + t • F₂ :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_line_segment_l2752_275212


namespace NUMINAMATH_CALUDE_max_n_sin_cos_inequality_l2752_275233

theorem max_n_sin_cos_inequality : 
  (∃ (n : ℕ), n > 0 ∧ ∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 1 / n) ∧ 
  (∀ (m : ℕ), m > 8 → ∃ (x : ℝ), (Real.sin x)^m + (Real.cos x)^m < 1 / m) :=
by sorry

end NUMINAMATH_CALUDE_max_n_sin_cos_inequality_l2752_275233


namespace NUMINAMATH_CALUDE_triangle_area_l2752_275213

theorem triangle_area (base height : ℝ) (h1 : base = 6) (h2 : height = 8) :
  (1 / 2) * base * height = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2752_275213


namespace NUMINAMATH_CALUDE_probability_8_of_hearts_or_spade_l2752_275291

def standard_deck : ℕ := 52

def probability_8_of_hearts : ℚ := 1 / standard_deck

def probability_spade : ℚ := 1 / 4

theorem probability_8_of_hearts_or_spade :
  probability_8_of_hearts + probability_spade = 7 / 26 := by
  sorry

end NUMINAMATH_CALUDE_probability_8_of_hearts_or_spade_l2752_275291


namespace NUMINAMATH_CALUDE_expression_evaluation_l2752_275250

theorem expression_evaluation :
  (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2752_275250


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2752_275228

theorem complex_equation_solution (z : ℂ) :
  z + (1 + 2*I) = 10 - 3*I → z = 9 - 5*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2752_275228


namespace NUMINAMATH_CALUDE_fraction_of_trunks_l2752_275243

/-- Given that 38% of garments are bikinis and 63% are either bikinis or trunks,
    prove that 25% of garments are trunks. -/
theorem fraction_of_trunks
  (bikinis : Real)
  (bikinis_or_trunks : Real)
  (h1 : bikinis = 0.38)
  (h2 : bikinis_or_trunks = 0.63) :
  bikinis_or_trunks - bikinis = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_trunks_l2752_275243


namespace NUMINAMATH_CALUDE_sum_of_large_numbers_l2752_275245

theorem sum_of_large_numbers : 800000000000 + 299999999999 = 1099999999999 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_large_numbers_l2752_275245


namespace NUMINAMATH_CALUDE_triangle_perimeter_bound_l2752_275225

theorem triangle_perimeter_bound : ∀ s : ℝ,
  s > 0 →
  s + 7 > 21 →
  s + 21 > 7 →
  7 + 21 > s →
  (∃ n : ℕ, n = 57 ∧ ∀ m : ℕ, m > (s + 7 + 21) → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_bound_l2752_275225


namespace NUMINAMATH_CALUDE_gcd_problem_l2752_275297

theorem gcd_problem : ∃! n : ℕ, 70 ≤ n ∧ n ≤ 80 ∧ Nat.gcd 35 n = 7 ∧ n = 77 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l2752_275297


namespace NUMINAMATH_CALUDE_total_wait_days_l2752_275273

/-- The number of days Mark waits for his first vaccine appointment -/
def first_appointment_wait : ℕ := 4

/-- The number of days Mark waits for his second vaccine appointment -/
def second_appointment_wait : ℕ := 20

/-- The number of weeks Mark waits for the vaccine to be fully effective -/
def full_effectiveness_wait_weeks : ℕ := 2

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- Theorem: The total number of days Mark waits is 38 -/
theorem total_wait_days : 
  first_appointment_wait + second_appointment_wait + (full_effectiveness_wait_weeks * days_per_week) = 38 := by
  sorry

end NUMINAMATH_CALUDE_total_wait_days_l2752_275273


namespace NUMINAMATH_CALUDE_monomial_condition_and_expression_evaluation_l2752_275253

/-- Given that -2a^2 * b^(y+3) and 4a^x * b^2 form a monomial when added together,
    prove that x = 2 and y = -1, and that under these conditions,
    2(x^2*y - 3*y^3 + 2*x) - 3(x + x^2*y - 2*y^3) - x = 4 -/
theorem monomial_condition_and_expression_evaluation 
  (a b : ℝ) (x y : ℤ) 
  (h : ∃ k, -2 * a^2 * b^(y+3) + 4 * a^x * b^2 = k * a^2 * b^2) :
  x = 2 ∧ y = -1 ∧ 
  2 * (x^2 * y - 3 * y^3 + 2 * x) - 3 * (x + x^2 * y - 2 * y^3) - x = 4 :=
by sorry

end NUMINAMATH_CALUDE_monomial_condition_and_expression_evaluation_l2752_275253


namespace NUMINAMATH_CALUDE_work_completion_time_l2752_275231

/-- 
Given a group of people who can complete a task in 12 days, 
prove that twice that number of people can complete half the task in 3 days.
-/
theorem work_completion_time 
  (people : ℕ) 
  (work : ℝ) 
  (h : people > 0) 
  (complete_time : ℝ → ℝ → ℝ → ℝ) 
  (h_complete : complete_time people work 12 = 1) :
  complete_time (2 * people) (work / 2) 3 = 1 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l2752_275231


namespace NUMINAMATH_CALUDE_nancy_vacation_pictures_l2752_275217

/-- The number of pictures Nancy took at the zoo -/
def zoo_pictures : ℕ := 49

/-- The number of pictures Nancy took at the museum -/
def museum_pictures : ℕ := 8

/-- The number of pictures Nancy deleted -/
def deleted_pictures : ℕ := 38

/-- The total number of pictures Nancy took during her vacation -/
def total_pictures : ℕ := zoo_pictures + museum_pictures

/-- The number of pictures Nancy has after deleting some -/
def remaining_pictures : ℕ := total_pictures - deleted_pictures

theorem nancy_vacation_pictures : remaining_pictures = 19 := by
  sorry

end NUMINAMATH_CALUDE_nancy_vacation_pictures_l2752_275217


namespace NUMINAMATH_CALUDE_harry_buckets_per_round_l2752_275261

theorem harry_buckets_per_round 
  (george_buckets : ℕ) 
  (total_buckets : ℕ) 
  (total_rounds : ℕ) 
  (h1 : george_buckets = 2)
  (h2 : total_buckets = 110)
  (h3 : total_rounds = 22) :
  (total_buckets - george_buckets * total_rounds) / total_rounds = 3 := by
  sorry

end NUMINAMATH_CALUDE_harry_buckets_per_round_l2752_275261


namespace NUMINAMATH_CALUDE_three_planes_division_l2752_275288

/-- A type representing the possible configurations of three non-coincident planes in space -/
inductive PlaneConfiguration
  | AllParallel
  | TwoParallelOneIntersecting
  | IntersectAlongLine
  | IntersectPairwiseParallelLines
  | IntersectAtPoint

/-- The number of parts that space is divided into by three non-coincident planes -/
def numParts (config : PlaneConfiguration) : ℕ :=
  match config with
  | .AllParallel => 4
  | .TwoParallelOneIntersecting => 6
  | .IntersectAlongLine => 6
  | .IntersectPairwiseParallelLines => 7
  | .IntersectAtPoint => 8

/-- Theorem stating that the number of parts is always 4, 6, 7, or 8 -/
theorem three_planes_division (config : PlaneConfiguration) :
  ∃ n : ℕ, (n = 4 ∨ n = 6 ∨ n = 7 ∨ n = 8) ∧ numParts config = n :=
sorry

end NUMINAMATH_CALUDE_three_planes_division_l2752_275288


namespace NUMINAMATH_CALUDE_area_is_100_l2752_275218

/-- The area enclosed by the graph of |x| + |2y| = 10 -/
def area_enclosed : ℝ := 100

/-- The equation of the graph -/
def graph_equation (x y : ℝ) : Prop := abs x + abs (2 * y) = 10

/-- The graph is symmetric across the x-axis -/
axiom symmetric_x_axis : ∀ x y : ℝ, graph_equation x y → graph_equation x (-y)

/-- The graph is symmetric across the y-axis -/
axiom symmetric_y_axis : ∀ x y : ℝ, graph_equation x y → graph_equation (-x) y

/-- The graph forms four congruent triangles -/
axiom four_congruent_triangles : ∃ A : ℝ, A > 0 ∧ area_enclosed = 4 * A

/-- Theorem: The area enclosed by the graph of |x| + |2y| = 10 is 100 square units -/
theorem area_is_100 : area_enclosed = 100 := by sorry

end NUMINAMATH_CALUDE_area_is_100_l2752_275218


namespace NUMINAMATH_CALUDE_quadratic_root_form_l2752_275277

theorem quadratic_root_form (d : ℝ) : 
  (∀ x : ℝ, x^2 + 6*x + d = 0 ↔ x = (-6 + Real.sqrt d) / 2 ∨ x = (-6 - Real.sqrt d) / 2) →
  d = 36 / 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_form_l2752_275277


namespace NUMINAMATH_CALUDE_multiplicative_inverse_289_mod_391_l2752_275258

theorem multiplicative_inverse_289_mod_391 
  (h : 136^2 + 255^2 = 289^2) : 
  (289 * 18) % 391 = 1 ∧ 0 ≤ 18 ∧ 18 < 391 := by
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_289_mod_391_l2752_275258


namespace NUMINAMATH_CALUDE_symmetric_even_function_value_l2752_275201

/-- A function that is symmetric about x=2 -/
def SymmetricAbout2 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (4 - x) = f x

/-- An even function -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

theorem symmetric_even_function_value (f : ℝ → ℝ) 
  (h_sym : SymmetricAbout2 f) (h_even : EvenFunction f) (h_val : f 3 = 3) : 
  f (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_even_function_value_l2752_275201


namespace NUMINAMATH_CALUDE_park_trees_count_l2752_275254

theorem park_trees_count : ∃! n : ℕ, 
  80 < n ∧ n < 150 ∧ 
  n % 4 = 2 ∧ 
  n % 5 = 3 ∧ 
  n % 6 = 4 ∧ 
  n = 98 := by sorry

end NUMINAMATH_CALUDE_park_trees_count_l2752_275254


namespace NUMINAMATH_CALUDE_negation_equivalence_l2752_275285

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, (x₀^2 + 1 > 0 ∨ x₀ > Real.sin x₀)) ↔ 
  (∀ x : ℝ, (x^2 + 1 ≤ 0 ∧ x ≤ Real.sin x)) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2752_275285


namespace NUMINAMATH_CALUDE_second_number_proof_l2752_275211

theorem second_number_proof : ∃ x : ℕ, 
  (1657 % 1 = 10) ∧ 
  (x % 1 = 7) ∧ 
  (∀ y : ℕ, y > x → ¬(y % 1 = 7)) ∧ 
  (x = 1655) := by
sorry

end NUMINAMATH_CALUDE_second_number_proof_l2752_275211


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2752_275296

theorem rationalize_denominator :
  ∃ (a b : ℝ), a + b * Real.sqrt 3 = -Real.sqrt 3 - 2 ∧
  (a + b * Real.sqrt 3) * (Real.sqrt 3 - 2) = 1 := by
sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2752_275296


namespace NUMINAMATH_CALUDE_divisibility_of_powers_l2752_275219

theorem divisibility_of_powers (a : ℕ) (ha : a > 0) :
  ∃ b : ℕ, b > a ∧ (1 + 2^b + 3^b) % (1 + 2^a + 3^a) = 0 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_powers_l2752_275219


namespace NUMINAMATH_CALUDE_two_roots_sum_greater_than_2a_l2752_275229

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x - 2

theorem two_roots_sum_greater_than_2a (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 → x₁ < x₂ → f a x₁ = 0 → f a x₂ = 0 → x₁ + x₂ > 2 * a := by
  sorry

end NUMINAMATH_CALUDE_two_roots_sum_greater_than_2a_l2752_275229


namespace NUMINAMATH_CALUDE_investment_rate_proof_l2752_275279

/-- Proves that the unknown investment rate is 0.18 given the problem conditions --/
theorem investment_rate_proof (total_investment : ℝ) (known_rate : ℝ) (total_interest : ℝ) (unknown_rate_investment : ℝ) 
  (h1 : total_investment = 22000)
  (h2 : known_rate = 0.14)
  (h3 : total_interest = 3360)
  (h4 : unknown_rate_investment = 7000)
  (h5 : unknown_rate_investment * r + (total_investment - unknown_rate_investment) * known_rate = total_interest) :
  r = 0.18 := by
  sorry

end NUMINAMATH_CALUDE_investment_rate_proof_l2752_275279


namespace NUMINAMATH_CALUDE_range_of_m_l2752_275247

theorem range_of_m (m : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + 2*x + m ≤ 0) ↔ m > 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l2752_275247


namespace NUMINAMATH_CALUDE_quadratic_function_value_l2752_275236

theorem quadratic_function_value (a b x₁ x₂ : ℝ) : 
  a ≠ 0 → 
  (∃ y₁ y₂ : ℝ, y₁ = a * x₁^2 + b * x₁ + 2009 ∧ 
                y₂ = a * x₂^2 + b * x₂ + 2009 ∧ 
                y₁ = 2012 ∧ 
                y₂ = 2012) → 
  a * (x₁ + x₂)^2 + b * (x₁ + x₂) + 2009 = 2009 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l2752_275236


namespace NUMINAMATH_CALUDE_not_p_or_q_l2752_275289

-- Define proposition p
def p : Prop := ∃ x : ℝ, Real.sin x > 1

-- Define proposition q
def q : Prop := ∀ x : ℝ, Real.exp (-x) < 0

-- Theorem to prove
theorem not_p_or_q : ¬(p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_not_p_or_q_l2752_275289


namespace NUMINAMATH_CALUDE_fraction_doubles_l2752_275235

theorem fraction_doubles (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (2*x)*(2*y) / ((2*x) + (2*y)) = 2 * (x*y / (x + y)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_doubles_l2752_275235


namespace NUMINAMATH_CALUDE_inscribed_hexagon_radius_theorem_l2752_275298

/-- A hexagon inscribed in a circle with radius R, where three consecutive sides are equal to a
    and the other three consecutive sides are equal to b. -/
structure InscribedHexagon (R a b : ℝ) : Prop where
  radius_positive : R > 0
  side_a_positive : a > 0
  side_b_positive : b > 0
  three_sides_a : ∃ (AB BC CD : ℝ), AB = a ∧ BC = a ∧ CD = a
  three_sides_b : ∃ (DE EF FA : ℝ), DE = b ∧ EF = b ∧ FA = b

/-- The theorem stating the relationship between the radius R and sides a and b of the inscribed hexagon. -/
theorem inscribed_hexagon_radius_theorem (R a b : ℝ) (h : InscribedHexagon R a b) :
  R^2 = (a^2 + b^2 + a*b) / 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_radius_theorem_l2752_275298


namespace NUMINAMATH_CALUDE_cubic_difference_division_l2752_275220

theorem cubic_difference_division (a b : ℝ) (ha : a = 6) (hb : b = 3) :
  (a^3 - b^3) / (a^2 + a*b + b^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_difference_division_l2752_275220


namespace NUMINAMATH_CALUDE_total_pages_calculation_l2752_275293

/-- The number of pages in each booklet -/
def pages_per_booklet : ℕ := 12

/-- The number of booklets in the short story section -/
def number_of_booklets : ℕ := 75

/-- The total number of pages in all booklets -/
def total_pages : ℕ := pages_per_booklet * number_of_booklets

theorem total_pages_calculation : total_pages = 900 := by
  sorry

end NUMINAMATH_CALUDE_total_pages_calculation_l2752_275293


namespace NUMINAMATH_CALUDE_lee_apple_harvest_l2752_275274

/-- The number of baskets Mr. Lee used to pack apples -/
def num_baskets : ℕ := 19

/-- The number of apples in each basket -/
def apples_per_basket : ℕ := 25

/-- The total number of apples harvested by Mr. Lee -/
def total_apples : ℕ := num_baskets * apples_per_basket

theorem lee_apple_harvest : total_apples = 475 := by
  sorry

end NUMINAMATH_CALUDE_lee_apple_harvest_l2752_275274


namespace NUMINAMATH_CALUDE_original_number_l2752_275226

theorem original_number : ∃ x : ℕ, x - (x / 3) = 36 ∧ x = 54 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l2752_275226


namespace NUMINAMATH_CALUDE_system_equivalence_l2752_275246

theorem system_equivalence (x y a b : ℝ) : 
  (2 * x + y = 5 ∧ a * x + 3 * y = -1) ∧
  (x - y = 1 ∧ 4 * x + b * y = 11) →
  a = -2 ∧ b = 3 := by
sorry

end NUMINAMATH_CALUDE_system_equivalence_l2752_275246


namespace NUMINAMATH_CALUDE_arrangements_count_is_24_l2752_275249

/-- The number of ways to arrange 5 people in a row with specific adjacency constraints -/
def arrangements_count : ℕ :=
  let total_people : ℕ := 5
  let adjacent_pair : ℕ := 2  -- A and B
  let non_adjacent : ℕ := 1   -- C
  (adjacent_pair.choose 1) * (adjacent_pair.factorial) * ((total_people - adjacent_pair).factorial)

/-- Theorem stating that the number of arrangements is 24 -/
theorem arrangements_count_is_24 : arrangements_count = 24 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_is_24_l2752_275249


namespace NUMINAMATH_CALUDE_equation_solution_l2752_275257

theorem equation_solution : ∃ x : ℚ, (3/4 : ℚ) + 1/x = (7/8 : ℚ) ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2752_275257


namespace NUMINAMATH_CALUDE_magical_stack_with_201_fixed_l2752_275278

/-- Definition of a magical stack of cards -/
def is_magical_stack (n : ℕ) : Prop :=
  ∃ (card_from_A card_from_B : ℕ), 
    card_from_A ≤ n ∧ 
    card_from_B > n ∧ 
    card_from_B ≤ 2*n ∧
    (card_from_A = 2 * ((card_from_A + 1) / 2) - 1 ∨
     card_from_B = 2 * (card_from_B / 2))

/-- Theorem stating the number of cards in a magical stack where card 201 retains its position -/
theorem magical_stack_with_201_fixed :
  ∃ (n : ℕ), 
    is_magical_stack n ∧ 
    201 ≤ n ∧
    201 = 2 * ((201 + 1) / 2) - 1 ∧
    n = 201 ∧
    2 * n = 402 := by
  sorry

end NUMINAMATH_CALUDE_magical_stack_with_201_fixed_l2752_275278


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2752_275227

theorem max_value_of_expression (x y z w : ℝ) (h : x + y + z + w = 1) :
  ∃ (M : ℝ), M = x*w + 2*y*w + 3*x*y + 3*z*w + 4*x*z + 5*y*z ∧
  M ≤ (3/2 : ℝ) ∧
  ∃ (x₀ y₀ z₀ w₀ : ℝ), x₀ + y₀ + z₀ + w₀ = 1 ∧
    (3/2 : ℝ) = x₀*w₀ + 2*y₀*w₀ + 3*x₀*y₀ + 3*z₀*w₀ + 4*x₀*z₀ + 5*y₀*z₀ :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2752_275227


namespace NUMINAMATH_CALUDE_solution_difference_l2752_275264

theorem solution_difference (r s : ℝ) : 
  ((r - 5) * (r + 5) = 24 * r - 120) →
  ((s - 5) * (s + 5) = 24 * s - 120) →
  r ≠ s →
  r > s →
  r - s = 14 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l2752_275264


namespace NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_18_l2752_275255

theorem factorization_of_2m_squared_minus_18 (m : ℝ) : 2 * m^2 - 18 = 2 * (m + 3) * (m - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_18_l2752_275255


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2752_275241

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ b.1 = k * a.1 ∧ b.2 = k * a.2

/-- Given vectors a = (2, 3) and b = (x, 6), if they are parallel, then x = 4 -/
theorem parallel_vectors_x_value :
  ∀ (x : ℝ),
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (x, 6)
  are_parallel a b → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2752_275241


namespace NUMINAMATH_CALUDE_max_value_theorem_l2752_275283

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 1) :
  2*a*b + 2*b*c*Real.sqrt 2 ≤ Real.sqrt 3 ∧ 
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ 
    a'^2 + b'^2 + c'^2 = 1 ∧ 
    2*a'*b' + 2*b'*c'*Real.sqrt 2 = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2752_275283


namespace NUMINAMATH_CALUDE_equality_condition_l2752_275237

theorem equality_condition (p q r : ℝ) : p + q * r = (p + q) * (p + r) ↔ p + q + r = 0 := by
  sorry

end NUMINAMATH_CALUDE_equality_condition_l2752_275237


namespace NUMINAMATH_CALUDE_adult_ticket_price_l2752_275268

-- Define the variables and constants
def adult_price : ℝ := sorry
def child_price : ℝ := 3.50
def total_tickets : ℕ := 21
def total_revenue : ℝ := 83.50
def adult_tickets : ℕ := 5

-- Theorem statement
theorem adult_ticket_price :
  adult_price = 5.50 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_adult_ticket_price_l2752_275268


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l2752_275224

/-- A random variable following a normal distribution with mean 1 and standard deviation σ > 0 -/
def ξ (σ : ℝ) : Type := Real

/-- The probability density function of ξ -/
noncomputable def pdf (σ : ℝ) (x : ℝ) : ℝ := sorry

/-- The probability that ξ falls within an interval (a, b) -/
noncomputable def prob (σ : ℝ) (a b : ℝ) : ℝ := sorry

theorem normal_distribution_probability 
  (σ : ℝ) 
  (h_σ_pos : σ > 0) 
  (h_prob : prob σ 0 1 = 0.4) : 
  prob σ 0 2 = 0.8 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l2752_275224


namespace NUMINAMATH_CALUDE_sandwich_combinations_l2752_275281

theorem sandwich_combinations (n_meat : ℕ) (n_cheese : ℕ) : n_meat = 8 → n_cheese = 7 →
  (n_meat.choose 2) * n_cheese = 196 := by sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l2752_275281


namespace NUMINAMATH_CALUDE_leah_peeled_18_l2752_275259

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  total_potatoes : ℕ
  homer_rate : ℕ
  leah_rate : ℕ
  homer_solo_time : ℕ

/-- Calculates the number of potatoes Leah peeled -/
def leah_potatoes (scenario : PotatoPeeling) : ℕ :=
  let potatoes_left := scenario.total_potatoes - scenario.homer_rate * scenario.homer_solo_time
  let combined_rate := scenario.homer_rate + scenario.leah_rate
  let combined_time := potatoes_left / combined_rate
  scenario.leah_rate * combined_time

/-- The theorem stating that Leah peeled 18 potatoes -/
theorem leah_peeled_18 (scenario : PotatoPeeling) 
  (h1 : scenario.total_potatoes = 50)
  (h2 : scenario.homer_rate = 3)
  (h3 : scenario.leah_rate = 4)
  (h4 : scenario.homer_solo_time = 6) :
  leah_potatoes scenario = 18 := by
  sorry

end NUMINAMATH_CALUDE_leah_peeled_18_l2752_275259


namespace NUMINAMATH_CALUDE_slope_of_line_l2752_275203

theorem slope_of_line (x y : ℝ) : 
  x + 3 * y + 3 = 0 → (
    let slope := -(1 : ℝ) / 3
    ∀ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ → 
    x₁ + 3 * y₁ + 3 = 0 → 
    x₂ + 3 * y₂ + 3 = 0 → 
    (y₂ - y₁) / (x₂ - x₁) = slope
  ) := by sorry

end NUMINAMATH_CALUDE_slope_of_line_l2752_275203


namespace NUMINAMATH_CALUDE_no_intersection_for_given_scenarios_l2752_275276

/-- Determines if two circles intersect based on their radii and the distance between their centers -/
def circlesIntersect (r1 r2 d : ℝ) : Prop :=
  |r1 - r2| ≤ d ∧ d ≤ r1 + r2

theorem no_intersection_for_given_scenarios :
  let r1 : ℝ := 3
  let r2 : ℝ := 5
  let d1 : ℝ := 9
  let d2 : ℝ := 1
  ¬(circlesIntersect r1 r2 d1) ∧ ¬(circlesIntersect r1 r2 d2) :=
by
  sorry

#check no_intersection_for_given_scenarios

end NUMINAMATH_CALUDE_no_intersection_for_given_scenarios_l2752_275276


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2752_275271

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 4| + 3 * x = 14 :=
by
  -- The unique solution is x = 4.5
  use 4.5
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2752_275271


namespace NUMINAMATH_CALUDE_factors_of_1320_l2752_275202

def n : ℕ := 1320

-- Count of distinct, positive factors
def count_factors (m : ℕ) : ℕ := sorry

-- Count of perfect square factors
def count_square_factors (m : ℕ) : ℕ := sorry

theorem factors_of_1320 :
  count_factors n = 24 ∧ count_square_factors n = 2 := by sorry

end NUMINAMATH_CALUDE_factors_of_1320_l2752_275202


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l2752_275204

/-- Two triangles XYZ and PQR are similar with a shared angle of 150 degrees. 
    Given the side lengths XY = 10, XZ = 20, and PR = 12, prove that PQ = 2.5. -/
theorem similar_triangles_side_length 
  (XY : ℝ) (XZ : ℝ) (PR : ℝ) (PQ : ℝ) 
  (h1 : XY = 10) 
  (h2 : XZ = 20) 
  (h3 : PR = 12) 
  (h4 : ∃ θ : ℝ, θ = 150 * π / 180) -- 150 degrees in radians
  (h5 : XY / PQ = XZ / PR) : -- similarity condition
  PQ = 2.5 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l2752_275204


namespace NUMINAMATH_CALUDE_evaluate_expression_l2752_275232

theorem evaluate_expression : 5 - 9 * (8 - 3 * 2) / 2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2752_275232


namespace NUMINAMATH_CALUDE_max_value_fraction_l2752_275270

theorem max_value_fraction (x y : ℝ) (hx : -3 ≤ x ∧ x ≤ -1) (hy : 3 ≤ y ∧ y ≤ 6) :
  (∀ a b, -3 ≤ a ∧ a ≤ -1 ∧ 3 ≤ b ∧ b ≤ 6 → (a + b) / a ≤ (x + y) / x) →
  (x + y) / x = -2 :=
sorry

end NUMINAMATH_CALUDE_max_value_fraction_l2752_275270


namespace NUMINAMATH_CALUDE_total_days_is_210_l2752_275284

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

end NUMINAMATH_CALUDE_total_days_is_210_l2752_275284


namespace NUMINAMATH_CALUDE_tammy_second_day_speed_l2752_275263

/-- Represents Tammy's mountain climbing over two days -/
structure MountainClimb where
  total_time : ℝ
  speed_increase : ℝ
  time_decrease : ℝ
  uphill_speed_decrease : ℝ
  downhill_speed_increase : ℝ
  total_distance : ℝ

/-- Calculates Tammy's average speed on the second day -/
def second_day_speed (climb : MountainClimb) : ℝ :=
  -- Definition to be proved
  4

/-- Theorem stating that Tammy's average speed on the second day was 4 km/h -/
theorem tammy_second_day_speed (climb : MountainClimb) 
  (h1 : climb.total_time = 14)
  (h2 : climb.speed_increase = 0.5)
  (h3 : climb.time_decrease = 2)
  (h4 : climb.uphill_speed_decrease = 1)
  (h5 : climb.downhill_speed_increase = 1)
  (h6 : climb.total_distance = 52) :
  second_day_speed climb = 4 := by
  sorry

end NUMINAMATH_CALUDE_tammy_second_day_speed_l2752_275263


namespace NUMINAMATH_CALUDE_length_AB_l2752_275299

/-- Parabola C: y^2 = 8x -/
def parabola_C (x y : ℝ) : Prop := y^2 = 8*x

/-- Line l: y = (√3/3)(x-2) -/
def line_l (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * (x - 2)

/-- A and B are intersection points of C and l -/
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola_C A.1 A.2 ∧ line_l A.1 A.2 ∧
  parabola_C B.1 B.2 ∧ line_l B.1 B.2 ∧
  A ≠ B

/-- The length of AB is 32 -/
theorem length_AB (A B : ℝ × ℝ) (h : intersection_points A B) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 32 := by sorry

end NUMINAMATH_CALUDE_length_AB_l2752_275299


namespace NUMINAMATH_CALUDE_subset_union_inclusion_fraction_inequality_ac_squared_eq_bc_squared_not_sufficient_negation_of_all_positive_real_l2752_275252

-- 1. Set inclusion property
theorem subset_union_inclusion (M N : Set α) : M ⊆ N → M ⊆ (M ∪ N) := by sorry

-- 2. Fraction inequality
theorem fraction_inequality (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  (b + m) / (a + m) > b / a := by sorry

-- 3. Counterexample for ac² = bc² implying a = b
theorem ac_squared_eq_bc_squared_not_sufficient :
  ∃ (a b c : ℝ), a * c^2 = b * c^2 ∧ a ≠ b := by sorry

-- 4. Negation of universal quantifier
theorem negation_of_all_positive_real :
  ¬(∀ (x : ℝ), x > 0) ≠ (∃ (x : ℝ), x < 0) := by sorry

end NUMINAMATH_CALUDE_subset_union_inclusion_fraction_inequality_ac_squared_eq_bc_squared_not_sufficient_negation_of_all_positive_real_l2752_275252


namespace NUMINAMATH_CALUDE_carrot_usage_l2752_275214

theorem carrot_usage (total_carrots : ℕ) (unused_carrots : ℕ) 
  (h1 : total_carrots = 300)
  (h2 : unused_carrots = 72) : 
  ∃ (x : ℚ), 
    x * total_carrots + (3/5 : ℚ) * (total_carrots - x * total_carrots) = total_carrots - unused_carrots ∧ 
    x = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_carrot_usage_l2752_275214


namespace NUMINAMATH_CALUDE_absolute_value_difference_l2752_275200

theorem absolute_value_difference (m n : ℝ) (hm : m < 0) (hmn : m * n < 0) :
  |n - m + 1| - |m - n - 5| = -4 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_difference_l2752_275200


namespace NUMINAMATH_CALUDE_equation_solution_l2752_275292

theorem equation_solution :
  ∃ x : ℝ, (1/8 : ℝ)^(3*x + 12) = (64 : ℝ)^(x + 4) ↔ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2752_275292


namespace NUMINAMATH_CALUDE_tim_speed_is_45_l2752_275208

/-- Represents the distance between Tim and Élan in miles -/
def initial_distance : ℝ := 150

/-- Represents Élan's initial speed in mph -/
def elan_initial_speed : ℝ := 5

/-- Represents the distance Tim travels until meeting Élan in miles -/
def tim_travel_distance : ℝ := 100

/-- Represents the number of hours until Tim and Élan meet -/
def meeting_time : ℕ := 2

/-- Represents Tim's initial speed in mph -/
def tim_initial_speed : ℝ := 45

/-- Theorem stating that given the conditions, Tim's initial speed is 45 mph -/
theorem tim_speed_is_45 :
  tim_initial_speed * (2^meeting_time - 1) = initial_distance - elan_initial_speed * (2^meeting_time - 1) :=
sorry

end NUMINAMATH_CALUDE_tim_speed_is_45_l2752_275208


namespace NUMINAMATH_CALUDE_ada_was_in_seat_two_l2752_275267

/-- Represents the seats in the row --/
inductive Seat
  | one
  | two
  | three
  | four
  | five

/-- Represents the friends --/
inductive Friend
  | ada
  | bea
  | ceci
  | dee
  | edie

/-- Represents the seating arrangement --/
def Arrangement := Friend → Seat

/-- The initial seating arrangement --/
def initial_arrangement : Arrangement := sorry

/-- The final seating arrangement after all movements --/
def final_arrangement : Arrangement := sorry

/-- Bea moves one seat to the right --/
def bea_moves (arr : Arrangement) : Arrangement := sorry

/-- Ceci moves left and then back --/
def ceci_moves (arr : Arrangement) : Arrangement := sorry

/-- Dee and Edie switch seats, then Edie moves right --/
def dee_edie_move (arr : Arrangement) : Arrangement := sorry

/-- Ada's original seat --/
def ada_original_seat : Seat := sorry

theorem ada_was_in_seat_two :
  ada_original_seat = Seat.two ∧
  final_arrangement = dee_edie_move (ceci_moves (bea_moves initial_arrangement)) ∧
  (final_arrangement Friend.ada = Seat.one ∨ final_arrangement Friend.ada = Seat.five) :=
sorry

end NUMINAMATH_CALUDE_ada_was_in_seat_two_l2752_275267


namespace NUMINAMATH_CALUDE_lee_cookies_l2752_275209

/-- Given that Lee can make 24 cookies with 3 cups of flour, 
    this function calculates how many cookies he can make with any number of cups. -/
def cookies_per_cups (cups : ℚ) : ℚ :=
  (24 / 3) * cups

/-- Theorem stating that Lee can make 40 cookies with 5 cups of flour. -/
theorem lee_cookies : cookies_per_cups 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_lee_cookies_l2752_275209


namespace NUMINAMATH_CALUDE_sqrt_x_plus_reciprocal_l2752_275230

theorem sqrt_x_plus_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_reciprocal_l2752_275230
