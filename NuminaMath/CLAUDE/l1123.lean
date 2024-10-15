import Mathlib

namespace NUMINAMATH_CALUDE_square_octagon_exterior_angle_l1123_112379

/-- The measure of an interior angle of a regular polygon with n sides -/
def interior_angle (n : ℕ) : ℚ := 180 * (n - 2) / n

/-- The configuration of a square and regular octagon sharing a side -/
structure SquareOctagonConfig where
  square_angle : ℚ  -- Interior angle of the square
  octagon_angle : ℚ -- Interior angle of the octagon
  common_side : ℚ   -- Length of the common side (not used in this problem, but included for completeness)

/-- The exterior angle formed by the non-shared sides of the square and octagon -/
def exterior_angle (config : SquareOctagonConfig) : ℚ :=
  360 - config.square_angle - config.octagon_angle

/-- Theorem: The exterior angle in the square-octagon configuration is 135° -/
theorem square_octagon_exterior_angle :
  ∀ (config : SquareOctagonConfig),
    config.square_angle = 90 ∧
    config.octagon_angle = interior_angle 8 →
    exterior_angle config = 135 := by
  sorry


end NUMINAMATH_CALUDE_square_octagon_exterior_angle_l1123_112379


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_l1123_112302

theorem mean_equality_implies_z (z : ℚ) : 
  (7 + 10 + 23) / 3 = (18 + z) / 2 → z = 26 / 3 := by
sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_l1123_112302


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1123_112374

/-- The perimeter of an equilateral triangle with side length 13/12 meters is 3.25 meters. -/
theorem equilateral_triangle_perimeter :
  let side_length : ℚ := 13/12
  let perimeter : ℚ := 3 * side_length
  perimeter = 13/4 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1123_112374


namespace NUMINAMATH_CALUDE_fractional_exponent_simplification_l1123_112300

theorem fractional_exponent_simplification (a : ℝ) (ha : a > 0) :
  a^2 / (a^(1/2) * a^(2/3)) = a^(5/6) := by
  sorry

end NUMINAMATH_CALUDE_fractional_exponent_simplification_l1123_112300


namespace NUMINAMATH_CALUDE_coffee_stock_problem_l1123_112326

/-- Proves that given the conditions of the coffee stock problem, 
    the percentage of the initial stock that was decaffeinated is 20%. -/
theorem coffee_stock_problem (initial_stock : ℝ) (additional_purchase : ℝ) 
  (decaf_percent_new : ℝ) (total_decaf_percent : ℝ) :
  initial_stock = 400 →
  additional_purchase = 100 →
  decaf_percent_new = 60 →
  total_decaf_percent = 28.000000000000004 →
  (initial_stock * (20 / 100) + additional_purchase * (decaf_percent_new / 100)) / 
  (initial_stock + additional_purchase) * 100 = total_decaf_percent :=
by
  sorry

end NUMINAMATH_CALUDE_coffee_stock_problem_l1123_112326


namespace NUMINAMATH_CALUDE_sum_of_squares_in_ratio_l1123_112307

theorem sum_of_squares_in_ratio (a b c : ℚ) : 
  (a : ℚ) + b + c = 15 →
  b = 2 * a →
  c = 4 * a →
  a^2 + b^2 + c^2 = 4725 / 49 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_in_ratio_l1123_112307


namespace NUMINAMATH_CALUDE_lipschitz_periodic_bound_l1123_112332

/-- A function f is k-Lipschitz if |f(x) - f(y)| ≤ k|x - y| for all x, y in the domain -/
def is_k_lipschitz (f : ℝ → ℝ) (k : ℝ) :=
  ∀ x y, |f x - f y| ≤ k * |x - y|

/-- A function f is periodic with period T if f(x + T) = f(x) for all x -/
def is_periodic (f : ℝ → ℝ) (T : ℝ) :=
  ∀ x, f (x + T) = f x

theorem lipschitz_periodic_bound
  (f : ℝ → ℝ)
  (h_lipschitz : is_k_lipschitz f 1)
  (h_periodic : is_periodic f 2) :
  ∀ x₁ x₂ : ℝ, |f x₁ - f x₂| ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_lipschitz_periodic_bound_l1123_112332


namespace NUMINAMATH_CALUDE_area_of_region_l1123_112350

theorem area_of_region (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 10 ∧ 
   A = Real.pi * (Real.sqrt ((x + 1)^2 + (y - 2)^2)) ^ 2 ∧
   x^2 + y^2 + 2*x - 4*y = 5) := by sorry

end NUMINAMATH_CALUDE_area_of_region_l1123_112350


namespace NUMINAMATH_CALUDE_negative_power_six_interpretation_l1123_112353

theorem negative_power_six_interpretation :
  -2^6 = -(2 * 2 * 2 * 2 * 2 * 2) := by
  sorry

end NUMINAMATH_CALUDE_negative_power_six_interpretation_l1123_112353


namespace NUMINAMATH_CALUDE_divisor_problem_l1123_112313

theorem divisor_problem (n : ℤ) (d : ℤ) : 
  (∃ k : ℤ, n = 18 * k + 10) → 
  (∃ q : ℤ, 2 * n = d * q + 2) → 
  d = 18 := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l1123_112313


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1123_112311

/-- The complex number -2i+1 corresponds to a point in the fourth quadrant of the complex plane -/
theorem complex_number_in_fourth_quadrant :
  let z : ℂ := -2 * Complex.I + 1
  (z.re > 0) ∧ (z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1123_112311


namespace NUMINAMATH_CALUDE_karen_cookie_distribution_l1123_112344

/-- Calculates the number of cookies each person in Karen's class receives -/
def cookies_per_person (total_cookies : ℕ) (kept_cookies : ℕ) (grandparents_cookies : ℕ) (class_size : ℕ) : ℕ :=
  (total_cookies - kept_cookies - grandparents_cookies) / class_size

/-- Theorem stating that each person in Karen's class receives 2 cookies -/
theorem karen_cookie_distribution :
  cookies_per_person 50 10 8 16 = 2 := by
  sorry

#eval cookies_per_person 50 10 8 16

end NUMINAMATH_CALUDE_karen_cookie_distribution_l1123_112344


namespace NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l1123_112327

/-- Given a line ax - by - 2 = 0 and a curve y = x³ with perpendicular tangents at point P(1,1),
    the value of b/a is -3. -/
theorem perpendicular_tangents_ratio (a b : ℝ) : 
  (∃ (x y : ℝ), a * x - b * y - 2 = 0 ∧ y = x^3) →  -- Line and curve equations
  (∃ (x y : ℝ), x = 1 ∧ y = 1 ∧ 
    (∀ (t : ℝ), a * t - b * (t^3) - 2 = 0 ↔ a * (x - t) + b * (y - t^3) = 0)) →  -- Perpendicular tangents at P(1,1)
  b / a = -3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l1123_112327


namespace NUMINAMATH_CALUDE_min_workers_for_profit_l1123_112336

/-- Represents the problem of determining the minimum number of workers needed for profit --/
theorem min_workers_for_profit 
  (maintenance_fee : ℝ) 
  (hourly_wage : ℝ) 
  (hourly_production : ℝ) 
  (widget_price : ℝ) 
  (work_hours : ℝ)
  (h1 : maintenance_fee = 600)
  (h2 : hourly_wage = 20)
  (h3 : hourly_production = 3)
  (h4 : widget_price = 2.80)
  (h5 : work_hours = 8) :
  (∃ n : ℕ, n * hourly_production * work_hours * widget_price > 
             maintenance_fee + n * hourly_wage * work_hours) ∧
  (∀ m : ℕ, m * hourly_production * work_hours * widget_price > 
             maintenance_fee + m * hourly_wage * work_hours → m ≥ 7) :=
by sorry

end NUMINAMATH_CALUDE_min_workers_for_profit_l1123_112336


namespace NUMINAMATH_CALUDE_beautiful_equations_proof_l1123_112349

/-- Two linear equations are "beautiful equations" if the sum of their solutions is 1 -/
def beautiful_equations (eq1 eq2 : ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), eq1 x ∧ eq2 y ∧ x + y = 1

/-- The first pair of equations -/
def eq1 (x : ℝ) : Prop := 4*x - (x + 5) = 1
def eq2 (y : ℝ) : Prop := -2*y - y = 3

/-- The second pair of equations with parameter n -/
def eq3 (n : ℝ) (x : ℝ) : Prop := 2*x - n + 3 = 0
def eq4 (n : ℝ) (x : ℝ) : Prop := x + 5*n - 1 = 0

theorem beautiful_equations_proof :
  (beautiful_equations eq1 eq2) ∧
  (∃ (n : ℝ), n = -1/3 ∧ beautiful_equations (eq3 n) (eq4 n)) :=
by sorry

end NUMINAMATH_CALUDE_beautiful_equations_proof_l1123_112349


namespace NUMINAMATH_CALUDE_six_people_lineup_permutations_l1123_112343

theorem six_people_lineup_permutations : Nat.factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_six_people_lineup_permutations_l1123_112343


namespace NUMINAMATH_CALUDE_mean_equality_implies_values_l1123_112310

theorem mean_equality_implies_values (x y : ℝ) : 
  (2 + 11 + 6 + x) / 4 = (14 + 9 + y) / 3 → x = -35 ∧ y = -35 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_values_l1123_112310


namespace NUMINAMATH_CALUDE_hexagon_interior_angles_sum_l1123_112348

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

/-- A hexagon has 6 sides -/
def hexagon_sides : ℕ := 6

theorem hexagon_interior_angles_sum :
  sum_interior_angles hexagon_sides = 720 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_interior_angles_sum_l1123_112348


namespace NUMINAMATH_CALUDE_no_solution_l1123_112319

theorem no_solution : ¬∃ (k j x : ℝ), 
  (64 / k = 8) ∧ 
  (k * j = 128) ∧ 
  (j - x = k) ∧ 
  (x^2 + j = 3 * k) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_l1123_112319


namespace NUMINAMATH_CALUDE_distance_thirty_students_l1123_112317

/-- The distance between the first and last student in a line of students -/
def distance_between_ends (num_students : ℕ) (gap_distance : ℝ) : ℝ :=
  (num_students - 1 : ℝ) * gap_distance

/-- Theorem: For 30 students standing in a line with 3 meters between adjacent students,
    the distance between the first and last student is 87 meters. -/
theorem distance_thirty_students :
  distance_between_ends 30 3 = 87 := by
  sorry

end NUMINAMATH_CALUDE_distance_thirty_students_l1123_112317


namespace NUMINAMATH_CALUDE_midpoint_value_part1_midpoint_value_part2_l1123_112322

-- Definition of midpoint value
def is_midpoint_value (a b : ℝ) : Prop := a^2 - b > 0

-- Part 1
theorem midpoint_value_part1 : 
  is_midpoint_value 4 3 ∧ ∀ x, x^2 - 8*x + 3 = 0 ↔ x^2 - 2*4*x + 3 = 0 :=
sorry

-- Part 2
theorem midpoint_value_part2 (m n : ℝ) : 
  (is_midpoint_value 3 n ∧ 
   ∀ x, x^2 - m*x + n = 0 ↔ x^2 - 2*3*x + n = 0 ∧
   (n^2 - m*n + n = 0)) →
  (n = 0 ∨ n = 5) :=
sorry

end NUMINAMATH_CALUDE_midpoint_value_part1_midpoint_value_part2_l1123_112322


namespace NUMINAMATH_CALUDE_fixed_fee_calculation_l1123_112369

/-- Represents the billing system for an online service provider -/
structure BillingSystem where
  fixed_fee : ℝ
  hourly_charge : ℝ

/-- Calculates the total bill given the connect time -/
def total_bill (bs : BillingSystem) (connect_time : ℝ) : ℝ :=
  bs.fixed_fee + bs.hourly_charge * connect_time

theorem fixed_fee_calculation (bs : BillingSystem) :
  total_bill bs 1 = 15.75 ∧ total_bill bs 3 = 24.45 → bs.fixed_fee = 11.40 := by
  sorry

end NUMINAMATH_CALUDE_fixed_fee_calculation_l1123_112369


namespace NUMINAMATH_CALUDE_directrix_of_given_parabola_l1123_112375

/-- A parabola in the xy-plane -/
structure Parabola where
  /-- The equation of the parabola in the form y = ax^2 + bx + c -/
  equation : ℝ → ℝ → Prop

/-- The directrix of a parabola -/
def directrix (p : Parabola) : ℝ → ℝ → Prop :=
  sorry

/-- The given parabola y = -3x^2 + 6x - 5 -/
def given_parabola : Parabola :=
  { equation := λ x y => y = -3 * x^2 + 6 * x - 5 }

theorem directrix_of_given_parabola :
  directrix given_parabola = λ x y => y = -35/18 :=
sorry

end NUMINAMATH_CALUDE_directrix_of_given_parabola_l1123_112375


namespace NUMINAMATH_CALUDE_factorization_of_quadratic_l1123_112370

theorem factorization_of_quadratic (x : ℝ) : x^2 - 3*x = x*(x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_quadratic_l1123_112370


namespace NUMINAMATH_CALUDE_ali_baba_maximum_value_l1123_112388

/-- Represents the problem of maximizing the value of gold and diamonds in one trip --/
theorem ali_baba_maximum_value :
  let gold_weight : ℝ := 200
  let diamond_weight : ℝ := 40
  let max_carry_weight : ℝ := 100
  let gold_value_per_kg : ℝ := 20
  let diamond_value_per_kg : ℝ := 60
  
  ∀ x y : ℝ,
  x ≥ 0 → y ≥ 0 →
  x + y = max_carry_weight →
  x * gold_value_per_kg + y * diamond_value_per_kg ≤ 3000 :=
by sorry

end NUMINAMATH_CALUDE_ali_baba_maximum_value_l1123_112388


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1123_112318

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 + (k + Complex.I)*x - 2 - k*Complex.I = 0) ↔ (k = 1 ∨ k = -1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1123_112318


namespace NUMINAMATH_CALUDE_molecular_weight_N2O5_l1123_112345

/-- The atomic weight of nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The molecular weight of a compound in g/mol -/
def molecular_weight (n_count : ℕ) (o_count : ℕ) : ℝ :=
  n_count * atomic_weight_N + o_count * atomic_weight_O

/-- Theorem stating that the molecular weight of N2O5 is 108.02 g/mol -/
theorem molecular_weight_N2O5 : 
  molecular_weight 2 5 = 108.02 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_N2O5_l1123_112345


namespace NUMINAMATH_CALUDE_train_speed_l1123_112377

/-- The speed of a train given its length and time to cross a post -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 250.02) (h2 : time = 22.5) :
  (length * 3600) / (time * 1000) = 40.0032 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1123_112377


namespace NUMINAMATH_CALUDE_circle_chord_theorem_l1123_112356

-- Define the circle equation
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 2*y + 2*a = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + y + 2 = 0

-- Define the chord length
def chord_length : ℝ := 4

-- Theorem statement
theorem circle_chord_theorem (a : ℝ) :
  (∀ x y : ℝ, circle_equation x y a ∧ line_equation x y) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_equation x₁ y₁ a ∧ line_equation x₁ y₁ ∧
    circle_equation x₂ y₂ a ∧ line_equation x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = chord_length^2) →
  a = -2 := by
  sorry

end NUMINAMATH_CALUDE_circle_chord_theorem_l1123_112356


namespace NUMINAMATH_CALUDE_angle_subtraction_theorem_l1123_112383

/-- Represents an angle in degrees and minutes -/
structure AngleDM where
  degrees : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Addition of angles in degrees and minutes -/
def add_angles (a b : AngleDM) : AngleDM :=
  let total_minutes := a.minutes + b.minutes
  let carry_degrees := total_minutes / 60
  { degrees := a.degrees + b.degrees + carry_degrees,
    minutes := total_minutes % 60,
    valid := by sorry }

/-- Subtraction of angles in degrees and minutes -/
def sub_angles (a b : AngleDM) : AngleDM :=
  let total_minutes := a.degrees * 60 + a.minutes - (b.degrees * 60 + b.minutes)
  { degrees := total_minutes / 60,
    minutes := total_minutes % 60,
    valid := by sorry }

/-- The main theorem to prove -/
theorem angle_subtraction_theorem :
  sub_angles { degrees := 72, minutes := 24, valid := by sorry }
              { degrees := 28, minutes := 36, valid := by sorry } =
  { degrees := 43, minutes := 48, valid := by sorry } := by sorry

end NUMINAMATH_CALUDE_angle_subtraction_theorem_l1123_112383


namespace NUMINAMATH_CALUDE_yellow_ball_estimate_l1123_112328

/-- Represents the contents of a bag with red and yellow balls -/
structure BagContents where
  red_balls : ℕ
  yellow_balls : ℕ

/-- Represents the result of multiple trials of drawing balls -/
structure TrialResults where
  num_trials : ℕ
  avg_red_ratio : ℝ

/-- Estimates the number of yellow balls in the bag based on trial results -/
def estimate_yellow_balls (bag : BagContents) (trials : TrialResults) : ℕ :=
  sorry

theorem yellow_ball_estimate (bag : BagContents) (trials : TrialResults) :
  bag.red_balls = 10 ∧ 
  trials.num_trials = 20 ∧ 
  trials.avg_red_ratio = 0.4 →
  estimate_yellow_balls bag trials = 15 :=
sorry

end NUMINAMATH_CALUDE_yellow_ball_estimate_l1123_112328


namespace NUMINAMATH_CALUDE_yearly_income_is_130_l1123_112396

/-- Calculates the yearly simple interest income given principal and rate -/
def simple_interest (principal : ℕ) (rate : ℕ) : ℕ :=
  principal * rate / 100

/-- Proves that the yearly annual income is 130 given the specified conditions -/
theorem yearly_income_is_130 (total : ℕ) (part1 : ℕ) (rate1 : ℕ) (rate2 : ℕ) 
  (h1 : total = 2500)
  (h2 : part1 = 2000)
  (h3 : rate1 = 5)
  (h4 : rate2 = 6) :
  simple_interest part1 rate1 + simple_interest (total - part1) rate2 = 130 := by
  sorry

#eval simple_interest 2000 5 + simple_interest 500 6

end NUMINAMATH_CALUDE_yearly_income_is_130_l1123_112396


namespace NUMINAMATH_CALUDE_older_ate_twelve_l1123_112333

/-- Represents the pancake eating scenario -/
structure PancakeScenario where
  initial_pancakes : ℕ
  final_pancakes : ℕ
  younger_eats : ℕ
  older_eats : ℕ
  grandma_bakes : ℕ

/-- Calculates the number of pancakes eaten by the older grandchild -/
def older_grandchild_pancakes (scenario : PancakeScenario) : ℕ :=
  let net_reduction := scenario.younger_eats + scenario.older_eats - scenario.grandma_bakes
  let cycles := (scenario.initial_pancakes - scenario.final_pancakes) / net_reduction
  scenario.older_eats * cycles

/-- Theorem stating that the older grandchild ate 12 pancakes in the given scenario -/
theorem older_ate_twelve (scenario : PancakeScenario) 
  (h1 : scenario.initial_pancakes = 19)
  (h2 : scenario.final_pancakes = 11)
  (h3 : scenario.younger_eats = 1)
  (h4 : scenario.older_eats = 3)
  (h5 : scenario.grandma_bakes = 2) :
  older_grandchild_pancakes scenario = 12 := by
  sorry

#eval older_grandchild_pancakes { 
  initial_pancakes := 19, 
  final_pancakes := 11, 
  younger_eats := 1, 
  older_eats := 3, 
  grandma_bakes := 2 
}

end NUMINAMATH_CALUDE_older_ate_twelve_l1123_112333


namespace NUMINAMATH_CALUDE_number_guessing_game_l1123_112367

theorem number_guessing_game (a b c d : ℕ) 
  (ha : a ≥ 10) 
  (hb : b < 10) (hc : c < 10) (hd : d < 10) : 
  ((((((a * 2 + 1) * 5 + b) * 2 + 1) * 5 + c) * 2 + 1) * 5 + d) - 555 = 1000 * a + 100 * b + 10 * c + d :=
by sorry

#check number_guessing_game

end NUMINAMATH_CALUDE_number_guessing_game_l1123_112367


namespace NUMINAMATH_CALUDE_sequence_with_least_period_l1123_112341

theorem sequence_with_least_period (p : ℕ) (h : p ≥ 2) :
  ∃ (x : ℕ → ℝ), 
    (∀ n, x (n + p) = x n) ∧ 
    (∀ n, x (n + 1) = x n - 1 / x n) ∧
    (∀ k, k < p → ¬(∀ n, x (n + k) = x n)) := by
  sorry

end NUMINAMATH_CALUDE_sequence_with_least_period_l1123_112341


namespace NUMINAMATH_CALUDE_number_of_divisors_of_36_l1123_112387

theorem number_of_divisors_of_36 : Finset.card (Finset.filter (· ∣ 36) (Finset.range 37)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_36_l1123_112387


namespace NUMINAMATH_CALUDE_inequality_implication_l1123_112368

theorem inequality_implication (a b c : ℝ) : a * c^2 > b * c^2 → a > b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l1123_112368


namespace NUMINAMATH_CALUDE_power_product_equals_ten_thousand_l1123_112340

theorem power_product_equals_ten_thousand : (2 ^ 4) * (5 ^ 4) = 10000 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_ten_thousand_l1123_112340


namespace NUMINAMATH_CALUDE_volleyball_lineup_combinations_l1123_112306

theorem volleyball_lineup_combinations (total_players : ℕ) 
  (starting_lineup_size : ℕ) (required_players : ℕ) : 
  total_players = 15 → 
  starting_lineup_size = 7 → 
  required_players = 3 → 
  Nat.choose (total_players - required_players) (starting_lineup_size - required_players) = 495 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_lineup_combinations_l1123_112306


namespace NUMINAMATH_CALUDE_painting_time_relation_l1123_112303

/-- Time taken by Taylor to paint the room alone -/
def taylor_time : ℝ := 12

/-- Time taken by Taylor and Jennifer together to paint the room -/
def combined_time : ℝ := 5.45454545455

/-- Time taken by Jennifer to paint the room alone -/
def jennifer_time : ℝ := 10.1538461538

/-- Theorem stating the relationship between individual and combined painting times -/
theorem painting_time_relation :
  1 / taylor_time + 1 / jennifer_time = 1 / combined_time :=
sorry

end NUMINAMATH_CALUDE_painting_time_relation_l1123_112303


namespace NUMINAMATH_CALUDE_linear_function_not_in_second_quadrant_l1123_112381

/-- A linear function f(x) = mx + b -/
structure LinearFunction where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- The four quadrants of the Cartesian plane -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- Determine if a point (x, y) is in a given quadrant -/
def inQuadrant (x y : ℝ) (q : Quadrant) : Prop :=
  match q with
  | Quadrant.I  => x > 0 ∧ y > 0
  | Quadrant.II => x < 0 ∧ y > 0
  | Quadrant.III => x < 0 ∧ y < 0
  | Quadrant.IV => x > 0 ∧ y < 0

/-- A linear function passes through a quadrant if there exists a point (x, y) in that quadrant satisfying the function equation -/
def passesThrough (f : LinearFunction) (q : Quadrant) : Prop :=
  ∃ x y : ℝ, y = f.m * x + f.b ∧ inQuadrant x y q

/-- The main theorem: the graph of y = 2x - 3 does not pass through the second quadrant -/
theorem linear_function_not_in_second_quadrant :
  ¬ passesThrough ⟨2, -3⟩ Quadrant.II := by
  sorry


end NUMINAMATH_CALUDE_linear_function_not_in_second_quadrant_l1123_112381


namespace NUMINAMATH_CALUDE_height_relation_l1123_112378

/-- Two right circular cylinders with equal volumes and related radii -/
structure TwoCylinders where
  r₁ : ℝ  -- radius of the first cylinder
  h₁ : ℝ  -- height of the first cylinder
  r₂ : ℝ  -- radius of the second cylinder
  h₂ : ℝ  -- height of the second cylinder
  r₁_pos : 0 < r₁
  h₁_pos : 0 < h₁
  r₂_pos : 0 < r₂
  h₂_pos : 0 < h₂
  equal_volume : r₁^2 * h₁ = r₂^2 * h₂
  radius_relation : r₂ = 1.2 * r₁

theorem height_relation (c : TwoCylinders) : c.h₁ = 1.44 * c.h₂ := by
  sorry

end NUMINAMATH_CALUDE_height_relation_l1123_112378


namespace NUMINAMATH_CALUDE_base7_arithmetic_l1123_112312

/-- Represents a number in base 7 --/
structure Base7 where
  digits : List Nat
  valid : ∀ d ∈ digits, d < 7

/-- Addition operation for Base7 numbers --/
def add_base7 (a b : Base7) : Base7 := sorry

/-- Subtraction operation for Base7 numbers --/
def sub_base7 (a b : Base7) : Base7 := sorry

/-- Conversion from a natural number to Base7 --/
def nat_to_base7 (n : Nat) : Base7 := sorry

theorem base7_arithmetic :
  let a := nat_to_base7 24
  let b := nat_to_base7 356
  let c := nat_to_base7 105
  let d := nat_to_base7 265
  sub_base7 (add_base7 a b) c = d := by sorry

end NUMINAMATH_CALUDE_base7_arithmetic_l1123_112312


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l1123_112391

theorem smallest_number_with_conditions : ∃ n : ℕ, 
  (∀ k : ℕ, k < n → ¬(11 ∣ k ∧ (∀ i : ℕ, 3 ≤ i ∧ i ≤ 7 → k % i = 2))) ∧ 
  11 ∣ n ∧ 
  (∀ i : ℕ, 3 ≤ i ∧ i ≤ 7 → n % i = 2) ∧ 
  n = 2102 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l1123_112391


namespace NUMINAMATH_CALUDE_sine_special_angle_l1123_112384

theorem sine_special_angle (α : Real) 
  (h1 : π / 2 < α ∧ α < π) 
  (h2 : Real.sin (-π - α) = Real.sqrt 5 / 5) : 
  Real.sin (α - 3 * π / 2) = 2 * Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_sine_special_angle_l1123_112384


namespace NUMINAMATH_CALUDE_log_eight_three_equals_five_twelve_l1123_112393

theorem log_eight_three_equals_five_twelve (x : ℝ) : 
  Real.log x / Real.log 8 = 3 → x = 512 := by
  sorry

end NUMINAMATH_CALUDE_log_eight_three_equals_five_twelve_l1123_112393


namespace NUMINAMATH_CALUDE_inequality_theorem_l1123_112323

theorem inequality_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_prod : a * b * c = 1) 
  (h_ineq : a^2011 + b^2011 + c^2011 < 1/a^2011 + 1/b^2011 + 1/c^2011) : 
  a + b + c < 1/a + 1/b + 1/c := by
sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1123_112323


namespace NUMINAMATH_CALUDE_four_common_divisors_l1123_112363

/-- The number of positive integer divisors that simultaneously divide 60, 84, and 126 -/
def common_divisors : Nat :=
  (Nat.divisors 60 ∩ Nat.divisors 84 ∩ Nat.divisors 126).card

/-- Theorem stating that there are exactly 4 positive integers that simultaneously divide 60, 84, and 126 -/
theorem four_common_divisors : common_divisors = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_common_divisors_l1123_112363


namespace NUMINAMATH_CALUDE_reflection_of_circle_center_l1123_112309

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem reflection_of_circle_center :
  let original_center : ℝ × ℝ := (8, -3)
  let reflected_center : ℝ × ℝ := reflect_about_y_eq_neg_x original_center
  reflected_center = (3, -8) := by sorry

end NUMINAMATH_CALUDE_reflection_of_circle_center_l1123_112309


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1123_112316

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- Given an arithmetic sequence a where a₁ + a₃ + a₅ = 3, prove a₂ + a₄ = 2 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h1 : ArithmeticSequence a) 
    (h2 : a 1 + a 3 + a 5 = 3) : a 2 + a 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1123_112316


namespace NUMINAMATH_CALUDE_max_value_theorem_l1123_112351

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 3) : 
  2*a*b + 2*b*c*Real.sqrt 3 ≤ 6 ∧ ∃ a b c, 2*a*b + 2*b*c*Real.sqrt 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1123_112351


namespace NUMINAMATH_CALUDE_product_of_numbers_l1123_112376

theorem product_of_numbers (x y : ℝ) 
  (h1 : (x - y)^2 / (x + y)^3 = 4 / 27)
  (h2 : x + y = 5 * (x - y) + 3) : 
  x * y = 15.75 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1123_112376


namespace NUMINAMATH_CALUDE_evaluate_expression_l1123_112304

theorem evaluate_expression (x z : ℝ) (hx : x = 4) (hz : z = 0) : z * (z - 4 * x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1123_112304


namespace NUMINAMATH_CALUDE_rainy_days_calculation_l1123_112337

/-- Calculates the number of rainy days in a week given cycling conditions --/
def rainy_days (rain_speed : ℕ) (snow_speed : ℕ) (snow_days : ℕ) (total_distance : ℕ) : ℕ :=
  let snow_distance := snow_speed * snow_days
  (total_distance - snow_distance) / rain_speed

theorem rainy_days_calculation :
  rainy_days 90 30 4 390 = 3 := by
  sorry

end NUMINAMATH_CALUDE_rainy_days_calculation_l1123_112337


namespace NUMINAMATH_CALUDE_nested_radical_value_l1123_112330

theorem nested_radical_value : 
  ∃ x : ℝ, x = Real.sqrt (20 + x) ∧ x > 0 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_value_l1123_112330


namespace NUMINAMATH_CALUDE_edward_scored_seven_l1123_112338

/-- Given the total points scored and the friend's score, calculate Edward's score. -/
def edward_score (total : ℕ) (friend_score : ℕ) : ℕ :=
  total - friend_score

/-- Theorem: Edward's score is 7 points when the total is 13 and his friend scored 6. -/
theorem edward_scored_seven :
  edward_score 13 6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_edward_scored_seven_l1123_112338


namespace NUMINAMATH_CALUDE_floor_of_5_7_l1123_112321

theorem floor_of_5_7 : ⌊(5.7 : ℝ)⌋ = 5 := by sorry

end NUMINAMATH_CALUDE_floor_of_5_7_l1123_112321


namespace NUMINAMATH_CALUDE_discount_calculation_l1123_112362

/-- Calculates the final amount paid after applying a discount -/
def finalAmount (initialAmount : ℕ) (discountPer100 : ℕ) : ℕ :=
  let fullDiscountUnits := initialAmount / 100
  let totalDiscount := fullDiscountUnits * discountPer100
  initialAmount - totalDiscount

/-- Theorem stating that for a $250 purchase with a $10 discount per $100 spent, the final amount is $230 -/
theorem discount_calculation :
  finalAmount 250 10 = 230 := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l1123_112362


namespace NUMINAMATH_CALUDE_concert_drive_l1123_112305

/-- Given a total distance and a distance already driven, calculate the remaining distance. -/
def remaining_distance (total : ℕ) (driven : ℕ) : ℕ :=
  total - driven

/-- Theorem: Given a total distance of 78 miles and having driven 32 miles, 
    the remaining distance to drive is 46 miles. -/
theorem concert_drive : remaining_distance 78 32 = 46 := by
  sorry

end NUMINAMATH_CALUDE_concert_drive_l1123_112305


namespace NUMINAMATH_CALUDE_mathematician_contemporaries_probability_l1123_112355

theorem mathematician_contemporaries_probability :
  let total_years : ℕ := 600
  let lifespan1 : ℕ := 120
  let lifespan2 : ℕ := 100
  let total_area : ℕ := total_years * total_years
  let overlap_area : ℕ := total_area - (lifespan1 * lifespan1 / 2 + lifespan2 * lifespan2 / 2)
  (overlap_area : ℚ) / total_area = 193 / 200 :=
by sorry

end NUMINAMATH_CALUDE_mathematician_contemporaries_probability_l1123_112355


namespace NUMINAMATH_CALUDE_square_difference_square_difference_40_l1123_112359

theorem square_difference (n : ℕ) : (n + 1)^2 - (n - 1)^2 = 4 * n := by
  -- The proof goes here
  sorry

-- Define the specific case for n = 40
def n : ℕ := 40

-- State the theorem for the specific case
theorem square_difference_40 : (n + 1)^2 - (n - 1)^2 = 160 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_square_difference_square_difference_40_l1123_112359


namespace NUMINAMATH_CALUDE_indeterminate_roots_of_related_quadratic_l1123_112347

/-- Given positive numbers a, b, c, and a quadratic equation with two equal real roots,
    the nature of the roots of a related quadratic equation cannot be determined. -/
theorem indeterminate_roots_of_related_quadratic
  (a b c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_equal_roots : ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ (∀ y : ℝ, a * y^2 + b * y + c = 0 → y = x)) :
  ∃ (r₁ r₂ : ℝ), (a + 1) * r₁^2 + (b + 2) * r₁ + (c + 1) = 0 ∧
                 (a + 1) * r₂^2 + (b + 2) * r₂ + (c + 1) = 0 ∧
                 (r₁ = r₂ ∨ r₁ ≠ r₂) :=
sorry

end NUMINAMATH_CALUDE_indeterminate_roots_of_related_quadratic_l1123_112347


namespace NUMINAMATH_CALUDE_right_triangle_expansion_l1123_112320

theorem right_triangle_expansion : ∃ (a b c : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  Nat.gcd a (Nat.gcd b c) = 1 ∧
  a^2 + b^2 = c^2 ∧
  (a + 100)^2 + (b + 100)^2 = (c + 140)^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_expansion_l1123_112320


namespace NUMINAMATH_CALUDE_equation_solution_l1123_112392

theorem equation_solution : 
  ∃ x : ℝ, (x + 1) / 6 = 4 / 3 - x ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1123_112392


namespace NUMINAMATH_CALUDE_f_properties_l1123_112365

open Real

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + a*x - 1

-- Define the derivative of f(x)
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*x + a

theorem f_properties (a : ℝ) :
  (f_deriv a 1 = 1) →
  (a = 2) ∧
  (∃ m b : ℝ, m = 9 ∧ b = 3 ∧ ∀ x y : ℝ, y = f a x → m*(-1) - y + b = 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1123_112365


namespace NUMINAMATH_CALUDE_perfect_square_solution_l1123_112394

theorem perfect_square_solution (t n : ℤ) : 
  (n > 0) → (n^2 + (4*t - 1)*n + 4*t^2 = 0) → ∃ k : ℤ, n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_solution_l1123_112394


namespace NUMINAMATH_CALUDE_expression_value_l1123_112315

theorem expression_value (x y : ℝ) (h1 : 2 * x + 3 * y = 5) (h2 : x = 4) :
  3 * x^2 + 12 * x * y + y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1123_112315


namespace NUMINAMATH_CALUDE_set_associativity_l1123_112352

theorem set_associativity (A B C : Set α) : 
  (A ∪ (B ∪ C) = (A ∪ B) ∪ C) ∧ (A ∩ (B ∩ C) = (A ∩ B) ∩ C) := by
  sorry

end NUMINAMATH_CALUDE_set_associativity_l1123_112352


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l1123_112354

/-- Given a circle with equation x^2 + y^2 - 4x = 0, its symmetric circle
    with respect to the line x = 0 has the equation x^2 + y^2 + 4x = 0 -/
theorem symmetric_circle_equation : 
  ∀ (x y : ℝ), (x^2 + y^2 - 4*x = 0) → 
  ∃ (x' y' : ℝ), (x'^2 + y'^2 + 4*x' = 0) ∧ (x' = -x) ∧ (y' = y) := by
sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l1123_112354


namespace NUMINAMATH_CALUDE_no_real_solutions_l1123_112342

theorem no_real_solutions :
  ¬∃ (x : ℝ), x + 2 * Real.sqrt (x - 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1123_112342


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l1123_112331

theorem rectangle_dimensions (length width : ℝ) : 
  length > 0 → width > 0 → 
  length * width = 120 → 
  2 * (length + width) = 46 → 
  min length width = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l1123_112331


namespace NUMINAMATH_CALUDE_unique_modular_solution_l1123_112382

theorem unique_modular_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ -250 ≡ n [ZMOD 23] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_solution_l1123_112382


namespace NUMINAMATH_CALUDE_stream_speed_l1123_112397

/-- Given a boat with speed 78 kmph in still water, if the time taken upstream is twice
    the time taken downstream, then the speed of the stream is 26 kmph. -/
theorem stream_speed (D : ℝ) (D_pos : D > 0) : 
  let boat_speed : ℝ := 78
  let stream_speed : ℝ := 26
  (D / (boat_speed - stream_speed) = 2 * (D / (boat_speed + stream_speed))) →
  stream_speed = 26 := by
sorry

end NUMINAMATH_CALUDE_stream_speed_l1123_112397


namespace NUMINAMATH_CALUDE_good_pair_exists_l1123_112324

theorem good_pair_exists (m : ℕ) : ∃ n : ℕ, n > m ∧ 
  ∃ a b : ℕ, m * n = a^2 ∧ (m + 1) * (n + 1) = b^2 := by
  sorry

end NUMINAMATH_CALUDE_good_pair_exists_l1123_112324


namespace NUMINAMATH_CALUDE_some_number_value_l1123_112390

theorem some_number_value : ∃ (some_number : ℝ), 
  (0.0077 * 3.6) / (0.04 * some_number * 0.007) = 990.0000000000001 ∧ some_number = 10 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l1123_112390


namespace NUMINAMATH_CALUDE_weighted_average_constants_l1123_112339

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define points A, B, C, P, Q
variable (A B C P Q : V)

-- Define the conditions
variable (hAPC : ∃ (k : ℝ), P - C = k • (A - C) ∧ k = 4/5)
variable (hBQC : ∃ (k : ℝ), Q - C = k • (B - C) ∧ k = 1/5)

-- Define r and s
variable (r s : ℝ)

-- Define the weighted average conditions
variable (hP : P = r • A + (1 - r) • C)
variable (hQ : Q = s • B + (1 - s) • C)

-- State the theorem
theorem weighted_average_constants : r = 1/5 ∧ s = 4/5 := by sorry

end NUMINAMATH_CALUDE_weighted_average_constants_l1123_112339


namespace NUMINAMATH_CALUDE_q_share_approx_l1123_112346

/-- Represents a partner in the partnership -/
inductive Partner
| P
| Q
| R

/-- Calculates the share ratio for a given partner -/
def shareRatio (partner : Partner) : Rat :=
  match partner with
  | Partner.P => 1/2
  | Partner.Q => 1/3
  | Partner.R => 1/4

/-- Calculates the investment duration for a given partner in months -/
def investmentDuration (partner : Partner) : ℕ :=
  match partner with
  | Partner.P => 2
  | _ => 12

/-- Calculates the capital ratio after p's withdrawal -/
def capitalRatioAfterWithdrawal (partner : Partner) : Rat :=
  match partner with
  | Partner.P => 1/4
  | _ => shareRatio partner

/-- The total profit in Rs -/
def totalProfit : ℕ := 378

/-- The total duration of the partnership in months -/
def totalDuration : ℕ := 12

/-- Calculates the investment parts for a given partner -/
def investmentParts (partner : Partner) : Rat :=
  (shareRatio partner * investmentDuration partner) +
  (capitalRatioAfterWithdrawal partner * (totalDuration - investmentDuration partner))

/-- Theorem stating that Q's share of the profit is approximately 123.36 Rs -/
theorem q_share_approx (ε : ℝ) (h : ε > 0) :
  ∃ (q_share : ℝ), abs (q_share - 123.36) < ε ∧
  q_share = (investmentParts Partner.Q / (investmentParts Partner.P + investmentParts Partner.Q + investmentParts Partner.R)) * totalProfit :=
sorry

end NUMINAMATH_CALUDE_q_share_approx_l1123_112346


namespace NUMINAMATH_CALUDE_optimal_sales_distribution_l1123_112398

/-- Represents the sales and profit model for a company selling robots in two locations --/
structure RobotSales where
  x : ℝ  -- Monthly sales volume in both locations
  production_cost : ℝ := 200
  price_A : ℝ := 500
  price_B : ℝ → ℝ := λ x => 1200 - x
  advert_cost_A : ℝ → ℝ := λ x => 100 * x + 10000
  advert_cost_B : ℝ := 50000
  total_sales : ℝ := 1000

/-- Calculates the profit for location A --/
def profit_A (model : RobotSales) : ℝ :=
  model.x * model.price_A - model.x * model.production_cost - model.advert_cost_A model.x

/-- Calculates the profit for location B --/
def profit_B (model : RobotSales) : ℝ :=
  model.x * model.price_B model.x - model.x * model.production_cost - model.advert_cost_B

/-- Calculates the total profit for both locations --/
def total_profit (model : RobotSales) : ℝ :=
  profit_A model + profit_B model

/-- Theorem stating the optimal sales distribution --/
theorem optimal_sales_distribution (model : RobotSales) :
  ∃ (x_A x_B : ℝ),
    x_A + x_B = model.total_sales ∧
    x_A = 600 ∧
    x_B = 400 ∧
    ∀ (y_A y_B : ℝ),
      y_A + y_B = model.total_sales →
      total_profit { model with x := y_A } + total_profit { model with x := y_B } ≤
      total_profit { model with x := x_A } + total_profit { model with x := x_B } :=
sorry

end NUMINAMATH_CALUDE_optimal_sales_distribution_l1123_112398


namespace NUMINAMATH_CALUDE_monotonically_decreasing_range_l1123_112371

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - x - 1

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + a*x - 1

-- Theorem statement
theorem monotonically_decreasing_range (a : ℝ) : 
  (∀ x : ℝ, f_derivative a x ≤ 0) → a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
by
  sorry

#check monotonically_decreasing_range

end NUMINAMATH_CALUDE_monotonically_decreasing_range_l1123_112371


namespace NUMINAMATH_CALUDE_gabriel_forgotten_days_l1123_112395

/-- The number of days in July -/
def days_in_july : ℕ := 31

/-- The number of days Gabriel took his capsules -/
def days_capsules_taken : ℕ := 28

/-- The number of days Gabriel forgot to take his capsules -/
def days_forgotten : ℕ := days_in_july - days_capsules_taken

theorem gabriel_forgotten_days :
  days_forgotten = 3 := by sorry

end NUMINAMATH_CALUDE_gabriel_forgotten_days_l1123_112395


namespace NUMINAMATH_CALUDE_total_visitors_is_440_l1123_112314

/-- Represents the survey results of visitors to a Picasso painting exhibition -/
structure SurveyResults where
  totalVisitors : ℕ
  didNotEnjoyOrUnderstand : ℕ
  enjoyedAndUnderstood : ℕ

/-- The conditions of the survey results -/
def surveyConditions (results : SurveyResults) : Prop :=
  results.didNotEnjoyOrUnderstand = 110 ∧
  results.enjoyedAndUnderstood = 3 * results.totalVisitors / 4 ∧
  results.totalVisitors = results.enjoyedAndUnderstood + results.didNotEnjoyOrUnderstand

/-- The theorem stating that given the survey conditions, the total number of visitors is 440 -/
theorem total_visitors_is_440 (results : SurveyResults) :
  surveyConditions results → results.totalVisitors = 440 := by
  sorry

#check total_visitors_is_440

end NUMINAMATH_CALUDE_total_visitors_is_440_l1123_112314


namespace NUMINAMATH_CALUDE_interior_edge_sum_is_twenty_l1123_112334

/-- A rectangular picture frame with specific properties -/
structure PictureFrame where
  outer_length : ℝ
  outer_width : ℝ
  frame_width : ℝ
  frame_area : ℝ
  outer_length_given : outer_length = 7
  frame_width_given : frame_width = 1
  frame_area_given : frame_area = 24
  positive_dimensions : outer_length > 0 ∧ outer_width > 0

/-- The sum of the interior edge lengths of the picture frame -/
def interior_edge_sum (frame : PictureFrame) : ℝ :=
  2 * (frame.outer_length - 2 * frame.frame_width) + 2 * (frame.outer_width - 2 * frame.frame_width)

/-- Theorem stating that the sum of interior edge lengths is 20 -/
theorem interior_edge_sum_is_twenty (frame : PictureFrame) :
  interior_edge_sum frame = 20 := by
  sorry


end NUMINAMATH_CALUDE_interior_edge_sum_is_twenty_l1123_112334


namespace NUMINAMATH_CALUDE_triangle_area_l1123_112329

theorem triangle_area (P Q R : ℝ) (r R : ℝ) (h1 : r = 3) (h2 : R = 15) 
  (h3 : 2 * Real.cos Q = Real.cos P + Real.cos R) : 
  ∃ (area : ℝ), area = 27 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1123_112329


namespace NUMINAMATH_CALUDE_binary_decimal_octal_equivalence_l1123_112372

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its octal representation as a list of digits -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
    aux n []

theorem binary_decimal_octal_equivalence :
  let binary := [true, true, false, true]  -- 1011 in binary (least significant bit first)
  let decimal := 11
  let octal := [1, 3]
  binary_to_decimal binary = decimal ∧
  decimal_to_octal decimal = octal :=
by sorry

end NUMINAMATH_CALUDE_binary_decimal_octal_equivalence_l1123_112372


namespace NUMINAMATH_CALUDE_range_of_a_l1123_112364

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x

theorem range_of_a (m n : ℝ) (h1 : 0 < m) (h2 : m < n) (h3 : 0 ≤ a) 
  (h4 : Set.Icc m n ⊆ Set.range (f a))
  (h5 : Set.Icc m n ⊆ Set.range (f a ∘ f a)) :
  1 - Real.exp (-1) ≤ a ∧ a < 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1123_112364


namespace NUMINAMATH_CALUDE_roots_of_equation_l1123_112335

theorem roots_of_equation : 
  let f : ℝ → ℝ := λ x => x * (x - 3)^2 * (5 + x)
  {x : ℝ | f x = 0} = {0, 3, -5} := by
sorry

end NUMINAMATH_CALUDE_roots_of_equation_l1123_112335


namespace NUMINAMATH_CALUDE_remainder_problem_l1123_112373

theorem remainder_problem (d r : ℤ) : 
  d > 1 →
  1223 % d = r →
  1625 % d = r →
  2513 % d = r →
  d - r = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1123_112373


namespace NUMINAMATH_CALUDE_graph_x_squared_minus_y_squared_l1123_112366

/-- The graph of x^2 - y^2 = 0 consists of two intersecting lines in the real plane -/
theorem graph_x_squared_minus_y_squared (x y : ℝ) :
  x^2 - y^2 = 0 ↔ (y = x ∨ y = -x) :=
sorry

end NUMINAMATH_CALUDE_graph_x_squared_minus_y_squared_l1123_112366


namespace NUMINAMATH_CALUDE_maria_stamp_collection_l1123_112380

/-- Given that Maria has 40 stamps and wants to increase her collection by 20%,
    prove that she will have a total of 48 stamps. -/
theorem maria_stamp_collection (initial_stamps : ℕ) (increase_percentage : ℚ) : 
  initial_stamps = 40 → 
  increase_percentage = 20 / 100 → 
  initial_stamps + (initial_stamps * increase_percentage).floor = 48 := by
  sorry

end NUMINAMATH_CALUDE_maria_stamp_collection_l1123_112380


namespace NUMINAMATH_CALUDE_square_root_squared_l1123_112361

theorem square_root_squared (a : ℝ) (ha : 0 ≤ a) : (Real.sqrt a) ^ 2 = a := by
  sorry

end NUMINAMATH_CALUDE_square_root_squared_l1123_112361


namespace NUMINAMATH_CALUDE_f_composition_fixed_points_l1123_112357

def f (x : ℝ) : ℝ := x^2 - 5*x + 6

theorem f_composition_fixed_points :
  {x : ℝ | f (f x) = f x} = {0, 2, 3, 5} := by sorry

end NUMINAMATH_CALUDE_f_composition_fixed_points_l1123_112357


namespace NUMINAMATH_CALUDE_time_addition_and_digit_sum_l1123_112301

/-- Represents time in a 12-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  isPM : Bool

/-- Represents a duration of time -/
structure Duration where
  hours : Nat
  minutes : Nat
  seconds : Nat

def addTime (t : Time) (d : Duration) : Time :=
  sorry

def sumDigits (t : Time) : Nat :=
  sorry

theorem time_addition_and_digit_sum :
  let initialTime : Time := ⟨3, 25, 15, true⟩
  let duration : Duration := ⟨137, 59, 59⟩
  let newTime := addTime initialTime duration
  newTime = ⟨9, 25, 14, true⟩ ∧ sumDigits newTime = 21 := by
  sorry

end NUMINAMATH_CALUDE_time_addition_and_digit_sum_l1123_112301


namespace NUMINAMATH_CALUDE_salary_restoration_l1123_112386

theorem salary_restoration (original_salary : ℝ) (reduced_salary : ℝ) : 
  reduced_salary = original_salary * (1 - 0.5) → 
  reduced_salary * 2 = original_salary :=
by sorry

end NUMINAMATH_CALUDE_salary_restoration_l1123_112386


namespace NUMINAMATH_CALUDE_silver_car_percentage_l1123_112360

/-- Calculates the percentage of silver cars in a car dealership's inventory after a new shipment. -/
theorem silver_car_percentage
  (initial_cars : ℕ)
  (initial_silver_percentage : ℚ)
  (new_shipment : ℕ)
  (new_non_silver_percentage : ℚ)
  (h1 : initial_cars = 40)
  (h2 : initial_silver_percentage = 1/10)
  (h3 : new_shipment = 80)
  (h4 : new_non_silver_percentage = 1/4)
  : ∃ (result : ℚ), abs (result - 53333/100000) < 1/10000 ∧
    result = (initial_silver_percentage * initial_cars + (1 - new_non_silver_percentage) * new_shipment) / (initial_cars + new_shipment) :=
by sorry

end NUMINAMATH_CALUDE_silver_car_percentage_l1123_112360


namespace NUMINAMATH_CALUDE_sample_size_is_75_l1123_112385

/-- Represents the sample size of a stratified sample -/
def sample_size (model_A_count : ℕ) (ratio_A ratio_B ratio_C : ℕ) : ℕ :=
  model_A_count * (ratio_A + ratio_B + ratio_C) / ratio_A

/-- Theorem stating that the sample size is 75 given the problem conditions -/
theorem sample_size_is_75 :
  sample_size 15 2 3 5 = 75 := by
  sorry

#eval sample_size 15 2 3 5

end NUMINAMATH_CALUDE_sample_size_is_75_l1123_112385


namespace NUMINAMATH_CALUDE_congruence_solution_l1123_112308

theorem congruence_solution (x : ℤ) :
  x ≡ 6 [ZMOD 17] → 15 * x + 2 ≡ 7 [ZMOD 17] := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l1123_112308


namespace NUMINAMATH_CALUDE_select_students_with_female_l1123_112389

/-- The number of male students -/
def num_male : ℕ := 5

/-- The number of female students -/
def num_female : ℕ := 2

/-- The total number of students to be selected -/
def num_selected : ℕ := 3

/-- The number of ways to select students with at least one female -/
def num_ways_with_female : ℕ := Nat.choose (num_male + num_female) num_selected - Nat.choose num_male num_selected

theorem select_students_with_female :
  num_ways_with_female = 25 := by
  sorry

end NUMINAMATH_CALUDE_select_students_with_female_l1123_112389


namespace NUMINAMATH_CALUDE_selling_price_proof_l1123_112399

/-- The selling price that results in the same profit as the loss -/
def selling_price : ℕ := 66

/-- The cost price of the article -/
def cost_price : ℕ := 59

/-- The price at which the article is sold at a loss -/
def loss_price : ℕ := 52

theorem selling_price_proof :
  (selling_price - cost_price = cost_price - loss_price) ∧
  (selling_price > cost_price) ∧
  (loss_price < cost_price) := by
  sorry

end NUMINAMATH_CALUDE_selling_price_proof_l1123_112399


namespace NUMINAMATH_CALUDE_english_math_only_count_l1123_112358

/-- The number of students taking at least one subject -/
def total_students : ℕ := 28

/-- The number of students taking Mathematics and History, but not English -/
def math_history_only : ℕ := 6

theorem english_math_only_count :
  ∀ (math_only english_math_only math_english_history english_history_only : ℕ),
  -- The number taking Mathematics and English only equals the number taking Mathematics only
  math_only = english_math_only →
  -- No student takes English only or History only
  -- Six students take Mathematics and History, but not English (already defined as math_history_only)
  -- The number taking English and History only is five times the number taking all three subjects
  english_history_only = 5 * math_english_history →
  -- The number taking all three subjects is even and non-zero
  math_english_history % 2 = 0 ∧ math_english_history > 0 →
  -- The total number of students is correct
  total_students = math_only + english_math_only + math_history_only + english_history_only + math_english_history →
  -- Prove that the number of students taking English and Mathematics only is 5
  english_math_only = 5 := by
sorry

end NUMINAMATH_CALUDE_english_math_only_count_l1123_112358


namespace NUMINAMATH_CALUDE_john_reps_per_set_l1123_112325

/-- Given the weight per rep, number of sets, and total weight moved,
    calculate the number of reps per set. -/
def reps_per_set (weight_per_rep : ℕ) (num_sets : ℕ) (total_weight : ℕ) : ℕ :=
  (total_weight / weight_per_rep) / num_sets

/-- Prove that under the given conditions, John does 10 reps per set. -/
theorem john_reps_per_set :
  let weight_per_rep : ℕ := 15
  let num_sets : ℕ := 3
  let total_weight : ℕ := 450
  reps_per_set weight_per_rep num_sets total_weight = 10 := by
sorry

end NUMINAMATH_CALUDE_john_reps_per_set_l1123_112325
