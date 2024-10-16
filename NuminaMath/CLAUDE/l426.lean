import Mathlib

namespace NUMINAMATH_CALUDE_kelsey_sister_age_difference_l426_42631

/-- Represents the age difference between Kelsey and her older sister -/
def age_difference (kelsey_birth_year : ℕ) (sister_birth_year : ℕ) : ℕ :=
  kelsey_birth_year - sister_birth_year

theorem kelsey_sister_age_difference :
  ∀ (kelsey_birth_year : ℕ) (sister_birth_year : ℕ),
  kelsey_birth_year + 25 = 1999 →
  sister_birth_year + 50 = 2021 →
  age_difference kelsey_birth_year sister_birth_year = 3 := by
  sorry

end NUMINAMATH_CALUDE_kelsey_sister_age_difference_l426_42631


namespace NUMINAMATH_CALUDE_license_plate_difference_l426_42624

/-- The number of possible letters in a license plate position -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate position -/
def num_digits : ℕ := 10

/-- The number of possible license plates for State A -/
def state_a_plates : ℕ := num_letters^5 * num_digits

/-- The number of possible license plates for State B -/
def state_b_plates : ℕ := num_letters^3 * num_digits^3

theorem license_plate_difference :
  state_a_plates - state_b_plates = 10123776 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_difference_l426_42624


namespace NUMINAMATH_CALUDE_soccer_games_per_month_l426_42641

/-- Given a total number of games and number of months in a season,
    calculate the number of games per month assuming equal distribution -/
def games_per_month (total_games : ℕ) (num_months : ℕ) : ℕ :=
  total_games / num_months

/-- Theorem: For 27 games over 3 months, there are 9 games per month -/
theorem soccer_games_per_month :
  games_per_month 27 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_soccer_games_per_month_l426_42641


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l426_42646

/-- The number of ways to choose a starting lineup from a basketball team -/
def number_of_lineups (total_players : ℕ) (center_players : ℕ) (lineup_size : ℕ) : ℕ :=
  center_players * (total_players - 1) * (total_players - 2) * (total_players - 3)

/-- Theorem stating that for a team of 12 players with 4 centers, there are 3960 ways to choose a starting lineup of 4 players -/
theorem basketball_lineup_combinations :
  number_of_lineups 12 4 4 = 3960 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l426_42646


namespace NUMINAMATH_CALUDE_analysis_method_seeks_sufficient_condition_l426_42632

/-- The type of condition sought in the analysis method for proving inequalities -/
inductive ConditionType
  | Necessary
  | Sufficient
  | NecessaryAndSufficient
  | NecessaryOrSufficient

/-- The analysis method for proving inequalities -/
structure AnalysisMethod where
  /-- The type of condition sought by the method -/
  condition_type : ConditionType

/-- Theorem: The analysis method for proving inequalities primarily seeks sufficient conditions -/
theorem analysis_method_seeks_sufficient_condition :
  ∀ (method : AnalysisMethod), method.condition_type = ConditionType.Sufficient :=
by
  sorry

end NUMINAMATH_CALUDE_analysis_method_seeks_sufficient_condition_l426_42632


namespace NUMINAMATH_CALUDE_line_canonical_equations_l426_42612

/-- The canonical equations of a line given by the intersection of two planes -/
theorem line_canonical_equations (x y z : ℝ) : 
  (x + 5*y - z + 11 = 0 ∧ x - y + 2*z - 1 = 0) → 
  ((x + 1)/9 = (y + 2)/(-3) ∧ (y + 2)/(-3) = z/(-6)) :=
by sorry

end NUMINAMATH_CALUDE_line_canonical_equations_l426_42612


namespace NUMINAMATH_CALUDE_peanut_butter_price_l426_42667

theorem peanut_butter_price 
  (spam_price : ℝ) 
  (bread_price : ℝ) 
  (spam_quantity : ℕ) 
  (peanut_butter_quantity : ℕ) 
  (bread_quantity : ℕ) 
  (total_paid : ℝ) 
  (h1 : spam_price = 3) 
  (h2 : bread_price = 2) 
  (h3 : spam_quantity = 12) 
  (h4 : peanut_butter_quantity = 3) 
  (h5 : bread_quantity = 4) 
  (h6 : total_paid = 59) :
  (total_paid - spam_price * spam_quantity - bread_price * bread_quantity) / peanut_butter_quantity = 5 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_price_l426_42667


namespace NUMINAMATH_CALUDE_temperature_difference_l426_42645

def highest_temp : ℝ := 10
def lowest_temp : ℝ := -1

theorem temperature_difference : highest_temp - lowest_temp = 11 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l426_42645


namespace NUMINAMATH_CALUDE_highest_salary_grade_is_six_l426_42618

/-- The minimum salary grade -/
def min_grade : ℕ := 1

/-- Function to calculate hourly wage based on salary grade -/
def hourly_wage (s : ℕ) : ℝ := 7.50 + 0.25 * (s - 1)

/-- The difference in hourly wage between the highest and lowest grade -/
def wage_difference : ℝ := 1.25

theorem highest_salary_grade_is_six :
  ∃ (max_grade : ℕ),
    (∀ (s : ℕ), min_grade ≤ s ∧ s ≤ max_grade) ∧
    (hourly_wage max_grade = hourly_wage min_grade + wage_difference) ∧
    max_grade = 6 :=
by sorry

end NUMINAMATH_CALUDE_highest_salary_grade_is_six_l426_42618


namespace NUMINAMATH_CALUDE_parking_cost_average_l426_42660

/-- Parking cost structure and calculation -/
theorem parking_cost_average (base_cost : ℝ) (base_hours : ℝ) (additional_cost : ℝ) (total_hours : ℝ) : 
  base_cost = 20 →
  base_hours = 2 →
  additional_cost = 1.75 →
  total_hours = 9 →
  (base_cost + (total_hours - base_hours) * additional_cost) / total_hours = 3.58 := by
sorry

end NUMINAMATH_CALUDE_parking_cost_average_l426_42660


namespace NUMINAMATH_CALUDE_least_five_digit_prime_congruent_to_7_mod_20_l426_42686

theorem least_five_digit_prime_congruent_to_7_mod_20 : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit number
  (n % 20 = 7) ∧              -- congruent to 7 (mod 20)
  Nat.Prime n ∧               -- prime number
  (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) → (m % 20 = 7) → Nat.Prime m → m ≥ n) ∧
  n = 10127 := by
sorry

end NUMINAMATH_CALUDE_least_five_digit_prime_congruent_to_7_mod_20_l426_42686


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l426_42628

theorem polynomial_division_quotient :
  let dividend := fun (z : ℚ) => 4 * z^5 + 2 * z^4 - 7 * z^3 + 5 * z^2 - 3 * z + 8
  let divisor := fun (z : ℚ) => 3 * z + 1
  let quotient := fun (z : ℚ) => (4/3) * z^4 - (19/3) * z^3 + (34/3) * z^2 - (61/9) * z - 1
  ∀ z : ℚ, dividend z = divisor z * quotient z + (275/27) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l426_42628


namespace NUMINAMATH_CALUDE_similar_numbers_same_outcome_l426_42619

/-- A number is good if the second player (Banana) has a winning strategy --/
def IsGood (k : ℕ) (n : ℕ) : Prop := sorry

theorem similar_numbers_same_outcome (k : ℕ) (n n' : ℕ) : 
  k ≥ 2 → n ≥ k → n' ≥ k → 
  (∀ p : ℕ, p.Prime → p ≤ k → (p ∣ n ↔ p ∣ n')) →
  (IsGood k n ↔ IsGood k n') := by sorry

end NUMINAMATH_CALUDE_similar_numbers_same_outcome_l426_42619


namespace NUMINAMATH_CALUDE_correlation_identification_l426_42604

-- Define the concept of a relationship
def Relationship : Type := Unit

-- Define specific relationships
def age_wealth : Relationship := ()
def curve_coordinates : Relationship := ()
def apple_production_climate : Relationship := ()
def tree_diameter_height : Relationship := ()

-- Define the property of being correlational
def is_correlational : Relationship → Prop := sorry

-- Define the property of being functional
def is_functional : Relationship → Prop := sorry

-- State that functional relationships are not correlational
axiom functional_not_correlational : 
  ∀ (r : Relationship), is_functional r → ¬is_correlational r

-- State the theorem
theorem correlation_identification :
  is_correlational age_wealth ∧
  is_correlational apple_production_climate ∧
  is_correlational tree_diameter_height ∧
  is_functional curve_coordinates :=
sorry

end NUMINAMATH_CALUDE_correlation_identification_l426_42604


namespace NUMINAMATH_CALUDE_sum_of_digits_l426_42600

def num1 : ℕ := 404040404040404040404040404040404040404040404040404040404040404040404040404040404040404040404040404
def num2 : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707

def product : ℕ := num1 * num2

def thousands_digit (n : ℕ) : ℕ := (n / 1000) % 10
def units_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits : 
  thousands_digit product + units_digit product = 10 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_l426_42600


namespace NUMINAMATH_CALUDE_intersection_points_distance_l426_42697

noncomputable section

-- Define the curve C in polar coordinates
def curve_C (a : ℝ) (θ : ℝ) : ℝ := 2 * Real.sin θ + 2 * a * Real.cos θ

-- Define the line l in parametric form
def line_l (t : ℝ) : ℝ × ℝ := (-2 + Real.sqrt 2 / 2 * t, Real.sqrt 2 / 2 * t)

-- Define point P in polar coordinates
def point_P : ℝ × ℝ := (2, Real.pi)

-- Theorem statement
theorem intersection_points_distance (a : ℝ) :
  a > 0 →
  ∃ (M N : ℝ × ℝ),
    (∃ (t₁ t₂ : ℝ), M = line_l t₁ ∧ N = line_l t₂) ∧
    (∃ (θ₁ θ₂ : ℝ), curve_C a θ₁ = Real.sqrt ((M.1)^2 + (M.2)^2) ∧
                    curve_C a θ₂ = Real.sqrt ((N.1)^2 + (N.2)^2)) ∧
    Real.sqrt ((M.1 - point_P.1)^2 + (M.2 - point_P.2)^2) +
    Real.sqrt ((N.1 - point_P.1)^2 + (N.2 - point_P.2)^2) = 5 * Real.sqrt 2 →
  a = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_intersection_points_distance_l426_42697


namespace NUMINAMATH_CALUDE_bridge_length_specific_bridge_length_l426_42621

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proof of the specific bridge length problem -/
theorem specific_bridge_length : 
  bridge_length 200 60 25 = 216.75 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_specific_bridge_length_l426_42621


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l426_42636

theorem quadratic_roots_relation (k : ℝ) : 
  (∀ x y : ℝ, x^2 + k*x + 12 = 0 ∧ y^2 - k*y + 12 = 0 → y = x + 7) → 
  k = -7 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l426_42636


namespace NUMINAMATH_CALUDE_determinant_in_terms_of_r_s_t_l426_42670

theorem determinant_in_terms_of_r_s_t (r s t : ℝ) (a b c : ℝ) : 
  (a^3 - r*a^2 + s*a - t = 0) →
  (b^3 - r*b^2 + s*b - t = 0) →
  (c^3 - r*c^2 + s*c - t = 0) →
  (a + b + c = r) →
  (a*b + a*c + b*c = s) →
  (a*b*c = t) →
  Matrix.det !![2+a, 2, 2; 2, 2+b, 2; 2, 2, 2+c] = t - 2*s := by
sorry

end NUMINAMATH_CALUDE_determinant_in_terms_of_r_s_t_l426_42670


namespace NUMINAMATH_CALUDE_solve_for_q_l426_42678

theorem solve_for_q (n m q : ℚ) 
  (eq1 : (3 : ℚ) / 4 = n / 88)
  (eq2 : (3 : ℚ) / 4 = (m + n) / 100)
  (eq3 : (3 : ℚ) / 4 = (q - m) / 150) :
  q = 121.5 := by
sorry

end NUMINAMATH_CALUDE_solve_for_q_l426_42678


namespace NUMINAMATH_CALUDE_set_union_implies_a_zero_l426_42668

theorem set_union_implies_a_zero (a : ℝ) : 
  let A : Set ℝ := {2^a, 3}
  let B : Set ℝ := {2, 3}
  A ∪ B = {1, 2, 3} → a = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_set_union_implies_a_zero_l426_42668


namespace NUMINAMATH_CALUDE_f_properties_l426_42613

-- Define the ⊕ operation
def oplus (a b : ℝ) : ℝ := a * b

-- Define the ⊗ operation
def otimes (a b : ℝ) : ℝ := a + b

-- Define the function f
def f (x : ℝ) : ℝ := otimes x 2 - oplus 2 x

-- Theorem statement
theorem f_properties :
  (¬ ∀ x, f (-x) = f x) ∧  -- not even
  (¬ ∀ x, f (-x) = -f x) ∧ -- not odd
  (∀ x y, x < y → f x > f y) -- decreasing
  := by sorry

end NUMINAMATH_CALUDE_f_properties_l426_42613


namespace NUMINAMATH_CALUDE_f_zero_at_one_f_zero_at_five_f_value_at_three_l426_42688

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := -2 * x^2 + 12 * x - 10

/-- The function f has a zero at x = 1 -/
theorem f_zero_at_one : f 1 = 0 := by sorry

/-- The function f has a zero at x = 5 -/
theorem f_zero_at_five : f 5 = 0 := by sorry

/-- The function f takes the value 8 when x = 3 -/
theorem f_value_at_three : f 3 = 8 := by sorry

end NUMINAMATH_CALUDE_f_zero_at_one_f_zero_at_five_f_value_at_three_l426_42688


namespace NUMINAMATH_CALUDE_dot_product_ab_bc_l426_42699

/-- Given two vectors AB and AC in 2D space, prove that their dot product with BC is -8. -/
theorem dot_product_ab_bc (AB AC : ℝ × ℝ) (h1 : AB = (4, 2)) (h2 : AC = (1, 4)) :
  AB • (AC - AB) = -8 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_ab_bc_l426_42699


namespace NUMINAMATH_CALUDE_complex_distance_bounds_l426_42654

theorem complex_distance_bounds (z : ℂ) (h : Complex.abs (z + 2 - 2*Complex.I) = 1) :
  (∃ w : ℂ, Complex.abs (w + 2 - 2*Complex.I) = 1 ∧ Complex.abs (w - 3 - 2*Complex.I) = 6) ∧
  (∃ v : ℂ, Complex.abs (v + 2 - 2*Complex.I) = 1 ∧ Complex.abs (v - 3 - 2*Complex.I) = 4) ∧
  (∀ u : ℂ, Complex.abs (u + 2 - 2*Complex.I) = 1 → 
    Complex.abs (u - 3 - 2*Complex.I) ≤ 6 ∧ Complex.abs (u - 3 - 2*Complex.I) ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_complex_distance_bounds_l426_42654


namespace NUMINAMATH_CALUDE_largest_fraction_l426_42605

theorem largest_fraction :
  let a := (1 : ℚ) / 3
  let b := (1 : ℚ) / 4
  let c := (3 : ℚ) / 8
  let d := (5 : ℚ) / 12
  let e := (7 : ℚ) / 24
  d > a ∧ d > b ∧ d > c ∧ d > e := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l426_42605


namespace NUMINAMATH_CALUDE_city_death_rate_l426_42651

/-- Represents the population dynamics of a city --/
structure CityPopulation where
  birth_rate : ℕ  -- Birth rate per two seconds
  net_increase : ℕ  -- Net population increase per day

/-- Calculates the death rate per two seconds given city population data --/
def death_rate (city : CityPopulation) : ℕ :=
  let seconds_per_day : ℕ := 24 * 60 * 60
  let birth_rate_per_second : ℕ := city.birth_rate / 2
  let net_increase_per_second : ℕ := city.net_increase / seconds_per_day
  2 * (birth_rate_per_second - net_increase_per_second)

/-- Theorem stating that for the given city data, the death rate is 6 people every two seconds --/
theorem city_death_rate :
  let city : CityPopulation := { birth_rate := 8, net_increase := 86400 }
  death_rate city = 6 := by
  sorry

end NUMINAMATH_CALUDE_city_death_rate_l426_42651


namespace NUMINAMATH_CALUDE_figure_to_square_l426_42643

/-- Represents a figure on a grid --/
structure GridFigure where
  width : ℕ
  height : ℕ
  area : ℕ

/-- Represents a part of the figure after cutting --/
structure FigurePart where
  area : ℕ

/-- Represents a square --/
structure Square where
  side_length : ℕ

/-- Function to cut the figure into parts --/
def cut_figure (f : GridFigure) (n : ℕ) : List FigurePart :=
  sorry

/-- Function to check if parts can form a square --/
def can_form_square (parts : List FigurePart) : Bool :=
  sorry

/-- Main theorem statement --/
theorem figure_to_square (f : GridFigure) 
  (h1 : f.width = 6) 
  (h2 : f.height = 6) 
  (h3 : f.area = 36) : 
  ∃ (parts : List FigurePart), 
    (parts.length = 4) ∧ 
    (∀ p ∈ parts, p.area = 9) ∧
    (can_form_square parts = true) :=
  sorry

end NUMINAMATH_CALUDE_figure_to_square_l426_42643


namespace NUMINAMATH_CALUDE_flag_design_count_l426_42694

/-- The number of possible colors for each stripe -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag -/
def num_stripes : ℕ := 3

/-- The number of possible flag designs -/
def num_flag_designs : ℕ := num_colors ^ num_stripes

theorem flag_design_count :
  num_flag_designs = 27 := by
  sorry

end NUMINAMATH_CALUDE_flag_design_count_l426_42694


namespace NUMINAMATH_CALUDE_no_double_apply_1987_function_l426_42608

theorem no_double_apply_1987_function :
  ¬∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 1987 := by
  sorry

end NUMINAMATH_CALUDE_no_double_apply_1987_function_l426_42608


namespace NUMINAMATH_CALUDE_no_valid_a_exists_l426_42656

theorem no_valid_a_exists : ¬ ∃ (a n : ℕ), 
  a > 1 ∧ 
  n > 0 ∧ 
  ∃ (k : ℕ), a * (10^n + 1) = k * a^2 := by
sorry

end NUMINAMATH_CALUDE_no_valid_a_exists_l426_42656


namespace NUMINAMATH_CALUDE_car_wash_earnings_per_car_l426_42625

/-- Proves that a car wash company making $2000 in 5 days while cleaning 80 cars per day earns $5 per car --/
theorem car_wash_earnings_per_car 
  (cars_per_day : ℕ) 
  (total_days : ℕ) 
  (total_earnings : ℕ) 
  (h1 : cars_per_day = 80) 
  (h2 : total_days = 5) 
  (h3 : total_earnings = 2000) : 
  total_earnings / (cars_per_day * total_days) = 5 := by
sorry

end NUMINAMATH_CALUDE_car_wash_earnings_per_car_l426_42625


namespace NUMINAMATH_CALUDE_round_robin_tournament_l426_42655

theorem round_robin_tournament (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_round_robin_tournament_l426_42655


namespace NUMINAMATH_CALUDE_fifth_term_product_l426_42687

/-- Given an arithmetic sequence a and a geometric sequence b with specified initial terms,
    prove that the product of their 5th terms is 80. -/
theorem fifth_term_product (a b : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  (∀ n, b (n + 1) / b n = b 2 / b 1) →  -- geometric sequence condition
  a 1 = 1 → b 1 = 1 → 
  a 2 = 2 → b 2 = 2 → 
  a 5 * b 5 = 80 := by
sorry


end NUMINAMATH_CALUDE_fifth_term_product_l426_42687


namespace NUMINAMATH_CALUDE_f_at_5_l426_42692

/-- The polynomial function f(x) = 2x^5 - 5x^4 - 4x^3 + 3x^2 - 524 -/
def f (x : ℝ) : ℝ := 2*x^5 - 5*x^4 - 4*x^3 + 3*x^2 - 524

/-- Theorem: The value of f(5) is 2176 -/
theorem f_at_5 : f 5 = 2176 := by
  sorry

end NUMINAMATH_CALUDE_f_at_5_l426_42692


namespace NUMINAMATH_CALUDE_inequalities_hold_l426_42601

theorem inequalities_hold (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  ab ≤ 1 ∧ a^2 + b^2 ≥ 2 ∧ 1/a + 1/b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l426_42601


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_true_l426_42650

theorem quadratic_inequality_always_true (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + 4 * m * x - 4 < 0) ↔ -1 < m ∧ m ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_true_l426_42650


namespace NUMINAMATH_CALUDE_unique_modular_integer_l426_42658

theorem unique_modular_integer : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2023 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_integer_l426_42658


namespace NUMINAMATH_CALUDE_monthly_income_calculation_l426_42681

/-- Proves that if 32% of a person's monthly income is Rs. 3800, then their monthly income is Rs. 11875. -/
theorem monthly_income_calculation (deposit : ℝ) (percentage : ℝ) (monthly_income : ℝ) 
  (h1 : deposit = 3800)
  (h2 : percentage = 32)
  (h3 : deposit = (percentage / 100) * monthly_income) :
  monthly_income = 11875 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_calculation_l426_42681


namespace NUMINAMATH_CALUDE_lisa_quiz_goal_impossible_l426_42663

theorem lisa_quiz_goal_impossible (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (completed_quizzes : ℕ) (earned_as : ℕ) : 
  total_quizzes = 60 → 
  goal_percentage = 9/10 → 
  completed_quizzes = 40 → 
  earned_as = 30 → 
  ¬ ∃ (remaining_non_as : ℕ), 
    earned_as + (total_quizzes - completed_quizzes - remaining_non_as) ≥ 
    ⌈goal_percentage * total_quizzes⌉ := by
  sorry

#check lisa_quiz_goal_impossible

end NUMINAMATH_CALUDE_lisa_quiz_goal_impossible_l426_42663


namespace NUMINAMATH_CALUDE_no_cube_in_sequence_l426_42669

theorem no_cube_in_sequence : ∀ (n : ℕ), ¬ ∃ (k : ℤ), 2^(2^n) + 1 = k^3 := by sorry

end NUMINAMATH_CALUDE_no_cube_in_sequence_l426_42669


namespace NUMINAMATH_CALUDE_locus_of_T_and_min_distance_l426_42642

-- Define the circle A
def circle_A (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 12

-- Define point B
def point_B : ℝ × ℝ := (1, 0)

-- Define the locus Γ
def Γ (x y : ℝ) : Prop := x^2 / 3 + y^2 / 2 = 1

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

theorem locus_of_T_and_min_distance :
  ∃ (P : ℝ × ℝ) (T : ℝ × ℝ),
    (circle_A P.1 P.2) ∧
    (∃ (M N : ℝ × ℝ) (H : ℝ × ℝ),
      (Γ M.1 M.2) ∧ (Γ N.1 N.2) ∧
      (H = ((M.1 + N.1) / 2, (M.2 + N.2) / 2)) ∧
      (unit_circle H.1 H.2)) →
    ((∀ (x y : ℝ), (Γ x y) ↔ (x^2 / 3 + y^2 / 2 = 1)) ∧
     (∃ (d : ℝ),
       d = 2 * Real.sqrt 6 / 5 ∧
       ∀ (M N : ℝ × ℝ),
         (Γ M.1 M.2) → (Γ N.1 N.2) →
         (unit_circle ((M.1 + N.1) / 2) ((M.2 + N.2) / 2)) →
         d ≤ Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2))) :=
by sorry


end NUMINAMATH_CALUDE_locus_of_T_and_min_distance_l426_42642


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_roots_l426_42616

theorem isosceles_right_triangle_roots (a b z₁ z₂ : ℂ) : 
  z₁^2 + a*z₁ + b = 0 → 
  z₂^2 + a*z₂ + b = 0 → 
  z₁ ≠ z₂ →
  (z₂ - 0) • (z₁ - 0) = 0 →  -- Perpendicular condition
  Complex.abs (z₂ - 0) = Complex.abs (z₁ - z₂) →  -- Isosceles condition
  a^2 / b = 2*Real.sqrt 2 + 2*Complex.I*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_roots_l426_42616


namespace NUMINAMATH_CALUDE_divisible_by_six_l426_42653

theorem divisible_by_six (n : ℤ) : ∃ k : ℤ, n * (n^2 + 5) = 6 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_l426_42653


namespace NUMINAMATH_CALUDE_handshake_count_couples_handshakes_l426_42682

theorem handshake_count (n : ℕ) : n > 0 → 
  (2 * n) * (2 * n - 2) / 2 = n * (2 * n - 2) :=
fun h => by sorry

theorem couples_handshakes :
  let couples : ℕ := 8
  let total_people : ℕ := 2 * couples
  let handshakes_per_person : ℕ := total_people - 2
  total_people * handshakes_per_person / 2 = 112 :=
by sorry

end NUMINAMATH_CALUDE_handshake_count_couples_handshakes_l426_42682


namespace NUMINAMATH_CALUDE_marble_probability_l426_42683

theorem marble_probability (total_marbles : ℕ) (p_white p_green p_yellow : ℚ) :
  total_marbles = 250 →
  p_white = 2 / 5 →
  p_green = 1 / 4 →
  p_yellow = 1 / 10 →
  1 - (p_white + p_green + p_yellow) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l426_42683


namespace NUMINAMATH_CALUDE_female_students_count_l426_42629

theorem female_students_count (total : ℕ) (sample : ℕ) (female_diff : ℕ) 
  (h_total : total = 1600)
  (h_sample : sample = 200)
  (h_female_diff : female_diff = 20)
  (h_ratio : (sample - female_diff) / (sample + female_diff) = 9 / 11) :
  ∃ F : ℕ, F = 720 ∧ F + (total - F) = total := by
  sorry

end NUMINAMATH_CALUDE_female_students_count_l426_42629


namespace NUMINAMATH_CALUDE_lioness_weight_l426_42664

/-- The weight of a lioness given the weights of her cubs -/
theorem lioness_weight (L F M : ℝ) : 
  L = 6 * F →  -- The weight of the lioness is six times the weight of her female cub
  L = 4 * M →  -- The weight of the lioness is four times the weight of her male cub
  M - F = 14 → -- The difference between the weights of the male and female cub is 14 kg
  L = 168 :=   -- The weight of the lioness is 168 kg
by sorry

end NUMINAMATH_CALUDE_lioness_weight_l426_42664


namespace NUMINAMATH_CALUDE_unique_solution_l426_42617

def system_solution (x y : ℝ) : Prop :=
  2 * x + y = 3 ∧ x - 2 * y = -1

theorem unique_solution : 
  {p : ℝ × ℝ | system_solution p.1 p.2} = {(1, 1)} := by sorry

end NUMINAMATH_CALUDE_unique_solution_l426_42617


namespace NUMINAMATH_CALUDE_sin_210_degrees_l426_42647

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l426_42647


namespace NUMINAMATH_CALUDE_pie_eating_contest_l426_42615

theorem pie_eating_contest (student1_session1 student1_session2 student2_session1 student2_session2 : ℚ)
  (h1 : student1_session1 = 7/8)
  (h2 : student1_session2 = 3/4)
  (h3 : student2_session1 = 5/6)
  (h4 : student2_session2 = 2/3) :
  (student1_session1 + student1_session2) - (student2_session1 + student2_session2) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l426_42615


namespace NUMINAMATH_CALUDE_remaining_jelly_beans_l426_42679

/-- Represents the distribution of jelly beans based on ID endings -/
structure JellyBeanDistribution :=
  (group1 : Nat) (group2 : Nat) (group3 : Nat)
  (group4 : Nat) (group5 : Nat) (group6 : Nat)

/-- Calculates the total number of jelly beans drawn -/
def totalJellyBeansDrawn (dist : JellyBeanDistribution) : Nat :=
  dist.group1 * 2 + dist.group2 * 4 + dist.group3 * 6 +
  dist.group4 * 8 + dist.group5 * 10 + dist.group6 * 12

/-- Theorem stating the number of remaining jelly beans -/
theorem remaining_jelly_beans
  (initial_jelly_beans : Nat)
  (total_children : Nat)
  (allowed_percentage : Rat)
  (dist : JellyBeanDistribution) :
  initial_jelly_beans = 2000 →
  total_children = 100 →
  allowed_percentage = 70 / 100 →
  dist.group1 = 9 →
  dist.group2 = 25 →
  dist.group3 = 20 →
  dist.group4 = 15 →
  dist.group5 = 15 →
  dist.group6 = 14 →
  initial_jelly_beans - totalJellyBeansDrawn dist = 1324 := by
  sorry

end NUMINAMATH_CALUDE_remaining_jelly_beans_l426_42679


namespace NUMINAMATH_CALUDE_blue_faces_cube_l426_42614

theorem blue_faces_cube (n : ℕ) (h : n > 0) : 
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1/3 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_faces_cube_l426_42614


namespace NUMINAMATH_CALUDE_total_balloons_l426_42659

theorem total_balloons (tom_balloons sara_balloons : ℕ) 
  (h1 : tom_balloons = 9) 
  (h2 : sara_balloons = 8) : 
  tom_balloons + sara_balloons = 17 := by
sorry

end NUMINAMATH_CALUDE_total_balloons_l426_42659


namespace NUMINAMATH_CALUDE_converse_proposition_l426_42695

theorem converse_proposition : 
  (∀ x : ℝ, x = 1 → x^2 = 1) → 
  (∀ x : ℝ, x^2 = 1 → x = 1) :=
by sorry

end NUMINAMATH_CALUDE_converse_proposition_l426_42695


namespace NUMINAMATH_CALUDE_cosine_of_point_on_terminal_side_l426_42603

def point_on_terminal_side (α : Real) (x y : Real) : Prop :=
  ∃ t : Real, t > 0 ∧ x = t * Real.cos α ∧ y = t * Real.sin α

theorem cosine_of_point_on_terminal_side (α : Real) :
  point_on_terminal_side α (-3) 4 → Real.cos α = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cosine_of_point_on_terminal_side_l426_42603


namespace NUMINAMATH_CALUDE_problem_1_l426_42610

theorem problem_1 : 2 * Real.tan (45 * π / 180) + (-1/2)^0 + |Real.sqrt 3 - 1| = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l426_42610


namespace NUMINAMATH_CALUDE_modulus_of_complex_square_root_l426_42690

theorem modulus_of_complex_square_root (w : ℂ) (h : w^2 = -48 + 36*I) : 
  Complex.abs w = 2 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_square_root_l426_42690


namespace NUMINAMATH_CALUDE_parabola_tangent_slope_l426_42626

-- Define the parabola
def parabola (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 9

-- Define the derivative of the parabola
def parabola_derivative (a b : ℝ) (x : ℝ) : ℝ := 2 * a * x + b

theorem parabola_tangent_slope (a b : ℝ) :
  parabola a b 2 = -1 →
  parabola_derivative a b 2 = 1 →
  a = 3 ∧ b = -11 := by
sorry

end NUMINAMATH_CALUDE_parabola_tangent_slope_l426_42626


namespace NUMINAMATH_CALUDE_green_pepper_weight_l426_42630

def hannah_peppers (total_weight red_weight green_weight : Real) : Prop :=
  total_weight = 0.66 ∧ 
  red_weight = 0.33 ∧ 
  green_weight = total_weight - red_weight

theorem green_pepper_weight : 
  ∀ (total_weight red_weight green_weight : Real),
  hannah_peppers total_weight red_weight green_weight →
  green_weight = 0.33 :=
by
  sorry

end NUMINAMATH_CALUDE_green_pepper_weight_l426_42630


namespace NUMINAMATH_CALUDE_distance_after_eight_hours_l426_42622

/-- The distance between two trains after a given time -/
def distance_between_trains (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) : ℝ :=
  (speed2 - speed1) * time

/-- Theorem: The distance between two trains after 8 hours -/
theorem distance_after_eight_hours :
  distance_between_trains 11 31 8 = 160 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_eight_hours_l426_42622


namespace NUMINAMATH_CALUDE_parabola_increasing_condition_l426_42635

/-- The parabola y = (a-1)x^2 + 1 increases as x increases when x ≥ 0 if and only if a > 1 -/
theorem parabola_increasing_condition (a : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → (∀ h : ℝ, h > 0 → ((a - 1) * (x + h)^2 + 1) > ((a - 1) * x^2 + 1))) ↔ 
  a > 1 := by sorry

end NUMINAMATH_CALUDE_parabola_increasing_condition_l426_42635


namespace NUMINAMATH_CALUDE_plant_growth_probability_l426_42611

theorem plant_growth_probability (p_1m : ℝ) (p_2m : ℝ) 
  (h1 : p_1m = 0.8) 
  (h2 : p_2m = 0.4) : 
  p_2m / p_1m = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_plant_growth_probability_l426_42611


namespace NUMINAMATH_CALUDE_arrangement_theorem_l426_42661

/-- The number of people standing in a row -/
def n : ℕ := 7

/-- Calculate the number of ways person A and B can stand next to each other -/
def adjacent_AB : ℕ := sorry

/-- Calculate the number of ways person A and B can stand not next to each other -/
def not_adjacent_AB : ℕ := sorry

/-- Calculate the number of ways person A, B, and C can stand so that no two of them are next to each other -/
def no_two_adjacent_ABC : ℕ := sorry

/-- Calculate the number of ways person A, B, and C can stand so that at most two of them are not next to each other -/
def at_most_two_not_adjacent_ABC : ℕ := sorry

theorem arrangement_theorem :
  adjacent_AB = 1440 ∧
  not_adjacent_AB = 3600 ∧
  no_two_adjacent_ABC = 1440 ∧
  at_most_two_not_adjacent_ABC = 4320 := by sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l426_42661


namespace NUMINAMATH_CALUDE_log_identity_l426_42680

/-- Given real numbers a and b greater than 1 satisfying lg(a + b) = lg(a) + lg(b),
    prove that lg(a - 1) + lg(b - 1) = 0 and lg(1/a + 1/b) = 0 -/
theorem log_identity (a b : ℝ) (ha : a > 1) (hb : b > 1) 
    (h : Real.log (a + b) = Real.log a + Real.log b) :
  Real.log (a - 1) + Real.log (b - 1) = 0 ∧ Real.log (1/a + 1/b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_log_identity_l426_42680


namespace NUMINAMATH_CALUDE_largest_kappa_l426_42609

theorem largest_kappa : ∃ κ : ℝ, κ = 2 ∧ 
  (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
    a^2 + d^2 = b^2 + c^2 → 
    a^2 + b^2 + c^2 + d^2 ≥ a*c + κ*b*d + a*d) ∧ 
  (∀ κ' : ℝ, κ' > κ → 
    ∃ a b c d : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ 
      a^2 + d^2 = b^2 + c^2 ∧ 
      a^2 + b^2 + c^2 + d^2 < a*c + κ'*b*d + a*d) :=
by sorry

end NUMINAMATH_CALUDE_largest_kappa_l426_42609


namespace NUMINAMATH_CALUDE_two_rats_through_wall_l426_42607

/-- The sum of lengths burrowed by two rats in n days -/
def S (n : ℕ) : ℚ :=
  (2^n - 1) + (2 - 1/(2^(n-1)))

/-- The problem statement -/
theorem two_rats_through_wall : S 5 = 32 + 15/16 := by
  sorry

end NUMINAMATH_CALUDE_two_rats_through_wall_l426_42607


namespace NUMINAMATH_CALUDE_cos_105_degrees_l426_42677

theorem cos_105_degrees : 
  Real.cos (105 * π / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_105_degrees_l426_42677


namespace NUMINAMATH_CALUDE_cary_walk_distance_l426_42627

/-- The number of calories Cary burns per mile walked -/
def calories_per_mile : ℝ := 150

/-- The number of calories in the candy bar Cary eats -/
def candy_bar_calories : ℝ := 200

/-- Cary's net calorie deficit -/
def net_calorie_deficit : ℝ := 250

/-- The number of miles Cary walked round-trip -/
def miles_walked : ℝ := 3

theorem cary_walk_distance :
  miles_walked * calories_per_mile - candy_bar_calories = net_calorie_deficit :=
by sorry

end NUMINAMATH_CALUDE_cary_walk_distance_l426_42627


namespace NUMINAMATH_CALUDE_isosceles_triangle_altitude_ratio_l426_42623

/-- An isosceles triangle with base to side ratio 4:3 has its altitude dividing the side in ratio 2:1 -/
theorem isosceles_triangle_altitude_ratio :
  ∀ (a b h m n : ℝ),
  a > 0 → b > 0 → h > 0 → m > 0 → n > 0 →
  b = (4/3) * a →  -- base to side ratio is 4:3
  h^2 = a^2 - (b/2)^2 →  -- height formula
  a^2 = h^2 + m^2 →  -- right triangle formed by altitude
  a = m + n →  -- side divided by altitude
  m / n = 2 / 1 :=  -- ratio in which altitude divides the side
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_altitude_ratio_l426_42623


namespace NUMINAMATH_CALUDE_count_multiples_l426_42675

theorem count_multiples (n : ℕ) : 
  (Finset.filter (fun x => x % 7 = 0 ∧ x % 14 ≠ 0) (Finset.range 350)).card = 25 := by
  sorry

end NUMINAMATH_CALUDE_count_multiples_l426_42675


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l426_42620

theorem prime_sum_theorem (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → 
  p + q = r → 1 < p → p < q → p = 2 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l426_42620


namespace NUMINAMATH_CALUDE_factorization_sum_l426_42698

theorem factorization_sum (a b : ℤ) :
  (∀ x, 25 * x^2 - 155 * x - 150 = (5 * x + a) * (5 * x + b)) →
  a + 2 * b = 27 := by
sorry

end NUMINAMATH_CALUDE_factorization_sum_l426_42698


namespace NUMINAMATH_CALUDE_triangle_inequality_l426_42671

/-- For any triangle ABC with sides a, b, and c, the sum of squares of the sides
    is greater than or equal to 4√3 times the area of the triangle. -/
theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  let S := Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l426_42671


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_intersection_l426_42673

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Finds the intersection point of two line segments -/
def lineIntersection (p1 p2 p3 p4 : Point) : Point := sorry

theorem quadrilateral_diagonal_intersection 
  (q : Quadrilateral) 
  (hConvex : isConvex q) 
  (hAB : distance q.A q.B = 12)
  (hCD : distance q.C q.D = 15)
  (hAC : distance q.A q.C = 18)
  (E : Point)
  (hE : E = lineIntersection q.A q.C q.B q.D)
  (hAreas : triangleArea q.A E q.D = triangleArea q.B E q.C) :
  distance q.A E = 8 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_intersection_l426_42673


namespace NUMINAMATH_CALUDE_solve_equation_l426_42689

-- Define the * operation
def star (a b : ℝ) : ℝ := 3 * a - b

-- State the theorem
theorem solve_equation : ∃ x : ℝ, star 2 (star 5 x) = 1 ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l426_42689


namespace NUMINAMATH_CALUDE_rope_division_l426_42640

theorem rope_division (rope_length : ℝ) (num_parts : ℕ) (part_length : ℝ) :
  rope_length = 5 →
  num_parts = 4 →
  rope_length = num_parts * part_length →
  part_length = 1.25 := by
sorry

end NUMINAMATH_CALUDE_rope_division_l426_42640


namespace NUMINAMATH_CALUDE_baker_cakes_left_l426_42634

/-- Given a baker who made a total of 217 cakes and sold 145 of them,
    prove that the number of cakes left is 72. -/
theorem baker_cakes_left (total : ℕ) (sold : ℕ) (h1 : total = 217) (h2 : sold = 145) :
  total - sold = 72 := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_left_l426_42634


namespace NUMINAMATH_CALUDE_count_triples_eq_two_l426_42676

/-- The number of positive integer triples (x, y, z) satisfying x · y = 6 and y · z = 15 -/
def count_triples : Nat :=
  (Finset.filter (fun t : Nat × Nat × Nat =>
    t.1 * t.2.1 = 6 ∧ t.2.1 * t.2.2 = 15 ∧
    t.1 > 0 ∧ t.2.1 > 0 ∧ t.2.2 > 0)
    (Finset.product (Finset.range 7) (Finset.product (Finset.range 4) (Finset.range 16)))).card

theorem count_triples_eq_two : count_triples = 2 := by
  sorry

end NUMINAMATH_CALUDE_count_triples_eq_two_l426_42676


namespace NUMINAMATH_CALUDE_committee_rearrangements_l426_42606

def word : String := "COMMITTEE"

def vowels : List Char := ['O', 'I', 'E', 'E']
def consonants : List Char := ['C', 'M', 'M', 'T', 'T']

def vowel_arrangements : ℕ := 12
def consonant_m_positions : ℕ := 10
def consonant_t_positions : ℕ := 3

theorem committee_rearrangements :
  (vowel_arrangements * consonant_m_positions * consonant_t_positions) = 360 :=
sorry

end NUMINAMATH_CALUDE_committee_rearrangements_l426_42606


namespace NUMINAMATH_CALUDE_linda_needs_one_train_l426_42691

/-- The number of trains Linda currently has -/
def current_trains : ℕ := 31

/-- The number of trains Linda wants in each row -/
def trains_per_row : ℕ := 8

/-- The function to calculate the smallest number of additional trains needed -/
def additional_trains_needed (current : ℕ) (per_row : ℕ) : ℕ :=
  (per_row - current % per_row) % per_row

/-- The theorem stating that Linda needs 1 additional train -/
theorem linda_needs_one_train : 
  additional_trains_needed current_trains trains_per_row = 1 := by
  sorry

end NUMINAMATH_CALUDE_linda_needs_one_train_l426_42691


namespace NUMINAMATH_CALUDE_sequence_properties_l426_42672

def S (n : ℕ) : ℤ := -n^2 + 7*n

def a (n : ℕ) : ℤ := S n - S (n-1)

theorem sequence_properties :
  (∀ n : ℕ, a n = -2*n + 8) ∧
  (∀ n : ℕ, n > 4 → a n < 0) ∧
  (∀ n : ℕ, S n ≤ S 3 ∧ S n ≤ S 4) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l426_42672


namespace NUMINAMATH_CALUDE_width_to_length_ratio_l426_42684

/-- A rectangle represents a rectangular hall --/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Properties of the rectangular hall --/
def RectangleProperties (r : Rectangle) : Prop :=
  r.width > 0 ∧ 
  r.length > 0 ∧ 
  r.width * r.length = 450 ∧ 
  r.length - r.width = 15

/-- Theorem stating the ratio of width to length --/
theorem width_to_length_ratio (r : Rectangle) 
  (h : RectangleProperties r) : r.width / r.length = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_width_to_length_ratio_l426_42684


namespace NUMINAMATH_CALUDE_larger_integer_is_48_l426_42666

theorem larger_integer_is_48 (x : ℤ) (h1 : x > 0) : 
  (x : ℚ) / (4 * x : ℚ) = 1 / 4 → 
  ((x + 12 : ℚ) / (4 * x : ℚ) = 1 / 2) → 
  4 * x = 48 := by
sorry

end NUMINAMATH_CALUDE_larger_integer_is_48_l426_42666


namespace NUMINAMATH_CALUDE_total_money_is_36_l426_42637

/-- Given Joanna's money, calculate the total money of Joanna, her brother, and her sister -/
def total_money (joanna_money : ℕ) : ℕ :=
  joanna_money + 3 * joanna_money + joanna_money / 2

/-- Theorem: The total money of Joanna, her brother, and her sister is $36 when Joanna has $8 -/
theorem total_money_is_36 : total_money 8 = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_money_is_36_l426_42637


namespace NUMINAMATH_CALUDE_cloth_weaving_problem_l426_42644

theorem cloth_weaving_problem (a₁ a₃₀ n : ℝ) (h1 : a₁ = 5) (h2 : a₃₀ = 1) (h3 : n = 30) :
  n / 2 * (a₁ + a₃₀) = 90 := by
  sorry

end NUMINAMATH_CALUDE_cloth_weaving_problem_l426_42644


namespace NUMINAMATH_CALUDE_jellybean_problem_l426_42685

theorem jellybean_problem :
  ∃ n : ℕ, n ≥ 200 ∧ n % 17 = 15 ∧ ∀ m : ℕ, m ≥ 200 ∧ m % 17 = 15 → m ≥ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_jellybean_problem_l426_42685


namespace NUMINAMATH_CALUDE_sequence_constant_condition_general_term_l426_42638

/-- Fibonacci sequence -/
def fib : ℕ → ℤ
| 0 => 1
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- The sequence a_n -/
noncomputable def a (x y : ℝ) : ℕ → ℝ
| 0 => x
| 1 => y
| (n + 2) => (a x y (n + 1) * a x y n + 1) / (a x y (n + 1) + a x y n)

theorem sequence_constant_condition (x y : ℝ) :
  (∃ n₀ : ℕ, ∀ n ≥ n₀, a x y (n + 1) = a x y n) ↔ (abs x = 1 ∧ y ≠ -x) :=
sorry

theorem general_term (x y : ℝ) (n : ℕ) :
  a x y n = ((x + 1)^(fib (n - 2)) * (y + 1)^(fib (n - 1)) + (x + 1)^(fib (n - 2)) * (y - 1)^(fib (n - 1))) /
            ((x + 1)^(fib (n - 2)) * (y + 1)^(fib (n - 1)) - (x - 1)^(fib (n - 2)) * (y - 1)^(fib (n - 1))) :=
sorry

end NUMINAMATH_CALUDE_sequence_constant_condition_general_term_l426_42638


namespace NUMINAMATH_CALUDE_ratio_change_proof_l426_42633

theorem ratio_change_proof (x y a : ℚ) : 
  y = 40 →
  x / y = 3 / 4 →
  (x + a) / (y + a) = 4 / 5 →
  a = 10 := by
sorry

end NUMINAMATH_CALUDE_ratio_change_proof_l426_42633


namespace NUMINAMATH_CALUDE_equation_solution_l426_42657

theorem equation_solution (a b c d : ℝ) : 
  a^3 + b^3 + c^3 + a^2 + b^2 + 1 = d^2 + d + Real.sqrt (a + b + c - 2*d) →
  d = 1 ∨ d = -4/3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l426_42657


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l426_42652

/-- If (a + i) / (1 - i) is a pure imaginary number, then a = 1 -/
theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (a + I) / (1 - I) = I * b) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l426_42652


namespace NUMINAMATH_CALUDE_sin_cos_equality_implies_pi_quarter_l426_42674

theorem sin_cos_equality_implies_pi_quarter (x : Real) :
  x ∈ Set.Icc 0 Real.pi →
  Real.sin (x + Real.sin x) = Real.cos (x - Real.cos x) →
  x = Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_equality_implies_pi_quarter_l426_42674


namespace NUMINAMATH_CALUDE_coefficient_x_five_in_expansion_l426_42693

theorem coefficient_x_five_in_expansion (x : ℝ) :
  ∃ (a₀ a₁ a₂ a₃ a₄ a₆ a₇ a₈ a₉ a₁₀ : ℝ),
    (x^2 - 2*x + 2)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + (-592)*x^5 + 
                        a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 :=
by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_five_in_expansion_l426_42693


namespace NUMINAMATH_CALUDE_prism_min_faces_and_pyramid_min_vertices_l426_42696

/-- A prism is a three-dimensional geometric shape with two parallel polygonal bases and rectangular faces connecting corresponding edges of the bases. -/
structure Prism where
  bases : ℕ -- number of sides in each base
  height : ℝ
  mk_pos : height > 0

/-- A pyramid is a three-dimensional geometric shape with a polygonal base and triangular faces meeting at a point (apex). -/
structure Pyramid where
  base_sides : ℕ -- number of sides in the base
  height : ℝ
  mk_pos : height > 0

/-- The number of faces in a prism. -/
def Prism.num_faces (p : Prism) : ℕ := p.bases + 2

/-- The number of vertices in a pyramid. -/
def Pyramid.num_vertices (p : Pyramid) : ℕ := p.base_sides + 1

theorem prism_min_faces_and_pyramid_min_vertices :
  (∀ p : Prism, p.num_faces ≥ 5) ∧
  (∀ p : Pyramid, p.num_vertices ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_prism_min_faces_and_pyramid_min_vertices_l426_42696


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_is_correct_l426_42648

/-- The value of a cow in dollars -/
def cow_value : ℕ := 400

/-- The value of a sheep in dollars -/
def sheep_value : ℕ := 280

/-- A debt is resolvable if it can be expressed as a linear combination of cow and sheep values -/
def is_resolvable (debt : ℕ) : Prop :=
  ∃ (c s : ℤ), debt = c * cow_value + s * sheep_value

/-- The smallest positive resolvable debt -/
def smallest_resolvable_debt : ℕ := 40

theorem smallest_resolvable_debt_is_correct :
  (is_resolvable smallest_resolvable_debt) ∧
  (∀ d : ℕ, 0 < d → d < smallest_resolvable_debt → ¬(is_resolvable d)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_is_correct_l426_42648


namespace NUMINAMATH_CALUDE_solution_range_l426_42662

-- Define the new operation
def otimes (x y : ℝ) : ℝ := x * (1 - y)

-- Theorem statement
theorem solution_range (a : ℝ) :
  (∃ x : ℝ, otimes x (x - a) > 1) ↔ (a < -3 ∨ a > 1) :=
sorry

end NUMINAMATH_CALUDE_solution_range_l426_42662


namespace NUMINAMATH_CALUDE_system_solution_existence_l426_42602

/-- Given a system of equations:
    1. y = b - x²
    2. x² + y² + 2a² = 4 - 2a(x + y)
    This theorem states the condition on b for the existence of at least one solution (x, y)
    for some real number a. -/
theorem system_solution_existence (b : ℝ) : 
  (∃ (a x y : ℝ), y = b - x^2 ∧ x^2 + y^2 + 2*a^2 = 4 - 2*a*(x + y)) ↔ 
  b ≥ -2 * Real.sqrt 2 - 1/4 := by
sorry


end NUMINAMATH_CALUDE_system_solution_existence_l426_42602


namespace NUMINAMATH_CALUDE_max_profit_year_l426_42665

/-- Represents the financial model of the environmentally friendly building materials factory. -/
structure FactoryFinances where
  initialInvestment : ℕ
  firstYearOperatingCosts : ℕ
  annualOperatingCostsIncrease : ℕ
  annualRevenue : ℕ

/-- Calculates the net profit for a given year. -/
def netProfitAtYear (f : FactoryFinances) (year : ℕ) : ℤ :=
  (f.annualRevenue * year : ℤ) -
  (f.initialInvestment : ℤ) -
  (f.firstYearOperatingCosts * year : ℤ) -
  (f.annualOperatingCostsIncrease * (year * (year - 1) / 2) : ℤ)

/-- Theorem stating that the net profit reaches its maximum in the 10th year. -/
theorem max_profit_year (f : FactoryFinances)
  (h1 : f.initialInvestment = 720000)
  (h2 : f.firstYearOperatingCosts = 120000)
  (h3 : f.annualOperatingCostsIncrease = 40000)
  (h4 : f.annualRevenue = 500000) :
  ∀ y : ℕ, y ≠ 10 → netProfitAtYear f y ≤ netProfitAtYear f 10 :=
sorry

end NUMINAMATH_CALUDE_max_profit_year_l426_42665


namespace NUMINAMATH_CALUDE_rental_distance_theorem_l426_42639

/-- Calculates the distance driven given rental parameters and total cost -/
def distance_driven (daily_rate : ℚ) (mile_rate : ℚ) (total_cost : ℚ) : ℚ :=
  (total_cost - daily_rate) / mile_rate

theorem rental_distance_theorem (daily_rate mile_rate total_cost : ℚ) :
  daily_rate = 29 →
  mile_rate = 0.08 →
  total_cost = 46.12 →
  distance_driven daily_rate mile_rate total_cost = 214 := by
  sorry

end NUMINAMATH_CALUDE_rental_distance_theorem_l426_42639


namespace NUMINAMATH_CALUDE_complex_ratio_max_value_l426_42649

theorem complex_ratio_max_value (z : ℂ) (h : Complex.abs z = 2) :
  (Complex.abs (z^2 - z + 1)) / (Complex.abs (2*z - 1 - Complex.I * Real.sqrt 3)) ≤ 3/2 ∧
  ∃ w : ℂ, Complex.abs w = 2 ∧
    (Complex.abs (w^2 - w + 1)) / (Complex.abs (2*w - 1 - Complex.I * Real.sqrt 3)) = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_complex_ratio_max_value_l426_42649
