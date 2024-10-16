import Mathlib

namespace NUMINAMATH_CALUDE_bicycle_distance_l758_75838

/-- The distance traveled by a bicycle in 30 minutes, given that it travels 1/2 as fast as a motorcycle going 90 miles per hour -/
theorem bicycle_distance (motorcycle_speed : ℝ) (bicycle_speed_ratio : ℝ) (time : ℝ) :
  motorcycle_speed = 90 →
  bicycle_speed_ratio = 1/2 →
  time = 1/2 →
  bicycle_speed_ratio * motorcycle_speed * time = 22.5 :=
by sorry

end NUMINAMATH_CALUDE_bicycle_distance_l758_75838


namespace NUMINAMATH_CALUDE_jennifer_museum_trips_l758_75861

/-- Calculates the total miles traveled for round trips to two museums -/
def total_miles_traveled (distance1 distance2 : ℕ) : ℕ :=
  2 * distance1 + 2 * distance2

/-- Theorem: Jennifer travels 40 miles in total to visit both museums -/
theorem jennifer_museum_trips : total_miles_traveled 5 15 = 40 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_museum_trips_l758_75861


namespace NUMINAMATH_CALUDE_gillian_spending_theorem_l758_75898

/-- Calculates the total amount Gillian spent at the farmer's market after tax -/
def gillian_total_spending (sandi_initial: ℝ) (sandi_market_fraction: ℝ) (sandi_discount: ℝ) 
  (gillian_extra: ℝ) (gillian_tax: ℝ) : ℝ :=
  let sandi_market := sandi_initial * sandi_market_fraction
  let sandi_after_discount := sandi_market * (1 - sandi_discount)
  let gillian_before_tax := 3 * sandi_after_discount + gillian_extra
  gillian_before_tax * (1 + gillian_tax)

/-- Theorem stating that Gillian's total spending at the farmer's market after tax is $957 -/
theorem gillian_spending_theorem :
  gillian_total_spending 600 0.5 0.2 150 0.1 = 957 := by
  sorry

end NUMINAMATH_CALUDE_gillian_spending_theorem_l758_75898


namespace NUMINAMATH_CALUDE_min_socks_for_pair_l758_75888

theorem min_socks_for_pair (n : ℕ) (h : n = 2019) : ∃ m : ℕ, m = n + 1 ∧ 
  (∀ k : ℕ, k < m → ∃ f : Fin k → Fin n, Function.Injective f) ∧
  (∀ g : Fin m → Fin n, ¬Function.Injective g) :=
by
  sorry

end NUMINAMATH_CALUDE_min_socks_for_pair_l758_75888


namespace NUMINAMATH_CALUDE_chores_per_week_l758_75869

theorem chores_per_week 
  (cookie_price : ℕ) 
  (cookies_per_pack : ℕ) 
  (budget : ℕ) 
  (cookies_per_chore : ℕ) 
  (weeks : ℕ) 
  (h1 : cookie_price = 3)
  (h2 : cookies_per_pack = 24)
  (h3 : budget = 15)
  (h4 : cookies_per_chore = 3)
  (h5 : weeks = 10)
  : (budget / cookie_price) * cookies_per_pack / weeks / cookies_per_chore = 4 := by
  sorry

end NUMINAMATH_CALUDE_chores_per_week_l758_75869


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l758_75896

/-- The problem statement -/
theorem min_reciprocal_sum (x y a b : ℝ) : 
  8 * x - y - 4 ≤ 0 →
  x + y + 1 ≥ 0 →
  y - 4 * x ≤ 0 →
  a > 0 →
  b > 0 →
  (∀ x' y', 8 * x' - y' - 4 ≤ 0 → x' + y' + 1 ≥ 0 → y' - 4 * x' ≤ 0 → a * x' + b * y' ≤ 2) →
  a * x + b * y = 2 →
  (∀ a' b', a' > 0 → b' > 0 → 
    (∀ x' y', 8 * x' - y' - 4 ≤ 0 → x' + y' + 1 ≥ 0 → y' - 4 * x' ≤ 0 → a' * x' + b' * y' ≤ 2) →
    1 / a + 1 / b ≤ 1 / a' + 1 / b') →
  1 / a + 1 / b = 9 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l758_75896


namespace NUMINAMATH_CALUDE_grass_field_width_l758_75834

/-- The width of a rectangular grass field with specific conditions -/
theorem grass_field_width : ∃ (w : ℝ), w = 40 ∧ w > 0 := by
  -- Define the length of the grass field
  let length : ℝ := 75

  -- Define the width of the path
  let path_width : ℝ := 2.5

  -- Define the cost per square meter of the path
  let cost_per_sqm : ℝ := 2

  -- Define the total cost of the path
  let total_cost : ℝ := 1200

  -- The width w satisfies the equation:
  -- 2 * (80 * (w + 5) - 75 * w) = 1200
  -- where 80 = length + 2 * path_width
  -- and 75 = length

  sorry

end NUMINAMATH_CALUDE_grass_field_width_l758_75834


namespace NUMINAMATH_CALUDE_min_y_value_l758_75854

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 14*x + 48*y) : y ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_min_y_value_l758_75854


namespace NUMINAMATH_CALUDE_money_division_l758_75889

theorem money_division (a b c : ℕ) (h1 : a = b / 2) (h2 : b = c / 2) (h3 : c = 400) :
  a + b + c = 700 := by
  sorry

end NUMINAMATH_CALUDE_money_division_l758_75889


namespace NUMINAMATH_CALUDE_angle_third_quadrant_tan_l758_75804

theorem angle_third_quadrant_tan (α : Real) : 
  (α > π ∧ α < 3*π/2) →  -- α is in the third quadrant
  -Real.cos α = 4/5 →    -- given condition
  Real.tan α = 3/4 :=    -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_angle_third_quadrant_tan_l758_75804


namespace NUMINAMATH_CALUDE_trajectory_of_Q_l758_75899

-- Define the points
def A : ℝ × ℝ := (0, 3)
def B : ℝ × ℝ := (0, 6)
def C : ℝ × ℝ := (0, -2)
def D : ℝ × ℝ := (0, 2)

-- Define the moving point P
def P : ℝ × ℝ → Prop :=
  λ p => ‖p - A‖ / ‖p - B‖ = 1 / 2

-- Define the perpendicular bisector of PC
def perpBisector (p : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ q => ‖q - p‖ = ‖q - C‖

-- Define point Q
def Q (p : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ q => perpBisector p q ∧ ∃ t : ℝ, q = p + t • (D - p)

-- State the theorem
theorem trajectory_of_Q :
  ∀ p : ℝ × ℝ, P p →
    ∀ q : ℝ × ℝ, Q p q →
      q.2^2 - q.1^2 / 3 = 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_Q_l758_75899


namespace NUMINAMATH_CALUDE_distance_between_trees_l758_75887

-- Define the yard length and number of trees
def yard_length : ℝ := 520
def num_trees : ℕ := 40

-- Theorem statement
theorem distance_between_trees :
  let num_spaces : ℕ := num_trees - 1
  let distance : ℝ := yard_length / num_spaces
  distance = 520 / 39 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l758_75887


namespace NUMINAMATH_CALUDE_smallest_divisible_by_12_20_6_sixty_divisible_by_12_20_6_sixty_is_smallest_l758_75852

theorem smallest_divisible_by_12_20_6 : ∀ n : ℕ, n > 0 → (12 ∣ n) → (20 ∣ n) → (6 ∣ n) → n ≥ 60 := by
  sorry

theorem sixty_divisible_by_12_20_6 : (12 ∣ 60) ∧ (20 ∣ 60) ∧ (6 ∣ 60) := by
  sorry

theorem sixty_is_smallest :
  ∀ n : ℕ, n > 0 → (12 ∣ n) → (20 ∣ n) → (6 ∣ n) → n = 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_12_20_6_sixty_divisible_by_12_20_6_sixty_is_smallest_l758_75852


namespace NUMINAMATH_CALUDE_perfect_square_power_of_two_l758_75886

theorem perfect_square_power_of_two (n : ℕ+) : 
  (∃ m : ℕ, 2^8 + 2^11 + 2^(n : ℕ) = m^2) ↔ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_power_of_two_l758_75886


namespace NUMINAMATH_CALUDE_inequality_solution_set_l758_75857

theorem inequality_solution_set (x : ℝ) : (2 * x - 1) / (3 * x + 1) > 0 ↔ x < -1/3 ∨ x > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l758_75857


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l758_75816

theorem solve_system_of_equations (b : ℝ) : 
  (∃ x : ℝ, 2 * x + 7 = 3 ∧ b * x - 10 = -2) → b = -4 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l758_75816


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l758_75833

theorem quadratic_equation_roots (a : ℝ) :
  let f : ℝ → ℝ := λ x => 4 * x^2 - 4 * (a + 2) * x + a^2 + 11
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ - x₂ = 3 → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l758_75833


namespace NUMINAMATH_CALUDE_target_row_sum_equals_2011_squared_l758_75879

/-- The row number where the sum of all numbers equals 2011² -/
def target_row : ℕ := 1006

/-- The number of elements in the nth row -/
def num_elements (n : ℕ) : ℕ := 2 * n - 1

/-- The sum of elements in the nth row -/
def row_sum (n : ℕ) : ℕ := (2 * n - 1)^2

/-- Theorem stating that the target_row is the row where the sum equals 2011² -/
theorem target_row_sum_equals_2011_squared :
  row_sum target_row = 2011^2 :=
sorry

end NUMINAMATH_CALUDE_target_row_sum_equals_2011_squared_l758_75879


namespace NUMINAMATH_CALUDE_line_equation_1_line_equation_2_line_equation_3_l758_75878

-- Define the types for points and lines
def Point := ℝ × ℝ
def Line := ℝ → ℝ → ℝ → Prop

-- Define what it means for a line to pass through a point
def passes_through (l : Line) (p : Point) : Prop :=
  l p.1 p.2 0

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, l1 x y 0 ↔ l2 (k * x) (k * y) 0

-- Define perpendicular lines
def perpendicular (l1 l2 : Line) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, l1 x y 0 ↔ l2 y (-x) 0

-- Define equal intercepts
def equal_intercepts (l : Line) : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ l a 0 0 ∧ l 0 a 0

-- Theorem 1
theorem line_equation_1 :
  let l : Line := λ x y c => x - 2 * y + 7 = c
  passes_through l (-1, 3) ∧ parallel l (λ x y c => x - 2 * y + 3 = c) := by sorry

-- Theorem 2
theorem line_equation_2 :
  let l : Line := λ x y c => x + 3 * y - 15 = c
  passes_through l (3, 4) ∧ perpendicular l (λ x y c => 3 * x - y + 2 = c) := by sorry

-- Theorem 3
theorem line_equation_3 :
  let l : Line := λ x y c => x + y - 3 = c
  passes_through l (1, 2) ∧ equal_intercepts l := by sorry

end NUMINAMATH_CALUDE_line_equation_1_line_equation_2_line_equation_3_l758_75878


namespace NUMINAMATH_CALUDE_statement_II_must_be_true_l758_75812

-- Define the possible contents of the card
inductive CardContent
| Number : Nat → CardContent
| Symbol : Char → CardContent

-- Define the statements
def statementI (c : CardContent) : Prop :=
  match c with
  | CardContent.Symbol _ => True
  | CardContent.Number _ => False

def statementII (c : CardContent) : Prop :=
  match c with
  | CardContent.Symbol '%' => False
  | _ => True

def statementIII (c : CardContent) : Prop :=
  c = CardContent.Number 3

def statementIV (c : CardContent) : Prop :=
  c ≠ CardContent.Number 4

-- Theorem statement
theorem statement_II_must_be_true :
  ∃ (c : CardContent),
    (statementI c ∧ statementII c ∧ statementIII c) ∨
    (statementI c ∧ statementII c ∧ statementIV c) ∨
    (statementI c ∧ statementIII c ∧ statementIV c) ∨
    (statementII c ∧ statementIII c ∧ statementIV c) :=
  sorry

end NUMINAMATH_CALUDE_statement_II_must_be_true_l758_75812


namespace NUMINAMATH_CALUDE_count_solutions_eq_4n_l758_75820

/-- The number of integer solutions (x, y) for |x| + |y| = n -/
def count_solutions (n : ℕ) : ℕ :=
  4 * n

/-- Theorem: For any positive integer n, the number of integer solutions (x, y) 
    satisfying |x| + |y| = n is equal to 4n -/
theorem count_solutions_eq_4n (n : ℕ) (hn : n > 0) : 
  count_solutions n = 4 * n := by sorry

end NUMINAMATH_CALUDE_count_solutions_eq_4n_l758_75820


namespace NUMINAMATH_CALUDE_problem_solution_l758_75842

theorem problem_solution (a e : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * e) : e = 49 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l758_75842


namespace NUMINAMATH_CALUDE_number_division_result_l758_75872

theorem number_division_result (x : ℝ) (h : 5 * x = 100) : x / 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_division_result_l758_75872


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l758_75819

-- Define the equation
def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 / (1 + m) + y^2 / (1 - m) = 1

-- Define what it means for the equation to represent a hyperbola
def is_hyperbola (m : ℝ) : Prop :=
  ∃ x y : ℝ, hyperbola_equation x y m ∧ 
  (1 + m > 0 ∧ 1 - m < 0) ∨ (1 + m < 0 ∧ 1 - m > 0)

-- Theorem stating the range of m for which the equation represents a hyperbola
theorem hyperbola_m_range :
  ∀ m : ℝ, is_hyperbola m ↔ m < -1 ∨ m > 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l758_75819


namespace NUMINAMATH_CALUDE_office_canteen_chairs_l758_75855

/-- The number of round tables in the office canteen -/
def num_round_tables : ℕ := 2

/-- The number of rectangular tables in the office canteen -/
def num_rectangular_tables : ℕ := 2

/-- The number of chairs per round table -/
def chairs_per_round_table : ℕ := 6

/-- The number of chairs per rectangular table -/
def chairs_per_rectangular_table : ℕ := 7

/-- The total number of chairs in the office canteen -/
def total_chairs : ℕ := num_round_tables * chairs_per_round_table + num_rectangular_tables * chairs_per_rectangular_table

theorem office_canteen_chairs : total_chairs = 26 := by
  sorry

end NUMINAMATH_CALUDE_office_canteen_chairs_l758_75855


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l758_75817

theorem quadratic_equation_roots (k : ℚ) :
  (∃ x, 5 * x^2 + k * x - 6 = 0 ∧ x = 2) →
  (∃ y, 5 * y^2 + k * y - 6 = 0 ∧ y = -3/5) ∧
  k = -7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l758_75817


namespace NUMINAMATH_CALUDE_line_length_difference_is_correct_l758_75808

/-- The length of the white line in inches -/
def white_line_length : ℝ := 7.678934

/-- The length of the blue line in inches -/
def blue_line_length : ℝ := 3.33457689

/-- The difference between the white and blue line lengths -/
def line_length_difference : ℝ := white_line_length - blue_line_length

/-- Theorem stating that the difference between the white and blue line lengths is 4.34435711 inches -/
theorem line_length_difference_is_correct : 
  line_length_difference = 4.34435711 := by sorry

end NUMINAMATH_CALUDE_line_length_difference_is_correct_l758_75808


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_23_l758_75822

theorem modular_inverse_of_5_mod_23 : ∃ x : ℕ, x < 23 ∧ (5 * x) % 23 = 1 :=
by
  use 14
  constructor
  · norm_num
  · norm_num

#eval (5 * 14) % 23  -- This should output 1

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_23_l758_75822


namespace NUMINAMATH_CALUDE_problem_statement_l758_75849

theorem problem_statement (x : ℝ) (h : x = -1) : 
  2 * (-x^2 + 3*x^3) - (2*x^3 - 2*x^2) + 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l758_75849


namespace NUMINAMATH_CALUDE_division_problem_l758_75863

theorem division_problem (k : ℕ) (h : k = 14) : 56 / k = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l758_75863


namespace NUMINAMATH_CALUDE_coefficient_of_x_plus_two_to_ten_l758_75867

theorem coefficient_of_x_plus_two_to_ten (x : ℝ) :
  ∃ (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ),
    (x + 1)^2 + (x + 1)^11 = a + a₁*(x + 2) + a₂*(x + 2)^2 + a₃*(x + 2)^3 + 
      a₄*(x + 2)^4 + a₅*(x + 2)^5 + a₆*(x + 2)^6 + a₇*(x + 2)^7 + 
      a₈*(x + 2)^8 + a₉*(x + 2)^9 + a₁₀*(x + 2)^10 + a₁₁*(x + 2)^11 ∧
    a₁₀ = -11 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_plus_two_to_ten_l758_75867


namespace NUMINAMATH_CALUDE_stratified_sampling_best_l758_75894

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | RandomNumberTable
  | Stratified

/-- Represents a high school population -/
structure HighSchoolPopulation where
  grades : List Nat
  students : Nat

/-- Represents a survey goal -/
inductive SurveyGoal
  | PsychologicalPressure

/-- Determines the best sampling method given a high school population and survey goal -/
def bestSamplingMethod (population : HighSchoolPopulation) (goal : SurveyGoal) : SamplingMethod :=
  sorry

/-- Theorem stating that stratified sampling is the best method for the given scenario -/
theorem stratified_sampling_best 
  (population : HighSchoolPopulation) 
  (h1 : population.grades.length > 1) 
  (goal : SurveyGoal) 
  (h2 : goal = SurveyGoal.PsychologicalPressure) : 
  bestSamplingMethod population goal = SamplingMethod.Stratified :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_best_l758_75894


namespace NUMINAMATH_CALUDE_stickers_per_page_l758_75850

theorem stickers_per_page (total_pages : ℕ) (remaining_stickers : ℕ) : 
  total_pages = 12 →
  remaining_stickers = 220 →
  (total_pages - 1) * (remaining_stickers / (total_pages - 1)) = remaining_stickers →
  remaining_stickers / (total_pages - 1) = 20 := by
sorry

end NUMINAMATH_CALUDE_stickers_per_page_l758_75850


namespace NUMINAMATH_CALUDE_unique_monic_quadratic_with_complex_root_l758_75877

/-- A monic quadratic polynomial with real coefficients -/
def MonicQuadraticPolynomial (a b : ℝ) : ℂ → ℂ := fun x ↦ x^2 + a*x + b

theorem unique_monic_quadratic_with_complex_root :
  ∃! (a b : ℝ), (MonicQuadraticPolynomial a b) (2 - 3*I) = 0 ∧
                a = -4 ∧ b = 13 := by
  sorry

end NUMINAMATH_CALUDE_unique_monic_quadratic_with_complex_root_l758_75877


namespace NUMINAMATH_CALUDE_system_solutions_l758_75843

/-- The system of equations has two solutions with distance 10 between them -/
theorem system_solutions (a : ℝ) : 
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁^2 + y₁^2 = 26 * (y₁ * Real.sin (2 * a) - x₁ * Real.cos (2 * a))) ∧
    (x₁^2 + y₁^2 = 26 * (y₁ * Real.cos (3 * a) - x₁ * Real.sin (3 * a))) ∧
    (x₂^2 + y₂^2 = 26 * (y₂ * Real.sin (2 * a) - x₂ * Real.cos (2 * a))) ∧
    (x₂^2 + y₂^2 = 26 * (y₂ * Real.cos (3 * a) - x₂ * Real.sin (3 * a))) ∧
    ((x₁ - x₂)^2 + (y₁ - y₂)^2 = 100)) ↔
  (∃ n : ℤ, 
    (a = π / 10 + 2 * π * n / 5) ∨
    (a = π / 10 + (2 / 5) * Real.arctan (12 / 5) + 2 * π * n / 5) ∨
    (a = π / 10 - (2 / 5) * Real.arctan (12 / 5) + 2 * π * n / 5)) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l758_75843


namespace NUMINAMATH_CALUDE_rational_identity_product_l758_75840

theorem rational_identity_product (M₁ M₂ : ℚ) : 
  (∀ x : ℚ, x ≠ 2 ∧ x ≠ 3 → (45 * x - 55) / (x^2 - 5*x + 6) = M₁ / (x - 2) + M₂ / (x - 3)) →
  M₁ * M₂ = 200 := by
sorry

end NUMINAMATH_CALUDE_rational_identity_product_l758_75840


namespace NUMINAMATH_CALUDE_tangent_point_x_coordinate_l758_75895

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

theorem tangent_point_x_coordinate 
  (a : ℝ) 
  (h1 : ∀ x, (deriv (f a)) x = (deriv (f a)) (-x))  -- f' is an odd function
  (h2 : ∃ x, (deriv (f a)) x = 3/2)  -- There exists a point with slope 3/2
  : ∃ x, (deriv (f a)) x = 3/2 ∧ x = Real.log ((3 + Real.sqrt 17) / 4) :=
sorry

end NUMINAMATH_CALUDE_tangent_point_x_coordinate_l758_75895


namespace NUMINAMATH_CALUDE_sum_of_cubes_l758_75818

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l758_75818


namespace NUMINAMATH_CALUDE_product_of_sum_and_difference_l758_75891

theorem product_of_sum_and_difference (x y : ℝ) : 
  x + y = 15 ∧ x - y = 11 → x * y = 26 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_difference_l758_75891


namespace NUMINAMATH_CALUDE_farmer_land_allocation_l758_75828

/-- Represents the ratios of land allocation for corn, sugar cane, and tobacco -/
structure LandRatio :=
  (corn : ℕ)
  (sugar_cane : ℕ)
  (tobacco : ℕ)

/-- The farmer's land allocation problem -/
theorem farmer_land_allocation
  (initial_ratio : LandRatio)
  (new_ratio : LandRatio)
  (tobacco_increase : ℕ) :
  initial_ratio.corn = 5 ∧
  initial_ratio.sugar_cane = 2 ∧
  initial_ratio.tobacco = 2 ∧
  new_ratio.corn = 2 ∧
  new_ratio.sugar_cane = 2 ∧
  new_ratio.tobacco = 5 ∧
  tobacco_increase = 450 →
  ∃ (total_land : ℕ), total_land = 1350 := by
sorry

end NUMINAMATH_CALUDE_farmer_land_allocation_l758_75828


namespace NUMINAMATH_CALUDE_min_sum_squares_l758_75868

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = 8 → x^2 + y^2 + z^2 ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l758_75868


namespace NUMINAMATH_CALUDE_arithmetic_mean_property_l758_75831

def number_set : List Nat := [9, 9999, 99999999, 999999999999, 9999999999999999, 99999999999999999999]

def arithmetic_mean (xs : List Nat) : Nat :=
  xs.sum / xs.length

def has_18_digits (n : Nat) : Prop :=
  n ≥ 10^17 ∧ n < 10^18

def all_digits_distinct (n : Nat) : Prop :=
  ∀ i j, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10

def does_not_contain_4 (n : Nat) : Prop :=
  ∀ i, (n / 10^i) % 10 ≠ 4

theorem arithmetic_mean_property :
  let mean := arithmetic_mean number_set
  has_18_digits mean ∧ all_digits_distinct mean ∧ does_not_contain_4 mean :=
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_property_l758_75831


namespace NUMINAMATH_CALUDE_sum_coordinates_of_B_is_zero_l758_75805

/-- Given that M(4,4) is the midpoint of segment AB and A(10,6), prove that the sum of coordinates of B is 0 -/
theorem sum_coordinates_of_B_is_zero (A B M : ℝ × ℝ) : 
  A = (10, 6) → M = (4, 4) → M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → B.1 + B.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_coordinates_of_B_is_zero_l758_75805


namespace NUMINAMATH_CALUDE_rectangular_field_area_l758_75835

/-- The area of a rectangular field with perimeter 70 meters and width 15 meters is 300 square meters. -/
theorem rectangular_field_area (perimeter width : ℝ) (h1 : perimeter = 70) (h2 : width = 15) :
  let length := (perimeter - 2 * width) / 2
  width * length = 300 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l758_75835


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l758_75892

theorem sum_of_a_and_b (a b : ℝ) (h1 : a + 2*b = 8) (h2 : 2*a + b = 4) : a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l758_75892


namespace NUMINAMATH_CALUDE_no_solution_eq1_unique_solution_eq2_l758_75810

-- Problem 1
theorem no_solution_eq1 : ¬∃ x : ℝ, (1 / (x - 2) + 3 = (1 - x) / (2 - x)) := by sorry

-- Problem 2
theorem unique_solution_eq2 : ∃! x : ℝ, (x / (x - 1) - 1 = 3 / (x^2 - 1)) ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_no_solution_eq1_unique_solution_eq2_l758_75810


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_sum_eighteen_makes_sum_integer_smallest_n_is_eighteen_l758_75870

theorem smallest_n_for_integer_sum : 
  ∀ n : ℕ+, (1/3 + 1/4 + 1/9 + 1/n : ℚ).isInt → n ≥ 18 := by
  sorry

theorem eighteen_makes_sum_integer : 
  (1/3 + 1/4 + 1/9 + 1/18 : ℚ).isInt := by
  sorry

theorem smallest_n_is_eighteen : 
  ∃! n : ℕ+, (1/3 + 1/4 + 1/9 + 1/n : ℚ).isInt ∧ ∀ m : ℕ+, (1/3 + 1/4 + 1/9 + 1/m : ℚ).isInt → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_sum_eighteen_makes_sum_integer_smallest_n_is_eighteen_l758_75870


namespace NUMINAMATH_CALUDE_larger_box_capacity_l758_75830

/-- Represents the capacity of a box in terms of volume and paperclips -/
structure Box where
  volume : ℝ
  paperclipCapacity : ℕ

/-- The maximum number of paperclips a box can hold -/
def maxPaperclips (b : Box) : ℕ := b.paperclipCapacity

theorem larger_box_capacity (smallBox largeBox : Box)
  (h1 : smallBox.volume = 20)
  (h2 : smallBox.paperclipCapacity = 80)
  (h3 : largeBox.volume = 100)
  (h4 : largeBox.paperclipCapacity = 380) :
  maxPaperclips largeBox = 380 := by
  sorry

end NUMINAMATH_CALUDE_larger_box_capacity_l758_75830


namespace NUMINAMATH_CALUDE_ellipse_points_equiv_target_set_l758_75844

/-- Represents an ellipse passing through (2,1) with a > b > 0 -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_a_gt_b : a > b
  h_passes_through : 4 / a^2 + 1 / b^2 = 1

/-- The set of points (x, y) on the ellipse satisfying |y| > 1 -/
def ellipse_points (e : Ellipse) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1 ∧ |p.2| > 1}

/-- The set of points (x, y) satisfying x^2 + y^2 < 5 and |y| > 1 -/
def target_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 < 5 ∧ |p.2| > 1}

/-- Theorem stating the equivalence of the two sets -/
theorem ellipse_points_equiv_target_set (e : Ellipse) :
  ellipse_points e = target_set := by sorry

end NUMINAMATH_CALUDE_ellipse_points_equiv_target_set_l758_75844


namespace NUMINAMATH_CALUDE_alexis_initial_budget_l758_75865

/-- Alexis's shopping expenses and remaining budget --/
structure ShoppingBudget where
  shirt : ℕ
  pants : ℕ
  coat : ℕ
  socks : ℕ
  belt : ℕ
  shoes : ℕ
  remaining : ℕ

/-- Calculate the initial budget given the shopping expenses and remaining amount --/
def initialBudget (s : ShoppingBudget) : ℕ :=
  s.shirt + s.pants + s.coat + s.socks + s.belt + s.shoes + s.remaining

/-- Alexis's actual shopping expenses and remaining budget --/
def alexisShopping : ShoppingBudget :=
  { shirt := 30
  , pants := 46
  , coat := 38
  , socks := 11
  , belt := 18
  , shoes := 41
  , remaining := 16 }

/-- Theorem stating that Alexis's initial budget was $200 --/
theorem alexis_initial_budget :
  initialBudget alexisShopping = 200 := by
  sorry

end NUMINAMATH_CALUDE_alexis_initial_budget_l758_75865


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_five_l758_75803

def sequence_u (n : ℕ) : ℚ :=
  sorry

theorem sum_of_coefficients_is_five :
  ∃ (a b c : ℚ),
    (∀ n : ℕ, sequence_u n = a * n^2 + b * n + c) ∧
    (sequence_u 1 = 5) ∧
    (∀ n : ℕ, sequence_u (n + 1) - sequence_u n = 3 + 4 * (n - 1)) ∧
    (a + b + c = 5) :=
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_five_l758_75803


namespace NUMINAMATH_CALUDE_cone_surface_area_l758_75875

/-- The surface area of a cone with given slant height and base circumference -/
theorem cone_surface_area (slant_height : ℝ) (base_circumference : ℝ) :
  slant_height = 2 →
  base_circumference = 2 * Real.pi →
  (π * (base_circumference / (2 * π))^2) + (π * (base_circumference / (2 * π)) * slant_height) = 3 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_surface_area_l758_75875


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l758_75876

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_parallel_planes 
  (m n : Line) (α β : Plane) 
  (h1 : perpendicular m α)
  (h2 : parallel_line_plane n β)
  (h3 : parallel_plane α β) :
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l758_75876


namespace NUMINAMATH_CALUDE_polygon_diagonals_theorem_l758_75848

theorem polygon_diagonals_theorem (n : ℕ) :
  n ≥ 3 →
  (n - 2 = 6) →
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_polygon_diagonals_theorem_l758_75848


namespace NUMINAMATH_CALUDE_tourist_tax_calculation_l758_75873

/-- Represents the tax system in Country B -/
structure TaxSystem where
  taxFreeLimit : ℝ
  bracket1Rate : ℝ
  bracket1Limit : ℝ
  bracket2Rate : ℝ
  bracket2Limit : ℝ
  bracket3Rate : ℝ
  electronicsRate : ℝ
  luxuryRate : ℝ
  studentDiscount : ℝ

/-- Represents a purchase made by a tourist -/
structure Purchase where
  totalValue : ℝ
  electronicsValue : ℝ
  luxuryValue : ℝ
  educationalValue : ℝ
  hasStudentID : Bool

def calculateTax (system : TaxSystem) (purchase : Purchase) : ℝ :=
  sorry

theorem tourist_tax_calculation (system : TaxSystem) (purchase : Purchase) :
  system.taxFreeLimit = 600 ∧
  system.bracket1Rate = 0.12 ∧
  system.bracket1Limit = 1000 ∧
  system.bracket2Rate = 0.18 ∧
  system.bracket2Limit = 1500 ∧
  system.bracket3Rate = 0.25 ∧
  system.electronicsRate = 0.05 ∧
  system.luxuryRate = 0.10 ∧
  system.studentDiscount = 0.05 ∧
  purchase.totalValue = 2100 ∧
  purchase.electronicsValue = 900 ∧
  purchase.luxuryValue = 820 ∧
  purchase.educationalValue = 380 ∧
  purchase.hasStudentID = true
  →
  calculateTax system purchase = 304 :=
by sorry

end NUMINAMATH_CALUDE_tourist_tax_calculation_l758_75873


namespace NUMINAMATH_CALUDE_acid_solution_concentration_l758_75824

/-- Proves that replacing half of a 50% acid solution with a solution of unknown concentration to obtain a 40% solution implies the unknown concentration is 30% -/
theorem acid_solution_concentration (original_concentration : ℝ) 
  (final_concentration : ℝ) (replaced_fraction : ℝ) (replacement_concentration : ℝ) :
  original_concentration = 50 →
  final_concentration = 40 →
  replaced_fraction = 0.5 →
  (1 - replaced_fraction) * original_concentration + replaced_fraction * replacement_concentration = 100 * final_concentration →
  replacement_concentration = 30 := by
sorry

end NUMINAMATH_CALUDE_acid_solution_concentration_l758_75824


namespace NUMINAMATH_CALUDE_union_necessary_not_sufficient_for_complement_l758_75874

universe u

variable {U : Type u}
variable (A B : Set U)

theorem union_necessary_not_sufficient_for_complement :
  (∀ (A B : Set U), B = Aᶜ → A ∪ B = Set.univ) ∧
  (∃ (A B : Set U), A ∪ B = Set.univ ∧ B ≠ Aᶜ) :=
sorry

end NUMINAMATH_CALUDE_union_necessary_not_sufficient_for_complement_l758_75874


namespace NUMINAMATH_CALUDE_return_trip_time_l758_75815

/-- Represents the flight scenario between two towns -/
structure FlightScenario where
  d : ℝ  -- distance between towns
  p : ℝ  -- plane speed in still air
  w : ℝ  -- wind speed
  t : ℝ  -- time for return trip in still air

/-- The conditions of the flight scenario -/
def flight_conditions (f : FlightScenario) : Prop :=
  f.w = (1/3) * f.p ∧  -- wind speed is one-third of plane speed
  f.d / (f.p - f.w) = 120 ∧  -- time against wind
  f.d / (f.p + f.w) = f.t - 20  -- time with wind

/-- The theorem to prove -/
theorem return_trip_time (f : FlightScenario) 
  (h : flight_conditions f) : f.d / (f.p + f.w) = 60 := by
  sorry

#check return_trip_time

end NUMINAMATH_CALUDE_return_trip_time_l758_75815


namespace NUMINAMATH_CALUDE_f_is_even_l758_75841

def f (x : ℝ) : ℝ := -3 * x^4

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by sorry

end NUMINAMATH_CALUDE_f_is_even_l758_75841


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_B_range_of_a_l758_75864

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2 * x - 4 ≥ x - 2}
def C (a : ℝ) : Set ℝ := {x | 2 * x + a > 0}

-- Theorem statements
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x < 3} := by sorry

theorem union_A_B : A ∪ B = {x : ℝ | x ≥ -1} := by sorry

theorem range_of_a (a : ℝ) (h : C a ∪ B = C a) : a > -4 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_B_range_of_a_l758_75864


namespace NUMINAMATH_CALUDE_edge_probability_first_20_rows_l758_75839

/-- Represents Pascal's Triangle up to a certain number of rows -/
structure PascalTriangle (n : ℕ) where
  total_elements : ℕ
  edge_numbers : ℕ

/-- The probability of selecting an edge number from Pascal's Triangle -/
def edge_probability (pt : PascalTriangle 20) : ℚ :=
  pt.edge_numbers / pt.total_elements

/-- Theorem: The probability of selecting an edge number from the first 20 rows of Pascal's Triangle is 13/70 -/
theorem edge_probability_first_20_rows :
  ∃ (pt : PascalTriangle 20),
    pt.total_elements = 210 ∧
    pt.edge_numbers = 39 ∧
    edge_probability pt = 13 / 70 := by
  sorry

end NUMINAMATH_CALUDE_edge_probability_first_20_rows_l758_75839


namespace NUMINAMATH_CALUDE_scientific_notation_125000_l758_75800

theorem scientific_notation_125000 : 
  125000 = 1.25 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_125000_l758_75800


namespace NUMINAMATH_CALUDE_bug_position_after_2015_jumps_l758_75866

/-- Represents the five points on the circle -/
inductive Point
  | one
  | two
  | three
  | four
  | five

/-- Determines if a point is odd-numbered -/
def isOdd (p : Point) : Bool :=
  match p with
  | Point.one => true
  | Point.two => false
  | Point.three => true
  | Point.four => false
  | Point.five => true

/-- Performs one jump according to the rules -/
def jump (p : Point) : Point :=
  match p with
  | Point.one => Point.three
  | Point.two => Point.three
  | Point.three => Point.five
  | Point.four => Point.five
  | Point.five => Point.two

/-- Performs n jumps starting from a given point -/
def jumpNTimes (start : Point) (n : Nat) : Point :=
  match n with
  | 0 => start
  | n + 1 => jump (jumpNTimes start n)

theorem bug_position_after_2015_jumps :
  jumpNTimes Point.five 2015 = Point.five :=
sorry

end NUMINAMATH_CALUDE_bug_position_after_2015_jumps_l758_75866


namespace NUMINAMATH_CALUDE_equidistant_from_axes_l758_75836

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the distance from a point to the x-axis
def distToXAxis (p : Point2D) : ℝ := |p.y|

-- Define the distance from a point to the y-axis
def distToYAxis (p : Point2D) : ℝ := |p.x|

-- State the theorem
theorem equidistant_from_axes (p : Point2D) :
  distToXAxis p = distToYAxis p ↔ |p.x| - |p.y| = 0 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_from_axes_l758_75836


namespace NUMINAMATH_CALUDE_dividend_calculation_l758_75893

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 38)
  (h2 : quotient = 19)
  (h3 : remainder = 7) :
  divisor * quotient + remainder = 729 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l758_75893


namespace NUMINAMATH_CALUDE_book_club_member_ratio_l758_75884

theorem book_club_member_ratio :
  ∀ (r p : ℕ), 
    r > 0 → p > 0 →
    (5 * r + 12 * p : ℚ) / (r + p) = 8 →
    (r : ℚ) / p = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_book_club_member_ratio_l758_75884


namespace NUMINAMATH_CALUDE_sum_powers_equality_l758_75832

theorem sum_powers_equality (a b c d : ℝ) 
  (sum_eq : a + b = c + d) 
  (cube_sum_eq : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 ∧ 
  ∃ (a b c d : ℝ), (a + b = c + d) ∧ (a^3 + b^3 = c^3 + d^3) ∧ (a^4 + b^4 ≠ c^4 + d^4) := by
  sorry

end NUMINAMATH_CALUDE_sum_powers_equality_l758_75832


namespace NUMINAMATH_CALUDE_f_min_at_neg_one_l758_75813

/-- The quadratic function we want to minimize -/
def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x - 2

/-- The theorem stating that f is minimized at x = -1 -/
theorem f_min_at_neg_one :
  ∀ x : ℝ, f (-1) ≤ f x :=
sorry

end NUMINAMATH_CALUDE_f_min_at_neg_one_l758_75813


namespace NUMINAMATH_CALUDE_horner_method_v3_horner_method_correct_l758_75847

def horner_polynomial (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

def horner_v3 (x : ℝ) : ℝ :=
  let v0 := 3
  let v1 := v0 * x + 5
  let v2 := v1 * x + 6
  v2 * x + 79

theorem horner_method_v3 :
  horner_v3 (-4) = -57 :=
by sorry

theorem horner_method_correct :
  horner_v3 (-4) = horner_polynomial (-4) :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v3_horner_method_correct_l758_75847


namespace NUMINAMATH_CALUDE_symmetry_of_expressions_l758_75827

-- Define a completely symmetric expression
def is_completely_symmetric (f : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ (a b c : ℝ), f a b c = f b a c ∧ f a b c = f a c b ∧ f a b c = f c b a

-- Define the three expressions
def expr1 (a b c : ℝ) : ℝ := (a - b)^2
def expr2 (a b c : ℝ) : ℝ := a * b + b * c + c * a
def expr3 (a b c : ℝ) : ℝ := a^2 * b + b^2 * c + c^2 * a

-- State the theorem
theorem symmetry_of_expressions :
  is_completely_symmetric expr1 ∧ 
  is_completely_symmetric expr2 ∧ 
  ¬is_completely_symmetric expr3 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_of_expressions_l758_75827


namespace NUMINAMATH_CALUDE_cos_tan_identity_l758_75837

theorem cos_tan_identity : 
  (Real.cos (10 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180))) / Real.cos (50 * π / 180) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_tan_identity_l758_75837


namespace NUMINAMATH_CALUDE_rectangular_plot_area_l758_75851

/-- The area of a rectangular plot with length thrice its breadth and breadth of 14 meters is 588 square meters. -/
theorem rectangular_plot_area (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  breadth = 14 →
  length = 3 * breadth →
  area = length * breadth →
  area = 588 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_area_l758_75851


namespace NUMINAMATH_CALUDE_minimum_guests_l758_75845

theorem minimum_guests (total_food : ℕ) (max_per_guest : ℕ) (h1 : total_food = 327) (h2 : max_per_guest = 2) :
  ∃ (min_guests : ℕ), min_guests = 164 ∧ min_guests * max_per_guest ≥ total_food ∧ (min_guests - 1) * max_per_guest < total_food :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_guests_l758_75845


namespace NUMINAMATH_CALUDE_adjacent_cells_difference_l758_75825

/-- A type representing a cell in an n × n grid --/
structure Cell (n : ℕ) where
  row : Fin n
  col : Fin n

/-- A function representing the placement of integers in the grid --/
def GridPlacement (n : ℕ) := Cell n → Fin (n^2)

/-- Two cells are adjacent if they share a side or a corner --/
def adjacent {n : ℕ} (c1 c2 : Cell n) : Prop :=
  (c1.row = c2.row ∧ c1.col.val + 1 = c2.col.val) ∨
  (c1.row = c2.row ∧ c2.col.val + 1 = c1.col.val) ∨
  (c1.col = c2.col ∧ c1.row.val + 1 = c2.row.val) ∨
  (c1.col = c2.col ∧ c2.row.val + 1 = c1.row.val) ∨
  (c1.row.val + 1 = c2.row.val ∧ c1.col.val + 1 = c2.col.val) ∨
  (c1.row.val + 1 = c2.row.val ∧ c2.col.val + 1 = c1.col.val) ∨
  (c2.row.val + 1 = c1.row.val ∧ c1.col.val + 1 = c2.col.val) ∨
  (c2.row.val + 1 = c1.row.val ∧ c2.col.val + 1 = c1.col.val)

/-- The main theorem --/
theorem adjacent_cells_difference {n : ℕ} (h : n > 0) (g : GridPlacement n) :
  ∃ (c1 c2 : Cell n), adjacent c1 c2 ∧ 
    ((g c1).val + n + 1 ≤ (g c2).val ∨ (g c2).val + n + 1 ≤ (g c1).val) :=
sorry

end NUMINAMATH_CALUDE_adjacent_cells_difference_l758_75825


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l758_75890

theorem function_inequality_implies_a_bound (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ 2 ∧ 0 ≤ x₂ ∧ x₂ ≤ 2 → 
    x₁^3 - 3*x₁ ≤ Real.exp x₂ - 2*a*x₂ + 2) → 
  a ≤ Real.exp 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l758_75890


namespace NUMINAMATH_CALUDE_remaining_slices_l758_75871

/-- Calculates the remaining slices of pie and cake after a weekend of consumption --/
theorem remaining_slices (initial_pies : ℕ) (initial_cake : ℕ) (slices_per_pie : ℕ) (slices_per_cake : ℕ)
  (friday_pie : ℕ) (friday_cake : ℕ) (saturday_pie_percent : ℚ) (saturday_cake_percent : ℚ)
  (sunday_morning_pie : ℕ) (sunday_morning_cake : ℕ) (sunday_evening_pie : ℕ) (sunday_evening_cake : ℕ) :
  initial_pies = 2 →
  initial_cake = 1 →
  slices_per_pie = 8 →
  slices_per_cake = 12 →
  friday_pie = 2 →
  friday_cake = 2 →
  saturday_pie_percent = 1/2 →
  saturday_cake_percent = 1/4 →
  sunday_morning_pie = 2 →
  sunday_morning_cake = 3 →
  sunday_evening_pie = 4 →
  sunday_evening_cake = 1 →
  ∃ (remaining_pie remaining_cake : ℕ),
    remaining_pie = 1 ∧ remaining_cake = 4 := by
  sorry

end NUMINAMATH_CALUDE_remaining_slices_l758_75871


namespace NUMINAMATH_CALUDE_hypotenuse_length_l758_75885

/-- Given a right triangle with an acute angle α and a circle of radius R
    touching the hypotenuse and the extensions of the two legs,
    the length of the hypotenuse is R * (1 - tan(α/2)) / cos(α) -/
theorem hypotenuse_length (α R : Real) (h1 : 0 < α ∧ α < π/2) (h2 : R > 0) :
  ∃ x, x > 0 ∧ x = R * (1 - Real.tan (α/2)) / Real.cos α :=
by sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l758_75885


namespace NUMINAMATH_CALUDE_simplify_expressions_l758_75802

theorem simplify_expressions :
  ((-2.48 + 4.33 + (-7.52) + (-4.33)) = -10) ∧
  ((7/13 * (-9) + 7/13 * (-18) + 7/13) = -14) ∧
  (-20 * (1/19) * 38 = -762) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l758_75802


namespace NUMINAMATH_CALUDE_notebooks_given_to_yujeong_l758_75826

/-- The number of notebooks Minyoung initially had -/
def initial_notebooks : ℕ := 17

/-- The number of notebooks Minyoung had left after giving some to Yujeong -/
def remaining_notebooks : ℕ := 8

/-- The number of notebooks Minyoung gave to Yujeong -/
def notebooks_given : ℕ := initial_notebooks - remaining_notebooks

theorem notebooks_given_to_yujeong :
  notebooks_given = 9 :=
sorry

end NUMINAMATH_CALUDE_notebooks_given_to_yujeong_l758_75826


namespace NUMINAMATH_CALUDE_gardening_project_total_cost_l758_75897

/-- The cost of the gardening project -/
def gardening_project_cost (
  num_rose_bushes : ℕ)
  (cost_per_rose_bush : ℕ)
  (gardener_hourly_rate : ℕ)
  (gardener_hours_per_day : ℕ)
  (gardener_days : ℕ)
  (soil_volume : ℕ)
  (soil_cost_per_unit : ℕ) : ℕ :=
  num_rose_bushes * cost_per_rose_bush +
  gardener_hourly_rate * gardener_hours_per_day * gardener_days +
  soil_volume * soil_cost_per_unit

/-- The theorem stating the total cost of the gardening project -/
theorem gardening_project_total_cost :
  gardening_project_cost 20 150 30 5 4 100 5 = 4100 := by
  sorry

end NUMINAMATH_CALUDE_gardening_project_total_cost_l758_75897


namespace NUMINAMATH_CALUDE_three_pair_prob_standard_deck_l758_75846

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (suits : Nat)

/-- Represents a "three pair" hand in poker -/
structure ThreePair :=
  (triplet_rank : Nat)
  (triplet_suit : Nat)
  (pair_rank : Nat)
  (pair_suit : Nat)

/-- The number of ways to choose 5 cards from a deck -/
def choose_five (d : Deck) : Nat :=
  Nat.choose d.cards 5

/-- The number of valid "three pair" hands -/
def count_three_pairs (d : Deck) : Nat :=
  d.ranks * d.suits * (d.ranks - 1) * d.suits

/-- The probability of getting a "three pair" hand -/
def three_pair_probability (d : Deck) : ℚ :=
  count_three_pairs d / choose_five d

/-- Theorem: The probability of a "three pair" in a standard deck is 2,496 / 2,598,960 -/
theorem three_pair_prob_standard_deck :
  three_pair_probability (Deck.mk 52 13 4) = 2496 / 2598960 := by
  sorry

end NUMINAMATH_CALUDE_three_pair_prob_standard_deck_l758_75846


namespace NUMINAMATH_CALUDE_unique_solution_implies_equal_absolute_values_l758_75883

theorem unique_solution_implies_equal_absolute_values (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∃! x, a * (x - a)^2 + b * (x - b)^2 = 0) → |a| = |b| :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_implies_equal_absolute_values_l758_75883


namespace NUMINAMATH_CALUDE_salary_increase_after_three_years_l758_75806

/-- The annual percentage increase in salary -/
def annual_increase : ℝ := 0.12

/-- The number of years of salary increase -/
def years : ℕ := 3

/-- The total percentage increase after a given number of years -/
def total_increase (y : ℕ) : ℝ := (1 + annual_increase) ^ y - 1

theorem salary_increase_after_three_years :
  ∃ ε > 0, abs (total_increase years - 0.4057) < ε :=
sorry

end NUMINAMATH_CALUDE_salary_increase_after_three_years_l758_75806


namespace NUMINAMATH_CALUDE_product_mod_five_l758_75829

theorem product_mod_five : (2023 * 2024 * 2025 * 2026) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_five_l758_75829


namespace NUMINAMATH_CALUDE_drivers_distance_comparison_l758_75811

/-- Conversion factor from miles to kilometers -/
def mile_to_km : ℝ := 1.60934

/-- Gervais's distance in miles per day -/
def gervais_miles_per_day : ℝ := 315

/-- Number of days Gervais drove -/
def gervais_days : ℕ := 3

/-- Gervais's speed in miles per hour -/
def gervais_speed : ℝ := 60

/-- Henri's total distance in miles -/
def henri_miles : ℝ := 1250

/-- Henri's speed in miles per hour -/
def henri_speed : ℝ := 50

/-- Madeleine's distance in miles per day -/
def madeleine_miles_per_day : ℝ := 100

/-- Number of days Madeleine drove -/
def madeleine_days : ℕ := 5

/-- Madeleine's speed in miles per hour -/
def madeleine_speed : ℝ := 40

/-- Calculate total distance driven by all three drivers in kilometers -/
def total_distance : ℝ :=
  (gervais_miles_per_day * gervais_days * mile_to_km) +
  (henri_miles * mile_to_km) +
  (madeleine_miles_per_day * madeleine_days * mile_to_km)

/-- Calculate Henri's distance in kilometers -/
def henri_distance : ℝ := henri_miles * mile_to_km

theorem drivers_distance_comparison :
  total_distance = 4337.16905 ∧
  henri_distance = 2011.675 ∧
  henri_distance > gervais_miles_per_day * gervais_days * mile_to_km ∧
  henri_distance > madeleine_miles_per_day * madeleine_days * mile_to_km :=
by sorry

end NUMINAMATH_CALUDE_drivers_distance_comparison_l758_75811


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l758_75862

def digit_sum (n : Nat) : Nat :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∀ n : Nat, 100 ≤ n ∧ n < 1000 ∧ n % 9 = 0 ∧ digit_sum n = 27 → n ≤ 999 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l758_75862


namespace NUMINAMATH_CALUDE_polynomial_expansion_l758_75853

theorem polynomial_expansion (x : ℝ) :
  (3*x^2 + 4*x + 8)*(x - 2) - (x - 2)*(x^2 + 5*x - 72) + (4*x - 15)*(x - 2)*(x + 6) =
  6*x^3 - 4*x^2 - 26*x + 20 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l758_75853


namespace NUMINAMATH_CALUDE_train_length_l758_75823

theorem train_length (t_platform : ℝ) (t_pole : ℝ) (l_platform : ℝ)
  (h1 : t_platform = 33)
  (h2 : t_pole = 18)
  (h3 : l_platform = 250) :
  ∃ l_train : ℝ, l_train = 300 ∧ (l_train + l_platform) / t_platform = l_train / t_pole :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l758_75823


namespace NUMINAMATH_CALUDE_ball_placement_theorem_l758_75881

/-- The number of ways to place balls in boxes under different conditions -/
theorem ball_placement_theorem :
  let n : ℕ := 4  -- number of balls and boxes
  -- 1. Distinct balls, exactly one empty box
  let distinct_one_empty : ℕ := n * (n - 1) * (n - 2) * 6
  -- 2. Identical balls, exactly one empty box
  let identical_one_empty : ℕ := n * (n - 1)
  -- 3. Distinct balls, empty boxes allowed
  let distinct_empty_allowed : ℕ := n^n
  -- 4. Identical balls, empty boxes allowed
  let identical_empty_allowed : ℕ := 
    1 + n * (n - 1) + (n * (n - 1) / 2) + n * (n - 1) / 2 + n
  ∀ (n : ℕ), n = 4 →
    (distinct_one_empty = 144) ∧
    (identical_one_empty = 12) ∧
    (distinct_empty_allowed = 256) ∧
    (identical_empty_allowed = 35) := by
  sorry

end NUMINAMATH_CALUDE_ball_placement_theorem_l758_75881


namespace NUMINAMATH_CALUDE_necessarily_negative_l758_75807

theorem necessarily_negative (y z : ℝ) (h1 : -1 < y) (h2 : y < 0) (h3 : 0 < z) (h4 : z < 1) :
  y - z < 0 := by
  sorry

end NUMINAMATH_CALUDE_necessarily_negative_l758_75807


namespace NUMINAMATH_CALUDE_discount_difference_l758_75880

theorem discount_difference (bill : ℝ) (d1 d2 d3 d4 : ℝ) :
  bill = 12000 ∧ d1 = 0.3 ∧ d2 = 0.2 ∧ d3 = 0.06 ∧ d4 = 0.04 →
  bill * (1 - d2) * (1 - d3) * (1 - d4) - bill * (1 - d1) = 263.04 := by
  sorry

end NUMINAMATH_CALUDE_discount_difference_l758_75880


namespace NUMINAMATH_CALUDE_initial_number_proof_l758_75860

theorem initial_number_proof (x : ℝ) : 
  x + 3889 - 47.80600000000004 = 3854.002 → x = 12.808000000000158 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l758_75860


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l758_75814

theorem quadratic_root_condition (a : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ > 1 ∧ r₂ < 1 ∧ r₁^2 + 2*a*r₁ + 1 = 0 ∧ r₂^2 + 2*a*r₂ + 1 = 0) → 
  a < -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l758_75814


namespace NUMINAMATH_CALUDE_share_division_l758_75856

theorem share_division (total : ℚ) (a b c : ℚ) : 
  total = 700 →
  a + b + c = total →
  a = b / 2 →
  b = c / 2 →
  c = 400 := by
sorry

end NUMINAMATH_CALUDE_share_division_l758_75856


namespace NUMINAMATH_CALUDE_sum_is_composite_l758_75882

theorem sum_is_composite (a b c d : ℕ) (h : a * b = c * d) :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ a + b + c + d = x * y :=
sorry

end NUMINAMATH_CALUDE_sum_is_composite_l758_75882


namespace NUMINAMATH_CALUDE_integral_multiple_equals_2400_l758_75809

theorem integral_multiple_equals_2400 : ∃ (x : ℤ), x = 4 * 595 ∧ x = 2400 := by
  sorry

end NUMINAMATH_CALUDE_integral_multiple_equals_2400_l758_75809


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l758_75801

/-- Calculates the length of a bridge given a person's walking speed and time to cross -/
theorem bridge_length_calculation (speed : ℝ) (time_minutes : ℝ) : 
  speed = 6 → time_minutes = 15 → speed * (time_minutes / 60) = 1.5 := by
  sorry

#check bridge_length_calculation

end NUMINAMATH_CALUDE_bridge_length_calculation_l758_75801


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l758_75859

theorem geometric_sequence_fourth_term 
  (x : ℝ) 
  (h1 : ∃ r : ℝ, (3*x + 3) = x * r) 
  (h2 : ∃ r : ℝ, (6*x + 6) = (3*x + 3) * r) :
  ∃ r : ℝ, -24 = (6*x + 6) * r :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l758_75859


namespace NUMINAMATH_CALUDE_evie_shells_left_l758_75821

/-- The number of shells Evie has left after collecting for 6 days and giving some to her brother -/
def shells_left (days : ℕ) (shells_per_day : ℕ) (shells_given : ℕ) : ℕ :=
  days * shells_per_day - shells_given

/-- Theorem stating that Evie has 58 shells left -/
theorem evie_shells_left : shells_left 6 10 2 = 58 := by
  sorry

end NUMINAMATH_CALUDE_evie_shells_left_l758_75821


namespace NUMINAMATH_CALUDE_necessary_condition_l758_75858

theorem necessary_condition (a b x y : ℤ) 
  (ha : 0 < a) (hb : 0 < b) 
  (h1 : x - y > a + b) (h2 : x * y > a * b) : 
  x > a ∧ y > b := by
  sorry

end NUMINAMATH_CALUDE_necessary_condition_l758_75858
