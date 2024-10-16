import Mathlib

namespace NUMINAMATH_CALUDE_probability_two_red_apples_l2643_264360

def total_apples : ℕ := 10
def red_apples : ℕ := 6
def green_apples : ℕ := 4
def chosen_apples : ℕ := 3

theorem probability_two_red_apples :
  (Nat.choose red_apples 2 * Nat.choose green_apples 1) / Nat.choose total_apples chosen_apples = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_apples_l2643_264360


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l2643_264331

theorem largest_prime_divisor_of_factorial_sum (n : ℕ) : 
  ∃ p : ℕ, p.Prime ∧ p ∣ (Nat.factorial 13 + Nat.factorial 14 * 2) ∧ 
  ∀ q : ℕ, q.Prime → q ∣ (Nat.factorial 13 + Nat.factorial 14 * 2) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l2643_264331


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2643_264378

theorem perfect_square_condition (a b : ℤ) : 
  (∀ m n : ℕ, ∃ k : ℕ, a * m^2 + b * n^2 = k^2) → a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2643_264378


namespace NUMINAMATH_CALUDE_inequality_proof_l2643_264317

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  2 * (x - y - 1) + 1 / (x^2 - 2*x*y + y^2) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2643_264317


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_l2643_264318

theorem quadratic_function_minimum (a b c : ℝ) (x₀ : ℝ) (ha : a > 0) (hx₀ : 2 * a * x₀ + b = 0) :
  ¬ (∀ x : ℝ, a * x^2 + b * x + c ≤ a * x₀^2 + b * x₀ + c) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_minimum_l2643_264318


namespace NUMINAMATH_CALUDE_thirteenth_result_l2643_264301

theorem thirteenth_result (total_count : Nat) (total_avg first_avg last_avg : ℚ) :
  total_count = 25 →
  total_avg = 19 →
  first_avg = 14 →
  last_avg = 17 →
  (total_count * total_avg - 12 * first_avg - 12 * last_avg : ℚ) = 103 :=
by sorry

end NUMINAMATH_CALUDE_thirteenth_result_l2643_264301


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l2643_264310

theorem quadratic_solution_sum (m n p : ℤ) : 
  (∃ x : ℚ, x * (5 * x - 11) = -6) ∧
  (∃ x y : ℚ, x = (m + n.sqrt : ℚ) / p ∧ y = (m - n.sqrt : ℚ) / p ∧ 
    x * (5 * x - 11) = -6 ∧ y * (5 * y - 11) = -6) ∧
  Nat.gcd (Nat.gcd m.natAbs n.natAbs) p.natAbs = 1 →
  m + n + p = 70 :=
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l2643_264310


namespace NUMINAMATH_CALUDE_floor_sqrt_sum_eq_floor_sqrt_sum_l2643_264312

theorem floor_sqrt_sum_eq_floor_sqrt_sum (n : ℕ) :
  ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_sum_eq_floor_sqrt_sum_l2643_264312


namespace NUMINAMATH_CALUDE_fourth_term_is_eleven_l2643_264369

/-- A sequence of 5 terms with specific properties -/
def CanSequence (a : Fin 5 → ℕ) : Prop :=
  a 0 = 2 ∧ 
  a 1 = 4 ∧ 
  a 2 = 7 ∧ 
  a 4 = 16 ∧
  ∀ i : Fin 3, (a (i + 1) - a i) - (a (i + 2) - a (i + 1)) = 1

theorem fourth_term_is_eleven (a : Fin 5 → ℕ) (h : CanSequence a) : a 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_eleven_l2643_264369


namespace NUMINAMATH_CALUDE_coin_arrangements_count_l2643_264368

/-- The number of indistinguishable gold coins -/
def num_gold_coins : ℕ := 5

/-- The number of indistinguishable silver coins -/
def num_silver_coins : ℕ := 5

/-- The total number of coins -/
def total_coins : ℕ := num_gold_coins + num_silver_coins

/-- The number of ways to arrange the gold and silver coins -/
def color_arrangements : ℕ := Nat.choose total_coins num_gold_coins

/-- The number of possible orientations to avoid face-to-face adjacency -/
def orientation_arrangements : ℕ := total_coins + 1

/-- The total number of distinguishable arrangements -/
def total_arrangements : ℕ := color_arrangements * orientation_arrangements

theorem coin_arrangements_count :
  total_arrangements = 2772 :=
sorry

end NUMINAMATH_CALUDE_coin_arrangements_count_l2643_264368


namespace NUMINAMATH_CALUDE_unique_partition_l2643_264392

/-- Represents the number of caps collected by each girl -/
def caps : List Nat := [20, 29, 31, 49, 51]

/-- Represents a partition of the caps into two boxes -/
structure Partition where
  red : List Nat
  blue : List Nat
  sum_red : red.sum = 60
  sum_blue : blue.sum = 120
  partition_complete : red ++ blue = caps

/-- The theorem to be proved -/
theorem unique_partition : ∃! p : Partition, True := by sorry

end NUMINAMATH_CALUDE_unique_partition_l2643_264392


namespace NUMINAMATH_CALUDE_root_minus_one_implies_k_eq_neg_two_l2643_264381

theorem root_minus_one_implies_k_eq_neg_two (k : ℝ) : 
  ((-1 : ℝ)^2 - k * (-1) + 1 = 0) → k = -2 := by
  sorry

end NUMINAMATH_CALUDE_root_minus_one_implies_k_eq_neg_two_l2643_264381


namespace NUMINAMATH_CALUDE_line_slope_l2643_264383

theorem line_slope (m n p K : ℝ) (h1 : p = 0.3333333333333333) : 
  (m = K * n + 5 ∧ m + 2 = K * (n + p) + 5) → K = 6 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l2643_264383


namespace NUMINAMATH_CALUDE_max_two_wins_l2643_264343

/-- Represents a single-elimination tournament --/
structure Tournament :=
  (participants : ℕ)

/-- Represents the number of participants who won exactly two matches --/
def exactlyTwoWins (t : Tournament) : ℕ := sorry

/-- The theorem stating the maximum number of participants who can win exactly two matches --/
theorem max_two_wins (t : Tournament) (h : t.participants = 100) : 
  exactlyTwoWins t ≤ 49 ∧ ∃ (strategy : Unit), exactlyTwoWins t = 49 := by sorry

end NUMINAMATH_CALUDE_max_two_wins_l2643_264343


namespace NUMINAMATH_CALUDE_square_quotient_property_l2643_264309

theorem square_quotient_property (a b : ℕ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h_div : (a * b + 1) ∣ (a^2 + b^2)) : 
  ∃ (k : ℕ), (a^2 + b^2) / (a * b + 1) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_square_quotient_property_l2643_264309


namespace NUMINAMATH_CALUDE_product_without_x3_x2_terms_l2643_264361

theorem product_without_x3_x2_terms (m n : ℝ) : 
  (∀ x : ℝ, (x^2 + m*x) * (x^2 - 2*x + n) = x^4 + m*n*x) → 
  m = 2 ∧ n = 4 := by
sorry

end NUMINAMATH_CALUDE_product_without_x3_x2_terms_l2643_264361


namespace NUMINAMATH_CALUDE_miss_one_out_of_three_l2643_264358

def free_throw_probability : ℝ := 0.9

theorem miss_one_out_of_three (p : ℝ) (hp : p = free_throw_probability) :
  p * p * (1 - p) + p * (1 - p) * p + (1 - p) * p * p = 0.243 := by
  sorry

end NUMINAMATH_CALUDE_miss_one_out_of_three_l2643_264358


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l2643_264391

theorem consecutive_negative_integers_sum (n : ℤ) : 
  n < 0 ∧ n * (n + 1) = 812 → n + (n + 1) = -57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l2643_264391


namespace NUMINAMATH_CALUDE_f_neg_x_properties_l2643_264393

-- Define the function f
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem f_neg_x_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, x < y → f (-x) > f (-y)) := by
  sorry

#check f_neg_x_properties

end NUMINAMATH_CALUDE_f_neg_x_properties_l2643_264393


namespace NUMINAMATH_CALUDE_sin_2x_value_l2643_264364

theorem sin_2x_value (x : Real) (h : Real.sin (x + π/4) = 1/4) : 
  Real.sin (2*x) = -7/8 := by
sorry

end NUMINAMATH_CALUDE_sin_2x_value_l2643_264364


namespace NUMINAMATH_CALUDE_linear_system_solution_inequality_system_solution_l2643_264314

-- Part 1: System of linear equations
theorem linear_system_solution :
  let x : ℝ := 5
  let y : ℝ := 1
  (x - 5 * y = 0) ∧ (3 * x + 2 * y = 17) := by sorry

-- Part 2: System of inequalities
theorem inequality_system_solution :
  ∀ x : ℝ, x < -1/5 →
    (2 * (x - 2) ≤ 3 - x) ∧ (1 - (2 * x + 1) / 3 > x + 1) := by sorry

end NUMINAMATH_CALUDE_linear_system_solution_inequality_system_solution_l2643_264314


namespace NUMINAMATH_CALUDE_megan_initial_files_l2643_264398

-- Define the problem parameters
def added_files : ℝ := 21.0
def files_per_folder : ℝ := 8.0
def final_folders : ℝ := 14.25

-- Define the theorem
theorem megan_initial_files :
  ∃ (initial_files : ℝ),
    (initial_files + added_files) / files_per_folder = final_folders ∧
    initial_files = 93.0 := by
  sorry

end NUMINAMATH_CALUDE_megan_initial_files_l2643_264398


namespace NUMINAMATH_CALUDE_president_vice_president_selection_l2643_264326

theorem president_vice_president_selection (n : ℕ) (h : n = 8) : 
  (n * (n - 1) : ℕ) = 56 := by
  sorry

end NUMINAMATH_CALUDE_president_vice_president_selection_l2643_264326


namespace NUMINAMATH_CALUDE_product_xyz_l2643_264397

theorem product_xyz (x y z : ℝ) 
  (sphere_eq : (x - 2)^2 + (y - 3)^2 + (z - 4)^2 = 9)
  (plane_eq : x + y + z = 12) :
  x * y * z = 42 := by
  sorry

end NUMINAMATH_CALUDE_product_xyz_l2643_264397


namespace NUMINAMATH_CALUDE_complex_solution_l2643_264344

-- Define the determinant operation
def det (a b c d : ℂ) : ℂ := a * d - b * c

-- State the theorem
theorem complex_solution :
  ∃ z : ℂ, det 2 (-1) z (z * Complex.I) = 1 + Complex.I ∧ z = 3/5 - 1/5 * Complex.I :=
sorry

end NUMINAMATH_CALUDE_complex_solution_l2643_264344


namespace NUMINAMATH_CALUDE_forgotten_angles_sum_l2643_264372

theorem forgotten_angles_sum (n : ℕ) (partial_sum : ℝ) : 
  n ≥ 3 → 
  partial_sum = 2797 → 
  (n - 2) * 180 - partial_sum = 83 :=
by sorry

end NUMINAMATH_CALUDE_forgotten_angles_sum_l2643_264372


namespace NUMINAMATH_CALUDE_julia_played_with_16_kids_l2643_264399

def kids_on_tuesday : ℕ := 4

def kids_difference : ℕ := 12

def kids_on_monday : ℕ := kids_on_tuesday + kids_difference

theorem julia_played_with_16_kids : kids_on_monday = 16 := by
  sorry

end NUMINAMATH_CALUDE_julia_played_with_16_kids_l2643_264399


namespace NUMINAMATH_CALUDE_athletes_leaving_hours_l2643_264379

/-- The number of hours athletes left the camp -/
def hours_athletes_left : ℕ := 4

/-- The initial number of athletes in the camp -/
def initial_athletes : ℕ := 300

/-- The rate at which athletes left the camp (per hour) -/
def leaving_rate : ℕ := 28

/-- The rate at which new athletes entered the camp (per hour) -/
def entering_rate : ℕ := 15

/-- The number of hours new athletes entered the camp -/
def entering_hours : ℕ := 7

/-- The difference in the total number of athletes over the two nights -/
def athlete_difference : ℕ := 7

theorem athletes_leaving_hours :
  initial_athletes - (leaving_rate * hours_athletes_left) + 
  (entering_rate * entering_hours) = initial_athletes + athlete_difference :=
by sorry

end NUMINAMATH_CALUDE_athletes_leaving_hours_l2643_264379


namespace NUMINAMATH_CALUDE_max_followers_after_three_weeks_l2643_264394

def susyInitialFollowers : ℕ := 100
def sarahInitialFollowers : ℕ := 50

def susyWeek1Gain : ℕ := 40
def sarahWeek1Gain : ℕ := 90

def susyTotalFollowers : ℕ := 
  susyInitialFollowers + susyWeek1Gain + (susyWeek1Gain / 2) + (susyWeek1Gain / 4)

def sarahTotalFollowers : ℕ := 
  sarahInitialFollowers + sarahWeek1Gain + (sarahWeek1Gain / 3) + (sarahWeek1Gain / 9)

theorem max_followers_after_three_weeks :
  max susyTotalFollowers sarahTotalFollowers = 180 := by
  sorry

end NUMINAMATH_CALUDE_max_followers_after_three_weeks_l2643_264394


namespace NUMINAMATH_CALUDE_min_four_digit_satisfying_condition_l2643_264302

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def satisfies_condition (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  let ab := 10 * a + b
  let cd := 10 * c + d
  (n + ab * cd) % 1111 = 0

theorem min_four_digit_satisfying_condition :
  ∃ (n : ℕ), is_four_digit n ∧ satisfies_condition n ∧
  ∀ (m : ℕ), is_four_digit m → satisfies_condition m → n ≤ m :=
by
  use 1729
  sorry

end NUMINAMATH_CALUDE_min_four_digit_satisfying_condition_l2643_264302


namespace NUMINAMATH_CALUDE_sonika_deposit_l2643_264342

theorem sonika_deposit (P R : ℝ) : 
  (P + (P * R * 3) / 100 = 11200) → 
  (P + (P * (R + 2) * 3) / 100 = 11680) → 
  P = 8000 := by
  sorry

end NUMINAMATH_CALUDE_sonika_deposit_l2643_264342


namespace NUMINAMATH_CALUDE_quadratic_equation_conversion_l2643_264375

theorem quadratic_equation_conversion :
  ∀ x : ℝ, (x - 3)^2 = 4 ↔ x^2 - 6*x + 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_conversion_l2643_264375


namespace NUMINAMATH_CALUDE_fraction_multiplication_invariance_l2643_264380

theorem fraction_multiplication_invariance (a b m : ℝ) (h : b ≠ 0) :
  ∀ x : ℝ, (a * (x - m)) / (b * (x - m)) = a / b ↔ x ≠ m :=
sorry

end NUMINAMATH_CALUDE_fraction_multiplication_invariance_l2643_264380


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l2643_264339

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 5| + |x - 3|

-- State the theorem
theorem min_value_and_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1/a + 1/b = Real.sqrt 3) : 
  (∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x : ℝ, f x = m) ∧ m = 2) ∧ 
  (1/a^2 + 2/b^2 ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l2643_264339


namespace NUMINAMATH_CALUDE_ab_minus_a_minus_b_even_l2643_264316

def S : Set ℕ := {1, 3, 5, 7, 9}

theorem ab_minus_a_minus_b_even (a b : ℕ) (ha : a ∈ S) (hb : b ∈ S) (hab : a ≠ b) :
  Even (a * b - a - b) :=
by
  sorry

end NUMINAMATH_CALUDE_ab_minus_a_minus_b_even_l2643_264316


namespace NUMINAMATH_CALUDE_fraction_equations_l2643_264307

theorem fraction_equations : 
  (5 / 6 - 2 / 3 = 1 / 6) ∧
  (1 / 2 + 1 / 4 = 3 / 4) ∧
  (9 / 7 - 7 / 21 = 17 / 21) ∧
  (4 / 8 - 1 / 4 = 3 / 8) := by
sorry

end NUMINAMATH_CALUDE_fraction_equations_l2643_264307


namespace NUMINAMATH_CALUDE_shortest_hexpath_distribution_l2643_264325

/-- Represents a direction in the hexagonal grid -/
inductive Direction
| Horizontal
| Diagonal1
| Diagonal2

/-- Represents a path in the hexagonal grid -/
structure HexPath where
  length : ℕ
  horizontal : ℕ
  diagonal1 : ℕ
  diagonal2 : ℕ
  sum_constraint : length = horizontal + diagonal1 + diagonal2

/-- A shortest path in a hexagonal grid -/
def is_shortest_path (path : HexPath) : Prop :=
  path.horizontal = path.diagonal1 + path.diagonal2

theorem shortest_hexpath_distribution (path : HexPath) 
  (h_shortest : is_shortest_path path) (h_length : path.length = 100) :
  path.horizontal = 50 ∧ path.diagonal1 + path.diagonal2 = 50 := by
  sorry

#check shortest_hexpath_distribution

end NUMINAMATH_CALUDE_shortest_hexpath_distribution_l2643_264325


namespace NUMINAMATH_CALUDE_zero_in_A_l2643_264319

def A : Set ℝ := {x | x * (x - 1) = 0}

theorem zero_in_A : 0 ∈ A := by
  sorry

end NUMINAMATH_CALUDE_zero_in_A_l2643_264319


namespace NUMINAMATH_CALUDE_original_number_proof_l2643_264335

theorem original_number_proof (n k : ℕ) : 
  (n + k = 3200) → 
  (k ≥ 0) →
  (k < 8) →
  (3200 % 8 = 0) →
  ((n + k) % 8 = 0) →
  (∀ m : ℕ, m < k → (n + m) % 8 ≠ 0) →
  n = 3199 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l2643_264335


namespace NUMINAMATH_CALUDE_equation_solution_l2643_264363

theorem equation_solution : 
  let eq (x : ℝ) := x^3 + Real.log 25 + Real.log 32 + Real.log 53 * x - Real.log 23 - Real.log 35 - Real.log 52 * x^2 - 1
  (eq (Real.log 23) = 0) ∧ (eq (Real.log 35) = 0) ∧ (eq (Real.log 52) = 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2643_264363


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2643_264328

theorem chess_tournament_games (n : ℕ) (h : n = 12) :
  2 * n * (n - 1) / 2 = 264 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l2643_264328


namespace NUMINAMATH_CALUDE_parabola_directrix_l2643_264349

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop := y = 4 * x^2

/-- The equation of the directrix -/
def directrix_equation (y : ℝ) : Prop := y = -1/16

/-- Theorem: The directrix of the parabola y = 4x^2 is y = -1/16 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola_equation x y → ∃ y_directrix : ℝ, directrix_equation y_directrix :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2643_264349


namespace NUMINAMATH_CALUDE_roots_of_quadratic_equation_l2643_264384

theorem roots_of_quadratic_equation :
  let f : ℝ → ℝ := λ x ↦ x^2 - 3*x
  (f 0 = 0) ∧ (f 3 = 0) ∧ (∀ x : ℝ, f x = 0 → x = 0 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_equation_l2643_264384


namespace NUMINAMATH_CALUDE_retirement_total_is_70_l2643_264355

/-- Represents the retirement eligibility rule for a company -/
structure RetirementRule where
  hireYear : ℕ
  hireAge : ℕ
  retirementYear : ℕ

/-- Calculates the required total of age and years of employment for retirement -/
def requiredTotal (rule : RetirementRule) : ℕ :=
  let ageAtRetirement := rule.hireAge + (rule.retirementYear - rule.hireYear)
  let yearsOfEmployment := rule.retirementYear - rule.hireYear
  ageAtRetirement + yearsOfEmployment

/-- Theorem stating that the required total for retirement is 70 -/
theorem retirement_total_is_70 (rule : RetirementRule) 
    (h1 : rule.hireYear = 1989)
    (h2 : rule.hireAge = 32)
    (h3 : rule.retirementYear = 2008) :
  requiredTotal rule = 70 := by
  sorry

end NUMINAMATH_CALUDE_retirement_total_is_70_l2643_264355


namespace NUMINAMATH_CALUDE_total_amount_is_70_l2643_264387

/-- Represents the distribution of money among three people -/
structure Distribution where
  x : ℚ  -- x's share in rupees
  y : ℚ  -- y's share in rupees
  z : ℚ  -- z's share in rupees

/-- Checks if a distribution satisfies the given conditions -/
def is_valid_distribution (d : Distribution) : Prop :=
  d.y = 0.45 * d.x ∧ d.z = 0.3 * d.x ∧ d.y = 18

/-- The theorem to prove -/
theorem total_amount_is_70 (d : Distribution) :
  is_valid_distribution d → d.x + d.y + d.z = 70 := by
  sorry


end NUMINAMATH_CALUDE_total_amount_is_70_l2643_264387


namespace NUMINAMATH_CALUDE_opposite_solutions_value_of_m_l2643_264365

theorem opposite_solutions_value_of_m : ∀ (x y m : ℝ),
  (3 * x + 5 * y = 2) →
  (2 * x + 7 * y = m - 18) →
  (x = -y) →
  m = 23 := by
  sorry

end NUMINAMATH_CALUDE_opposite_solutions_value_of_m_l2643_264365


namespace NUMINAMATH_CALUDE_profit_A_range_max_a_value_l2643_264351

-- Define the profit functions
def profit_A_before (x : ℝ) : ℝ := 120000 * 500

def profit_A_after (x : ℝ) : ℝ := 120000 * (500 - x) * (1 + 0.005 * x)

def profit_B (x a : ℝ) : ℝ := 120000 * x * (a - 0.013 * x)

-- Theorem for part (I)
theorem profit_A_range (x : ℝ) :
  (0 < x ∧ x ≤ 300) ↔ profit_A_after x ≥ profit_A_before x :=
sorry

-- Theorem for part (II)
theorem max_a_value :
  ∃ (a : ℝ), a = 5.5 ∧
  ∀ (x : ℝ), 0 < x → x ≤ 300 →
  (∀ (a' : ℝ), a' > 0 → profit_B x a' ≤ profit_A_after x → a' ≤ a) :=
sorry

end NUMINAMATH_CALUDE_profit_A_range_max_a_value_l2643_264351


namespace NUMINAMATH_CALUDE_simplify_tan_product_l2643_264305

theorem simplify_tan_product : (1 + Real.tan (30 * π / 180)) * (1 + Real.tan (15 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_tan_product_l2643_264305


namespace NUMINAMATH_CALUDE_paint_container_rectangle_perimeter_l2643_264396

theorem paint_container_rectangle_perimeter :
  ∀ x : ℝ,
  -- Old rectangle conditions
  x > 0 →
  let old_width := x
  let old_length := 3 * x
  let old_area := old_width * old_length
  -- New rectangle conditions
  let new_width := x + 8
  let new_length := 3 * x - 18
  let new_area := new_width * new_length
  -- Equal area condition
  old_area = new_area →
  -- Perimeter calculation
  let new_perimeter := 2 * (new_width + new_length)
  -- Theorem statement
  new_perimeter = 172 :=
by
  sorry


end NUMINAMATH_CALUDE_paint_container_rectangle_perimeter_l2643_264396


namespace NUMINAMATH_CALUDE_S_singleton_I_singleton_l2643_264321

-- Define the set X
inductive X
| zero : X
| a : X
| b : X
| c : X

-- Define the addition operation on X
def add : X → X → X
| X.zero, y => y
| X.a, X.zero => X.a
| X.a, X.a => X.zero
| X.a, X.b => X.c
| X.a, X.c => X.b
| X.b, X.zero => X.b
| X.b, X.a => X.c
| X.b, X.b => X.zero
| X.b, X.c => X.a
| X.c, X.zero => X.c
| X.c, X.a => X.b
| X.c, X.b => X.a
| X.c, X.c => X.zero

-- Define the set of all functions from X to X
def M : Type := X → X

-- Define the set S
def S : Set M := {f : M | ∀ x y : X, f (add (add x y) x) = add (add (f x) (f y)) (f x)}

-- Define the set I
def I : Set M := {f : M | ∀ x : X, f (add x x) = add (f x) (f x)}

-- Theorem: S contains only one function (the zero function)
theorem S_singleton : ∃! f : M, f ∈ S := sorry

-- Theorem: I contains only one function (the zero function)
theorem I_singleton : ∃! f : M, f ∈ I := sorry

end NUMINAMATH_CALUDE_S_singleton_I_singleton_l2643_264321


namespace NUMINAMATH_CALUDE_least_x_value_l2643_264389

theorem least_x_value (x p : ℕ) (h1 : x > 0) (h2 : Nat.Prime p) 
  (h3 : ∃ q : ℕ, Nat.Prime q ∧ q % 2 = 1 ∧ x = 11 * p * q) : 
  x ≥ 66 ∧ ∃ x0 : ℕ, x0 = 66 ∧ x0 > 0 ∧ Nat.Prime p ∧ 
  ∃ q0 : ℕ, Nat.Prime q0 ∧ q0 % 2 = 1 ∧ x0 = 11 * p * q0 :=
sorry

end NUMINAMATH_CALUDE_least_x_value_l2643_264389


namespace NUMINAMATH_CALUDE_fish_left_in_tank_l2643_264359

/-- The number of fish left in Lucy's first tank after moving some to another tank -/
theorem fish_left_in_tank (initial_fish : ℝ) (moved_fish : ℝ) 
  (h1 : initial_fish = 212.0)
  (h2 : moved_fish = 68.0) : 
  initial_fish - moved_fish = 144.0 := by
  sorry

end NUMINAMATH_CALUDE_fish_left_in_tank_l2643_264359


namespace NUMINAMATH_CALUDE_cos_300_degrees_l2643_264338

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l2643_264338


namespace NUMINAMATH_CALUDE_infinite_square_free_triples_l2643_264374

/-- A positive integer is square-free if it's not divisible by any perfect square greater than 1 -/
def IsSquareFree (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 1 → k * k ∣ n → k = 1

/-- The set of positive integers n for which n, n+1, and n+2 are all square-free -/
def SquareFreeTriples : Set ℕ :=
  {n : ℕ | n > 0 ∧ IsSquareFree n ∧ IsSquareFree (n + 1) ∧ IsSquareFree (n + 2)}

/-- The set of positive integers n for which n, n+1, and n+2 are all square-free is infinite -/
theorem infinite_square_free_triples : Set.Infinite SquareFreeTriples :=
sorry

end NUMINAMATH_CALUDE_infinite_square_free_triples_l2643_264374


namespace NUMINAMATH_CALUDE_problem_statement_l2643_264352

theorem problem_statement : ((18^18 / 18^17)^3 * 9^3) / 3^6 = 5832 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2643_264352


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l2643_264334

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line
  (A : Point)
  (given_line : Line)
  (h_A : A.x = 3 ∧ A.y = 2)
  (h_given : given_line.a = 4 ∧ given_line.b = 1 ∧ given_line.c = -2)
  : ∃ (result_line : Line),
    result_line.a = 4 ∧ 
    result_line.b = 1 ∧ 
    result_line.c = -14 ∧
    A.liesOn result_line ∧
    result_line.isParallelTo given_line :=
  sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l2643_264334


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l2643_264377

-- Define the function f
def f (x : ℝ) : ℝ := (x+3)*(x+2)*(x+1)*x*(x-1)*(x-2)*(x-3)

-- State the theorem
theorem f_derivative_at_one : 
  deriv f 1 = 48 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l2643_264377


namespace NUMINAMATH_CALUDE_sum_of_complex_roots_l2643_264382

theorem sum_of_complex_roots (a b c : ℂ) 
  (eq1 : a^2 = b - c) 
  (eq2 : b^2 = c - a) 
  (eq3 : c^2 = a - b) : 
  (a + b + c = 0) ∨ (a + b + c = Complex.I * Real.sqrt 6) ∨ (a + b + c = -Complex.I * Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_complex_roots_l2643_264382


namespace NUMINAMATH_CALUDE_perpendicular_planes_from_lines_l2643_264304

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_from_lines 
  (m n : Line) (α β : Plane) :
  perpendicular m α → 
  parallel_lines m n → 
  parallel_line_plane n β → 
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_from_lines_l2643_264304


namespace NUMINAMATH_CALUDE_pencil_price_calculation_l2643_264308

theorem pencil_price_calculation (num_pens num_pencils total_cost pen_avg_price : ℝ) 
  (h1 : num_pens = 30)
  (h2 : num_pencils = 75)
  (h3 : total_cost = 750)
  (h4 : pen_avg_price = 20) :
  (total_cost - num_pens * pen_avg_price) / num_pencils = 2 := by
  sorry

#check pencil_price_calculation

end NUMINAMATH_CALUDE_pencil_price_calculation_l2643_264308


namespace NUMINAMATH_CALUDE_hexagon_side_length_l2643_264347

/-- A hexagon is a polygon with 6 sides -/
def Hexagon : ℕ := 6

/-- Given a hexagon with perimeter 42 inches, prove that each side length is 7 inches -/
theorem hexagon_side_length (perimeter : ℝ) (h1 : perimeter = 42) :
  perimeter / Hexagon = 7 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_side_length_l2643_264347


namespace NUMINAMATH_CALUDE_expand_expression_l2643_264341

theorem expand_expression (x : ℝ) : (5 * x^2 + 3) * 4 * x^3 = 20 * x^5 + 12 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2643_264341


namespace NUMINAMATH_CALUDE_first_hit_not_binomial_l2643_264303

/-- A random variable follows a binomial distribution -/
def is_binomial_distribution (X : ℕ → ℝ) : Prop :=
  ∃ (n : ℕ) (p : ℝ), 0 ≤ p ∧ p ≤ 1 ∧
    ∀ k, 0 ≤ k ∧ k ≤ n → X k = (n.choose k : ℝ) * p^k * (1-p)^(n-k)

/-- Computer virus infection scenario -/
def computer_infection (n : ℕ) : ℕ → ℝ := sorry

/-- First hit scenario -/
def first_hit : ℕ → ℝ := sorry

/-- Multiple shots scenario -/
def multiple_shots (n : ℕ) : ℕ → ℝ := sorry

/-- Car refueling scenario -/
def car_refueling : ℕ → ℝ := sorry

/-- Theorem stating that the first hit scenario is not a binomial distribution -/
theorem first_hit_not_binomial :
  is_binomial_distribution (computer_infection 10) ∧
  is_binomial_distribution (multiple_shots 10) ∧
  is_binomial_distribution car_refueling →
  ¬ is_binomial_distribution first_hit := by sorry

end NUMINAMATH_CALUDE_first_hit_not_binomial_l2643_264303


namespace NUMINAMATH_CALUDE_salary_change_percentage_l2643_264315

theorem salary_change_percentage (initial_salary : ℝ) (h : initial_salary > 0) :
  let decreased_salary := initial_salary * (1 - 0.4)
  let final_salary := decreased_salary * (1 + 0.4)
  (initial_salary - final_salary) / initial_salary = 0.16 := by
sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l2643_264315


namespace NUMINAMATH_CALUDE_root_analysis_uses_classification_and_discussion_l2643_264370

/-- A mathematical thinking method -/
inductive MathThinkingMethod
| Transformation
| Equation
| ClassificationAndDiscussion
| NumberAndShapeCombination

/-- A number category for root analysis -/
inductive NumberCategory
| Positive
| Zero
| Negative

/-- Represents the analysis of roots -/
structure RootAnalysis where
  categories : List NumberCategory
  method : MathThinkingMethod

/-- The specific root analysis we're considering -/
def squareAndCubeRootAnalysis : RootAnalysis :=
  { categories := [NumberCategory.Positive, NumberCategory.Zero, NumberCategory.Negative],
    method := MathThinkingMethod.ClassificationAndDiscussion }

/-- Theorem stating that the given root analysis uses classification and discussion thinking -/
theorem root_analysis_uses_classification_and_discussion :
  squareAndCubeRootAnalysis.method = MathThinkingMethod.ClassificationAndDiscussion :=
by sorry

end NUMINAMATH_CALUDE_root_analysis_uses_classification_and_discussion_l2643_264370


namespace NUMINAMATH_CALUDE_circle_line_intersection_range_l2643_264324

/-- Given a line and a circle with common points, prove the range of the circle's center x-coordinate. -/
theorem circle_line_intersection_range (a : ℝ) : 
  (∃ x y : ℝ, x - y + 1 = 0 ∧ (x - a)^2 + y^2 = 2) → 
  -3 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_range_l2643_264324


namespace NUMINAMATH_CALUDE_bus_problem_l2643_264337

/-- Calculates the final number of people on a bus given initial count and changes -/
def final_bus_count (initial : ℕ) (getting_on : ℕ) (getting_off : ℕ) : ℕ :=
  initial + getting_on - getting_off

/-- Theorem stating that given 22 initial people, 4 getting on, and 8 getting off, 
    the final count is 18 -/
theorem bus_problem : final_bus_count 22 4 8 = 18 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_l2643_264337


namespace NUMINAMATH_CALUDE_six_to_six_sum_l2643_264330

theorem six_to_six_sum : (6^6 : ℕ) + 6^6 + 6^6 + 6^6 + 6^6 + 6^6 = 6^7 := by
  sorry

end NUMINAMATH_CALUDE_six_to_six_sum_l2643_264330


namespace NUMINAMATH_CALUDE_max_a_for_quadratic_inequality_l2643_264395

theorem max_a_for_quadratic_inequality :
  (∃ (a : ℝ), ∀ (x : ℝ), x^2 - 2*x - a ≥ 0) ∧
  (∀ (a : ℝ), (∀ (x : ℝ), x^2 - 2*x - a ≥ 0) → a ≤ -1) ∧
  (∀ (x : ℝ), x^2 - 2*x - (-1) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_max_a_for_quadratic_inequality_l2643_264395


namespace NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_l2643_264367

-- Problem 1
theorem simplify_expression (x y : ℝ) : 
  x - (2*x - y) + (3*x - 2*y) = 2*x - y := by sorry

-- Problem 2
theorem evaluate_expression : 
  let x : ℚ := -2/3
  let y : ℚ := 3/2
  2*x*y + (-3*x^3 + 5*x*y + 2) - 3*(2*x*y - x^3 + 1) = -2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_l2643_264367


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l2643_264333

theorem rectangle_area_diagonal (l w d : ℝ) (h1 : l / w = 5 / 2) (h2 : l^2 + w^2 = d^2) :
  l * w = (10 / 29) * d^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l2643_264333


namespace NUMINAMATH_CALUDE_triplet_equality_l2643_264313

theorem triplet_equality (p q r s t u v : ℤ) : 
  -- Formulation 1
  (q + r + (p + r) + (2*p + 2*q + r) = r + (p + 2*q + r) + (2*p + q + r)) ∧
  ((q + r)^2 + (p + r)^2 + (2*p + 2*q + r)^2 = r^2 + (p + 2*q + r)^2 + (2*p + q + r)^2) ∧
  -- Formulation 2
  (u*v = s*t → 
    (s + t + (u + v) = u + v + (s + t)) ∧
    (s^2 + t^2 + (u + v)^2 = u^2 + v^2 + (s + t)^2)) := by
  sorry

end NUMINAMATH_CALUDE_triplet_equality_l2643_264313


namespace NUMINAMATH_CALUDE_james_total_matches_l2643_264322

/-- The number of boxes in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of boxes James has -/
def james_dozens : ℕ := 5

/-- The number of matches in each box -/
def matches_per_box : ℕ := 20

/-- Theorem: James has 1200 matches in total -/
theorem james_total_matches :
  james_dozens * dozen * matches_per_box = 1200 := by
  sorry

end NUMINAMATH_CALUDE_james_total_matches_l2643_264322


namespace NUMINAMATH_CALUDE_rms_geq_cube_root_avg_product_l2643_264354

theorem rms_geq_cube_root_avg_product (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  Real.sqrt ((a^2 + b^2 + c^2 + d^2) / 4) ≥ (((a*b*c + a*b*d + a*c*d + b*c*d) / 4) ^ (1/3 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_rms_geq_cube_root_avg_product_l2643_264354


namespace NUMINAMATH_CALUDE_problem_solution_l2643_264300

theorem problem_solution (x y : ℚ) (h1 : x - y = 8) (h2 : x + 2*y = 10) : x = 26/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2643_264300


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l2643_264306

def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

def is_center (h k : ℝ) : Prop := ∀ x y : ℝ, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = 4

theorem circle_center_and_radius :
  is_center 2 0 ∧ ∀ x y : ℝ, circle_equation x y → (x - 2)^2 + y^2 ≤ 4 := by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l2643_264306


namespace NUMINAMATH_CALUDE_potatoes_bought_l2643_264390

/-- The number of potatoes mother bought can be calculated by summing the potatoes used for salads, 
    mashed potatoes, and the potatoes left. -/
theorem potatoes_bought (salad_potatoes mashed_potatoes left_potatoes : ℕ) :
  salad_potatoes = 15 → mashed_potatoes = 24 → left_potatoes = 13 →
  salad_potatoes + mashed_potatoes + left_potatoes = 52 := by
  sorry

end NUMINAMATH_CALUDE_potatoes_bought_l2643_264390


namespace NUMINAMATH_CALUDE_cost_of_500_candies_l2643_264346

def candies_per_box : ℕ := 20
def cost_per_box : ℚ := 8
def discount_percentage : ℚ := 0.1
def discount_threshold : ℕ := 400
def order_size : ℕ := 500

theorem cost_of_500_candies : 
  let boxes_needed : ℕ := order_size / candies_per_box
  let total_cost : ℚ := boxes_needed * cost_per_box
  let discount : ℚ := if order_size > discount_threshold then discount_percentage * total_cost else 0
  let final_cost : ℚ := total_cost - discount
  final_cost = 180 := by sorry

end NUMINAMATH_CALUDE_cost_of_500_candies_l2643_264346


namespace NUMINAMATH_CALUDE_simplify_expressions_l2643_264366

open Real

theorem simplify_expressions (θ : ℝ) :
  (sqrt (1 - 2 * sin (135 * π / 180) * cos (135 * π / 180))) / 
  (sin (135 * π / 180) + sqrt (1 - sin (135 * π / 180) ^ 2)) = 1 ∧
  (sin (θ - 5 * π) * cos (-π / 2 - θ) * cos (8 * π - θ)) / 
  (sin (θ - 3 * π / 2) * sin (-θ - 4 * π)) = -sin (θ - 5 * π) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l2643_264366


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2643_264386

/-- Given an ellipse C with equation x²/a² + y²/b² = 1 where a > b > 0,
    and points A(-a,0), B(a,0), M(x,y), N(x,-y) on C,
    prove that if the product of slopes of AM and BN is 4/9,
    then the eccentricity of C is √5/3 -/
theorem ellipse_eccentricity (a b : ℝ) (x y : ℝ) :
  a > b → b > 0 →
  x^2 / a^2 + y^2 / b^2 = 1 →
  (y / (x + a)) * (-y / (x - a)) = 4/9 →
  Real.sqrt (1 - b^2 / a^2) = Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2643_264386


namespace NUMINAMATH_CALUDE_unique_real_solution_iff_a_in_range_l2643_264376

/-- The equation x^3 - ax^2 - 4ax + 4a^2 - 1 = 0 has exactly one real solution in x if and only if a ∈ (-∞, 3/4). -/
theorem unique_real_solution_iff_a_in_range (a : ℝ) : 
  (∃! x : ℝ, x^3 - a*x^2 - 4*a*x + 4*a^2 - 1 = 0) ↔ a < 3/4 :=
sorry

end NUMINAMATH_CALUDE_unique_real_solution_iff_a_in_range_l2643_264376


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l2643_264320

/-- Given a quadratic equation (2√3 + √2)x² + 2(√3 + √2)x + (√2 - 2√3) = 0,
    prove that the sum of its roots is -(4 + √6)/5 and the product of its roots is (2√6 - 7)/5 -/
theorem quadratic_roots_sum_and_product :
  let a : ℝ := 2 * Real.sqrt 3 + Real.sqrt 2
  let b : ℝ := 2 * (Real.sqrt 3 + Real.sqrt 2)
  let c : ℝ := Real.sqrt 2 - 2 * Real.sqrt 3
  (-(b / a) = -(4 + Real.sqrt 6) / 5) ∧ 
  (c / a = (2 * Real.sqrt 6 - 7) / 5) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l2643_264320


namespace NUMINAMATH_CALUDE_cloth_cost_price_l2643_264345

/-- Represents the cost price per metre of cloth -/
def cost_price_per_metre (total_length : ℕ) (total_selling_price : ℕ) (loss_per_metre : ℕ) : ℕ :=
  (total_selling_price + total_length * loss_per_metre) / total_length

/-- Theorem: Given a cloth of 300 metres sold for Rs. 9000 with a loss of Rs. 6 per metre,
    the cost price for one metre of cloth is Rs. 36 -/
theorem cloth_cost_price :
  cost_price_per_metre 300 9000 6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l2643_264345


namespace NUMINAMATH_CALUDE_alpha_beta_sum_l2643_264327

theorem alpha_beta_sum (α β : ℝ) 
  (h1 : α^3 - 3*α^2 + 5*α - 4 = 0)
  (h2 : β^3 - 3*β^2 + 5*β - 2 = 0) : 
  α + β = 2 := by
  sorry

end NUMINAMATH_CALUDE_alpha_beta_sum_l2643_264327


namespace NUMINAMATH_CALUDE_only_C_not_proportional_l2643_264357

-- Define the groups of line segments
def group_A : (ℚ × ℚ × ℚ × ℚ) := (3, 6, 2, 4)
def group_B : (ℚ × ℚ × ℚ × ℚ) := (1, 2, 2, 4)
def group_C : (ℚ × ℚ × ℚ × ℚ) := (4, 6, 5, 10)
def group_D : (ℚ × ℚ × ℚ × ℚ) := (1, 1/2, 1/6, 1/3)

-- Define a function to check if a group is proportional
def is_proportional (group : ℚ × ℚ × ℚ × ℚ) : Prop :=
  let (a, b, c, d) := group
  a / b = c / d

-- Theorem stating that only group C is not proportional
theorem only_C_not_proportional :
  is_proportional group_A ∧
  is_proportional group_B ∧
  ¬is_proportional group_C ∧
  is_proportional group_D :=
by sorry

end NUMINAMATH_CALUDE_only_C_not_proportional_l2643_264357


namespace NUMINAMATH_CALUDE_triangle_radius_equality_l2643_264323

/-- For a triangle with sides a, b, c, circumradius R, inradius r, and semi-perimeter p,
    prove that ab + bc + ac = r² + p² + 4Rr -/
theorem triangle_radius_equality (a b c R r p : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0 ∧ r > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_semi_perimeter : p = (a + b + c) / 2)
  (h_circumradius : R = (a * b * c) / (4 * (p * (p - a) * (p - b) * (p - c))^(1/2)))
  (h_inradius : r = (p * (p - a) * (p - b) * (p - c))^(1/2) / p) :
  a * b + b * c + a * c = r^2 + p^2 + 4 * R * r := by
sorry

end NUMINAMATH_CALUDE_triangle_radius_equality_l2643_264323


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2643_264373

theorem absolute_value_equation_solution :
  ∀ x : ℝ, (|2*x + 1| - |x - 5| = 6) ↔ (x = -12 ∨ x = 10/3) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2643_264373


namespace NUMINAMATH_CALUDE_complement_of_angle_l2643_264353

theorem complement_of_angle (A : ℝ) : A = 35 → 180 - A = 145 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_angle_l2643_264353


namespace NUMINAMATH_CALUDE_right_triangle_area_l2643_264371

-- Define the right triangle
def RightTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

-- Define the incircle radius formula for a right triangle
def IncircleRadius (a b c r : ℝ) : Prop :=
  r = (a + b - c) / 2

-- Theorem statement
theorem right_triangle_area (a b c r : ℝ) :
  RightTriangle a b c →
  IncircleRadius a b c r →
  a = 3 →
  r = 3/8 →
  (1/2) * a * b = 21/16 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2643_264371


namespace NUMINAMATH_CALUDE_combined_mean_score_l2643_264362

/-- Given two sections of algebra students with different mean scores and a ratio of students between sections, calculate the combined mean score. -/
theorem combined_mean_score (mean1 mean2 : ℚ) (ratio : ℚ) : 
  mean1 = 92 →
  mean2 = 78 →
  ratio = 5/7 →
  (mean1 * ratio + mean2) / (ratio + 1) = 1006/12 := by
  sorry

end NUMINAMATH_CALUDE_combined_mean_score_l2643_264362


namespace NUMINAMATH_CALUDE_g_x_plus_3_l2643_264329

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x + 1

-- State the theorem
theorem g_x_plus_3 : ∀ x : ℝ, g (x + 3) = 3 * x + 10 := by
  sorry

end NUMINAMATH_CALUDE_g_x_plus_3_l2643_264329


namespace NUMINAMATH_CALUDE_failed_both_subjects_percentage_l2643_264332

def total_candidates : ℕ := 3000
def failed_english_percent : ℚ := 49 / 100
def failed_hindi_percent : ℚ := 36 / 100
def passed_english_alone : ℕ := 630

theorem failed_both_subjects_percentage :
  let passed_english_alone_percent : ℚ := passed_english_alone / total_candidates
  let passed_english_percent : ℚ := 1 - failed_english_percent
  let passed_hindi_percent : ℚ := 1 - failed_hindi_percent
  let passed_both_percent : ℚ := passed_english_percent - passed_english_alone_percent
  let passed_hindi_alone_percent : ℚ := passed_hindi_percent - passed_both_percent
  let failed_both_percent : ℚ := 1 - (passed_english_alone_percent + passed_hindi_alone_percent + passed_both_percent)
  failed_both_percent = 15 / 100 := by
  sorry

end NUMINAMATH_CALUDE_failed_both_subjects_percentage_l2643_264332


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l2643_264311

/-- Given a right triangle with legs a and b, and hypotenuse c, 
    the radius r of its inscribed circle is (a + b - c) / 2 -/
theorem inscribed_circle_radius_right_triangle 
  (a b c : ℝ) (h_right : a^2 + b^2 = c^2) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ r : ℝ, r > 0 ∧ r = (a + b - c) / 2 ∧ 
    r * (a + b + c) / 2 = a * b / 2 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l2643_264311


namespace NUMINAMATH_CALUDE_prob_no_consecutive_heads_10_is_9_64_l2643_264385

/-- The number of coin tosses -/
def n : ℕ := 10

/-- The probability of getting heads on a single toss -/
def p : ℚ := 1/2

/-- The number of ways to arrange k heads in n tosses without consecutive heads -/
def non_consecutive_heads (n k : ℕ) : ℕ := Nat.choose (n - k + 1) k

/-- The total number of favorable outcomes -/
def total_favorable_outcomes (n : ℕ) : ℕ :=
  (List.range (n/2 + 1)).map (non_consecutive_heads n) |>.sum

/-- The probability of not having consecutive heads in n fair coin tosses -/
def prob_no_consecutive_heads (n : ℕ) : ℚ :=
  (total_favorable_outcomes n : ℚ) / 2^n

/-- The main theorem -/
theorem prob_no_consecutive_heads_10_is_9_64 :
  prob_no_consecutive_heads n = 9/64 := by sorry

end NUMINAMATH_CALUDE_prob_no_consecutive_heads_10_is_9_64_l2643_264385


namespace NUMINAMATH_CALUDE_approximation_theorem_l2643_264336

theorem approximation_theorem (a b : ℝ) (ε : ℝ) (hε : ε > 0) :
  ∃ (k m : ℤ) (n : ℕ), |n • a - k| < ε ∧ |n • b - m| < ε := by
  sorry

end NUMINAMATH_CALUDE_approximation_theorem_l2643_264336


namespace NUMINAMATH_CALUDE_series_sum_l2643_264388

theorem series_sum : 
  (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l2643_264388


namespace NUMINAMATH_CALUDE_base_5_divisible_by_7_l2643_264350

def base_5_to_10 (d : ℕ) : ℕ := 3 * 5^3 + d * 5^2 + d * 5 + 4

theorem base_5_divisible_by_7 :
  ∃! d : ℕ, d < 5 ∧ (base_5_to_10 d) % 7 = 0 :=
by sorry

end NUMINAMATH_CALUDE_base_5_divisible_by_7_l2643_264350


namespace NUMINAMATH_CALUDE_chess_game_draw_probability_l2643_264348

/-- Given a chess game between A and B:
    * The game can end in A winning, B winning, or a draw.
    * The probability of A not losing is 0.6.
    * The probability of B not losing is 0.7.
    This theorem proves that the probability of the game ending in a draw is 0.3. -/
theorem chess_game_draw_probability :
  ∀ (p_a_win p_b_win p_draw : ℝ),
    p_a_win + p_b_win + p_draw = 1 →
    p_a_win + p_draw = 0.6 →
    p_b_win + p_draw = 0.7 →
    p_draw = 0.3 :=
by sorry

end NUMINAMATH_CALUDE_chess_game_draw_probability_l2643_264348


namespace NUMINAMATH_CALUDE_anthony_jim_difference_l2643_264340

/-- The number of pairs of shoes Scott has -/
def scott_shoes : ℕ := 7

/-- The number of pairs of shoes Anthony has -/
def anthony_shoes : ℕ := 3 * scott_shoes

/-- The number of pairs of shoes Jim has -/
def jim_shoes : ℕ := anthony_shoes - 2

/-- Theorem: Anthony has 2 more pairs of shoes than Jim -/
theorem anthony_jim_difference : anthony_shoes - jim_shoes = 2 := by
  sorry

end NUMINAMATH_CALUDE_anthony_jim_difference_l2643_264340


namespace NUMINAMATH_CALUDE_arithmetic_sequence_neither_necessary_nor_sufficient_l2643_264356

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_neither_necessary_nor_sufficient :
  ∃ (a : ℕ → ℝ) (m n p q : ℕ),
    arithmetic_sequence a ∧
    m > 0 ∧ n > 0 ∧ p > 0 ∧ q > 0 ∧
    (a m + a n > a p + a q ∧ m + n ≤ p + q) ∧
    (m + n > p + q ∧ a m + a n ≤ a p + a q) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_neither_necessary_nor_sufficient_l2643_264356
