import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_number_and_square_l1695_169517

theorem sum_of_number_and_square : 
  let n : ℕ := 15
  n + n^2 = 240 := by sorry

end NUMINAMATH_CALUDE_sum_of_number_and_square_l1695_169517


namespace NUMINAMATH_CALUDE_equation_transformation_l1695_169570

theorem equation_transformation (x : ℝ) (y : ℝ) (h1 : x ≠ 0) (h2 : x^2 ≠ 2) :
  y = (x^2 - 2) / x ∧ (x^2 - 2) / x + 2 * x / (x^2 - 2) = 5 → y^2 - 5*y + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l1695_169570


namespace NUMINAMATH_CALUDE_sequence_length_bound_l1695_169568

theorem sequence_length_bound (N : ℕ) (m : ℕ) (a : ℕ → ℕ) :
  (∀ i j, 1 ≤ i → i < j → j ≤ m → a i < a j) →
  (∀ i, 1 ≤ i → i ≤ m → a i ≤ N) →
  (∀ i j, 1 ≤ i → i < j → j ≤ m → Nat.lcm (a i) (a j) ≤ N) →
  m ≤ 2 * Int.floor (Real.sqrt N) :=
by sorry

end NUMINAMATH_CALUDE_sequence_length_bound_l1695_169568


namespace NUMINAMATH_CALUDE_binomial_arithmetic_sequence_l1695_169508

theorem binomial_arithmetic_sequence (n k : ℕ) :
  (∃ (a : ℕ), Nat.choose n (k-1) + a = Nat.choose n k ∧ Nat.choose n k + a = Nat.choose n (k+1)) ↔
  (∃ (u : ℕ), u ≥ 3 ∧ n = u^2 - 2 ∧ (k = Nat.choose u 2 - 1 ∨ k = Nat.choose (u+1) 2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_binomial_arithmetic_sequence_l1695_169508


namespace NUMINAMATH_CALUDE_parabola_m_range_l1695_169541

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadratic function of the form y = ax² + 4ax + c -/
structure QuadraticFunction where
  a : ℝ
  c : ℝ
  h : a ≠ 0

theorem parabola_m_range 
  (f : QuadraticFunction)
  (A B C : Point)
  (h1 : A.x = m)
  (h2 : B.x = m + 2)
  (h3 : C.x = -2)  -- vertex x-coordinate for y = ax² + 4ax + c is always -2
  (h4 : A.y = f.a * A.x^2 + 4 * f.a * A.x + f.c)
  (h5 : B.y = f.a * B.x^2 + 4 * f.a * B.x + f.c)
  (h6 : C.y = f.a * C.x^2 + 4 * f.a * C.x + f.c)
  (h7 : C.y ≥ B.y)
  (h8 : B.y > A.y)
  : m < -3 := by sorry

end NUMINAMATH_CALUDE_parabola_m_range_l1695_169541


namespace NUMINAMATH_CALUDE_like_terms_exponent_l1695_169519

theorem like_terms_exponent (m n : ℕ) : 
  (∃ (x y : ℝ), -7 * x^(m+2) * y^2 = -3 * x^3 * y^n) → m^n = 1 := by
sorry

end NUMINAMATH_CALUDE_like_terms_exponent_l1695_169519


namespace NUMINAMATH_CALUDE_triple_base_exponent_l1695_169567

theorem triple_base_exponent (a b : ℤ) (x : ℚ) (h1 : b ≠ 0) :
  (3 * a) ^ (3 * b) = a ^ b * x ^ b → x = 27 * a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_triple_base_exponent_l1695_169567


namespace NUMINAMATH_CALUDE_polynomial_constant_term_l1695_169502

theorem polynomial_constant_term (a b c d e : ℝ) :
  (2^7 * a + 2^5 * b + 2^3 * c + 2 * d + e = 23) →
  ((-2)^7 * a + (-2)^5 * b + (-2)^3 * c + (-2) * d + e = -35) →
  e = -6 := by sorry

end NUMINAMATH_CALUDE_polynomial_constant_term_l1695_169502


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l1695_169565

theorem no_positive_integer_solutions :
  ∀ A : ℕ, 1 ≤ A → A ≤ 9 →
  ¬∃ x : ℕ, x > 0 ∧ x^2 - (10 * A + 1) * x + (10 * A + A) = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l1695_169565


namespace NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l1695_169557

theorem neither_sufficient_nor_necessary : ¬(∀ x : ℝ, -1 < x ∧ x < 2 → |x - 2| < 1) ∧
                                           ¬(∀ x : ℝ, |x - 2| < 1 → -1 < x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l1695_169557


namespace NUMINAMATH_CALUDE_parabola_point_movement_l1695_169542

/-- Represents a parabola of the form y = x^2 - 2mx - 3 -/
structure Parabola where
  m : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on the parabola -/
def on_parabola (p : Parabola) (pt : Point) : Prop :=
  pt.y = pt.x^2 - 2*p.m*pt.x - 3

/-- Calculates the vertex of the parabola -/
def vertex (p : Parabola) : Point :=
  { x := p.m, y := -(p.m^2) - 3 }

theorem parabola_point_movement (p : Parabola) (A : Point) (n b : ℝ) :
  on_parabola p { x := -2, y := n } →
  { x := 1, y := n - b } = vertex p →
  b = 9 := by sorry

end NUMINAMATH_CALUDE_parabola_point_movement_l1695_169542


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1695_169533

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (5 * x + 9) = 12 → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1695_169533


namespace NUMINAMATH_CALUDE_big_eight_football_league_games_l1695_169585

theorem big_eight_football_league_games (num_divisions : Nat) (teams_per_division : Nat) : 
  num_divisions = 3 → 
  teams_per_division = 4 → 
  (num_divisions * teams_per_division * (teams_per_division - 1) + 
   num_divisions * teams_per_division * (num_divisions - 1) * teams_per_division / 2) = 228 := by
  sorry

#check big_eight_football_league_games

end NUMINAMATH_CALUDE_big_eight_football_league_games_l1695_169585


namespace NUMINAMATH_CALUDE_evas_shoes_l1695_169550

def total_laces : ℕ := 52
def laces_per_pair : ℕ := 2

theorem evas_shoes : 
  total_laces / laces_per_pair = 26 := by sorry

end NUMINAMATH_CALUDE_evas_shoes_l1695_169550


namespace NUMINAMATH_CALUDE_alcohol_solution_percentage_l1695_169525

theorem alcohol_solution_percentage (initial_volume : ℝ) (initial_percentage : ℝ) (added_alcohol : ℝ) : 
  initial_volume = 6 →
  initial_percentage = 0.2 →
  added_alcohol = 3.6 →
  let final_volume := initial_volume + added_alcohol
  let initial_alcohol := initial_volume * initial_percentage
  let final_alcohol := initial_alcohol + added_alcohol
  let final_percentage := final_alcohol / final_volume
  final_percentage = 0.5 := by sorry

end NUMINAMATH_CALUDE_alcohol_solution_percentage_l1695_169525


namespace NUMINAMATH_CALUDE_chair_capacity_l1695_169558

theorem chair_capacity (total_chairs : ℕ) (attended : ℕ) : 
  total_chairs = 40 →
  (2 : ℚ) / 5 * total_chairs = total_chairs - (3 : ℚ) / 5 * total_chairs →
  2 * ((3 : ℚ) / 5 * total_chairs) = attended →
  attended = 48 →
  ∃ (capacity : ℕ), capacity = 48 ∧ capacity * total_chairs = capacity * attended :=
by
  sorry

end NUMINAMATH_CALUDE_chair_capacity_l1695_169558


namespace NUMINAMATH_CALUDE_base_3_to_base_9_first_digit_l1695_169597

def base_3_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (3 ^ i)) 0

def first_digit_base_9 (n : Nat) : Nat :=
  Nat.log 9 n

theorem base_3_to_base_9_first_digit :
  let y : Nat := base_3_to_decimal [2, 0, 2, 2, 1, 1, 2, 2, 2, 0, 1, 2, 0, 1, 0, 1, 1, 1]
  first_digit_base_9 y = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_3_to_base_9_first_digit_l1695_169597


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1695_169546

def polynomial (x : ℝ) : ℝ :=
  3 * (x^8 - 2*x^5 + 4*x^3 - 6) + 5 * (x^4 + 3*x^2 - 2*x) - 4 * (2*x^6 - 5)

theorem sum_of_coefficients : 
  (polynomial 1) = -31 :=
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1695_169546


namespace NUMINAMATH_CALUDE_fourth_square_area_l1695_169573

-- Define the triangles and their properties
structure Triangle (X Y Z : ℝ × ℝ) where
  is_right : (Y.1 - X.1) * (Z.1 - X.1) + (Y.2 - X.2) * (Z.2 - X.2) = 0

-- Define the theorem
theorem fourth_square_area 
  (XYZ : Triangle X Y Z) 
  (XZW : Triangle X Z W) 
  (square1_area : ℝ) 
  (square2_area : ℝ) 
  (square3_area : ℝ) 
  (h1 : square1_area = 25) 
  (h2 : square2_area = 4) 
  (h3 : square3_area = 49) : 
  ∃ (fourth_square_area : ℝ), fourth_square_area = 78 := by
  sorry

end NUMINAMATH_CALUDE_fourth_square_area_l1695_169573


namespace NUMINAMATH_CALUDE_trig_identity_l1695_169589

theorem trig_identity (α : Real) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  2 * (Real.cos (π / 6 + α / 2))^2 - 1 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1695_169589


namespace NUMINAMATH_CALUDE_beverage_selection_probabilities_l1695_169516

/-- The number of cups of Beverage A -/
def num_a : ℕ := 3

/-- The number of cups of Beverage B -/
def num_b : ℕ := 2

/-- The total number of cups -/
def total_cups : ℕ := num_a + num_b

/-- The number of cups to be selected -/
def select_cups : ℕ := 3

/-- The probability of selecting all cups of Beverage A -/
def prob_excellent : ℚ := 1 / 10

/-- The probability of selecting at least 2 cups of Beverage A -/
def prob_good_or_above : ℚ := 7 / 10

theorem beverage_selection_probabilities :
  (Nat.choose total_cups select_cups : ℚ) * prob_excellent = Nat.choose num_a select_cups ∧
  (Nat.choose total_cups select_cups : ℚ) * prob_good_or_above = 
    Nat.choose num_a select_cups + Nat.choose num_a 2 * Nat.choose num_b 1 := by
  sorry

end NUMINAMATH_CALUDE_beverage_selection_probabilities_l1695_169516


namespace NUMINAMATH_CALUDE_yellow_balls_count_l1695_169524

theorem yellow_balls_count (total : ℕ) (white green red purple : ℕ) (prob : ℚ) :
  total = 100 ∧ 
  white = 20 ∧ 
  green = 30 ∧ 
  red = 37 ∧ 
  purple = 3 ∧ 
  prob = 6/10 ∧ 
  (white + green : ℚ) / total + (total - white - green - red - purple : ℚ) / total = prob →
  total - white - green - red - purple = 10 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l1695_169524


namespace NUMINAMATH_CALUDE_min_value_plus_argmin_l1695_169544

open Real

noncomputable def f (x : ℝ) : ℝ := 9 / (8 * cos (2 * x) + 16) - sin x ^ 2

theorem min_value_plus_argmin (m n : ℝ) 
  (hm : ∀ x, f x ≥ m)
  (hn : f n = m)
  (hp : ∀ x, 0 < x → x < n → f x > m) : 
  m + n = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_plus_argmin_l1695_169544


namespace NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_half_l1695_169504

/-- The system of equations has no solution if and only if n = -1/2 -/
theorem no_solution_iff_n_eq_neg_half (n : ℝ) : 
  (∀ x y z : ℝ, ¬(2*n*x + y = 2 ∧ n*y + 2*z = 2 ∧ x + 2*n*z = 2)) ↔ n = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_half_l1695_169504


namespace NUMINAMATH_CALUDE_same_solution_implies_c_value_l1695_169530

theorem same_solution_implies_c_value (x c : ℚ) : 
  (3 * x + 9 = 0) ∧ (c * x^2 - 7 = 6) → c = 13/9 := by sorry

end NUMINAMATH_CALUDE_same_solution_implies_c_value_l1695_169530


namespace NUMINAMATH_CALUDE_horner_operations_degree_5_l1695_169562

/-- The number of operations required to evaluate a polynomial using Horner's method -/
def horner_operations (degree : ℕ) : ℕ :=
  2 * degree

/-- Theorem: The number of operations to evaluate a polynomial of degree 5 using Horner's method is 10 -/
theorem horner_operations_degree_5 :
  horner_operations 5 = 10 := by
  sorry

#eval horner_operations 5

end NUMINAMATH_CALUDE_horner_operations_degree_5_l1695_169562


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l1695_169523

theorem sum_of_four_numbers : 4321 + 3214 + 2143 + 1432 = 11110 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l1695_169523


namespace NUMINAMATH_CALUDE_show_episodes_per_week_l1695_169575

/-- Calculates the number of episodes shown per week given the episode length,
    filming time multiplier, and total filming time for a certain number of weeks. -/
def episodes_per_week (episode_length : ℕ) (filming_multiplier : ℚ) (total_filming_time : ℕ) (num_weeks : ℕ) : ℚ :=
  let filming_time_per_episode : ℚ := episode_length * filming_multiplier
  let total_minutes : ℕ := total_filming_time * 60
  let total_episodes : ℚ := total_minutes / filming_time_per_episode
  total_episodes / num_weeks

/-- Proves that the number of episodes shown each week is 5 under the given conditions. -/
theorem show_episodes_per_week :
  episodes_per_week 20 (3/2) 10 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_show_episodes_per_week_l1695_169575


namespace NUMINAMATH_CALUDE_unique_power_inequality_l1695_169584

theorem unique_power_inequality : ∃! (n : ℕ), n > 0 ∧ ∀ (m : ℕ), m > 0 → n^m ≥ m^n :=
by sorry

end NUMINAMATH_CALUDE_unique_power_inequality_l1695_169584


namespace NUMINAMATH_CALUDE_jason_debt_l1695_169503

def mowing_value (hour : ℕ) : ℕ :=
  match hour % 3 with
  | 1 => 3
  | 2 => 5
  | 0 => 7
  | _ => 0

def total_earnings (hours : ℕ) : ℕ :=
  (List.range hours).map mowing_value |>.sum

theorem jason_debt : total_earnings 25 = 123 := by
  sorry

end NUMINAMATH_CALUDE_jason_debt_l1695_169503


namespace NUMINAMATH_CALUDE_sausages_problem_l1695_169520

def sausages_left (initial : ℕ) : ℕ :=
  let after_monday := initial - (2 * initial / 5)
  let after_tuesday := after_monday - (after_monday / 2)
  let after_wednesday := after_tuesday - (after_tuesday / 4)
  let after_thursday := after_wednesday - (after_wednesday / 3)
  after_thursday - (3 * after_thursday / 5)

theorem sausages_problem : sausages_left 1200 = 72 := by
  sorry

end NUMINAMATH_CALUDE_sausages_problem_l1695_169520


namespace NUMINAMATH_CALUDE_profit_calculation_correct_l1695_169563

/-- Represents the demand scenario --/
inductive DemandScenario
  | High
  | Moderate
  | Low

/-- Calculates the profit for a given demand scenario --/
def calculate_profit (scenario : DemandScenario) : ℚ :=
  let total_cloth : ℕ := 40
  let profit_per_meter : ℚ := 35
  let high_discount : ℚ := 0.1
  let moderate_discount : ℚ := 0.05
  let sales_tax : ℚ := 0.05
  let low_demand_cloth : ℕ := 30

  match scenario with
  | DemandScenario.High =>
    let original_profit := total_cloth * profit_per_meter
    let discounted_profit := original_profit * (1 - high_discount)
    discounted_profit * (1 - sales_tax)
  | DemandScenario.Moderate =>
    let original_profit := total_cloth * profit_per_meter
    let discounted_profit := original_profit * (1 - moderate_discount)
    discounted_profit * (1 - sales_tax)
  | DemandScenario.Low =>
    let original_profit := low_demand_cloth * profit_per_meter
    original_profit * (1 - sales_tax)

theorem profit_calculation_correct :
  (calculate_profit DemandScenario.High = 1197) ∧
  (calculate_profit DemandScenario.Moderate = 1263.5) ∧
  (calculate_profit DemandScenario.Low = 997.5) :=
by sorry

end NUMINAMATH_CALUDE_profit_calculation_correct_l1695_169563


namespace NUMINAMATH_CALUDE_power_of_negative_power_l1695_169595

theorem power_of_negative_power (a : ℝ) : (-a^5)^2 = a^10 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_power_l1695_169595


namespace NUMINAMATH_CALUDE_quadratic_inequality_all_reals_l1695_169512

/-- The quadratic inequality ax² + bx + c < 0 has a solution set of all real numbers
    if and only if a < 0 and b² - 4ac < 0 -/
theorem quadratic_inequality_all_reals
  (a b c : ℝ) (h_a : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c < 0) ↔ (a < 0 ∧ b^2 - 4*a*c < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_all_reals_l1695_169512


namespace NUMINAMATH_CALUDE_min_value_x2_plus_y2_l1695_169599

theorem min_value_x2_plus_y2 (x y : ℝ) (h : x^2 + 2*x*y - y^2 = 7) :
  ∃ (m : ℝ), m = (7 * Real.sqrt 2) / 2 ∧ x^2 + y^2 ≥ m ∧ ∃ (x' y' : ℝ), x'^2 + 2*x'*y' - y'^2 = 7 ∧ x'^2 + y'^2 = m :=
sorry

end NUMINAMATH_CALUDE_min_value_x2_plus_y2_l1695_169599


namespace NUMINAMATH_CALUDE_blood_expires_same_day_l1695_169537

/-- The number of seconds in a day -/
def seconds_per_day : ℕ := 86400

/-- The factorial of 7 -/
def blood_expiration_seconds : ℕ := 7 * 6 * 5 * 4 * 3 * 2 * 1

/-- Proposition: Blood donated at noon expires on the same day -/
theorem blood_expires_same_day : blood_expiration_seconds < seconds_per_day := by
  sorry

#eval blood_expiration_seconds
#eval seconds_per_day

end NUMINAMATH_CALUDE_blood_expires_same_day_l1695_169537


namespace NUMINAMATH_CALUDE_broken_seashells_l1695_169553

theorem broken_seashells (total : ℕ) (unbroken : ℕ) (h1 : total = 6) (h2 : unbroken = 2) :
  total - unbroken = 4 := by
  sorry

end NUMINAMATH_CALUDE_broken_seashells_l1695_169553


namespace NUMINAMATH_CALUDE_square_of_fraction_l1695_169551

theorem square_of_fraction (a b c : ℝ) (hc : c ≠ 0) :
  ((-2 * a^2 * b) / (3 * c))^2 = (4 * a^4 * b^2) / (9 * c^2) := by
  sorry

end NUMINAMATH_CALUDE_square_of_fraction_l1695_169551


namespace NUMINAMATH_CALUDE_fourth_draw_probability_problem_solution_l1695_169571

/-- A box containing red and black balls -/
structure Box where
  red_balls : ℕ
  black_balls : ℕ

/-- The probability of selecting a black ball from a box -/
def prob_black (b : Box) : ℚ :=
  b.black_balls / (b.red_balls + b.black_balls)

/-- The box described in the problem -/
def problem_box : Box :=
  { red_balls := 4, black_balls := 4 }

theorem fourth_draw_probability (b : Box) :
  prob_black b = 1 / 2 →
  (∀ n : ℕ, n > 0 → prob_black { red_balls := b.red_balls - min n b.red_balls,
                                 black_balls := b.black_balls - min n b.black_balls } = 1 / 2) →
  prob_black { red_balls := b.red_balls - min 3 b.red_balls,
               black_balls := b.black_balls - min 3 b.black_balls } = 1 / 2 :=
by sorry

theorem problem_solution :
  prob_black problem_box = 1 / 2 ∧
  (∀ n : ℕ, n > 0 → prob_black { red_balls := problem_box.red_balls - min n problem_box.red_balls,
                                 black_balls := problem_box.black_balls - min n problem_box.black_balls } = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_fourth_draw_probability_problem_solution_l1695_169571


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1695_169500

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_x_value :
  let m : ℝ × ℝ := (4, 2)
  let n : ℝ × ℝ := (x, -3)
  parallel m n → x = -6 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1695_169500


namespace NUMINAMATH_CALUDE_functional_equation_implies_linearity_l1695_169594

theorem functional_equation_implies_linearity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * (f x^2 - f x * f y + f y^2)) :
  ∀ x : ℝ, f (2005 * x) = 2005 * f x :=
sorry

end NUMINAMATH_CALUDE_functional_equation_implies_linearity_l1695_169594


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1695_169538

theorem trigonometric_identities : 
  (2 * Real.sin (67.5 * π / 180) * Real.cos (67.5 * π / 180) = Real.sqrt 2 / 2) ∧
  (1 - 2 * (Real.sin (22.5 * π / 180))^2 = Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1695_169538


namespace NUMINAMATH_CALUDE_contracting_schemes_l1695_169576

def number_of_projects : ℕ := 6
def projects_for_A : ℕ := 3
def projects_for_B : ℕ := 2
def projects_for_C : ℕ := 1

theorem contracting_schemes :
  (number_of_projects.choose projects_for_A) *
  ((number_of_projects - projects_for_A).choose projects_for_B) *
  ((number_of_projects - projects_for_A - projects_for_B).choose projects_for_C) = 60 := by
  sorry

end NUMINAMATH_CALUDE_contracting_schemes_l1695_169576


namespace NUMINAMATH_CALUDE_cheryl_pesto_production_l1695_169531

/-- Prove that Cheryl can make 32 cups of pesto given the harvesting conditions --/
theorem cheryl_pesto_production (basil_per_pesto : ℕ) (basil_per_week : ℕ) (weeks : ℕ)
  (h1 : basil_per_pesto = 4)
  (h2 : basil_per_week = 16)
  (h3 : weeks = 8) :
  (basil_per_week * weeks) / basil_per_pesto = 32 := by
  sorry


end NUMINAMATH_CALUDE_cheryl_pesto_production_l1695_169531


namespace NUMINAMATH_CALUDE_division_by_negative_l1695_169556

theorem division_by_negative : 15 / (-3 : ℤ) = -5 := by sorry

end NUMINAMATH_CALUDE_division_by_negative_l1695_169556


namespace NUMINAMATH_CALUDE_tensor_solution_l1695_169572

/-- Custom operation ⊗ -/
def tensor (a b : ℝ) : ℝ := a * b + a + b^2

theorem tensor_solution :
  ∀ m : ℝ, m > 0 → tensor 1 m = 3 → m = 1 := by sorry

end NUMINAMATH_CALUDE_tensor_solution_l1695_169572


namespace NUMINAMATH_CALUDE_circle_center_correct_l1695_169554

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y - 4 = 0

/-- The center of a circle -/
def circle_center : ℝ × ℝ := (1, -2)

/-- Theorem: The center of the circle defined by the given equation is (1, -2) -/
theorem circle_center_correct :
  ∀ x y : ℝ, circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 9 :=
sorry

end NUMINAMATH_CALUDE_circle_center_correct_l1695_169554


namespace NUMINAMATH_CALUDE_volume_eq_cross_section_area_l1695_169501

/-- A right prism with an equilateral triangular base -/
structure EquilateralPrism where
  /-- Side length of the equilateral triangular base -/
  a : ℝ
  /-- Angle between the cross-section plane and the base -/
  φ : ℝ
  /-- Area of the cross-section -/
  Q : ℝ
  /-- Side length is positive -/
  h_a_pos : 0 < a
  /-- Angle is between 0 and π/2 -/
  h_φ_range : 0 < φ ∧ φ < Real.pi / 2
  /-- Area is positive -/
  h_Q_pos : 0 < Q

/-- The volume of the equilateral prism -/
def volume (p : EquilateralPrism) : ℝ := p.Q

theorem volume_eq_cross_section_area (p : EquilateralPrism) :
  volume p = p.Q := by sorry

end NUMINAMATH_CALUDE_volume_eq_cross_section_area_l1695_169501


namespace NUMINAMATH_CALUDE_cafeteria_fruit_sale_l1695_169532

/-- Cafeteria fruit sale problem -/
theorem cafeteria_fruit_sale
  (initial_apples : ℕ) 
  (initial_oranges : ℕ) 
  (apple_price : ℚ) 
  (orange_price : ℚ) 
  (total_earnings : ℚ) 
  (apples_left : ℕ) 
  (h1 : initial_apples = 50)
  (h2 : initial_oranges = 40)
  (h3 : apple_price = 4/5)
  (h4 : orange_price = 1/2)
  (h5 : total_earnings = 49)
  (h6 : apples_left = 10) :
  initial_oranges - (total_earnings - (initial_apples - apples_left) * apple_price) / orange_price = 6 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_fruit_sale_l1695_169532


namespace NUMINAMATH_CALUDE_unique_six_digit_number_l1695_169515

/-- A function that checks if a number is a 6-digit number beginning and ending with 2 -/
def is_valid_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧ n % 10 = 2 ∧ n / 100000 = 2

/-- A function that checks if a number is the product of three consecutive even integers -/
def is_product_of_three_consecutive_even (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (2*k) * (2*k + 2) * (2*k + 4)

theorem unique_six_digit_number : 
  ∀ n : ℕ, is_valid_number n ∧ is_product_of_three_consecutive_even n ↔ n = 287232 :=
sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_l1695_169515


namespace NUMINAMATH_CALUDE_initial_avg_equals_correct_avg_l1695_169564

-- Define the number of elements
def n : ℕ := 10

-- Define the correct average
def correct_avg : ℚ := 22

-- Define the difference between the correct and misread value
def misread_diff : ℤ := 10

-- Theorem statement
theorem initial_avg_equals_correct_avg :
  let correct_sum := n * correct_avg
  let initial_sum := correct_sum - misread_diff
  initial_sum / n = correct_avg := by
sorry

end NUMINAMATH_CALUDE_initial_avg_equals_correct_avg_l1695_169564


namespace NUMINAMATH_CALUDE_two_segment_journey_average_speed_l1695_169580

/-- Calculates the average speed of a two-segment journey -/
theorem two_segment_journey_average_speed 
  (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) 
  (h1 : distance1 = 20) (h2 : speed1 = 10) (h3 : distance2 = 30) (h4 : speed2 = 20) :
  (distance1 + distance2) / ((distance1 / speed1) + (distance2 / speed2)) = 50 / 3.5 := by
sorry

#eval (20 + 30) / ((20 / 10) + (30 / 20)) -- To verify the result

end NUMINAMATH_CALUDE_two_segment_journey_average_speed_l1695_169580


namespace NUMINAMATH_CALUDE_function_derivative_l1695_169521

/-- Given a function f(x) = α² - cos(x), prove that its derivative f'(x) = sin(x) -/
theorem function_derivative (α : ℝ) : 
  let f : ℝ → ℝ := λ x => α^2 - Real.cos x
  deriv f = λ x => Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_function_derivative_l1695_169521


namespace NUMINAMATH_CALUDE_expression_value_l1695_169596

theorem expression_value (m n : ℝ) 
  (h1 : m^2 + 2*m*n = 3) 
  (h2 : 2*n^2 + 3*m*n = 5) : 
  2*m^2 + 13*m*n + 6*n^2 = 21 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l1695_169596


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l1695_169592

-- Define the quadratic equations
def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x < 0 ∧ y < 0 ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

-- Define the set (-2,0) ∪ (1,3)
def target_set (m : ℝ) : Prop := (m > -2 ∧ m < 0) ∨ (m > 1 ∧ m < 3)

-- State the theorem
theorem quadratic_roots_theorem (m : ℝ) :
  ((p m ∨ q m) ∧ ¬(p m ∧ q m)) ↔ target_set m :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l1695_169592


namespace NUMINAMATH_CALUDE_tims_income_percentage_l1695_169513

theorem tims_income_percentage (tim mart juan : ℝ) 
  (h1 : mart = tim + 0.6 * tim) 
  (h2 : mart = 0.9599999999999999 * juan) : 
  tim = 0.6 * juan := by sorry

end NUMINAMATH_CALUDE_tims_income_percentage_l1695_169513


namespace NUMINAMATH_CALUDE_square_less_than_triple_l1695_169590

theorem square_less_than_triple (n : ℕ) : n > 0 → (n^2 < 3*n ↔ n = 1 ∨ n = 2) := by
  sorry

end NUMINAMATH_CALUDE_square_less_than_triple_l1695_169590


namespace NUMINAMATH_CALUDE_min_sum_squares_l1695_169582

def S : Finset Int := {-8, -6, -4, -1, 1, 3, 5, 14}

theorem min_sum_squares (a b c d e f g h : Int)
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) (hd : d ∈ S)
  (he : e ∈ S) (hf : f ∈ S) (hg : g ∈ S) (hh : h ∈ S)
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
               b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
               c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
               d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
               e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
               f ≠ g ∧ f ≠ h ∧
               g ≠ h) :
  (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1695_169582


namespace NUMINAMATH_CALUDE_largest_packet_size_l1695_169522

theorem largest_packet_size (jonathan_sets elena_sets : ℕ) 
  (h1 : jonathan_sets = 36) 
  (h2 : elena_sets = 60) : 
  Nat.gcd jonathan_sets elena_sets = 12 := by
  sorry

end NUMINAMATH_CALUDE_largest_packet_size_l1695_169522


namespace NUMINAMATH_CALUDE_square_number_ones_digit_l1695_169527

/-- A number is a square number if it's the square of an integer -/
def IsSquareNumber (a : ℕ) : Prop := ∃ x : ℕ, a = x^2

/-- Get the tens digit of a natural number -/
def TensDigit (n : ℕ) : ℕ := (n / 10) % 10

/-- Get the ones digit of a natural number -/
def OnesDigit (n : ℕ) : ℕ := n % 10

/-- A number is odd if it's not divisible by 2 -/
def IsOdd (n : ℕ) : Prop := n % 2 = 1

theorem square_number_ones_digit
  (a : ℕ)
  (h1 : IsSquareNumber a)
  (h2 : IsOdd (TensDigit a)) :
  OnesDigit a = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_number_ones_digit_l1695_169527


namespace NUMINAMATH_CALUDE_room_population_l1695_169510

theorem room_population (total : ℕ) (women : ℕ) (married : ℕ) (unmarried_women : ℕ) :
  women = total / 4 →
  married = 3 * total / 4 →
  unmarried_women ≤ 20 →
  unmarried_women = total - married →
  total = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_room_population_l1695_169510


namespace NUMINAMATH_CALUDE_ash_cloud_radius_l1695_169528

theorem ash_cloud_radius (height : ℝ) (diameter_ratio : ℝ) : 
  height = 300 → diameter_ratio = 18 → (diameter_ratio * height) / 2 = 2700 := by
  sorry

end NUMINAMATH_CALUDE_ash_cloud_radius_l1695_169528


namespace NUMINAMATH_CALUDE_monotone_increasing_iff_a_in_range_l1695_169545

/-- A quadratic function f(x) = ax^2 + 2x - 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x - 3

/-- The statement that f is monotonically increasing on (-∞, 4) iff a ∈ [-1/4, 0] -/
theorem monotone_increasing_iff_a_in_range (a : ℝ) :
  (∀ x y, x < y → x < 4 → f a x < f a y) ↔ -1/4 ≤ a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_monotone_increasing_iff_a_in_range_l1695_169545


namespace NUMINAMATH_CALUDE_ceiling_sum_sqrt_l1695_169587

theorem ceiling_sum_sqrt : ⌈Real.sqrt 5⌉ + ⌈Real.sqrt 50⌉ + ⌈Real.sqrt 500⌉ + ⌈Real.sqrt 1000⌉ = 66 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_sqrt_l1695_169587


namespace NUMINAMATH_CALUDE_integer_solutions_l1695_169566

def is_solution (x y z : ℤ) : Prop :=
  x + y + z = 3 ∧ x^3 + y^3 + z^3 = 3

theorem integer_solutions :
  ∀ x y z : ℤ, is_solution x y z ↔ 
    ((x = 1 ∧ y = 1 ∧ z = 1) ∨
     (x = 4 ∧ y = 4 ∧ z = -5) ∨
     (x = 4 ∧ y = -5 ∧ z = 4) ∨
     (x = -5 ∧ y = 4 ∧ z = 4)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_l1695_169566


namespace NUMINAMATH_CALUDE_f_properties_l1695_169559

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2 + 1/2

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-1/2) (Real.sqrt 3 / 2) ↔
    ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi/4) ∧ f x = y) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1695_169559


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_singleton_one_l1695_169518

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a : ℕ+, x = 2 * a - 1}

theorem M_intersect_N_eq_singleton_one : M ∩ N = {1} := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_singleton_one_l1695_169518


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1695_169583

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, prove the measure of angle A
    and the perimeter of the triangle under specific conditions. -/
theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a * Real.sin B = -Real.sqrt 3 * b * Real.cos A →
  b = 4 →
  S = 2 * Real.sqrt 3 →
  S = (1/2) * b * c * Real.sin A →
  (A = (2/3) * Real.pi ∧ a + b + c = 6 + 2 * Real.sqrt 7) := by
  sorry


end NUMINAMATH_CALUDE_triangle_abc_properties_l1695_169583


namespace NUMINAMATH_CALUDE_infinite_equal_terms_l1695_169549

/-- An infinite sequence with two ends satisfying the given recurrence relation -/
def InfiniteSequence := ℤ → ℝ

/-- The recurrence relation for the sequence -/
def SatisfiesRecurrence (a : InfiniteSequence) : Prop :=
  ∀ k : ℤ, a k = (1/4) * (a (k-1) + a (k+1))

theorem infinite_equal_terms
  (a : InfiniteSequence)
  (h_recurrence : SatisfiesRecurrence a)
  (h_equal : ∃ k p : ℤ, k < p ∧ a k = a p) :
  ∀ n : ℕ, ∃ k p : ℤ, k < p ∧ a (k - n) = a (p + n) :=
sorry

end NUMINAMATH_CALUDE_infinite_equal_terms_l1695_169549


namespace NUMINAMATH_CALUDE_solve_equation_l1695_169540

/-- Custom operation # -/
def hash (a b : ℝ) : ℝ := a * b - b + b^2

/-- Theorem statement -/
theorem solve_equation (x : ℝ) (h : hash x 7 = 63) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1695_169540


namespace NUMINAMATH_CALUDE_P_divisibility_l1695_169552

/-- The polynomial P(x) defined in terms of a and b -/
def P (a b x : ℚ) : ℚ := (a + b) * x^5 + a * b * x^2 + 1

/-- The theorem stating the conditions for P(x) to be divisible by x^2 - 3x + 2 -/
theorem P_divisibility (a b : ℚ) : 
  (∀ x, (x^2 - 3*x + 2) ∣ P a b x) ↔ 
  ((a = -1 ∧ b = 31/28) ∨ (a = 31/28 ∧ b = -1)) :=
sorry

end NUMINAMATH_CALUDE_P_divisibility_l1695_169552


namespace NUMINAMATH_CALUDE_base8_to_base10_157_l1695_169536

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The base-8 representation of the number --/
def base8Number : List Nat := [7, 5, 1]

theorem base8_to_base10_157 :
  base8ToBase10 base8Number = 111 := by
  sorry

end NUMINAMATH_CALUDE_base8_to_base10_157_l1695_169536


namespace NUMINAMATH_CALUDE_class_size_ratio_l1695_169569

/-- Given three classes A, B, and C, prove that the ratio of the size of Class A to Class C is 1/3 -/
theorem class_size_ratio (size_A size_B size_C : ℕ) : 
  size_A = 2 * size_B → 
  size_B = 20 → 
  size_C = 120 → 
  (size_A : ℚ) / size_C = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_class_size_ratio_l1695_169569


namespace NUMINAMATH_CALUDE_congruence_solutions_count_l1695_169543

theorem congruence_solutions_count :
  ∃ (S : Finset ℕ), 
    (∀ x ∈ S, x > 0 ∧ x < 120 ∧ (x + 17) % 38 = 75 % 38) ∧
    (∀ x : ℕ, x > 0 ∧ x < 120 ∧ (x + 17) % 38 = 75 % 38 → x ∈ S) ∧
    Finset.card S = 3 :=
by sorry

end NUMINAMATH_CALUDE_congruence_solutions_count_l1695_169543


namespace NUMINAMATH_CALUDE_direct_square_variation_theorem_l1695_169529

/-- A function representing direct variation with the square of x -/
def direct_square_variation (k : ℝ) (x : ℝ) : ℝ := k * x^2

theorem direct_square_variation_theorem (y : ℝ → ℝ) :
  (∃ k : ℝ, ∀ x, y x = direct_square_variation k x) →
  y 3 = 18 →
  y 6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_direct_square_variation_theorem_l1695_169529


namespace NUMINAMATH_CALUDE_min_sum_squares_l1695_169539

theorem min_sum_squares (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) 
  (heq : a^2 - 2015*a = b^2 - 2015*b) : 
  ∀ x y : ℝ, x > 0 → y > 0 → x ≠ y → x^2 - 2015*x = y^2 - 2015*y → 
  a^2 + b^2 ≤ x^2 + y^2 ∧ a^2 + b^2 = (2015^2) / 2 := by
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1695_169539


namespace NUMINAMATH_CALUDE_partnership_profit_share_l1695_169560

/-- 
Given a partnership where:
- A invests 3 times as much as B
- B invests two-thirds of what C invests
- The total profit is 6600
Prove that B's share of the profit is 1200
-/
theorem partnership_profit_share 
  (c_investment : ℝ) 
  (total_profit : ℝ) 
  (h1 : total_profit = 6600) 
  (h2 : c_investment > 0) : 
  let b_investment := (2/3) * c_investment
  let a_investment := 3 * b_investment
  let total_investment := a_investment + b_investment + c_investment
  b_investment / total_investment * total_profit = 1200 := by
sorry

end NUMINAMATH_CALUDE_partnership_profit_share_l1695_169560


namespace NUMINAMATH_CALUDE_correct_units_l1695_169534

/-- Represents the number of units in a building project -/
structure BuildingProject where
  first_building : ℕ
  second_building : ℕ
  third_building : ℕ
  apartments : ℕ
  condos : ℕ
  townhouses : ℕ
  bungalows : ℕ

/-- Calculates the correct number of units for each type in the building project -/
def calculate_units (project : BuildingProject) : Prop :=
  -- First building conditions
  project.first_building = 4000 ∧
  project.apartments ≥ 2000 ∧
  project.condos ≥ 2000 ∧
  -- Second building conditions
  project.second_building = (2 : ℕ) * project.first_building / 5 ∧
  -- Third building conditions
  project.third_building = (6 : ℕ) * project.second_building / 5 ∧
  project.townhouses = (3 : ℕ) * project.third_building / 5 ∧
  project.bungalows = (2 : ℕ) * project.third_building / 5 ∧
  -- Total units calculation
  project.apartments = 3200 ∧
  project.condos = 2400 ∧
  project.townhouses = 1152 ∧
  project.bungalows = 768

/-- Theorem stating that the calculated units are correct -/
theorem correct_units (project : BuildingProject) : 
  calculate_units project → 
  project.apartments = 3200 ∧ 
  project.condos = 2400 ∧ 
  project.townhouses = 1152 ∧ 
  project.bungalows = 768 := by
  sorry

end NUMINAMATH_CALUDE_correct_units_l1695_169534


namespace NUMINAMATH_CALUDE_travel_cost_for_twenty_days_l1695_169577

/-- Calculate the total travel cost for a given number of working days and one-way trip cost. -/
def totalTravelCost (workingDays : ℕ) (oneWayCost : ℕ) : ℕ :=
  workingDays * (2 * oneWayCost)

/-- Theorem: The total travel cost for 20 working days with a one-way cost of $24 is $960. -/
theorem travel_cost_for_twenty_days :
  totalTravelCost 20 24 = 960 := by
  sorry

end NUMINAMATH_CALUDE_travel_cost_for_twenty_days_l1695_169577


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1695_169509

/-- The function f(x) = x³ - x + 3 -/
def f (x : ℝ) : ℝ := x^3 - x + 3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

theorem tangent_line_equation :
  let P : ℝ × ℝ := (1, 3)
  let m : ℝ := f' P.1
  let tangent_eq (x y : ℝ) : Prop := 2 * x - y + 1 = 0
  tangent_eq P.1 P.2 ∧ ∀ x y, tangent_eq x y ↔ y - P.2 = m * (x - P.1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1695_169509


namespace NUMINAMATH_CALUDE_expression_value_l1695_169526

theorem expression_value (x y : ℤ) (hx : x = -6) (hy : y = -3) :
  4 * (x - y)^2 - x * y = 18 := by sorry

end NUMINAMATH_CALUDE_expression_value_l1695_169526


namespace NUMINAMATH_CALUDE_rectangular_area_equation_l1695_169598

/-- Represents a rectangular area with length and width in meters -/
structure RectangularArea where
  length : ℝ
  width : ℝ

/-- The area of a rectangle is the product of its length and width -/
def area (r : RectangularArea) : ℝ := r.length * r.width

theorem rectangular_area_equation (x : ℝ) :
  let r : RectangularArea := { length := x, width := x - 6 }
  area r = 720 → x * (x - 6) = 720 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_area_equation_l1695_169598


namespace NUMINAMATH_CALUDE_a_collinear_b_l1695_169574

/-- Two 2D vectors are collinear if and only if their cross product is zero -/
def collinear (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 - v.2 * w.1 = 0

/-- The vector a -/
def a : ℝ × ℝ := (1, 2)

/-- The vector b -/
def b : ℝ × ℝ := (-1, -2)

/-- Proof that vectors a and b are collinear -/
theorem a_collinear_b : collinear a b := by
  sorry

end NUMINAMATH_CALUDE_a_collinear_b_l1695_169574


namespace NUMINAMATH_CALUDE_russian_alphabet_sum_sequence_exists_l1695_169555

theorem russian_alphabet_sum_sequence_exists : ∃ (π : Fin 33 → Fin 33), Function.Bijective π ∧
  ∀ (i j : Fin 33), i ≠ j → (π i + i : Fin 33) ≠ (π j + j : Fin 33) := by
  sorry

end NUMINAMATH_CALUDE_russian_alphabet_sum_sequence_exists_l1695_169555


namespace NUMINAMATH_CALUDE_parabola_point_y_coordinate_l1695_169588

theorem parabola_point_y_coordinate 
  (M : ℝ × ℝ) -- Point M with coordinates (x, y)
  (h1 : M.2 = 2 * M.1^2) -- M is on the parabola y = 2x^2
  (h2 : (M.1 - 0)^2 + (M.2 - 1/4)^2 = 1) -- Distance from M to focus (0, 1/4) is 1
  : M.2 = 7/8 := by
sorry

end NUMINAMATH_CALUDE_parabola_point_y_coordinate_l1695_169588


namespace NUMINAMATH_CALUDE_sequence_sum_equals_33_l1695_169511

def arithmetic_sequence (n : ℕ) : ℕ := 3 * n - 2

def geometric_sequence (n : ℕ) : ℕ := 3^(n - 1)

theorem sequence_sum_equals_33 :
  arithmetic_sequence (geometric_sequence 1) +
  arithmetic_sequence (geometric_sequence 2) +
  arithmetic_sequence (geometric_sequence 3) = 33 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_equals_33_l1695_169511


namespace NUMINAMATH_CALUDE_negative_three_less_than_negative_two_l1695_169593

theorem negative_three_less_than_negative_two : -3 < -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_less_than_negative_two_l1695_169593


namespace NUMINAMATH_CALUDE_square_difference_l1695_169579

theorem square_difference (x y : ℝ) 
  (h1 : (x + y)^2 = 81) 
  (h2 : x * y = 18) : 
  (x - y)^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l1695_169579


namespace NUMINAMATH_CALUDE_nominations_distribution_l1695_169547

/-- The number of ways to distribute nominations among schools -/
def distribute_nominations (total_nominations : ℕ) (num_schools : ℕ) : ℕ :=
  Nat.choose (total_nominations - num_schools + num_schools - 1) (num_schools - 1)

/-- Theorem stating that there are 84 ways to distribute 10 nominations among 7 schools -/
theorem nominations_distribution :
  distribute_nominations 10 7 = 84 := by
  sorry

end NUMINAMATH_CALUDE_nominations_distribution_l1695_169547


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_relation_l1695_169506

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ a₂ : ℝ) : Prop :=
  ∃ d : ℝ, a₂ - a₁ = d ∧ a₁ - (-9) = d ∧ (-1) - a₂ = d

-- Define the geometric sequence
def geometric_sequence (b₁ b₂ b₃ : ℝ) : Prop :=
  ∃ r : ℝ, b₁ / (-9) = r ∧ b₂ / b₁ = r ∧ b₃ / b₂ = r ∧ (-1) / b₃ = r

-- State the theorem
theorem arithmetic_geometric_sequence_relation :
  ∀ a₁ a₂ b₁ b₂ b₃ : ℝ,
  arithmetic_sequence a₁ a₂ →
  geometric_sequence b₁ b₂ b₃ →
  a₂ * b₂ - a₁ * b₂ = -8 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_relation_l1695_169506


namespace NUMINAMATH_CALUDE_quadratic_properties_l1695_169578

-- Define the quadratic function
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 4 * m * x + m - 2

-- Define the conditions
theorem quadratic_properties (m : ℝ) (h_m_neq_0 : m ≠ 0) 
  (h_distinct_roots : ∃ M N : ℝ, M ≠ N ∧ quadratic_function m M = 0 ∧ quadratic_function m N = 0)
  (h_passes_through_A : quadratic_function m 3 = 0) :
  -- 1. The value of m is -1
  m = -1 ∧
  -- 2. The vertex coordinates are (2, 1)
  (let vertex_x := 2; let vertex_y := 1;
   quadratic_function m vertex_x = vertex_y ∧
   ∀ x : ℝ, quadratic_function m x ≤ vertex_y) ∧
  -- 3. When m < 0 and MN ≤ 4, the range of m is m < 0
  (m < 0 → ∀ M N : ℝ, M ≠ N → quadratic_function m M = 0 → quadratic_function m N = 0 →
    (M - N)^2 ≤ 4^2 → m < 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1695_169578


namespace NUMINAMATH_CALUDE_factorization_mx_plus_my_l1695_169581

theorem factorization_mx_plus_my (m x y : ℝ) : m * x + m * y = m * (x + y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_mx_plus_my_l1695_169581


namespace NUMINAMATH_CALUDE_whiteboard_numbers_l1695_169548

theorem whiteboard_numbers (n k : ℕ) : 
  n > 0 ∧ k > 0 ∧ k ≤ n ∧ 
  Odd n ∧
  (((n * (n + 1)) / 2 - k) : ℚ) / (n - 1) = 22 →
  n = 43 ∧ k = 22 := by
  sorry

end NUMINAMATH_CALUDE_whiteboard_numbers_l1695_169548


namespace NUMINAMATH_CALUDE_jacksons_vacation_months_l1695_169535

/-- Proves that Jackson's vacation is 15 months away given his saving plan -/
theorem jacksons_vacation_months (total_savings : ℝ) (paychecks_per_month : ℕ) (savings_per_paycheck : ℝ)
  (h1 : total_savings = 3000)
  (h2 : paychecks_per_month = 2)
  (h3 : savings_per_paycheck = 100) :
  (total_savings / savings_per_paycheck) / paychecks_per_month = 15 := by
  sorry

end NUMINAMATH_CALUDE_jacksons_vacation_months_l1695_169535


namespace NUMINAMATH_CALUDE_rice_division_l1695_169507

theorem rice_division (total_weight : ℚ) (num_containers : ℕ) (ounces_per_pound : ℕ) :
  total_weight = 29 / 4 →
  num_containers = 4 →
  ounces_per_pound = 16 →
  (total_weight / num_containers) * ounces_per_pound = 29 := by
sorry

end NUMINAMATH_CALUDE_rice_division_l1695_169507


namespace NUMINAMATH_CALUDE_expression_evaluation_l1695_169561

theorem expression_evaluation (x y : ℕ) (hx : x = 3) (hy : y = 2) :
  3 * x^y + 4 * y^x - 2 * x * y = 47 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1695_169561


namespace NUMINAMATH_CALUDE_sqrt_three_expression_l1695_169586

theorem sqrt_three_expression : 
  (Real.sqrt 3 + 2)^2023 * (Real.sqrt 3 - 2)^2024 = -Real.sqrt 3 + 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_expression_l1695_169586


namespace NUMINAMATH_CALUDE_cone_sphere_ratio_l1695_169505

theorem cone_sphere_ratio (r h : ℝ) (hr : r > 0) : 
  (1 / 3 * π * r^2 * h = 1 / 3 * (4 / 3 * π * r^3)) → h / r = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_sphere_ratio_l1695_169505


namespace NUMINAMATH_CALUDE_max_value_implies_m_l1695_169591

noncomputable def f (x m : ℝ) : ℝ :=
  2 * (Real.cos x) * (Real.cos x) + Real.sqrt 3 * Real.sin (2 * x) + m

theorem max_value_implies_m (h : ∀ x ∈ Set.Icc 0 (Real.pi / 6), f x 1 ≤ 4) :
  ∃ x ∈ Set.Icc 0 (Real.pi / 6), f x 1 = 4 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_m_l1695_169591


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_l1695_169514

/-- Given an arithmetic progression where the sum of the 4th and 12th terms is 10,
    prove that the sum of the first 15 terms is 75. -/
theorem arithmetic_progression_sum (a d : ℝ) : 
  (a + 3*d) + (a + 11*d) = 10 → 
  (15 : ℝ) / 2 * (2*a + 14*d) = 75 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_l1695_169514
