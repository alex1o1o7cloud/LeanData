import Mathlib

namespace NUMINAMATH_CALUDE_exists_number_with_three_prime_factors_l50_5091

def M (n : ℕ) : Set ℕ := {m | n ≤ m ∧ m ≤ n + 9}

def has_at_least_three_prime_factors (k : ℕ) : Prop :=
  ∃ (p q r : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ 
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ k % p = 0 ∧ k % q = 0 ∧ k % r = 0

theorem exists_number_with_three_prime_factors (n : ℕ) (h : n ≥ 93) :
  ∃ k ∈ M n, has_at_least_three_prime_factors k := by
  sorry

end NUMINAMATH_CALUDE_exists_number_with_three_prime_factors_l50_5091


namespace NUMINAMATH_CALUDE_area_between_semicircles_l50_5005

/-- Given a semicircle with diameter D, which is divided into two parts,
    and semicircles constructed on each part inside the given semicircle,
    the area enclosed between the three semicircles is equal to πCD²/4,
    where CD is the length of the perpendicular from the division point to the semicircle. -/
theorem area_between_semicircles (D r : ℝ) (h : 0 < r ∧ r < D) : 
  let R := D / 2
  let area := π * r * (R - r)
  let CD := Real.sqrt (2 * r * (D - r))
  area = π * CD^2 / 4 := by sorry

end NUMINAMATH_CALUDE_area_between_semicircles_l50_5005


namespace NUMINAMATH_CALUDE_shoeing_problem_solution_l50_5059

/-- Represents the shoeing problem with given parameters -/
structure ShoeingProblem where
  blacksmiths : ℕ
  horses : ℕ
  time_per_hoof : ℕ
  hooves_per_horse : ℕ
  min_hooves_on_ground : ℕ

/-- Calculates the minimum time required to shoe all horses -/
def minimum_shoeing_time (problem : ShoeingProblem) : ℕ :=
  let total_hooves := problem.horses * problem.hooves_per_horse
  let total_time := total_hooves * problem.time_per_hoof
  let time_per_blacksmith := total_time / problem.blacksmiths
  let horses_at_once := problem.blacksmiths / problem.hooves_per_horse
  let sets_needed := (problem.horses + horses_at_once - 1) / horses_at_once
  sets_needed * time_per_blacksmith

/-- Theorem stating that for the given problem, the minimum shoeing time is 125 minutes -/
theorem shoeing_problem_solution :
  let problem : ShoeingProblem := {
    blacksmiths := 48,
    horses := 60,
    time_per_hoof := 5,
    hooves_per_horse := 4,
    min_hooves_on_ground := 3
  }
  minimum_shoeing_time problem = 125 := by
  sorry

#eval minimum_shoeing_time {
  blacksmiths := 48,
  horses := 60,
  time_per_hoof := 5,
  hooves_per_horse := 4,
  min_hooves_on_ground := 3
}

end NUMINAMATH_CALUDE_shoeing_problem_solution_l50_5059


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l50_5028

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h1 : ∀ x, ax^2 + b*x + c > 0 ↔ 2 < x ∧ x < 3) 
  (h2 : a < 0) :
  ∀ x, c*x^2 - b*x + a > 0 ↔ -1/2 < x ∧ x < -1/3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l50_5028


namespace NUMINAMATH_CALUDE_inequality_proof_l50_5051

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c = d) :
  a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l50_5051


namespace NUMINAMATH_CALUDE_alice_expected_games_l50_5080

/-- Represents a tournament with n competitors -/
structure Tournament (n : ℕ) where
  skillLevels : Fin n → ℕ
  distinctSkills : ∀ i j, i ≠ j → skillLevels i ≠ skillLevels j

/-- The expected number of games played by a competitor with a given skill level -/
noncomputable def expectedGames (t : Tournament 21) (skillLevel : ℕ) : ℚ :=
  sorry

/-- Theorem stating the expected number of games for Alice -/
theorem alice_expected_games (t : Tournament 21) (h : t.skillLevels 10 = 11) :
  expectedGames t 11 = 47 / 42 :=
sorry

end NUMINAMATH_CALUDE_alice_expected_games_l50_5080


namespace NUMINAMATH_CALUDE_ab_value_l50_5000

theorem ab_value (a b : ℝ) (h1 : a^2 + b^2 = 5) (h2 : a + b = 3) : a * b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l50_5000


namespace NUMINAMATH_CALUDE_cube_root_of_negative_27_l50_5001

theorem cube_root_of_negative_27 : ∃ x : ℝ, x^3 = -27 ∧ x = -3 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_27_l50_5001


namespace NUMINAMATH_CALUDE_triangle_area_perimeter_inequality_triangle_area_perimeter_equality_l50_5012

/-- Represents a triangle with area and perimeter -/
structure Triangle where
  area : ℝ
  perimeter : ℝ

/-- Predicate to check if a triangle is equilateral -/
def IsEquilateral (t : Triangle) : Prop :=
  sorry -- Definition of equilateral triangle

theorem triangle_area_perimeter_inequality (t : Triangle) :
  36 * t.area ≤ t.perimeter^2 * Real.sqrt 3 :=
sorry

theorem triangle_area_perimeter_equality (t : Triangle) :
  36 * t.area = t.perimeter^2 * Real.sqrt 3 ↔ IsEquilateral t :=
sorry

end NUMINAMATH_CALUDE_triangle_area_perimeter_inequality_triangle_area_perimeter_equality_l50_5012


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l50_5052

/-- A hyperbola with equation x^2 - y^2/b^2 = 1 where b > 0 -/
structure Hyperbola where
  b : ℝ
  h_pos : b > 0

/-- The asymptote of a hyperbola is parallel to a line -/
def asymptote_parallel (h : Hyperbola) (m : ℝ) : Prop :=
  ∃ (c : ℝ), ∀ (x y : ℝ), y = m * x + c → (x^2 - y^2 / h.b^2 = 1 → False)

theorem hyperbola_asymptote_slope (h : Hyperbola) 
  (parallel : asymptote_parallel h 2) : h.b = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l50_5052


namespace NUMINAMATH_CALUDE_zigzag_angle_l50_5004

theorem zigzag_angle (ACB FEG DCE DEC : Real) (h1 : ACB = 80)
  (h2 : FEG = 64) (h3 : DCE + 80 + 14 = 180) (h4 : DEC + 64 + 33 = 180) :
  180 - DCE - DEC = 11 := by
  sorry

end NUMINAMATH_CALUDE_zigzag_angle_l50_5004


namespace NUMINAMATH_CALUDE_pizza_cost_distribution_l50_5016

theorem pizza_cost_distribution (total_cost : ℚ) (num_students : ℕ) 
  (price1 price2 : ℚ) (h1 : total_cost = 26) (h2 : num_students = 7) 
  (h3 : price1 = 371/100) (h4 : price2 = 372/100) : 
  ∃ (x y : ℕ), x + y = num_students ∧ 
  x * price1 + y * price2 = total_cost ∧ 
  y = 3 := by sorry

end NUMINAMATH_CALUDE_pizza_cost_distribution_l50_5016


namespace NUMINAMATH_CALUDE_wednesday_temperature_l50_5042

/-- Given the high temperatures for three consecutive days (Monday, Tuesday, Wednesday),
    prove that Wednesday's temperature is 12°C. -/
theorem wednesday_temperature
  (monday tuesday wednesday : ℝ)
  (h1 : tuesday = monday + 4)
  (h2 : wednesday = monday - 6)
  (h3 : tuesday = 22) :
  wednesday = 12 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_temperature_l50_5042


namespace NUMINAMATH_CALUDE_bennett_brothers_count_l50_5095

theorem bennett_brothers_count (aaron_brothers : ℕ) (bennett_brothers : ℕ) 
  (h1 : aaron_brothers = 4)
  (h2 : bennett_brothers = 2 * aaron_brothers - 2) :
  bennett_brothers = 6 := by
sorry

end NUMINAMATH_CALUDE_bennett_brothers_count_l50_5095


namespace NUMINAMATH_CALUDE_four_solutions_l50_5073

/-- The number of solutions to the equation 4/m + 2/n = 1 where m and n are positive integers -/
def num_solutions : ℕ := 4

/-- A function that checks if a pair of positive integers satisfies the equation 4/m + 2/n = 1 -/
def satisfies_equation (m n : ℕ+) : Prop :=
  (4 : ℚ) / m.val + (2 : ℚ) / n.val = 1

/-- The theorem stating that there are exactly 4 solutions to the equation -/
theorem four_solutions :
  ∃! (solutions : Finset (ℕ+ × ℕ+)),
    solutions.card = num_solutions ∧
    ∀ (pair : ℕ+ × ℕ+), pair ∈ solutions ↔ satisfies_equation pair.1 pair.2 :=
sorry

end NUMINAMATH_CALUDE_four_solutions_l50_5073


namespace NUMINAMATH_CALUDE_gold_alloy_percentage_l50_5069

/-- Proves that adding pure gold to an alloy results in a specific gold percentage -/
theorem gold_alloy_percentage 
  (original_weight : ℝ) 
  (original_percentage : ℝ) 
  (added_gold : ℝ) 
  (h1 : original_weight = 48) 
  (h2 : original_percentage = 0.25) 
  (h3 : added_gold = 12) : 
  (original_percentage * original_weight + added_gold) / (original_weight + added_gold) = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_gold_alloy_percentage_l50_5069


namespace NUMINAMATH_CALUDE_probability_of_exactly_three_primes_out_of_five_dice_l50_5085

def is_prime (n : ℕ) : Prop := sorry

def number_of_primes_up_to_20 : ℕ := 8

def probability_of_prime_on_20_sided_die : ℚ := 
  number_of_primes_up_to_20 / 20

def number_of_dice : ℕ := 5

def number_of_dice_showing_prime : ℕ := 3

theorem probability_of_exactly_three_primes_out_of_five_dice : 
  (Nat.choose number_of_dice number_of_dice_showing_prime : ℚ) * 
  (probability_of_prime_on_20_sided_die ^ number_of_dice_showing_prime) *
  ((1 - probability_of_prime_on_20_sided_die) ^ (number_of_dice - number_of_dice_showing_prime)) = 
  5 / 16 :=
sorry

end NUMINAMATH_CALUDE_probability_of_exactly_three_primes_out_of_five_dice_l50_5085


namespace NUMINAMATH_CALUDE_algebraic_expression_proof_l50_5027

theorem algebraic_expression_proof (a b : ℝ) : 
  3 * a^2 + (4 * a * b - a^2) - 2 * (a^2 + 2 * a * b - b^2) = 2 * b^2 ∧
  3 * a^2 + (4 * a * (-2) - a^2) - 2 * (a^2 + 2 * a * (-2) - (-2)^2) = 8 :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expression_proof_l50_5027


namespace NUMINAMATH_CALUDE_burning_time_3x5_grid_l50_5040

/-- Represents a rectangular grid of toothpicks -/
structure ToothpickGrid :=
  (rows : ℕ)
  (cols : ℕ)
  (toothpicks : ℕ)

/-- Represents the burning properties of toothpicks -/
structure BurningProperties :=
  (burn_time : ℕ)  -- Time for one toothpick to burn completely
  (spread_speed : ℕ)  -- Speed at which fire spreads (assumed constant)

/-- Calculates the total burning time for a toothpick grid -/
def total_burning_time (grid : ToothpickGrid) (props : BurningProperties) : ℕ :=
  sorry

/-- Theorem stating the burning time for the specific problem -/
theorem burning_time_3x5_grid :
  ∀ (grid : ToothpickGrid) (props : BurningProperties),
    grid.rows = 3 ∧ 
    grid.cols = 5 ∧ 
    grid.toothpicks = 38 ∧
    props.burn_time = 10 ∧
    props.spread_speed = 1 →
    total_burning_time grid props = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_burning_time_3x5_grid_l50_5040


namespace NUMINAMATH_CALUDE_unique_multiplication_with_repeated_digit_l50_5019

theorem unique_multiplication_with_repeated_digit :
  ∃! (a b c d e f g h i j z : ℕ),
    (0 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ c ∧ c ≤ 9) ∧
    (0 ≤ d ∧ d ≤ 9) ∧ (0 ≤ e ∧ e ≤ 9) ∧ (0 ≤ f ∧ f ≤ 9) ∧
    (0 ≤ g ∧ g ≤ 9) ∧ (0 ≤ h ∧ h ≤ 9) ∧ (0 ≤ i ∧ i ≤ 9) ∧
    (0 ≤ j ∧ j ≤ 9) ∧ (0 ≤ z ∧ z ≤ 9) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
    g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
    h ≠ i ∧ h ≠ j ∧
    i ≠ j ∧
    (a * 1000000 + b * 100000 + z * 10000 + c * 1000 + d * 100 + e * 10 + z) *
    (f * 100000 + g * 10000 + h * 1000 + i * 100 + z * 10 + j) =
    423416204528 :=
by sorry

end NUMINAMATH_CALUDE_unique_multiplication_with_repeated_digit_l50_5019


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_a_plus_2b_necessary_not_sufficient_l50_5021

/-- For all x in [0, 1], a+2b>0 is a necessary but not sufficient condition for ax+b>0 to always hold true -/
theorem necessary_not_sufficient_condition (a b : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → (a * x + b > 0)) ↔ (b > 0 ∧ a + b > 0) :=
by sorry

/-- a+2b>0 is necessary but not sufficient for the above condition -/
theorem a_plus_2b_necessary_not_sufficient (a b : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → (a * x + b > 0)) → (a + 2*b > 0) ∧
  ¬(∀ a b : ℝ, (a + 2*b > 0) → (∀ x : ℝ, x ∈ Set.Icc 0 1 → (a * x + b > 0))) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_a_plus_2b_necessary_not_sufficient_l50_5021


namespace NUMINAMATH_CALUDE_expression_equals_2x_to_4th_l50_5050

theorem expression_equals_2x_to_4th (x : ℝ) :
  let A := x^4 * x^4
  let B := x^4 + x^4
  let C := 2*x^2 + x^2
  let D := 2*x * x^4
  B = 2 * x^4 := by sorry

end NUMINAMATH_CALUDE_expression_equals_2x_to_4th_l50_5050


namespace NUMINAMATH_CALUDE_polynomial_remainder_l50_5071

theorem polynomial_remainder (x : ℝ) : 
  (8 * x^3 - 20 * x^2 + 28 * x - 30) % (4 * x - 8) = 10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l50_5071


namespace NUMINAMATH_CALUDE_max_value_x_plus_reciprocal_l50_5087

theorem max_value_x_plus_reciprocal (x : ℝ) (h : 9 = x^3 + 1/x^3) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ 3 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_plus_reciprocal_l50_5087


namespace NUMINAMATH_CALUDE_valid_assignments_l50_5045

-- Define a type for statements
inductive Statement
| Assign1 : Statement  -- x←1, y←2, z←3
| Assign2 : Statement  -- S^2←4
| Assign3 : Statement  -- i←i+2
| Assign4 : Statement  -- x+1←x

-- Define a predicate for valid assignment statements
def is_valid_assignment (s : Statement) : Prop :=
  match s with
  | Statement.Assign1 => True
  | Statement.Assign2 => False
  | Statement.Assign3 => True
  | Statement.Assign4 => False

-- Theorem stating which statements are valid assignments
theorem valid_assignments :
  (is_valid_assignment Statement.Assign1) ∧
  (¬is_valid_assignment Statement.Assign2) ∧
  (is_valid_assignment Statement.Assign3) ∧
  (¬is_valid_assignment Statement.Assign4) := by
  sorry

end NUMINAMATH_CALUDE_valid_assignments_l50_5045


namespace NUMINAMATH_CALUDE_exists_cycle_l50_5056

structure Team :=
  (id : Nat)

structure Tournament :=
  (teams : Finset Team)
  (score : Team → Nat)
  (beats : Team → Team → Prop)
  (round_robin : ∀ t1 t2 : Team, t1 ≠ t2 → (beats t1 t2 ∨ beats t2 t1))

theorem exists_cycle (t : Tournament) 
  (h : ∃ t1 t2 : Team, t1 ∈ t.teams ∧ t2 ∈ t.teams ∧ t1 ≠ t2 ∧ t.score t1 = t.score t2) :
  ∃ A B C : Team, A ∈ t.teams ∧ B ∈ t.teams ∧ C ∈ t.teams ∧ 
    t.beats A B ∧ t.beats B C ∧ t.beats C A :=
sorry

end NUMINAMATH_CALUDE_exists_cycle_l50_5056


namespace NUMINAMATH_CALUDE_ab_squared_commutes_l50_5079

theorem ab_squared_commutes (a b : ℝ) : a * b^2 - b^2 * a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_squared_commutes_l50_5079


namespace NUMINAMATH_CALUDE_last_two_digits_of_7_power_10_l50_5062

theorem last_two_digits_of_7_power_10 : 7^10 ≡ 49 [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_7_power_10_l50_5062


namespace NUMINAMATH_CALUDE_triangle_properties_l50_5014

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C) →
  -- Given conditions
  Real.sqrt 3 * (a - c * Real.cos B) = b * Real.sin C →
  (1/2) * a * b * Real.sin C = Real.sqrt 3 / 3 →
  a + b = 4 →
  -- Conclusions
  C = π / 3 ∧
  Real.sin A * Real.sin B = 1/12 ∧
  Real.cos A * Real.cos B = 5/12 := by
sorry


end NUMINAMATH_CALUDE_triangle_properties_l50_5014


namespace NUMINAMATH_CALUDE_ball_distribution_l50_5048

/-- Represents the number of ways to distribute balls into boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to place 7 distinguishable balls into 3 boxes,
    where one box is red and the other two are indistinguishable -/
theorem ball_distribution : distribute_balls 7 3 = 64 := by sorry

end NUMINAMATH_CALUDE_ball_distribution_l50_5048


namespace NUMINAMATH_CALUDE_books_on_cart_l50_5024

/-- The number of books on a cart -/
theorem books_on_cart 
  (fiction : ℕ) 
  (non_fiction : ℕ) 
  (autobiographies : ℕ) 
  (picture : ℕ) 
  (h1 : fiction = 5)
  (h2 : non_fiction = fiction + 4)
  (h3 : autobiographies = 2 * fiction)
  (h4 : picture = 11) :
  fiction + non_fiction + autobiographies + picture = 35 := by
sorry

end NUMINAMATH_CALUDE_books_on_cart_l50_5024


namespace NUMINAMATH_CALUDE_three_digit_number_problem_l50_5011

theorem three_digit_number_problem (a b c d e f : ℕ) :
  (100 ≤ 100 * a + 10 * b + c) ∧ (100 * a + 10 * b + c < 1000) ∧
  (100 ≤ 100 * d + 10 * e + f) ∧ (100 * d + 10 * e + f < 1000) ∧
  (a = b + 1) ∧ (b = c + 2) ∧
  ((100 * a + 10 * b + c) * 3 + 4 = 100 * d + 10 * e + f) →
  100 * d + 10 * e + f = 964 := by
  sorry


end NUMINAMATH_CALUDE_three_digit_number_problem_l50_5011


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l50_5099

/-- A rectangle with perimeter 40 and area 96 has dimensions (12, 8) or (8, 12) -/
theorem rectangle_dimensions : 
  ∀ a b : ℝ, 
  (2 * a + 2 * b = 40) →  -- perimeter condition
  (a * b = 96) →          -- area condition
  ((a = 12 ∧ b = 8) ∨ (a = 8 ∧ b = 12)) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l50_5099


namespace NUMINAMATH_CALUDE_solve_star_equation_l50_5033

-- Define the * operation
def star (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

-- State the theorem
theorem solve_star_equation : 
  ∃! x : ℝ, star (x - 4) 1 = 0 ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_solve_star_equation_l50_5033


namespace NUMINAMATH_CALUDE_park_bench_spaces_l50_5003

/-- Calculates the number of available spaces on benches in a park. -/
def availableSpaces (numBenches : ℕ) (capacityPerBench : ℕ) (peopleSitting : ℕ) : ℕ :=
  numBenches * capacityPerBench - peopleSitting

/-- Theorem stating that there are 120 available spaces on the benches. -/
theorem park_bench_spaces :
  availableSpaces 50 4 80 = 120 := by
  sorry

end NUMINAMATH_CALUDE_park_bench_spaces_l50_5003


namespace NUMINAMATH_CALUDE_vector_dot_product_theorem_l50_5032

def orthogonal_unit_vectors (i j : ℝ × ℝ) : Prop :=
  i.1 * j.1 + i.2 * j.2 = 0 ∧ i.1^2 + i.2^2 = 1 ∧ j.1^2 + j.2^2 = 1

def vector_a (i j : ℝ × ℝ) : ℝ × ℝ := (2 * i.1 + 3 * j.1, 2 * i.2 + 3 * j.2)

def vector_b (i j : ℝ × ℝ) (m : ℝ) : ℝ × ℝ := (i.1 - m * j.1, i.2 - m * j.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem vector_dot_product_theorem (i j : ℝ × ℝ) (m : ℝ) :
  orthogonal_unit_vectors i j →
  dot_product (vector_a i j) (vector_b i j m) = 1 →
  m = 1/3 := by sorry

end NUMINAMATH_CALUDE_vector_dot_product_theorem_l50_5032


namespace NUMINAMATH_CALUDE_scalene_triangle_two_angles_less_than_60_l50_5006

/-- A scalene triangle with side lengths in arithmetic progression has two angles less than 60 degrees. -/
theorem scalene_triangle_two_angles_less_than_60 (a d : ℝ) 
  (h_d_pos : d > 0) 
  (h_scalene : a - d ≠ a ∧ a ≠ a + d ∧ a - d ≠ a + d) :
  ∃ (α β : ℝ), α + β + (180 - α - β) = 180 ∧ 
               0 < α ∧ α < 60 ∧ 
               0 < β ∧ β < 60 := by
  sorry


end NUMINAMATH_CALUDE_scalene_triangle_two_angles_less_than_60_l50_5006


namespace NUMINAMATH_CALUDE_twentyFiveCentCoins_l50_5017

/-- Represents the number of coins of each denomination -/
structure CoinCounts where
  five : ℕ
  ten : ℕ
  twentyFive : ℕ

/-- Calculates the total number of coins -/
def totalCoins (c : CoinCounts) : ℕ := c.five + c.ten + c.twentyFive

/-- Calculates the number of different values that can be obtained -/
def differentValues (c : CoinCounts) : ℕ :=
  74 - 4 * c.five - 3 * c.ten

/-- Main theorem -/
theorem twentyFiveCentCoins (c : CoinCounts) :
  totalCoins c = 15 ∧ differentValues c = 30 → c.twentyFive = 2 := by
  sorry

end NUMINAMATH_CALUDE_twentyFiveCentCoins_l50_5017


namespace NUMINAMATH_CALUDE_range_of_a_l50_5070

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x^3 + x^2 + 1 else Real.exp (a * x)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-2) 3, f a x ≤ 2) ∧
  (∃ x ∈ Set.Icc (-2) 3, f a x = 2) →
  a ≤ (1/3) * Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l50_5070


namespace NUMINAMATH_CALUDE_unique_solution_factorial_equation_l50_5044

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem unique_solution_factorial_equation :
  ∃! (k n : ℕ), factorial n + 3 * n + 8 = k^2 ∧ k = 4 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_factorial_equation_l50_5044


namespace NUMINAMATH_CALUDE_area_of_curve_l50_5081

-- Define the curve
def curve (x y : ℝ) : Prop := |x - 1| + |y - 1| = 1

-- Define the area enclosed by the curve
noncomputable def enclosed_area : ℝ := sorry

-- Theorem statement
theorem area_of_curve : enclosed_area = 2 := by sorry

end NUMINAMATH_CALUDE_area_of_curve_l50_5081


namespace NUMINAMATH_CALUDE_no_solution_factorial_equality_l50_5096

theorem no_solution_factorial_equality (n m : ℕ) (h : m ≥ 2) :
  n.factorial ≠ 2^m * m.factorial := by
  sorry

end NUMINAMATH_CALUDE_no_solution_factorial_equality_l50_5096


namespace NUMINAMATH_CALUDE_complex_multiplication_complex_division_l50_5029

-- Define complex numbers
def i : ℂ := Complex.I

-- Part 1
theorem complex_multiplication :
  (1 - 2*i) * (3 + 4*i) * (-2 + i) = -20 + 15*i := by sorry

-- Part 2
theorem complex_division (x a : ℝ) (h1 : 0 < x) (h2 : x < 1) (h3 : x > a) :
  (Complex.ofReal x) / (Complex.ofReal (x - a)) = -1/5 + 2/5*i := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_complex_division_l50_5029


namespace NUMINAMATH_CALUDE_max_parts_quadratic_trinomials_l50_5088

/-- The maximum number of parts into which the coordinate plane can be divided by n quadratic trinomials -/
def max_parts (n : ℕ) : ℕ := n^2 + 1

/-- Theorem: The maximum number of parts into which the coordinate plane can be divided by n quadratic trinomials is n^2 + 1 -/
theorem max_parts_quadratic_trinomials (n : ℕ) :
  max_parts n = n^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_max_parts_quadratic_trinomials_l50_5088


namespace NUMINAMATH_CALUDE_candy_bar_difference_l50_5009

theorem candy_bar_difference (lena nicole kevin : ℕ) : 
  lena = 23 →
  lena + 7 = 4 * kevin →
  nicole = kevin + 6 →
  lena - nicole = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_bar_difference_l50_5009


namespace NUMINAMATH_CALUDE_sum_of_distances_inequality_minimum_value_of_expression_l50_5066

-- Part 1
theorem sum_of_distances_inequality (x y : ℝ) :
  Real.sqrt (x^2 + y^2) + Real.sqrt ((x-1)^2 + y^2) +
  Real.sqrt (x^2 + (y-1)^2) + Real.sqrt ((x-1)^2 + (y-1)^2) ≥ 2 * Real.sqrt 2 :=
sorry

-- Part 2
theorem minimum_value_of_expression :
  ∃ (min : ℝ), min = 8 ∧
  ∀ (a b : ℝ), abs a ≤ Real.sqrt 2 → b > 0 →
  (a - b)^2 + (Real.sqrt (2 - a^2) - 9 / b)^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_sum_of_distances_inequality_minimum_value_of_expression_l50_5066


namespace NUMINAMATH_CALUDE_expansion_and_factorization_l50_5035

theorem expansion_and_factorization :
  (∀ y : ℝ, (y - 1) * (y + 5) = y^2 + 4*y - 5) ∧
  (∀ x y : ℝ, -x^2 + 4*x*y - 4*y^2 = -(x - 2*y)^2) := by
  sorry

end NUMINAMATH_CALUDE_expansion_and_factorization_l50_5035


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l50_5092

-- Define the quadratic function f(x)
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_function_properties :
  ∀ b c : ℝ,
  (f b c 1 = 0) →
  (f b c 3 = 0) →
  (b = -4 ∧ c = 3) ∧
  (∀ x y : ℝ, 2 < x → x < y → f (-4) 3 x < f (-4) 3 y) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l50_5092


namespace NUMINAMATH_CALUDE_chord_length_when_a_is_3_2_symmetrical_circle_equation_l50_5055

-- Define the circle C
def circle_C (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 4*a*y + 4*a^2 + 1 = 0

-- Define the line l
def line_l (a : ℝ) (x y : ℝ) : Prop :=
  a*x + y + 2*a = 0

-- Part 1: Length of chord AB when a = 3/2
theorem chord_length_when_a_is_3_2 :
  ∃ (A B : ℝ × ℝ),
    circle_C (3/2) A.1 A.2 ∧
    circle_C (3/2) B.1 B.2 ∧
    line_l (3/2) A.1 A.2 ∧
    line_l (3/2) B.1 B.2 ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2) = (2*Real.sqrt 39 / 13)^2 :=
sorry

-- Part 2: Equation of symmetrical circle C' when line l is tangent to circle C
theorem symmetrical_circle_equation (a : ℝ) :
  a > 0 →
  (∃ (x₀ y₀ : ℝ), circle_C a x₀ y₀ ∧ line_l a x₀ y₀ ∧
    ∀ (x y : ℝ), circle_C a x y → line_l a x y → x = x₀ ∧ y = y₀) →
  ∃ (x₁ y₁ : ℝ),
    x₁ = -5 ∧ y₁ = Real.sqrt 3 ∧
    ∀ (x y : ℝ), (x - x₁)^2 + (y - y₁)^2 = 3 ↔
      circle_C a (2*x₁ - x) (2*y₁ - y) :=
sorry

end NUMINAMATH_CALUDE_chord_length_when_a_is_3_2_symmetrical_circle_equation_l50_5055


namespace NUMINAMATH_CALUDE_constant_t_equation_l50_5047

theorem constant_t_equation : ∃! t : ℝ, 
  ∀ x : ℝ, (2*x^2 - 3*x + 4)*(5*x^2 + t*x + 9) = 10*x^4 - t^2*x^3 + 23*x^2 - 27*x + 36 ∧ t = -5 := by
  sorry

end NUMINAMATH_CALUDE_constant_t_equation_l50_5047


namespace NUMINAMATH_CALUDE_equation_solution_l50_5030

theorem equation_solution :
  ∀ (A B C : ℕ),
    3 * A - A = 10 →
    B + A = 12 →
    C - B = 6 →
    A ≠ B →
    B ≠ C →
    A ≠ C →
    C = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l50_5030


namespace NUMINAMATH_CALUDE_point_translation_coordinates_equal_l50_5037

theorem point_translation_coordinates_equal (m : ℝ) : 
  let A : ℝ × ℝ := (m, 2)
  let B : ℝ × ℝ := (m + 1, 5)
  (B.1 = B.2) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_point_translation_coordinates_equal_l50_5037


namespace NUMINAMATH_CALUDE_old_edition_pages_l50_5057

theorem old_edition_pages (new_edition : ℕ) (old_edition : ℕ) 
  (h1 : new_edition = 450) 
  (h2 : new_edition = 2 * old_edition - 230) : old_edition = 340 := by
  sorry

end NUMINAMATH_CALUDE_old_edition_pages_l50_5057


namespace NUMINAMATH_CALUDE_hadley_books_added_l50_5008

theorem hadley_books_added (initial_books : ℕ) (borrowed_by_lunch : ℕ) (borrowed_by_evening : ℕ) (remaining_books : ℕ) : 
  initial_books = 100 →
  borrowed_by_lunch = 50 →
  borrowed_by_evening = 30 →
  remaining_books = 60 →
  initial_books - borrowed_by_lunch + (remaining_books + borrowed_by_evening - (initial_books - borrowed_by_lunch)) = 100 :=
by
  sorry

#check hadley_books_added

end NUMINAMATH_CALUDE_hadley_books_added_l50_5008


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l50_5053

/-- Given an incident ray along the line 2x - y + 2 = 0 reflected off the y-axis,
    the equation of the line containing the reflected ray is 2x + y - 2 = 0 -/
theorem reflected_ray_equation (x y : ℝ) :
  (2 * x - y + 2 = 0) →  -- incident ray equation
  (∃ (x' y' : ℝ), 2 * x' + y' - 2 = 0) -- reflected ray equation
  := by sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l50_5053


namespace NUMINAMATH_CALUDE_min_value_cos_squared_minus_sin_squared_l50_5077

theorem min_value_cos_squared_minus_sin_squared :
  ∃ (m : ℝ), (∀ x, m ≤ (Real.cos (x/2))^2 - (Real.sin (x/2))^2) ∧ 
  (∃ x₀, m = (Real.cos (x₀/2))^2 - (Real.sin (x₀/2))^2) ∧
  m = -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_cos_squared_minus_sin_squared_l50_5077


namespace NUMINAMATH_CALUDE_gcd_84_210_l50_5031

theorem gcd_84_210 : Nat.gcd 84 210 = 42 := by
  sorry

end NUMINAMATH_CALUDE_gcd_84_210_l50_5031


namespace NUMINAMATH_CALUDE_attendance_problem_l50_5086

/-- Proves that the number of people who didn't show up is 12 --/
theorem attendance_problem (total_invited : ℕ) (tables_used : ℕ) (table_capacity : ℕ) : 
  total_invited - (tables_used * table_capacity) = 12 :=
by
  sorry

#check attendance_problem 18 2 3

end NUMINAMATH_CALUDE_attendance_problem_l50_5086


namespace NUMINAMATH_CALUDE_min_total_cost_l50_5075

/-- Represents the transportation problem with two warehouses and two construction sites -/
structure TransportationProblem where
  warehouseA_capacity : ℝ
  warehouseB_capacity : ℝ
  siteA_demand : ℝ
  siteB_demand : ℝ
  costA_to_A : ℝ
  costA_to_B : ℝ
  costB_to_A : ℝ
  costB_to_B : ℝ

/-- The specific transportation problem instance -/
def problem : TransportationProblem :=
  { warehouseA_capacity := 800
  , warehouseB_capacity := 1200
  , siteA_demand := 1300
  , siteB_demand := 700
  , costA_to_A := 12
  , costA_to_B := 15
  , costB_to_A := 10
  , costB_to_B := 18
  }

/-- The cost reduction from Warehouse A to Site A -/
def cost_reduction (a : ℝ) : Prop := 2 ≤ a ∧ a ≤ 6

/-- The amount transported from Warehouse A to Site A -/
def transport_amount (x : ℝ) : Prop := 100 ≤ x ∧ x ≤ 800

/-- The theorem stating the minimum total transportation cost after cost reduction -/
theorem min_total_cost (p : TransportationProblem) (a : ℝ) (x : ℝ) 
  (h1 : p = problem) (h2 : cost_reduction a) (h3 : transport_amount x) : 
  ∃ y : ℝ, y = 22400 ∧ ∀ z : ℝ, z ≥ y := by
  sorry

end NUMINAMATH_CALUDE_min_total_cost_l50_5075


namespace NUMINAMATH_CALUDE_problem_statement_l50_5036

theorem problem_statement : |1 - Real.sqrt 3| - Real.sqrt 3 * (Real.sqrt 3 + 1) = -4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l50_5036


namespace NUMINAMATH_CALUDE_parallel_lines_b_value_l50_5067

theorem parallel_lines_b_value (b : ℝ) : 
  (∃ (m₁ m₂ : ℝ), (∀ x y : ℝ, 3 * y - 4 * b = 9 * x ↔ y = m₁ * x + (4 * b / 3)) ∧
                   (∀ x y : ℝ, y - 2 = (b + 10) * x ↔ y = m₂ * x + 2) ∧
                   m₁ = m₂) →
  b = -7 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_b_value_l50_5067


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l50_5083

/-- The y-intercept of the line 4x + 7y = 28 is the point (0, 4). -/
theorem y_intercept_of_line (x y : ℝ) :
  (4 * x + 7 * y = 28) → (x = 0 → y = 4) :=
by sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l50_5083


namespace NUMINAMATH_CALUDE_cos_45_degrees_l50_5002

theorem cos_45_degrees : Real.cos (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_45_degrees_l50_5002


namespace NUMINAMATH_CALUDE_number_exceeding_fraction_l50_5064

theorem number_exceeding_fraction (x : ℚ) : 
  x = (3/8) * x + 35 → x = 56 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_fraction_l50_5064


namespace NUMINAMATH_CALUDE_rem_five_sevenths_three_fourths_l50_5082

def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

theorem rem_five_sevenths_three_fourths :
  rem (5/7) (3/4) = 5/7 := by sorry

end NUMINAMATH_CALUDE_rem_five_sevenths_three_fourths_l50_5082


namespace NUMINAMATH_CALUDE_y₁_less_than_y₂_l50_5041

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := -x^2 - 4*x + 1

/-- y₁ is the y-coordinate of the point (-3, y₁) on the parabola -/
def y₁ : ℝ := parabola (-3)

/-- y₂ is the y-coordinate of the point (-2, y₂) on the parabola -/
def y₂ : ℝ := parabola (-2)

/-- Theorem stating that y₁ < y₂ for the given parabola and points -/
theorem y₁_less_than_y₂ : y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y₁_less_than_y₂_l50_5041


namespace NUMINAMATH_CALUDE_no_valid_tiling_l50_5022

/-- Represents a chessboard with one corner removed -/
def ChessboardWithCornerRemoved := Fin 8 × Fin 8

/-- Represents a trimino (3x1 rectangle) -/
def Trimino := Fin 3 × Fin 1

/-- A tiling of the chessboard with triminos -/
def Tiling := ChessboardWithCornerRemoved → Option Trimino

/-- Predicate to check if a tiling is valid -/
def is_valid_tiling (t : Tiling) : Prop :=
  -- Each square is either covered by a trimino or is the removed corner
  ∀ (x : ChessboardWithCornerRemoved), 
    (x ≠ (7, 7) → t x ≠ none) ∧ 
    (x = (7, 7) → t x = none) ∧
  -- Each trimino covers exactly three squares
  ∀ (p : Trimino), ∃! (x y z : ChessboardWithCornerRemoved), 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    t x = some p ∧ t y = some p ∧ t z = some p

/-- Theorem stating that no valid tiling exists -/
theorem no_valid_tiling : ¬∃ (t : Tiling), is_valid_tiling t := by
  sorry

end NUMINAMATH_CALUDE_no_valid_tiling_l50_5022


namespace NUMINAMATH_CALUDE_x_positive_necessary_not_sufficient_for_abs_x_minus_one_less_than_one_l50_5097

theorem x_positive_necessary_not_sufficient_for_abs_x_minus_one_less_than_one :
  (∃ x : ℝ, x > 0 ∧ ¬(|x - 1| < 1)) ∧
  (∀ x : ℝ, |x - 1| < 1 → x > 0) :=
by sorry

end NUMINAMATH_CALUDE_x_positive_necessary_not_sufficient_for_abs_x_minus_one_less_than_one_l50_5097


namespace NUMINAMATH_CALUDE_rays_dog_walks_66_blocks_per_day_l50_5026

/-- Represents the number of blocks Ray walks in each segment of his route -/
structure RouteSegments where
  toPark : ℕ
  toHighSchool : ℕ
  toHome : ℕ

/-- Calculates the total number of blocks walked in one complete route -/
def totalBlocksPerWalk (route : RouteSegments) : ℕ :=
  route.toPark + route.toHighSchool + route.toHome

/-- Represents Ray's daily dog walking routine -/
structure DailyWalk where
  route : RouteSegments
  frequency : ℕ

/-- Calculates the total number of blocks walked per day -/
def totalBlocksPerDay (daily : DailyWalk) : ℕ :=
  (totalBlocksPerWalk daily.route) * daily.frequency

/-- Theorem: Ray's dog walks 66 blocks each day -/
theorem rays_dog_walks_66_blocks_per_day :
  ∀ (daily : DailyWalk),
    daily.route.toPark = 4 →
    daily.route.toHighSchool = 7 →
    daily.route.toHome = 11 →
    daily.frequency = 3 →
    totalBlocksPerDay daily = 66 := by
  sorry


end NUMINAMATH_CALUDE_rays_dog_walks_66_blocks_per_day_l50_5026


namespace NUMINAMATH_CALUDE_magic_square_d_plus_e_l50_5061

/-- Represents a 3x3 magic square with some known values and variables -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  sum : ℕ
  sum_eq_row1 : sum = 30 + e + 24
  sum_eq_row2 : sum = 15 + c + d
  sum_eq_row3 : sum = a + 28 + b
  sum_eq_col1 : sum = 30 + 15 + a
  sum_eq_col2 : sum = e + c + 28
  sum_eq_col3 : sum = 24 + d + b
  sum_eq_diag1 : sum = 30 + c + b
  sum_eq_diag2 : sum = a + c + 24

theorem magic_square_d_plus_e (sq : MagicSquare) : sq.d + sq.e = 48 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_d_plus_e_l50_5061


namespace NUMINAMATH_CALUDE_max_concert_tickets_l50_5020

theorem max_concert_tickets (ticket_price : ℕ) (budget : ℕ) : 
  ticket_price = 15 → budget = 120 → 
  ∃ (max_tickets : ℕ), max_tickets = 8 ∧ 
    (∀ n : ℕ, n * ticket_price ≤ budget → n ≤ max_tickets) :=
by sorry

end NUMINAMATH_CALUDE_max_concert_tickets_l50_5020


namespace NUMINAMATH_CALUDE_tan_sum_problem_l50_5049

theorem tan_sum_problem (α β : Real) 
  (h1 : Real.tan α = 3) 
  (h2 : Real.tan (α + β) = 2) : 
  Real.tan β = -1/7 := by sorry

end NUMINAMATH_CALUDE_tan_sum_problem_l50_5049


namespace NUMINAMATH_CALUDE_linear_equation_condition_l50_5015

theorem linear_equation_condition (a : ℝ) : 
  (∀ x y : ℝ, ∃ k l m : ℝ, (a^2 - 4) * x^2 + (2 - 3*a) * x + (a + 1) * y + 3*a = k * x + l * y + m) → 
  (a = 2 ∨ a = -2) :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l50_5015


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1001_l50_5098

theorem largest_prime_factor_of_1001 : 
  ∃ (p : Nat), Nat.Prime p ∧ p ∣ 1001 ∧ ∀ (q : Nat), Nat.Prime q → q ∣ 1001 → q ≤ p ∧ p = 13 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1001_l50_5098


namespace NUMINAMATH_CALUDE_train_passing_platform_l50_5090

/-- Calculates the time taken for a train to pass a platform -/
theorem train_passing_platform 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (pole_passing_time : ℝ) 
  (h1 : train_length = 500)
  (h2 : platform_length = 500)
  (h3 : pole_passing_time = 50) :
  (train_length + platform_length) / (train_length / pole_passing_time) = 100 :=
by sorry

end NUMINAMATH_CALUDE_train_passing_platform_l50_5090


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l50_5010

theorem sum_of_fractions_equals_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h_sum : a + b + c = 1) : 
  (a^2 * b^2 / ((a^2 + b*c) * (b^2 + a*c))) + 
  (a^2 * c^2 / ((a^2 + b*c) * (c^2 + a*b))) + 
  (b^2 * c^2 / ((b^2 + a*c) * (c^2 + a*b))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l50_5010


namespace NUMINAMATH_CALUDE_circumscribed_sphere_volume_l50_5025

/-- Given a regular tetrahedron with an inscribed sphere of volume 1 and the ratio of the radius of 
    the circumscribed sphere to the radius of the inscribed sphere is 3:1, 
    the volume of the circumscribed sphere is 27. -/
theorem circumscribed_sphere_volume (r R : ℝ) : 
  (4 / 3) * Real.pi * r^3 = 1 → R / r = 3 → (4 / 3) * Real.pi * R^3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_volume_l50_5025


namespace NUMINAMATH_CALUDE_gravel_path_rate_l50_5089

/-- Calculates the rate per square meter for gravelling a path around a rectangular plot. -/
theorem gravel_path_rate (length width path_width total_cost : ℝ) 
  (h1 : length = 110)
  (h2 : width = 65)
  (h3 : path_width = 2.5)
  (h4 : total_cost = 680) : 
  total_cost / ((length * width) - ((length - 2 * path_width) * (width - 2 * path_width))) = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_gravel_path_rate_l50_5089


namespace NUMINAMATH_CALUDE_candidate_vote_difference_l50_5060

theorem candidate_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 7800 →
  candidate_percentage = 35 / 100 →
  (total_votes : ℚ) * candidate_percentage < (total_votes : ℚ) * (1 - candidate_percentage) →
  (total_votes : ℚ) * (1 - candidate_percentage) - (total_votes : ℚ) * candidate_percentage = 2340 := by
  sorry

end NUMINAMATH_CALUDE_candidate_vote_difference_l50_5060


namespace NUMINAMATH_CALUDE_max_teams_in_tournament_l50_5074

/-- The number of players in each team -/
def players_per_team : ℕ := 3

/-- The maximum number of games that can be played in the tournament -/
def max_games : ℕ := 150

/-- The number of games played between two teams -/
def games_between_teams : ℕ := players_per_team * players_per_team

/-- The maximum number of team pairs that can play within the game limit -/
def max_team_pairs : ℕ := max_games / games_between_teams

/-- The function to calculate the number of unique pairs of teams -/
def team_pairs (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The theorem stating the maximum number of teams that can participate -/
theorem max_teams_in_tournament : 
  ∃ (n : ℕ), n > 0 ∧ team_pairs n ≤ max_team_pairs ∧ 
  ∀ (m : ℕ), m > n → team_pairs m > max_team_pairs :=
sorry

end NUMINAMATH_CALUDE_max_teams_in_tournament_l50_5074


namespace NUMINAMATH_CALUDE_unicorn_flowers_theorem_l50_5039

/-- The number of flowers that bloom per unicorn step -/
def flowers_per_step (num_unicorns : ℕ) (total_distance : ℕ) (step_length : ℕ) (total_flowers : ℕ) : ℚ :=
  total_flowers / (num_unicorns * (total_distance * 1000 / step_length))

/-- Theorem: Given the conditions, 4 flowers bloom per unicorn step -/
theorem unicorn_flowers_theorem (num_unicorns : ℕ) (total_distance : ℕ) (step_length : ℕ) (total_flowers : ℕ)
  (h1 : num_unicorns = 6)
  (h2 : total_distance = 9)
  (h3 : step_length = 3)
  (h4 : total_flowers = 72000) :
  flowers_per_step num_unicorns total_distance step_length total_flowers = 4 := by
  sorry

end NUMINAMATH_CALUDE_unicorn_flowers_theorem_l50_5039


namespace NUMINAMATH_CALUDE_smallest_p_is_12_l50_5054

/-- The set of complex numbers with real part between 1/2 and √2/2 -/
def T : Set ℂ := {z : ℂ | 1/2 ≤ z.re ∧ z.re ≤ Real.sqrt 2 / 2}

/-- The property that for all n ≥ p, there exists z ∈ T such that z^n = 1 -/
def has_root_in_T (p : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ p → ∃ z ∈ T, z^n = 1

/-- 12 is the smallest positive integer satisfying the property -/
theorem smallest_p_is_12 : 
  has_root_in_T 12 ∧ ∀ p : ℕ, 0 < p → p < 12 → ¬has_root_in_T p :=
sorry

end NUMINAMATH_CALUDE_smallest_p_is_12_l50_5054


namespace NUMINAMATH_CALUDE_greatest_common_divisor_under_30_l50_5084

theorem greatest_common_divisor_under_30 : ∃ (d : ℕ), d = 18 ∧ 
  d ∣ 450 ∧ d ∣ 90 ∧ d < 30 ∧ 
  ∀ (x : ℕ), x ∣ 450 ∧ x ∣ 90 ∧ x < 30 → x ≤ d :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_under_30_l50_5084


namespace NUMINAMATH_CALUDE_clock_divisibility_impossible_l50_5046

theorem clock_divisibility_impossible (a b : ℕ) : 
  0 < a → a ≤ 12 → b < 60 → 
  ¬ (∃ k : ℕ, (120 * a + 2 * b) = k * (100 * a + b)) := by
  sorry

end NUMINAMATH_CALUDE_clock_divisibility_impossible_l50_5046


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l50_5063

def U : Finset ℕ := {2, 0, 1, 5}
def A : Finset ℕ := {0, 2}

theorem complement_of_A_in_U :
  (U \ A) = {1, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l50_5063


namespace NUMINAMATH_CALUDE_intersection_ratio_l50_5072

-- Define the slopes and y-intercepts of the two lines
variable (k₁ k₂ : ℝ)

-- Define the condition that the lines intersect on the x-axis
def intersect_on_x_axis (k₁ k₂ : ℝ) : Prop :=
  ∃ x : ℝ, k₁ * x + 4 = 0 ∧ k₂ * x - 2 = 0

-- Theorem statement
theorem intersection_ratio (k₁ k₂ : ℝ) (h : intersect_on_x_axis k₁ k₂) (h₁ : k₁ ≠ 0) (h₂ : k₂ ≠ 0) :
  k₁ / k₂ = -2 :=
sorry

end NUMINAMATH_CALUDE_intersection_ratio_l50_5072


namespace NUMINAMATH_CALUDE_custom_op_three_six_l50_5076

/-- Custom operation @ for positive integers -/
def custom_op (a b : ℕ+) : ℚ :=
  (a.val ^ 2 * b.val : ℚ) / (a.val + b.val)

/-- Theorem stating that 3 @ 6 = 6 -/
theorem custom_op_three_six :
  custom_op 3 6 = 6 := by sorry

end NUMINAMATH_CALUDE_custom_op_three_six_l50_5076


namespace NUMINAMATH_CALUDE_garage_cars_count_l50_5034

theorem garage_cars_count (total_wheels : ℕ) (total_bicycles : ℕ) 
  (bicycle_wheels : ℕ) (car_wheels : ℕ) :
  total_wheels = 82 →
  total_bicycles = 9 →
  bicycle_wheels = 2 →
  car_wheels = 4 →
  ∃ (total_cars : ℕ), 
    total_wheels = (total_bicycles * bicycle_wheels) + (total_cars * car_wheels) ∧
    total_cars = 16 := by
  sorry

end NUMINAMATH_CALUDE_garage_cars_count_l50_5034


namespace NUMINAMATH_CALUDE_third_term_not_unique_l50_5094

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- The product of the first 5 terms of a sequence equals 32 -/
def ProductEquals32 (a : ℕ → ℝ) : Prop :=
  a 1 * a 2 * a 3 * a 4 * a 5 = 32

/-- The third term of a geometric sequence cannot be uniquely determined
    given only that the product of the first 5 terms equals 32 -/
theorem third_term_not_unique (a : ℕ → ℝ) 
    (h1 : GeometricSequence a) (h2 : ProductEquals32 a) :
    ¬∃! x : ℝ, a 3 = x :=
  sorry

end NUMINAMATH_CALUDE_third_term_not_unique_l50_5094


namespace NUMINAMATH_CALUDE_fourth_root_equation_implies_x_power_eight_zero_l50_5078

theorem fourth_root_equation_implies_x_power_eight_zero (x : ℝ) :
  (((1 - x^4 : ℝ)^(1/4) + (1 + x^4 : ℝ)^(1/4)) = 2) → x^8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_implies_x_power_eight_zero_l50_5078


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_l50_5013

theorem reciprocal_of_negative_two :
  ∃ x : ℚ, x * (-2) = 1 ∧ x = -1/2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_l50_5013


namespace NUMINAMATH_CALUDE_congruence_solution_l50_5058

theorem congruence_solution (x : ℤ) : 
  (10 * x + 3) % 18 = 11 % 18 → x % 9 = 8 % 9 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l50_5058


namespace NUMINAMATH_CALUDE_linda_candy_count_l50_5065

def candy_problem (initial : ℕ) (given_away : ℕ) (received : ℕ) : ℕ :=
  initial - given_away + received

theorem linda_candy_count : candy_problem 34 28 15 = 21 := by
  sorry

end NUMINAMATH_CALUDE_linda_candy_count_l50_5065


namespace NUMINAMATH_CALUDE_sum_of_first_n_naturals_l50_5093

theorem sum_of_first_n_naturals (n : ℕ) : 
  (List.range (n + 1)).sum = n * (n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_n_naturals_l50_5093


namespace NUMINAMATH_CALUDE_total_drying_time_in_hours_l50_5023

/-- Time to dry a short-haired dog in minutes -/
def short_hair_time : ℕ := 10

/-- Time to dry a full-haired dog in minutes -/
def full_hair_time : ℕ := 2 * short_hair_time

/-- Time to dry a medium-haired dog in minutes -/
def medium_hair_time : ℕ := 15

/-- Number of short-haired dogs -/
def short_hair_count : ℕ := 12

/-- Number of full-haired dogs -/
def full_hair_count : ℕ := 15

/-- Number of medium-haired dogs -/
def medium_hair_count : ℕ := 8

/-- Total time to dry all dogs in minutes -/
def total_time : ℕ := 
  short_hair_time * short_hair_count + 
  full_hair_time * full_hair_count + 
  medium_hair_time * medium_hair_count

theorem total_drying_time_in_hours : 
  total_time / 60 = 9 := by sorry

end NUMINAMATH_CALUDE_total_drying_time_in_hours_l50_5023


namespace NUMINAMATH_CALUDE_corn_height_after_three_weeks_l50_5018

/-- The height of corn plants after three weeks of growth -/
def cornHeight (firstWeekGrowth : ℕ) : ℕ :=
  let secondWeekGrowth := 2 * firstWeekGrowth
  let thirdWeekGrowth := 4 * secondWeekGrowth
  firstWeekGrowth + secondWeekGrowth + thirdWeekGrowth

/-- Theorem stating that the corn plants grow to 22 inches after three weeks -/
theorem corn_height_after_three_weeks :
  cornHeight 2 = 22 := by
  sorry


end NUMINAMATH_CALUDE_corn_height_after_three_weeks_l50_5018


namespace NUMINAMATH_CALUDE_basketball_team_callbacks_l50_5038

theorem basketball_team_callbacks (girls boys cut : ℕ) 
  (h1 : girls = 17)
  (h2 : boys = 32)
  (h3 : cut = 39) :
  girls + boys - cut = 10 := by
sorry

end NUMINAMATH_CALUDE_basketball_team_callbacks_l50_5038


namespace NUMINAMATH_CALUDE_response_rate_is_sixty_percent_l50_5043

/-- The response rate percentage for a questionnaire mailing --/
def response_rate_percentage (responses_needed : ℕ) (questionnaires_mailed : ℕ) : ℚ :=
  (responses_needed : ℚ) / (questionnaires_mailed : ℚ) * 100

/-- Theorem stating that the response rate percentage is 60% given the specified conditions --/
theorem response_rate_is_sixty_percent 
  (responses_needed : ℕ) 
  (questionnaires_mailed : ℕ) 
  (h1 : responses_needed = 300) 
  (h2 : questionnaires_mailed = 500) : 
  response_rate_percentage responses_needed questionnaires_mailed = 60 := by
  sorry

#eval response_rate_percentage 300 500

end NUMINAMATH_CALUDE_response_rate_is_sixty_percent_l50_5043


namespace NUMINAMATH_CALUDE_soccer_ball_max_height_l50_5007

/-- The height function of a soccer ball kicked vertically -/
def h (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 20

/-- The maximum height achieved by the soccer ball -/
def max_height : ℝ := 40

/-- Theorem stating that the maximum height of the soccer ball is 40 feet -/
theorem soccer_ball_max_height :
  ∀ t : ℝ, h t ≤ max_height :=
by sorry

end NUMINAMATH_CALUDE_soccer_ball_max_height_l50_5007


namespace NUMINAMATH_CALUDE_infinite_perfect_squares_l50_5068

theorem infinite_perfect_squares (k : ℕ+) : 
  ∃ n : ℕ+, ∃ m : ℕ, (n * 2^k.val - 7 : ℤ) = m^2 :=
sorry

end NUMINAMATH_CALUDE_infinite_perfect_squares_l50_5068
