import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l2209_220960

theorem problem_solution (m n : ℝ) 
  (h1 : m^2 + m*n = -3) 
  (h2 : n^2 - 3*m*n = 18) : 
  m^2 + 4*m*n - n^2 = -21 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2209_220960


namespace NUMINAMATH_CALUDE_relay_station_problem_l2209_220972

theorem relay_station_problem (x : ℝ) (h : x > 3) : 
  (∃ (slow_speed fast_speed : ℝ),
    slow_speed > 0 ∧ 
    fast_speed > 0 ∧
    fast_speed = 2 * slow_speed ∧
    900 / (x + 1) = slow_speed ∧
    900 / (x - 3) = fast_speed) ↔ 
  2 * (900 / (x + 1)) = 900 / (x - 3) :=
by sorry

end NUMINAMATH_CALUDE_relay_station_problem_l2209_220972


namespace NUMINAMATH_CALUDE_cube_surface_area_l2209_220917

/-- The surface area of a cube given the sum of its edge lengths -/
theorem cube_surface_area (sum_of_edges : ℝ) (h : sum_of_edges = 36) : 
  6 * (sum_of_edges / 12)^2 = 54 := by
  sorry

#check cube_surface_area

end NUMINAMATH_CALUDE_cube_surface_area_l2209_220917


namespace NUMINAMATH_CALUDE_football_club_arrangements_l2209_220939

theorem football_club_arrangements (n : ℕ) (k : ℕ) 
  (h1 : n = 9) 
  (h2 : k = 2) : 
  (Nat.factorial n) * (Nat.choose n k) = 13063680 :=
by sorry

end NUMINAMATH_CALUDE_football_club_arrangements_l2209_220939


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l2209_220936

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- The slope of the first line y = 5x + 3 -/
def slope1 : ℝ := 5

/-- The slope of the second line y = (3k)x + 7 -/
def slope2 (k : ℝ) : ℝ := 3 * k

/-- Theorem: If the lines y = 5x + 3 and y = (3k)x + 7 are parallel, then k = 5/3 -/
theorem parallel_lines_k_value (k : ℝ) :
  parallel slope1 (slope2 k) → k = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l2209_220936


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_l2209_220904

/-- The function f(x) = x^2 / 2^x is monotonically increasing on the interval (0, 2/ln(2)) -/
theorem monotonic_increasing_interval (f : ℝ → ℝ) : 
  (∀ x, f x = x^2 / 2^x) →
  (∃ a b, a = 0 ∧ b = 2 / Real.log 2 ∧
    ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_l2209_220904


namespace NUMINAMATH_CALUDE_salary_problem_l2209_220975

theorem salary_problem (initial_average : ℝ) (initial_workers : ℕ) (initial_supervisors : ℕ)
  (initial_supervisor_salary : ℝ) (new_supervisor_salary : ℝ) (new_average : ℝ)
  (h1 : initial_average = 430)
  (h2 : initial_workers = 8)
  (h3 : initial_supervisors = 1)
  (h4 : initial_supervisor_salary = 870)
  (h5 : new_supervisor_salary = 870)
  (h6 : new_average = 430) :
  ∃ (new_total : ℕ), new_total = 9 ∧
    new_total * new_average = initial_workers * initial_average + new_supervisor_salary :=
by sorry

end NUMINAMATH_CALUDE_salary_problem_l2209_220975


namespace NUMINAMATH_CALUDE_initial_typists_count_l2209_220971

/-- The number of typists in the initial group -/
def initial_typists : ℕ := 25

/-- The number of letters the initial group can type in 20 minutes -/
def letters_in_20_min : ℕ := 60

/-- The number of typists in the second group -/
def second_group_typists : ℕ := 75

/-- The number of letters the second group can type in 60 minutes -/
def letters_in_60_min : ℕ := 540

/-- The time ratio between the two scenarios -/
def time_ratio : ℚ := 3

theorem initial_typists_count :
  initial_typists * second_group_typists * letters_in_20_min * time_ratio = 
  letters_in_60_min * initial_typists * time_ratio :=
sorry

end NUMINAMATH_CALUDE_initial_typists_count_l2209_220971


namespace NUMINAMATH_CALUDE_circle_center_correct_l2209_220950

/-- The equation of a circle in the form ax² + bx + cy² + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Given a circle equation, return its center -/
def findCenter (eq : CircleEquation) : CircleCenter :=
  sorry

theorem circle_center_correct :
  let eq := CircleEquation.mk 1 4 1 (-6) (-20)
  findCenter eq = CircleCenter.mk (-2) 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_correct_l2209_220950


namespace NUMINAMATH_CALUDE_hash_difference_l2209_220940

-- Define the # operation
def hash (x y : ℤ) : ℤ := x * y - 3 * x + y

-- State the theorem
theorem hash_difference : hash 8 5 - hash 5 8 = -12 := by
  sorry

end NUMINAMATH_CALUDE_hash_difference_l2209_220940


namespace NUMINAMATH_CALUDE_hen_count_l2209_220987

theorem hen_count (total_animals : ℕ) (total_feet : ℕ) (hen_feet : ℕ) (cow_feet : ℕ) 
  (h1 : total_animals = 50)
  (h2 : total_feet = 144)
  (h3 : hen_feet = 2)
  (h4 : cow_feet = 4) :
  ∃ (hens : ℕ) (cows : ℕ),
    hens + cows = total_animals ∧
    hens * hen_feet + cows * cow_feet = total_feet ∧
    hens = 28 :=
by sorry

end NUMINAMATH_CALUDE_hen_count_l2209_220987


namespace NUMINAMATH_CALUDE_eagle_count_theorem_l2209_220926

/-- The total number of unique types of eagles across all sections of the mountain -/
def total_unique_eagles (lower middle upper overlapping : ℕ) : ℕ :=
  lower + middle + upper - overlapping

/-- Theorem stating that the total number of unique types of eagles is 32 -/
theorem eagle_count_theorem (lower middle upper overlapping : ℕ) 
  (h1 : lower = 12)
  (h2 : middle = 8)
  (h3 : upper = 16)
  (h4 : overlapping = 4) :
  total_unique_eagles lower middle upper overlapping = 32 := by
  sorry

end NUMINAMATH_CALUDE_eagle_count_theorem_l2209_220926


namespace NUMINAMATH_CALUDE_rectangle_folding_l2209_220934

/-- Rectangle ABCD with given side lengths -/
structure Rectangle :=
  (A B C D : ℝ × ℝ)
  (is_rectangle : sorry)
  (AD_length : dist A D = 4)
  (AB_length : dist A B = 3)

/-- Point B₁ after folding along diagonal AC -/
def B₁ (rect : Rectangle) : ℝ × ℝ := sorry

/-- Dihedral angle between two planes -/
def dihedral_angle (p₁ p₂ p₃ : ℝ × ℝ) (q₁ q₂ q₃ : ℝ × ℝ) : ℝ := sorry

/-- Distance between two skew lines -/
def skew_line_distance (p₁ p₂ q₁ q₂ : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem rectangle_folding (rect : Rectangle) :
  let b₁ := B₁ rect
  dihedral_angle b₁ rect.D rect.C rect.A rect.C rect.D = Real.arctan (15/16) ∧
  skew_line_distance rect.A b₁ rect.C rect.D = 10 * Real.sqrt 34 / 17 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_folding_l2209_220934


namespace NUMINAMATH_CALUDE_simplify_sqrt_m_squared_n_l2209_220942

theorem simplify_sqrt_m_squared_n
  (m n : ℝ)
  (h1 : m < 0)
  (h2 : m^2 * n ≥ 0) :
  Real.sqrt (m^2 * n) = -m * Real.sqrt n :=
by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_m_squared_n_l2209_220942


namespace NUMINAMATH_CALUDE_decimal_digits_correct_l2209_220976

/-- The number of digits to the right of the decimal point when 5^8 / (10^6 * 216) is expressed as a decimal -/
def decimal_digits : ℕ :=
  let fraction := (5^8 : ℚ) / ((10^6 * 216) : ℚ)
  5

theorem decimal_digits_correct :
  decimal_digits = 5 := by sorry

end NUMINAMATH_CALUDE_decimal_digits_correct_l2209_220976


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l2209_220957

theorem pure_imaginary_product (a : ℝ) : 
  (∃ b : ℝ, (a + Complex.I) * (2 - Complex.I) = Complex.I * b) → a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l2209_220957


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2209_220965

/-- The line l is defined by the equation x + y - 1 = 0 --/
def line_l (x y : ℝ) : Prop := x + y - 1 = 0

/-- A point P lies on line l if its coordinates satisfy the line equation --/
def point_on_line_l (x y : ℝ) : Prop := line_l x y

/-- The specific condition we're examining --/
def specific_condition (x y : ℝ) : Prop := x = 2 ∧ y = -1

theorem sufficient_not_necessary :
  (∀ x y : ℝ, specific_condition x y → point_on_line_l x y) ∧
  ¬(∀ x y : ℝ, point_on_line_l x y → specific_condition x y) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2209_220965


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l2209_220921

def f (x : ℝ) : ℝ := x^3 + x^2 - 5*x - 5

theorem f_strictly_increasing :
  (∀ x y, x < y ∧ ((x < -5/3 ∧ y < -5/3) ∨ (x > 1 ∧ y > 1)) → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l2209_220921


namespace NUMINAMATH_CALUDE_max_sum_of_square_roots_max_sum_of_square_roots_achievable_l2209_220903

theorem max_sum_of_square_roots (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 8) :
  Real.sqrt (3 * a + 2) + Real.sqrt (3 * b + 2) + Real.sqrt (3 * c + 2) ≤ Real.sqrt 78 :=
by sorry

theorem max_sum_of_square_roots_achievable :
  ∃ a b c : ℝ, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 8 ∧
  Real.sqrt (3 * a + 2) + Real.sqrt (3 * b + 2) + Real.sqrt (3 * c + 2) = Real.sqrt 78 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_square_roots_max_sum_of_square_roots_achievable_l2209_220903


namespace NUMINAMATH_CALUDE_rationalize_and_simplify_l2209_220984

theorem rationalize_and_simplify : 
  3 / (Real.sqrt 75 + Real.sqrt 3) = Real.sqrt 3 / 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_simplify_l2209_220984


namespace NUMINAMATH_CALUDE_remainder_of_198_digits_mod_9_l2209_220951

/-- Represents the sequence of digits formed by concatenating consecutive natural numbers -/
def consecutiveDigitSequence (n : ℕ) : List ℕ :=
  sorry

/-- Computes the sum of digits in the sequence up to the nth digit -/
def sumOfDigits (n : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that the sum of the first 198 digits in the sequence,
    when divided by 9, has a remainder of 6 -/
theorem remainder_of_198_digits_mod_9 :
  sumOfDigits 198 % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_198_digits_mod_9_l2209_220951


namespace NUMINAMATH_CALUDE_rectangular_paper_area_l2209_220900

theorem rectangular_paper_area (L W : ℝ) 
  (h1 : L + 2*W = 34) 
  (h2 : 2*L + W = 38) : 
  L * W = 140 := by
sorry

end NUMINAMATH_CALUDE_rectangular_paper_area_l2209_220900


namespace NUMINAMATH_CALUDE_person_y_speed_l2209_220906

-- Define the river and docks
structure River :=
  (current_speed : ℝ)

structure Dock :=
  (position : ℝ)

-- Define the persons and their boats
structure Person :=
  (rowing_speed : ℝ)
  (starting_dock : Dock)

-- Define the scenario
def Scenario (river : River) (x y : Person) :=
  (x.rowing_speed = 6) ∧ 
  (x.starting_dock.position < y.starting_dock.position) ∧
  (∃ t : ℝ, t > 0 ∧ t * (x.rowing_speed - river.current_speed) = t * (y.rowing_speed + river.current_speed)) ∧
  (∃ t : ℝ, t > 0 ∧ t * (y.rowing_speed + river.current_speed) = t * (x.rowing_speed + river.current_speed) + 4 * (y.rowing_speed - x.rowing_speed)) ∧
  (4 * (x.rowing_speed - river.current_speed + y.rowing_speed + river.current_speed) = 16 * (y.rowing_speed - x.rowing_speed))

-- Theorem statement
theorem person_y_speed (river : River) (x y : Person) 
  (h : Scenario river x y) : y.rowing_speed = 10 :=
sorry

end NUMINAMATH_CALUDE_person_y_speed_l2209_220906


namespace NUMINAMATH_CALUDE_unique_root_condition_l2209_220961

/-- The equation ln(x+a) - 4(x+a)^2 + a = 0 has a unique root if and only if a = (3 ln 2 + 1) / 2 -/
theorem unique_root_condition (a : ℝ) :
  (∃! x : ℝ, Real.log (x + a) - 4 * (x + a)^2 + a = 0) ↔ 
  a = (3 * Real.log 2 + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_unique_root_condition_l2209_220961


namespace NUMINAMATH_CALUDE_second_number_20th_row_l2209_220991

/-- The first number in the nth row of the sequence -/
def first_number (n : ℕ) : ℕ := (n + 1)^2 - 1

/-- The second number in the nth row of the sequence -/
def second_number (n : ℕ) : ℕ := first_number n - 1

/-- Theorem stating that the second number in the 20th row is 439 -/
theorem second_number_20th_row : second_number 20 = 439 := by sorry

end NUMINAMATH_CALUDE_second_number_20th_row_l2209_220991


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l2209_220928

theorem bowling_ball_weight (b c : ℝ) 
  (h1 : 8 * b = 5 * c)  -- 8 bowling balls weigh the same as 5 canoes
  (h2 : 3 * c = 135)    -- 3 canoes weigh 135 pounds
  : b = 28.125 :=       -- One bowling ball weighs 28.125 pounds
by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l2209_220928


namespace NUMINAMATH_CALUDE_exp_gt_one_plus_x_l2209_220931

theorem exp_gt_one_plus_x : ∀ x : ℝ, x > 0 → Real.exp x > 1 + x := by sorry

end NUMINAMATH_CALUDE_exp_gt_one_plus_x_l2209_220931


namespace NUMINAMATH_CALUDE_fermat_numbers_coprime_l2209_220992

theorem fermat_numbers_coprime (n m : ℕ) (h : n ≠ m) :
  Nat.gcd (2^(2^(n-1)) + 1) (2^(2^(m-1)) + 1) = 1 := by
sorry

end NUMINAMATH_CALUDE_fermat_numbers_coprime_l2209_220992


namespace NUMINAMATH_CALUDE_solution_for_n_equals_S_plus_U_squared_l2209_220925

def S (n : ℕ) : ℕ := sorry  -- Sum of digits of n

def U (n : ℕ) : ℕ := sorry  -- Unit digit of n

theorem solution_for_n_equals_S_plus_U_squared :
  ∀ n : ℕ, n > 0 → (n = S n + (U n)^2) ↔ (n = 13 ∨ n = 46 ∨ n = 99) := by
  sorry

end NUMINAMATH_CALUDE_solution_for_n_equals_S_plus_U_squared_l2209_220925


namespace NUMINAMATH_CALUDE_experts_win_probability_l2209_220946

/-- The probability of Experts winning a single round -/
def p_win : ℝ := 0.6

/-- The probability of Experts losing a single round -/
def p_lose : ℝ := 1 - p_win

/-- The current score of Experts -/
def experts_score : ℕ := 3

/-- The current score of Viewers -/
def viewers_score : ℕ := 4

/-- The number of rounds needed to win the game -/
def winning_score : ℕ := 6

/-- The probability that Experts will win the game from the current position -/
def experts_win_prob : ℝ :=
  p_win ^ 3 + 3 * p_win ^ 3 * p_lose

theorem experts_win_probability :
  experts_win_prob = 0.4752 :=
sorry

end NUMINAMATH_CALUDE_experts_win_probability_l2209_220946


namespace NUMINAMATH_CALUDE_least_n_for_length_50_l2209_220945

-- Define the points A_n on the x-axis
def A (n : ℕ) : ℝ × ℝ := (0, 0)  -- We only need A_0 for the statement

-- Define the points B_n on y = x^2
def B (n : ℕ) : ℝ × ℝ := sorry

-- Define the property that A_{n-1}B_nA_n is an equilateral triangle
def is_equilateral_triangle (n : ℕ) : Prop := sorry

-- Define the length of A_0A_n
def length_A0An (n : ℕ) : ℝ := sorry

-- The main theorem
theorem least_n_for_length_50 :
  ∃ n : ℕ, (∀ m : ℕ, m < n → length_A0An m < 50) ∧ length_A0An n ≥ 50 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_least_n_for_length_50_l2209_220945


namespace NUMINAMATH_CALUDE_pauls_diner_cost_l2209_220935

/-- Represents the pricing and discount policy at Paul's Diner -/
structure PaulsDiner where
  sandwich_price : ℕ
  soda_price : ℕ
  discount_threshold : ℕ
  discount_amount : ℕ

/-- Calculates the total cost for a purchase at Paul's Diner -/
def total_cost (diner : PaulsDiner) (sandwiches : ℕ) (sodas : ℕ) : ℕ :=
  let sandwich_cost := diner.sandwich_price * sandwiches
  let soda_cost := diner.soda_price * sodas
  let subtotal := sandwich_cost + soda_cost
  if sandwiches > diner.discount_threshold then
    subtotal - diner.discount_amount
  else
    subtotal

/-- Theorem stating that the total cost for 6 sandwiches and 3 sodas is 29 -/
theorem pauls_diner_cost :
  ∃ (d : PaulsDiner), total_cost d 6 3 = 29 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_pauls_diner_cost_l2209_220935


namespace NUMINAMATH_CALUDE_purely_imaginary_iff_a_eq_one_l2209_220993

theorem purely_imaginary_iff_a_eq_one (a : ℝ) : 
  (Complex.I * (a * Complex.I) = (a^2 - a) + a * Complex.I) ↔ a = 1 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_iff_a_eq_one_l2209_220993


namespace NUMINAMATH_CALUDE_soris_population_2080_l2209_220996

/-- The population growth function for Soris island -/
def soris_population (initial_population : ℕ) (years_passed : ℕ) : ℕ :=
  initial_population * (2 ^ (years_passed / 20))

/-- Theorem stating the population of Soris in 2080 -/
theorem soris_population_2080 :
  soris_population 500 80 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_soris_population_2080_l2209_220996


namespace NUMINAMATH_CALUDE_angle_C_measure_l2209_220995

-- Define the angles A, B, and C
variable (A B C : ℝ)

-- Define the parallel lines condition
variable (p_parallel_q : Bool)

-- State the theorem
theorem angle_C_measure :
  p_parallel_q = true →
  A = (1/4) * B →
  B + C = 180 →
  C = 36 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l2209_220995


namespace NUMINAMATH_CALUDE_population_model_steps_l2209_220912

/- Define the steps as an inductive type -/
inductive ModelingStep
  | observe : ModelingStep
  | test : ModelingStep
  | propose : ModelingStep
  | express : ModelingStep

/- Define a function to represent the correct order of steps -/
def correct_order : List ModelingStep :=
  [ModelingStep.observe, ModelingStep.propose, ModelingStep.express, ModelingStep.test]

/- Define a predicate to check if a given order is correct -/
def is_correct_order (order : List ModelingStep) : Prop :=
  order = correct_order

/- Theorem stating that the specified order is correct -/
theorem population_model_steps :
  is_correct_order [ModelingStep.observe, ModelingStep.propose, ModelingStep.express, ModelingStep.test] :=
by sorry

end NUMINAMATH_CALUDE_population_model_steps_l2209_220912


namespace NUMINAMATH_CALUDE_guessing_game_scores_l2209_220963

-- Define the players and their scores
def Hajar : ℕ := 42
def Farah : ℕ := Hajar + 24
def Sami : ℕ := Farah + 18

-- Theorem statement
theorem guessing_game_scores :
  Hajar = 42 ∧ Farah = 66 ∧ Sami = 84 ∧
  Farah - Hajar = 24 ∧ Sami - Farah = 18 ∧
  Farah > Hajar ∧ Sami > Hajar :=
by
  sorry


end NUMINAMATH_CALUDE_guessing_game_scores_l2209_220963


namespace NUMINAMATH_CALUDE_colton_sticker_distribution_l2209_220947

/-- Proves that Colton gave 4 stickers to each of his 3 friends --/
theorem colton_sticker_distribution :
  ∀ (initial_stickers : ℕ) 
    (friends : ℕ) 
    (remaining_stickers : ℕ) 
    (stickers_to_friend : ℕ),
  initial_stickers = 72 →
  friends = 3 →
  remaining_stickers = 42 →
  initial_stickers = 
    remaining_stickers + 
    (friends * stickers_to_friend) + 
    (friends * stickers_to_friend + 2) + 
    (friends * stickers_to_friend - 8) →
  stickers_to_friend = 4 := by
sorry

end NUMINAMATH_CALUDE_colton_sticker_distribution_l2209_220947


namespace NUMINAMATH_CALUDE_function_properties_l2209_220962

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x) * Real.cos (ω * x) - 2 * Real.sqrt 3 * (Real.sin (ω * x))^2 + Real.sqrt 3

def is_symmetry_axis (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem function_properties (ω : ℝ) (h_ω : ω > 0) 
  (h_symmetry : ∃ x₁ x₂, is_symmetry_axis (f ω) x₁ ∧ is_symmetry_axis (f ω) x₂)
  (h_min_dist : ∃ x₁ x₂, is_symmetry_axis (f ω) x₁ ∧ is_symmetry_axis (f ω) x₂ ∧ |x₁ - x₂| ≥ π/2 ∧ 
    ∀ y₁ y₂, is_symmetry_axis (f ω) y₁ ∧ is_symmetry_axis (f ω) y₂ → |y₁ - y₂| ≥ |x₁ - x₂|) :
  (ω = 1) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (-5*π/12 + k*π) (π/12 + k*π), 
    ∀ y ∈ Set.Icc (-5*π/12 + k*π) (π/12 + k*π), x < y → (f ω x) < (f ω y)) ∧
  (∀ α : ℝ, f ω α = 2/3 → Real.sin (5*π/6 - 4*α) = -7/9) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l2209_220962


namespace NUMINAMATH_CALUDE_machine_tool_supervision_probability_l2209_220914

theorem machine_tool_supervision_probability :
  let p_no_supervision : ℝ := 0.8000
  let n_tools : ℕ := 4
  let p_at_most_two_require_supervision : ℝ := 1 - (Nat.choose n_tools 3 * (1 - p_no_supervision)^3 * p_no_supervision + Nat.choose n_tools 4 * (1 - p_no_supervision)^4)
  p_at_most_two_require_supervision = 0.9728 := by
sorry

end NUMINAMATH_CALUDE_machine_tool_supervision_probability_l2209_220914


namespace NUMINAMATH_CALUDE_sheet_area_calculation_l2209_220937

/-- Represents a rectangular sheet of paper. -/
structure Sheet where
  length : ℝ
  width : ℝ

/-- Represents the perimeters of the three rectangles after folding. -/
structure Perimeters where
  p1 : ℝ
  p2 : ℝ
  p3 : ℝ

/-- Calculates the perimeters of the three rectangles after folding. -/
def calculatePerimeters (s : Sheet) : Perimeters :=
  { p1 := 2 * s.length,
    p2 := 2 * s.width,
    p3 := 2 * (s.length - s.width) }

/-- The main theorem stating the conditions and the result to be proved. -/
theorem sheet_area_calculation (s : Sheet) :
  let p := calculatePerimeters s
  p.p1 = p.p2 + 20 ∧ p.p2 = p.p3 + 16 →
  s.length * s.width = 504 := by
  sorry


end NUMINAMATH_CALUDE_sheet_area_calculation_l2209_220937


namespace NUMINAMATH_CALUDE_ellens_snack_calories_l2209_220916

/-- Calculates the calories of an afternoon snack given the total daily allowance and the calories consumed in other meals. -/
def afternoon_snack_calories (daily_allowance breakfast lunch dinner : ℕ) : ℕ :=
  daily_allowance - breakfast - lunch - dinner

/-- Proves that Ellen's afternoon snack was 130 calories given her daily allowance and other meal calorie counts. -/
theorem ellens_snack_calories :
  afternoon_snack_calories 2200 353 885 832 = 130 := by
  sorry

end NUMINAMATH_CALUDE_ellens_snack_calories_l2209_220916


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l2209_220979

/-- Given a man's speed with the current and the speed of the current, 
    calculates the man's speed against the current. -/
def speed_against_current (speed_with_current speed_of_current : ℝ) : ℝ :=
  speed_with_current - 2 * speed_of_current

/-- Theorem stating that given the specific speeds in the problem, 
    the man's speed against the current is 12 km/hr. -/
theorem mans_speed_against_current :
  speed_against_current 22 5 = 12 := by
  sorry

#eval speed_against_current 22 5

end NUMINAMATH_CALUDE_mans_speed_against_current_l2209_220979


namespace NUMINAMATH_CALUDE_train_distance_l2209_220982

/-- Proves that a train traveling at a rate of 1 mile per 1.5 minutes will cover 40 miles in 60 minutes -/
theorem train_distance (rate : ℝ) (time : ℝ) (distance : ℝ) : 
  rate = 1 / 1.5 → time = 60 → distance = rate * time → distance = 40 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_l2209_220982


namespace NUMINAMATH_CALUDE_probability_of_even_sum_l2209_220911

theorem probability_of_even_sum (p1 p2 : ℝ) 
  (h1 : p1 = 1/2)  -- Probability of even number from first wheel
  (h2 : p2 = 1/3)  -- Probability of even number from second wheel
  : p1 * p2 + (1 - p1) * (1 - p2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_even_sum_l2209_220911


namespace NUMINAMATH_CALUDE_cube_cut_forms_regular_hexagons_l2209_220905

-- Define a cube
structure Cube where
  side : ℝ
  side_positive : side > 0

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a plane in 3D space
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define a regular hexagon
structure RegularHexagon where
  side : ℝ
  side_positive : side > 0

-- Function to get midpoints of cube edges
def getMidpoints (c : Cube) : List Point3D :=
  sorry

-- Function to define a plane through midpoints
def planeThroughMidpoints (midpoints : List Point3D) : Plane3D :=
  sorry

-- Function to determine if a plane intersects a cube to form regular hexagons
def intersectionFormsRegularHexagons (c : Cube) (p : Plane3D) : Prop :=
  sorry

-- Theorem statement
theorem cube_cut_forms_regular_hexagons (c : Cube) :
  let midpoints := getMidpoints c
  let cuttingPlane := planeThroughMidpoints midpoints
  intersectionFormsRegularHexagons c cuttingPlane :=
sorry

end NUMINAMATH_CALUDE_cube_cut_forms_regular_hexagons_l2209_220905


namespace NUMINAMATH_CALUDE_complex_sum_conjugate_l2209_220949

open Complex

theorem complex_sum_conjugate (α β γ : ℝ) 
  (h : exp (I * α) + exp (I * β) + exp (I * γ) = (1 / 3 : ℂ) + (1 / 2 : ℂ) * I) : 
  exp (-I * α) + exp (-I * β) + exp (-I * γ) = (1 / 3 : ℂ) - (1 / 2 : ℂ) * I := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_conjugate_l2209_220949


namespace NUMINAMATH_CALUDE_stock_decrease_duration_l2209_220973

/-- The number of bicycles the stock decreases each month -/
def monthly_decrease : ℕ := 2

/-- The number of months from January 1 to September 1 -/
def months_jan_to_sep : ℕ := 8

/-- The total decrease in bicycles from January 1 to September 1 -/
def total_decrease : ℕ := 18

/-- The number of months the stock has been decreasing -/
def months_decreasing : ℕ := 1

theorem stock_decrease_duration :
  monthly_decrease * months_decreasing + monthly_decrease * months_jan_to_sep = total_decrease :=
by sorry

end NUMINAMATH_CALUDE_stock_decrease_duration_l2209_220973


namespace NUMINAMATH_CALUDE_triangle_properties_l2209_220985

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Main theorem about the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.c * (1 + Real.cos t.A) = Real.sqrt 3 * t.a * Real.sin t.C)
  (h2 : t.a = Real.sqrt 7)
  (h3 : t.b = 1) :
  t.A = π / 3 ∧ 
  (1 / 2 : ℝ) * t.b * t.c * Real.sin t.A = (3 : ℝ) * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2209_220985


namespace NUMINAMATH_CALUDE_jellybean_count_l2209_220953

theorem jellybean_count (nephews nieces jellybeans_per_child : ℕ) 
  (h1 : nephews = 3)
  (h2 : nieces = 2)
  (h3 : jellybeans_per_child = 14) :
  (nephews + nieces) * jellybeans_per_child = 70 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_count_l2209_220953


namespace NUMINAMATH_CALUDE_balance_after_school_days_l2209_220977

/-- Represents the balance after spending money for a certain number of days. -/
def balance (initial_balance : ℝ) (daily_spending : ℝ) (days : ℝ) : ℝ :=
  initial_balance - daily_spending * days

/-- Theorem stating the relationship between balance and days spent at school. -/
theorem balance_after_school_days 
  (initial_balance : ℝ) 
  (daily_spending : ℝ) 
  (days : ℝ) 
  (h1 : initial_balance = 200)
  (h2 : daily_spending = 36)
  (h3 : 0 ≤ days)
  (h4 : days ≤ 5) :
  balance initial_balance daily_spending days = 200 - 36 * days :=
by sorry

end NUMINAMATH_CALUDE_balance_after_school_days_l2209_220977


namespace NUMINAMATH_CALUDE_fruit_display_total_l2209_220902

/-- Proves that the total number of fruits on a display is 35, given the specified conditions. -/
theorem fruit_display_total (bananas oranges apples : ℕ) : 
  bananas = 5 → 
  oranges = 2 * bananas → 
  apples = 2 * oranges → 
  bananas + oranges + apples = 35 := by
sorry

end NUMINAMATH_CALUDE_fruit_display_total_l2209_220902


namespace NUMINAMATH_CALUDE_train_crossing_time_l2209_220943

/-- Represents the time it takes for a train to cross a tree given its length and the time it takes to pass a platform of known length. -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (platform_crossing_time : ℝ) 
  (h1 : train_length = 1200)
  (h2 : platform_length = 1000)
  (h3 : platform_crossing_time = 220) :
  (train_length / ((train_length + platform_length) / platform_crossing_time)) = 120 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l2209_220943


namespace NUMINAMATH_CALUDE_two_digit_multiplication_swap_l2209_220929

theorem two_digit_multiplication_swap (a b c d : Nat) : 
  (a ≥ 1 ∧ a ≤ 9) →
  (b ≥ 0 ∧ b ≤ 9) →
  (c ≥ 1 ∧ c ≤ 9) →
  (d ≥ 0 ∧ d ≤ 9) →
  ((10 * a + b) * (10 * c + d) - (10 * b + a) * (10 * c + d) = 4248) →
  ((10 * a + b) * (10 * c + d) = 5369 ∨ (10 * a + b) * (10 * c + d) = 4720) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_multiplication_swap_l2209_220929


namespace NUMINAMATH_CALUDE_empty_solution_set_range_l2209_220922

theorem empty_solution_set_range (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x - 2| > a^2 + a + 1) ↔ -1 < a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_range_l2209_220922


namespace NUMINAMATH_CALUDE_candy_distribution_count_l2209_220956

/-- The number of ways to distribute n distinct items among k bags, where each bag must receive at least one item. -/
def distribute (n k : ℕ) : ℕ := k^n - k * ((k-1)^n - (k-1))

/-- The number of ways to distribute 9 distinct pieces of candy among 3 bags, where each bag must receive at least one piece of candy. -/
def candy_distribution : ℕ := distribute 9 3

theorem candy_distribution_count : candy_distribution = 18921 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_count_l2209_220956


namespace NUMINAMATH_CALUDE_max_a_value_l2209_220970

/-- A lattice point in an xy-coordinate system is any point (x, y) where both x and y are integers. -/
def is_lattice_point (x y : ℤ) : Prop := True

/-- The equation y = mx + 3 -/
def equation (m : ℚ) (x y : ℤ) : Prop := y = m * x + 3

/-- The condition that the equation has no lattice point solutions for 0 < x ≤ 150 -/
def no_lattice_solutions (m : ℚ) : Prop :=
  ∀ x y : ℤ, 0 < x → x ≤ 150 → is_lattice_point x y → ¬equation m x y

/-- The theorem stating that 101/150 is the maximum value of a satisfying the given conditions -/
theorem max_a_value : 
  (∃ a : ℚ, a = 101/150 ∧ 
    (∀ m : ℚ, 2/3 < m → m < a → no_lattice_solutions m) ∧
    (∀ b : ℚ, b > a → ∃ m : ℚ, 2/3 < m ∧ m < b ∧ ¬no_lattice_solutions m)) :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l2209_220970


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l2209_220954

/-- The area of the triangle formed by y = 3x - 6, y = -4x + 24, and the y-axis -/
theorem triangle_area : ℝ → Prop :=
  λ area : ℝ =>
    let line1 : ℝ → ℝ := λ x => 3 * x - 6
    let line2 : ℝ → ℝ := λ x => -4 * x + 24
    let y_axis : ℝ → ℝ := λ x => 0
    let intersection_x : ℝ := 30 / 7
    let intersection_y : ℝ := line1 intersection_x
    let y_intercept1 : ℝ := line1 0
    let y_intercept2 : ℝ := line2 0
    area = 450 / 7 ∧
    area = (1 / 2) * (y_intercept2 - y_intercept1) * intersection_x

/-- Proof of the triangle area theorem -/
theorem triangle_area_proof : triangle_area (450 / 7) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l2209_220954


namespace NUMINAMATH_CALUDE_z_gets_30_paisa_l2209_220959

/-- Represents the division of money among three parties -/
structure MoneyDivision where
  total : ℚ
  y_share : ℚ
  y_rate : ℚ

/-- Calculates the share of z given a money division -/
def z_share (md : MoneyDivision) : ℚ :=
  md.total - md.y_share - (md.y_share / md.y_rate)

/-- Calculates the rate at which z receives money compared to x -/
def z_rate (md : MoneyDivision) : ℚ :=
  (z_share md) / (md.y_share / md.y_rate)

/-- Theorem stating that z gets 30 paisa for each rupee x gets -/
theorem z_gets_30_paisa (md : MoneyDivision) 
  (h1 : md.total = 105)
  (h2 : md.y_share = 27)
  (h3 : md.y_rate = 45/100) : 
  z_rate md = 30/100 := by
  sorry

end NUMINAMATH_CALUDE_z_gets_30_paisa_l2209_220959


namespace NUMINAMATH_CALUDE_french_exam_min_words_to_learn_l2209_220968

/-- The minimum number of words to learn for a 90% score on a French vocabulary exam -/
theorem french_exam_min_words_to_learn :
  ∀ (total_words : ℕ) (guess_success_rate : ℚ) (target_score : ℚ),
    total_words = 800 →
    guess_success_rate = 1/10 →
    target_score = 9/10 →
    ∃ (words_to_learn : ℕ),
      words_to_learn ≥ 712 ∧
      (words_to_learn : ℚ) / total_words +
        guess_success_rate * ((total_words : ℚ) - words_to_learn) / total_words ≥ target_score :=
by sorry

end NUMINAMATH_CALUDE_french_exam_min_words_to_learn_l2209_220968


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2209_220952

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def N : Set ℝ := {x | x ≥ 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2209_220952


namespace NUMINAMATH_CALUDE_correct_answer_calculation_l2209_220941

theorem correct_answer_calculation (x y : ℝ) : 
  (y = x + 2 * 0.42) → (x = y - 2 * 0.42) :=
by
  sorry

#eval (0.9 : ℝ) - 2 * 0.42

end NUMINAMATH_CALUDE_correct_answer_calculation_l2209_220941


namespace NUMINAMATH_CALUDE_exists_homogeneous_polynomial_for_irreducible_lattice_points_l2209_220955

-- Define an irreducible lattice point
def irreducible_lattice_point (p : ℤ × ℤ) : Prop :=
  Int.gcd p.1 p.2 = 1

-- Define a homogeneous polynomial with integer coefficients
def homogeneous_polynomial (f : ℤ → ℤ → ℤ) (d : ℕ) : Prop :=
  ∀ (c : ℤ) (x y : ℤ), f (c * x) (c * y) = c^d * f x y

-- The main theorem
theorem exists_homogeneous_polynomial_for_irreducible_lattice_points 
  (S : Finset (ℤ × ℤ)) (h : ∀ p ∈ S, irreducible_lattice_point p) :
  ∃ (f : ℤ → ℤ → ℤ) (d : ℕ), 
    d ≥ 1 ∧ 
    homogeneous_polynomial f d ∧ 
    (∀ p ∈ S, f p.1 p.2 = 1) := by
  sorry


end NUMINAMATH_CALUDE_exists_homogeneous_polynomial_for_irreducible_lattice_points_l2209_220955


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2209_220988

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  (2 * a - c) * Real.cos B = b * Real.cos C →
  a = 2 →
  c = 3 →
  B = π / 3 ∧ Real.sin C = (3 * Real.sqrt 14) / 14 := by
sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2209_220988


namespace NUMINAMATH_CALUDE_only_one_true_proposition_l2209_220974

theorem only_one_true_proposition :
  (∃! n : Fin 4, 
    (n = 0 → (∀ a b : ℝ, a > b ↔ a^2 > b^2)) ∧
    (n = 1 → (∀ a b : ℝ, a > b ↔ a^3 > b^3)) ∧
    (n = 2 → (∀ a b : ℝ, a > b → |a| > |b|)) ∧
    (n = 3 → (∀ a b c : ℝ, a * c^2 ≤ b * c^2 → a > b))) :=
by sorry

end NUMINAMATH_CALUDE_only_one_true_proposition_l2209_220974


namespace NUMINAMATH_CALUDE_opposite_of_negative_six_l2209_220909

-- Define the concept of opposite
def opposite (a : ℝ) : ℝ := -a

-- State the theorem
theorem opposite_of_negative_six :
  opposite (-6) = 6 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_six_l2209_220909


namespace NUMINAMATH_CALUDE_income_2005_between_3600_and_3800_l2209_220933

/-- Represents the income data for farmers in a certain region -/
structure FarmerIncome where
  initialYear : Nat
  initialWageIncome : ℝ
  initialOtherIncome : ℝ
  wageGrowthRate : ℝ
  otherIncomeIncrease : ℝ

/-- Calculates the average income of farmers after a given number of years -/
def averageIncomeAfterYears (data : FarmerIncome) (years : Nat) : ℝ :=
  data.initialWageIncome * (1 + data.wageGrowthRate) ^ years +
  data.initialOtherIncome + data.otherIncomeIncrease * years

/-- Theorem stating that the average income in 2005 will be between 3600 and 3800 yuan -/
theorem income_2005_between_3600_and_3800 (data : FarmerIncome) 
  (h1 : data.initialYear = 2003)
  (h2 : data.initialWageIncome = 1800)
  (h3 : data.initialOtherIncome = 1350)
  (h4 : data.wageGrowthRate = 0.06)
  (h5 : data.otherIncomeIncrease = 160) :
  3600 ≤ averageIncomeAfterYears data 2 ∧ averageIncomeAfterYears data 2 ≤ 3800 := by
  sorry

#eval averageIncomeAfterYears 
  { initialYear := 2003
    initialWageIncome := 1800
    initialOtherIncome := 1350
    wageGrowthRate := 0.06
    otherIncomeIncrease := 160 } 2

end NUMINAMATH_CALUDE_income_2005_between_3600_and_3800_l2209_220933


namespace NUMINAMATH_CALUDE_scores_with_two_ways_exist_l2209_220964

/-- Represents a scoring configuration for a test -/
structure ScoringConfig where
  total_questions : ℕ
  correct_points : ℕ
  unanswered_points : ℕ
  incorrect_points : ℕ

/-- Represents a possible answer combination -/
structure AnswerCombination where
  correct : ℕ
  unanswered : ℕ
  incorrect : ℕ

/-- Calculates the score for a given answer combination -/
def calculate_score (config : ScoringConfig) (answers : AnswerCombination) : ℕ :=
  answers.correct * config.correct_points + 
  answers.unanswered * config.unanswered_points +
  answers.incorrect * config.incorrect_points

/-- Checks if an answer combination is valid for a given configuration -/
def is_valid_combination (config : ScoringConfig) (answers : AnswerCombination) : Prop :=
  answers.correct + answers.unanswered + answers.incorrect = config.total_questions

/-- Defines the existence of scores with exactly two ways to achieve them -/
def exists_scores_with_two_ways (config : ScoringConfig) : Prop :=
  ∃ S : ℕ, 
    0 ≤ S ∧ S ≤ 175 ∧
    (∃ (a b : AnswerCombination),
      a ≠ b ∧
      is_valid_combination config a ∧
      is_valid_combination config b ∧
      calculate_score config a = S ∧
      calculate_score config b = S ∧
      ∀ c : AnswerCombination, 
        is_valid_combination config c ∧ calculate_score config c = S → (c = a ∨ c = b))

/-- The main theorem to prove -/
theorem scores_with_two_ways_exist : 
  let config : ScoringConfig := {
    total_questions := 25,
    correct_points := 7,
    unanswered_points := 3,
    incorrect_points := 0
  }
  exists_scores_with_two_ways config := by
  sorry

end NUMINAMATH_CALUDE_scores_with_two_ways_exist_l2209_220964


namespace NUMINAMATH_CALUDE_prime_sum_2003_l2209_220967

theorem prime_sum_2003 (a b : ℕ) (ha : Prime a) (hb : Prime b) (h : a^2 + b = 2003) : 
  a + b = 2001 := by sorry

end NUMINAMATH_CALUDE_prime_sum_2003_l2209_220967


namespace NUMINAMATH_CALUDE_time_to_hospital_l2209_220990

/-- Proves that given a distance of 0.09 kilometers to the hospital and a speed of 3 meters per 4 seconds, it takes 120 seconds for Ayeon to reach the hospital. -/
theorem time_to_hospital (distance_km : ℝ) (speed_m : ℝ) (speed_s : ℝ) : 
  distance_km = 0.09 →
  speed_m = 3 →
  speed_s = 4 →
  (distance_km * 1000) / (speed_m / speed_s) = 120 := by
sorry

end NUMINAMATH_CALUDE_time_to_hospital_l2209_220990


namespace NUMINAMATH_CALUDE_square_side_length_l2209_220998

theorem square_side_length (x : ℝ) : 
  x > 0 ∧ 
  x + (x + 17) + (x + 11) = 52 → 
  x = 8 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l2209_220998


namespace NUMINAMATH_CALUDE_elixir_combinations_eq_18_l2209_220923

/-- Represents the number of magical herbs available. -/
def num_herbs : ℕ := 4

/-- Represents the number of enchanted gems available. -/
def num_gems : ℕ := 6

/-- Represents the number of incompatible gem-herb pairs. -/
def num_incompatible : ℕ := 6

/-- Calculates the number of ways the sorcerer can prepare the elixir. -/
def num_elixir_combinations : ℕ := num_herbs * num_gems - num_incompatible

/-- Proves that the number of ways to prepare the elixir is 18. -/
theorem elixir_combinations_eq_18 : num_elixir_combinations = 18 := by
  sorry

end NUMINAMATH_CALUDE_elixir_combinations_eq_18_l2209_220923


namespace NUMINAMATH_CALUDE_butterflies_in_garden_l2209_220958

theorem butterflies_in_garden (total : ℕ) (flew_away_fraction : ℚ) (left : ℕ) : 
  total = 150 →
  flew_away_fraction = 11 / 13 →
  left = total - Int.floor (↑total * flew_away_fraction) →
  left = 23 := by
  sorry

end NUMINAMATH_CALUDE_butterflies_in_garden_l2209_220958


namespace NUMINAMATH_CALUDE_paper_tearing_impossibility_l2209_220997

theorem paper_tearing_impossibility : ∀ n : ℕ, 
  n % 3 = 2 → 
  ¬ (∃ (sequence : ℕ → ℕ), 
    sequence 0 = 1 ∧ 
    (∀ i : ℕ, sequence (i + 1) = sequence i + 3 ∨ sequence (i + 1) = sequence i + 9) ∧
    (∃ k : ℕ, sequence k = n)) :=
by sorry

end NUMINAMATH_CALUDE_paper_tearing_impossibility_l2209_220997


namespace NUMINAMATH_CALUDE_girls_in_school_l2209_220915

/-- Proves the number of girls in a school given stratified sampling conditions -/
theorem girls_in_school (total_students : ℕ) (sample_size : ℕ) (girls_boys_diff : ℕ) :
  total_students = 2400 →
  sample_size = 200 →
  girls_boys_diff = 10 →
  ∃ (girls_in_sample : ℕ) (girls_in_school : ℕ),
    girls_in_sample + (girls_in_sample + girls_boys_diff) = sample_size ∧
    (girls_in_sample : ℚ) / sample_size = (girls_in_school : ℚ) / total_students ∧
    girls_in_school = 1140 :=
by sorry

end NUMINAMATH_CALUDE_girls_in_school_l2209_220915


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_l2209_220944

def N : ℕ := 36 * 72 * 50 * 81

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors N : ℚ) / (sum_even_divisors N : ℚ) = 1 / 126 := by sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_l2209_220944


namespace NUMINAMATH_CALUDE_mrs_hilt_coin_value_l2209_220907

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The number of quarters Mrs. Hilt found -/
def num_quarters : ℕ := 4

/-- The number of dimes Mrs. Hilt found -/
def num_dimes : ℕ := 6

/-- The number of nickels Mrs. Hilt found -/
def num_nickels : ℕ := 8

/-- The number of pennies Mrs. Hilt found -/
def num_pennies : ℕ := 12

/-- The total value of the coins Mrs. Hilt found -/
theorem mrs_hilt_coin_value : 
  (num_quarters : ℚ) * quarter_value + 
  (num_dimes : ℚ) * dime_value + 
  (num_nickels : ℚ) * nickel_value + 
  (num_pennies : ℚ) * penny_value = 2.12 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_coin_value_l2209_220907


namespace NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l2209_220932

theorem greatest_integer_quadratic_inequality :
  ∃ (n : ℤ), n^2 - 17*n + 72 ≤ 0 ∧ n = 9 ∧ ∀ (m : ℤ), m^2 - 17*m + 72 ≤ 0 → m ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l2209_220932


namespace NUMINAMATH_CALUDE_article_cost_price_l2209_220927

theorem article_cost_price (selling_price selling_price_increased : ℝ) 
  (h1 : selling_price = 0.75 * 1250)
  (h2 : selling_price_increased = selling_price + 500)
  (h3 : selling_price_increased = 1.15 * 1250) : 1250 = 1250 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_price_l2209_220927


namespace NUMINAMATH_CALUDE_no_real_roots_l2209_220978

theorem no_real_roots : ¬∃ x : ℝ, Real.sqrt (x + 9) - Real.sqrt (x - 1) + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l2209_220978


namespace NUMINAMATH_CALUDE_train_speed_l2209_220930

/-- The speed of a train given its length and time to cross a fixed point. -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 900) (h2 : time = 12) :
  length / time = 75 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l2209_220930


namespace NUMINAMATH_CALUDE_avalon_quest_probability_l2209_220986

theorem avalon_quest_probability :
  let total_players : ℕ := 10
  let bad_players : ℕ := 4
  let quest_size : ℕ := 3
  let good_players : ℕ := total_players - bad_players
  let total_quests : ℕ := Nat.choose total_players quest_size
  let failed_quests : ℕ := total_quests - Nat.choose good_players quest_size
  let one_bad_quests : ℕ := Nat.choose bad_players 1 * Nat.choose good_players (quest_size - 1)
  (failed_quests > 0) →
  (one_bad_quests : ℚ) / failed_quests = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_avalon_quest_probability_l2209_220986


namespace NUMINAMATH_CALUDE_perpendicular_m_value_parallel_distance_l2209_220966

-- Define the lines l1 and l2
def l1 (m : ℝ) (x y : ℝ) : Prop := x + (m - 3) * y + m = 0
def l2 (m : ℝ) (x y : ℝ) : Prop := m * x - 2 * y + 4 = 0

-- Define perpendicularity condition
def perpendicular (m : ℝ) : Prop := (-1 : ℝ) / (m - 3) * (m / 2) = -1

-- Define parallelism condition
def parallel (m : ℝ) : Prop := 1 * (-2) = m * (m - 3)

-- Theorem for perpendicular case
theorem perpendicular_m_value : ∃ m : ℝ, perpendicular m ∧ m = 6 := by sorry

-- Theorem for parallel case
theorem parallel_distance : 
  ∃ m : ℝ, parallel m ∧ 
  (let d := |4 - 1| / Real.sqrt (1^2 + (-2)^2);
   d = 3 * Real.sqrt 5 / 5) := by sorry

end NUMINAMATH_CALUDE_perpendicular_m_value_parallel_distance_l2209_220966


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_l2209_220938

theorem ellipse_parabola_intersection (n m : ℝ) :
  (∀ x y : ℝ, x^2/n + y^2/9 = 1 ∧ y = x^2 - m → 
    (3/n < m ∧ m < (4*m^2 + 9)/(4*m) ∧ m > 3/2)) ∧
  (m = 4 ∧ n = 4 → 
    ∀ x y : ℝ, x^2/n + y^2/9 = 1 ∧ y = x^2 - m → 
      4*x^2 + 4*y^2 - 5*y - 16 = 0) := by
sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_l2209_220938


namespace NUMINAMATH_CALUDE_tan_45_degrees_l2209_220918

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l2209_220918


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2209_220919

theorem absolute_value_inequality (a : ℝ) : 
  (∀ x : ℝ, |2*x - 3| - 2*a > |x + a|) ↔ -3/2 ≤ a ∧ a < -1/2 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2209_220919


namespace NUMINAMATH_CALUDE_harper_consumption_l2209_220989

/-- Represents the mineral water consumption problem -/
structure MineralWaterConsumption where
  bottles_per_case : ℕ
  cost_per_case : ℚ
  total_spent : ℚ
  days_supply : ℕ

/-- Calculates the daily mineral water consumption given the problem parameters -/
def daily_consumption (m : MineralWaterConsumption) : ℚ :=
  (m.total_spent / m.cost_per_case * m.bottles_per_case) / m.days_supply

/-- Theorem stating that Harper's daily mineral water consumption is 0.5 bottles -/
theorem harper_consumption :
  ∃ (m : MineralWaterConsumption),
    m.bottles_per_case = 24 ∧
    m.cost_per_case = 12 ∧
    m.total_spent = 60 ∧
    m.days_supply = 240 ∧
    daily_consumption m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_harper_consumption_l2209_220989


namespace NUMINAMATH_CALUDE_max_profit_min_sales_for_profit_l2209_220910

-- Define the cost per unit
def cost : ℝ := 20

-- Define the relationship between price and sales volume
def sales_volume (x : ℝ) : ℝ := -10 * x + 500

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - cost) * sales_volume x

-- Define the price constraints
def price_constraint (x : ℝ) : Prop := 25 ≤ x ∧ x ≤ 38

-- Theorem 1: Maximum profit occurs at x = 35 and is equal to 2250
theorem max_profit :
  ∃ (x : ℝ), price_constraint x ∧
  profit x = 2250 ∧
  ∀ (y : ℝ), price_constraint y → profit y ≤ profit x :=
sorry

-- Theorem 2: At price 38, selling 120 units yields a profit of at least 2000
theorem min_sales_for_profit :
  sales_volume 38 ≥ 120 ∧ profit 38 ≥ 2000 :=
sorry

end NUMINAMATH_CALUDE_max_profit_min_sales_for_profit_l2209_220910


namespace NUMINAMATH_CALUDE_goldfish_remaining_l2209_220913

/-- Given Finn's initial number of goldfish and the number that die,
    prove the number of goldfish left. -/
theorem goldfish_remaining (initial : ℕ) (died : ℕ) :
  initial ≥ died →
  initial - died = initial - died :=
by sorry

end NUMINAMATH_CALUDE_goldfish_remaining_l2209_220913


namespace NUMINAMATH_CALUDE_swimmer_speed_ratio_l2209_220983

theorem swimmer_speed_ratio :
  ∀ (v₁ v₂ : ℝ),
    v₁ > v₂ →
    v₁ > 0 →
    v₂ > 0 →
    (v₁ + v₂) * 3 = 12 →
    (v₁ - v₂) * 6 = 12 →
    v₁ / v₂ = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_swimmer_speed_ratio_l2209_220983


namespace NUMINAMATH_CALUDE_max_product_for_2017_l2209_220908

/-- The maximum product of positive integers whose sum is 2017 -/
def max_product : ℕ := 4 * 3^671

/-- The sum of the positive integers used to achieve the maximum product -/
def sum_of_factors : ℕ := 2017

theorem max_product_for_2017 :
  ∀ (factors : List ℕ),
  (factors.sum = sum_of_factors) →
  (factors.prod ≤ max_product) :=
sorry

end NUMINAMATH_CALUDE_max_product_for_2017_l2209_220908


namespace NUMINAMATH_CALUDE_min_distance_circle_to_line_l2209_220948

/-- The minimum distance from a point on the circle x^2 + y^2 = 1 to the line 3x + 4y - 25 = 0 is 4 -/
theorem min_distance_circle_to_line :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let line := {(x, y) : ℝ × ℝ | 3*x + 4*y - 25 = 0}
  ∃ (d : ℝ), d = 4 ∧ ∀ (p : ℝ × ℝ), p ∈ circle →
    ∀ (q : ℝ × ℝ), q ∈ line →
      d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_circle_to_line_l2209_220948


namespace NUMINAMATH_CALUDE_negative_slope_implies_negative_correlation_l2209_220980

/-- Represents a linear regression equation -/
structure LinearRegression where
  a : ℝ
  b : ℝ

/-- The correlation coefficient between two variables -/
def correlation_coefficient (x y : ℝ → ℝ) : ℝ := sorry

/-- Theorem: Given a linear regression with negative slope, 
    the correlation coefficient is between -1 and 0 -/
theorem negative_slope_implies_negative_correlation 
  (reg : LinearRegression) 
  (x y : ℝ → ℝ) 
  (h_reg : ∀ t, y t = reg.a + reg.b * x t) 
  (h_neg : reg.b < 0) : 
  -1 < correlation_coefficient x y ∧ correlation_coefficient x y < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_slope_implies_negative_correlation_l2209_220980


namespace NUMINAMATH_CALUDE_f_composition_equals_sqrt2_over_2_minus_1_l2209_220994

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x - 1 else Real.log x / Real.log 3 + 1

theorem f_composition_equals_sqrt2_over_2_minus_1 :
  f (f (Real.sqrt 3 / 9)) = Real.sqrt 2 / 2 - 1 := by sorry

end NUMINAMATH_CALUDE_f_composition_equals_sqrt2_over_2_minus_1_l2209_220994


namespace NUMINAMATH_CALUDE_min_visible_pairs_l2209_220901

/-- Represents the number of birds on the circle -/
def num_birds : ℕ := 155

/-- Represents the maximum arc length for mutual visibility in degrees -/
def visibility_arc : ℝ := 10

/-- Calculates the number of pairs in a group of n birds -/
def pairs_in_group (n : ℕ) : ℕ := n.choose 2

/-- Represents the optimal grouping of birds -/
def optimal_grouping : List ℕ := [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

theorem min_visible_pairs :
  (List.sum (List.map pairs_in_group optimal_grouping) = 270) ∧
  (List.sum optimal_grouping = num_birds) ∧
  (List.length optimal_grouping * visibility_arc ≥ 360) ∧
  (∀ (grouping : List ℕ), 
    (List.sum grouping = num_birds) →
    (List.length grouping * visibility_arc ≥ 360) →
    (List.sum (List.map pairs_in_group grouping) ≥ 270)) := by
  sorry

end NUMINAMATH_CALUDE_min_visible_pairs_l2209_220901


namespace NUMINAMATH_CALUDE_a_minus_b_equals_two_l2209_220924

-- Define the functions f, g, h, and h_inv
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -4 * x + 3
def h (a b : ℝ) (x : ℝ) : ℝ := f a b (g x)
def h_inv (x : ℝ) : ℝ := x + 3

-- State the theorem
theorem a_minus_b_equals_two (a b : ℝ) : 
  (∀ x, h a b x = x - 3) → 
  (∀ x, h a b (h_inv x) = x) → 
  a - b = 2 := by
  sorry


end NUMINAMATH_CALUDE_a_minus_b_equals_two_l2209_220924


namespace NUMINAMATH_CALUDE_function_forms_correctness_l2209_220969

-- Define a linear function
def linear_function (a b x : ℝ) : ℝ := a * x + b

-- Define a special case of linear function
def linear_function_special (a x : ℝ) : ℝ := a * x

-- Define a quadratic function
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define special cases of quadratic function
def quadratic_function_special1 (a c x : ℝ) : ℝ := a * x^2 + c
def quadratic_function_special2 (a x : ℝ) : ℝ := a * x^2

-- Theorem stating the correctness of these function definitions
theorem function_forms_correctness (a b c x : ℝ) (h : a ≠ 0) :
  (∃ y, y = linear_function a b x) ∧
  (∃ y, y = linear_function_special a x) ∧
  (∃ y, y = quadratic_function a b c x) ∧
  (∃ y, y = quadratic_function_special1 a c x) ∧
  (∃ y, y = quadratic_function_special2 a x) :=
sorry

end NUMINAMATH_CALUDE_function_forms_correctness_l2209_220969


namespace NUMINAMATH_CALUDE_correct_system_of_equations_l2209_220981

/-- Represents the price of a basketball in yuan -/
def basketball_price : ℝ := sorry

/-- Represents the price of a soccer ball in yuan -/
def soccer_ball_price : ℝ := sorry

/-- The total cost of the purchase in yuan -/
def total_cost : ℝ := 445

/-- The number of basketballs purchased -/
def num_basketballs : ℕ := 3

/-- The number of soccer balls purchased -/
def num_soccer_balls : ℕ := 7

/-- The price difference between a basketball and a soccer ball in yuan -/
def price_difference : ℝ := 5

/-- Theorem stating that the system of equations correctly represents the given conditions -/
theorem correct_system_of_equations : 
  (num_basketballs * basketball_price + num_soccer_balls * soccer_ball_price = total_cost) ∧ 
  (basketball_price = soccer_ball_price + price_difference) := by
  sorry

end NUMINAMATH_CALUDE_correct_system_of_equations_l2209_220981


namespace NUMINAMATH_CALUDE_grace_age_l2209_220920

-- Define the ages as natural numbers
def Harriet : ℕ := 18
def Ian : ℕ := Harriet + 5
def Jack : ℕ := Ian - 7
def Grace : ℕ := 2 * Jack

-- Theorem statement
theorem grace_age : Grace = 32 := by
  sorry

end NUMINAMATH_CALUDE_grace_age_l2209_220920


namespace NUMINAMATH_CALUDE_element_in_set_l2209_220999

theorem element_in_set : 
  let M : Set ℕ := {0, 1, 2}
  let a : ℕ := 0
  a ∈ M :=
by sorry

end NUMINAMATH_CALUDE_element_in_set_l2209_220999
