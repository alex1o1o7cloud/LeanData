import Mathlib

namespace NUMINAMATH_CALUDE_speed_difference_l2430_243063

/-- Given a truck and a car traveling the same distance, prove the difference in their average speeds -/
theorem speed_difference (distance : ℝ) (truck_time car_time : ℝ) 
  (h1 : distance = 240)
  (h2 : truck_time = 8)
  (h3 : car_time = 5) : 
  (distance / car_time) - (distance / truck_time) = 18 := by
  sorry

end NUMINAMATH_CALUDE_speed_difference_l2430_243063


namespace NUMINAMATH_CALUDE_inequality_and_system_solution_l2430_243006

theorem inequality_and_system_solution :
  (∀ x : ℝ, (2*x - 3)/3 > (3*x + 1)/6 - 1 ↔ x > 1) ∧
  (∀ x : ℝ, x ≤ 3*x - 6 ∧ 3*x + 1 > 2*(x - 1) ↔ x ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_system_solution_l2430_243006


namespace NUMINAMATH_CALUDE_wall_bricks_count_l2430_243003

theorem wall_bricks_count :
  ∀ (x : ℕ),
  (∃ (rate1 rate2 : ℚ),
    rate1 = x / 9 ∧
    rate2 = x / 10 ∧
    5 * (rate1 + rate2 - 10) = x) →
  x = 900 := by
sorry

end NUMINAMATH_CALUDE_wall_bricks_count_l2430_243003


namespace NUMINAMATH_CALUDE_inequality_order_l2430_243025

theorem inequality_order (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * a * b) / (a + b) ≤ Real.sqrt (a * b) ∧ Real.sqrt (a * b) ≤ (a + b) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_order_l2430_243025


namespace NUMINAMATH_CALUDE_zero_point_in_interval_l2430_243058

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

theorem zero_point_in_interval :
  ∃ x : ℝ, 2 < x ∧ x < 3 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_zero_point_in_interval_l2430_243058


namespace NUMINAMATH_CALUDE_matrix_sum_equality_l2430_243060

def matrix1 : Matrix (Fin 3) (Fin 3) ℤ := !![2, -1, 3; 0, 4, -2; 5, -3, 1]
def matrix2 : Matrix (Fin 3) (Fin 3) ℤ := !![-3, 2, -4; 1, -6, 3; -2, 4, 0]
def result : Matrix (Fin 3) (Fin 3) ℤ := !![-1, 1, -1; 1, -2, 1; 3, 1, 1]

theorem matrix_sum_equality : matrix1 + matrix2 = result := by
  sorry

end NUMINAMATH_CALUDE_matrix_sum_equality_l2430_243060


namespace NUMINAMATH_CALUDE_afternoon_more_than_morning_l2430_243093

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 6

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 8

/-- The difference in emails between afternoon and morning -/
def email_difference : ℕ := afternoon_emails - morning_emails

theorem afternoon_more_than_morning : email_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_more_than_morning_l2430_243093


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_value_l2430_243020

/-- A hyperbola with equation -y²/a² + x²/b² = 1 and eccentricity e -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  e : ℝ

/-- A parabola with equation y² = 16x -/
structure Parabola where

/-- The right focus of a hyperbola coincides with the focus of a parabola y² = 16x -/
def right_focus_coincides (h : Hyperbola) (p : Parabola) : Prop :=
  h.e * h.b = 4

theorem hyperbola_eccentricity_value (h : Hyperbola) (p : Parabola) 
  (h_eq : -h.a^2 + h.b^2 = h.a^2 * h.b^2) 
  (coincide : right_focus_coincides h p) : 
  h.e = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_value_l2430_243020


namespace NUMINAMATH_CALUDE_beijing_shanghai_train_time_l2430_243089

/-- The function relationship between total travel time and average speed for a train on the Beijing-Shanghai railway line -/
theorem beijing_shanghai_train_time (t : ℝ) (v : ℝ) (h : v ≠ 0) : 
  (t = 1463 / v) ↔ (1463 = t * v) :=
by sorry

end NUMINAMATH_CALUDE_beijing_shanghai_train_time_l2430_243089


namespace NUMINAMATH_CALUDE_two_thirds_of_number_l2430_243040

theorem two_thirds_of_number (y : ℝ) : (2 / 3) * y = 40 → y = 60 := by
  sorry

end NUMINAMATH_CALUDE_two_thirds_of_number_l2430_243040


namespace NUMINAMATH_CALUDE_negative_one_power_difference_l2430_243057

theorem negative_one_power_difference : (-1 : ℤ)^5 - (-1 : ℤ)^4 = -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_power_difference_l2430_243057


namespace NUMINAMATH_CALUDE_unique_integer_solution_l2430_243078

theorem unique_integer_solution :
  ∃! (a b c : ℤ), a^2 + b^2 + c^2 + 3 < a*b + 3*b + 2*c ∧ a = 1 ∧ b = 2 ∧ c = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l2430_243078


namespace NUMINAMATH_CALUDE_andrew_payment_l2430_243062

/-- The total amount Andrew paid for grapes and mangoes -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Andrew paid 1376 for his purchase -/
theorem andrew_payment :
  total_amount 14 54 10 62 = 1376 := by
  sorry

end NUMINAMATH_CALUDE_andrew_payment_l2430_243062


namespace NUMINAMATH_CALUDE_animal_books_count_animal_books_proof_l2430_243075

def book_price : ℕ := 16
def space_books : ℕ := 1
def train_books : ℕ := 3
def total_spent : ℕ := 224

theorem animal_books_count : ℕ :=
  (total_spent - book_price * (space_books + train_books)) / book_price

#check animal_books_count

theorem animal_books_proof :
  animal_books_count = 10 :=
by sorry

end NUMINAMATH_CALUDE_animal_books_count_animal_books_proof_l2430_243075


namespace NUMINAMATH_CALUDE_stratified_sampling_sample_size_l2430_243004

theorem stratified_sampling_sample_size 
  (total_employees : ℕ) 
  (young_workers : ℕ) 
  (sample_young : ℕ) 
  (h1 : total_employees = 750) 
  (h2 : young_workers = 350) 
  (h3 : sample_young = 7) : 
  ∃ (sample_size : ℕ), 
    sample_size * young_workers = sample_young * total_employees ∧ 
    sample_size = 15 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_sample_size_l2430_243004


namespace NUMINAMATH_CALUDE_siblings_age_problem_l2430_243047

theorem siblings_age_problem (b s : ℕ) : 
  (b - 3 = 7 * (s - 3)) →
  (b - 2 = 4 * (s - 2)) →
  (b - 1 = 3 * (s - 1)) →
  (b = (5 * s) / 2) →
  (b = 10 ∧ s = 4) :=
by sorry

end NUMINAMATH_CALUDE_siblings_age_problem_l2430_243047


namespace NUMINAMATH_CALUDE_octagon_diagonals_l2430_243073

/-- The number of internal diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: An octagon has 20 internal diagonals -/
theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l2430_243073


namespace NUMINAMATH_CALUDE_perpendicular_planes_l2430_243066

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Define the theorem
theorem perpendicular_planes 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) 
  (h_m_perp_α : perpendicular m α) 
  (h_m_para_β : parallel m β) : 
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_l2430_243066


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2430_243061

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℝ),
  (X^4 : Polynomial ℝ) + X^3 + 1 = (X^2 - 2*X + 3) * q + r ∧
  r.degree < (X^2 - 2*X + 3).degree ∧
  r = -3*X - 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2430_243061


namespace NUMINAMATH_CALUDE_exactly_one_sunny_day_probability_l2430_243032

theorem exactly_one_sunny_day_probability :
  let num_days : ℕ := 4
  let rain_probability : ℚ := 3/4
  let sunny_probability : ℚ := 1 - rain_probability
  let choose_one_day : ℕ := num_days.choose 1
  let prob_three_rain_one_sunny : ℚ := rain_probability^3 * sunny_probability^1
  choose_one_day * prob_three_rain_one_sunny = 27/64 :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_sunny_day_probability_l2430_243032


namespace NUMINAMATH_CALUDE_equation_solution_l2430_243051

theorem equation_solution : ∃ x : ℝ, 45 - (28 - (37 - (15 - x))) = 54 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2430_243051


namespace NUMINAMATH_CALUDE_labyrinth_paths_count_l2430_243011

/-- Represents a point in the labyrinth -/
structure Point where
  x : Nat
  y : Nat

/-- Represents a direction of movement in the labyrinth -/
inductive Direction
  | Right
  | Down
  | Up

/-- Represents the labyrinth structure -/
structure Labyrinth where
  entrance : Point
  exit : Point
  branchPoints : List Point
  isValidMove : Point → Direction → Bool

/-- Counts the number of distinct paths from entrance to exit -/
def countPaths (lab : Labyrinth) : Nat :=
  sorry

/-- The specific labyrinth structure from the problem -/
def problemLabyrinth : Labyrinth :=
  { entrance := { x := 0, y := 1 }
  , exit := { x := 4, y := 1 }
  , branchPoints := [{ x := 1, y := 1 }, { x := 1, y := 2 }, { x := 2, y := 1 }, { x := 2, y := 2 }, { x := 3, y := 1 }, { x := 3, y := 2 }]
  , isValidMove := sorry }

theorem labyrinth_paths_count :
  countPaths problemLabyrinth = 16 :=
by sorry

end NUMINAMATH_CALUDE_labyrinth_paths_count_l2430_243011


namespace NUMINAMATH_CALUDE_y₁_less_than_y₂_l2430_243046

/-- A linear function f(x) = -x + 5 -/
def f (x : ℝ) : ℝ := -x + 5

/-- P₁ is a point on the graph of f with x-coordinate -2 -/
def P₁ (y₁ : ℝ) : Prop := f (-2) = y₁

/-- P₂ is a point on the graph of f with x-coordinate -3 -/
def P₂ (y₂ : ℝ) : Prop := f (-3) = y₂

/-- Theorem: If P₁(-2, y₁) and P₂(-3, y₂) are points on the graph of f, then y₁ < y₂ -/
theorem y₁_less_than_y₂ (y₁ y₂ : ℝ) (h₁ : P₁ y₁) (h₂ : P₂ y₂) : y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y₁_less_than_y₂_l2430_243046


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l2430_243083

def f (x : ℝ) : ℝ := 4 * x^5 + 13 * x^4 - 30 * x^3 + 8 * x^2

theorem roots_of_polynomial :
  (∀ x : ℝ, f x = 0 ↔ x = 0 ∨ x = (1 : ℝ) / 2 ∨ x = -2 + 2 * Real.sqrt 2 ∨ x = -2 - 2 * Real.sqrt 2) ∧
  (∃ ε > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < ε → f x / x^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l2430_243083


namespace NUMINAMATH_CALUDE_initial_men_count_l2430_243029

/-- The number of days it takes for the initial group to complete the work -/
def initial_days : ℕ := 70

/-- The number of days it takes for 40 men to complete the work -/
def new_days : ℕ := 63

/-- The number of men in the new group -/
def new_men : ℕ := 40

/-- The amount of work is constant and can be represented as men * days -/
axiom work_constant (m1 m2 : ℕ) (d1 d2 : ℕ) : m1 * d1 = m2 * d2

/-- The theorem to be proved -/
theorem initial_men_count : ∃ x : ℕ, x * initial_days = new_men * new_days ∧ x = 36 := by
  sorry

end NUMINAMATH_CALUDE_initial_men_count_l2430_243029


namespace NUMINAMATH_CALUDE_f_geq_f1_iff_a_in_range_l2430_243000

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 1 then (1/3) * x^3 - a * x + 1
  else if x ≥ 1 then a * Real.log x
  else 0  -- This case should never occur in our problem, but Lean requires it for completeness

-- State the theorem
theorem f_geq_f1_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f a x ≥ f a 1) ↔ (0 < a ∧ a ≤ 4/3) :=
by sorry

end NUMINAMATH_CALUDE_f_geq_f1_iff_a_in_range_l2430_243000


namespace NUMINAMATH_CALUDE_apple_ratio_l2430_243052

def total_apples : ℕ := 496
def green_apples : ℕ := 124

theorem apple_ratio : 
  let red_apples := total_apples - green_apples
  (red_apples : ℚ) / green_apples = 93 / 31 := by
sorry

end NUMINAMATH_CALUDE_apple_ratio_l2430_243052


namespace NUMINAMATH_CALUDE_inequality_range_l2430_243080

theorem inequality_range (k : ℝ) : 
  (∀ x : ℝ, x^4 + (k-1)*x^2 + 1 ≥ 0) ↔ k ≥ 1 := by sorry

end NUMINAMATH_CALUDE_inequality_range_l2430_243080


namespace NUMINAMATH_CALUDE_next_divisible_by_sum_of_digits_l2430_243043

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Check if a number is divisible by the sum of its digits -/
def isDivisibleBySumOfDigits (n : ℕ) : Prop :=
  n % sumOfDigits n = 0

/-- The next number after 1232 that is divisible by the sum of its digits -/
theorem next_divisible_by_sum_of_digits :
  ∃ (n : ℕ), n > 1232 ∧
    isDivisibleBySumOfDigits n ∧
    ∀ (m : ℕ), 1232 < m ∧ m < n → ¬isDivisibleBySumOfDigits m :=
by sorry

end NUMINAMATH_CALUDE_next_divisible_by_sum_of_digits_l2430_243043


namespace NUMINAMATH_CALUDE_league_games_l2430_243067

theorem league_games (n : ℕ) (h : n = 12) : (n * (n - 1)) / 2 = 66 := by
  sorry

end NUMINAMATH_CALUDE_league_games_l2430_243067


namespace NUMINAMATH_CALUDE_intersection_M_N_l2430_243013

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x : ℝ | x^2 = 2*x}

theorem intersection_M_N : M ∩ N = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2430_243013


namespace NUMINAMATH_CALUDE_amp_specific_value_l2430_243070

/-- The operation & defined for real numbers -/
def amp (a b c d : ℝ) : ℝ := b^2 - 4*a*c + d

/-- Theorem stating that &(2, -3, 1, 5) = 6 -/
theorem amp_specific_value : amp 2 (-3) 1 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_amp_specific_value_l2430_243070


namespace NUMINAMATH_CALUDE_intersection_points_form_hyperbola_l2430_243076

/-- Given real t, the point (x, y) satisfies both equations -/
def satisfies_equations (x y t : ℝ) : Prop :=
  2 * t * x - 3 * y - 4 * t = 0 ∧ 2 * x - 3 * t * y + 4 = 0

/-- The locus of points (x, y) satisfying the equations for all t forms a hyperbola -/
theorem intersection_points_form_hyperbola :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
  ∀ x y : ℝ, (∃ t : ℝ, satisfies_equations x y t) →
  x^2 / a^2 - y^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_points_form_hyperbola_l2430_243076


namespace NUMINAMATH_CALUDE_rational_sum_zero_l2430_243022

theorem rational_sum_zero (x₁ x₂ x₃ x₄ : ℚ) 
  (h₁ : x₁ = x₂ + x₃ + x₄)
  (h₂ : x₂ = x₁ + x₃ + x₄)
  (h₃ : x₃ = x₁ + x₂ + x₄)
  (h₄ : x₄ = x₁ + x₂ + x₃) :
  x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_sum_zero_l2430_243022


namespace NUMINAMATH_CALUDE_julio_mocktail_lime_juice_l2430_243064

/-- Proves that Julio uses 1 tablespoon of lime juice per mocktail -/
theorem julio_mocktail_lime_juice :
  -- Define the problem parameters
  let days : ℕ := 30
  let mocktails_per_day : ℕ := 1
  let lime_juice_per_lime : ℚ := 2
  let limes_per_dollar : ℚ := 3
  let total_spent : ℚ := 5

  -- Calculate the total number of limes bought
  let total_limes : ℚ := total_spent * limes_per_dollar

  -- Calculate the total amount of lime juice
  let total_lime_juice : ℚ := total_limes * lime_juice_per_lime

  -- Calculate the amount of lime juice per mocktail
  let lime_juice_per_mocktail : ℚ := total_lime_juice / (days * mocktails_per_day)

  -- Prove that the amount of lime juice per mocktail is 1 tablespoon
  lime_juice_per_mocktail = 1 := by sorry

end NUMINAMATH_CALUDE_julio_mocktail_lime_juice_l2430_243064


namespace NUMINAMATH_CALUDE_serenas_age_problem_l2430_243037

/-- Proves that in 6 years, Serena's mother will be three times as old as Serena. -/
theorem serenas_age_problem (serena_age : ℕ) (mother_age : ℕ) 
  (h1 : serena_age = 9) (h2 : mother_age = 39) : 
  ∃ (years : ℕ), years = 6 ∧ mother_age + years = 3 * (serena_age + years) := by
  sorry

end NUMINAMATH_CALUDE_serenas_age_problem_l2430_243037


namespace NUMINAMATH_CALUDE_yellow_yellow_pairs_count_l2430_243088

/-- Represents the student pairing scenario in a math contest --/
structure ContestPairing where
  total_students : ℕ
  blue_students : ℕ
  yellow_students : ℕ
  total_pairs : ℕ
  blue_blue_pairs : ℕ

/-- The specific contest pairing scenario from the problem --/
def mathContest : ContestPairing := {
  total_students := 144
  blue_students := 63
  yellow_students := 81
  total_pairs := 72
  blue_blue_pairs := 27
}

/-- Theorem stating that the number of yellow-yellow pairs is 36 --/
theorem yellow_yellow_pairs_count (contest : ContestPairing) 
  (h1 : contest.total_students = contest.blue_students + contest.yellow_students)
  (h2 : contest.total_pairs * 2 = contest.total_students)
  (h3 : contest = mathContest) : 
  contest.yellow_students - (contest.total_pairs - contest.blue_blue_pairs - 
  (contest.blue_students - 2 * contest.blue_blue_pairs)) = 36 := by
  sorry

#check yellow_yellow_pairs_count

end NUMINAMATH_CALUDE_yellow_yellow_pairs_count_l2430_243088


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l2430_243068

theorem max_sum_of_factors (heart club : ℕ) : 
  heart * club = 42 → (∀ x y : ℕ, x * y = 42 → x + y ≤ heart + club) → heart + club = 43 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l2430_243068


namespace NUMINAMATH_CALUDE_same_gender_probability_l2430_243097

def num_male : ℕ := 3
def num_female : ℕ := 2
def total_volunteers : ℕ := num_male + num_female
def num_to_select : ℕ := 2

-- Function to calculate combinations
def combination (n k : ℕ) : ℕ := 
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def total_ways : ℕ := combination total_volunteers num_to_select
def same_gender_ways : ℕ := combination num_male num_to_select + combination num_female num_to_select

theorem same_gender_probability : 
  (same_gender_ways : ℚ) / total_ways = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_same_gender_probability_l2430_243097


namespace NUMINAMATH_CALUDE_smallest_other_integer_l2430_243098

theorem smallest_other_integer (a b x : ℕ+) : 
  (a = 70 ∨ b = 70) →
  (Nat.gcd a b = x + 7) →
  (Nat.lcm a b = x * (x + 7)) →
  (min a b ≠ 70 → min a b ≥ 20) :=
by sorry

end NUMINAMATH_CALUDE_smallest_other_integer_l2430_243098


namespace NUMINAMATH_CALUDE_cost_for_100km_l2430_243001

/-- Represents the cost of a taxi ride in dollars -/
def taxi_cost (distance : ℝ) : ℝ := sorry

/-- The taxi fare is directly proportional to distance traveled -/
axiom fare_proportional (d1 d2 : ℝ) : d1 ≠ 0 → d2 ≠ 0 → 
  taxi_cost d1 / d1 = taxi_cost d2 / d2

/-- Bob's actual ride: 80 km for $160 -/
axiom bob_ride : taxi_cost 80 = 160

/-- The theorem to prove -/
theorem cost_for_100km : taxi_cost 100 = 200 := by sorry

end NUMINAMATH_CALUDE_cost_for_100km_l2430_243001


namespace NUMINAMATH_CALUDE_probability_male_saturday_female_sunday_l2430_243031

/-- The number of male students -/
def num_male : ℕ := 2

/-- The number of female students -/
def num_female : ℕ := 2

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

/-- The number of days in the event -/
def num_days : ℕ := 2

/-- The probability of selecting a male student for Saturday and a female student for Sunday -/
theorem probability_male_saturday_female_sunday :
  (num_male * num_female) / (total_students * (total_students - 1)) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_male_saturday_female_sunday_l2430_243031


namespace NUMINAMATH_CALUDE_additional_sheep_problem_l2430_243079

theorem additional_sheep_problem (mary_sheep : ℕ) (bob_additional : ℕ) :
  mary_sheep = 300 →
  (mary_sheep + 266 = 2 * mary_sheep + bob_additional - 69) →
  bob_additional = 35 := by
  sorry

end NUMINAMATH_CALUDE_additional_sheep_problem_l2430_243079


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_M_l2430_243084

def M : ℕ := 36 * 36 * 98 * 150

/-- Sum of odd divisors of a natural number -/
def sum_odd_divisors (n : ℕ) : ℕ := sorry

/-- Sum of even divisors of a natural number -/
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors_M :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 62 := by sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_M_l2430_243084


namespace NUMINAMATH_CALUDE_polynomial_evaluation_and_subtraction_l2430_243071

theorem polynomial_evaluation_and_subtraction :
  let x : ℝ := 2
  20 - 2 * (3 * x^2 - 4 * x + 8) = -4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_and_subtraction_l2430_243071


namespace NUMINAMATH_CALUDE_inequality_proof_l2430_243026

theorem inequality_proof (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) 
  (sum_condition : a + b + c = 2) : 
  (1 / (1 + a * b) + 1 / (1 + b * c) + 1 / (1 + c * a) ≥ 27 / 13) ∧ 
  ((1 / (1 + a * b) + 1 / (1 + b * c) + 1 / (1 + c * a) = 27 / 13) ↔ 
   (a = 2/3 ∧ b = 2/3 ∧ c = 2/3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2430_243026


namespace NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l2430_243086

theorem sum_of_squares_lower_bound (a b c : ℝ) (h : a + b + c = 1) :
  a^2 + b^2 + c^2 ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l2430_243086


namespace NUMINAMATH_CALUDE_equation_solution_l2430_243081

theorem equation_solution : 
  ∀ x : ℝ, (Real.sqrt (x + 16) - 8 / Real.sqrt (x + 16) = 4) ↔ 
  (x = 20 + 8 * Real.sqrt 3 ∨ x = 20 - 8 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2430_243081


namespace NUMINAMATH_CALUDE_certain_number_value_l2430_243014

/-- Given two sets of numbers with known means, prove the value of an unknown number in the first set. -/
theorem certain_number_value (x y : ℝ) : 
  (28 + x + y + 78 + 104) / 5 = 62 →
  (48 + 62 + 98 + 124 + x) / 5 = 78 →
  y = 42 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l2430_243014


namespace NUMINAMATH_CALUDE_sunset_time_calculation_l2430_243015

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  valid : hours < 24 ∧ minutes < 60

/-- Represents duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat

/-- Adds a duration to a time -/
def addDuration (t : Time) (d : Duration) : Time :=
  sorry

theorem sunset_time_calculation (sunrise : Time) (daylight : Duration) : 
  sunrise.hours = 7 → 
  sunrise.minutes = 3 → 
  daylight.hours = 12 → 
  daylight.minutes = 36 → 
  (addDuration sunrise daylight).hours = 19 ∧ 
  (addDuration sunrise daylight).minutes = 39 := by
  sorry

#check sunset_time_calculation

end NUMINAMATH_CALUDE_sunset_time_calculation_l2430_243015


namespace NUMINAMATH_CALUDE_square_sum_problem_l2430_243042

theorem square_sum_problem (a b : ℝ) (h1 : a + b = -9) (h2 : a = 30 / b) : a^2 + b^2 = 61 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_problem_l2430_243042


namespace NUMINAMATH_CALUDE_quadratic_equation_implication_l2430_243050

theorem quadratic_equation_implication (x : ℝ) : 2 * x^2 + 1 = 17 → 4 * x^2 + 1 = 33 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_implication_l2430_243050


namespace NUMINAMATH_CALUDE_ab_geq_4_and_a_plus_b_geq_4_relationship_l2430_243054

theorem ab_geq_4_and_a_plus_b_geq_4_relationship (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ a b, a > 0 → b > 0 → a * b ≥ 4 → a + b ≥ 4) ∧
  (∃ a b, a > 0 ∧ b > 0 ∧ a + b ≥ 4 ∧ a * b < 4) := by
  sorry

end NUMINAMATH_CALUDE_ab_geq_4_and_a_plus_b_geq_4_relationship_l2430_243054


namespace NUMINAMATH_CALUDE_sequence_term_formula_l2430_243012

/-- The nth term of the sequence defined by the sum of consecutive integers from 1 to n -/
def sequence_term (n : ℕ) : ℕ := (List.range n).sum + n

/-- Theorem stating that the nth term of the sequence is equal to n(n+1)/2 -/
theorem sequence_term_formula (n : ℕ) : sequence_term n = n * (n + 1) / 2 := by
  sorry

#check sequence_term_formula

end NUMINAMATH_CALUDE_sequence_term_formula_l2430_243012


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l2430_243055

theorem cubic_equation_roots (p q : ℝ) :
  let x₁ := (-p + Real.sqrt (p^2 - 4*q)) / 2
  let x₂ := (-p - Real.sqrt (p^2 - 4*q)) / 2
  let cubic := fun y : ℝ => y^3 - (p^2 - q)*y^2 + (p^2*q - q^2)*y - q^3
  (cubic x₁^2 = 0) ∧ (cubic (x₁*x₂) = 0) ∧ (cubic x₂^2 = 0) :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l2430_243055


namespace NUMINAMATH_CALUDE_factorization_problems_l2430_243045

variable (x y : ℝ)

theorem factorization_problems :
  (x^2 + 3*x = x*(x + 3)) ∧ (x^2 - 2*x*y + y^2 = (x - y)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l2430_243045


namespace NUMINAMATH_CALUDE_regression_line_properties_l2430_243096

-- Define random variables x and y
variable (x y : ℝ → ℝ)

-- Define that x and y are correlated
variable (h_correlated : Correlated x y)

-- Define the mean of x and y
def x_mean : ℝ := sorry
def y_mean : ℝ := sorry

-- Define the regression line
def regression_line (x y : ℝ → ℝ) : ℝ → ℝ := sorry

-- Define the slope and intercept of the regression line
def a : ℝ := 0.2
def b : ℝ := 12

theorem regression_line_properties :
  (∀ t : ℝ, regression_line x y t = a * t + b) →
  (regression_line x y x_mean = y_mean) ∧
  (∀ δ : ℝ, regression_line x y (x_mean + δ) - regression_line x y x_mean = a * δ) :=
sorry

end NUMINAMATH_CALUDE_regression_line_properties_l2430_243096


namespace NUMINAMATH_CALUDE_buses_passed_count_l2430_243087

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Represents a bus schedule -/
structure BusSchedule where
  startTime : Time
  interval : Nat

/-- Calculates the number of buses passed during a journey -/
def busesPassed (departureTime : Time) (journeyDuration : Nat) (cityASchedule : BusSchedule) (cityBSchedule : BusSchedule) : Nat :=
  sorry

theorem buses_passed_count :
  let cityASchedule : BusSchedule := ⟨⟨6, 0, by sorry, by sorry⟩, 2⟩
  let cityBSchedule : BusSchedule := ⟨⟨6, 30, by sorry, by sorry⟩, 1⟩
  let departureTime : Time := ⟨14, 30, by sorry, by sorry⟩
  let journeyDuration : Nat := 8
  busesPassed departureTime journeyDuration cityASchedule cityBSchedule = 5 := by
  sorry

end NUMINAMATH_CALUDE_buses_passed_count_l2430_243087


namespace NUMINAMATH_CALUDE_partnership_contribution_time_l2430_243085

/-- Proves that given the conditions of the partnership problem, A contributed for 8 months -/
theorem partnership_contribution_time (a_contribution b_contribution total_profit a_share : ℚ)
  (b_time : ℕ) :
  a_contribution = 5000 →
  b_contribution = 6000 →
  b_time = 5 →
  total_profit = 8400 →
  a_share = 4800 →
  ∃ (a_time : ℕ),
    a_time = 8 ∧
    a_share / total_profit = (a_contribution * a_time) / (a_contribution * a_time + b_contribution * b_time) :=
by sorry

end NUMINAMATH_CALUDE_partnership_contribution_time_l2430_243085


namespace NUMINAMATH_CALUDE_least_multiple_of_29_above_500_l2430_243053

theorem least_multiple_of_29_above_500 : 
  ∀ n : ℕ, n > 0 ∧ 29 ∣ n ∧ n > 500 → n ≥ 522 := by
  sorry

end NUMINAMATH_CALUDE_least_multiple_of_29_above_500_l2430_243053


namespace NUMINAMATH_CALUDE_certain_number_proof_l2430_243005

theorem certain_number_proof : ∃ x : ℝ, 45 * x = 0.35 * 900 ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2430_243005


namespace NUMINAMATH_CALUDE_EL_length_l2430_243033

-- Define the rectangle
def rectangle_EFGH : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define points E and K
def E : ℝ × ℝ := (0, 1)
def K : ℝ × ℝ := (1, 0)

-- Define the inscribed circle ω
def ω : Set (ℝ × ℝ) :=
  {p | (p.1 - 1)^2 + (p.2 - 0.5)^2 = 0.25}

-- Define the line EK
def line_EK (x : ℝ) : ℝ := -x + 1

-- Define point L as the intersection of EK and ω (different from K)
def L : ℝ × ℝ :=
  let x := 0.5
  (x, line_EK x)

-- Theorem statement
theorem EL_length :
  let el_length := Real.sqrt ((L.1 - E.1)^2 + (L.2 - E.2)^2)
  el_length = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_EL_length_l2430_243033


namespace NUMINAMATH_CALUDE_hcf_of_4_and_18_l2430_243023

theorem hcf_of_4_and_18 :
  let a : ℕ := 4
  let b : ℕ := 18
  let lcm_ab : ℕ := 36
  Nat.lcm a b = lcm_ab →
  Nat.gcd a b = 2 := by
sorry

end NUMINAMATH_CALUDE_hcf_of_4_and_18_l2430_243023


namespace NUMINAMATH_CALUDE_sqrt_calculation_l2430_243002

theorem sqrt_calculation : Real.sqrt 24 * Real.sqrt (1/6) - (-Real.sqrt 7)^2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculation_l2430_243002


namespace NUMINAMATH_CALUDE_square_perimeter_l2430_243007

theorem square_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 360 →
  area = side ^ 2 →
  perimeter = 4 * side →
  perimeter = 24 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_l2430_243007


namespace NUMINAMATH_CALUDE_intersection_points_properties_l2430_243019

/-- The curve equation -/
def curve (x : ℝ) : ℝ := x^2 - 5*x + 4

/-- The line equation -/
def line (p : ℝ) : ℝ := p

/-- Theorem stating the properties of the intersection points -/
theorem intersection_points_properties (a b p : ℝ) : 
  (curve a = line p) ∧ 
  (curve b = line p) ∧ 
  (a ≠ b) ∧
  (a^4 + b^4 = 1297) →
  (a = 6 ∧ b = -1) ∨ (a = -1 ∧ b = 6) := by
  sorry

#check intersection_points_properties

end NUMINAMATH_CALUDE_intersection_points_properties_l2430_243019


namespace NUMINAMATH_CALUDE_bottom_face_points_l2430_243030

/-- Represents the number of points on each face of a cube -/
structure CubePoints where
  front : ℕ
  back : ℕ
  left : ℕ
  right : ℕ
  top : ℕ
  bottom : ℕ

/-- Theorem stating the number of points on the bottom face of the cube -/
theorem bottom_face_points (c : CubePoints) 
  (opposite_sum : c.front + c.back = 13 ∧ c.left + c.right = 13 ∧ c.top + c.bottom = 13)
  (front_left_top_sum : c.front + c.left + c.top = 16)
  (top_right_back_sum : c.top + c.right + c.back = 24) :
  c.bottom = 6 := by
  sorry

end NUMINAMATH_CALUDE_bottom_face_points_l2430_243030


namespace NUMINAMATH_CALUDE_scarves_difference_formula_l2430_243018

/-- Calculates the difference in scarves produced between a normal day and a tiring day. -/
def scarves_difference (h : ℝ) : ℝ :=
  let s := 3 * h
  let normal_day := s * h
  let tiring_day := (s - 2) * (h - 3)
  normal_day - tiring_day

/-- Theorem stating that the difference in scarves produced is 11h - 6. -/
theorem scarves_difference_formula (h : ℝ) :
  scarves_difference h = 11 * h - 6 := by
  sorry

end NUMINAMATH_CALUDE_scarves_difference_formula_l2430_243018


namespace NUMINAMATH_CALUDE_min_sum_of_squares_with_diff_l2430_243048

theorem min_sum_of_squares_with_diff (x y : ℕ+) : 
  x.val^2 - y.val^2 = 145 → x.val^2 + y.val^2 ≥ 433 := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_with_diff_l2430_243048


namespace NUMINAMATH_CALUDE_circle_reflection_translation_l2430_243044

def reflect_across_y_axis (x y : ℝ) : ℝ × ℝ := (-x, y)

def translate_up (point : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (point.1, point.2 + units)

theorem circle_reflection_translation (center : ℝ × ℝ) :
  center = (3, -4) →
  (translate_up (reflect_across_y_axis center.1 center.2) 5) = (-3, 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_reflection_translation_l2430_243044


namespace NUMINAMATH_CALUDE_sum_of_digits_An_l2430_243092

-- Define the product An
def An (n : ℕ) : ℕ :=
  (List.range (n + 1)).foldl (λ acc i => acc * (10^(2^i) - 1)) 9

-- Define the sum of digits function
def sumOfDigits (m : ℕ) : ℕ :=
  if m < 10 then m else m % 10 + sumOfDigits (m / 10)

-- State the theorem
theorem sum_of_digits_An (n : ℕ) :
  sumOfDigits (An n) = 9 * 2^n := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_An_l2430_243092


namespace NUMINAMATH_CALUDE_fraction_simplification_l2430_243021

theorem fraction_simplification (a : ℝ) (h : a ≠ 0) : (a^2 - 1) / a + 1 / a = a := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2430_243021


namespace NUMINAMATH_CALUDE_expression_evaluation_l2430_243082

theorem expression_evaluation (x y : ℝ) (hx : x = -2) (hy : y = 1/3) :
  (2*y + 3*x^2) - (x^2 - y) - x^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2430_243082


namespace NUMINAMATH_CALUDE_expression_equality_l2430_243077

theorem expression_equality : 784 + 2 * 28 * 7 + 49 = 1225 := by sorry

end NUMINAMATH_CALUDE_expression_equality_l2430_243077


namespace NUMINAMATH_CALUDE_solve_equation_l2430_243036

theorem solve_equation (b : ℚ) (h : b + b / 4 = 5 / 2) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2430_243036


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2430_243009

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (-x) + Real.cos x

theorem inequality_solution_set (m : ℝ) :
  f (2 * m) > f (m - 2) ↔ m < -2 ∨ m > 2/3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2430_243009


namespace NUMINAMATH_CALUDE_max_m_value_l2430_243094

theorem max_m_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = x + 2 * y) :
  ∀ m : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → x * y = x + 2 * y → x * y ≥ m - 2) → m ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l2430_243094


namespace NUMINAMATH_CALUDE_smartphone_savings_proof_l2430_243095

/-- Calculates the required weekly savings to reach a target amount. -/
def weekly_savings (smartphone_cost : ℚ) (current_savings : ℚ) (saving_weeks : ℕ) : ℚ :=
  (smartphone_cost - current_savings) / saving_weeks

/-- Proves that the weekly savings required to buy a $160 smartphone
    with $40 current savings over 8 weeks is $15. -/
theorem smartphone_savings_proof :
  let smartphone_cost : ℚ := 160
  let current_savings : ℚ := 40
  let saving_weeks : ℕ := 8
  weekly_savings smartphone_cost current_savings saving_weeks = 15 := by
sorry

end NUMINAMATH_CALUDE_smartphone_savings_proof_l2430_243095


namespace NUMINAMATH_CALUDE_equation_solution_l2430_243027

theorem equation_solution : ∃ x : ℝ, x > 0 ∧ 6 * x^(1/3) - 3 * (x / x^(2/3)) = -1 + 2 * x^(1/3) + 4 ∧ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2430_243027


namespace NUMINAMATH_CALUDE_complex_number_property_l2430_243049

theorem complex_number_property : 
  let z : ℂ := (-2 * Complex.I) / (1 + Complex.I)
  (z + 1).im ≠ 0 ∧ (z + 1).re = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_property_l2430_243049


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2430_243039

theorem binomial_expansion_coefficient (m : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2 - m * x)^5 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5) →
  a₃ = 40 →
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2430_243039


namespace NUMINAMATH_CALUDE_f_value_at_one_l2430_243090

/-- The polynomial g(x) -/
def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x + 10

/-- The polynomial f(x) -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + b*x^2 + 100*x + c

theorem f_value_at_one (a b c : ℝ) : 
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    g a r₁ = 0 ∧ g a r₂ = 0 ∧ g a r₃ = 0) →
  (∀ x : ℝ, g a x = 0 → f b c x = 0) →
  f b c 1 = -7007 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_one_l2430_243090


namespace NUMINAMATH_CALUDE_sine_midpoint_inequality_l2430_243074

theorem sine_midpoint_inequality (x₁ x₂ : ℝ) 
  (h₁ : 0 < x₁) (h₂ : x₁ < π) (h₃ : 0 < x₂) (h₄ : x₂ < π) (h₅ : x₁ ≠ x₂) : 
  (Real.sin x₁ + Real.sin x₂) / 2 < Real.sin ((x₁ + x₂) / 2) := by
  sorry

end NUMINAMATH_CALUDE_sine_midpoint_inequality_l2430_243074


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2430_243010

theorem imaginary_part_of_z (z : ℂ) : z = 2 / (-1 + Complex.I) → z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2430_243010


namespace NUMINAMATH_CALUDE_impossibility_of_zero_sum_l2430_243038

/-- The sum of the first n natural numbers -/
def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents a configuration of signs between numbers 1 to 10 -/
def SignConfiguration := Fin 9 → Bool

/-- Calculates the sum based on a given sign configuration -/
def calculate_sum (config : SignConfiguration) : ℤ :=
  sorry

theorem impossibility_of_zero_sum : ∀ (config : SignConfiguration), 
  calculate_sum config ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_impossibility_of_zero_sum_l2430_243038


namespace NUMINAMATH_CALUDE_jaydon_rachel_ratio_l2430_243065

-- Define the number of cans for each person
def mark_cans : ℕ := 100
def jaydon_cans : ℕ := 25
def rachel_cans : ℕ := 10

-- Define the total number of cans
def total_cans : ℕ := 135

-- Define the conditions
axiom mark_jaydon_relation : mark_cans = 4 * jaydon_cans
axiom total_cans_sum : total_cans = mark_cans + jaydon_cans + rachel_cans
axiom jaydon_rachel_relation : ∃ k : ℕ, jaydon_cans = k * rachel_cans + 5

-- Theorem to prove
theorem jaydon_rachel_ratio : 
  (jaydon_cans : ℚ) / rachel_cans = 5 / 2 := by sorry

end NUMINAMATH_CALUDE_jaydon_rachel_ratio_l2430_243065


namespace NUMINAMATH_CALUDE_not_always_left_to_right_l2430_243028

theorem not_always_left_to_right : ∃ (a b c : ℕ), a + b * c ≠ (a + b) * c := by sorry

end NUMINAMATH_CALUDE_not_always_left_to_right_l2430_243028


namespace NUMINAMATH_CALUDE_slurpee_purchase_l2430_243008

theorem slurpee_purchase (money_given : ℕ) (slurpee_cost : ℕ) (change : ℕ) : 
  money_given = 20 ∧ slurpee_cost = 2 ∧ change = 8 → 
  (money_given - change) / slurpee_cost = 6 := by
  sorry

end NUMINAMATH_CALUDE_slurpee_purchase_l2430_243008


namespace NUMINAMATH_CALUDE_function_value_at_log_third_l2430_243069

/-- Given a function f and a real number a, proves that f(ln(1/3)) = -1 -/
theorem function_value_at_log_third (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 2^x / (2^x + 1) + a * x
  f (Real.log 3) = 2 → f (Real.log (1/3)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_log_third_l2430_243069


namespace NUMINAMATH_CALUDE_three_digit_number_divisible_by_seven_l2430_243017

theorem three_digit_number_divisible_by_seven :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 
  n % 10 = 4 ∧ 
  (n / 100) % 10 = 5 ∧ 
  n % 7 = 0 ∧
  n = 534 := by
sorry

end NUMINAMATH_CALUDE_three_digit_number_divisible_by_seven_l2430_243017


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2430_243056

theorem arithmetic_calculations :
  (24 - (-16) + (-25) - 15 = 0) ∧
  ((-81) + 2.25 * (4/9) / (-16) = -81 - 1/16) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2430_243056


namespace NUMINAMATH_CALUDE_unpainted_cubes_in_6x6x6_l2430_243041

/-- Represents a cube composed of unit cubes -/
structure Cube where
  size : Nat
  total_units : Nat
  painted_per_face : Nat

/-- Calculates the number of unpainted unit cubes in a cube with painted cross patterns on each face -/
def unpainted_cubes (c : Cube) : Nat :=
  c.total_units - (c.painted_per_face * 6 - 24 - 12)

/-- Theorem stating the number of unpainted cubes in the specific 6x6x6 cube -/
theorem unpainted_cubes_in_6x6x6 :
  let c : Cube := { size := 6, total_units := 216, painted_per_face := 10 }
  unpainted_cubes c = 180 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_in_6x6x6_l2430_243041


namespace NUMINAMATH_CALUDE_quadratic_b_value_l2430_243099

/-- The value of b in a quadratic function y = x² - bx + c passing through (1,n) and (3,n) -/
theorem quadratic_b_value (n : ℝ) : 
  ∃ (b c : ℝ), (∀ x : ℝ, x^2 - b*x + c = n ↔ x = 1 ∨ x = 3) → b = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_b_value_l2430_243099


namespace NUMINAMATH_CALUDE_classroom_window_2023_l2430_243024

/-- Represents a digit as seen through a transparent surface --/
inductive MirroredDigit
| Zero
| Two
| Three

/-- Represents the appearance of a number when viewed through a transparent surface --/
def mirror_number (n : List Nat) : List MirroredDigit :=
  sorry

/-- The property of being viewed from the opposite side of a transparent surface --/
def viewed_from_opposite_side (original : List Nat) (mirrored : List MirroredDigit) : Prop :=
  mirror_number original = mirrored.reverse

theorem classroom_window_2023 :
  viewed_from_opposite_side [2, 0, 2, 3] [MirroredDigit.Three, MirroredDigit.Two, MirroredDigit.Zero, MirroredDigit.Two] :=
by sorry

end NUMINAMATH_CALUDE_classroom_window_2023_l2430_243024


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2430_243072

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define the eccentricity of the hyperbola
def hyperbola_eccentricity : ℝ := 2

-- Define the standard form of a hyperbola
def is_standard_hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Theorem statement
theorem hyperbola_equation : 
  ∃ (a b : ℝ), a^2 = 4 ∧ b^2 = 12 ∧ 
  (∀ (x y : ℝ), is_standard_hyperbola a b x y) ∧
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ 
   c = hyperbola_eccentricity * a ∧
   c^2 = 25 - 9) := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2430_243072


namespace NUMINAMATH_CALUDE_inverse_variation_proof_l2430_243035

/-- Given that x² varies inversely with y⁴, prove that x² = 4 when y = 4, given x = 8 when y = 2 -/
theorem inverse_variation_proof (x y : ℝ) (h1 : ∃ k : ℝ, ∀ x y, x^2 * y^4 = k) 
  (h2 : ∃ x₀ y₀ : ℝ, x₀ = 8 ∧ y₀ = 2 ∧ x₀^2 * y₀^4 = k) : 
  ∃ x₁ : ℝ, x₁^2 = 4 ∧ x₁^2 * 4^4 = k :=
sorry

end NUMINAMATH_CALUDE_inverse_variation_proof_l2430_243035


namespace NUMINAMATH_CALUDE_arrangement_theorem_l2430_243091

/-- The number of ways to arrange 3 male and 2 female students in a row with females not at ends -/
def arrangement_count : ℕ := sorry

/-- There are 3 male students -/
def male_count : ℕ := 3

/-- There are 2 female students -/
def female_count : ℕ := 2

/-- Total number of students -/
def total_students : ℕ := male_count + female_count

/-- Number of positions where female students can stand (not at ends) -/
def female_positions : ℕ := total_students - 2

theorem arrangement_theorem : arrangement_count = 36 := by sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l2430_243091


namespace NUMINAMATH_CALUDE_nested_series_sum_l2430_243034

def nested_series : ℕ → ℕ
  | 0 => 2
  | n + 1 => 2 * (1 + nested_series n)

theorem nested_series_sum : nested_series 5 = 126 := by
  sorry

end NUMINAMATH_CALUDE_nested_series_sum_l2430_243034


namespace NUMINAMATH_CALUDE_power_exceeds_any_number_l2430_243059

theorem power_exceeds_any_number (p M : ℝ) (hp : p > 0) (hM : M > 0) :
  ∃ n : ℕ, (1 + p)^n > M := by sorry

end NUMINAMATH_CALUDE_power_exceeds_any_number_l2430_243059


namespace NUMINAMATH_CALUDE_peanut_cluster_probability_theorem_l2430_243016

/-- Represents the composition of a box of chocolates -/
structure ChocolateBox where
  total : Nat
  caramels : Nat
  nougats : Nat
  truffles : Nat
  peanut_clusters : Nat

/-- Calculates the probability of selecting a peanut cluster -/
def peanut_cluster_probability (box : ChocolateBox) : Rat :=
  box.peanut_clusters / box.total

/-- Theorem stating the probability of selecting a peanut cluster -/
theorem peanut_cluster_probability_theorem (box : ChocolateBox) 
  (h1 : box.total = 50)
  (h2 : box.caramels = 3)
  (h3 : box.nougats = 2 * box.caramels)
  (h4 : box.truffles = box.caramels + 6)
  (h5 : box.peanut_clusters = box.total - box.caramels - box.nougats - box.truffles) :
  peanut_cluster_probability box = 32 / 50 := by
  sorry

#eval (32 : Rat) / 50

end NUMINAMATH_CALUDE_peanut_cluster_probability_theorem_l2430_243016
