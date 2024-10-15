import Mathlib

namespace NUMINAMATH_CALUDE_five_graduates_three_companies_l3900_390014

/-- The number of ways to assign n graduates to k companies, with each company hiring at least one person -/
def assignGraduates (n k : ℕ) : ℕ :=
  sorry

theorem five_graduates_three_companies : 
  assignGraduates 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_five_graduates_three_companies_l3900_390014


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l3900_390002

theorem ceiling_floor_sum : ⌈(7:ℝ)/3⌉ + ⌊-(7:ℝ)/3⌋ = 0 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l3900_390002


namespace NUMINAMATH_CALUDE_product_digit_sum_l3900_390073

theorem product_digit_sum : 
  let product := 2 * 3 * 5 * 7 * 11 * 13 * 17
  ∃ (digits : List Nat), 
    (∀ d ∈ digits, d < 10) ∧ 
    (product.repr.toList.map (λ c => c.toNat - '0'.toNat) = digits) ∧
    (digits.sum = 12) := by
  sorry

end NUMINAMATH_CALUDE_product_digit_sum_l3900_390073


namespace NUMINAMATH_CALUDE_smallest_n_doughnuts_l3900_390076

theorem smallest_n_doughnuts : ∃ n : ℕ+, 
  (∀ m : ℕ+, (15 * m.val - 1) % 11 = 0 → n ≤ m) ∧
  (15 * n.val - 1) % 11 = 0 ∧
  n.val = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_doughnuts_l3900_390076


namespace NUMINAMATH_CALUDE_book_distribution_l3900_390000

def number_of_books : ℕ := 6
def number_of_people : ℕ := 3

def distribute_evenly (n m : ℕ) : ℕ :=
  Nat.choose n 2 * Nat.choose (n - 2) 2

def distribute_fixed (n : ℕ) : ℕ :=
  Nat.choose n 1 * Nat.choose (n - 1) 2

def distribute_variable (n m : ℕ) : ℕ :=
  Nat.choose n 1 * Nat.choose (n - 1) 2 * Nat.factorial m

theorem book_distribution :
  (distribute_evenly number_of_books number_of_people = 90) ∧
  (distribute_fixed number_of_books = 60) ∧
  (distribute_variable number_of_books number_of_people = 360) := by
  sorry

end NUMINAMATH_CALUDE_book_distribution_l3900_390000


namespace NUMINAMATH_CALUDE_mathematics_letter_probability_l3900_390097

theorem mathematics_letter_probability : 
  let alphabet_size : ℕ := 26
  let unique_letters : ℕ := 8
  let probability : ℚ := unique_letters / alphabet_size
  probability = 4 / 13 := by
sorry

end NUMINAMATH_CALUDE_mathematics_letter_probability_l3900_390097


namespace NUMINAMATH_CALUDE_train_speed_l3900_390015

/-- The speed of a train given its length and time to cross an electric pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 700) (h2 : time = 20) :
  length / time = 35 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3900_390015


namespace NUMINAMATH_CALUDE_maxwells_walking_speed_l3900_390009

/-- Proves that Maxwell's walking speed is 24 km/h given the problem conditions -/
theorem maxwells_walking_speed 
  (total_distance : ℝ) 
  (brads_speed : ℝ) 
  (maxwell_distance : ℝ) 
  (h1 : total_distance = 72) 
  (h2 : brads_speed = 12) 
  (h3 : maxwell_distance = 24) : 
  maxwell_distance / (maxwell_distance / brads_speed) = 24 := by
  sorry

end NUMINAMATH_CALUDE_maxwells_walking_speed_l3900_390009


namespace NUMINAMATH_CALUDE_rectangle_area_problem_l3900_390074

theorem rectangle_area_problem (A : ℝ) : 
  let square_side : ℝ := 12
  let new_horizontal : ℝ := square_side + 3
  let new_vertical : ℝ := square_side - A
  let new_area : ℝ := 120
  new_horizontal * new_vertical = new_area → A = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_problem_l3900_390074


namespace NUMINAMATH_CALUDE_rectangular_prism_base_area_l3900_390027

theorem rectangular_prism_base_area :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a % 5 = 0 ∧ b % 5 = 0 ∧ a * b = 450 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_base_area_l3900_390027


namespace NUMINAMATH_CALUDE_parabola_equation_l3900_390045

/-- Given a parabola y = 2px (p > 0) and a point M on it with abscissa 3,
    if |MF| = 2p, then the equation of the parabola is y^2 = 4x -/
theorem parabola_equation (p : ℝ) (h1 : p > 0) :
  ∃ (M : ℝ × ℝ),
    M.1 = 3 ∧
    M.2 = 2 * p * M.1 ∧
    |M.1 - (-p/2)| + M.2 = 2 * p →
  ∀ (x y : ℝ), y = 2 * p * x ↔ y^2 = 4 * x :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l3900_390045


namespace NUMINAMATH_CALUDE_perfect_squares_between_210_and_560_l3900_390069

theorem perfect_squares_between_210_and_560 :
  (Finset.filter (fun n => 210 < n^2 ∧ n^2 < 560) (Finset.range 24)).card = 9 :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_between_210_and_560_l3900_390069


namespace NUMINAMATH_CALUDE_f_4_1981_equals_tower_exp_l3900_390033

/-- A function f : ℕ → ℕ → ℕ satisfying the given recursive conditions -/
noncomputable def f : ℕ → ℕ → ℕ
| 0, y => y + 1
| x + 1, 0 => f x 1
| x + 1, y + 1 => f x (f (x + 1) y)

/-- Helper function to represent towering exponentiation -/
def tower_exp : ℕ → ℕ → ℕ
| 0, n => n
| m + 1, n => 2^(tower_exp m n)

/-- The main theorem stating that f(4, 1981) is equal to a specific towering exponentiation -/
theorem f_4_1981_equals_tower_exp : 
  f 4 1981 = tower_exp 12 (2^2) :=
sorry

end NUMINAMATH_CALUDE_f_4_1981_equals_tower_exp_l3900_390033


namespace NUMINAMATH_CALUDE_stairs_height_l3900_390059

theorem stairs_height (h : ℝ) 
  (total_height : 3 * h + h / 2 + (h / 2 + 10) = 70) : h = 15 := by
  sorry

end NUMINAMATH_CALUDE_stairs_height_l3900_390059


namespace NUMINAMATH_CALUDE_linear_function_composition_l3900_390080

-- Define a linear function
def IsLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, f x = a * x + b

-- State the theorem
theorem linear_function_composition (f : ℝ → ℝ) :
  IsLinearFunction f → (∀ x, f (f x) = 4 * x + 8) →
  (∀ x, f x = 2 * x + 8 / 3) ∨ (∀ x, f x = -2 * x - 8) :=
by
  sorry

end NUMINAMATH_CALUDE_linear_function_composition_l3900_390080


namespace NUMINAMATH_CALUDE_system_solution_l3900_390063

theorem system_solution : ∃ (a b c d : ℝ), 
  (a + c = -4 ∧ 
   a * c + b + d = 6 ∧ 
   a * d + b * c = -5 ∧ 
   b * d = 2) ∧
  ((a = -3 ∧ b = 2 ∧ c = -1 ∧ d = 1) ∨
   (a = -1 ∧ b = 1 ∧ c = -3 ∧ d = 2)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3900_390063


namespace NUMINAMATH_CALUDE_ellipse_equation_l3900_390029

-- Define the ellipse
structure Ellipse where
  foci : (ℝ × ℝ) × (ℝ × ℝ)
  majorAxisLength : ℝ

-- Define the standard form of an ellipse equation
def StandardEllipseEquation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Theorem statement
theorem ellipse_equation (e : Ellipse) : 
  e.foci = ((-2, 0), (2, 0)) ∧ e.majorAxisLength = 10 →
  ∀ x y : ℝ, StandardEllipseEquation 25 21 x y :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3900_390029


namespace NUMINAMATH_CALUDE_sqrt_five_minus_one_gt_one_l3900_390017

theorem sqrt_five_minus_one_gt_one : Real.sqrt 5 - 1 > 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_minus_one_gt_one_l3900_390017


namespace NUMINAMATH_CALUDE_c_minus_a_positive_l3900_390036

/-- A quadratic function y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The graph of the quadratic function is a downward-opening parabola -/
def is_downward_opening (f : QuadraticFunction) : Prop :=
  f.a < 0

/-- The y-intercept of the quadratic function is positive -/
def has_positive_y_intercept (f : QuadraticFunction) : Prop :=
  f.c > 0

/-- Theorem stating that if a quadratic function's graph is a downward-opening parabola
    with a positive y-intercept, then c - a > 0 -/
theorem c_minus_a_positive (f : QuadraticFunction)
  (h1 : is_downward_opening f)
  (h2 : has_positive_y_intercept f) :
  f.c - f.a > 0 := by
  sorry

end NUMINAMATH_CALUDE_c_minus_a_positive_l3900_390036


namespace NUMINAMATH_CALUDE_altitude_segment_length_l3900_390086

/-- Represents an acute triangle with two altitudes dividing the sides -/
structure AcuteTriangleWithAltitudes where
  /-- Length of one segment created by an altitude -/
  segment1 : ℝ
  /-- Length of another segment created by an altitude -/
  segment2 : ℝ
  /-- Length of a third segment created by an altitude -/
  segment3 : ℝ
  /-- Length of the fourth segment created by an altitude -/
  segment4 : ℝ
  /-- The triangle is acute -/
  acute : segment1 > 0 ∧ segment2 > 0 ∧ segment3 > 0 ∧ segment4 > 0

/-- The theorem stating that for the given acute triangle with altitudes, the fourth segment length is 4.5 -/
theorem altitude_segment_length (t : AcuteTriangleWithAltitudes) 
  (h1 : t.segment1 = 4) 
  (h2 : t.segment2 = 6) 
  (h3 : t.segment3 = 3) : 
  t.segment4 = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_altitude_segment_length_l3900_390086


namespace NUMINAMATH_CALUDE_matrix_power_2018_l3900_390093

def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 1; 1, 1]

theorem matrix_power_2018 :
  A ^ 2018 = !![2^2017, 2^2017; 2^2017, 2^2017] := by sorry

end NUMINAMATH_CALUDE_matrix_power_2018_l3900_390093


namespace NUMINAMATH_CALUDE_sum_of_four_cubes_equals_1812_l3900_390025

theorem sum_of_four_cubes_equals_1812 :
  (303 : ℤ)^3 + (301 : ℤ)^3 + (-302 : ℤ)^3 + (-302 : ℤ)^3 = 1812 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_cubes_equals_1812_l3900_390025


namespace NUMINAMATH_CALUDE_double_inequality_solution_l3900_390031

theorem double_inequality_solution (x : ℝ) : 
  (4 * x - 3 < (x - 2)^2 ∧ (x - 2)^2 < 6 * x - 5) ↔ (7 < x ∧ x < 9) :=
sorry

end NUMINAMATH_CALUDE_double_inequality_solution_l3900_390031


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3900_390040

/-- Given a principal amount and an interest rate, if increasing the rate by 4% for 2 years
    yields Rs. 60 more in interest, then the principal amount is Rs. 750. -/
theorem simple_interest_problem (P R : ℝ) (h : P > 0) (k : R > 0) :
  (P * (R + 4) * 2) / 100 = (P * R * 2) / 100 + 60 → P = 750 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3900_390040


namespace NUMINAMATH_CALUDE_car_distances_theorem_l3900_390090

/-- Represents the distance traveled by a car -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem stating the distances traveled by two cars under given conditions -/
theorem car_distances_theorem (distance_AB : ℝ) (speed_car1 : ℝ) (speed_car2 : ℝ) 
  (h1 : distance_AB = 70)
  (h2 : speed_car1 = 30)
  (h3 : speed_car2 = 40)
  (h4 : speed_car1 + speed_car2 > 0) -- Ensure division by zero is avoided
  : ∃ (time : ℝ), 
    distance speed_car1 time = 150 ∧ 
    distance speed_car2 time = 200 ∧
    time * (speed_car1 + speed_car2) = 5 * distance_AB :=
sorry

end NUMINAMATH_CALUDE_car_distances_theorem_l3900_390090


namespace NUMINAMATH_CALUDE_factorization_of_polynomial_l3900_390062

theorem factorization_of_polynomial (z : ℝ) : 
  75 * z^24 + 225 * z^48 = 75 * z^24 * (1 + 3 * z^24) := by
sorry

end NUMINAMATH_CALUDE_factorization_of_polynomial_l3900_390062


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l3900_390082

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 10) / (Nat.factorial 4)) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l3900_390082


namespace NUMINAMATH_CALUDE_cookies_per_bag_l3900_390022

theorem cookies_per_bag (total_cookies : ℕ) (num_bags : ℕ) (h1 : total_cookies = 703) (h2 : num_bags = 37) :
  total_cookies / num_bags = 19 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l3900_390022


namespace NUMINAMATH_CALUDE_equation_solution_l3900_390041

/-- Given an equation x^4 - 10x^3 - 2(a-11)x^2 + 2(5a+6)x + 2a + a^2 = 0,
    where a is a constant and a ≥ -6, prove the solutions for a and x. -/
theorem equation_solution (a x : ℝ) (h : a ≥ -6) :
  x^4 - 10*x^3 - 2*(a-11)*x^2 + 2*(5*a+6)*x + 2*a + a^2 = 0 →
  ((a = x^2 - 4*x - 2) ∨ (a = x^2 - 6*x)) ∧
  ((∃ (i : Fin 2), x = 2 + (-1)^(i : ℕ) * Real.sqrt (a + 6)) ∨
   (∃ (i : Fin 2), x = 3 + (-1)^(i : ℕ) * Real.sqrt (a + 9))) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3900_390041


namespace NUMINAMATH_CALUDE_student_sums_correct_l3900_390005

theorem student_sums_correct (wrong_sums correct_sums total_sums : ℕ) : 
  wrong_sums = 2 * correct_sums →
  total_sums = 36 →
  wrong_sums + correct_sums = total_sums →
  correct_sums = 12 := by
  sorry

end NUMINAMATH_CALUDE_student_sums_correct_l3900_390005


namespace NUMINAMATH_CALUDE_strategy_game_cost_l3900_390094

def total_spent : ℝ := 35.52
def football_cost : ℝ := 14.02
def batman_cost : ℝ := 12.04

theorem strategy_game_cost :
  total_spent - football_cost - batman_cost = 9.46 := by
  sorry

end NUMINAMATH_CALUDE_strategy_game_cost_l3900_390094


namespace NUMINAMATH_CALUDE_function_properties_l3900_390030

/-- The function f(x) defined on the real line -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 - 4 * a * x + b

theorem function_properties (a b : ℝ) (h_a : a > 0) 
  (h_max : ∀ x ∈ Set.Icc 0 1, f a b x ≤ 1) 
  (h_max_exists : ∃ x ∈ Set.Icc 0 1, f a b x = 1)
  (h_min : ∀ x ∈ Set.Icc 0 1, f a b x ≥ -2) 
  (h_min_exists : ∃ x ∈ Set.Icc 0 1, f a b x = -2) :
  (a = 1 ∧ b = 1) ∧ 
  (∀ m : ℝ, (∀ x ∈ Set.Icc (-1) 1, f a b x > -x + m) ↔ m < -1) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l3900_390030


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3900_390054

theorem solve_linear_equation (x : ℝ) : 3 * x + 7 = -2 → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3900_390054


namespace NUMINAMATH_CALUDE_andy_remaining_demerits_l3900_390096

/-- The maximum number of demerits Andy can get in a month before getting fired -/
def max_demerits : ℕ := 50

/-- The number of times Andy showed up late in the first week -/
def late_instances : ℕ := 6

/-- The number of demerits Andy gets for each late instance -/
def late_demerits : ℕ := 2

/-- The number of demerits Andy got for making an inappropriate joke in the second week -/
def joke_demerits : ℕ := 15

/-- The number of times Andy used his phone during work hours in the third week -/
def phone_instances : ℕ := 4

/-- The number of demerits Andy gets for each phone use instance -/
def phone_demerits : ℕ := 3

/-- The number of days Andy didn't tidy up his work area in the fourth week -/
def untidy_days : ℕ := 5

/-- The number of demerits Andy gets for each day of not tidying up -/
def untidy_demerits : ℕ := 1

/-- The total number of demerits Andy has accumulated so far -/
def total_demerits : ℕ := 
  late_instances * late_demerits + 
  joke_demerits + 
  phone_instances * phone_demerits + 
  untidy_days * untidy_demerits

/-- The number of additional demerits Andy can receive before getting fired -/
def additional_demerits : ℕ := max_demerits - total_demerits

theorem andy_remaining_demerits : additional_demerits = 6 := by
  sorry

end NUMINAMATH_CALUDE_andy_remaining_demerits_l3900_390096


namespace NUMINAMATH_CALUDE_binomial_coefficient_seven_two_l3900_390039

theorem binomial_coefficient_seven_two : 
  Nat.choose 7 2 = 21 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_seven_two_l3900_390039


namespace NUMINAMATH_CALUDE_least_frood_drop_beats_eat_l3900_390035

theorem least_frood_drop_beats_eat : 
  ∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, k > 0 → k < n → (k * (k + 1)) / 2 ≤ 15 * k) ∧ (n * (n + 1)) / 2 > 15 * n :=
by sorry

end NUMINAMATH_CALUDE_least_frood_drop_beats_eat_l3900_390035


namespace NUMINAMATH_CALUDE_a2_value_l3900_390085

theorem a2_value (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, x^4 = a₀ + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3 + a₄*(x-2)^4) →
  a₂ = 24 := by
sorry

end NUMINAMATH_CALUDE_a2_value_l3900_390085


namespace NUMINAMATH_CALUDE_women_married_fraction_l3900_390056

theorem women_married_fraction (total : ℕ) (women : ℕ) (married : ℕ) (men : ℕ) :
  women = (61 * total) / 100 →
  married = (60 * total) / 100 →
  men = total - women →
  (men - (men / 3)) * 3 = 2 * men →
  (married - (men / 3) : ℚ) / women = 47 / 61 :=
by
  sorry

end NUMINAMATH_CALUDE_women_married_fraction_l3900_390056


namespace NUMINAMATH_CALUDE_fraction_sum_division_specific_fraction_sum_division_l3900_390091

theorem fraction_sum_division (a b c d : ℚ) :
  (a / b + c / d) / 4 = (a * d + b * c) / (4 * b * d) :=
by sorry

theorem specific_fraction_sum_division :
  (2 / 5 + 1 / 3) / 4 = 11 / 60 :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_division_specific_fraction_sum_division_l3900_390091


namespace NUMINAMATH_CALUDE_terminal_side_first_quadrant_l3900_390012

-- Define the angle in degrees
def angle : ℤ := -685

-- Define a function to normalize an angle to the range [0, 360)
def normalizeAngle (a : ℤ) : ℤ :=
  (a % 360 + 360) % 360

-- Define a function to determine the quadrant of an angle
def quadrant (a : ℤ) : ℕ :=
  let normalizedAngle := normalizeAngle a
  if 0 ≤ normalizedAngle ∧ normalizedAngle < 90 then 1
  else if 90 ≤ normalizedAngle ∧ normalizedAngle < 180 then 2
  else if 180 ≤ normalizedAngle ∧ normalizedAngle < 270 then 3
  else 4

-- Theorem statement
theorem terminal_side_first_quadrant :
  quadrant angle = 1 := by sorry

end NUMINAMATH_CALUDE_terminal_side_first_quadrant_l3900_390012


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_integers_with_five_l3900_390042

def is_divisible (a b : ℤ) : Prop := ∃ k : ℤ, a = b * k

theorem largest_divisor_of_consecutive_integers_with_five (n : ℤ) :
  (is_divisible n 5 ∨ is_divisible (n + 1) 5 ∨ is_divisible (n + 2) 5) →
  is_divisible (n * (n + 1) * (n + 2)) 15 ∧
  ∀ m : ℤ, m > 15 → ¬(∀ k : ℤ, (is_divisible k 5 ∨ is_divisible (k + 1) 5 ∨ is_divisible (k + 2) 5) →
                              is_divisible (k * (k + 1) * (k + 2)) m) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_integers_with_five_l3900_390042


namespace NUMINAMATH_CALUDE_pentagon_fencing_cost_l3900_390034

/-- Calculates the total cost of fencing a pentagon park -/
def fencing_cost (sides : Fin 5 → ℝ) (costs : Fin 5 → ℝ) : ℝ :=
  (sides 0 * costs 0) + (sides 1 * costs 1) + (sides 2 * costs 2) + 
  (sides 3 * costs 3) + (sides 4 * costs 4)

theorem pentagon_fencing_cost :
  let sides : Fin 5 → ℝ := ![50, 75, 60, 80, 65]
  let costs : Fin 5 → ℝ := ![2, 3, 4, 3.5, 5]
  fencing_cost sides costs = 1170 := by sorry

end NUMINAMATH_CALUDE_pentagon_fencing_cost_l3900_390034


namespace NUMINAMATH_CALUDE_handshake_count_l3900_390057

theorem handshake_count (n : ℕ) (h : n = 11) : 
  (n * (n - 1)) / 2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_handshake_count_l3900_390057


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3900_390083

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (5 * x₁^2 + 8 * x₁ - 7 = 0) → 
  (5 * x₂^2 + 8 * x₂ - 7 = 0) → 
  (x₁ ≠ x₂) →
  (x₁^2 + x₂^2 = 134/25) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3900_390083


namespace NUMINAMATH_CALUDE_units_digit_of_product_l3900_390095

theorem units_digit_of_product (a b c : ℕ) : (4^1001 * 8^1002 * 12^1003) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l3900_390095


namespace NUMINAMATH_CALUDE_smallest_sum_of_primes_with_digit_conditions_l3900_390058

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def digit_count (n : ℕ) (d : ℕ) : ℕ := 
  (n.digits 10).count d

def satisfies_conditions (primes : List ℕ) : Prop :=
  primes.length = 5 ∧
  (∀ p ∈ primes, is_prime p) ∧
  (primes.map (digit_count · 3)).sum = 2 ∧
  (primes.map (digit_count · 7)).sum = 2 ∧
  (primes.map (digit_count · 8)).sum = 2 ∧
  (∀ d ∈ [1, 2, 4, 5, 6, 9], (primes.map (digit_count · d)).sum = 1)

theorem smallest_sum_of_primes_with_digit_conditions :
  ∃ (primes : List ℕ),
    satisfies_conditions primes ∧
    primes.sum = 2063 ∧
    (∀ other_primes : List ℕ, satisfies_conditions other_primes → other_primes.sum ≥ 2063) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_primes_with_digit_conditions_l3900_390058


namespace NUMINAMATH_CALUDE_curve_constants_sum_l3900_390081

/-- Given a curve y = ax² + b/x passing through the point (2, -5) with a tangent at this point
    parallel to the line 7x + 2y + 3 = 0, prove that a + b = -43/20 -/
theorem curve_constants_sum (a b : ℝ) : 
  (4 * a + b / 2 = -5) →  -- Curve passes through (2, -5)
  (4 * a - b / 4 = -7/2) →  -- Tangent at (2, -5) is parallel to 7x + 2y + 3 = 0
  a + b = -43/20 := by
  sorry

end NUMINAMATH_CALUDE_curve_constants_sum_l3900_390081


namespace NUMINAMATH_CALUDE_warden_citations_l3900_390079

/-- The total number of citations issued by a park warden -/
theorem warden_citations (littering : ℕ) (off_leash : ℕ) (parking : ℕ) 
  (h1 : littering = off_leash)
  (h2 : parking = 2 * littering)
  (h3 : littering = 4) : 
  littering + off_leash + parking = 16 := by
  sorry

end NUMINAMATH_CALUDE_warden_citations_l3900_390079


namespace NUMINAMATH_CALUDE_quadrilateral_and_triangle_theorem_l3900_390032

/-- Represents a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Intersection point of two lines -/
def intersect (l₁ l₂ : Line) : Point := sorry

/-- Line passing through two points -/
def line_through (p q : Point) : Line := sorry

/-- Point where a line parallel to a given direction through a point intersects another line -/
def parallel_intersect (p : Point) (l : Line) (dir : ℝ) : Point := sorry

/-- Check if three points are collinear -/
def collinear (p q r : Point) : Prop := sorry

/-- Check if two triangles are perspective -/
def perspective (t₁ t₂ : Point × Point × Point) : Prop := sorry

/-- The main theorem -/
theorem quadrilateral_and_triangle_theorem 
  (A B C D E F : Point) 
  (dir₁ dir₂ : ℝ) :
  let EF := line_through E F
  let A₁ := parallel_intersect A EF dir₁
  let B₁ := parallel_intersect B EF dir₁
  let C₁ := parallel_intersect C EF dir₁
  let D₁ := parallel_intersect D EF dir₁
  let desargues_line := sorry -- Definition of Desargues line
  let A' := parallel_intersect A desargues_line dir₂
  let B' := parallel_intersect B desargues_line dir₂
  let C' := parallel_intersect C desargues_line dir₂
  let A₁' := parallel_intersect A₁ desargues_line dir₂
  let B₁' := parallel_intersect B₁ desargues_line dir₂
  let C₁' := parallel_intersect C₁ desargues_line dir₂
  collinear E F (intersect (line_through A C) (line_through B D)) ∧
  perspective (A, B, C) (A₁, B₁, C₁) →
  (1 / distance A A₁ + 1 / distance C C₁ = 1 / distance B B₁ + 1 / distance D D₁) ∧
  (1 / distance A A' + 1 / distance B B' + 1 / distance C C' = 
   1 / distance A₁ A₁' + 1 / distance B₁ B₁' + 1 / distance C₁ C₁') := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_and_triangle_theorem_l3900_390032


namespace NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l3900_390089

theorem a_equals_one_sufficient_not_necessary_for_abs_a_equals_one :
  (∃ a : ℝ, a = 1 → abs a = 1) ∧
  (∃ a : ℝ, a ≠ 1 ∧ abs a = 1) := by
  sorry

end NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l3900_390089


namespace NUMINAMATH_CALUDE_four_distinct_solutions_l3900_390008

theorem four_distinct_solutions (p q : ℝ) :
  (∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    (x₁^2 + p * |x₁| = q * x₁ - 1) ∧
    (x₂^2 + p * |x₂| = q * x₂ - 1) ∧
    (x₃^2 + p * |x₃| = q * x₃ - 1) ∧
    (x₄^2 + p * |x₄| = q * x₄ - 1)) ↔
  (p + |q| + 2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_four_distinct_solutions_l3900_390008


namespace NUMINAMATH_CALUDE_triangle_existence_l3900_390099

theorem triangle_existence (n : ℕ) (points : Finset (ℝ × ℝ × ℝ)) (segments : Finset ((ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ))) :
  points.card = 2 * n →
  segments.card = n^2 + 1 →
  ∃ (a b c : ℝ × ℝ × ℝ), 
    a ∈ points ∧ b ∈ points ∧ c ∈ points ∧
    (a, b) ∈ segments ∧ (b, c) ∈ segments ∧ (a, c) ∈ segments :=
by sorry

end NUMINAMATH_CALUDE_triangle_existence_l3900_390099


namespace NUMINAMATH_CALUDE_pie_slices_l3900_390024

/-- Proves that if 3/4 of a pie is given away and 2 slices are left, then the pie was sliced into 8 pieces. -/
theorem pie_slices (total_slices : ℕ) : 
  (3 : ℚ) / 4 * total_slices + 2 = total_slices → total_slices = 8 := by
  sorry

#check pie_slices

end NUMINAMATH_CALUDE_pie_slices_l3900_390024


namespace NUMINAMATH_CALUDE_store_turnover_equation_l3900_390050

/-- Represents the equation for the total turnover in the first quarter of a store,
    given an initial turnover and a monthly growth rate. -/
theorem store_turnover_equation (initial_turnover : ℝ) (growth_rate : ℝ) :
  initial_turnover = 50 →
  initial_turnover * (1 + (1 + growth_rate) + (1 + growth_rate)^2) = 600 :=
by sorry

end NUMINAMATH_CALUDE_store_turnover_equation_l3900_390050


namespace NUMINAMATH_CALUDE_weight_loss_calculation_l3900_390087

theorem weight_loss_calculation (W : ℝ) (x : ℝ) : 
  W * (1 - x / 100 + 2 / 100) = W * (100 - 10.24) / 100 → x = 12.24 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_calculation_l3900_390087


namespace NUMINAMATH_CALUDE_odd_number_multiple_square_differences_l3900_390055

theorem odd_number_multiple_square_differences : ∃ (n : ℕ), 
  Odd n ∧ (∃ (a b c d : ℕ), a ≠ c ∧ b ≠ d ∧ n = a^2 - b^2 ∧ n = c^2 - d^2) := by
  sorry

end NUMINAMATH_CALUDE_odd_number_multiple_square_differences_l3900_390055


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l3900_390066

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_sequence_third_term (a : ℕ → ℝ) :
  arithmetic_sequence a → a 1 = 2 → a 5 + a 7 = 2 * a 4 + 4 → a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l3900_390066


namespace NUMINAMATH_CALUDE_triangle_classification_l3900_390071

theorem triangle_classification (a b c : ℝ) (h : (b^2 + a^2) * (b - a) = b * c^2 - a * c^2) :
  a = b ∨ a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_classification_l3900_390071


namespace NUMINAMATH_CALUDE_qi_winning_probability_l3900_390067

-- Define the horse strengths
structure HorseStrengths where
  tian_top_better_than_qi_middle : Prop
  tian_top_worse_than_qi_top : Prop
  tian_middle_better_than_qi_bottom : Prop
  tian_middle_worse_than_qi_middle : Prop
  tian_bottom_worse_than_qi_bottom : Prop

-- Define the probability of Qi's horse winning
def probability_qi_wins (strengths : HorseStrengths) : ℚ := 2/3

-- Theorem statement
theorem qi_winning_probability (strengths : HorseStrengths) :
  probability_qi_wins strengths = 2/3 := by sorry

end NUMINAMATH_CALUDE_qi_winning_probability_l3900_390067


namespace NUMINAMATH_CALUDE_bridge_length_proof_l3900_390001

/-- Calculate the distance traveled with constant acceleration -/
def distance_traveled (initial_velocity : ℝ) (acceleration : ℝ) (time : ℝ) : ℝ :=
  initial_velocity * time + 0.5 * acceleration * time^2

/-- Convert kilometers to meters -/
def km_to_meters (km : ℝ) : ℝ := km * 1000

theorem bridge_length_proof (initial_velocity : ℝ) (acceleration : ℝ) (time : ℝ) 
  (h1 : initial_velocity = 3)
  (h2 : acceleration = 0.2)
  (h3 : time = 0.25) :
  km_to_meters (distance_traveled initial_velocity acceleration time) = 756.25 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_proof_l3900_390001


namespace NUMINAMATH_CALUDE_hundred_from_twos_l3900_390013

theorem hundred_from_twos : (222 / 2) - (22 / 2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_hundred_from_twos_l3900_390013


namespace NUMINAMATH_CALUDE_foil_covered_prism_width_l3900_390049

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism -/
def volume (d : PrismDimensions) : ℝ := d.length * d.width * d.height

/-- Represents the inner dimensions of the prism (not covered by foil) -/
def inner_prism : PrismDimensions :=
  { length := 4,
    width := 8,
    height := 4 }

/-- Represents the outer dimensions of the prism (covered by foil) -/
def outer_prism : PrismDimensions :=
  { length := inner_prism.length + 2,
    width := inner_prism.width + 2,
    height := inner_prism.height + 2 }

/-- The main theorem to prove -/
theorem foil_covered_prism_width :
  (volume inner_prism = 128) →
  (inner_prism.width = 2 * inner_prism.length) →
  (inner_prism.width = 2 * inner_prism.height) →
  (outer_prism.width = 10) := by
  sorry

end NUMINAMATH_CALUDE_foil_covered_prism_width_l3900_390049


namespace NUMINAMATH_CALUDE_number_ratio_l3900_390060

theorem number_ratio (x : ℝ) (h : 3 * (2 * x + 9) = 63) : x / (2 * x) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_l3900_390060


namespace NUMINAMATH_CALUDE_intercept_sum_l3900_390098

/-- The line equation 2x - y + 4 = 0 -/
def line_equation (x y : ℝ) : Prop := 2 * x - y + 4 = 0

/-- The x-intercept of the line -/
def x_intercept : ℝ := -2

/-- The y-intercept of the line -/
def y_intercept : ℝ := 4

/-- Theorem: The sum of the x-intercept and y-intercept of the line 2x - y + 4 = 0 is equal to 2 -/
theorem intercept_sum : x_intercept + y_intercept = 2 := by
  sorry

end NUMINAMATH_CALUDE_intercept_sum_l3900_390098


namespace NUMINAMATH_CALUDE_binary_110110_is_54_l3900_390084

def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_110110_is_54 :
  binary_to_decimal [true, true, false, true, true, false] = 54 := by
  sorry

end NUMINAMATH_CALUDE_binary_110110_is_54_l3900_390084


namespace NUMINAMATH_CALUDE_min_value_expression_l3900_390026

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 + a^2 * b^2) / (a * b) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3900_390026


namespace NUMINAMATH_CALUDE_lassis_from_ten_mangoes_l3900_390021

/-- A recipe for making lassis from mangoes -/
structure Recipe where
  mangoes : ℕ
  lassis : ℕ

/-- Given a recipe and a number of mangoes, calculate the number of lassis that can be made -/
def makeLassis (recipe : Recipe) (numMangoes : ℕ) : ℕ :=
  (recipe.lassis * numMangoes) / recipe.mangoes

theorem lassis_from_ten_mangoes (recipe : Recipe) 
  (h1 : recipe.mangoes = 3) 
  (h2 : recipe.lassis = 15) : 
  makeLassis recipe 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_lassis_from_ten_mangoes_l3900_390021


namespace NUMINAMATH_CALUDE_town_population_growth_l3900_390020

theorem town_population_growth (r : ℕ) (h1 : r^3 + 200 = (r + 1)^3 + 27) 
  (h2 : (r + 1)^3 + 300 = (r + 1)^3) : 
  (((r + 1)^3 - r^3) * 100 : ℚ) / r^3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_town_population_growth_l3900_390020


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l3900_390004

theorem nested_fraction_evaluation :
  (2 : ℚ) / (2 + 1 / (3 + 1 / 4)) = 13 / 15 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l3900_390004


namespace NUMINAMATH_CALUDE_turtle_ratio_l3900_390006

theorem turtle_ratio (total : ℕ) (green : ℕ) (h1 : total = 3200) (h2 : green = 800) :
  (total - green) / green = 3 := by
  sorry

end NUMINAMATH_CALUDE_turtle_ratio_l3900_390006


namespace NUMINAMATH_CALUDE_exactly_two_true_with_converse_l3900_390037

-- Define the propositions
def proposition1 (a b : ℝ) : Prop := (a / b < 1) → (a < b)
def proposition2 (sides : Fin 4 → ℝ) : Prop := ∀ i j, sides i = sides j
def proposition3 (angles : Fin 3 → ℝ) : Prop := angles 0 = angles 1

-- Define the converses
def converse1 (a b : ℝ) : Prop := (a < b) → (a / b < 1)
def converse2 (sides : Fin 4 → ℝ) : Prop := (∀ i j, sides i = sides j) → (∃ r : ℝ, ∀ i, sides i = r)
def converse3 (angles : Fin 3 → ℝ) : Prop := (angles 0 = angles 1) → (∃ s : ℝ, angles 0 = s ∧ angles 1 = s)

-- Theorem statement
theorem exactly_two_true_with_converse :
  ∃! n : ℕ, n = 2 ∧
  (∀ a b : ℝ, proposition1 a b ∧ converse1 a b) ∨
  (∀ sides : Fin 4 → ℝ, proposition2 sides ∧ converse2 sides) ∨
  (∀ angles : Fin 3 → ℝ, proposition3 angles ∧ converse3 angles) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_true_with_converse_l3900_390037


namespace NUMINAMATH_CALUDE_zekes_estimate_l3900_390070

theorem zekes_estimate (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0) (h : x > 2*y) :
  (x + k) - 2*(y + k) < x - 2*y := by
sorry

end NUMINAMATH_CALUDE_zekes_estimate_l3900_390070


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3900_390016

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    focal length 2√6, and an asymptote l such that the distance from (1,0) to l is √6/3,
    prove that the equation of the hyperbola is x²/2 - y²/4 = 1. -/
theorem hyperbola_equation (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
  (h_focal : Real.sqrt (a^2 + b^2) = Real.sqrt 6)
  (h_asymptote : ∃ (k : ℝ), k * a = b ∧ k * b = a)
  (h_distance : (b / Real.sqrt (a^2 + b^2)) = Real.sqrt 6 / 3) :
  a^2 = 2 ∧ b^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3900_390016


namespace NUMINAMATH_CALUDE_hcf_of_8_and_12_l3900_390047

theorem hcf_of_8_and_12 :
  let a : ℕ := 8
  let b : ℕ := 12
  Nat.lcm a b = 24 →
  Nat.gcd a b = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_hcf_of_8_and_12_l3900_390047


namespace NUMINAMATH_CALUDE_total_streets_patrolled_in_one_hour_l3900_390046

/-- Represents the patrol rate of a police officer -/
structure PatrolRate where
  streets : ℕ
  hours : ℕ

/-- Calculates the number of streets patrolled per hour -/
def streetsPerHour (rate : PatrolRate) : ℚ :=
  rate.streets / rate.hours

/-- The patrol rates of three officers -/
def officerA : PatrolRate := { streets := 36, hours := 4 }
def officerB : PatrolRate := { streets := 55, hours := 5 }
def officerC : PatrolRate := { streets := 42, hours := 6 }

/-- The total number of streets patrolled by all three officers in one hour -/
def totalStreetsPerHour : ℚ :=
  streetsPerHour officerA + streetsPerHour officerB + streetsPerHour officerC

theorem total_streets_patrolled_in_one_hour :
  totalStreetsPerHour = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_streets_patrolled_in_one_hour_l3900_390046


namespace NUMINAMATH_CALUDE_inequality_solution_l3900_390023

theorem inequality_solution (x : ℝ) : 
  (-1 ≤ (x^2 + 3*x - 1) / (4 - x^2) ∧ (x^2 + 3*x - 1) / (4 - x^2) < 1) ↔ 
  (x < -5/2 ∨ (-1 ≤ x ∧ x < 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3900_390023


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l3900_390053

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the problem
theorem triangle_angle_proof (t : Triangle) (m n : ℝ × ℝ) :
  m = (t.a + t.c, -t.b) →
  n = (t.a - t.c, t.b) →
  m.1 * n.1 + m.2 * n.2 = t.b * t.c →
  0 < t.A →
  t.A < π →
  t.A = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l3900_390053


namespace NUMINAMATH_CALUDE_hyperbola_C_different_asymptote_l3900_390088

-- Define the hyperbolas
def hyperbola_A (x y : ℝ) : Prop := x^2 / 9 - y^2 / 4 = 1
def hyperbola_B (x y : ℝ) : Prop := y^2 / 4 - x^2 / 9 = 1
def hyperbola_C (x y : ℝ) : Prop := x^2 / 4 - y^2 / 9 = 1
def hyperbola_D (x y : ℝ) : Prop := y^2 / 12 - x^2 / 27 = 1

-- Define the asymptote
def is_asymptote (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), ∀ (x y : ℝ), f x y → (2 * x = 3 * y ∨ 2 * x = -3 * y)

-- Theorem statement
theorem hyperbola_C_different_asymptote :
  ¬(is_asymptote hyperbola_C) ∧
  (is_asymptote hyperbola_A) ∧
  (is_asymptote hyperbola_B) ∧
  (is_asymptote hyperbola_D) := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_C_different_asymptote_l3900_390088


namespace NUMINAMATH_CALUDE_max_intersections_theorem_l3900_390068

/-- Represents a convex polygon in a plane -/
structure ConvexPolygon where
  sides : ℕ
  convex : sides ≥ 3

/-- Represents the configuration of two convex polygons -/
structure TwoPolygons where
  P₁ : ConvexPolygon
  P₂ : ConvexPolygon
  sameplane : True  -- Represents that P₁ and P₂ are on the same plane
  no_overlap : True  -- Represents that P₁ and P₂ do not have overlapping line segments
  size_order : P₁.sides ≤ P₂.sides

/-- The function that calculates the maximum number of intersection points -/
def max_intersections (tp : TwoPolygons) : ℕ := 2 * tp.P₁.sides

/-- The theorem stating the maximum number of intersection points -/
theorem max_intersections_theorem (tp : TwoPolygons) : 
  max_intersections tp = 2 * tp.P₁.sides := by sorry

end NUMINAMATH_CALUDE_max_intersections_theorem_l3900_390068


namespace NUMINAMATH_CALUDE_vector_simplification_l3900_390051

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_simplification (a b : V) : 
  2 • (a + b) - a = a + 2 • b := by sorry

end NUMINAMATH_CALUDE_vector_simplification_l3900_390051


namespace NUMINAMATH_CALUDE_chinese_english_total_score_l3900_390077

theorem chinese_english_total_score 
  (average_score : ℝ) 
  (math_score : ℝ) 
  (num_subjects : ℕ) 
  (h1 : average_score = 97) 
  (h2 : math_score = 100) 
  (h3 : num_subjects = 3) :
  average_score * num_subjects - math_score = 191 :=
by sorry

end NUMINAMATH_CALUDE_chinese_english_total_score_l3900_390077


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3900_390018

theorem inequality_equivalence (x : ℝ) : -1/2 * x + 3 < 0 ↔ x > 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3900_390018


namespace NUMINAMATH_CALUDE_moles_of_CH3Cl_formed_l3900_390065

-- Define the chemical reaction
structure Reaction where
  reactant1 : String
  reactant2 : String
  product1 : String
  product2 : String

-- Define the available moles
def available_CH4 : ℝ := 1
def available_Cl2 : ℝ := 1

-- Define the reaction
def methane_chlorine_reaction : Reaction :=
  { reactant1 := "CH4"
  , reactant2 := "Cl2"
  , product1 := "CH3Cl"
  , product2 := "HCl" }

-- Theorem statement
theorem moles_of_CH3Cl_formed (reaction : Reaction) 
  (h1 : reaction = methane_chlorine_reaction)
  (h2 : available_CH4 = 1)
  (h3 : available_Cl2 = 1) :
  ∃ (moles_CH3Cl : ℝ), moles_CH3Cl = 1 :=
sorry

end NUMINAMATH_CALUDE_moles_of_CH3Cl_formed_l3900_390065


namespace NUMINAMATH_CALUDE_hainan_scientific_notation_l3900_390048

theorem hainan_scientific_notation :
  48500000 = 4.85 * (10 ^ 7) := by
  sorry

end NUMINAMATH_CALUDE_hainan_scientific_notation_l3900_390048


namespace NUMINAMATH_CALUDE_pascal_39th_number_40th_row_l3900_390072

-- Define Pascal's triangle coefficient
def pascal (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement
theorem pascal_39th_number_40th_row : pascal 40 38 = 780 := by
  sorry

end NUMINAMATH_CALUDE_pascal_39th_number_40th_row_l3900_390072


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_l3900_390061

theorem quadratic_vertex_form (a b c : ℝ) (h k : ℝ) :
  (∀ x, a * x^2 + b * x + c = a * (x - h)^2 + k) →
  (a = 3 ∧ b = 9 ∧ c = 20) →
  h = -1.5 := by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_l3900_390061


namespace NUMINAMATH_CALUDE_max_length_complex_l3900_390043

theorem max_length_complex (ω : ℂ) (h : Complex.abs ω = 1) :
  ∃ (max : ℝ), max = 108 ∧ ∀ (z : ℂ), Complex.abs ((ω + 2)^3 * (ω - 3)^2) ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_length_complex_l3900_390043


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_six_l3900_390003

theorem sum_of_roots_equals_six : 
  let f (x : ℝ) := (x^3 - 3*x^2 - 9*x) / (x + 3)
  ∃ (a b : ℝ), (f a = 9 ∧ f b = 9 ∧ a ≠ b) ∧ a + b = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_six_l3900_390003


namespace NUMINAMATH_CALUDE_order_of_expressions_l3900_390028

theorem order_of_expressions :
  let a : ℝ := 3^(3/2)
  let b : ℝ := 3^(5/2)
  let c : ℝ := Real.log 3 / Real.log 0.5
  c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_order_of_expressions_l3900_390028


namespace NUMINAMATH_CALUDE_abs_le_two_necessary_not_sufficient_for_zero_le_x_le_two_l3900_390078

theorem abs_le_two_necessary_not_sufficient_for_zero_le_x_le_two :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → |x| ≤ 2) ∧
  ¬(∀ x : ℝ, |x| ≤ 2 → 0 ≤ x ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_abs_le_two_necessary_not_sufficient_for_zero_le_x_le_two_l3900_390078


namespace NUMINAMATH_CALUDE_quadratic_roots_real_l3900_390007

theorem quadratic_roots_real (a b c : ℝ) : 
  let discriminant := 4 * (b^2 + c^2)
  discriminant ≥ 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_real_l3900_390007


namespace NUMINAMATH_CALUDE_product_of_sum_of_roots_l3900_390011

theorem product_of_sum_of_roots (x : ℝ) :
  (Real.sqrt (8 + x) + Real.sqrt (15 - x) = 6) →
  (8 + x) * (15 - x) = 169 / 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_of_roots_l3900_390011


namespace NUMINAMATH_CALUDE_interval_for_quadratic_function_l3900_390092

/-- The function f(x) = -x^2 -/
def f (x : ℝ) : ℝ := -x^2

theorem interval_for_quadratic_function (a b : ℝ) :
  (∀ x ∈ Set.Icc a b, f x ≥ 2*a) ∧  -- minimum value condition
  (∀ x ∈ Set.Icc a b, f x ≤ 2*b) ∧  -- maximum value condition
  (∃ x ∈ Set.Icc a b, f x = 2*a) ∧  -- minimum value is achieved
  (∃ x ∈ Set.Icc a b, f x = 2*b) →  -- maximum value is achieved
  a = 1 ∧ b = 3 :=
by sorry

end NUMINAMATH_CALUDE_interval_for_quadratic_function_l3900_390092


namespace NUMINAMATH_CALUDE_platform_length_platform_length_proof_l3900_390075

/-- Calculates the length of a platform given the length of a train, its speed, and the time it takes to cross the platform. -/
theorem platform_length 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_length = 160) 
  (h2 : train_speed_kmph = 72) 
  (h3 : crossing_time = 25) : ℝ :=
let train_speed_mps := train_speed_kmph * (1000 / 3600)
let total_distance := train_speed_mps * crossing_time
let platform_length := total_distance - train_length
340

theorem platform_length_proof : platform_length 160 72 25 rfl rfl rfl = 340 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_platform_length_proof_l3900_390075


namespace NUMINAMATH_CALUDE_pressure_valve_problem_l3900_390064

/-- Represents the constant ratio between pressure change and temperature -/
def k : ℚ := (5 * 4 - 6) / (10 + 20)

/-- The pressure-temperature relationship function -/
def pressure_temp_relation (x t : ℚ) : Prop :=
  (5 * x - 6) / (t + 20) = k

theorem pressure_valve_problem :
  pressure_temp_relation 4 10 →
  pressure_temp_relation (34/5) 40 :=
by sorry

end NUMINAMATH_CALUDE_pressure_valve_problem_l3900_390064


namespace NUMINAMATH_CALUDE_movie_production_profit_l3900_390010

def movie_production (main_actor_fee supporting_actor_fee extra_fee : ℕ)
                     (main_actor_food supporting_actor_food crew_food : ℕ)
                     (post_production_cost revenue : ℕ) : Prop :=
  let num_main_actors : ℕ := 2
  let num_supporting_actors : ℕ := 3
  let num_extra : ℕ := 1
  let total_people : ℕ := 50
  let actor_fees := num_main_actors * main_actor_fee + 
                    num_supporting_actors * supporting_actor_fee + 
                    num_extra * extra_fee
  let food_cost := num_main_actors * main_actor_food + 
                   (num_supporting_actors + num_extra) * supporting_actor_food + 
                   (total_people - num_main_actors - num_supporting_actors - num_extra) * crew_food
  let equipment_cost := 2 * (actor_fees + food_cost)
  let total_cost := actor_fees + food_cost + equipment_cost + post_production_cost
  let profit := revenue - total_cost
  profit = 4584

theorem movie_production_profit :
  movie_production 500 100 50 10 5 3 850 10000 := by
  sorry

end NUMINAMATH_CALUDE_movie_production_profit_l3900_390010


namespace NUMINAMATH_CALUDE_new_speed_calculation_l3900_390052

theorem new_speed_calculation (distance : ℝ) (original_time : ℝ) (time_factor : ℝ) 
  (h1 : distance = 252)
  (h2 : original_time = 6)
  (h3 : time_factor = 3/2) :
  let new_time := original_time * time_factor
  let new_speed := distance / new_time
  new_speed = 28 := by sorry

end NUMINAMATH_CALUDE_new_speed_calculation_l3900_390052


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3900_390019

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℚ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = 4) :
  ∃ d : ℚ, d = 3/2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3900_390019


namespace NUMINAMATH_CALUDE_abc_solution_l3900_390038

/-- Converts a base 7 number to its decimal representation -/
def toDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to its base 7 representation -/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Represents a two-digit number in base 7 -/
def twoDigitBase7 (tens : ℕ) (ones : ℕ) : ℕ := 7 * tens + ones

/-- Represents a three-digit number in base 7 -/
def threeDigitBase7 (hundreds : ℕ) (tens : ℕ) (ones : ℕ) : ℕ := 49 * hundreds + 7 * tens + ones

theorem abc_solution (A B C : ℕ) : 
  (A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0) →  -- non-zero digits
  (A < 7 ∧ B < 7 ∧ C < 7) →  -- less than 7
  (A ≠ B ∧ B ≠ C ∧ A ≠ C) →  -- distinct digits
  (twoDigitBase7 A B + C = twoDigitBase7 C 0) →  -- AB₇ + C₇ = C0₇
  (twoDigitBase7 A B + twoDigitBase7 B A = twoDigitBase7 C C) →  -- AB₇ + BA₇ = CC₇
  threeDigitBase7 A B C = 643  -- ABC = 643 in base 7
  := by sorry


end NUMINAMATH_CALUDE_abc_solution_l3900_390038


namespace NUMINAMATH_CALUDE_units_digit_of_composite_product_l3900_390044

def first_four_composites : List Nat := [4, 6, 8, 9]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (· * ·) 1

def units_digit (n : Nat) : Nat :=
  n % 10

theorem units_digit_of_composite_product :
  units_digit (product_of_list first_four_composites) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_composite_product_l3900_390044
