import Mathlib

namespace NUMINAMATH_CALUDE_exp_25pi_i_div_2_equals_i_l2127_212713

theorem exp_25pi_i_div_2_equals_i :
  Complex.exp (25 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_exp_25pi_i_div_2_equals_i_l2127_212713


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l2127_212729

theorem modulus_of_complex_number (z : ℂ) : z = 1 + 2*I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l2127_212729


namespace NUMINAMATH_CALUDE_sin_cos_product_l2127_212732

theorem sin_cos_product (θ : Real) 
  (h : (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 2) : 
  Real.sin θ * Real.cos θ = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_product_l2127_212732


namespace NUMINAMATH_CALUDE_queue_properties_l2127_212760

/-- A queue with Slowpokes and Quickies -/
structure Queue where
  m : ℕ  -- number of Slowpokes
  n : ℕ  -- number of Quickies
  a : ℕ  -- time taken by Quickies
  b : ℕ  -- time taken by Slowpokes

/-- Combinatorial choose function -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Minimum total waiting time -/
def T_min (q : Queue) : ℕ :=
  q.a * choose q.n 2 + q.a * q.m * q.n + q.b * choose q.m 2

/-- Maximum total waiting time -/
def T_max (q : Queue) : ℕ :=
  q.a * choose q.n 2 + q.b * q.m * q.n + q.b * choose q.m 2

/-- Expected total waiting time -/
def E_T (q : Queue) : ℚ :=
  (choose (q.n + q.m) 2 : ℚ) * (q.b * q.m + q.a * q.n) / (q.m + q.n)

/-- Theorem stating the properties of the queue -/
theorem queue_properties (q : Queue) :
  (T_min q ≤ T_max q) ∧
  (E_T q ≥ (T_min q : ℚ)) ∧
  (E_T q ≤ (T_max q : ℚ)) :=
sorry

end NUMINAMATH_CALUDE_queue_properties_l2127_212760


namespace NUMINAMATH_CALUDE_students_not_in_biology_l2127_212752

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) 
  (h1 : total_students = 880) 
  (h2 : biology_percentage = 275 / 1000) : 
  total_students - (total_students * biology_percentage).floor = 638 := by
  sorry

end NUMINAMATH_CALUDE_students_not_in_biology_l2127_212752


namespace NUMINAMATH_CALUDE_remainder_problem_l2127_212777

theorem remainder_problem (n m p : ℕ) 
  (hn : n % 4 = 3)
  (hm : m % 7 = 5)
  (hp : p % 5 = 2) :
  (7 * n + 3 * m - p) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2127_212777


namespace NUMINAMATH_CALUDE_derivative_at_two_l2127_212779

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 2*x + 1

theorem derivative_at_two :
  (deriv f) 2 = 2 := by sorry

end NUMINAMATH_CALUDE_derivative_at_two_l2127_212779


namespace NUMINAMATH_CALUDE_trig_identity_l2127_212739

/-- Prove that (cos 70° * cos 20°) / (1 - 2 * sin² 25°) = 1/2 -/
theorem trig_identity : 
  (Real.cos (70 * π / 180) * Real.cos (20 * π / 180)) / 
  (1 - 2 * Real.sin (25 * π / 180) ^ 2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2127_212739


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2127_212763

/-- Given two vectors a and b in R², where a is perpendicular to (a - b),
    prove that the y-coordinate of b must be 3. -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h : a.1 = 1 ∧ a.2 = 2 ∧ b.1 = -1) :
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0) → b.2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2127_212763


namespace NUMINAMATH_CALUDE_three_legged_dogs_carly_three_legged_dogs_l2127_212709

theorem three_legged_dogs (total_nails : ℕ) (total_dogs : ℕ) : ℕ :=
  let nails_per_paw := 4
  let paws_per_dog := 4
  let nails_per_dog := nails_per_paw * paws_per_dog
  let expected_total_nails := total_dogs * nails_per_dog
  let missing_nails := expected_total_nails - total_nails
  missing_nails / nails_per_paw

theorem carly_three_legged_dogs :
  three_legged_dogs 164 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_legged_dogs_carly_three_legged_dogs_l2127_212709


namespace NUMINAMATH_CALUDE_box_volume_calculation_l2127_212762

/-- Given a rectangular metallic sheet and squares cut from each corner, calculate the volume of the resulting box. -/
theorem box_volume_calculation (sheet_length sheet_width cut_square_side : ℝ) 
  (h1 : sheet_length = 48)
  (h2 : sheet_width = 36)
  (h3 : cut_square_side = 4) :
  (sheet_length - 2 * cut_square_side) * (sheet_width - 2 * cut_square_side) * cut_square_side = 4480 := by
  sorry

#check box_volume_calculation

end NUMINAMATH_CALUDE_box_volume_calculation_l2127_212762


namespace NUMINAMATH_CALUDE_dishwasher_manager_wage_ratio_l2127_212736

/-- Proves that the ratio of dishwasher's wage to manager's wage is 0.5 -/
theorem dishwasher_manager_wage_ratio :
  ∀ (dishwasher_wage chef_wage manager_wage : ℝ),
  manager_wage = 7.5 →
  chef_wage = manager_wage - 3 →
  chef_wage = dishwasher_wage * 1.2 →
  dishwasher_wage / manager_wage = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_dishwasher_manager_wage_ratio_l2127_212736


namespace NUMINAMATH_CALUDE_mighty_l_league_teams_l2127_212716

/-- The number of teams in the league -/
def n : ℕ := 8

/-- The total number of games played -/
def total_games : ℕ := 28

/-- Formula for the number of games in a round-robin tournament -/
def games (x : ℕ) : ℕ := x * (x - 1) / 2

theorem mighty_l_league_teams :
  (n ≥ 2) ∧ (games n = total_games) := by sorry

end NUMINAMATH_CALUDE_mighty_l_league_teams_l2127_212716


namespace NUMINAMATH_CALUDE_expression_evaluation_l2127_212791

theorem expression_evaluation (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxy : x > y) (hyz : y > z) :
  (x^z * y^x * z^y) / (z^z * y^y * x^x) = x^(z-x) * y^(x-y) * z^(y-z) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2127_212791


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2127_212726

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 2 + a 4 + a 9 + a 11 = 32 →
  a 6 + a 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2127_212726


namespace NUMINAMATH_CALUDE_equation_solutions_l2127_212773

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 6*x + 1 = 0 ↔ x = 3 + 2*Real.sqrt 2 ∨ x = 3 - 2*Real.sqrt 2) ∧
  (∀ x : ℝ, (2*x - 3)^2 = 5*(2*x - 3) ↔ x = 3/2 ∨ x = 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2127_212773


namespace NUMINAMATH_CALUDE_simplify_fraction_l2127_212714

theorem simplify_fraction : (120 : ℚ) / 180 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2127_212714


namespace NUMINAMATH_CALUDE_coat_drive_l2127_212759

theorem coat_drive (total_coats : ℕ) (high_school_coats : ℕ) (elementary_coats : ℕ) :
  total_coats = 9437 →
  high_school_coats = 6922 →
  elementary_coats = total_coats - high_school_coats →
  elementary_coats = 2515 := by
  sorry

end NUMINAMATH_CALUDE_coat_drive_l2127_212759


namespace NUMINAMATH_CALUDE_batch_size_proof_l2127_212735

/-- The number of days it takes person A to complete the batch alone -/
def days_a : ℕ := 10

/-- The number of days it takes person B to complete the batch alone -/
def days_b : ℕ := 12

/-- The difference in parts processed by person A and B after working together for 1 day -/
def difference : ℕ := 40

/-- The total number of parts in the batch -/
def total_parts : ℕ := 2400

theorem batch_size_proof :
  (1 / days_a - 1 / days_b : ℚ) * total_parts = difference := by
  sorry

end NUMINAMATH_CALUDE_batch_size_proof_l2127_212735


namespace NUMINAMATH_CALUDE_area_of_inscribed_hexagon_l2127_212772

/-- The area of a regular hexagon inscribed in a circle with radius 3 units is 13.5√3 square units. -/
theorem area_of_inscribed_hexagon : 
  let r : ℝ := 3  -- radius of the circle
  let hexagon_area : ℝ := 6 * (r^2 * Real.sqrt 3 / 4)  -- area of hexagon as 6 times the area of an equilateral triangle
  hexagon_area = 13.5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_area_of_inscribed_hexagon_l2127_212772


namespace NUMINAMATH_CALUDE_functional_equation_identity_l2127_212786

open Function

theorem functional_equation_identity (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (y - f x) = f x - 2 * x + f (f y)) →
  (∀ x : ℝ, f x = x) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_identity_l2127_212786


namespace NUMINAMATH_CALUDE_digit_sum_in_base_d_l2127_212727

/-- A function to represent a two-digit number in base d -/
def two_digit_number (d a b : ℕ) : ℕ := a * d + b

/-- The problem statement -/
theorem digit_sum_in_base_d (d A B : ℕ) : 
  d > 8 →
  A < d →
  B < d →
  two_digit_number d A B + two_digit_number d A A - two_digit_number d B A = 180 →
  A + B = 10 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_in_base_d_l2127_212727


namespace NUMINAMATH_CALUDE_simplify_fraction_l2127_212767

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2127_212767


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2127_212794

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 6*p^2 + 8*p - 3 = 0 →
  q^3 - 6*q^2 + 8*q - 3 = 0 →
  r^3 - 6*r^2 + 8*r - 3 = 0 →
  p / (q*r + 2) + q / (p*r + 2) + r / (p*q + 2) = 6/5 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2127_212794


namespace NUMINAMATH_CALUDE_distance_before_gas_is_32_l2127_212796

/-- The distance driven before stopping for gas -/
def distance_before_gas (total_distance remaining_distance : ℕ) : ℕ :=
  total_distance - remaining_distance

/-- Theorem: The distance driven before stopping for gas is 32 miles -/
theorem distance_before_gas_is_32 :
  distance_before_gas 78 46 = 32 := by
  sorry

end NUMINAMATH_CALUDE_distance_before_gas_is_32_l2127_212796


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l2127_212701

/-- A quadratic polynomial that satisfies specific conditions -/
def q (x : ℚ) : ℚ := (6/5) * x^2 - (4/5) * x + 8/5

/-- Theorem stating that q satisfies the given conditions -/
theorem q_satisfies_conditions :
  q (-2) = 8 ∧ q 1 = 2 ∧ q 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_l2127_212701


namespace NUMINAMATH_CALUDE_evaluate_f_l2127_212743

/-- The function f(x) = x^3 + 3∛x -/
def f (x : ℝ) : ℝ := x^3 + 3 * (x^(1/3))

/-- Theorem stating that 3f(3) + f(27) = 19818 -/
theorem evaluate_f : 3 * f 3 + f 27 = 19818 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_f_l2127_212743


namespace NUMINAMATH_CALUDE_quadratic_sum_l2127_212725

/-- Given a quadratic equation 100x^2 + 80x - 144 = 0, rewritten as (dx + e)^2 = f,
    where d, e, and f are integers and d > 0, prove that d + e + f = 174 -/
theorem quadratic_sum (d e f : ℤ) : 
  d > 0 → 
  (∀ x, 100 * x^2 + 80 * x - 144 = 0 ↔ (d * x + e)^2 = f) →
  d + e + f = 174 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2127_212725


namespace NUMINAMATH_CALUDE_factorization_3ax2_minus_3ay2_l2127_212731

theorem factorization_3ax2_minus_3ay2 (a x y : ℝ) : 3*a*x^2 - 3*a*y^2 = 3*a*(x+y)*(x-y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_3ax2_minus_3ay2_l2127_212731


namespace NUMINAMATH_CALUDE_linear_function_unique_solution_l2127_212717

/-- A linear function is a function of the form f(x) = mx + b for some constants m and b. -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

theorem linear_function_unique_solution (f : ℝ → ℝ) 
  (h_linear : LinearFunction f) (h1 : f 2 = 1) (h2 : f (-1) = -5) :
  ∀ x, f x = 2 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_unique_solution_l2127_212717


namespace NUMINAMATH_CALUDE_days_worked_together_l2127_212746

-- Define the total work as a positive real number
variable (W : ℝ) (hW : W > 0)

-- Define the time taken by a and b together to finish the work
def time_together : ℝ := 40

-- Define the time taken by a alone to finish the work
def time_a_alone : ℝ := 12

-- Define the additional time a worked after b left
def additional_time_a : ℝ := 9

-- Define the function to calculate the work done in a given time at a given rate
def work_done (time : ℝ) (rate : ℝ) : ℝ := time * rate

-- Define the theorem to prove
theorem days_worked_together (W : ℝ) (hW : W > 0) : 
  ∃ x : ℝ, x > 0 ∧ 
    work_done x (W / time_together) + 
    work_done additional_time_a (W / time_a_alone) = W ∧
    x = 10 := by
  sorry

end NUMINAMATH_CALUDE_days_worked_together_l2127_212746


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2127_212748

theorem imaginary_part_of_z (z : ℂ) : z = ((Complex.I - 1)^2 + 4) / (Complex.I + 1) → z.im = -3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2127_212748


namespace NUMINAMATH_CALUDE_platform_length_calculation_l2127_212792

/-- Calculates the length of a platform given train parameters -/
theorem platform_length_calculation (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ) :
  train_length = 300 →
  time_platform = 39 →
  time_pole = 18 →
  ∃ platform_length : ℝ,
    (platform_length > 350.12 ∧ platform_length < 350.14) ∧
    platform_length = train_length * (time_platform / time_pole - 1) :=
by
  sorry

#check platform_length_calculation

end NUMINAMATH_CALUDE_platform_length_calculation_l2127_212792


namespace NUMINAMATH_CALUDE_initial_eggs_correct_l2127_212784

/-- The number of eggs initially in the basket -/
def initial_eggs : ℕ := 14

/-- The number of eggs remaining after a customer buys eggs -/
def remaining_eggs (n : ℕ) (eggs : ℕ) : ℕ :=
  eggs - (eggs / 2 + 1)

/-- Theorem stating that the initial number of eggs satisfies the given conditions -/
theorem initial_eggs_correct : 
  let eggs1 := remaining_eggs initial_eggs initial_eggs
  let eggs2 := remaining_eggs eggs1 eggs1
  let eggs3 := remaining_eggs eggs2 eggs2
  eggs3 = 0 := by sorry

end NUMINAMATH_CALUDE_initial_eggs_correct_l2127_212784


namespace NUMINAMATH_CALUDE_sqrt_less_than_3y_iff_l2127_212703

theorem sqrt_less_than_3y_iff (y : ℝ) (h : y > 0) : 
  Real.sqrt y < 3 * y ↔ y > 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_sqrt_less_than_3y_iff_l2127_212703


namespace NUMINAMATH_CALUDE_quadratic_roots_l2127_212728

theorem quadratic_roots (p q : ℚ) : 
  (∃ f : ℚ → ℚ, (∀ x, f x = x^2 + p*x + q) ∧ f p = 0 ∧ f q = 0) ↔ 
  ((p = 0 ∧ q = 0) ∨ (p = -1/2 ∧ q = -1/2) ∨ (p = 1 ∧ q = -2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2127_212728


namespace NUMINAMATH_CALUDE_least_clock_equivalent_hour_l2127_212774

theorem least_clock_equivalent_hour : 
  ∃ (h : ℕ), h > 6 ∧ 
             h % 12 = (h^2) % 12 ∧ 
             h % 12 = (h^3) % 12 ∧ 
             (∀ (k : ℕ), k > 6 ∧ k < h → 
               (k % 12 ≠ (k^2) % 12 ∨ k % 12 ≠ (k^3) % 12)) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_least_clock_equivalent_hour_l2127_212774


namespace NUMINAMATH_CALUDE_factorial_fraction_l2127_212737

theorem factorial_fraction (N : ℕ) :
  (Nat.factorial (N + 1)) / ((Nat.factorial (N + 2)) + (Nat.factorial N)) = 
  (N + 1) / (N^2 + 3*N + 3) := by
sorry

end NUMINAMATH_CALUDE_factorial_fraction_l2127_212737


namespace NUMINAMATH_CALUDE_swimming_pool_width_l2127_212712

/-- Represents the dimensions and area of a rectangular swimming pool -/
structure SwimmingPool where
  length : ℝ
  width : ℝ
  area : ℝ

/-- Theorem: Given a rectangular swimming pool with area 143.2 m² and length 4 m, its width is 35.8 m -/
theorem swimming_pool_width (pool : SwimmingPool) 
  (h_area : pool.area = 143.2)
  (h_length : pool.length = 4)
  (h_rectangle : pool.area = pool.length * pool.width) : 
  pool.width = 35.8 := by
  sorry

end NUMINAMATH_CALUDE_swimming_pool_width_l2127_212712


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_l2127_212721

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem f_monotone_decreasing : 
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y :=
sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_l2127_212721


namespace NUMINAMATH_CALUDE_horner_method_for_f_at_3_l2127_212708

/-- Horner's method for a polynomial of degree 4 -/
def horner_method (a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  ((((a₄ * x + a₃) * x + a₂) * x + a₁) * x + a₀)

/-- The polynomial f(x) = 2x⁴ - x³ + 3x² + 7 -/
def f (x : ℝ) : ℝ := 2 * x^4 - x^3 + 3 * x^2 + 7

theorem horner_method_for_f_at_3 :
  horner_method 2 (-1) 3 0 7 3 = f 3 ∧ horner_method 2 (-1) 3 0 7 3 = 54 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_for_f_at_3_l2127_212708


namespace NUMINAMATH_CALUDE_sams_money_l2127_212789

/-- Given that Sam and Erica have $91 together and Erica has $53, 
    prove that Sam has $38. -/
theorem sams_money (total : ℕ) (ericas_money : ℕ) (sams_money : ℕ) : 
  total = 91 → ericas_money = 53 → sams_money = total - ericas_money → sams_money = 38 := by
  sorry

end NUMINAMATH_CALUDE_sams_money_l2127_212789


namespace NUMINAMATH_CALUDE_jerry_collection_cost_l2127_212705

/-- The amount of money Jerry needs to finish his action figure collection -/
def money_needed (current : ℕ) (total : ℕ) (cost : ℕ) : ℕ :=
  (total - current) * cost

/-- Theorem: Jerry needs $216 to finish his collection -/
theorem jerry_collection_cost :
  money_needed 9 27 12 = 216 := by
  sorry

end NUMINAMATH_CALUDE_jerry_collection_cost_l2127_212705


namespace NUMINAMATH_CALUDE_initial_cabinets_l2127_212788

theorem initial_cabinets (total : ℕ) (additional : ℕ) (counters : ℕ) : 
  total = 26 → 
  additional = 5 → 
  counters = 3 → 
  ∃ initial : ℕ, initial + counters * (2 * initial) + additional = total ∧ initial = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_cabinets_l2127_212788


namespace NUMINAMATH_CALUDE_regular_hexagon_angles_l2127_212795

/-- A regular hexagon is a polygon with 6 sides of equal length and 6 angles of equal measure. -/
structure RegularHexagon where
  -- We don't need to define any specific fields for this problem

/-- The measure of an interior angle of a regular hexagon -/
def interior_angle_measure (h : RegularHexagon) : ℝ := 120

/-- The sum of all exterior angles of a regular hexagon -/
def sum_exterior_angles (h : RegularHexagon) : ℝ := 360

theorem regular_hexagon_angles (h : RegularHexagon) : 
  (interior_angle_measure h = 120) ∧ (sum_exterior_angles h = 360) := by
  sorry

#check regular_hexagon_angles

end NUMINAMATH_CALUDE_regular_hexagon_angles_l2127_212795


namespace NUMINAMATH_CALUDE_office_average_age_l2127_212747

/-- The average age of all persons in an office, given specific conditions -/
theorem office_average_age :
  let total_persons : ℕ := 18
  let group1_size : ℕ := 5
  let group1_avg : ℚ := 14
  let group2_size : ℕ := 9
  let group2_avg : ℚ := 16
  let person15_age : ℕ := 56
  (total_persons : ℚ) * (average_age : ℚ) =
    (group1_size : ℚ) * group1_avg +
    (group2_size : ℚ) * group2_avg +
    (person15_age : ℚ) +
    ((total_persons - group1_size - group2_size - 1) : ℚ) * average_age →
  average_age = 270 / 14 := by
sorry

end NUMINAMATH_CALUDE_office_average_age_l2127_212747


namespace NUMINAMATH_CALUDE_outfit_combinations_l2127_212700

/-- Represents the number of shirts Li Fang has -/
def num_shirts : ℕ := 4

/-- Represents the number of skirts Li Fang has -/
def num_skirts : ℕ := 3

/-- Represents the number of dresses Li Fang has -/
def num_dresses : ℕ := 2

/-- Calculates the total number of outfit combinations -/
def total_outfits : ℕ := num_shirts * num_skirts + num_dresses

/-- Theorem stating that the total number of outfit combinations is 14 -/
theorem outfit_combinations : total_outfits = 14 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l2127_212700


namespace NUMINAMATH_CALUDE_tshirt_cost_l2127_212749

def initial_amount : ℕ := 91
def sweater_cost : ℕ := 24
def shoes_cost : ℕ := 11
def amount_left : ℕ := 50

theorem tshirt_cost :
  ∃ (tshirt_cost : ℕ), 
    initial_amount = sweater_cost + shoes_cost + tshirt_cost + amount_left ∧
    tshirt_cost = 6 :=
by sorry

end NUMINAMATH_CALUDE_tshirt_cost_l2127_212749


namespace NUMINAMATH_CALUDE_max_perimeter_special_triangle_l2127_212754

theorem max_perimeter_special_triangle :
  ∀ a b c : ℕ,
  (a = 4 * b) →
  (c = 20) →
  (a + b + c > a) →
  (a + b + c > b) →
  (a + b + c > c) →
  (a + b + c ≤ 50) :=
by sorry

end NUMINAMATH_CALUDE_max_perimeter_special_triangle_l2127_212754


namespace NUMINAMATH_CALUDE_furniture_cost_price_l2127_212742

theorem furniture_cost_price (final_price : ℝ) : 
  final_price = 9522.84 →
  ∃ (cost_price : ℝ),
    cost_price = 7695 ∧
    final_price = (1.12 * (0.85 * (1.3 * cost_price))) :=
by sorry

end NUMINAMATH_CALUDE_furniture_cost_price_l2127_212742


namespace NUMINAMATH_CALUDE_prize_problem_solution_l2127_212740

/-- Represents the cost and quantity of pens and notebooks --/
structure PrizeInfo where
  pen_cost : ℚ
  notebook_cost : ℚ
  total_prizes : ℕ
  max_total_cost : ℚ

/-- Theorem stating the solution to the prize problem --/
theorem prize_problem_solution (info : PrizeInfo) 
  (h1 : 2 * info.pen_cost + 3 * info.notebook_cost = 62)
  (h2 : 5 * info.pen_cost + info.notebook_cost = 90)
  (h3 : info.total_prizes = 80)
  (h4 : info.max_total_cost = 1100) :
  info.pen_cost = 16 ∧ 
  info.notebook_cost = 10 ∧ 
  (∀ m : ℕ, m * info.pen_cost + (info.total_prizes - m) * info.notebook_cost ≤ info.max_total_cost → m ≤ 50) :=
by sorry


end NUMINAMATH_CALUDE_prize_problem_solution_l2127_212740


namespace NUMINAMATH_CALUDE_initially_calculated_average_weight_l2127_212724

/-- Given a class of boys, prove that the initially calculated average weight
    is correct based on the given conditions. -/
theorem initially_calculated_average_weight
  (num_boys : ℕ)
  (correct_avg_weight : ℝ)
  (misread_weight : ℝ)
  (correct_weight : ℝ)
  (h1 : num_boys = 20)
  (h2 : correct_avg_weight = 58.6)
  (h3 : misread_weight = 56)
  (h4 : correct_weight = 60) :
  let correct_total_weight := correct_avg_weight * num_boys
  let initial_total_weight := correct_total_weight - (correct_weight - misread_weight)
  let initial_avg_weight := initial_total_weight / num_boys
  initial_avg_weight = 58.4 := by
sorry

end NUMINAMATH_CALUDE_initially_calculated_average_weight_l2127_212724


namespace NUMINAMATH_CALUDE_sum_of_digits_of_expression_l2127_212711

/-- The sum of digits of a natural number in base 10 -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The expression (10^(4n^2 + 8) + 1)^2 -/
def expression (n : ℕ) : ℕ := (10^(4*n^2 + 8) + 1)^2

theorem sum_of_digits_of_expression (n : ℕ) (h : n > 0) : 
  sumOfDigits (expression n) = 4 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_expression_l2127_212711


namespace NUMINAMATH_CALUDE_scientific_notation_218_million_l2127_212764

theorem scientific_notation_218_million :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ a ∧ a < 10 ∧ 
    218000000 = a * (10 : ℝ) ^ n ∧
    a = 2.18 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_218_million_l2127_212764


namespace NUMINAMATH_CALUDE_phi_value_for_even_shifted_function_l2127_212718

/-- Given a function f and a real number φ, proves that if f(x) = (1/2) * sin(2x + π/6)
    and f(x - φ) is an even function, then φ = -π/6 -/
theorem phi_value_for_even_shifted_function 
  (f : ℝ → ℝ) 
  (φ : ℝ) 
  (h1 : ∀ x, f x = (1/2) * Real.sin (2*x + π/6))
  (h2 : ∀ x, f (x - φ) = f (φ - x)) :
  φ = -π/6 := by
  sorry


end NUMINAMATH_CALUDE_phi_value_for_even_shifted_function_l2127_212718


namespace NUMINAMATH_CALUDE_intersection_on_y_axis_l2127_212738

/-- Proves that the intersection point of two specific polynomial graphs is on the y-axis -/
theorem intersection_on_y_axis (a b c : ℝ) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃! x y : ℝ, (a * x^2 + b^2 * x^3 + c = y) ∧ (a * x^2 - b^2 * x^3 + c = y) ∧ x = 0 ∧ y = c := by
  sorry

end NUMINAMATH_CALUDE_intersection_on_y_axis_l2127_212738


namespace NUMINAMATH_CALUDE_min_value_ab_l2127_212723

/-- Given that ab > 0 and points A(a,0), B(0,b), and C(-2,-2) are collinear, 
    the minimum value of ab is 16 -/
theorem min_value_ab (a b : ℝ) (hab : a * b > 0) 
    (hcollinear : (0 - a) * (b + 2) = (b - 0) * (0 + 2)) : 
  ∀ x y : ℝ, x * y > 0 ∧ 
    (0 - x) * (y + 2) = (y - 0) * (0 + 2) → 
    a * b ≤ x * y ∧ a * b = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_ab_l2127_212723


namespace NUMINAMATH_CALUDE_average_height_calculation_l2127_212797

theorem average_height_calculation (total_members : ℕ) (average_height : ℝ) 
  (two_member_height : ℝ) (remaining_members : ℕ) :
  total_members = 11 →
  average_height = 145.7 →
  two_member_height = 142.1 →
  remaining_members = total_members - 2 →
  (total_members * average_height - 2 * two_member_height) / remaining_members = 146.5 := by
sorry

end NUMINAMATH_CALUDE_average_height_calculation_l2127_212797


namespace NUMINAMATH_CALUDE_unique_congruence_in_range_l2127_212753

theorem unique_congruence_in_range : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 10 ∧ n ≡ 123456 [ZMOD 11] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_in_range_l2127_212753


namespace NUMINAMATH_CALUDE_fedya_deposit_l2127_212745

theorem fedya_deposit (n : ℕ) (X : ℕ) : 
  n < 30 →
  X * (100 - n) = 847 * 100 →
  X = 1100 := by
sorry

end NUMINAMATH_CALUDE_fedya_deposit_l2127_212745


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_equality_l2127_212757

theorem min_reciprocal_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z = 3) :
  (1 / x + 1 / y + 1 / z) ≥ 3 := by
  sorry

theorem min_reciprocal_sum_equality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z = 3) :
  (1 / x + 1 / y + 1 / z = 3) ↔ (x = 1 ∧ y = 1 ∧ z = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_equality_l2127_212757


namespace NUMINAMATH_CALUDE_max_y_value_l2127_212730

theorem max_y_value (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0)
  (h_eq : x^3 + y^3 = (4*x - 5*y)*y) :
  y ≤ (1/3 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_max_y_value_l2127_212730


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2127_212761

/-- Given a rectangle with length thrice its breadth and area 147,
    prove that its perimeter is 56 -/
theorem rectangle_perimeter (breadth : ℝ) (length : ℝ) :
  length = 3 * breadth →
  breadth * length = 147 →
  2 * (length + breadth) = 56 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2127_212761


namespace NUMINAMATH_CALUDE_tuning_day_method_pi_approximation_l2127_212715

/-- The "Tuning Day Method" for approximating a real number -/
def tuningDayMethod (a b c d : ℕ) : ℚ := (b + d) / (a + c)

/-- Apply the Tuning Day Method n times -/
def applyTuningDayMethod (n : ℕ) (a b c d : ℕ) : ℚ :=
  match n with
  | 0 => b / a
  | n+1 => tuningDayMethod a b c d

theorem tuning_day_method_pi_approximation :
  applyTuningDayMethod 3 10 31 5 16 = 22 / 7 := by
  sorry


end NUMINAMATH_CALUDE_tuning_day_method_pi_approximation_l2127_212715


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l2127_212776

/-- The line y = mx + 2 is tangent to the ellipse x^2 + 9y^2 = 9 if and only if m^2 = 1/3 -/
theorem line_tangent_to_ellipse (m : ℝ) :
  (∀ x y : ℝ, y = m * x + 2 ∧ x^2 + 9 * y^2 = 9 → (∃! p : ℝ × ℝ, p.1^2 + 9 * p.2^2 = 9 ∧ p.2 = m * p.1 + 2)) ↔
  m^2 = 1/3 := by
sorry


end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l2127_212776


namespace NUMINAMATH_CALUDE_simplify_expression_1_expand_expression_2_l2127_212720

-- First expression
theorem simplify_expression_1 (x y : ℝ) (h : y ≠ 0) :
  (3 * x^2 * y - 6 * x * y) / (3 * x * y) = x - 2 := by sorry

-- Second expression
theorem expand_expression_2 (a b : ℝ) :
  (a + b + 2) * (a + b - 2) = a^2 + 2*a*b + b^2 - 4 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_expand_expression_2_l2127_212720


namespace NUMINAMATH_CALUDE_projectile_height_l2127_212744

theorem projectile_height (t : ℝ) : 
  t > 0 ∧ -16 * t^2 + 60 * t = 56 ∧ 
  ∀ s, s > 0 ∧ -16 * s^2 + 60 * s = 56 → t ≤ s → 
  t = 1.75 := by
sorry

end NUMINAMATH_CALUDE_projectile_height_l2127_212744


namespace NUMINAMATH_CALUDE_some_dragons_not_breathe_fire_negates_all_dragons_breathe_fire_l2127_212719

-- Define the universe of discourse
def Dragon : Type := sorry

-- Define the property of breathing fire
def breathes_fire : Dragon → Prop := sorry

-- Theorem: "Some dragons do not breathe fire" is equivalent to 
-- the negation of "All dragons breathe fire"
theorem some_dragons_not_breathe_fire_negates_all_dragons_breathe_fire :
  (∃ d : Dragon, ¬(breathes_fire d)) ↔ ¬(∀ d : Dragon, breathes_fire d) := by
  sorry

end NUMINAMATH_CALUDE_some_dragons_not_breathe_fire_negates_all_dragons_breathe_fire_l2127_212719


namespace NUMINAMATH_CALUDE_show_end_time_l2127_212702

/-- Represents time in 24-hour format -/
structure Time where
  hour : Nat
  minute : Nat
  h_valid : hour < 24
  m_valid : minute < 60

/-- Represents a TV show -/
structure TVShow where
  start_time : Time
  end_time : Time
  weekday_only : Bool

def total_watch_time (s : TVShow) (days_watched : Nat) : Nat :=
  days_watched * (s.end_time.hour * 60 + s.end_time.minute - s.start_time.hour * 60 - s.start_time.minute)

theorem show_end_time (s : TVShow) 
  (h1 : s.start_time = ⟨14, 0, by norm_num, by norm_num⟩)
  (h2 : s.weekday_only = true)
  (h3 : total_watch_time s 4 = 120) :
  s.end_time = ⟨14, 30, by norm_num, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_show_end_time_l2127_212702


namespace NUMINAMATH_CALUDE_total_solar_systems_and_planets_l2127_212707

/-- The number of planets in the galaxy -/
def num_planets : ℕ := 20

/-- The number of additional solar systems for each planet -/
def additional_solar_systems : ℕ := 8

/-- The total number of solar systems and planets in the galaxy -/
def total_count : ℕ := num_planets * (additional_solar_systems + 1) + num_planets

theorem total_solar_systems_and_planets :
  total_count = 200 :=
by sorry

end NUMINAMATH_CALUDE_total_solar_systems_and_planets_l2127_212707


namespace NUMINAMATH_CALUDE_class_size_l2127_212750

theorem class_size (dog_video_percentage : ℚ) (dog_movie_percentage : ℚ) (dog_preference_count : ℕ) :
  dog_video_percentage = 1/2 →
  dog_movie_percentage = 1/10 →
  dog_preference_count = 18 →
  (dog_video_percentage + dog_movie_percentage) * ↑dog_preference_count / (dog_video_percentage + dog_movie_percentage) = 30 :=
by sorry

end NUMINAMATH_CALUDE_class_size_l2127_212750


namespace NUMINAMATH_CALUDE_reflection_property_l2127_212758

/-- A reflection in R² --/
structure Reflection where
  line : ℝ × ℝ  -- Vector representing the line of reflection

/-- Apply a reflection to a point --/
def apply_reflection (r : Reflection) (p : ℝ × ℝ) : ℝ × ℝ := sorry

theorem reflection_property :
  ∃ (r : Reflection),
    apply_reflection r (3, 5) = (7, 1) ∧
    apply_reflection r (2, 7) = (-7, -2) := by
  sorry

end NUMINAMATH_CALUDE_reflection_property_l2127_212758


namespace NUMINAMATH_CALUDE_range_of_k_value_of_k_with_condition_l2127_212704

-- Define the quadratic equation
def quadratic (k x : ℝ) : ℝ := x^2 + (2*k - 1)*x + k^2 - 1

-- Define the condition for two real roots
def has_two_real_roots (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic k x₁ = 0 ∧ quadratic k x₂ = 0

-- Define the condition for the sum of squares
def sum_of_squares_condition (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, quadratic k x₁ = 0 ∧ quadratic k x₂ = 0 ∧ x₁^2 + x₂^2 = 16 + x₁*x₂

-- Theorem for the range of k
theorem range_of_k :
  ∀ k : ℝ, has_two_real_roots k → k ≤ 5/4 :=
sorry

-- Theorem for the value of k when sum of squares condition is satisfied
theorem value_of_k_with_condition :
  ∀ k : ℝ, has_two_real_roots k → sum_of_squares_condition k → k = -2 :=
sorry

end NUMINAMATH_CALUDE_range_of_k_value_of_k_with_condition_l2127_212704


namespace NUMINAMATH_CALUDE_mrs_lim_revenue_l2127_212778

/-- Calculates the revenue from milk sales given the milk production and sales data --/
def milk_revenue (yesterday_morning : ℕ) (yesterday_evening : ℕ) (morning_decrease : ℕ) (remaining : ℕ) (price_per_gallon : ℚ) : ℚ :=
  let total_yesterday := yesterday_morning + yesterday_evening
  let this_morning := yesterday_morning - morning_decrease
  let total_milk := total_yesterday + this_morning
  let sold_milk := total_milk - remaining
  sold_milk * price_per_gallon

/-- Theorem stating that Mrs. Lim's revenue is $616 given the specified conditions --/
theorem mrs_lim_revenue :
  milk_revenue 68 82 18 24 (350/100) = 616 := by
  sorry

end NUMINAMATH_CALUDE_mrs_lim_revenue_l2127_212778


namespace NUMINAMATH_CALUDE_log_proportionality_l2127_212782

theorem log_proportionality (P K a b : ℝ) 
  (hP : P > 0) (hK : K > 0) (ha : a > 1) (hb : b > 1) :
  (Real.log P / Real.log a) / (Real.log P / Real.log b) = 
  (Real.log K / Real.log a) / (Real.log K / Real.log b) := by
sorry

end NUMINAMATH_CALUDE_log_proportionality_l2127_212782


namespace NUMINAMATH_CALUDE_cloth_cost_per_metre_l2127_212783

theorem cloth_cost_per_metre (total_length : Real) (total_cost : Real) 
  (h1 : total_length = 9.25)
  (h2 : total_cost = 425.50) :
  total_cost / total_length = 46 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_per_metre_l2127_212783


namespace NUMINAMATH_CALUDE_ticket_sales_proof_l2127_212785

theorem ticket_sales_proof (total_tickets : ℕ) (reduced_price_tickets : ℕ) (full_price_ratio : ℕ) :
  total_tickets = 25200 →
  reduced_price_tickets = 5400 →
  full_price_ratio = 5 →
  reduced_price_tickets + full_price_ratio * reduced_price_tickets = total_tickets →
  full_price_ratio * reduced_price_tickets = 21000 :=
by
  sorry

end NUMINAMATH_CALUDE_ticket_sales_proof_l2127_212785


namespace NUMINAMATH_CALUDE_candy_ratio_l2127_212733

theorem candy_ratio (adam james rubert : ℕ) : 
  adam = 6 →
  james = 3 * adam →
  adam + james + rubert = 96 →
  rubert = 4 * james :=
by
  sorry

end NUMINAMATH_CALUDE_candy_ratio_l2127_212733


namespace NUMINAMATH_CALUDE_equidistant_point_location_l2127_212734

-- Define a 2D point
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a quadrilateral
structure Quadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

-- Define the property of being convex
def isConvex (q : Quadrilateral) : Prop := sorry

-- Define the distance between two points
def distance (p1 p2 : Point2D) : ℝ := sorry

-- Define the property of a point being equidistant from all vertices
def isEquidistant (p : Point2D) (q : Quadrilateral) : Prop :=
  distance p q.A = distance p q.B ∧
  distance p q.A = distance p q.C ∧
  distance p q.A = distance p q.D

-- Define the property of a point being inside a quadrilateral
def isInside (p : Point2D) (q : Quadrilateral) : Prop := sorry

-- Define the property of a point being outside a quadrilateral
def isOutside (p : Point2D) (q : Quadrilateral) : Prop := sorry

-- Define the property of a point being on the boundary of a quadrilateral
def isOnBoundary (p : Point2D) (q : Quadrilateral) : Prop := sorry

theorem equidistant_point_location (q : Quadrilateral) (h : isConvex q) :
  ∃ p : Point2D, isEquidistant p q ∧
    (isInside p q ∨ isOutside p q ∨ isOnBoundary p q) := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_location_l2127_212734


namespace NUMINAMATH_CALUDE_platform_length_calculation_l2127_212793

/-- Calculates the length of a platform given train length and crossing times -/
theorem platform_length_calculation (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) :
  train_length = 300 →
  platform_time = 33 →
  pole_time = 18 →
  ∃ (platform_length : ℝ),
    platform_length = platform_time * (train_length / pole_time) - train_length ∧
    platform_length = 250 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_calculation_l2127_212793


namespace NUMINAMATH_CALUDE_lucas_50th_mod5_lucas_50th_remainder_l2127_212770

/-- Lucas sequence -/
def lucas : ℕ → ℤ
  | 0 => 2
  | 1 => 1
  | n + 2 => lucas (n + 1) + lucas n

/-- Lucas sequence modulo 5 -/
def lucas_mod5 (n : ℕ) : ℤ := lucas n % 5

/-- The Lucas sequence modulo 5 has a period of 4 -/
axiom lucas_mod5_period : ∀ n, lucas_mod5 (n + 4) = lucas_mod5 n

/-- The 50th term of the Lucas sequence modulo 5 equals the 2nd term modulo 5 -/
theorem lucas_50th_mod5 : lucas_mod5 50 = lucas_mod5 2 := by sorry

/-- The remainder when the 50th term of the Lucas sequence is divided by 5 is 1 -/
theorem lucas_50th_remainder : lucas 50 % 5 = 1 := by sorry

end NUMINAMATH_CALUDE_lucas_50th_mod5_lucas_50th_remainder_l2127_212770


namespace NUMINAMATH_CALUDE_triangle_side_length_l2127_212766

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  b = 2 * Real.sqrt 3 →
  B = 2 * π / 3 →
  C = π / 6 →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2127_212766


namespace NUMINAMATH_CALUDE_problem1_l2127_212751

theorem problem1 : |-3| - Real.sqrt 12 + 2 * Real.sin (30 * π / 180) + (-1) ^ 2021 = 3 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem1_l2127_212751


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l2127_212722

/-- A point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Curve C₁ in polar coordinates -/
def C₁ (p : PolarPoint) : Prop :=
  p.ρ * (Real.sqrt 2 * Real.cos p.θ + Real.sin p.θ) = 1

/-- Curve C₂ in polar coordinates -/
def C₂ (a : ℝ) (p : PolarPoint) : Prop :=
  p.ρ = a

/-- A point is on the polar axis if its θ coordinate is 0 or π -/
def onPolarAxis (p : PolarPoint) : Prop :=
  p.θ = 0 ∨ p.θ = Real.pi

theorem intersection_implies_a_value (a : ℝ) (h_a_pos : a > 0) :
  (∃ p : PolarPoint, C₁ p ∧ C₂ a p ∧ onPolarAxis p) →
  a = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l2127_212722


namespace NUMINAMATH_CALUDE_square_root_divided_by_19_equals_4_l2127_212769

theorem square_root_divided_by_19_equals_4 : 
  ∃ (x : ℝ), x > 0 ∧ (Real.sqrt x) / 19 = 4 ∧ x = 5776 := by
  sorry

end NUMINAMATH_CALUDE_square_root_divided_by_19_equals_4_l2127_212769


namespace NUMINAMATH_CALUDE_distance_between_points_l2127_212710

def point1 : ℝ × ℝ := (2, -3)
def point2 : ℝ × ℝ := (13, 4)

theorem distance_between_points : 
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = Real.sqrt 170 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2127_212710


namespace NUMINAMATH_CALUDE_compound_interest_equation_l2127_212756

/-- The initial sum of money lent out -/
def P : ℝ := sorry

/-- The final amount after 2 years -/
def final_amount : ℝ := 341

/-- The semi-annual interest rate for the first year -/
def r1 : ℝ := 0.025

/-- The semi-annual interest rate for the second year -/
def r2 : ℝ := 0.03

/-- The number of compounding periods per year -/
def n : ℕ := 2

/-- The total number of compounding periods -/
def total_periods : ℕ := 4

theorem compound_interest_equation :
  P * (1 + r1)^n * (1 + r2)^n = final_amount := by sorry

end NUMINAMATH_CALUDE_compound_interest_equation_l2127_212756


namespace NUMINAMATH_CALUDE_positive_integers_satisfying_condition_l2127_212799

theorem positive_integers_satisfying_condition :
  ∀ n : ℕ+, (25 - 3 * n.val ≥ 4) ↔ n.val ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_positive_integers_satisfying_condition_l2127_212799


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2127_212765

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d > 0 ∧
  (∀ n, a n > 0) ∧
  (a 1 = 1) ∧
  (∀ n, a (n + 1) = a n + d) ∧
  (a 3 * a 11 = (a 4 + 5/2)^2)

/-- Theorem stating the difference between two terms -/
theorem arithmetic_sequence_difference
  (a : ℕ → ℝ) (m n : ℕ) (h : ArithmeticSequence a) (h_diff : m - n = 8) :
  a m - a n = 12 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2127_212765


namespace NUMINAMATH_CALUDE_nickel_difference_formula_l2127_212790

/-- The number of nickels equivalent to one quarter -/
def nickels_per_quarter : ℕ := 5

/-- Alice's quarters as a function of q -/
def alice_quarters (q : ℕ) : ℕ := 10 * q + 2

/-- Bob's quarters as a function of q -/
def bob_quarters (q : ℕ) : ℕ := 2 * q + 10

/-- The difference in nickels between Alice and Bob -/
def nickel_difference (q : ℕ) : ℤ :=
  (alice_quarters q - bob_quarters q) * nickels_per_quarter

theorem nickel_difference_formula (q : ℕ) :
  nickel_difference q = 40 * (q - 1) := by sorry

end NUMINAMATH_CALUDE_nickel_difference_formula_l2127_212790


namespace NUMINAMATH_CALUDE_completePassage_correct_l2127_212741

/-- Represents an incomplete sentence or passage -/
inductive IncompleteSentence : Type
| Wei : IncompleteSentence
| Zhuangzi : IncompleteSentence
| TaoYuanming : IncompleteSentence
| LiBai : IncompleteSentence
| SuShi : IncompleteSentence
| XinQiji : IncompleteSentence
| Analects : IncompleteSentence
| LiuYuxi : IncompleteSentence

/-- Represents the correct completion for a sentence -/
def Completion : Type := String

/-- A function that returns the correct completion for a given incomplete sentence -/
def completePassage : IncompleteSentence → Completion
| IncompleteSentence.Wei => "垝垣"
| IncompleteSentence.Zhuangzi => "水之积也不厚"
| IncompleteSentence.TaoYuanming => "仰而视之"
| IncompleteSentence.LiBai => "扶疏荫初上"
| IncompleteSentence.SuShi => "举匏樽"
| IncompleteSentence.XinQiji => "骑鲸鱼"
| IncompleteSentence.Analects => "切问而近思"
| IncompleteSentence.LiuYuxi => "莫是银屏"

/-- Theorem stating that the completePassage function returns the correct completion for each incomplete sentence -/
theorem completePassage_correct :
  ∀ (s : IncompleteSentence), 
    (s = IncompleteSentence.Wei → completePassage s = "垝垣") ∧
    (s = IncompleteSentence.Zhuangzi → completePassage s = "水之积也不厚") ∧
    (s = IncompleteSentence.TaoYuanming → completePassage s = "仰而视之") ∧
    (s = IncompleteSentence.LiBai → completePassage s = "扶疏荫初上") ∧
    (s = IncompleteSentence.SuShi → completePassage s = "举匏樽") ∧
    (s = IncompleteSentence.XinQiji → completePassage s = "骑鲸鱼") ∧
    (s = IncompleteSentence.Analects → completePassage s = "切问而近思") ∧
    (s = IncompleteSentence.LiuYuxi → completePassage s = "莫是银屏") :=
by sorry


end NUMINAMATH_CALUDE_completePassage_correct_l2127_212741


namespace NUMINAMATH_CALUDE_infinite_sqrt_two_plus_l2127_212798

theorem infinite_sqrt_two_plus (x : ℝ) : x > 0 ∧ x^2 = 2 + x → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_sqrt_two_plus_l2127_212798


namespace NUMINAMATH_CALUDE_last_digit_389_base4_l2127_212706

def decimal_to_base4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem last_digit_389_base4 :
  (decimal_to_base4 389).getLast? = some 1 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_389_base4_l2127_212706


namespace NUMINAMATH_CALUDE_percentage_not_participating_l2127_212768

theorem percentage_not_participating (total_students : ℕ) (music_and_sports : ℕ) (music_only : ℕ) (sports_only : ℕ) :
  total_students = 50 →
  music_and_sports = 5 →
  music_only = 15 →
  sports_only = 20 →
  (total_students - (music_and_sports + music_only + sports_only)) / total_students * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_not_participating_l2127_212768


namespace NUMINAMATH_CALUDE_angle_B_measure_l2127_212775

theorem angle_B_measure (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 3 →
  b = Real.sqrt 2 →
  A = 60 * π / 180 →
  -- Sine Rule
  a / Real.sin A = b / Real.sin B →
  -- Triangle inequality (ensuring it's a valid triangle)
  a + b > c ∧ b + c > a ∧ c + a > b →
  -- Sum of angles in a triangle is π
  A + B + C = π →
  B = 45 * π / 180 := by sorry

end NUMINAMATH_CALUDE_angle_B_measure_l2127_212775


namespace NUMINAMATH_CALUDE_triangle_inequality_and_equality_condition_l2127_212755

theorem triangle_inequality_and_equality_condition (a b c : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) : 
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧ 
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_and_equality_condition_l2127_212755


namespace NUMINAMATH_CALUDE_water_left_l2127_212780

theorem water_left (initial_water : ℚ) (used_water : ℚ) (water_left : ℚ) : 
  initial_water = 3 ∧ used_water = 11/4 → water_left = initial_water - used_water → water_left = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_water_left_l2127_212780


namespace NUMINAMATH_CALUDE_correct_result_l2127_212787

variables (a b c : ℝ)

def A : ℝ := 3 * a * b - 2 * a * c + 5 * b * c + 2 * (a * b + 2 * b * c - 4 * a * c)

theorem correct_result :
  A a b c - 2 * (a * b + 2 * b * c - 4 * a * c) = -a * b + 14 * a * c - 3 * b * c := by
  sorry

end NUMINAMATH_CALUDE_correct_result_l2127_212787


namespace NUMINAMATH_CALUDE_horner_method_correctness_horner_method_equivalence_l2127_212771

/-- Horner's Method evaluation for a specific polynomial -/
def horner_eval (x : ℝ) : ℝ := 
  (((((4 * x - 3) * x + 4) * x - 2) * x - 2) * x + 3)

/-- Count of multiplication operations in Horner's Method for this polynomial -/
def horner_mult_count : ℕ := 5

/-- Count of addition operations in Horner's Method for this polynomial -/
def horner_add_count : ℕ := 5

/-- Theorem stating the correctness of Horner's Method for the given polynomial -/
theorem horner_method_correctness : 
  horner_eval 3 = 816 ∧ 
  horner_mult_count = 5 ∧ 
  horner_add_count = 5 := by sorry

/-- Theorem stating that Horner's Method gives the same result as direct polynomial evaluation -/
theorem horner_method_equivalence (x : ℝ) : 
  horner_eval x = 4 * x^5 - 3 * x^4 + 4 * x^3 - 2 * x^2 - 2 * x + 3 := by sorry

end NUMINAMATH_CALUDE_horner_method_correctness_horner_method_equivalence_l2127_212771


namespace NUMINAMATH_CALUDE_sticks_for_800_hexagons_l2127_212781

/-- The number of sticks required to form a row of n hexagons -/
def sticksForHexagons (n : ℕ) : ℕ :=
  if n = 0 then 0 else 6 + 5 * (n - 1)

/-- Theorem: The number of sticks required for 800 hexagons is 4001 -/
theorem sticks_for_800_hexagons : sticksForHexagons 800 = 4001 := by
  sorry

#eval sticksForHexagons 800  -- To verify the result

end NUMINAMATH_CALUDE_sticks_for_800_hexagons_l2127_212781
