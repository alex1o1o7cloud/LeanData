import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_squared_l2229_222917

theorem sqrt_x_minus_one_squared (x : ℝ) (h : |1 - x| = 1 + |x|) : 
  Real.sqrt ((x - 1)^2) = 1 - x :=
by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_squared_l2229_222917


namespace NUMINAMATH_CALUDE_sock_selection_theorem_l2229_222924

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of socks in the drawer -/
def total_socks : ℕ := 8

/-- The number of socks to be chosen -/
def socks_to_choose : ℕ := 4

/-- The number of non-red socks -/
def non_red_socks : ℕ := 7

theorem sock_selection_theorem :
  choose total_socks socks_to_choose - choose non_red_socks socks_to_choose = 35 := by sorry

end NUMINAMATH_CALUDE_sock_selection_theorem_l2229_222924


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2229_222939

theorem constant_term_binomial_expansion :
  ∃ (c : ℝ), ∀ (x : ℝ),
    (λ x => (4^x - 2^(-x))^6) x = c + (λ x => (4^x - 2^(-x))^6 - c) x ∧ c = 15 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2229_222939


namespace NUMINAMATH_CALUDE_simplify_expression_value_under_condition_independence_condition_l2229_222902

/-- Given algebraic expressions A and B -/
def A (m y : ℝ) : ℝ := 2 * m^2 + 3 * m * y + 2 * y - 1

def B (m y : ℝ) : ℝ := m^2 - m * y

/-- Theorem 1: Simplification of 3A - 2(A + B) -/
theorem simplify_expression (m y : ℝ) :
  3 * A m y - 2 * (A m y + B m y) = 5 * m * y + 2 * y - 1 := by sorry

/-- Theorem 2: Value of 3A - 2(A + B) under specific condition -/
theorem value_under_condition (m y : ℝ) :
  (m - 1)^2 + |y + 2| = 0 →
  3 * A m y - 2 * (A m y + B m y) = -15 := by sorry

/-- Theorem 3: Condition for 3A - 2(A + B) to be independent of y -/
theorem independence_condition (m : ℝ) :
  (∀ y : ℝ, 3 * A m y - 2 * (A m y + B m y) = 5 * m * y + 2 * y - 1) →
  m = -2/5 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_value_under_condition_independence_condition_l2229_222902


namespace NUMINAMATH_CALUDE_dogs_not_doing_anything_l2229_222967

/-- Proves that the number of dogs not doing anything is 10, given the total number of dogs and the number of dogs engaged in each activity. -/
theorem dogs_not_doing_anything (total : ℕ) (running : ℕ) (playing : ℕ) (barking : ℕ) : 
  total = 88 → 
  running = 12 → 
  playing = total / 2 → 
  barking = total / 4 → 
  total - (running + playing + barking) = 10 := by
sorry

end NUMINAMATH_CALUDE_dogs_not_doing_anything_l2229_222967


namespace NUMINAMATH_CALUDE_smallest_start_for_five_odd_squares_l2229_222950

theorem smallest_start_for_five_odd_squares : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (s : Finset ℕ), s.card = 5 ∧ 
    (∀ m ∈ s, n ≤ m ∧ m ≤ 100 ∧ Odd m ∧ ∃ k : ℕ, m = k^2) ∧
    (∀ m : ℕ, n ≤ m ∧ m ≤ 100 ∧ Odd m ∧ (∃ k : ℕ, m = k^2) → m ∈ s)) ∧
  (∀ n' : ℕ, 0 < n' ∧ n' < n → 
    ¬∃ (s : Finset ℕ), s.card = 5 ∧ 
      (∀ m ∈ s, n' ≤ m ∧ m ≤ 100 ∧ Odd m ∧ ∃ k : ℕ, m = k^2) ∧
      (∀ m : ℕ, n' ≤ m ∧ m ≤ 100 ∧ Odd m ∧ (∃ k : ℕ, m = k^2) → m ∈ s)) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_start_for_five_odd_squares_l2229_222950


namespace NUMINAMATH_CALUDE_simplify_fraction_l2229_222911

theorem simplify_fraction : (5 : ℚ) * (13 / 3) * (21 / -65) = -7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2229_222911


namespace NUMINAMATH_CALUDE_part_one_part_two_l2229_222972

-- Define propositions P and Q
def P (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def Q (x : ℝ) : Prop := |x - 3| ≤ 1

-- Part 1
theorem part_one (x : ℝ) (h : P x 1 ∨ Q x) : 1 < x ∧ x ≤ 4 := by sorry

-- Part 2
theorem part_two (a : ℝ) (h : ∀ x, ¬(P x a) → ¬(Q x)) 
  (h_not_sufficient : ∃ x, ¬(P x a) ∧ Q x) : 4/3 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2229_222972


namespace NUMINAMATH_CALUDE_grid_sum_l2229_222922

theorem grid_sum (X Y Z : ℝ) 
  (row1_sum : 1 + X + 3 = 9)
  (row2_sum : 2 + Y + Z = 9) :
  X + Y + Z = 12 := by sorry

end NUMINAMATH_CALUDE_grid_sum_l2229_222922


namespace NUMINAMATH_CALUDE_min_value_f_max_value_y_l2229_222903

-- Problem 1
theorem min_value_f (x : ℝ) (hx : x > 0) :
  2 / x + 2 * x ≥ 4 ∧ (2 / x + 2 * x = 4 ↔ x = 1) :=
by sorry

-- Problem 2
theorem max_value_y (x : ℝ) (hx1 : x > 0) (hx2 : x < 1/3) :
  x * (1 - 3 * x) ≤ 1/12 :=
by sorry

end NUMINAMATH_CALUDE_min_value_f_max_value_y_l2229_222903


namespace NUMINAMATH_CALUDE_annual_lesson_cost_difference_l2229_222929

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- The hourly rate for clarinet lessons in dollars -/
def clarinet_rate : ℕ := 40

/-- The number of hours per week of clarinet lessons -/
def clarinet_hours : ℕ := 3

/-- The hourly rate for piano lessons in dollars -/
def piano_rate : ℕ := 28

/-- The number of hours per week of piano lessons -/
def piano_hours : ℕ := 5

/-- The difference in annual spending between piano and clarinet lessons -/
theorem annual_lesson_cost_difference :
  (piano_rate * piano_hours - clarinet_rate * clarinet_hours) * weeks_per_year = 1040 := by
  sorry

end NUMINAMATH_CALUDE_annual_lesson_cost_difference_l2229_222929


namespace NUMINAMATH_CALUDE_quadratic_linear_intersection_l2229_222970

/-- Given a quadratic function f(x) = ax² + bx + c and a linear function g(x) = -bx,
    where a > b > c and f(1) = 0, prove that f and g intersect at two distinct points. -/
theorem quadratic_linear_intersection
  (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : a * 1^2 + b * 1 + c = 0) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    a * x₁^2 + b * x₁ + c = -b * x₁ ∧
    a * x₂^2 + b * x₂ + c = -b * x₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_linear_intersection_l2229_222970


namespace NUMINAMATH_CALUDE_complex_point_C_l2229_222909

/-- Given points A, B, and C in the complex plane, prove that C corresponds to 4-2i -/
theorem complex_point_C (A B C : ℂ) : 
  A = 2 + I →
  B - A = 1 + 2*I →
  C - B = 3 - I →
  C = 4 - 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_point_C_l2229_222909


namespace NUMINAMATH_CALUDE_log_cutting_theorem_l2229_222983

/-- The number of pieces of wood after cutting a log -/
def num_pieces (initial_logs : ℕ) (num_cuts : ℕ) : ℕ :=
  initial_logs + num_cuts

/-- Theorem: Cutting a single log 10 times results in 11 pieces -/
theorem log_cutting_theorem :
  num_pieces 1 10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_log_cutting_theorem_l2229_222983


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l2229_222900

theorem rhombus_diagonal (d1 : ℝ) (area : ℝ) (d2 : ℝ) : 
  d1 = 30 → area = 180 → area = (d1 * d2) / 2 → d2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l2229_222900


namespace NUMINAMATH_CALUDE_road_graveling_cost_l2229_222953

/-- Calculate the cost of graveling two intersecting roads on a rectangular lawn -/
theorem road_graveling_cost (lawn_length lawn_width road_width gravel_cost : ℝ) 
  (h1 : lawn_length = 90)
  (h2 : lawn_width = 60)
  (h3 : road_width = 10)
  (h4 : gravel_cost = 3) : 
  (lawn_length * road_width + lawn_width * road_width - road_width * road_width) * gravel_cost = 4200 := by
  sorry

#check road_graveling_cost

end NUMINAMATH_CALUDE_road_graveling_cost_l2229_222953


namespace NUMINAMATH_CALUDE_twentyseven_eighths_two_thirds_power_l2229_222938

theorem twentyseven_eighths_two_thirds_power :
  (27 / 8 : ℝ) ^ (2 / 3) = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_twentyseven_eighths_two_thirds_power_l2229_222938


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_sum_l2229_222943

theorem geometric_sequence_common_ratio_sum 
  (k p r : ℝ) 
  (h_nonconstant_p : p ≠ 1) 
  (h_nonconstant_r : r ≠ 1) 
  (h_different_ratios : p ≠ r) 
  (h_relation : k * p^2 - k * r^2 = 5 * (k * p - k * r)) :
  p + r = 5 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_sum_l2229_222943


namespace NUMINAMATH_CALUDE_researcher_reading_rate_l2229_222907

theorem researcher_reading_rate 
  (total_pages : ℕ) 
  (total_hours : ℕ) 
  (h1 : total_pages = 30000) 
  (h2 : total_hours = 150) : 
  (total_pages : ℚ) / total_hours = 200 := by
  sorry

end NUMINAMATH_CALUDE_researcher_reading_rate_l2229_222907


namespace NUMINAMATH_CALUDE_right_triangle_properties_l2229_222905

-- Define a right triangle with hypotenuse 13 and one side 5
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_right : a^2 + b^2 = c^2
  hypotenuse_length : c = 13
  side_length : a = 5

-- Theorem statement
theorem right_triangle_properties (t : RightTriangle) :
  t.b = 12 ∧
  (1/2 : ℝ) * t.a * t.b = 30 ∧
  t.a + t.b + t.c = 30 ∧
  (∃ θ₁ θ₂ : ℝ, 0 < θ₁ ∧ θ₁ < π/2 ∧ 0 < θ₂ ∧ θ₂ < π/2 ∧ θ₁ + θ₂ = π/2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_properties_l2229_222905


namespace NUMINAMATH_CALUDE_tangent_points_parallel_to_line_tangent_points_on_curve_unique_tangent_points_main_theorem_l2229_222942

-- Define the function f(x) = x^3 + x - 2
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f(x)
def f_derivative (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_points_parallel_to_line (x : ℝ) :
  (f_derivative x = 4) ↔ (x = 1 ∨ x = -1) := by sorry

theorem tangent_points_on_curve :
  f 1 = 0 ∧ f (-1) = -4 := by sorry

theorem unique_tangent_points :
  ∀ x : ℝ, f_derivative x = 4 → (x = 1 ∨ x = -1) := by sorry

-- Main theorem
theorem main_theorem :
  ∃! s : Set (ℝ × ℝ), s = {(1, 0), (-1, -4)} ∧
  (∀ (x y : ℝ), (x, y) ∈ s ↔ f x = y ∧ f_derivative x = 4) := by sorry

end NUMINAMATH_CALUDE_tangent_points_parallel_to_line_tangent_points_on_curve_unique_tangent_points_main_theorem_l2229_222942


namespace NUMINAMATH_CALUDE_distance_is_8_sqrt2_div_3_l2229_222977

/-- Two lines l₁ and l₂ in the plane -/
structure ParallelLines where
  a : ℝ
  l₁ : ℝ × ℝ → Prop
  l₂ : ℝ × ℝ → Prop
  l₁_eq : ∀ x y, l₁ (x, y) ↔ x + a * y + 6 = 0
  l₂_eq : ∀ x y, l₂ (x, y) ↔ (a - 2) * x + 3 * y + 2 * a = 0
  parallel : ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, l₁ (x, y) ↔ l₂ (k * x, k * y)

/-- The distance between two parallel lines -/
def distance (lines : ParallelLines) : ℝ := sorry

/-- Theorem: The distance between the parallel lines is 8√2/3 -/
theorem distance_is_8_sqrt2_div_3 (lines : ParallelLines) :
  distance lines = 8 * Real.sqrt 2 / 3 := by sorry

end NUMINAMATH_CALUDE_distance_is_8_sqrt2_div_3_l2229_222977


namespace NUMINAMATH_CALUDE_matilda_age_l2229_222949

/-- Given the ages of Louis, Jerica, and Matilda, prove Matilda's age -/
theorem matilda_age (louis_age jerica_age matilda_age : ℕ) : 
  louis_age = 14 →
  jerica_age = 2 * louis_age →
  matilda_age = jerica_age + 7 →
  matilda_age = 35 := by
sorry

end NUMINAMATH_CALUDE_matilda_age_l2229_222949


namespace NUMINAMATH_CALUDE_digit_sum_problem_l2229_222923

theorem digit_sum_problem (w x y z : ℕ) : 
  w ≤ 9 ∧ x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9 ∧  -- digits are between 0 and 9
  w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧  -- all digits are different
  y + w = 10 ∧  -- sum in ones place
  x + y + 1 = 10 ∧  -- sum in tens place with carry
  w + z + 1 = 11  -- sum in hundreds place with carry
  →
  w + x + y + z = 23 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l2229_222923


namespace NUMINAMATH_CALUDE_sequence_bounds_l2229_222991

def a : ℕ → ℚ
  | 0 => 2
  | n + 1 => (2 * a n + 1) / (a n + 2)

theorem sequence_bounds : ∀ n : ℕ, n ≥ 1 → 1 < a n ∧ a n < 1 + 1 / 3^n := by
  sorry

end NUMINAMATH_CALUDE_sequence_bounds_l2229_222991


namespace NUMINAMATH_CALUDE_fair_coin_probability_l2229_222992

theorem fair_coin_probability : 
  let n : ℕ := 8  -- number of coin tosses
  let p : ℚ := 1/2  -- probability of heads for a fair coin
  let favorable_outcomes : ℕ := (n.choose 2) + (n.choose 3) + (n.choose 4)
  let total_outcomes : ℕ := 2^n
  (favorable_outcomes : ℚ) / total_outcomes = 77/128 := by sorry

end NUMINAMATH_CALUDE_fair_coin_probability_l2229_222992


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_two_l2229_222921

theorem fraction_zero_implies_x_negative_two (x : ℝ) :
  (|x| - 2) / (x - 2) = 0 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_two_l2229_222921


namespace NUMINAMATH_CALUDE_new_person_weight_l2229_222933

def initial_persons : ℕ := 10
def average_weight_increase : ℚ := 63/10
def replaced_person_weight : ℚ := 65

theorem new_person_weight :
  let total_weight_increase : ℚ := initial_persons * average_weight_increase
  let new_person_weight : ℚ := replaced_person_weight + total_weight_increase
  new_person_weight = 128 := by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2229_222933


namespace NUMINAMATH_CALUDE_x_plus_inv_x_power_n_is_integer_l2229_222928

theorem x_plus_inv_x_power_n_is_integer
  (x : ℝ) (h : ∃ (k : ℤ), x + 1 / x = k) :
  ∀ n : ℕ, ∃ (m : ℤ), x^n + 1 / x^n = m :=
by sorry

end NUMINAMATH_CALUDE_x_plus_inv_x_power_n_is_integer_l2229_222928


namespace NUMINAMATH_CALUDE_complex_exp_thirteen_pi_over_two_l2229_222966

theorem complex_exp_thirteen_pi_over_two : Complex.exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_thirteen_pi_over_two_l2229_222966


namespace NUMINAMATH_CALUDE_min_trips_for_field_trip_l2229_222945

/-- The minimum number of trips required to transport all students -/
def min_trips (total_students : ℕ) (num_buses : ℕ) (bus_capacity : ℕ) : ℕ :=
  (total_students + num_buses * bus_capacity - 1) / (num_buses * bus_capacity)

theorem min_trips_for_field_trip :
  min_trips 520 5 45 = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_trips_for_field_trip_l2229_222945


namespace NUMINAMATH_CALUDE_unique_two_digit_number_mod_4_17_l2229_222910

theorem unique_two_digit_number_mod_4_17 : 
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ n % 4 = 1 ∧ n % 17 = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_mod_4_17_l2229_222910


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l2229_222948

-- Define the operation
noncomputable def bowtie (c x : ℝ) : ℝ := c + Real.sqrt (x + Real.sqrt (x + Real.sqrt (x + Real.sqrt x)))

-- State the theorem
theorem bowtie_equation_solution :
  ∃ x : ℝ, bowtie 5 x = 11 ∧ x = 30 := by sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l2229_222948


namespace NUMINAMATH_CALUDE_age_difference_l2229_222963

theorem age_difference (a b : ℕ) : 
  (a > 0) → 
  (b > 0) → 
  (a < 100) → 
  (b < 100) → 
  (a = 10 * (b % 10) + (a / 10)) → 
  (b = 10 * (a % 10) + (b / 10)) → 
  (a + 7 = 3 * (b + 7)) → 
  (a - b = 45) := by
sorry

end NUMINAMATH_CALUDE_age_difference_l2229_222963


namespace NUMINAMATH_CALUDE_gold_alloy_ratio_l2229_222981

/-- Proves that adding 62.5 ounces of pure gold to an alloy of 100 ounces that is 35% gold 
    will result in a new alloy that is 60% gold -/
theorem gold_alloy_ratio (original_weight : ℝ) (original_gold_ratio : ℝ) 
    (new_gold_ratio : ℝ) (added_gold : ℝ) : 
    original_weight = 100 →
    original_gold_ratio = 0.35 →
    new_gold_ratio = 0.60 →
    added_gold = 62.5 →
    (original_weight * original_gold_ratio + added_gold) / (original_weight + added_gold) = new_gold_ratio := by
  sorry

#check gold_alloy_ratio

end NUMINAMATH_CALUDE_gold_alloy_ratio_l2229_222981


namespace NUMINAMATH_CALUDE_volume_of_inscribed_sphere_l2229_222956

/-- The volume of a sphere inscribed in a cube with edge length 12 inches is 288π cubic inches. -/
theorem volume_of_inscribed_sphere (π : ℝ) :
  let cube_edge : ℝ := 12
  let sphere_radius : ℝ := cube_edge / 2
  let sphere_volume : ℝ := (4 / 3) * π * sphere_radius ^ 3
  sphere_volume = 288 * π := by sorry

end NUMINAMATH_CALUDE_volume_of_inscribed_sphere_l2229_222956


namespace NUMINAMATH_CALUDE_definite_integral_tan_trig_l2229_222987

theorem definite_integral_tan_trig : 
  ∃ (f : ℝ → ℝ), (∀ x ∈ Set.Icc (π / 4) (Real.arcsin (Real.sqrt (2 / 3))), 
    HasDerivAt f ((8 * Real.tan x) / (3 * (Real.cos x)^2 + 8 * Real.sin (2 * x) - 7)) x) ∧ 
  (f (Real.arcsin (Real.sqrt (2 / 3))) - f (π / 4) = 
    (4 / 21) * Real.log (abs ((7 * Real.sqrt 2 - 2) / 5)) - 
    (4 / 3) * Real.log (abs (2 - Real.sqrt 2))) := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_tan_trig_l2229_222987


namespace NUMINAMATH_CALUDE_irreducible_fractions_count_l2229_222940

/-- A rational number between 0 and 1 in irreducible fraction form -/
structure IrreducibleFraction :=
  (numerator : ℕ)
  (denominator : ℕ)
  (is_between_0_and_1 : numerator < denominator)
  (is_irreducible : Nat.gcd numerator denominator = 1)
  (product_is_20 : numerator * denominator = 20)

/-- The count of irreducible fractions between 0 and 1 with numerator-denominator product of 20 -/
def count_irreducible_fractions : ℕ := sorry

/-- The main theorem stating there are 128 such fractions -/
theorem irreducible_fractions_count :
  count_irreducible_fractions = 128 := by sorry

end NUMINAMATH_CALUDE_irreducible_fractions_count_l2229_222940


namespace NUMINAMATH_CALUDE_division_problem_l2229_222936

theorem division_problem (divisor quotient remainder : ℕ) (h1 : divisor = 21) (h2 : quotient = 8) (h3 : remainder = 3) :
  divisor * quotient + remainder = 171 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2229_222936


namespace NUMINAMATH_CALUDE_downstream_distance_84km_l2229_222990

/-- Represents the swimming scenario --/
structure SwimmingScenario where
  current_speed : ℝ
  upstream_distance : ℝ
  upstream_time : ℝ
  downstream_time : ℝ

/-- Calculates the downstream distance given a swimming scenario --/
def downstream_distance (s : SwimmingScenario) : ℝ :=
  sorry

/-- Theorem stating the downstream distance for the given scenario --/
theorem downstream_distance_84km (s : SwimmingScenario) 
  (h1 : s.current_speed = 2.5)
  (h2 : s.upstream_distance = 24)
  (h3 : s.upstream_time = 8)
  (h4 : s.downstream_time = 8) :
  downstream_distance s = 84 := by
  sorry

end NUMINAMATH_CALUDE_downstream_distance_84km_l2229_222990


namespace NUMINAMATH_CALUDE_pizza_slices_per_person_l2229_222959

/-- Given a pizza with 12 slices shared equally among Ron and his 2 friends, 
    prove that each person ate 4 slices. -/
theorem pizza_slices_per_person (total_slices : Nat) (num_friends : Nat) :
  total_slices = 12 →
  num_friends = 2 →
  total_slices / (num_friends + 1) = 4 :=
by sorry

end NUMINAMATH_CALUDE_pizza_slices_per_person_l2229_222959


namespace NUMINAMATH_CALUDE_greg_and_earl_final_amount_l2229_222952

/-- Represents the financial state of three individuals and their debts --/
structure FinancialState where
  earl_initial : ℕ
  fred_initial : ℕ
  greg_initial : ℕ
  earl_owes_fred : ℕ
  fred_owes_greg : ℕ
  greg_owes_earl : ℕ

/-- Calculates the final amount Greg and Earl have together after all debts are paid --/
def final_amount (state : FinancialState) : ℕ :=
  (state.earl_initial - state.earl_owes_fred + state.greg_owes_earl) +
  (state.greg_initial + state.fred_owes_greg - state.greg_owes_earl)

/-- Theorem stating that Greg and Earl will have $130 together after all debts are paid --/
theorem greg_and_earl_final_amount (state : FinancialState)
  (h1 : state.earl_initial = 90)
  (h2 : state.fred_initial = 48)
  (h3 : state.greg_initial = 36)
  (h4 : state.earl_owes_fred = 28)
  (h5 : state.fred_owes_greg = 32)
  (h6 : state.greg_owes_earl = 40) :
  final_amount state = 130 := by
  sorry

end NUMINAMATH_CALUDE_greg_and_earl_final_amount_l2229_222952


namespace NUMINAMATH_CALUDE_maxine_purchase_l2229_222974

theorem maxine_purchase (x y z : ℕ) : 
  x + y + z = 40 ∧ 
  50 * x + 400 * y + 500 * z = 10000 →
  x = 40 ∧ y = 0 ∧ z = 0 := by
sorry

end NUMINAMATH_CALUDE_maxine_purchase_l2229_222974


namespace NUMINAMATH_CALUDE_pyramid_volume_l2229_222947

/-- The volume of a pyramid with a square base and given dimensions -/
theorem pyramid_volume (base_side : ℝ) (edge_length : ℝ) (h : base_side = 10 ∧ edge_length = 17) :
  (1 / 3 : ℝ) * base_side ^ 2 * Real.sqrt (edge_length ^ 2 - (base_side ^ 2 / 2)) = 
    (100 * Real.sqrt 239) / 3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_l2229_222947


namespace NUMINAMATH_CALUDE_triangle_properties_l2229_222916

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem about the triangle with specific properties -/
theorem triangle_properties (t : Triangle) 
  (h1 : (2 * t.c - t.b) / t.a = Real.cos t.B / Real.cos t.A) 
  (h2 : t.a = 2 * Real.sqrt 5) : 
  t.A = π / 3 ∧ 
  (∃ (S : ℝ), S = 5 * Real.sqrt 3 ∧ ∀ (area : ℝ), area ≤ S) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l2229_222916


namespace NUMINAMATH_CALUDE_passing_mark_is_200_l2229_222969

/-- Represents an exam with a total number of marks and a passing mark. -/
structure Exam where
  total_marks : ℕ
  passing_mark : ℕ

/-- Defines the conditions of the exam as described in the problem. -/
def exam_conditions (e : Exam) : Prop :=
  (e.total_marks * 30 / 100 + 50 = e.passing_mark) ∧
  (e.total_marks * 45 / 100 = e.passing_mark + 25)

/-- Theorem stating that under the given conditions, the passing mark is 200. -/
theorem passing_mark_is_200 :
  ∃ e : Exam, exam_conditions e ∧ e.passing_mark = 200 := by
  sorry


end NUMINAMATH_CALUDE_passing_mark_is_200_l2229_222969


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2229_222975

theorem inequality_solution_set (m : ℝ) :
  let S := {x : ℝ | x^2 + (m - 1) * x - m > 0}
  (m = -1 → S = {x : ℝ | x ≠ 1}) ∧
  (m > -1 → S = {x : ℝ | x < -m ∨ x > 1}) ∧
  (m < -1 → S = {x : ℝ | x < 1 ∨ x > -m}) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2229_222975


namespace NUMINAMATH_CALUDE_midpoint_property_l2229_222944

/-- Given two points A and B in ℝ², prove that if C is their midpoint,
    then 2x - 4y = -22 where (x, y) are the coordinates of C. -/
theorem midpoint_property (A B : ℝ × ℝ) (h1 : A = (15, 10)) (h2 : B = (-5, 6)) :
  let C := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  2 * C.1 - 4 * C.2 = -22 := by
sorry

end NUMINAMATH_CALUDE_midpoint_property_l2229_222944


namespace NUMINAMATH_CALUDE_five_twelve_thirteen_pythagorean_l2229_222946

/-- Definition of a Pythagorean triple -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

/-- Proof that (5, 12, 13) is a Pythagorean triple -/
theorem five_twelve_thirteen_pythagorean : is_pythagorean_triple 5 12 13 := by
  sorry

end NUMINAMATH_CALUDE_five_twelve_thirteen_pythagorean_l2229_222946


namespace NUMINAMATH_CALUDE_barry_head_stand_theorem_l2229_222927

/-- The number of turns Barry can take standing on his head during a 2-hour period -/
def barry_head_stand_turns : ℕ :=
  let head_stand_time : ℕ := 10  -- minutes
  let sit_time : ℕ := 5  -- minutes
  let total_period : ℕ := 2 * 60  -- 2 hours in minutes
  let time_per_turn : ℕ := head_stand_time + sit_time
  total_period / time_per_turn

theorem barry_head_stand_theorem :
  barry_head_stand_turns = 8 := by
  sorry

end NUMINAMATH_CALUDE_barry_head_stand_theorem_l2229_222927


namespace NUMINAMATH_CALUDE_unique_charming_number_l2229_222984

def is_charming (n : ℕ) : Prop :=
  ∃ a b : ℕ, 
    n = 10 * a + b ∧ 
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧
    n = 2 * a + b^3

theorem unique_charming_number : 
  ∃! n : ℕ, is_charming n := by sorry

end NUMINAMATH_CALUDE_unique_charming_number_l2229_222984


namespace NUMINAMATH_CALUDE_no_solution_implies_m_leq_one_l2229_222932

theorem no_solution_implies_m_leq_one :
  (∀ x : ℝ, ¬(2*x - 1 > 1 ∧ x < m)) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_m_leq_one_l2229_222932


namespace NUMINAMATH_CALUDE_no_savings_on_joint_purchase_l2229_222930

/-- Calculates the cost of windows given the quantity and discount offer -/
def windowCost (quantity : ℕ) (regularPrice : ℕ) (discountRate : ℕ) : ℕ :=
  let discountedQuantity := quantity - (quantity / (discountRate + 2)) * 2
  discountedQuantity * regularPrice

/-- Proves that joint purchase does not lead to savings compared to separate purchases -/
theorem no_savings_on_joint_purchase (regularPrice : ℕ) (discountRate : ℕ) :
  windowCost 22 regularPrice discountRate =
  windowCost 10 regularPrice discountRate + windowCost 12 regularPrice discountRate :=
by sorry

end NUMINAMATH_CALUDE_no_savings_on_joint_purchase_l2229_222930


namespace NUMINAMATH_CALUDE_seans_sandwiches_l2229_222976

/-- Calculates the number of sandwiches Sean bought given the costs of items and total cost -/
theorem seans_sandwiches
  (soda_cost : ℕ)
  (soup_cost : ℕ)
  (sandwich_cost : ℕ)
  (total_cost : ℕ)
  (h1 : soda_cost = 3)
  (h2 : soup_cost = 6)
  (h3 : sandwich_cost = 9)
  (h4 : total_cost = 18)
  : (total_cost - soda_cost - soup_cost) / sandwich_cost = 1 := by
  sorry

#check seans_sandwiches

end NUMINAMATH_CALUDE_seans_sandwiches_l2229_222976


namespace NUMINAMATH_CALUDE_overlap_area_of_specific_triangles_l2229_222954

/-- A point in a 2D grid. -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- A triangle defined by three points in a 2D grid. -/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- Calculate the area of a triangle given its base and height. -/
def triangleArea (base height : ℕ) : ℚ :=
  (base * height : ℚ) / 2

/-- The main theorem stating the area of overlap between two specific triangles. -/
theorem overlap_area_of_specific_triangles :
  let triangleA : GridTriangle := ⟨⟨0, 0⟩, ⟨2, 0⟩, ⟨2, 2⟩⟩
  let triangleB : GridTriangle := ⟨⟨0, 2⟩, ⟨2, 2⟩, ⟨0, 0⟩⟩
  triangleArea 2 2 = 2 := by sorry

end NUMINAMATH_CALUDE_overlap_area_of_specific_triangles_l2229_222954


namespace NUMINAMATH_CALUDE_no_solution_iff_parallel_equation_no_solution_iff_l2229_222978

/-- Two 2D vectors are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, v.1 = c * w.1 ∧ v.2 = c * w.2

/-- The equation has no solution if and only if the direction vectors are parallel -/
theorem no_solution_iff_parallel (m : ℝ) : Prop :=
  parallel (5, 2) (-2, m)

/-- The main theorem: the equation has no solution if and only if m = -4/5 -/
theorem equation_no_solution_iff (m : ℝ) : 
  no_solution_iff_parallel m ↔ m = -4/5 := by sorry

end NUMINAMATH_CALUDE_no_solution_iff_parallel_equation_no_solution_iff_l2229_222978


namespace NUMINAMATH_CALUDE_light_travel_distance_l2229_222997

/-- The distance light travels in one year in miles -/
def light_year_distance : ℝ := 5870000000000

/-- The number of years we want to calculate the light travel distance for -/
def years : ℕ := 50

/-- Theorem stating that the distance light travels in 50 years
    is equal to 293.5 × 10^12 miles -/
theorem light_travel_distance : 
  (light_year_distance * years : ℝ) = 293.5 * (10 ^ 12) := by
  sorry

end NUMINAMATH_CALUDE_light_travel_distance_l2229_222997


namespace NUMINAMATH_CALUDE_x_2021_minus_one_values_l2229_222994

theorem x_2021_minus_one_values (x : ℝ) :
  (x - 1) * (x^5 + x^4 + x^3 + x^2 + x + 1) = 0 →
  x^2021 - 1 = 0 ∨ x^2021 - 1 = -2 := by
sorry

end NUMINAMATH_CALUDE_x_2021_minus_one_values_l2229_222994


namespace NUMINAMATH_CALUDE_cadence_earnings_increase_l2229_222901

/-- Proves that the percentage increase in Cadence's monthly earnings at her new company
    compared to her old company is 20%, given the specified conditions. -/
theorem cadence_earnings_increase (
  old_company_duration : ℕ := 3 * 12
  ) (old_company_monthly_salary : ℕ := 5000)
  (new_company_duration_increase : ℕ := 5)
  (total_earnings : ℕ := 426000) : Real :=
  by
  -- Define the duration of employment at the new company
  let new_company_duration : ℕ := old_company_duration + new_company_duration_increase

  -- Calculate total earnings from the old company
  let old_company_total : ℕ := old_company_duration * old_company_monthly_salary

  -- Calculate total earnings from the new company
  let new_company_total : ℕ := total_earnings - old_company_total

  -- Calculate monthly salary at the new company
  let new_company_monthly_salary : ℕ := new_company_total / new_company_duration

  -- Calculate the percentage increase
  let percentage_increase : Real := 
    (new_company_monthly_salary - old_company_monthly_salary : Real) / old_company_monthly_salary * 100

  -- Prove that the percentage increase is 20%
  sorry


end NUMINAMATH_CALUDE_cadence_earnings_increase_l2229_222901


namespace NUMINAMATH_CALUDE_trajectory_of_point_m_l2229_222985

/-- The trajectory of point M given a circle and specific conditions -/
theorem trajectory_of_point_m (x y : ℝ) : 
  (∃ m n : ℝ, 
    m^2 + n^2 = 9 ∧  -- P(m, n) is on the circle
    (x - m)^2 + y^2 = ((m - x)^2 + y^2) / 4) -- PM = 2MP'
  → x^2 / 9 + y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_of_point_m_l2229_222985


namespace NUMINAMATH_CALUDE_collisions_100_balls_l2229_222906

/-- The number of collisions between n identical balls moving along a single dimension,
    where each pair of balls can collide exactly once. -/
def numCollisions (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that for 100 balls, the number of collisions is 4950. -/
theorem collisions_100_balls :
  numCollisions 100 = 4950 := by
  sorry

end NUMINAMATH_CALUDE_collisions_100_balls_l2229_222906


namespace NUMINAMATH_CALUDE_points_per_correct_answer_l2229_222957

theorem points_per_correct_answer 
  (total_problems : ℕ) 
  (total_score : ℕ) 
  (wrong_answers : ℕ) 
  (points_per_wrong : ℕ) 
  (h1 : total_problems = 25)
  (h2 : total_score = 85)
  (h3 : wrong_answers = 3)
  (h4 : points_per_wrong = 1) :
  (total_score + wrong_answers * points_per_wrong) / (total_problems - wrong_answers) = 4 := by
sorry

end NUMINAMATH_CALUDE_points_per_correct_answer_l2229_222957


namespace NUMINAMATH_CALUDE_simple_interest_rate_example_l2229_222915

/-- Calculates the simple interest rate given principal, amount, and time -/
def simple_interest_rate (principal amount : ℚ) (time : ℕ) : ℚ :=
  ((amount - principal) * 100) / (principal * time)

theorem simple_interest_rate_example :
  simple_interest_rate 750 1050 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_example_l2229_222915


namespace NUMINAMATH_CALUDE_optimal_strategy_with_bicycle_l2229_222925

/-- The optimal strategy for two people to reach a destination with one bicycle. -/
theorem optimal_strategy_with_bicycle 
  (total_distance : ℝ) 
  (walking_speed : ℝ) 
  (cycling_speed : ℝ) 
  (ha : total_distance > 0) 
  (hw : walking_speed > 0) 
  (hc : cycling_speed > walking_speed) :
  ∃ (x : ℝ), 
    0 < x ∧ 
    x < total_distance ∧ 
    (x / walking_speed + (total_distance - x) / cycling_speed = 
     x / walking_speed + (total_distance - x) / walking_speed) ∧
    ∀ (y : ℝ), 
      0 < y → 
      y < total_distance → 
      (y / walking_speed + (total_distance - y) / cycling_speed ≥
       x / walking_speed + (total_distance - x) / cycling_speed) :=
by sorry

end NUMINAMATH_CALUDE_optimal_strategy_with_bicycle_l2229_222925


namespace NUMINAMATH_CALUDE_oranges_sum_l2229_222941

/-- The number of oranges Janet has -/
def janet_oranges : ℕ := 9

/-- The number of oranges Sharon has -/
def sharon_oranges : ℕ := 7

/-- The total number of oranges Janet and Sharon have together -/
def total_oranges : ℕ := janet_oranges + sharon_oranges

theorem oranges_sum : total_oranges = 16 := by
  sorry

end NUMINAMATH_CALUDE_oranges_sum_l2229_222941


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2229_222919

/-- Given a quadratic equation ax² + bx + c = 0, returns the coefficient of x² (a) -/
def quadratic_coefficient (a b c : ℚ) : ℚ := a

/-- Given a quadratic equation ax² + bx + c = 0, returns the constant term (c) -/
def constant_term (a b c : ℚ) : ℚ := c

theorem quadratic_equation_coefficients :
  let a : ℚ := 3
  let b : ℚ := -6
  let c : ℚ := -7
  quadratic_coefficient a b c = 3 ∧ constant_term a b c = -7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2229_222919


namespace NUMINAMATH_CALUDE_am_gm_positive_condition_l2229_222914

theorem am_gm_positive_condition (a b : ℝ) (h : a * b ≠ 0) :
  (a > 0 ∧ b > 0) ↔ ((a + b) / 2 ≥ Real.sqrt (a * b)) :=
sorry

end NUMINAMATH_CALUDE_am_gm_positive_condition_l2229_222914


namespace NUMINAMATH_CALUDE_olivias_papers_l2229_222988

/-- Given an initial number of papers and a number of papers used,
    calculate the remaining number of papers. -/
def remaining_papers (initial : ℕ) (used : ℕ) : ℕ :=
  initial - used

/-- Theorem stating that given 81 initial papers and 56 used papers,
    the remaining number is 25. -/
theorem olivias_papers :
  remaining_papers 81 56 = 25 := by
  sorry

end NUMINAMATH_CALUDE_olivias_papers_l2229_222988


namespace NUMINAMATH_CALUDE_problem_solution_l2229_222999

theorem problem_solution (x y : ℝ) (h : 3 * x - y ≤ Real.log (x + 2 * y - 3) + Real.log (2 * x - 3 * y + 5)) : 
  x + y = 16 / 7 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2229_222999


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l2229_222935

theorem geometric_sequence_ratio_sum (k p r : ℝ) (h1 : p ≠ 1) (h2 : r ≠ 1) (h3 : p ≠ r) 
  (h4 : k ≠ 0) (h5 : k * p^2 - k * r^2 = 3 * (k * p - k * r)) : p + r = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l2229_222935


namespace NUMINAMATH_CALUDE_basketball_tournament_l2229_222937

/-- The number of teams in the basketball tournament --/
def n : ℕ := 12

/-- The total number of matches played in the tournament --/
def total_matches (n : ℕ) : ℕ := n * (n - 1)

/-- The total number of points distributed in the tournament --/
def total_points (n : ℕ) : ℕ := 2 * total_matches n

/-- The number of teams scoring 24 points --/
def a (n : ℕ) : ℤ := n * (n - 1) - 11 * n + 33

/-- The number of teams scoring 22 points --/
def b (n : ℕ) : ℤ := -n^2 + 12 * n - 36

theorem basketball_tournament :
  (∃ (winner : ℕ) (last1 last2 : ℕ),
    winner = 26 ∧ 
    last1 = 20 ∧ 
    last2 = 20 ∧ 
    winner + last1 + last2 + 24 * (a n) + 22 * (b n) = total_points n) ∧
  a n ≥ 0 ∧
  b n ≥ 0 ∧
  a n + b n = n - 3 :=
by sorry

end NUMINAMATH_CALUDE_basketball_tournament_l2229_222937


namespace NUMINAMATH_CALUDE_sarah_candy_consumption_l2229_222998

theorem sarah_candy_consumption 
  (candy_from_neighbors : ℕ)
  (candy_from_sister : ℕ)
  (days_lasted : ℕ)
  (h1 : candy_from_neighbors = 66)
  (h2 : candy_from_sister = 15)
  (h3 : days_lasted = 9)
  (h4 : days_lasted > 0) :
  (candy_from_neighbors + candy_from_sister) / days_lasted = 9 :=
by sorry

end NUMINAMATH_CALUDE_sarah_candy_consumption_l2229_222998


namespace NUMINAMATH_CALUDE_specific_ellipse_foci_distance_l2229_222961

/-- An ellipse with axes parallel to the coordinate axes -/
structure ParallelAxisEllipse where
  /-- The point where the ellipse is tangent to the x-axis -/
  x_tangent : ℝ × ℝ
  /-- The point where the ellipse is tangent to the y-axis -/
  y_tangent : ℝ × ℝ

/-- The distance between the foci of an ellipse -/
def foci_distance (e : ParallelAxisEllipse) : ℝ := sorry

/-- Theorem stating the distance between foci for a specific ellipse -/
theorem specific_ellipse_foci_distance :
  ∃ (e : ParallelAxisEllipse),
    e.x_tangent = (5, 0) ∧
    e.y_tangent = (0, 2) ∧
    foci_distance e = 2 * Real.sqrt 21 :=
  sorry

end NUMINAMATH_CALUDE_specific_ellipse_foci_distance_l2229_222961


namespace NUMINAMATH_CALUDE_camping_trip_percentage_l2229_222904

theorem camping_trip_percentage (total_students : ℕ) 
  (march_trip_percentage : ℝ) (march_over_100_percentage : ℝ)
  (june_trip_percentage : ℝ) (june_over_100_percentage : ℝ)
  (over_100_march_percentage : ℝ) :
  march_trip_percentage = 0.2 →
  march_over_100_percentage = 0.35 →
  june_trip_percentage = 0.15 →
  june_over_100_percentage = 0.4 →
  over_100_march_percentage = 0.7 →
  (march_trip_percentage + june_trip_percentage) * total_students = 
    0.35 * total_students :=
by sorry

end NUMINAMATH_CALUDE_camping_trip_percentage_l2229_222904


namespace NUMINAMATH_CALUDE_smaller_k_implies_smaller_certainty_l2229_222996

/-- Represents the observed value of the random variable K² -/
def observed_value (k : ℝ) : Prop := k ≥ 0

/-- Represents the certainty of the relationship between categorical variables -/
def relationship_certainty (c : ℝ) : Prop := c ≥ 0 ∧ c ≤ 1

/-- Theorem stating the relationship between observed K² value and relationship certainty -/
theorem smaller_k_implies_smaller_certainty 
  (X Y : Type) [Finite X] [Finite Y] 
  (k₁ k₂ c₁ c₂ : ℝ) 
  (hk₁ : observed_value k₁) 
  (hk₂ : observed_value k₂) 
  (hc₁ : relationship_certainty c₁) 
  (hc₂ : relationship_certainty c₂) :
  k₁ < k₂ → c₁ < c₂ :=
sorry

end NUMINAMATH_CALUDE_smaller_k_implies_smaller_certainty_l2229_222996


namespace NUMINAMATH_CALUDE_base_number_proof_l2229_222968

theorem base_number_proof (x : ℝ) : 16^7 = x^14 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l2229_222968


namespace NUMINAMATH_CALUDE_expression_equals_three_l2229_222980

theorem expression_equals_three : 
  3⁻¹ + (Real.sqrt 2 - 1)^0 + 2 * Real.sin (30 * π / 180) - (-2/3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_three_l2229_222980


namespace NUMINAMATH_CALUDE_pen_cost_l2229_222951

theorem pen_cost (cost : ℝ) (has : ℝ) (needs : ℝ) : 
  has = cost / 3 → needs = 20 → has + needs = cost → cost = 30 := by
  sorry

end NUMINAMATH_CALUDE_pen_cost_l2229_222951


namespace NUMINAMATH_CALUDE_math_marks_proof_l2229_222979

/-- Calculates the marks in Mathematics given marks in other subjects and the average -/
def calculate_math_marks (english physics chemistry biology average : ℕ) : ℕ :=
  5 * average - (english + physics + chemistry + biology)

theorem math_marks_proof (english physics chemistry biology average : ℕ) 
  (h_english : english = 96)
  (h_physics : physics = 82)
  (h_chemistry : chemistry = 97)
  (h_biology : biology = 95)
  (h_average : average = 93) :
  calculate_math_marks english physics chemistry biology average = 95 := by
  sorry

end NUMINAMATH_CALUDE_math_marks_proof_l2229_222979


namespace NUMINAMATH_CALUDE_wendy_bags_proof_l2229_222962

/-- The number of points Wendy earns per bag of cans recycled -/
def points_per_bag : ℕ := 5

/-- The number of bags Wendy didn't recycle -/
def unrecycled_bags : ℕ := 2

/-- The total points Wendy would earn if she recycled all but 2 bags -/
def total_points : ℕ := 45

/-- The initial number of bags Wendy had -/
def initial_bags : ℕ := 11

theorem wendy_bags_proof :
  points_per_bag * (initial_bags - unrecycled_bags) = total_points :=
by sorry

end NUMINAMATH_CALUDE_wendy_bags_proof_l2229_222962


namespace NUMINAMATH_CALUDE_joe_fruit_probability_l2229_222958

/-- The number of meals in a day -/
def num_meals : ℕ := 3

/-- The number of fruit types available -/
def num_fruits : ℕ := 4

/-- The probability of choosing a specific fruit at any meal -/
def prob_fruit : ℚ := 1 / num_fruits

/-- The probability of eating the same fruit for all meals -/
def prob_same_fruit : ℚ := num_fruits * (prob_fruit ^ num_meals)

theorem joe_fruit_probability :
  1 - prob_same_fruit = 15 / 16 :=
sorry

end NUMINAMATH_CALUDE_joe_fruit_probability_l2229_222958


namespace NUMINAMATH_CALUDE_average_of_solutions_eq_neg_two_thirds_l2229_222986

theorem average_of_solutions_eq_neg_two_thirds : 
  let f (x : ℝ) := 3 * x^2 + 4 * x + 1
  let solutions := {x : ℝ | f x = 28}
  ∃ (x₁ x₂ : ℝ), solutions = {x₁, x₂} ∧ (x₁ + x₂) / 2 = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_solutions_eq_neg_two_thirds_l2229_222986


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2229_222918

theorem complex_equation_solution (a b : ℝ) (h : (Complex.I + a) * (1 + Complex.I) = b * Complex.I) : 
  Complex.mk a b = 1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2229_222918


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2229_222964

theorem perfect_square_condition (m : ℝ) : 
  (∃ x : ℝ, ∃ k : ℝ, x^2 + 2*(m-3)*x + 16 = k^2) → (m = 7 ∨ m = -1) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2229_222964


namespace NUMINAMATH_CALUDE_bond_return_rate_l2229_222995

-- Define the total investment
def total_investment : ℝ := 10000

-- Define the bank interest rate
def bank_interest_rate : ℝ := 0.05

-- Define the total annual income
def total_annual_income : ℝ := 660

-- Define the amount invested in each method
def investment_per_method : ℝ := 6000

-- Theorem statement
theorem bond_return_rate :
  let bank_income := investment_per_method * bank_interest_rate
  let bond_income := total_annual_income - bank_income
  bond_income / investment_per_method = 0.06 := by
sorry


end NUMINAMATH_CALUDE_bond_return_rate_l2229_222995


namespace NUMINAMATH_CALUDE_soup_problem_solution_l2229_222973

/-- Represents the number of people a can of soup can feed -/
structure CanCapacity where
  adults : Nat
  children : Nat

/-- Represents the problem setup -/
structure SoupProblem where
  capacity : CanCapacity
  totalCans : Nat
  childrenFed : Nat

/-- Calculates the number of adults that can be fed with the remaining soup -/
def remainingAdults (problem : SoupProblem) : Nat :=
  let cansUsedForChildren := problem.childrenFed / problem.capacity.children
  let remainingCans := problem.totalCans - cansUsedForChildren
  remainingCans * problem.capacity.adults

/-- Proves that given the problem conditions, 16 adults can be fed with the remaining soup -/
theorem soup_problem_solution (problem : SoupProblem) 
  (h1 : problem.capacity = ⟨4, 6⟩) 
  (h2 : problem.totalCans = 8) 
  (h3 : problem.childrenFed = 24) : 
  remainingAdults problem = 16 := by
  sorry

#eval remainingAdults ⟨⟨4, 6⟩, 8, 24⟩

end NUMINAMATH_CALUDE_soup_problem_solution_l2229_222973


namespace NUMINAMATH_CALUDE_total_winter_clothing_l2229_222960

-- Define a structure for a box of winter clothing
structure WinterClothingBox where
  scarves : Nat
  mittens : Nat
  hats : Nat

-- Define the contents of each box
def box1 : WinterClothingBox := ⟨2, 3, 1⟩
def box2 : WinterClothingBox := ⟨4, 2, 2⟩
def box3 : WinterClothingBox := ⟨1, 5, 3⟩
def box4 : WinterClothingBox := ⟨3, 4, 1⟩
def box5 : WinterClothingBox := ⟨5, 3, 2⟩
def box6 : WinterClothingBox := ⟨2, 6, 0⟩
def box7 : WinterClothingBox := ⟨4, 1, 3⟩
def box8 : WinterClothingBox := ⟨3, 2, 4⟩
def box9 : WinterClothingBox := ⟨1, 4, 5⟩

-- Define a function to count items in a box
def countItems (box : WinterClothingBox) : Nat :=
  box.scarves + box.mittens + box.hats

-- Theorem statement
theorem total_winter_clothing :
  countItems box1 + countItems box2 + countItems box3 +
  countItems box4 + countItems box5 + countItems box6 +
  countItems box7 + countItems box8 + countItems box9 = 76 := by
  sorry

end NUMINAMATH_CALUDE_total_winter_clothing_l2229_222960


namespace NUMINAMATH_CALUDE_c_profit_share_is_400_l2229_222982

/-- Represents an investor in the business --/
structure Investor where
  name : String
  investment : ℕ

/-- Represents the business venture --/
structure Business where
  investors : List Investor
  duration : ℕ
  total_profit : ℕ

/-- Calculates an investor's share of the profit --/
def profit_share (b : Business) (i : Investor) : ℕ :=
  let total_investment := b.investors.map (·.investment) |>.sum
  (i.investment * b.total_profit) / total_investment

theorem c_profit_share_is_400 (b : Business) (c : Investor) :
  b.investors = [⟨"a", 800⟩, ⟨"b", 1000⟩, c] →
  b.duration = 2 →
  b.total_profit = 1000 →
  c.investment = 1200 →
  profit_share b c = 400 := by
  sorry

#eval profit_share
  ⟨[⟨"a", 800⟩, ⟨"b", 1000⟩, ⟨"c", 1200⟩], 2, 1000⟩
  ⟨"c", 1200⟩

end NUMINAMATH_CALUDE_c_profit_share_is_400_l2229_222982


namespace NUMINAMATH_CALUDE_mrs_hilt_hot_dog_cost_l2229_222926

/-- The total cost of hot dogs in cents -/
def total_cost (num_hot_dogs : ℕ) (cost_per_hot_dog : ℕ) : ℕ :=
  num_hot_dogs * cost_per_hot_dog

/-- Theorem: Mrs. Hilt's total cost for hot dogs is 300 cents -/
theorem mrs_hilt_hot_dog_cost :
  total_cost 6 50 = 300 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_hot_dog_cost_l2229_222926


namespace NUMINAMATH_CALUDE_consecutive_days_sum_l2229_222971

theorem consecutive_days_sum (x : ℕ) : 
  x + (x + 1) + (x + 2) = 33 → x = 10 ∧ x + 1 = 11 ∧ x + 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_days_sum_l2229_222971


namespace NUMINAMATH_CALUDE_intersection_range_distance_when_b_is_one_l2229_222955

/-- The line y = x + b intersects the ellipse x^2/2 + y^2 = 1 at two distinct points -/
def intersects_at_two_points (b : ℝ) : Prop :=
  ∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    y₁ = x₁ + b ∧ y₂ = x₂ + b ∧
    x₁^2/2 + y₁^2 = 1 ∧ x₂^2/2 + y₂^2 = 1

/-- The range of b for which the line intersects the ellipse at two distinct points -/
theorem intersection_range :
  ∀ b : ℝ, intersects_at_two_points b ↔ -Real.sqrt 3 < b ∧ b < Real.sqrt 3 :=
sorry

/-- The distance between intersection points when b = 1 -/
theorem distance_when_b_is_one :
  ∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    y₁ = x₁ + 1 ∧ y₂ = x₂ + 1 ∧
    x₁^2/2 + y₁^2 = 1 ∧ x₂^2/2 + y₂^2 = 1 ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 4 * Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_distance_when_b_is_one_l2229_222955


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l2229_222993

theorem chess_tournament_participants (n : ℕ) (h : n > 0) : 
  (n * (n - 1)) / 2 = 120 → n = 16 := by sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l2229_222993


namespace NUMINAMATH_CALUDE_burn_all_bridges_probability_l2229_222989

/-- The number of islands in the lake -/
def num_islands : ℕ := 2013

/-- The probability of choosing a new bridge at each step -/
def prob_new_bridge : ℚ := 2/3

/-- The probability of burning all bridges -/
def prob_burn_all : ℚ := num_islands * prob_new_bridge ^ (num_islands - 1)

/-- Theorem stating the probability of burning all bridges -/
theorem burn_all_bridges_probability :
  prob_burn_all = num_islands * (2/3) ^ (num_islands - 1) := by sorry

end NUMINAMATH_CALUDE_burn_all_bridges_probability_l2229_222989


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l2229_222931

/-- Given a hyperbola E and a parabola C with specific properties, 
    prove that the eccentricity of E is in the range (1, 3√2/4] -/
theorem hyperbola_eccentricity_range 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (E : Set (ℝ × ℝ)) 
  (C : Set (ℝ × ℝ)) 
  (A : ℝ × ℝ) 
  (F : ℝ × ℝ) 
  (P : ℝ × ℝ) 
  (hE : E = {(x, y) | x^2/a^2 - y^2/b^2 = 1})
  (hC : C = {(x, y) | y^2 = 8*a*x})
  (hA : A = (a, 0))
  (hF : F = (2*a, 0))
  (hP : P ∈ {(x, y) | y = (b/a)*x})  -- P is on the asymptote of E
  (hPerp : (P.1 - A.1) * (P.1 - F.1) + (P.2 - A.2) * (P.2 - F.2) = 0)  -- AP ⊥ FP
  : 1 < Real.sqrt (a^2 + b^2) / a ∧ Real.sqrt (a^2 + b^2) / a ≤ 3 * Real.sqrt 2 / 4 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l2229_222931


namespace NUMINAMATH_CALUDE_variance_of_data_l2229_222934

def data : List ℝ := [3, 2, 1, 0, 0, 0, 1]

theorem variance_of_data : 
  let n : ℝ := data.length
  let mean := (data.sum) / n
  let variance := (data.map (fun x => (x - mean)^2)).sum / n
  variance = 8/7 := by sorry

end NUMINAMATH_CALUDE_variance_of_data_l2229_222934


namespace NUMINAMATH_CALUDE_expand_product_l2229_222920

theorem expand_product (x : ℝ) : (3 * x + 4) * (2 * x - 6) = 6 * x^2 - 10 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2229_222920


namespace NUMINAMATH_CALUDE_crayon_division_l2229_222908

theorem crayon_division (total_crayons : ℕ) (num_people : ℕ) (crayons_per_person : ℕ) :
  total_crayons = 24 →
  num_people = 3 →
  total_crayons = num_people * crayons_per_person →
  crayons_per_person = 8 :=
by sorry

end NUMINAMATH_CALUDE_crayon_division_l2229_222908


namespace NUMINAMATH_CALUDE_average_weight_increase_l2229_222965

/-- Proves that replacing a person weighing 60 kg with a person weighing 80 kg
    in a group of 8 people increases the average weight by 2.5 kg -/
theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total := 8 * initial_average
  let new_total := initial_total - 60 + 80
  let new_average := new_total / 8
  new_average - initial_average = 2.5 := by sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2229_222965


namespace NUMINAMATH_CALUDE_power_tower_mod_2000_l2229_222912

theorem power_tower_mod_2000 : 7^(7^(7^7)) ≡ 343 [ZMOD 2000] := by sorry

end NUMINAMATH_CALUDE_power_tower_mod_2000_l2229_222912


namespace NUMINAMATH_CALUDE_no_solution_inequality_l2229_222913

theorem no_solution_inequality (m : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 5| > m) ↔ m < 6 := by sorry

end NUMINAMATH_CALUDE_no_solution_inequality_l2229_222913
