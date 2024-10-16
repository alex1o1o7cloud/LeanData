import Mathlib

namespace NUMINAMATH_CALUDE_lcm_sum_ratio_problem_l1679_167998

theorem lcm_sum_ratio_problem (A B x y : ℕ+) : 
  Nat.lcm A B = 60 →
  A + B = 50 →
  x > y →
  A * y = B * x →
  x = 3 ∧ y = 2 := by sorry

end NUMINAMATH_CALUDE_lcm_sum_ratio_problem_l1679_167998


namespace NUMINAMATH_CALUDE_ratio_abc_l1679_167914

theorem ratio_abc (a b c : ℝ) (ha : a ≠ 0) 
  (h : 14 * (a^2 + b^2 + c^2) = (a + 2*b + 3*c)^2) : 
  ∃ (k : ℝ), k ≠ 0 ∧ a = k ∧ b = 2*k ∧ c = 3*k := by
  sorry

end NUMINAMATH_CALUDE_ratio_abc_l1679_167914


namespace NUMINAMATH_CALUDE_stream_speed_l1679_167949

/-- Given a river with stream speed v and a rower with speed u in still water,
    if the rower travels 27 km upstream and 81 km downstream, each in 9 hours,
    then the speed of the stream v is 3 km/h. -/
theorem stream_speed (v u : ℝ) 
  (h1 : 27 / (u - v) = 9)  -- Upstream condition
  (h2 : 81 / (u + v) = 9)  -- Downstream condition
  : v = 3 := by
  sorry


end NUMINAMATH_CALUDE_stream_speed_l1679_167949


namespace NUMINAMATH_CALUDE_exponent_multiplication_l1679_167981

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l1679_167981


namespace NUMINAMATH_CALUDE_box_capacity_l1679_167937

-- Define the volumes and capacities
def small_box_volume : ℝ := 24
def small_box_paperclips : ℕ := 60
def large_box_volume : ℝ := 72
def large_box_staples : ℕ := 90

-- Define the theorem
theorem box_capacity :
  ∃ (large_box_paperclips large_box_mixed_staples : ℕ),
    large_box_paperclips = 90 ∧ 
    large_box_mixed_staples = 45 ∧
    (large_box_paperclips : ℝ) / (large_box_volume / 2) = (small_box_paperclips : ℝ) / small_box_volume ∧
    (large_box_mixed_staples : ℝ) / (large_box_volume / 2) = (large_box_staples : ℝ) / large_box_volume :=
by
  sorry

end NUMINAMATH_CALUDE_box_capacity_l1679_167937


namespace NUMINAMATH_CALUDE_double_march_earnings_cars_l1679_167941

/-- Represents the earnings of a car salesman -/
structure CarSalesmanEarnings where
  baseSalary : ℕ
  commissionPerCar : ℕ
  marchEarnings : ℕ

/-- Calculates the number of cars needed to be sold to reach a target earning -/
def carsNeededForTarget (e : CarSalesmanEarnings) (targetEarnings : ℕ) : ℕ :=
  ((targetEarnings - e.baseSalary) + e.commissionPerCar - 1) / e.commissionPerCar

/-- Theorem: The number of cars needed to double March earnings is 15 -/
theorem double_march_earnings_cars (e : CarSalesmanEarnings) 
    (h1 : e.baseSalary = 1000)
    (h2 : e.commissionPerCar = 200)
    (h3 : e.marchEarnings = 2000) : 
    carsNeededForTarget e (2 * e.marchEarnings) = 15 := by
  sorry

end NUMINAMATH_CALUDE_double_march_earnings_cars_l1679_167941


namespace NUMINAMATH_CALUDE_distribute_negative_two_over_parentheses_l1679_167954

theorem distribute_negative_two_over_parentheses (x : ℝ) : -2 * (x - 3) = -2 * x + 6 := by
  sorry

end NUMINAMATH_CALUDE_distribute_negative_two_over_parentheses_l1679_167954


namespace NUMINAMATH_CALUDE_hari_joined_after_five_months_l1679_167974

/-- Represents the business scenario with two partners --/
structure Business where
  praveen_investment : ℕ
  hari_investment : ℕ
  profit_ratio_praveen : ℕ
  profit_ratio_hari : ℕ
  total_duration : ℕ

/-- Calculates the number of months after which Hari joined the business --/
def months_until_hari_joined (b : Business) : ℕ :=
  let x := b.total_duration - (b.praveen_investment * b.total_duration * b.profit_ratio_hari) / 
           (b.hari_investment * b.profit_ratio_praveen)
  x

/-- Theorem stating that Hari joined 5 months after Praveen started the business --/
theorem hari_joined_after_five_months (b : Business) 
  (h1 : b.praveen_investment = 3220)
  (h2 : b.hari_investment = 8280)
  (h3 : b.profit_ratio_praveen = 2)
  (h4 : b.profit_ratio_hari = 3)
  (h5 : b.total_duration = 12) :
  months_until_hari_joined b = 5 := by
  sorry

#eval months_until_hari_joined ⟨3220, 8280, 2, 3, 12⟩

end NUMINAMATH_CALUDE_hari_joined_after_five_months_l1679_167974


namespace NUMINAMATH_CALUDE_angle_C_measure_triangle_perimeter_l1679_167987

-- Define the right triangle ABC
structure RightTriangle where
  A : Real
  B : Real
  C : Real
  AB : Real
  BC : Real
  AC : Real
  right_angle : C = 90
  angle_sum : A + B + C = 180

-- Define the given condition
def tan_condition (t : RightTriangle) : Prop :=
  Real.tan t.A + Real.tan t.B + Real.tan t.A * Real.tan t.B = 1

-- Theorem for part 1
theorem angle_C_measure (t : RightTriangle) (h : tan_condition t) : t.C = 135 := by
  sorry

-- Theorem for part 2
theorem triangle_perimeter 
  (t : RightTriangle) 
  (h1 : tan_condition t) 
  (h2 : t.A = 15) 
  (h3 : t.AB = Real.sqrt 2) : 
  t.AB + t.BC + t.AC = (2 + Real.sqrt 6 + Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_triangle_perimeter_l1679_167987


namespace NUMINAMATH_CALUDE_rat_digging_difference_l1679_167912

/-- The distance dug by the large rat after n days -/
def large_rat_distance (n : ℕ) : ℚ := 2^n - 1

/-- The distance dug by the small rat after n days -/
def small_rat_distance (n : ℕ) : ℚ := 2 - 1 / 2^(n-1)

/-- The difference in distance dug between the large and small rat after n days -/
def distance_difference (n : ℕ) : ℚ := large_rat_distance n - small_rat_distance n

theorem rat_digging_difference :
  distance_difference 5 = 29 / 16 := by sorry

end NUMINAMATH_CALUDE_rat_digging_difference_l1679_167912


namespace NUMINAMATH_CALUDE_lawnmower_initial_price_l1679_167957

/-- Proves that the initial price of a lawnmower was $100 given specific depreciation rates and final value -/
theorem lawnmower_initial_price (initial_price : ℝ) : 
  let price_after_six_months := initial_price * 0.75
  let final_price := price_after_six_months * 0.8
  final_price = 60 →
  initial_price = 100 := by
  sorry

end NUMINAMATH_CALUDE_lawnmower_initial_price_l1679_167957


namespace NUMINAMATH_CALUDE_probability_two_non_defective_pens_l1679_167917

theorem probability_two_non_defective_pens (total_pens : ℕ) (defective_pens : ℕ) 
  (h1 : total_pens = 10) (h2 : defective_pens = 3) :
  let non_defective := total_pens - defective_pens
  let prob_first := non_defective / total_pens
  let prob_second := (non_defective - 1) / (total_pens - 1)
  prob_first * prob_second = 7 / 15 := by sorry

end NUMINAMATH_CALUDE_probability_two_non_defective_pens_l1679_167917


namespace NUMINAMATH_CALUDE_smallest_base_for_120_l1679_167962

theorem smallest_base_for_120 : ∃ (b : ℕ), b = 5 ∧ b^2 ≤ 120 ∧ 120 < b^3 ∧ ∀ (x : ℕ), x < b → (x^2 ≤ 120 → 120 ≥ x^3) := by
  sorry

end NUMINAMATH_CALUDE_smallest_base_for_120_l1679_167962


namespace NUMINAMATH_CALUDE_chime_time_at_12_l1679_167968

/-- Represents a clock with hourly chimes -/
structure ChimeClock where
  /-- The time in seconds it takes to complete chimes at 4 o'clock -/
  time_at_4 : ℕ
  /-- Assertion that the clock chimes once every hour -/
  chimes_hourly : Prop

/-- Calculates the time it takes to complete chimes at a given hour -/
def chime_time (clock : ChimeClock) (hour : ℕ) : ℕ :=
  sorry

/-- Theorem stating that it takes 44 seconds to complete chimes at 12 o'clock -/
theorem chime_time_at_12 (clock : ChimeClock) 
  (h1 : clock.time_at_4 = 12) 
  (h2 : clock.chimes_hourly) : 
  chime_time clock 12 = 44 :=
sorry

end NUMINAMATH_CALUDE_chime_time_at_12_l1679_167968


namespace NUMINAMATH_CALUDE_final_student_count_l1679_167930

def initial_students : ℕ := 31
def students_left : ℕ := 5
def new_students : ℕ := 11

theorem final_student_count : 
  initial_students - students_left + new_students = 37 := by
  sorry

end NUMINAMATH_CALUDE_final_student_count_l1679_167930


namespace NUMINAMATH_CALUDE_salary_change_l1679_167923

theorem salary_change (original_salary : ℝ) (h : original_salary > 0) :
  let increased_salary := original_salary * 1.15
  let final_salary := increased_salary * 0.85
  let net_change := (final_salary - original_salary) / original_salary
  net_change = -0.0225 := by
sorry

end NUMINAMATH_CALUDE_salary_change_l1679_167923


namespace NUMINAMATH_CALUDE_half_minus_quarter_equals_two_l1679_167925

theorem half_minus_quarter_equals_two (n : ℝ) : n = 8 → (0.5 * n) - (0.25 * n) = 2 := by
  sorry

end NUMINAMATH_CALUDE_half_minus_quarter_equals_two_l1679_167925


namespace NUMINAMATH_CALUDE_conditional_probability_fair_die_l1679_167967

-- Define the sample space
def S : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define events A and B
def A : Finset Nat := {2, 3, 5}
def B : Finset Nat := {1, 2, 4, 5, 6}

-- Define the probability measure
def P (X : Finset Nat) : ℚ := (X.card : ℚ) / (S.card : ℚ)

-- Define the intersection of events
def AB : Finset Nat := A ∩ B

-- Theorem statement
theorem conditional_probability_fair_die :
  P AB / P B = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_fair_die_l1679_167967


namespace NUMINAMATH_CALUDE_book_sales_l1679_167910

theorem book_sales (wednesday_sales : ℕ) : 
  wednesday_sales + 3 * wednesday_sales + 3 * wednesday_sales / 5 = 69 → 
  wednesday_sales = 15 := by
sorry

end NUMINAMATH_CALUDE_book_sales_l1679_167910


namespace NUMINAMATH_CALUDE_ellipse_proof_hyperbola_proof_l1679_167948

-- Ellipse
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 36 + y^2 / 20 = 1

theorem ellipse_proof (major_axis_length : ℝ) (eccentricity : ℝ) 
  (h1 : major_axis_length = 12) 
  (h2 : eccentricity = 2/3) : 
  ∀ x y : ℝ, ellipse_equation x y ↔ 
    ∃ a b : ℝ, a^2 * y^2 + b^2 * x^2 = a^2 * b^2 ∧ 
    2 * a = major_axis_length ∧ 
    (a^2 - b^2) / a^2 = eccentricity^2 :=
sorry

-- Hyperbola
def original_hyperbola (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

def new_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 24 = 1

theorem hyperbola_proof :
  ∀ x y : ℝ, new_hyperbola x y ↔ 
    (∃ c : ℝ, (∀ x₀ y₀ : ℝ, original_hyperbola x₀ y₀ → 
      (x₀ - c)^2 - y₀^2 = c^2 ∧ (x₀ + c)^2 - y₀^2 = c^2) ∧
    new_hyperbola (-Real.sqrt 5 / 2) (-Real.sqrt 6)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_proof_hyperbola_proof_l1679_167948


namespace NUMINAMATH_CALUDE_barbara_candy_distribution_l1679_167939

/-- Represents the candy distribution problem --/
structure CandyProblem where
  original_candies : Nat
  bought_candies : Nat
  num_friends : Nat

/-- Calculates the number of candies each friend receives --/
def candies_per_friend (problem : CandyProblem) : Nat :=
  (problem.original_candies + problem.bought_candies) / problem.num_friends

/-- Theorem stating that each friend receives 4 candies --/
theorem barbara_candy_distribution :
  ∀ (problem : CandyProblem),
    problem.original_candies = 9 →
    problem.bought_candies = 18 →
    problem.num_friends = 6 →
    candies_per_friend problem = 4 :=
by
  sorry

#eval candies_per_friend { original_candies := 9, bought_candies := 18, num_friends := 6 }

end NUMINAMATH_CALUDE_barbara_candy_distribution_l1679_167939


namespace NUMINAMATH_CALUDE_simplify_expression_l1679_167938

theorem simplify_expression (a : ℝ) : ((4 * a + 6) - 7 * a) / 3 = -a + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1679_167938


namespace NUMINAMATH_CALUDE_range_of_a_l1679_167944

def A (a : ℝ) : Set ℝ := {x : ℝ | (x - 1) * (x - a) ≥ 0}

def B (a : ℝ) : Set ℝ := {x : ℝ | x ≥ a - 1}

theorem range_of_a (a : ℝ) (h : A a ∪ B a = Set.univ) : a ≤ 2 := by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_range_of_a_l1679_167944


namespace NUMINAMATH_CALUDE_differential_y_differential_F_differential_z_dz_at_zero_l1679_167902

noncomputable section

-- Function definitions
def y (x : ℝ) := x^3 - 3^x
def F (φ : ℝ) := Real.cos (φ/3) + Real.sin (3/φ)
def z (x : ℝ) := Real.log (1 + Real.exp (10*x)) + Real.arctan (Real.exp (5*x))⁻¹

-- Theorem statements
theorem differential_y (x : ℝ) :
  deriv y x = 3*x^2 - 3^x * Real.log 3 :=
sorry

theorem differential_F (φ : ℝ) (h : φ ≠ 0) :
  deriv F φ = -1/3 * Real.sin (φ/3) - 3 * Real.cos (3/φ) / φ^2 :=
sorry

theorem differential_z (x : ℝ) :
  deriv z x = (5 * Real.exp (5*x) * (2 * Real.exp (5*x) - 1)) / (1 + Real.exp (10*x)) :=
sorry

theorem dz_at_zero :
  (deriv z 0) * 0.1 = 0.25 :=
sorry

end

end NUMINAMATH_CALUDE_differential_y_differential_F_differential_z_dz_at_zero_l1679_167902


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_7_mod_21_l1679_167995

theorem least_five_digit_congruent_to_7_mod_21 :
  ∀ n : ℕ, 
    n ≥ 10000 ∧ n ≤ 99999 ∧ n % 21 = 7 → n ≥ 10003 :=
by
  sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_7_mod_21_l1679_167995


namespace NUMINAMATH_CALUDE_max_value_of_f_range_of_t_inequality_for_positive_reals_l1679_167969

-- Define the function f(x) = |x+1| - |x-2|
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Theorem 1: The maximum value of f(x) is 3
theorem max_value_of_f : ∀ x : ℝ, f x ≤ 3 :=
sorry

-- Theorem 2: The range of t given the inequality
theorem range_of_t : ∀ t : ℝ, (∃ x : ℝ, f x ≥ |t - 1| + t) ↔ t ≤ 2 :=
sorry

-- Theorem 3: Inequality for positive real numbers
theorem inequality_for_positive_reals :
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 2*a + b + c = 2 → a^2 + b^2 + c^2 ≥ 2/3 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_range_of_t_inequality_for_positive_reals_l1679_167969


namespace NUMINAMATH_CALUDE_odometer_problem_l1679_167971

theorem odometer_problem (a b c : ℕ) (n : ℕ+) :
  100 ≤ 100 * a + 10 * b + c →
  100 * a + 10 * b + c ≤ 999 →
  a ≥ 1 →
  a + b + c ≤ 7 →
  100 * c + 10 * b + a - (100 * a + 10 * b + c) = 55 * n →
  a^2 + b^2 + c^2 = 37 := by
sorry

end NUMINAMATH_CALUDE_odometer_problem_l1679_167971


namespace NUMINAMATH_CALUDE_trajectory_of_Q_l1679_167991

/-- The trajectory of a point Q derived from a point P on a unit circle. -/
theorem trajectory_of_Q (x y u v : ℝ) : 
  (x^2 + y^2 = 1) →  -- P is on the unit circle
  (u = x + y) →      -- First coordinate of Q
  (v = x * y) →      -- Second coordinate of Q
  (u^2 - 2*v = 1)    -- Equation of a parabola
  := by sorry

end NUMINAMATH_CALUDE_trajectory_of_Q_l1679_167991


namespace NUMINAMATH_CALUDE_car_rental_rates_equal_l1679_167961

/-- The daily rate of Sunshine Car Rentals -/
def sunshine_daily_rate : ℝ := 17.99

/-- The per-mile rate of Sunshine Car Rentals -/
def sunshine_mile_rate : ℝ := 0.18

/-- The per-mile rate of the second car rental company -/
def second_company_mile_rate : ℝ := 0.16

/-- The number of miles driven -/
def miles_driven : ℝ := 48

/-- The daily rate of the second car rental company -/
def second_company_daily_rate : ℝ := 18.95

theorem car_rental_rates_equal :
  sunshine_daily_rate + sunshine_mile_rate * miles_driven =
  second_company_daily_rate + second_company_mile_rate * miles_driven :=
by sorry

end NUMINAMATH_CALUDE_car_rental_rates_equal_l1679_167961


namespace NUMINAMATH_CALUDE_smallest_w_value_l1679_167965

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w_value : 
  ∃ w : ℕ, w > 0 ∧ 
    is_factor (2^7) (936 * w) ∧
    is_factor (3^4) (936 * w) ∧
    is_factor (5^3) (936 * w) ∧
    is_factor (7^2) (936 * w) ∧
    is_factor (11^2) (936 * w) ∧
    (∀ v : ℕ, v > 0 ∧ 
      is_factor (2^7) (936 * v) ∧
      is_factor (3^4) (936 * v) ∧
      is_factor (5^3) (936 * v) ∧
      is_factor (7^2) (936 * v) ∧
      is_factor (11^2) (936 * v) → 
      w ≤ v) ∧
    w = 320166000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_w_value_l1679_167965


namespace NUMINAMATH_CALUDE_derricks_yard_length_l1679_167956

theorem derricks_yard_length :
  ∀ (derrick_length alex_length brianne_length : ℝ),
    brianne_length = 30 →
    alex_length = derrick_length / 2 →
    brianne_length = 6 * alex_length →
    derrick_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_derricks_yard_length_l1679_167956


namespace NUMINAMATH_CALUDE_sequence_properties_l1679_167986

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem sequence_properties
  (a : ℕ → ℕ)
  (h_increasing : ∀ n, a n < a (n + 1))
  (h_positive : ∀ n, 0 < a n)
  (b : ℕ → ℕ)
  (h_b : ∀ n, b n = a (a n))
  (c : ℕ → ℕ)
  (h_c : ∀ n, c n = a (a (n + 1)))
  (h_b_value : ∀ n, b n = 3 * n)
  (h_c_arithmetic : is_arithmetic_sequence c ∧ ∀ n, c (n + 1) = c n + 1) :
  a 1 = 2 ∧ c 1 = 6 ∧ is_arithmetic_sequence a :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1679_167986


namespace NUMINAMATH_CALUDE_find_a_l1679_167984

def U (a : ℤ) : Set ℤ := {2, 4, a^2 - a + 1}

def A (a : ℤ) : Set ℤ := {a+4, 4}

def complement_A (a : ℤ) : Set ℤ := {7}

theorem find_a : ∃ a : ℤ, 
  (U a = {2, 4, a^2 - a + 1}) ∧ 
  (A a = {a+4, 4}) ∧ 
  (complement_A a = {7}) ∧
  (Set.inter (A a) (complement_A a) = ∅) ∧
  (Set.union (A a) (complement_A a) = U a) ∧
  (a = -2) := by sorry

end NUMINAMATH_CALUDE_find_a_l1679_167984


namespace NUMINAMATH_CALUDE_tangent_ratio_bounds_l1679_167994

noncomputable def f (x : ℝ) : ℝ := |Real.exp x - 1|

theorem tangent_ratio_bounds (x₁ x₂ : ℝ) (h₁ : x₁ < 0) (h₂ : x₂ > 0) :
  let A := (x₁, f x₁)
  let B := (x₂, f x₂)
  let M := (0, (1 - Real.exp x₁) + x₁ * Real.exp x₁)
  let N := (0, (Real.exp x₂ - 1) - x₂ * Real.exp x₂)
  let tangent_slope_A := -Real.exp x₁
  let tangent_slope_B := Real.exp x₂
  tangent_slope_A * tangent_slope_B = -1 →
  let AM := Real.sqrt ((x₁ - 0)^2 + (f x₁ - M.2)^2)
  let BN := Real.sqrt ((x₂ - 0)^2 + (f x₂ - N.2)^2)
  0 < AM / BN ∧ AM / BN < 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_ratio_bounds_l1679_167994


namespace NUMINAMATH_CALUDE_vertex_coordinates_l1679_167929

def f (x : ℝ) := (x - 1)^2 - 2

theorem vertex_coordinates :
  ∃ (x y : ℝ), (x = 1 ∧ y = -2) ∧
  ∀ (t : ℝ), f t ≥ f x :=
by sorry

end NUMINAMATH_CALUDE_vertex_coordinates_l1679_167929


namespace NUMINAMATH_CALUDE_michael_saved_five_cookies_l1679_167926

/-- The number of cookies Michael saved to give Sarah -/
def michaels_cookies (sarahs_initial_cupcakes : ℕ) (sarahs_final_desserts : ℕ) : ℕ :=
  sarahs_final_desserts - (sarahs_initial_cupcakes - sarahs_initial_cupcakes / 3)

theorem michael_saved_five_cookies :
  michaels_cookies 9 11 = 5 :=
by sorry

end NUMINAMATH_CALUDE_michael_saved_five_cookies_l1679_167926


namespace NUMINAMATH_CALUDE_initially_calculated_average_of_class_l1679_167901

/-- The initially calculated average height of a class of boys -/
def initially_calculated_average (num_boys : ℕ) (actual_average : ℚ) (initial_error : ℕ) : ℚ :=
  actual_average + (initial_error : ℚ) / num_boys

/-- Theorem stating the initially calculated average height -/
theorem initially_calculated_average_of_class (num_boys : ℕ) (actual_average : ℚ) (initial_error : ℕ) 
  (h1 : num_boys = 35)
  (h2 : actual_average = 178)
  (h3 : initial_error = 50) :
  initially_calculated_average num_boys actual_average initial_error = 179 + 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_initially_calculated_average_of_class_l1679_167901


namespace NUMINAMATH_CALUDE_percentage_of_democrat_voters_l1679_167952

theorem percentage_of_democrat_voters (d r : ℝ) : 
  d + r = 100 →
  0.65 * d + 0.2 * r = 47 →
  d = 60 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_democrat_voters_l1679_167952


namespace NUMINAMATH_CALUDE_derivative_zero_sufficient_not_necessary_for_extremum_l1679_167970

-- Define a differentiable function f
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define the property of having an extremum at a point
def HasExtremumAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x, |x - x₀| < ε → f x ≤ f x₀ ∨ ∀ x, |x - x₀| < ε → f x ≥ f x₀

-- State the theorem
theorem derivative_zero_sufficient_not_necessary_for_extremum :
  (∃ x₀ : ℝ, deriv f x₀ = 0 → HasExtremumAt f x₀) ∧
  (∃ x₀ : ℝ, HasExtremumAt f x₀ ∧ deriv f x₀ ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_derivative_zero_sufficient_not_necessary_for_extremum_l1679_167970


namespace NUMINAMATH_CALUDE_third_number_in_set_l1679_167975

theorem third_number_in_set (x : ℝ) : 
  let set1 := [10, 70, 28]
  let set2 := [20, 40, x]
  (set2.sum / set2.length : ℝ) = (set1.sum / set1.length : ℝ) + 4 →
  x = 60 := by
sorry

end NUMINAMATH_CALUDE_third_number_in_set_l1679_167975


namespace NUMINAMATH_CALUDE_no_function_satisfies_condition_l1679_167988

theorem no_function_satisfies_condition : ¬∃ f : ℝ → ℝ, ∀ x y : ℝ, f x * f y - f (x * y) = 2 * x + 2 * y := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_condition_l1679_167988


namespace NUMINAMATH_CALUDE_new_drive_free_space_l1679_167979

/-- Calculates the free space on a new external drive after file operations and transfer. -/
theorem new_drive_free_space 
  (initial_free : ℝ) 
  (initial_used : ℝ) 
  (deleted_size : ℝ) 
  (new_files_size : ℝ) 
  (new_drive_size : ℝ)
  (h1 : initial_free = 2.4)
  (h2 : initial_used = 12.6)
  (h3 : deleted_size = 4.6)
  (h4 : new_files_size = 2)
  (h5 : new_drive_size = 20) :
  new_drive_size - (initial_used - deleted_size + new_files_size) = 10 := by
  sorry

#check new_drive_free_space

end NUMINAMATH_CALUDE_new_drive_free_space_l1679_167979


namespace NUMINAMATH_CALUDE_c_share_is_27_l1679_167918

/-- Represents the rent share calculation for a pasture -/
structure PastureRent where
  a_oxen : ℕ
  a_months : ℕ
  b_oxen : ℕ
  b_months : ℕ
  c_oxen : ℕ
  c_months : ℕ
  total_rent : ℕ

/-- Calculates the share of rent for person C -/
def calculate_c_share (pr : PastureRent) : ℚ :=
  let total_ox_months := pr.a_oxen * pr.a_months + pr.b_oxen * pr.b_months + pr.c_oxen * pr.c_months
  let rent_per_ox_month := pr.total_rent / total_ox_months
  (pr.c_oxen * pr.c_months * rent_per_ox_month : ℚ)

/-- Theorem stating that C's share of rent is 27 Rs -/
theorem c_share_is_27 (pr : PastureRent) 
  (h1 : pr.a_oxen = 10) (h2 : pr.a_months = 7)
  (h3 : pr.b_oxen = 12) (h4 : pr.b_months = 5)
  (h5 : pr.c_oxen = 15) (h6 : pr.c_months = 3)
  (h7 : pr.total_rent = 105) : 
  calculate_c_share pr = 27 := by
  sorry


end NUMINAMATH_CALUDE_c_share_is_27_l1679_167918


namespace NUMINAMATH_CALUDE_dot_product_a_b_l1679_167959

-- Define the vectors
def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![3, -2]

-- Define the dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

-- Theorem statement
theorem dot_product_a_b : dot_product a b = 4 := by sorry

end NUMINAMATH_CALUDE_dot_product_a_b_l1679_167959


namespace NUMINAMATH_CALUDE_max_sphere_area_from_cube_l1679_167982

/-- The maximum surface area of a sphere carved from a cube -/
theorem max_sphere_area_from_cube (cube_side : ℝ) (sphere_radius : ℝ) : 
  cube_side = 2 →
  sphere_radius ≤ 1 →
  sphere_radius > 0 →
  (4 : ℝ) * Real.pi * sphere_radius^2 ≤ 4 * Real.pi :=
by
  sorry

#check max_sphere_area_from_cube

end NUMINAMATH_CALUDE_max_sphere_area_from_cube_l1679_167982


namespace NUMINAMATH_CALUDE_mixed_doubles_pairings_l1679_167903

theorem mixed_doubles_pairings (n_men : Nat) (n_women : Nat) : 
  n_men = 5 → n_women = 4 → (n_men.choose 2) * (n_women.choose 2) * 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_mixed_doubles_pairings_l1679_167903


namespace NUMINAMATH_CALUDE_investment_growth_l1679_167942

/-- The annual interest rate as a decimal -/
def interest_rate : ℝ := 0.10

/-- The number of years the investment grows -/
def years : ℕ := 4

/-- The initial investment amount -/
def initial_investment : ℝ := 300

/-- The final value after compounding -/
def final_value : ℝ := 439.23

/-- Theorem stating that the initial investment grows to the final value 
    when compounded annually at the given interest rate for the specified number of years -/
theorem investment_growth :
  initial_investment * (1 + interest_rate) ^ years = final_value := by
  sorry

end NUMINAMATH_CALUDE_investment_growth_l1679_167942


namespace NUMINAMATH_CALUDE_trig_simplification_l1679_167911

theorem trig_simplification (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.sin (x + y) = Real.cos x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l1679_167911


namespace NUMINAMATH_CALUDE_new_person_age_l1679_167976

/-- Given a group of 10 persons where replacing a 46-year-old person with a new person
    decreases the average age by 3 years, prove that the age of the new person is 16 years. -/
theorem new_person_age (T : ℝ) (A : ℝ) : 
  (T / 10 = (T - 46 + A) / 10 + 3) → A = 16 := by
  sorry

end NUMINAMATH_CALUDE_new_person_age_l1679_167976


namespace NUMINAMATH_CALUDE_kelly_nintendo_games_l1679_167990

theorem kelly_nintendo_games (initial : ℕ) : 
  initial + 31 - 105 = 6 → initial = 80 := by
  sorry

end NUMINAMATH_CALUDE_kelly_nintendo_games_l1679_167990


namespace NUMINAMATH_CALUDE_fraction_equals_five_l1679_167908

theorem fraction_equals_five (a b : ℕ+) (k : ℕ+) 
  (h : (a.val^2 + b.val^2 : ℚ) / (a.val * b.val - 1) = k.val) : k = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_five_l1679_167908


namespace NUMINAMATH_CALUDE_total_surfers_l1679_167989

/-- The number of surfers on Malibu beach -/
def malibu_surfers : ℕ := 50

/-- The number of surfers on Santa Monica beach -/
def santa_monica_surfers : ℕ := 30

/-- The number of surfers on Venice beach -/
def venice_surfers : ℕ := 20

/-- The ratio of surfers on Malibu beach -/
def malibu_ratio : ℕ := 5

/-- The ratio of surfers on Santa Monica beach -/
def santa_monica_ratio : ℕ := 3

/-- The ratio of surfers on Venice beach -/
def venice_ratio : ℕ := 2

theorem total_surfers : 
  malibu_surfers + santa_monica_surfers + venice_surfers = 100 ∧
  malibu_surfers * santa_monica_ratio = santa_monica_surfers * malibu_ratio ∧
  venice_surfers * santa_monica_ratio = santa_monica_surfers * venice_ratio :=
by sorry

end NUMINAMATH_CALUDE_total_surfers_l1679_167989


namespace NUMINAMATH_CALUDE_square_equals_25_l1679_167935

theorem square_equals_25 : {x : ℝ | x^2 = 25} = {-5, 5} := by sorry

end NUMINAMATH_CALUDE_square_equals_25_l1679_167935


namespace NUMINAMATH_CALUDE_min_yz_minus_xy_l1679_167915

/-- Represents a triangle with integer side lengths -/
structure Triangle :=
  (xy yz xz : ℕ)

/-- The perimeter of the triangle -/
def Triangle.perimeter (t : Triangle) : ℕ := t.xy + t.yz + t.xz

/-- Predicate for a valid triangle satisfying the given conditions -/
def isValidTriangle (t : Triangle) : Prop :=
  t.xy < t.yz ∧ t.yz ≤ t.xz ∧
  t.perimeter = 2010 ∧
  t.xy + t.yz > t.xz ∧ t.xy + t.xz > t.yz ∧ t.yz + t.xz > t.xy

theorem min_yz_minus_xy (t : Triangle) (h : isValidTriangle t) :
  ∀ (t' : Triangle), isValidTriangle t' → t'.yz - t'.xy ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_min_yz_minus_xy_l1679_167915


namespace NUMINAMATH_CALUDE_misses_both_mutually_exclusive_not_contradictory_l1679_167913

-- Define the sample space for two shots
inductive ShotOutcome
  | Miss
  | Hit

-- Define the event of hitting exactly once
def hits_exactly_once (outcome : ShotOutcome × ShotOutcome) : Prop :=
  (outcome.1 = ShotOutcome.Hit ∧ outcome.2 = ShotOutcome.Miss) ∨
  (outcome.1 = ShotOutcome.Miss ∧ outcome.2 = ShotOutcome.Hit)

-- Define the event of missing both times
def misses_both_times (outcome : ShotOutcome × ShotOutcome) : Prop :=
  outcome.1 = ShotOutcome.Miss ∧ outcome.2 = ShotOutcome.Miss

-- Theorem statement
theorem misses_both_mutually_exclusive_not_contradictory :
  (∀ outcome : ShotOutcome × ShotOutcome, ¬(hits_exactly_once outcome ∧ misses_both_times outcome)) ∧
  (∃ outcome : ShotOutcome × ShotOutcome, hits_exactly_once outcome ∨ misses_both_times outcome) :=
sorry

end NUMINAMATH_CALUDE_misses_both_mutually_exclusive_not_contradictory_l1679_167913


namespace NUMINAMATH_CALUDE_base_number_proof_l1679_167934

/-- 
Given a real number x, if (x^4 * 3.456789)^12 has 24 digits to the right of the decimal place 
when written as a single term, then x = 10^12.
-/
theorem base_number_proof (x : ℝ) : 
  (∃ n : ℕ, (x^4 * 3.456789)^12 * 10^24 = n) → x = 10^12 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l1679_167934


namespace NUMINAMATH_CALUDE_negation_theorem1_negation_theorem2_l1679_167973

-- Define a type for triangles
structure Triangle where
  -- You might add more properties here if needed
  interiorAngleSum : ℝ

-- Define the propositions
def proposition1 : Prop := ∃ t : Triangle, t.interiorAngleSum ≠ 180

def proposition2 : Prop := ∀ x : ℝ, |x| + x^2 ≥ 0

-- State the theorems
theorem negation_theorem1 : 
  (¬ proposition1) ↔ (∀ t : Triangle, t.interiorAngleSum = 180) :=
sorry

theorem negation_theorem2 : 
  (¬ proposition2) ↔ (∃ x : ℝ, |x| + x^2 < 0) :=
sorry

end NUMINAMATH_CALUDE_negation_theorem1_negation_theorem2_l1679_167973


namespace NUMINAMATH_CALUDE_utility_graph_non_planar_l1679_167922

/-- A graph representing the connection of houses to utilities -/
structure UtilityGraph where
  houses : Finset (Fin 3)
  utilities : Finset (Fin 3)
  connections : Set (Fin 3 × Fin 3)

/-- The utility graph is complete bipartite -/
def is_complete_bipartite (g : UtilityGraph) : Prop :=
  ∀ h ∈ g.houses, ∀ u ∈ g.utilities, (h, u) ∈ g.connections

/-- A graph is planar if it can be drawn on a plane without edge crossings -/
def is_planar (g : UtilityGraph) : Prop :=
  sorry -- Definition of planarity

/-- The theorem stating that the utility graph is non-planar -/
theorem utility_graph_non_planar (g : UtilityGraph) 
  (h1 : g.houses.card = 3) 
  (h2 : g.utilities.card = 3) 
  (h3 : is_complete_bipartite g) : 
  ¬ is_planar g :=
sorry

end NUMINAMATH_CALUDE_utility_graph_non_planar_l1679_167922


namespace NUMINAMATH_CALUDE_bacteria_population_growth_l1679_167919

def bacteria_count (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    2^(n/2 + 1)
  else
    2^((n-1)/2 + 1)

theorem bacteria_population_growth (n : ℕ) :
  bacteria_count n = if n % 2 = 0 then 2^(n/2 + 1) else 2^((n-1)/2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_bacteria_population_growth_l1679_167919


namespace NUMINAMATH_CALUDE_unique_perfect_square_grid_l1679_167921

/-- A type representing a 2x3 grid of natural numbers -/
def Grid := Fin 2 → Fin 3 → ℕ

/-- Check if a natural number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- Check if a Grid forms valid perfect squares horizontally and vertically -/
def is_valid_grid (g : Grid) : Prop :=
  (is_perfect_square (g 0 0 * 100 + g 0 1 * 10 + g 0 2)) ∧
  (is_perfect_square (g 1 0 * 100 + g 1 1 * 10 + g 1 2)) ∧
  (is_perfect_square (g 0 0 * 10 + g 1 0)) ∧
  (is_perfect_square (g 0 1 * 10 + g 1 1)) ∧
  (is_perfect_square (g 0 2 * 10 + g 1 2)) ∧
  (∀ i j, g i j < 10)

/-- The main theorem stating the existence and uniqueness of the solution -/
theorem unique_perfect_square_grid :
  ∃! g : Grid, is_valid_grid g ∧ g 0 0 = 8 ∧ g 0 1 = 4 ∧ g 0 2 = 1 ∧
                               g 1 0 = 1 ∧ g 1 1 = 9 ∧ g 1 2 = 6 :=
sorry

end NUMINAMATH_CALUDE_unique_perfect_square_grid_l1679_167921


namespace NUMINAMATH_CALUDE_min_value_F_negative_reals_l1679_167964

-- Define the real-valued functions f, g, and F
variable (f g : ℝ → ℝ)
variable (a b : ℝ)
def F (x : ℝ) := a * f x + b * g x + 2

-- State the theorem
theorem min_value_F_negative_reals
  (hf : ∀ x, f (-x) = -f x)  -- f is odd
  (hg : ∀ x, g (-x) = -g x)  -- g is odd
  (hab : a * b ≠ 0)  -- ab ≠ 0
  (hmax : ∃ x > 0, ∀ y > 0, F f g a b y ≤ F f g a b x ∧ F f g a b x = 5)  -- F has max value 5 on (0, +∞)
  : ∃ x < 0, ∀ y < 0, F f g a b y ≥ F f g a b x ∧ F f g a b x = -1 :=
sorry

end NUMINAMATH_CALUDE_min_value_F_negative_reals_l1679_167964


namespace NUMINAMATH_CALUDE_extreme_value_and_monotonicity_l1679_167900

/-- The function f(x) = x³ + ax² + bx + a² -/
def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f(x) -/
def f' (x a b : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extreme_value_and_monotonicity (a b : ℝ) :
  (f 1 a b = 10 ∧ f' 1 a b = 0) →
  (a = 4 ∧ b = -11) ∧
  (∀ x : ℝ, 
    (b = -a^2 → 
      (a > 0 → 
        ((x < -a ∨ x > a/3) → f' x a (-a^2) > 0) ∧
        ((-a < x ∧ x < a/3) → f' x a (-a^2) < 0)) ∧
      (a < 0 → 
        ((x < a/3 ∨ x > -a) → f' x a (-a^2) > 0) ∧
        ((a/3 < x ∧ x < -a) → f' x a (-a^2) < 0)) ∧
      (a = 0 → f' x a (-a^2) > 0))) :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_and_monotonicity_l1679_167900


namespace NUMINAMATH_CALUDE_sphere_in_cube_ratios_l1679_167999

/-- The ratio of volumes and surface areas for a sphere inscribed in a cube -/
theorem sphere_in_cube_ratios (s : ℝ) (h : s > 0) :
  let sphere_volume := (4 / 3) * Real.pi * s^3
  let cube_volume := (2 * s)^3
  let sphere_surface_area := 4 * Real.pi * s^2
  let cube_surface_area := 6 * (2 * s)^2
  (sphere_volume / cube_volume = Real.pi / 6) ∧
  (sphere_surface_area / cube_surface_area = Real.pi / 6) := by
  sorry

end NUMINAMATH_CALUDE_sphere_in_cube_ratios_l1679_167999


namespace NUMINAMATH_CALUDE_sets_relationship_l1679_167966

def M : Set ℝ := {x | ∃ m : ℤ, x = m + 1/6}
def N : Set ℝ := {x | ∃ n : ℤ, x = n/2 - 1/3}
def P : Set ℝ := {x | ∃ p : ℤ, x = p/2 + 1/6}

theorem sets_relationship : N = P ∧ M ≠ N :=
sorry

end NUMINAMATH_CALUDE_sets_relationship_l1679_167966


namespace NUMINAMATH_CALUDE_football_players_l1679_167997

theorem football_players (total : ℕ) (cricket : ℕ) (neither : ℕ) (both : ℕ) 
  (h1 : total = 250)
  (h2 : cricket = 90)
  (h3 : neither = 50)
  (h4 : both = 50) :
  total - neither - (cricket - both) = 160 :=
by
  sorry

#check football_players

end NUMINAMATH_CALUDE_football_players_l1679_167997


namespace NUMINAMATH_CALUDE_dogs_and_bunnies_total_l1679_167985

/-- Represents the number of animals in a pet shop -/
structure PetShop where
  dogs : ℕ
  cats : ℕ
  bunnies : ℕ

/-- Defines the conditions of the pet shop problem -/
def pet_shop_problem (shop : PetShop) : Prop :=
  shop.dogs = 51 ∧
  shop.dogs * 5 = shop.cats * 3 ∧
  shop.dogs * 9 = shop.bunnies * 3

/-- Theorem stating the total number of dogs and bunnies in the pet shop -/
theorem dogs_and_bunnies_total (shop : PetShop) :
  pet_shop_problem shop → shop.dogs + shop.bunnies = 204 := by
  sorry


end NUMINAMATH_CALUDE_dogs_and_bunnies_total_l1679_167985


namespace NUMINAMATH_CALUDE_point_on_curve_l1679_167920

theorem point_on_curve : ∃ θ : ℝ, 1 + Real.sin θ = 3/2 ∧ Real.sin (2*θ) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_curve_l1679_167920


namespace NUMINAMATH_CALUDE_one_carbon_per_sheet_l1679_167945

/-- Represents the number of carbon copies produced when sheets are folded and typed on -/
def carbon_copies_when_folded : ℕ := 2

/-- Represents the total number of sheets -/
def total_sheets : ℕ := 3

/-- Represents the number of carbons in each sheet -/
def carbons_per_sheet : ℕ := 1

/-- Theorem stating that there is 1 carbon in each sheet -/
theorem one_carbon_per_sheet :
  (carbons_per_sheet = 1) ∧ 
  (carbon_copies_when_folded = 2) ∧
  (total_sheets = 3) := by
  sorry

end NUMINAMATH_CALUDE_one_carbon_per_sheet_l1679_167945


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l1679_167972

theorem inscribed_rectangle_area (square_area : ℝ) (ratio : ℝ) 
  (h_square_area : square_area = 18)
  (h_ratio : ratio = 2)
  (h_positive : square_area > 0) :
  let square_side := Real.sqrt square_area
  let rect_short_side := 2 * square_side / (ratio + 1 + Real.sqrt (ratio^2 + 1))
  let rect_long_side := ratio * rect_short_side
  rect_short_side * rect_long_side = 8 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_rectangle_area_l1679_167972


namespace NUMINAMATH_CALUDE_batsman_running_fraction_l1679_167950

/-- Represents the score of a batsman in cricket --/
structure BatsmanScore where
  total_runs : ℕ
  boundaries : ℕ
  sixes : ℕ

/-- Calculates the fraction of runs made by running between wickets --/
def runningFraction (score : BatsmanScore) : ℚ :=
  let boundary_runs := 4 * score.boundaries
  let six_runs := 6 * score.sixes
  let running_runs := score.total_runs - (boundary_runs + six_runs)
  (running_runs : ℚ) / score.total_runs

theorem batsman_running_fraction :
  let score : BatsmanScore := ⟨250, 15, 10⟩
  runningFraction score = 13 / 25 := by
  sorry

end NUMINAMATH_CALUDE_batsman_running_fraction_l1679_167950


namespace NUMINAMATH_CALUDE_floor_ceiling_difference_l1679_167963

theorem floor_ceiling_difference : 
  ⌊(14 : ℝ) / 5 * (31 : ℝ) / 4⌋ - ⌈(14 : ℝ) / 5 * ⌈(31 : ℝ) / 4⌉⌉ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_difference_l1679_167963


namespace NUMINAMATH_CALUDE_jeremy_songs_l1679_167951

theorem jeremy_songs (x : ℕ) (h1 : x + (x + 5) = 23) : x + 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_songs_l1679_167951


namespace NUMINAMATH_CALUDE_class_average_theorem_l1679_167980

theorem class_average_theorem (total_students : ℕ) (students_without_two : ℕ) 
  (avg_without_two : ℚ) (score1 : ℕ) (score2 : ℕ) :
  total_students = students_without_two + 2 →
  (students_without_two : ℚ) * avg_without_two + score1 + score2 = total_students * 80 :=
by
  sorry

#check class_average_theorem 40 38 79 98 100

end NUMINAMATH_CALUDE_class_average_theorem_l1679_167980


namespace NUMINAMATH_CALUDE_sum_ratio_equals_half_l1679_167906

theorem sum_ratio_equals_half : (1 + 2 + 3 + 4 + 5) / (2 + 4 + 6 + 8 + 10) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_ratio_equals_half_l1679_167906


namespace NUMINAMATH_CALUDE_parabola_kite_sum_l1679_167978

/-- Represents a parabola of the form y = ax^2 + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Represents a kite formed by the intersection points of two parabolas with the coordinate axes -/
structure Kite where
  p1 : Parabola
  p2 : Parabola

theorem parabola_kite_sum (k : Kite) : 
  k.p1.a > 0 ∧ k.p2.a < 0 ∧  -- Ensure parabolas open in opposite directions
  k.p1.b = -4 ∧ k.p2.b = 6 ∧  -- Specific y-intercepts
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ 
    k.p1.a * x^2 + k.p1.b = 0 ∧   -- x-intercepts of first parabola
    k.p2.a * x^2 + k.p2.b = 0 ∧   -- x-intercepts of second parabola
    k.p1.a * y^2 + k.p1.b = k.p2.a * y^2 + k.p2.b) ∧  -- Intersection point
  (1/2 * (2 * (k.p2.b - k.p1.b)) * (2 * Real.sqrt (k.p2.b / (-k.p2.a))) = 24) →  -- Area of kite
  k.p1.a + (-k.p2.a) = 125/72 := by
sorry

end NUMINAMATH_CALUDE_parabola_kite_sum_l1679_167978


namespace NUMINAMATH_CALUDE_farmer_seeds_l1679_167907

theorem farmer_seeds (final_buckets sowed_buckets : ℝ) 
  (h1 : final_buckets = 6)
  (h2 : sowed_buckets = 2.75) : 
  final_buckets + sowed_buckets = 8.75 := by
  sorry

end NUMINAMATH_CALUDE_farmer_seeds_l1679_167907


namespace NUMINAMATH_CALUDE_alvin_coconut_trees_l1679_167904

/-- The number of coconuts each tree yields -/
def coconuts_per_tree : ℕ := 5

/-- The price of each coconut in dollars -/
def price_per_coconut : ℕ := 3

/-- The amount Alvin needs to earn in dollars -/
def target_earnings : ℕ := 90

/-- The number of coconut trees Alvin needs to harvest -/
def trees_to_harvest : ℕ := 6

theorem alvin_coconut_trees :
  trees_to_harvest * coconuts_per_tree * price_per_coconut = target_earnings :=
sorry

end NUMINAMATH_CALUDE_alvin_coconut_trees_l1679_167904


namespace NUMINAMATH_CALUDE_power_calculation_l1679_167960

theorem power_calculation : 9^6 * 3^9 / 27^5 = 729 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l1679_167960


namespace NUMINAMATH_CALUDE_square_side_length_l1679_167992

theorem square_side_length (area : ℝ) (side : ℝ) : 
  area = 1/9 → side^2 = area → side = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1679_167992


namespace NUMINAMATH_CALUDE_inverse_of_B_cubed_l1679_167983

/-- Given a 2x2 matrix B with its inverse, prove that the inverse of B^3 is as stated. -/
theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = ![![3, 4], ![-2, -2]]) : 
  (B^3)⁻¹ = ![![3, 4], ![-6, -28]] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_B_cubed_l1679_167983


namespace NUMINAMATH_CALUDE_baseball_gear_expense_l1679_167947

theorem baseball_gear_expense (initial_amount : ℕ) (remaining_amount : ℕ) 
  (h1 : initial_amount = 79)
  (h2 : remaining_amount = 32) :
  initial_amount - remaining_amount = 47 := by
  sorry

end NUMINAMATH_CALUDE_baseball_gear_expense_l1679_167947


namespace NUMINAMATH_CALUDE_ice_cream_sundaes_l1679_167953

theorem ice_cream_sundaes (total_flavors : ℕ) (h : total_flavors = 8) :
  let vanilla_sundaes := total_flavors - 1
  vanilla_sundaes = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sundaes_l1679_167953


namespace NUMINAMATH_CALUDE_negative_sqrt_two_squared_equals_two_l1679_167932

theorem negative_sqrt_two_squared_equals_two :
  (-Real.sqrt 2)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_two_squared_equals_two_l1679_167932


namespace NUMINAMATH_CALUDE_limit_rational_function_l1679_167936

/-- The limit of (2x³ - 3x² + 5x + 7) / (3x³ + 4x² - x + 2) as x approaches infinity is 2/3 -/
theorem limit_rational_function : 
  ∀ ε > 0, ∃ N : ℝ, ∀ x : ℝ, x > N → 
    |((2 * x^3 - 3 * x^2 + 5 * x + 7) / (3 * x^3 + 4 * x^2 - x + 2)) - 2/3| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_rational_function_l1679_167936


namespace NUMINAMATH_CALUDE_fraction_simplification_l1679_167996

theorem fraction_simplification : (5 * 6 - 4) / 8 = 13 / 4 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1679_167996


namespace NUMINAMATH_CALUDE_a_1992_b_1992_values_l1679_167909

def a : ℕ → ℤ
| 0 => 0
| (n + 1) => 2 * a n - a (n - 1) + 2

def b : ℕ → ℤ
| 0 => 8
| (n + 1) => 2 * b n - b (n - 1)

axiom square_sum : ∀ n > 0, ∃ k : ℤ, a n ^ 2 + b n ^ 2 = k ^ 2

theorem a_1992_b_1992_values : 
  (a 1992 = 1992^2 ∧ b 1992 = 7976) ∨ (a 1992 = 1992^2 ∧ b 1992 = -7960) := by
  sorry

end NUMINAMATH_CALUDE_a_1992_b_1992_values_l1679_167909


namespace NUMINAMATH_CALUDE_point_below_line_implies_a_greater_than_three_l1679_167946

/-- A point is below a line if its y-coordinate is less than the y-coordinate of the point on the line with the same x-coordinate. -/
def PointBelowLine (x y : ℝ) (m b : ℝ) : Prop :=
  y < m * x + b

theorem point_below_line_implies_a_greater_than_three (a : ℝ) :
  PointBelowLine a 3 2 3 → a > 3 := by
  sorry

end NUMINAMATH_CALUDE_point_below_line_implies_a_greater_than_three_l1679_167946


namespace NUMINAMATH_CALUDE_unique_four_digit_cube_divisible_by_16_and_9_l1679_167931

theorem unique_four_digit_cube_divisible_by_16_and_9 :
  ∃! n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 
  (∃ m : ℕ, n = m^3) ∧ 
  n % 16 = 0 ∧ n % 9 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_cube_divisible_by_16_and_9_l1679_167931


namespace NUMINAMATH_CALUDE_female_grade_one_jiu_is_set_l1679_167958

-- Define the universe of students
def Student : Type := sorry

-- Define the property of being female
def is_female : Student → Prop := sorry

-- Define the property of being in grade one of Jiu Middle School
def is_grade_one_jiu : Student → Prop := sorry

-- Define our set
def female_grade_one_jiu : Set Student :=
  {s : Student | is_female s ∧ is_grade_one_jiu s}

-- Theorem stating that female_grade_one_jiu is a well-defined set
theorem female_grade_one_jiu_is_set :
  ∀ (s : Student), Decidable (s ∈ female_grade_one_jiu) :=
sorry

end NUMINAMATH_CALUDE_female_grade_one_jiu_is_set_l1679_167958


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l1679_167993

theorem smallest_number_with_remainders : ∃ (a : ℕ), 
  (a % 3 = 2) ∧ (a % 5 = 3) ∧ (a % 7 = 3) ∧
  (∀ (b : ℕ), b < a → ¬((b % 3 = 2) ∧ (b % 5 = 3) ∧ (b % 7 = 3))) ∧
  a = 98 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l1679_167993


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l1679_167916

noncomputable def angle_between (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  (Real.sqrt (a.1^2 + a.2^2) = 4) →
  (Real.sqrt (b.1^2 + b.2^2) = 3) →
  (angle_between a b = Real.pi / 3) →
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = Real.sqrt 37 :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l1679_167916


namespace NUMINAMATH_CALUDE_not_perfect_square_l1679_167924

theorem not_perfect_square (n : ℕ+) : ¬ ∃ m : ℤ, (2551 * 543^n.val - 2008 * 7^n.val : ℤ) = m^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1679_167924


namespace NUMINAMATH_CALUDE_average_exists_l1679_167940

theorem average_exists : ∃ N : ℝ, 11 < N ∧ N < 19 ∧ (8 + 12 + N) / 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_exists_l1679_167940


namespace NUMINAMATH_CALUDE_gcd_12345_67890_l1679_167905

theorem gcd_12345_67890 : Nat.gcd 12345 67890 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12345_67890_l1679_167905


namespace NUMINAMATH_CALUDE_product_of_fractions_l1679_167943

theorem product_of_fractions : 
  (1 - 1/2) * (1 - 1/3) * (1 - 1/4) * (1 - 1/5) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l1679_167943


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1679_167928

-- Define the quadratic function f
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_inequality (b c : ℝ) 
  (h : ∀ x, f b c (3 + x) = f b c (3 - x)) : 
  f b c 4 < f b c 1 ∧ f b c 1 < f b c (-1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1679_167928


namespace NUMINAMATH_CALUDE_two_x_plus_y_equals_seven_l1679_167927

theorem two_x_plus_y_equals_seven 
  (h1 : (x + y) / 3 = 1.6666666666666667)
  (h2 : x + 2 * y = 8) : 
  2 * x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_two_x_plus_y_equals_seven_l1679_167927


namespace NUMINAMATH_CALUDE_toy_production_difference_l1679_167955

/-- The difference in daily toy production between actual and planned rates --/
theorem toy_production_difference (total_toys : ℕ) (planned_days : ℕ) (actual_days : ℕ) 
  (h1 : total_toys = 10080)
  (h2 : planned_days = 14)
  (h3 : actual_days = planned_days - 2) :
  (total_toys / actual_days) - (total_toys / planned_days) = 120 := by
  sorry

end NUMINAMATH_CALUDE_toy_production_difference_l1679_167955


namespace NUMINAMATH_CALUDE_abs_neg_five_equals_five_l1679_167977

theorem abs_neg_five_equals_five : |(-5 : ℤ)| = 5 := by sorry

end NUMINAMATH_CALUDE_abs_neg_five_equals_five_l1679_167977


namespace NUMINAMATH_CALUDE_lg_graph_property_l1679_167933

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define what it means for a point to be on the graph of y = lg x
def on_lg_graph (p : ℝ × ℝ) : Prop := p.2 = lg p.1

-- Theorem statement
theorem lg_graph_property (a b : ℝ) (h1 : on_lg_graph (a, b)) (h2 : a ≠ 1) :
  on_lg_graph (a^2, 2*b) :=
sorry

end NUMINAMATH_CALUDE_lg_graph_property_l1679_167933
