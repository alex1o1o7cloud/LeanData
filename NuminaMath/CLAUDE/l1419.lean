import Mathlib

namespace NUMINAMATH_CALUDE_vectors_form_basis_l1419_141945

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (3, 1)

theorem vectors_form_basis : LinearIndependent ℝ ![a, b] ∧ Submodule.span ℝ {a, b} = ⊤ := by
  sorry

end NUMINAMATH_CALUDE_vectors_form_basis_l1419_141945


namespace NUMINAMATH_CALUDE_y_relationship_l1419_141902

/-- The quadratic function f(x) = x² + 4x - 5 --/
def f (x : ℝ) : ℝ := x^2 + 4*x - 5

/-- y₁ is the y-coordinate of point A(-4, y₁) on the graph of f --/
def y₁ : ℝ := f (-4)

/-- y₂ is the y-coordinate of point B(-3, y₂) on the graph of f --/
def y₂ : ℝ := f (-3)

/-- y₃ is the y-coordinate of point C(1, y₃) on the graph of f --/
def y₃ : ℝ := f 1

/-- Theorem stating the relationship between y₁, y₂, and y₃ --/
theorem y_relationship : y₂ < y₁ ∧ y₁ < y₃ := by
  sorry

end NUMINAMATH_CALUDE_y_relationship_l1419_141902


namespace NUMINAMATH_CALUDE_bowling_ball_weighs_18_pounds_l1419_141929

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := sorry

/-- The weight of one canoe in pounds -/
def canoe_weight : ℝ := sorry

/-- Theorem stating the weight of one bowling ball is 18 pounds -/
theorem bowling_ball_weighs_18_pounds :
  (10 * bowling_ball_weight = 6 * canoe_weight) →
  (4 * canoe_weight = 120) →
  bowling_ball_weight = 18 := by sorry

end NUMINAMATH_CALUDE_bowling_ball_weighs_18_pounds_l1419_141929


namespace NUMINAMATH_CALUDE_sum_of_fractions_between_18_and_19_l1419_141933

theorem sum_of_fractions_between_18_and_19 :
  let a : ℚ := 2 + 3/8
  let b : ℚ := 4 + 1/3
  let c : ℚ := 5 + 2/21
  let d : ℚ := 6 + 1/11
  18 < a + b + c + d ∧ a + b + c + d < 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_between_18_and_19_l1419_141933


namespace NUMINAMATH_CALUDE_solve_for_y_l1419_141965

theorem solve_for_y (x y : ℝ) (h1 : x - y = 16) (h2 : x + y = 8) : y = -4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1419_141965


namespace NUMINAMATH_CALUDE_shadow_growth_rate_l1419_141927

theorem shadow_growth_rate (shadow_length : ℝ) (hours_past_noon : ℝ) :
  shadow_length = 360 ∧ hours_past_noon = 6 →
  (shadow_length / 12) / hours_past_noon = 5 :=
by sorry

end NUMINAMATH_CALUDE_shadow_growth_rate_l1419_141927


namespace NUMINAMATH_CALUDE_total_protest_days_equals_29_625_l1419_141914

/-- Calculates the total number of days spent at four protests -/
def total_protest_days (first_protest : ℝ) (second_increase : ℝ) (third_increase : ℝ) (fourth_increase : ℝ) : ℝ :=
  let second_protest := first_protest * (1 + second_increase)
  let third_protest := second_protest * (1 + third_increase)
  let fourth_protest := third_protest * (1 + fourth_increase)
  first_protest + second_protest + third_protest + fourth_protest

/-- Theorem stating that the total number of days spent at four protests equals 29.625 -/
theorem total_protest_days_equals_29_625 :
  total_protest_days 4 0.25 0.5 0.75 = 29.625 := by
  sorry

end NUMINAMATH_CALUDE_total_protest_days_equals_29_625_l1419_141914


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_one_l1419_141919

theorem simplify_and_evaluate_one (x y : ℚ) :
  x = 1/2 ∧ y = -1 →
  (1 * (2*x + y) * (2*x - y)) - 4*x*(x - y) = -3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_one_l1419_141919


namespace NUMINAMATH_CALUDE_overlapping_circles_area_ratio_l1419_141946

/-- Given two overlapping circles, this theorem proves the ratio of their areas. -/
theorem overlapping_circles_area_ratio
  (L S A : ℝ)  -- L: area of large circle, S: area of small circle, A: overlapped area
  (h1 : A = 3/5 * S)  -- Overlapped area is 3/5 of small circle
  (h2 : A = 6/25 * L)  -- Overlapped area is 6/25 of large circle
  : S / L = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_overlapping_circles_area_ratio_l1419_141946


namespace NUMINAMATH_CALUDE_max_imaginary_part_of_roots_l1419_141936

theorem max_imaginary_part_of_roots (z : ℂ) (θ : ℝ) :
  z^12 - z^9 + z^6 - z^3 + 1 = 0 →
  -π/2 ≤ θ ∧ θ ≤ π/2 →
  z.im = Real.sin θ →
  z.im ≤ Real.sin (84 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_max_imaginary_part_of_roots_l1419_141936


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1419_141968

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 3| = 5 - x :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1419_141968


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_value_l1419_141918

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → y ≠ x-5 → ¬∃ (a b : ℝ), b > 0 ∧ b ≠ 1 ∧ x-5 = a^2 * b

theorem simplest_quadratic_radical_value :
  ∀ x : ℝ, x ∈ ({11, 13, 21, 29} : Set ℝ) →
    (is_simplest_quadratic_radical x ↔ x = 11) :=
by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_value_l1419_141918


namespace NUMINAMATH_CALUDE_cone_volume_proof_l1419_141958

theorem cone_volume_proof (a b c r h : ℝ) : 
  a = 3 → b = 4 → c^2 = a^2 + b^2 → 
  2 * r = c → h^2 + r^2 = 3^2 →
  (1/3) * π * r^2 * h = (25 * π * Real.sqrt 11) / 24 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_proof_l1419_141958


namespace NUMINAMATH_CALUDE_coffee_machine_discount_l1419_141907

def coffee_machine_problem (original_price : ℝ) (home_cost : ℝ) (previous_coffees : ℕ) (previous_price : ℝ) (payoff_days : ℕ) : Prop :=
  let previous_daily_cost := previous_coffees * previous_price
  let daily_savings := previous_daily_cost - home_cost
  let total_savings := daily_savings * payoff_days
  let discount := original_price - total_savings
  original_price = 200 ∧ 
  home_cost = 3 ∧ 
  previous_coffees = 2 ∧ 
  previous_price = 4 ∧ 
  payoff_days = 36 →
  discount = 20

theorem coffee_machine_discount :
  coffee_machine_problem 200 3 2 4 36 :=
by sorry

end NUMINAMATH_CALUDE_coffee_machine_discount_l1419_141907


namespace NUMINAMATH_CALUDE_investment_income_percentage_l1419_141910

/-- Proves that the total annual income from two investments is equal to 6% of the total investment amount. -/
theorem investment_income_percentage (initial_investment : ℝ) (additional_investment : ℝ) 
  (initial_rate : ℝ) (additional_rate : ℝ) :
  initial_investment = 2400 →
  additional_investment = 599.9999999999999 →
  initial_rate = 0.05 →
  additional_rate = 0.10 →
  let total_investment := initial_investment + additional_investment
  let total_income := initial_investment * initial_rate + additional_investment * additional_rate
  (total_income / total_investment) * 100 = 6 := by
  sorry

end NUMINAMATH_CALUDE_investment_income_percentage_l1419_141910


namespace NUMINAMATH_CALUDE_negative_two_cubed_equality_l1419_141904

theorem negative_two_cubed_equality : (-2)^3 = -2^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_cubed_equality_l1419_141904


namespace NUMINAMATH_CALUDE_coordinates_of_M_l1419_141985

/-- Point in 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance from a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ := |p.x|

/-- Check if a point is on the angle bisector of the first and third quadrants -/
def isOnAngleBisector (p : Point) : Prop := p.x = p.y

/-- Given point M with coordinates (2-m, 1+2m) -/
def M (m : ℝ) : Point := ⟨2 - m, 1 + 2*m⟩

theorem coordinates_of_M (m : ℝ) :
  (distanceToYAxis (M m) = 3 → (M m = ⟨3, -1⟩ ∨ M m = ⟨-3, 11⟩)) ∧
  (isOnAngleBisector (M m) → M m = ⟨5/3, 5/3⟩) := by
  sorry

end NUMINAMATH_CALUDE_coordinates_of_M_l1419_141985


namespace NUMINAMATH_CALUDE_meeting_time_l1419_141956

/-- The speed of l in km/hr -/
def speed_l : ℝ := 50

/-- The speed of k in km/hr -/
def speed_k : ℝ := speed_l * 1.5

/-- The time difference between k's and l's start times in hours -/
def time_difference : ℝ := 1

/-- The total distance between k and l in km -/
def total_distance : ℝ := 300

/-- The time when l starts -/
def start_time_l : ℕ := 9

/-- The time when k starts -/
def start_time_k : ℕ := 10

theorem meeting_time :
  let distance_traveled_by_l := speed_l * time_difference
  let remaining_distance := total_distance - distance_traveled_by_l
  let relative_speed := speed_l + speed_k
  let time_to_meet := remaining_distance / relative_speed
  start_time_k + ⌊time_to_meet⌋ = 12 := by sorry

end NUMINAMATH_CALUDE_meeting_time_l1419_141956


namespace NUMINAMATH_CALUDE_final_eraser_count_l1419_141972

/-- Represents the state of erasers in three drawers -/
structure EraserState where
  drawer1 : ℕ
  drawer2 : ℕ
  drawer3 : ℕ

/-- Initial state of erasers -/
def initial_state : EraserState := ⟨139, 95, 75⟩

/-- State after Monday's changes -/
def monday_state (s : EraserState) : EraserState :=
  ⟨s.drawer1 + 50, s.drawer2 - 50, s.drawer3⟩

/-- State after Tuesday's changes -/
def tuesday_state (s : EraserState) : EraserState :=
  ⟨s.drawer1 - 35, s.drawer2, s.drawer3 - 20⟩

/-- Final state after changes later in the week -/
def final_state (s : EraserState) : EraserState :=
  ⟨s.drawer1 + 131, s.drawer2 - 30, s.drawer3⟩

/-- Total number of erasers in all drawers -/
def total_erasers (s : EraserState) : ℕ :=
  s.drawer1 + s.drawer2 + s.drawer3

/-- Theorem stating the final number of erasers -/
theorem final_eraser_count :
  total_erasers (final_state (tuesday_state (monday_state initial_state))) = 355 := by
  sorry


end NUMINAMATH_CALUDE_final_eraser_count_l1419_141972


namespace NUMINAMATH_CALUDE_parabola_directrix_l1419_141971

/-- Given a parabola with equation y = 8x^2, its directrix has equation y = -1/32 -/
theorem parabola_directrix (x y : ℝ) :
  y = 8 * x^2 →
  ∃ (p : ℝ), p > 0 ∧ x^2 = 4 * p * y ∧ -p = -(1/32) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1419_141971


namespace NUMINAMATH_CALUDE_negation_quadratic_roots_l1419_141997

theorem negation_quadratic_roots (a b c : ℝ) :
  (¬(b^2 - 4*a*c < 0 → ∀ x, a*x^2 + b*x + c ≠ 0)) ↔
  (b^2 - 4*a*c ≥ 0 → ∃ x, a*x^2 + b*x + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_quadratic_roots_l1419_141997


namespace NUMINAMATH_CALUDE_tailors_hourly_rate_l1419_141943

theorem tailors_hourly_rate (num_shirts : ℕ) (num_pants : ℕ) (shirt_time : ℚ) (total_cost : ℚ) :
  num_shirts = 10 →
  num_pants = 12 →
  shirt_time = 3/2 →
  total_cost = 1530 →
  (num_shirts * shirt_time + num_pants * (2 * shirt_time)) * (total_cost / (num_shirts * shirt_time + num_pants * (2 * shirt_time))) = 30 := by
  sorry

end NUMINAMATH_CALUDE_tailors_hourly_rate_l1419_141943


namespace NUMINAMATH_CALUDE_floor_of_e_l1419_141957

theorem floor_of_e : ⌊Real.exp 1⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_e_l1419_141957


namespace NUMINAMATH_CALUDE_integral_proof_l1419_141974

open Real

theorem integral_proof (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ -2) :
  deriv (fun x => log (abs (x + 1)) - 1 / (2 * (x + 2)^2)) x =
  (x^3 + 6*x^2 + 13*x + 9) / ((x + 1) * (x + 2)^3) := by
sorry

end NUMINAMATH_CALUDE_integral_proof_l1419_141974


namespace NUMINAMATH_CALUDE_no_complete_non_self_intersecting_path_l1419_141966

/-- Represents the surface of a Rubik's cube -/
structure RubiksCubeSurface where
  squares : Nat
  diagonals : Nat
  vertices : Nat

/-- The surface of a standard Rubik's cube -/
def standardRubiksCube : RubiksCubeSurface :=
  { squares := 54
  , diagonals := 54
  , vertices := 56 }

/-- A path on the surface of a Rubik's cube -/
structure DiagonalPath (surface : RubiksCubeSurface) where
  length : Nat
  is_non_self_intersecting : Bool

/-- Theorem stating the impossibility of creating a non-self-intersecting path
    using all diagonals on the surface of a standard Rubik's cube -/
theorem no_complete_non_self_intersecting_path 
  (surface : RubiksCubeSurface) 
  (h_surface : surface = standardRubiksCube) :
  ¬∃ (path : DiagonalPath surface), 
    path.length = surface.diagonals ∧ 
    path.is_non_self_intersecting = true := by
  sorry


end NUMINAMATH_CALUDE_no_complete_non_self_intersecting_path_l1419_141966


namespace NUMINAMATH_CALUDE_binomial_product_minus_240_l1419_141940

theorem binomial_product_minus_240 : 
  (Nat.choose 10 3) * (Nat.choose 8 3) - 240 = 6480 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_minus_240_l1419_141940


namespace NUMINAMATH_CALUDE_circle_diameter_from_intersection_l1419_141911

-- Define the given circle
def given_circle (x y : ℝ) : Prop := x^2 + y^2 + x - 6*y + 3 = 0

-- Define the given line
def given_line (x y : ℝ) : Prop := x + 2*y - 3 = 0

-- Define the resulting circle
def result_circle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define a function to represent a point
def Point := ℝ × ℝ

-- Theorem statement
theorem circle_diameter_from_intersection :
  ∃ (P Q : Point),
    (given_circle P.1 P.2 ∧ given_line P.1 P.2) ∧
    (given_circle Q.1 Q.2 ∧ given_line Q.1 Q.2) ∧
    (∀ (x y : ℝ), result_circle x y ↔ 
      ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
        x = P.1 * (1 - t) + Q.1 * t ∧
        y = P.2 * (1 - t) + Q.2 * t) :=
sorry

end NUMINAMATH_CALUDE_circle_diameter_from_intersection_l1419_141911


namespace NUMINAMATH_CALUDE_max_sum_of_shorter_l1419_141986

/-- Represents the configuration of houses -/
structure HouseConfig where
  one_story : ℕ
  two_story : ℕ

/-- The total number of floors in the city -/
def total_floors : ℕ := 30

/-- Calculates the sum of shorter houses seen from each roof -/
def sum_of_shorter (config : HouseConfig) : ℕ :=
  config.one_story * config.two_story

/-- Checks if a configuration is valid (i.e., totals 30 floors) -/
def is_valid_config (config : HouseConfig) : Prop :=
  config.one_story + 2 * config.two_story = total_floors

/-- The theorem to be proved -/
theorem max_sum_of_shorter :
  ∃ (config1 config2 : HouseConfig),
    is_valid_config config1 ∧
    is_valid_config config2 ∧
    sum_of_shorter config1 = 112 ∧
    sum_of_shorter config2 = 112 ∧
    (∀ (config : HouseConfig), is_valid_config config → sum_of_shorter config ≤ 112) ∧
    ((config1.one_story = 16 ∧ config1.two_story = 7) ∨
     (config1.one_story = 14 ∧ config1.two_story = 8)) ∧
    ((config2.one_story = 16 ∧ config2.two_story = 7) ∨
     (config2.one_story = 14 ∧ config2.two_story = 8)) ∧
    config1 ≠ config2 :=
  sorry

end NUMINAMATH_CALUDE_max_sum_of_shorter_l1419_141986


namespace NUMINAMATH_CALUDE_zeros_of_log_linear_function_l1419_141928

open Real

theorem zeros_of_log_linear_function (m : ℝ) (x₁ x₂ : ℝ) 
  (hm : m > 0) 
  (hx : x₁ < x₂) 
  (hz₁ : m * log x₁ = x₁) 
  (hz₂ : m * log x₂ = x₂) : 
  x₁ < exp 1 ∧ exp 1 < x₂ := by
sorry

end NUMINAMATH_CALUDE_zeros_of_log_linear_function_l1419_141928


namespace NUMINAMATH_CALUDE_problem_1_l1419_141952

theorem problem_1 : (-2.4) + (-3.7) + (-4.6) + 5.7 = -5 := by
  sorry

#eval (-2.4) + (-3.7) + (-4.6) + 5.7

end NUMINAMATH_CALUDE_problem_1_l1419_141952


namespace NUMINAMATH_CALUDE_f_tangent_perpendicular_range_l1419_141926

open Real

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x * (m + exp (-x))

theorem f_tangent_perpendicular_range :
  ∃ (a b : Set ℝ), a = Set.Ioo 0 (exp (-2)) ∧
  (∀ m : ℝ, m ∈ a ↔ 
    ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
      (deriv (f m)) x₁ = 0 ∧ 
      (deriv (f m)) x₂ = 0) :=
sorry

end NUMINAMATH_CALUDE_f_tangent_perpendicular_range_l1419_141926


namespace NUMINAMATH_CALUDE_problem1_problem2_problem3_problem4_l1419_141921

theorem problem1 : 5 / 7 + (-5 / 6) - (-2 / 7) + 1 + 1 / 6 = 4 / 3 := by sorry

theorem problem2 : (1 / 2 - (1 + 1 / 3) + 3 / 8) / (-1 / 24) = 11 := by sorry

theorem problem3 : (-3)^3 + (-5)^2 - |(-3)| * 4 = -14 := by sorry

theorem problem4 : -(1^101) - (-0.5 - (1 - 3 / 5 * 0.7) / (-1 / 2)^2) = 91 / 50 := by sorry

end NUMINAMATH_CALUDE_problem1_problem2_problem3_problem4_l1419_141921


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l1419_141996

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw_x : w / x = 4 / 3)
  (hy_z : y / z = 5 / 3)
  (hz_x : z / x = 1 / 5) :
  w / y = 4 / 1 := by
sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l1419_141996


namespace NUMINAMATH_CALUDE_arithmetic_equation_l1419_141916

theorem arithmetic_equation : 4 * (8 - 2 + 3) - 7 = 29 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equation_l1419_141916


namespace NUMINAMATH_CALUDE_exterior_angle_pentagon_octagon_exterior_angle_pentagon_octagon_is_117_l1419_141983

/-- The measure of an exterior angle formed by a regular pentagon and a regular octagon sharing a side -/
theorem exterior_angle_pentagon_octagon : ℝ :=
  let pentagon_interior_angle : ℝ := (180 * (5 - 2)) / 5
  let octagon_interior_angle : ℝ := (180 * (8 - 2)) / 8
  360 - (pentagon_interior_angle + octagon_interior_angle)

/-- The exterior angle formed by a regular pentagon and a regular octagon sharing a side is 117 degrees -/
theorem exterior_angle_pentagon_octagon_is_117 :
  exterior_angle_pentagon_octagon = 117 := by sorry

end NUMINAMATH_CALUDE_exterior_angle_pentagon_octagon_exterior_angle_pentagon_octagon_is_117_l1419_141983


namespace NUMINAMATH_CALUDE_complex_power_one_minus_i_six_l1419_141951

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_one_minus_i_six :
  (1 - i)^6 = 8*i := by sorry

end NUMINAMATH_CALUDE_complex_power_one_minus_i_six_l1419_141951


namespace NUMINAMATH_CALUDE_max_ab_line_tangent_circle_l1419_141967

/-- The maximum value of ab when a line is tangent to a circle -/
theorem max_ab_line_tangent_circle (a b : ℝ) : 
  -- Line equation: x + 2y = 0
  -- Circle equation: (x-a)² + (y-b)² = 5
  -- Line is tangent to circle
  (∃ x y : ℝ, x + 2*y = 0 ∧ (x-a)^2 + (y-b)^2 = 5 ∧ 
    ∀ x' y' : ℝ, x' + 2*y' = 0 → (x'-a)^2 + (y'-b)^2 ≥ 5) →
  -- Center of circle is above the line
  a + 2*b > 0 →
  -- The maximum value of ab is 25/8
  a * b ≤ 25/8 :=
by sorry

end NUMINAMATH_CALUDE_max_ab_line_tangent_circle_l1419_141967


namespace NUMINAMATH_CALUDE_sharon_coffee_pods_l1419_141903

/-- Calculates the number of pods in a box given vacation details and spending -/
def pods_per_box (vacation_days : ℕ) (daily_pods : ℕ) (total_spent : ℕ) (price_per_box : ℕ) : ℕ :=
  let total_pods := vacation_days * daily_pods
  let boxes_bought := total_spent / price_per_box
  total_pods / boxes_bought

/-- Proves that the number of pods in a box is 30 given the specific vacation details -/
theorem sharon_coffee_pods :
  pods_per_box 40 3 32 8 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sharon_coffee_pods_l1419_141903


namespace NUMINAMATH_CALUDE_power_sum_and_division_equals_82_l1419_141930

theorem power_sum_and_division_equals_82 : 2^0 + 9^5 / 9^3 = 82 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_and_division_equals_82_l1419_141930


namespace NUMINAMATH_CALUDE_subcommittees_count_l1419_141954

def planning_committee_size : ℕ := 10
def teacher_count : ℕ := 4
def subcommittee_size : ℕ := 4

/-- The number of distinct subcommittees with at least one teacher -/
def subcommittees_with_teacher : ℕ :=
  Nat.choose planning_committee_size subcommittee_size -
  Nat.choose (planning_committee_size - teacher_count) subcommittee_size

theorem subcommittees_count :
  subcommittees_with_teacher = 195 :=
sorry

end NUMINAMATH_CALUDE_subcommittees_count_l1419_141954


namespace NUMINAMATH_CALUDE_f_zero_at_three_l1419_141917

def f (x s : ℝ) : ℝ := 3 * x^5 + 2 * x^4 - x^3 + 4 * x^2 - 5 * x + s

theorem f_zero_at_three (s : ℝ) : f 3 s = 0 ↔ s = -885 := by sorry

end NUMINAMATH_CALUDE_f_zero_at_three_l1419_141917


namespace NUMINAMATH_CALUDE_mixed_tea_sale_price_l1419_141980

/-- Represents the types of tea in the mixture -/
inductive TeaType
| First
| Second
| Third

/-- Represents the properties of each tea type -/
def tea_properties : TeaType → (Nat × Nat × Nat) :=
  fun t => match t with
  | TeaType.First  => (120, 30, 50)
  | TeaType.Second => (45, 40, 30)
  | TeaType.Third  => (35, 60, 25)

/-- Calculates the selling price for a given tea type -/
def selling_price (t : TeaType) : Nat :=
  let (weight, cost, profit) := tea_properties t
  weight * cost * (100 + profit) / 100

/-- Theorem stating the sale price of the mixed tea per kg -/
theorem mixed_tea_sale_price :
  (selling_price TeaType.First + selling_price TeaType.Second + selling_price TeaType.Third) /
  (120 + 45 + 35 : Nat) = 51825 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_mixed_tea_sale_price_l1419_141980


namespace NUMINAMATH_CALUDE_two_oplus_neg_three_l1419_141937

/-- The ⊕ operation for rational numbers -/
def oplus (α β : ℚ) : ℚ := α * β + 1

/-- Theorem stating that 2 ⊕ (-3) = -5 -/
theorem two_oplus_neg_three : oplus 2 (-3) = -5 := by
  sorry

end NUMINAMATH_CALUDE_two_oplus_neg_three_l1419_141937


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1419_141979

theorem arithmetic_sequence_sum : 
  let a₁ : ℕ := 2  -- first term
  let aₙ : ℕ := 29 -- last term
  let d : ℕ := 3   -- common difference
  let n : ℕ := (aₙ - a₁) / d + 1 -- number of terms
  (n : ℝ) * (a₁ + aₙ) / 2 = 155 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1419_141979


namespace NUMINAMATH_CALUDE_number_representation_proof_l1419_141939

theorem number_representation_proof (n a b c : ℕ) : 
  (n = 14^2 * a + 14 * b + c) →
  (n = 15^2 * a + 15 * c + b) →
  (n = 6^3 * a + 6^2 * c + 6 * a + c) →
  (a > 0) →
  (a < 6 ∧ b < 14 ∧ c < 6) →
  (n = 925) := by
sorry

end NUMINAMATH_CALUDE_number_representation_proof_l1419_141939


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1419_141908

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3*y = 1) :
  1/x + 3/y ≥ 16 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3*y₀ = 1 ∧ 1/x₀ + 3/y₀ = 16 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1419_141908


namespace NUMINAMATH_CALUDE_total_games_is_105_l1419_141991

/-- The number of teams in the league -/
def num_teams : ℕ := 15

/-- The total number of games played in the league -/
def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that the total number of games played is 105 -/
theorem total_games_is_105 : total_games num_teams = 105 := by
  sorry

end NUMINAMATH_CALUDE_total_games_is_105_l1419_141991


namespace NUMINAMATH_CALUDE_roller_coaster_tickets_l1419_141931

/-- The number of friends going on the roller coaster ride -/
def num_friends : ℕ := 8

/-- The total number of tickets needed for all friends -/
def total_tickets : ℕ := 48

/-- The number of tickets required per ride -/
def tickets_per_ride : ℕ := total_tickets / num_friends

theorem roller_coaster_tickets : tickets_per_ride = 6 := by
  sorry

end NUMINAMATH_CALUDE_roller_coaster_tickets_l1419_141931


namespace NUMINAMATH_CALUDE_box_two_neg_one_zero_l1419_141935

def box (a b c : ℤ) : ℚ := a^b - b^c + c^a

theorem box_two_neg_one_zero : box 2 (-1) 0 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_box_two_neg_one_zero_l1419_141935


namespace NUMINAMATH_CALUDE_scheme_probability_l1419_141949

theorem scheme_probability (p_both : ℝ) (h1 : p_both = 0.3) :
  1 - (1 - p_both) * (1 - p_both) = 0.51 := by
sorry

end NUMINAMATH_CALUDE_scheme_probability_l1419_141949


namespace NUMINAMATH_CALUDE_segment_length_l1419_141973

/-- The length of a segment with endpoints (1,1) and (8,17) is √305 -/
theorem segment_length : Real.sqrt ((8 - 1)^2 + (17 - 1)^2) = Real.sqrt 305 := by
  sorry

end NUMINAMATH_CALUDE_segment_length_l1419_141973


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1419_141912

/-- An arithmetic sequence with given first three terms -/
def arithmetic_sequence (x : ℝ) (n : ℕ) : ℝ :=
  let a₁ := x - 1
  let a₂ := x + 1
  let a₃ := 2 * x + 3
  let d := a₂ - a₁  -- common difference
  a₁ + (n - 1) * d

/-- Theorem stating the general formula for the given arithmetic sequence -/
theorem arithmetic_sequence_formula (x : ℝ) (n : ℕ) :
  arithmetic_sequence x n = 2 * n - 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1419_141912


namespace NUMINAMATH_CALUDE_negation_forall_squared_gt_neg_one_negation_exists_squared_leq_nine_abs_gt_not_necessary_for_gt_m_lt_zero_iff_one_positive_one_negative_root_l1419_141982

-- Statement 1
theorem negation_forall_squared_gt_neg_one :
  (¬ ∀ x : ℝ, x^2 > -1) ↔ (∃ x : ℝ, x^2 ≤ -1) := by sorry

-- Statement 2
theorem negation_exists_squared_leq_nine :
  (¬ ∃ x : ℝ, x > -3 ∧ x^2 ≤ 9) ↔ (∀ x : ℝ, x > -3 → x^2 > 9) := by sorry

-- Statement 3
theorem abs_gt_not_necessary_for_gt :
  ∃ x y : ℝ, (abs x > abs y) ∧ (x ≤ y) := by sorry

-- Statement 4
theorem m_lt_zero_iff_one_positive_one_negative_root :
  ∀ m : ℝ, (m < 0) ↔ 
    (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 - 2*x + m = 0 ∧ y^2 - 2*y + m = 0 ∧ 
      (∀ z : ℝ, z^2 - 2*z + m = 0 → z = x ∨ z = y)) := by sorry

end NUMINAMATH_CALUDE_negation_forall_squared_gt_neg_one_negation_exists_squared_leq_nine_abs_gt_not_necessary_for_gt_m_lt_zero_iff_one_positive_one_negative_root_l1419_141982


namespace NUMINAMATH_CALUDE_arithmetic_geometric_condition_l1419_141948

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) : Prop :=
  b * b = a * c

theorem arithmetic_geometric_condition (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d ∧ a 1 = 2 →
  (d = 4 → geometric_sequence (a 1) (a 2) (a 5)) ∧
  ¬(geometric_sequence (a 1) (a 2) (a 5) → d = 4) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_condition_l1419_141948


namespace NUMINAMATH_CALUDE_no_real_solutions_l1419_141906

theorem no_real_solutions : ¬∃ x : ℝ, x + 36 / (x - 3) = -9 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1419_141906


namespace NUMINAMATH_CALUDE_dave_total_rides_l1419_141950

/-- The number of rides Dave took on the first day -/
def first_day_rides : ℕ := 4

/-- The number of rides Dave took on the second day -/
def second_day_rides : ℕ := 3

/-- The total number of rides Dave took over two days -/
def total_rides : ℕ := first_day_rides + second_day_rides

theorem dave_total_rides : total_rides = 7 := by
  sorry

end NUMINAMATH_CALUDE_dave_total_rides_l1419_141950


namespace NUMINAMATH_CALUDE_fraction_simplification_l1419_141942

theorem fraction_simplification :
  (3 / 7 + 4 / 5) / (5 / 11 + 2 / 9) = 4257 / 2345 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1419_141942


namespace NUMINAMATH_CALUDE_doug_money_l1419_141995

theorem doug_money (j d b s : ℚ) : 
  j + d + b + s = 150 →
  j = 2 * b →
  j = (3/4) * d →
  s = (1/2) * (j + d + b) →
  d = (4/3) * (150 * 12/41) := by
sorry

end NUMINAMATH_CALUDE_doug_money_l1419_141995


namespace NUMINAMATH_CALUDE_triangle_formation_l1419_141934

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_formation :
  triangle_inequality 2 3 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l1419_141934


namespace NUMINAMATH_CALUDE_saramago_readers_l1419_141981

theorem saramago_readers (total_workers : ℕ) (kureishi_readers : ℚ) 
  (both_readers : ℕ) (s : ℚ) : 
  total_workers = 40 →
  kureishi_readers = 5/8 →
  both_readers = 2 →
  (s * total_workers - both_readers - 1 : ℚ) = 
    (total_workers * (1 - kureishi_readers - s) : ℚ) →
  s = 9/40 := by sorry

end NUMINAMATH_CALUDE_saramago_readers_l1419_141981


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_18_24_42_l1419_141975

theorem arithmetic_mean_of_18_24_42 :
  let numbers : List ℕ := [18, 24, 42]
  (numbers.sum : ℚ) / numbers.length = 28 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_18_24_42_l1419_141975


namespace NUMINAMATH_CALUDE_f_max_at_a_l1419_141909

/-- The function f(x) = x^3 - 12x -/
def f (x : ℝ) : ℝ := x^3 - 12*x

/-- The maximum value point of f(x) -/
def a : ℝ := -2

theorem f_max_at_a : IsLocalMax f a := by sorry

end NUMINAMATH_CALUDE_f_max_at_a_l1419_141909


namespace NUMINAMATH_CALUDE_remaining_work_time_for_x_l1419_141944

-- Define the work rates and work durations
def x_rate : ℚ := 1 / 30
def y_rate : ℚ := 1 / 15
def z_rate : ℚ := 1 / 20
def y_work_days : ℕ := 10
def z_work_days : ℕ := 5

-- Define the theorem
theorem remaining_work_time_for_x :
  let total_work : ℚ := 1
  let work_done_by_y : ℚ := y_rate * y_work_days
  let work_done_by_z : ℚ := z_rate * z_work_days
  let remaining_work : ℚ := total_work - (work_done_by_y + work_done_by_z)
  remaining_work / x_rate = 5 / 2 := by sorry

end NUMINAMATH_CALUDE_remaining_work_time_for_x_l1419_141944


namespace NUMINAMATH_CALUDE_alex_and_sam_speeds_l1419_141947

-- Define the variables
def alex_downstream_distance : ℝ := 36
def alex_downstream_time : ℝ := 6
def alex_upstream_time : ℝ := 9
def sam_downstream_distance : ℝ := 48
def sam_downstream_time : ℝ := 8
def sam_upstream_time : ℝ := 12

-- Define the theorem
theorem alex_and_sam_speeds :
  ∃ (alex_speed sam_speed current_speed : ℝ),
    alex_speed > 0 ∧ sam_speed > 0 ∧
    (alex_speed + current_speed) * alex_downstream_time = alex_downstream_distance ∧
    (alex_speed - current_speed) * alex_upstream_time = alex_downstream_distance ∧
    (sam_speed + current_speed) * sam_downstream_time = sam_downstream_distance ∧
    (sam_speed - current_speed) * sam_upstream_time = sam_downstream_distance ∧
    alex_speed = 5 ∧ sam_speed = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_alex_and_sam_speeds_l1419_141947


namespace NUMINAMATH_CALUDE_parallel_lines_parameter_sum_l1419_141905

/-- Given two parallel lines with a specific distance between them, prove that the sum of their parameters is either 3 or -3. -/
theorem parallel_lines_parameter_sum (n m : ℝ) : 
  (∀ x y : ℝ, 2 * x + y + n = 0 ↔ 4 * x + m * y - 4 = 0) →  -- parallelism condition
  (∃ d : ℝ, d = (3 / 5) * Real.sqrt 5 ∧ 
    d = |n + 2| / Real.sqrt 5) →  -- distance condition
  m = 2 →  -- parallelism implies m = 2
  (m + n = 3 ∨ m + n = -3) :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_parameter_sum_l1419_141905


namespace NUMINAMATH_CALUDE_total_production_is_29621_l1419_141984

/-- Represents the production numbers for a specific region -/
structure RegionProduction where
  sedans : Nat
  suvs : Nat
  pickups : Nat

/-- Calculates the total production for a region -/
def total_region_production (r : RegionProduction) : Nat :=
  r.sedans + r.suvs + r.pickups

/-- Represents the production data for all regions -/
structure GlobalProduction where
  north_america : RegionProduction
  europe : RegionProduction
  asia : RegionProduction
  south_america : RegionProduction

/-- Calculates the total global production -/
def total_global_production (g : GlobalProduction) : Nat :=
  total_region_production g.north_america +
  total_region_production g.europe +
  total_region_production g.asia +
  total_region_production g.south_america

/-- The production data for the 5-month period -/
def production_data : GlobalProduction := {
  north_america := { sedans := 3884, suvs := 2943, pickups := 1568 }
  europe := { sedans := 2871, suvs := 2145, pickups := 643 }
  asia := { sedans := 5273, suvs := 3881, pickups := 2338 }
  south_america := { sedans := 1945, suvs := 1365, pickups := 765 }
}

/-- Theorem stating that the total global production equals 29621 -/
theorem total_production_is_29621 :
  total_global_production production_data = 29621 := by
  sorry

end NUMINAMATH_CALUDE_total_production_is_29621_l1419_141984


namespace NUMINAMATH_CALUDE_tyler_sanctuary_species_l1419_141915

/-- The number of pairs of birds per species in Tyler's sanctuary -/
def pairs_per_species : ℕ := 7

/-- The total number of pairs of birds in Tyler's sanctuary -/
def total_pairs : ℕ := 203

/-- The number of endangered bird species in Tyler's sanctuary -/
def num_species : ℕ := total_pairs / pairs_per_species

theorem tyler_sanctuary_species :
  num_species = 29 :=
sorry

end NUMINAMATH_CALUDE_tyler_sanctuary_species_l1419_141915


namespace NUMINAMATH_CALUDE_double_root_values_l1419_141992

def polynomial (b₃ b₂ b₁ : ℤ) (x : ℤ) : ℤ := x^4 + b₃ * x^3 + b₂ * x^2 + b₁ * x + 50

def is_double_root (p : ℤ → ℤ) (s : ℤ) : Prop :=
  p s = 0 ∧ (∃ q : ℤ → ℤ, ∀ x, p x = (x - s)^2 * q x)

theorem double_root_values (b₃ b₂ b₁ : ℤ) (s : ℤ) :
  is_double_root (polynomial b₃ b₂ b₁) s → s ∈ ({-5, -2, -1, 1, 2, 5} : Set ℤ) :=
by sorry

end NUMINAMATH_CALUDE_double_root_values_l1419_141992


namespace NUMINAMATH_CALUDE_coprime_27x_plus_4_and_18x_plus_3_l1419_141988

theorem coprime_27x_plus_4_and_18x_plus_3 (x : ℕ) : Nat.gcd (27 * x + 4) (18 * x + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_coprime_27x_plus_4_and_18x_plus_3_l1419_141988


namespace NUMINAMATH_CALUDE_max_blocks_fit_l1419_141962

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ := d.length * d.width * d.height

/-- Calculates the number of blocks that can fit in one layer of the larger box -/
def blocksPerLayer (largeBox : BoxDimensions) (smallBox : BoxDimensions) : ℕ :=
  (largeBox.length / smallBox.length) * (largeBox.width / smallBox.width)

/-- The main theorem stating the maximum number of blocks that can fit -/
theorem max_blocks_fit (largeBox smallBox : BoxDimensions) :
  largeBox = BoxDimensions.mk 5 4 4 →
  smallBox = BoxDimensions.mk 3 2 1 →
  blocksPerLayer largeBox smallBox * (largeBox.height / smallBox.height) = 12 :=
sorry

end NUMINAMATH_CALUDE_max_blocks_fit_l1419_141962


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_l1419_141999

/-- The expression to be simplified -/
def expression (x : ℝ) : ℝ := 3 * (x^3 - 2*x^2 + 3) - 5 * (x^4 - 4*x^2 + 2)

/-- The coefficients of the fully simplified expression -/
def coefficients : List ℝ := [-5, 3, 14, -1]

/-- Theorem: The sum of the squares of the coefficients of the fully simplified expression is 231 -/
theorem sum_of_squared_coefficients :
  (coefficients.map (λ c => c^2)).sum = 231 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_l1419_141999


namespace NUMINAMATH_CALUDE_largest_integer_l1419_141989

theorem largest_integer (a b c d : ℤ) 
  (sum1 : a + b + c = 163)
  (sum2 : a + b + d = 178)
  (sum3 : a + c + d = 184)
  (sum4 : b + c + d = 194) :
  max a (max b (max c d)) = 77 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_l1419_141989


namespace NUMINAMATH_CALUDE_product_of_four_is_perfect_square_l1419_141955

theorem product_of_four_is_perfect_square 
  (nums : Finset ℕ) 
  (h_card : nums.card = 48) 
  (h_primes : (nums.prod id).factorization.support.card = 10) : 
  ∃ (subset : Finset ℕ), subset ⊆ nums ∧ subset.card = 4 ∧ 
  ∃ (m : ℕ), (subset.prod id) = m^2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_four_is_perfect_square_l1419_141955


namespace NUMINAMATH_CALUDE_f_monotone_and_inequality_l1419_141987

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.log x - a * x + a

theorem f_monotone_and_inequality (a : ℝ) : 
  (a > 0 ∧ a ≤ 2) ↔ 
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → f a x < f a y) ∧
  (∀ x : ℝ, x > 0 → (x - 1) * f a x ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_f_monotone_and_inequality_l1419_141987


namespace NUMINAMATH_CALUDE_alcohol_percentage_in_first_vessel_alcohol_percentage_proof_l1419_141901

theorem alcohol_percentage_in_first_vessel : ℝ → Prop :=
  fun x =>
    let vessel1_capacity : ℝ := 2
    let vessel2_capacity : ℝ := 6
    let vessel2_alcohol_percentage : ℝ := 40
    let total_liquid : ℝ := 8
    let new_mixture_concentration : ℝ := 30

    let vessel2_alcohol_amount : ℝ := vessel2_capacity * (vessel2_alcohol_percentage / 100)
    let total_alcohol_amount : ℝ := total_liquid * (new_mixture_concentration / 100)
    let vessel1_alcohol_amount : ℝ := vessel1_capacity * (x / 100)

    vessel1_alcohol_amount + vessel2_alcohol_amount = total_alcohol_amount →
    x = 0

theorem alcohol_percentage_proof : alcohol_percentage_in_first_vessel 0 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_percentage_in_first_vessel_alcohol_percentage_proof_l1419_141901


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l1419_141924

theorem matrix_multiplication_result : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 5, 2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![0, 6; -2, 1]
  A * B = !![6, 21; -4, 32] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l1419_141924


namespace NUMINAMATH_CALUDE_cost_for_23_days_l1419_141990

/-- Calculates the total cost of staying in a student youth hostel for a given number of days. -/
def hostelCost (days : ℕ) : ℚ :=
  let firstWeekRate : ℚ := 18
  let additionalWeekRate : ℚ := 13
  let firstWeekDays : ℕ := min days 7
  let additionalDays : ℕ := days - firstWeekDays
  firstWeekRate * firstWeekDays + additionalWeekRate * additionalDays

/-- Proves that the cost of staying for 23 days in the student youth hostel is $334.00. -/
theorem cost_for_23_days : hostelCost 23 = 334 := by
  sorry

end NUMINAMATH_CALUDE_cost_for_23_days_l1419_141990


namespace NUMINAMATH_CALUDE_fraction_simplification_l1419_141993

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  2 / (x - 1) - 1 / (x + 1) = (x + 3) / (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1419_141993


namespace NUMINAMATH_CALUDE_max_value_theorem_max_value_achievable_l1419_141922

theorem max_value_theorem (x y : ℝ) :
  (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 2) ≤ Real.sqrt 29 :=
by sorry

theorem max_value_achievable :
  ∃ x y : ℝ, (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 2) = Real.sqrt 29 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_max_value_achievable_l1419_141922


namespace NUMINAMATH_CALUDE_proper_subsets_of_A_l1419_141959

def U : Finset ℕ := {0,1,2,3,4,5}

def C_U_A : Finset ℕ := {1,2,3}

def A : Finset ℕ := U \ C_U_A

theorem proper_subsets_of_A : Finset.card (Finset.powerset A \ {A}) = 7 := by
  sorry

end NUMINAMATH_CALUDE_proper_subsets_of_A_l1419_141959


namespace NUMINAMATH_CALUDE_logarithm_sum_simplification_l1419_141977

theorem logarithm_sum_simplification :
  1 / (Real.log 3 / Real.log 12 + 1) +
  1 / (Real.log 2 / Real.log 8 + 1) +
  1 / (Real.log 7 / Real.log 14 + 1) = 1.5 := by sorry

end NUMINAMATH_CALUDE_logarithm_sum_simplification_l1419_141977


namespace NUMINAMATH_CALUDE_prime_odd_sum_product_l1419_141913

theorem prime_odd_sum_product (p q : ℕ) : 
  Prime p → 
  Odd q → 
  q > 0 → 
  p^2 + q = 125 → 
  p * q = 242 := by
sorry

end NUMINAMATH_CALUDE_prime_odd_sum_product_l1419_141913


namespace NUMINAMATH_CALUDE_abc_inequality_l1419_141963

theorem abc_inequality (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) : 
  a + b + c + 2 * a * b * c > a * b + b * c + c * a + 2 * Real.sqrt (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1419_141963


namespace NUMINAMATH_CALUDE_machine_work_rate_l1419_141920

theorem machine_work_rate (x : ℝ) : 
  (1 / (x + 4) + 1 / (x + 3) + 1 / (x + 2) = 1 / x) → x = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_machine_work_rate_l1419_141920


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_equals_four_l1419_141976

theorem sum_of_a_and_b_equals_four (a b : ℝ) (h : b + (a - 2) * Complex.I = 1 + Complex.I) : a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_equals_four_l1419_141976


namespace NUMINAMATH_CALUDE_min_sum_squares_reciprocal_inequality_l1419_141970

-- Define the set D
def D : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2 ∧ p.1 > 0 ∧ p.2 > 0}

-- Theorem 1: Minimum value of x₁² + x₂²
theorem min_sum_squares (p : ℝ × ℝ) (h : p ∈ D) : p.1^2 + p.2^2 ≥ 2 := by
  sorry

-- Theorem 2: Inequality for reciprocals
theorem reciprocal_inequality (p : ℝ × ℝ) (h : p ∈ D) :
  1 / (p.1 + 2*p.2) + 1 / (2*p.1 + p.2) ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_reciprocal_inequality_l1419_141970


namespace NUMINAMATH_CALUDE_cost_price_of_toy_cost_price_is_800_l1419_141923

/-- The cost price of a toy given the selling price and gain conditions -/
theorem cost_price_of_toy (total_sale : ℕ) (num_toys : ℕ) (gain_in_toys : ℕ) : ℕ :=
  let selling_price := total_sale / num_toys
  let cost_price := selling_price / (1 + gain_in_toys / num_toys)
  cost_price
  
/-- Proof that the cost price of a toy is 800 given the conditions -/
theorem cost_price_is_800 : cost_price_of_toy 16800 18 3 = 800 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_of_toy_cost_price_is_800_l1419_141923


namespace NUMINAMATH_CALUDE_percentage_equation_l1419_141961

theorem percentage_equation (x : ℝ) : (35 / 100 * 400 = 20 / 100 * x) → x = 700 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equation_l1419_141961


namespace NUMINAMATH_CALUDE_x_0_value_l1419_141925

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem x_0_value (x₀ : ℝ) (h : x₀ > 0) :
  (deriv f x₀ = 2) → x₀ = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_x_0_value_l1419_141925


namespace NUMINAMATH_CALUDE_stratified_sampling_male_count_l1419_141900

theorem stratified_sampling_male_count 
  (total_students : ℕ) 
  (male_students : ℕ) 
  (female_students : ℕ) 
  (sample_size : ℕ) :
  total_students = male_students + female_students →
  total_students = 700 →
  male_students = 400 →
  female_students = 300 →
  sample_size = 35 →
  (male_students * sample_size) / total_students = 20 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_male_count_l1419_141900


namespace NUMINAMATH_CALUDE_part_time_employees_l1419_141953

/-- Represents the number of employees in a corporation -/
structure Corporation where
  total : ℕ
  fullTime : ℕ
  partTime : ℕ

/-- The total number of employees is the sum of full-time and part-time employees -/
axiom total_eq_sum (c : Corporation) : c.total = c.fullTime + c.partTime

/-- Theorem: Given a corporation with 65,134 total employees and 63,093 full-time employees,
    the number of part-time employees is 2,041 -/
theorem part_time_employees (c : Corporation) 
    (h1 : c.total = 65134) 
    (h2 : c.fullTime = 63093) : 
    c.partTime = 2041 := by
  sorry


end NUMINAMATH_CALUDE_part_time_employees_l1419_141953


namespace NUMINAMATH_CALUDE_circumscribed_equal_triangulation_only_square_l1419_141941

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : n > 3

/-- A polygon is circumscribed if all its sides are tangent to a common circle -/
def IsCircumscribed (P : ConvexPolygon n) : Prop :=
  sorry

/-- A polygon can be dissected into equal triangles by non-intersecting diagonals -/
def HasEqualTriangulation (P : ConvexPolygon n) : Prop :=
  sorry

/-- The main theorem -/
theorem circumscribed_equal_triangulation_only_square
  (n : ℕ) (P : ConvexPolygon n)
  (h_circ : IsCircumscribed P)
  (h_triang : HasEqualTriangulation P) :
  n = 4 :=
sorry

end NUMINAMATH_CALUDE_circumscribed_equal_triangulation_only_square_l1419_141941


namespace NUMINAMATH_CALUDE_at_least_three_babies_speak_l1419_141998

def probability_baby_speaks : ℚ := 2/5

def number_of_babies : ℕ := 6

def probability_at_least_three_speak : ℚ := 7120/15625

theorem at_least_three_babies_speak :
  probability_at_least_three_speak =
    1 - (Nat.choose number_of_babies 0 * (1 - probability_baby_speaks)^number_of_babies +
         Nat.choose number_of_babies 1 * probability_baby_speaks * (1 - probability_baby_speaks)^(number_of_babies - 1) +
         Nat.choose number_of_babies 2 * probability_baby_speaks^2 * (1 - probability_baby_speaks)^(number_of_babies - 2)) :=
by sorry

end NUMINAMATH_CALUDE_at_least_three_babies_speak_l1419_141998


namespace NUMINAMATH_CALUDE_total_green_marbles_l1419_141938

/-- The number of green marbles Sara has -/
def sara_green : ℕ := 3

/-- The number of green marbles Tom has -/
def tom_green : ℕ := 4

/-- The total number of green marbles Sara and Tom have -/
def total_green : ℕ := sara_green + tom_green

theorem total_green_marbles : total_green = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_green_marbles_l1419_141938


namespace NUMINAMATH_CALUDE_algorithm_characteristic_is_determinacy_l1419_141978

-- Define the concept of an algorithm step
structure AlgorithmStep where
  definite : Bool
  executable : Bool
  yieldsDefiniteResult : Bool

-- Define the characteristic of determinacy
def isDeterminacy (step : AlgorithmStep) : Prop :=
  step.definite ∧ step.executable ∧ step.yieldsDefiniteResult

-- Theorem statement
theorem algorithm_characteristic_is_determinacy (step : AlgorithmStep) :
  step.definite ∧ step.executable ∧ step.yieldsDefiniteResult → isDeterminacy step :=
by
  sorry

#check algorithm_characteristic_is_determinacy

end NUMINAMATH_CALUDE_algorithm_characteristic_is_determinacy_l1419_141978


namespace NUMINAMATH_CALUDE_stationery_box_sheets_l1419_141932

/-- Represents the contents of a stationery box -/
structure StationeryBox where
  sheets : ℕ
  envelopes : ℕ

/-- Represents a person who uses the stationery box -/
structure Person where
  name : String
  box : StationeryBox
  pagesPerLetter : ℕ

theorem stationery_box_sheets (ann sue : Person) : 
  ann.name = "Ann" →
  sue.name = "Sue" →
  ann.pagesPerLetter = 1 →
  sue.pagesPerLetter = 3 →
  ann.box = sue.box →
  ann.box.sheets - ann.box.envelopes = 50 →
  sue.box.envelopes - sue.box.sheets / 3 = 50 →
  ann.box.sheets = 150 := by
sorry

end NUMINAMATH_CALUDE_stationery_box_sheets_l1419_141932


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1419_141994

/-- Given a boat traveling downstream with a current of 5 km/hr,
    if it covers a distance of 7.5 km in 18 minutes,
    then its speed in still water is 20 km/hr. -/
theorem boat_speed_in_still_water 
  (current_speed : ℝ) 
  (distance_downstream : ℝ) 
  (time_minutes : ℝ) 
  (boat_speed : ℝ) :
  current_speed = 5 →
  distance_downstream = 7.5 →
  time_minutes = 18 →
  distance_downstream = (boat_speed + current_speed) * (time_minutes / 60) →
  boat_speed = 20 :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1419_141994


namespace NUMINAMATH_CALUDE_projection_x_coordinate_l1419_141969

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Theorem: The x-coordinate of the projection of a point on a circle onto the x-axis -/
theorem projection_x_coordinate 
  (circle : Circle)
  (start : Point)
  (B : Point)
  (angle : ℝ) :
  circle.center = Point.mk 0 0 →
  circle.radius = 4 →
  start = Point.mk 4 0 →
  B.x = 4 * Real.cos angle →
  B.y = 4 * Real.sin angle →
  angle ≥ 0 →
  4 * Real.cos angle = (Point.mk (B.x) 0).x :=
by sorry

end NUMINAMATH_CALUDE_projection_x_coordinate_l1419_141969


namespace NUMINAMATH_CALUDE_triangle_inequality_l1419_141964

open Real

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point M
def M : ℝ × ℝ := sorry

-- Define the semi-perimeter p
def semiPerimeter (t : Triangle) : ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle between three points
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_inequality (t : Triangle) :
  let p := semiPerimeter t
  distance M t.A * cos (angle t.B t.A t.C / 2) +
  distance M t.B * cos (angle t.A t.B t.C / 2) +
  distance M t.C * cos (angle t.A t.C t.B / 2) ≥ p := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1419_141964


namespace NUMINAMATH_CALUDE_seonyeong_class_size_l1419_141960

/-- The number of rows of students -/
def num_rows : ℕ := 12

/-- The number of students in each row -/
def students_per_row : ℕ := 4

/-- The number of additional students -/
def additional_students : ℕ := 3

/-- The number of students in Jieun's class -/
def jieun_class_size : ℕ := 12

/-- The total number of students -/
def total_students : ℕ := num_rows * students_per_row + additional_students

/-- Theorem: The number of students in Seonyeong's class is 39 -/
theorem seonyeong_class_size : total_students - jieun_class_size = 39 := by
  sorry

end NUMINAMATH_CALUDE_seonyeong_class_size_l1419_141960
