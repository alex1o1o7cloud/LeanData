import Mathlib

namespace NUMINAMATH_CALUDE_units_digit_of_sequence_sum_l2732_273292

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sequence_term (n : ℕ) : ℕ := factorial n + n

def sum_sequence (n : ℕ) : ℕ := (List.range n).map sequence_term |>.sum

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_sequence_sum :
  units_digit (sum_sequence 12) = 1 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_sequence_sum_l2732_273292


namespace NUMINAMATH_CALUDE_square_root_of_nine_l2732_273256

-- Define the square root operation
def square_root (x : ℝ) : Set ℝ := {y : ℝ | y^2 = x}

-- Theorem statement
theorem square_root_of_nine : square_root 9 = {-3, 3} := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l2732_273256


namespace NUMINAMATH_CALUDE_max_value_interval_l2732_273299

open Real

noncomputable def f (a x : ℝ) : ℝ := 3 * log x - x^2 + (a - 1/2) * x

theorem max_value_interval (a : ℝ) :
  (∃ x ∈ Set.Ioo 1 3, ∀ y ∈ Set.Ioo 1 3, f a x ≥ f a y) ↔ a ∈ Set.Ioo (-1/2) (11/2) := by
  sorry

end NUMINAMATH_CALUDE_max_value_interval_l2732_273299


namespace NUMINAMATH_CALUDE_sequence_general_term_l2732_273293

/-- Given a sequence {a_n} with sum of first n terms S_n = (2/3)a_n + 1/3,
    prove that the general term formula is a_n = (-2)^(n-1) -/
theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, S n = (2/3) * a n + 1/3) →
  ∃ C : ℝ, ∀ n : ℕ, a n = C * (-2)^(n-1) :=
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2732_273293


namespace NUMINAMATH_CALUDE_pipe_cutting_time_l2732_273230

/-- The time needed to cut a pipe into sections -/
def cut_time (sections : ℕ) (time_per_cut : ℕ) : ℕ :=
  (sections - 1) * time_per_cut

/-- Theorem: The time needed to cut a pipe into 5 sections is 24 minutes -/
theorem pipe_cutting_time : cut_time 5 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_pipe_cutting_time_l2732_273230


namespace NUMINAMATH_CALUDE_ascending_order_abc_l2732_273275

theorem ascending_order_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (eq_a : Real.cos a = a) (eq_b : Real.sin (Real.cos b) = b) (eq_c : Real.cos (Real.sin c) = c) :
  b < a ∧ a < c :=
by sorry

end NUMINAMATH_CALUDE_ascending_order_abc_l2732_273275


namespace NUMINAMATH_CALUDE_b_contribution_is_16200_l2732_273265

/-- Calculates the partner's contribution given the investment details and profit ratio -/
def calculate_partner_contribution (a_investment : ℕ) (total_months : ℕ) (b_join_month : ℕ) (a_profit_share : ℕ) (b_profit_share : ℕ) : ℕ :=
  let a_months := total_months
  let b_months := total_months - b_join_month
  (a_investment * a_months * b_profit_share) / (a_profit_share * b_months)

/-- Proves that B's contribution to the capital is 16200 rs given the problem conditions -/
theorem b_contribution_is_16200 :
  let a_investment := 4500
  let total_months := 12
  let b_join_month := 7
  let a_profit_share := 2
  let b_profit_share := 3
  calculate_partner_contribution a_investment total_months b_join_month a_profit_share b_profit_share = 16200 := by
  sorry

end NUMINAMATH_CALUDE_b_contribution_is_16200_l2732_273265


namespace NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_equals_3_l2732_273250

theorem sqrt_27_div_sqrt_3_equals_3 : Real.sqrt 27 / Real.sqrt 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_equals_3_l2732_273250


namespace NUMINAMATH_CALUDE_number_of_valid_divisors_l2732_273231

def total_marbles : ℕ := 720

theorem number_of_valid_divisors :
  (Finset.filter (fun m => m > 1 ∧ m < total_marbles ∧ total_marbles % m = 0) 
    (Finset.range (total_marbles + 1))).card = 28 := by
  sorry

end NUMINAMATH_CALUDE_number_of_valid_divisors_l2732_273231


namespace NUMINAMATH_CALUDE_complement_of_N_in_U_l2732_273226

def U : Set ℕ := {x | x > 0 ∧ x ≤ 5}
def N : Set ℕ := {2, 4}

theorem complement_of_N_in_U :
  U \ N = {1, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_N_in_U_l2732_273226


namespace NUMINAMATH_CALUDE_vector_operation_l2732_273232

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)

theorem vector_operation : 
  (3 • a - 2 • b : ℝ × ℝ) = (1, 5) := by sorry

end NUMINAMATH_CALUDE_vector_operation_l2732_273232


namespace NUMINAMATH_CALUDE_log_ratio_squared_l2732_273296

theorem log_ratio_squared (a b : ℝ) (ha : a > 0) (hb : b > 0) (ha1 : a ≠ 1) (hb1 : b ≠ 1)
  (h1 : Real.log a / Real.log 3 = Real.log 81 / Real.log b) (h2 : a * b = 243) :
  (Real.log (a / b) / Real.log 3)^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_log_ratio_squared_l2732_273296


namespace NUMINAMATH_CALUDE_marble_problem_l2732_273208

theorem marble_problem (katrina amanda carlos mabel : ℕ) : 
  amanda + 12 = 2 * katrina →
  mabel = 5 * katrina →
  carlos = 3 * katrina →
  mabel = 85 →
  mabel - (amanda + carlos) = 12 := by
sorry

end NUMINAMATH_CALUDE_marble_problem_l2732_273208


namespace NUMINAMATH_CALUDE_cylinder_ellipse_intersection_l2732_273249

/-- Represents a right circular cylinder -/
structure RightCircularCylinder where
  radius : ℝ

/-- Represents an ellipse formed by a plane intersecting a cylinder -/
structure Ellipse where
  majorAxis : ℝ
  minorAxis : ℝ

/-- The theorem stating the relationship between the cylinder and the ellipse -/
theorem cylinder_ellipse_intersection
  (c : RightCircularCylinder)
  (e : Ellipse)
  (h1 : c.radius = 3)
  (h2 : e.minorAxis = 2 * c.radius)
  (h3 : e.majorAxis = e.minorAxis * 1.6)
  : e.majorAxis = 9.6 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_ellipse_intersection_l2732_273249


namespace NUMINAMATH_CALUDE_no_solution_iff_a_leq_8_l2732_273217

theorem no_solution_iff_a_leq_8 :
  ∀ a : ℝ, (∀ x : ℝ, ¬(|x - 5| + |x + 3| < a)) ↔ a ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_a_leq_8_l2732_273217


namespace NUMINAMATH_CALUDE_function_inequality_l2732_273215

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_periodic_2 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) = f (x - 1)

def monotone_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem function_inequality (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_periodic : is_periodic_2 f)
  (h_monotone : monotone_increasing_on f 0 1) :
  f (-3/2) < f (4/3) ∧ f (4/3) < f 1 :=
sorry

end NUMINAMATH_CALUDE_function_inequality_l2732_273215


namespace NUMINAMATH_CALUDE_max_value_of_exponential_difference_l2732_273207

theorem max_value_of_exponential_difference :
  ∃ (M : ℝ), M = 1/4 ∧ ∀ x : ℝ, 5^x - 25^x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_exponential_difference_l2732_273207


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2732_273288

/-- The eccentricity of a hyperbola with equation y²/9 - x²/4 = 1 is √13/3 -/
theorem hyperbola_eccentricity : ∃ e : ℝ, e = Real.sqrt 13 / 3 ∧
  ∀ x y : ℝ, y^2 / 9 - x^2 / 4 = 1 → 
  e = Real.sqrt ((3:ℝ)^2 + (2:ℝ)^2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2732_273288


namespace NUMINAMATH_CALUDE_novelists_count_l2732_273280

theorem novelists_count (total : ℕ) (ratio_novelists : ℕ) (ratio_poets : ℕ) (novelists : ℕ) : 
  total = 24 →
  ratio_novelists = 5 →
  ratio_poets = 3 →
  ratio_novelists + ratio_poets = novelists + (total - novelists) →
  novelists * (ratio_novelists + ratio_poets) = total * ratio_novelists →
  novelists = 15 := by
sorry

end NUMINAMATH_CALUDE_novelists_count_l2732_273280


namespace NUMINAMATH_CALUDE_largest_decimal_l2732_273204

theorem largest_decimal : 
  let a := 0.938
  let b := 0.9389
  let c := 0.93809
  let d := 0.839
  let e := 0.893
  b = max a (max b (max c (max d e))) := by
  sorry

end NUMINAMATH_CALUDE_largest_decimal_l2732_273204


namespace NUMINAMATH_CALUDE_vacation_cost_l2732_273228

theorem vacation_cost (C : ℝ) : C / 3 - C / 4 = 40 → C = 480 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_l2732_273228


namespace NUMINAMATH_CALUDE_sinusoidal_function_values_l2732_273209

theorem sinusoidal_function_values (a b : ℝ) (h_a_neg : a < 0) :
  (∀ x, a * Real.sin x + b ≤ 3) ∧
  (∀ x, a * Real.sin x + b ≥ -1) ∧
  (∃ x, a * Real.sin x + b = 3) ∧
  (∃ x, a * Real.sin x + b = -1) →
  a = -2 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_sinusoidal_function_values_l2732_273209


namespace NUMINAMATH_CALUDE_smallest_X_value_l2732_273220

/-- A function that checks if a natural number is composed only of 0s and 1s -/
def isComposedOf0sAnd1s (n : ℕ) : Prop := sorry

/-- The smallest positive integer composed of 0s and 1s that is divisible by 6 -/
def smallestValidT : ℕ := 1110

theorem smallest_X_value :
  ∀ T : ℕ,
  T > 0 →
  isComposedOf0sAnd1s T →
  T % 6 = 0 →
  T / 6 ≥ 185 :=
sorry

end NUMINAMATH_CALUDE_smallest_X_value_l2732_273220


namespace NUMINAMATH_CALUDE_sixThousandthTerm_l2732_273240

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term
  a₁ : ℝ
  -- Common difference
  d : ℝ
  -- Parameters p and r
  p : ℝ
  r : ℝ
  -- Conditions on the first four terms
  h₁ : a₁ = 2 * p
  h₂ : a₁ + d = 14
  h₃ : a₁ + 2 * d = 4 * p - r
  h₄ : a₁ + 3 * d = 4 * p + r

/-- The nth term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a₁ + (n - 1) * seq.d

/-- Theorem stating that the 6000th term is 24006 -/
theorem sixThousandthTerm (seq : ArithmeticSequence) : nthTerm seq 6000 = 24006 := by
  sorry

end NUMINAMATH_CALUDE_sixThousandthTerm_l2732_273240


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2732_273218

/-- Given a geometric sequence {a_n} with common ratio q > 0,
    where a_2 = 1 and a_{n+2} + a_{n+1} = 6a_n,
    prove that the sum of the first four terms (S_4) is equal to 15/2. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h_q_pos : q > 0)
  (h_geometric : ∀ n, a (n + 1) = a n * q)
  (h_a2 : a 2 = 1)
  (h_relation : ∀ n, a (n + 2) + a (n + 1) = 6 * a n) :
  a 1 + a 2 + a 3 + a 4 = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2732_273218


namespace NUMINAMATH_CALUDE_dodecagon_diagonal_equality_l2732_273242

/-- A regular dodecagon -/
structure RegularDodecagon where
  /-- Side length of the dodecagon -/
  a : ℝ
  /-- Length of shortest diagonal spanning three sides -/
  b : ℝ
  /-- Length of longest diagonal spanning six sides -/
  d : ℝ
  /-- Positive side length -/
  a_pos : 0 < a

/-- In a regular dodecagon, the length of the shortest diagonal spanning three sides
    is equal to the length of the longest diagonal spanning six sides -/
theorem dodecagon_diagonal_equality (poly : RegularDodecagon) : poly.b = poly.d := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonal_equality_l2732_273242


namespace NUMINAMATH_CALUDE_asima_integer_possibilities_l2732_273283

theorem asima_integer_possibilities (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
  (h3 : 4 * (2 * a - 10) + 4 * (2 * b - 10) = 440) :
  ∃ (n : ℕ), n = 64 ∧ (∀ x : ℕ, x > 0 ∧ x ≤ n → ∃ y : ℕ, y > 0 ∧ 4 * (2 * x - 10) + 4 * (2 * y - 10) = 440) :=
sorry

end NUMINAMATH_CALUDE_asima_integer_possibilities_l2732_273283


namespace NUMINAMATH_CALUDE_max_value_theorem_l2732_273279

theorem max_value_theorem (x y : ℝ) (h : 4 * x^2 + y^2 + x * y = 1) :
  ∃ (max : ℝ), (∀ (x' y' : ℝ), 4 * x'^2 + y'^2 + x' * y' = 1 → 2 * x' + y' ≤ max) ∧
                max = 2 * Real.sqrt 10 / 5 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2732_273279


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l2732_273248

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 9*x + y = x*y) :
  x + y ≥ 16 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 9*x₀ + y₀ = x₀*y₀ ∧ x₀ + y₀ = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l2732_273248


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2732_273258

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₁ + a₉ = 10, prove that a₅ = 5 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 1 + a 9 = 10) : 
  a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2732_273258


namespace NUMINAMATH_CALUDE_variance_of_data_l2732_273200

/-- Given a list of 5 real numbers with an average of 5 and an average of squares of 33,
    prove that the variance of the list is 8. -/
theorem variance_of_data (x : List ℝ) (hx : x.length = 5)
  (h_avg : x.sum / 5 = 5)
  (h_avg_sq : (x.map (λ xi => xi^2)).sum / 5 = 33) :
  (x.map (λ xi => (xi - 5)^2)).sum / 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_data_l2732_273200


namespace NUMINAMATH_CALUDE_range_of_f_l2732_273266

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x

-- Define the domain
def domain : Set ℝ := { x | 1 ≤ x ∧ x < 5 }

-- State the theorem
theorem range_of_f :
  { y | ∃ x ∈ domain, f x = y } = { y | -4 ≤ y ∧ y < 5 } := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2732_273266


namespace NUMINAMATH_CALUDE_dig_time_proof_l2732_273272

/-- Represents the time (in days) it takes for a person to dig a well alone -/
structure DigTime :=
  (days : ℝ)
  (pos : days > 0)

/-- Given the dig times for three people and their combined dig time,
    proves that if two people's dig times are 24 and 48 days,
    the third person's dig time is 16 days -/
theorem dig_time_proof
  (combined_time : ℝ)
  (combined_time_pos : combined_time > 0)
  (combined_time_eq : combined_time = 8)
  (time1 time2 time3 : DigTime)
  (time2_eq : time2.days = 24)
  (time3_eq : time3.days = 48)
  (combined_rate_eq : 1 / combined_time = 1 / time1.days + 1 / time2.days + 1 / time3.days) :
  time1.days = 16 := by
sorry


end NUMINAMATH_CALUDE_dig_time_proof_l2732_273272


namespace NUMINAMATH_CALUDE_janelle_has_72_marbles_l2732_273235

/-- The number of marbles Janelle has after buying blue marbles and giving some away as a gift. -/
def janelles_marbles : ℕ :=
  let initial_green : ℕ := 26
  let blue_bags : ℕ := 6
  let marbles_per_bag : ℕ := 10
  let gift_green : ℕ := 6
  let gift_blue : ℕ := 8
  let total_blue : ℕ := blue_bags * marbles_per_bag
  let total_before_gift : ℕ := initial_green + total_blue
  let total_gift : ℕ := gift_green + gift_blue
  total_before_gift - total_gift

/-- Theorem stating that Janelle has 72 marbles after the transactions. -/
theorem janelle_has_72_marbles : janelles_marbles = 72 := by
  sorry

end NUMINAMATH_CALUDE_janelle_has_72_marbles_l2732_273235


namespace NUMINAMATH_CALUDE_betty_morning_flies_l2732_273290

/-- The number of flies Betty caught in the morning -/
def morning_flies : ℕ := 5

/-- The number of flies a frog eats per day -/
def flies_per_day : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of flies Betty caught in the afternoon -/
def afternoon_flies : ℕ := 6

/-- The number of flies that escaped -/
def escaped_flies : ℕ := 1

/-- The number of additional flies Betty needs -/
def additional_flies_needed : ℕ := 4

theorem betty_morning_flies :
  morning_flies = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_betty_morning_flies_l2732_273290


namespace NUMINAMATH_CALUDE_adams_dog_food_packages_l2732_273233

theorem adams_dog_food_packages (cat_packages : ℕ) (cat_cans_per_package : ℕ) (dog_cans_per_package : ℕ) (cat_dog_can_difference : ℕ) :
  cat_packages = 9 →
  cat_cans_per_package = 10 →
  dog_cans_per_package = 5 →
  cat_dog_can_difference = 55 →
  ∃ (dog_packages : ℕ),
    cat_packages * cat_cans_per_package = dog_packages * dog_cans_per_package + cat_dog_can_difference ∧
    dog_packages = 7 :=
by
  sorry


end NUMINAMATH_CALUDE_adams_dog_food_packages_l2732_273233


namespace NUMINAMATH_CALUDE_system_solution_l2732_273206

theorem system_solution : ∃ (x y : ℝ), 
  (x - 2*y = 3) ∧ (3*x - y = 4) ∧ (x = 1) ∧ (y = -1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2732_273206


namespace NUMINAMATH_CALUDE_animal_ratio_proof_l2732_273247

/-- Given ratios between animals, prove the final ratio of all animals -/
theorem animal_ratio_proof 
  (chicken_pig_ratio : ℚ × ℚ)
  (sheep_horse_ratio : ℚ × ℚ)
  (pig_horse_ratio : ℚ × ℚ)
  (h1 : chicken_pig_ratio = (26, 5))
  (h2 : sheep_horse_ratio = (25, 9))
  (h3 : pig_horse_ratio = (10, 3)) :
  ∃ (k : ℚ), k > 0 ∧ 
    k * 156 = chicken_pig_ratio.1 * pig_horse_ratio.2 ∧
    k * 30 = chicken_pig_ratio.2 * pig_horse_ratio.2 ∧
    k * 9 = pig_horse_ratio.2 ∧
    k * 25 = sheep_horse_ratio.1 * pig_horse_ratio.2 / sheep_horse_ratio.2 :=
by
  sorry

end NUMINAMATH_CALUDE_animal_ratio_proof_l2732_273247


namespace NUMINAMATH_CALUDE_bowling_ball_weight_calculation_l2732_273281

-- Define the weight of a canoe
def canoe_weight : ℚ := 32

-- Define the number of canoes and bowling balls
def num_canoes : ℕ := 4
def num_bowling_balls : ℕ := 5

-- Define the total weight of canoes
def total_canoe_weight : ℚ := num_canoes * canoe_weight

-- Define the weight of one bowling ball
def bowling_ball_weight : ℚ := total_canoe_weight / num_bowling_balls

-- Theorem statement
theorem bowling_ball_weight_calculation :
  bowling_ball_weight = 128 / 5 := by sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_calculation_l2732_273281


namespace NUMINAMATH_CALUDE_history_books_count_l2732_273269

theorem history_books_count (total : ℕ) (reading : ℕ) (math : ℕ) (science : ℕ) (history : ℕ) : 
  total = 10 →
  reading = 2 * total / 5 →
  math = 3 * total / 10 →
  science = math - 1 →
  history = total - (reading + math + science) →
  history = 1 := by
sorry

end NUMINAMATH_CALUDE_history_books_count_l2732_273269


namespace NUMINAMATH_CALUDE_ellipse_line_slope_l2732_273243

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 3 = 1

-- Define a line l
def line (k m : ℝ) (x y : ℝ) : Prop := y = k * x + m

-- Define the midpoint condition
def is_midpoint (x₁ y₁ x₂ y₂ xₘ yₘ : ℝ) : Prop :=
  xₘ = (x₁ + x₂) / 2 ∧ yₘ = (y₁ + y₂) / 2

-- Theorem statement
theorem ellipse_line_slope (x₁ y₁ x₂ y₂ k m : ℝ) :
  ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
  line k m x₁ y₁ ∧ line k m x₂ y₂ ∧
  is_midpoint x₁ y₁ x₂ y₂ 1 1 →
  k = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_line_slope_l2732_273243


namespace NUMINAMATH_CALUDE_seating_arrangement_solution_l2732_273251

/-- Represents a seating arrangement --/
structure SeatingArrangement where
  total_people : ℕ
  row_size_1 : ℕ
  row_size_2 : ℕ
  rows_of_size_1 : ℕ
  rows_of_size_2 : ℕ

/-- Defines a valid seating arrangement --/
def is_valid_arrangement (s : SeatingArrangement) : Prop :=
  s.total_people = s.row_size_1 * s.rows_of_size_1 + s.row_size_2 * s.rows_of_size_2

/-- The specific seating arrangement for our problem --/
def problem_arrangement : SeatingArrangement :=
  { total_people := 58
  , row_size_1 := 7
  , row_size_2 := 9
  , rows_of_size_1 := 7  -- This value is not given in the problem, but needed for the structure
  , rows_of_size_2 := 1  -- This is what we want to prove
  }

/-- The main theorem to prove --/
theorem seating_arrangement_solution :
  is_valid_arrangement problem_arrangement ∧
  ∀ s : SeatingArrangement,
    s.total_people = problem_arrangement.total_people ∧
    s.row_size_1 = problem_arrangement.row_size_1 ∧
    s.row_size_2 = problem_arrangement.row_size_2 ∧
    is_valid_arrangement s →
    s.rows_of_size_2 = problem_arrangement.rows_of_size_2 :=
by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_solution_l2732_273251


namespace NUMINAMATH_CALUDE_test_question_count_l2732_273202

theorem test_question_count (total_points : ℕ) (total_questions : ℕ) 
  (h1 : total_points = 200)
  (h2 : total_questions = 30)
  (h3 : ∃ (five_point_count ten_point_count : ℕ), 
    five_point_count + ten_point_count = total_questions ∧
    5 * five_point_count + 10 * ten_point_count = total_points) :
  ∃ (five_point_count : ℕ), five_point_count = 20 ∧
    ∃ (ten_point_count : ℕ), 
      five_point_count + ten_point_count = total_questions ∧
      5 * five_point_count + 10 * ten_point_count = total_points :=
by
  sorry

end NUMINAMATH_CALUDE_test_question_count_l2732_273202


namespace NUMINAMATH_CALUDE_fine_on_fifth_day_l2732_273236

/-- Calculates the fine for a given day based on the previous day's fine -/
def nextDayFine (prevFine : ℚ) : ℚ :=
  min (prevFine + 0.3) (prevFine * 2)

/-- Calculates the total fine for a given number of days -/
def totalFine : ℕ → ℚ
  | 0 => 0
  | 1 => 0.05
  | n + 1 => nextDayFine (totalFine n)

theorem fine_on_fifth_day :
  totalFine 5 = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_fine_on_fifth_day_l2732_273236


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l2732_273297

/-- The length of the real axis of a hyperbola with equation x^2 - y^2/9 = 1 is 2 -/
theorem hyperbola_real_axis_length :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 - y^2/9 = 1}
  ∃ (a : ℝ), a > 0 ∧ (∀ (p : ℝ × ℝ), p ∈ hyperbola → p.1 ≤ a) ∧ 2 * a = 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l2732_273297


namespace NUMINAMATH_CALUDE_remainder_of_B_l2732_273268

theorem remainder_of_B (A B : ℕ) (h : B = 9 * A + 13) : B % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_B_l2732_273268


namespace NUMINAMATH_CALUDE_worker_overtime_rate_l2732_273277

/-- A worker's wage calculation problem -/
theorem worker_overtime_rate
  (ordinary_rate : ℚ)
  (total_hours : ℕ)
  (overtime_hours : ℕ)
  (total_earnings : ℚ)
  (h1 : ordinary_rate = 60 / 100)  -- 60 cents per hour for ordinary time
  (h2 : total_hours = 50)  -- 50-hour week
  (h3 : overtime_hours = 8)  -- 8 hours of overtime
  (h4 : total_earnings = 3240 / 100)  -- $32.40 total earnings
  : ∃ (overtime_rate : ℚ),
    overtime_rate = 90 / 100 ∧  -- $0.90 per hour for overtime
    total_earnings = ordinary_rate * (total_hours - overtime_hours) + overtime_rate * overtime_hours :=
by sorry

end NUMINAMATH_CALUDE_worker_overtime_rate_l2732_273277


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l2732_273216

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 6*y - x*y = 0) :
  ∀ a b : ℝ, a > 0 → b > 0 → 2*a + 6*b - a*b = 0 → x + y ≤ a + b ∧ x + y = 8 + 4*Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l2732_273216


namespace NUMINAMATH_CALUDE_table_tennis_ball_surface_area_l2732_273261

/-- The surface area of a sphere with diameter 40 millimeters is approximately 5026.55 square millimeters. -/
theorem table_tennis_ball_surface_area :
  let diameter : ℝ := 40
  let radius : ℝ := diameter / 2
  let surface_area : ℝ := 4 * Real.pi * radius^2
  ∃ ε > 0, abs (surface_area - 5026.55) < ε :=
by sorry

end NUMINAMATH_CALUDE_table_tennis_ball_surface_area_l2732_273261


namespace NUMINAMATH_CALUDE_highest_score_in_test_l2732_273298

/-- Given a math test with scores, prove the highest score -/
theorem highest_score_in_test (mark_score least_score highest_score : ℕ) : 
  mark_score = 2 * least_score →
  mark_score = 46 →
  highest_score - least_score = 75 →
  highest_score = 98 :=
by sorry

end NUMINAMATH_CALUDE_highest_score_in_test_l2732_273298


namespace NUMINAMATH_CALUDE_parabola_vertex_l2732_273203

/-- The parabola is defined by the equation y = (x - 1)² - 3 -/
def parabola (x : ℝ) : ℝ := (x - 1)^2 - 3

/-- The x-coordinate of the vertex -/
def vertex_x : ℝ := 1

/-- The y-coordinate of the vertex -/
def vertex_y : ℝ := -3

/-- Theorem: The vertex of the parabola y = (x - 1)² - 3 has coordinates (1, -3) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola vertex_x) ∧ 
  parabola vertex_x = vertex_y := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2732_273203


namespace NUMINAMATH_CALUDE_cone_volume_l2732_273282

/-- Given a cone with base radius 1 and lateral area 2π, its volume is (√3/3)π -/
theorem cone_volume (r h : ℝ) : 
  r = 1 → 
  π * r * (r^2 + h^2).sqrt = 2 * π → 
  (1/3) * π * r^2 * h = (Real.sqrt 3 / 3) * π :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l2732_273282


namespace NUMINAMATH_CALUDE_M_mod_55_l2732_273286

def M : ℕ := sorry

theorem M_mod_55 : M % 55 = 50 := by sorry

end NUMINAMATH_CALUDE_M_mod_55_l2732_273286


namespace NUMINAMATH_CALUDE_smallest_square_side_length_l2732_273273

theorem smallest_square_side_length :
  ∀ (n : ℕ),
  (∃ (a b c d : ℕ),
    n * n = a + 4 * b + 9 * c ∧
    14 = a + b + c ∧
    a ≥ 10 ∧ b ≥ 3 ∧ c ≥ 1) →
  n ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_side_length_l2732_273273


namespace NUMINAMATH_CALUDE_division_problem_l2732_273211

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 12401 → 
  divisor = 163 → 
  remainder = 13 → 
  dividend = divisor * quotient + remainder → 
  quotient = 76 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2732_273211


namespace NUMINAMATH_CALUDE_unique_sequence_exists_l2732_273244

def is_valid_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧
  a 2 > 1 ∧
  ∀ n : ℕ, n ≥ 1 → (a (n + 1))^3 + 1 = (a n) * (a (n + 2))

theorem unique_sequence_exists : ∃! a : ℕ → ℤ, is_valid_sequence a := by
  sorry

end NUMINAMATH_CALUDE_unique_sequence_exists_l2732_273244


namespace NUMINAMATH_CALUDE_principal_calculation_l2732_273270

/-- Simple interest calculation --/
def simple_interest (principal rate time : ℝ) : ℝ := principal * (1 + rate * time)

/-- Theorem: Principal calculation given two-year and three-year amounts --/
theorem principal_calculation (amount_2_years amount_3_years : ℝ) 
  (h1 : amount_2_years = 3450)
  (h2 : amount_3_years = 3655)
  (h3 : ∃ (p r : ℝ), simple_interest p r 2 = amount_2_years ∧ simple_interest p r 3 = amount_3_years) :
  ∃ (p r : ℝ), p = 3245 ∧ simple_interest p r 2 = amount_2_years ∧ simple_interest p r 3 = amount_3_years := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l2732_273270


namespace NUMINAMATH_CALUDE_division_problem_l2732_273213

theorem division_problem :
  ∀ (dividend divisor quotient remainder : ℕ),
    dividend = 171 →
    divisor = 21 →
    remainder = 3 →
    dividend = divisor * quotient + remainder →
    quotient = 8 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2732_273213


namespace NUMINAMATH_CALUDE_movie_ticket_price_l2732_273294

/-- The cost of a movie date, given ticket price, combo meal price, and candy price -/
def movie_date_cost (ticket_price : ℚ) (combo_price : ℚ) (candy_price : ℚ) : ℚ :=
  2 * ticket_price + combo_price + 2 * candy_price

/-- Theorem stating that the movie ticket price is $10.00 given the conditions of Connor's date -/
theorem movie_ticket_price :
  ∃ (ticket_price : ℚ),
    movie_date_cost ticket_price 11 2.5 = 36 ∧
    ticket_price = 10 := by
  sorry

end NUMINAMATH_CALUDE_movie_ticket_price_l2732_273294


namespace NUMINAMATH_CALUDE_expression_simplification_l2732_273253

theorem expression_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 - 11*x + 13*x^2 - 15 + 17*x + 19*x^2 = 25*x^2 + x - 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2732_273253


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2732_273223

def U : Set Nat := {0, 1, 2, 3, 4, 5}
def M : Set Nat := {0, 3, 5}
def N : Set Nat := {1, 4, 5}

theorem intersection_complement_equality : M ∩ (U \ N) = {0, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2732_273223


namespace NUMINAMATH_CALUDE_workers_calculation_l2732_273271

/-- The initial number of workers on a job -/
def initial_workers : ℕ := 20

/-- The number of days to complete the job with the initial number of workers -/
def initial_days : ℕ := 30

/-- The number of days worked before some workers leave -/
def days_before_leaving : ℕ := 15

/-- The number of workers that leave the job -/
def workers_leaving : ℕ := 5

/-- The total number of days to complete the job after some workers leave -/
def total_days : ℕ := 35

theorem workers_calculation :
  (initial_workers * days_before_leaving = (initial_workers - workers_leaving) * (total_days - days_before_leaving)) ∧
  (initial_workers * initial_days = initial_workers * days_before_leaving + (initial_workers - workers_leaving) * (total_days - days_before_leaving)) :=
sorry

end NUMINAMATH_CALUDE_workers_calculation_l2732_273271


namespace NUMINAMATH_CALUDE_bench_wood_length_l2732_273254

theorem bench_wood_length (num_long_pieces : ℕ) (long_piece_length : ℝ) (total_wood : ℝ) :
  num_long_pieces = 6 →
  long_piece_length = 4 →
  total_wood = 28 →
  total_wood - (num_long_pieces : ℝ) * long_piece_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_bench_wood_length_l2732_273254


namespace NUMINAMATH_CALUDE_root_sum_zero_l2732_273285

theorem root_sum_zero (a b : ℝ) : 
  (Complex.I + 1) ^ 2 + a * (Complex.I + 1) + b = 0 → a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_zero_l2732_273285


namespace NUMINAMATH_CALUDE_smallest_base_10_integer_l2732_273259

def is_valid_base_6_digit (x : ℕ) : Prop := x ≥ 0 ∧ x ≤ 5

def is_valid_base_8_digit (y : ℕ) : Prop := y ≥ 0 ∧ y ≤ 7

def base_6_to_decimal (x : ℕ) : ℕ := 6 * x + x

def base_8_to_decimal (y : ℕ) : ℕ := 8 * y + y

theorem smallest_base_10_integer : 
  ∃ (x y : ℕ), 
    is_valid_base_6_digit x ∧ 
    is_valid_base_8_digit y ∧ 
    base_6_to_decimal x = 63 ∧ 
    base_8_to_decimal y = 63 ∧ 
    (∀ (x' y' : ℕ), 
      is_valid_base_6_digit x' ∧ 
      is_valid_base_8_digit y' ∧ 
      base_6_to_decimal x' = base_8_to_decimal y' → 
      base_6_to_decimal x' ≥ 63) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_10_integer_l2732_273259


namespace NUMINAMATH_CALUDE_divisibility_properties_l2732_273238

theorem divisibility_properties (n : ℤ) : 
  (3 ∣ (n^3 - n)) ∧ 
  (5 ∣ (n^5 - n)) ∧ 
  (7 ∣ (n^7 - n)) ∧ 
  (11 ∣ (n^11 - n)) ∧ 
  (13 ∣ (n^13 - n)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_properties_l2732_273238


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2732_273289

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2732_273289


namespace NUMINAMATH_CALUDE_ribbon_boxes_theorem_l2732_273263

theorem ribbon_boxes_theorem (total_ribbon : ℝ) (ribbon_per_box : ℝ) (leftover : ℝ) :
  total_ribbon = 12.5 ∧ 
  ribbon_per_box = 1.75 ∧ 
  leftover = 0.3 → 
  ⌊total_ribbon / (ribbon_per_box + leftover)⌋ = 6 :=
by sorry

end NUMINAMATH_CALUDE_ribbon_boxes_theorem_l2732_273263


namespace NUMINAMATH_CALUDE_train_length_l2732_273245

/-- The length of a train given its speed and the time it takes to cross a bridge of known length. -/
theorem train_length (v : ℝ) (t : ℝ) (bridge_length : ℝ) (h1 : v = 36 * (1000 / 3600)) (h2 : t = 26.997840172786177) (h3 : bridge_length = 150) :
  v * t - bridge_length = 119.97840172786177 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l2732_273245


namespace NUMINAMATH_CALUDE_center_square_side_length_l2732_273284

theorem center_square_side_length :
  ∀ (total_side : ℝ) (l_region_count : ℕ) (l_region_fraction : ℝ),
    total_side = 20 →
    l_region_count = 4 →
    l_region_fraction = 1/5 →
    let total_area := total_side^2
    let l_regions_area := l_region_count * l_region_fraction * total_area
    let center_area := total_area - l_regions_area
    center_area.sqrt = 4 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_center_square_side_length_l2732_273284


namespace NUMINAMATH_CALUDE_at_least_one_not_perfect_square_l2732_273225

theorem at_least_one_not_perfect_square (d : ℕ+) :
  ¬(∃ x y z : ℕ, (2 * d - 1 = x^2) ∧ (5 * d - 1 = y^2) ∧ (13 * d - 1 = z^2)) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_not_perfect_square_l2732_273225


namespace NUMINAMATH_CALUDE_imaginary_part_of_product_l2732_273262

theorem imaginary_part_of_product : Complex.im ((1 - 3*Complex.I) * (1 - Complex.I)) = -4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_product_l2732_273262


namespace NUMINAMATH_CALUDE_smallest_rearranged_multiple_of_nine_l2732_273234

/-- A function that returns the digits of a natural number as a list -/
def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

/-- A predicate that checks if two natural numbers have the same digits -/
def same_digits (a b : ℕ) : Prop :=
  digits a = digits b

/-- The theorem stating that 1089 is the smallest natural number
    that when multiplied by 9, results in a number with the same digits -/
theorem smallest_rearranged_multiple_of_nine :
  (∀ n : ℕ, n < 1089 → ¬(same_digits n (9 * n))) ∧
  (same_digits 1089 (9 * 1089)) :=
sorry

end NUMINAMATH_CALUDE_smallest_rearranged_multiple_of_nine_l2732_273234


namespace NUMINAMATH_CALUDE_rectangular_triangular_field_equal_area_l2732_273221

/-- Proves that a rectangular field with width 4 m and length 6.3 m has the same area
    as a triangular field with base 7.2 m and height 7 m. -/
theorem rectangular_triangular_field_equal_area :
  let triangle_base : ℝ := 7.2
  let triangle_height : ℝ := 7
  let triangle_area : ℝ := (triangle_base * triangle_height) / 2
  let rectangle_width : ℝ := 4
  let rectangle_length : ℝ := 6.3
  let rectangle_area : ℝ := rectangle_width * rectangle_length
  triangle_area = rectangle_area := by sorry

end NUMINAMATH_CALUDE_rectangular_triangular_field_equal_area_l2732_273221


namespace NUMINAMATH_CALUDE_rohan_house_rent_percentage_l2732_273276

/-- Rohan's monthly expenses and savings --/
def rohan_finances (house_rent_percentage : ℚ) : Prop :=
  let salary : ℚ := 5000
  let food_percentage : ℚ := 40 / 100
  let entertainment_percentage : ℚ := 10 / 100
  let conveyance_percentage : ℚ := 10 / 100
  let savings : ℚ := 1000
  (food_percentage + house_rent_percentage + entertainment_percentage + conveyance_percentage + savings / salary = 1)

/-- Theorem stating that Rohan spends 20% of his salary on house rent --/
theorem rohan_house_rent_percentage : 
  ∃ (house_rent_percentage : ℚ), rohan_finances house_rent_percentage ∧ house_rent_percentage = 20 / 100 := by
  sorry

end NUMINAMATH_CALUDE_rohan_house_rent_percentage_l2732_273276


namespace NUMINAMATH_CALUDE_twin_running_problem_l2732_273278

theorem twin_running_problem (x : ℝ) :
  (x ≥ 0) →  -- Ensure distance is non-negative
  (2 * x = 25) →  -- Final distance equation
  (x = 12.5) :=
by
  sorry

end NUMINAMATH_CALUDE_twin_running_problem_l2732_273278


namespace NUMINAMATH_CALUDE_max_principals_l2732_273246

/-- Represents the number of years in a period -/
def period : ℕ := 10

/-- Represents the minimum term length for a principal -/
def minTerm : ℕ := 3

/-- Represents the maximum term length for a principal -/
def maxTerm : ℕ := 5

/-- Represents a valid principal term length -/
def ValidTerm (t : ℕ) : Prop := minTerm ≤ t ∧ t ≤ maxTerm

/-- 
Theorem: The maximum number of principals during a continuous 10-year period is 3,
given that each principal's term can be between 3 and 5 years.
-/
theorem max_principals :
  ∃ (n : ℕ) (terms : List ℕ),
    (∀ t ∈ terms, ValidTerm t) ∧ 
    (terms.sum ≥ period) ∧
    (terms.length = n) ∧
    (∀ m : ℕ, m > n → 
      ¬∃ (terms' : List ℕ), 
        (∀ t ∈ terms', ValidTerm t) ∧ 
        (terms'.sum ≥ period) ∧
        (terms'.length = m)) ∧
    n = 3 :=
  sorry

end NUMINAMATH_CALUDE_max_principals_l2732_273246


namespace NUMINAMATH_CALUDE_chef_potato_problem_chef_potato_solution_l2732_273210

theorem chef_potato_problem (already_cooked : ℕ) (cooking_time_per_potato : ℕ) (remaining_cooking_time : ℕ) : ℕ :=
  let remaining_potatoes := remaining_cooking_time / cooking_time_per_potato
  let total_potatoes := already_cooked + remaining_potatoes
  total_potatoes

#check chef_potato_problem 8 9 63

theorem chef_potato_solution :
  chef_potato_problem 8 9 63 = 15 := by
  sorry

end NUMINAMATH_CALUDE_chef_potato_problem_chef_potato_solution_l2732_273210


namespace NUMINAMATH_CALUDE_math_basketball_count_l2732_273201

/-- Represents the number of students in a school with various club and team memberships -/
structure SchoolMembership where
  total : ℕ
  science_club : ℕ
  math_club : ℕ
  football_team : ℕ
  basketball_team : ℕ
  science_football : ℕ

/-- Conditions for the school membership problem -/
def school_conditions (s : SchoolMembership) : Prop :=
  s.total = 60 ∧
  s.science_club + s.math_club = s.total ∧
  s.football_team + s.basketball_team = s.total ∧
  s.science_football = 20 ∧
  s.math_club = 36 ∧
  s.basketball_team = 22

/-- Theorem stating the number of students in both math club and basketball team -/
theorem math_basketball_count (s : SchoolMembership) 
  (h : school_conditions s) : 
  s.math_club + s.basketball_team - s.total = 18 := by
  sorry

#check math_basketball_count

end NUMINAMATH_CALUDE_math_basketball_count_l2732_273201


namespace NUMINAMATH_CALUDE_original_fraction_value_l2732_273255

theorem original_fraction_value (n : ℚ) : 
  (n + 1) / (n + 6) = 7 / 12 → n / (n + 5) = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_original_fraction_value_l2732_273255


namespace NUMINAMATH_CALUDE_percent_decrease_cars_sold_car_sales_decrease_proof_l2732_273224

/-- Calculates the percent decrease in cars sold given the increase in total profit and average profit per car -/
theorem percent_decrease_cars_sold 
  (total_profit_increase : ℝ) 
  (avg_profit_per_car_increase : ℝ) : ℝ :=
  let new_total_profit_ratio := 1 + total_profit_increase
  let new_avg_profit_ratio := 1 + avg_profit_per_car_increase
  let cars_sold_ratio := new_total_profit_ratio / new_avg_profit_ratio
  (1 - cars_sold_ratio) * 100

/-- The percent decrease in cars sold is approximately 30% when total profit increases by 30% and average profit per car increases by 85.71% -/
theorem car_sales_decrease_proof : 
  abs (percent_decrease_cars_sold 0.30 0.8571 - 30) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_percent_decrease_cars_sold_car_sales_decrease_proof_l2732_273224


namespace NUMINAMATH_CALUDE_base_9_101_to_decimal_l2732_273295

def base_9_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ (digits.length - 1 - i))) 0

theorem base_9_101_to_decimal :
  base_9_to_decimal [1, 0, 1] = 82 := by
  sorry

end NUMINAMATH_CALUDE_base_9_101_to_decimal_l2732_273295


namespace NUMINAMATH_CALUDE_scientific_notation_of_1010659_l2732_273227

/-- The original number to be expressed in scientific notation -/
def original_number : ℕ := 1010659

/-- The number of significant figures to keep -/
def significant_figures : ℕ := 3

/-- Function to convert a natural number to scientific notation with given significant figures -/
noncomputable def to_scientific_notation (n : ℕ) (sig_figs : ℕ) : ℝ × ℤ := sorry

/-- Theorem stating that the scientific notation of 1,010,659 with three significant figures is 1.01 × 10^6 -/
theorem scientific_notation_of_1010659 :
  to_scientific_notation original_number significant_figures = (1.01, 6) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1010659_l2732_273227


namespace NUMINAMATH_CALUDE_angle_AOC_equals_negative_150_l2732_273287

-- Define the rotation angles
def counterclockwise_rotation : ℝ := 120
def clockwise_rotation : ℝ := 270

-- Define the resulting angle
def angle_AOC : ℝ := counterclockwise_rotation - clockwise_rotation

-- Theorem statement
theorem angle_AOC_equals_negative_150 : angle_AOC = -150 := by
  sorry

end NUMINAMATH_CALUDE_angle_AOC_equals_negative_150_l2732_273287


namespace NUMINAMATH_CALUDE_ellipse_chord_slope_l2732_273205

/-- The slope of a chord of an ellipse bisected by a given point -/
theorem ellipse_chord_slope (x₁ y₁ x₂ y₂ : ℝ) : 
  (x₁^2 / 36 + y₁^2 / 9 = 1) →  -- P(x₁, y₁) is on the ellipse
  (x₂^2 / 36 + y₂^2 / 9 = 1) →  -- Q(x₂, y₂) is on the ellipse
  ((x₁ + x₂) / 2 = 1) →         -- Midpoint x-coordinate is 1
  ((y₁ + y₂) / 2 = 1) →         -- Midpoint y-coordinate is 1
  (y₂ - y₁) / (x₂ - x₁) = -1/4  -- Slope of PQ is -1/4
:= by sorry

end NUMINAMATH_CALUDE_ellipse_chord_slope_l2732_273205


namespace NUMINAMATH_CALUDE_nancy_carrots_l2732_273260

/-- The total number of carrots Nancy has after two days of picking and throwing out some -/
def total_carrots (initial : ℕ) (thrown_out : ℕ) (picked_next_day : ℕ) : ℕ :=
  initial - thrown_out + picked_next_day

/-- Theorem stating that Nancy's total carrots is 31 given the specific numbers in the problem -/
theorem nancy_carrots : total_carrots 12 2 21 = 31 := by
  sorry

end NUMINAMATH_CALUDE_nancy_carrots_l2732_273260


namespace NUMINAMATH_CALUDE_goat_kangaroo_ratio_l2732_273212

theorem goat_kangaroo_ratio : 
  ∀ (num_goats : ℕ), 
    (2 * 23 + 4 * num_goats = 322) → 
    (num_goats : ℚ) / 23 = 3 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_goat_kangaroo_ratio_l2732_273212


namespace NUMINAMATH_CALUDE_rebecca_current_income_l2732_273237

/-- Rebecca's current yearly income --/
def rebecca_income : ℝ := sorry

/-- Jimmy's annual income --/
def jimmy_income : ℝ := 18000

/-- The increase in Rebecca's income --/
def income_increase : ℝ := 7000

/-- The percentage of Rebecca's new income in their combined income --/
def rebecca_percentage : ℝ := 0.55

theorem rebecca_current_income :
  rebecca_income = 15000 ∧
  (rebecca_income + income_increase) = 
    rebecca_percentage * (rebecca_income + income_increase + jimmy_income) :=
by sorry

end NUMINAMATH_CALUDE_rebecca_current_income_l2732_273237


namespace NUMINAMATH_CALUDE_sector_perimeter_ratio_l2732_273252

theorem sector_perimeter_ratio (α : ℝ) (r R : ℝ) (h_positive : 0 < α ∧ 0 < r ∧ 0 < R) 
  (h_area_ratio : (α * r^2) / (α * R^2) = 1/4) : 
  (2*r + α*r) / (2*R + α*R) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sector_perimeter_ratio_l2732_273252


namespace NUMINAMATH_CALUDE_mark_sold_eight_less_l2732_273267

theorem mark_sold_eight_less (total : ℕ) (mark_sold : ℕ) (ann_sold : ℕ) 
  (h_total : total = 9)
  (h_mark : mark_sold < total)
  (h_ann : ann_sold = total - 2)
  (h_mark_positive : mark_sold ≥ 1)
  (h_ann_positive : ann_sold ≥ 1)
  (h_total_greater : mark_sold + ann_sold < total) :
  total - mark_sold = 8 := by
sorry

end NUMINAMATH_CALUDE_mark_sold_eight_less_l2732_273267


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2732_273241

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

/-- Theorem: Given S_3 = 2 and S_6 = 8, then S_9 = 18 -/
theorem arithmetic_sequence_sum 
  (seq : ArithmeticSequence) 
  (h1 : seq.S 3 = 2) 
  (h2 : seq.S 6 = 8) : 
  seq.S 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2732_273241


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l2732_273239

/-- A right triangle with sides 5, 12, and 13 -/
structure RightTriangle where
  de : ℝ
  ef : ℝ
  df : ℝ
  de_eq : de = 5
  ef_eq : ef = 12
  df_eq : df = 13
  right_angle : de^2 + ef^2 = df^2

/-- A square inscribed in the right triangle -/
structure InscribedSquare (t : RightTriangle) where
  side_length : ℝ
  on_df : side_length ≤ t.df
  on_de : side_length ≤ t.de
  on_ef : side_length ≤ t.ef

/-- The theorem stating the side length of the inscribed square -/
theorem inscribed_square_side_length (t : RightTriangle) (s : InscribedSquare t) :
  s.side_length = 10140 / 229 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l2732_273239


namespace NUMINAMATH_CALUDE_expression_value_l2732_273222

theorem expression_value :
  let x : ℝ := 1
  let y : ℝ := 1
  let z : ℝ := 3
  let p : ℝ := 2
  let q : ℝ := 4
  let r : ℝ := 2
  let s : ℝ := 3
  let t : ℝ := 3
  (p + x)^2 * y * z - q * r * (x * y * z)^2 + s^t = -18 := by sorry

end NUMINAMATH_CALUDE_expression_value_l2732_273222


namespace NUMINAMATH_CALUDE_sin_600_degrees_l2732_273219

theorem sin_600_degrees : Real.sin (600 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_600_degrees_l2732_273219


namespace NUMINAMATH_CALUDE_system_solution_l2732_273264

theorem system_solution :
  ∃ (A B C D : ℚ),
    A = 1/42 ∧
    B = 1/7 ∧
    C = 1/3 ∧
    D = 1/2 ∧
    A = B * C * D ∧
    A + B = C * D ∧
    A + B + C = D ∧
    A + B + C + D = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2732_273264


namespace NUMINAMATH_CALUDE_consecutive_product_prime_power_and_perfect_power_l2732_273291

theorem consecutive_product_prime_power_and_perfect_power (m : ℕ) : m ≥ 1 → (
  (∃ (p : ℕ) (k : ℕ), Prime p ∧ m * (m + 1) = p^k) ↔ m = 1
) ∧ 
¬(∃ (a k : ℕ), a ≥ 1 ∧ k ≥ 2 ∧ m * (m + 1) = a^k) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_prime_power_and_perfect_power_l2732_273291


namespace NUMINAMATH_CALUDE_min_socks_for_twelve_pairs_l2732_273229

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer :=
  (red : ℕ)
  (green : ℕ)
  (blue : ℕ)
  (purple : ℕ)

/-- Calculates the minimum number of socks needed to guarantee a certain number of pairs -/
def minSocksForPairs (drawer : SockDrawer) (pairs : ℕ) : ℕ :=
  sorry

/-- Theorem stating that 27 socks are needed to guarantee 12 pairs in the given drawer -/
theorem min_socks_for_twelve_pairs :
  let drawer : SockDrawer := { red := 90, green := 70, blue := 50, purple := 30 }
  minSocksForPairs drawer 12 = 27 := by sorry

end NUMINAMATH_CALUDE_min_socks_for_twelve_pairs_l2732_273229


namespace NUMINAMATH_CALUDE_cos_inequality_range_l2732_273257

theorem cos_inequality_range (θ : Real) : 
  θ ∈ Set.Icc (-Real.pi) Real.pi →
  (3 * Real.sqrt 2 * Real.cos (θ + Real.pi / 4) < 4 * (Real.sin θ ^ 3 - Real.cos θ ^ 3)) ↔
  θ ∈ Set.Ioc (-Real.pi) (-3 * Real.pi / 4) ∪ Set.Ioo (Real.pi / 4) Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cos_inequality_range_l2732_273257


namespace NUMINAMATH_CALUDE_largest_number_with_digit_sum_23_l2732_273274

def digit_sum (n : Nat) : Nat :=
  let digits := n.digits 10
  digits.sum

def all_digits_different (n : Nat) : Prop :=
  let digits := n.digits 10
  digits.length = digits.toFinset.card

theorem largest_number_with_digit_sum_23 :
  ∀ n : Nat, n ≤ 999 →
    (digit_sum n = 23 ∧ all_digits_different n) →
    n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_number_with_digit_sum_23_l2732_273274


namespace NUMINAMATH_CALUDE_sequence_formula_l2732_273214

theorem sequence_formula (n : ℕ) : 
  let a : ℕ → ℕ := λ k => 2^k + 1
  (a 1 = 3) ∧ (a 2 = 5) ∧ (a 3 = 9) ∧ (a 4 = 17) ∧ (a 5 = 33) :=
by sorry

end NUMINAMATH_CALUDE_sequence_formula_l2732_273214
