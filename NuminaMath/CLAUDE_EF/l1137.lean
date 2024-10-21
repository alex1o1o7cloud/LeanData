import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_algebra_sum_is_five_l1137_113767

def letter_value (position : ℕ) : ℤ :=
  match position % 8 with
  | 1 => 3
  | 2 => 1
  | 3 => 0
  | 4 => -1
  | 5 => -3
  | 6 => -1
  | 7 => 0
  | 0 => 1
  | _ => 0  -- This case should never occur, but Lean requires it for completeness

def alphabet_position (letter : Char) : ℕ :=
  match letter with
  | 'a' => 1
  | 'l' => 12
  | 'g' => 7
  | 'e' => 5
  | 'b' => 2
  | 'r' => 18
  | _ => 0  -- Default case for other letters

def sum_word_values (word : String) : ℤ :=
  word.data.map (λ c => letter_value (alphabet_position c)) |>.sum

theorem algebra_sum_is_five : sum_word_values "algebra" = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_algebra_sum_is_five_l1137_113767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_problem_inequality_system_solutions_l1137_113710

-- Problem 1
theorem calculation_problem :
  Real.sqrt 12 - 4 * Real.cos (30 * π / 180) + (3.14 - Real.pi) ^ 0 + |1 - Real.sqrt 2| = Real.sqrt 2 := by
  sorry

-- Problem 2
def inequality_system (x : ℤ) : Prop :=
  2 * x - (x + 3) / 2 ≤ 0 ∧ 5 * x + 1 > 3 * (x - 1)

theorem inequality_system_solutions :
  ∀ x : ℤ, inequality_system x ↔ x = -1 ∨ x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_problem_inequality_system_solutions_l1137_113710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_parts_sum_power_l1137_113761

/-- The decimal part of a real number x -/
noncomputable def decimalPart (x : ℝ) : ℝ := x - ⌊x⌋

theorem decimal_parts_sum_power (a b : ℝ) : 
  decimalPart (5 + Real.sqrt 7) = a →
  decimalPart (5 - Real.sqrt 7) = b →
  (a + b)^2023 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_parts_sum_power_l1137_113761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inheritance_tax_problem_l1137_113733

theorem inheritance_tax_problem (total_tax : ℝ) (federal_rate : ℝ) (state_rate : ℝ) 
  (h1 : total_tax = 16800)
  (h2 : federal_rate = 0.25)
  (h3 : state_rate = 0.12) :
  ∃ (original : ℝ), 
    (federal_rate * original + state_rate * (original - federal_rate * original) = total_tax) ∧ 
    (abs (original - 49412) < 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inheritance_tax_problem_l1137_113733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_increasing_condition_l1137_113774

-- Define the function f(x) = (a - 1)^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) ^ x

-- State the theorem
theorem exponential_increasing_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → a > 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_increasing_condition_l1137_113774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_e_l1137_113730

def polynomial (a b c d e : ℤ) (x : ℚ) : ℚ :=
  a * x^4 + b * x^3 + c * x^2 + d * x + e

theorem smallest_positive_e : 
  ∀ a b c d e : ℤ, 
    (∀ x : ℚ, polynomial a b c d e x = 0 ↔ x = -4 ∨ x = 3 ∨ x = 7 ∨ x = (1:ℚ)/2) →
    e > 0 →
    ∀ e' : ℤ, e' > 0 → 
      (∃ a' b' c' d' : ℤ, ∀ x : ℚ, polynomial a' b' c' d' e' x = 0 ↔ x = -4 ∨ x = 3 ∨ x = 7 ∨ x = (1:ℚ)/2) →
      e' ≥ 42 :=
by sorry

#check smallest_positive_e

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_e_l1137_113730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_sum_l1137_113773

/-- Given a triangle with angles A, B, C, prove that under certain conditions,
    there exist positive integers x, y, z, w satisfying specific properties. -/
theorem triangle_special_sum : ∀ (B C : Real),
  -- A = 60°
  let A : Real := Real.pi / 3
  -- First given equation
  (Real.cos B)^2 + (Real.cos C)^2 + 2 * Real.sin B * Real.sin C * Real.cos A = 13/7 →
  -- Second given equation
  (Real.cos C)^2 + (Real.cos A)^2 + 2 * Real.sin C * Real.sin A * Real.cos B = 12/5 →
  -- Third equation with integer conditions
  ∃ (x y z w : ℕ),
    x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧
    (Real.cos A)^2 + (Real.cos B)^2 + 2 * Real.sin A * Real.sin B * Real.cos C = (x - y * Real.sqrt z) / w ∧
    Int.gcd (x + y) w = 1 ∧
    ∀ (p : ℕ), Nat.Prime p → ¬(p^2 ∣ z) ∧
    x + y + z + w = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_sum_l1137_113773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_quiz_goal_l1137_113784

theorem lisa_quiz_goal (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (current_As : ℕ) (quizzes_taken : ℕ) : 
  total_quizzes = 50 →
  goal_percentage = 4/5 →
  current_As = 22 →
  quizzes_taken = 30 →
  ∃ (max_non_As : ℕ), 
    max_non_As = 2 ∧ 
    (total_quizzes - quizzes_taken - max_non_As) + current_As ≥ 
      (goal_percentage * total_quizzes).floor := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_quiz_goal_l1137_113784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_maximized_l1137_113794

/-- The function S in terms of x -/
noncomputable def S (x : ℝ) : ℝ := x * (180 - 3 / x + 400 / x^2)

/-- The theorem stating that S is maximized when x = 40 and y = 45 -/
theorem S_maximized (x y : ℝ) (h : x * y = 1800) :
  S x ≤ S 40 ∧ y = 45 := by
  sorry

#check S_maximized

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_maximized_l1137_113794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1137_113721

noncomputable def f (A ω α : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + α)

theorem function_properties (A ω α : ℝ) 
  (h1 : A > 0) (h2 : ω > 0) (h3 : -π/2 < α ∧ α < π/2)
  (h4 : ∀ x, f A ω α (x + π) = f A ω α x)
  (h5 : ∀ x, f A ω α x ≤ 3)
  (h6 : f A ω α (π/6) = 3) :
  ∃ m : ℝ, m > 0 ∧ 
    (∀ x, f A ω α x = 3 * Real.sin (2*x + π/6)) ∧
    (∀ g : ℝ → ℝ, (∀ x, g x = f A ω α (x - m)) → 
      (∀ x, g (-x) = g x) → 
      ∀ m' > 0, m' ≥ m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1137_113721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equations_not_generally_true_l1137_113728

-- Define the variables and conditions
variable (t : ℝ)

-- Define x and y as functions of t
noncomputable def x (t : ℝ) : ℝ := t^(2/(t-1))
noncomputable def y (t : ℝ) : ℝ := t^((2*t-1)/(t-1))

-- State the theorem
theorem equations_not_generally_true : 
  ¬(∀ t : ℝ, t > 0 → t ≠ 1 → 
    (y t)^(x t) = (x t)^(y t) ∨ 
    (y t)^(1/(x t)) = (x t)^(y t) ∨ 
    (y t)/(x t) = (x t)/(y t) ∨ 
    (x t)^(-(x t)) = (y t)^(-(y t))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equations_not_generally_true_l1137_113728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_profit_loss_selling_price_l1137_113751

/-- Proves that the selling price for equal percentage profit and loss is correct --/
theorem equal_profit_loss_selling_price 
  (cost_price : ℝ) 
  (loss_price : ℝ) 
  (profit_price : ℝ) 
  (h1 : cost_price = 1500) 
  (h2 : loss_price = 1280) 
  (h3 : profit_price = 1875) 
  (h4 : profit_price = cost_price * 1.25) : 
  ∃ (equal_price : ℝ), 
    (equal_price - cost_price) / cost_price = (cost_price - loss_price) / cost_price ∧ 
    abs (equal_price - 1720.05) < 0.01 := by
  sorry

#check equal_profit_loss_selling_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_profit_loss_selling_price_l1137_113751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sail_max_power_speed_l1137_113795

/-- The force equation for the airflow acting on the sail -/
noncomputable def force (B ρ v₀ v : ℝ) : ℝ := (B * 7 * ρ * (v₀ - v)^2) / 2

/-- The instantaneous power of the wind -/
noncomputable def power (B ρ v₀ v : ℝ) : ℝ := force B ρ v₀ v * v

/-- The wind speed -/
def wind_speed : ℝ := 6.3

theorem sail_max_power_speed :
  ∃ (v : ℝ), v = wind_speed / 3 ∧
  ∀ (B ρ : ℝ), B > 0 → ρ > 0 →
    ∀ (u : ℝ), power B ρ wind_speed v ≥ power B ρ wind_speed u := by
  sorry

#check sail_max_power_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sail_max_power_speed_l1137_113795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_tan_alpha_l1137_113766

theorem sin_tan_alpha (α : ℝ) :
  let P : ℝ × ℝ := (3/5, -4/5)
  (∃ t : ℝ, t > 0 ∧ t • (Real.cos α, Real.sin α) = P) →
  Real.sin α * Real.tan α = 16/15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_tan_alpha_l1137_113766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_north_southville_population_increase_correct_l1137_113740

/-- Calculates the annual population increase in North Southville and rounds it to the nearest hundred. -/
def north_southville_population_increase 
  (birth_interval : ℚ) 
  (death_interval : ℚ) 
  (hours_per_day : ℚ) 
  (days_per_year : ℚ) : ℕ :=
  let births_per_day := hours_per_day / birth_interval
  let deaths_per_day := hours_per_day / death_interval
  let net_increase_per_day := births_per_day - deaths_per_day
  let annual_increase := net_increase_per_day * days_per_year
  (round (annual_increase / 100) * 100).toNat

/-- Proves that the calculation for North Southville's population increase is correct. -/
theorem north_southville_population_increase_correct : 
  north_southville_population_increase 6 36 24 365 = 1200 := by
  -- Unfold the definition and simplify
  unfold north_southville_population_increase
  -- Perform the calculation
  simp [round]
  -- The rest of the proof would go here
  sorry

#eval north_southville_population_increase 6 36 24 365

end NUMINAMATH_CALUDE_ERRORFEEDBACK_north_southville_population_increase_correct_l1137_113740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_length_l1137_113704

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := (1 - Real.exp x - Real.exp (-x)) / 2

-- Define the domain
def a : ℝ := 0
def b : ℝ := 3

-- State the theorem
theorem curve_length : 
  (∫ x in a..b, Real.sqrt (1 + (deriv f x)^2)) = Real.sinh b := by
  sorry

-- Additional lemma to show that f is differentiable
lemma f_differentiable : Differentiable ℝ f := by
  sorry

-- Lemma to show that the derivative of f is as expected
lemma deriv_f (x : ℝ) : deriv f x = -Real.sinh x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_length_l1137_113704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_arithmetic_progression_implies_not_arithmetic_progression_l1137_113765

theorem sine_arithmetic_progression_implies_not_arithmetic_progression 
  (x₁ x₂ x₃ : ℝ) 
  (h_distinct : x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) 
  (h_interval : 0 < x₁ ∧ x₁ < π/2 ∧ 0 < x₂ ∧ x₂ < π/2 ∧ 0 < x₃ ∧ x₃ < π/2) 
  (h_sine_ap : ∃ d : ℝ, Real.sin x₂ - Real.sin x₁ = d ∧ Real.sin x₃ - Real.sin x₂ = d) : 
  ¬(∃ r : ℝ, x₂ - x₁ = r ∧ x₃ - x₂ = r) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_arithmetic_progression_implies_not_arithmetic_progression_l1137_113765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graphs_are_different_l1137_113775

-- Define the three functions
noncomputable def f₁ (x : ℝ) : ℝ := x + 3
noncomputable def f₂ (x : ℝ) : ℝ := (x^2 - 9) / (x - 3)
def f₃ (x y : ℝ) : Prop := (x - 3) * y = x^2 - 9

-- Theorem stating that the graphs are all different
theorem graphs_are_different : 
  (∃ x : ℝ, f₁ x ≠ f₂ x) ∧ 
  (∃ x y : ℝ, f₃ x y ∧ y ≠ f₁ x) ∧ 
  (∃ x y : ℝ, f₃ x y ∧ y ≠ f₂ x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graphs_are_different_l1137_113775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_ride_cost_correct_l1137_113753

/-- Calculates the total cost of a taxi ride based on given parameters --/
def taxi_ride_cost (base_fare : ℚ) (additional_fare : ℚ) (waiting_charge : ℚ) 
  (delay_charge : ℚ) (toll_fee : ℚ) (discount_rate : ℚ) (tip_rate : ℚ) 
  (distance : ℚ) (waiting_time : ℕ) (delay_time : ℕ) (has_toll : Bool) 
  (is_discounted : Bool) : ℚ :=
  sorry

/-- The total cost of the taxi ride is correct --/
theorem taxi_ride_cost_correct : 
  (let base_fare : ℚ := 5/2
  let additional_fare : ℚ := 2/5
  let waiting_charge : ℚ := 1/4
  let delay_charge : ℚ := 3/20
  let toll_fee : ℚ := 3
  let discount_rate : ℚ := 1/10
  let tip_rate : ℚ := 3/20
  let distance : ℚ := 8
  let waiting_time : ℕ := 12
  let delay_time : ℕ := 25
  let has_toll : Bool := true
  let is_discounted : Bool := true
  taxi_ride_cost base_fare additional_fare waiting_charge delay_charge 
    toll_fee discount_rate tip_rate distance waiting_time delay_time 
    has_toll is_discounted) = 519/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_ride_cost_correct_l1137_113753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salad_dressing_composition_l1137_113724

theorem salad_dressing_composition (p_vinegar_ratio : Real) (q_vinegar_ratio : Real) 
  (new_vinegar_ratio : Real) (p_ratio_in_new : Real) :
  p_vinegar_ratio = 0.3 →
  p_ratio_in_new = 0.1 →
  new_vinegar_ratio = 0.12 →
  p_ratio_in_new * p_vinegar_ratio + (1 - p_ratio_in_new) * q_vinegar_ratio = new_vinegar_ratio →
  q_vinegar_ratio = 0.1 := by
  intro h1 h2 h3 h4
  -- The proof steps would go here
  sorry

#check salad_dressing_composition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salad_dressing_composition_l1137_113724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_sides_l1137_113786

/-- The number of sides of regular polygon P -/
def n : ℕ := 40

/-- The number of sides of regular polygon Q -/
def m : ℕ := n - 4

/-- The measure of an interior angle of a regular polygon with k sides -/
noncomputable def interior_angle (k : ℕ) : ℚ := (k - 2) * 180 / k

theorem regular_polygon_sides : 
  interior_angle n - interior_angle m = 1 ↔ n = 40 := by
  sorry

#eval n
#eval m

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_sides_l1137_113786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_proof_l1137_113741

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates a quadratic function at a given x -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- Calculates the vertex of a quadratic function -/
noncomputable def QuadraticFunction.vertex (f : QuadraticFunction) : ℝ × ℝ :=
  let x := -f.b / (2 * f.a)
  (x, f.eval x)

theorem parabola_equation_proof (f : QuadraticFunction)
    (h1 : f.a = -2 ∧ f.b = 12 ∧ f.c = -13)
    (h2 : f.vertex = (3, 5))
    (h3 : f.eval 4 = 3) : True := by
  sorry

#check parabola_equation_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_proof_l1137_113741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distances_l1137_113745

def A : Fin 3 → ℝ := ![10, 0, 0]
def B : Fin 3 → ℝ := ![0, -6, 0]
def C : Fin 3 → ℝ := ![0, 0, 8]
def D : Fin 3 → ℝ := ![0, 0, 0]
def E : Fin 3 → ℝ := ![5, -3, 4]
def P : Fin 3 → ℝ := ![5, -3, 4]

noncomputable def distance (X Y : Fin 3 → ℝ) : ℝ :=
  Real.sqrt ((X 0 - Y 0)^2 + (X 1 - Y 1)^2 + (X 2 - Y 2)^2)

theorem equal_distances :
  distance A P = distance B P ∧
  distance A P = distance C P ∧
  distance A P = distance D P ∧
  distance A P = distance E P := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distances_l1137_113745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_four_percent_l1137_113706

/-- Calculates the annual interest rate given present value, future value, and time period. -/
noncomputable def calculate_interest_rate (present_value : ℝ) (future_value : ℝ) (time : ℝ) : ℝ :=
  ((future_value / present_value) ^ (1 / time)) - 1

/-- Proves that the interest rate is 0.04 given the problem conditions. -/
theorem interest_rate_is_four_percent :
  let present_value : ℝ := 625
  let future_value : ℝ := 676
  let time : ℝ := 2
  let calculated_rate := calculate_interest_rate present_value future_value time
  abs (calculated_rate - 0.04) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_four_percent_l1137_113706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_above_threshold_l1137_113744

def jungkook : ℚ := 4/5
def yoongi : ℚ := 1/2
def yoojung : ℚ := 9/10
def yuna : ℚ := 1/3

def threshold : ℚ := 3/10

def count_above_threshold (j y1 y2 y3 t : ℚ) : ℕ :=
  (if j > t then 1 else 0) +
  (if y1 > t then 1 else 0) +
  (if y2 > t then 1 else 0) +
  (if y3 > t then 1 else 0)

theorem all_above_threshold :
  count_above_threshold jungkook yoongi yoojung yuna threshold = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_above_threshold_l1137_113744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_count_l1137_113783

def is_valid_number (n : ℕ) : Bool :=
  n ≥ 1000 && n < 10000 && 
  (n % 10 ≠ 2) &&
  (List.range 4).all (fun d => (n.digits 10).count d ≤ 1)

def count_valid_numbers : ℕ := (List.range 10000).filter is_valid_number |>.length

theorem valid_numbers_count : count_valid_numbers = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_count_l1137_113783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_percentage_increase_l1137_113739

/-- The highest average cost of milk in City A -/
noncomputable def highest_cost_A : ℝ := 45

/-- The lowest average cost of milk in City B -/
noncomputable def lowest_cost_B : ℝ := 25

/-- The percentage increase from the lowest cost in City B to the highest cost in City A -/
noncomputable def percentage_increase : ℝ := (highest_cost_A - lowest_cost_B) / lowest_cost_B * 100

theorem cost_percentage_increase :
  percentage_increase = 80 := by
  -- Unfold the definitions
  unfold percentage_increase highest_cost_A lowest_cost_B
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_percentage_increase_l1137_113739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_properties_l1137_113770

-- Define the line l
def line_l (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 2 = 0

-- Define the slope of a line
def line_slope (m : ℝ) (x y : ℝ) : Prop := y = m * x + 2

-- Define a directional vector
def directional_vector (v : ℝ × ℝ) (x y : ℝ) : Prop :=
  ∃ (t : ℝ), x = v.1 * t ∧ y = v.2 * t

-- Define the third quadrant
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

theorem line_l_properties :
  (∀ x y, line_l x y → line_slope (-Real.sqrt 3) x y) ∧
  (∀ x y, line_l x y → directional_vector (-Real.sqrt 3, 3) x y) ∧
  (∀ x y, line_l x y → ¬ third_quadrant x y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_properties_l1137_113770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_willam_land_percentage_l1137_113716

/-- Represents the tax rates for different land categories -/
structure TaxRates where
  x : Float
  y : Float
  z : Float

/-- Represents the tax payments made by Mr. Willam -/
structure WillamPayments where
  x : Float
  y : Float
  z : Float

/-- Calculates the percentage of land owned by Mr. Willam -/
def calculate_land_percentage (
  tax_rates : TaxRates)
  (willam_payments : WillamPayments)
  (total_tax : Float)
  (total_land : Float)
  : Float :=
  let willam_total_tax := willam_payments.x + willam_payments.y + willam_payments.z
  (willam_total_tax / total_tax) * 100

/-- Theorem stating that Mr. Willam owns approximately 72.31% of the total taxable land -/
theorem willam_land_percentage :
  let tax_rates : TaxRates := { x := 0.45, y := 0.55, z := 0.65 }
  let willam_payments : WillamPayments := { x := 1200, y := 1600, z := 1900 }
  let total_tax : Float := 6500
  let total_land : Float := 1000
  (calculate_land_percentage tax_rates willam_payments total_tax total_land - 72.31).abs < 0.01 := by
  sorry

#eval calculate_land_percentage 
  { x := 0.45, y := 0.55, z := 0.65 }
  { x := 1200, y := 1600, z := 1900 }
  6500
  1000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_willam_land_percentage_l1137_113716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l1137_113726

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x

-- Define the theorem
theorem max_k_value : 
  ∃ (k : ℤ), (∀ (x : ℝ), x > 2 → k * (x - 2) < f x) ∧ 
  (∀ (m : ℤ), (∀ (x : ℝ), x > 2 → m * (x - 2) < f x) → m ≤ k) ∧
  k = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l1137_113726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_volume_l1137_113746

/-- Triangle vertices -/
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (42, 0)
def C : ℝ × ℝ := (18, 30)

/-- Midpoints of the triangle sides -/
noncomputable def D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
noncomputable def E : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
noncomputable def F : ℝ × ℝ := ((C.1 + A.1) / 2, (C.2 + A.2) / 2)

/-- Centroid of the triangle -/
noncomputable def G : ℝ × ℝ := ((D.1 + E.1 + F.1) / 3, (D.2 + E.2 + F.2) / 3)

/-- Area of the midpoint triangle -/
noncomputable def midpointTriangleArea : ℝ :=
  (1 / 2) * abs (D.1 * (E.2 - F.2) + E.1 * (F.2 - D.2) + F.1 * (D.2 - E.2))

/-- Height of the pyramid -/
noncomputable def pyramidHeight : ℝ := G.2

/-- Volume of the pyramid -/
noncomputable def pyramidVolume : ℝ := (1 / 3) * midpointTriangleArea * pyramidHeight

/-- Theorem: The volume of the triangular pyramid is 1050 -/
theorem triangular_pyramid_volume : pyramidVolume = 1050 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_volume_l1137_113746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_B_l1137_113749

noncomputable section

/-- Curve C₁ -/
def C₁ (t : ℝ) : ℝ × ℝ := (Real.cos t, 2 * Real.sin t)

/-- Line l -/
noncomputable def l (x : ℝ) : ℝ := Real.sqrt 3 * x

/-- Curve C₂ in polar coordinates -/
def C₂ (θ : ℝ) : ℝ := -8 * Real.cos θ

/-- Point A: intersection of C₂ and l in the third quadrant -/
noncomputable def A : ℝ × ℝ := sorry

/-- Point B: intersection of C₁ and l in the first quadrant -/
noncomputable def B : ℝ × ℝ := sorry

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_A_B : distance A B = (4 * Real.sqrt 7) / 7 + 4 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_B_l1137_113749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l1137_113750

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- Axiom for the symmetry of f about (-1, 0)
axiom f_symmetry (x : ℝ) : f x = f (-2 - x)

-- Axiom for the derivative condition
axiom f'_condition (x : ℝ) : x < -1 → (x + 1) * (f x + (x + 1) * f' x) < 0

-- Theorem statement
theorem solution_set_of_inequality :
  {x : ℝ | x * f' (x - 1) > f' 0} = Set.Ioo (-1 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l1137_113750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angle_relations_l1137_113700

theorem triangle_right_angle_relations (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 ≤ A) → (A ≤ Real.pi) →
  (0 ≤ B) → (B ≤ Real.pi) →
  (0 ≤ C) → (C ≤ Real.pi) →
  (A + B + C = Real.pi) →
  (a * Real.sin B = b * Real.sin A) →
  (b * Real.sin C = c * Real.sin B) →
  (c * Real.sin A = a * Real.sin C) →
  ((C = Real.pi / 2) → (a^2 + b^2 = c^2)) ∧
  ((B = Real.pi / 2) → (a^2 + c^2 = b^2)) ∧
  ((A = Real.pi / 2) → (b^2 + c^2 = a^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angle_relations_l1137_113700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_student_pullups_l1137_113778

/-- Represents the number of pull-ups done by each student -/
def PullUps : Fin 5 → ℕ := sorry

/-- The average number of pull-ups per person -/
def average : ℕ := 10

/-- The total number of students -/
def total_students : ℕ := 5

theorem fourth_student_pullups
  (h1 : PullUps 0 = 9)
  (h2 : PullUps 1 = 12)
  (h3 : PullUps 2 = 9)
  (h5 : PullUps 4 = 8)
  (h_avg : (PullUps 0 + PullUps 1 + PullUps 2 + PullUps 3 + PullUps 4) / total_students = average) :
  PullUps 3 = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_student_pullups_l1137_113778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_three_fifth_power_seven_l1137_113796

theorem cube_root_of_three_fifth_power_seven (x : ℝ) :
  x = (5^7 + 5^7 + 5^7)^(1/3) → x = 25 * 25^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_three_fifth_power_seven_l1137_113796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_in_hexadecagon_l1137_113720

/-- The area of a triangle in a regular hexadecagon -/
theorem area_triangle_in_hexadecagon (side_length : ℝ) (h : side_length = 4) :
  let angle_center : ℝ := 2 * Real.pi / 16
  let R : ℝ := side_length / (2 * Real.sin (angle_center / 2))
  let triangle_angle : ℝ := 3 * angle_center
  let triangle_side : ℝ := 2 * R * Real.sin (triangle_angle / 2)
  (1 / 2) * triangle_side * triangle_side * Real.sin triangle_angle =
    8 * (Real.sin (33.75 * Real.pi / 180))^2 * Real.sin (67.5 * Real.pi / 180) /
    (Real.sin (11.25 * Real.pi / 180))^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_in_hexadecagon_l1137_113720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1137_113715

open Real

noncomputable def f (a : ℝ) (x : ℝ) := a * x - log x

theorem f_properties :
  ∀ a : ℝ, ∀ x : ℝ, x > 0 →
  (∃ y : ℝ, (f 2 x - f 2 1) = (x - 1) ∧ y = f 2 x ∧ x - y + 1 = 0) ∧
  (∀ x : ℝ, x > 0 → (deriv (f a) 1 = 0 → (∀ z : ℝ, z > 1 → deriv (f a) z > 0))) ∧
  (∃ m : ℝ, m = 3 ∧ (∀ z : ℝ, 0 < z ∧ z ≤ exp 1 → f a z ≥ m) → a = exp 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1137_113715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1137_113757

/-- A hyperbola with a vertex at (2, 0) and an asymptote y = √2x has the equation x²/4 - y²/8 = 1 -/
theorem hyperbola_equation (h : Set (ℝ × ℝ)) : 
  (∃ v : ℝ × ℝ, v ∈ h ∧ v = (2, 0)) → 
  (∃ f : ℝ → ℝ, (∀ x y, (x, y) ∈ h → y = f x) ∧ ∀ x, f x = Real.sqrt 2 * x) → 
  ∀ x y, (x, y) ∈ h ↔ x^2 / 4 - y^2 / 8 = 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1137_113757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_yOz_coords_symmetry_point_A_l1137_113742

/-- Symmetry with respect to the yOz plane in a spatial rectangular coordinate system -/
def symmetry_yOz (A : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-(A.1), A.2.1, A.2.2)

theorem symmetry_yOz_coords (x y z : ℝ) :
  symmetry_yOz (x, y, z) = (-x, y, z) := by
  sorry

/-- The specific point A(2, -3, 4) -/
def point_A : ℝ × ℝ × ℝ := (2, -3, 4)

theorem symmetry_point_A :
  symmetry_yOz point_A = (-2, -3, 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_yOz_coords_symmetry_point_A_l1137_113742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_analysis_l1137_113790

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x + φ) + Real.cos (2 * x + φ)

theorem function_analysis (φ : ℝ) (h1 : |φ| < π / 2) :
  ∃ (g : ℝ → ℝ), 
    (∀ x, g x = f (x + π / 3) φ) ∧ 
    (∀ x, g x = g (-x)) ∧
    (∀ x, f x φ = 2 * Real.sin (2 * x - π / 6)) ∧
    (∀ a : ℝ, (∃! (r1 r2 : ℝ), r1 ∈ Set.Icc (π / 6) (5 * π / 12) ∧ 
                               r2 ∈ Set.Icc (π / 6) (5 * π / 12) ∧ 
                               r1 ≠ r2 ∧ 
                               f r1 φ = a ∧ 
                               f r2 φ = a) 
              → a ∈ Set.Icc (Real.sqrt 3) 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_analysis_l1137_113790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_8_ball_probability_l1137_113727

open Real BigOperators

theorem magic_8_ball_probability :
  let n : ℕ := 7  -- total number of questions
  let k : ℕ := 3  -- number of positive answers
  let p : ℝ := 2/5  -- probability of a positive answer
  Nat.choose n k * p^k * (1-p)^(n-k) = 22680/78125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_8_ball_probability_l1137_113727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_encounter_time_correct_l1137_113711

/-- The time until the first encounter of two bodies moving on a circle -/
noncomputable def time_to_first_encounter (v a : ℝ) : ℝ := v * (Real.sqrt 5 - 1) / a

/-- The circumference of the circle -/
noncomputable def circle_circumference (v a : ℝ) : ℝ := 2 * v^2 / a

/-- The time until the second encounter (back at the starting point) -/
noncomputable def time_to_second_encounter (v a : ℝ) : ℝ := circle_circumference v a / v

theorem first_encounter_time_correct (v a : ℝ) (hv : v > 0) (ha : a > 0) :
  let t₁ := time_to_first_encounter v a
  let t₂ := time_to_second_encounter v a
  let l := circle_circumference v a
  v * t₁ + a * t₁^2 / 2 = l ∧
  v * t₂ = l ∧
  a * t₂^2 / 2 = l :=
by sorry

#check first_encounter_time_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_encounter_time_correct_l1137_113711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_equation_slope_of_given_line_l1137_113780

/-- The slope of a line given by the equation ax + by = c, where b ≠ 0 -/
def line_slope (a b c : ℚ) : ℚ := -a / b

theorem line_slope_equation (a b c : ℚ) (h : b ≠ 0) :
  line_slope a b c = -a / b :=
by
  unfold line_slope
  rfl

theorem slope_of_given_line :
  line_slope 4 7 28 = -4 / 7 :=
by
  unfold line_slope
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_equation_slope_of_given_line_l1137_113780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_lambda_inverse_l1137_113798

theorem xyz_lambda_inverse (x y z l : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) (pos_l : l > 0)
  (eq1 : y * z = 6 * l * x)
  (eq2 : x * z = 6 * l * y)
  (eq3 : x * y = 6 * l * z)
  (eq4 : x^2 + y^2 + z^2 = 1) :
  (x * y * z * l)⁻¹ = 54 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_lambda_inverse_l1137_113798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1137_113731

-- Define the ⊕ operation
noncomputable def oplus (a b : ℝ) : ℝ :=
  if a ≥ b then a else b^2

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  (oplus 1 x) * x - (oplus 2 x)

-- State the theorem
theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) 2 ∧
  f x = 6 ∧
  ∀ (y : ℝ), y ∈ Set.Icc (-2 : ℝ) 2 → f y ≤ f x := by
  -- Proof goes here
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1137_113731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_theta_pi_sixth_l1137_113703

noncomputable def f (x θ : ℝ) : ℝ := Real.sin (x + θ) + Real.sqrt 3 * Real.cos (x + θ)

theorem even_function_implies_theta_pi_sixth (θ : ℝ) 
  (h1 : θ ≥ -π/2 ∧ θ < π/2) 
  (h2 : ∀ x, f x θ = f (-x) θ) : 
  θ = π/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_theta_pi_sixth_l1137_113703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_product_plus_positive_l1137_113736

theorem negative_product_plus_positive (a b : ℤ) (ha : a > 0) (hb : b > 0) : 
  (-11 * a) * (-8 * b) + a * b = 89 * a * b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_product_plus_positive_l1137_113736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_justin_current_age_l1137_113735

/-- Represents a person's age --/
structure Age where
  years : ℕ

/-- Represents the current time --/
def CurrentTime : Type := Unit

/-- Represents a future time --/
def FutureTime : Type := Unit

/-- The age of a person at a given time --/
def age_at (person : String) (time : Type) : Age :=
  ⟨0⟩  -- Default implementation, replace with actual logic if needed

/-- The difference in years between two ages --/
def age_difference (a b : Age) : ℕ :=
  if a.years ≥ b.years then a.years - b.years else b.years - a.years

theorem justin_current_age :
  let angelina_future_age := age_at "Angelina" FutureTime
  let angelina_current_age := age_at "Angelina" CurrentTime
  let justin_current_age := age_at "Justin" CurrentTime
  (angelina_future_age.years = 40) →
  (age_difference angelina_future_age angelina_current_age = 5) →
  (age_difference angelina_current_age justin_current_age = 4) →
  justin_current_age.years = 31 :=
by
  intros h1 h2 h3
  sorry  -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_justin_current_age_l1137_113735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_result_l1137_113788

-- Define the participants
inductive Participant
| Olya
| Oleg
| Pasha

-- Define the possible places
inductive Place
| First
| Second
| Third

-- Define a function to represent the placement of participants
def placement : Participant → Place
| _ => Place.First  -- Default value, will be overridden by axioms

-- Define a function to represent whether a participant is a boy
def is_boy : Participant → Prop
| _ => False  -- Default value, will be overridden by axioms

-- Define a function to represent whether a participant is telling the truth
def is_truthful : Participant → Prop
| _ => False  -- Default value, will be overridden by axioms

-- All participants claim to be first
axiom all_claim_first :
  ∀ p : Participant, is_truthful p → placement p = Place.First

-- Olya's statement about odd places
axiom olya_odd_places :
  is_truthful Participant.Olya →
  (placement Participant.Olya = Place.First ∨ placement Participant.Olya = Place.Third) →
  is_boy Participant.Olya

-- Oleg's statement about Olya
axiom oleg_about_olya :
  is_truthful Participant.Oleg →
  ¬(is_truthful Participant.Olya)

-- All participants either always lie or always tell the truth
axiom consistent_truthfulness :
  (∀ p : Participant, is_truthful p) ∨ (∀ p : Participant, ¬is_truthful p)

-- All places are taken
axiom all_places_taken :
  ∃! p₁ : Participant, placement p₁ = Place.First ∧
  ∃! p₂ : Participant, placement p₂ = Place.Second ∧
  ∃! p₃ : Participant, placement p₃ = Place.Third

-- The main theorem to prove
theorem competition_result :
  placement Participant.Oleg = Place.First ∧
  placement Participant.Pasha = Place.Second ∧
  placement Participant.Olya = Place.Third := by
  sorry

#check competition_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_result_l1137_113788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_dozen_pens_l1137_113713

-- Define the cost ratio of pen to pencil
def pen_pencil_ratio : ℚ := 5 / 1

-- Define the total cost of 3 pens and 5 pencils
def total_cost : ℚ := 150

-- Define the number of pens and pencils in the given set
def num_pens : ℕ := 3
def num_pencils : ℕ := 5

-- Define the number of pens in a dozen
def dozen : ℕ := 12

-- Calculate the cost of one pen
def pen_cost : ℚ := total_cost * pen_pencil_ratio / (num_pens * pen_pencil_ratio + num_pencils)

-- Theorem statement
theorem cost_of_dozen_pens : dozen * pen_cost = 450 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_dozen_pens_l1137_113713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_J_radius_l1137_113789

-- Define the radius of circle Z
noncomputable def radius_Z : ℝ := 15

-- Define the radius of circle G
noncomputable def radius_G : ℝ := 5

-- Define the radius of circles H and I
noncomputable def radius_HI : ℝ := 3

-- Define the radius of circle F
noncomputable def radius_F : ℝ := 1

-- Define the radius of circle J
noncomputable def radius_J : ℝ := 137 / 8

theorem circle_J_radius :
  let ZJ := radius_Z - radius_G + radius_J
  let ZHI := radius_Z - radius_HI + radius_J
  (ZHI^2 : ℝ) = ZJ^2 + radius_Z^2 - radius_Z^2 * (1/2) :=
by sorry

#eval Int.gcd 137 8
#eval 137 + 8

-- The final answer is 145

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_J_radius_l1137_113789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_primes_in_E_l1137_113769

-- Define the property of E containing all prime factors of the product minus one
def contains_prime_factors (E : Set Nat) : Prop :=
  ∀ (S : Finset Nat), S.Nonempty → (∀ p ∈ S, p ∈ E ∧ Nat.Prime p) →
    ∀ q, Nat.Prime q → q ∣ (S.prod id - 1) → q ∈ E

-- State the theorem
theorem all_primes_in_E (E : Set Nat) 
    (h1 : ∃ p q, p ∈ E ∧ q ∈ E ∧ p ≠ q ∧ Nat.Prime p ∧ Nat.Prime q)
    (h2 : contains_prime_factors E) :
    ∀ p, Nat.Prime p → p ∈ E := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_primes_in_E_l1137_113769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1137_113737

theorem lambda_range (l : ℝ) : (∀ a b : ℝ, a^2 + 8*b^2 ≥ l*b*(a+b)) ↔ -8 ≤ l ∧ l ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1137_113737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_min_value_is_six_min_value_achieved_l1137_113791

theorem min_value_of_expression (x y : ℝ) (h : x + 2*y = 2) : 
  ∀ a b : ℝ, a + 2*b = 2 → (3 : ℝ)^x + (3 : ℝ)^(2*y) ≤ (3 : ℝ)^a + (3 : ℝ)^(2*b) :=
by
  sorry

theorem min_value_is_six (x y : ℝ) (h : x + 2*y = 2) : 
  (3 : ℝ)^x + (3 : ℝ)^(2*y) ≥ 6 :=
by
  sorry

theorem min_value_achieved (x y : ℝ) (h : x + 2*y = 2) : 
  ∃ a b : ℝ, a + 2*b = 2 ∧ (3 : ℝ)^a + (3 : ℝ)^(2*b) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_min_value_is_six_min_value_achieved_l1137_113791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_M_l1137_113714

theorem sum_of_digits_M (M : ℕ) (h : M ^ 2 = 36 ^ 49 * 49 ^ 36) : 
  (Nat.digits 10 M).sum = 270 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_M_l1137_113714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_minimum_t_value_ln_inequality_l1137_113719

noncomputable section

def f (x : ℝ) := (Real.log x) / (x + 1)

theorem tangent_line_at_one (x y : ℝ) :
  HasDerivAt f (1/2) 1 → x - 2*y - 1 = 0 := by sorry

theorem minimum_t_value (t : ℝ) :
  (∀ x > 0, f x + t/x ≥ 2/(x+1)) → t ≥ 1 := by sorry

theorem ln_inequality (n : ℕ) (hn : n ≥ 2) :
  Real.log n > (Finset.range (n-1)).sum (λ i ↦ 1 / ((i+2) : ℝ)) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_minimum_t_value_ln_inequality_l1137_113719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_x_value_range_of_g_l1137_113792

-- Define the vectors
noncomputable def a : ℝ × ℝ := (Real.sqrt 3, 1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x, -Real.cos x)

-- Part 1
theorem parallel_vectors_x_value (x : ℝ) (h1 : x ∈ Set.Icc 0 Real.pi) 
  (h2 : ∃ (k : ℝ), a = k • (b x)) : x = (2 / 3) * Real.pi := by sorry

-- Part 2
-- Define the dot product of vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define functions f and g
noncomputable def f (x : ℝ) : ℝ := dot_product a (b x)
noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 3)

-- Theorem for the range of g(x)
theorem range_of_g : Set.range g = Set.Icc (-2) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_x_value_range_of_g_l1137_113792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_same_digits_l1137_113752

def base_number : ℕ := 15873

def sequence_term (n : ℕ) : ℕ := n * base_number

theorem all_same_digits (n : ℕ) (h : 1 ≤ n ∧ n ≤ 9) : 
  ∃ (d : ℕ), sequence_term n * 7 = d * 111111 := by
  sorry

#eval base_number * 7  -- This should output 111111

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_same_digits_l1137_113752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Fe_approx_l1137_113771

-- Define the molar masses
noncomputable def molar_mass_Fe : ℝ := 55.845
noncomputable def molar_mass_O : ℝ := 15.999

-- Define the molar mass of Fe2O3
noncomputable def molar_mass_Fe2O3 : ℝ := 2 * molar_mass_Fe + 3 * molar_mass_O

-- Define the mass percentage of Fe in Fe2O3
noncomputable def mass_percentage_Fe : ℝ := (2 * molar_mass_Fe / molar_mass_Fe2O3) * 100

-- Define the given mass percentage of the compound
noncomputable def given_mass_percentage : ℝ := 70

-- Theorem statement
theorem mass_percentage_Fe_approx :
  ∃ ε > 0, abs (mass_percentage_Fe - 69.94) < ε ∧ 
  abs (given_mass_percentage - mass_percentage_Fe) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Fe_approx_l1137_113771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_of_f_l1137_113701

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) := x + 4 / x

-- State the theorem
theorem minimum_value_of_f :
  (∀ x > 0, f x ≥ 4) ∧ (∃ x > 0, f x = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_of_f_l1137_113701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_table_ants_cannot_visit_all_points_l1137_113702

/-- Represents a polygonal table -/
structure PolygonalTable where
  sides : List ℝ
  all_sides_longer_than_1 : ∀ s ∈ sides, s > 1

/-- Represents the position of an ant on the table -/
structure AntPosition where
  edge : ℕ
  position : ℝ

/-- Represents the state of two ants on the table -/
structure TwoAntsState where
  table : PolygonalTable
  ant1 : AntPosition
  ant2 : AntPosition
  distance_10cm : abs (ant1.position - ant2.position) = 0.1
  on_same_side_initially : ant1.edge = ant2.edge

/-- Theorem: There exists a polygonal table where two ants cannot visit all points -/
theorem exists_table_ants_cannot_visit_all_points :
  ∃ (table : PolygonalTable),
    ∀ (initial_state : TwoAntsState),
    ∃ (unvisited_point : AntPosition),
      ∀ (final_state : TwoAntsState),
        final_state.table = initial_state.table →
        unvisited_point.edge ∈ List.range (List.length table.sides) →
        unvisited_point ≠ final_state.ant1 ∧ unvisited_point ≠ final_state.ant2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_table_ants_cannot_visit_all_points_l1137_113702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_numbers_match_first_digit_l1137_113722

/-- Represents a four-digit number formed from digits 1, 2, and 3 -/
def FourDigitNumber := Fin 4 → Fin 3

/-- Represents the assignment of 1, 2, or 3 to each four-digit number -/
def Assignment := FourDigitNumber → Fin 3

/-- Checks if a four-digit number has no matching digits -/
def hasNoMatchingDigits (n : FourDigitNumber) : Prop :=
  ∀ i j : Fin 4, i ≠ j → n i ≠ n j

/-- The main theorem to be proved -/
theorem all_numbers_match_first_digit 
  (assign : Assignment)
  (h1 : ∃ S : Finset FourDigitNumber, S.card = 74 ∧ 
    (∀ n, n ∈ S → hasNoMatchingDigits n) ∧ 
    (∀ n m, n ∈ S → m ∈ S → n ≠ m → assign n ≠ assign m))
  (h2 : assign (λ _ => 0) = 0)
  (h3 : assign (λ _ => 1) = 1)
  (h4 : assign (λ _ => 2) = 2)
  (h5 : assign (λ i => if i = 0 then 0 else 1) = 0)
  : ∀ n : FourDigitNumber, assign n = n 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_numbers_match_first_digit_l1137_113722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_in_A_l1137_113734

def set_A : Set ℝ := {x : ℝ | |x - 2| ≤ 5}

theorem smallest_integer_in_A : 
  ∃ (n : ℤ), (n : ℝ) ∈ set_A ∧ (∀ (m : ℤ), (m : ℝ) ∈ set_A → n ≤ m) ∧ n = -3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_in_A_l1137_113734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_triangle_angles_l1137_113705

/-- A triangle with sides forming a geometric sequence --/
structure GeometricTriangle where
  a : ℝ
  q : ℝ
  h_a_pos : 0 < a
  h_q_pos : 0 < q

/-- The angles of a geometric triangle --/
noncomputable def angles (t : GeometricTriangle) : ℝ × ℝ × ℝ :=
  let α := Real.arccos ((t.q^4 + t.q^2 - 1) / (2 * t.q^3))
  let β := Real.arccos ((1 - t.q^2 + t.q^4) / (2 * t.q^2))
  let γ := Real.arccos ((1 + t.q^2 - t.q^4) / (2 * t.q))
  (α, β, γ)

theorem geometric_triangle_angles (t : GeometricTriangle) :
  let (α, β, γ) := angles t
  Real.cos α = (t.q^4 + t.q^2 - 1) / (2 * t.q^3) ∧
  Real.cos β = (1 - t.q^2 + t.q^4) / (2 * t.q^2) ∧
  Real.cos γ = (1 + t.q^2 - t.q^4) / (2 * t.q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_triangle_angles_l1137_113705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_difference_is_zero_l1137_113768

noncomputable def log_product (bases : List ℕ) (args : List ℕ) : ℝ :=
  (args.map Real.log).prod / (bases.map Real.log).prod

theorem log_product_difference_is_zero :
  let bases := List.range 50 |>.map (λ i => i + 52)
  let args := List.range 50 |>.map (λ i => i + 2)
  ∀ (perm_bases : List ℕ) (perm_args : List ℕ),
    perm_bases.Perm bases → perm_args.Perm args →
    log_product perm_bases perm_args = log_product bases args :=
by
  sorry

#check log_product_difference_is_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_difference_is_zero_l1137_113768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_decagon_angle_measure_l1137_113787

/-- The number of sides in a regular decagon -/
def n : ℕ := 10

/-- The sum of interior angles of a polygon -/
noncomputable def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- The measure of one angle in a regular polygon -/
noncomputable def angle_measure (n : ℕ) : ℝ := sum_interior_angles n / n

/-- Theorem: The measure of one angle of a regular decagon is 144 degrees -/
theorem regular_decagon_angle_measure :
  angle_measure n = 144 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_decagon_angle_measure_l1137_113787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1137_113763

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) (lambda : ℝ) : 
  let m : ℝ × ℝ := (c * Real.cos C, Real.sqrt 3)
  let n : ℝ × ℝ := (2, a * Real.cos B + b * Real.cos A)
  (m.1 * n.1 + m.2 * n.2 = 0) →
  (c^2 = (5 + 2 * Real.sqrt 3) * b^2) →
  (1/2 * a * b * Real.sin C = 1/2) →
  (lambda * Real.sin (4*A) = Real.sin (2*A) + Real.cos (2*A)) →
  (a = 2 ∧ b = 1 ∧ 
   ∀ mu : ℝ, (mu * Real.sin (4*A) = Real.sin (2*A) + Real.cos (2*A)) → mu ≥ 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1137_113763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_zero_two_intersections_l1137_113777

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 + x * Real.sin x + Real.cos x

-- Theorem 1: The curve y = f(x) is tangent to y = 1 at (0, 1)
theorem tangent_at_zero :
  (f 0 = 1) ∧ 
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-ε) ε \ {0}, f x > 1) := by
  sorry

-- Theorem 2: f(x) intersects any horizontal line y = b where b > 1 at exactly two points
theorem two_intersections (b : ℝ) (h : b > 1) :
  ∃! (x₁ x₂ : ℝ), x₁ < x₂ ∧ f x₁ = b ∧ f x₂ = b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_zero_two_intersections_l1137_113777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_N_l1137_113779

-- Define the circle
def myCircle (x y : ℝ) : Prop := x^2 + y^2 - 8*x = 0

-- Define the chord OA passing through the origin
def chordOA (x y : ℝ) : Prop := ∃ t : ℝ, x = t ∧ y = t * (y / x)

-- Define the extension of OA to N such that |OA| = |AN|
def extendOAtoN (x y : ℝ) : Prop := ∃ a : ℝ × ℝ, 
  chordOA a.1 a.2 ∧ 
  myCircle a.1 a.2 ∧ 
  x = 2 * a.1 ∧ 
  y = 2 * a.2

-- Theorem stating the locus of point N
theorem locus_of_N : 
  ∀ x y : ℝ, extendOAtoN x y → x^2 + y^2 - 16*x = 0 := by
  sorry

#check locus_of_N

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_N_l1137_113779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_for_odd_g_l1137_113797

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := Real.cos x * (Real.sin x + Real.sqrt 3 * Real.cos x) - Real.sqrt 3 / 2

/-- The function g(x) as defined in the problem -/
noncomputable def g (α : ℝ) (x : ℝ) : ℝ := f (x + α)

/-- A function is odd if f(-x) = -f(x) for all x -/
def is_odd (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = -h x

/-- The main theorem -/
theorem alpha_value_for_odd_g (α : ℝ) (h_α_pos : α > 0) (h_g_odd : is_odd (g α)) :
  ∃ k : ℕ, k > 0 ∧ α = k * Real.pi / 2 - Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_for_odd_g_l1137_113797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1137_113732

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (x + Real.pi / 4)

theorem f_increasing_on_interval :
  ∀ x ∈ Set.Icc 0 (Real.pi / 4), StrictMono (fun x ↦ f x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1137_113732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excitedCellCountCorrect_l1137_113725

/-- Represents the state of a cell: rest or excited -/
inductive CellState
| Rest
| Excited

/-- Represents an infinite chain of nerve cells -/
def InfiniteChain := ℤ → CellState

/-- Counts the number of 1's in the binary representation of a natural number -/
def countOnes : ℕ → ℕ := sorry

/-- Calculates the number of excited cells at time t -/
def excitedCellCount (t : ℕ) : ℕ := 2^(countOnes t)

/-- Simulates the state of the infinite chain at time t -/
def chainState : ℕ → InfiniteChain := sorry

/-- Initial state with only one cell excited -/
def initialState : InfiniteChain :=
  fun i => if i = 0 then CellState.Excited else CellState.Rest

/-- Counts the number of excited cells in a given chain state -/
def countExcitedCells (chain : InfiniteChain) : ℕ := sorry

theorem excitedCellCountCorrect (t : ℕ) :
  countExcitedCells (chainState t) = excitedCellCount t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_excitedCellCountCorrect_l1137_113725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_installment_calculation_l1137_113755

/-- Calculates the first installment payment for a discounted television purchase -/
theorem first_installment_calculation (original_price : ℚ) (discount_rate : ℚ) 
  (num_installments : ℕ) (installment_amount : ℚ) : 
  original_price = 480 →
  discount_rate = 5 / 100 →
  num_installments = 3 →
  installment_amount = 102 →
  original_price * (1 - discount_rate) - num_installments * installment_amount = 150 := by
  intros h1 h2 h3 h4
  -- Proof steps would go here
  sorry

#check first_installment_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_installment_calculation_l1137_113755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1137_113738

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos x)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem triangle_area (a b c : ℝ) (A : ℝ) (h1 : a = 1) (h2 : b + c = 2) (h3 : f A = 1) :
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1137_113738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_monotonicity_l1137_113759

-- Define the power function
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(-2*m - 1)

-- State the theorem
theorem power_function_monotonicity :
  ∃ (m : ℝ), ∀ (x y : ℝ), 0 < x → 0 < y → x < y → f m x < f m y ↔ m = -1 :=
by
  -- We use 'sorry' to skip the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_monotonicity_l1137_113759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_a_range_l1137_113747

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 - (x - 1) * Real.exp x

-- State the theorem
theorem function_inequality_implies_a_range :
  ∀ a : ℝ,
  (∀ x₁ x₂ x₃ : ℝ, x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 → x₃ ∈ Set.Icc 0 1 →
    f a x₁ + f a x₂ ≥ f a x₃) →
  a ∈ Set.Icc 1 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_a_range_l1137_113747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_edmonton_to_red_deer_distance_l1137_113758

/-- The distance between two cities in kilometers -/
def distance (city1 city2 : String) : ℝ := sorry

/-- The time taken to travel between two cities in hours -/
def travel_time (city1 city2 : String) : ℝ := sorry

/-- The speed of travel in kilometers per hour -/
def speed : ℝ := 110

/-- Edmonton is north of Red Deer -/
axiom edmonton_north_of_red_deer : distance "Edmonton" "Red Deer" > 0

/-- Calgary is 110 kilometers south of Red Deer -/
axiom calgary_south_of_red_deer : distance "Calgary" "Red Deer" = 110

/-- It takes 3 hours to get from Edmonton to Calgary -/
axiom edmonton_to_calgary_time : travel_time "Edmonton" "Calgary" = 3

theorem edmonton_to_red_deer_distance :
  distance "Edmonton" "Red Deer" = 220 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_edmonton_to_red_deer_distance_l1137_113758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_A_on_transformed_plane_point_A_in_image_of_plane_l1137_113717

/-- The original plane equation -/
noncomputable def plane_equation (x y z : ℝ) : Prop := 3 * x - y + 5 * z - 6 = 0

/-- The scale factor of the similarity transformation -/
noncomputable def k : ℝ := 5 / 6

/-- The point A -/
noncomputable def point_A : ℝ × ℝ × ℝ := (1/3, 1, 1)

/-- The transformed plane equation after similarity transformation -/
noncomputable def transformed_plane_equation (x y z : ℝ) : Prop := 3 * x - y + 5 * z - 5 = 0

/-- Theorem stating that point A belongs to the transformed plane -/
theorem point_A_on_transformed_plane : 
  transformed_plane_equation point_A.1 point_A.2.1 point_A.2.2 := by
  sorry

/-- Main theorem: Point A belongs to the image of plane a after similarity transformation -/
theorem point_A_in_image_of_plane : 
  ∃ (x y z : ℝ), plane_equation x y z ∧ 
  transformed_plane_equation (k * x) (k * y) (k * z) ∧
  (k * x, k * y, k * z) = point_A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_A_on_transformed_plane_point_A_in_image_of_plane_l1137_113717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l1137_113756

/-- Evaluates an expression with right-to-left grouping -/
noncomputable def evaluateRightToLeft (a b c d e : ℝ) : ℝ :=
  a / (b - c * (d + e))

/-- Proves that the given expression with right-to-left grouping 
    is equivalent to a / (b - c * (d + e)) -/
theorem expression_evaluation (a b c d e : ℝ) :
  evaluateRightToLeft a b c d e = a / (b - c * (d + e)) := by
  -- Unfold the definition of evaluateRightToLeft
  unfold evaluateRightToLeft
  -- The equality is true by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l1137_113756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_alpha_value_l1137_113748

noncomputable def f (x : ℝ) := Real.cos x * (Real.sin x + Real.sqrt 3 * Real.cos x) - Real.sqrt 3 / 2

theorem odd_function_alpha_value (α : ℝ) (h1 : α > 0) 
  (h2 : ∀ x, f (x + α) = -f (-x - α)) : 
  ∃ k : ℕ, α = (2 * k - 1) * Real.pi / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_alpha_value_l1137_113748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_521_l1137_113729

/-- The binary representation of a natural number -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec go (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
  go n

/-- Convert a list of booleans to a binary string -/
def binaryToString (l : List Bool) : String :=
  l.reverse.map (fun b => if b then '1' else '0') |> String.mk

theorem binary_521 :
  binaryToString (toBinary 521) = "1000001001" := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_521_l1137_113729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_shortest_chord_l1137_113760

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on the circle
structure PointOnCircle (c : Circle) where
  point : ℝ × ℝ
  on_circle : (point.1 - c.center.1)^2 + (point.2 - c.center.2)^2 = c.radius^2

-- Define a chord
structure Chord (c : Circle) where
  start : ℝ × ℝ
  endpoint : ℝ × ℝ
  on_circle : (start.1 - c.center.1)^2 + (start.2 - c.center.2)^2 = c.radius^2 ∧
              (endpoint.1 - c.center.1)^2 + (endpoint.2 - c.center.2)^2 = c.radius^2

-- Define the length of a chord
noncomputable def chord_length (chord : Chord c) : ℝ :=
  Real.sqrt ((chord.endpoint.1 - chord.start.1)^2 + (chord.endpoint.2 - chord.start.2)^2)

-- Define a diameter
def is_diameter (chord : Chord c) : Prop :=
  ∃ (t : ℝ), chord.start = (c.center.1 + t * (chord.endpoint.1 - c.center.1),
                            c.center.2 + t * (chord.endpoint.2 - c.center.2))

-- Define perpendicularity
def perpendicular (chord1 chord2 : Chord c) : Prop :=
  (chord1.endpoint.1 - chord1.start.1) * (chord2.endpoint.1 - chord2.start.1) +
  (chord1.endpoint.2 - chord1.start.2) * (chord2.endpoint.2 - chord2.start.2) = 0

-- Theorem statement
theorem longest_shortest_chord (c : Circle) (M : PointOnCircle c) :
  ∃ (longest shortest : Chord c),
    (M.point = longest.start ∨ M.point = longest.endpoint) ∧
    (M.point = shortest.start ∨ M.point = shortest.endpoint) ∧
    is_diameter longest ∧
    perpendicular longest shortest ∧
    (∀ (chord : Chord c), (M.point = chord.start ∨ M.point = chord.endpoint) →
      chord_length chord ≤ chord_length longest ∧
      chord_length shortest ≤ chord_length chord) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_shortest_chord_l1137_113760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_problem_l1137_113793

theorem factor_problem (x y : ℕ) (h1 : y = x + 10) 
  (h2 : (x * y - 40) / x = 39) (h3 : (x * y - 40) % x = 22) : 
  x = 31 ∧ y = 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_problem_l1137_113793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_sum_zero_l1137_113776

-- Define the function f
noncomputable def f (x b : ℝ) : ℝ := 2016 * x^3 - Real.sin x + b + 2

-- State the theorem
theorem odd_function_sum_zero 
  (a b : ℝ) 
  (h_domain : Set.Icc (a - 4) (2 * a - 2) = Set.Icc (a - 4) (a + 2))
  (h_odd : ∀ x, f x b = -f (-x) b) :
  f a b + f b b = 0 := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_sum_zero_l1137_113776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_ratio_equals_five_fourths_l1137_113762

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define the triangle
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

def triangle_satisfies_conditions (t : Triangle) : Prop :=
  t.A = (-4, 0) ∧ t.C = (4, 0) ∧ is_on_ellipse t.B.1 t.B.2

-- Define the angle function (this is a placeholder)
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Define the sine ratio
noncomputable def sine_ratio (t : Triangle) : ℝ :=
  (Real.sin (angle t.B t.A t.C) + Real.sin (angle t.B t.C t.A)) / Real.sin (angle t.A t.B t.C)

-- The theorem
theorem sine_ratio_equals_five_fourths (t : Triangle) 
  (h : triangle_satisfies_conditions t) : 
  sine_ratio t = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_ratio_equals_five_fourths_l1137_113762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_ratio_triangle_l1137_113709

/-- Given a triangle with angles in the ratio 3:4:5, the largest angle is 75 degrees -/
theorem largest_angle_in_ratio_triangle : 
  ∀ (a b c : ℝ), 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 180 →
  (a : ℝ) / 3 = (b : ℝ) / 4 ∧ (b : ℝ) / 4 = (c : ℝ) / 5 →
  max a (max b c) = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_ratio_triangle_l1137_113709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_power_two_and_181_l1137_113781

theorem smallest_difference_power_two_and_181 :
  (∀ m n : ℕ, m > 0 → n > 0 → |((2 : ℤ)^m - (181 : ℤ)^n)| ≥ 7) ∧
  (∃ m n : ℕ, m > 0 ∧ n > 0 ∧ |((2 : ℤ)^m - (181 : ℤ)^n)| = 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_power_two_and_181_l1137_113781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_needed_proof_l1137_113743

/-- Represents the dimensions of a rectangular area in feet -/
structure Dimensions where
  length : ℚ
  width : ℚ

/-- Calculates the area of a rectangular space given its dimensions -/
def area (d : Dimensions) : ℚ := d.length * d.width

/-- Converts inches to feet -/
def inchesToFeet (inches : ℚ) : ℚ := inches / 12

/-- The dimensions of the entire floor -/
def floorDimensions : Dimensions := ⟨15, 20⟩

/-- The dimensions of a single tile in inches -/
def tileDimensionsInches : Dimensions := ⟨3, 9⟩

/-- The width of the border in feet -/
def borderWidth : ℚ := 1

/-- The dimensions of the main area to be tiled -/
def mainAreaDimensions : Dimensions := 
  ⟨floorDimensions.length - 2 * borderWidth, floorDimensions.width - 2 * borderWidth⟩

/-- The dimensions of a single tile in feet -/
def tileDimensionsFeet : Dimensions := 
  ⟨inchesToFeet tileDimensionsInches.length, inchesToFeet tileDimensionsInches.width⟩

theorem tiles_needed_proof : 
  (area mainAreaDimensions / area tileDimensionsFeet).floor = 1248 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_needed_proof_l1137_113743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_plus_2cos6_l1137_113707

theorem min_sin6_plus_2cos6 :
  ∃ (x : ℝ), ∀ (y : ℝ), Real.sin y ^ 6 + 2 * Real.cos y ^ 6 ≥ 2/3 ∧ Real.sin x ^ 6 + 2 * Real.cos x ^ 6 = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_plus_2cos6_l1137_113707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_odd_integers_l1137_113754

theorem min_odd_integers (a b c d e f : ℤ) 
  (sum_abc : a + b + c = 30)
  (sum_abcde : a + b + c + d + e = 48)
  (sum_abcdef : a + b + c + d + e + f = 59) :
  ∃ (odd_count : ℕ), odd_count ≥ 1 ∧
  odd_count = (if a % 2 ≠ 0 then 1 else 0) +
              (if b % 2 ≠ 0 then 1 else 0) +
              (if c % 2 ≠ 0 then 1 else 0) +
              (if d % 2 ≠ 0 then 1 else 0) +
              (if e % 2 ≠ 0 then 1 else 0) +
              (if f % 2 ≠ 0 then 1 else 0) ∧
  ∀ (other_odd_count : ℕ),
    other_odd_count = (if a % 2 ≠ 0 then 1 else 0) +
                      (if b % 2 ≠ 0 then 1 else 0) +
                      (if c % 2 ≠ 0 then 1 else 0) +
                      (if d % 2 ≠ 0 then 1 else 0) +
                      (if e % 2 ≠ 0 then 1 else 0) +
                      (if f % 2 ≠ 0 then 1 else 0) →
    odd_count ≤ other_odd_count := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_odd_integers_l1137_113754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1137_113718

noncomputable def f (x : ℝ) : ℝ := 
  if x < 1 then |x| + 2 else x + 2/x

theorem a_range (a : ℝ) : 
  (∀ x : ℝ, f x ≥ |x/2 + a|) → a ∈ Set.Icc (-2) 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1137_113718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l1137_113764

theorem remainder_problem (k : ℕ) 
  (h1 : k % 5 = 2)
  (h2 : k % 6 = 5)
  (h3 : k < 39) :
  k % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l1137_113764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_fraction_to_two_l1137_113782

theorem limit_fraction_to_two :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |((2 * (n : ℝ) - 5) / ((n : ℝ) + 1)) - 2| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_fraction_to_two_l1137_113782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_linear_functions_touch_theorem_l1137_113723

-- Define linear functions
def linear_function (a b : ℝ) : ℝ → ℝ := λ x ↦ a * x + b

-- Define the condition for two lines to be parallel
def parallel (f g : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ f = linear_function a b ∧ g = linear_function a c

-- Define the condition for a function to touch another function
def touches (f g : ℝ → ℝ) : Prop :=
  ∃! x, f x = g x

theorem parallel_linear_functions_touch_theorem
  (f g : ℝ → ℝ)
  (h_parallel : parallel f g)
  (h_not_horizontal : ∀ (k : ℝ), f ≠ λ x ↦ k)
  (h_touch : touches (λ x ↦ (f x)^2) (λ x ↦ -50 * g x)) :
  ∃! (A : ℝ), A = (1 : ℝ) / 50 ∧ touches (λ x ↦ (g x)^2) (λ x ↦ (f x) / A) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_linear_functions_touch_theorem_l1137_113723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1137_113799

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x * (2 - x))

-- Define the domain of f
def domain_f : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f (2 * x) / (x - 1)

-- Define the domain of g
def domain_g : Set ℝ := { x | 0 ≤ x ∧ x < 1 }

-- Theorem statement
theorem domain_of_g : 
  ∀ x : ℝ, x ∈ domain_g ↔ (∃ y : ℝ, y ∈ domain_f ∧ y = 2 * x) ∧ x ≠ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1137_113799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_difference_reversed_digits_l1137_113708

/-- Two-digit positive integer -/
def TwoDigitInteger (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- Reverses the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem greatest_difference_reversed_digits (q r : ℕ) :
  TwoDigitInteger q →
  TwoDigitInteger r →
  r = reverseDigits q →
  q ≥ r →
  q - r < 30 →
  ∀ (s t : ℕ), TwoDigitInteger s →
               TwoDigitInteger t →
               t = reverseDigits s →
               s ≥ t →
               s - t < 30 →
               q - r ≥ s - t →
  q - r = 27 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_difference_reversed_digits_l1137_113708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_poll_total_count_l1137_113772

theorem poll_total_count : ∀ (total_women : ℚ) (total_men : ℚ),
  total_women = total_men →
  (35 : ℚ) / 100 * total_women + 39 = total_women →
  total_women + total_men = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_poll_total_count_l1137_113772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trailing_zeroes_sum_factorials_l1137_113712

/-- Count the number of trailing zeroes in a factorial -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The number of trailing zeroes in the sum of factorials 73! + 79! + 83! -/
theorem trailing_zeroes_sum_factorials :
  trailingZeroes 73 = trailingZeroes (factorial 73 + factorial 79 + factorial 83) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trailing_zeroes_sum_factorials_l1137_113712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1137_113785

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 3 * Real.sin (ω * x + φ)

theorem function_properties (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : -π/2 < φ ∧ φ < π/2) 
  (h_sym : ∀ x, f ω φ (2*π/3 - x) = f ω φ (2*π/3 + x)) 
  (h_period : ∀ x, f ω φ (x + π) = f ω φ x) :
  ω = 2 ∧ 
  φ = π/6 ∧ 
  f ω φ (5*π/12) = 0 := by
  sorry

#check function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1137_113785
