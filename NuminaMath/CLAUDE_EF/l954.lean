import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_difference_equals_sqrt_three_over_two_l954_95408

theorem cos_squared_difference_equals_sqrt_three_over_two :
  (Real.cos (π / 12))^2 - (Real.cos (5 * π / 12))^2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_difference_equals_sqrt_three_over_two_l954_95408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_removed_triangles_area_l954_95438

theorem removed_triangles_area (s h : ℝ) (h_positive : h > 0) : 
  (4 * ((1 / 2) * (h / Real.sqrt 2) ^ 2)) = 64 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_removed_triangles_area_l954_95438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l954_95411

/-- The function f(x) is defined as the minimum of three linear functions -/
noncomputable def f (x : ℝ) : ℝ := min (3*x - 1) (min (-x + 4) (2*x + 5))

/-- The maximum value of f(x) is 11/4 -/
theorem max_value_of_f : 
  ∃ (M : ℝ), M = 11/4 ∧ ∀ (x : ℝ), f x ≤ M ∧ ∃ (x₀ : ℝ), f x₀ = M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l954_95411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l954_95425

theorem triangle_problem (A B C : Real) (a b c : Real) 
  (m n : Fin 2 → Real) :
  m = (λ i => if i = 0 then Real.sin A else Real.cos A) →
  n = (λ i => if i = 0 then Real.cos B else Real.sin B) →
  (m 0 * n 0 + m 1 * n 1) = Real.sin (2 * C) →
  ∃ r : Real, Real.sin A = r * Real.sin C ∧ Real.sin C = r * Real.sin B →
  a * b * Real.cos C = 18 →
  C = Real.pi / 3 ∧ c = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l954_95425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_undefined_at_one_l954_95455

/-- The function g(x) = (x-5)/(x-6) -/
noncomputable def g (x : ℝ) : ℝ := (x - 5) / (x - 6)

/-- The theorem stating that the inverse of g is undefined at x = 1 -/
theorem g_inverse_undefined_at_one : 
  ∀ f : ℝ → ℝ, (∀ x ≠ 1, f (g x) = x ∧ g (f x) = x) → 
  ¬∃ y : ℝ, g y = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_undefined_at_one_l954_95455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l954_95409

-- Define the parabola C₁
def C₁ (x y : ℝ) : Prop := y^2 = 8*x

-- Define the hyperbola C₂
def C₂ (x y a b : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

-- Define the directrix of C₁
def directrix (x : ℝ) : Prop := x = -2

-- Define the focus of C₁
def focus : ℝ × ℝ := (2, 0)

-- Define the asymptote of C₂
def asymptote (x y : ℝ) : Prop := y = (4*Real.sqrt 3/3)*x

-- Define points A and B
noncomputable def A : ℝ × ℝ := (-2, 4*Real.sqrt 3/3)
noncomputable def B : ℝ × ℝ := (-2, -4*Real.sqrt 3/3)

-- Define the equilateral triangle property
noncomputable def is_equilateral (F A B : ℝ × ℝ) : Prop :=
  let d₁ := Real.sqrt ((F.1 - A.1)^2 + (F.2 - A.2)^2)
  let d₂ := Real.sqrt ((F.1 - B.1)^2 + (F.2 - B.2)^2)
  let d₃ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  d₁ = d₂ ∧ d₂ = d₃ ∧ d₃ = d₁

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y, C₁ x y → directrix x → C₂ x y a b) ∧
  (∀ x y, asymptote x y) ∧
  is_equilateral focus A B →
  a^2 = 3 ∧ b^2 = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l954_95409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tripled_odot_nine_four_l954_95444

-- Define the custom operation
noncomputable def odot (a b : ℝ) : ℝ := a + (3 * a) / (2 * b)

-- State the theorem
theorem tripled_odot_nine_four : 
  3 * (odot 9 4) = 37.125 := by
  -- Unfold the definition of odot
  unfold odot
  -- Simplify the expression
  simp [mul_add, mul_div_assoc]
  -- Perform the numerical calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tripled_odot_nine_four_l954_95444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_g_exp_log_inequality_min_value_mn_l954_95448

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x
noncomputable def g (x : ℝ) : ℝ := x / (Real.exp x)

-- Statement 1: Maximum value of f and g
theorem max_value_f_g : 
  (∃ x₁ : ℝ, x₁ > 0 ∧ ∀ x > 0, f x ≤ f x₁) ∧ 
  (∃ x₂ : ℝ, x₂ > 0 ∧ ∀ x > 0, g x ≤ g x₂) ∧ 
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ f x₁ = g x₂ ∧ f x₁ = 1 / Real.exp 1) :=
sorry

-- Statement 2: Inequality involving e^x, ln x, and 2x
theorem exp_log_inequality :
  ∀ x : ℝ, Real.exp x - Real.log x > 2 * x :=
sorry

-- Statement 3: Minimum value of mn
theorem min_value_mn :
  ∀ m n : ℝ, (Real.log m) / m = n / (Real.exp n) → n < 0 → 
  m * n ≥ -1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_g_exp_log_inequality_min_value_mn_l954_95448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_max_m_value_a_plus_b_bound_l954_95465

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| + 2*|x + 1|

-- Theorem for the solution set of f(x) ≤ 5
theorem solution_set_f : 
  {x : ℝ | f x ≤ 5} = {x : ℝ | -3/2 ≤ x ∧ x ≤ 1} := by
  sorry

-- Theorem for the maximum value of m
theorem max_m_value :
  ∃ (M : ℝ), M = 2 ∧ ∀ m, (∃ x₀, f x₀ ≤ 5 + m - m^2) → m ≤ M := by
  sorry

-- Theorem for the inequality involving a and b
theorem a_plus_b_bound (a b : ℝ) (h : a^3 + b^3 = 2) :
  0 < a + b ∧ a + b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_max_m_value_a_plus_b_bound_l954_95465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sum_reciprocals_l954_95400

/-- Given a line 2ax - by + 2 = 0 passing through the center of the circle x² + y² + 2x - 4y + 1 = 0,
    where a > 0 and b > 0, the minimum value of 1/a + 1/b is 4. -/
theorem min_value_sum_reciprocals (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ x y : ℝ, x^2 + y^2 + 2*x - 4*y + 1 = 0 ∧ 2*a*x - b*y + 2 = 0) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    (∃ x y : ℝ, x^2 + y^2 + 2*x - 4*y + 1 = 0 ∧ 2*a'*x - b'*y + 2 = 0) →
    1/a + 1/b ≤ 1/a' + 1/b') →
  1/a + 1/b = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sum_reciprocals_l954_95400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_range_l954_95403

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x) ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.sin (x + Real.pi / 2)

theorem f_period_and_range :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧ T = Real.pi) ∧
  (Set.Icc 0 3 = {y | ∃ x ∈ Set.Icc 0 (2 * Real.pi / 3), f x = y}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_range_l954_95403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifty_two_million_scientific_notation_l954_95424

-- Define scientific notation
noncomputable def scientific_notation (a : ℝ) (n : ℤ) : ℝ := a * (10 : ℝ) ^ n

-- Define the condition for valid scientific notation
def valid_scientific_notation (a : ℝ) : Prop := 1 ≤ |a| ∧ |a| < 10

-- Theorem statement
theorem fifty_two_million_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), valid_scientific_notation a ∧ 
  scientific_notation a n = 52000000 ∧
  a = 5.2 ∧ n = 7 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifty_two_million_scientific_notation_l954_95424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_count_is_four_l954_95416

def number : Nat := 34698

def is_divisor (d : Nat) : Bool := number % d = 0

def divisor_count : Nat :=
  (List.range 9).filter (λ i => is_divisor (i + 1)) |>.length

theorem divisor_count_is_four : divisor_count = 4 := by
  -- Proof goes here
  sorry

#eval divisor_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_count_is_four_l954_95416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_value_l954_95493

theorem sin_plus_cos_value (α : ℝ) (h1 : α ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi)) 
  (h2 : Real.cos (α + 2017 * Real.pi / 2) = 3 / 5) : 
  Real.sin α + Real.cos α = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_value_l954_95493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l954_95464

/-- The area of a parallelogram with given base, slant height, and angle -/
theorem parallelogram_area (base : ℝ) (slant_height : ℝ) (angle : ℝ) :
  base = 22 →
  slant_height = 18 →
  angle = 35 * π / 180 →
  ∃ (area : ℝ), abs (area - base * (slant_height * Real.cos angle)) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l954_95464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_efficient_packing_correct_l954_95458

def most_efficient_packing (cd_quantities : List Nat) (box_sizes : List Nat) : Nat :=
  let gcd_quantities := cd_quantities.foldl Nat.gcd 0
  let valid_sizes := box_sizes.filter (· ∣ gcd_quantities)
  valid_sizes.foldl max 0

#eval most_efficient_packing [21, 18, 15, 12, 9] [3, 6, 9]

theorem most_efficient_packing_correct 
  (cd_quantities : List Nat) (box_sizes : List Nat) 
  (h_cd : cd_quantities = [21, 18, 15, 12, 9]) 
  (h_box : box_sizes = [3, 6, 9]) :
  most_efficient_packing cd_quantities box_sizes = 3 := by
  sorry -- Placeholder for the actual proof

#check most_efficient_packing_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_efficient_packing_correct_l954_95458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_vector_expression_l954_95422

/-- Given vectors a, b, c with specified norms, the maximum value of 
    ‖a - 3b‖² + ‖b - 3c‖² + ‖c - 3a‖² is 429. -/
theorem max_value_vector_expression {n : ℕ} (a b c : EuclideanSpace ℝ (Fin n)) 
    (ha : ‖a‖ = 2) (hb : ‖b‖ = 3) (hc : ‖c‖ = 4) :
    (∀ x y z : EuclideanSpace ℝ (Fin n), ‖x‖ = 2 → ‖y‖ = 3 → ‖z‖ = 4 → 
    ‖x - 3 • y‖^2 + ‖y - 3 • z‖^2 + ‖z - 3 • x‖^2 ≤ 429) ∧
    (∃ x y z : EuclideanSpace ℝ (Fin n), ‖x‖ = 2 ∧ ‖y‖ = 3 ∧ ‖z‖ = 4 ∧
    ‖x - 3 • y‖^2 + ‖y - 3 • z‖^2 + ‖z - 3 • x‖^2 = 429) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_vector_expression_l954_95422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l954_95445

-- Define the function f(x) = 2^x - 1/(x-1)
noncomputable def f (x : ℝ) : ℝ := 2^x - 1/(x-1)

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, x < 0 ∧ f x = a) → 0 < a ∧ a < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l954_95445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longer_candle_burn_time_l954_95499

/-- Represents a candle with its length -/
structure Candle where
  length : ℝ

/-- Represents the state of two burning candles at a given time -/
structure BurningCandles where
  longer : Candle
  shorter : Candle
  time : ℝ

/-- The burn rate of a candle (length units per minute) -/
noncomputable def burnRate (c : Candle) : ℝ := c.length / (c.length / 0.75)

/-- Calculate the remaining burn time for a candle -/
noncomputable def remainingBurnTime (c : Candle) : ℝ := c.length / burnRate c

/-- The main theorem to prove -/
theorem longer_candle_burn_time 
  (initial : BurningCandles) 
  (after18min : BurningCandles) :
  initial.time = 0 ∧
  initial.longer.length / initial.shorter.length = 21 / 16 ∧
  after18min.time = 18 ∧
  after18min.longer.length / after18min.shorter.length = 15 / 11 →
  remainingBurnTime after18min.longer = 150 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longer_candle_burn_time_l954_95499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goldfish_sold_is_110_l954_95439

/-- Calculates the number of goldfish sold in a week -/
def goldfish_sold (buy_price sell_price tank_cost : ℚ) (percent_short : ℚ) : ℕ :=
  let profit_per_fish := sell_price - buy_price
  let total_profit := tank_cost * (1 - percent_short)
  (total_profit / profit_per_fish).floor.toNat

/-- Proves that the number of goldfish sold in a week is 110 -/
theorem goldfish_sold_is_110 :
  goldfish_sold (25/100) (75/100) 100 (45/100) = 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goldfish_sold_is_110_l954_95439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_condition_l954_95451

/-- The function f(x) = x³ - 2ax + a has a local minimum in (0, 1) iff 0 < a < 3/2 -/
theorem local_minimum_condition (a : ℝ) : 
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ 
    (∀ y : ℝ, 0 < y ∧ y < 1 → 
      (x^3 - 2*a*x + a) ≤ (y^3 - 2*a*y + a))) ↔ 
  (0 < a ∧ a < 3/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_condition_l954_95451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_hyperbola_l954_95450

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  2 * x^2 - y^2 + 8 * x - 6 * y - 12 = 0

/-- The x-coordinate of the focus -/
noncomputable def focus_x : ℝ := -2 + Real.sqrt 19.5

/-- The y-coordinate of the focus -/
def focus_y : ℝ := -3

/-- Theorem stating that (focus_x, focus_y) is a focus of the hyperbola -/
theorem focus_of_hyperbola :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (x y : ℝ), hyperbola_equation x y ↔
    ((y - focus_y)^2 / a^2) - ((x - (-2))^2 / b^2) = 1 ∧
    a^2 - b^2 = 19.5 := by
  sorry

#check focus_of_hyperbola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_hyperbola_l954_95450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l954_95467

def arithmetic_sequence (s : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, s (n + 1) - s n = d

theorem sequence_property (a : ℕ → ℚ) 
  (h1 : a 3 = 2)
  (h2 : a 5 = 1)
  (h3 : arithmetic_sequence (λ n ↦ 1 / (1 + a n))) :
  a 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l954_95467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_1998_l954_95471

def sequence_sum (n : ℕ) : ℤ :=
  let groups := n / 3
  let remainder := n % 3
  let group_sum := (groups / 2) * (-1 : ℤ) + (groups - groups / 2) * 3
  let last_terms := match remainder with
    | 0 => 0
    | 1 => n
    | 2 => n + (n - 1)
    | _ => 0
  group_sum * 3 + last_terms

theorem sequence_sum_1998 :
  sequence_sum 1998 = 2665 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_1998_l954_95471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_q_value_l954_95473

noncomputable section

-- Define the polynomial Q(x)
def Q (x p q r : ℝ) : ℝ := x^3 + p*x^2 + q*x + r

-- Define the mean of zeros
def meanOfZeros (p : ℝ) : ℝ := -p / 3

-- Define the product of zeros
def productOfZeros (r : ℝ) : ℝ := -r

-- Define the sum of coefficients
def sumOfCoefficients (p q r : ℝ) : ℝ := 1 + p + q + r

-- Theorem statement
theorem polynomial_q_value 
  (p q r : ℝ) 
  (h1 : meanOfZeros p = productOfZeros r)
  (h2 : meanOfZeros p = sumOfCoefficients p q r)
  (h3 : r = 3) :
  q = -16 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_q_value_l954_95473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_cars_l954_95481

/-- Represents the position of a car at time t -/
noncomputable def position (v₀ v a : ℝ) (t : ℝ) : ℝ := v₀ + v * t + (1/2) * a * t^2

/-- The distance between two cars at time t -/
noncomputable def distance (t : ℝ) : ℝ :=
  position 100 0 20 t - position 0 50 0 t

theorem min_distance_between_cars :
  ∃ (t : ℝ), t > 0 ∧ ∀ (s : ℝ), s ≥ 0 → distance t ≤ distance s ∧ distance t = 37.5 := by
  sorry

#check min_distance_between_cars

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_cars_l954_95481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_trigonometric_system_l954_95417

theorem unique_solution_trigonometric_system :
  ∃! (x y z : ℝ), 
    Real.sin y - Real.sin x = x - y ∧
    Real.sin y - Real.sin z = z - y ∧
    x - y + z = Real.pi ∧
    x = Real.pi ∧ y = Real.pi ∧ z = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_trigonometric_system_l954_95417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_third_quadrant_tan_identity_sector_area_terminal_side_angles_l954_95447

noncomputable def quadrant (θ : Real) : Nat :=
  if Real.cos θ ≥ 0 && Real.sin θ ≥ 0 then 1
  else if Real.cos θ < 0 && Real.sin θ ≥ 0 then 2
  else if Real.cos θ < 0 && Real.sin θ < 0 then 3
  else 4

-- Statement 1
theorem not_third_quadrant : quadrant (-7 * Real.pi / 6) ≠ 3 := by sorry

-- Statement 2
theorem tan_identity (α : Real) (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 := by sorry

-- Statement 3
theorem sector_area (θ : Real) (l : Real) (h1 : θ = Real.pi / 3) (h2 : l = Real.pi) :
  let r := l / θ
  (1/2) * r^2 * θ = (3 * Real.pi) / 2 := by sorry

-- Statement 4
theorem terminal_side_angles (m : Real) (h : m > 0) :
  {α : Real | ∃ k : Int, α = Real.pi / 4 + 2 * Real.pi * ↑k} = 
  {α : Real | ∃ t : Real, t > 0 ∧ (t * Real.cos α, t * Real.sin α) = (m, m)} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_third_quadrant_tan_identity_sector_area_terminal_side_angles_l954_95447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_convincing_statement_l954_95486

-- Define a Transylvanian
structure Transylvanian where
  name : String

-- Define a statement
structure Statement where
  content : Prop
  speaker : Transylvanian

-- Define what it means for a statement to be indeterminate
def is_indeterminate (p : Prop) : Prop :=
  ¬(p ∨ ¬p)

-- Define Dracula's alive state
def dracula_alive : Prop := sorry

-- Theorem statement
theorem exists_convincing_statement :
  ∃ (t : Transylvanian) (s : Statement),
    s.speaker = t ∧
    (s.content → dracula_alive) ∧
    is_indeterminate s.content := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_convincing_statement_l954_95486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_revolution_surface_area_l954_95430

/-- The surface area of a solid of revolution formed by rotating an isosceles triangle -/
noncomputable def surfaceAreaIsoscelesRevolution (a : ℝ) (α : ℝ) : ℝ :=
  8 * Real.pi * a^2 * Real.sin α * Real.cos (α/2) * 
  Real.cos (Real.pi/6 + α/2) * Real.cos (Real.pi/6 - α/2)

/-- Theorem: The surface area of the solid of revolution formed by rotating an isosceles triangle
    with lateral side a and base angle α around a line parallel to the angle bisector of α
    and passing through the vertex opposite the base is equal to 
    8πa²sinα cos(α/2) cos(π/6 + α/2) cos(π/6 - α/2) -/
theorem isosceles_revolution_surface_area (a α : ℝ) 
  (ha : a > 0) (hα : 0 < α ∧ α < Real.pi/2) :
  surfaceAreaIsoscelesRevolution a α = 
    8 * Real.pi * a^2 * Real.sin α * Real.cos (α/2) * 
    Real.cos (Real.pi/6 + α/2) * Real.cos (Real.pi/6 - α/2) := by
  -- The proof is skipped for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_revolution_surface_area_l954_95430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l954_95474

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if c - a * cos(B) = (2a - b) * cos(A), then the triangle is either isosceles or right-angled -/
theorem triangle_shape (a b c A B C : ℝ) 
    (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c)
    (h_angles : A + B + C = π) 
    (h_condition : c - a * Real.cos B = (2*a - b) * Real.cos A) :
    (A = B) ∨ (A = π/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l954_95474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_functions_with_nonnegative_range_l954_95432

-- Define the functions
noncomputable def f1 (x : ℝ) : ℝ := |x|
noncomputable def f2 (x : ℝ) : ℝ := x^3
noncomputable def f3 (x : ℝ) : ℝ := 2^(|x|)
noncomputable def f4 (x : ℝ) : ℝ := x^2 + |x|

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the range of a function
def range (f : ℝ → ℝ) : Set ℝ := {y | ∃ x, f x = y}

-- State the theorem
theorem even_functions_with_nonnegative_range :
  (is_even f1 ∧ range f1 = Set.Ici 0) ∧
  (is_even f4 ∧ range f4 = Set.Ici 0) ∧
  ¬(is_even f2 ∧ range f2 = Set.Ici 0) ∧
  ¬(is_even f3 ∧ range f3 = Set.Ici 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_functions_with_nonnegative_range_l954_95432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_l954_95426

/-- Regular triangular pyramid -/
structure RegularTriangularPyramid where
  a : ℝ  -- Side length of the base triangle
  β : ℝ  -- Angle between a lateral face and the base plane

/-- Volume of a regular triangular pyramid -/
noncomputable def volume (pyramid : RegularTriangularPyramid) : ℝ :=
  (pyramid.a^3 * Real.tan pyramid.β) / 24

theorem volume_formula (pyramid : RegularTriangularPyramid) :
  volume pyramid = (pyramid.a^3 * Real.tan pyramid.β) / 24 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_l954_95426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_numbers_dividing_all_relatives_l954_95479

def is_relative (ab n : ℕ) : Prop :=
  let a := ab / 10
  let b := ab % 10
  n % 10 = b ∧ (Nat.digits 10 (n / 10)).sum = a ∧ ∀ d ∈ (Nat.digits 10 (n / 10)), d ≠ 0

def divides_all_relatives (ab : ℕ) : Prop :=
  ∀ n : ℕ, is_relative ab n → ab ∣ n

theorem two_digit_numbers_dividing_all_relatives :
  {ab : ℕ | 10 ≤ ab ∧ ab < 100 ∧ divides_all_relatives ab} = {15, 18, 30, 45, 90} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_numbers_dividing_all_relatives_l954_95479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meet_time_theorem_l954_95476

-- Define the track length in meters
noncomputable def track_length : ℝ := 200

-- Define the speeds in km/h
noncomputable def speed_a : ℝ := 36
noncomputable def speed_b : ℝ := 72

-- Convert km/h to m/s
noncomputable def mps_conversion (kmph : ℝ) : ℝ := kmph * (1000 / 3600)

-- Calculate the speeds in m/s
noncomputable def speed_a_mps : ℝ := mps_conversion speed_a
noncomputable def speed_b_mps : ℝ := mps_conversion speed_b

-- Calculate the time for one lap for each object
noncomputable def time_lap_a : ℝ := track_length / speed_a_mps
noncomputable def time_lap_b : ℝ := track_length / speed_b_mps

-- Define the theorem
theorem meet_time_theorem : 
  ∃ (t : ℝ), t > 0 ∧ 
  (∃ (n : ℤ), t = n * time_lap_a) ∧ 
  (∃ (m : ℤ), t = m * time_lap_b) ∧ 
  t = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meet_time_theorem_l954_95476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_octagon_area_ratio_l954_95443

/-- Regular octagon with side length s -/
structure RegularOctagon :=
  (s : ℝ)
  (s_pos : 0 < s)

/-- Area of a regular octagon -/
noncomputable def area (o : RegularOctagon) : ℝ := 2 * o.s^2 * (1 + Real.sqrt 2)

/-- Smaller octagon formed by connecting midpoints -/
noncomputable def midpointOctagon (o : RegularOctagon) : RegularOctagon :=
  { s := o.s / Real.sqrt 2
  , s_pos := by 
      have h : 0 < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
      exact div_pos o.s_pos h }

/-- Theorem: Area of midpoint octagon is 1/4 of original octagon -/
theorem midpoint_octagon_area_ratio (o : RegularOctagon) :
  area (midpointOctagon o) = (1 / 4) * area o := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_octagon_area_ratio_l954_95443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_circumference_l954_95452

/-- Represents a right circular cylinder -/
structure Cylinder where
  height : ℝ
  circumference : ℝ

/-- Calculates the volume of a cylinder -/
noncomputable def cylinderVolume (c : Cylinder) : ℝ :=
  c.circumference ^ 2 * c.height / (4 * Real.pi)

theorem tank_circumference (tankA tankB : Cylinder)
    (hA_height : tankA.height = 10)
    (hB_height : tankB.height = 7)
    (hB_circ : tankB.circumference = 10)
    (hVolume : cylinderVolume tankA = 0.7 * cylinderVolume tankB) :
    tankA.circumference = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_circumference_l954_95452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l954_95449

/-- An ellipse Γ with equation x^2/a^2 + y^2 = 1 and a > 1 -/
structure Ellipse :=
  (a : ℝ)
  (h_a : a > 1)

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (γ : Ellipse) : ℝ :=
  Real.sqrt (1 - 1 / γ.a^2)

/-- A circle centered at (0,1) -/
structure Circle :=
  (r : ℝ)
  (h_r : r > 0)

/-- The number of intersection points between an ellipse and a circle -/
def num_intersections (γ : Ellipse) (c : Circle) : ℕ := sorry

/-- The main theorem -/
theorem eccentricity_range (γ : Ellipse) :
  (∀ c : Circle, num_intersections γ c ≤ 3) →
  0 < eccentricity γ ∧ eccentricity γ ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l954_95449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_pen_is_14_l954_95407

-- Define the range of pen numbers
def PenNumbers : Finset ℕ := Finset.range 20

-- Define the random number table as a list of natural numbers
def RandomTable : List ℕ := [9, 5, 2, 2, 6, 0, 0, 0, 4, 9, 8, 4, 0, 1, 2, 8, 6, 6, 1, 7, 5, 1, 6, 8, 3, 9, 6, 8, 2, 9, 2, 7, 4, 3, 7, 7, 2, 3, 6, 6, 2, 7, 0, 9, 6, 6, 2, 3]

-- Function to check if a number is valid (within the pen number range)
def isValidNumber (n : ℕ) : Bool := n ∈ PenNumbers

-- Function to select valid numbers from the random table
def selectValidNumbers (table : List ℕ) : List ℕ :=
  table.filter isValidNumber

-- Theorem stating that the 6th selected pen number is 14
theorem sixth_pen_is_14 : 
  (selectValidNumbers RandomTable).get? 5 = some 14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_pen_is_14_l954_95407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_equation_C_intersection_range_m_l954_95440

noncomputable section

-- Define the curve C
def curve_C (t : ℝ) : ℝ × ℝ :=
  (2 * Real.sqrt t / (1 + t), 2 / (1 + t))

-- Define the line l
def line_l (m : ℝ) (ρ θ : ℝ) : Prop :=
  Real.sqrt 2 * ρ * Real.cos (θ + Real.pi / 4) + m = 0

-- Statement for the Cartesian equation of curve C
theorem cartesian_equation_C :
  ∀ x y : ℝ, x ≥ 0 → y ≠ 0 →
  (∃ t : ℝ, curve_C t = (x, y)) ↔ x^2 + y^2 - 2*y = 0 := by
  sorry

-- Statement for the range of m
theorem intersection_range_m :
  ∀ m : ℝ, (∃ x y : ℝ, x ≥ 0 ∧ y ≠ 0 ∧
    x^2 + y^2 - 2*y = 0 ∧
    (∃ ρ θ : ℝ, x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ line_l m ρ θ)) ↔
  m ≥ -Real.sqrt 2 + 1 ∧ m ≤ 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_equation_C_intersection_range_m_l954_95440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joshes_friends_l954_95480

theorem joshes_friends (n : ℕ) : 
  (∀ (person : ℕ), person ≤ n + 1 → (5 : ℝ) = 5) →  -- Everyone puts $5
  (0.8 * (5 * (n + 1 : ℝ)) = 5 * (n + 1 : ℝ) - 8) →   -- First place gets 80%, rest is 8
  (4 = 0.1 * (5 * (n + 1 : ℝ))) →                 -- Third place gets $4, which is half of the remaining 20%
  n = 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joshes_friends_l954_95480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_2_3_4_is_obtuse_right_triangle_3_4_x_special_right_triangle_l954_95423

-- Define triangle types
inductive TriangleType
  | Acute
  | Right
  | Obtuse

-- Function to determine triangle type
noncomputable def triangleType (a b c : ℝ) : TriangleType :=
  let max := max a (max b c)
  if max^2 = a^2 + b^2 + c^2 - max^2 then TriangleType.Right
  else if max^2 > a^2 + b^2 + c^2 - max^2 then TriangleType.Obtuse
  else TriangleType.Acute

-- Theorem 1: Triangle with sides 2, 3, 4 is obtuse
theorem triangle_2_3_4_is_obtuse :
  triangleType 2 3 4 = TriangleType.Obtuse := by sorry

-- Theorem 2: In a right triangle with sides 3, 4, and x, x = 5 or √7
theorem right_triangle_3_4_x (x : ℝ) :
  triangleType 3 4 x = TriangleType.Right → x = 5 ∨ x = Real.sqrt 7 := by sorry

-- Theorem 3: Triangle with sides (m²-n²)/2, mn, (m²+n²)/2 is right
theorem special_right_triangle (m n : ℝ) :
  triangleType ((m^2 - n^2) / 2) (m * n) ((m^2 + n^2) / 2) = TriangleType.Right := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_2_3_4_is_obtuse_right_triangle_3_4_x_special_right_triangle_l954_95423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_complex_condition_l954_95494

/-- Given a complex number Z = x + yi (x, y ∈ ℝ) satisfying |Z - 4i| = |Z + 2|,
    the minimum value of 2^x + 4^y is 4√2. -/
theorem min_value_of_complex_condition (x y : ℝ) :
  let Z : ℂ := Complex.mk x y
  Complex.abs (Z - 4*Complex.I) = Complex.abs (Z + 2) →
  (∀ a b : ℝ, let W : ℂ := Complex.mk a b
    Complex.abs (W - 4*Complex.I) = Complex.abs (W + 2) →
    (2:ℝ)^x + (4:ℝ)^y ≤ (2:ℝ)^a + (4:ℝ)^b) →
  (2:ℝ)^x + (4:ℝ)^y = 4 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_complex_condition_l954_95494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_numbers_average_l954_95453

theorem middle_numbers_average (a b c d : ℚ) : 
  a < b ∧ b < c ∧ c < d ∧  -- Four different positive rational numbers
  (a + b + c + d) / 4 = 7 ∧  -- Average is 7
  d - a = 17 →  -- Maximum possible difference (19 - 2 = 17)
  (b + c) / 2 = (7 : ℚ) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_numbers_average_l954_95453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_bipartite_l954_95487

/-- A connected graph with n vertices and m edges -/
structure ConnectedGraph (n m : ℕ) where
  (n_ge_3 : n ≥ 3)
  (is_connected : True)  -- We assume the graph is connected without formally defining connectedness

/-- The property that a graph can be made bipartite by deleting at most k(m - ⌊n/2⌋) edges -/
def CanMakeBipartite (k : ℝ) (n m : ℕ) : Prop :=
  ∀ (G : ConnectedGraph n m), ∃ (deleted_edges : ℕ), 
    (↑deleted_edges : ℝ) ≤ k * ((m : ℝ) - ↑(n / 2)) ∧ 
    True  -- Represents that the resulting graph is bipartite

/-- The main theorem stating that 1/2 is the smallest k that satisfies the condition -/
theorem smallest_k_for_bipartite :
  (∀ (n m : ℕ), CanMakeBipartite (1/2) n m) ∧
  (∀ (k : ℝ), k < 1/2 → ∃ (n m : ℕ), ¬CanMakeBipartite k n m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_bipartite_l954_95487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_theorem_l954_95401

-- Define the quadrilateral ABCD
structure Quadrilateral (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
  (A B C D : V)

-- Define the properties of the quadrilateral
def is_convex_quadrilateral {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (q : Quadrilateral V) : Prop := sorry

noncomputable def angle {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (A B C : V) : ℝ := sorry

theorem quadrilateral_theorem 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (q : Quadrilateral V) (E : V)
  (h_convex : is_convex_quadrilateral q)
  (h_AB_BC : ‖q.A - q.B‖ > ‖q.B - q.C‖)
  (h_CD_DA : ‖q.C - q.D‖ = ‖q.D - q.A‖)
  (h_angle_ABD_DBC : angle q.A q.B q.D = angle q.D q.B q.C)
  (h_E_on_AB : ∃ t : ℝ, E = q.A + t • (q.B - q.A))
  (h_angle_DEB : angle q.D E q.B = π / 2) :
  ‖q.A - E‖ = (‖q.A - q.B‖ - ‖q.B - q.C‖) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_theorem_l954_95401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridget_erasers_given_l954_95435

/-- The number of erasers Bridget gave to Peter -/
def erasers_given : ℕ := 3

/-- Peter's initial number of erasers -/
def initial_erasers : ℕ := 8

/-- Peter's final number of erasers -/
def final_erasers : ℕ := 11

/-- Theorem stating that the number of erasers Bridget gave to Peter
    is equal to the difference between Peter's final and initial number of erasers -/
theorem bridget_erasers_given :
  erasers_given = final_erasers - initial_erasers := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridget_erasers_given_l954_95435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_fraction_decomposition_product_l954_95456

noncomputable def f (x : ℝ) : ℝ := (x^2 - 12) / (x^3 - 3*x^2 - 4*x + 12)

noncomputable def g (A B C x : ℝ) : ℝ := A / (x - 2) + B / (x + 2) + C / (x - 3)

theorem partial_fraction_decomposition_product :
  ∃ (A B C : ℝ), (∀ x, f x = g A B C x) ∧ (A * B * C = -12/25) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_fraction_decomposition_product_l954_95456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_relationship_in_plane_l954_95490

/-- A 2D point -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A 2D line -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point is on a line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Two lines are intersecting if they have exactly one point in common -/
def Intersecting (l₁ l₂ : Line2D) : Prop :=
  ∃ p : Point2D, pointOnLine p l₁ ∧ pointOnLine p l₂ ∧
    ∀ q : Point2D, pointOnLine q l₁ ∧ pointOnLine q l₂ → q = p

/-- Two lines are parallel if they do not intersect -/
def Parallel (l₁ l₂ : Line2D) : Prop :=
  ∀ p : Point2D, ¬(pointOnLine p l₁ ∧ pointOnLine p l₂)

/-- Two straight lines in a plane are either intersecting or parallel -/
theorem line_relationship_in_plane (l₁ l₂ : Line2D) (h : l₁ ≠ l₂) :
  (Intersecting l₁ l₂) ∨ (Parallel l₁ l₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_relationship_in_plane_l954_95490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l954_95431

-- Define the circle C
def circle_center : ℝ × ℝ := (1, 0)
def circle_radius : ℝ := 2

-- Define the line l
def line_equation (x y : ℝ) : Prop := y = x

-- Theorem statement
theorem intersection_chord_length :
  let C := {p : ℝ × ℝ | (p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2 = circle_radius^2}
  let l := {p : ℝ × ℝ | line_equation p.1 p.2}
  let intersection := C ∩ l
  ∃ A B, A ∈ intersection ∧ B ∈ intersection ∧ ‖A - B‖ = Real.sqrt 14 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l954_95431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_tangent_properties_l954_95404

-- Define the ellipse C1
def C1 (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the parabola C2
def C2 (x y : ℝ) : Prop := y^2 = 4*x

-- Define a point P on C2 (not the origin)
def P_on_C2 (x y : ℝ) : Prop := C2 x y ∧ (x ≠ 0 ∨ y ≠ 0)

-- Define the tangent line l at point P
def tangent_line (P_x P_y x y : ℝ) : Prop :=
  ∃ (k : ℝ), y = k*(x - P_x) + P_y ∧ k = 1/(2*P_y)

-- Define the intersection points A and B
def intersection_points (A_x A_y B_x B_y : ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  C1 A_x A_y ∧ C1 B_x B_y ∧ l A_x A_y ∧ l B_x B_y ∧ (A_x ≠ B_x ∨ A_y ≠ B_y)

-- Define the x-intercept of line l
noncomputable def x_intercept (P_x P_y : ℝ) : ℝ := -P_x^2

-- Define the area of triangle AOB
noncomputable def triangle_area (A_x A_y B_x B_y : ℝ) : ℝ :=
  abs ((A_x * B_y - B_x * A_y) / 2)

theorem ellipse_parabola_tangent_properties :
  ∀ (P_x P_y : ℝ),
    P_on_C2 P_x P_y →
    (∀ (x y : ℝ), tangent_line P_x P_y x y → 
      (-4 < x_intercept P_x P_y ∧ x_intercept P_x P_y < 0)) ∧
    (∃ (A_x A_y B_x B_y : ℝ),
      intersection_points A_x A_y B_x B_y (tangent_line P_x P_y) ∧
      (∀ (X_x X_y Y_x Y_y : ℝ),
        intersection_points X_x X_y Y_x Y_y (tangent_line P_x P_y) →
        triangle_area X_x X_y Y_x Y_y ≤ Real.sqrt 3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_tangent_properties_l954_95404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bubble_sort_rounds_needed_l954_95462

def initial_sequence : List ℕ := [37, 21, 3, 56, 9, 7]
def target_sequence : List ℕ := [3, 9, 7, 21, 37, 56]

def bubble_sort_step (lst : List ℕ) : List ℕ :=
  match lst with
  | [] => []
  | [x] => [x]
  | x :: y :: rest =>
    if x > y then
      y :: bubble_sort_step (x :: rest)
    else
      x :: bubble_sort_step (y :: rest)

def bubble_sort_rounds (lst : List ℕ) : ℕ → List ℕ
  | 0 => lst
  | n + 1 => bubble_sort_rounds (bubble_sort_step lst) n

theorem bubble_sort_rounds_needed :
  ∃ (n : ℕ), n = 2 ∧ bubble_sort_rounds initial_sequence n = target_sequence :=
by
  use 2
  apply And.intro
  · rfl
  · sorry  -- The actual proof would go here

#eval bubble_sort_rounds initial_sequence 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bubble_sort_rounds_needed_l954_95462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_odd_cycles_after_removal_l954_95460

/-- A graph with 2000 vertices where each vertex has exactly 3 edges -/
structure CityGraph where
  vertices : Finset ℕ
  edges : Finset (ℕ × ℕ)
  vertex_count : vertices.card = 2000
  edge_degree : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 3

/-- A subset of edges to be removed -/
def RemovedEdges (g : CityGraph) := { e : Finset (ℕ × ℕ) // e ⊆ g.edges ∧ e.card = 1000 }

/-- An odd cycle in a graph -/
def OddCycle (g : CityGraph) (cycle : List ℕ) : Prop :=
  cycle.length % 2 = 1 ∧
  cycle.length > 0 ∧
  (∀ i, i < cycle.length → (cycle[i]!, cycle[(i+1) % cycle.length]!) ∈ g.edges)

/-- The graph after removing edges -/
def RemainingGraph (g : CityGraph) (removed : RemovedEdges g) : CityGraph :=
  { vertices := g.vertices
    edges := g.edges \ removed.val
    vertex_count := g.vertex_count
    edge_degree := sorry }

theorem no_odd_cycles_after_removal (g : CityGraph) :
  ∃ (removed : RemovedEdges g), ∀ cycle, ¬OddCycle (RemainingGraph g removed) cycle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_odd_cycles_after_removal_l954_95460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_union_inequality_l954_95484

theorem subset_union_inequality (t : ℕ) (A : Fin t → Set α) :
  ∃ (S : Finset (Fin t)), 
    S.card = ⌊Real.sqrt (t : ℝ)⌋ ∧ 
    ∀ (x y z : Fin t), x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z → 
      A x ∪ A y ≠ A z :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_union_inequality_l954_95484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_one_equals_three_l954_95489

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 3 then x^2 - 2*x else 2*x + 1

-- State the theorem
theorem f_of_f_one_equals_three : f (f 1) = 3 := by
  -- Evaluate f(1)
  have h1 : f 1 = 3 := by
    -- Simplify the definition of f for x < 3
    simp [f]
    norm_num

  -- Evaluate f(f(1)) = f(3)
  have h2 : f 3 = 3 := by
    -- Simplify the definition of f for x ≥ 3
    simp [f]
    norm_num

  -- Combine the results
  rw [h1, h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_one_equals_three_l954_95489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l954_95457

noncomputable def f (x : ℝ) := (Real.log x) / Real.sqrt (1 - x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | 0 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l954_95457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_point_three_repeating_equals_twenty_two_thirds_l954_95498

/-- Represents a repeating decimal where the fractional part is an infinite repetition of a single digit -/
def repeatingDecimal (whole : ℕ) (repeatDigit : ℕ) : ℚ :=
  whole + (repeatDigit : ℚ) / 9

/-- The theorem states that 7.3333... (where 3 repeats infinitely) is equal to 22/3 -/
theorem seven_point_three_repeating_equals_twenty_two_thirds :
  repeatingDecimal 7 3 = 22 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_point_three_repeating_equals_twenty_two_thirds_l954_95498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_growth_l954_95428

-- Define the function f
variable (f : ℝ → ℝ)

-- Define f' as the derivative of f
variable (f' : ℝ → ℝ)

-- State the theorem
theorem function_growth (h : ∀ x, HasDerivAt f (f' x) x) 
  (h_growth : ∀ x, f' x > f x) : f 1 > Real.exp 1 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_growth_l954_95428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_circle_properties_l954_95405

/-- Parabola struct representing y² = 2px -/
structure Parabola where
  p : ℝ
  h : 0 < p ∧ p ≤ 8

/-- Circle struct representing (x-3)² + y² = 1 -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  h : center = (3, 0) ∧ radius = 1

/-- Tangent line from focus of parabola to circle -/
noncomputable def tangentLine (Ω : Parabola) (C : Circle) : ℝ := sorry

/-- Intersection points of a tangent to circle with parabola -/
noncomputable def intersectionPoints (Ω : Parabola) (C : Circle) : Set (ℝ × ℝ) := sorry

theorem parabola_and_circle_properties (Ω : Parabola) (C : Circle) :
  tangentLine Ω C = Real.sqrt 3 →
  (∀ x y, y^2 = 2 * Ω.p * x ↔ y^2 = 4 * x) ∧
  (∃ m : ℝ, (∀ A B, A ∈ intersectionPoints Ω C → B ∈ intersectionPoints Ω C →
    |A.1 + 1| * |B.1 + 1| ≥ m) ∧
    (∃ A' B', A' ∈ intersectionPoints Ω C ∧ B' ∈ intersectionPoints Ω C ∧ 
    |A'.1 + 1| * |B'.1 + 1| = m)) ∧
  m = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_circle_properties_l954_95405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_l954_95442

theorem necessary_not_sufficient : 
  (∀ x : ℝ, (|x - 1| < 1 → x * (x - 4) < 0)) ∧ 
  (∃ x : ℝ, x * (x - 4) < 0 ∧ |x - 1| ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_l954_95442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_video_games_spending_is_12_5_l954_95491

noncomputable def total_allowance : ℝ := 50

noncomputable def books_fraction : ℝ := 1/4
noncomputable def snacks_fraction : ℝ := 1/5
noncomputable def crafts_fraction : ℝ := 3/10

noncomputable def video_games_spending (total : ℝ) (books : ℝ) (snacks : ℝ) (crafts : ℝ) : ℝ :=
  total - (books * total + snacks * total + crafts * total)

theorem video_games_spending_is_12_5 : 
  video_games_spending total_allowance books_fraction snacks_fraction crafts_fraction = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_video_games_spending_is_12_5_l954_95491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_cost_july_is_5_1_l954_95468

/-- The cost of a mixture of milk powder and coffee in July -/
noncomputable def mixture_cost_july (june_cost : ℝ) : ℝ :=
  let coffee_july := 3 * june_cost
  let milk_july := 0.4
  let mixture_weight := 3
  (mixture_weight / 2) * (coffee_july + milk_july)

/-- Theorem stating the cost of the mixture in July -/
theorem mixture_cost_july_is_5_1 :
  ∃ (june_cost : ℝ),
    june_cost > 0 ∧
    0.4 * june_cost = 0.4 ∧
    mixture_cost_july june_cost = 5.1 := by
  -- We'll use 1 as the june_cost
  use 1
  constructor
  · -- Prove june_cost > 0
    linarith
  constructor
  · -- Prove 0.4 * june_cost = 0.4
    norm_num
  · -- Prove mixture_cost_july june_cost = 5.1
    unfold mixture_cost_july
    norm_num
    

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_cost_july_is_5_1_l954_95468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_f₁_min_value_f₂_l954_95406

-- Part 1
noncomputable def f₁ (x : ℝ) : ℝ := -1/3 * x^3 - 1/2 * x^2 + 6*x

theorem extreme_values_f₁ :
  (∃ x, f₁ x = 22/3 ∧ ∀ y, f₁ y ≤ f₁ x) ∧
  (∃ x, f₁ x = -27/2 ∧ ∀ y, f₁ y ≥ f₁ x) := by
  sorry

-- Part 2
noncomputable def f₂ (m : ℝ) (x : ℝ) : ℝ := 1/3 * x^3 - 1/2 * x^2 + 2*m*x

theorem min_value_f₂ (m : ℝ) (h₁ : -2 < m) (h₂ : m < 0)
  (h₃ : ∃ x ∈ Set.Icc 1 4, f₂ m x = 16/3 ∧ ∀ y ∈ Set.Icc 1 4, f₂ m y ≤ f₂ m x) :
  ∃ x ∈ Set.Icc 1 4, f₂ m x = -10/3 ∧ ∀ y ∈ Set.Icc 1 4, f₂ m y ≥ f₂ m x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_f₁_min_value_f₂_l954_95406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_arrays_48_chairs_l954_95421

/-- The number of different rectangular arrays that can be formed with a given number of chairs. -/
def num_arrays (total_chairs : ℕ) : ℕ :=
  2 * (Finset.filter (λ x ↦ x ≥ 2 ∧ total_chairs / x ≥ 2 ∧ total_chairs % x = 0)
        (Finset.range (total_chairs + 1))).card

/-- Theorem stating that the number of different rectangular arrays with 48 chairs is 8. -/
theorem num_arrays_48_chairs :
  num_arrays 48 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_arrays_48_chairs_l954_95421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_prob_smaller_yellow_prob_three_fifths_add_25_red_prob_two_thirds_equal_prob_after_addition_l954_95496

/-- A bag containing red and yellow balls -/
structure Bag where
  total : ℕ
  red : ℕ
  yellow : ℕ
  h_total : total = red + yellow

/-- The probability of drawing a ball of a specific color -/
def prob (b : Bag) (color : ℕ) : ℚ := color / b.total

/-- The initial bag configuration -/
def initial_bag : Bag := {
  total := 20,
  red := 5,
  yellow := 15,
  h_total := by rfl
}

theorem red_prob_smaller (b : Bag) (h : b.red < b.yellow) : 
  prob b b.red < prob b b.yellow := by sorry

theorem yellow_prob_three_fifths : 
  prob initial_bag initial_bag.yellow = 3/5 := by sorry

theorem add_25_red_prob_two_thirds (b : Bag) :
  let new_bag : Bag := {
    total := b.total + 25,
    red := b.red + 25,
    yellow := b.yellow,
    h_total := by simp [b.h_total]; ring
  }
  prob new_bag new_bag.red = 2/3 := by sorry

theorem equal_prob_after_addition (b : Bag) :
  let new_bag : Bag := {
    total := b.total + 20,
    red := b.red + 15,
    yellow := b.yellow + 5,
    h_total := by simp [b.h_total]; ring
  }
  prob new_bag new_bag.red = prob new_bag new_bag.yellow := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_prob_smaller_yellow_prob_three_fifths_add_25_red_prob_two_thirds_equal_prob_after_addition_l954_95496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_1000_equals_negative_one_l954_95495

def sequence_a : ℕ → ℤ
  | 0 => 1  -- Adding this case for n = 0
  | 1 => 1
  | 2 => 5
  | n + 3 => sequence_a (n + 2) - sequence_a (n + 1)

theorem a_1000_equals_negative_one :
  sequence_a 1000 = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_1000_equals_negative_one_l954_95495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oranges_at_opening_l954_95461

/-- Represents the state of the grocery store inventory --/
structure GroceryStore where
  initialLemons : ℕ
  finalLemons : ℕ
  finalOranges : ℕ
  ratioDecrease : ℚ

/-- Calculates the number of oranges at store opening --/
def initialOranges (store : GroceryStore) : ℕ :=
  sorry

/-- Theorem stating the number of oranges at store opening --/
theorem oranges_at_opening (store : GroceryStore) 
  (h1 : store.initialLemons = 50)
  (h2 : store.finalLemons = 20)
  (h3 : store.finalOranges = 40)
  (h4 : store.ratioDecrease = 2/5) : 
  initialOranges store = 60 := by
  sorry

#check oranges_at_opening

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oranges_at_opening_l954_95461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l954_95485

/-- Given vectors a and b, if a is perpendicular to λa + b, then λ = -2 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h : a = (2, 1) ∧ b = (3, 4)) :
  (∃ l : ℝ, a.1 * (l * a.1 + b.1) + a.2 * (l * a.2 + b.2) = 0) →
  (∃ l : ℝ, l = -2 ∧ a.1 * (l * a.1 + b.1) + a.2 * (l * a.2 + b.2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l954_95485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_ming_expense_increase_l954_95419

/-- Calculates the percentage increase in total expenses --/
noncomputable def percentage_increase (food_last : ℝ) (other_last : ℝ) (food_rate : ℝ) (other_rate : ℝ) : ℝ :=
  let food_this := food_last * (1 + food_rate)
  let other_this := other_last * (1 + other_rate)
  let total_last := food_last + other_last
  let total_this := food_this + other_this
  (total_this - total_last) / total_last * 100

/-- Theorem stating that the percentage increase in Xiao Ming's family's total expenses is 16.4% --/
theorem xiao_ming_expense_increase : 
  percentage_increase 360 640 0.1 0.2 = 16.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_ming_expense_increase_l954_95419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_general_term_l954_95433

/-- The sequence a_n defined recursively -/
def a : ℕ → ℤ
  | 0 => 1  -- Added case for n = 0
  | 1 => 1
  | n + 2 => 2 * a (n + 1) + (n + 2) - 2

/-- The proposed general term for a_n -/
def a_general (n : ℕ) : ℤ := 2^n - n

/-- Theorem stating that a_n equals the proposed general term for all n ≥ 1 -/
theorem a_equals_general_term : ∀ n : ℕ, n ≥ 1 → a n = a_general n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_general_term_l954_95433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_resolvable_debt_l954_95420

theorem smallest_resolvable_debt (apple_value orange_value : ℕ) 
  (h_apple : apple_value = 240) (h_orange : orange_value = 180) : ℕ := by
  let D := 60
  have h1 : ∀ d : ℕ, d > 0 → d < D → ¬∃ (a o : ℤ), d = apple_value * a + orange_value * o := by
    sorry
  have h2 : ∃ (a o : ℤ), D = apple_value * a + orange_value * o := by
    sorry
  exact D

#check smallest_resolvable_debt

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_resolvable_debt_l954_95420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_with_equal_intercepts_l954_95477

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 2

-- Define a line with slope m and y-intercept b
def my_line (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

-- Define tangency condition
def is_tangent (m b : ℝ) : Prop :=
  ∃ x y, my_circle x y ∧ my_line m b x y ∧
  ∀ x' y', my_circle x' y' ∧ my_line m b x' y' → (x' = x ∧ y' = y)

-- Define equal intercepts condition
def has_equal_intercepts (m b : ℝ) : Prop :=
  ∃ a, a ≠ 0 ∧ my_line m b a 0 ∧ my_line m b 0 a

-- Main theorem
theorem tangent_lines_with_equal_intercepts :
  ∀ m b : ℝ,
    (is_tangent m b ∧ has_equal_intercepts m b) ↔
    ((m = 1 ∧ b = 0) ∨ (m = -1 ∧ b = 0) ∨ (m = -1 ∧ b = 4)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_with_equal_intercepts_l954_95477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l954_95482

/-- The function f(x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x - Real.sqrt 3 * (Real.cos x)^2

/-- The function g(x) obtained by shifting f(x) left by π/4 units -/
noncomputable def g (x : ℝ) : ℝ := Real.sin (2*x + Real.pi/6) - Real.sqrt 3 / 2

/-- Theorem stating the maximum value of a -/
theorem max_a_value : 
  ∃ (a_max : ℝ), a_max = Real.sqrt 3 ∧ 
  (∀ (a : ℝ), a > 0 → 
    (∀ (x : ℝ), x ∈ Set.Icc (Real.pi/6) (Real.pi/3) → 
      a * f x + g x ≥ Real.sqrt (a^2 + 1) / 2 - Real.sqrt 3 / 2 * (a + 1)) → 
    a ≤ a_max) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (Real.pi/6) (Real.pi/3) → 
    a_max * f x + g x ≥ Real.sqrt (a_max^2 + 1) / 2 - Real.sqrt 3 / 2 * (a_max + 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l954_95482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l954_95436

def M : Set ℝ := {x | x^2 + x - 6 < 0}
def N : Set ℝ := {x | |x - 1| ≤ 2}

theorem intersection_M_N : M ∩ N = Set.Ioc (-1) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l954_95436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ferris_wheel_time_l954_95427

theorem ferris_wheel_time (radius : ℝ) (revolution_time : ℝ) (height : ℝ) : 
  radius = 30 →
  revolution_time = 120 →
  height = 15 →
  let period := 2 * Real.pi / revolution_time
  let amplitude := radius
  let vertical_shift := radius
  let height_function := λ (t : ℝ) ↦ amplitude * Real.cos (period * t) + vertical_shift
  ∃ t : ℝ, t ≥ 0 ∧ t ≤ revolution_time / 4 ∧ height_function t = vertical_shift + height ∧ t = 40 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ferris_wheel_time_l954_95427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_sums_l954_95478

theorem polynomial_identity_sums (b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) 
  (h : ∀ x : ℝ, x^6 - 2*x^5 + 3*x^4 - 3*x^3 + 3*x^2 - 2*x + 1 = 
    (x^2 + b₁*x + c₁) * (x^2 + b₂*x + c₂) * (x^2 + b₃*x + c₃)) : 
  b₁*c₁ + b₂*c₂ + b₃*c₃ = -2 ∧ b₁^2*c₁^2 + b₂^2*c₂^2 + b₃^2*c₃^2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_sums_l954_95478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_b_c_l954_95414

-- Define the vectors as functions of x
def a : ℝ → ℝ × ℝ := λ _ ↦ (2, 1)
def b : ℝ → ℝ × ℝ := λ x ↦ (1 - x, x)
def c : ℝ → ℝ × ℝ := λ x ↦ (-3*x, 3*x)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 * w.2 = k * v.2 * w.1

-- Define the cosine of the angle between two vectors
noncomputable def cos_angle (v w : ℝ × ℝ) : ℝ :=
  (v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2))

-- State the theorem
theorem cos_angle_b_c :
  ∃ x : ℝ, parallel (a x) (b x) ∧ cos_angle (b x) (c x) = -Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_b_c_l954_95414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gold_cube_value_is_6000_l954_95402

/-- Represents the properties of a cube made of a specific material -/
structure Cube where
  sideLength : ℝ
  weight : ℝ
  value : ℝ

/-- Calculates the value of a gold cube based on a silver cube's properties -/
noncomputable def goldCubeValue (silverCube : Cube) : ℝ :=
  let goldWeight := 2 * silverCube.weight
  let silverPricePerPound := silverCube.value / silverCube.weight
  let goldPricePerPound := 3 * silverPricePerPound
  goldWeight * goldPricePerPound

/-- Theorem stating the value of a gold cube given specific conditions -/
theorem gold_cube_value_is_6000 (silverCube : Cube) 
    (h1 : silverCube.sideLength = 4)
    (h2 : silverCube.weight = 5)
    (h3 : silverCube.value = 1000) :
  goldCubeValue silverCube = 6000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gold_cube_value_is_6000_l954_95402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scrooge_pie_share_l954_95483

theorem scrooge_pie_share :
  ∀ (x : ℚ),
  let total_pie : ℚ := 1
  let leftover_pie : ℚ := 8/9 * total_pie
  let num_people : ℕ := 4
  let scrooge_multiplier : ℕ := 2
  leftover_pie = (scrooge_multiplier * x + (num_people - 1) * x) →
  x = leftover_pie / (scrooge_multiplier + num_people - 1) →
  scrooge_multiplier * x = 16/45 :=
by
  intro x total_pie leftover_pie num_people scrooge_multiplier
  intro h1 h2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scrooge_pie_share_l954_95483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangent_externally_l954_95463

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

-- Define the centers and radii of the circles
def center_C1 : ℝ × ℝ := (0, 0)
def radius_C1 : ℝ := 1
def center_C2 : ℝ × ℝ := (2, 0)
def radius_C2 : ℝ := 1

-- Define the distance between the centers
noncomputable def distance_between_centers : ℝ := 
  Real.sqrt ((center_C2.1 - center_C1.1)^2 + (center_C2.2 - center_C1.2)^2)

-- Theorem: The circles are tangent externally
theorem circles_tangent_externally :
  distance_between_centers = radius_C1 + radius_C2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangent_externally_l954_95463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2010_equals_1_l954_95470

/-- A function f defined as f(x) = a*sin(π*x + α) + b*cos(π*x + β) -/
noncomputable def f (a b α β : ℝ) (x : ℝ) : ℝ := 
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β)

/-- Theorem: If f(2009) = -1, then f(2010) = 1 -/
theorem f_2010_equals_1 (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) 
  (h : f a b α β 2009 = -1) : f a b α β 2010 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2010_equals_1_l954_95470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l954_95437

theorem lambda_range (l : ℝ) :
  (∀ a b : ℝ, a^2 + 8*b^2 ≥ l*b*(a + b)) → -8 ≤ l ∧ l ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l954_95437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_theorem_l954_95497

/-- The circle with center (1, 0) and radius 5 -/
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

/-- The point P -/
def P : ℝ × ℝ := (2, -1)

/-- The line passing through P that forms the shortest chord on the circle -/
def shortest_chord_line (x y : ℝ) : Prop := x - y - 3 = 0

/-- 
Theorem: The line x - y - 3 = 0 forms the shortest chord of the circle 
(x-1)^2 + y^2 = 25 passing through the point P(2, -1)
-/
theorem shortest_chord_theorem :
  ∀ (x y : ℝ), 
    shortest_chord_line x y ↔ 
      (∃ (a b : ℝ), 
        my_circle a b ∧ 
        shortest_chord_line a b ∧
        (∀ (c d : ℝ), my_circle c d → (c - 2)^2 + (d + 1)^2 ≥ (a - 2)^2 + (b + 1)^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_theorem_l954_95497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_10_l954_95410

-- Define the length of the train in meters
noncomputable def train_length : ℝ := 400

-- Define the time taken to cross the electric pole in seconds
noncomputable def crossing_time : ℝ := 40

-- Define the speed of the train in meters per second
noncomputable def train_speed : ℝ := train_length / crossing_time

-- Theorem statement
theorem train_speed_is_10 : train_speed = 10 := by
  -- Unfold the definitions
  unfold train_speed train_length crossing_time
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_10_l954_95410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_less_than_sum_of_cos_l954_95469

-- Define acute angles
def is_acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < Real.pi / 2

-- State the theorem
theorem cos_sum_less_than_sum_of_cos 
  (α β : ℝ) 
  (h_α : is_acute_angle α) 
  (h_β : is_acute_angle β) : 
  Real.cos (α + β) < Real.cos α + Real.cos β := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_less_than_sum_of_cos_l954_95469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l954_95459

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (x^2 + 6)

theorem f_properties :
  (∃ k : ℝ, (∀ x : ℝ, f x > k ↔ x < -3 ∨ x > -2) ∧ k = -2/5) ∧
  (∀ x : ℝ, x > 0 → f x ≤ Real.sqrt 6 / 6) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l954_95459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_altitude_l954_95413

-- Define a triangle with given side lengths
def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the area of a triangle using Heron's formula
noncomputable def area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Define the altitude corresponding to the largest side
noncomputable def altitude (a b c : ℝ) : ℝ :=
  2 * (area a b c) / (max a (max b c))

-- Theorem statement
theorem triangle_area_and_altitude :
  Triangle 41 28 15 →
  area 41 28 15 = 126 ∧
  altitude 41 28 15 = 9 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_altitude_l954_95413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_participants_selected_l954_95429

/-- Represents a random number table -/
def RandomNumberTable := List (List Nat)

/-- Represents a student number -/
def StudentNumber := Fin 1000

/-- Selects participants based on the given conditions -/
def selectParticipants (table : RandomNumberTable) (totalStudents : Nat) (numToSelect : Nat) 
  (startRow : Nat) (startCol : Nat) : List StudentNumber :=
  sorry

/-- The theorem to prove -/
theorem correct_participants_selected 
  (table : RandomNumberTable) 
  (totalStudents : Nat) 
  (numToSelect : Nat) 
  (startRow : Nat) 
  (startCol : Nat) :
  totalStudents = 247 →
  numToSelect = 4 →
  startRow = 4 →
  startCol = 9 →
  selectParticipants table totalStudents numToSelect startRow startCol = 
    [⟨050, sorry⟩, ⟨121, sorry⟩, ⟨014, sorry⟩, ⟨218, sorry⟩] :=
by
  sorry

#eval (⟨050, sorry⟩ : StudentNumber)
#eval (⟨121, sorry⟩ : StudentNumber)
#eval (⟨014, sorry⟩ : StudentNumber)
#eval (⟨218, sorry⟩ : StudentNumber)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_participants_selected_l954_95429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grinder_loss_percentage_l954_95412

/-- Represents the financial transaction of buying and selling items -/
structure Transaction where
  grinder_cp : ℚ  -- Cost price of grinder
  mobile_cp : ℚ   -- Cost price of mobile
  mobile_profit_percent : ℚ  -- Profit percentage on mobile
  total_profit : ℚ  -- Overall profit
  grinder_loss_percent : ℚ  -- Loss percentage on grinder (to be proved)

/-- Calculates the selling price of an item given its cost price and profit/loss percentage -/
def selling_price (cost : ℚ) (percent : ℚ) : ℚ :=
  cost * (1 + percent / 100)

/-- Theorem stating that the loss percentage on the grinder is 5% -/
theorem grinder_loss_percentage (t : Transaction) 
  (h1 : t.grinder_cp = 15000)
  (h2 : t.mobile_cp = 8000)
  (h3 : t.mobile_profit_percent = 10)
  (h4 : t.total_profit = 50)
  : t.grinder_loss_percent = 5 := by
  sorry

#check grinder_loss_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grinder_loss_percentage_l954_95412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_copper_electron_transfer_l954_95472

/-- Avogadro's constant -/
axiom N_A : ℝ

/-- Mass of copper in grams -/
def mass_Cu : ℝ := 32

/-- Molar mass of copper in g/mol -/
def molar_mass_Cu : ℝ := 63.546

/-- Amount of substance of copper in moles -/
noncomputable def amount_Cu : ℝ := mass_Cu / molar_mass_Cu

/-- Valency of copper in the reaction with nitric acid -/
def valency_Cu : ℕ := 2

/-- Theorem stating that the number of electrons transferred by copper is equal to Avogadro's constant -/
theorem copper_electron_transfer :
  (amount_Cu * (valency_Cu : ℝ) * N_A) = N_A := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_copper_electron_transfer_l954_95472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skipping_performance_most_suitable_l954_95488

-- Define a structure for survey methods
structure SurveyMethod where
  name : String
  covers : String → Prop
  usesSampling : Prop

-- Define the characteristics of a comprehensive survey
def comprehensiveSurvey (method : SurveyMethod) : Prop :=
  (∀ unit, method.covers unit) ∧ ¬method.usesSampling

-- Define the four survey methods
def milkHygieneTest : SurveyMethod := {
  name := "Testing whether a certain brand of fresh milk meets food hygiene standards",
  covers := λ _ => false,  -- Assume it doesn't cover all units
  usesSampling := true
}

def skippingPerformance : SurveyMethod := {
  name := "Understanding the one-minute skipping performance of a class of students",
  covers := λ _ => true,  -- Covers all students in the class
  usesSampling := false
}

def beijingStudentVision : SurveyMethod := {
  name := "Understanding the vision of middle school students in Beijing",
  covers := λ _ => false,  -- Assume it doesn't cover all students
  usesSampling := true
}

def carImpactResistance : SurveyMethod := {
  name := "Investigating the impact resistance of a batch of cars",
  covers := λ _ => false,  -- Assume it doesn't cover all cars
  usesSampling := true
}

-- Define the theorem
theorem skipping_performance_most_suitable :
  comprehensiveSurvey skippingPerformance ∧
  ¬comprehensiveSurvey milkHygieneTest ∧
  ¬comprehensiveSurvey beijingStudentVision ∧
  ¬comprehensiveSurvey carImpactResistance :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_skipping_performance_most_suitable_l954_95488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_to_pen_ratio_2006_l954_95434

/-- Represents the revenue from pens in 2006 -/
def P : ℝ := sorry

/-- Represents the revenue from pencils in 2006 -/
def C : ℝ := sorry

/-- The revenue from pens in 2007 increased by 5% -/
def pen_revenue_2007 : ℝ := 1.05 * P

/-- The revenue from pencils in 2007 decreased by 13% -/
def pencil_revenue_2007 : ℝ := 0.87 * C

/-- The overall revenue in 2006 -/
def overall_revenue_2006 : ℝ := P + C

/-- The overall revenue in 2007 decreased by 1% -/
def overall_revenue_2007 : ℝ := 0.99 * overall_revenue_2006

/-- Theorem stating that the ratio of pencil revenue to pen revenue in 2006 is 1/2 -/
theorem pencil_to_pen_ratio_2006 : C / P = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_to_pen_ratio_2006_l954_95434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_group_b_l954_95446

/-- Represents a group in the math modeling interest groups --/
structure MathGroup where
  members : ℕ

/-- Represents the college's math modeling interest groups --/
structure College where
  groupA : MathGroup
  groupB : MathGroup
  groupC : MathGroup

/-- Calculates the total number of members in all groups --/
def totalMembers (c : College) : ℕ :=
  c.groupA.members + c.groupB.members + c.groupC.members

/-- Calculates the number of people selected from a group in stratified sampling --/
def selectedFromGroup (g : MathGroup) (sampleSize : ℕ) (c : College) : ℚ :=
  (g.members : ℚ) * (sampleSize : ℚ) / (totalMembers c : ℚ)

/-- The main theorem to prove --/
theorem stratified_sampling_group_b (c : College) (sampleSize : ℕ) :
  c.groupA.members = 45 → 
  c.groupB.members = 45 → 
  c.groupC.members = 60 → 
  sampleSize = 10 →
  selectedFromGroup c.groupB sampleSize c = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_group_b_l954_95446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_close_points_l954_95441

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define an equilateral triangle
def EquilateralTriangle := Point × Point × Point

-- Function to check if a point is inside a triangle
def isInside (p : Point) (t : EquilateralTriangle) : Prop := sorry

-- Function to calculate distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Main theorem
theorem exist_close_points 
  (triangle : EquilateralTriangle) 
  (side_length : ℝ) 
  (h_side : side_length = 2) 
  (points : Finset Point) 
  (h_count : points.card = 5) 
  (h_inside : ∀ p, p ∈ points → isInside p triangle) :
  ∃ p1 p2, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ distance p1 p2 ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_close_points_l954_95441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_distance_bounds_l954_95418

/-- Given a circle with center O and radius 10, and a point M at distance 15 from O,
    prove that the minimum and maximum distances from M to any point on the circle
    are 5 and 25 respectively. -/
theorem circle_distance_bounds (O M : EuclideanSpace ℝ (Fin 2)) 
  (X : Set (EuclideanSpace ℝ (Fin 2))) : 
  (∀ x ∈ X, ‖x - O‖ = 10) →  -- X represents points on the circle
  ‖M - O‖ = 15 → 
  ∃ (min max : ℝ), 
    (∀ x ∈ X, min ≤ ‖M - x‖ ∧ ‖M - x‖ ≤ max) ∧ 
    min = 5 ∧ max = 25 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_distance_bounds_l954_95418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_l954_95475

def A : ℝ × ℝ × ℝ := (1, 3, 5)
def B : ℝ × ℝ × ℝ := (-3, 6, -7)

theorem distance_AB : Real.sqrt ((A.fst - B.fst)^2 + (A.snd.fst - B.snd.fst)^2 + (A.snd.snd - B.snd.snd)^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_l954_95475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_highway_miles_per_tankful_l954_95466

/-- Represents the fuel efficiency and travel distance of a car in different settings. -/
structure CarFuelData where
  city_miles_per_tankful : ℚ
  city_miles_per_gallon : ℚ
  highway_city_mpg_difference : ℚ

/-- Calculates the miles per tankful on the highway given car fuel data. -/
def highway_miles_per_tankful (data : CarFuelData) : ℚ :=
  let tank_size := data.city_miles_per_tankful / data.city_miles_per_gallon
  let highway_mpg := data.city_miles_per_gallon + data.highway_city_mpg_difference
  tank_size * highway_mpg

/-- Theorem stating that given the specified conditions, the car travels 462 miles per tankful on the highway. -/
theorem car_highway_miles_per_tankful (data : CarFuelData) 
    (h1 : data.city_miles_per_tankful = 336)
    (h2 : data.city_miles_per_gallon = 40)
    (h3 : data.highway_city_mpg_difference = 15) :
    highway_miles_per_tankful data = 462 := by
  sorry

#eval highway_miles_per_tankful ⟨336, 40, 15⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_highway_miles_per_tankful_l954_95466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_not_two_pi_over_nine_l954_95492

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.cos (2 * x) * Real.cos (4 * x)

-- State the theorem
theorem alpha_not_two_pi_over_nine (α : ℝ) (h : f α = 1/8) : α ≠ 2*Real.pi/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_not_two_pi_over_nine_l954_95492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l954_95454

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (1 - 1 / (x + 3))

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < -3 ∨ x > -2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l954_95454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_implies_a_range_l954_95415

theorem quadratic_inequality_implies_a_range :
  (∀ x : ℝ, a * x^2 - a * x - 2 ≤ 0) → a ∈ Set.Icc (-8 : ℝ) 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_implies_a_range_l954_95415
