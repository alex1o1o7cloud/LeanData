import Mathlib

namespace NUMINAMATH_CALUDE_reading_activity_results_l4051_405130

def characters_per_day : ℕ := 850
def days_per_week : ℕ := 7
def total_weeks : ℕ := 20

def characters_per_week : ℕ := characters_per_day * days_per_week
def total_characters : ℕ := characters_per_week * total_weeks

def approximate_ten_thousands (n : ℕ) : ℕ :=
  (n + 5000) / 10000

theorem reading_activity_results :
  characters_per_week = 5950 ∧
  total_characters = 119000 ∧
  approximate_ten_thousands total_characters = 12 :=
by sorry

end NUMINAMATH_CALUDE_reading_activity_results_l4051_405130


namespace NUMINAMATH_CALUDE_min_value_fraction_l4051_405105

theorem min_value_fraction (x : ℝ) (h : x > 9) :
  x^2 / (x - 9) ≥ 36 ∧ ∃ y > 9, y^2 / (y - 9) = 36 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l4051_405105


namespace NUMINAMATH_CALUDE_function_value_at_2004_l4051_405129

/-- Given a function f(x) = a*sin(πx + α) + b*cos(πx + β) + 4,
    where α, β, a, and b are non-zero real numbers, and f(2003) = 6,
    prove that f(2004) = 2. -/
theorem function_value_at_2004 
  (α β a b : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) 
  (f : ℝ → ℝ) 
  (hf : ∀ x, f x = a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4) 
  (h2003 : f 2003 = 6) : 
  f 2004 = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_2004_l4051_405129


namespace NUMINAMATH_CALUDE_grocery_store_order_l4051_405174

theorem grocery_store_order (peas carrots corn : ℕ) 
  (h_peas : peas = 810) 
  (h_carrots : carrots = 954) 
  (h_corn : corn = 675) : 
  ∃ (boxes packs cases : ℕ), 
    boxes * 4 ≥ peas ∧ 
    (boxes - 1) * 4 < peas ∧ 
    packs * 6 = carrots ∧ 
    cases * 5 = corn ∧ 
    boxes = 203 ∧ 
    packs = 159 ∧ 
    cases = 135 := by
  sorry

#check grocery_store_order

end NUMINAMATH_CALUDE_grocery_store_order_l4051_405174


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l4051_405121

theorem polynomial_coefficient_sum (A B C D : ℝ) : 
  (∀ x : ℝ, (x - 3) * (4 * x^2 + 2 * x - 6) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l4051_405121


namespace NUMINAMATH_CALUDE_tylers_puppies_l4051_405153

theorem tylers_puppies (num_dogs : ℕ) (puppies_per_dog : ℕ) : 
  num_dogs = 15 → puppies_per_dog = 5 → num_dogs * puppies_per_dog = 75 := by
  sorry

end NUMINAMATH_CALUDE_tylers_puppies_l4051_405153


namespace NUMINAMATH_CALUDE_rectangle_max_area_l4051_405154

theorem rectangle_max_area (l w : ℕ) : 
  (2 * l + 2 * w = 40) → 
  (∀ a b : ℕ, 2 * a + 2 * b = 40 → l * w ≥ a * b) → 
  l * w = 100 := by
sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l4051_405154


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_system_of_equations_solution_l4051_405100

-- Problem 1
theorem sqrt_sum_equality : |Real.sqrt 2 - Real.sqrt 5| + 2 * Real.sqrt 2 = Real.sqrt 5 + Real.sqrt 2 := by
  sorry

-- Problem 2
theorem system_of_equations_solution :
  ∃ (x y : ℝ), 4 * x + y = 15 ∧ 3 * x - 2 * y = 3 ∧ x = 3 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_system_of_equations_solution_l4051_405100


namespace NUMINAMATH_CALUDE_good_pair_exists_l4051_405127

theorem good_pair_exists (m : ℕ) : ∃ n : ℕ, n > m ∧ 
  ∃ a b : ℕ, m * n = a ^ 2 ∧ (m + 1) * (n + 1) = b ^ 2 := by
  let n := m * (4 * m + 3) ^ 2
  have h1 : n > m := sorry
  have h2 : ∃ a : ℕ, m * n = a ^ 2 := sorry
  have h3 : ∃ b : ℕ, (m + 1) * (n + 1) = b ^ 2 := sorry
  exact ⟨n, h1, h2.choose, h3.choose, h2.choose_spec, h3.choose_spec⟩

end NUMINAMATH_CALUDE_good_pair_exists_l4051_405127


namespace NUMINAMATH_CALUDE_quadratic_complex_roots_l4051_405128

theorem quadratic_complex_roots : ∃ (z₁ z₂ : ℂ),
  z₁ = Complex.mk (Real.sqrt 7 - 1) ((Real.sqrt 7) / 2) ∧
  z₂ = Complex.mk (-(Real.sqrt 7) - 1) (-(Real.sqrt 7) / 2) ∧
  z₁^2 + 2*z₁ = Complex.mk 3 7 ∧
  z₂^2 + 2*z₂ = Complex.mk 3 7 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_complex_roots_l4051_405128


namespace NUMINAMATH_CALUDE_unique_common_tangent_parabolas_l4051_405122

/-- Given two parabolas C₁ and C₂, prove that if they have exactly one common tangent,
    then a = -1/2 and the equation of the common tangent is y = x - 1/4 -/
theorem unique_common_tangent_parabolas (x y : ℝ) :
  let C₁ : ℝ → ℝ := λ x => x^2 + 2*x
  let C₂ : ℝ → ℝ → ℝ := λ a x => -x^2 + a
  ∃! (m b a : ℝ), (∀ x, (y = m*x + b) → (y = C₁ x ∨ y = C₂ a x) → 
    (∃ x₀, (C₁ x₀ = m*x₀ + b ∧ (2*x₀ + 2 = m)) ∧ 
           (C₂ a x₀ = m*x₀ + b ∧ (-2*x₀ = m))))
  → a = -1/2 ∧ m = 1 ∧ b = -1/4 := by
sorry

end NUMINAMATH_CALUDE_unique_common_tangent_parabolas_l4051_405122


namespace NUMINAMATH_CALUDE_product_equals_difference_of_squares_l4051_405168

theorem product_equals_difference_of_squares (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_difference_of_squares_l4051_405168


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l4051_405163

/-- Given a geometric sequence of positive integers where the first term is 3
    and the fifth term is 243, the seventh term is 2187. -/
theorem geometric_sequence_seventh_term :
  ∀ (a : ℕ → ℕ),
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 3 →                            -- first term is 3
  a 5 = 243 →                          -- fifth term is 243
  a 7 = 2187 :=                        -- seventh term is 2187
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l4051_405163


namespace NUMINAMATH_CALUDE_rectangular_box_dimensions_sum_l4051_405173

theorem rectangular_box_dimensions_sum (A B C : ℝ) 
  (h1 : A * B = 40)
  (h2 : B * C = 90)
  (h3 : C * A = 100) :
  A + B + C = 83/3 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_dimensions_sum_l4051_405173


namespace NUMINAMATH_CALUDE_f_continuity_and_discontinuity_l4051_405175

-- Define the piecewise function
noncomputable def f (x : ℝ) : ℝ :=
  if x < -3 then (x^2 + 3*x - 1) / (x + 2)
  else if x ≤ 4 then (x + 2)^2
  else 9*x + 1

-- Define continuity at a point
def continuous_at (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - a| < δ → |f x - f a| < ε

-- Define left and right limits
def has_limit_at_left (f : ℝ → ℝ) (a : ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, a - δ < x ∧ x < a → |f x - L| < ε

def has_limit_at_right (f : ℝ → ℝ) (a : ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, a < x ∧ x < a + δ → |f x - L| < ε

-- Define jump discontinuity
def jump_discontinuity (f : ℝ → ℝ) (a : ℝ) (jump : ℝ) : Prop :=
  ∃ L₁ L₂, has_limit_at_left f a L₁ ∧ has_limit_at_right f a L₂ ∧ L₂ - L₁ = jump

-- Theorem statement
theorem f_continuity_and_discontinuity :
  continuous_at f (-3) ∧ jump_discontinuity f 4 1 :=
sorry

end NUMINAMATH_CALUDE_f_continuity_and_discontinuity_l4051_405175


namespace NUMINAMATH_CALUDE_x_convergence_bound_l4051_405182

def x : ℕ → ℚ
  | 0 => 3
  | n + 1 => (x n ^ 2 + 6 * x n + 8) / (x n + 7)

theorem x_convergence_bound :
  ∃ m : ℕ, 31 ≤ m ∧ m ≤ 90 ∧ 
    x m ≤ 5 + 1 / (2^15) ∧
    ∀ k : ℕ, 0 < k → k < m → x k > 5 + 1 / (2^15) := by
  sorry

end NUMINAMATH_CALUDE_x_convergence_bound_l4051_405182


namespace NUMINAMATH_CALUDE_perpendicular_parallel_lines_to_plane_l4051_405177

/-- Two lines are parallel -/
def parallel_lines (a b : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry

/-- The theorem to be proved -/
theorem perpendicular_parallel_lines_to_plane 
  (a b : Line) (α : Plane) 
  (h1 : a ≠ b) 
  (h2 : parallel_lines a b) 
  (h3 : perpendicular_line_plane a α) : 
  perpendicular_line_plane b α := by sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_lines_to_plane_l4051_405177


namespace NUMINAMATH_CALUDE_brother_is_tweedledee_l4051_405171

-- Define the two brothers
inductive Brother
| tweedledee
| tweedledum

-- Define a proposition for "lying today"
def lying_today (b : Brother) : Prop := sorry

-- Define the statement made by the brother
def brother_statement (b : Brother) : Prop :=
  lying_today b ∨ b = Brother.tweedledee

-- Theorem stating that the brother must be Tweedledee
theorem brother_is_tweedledee (b : Brother) : 
  brother_statement b → b = Brother.tweedledee :=
by sorry

end NUMINAMATH_CALUDE_brother_is_tweedledee_l4051_405171


namespace NUMINAMATH_CALUDE_remainder_123456789012_mod_252_l4051_405160

theorem remainder_123456789012_mod_252 : 
  123456789012 % 252 = 156 := by sorry

end NUMINAMATH_CALUDE_remainder_123456789012_mod_252_l4051_405160


namespace NUMINAMATH_CALUDE_unique_number_with_conditions_l4051_405118

theorem unique_number_with_conditions : ∃! b : ℤ, 
  40 < b ∧ b < 120 ∧ 
  b % 4 = 3 ∧ 
  b % 5 = 3 ∧ 
  b % 6 = 3 ∧
  b = 63 := by sorry

end NUMINAMATH_CALUDE_unique_number_with_conditions_l4051_405118


namespace NUMINAMATH_CALUDE_fraction_equals_sqrt_two_l4051_405112

theorem fraction_equals_sqrt_two (a b : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : a^2 + b^2 = 6*a*b) : 
  (a + b) / (a - b) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_equals_sqrt_two_l4051_405112


namespace NUMINAMATH_CALUDE_basketball_team_sales_l4051_405162

/-- The number of cupcakes sold by the basketball team -/
def num_cupcakes : ℕ := 50

/-- The price of each cupcake in dollars -/
def cupcake_price : ℚ := 2

/-- The number of cookies sold -/
def num_cookies : ℕ := 40

/-- The price of each cookie in dollars -/
def cookie_price : ℚ := 1/2

/-- The number of basketballs bought -/
def num_basketballs : ℕ := 2

/-- The price of each basketball in dollars -/
def basketball_price : ℚ := 40

/-- The number of energy drinks bought -/
def num_energy_drinks : ℕ := 20

/-- The price of each energy drink in dollars -/
def energy_drink_price : ℚ := 2

theorem basketball_team_sales :
  (num_cupcakes * cupcake_price + num_cookies * cookie_price : ℚ) =
  (num_basketballs * basketball_price + num_energy_drinks * energy_drink_price : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_sales_l4051_405162


namespace NUMINAMATH_CALUDE_rotated_point_x_coordinate_l4051_405103

theorem rotated_point_x_coordinate 
  (P : ℝ × ℝ) 
  (h_unit_circle : P.1^2 + P.2^2 = 1) 
  (h_P : P = (4/5, -3/5)) : 
  let Q := (
    P.1 * Real.cos (π/3) - P.2 * Real.sin (π/3),
    P.1 * Real.sin (π/3) + P.2 * Real.cos (π/3)
  )
  Q.1 = (4 + 3 * Real.sqrt 3) / 10 := by
sorry

end NUMINAMATH_CALUDE_rotated_point_x_coordinate_l4051_405103


namespace NUMINAMATH_CALUDE_evaluate_expression_l4051_405104

theorem evaluate_expression : -((18 / 3)^2 * 4 - 80 + 5 * 7) = -99 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l4051_405104


namespace NUMINAMATH_CALUDE_vertex_of_our_parabola_l4051_405190

/-- A parabola is defined by the equation y = (x - h)^2 + k, where (h, k) is its vertex -/
structure Parabola where
  h : ℝ
  k : ℝ

/-- The parabola y = (x - 2)^2 + 1 -/
def our_parabola : Parabola := { h := 2, k := 1 }

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := (p.h, p.k)

theorem vertex_of_our_parabola :
  vertex our_parabola = (2, 1) := by sorry

end NUMINAMATH_CALUDE_vertex_of_our_parabola_l4051_405190


namespace NUMINAMATH_CALUDE_pigs_in_barn_l4051_405164

/-- The total number of pigs after more pigs join the barn -/
def total_pigs (initial : Float) (joined : Float) : Float :=
  initial + joined

/-- Theorem stating that given 64.0 initial pigs and 86.0 pigs joining, the total is 150.0 -/
theorem pigs_in_barn : total_pigs 64.0 86.0 = 150.0 := by
  sorry

end NUMINAMATH_CALUDE_pigs_in_barn_l4051_405164


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4051_405183

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a b : ℕ → ℝ) :
  ArithmeticSequence a →
  ArithmeticSequence b →
  a 1 + b 1 = 7 →
  a 3 + b 3 = 21 →
  a 5 + b 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4051_405183


namespace NUMINAMATH_CALUDE_happy_boys_count_l4051_405116

theorem happy_boys_count (total_children total_happy total_sad total_neutral total_boys total_girls sad_girls : ℕ)
  (h1 : total_children = 60)
  (h2 : total_happy = 30)
  (h3 : total_sad = 10)
  (h4 : total_neutral = 20)
  (h5 : total_boys = 18)
  (h6 : total_girls = 42)
  (h7 : sad_girls = 4)
  (h8 : total_children = total_happy + total_sad + total_neutral)
  (h9 : total_children = total_boys + total_girls)
  (h10 : ∃ (happy_boys : ℕ), happy_boys > 0) :
  ∃ (happy_boys : ℕ), happy_boys = 12 :=
sorry

end NUMINAMATH_CALUDE_happy_boys_count_l4051_405116


namespace NUMINAMATH_CALUDE_workshop_workers_correct_l4051_405131

/-- The number of workers in a workshop with given salary conditions -/
def workshop_workers : ℕ :=
  let average_salary : ℚ := 750
  let technician_count : ℕ := 5
  let technician_salary : ℚ := 900
  let non_technician_salary : ℚ := 700
  20

/-- Proof that the number of workers in the workshop is correct -/
theorem workshop_workers_correct :
  let average_salary : ℚ := 750
  let technician_count : ℕ := 5
  let technician_salary : ℚ := 900
  let non_technician_salary : ℚ := 700
  let total_workers := workshop_workers
  (average_salary * total_workers : ℚ) =
    technician_salary * technician_count +
    non_technician_salary * (total_workers - technician_count) :=
by
  sorry

#eval workshop_workers

end NUMINAMATH_CALUDE_workshop_workers_correct_l4051_405131


namespace NUMINAMATH_CALUDE_sequence_problem_l4051_405184

def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b

def is_geometric_sequence (a b c : ℝ) : Prop :=
  b / a = c / b

theorem sequence_problem (x y : ℝ) :
  is_arithmetic_sequence (2 * x) 1 (y - 1) →
  is_geometric_sequence (y + 3) (|x + 1| + |x - 1|) (Real.cos (Real.arcsin (Real.sqrt (1 - x^2)))) →
  (x + 1) * (y + 1) = 4 ∨ (x + 1) * (y + 1) = 2 * (Real.sqrt 17 - 3) := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l4051_405184


namespace NUMINAMATH_CALUDE_composite_function_inverse_l4051_405165

theorem composite_function_inverse (a b : ℝ) : 
  let f (x : ℝ) := a * x + b
  let g (x : ℝ) := -2 * x^2 + 4 * x - 1
  let h := f ∘ g
  (∀ x, h.invFun x = 2 * x - 3) →
  2 * a - 3 * b = -91 / 32 := by
sorry

end NUMINAMATH_CALUDE_composite_function_inverse_l4051_405165


namespace NUMINAMATH_CALUDE_f_properties_l4051_405145

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.log x + a) / x

def monotonicity_intervals (a : ℝ) : Prop :=
  (a > 0 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.exp (1 - a) → f a x₁ < f a x₂) ∧
            (∀ x₁ x₂, Real.exp (1 - a) < x₁ ∧ x₁ < x₂ → f a x₁ > f a x₂)) ∧
  (a < 0 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.exp (1 - a) → f a x₁ > f a x₂) ∧
            (∀ x₁ x₂, Real.exp (1 - a) < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂))

def has_root_in_interval (a : ℝ) : Prop :=
  ∃ x, Real.exp 1 < x ∧ f a x = 0

theorem f_properties (a : ℝ) (h : a ≠ 0) :
  monotonicity_intervals a ∧ (has_root_in_interval a ↔ a < -1) :=
sorry

end

end NUMINAMATH_CALUDE_f_properties_l4051_405145


namespace NUMINAMATH_CALUDE_cars_sold_per_day_second_period_l4051_405133

def total_quota : ℕ := 50
def total_days : ℕ := 30
def first_period : ℕ := 3
def second_period : ℕ := 4
def cars_per_day_first_period : ℕ := 5
def remaining_cars : ℕ := 23

theorem cars_sold_per_day_second_period :
  let cars_sold_first_period := first_period * cars_per_day_first_period
  let remaining_after_first_period := total_quota - cars_sold_first_period
  let cars_to_sell_second_period := remaining_after_first_period - remaining_cars
  cars_to_sell_second_period / second_period = 3 := by sorry

end NUMINAMATH_CALUDE_cars_sold_per_day_second_period_l4051_405133


namespace NUMINAMATH_CALUDE_square_of_1307_squared_l4051_405117

theorem square_of_1307_squared : (1307 * 1307)^2 = 2918129502401 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1307_squared_l4051_405117


namespace NUMINAMATH_CALUDE_cave_door_weight_l4051_405134

/-- The weight already on the switch in pounds -/
def initial_weight : ℕ := 234

/-- The additional weight needed in pounds -/
def additional_weight : ℕ := 478

/-- The total weight needed to open the cave doors in pounds -/
def total_weight : ℕ := initial_weight + additional_weight

/-- Theorem stating that the total weight needed to open the cave doors is 712 pounds -/
theorem cave_door_weight : total_weight = 712 := by
  sorry

end NUMINAMATH_CALUDE_cave_door_weight_l4051_405134


namespace NUMINAMATH_CALUDE_max_n_with_divisor_condition_l4051_405113

theorem max_n_with_divisor_condition (N : ℕ) : 
  (∃ d₁ d₂ d₃ : ℕ, 
    d₁ ∣ N ∧ d₂ ∣ N ∧ d₃ ∣ N ∧
    d₁ < d₂ ∧ 
    (∀ d : ℕ, d ∣ N → d ≤ d₁ ∨ d ≥ d₂) ∧
    (∀ d : ℕ, d ∣ N → d ≤ d₃ ∨ d > N / d₃) ∧
    d₃ = 21 * d₂) →
  N ≤ 441 := by
sorry

end NUMINAMATH_CALUDE_max_n_with_divisor_condition_l4051_405113


namespace NUMINAMATH_CALUDE_cindy_calculation_l4051_405148

theorem cindy_calculation (x : ℝ) (h : (x - 9) / 3 = 43) : (x - 3) / 9 = 15 := by
  sorry

end NUMINAMATH_CALUDE_cindy_calculation_l4051_405148


namespace NUMINAMATH_CALUDE_maurice_cookout_invites_l4051_405166

/-- The number of people Maurice can invite to the cookout --/
def people_invited : ℕ := by sorry

theorem maurice_cookout_invites :
  let packages : ℕ := 4
  let pounds_per_package : ℕ := 5
  let pounds_per_burger : ℕ := 2
  let total_pounds : ℕ := packages * pounds_per_package
  let total_burgers : ℕ := total_pounds / pounds_per_burger
  people_invited = total_burgers - 1 := by sorry

end NUMINAMATH_CALUDE_maurice_cookout_invites_l4051_405166


namespace NUMINAMATH_CALUDE_subset_implies_m_eq_neg_two_l4051_405126

def set_A (m : ℝ) : Set ℝ := {3, 4, 4*m - 4}
def set_B (m : ℝ) : Set ℝ := {3, m^2}

theorem subset_implies_m_eq_neg_two (m : ℝ) :
  set_B m ⊆ set_A m → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_eq_neg_two_l4051_405126


namespace NUMINAMATH_CALUDE_factory_produces_160_crayons_in_4_hours_l4051_405198

/-- Represents a crayon factory with given specifications -/
structure CrayonFactory where
  num_colors : ℕ
  crayons_per_color_per_box : ℕ
  boxes_per_hour : ℕ

/-- Calculates the total number of crayons produced in a given number of hours -/
def total_crayons_produced (factory : CrayonFactory) (hours : ℕ) : ℕ :=
  factory.num_colors * factory.crayons_per_color_per_box * factory.boxes_per_hour * hours

/-- Theorem stating that a factory with given specifications produces 160 crayons in 4 hours -/
theorem factory_produces_160_crayons_in_4_hours 
  (factory : CrayonFactory) 
  (h1 : factory.num_colors = 4) 
  (h2 : factory.crayons_per_color_per_box = 2) 
  (h3 : factory.boxes_per_hour = 5) : 
  total_crayons_produced factory 4 = 160 := by
  sorry

end NUMINAMATH_CALUDE_factory_produces_160_crayons_in_4_hours_l4051_405198


namespace NUMINAMATH_CALUDE_range_of_m_l4051_405157

theorem range_of_m (a b m : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : ∀ (a b : ℝ), a > 0 → b > 0 → (1/a + 1/b) * Real.sqrt (a^2 + b^2) ≥ 2*m - 4) : 
  m ≤ 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l4051_405157


namespace NUMINAMATH_CALUDE_south_cyclist_speed_l4051_405146

/-- The speed of a cyclist going south, given two cyclists start from the same place
    in opposite directions, one going north at 10 kmph, and they are 50 km apart after 1 hour. -/
def speed_of_south_cyclist : ℝ :=
  let speed_north : ℝ := 10
  let time : ℝ := 1
  let distance_apart : ℝ := 50
  distance_apart - speed_north * time

theorem south_cyclist_speed : speed_of_south_cyclist = 40 := by
  sorry

end NUMINAMATH_CALUDE_south_cyclist_speed_l4051_405146


namespace NUMINAMATH_CALUDE_perimeter_equals_127_32_l4051_405150

/-- The perimeter of a figure constructed with 6 equilateral triangles, where the first triangle
    has a side length of 1 cm and each subsequent triangle has sides equal to half the length
    of the previous triangle. -/
def perimeter_of_triangles : ℚ :=
  let side_lengths : List ℚ := [1, 1/2, 1/4, 1/8, 1/16, 1/32]
  let unique_segments : List ℚ := [1, 1, 1/2, 1/2, 1/4, 1/4, 1/8, 1/8, 1/16, 1/16, 1/32, 1/32, 1/32]
  unique_segments.sum

theorem perimeter_equals_127_32 : perimeter_of_triangles = 127 / 32 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_equals_127_32_l4051_405150


namespace NUMINAMATH_CALUDE_job_crop_production_l4051_405181

/-- Represents the land allocation of Job's farm in hectares -/
structure FarmLand where
  total : ℕ
  house_and_machinery : ℕ
  future_expansion : ℕ
  cattle : ℕ

/-- Calculates the land used for crop production given a FarmLand allocation -/
def crop_production (farm : FarmLand) : ℕ :=
  farm.total - (farm.house_and_machinery + farm.future_expansion + farm.cattle)

/-- Theorem stating that for Job's specific land allocation, the crop production area is 70 hectares -/
theorem job_crop_production :
  let job_farm := FarmLand.mk 150 25 15 40
  crop_production job_farm = 70 := by
  sorry

end NUMINAMATH_CALUDE_job_crop_production_l4051_405181


namespace NUMINAMATH_CALUDE_unique_score_is_correct_l4051_405132

/-- Represents the score on the Mini-AHSME exam -/
structure MiniAHSMEScore where
  total : ℕ
  correct : ℕ
  wrong : ℕ
  h_total : total = 20 + 3 * correct - wrong
  h_questions : correct + wrong ≤ 20

/-- The unique score that satisfies all conditions of the problem -/
def unique_score : MiniAHSMEScore := ⟨53, 11, 0, by simp, by simp⟩

theorem unique_score_is_correct :
  ∀ s : MiniAHSMEScore,
    s.total > 50 →
    (∀ t : MiniAHSMEScore, t.total > 50 ∧ t.total < s.total → 
      ∃ u : MiniAHSMEScore, u.total = t.total ∧ u.correct ≠ t.correct) →
    s = unique_score := by sorry

end NUMINAMATH_CALUDE_unique_score_is_correct_l4051_405132


namespace NUMINAMATH_CALUDE_equation_solution_l4051_405179

theorem equation_solution :
  let f (x : ℝ) := x^2 + 2*x + 1
  let g (x : ℝ) := |3*x - 2|
  let sol₁ := (-7 + Real.sqrt 37) / 2
  let sol₂ := (-7 - Real.sqrt 37) / 2
  (∀ x : ℝ, f x = g x ↔ x = sol₁ ∨ x = sol₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l4051_405179


namespace NUMINAMATH_CALUDE_min_value_expression_l4051_405136

theorem min_value_expression (a b c : ℝ) (h1 : b > c) (h2 : c > a) (h3 : b ≠ 0) :
  ((a + b)^2 + (b - c)^2 + (c - a)^2) / b^2 ≥ 4/3 ∧
  ∃ (a' b' c' : ℝ), b' > c' ∧ c' > a' ∧ b' ≠ 0 ∧
    ((a' + b')^2 + (b' - c')^2 + (c' - a')^2) / b'^2 = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l4051_405136


namespace NUMINAMATH_CALUDE_grade_distribution_l4051_405137

theorem grade_distribution (total_students : ℕ) 
  (below_b_percent : ℚ) (b_or_bplus_percent : ℚ) (a_or_aminus_percent : ℚ) (aplus_percent : ℚ) :
  total_students = 60 →
  below_b_percent = 40 / 100 →
  b_or_bplus_percent = 30 / 100 →
  a_or_aminus_percent = 20 / 100 →
  aplus_percent = 10 / 100 →
  below_b_percent + b_or_bplus_percent + a_or_aminus_percent + aplus_percent = 1 →
  (b_or_bplus_percent + a_or_aminus_percent) * total_students = 30 := by
  sorry

end NUMINAMATH_CALUDE_grade_distribution_l4051_405137


namespace NUMINAMATH_CALUDE_tan_monotone_or_angle_sin_equivalence_l4051_405139

-- Define the tangent function
noncomputable def tan (x : ℝ) : ℝ := Real.tan x

-- Define a predicate for monotonically increasing functions
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define a triangle
structure Triangle :=
  (A B C : ℝ)
  (angle_sum : A + B + C = π)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

-- State the theorem
theorem tan_monotone_or_angle_sin_equivalence :
  (MonotonicallyIncreasing tan) ∨ 
  (∀ t : Triangle, t.A > t.B ↔ Real.sin t.A > Real.sin t.B) :=
sorry

end NUMINAMATH_CALUDE_tan_monotone_or_angle_sin_equivalence_l4051_405139


namespace NUMINAMATH_CALUDE_max_total_score_is_four_l4051_405193

/-- Represents an instructor's scoring for a set of problems -/
structure InstructorScoring :=
  (scores : List ℕ)
  (one_count : ℕ)
  (h_scores : ∀ s ∈ scores, s = 0 ∨ s = 1)
  (h_one_count : one_count = 3)
  (h_one_count_correct : scores.count 1 = one_count)

/-- Calculates the rounded mean of three scores -/
def roundedMean (a b c : ℕ) : ℕ :=
  (a + b + c + 1) / 3

/-- Calculates the total score based on three instructors' scorings -/
def totalScore (i1 i2 i3 : InstructorScoring) : ℕ :=
  List.sum (List.zipWith3 roundedMean i1.scores i2.scores i3.scores)

/-- The main theorem stating that the maximum possible total score is 4 -/
theorem max_total_score_is_four (i1 i2 i3 : InstructorScoring) :
  totalScore i1 i2 i3 ≤ 4 :=
sorry

#check max_total_score_is_four

end NUMINAMATH_CALUDE_max_total_score_is_four_l4051_405193


namespace NUMINAMATH_CALUDE_wam_gm_difference_bound_l4051_405189

theorem wam_gm_difference_bound (k b : ℝ) (h1 : 0 < k) (h2 : k < 1) (h3 : b > 0) : 
  let a := k * b
  let c := (a + b) / 2
  let wam := (2 * a + 3 * b + 4 * c) / 9
  let gm := (a * b * c) ^ (1/3 : ℝ)
  (wam - gm = b * ((5 * k + 5) / 9 - ((k * (k + 1) * b^2) / 2) ^ (1/3 : ℝ))) ∧
  (wam - gm < ((1 - k)^2 * b) / (8 * k)) := by sorry

end NUMINAMATH_CALUDE_wam_gm_difference_bound_l4051_405189


namespace NUMINAMATH_CALUDE_parallelogram_area_12_8_l4051_405102

/-- The area of a parallelogram with given base and height -/
def parallelogramArea (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 12 cm and height 8 cm is 96 square centimeters -/
theorem parallelogram_area_12_8 : parallelogramArea 12 8 = 96 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_12_8_l4051_405102


namespace NUMINAMATH_CALUDE_sum_ages_five_years_from_now_l4051_405123

/-- Represents the ages of Viggo, his younger brother, and his sister -/
structure FamilyAges where
  viggo : ℕ
  brother : ℕ
  sister : ℕ

/-- Calculates the sum of ages after a given number of years -/
def sumAgesAfter (ages : FamilyAges) (years : ℕ) : ℕ :=
  ages.viggo + ages.brother + ages.sister + 3 * years

/-- Theorem stating the sum of ages five years from now -/
theorem sum_ages_five_years_from_now :
  ∃ (initialAges : FamilyAges),
    (initialAges.viggo = 2 * initialAges.brother + 10) →
    (initialAges.sister = initialAges.viggo + 5) →
    (initialAges.brother + 8 = 10) →
    (sumAgesAfter initialAges 5 = 74) := by
  sorry

end NUMINAMATH_CALUDE_sum_ages_five_years_from_now_l4051_405123


namespace NUMINAMATH_CALUDE_probability_different_colors_is_seven_ninths_l4051_405143

/-- The number of color options for socks -/
def sock_colors : ℕ := 3

/-- The number of color options for headband -/
def headband_colors : ℕ := 3

/-- The number of colors shared between socks and headband options -/
def shared_colors : ℕ := 1

/-- The total number of possible combinations -/
def total_combinations : ℕ := sock_colors * headband_colors

/-- The number of combinations where socks and headband have different colors -/
def different_color_combinations : ℕ := 
  sock_colors * headband_colors - sock_colors * shared_colors

/-- The probability of selecting different colors for socks and headband -/
def probability_different_colors : ℚ := 
  different_color_combinations / total_combinations

theorem probability_different_colors_is_seven_ninths : 
  probability_different_colors = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_different_colors_is_seven_ninths_l4051_405143


namespace NUMINAMATH_CALUDE_y_investment_calculation_l4051_405141

/-- Represents the investment and profit sharing of two business partners -/
structure BusinessPartnership where
  /-- The amount invested by partner X -/
  x_investment : ℕ
  /-- The amount invested by partner Y -/
  y_investment : ℕ
  /-- The profit share ratio of partner X -/
  x_profit_ratio : ℕ
  /-- The profit share ratio of partner Y -/
  y_profit_ratio : ℕ

/-- Theorem stating that if the profit is shared in ratio 2:6 and X invested 5000, then Y invested 15000 -/
theorem y_investment_calculation (bp : BusinessPartnership) 
  (h1 : bp.x_investment = 5000)
  (h2 : bp.x_profit_ratio = 2)
  (h3 : bp.y_profit_ratio = 6) :
  bp.y_investment = 15000 := by
  sorry


end NUMINAMATH_CALUDE_y_investment_calculation_l4051_405141


namespace NUMINAMATH_CALUDE_min_value_of_f_l4051_405167

-- Define second-order product sum
def second_order_sum (a b c d : ℤ) : ℤ := a * d + b * c

-- Define third-order product sum
def third_order_sum (a1 a2 a3 b1 b2 b3 c1 c2 c3 : ℤ) : ℤ :=
  a1 * (second_order_sum b2 b3 c2 c3) +
  a2 * (second_order_sum b1 b3 c1 c3) +
  a3 * (second_order_sum b1 b2 c1 c2)

-- Define the function f
def f (n : ℕ+) : ℤ := third_order_sum n 2 (-9) n 1 n 1 2 n

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℤ), m = -21 ∧ ∀ (n : ℕ+), f n ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l4051_405167


namespace NUMINAMATH_CALUDE_number_of_divisors_of_30_l4051_405101

theorem number_of_divisors_of_30 : Finset.card (Nat.divisors 30) = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_30_l4051_405101


namespace NUMINAMATH_CALUDE_lines_are_parallel_l4051_405194

/-- Represents a line in the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if they have the same slope -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

theorem lines_are_parallel : 
  let line1 : Line := { a := 2, b := -1, c := 7 }
  let line2 : Line := { a := 2, b := -1, c := 1 }
  parallel line1 line2 := by
  sorry

end NUMINAMATH_CALUDE_lines_are_parallel_l4051_405194


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l4051_405119

theorem rectangular_to_polar_conversion :
  let x : ℝ := 8
  let y : ℝ := 2 * Real.sqrt 6
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x)
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧ r = 2 * Real.sqrt 22 ∧ θ = Real.arctan (Real.sqrt 6 / 4) := by
  sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l4051_405119


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l4051_405188

-- Define a real-valued function on real numbers
def RealFunction := ℝ → ℝ

-- Define the property of having extreme values
def has_extreme_values (f : RealFunction) : Prop :=
  ∃ x : ℝ, ∀ y : ℝ, f y ≤ f x ∨ f y ≥ f x

-- Define the property of the derivative having real roots
def derivative_has_real_roots (f : RealFunction) : Prop :=
  ∃ x : ℝ, deriv f x = 0

-- Theorem statement
theorem necessary_not_sufficient_condition 
  (f : RealFunction) (hf : Differentiable ℝ f) :
  (has_extreme_values f → derivative_has_real_roots f) ∧
  ¬(derivative_has_real_roots f → has_extreme_values f) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l4051_405188


namespace NUMINAMATH_CALUDE_grape_juice_mixture_l4051_405147

theorem grape_juice_mixture (initial_volume : ℝ) (initial_percentage : ℝ) (added_volume : ℝ) :
  initial_volume = 40 →
  initial_percentage = 20 →
  added_volume = 10 →
  let initial_grape_juice := initial_volume * (initial_percentage / 100)
  let total_grape_juice := initial_grape_juice + added_volume
  let final_volume := initial_volume + added_volume
  let final_percentage := (total_grape_juice / final_volume) * 100
  final_percentage = 36 := by
  sorry

end NUMINAMATH_CALUDE_grape_juice_mixture_l4051_405147


namespace NUMINAMATH_CALUDE_sum_of_digits_9ab_is_18000_l4051_405159

/-- 
Represents an integer consisting of n repetitions of a digit d.
For example, repeat_digit 3 2000 represents 333...333 (2000 threes).
-/
def repeat_digit (d : ℕ) (n : ℕ) : ℕ :=
  (d * (10^n - 1)) / 9

/-- 
Calculates the sum of digits of a natural number in base 10.
-/
def sum_of_digits (n : ℕ) : ℕ :=
  sorry

theorem sum_of_digits_9ab_is_18000 : 
  let a := repeat_digit 3 2000
  let b := repeat_digit 7 2000
  sum_of_digits (9 * a * b) = 18000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_9ab_is_18000_l4051_405159


namespace NUMINAMATH_CALUDE_xy_value_l4051_405140

theorem xy_value (x y : ℝ) (h : (x^2 + 6*x + 12) * (5*y^2 + 2*y + 1) = 12/5) : 
  x * y = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l4051_405140


namespace NUMINAMATH_CALUDE_min_vertical_distance_l4051_405169

-- Define the two functions
def f (x : ℝ) : ℝ := abs x
def g (x : ℝ) : ℝ := -x^2 - 3*x - 2

-- Define the vertical distance between the two functions
def verticalDistance (x : ℝ) : ℝ := abs (f x - g x)

-- Theorem statement
theorem min_vertical_distance :
  ∃ (x : ℝ), verticalDistance x = 1 ∧ ∀ (y : ℝ), verticalDistance y ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_min_vertical_distance_l4051_405169


namespace NUMINAMATH_CALUDE_language_selection_theorem_l4051_405196

theorem language_selection_theorem (n : ℕ) :
  ∀ (employees : Finset (Finset ℕ)),
    (employees.card = 500) →
    (∀ e ∈ employees, e ⊆ Finset.range (2 * n)) →
    (∀ e ∈ employees, e.card ≥ n) →
    ∃ (selected : Finset ℕ),
      selected.card = 14 ∧
      selected ⊆ Finset.range (2 * n) ∧
      ∀ e ∈ employees, ∃ l ∈ selected, l ∈ e :=
by sorry

end NUMINAMATH_CALUDE_language_selection_theorem_l4051_405196


namespace NUMINAMATH_CALUDE_door_height_calculation_l4051_405156

/-- Calculates the height of a door in a room given the room dimensions, door width, window dimensions, number of windows, cost of white washing per square foot, and total cost of white washing. -/
theorem door_height_calculation (room_length room_width room_height : ℝ)
                                (door_width : ℝ)
                                (window_length window_width : ℝ)
                                (num_windows : ℕ)
                                (cost_per_sqft : ℝ)
                                (total_cost : ℝ) :
  room_length = 25 ∧ room_width = 15 ∧ room_height = 12 ∧
  door_width = 3 ∧
  window_length = 4 ∧ window_width = 3 ∧
  num_windows = 3 ∧
  cost_per_sqft = 3 ∧
  total_cost = 2718 →
  ∃ (door_height : ℝ),
    door_height = 6 ∧
    total_cost = (2 * (room_length * room_height + room_width * room_height) -
                  (door_height * door_width + ↑num_windows * window_length * window_width)) * cost_per_sqft :=
by sorry

end NUMINAMATH_CALUDE_door_height_calculation_l4051_405156


namespace NUMINAMATH_CALUDE_a_range_l4051_405109

-- Define the linear equation
def linear_equation (a x : ℝ) : ℝ := a * x + x + 4

-- Define the condition that the root is within [-2, 1]
def root_in_interval (a : ℝ) : Prop :=
  ∃ x, x ∈ Set.Icc (-2) 1 ∧ linear_equation a x = 0

-- State the theorem
theorem a_range (a : ℝ) : 
  root_in_interval a ↔ a ∈ Set.Ioi 1 ∪ Set.Iio (-5) :=
sorry

end NUMINAMATH_CALUDE_a_range_l4051_405109


namespace NUMINAMATH_CALUDE_scientists_from_usa_l4051_405172

theorem scientists_from_usa (total : ℕ) (europe : ℕ) (canada : ℕ) (usa : ℕ)
  (h_total : total = 70)
  (h_europe : europe = total / 2)
  (h_canada : canada = total / 5)
  (h_sum : total = europe + canada + usa) :
  usa = 21 := by
  sorry

end NUMINAMATH_CALUDE_scientists_from_usa_l4051_405172


namespace NUMINAMATH_CALUDE_blue_candy_count_l4051_405114

theorem blue_candy_count (total : ℕ) (red : ℕ) (h1 : total = 11567) (h2 : red = 792) :
  total - red = 10775 := by
  sorry

end NUMINAMATH_CALUDE_blue_candy_count_l4051_405114


namespace NUMINAMATH_CALUDE_smallest_positive_period_l4051_405142

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the symmetric points
variable (a b y₀ : ℝ)

-- Define the symmetry property
def isSymmetric (f : ℝ → ℝ) (x₁ x₂ y : ℝ) : Prop :=
  ∀ t, f (x₁ - t) = 2 * y - f (x₂ + t)

-- State the theorem
theorem smallest_positive_period
  (h₁ : isSymmetric f a a y₀)
  (h₂ : isSymmetric f b b y₀)
  (h₃ : ∀ x, a < x → x < b → ¬ isSymmetric f x x y₀)
  (h₄ : a < b) :
  ∃ T, T > 0 ∧ (∀ x, f (x + T) = f x) ∧
    (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = 2 * (b - a) :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_period_l4051_405142


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_thirds_l4051_405195

theorem reciprocal_of_negative_two_thirds :
  let x : ℚ := -2/3
  let y : ℚ := -3/2
  (x * y = 1) → y = x⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_thirds_l4051_405195


namespace NUMINAMATH_CALUDE_always_two_real_roots_root_less_than_one_implies_k_negative_l4051_405111

-- Define the quadratic equation
def quadratic (x k : ℝ) : ℝ := x^2 - (k+3)*x + 2*k + 2

-- Theorem 1: The equation always has two real roots
theorem always_two_real_roots (k : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic x₁ k = 0 ∧ quadratic x₂ k = 0 :=
sorry

-- Theorem 2: When one root is less than 1, k < 0
theorem root_less_than_one_implies_k_negative (k : ℝ) :
  (∃ x : ℝ, x < 1 ∧ quadratic x k = 0) → k < 0 :=
sorry

end NUMINAMATH_CALUDE_always_two_real_roots_root_less_than_one_implies_k_negative_l4051_405111


namespace NUMINAMATH_CALUDE_max_golf_rounds_is_eight_l4051_405178

/-- Calculates the maximum number of golf rounds that can be played given the specified conditions. -/
def maxGolfRounds (initialCost : ℚ) (membershipFee : ℚ) (budget : ℚ) 
  (discount2nd : ℚ) (discount3rd : ℚ) (discountSubsequent : ℚ) : ℕ :=
  let totalBudget := budget + membershipFee
  let cost1st := initialCost
  let cost2nd := initialCost * (1 - discount2nd)
  let cost3rd := initialCost * (1 - discount3rd)
  let costSubsequent := initialCost * (1 - discountSubsequent)
  let remainingAfter3 := totalBudget - cost1st - cost2nd - cost3rd
  let additionalRounds := (remainingAfter3 / costSubsequent).floor
  3 + additionalRounds.toNat

/-- Theorem stating that the maximum number of golf rounds is 8 under the given conditions. -/
theorem max_golf_rounds_is_eight :
  maxGolfRounds 80 100 400 (1/10) (1/5) (3/10) = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_golf_rounds_is_eight_l4051_405178


namespace NUMINAMATH_CALUDE_triangle_perimeter_l4051_405151

/-- Given a triangle with inradius 1.5 cm and area 29.25 cm², its perimeter is 39 cm. -/
theorem triangle_perimeter (inradius : ℝ) (area : ℝ) (perimeter : ℝ) : 
  inradius = 1.5 → area = 29.25 → perimeter = area / inradius * 2 → perimeter = 39 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l4051_405151


namespace NUMINAMATH_CALUDE_marble_distribution_l4051_405170

theorem marble_distribution (total_marbles : ℕ) (num_friends : ℕ) 
  (h1 : total_marbles = 60) (h2 : num_friends = 4) :
  total_marbles / num_friends = 15 := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l4051_405170


namespace NUMINAMATH_CALUDE_group_size_calculation_l4051_405108

theorem group_size_calculation (initial_avg : ℝ) (new_person_age : ℝ) (new_avg : ℝ) : 
  initial_avg = 15 → new_person_age = 37 → new_avg = 17 → 
  ∃ n : ℕ, (n : ℝ) * initial_avg + new_person_age = (n + 1) * new_avg ∧ n = 10 :=
by sorry

end NUMINAMATH_CALUDE_group_size_calculation_l4051_405108


namespace NUMINAMATH_CALUDE_reciprocal_sum_l4051_405155

theorem reciprocal_sum : (1 / (1/4 + 1/5) : ℚ) = 20/9 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_l4051_405155


namespace NUMINAMATH_CALUDE_greatest_c_value_l4051_405180

theorem greatest_c_value (c : ℝ) : 
  (∀ x : ℝ, -x^2 + 9*x - 20 ≥ 0 → x ≤ 5) ∧ 
  (-5^2 + 9*5 - 20 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_greatest_c_value_l4051_405180


namespace NUMINAMATH_CALUDE_bingo_prize_distribution_l4051_405110

theorem bingo_prize_distribution (total_prize : ℝ) (remaining_winners : ℕ) (each_remaining_prize : ℝ) 
  (h1 : total_prize = 2400)
  (h2 : remaining_winners = 10)
  (h3 : each_remaining_prize = 160)
  (h4 : ∀ f : ℝ, (1 - f) * total_prize / remaining_winners = each_remaining_prize → f = 1/3) :
  ∃ f : ℝ, f * total_prize = total_prize / 3 ∧ 
    (1 - f) * total_prize / remaining_winners = each_remaining_prize := by
  sorry

#check bingo_prize_distribution

end NUMINAMATH_CALUDE_bingo_prize_distribution_l4051_405110


namespace NUMINAMATH_CALUDE_product_of_distinct_nonzero_reals_l4051_405192

theorem product_of_distinct_nonzero_reals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y)
  (h : x - 2 / x = y - 2 / y) : x * y = -2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_distinct_nonzero_reals_l4051_405192


namespace NUMINAMATH_CALUDE_max_area_is_eight_l4051_405120

/-- A line in the form kx - y + 2 = 0 -/
structure Line where
  k : ℝ

/-- A circle in the form x^2 + y^2 - 4x - 12 = 0 -/
def Circle : Type := Unit

/-- Points of intersection between the line and the circle -/
structure Intersection where
  Q : ℝ × ℝ
  R : ℝ × ℝ

/-- The maximum area of triangle QRC given a line and a circle -/
def max_area (l : Line) (C : Circle) (i : Intersection) : ℝ := 8

/-- Theorem stating that the maximum area of triangle QRC is 8 -/
theorem max_area_is_eight (l : Line) (C : Circle) (i : Intersection) :
  max_area l C i = 8 := by sorry

end NUMINAMATH_CALUDE_max_area_is_eight_l4051_405120


namespace NUMINAMATH_CALUDE_pauls_birthday_crayons_l4051_405106

/-- The number of crayons Paul received for his birthday -/
def crayons_received (crayons_left : ℕ) (crayons_lost_or_given : ℕ) 
  (crayons_lost : ℕ) (crayons_given : ℕ) : ℕ :=
  crayons_left + crayons_lost_or_given

/-- Theorem stating the number of crayons Paul received for his birthday -/
theorem pauls_birthday_crayons :
  ∃ (crayons_lost crayons_given : ℕ),
    crayons_lost = 2 * crayons_given ∧
    crayons_lost + crayons_given = 9750 ∧
    crayons_received 2560 9750 crayons_lost crayons_given = 12310 := by
  sorry


end NUMINAMATH_CALUDE_pauls_birthday_crayons_l4051_405106


namespace NUMINAMATH_CALUDE_prop_one_correct_prop_two_not_always_true_prop_three_not_always_true_l4051_405125

-- Define the custom distance function
def customDist (x₁ y₁ x₂ y₂ : ℝ) : ℝ := |x₂ - x₁| + |y₂ - y₁|

-- Proposition 1
theorem prop_one_correct (x₁ y₁ x₂ y₂ x₀ y₀ : ℝ) 
  (h₁ : x₀ ∈ Set.Icc x₁ x₂) (h₂ : y₀ ∈ Set.Icc y₁ y₂) :
  customDist x₁ y₁ x₀ y₀ + customDist x₀ y₀ x₂ y₂ = customDist x₁ y₁ x₂ y₂ := by sorry

-- Proposition 2
theorem prop_two_not_always_true :
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ, 
    customDist x₁ y₁ x₂ y₂ + customDist x₂ y₂ x₃ y₃ ≤ customDist x₁ y₁ x₃ y₃ := by sorry

-- Proposition 3
theorem prop_three_not_always_true :
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ, 
    (x₂ - x₁) * (x₃ - x₁) + (y₂ - y₁) * (y₃ - y₁) = 0 ∧ 
    (customDist x₁ y₁ x₂ y₂)^2 + (customDist x₁ y₁ x₃ y₃)^2 ≠ (customDist x₂ y₂ x₃ y₃)^2 := by sorry

end NUMINAMATH_CALUDE_prop_one_correct_prop_two_not_always_true_prop_three_not_always_true_l4051_405125


namespace NUMINAMATH_CALUDE_cubic_polynomials_with_constant_difference_l4051_405158

/-- Two monic cubic polynomials with specific roots and a constant difference -/
theorem cubic_polynomials_with_constant_difference 
  (f g : ℝ → ℝ) 
  (r : ℝ) 
  (hf : ∃ a : ℝ, ∀ x, f x = (x - (r + 2)) * (x - (r + 8)) * (x - a))
  (hg : ∃ b : ℝ, ∀ x, g x = (x - (r + 4)) * (x - (r + 10)) * (x - b))
  (h_diff : ∀ x, f x - g x = r) :
  r = 32 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomials_with_constant_difference_l4051_405158


namespace NUMINAMATH_CALUDE_ordered_numbers_count_l4051_405199

/-- Counts the numbers from 000 to 999 with digits in non-decreasing or non-increasing order -/
def count_ordered_numbers : ℕ :=
  let non_decreasing := Nat.choose 12 9
  let non_increasing := Nat.choose 12 9
  let double_counted := 10  -- Numbers with all identical digits
  non_decreasing + non_increasing - double_counted

/-- The count of numbers from 000 to 999 with digits in non-decreasing or non-increasing order is 430 -/
theorem ordered_numbers_count : count_ordered_numbers = 430 := by
  sorry

end NUMINAMATH_CALUDE_ordered_numbers_count_l4051_405199


namespace NUMINAMATH_CALUDE_unique_pair_with_single_solution_l4051_405107

theorem unique_pair_with_single_solution :
  ∃! p : ℕ × ℕ, 
    let b := p.1
    let c := p.2
    b > 0 ∧ c > 0 ∧
    (∃! x : ℝ, x^2 + b*x + c = 0) ∧
    (∃! x : ℝ, x^2 + c*x + b = 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_pair_with_single_solution_l4051_405107


namespace NUMINAMATH_CALUDE_call_center_efficiency_l4051_405187

/-- Represents the efficiency and agent ratios of three call center teams -/
structure CallCenterTeams where
  team_a_efficiency : Rat -- Team A's efficiency relative to Team B
  team_c_efficiency : Rat -- Team C's efficiency relative to Team B
  team_a_agents : Rat -- Team A's number of agents relative to Team B
  team_c_agents : Rat -- Team C's number of agents relative to Team B

/-- Calculates the fraction of total calls processed by all three teams combined -/
def total_calls_fraction (teams : CallCenterTeams) : Rat :=
  sorry

/-- Theorem stating that given the specific ratios, the fraction of total calls processed is 19/32 -/
theorem call_center_efficiency (teams : CallCenterTeams)
    (h1 : teams.team_a_efficiency = 1/5)
    (h2 : teams.team_c_efficiency = 7/8)
    (h3 : teams.team_a_agents = 5/8)
    (h4 : teams.team_c_agents = 3/4) :
    total_calls_fraction teams = 19/32 := by
  sorry

end NUMINAMATH_CALUDE_call_center_efficiency_l4051_405187


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l4051_405144

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (2 : ℂ) / (1 + i) = 1 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l4051_405144


namespace NUMINAMATH_CALUDE_complement_of_event_A_l4051_405135

/-- The total number of products in the batch -/
def total_products : ℕ := 10

/-- Event A: There are at least 2 defective products -/
def event_A (defective : ℕ) : Prop := defective ≥ 2

/-- The complement of event A -/
def complement_A (defective : ℕ) : Prop := defective ≤ 1

/-- Theorem stating that the complement of event A is correctly defined -/
theorem complement_of_event_A :
  ∀ defective : ℕ, defective ≤ total_products →
    (¬ event_A defective ↔ complement_A defective) :=
by sorry

end NUMINAMATH_CALUDE_complement_of_event_A_l4051_405135


namespace NUMINAMATH_CALUDE_least_product_of_three_primes_over_50_l4051_405197

theorem least_product_of_three_primes_over_50 :
  ∃ (p q r : Nat),
    Prime p ∧ Prime q ∧ Prime r ∧
    p > 50 ∧ q > 50 ∧ r > 50 ∧
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    p * q * r = 190847 ∧
    ∀ (a b c : Nat),
      Prime a → Prime b → Prime c →
      a > 50 → b > 50 → c > 50 →
      a ≠ b → a ≠ c → b ≠ c →
      a * b * c ≥ 190847 :=
by
  sorry

end NUMINAMATH_CALUDE_least_product_of_three_primes_over_50_l4051_405197


namespace NUMINAMATH_CALUDE_valid_sequences_are_correct_l4051_405152

def is_valid_sequence (s : List ℕ) : Prop :=
  s.length = 8 ∧
  s.toFinset.card = 8 ∧
  (∀ x ∈ s, 1 ≤ x ∧ x ≤ 11) ∧
  (∀ n ∈ Finset.range 8, (s.take (n + 1)).sum % (n + 1) = 0)

def valid_sequences : List (List ℕ) :=
  [[1, 3, 2, 6, 8, 4, 11, 5],
   [3, 1, 2, 6, 8, 4, 11, 5],
   [2, 6, 1, 3, 8, 4, 11, 5],
   [6, 2, 1, 3, 8, 4, 11, 5],
   [9, 11, 10, 6, 4, 8, 1, 7],
   [11, 9, 10, 6, 4, 8, 1, 7],
   [10, 6, 11, 9, 4, 8, 1, 7],
   [6, 10, 11, 9, 4, 8, 1, 7]]

theorem valid_sequences_are_correct :
  ∀ s ∈ valid_sequences, is_valid_sequence s ∧
  ∀ s, is_valid_sequence s → s ∈ valid_sequences :=
by sorry

end NUMINAMATH_CALUDE_valid_sequences_are_correct_l4051_405152


namespace NUMINAMATH_CALUDE_product_ratio_theorem_l4051_405161

theorem product_ratio_theorem (a b c d e f : ℝ) 
  (h1 : a * b * c = 195)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_product_ratio_theorem_l4051_405161


namespace NUMINAMATH_CALUDE_disjoint_sets_imply_m_leq_neg_one_l4051_405149

def A : Set (ℝ × ℝ) := {p | p.2 = Real.log (p.1 + 1) - 1}

def B (m : ℝ) : Set (ℝ × ℝ) := {p | p.1 = m}

theorem disjoint_sets_imply_m_leq_neg_one (m : ℝ) :
  A ∩ B m = ∅ → m ≤ -1 := by sorry

end NUMINAMATH_CALUDE_disjoint_sets_imply_m_leq_neg_one_l4051_405149


namespace NUMINAMATH_CALUDE_percent_of_whole_l4051_405191

theorem percent_of_whole (part whole : ℝ) (h : whole ≠ 0) :
  (part / whole) * 100 = 50 ↔ part = (1/2) * whole :=
sorry

end NUMINAMATH_CALUDE_percent_of_whole_l4051_405191


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l4051_405138

theorem simplify_trig_expression : 
  Real.sqrt (1 - Real.sin (3 * Real.pi / 5) ^ 2) = -Real.cos (3 * Real.pi / 5) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l4051_405138


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l4051_405115

/-- A geometric sequence with common ratio r -/
def geometricSequence (r : ℝ) : ℕ → ℝ := fun n => r^(n-1)

/-- The third term of a geometric sequence -/
def a₃ (r : ℝ) : ℝ := geometricSequence r 3

/-- The seventh term of a geometric sequence -/
def a₇ (r : ℝ) : ℝ := geometricSequence r 7

/-- The fifth term of a geometric sequence -/
def a₅ (r : ℝ) : ℝ := geometricSequence r 5

theorem geometric_sequence_fifth_term (r : ℝ) :
  (a₃ r)^2 - 4*(a₃ r) + 3 = 0 ∧ (a₇ r)^2 - 4*(a₇ r) + 3 = 0 → a₅ r = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l4051_405115


namespace NUMINAMATH_CALUDE_certain_number_problem_l4051_405186

theorem certain_number_problem (x : ℝ) : 
  x - (1/4)*2 - (1/3)*3 - (1/7)*x = 27 → 
  (10/100) * x = 3.325 := by
sorry

end NUMINAMATH_CALUDE_certain_number_problem_l4051_405186


namespace NUMINAMATH_CALUDE_special_function_property_l4051_405185

/-- A function that is monotonically increasing on [0,2] and f(x+2) is even -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 2 → f x ≤ f y) ∧
  (∀ x, f (x + 2) = f (-x + 2))

theorem special_function_property (f : ℝ → ℝ) (h : special_function f) :
  f (7/2) < f 1 ∧ f 1 < f (5/2) := by
  sorry

end NUMINAMATH_CALUDE_special_function_property_l4051_405185


namespace NUMINAMATH_CALUDE_tens_digit_of_23_pow_2023_l4051_405124

theorem tens_digit_of_23_pow_2023 : ∃ n : ℕ, 23^2023 ≡ 60 + n [ZMOD 100] ∧ 0 ≤ n ∧ n < 10 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_23_pow_2023_l4051_405124


namespace NUMINAMATH_CALUDE_hexagon_dimension_theorem_l4051_405176

/-- Represents a hexagon that can be part of a rectangle and repositioned to form a square --/
structure Hexagon where
  area : ℝ
  significantDimension : ℝ

/-- Represents a rectangle that can be divided into two congruent hexagons --/
structure Rectangle where
  width : ℝ
  height : ℝ
  hexagons : Fin 2 → Hexagon
  isCongruent : hexagons 0 = hexagons 1

/-- Represents a square formed by repositioning two hexagons --/
structure Square where
  sideLength : ℝ

/-- Theorem stating the relationship between the rectangle, hexagons, and resulting square --/
theorem hexagon_dimension_theorem (rect : Rectangle) (sq : Square) : 
  rect.width = 9 ∧ 
  rect.height = 16 ∧ 
  (rect.width * rect.height = sq.sideLength * sq.sideLength) ∧
  (rect.hexagons 0).significantDimension = 6 := by
  sorry

#check hexagon_dimension_theorem

end NUMINAMATH_CALUDE_hexagon_dimension_theorem_l4051_405176
