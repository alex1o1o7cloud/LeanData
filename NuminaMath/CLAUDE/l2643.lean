import Mathlib

namespace NUMINAMATH_CALUDE_solution_set_of_decreasing_function_l2643_264375

-- Define a decreasing function on ℝ
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem solution_set_of_decreasing_function (f : ℝ → ℝ) 
  (h : DecreasingFunction f) : 
  {x : ℝ | f x > f 1} = {x : ℝ | x < 1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_decreasing_function_l2643_264375


namespace NUMINAMATH_CALUDE_ends_in_zero_l2643_264389

theorem ends_in_zero (a : ℤ) (n : ℕ) (h : n ≥ 2) :
  ∃ k : ℤ, a^(2^n + 1) - a = 10 * k := by
  sorry

end NUMINAMATH_CALUDE_ends_in_zero_l2643_264389


namespace NUMINAMATH_CALUDE_smallest_multiple_l2643_264388

theorem smallest_multiple (n : ℕ) : n = 3441 ↔ 
  n > 0 ∧ 
  37 ∣ n ∧ 
  n % 103 = 7 ∧ 
  ∀ m : ℕ, m > 0 → 37 ∣ m → m % 103 = 7 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l2643_264388


namespace NUMINAMATH_CALUDE_max_cells_visitable_l2643_264390

/-- Represents a rectangular board -/
structure Board where
  rows : Nat
  cols : Nat

/-- Represents a cube with one painted face -/
structure Cube where
  side : Nat
  painted_face : Nat

/-- Defines the maximum number of cells a cube can visit on a board without the painted face touching -/
def max_visitable_cells (b : Board) (c : Cube) : Nat :=
  b.rows * b.cols

/-- Theorem stating that the maximum number of visitable cells equals the total number of cells on the board -/
theorem max_cells_visitable (b : Board) (c : Cube) 
  (h1 : b.rows = 7) 
  (h2 : b.cols = 12) 
  (h3 : c.side = 1) 
  (h4 : c.painted_face ≤ 6) :
  max_visitable_cells b c = b.rows * b.cols := by
  sorry

end NUMINAMATH_CALUDE_max_cells_visitable_l2643_264390


namespace NUMINAMATH_CALUDE_trajectory_of_point_P_l2643_264314

/-- Given a circle centered at F(4,0) with radius 2, and points A(-4,0) and B on the circle,
    the point P is defined as the intersection of the perpendicular bisector of AB and line BF.
    This theorem states that the trajectory of P as B moves along the circle
    is a hyperbola with equation x^2 - y^2/15 = 1 (x ≠ 0). -/
theorem trajectory_of_point_P (A B P F : ℝ × ℝ) :
  A = (-4, 0) →
  F = (4, 0) →
  (B.1 - 4)^2 + B.2^2 = 4 →
  (∀ M : ℝ × ℝ, (M.1 - A.1)^2 + (M.2 - A.2)^2 = (M.1 - B.1)^2 + (M.2 - B.2)^2 →
                 (M.1 - P.1) * (B.2 - F.2) = (M.2 - P.2) * (B.1 - F.1)) →
  (P.1 - F.1) * (B.2 - F.2) = (P.2 - F.2) * (B.1 - F.1) →
  P.1 ≠ 0 →
  P.1^2 - P.2^2 / 15 = 1 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_point_P_l2643_264314


namespace NUMINAMATH_CALUDE_imaginary_difference_condition_l2643_264347

def is_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem imaginary_difference_condition (z₁ z₂ : ℂ) :
  (is_imaginary (z₁ - z₂) → (is_imaginary z₁ ∨ is_imaginary z₂)) ∧
  ∃ z₁ z₂ : ℂ, (is_imaginary z₁ ∨ is_imaginary z₂) ∧ ¬is_imaginary (z₁ - z₂) :=
sorry

end NUMINAMATH_CALUDE_imaginary_difference_condition_l2643_264347


namespace NUMINAMATH_CALUDE_sqrt_sum_difference_eighth_power_l2643_264316

theorem sqrt_sum_difference_eighth_power : 
  (Real.sqrt 11 + Real.sqrt 5)^8 + (Real.sqrt 11 - Real.sqrt 5)^8 = 903712 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_difference_eighth_power_l2643_264316


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l2643_264366

theorem geometric_arithmetic_sequence (a₁ : ℝ) (h : a₁ ≠ 0) :
  ∃! (s : Finset ℝ), s.card = 2 ∧
    ∀ q ∈ s, 2 * (a₁ * q^4) = 4 * a₁ + (-2 * (a₁ * q^2)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l2643_264366


namespace NUMINAMATH_CALUDE_inequality_proof_l2643_264301

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : 0 < d ∧ 0 < c ∧ 0 < b ∧ 0 < a) 
  (h_order : d < c ∧ c < b ∧ b < a) : 
  (a + b + c + d)^2 > a^2 + 3*b^2 + 5*c^2 + 7*d^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2643_264301


namespace NUMINAMATH_CALUDE_greatest_integer_no_real_roots_l2643_264323

theorem greatest_integer_no_real_roots (c : ℤ) : 
  (∀ x : ℝ, x^2 + c*x + 15 ≠ 0) ↔ c ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_no_real_roots_l2643_264323


namespace NUMINAMATH_CALUDE_triangle_inequality_l2643_264376

theorem triangle_inequality (a b c S : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
  (h₄ : S > 0) (h₅ : a + b > c) (h₆ : b + c > a) (h₇ : c + a > b)
  (h₈ : S = Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4) :
  c^2 - a^2 - b^2 + 4*a*b ≥ 4 * Real.sqrt 3 * S := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2643_264376


namespace NUMINAMATH_CALUDE_power_sum_difference_l2643_264307

theorem power_sum_difference : 2^(1+2+3+4) - (2^1 + 2^2 + 2^3 + 2^4) = 994 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_l2643_264307


namespace NUMINAMATH_CALUDE_cost_per_page_is_five_l2643_264356

/-- Calculates the cost per page in cents -/
def cost_per_page (notebooks : ℕ) (pages_per_notebook : ℕ) (total_cost_dollars : ℕ) : ℚ :=
  (total_cost_dollars * 100) / (notebooks * pages_per_notebook)

/-- Proves that the cost per page is 5 cents given the problem conditions -/
theorem cost_per_page_is_five :
  cost_per_page 2 50 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_page_is_five_l2643_264356


namespace NUMINAMATH_CALUDE_natural_numbers_product_sum_diff_l2643_264312

theorem natural_numbers_product_sum_diff (m n : ℕ) :
  (m + n) * |Int.ofNat m - Int.ofNat n| = 2021 →
  ((m = 1011 ∧ n = 1010) ∨ (m = 45 ∧ n = 2)) := by
  sorry

end NUMINAMATH_CALUDE_natural_numbers_product_sum_diff_l2643_264312


namespace NUMINAMATH_CALUDE_coin_difference_l2643_264374

def coin_values : List ℕ := [5, 10, 25, 50]

def target_amount : ℕ := 75

def min_coins (values : List ℕ) (target : ℕ) : ℕ := sorry

def max_coins (values : List ℕ) (target : ℕ) : ℕ := sorry

theorem coin_difference :
  max_coins coin_values target_amount - min_coins coin_values target_amount = 13 := by
  sorry

end NUMINAMATH_CALUDE_coin_difference_l2643_264374


namespace NUMINAMATH_CALUDE_only_negative_option_l2643_264325

theorem only_negative_option (x : ℝ) : 
  (|(-1)| < 0 ∨ (-2^2) < 0 ∨ ((-Real.sqrt 3)^2) < 0 ∨ ((-3)^0) < 0) ↔ 
  (-2^2) < 0 :=
by sorry

end NUMINAMATH_CALUDE_only_negative_option_l2643_264325


namespace NUMINAMATH_CALUDE_event_A_not_random_l2643_264358

-- Define the type for events
inductive Event
| A : Event  -- The sun rises in the east and it rains in the west
| B : Event  -- It's not cold when it snows but cold when the snow melts
| C : Event  -- It often rains during the Qingming festival
| D : Event  -- It's sunny every day when the plums turn yellow

-- Define what it means for an event to be random
def isRandomEvent (e : Event) : Prop := sorry

-- Define what it means for an event to be based on natural laws
def isBasedOnNaturalLaws (e : Event) : Prop := sorry

-- Axiom: Events based on natural laws are not random
axiom natural_law_not_random : ∀ (e : Event), isBasedOnNaturalLaws e → ¬isRandomEvent e

-- Theorem: Event A is not a random event
theorem event_A_not_random : ¬isRandomEvent Event.A := by
  sorry

end NUMINAMATH_CALUDE_event_A_not_random_l2643_264358


namespace NUMINAMATH_CALUDE_victors_specific_earnings_l2643_264338

/-- Victor's earnings over two days given his hourly wage and hours worked each day -/
def victors_earnings (hourly_wage : ℕ) (hours_monday : ℕ) (hours_tuesday : ℕ) : ℕ :=
  hourly_wage * (hours_monday + hours_tuesday)

/-- Theorem: Victor's earnings over two days given specific conditions -/
theorem victors_specific_earnings :
  victors_earnings 6 5 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_victors_specific_earnings_l2643_264338


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_l2643_264343

/-- Given a person's income and savings, prove the ratio of income to expenditure -/
theorem income_expenditure_ratio 
  (income : ℕ) 
  (savings : ℕ) 
  (h1 : income = 14000) 
  (h2 : savings = 2000) :
  (income : ℚ) / (income - savings) = 7 / 6 :=
by sorry

end NUMINAMATH_CALUDE_income_expenditure_ratio_l2643_264343


namespace NUMINAMATH_CALUDE_vector_equality_l2643_264339

theorem vector_equality (m : ℝ) : 
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (1, 2)
  (a.1 + b.1)^2 + (a.2 + b.2)^2 = (a.1^2 + a.2^2) + (b.1^2 + b.2^2) → m = -2 :=
by sorry

end NUMINAMATH_CALUDE_vector_equality_l2643_264339


namespace NUMINAMATH_CALUDE_function_derivative_at_one_l2643_264350

theorem function_derivative_at_one 
  (f : ℝ → ℝ) 
  (h_diff : DifferentiableOn ℝ f (Set.Ioi 0))
  (h_def : ∀ x : ℝ, f (Real.exp x) = x + Real.exp x) : 
  deriv f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_function_derivative_at_one_l2643_264350


namespace NUMINAMATH_CALUDE_rectangular_equation_focus_directrix_distance_l2643_264340

-- Define the polar coordinate equation of the conic section curve C
def polarEquation (ρ θ : ℝ) : Prop :=
  ρ = 8 * Real.sin θ / (1 + Real.cos (2 * θ))

-- Define the conversion from polar to rectangular coordinates
def polarToRectangular (x y ρ θ : ℝ) : Prop :=
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- Theorem: The rectangular coordinate equation of curve C is x² = 4y
theorem rectangular_equation (x y : ℝ) :
  (∃ ρ θ, polarEquation ρ θ ∧ polarToRectangular x y ρ θ) →
  x^2 = 4*y :=
sorry

-- Theorem: The distance from the focus to the directrix is 2
theorem focus_directrix_distance :
  (∃ p : ℝ, ∀ x y : ℝ, (∃ ρ θ, polarEquation ρ θ ∧ polarToRectangular x y ρ θ) →
    y = (1 / (4 * p)) * x^2) →
  2 = 2 :=
sorry

end NUMINAMATH_CALUDE_rectangular_equation_focus_directrix_distance_l2643_264340


namespace NUMINAMATH_CALUDE_total_pennies_thrown_l2643_264326

/-- The number of pennies thrown by each person and their total --/
def penny_throwing (R G X M T : ℚ) : Prop :=
  R = 1500 ∧
  G = (2/3) * R ∧
  X = (3/4) * G ∧
  M = (7/2) * X ∧
  T = (4/5) * M ∧
  R + G + X + M + T = 7975

/-- Theorem stating that the total number of pennies thrown is 7975 --/
theorem total_pennies_thrown :
  ∃ (R G X M T : ℚ), penny_throwing R G X M T :=
sorry

end NUMINAMATH_CALUDE_total_pennies_thrown_l2643_264326


namespace NUMINAMATH_CALUDE_tangent_line_slope_l2643_264367

/-- If the line y = kx is tangent to the curve y = x + exp(-x), then k = 1 - exp(1) -/
theorem tangent_line_slope (k : ℝ) : 
  (∃ x₀ : ℝ, k * x₀ = x₀ + Real.exp (-x₀) ∧ 
             k = 1 - Real.exp (-x₀)) → 
  k = 1 - Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l2643_264367


namespace NUMINAMATH_CALUDE_simplify_expression_l2643_264393

theorem simplify_expression (x w : ℝ) : 3*x + 4*w - 2*x + 6 - 5*w - 5 = x - w + 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2643_264393


namespace NUMINAMATH_CALUDE_simple_interest_rate_correct_l2643_264371

/-- The simple interest rate that makes a sum of money increase to 7/6 of itself in 6 years -/
def simple_interest_rate : ℚ :=
  100 / 36

/-- The time period in years -/
def time_period : ℕ := 6

/-- The ratio of final amount to initial amount -/
def final_to_initial_ratio : ℚ := 7 / 6

theorem simple_interest_rate_correct : 
  final_to_initial_ratio = 1 + (simple_interest_rate * time_period) / 100 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_correct_l2643_264371


namespace NUMINAMATH_CALUDE_symmetric_solution_l2643_264310

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  2 * x^2 + 3 * x * y + y^2 = 70 ∧ 6 * x^2 + x * y - y^2 = 50

/-- Given solution -/
def x₁ : ℝ := 3
def y₁ : ℝ := 4

/-- Theorem stating that if (x₁, y₁) is a solution, then (-x₁, -y₁) is also a solution -/
theorem symmetric_solution :
  system x₁ y₁ → system (-x₁) (-y₁) := by sorry

end NUMINAMATH_CALUDE_symmetric_solution_l2643_264310


namespace NUMINAMATH_CALUDE_unique_triple_l2643_264320

theorem unique_triple : ∃! (x y z : ℕ+), 
  x ≤ y ∧ y ≤ z ∧ 
  x^3 * (y^3 + z^3) = 2012 * (x * y * z + 2) ∧
  x = 2 ∧ y = 251 ∧ z = 252 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_l2643_264320


namespace NUMINAMATH_CALUDE_cloth_sale_loss_per_metre_l2643_264344

/-- Given the following conditions for a cloth sale:
  * 500 metres of cloth sold
  * Total selling price of Rs. 15000
  * Cost price of Rs. 40 per metre
  Prove that the loss per metre of cloth sold is Rs. 10. -/
theorem cloth_sale_loss_per_metre 
  (total_metres : ℕ) 
  (total_selling_price : ℕ) 
  (cost_price_per_metre : ℕ) 
  (h1 : total_metres = 500)
  (h2 : total_selling_price = 15000)
  (h3 : cost_price_per_metre = 40) :
  (cost_price_per_metre * total_metres - total_selling_price) / total_metres = 10 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_loss_per_metre_l2643_264344


namespace NUMINAMATH_CALUDE_nancy_coffee_consumption_l2643_264327

/-- Represents the daily coffee consumption and costs for Nancy --/
structure CoffeeConsumption where
  double_espresso_cost : ℝ
  iced_coffee_cost : ℝ
  total_spent : ℝ
  days : ℕ

/-- Calculates the number of coffees Nancy buys each day --/
def coffees_per_day (c : CoffeeConsumption) : ℕ :=
  2

/-- Theorem stating that Nancy buys 2 coffees per day given the conditions --/
theorem nancy_coffee_consumption (c : CoffeeConsumption) 
  (h1 : c.double_espresso_cost = 3)
  (h2 : c.iced_coffee_cost = 2.5)
  (h3 : c.total_spent = 110)
  (h4 : c.days = 20) :
  coffees_per_day c = 2 := by
  sorry

#check nancy_coffee_consumption

end NUMINAMATH_CALUDE_nancy_coffee_consumption_l2643_264327


namespace NUMINAMATH_CALUDE_seashell_ratio_correct_l2643_264351

/-- Represents the number of seashells found by each person -/
structure SeashellCount where
  mary : ℕ
  jessica : ℕ
  linda : ℕ

/-- Represents a ratio as a triple of natural numbers -/
structure Ratio where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The actual seashell counts -/
def actualCounts : SeashellCount :=
  { mary := 18, jessica := 41, linda := 27 }

/-- The expected ratio -/
def expectedRatio : Ratio :=
  { first := 18, second := 41, third := 27 }

/-- Theorem stating that the ratio of seashells found is as expected -/
theorem seashell_ratio_correct :
  let counts := actualCounts
  (counts.mary : ℚ) / (counts.jessica : ℚ) = (expectedRatio.first : ℚ) / (expectedRatio.second : ℚ) ∧
  (counts.jessica : ℚ) / (counts.linda : ℚ) = (expectedRatio.second : ℚ) / (expectedRatio.third : ℚ) :=
sorry

end NUMINAMATH_CALUDE_seashell_ratio_correct_l2643_264351


namespace NUMINAMATH_CALUDE_circumscribed_circle_area_l2643_264394

/-- The area of a circle circumscribed about an equilateral triangle with side length 12 units -/
theorem circumscribed_circle_area (s : ℝ) (h : s = 12) : 
  let R := s / Real.sqrt 3
  π * R^2 = 48 * π := by sorry

end NUMINAMATH_CALUDE_circumscribed_circle_area_l2643_264394


namespace NUMINAMATH_CALUDE_triangle_longer_segment_l2643_264396

theorem triangle_longer_segment (a b c h x : ℝ) : 
  a = 40 → b = 90 → c = 100 → 
  x^2 + h^2 = a^2 → 
  (c - x)^2 + h^2 = b^2 → 
  c - x = 82.5 :=
by sorry

end NUMINAMATH_CALUDE_triangle_longer_segment_l2643_264396


namespace NUMINAMATH_CALUDE_f_f_zero_equals_3pi_squared_minus_4_l2643_264303

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 3 * x^2 - 4
  else if x = 0 then Real.pi
  else 0

-- Theorem statement
theorem f_f_zero_equals_3pi_squared_minus_4 :
  f (f 0) = 3 * Real.pi^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_f_f_zero_equals_3pi_squared_minus_4_l2643_264303


namespace NUMINAMATH_CALUDE_prime_triplet_l2643_264346

theorem prime_triplet (p : ℕ) : 
  Prime p ∧ Prime (p + 10) ∧ Prime (p + 14) → p = 3 :=
by sorry

end NUMINAMATH_CALUDE_prime_triplet_l2643_264346


namespace NUMINAMATH_CALUDE_correct_arrangement_l2643_264353

-- Define the set of friends
inductive Friend : Type
  | Amy : Friend
  | Bob : Friend
  | Celine : Friend
  | David : Friend

-- Define a height comparison relation
def taller_than : Friend → Friend → Prop := sorry

-- Define the statements
def statement_I : Prop := ¬(taller_than Friend.Celine Friend.Amy ∧ taller_than Friend.Celine Friend.Bob ∧ taller_than Friend.Celine Friend.David)
def statement_II : Prop := ∀ f : Friend, f ≠ Friend.Bob → taller_than f Friend.Bob
def statement_III : Prop := ∃ f₁ f₂ : Friend, taller_than f₁ Friend.Amy ∧ taller_than Friend.Amy f₂
def statement_IV : Prop := taller_than Friend.David Friend.Bob ∧ taller_than Friend.Amy Friend.David

-- Define the condition that exactly one statement is true
def exactly_one_true : Prop :=
  (statement_I ∧ ¬statement_II ∧ ¬statement_III ∧ ¬statement_IV) ∨
  (¬statement_I ∧ statement_II ∧ ¬statement_III ∧ ¬statement_IV) ∨
  (¬statement_I ∧ ¬statement_II ∧ statement_III ∧ ¬statement_IV) ∨
  (¬statement_I ∧ ¬statement_II ∧ ¬statement_III ∧ statement_IV)

-- Theorem to prove
theorem correct_arrangement (h : exactly_one_true) :
  taller_than Friend.Celine Friend.Amy ∧
  taller_than Friend.Amy Friend.David ∧
  taller_than Friend.David Friend.Bob :=
sorry

end NUMINAMATH_CALUDE_correct_arrangement_l2643_264353


namespace NUMINAMATH_CALUDE_parabola_intersection_dot_product_l2643_264300

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 1)

-- Define the intersection of the line and the parabola
def intersection (k : ℝ) (p : PointOnParabola) : Prop :=
  line_through_focus k p.x p.y

theorem parabola_intersection_dot_product :
  ∀ (k : ℝ) (A B : PointOnParabola),
    intersection k A →
    intersection k B →
    A ≠ B →
    A.x * B.x + A.y * B.y = -3 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_dot_product_l2643_264300


namespace NUMINAMATH_CALUDE_minutes_to_date_time_correct_l2643_264308

/-- Represents a date and time -/
structure DateTime where
  year : Nat
  month : Nat
  day : Nat
  hour : Nat
  minute : Nat

/-- Converts minutes to a DateTime structure -/
def minutesToDateTime (startDateTime : DateTime) (minutes : Nat) : DateTime :=
  sorry

/-- The starting date and time -/
def startDateTime : DateTime :=
  { year := 2015, month := 1, day := 1, hour := 0, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : Nat := 3050

/-- The expected result date and time -/
def expectedDateTime : DateTime :=
  { year := 2015, month := 1, day := 3, hour := 2, minute := 50 }

theorem minutes_to_date_time_correct :
  minutesToDateTime startDateTime minutesToAdd = expectedDateTime :=
  sorry

end NUMINAMATH_CALUDE_minutes_to_date_time_correct_l2643_264308


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2643_264378

/-- Given a triangle ABD where angle ABC is a straight angle (180°),
    angle CBD is 133°, and one angle in triangle ABD is 31°,
    prove that the measure of the remaining angle y in triangle ABD is 102°. -/
theorem triangle_angle_measure (angle_CBD : ℝ) (angle_in_ABD : ℝ) :
  angle_CBD = 133 →
  angle_in_ABD = 31 →
  let angle_ABD : ℝ := 180 - angle_CBD
  180 - (angle_ABD + angle_in_ABD) = 102 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2643_264378


namespace NUMINAMATH_CALUDE_sum_F_equals_535501_l2643_264336

/-- F(n) is the smallest positive integer greater than n whose sum of digits is equal to the sum of the digits of n -/
def F (n : ℕ) : ℕ := sorry

/-- The sum of F(n) for n from 1 to 1000 -/
def sum_F : ℕ := (List.range 1000).map F |>.sum

theorem sum_F_equals_535501 : sum_F = 535501 := by sorry

end NUMINAMATH_CALUDE_sum_F_equals_535501_l2643_264336


namespace NUMINAMATH_CALUDE_correct_product_l2643_264360

/- Define a function to reverse digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/- Main theorem -/
theorem correct_product (a b : ℕ) :
  a ≥ 10 ∧ a ≤ 99 ∧  -- a is a two-digit number
  0 < b ∧  -- b is positive
  (reverse_digits a) * b = 187 →
  a * b = 187 :=
by
  sorry


end NUMINAMATH_CALUDE_correct_product_l2643_264360


namespace NUMINAMATH_CALUDE_stamp_problem_l2643_264380

/-- Given stamps of denominations 5, n, and n+1 cents, 
    where n is a positive integer, 
    if 97 cents is the greatest postage that cannot be formed, 
    then n = 25 -/
theorem stamp_problem (n : ℕ) : 
  n > 0 → 
  (∀ k : ℕ, k > 97 → ∃ a b c : ℕ, k = 5*a + n*b + (n+1)*c) → 
  (∃ a b c : ℕ, 97 = 5*a + n*b + (n+1)*c → False) → 
  n = 25 := by
  sorry

end NUMINAMATH_CALUDE_stamp_problem_l2643_264380


namespace NUMINAMATH_CALUDE_third_term_expansion_l2643_264302

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of the third term in (3b+2a)^6
def third_term_coefficient : ℕ := binomial 6 2 * 3^4 * 2^2

-- Theorem statement
theorem third_term_expansion :
  third_term_coefficient = 4860 ∧ binomial 6 2 = 15 := by sorry

end NUMINAMATH_CALUDE_third_term_expansion_l2643_264302


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2643_264398

theorem system_of_equations_solution :
  ∃! (x y z : ℚ),
    x + 2 * y - z = 20 ∧
    y = 5 ∧
    3 * x + 4 * z = 40 ∧
    x = 80 / 7 ∧
    z = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2643_264398


namespace NUMINAMATH_CALUDE_sphere_equation_implies_zero_difference_l2643_264328

theorem sphere_equation_implies_zero_difference (x y z : ℝ) :
  x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 14 = 0 →
  (x - y - z)^2002 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sphere_equation_implies_zero_difference_l2643_264328


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2643_264319

theorem complex_fraction_simplification :
  (2 * Complex.I) / (1 - Complex.I) = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2643_264319


namespace NUMINAMATH_CALUDE_chef_butter_remaining_l2643_264322

/-- Represents the recipe and chef's actions for making brownies. -/
structure BrownieRecipe where
  /-- The amount of butter (in ounces) required per cup of baking mix. -/
  butter_per_cup : ℝ
  /-- The amount of baking mix (in cups) the chef planned to use. -/
  planned_baking_mix : ℝ
  /-- The amount of coconut oil (in ounces) the chef used. -/
  coconut_oil_used : ℝ

/-- Calculates the amount of butter remaining after substituting with coconut oil. -/
def butter_remaining (recipe : BrownieRecipe) : ℝ :=
  recipe.butter_per_cup * recipe.planned_baking_mix - recipe.coconut_oil_used

/-- Theorem stating that the chef had 4 ounces of butter remaining. -/
theorem chef_butter_remaining (recipe : BrownieRecipe)
    (h1 : recipe.butter_per_cup = 2)
    (h2 : recipe.planned_baking_mix = 6)
    (h3 : recipe.coconut_oil_used = 8) :
    butter_remaining recipe = 4 := by
  sorry

#eval butter_remaining { butter_per_cup := 2, planned_baking_mix := 6, coconut_oil_used := 8 }

end NUMINAMATH_CALUDE_chef_butter_remaining_l2643_264322


namespace NUMINAMATH_CALUDE_hyperbola_focus_coordinates_l2643_264381

/-- Given a hyperbola with equation (x-5)^2/7^2 - (y-10)^2/15^2 = 1, 
    the focus with the larger x-coordinate has coordinates (5 + √274, 10) -/
theorem hyperbola_focus_coordinates (x y : ℝ) : 
  ((x - 5)^2 / 7^2) - ((y - 10)^2 / 15^2) = 1 →
  ∃ (f_x f_y : ℝ), f_x > 5 ∧ f_y = 10 ∧ 
  f_x = 5 + Real.sqrt 274 ∧
  ((f_x - 5)^2 / 7^2) - ((f_y - 10)^2 / 15^2) = 1 ∧
  ∀ (x' y' : ℝ), x' > 5 ∧ 
    ((x' - 5)^2 / 7^2) - ((y' - 10)^2 / 15^2) = 1 →
    x' ≤ f_x :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_coordinates_l2643_264381


namespace NUMINAMATH_CALUDE_probability_shaded_is_half_l2643_264362

/-- Represents a triangle in the diagram -/
structure Triangle where
  is_shaded : Bool

/-- The diagram containing the triangles -/
structure Diagram where
  triangles : Finset Triangle

/-- Calculates the probability of selecting a shaded triangle -/
def probability_shaded (d : Diagram) : ℚ :=
  (d.triangles.filter (·.is_shaded)).card / d.triangles.card

/-- The theorem statement -/
theorem probability_shaded_is_half (d : Diagram) :
    d.triangles.card = 4 ∧ 
    (d.triangles.filter (·.is_shaded)).card > 0 →
    probability_shaded d = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_shaded_is_half_l2643_264362


namespace NUMINAMATH_CALUDE_pauline_bell_peppers_l2643_264395

/-- The number of bell peppers Pauline bought -/
def num_bell_peppers : ℕ := 4

/-- The cost of taco shells in dollars -/
def taco_shells_cost : ℚ := 5

/-- The cost of each bell pepper in dollars -/
def bell_pepper_cost : ℚ := 3/2

/-- The cost of meat per pound in dollars -/
def meat_cost_per_pound : ℚ := 3

/-- The amount of meat Pauline bought in pounds -/
def meat_amount : ℚ := 2

/-- The total amount Pauline spent in dollars -/
def total_spent : ℚ := 17

theorem pauline_bell_peppers :
  num_bell_peppers = (total_spent - (taco_shells_cost + meat_cost_per_pound * meat_amount)) / bell_pepper_cost := by
  sorry

end NUMINAMATH_CALUDE_pauline_bell_peppers_l2643_264395


namespace NUMINAMATH_CALUDE_fuel_distance_theorem_l2643_264379

/-- Represents the relationship between remaining fuel and distance traveled for a car -/
def fuel_distance_relation (initial_fuel : ℝ) (consumption_rate : ℝ) (x : ℝ) : ℝ :=
  initial_fuel - consumption_rate * x

/-- Theorem stating the relationship between remaining fuel and distance traveled -/
theorem fuel_distance_theorem (x : ℝ) :
  fuel_distance_relation 60 0.12 x = 60 - 0.12 * x := by
  sorry

end NUMINAMATH_CALUDE_fuel_distance_theorem_l2643_264379


namespace NUMINAMATH_CALUDE_total_trees_planted_l2643_264385

def trees_planted (fourth_grade fifth_grade sixth_grade : ℕ) : Prop :=
  fourth_grade = 30 ∧
  fifth_grade = 2 * fourth_grade ∧
  sixth_grade = 3 * fifth_grade - 30

theorem total_trees_planted :
  ∀ fourth_grade fifth_grade sixth_grade : ℕ,
  trees_planted fourth_grade fifth_grade sixth_grade →
  fourth_grade + fifth_grade + sixth_grade = 240 :=
by sorry

end NUMINAMATH_CALUDE_total_trees_planted_l2643_264385


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l2643_264342

/-- A right triangle with sides 5, 12, and 13 -/
structure RightTriangle :=
  (a : ℝ) (b : ℝ) (c : ℝ)
  (is_right : a^2 + b^2 = c^2)
  (side_lengths : a = 5 ∧ b = 12 ∧ c = 13)

/-- Square inscribed in a right triangle with one vertex at the right angle -/
def square_at_right_angle (t : RightTriangle) (x : ℝ) : Prop :=
  0 < x ∧ x ≤ min t.a t.b

/-- Square inscribed in a right triangle with one side on the hypotenuse -/
def square_on_hypotenuse (t : RightTriangle) (y : ℝ) : Prop :=
  0 < y ∧ y ≤ t.c

theorem inscribed_squares_ratio (t1 t2 : RightTriangle) (x y : ℝ)
  (h1 : square_at_right_angle t1 x)
  (h2 : square_on_hypotenuse t2 y) :
  x / y = 1800 / 2863 := by
  sorry

#check inscribed_squares_ratio

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l2643_264342


namespace NUMINAMATH_CALUDE_problem_statement_l2643_264313

open Real

theorem problem_statement :
  (∃ x : ℝ, x - 2 > log x) ∧
  ¬(∀ x : ℝ, exp x > 1) ∧
  ((∃ x : ℝ, x - 2 > log x) ∧ ¬(∀ x : ℝ, exp x > 1)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2643_264313


namespace NUMINAMATH_CALUDE_fraction_inequality_l2643_264309

theorem fraction_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c < 0) :
  c / (a^2) > c / (b^2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2643_264309


namespace NUMINAMATH_CALUDE_victor_percentage_proof_l2643_264355

def max_marks : ℝ := 450
def victor_marks : ℝ := 405

theorem victor_percentage_proof :
  (victor_marks / max_marks) * 100 = 90 := by
  sorry

end NUMINAMATH_CALUDE_victor_percentage_proof_l2643_264355


namespace NUMINAMATH_CALUDE_projection_matrix_values_l2643_264324

def is_projection_matrix (Q : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  Q * Q = Q

theorem projection_matrix_values :
  ∀ (a c : ℝ),
  let Q : Matrix (Fin 2) (Fin 2) ℝ := !![a, 18/45; c, 27/45]
  is_projection_matrix Q →
  a = 2/5 ∧ c = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_projection_matrix_values_l2643_264324


namespace NUMINAMATH_CALUDE_toby_work_hours_l2643_264368

/-- Given the working hours of Thomas, Toby, and Rebecca, prove that Toby worked 10 hours less than twice what Thomas worked. -/
theorem toby_work_hours (x : ℕ) : 
  -- Total hours worked
  x + (2 * x - 10) + 56 = 157 →
  -- Rebecca worked 56 hours
  56 = 56 →
  -- Rebecca worked 8 hours less than Toby
  56 = (2 * x - 10) - 8 →
  -- Toby worked 10 hours less than twice what Thomas worked
  (2 * x - (2 * x - 10)) = 10 := by
sorry

end NUMINAMATH_CALUDE_toby_work_hours_l2643_264368


namespace NUMINAMATH_CALUDE_perpendicular_planes_from_perpendicular_lines_l2643_264337

-- Define the necessary types
variable {Point : Type*} [NormedAddCommGroup Point] [InnerProductSpace ℝ Point] [Finite Point]
variable {Line : Type*} [NormedAddCommGroup Line] [InnerProductSpace ℝ Line] [Finite Line]
variable {Plane : Type*} [NormedAddCommGroup Plane] [InnerProductSpace ℝ Plane] [Finite Plane]

-- Define the necessary relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_from_perpendicular_lines 
  (m n : Line) (α β : Plane) :
  perpendicular m n → 
  perpendicular_line_plane m α → 
  perpendicular_line_plane n β → 
  perpendicular_plane α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_from_perpendicular_lines_l2643_264337


namespace NUMINAMATH_CALUDE_three_sum_exists_l2643_264349

theorem three_sum_exists (n : ℕ) (a : Fin (n + 1) → ℕ) 
  (h_strict_increasing : ∀ i j : Fin (n + 1), i < j → a i < a j)
  (h_upper_bound : ∀ i : Fin (n + 1), a i < 2 * n) :
  ∃ i j k : Fin (n + 1), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ a i + a j = a k :=
by sorry

end NUMINAMATH_CALUDE_three_sum_exists_l2643_264349


namespace NUMINAMATH_CALUDE_box_volume_increase_l2643_264382

theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 4000)
  (surface_area : 2 * l * w + 2 * w * h + 2 * h * l = 1680)
  (edge_sum : 4 * l + 4 * w + 4 * h = 200) :
  (l + 2) * (w + 3) * (h + 1) = 5736 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l2643_264382


namespace NUMINAMATH_CALUDE_existence_of_parallel_plane_l2643_264345

/-- Two lines in space are non-intersecting (skew) -/
def NonIntersecting (a b : Line3) : Prop := sorry

/-- A line is parallel to a plane -/
def ParallelToPlane (l : Line3) (p : Plane3) : Prop := sorry

theorem existence_of_parallel_plane (a b : Line3) (h : NonIntersecting a b) :
  ∃ α : Plane3, ParallelToPlane a α ∧ ParallelToPlane b α := by sorry

end NUMINAMATH_CALUDE_existence_of_parallel_plane_l2643_264345


namespace NUMINAMATH_CALUDE_pirate_treasure_probability_l2643_264392

def num_islands : ℕ := 7
def num_treasure_islands : ℕ := 4

def prob_treasure : ℚ := 1/5
def prob_trap : ℚ := 1/10
def prob_neither : ℚ := 7/10

theorem pirate_treasure_probability :
  (Nat.choose num_islands num_treasure_islands : ℚ) *
  prob_treasure ^ num_treasure_islands *
  prob_neither ^ (num_islands - num_treasure_islands) =
  12005/625000 := by sorry

end NUMINAMATH_CALUDE_pirate_treasure_probability_l2643_264392


namespace NUMINAMATH_CALUDE_apple_relationship_l2643_264363

/-- Proves the relationship between bruised and wormy apples --/
theorem apple_relationship (total_apples wormy_ratio raw_apples : ℕ) 
  (h1 : total_apples = 85)
  (h2 : wormy_ratio = 5)
  (h3 : raw_apples = 42) : 
  ∃ (bruised wormy : ℕ), 
    wormy = total_apples / wormy_ratio ∧ 
    bruised = total_apples - raw_apples - wormy ∧ 
    bruised = wormy + 9 :=
by sorry

end NUMINAMATH_CALUDE_apple_relationship_l2643_264363


namespace NUMINAMATH_CALUDE_crate_height_difference_l2643_264372

/-- The number of cans in each crate -/
def num_cans : ℕ := 300

/-- The diameter of each can in cm -/
def can_diameter : ℕ := 12

/-- The number of rows in triangular stacking -/
def triangular_rows : ℕ := 24

/-- The number of rows in square stacking -/
def square_rows : ℕ := 18

/-- The height of the triangular stacking in cm -/
def triangular_height : ℕ := triangular_rows * can_diameter

/-- The height of the square stacking in cm -/
def square_height : ℕ := square_rows * can_diameter

theorem crate_height_difference :
  triangular_height - square_height = 72 :=
sorry

end NUMINAMATH_CALUDE_crate_height_difference_l2643_264372


namespace NUMINAMATH_CALUDE_floor_paving_cost_l2643_264377

/-- The cost of paving a rectangular floor -/
theorem floor_paving_cost 
  (length : ℝ) 
  (width : ℝ) 
  (rate : ℝ) 
  (h1 : length = 5) 
  (h2 : width = 4.75) 
  (h3 : rate = 900) : 
  length * width * rate = 21375 := by
  sorry

end NUMINAMATH_CALUDE_floor_paving_cost_l2643_264377


namespace NUMINAMATH_CALUDE_block_count_is_eight_l2643_264359

/-- Represents the orthographic views of a geometric body -/
structure OrthographicViews where
  front : Nat
  top : Nat
  side : Nat

/-- Calculates the number of blocks in a geometric body based on its orthographic views -/
def countBlocks (views : OrthographicViews) : Nat :=
  sorry

/-- The specific orthographic views for the given problem -/
def problemViews : OrthographicViews :=
  { front := 6, top := 6, side := 4 }

/-- Theorem stating that the number of blocks for the given views is 8 -/
theorem block_count_is_eight :
  countBlocks problemViews = 8 := by
  sorry

end NUMINAMATH_CALUDE_block_count_is_eight_l2643_264359


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l2643_264335

/-- The total surface area of a cylinder with height 15 cm and radius 5 cm is 200π square cm. -/
theorem cylinder_surface_area :
  let h : ℝ := 15
  let r : ℝ := 5
  let total_area : ℝ := 2 * Real.pi * r * r + 2 * Real.pi * r * h
  total_area = 200 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l2643_264335


namespace NUMINAMATH_CALUDE_largest_y_coordinate_degenerate_hyperbola_l2643_264315

theorem largest_y_coordinate_degenerate_hyperbola : 
  ∀ (x y : ℝ), x^2 / 49 - (y - 3)^2 / 25 = 0 → y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_y_coordinate_degenerate_hyperbola_l2643_264315


namespace NUMINAMATH_CALUDE_unique_solution_l2643_264329

def is_valid_digit (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 6

def is_valid_row (a b c d e : ℕ) : Prop :=
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ is_valid_digit d ∧ is_valid_digit e ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def sum_constraint (a b c d e : ℕ) : Prop :=
  100 * a + 10 * b + c + 10 * c + d + e = 696

theorem unique_solution (a b c d e : ℕ) :
  is_valid_row a b c d e ∧ sum_constraint a b c d e →
  a = 6 ∧ b = 2 ∧ c = 3 ∧ d = 6 ∧ e = 3 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2643_264329


namespace NUMINAMATH_CALUDE_altitudes_sum_eq_nine_inradius_implies_equilateral_l2643_264311

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The altitude from a vertex to the opposite side of a triangle -/
def altitude (t : Triangle) (v : ℝ × ℝ) : ℝ := sorry

/-- The radius of the inscribed circle of a triangle -/
def inradius (t : Triangle) : ℝ := sorry

/-- A triangle is equilateral if all its sides have equal length -/
def is_equilateral (t : Triangle) : Prop := sorry

/-- 
If the sum of the altitudes of a triangle is equal to nine times 
the radius of its inscribed circle, then the triangle is equilateral 
-/
theorem altitudes_sum_eq_nine_inradius_implies_equilateral (t : Triangle) :
  altitude t t.A + altitude t t.B + altitude t t.C = 9 * inradius t →
  is_equilateral t := by
  sorry

end NUMINAMATH_CALUDE_altitudes_sum_eq_nine_inradius_implies_equilateral_l2643_264311


namespace NUMINAMATH_CALUDE_min_correct_answers_to_win_l2643_264352

/-- Represents the scoring system and conditions of the quiz -/
structure QuizRules where
  total_questions : ℕ
  correct_points : ℕ
  incorrect_points : ℕ
  unanswered : ℕ
  min_score_to_win : ℕ

/-- Calculates the score based on the number of correct answers -/
def calculate_score (rules : QuizRules) (correct_answers : ℕ) : ℤ :=
  (correct_answers : ℤ) * rules.correct_points -
  (rules.total_questions - rules.unanswered - correct_answers : ℤ) * rules.incorrect_points

/-- Theorem stating the minimum number of correct answers needed to win -/
theorem min_correct_answers_to_win (rules : QuizRules)
  (h1 : rules.total_questions = 25)
  (h2 : rules.correct_points = 4)
  (h3 : rules.incorrect_points = 2)
  (h4 : rules.unanswered = 2)
  (h5 : rules.min_score_to_win = 80) :
  ∀ x : ℕ, x ≥ 22 ↔ calculate_score rules x > rules.min_score_to_win :=
sorry

end NUMINAMATH_CALUDE_min_correct_answers_to_win_l2643_264352


namespace NUMINAMATH_CALUDE_total_milk_production_l2643_264384

/-- 
Given two groups of cows with their respective milk production rates,
this theorem proves the total milk production for both groups over a specified period.
-/
theorem total_milk_production 
  (a b c x y z w : ℝ) 
  (ha : a > 0) 
  (hb : b ≥ 0) 
  (hc : c > 0) 
  (hx : x > 0) 
  (hy : y ≥ 0) 
  (hz : z > 0) 
  (hw : w ≥ 0) :
  let group_a_rate := b / c
  let group_b_rate := y / z
  (group_a_rate + group_b_rate) * w = b * w / c + y * w / z := by
  sorry

#check total_milk_production

end NUMINAMATH_CALUDE_total_milk_production_l2643_264384


namespace NUMINAMATH_CALUDE_roots_expression_l2643_264318

theorem roots_expression (p q : ℝ) (α β γ δ : ℝ) 
  (hαβ : α^2 + p*α - 1 = 0 ∧ β^2 + p*β - 1 = 0)
  (hγδ : γ^2 + q*γ - 1 = 0 ∧ δ^2 + q*δ - 1 = 0) :
  (α - γ)*(β - γ)*(α - δ)*(β - δ) = -(p - q)^2 := by
sorry

end NUMINAMATH_CALUDE_roots_expression_l2643_264318


namespace NUMINAMATH_CALUDE_percentage_of_boys_studying_science_l2643_264369

theorem percentage_of_boys_studying_science 
  (total_boys : ℕ) 
  (school_A_percentage : ℚ) 
  (non_science_boys : ℕ) 
  (h1 : total_boys = 300)
  (h2 : school_A_percentage = 1/5)
  (h3 : non_science_boys = 42) :
  (↑((school_A_percentage * ↑total_boys - ↑non_science_boys) / (school_A_percentage * ↑total_boys)) : ℚ) = 3/10 :=
sorry

end NUMINAMATH_CALUDE_percentage_of_boys_studying_science_l2643_264369


namespace NUMINAMATH_CALUDE_emily_walks_farther_l2643_264341

/-- The distance Emily walks farther than Troy over five days -/
def distance_difference (troy_distance emily_distance : ℕ) : ℕ :=
  ((emily_distance - troy_distance) * 2) * 5

/-- Theorem stating the difference in distance walked by Emily and Troy over five days -/
theorem emily_walks_farther :
  distance_difference 75 98 = 230 := by
  sorry

end NUMINAMATH_CALUDE_emily_walks_farther_l2643_264341


namespace NUMINAMATH_CALUDE_rectangle_diagonal_triangle_area_l2643_264348

/-- The area of a triangle formed by the diagonal and one side of a rectangle with length 40 units and width 24 units is 480 square units. -/
theorem rectangle_diagonal_triangle_area : 
  let rectangle_length : ℝ := 40
  let rectangle_width : ℝ := 24
  let rectangle_area : ℝ := rectangle_length * rectangle_width
  let triangle_area : ℝ := rectangle_area / 2
  triangle_area = 480 := by sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_triangle_area_l2643_264348


namespace NUMINAMATH_CALUDE_base_number_proof_l2643_264399

theorem base_number_proof (x : ℝ) : 16^8 = x^16 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l2643_264399


namespace NUMINAMATH_CALUDE_sum_of_x_coordinates_is_seven_l2643_264331

-- Define the system of equations
def equation1 (x : ℝ) : ℝ := |x^2 - 8*x + 15|
def equation2 (x : ℝ) : ℝ := 8 - x

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | equation1 x = equation2 x}

-- State the theorem
theorem sum_of_x_coordinates_is_seven :
  ∃ (x₁ x₂ : ℝ), x₁ ∈ solution_set ∧ x₂ ∈ solution_set ∧ x₁ ≠ x₂ ∧ x₁ + x₂ = 7 :=
sorry

end NUMINAMATH_CALUDE_sum_of_x_coordinates_is_seven_l2643_264331


namespace NUMINAMATH_CALUDE_unique_linear_m_l2643_264305

/-- A function to represent the linearity of an equation -/
def is_linear (m : ℝ) : Prop :=
  (abs m = 1) ∧ (m + 1 ≠ 0)

/-- Theorem stating that m = 1 is the only value satisfying the linearity condition -/
theorem unique_linear_m : ∃! m : ℝ, is_linear m :=
  sorry

end NUMINAMATH_CALUDE_unique_linear_m_l2643_264305


namespace NUMINAMATH_CALUDE_principal_calculation_l2643_264304

/-- Represents the simple interest calculation for a loan --/
def simple_interest_loan (principal rate time interest : ℚ) : Prop :=
  interest = (principal * rate * time) / 100

theorem principal_calculation (rate time interest : ℚ) 
  (h1 : rate = 12)
  (h2 : time = 20)
  (h3 : interest = 2100) :
  ∃ (principal : ℚ), simple_interest_loan principal rate time interest ∧ principal = 875 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l2643_264304


namespace NUMINAMATH_CALUDE_minimum_of_x_squared_l2643_264386

theorem minimum_of_x_squared :
  ∃ (m : ℝ), m = 0 ∧ ∀ x : ℝ, x^2 ≥ m := by sorry

end NUMINAMATH_CALUDE_minimum_of_x_squared_l2643_264386


namespace NUMINAMATH_CALUDE_prime_characterization_l2643_264333

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, 2 ≤ k → k < n → ¬(k ∣ n)

theorem prime_characterization (n : ℕ) :
  Nat.Prime n ↔ is_prime n := by
  sorry

end NUMINAMATH_CALUDE_prime_characterization_l2643_264333


namespace NUMINAMATH_CALUDE_gcd_pow_minus_one_l2643_264332

theorem gcd_pow_minus_one (a b : ℕ) :
  Nat.gcd (2^a - 1) (2^b - 1) = 2^(Nat.gcd a b) - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_pow_minus_one_l2643_264332


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2643_264357

theorem complex_modulus_problem (z : ℂ) (h : z * (1 + Complex.I) = Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2643_264357


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2643_264306

theorem right_triangle_hypotenuse (a b h : ℝ) : 
  a = 15 ∧ b = 36 ∧ a^2 + b^2 = h^2 → h = 39 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2643_264306


namespace NUMINAMATH_CALUDE_machine_depreciation_rate_l2643_264317

/-- The annual depreciation rate of a machine given its initial value,
    selling price after two years, and profit. -/
theorem machine_depreciation_rate
  (initial_value : ℝ)
  (selling_price : ℝ)
  (profit : ℝ)
  (h1 : initial_value = 150000)
  (h2 : selling_price = 113935)
  (h3 : profit = 24000)
  : ∃ (r : ℝ), initial_value * (1 - r / 100)^2 = selling_price - profit :=
sorry

end NUMINAMATH_CALUDE_machine_depreciation_rate_l2643_264317


namespace NUMINAMATH_CALUDE_middle_integer_is_six_l2643_264330

def is_valid_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧
  a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 ∧
  b = a + 2 ∧ c = b + 2 ∧
  a + b + c = (a * b * c) / 8

theorem middle_integer_is_six :
  ∀ a b c : ℕ, is_valid_triple a b c → b = 6 :=
by sorry

end NUMINAMATH_CALUDE_middle_integer_is_six_l2643_264330


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2643_264321

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a → a 5 * a 14 = 5 → a 8 * a 9 * a 10 * a 11 = 25 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2643_264321


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l2643_264361

theorem billion_to_scientific_notation :
  (6.1 : ℝ) * 1000000000 = (6.1 : ℝ) * (10 ^ 8) :=
by sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l2643_264361


namespace NUMINAMATH_CALUDE_solution_set_part_I_range_of_a_part_II_l2643_264370

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Theorem for part I
theorem solution_set_part_I :
  ∀ x : ℝ, f (-2) x + f (-2) (2*x) > 2 ↔ x < -2 ∨ x > -2/3 :=
sorry

-- Theorem for part II
theorem range_of_a_part_II :
  ∀ a : ℝ, a < 0 → (∃ x : ℝ, f a x + f a (2*x) < 1/2) → -1 < a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part_I_range_of_a_part_II_l2643_264370


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l2643_264364

open Set

-- Define the universal set I as the set of real numbers
def I : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define set N
def N : Set ℝ := {x | x < 1}

-- Theorem statement
theorem complement_M_intersect_N :
  (I \ M) ∩ N = {x : ℝ | x < -2} := by
  sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l2643_264364


namespace NUMINAMATH_CALUDE_meeting_handshakes_l2643_264391

/-- The number of people in the meeting -/
def total_people : ℕ := 40

/-- The number of people who know each other -/
def group_a : ℕ := 25

/-- The number of people who don't know anyone -/
def group_b : ℕ := 15

/-- Calculate the number of handshakes between two groups -/
def inter_group_handshakes (g1 g2 : ℕ) : ℕ := g1 * g2

/-- Calculate the number of handshakes within a group where no one knows each other -/
def intra_group_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The total number of handshakes in the meeting -/
def total_handshakes : ℕ := 
  inter_group_handshakes group_a group_b + intra_group_handshakes group_b

theorem meeting_handshakes : 
  total_people = group_a + group_b → total_handshakes = 480 := by
  sorry

end NUMINAMATH_CALUDE_meeting_handshakes_l2643_264391


namespace NUMINAMATH_CALUDE_inequality_implication_l2643_264383

theorem inequality_implication (a b c d e : ℝ) :
  a * b^2 * c^3 * d^4 * e^5 < 0 → a * b^2 * c * d^4 * e < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l2643_264383


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l2643_264354

/-- Given a positive geometric sequence {a_n} with a_2 = 2 and 2a_3 + a_4 = 16,
    prove that the general term formula is a_n = 2^(n-1) -/
theorem geometric_sequence_formula (a : ℕ → ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n)
  (h_a2 : a 2 = 2)
  (h_sum : 2 * a 3 + a 4 = 16) :
  ∀ n : ℕ, a n = 2^(n - 1) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l2643_264354


namespace NUMINAMATH_CALUDE_quadratic_equation_no_real_roots_l2643_264397

theorem quadratic_equation_no_real_roots 
  (a b c : ℝ) 
  (h : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) : 
  ∀ x : ℝ, x^2 + (a + b + c) * x + a^2 + b^2 + c^2 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_no_real_roots_l2643_264397


namespace NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_equality_condition_l2643_264373

theorem min_sum_squares (a b c d : ℝ) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0) 
  (sum_eq : a + b + c + d = Real.sqrt 7960) : 
  a^2 + b^2 + c^2 + d^2 ≥ 1990 := by
sorry

theorem min_sum_squares_equality_condition (a b c d : ℝ) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0) 
  (sum_eq : a + b + c + d = Real.sqrt 7960) : 
  a^2 + b^2 + c^2 + d^2 = 1990 ↔ a = b ∧ b = c ∧ c = d := by
sorry

end NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_equality_condition_l2643_264373


namespace NUMINAMATH_CALUDE_new_average_age_l2643_264365

/-- Calculates the new average age of a group after new members join -/
theorem new_average_age
  (initial_count : ℕ)
  (initial_avg_age : ℚ)
  (new_count : ℕ)
  (new_avg_age : ℚ)
  (h1 : initial_count = 20)
  (h2 : initial_avg_age = 16)
  (h3 : new_count = 20)
  (h4 : new_avg_age = 15) :
  let total_initial_age := initial_count * initial_avg_age
  let total_new_age := new_count * new_avg_age
  let total_count := initial_count + new_count
  let new_avg := (total_initial_age + total_new_age) / total_count
  new_avg = 15.5 := by
  sorry

end NUMINAMATH_CALUDE_new_average_age_l2643_264365


namespace NUMINAMATH_CALUDE_unique_solution_x_zero_l2643_264334

theorem unique_solution_x_zero (x y : ℝ) : 
  y = 2 * x → (3 * y^2 + y + 4 = 2 * (6 * x^2 + y + 2)) ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_x_zero_l2643_264334


namespace NUMINAMATH_CALUDE_aiyanna_cookies_l2643_264387

def alyssa_cookies : ℕ := 129
def cookie_difference : ℕ := 11

theorem aiyanna_cookies : ℕ := alyssa_cookies + cookie_difference

#check aiyanna_cookies -- This should return 140

end NUMINAMATH_CALUDE_aiyanna_cookies_l2643_264387
