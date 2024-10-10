import Mathlib

namespace jean_initial_stuffies_l442_44251

/-- Proves that Jean initially had 60 stuffies given the problem conditions -/
theorem jean_initial_stuffies :
  ∀ (initial : ℕ),
  (initial : ℚ) * (2/3) * (1/4) = 10 →
  initial = 60 := by
sorry

end jean_initial_stuffies_l442_44251


namespace subset_collection_m_eq_seven_l442_44215

/-- A structure representing a collection of 3-element subsets of {1, ..., n} -/
structure SubsetCollection (n : ℕ) where
  m : ℕ
  subsets : Fin m → Finset (Fin n)
  m_gt_one : m > 1
  three_elements : ∀ i, (subsets i).card = 3
  unique_pairs : ∀ {x y : Fin n}, x ≠ y → ∃! i, {x, y} ⊆ subsets i
  one_common : ∀ {i j : Fin m}, i ≠ j → ∃! x, x ∈ subsets i ∩ subsets j

/-- The main theorem stating that for any valid SubsetCollection, m = 7 -/
theorem subset_collection_m_eq_seven {n : ℕ} (sc : SubsetCollection n) : sc.m = 7 :=
sorry

end subset_collection_m_eq_seven_l442_44215


namespace min_difference_of_extreme_points_l442_44208

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2*x - 1/x - a * log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - x + 2*a * log x

theorem min_difference_of_extreme_points (a : ℝ) (x₁ x₂ : ℝ) : 
  x₁ ∈ Set.Icc 0 1 → 
  (∀ x, x ≠ x₁ → x ≠ x₂ → g a x ≥ min (g a x₁) (g a x₂)) →
  g a x₁ - g a x₂ ≥ 0 :=
sorry

end min_difference_of_extreme_points_l442_44208


namespace magnet_area_theorem_l442_44253

/-- Represents a rectangular magnet with length and width in centimeters. -/
structure Magnet where
  length : ℝ
  width : ℝ

/-- Calculates the area of a magnet in square centimeters. -/
def area (m : Magnet) : ℝ := m.length * m.width

/-- Calculates the circumference of two identical magnets attached horizontally. -/
def totalCircumference (m : Magnet) : ℝ := 2 * (2 * m.length + 2 * m.width)

/-- Theorem: Given two identical rectangular magnets with a total circumference of 70 cm
    and a total length of 15 cm when attached horizontally, the area of one magnet is 150 cm². -/
theorem magnet_area_theorem (m : Magnet) 
    (h1 : totalCircumference m = 70)
    (h2 : 2 * m.length = 15) : 
  area m = 150 := by
  sorry

end magnet_area_theorem_l442_44253


namespace smallest_period_scaled_l442_44224

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x - p) = f x

theorem smallest_period_scaled (f : ℝ → ℝ) (h : is_periodic f 30) :
  ∃ b : ℝ, b > 0 ∧ (∀ x, f ((x - b) / 3) = f (x / 3)) ∧
  ∀ b' : ℝ, 0 < b' ∧ (∀ x, f ((x - b') / 3) = f (x / 3)) → b ≤ b' :=
sorry

end smallest_period_scaled_l442_44224


namespace infinitely_many_composite_sums_l442_44283

theorem infinitely_many_composite_sums : 
  ∃ (f : ℕ → ℕ), Function.Injective f ∧ 
  ∀ (k n : ℕ), ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ n^4 + (f k)^4 = x * y :=
sorry

end infinitely_many_composite_sums_l442_44283


namespace equation_system_solution_l442_44280

theorem equation_system_solution :
  ∃ (x₁ x₂ : ℝ),
    (∀ x y : ℝ, 5 * y^2 + 3 * y + 2 = 2 * (10 * x^2 + 3 * y + 3) ∧ y = 3 * x + 1 →
      x = x₁ ∨ x = x₂) ∧
    x₁ = (-21 + Real.sqrt 641) / 50 ∧
    x₂ = (-21 - Real.sqrt 641) / 50 := by
  sorry

end equation_system_solution_l442_44280


namespace quadratic_intersects_x_axis_symmetric_roots_l442_44258

/-- The quadratic function f(x) = x^2 - 2kx - 1 -/
def f (k : ℝ) (x : ℝ) : ℝ := x^2 - 2*k*x - 1

theorem quadratic_intersects_x_axis (k : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0 :=
sorry

theorem symmetric_roots :
  f 0 1 = 0 ∧ f 0 (-1) = 0 :=
sorry

end quadratic_intersects_x_axis_symmetric_roots_l442_44258


namespace angle_1303_equiv_neg137_l442_44266

-- Define a function to represent angles with the same terminal side
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ n : ℤ, β = α + n * 360

-- State the theorem
theorem angle_1303_equiv_neg137 :
  same_terminal_side 1303 (-137) :=
sorry

end angle_1303_equiv_neg137_l442_44266


namespace negation_equivalence_l442_44276

theorem negation_equivalence :
  (¬ ∃ n : ℕ, 2^n > 1000) ↔ (∀ n : ℕ, 2^n ≤ 1000) := by sorry

end negation_equivalence_l442_44276


namespace min_value_ab_l442_44230

/-- Given that ab > 0 and points A(a, 0), B(0, b), and C(-2, -2) are collinear,
    the minimum value of ab is 16. -/
theorem min_value_ab (a b : ℝ) (hab : a * b > 0)
  (h_collinear : (a - 0) * (-2 - b) = (-2 - a) * (b - 0)) :
  ∀ x y : ℝ, x * y > 0 → (x - 0) * (-2 - y) = (-2 - x) * (y - 0) → a * b ≤ x * y → a * b = 16 := by
  sorry

end min_value_ab_l442_44230


namespace roberta_listening_time_l442_44202

/-- The number of days it takes Roberta to listen to her entire record collection -/
def listen_time (initial_records : ℕ) (gift_records : ℕ) (bought_records : ℕ) (days_per_record : ℕ) : ℕ :=
  (initial_records + gift_records + bought_records) * days_per_record

theorem roberta_listening_time :
  listen_time 8 12 30 2 = 100 := by
  sorry

end roberta_listening_time_l442_44202


namespace line_passes_through_quadrants_l442_44203

/-- A line in the plane defined by the equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The quadrants of the Cartesian plane -/
inductive Quadrant
  | first
  | second
  | third
  | fourth

/-- Predicate to check if a line passes through a quadrant -/
def passes_through (l : Line) (q : Quadrant) : Prop := sorry

/-- Theorem stating that under given conditions, the line passes through specific quadrants -/
theorem line_passes_through_quadrants (l : Line) 
  (h1 : l.A * l.C < 0) (h2 : l.B * l.C < 0) : 
  passes_through l Quadrant.first ∧ 
  passes_through l Quadrant.second ∧ 
  passes_through l Quadrant.fourth :=
sorry

end line_passes_through_quadrants_l442_44203


namespace ship_passengers_ship_passengers_proof_l442_44263

theorem ship_passengers : ℕ → Prop :=
  fun total_passengers =>
    (total_passengers : ℚ) = (1 / 12 + 1 / 4 + 1 / 9 + 1 / 6) * total_passengers + 42 →
    total_passengers = 108

-- Proof
theorem ship_passengers_proof : ship_passengers 108 := by
  sorry

end ship_passengers_ship_passengers_proof_l442_44263


namespace min_value_x_plus_y_l442_44232

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (heq : y + 9 * x = x * y) :
  ∀ a b : ℝ, a > 0 ∧ b > 0 ∧ b + 9 * a = a * b → x + y ≤ a + b ∧ x + y = 16 := by
  sorry

end min_value_x_plus_y_l442_44232


namespace root_expression_value_l442_44271

theorem root_expression_value (x₁ x₂ : ℝ) 
  (h₁ : x₁^2 - x₁ - 2022 = 0) 
  (h₂ : x₂^2 - x₂ - 2022 = 0) : 
  x₁^3 - 2022*x₁ + x₂^2 = 4045 := by
  sorry

end root_expression_value_l442_44271


namespace fuel_station_problem_l442_44228

/-- Represents the problem of calculating the number of mini-vans filled up at a fuel station. -/
theorem fuel_station_problem (service_cost : ℝ) (fuel_cost_per_liter : ℝ) (total_cost : ℝ) 
  (minivan_tank : ℝ) (truck_tank : ℝ) (num_trucks : ℕ) :
  service_cost = 2.20 →
  fuel_cost_per_liter = 0.70 →
  total_cost = 347.7 →
  minivan_tank = 65 →
  truck_tank = minivan_tank * 2.2 →
  num_trucks = 2 →
  ∃ (num_minivans : ℕ), 
    (num_minivans : ℝ) * (service_cost + fuel_cost_per_liter * minivan_tank) + 
    (num_trucks : ℝ) * (service_cost + fuel_cost_per_liter * truck_tank) = total_cost ∧
    num_minivans = 3 :=
by sorry

end fuel_station_problem_l442_44228


namespace number_of_correct_statements_l442_44299

-- Define the properties
def is_rational (m : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ m = a / b
def is_real (m : ℝ) : Prop := True

def tan_equal (A B : ℝ) : Prop := Real.tan A = Real.tan B
def angle_equal (A B : ℝ) : Prop := A = B

def quadratic_equation (x : ℝ) : Prop := x^2 - 2*x - 3 = 0
def x_equals_3 (x : ℝ) : Prop := x = 3

-- Define the statements
def statement1 : Prop := 
  (∀ m : ℝ, is_rational m → is_real m) ∧ 
  ¬(∀ m : ℝ, is_real m → is_rational m)

def statement2 : Prop := 
  (∀ A B : ℝ, tan_equal A B → angle_equal A B) ∧ 
  ¬(∀ A B : ℝ, angle_equal A B → tan_equal A B)

def statement3 : Prop := 
  (∀ x : ℝ, x_equals_3 x → quadratic_equation x) ∧ 
  ¬(∀ x : ℝ, quadratic_equation x → x_equals_3 x)

-- Theorem to prove
theorem number_of_correct_statements : 
  (statement1 ∧ ¬statement2 ∧ statement3) → 
  (Nat.card {s | s = statement1 ∨ s = statement2 ∨ s = statement3 ∧ s} = 2) :=
sorry

end number_of_correct_statements_l442_44299


namespace horner_method_v3_l442_44291

def f (x : ℝ) : ℝ := x^5 + 2*x^4 + x^3 - x^2 + 3*x - 5

def horner_v3 (a₅ a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  ((((a₅ * x + a₄) * x + a₃) * x + a₂) * x + a₁) * x + a₀

theorem horner_method_v3 :
  horner_v3 1 2 1 (-1) 3 (-5) 5 = 179 :=
sorry

end horner_method_v3_l442_44291


namespace lines_not_parallel_l442_44252

-- Define the types for our objects
variable (Point Line Plane : Type)

-- Define the relationships
variable (contains : Plane → Line → Prop)
variable (not_contains : Plane → Line → Prop)
variable (on_line : Point → Line → Prop)
variable (in_plane : Point → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_not_parallel 
  (m n : Line) (α : Plane) (A : Point)
  (h1 : not_contains α m)
  (h2 : contains α n)
  (h3 : on_line A m)
  (h4 : in_plane A α) :
  ¬(parallel m n) :=
by sorry

end lines_not_parallel_l442_44252


namespace elevator_max_velocity_l442_44294

/-- Represents the state of the elevator at a given time -/
structure ElevatorState where
  time : ℝ
  velocity : ℝ

/-- The elevator's motion profile -/
def elevatorMotion : ℝ → ElevatorState := sorry

/-- The acceleration period of the elevator -/
def accelerationPeriod : Set ℝ := {t | 2 ≤ t ∧ t ≤ 4}

/-- The deceleration period of the elevator -/
def decelerationPeriod : Set ℝ := {t | 22 ≤ t ∧ t ≤ 24}

/-- The constant speed period of the elevator -/
def constantSpeedPeriod : Set ℝ := {t | 4 < t ∧ t < 22}

/-- The maximum downward velocity of the elevator -/
def maxDownwardVelocity : ℝ := sorry

theorem elevator_max_velocity :
  ∀ t ∈ constantSpeedPeriod,
    (elevatorMotion t).velocity = maxDownwardVelocity ∧
    ∀ s, (elevatorMotion s).velocity ≤ maxDownwardVelocity := by
  sorry

#check elevator_max_velocity

end elevator_max_velocity_l442_44294


namespace profit_share_difference_theorem_l442_44269

/-- Represents an investor's contribution to the business --/
structure Investor where
  investment : ℕ
  duration : ℕ

/-- Calculates the difference in profit shares between two investors --/
def profit_share_difference (suresh rohan sudhir : Investor) (total_profit : ℕ) : ℕ :=
  let total_investment_months := suresh.investment * suresh.duration + 
                                 rohan.investment * rohan.duration + 
                                 sudhir.investment * sudhir.duration
  let rohan_share := (rohan.investment * rohan.duration * total_profit) / total_investment_months
  let sudhir_share := (sudhir.investment * sudhir.duration * total_profit) / total_investment_months
  rohan_share - sudhir_share

/-- Theorem stating the difference in profit shares --/
theorem profit_share_difference_theorem (suresh rohan sudhir : Investor) (total_profit : ℕ) :
  suresh.investment = 18000 ∧ suresh.duration = 12 ∧
  rohan.investment = 12000 ∧ rohan.duration = 9 ∧
  sudhir.investment = 9000 ∧ sudhir.duration = 8 ∧
  total_profit = 3795 →
  profit_share_difference suresh rohan sudhir total_profit = 345 := by
  sorry

end profit_share_difference_theorem_l442_44269


namespace smallest_multiple_of_4_and_14_l442_44221

theorem smallest_multiple_of_4_and_14 : ∀ a : ℕ, a > 0 ∧ 4 ∣ a ∧ 14 ∣ a → a ≥ 28 := by
  sorry

end smallest_multiple_of_4_and_14_l442_44221


namespace complex_equality_l442_44275

theorem complex_equality (a : ℝ) : 
  (1 + (a - 2) * Complex.I).im = 0 → (a + Complex.I) / Complex.I = 1 - 2 * Complex.I :=
by sorry

end complex_equality_l442_44275


namespace f_derivative_at_one_l442_44244

noncomputable def f (x : ℝ) : ℝ := Real.sin 1 - Real.cos x

theorem f_derivative_at_one : 
  deriv f 1 = Real.sin 1 := by sorry

end f_derivative_at_one_l442_44244


namespace pure_imaginary_second_quadrant_l442_44290

-- Define the complex number z as a function of real number m
def z (m : ℝ) : ℂ := Complex.mk (m^2 - 2*m - 3) (m^2 + 3*m + 2)

-- Theorem 1: z is a pure imaginary number if and only if m = 3
theorem pure_imaginary (m : ℝ) : z m = Complex.I * (z m).im ↔ m = 3 := by
  sorry

-- Theorem 2: z is in the second quadrant if and only if -1 < m < 3
theorem second_quadrant (m : ℝ) : (z m).re < 0 ∧ (z m).im > 0 ↔ -1 < m ∧ m < 3 := by
  sorry

end pure_imaginary_second_quadrant_l442_44290


namespace vector_magnitude_problem_l442_44211

theorem vector_magnitude_problem (a b : ℝ × ℝ) :
  let angle := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  let magnitude_2a_plus_b := Real.sqrt ((2*a.1 + b.1)^2 + (2*a.2 + b.2)^2)
  angle = π/4 ∧ magnitude_a = 1 ∧ magnitude_2a_plus_b = Real.sqrt 10 →
  Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 2 := by
sorry


end vector_magnitude_problem_l442_44211


namespace trivia_team_absence_l442_44213

theorem trivia_team_absence (total_members : ℕ) (points_per_member : ℕ) (total_score : ℕ) 
  (h1 : total_members = 14)
  (h2 : points_per_member = 5)
  (h3 : total_score = 35) :
  total_members - (total_score / points_per_member) = 7 := by
  sorry

end trivia_team_absence_l442_44213


namespace smallest_n_with_three_triples_l442_44292

/-- Function that counts the number of distinct ordered triples (a, b, c) of positive integers
    such that a^2 + b^2 + c^2 = n -/
def g (n : ℕ) : ℕ := 
  (Finset.filter (fun t : ℕ × ℕ × ℕ => 
    t.1 > 0 ∧ t.2.1 > 0 ∧ t.2.2 > 0 ∧ 
    t.1^2 + t.2.1^2 + t.2.2^2 = n) (Finset.product (Finset.range n) (Finset.product (Finset.range n) (Finset.range n)))).card

/-- 11 is the smallest positive integer n for which g(n) = 3 -/
theorem smallest_n_with_three_triples : 
  (∀ m : ℕ, m > 0 ∧ m < 11 → g m ≠ 3) ∧ g 11 = 3 :=
sorry

end smallest_n_with_three_triples_l442_44292


namespace polynomial_root_product_l442_44262

theorem polynomial_root_product (k : ℝ) : 
  (∃ (x₁ x₂ x₃ x₄ : ℝ), 
    (x₁^4 - 18*x₁^3 + k*x₁^2 + 200*x₁ - 1984 = 0) ∧
    (x₂^4 - 18*x₂^3 + k*x₂^2 + 200*x₂ - 1984 = 0) ∧
    (x₃^4 - 18*x₃^3 + k*x₃^2 + 200*x₃ - 1984 = 0) ∧
    (x₄^4 - 18*x₄^3 + k*x₄^2 + 200*x₄ - 1984 = 0) ∧
    (x₁ * x₂ = -32 ∨ x₁ * x₃ = -32 ∨ x₁ * x₄ = -32 ∨ 
     x₂ * x₃ = -32 ∨ x₂ * x₄ = -32 ∨ x₃ * x₄ = -32)) →
  k = 86 :=
by sorry

end polynomial_root_product_l442_44262


namespace range_of_m_for_quadratic_inequality_l442_44217

theorem range_of_m_for_quadratic_inequality (m : ℝ) : 
  m ≠ 0 → 
  (∀ x : ℝ, x ∈ Set.Icc 1 3 → m * x^2 - m * x - 1 < -m + 5) ↔ 
  m > 0 ∧ m < 6/7 := by
sorry

end range_of_m_for_quadratic_inequality_l442_44217


namespace pole_not_perpendicular_l442_44212

theorem pole_not_perpendicular (h : Real) (d : Real) (c : Real) 
  (h_val : h = 1.4)
  (d_val : d = 2)
  (c_val : c = 2.5) : 
  h^2 + d^2 ≠ c^2 := by
  sorry

end pole_not_perpendicular_l442_44212


namespace derivative_evaluation_l442_44206

theorem derivative_evaluation (x : ℝ) (h : x > 0) :
  let F : ℝ → ℝ := λ x => (1 - Real.sqrt x)^2 / x
  let F' : ℝ → ℝ := λ x => -1/x^2 + 1/x^(3/2)
  F' 0.01 = -9000 := by sorry

end derivative_evaluation_l442_44206


namespace collinear_points_ratio_l442_44267

/-- Given four collinear points E, F, G, H in that order, with EF = 3, FG = 6, and EH = 20,
    prove that the ratio of EG to FH is 9/17. -/
theorem collinear_points_ratio (E F G H : ℝ) : 
  (F - E = 3) → (G - F = 6) → (H - E = 20) → 
  (E < F) → (F < G) → (G < H) →
  (G - E) / (H - F) = 9 / 17 := by
sorry

end collinear_points_ratio_l442_44267


namespace parabola_tangent_values_l442_44225

/-- A parabola tangent to a line -/
structure ParabolaTangentToLine where
  /-- Coefficient of x^2 term in the parabola equation -/
  a : ℝ
  /-- Coefficient of x term in the parabola equation -/
  b : ℝ
  /-- The parabola y = ax^2 + bx is tangent to the line y = 2x + 4 -/
  is_tangent : ∃ (x : ℝ), a * x^2 + b * x = 2 * x + 4
  /-- The x-coordinate of the point of tangency is 1 -/
  tangent_point : ∃ (y : ℝ), a * 1^2 + b * 1 = 2 * 1 + 4 ∧ a * 1^2 + b * 1 = y

/-- The values of a and b for the parabola tangent to the line -/
theorem parabola_tangent_values (p : ParabolaTangentToLine) : p.a = 4/3 ∧ p.b = 10/3 := by
  sorry

end parabola_tangent_values_l442_44225


namespace alice_wins_iff_m_even_or_n_odd_l442_44274

/-- The game state on an n×n grid where players can color an m×m subgrid or a single cell -/
structure GameState (m n : ℕ+) where
  grid : Fin n → Fin n → Bool

/-- The result of the game -/
inductive GameResult
  | AliceWins
  | BobWins

/-- An optimal strategy for the game -/
def OptimalStrategy (m n : ℕ+) : GameState m n → GameResult := sorry

/-- The main theorem: Alice wins with optimal play if and only if m is even or n is odd -/
theorem alice_wins_iff_m_even_or_n_odd (m n : ℕ+) :
  (∀ initial : GameState m n, OptimalStrategy m n initial = GameResult.AliceWins) ↔ 
  (Even m.val ∨ Odd n.val) := by sorry

end alice_wins_iff_m_even_or_n_odd_l442_44274


namespace hyperbola_and_line_theorem_l442_44245

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 - (4 * y^2 / 33) = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define point P
def P : ℝ × ℝ := (7, 12)

-- Define the asymptotes
def asymptote_positive (x y : ℝ) : Prop := y = (Real.sqrt 33 / 2) * x
def asymptote_negative (x y : ℝ) : Prop := y = -(Real.sqrt 33 / 2) * x

-- Define line l
def line_l (x y t : ℝ) : Prop := y = x + t

-- Define perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem hyperbola_and_line_theorem :
  -- Hyperbola C passes through P
  hyperbola_C P.1 P.2 →
  -- There exist points A and B on C and l
  ∃ (A B : ℝ × ℝ) (t : ℝ),
    hyperbola_C A.1 A.2 ∧ 
    hyperbola_C B.1 B.2 ∧
    line_l A.1 A.2 t ∧
    line_l B.1 B.2 t ∧
    -- A and B are perpendicular from the origin
    perpendicular A.1 A.2 B.1 B.2 →
  -- Then the equation of line l is y = x ± √(66/29)
  t = Real.sqrt (66 / 29) ∨ t = -Real.sqrt (66 / 29) :=
sorry

end hyperbola_and_line_theorem_l442_44245


namespace arithmetic_calculation_l442_44243

theorem arithmetic_calculation : 8 / 4 - 3 - 9 + 3 * 9 = 17 := by
  sorry

end arithmetic_calculation_l442_44243


namespace ellen_smoothie_total_cups_l442_44257

/-- Represents the ingredients used in Ellen's smoothie recipe -/
structure SmoothieIngredients where
  strawberries : Float
  yogurt : Float
  orange_juice : Float
  honey : Float
  chia_seeds : Float
  spinach : Float

/-- Conversion factors for measurements -/
def ounce_to_cup : Float := 0.125
def tablespoon_to_cup : Float := 0.0625

/-- Ellen's smoothie recipe -/
def ellen_smoothie : SmoothieIngredients := {
  strawberries := 0.2,
  yogurt := 0.1,
  orange_juice := 0.2,
  honey := 1 * ounce_to_cup,
  chia_seeds := 2 * tablespoon_to_cup,
  spinach := 0.5
}

/-- Theorem stating the total cups of ingredients in Ellen's smoothie -/
theorem ellen_smoothie_total_cups : 
  ellen_smoothie.strawberries + 
  ellen_smoothie.yogurt + 
  ellen_smoothie.orange_juice + 
  ellen_smoothie.honey + 
  ellen_smoothie.chia_seeds + 
  ellen_smoothie.spinach = 1.25 := by sorry

end ellen_smoothie_total_cups_l442_44257


namespace sum_of_roots_l442_44241

theorem sum_of_roots (a b : ℝ) (ha : a * (a - 4) = 5) (hb : b * (b - 4) = 5) (hab : a ≠ b) : a + b = 4 := by
  sorry

end sum_of_roots_l442_44241


namespace parametric_line_point_at_zero_l442_44220

/-- A parametric line in 2D space -/
structure ParametricLine where
  /-- The position vector for a given parameter t -/
  pos : ℝ → (ℝ × ℝ)

/-- Theorem: Given a parametric line with specific points, find the point at t = 0 -/
theorem parametric_line_point_at_zero
  (line : ParametricLine)
  (h1 : line.pos 1 = (2, 3))
  (h4 : line.pos 4 = (6, -12)) :
  line.pos 0 = (2/3, 8) := by
  sorry

end parametric_line_point_at_zero_l442_44220


namespace star_three_five_l442_44298

-- Define the star operation
def star (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2

-- Theorem statement
theorem star_three_five : star 3 5 = 4 := by
  sorry

end star_three_five_l442_44298


namespace expression_factorization_l442_44240

theorem expression_factorization (x : ℝ) :
  (12 * x^3 + 45 * x - 3) - (-3 * x^3 + 5 * x - 2) = 5 * x * (3 * x^2 + 8) - 1 :=
by sorry

end expression_factorization_l442_44240


namespace log_power_equality_l442_44297

theorem log_power_equality (a N m : ℝ) (ha : a > 0) (hN : N > 0) (hm : m ≠ 0) :
  Real.log N^m / Real.log (a^m) = Real.log N / Real.log a := by
  sorry

end log_power_equality_l442_44297


namespace subtract_3a_from_expression_l442_44237

variable (a : ℝ)

theorem subtract_3a_from_expression : (9 * a^2 - 3 * a + 8) - 3 * a = 9 * a^2 + 8 := by
  sorry

end subtract_3a_from_expression_l442_44237


namespace cube_root_nine_thirty_two_squared_l442_44250

theorem cube_root_nine_thirty_two_squared :
  (((9 : ℝ) / 32) ^ (1/3 : ℝ)) ^ 2 = 3 / 8 := by sorry

end cube_root_nine_thirty_two_squared_l442_44250


namespace distinct_cube_models_count_l442_44238

/-- The number of vertices in a cube -/
def cube_vertices : ℕ := 8

/-- The number of colors available -/
def available_colors : ℕ := 8

/-- The number of rotational symmetries of a cube -/
def cube_rotations : ℕ := 24

/-- The number of distinct models of cubes with differently colored vertices -/
def distinct_cube_models : ℕ := Nat.factorial available_colors / cube_rotations

theorem distinct_cube_models_count :
  distinct_cube_models = 1680 := by sorry

end distinct_cube_models_count_l442_44238


namespace average_of_remaining_numbers_l442_44222

theorem average_of_remaining_numbers
  (total_count : Nat)
  (total_average : ℚ)
  (subset_count : Nat)
  (subset_average : ℚ)
  (h1 : total_count = 50)
  (h2 : total_average = 76)
  (h3 : subset_count = 40)
  (h4 : subset_average = 80)
  (h5 : subset_count < total_count) :
  let remaining_count := total_count - subset_count
  let remaining_sum := total_count * total_average - subset_count * subset_average
  remaining_sum / remaining_count = 60 := by
sorry

end average_of_remaining_numbers_l442_44222


namespace weekly_commute_cost_l442_44270

-- Define the parameters
def workDays : ℕ := 5
def carToll : ℚ := 12.5
def motorcycleToll : ℚ := 7
def milesPerGallon : ℚ := 35
def commuteDistance : ℚ := 14
def gasPrice : ℚ := 3.75
def carTrips : ℕ := 3
def motorcycleTrips : ℕ := 2

-- Define the theorem
theorem weekly_commute_cost :
  let carTollCost := carToll * carTrips
  let motorcycleTollCost := motorcycleToll * motorcycleTrips
  let totalDistance := commuteDistance * 2 * workDays
  let totalGasUsed := totalDistance / milesPerGallon
  let gasCost := totalGasUsed * gasPrice
  let totalCost := carTollCost + motorcycleTollCost + gasCost
  totalCost = 59 := by sorry

end weekly_commute_cost_l442_44270


namespace triangle_area_triangle_area_proof_l442_44284

/-- The area of a triangle with side lengths 15, 36, and 39 is 270 -/
theorem triangle_area : ℝ → Prop :=
  fun a => ∀ s₁ s₂ s₃ : ℝ,
    s₁ = 15 ∧ s₂ = 36 ∧ s₃ = 39 →
    (∃ A : ℝ, A = a ∧ A = 270)

/-- Proof of the theorem -/
theorem triangle_area_proof : triangle_area 270 := by
  sorry

end triangle_area_triangle_area_proof_l442_44284


namespace intersection_coordinate_sum_l442_44219

/-- Given a triangle ABC with A(0,6), B(0,0), C(8,0), 
    D is the midpoint of AB, 
    E is on BC such that BE is one-third of BC,
    F is the intersection of AE and CD.
    Prove that the sum of x and y coordinates of F is 56/11. -/
theorem intersection_coordinate_sum (A B C D E F : ℝ × ℝ) : 
  A = (0, 6) →
  B = (0, 0) →
  C = (8, 0) →
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  E.1 = B.1 + (C.1 - B.1) / 3 →
  E.2 = B.2 + (C.2 - B.2) / 3 →
  (F.2 - A.2) / (F.1 - A.1) = (E.2 - A.2) / (E.1 - A.1) →
  (F.2 - C.2) / (F.1 - C.1) = (D.2 - C.2) / (D.1 - C.1) →
  F.1 + F.2 = 56 / 11 := by
  sorry


end intersection_coordinate_sum_l442_44219


namespace apple_distribution_l442_44295

theorem apple_distribution (n : ℕ) (k : ℕ) (min_apples : ℕ) : 
  n = 24 → k = 3 → min_apples = 2 → 
  (Nat.choose (n - k * min_apples + k - 1) (k - 1)) = 190 := by
  sorry

end apple_distribution_l442_44295


namespace cube_equation_solution_l442_44214

theorem cube_equation_solution (a w : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * w) : w = 49 := by
  sorry

end cube_equation_solution_l442_44214


namespace sandwich_problem_l442_44265

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℚ := 349/100

/-- The cost of a soda in dollars -/
def soda_cost : ℚ := 87/100

/-- The number of sodas bought -/
def num_sodas : ℕ := 4

/-- The total cost of the purchase in dollars -/
def total_cost : ℚ := 1046/100

/-- The number of sandwiches bought -/
def num_sandwiches : ℕ := 2

theorem sandwich_problem :
  sandwich_cost * num_sandwiches + soda_cost * num_sodas = total_cost :=
sorry

end sandwich_problem_l442_44265


namespace competition_probabilities_l442_44260

/-- Represents the type of question in the competition -/
inductive QuestionType
| MultipleChoice
| TrueFalse

/-- Represents a question in the competition -/
structure Question where
  id : Nat
  type : QuestionType

/-- Represents the competition setup -/
structure Competition where
  questions : Finset Question
  numMultipleChoice : Nat
  numTrueFalse : Nat

/-- Represents a draw outcome for two participants -/
structure DrawOutcome where
  questionA : Question
  questionB : Question

/-- The probability of A drawing a multiple-choice question and B drawing a true/false question -/
def probAMultipleBTrue (c : Competition) : ℚ :=
  sorry

/-- The probability of at least one of A and B drawing a multiple-choice question -/
def probAtLeastOneMultiple (c : Competition) : ℚ :=
  sorry

/-- The main theorem stating the probabilities for the given competition setup -/
theorem competition_probabilities (c : Competition) 
  (h1 : c.questions.card = 4)
  (h2 : c.numMultipleChoice = 2)
  (h3 : c.numTrueFalse = 2) :
  probAMultipleBTrue c = 1/3 ∧ probAtLeastOneMultiple c = 5/6 := by
  sorry

end competition_probabilities_l442_44260


namespace parabola_chord_length_l442_44226

/-- Given a parabola y^2 = 4x and a line passing through its focus intersecting 
    the parabola at points P(x₁, y₁) and Q(x₂, y₂) such that x₁ + x₂ = 6, 
    prove that the length |PQ| = 8. -/
theorem parabola_chord_length (x₁ x₂ y₁ y₂ : ℝ) : 
  y₁^2 = 4*x₁ →  -- P is on the parabola
  y₂^2 = 4*x₂ →  -- Q is on the parabola
  x₁ + x₂ = 6 →  -- Given condition
  (∃ t : ℝ, t*x₁ + (1-t)*1 = 0 ∧ t*y₁ = 0) →  -- Line PQ passes through focus (1,0)
  (∃ s : ℝ, s*x₂ + (1-s)*1 = 0 ∧ s*y₂ = 0) →  -- Line PQ passes through focus (1,0)
  ((x₁ - x₂)^2 + (y₁ - y₂)^2)^(1/2) = 8 :=  -- |PQ| = 8
by sorry

end parabola_chord_length_l442_44226


namespace courtyard_width_is_16_meters_l442_44207

def courtyard_length : ℝ := 25
def brick_length : ℝ := 0.2
def brick_width : ℝ := 0.1
def total_bricks : ℕ := 20000

theorem courtyard_width_is_16_meters :
  let brick_area : ℝ := brick_length * brick_width
  let total_area : ℝ := (total_bricks : ℝ) * brick_area
  let courtyard_width : ℝ := total_area / courtyard_length
  courtyard_width = 16 := by sorry

end courtyard_width_is_16_meters_l442_44207


namespace exp_addition_property_l442_44254

open Real

-- Define the exponential function
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- State the theorem
theorem exp_addition_property (x y : ℝ) : f (x + y) = f x * f y := by
  sorry

end exp_addition_property_l442_44254


namespace eccentricity_relation_l442_44281

-- Define the eccentricities and point coordinates
variable (e₁ e₂ : ℝ)
variable (O F₁ F₂ P : ℝ × ℝ)

-- Define the conditions
def is_standard_ellipse_hyperbola : Prop :=
  0 < e₁ ∧ e₁ < 1 ∧ e₂ > 1

def foci_on_x_axis : Prop :=
  ∃ c : ℝ, F₁ = (c, 0) ∧ F₂ = (-c, 0)

def O_is_origin : Prop :=
  O = (0, 0)

def P_on_both_curves : Prop :=
  ∃ (x y : ℝ), P = (x, y)

def distance_condition : Prop :=
  2 * ‖P - O‖ = ‖F₁ - F₂‖

-- State the theorem
theorem eccentricity_relation
  (h₁ : is_standard_ellipse_hyperbola e₁ e₂)
  (h₂ : foci_on_x_axis F₁ F₂)
  (h₃ : O_is_origin O)
  (h₄ : P_on_both_curves P)
  (h₅ : distance_condition O F₁ F₂ P) :
  (e₁ * e₂) / Real.sqrt (e₁^2 + e₂^2) = Real.sqrt 2 / 2 :=
sorry

end eccentricity_relation_l442_44281


namespace root_shift_polynomial_l442_44239

theorem root_shift_polynomial (a b c : ℂ) : 
  (a^3 - 5*a^2 + 7*a - 2 = 0) ∧ 
  (b^3 - 5*b^2 + 7*b - 2 = 0) ∧ 
  (c^3 - 5*c^2 + 7*c - 2 = 0) → 
  ((a - 3)^3 + 4*(a - 3)^2 + 4*(a - 3) + 1 = 0) ∧
  ((b - 3)^3 + 4*(b - 3)^2 + 4*(b - 3) + 1 = 0) ∧
  ((c - 3)^3 + 4*(c - 3)^2 + 4*(c - 3) + 1 = 0) :=
by sorry

end root_shift_polynomial_l442_44239


namespace line_parallel_to_parallel_plane_l442_44249

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_parallel_plane 
  (α β : Plane) (m : Line)
  (h1 : subset m α)
  (h2 : parallel α β) :
  line_parallel m β :=
sorry

end line_parallel_to_parallel_plane_l442_44249


namespace function_property_l442_44229

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem function_property (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : ∀ x, f (x + 3) = -f x)
  (h_f1 : f 1 = 1) :
  f 2015 + f 2016 = -1 := by
  sorry

end function_property_l442_44229


namespace tan_sin_30_identity_l442_44278

theorem tan_sin_30_identity : 
  (Real.tan (30 * π / 180))^2 - (Real.sin (30 * π / 180))^2 = 
  (4 / 3) * (Real.tan (30 * π / 180))^2 * (Real.sin (30 * π / 180))^2 := by
sorry

end tan_sin_30_identity_l442_44278


namespace ordered_pairs_satisfying_inequalities_l442_44200

theorem ordered_pairs_satisfying_inequalities :
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (a b : ℤ), (a, b) ∈ s ↔ 
      (a^2 + b^2 < 16 ∧ 
       a^2 + b^2 < 8*a ∧ 
       a^2 + b^2 < 8*b)) ∧
    s.card = 6 := by
  sorry

end ordered_pairs_satisfying_inequalities_l442_44200


namespace bisection_method_root_existence_l442_44273

theorem bisection_method_root_existence
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h_cont : ContinuousOn f (Set.Icc a b))
  (h_sign : f a * f b < 0)
  (h_a_neg : f a < 0)
  (h_b_pos : f b > 0)
  (h_mid_pos : f ((a + b) / 2) > 0) :
  ∃ x ∈ Set.Ioo a ((a + b) / 2), f x = 0 :=
by sorry

end bisection_method_root_existence_l442_44273


namespace sales_volume_linear_profit_quadratic_max_profit_profit_3000_min_inventory_l442_44272

/-- Represents the daily sales model for a specialty store -/
structure SalesModel where
  x : ℝ  -- Selling price per item in yuan
  y : ℝ  -- Daily sales volume in items
  W : ℝ  -- Daily total profit in yuan
  h1 : 16 ≤ x ∧ x ≤ 48  -- Price constraints
  h2 : y = -10 * x + 560  -- Relationship between y and x
  h3 : W = (x - 16) * y  -- Definition of total profit

/-- The daily sales volume is a linear function of the selling price -/
theorem sales_volume_linear (model : SalesModel) :
  ∃ a b : ℝ, model.y = a * model.x + b :=
sorry

/-- The daily total profit is a quadratic function of the selling price -/
theorem profit_quadratic (model : SalesModel) :
  ∃ a b c : ℝ, model.W = a * model.x^2 + b * model.x + c :=
sorry

/-- The maximum daily profit occurs when the selling price is 36 yuan and equals 4000 yuan -/
theorem max_profit (model : SalesModel) :
  (∀ x : ℝ, 16 ≤ x ∧ x ≤ 48 → model.W ≤ 4000) ∧
  (∃ model' : SalesModel, model'.x = 36 ∧ model'.W = 4000) :=
sorry

/-- There exists a selling price that ensures a daily profit of 3000 yuan while minimizing inventory -/
theorem profit_3000_min_inventory (model : SalesModel) :
  ∃ x : ℝ, 16 ≤ x ∧ x ≤ 48 ∧
  (∃ model' : SalesModel, model'.x = x ∧ model'.W = 3000) ∧
  (∀ x' : ℝ, 16 ≤ x' ∧ x' ≤ 48 →
    (∃ model'' : SalesModel, model''.x = x' ∧ model''.W = 3000) →
    x ≤ x') :=
sorry

end sales_volume_linear_profit_quadratic_max_profit_profit_3000_min_inventory_l442_44272


namespace z_equals_3s_l442_44282

theorem z_equals_3s (z s : ℝ) (hz : z ≠ 0) (heq : z = Real.sqrt (6 * z * s - 9 * s^2)) : z = 3 * s := by
  sorry

end z_equals_3s_l442_44282


namespace sector_area_l442_44223

/-- The area of a circular sector with central angle 54° and radius 20 cm is 60π cm² -/
theorem sector_area (θ : Real) (r : Real) : 
  θ = 54 * π / 180 → r = 20 → (1/2) * r^2 * θ = 60 * π := by
  sorry

end sector_area_l442_44223


namespace probability_not_above_x_axis_l442_44247

/-- Parallelogram ABCD with given vertices -/
structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The specific parallelogram ABCD from the problem -/
def ABCD : Parallelogram :=
  { A := (4, 4)
    B := (-2, -2)
    C := (-8, -2)
    D := (0, 4) }

/-- Function to calculate the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := sorry

/-- Function to calculate the area of the part of the parallelogram below the x-axis -/
def areaBelowXAxis (p : Parallelogram) : ℝ := sorry

/-- Theorem stating the probability of a point not being above the x-axis -/
theorem probability_not_above_x_axis (p : Parallelogram) :
  p = ABCD →
  (areaBelowXAxis p) / (area p) = 1/3 := by sorry

end probability_not_above_x_axis_l442_44247


namespace part_one_part_two_l442_44227

-- Define the line l: y = k(x-n)
def line (k n x : ℝ) : ℝ := k * (x - n)

-- Define the parabola y^2 = 4x
def parabola (x : ℝ) : ℝ := 4 * x

-- Define the intersection points
structure Point where
  x : ℝ
  y : ℝ

-- Theorem for part (I)
theorem part_one (k n : ℝ) (A B : Point) (h1 : A.y = line k n A.x) (h2 : B.y = line k n B.x)
  (h3 : A.y^2 = parabola A.x) (h4 : B.y^2 = parabola B.x) (h5 : A.x * B.x ≠ 0)
  (h6 : line k n 1 = 0) : A.x * B.x = 1 := by sorry

-- Theorem for part (II)
theorem part_two (k n : ℝ) (A B : Point) (h1 : A.y = line k n A.x) (h2 : B.y = line k n B.x)
  (h3 : A.y^2 = parabola A.x) (h4 : B.y^2 = parabola B.x) (h5 : A.x * B.x ≠ 0)
  (h6 : A.x * B.x + A.y * B.y = 0) : n = 4 := by sorry

end part_one_part_two_l442_44227


namespace complex_product_equals_negative_25i_l442_44277

theorem complex_product_equals_negative_25i :
  let Q : ℂ := 3 + 4*Complex.I
  let E : ℂ := -Complex.I
  let D : ℂ := 3 - 4*Complex.I
  Q * E * D = -25 * Complex.I :=
by
  sorry

end complex_product_equals_negative_25i_l442_44277


namespace arccos_one_half_equals_pi_third_l442_44235

theorem arccos_one_half_equals_pi_third : Real.arccos (1/2) = π/3 := by
  sorry

end arccos_one_half_equals_pi_third_l442_44235


namespace school_capacity_l442_44296

theorem school_capacity (total_classrooms : ℕ) 
  (desks_type1 desks_type2 desks_type3 : ℕ) : 
  total_classrooms = 30 →
  desks_type1 = 40 →
  desks_type2 = 35 →
  desks_type3 = 28 →
  (total_classrooms / 5 * desks_type1 + 
   total_classrooms / 3 * desks_type2 + 
   (total_classrooms - total_classrooms / 5 - total_classrooms / 3) * desks_type3) = 982 :=
by
  sorry

#check school_capacity

end school_capacity_l442_44296


namespace polynomial_division_remainder_l442_44279

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℝ),
  X^4 + 1 = (X^2 - 4*X + 6) * q + r ∧
  r.degree < (X^2 - 4*X + 6).degree ∧
  r = 16*X - 59 := by
  sorry

end polynomial_division_remainder_l442_44279


namespace min_value_sum_reciprocals_l442_44216

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  b * c / a + a * c / b + a * b / c ≥ 1 := by
  sorry

end min_value_sum_reciprocals_l442_44216


namespace sum_of_divisors_143_l442_44246

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_143 : sum_of_divisors 143 = 168 := by sorry

end sum_of_divisors_143_l442_44246


namespace expression_simplification_l442_44248

theorem expression_simplification : (((2 + 3 + 6 + 7) / 3) + ((3 * 6 + 9) / 4)) = 12.75 := by
  sorry

end expression_simplification_l442_44248


namespace square_perimeters_sum_l442_44293

theorem square_perimeters_sum (x y : ℝ) 
  (h1 : x^2 + y^2 = 113) 
  (h2 : x^2 - y^2 = 47) 
  (h3 : x ≥ y) : 
  3 * (4 * x) + 4 * y = 48 * Real.sqrt 5 + 4 * Real.sqrt 33 := by
sorry

end square_perimeters_sum_l442_44293


namespace spade_evaluation_l442_44231

def spade (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

theorem spade_evaluation : spade 2 (spade 3 4) = 384 := by
  sorry

end spade_evaluation_l442_44231


namespace dog_age_difference_l442_44209

/-- The age difference between the 1st and 2nd fastest dogs -/
def age_difference (d1 d2 d3 d4 d5 : ℕ) : ℕ := d1 - d2

theorem dog_age_difference :
  ∀ d1 d2 d3 d4 d5 : ℕ,
  (d1 + d5) / 2 = 18 →  -- Average age of 1st and 5th dogs
  d1 = 10 →             -- Age of 1st dog
  d2 = d1 - 2 →         -- Age of 2nd dog
  d3 = d2 + 4 →         -- Age of 3rd dog
  d4 * 2 = d3 →         -- Age of 4th dog
  d5 = d4 + 20 →        -- Age of 5th dog
  age_difference d1 d2 = 2 := by
sorry

end dog_age_difference_l442_44209


namespace simplify_expression_l442_44236

theorem simplify_expression (x : ℝ) : 2*x^3 - (7*x^2 - 9*x) - 2*(x^3 - 3*x^2 + 4*x) = -x^2 + x := by
  sorry

end simplify_expression_l442_44236


namespace power_of_product_l442_44264

theorem power_of_product (a b : ℝ) : (b^2 * a)^3 = a^3 * b^6 := by sorry

end power_of_product_l442_44264


namespace range_of_a_given_points_on_opposite_sides_l442_44289

/-- Given points M(1, -a) and N(a, 1) are on opposite sides of the line 2x-3y+1=0,
    prove that the range of the real number a is -1 < a < 1. -/
theorem range_of_a_given_points_on_opposite_sides (a : ℝ) : 
  (∃ (M N : ℝ × ℝ), 
    M = (1, -a) ∧ 
    N = (a, 1) ∧ 
    (2 * M.1 - 3 * M.2 + 1) * (2 * N.1 - 3 * N.2 + 1) < 0) →
  -1 < a ∧ a < 1 :=
by sorry

end range_of_a_given_points_on_opposite_sides_l442_44289


namespace easter_egg_ratio_l442_44259

def total_eggs : ℕ := 63
def hannah_eggs : ℕ := 42

theorem easter_egg_ratio :
  let helen_eggs := total_eggs - hannah_eggs
  (hannah_eggs : ℚ) / helen_eggs = 2 / 1 := by sorry

end easter_egg_ratio_l442_44259


namespace sin_2alpha_value_l442_44233

theorem sin_2alpha_value (α : Real) (h : Real.sin α + Real.cos α = 1/3) : 
  Real.sin (2 * α) = -8/9 := by
  sorry

end sin_2alpha_value_l442_44233


namespace rectangular_field_area_l442_44218

/-- 
Given a rectangular field with one side of length 20 feet and a perimeter 
(excluding that side) of 85 feet, the area of the field is 650 square feet.
-/
theorem rectangular_field_area : 
  ∀ (length width : ℝ), 
    length = 20 →
    2 * width + length = 85 →
    length * width = 650 := by
  sorry

end rectangular_field_area_l442_44218


namespace netflix_series_seasons_l442_44288

theorem netflix_series_seasons (episodes_per_season : ℕ) (episodes_remaining : ℕ) : 
  episodes_per_season = 20 →
  episodes_remaining = 160 →
  (∃ (total_episodes : ℕ), 
    total_episodes * (1 / 3 : ℚ) = total_episodes - episodes_remaining ∧
    total_episodes / episodes_per_season = 12) :=
by sorry

end netflix_series_seasons_l442_44288


namespace complement_M_intersect_N_l442_44210

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem complement_M_intersect_N :
  (U \ M) ∩ N = {3} := by sorry

end complement_M_intersect_N_l442_44210


namespace articles_count_l442_44201

/-- 
Given:
- The selling price is double the cost price
- The cost price of X articles equals the selling price of 25 articles
Prove that X = 50
-/
theorem articles_count (cost_price selling_price : ℝ) (X : ℕ) 
  (h1 : selling_price = 2 * cost_price) 
  (h2 : X * cost_price = 25 * selling_price) : 
  X = 50 := by
  sorry

end articles_count_l442_44201


namespace white_balls_count_l442_44268

theorem white_balls_count (red : ℕ) (yellow : ℕ) (white : ℕ) 
  (h_red : red = 3)
  (h_yellow : yellow = 2)
  (h_prob : (yellow : ℚ) / (red + yellow + white) = 1/4) :
  white = 3 := by
sorry

end white_balls_count_l442_44268


namespace sqrt_eight_and_one_ninth_l442_44234

theorem sqrt_eight_and_one_ninth (x : ℝ) : x = Real.sqrt (8 + 1/9) → x = Real.sqrt 73 / 3 := by
  sorry

end sqrt_eight_and_one_ninth_l442_44234


namespace sin_polar_complete_circle_l442_44285

open Real

theorem sin_polar_complete_circle (t : ℝ) :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t → ∃ r : ℝ, r = sin θ) →
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t → sin θ = sin (θ + t)) →
  t = 2 * π :=
sorry

end sin_polar_complete_circle_l442_44285


namespace prism_faces_l442_44287

/-- Represents a prism with n-sided polygonal bases -/
structure Prism where
  n : ℕ
  vertices : ℕ := 2 * n
  edges : ℕ := 3 * n
  faces : ℕ := n + 2

/-- Theorem: A prism with 40 as the sum of its vertices and edges has 10 faces -/
theorem prism_faces (p : Prism) (h : p.vertices + p.edges = 40) : p.faces = 10 := by
  sorry


end prism_faces_l442_44287


namespace root_sum_reciprocals_l442_44261

-- Define the polynomial
def f (x : ℂ) : ℂ := x^4 + 10*x^3 + 20*x^2 + 15*x + 6

-- Define the roots
axiom p : ℂ
axiom q : ℂ
axiom r : ℂ
axiom s : ℂ

-- Axiom that p, q, r, s are roots of f
axiom root_p : f p = 0
axiom root_q : f q = 0
axiom root_r : f r = 0
axiom root_s : f s = 0

-- The theorem to prove
theorem root_sum_reciprocals :
  1 / (p * q) + 1 / (p * r) + 1 / (p * s) + 1 / (q * r) + 1 / (q * s) + 1 / (r * s) = -10 / 3 := by
  sorry

end root_sum_reciprocals_l442_44261


namespace logical_equivalence_l442_44204

theorem logical_equivalence (P Q : Prop) :
  (¬P → ¬Q) ↔ (Q → P) := by sorry

end logical_equivalence_l442_44204


namespace magic_square_a_plus_b_l442_44286

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (w y a b z : ℕ)
  (magic_sum : ℕ)
  (top_row : 19 + w + 23 = magic_sum)
  (middle_row : 22 + y + a = magic_sum)
  (bottom_row : b + 18 + z = magic_sum)
  (left_column : 19 + 22 + b = magic_sum)
  (middle_column : w + y + 18 = magic_sum)
  (right_column : 23 + a + z = magic_sum)
  (main_diagonal : 19 + y + z = magic_sum)
  (secondary_diagonal : 23 + y + b = magic_sum)

/-- The sum of a and b in the magic square is 23 -/
theorem magic_square_a_plus_b (ms : MagicSquare) : ms.a + ms.b = 23 := by
  sorry

end magic_square_a_plus_b_l442_44286


namespace telecom_plans_l442_44242

/-- Represents the monthly fee for Plan A given the call duration -/
def plan_a_fee (x : ℝ) : ℝ := 0.4 * x + 50

/-- Represents the monthly fee for Plan B given the call duration -/
def plan_b_fee (x : ℝ) : ℝ := 0.6 * x

theorem telecom_plans :
  (∀ x : ℝ, plan_a_fee x = 0.4 * x + 50) ∧
  (∀ x : ℝ, plan_b_fee x = 0.6 * x) ∧
  (plan_a_fee 300 < plan_b_fee 300) ∧
  (∃ x : ℝ, x = 250 ∧ plan_a_fee x = plan_b_fee x) :=
sorry

end telecom_plans_l442_44242


namespace max_value_of_f_l442_44256

/-- The function f represents the quadratic equation y = -3x^2 + 12x + 4 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 12 * x + 4

/-- The theorem states that the maximum value of f is 16 and occurs at x = 2 -/
theorem max_value_of_f :
  (∃ (x_max : ℝ), f x_max = 16 ∧ ∀ (x : ℝ), f x ≤ f x_max) ∧
  (f 2 = 16 ∧ ∀ (x : ℝ), f x ≤ 16) :=
sorry

end max_value_of_f_l442_44256


namespace paperclips_exceed_200_l442_44255

def paperclips (k : ℕ) : ℕ := 3 * 2^k

theorem paperclips_exceed_200 : ∀ k : ℕ, paperclips k ≤ 200 ↔ k < 7 := by sorry

end paperclips_exceed_200_l442_44255


namespace sandy_marks_per_correct_sum_l442_44205

/-- 
Given:
- Sandy attempts 30 sums
- Sandy obtains 45 marks in total
- Sandy got 21 sums correct
- Sandy loses 2 marks for each incorrect sum

Prove that Sandy gets 3 marks for each correct sum
-/
theorem sandy_marks_per_correct_sum 
  (total_sums : ℕ) 
  (total_marks : ℕ) 
  (correct_sums : ℕ) 
  (penalty_per_incorrect : ℕ) 
  (h1 : total_sums = 30)
  (h2 : total_marks = 45)
  (h3 : correct_sums = 21)
  (h4 : penalty_per_incorrect = 2) :
  (total_marks + penalty_per_incorrect * (total_sums - correct_sums)) / correct_sums = 3 := by
  sorry

end sandy_marks_per_correct_sum_l442_44205
