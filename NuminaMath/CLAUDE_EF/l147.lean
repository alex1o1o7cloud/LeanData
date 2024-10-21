import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_properties_l147_14770

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : ℝ) : ℝ := 2 * Real.sin θ

-- Define the line l in parametric form
def line_l (t : ℝ) : ℝ × ℝ := (1 + t, 2 - t)

-- Standard equation of curve C
def standard_equation_C (x y : ℝ) : Prop :=
  x^2 + (y - 1)^2 = 1

-- Standard equation of line l
def standard_equation_l (x y : ℝ) : Prop :=
  x + y - 3 = 0

-- Equation of line l' parallel to l and tangent to C
def equation_l' (x y : ℝ) : Prop :=
  (x + y - 1 + Real.sqrt 2 = 0) ∨ (x + y - 1 - Real.sqrt 2 = 0)

theorem curve_and_line_properties :
  ∀ (θ t x y : ℝ),
  -- Given the curve C and line l
  (x^2 + y^2 = (curve_C θ)^2) →
  ((x, y) = line_l t) →
  -- Prove the standard equations and the equation of l'
  (standard_equation_C x y ∧
   standard_equation_l x y ∧
   equation_l' x y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_properties_l147_14770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_with_12_factors_l147_14776

theorem least_integer_with_12_factors : 
  ∃ n : ℕ, (n > 0) ∧ 
    (∀ m : ℕ, m > 0 → (Nat.card (Nat.divisors m) = 12) → n ≤ m) ∧ 
    (Nat.card (Nat.divisors n) = 12) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_with_12_factors_l147_14776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coords_of_point_M_l147_14781

/-- Given a point M with Cartesian coordinates (1, -√3), its polar coordinates are (2, -π/3) -/
theorem polar_coords_of_point_M :
  let x : ℝ := 1
  let y : ℝ := -Real.sqrt 3
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x)
  (r = 2) ∧ (θ = -π/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coords_of_point_M_l147_14781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_problem_l147_14771

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_problem :
  ∀ f : ℝ → ℝ,
  (∃ a b c : ℝ, ∀ x, f x = quadratic_function a b c x) →
  f (-1) = -8 →
  (∃ y : ℝ, ∀ x : ℝ, f x ≤ f 2) →
  f 2 = 1 →
  (∀ x, f x = -(x - 2)^2 + 1) ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ = 1 ∧ x₂ = 3) ∧
  f 0 = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_problem_l147_14771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_person_weight_l147_14785

theorem new_person_weight 
  (initial_count : ℕ) 
  (replaced_weight : ℝ) 
  (avg_increase : ℝ) 
  (h1 : initial_count = 5)
  (h2 : replaced_weight = 68)
  (h3 : avg_increase = 5.5)
  : replaced_weight + initial_count * avg_increase = 95.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_person_weight_l147_14785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l147_14742

noncomputable def monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f y < f x

def intersects_x_axis_twice (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0

noncomputable def log_a (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem problem_statement (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let p := monotonically_decreasing (λ x => log_a a (x + 1))
  let q := intersects_x_axis_twice (λ x => x^2 + (2*a - 3)*x + 1)
  (p ∨ q) ∧ ¬(p ∧ q) → a ∈ Set.Icc (1/2) 1 ∪ Set.Ioi (5/2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l147_14742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l147_14760

noncomputable def f (x : ℝ) := 2 * (Real.sin x)^2 + 2 * Real.sin x - 1/2

theorem min_value_of_f :
  ∃ (min : ℝ), min = 1 ∧
  ∀ x, x ∈ Set.Icc (π/6) (5*π/6) →
  f x ≥ min ∧
  ∃ x₀, x₀ ∈ Set.Icc (π/6) (5*π/6) ∧ f x₀ = min :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l147_14760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l147_14791

/-- The speed of a train given its length, time to cross a man, and the man's speed --/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed : ℝ) :
  train_length = 400 ∧ 
  crossing_time = 35.99712023038157 ∧ 
  man_speed = 6 →
  ∃ (train_speed : ℝ), 
    abs (train_speed - 45.9632) < 0.0001 ∧ 
    train_speed * 1000 / 3600 = train_length / crossing_time + man_speed * 1000 / 3600 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l147_14791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_workers_for_profit_is_32_l147_14703

/-- The minimum number of workers needed for a company to make a profit -/
def min_workers_for_profit : ℕ :=
  let maintenance_fee : ℚ := 550
  let setup_cost : ℚ := 200
  let hourly_wage : ℚ := 18
  let widgets_per_hour : ℚ := 6
  let widget_price : ℚ := 3.5
  let work_hours : ℚ := 8

  let daily_wage := hourly_wage * work_hours
  let widgets_per_day := widgets_per_hour * work_hours
  let revenue_per_worker := widgets_per_day * widget_price

  (⌈(maintenance_fee + setup_cost) / (revenue_per_worker - daily_wage)⌉).toNat

theorem min_workers_for_profit_is_32 :
  min_workers_for_profit = 32 := by
  sorry

#eval min_workers_for_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_workers_for_profit_is_32_l147_14703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l147_14735

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x < 0}
def B : Set ℝ := {x | x ≥ 1}

-- Define the complement of B in the universal set of real numbers
def C_U_B : Set ℝ := {x | x < 1}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ C_U_B = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l147_14735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meet_diagonal_corner_l147_14768

/-- Represents a person walking around a rectangle -/
structure Walker where
  speed : ℝ
  startCorner : ℕ
  direction : Int  -- Changed to Int: 1 for clockwise, -1 for counterclockwise

/-- Represents a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- The problem setup -/
noncomputable def setupProblem : Rectangle × Walker × Walker := sorry

/-- The perimeter of the rectangle is 24 -/
axiom perimeter_is_24 : (setupProblem.1.length + setupProblem.1.width) * 2 = 24

/-- The ratio of the rectangle's sides is 3:2 -/
axiom side_ratio : setupProblem.1.length / setupProblem.1.width = 3 / 2

/-- Jane's speed is twice Hector's speed -/
axiom speed_ratio : setupProblem.2.1.speed = 2 * setupProblem.2.2.speed

/-- They start at the same corner -/
axiom same_start : setupProblem.2.1.startCorner = setupProblem.2.2.startCorner

/-- They walk in opposite directions -/
axiom opposite_directions : setupProblem.2.1.direction = -setupProblem.2.2.direction

/-- Function to calculate the meeting point -/
noncomputable def meetingPoint (r : Rectangle) (w1 w2 : Walker) : ℝ := sorry

/-- Theorem: They meet closest to the corner diagonally opposite their starting point -/
theorem meet_diagonal_corner :
  let (r, w1, w2) := setupProblem
  let mp := meetingPoint r w1 w2
  mp > r.length + r.width ∧ mp < 2 * r.length + r.width := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meet_diagonal_corner_l147_14768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_has_six_sides_l147_14709

/-- A convex polygon with interior angles forming an arithmetic sequence -/
structure ConvexPolygon where
  n : ℕ  -- number of sides
  smallest_angle : ℚ  -- smallest interior angle in degrees
  largest_angle : ℚ  -- largest interior angle in degrees
  is_convex : n ≥ 3
  is_arithmetic_sequence : True  -- assume the angles form an arithmetic sequence

/-- The sum of interior angles of a polygon with n sides is (n-2) * 180° -/
def sum_of_interior_angles (p : ConvexPolygon) : ℚ :=
  (p.n - 2) * 180

/-- The sum of interior angles for an arithmetic sequence -/
def sum_of_arithmetic_sequence (p : ConvexPolygon) : ℚ :=
  (p.n : ℚ) / 2 * (p.smallest_angle + p.largest_angle)

/-- Theorem: A convex polygon with interior angles forming an arithmetic sequence,
    where the smallest angle is 100° and the largest angle is 140°, has 6 sides -/
theorem convex_polygon_has_six_sides (p : ConvexPolygon)
  (h1 : p.smallest_angle = 100)
  (h2 : p.largest_angle = 140) :
  p.n = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_has_six_sides_l147_14709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_of_g_l147_14727

noncomputable def f (x : ℝ) : ℝ := (x - 1) / x

noncomputable def g (x : ℝ) : ℝ := f (4 * x) - x

theorem zero_point_of_g :
  ∃ x : ℝ, g x = 0 ∧ x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_of_g_l147_14727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_investment_return_l147_14743

/-- Represents the annual returns for a stock over 4 years -/
structure StockReturns :=
  (year1 : ℝ) (year2 : ℝ) (year3 : ℝ) (year4 : ℝ)

/-- Calculates the compound return for a stock over 4 years -/
noncomputable def compoundReturn (s : StockReturns) : ℝ :=
  (1 + s.year1) * (1 + s.year2) * (1 + s.year3) * (1 + s.year4)

/-- The given stock returns for the 5 stocks -/
def stockA : StockReturns := ⟨0.10, 0.05, -0.15, 0.25⟩
def stockB : StockReturns := ⟨0.08, -0.12, -0.05, 0.20⟩
def stockC : StockReturns := ⟨-0.05, 0.10, 0.15, -0.08⟩
def stockD : StockReturns := ⟨0.18, 0.04, -0.10, -0.02⟩
def stockE : StockReturns := ⟨0.03, -0.18, 0.22, 0.11⟩

/-- The average return across all stocks -/
noncomputable def averageReturn : ℝ :=
  (compoundReturn stockA + compoundReturn stockB + compoundReturn stockC +
   compoundReturn stockD + compoundReturn stockE) / 5

theorem stock_investment_return :
  abs (averageReturn - 1.119251) < 0.000001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_investment_return_l147_14743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_excess_income_l147_14794

def tax_rate_first_40k : ℝ := 0.12
def total_tax : ℝ := 8000
def total_income : ℝ := 56000
def first_40k : ℝ := 40000

theorem tax_rate_excess_income : 
  let tax_on_first_40k := tax_rate_first_40k * first_40k
  let remaining_tax := total_tax - tax_on_first_40k
  let excess_income := total_income - first_40k
  let tax_rate_excess := remaining_tax / excess_income
  tax_rate_excess * 100 = 20 := by
    -- Proof steps would go here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_excess_income_l147_14794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bookstore_problem_l147_14729

/-- The original difference in the number of books between two branches of a bookstore -/
def original_difference : ℤ := 3000

theorem bookstore_problem (branch_a branch_b : ℤ) : 
  branch_a + branch_b = 5000 →
  branch_b + 400 = (branch_a - 400) / 2 - 400 →
  |branch_a - branch_b| = original_difference :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bookstore_problem_l147_14729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_numbers_problem_l147_14773

theorem two_digit_numbers_problem :
  ∀ X Y : ℤ,
  (10 ≤ X ∧ X < 100) →  -- X is a two-digit number
  (10 ≤ Y ∧ Y < 100) →  -- Y is a two-digit number
  (X = 2 * Y) →  -- X is twice Y
  (∃ a b : ℤ, 
    X = 10 * a + b ∧  -- X's digits
    Y = 10 * (a + b) + abs (a - b)) →  -- Y's digits are sum and difference of X's digits
  (X = 34 ∧ Y = 17) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_numbers_problem_l147_14773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_multiples_of_six_l147_14724

theorem two_digit_multiples_of_six : ∃ n : ℕ, n = 15 ∧ 
  n = (Finset.filter (fun x => 10 ≤ x ∧ x ≤ 99 ∧ x % 6 = 0) (Finset.range 100)).card :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_multiples_of_six_l147_14724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l147_14750

/-- A geometric sequence with first term a and common ratio r -/
def geometric_sequence (a r : ℝ) : ℕ → ℝ := λ n ↦ a * r^(n - 1)

/-- The theorem statement -/
theorem geometric_sequence_problem (a r : ℝ) :
  let seq := geometric_sequence a r
  (seq 2 + seq 4 = 20) ∧ (seq 3 + seq 5 = 40) → seq 5 + seq 7 = 160 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l147_14750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_pass_bridge_time_l147_14767

/-- The time (in seconds) it takes for a train to pass a bridge -/
noncomputable def train_pass_time (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

/-- Theorem: A train 360 m long running at 45 kmph takes 40 seconds to pass a bridge 140 m long -/
theorem train_pass_bridge_time :
  train_pass_time 360 140 45 = 40 := by
  -- Unfold the definition of train_pass_time
  unfold train_pass_time
  -- Simplify the expression
  simp
  -- The proof is skipped using sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_pass_bridge_time_l147_14767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_max_value_l147_14777

/-- The function f(x) = sin(x/3) + cos(x/3) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

theorem f_period_and_max_value :
  (∃ (T : ℝ), T > 0 ∧ T = 6 * Real.pi ∧ (∀ x, f (x + T) = f x) ∧ 
    (∀ S, S > 0 → (∀ x, f (x + S) = f x) → T ≤ S)) ∧
  (∃ (M : ℝ), M = Real.sqrt 2 ∧ (∀ x, f x ≤ M) ∧ (∃ y, f y = M)) := by
  sorry

#check f_period_and_max_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_max_value_l147_14777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_schoolchildren_speed_l147_14775

theorem schoolchildren_speed (v : ℝ) (h_pos : v > 0) : ∃ ε > 0, |v - 63.24| < ε :=
  let distance_to_school : ℝ := 400
  let return_speed_increase : ℝ := 60
  let initial_walk_time : ℝ := 3

  have h1 : (400 + 3 * v) / (v + 60) = (400 - 3 * v) / v := by sorry
  have h2 : v^2 = 4000 := by sorry

  sorry

#check schoolchildren_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_schoolchildren_speed_l147_14775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_log_base_2_l147_14784

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the proposed inverse function
noncomputable def g (x : ℝ) : ℝ := 2^x

-- Theorem statement
theorem inverse_log_base_2 : Function.LeftInverse g f ∧ Function.RightInverse g f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_log_base_2_l147_14784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_rectangle_diagonal_l147_14732

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  width : ℝ
  length : ℝ
  perimeter_eq : width * 2 + length * 2 = 30
  length_eq : length = width + 3

/-- The diagonal of a rectangle -/
noncomputable def diagonal (r : SpecialRectangle) : ℝ := 
  Real.sqrt (r.width ^ 2 + r.length ^ 2)

/-- Theorem: The diagonal of the special rectangle is √117 -/
theorem special_rectangle_diagonal :
  ∀ r : SpecialRectangle, diagonal r = Real.sqrt 117 := by
  intro r
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_rectangle_diagonal_l147_14732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_zero_plus_inverse_third_equals_four_l147_14799

theorem power_zero_plus_inverse_third_equals_four :
  (2021 : ℝ)^0 + (1/3 : ℝ)⁻¹ = 4 := by
  have h1 : (2021 : ℝ)^0 = 1 := by exact pow_zero 2021
  have h2 : (1/3 : ℝ)⁻¹ = 3 := by
    rw [inv_eq_one_div]
    norm_num
  calc
    (2021 : ℝ)^0 + (1/3 : ℝ)⁻¹ = 1 + 3 := by rw [h1, h2]
    _ = 4 := by norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_zero_plus_inverse_third_equals_four_l147_14799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l147_14756

variable (a₁ a₂ a₃ a₄ : ℝ)
variable (x₁ x₂ x₃ x₄ : ℝ)

-- Assumption that a₁, a₂, a₃, a₄ are distinct
axiom distinct_a : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₃ ≠ a₄

-- The system of equations
def eq1 (a₁ a₂ a₃ a₄ x₁ x₂ x₃ x₄ : ℝ) : Prop := 
  |a₁ - a₂| * x₂ + |a₁ - a₃| * x₃ + |a₁ - a₄| * x₄ = 1

def eq2 (a₁ a₂ a₃ a₄ x₁ x₂ x₃ x₄ : ℝ) : Prop := 
  |a₂ - a₁| * x₁ + |a₂ - a₃| * x₃ + |a₂ - a₄| * x₄ = 1

def eq3 (a₁ a₂ a₃ a₄ x₁ x₂ x₃ x₄ : ℝ) : Prop := 
  |a₃ - a₁| * x₁ + |a₃ - a₂| * x₂ + |a₃ - a₄| * x₄ = 1

def eq4 (a₁ a₂ a₃ a₄ x₁ x₂ x₃ x₄ : ℝ) : Prop := 
  |a₄ - a₁| * x₁ + |a₄ - a₂| * x₂ + |a₄ - a₃| * x₃ = 1

-- The theorem to prove
theorem solution_exists : 
  ∃ (x₁ x₂ x₃ x₄ : ℝ), 
    x₁ = 1 / (a₁ - a₄) ∧ 
    x₂ = 0 ∧ 
    x₃ = 0 ∧ 
    x₄ = 1 / (a₁ - a₄) ∧
    eq1 a₁ a₂ a₃ a₄ x₁ x₂ x₃ x₄ ∧
    eq2 a₁ a₂ a₃ a₄ x₁ x₂ x₃ x₄ ∧
    eq3 a₁ a₂ a₃ a₄ x₁ x₂ x₃ x₄ ∧
    eq4 a₁ a₂ a₃ a₄ x₁ x₂ x₃ x₄ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l147_14756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_geometric_progression_ratio_l147_14731

/-- A geometric progression with positive terms where any term equals the sum of the next three following terms -/
structure SpecialGeometricProgression where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  a_pos : a > 0
  r_pos : r > 0
  sum_property : ∀ n : ℕ, a * r^n = a * r^(n+1) + a * r^(n+2) + a * r^(n+3)

/-- The common ratio of a special geometric progression -/
noncomputable def special_ratio : ℝ := (-1 + (19 + 3 * Real.sqrt 33) ^ (1/3) + (19 - 3 * Real.sqrt 33) ^ (1/3)) / 3

/-- Theorem stating that the common ratio of a special geometric progression is equal to the special_ratio -/
theorem special_geometric_progression_ratio (gp : SpecialGeometricProgression) : 
  gp.r = special_ratio := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_geometric_progression_ratio_l147_14731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_l147_14759

theorem downstream_distance
  (boat_speed : ℝ)
  (current_speed : ℝ)
  (time_minutes : ℝ)
  (h1 : boat_speed = 20)
  (h2 : current_speed = 5)
  (h3 : time_minutes = 15) :
  boat_speed + current_speed * (time_minutes / 60) = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_l147_14759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_centers_coincide_iff_skew_edges_equal_l147_14713

-- Define the Point type
structure Point where
  x : Real
  y : Real
  z : Real

-- Define the Tetrahedron structure
structure Tetrahedron where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define the circumcenter function
noncomputable def circumcenter (t : Tetrahedron) : Point := sorry

-- Define the incenter function
noncomputable def incenter (t : Tetrahedron) : Point := sorry

-- Define the skew_edges_equal predicate
def skew_edges_equal (t : Tetrahedron) : Prop := sorry

-- State and prove the theorem
theorem tetrahedron_centers_coincide_iff_skew_edges_equal (t : Tetrahedron) :
  circumcenter t = incenter t ↔ skew_edges_equal t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_centers_coincide_iff_skew_edges_equal_l147_14713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_lines_circle_theorem_l147_14741

structure Line where
  -- Placeholder for line definition

structure Point where
  -- Placeholder for point definition

structure Circle where
  -- Placeholder for circle definition

def general_position (l₁ l₂ l₃ l₄ : Line) : Prop :=
  sorry -- Definition of general position for four lines

def point_on_line (p : Point) (l : Line) : Prop :=
  sorry -- Definition of a point being on a line

def points_on_circle (c : Circle) (p₁ p₂ p₃ p₄ : Point) : Prop :=
  sorry -- Definition of points lying on the same circle

def point_for_triple (l₁ l₂ l₃ : Line) : Point :=
  sorry -- Definition of the point corresponding to a triple of lines

theorem four_lines_circle_theorem 
  (l₁ l₂ l₃ l₄ : Line) 
  (p₁ p₂ p₃ p₄ : Point) 
  (c : Circle) :
  general_position l₁ l₂ l₃ l₄ →
  point_on_line p₁ l₁ →
  point_on_line p₂ l₂ →
  point_on_line p₃ l₃ →
  point_on_line p₄ l₄ →
  points_on_circle c p₁ p₂ p₃ p₄ →
  ∃ c' : Circle, points_on_circle c' 
    (point_for_triple l₂ l₃ l₄)
    (point_for_triple l₁ l₃ l₄)
    (point_for_triple l₁ l₂ l₄)
    (point_for_triple l₁ l₂ l₃) := by
  sorry -- Proof skipped


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_lines_circle_theorem_l147_14741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l147_14717

def f (x : ℝ) := 2 * x^2 - 4 * x + 3

theorem quadratic_function_properties :
  (∀ x, f x ≥ 1) ∧  -- minimum value is 1
  f 0 = 3 ∧ f 2 = 3 →
  (∀ x, f x = 2 * x^2 - 4 * x + 3) ∧
  (∀ a : ℝ, (∃ x y, x ∈ Set.Icc (2 * a) (a + 1) ∧ y ∈ Set.Icc (2 * a) (a + 1) ∧ f x < f y ∧ f y < f x) ↔ 0 < a ∧ a < 1/2) ∧
  (∀ m : ℝ, (∀ x ∈ Set.Icc (-1) 1, f x > 2 * x + 2 * m + 1) ↔ m < -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l147_14717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_count_is_1500_l147_14706

/-- A function that checks if a 4-digit number satisfies the given conditions -/
def is_valid (n : ℕ) : Bool :=
  1000 ≤ n ∧ n ≤ 9999 ∧  -- 4-digit number
  (n % 10 ≥ 2 * ((n / 10) % 10)) ∧  -- units digit is at least twice the tens digit
  (n / 1000 % 2 = 1)  -- thousands digit is odd

/-- The count of valid 4-digit numbers -/
def valid_count : ℕ := (List.range 10000).filter is_valid |>.length

/-- The main theorem stating that the count of valid 4-digit numbers is 1500 -/
theorem valid_count_is_1500 : valid_count = 1500 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_count_is_1500_l147_14706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_sum_equality_l147_14748

/-- Represents a die with 8 faces -/
structure Die where
  faces : Finset ℕ
  fair : faces.card = 8

/-- Represents a pair of dice -/
structure DicePair where
  die1 : Die
  die2 : Die

/-- Calculates the probability distribution of sums for a pair of dice -/
noncomputable def sumDistribution (dp : DicePair) : Finset ℕ → ℚ :=
  sorry

/-- Kelvin's standard 8-sided dice pair -/
def kelvinDice : DicePair :=
  sorry

/-- Alex's specially labeled 8-sided dice pair -/
def alexDice (a b : ℕ) : DicePair :=
  sorry

/-- The main theorem -/
theorem dice_sum_equality
  (a b : ℕ)
  (h_neq : a ≠ b)
  (h_equal_dist : sumDistribution (alexDice a b) = sumDistribution kelvinDice) :
  min a b ∈ ({24, 28, 32} : Set ℕ) := by
  sorry

#check dice_sum_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_sum_equality_l147_14748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_books_bought_at_cost_price_proof_l147_14720

/-- The number of books bought at cost price, given specific cost and selling price conditions -/
def books_bought_at_cost_price : ℕ := 8

/-- Proof of the theorem -/
theorem books_bought_at_cost_price_proof : books_bought_at_cost_price = 8 := by
  -- Define constants
  let cost_price : ℝ := 1  -- Arbitrary non-zero value
  let selling_price : ℝ := cost_price * (1 - 0.5)  -- 50% less than cost price
  let num_books : ℕ := 16

  -- State the main equation
  have h1 : cost_price * books_bought_at_cost_price = selling_price * num_books := by
    sorry

  -- Define the relationship between selling price and cost price
  have h2 : selling_price = cost_price * (1 - 0.5) := by
    sorry

  -- Prove that books_bought_at_cost_price equals 8
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_books_bought_at_cost_price_proof_l147_14720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_amounts_correct_l147_14789

noncomputable def total_sales : ℝ := 5000

def credit_percent : ℝ := 35
def cash_percent : ℝ := 25
def debit_percent : ℝ := 20
def check_percent : ℝ := 10
def electronic_percent : ℝ := 10

noncomputable def credit_amount : ℝ := total_sales * (credit_percent / 100)
noncomputable def cash_amount : ℝ := total_sales * (cash_percent / 100)
noncomputable def debit_amount : ℝ := total_sales * (debit_percent / 100)
noncomputable def check_amount : ℝ := total_sales * (check_percent / 100)
noncomputable def electronic_amount : ℝ := total_sales * (electronic_percent / 100)

theorem payment_amounts_correct :
  credit_amount = 1750 ∧
  cash_amount = 1250 ∧
  debit_amount = 1000 ∧
  check_amount = 500 ∧
  electronic_amount = 500 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_amounts_correct_l147_14789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_area_of_triangle_l147_14739

/-- The area of the circumcircle of a triangle ABC with given side lengths and angle -/
theorem circumcircle_area_of_triangle (a b c : ℝ) (angle_A : ℝ) : 
  a = 3 → b = 2 → angle_A = π / 3 →
  let c_squared := a^2 + b^2 - 2 * a * b * Real.cos angle_A
  let c := Real.sqrt c_squared
  let R := c / (2 * Real.sin angle_A)
  π * R^2 = 7 * π / 3 := by
  sorry

#check circumcircle_area_of_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_area_of_triangle_l147_14739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_series_sum_l147_14774

def alternating_series (start : ℕ) (step : ℕ) (end_val : ℕ) : List ℤ :=
  let terms := List.range ((start - end_val) / step + 1)
  terms.map (λ i => (if i % 2 = 0 then -1 else 1) * (start - i * step))

theorem alternating_series_sum :
  (alternating_series 2020 10 20).sum = -950 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_series_sum_l147_14774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_problem_l147_14744

theorem log_problem (x : ℝ) (h1 : x < 1) (h2 : (Real.log x)^3 - Real.log (x^3) = 243 * (Real.log 10)) :
  (Real.log x)^4 - Real.log (x^4) = 6597 * (Real.log 10) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_problem_l147_14744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_at_corner_l147_14782

/-- Represents the faces of a cube --/
inductive CubeFaces
  | Face1 | Face2 | Face3 | Face4 | Face5 | Face6

/-- The values on the faces of the cube --/
def face_values : CubeFaces → ℕ
  | CubeFaces.Face1 => 3
  | CubeFaces.Face2 => 4
  | CubeFaces.Face3 => 5
  | CubeFaces.Face4 => 6
  | CubeFaces.Face5 => 7
  | CubeFaces.Face6 => 8

/-- Predicate to check if three faces can meet at a corner --/
def can_meet_at_corner (a b c : CubeFaces) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  (face_values a + face_values b + face_values c ≠ 7) ∧
  (face_values a + face_values b + face_values c ≠ 11) ∧
  (face_values a + face_values b + face_values c ≠ 15)

/-- The theorem to be proved --/
theorem max_product_at_corner : 
  (∀ a b c : CubeFaces, can_meet_at_corner a b c → 
    face_values a * face_values b * face_values c ≤ 280) ∧
  (∃ a b c : CubeFaces, can_meet_at_corner a b c ∧ 
    face_values a * face_values b * face_values c = 280) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_at_corner_l147_14782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_division_multiplication_l147_14725

theorem cube_root_division_multiplication : 
  (Real.rpow (6 / 18) (1 / 3 : ℝ)) * 2 = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_division_multiplication_l147_14725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l147_14786

-- Statement 1
def divisible_by_three (n : ℕ) : Prop := n % 3 = 0

-- Statement 2
noncomputable def circle_area_increase (r : ℝ) : ℝ := (Real.pi * (1.2 * r)^2) / (Real.pi * r^2) - 1

-- Statement 3
def count_divisors (n : ℕ) : ℕ := (Finset.filter (λ x => n % x = 0) (Finset.range (n + 1))).card

-- Statement 4
def geometric_progression (a : ℝ) (n : ℕ) : ℝ := a * (-2)^(n - 1)

-- Statement 5
def arithmetic_progression (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Statement 6
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

theorem problem_solution :
  (¬ (divisible_by_three 123 ∧ divisible_by_three 365 ∧ divisible_by_three 293 ∧ divisible_by_three 18)) ∧
  (circle_area_increase 1 = 0.44) ∧
  (¬ (count_divisors 45 > count_divisors 36)) ∧
  (∃ a : ℝ, (geometric_progression a 1 + geometric_progression a 2 + geometric_progression a 3) / 3 = geometric_progression a 1) ∧
  (∀ a₁ d : ℝ, arithmetic_progression a₁ d 10 < 5 → arithmetic_progression a₁ d 12 > 7 → d > 1) ∧
  (is_perfect_square 640000000000) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l147_14786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_pattern_area_l147_14740

/-- The area of a shaded region formed by semicircles -/
theorem semicircle_pattern_area (diameter : ℝ) (pattern_length : ℝ) : 
  diameter = 3 →
  pattern_length = 18 →
  (pattern_length / diameter * π * (diameter / 2)^2) = (27/2) * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_pattern_area_l147_14740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l147_14707

/-- Represents a hyperbola in the Cartesian coordinate system -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h : a > 0 ∧ b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Theorem: A hyperbola with an asymptote y = √3x has eccentricity 2 -/
theorem hyperbola_eccentricity (h : Hyperbola) 
    (asymptote : h.b / h.a = Real.sqrt 3) : eccentricity h = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l147_14707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_needed_l147_14736

/-- The number of posts to be painted -/
def num_posts : ℕ := 20

/-- The height of each post in feet -/
def post_height : ℝ := 15

/-- The diameter of each post in feet -/
def post_diameter : ℝ := 8

/-- The area that one gallon of paint can cover in square feet -/
def paint_coverage : ℝ := 300

/-- Calculates the minimum number of full gallons of paint needed -/
noncomputable def min_gallons_needed : ℕ := 
  let post_radius : ℝ := post_diameter / 2
  let single_post_area : ℝ := 2 * Real.pi * post_radius * post_height
  let total_area : ℝ := single_post_area * (num_posts : ℝ)
  let gallons_float : ℝ := total_area / paint_coverage
  Int.toNat (Int.ceil gallons_float)

/-- Theorem stating that the minimum number of gallons needed is 26 -/
theorem paint_needed : min_gallons_needed = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_needed_l147_14736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_reflection_150_degrees_l147_14722

noncomputable def rotation_matrix (θ : Real) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![Real.cos θ, -Real.sin θ; Real.sin θ, Real.cos θ]

def reflection_x_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  !![ 1,  0;
      0, -1]

noncomputable def combined_transformation (θ : Real) : Matrix (Fin 2) (Fin 2) ℝ :=
  reflection_x_matrix * rotation_matrix θ

theorem rotation_reflection_150_degrees :
  combined_transformation (150 * π / 180) = !![-(Real.sqrt 3) / 2,  1 / 2;
                                                1 / 2, (Real.sqrt 3) / 2] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_reflection_150_degrees_l147_14722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_point_properties_l147_14766

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 25

-- Define point P
def point_P : ℝ × ℝ := (2, 1)

-- Define a function to check if a point is inside the circle
def is_inside_circle (p : ℝ × ℝ) : Prop :=
  (p.1 + 1)^2 + (p.2 - 2)^2 < 25

-- Define a function to represent a line
def line (a b c : ℝ) (x y : ℝ) : Prop := a*x + b*y + c = 0

-- Define a function to check if a line passes through a point
def passes_through (a b c : ℝ) (p : ℝ × ℝ) : Prop :=
  line a b c p.1 p.2

-- Define a function to calculate the chord length cut by a line on the circle
noncomputable def chord_length (a b c : ℝ) : ℝ :=
  2 * Real.sqrt (25 - ((a + b*2 - c)^2 / (a^2 + b^2)))

theorem circle_and_point_properties :
  is_inside_circle point_P ∧
  (∃ a b c : ℝ, 
    passes_through a b c point_P ∧ 
    chord_length a b c = 8 ∧
    ((a = 4 ∧ b = 3 ∧ c = -11) ∨ (a = 1 ∧ b = 0 ∧ c = -2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_point_properties_l147_14766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_constructible_iff_l147_14792

/-- Represents a quadrilateral with sides a, b, c, d and midpoint distance l -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  l : ℝ

/-- Condition for constructibility of a quadrilateral -/
def is_constructible (q : Quadrilateral) : Prop :=
  ∃ m : ℝ,
    abs (q.b - q.d) < 2 * q.l ∧
    2 * q.l < q.b + q.d ∧
    abs (q.a - q.c) < 2 * m ∧
    2 * m < q.a + q.c

/-- Theorem stating the necessary and sufficient conditions for quadrilateral constructibility -/
theorem quadrilateral_constructible_iff (q : Quadrilateral) :
  is_constructible q ↔
  (∃ (A B C D : ℝ × ℝ), 
    -- ABCD is a quadrilateral with sides q.a, q.b, q.c, q.d
    -- and midpoint distance q.l between AB and CD
    True) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_constructible_iff_l147_14792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_n_A_is_fibonacci_go_kart_routes_l147_14751

/-- Represents the number of routes ending at point A after n minutes -/
def M_n_A : ℕ → ℕ
| 0 => 0
| 1 => 0
| 2 => 1
| n + 3 => M_n_A (n + 1) + M_n_A n

/-- The Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| n + 2 => fib n + fib (n + 1)

/-- The main theorem: M_n_A follows the Fibonacci sequence -/
theorem M_n_A_is_fibonacci (n : ℕ) : M_n_A (2 * n) = fib n := by
  sorry

/-- The solution to the original problem -/
theorem go_kart_routes : M_n_A 10 = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_n_A_is_fibonacci_go_kart_routes_l147_14751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_value_at_27_l147_14788

-- Define the polynomial h
def h (x : ℝ) : ℝ := x^3 - 2*x + 1

-- Define the properties of k
def k_properties (k : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, k = λ x ↦ (x - a^3) * (x - b^3) * (x - c^3)) ∧
  (k 0 = 1) ∧
  (∀ r : ℝ, h r = 0 → k (r^3) = 0)

theorem k_value_at_27 (k : ℝ → ℝ) (h_k : k_properties k) : k 27 = -704 := by
  sorry

#check k_value_at_27

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_value_at_27_l147_14788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_equals_given_l147_14716

def U : Set ℕ := {2, 4, 6, 8}

def A (m : ℤ) : Set ℕ := {2, (|m - 6|).toNat}

theorem complement_A_equals_given (m : ℤ) : 
  (A m ⊆ U) → (U \ A m = {6, 8}) → (m = 2 ∨ m = 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_equals_given_l147_14716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_real_axis_length_l147_14764

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the point P
def point_P : ℝ × ℝ := (2, -1)

-- Define the tangent line to the circle at P
def tangent_line (x y : ℝ) : Prop := y = 2*x - 5

-- Define the hyperbola C
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : (x y : ℝ) → Prop
  centered_at_origin : equation 0 0
  symmetric_about_axes : ∀ x y, equation x y ↔ equation (-x) y ∧ equation x (-y)
  intersects_circle : equation point_P.1 point_P.2
  asymptote_parallel_to_tangent : b/a = 2

-- State the theorem
theorem hyperbola_real_axis_length (C : Hyperbola) : 
  2 * C.a = Real.sqrt 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_real_axis_length_l147_14764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinearity_condition_l147_14765

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def collinear (a b : V) : Prop :=
  ∃! (k : ℝ), b = k • a

theorem collinearity_condition (a b : V) (ha : a ≠ 0) :
  collinear a b ↔ ∃! (k : ℝ), b = k • a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinearity_condition_l147_14765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_shorter_when_closer_to_lamp_l147_14780

/-- Represents the distance between a person and a street lamp -/
def distance_to_lamp : ℝ → ℝ := sorry

/-- Represents the length of a person's shadow -/
def shadow_length : ℝ → ℝ := sorry

/-- States that as the distance to the lamp decreases, the shadow length decreases -/
theorem shadow_shorter_when_closer_to_lamp :
  ∀ (d₁ d₂ : ℝ), d₁ < d₂ → distance_to_lamp d₁ < distance_to_lamp d₂ →
  shadow_length d₁ < shadow_length d₂ := by
  sorry

#check shadow_shorter_when_closer_to_lamp

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_shorter_when_closer_to_lamp_l147_14780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_cost_l147_14714

-- Define the variables
variable (x y z a b c : ℝ)

-- Define the conditions
def room_areas (x y z : ℝ) : Prop := x < y ∧ y < z
def paint_costs (a b c : ℝ) : Prop := a < b ∧ b < c

-- Define the cost functions
def cost_A (x y z a b c : ℝ) : ℝ := a*x + b*y + c*z
def cost_B (x y z a b c : ℝ) : ℝ := a*z + b*y + c*x
def cost_C (x y z a b c : ℝ) : ℝ := a*y + b*z + c*x
def cost_D (x y z a b c : ℝ) : ℝ := a*y + b*x + c*z

-- Theorem statement
theorem lowest_cost (h1 : room_areas x y z) (h2 : paint_costs a b c) :
  cost_B x y z a b c < cost_A x y z a b c ∧ 
  cost_B x y z a b c < cost_C x y z a b c ∧ 
  cost_B x y z a b c < cost_D x y z a b c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_cost_l147_14714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_k_value_l147_14705

/-- Three points are collinear if they lie on the same straight line -/
def are_collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

theorem collinear_points_k_value :
  ∃! k : ℝ, are_collinear (1, 2) (3, 8) (4, k/3) ∧ k = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_k_value_l147_14705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l147_14701

-- Define the function f(x)
noncomputable def f (x : ℝ) := Real.log x + 2 * x - 7

-- State the theorem
theorem root_in_interval :
  -- Conditions
  (∀ x > 0, ContinuousAt (f) x) →  -- f is continuous on (0, +∞)
  (∀ x y, 0 < x ∧ x < y → f x < f y) →  -- f is monotonically increasing on (0, +∞)
  f 2 < 0 →  -- f(2) < 0
  0 < f 3 →  -- f(3) > 0
  -- Conclusion
  ∃ x, 2 < x ∧ x < 3 ∧ f x = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l147_14701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_CDFE_is_four_ninths_l147_14712

/-- Rectangle with given dimensions and points -/
structure Rectangle where
  length : ℝ
  width : ℝ
  E : ℝ  -- Position of E on AB
  F : ℝ  -- Position of F on AD

/-- The area of quadrilateral CDFE in the given rectangle -/
noncomputable def area_CDFE (r : Rectangle) : ℝ :=
  -- Area of triangle EGF + Area of triangle EGC
  (1/2 * (1/3) * (2/3)) + (1/2 * 1 * (2/3))

/-- Theorem stating the area of CDFE in the specific rectangle -/
theorem area_CDFE_is_four_ninths :
  ∀ (r : Rectangle),
    r.length = 2 ∧ 
    r.width = 1 ∧ 
    r.E = 1/3 ∧ 
    r.F = 2/3 →
    area_CDFE r = 4/9 :=
by
  intro r h
  simp [area_CDFE]
  -- The proof steps would go here
  sorry

#eval (1/9 : ℚ) + (1/3 : ℚ)  -- To verify the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_CDFE_is_four_ninths_l147_14712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_implies_a_range_l147_14708

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + a * (1/2)^x + (1/4)^x

-- State the theorem
theorem f_upper_bound_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f a x ≤ 3) → a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_implies_a_range_l147_14708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_insufficient_information_for_unique_compound_l147_14796

/-- Represents the mass percentage of an element in a compound -/
def MassPercentage := Float

/-- Represents a chemical compound -/
structure Compound where
  name : String
  oxygenPercentage : MassPercentage

/-- Given a mass percentage of oxygen, this function returns all possible compounds
    that match that percentage -/
noncomputable def possibleCompounds (oxygenPercentage : MassPercentage) : Set Compound :=
  sorry

theorem insufficient_information_for_unique_compound :
  ∃ (c1 c2 : Compound), c1 ≠ c2 ∧ c1.oxygenPercentage = (28.57 : Float) ∧ c2.oxygenPercentage = (28.57 : Float) :=
by
  sorry

#eval (28.57 : Float)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_insufficient_information_for_unique_compound_l147_14796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_block_length_is_30_7_l147_14797

/-- The length of the wooden block in meters -/
noncomputable def block_length : ℝ :=
  31 - (30 / 100)

/-- Theorem stating that the length of the wooden block is 30.7 meters -/
theorem block_length_is_30_7 : block_length = 30.7 := by
  -- Unfold the definition of block_length
  unfold block_length
  -- Simplify the arithmetic expression
  simp [sub_eq_add_neg, div_eq_mul_inv]
  -- Prove equality using real number properties
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_block_length_is_30_7_l147_14797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_l147_14723

-- Define the ceiling function {m}
noncomputable def ceiling (m : ℝ) : ℤ := Int.ceil m

-- Define the floor function [m]
noncomputable def floor (m : ℝ) : ℤ := Int.floor m

-- Define the problem
theorem problem (x y : ℤ) 
  (eq1 : 3 * (floor (x : ℝ)) + 2 * (ceiling (y : ℝ)) = 2011)
  (eq2 : 2 * (ceiling (x : ℝ)) - (floor (y : ℝ)) = 2) :
  x + y = 861 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_l147_14723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_skew_iff_b_neq_l147_14772

/-- Two lines in ℝ³ given by their parametric equations -/
structure Line3D where
  point : Fin 3 → ℝ
  direction : Fin 3 → ℝ

/-- Check if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  ∃ (b : ℝ), l1.point 1 = b ∧ 
  ¬∃ (t u : ℝ), 
    (∀ i : Fin 3, l1.point i + t * l1.direction i = l2.point i + u * l2.direction i)

theorem lines_skew_iff_b_neq (b : ℝ) : 
  are_skew 
    (Line3D.mk (fun i => [2, b, 4].get i) (fun i => [3, 4, 5].get i))
    (Line3D.mk (fun i => [3, 2, 1].get i) (fun i => [6, 5, 2].get i))
  ↔ 
  b ≠ 448/105 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_skew_iff_b_neq_l147_14772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_square_min_area_l147_14710

/-- Regular polygon with n sides --/
structure RegularPolygon where
  n : ℕ
  a : ℝ  -- side length
  R : ℝ  -- circumradius
  r : ℝ  -- inradius
  h_n : n ≥ 3
  h_R : R > 0
  h_r : r > 0
  h_a : a > 0
  h_bounds : R - a / 2 ≤ r

/-- Circle k with radius ρ --/
def CircleK (poly : RegularPolygon) (ρ : ℝ) : Prop :=
  poly.R - poly.a / 2 ≤ ρ ∧ ρ ≤ poly.r

/-- Covered area by circle k and n touching circles --/
noncomputable def CoveredArea (poly : RegularPolygon) (ρ : ℝ) : ℝ :=
  Real.pi * ρ^2 + poly.n * (Real.pi / 2 - Real.pi / poly.n) * (poly.R - ρ)^2

theorem triangle_max_area (poly : RegularPolygon) (h_triangle : poly.n = 3) :
  ∃ ρ, CircleK poly ρ ∧
    ∀ ρ', CircleK poly ρ' → CoveredArea poly ρ ≥ CoveredArea poly ρ' := by
  sorry

theorem square_min_area (poly : RegularPolygon) (h_square : poly.n = 4) :
  ∃ ρ, CircleK poly ρ ∧
    ∀ ρ', CircleK poly ρ' → CoveredArea poly ρ ≤ CoveredArea poly ρ' := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_square_min_area_l147_14710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l147_14761

def M : Set ℝ := {x | (2 - x) / (x + 1) ≥ 0}
def N : Set ℝ := Set.univ

theorem intersection_M_N : M ∩ N = Set.Ioo (-1 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l147_14761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_good_placements_count_l147_14795

/-- Represents a card with two distinct numbers -/
structure Card where
  num1 : Fin 5
  num2 : Fin 5
  h : num1 ≠ num2

/-- Represents a placement of cards into boxes -/
def Placement := Fin 5 → Finset Card

/-- The set of all possible cards -/
def allCards : Finset Card :=
  sorry

/-- A placement is valid if each card is in a correct box -/
def isValidPlacement (p : Placement) : Prop :=
  ∀ c ∈ allCards, ∃ i : Fin 5, c ∈ p i ∧ (c.num1 = i ∨ c.num2 = i)

/-- A placement is good if box 1 has more cards than any other box -/
def isGoodPlacement (p : Placement) : Prop :=
  ∀ i : Fin 5, i ≠ 0 → (p 0).card > (p i).card

/-- Assume Placement is finite -/
instance : Fintype Placement := sorry

/-- Assume isValidPlacement is decidable -/
instance (p : Placement) : Decidable (isValidPlacement p) := sorry

/-- Assume isGoodPlacement is decidable -/
instance (p : Placement) : Decidable (isGoodPlacement p) := sorry

/-- The main theorem: there are 120 good placements -/
theorem good_placements_count :
  (Finset.filter (λ p : Placement => isValidPlacement p ∧ isGoodPlacement p) (Finset.univ)).card = 120 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_good_placements_count_l147_14795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_relation_triangle_perimeter_l147_14734

/-- Definition of a triangle with sides a, b, c and angles A, B, C -/
def triangle (a b c A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  A > 0 ∧ B > 0 ∧ C > 0 ∧
  A + B + C = Real.pi ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if 3(sin²A + sin²C) = 2sinAsinC + 8sinAsinCcosB, then a + c = 2b. -/
theorem triangle_side_relation (a b c A B C : ℝ) 
  (h_triangle : triangle a b c A B C)
  (h_relation : 3 * (Real.sin A ^ 2 + Real.sin C ^ 2) = 
                2 * Real.sin A * Real.sin C + 
                8 * Real.sin A * Real.sin C * Real.cos B) :
  a + c = 2 * b :=
by sorry

/-- If cosB = 11/14 and the area of triangle ABC is 15/4√3, 
    find the perimeter of triangle ABC. -/
theorem triangle_perimeter (a b c A B C : ℝ) 
  (h_triangle : triangle a b c A B C)
  (h_cosB : Real.cos B = 11 / 14)
  (h_area : (1 / 2) * a * c * Real.sin B = (15 / 4) * Real.sqrt 3) :
  a + b + c = 15 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_relation_triangle_perimeter_l147_14734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_well_defined_sum_never_zero_l147_14762

/-- Definition of the sequences x_n, y_n, and z_n -/
def x : ℕ → ℚ
  | 0 => 2
  | n + 1 => 2 * x n / ((x n)^2 - 1)

def y : ℕ → ℚ
  | 0 => 4
  | n + 1 => 2 * y n / ((y n)^2 - 1)

def z : ℕ → ℚ
  | 0 => 6/7
  | n + 1 => 2 * z n / ((z n)^2 - 1)

/-- The sequences are well-defined for all natural numbers n -/
theorem sequences_well_defined : ∀ n : ℕ, x n ≠ 0 ∧ y n ≠ 0 ∧ z n ≠ 0 := by
  sorry

/-- The sum of x_n, y_n, and z_n is never zero -/
theorem sum_never_zero : ∀ n : ℕ, x n + y n + z n ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_well_defined_sum_never_zero_l147_14762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratatouille_cost_per_quart_l147_14779

/-- Calculates the cost per quart of ratatouille given the ingredients and their prices --/
theorem ratatouille_cost_per_quart 
  (eggplant_quantity : ℕ) (eggplant_price : ℚ)
  (zucchini_quantity : ℕ) (zucchini_price : ℚ)
  (tomato_quantity : ℕ) (tomato_price : ℚ)
  (onion_quantity : ℕ) (onion_price : ℚ)
  (basil_quantity : ℕ) (basil_half_pound_price : ℚ)
  (yield_quarts : ℕ)
  (h1 : eggplant_quantity = 5)
  (h2 : eggplant_price = 2)
  (h3 : zucchini_quantity = 4)
  (h4 : zucchini_price = 2)
  (h5 : tomato_quantity = 4)
  (h6 : tomato_price = 7/2)
  (h7 : onion_quantity = 3)
  (h8 : onion_price = 1)
  (h9 : basil_quantity = 1)
  (h10 : basil_half_pound_price = 5/2)
  (h11 : yield_quarts = 4) :
  (eggplant_quantity * eggplant_price + 
   zucchini_quantity * zucchini_price + 
   tomato_quantity * tomato_price + 
   onion_quantity * onion_price + 
   basil_quantity * (2 * basil_half_pound_price)) / yield_quarts = 10 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratatouille_cost_per_quart_l147_14779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_possible_price_max_price_is_nine_l147_14730

def entrance_fee : ℚ := 5
def num_caps : ℚ := 20
def total_budget : ℚ := 200
def tax_rate : ℚ := 8 / 100

noncomputable def max_price_per_cap : ℚ :=
  ⌊(total_budget - entrance_fee) / (1 + tax_rate) / num_caps⌋

theorem highest_possible_price (price : ℚ) :
  (price ≤ max_price_per_cap) ↔
  (price * num_caps * (1 + tax_rate) + entrance_fee ≤ total_budget) :=
by sorry

theorem max_price_is_nine : max_price_per_cap = 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_possible_price_max_price_is_nine_l147_14730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ap_sum_15_is_180_l147_14763

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
structure ArithmeticProgression where
  first : ℝ
  diff : ℝ

/-- The nth term of an arithmetic progression -/
noncomputable def ArithmeticProgression.nthTerm (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  ap.first + (n - 1 : ℝ) * ap.diff

/-- The sum of the first n terms of an arithmetic progression -/
noncomputable def ArithmeticProgression.sumFirstN (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  n / 2 * (2 * ap.first + (n - 1 : ℝ) * ap.diff)

/-- Theorem: In an arithmetic progression where the sum of the 4th and 12th terms is 24,
    the sum of the first 15 terms is 180. -/
theorem ap_sum_15_is_180 (ap : ArithmeticProgression) 
    (h : ap.nthTerm 4 + ap.nthTerm 12 = 24) : 
    ap.sumFirstN 15 = 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ap_sum_15_is_180_l147_14763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l147_14718

noncomputable def f (ω a x : ℝ) : ℝ := (1/2) * (Real.sin (ω * x) + a * Real.cos (ω * x))

theorem problem_solution (a : ℝ) (ω : ℝ) (h1 : 0 < ω) (h2 : ω ≤ 1)
  (h3 : ∀ x, f ω a x = f ω a (π/3 - x))
  (h4 : ∀ x, f ω a (x - π) = f ω a (x + π)) :
  ω = 1 ∧
  a = Real.sqrt 3 ∧
  (∀ x, f ω a x = Real.sin (x + π/3)) ∧
  (∀ x₁ x₂, x₁ ≠ x₂ → x₁ ∈ Set.Ioo (-π/3) (5*π/3) → x₂ ∈ Set.Ioo (-π/3) (5*π/3) →
    f ω a x₁ = -1/2 → f ω a x₂ = -1/2 → x₁ + x₂ = 7*π/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l147_14718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_bc_equation_l147_14798

/-- Given a triangle ABC with specific properties, prove that the equation of side BC is 2x + 9y - 65 = 0 -/
theorem side_bc_equation (A B C : ℝ × ℝ) : 
  A = (3, -1) →
  (∃ k : ℝ, ∀ x y : ℝ, (x, y) ∈ Set.range (λ t : ℝ ↦ (1 - t) • A + t • ((B + C) / 2)) ↔ 6*x + 10*y = 59) →
  (∃ m : ℝ, ∀ x y : ℝ, (x, y) ∈ Set.range (λ t : ℝ ↦ (1 - t) • B + t • ((A + C) / 2)) ↔ x - 4*y + 10 = 0) →
  ∃ n : ℝ, ∀ x y : ℝ, (x, y) ∈ Set.range (λ t : ℝ ↦ (1 - t) • B + t • C) ↔ 2*x + 9*y = 65 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_bc_equation_l147_14798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l147_14783

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 2 then (a - 2) * x else (1/2)^x - 1

theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  a ∈ Set.Iic (13/8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l147_14783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_continuous_positive_function_satisfies_conditions_l147_14721

open MeasureTheory Measure Real

theorem no_continuous_positive_function_satisfies_conditions 
  (α : ℝ) : ¬∃ (f : ℝ → ℝ), 
  (ContinuousOn f (Set.Icc 0 1)) ∧ 
  (∀ x ∈ Set.Icc 0 1, f x > 0) ∧
  (∫ x in Set.Icc 0 1, f x = 1) ∧
  (∫ x in Set.Icc 0 1, x * f x = α) ∧
  (∫ x in Set.Icc 0 1, x^2 * f x = α^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_continuous_positive_function_satisfies_conditions_l147_14721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_k_value_l147_14752

/-- Given vectors a and b, if (a + 2b) is parallel to (3a - b), then k = -6 -/
theorem parallel_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
  (ha : a = (1, 3))
  (hb : b = (-2, k))
  (h_parallel : ∃ (c : ℝ), c ≠ 0 ∧ (a + 2 • b) = c • (3 • a - b)) :
  k = -6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_k_value_l147_14752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C_increases_l147_14700

variable (e R r : ℝ)

noncomputable def C (e R r n : ℝ) : ℝ := (e * n^2) / (R + n * r)

theorem C_increases (e R r : ℝ) (h_e : e > 0) (h_R : R > 0) (h_r : r > 0) :
  ∀ n₁ n₂ : ℝ, 0 < n₁ → n₁ < n₂ → C e R r n₁ < C e R r n₂ := by
  sorry

#check C_increases

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C_increases_l147_14700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bounces_to_target_height_l147_14793

/-- The number of bounces required for a ball to reach a height less than 2 feet -/
def num_bounces : ℕ := 9

/-- The initial height of the ball in feet -/
noncomputable def initial_height : ℝ := 20

/-- The ratio of the new height to the previous height after each bounce -/
noncomputable def bounce_ratio : ℝ := 3/4

/-- The target height in feet -/
noncomputable def target_height : ℝ := 2

/-- Theorem stating that num_bounces is the smallest natural number satisfying the condition -/
theorem min_bounces_to_target_height :
  (∀ k : ℕ, k < num_bounces → initial_height * bounce_ratio ^ k ≥ target_height) ∧
  (initial_height * bounce_ratio ^ num_bounces < target_height) := by
  sorry

#check min_bounces_to_target_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bounces_to_target_height_l147_14793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l147_14737

theorem trigonometric_identities :
  (∀ (x y : ℝ), x = 20 * π / 180 ∧ y = 40 * π / 180 →
    Real.tan x + Real.tan y + Real.sqrt 3 * Real.tan x * Real.tan y = Real.sqrt 3)
  ∧
  (∀ (z w : ℝ), z = 50 * π / 180 ∧ w = 10 * π / 180 →
    Real.sin z * (1 + Real.sqrt 3 * Real.tan w) = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l147_14737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_range_l147_14702

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = y

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the line passing through two points
noncomputable def line_through (A B : PointOnParabola) (x : ℝ) : ℝ :=
  (B.y - A.y) / (B.x - A.x) * (x - A.x) + A.y

-- Define the y-intercept of the line
noncomputable def y_intercept (A B : PointOnParabola) : ℝ :=
  line_through A B 0

-- Define the dot product of two vectors from origin to points
def dot_product (A B : PointOnParabola) : ℝ :=
  A.x * B.x + A.y * B.y

theorem parabola_intersection_range (A B : PointOnParabola) 
  (h1 : A.x < 0) (h2 : B.x > 0) (h3 : dot_product A B > 0) :
  y_intercept A B > 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_range_l147_14702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_grazing_area_at_B_max_grazing_area_value_l147_14745

/-- Represents a point on the edge of the pond -/
inductive StakePosition
| A
| B
| C
| D

/-- Calculates the grazing area for a given stake position -/
noncomputable def grazingArea (pos : StakePosition) : ℝ :=
  match pos with
  | StakePosition.A => 8.25 * Real.pi
  | StakePosition.B => 12 * Real.pi
  | StakePosition.C => 8.25 * Real.pi
  | StakePosition.D => 8 * Real.pi

/-- Theorem stating that position B maximizes the grazing area -/
theorem max_grazing_area_at_B :
  ∀ pos : StakePosition, grazingArea StakePosition.B ≥ grazingArea pos :=
by sorry

/-- Corollary: The maximum grazing area is exactly 12π square meters -/
theorem max_grazing_area_value :
  (grazingArea StakePosition.B) = 12 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_grazing_area_at_B_max_grazing_area_value_l147_14745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_integer_sum_l147_14787

theorem inequality_system_integer_sum : ∃ (S : Finset ℤ), 
  (∀ x ∈ S, (-x - 2*(x+1) ≤ 1 ∧ (x+1)/3 > x-1)) ∧ 
  (∀ x : ℤ, (-x - 2*(x+1) ≤ 1 ∧ (x+1)/3 > x-1) → x ∈ S) ∧
  (S.sum id) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_integer_sum_l147_14787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l147_14747

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.sin x * Real.sin (x + Real.pi/2) + (Real.cos x)^2 - 1/2

theorem f_properties :
  ∃ (monotonic_decrease : Set ℝ) (range : Set ℝ) (min_phi : ℝ),
    -- 1. Intervals of monotonic decrease
    (∀ k : ℤ, Set.Icc (k * Real.pi + Real.pi/6) (k * Real.pi + 2*Real.pi/3) ⊆ monotonic_decrease) ∧
    (∀ x y : ℝ, x ∈ monotonic_decrease → y ∈ monotonic_decrease → x < y → f x ≥ f y) ∧
    
    -- 2. Range on [0, π/2]
    range = Set.Icc (-1/2) 1 ∧
    (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi/2) → f x ∈ range) ∧
    (∀ y : ℝ, y ∈ range → ∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi/2) ∧ f x = y) ∧
    
    -- 3. Minimum value of φ
    min_phi = 5*Real.pi/6 ∧
    (∀ φ : ℝ, φ > 0 →
      (∀ x : ℝ, f ((x + φ)/2) = -f ((-x + φ)/2)) →
      φ ≥ min_phi) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l147_14747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_to_parallel_plane_perpendicular_to_parallel_planes_l147_14746

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- Proposition 1
theorem perpendicular_to_parallel_plane 
  (m n : Line) (α : Plane) 
  (h1 : perpendicular m α) 
  (h2 : parallel_line_plane n α) :
  perpendicular_lines m n :=
sorry

-- Proposition 4
theorem perpendicular_to_parallel_planes 
  (m : Line) (α β γ : Plane)
  (h1 : parallel_planes α β)
  (h2 : parallel_planes β γ)
  (h3 : perpendicular m α) :
  perpendicular m γ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_to_parallel_plane_perpendicular_to_parallel_planes_l147_14746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_basketball_shot_probability_l147_14778

/-- The probability of making a single shot -/
noncomputable def p : ℝ := 0.4

/-- The number of shots taken -/
def n : ℕ := 10

/-- The number of successful shots required -/
def k : ℕ := 4

/-- The maximum allowed ratio of successful shots to total shots at any point -/
noncomputable def max_ratio : ℝ := 0.4

/-- The probability of the specific scenario described in the problem -/
noncomputable def scenario_probability : ℝ := 25 * 2^4 * 3^6 / 5^10

/-- 
  Theorem stating that the probability of the specific scenario 
  (4 successful shots out of 10, with the 10th shot being successful, 
  and maintaining a ratio ≤ 0.4 at all points before) 
  is equal to 25 * 2^4 * 3^6 / 5^10
-/
theorem basketball_shot_probability : 
  scenario_probability = 25 * 2^4 * 3^6 / 5^10 := by
  sorry

#check basketball_shot_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_basketball_shot_probability_l147_14778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_growth_approximation_l147_14711

/-- The growth factor for an annual increase of 1/6 -/
noncomputable def growthFactor : ℝ := 1 + 1/6

/-- The initial amount in rupees -/
def initialAmount : ℝ := 64000

/-- The number of years -/
def years : ℕ := 2

/-- The final amount after compound growth -/
noncomputable def finalAmount : ℝ := initialAmount * growthFactor ^ years

/-- Theorem stating that the final amount is approximately 87030.40 -/
theorem compound_growth_approximation :
  abs (finalAmount - 87030.40) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_growth_approximation_l147_14711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_curve_satisfies_conditions_l147_14719

/-- The integral curve of the differential equation y'' + 2y' + 2y = 0 
    passing through (0, 1) and tangent to y = x + 1 at that point -/
noncomputable def integral_curve (x : ℝ) : ℝ :=
  Real.exp (-x) * (Real.cos x + 2 * Real.sin x)

/-- The differential equation -/
def differential_equation (y : ℝ → ℝ) : Prop :=
  ∀ x, (deriv (deriv y)) x + 2 * (deriv y x) + 2 * (y x) = 0

theorem integral_curve_satisfies_conditions :
  differential_equation integral_curve ∧ 
  integral_curve 0 = 1 ∧ 
  (deriv integral_curve) 0 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_curve_satisfies_conditions_l147_14719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_magnitude_l147_14757

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define vectors m and n
noncomputable def m (t : Triangle) (A B : ℝ) : ℝ × ℝ := 
  (Real.sin B - Real.sin A, Real.sqrt 3 * t.a + t.c)

noncomputable def n (t : Triangle) (C : ℝ) : ℝ × ℝ := 
  (Real.sin C, t.a + t.b)

-- State the theorem
theorem angle_B_magnitude (t : Triangle) (A B C : ℝ) :
  (∃ k : ℝ, m t A B = k • (n t C)) →  -- m and n are parallel
  B = 5 * Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_magnitude_l147_14757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_area_ratio_l147_14726

/-- Given a semicircle with diameter AB = 2r and two smaller semicircles with diameters AC = r and CB = r,
    where CD is perpendicular to AB, the ratio of the area between the large semicircle and the two
    smaller semicircles to the area of a circle with radius r is 1/4. -/
theorem semicircle_area_ratio (r : ℝ) (h : r > 0) : 
  (π * r^2 / 2 - 2 * (π * (r/2)^2 / 2)) / (π * r^2) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_area_ratio_l147_14726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_a_f_a_min_f_a_max_l147_14754

/-- Given a linear function f(x) = ax + b where the integral of f(x)^2 from -1 to 1 equals 2,
    prove that the range of f(a) is [-1, 37/12]. -/
theorem range_of_f_a (a b : ℝ) (h : ∫ (x : ℝ) in Set.Icc (-1) 1, (a * x + b)^2 = 2) :
  ∀ y, y = a^2 + b → -1 ≤ y ∧ y ≤ 37/12 := by
sorry

/-- Prove that f(a) achieves its minimum value of -1. -/
theorem f_a_min (a b : ℝ) (h : ∫ (x : ℝ) in Set.Icc (-1) 1, (a * x + b)^2 = 2) :
  ∃ a₀ b₀, a₀^2 + b₀ = -1 := by
sorry

/-- Prove that f(a) achieves its maximum value of 37/12. -/
theorem f_a_max (a b : ℝ) (h : ∫ (x : ℝ) in Set.Icc (-1) 1, (a * x + b)^2 = 2) :
  ∃ a₀ b₀, a₀^2 + b₀ = 37/12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_a_f_a_min_f_a_max_l147_14754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_l147_14728

/-- The distance formula from a point to a line -/
noncomputable def distanceToLine (x y : ℝ) (a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2)

/-- Theorem: If P is on the x-axis and its distance to 3x - 4y + 6 = 0 is 6, then its x-coordinate is -12 or 8 -/
theorem point_on_line (x : ℝ) :
  (distanceToLine x 0 3 (-4) 6 = 6) →
  (x = -12 ∨ x = 8) := by
  intro h
  sorry

#check point_on_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_l147_14728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_slant_height_calculation_l147_14715

/-- Represents the properties of a cone -/
structure Cone where
  radius : ℝ
  slantHeight : ℝ
  curvedSurfaceArea : ℝ

/-- The formula for the curved surface area of a cone -/
noncomputable def curvedSurfaceAreaFormula (c : Cone) : ℝ :=
  Real.pi * c.radius * c.slantHeight

/-- Theorem: Given a cone with radius 5 cm and curved surface area 157.07963267948966 cm², 
    its slant height is 10 cm -/
theorem cone_slant_height_calculation (c : Cone) 
    (h1 : c.radius = 5)
    (h2 : c.curvedSurfaceArea = 157.07963267948966) :
    c.slantHeight = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_slant_height_calculation_l147_14715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2α_value_l147_14733

noncomputable def a (α : ℝ) : ℝ × ℝ := (1/3, Real.tan α)
noncomputable def b (α : ℝ) : ℝ := Real.cos α

theorem cos_2α_value (α : ℝ) 
  (h_parallel : ∃ k : ℝ, k ≠ 0 ∧ a α = (k * b α, k * Real.tan α)) : 
  Real.cos (2 * α) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2α_value_l147_14733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_two_equations_have_solutions_l147_14758

theorem at_least_two_equations_have_solutions (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ∃ (i j : Fin 3), i ≠ j ∧
    (∃ x : ℝ, match i with
      | 0 => (x - b) * (x - c) = x - a
      | 1 => (x - c) * (x - a) = x - b
      | 2 => (x - a) * (x - b) = x - c) ∧
    (∃ y : ℝ, match j with
      | 0 => (y - b) * (y - c) = y - a
      | 1 => (y - c) * (y - a) = y - b
      | 2 => (y - a) * (y - b) = y - c) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_two_equations_have_solutions_l147_14758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_problem_l147_14749

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {2, 4}
def B : Set ℕ := {3, 4}

theorem complement_union_problem : (U \ A) ∪ B = {1, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_problem_l147_14749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_ABC_l147_14738

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the conditions
def conditions (a : ℝ) : Prop :=
  ∃ (b c : ℝ), triangle_ABC a b c ∧ c = 2 ∧ b = Real.sqrt 2 * a

-- Define the area function
noncomputable def area (a : ℝ) : ℝ :=
  let b := Real.sqrt 2 * a
  let c := 2
  Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4

-- State the theorem
theorem max_area_triangle_ABC :
  ∃ (max_area : ℝ), max_area = 2 * Real.sqrt 2 ∧
  ∀ (a : ℝ), conditions a → area a ≤ max_area :=
by
  sorry

#check max_area_triangle_ABC

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_ABC_l147_14738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_tangent_to_circumcircle_l147_14755

-- Define a right triangle ABC
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  right_angle : (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0

-- Define the circumcenter
noncomputable def circumcenter (t : RightTriangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1) / 2, (t.A.2 + t.B.2) / 2)

-- Define the incenter
noncomputable def incenter (t : RightTriangle) : ℝ × ℝ :=
  sorry  -- Actual calculation of incenter coordinates

-- Define the circumradius
noncomputable def circumradius (t : RightTriangle) : ℝ :=
  sorry  -- Actual calculation of circumradius

-- Define the inradius
noncomputable def inradius (t : RightTriangle) : ℝ :=
  sorry  -- Actual calculation of inradius

-- Define the homothety transformation
def homothety (center : ℝ × ℝ) (scale : ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  (center.1 + scale * (point.1 - center.1), center.2 + scale * (point.2 - center.2))

-- Theorem statement
theorem incircle_tangent_to_circumcircle (t : RightTriangle) :
  let O := circumcenter t
  let I := incenter t
  let I' := homothety t.C 2 I
  let R := circumradius t
  let r := inradius t
  Real.sqrt ((I'.1 - O.1)^2 + (I'.2 - O.2)^2) = R - 2*r :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_tangent_to_circumcircle_l147_14755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_opposite_angles_l147_14704

/-- Represents a tetrahedron with equal opposite edges -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0

/-- Calculates the angle opposite a pair of edges in the tetrahedron -/
noncomputable def oppositeAngle (t : Tetrahedron) (x y z : ℝ) : ℝ :=
  Real.arccos ((y^2 - x^2) / z^2)

/-- Theorem stating the angles opposite each pair of opposite edges in the tetrahedron -/
theorem tetrahedron_opposite_angles (t : Tetrahedron) :
  (oppositeAngle t t.b t.a t.c, oppositeAngle t t.a t.c t.b, oppositeAngle t t.c t.b t.a) =
  (Real.arccos ((t.b^2 - t.a^2) / t.c^2),
   Real.arccos ((t.a^2 - t.c^2) / t.b^2),
   Real.arccos ((t.c^2 - t.b^2) / t.a^2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_opposite_angles_l147_14704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_sum_at_specific_points_l147_14769

open Real

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 3 * sin (ω * x) * cos (ω * x) - 4 * (cos (ω * x))^2

-- State the theorem
theorem function_sum_at_specific_points
  (ω : ℝ) (θ : ℝ) (h_ω_pos : ω > 0) (h_period : ∀ x, f ω (x + π) = f ω x) (h_f_theta : f ω θ = 1/2) :
  f ω (θ + π/2) + f ω (θ - π/4) = -13/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_sum_at_specific_points_l147_14769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_diagonal_length_l147_14753

/-- Given a quadrilateral PQRS with right angles at Q and R, and side lengths PQ = 7, QR = 12, RS = 25,
    prove that the length of PS is √313. -/
theorem quadrilateral_diagonal_length (P Q R S : ℝ × ℝ) :
  let pq := ‖P - Q‖
  let qr := ‖Q - R‖
  let rs := ‖R - S‖
  let ps := ‖P - S‖
  pq = 7 ∧ 
  qr = 12 ∧ 
  rs = 25 ∧
  (Q.1 - P.1) * (R.1 - Q.1) + (Q.2 - P.2) * (R.2 - Q.2) = 0 ∧
  (R.1 - Q.1) * (S.1 - R.1) + (R.2 - Q.2) * (S.2 - R.2) = 0 →
  ps = Real.sqrt 313 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_diagonal_length_l147_14753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_graph_properties_l147_14790

/-- A graph defined on vertices (x, y) where x, y ∈ {0, 1, ..., p-1} for a prime p -/
def ModularGraph (p : ℕ) (hp : Nat.Prime p) := (Fin p) × (Fin p)

/-- An edge exists between vertices if and only if xx' + yy' ≡ 1 (mod p) -/
def ModularGraphEdge {p : ℕ} (hp : Nat.Prime p) (v w : ModularGraph p hp) : Prop :=
  (v.1 * w.1 + v.2 * w.2) % p = 1

/-- A path of length 4 in the graph -/
def Path4 {p : ℕ} (hp : Nat.Prime p) (a b c d : ModularGraph p hp) : Prop :=
  ModularGraphEdge hp a b ∧ ModularGraphEdge hp b c ∧ ModularGraphEdge hp c d ∧ ModularGraphEdge hp d a

/-- The number of edges in the graph -/
def EdgeCount (p : ℕ) : ℕ :=
  (p^3 - 2*p^2 + 3*p - 2) / 2

theorem modular_graph_properties {p : ℕ} (hp : Nat.Prime p) :
  (∀ a b c d : ModularGraph p hp, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a → ¬Path4 hp a b c d) ∧
  EdgeCount p ≥ (p^3 / 2) - p^2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_graph_properties_l147_14790
