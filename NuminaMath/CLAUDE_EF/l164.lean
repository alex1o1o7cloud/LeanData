import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_square_area_is_94_l164_16422

/-- Represents a cube with green border and white square on each face -/
structure DecoratedCube where
  edge_length : ℚ
  total_green_paint : ℚ

/-- Calculates the area of the white square on each face of the decorated cube -/
def white_square_area (cube : DecoratedCube) : ℚ :=
  let face_area := cube.edge_length ^ 2
  let green_area_per_face := cube.total_green_paint / 6
  face_area - green_area_per_face

/-- Theorem stating that for a cube with edge length 12 and 300 sq ft of green paint,
    the area of the white square on each face is 94 sq ft -/
theorem white_square_area_is_94 :
  let cube := DecoratedCube.mk 12 300
  white_square_area cube = 94 := by
  -- Unfold the definition of white_square_area
  unfold white_square_area
  -- Simplify the arithmetic
  simp [DecoratedCube.mk]
  -- The proof is complete
  rfl

#eval white_square_area (DecoratedCube.mk 12 300)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_square_area_is_94_l164_16422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_X_l164_16445

-- Define the points as pairs of real numbers
def Point := ℝ × ℝ

-- Define the distance between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem sum_of_coordinates_X (X Y Z : Point) : 
  distance X Z / distance X Y = 1/2 →
  distance Z Y / distance X Y = 1/2 →
  Y = (1, 7) →
  Z = (-1, -7) →
  X.1 + X.2 = -24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_X_l164_16445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_range_l164_16496

theorem quadratic_root_range (a : ℝ) : 
  (∃ x : ℝ, 0 < x ∧ x ≤ 3 ∧ a * x^2 + x + 3 * a + 1 = 0) →
  (a ∈ Set.Icc (-(1/2)) (-(1/3))) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_range_l164_16496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_a_value_l164_16450

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 + a^2 * x) / (a^2 - x)

theorem symmetry_implies_a_value (a : ℝ) :
  (∀ x : ℝ, f a x = -2 - f a (2 - x)) → (a = 1 ∨ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_a_value_l164_16450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l164_16442

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 - Real.sqrt (5 - Real.sqrt (6 - x)))

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Icc (-19 : ℝ) 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l164_16442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_minutes_to_fill_barrels_l164_16472

/-- Represents the state of Mac's barrel-filling process -/
structure BarrelState where
  energy : ℕ
  filled_barrels : ℕ

/-- Represents a single action Mac can take -/
inductive MacAction
  | Rest
  | Fill (k : ℕ)

/-- Applies an action to a state, returning the new state -/
def apply_action (state : BarrelState) (action : MacAction) : BarrelState :=
  match action with
  | MacAction.Rest => ⟨state.energy + 1, state.filled_barrels⟩
  | MacAction.Fill k => 
    if k ≤ state.energy then
      ⟨state.energy - k, state.filled_barrels + state.energy * (k + 1)⟩
    else
      state

/-- Returns true if all barrels are filled -/
def all_filled (state : BarrelState) : Prop :=
  state.filled_barrels ≥ 2012

/-- Theorem: The minimal number of minutes to fill 2012 barrels is 46 -/
theorem min_minutes_to_fill_barrels : 
  ∃ (actions : List MacAction), 
    actions.length = 46 ∧ 
    all_filled (actions.foldl apply_action ⟨0, 0⟩) ∧
    ∀ (other_actions : List MacAction),
      other_actions.length < 46 → 
      ¬(all_filled (other_actions.foldl apply_action ⟨0, 0⟩)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_minutes_to_fill_barrels_l164_16472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_2014_integers_l164_16440

theorem exact_2014_integers (a : ℕ) : 
  (∃! (s : Finset ℕ), s.card = 2014 ∧ ∀ b ∈ s, (2 : ℚ) ≤ (a : ℚ) / b ∧ (a : ℚ) / b ≤ 5) ↔ 
  (a = 6710 ∨ a = 6712 ∨ a = 6713) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_2014_integers_l164_16440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_nine_to_twelve_l164_16489

theorem fourth_root_nine_to_twelve : ((9 : ℝ) ^ (1/4)) ^ 12 = 729 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_nine_to_twelve_l164_16489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l164_16420

-- Define the triangle and its properties
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  m : ℝ × ℝ
  n : ℝ × ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.m = (Real.sin t.A, Real.cos t.A))
  (h2 : t.n = (Real.cos t.B, Real.sin t.B))
  (h3 : t.m.1 * t.n.1 + t.m.2 * t.n.2 = Real.sin (2 * t.C))
  (h4 : t.A + t.B + t.C = Real.pi)
  (h5 : Real.sin t.A * Real.sin t.B = (Real.sin t.C)^2)
  (h6 : t.a * t.b * Real.cos t.C = 18)
  (h7 : t.c^2 = t.a * t.b) :
  t.C = Real.pi / 3 ∧ t.c = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l164_16420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l164_16454

-- Define the function f(x)
noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x + m / x

-- State the theorem
theorem function_properties :
  ∃ (m : ℝ), 
    (f 1 m = 5) ∧ 
    (∀ x₁ x₂ : ℝ, 2 < x₁ ∧ x₁ < x₂ → f x₁ m < f x₂ m) ∧
    (f (5/2) m = 41/10) ∧
    (f (10/3) m = 68/15) :=
by
  -- Introduce m and prove it equals 4
  use 4
  constructor
  · -- Prove f(1) = 5
    simp [f]
    norm_num
  constructor
  · -- Prove monotonicity
    intros x₁ x₂ h
    simp [f]
    sorry -- Detailed proof omitted
  constructor
  · -- Prove f(5/2) = 41/10
    simp [f]
    norm_num
  · -- Prove f(10/3) = 68/15
    simp [f]
    norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l164_16454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_A_different_from_M_l164_16423

/-- Two points in polar coordinates are equivalent if they represent the same point in the plane --/
def polar_equivalent (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : Prop :=
  (r1 = r2 ∧ ∃ k : ℤ, θ1 = θ2 + 2 * Real.pi * k) ∨
  (r1 = -r2 ∧ ∃ k : ℤ, θ1 = θ2 + Real.pi + 2 * Real.pi * k)

theorem point_A_different_from_M : 
  ¬ polar_equivalent 5 (-Real.pi/3) (-5) (Real.pi/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_A_different_from_M_l164_16423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_is_five_percent_l164_16497

/-- Calculates the discount percentage given cost price, marked price, and profit percentage -/
noncomputable def discount_percentage (cost_price marked_price profit_percentage : ℚ) : ℚ :=
  let selling_price := cost_price * (1 + profit_percentage / 100)
  let discount := marked_price - selling_price
  (discount / marked_price) * 100

/-- Theorem stating that the discount percentage is 5% given the problem conditions -/
theorem discount_percentage_is_five_percent :
  discount_percentage (47.50 : ℚ) 65 30 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_is_five_percent_l164_16497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_neg_480_deg_l164_16429

theorem cos_neg_480_deg : Real.cos (-480 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_neg_480_deg_l164_16429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_A_paradox_l164_16400

-- Define the types of statements
inductive Statement
| Liar : Statement
| TwoPlusTwoIsFive : Statement

-- Define the types of speakers
inductive Speaker
| Knight : Speaker
| Liar : Speaker

-- Define the statement made by A
def statement_A : Statement → Prop
| Statement.Liar => True
| Statement.TwoPlusTwoIsFive => True

-- Define the truth value of statements
def is_true : Statement → Prop
| Statement.Liar => False
| Statement.TwoPlusTwoIsFive => False

-- Define how speakers evaluate statements
def evaluates (s : Speaker) (p : Statement → Prop) : Prop :=
  match s with
  | Speaker.Knight => p Statement.Liar ∨ p Statement.TwoPlusTwoIsFive
  | Speaker.Liar => ¬(p Statement.Liar ∨ p Statement.TwoPlusTwoIsFive)

-- Theorem stating that A's statement leads to a contradiction
theorem statement_A_paradox :
  ¬∃ (s : Speaker), (evaluates s statement_A) = (is_true (Statement.Liar) ∨ is_true (Statement.TwoPlusTwoIsFive)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_A_paradox_l164_16400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_solid_properties_l164_16432

/-- Represents a rectangular solid with dimensions in geometric progression -/
structure RectangularSolid where
  a : ℝ  -- Common ratio
  r : ℝ  -- Geometric progression ratio

/-- The volume of the rectangular solid -/
noncomputable def volume (s : RectangularSolid) : ℝ := s.a^3

/-- The surface area of the rectangular solid -/
noncomputable def surfaceArea (s : RectangularSolid) : ℝ := 2 * (s.a^2/s.r + s.a^2 * s.r + s.a^2)

/-- The sum of the lengths of all edges of the rectangular solid -/
noncomputable def edgeSum (s : RectangularSolid) : ℝ := 4 * (s.a/s.r + s.a + s.a*s.r)

/-- Theorem stating the properties of the specific rectangular solid -/
theorem rectangular_solid_properties :
  ∃ (s : RectangularSolid), 
    volume s = 8 ∧ 
    surfaceArea s = 32 ∧ 
    edgeSum s = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_solid_properties_l164_16432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_travel_time_l164_16412

/-- The time for two objects traveling perpendicular to each other to reach a certain distance apart -/
noncomputable def time_to_distance (v1 v2 d : ℝ) : ℝ :=
  d / Real.sqrt (v1^2 + v2^2)

/-- Theorem stating that the time for two objects traveling at 10 mph and 12 mph
    to be 130 miles apart is 65/√61 hours -/
theorem perpendicular_travel_time :
  time_to_distance 10 12 130 = 65 / Real.sqrt 61 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_travel_time_l164_16412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_sales_theorem_l164_16410

/-- Represents the sales and pricing data for a supermarket's fruit sales --/
structure FruitSales where
  initial_cost : ℝ
  initial_price : ℝ
  june_sales : ℝ
  august_sales : ℝ
  sales_increase_rate : ℝ
  target_profit : ℝ

/-- Calculates the monthly average growth rate between June and August --/
noncomputable def monthly_growth_rate (data : FruitSales) : ℝ :=
  (data.august_sales / data.june_sales) ^ (1/2) - 1

/-- Calculates the price reduction needed to achieve the target profit in September --/
noncomputable def price_reduction (data : FruitSales) : ℝ :=
  ((data.initial_price - data.initial_cost) * data.august_sales - data.target_profit) /
  (data.sales_increase_rate * (data.initial_price - data.initial_cost) - data.august_sales)

/-- Theorem stating the correctness of the calculated monthly growth rate and price reduction --/
theorem fruit_sales_theorem (data : FruitSales) 
  (h1 : data.initial_cost = 25)
  (h2 : data.initial_price = 40)
  (h3 : data.june_sales = 256)
  (h4 : data.august_sales = 400)
  (h5 : data.sales_increase_rate = 5)
  (h6 : data.target_profit = 4250) :
  monthly_growth_rate data = 0.25 ∧ price_reduction data = 5 := by
  sorry

-- Remove the #eval statements as they are not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_sales_theorem_l164_16410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_result_value_l164_16478

noncomputable def weighted_average (sum : ℝ) (weight : ℝ) : ℝ := sum / weight

theorem sixth_result_value
  (n : ℕ)
  (total_average : ℝ)
  (first_six_average : ℝ)
  (first_six_weight : ℝ)
  (last_six_average : ℝ)
  (last_six_weight : ℝ)
  (sixth_weight : ℝ)
  (h1 : n = 11)
  (h2 : total_average = 58)
  (h3 : first_six_average = 54)
  (h4 : first_six_weight = 20)
  (h5 : last_six_average = 62)
  (h6 : last_six_weight = 30)
  (h7 : sixth_weight = 10) :
  ∃ (first_five_sum last_five_sum sixth_result : ℝ),
    weighted_average (first_five_sum + sixth_result) first_six_weight = first_six_average ∧
    weighted_average (sixth_result + last_five_sum) last_six_weight = last_six_average ∧
    weighted_average (first_five_sum + sixth_result + last_five_sum) (first_six_weight + last_six_weight - sixth_weight) = total_average ∧
    sixth_result = 620 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_result_value_l164_16478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_root_modulus_one_l164_16438

theorem complex_root_modulus_one (n : ℕ) :
  (∃ z : ℂ, z^(n+1) - z^n - 1 = 0 ∧ Complex.abs z = 1) ↔ (n + 2) % 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_root_modulus_one_l164_16438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_specific_l164_16447

/-- The area of a rhombus with given side length and diagonal difference -/
noncomputable def rhombus_area (side_length : ℝ) (diagonal_difference : ℝ) : ℝ :=
  let x := (side_length^2 - (diagonal_difference/2)^2) / (2 * diagonal_difference)
  (2*x) * (2*x + diagonal_difference) / 2

/-- Theorem: The area of a rhombus with side length 9 and diagonals differing by 10 is 72 -/
theorem rhombus_area_specific : rhombus_area 9 10 = 72 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_specific_l164_16447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_condition_for_perpendicular_lines_l164_16411

-- Define the types for lines and planes
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
def Line (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] := V
def Plane (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] := V

-- Define the relationships between lines and planes
def parallel {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (l : Line V) (p : Plane V) : Prop := sorry
def perpendicular {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (l : Line V) (p : Plane V) : Prop := sorry
def parallel_planes {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (p1 p2 : Plane V) : Prop := sorry
def perpendicular_lines {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (l1 l2 : Line V) : Prop := sorry

theorem sufficient_condition_for_perpendicular_lines
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (a b : Line V) (α β : Plane V)
  (h1 : a ≠ b)
  (h2 : α ≠ β)
  (h3 : parallel a α)
  (h4 : perpendicular b β)
  (h5 : parallel_planes α β) :
  perpendicular_lines a b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_condition_for_perpendicular_lines_l164_16411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_entrance_fee_increase_l164_16428

/-- Represents the fair entrance and ride costs --/
structure FairCosts where
  under18Fee : ℚ
  ridePrice : ℚ

/-- Represents the group's fair activities --/
structure FairVisit where
  numUnder18 : ℕ
  numRidesEach : ℕ
  totalSpent : ℚ

/-- Calculates the percentage increase in entrance fee for those over 18 --/
def entranceFeeIncreasePercent (costs : FairCosts) (visit : FairVisit) : ℚ :=
  let under18Total := costs.under18Fee * visit.numUnder18 + 
                      costs.ridePrice * visit.numUnder18 * visit.numRidesEach
  let over18Spent := visit.totalSpent - under18Total
  let over18Fee := over18Spent - costs.ridePrice * visit.numRidesEach
  let increase := over18Fee - costs.under18Fee
  (increase / costs.under18Fee) * 100

theorem fair_entrance_fee_increase 
  (costs : FairCosts) 
  (visit : FairVisit) 
  (h1 : costs.under18Fee = 5)
  (h2 : costs.ridePrice = 1/2)
  (h3 : visit.numUnder18 = 2)
  (h4 : visit.numRidesEach = 3)
  (h5 : visit.totalSpent = 41/2) :
  entranceFeeIncreasePercent costs visit = 20 := by
  sorry

#eval entranceFeeIncreasePercent 
  { under18Fee := 5, ridePrice := 1/2 } 
  { numUnder18 := 2, numRidesEach := 3, totalSpent := 41/2 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_entrance_fee_increase_l164_16428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_maximum_l164_16460

/-- Geometric sequence with common ratio √2 -/
noncomputable def geometric_sequence (a₁ : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => geometric_sequence a₁ n * Real.sqrt 2

/-- Sum of the first n terms of the geometric sequence -/
noncomputable def S (a₁ : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - (Real.sqrt 2) ^ n) / (1 - Real.sqrt 2)

/-- Definition of T_n -/
noncomputable def T (a₁ : ℝ) (n : ℕ+) : ℝ :=
  (17 * S a₁ n - S a₁ (2 * n)) / (geometric_sequence a₁ n)

/-- The index at which T_n reaches its maximum value -/
def n₀ : ℕ+ := 4

/-- Theorem stating that T_n reaches its maximum at n₀ -/
theorem T_maximum (a₁ : ℝ) (h : a₁ > 0) :
  ∀ n : ℕ+, T a₁ n₀ ≥ T a₁ n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_maximum_l164_16460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l164_16455

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 30 / 3600 → speed * time * 1000 = 500 := by
  intros h_speed h_time
  rw [h_speed, h_time]
  norm_num
  
#check train_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l164_16455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_unsolvable_equation_l164_16433

theorem infinite_unsolvable_equation :
  ∃ (N : Set ℕ), Set.Infinite N ∧
    ∀ n ∈ N, ¬∃ (x y z : ℤ), x^2 + y^11 - z^(Nat.factorial 2022) = n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_unsolvable_equation_l164_16433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_increase_l164_16482

/-- Calculates the simple interest for a given principal, rate, and time -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

theorem interest_rate_increase (originalRate : ℝ) : 
  let principal : ℝ := 400
  let time : ℝ := 10
  let increasedInterest : ℝ := simpleInterest principal originalRate time + 200
  ∃ increasedRate : ℝ, 
    simpleInterest principal increasedRate time = increasedInterest ∧ 
    (increasedRate - originalRate) / originalRate = 0.5 := by
  sorry

#check interest_rate_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_increase_l164_16482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_two_axes_symmetry_axes_perpendicular_to_sides_l164_16414

-- Define a rectangle
structure Rectangle where
  length : ℝ
  width : ℝ
  length_positive : length > 0
  width_positive : width > 0

-- Define an axis of symmetry for a rectangle
def AxisOfSymmetry (r : Rectangle) := 
  {axis : Set (ℝ × ℝ) // ∀ p : ℝ × ℝ, 
    p ∈ axis ↔ p ∈ Set.univ}

-- State the theorem
theorem rectangle_two_axes_symmetry (r : Rectangle) : 
  ∃! (axes : Finset (AxisOfSymmetry r)), axes.card = 2 := by
  sorry

-- Additional helper theorem to capture the perpendicularity property
theorem axes_perpendicular_to_sides (r : Rectangle) 
  (axes : Finset (AxisOfSymmetry r)) : 
  axes.card = 2 → ∀ axis ∈ axes, True := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_two_axes_symmetry_axes_perpendicular_to_sides_l164_16414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_distinct_values_l164_16487

-- Define the expression
def base : ℕ := 3
def exponent_height : ℕ := 4

-- Define the function to calculate the number of distinct values
def distinct_values (b : ℕ) (h : ℕ) : ℕ :=
  -- The implementation is not provided, as per the instructions
  sorry

-- Theorem statement
theorem three_distinct_values : distinct_values base exponent_height = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_distinct_values_l164_16487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_inequality_solution_set_l164_16403

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x + Real.pi / 4)

-- Define the solution set
def solution_set : Set ℝ :=
  {x | ∃ k : ℤ, Real.pi/24 + k*Real.pi/2 ≤ x ∧ x < Real.pi/8 + k*Real.pi/2}

-- Theorem statement
theorem tan_inequality_solution_set :
  {x : ℝ | f x ≥ Real.sqrt 3} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_inequality_solution_set_l164_16403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_in_xz_plane_l164_16456

theorem equidistant_point_in_xz_plane :
  let p : ℝ × ℝ × ℝ := (61/24, 0, 5/48)
  let p1 : ℝ × ℝ × ℝ := (1, 0, 0)
  let p2 : ℝ × ℝ × ℝ := (3, 1, 3)
  let p3 : ℝ × ℝ × ℝ := (4, 3, -2)
  let dist (a b : ℝ × ℝ × ℝ) := Real.sqrt ((a.1 - b.1)^2 + (a.2.1 - b.2.1)^2 + (a.2.2 - b.2.2)^2)
  dist p p1 = dist p p2 ∧ dist p p1 = dist p p3 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_in_xz_plane_l164_16456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_symmetry_l164_16462

/-- The inverse proportion function f(x) = -2/x -/
noncomputable def f (x : ℝ) : ℝ := -2 / x

/-- A point (x, y) lies on the graph of f if y = f(x) -/
def lies_on_graph (x y : ℝ) : Prop := y = f x

theorem inverse_proportion_symmetry (m n : ℝ) (h : m ≠ 0) (h' : n ≠ 0) :
  lies_on_graph m n → lies_on_graph n m := by
  intro h1
  unfold lies_on_graph at *
  unfold f at *
  rw [h1]
  field_simp
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_symmetry_l164_16462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sibling_age_problem_l164_16437

theorem sibling_age_problem : 
  ∃ (x y z : ℕ), 
    x - y = 3 ∧
    z - 1 = 2 * ((x - 1) + (y - 1)) ∧
    z + 20 = (x + 20) + (y + 20) ∧
    x = 13 ∧ y = 10 ∧ z = 43 := by
  sorry

#check sibling_age_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sibling_age_problem_l164_16437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l164_16404

theorem equation_solution :
  ∃ x : ℝ, 64 = 4 * (16 : ℝ) ^ (x - 2) ↔ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l164_16404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_theorem_l164_16406

theorem cosine_sum_theorem (A B C : ℝ) 
  (h1 : Real.sin A + Real.sin B + Real.sin C = 0) 
  (h2 : Real.cos A + Real.cos B + Real.cos C = 0) : 
  Real.cos (A - B) + Real.cos (B - C) + Real.cos (C - A) = -3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_theorem_l164_16406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonpositive_implies_a_range_l164_16480

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x^2 + a

-- State the theorem
theorem f_nonpositive_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f a x ≤ 0) → a ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonpositive_implies_a_range_l164_16480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_satisfies_conditions_principal_approximation_l164_16402

/-- The principal amount that satisfies the given conditions -/
noncomputable def principal : ℝ :=
  12 / ((1.05^2 * 1.03^4) - 1 - 0.22)

/-- Theorem stating that the principal amount satisfies the given conditions -/
theorem principal_satisfies_conditions :
  principal * ((1.05^2 * 1.03^4) - 1 - 0.22) = 12 := by
  sorry

/-- The compound interest rate for the first year -/
def first_year_rate : ℝ := 0.10

/-- The compound interest rate for the second year -/
def first_year_periods : ℕ := 2

/-- The number of compounding periods in the second year -/
def second_year_periods : ℕ := 4

/-- The difference between compound interest and simple interest -/
def interest_difference : ℝ := 12

/-- Theorem stating that the principal amount is approximately 597.01 -/
theorem principal_approximation :
  ∃ ε > 0, |principal - 597.01| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_satisfies_conditions_principal_approximation_l164_16402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l164_16490

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n ↦ a₁ + (n - 1) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (a₁ + arithmetic_sequence a₁ d n) / 2

theorem arithmetic_sequence_common_difference :
  ∃ d : ℝ, arithmetic_sum 1 d 9 = 45 ∧ d = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l164_16490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_minimized_at_6_l164_16470

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  first_term : a 1 = -11
  sum_4_6 : a 4 + a 6 = -6
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a 1 + (n - 1) * (seq.a 2 - seq.a 1))

/-- The theorem stating that the sum is minimized when n = 6 -/
theorem sum_minimized_at_6 (seq : ArithmeticSequence) :
  ∀ n : ℕ, n ≠ 0 → sum_n_terms seq 6 ≤ sum_n_terms seq n :=
by
  sorry

#check sum_minimized_at_6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_minimized_at_6_l164_16470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC₁_l164_16435

-- Define the circle C₁
noncomputable def C₁ (x y : ℝ) : Prop := (x + Real.sqrt 3)^2 + y^2 = 4

-- Define the curve C₂
def C₂ (x y θ : ℝ) : Prop := x = 2 + 2 * Real.cos θ ∧ y = 2 * Real.sin θ

-- Define the line C₃ in polar coordinates
def C₃ (θ : ℝ) : Prop := θ = Real.pi / 3

-- Define the center of circle C₁
noncomputable def C₁_center : ℝ × ℝ := (-Real.sqrt 3, 0)

-- Define the intersection points A and B
def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (1, Real.sqrt 3)

-- State the theorem
theorem area_of_triangle_ABC₁ :
  let triangle_area := (1/2) * ‖A - C₁_center‖ * ‖B - A‖ * Real.sin (2*Real.pi/3)
  triangle_area = 3/2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC₁_l164_16435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_plane_intersection_not_always_parallel_perpendicular_line_plane_implies_planes_parallel_parallel_lines_perpendicular_to_plane_perpendicular_lines_to_plane_not_always_parallel_l164_16408

-- Define the types for lines and planes
variable (L : Type*) (P : Type*)

-- Define the parallel and intersection relations
variable (parallel_line_plane : L → P → Prop)
variable (parallel_lines : L → L → Prop)
variable (perpendicular_line_plane : L → P → Prop)
variable (intersect_planes : P → P → L → Prop)

-- State the theorem
theorem parallel_line_plane_intersection_not_always_parallel :
  ¬ (∀ (m n : L) (α β : P),
    parallel_line_plane m α →
    intersect_planes α β n →
    parallel_lines m n) :=
sorry

-- Additional theorems for other options
theorem perpendicular_line_plane_implies_planes_parallel
  (m : L) (α β : P) :
  perpendicular_line_plane m α →
  perpendicular_line_plane m β →
  -- Assuming we have a relation for parallel planes
  True :=  -- Replace 'True' with the actual relation for parallel planes
sorry

theorem parallel_lines_perpendicular_to_plane
  (m n : L) (α : P) :
  parallel_lines m n →
  perpendicular_line_plane m α →
  perpendicular_line_plane n α :=
sorry

theorem perpendicular_lines_to_plane_not_always_parallel
  (m n : L) (α : P) :
  perpendicular_line_plane m α →
  perpendicular_line_plane n α →
  ¬ (parallel_lines m n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_plane_intersection_not_always_parallel_perpendicular_line_plane_implies_planes_parallel_parallel_lines_perpendicular_to_plane_perpendicular_lines_to_plane_not_always_parallel_l164_16408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_circumcenter_distance_l164_16426

/-- Given a triangle with orthocenter H, circumcenter O, angles A, B, and C, and circumradius R,
    the squared distance between H and O is equal to R^2(1 - 8 cos A · cos B · cos C) -/
theorem orthocenter_circumcenter_distance 
  (H O : ℝ × ℝ) -- orthocenter and circumcenter as points in 2D plane
  (A B C : ℝ) -- angles of the triangle
  (R : ℝ) -- circumradius
  (h_triangle : A + B + C = π) -- sum of angles in a triangle is π
  (h_R_pos : R > 0) -- circumradius is positive
  : ‖H - O‖^2 = R^2 * (1 - 8 * Real.cos A * Real.cos B * Real.cos C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_circumcenter_distance_l164_16426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_negation_l164_16486

theorem unique_solution_negation {α : Type*} (P : α → Prop) :
  ¬(∃! x, P x) ↔ (∃ x y, P x ∧ P y ∧ x ≠ y) ∨ (∀ x, ¬P x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_negation_l164_16486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_contains_two_points_from_L_l164_16499

/-- Definition of the set L -/
def L : Set (ℤ × ℤ) := {p | ∃ x y : ℤ, p = (41*x + 2*y, 59*x + 15*y)}

/-- Definition of a parallelogram centered at the origin -/
def is_centered_parallelogram (P : Set (ℤ × ℤ)) : Prop :=
  ∃ v w : ℤ × ℤ, P = {p | ∃ a b : ℚ, -1 ≤ a ∧ a ≤ 1 ∧ -1 ≤ b ∧ b ≤ 1 ∧ p = (⌊a * v.1 + b * w.1⌋, ⌊a * v.2 + b * w.2⌋)}

/-- Area of a parallelogram given its two vectors -/
def parallelogram_area (v w : ℤ × ℤ) : ℤ :=
  abs (v.1 * w.2 - v.2 * w.1)

/-- Main theorem -/
theorem parallelogram_contains_two_points_from_L :
  ∀ P : Set (ℤ × ℤ),
  is_centered_parallelogram P →
  (∃ v w : ℤ × ℤ, P = {p | ∃ a b : ℚ, -1 ≤ a ∧ a ≤ 1 ∧ -1 ≤ b ∧ b ≤ 1 ∧ p = (⌊a * v.1 + b * w.1⌋, ⌊a * v.2 + b * w.2⌋)} ∧ parallelogram_area v w = 2008) →
  ∃ p₁ p₂ : ℤ × ℤ, p₁ ∈ L ∧ p₂ ∈ L ∧ p₁ ∈ P ∧ p₂ ∈ P ∧ p₁ ≠ p₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_contains_two_points_from_L_l164_16499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_areas_l164_16481

/-- A figure within the rectangle --/
structure Figure where
  area : ℝ

/-- The rectangle containing the figures --/
structure Rectangle where
  area : ℝ
  figures : Finset Figure
  figure_count : Nat
  figure_area : ℝ

/-- The problem statement --/
theorem intersection_areas 
  (rect : Rectangle) 
  (h_rect_area : rect.area = 1) 
  (h_figure_count : rect.figure_count = 5) 
  (h_figure_area : ∀ f ∈ rect.figures, f.area = 1/2) : 
  (∃ i j : Figure, i ∈ rect.figures ∧ j ∈ rect.figures ∧ i ≠ j ∧ (i.area ⊓ j.area : ℝ) ≥ 3/20) ∧ 
  (∃ i j : Figure, i ∈ rect.figures ∧ j ∈ rect.figures ∧ i ≠ j ∧ (i.area ⊓ j.area : ℝ) ≥ 1/5) ∧ 
  (∃ i j k : Figure, i ∈ rect.figures ∧ j ∈ rect.figures ∧ k ∈ rect.figures ∧ 
    i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ (i.area ⊓ j.area ⊓ k.area : ℝ) ≥ 1/20) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_areas_l164_16481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_volume_is_216_l164_16401

/-- A square-based right prism with base edges of 6 cm and side edges of 12 cm -/
structure Prism where
  base_edge : ℝ
  side_edge : ℝ
  base_edge_eq : base_edge = 6
  side_edge_eq : side_edge = 12

/-- Volume of a pyramid with triangular base and given height -/
noncomputable def pyramid_volume (base_area : ℝ) (height : ℝ) : ℝ :=
  (1/3) * base_area * height

/-- Volume of the remaining solid after cutting four pyramids from the prism -/
noncomputable def remaining_volume (p : Prism) : ℝ :=
  let prism_volume := p.base_edge^2 * p.side_edge
  let pyramid_base_area := (1/2) * p.base_edge^2
  let large_pyramid_volume := pyramid_volume pyramid_base_area p.side_edge
  let small_pyramid_volume := pyramid_volume ((1/4) * pyramid_base_area) (p.side_edge / 2)
  prism_volume - 4 * large_pyramid_volume + 4 * small_pyramid_volume

/-- The theorem stating that the remaining volume is 216 cm³ -/
theorem remaining_volume_is_216 (p : Prism) : remaining_volume p = 216 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_volume_is_216_l164_16401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fox_hunt_theorem_l164_16431

/-- The probability of hitting a target at a given distance -/
noncomputable def hit_probability (initial_distance : ℝ) (distance : ℝ) : ℝ :=
  (1 / 2) * (initial_distance / distance)^2

/-- The probability of hitting the fox in at least one of three shots -/
noncomputable def fox_hunt_probability (initial_distance : ℝ) (move_distance : ℝ) : ℝ :=
  let p1 := hit_probability initial_distance initial_distance
  let p2 := hit_probability initial_distance (initial_distance + move_distance)
  let p3 := hit_probability initial_distance (initial_distance + 2 * move_distance)
  p1 + p2 + p3 - (p1 * p2) - (p1 * p3) - (p2 * p3) + (p1 * p2 * p3)

theorem fox_hunt_theorem :
  fox_hunt_probability 100 50 = 95 / 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fox_hunt_theorem_l164_16431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sine_line_l164_16441

/-- 
Given a line y = a intersecting the curve y = sin x (0 ≤ x ≤ π) at points A and B,
if the distance between A and B is π/5, then a = sin(2π/5).
-/
theorem intersection_sine_line (a : ℝ) : 
  ∃ (x₁ x₂ : ℝ), 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ π ∧
  Real.sin x₁ = a ∧ Real.sin x₂ = a ∧
  x₂ - x₁ = π / 5 →
  a = Real.sin (2 * π / 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sine_line_l164_16441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_range_l164_16461

theorem count_integers_in_range : 
  ∃ (S : Finset ℤ), (∀ x : ℤ, x ∈ S ↔ -6 ≤ 2*x + 2 ∧ 2*x + 2 ≤ 4) ∧ Finset.card S = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_range_l164_16461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_same_number_is_one_tenth_l164_16453

-- Define the set of integers from 1 to 10
def S : Finset ℕ := Finset.range 10

-- Define the probability of picking the same number
def prob_same_number : ℚ :=
  (S.card : ℚ) / ((S.card * S.card) : ℚ)

-- Theorem statement
theorem prob_same_number_is_one_tenth :
  prob_same_number = 1 / 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_same_number_is_one_tenth_l164_16453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bonus_redistribution_l164_16407

def bonuses : List ℕ := [30, 40, 50, 60, 70]

theorem bonus_redistribution :
  let total := bonuses.sum
  let equal_share := total / bonuses.length
  let highest_bonus := bonuses.maximum?.getD 0
  highest_bonus = 70 ∧
  highest_bonus - equal_share = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bonus_redistribution_l164_16407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_proof_l164_16419

/-- Given a > 0 and f : (0, +∞) → ℝ satisfying certain conditions, prove f is constant -/
theorem constant_function_proof (a : ℝ) (ha : a > 0) 
  (f : ℝ → ℝ) (hf : ∀ x > 0, f x ∈ Set.Ioi 0) :
  (f a = 1) →
  (∀ x y, x > 0 → y > 0 → f x * f y + f (a / x) * f (a / y) = 2 * f (x * y)) →
  ∀ x > 0, f x = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_proof_l164_16419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_walking_speed_l164_16476

noncomputable def bridge_length : ℝ := 1500 -- in meters
noncomputable def crossing_time : ℝ := 15 -- in minutes

noncomputable def walking_speed : ℝ :=
  (bridge_length / 1000) / (crossing_time / 60)

theorem mans_walking_speed :
  walking_speed = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_walking_speed_l164_16476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2014_derivative_l164_16443

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.exp x + x^2013

-- Define the recursive derivative function
noncomputable def f_n : ℕ → (ℝ → ℝ)
  | 0 => f
  | n+1 => deriv (f_n n)

-- State the theorem
theorem f_2014_derivative (x : ℝ) : f_n 2014 x = -Real.sin x + Real.exp x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2014_derivative_l164_16443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_l164_16491

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * seq.a 1 + n * (n - 1) / 2 * seq.d

/-- Theorem: If S_30 > 0 and S_31 < 0, then S_n is maximum when n = 15 -/
theorem arithmetic_sequence_max_sum
  (seq : ArithmeticSequence)
  (h30 : S seq 30 > 0)
  (h31 : S seq 31 < 0) :
  ∀ n : ℕ, S seq n ≤ S seq 15 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_l164_16491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_range_of_x_l164_16425

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for the solution set of f(x) ≤ 3
theorem solution_set_f (x : ℝ) : f x ≤ 3 ↔ x ∈ Set.Icc 0 3 := by sorry

-- Theorem for the range of x given the inequality
theorem range_of_x (a b x : ℝ) (ha : a ≠ 0) :
  (∀ a b, abs (abs (a + b) - abs (a - b)) ≤ abs a * f x) →
  x ≤ 1/2 ∨ x ≥ 5/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_range_of_x_l164_16425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_folding_properties_l164_16477

/-- Represents the dimensions of a rectangular paper. -/
structure PaperDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the number of different shapes after n folds. -/
def num_shapes (n : ℕ) : ℕ := n + 1

/-- Calculates the sum of areas after n folds. -/
noncomputable def sum_areas (n : ℕ) : ℝ := 240 * (3 - (n + 3) / 2^n)

/-- Theorem about paper folding properties. -/
theorem paper_folding_properties (paper : PaperDimensions) 
  (h1 : paper.length = 20)
  (h2 : paper.width = 12) :
  num_shapes 4 = 5 ∧ 
  ∀ n : ℕ, sum_areas n = 240 * (3 - (n + 3) / 2^n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_folding_properties_l164_16477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dominating_set_size_l164_16464

theorem min_dominating_set_size (G : SimpleGraph (Fin 2017)) (hconn : G.Connected) :
  ∃ (S : Finset (Fin 2017)), (S.card = 1344 ∧ 
    (∀ v : Fin 2017, v ∈ S ∨ ∃ u ∈ S, G.Adj v u) ∧
    ∀ T : Finset (Fin 2017), (∀ v : Fin 2017, v ∈ T ∨ ∃ u ∈ T, G.Adj v u) → T.card ≥ 1344) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dominating_set_size_l164_16464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyarelal_loss_calculation_l164_16483

/-- Calculates Pyarelal's loss given the total loss and the ratio of investments -/
noncomputable def pyarelal_loss (total_loss : ℝ) (ratio : ℝ) : ℝ :=
  let pyarelal_investment := total_loss / (0.12 * ratio + 0.09)
  0.09 * pyarelal_investment

theorem pyarelal_loss_calculation (total_loss : ℝ) (h1 : total_loss = 2100) :
  ∃ (ε : ℝ), abs (pyarelal_loss total_loss (1/9) - 1829.32) < ε ∧ ε > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyarelal_loss_calculation_l164_16483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_alpha_beta_l164_16459

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x + a * Real.cos x

theorem sin_sum_alpha_beta (a : ℝ) (α β : ℝ) :
  (∀ x, f a (x + π/4) = f a (-x - π/4)) →
  α ∈ Set.Ioo 0 (π/2) →
  β ∈ Set.Ioo 0 (π/2) →
  f (-1) (α + π/4) = Real.sqrt 10 / 5 →
  f (-1) (β + 3*π/4) = 3 * Real.sqrt 5 / 5 →
  Real.sin (α + β) = Real.sqrt 2 / 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_alpha_beta_l164_16459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chinese_chess_probabilities_l164_16498

/-- Represents the types of players in the Chinese chess competition -/
inductive PlayerType
| Type1
| Type2
| Type3

/-- The probability distribution of player types -/
noncomputable def player_type_prob : PlayerType → ℝ
| PlayerType.Type1 => 0.5
| PlayerType.Type2 => 0.25
| PlayerType.Type3 => 0.25

/-- The probability of Xiao Ming winning against each player type -/
noncomputable def win_prob : PlayerType → ℝ
| PlayerType.Type1 => 0.3
| PlayerType.Type2 => 0.4
| PlayerType.Type3 => 0.5

/-- The overall probability of Xiao Ming winning -/
noncomputable def overall_win_prob : ℝ := 
  (player_type_prob PlayerType.Type1 * win_prob PlayerType.Type1) +
  (player_type_prob PlayerType.Type2 * win_prob PlayerType.Type2) +
  (player_type_prob PlayerType.Type3 * win_prob PlayerType.Type3)

/-- The probability that the opponent is Type 1 given Xiao Ming wins -/
noncomputable def prob_type1_given_win : ℝ :=
  (player_type_prob PlayerType.Type1 * win_prob PlayerType.Type1) / overall_win_prob

theorem chinese_chess_probabilities :
  overall_win_prob = 0.375 ∧ prob_type1_given_win = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chinese_chess_probabilities_l164_16498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_at_one_l164_16413

/-- The function f(x) that has an extreme value at x = 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x

/-- The derivative of f(x) -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := a / x + 1

theorem extreme_value_at_one (a : ℝ) :
  (∀ x > 0, f_derivative a x = 0 → x = 1) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_at_one_l164_16413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_not_uniform_l164_16439

-- Define the volumes of geometric shapes
noncomputable def cube_volume (base_area : ℝ) (height : ℝ) : ℝ := base_area * height
noncomputable def rectangular_prism_volume (base_area : ℝ) (height : ℝ) : ℝ := base_area * height
noncomputable def cone_volume (base_area : ℝ) (height : ℝ) : ℝ := (1/3) * base_area * height

-- Theorem statement
theorem volume_formula_not_uniform :
  ¬(∀ (base_area height : ℝ),
    cube_volume base_area height = rectangular_prism_volume base_area height ∧
    cube_volume base_area height = cone_volume base_area height) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_not_uniform_l164_16439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_catch_up_l164_16463

-- Define the speeds of the cars in miles per hour
noncomputable def first_car_speed : ℝ := 30
noncomputable def second_car_speed : ℝ := 60

-- Define the total trip distance in miles
noncomputable def total_distance : ℝ := 80

-- Define the time difference between the cars' departures in hours
noncomputable def time_difference : ℝ := 1 / 6

-- Define the catch-up time in hours after the second car's departure
noncomputable def catch_up_time : ℝ := 1 / 6

-- Theorem statement
theorem cars_catch_up :
  let distance_traveled_by_first_car := first_car_speed * time_difference
  let relative_speed := second_car_speed - first_car_speed
  distance_traveled_by_first_car / relative_speed = catch_up_time := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_catch_up_l164_16463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l164_16446

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 7 = 0

-- Define the line
def line_eq (x y : ℝ) : Prop := x - y - 5 = 0

-- State the theorem
theorem max_distance_circle_to_line :
  ∃ (d : ℝ), d = 5*Real.sqrt 2/2 + 1 ∧
  (∀ (x y : ℝ), circle_eq x y →
    ∀ (x' y' : ℝ), line_eq x' y' →
      Real.sqrt ((x - x')^2 + (y - y')^2) ≤ d) ∧
  (∃ (x y : ℝ), circle_eq x y ∧
    ∃ (x' y' : ℝ), line_eq x' y' ∧
      Real.sqrt ((x - x')^2 + (y - y')^2) = d) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l164_16446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l164_16466

noncomputable def f (x : ℝ) : ℝ := (x^5 - 5*x^4 + 10*x^3 - 10*x^2 + 5*x - 1) / (x^2 - 9)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ -3 ∧ x ≠ 3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l164_16466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_isosceles_l164_16492

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a*cos(B) + b*cos(A) = a, then the triangle is isosceles with a = c -/
theorem triangle_isosceles (a b c : ℝ) (A B C : ℝ) 
    (h1 : a > 0 ∧ b > 0 ∧ c > 0) -- Positive side lengths
    (h2 : A > 0 ∧ B > 0 ∧ C > 0) -- Positive angles
    (h3 : A + B + C = Real.pi) -- Angle sum in a triangle
    (h4 : a * Real.cos B + b * Real.cos A = a) -- Given condition
    (h5 : Real.cos A = (b^2 + c^2 - a^2) / (2*b*c)) -- Cosine rule
    (h6 : Real.cos B = (a^2 + c^2 - b^2) / (2*a*c)) -- Cosine rule
    : a = c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_isosceles_l164_16492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_n_minus_one_squared_l164_16444

theorem divisibility_by_n_minus_one_squared (n : ℕ) (h : n > 1) :
  ∃ k : ℤ, (n : ℤ)^n - (n : ℤ)^2 + n - 1 = k * (n - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_n_minus_one_squared_l164_16444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_triangles_l164_16469

/-- A right triangle with positive integer leg lengths -/
structure RightTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  is_right : c ^ 2 = a ^ 2 + b ^ 2

/-- The area of a right triangle -/
def area (t : RightTriangle) : ℚ :=
  (t.a.val * t.b.val : ℚ) / 2

/-- The perimeter of a right triangle -/
def perimeter (t : RightTriangle) : ℕ :=
  t.a.val + t.b.val + t.c.val

/-- A right triangle satisfying the area-perimeter condition -/
def satisfies_condition (t : RightTriangle) : Prop :=
  area t = 5 * (perimeter t : ℚ)

/-- Two right triangles are congruent if they have the same leg lengths (up to order) -/
def congruent (t1 t2 : RightTriangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b) ∨ (t1.a = t2.b ∧ t1.b = t2.a)

/-- The main theorem -/
theorem count_satisfying_triangles :
  ∃ (S : Finset RightTriangle),
    (∀ t, t ∈ S → satisfies_condition t) ∧ 
    (∀ t, satisfies_condition t → ∃ t', t' ∈ S ∧ congruent t t') ∧
    (∀ t1 t2, t1 ∈ S → t2 ∈ S → t1 ≠ t2 → ¬congruent t1 t2) ∧
    S.card = 7 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_triangles_l164_16469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l164_16434

-- Define the function (marked as noncomputable due to use of Real.sqrt and abs)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 2) / (abs x - 3)

-- Define the domain
def domain (x : ℝ) : Prop := x ≥ 2 ∧ x ≠ 3

-- Theorem statement
theorem f_domain : 
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ domain x := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l164_16434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l164_16416

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1

-- Define the line with slope 1 from left vertex
def line_from_vertex (a : ℝ) (x y : ℝ) : Prop :=
  y = x + a

-- Define the asymptotes
def asymptote1 (a b : ℝ) (x y : ℝ) : Prop :=
  b * x - a * y = 0

def asymptote2 (a b : ℝ) (x y : ℝ) : Prop :=
  b * x + a * y = 0

-- Define the condition AB = 1/2 BC
def vector_condition (a b : ℝ) : Prop :=
  ∃ x1 y1 x2 y2 : ℝ,
    line_from_vertex a x1 y1 ∧
    asymptote1 a b x1 y1 ∧
    asymptote2 a b x2 y2 ∧
    x1 - a = (x2 - x1) / 2 ∧
    y1 = (y2 - y1) / 2

-- Theorem statement
theorem hyperbola_eccentricity (a b : ℝ) :
  hyperbola a b a 0 →
  vector_condition a b →
  ∃ c : ℝ, c^2 = a^2 + b^2 ∧ c / a = Real.sqrt 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l164_16416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expressions_l164_16452

theorem calculate_expressions :
  (6 * Real.sqrt (1 / 9) - (27 : ℝ) ^ (1/3) + (Real.sqrt 2) ^ 2 = 1) ∧
  (-1 ^ 2022 + Real.sqrt ((-2) ^ 2) + abs (2 - Real.sqrt 3) = 3 - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expressions_l164_16452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_range_l164_16448

/-- The function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + 1 + a * Real.log x

/-- The derivative of f(x) -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 2*x - 2 + a/x

theorem extreme_points_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
   f_deriv a x₁ = 0 ∧ f_deriv a x₂ = 0 ∧
   ∀ x, x > 0 → (f_deriv a x = 0 → x = x₁ ∨ x = x₂)) →
  ∃ x₁ : ℝ, f a x₁ > 1/4 - 1/2 * Real.log 2 ∧
    ∀ x, (∃ y, f_deriv a y = 0 ∧ y < x) → f a x ≥ f a x₁ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_range_l164_16448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_solution_l164_16494

open Real

theorem floor_equation_solution (x : ℝ) : 
  (⌊⌊3 * x⌋ - 1/3⌋ = ⌊x + 3⌋) ↔ 
  ((2 ≤ x ∧ x < 3) ∨ (5/3 ≤ x ∧ x < 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_solution_l164_16494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l164_16458

/-- Simple interest calculation function -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  (principal * rate * time) / 100

/-- Compound interest calculation function -/
noncomputable def compound_interest (principal rate time : ℝ) : ℝ :=
  principal * (1 + rate / 100) ^ time - principal

theorem investment_problem (x y : ℝ) 
  (h1 : simple_interest x y 2 = 800)
  (h2 : compound_interest x y 2 = 820) : 
  x = 8000 := by
  sorry

#check investment_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l164_16458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonal_length_l164_16471

/-- Represents a rhombus with diagonals d1 and d2 -/
structure Rhombus where
  d1 : ℝ
  d2 : ℝ

/-- Calculates the area of a rhombus given its diagonals -/
noncomputable def Rhombus.area (r : Rhombus) : ℝ := (r.d1 * r.d2) / 2

theorem rhombus_diagonal_length (r : Rhombus) (h1 : r.d2 = 15) (h2 : r.area = 90) : r.d1 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonal_length_l164_16471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equilateral_triangle_area_in_rectangle_l164_16449

/-- The maximum area of an equilateral triangle inscribed in a 12x13 rectangle --/
theorem max_equilateral_triangle_area_in_rectangle : 
  ∃ (a b c : ℕ), 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧ 
    (∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ b)) ∧
    (let rectangle_width : ℝ := 12
     let rectangle_height : ℝ := 13
     let max_area : ℝ := (a : ℝ) * Real.sqrt (b : ℝ) - (c : ℝ)
     ∀ s : ℝ, 
       s > 0 → 
       s ≤ min rectangle_width rectangle_height → 
       (Real.sqrt 3 / 4 : ℝ) * s^2 ≤ max_area) ∧
    a + b + c = 433 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equilateral_triangle_area_in_rectangle_l164_16449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rose_percentage_is_71_25_l164_16467

/-- Represents a flower bouquet with white and purple tulips and roses -/
structure Bouquet where
  white_tulips : ℚ
  purple_tulips : ℚ
  white_roses : ℚ
  purple_roses : ℚ

/-- The percentage of roses in a bouquet with given conditions -/
def rose_percentage (b : Bouquet) : ℚ :=
  ((b.white_roses + b.purple_roses) / (b.white_tulips + b.purple_tulips + b.white_roses + b.purple_roses)) * 100

/-- Theorem stating that under given conditions, the percentage of roses is 71.25% -/
theorem rose_percentage_is_71_25 (b : Bouquet) 
  (h1 : b.white_tulips = (1/4) * (b.white_tulips + b.white_roses))
  (h2 : b.purple_roses = (3/5) * (b.purple_tulips + b.purple_roses))
  (h3 : b.white_tulips + b.white_roses = (3/4) * (b.white_tulips + b.purple_tulips + b.white_roses + b.purple_roses)) :
  rose_percentage b = 71.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rose_percentage_is_71_25_l164_16467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_plus_cos6_l164_16421

theorem min_sin6_plus_cos6 (x : ℝ) : Real.sin x ^ 6 + Real.cos x ^ 6 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_plus_cos6_l164_16421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_li_bullseyes_l164_16479

/-- Given that:
  1. A rifle shooter starts with 10 bullets and gets 2 additional bullets for each bullseye.
  2. A pistol shooter starts with 14 bullets and gets 4 additional bullets for each bullseye.
  3. Xiao Wang uses a rifle and hits 30 bullseyes.
  4. Xiao Li uses a pistol.
  5. Both shooters end up shooting the same total number of bullets.

  Prove that Xiao Li hits 14 bullseyes. -/
theorem xiao_li_bullseyes 
  (rifle_start : ℕ) 
  (rifle_reward : ℕ) 
  (pistol_start : ℕ) 
  (pistol_reward : ℕ) 
  (xiao_wang_bullseyes : ℕ) 
  (xiao_li_bullseyes : ℕ)
  (h1 : rifle_start = 10)
  (h2 : rifle_reward = 2)
  (h3 : pistol_start = 14)
  (h4 : pistol_reward = 4)
  (h5 : xiao_wang_bullseyes = 30)
  (h6 : rifle_start + xiao_wang_bullseyes * rifle_reward = 
        pistol_start + xiao_li_bullseyes * pistol_reward) :
  xiao_li_bullseyes = 14 :=
by
  sorry

#check xiao_li_bullseyes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_li_bullseyes_l164_16479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l164_16436

/-- The equation of the directrix of a parabola y = ax^2 is y = -1/(4a) -/
theorem parabola_directrix (a : ℝ) (h : a ≠ 0) :
  let parabola := λ x : ℝ ↦ a * x^2
  let directrix := λ x : ℝ ↦ -1 / (4 * a)
  ∀ x : ℝ, parabola x = 2 * x^2 → directrix x = -1/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l164_16436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l164_16409

noncomputable def p : ℝ × ℝ := (-1, 2)
def A : ℝ × ℝ := (8, 0)

def B (n t : ℝ) : ℝ × ℝ := (n, t)
noncomputable def C (k θ t : ℝ) : ℝ × ℝ := (k * Real.sin θ, t)

def vector_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

def vector_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (c : ℝ), v = (c * w.1, c * w.2)

noncomputable def vector_length (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

theorem problem_solution :
  ∀ (n t k θ : ℝ),
  0 ≤ θ ∧ θ ≤ Real.pi / 2 →
  vector_perpendicular (B n t - A) p →
  vector_length (B n t - A) = Real.sqrt 5 * vector_length A →
  vector_parallel (C k θ t - A) p →
  k > 4 →
  (∀ θ', 0 ≤ θ' ∧ θ' ≤ Real.pi / 2 → t * Real.sin θ' ≤ 4) →
  t * Real.sin θ = 4 →
  ((B n t = (24, 8) ∨ B n t = (-8, -8)) ∧
   (Real.tan (Real.arccos ((A.1 * (C k θ t).1 + A.2 * (C k θ t).2) /
     (vector_length A * vector_length (C k θ t)))) = 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l164_16409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jim_reading_theorem_l164_16465

/-- Calculates the number of pages Jim reads per week after increasing his speed and reducing reading time -/
noncomputable def pages_read_per_week (initial_rate : ℝ) (initial_pages : ℝ) (speed_increase : ℝ) (time_reduction : ℝ) : ℝ :=
  let initial_hours := initial_pages / initial_rate
  let new_rate := initial_rate * speed_increase
  let new_hours := initial_hours - time_reduction
  new_rate * new_hours

/-- Theorem stating that Jim reads 660 pages per week after changes in reading speed and time -/
theorem jim_reading_theorem :
  pages_read_per_week 40 600 1.5 4 = 660 := by
  -- Unfold the definition of pages_read_per_week
  unfold pages_read_per_week
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jim_reading_theorem_l164_16465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_l164_16415

-- Define the basic structures
structure Plane : Type
structure Line : Type

-- Define the propositions
def no_common_points (a b : Line) : Prop := sorry

def parallel (α β : Plane) : Prop := sorry

-- Define the subset relation
def subset (a : Line) (α : Plane) : Prop := sorry

-- State the theorem
theorem necessary_but_not_sufficient 
  (α β : Plane) (a b : Line) 
  (h1 : α ≠ β) 
  (h2 : subset a α) 
  (h3 : subset b β) : 
  (parallel α β → no_common_points a b) ∧ 
  ¬(no_common_points a b → parallel α β) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_l164_16415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l164_16457

noncomputable def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

noncomputable def right_focus : ℝ × ℝ := (Real.sqrt 2, 0)

def asymptotes (x y : ℝ) : Prop := x + y = 0 ∨ x - y = 0

def point_P : ℝ × ℝ := (0, -1)

noncomputable def circle_equation (x y : ℝ) : Prop := (x - Real.sqrt 2)^2 + y^2 = 1

def y_intercept_range (t : ℝ) : Prop := t > 2

theorem hyperbola_properties :
  (∀ x y, hyperbola x y → (∃ t, y_intercept_range t ∧
    (∃ k, 1 < k ∧ k < Real.sqrt 2 ∧
      (∃ x1 y1 x2 y2, hyperbola x1 y1 ∧ hyperbola x2 y2 ∧
        y1 = k * x1 - 1 ∧ y2 = k * x2 - 1 ∧
        t = 2 / (k^2 - 1))))) ∧
  (∀ x y, circle_equation x y ↔
    (x - (right_focus.1))^2 + y^2 = (right_focus.1 - 0)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l164_16457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_45_degrees_perpendicular_lines_l164_16484

-- Define the slope of line l₁
def slope_l1 : ℚ := 2

-- Define the coordinates of points A and B on line l₂
def point_A (m : ℚ) : ℚ × ℚ := (3*m, 2*m - 1)
def point_B (m : ℚ) : ℚ × ℚ := (2, m - 3)

-- Define the slope of line l₂
noncomputable def slope_l2 (m : ℚ) : ℚ := 
  ((point_B m).2 - (point_A m).2) / ((point_B m).1 - (point_A m).1)

-- Theorem for Question 1
theorem slope_45_degrees (m : ℚ) : 
  slope_l2 m = 1 → m = 2 := by sorry

-- Theorem for Question 2
theorem perpendicular_lines (m : ℚ) :
  slope_l2 m = -1 / slope_l1 → m = -2/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_45_degrees_perpendicular_lines_l164_16484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_intersects_parabola_l164_16474

noncomputable def parabola (x : ℝ) : ℝ := x^2

noncomputable def normal_slope (x : ℝ) : ℝ := 1 / (2 * x)

noncomputable def normal_line (x : ℝ) : ℝ := normal_slope (-1) * (x + 1) + 1

theorem normal_intersects_parabola :
  let A : ℝ × ℝ := (-1, 1)
  let B : ℝ × ℝ := (1.5, 2.25)
  (parabola A.1 = A.2) →
  (normal_line B.1 = parabola B.1) ∧
  (B.2 = parabola B.1) :=
by sorry

#check normal_intersects_parabola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_intersects_parabola_l164_16474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_of_quadratic_l164_16475

theorem factor_of_quadratic (x t : ℝ) : 
  (∃ k : ℝ, 4 * x^2 + 11 * x - 3 = (x - t) * k) ↔ t = 1/4 ∨ t = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_of_quadratic_l164_16475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_difference_l164_16495

-- Define the points and circles
variable (A B C : ℝ × ℝ)
variable (C₁ C₂ : Set (ℝ × ℝ))

-- Define the conditions
def collinear (A B C : ℝ × ℝ) : Prop := sorry
def distance (P Q : ℝ × ℝ) : ℝ := sorry
def circleSet (center : ℝ × ℝ) (passThrough : ℝ × ℝ) : Set (ℝ × ℝ) := sorry
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

-- State the theorem
theorem circle_area_difference 
  (h_collinear : collinear A B C)
  (h_AB : distance A B = 1)
  (h_BC : distance B C = 1)
  (h_AC : distance A C = 2)
  (h_C₁ : C₁ = circleSet A B)
  (h_C₂ : C₂ = circleSet A C) :
  area (C₂ \ C₁) = 3 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_difference_l164_16495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_100_l164_16485

noncomputable section

/-- Cost function in million yuan -/
noncomputable def C (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then 10 * x^2 + 100 * x
  else if x ≥ 40 then 501 * x + 10000 / x - 4500
  else 0

/-- Profit function in million yuan -/
noncomputable def L (x : ℝ) : ℝ :=
  500 * x - 20 - C x

/-- The statement to be proved -/
theorem max_profit_at_100 :
  ∃ (x : ℝ), x > 0 ∧ L x = 2300 ∧ ∀ (y : ℝ), y > 0 → L y ≤ L x := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_100_l164_16485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_min_chord_sum_l164_16493

noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

noncomputable def left_focus (a : ℝ) (e : ℝ) : ℝ := -a * e
noncomputable def right_focus (a : ℝ) (e : ℝ) : ℝ := a * e

noncomputable def inradius (a b c s : ℝ) : ℝ :=
  s / ((a + b + c) / 2)

noncomputable def triangle_area (base height : ℝ) : ℝ :=
  (1 / 2) * base * height

theorem ellipse_equation_and_min_chord_sum 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : eccentricity a b = 1/2) 
  (h4 : ∃ (x y : ℝ), ellipse a b x y ∧ 
    inradius (2*a) (2*a) (2*a) 1 = 1 ∧
    ∃ (x1 y1 x2 y2 : ℝ), 
      triangle_area (Real.sqrt ((x-left_focus a (1/2))^2 + y^2)) 1 +
      triangle_area (Real.sqrt ((x-right_focus a (1/2))^2 + y^2)) 1 = 2) :
  (a = 2 ∧ b = Real.sqrt 3) ∧
  (∀ (m : ℝ), 
    let chord_sum := 
      Real.sqrt (1 + m^2) * Real.sqrt (36*m^2 + 36*(3*m^2 + 4)) / (3*m^2 + 4) +
      Real.sqrt (1 + 1/m^2) * Real.sqrt (36/m^2 + 36*(3/m^2 + 4)) / (3/m^2 + 4)
    chord_sum ≥ 48/7) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_min_chord_sum_l164_16493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_second_quadrant_l164_16405

theorem sin_double_angle_second_quadrant (α : Real) 
  (h1 : Real.sin α = 1/5) 
  (h2 : α ∈ Set.Ioo (π/2) π) : 
  Real.sin (2*α) = -4*Real.sqrt 6/25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_second_quadrant_l164_16405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l164_16424

theorem power_equation (a b : ℝ) (h1 : (27 : ℝ)^a = 5) (h2 : (9 : ℝ)^b = 10) : 
  (3 : ℝ)^(3*a + 2*b) = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l164_16424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_queue_length_calculation_l164_16468

/-- The length of a moving queue given the queue's speed, a runner's speed, and the time taken for a round trip --/
noncomputable def queue_length (queue_speed : ℝ) (runner_speed : ℝ) (round_trip_time : ℝ) : ℝ :=
  (runner_speed * queue_speed * round_trip_time) / (2 * (runner_speed - queue_speed))

theorem queue_length_calculation (queue_speed runner_speed round_trip_time : ℝ) 
  (h1 : queue_speed = 8)
  (h2 : runner_speed = 12)
  (h3 : round_trip_time = 7.2 / 60) :
  queue_length queue_speed runner_speed round_trip_time * 1000 = 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_queue_length_calculation_l164_16468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meet_time_is_half_harmonic_mean_xiao_dong_and_grandfather_meet_time_l164_16418

noncomputable def time_to_meet (t1 t2 : ℝ) : ℝ := 1 / ((1 / t1) + (1 / t2))

theorem meet_time_is_half_harmonic_mean (t_xiao_dong t_grandfather : ℝ) 
  (h1 : t_xiao_dong > 0) (h2 : t_grandfather > 0) :
  time_to_meet t_xiao_dong t_grandfather = (t_xiao_dong * t_grandfather) / (t_xiao_dong + t_grandfather) / 2 := by
  sorry

theorem xiao_dong_and_grandfather_meet_time :
  time_to_meet 10 16 = 80 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meet_time_is_half_harmonic_mean_xiao_dong_and_grandfather_meet_time_l164_16418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_subtraction_l164_16451

theorem digit_sum_subtraction (N : ℕ) : 
  2010 ≤ N ∧ N ≤ 2019 → N - (Nat.sum (Nat.digits 10 N)) = 2007 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_subtraction_l164_16451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pebble_collection_15_days_l164_16473

def pebble_collection (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n % 2 = 1 then (n + 1) / 2
  else 2 * pebble_collection (n - 1)

def total_pebbles (n : ℕ) : ℕ :=
  (List.range n).map pebble_collection |>.sum

theorem pebble_collection_15_days :
  total_pebbles 15 = 152 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pebble_collection_15_days_l164_16473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l164_16430

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The slope of a line, if it exists -/
noncomputable def Line.slope (l : Line) : Option ℝ :=
  if l.b ≠ 0 then some (-l.a / l.b) else none

/-- Check if a point (x, y) is on the line -/
def Line.contains_point (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  match l1.slope, l2.slope with
  | some m1, some m2 => m1 * m2 = -1
  | _, _ => False

theorem perpendicular_line_equation (l : Line) :
  l.contains_point 0 3 ∧
  perpendicular l (Line.mk 1 1 1) →
  l = Line.mk 1 (-1) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l164_16430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l164_16488

/-- Represents a geometric sequence with 2n terms -/
structure GeometricSequence where
  n : ℕ
  r : ℝ
  first_term : ℝ

/-- Sum of odd-numbered terms in the sequence -/
noncomputable def sum_odd (seq : GeometricSequence) : ℝ :=
  (1 - seq.r^(2 * seq.n)) / (1 - seq.r^2)

/-- Sum of even-numbered terms in the sequence -/
noncomputable def sum_even (seq : GeometricSequence) : ℝ :=
  (seq.r * (1 - seq.r^(2 * seq.n))) / (1 - seq.r^2)

/-- Sum of the two middle terms in the sequence -/
noncomputable def sum_middle_terms (seq : GeometricSequence) : ℝ :=
  seq.r^(seq.n - 1) + seq.r^seq.n

/-- Theorem stating the properties of the geometric sequence and its number of terms -/
theorem geometric_sequence_properties (seq : GeometricSequence) :
  seq.first_term = 1 ∧
  sum_even seq = 2 * sum_odd seq ∧
  sum_middle_terms seq = 24 →
  seq.n = 4 := by
  sorry

#check geometric_sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l164_16488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_gain_percentage_l164_16427

/-- Calculates the gain percentage when a watch is sold with increased price -/
theorem watch_gain_percentage 
  (cost_price : ℝ) 
  (initial_loss_percentage : ℝ) 
  (price_increase : ℝ) 
  (h1 : cost_price = 2000) 
  (h2 : initial_loss_percentage = 10) 
  (h3 : price_increase = 280) : 
  (cost_price * (1 - initial_loss_percentage / 100) + price_increase - cost_price) / cost_price * 100 = 4 := by
  -- Calculate initial selling price
  have initial_selling_price := cost_price * (1 - initial_loss_percentage / 100)
  -- Calculate new selling price
  have new_selling_price := initial_selling_price + price_increase
  -- Calculate gain
  have gain := new_selling_price - cost_price
  -- Calculate gain percentage
  have gain_percentage := (gain / cost_price) * 100
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_gain_percentage_l164_16427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_solutions_exist_x_not_uniquely_determined_l164_16417

/-- The distance from a point (x, y) to the line ax + by + c = 0 -/
noncomputable def distanceToLine (x y a b c : ℝ) : ℝ :=
  |a * x + b * y + c| / Real.sqrt (a^2 + b^2)

/-- The theorem stating that there are multiple solutions for x -/
theorem multiple_solutions_exist : ∃ x₁ x₂ y₁ y₂ : ℝ, 
  x₁ ≠ x₂ ∧ 
  (distanceToLine x₁ y₁ 1 1 (-2) = |y₁|) ∧
  (distanceToLine x₁ y₁ 1 1 (-2) = |x₁|) ∧
  (distanceToLine x₂ y₂ 1 1 (-2) = |y₂|) ∧
  (distanceToLine x₂ y₂ 1 1 (-2) = |x₂|) := by
  sorry

/-- The main theorem proving that x cannot be uniquely determined -/
theorem x_not_uniquely_determined : 
  ¬ ∃! x : ℝ, ∃ y : ℝ, (distanceToLine x y 1 1 (-2) = |y|) ∧ 
                        (distanceToLine x y 1 1 (-2) = |x|) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_solutions_exist_x_not_uniquely_determined_l164_16417
