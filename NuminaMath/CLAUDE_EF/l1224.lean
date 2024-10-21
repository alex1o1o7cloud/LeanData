import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_total_profit_l1224_122435

/-- Calculates the total profit given investments and A's profit share -/
theorem calculate_total_profit
  (a_investment : ℚ)
  (a_months : ℚ)
  (b_investment : ℚ)
  (b_months : ℚ)
  (a_profit_share : ℚ)
  (h1 : a_investment = 400)
  (h2 : a_months = 12)
  (h3 : b_investment = 200)
  (h4 : b_months = 6)
  (h5 : a_profit_share = 80) :
  (a_investment * a_months + b_investment * b_months) * a_profit_share /
  (a_investment * a_months) = 120 := by
  sorry

#check calculate_total_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_total_profit_l1224_122435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_filter_kit_price_difference_l1224_122462

/-- Represents the price of a camera lens filter kit and its individual components. -/
structure FilterKit where
  kit_price : ℚ
  filter_price_1 : ℚ
  filter_price_2 : ℚ
  filter_price_3 : ℚ
  num_filter_1 : ℕ
  num_filter_2 : ℕ
  num_filter_3 : ℕ

/-- Calculates the total price of filters if purchased individually. -/
def total_individual_price (fk : FilterKit) : ℚ :=
  fk.filter_price_1 * fk.num_filter_1 + fk.filter_price_2 * fk.num_filter_2 + fk.filter_price_3 * fk.num_filter_3

/-- Calculates the percentage difference between kit price and total individual price. -/
def price_difference_percentage (fk : FilterKit) : ℚ :=
  (fk.kit_price - total_individual_price fk) / total_individual_price fk * 100

/-- Theorem stating that the price difference percentage for the given filter kit is -8.7%. -/
theorem filter_kit_price_difference :
  let fk : FilterKit := {
    kit_price := 87.5,
    filter_price_1 := 16.45,
    filter_price_2 := 14.05,
    filter_price_3 := 19.5,
    num_filter_1 := 2,
    num_filter_2 := 2,
    num_filter_3 := 1
  }
  abs (price_difference_percentage fk - (-87/10)) < 1/10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_filter_kit_price_difference_l1224_122462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mail_distribution_l1224_122413

def number_of_distributions (n m : ℕ) : ℕ := m^n

theorem mail_distribution (n m : ℕ) : 
  n > 0 → m > 0 → number_of_distributions n m = m^n :=
by
  intros hn hm
  unfold number_of_distributions
  rfl

#eval number_of_distributions 3 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mail_distribution_l1224_122413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_condition_l1224_122470

/-- The function g --/
noncomputable def g (c : ℝ) (x : ℝ) : ℝ := 1 / (3 * x + c)

/-- The inverse function of g --/
noncomputable def g_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

/-- Theorem stating that c must equal 3/2 for the given conditions to hold --/
theorem g_inverse_condition (c : ℝ) : 
  (∀ x, g c x ≠ 0 → g_inv (g c x) = x) → c = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_condition_l1224_122470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_eraser_cost_l1224_122471

theorem pencil_eraser_cost (x y : ℤ) : 
  14 * x + 3 * y = 107 →
  abs (x - y) ≤ 5 →
  x > 0 →
  y > 0 →
  x + y = 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_eraser_cost_l1224_122471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_theorem_l1224_122415

-- Define the sets A and B
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | x < 1}

-- Define the complement of B in the universal set ℝ
def C_U_B : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem intersection_theorem : A ∩ C_U_B = Set.Icc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_theorem_l1224_122415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_term_is_three_l1224_122491

def mySequence (n : ℕ) : ℚ :=
  if n % 2 = 1 then 3 else 4

theorem fifteenth_term_is_three :
  let s := mySequence
  (∀ n : ℕ, n ≥ 1 → s (n + 1) * s n = s 2 * s 1) →  -- constant proportionality
  s 1 = 3 →                                         -- first term is 3
  s 2 = 4 →                                         -- second term is 4
  s 15 = 3                                          -- 15th term is 3
:= by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_term_is_three_l1224_122491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_sum_l1224_122428

/-- Given the conditions, prove that k + m + n = 27 -/
theorem trigonometric_identity_sum (t : ℝ) (k m n : ℕ+) :
  (1 + Real.sin t) * (1 + Real.cos t) = 5/4 →
  (1 - Real.sin t) * (1 - Real.cos t) = m / n - Real.sqrt k →
  Nat.Coprime m.val n.val →
  k + m + n = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_sum_l1224_122428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_line_intersection_common_point_l1224_122493

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a triangle
structure Triangle where
  p1 : Point2D
  p2 : Point2D
  p3 : Point2D

-- Define a circle
structure Circle where
  center : Point2D
  radius : ℝ

-- Define membership for Point2D in Line2D
def Point2D.mem (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define membership for Point2D in Circle
def Point2D.memCircle (p : Point2D) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

instance : Membership Point2D Line2D where
  mem := Point2D.mem

instance : Membership Point2D Circle where
  mem := Point2D.memCircle

-- Define the theorem
theorem four_line_intersection_common_point 
  (l1 l2 l3 l4 : Line2D) 
  (A B C D E F : Point2D) 
  (t1 t2 t3 t4 : Triangle) 
  (c1 c2 c3 c4 : Circle) :
  (-- Conditions
   -- Lines intersect to form points
   (A ∈ l1 ∧ A ∈ l2) ∧
   (B ∈ l1 ∧ B ∈ l3) ∧
   (C ∈ l1 ∧ C ∈ l4) ∧
   (D ∈ l2 ∧ D ∈ l3) ∧
   (E ∈ l2 ∧ E ∈ l4) ∧
   (F ∈ l3 ∧ F ∈ l4) ∧
   -- Triangles are formed by the intersections
   t1 = Triangle.mk A E C ∧
   t2 = Triangle.mk B D C ∧
   t3 = Triangle.mk A B F ∧
   t4 = Triangle.mk E D F ∧
   -- Circles are circumcircles of the triangles
   (c1.center ≠ A ∧ c1.center ≠ E ∧ c1.center ≠ C) ∧
   (c2.center ≠ B ∧ c2.center ≠ D ∧ c2.center ≠ C) ∧
   (c3.center ≠ A ∧ c3.center ≠ B ∧ c3.center ≠ F) ∧
   (c4.center ≠ E ∧ c4.center ≠ D ∧ c4.center ≠ F) ∧
   (A ∈ c1) ∧ (E ∈ c1) ∧ (C ∈ c1) ∧
   (B ∈ c2) ∧ (D ∈ c2) ∧ (C ∈ c2) ∧
   (A ∈ c3) ∧ (B ∈ c3) ∧ (F ∈ c3) ∧
   (E ∈ c4) ∧ (D ∈ c4) ∧ (F ∈ c4)) →
  -- Conclusion
  ∃ (P : Point2D), P ∈ c1 ∧ P ∈ c2 ∧ P ∈ c3 ∧ P ∈ c4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_line_intersection_common_point_l1224_122493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_evaluation_l1224_122417

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem floor_expression_evaluation :
  (floor 6.5 : ℝ) * (floor (2 / 3) : ℝ) + (floor 2 : ℝ) * 7.2 + (floor 8.4 : ℝ) - 9.8 = 12.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_evaluation_l1224_122417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_ratio_l1224_122404

theorem square_perimeter_ratio (s₁ s₂ : ℝ) (h : s₁ > 0 ∧ s₂ > 0) :
  s₂ * Real.sqrt 2 = 2 * (s₁ * Real.sqrt 2) →
  (4 * s₂) / (4 * s₁) = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_ratio_l1224_122404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_problem_solution_l1224_122490

/-- Two people walking towards each other -/
structure WalkingProblem where
  initial_distance : ℝ
  speed_person1 : ℝ
  speed_person2 : ℝ

/-- Calculate the distance walked by one person when they meet -/
noncomputable def distance_walked (p : WalkingProblem) : ℝ :=
  (p.speed_person1 * p.initial_distance) / (p.speed_person1 + p.speed_person2)

/-- Theorem stating that in the given scenario, one person walks 25 miles -/
theorem walking_problem_solution :
  ∃ (p : WalkingProblem),
    p.initial_distance = 50 ∧
    p.speed_person1 = 5 ∧
    p.speed_person2 = 5 ∧
    distance_walked p = 25 := by
  -- Construct the WalkingProblem
  let p : WalkingProblem := {
    initial_distance := 50
    speed_person1 := 5
    speed_person2 := 5
  }
  -- Show that this WalkingProblem satisfies all conditions
  have h1 : p.initial_distance = 50 := rfl
  have h2 : p.speed_person1 = 5 := rfl
  have h3 : p.speed_person2 = 5 := rfl
  -- Calculate the distance walked
  have h4 : distance_walked p = 25 := by
    unfold distance_walked
    simp [h1, h2, h3]
    norm_num
  -- Prove the existence
  exact ⟨p, h1, h2, h3, h4⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_problem_solution_l1224_122490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_problem_l1224_122439

theorem complex_magnitude_problem (n : ℝ) :
  n > 0 →
  Complex.abs (⟨5, n⟩ : ℂ) = 5 * Real.sqrt 13 →
  n = 10 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_problem_l1224_122439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_tangents_inequality_l1224_122482

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1/a) * Real.log x + 1/x - x

/-- The derivative of f(x) -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := (a + 1/a) / x - 1 / (x^2) - 1

theorem parallel_tangents_inequality (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a ≥ 3) 
  (hx₁ : x₁ > 0) 
  (hx₂ : x₂ > 0) 
  (hx_ne : x₁ ≠ x₂) 
  (h_parallel : f_derivative a x₁ = f_derivative a x₂) : 
  x₁ + x₂ > 6/5 := by
  sorry

#check parallel_tangents_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_tangents_inequality_l1224_122482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_angle_relation_l1224_122475

theorem sin_angle_relation (α : ℝ) :
  Real.sin (π / 3 - α) = 1 / 4 → Real.sin (π / 6 - 2 * α) = -7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_angle_relation_l1224_122475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1224_122498

noncomputable def f (x : ℝ) : ℝ := |x| / (Real.sqrt (1 + x^2) * Real.sqrt (4 + x^2))

theorem f_max_value : 
  (∀ x : ℝ, f x ≤ 1/3) ∧ (∃ x : ℝ, f x = 1/3) := by
  sorry

#eval "The theorem has been stated and the proof is left as 'sorry'."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1224_122498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_household_income_savings_regression_l1224_122405

/-- Represents a sample of household incomes and savings -/
structure HouseholdSample where
  n : ℕ
  sum_x : ℝ
  sum_y : ℝ
  sum_xy : ℝ
  sum_x_sq : ℝ

/-- Calculates the linear regression coefficients -/
noncomputable def calculateRegressionCoefficients (sample : HouseholdSample) : ℝ × ℝ :=
  let x_mean := sample.sum_x / sample.n
  let y_mean := sample.sum_y / sample.n
  let b := (sample.sum_xy - sample.n * x_mean * y_mean) / (sample.sum_x_sq - sample.n * x_mean^2)
  let a := y_mean - b * x_mean
  (b, a)

/-- Predicts savings based on the regression equation -/
noncomputable def predictSavings (coefficients : ℝ × ℝ) (income : ℝ) : ℝ :=
  let (b, a) := coefficients
  b * income + a

theorem household_income_savings_regression 
  (sample : HouseholdSample)
  (h_n : sample.n = 10)
  (h_sum_x : sample.sum_x = 80)
  (h_sum_y : sample.sum_y = 20)
  (h_sum_xy : sample.sum_xy = 184)
  (h_sum_x_sq : sample.sum_x_sq = 720) :
  let coefficients := calculateRegressionCoefficients sample
  let (b, a) := coefficients
  b = 0.3 ∧ 
  a = -0.4 ∧ 
  b > 0 ∧ 
  predictSavings coefficients 7 = 1.7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_household_income_savings_regression_l1224_122405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_spanning_tree_exists_l1224_122467

/-- A color for graph edges -/
inductive Color
| Red
| Green
| Blue

/-- A graph with colored edges -/
structure ColoredGraph (V : Type) [Fintype V] where
  edges : V → V → Option Color

/-- A spanning tree of a graph -/
def SpanningTree (V : Type) [Fintype V] (g : ColoredGraph V) := V → V → Prop

/-- The number of edges of a given color in a spanning tree -/
def ColorCount (V : Type) [Fintype V] (g : ColoredGraph V) (t : SpanningTree V g) (c : Color) : ℕ := sorry

/-- A graph is connected if there's a path between any two vertices -/
def Connected (V : Type) [Fintype V] (g : ColoredGraph V) : Prop := sorry

theorem colored_spanning_tree_exists 
  {V : Type} [Fintype V]
  (r g b : ℕ) 
  (Γ : ColoredGraph V) 
  (h_connected : Connected V Γ)
  (h_vertex_count : Fintype.card V = r + g + b + 1)
  (h_red_tree : ∃ t : SpanningTree V Γ, ColorCount V Γ t Color.Red = r)
  (h_green_tree : ∃ t : SpanningTree V Γ, ColorCount V Γ t Color.Green = g)
  (h_blue_tree : ∃ t : SpanningTree V Γ, ColorCount V Γ t Color.Blue = b) :
  ∃ t : SpanningTree V Γ, 
    ColorCount V Γ t Color.Red = r ∧ 
    ColorCount V Γ t Color.Green = g ∧ 
    ColorCount V Γ t Color.Blue = b :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_spanning_tree_exists_l1224_122467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_integer_pairs_count_l1224_122492

theorem distinct_integer_pairs_count : 
  ∃! n : ℕ, n = Finset.card (Finset.filter 
    (fun p : ℕ × ℕ => 
      let (x, y) := p
      0 < x ∧ x < y ∧ (Real.sqrt 1690 : ℝ) = Real.sqrt (x : ℝ) + Real.sqrt (y : ℝ))
    (Finset.range 1691 ×ˢ Finset.range 1691)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_integer_pairs_count_l1224_122492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_natural_with_ten_divisors_l1224_122458

def has_exactly_ten_divisors (n : ℕ) : Prop :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 10

theorem smallest_natural_with_ten_divisors :
  ∃ n : ℕ, has_exactly_ten_divisors n ∧ ∀ m : ℕ, m < n → ¬has_exactly_ten_divisors m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_natural_with_ten_divisors_l1224_122458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_equality_l1224_122412

theorem angle_sum_equality (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sin α = Real.sqrt 5 / 5) (h4 : Real.cos β = 3 * Real.sqrt 10 / 10) :
  α + β = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_equality_l1224_122412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chips_required_l1224_122496

/-- Represents a point on the line -/
inductive Point
| A : Fin 10 → Point

/-- Represents the state of chips on the line -/
def State := Point → ℕ

/-- Represents a move in the game -/
inductive Move
| move1 : Fin 9 → Move
| move2 : Fin 8 → Move

/-- Applies a move to a state -/
def applyMove (s : State) (m : Move) : State :=
  match m with
  | Move.move1 i =>
    fun p => match p with
      | Point.A j =>
        if j = i then s p - 2
        else if j = i.succ then s p + 1
        else s p
  | Move.move2 i =>
    fun p => match p with
      | Point.A j =>
        if j = i then s p + 1
        else if j = i.succ then s p - 2
        else if j = i.succ.succ then s p + 1
        else s p

/-- Checks if a state is valid (no negative chips) -/
def isValidState (s : State) : Prop :=
  ∀ p, s p ≥ 0

/-- Checks if a state has a chip on A₁₀ -/
def hasChipOnA10 (s : State) : Prop :=
  s (Point.A ⟨9, by norm_num⟩) > 0

/-- Defines the initial state with n chips on A₁ -/
def initialState (n : ℕ) : State :=
  fun p => match p with
  | Point.A ⟨0, _⟩ => n
  | _ => 0

/-- Defines the reachability of a state through a sequence of moves -/
inductive Reachable : State → Prop
| initial {n : ℕ} : Reachable (initialState n)
| step {s s' : State} {m : Move} :
    Reachable s → isValidState s' → s' = applyMove s m → Reachable s'

/-- The main theorem: 46 is the minimum number of chips required -/
theorem min_chips_required :
  (∃ s, Reachable s ∧ hasChipOnA10 s) ↔ ∃ n ≥ 46, Reachable (initialState n) ∧ hasChipOnA10 (initialState n) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chips_required_l1224_122496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1224_122484

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (x + 2)

-- State the theorem
theorem f_domain : ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ x > -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1224_122484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sailboat_max_power_speed_l1224_122411

noncomputable def force (C S ρ v₀ v : ℝ) : ℝ := (C * S * ρ * (v₀ - v)^2) / 2

noncomputable def power (C S ρ v₀ v : ℝ) : ℝ := force C S ρ v₀ v * v

def wind_speed : ℝ := 6

theorem sailboat_max_power_speed (C S ρ : ℝ) (hC : C > 0) (hS : S > 0) (hρ : ρ > 0) :
  ∃ v : ℝ, v > 0 ∧ v < wind_speed ∧
    (∀ u : ℝ, u > 0 → u < wind_speed → power C S ρ wind_speed v ≥ power C S ρ wind_speed u) ∧
    v = wind_speed / 3 := by
  sorry

#check sailboat_max_power_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sailboat_max_power_speed_l1224_122411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1224_122489

-- Define a triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side a
  b : ℝ  -- Side b
  c : ℝ  -- Side c
  h1 : A + B + C = π  -- Sum of angles in a triangle
  h2 : a > 0 ∧ b > 0 ∧ c > 0  -- Positive side lengths

-- State the theorem
theorem triangle_problem (t : Triangle) 
  (h3 : (Real.cos t.A) / t.a + (Real.cos t.B) / t.b = (2 * Real.sqrt 3 * Real.sin t.C) / (3 * t.a))
  (h4 : (1/2) * t.a * t.c * Real.sin t.B = Real.sqrt 3 / 2)
  (h5 : t.B > π/2) :
  t.B = 2*π/3 ∧ t.b ≥ Real.sqrt 6 ∧ (t.b = Real.sqrt 6 ↔ t.a = t.c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1224_122489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_K_l1224_122429

def K : ℕ := 2^4 * 3^3 * 5^2 * 7^1

theorem number_of_factors_K : (Finset.filter (·∣K) (Finset.range (K + 1))).card = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_K_l1224_122429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_is_correct_l1224_122436

/-- The curve function f(x) = x³ - 2x + 1 -/
def f (x : ℝ) : ℝ := x^3 - 2*x + 1

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 2

/-- The point of tangency -/
def point : ℝ × ℝ := (1, 0)

/-- The slope of the tangent line at the point of tangency -/
def tangent_slope : ℝ := f' point.1

/-- The proposed equation of the tangent line -/
def tangent_line (x y : ℝ) : Prop := x - y - 1 = 0

theorem tangent_line_is_correct :
  (f point.1 = point.2) ∧ 
  (∀ x y : ℝ, tangent_line x y ↔ y - point.2 = tangent_slope * (x - point.1)) :=
by sorry

#check tangent_line_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_is_correct_l1224_122436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1224_122401

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 = 2*y

-- Define point P
def point_P : ℝ × ℝ := (0, 3)

-- Define the distance function
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem intersection_distance_sum :
  ∀ A B : ℝ × ℝ,
  line_l A.1 A.2 ∧ curve_C A.1 A.2 ∧
  line_l B.1 B.2 ∧ curve_C B.1 B.2 ∧
  A ≠ B →
  distance A.1 A.2 point_P.1 point_P.2 + distance B.1 B.2 point_P.1 point_P.2 = 2 * Real.sqrt 14 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1224_122401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_statements_correct_l1224_122402

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define properties of functions
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y
def is_decreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x > f y

-- Define the statements
def statement1 (f : ℝ → ℝ) : Prop := f (-3) = -f 3 → is_odd f
def statement2 (f : ℝ → ℝ) : Prop := f (-3) ≠ f 3 → ¬(is_even f)
def statement3 (f : ℝ → ℝ) : Prop := f 1 < f 2 → is_increasing f
def statement4 (f : ℝ → ℝ) : Prop := f 1 < f 2 → ¬(is_decreasing f)

-- Theorem stating that exactly two statements are correct
theorem two_statements_correct (f : ℝ → ℝ) :
  (statement1 f ∨ statement2 f ∨ statement3 f ∨ statement4 f) ∧
  (statement1 f ∧ statement2 f ∨
   statement1 f ∧ statement3 f ∨
   statement1 f ∧ statement4 f ∨
   statement2 f ∧ statement3 f ∨
   statement2 f ∧ statement4 f ∨
   statement3 f ∧ statement4 f) ∧
  ¬(statement1 f ∧ statement2 f ∧ statement3 f ∨
    statement1 f ∧ statement2 f ∧ statement4 f ∨
    statement1 f ∧ statement3 f ∧ statement4 f ∨
    statement2 f ∧ statement3 f ∧ statement4 f) ∧
  ¬(statement1 f ∧ statement2 f ∧ statement3 f ∧ statement4 f) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_statements_correct_l1224_122402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_l1224_122494

/-- The number of ways to arrange 5 distinct events with constraints -/
theorem arrangement_count : ℕ := by
  -- Define the total number of events
  let total_events : ℕ := 5

  -- Define the number of positions available for the constrained event (not first)
  let positions_for_constrained : ℕ := total_events - 2

  -- Define the number of remaining events to arrange
  let remaining_events : ℕ := total_events - 2

  -- Calculate the number of arrangements
  let arrangements : ℕ := positions_for_constrained * (Nat.factorial remaining_events)

  -- Assert that the number of arrangements is 18
  have h : arrangements = 18 := by sorry

  -- Prove the theorem by providing the calculated result
  exact 18


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_l1224_122494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_horses_count_l1224_122403

/-- Represents the pasture rental problem -/
structure PastureRental where
  total_rent : ℕ
  a_months : ℕ
  b_horses : ℕ
  b_months : ℕ
  c_horses : ℕ
  c_months : ℕ
  b_payment : ℕ

/-- Calculates the number of horses a put in the pasture -/
def calculate_a_horses (pr : PastureRental) : ℕ :=
  let ac_cost := pr.total_rent - pr.b_payment
  let c_cost := pr.c_horses * pr.c_months
  let a_cost := ac_cost - c_cost
  a_cost / pr.a_months

/-- Theorem stating that a put 73 horses in the pasture -/
theorem a_horses_count (pr : PastureRental) 
  (h1 : pr.total_rent = 841)
  (h2 : pr.a_months = 8)
  (h3 : pr.b_horses = 16)
  (h4 : pr.b_months = 9)
  (h5 : pr.c_horses = 18)
  (h6 : pr.c_months = 6)
  (h7 : pr.b_payment = 348) :
  calculate_a_horses pr = 73 := by
  sorry

#eval calculate_a_horses {
  total_rent := 841,
  a_months := 8,
  b_horses := 16,
  b_months := 9,
  c_horses := 18,
  c_months := 6,
  b_payment := 348
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_horses_count_l1224_122403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dataset_mean_unchanged_l1224_122461

theorem dataset_mean_unchanged (n : ℕ) (initial_mean : ℝ) : 
  n = 150 ∧ initial_mean = 300 →
  let multiples_of_3 := n / 3
  let non_multiples := n - multiples_of_3
  let total_decrement := (multiples_of_3 : ℝ) * (-30)
  let total_increment := (non_multiples : ℝ) * 15
  total_decrement + total_increment = 0 →
  initial_mean = 300 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dataset_mean_unchanged_l1224_122461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_multiple_of_18_to_2500_is_correct_l1224_122499

def closest_multiple_of_18_to_2500 : ℕ := 2502

theorem closest_multiple_of_18_to_2500_is_correct :
  closest_multiple_of_18_to_2500 = 2502 ∧
  closest_multiple_of_18_to_2500 % 18 = 0 ∧
  ∀ n : ℕ, n % 18 = 0 → (n : ℤ) - 2500 ≥ (closest_multiple_of_18_to_2500 : ℤ) - 2500 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_multiple_of_18_to_2500_is_correct_l1224_122499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_results_l1224_122479

-- Define the movements of Xiao Wen and Xiao Li
def xiao_wen_movements : List Int := [3, -2, -1, 4, -5]
def xiao_li_movements : List Int := [-4, 3, 5, -2]

-- Define Xiao Li's final position
def xiao_li_final_position : Int := 5

-- Define Xiao Wen's scoring system
def eastward_score : Int := 3
def westward_score : Int := 2

-- Theorem to prove
theorem game_results :
  -- 1. Xiao Wen's final position
  (List.sum xiao_wen_movements = -1) ∧
  -- 2. Xiao Li's movement in the 5th game
  (List.sum xiao_li_movements + 3 = xiao_li_final_position) ∧
  -- 3. Difference in scores
  (let xiao_wen_score := (List.sum (List.filter (· > 0) xiao_wen_movements) * eastward_score -
                          List.sum (List.map Int.natAbs (List.filter (· < 0) xiao_wen_movements)) * westward_score)
   let xiao_li_score := ((List.sum (List.filter (· > 0) xiao_li_movements) + 3) * eastward_score -
                         List.sum (List.map Int.natAbs (List.filter (· < 0) xiao_li_movements)) * westward_score)
   xiao_li_score - xiao_wen_score = 16) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_results_l1224_122479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_printer_time_proof_l1224_122443

def pages_to_print : ℕ := 300
def pages_per_minute : ℕ := 24

def time_to_print : ℚ := pages_to_print / pages_per_minute

-- Use Int.floor instead of Rat.floor, and convert to ℕ
def rounded_time_to_print : ℕ := (Int.floor (time_to_print + 1/2)).toNat

theorem printer_time_proof : rounded_time_to_print = 13 := by
  -- Proof steps would go here
  sorry

#eval rounded_time_to_print  -- This will evaluate the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_printer_time_proof_l1224_122443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_of_four_five_digit_numbers_l1224_122445

/-- A list representing a five-digit number -/
def FiveDigitNumber := List Nat

/-- Check if a number is a valid five-digit number -/
def isValidFiveDigitNumber (n : FiveDigitNumber) : Bool :=
  n.length = 5 && n.all (· < 10)

/-- Check if a list of numbers uses each digit 0,1,2,3,4 exactly twice -/
def usesDigitsTwice (numbers : List FiveDigitNumber) : Prop :=
  let allDigits := numbers.join
  ∀ d : Nat, d ≤ 4 → (allDigits.count d = 2)

/-- Convert a FiveDigitNumber to its numerical value -/
def toNumber (n : FiveDigitNumber) : Nat :=
  n.foldl (fun acc d => acc * 10 + d) 0

/-- Sum of a list of FiveDigitNumbers -/
def sumNumbers (numbers : List FiveDigitNumber) : Nat :=
  numbers.map toNumber |>.sum

/-- The main theorem -/
theorem largest_sum_of_four_five_digit_numbers :
  ∃ (numbers : List FiveDigitNumber),
    numbers.length = 4 ∧
    numbers.all isValidFiveDigitNumber ∧
    usesDigitsTwice numbers ∧
    sumNumbers numbers = 150628 ∧
    ∀ (otherNumbers : List FiveDigitNumber),
      otherNumbers.length = 4 →
      otherNumbers.all isValidFiveDigitNumber →
      usesDigitsTwice otherNumbers →
      sumNumbers otherNumbers ≤ 150628 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_of_four_five_digit_numbers_l1224_122445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_line_l1224_122441

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 6*y + 17 = 0

-- Define a line passing through the origin
def line_through_origin (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x

-- Define the concept of a line intersecting the circle
def intersects_circle (k : ℝ) : Prop :=
  ∃ x y : ℝ, circle_C x y ∧ line_through_origin k x y

-- Define the concept of chord length
-- (We don't implement the actual calculation, just define the concept)
noncomputable def chord_length (k : ℝ) : ℝ := sorry

-- Define the concept of maximum chord length
def has_max_chord_length (k : ℝ) : Prop :=
  intersects_circle k ∧
  ∀ k' : ℝ, intersects_circle k' → chord_length k ≥ chord_length k'

-- The theorem to prove
theorem max_chord_line :
  ∃ k : ℝ, has_max_chord_length k ∧ (∀ x y : ℝ, line_through_origin k x y ↔ x - y = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_line_l1224_122441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_A_is_zero_matrix_l1224_122426

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, 10; -8, -20]

theorem inverse_of_A_is_zero_matrix :
  A⁻¹ = !![0, 0; 0, 0] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_A_is_zero_matrix_l1224_122426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_bound_l1224_122432

def sequence_a : ℕ → ℕ
  | 0 => 1
  | n + 1 => sequence_a n + (n + 1)

theorem lambda_bound (lambda : ℝ) :
  (∀ n : ℕ, n > 0 → lambda / n > (n + 1) / (sequence_a n + 1)) →
  lambda ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_bound_l1224_122432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_negative_one_l1224_122442

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

theorem tangent_line_at_negative_one (a : ℝ) :
  (∀ x, HasDerivAt (f a) (3 * a * x^2 + 6 * x) x) →
  HasDerivAt (f a) 3 (-1) →
  ∃ m b, ∀ x, m * x + b = 3 * x + 5 ∧
    HasDerivAt (fun x => m * x + b) m (-1) ∧
    (m * (-1) + b = f a (-1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_negative_one_l1224_122442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_fraction_l1224_122477

theorem decimal_to_fraction : 
  ∃ (s : ℚ), (∀ n : ℕ, s * 10^(6*n) - ⌊s * 10^(6*n)⌋ = 0.142857) ∧ s = 1/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_fraction_l1224_122477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1224_122463

theorem triangle_problem (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + c^2 - b^2 = Real.sqrt 3 * a * c) :
  let A := Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))
  let B := Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))
  let C := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  ∃ (S : ℝ), 
    B = π / 6 ∧ 
    (b = 2 ∧ c = 2 * Real.sqrt 3 → S = Real.sqrt 3 ∨ S = 2 * Real.sqrt 3) ∧
    S = 1/2 * a * c * Real.sin B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1224_122463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_exists_l1224_122480

-- Define the function
noncomputable def f (a x : ℝ) : ℝ := a^(2*x - 1) - 2

-- State the theorem
theorem fixed_point_exists (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  f a (1/2) = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_exists_l1224_122480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jessica_money_l1224_122406

noncomputable def weekly_allowance : ℚ := 10
noncomputable def movie_expense : ℚ := weekly_allowance / 2
noncomputable def car_wash_earnings : ℚ := 6

theorem jessica_money : 
  weekly_allowance - movie_expense + car_wash_earnings = 11 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jessica_money_l1224_122406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_proof_standard_to_final_form_l1224_122453

-- Define the hyperbola
structure Hyperbola where
  center : ℝ × ℝ
  foci_axis : ℝ × ℝ → Prop
  axes_equal : Prop
  focus_to_asymptote : ℝ

-- Define the properties of our specific hyperbola
noncomputable def our_hyperbola : Hyperbola where
  center := (0, 0)
  foci_axis := λ (x, y) ↦ y = 0
  axes_equal := True
  focus_to_asymptote := Real.sqrt 2

-- Define the equation of a hyperbola
def hyperbola_equation (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / a^2 = 1

-- Theorem statement
theorem hyperbola_equation_proof (h : Hyperbola) :
  h = our_hyperbola → hyperbola_equation (Real.sqrt 2) x y :=
by
  sorry

-- Additional theorem to connect the standard form to the final equation
theorem standard_to_final_form (x y : ℝ) :
  hyperbola_equation (Real.sqrt 2) x y ↔ x^2 - y^2 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_proof_standard_to_final_form_l1224_122453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1224_122448

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + a) / x

-- State the theorem
theorem f_properties (a : ℝ) :
  (f a 1 = 2) →
  (∀ x ≠ 0, f a (-x) = -(f a x)) ∧
  (∀ x > 1, (deriv (f a)) x > 0) ∧
  (∃ x_max ∈ Set.Icc 2 5, ∀ x ∈ Set.Icc 2 5, f a x ≤ f a x_max) ∧
  (∃ x_min ∈ Set.Icc 2 5, ∀ x ∈ Set.Icc 2 5, f a x ≥ f a x_min) ∧
  (f a 5 = 26/5) ∧
  (f a 2 = 5/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1224_122448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_fraction_product_l1224_122418

theorem inverse_sum_fraction_product : (12 : ℚ) * ((1/3 : ℚ) + (1/4 : ℚ) + (1/6 : ℚ))⁻¹ = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_fraction_product_l1224_122418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fiftiethTerm_eq_1284_l1224_122481

/-- The sequence of positive integers that are either powers of 4 or sums of distinct powers of 4 -/
def powerOf4Sequence : ℕ → ℕ := sorry

/-- The 50th term of the powerOf4Sequence -/
def fiftiethTerm : ℕ := powerOf4Sequence 50

/-- Theorem stating that the 50th term of the powerOf4Sequence is 1284 -/
theorem fiftiethTerm_eq_1284 : fiftiethTerm = 1284 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fiftiethTerm_eq_1284_l1224_122481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_is_correct_l1224_122456

/-- The equation of an ellipse -/
noncomputable def ellipse_equation (x y : ℝ) : Prop := y^2 / 49 + x^2 / 24 = 1

/-- The eccentricity of the hyperbola -/
def hyperbola_eccentricity : ℚ := 5 / 4

/-- The equation of the hyperbola we want to prove -/
noncomputable def hyperbola_equation (x y : ℝ) : Prop := y^2 / 16 - x^2 / 9 = 1

/-- Theorem stating that the given hyperbola equation is correct -/
theorem hyperbola_equation_is_correct :
  ∀ x y : ℝ, ellipse_equation x y →
  ∃ (focus_x focus_y : ℝ), 
    (focus_x = 0 ∧ (focus_y = 5 ∨ focus_y = -5)) →
    hyperbola_equation x y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_is_correct_l1224_122456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l1224_122455

noncomputable def a : ℕ → ℝ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 1
  | (n + 3) => (n.succ.succ^2 * a (n + 2)^2 + 5) / ((n.succ.succ^2 - 1) * a (n + 1))

theorem a_formula (n : ℕ) (hn : n > 0): 
  a n = (1 / n) * ((63 - 13 * Real.sqrt 21) / 42 * ((5 + Real.sqrt 21) / 2)^n + 
                   (63 + 13 * Real.sqrt 21) / 42 * ((5 - Real.sqrt 21) / 2)^n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l1224_122455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_side_l1224_122446

/-- Two similar triangles with given properties -/
structure SimilarTriangles where
  smaller_area : ℝ
  smaller_side : ℝ
  area_ratio : ℝ

/-- The corresponding side of the larger triangle -/
noncomputable def larger_side (t : SimilarTriangles) : ℝ :=
  t.smaller_side * (t.area_ratio ^ (1/2 : ℝ))

/-- Theorem stating the relationship between the triangles -/
theorem similar_triangles_side (t : SimilarTriangles) 
  (h1 : t.smaller_area = 16)
  (h2 : t.smaller_side = 4)
  (h3 : t.area_ratio = 9) :
  larger_side t = 12 := by
  sorry

#check similar_triangles_side

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_side_l1224_122446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_equation_l1224_122438

-- Define the line x - y + 2 = 0
def line_xy (x y : ℝ) : Prop := x - y + 2 = 0

-- Define the angle of inclination of a line
noncomputable def angle_of_inclination (m : ℝ) : ℝ := Real.arctan m

-- Define the point P
def point_P : ℝ × ℝ := (1, -1)

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  ∃ (m : ℝ), y - point_P.2 = m * (x - point_P.1) ∧
  angle_of_inclination m = 2 * angle_of_inclination 1

-- Theorem statement
theorem line_l_equation :
  ∀ x y : ℝ, line_l x y ↔ x = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_equation_l1224_122438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_formula_l1224_122427

-- Define the rectangle's properties
def rectangle_perimeter (a : ℝ) : ℝ :=
  let width := a
  let length := 2 * a + 1
  2 * (width + length)

-- Theorem statement
theorem rectangle_perimeter_formula (a : ℝ) :
  rectangle_perimeter a = 6 * a + 2 := by
  -- Unfold the definition of rectangle_perimeter
  unfold rectangle_perimeter
  -- Simplify the expression
  simp
  -- Perform algebraic manipulations
  ring

-- Example usage
example (a : ℝ) : rectangle_perimeter a = 6 * a + 2 :=
  rectangle_perimeter_formula a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_formula_l1224_122427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_fh_l1224_122450

-- Define the functions f, h, and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def h : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Define the ranges of f and h
axiom f_range : ∀ x, -3 ≤ f x ∧ f x ≤ 4
axiom h_range : ∀ x, -1 ≤ h x ∧ h x ≤ 2

-- Define the relationship between h and g
axiom h_def : ∀ x, h x = g x + 1

-- Theorem statement
theorem max_product_fh : 
  (∀ x, f x * h x ≤ 12) ∧ (∃ x, f x * h x = 12) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_fh_l1224_122450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rounded_octahedron_properties_l1224_122444

/-- A regular octahedron with vertices rounded by a sphere that touches all its edges. -/
structure RoundedOctahedron where
  /-- The edge length of the original regular octahedron. -/
  a : ℝ
  /-- The sphere touches all edges of the octahedron. -/
  sphere_touches_edges : True
  /-- The sphere's center coincides with the octahedron's center. -/
  sphere_center_coincides : True

/-- The surface area of a rounded octahedron. -/
noncomputable def surface_area (o : RoundedOctahedron) : ℝ :=
  Real.pi * (4 * Real.sqrt 6 - 7) / 3 * o.a^2

/-- The volume of a rounded octahedron. -/
noncomputable def volume (o : RoundedOctahedron) : ℝ :=
  Real.pi * (14 * Real.sqrt 6 - 27) / 54 * o.a^3

/-- Theorem stating the correct surface area and volume of a rounded octahedron. -/
theorem rounded_octahedron_properties (o : RoundedOctahedron) :
  (surface_area o = Real.pi * (4 * Real.sqrt 6 - 7) / 3 * o.a^2) ∧
  (volume o = Real.pi * (14 * Real.sqrt 6 - 27) / 54 * o.a^3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rounded_octahedron_properties_l1224_122444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1224_122469

noncomputable def f (x : ℝ) : ℝ := (2 - Real.sqrt 2 * Real.sin (Real.pi * x / 4)) / (x^2 + 4*x + 5)

theorem max_value_of_f :
  ∃ (M : ℝ), M = 2 + Real.sqrt 2 ∧
  ∀ (x : ℝ), -4 ≤ x ∧ x ≤ 0 → f x ≤ M :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1224_122469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_five_digits_l1224_122419

/-- Represents the sequence of digits in Bill's list -/
def billList : ℕ → ℕ := sorry

/-- The number of digits written before a given position in the list -/
def digitCount : ℕ → ℕ := sorry

/-- The five-digit number formed by digits at positions n to n+4 in the list -/
def fiveDigitNumber (n : ℕ) : ℕ := sorry

/-- The list starts with 2 and increases -/
axiom list_start : billList 1 = 2

/-- The list is in increasing order -/
axiom list_increasing : ∀ n : ℕ, billList n < billList (n + 1)

/-- The digit count is correct -/
axiom digit_count_correct : digitCount 1250 = 1250

/-- The main theorem: the five-digit number formed by the 1246th to 1250th digits is 20002 -/
theorem last_five_digits : fiveDigitNumber 1246 = 20002 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_five_digits_l1224_122419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_price_after_discounts_l1224_122430

/-- Given a dress with original price d, prove that after applying a 55% discount
    and then an additional 50% staff discount, the final price is 0.225d. -/
theorem dress_price_after_discounts (d : ℝ) :
  d * (1 - 0.55) * (1 - 0.50) = d * 0.225 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_price_after_discounts_l1224_122430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_lines_through_single_point_l1224_122447

/-- A line in a plane --/
structure Line where
  -- Define a line (you might want to use a more specific representation depending on your needs)
  mk :: -- Add a constructor

/-- A point in a plane --/
structure Point where
  -- Define a point (you might want to use a more specific representation depending on your needs)
  mk :: -- Add a constructor

/-- Check if two lines are parallel --/
def are_parallel (l1 l2 : Line) : Prop :=
  sorry -- Define what it means for two lines to be parallel

/-- Check if a point lies on a line --/
def point_on_line (p : Point) (l : Line) : Prop :=
  sorry -- Define what it means for a point to lie on a line

/-- Check if a point is an intersection of two lines --/
def is_intersection (p : Point) (l1 l2 : Line) : Prop :=
  point_on_line p l1 ∧ point_on_line p l2

/-- The main theorem --/
theorem all_lines_through_single_point 
  (lines : Finset Line) 
  (h1 : ∀ l1 l2 : Line, l1 ∈ lines → l2 ∈ lines → l1 ≠ l2 → ¬are_parallel l1 l2)
  (h2 : ∀ l1 l2 l3 : Line, l1 ∈ lines → l2 ∈ lines → l3 ∈ lines → l1 ≠ l2 → 
        ∃ p, is_intersection p l1 l2 → ∃ l ∈ lines, l ≠ l1 ∧ l ≠ l2 ∧ point_on_line p l) :
  ∃ p, ∀ l ∈ lines, point_on_line p l :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_lines_through_single_point_l1224_122447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_ratio_l1224_122437

/-- Represents a square pyramid -/
structure SquarePyramid where
  baseEdge : ℝ
  altitude : ℝ

/-- Calculate the volume of a square pyramid -/
noncomputable def pyramidVolume (p : SquarePyramid) : ℝ :=
  (1/3) * p.baseEdge^2 * p.altitude

/-- Calculate the volume of a frustum formed by removing a smaller similar pyramid -/
noncomputable def frustumVolume (p : SquarePyramid) (ratio : ℝ) : ℝ :=
  pyramidVolume p - pyramidVolume { baseEdge := p.baseEdge * ratio, altitude := p.altitude * ratio }

/-- The main theorem -/
theorem frustum_volume_ratio (p : SquarePyramid) (h1 : p.baseEdge = 48) (h2 : p.altitude = 18) :
  frustumVolume p (1/3) / pyramidVolume p = 26/27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_ratio_l1224_122437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_of_implication_l1224_122408

theorem converse_of_implication (p q : Prop) : 
  (q → p) = (q → p) := by 
  rfl

#check converse_of_implication

end NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_of_implication_l1224_122408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tamika_logan_distance_difference_l1224_122424

/-- Given Tamika's and Logan's driving times and speeds, calculates how many miles farther Tamika drove. -/
theorem tamika_logan_distance_difference 
  (tamika_time : ℝ) (tamika_speed : ℝ) (logan_time : ℝ) (logan_speed : ℝ)
  (h1 : tamika_time = 8)
  (h2 : tamika_speed = 45)
  (h3 : logan_time = 5)
  (h4 : logan_speed = 55) :
  tamika_time * tamika_speed - logan_time * logan_speed = 85 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tamika_logan_distance_difference_l1224_122424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odot_eight_six_l1224_122486

-- Define the ⊙ operation
noncomputable def odot (a b : ℝ) : ℝ := a + (3 * a) / (2 * b)

-- State the theorem
theorem odot_eight_six : odot 8 6 = 10 := by
  -- Unfold the definition of odot
  unfold odot
  -- Simplify the expression
  simp [add_div, mul_div_assoc]
  -- Perform the arithmetic
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odot_eight_six_l1224_122486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_property_max_value_on_interval_l1224_122454

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x ≤ y → f y ≤ f x

theorem even_function_property (f : ℝ → ℝ) (a : ℝ) 
  (h1 : is_even_function f)
  (h2 : is_decreasing_on f (Set.Iic 0))
  (h3 : f a ≤ f 2) :
  -2 ≤ a ∧ a ≤ 2 := by
  sorry

theorem max_value_on_interval (a : ℝ) (h : -2 ≤ a ∧ a ≤ 2) :
  a^2 - 2*a + 2 ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_property_max_value_on_interval_l1224_122454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_intersecting_ellipse_l1224_122466

/-- Given an ellipse C and a line l with slope 1 intersecting C at points A and B,
    prove that the equation of l is y = x ± 1 if the distance between A and B is 3√2/2 -/
theorem line_equation_intersecting_ellipse (C l : Set (ℝ × ℝ)) 
  (A B : ℝ × ℝ) (m : ℝ) :
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2/3 + y^2 = 1) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ y = x + m) →
  A ∈ C ∧ A ∈ l →
  B ∈ C ∧ B ∈ l →
  A ≠ B →
  ‖A - B‖ = 3 * Real.sqrt 2 / 2 →
  m = 1 ∨ m = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_intersecting_ellipse_l1224_122466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_period_l1224_122434

open Real Matrix

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![cos θ, -sin θ],
    ![sin θ,  cos θ]]

def is_identity (M : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  M = ![![1, 0],
       ![0, 1]]

theorem smallest_rotation_period :
  (∀ k : ℕ, k > 0 → k < 12 → ¬(is_identity ((rotation_matrix (150 * π / 180))^k))) ∧
  (is_identity ((rotation_matrix (150 * π / 180))^12)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_period_l1224_122434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1224_122414

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x^2 - x^4)) / (abs (x - 1) - 1)

-- Define the domain of f
def domain (x : ℝ) : Prop := x ∈ Set.Icc (-1) 0 ∪ Set.Ioc 0 1

-- Theorem stating the properties of f
theorem f_properties :
  (∀ x, domain x → f x ∈ Set.Ioo (-1) 1) ∧ 
  (∀ x, domain x → domain (-x) ∧ f (-x) = -f x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1224_122414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_unity_sum_l1224_122465

theorem fourth_root_unity_sum (ω : ℂ) : 
  ω^4 = 1 ∧ ω ≠ 1 ∧ ω ≠ -1 → (1 - ω + ω^3)^4 + (1 + ω - ω^3)^4 = -14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_unity_sum_l1224_122465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_purchase_total_check_result_l1224_122449

/-- Calculates the discounted price based on the given rules -/
noncomputable def discounted_price (price : ℝ) : ℝ :=
  if price ≤ 200 then price
  else if price ≤ 500 then price * 0.9
  else 500 * 0.9 + (price - 500) * 0.8

/-- Theorem stating that the combined purchase results in the correct total payment -/
theorem combined_purchase_total (purchase1 purchase2 : ℝ) 
  (h1 : purchase1 = 168) (h2 : purchase2 = 423) : 
  discounted_price (purchase1 + (purchase2 / 0.9)) = 560.4 := by
  sorry

-- Use #eval only for computable functions
-- Instead, we can use the following to check the result:
theorem check_result : 
  discounted_price (168 + (423 / 0.9)) = 560.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_purchase_total_check_result_l1224_122449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_fraction_l1224_122497

def target : ℚ := 20.14

def fractions : List ℚ := [101/5, 141/7, 181/9, 161/8]

def is_simplest (q : ℚ) : Prop := 
  ∀ a b : ℤ, (a : ℚ) / b = q → b > 0 → Int.gcd a b = 1

def denominator_less_than_10 (q : ℚ) : Prop :=
  (q.den : ℕ) < 10

theorem closest_fraction :
  ∀ q ∈ fractions, 
    is_simplest q → 
    denominator_less_than_10 q →
    ∀ r ∈ fractions, 
      is_simplest r → 
      denominator_less_than_10 r →
      |q - target| ≤ |r - target| →
      q = 141/7 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_fraction_l1224_122497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_area_ratio_l1224_122473

noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

noncomputable def ring_area (outer_r inner_r : ℝ) : ℝ := circle_area outer_r - circle_area inner_r

theorem concentric_circles_area_ratio :
  let r1 : ℝ := 3
  let r2 : ℝ := 5
  let r3 : ℝ := 7
  let r4 : ℝ := 9
  let white_area : ℝ := circle_area r1 + ring_area r3 r2
  let black_area : ℝ := ring_area r2 r1 + ring_area r4 r3
  black_area / white_area = 16 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_area_ratio_l1224_122473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_composition_equals_sqrt_six_l1224_122459

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Define the inverse functions
noncomputable def f_inv : ℝ → ℝ := Function.invFun f
noncomputable def g_inv : ℝ → ℝ := Function.invFun g

-- State the given condition
axiom condition : ∀ x, f_inv (g x) = x^4 - 4*x^2 + 3

-- State that g has an inverse
axiom g_has_inverse : Function.Bijective g

-- Theorem to prove
theorem inverse_composition_equals_sqrt_six :
  g_inv (f 15) = Real.sqrt 6 ∨ g_inv (f 15) = -Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_composition_equals_sqrt_six_l1224_122459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_equivalence_l1224_122410

-- Define the function
noncomputable def f (θ : Real) := Real.log (Real.cos θ * Real.tan θ)

-- Define the domain of the function
def domain (θ : Real) : Prop :=
  Real.cos θ * Real.tan θ > 0 ∧ Real.cos θ * Real.tan θ ≠ 1

-- Define the first and second quadrant
def first_or_second_quadrant (θ : Real) : Prop :=
  0 < Real.sin θ ∧ Real.sin θ < 1

-- Theorem statement
theorem domain_equivalence :
  ∀ θ : Real, domain θ ↔ first_or_second_quadrant θ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_equivalence_l1224_122410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_result_l1224_122485

/-- Calculates the final amount after compound interest --/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) (frequency : ℝ) : ℝ :=
  principal * (1 + rate / frequency) ^ (frequency * time)

/-- Theorem: An investment of $3000 at 10% annual compound interest for 2 years, 
    compounded annually, will result in a final amount of $3630 --/
theorem investment_result : 
  let principal : ℝ := 3000
  let rate : ℝ := 0.10
  let time : ℝ := 2
  let frequency : ℝ := 1
  compound_interest principal rate time frequency = 3630 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_result_l1224_122485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_values_for_point_l1224_122460

def point_on_terminal_side (a : ℝ) (α : ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ r * (Real.cos α) = -15 * a ∧ r * (Real.sin α) = 8 * a

theorem trig_values_for_point (a : ℝ) (α : ℝ) (h : a ≠ 0) 
  (h_terminal : point_on_terminal_side a α) :
  (a > 0 → 
    Real.sin α = 8/17 ∧ 
    Real.cos α = -15/17 ∧ 
    Real.tan α = -8/15 ∧ 
    (1 / Real.sin α) = 17/8 ∧ 
    (1 / Real.cos α) = -17/15 ∧ 
    (1 / Real.tan α) = -15/8) ∧
  (a < 0 → 
    Real.sin α = -8/17 ∧ 
    Real.cos α = 15/17 ∧ 
    Real.tan α = -8/15 ∧ 
    (1 / Real.sin α) = -17/8 ∧ 
    (1 / Real.cos α) = 17/15 ∧ 
    (1 / Real.tan α) = -15/8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_values_for_point_l1224_122460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1224_122423

noncomputable def f (a k x : ℝ) : ℝ := a^x + k*a^(-x)

theorem problem_solution (a k : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x, f a k x = -f a k (-x)) →
  f a k 1 = 3/2 →
  (a = 2 ∧ k = -1) ∧
  (∀ m : ℝ, (∃ x ∈ Set.Icc (1/2 : ℝ) 2, f a k (2*x^2) + f a k (1 - m*x) < 0) ↔ m > 2*Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1224_122423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tim_john_age_ratio_l1224_122488

theorem tim_john_age_ratio : 
  let john_age : ℕ := 35
  let tim_age : ℕ := 79
  let R : ℚ := tim_age / john_age
  tim_age = (R * john_age).floor - 5 →
  R = 2.4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tim_john_age_ratio_l1224_122488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_problem_l1224_122440

theorem wire_length_problem (total_wires : ℕ) (avg_length_all : ℝ) (avg_length_third : ℝ) :
  total_wires = 6 →
  avg_length_all = 80 →
  avg_length_third = 70 →
  let total_length : ℝ := (total_wires : ℝ) * avg_length_all
  let third_wires : ℕ := total_wires / 3
  let length_third : ℝ := (third_wires : ℝ) * avg_length_third
  let remaining_wires : ℕ := total_wires - third_wires
  let remaining_length : ℝ := total_length - length_third
  remaining_length / (remaining_wires : ℝ) = 85 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_problem_l1224_122440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_shift_l1224_122495

-- Define the two functions as noncomputable
noncomputable def f (x : ℝ) : ℝ := 3 * Real.cos (2 * x - Real.pi / 2)
noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 4)

-- Theorem statement
theorem horizontal_shift : 
  ∀ x : ℝ, f x = g (x + Real.pi / 8) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_shift_l1224_122495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_property_l1224_122457

-- Define the ellipse
def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define a point on the ellipse
def point_on_ellipse (a b : ℝ) (p : ℝ × ℝ) : Prop :=
  ellipse a b p.fst p.snd

-- Define the right focus of the ellipse
noncomputable def right_focus (a b : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 - b^2), 0)

-- Define a line passing through a point
def line_through_point (p : ℝ × ℝ) (x y : ℝ) : Prop :=
  ∃ (m c : ℝ), y = m * x + c ∧ p.snd = m * p.fst + c

-- Define the intersection of a line with the y-axis
def intersect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (0, p.snd)

-- Main theorem
theorem ellipse_focal_property (a b : ℝ) (A B P : ℝ × ℝ) :
  a > 0 → b > 0 → a > b →
  point_on_ellipse a b A →
  point_on_ellipse a b B →
  line_through_point (right_focus a b) A.fst A.snd →
  line_through_point (right_focus a b) B.fst B.snd →
  P = intersect_y_axis (right_focus a b) →
  dist P A + dist P B = 2 * a :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_property_l1224_122457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_properties_l1224_122474

theorem quadratic_equation_properties :
  ∀ m : ℝ,
  let a : ℝ := 1
  let b : ℝ := -3
  let c : ℝ := 2 - m^2 - m
  let discriminant : ℝ := b^2 - 4*a*c
  let α : ℝ := (-b + Real.sqrt discriminant) / (2*a)
  let β : ℝ := (-b - Real.sqrt discriminant) / (2*a)
  (discriminant ≥ 0) ∧
  (α^2 + β^2 = 9 → m = -2 ∨ m = 1) := by
  intro m
  sorry

#check quadratic_equation_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_properties_l1224_122474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_intervals_of_f_l1224_122425

noncomputable def f (x : ℝ) : ℝ := Real.tan (x + Real.pi / 4)

theorem monotonic_increasing_intervals_of_f :
  ∀ k : ℤ, StrictMonoOn f (Set.Ioo ((k : ℝ) * Real.pi - 3 * Real.pi / 4) ((k : ℝ) * Real.pi + Real.pi / 4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_intervals_of_f_l1224_122425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l1224_122407

/-- Compound interest calculation --/
noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

/-- Proof that the initial investment of $36,113 grows to approximately $70,000 --/
theorem investment_growth : 
  let P : ℝ := 36113
  let r : ℝ := 0.08
  let n : ℝ := 12
  let t : ℝ := 8
  abs (compound_interest P r n t - 70000) < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l1224_122407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_m_l1224_122476

-- Define the hyperbola
noncomputable def hyperbola (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / m - y^2 / 3 = 1

-- Define the eccentricity
noncomputable def eccentricity (m : ℝ) : ℝ :=
  Real.sqrt (1 + 3 / m)

-- Theorem statement
theorem hyperbola_eccentricity_m (m : ℝ) :
  (∀ x y, hyperbola m x y) → eccentricity m = 2 → m = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_m_l1224_122476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peregrine_falcon_diving_time_l1224_122409

/-- The diving speed of a bald eagle in miles per hour -/
noncomputable def bald_eagle_speed : ℝ := 100

/-- The diving speed of a peregrine falcon in miles per hour -/
noncomputable def peregrine_falcon_speed : ℝ := 2 * bald_eagle_speed

/-- The time it takes a bald eagle to dive to the ground in seconds -/
noncomputable def bald_eagle_time : ℝ := 30

/-- The diving distance in miles -/
noncomputable def diving_distance : ℝ := (bald_eagle_speed / 3600) * bald_eagle_time

/-- The theorem stating the time it takes a peregrine falcon to dive the same distance -/
theorem peregrine_falcon_diving_time :
  diving_distance / (peregrine_falcon_speed / 3600) = 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_peregrine_falcon_diving_time_l1224_122409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marie_speed_is_12_l1224_122487

/-- Marie's biking speed -/
noncomputable def marie_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

/-- Theorem: Marie's speed is 12 miles per hour -/
theorem marie_speed_is_12 : marie_speed 372 31 = 12 := by
  -- Unfold the definition of marie_speed
  unfold marie_speed
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marie_speed_is_12_l1224_122487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_tube_volume_difference_l1224_122420

noncomputable section

/-- The volume of a cylindrical tube formed by rolling a rectangular sheet of paper. -/
def tube_volume (width : ℝ) (height : ℝ) : ℝ :=
  (width ^ 2 * height) / (4 * Real.pi)

/-- The positive difference in volumes of two cylindrical tubes multiplied by π. -/
def volume_difference (width1 height1 width2 height2 : ℝ) : ℝ :=
  Real.pi * |tube_volume width1 height1 - tube_volume width2 height2|

theorem paper_tube_volume_difference :
  volume_difference 7 10 12 9 = 262.75 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_tube_volume_difference_l1224_122420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_pyramid_ABGF_l1224_122451

-- Define the cube
def cube_ABCDEFGH : Set (Fin 3 → ℝ) :=
  {p | ∀ i, 0 ≤ p i ∧ p i ≤ 2}

-- Define points A, B, G, F
def A : Fin 3 → ℝ := λ i => if i = 0 then 0 else if i = 1 then 0 else 0
def B : Fin 3 → ℝ := λ i => if i = 0 then 2 else if i = 1 then 0 else 0
def G : Fin 3 → ℝ := λ i => if i = 0 then 2 else if i = 1 then 2 else 2
def F : Fin 3 → ℝ := λ i => if i = 0 then 0 else if i = 1 then 2 else 2

-- Define the pyramid ABGF
def pyramid_ABGF : Set (Fin 3 → ℝ) :=
  {p | ∃ t₁ t₂ t₃ t₄ : ℝ, t₁ ≥ 0 ∧ t₂ ≥ 0 ∧ t₃ ≥ 0 ∧ t₄ ≥ 0 ∧
    t₁ + t₂ + t₃ + t₄ = 1 ∧
    p = λ i => t₁ * (A i) + t₂ * (B i) + t₃ * (G i) + t₄ * (F i)}

-- Define the volume function (as a placeholder)
noncomputable def volume (S : Set (Fin 3 → ℝ)) : ℝ := sorry

-- State the theorem
theorem volume_pyramid_ABGF :
  volume pyramid_ABGF = 4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_pyramid_ABGF_l1224_122451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_head_l1224_122433

-- Define a fair coin
noncomputable def fairCoin : ℚ := 1 / 2

-- Define the number of tosses
def numTosses : ℕ := 3

-- Theorem statement
theorem prob_at_least_one_head (p : ℚ) (n : ℕ) 
  (h_fair : p = fairCoin) 
  (h_tosses : n = numTosses) : 
  1 - (1 - p)^n = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_head_l1224_122433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_closed_path_l1224_122472

/-- Represents a 3x3x3 cube with a 1x1x3 core removed --/
structure Pipe :=
  (surface : Set (Fin 3 × Fin 3 × Fin 3))
  (is_valid_surface : ∀ (x y z : Fin 3), (x, y, z) ∈ surface ↔ (x = 0 ∨ x = 2 ∨ y = 0 ∨ y = 2 ∨ z = 0 ∨ z = 2))

/-- Represents a diagonal on the surface of the Pipe --/
inductive SurfaceDiagonal (p : Pipe)
  | mk : (v1 v2 : Fin 3 × Fin 3 × Fin 3) → v1 ∈ p.surface → v2 ∈ p.surface → SurfaceDiagonal p

/-- Represents a path on the surface of the Pipe --/
inductive SurfacePath (p : Pipe)
  | nil : SurfacePath p
  | cons : SurfaceDiagonal p → SurfacePath p → SurfacePath p

/-- Predicate for a closed path --/
def is_closed (p : Pipe) (path : SurfacePath p) : Prop :=
  sorry

/-- Predicate for a path that visits each vertex exactly once --/
def visits_each_vertex_once (p : Pipe) (path : SurfacePath p) : Prop :=
  sorry

/-- Predicate for a closed path that visits each vertex exactly once --/
def is_valid_closed_path (p : Pipe) (path : SurfacePath p) : Prop :=
  is_closed p path ∧ visits_each_vertex_once p path

theorem no_valid_closed_path (p : Pipe) :
  ¬∃ (path : SurfacePath p), is_valid_closed_path p path :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_closed_path_l1224_122472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daps_equivalent_to_42_dips_l1224_122464

/-- Represents the number of units of a given type -/
@[ext]
structure MyUnits (α : Type) where
  count : ℕ

/-- Represents an equivalence between two different types of units -/
structure MyEquivalence (α β : Type) where
  left : MyUnits α
  right : MyUnits β

variable (dap dop dip : Type)

/-- 4 daps are equivalent to 3 dops -/
def daps_to_dops : MyEquivalence dap dop :=
  ⟨⟨4⟩, ⟨3⟩⟩

/-- 2 dops are equivalent to 7 dips -/
def dops_to_dips : MyEquivalence dop dip :=
  ⟨⟨2⟩, ⟨7⟩⟩

/-- The number of daps equivalent to 42 dips is 16 -/
theorem daps_equivalent_to_42_dips :
  ∃ (eq : MyEquivalence dap dip), eq.left.count = 16 ∧ eq.right.count = 42 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_daps_equivalent_to_42_dips_l1224_122464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1224_122431

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (2*x - 4)) / (Real.sqrt (9 - 3*x) + Real.sqrt (x - 1))

theorem domain_of_f :
  ∀ x : ℝ, (2 ≤ x ∧ x ≤ 3) ↔ 
    (2*x - 4 ≥ 0 ∧ 9 - 3*x ≥ 0 ∧ x - 1 ≥ 0 ∧ 
     Real.sqrt (9 - 3*x) + Real.sqrt (x - 1) ≠ 0 ∧
     f x = f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1224_122431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exceptional_points_on_circle_l1224_122416

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in a plane -/
def Point : Type := ℝ × ℝ

/-- Checks if two circles are non-overlapping and mutually external -/
def are_external (c1 c2 : Circle) : Prop := sorry

/-- Checks if a point is outside a circle -/
def is_outside (p : Point) (c : Circle) : Prop := sorry

/-- Checks if a line through two points is tangent to a circle -/
def is_tangent (p1 p2 : Point) (c : Circle) : Prop := sorry

/-- Checks if three lines defined by pairs of points are concurrent -/
def are_concurrent (p1 p2 p3 p4 p5 p6 : Point) : Prop := sorry

/-- Defines membership of a point in a circle -/
def point_in_circle (p : Point) (c : Circle) : Prop := sorry

/-- Theorem: All exceptional points lie on the same circle -/
theorem exceptional_points_on_circle 
  (Γ₁ Γ₂ Γ₃ : Circle)
  (h_external : are_external Γ₁ Γ₂ ∧ are_external Γ₂ Γ₃ ∧ are_external Γ₁ Γ₃) :
  ∃ (Γ : Circle),
    ∀ (P : Point),
      (is_outside P Γ₁ ∧ is_outside P Γ₂ ∧ is_outside P Γ₃) →
      (∃ (A₁ B₁ A₂ B₂ A₃ B₃ : Point),
        (is_tangent P A₁ Γ₁ ∧ is_tangent P B₁ Γ₁) ∧
        (is_tangent P A₂ Γ₂ ∧ is_tangent P B₂ Γ₂) ∧
        (is_tangent P A₃ Γ₃ ∧ is_tangent P B₃ Γ₃) ∧
        are_concurrent A₁ B₁ A₂ B₂ A₃ B₃) →
      point_in_circle P Γ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exceptional_points_on_circle_l1224_122416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_probabilities_expected_value_is_one_l1224_122422

/-- Definition of the random variable X --/
def X : ℝ → ℝ := sorry

/-- Probability mass function for X --/
noncomputable def P (x : ℝ) : ℝ :=
  if x = 0 then 8/27
  else if x = 1 then 4/9
  else if x = 2 then 2/9
  else if x = 3 then 1/27
  else 0

/-- Theorem stating that the sum of probabilities equals 1 --/
theorem sum_of_probabilities : P 0 + P 1 + P 2 + P 3 = 1 := by sorry

/-- Definition of expected value for discrete random variable --/
noncomputable def expected_value : ℝ := 0 * P 0 + 1 * P 1 + 2 * P 2 + 3 * P 3

/-- Main theorem: The expected value of X is 1 --/
theorem expected_value_is_one : expected_value = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_probabilities_expected_value_is_one_l1224_122422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_side_length_l1224_122468

-- Define the quadrilateral ADCB
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the lengths of the sides and diagonal
noncomputable def AD : ℝ := 6
noncomputable def DC : ℝ := 11
noncomputable def CB : ℝ := 6
noncomputable def AC : ℝ := 14

-- Define the length of side BA as x
noncomputable def BA : ℝ := Real.sqrt 557

-- Theorem statement
theorem quadrilateral_side_length : BA = Real.sqrt 557 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_side_length_l1224_122468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_angle_C_l1224_122483

noncomputable section

open Real

def m (x : ℝ) : ℝ × ℝ := (2 * sin x, 1)
def n (x : ℝ) : ℝ × ℝ := (sqrt 3 * cos x, 2 * (cos x)^2)

def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi ∧
  a / sin A = b / sin B ∧ b / sin B = c / sin C

theorem f_range_and_angle_C :
  (∀ x ∈ Set.Icc 0 (Real.pi/2), 0 ≤ f x ∧ f x ≤ 3) ∧
  ∀ A B C : ℝ, triangle_ABC A B C 1 (sqrt 3) c →
    f A = 3 → C = Real.pi/2 ∨ C = Real.pi/6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_angle_C_l1224_122483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_trig_expression_l1224_122478

open Real

-- Define the angle θ
variable (θ : ℝ)

-- Define the condition that the terminal side of θ passes through (4, -3)
def terminal_point : ℝ × ℝ := (4, -3)

-- State the theorems to be proved
theorem tan_value : Real.tan θ = -3/4 := by sorry

theorem trig_expression : 
  (Real.sin (θ + π/2) + Real.cos θ) / (Real.sin θ - Real.cos (θ - π)) = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_trig_expression_l1224_122478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_area_circular_garden_l1224_122452

/-- The area of a circle with diameter 10 meters -/
noncomputable def circle_area (π : ℝ) : ℝ :=
  π * (5 ^ 2)

/-- Half the area of a circle with diameter 10 meters -/
noncomputable def half_circle_area (π : ℝ) : ℝ :=
  circle_area π / 2

/-- Theorem: Half the area of a circular garden with diameter 10 meters is 12.5π square meters -/
theorem half_area_circular_garden (π : ℝ) :
  half_circle_area π = 12.5 * π := by
  unfold half_circle_area circle_area
  simp [mul_div_cancel]
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_area_circular_garden_l1224_122452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_absolute_values_l1224_122421

noncomputable def x : ℝ := 1 + Real.sqrt 3 / (1 + Real.sqrt 3 / (1 + Real.sqrt 3 / (1 + Real.sqrt 3 / (1 + Real.sqrt 3 / (1 + Real.sqrt 3)))))

noncomputable def inverse_expression (x : ℝ) : ℝ := 1 / ((x + 1) * (x - 2))

def simplified_form (A B C : ℤ) : Prop :=
  inverse_expression x = (A + Real.sqrt B : ℝ) / C

theorem sum_of_absolute_values : ∃ A B C : ℤ, simplified_form A B C ∧ |A| + |B| + |C| = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_absolute_values_l1224_122421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l1224_122400

theorem function_property (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (f x) = x^2 * f x - x + 1) : 
  f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l1224_122400
