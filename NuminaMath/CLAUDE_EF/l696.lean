import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l696_69682

def a : ℝ × ℝ := (-4, 3)
def b (x y : ℝ) : ℝ × ℝ := (2*x, y)
def c (x y : ℝ) : ℝ × ℝ := (x+y, 1)

theorem vector_problem (x y : ℝ) :
  (∃ (k : ℝ), b x y = k • a) ∧  -- a is parallel to b
  (a.1 * (c x y).1 + a.2 * (c x y).2 = 0) →  -- a is perpendicular to c
  x = -3/2 ∧ y = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l696_69682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l696_69663

/-- Given a hyperbola C with the equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    one asymptote y = √3x, and right focus F(2, 0).
    A line l passes through F and intersects the right branch of C at P and Q.
    Point M satisfies FP = QM. -/
theorem hyperbola_properties (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (C : Set (ℝ × ℝ)) (F P Q M : ℝ × ℝ) (l : Set (ℝ × ℝ)) :
  C = {(x, y) | x^2 / a^2 - y^2 / b^2 = 1} →
  (∃ x, (x, Real.sqrt 3 * x) ∈ C) →
  F = (2, 0) →
  l ∈ Set.powerset C →
  F ∈ l ∧ P ∈ l ∧ Q ∈ l →
  P.1 > 0 ∧ Q.1 > 0 →
  (P.1 - F.1, P.2 - F.2) = (M.1 - Q.1, M.2 - Q.2) →
  (∃ E₁ E₂ : ℝ × ℝ, 
    (C = {(x, y) | x^2 - y^2 / 3 = 1}) ∧
    (E₁ = (-4, 0) ∧ E₂ = (4, 0) ∧
     |M.1 - E₁.1| - |M.1 - E₂.1| = 4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l696_69663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bromine_atomic_weight_l696_69649

/-- The atomic weight of nitrogen in g/mol -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The molecular weight of the compound in g/mol -/
def compound_weight : ℝ := 98

/-- The number of nitrogen atoms in the compound -/
def nitrogen_count : ℕ := 1

/-- The number of hydrogen atoms in the compound -/
def hydrogen_count : ℕ := 4

/-- The number of bromine atoms in the compound -/
def bromine_count : ℕ := 1

/-- The atomic weight of bromine in g/mol -/
def bromine_weight : ℝ := compound_weight - (nitrogen_count * nitrogen_weight + hydrogen_count * hydrogen_weight)

theorem bromine_atomic_weight :
  ∃ ε > 0, |bromine_weight - 79.958| < ε := by
  -- The proof is omitted for brevity
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bromine_atomic_weight_l696_69649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l696_69607

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) lies on the line -/
noncomputable def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- The y-intercept of a line -/
noncomputable def Line.yIntercept (l : Line) : ℝ :=
  -l.c / l.b

theorem line_equation_proof (l : Line) : 
  (l.a = 5 ∧ l.b = 4 ∧ l.c = -8) → 
  (l.contains 4 (-3) ∧ l.yIntercept = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l696_69607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_c_all_reals_l696_69604

noncomputable def c (m : ℝ) (x : ℝ) : ℝ := (3 * m * x^2 + m * x - 4) / (m * x^2 - 3 * x + 2 * m)

theorem domain_c_all_reals (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, c m x = y) ↔ 
  (m < -3 * Real.sqrt 2 / 4 ∨ m > 3 * Real.sqrt 2 / 4) := by
  sorry

#check domain_c_all_reals

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_c_all_reals_l696_69604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_people_round_table_l696_69654

def round_table_arrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

theorem six_people_round_table :
  round_table_arrangements 6 = 120 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_people_round_table_l696_69654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_same_heads_value_l696_69642

/-- Represents the outcome of flipping a coin -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents a coin with its probability of getting heads -/
structure Coin where
  prob_heads : ℚ
  prob_tails : ℚ
  prob_sum_one : prob_heads + prob_tails = 1

/-- A fair coin has equal probability of heads and tails -/
def fair_coin : Coin where
  prob_heads := 1/2
  prob_tails := 1/2
  prob_sum_one := by norm_num

/-- A biased coin with 3/5 probability of heads -/
def biased_coin : Coin where
  prob_heads := 3/5
  prob_tails := 2/5
  prob_sum_one := by norm_num

/-- The set of coins each person has -/
def person_coins : List Coin := [fair_coin, fair_coin, biased_coin]

/-- The probability of getting a specific number of heads when flipping the coins -/
noncomputable def prob_heads (n : ℕ) (coins : List Coin) : ℚ :=
  sorry

/-- The probability of two people getting the same number of heads -/
noncomputable def prob_same_heads : ℚ :=
  sorry

/-- Theorem stating the probability of two people getting the same number of heads -/
theorem prob_same_heads_value : prob_same_heads = 63/200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_same_heads_value_l696_69642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l696_69681

theorem sin_alpha_value (α : Real) (h1 : 0 < α ∧ α < π/2) (h2 : Real.cos (α + π/6) = 2/3) :
  Real.sin α = (Real.sqrt 15 - 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l696_69681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trinomial_expansion_constant_term_l696_69665

-- Define the trinomial
noncomputable def trinomial (x : ℝ) : ℝ := x + 1 / (2 * x)

-- Define the constant term of the expansion
noncomputable def constant_term : ℝ := (1 / 2)^3 * (Nat.choose 6 3)

-- Theorem statement
theorem trinomial_expansion_constant_term :
  constant_term = 5/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trinomial_expansion_constant_term_l696_69665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l696_69630

noncomputable section

-- Define the function f
def f (a m : ℝ) (x : ℝ) : ℝ := m + (Real.log x) / (Real.log a)

-- Define the function g
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := 2 * f x - f (x - 1)

-- State the theorem
theorem function_properties (a m : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  f a m 8 = 2 ∧ f a m 1 = -1 →
  (∀ x, f a m x = -1 + Real.log x / Real.log 2) ∧
  (∃ x, g (f a m) x = 1 ∧ ∀ y, g (f a m) y ≥ g (f a m) x) ∧
  (g (f a m) 2 = 1 ∧ ∀ y, g (f a m) y ≥ 1) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l696_69630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_in_first_quadrant_implies_k_range_l696_69608

/-- The intersection point of two lines in the first quadrant -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ
  first_quadrant : x > 0 ∧ y > 0

/-- Definition of line l₁ -/
noncomputable def l₁ (k : ℝ) (x : ℝ) : ℝ := 2 * x - 5 * k + 7

/-- Definition of line l₂ -/
noncomputable def l₂ (x : ℝ) : ℝ := -1/2 * x + 2

/-- Theorem stating the range of k given the intersection point is in the first quadrant -/
theorem intersection_in_first_quadrant_implies_k_range 
  (k : ℝ) 
  (p : IntersectionPoint) 
  (h₁ : l₁ k p.x = p.y) 
  (h₂ : l₂ p.x = p.y) : 
  1 < k ∧ k < 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_in_first_quadrant_implies_k_range_l696_69608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identical_cones_apex_angle_l696_69635

/-- Given three identical cones with vertex A touching each other externally, 
    and a fourth cone with vertex A and apex angle 2π/3 touching the other three internally,
    prove that the apex angle of the identical cones is 2 * arctan((3 / (4 + √3))). -/
theorem identical_cones_apex_angle : 
  ∀ (α : ℝ), 
  (∃ (r R : ℝ), 
    r > 0 ∧ R > 0 ∧
    (2 * r / Real.sqrt 3 = (R - r) / 2) ∧
    (R = r * (Real.tan (π / 3)) / (Real.tan α))) →
  2 * α = 2 * Real.arctan ((3 : ℝ) / (4 + Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identical_cones_apex_angle_l696_69635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hired_candidate_is_6_l696_69656

def Candidate := Fin 6

structure Prediction where
  person : Fin 4
  content : Candidate → Bool

def predictions : List Prediction := [
  ⟨0, fun c => c.val ≠ 5⟩,
  ⟨1, fun c => c.val = 3 ∨ c.val = 4⟩,
  ⟨2, fun c => c.val = 0 ∨ c.val = 1 ∨ c.val = 2⟩,
  ⟨3, fun c => c.val ≠ 0 ∧ c.val ≠ 1 ∧ c.val ≠ 2⟩
]

def correctPredictions (hired : Candidate) : List Prediction :=
  predictions.filter (fun p => p.content hired)

theorem hired_candidate_is_6 :
  ∃! hired : Candidate, (correctPredictions hired).length = 1 ∧ hired.val = 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hired_candidate_is_6_l696_69656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_max_k_condition_l696_69625

noncomputable def f (x : ℝ) : ℝ := 5 + Real.log x

noncomputable def g (k : ℝ) (x : ℝ) : ℝ := (k * x) / (x + 1)

theorem tangent_line_condition (k : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ (deriv (g k)) x₀ = 1 ∧ g k x₀ = x₀ + 4) ↔ (k = 1 ∨ k = 9) :=
sorry

theorem max_k_condition : 
  (∃ k : ℕ, k > 0 ∧ ∀ k' : ℕ, k' > k → ∃ x : ℝ, x > 1 ∧ f x ≤ g k' x) ∧
  (∀ x : ℝ, x > 1 → f x > g 7 x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_max_k_condition_l696_69625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_diet_start_time_l696_69637

/-- Represents the weight loss problem for Joe --/
structure WeightLossProblem where
  initial_weight : ℝ
  current_weight : ℝ
  future_weight : ℝ
  future_months : ℝ
  weight_loss_rate : ℝ

/-- Calculates the number of months since Joe started his diet --/
noncomputable def months_since_diet_start (p : WeightLossProblem) : ℝ :=
  (p.initial_weight - p.current_weight) / p.weight_loss_rate

/-- Theorem stating that Joe started his diet 3 months ago --/
theorem joe_diet_start_time (p : WeightLossProblem) 
  (h1 : p.initial_weight = 222)
  (h2 : p.current_weight = 198)
  (h3 : p.future_weight = 170)
  (h4 : p.future_months = 3.5)
  (h5 : p.weight_loss_rate = (p.current_weight - p.future_weight) / p.future_months)
  : months_since_diet_start p = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_diet_start_time_l696_69637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l696_69660

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)

-- State the theorem
theorem monotonic_increasing_interval_of_f :
  ∃ (a b : ℝ), a = 0 ∧ b = Real.pi / 8 ∧
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ Real.pi / 2 →
    (x ≤ b → f x ≤ f y) ∧
    (b < x → ¬(f x ≤ f y))) := by
  sorry

#check monotonic_increasing_interval_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l696_69660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_comparison_l696_69629

noncomputable section

variable (f : ℝ → ℝ)

axiom symmetry : ∀ x : ℝ, f (3 - x) = f x
axiom derivative_condition : ∀ x : ℝ, (x - 3/2) * (deriv f x) < 0

theorem function_comparison (x₁ x₂ : ℝ) (h1 : x₁ < x₂) (h2 : x₁ + x₂ > 3) :
  f x₁ > f x₂ := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_comparison_l696_69629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_transformation_l696_69661

/-- The original parabola function -/
noncomputable def original_parabola (x : ℝ) : ℝ := (1/4) * x^2

/-- The transformation that shifts a function to the left by 2 units -/
def shift_left (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x + 2)

/-- The transformation that shifts a function down by 3 units -/
def shift_down (f : ℝ → ℝ) (x : ℝ) : ℝ := f x - 3

/-- The resulting parabola after both transformations -/
noncomputable def transformed_parabola (x : ℝ) : ℝ := (1/4) * (x + 2)^2 - 3

theorem parabola_transformation :
  ∀ x : ℝ, (shift_down (shift_left original_parabola)) x = transformed_parabola x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_transformation_l696_69661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l696_69691

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence definition
  q > 0 →  -- common ratio is positive
  a 3 * a 9 = 2 * (a 5)^2 →  -- given condition
  a 2 = 1 →  -- given condition
  a 1 = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l696_69691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_A_production_rate_l696_69603

-- Define the number of sprockets produced by each machine
def total_sprockets : ℕ := 440

-- Define the production rates of Machine A and Machine B
variable (rate_A : ℚ)
variable (rate_B : ℚ)

-- Define the time taken by each machine
variable (time_A : ℚ)
variable (time_B : ℚ)

-- State the conditions
axiom time_difference : time_A = time_B + 10
axiom production_rate_relation : rate_B = 1.1 * rate_A
axiom total_production_A : total_sprockets = rate_A * time_A
axiom total_production_B : total_sprockets = rate_B * time_B

-- State the theorem to be proved
theorem machine_A_production_rate : rate_A = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_A_production_rate_l696_69603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosAOB_specific_rectangle_l696_69688

/-- A rectangle with given side lengths -/
structure Rectangle where
  AB : ℝ
  BC : ℝ

/-- The cosine of the angle AOB in a rectangle -/
noncomputable def cosAOB (r : Rectangle) : ℝ :=
  r.AB / (Real.sqrt (r.AB ^ 2 + r.BC ^ 2))

/-- Theorem: For a rectangle with AB = 15 and BC = 35, cos∠AOB = (15√1450) / 1450 -/
theorem cosAOB_specific_rectangle :
  let r := Rectangle.mk 15 35
  cosAOB r = (15 * Real.sqrt 1450) / 1450 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosAOB_specific_rectangle_l696_69688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_catchup_time_is_seven_l696_69685

/-- Represents the problem of a car catching up to a truck --/
structure CatchUpProblem where
  truckSpeed : ℚ
  carInitialSpeed : ℚ
  carSpeedIncrease : ℚ
  initialDistance : ℚ

/-- Calculates the time it takes for the car to catch up with the truck --/
noncomputable def catchUpTime (p : CatchUpProblem) : ℚ :=
  let a := p.carSpeedIncrease / 2
  let b := p.carInitialSpeed - p.truckSpeed + p.carSpeedIncrease / 2
  let c := -p.initialDistance
  ((-b + (b^2 - 4*a*c).sqrt) / (2*a))

/-- The main theorem stating that given the problem conditions, 
    the catch-up time is 7 hours --/
theorem catchup_time_is_seven : 
  let problem := CatchUpProblem.mk 40 50 5 175
  catchUpTime problem = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_catchup_time_is_seven_l696_69685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_l696_69610

theorem log_equation (a : ℝ) (h : (3 : ℝ)^a = 2) : 
  2 * Real.log 6 / Real.log 3 - Real.log 8 / Real.log 3 = 2 - a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_l696_69610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_satisfies_equation_l696_69641

/-- The perpendicular bisector of a line segment AB intersects AB at its midpoint -/
axiom perpendicular_bisector_intersects_at_midpoint 
  (A B : ℝ × ℝ) : 
  let C := ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)
  C = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

/-- Given points A and B, prove that the x and y coordinates of their midpoint 
    satisfy the equation 2x - 4y = -8 -/
theorem midpoint_satisfies_equation 
  (A B : ℝ × ℝ) 
  (hA : A = (20, 10)) 
  (hB : B = (4, 6)) : 
  let C := ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)
  2 * C.fst - 4 * C.snd = -8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_satisfies_equation_l696_69641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_is_j_l696_69689

noncomputable def h (x : ℝ) : ℝ := 7 + 3 * x

noncomputable def j (x : ℝ) : ℝ := (x - 7) / 3

theorem h_inverse_is_j : Function.LeftInverse j h ∧ Function.RightInverse j h := by
  constructor
  · -- Prove left inverse
    intro x
    simp [h, j]
    field_simp
    ring
  · -- Prove right inverse
    intro x
    simp [h, j]
    field_simp
    ring

#check h_inverse_is_j

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_is_j_l696_69689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_triangle_area_l696_69648

/-- The side length of each hexagon -/
noncomputable def hexagon_side_length : ℝ := Real.sqrt 2

/-- The number of surrounding hexagons -/
def num_surrounding_hexagons : ℕ := 6

/-- The number of hexagons selected to form the triangle -/
def num_selected_hexagons : ℕ := 3

/-- Represents a regular hexagon -/
structure RegularHexagon where
  side_length : ℝ
  center : ℝ × ℝ

/-- Represents the configuration of hexagons -/
structure HexagonConfiguration where
  central_hexagon : RegularHexagon
  surrounding_hexagons : Finset RegularHexagon

/-- Represents a triangle formed by the centers of three hexagons -/
structure CenterTriangle where
  vertices : Finset (ℝ × ℝ)

/-- Function to calculate the area of a triangle given its side length -/
noncomputable def triangle_area (side_length : ℝ) : ℝ :=
  (Real.sqrt 3 / 4) * side_length^2

/-- Theorem stating that the area of the triangle formed by the centers
    of three randomly selected surrounding hexagons is 2√3 -/
theorem center_triangle_area
  (config : HexagonConfiguration)
  (selected_centers : Finset (ℝ × ℝ))
  (h1 : config.surrounding_hexagons.card = num_surrounding_hexagons)
  (h2 : ∀ h ∈ config.surrounding_hexagons, h.side_length = hexagon_side_length)
  (h3 : selected_centers.card = num_selected_hexagons)
  (h4 : ∀ c ∈ selected_centers, ∃ h ∈ config.surrounding_hexagons, h.center = c) :
  triangle_area (2 * hexagon_side_length) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_triangle_area_l696_69648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_transformation_l696_69621

-- Define the function g as noncomputable due to the use of Real.sqrt
noncomputable def g : ℝ → ℝ
| x => if x ≥ -2 ∧ x ≤ 1 then 2 - x
       else if x > 1 ∧ x ≤ 3 then Real.sqrt (4 - (x - 1)^2)
       else if x > 3 ∧ x ≤ 5 then x - 3
       else 0  -- For values outside the defined range

-- State the theorem
theorem graph_transformation (x y : ℝ) :
  y = g (x - 1) + 3 ↔ y - 3 = g (x - 1) :=
by
  -- The proof is skipped using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_transformation_l696_69621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_daily_profit_l696_69612

/-- Represents the daily profit function for greeting cards -/
noncomputable def W (x : ℝ) : ℝ := 60 - 120 / x

/-- Theorem stating the maximum daily profit under given conditions -/
theorem max_daily_profit :
  ∀ x : ℝ, x > 0 → x ≤ 10 → W x ≤ 48 ∧ W 10 = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_daily_profit_l696_69612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l696_69659

def A : Set ℝ := {0, 1}
def B (a : ℝ) : Set ℝ := {a^2, 2*a}

def sum_sets (A B : Set ℝ) : Set ℝ :=
  {x | ∃ (x₁ : ℝ) (x₂ : ℝ), x₁ ∈ A ∧ x₂ ∈ B ∧ x = x₁ + x₂}

theorem a_range (a : ℝ) : 
  (∀ x ∈ sum_sets A (B a), x ≤ 2*a + 1) ∧ 
  (2*a + 1 ∈ sum_sets A (B a)) → 
  0 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l696_69659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_same_initial_letter_l696_69673

/-- Represents a digit (0-9) --/
inductive Digit : Type
| zero | one | two | three | four | five | six | seven | eight | nine

/-- Returns true if the Russian word for the digit starts with 's' --/
def startsWithS : Digit → Bool
  | Digit.seven => true
  | _ => false

/-- Converts a Digit to its numerical value --/
def digitToNat : Digit → Nat
  | Digit.zero => 0
  | Digit.one => 1
  | Digit.two => 2
  | Digit.three => 3
  | Digit.four => 4
  | Digit.five => 5
  | Digit.six => 6
  | Digit.seven => 7
  | Digit.eight => 8
  | Digit.nine => 9

/-- Checks if three digits form a valid three-digit number in ascending order --/
def isValidNumber (a b c : Digit) : Prop :=
  digitToNat a < digitToNat b ∧ digitToNat b < digitToNat c

/-- The main theorem --/
theorem unique_number_with_same_initial_letter :
  ∀ a b c : Digit,
    isValidNumber a b c ∧
    (startsWithS a ∧ startsWithS b ∧ startsWithS c) →
    digitToNat a = 1 ∧ digitToNat b = 4 ∧ digitToNat c = 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_same_initial_letter_l696_69673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_in_list_l696_69699

/-- A list of 12 consecutive integers -/
def ConsecutiveIntegers (start : ℤ) : List ℤ :=
  List.range 12 |>.map (fun i => start + i)

/-- The range of a list of integers -/
def Range (l : List ℤ) : ℤ :=
  match l.maximum, l.minimum with
  | some max, some min => max - min
  | _, _ => 0

theorem least_integer_in_list (start : ℤ) :
  let K := ConsecutiveIntegers start
  Range (K.filter (· > 0)) = 7 →
  K.minimum? = some (-1) := by
  sorry

#eval ConsecutiveIntegers (-1)
#eval Range (ConsecutiveIntegers (-1))
#eval Range ((ConsecutiveIntegers (-1)).filter (· > 0))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_in_list_l696_69699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_larger_square_l696_69640

/-- Represents the weight of a square piece of wood -/
noncomputable def weight (side_length : ℝ) : ℝ :=
  16 * (side_length / 4) ^ 2

/-- Theorem stating that a square piece of wood with side length 6 inches weighs 36 ounces -/
theorem weight_of_larger_square :
  weight 6 = 36 :=
by
  -- Unfold the definition of weight
  unfold weight
  -- Simplify the expression
  simp [pow_two]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_larger_square_l696_69640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_integer_in_sequence_one_million_is_last_integer_l696_69666

def sequenceTermQ (n : ℕ) : ℚ :=
  (1000000 : ℚ) / (3 ^ n)

def sequenceTermInt (n : ℕ) : ℤ :=
  (1000000 : ℤ) / (3 ^ n)

theorem last_integer_in_sequence :
  ∀ n : ℕ, n > 0 → ¬ (sequenceTermQ n).isInt :=
by
  sorry

theorem one_million_is_last_integer :
  (sequenceTermInt 0 = 1000000) ∧ ∀ n : ℕ, n > 0 → ¬ (sequenceTermQ n).isInt :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_integer_in_sequence_one_million_is_last_integer_l696_69666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perfect_number_l696_69693

def is_proper_divisor (d n : ℕ) : Prop := d ∣ n ∧ d < n

noncomputable def sum_proper_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ d => d ∣ n ∧ d < n) (Finset.range n)).sum id

def is_perfect (n : ℕ) : Prop := n > 0 ∧ sum_proper_divisors n = n

theorem smallest_perfect_number :
  ∃ n : ℕ, is_perfect n ∧ ∀ m : ℕ, is_perfect m → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perfect_number_l696_69693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l696_69647

theorem product_remainder (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a % 7 = 2 →
  b % 7 = 3 →
  c % 7 = 5 →
  (a + b + c) % 7 = 3 →
  (a * b * c) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l696_69647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l696_69626

/-- A power function that passes through the point (√2, 2√2) -/
noncomputable def f (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

theorem power_function_through_point (α : ℝ) :
  f α (Real.sqrt 2) = 2 * Real.sqrt 2 → f α 2 = 8 := by
  intro h
  -- The proof steps would go here
  sorry

#check power_function_through_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l696_69626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stop_walk_time_l696_69690

/-- The time in minutes to walk to the bus stop at the usual speed -/
noncomputable def usual_time : ℝ := sorry

/-- The time in minutes to walk to the bus stop at 4/5 of the usual speed -/
noncomputable def slower_time : ℝ := usual_time + 10

/-- The ratio of the usual speed to the slower speed -/
noncomputable def speed_ratio : ℝ := 5 / 4

theorem bus_stop_walk_time : usual_time = 40 := by
  have h1 : speed_ratio = slower_time / usual_time := by sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stop_walk_time_l696_69690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l696_69662

noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.sin (x + Real.pi/3) - Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3 / 4

theorem f_properties :
  -- The smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  -- The maximum value in [-π/4, π/4] is 1/4
  (∀ (x : ℝ), -Real.pi/4 ≤ x ∧ x ≤ Real.pi/4 → f x ≤ 1/4) ∧
  (∃ (x : ℝ), -Real.pi/4 ≤ x ∧ x ≤ Real.pi/4 ∧ f x = 1/4) ∧
  -- The minimum value in [-π/4, π/4] is -1/2
  (∀ (x : ℝ), -Real.pi/4 ≤ x ∧ x ≤ Real.pi/4 → f x ≥ -1/2) ∧
  (∃ (x : ℝ), -Real.pi/4 ≤ x ∧ x ≤ Real.pi/4 ∧ f x = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l696_69662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pressure_increases_with_submerged_block_l696_69644

/-- The pressure at the bottom of a container containing water -/
noncomputable def pressure_at_bottom (h : ℝ) : ℝ := sorry

/-- The height of the water column in the container -/
noncomputable def water_height : ℝ := sorry

/-- The volume of water displaced by the submerged wooden block -/
noncomputable def displaced_volume : ℝ := sorry

/-- Assertion that the displaced volume is positive -/
axiom displaced_volume_positive : displaced_volume > 0

/-- The new height of the water column after the block is submerged -/
noncomputable def new_water_height : ℝ := water_height + displaced_volume / sorry

/-- Assertion that the pressure depends only on the water height -/
axiom pressure_depends_on_height :
  ∀ h₁ h₂ : ℝ, h₁ > h₂ → pressure_at_bottom h₁ > pressure_at_bottom h₂

/-- Theorem: The pressure at the bottom increases when a wooden block is submerged -/
theorem pressure_increases_with_submerged_block :
  pressure_at_bottom new_water_height > pressure_at_bottom water_height := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pressure_increases_with_submerged_block_l696_69644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_terms_are_integers_l696_69645

def a : ℕ → ℚ
  | 0 => 1
  | 1 => 2
  | (n + 2) => ((6 * n + 9) * a (n + 1) - n * a n) / (n + 3)

theorem all_terms_are_integers :
  ∀ n : ℕ, ∃ m : ℤ, a n = m :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_terms_are_integers_l696_69645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l696_69655

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := y^2 / 2 + x^2 = 1

/-- A line passing through the foci of the ellipse -/
structure FocalLine where
  k : ℝ
  eq : ℝ → ℝ → Prop
  eq_def : ∀ x y, eq x y ↔ y = k * x + 1

/-- Points of intersection between the focal line and the ellipse -/
structure IntersectionPoints (l : FocalLine) where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  P_on_ellipse : ellipse P.1 P.2
  Q_on_ellipse : ellipse Q.1 Q.2
  P_on_line : l.eq P.1 P.2
  Q_on_line : l.eq Q.1 Q.2

/-- The point M on the x-axis -/
noncomputable def M (l : FocalLine) : ℝ × ℝ := (l.k / (l.k^2 + 2), 0)

/-- The area of triangle MPQ -/
noncomputable def triangleArea (l : FocalLine) (pts : IntersectionPoints l) : ℝ :=
  let P := pts.P
  let Q := pts.Q
  let M := M l
  -- Area calculation (placeholder)
  0

/-- The theorem stating the maximum area of triangle MPQ -/
theorem max_triangle_area :
  ∃ (l : FocalLine) (pts : IntersectionPoints l),
    ∀ (l' : FocalLine) (pts' : IntersectionPoints l'),
      triangleArea l pts ≥ triangleArea l' pts' ∧
      triangleArea l pts = 3 * Real.sqrt 6 / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l696_69655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_sum_l696_69627

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x^2 - 1

-- State the theorem
theorem f_derivative_sum : 
  (deriv f 1) + (deriv (λ _ : ℝ ↦ f 1) 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_sum_l696_69627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lateral_surface_area_equal_volume_ratio_l696_69652

/-- Cone with base radius R and height H -/
structure Cone where
  R : ℝ
  H : ℝ

/-- Inscribed cylinder with height x -/
structure InscribedCylinder (cone : Cone) where
  x : ℝ

/-- Theorem for the maximum lateral surface area of an inscribed cylinder -/
theorem max_lateral_surface_area (cone : Cone) :
  ∃ (cyl : InscribedCylinder cone),
    ∀ (other : InscribedCylinder cone),
      2 * Real.pi * cyl.x * (cone.R - (cone.R / cone.H) * cyl.x) ≥
      2 * Real.pi * other.x * (cone.R - (cone.R / cone.H) * other.x) ∧
      2 * Real.pi * cyl.x * (cone.R - (cone.R / cone.H) * cyl.x) = 
      (1/2) * Real.pi * cone.R * cone.H := by
  sorry

/-- Theorem for the ratio of heights when the cone is divided into equal volumes -/
theorem equal_volume_ratio (cone : Cone) :
  ∃ (h_small : ℝ) (h_frustum : ℝ),
    h_small + h_frustum = cone.H ∧
    (1/3) * Real.pi * (cone.R * h_small / cone.H)^2 * h_small = 
    (1/3) * Real.pi * cone.R^2 * cone.H - 
    (1/3) * Real.pi * (cone.R * h_small / cone.H)^2 * h_small ∧
    h_small / h_frustum = Real.rpow 4 (1/3) / (2 - Real.rpow 4 (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lateral_surface_area_equal_volume_ratio_l696_69652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_locus_of_circle_line_intersection_l696_69695

/-- A circle in the 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane passing through the origin --/
structure Line where
  slope : ℝ

/-- The locus of points satisfying a condition --/
structure Locus where
  points : Set (ℝ × ℝ)

/-- Given a circle C₁ and a line l passing through the origin,
    prove that the locus of midpoints of the intersections
    satisfies the given equation --/
theorem midpoint_locus_of_circle_line_intersection
  (C₁ : Circle)
  (l : Line)
  (h₁ : C₁.center = (3, 0))
  (h₂ : C₁.radius = 2)
  (h₃ : ∀ x y, x^2 + y^2 - 6*x + 5 = 0 ↔ ((x - 3)^2 + y^2 = 4)) :
  ∃ (C : Locus),
    C.points = {(x, y) | (x - 3/2)^2 + y^2 = 9/4 ∧ 5/3 < x ∧ x ≤ 3} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_locus_of_circle_line_intersection_l696_69695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_20n_l696_69606

noncomputable def mySequence (a : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => (Real.sqrt 3 * mySequence a n + 1) / (Real.sqrt 3 - mySequence a n)

theorem mySequence_20n (a : ℝ) (n : ℕ) :
  mySequence a (20 * n) = (a + Real.sqrt 3) / (1 - Real.sqrt 3 * a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_20n_l696_69606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_in_domain_of_f_of_f_l696_69623

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 5)

-- State the theorem
theorem smallest_x_in_domain_of_f_of_f : 
  ∀ x : ℝ, (∃ y : ℝ, f (f x) = y) → x ≥ 30 := by
  sorry

-- Optional: Add a lemma to break down the proof
lemma f_of_f_defined (x : ℝ) : 
  (∃ y : ℝ, f (f x) = y) ↔ x ≥ 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_in_domain_of_f_of_f_l696_69623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_equality_part2_equality_l696_69697

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Part 1
theorem part1_equality : (lg 2)^2 + lg 2 * lg 5 + lg 5 = 1 := by sorry

-- Part 2
theorem part2_equality : 
  (Real.rpow 2 (1/3 : ℝ) * Real.sqrt 3)^6 - 8 * Real.rpow (16/49) (-1/2) - Real.rpow 2 (1/4) * Real.rpow 8 (1/4) - Real.rpow (-2016) 0 = 91 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_equality_part2_equality_l696_69697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_shortest_chord_l696_69634

-- Define the circle C
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

-- Define the line AB
def line_eq (m x y : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

-- Theorem 1: Line AB always passes through the fixed point (3,1)
theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line_eq m 3 1 := by sorry

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Define the chord length as a function of m
noncomputable def chord_length (m : ℝ) : ℝ :=
  let x1 := (sorry : ℝ)  -- First intersection point x-coordinate
  let y1 := (sorry : ℝ)  -- First intersection point y-coordinate
  let x2 := (sorry : ℝ)  -- Second intersection point x-coordinate
  let y2 := (sorry : ℝ)  -- Second intersection point y-coordinate
  distance x1 y1 x2 y2

-- Theorem 2: The chord is shortest when m = -3/4 and its length is 4√5
theorem shortest_chord :
  ∃ m : ℝ, m = -3/4 ∧ 
  (∀ m' : ℝ, chord_length m ≤ chord_length m') ∧
  chord_length m = 4 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_shortest_chord_l696_69634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motel_rent_theorem_l696_69658

/-- Represents the total rent charged by a motel on a Saturday night. -/
def TotalRent (r40 r60 : ℕ) : ℕ := 40 * r40 + 60 * r60

/-- Represents the new total rent if 10 rooms were switched from $60 to $40. -/
def NewTotalRent (r40 r60 : ℕ) : ℕ := 40 * (r40 + 10) + 60 * (r60 - 10)

/-- The theorem stating that the total rent is $500. -/
theorem motel_rent_theorem (r40 r60 : ℕ) : 
  (NewTotalRent r40 r60 = (TotalRent r40 r60 * 6) / 10) → 
  TotalRent r40 r60 = 500 := by
  sorry

#check motel_rent_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_motel_rent_theorem_l696_69658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l696_69639

/-- The parabola function y = x^3 -/
def parabola (x : ℝ) : ℝ := x^3

/-- The slope of the tangent line to the parabola at x -/
noncomputable def tangent_slope (x : ℝ) : ℝ := 3 * x^2

/-- The slope of the normal line to the parabola at x -/
noncomputable def normal_slope (x : ℝ) : ℝ := -1 / tangent_slope x

/-- The normal line to the parabola at point A (1, 1) -/
noncomputable def normal_line (x : ℝ) : ℝ := normal_slope 1 * (x - 1) + 1

theorem intersection_point :
  let x_B : ℝ := -4/3
  let y_B : ℝ := -64/27
  parabola x_B = y_B ∧ normal_line x_B = y_B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l696_69639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_equals_P_odd_div_P_even_l696_69613

/-- Given a list of natural numbers, compute the product of LCMs of all subsets with odd cardinality -/
def P_odd (a : List ℕ) : ℕ := sorry

/-- Given a list of natural numbers, compute the product of LCMs of all subsets with even cardinality -/
def P_even (a : List ℕ) : ℕ := sorry

/-- The GCD of a list of natural numbers -/
def list_gcd : List ℕ → ℕ
  | [] => 0
  | (x :: xs) => Nat.gcd x (list_gcd xs)

theorem gcd_equals_P_odd_div_P_even (a : List ℕ) (h : a.Pairwise (· ≠ ·)) :
  list_gcd a = P_odd a / P_even a := by
  sorry

#check gcd_equals_P_odd_div_P_even

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_equals_P_odd_div_P_even_l696_69613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_DSO_measure_l696_69696

-- Define the triangle DOG
structure Triangle (D O G : Point) where
  -- We don't need to add specific conditions here for this problem

-- Define the angle measure in degrees
noncomputable def angle_measure (A B C : Point) : ℝ := sorry

-- State the theorem
theorem angle_DSO_measure 
  (D O G S : Point) 
  (tri : Triangle D O G) 
  (h1 : angle_measure D G O = angle_measure D O G)
  (h2 : angle_measure D O G = 40)
  (h3 : angle_measure D O S = angle_measure D O G / 2) :
  angle_measure D S O = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_DSO_measure_l696_69696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_in_interval_l696_69669

open Real

theorem unique_root_in_interval :
  ∃! x : ℝ, x ∈ Set.Ioo (-π/2) 0 ∧ 
    1 / (cos x)^3 - 1 / (sin x)^3 = 4 * Real.sqrt 2 ∧
    x = -π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_in_interval_l696_69669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_tangent_l696_69628

-- Define the triangle DEF
structure Triangle (D E F : ℝ × ℝ) : Prop where
  right_angle_at_E : (E.1 - D.1) * (F.1 - E.1) + (E.2 - D.2) * (F.2 - E.2) = 0
  df_length : (F.1 - D.1)^2 + (F.2 - D.2)^2 = 85
  de_length : (E.1 - D.1)^2 + (E.2 - D.2)^2 = 49

-- Define the circle
structure Circle (center : ℝ × ℝ) (radius : ℝ) (D E F : ℝ × ℝ) : Prop where
  center_on_DE : ∃ t : ℝ, center = (D.1 + t * (E.1 - D.1), D.2 + t * (E.2 - D.2))
  tangent_to_DF : ∃ Q : ℝ × ℝ, (Q.1 - D.1)^2 + (Q.2 - D.2)^2 = radius^2 ∧
                               (Q.1 - F.1)^2 + (Q.2 - F.2)^2 = radius^2
  tangent_to_EF : (F.1 - center.1)^2 + (F.2 - center.2)^2 = radius^2

theorem triangle_circle_tangent (D E F : ℝ × ℝ) (center : ℝ × ℝ) (radius : ℝ)
  (h_triangle : Triangle D E F) (h_circle : Circle center radius D E F) :
  ∃ Q : ℝ × ℝ, (F.1 - Q.1)^2 + (F.2 - Q.2)^2 = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_tangent_l696_69628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_theorem_l696_69687

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- State the theorem
theorem solution_set_theorem (h1 : ∀ x, HasDerivAt f (f' x) x)
                             (h2 : ∀ x, f x < 2 * f' x)
                             (h3 : f (Real.log 4) = 2) :
  {x : ℝ | f x > Real.exp (x / 2)} = Set.Ioi (2 * Real.log 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_theorem_l696_69687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_platform_time_l696_69671

/-- Calculates the time for a train to cross a platform given the train length, platform length, and time to cross a signal pole. -/
noncomputable def time_to_cross_platform (train_length platform_length signal_pole_time : ℝ) : ℝ :=
  let train_speed := train_length / signal_pole_time
  let total_distance := train_length + platform_length
  total_distance / train_speed

/-- Theorem stating that a 300m train crossing a 250m platform takes 33 seconds, given it crosses a signal pole in 18 seconds. -/
theorem train_crossing_platform_time :
  time_to_cross_platform 300 250 18 = 33 := by
  -- Unfold the definition of time_to_cross_platform
  unfold time_to_cross_platform
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_platform_time_l696_69671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l696_69698

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the vectors
def m (t : Triangle) : ℝ × ℝ := (t.b, t.c - t.a)
noncomputable def n (t : Triangle) : ℝ × ℝ := (Real.sin t.C + Real.sin t.A, Real.sin t.C - Real.sin t.B)

-- Define the parallel condition
def parallel (v w : ℝ × ℝ) : Prop := v.1 * w.2 = v.2 * w.1

-- Define the area of the triangle
noncomputable def area (t : Triangle) : ℝ := (1/2) * t.b * t.c * Real.sin t.A

-- State the theorem
theorem triangle_proof (t : Triangle) 
  (h1 : parallel (m t) (n t))
  (h2 : t.b + t.c = 4)
  (h3 : area t = 3 * Real.sqrt 3 / 4) :
  t.A = π/3 ∧ t.a = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l696_69698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_game_probability_l696_69657

/-- The probability of getting exactly two heads in n flips of a biased coin -/
noncomputable def prob_two_heads (n : ℕ) : ℝ :=
  (n.choose 2) * (1/3)^2 * (2/3)^(n-2)

/-- The probability that all three players flip their coins the same number of times -/
noncomputable def prob_same_flips : ℝ := ∑' n, if n ≥ 2 then (prob_two_heads n)^3 else 0

theorem coin_flip_game_probability :
  prob_same_flips = ∑' n, if n ≥ 2 then ((n.choose 2) * (1/3)^2 * (2/3)^(n-2))^3 else 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_game_probability_l696_69657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l696_69620

theorem problem_statement (n a : ℝ) (b : ℝ) :
  n = 2 ^ (1 / 10 : ℝ) →
  n ^ b = a →
  b = 40.00000000000002 →
  ∃ (k : ℤ), a = k →
  a > 0 →
  a = 16 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l696_69620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_cylindrical_specific_point_l696_69672

noncomputable def rectangular_to_cylindrical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 then Real.arctan (y / x) + Real.pi
           else if y ≥ 0 then Real.pi / 2
           else 3 * Real.pi / 2
  (r, θ, z)

theorem rectangular_to_cylindrical_specific_point :
  let (r, θ, z) := rectangular_to_cylindrical 3 (-3 * Real.sqrt 3) 2
  r = 6 ∧ θ = 5 * Real.pi / 3 ∧ z = 2 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_cylindrical_specific_point_l696_69672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_for_x_power_seven_l696_69684

theorem min_n_for_x_power_seven (n : ℕ) (hn : n > 0) : 
  (∃ k : ℕ, (Nat.choose n k) * (2 * n - 5 * k) = 7) → n ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_for_x_power_seven_l696_69684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trenton_commission_rate_l696_69674

/-- Trenton's fixed weekly earnings in dollars -/
noncomputable def fixed_earnings : ℝ := 190

/-- Trenton's goal earnings in dollars -/
noncomputable def goal_earnings : ℝ := 500

/-- Minimum sales required to reach the goal in dollars -/
noncomputable def min_sales : ℝ := 7750

/-- Trenton's commission rate as a decimal -/
noncomputable def commission_rate : ℝ := (goal_earnings - fixed_earnings) / min_sales

theorem trenton_commission_rate :
  commission_rate = 0.04 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trenton_commission_rate_l696_69674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_rectangles_in_partitioned_square_l696_69611

/-- Represents a square partitioned by horizontal and vertical line segments. -/
structure PartitionedSquare where
  /-- The number of rectangles intersected by any vertical line -/
  k : ℕ
  /-- The number of rectangles intersected by any horizontal line -/
  l : ℕ

/-- Function to calculate the number of rectangles in a partitioned square -/
def number_of_rectangles (sq : PartitionedSquare) : ℕ := sq.k * sq.l

/-- Theorem stating that the number of rectangles in a partitioned square is k * l -/
theorem num_rectangles_in_partitioned_square (sq : PartitionedSquare) :
  number_of_rectangles sq = sq.k * sq.l := by
  -- Unfold the definition of number_of_rectangles
  unfold number_of_rectangles
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_rectangles_in_partitioned_square_l696_69611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_triple_count_l696_69653

theorem lcm_triple_count :
  let S := {(x, y, z) : ℕ × ℕ × ℕ | 
    x > 0 ∧ y > 0 ∧ z > 0 ∧
    Nat.lcm x y = 180 ∧
    Nat.lcm x z = 450 ∧
    Nat.lcm y z = 600}
  Finset.card (Finset.filter (fun (x, y, z) => 
    x > 0 ∧ y > 0 ∧ z > 0 ∧
    Nat.lcm x y = 180 ∧
    Nat.lcm x z = 450 ∧
    Nat.lcm y z = 600) (Finset.range 1000 ×ˢ Finset.range 1000 ×ˢ Finset.range 1000)) = 6 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_triple_count_l696_69653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l696_69694

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 4)^2 + (y - 1)^2 = 1

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem min_distance_sum :
  ∀ M A : ℝ × ℝ,
  parabola M.1 M.2 →
  circle_eq A.1 A.2 →
  ∀ ε > 0,
  ∃ M' A' : ℝ × ℝ,
  parabola M'.1 M'.2 ∧
  circle_eq A'.1 A'.2 ∧
  distance M' A' + distance M' focus < 4 + ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l696_69694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l696_69632

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x + 1/x
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := x^2 + x - b

-- Define the derivative of f
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := a/x - 1 - 1/x^2

-- Define the function h
noncomputable def h (a b : ℝ) (x : ℝ) : ℝ := f a x / g b x

-- State the theorem
theorem problem_solution :
  -- Part 1: Correct values of a and b
  (∃ (x : ℝ), f 2 x = 0 ∧ g 2 x = 0 ∧ f' 2 x = 0) ∧
  -- Part 2: Sign of h(x)
  (∀ x : ℝ, x > 0 ∧ x ≠ 1 → h 2 2 x < 0) ∧
  -- Part 3: Inequality for sum of reciprocals
  (∀ n : ℕ, n ≥ 2 → (Finset.range n).sum (λ i ↦ 1 / (i + 1 : ℝ)) > Real.log n + (n + 1 : ℝ) / (2 * n)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l696_69632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_and_perimeter_l696_69664

/-- A rhombus with given diagonal lengths -/
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ

/-- Calculate the area of a rhombus -/
noncomputable def area (r : Rhombus) : ℝ :=
  r.diagonal1 * r.diagonal2 / 2

/-- Calculate the side length of a rhombus using the Pythagorean theorem -/
noncomputable def sideLength (r : Rhombus) : ℝ :=
  Real.sqrt ((r.diagonal1 / 2) ^ 2 + (r.diagonal2 / 2) ^ 2)

/-- Calculate the perimeter of a rhombus -/
noncomputable def perimeter (r : Rhombus) : ℝ :=
  4 * sideLength r

theorem rhombus_area_and_perimeter (r : Rhombus) 
    (h1 : r.diagonal1 = 6) 
    (h2 : r.diagonal2 = 8) : 
    area r = 24 ∧ perimeter r = 20 := by
  sorry

#check rhombus_area_and_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_and_perimeter_l696_69664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_when_x_greater_than_one_smallest_a_for_inequality_l696_69650

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x - (1 - 1/x)

-- Theorem 1
theorem f_positive_when_x_greater_than_one :
  ∀ x : ℝ, x > 1 → f x > 0 := by
  sorry

-- Theorem 2
theorem smallest_a_for_inequality :
  (∀ x : ℝ, x > 1 → Real.log x / x < a * (x - 1)) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_when_x_greater_than_one_smallest_a_for_inequality_l696_69650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l696_69636

/-- The constant term in the expansion of (2x - 1/x)^6 is -160 -/
theorem constant_term_expansion : 
  ∃ (f : ℝ → ℝ), (∀ x : ℝ, x ≠ 0 → f x = (2*x - 1/x)^6) ∧ 
  (∃ c : ℝ, c = -160 ∧ ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ → |f x - c| < ε) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l696_69636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_three_digits_after_decimal_l696_69651

/-- The first three digits to the right of the decimal point in (10^3010 + 1)^(5/9) are 555 -/
theorem first_three_digits_after_decimal (n : ℕ) (x : ℝ) : 
  n = 3010 → x = (10^n + 1)^(5/9) → 
  ∃ (y : ℝ), x = y + 0.555 ∧ y = ⌊y⌋ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_three_digits_after_decimal_l696_69651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_line_intersection_l696_69601

-- Define the ellipse
def ellipse (x y a : ℝ) : Prop := x^2 / a^2 + y^2 / 3 = 1

-- Define the line
def line (x y m : ℝ) : Prop := x = m * y + 3

-- Define the perpendicularity condition
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Define the area of triangle PMN
noncomputable def triangle_area (y1 y2 m : ℝ) : ℝ := 
  2 * Real.sqrt 3 * Real.sqrt ((m^2 + 1) / ((m^2 + 4)^2))

theorem ellipse_focus_line_intersection 
  (a : ℝ) 
  (ha : a > Real.sqrt 10) 
  (hm : ∀ m : ℝ, m ≠ 0 → 
    ∃ x1 y1 x2 y2 : ℝ, 
      ellipse x1 y1 a ∧ ellipse x2 y2 a ∧ 
      line x1 y1 m ∧ line x2 y2 m) :
  -- 1. Equation of the ellipse
  (∀ x y : ℝ, ellipse x y a ↔ x^2 / 12 + y^2 / 3 = 1) ∧
  -- 2. Value of m when OM ⟂ ON
  (∀ x1 y1 x2 y2 m : ℝ, 
    ellipse x1 y1 a → ellipse x2 y2 a → 
    line x1 y1 m → line x2 y2 m → 
    perpendicular x1 y1 x2 y2 → 
    m = Real.sqrt 11 / 2 ∨ m = -Real.sqrt 11 / 2) ∧
  -- 3. Maximum area of triangle PMN
  (∀ m : ℝ, m ≠ 0 → 
    ∃ y1 y2 : ℝ, triangle_area y1 y2 m ≤ 1 ∧
    (∃ m0 : ℝ, triangle_area y1 y2 m0 = 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_line_intersection_l696_69601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_30_equals_2cos_30_l696_69679

theorem cot_30_equals_2cos_30 : Real.tan (π / 6)⁻¹ = 2 * Real.cos (π / 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_30_equals_2cos_30_l696_69679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_solutions_l696_69683

/-- A function satisfying the given functional equation. -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y + f (x + y) = x * y

/-- The set of functions that satisfy the equation. -/
def SolutionSet : Set (ℝ → ℝ) :=
  { f | f = (λ _ ↦ 0) ∨ f = (λ x ↦ x - 1) ∨ f = (λ x ↦ -x - 1) }

/-- Theorem stating that a function satisfies the equation if and only if it's in the solution set. -/
theorem characterization_of_solutions :
    ∀ f : ℝ → ℝ, SatisfiesEquation f ↔ f ∈ SolutionSet := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_solutions_l696_69683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_ratio_ellipse_hyperbola_l696_69616

/-- Predicate for a point being on an ellipse with given foci and eccentricity -/
def is_ellipse (F₁ F₂ : ℝ × ℝ) (e : ℝ) : Prop :=
  0 < e ∧ e < 1 ∧ ∀ P, (‖P - F₁‖ + ‖P - F₂‖ = 2 * Real.sqrt (1 - e^2) * ‖F₁ - F₂‖ / (2 * e))

/-- Predicate for a point being on a hyperbola with given foci and eccentricity -/
def is_hyperbola (F₁ F₂ : ℝ × ℝ) (e : ℝ) : Prop :=
  e > 1 ∧ ∀ P, (|‖P - F₁‖ - ‖P - F₂‖| = 2 * Real.sqrt (e^2 - 1) * ‖F₁ - F₂‖ / (2 * e))

/-- The set of points on an ellipse with given foci and eccentricity -/
def ellipse (F₁ F₂ : ℝ × ℝ) (e : ℝ) : Set (ℝ × ℝ) :=
  {P | ‖P - F₁‖ + ‖P - F₂‖ = 2 * Real.sqrt (1 - e^2) * ‖F₁ - F₂‖ / (2 * e)}

/-- The set of points on a hyperbola with given foci and eccentricity -/
def hyperbola (F₁ F₂ : ℝ × ℝ) (e : ℝ) : Set (ℝ × ℝ) :=
  {P | |‖P - F₁‖ - ‖P - F₂‖| = 2 * Real.sqrt (e^2 - 1) * ‖F₁ - F₂‖ / (2 * e)}

/-- Given an ellipse and a hyperbola with common foci and a common point P,
    prove that the ratio of their eccentricities is equal to √2/2 -/
theorem eccentricity_ratio_ellipse_hyperbola 
  (e₁ e₂ : ℝ) 
  (F₁ F₂ P : ℝ × ℝ) 
  (h_ellipse : is_ellipse F₁ F₂ e₁)
  (h_hyperbola : is_hyperbola F₁ F₂ e₂)
  (h_common_point : P ∈ ellipse F₁ F₂ e₁ ∧ P ∈ hyperbola F₁ F₂ e₂)
  (h_vector_sum : ‖P - F₁ + (P - F₂)‖ = ‖F₁ - F₂‖) :
  e₁ * e₂ / Real.sqrt (e₁^2 + e₂^2) = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_ratio_ellipse_hyperbola_l696_69616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l696_69619

noncomputable def f (x : ℝ) : ℝ := Real.cos (x - Real.pi/3)

theorem range_of_f :
  ∀ y ∈ Set.range f,
  (∃ x ∈ Set.Icc 0 (Real.pi/2), f x = y) ↔ y ∈ Set.Icc (1/2) 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l696_69619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_coin_flip_probability_l696_69614

def coin_flip_probability : ℚ := 79 / 4096

theorem fair_coin_flip_probability :
  (let n : ℕ := 12  -- number of coin flips
   let k : ℕ := 10  -- minimum number of heads
   let p : ℚ := 1/2 -- probability of heads for a fair coin
   (Finset.range (n+1)).sum (λ i ↦ if i ≥ k then (n.choose i) * p^i * (1-p)^(n-i) else 0)) = coin_flip_probability :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_coin_flip_probability_l696_69614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l696_69692

-- Define an odd function f : ℝ → ℝ
noncomputable def f : ℝ → ℝ := fun x => 
  if x < 0 then x + 2 else x - 2

-- State the theorem
theorem inequality_solution_set :
  ∀ x : ℝ, 2 * f x - 1 < 0 ↔ x < -3/2 ∨ (0 ≤ x ∧ x < 5/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l696_69692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_intersect_Q_equals_one_two_l696_69615

-- Define set P
def P : Set ℕ := {x | 1 ≤ x ∧ x ≤ 10}

-- Define set Q
def Q : Set ℝ := {x | x^2 - x - 6 < 0}

-- Define the intersection of P and Q
def P_intersect_Q : Set ℕ := {x ∈ P | (x : ℝ) ∈ Q}

-- Theorem to prove
theorem P_intersect_Q_equals_one_two : P_intersect_Q = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_intersect_Q_equals_one_two_l696_69615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l696_69617

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let dAC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let dBC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let dAB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  dAC = 10 ∧ dBC = 10 ∧ dAB = 6

-- Define point D on line AB
def PointD (A B D : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ D.1 = A.1 + t * (B.1 - A.1) ∧ D.2 = A.2 + t * (B.2 - A.2)

-- Define the distance between two points
noncomputable def Distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

-- Theorem statement
theorem triangle_side_length 
  (A B C D : ℝ × ℝ) 
  (h1 : Triangle A B C) 
  (h2 : PointD A B D) 
  (h3 : Distance C D = 12) : 
  Distance B D = Real.sqrt 53 - 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l696_69617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_comparability_l696_69605

/-- A rectangle represented by its width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Defines when one rectangle is comparable to another -/
def comparable (r1 r2 : Rectangle) : Prop :=
  (r1.width ≤ r2.width ∧ r1.height ≤ r2.height) ∨
  (r1.width ≤ r2.height ∧ r1.height ≤ r2.width)

/-- The theorem to be proved -/
theorem rectangle_comparability {n : ℕ} (h : n^2 > 1) :
  ∃ (subset : Finset Rectangle),
    subset.card = 2*n ∧
    (∀ r1 r2, r1 ∈ subset → r2 ∈ subset → comparable r1 r2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_comparability_l696_69605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiffany_blocks_proof_l696_69675

/-- The number of blocks Moses runs -/
def moses_blocks : ℕ := 12

/-- The time Moses takes to run, in minutes -/
def moses_time : ℕ := 8

/-- The time Tiffany takes to run, in minutes -/
def tiffany_time : ℕ := 3

/-- The speed of the faster runner, in blocks per minute -/
def faster_speed : ℚ := 2

/-- The average speed of Moses, in blocks per minute -/
def moses_speed : ℚ := moses_blocks / moses_time

/-- The number of blocks Tiffany runs -/
def tiffany_blocks : ℕ := 6

theorem tiffany_blocks_proof : tiffany_blocks = 6 := by
  -- Proof steps would go here
  sorry

#eval tiffany_blocks -- Should output 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiffany_blocks_proof_l696_69675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l696_69676

theorem negation_of_proposition :
  (¬(∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1)) ↔ 
  (∃ x₀ : ℝ, x₀ > 0 ∧ (x₀ + 1) * Real.exp x₀ ≤ 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l696_69676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l696_69600

/-- The length of a train given its speed, time to cross a tunnel, and the tunnel's length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (tunnel_length : ℝ) :
  train_speed = 72 * (1000 / 3600) →
  crossing_time = 74.994 →
  tunnel_length = 1400 →
  ∃ (train_length : ℝ), abs (train_length - 99.88) < 0.01 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l696_69600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_A_in_triangle_l696_69609

/-- Given a triangle ABC with side lengths a and b, and angle B,
    prove that if a = 4, b = 6, and B = 60°, then sin A = √3/3 -/
theorem sin_A_in_triangle (a b : ℝ) (A B C : Real) :
  a = 4 → b = 6 → B = π/3 → Real.sin A = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_A_in_triangle_l696_69609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_interval_l696_69638

open Real

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (cos (x - π/6))^2

-- Define the monotonic increasing interval
def is_monotonic_increasing_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- Theorem statement
theorem f_monotonic_increasing_interval :
  ∀ k : ℤ, is_monotonic_increasing_interval f (-π/3 + ↑k*π) (π/6 + ↑k*π) := by
  sorry

#check f_monotonic_increasing_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_interval_l696_69638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l696_69622

/-- The equation of the asymptotes of a hyperbola with given parameters -/
def asymptote_equation (a b : ℝ) (x y : ℝ) : Prop :=
  b * x = a * y ∨ b * x = -a * y

/-- The equation of a hyperbola with given parameters -/
def hyperbola_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

theorem hyperbola_asymptotes :
  ∀ (x y : ℝ),
  hyperbola_equation 3 4 x y →
  asymptote_equation 3 4 x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l696_69622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_decreasing_others_not_l696_69668

-- Define the interval [0, π/2]
def I : Set ℝ := Set.Icc 0 (Real.pi / 2)

-- Define the functions
noncomputable def f₁ (x : ℝ) : ℝ := Real.cos x
noncomputable def f₂ (x : ℝ) : ℝ := Real.sin x
noncomputable def f₃ (x : ℝ) : ℝ := Real.tan x
noncomputable def f₄ (x : ℝ) : ℝ := Real.sin (x - Real.pi / 3)

-- State the theorem
theorem cos_decreasing_others_not :
  (∀ x y, x ∈ I → y ∈ I → x < y → f₁ y < f₁ x) ∧
  ¬(∀ x y, x ∈ I → y ∈ I → x < y → f₂ y < f₂ x) ∧
  ¬(∀ x y, x ∈ I → y ∈ I → x < y → f₃ y < f₃ x) ∧
  ¬(∀ x y, x ∈ I → y ∈ I → x < y → f₄ y < f₄ x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_decreasing_others_not_l696_69668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sneaker_lace_length_l696_69624

/-- Represents the dimensions and configuration of a sneaker's eyelet arrangement -/
structure SneakerEyelet where
  width : ℝ
  length : ℝ
  num_eyelets : ℕ

/-- Calculates the minimum lace length required for a given sneaker eyelet configuration -/
noncomputable def min_lace_length (s : SneakerEyelet) : ℝ :=
  let segment_length := s.length / (s.num_eyelets / 2 - 1)
  let diagonal := Real.sqrt (s.width^2 + segment_length^2)
  (s.num_eyelets - 2) * diagonal + s.width + 2 * 200

/-- Theorem stating the minimum lace length for the given sneaker configuration -/
theorem sneaker_lace_length :
  let s : SneakerEyelet := { width := 50, length := 80, num_eyelets := 8 }
  min_lace_length s = 790 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sneaker_lace_length_l696_69624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l696_69677

theorem sin_2alpha_value (α β : ℝ) 
  (h1 : 0 < β) (h2 : β < α) (h3 : α < π/4)
  (h4 : Real.cos (α - β) = 12/13)
  (h5 : Real.sin (α + β) = 4/5) : 
  Real.sin (2*α) = 63/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l696_69677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_draw_theorem_l696_69686

/-- Represents a card in a standard deck --/
structure Card where
  suit : Fin 4
  value : Fin 13
  deriving Repr

/-- Represents a draw of cards from a standard deck --/
structure Draw where
  hearts : Fin 13 → Nat
  spades : Fin 13 → Nat
  diamonds : Fin 13 → Nat
  clubs : Fin 13 → Nat

def Draw.total_cards (d : Draw) : Nat :=
  (Finset.sum Finset.univ d.hearts) + (Finset.sum Finset.univ d.spades) +
  (Finset.sum Finset.univ d.diamonds) + (Finset.sum Finset.univ d.clubs)

def Draw.sum_values (d : Draw) : Nat :=
  Finset.sum (Finset.range 13) (λ i => (i + 1) * (d.hearts i + d.spades i + d.diamonds i + d.clubs i))

def Draw.count_twos (d : Draw) : Nat :=
  d.hearts 1 + d.spades 1 + d.diamonds 1 + d.clubs 1

theorem card_draw_theorem (d : Draw) :
  d.total_cards = 14 ∧
  (Finset.sum Finset.univ d.hearts) = 2 ∧
  (Finset.sum Finset.univ d.spades) = 3 ∧
  (Finset.sum Finset.univ d.diamonds) = 4 ∧
  (Finset.sum Finset.univ d.clubs) = 5 ∧
  d.sum_values = 34 →
  d.count_twos = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_draw_theorem_l696_69686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l696_69631

-- Define the function y as noncomputable
noncomputable def y (x : ℝ) : ℝ := Real.tan (x + 2 * Real.pi / 3) - Real.tan (x + Real.pi / 6) + Real.cos (x + Real.pi / 6)

-- Define the interval
def I : Set ℝ := { x | -5 * Real.pi / 12 ≤ x ∧ x ≤ -Real.pi / 3 }

-- Theorem statement
theorem max_value_of_y :
  ∃ (M : ℝ), M = (11 * Real.sqrt 3) / 6 ∧ 
  (∀ x ∈ I, y x ≤ M) ∧
  (∃ x ∈ I, y x = M) := by
  sorry

#check max_value_of_y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l696_69631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_implies_base_equality_l696_69643

theorem log_equality_implies_base_equality :
  ∀ x : ℝ, x > 0 → (Real.log 256 / Real.log x = Real.log 256 / Real.log 4) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_implies_base_equality_l696_69643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_f_difference_bound_l696_69618

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_domain : ∀ x, x ∈ Set.Icc (-1) 1 → f x ∈ Set.Icc (-1) 1
axiom f_endpoints : f (-1) = 0 ∧ f 1 = 0
axiom f_lipschitz : ∀ u v, u ∈ Set.Icc (-1) 1 → v ∈ Set.Icc (-1) 1 → |f u - f v| ≤ |u - v|

-- Theorem 1
theorem f_bounds :
  ∀ x, x ∈ Set.Icc (-1) 1 → x - 1 ≤ f x ∧ f x ≤ 1 - x := by
  sorry

-- Theorem 2
theorem f_difference_bound :
  ∀ u v, u ∈ Set.Icc (-1) 1 → v ∈ Set.Icc (-1) 1 → |f u - f v| ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_f_difference_bound_l696_69618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_bound_l696_69667

theorem polynomial_value_bound (P : Polynomial ℤ) (n : ℤ) :
  (∃ (roots : Finset ℤ), roots.card ≥ 13 ∧ ∀ r ∈ roots, P.eval r = 0) →
  P.eval n ≠ 0 →
  |P.eval n| ≥ 7 * (Nat.factorial 6)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_bound_l696_69667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jonathan_total_exercise_time_l696_69670

/-- Represents Jonathan's exercise routine for a specific day -/
structure ExerciseDay where
  speed : ℚ
  distance : ℚ

/-- Calculates the time spent exercising on a given day -/
def exerciseTime (day : ExerciseDay) : ℚ :=
  day.distance / day.speed

/-- Jonathan's weekly exercise routine -/
def jonathan_routine : List ExerciseDay :=
  [{ speed := 2, distance := 6 },  -- Monday
   { speed := 3, distance := 6 },  -- Wednesday
   { speed := 6, distance := 6 }]  -- Friday

theorem jonathan_total_exercise_time :
  (jonathan_routine.map exerciseTime).sum = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jonathan_total_exercise_time_l696_69670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_l696_69646

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) ^ 2 - (Real.sin x) ^ 2 + 1

theorem f_period : ∃ (p : ℝ), p > 0 ∧ p = π ∧ ∀ (x : ℝ), f (x + p) = f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_l696_69646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_order_determinable_l696_69678

-- Define the type for cars
inductive Car : Type
  | A | B | C | D

-- Define a type for the race order
def RaceOrder := List Car

-- Define a function to represent position exchanges
def exchange_count (car1 car2 : Car) : ℕ := sorry

-- Define the initial order
def initial_order : RaceOrder := [Car.A, Car.B, Car.C, Car.D]

-- Define the known exchanges
axiom AB_exchanges : exchange_count Car.A Car.B = 9
axiom BC_exchanges : exchange_count Car.B Car.C = 8

-- Define a function to represent asking a question about exchanges
def ask_exchange_question (car1 car2 : Car) : ℕ := sorry

-- Theorem statement
theorem race_order_determinable :
  ∃ (q1 q2 q3 : Car × Car) (final_order : RaceOrder),
    (q1 ≠ q2 ∧ q1 ≠ q3 ∧ q2 ≠ q3) ∧
    (∀ (other_final_order : RaceOrder),
      (ask_exchange_question q1.1 q1.2 = exchange_count q1.1 q1.2) →
      (ask_exchange_question q2.1 q2.2 = exchange_count q2.1 q2.2) →
      (ask_exchange_question q3.1 q3.2 = exchange_count q3.1 q3.2) →
      other_final_order = final_order) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_order_determinable_l696_69678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_decreasing_function_theorem_l696_69633

def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def DecreasingOn (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x > f y

theorem even_decreasing_function_theorem (f : ℝ → ℝ) (a : ℝ) 
  (h1 : EvenFunction f)
  (h2 : DecreasingOn f (Set.Iic 0))
  (h3 : f a ≥ f 2) :
  a ≤ -2 ∨ a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_decreasing_function_theorem_l696_69633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_second_trip_length_l696_69602

/-- Calculates the length of the second trip given fuel consumption rate, total fuel, and length of the first trip -/
noncomputable def second_trip_length (fuel_consumption : ℝ) (total_fuel : ℝ) (first_trip_length : ℝ) : ℝ :=
  (total_fuel / fuel_consumption) - first_trip_length

/-- Theorem: Given John's fuel consumption, total fuel, and first trip length, the second trip length is 30 km -/
theorem john_second_trip_length :
  second_trip_length 5 250 20 = 30 := by
  -- Unfold the definition of second_trip_length
  unfold second_trip_length
  -- Simplify the arithmetic
  simp [div_eq_mul_inv]
  -- The proof is complete
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_second_trip_length_l696_69602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_implies_k_bound_l696_69680

/-- The function f(x) = kx² - ln(x) -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 - Real.log x

/-- Theorem stating that if f(x) > 0 for all x > 0, then k > 1/(2e) -/
theorem f_positive_implies_k_bound (k : ℝ) :
  (∀ x : ℝ, x > 0 → f k x > 0) → k > 1 / (2 * Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_implies_k_bound_l696_69680
