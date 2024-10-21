import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_qin_jiushao_formula_l347_34717

noncomputable def triangle_area (a b c : ℝ) : ℝ := 
  Real.sqrt ((1/4) * (c^2 * a^2 - ((c^2 + a^2 - b^2)/2)^2))

theorem qin_jiushao_formula (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a = 15) (h4 : b = 14) (h5 : c = 13) : 
  triangle_area a b c = 84 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval triangle_area 15 14 13

end NUMINAMATH_CALUDE_ERRORFEEDBACK_qin_jiushao_formula_l347_34717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factors_count_l347_34768

theorem prime_factors_count (n : ℕ) : 
  (∀ p : ℕ, Nat.Prime p → p ∣ n → p = 2 ∨ p = 7 ∨ p = 11) →
  (4 = 2^2) →
  Nat.Prime 7 →
  Nat.Prime 11 →
  n = 4^11 * 7^5 * 11^2 →
  (Finset.card (Finset.filter (fun p => Nat.Prime p ∧ p ∣ n) (Finset.range (n + 1)))) = 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factors_count_l347_34768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_squared_minus_3B_l347_34709

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 3, 2]

theorem det_B_squared_minus_3B : 
  Matrix.det (B^2 - 3 • B) = 88 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_squared_minus_3B_l347_34709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_average_weight_l347_34778

theorem girls_average_weight 
  (num_boys : ℕ) 
  (num_total : ℕ) 
  (avg_boys : ℚ) 
  (avg_total : ℚ) 
  (h1 : num_boys = 15)
  (h2 : num_total = 25)
  (h3 : avg_boys = 48)
  (h4 : avg_total = 45) :
  (num_total * avg_total - num_boys * avg_boys) / (num_total - num_boys) = 40.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_average_weight_l347_34778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_cardinality_l347_34748

-- Define our sets
def A : Finset String := {("circle")}
def B : Finset String := {("line")}

-- State the theorem
theorem intersection_cardinality : Finset.card (A ∩ B) = 0 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_cardinality_l347_34748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_range_l347_34703

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the length of a side
noncomputable def sideLength (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the median AD
noncomputable def median (t : Triangle) : ℝ × ℝ :=
  ((t.B.1 + t.C.1) / 2, (t.B.2 + t.C.2) / 2)

theorem median_range (t : Triangle) :
  let AB := sideLength t.A t.B
  let AC := sideLength t.A t.C
  let AD := sideLength t.A (median t)
  AC = (Real.sqrt (5 - AB) + Real.sqrt (2 * AB - 10) + AB + 1) / 2 →
  1 < AD ∧ AD < 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_range_l347_34703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_and_sum_properties_l347_34702

def arithmetic_sequence (a b c : ℝ) : Prop := b - a = c - b

def sum_formula (S a : ℕ → ℝ) : Prop := ∀ n, S n = 2 * a n - a 1

theorem sequence_and_sum_properties
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h_sum : sum_formula S a)
  (h_arith : arithmetic_sequence (a 1) (a 2 + 1) (a 2)) :
  (∀ n : ℕ, a n = (2 : ℝ)^n) ∧
  (∀ n : ℕ, n ≥ 10 → |1 - 1 / ((2 : ℝ)^(n+1) - 1)| < 1 / 2016) ∧
  (∀ n : ℕ, n < 10 → |1 - 1 / ((2 : ℝ)^(n+1) - 1)| ≥ 1 / 2016) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_and_sum_properties_l347_34702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_measure_one_liter_a_measure_one_liter_b_l347_34750

-- Define the bucket capacities
def bucket1 : ℚ := 5
def bucket2 : ℚ := 7

noncomputable def bucket3 : ℝ := 2 - Real.sqrt 2
noncomputable def bucket4 : ℝ := Real.sqrt 2

-- Theorem for part a
theorem measure_one_liter_a : ∃ (k l : ℤ), (1 : ℚ) = k * bucket1 + l * bucket2 := by sorry

-- Theorem for part b
theorem measure_one_liter_b : ¬∃ (m n : ℤ), (1 : ℝ) = m * bucket3 + n * bucket4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_measure_one_liter_a_measure_one_liter_b_l347_34750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_travel_distance_l347_34789

/-- Represents the configuration of an arc in the track -/
structure Arc where
  radius : ℝ
  fraction : ℝ
  isOutside : Bool

/-- Calculates the distance traveled by the ball's center on a single arc -/
noncomputable def distanceOnArc (ballDiameter : ℝ) (arc : Arc) : ℝ :=
  let adjustedRadius := if arc.isOutside then arc.radius - ballDiameter / 2 else arc.radius + ballDiameter / 2
  arc.fraction * adjustedRadius * Real.pi

/-- Theorem: The total distance traveled by the center of the ball -/
theorem ball_travel_distance (ballDiameter : ℝ) (arcs : List Arc) : 
  ballDiameter = 3 →
  arcs = [
    { radius := 120, fraction := 1, isOutside := true },
    { radius := 50, fraction := 1, isOutside := false },
    { radius := 75, fraction := 1/2, isOutside := false },
    { radius := 20, fraction := 1, isOutside := true }
  ] →
  (arcs.map (distanceOnArc ballDiameter)).sum = 226.75 * Real.pi := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_travel_distance_l347_34789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_fourth_prop_true_l347_34735

-- Define the propositions
def prop1 : Prop := ∀ x : ℝ, x^2 + 1 < 3*x ↔ ¬∃ x : ℝ, x^2 + 1 > 3*x
def prop2 : Prop := ∀ a : ℝ, (a > 2 → a > 5) ∧ ¬(a > 5 → a > 2)
def prop3 : Prop := ∀ x y : ℝ, (¬(x = 0 ∧ y = 0) → x * y ≠ 0)
def prop4 : Prop := ∀ p q : Prop, ¬(p ∨ q) → (¬p ∧ ¬q)

-- Theorem stating that only the fourth proposition is true
theorem only_fourth_prop_true : ¬prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ prop4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_fourth_prop_true_l347_34735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l347_34712

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in parametric form -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ
  α : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem about the line and its properties -/
theorem line_properties (P : Point) (l : ParametricLine) (A B : Point) :
  P.x = 2 ∧ P.y = 1 ∧
  (∀ t, l.x t = 2 + t * Real.cos l.α) ∧
  (∀ t, l.y t = 1 + t * Real.sin l.α) ∧
  A.y = 0 ∧ B.x = 0 ∧
  (∃ t₁, l.x t₁ = A.x ∧ l.y t₁ = A.y) ∧
  (∃ t₂, l.x t₂ = B.x ∧ l.y t₂ = B.y) ∧
  distance P A * distance P B = 4 →
  l.α = 3 * Real.pi / 4 ∧
  (∀ θ ρ, ρ * (Real.sin θ + Real.cos θ) = 3 ↔ 
    ρ * Real.cos θ + ρ * Real.sin θ - 3 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l347_34712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l347_34736

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a / x

theorem problem_solution (a : ℝ) :
  (f a 1 = 10) →
  (a = 9) ∧
  (∀ x : ℝ, x ≠ 0 → f 9 (-x) = -(f 9 x)) ∧
  (∀ x₁ x₂ : ℝ, 3 < x₁ ∧ x₁ < x₂ → f 9 x₁ < f 9 x₂) :=
by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l347_34736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_of_squares_l347_34724

/-- Given a line segment AB with point O on it such that AO : OB = a : b (a > 0, b > 0),
    and a point P on circle O(r) with radius r, prove that b · PA² + a · PB² is constant. -/
theorem constant_sum_of_squares (A B O P : ℝ × ℝ) (a b r k : ℝ) : 
  a > 0 → b > 0 → k > 0 →
  O = (0, 0) →
  A = (-k * a, 0) →
  B = (k * b, 0) →
  ‖P - O‖ = r →
  b * ‖P - A‖^2 + a * ‖P - B‖^2 = r^2 * (a + b) + k^2 * a * b * (a + b) := by
  sorry

#check constant_sum_of_squares

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_of_squares_l347_34724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l347_34783

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Two non-zero vectors are not collinear -/
def NotCollinear (a b : V) : Prop := a ≠ 0 ∧ b ≠ 0 ∧ ∀ (r : ℝ), a ≠ r • b

/-- Three points are collinear if one is between the other two -/
def AreCollinear (A B D : V) : Prop := ∃ (t : ℝ), D - A = t • (B - A) ∨ B - A = t • (D - A)

/-- Main theorem -/
theorem vector_collinearity 
  (a b : V) 
  (h_not_collinear : NotCollinear a b)
  (AB BC CD : V)
  (h_AB : AB = a + b)
  (h_BC : BC = 2 • a + 8 • b)
  (h_CD : CD = 3 • (a - b)) :
  (∃ (A B D : V), AreCollinear A B D) ∧
  (∀ (k : ℝ), (∃ (r : ℝ), k • a + b = r • (a + k • b)) ↔ k = 1 ∨ k = -1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l347_34783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_passing_time_l347_34745

/-- The time taken for a goods train to pass a man in another train -/
noncomputable def time_to_pass (mans_train_speed goods_train_speed : ℝ) (goods_train_length : ℝ) : ℝ :=
  let relative_speed := mans_train_speed + goods_train_speed
  let relative_speed_mps := relative_speed * 1000 / 3600
  goods_train_length / relative_speed_mps

/-- Theorem stating the time taken for the goods train to pass the man -/
theorem goods_train_passing_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |time_to_pass 15 97 280 - 8.99| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_passing_time_l347_34745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l347_34716

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (compounds_per_year : ℕ) (time : ℝ) : ℝ :=
  principal * (1 + rate / (compounds_per_year : ℝ)) ^ ((compounds_per_year : ℝ) * time)

theorem investment_growth :
  let principal : ℝ := 8000
  let rate : ℝ := 0.10
  let compounds_per_year : ℕ := 2
  let time : ℝ := 1
  compound_interest principal rate compounds_per_year time = 8820 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l347_34716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_composition_l347_34743

/-- Given an odd function f defined on ℝ where f(x) = (3^x - 4) / 3^x for x > 0,
    prove that f[f(log₃2)] = 1/3 -/
theorem odd_function_composition (f : ℝ → ℝ) : 
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x > 0, f x = (3^x - 4) / 3^x) →  -- definition of f for x > 0
  f (f (Real.log 2 / Real.log 3)) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_composition_l347_34743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_x_cubed_plus_two_l347_34718

open MeasureTheory Interval Real

theorem definite_integral_x_cubed_plus_two :
  ∫ x in (-2)..2, (x^3 + 2) = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_x_cubed_plus_two_l347_34718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_non_random_events_l347_34782

-- Define an event
def Event : Type := String

-- Define a function to determine if an event is random
def isRandomEvent : Event → Bool
| _ => false  -- Placeholder implementation

-- Define our 5 events
def event1 : Event := "Water turns into oil by itself"
def event2 : Event := "It will rain tomorrow"
def event3 : Event := "Xiao Ming scores a 10 in shooting"
def event4 : Event := "Ice melts under normal temperature and pressure"
def event5 : Event := "January of year 13 has 31 days"

-- List of all events
def allEvents : List Event := [event1, event2, event3, event4, event5]

-- Theorem: There are exactly 3 non-random events in our list
theorem three_non_random_events : 
  (allEvents.filter (fun e => ¬(isRandomEvent e))).length = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_non_random_events_l347_34782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_cost_proof_l347_34737

theorem cookie_cost_proof (selling_price_percentage : ℝ) 
                          (cookies_sold : ℕ) 
                          (total_earnings : ℝ) : ℝ := by
  -- Define the conditions
  have h1 : selling_price_percentage = 120 := by sorry
  have h2 : cookies_sold = 50 := by sorry
  have h3 : total_earnings = 60 := by sorry

  -- Calculate the selling price per cookie
  let selling_price_per_cookie := total_earnings / (cookies_sold : ℝ)

  -- Calculate the cost price per cookie
  let cost_per_cookie := selling_price_per_cookie / (selling_price_percentage / 100)

  -- Prove that the cost per cookie is 1
  have : cost_per_cookie = 1 := by
    sorry

  exact cost_per_cookie

#check cookie_cost_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_cost_proof_l347_34737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l347_34773

/-- The area of a triangle with vertices at (3, 2), (3, -4), and (11, 2) is 24 square units. -/
theorem triangle_area : ∃ (area : ℝ), area = 24 := by
  let vertex1 : ℝ × ℝ := (3, 2)
  let vertex2 : ℝ × ℝ := (3, -4)
  let vertex3 : ℝ × ℝ := (11, 2)
  
  -- Calculate the base and height of the triangle
  let base : ℝ := abs (vertex1.2 - vertex2.2)
  let height : ℝ := abs (vertex3.1 - vertex1.1)
  
  -- Calculate the area
  let area : ℝ := (1/2) * base * height
  
  -- Prove that the area is 24
  have h : area = 24 := by
    -- Expand the definitions and simplify
    simp [area, base, height]
    -- Perform the arithmetic
    norm_num
  
  -- Conclude the proof
  exact ⟨area, h⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l347_34773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_B_value_l347_34732

/-- The probability that A speaks the truth -/
noncomputable def prob_A : ℝ := 0.80

/-- The probability that both A and B speak the truth -/
noncomputable def prob_A_and_B : ℝ := 0.48

/-- The probability that B speaks the truth -/
noncomputable def prob_B : ℝ := prob_A_and_B / prob_A

theorem prob_B_value : prob_B = 0.60 := by
  -- Unfold the definitions
  unfold prob_B prob_A_and_B prob_A
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_B_value_l347_34732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_proper_subsets_l347_34747

def U : Finset Nat := {1, 2, 3}

theorem number_of_proper_subsets (A : Finset Nat) 
  (h : U \ A = {2}) : 
  (A.powerset.filter (· ⊂ A)).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_proper_subsets_l347_34747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_points_distance_l347_34730

/-- Two circles with centers at (3,5) and (20,15), both tangent to the x-axis -/
def circle1_center : ℝ × ℝ := (3, 5)
def circle2_center : ℝ × ℝ := (20, 15)

/-- The distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The theorem to be proved -/
theorem closest_points_distance :
  distance circle1_center circle2_center - (circle1_center.2 + circle2_center.2) = Real.sqrt 389 - 20 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_points_distance_l347_34730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taylor_formulas_l347_34728

-- Define the complex number i
def i : ℂ := Complex.I

-- Taylor expansion for e^x (up to 4th term)
noncomputable def e_taylor (x : ℝ) : ℝ := 1 + x + x^2 / 2 + x^3 / 6 + x^4 / 24

-- Taylor expansion for sin x (up to 4th term)
noncomputable def sin_taylor (x : ℝ) : ℝ := x - x^3 / 6 + x^5 / 120

-- Taylor expansion for cos x (up to 4th term)
noncomputable def cos_taylor (x : ℝ) : ℝ := 1 - x^2 / 2 + x^4 / 24

theorem taylor_formulas :
  (∀ x : ℝ, Complex.exp (i * x) = Complex.cos x + i * Complex.sin x) ∧
  (∀ x : ℝ, x ≥ 0 → (2 : ℝ)^x ≥ 1 + x * Real.log 2 + (x * Real.log 2)^2 / 2) ∧
  (∀ x : ℝ, x > 0 ∧ x < 1 → Real.cos x ≤ 1 - x^2 / 2 + x^4 / 24) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taylor_formulas_l347_34728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_condition_l347_34787

-- Define the line l
noncomputable def line_l (x y : ℝ) : Prop := 4 * x + 3 * y - 8 = 0

-- Define the circle C
noncomputable def circle_C (x y a : ℝ) : Prop := x^2 + y^2 - a * y = 0

-- Define the radius of the circle
noncomputable def radius (a : ℝ) : ℝ := a / 2

-- Define the distance from the center of the circle to the line
noncomputable def distance_center_to_line (a : ℝ) : ℝ := |3 * a / 2 - 8| / 5

-- Theorem statement
theorem chord_length_condition (a : ℝ) :
  (∀ x y, line_l x y → circle_C x y a) →
  ((radius a)^2 - (distance_center_to_line a)^2) * 12 = (radius a)^2 * 3 →
  a = 32 ∨ a = 32/11 := by
  sorry

#check chord_length_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_condition_l347_34787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_focus_coincidence_l347_34779

/-- Represents a parabola of the form y = ax² -/
structure Parabola where
  a : ℝ

/-- Represents a hyperbola of the form y²/3 - x² = 1 -/
structure Hyperbola

/-- The directrix of a parabola -/
noncomputable def directrix (p : Parabola) : ℝ := -1 / (4 * p.a)

/-- The focus of a parabola -/
noncomputable def parabola_focus (p : Parabola) : ℝ := 1 / (4 * p.a)

/-- The foci of the hyperbola y²/3 - x² = 1 -/
def hyperbola_foci : Set ℝ := {-2, 2}

/-- 
  If the directrix of the parabola y = ax² is y = 2, 
  then the focus of this parabola coincides with one of the foci of the hyperbola y²/3 - x² = 1
-/
theorem parabola_hyperbola_focus_coincidence (p : Parabola) : 
  directrix p = 2 → parabola_focus p ∈ hyperbola_foci := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_focus_coincidence_l347_34779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_erdos_series_permutation_theorem_l347_34772

/-- Given a sequence (a_k) where 0 < a_k < 1 for all k, 
    the existence of a permutation π for every x ∈ (0, 1) such that 
    x = ∑(k=1 to ∞) a_(π(k)) / 2^k is equivalent to inf a_k = 0 and sup a_k = 1 -/
theorem erdos_series_permutation_theorem (a : ℕ → ℝ) 
    (h : ∀ k, 0 < a k ∧ a k < 1) : 
  (∀ x, 0 < x → x < 1 → ∃ π : ℕ → ℕ, Function.Bijective π ∧ x = ∑' k, a (π k) / (2 ^ k : ℝ)) ↔ 
  (∀ ε > 0, ∃ k, a k < ε) ∧ (∀ ε > 0, ∃ k, a k > 1 - ε) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_erdos_series_permutation_theorem_l347_34772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_amount_is_800_l347_34754

/-- Calculates the final amount after simple interest is applied -/
noncomputable def final_amount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time / 100)

/-- Proves that the initial amount is 800 given the conditions -/
theorem initial_amount_is_800 
  (initial_amount rate : ℝ)
  (h1 : final_amount initial_amount rate 3 = 956)
  (h2 : final_amount initial_amount (rate + 4) 3 = 1052) :
  initial_amount = 800 := by
  sorry

#check initial_amount_is_800

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_amount_is_800_l347_34754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_value_l347_34757

def sequenceList : List ℕ := [2, 4, 8, 14, 32]

def difference_pattern (seq : List ℕ) : Prop :=
  ∀ i, i + 1 < seq.length → seq[i+1]! - seq[i]! = 2 * (i + 1)

theorem fifth_term_value (seq : List ℕ) (h : difference_pattern seq) :
  ∃ x : ℕ, x = 22 ∧ (seq.take 4 ++ [x] ++ seq.drop 4) = [2, 4, 8, 14, 22, 32] :=
by
  sorry

#eval sequenceList

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_value_l347_34757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_vertices_l347_34776

/-- The distance between the vertices of the hyperbola 16x^2 - 32x - y^2 + 10y + 19 = 0 -/
noncomputable def hyperbola_vertex_distance : ℝ :=
  Real.sqrt 7

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  16 * x^2 - 32 * x - y^2 + 10 * y + 19 = 0

theorem distance_between_vertices :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola_equation x₁ y₁ ∧
    hyperbola_equation x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = hyperbola_vertex_distance^2 ∧
    ∀ (x y : ℝ), hyperbola_equation x y →
      (x - x₁)^2 + (y - y₁)^2 ≤ hyperbola_vertex_distance^2 ∧
      (x - x₂)^2 + (y - y₂)^2 ≤ hyperbola_vertex_distance^2 :=
by
  sorry

#check distance_between_vertices

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_vertices_l347_34776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annas_spending_difference_l347_34769

/-- Represents the exchange rate from USD to LCU -/
structure ExchangeRate where
  rate : ℝ
  pos : rate > 0

/-- Calculates the difference in USD between lunch and breakfast costs for a given day -/
noncomputable def dailyDifference (lunch_cost breakfast_cost : ℝ) (exchange_rate : ExchangeRate) : ℝ :=
  (lunch_cost - breakfast_cost) / exchange_rate.rate

/-- The problem statement -/
theorem annas_spending_difference 
  (day1_rate : ExchangeRate)
  (day2_rate : ExchangeRate)
  (h1 : day1_rate.rate = 1.20)
  (h2 : day2_rate.rate = 1.25)
  (lunch_cost : ℝ)
  (breakfast_cost : ℝ)
  (h3 : lunch_cost = 9.282)
  (h4 : breakfast_cost = 2.76) :
  (dailyDifference lunch_cost breakfast_cost day1_rate) - 
  (dailyDifference lunch_cost breakfast_cost day2_rate) = 0.2174 := by
  sorry

#eval Float.toString ((9.282 - 2.76) / 1.20 - (9.282 - 2.76) / 1.25)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annas_spending_difference_l347_34769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_multiples_of_5_and_7_l347_34797

theorem three_digit_multiples_of_5_and_7 : 
  let multiples := { n : ℕ | 100 ≤ n ∧ n < 1000 ∧ 5 ∣ n ∧ 7 ∣ n }
  Finset.card (Finset.filter (λ n => 100 ≤ n ∧ n < 1000 ∧ 5 ∣ n ∧ 7 ∣ n) (Finset.range 1000)) = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_multiples_of_5_and_7_l347_34797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_space_partition_exists_l347_34792

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a unit cube in 3D space -/
structure UnitCube where
  corner : Point3D

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  vertices : Fin 4 → Point3D

/-- Represents a regular octahedron -/
structure RegularOctahedron where
  vertices : Fin 6 → Point3D

/-- Represents a partition of 3D space -/
structure SpacePartition where
  tetrahedra : Set RegularTetrahedron
  octahedra : Set RegularOctahedron

/-- Function to decompose space into unit cubes -/
noncomputable def decomposeIntoCubes : Set UnitCube := sorry

/-- Function to color cubes in a checkerboard pattern -/
noncomputable def colorCubesCheckerboard (cubes : Set UnitCube) : UnitCube → Bool := sorry

/-- Function to inscribe a tetrahedron in a cube -/
noncomputable def inscribeTetrahedron (cube : UnitCube) (color : Bool) : RegularTetrahedron := sorry

/-- Function to form octahedra from remaining space -/
noncomputable def formOctahedra (cubes : Set UnitCube) : Set RegularOctahedron := sorry

/-- Theorem stating that 3D space can be partitioned into regular tetrahedra and octahedra -/
theorem space_partition_exists : ∃ (p : SpacePartition), 
  (∀ (x y z : ℝ), ∃ (t : RegularTetrahedron) (o : RegularOctahedron), 
    (t ∈ p.tetrahedra ∧ (∃ i : Fin 4, (t.vertices i).x = x ∧ (t.vertices i).y = y ∧ (t.vertices i).z = z)) ∨ 
    (o ∈ p.octahedra ∧ (∃ i : Fin 6, (o.vertices i).x = x ∧ (o.vertices i).y = y ∧ (o.vertices i).z = z))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_space_partition_exists_l347_34792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_satisfied_by_nine_l347_34705

/-- The sum of an infinite geometric series with first term a and common ratio r -/
noncomputable def geometricSum (a r : ℝ) : ℝ := a / (1 - r)

/-- The left-hand side of the equation -/
noncomputable def leftHandSide : ℝ := (geometricSum 1 (1/3)) * (geometricSum 1 (-1/3))

/-- The right-hand side of the equation -/
noncomputable def rightHandSide (y : ℝ) : ℝ := geometricSum 1 (1/y)

/-- The theorem stating that y = 9 satisfies the equation -/
theorem equation_satisfied_by_nine : leftHandSide = rightHandSide 9 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_satisfied_by_nine_l347_34705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_count_l347_34765

-- Define the type for integer points on the circle
def CirclePoint : Type := { p : ℤ × ℤ // p.1^2 + p.2^2 = 25 }

-- Define the condition for a line to pass through a point on the circle
def LinePassesThrough (a b : ℝ) (p : CirclePoint) : Prop :=
  a * p.val.1 + b * p.val.2 = 1

-- Define the main theorem
theorem circle_line_intersection_count :
  ∃ (S : Finset (ℝ × ℝ)), 
    (∀ (a b : ℝ), (a, b) ∈ S ↔ 
      ∃ (p : CirclePoint), LinePassesThrough a b p) ∧
    Finset.card S = 72 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_count_l347_34765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_masha_time_difference_l347_34762

/-- Represents the time taken by a runner to complete the race -/
structure RunTime where
  time : ℝ

/-- Petya's running speed relative to Kolya -/
def petyaToKolyaSpeedRatio : ℝ := 2

/-- Petya's running speed relative to Masha -/
def petyaToMashaSpeedRatio : ℝ := 3

/-- Time difference between Petya and Kolya finishing the race -/
def petyaKolyaTimeDifference : ℝ := 12

/-- Proves that Petya finishes 24 seconds before Masha given the conditions -/
theorem petya_masha_time_difference :
  ∀ (t_p t_k t_m : RunTime),
    t_k.time = petyaToKolyaSpeedRatio * t_p.time →
    t_m.time = petyaToMashaSpeedRatio * t_p.time →
    t_k.time = t_p.time + petyaKolyaTimeDifference →
    t_m.time - t_p.time = 24 := by
  sorry

#check petya_masha_time_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_masha_time_difference_l347_34762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_60_properties_l347_34755

/-- A right triangle with one angle of 60 degrees --/
structure RightTriangle60 where
  /-- Length of the side adjacent to the 60-degree angle --/
  df : ℝ
  /-- df is positive --/
  df_pos : df > 0

/-- Calculate the radius of the incircle for a RightTriangle60 --/
noncomputable def incircleRadius (t : RightTriangle60) : ℝ :=
  3 * (Real.sqrt 3 - 1) * t.df / 6

/-- Calculate the circumference of the circumscribed circle for a RightTriangle60 --/
noncomputable def circumcircleCircumference (t : RightTriangle60) : ℝ :=
  2 * Real.pi * t.df

/-- Main theorem stating the properties of a specific RightTriangle60 --/
theorem right_triangle_60_properties :
  let t : RightTriangle60 := ⟨6, by norm_num⟩
  incircleRadius t = 3 * (Real.sqrt 3 - 1) ∧
  circumcircleCircumference t = 12 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_60_properties_l347_34755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_perimeter_l347_34711

noncomputable section

/-- A right triangle with legs a and b -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  a_pos : 0 < a
  b_pos : 0 < b

/-- The volume of the cone formed by rotating the triangle around its larger leg -/
def coneVolume (t : RightTriangle) : ℝ := (1/3) * Real.pi * t.a^2 * t.b

/-- The radius of the inscribed circle -/
def inRadius (t : RightTriangle) : ℝ := 
  (t.a * t.b) / (t.a + t.b + Real.sqrt (t.a^2 + t.b^2))

/-- The radius of the circumscribed circle -/
def circumRadius (t : RightTriangle) : ℝ := 
  Real.sqrt (t.a^2 + t.b^2) / 2

/-- The perimeter of the triangle -/
def perimeter (t : RightTriangle) : ℝ := 
  t.a + t.b + Real.sqrt (t.a^2 + t.b^2)

/-- The main theorem -/
theorem right_triangle_perimeter 
  (t : RightTriangle) 
  (h1 : coneVolume t = 100 * Real.pi)
  (h2 : 2 * inRadius t + 2 * circumRadius t = 17) :
  perimeter t = 30 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_perimeter_l347_34711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l347_34700

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 - x) / (x - 1)

-- Define the domain of f
def domain_f : Set ℝ := {x | x ≤ 4 ∧ x ≠ 1}

-- Theorem stating that domain_f is the correct domain for f
theorem domain_of_f : 
  ∀ x : ℝ, x ∈ domain_f ↔ (∃ y : ℝ, f x = y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l347_34700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basic_computer_price_proof_l347_34751

/-- The price of the basic computer and printer -/
noncomputable def total_price : ℝ := 2500

/-- The price increment for the first enhanced computer -/
noncomputable def price_increment : ℝ := 800

/-- The fraction of the total cost before tax that the printer represents -/
noncomputable def printer_fraction : ℝ := 1/5

/-- The tax rate for the first enhanced computer scenario -/
noncomputable def tax_rate : ℝ := 0.05

/-- The price of the basic computer before tax -/
noncomputable def basic_computer_price : ℝ := 1184.13

theorem basic_computer_price_proof :
  let enhanced_price := basic_computer_price + price_increment
  let total_before_tax := enhanced_price + printer_fraction * enhanced_price
  total_price = (1 + tax_rate) * total_before_tax :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_basic_computer_price_proof_l347_34751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_removed_edge_distance_range_all_distances_achievable_l347_34734

/-- Represents a regular octahedron -/
structure RegularOctahedron :=
  (side_length : ℝ)
  (side_length_pos : 0 < side_length)

/-- Represents the configuration of a regular octahedron with two adjacent faces removed -/
structure OctahedronWithRemovedFaces extends RegularOctahedron :=
  (removed_edge_distance : ℝ)
  (distance_in_range : 0 ≤ removed_edge_distance ∧ removed_edge_distance ≤ 2 * side_length)

/-- The main theorem stating the range of possible distances between the endpoints of the removed edge -/
theorem removed_edge_distance_range (octahedron : OctahedronWithRemovedFaces) :
  0 ≤ octahedron.removed_edge_distance ∧ octahedron.removed_edge_distance ≤ 2 * octahedron.side_length :=
octahedron.distance_in_range

/-- Every distance within the range is achievable -/
theorem all_distances_achievable :
  ∀ (d : ℝ) (h : 0 ≤ d ∧ d ≤ 2 * 10),
  ∃ (config : OctahedronWithRemovedFaces),
  config.side_length = 10 ∧ config.removed_edge_distance = d :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_removed_edge_distance_range_all_distances_achievable_l347_34734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_proof_l347_34746

/-- Calculates the principal amount given the final sum, interest rate, compounding frequency, and time period. -/
noncomputable def calculate_principal (final_sum : ℝ) (interest_rate : ℝ) (compounds_per_year : ℝ) (years : ℝ) : ℝ :=
  final_sum / ((1 + interest_rate / compounds_per_year) ^ (compounds_per_year * years))

/-- Proves that the principal amount is approximately 673,014.35 given the problem conditions. -/
theorem principal_amount_proof :
  let final_sum : ℝ := 1000000
  let interest_rate : ℝ := 0.08
  let compounds_per_year : ℝ := 4
  let years : ℝ := 5
  let calculated_principal := calculate_principal final_sum interest_rate compounds_per_year years
  ∃ ε > 0, |calculated_principal - 673014.35| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_proof_l347_34746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hotel_flat_fee_l347_34710

/-- Proves that the flat fee for the first night is $56.25 given the conditions -/
theorem hotel_flat_fee (f n : ℚ) : 
  f + 3 * n = 195 →
  f + 7 * n = 380 →
  f = 56.25 := by
  intros h1 h2
  -- The proof steps would go here
  sorry

#check hotel_flat_fee

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hotel_flat_fee_l347_34710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_perpendicular_lines_l347_34759

-- Define a structure for a 3D space
structure Space3D where
  -- Add necessary fields here
  dummy : Unit

-- Define a line in 3D space
structure Line where
  -- Add necessary fields here
  dummy : Unit

-- Define a plane in 3D space
structure Plane where
  -- Add necessary fields here
  dummy : Unit

-- Define perpendicularity between a line and a plane
def is_perpendicular (l : Line) (α : Plane) : Prop :=
  sorry

-- Define a set of lines in a plane that are perpendicular to a given line
def perpendicular_lines_in_plane (l : Line) (α : Plane) : Set Line :=
  sorry

-- The main theorem
theorem infinitely_many_perpendicular_lines
  (l : Line) (α : Plane) (h : ¬is_perpendicular l α) :
  Set.Infinite (perpendicular_lines_in_plane l α) :=
by
  sorry

#check infinitely_many_perpendicular_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_perpendicular_lines_l347_34759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_equation_solution_l347_34788

-- Define the diamond operation
noncomputable def diamond (a b : ℝ) : ℝ := a * Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- State the theorem
theorem diamond_equation_solution :
  ∃ h : ℝ, diamond 2 h = 8 ∧ h = 12 :=
by
  -- We'll use 12 as the value of h
  use 12
  
  -- Split the goal into two parts
  constructor
  
  -- Prove that diamond 2 12 = 8
  · have h1 : Real.sqrt (12 + Real.sqrt (12 + Real.sqrt (12 + Real.sqrt 12))) = 4 := by
      sorry -- This step requires numerical approximation, which is challenging in Lean
    calc
      diamond 2 12 = 2 * Real.sqrt (12 + Real.sqrt (12 + Real.sqrt (12 + Real.sqrt 12))) := rfl
      _ = 2 * 4 := by rw [h1]
      _ = 8 := by norm_num

  -- Prove that h = 12
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_equation_solution_l347_34788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_sum_l347_34767

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ
  f : ℝ → ℝ := λ x => (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)

/-- The vertex of a quadratic function -/
noncomputable def vertex (q : QuadraticFunction) : ℝ × ℝ :=
  (-(q.b : ℝ) / (2 * (q.a : ℝ)), q.f (-(q.b : ℝ) / (2 * (q.a : ℝ))))

theorem quadratic_sum (q : QuadraticFunction) 
  (h1 : vertex q = (2, -3))
  (h2 : q.f 3 = 0) :
  q.a - q.b + q.c = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_sum_l347_34767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_balls_count_l347_34733

/-- Represents the number of green balls in the bag -/
def G : ℕ := 7

/-- The probability of drawing two balls of the same color -/
def prob_same_color : ℝ := 0.46153846153846156

/-- The total number of balls in the bag -/
def total_balls : ℕ := 2 * G

theorem green_balls_count :
  (((G : ℝ) - 1) / ((total_balls : ℝ) - 1) = prob_same_color) →
  G = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_balls_count_l347_34733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_day_crew_fraction_l347_34727

theorem day_crew_fraction (D W : ℚ) (hD : D > 0) (hW : W > 0) : 
  (D * W) / ((D * W) + ((3 / 4 * D) * (3 / 4 * W))) = 16 / 25 := by
  -- Define intermediate variables
  let night_boxes_per_worker := (3 : ℚ) / 4 * D
  let night_workers := (3 : ℚ) / 4 * W
  let day_total := D * W
  let night_total := night_boxes_per_worker * night_workers
  let total := day_total + night_total

  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_day_crew_fraction_l347_34727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yacht_arrangements_l347_34731

theorem yacht_arrangements (n : ℕ) (k : ℕ) : 
  n = 5 → k = 2 → (Nat.choose n k) * (Nat.factorial k) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yacht_arrangements_l347_34731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_factors_of_3_pow_12_minus_1_l347_34715

theorem two_digit_factors_of_3_pow_12_minus_1 : 
  (Finset.filter (fun n : ℕ => 10 ≤ n ∧ n < 100 ∧ (3^12 - 1) % n = 0) (Finset.range 100)).card = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_factors_of_3_pow_12_minus_1_l347_34715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_height_ratio_time_l347_34760

noncomputable section

-- Define the initial height of candles
def initial_height : ℝ := 1

-- Define the burn rates of candles
def burn_rate_1 : ℝ := 1 / 4
def burn_rate_2 : ℝ := 1 / 3

-- Define the height of each candle as a function of time
def height_1 (t : ℝ) : ℝ := initial_height - burn_rate_1 * t
def height_2 (t : ℝ) : ℝ := initial_height - burn_rate_2 * t

end noncomputable section

-- Theorem stating the time when the first candle is twice the height of the second
theorem candle_height_ratio_time : 
  ∃ t : ℝ, t = 12/5 ∧ height_1 t = 2 * height_2 t := by
  -- The proof goes here
  sorry

#check candle_height_ratio_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_height_ratio_time_l347_34760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_side_length_l347_34701

theorem triangle_max_side_length (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- Two sides have lengths 7 and 24
  (a = 7 ∨ b = 7 ∨ c = 7) ∧ (a = 24 ∨ b = 24 ∨ c = 24) →
  -- The sine condition holds
  Real.sin (2 * A) + Real.sin (2 * B) + Real.sin (2 * C) = 0 →
  -- The maximum length of the third side is 25
  max a (max b c) ≤ 25 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_side_length_l347_34701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_l347_34729

def expansion_sum (n : ℕ) : ℕ → ℕ
| 0 => 1
| k + 1 => expansion_sum n k + n^(k + 1)

theorem coefficient_of_x_squared : 
  (Finset.range 11).sum (λ k => Nat.choose (k + 1) 2) = Nat.choose 12 3 := by
  sorry

#eval Nat.choose 12 3  -- Should output 220

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_l347_34729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l347_34771

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + abs x) - 1 / (1 + x^2)

theorem f_inequality (x : ℝ) : f x > f (2*x - 1) ↔ x ∈ Set.Ioo (1/3) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l347_34771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_prime_rep_441_l347_34780

/-- Base prime representation of a natural number up to 7 -/
def BasePrimeRep (n : ℕ) : Fin 4 → ℕ :=
  fun i => match i with
    | 0 => (Nat.factorization n 2)
    | 1 => (Nat.factorization n 3)
    | 2 => (Nat.factorization n 5)
    | 3 => (Nat.factorization n 7)

theorem base_prime_rep_441 : BasePrimeRep 441 = ![0, 2, 0, 2] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_prime_rep_441_l347_34780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_sales_is_88_l347_34725

noncomputable def monthly_sales : List ℚ := [110, 90, 50, 130, 100, 60]
def discount_rate : ℚ := 1/5

noncomputable def average_monthly_sales (sales : List ℚ) (discount : ℚ) : ℚ :=
  let adjusted_last_sale := sales.getLast! * (1 - discount)
  let total_sales := sales.sum - sales.getLast! + adjusted_last_sale
  total_sales / sales.length

theorem average_sales_is_88 :
  average_monthly_sales monthly_sales discount_rate = 88 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_sales_is_88_l347_34725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l347_34775

open Real

/-- The curve C defined by the equation (x - arcsin α)(x - arccos α) + (y - arcsin α)(y + arccos α) = 0 -/
def C (α : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - arcsin α) * (p.1 - arccos α) + (p.2 - arcsin α) * (p.2 + arccos α) = 0}

/-- The line x = π/4 -/
def L : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = π/4}

/-- The chord length function -/
noncomputable def chordLength (α : ℝ) : ℝ :=
  let y₁ := arcsin α
  let y₂ := -arccos α
  |y₁ - y₂|

theorem min_chord_length :
  ∃ (d : ℝ), d = π/2 ∧ ∀ (α : ℝ), chordLength α ≥ d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l347_34775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_balance_theorem_l347_34741

noncomputable def initial_deposit : ℝ := 100

noncomputable def interest_rate (year : ℕ) (balance : ℝ) : ℝ :=
  if year ≤ 3 then 0.1
  else if balance ≤ 300 then 0.08
  else 0.12

noncomputable def annual_deposit (year : ℕ) (balance : ℝ) : ℝ :=
  if year ≤ 2 then 10
  else if balance < 250 then 25
  else 15

noncomputable def balance_after_year (year : ℕ) (prev_balance : ℝ) : ℝ :=
  let interest := prev_balance * interest_rate year prev_balance
  let deposit := annual_deposit year prev_balance
  prev_balance + interest + deposit

noncomputable def final_balance : ℝ :=
  let year1 := balance_after_year 1 initial_deposit
  let year2 := balance_after_year 2 year1
  let year3 := balance_after_year 3 year2
  let year4 := balance_after_year 4 year3
  balance_after_year 5 year4

theorem final_balance_theorem : 
  (⌊final_balance * 100⌋ : ℝ) / 100 = 245.86 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_balance_theorem_l347_34741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_bc_length_l347_34713

noncomputable section

-- Define the trapezoid ABCD
def Trapezoid (A B C D : ℝ × ℝ) : Prop :=
  ∃ (h : ℝ), h > 0 ∧ 
  (C.2 - B.2 = h) ∧ 
  (D.2 - A.2 = h) ∧
  (B.1 < C.1) ∧ (A.1 < D.1)

-- Define the area of a trapezoid
noncomputable def TrapezoidArea (A B C D : ℝ × ℝ) : ℝ :=
  let h := C.2 - B.2
  let a := D.1 - A.1
  let b := C.1 - B.1
  (a + b) * h / 2

-- Define the length of a line segment
noncomputable def SegmentLength (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

theorem trapezoid_bc_length 
  (A B C D : ℝ × ℝ) 
  (h_trap : Trapezoid A B C D)
  (h_area : TrapezoidArea A B C D = 272)
  (h_altitude : C.2 - B.2 = 10)
  (h_AB : SegmentLength A B = 12)
  (h_CD : SegmentLength C D = 22) :
  SegmentLength B C = (272 - 5 * Real.sqrt 44 - 5 * Real.sqrt 384) / 10 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_bc_length_l347_34713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_min_at_6_l347_34721

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ := λ n => a₁ + (n - 1) * d

noncomputable def S_n (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_min_at_6 (d : ℝ) :
  let a₁ := -20
  let a_n := arithmetic_sequence a₁ d
  let sum := S_n a₁ d
  (∀ n : ℕ, sum n ≥ sum 6) → (10 / 3 < d ∧ d < 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_min_at_6_l347_34721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_four_l347_34766

/-- The length of the chord intercepted on a circle by a line -/
def chord_length (circle_eq : ℝ → ℝ → Prop) (line_eq : ℝ → ℝ → Prop) : ℝ :=
  sorry

/-- The circle equation x^2 + y^2 - 2x - 4y = 0 -/
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y = 0

/-- The line equation x + 2y - 5 + √5 = 0 -/
def line_eq (x y : ℝ) : Prop :=
  x + 2*y - 5 + Real.sqrt 5 = 0

theorem chord_length_is_four :
  chord_length circle_eq line_eq = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_four_l347_34766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l347_34761

-- Define the original expression
noncomputable def original_expr := (Real.sqrt 18 + Real.sqrt 2) / (Real.sqrt 3 - Real.sqrt 2)

-- Define the rationalized expression
noncomputable def rationalized_expr := 4 * Real.sqrt 6 + 8

-- Theorem statement
theorem rationalize_denominator :
  ∃ (x : ℝ), x = original_expr ∧ x = rationalized_expr := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l347_34761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_christmas_tree_count_l347_34738

structure Box where
  tinsel : ℕ
  trees : ℕ
  snow_globes : ℕ

theorem christmas_tree_count (T : ℕ) : 
  (∀ box : Box, box.tinsel = 4 ∧ box.trees = T ∧ box.snow_globes = 5) →
  (12 : ℕ) * (4 + T + 5) = 120 →
  T = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_christmas_tree_count_l347_34738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l347_34758

def A : Set ℤ := {x : ℤ | x ≥ 1 ∧ x ≤ 5 ∧ x % 2 = 1}

def B : Set ℤ := {-3, 2, 3}

theorem union_of_A_and_B :
  A ∪ B = {-3, 1, 2, 3, 5} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l347_34758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_at_fixed_distance_l347_34786

-- Define a circle in 2D plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point in 2D plane
def Point := ℝ × ℝ

-- Function to calculate distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem max_points_at_fixed_distance 
  (D : Circle) (Q : Point) (r : ℝ) 
  (h1 : distance Q D.center > D.radius) -- Q is outside D
  (h2 : r > 0) -- fixed distance is positive
  : 
  (∃ (n : ℕ), n ≤ 2 ∧ 
    (∀ (S : Finset Point), 
      (∀ p ∈ S, distance p Q = r ∧ distance p D.center = D.radius) → 
      S.card ≤ n) ∧
    (∃ S : Finset Point, 
      (∀ p ∈ S, distance p Q = r ∧ distance p D.center = D.radius) ∧ 
      S.card = n)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_at_fixed_distance_l347_34786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_3_implies_expression_equals_2_l347_34793

theorem tan_alpha_3_implies_expression_equals_2 (α : ℝ) (h : Real.tan α = 3) :
  (Real.sin (α - π) + Real.cos (π - α)) / (Real.sin (π / 2 - α) + Real.cos (π / 2 + α)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_3_implies_expression_equals_2_l347_34793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_properties_l347_34740

theorem divisibility_properties (a n : ℕ) (ha : a > 1) :
  (∃ k₁ k₂ : ℕ, (a + 1)^(a^n) - 1 = k₁ * a^(n + 1) ∧ (a - 1)^(a^n) + 1 = k₂ * a^(n + 1)) ∧
  (∀ m₁ m₂ : ℕ, (a + 1)^(a^n) - 1 ≠ m₁ * a^(n + 2) ∧ (a - 1)^(a^n) + 1 ≠ m₂ * a^(n + 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_properties_l347_34740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_ab_solutions_system_c_solution_l347_34752

-- Define the system of equations for parts (a) and (b)
def system_ab (a x y : ℝ) : Prop :=
  (a * x + y = a^2) ∧ (x + a * y = 1)

-- Theorem for part (a) and (b)
theorem system_ab_solutions (a : ℝ) :
  (a = -1 → ¬∃ x y, system_ab a x y) ∧
  (a = 1 → ∀ x, ∃ y, system_ab a x y) :=
by
  sorry

-- Define the system of equations for part (c)
def system_c (a x y z : ℝ) : Prop :=
  (a * x + y + z = 1) ∧ (x + a * y + z = a) ∧ (x + y + a * z = a^2)

-- Theorem for part (c)
theorem system_c_solution (a : ℝ) (h : a ≠ 1) :
  ∃! x y z, system_c a x y z :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_ab_solutions_system_c_solution_l347_34752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_value_increase_factor_l347_34770

/-- The factor by which the value of games increased -/
noncomputable def value_increase_factor (initial_cost : ℝ) (sold_percentage : ℝ) (sale_price : ℝ) : ℝ :=
  sale_price / (sold_percentage * initial_cost)

/-- Theorem: Given the problem conditions, the value increase factor is 3 -/
theorem game_value_increase_factor :
  value_increase_factor 200 0.4 240 = 3 := by
  -- Unfold the definition of value_increase_factor
  unfold value_increase_factor
  -- Simplify the expression
  simp
  -- Perform the numerical calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_value_increase_factor_l347_34770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_circumradius_ratio_not_always_equal_l347_34784

/-- Represents an isosceles triangle with two equal sides and one different side -/
structure IsoscelesTriangle where
  equalSide : ℝ
  diffSide : ℝ
  hDiff : equalSide ≠ diffSide

/-- The perimeter of an isosceles triangle -/
noncomputable def perimeter (t : IsoscelesTriangle) : ℝ := 2 * t.equalSide + t.diffSide

/-- The circumradius of an isosceles triangle, assuming the angle between equal sides doesn't affect it significantly -/
noncomputable def circumradius (t : IsoscelesTriangle) (α : ℝ) : ℝ := t.equalSide / (2 * Real.sin α)

/-- Theorem stating that the ratio of perimeters is not always equal to the ratio of circumradii for two isosceles triangles with equal angles -/
theorem perimeter_circumradius_ratio_not_always_equal :
  ¬ ∀ (t1 t2 : IsoscelesTriangle) (α : ℝ), 
    perimeter t1 / perimeter t2 = circumradius t1 α / circumradius t2 α :=
by
  sorry

#check perimeter_circumradius_ratio_not_always_equal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_circumradius_ratio_not_always_equal_l347_34784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stretched_shifted_sine_form_l347_34749

/-- Given a function f, stretching its horizontal coordinates by a factor of 2
    and shifting it right by π/6 results in g(x) = sin(2x) --/
def is_stretched_and_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x, Real.sin (2 * x) = f ((x - Real.pi/6) / 2)

/-- Theorem stating that if f satisfies the stretch and shift condition,
    then f(x) = sin(4x + π/3) --/
theorem stretched_shifted_sine_form (f : ℝ → ℝ) 
    (h : is_stretched_and_shifted f) : 
    ∀ x, f x = Real.sin (4 * x + Real.pi/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stretched_shifted_sine_form_l347_34749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curve_l347_34785

-- Define the function representing the curve
noncomputable def f (x : ℝ) : ℝ := 1 / x

-- Define the boundaries
noncomputable def left_boundary : ℝ := 1 / Real.exp 1
noncomputable def right_boundary : ℝ := Real.exp 1

-- State the theorem
theorem area_enclosed_by_curve : 
  (∫ x in left_boundary..right_boundary, f x) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curve_l347_34785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_collinear_l347_34707

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Calculates the intersection point of two lines -/
noncomputable def intersectionPoint (l1 l2 : Line) : Point :=
  { x := (l2.intercept - l1.intercept) / (l1.slope - l2.slope),
    y := l1.slope * (l2.intercept - l1.intercept) / (l1.slope - l2.slope) + l1.intercept }

/-- Checks if three points are collinear -/
def areCollinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p2.x) = (p3.y - p2.y) * (p2.x - p1.x)

theorem intersection_points_collinear :
  let A : Point := { x := 0, y := 0 }
  let B : Point := { x := 0, y := 2 }
  let l1 : Line := { slope := Real.tan (45 * π / 180), intercept := 0 }
  let l2 : Line := { slope := Real.tan (75 * π / 180), intercept := 0 }
  let l3 : Line := { slope := -Real.tan (45 * π / 180), intercept := 2 }
  let l4 : Line := { slope := -Real.tan (75 * π / 180), intercept := 2 }
  let p1 := intersectionPoint l1 l3
  let p2 := intersectionPoint l2 l4
  let p3 : Point := { x := 5, y := 1 }  -- An arbitrary point on the line y = 1
  areCollinear p1 p2 p3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_collinear_l347_34707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l347_34722

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3)

theorem f_properties :
  (∀ x : ℝ, f (Real.pi/3 + x) = -f (Real.pi/3 - x)) ∧
  (∃ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi/2) → f x + a ≥ Real.sqrt 3) ∧
             (∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi/2) ∧ f x + a = Real.sqrt 3) ∧
             a = 2 * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l347_34722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_3_5_7_count_l347_34742

theorem divisible_by_3_5_7_count : 
  (Finset.filter (fun n => n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0) (Finset.range 1000)).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_3_5_7_count_l347_34742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmers_pass_count_l347_34796

/-- Represents a swimmer with a given speed -/
structure Swimmer where
  speed : ℚ
  deriving Repr

/-- Represents the swimming scenario -/
structure SwimmingScenario where
  poolLength : ℚ
  swimmer1 : Swimmer
  swimmer2 : Swimmer
  totalTime : ℚ
  deriving Repr

/-- Calculates the number of times swimmers pass each other -/
def countPasses (scenario : SwimmingScenario) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem swimmers_pass_count (scenario : SwimmingScenario) 
  (h1 : scenario.poolLength = 100)
  (h2 : scenario.swimmer1.speed = 4)
  (h3 : scenario.swimmer2.speed = 5)
  (h4 : scenario.totalTime = 15 * 60) : -- 15 minutes in seconds
  countPasses scenario = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmers_pass_count_l347_34796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_periodic_sin_plus_sin_ax_l347_34774

/-- A function f is periodic with period T if f(x + T) = f(x) for all x -/
def IsPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

/-- The function f(x) = sin x + sin(ax) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x + Real.sin (a * x)

/-- Theorem: f(x) = sin x + sin(ax) is not periodic when a is irrational -/
theorem not_periodic_sin_plus_sin_ax (a : ℝ) (h : Irrational a) :
  ¬ ∃ T : ℝ, T ≠ 0 ∧ IsPeriodic (f a) T := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_periodic_sin_plus_sin_ax_l347_34774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_inequality_l347_34799

theorem fifth_inequality : 
  1 + (1 / 2^2 : ℝ) + (1 / 3^2 : ℝ) + (1 / 4^2 : ℝ) + (1 / 5^2 : ℝ) + (1 / 6^2 : ℝ) < 11 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_inequality_l347_34799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_value_max_area_value_l347_34739

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  t.b * Real.cos t.A = Real.sqrt 3 * Real.sin t.B

def condition2 (t : Triangle) : Prop :=
  t.b + t.c = 4

-- Define the area of the triangle
noncomputable def area (t : Triangle) : ℝ :=
  (1 / 2) * t.b * t.c * Real.sin t.A

-- Theorem 1: If condition1 holds, then A = π/6
theorem angle_A_value (t : Triangle) (h : condition1 t) : t.A = π / 6 := by
  sorry

-- Theorem 2: If condition2 holds, then the maximum area is 1
theorem max_area_value (t : Triangle) (h : condition2 t) : 
  ∃ (max_area : ℝ), max_area = 1 ∧ ∀ (s : ℝ), s = area t → s ≤ max_area := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_value_max_area_value_l347_34739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_tangent_circle_radius_l347_34798

/-- A circle is tangent to the hypotenuse and the extensions of the legs of a right triangle. -/
def is_radius_of_circle_tangent_to_hypotenuse_and_legs (r a b c : ℝ) : Prop :=
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = c ∧ x + r = b ∧ y + r = a

/-- Given a right triangle with sides a, b, and c (c being the hypotenuse),
    and a circle with radius r tangent to the hypotenuse and the extensions of the legs of the triangle,
    prove that r = (a + b + c) / 2. -/
theorem right_triangle_tangent_circle_radius
  (a b c r : ℝ) 
  (h_right_triangle : a^2 + b^2 = c^2)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0)
  (h_tangent : is_radius_of_circle_tangent_to_hypotenuse_and_legs r a b c) :
  r = (a + b + c) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_tangent_circle_radius_l347_34798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_properties_l347_34708

-- Define the sequence of functions g_n
noncomputable def g : ℕ → (ℝ → ℝ)
  | 0 => fun x => Real.sqrt (2 - x)  -- Adding case for 0 to cover all natural numbers
  | (n+1) => fun x => g n (Real.sqrt ((n+1)^2 + (n+1) - x))

-- Define the domain of a function
def domain (f : ℝ → ℝ) : Set ℝ :=
  {x | ∃ y, f x = y}

-- Statement to prove
theorem g_domain_properties :
  (∃ N : ℕ, (∀ n > N, domain (g n) = ∅) ∧
             domain (g N) ≠ ∅ ∧
             domain (g (N+1)) = ∅) ∧
  (domain (g 5) = {-370}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_properties_l347_34708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_decreasing_l347_34756

noncomputable def f (x : ℝ) : ℝ := 2^(-x) - 2^x

theorem f_odd_and_decreasing : 
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x > f y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_decreasing_l347_34756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_K_after_reaction_l347_34753

/-- Molar mass of KBr in g/mol -/
noncomputable def molar_mass_KBr : ℝ := 119

/-- Molar mass of KBrO3 in g/mol -/
noncomputable def molar_mass_KBrO3 : ℝ := 167

/-- Molar mass of K in g/mol -/
noncomputable def molar_mass_K : ℝ := 39

/-- Mass of KBr in the initial mixture in grams -/
noncomputable def mass_KBr : ℝ := 12

/-- Mass of KBrO3 in the initial mixture in grams -/
noncomputable def mass_KBrO3 : ℝ := 18

/-- Function to calculate moles of K from a given mass and molar mass -/
noncomputable def moles_K (mass : ℝ) (molar_mass : ℝ) : ℝ :=
  mass / molar_mass

/-- Function to calculate mass percentage -/
noncomputable def mass_percentage (mass_part : ℝ) (mass_total : ℝ) : ℝ :=
  (mass_part / mass_total) * 100

/-- Theorem stating that the mass percentage of K in the mixture after the reaction is approximately 27.118% -/
theorem mass_percentage_K_after_reaction :
  let moles_K_KBr := moles_K mass_KBr molar_mass_KBr
  let moles_K_KBrO3 := moles_K mass_KBrO3 molar_mass_KBrO3
  let total_moles_K := moles_K_KBr + moles_K_KBrO3
  let mass_K := total_moles_K * molar_mass_K
  let total_mass := mass_KBr + mass_KBrO3
  let percentage_K := mass_percentage mass_K total_mass
  abs (percentage_K - 27.118) < 0.001 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_K_after_reaction_l347_34753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_earliest_eight_different_digits_l347_34763

/-- Represents a date in DD.MM.YYYY format -/
structure Date where
  day : Nat
  month : Nat
  year : Nat
  deriving Repr

/-- Check if a date is valid -/
def Date.isValid (d : Date) : Bool :=
  d.day ≥ 1 && d.day ≤ 31 && d.month ≥ 1 && d.month ≤ 12 && d.year ≥ 1

/-- Convert a date to a list of digits -/
def Date.toDigits (d : Date) : List Nat :=
  (d.day.repr ++ d.month.repr ++ d.year.repr).toList.map (fun c => c.toNat - 48)

/-- Check if a date uses 8 different digits -/
def Date.hasEightDifferentDigits (d : Date) : Bool :=
  (d.toDigits.eraseDups).length = 8

/-- Check if a date is after another date -/
def Date.isAfter (d1 d2 : Date) : Bool :=
  d1.year > d2.year || 
  (d1.year = d2.year && d1.month > d2.month) ||
  (d1.year = d2.year && d1.month = d2.month && d1.day > d2.day)

/-- The starting date: 11.08.1999 -/
def startDate : Date := ⟨11, 8, 1999⟩

/-- The date to be proven: 17.06.2345 -/
def targetDate : Date := ⟨17, 6, 2345⟩

theorem earliest_eight_different_digits : 
  targetDate.isValid ∧
  targetDate.hasEightDifferentDigits ∧
  targetDate.isAfter startDate ∧
  ∀ d : Date, d.isValid → d.hasEightDifferentDigits → d.isAfter startDate → 
    ¬(d.isAfter targetDate) := by
  sorry

#eval targetDate.isValid
#eval targetDate.hasEightDifferentDigits
#eval targetDate.isAfter startDate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_earliest_eight_different_digits_l347_34763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b3_set_equality_l347_34764

/-- Definition of a B₃-set -/
def is_B3_set (A : Set ℝ) : Prop :=
  ∀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ,
    a₁ ∈ A → a₂ ∈ A → a₃ ∈ A → a₄ ∈ A → a₅ ∈ A → a₆ ∈ A →
    a₁ + a₂ + a₃ = a₄ + a₅ + a₆ →
    ∃ σ : Fin 3 → Fin 3, Function.Bijective σ ∧ (fun i ↦ [a₁, a₂, a₃].get i) = (fun i ↦ [a₄, a₅, a₆].get (σ i))

/-- Definition of the difference set -/
def difference_set (X : Set ℝ) : Set ℝ :=
  {d | ∃ x y, x ∈ X ∧ y ∈ X ∧ d = |x - y|}

/-- Definition of an infinite strictly increasing sequence starting with 0 -/
def is_increasing_sequence_from_zero (s : ℕ → ℝ) : Prop :=
  s 0 = 0 ∧ ∀ n : ℕ, s n < s (n + 1)

/-- The main theorem -/
theorem b3_set_equality (A B : Set ℝ) (sA sB : ℕ → ℝ) :
  A.Nonempty →
  is_B3_set A →
  (∀ n, sA n ∈ A) →
  (∀ n, sB n ∈ B) →
  is_increasing_sequence_from_zero sA →
  is_increasing_sequence_from_zero sB →
  difference_set A = difference_set B →
  A = B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b3_set_equality_l347_34764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_range_l347_34726

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 - x + 3/2) / Real.log a

theorem f_increasing_implies_a_range (a : ℝ) :
  (0 < a ∧ a < 1) →
  (∀ x y, 1 ≤ x ∧ x < y ∧ y ≤ 2 → f a x < f a y) →
  1/8 < a ∧ a ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_range_l347_34726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_tripled_sphere_volume_tripled_incorrect_statements_l347_34706

-- Define the square area function
noncomputable def square_area (s : ℝ) : ℝ := s^2

-- Define the sphere volume function
noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

-- Theorem for square area
theorem square_area_tripled (s : ℝ) :
  square_area (3 * s) ≠ 3 * square_area s := by
  sorry

-- Theorem for sphere volume
theorem sphere_volume_tripled (r : ℝ) :
  sphere_volume (3 * r) ≠ 4 * sphere_volume r := by
  sorry

-- Main theorem combining both incorrect statements
theorem incorrect_statements :
  ∃ (s r : ℝ), (square_area (3 * s) ≠ 3 * square_area s) ∧ 
               (sphere_volume (3 * r) ≠ 4 * sphere_volume r) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_tripled_sphere_volume_tripled_incorrect_statements_l347_34706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distributor_profit_is_87_point_5_percent_l347_34719

/-- Calculates the distributor's profit percentage given the commission rate,
    product cost, and observed online price. -/
noncomputable def distributor_profit_percentage (commission_rate : ℝ)
                                  (product_cost : ℝ)
                                  (online_price : ℝ) : ℝ :=
  let distributor_price := online_price / (1 - commission_rate)
  let profit := distributor_price - product_cost
  (profit / product_cost) * 100

/-- Proves that the distributor's profit percentage is 87.5% given the
    specified conditions. -/
theorem distributor_profit_is_87_point_5_percent :
  distributor_profit_percentage 0.2 20 30 = 87.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distributor_profit_is_87_point_5_percent_l347_34719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_lambda_l347_34790

/-- Given two plane vectors a and b, prove that if λa + b is perpendicular to a, then λ = -1 -/
theorem perpendicular_vector_lambda (a b : ℝ × ℝ) (lambda : ℝ) :
  a = (1, -3) →
  b = (4, -2) →
  (lambda • a + b) • a = 0 →
  lambda = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_lambda_l347_34790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l347_34781

theorem divisibility_property (n : ℕ) (h : n > 1) : 
  ∃ (S : Finset ℕ), (Finset.card S = n) ∧ 
  (∀ a b, a ∈ S → b ∈ S → a ≠ b → (a - b) ∣ (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l347_34781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l347_34791

/-- The distance between two parallel lines given by their coefficients -/
noncomputable def distance_parallel_lines (A B C1 C2 : ℝ) : ℝ :=
  |C2 - C1| / Real.sqrt (A^2 + B^2)

/-- The first line: x - y + 1 = 0 -/
def line1 : ℝ × ℝ × ℝ := (1, -1, 1)

/-- The second line: 3x - 3y + 1 = 0 -/
def line2 : ℝ × ℝ × ℝ := (3, -3, 1)

theorem distance_between_given_lines :
  distance_parallel_lines line1.1 line1.2.1 line1.2.2 (line2.2.2 / line2.1) = Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l347_34791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_sequence_has_large_number_l347_34723

def adjacent_sum (seq : List ℤ) : List ℤ :=
  List.zipWith (· + ·) seq (seq.rotate 1)

def iterate_adjacent_sum (seq : List ℤ) : ℕ → List ℤ
  | 0 => seq
  | n + 1 => iterate_adjacent_sum (adjacent_sum seq) n

def initial_sequence : List ℤ := List.replicate 13 1 ++ List.replicate 12 (-1)

theorem final_sequence_has_large_number :
  ∃ x ∈ (iterate_adjacent_sum initial_sequence 100), x > 10^20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_sequence_has_large_number_l347_34723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l347_34794

-- Define a and b
noncomputable def a : ℝ := Real.log 0.3 / Real.log 0.2
noncomputable def b : ℝ := Real.log 0.3 / Real.log 2

-- State the theorem
theorem log_inequality : a * b < a + b ∧ a + b < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l347_34794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_sine_symmetry_l347_34777

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (3 * x + φ)

theorem shifted_sine_symmetry (φ : ℝ) 
  (h1 : -π < φ ∧ φ < 0) 
  (h2 : ∀ x, f φ (x + π/12) = f φ (-x + π/12)) : 
  φ = -π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_sine_symmetry_l347_34777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l347_34720

/-- The hyperbola C with equation x²/a² - y²/b² = 1 -/
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

/-- The right focus F of the hyperbola -/
noncomputable def rightFocus (a b : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 + b^2), 0)

/-- The circle Ω with diameter being the real axis of hyperbola C -/
def circleOmega (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = a^2

/-- The slope of line FP -/
noncomputable def slopeFP (a b : ℝ) : ℝ := -b/a

/-- The point P where circle Ω intersects the asymptote in the first quadrant -/
noncomputable def pointP (a b : ℝ) : ℝ × ℝ := 
  let c := Real.sqrt (a^2 + b^2)
  (c/2, b*c/(2*a))

theorem hyperbola_asymptotes (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y : ℝ, hyperbola a b x y → 
    ∃ k : ℝ, k = 1 ∨ k = -1 ∧ y = k*x) ↔ 
  (∃ (P : ℝ × ℝ), P = pointP a b ∧ 
    circleOmega a P.1 P.2 ∧ 
    slopeFP a b = (P.2 - (rightFocus a b).2) / (P.1 - (rightFocus a b).1)) := 
  by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l347_34720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extracurricular_study_group_count_l347_34744

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem extracurricular_study_group_count :
  let total_female : ℕ := 10
  let total_male : ℕ := 5
  let group_size : ℕ := 6
  let female_in_group : ℕ := 4
  let male_in_group : ℕ := 2
  (choose total_female female_in_group) * (choose total_male male_in_group) = 2100 :=
by
  -- Unfold the definitions
  simp [choose]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#eval choose 10 4 * choose 5 2  -- This should output 2100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extracurricular_study_group_count_l347_34744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sarah_picked_more_apples_l347_34704

/-- The number of apples Sarah picked -/
noncomputable def sarah_apples : ℝ := 45.3

/-- The number of apples Jason picked -/
noncomputable def jason_apples : ℝ := 9.1

/-- The number of apples Emily picked -/
noncomputable def emily_apples : ℝ := 12.4

/-- The total number of apples picked by Jason and Emily -/
noncomputable def jason_emily_total : ℝ := jason_apples + emily_apples

/-- The difference between Sarah's apples and the total apples picked by Jason and Emily -/
noncomputable def apple_difference : ℝ := sarah_apples - jason_emily_total

/-- The percentage more apples Sarah picked compared to Jason and Emily -/
noncomputable def percentage_more : ℝ := (apple_difference / jason_emily_total) * 100

theorem sarah_picked_more_apples : 
  110.6 < percentage_more ∧ percentage_more < 110.8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sarah_picked_more_apples_l347_34704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cauchy_principal_value_sin_2x_l347_34795

/-- The Cauchy principal value of the integral of sin(2x) from negative infinity to positive infinity is zero. -/
theorem cauchy_principal_value_sin_2x : 
  ∃ (I : ℝ), (∀ ε > 0, ∃ A > 0, ∀ a ≥ A, 
    |∫ x in (-a)..a, Real.sin (2 * x) - I| < ε) ∧ I = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cauchy_principal_value_sin_2x_l347_34795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_representation_bound_l347_34714

/-- 
Represents a positive integer as the sum of non-negative integer powers of 2.
Two representations that differ only in the order of summation are considered the same.
-/
def represent (n : ℕ+) : Finset (Multiset ℕ) :=
  sorry

/-- 
The number of different representations of a positive integer n 
as the sum of non-negative integer powers of 2.
-/
def f (n : ℕ+) : ℕ :=
  (represent n).card

/-- 
For any integer n ≥ 3, 2^(n^2/4) < f(2^n) < 2^(n^2/2)
-/
theorem representation_bound (n : ℕ) (h : n ≥ 3) : 
  2^((n^2 : ℝ)/4) < (f (2^n : ℕ+) : ℝ) ∧ (f (2^n : ℕ+) : ℝ) < 2^((n^2 : ℝ)/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_representation_bound_l347_34714
