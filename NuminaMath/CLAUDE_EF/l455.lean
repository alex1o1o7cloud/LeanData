import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l455_45560

-- Define the function f(x) = log(x^2)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2)

-- State the theorem
theorem f_decreasing_interval :
  ∀ x y : ℝ, x < y ∧ x < 0 ∧ y < 0 → f y < f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l455_45560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l455_45596

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- State the theorem
theorem range_of_x (x : ℝ) : 
  (∀ a b : ℝ, a ≠ 0 → |a + b| + |a - b| ≥ |a| * f x) → 
  1 ≤ x ∧ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l455_45596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_x_period_l455_45571

theorem sin_pi_x_period : 
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), Real.sin (π * x) = Real.sin (π * (x + p))) ∧ 
  (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), Real.sin (π * x) = Real.sin (π * (x + q))) → p ≤ q) ∧
  p = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_x_period_l455_45571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_jet_set_numbers_l455_45553

def is_permutation_of_1_to_7 (n : ℕ) : Prop :=
  ∃ (d₁ d₂ d₃ d₄ d₅ d₆ d₇ : ℕ),
    n = d₁ * 1000000 + d₂ * 100000 + d₃ * 10000 + d₄ * 1000 + d₅ * 100 + d₆ * 10 + d₇ ∧
    Finset.toSet {d₁, d₂, d₃, d₄, d₅, d₆, d₇} = Finset.toSet {1, 2, 3, 4, 5, 6, 7}

def prefix_divisible (n : ℕ) : Prop :=
  ∀ k : ℕ, k ∈ Finset.range 7 → k ≠ 0 →
    (n / (10^(7-k))) % k = 0

def jet_set_number (n : ℕ) : Prop :=
  1000000 ≤ n ∧ n < 10000000 ∧
  is_permutation_of_1_to_7 n ∧
  prefix_divisible n

theorem count_jet_set_numbers :
  ∃! (s : Finset ℕ), (∀ n, n ∈ s ↔ jet_set_number n) ∧ s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_jet_set_numbers_l455_45553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_notebook_cost_l455_45521

/-- Proves that the cost of each notebook is $4 given the conditions of Rubble's purchase. -/
theorem notebook_cost (initial_amount : ℝ) (notebooks_count : ℕ) (pens_count : ℕ) 
  (pen_cost : ℝ) (remaining_amount : ℝ) : ℝ :=
by
  have h1 : initial_amount = 15 := by sorry
  have h2 : notebooks_count = 2 := by sorry
  have h3 : pens_count = 2 := by sorry
  have h4 : pen_cost = 1.5 := by sorry
  have h5 : remaining_amount = 4 := by sorry

  -- The cost of each notebook
  let notebook_cost : ℝ := (initial_amount - remaining_amount - pen_cost * pens_count) / notebooks_count

  -- Prove that notebook_cost = 4
  have h6 : notebook_cost = 4 := by sorry

  exact notebook_cost

-- Example usage (commented out to avoid compilation issues)
-- #eval notebook_cost 15 2 2 1.5 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_notebook_cost_l455_45521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_count_bound_l455_45506

/-- A set of points in the Cartesian plane -/
def PointSet := Set (ℝ × ℝ)

/-- A rectangle with vertices from a given set and sides parallel to coordinate axes -/
def IsValidRectangle (M : PointSet) (r : Set (ℝ × ℝ)) : Prop :=
  r.Finite ∧ r.ncard = 4 ∧ r ⊆ M ∧
  ∃ x₁ x₂ y₁ y₂ : ℝ, x₁ < x₂ ∧ y₁ < y₂ ∧
    r = {(x₁, y₁), (x₁, y₂), (x₂, y₁), (x₂, y₂)}

/-- The set of all valid rectangles for a given point set -/
def ValidRectangles (M : PointSet) : Set (Set (ℝ × ℝ)) :=
  {r | IsValidRectangle M r}

/-- Theorem: The number of rectangles with vertices in a set of 100 points
    and sides parallel to coordinate axes is at most 2025 -/
theorem rectangle_count_bound (M : PointSet) (h : M.ncard = 100) :
    (ValidRectangles M).ncard ≤ 2025 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_count_bound_l455_45506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_half_l455_45517

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = x / ((2x+1)(x-a)) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x / ((2*x + 1) * (x - a))

/-- If f(x) = x / ((2x+1)(x-a)) is an odd function, then a = 1/2 -/
theorem odd_function_implies_a_half :
  ∀ a : ℝ, IsOdd (f a) → a = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_half_l455_45517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l455_45531

def A : Set ℝ := {x | (2 : ℝ)^x > 1}
def B : Set ℝ := {x | |x| < 3}

theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l455_45531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_new_students_is_9_88_l455_45501

/-- The number of elementary schools in Lansing -/
noncomputable def num_schools : ℝ := 25.0

/-- The total number of new elementary students in Lansing -/
noncomputable def total_new_students : ℝ := 247.0

/-- The average number of new students per school -/
noncomputable def average_new_students : ℝ := total_new_students / num_schools

/-- Theorem stating that the average number of new students per school is 9.88 -/
theorem average_new_students_is_9_88 : 
  average_new_students = 9.88 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_new_students_is_9_88_l455_45501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_corresponding_point_l455_45526

/-- A square in a 2D plane --/
structure Square where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_square : -- Add conditions for a square
    (B.1 - A.1) = (C.1 - B.1) ∧
    (C.1 - B.1) = (D.1 - C.1) ∧
    (D.1 - C.1) = (A.1 - D.1) ∧
    (B.2 - A.2) = (C.2 - B.2) ∧
    (C.2 - B.2) = (D.2 - C.2) ∧
    (D.2 - C.2) = (A.2 - D.2)

/-- Two squares are similar --/
def similar (s1 s2 : Square) : Prop := 
  ∃ k : ℝ, k > 0 ∧ 
    (s2.B.1 - s2.A.1) = k * (s1.B.1 - s1.A.1) ∧
    (s2.B.2 - s2.A.2) = k * (s1.B.2 - s1.A.2)

/-- A point is inside a square --/
def inside (p : ℝ × ℝ) (s : Square) : Prop :=
  s.A.1 ≤ p.1 ∧ p.1 ≤ s.C.1 ∧ s.A.2 ≤ p.2 ∧ p.2 ≤ s.C.2

/-- The ratio of distances from a point to two parallel sides of a square --/
noncomputable def ratio (p : ℝ × ℝ) (s : Square) : ℝ := 
  (p.1 - s.A.1) / (s.C.1 - s.A.1)

theorem unique_corresponding_point (s1 s2 : Square) (h : similar s1 s2) :
  ∃! p : ℝ × ℝ, inside p s2 ∧ 
    ratio p s2 = ratio (p.1 * (s1.C.1 - s1.A.1) / (s2.C.1 - s2.A.1), 
                        p.2 * (s1.C.2 - s1.A.2) / (s2.C.2 - s2.A.2)) s1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_corresponding_point_l455_45526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_theorem_l455_45544

theorem log_product_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.log (x^2) / Real.log (y^4)) * (Real.log y / Real.log (x^3)) * 
  (Real.log (x^4) / Real.log (y^3)) * (Real.log (y^3) / Real.log (x^2)) * 
  (Real.log (x^3) / Real.log y) = Real.log x / Real.log y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_theorem_l455_45544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_29pi_l455_45511

-- Define the points A and B
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (8, 7)

-- Define the diameter as the distance between A and B
noncomputable def diameter : ℝ := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- Define the radius as half the diameter
noncomputable def radius : ℝ := diameter / 2

-- Define the area of the circle
noncomputable def circle_area : ℝ := Real.pi * radius^2

-- Theorem statement
theorem circle_area_is_29pi : circle_area = 29 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_29pi_l455_45511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_sum_constant_l455_45568

/-- The ellipse E defined by (x^2/25) + (y^2/9) = 1 --/
def E : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 25) + (p.2^2 / 9) = 1}

/-- The left focus of the ellipse E --/
def leftFocus : ℝ × ℝ := (-4, 0)

/-- A chord of the ellipse E passing through the left focus --/
structure Chord where
  slope : ℝ
  passes_through_focus : (Set.inter {p : ℝ × ℝ | p.2 = slope * (p.1 + 4)} E).Nonempty

/-- The length of a chord --/
noncomputable def chordLength (c : Chord) : ℝ :=
  90 * (1 + c.slope^2) / (25 * c.slope^2 + 9)

/-- The theorem to be proved --/
theorem chord_sum_constant (c1 c2 : Chord) (h : c1.slope * c2.slope = 1) :
  1 / chordLength c1 + 1 / chordLength c2 = 17 / 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_sum_constant_l455_45568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_2_8_plus_5_5_l455_45561

theorem greatest_prime_factor_of_2_8_plus_5_5 :
  (Nat.factors (2^8 + 5^5)).maximum? = some 3381 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_2_8_plus_5_5_l455_45561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_routes_theorem_example_satisfies_conditions_l455_45527

-- Define the number of cities
def num_cities : ℕ := 15

-- Define the number of airlines
def num_airlines : ℕ := 3

-- Define a type for airlines
inductive Airline : Type
| A : Airline
| B : Airline
| C : Airline

-- Define a function to represent the number of routes for each airline
def routes : Airline → ℕ := sorry

-- Define the connectivity condition
def connectivity_condition (r : Airline → ℕ) : Prop :=
  ∀ (x y : Airline), x ≠ y → r x + r y ≥ num_cities - 1

-- Theorem statement
theorem min_routes_theorem (r : Airline → ℕ) :
  connectivity_condition r →
  (r Airline.A + r Airline.B + r Airline.C ≥ 21) := by
  sorry

-- Example construction
def example_routes : Airline → ℕ
| Airline.A => 7
| Airline.B => 7
| Airline.C => 7

-- Proof that the example satisfies the conditions
theorem example_satisfies_conditions :
  connectivity_condition example_routes ∧
  (example_routes Airline.A + example_routes Airline.B + example_routes Airline.C = 21) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_routes_theorem_example_satisfies_conditions_l455_45527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_order_linear_approximation_l455_45588

noncomputable def f (x : ℝ) := x + 1/x

noncomputable def vector_ON (lambda : ℝ) (A B : ℝ × ℝ) : ℝ × ℝ :=
  (lambda * A.1 + (1 - lambda) * B.1, lambda * A.2 + (1 - lambda) * B.2)

noncomputable def vector_MN (x : ℝ) : ℝ := |f x - (x/2 + 3/2)|

theorem k_order_linear_approximation :
  ∃ (k : ℝ), k = 3/2 - Real.sqrt 2 ∧
  (∀ (x : ℝ), x ∈ Set.Icc 1 2 → vector_MN x ≤ k) ∧
  (∀ (ε : ℝ), ε > 0 → ∃ (x : ℝ), x ∈ Set.Icc 1 2 ∧ vector_MN x > k - ε) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_order_linear_approximation_l455_45588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_ticket_price_l455_45559

/-- Calculates the final price of a movie ticket after a series of percentage changes. -/
noncomputable def final_price (initial_price : ℝ) (changes : List ℝ) : ℝ :=
  changes.foldl (fun price change => price * (1 + change / 100)) initial_price

/-- Theorem stating the final price of a movie ticket after specific percentage changes. -/
theorem movie_ticket_price : 
  let initial_price : ℝ := 100
  let changes : List ℝ := [12, -5, 8, -4, 6]
  abs (final_price initial_price changes - 116.93) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_ticket_price_l455_45559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_parameter_bound_l455_45507

open Real

theorem function_inequality_implies_parameter_bound (a : ℝ) :
  (∀ p q : ℝ, 0 < p ∧ p < 1 ∧ 0 < q ∧ q < 1 ∧ p ≠ q →
    (a * log p - 2 * p^2 - (a * log q - 2 * q^2)) / (p - q) > 1) →
  a ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_parameter_bound_l455_45507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eiffel_tower_scale_model_l455_45516

/-- Represents the height of the Eiffel Tower in feet -/
noncomputable def TowerHeight : ℝ := 984

/-- Represents the height of a scale model in inches -/
noncomputable def ModelHeight : ℝ := 6

/-- Represents the number of feet of the tower that one inch of the model represents -/
noncomputable def FeetPerInch : ℝ := TowerHeight / ModelHeight

/-- Theorem stating that one inch of the model represents 164 feet of the Eiffel Tower -/
theorem eiffel_tower_scale_model :
  FeetPerInch = 164 := by
  -- Unfold the definitions
  unfold FeetPerInch TowerHeight ModelHeight
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eiffel_tower_scale_model_l455_45516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l455_45520

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - a + 1

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a (x + a) - 1

noncomputable def h (a : ℝ) (x : ℝ) : ℝ := (x - 7) / 3

theorem problem_solution (a : ℝ) (h_a_pos : a > 0) (h_a_neq_one : a ≠ 1) 
  (h_f_fixed_point : f a 3 = 2) :
  a = 3 ∧
  (∀ x, g a (h a x) = x) ∧
  (∀ x ∈ Set.Icc 1 3, (h a x + 2)^2 ≤ h a (x^2) + 5 + 2) ∧
  (∀ m : ℝ, (∀ x ∈ Set.Icc 1 3, (h a x + 2)^2 ≤ h a (x^2) + m + 2) → m ≥ 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l455_45520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_walk_paths_l455_45562

theorem grid_walk_paths (n : ℕ) :
  n > 0 →
  (number_of_paths : ℕ) →
  number_of_paths = 2^(n - 1) :=
by
  intros h_n_pos h_number_of_paths
  -- The proof would go here
  sorry

#check grid_walk_paths

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_walk_paths_l455_45562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_point_check_l455_45548

noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

def y_increases_with_x (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂

theorem inverse_proportion_point_check
  (k : ℝ)
  (h1 : k ≠ 0)
  (h2 : y_increases_with_x (inverse_proportion k)) :
  (inverse_proportion k (-2) = 3) ∧
  (inverse_proportion k 2 ≠ 3) ∧
  (inverse_proportion k 3 ≠ 0) ∧
  (inverse_proportion k (-3) ≠ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_point_check_l455_45548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_polynomial_non_negative_coefficients_l455_45586

/-- A polynomial with real coefficients that is positive for non-negative x -/
structure PositivePolynomial where
  p : Polynomial ℝ
  positive : ∀ x : ℝ, x ≥ 0 → p.eval x > 0

/-- The theorem stating that for any positive polynomial, there exists a positive integer n
    such that (1 + x)^n * p(x) has non-negative coefficients -/
theorem positive_polynomial_non_negative_coefficients (pp : PositivePolynomial) :
  ∃ n : ℕ+, ∀ i : ℕ, (((1 + X : Polynomial ℝ)^n.val * pp.p).coeff i) ≥ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_polynomial_non_negative_coefficients_l455_45586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_rounds_is_16_div_3_l455_45538

/-- Represents the state of the game -/
inductive GameState
  | Equal : GameState
  | AheadByOne : GameState
  | BehindByOne : GameState

/-- Represents the current round number -/
def Round := Nat

/-- The probability of winning for the favored player in a round -/
def winProbability : ℚ := 3/4

/-- The game ends when a player is ahead by two rounds -/
def isGameOver (state : GameState) : Bool :=
  match state with
  | GameState.Equal => false
  | GameState.AheadByOne => false
  | GameState.BehindByOne => false

/-- The probability of transitioning from one state to another in a single round -/
def transitionProbability (state1 state2 : GameState) (round : Round) : ℚ :=
  sorry

/-- The expected number of additional rounds from a given state -/
noncomputable def expectedAdditionalRounds (state : GameState) : ℚ :=
  sorry

theorem expected_rounds_is_16_div_3 :
  expectedAdditionalRounds GameState.Equal = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_rounds_is_16_div_3_l455_45538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_radius_l455_45529

-- Define the wire length
noncomputable def wire_length : ℝ := 20

-- Define the area function for a circular sector
noncomputable def sector_area (r : ℝ) : ℝ := r * (wire_length - 2 * r) / 2

-- State the theorem
theorem max_area_radius :
  ∃ (r : ℝ), r > 0 ∧ r < wire_length / 2 ∧
  ∀ (x : ℝ), x > 0 → x < wire_length / 2 → sector_area r ≥ sector_area x ∧
  r = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_radius_l455_45529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_negative_eight_equals_negative_thirty_seven_sixteenths_l455_45574

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x - 9

noncomputable def g (x : ℝ) : ℝ := 3 * ((f⁻¹ x) ^ 2) + 6 * (f⁻¹ x) - 4

-- Theorem statement
theorem g_of_negative_eight_equals_negative_thirty_seven_sixteenths :
  g (-8) = -37 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_negative_eight_equals_negative_thirty_seven_sixteenths_l455_45574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_general_form_power_function_passes_through_point_l455_45566

-- Define a power function
noncomputable def power_function (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

-- Theorem stating that the general form of a power function remains x^α
theorem power_function_general_form (α : ℝ) :
  ∀ x : ℝ, power_function α x = x ^ α := by
  intro x
  -- Unfold the definition of power_function
  unfold power_function
  -- The equality is true by definition
  rfl

-- Theorem stating that for any α, the function passes through (3, 3^α)
theorem power_function_passes_through_point (α : ℝ) :
  power_function α 3 = 3 ^ α := by
  -- Unfold the definition of power_function
  unfold power_function
  -- The equality is true by definition
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_general_form_power_function_passes_through_point_l455_45566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lindas_initial_quarters_l455_45514

-- Define the initial coin counts
def initial_dimes : ℕ := 2
def initial_nickels : ℕ := 5

-- Define the additional coins given by mother
def additional_dimes : ℕ := 2
def additional_quarters : ℕ := 10

-- Define the total number of coins after receiving additional coins
def total_coins : ℕ := 35

-- Theorem to prove
theorem lindas_initial_quarters : ∃ Q : ℕ, 
  (initial_dimes + additional_dimes) + 
  (Q + additional_quarters) + 
  (initial_nickels + 2 * initial_nickels) = total_coins ∧ 
  Q = 6 := by
  -- Let Q be the initial number of quarters
  let Q : ℕ := 6

  -- Calculate the final number of dimes
  let final_dimes := initial_dimes + additional_dimes

  -- Calculate the final number of quarters
  let final_quarters := Q + additional_quarters

  -- Calculate the final number of nickels
  let final_nickels := initial_nickels + 2 * initial_nickels

  -- Assert that the sum of all coins after receiving additional coins equals the total
  have h : final_dimes + final_quarters + final_nickels = total_coins := by
    -- The proof goes here
    sorry

  -- Return the existence of Q satisfying the conditions
  exact ⟨Q, h, rfl⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lindas_initial_quarters_l455_45514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_k_value_l455_45591

/-- The Plane type represents a plane in 3D space -/
structure Plane where
  normal : ℝ × ℝ × ℝ

/-- Two planes are parallel if and only if their normal vectors are proportional -/
axiom planes_parallel_iff_normal_vectors_proportional {α β : Plane} :
  α.normal = β.normal ↔ ∃ (c : ℝ) (hc : c ≠ 0), α.normal = (c, c, c) • β.normal

theorem parallel_planes_k_value (α β : Plane)
  (h1 : α.normal = (1, 2, -2))
  (h2 : ∃ k, β.normal = (-2, -4, k))
  (h3 : α.normal = β.normal) :
  ∃ k, β.normal = (-2, -4, k) ∧ k = 4 := by
  sorry

#check parallel_planes_k_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_k_value_l455_45591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_odd_powers_l455_45592

/-- Polynomial type with integer coefficients -/
def MyPolynomial (α : Type) := List α

/-- Coefficient of x^n in the product of two polynomials -/
def coeff_product (p q : MyPolynomial ℤ) (n : ℕ) : ℤ :=
  sorry

/-- The first polynomial (x³ - x² - 6x + 2) -/
def poly1 : MyPolynomial ℤ := [2, -6, -1, 1]

/-- The second polynomial (x³ + px² + qx + r) -/
def poly2 (p q r : ℤ) : MyPolynomial ℤ := [r, q, p, 1]

/-- Theorem: The product of poly1 and poly2 has no odd-power terms
    if and only if p = 1, q = -6, and r = -2 -/
theorem no_odd_powers (p q r : ℤ) :
  (coeff_product poly1 (poly2 p q r) 1 = 0 ∧
   coeff_product poly1 (poly2 p q r) 3 = 0 ∧
   coeff_product poly1 (poly2 p q r) 5 = 0) ↔
  (p = 1 ∧ q = -6 ∧ r = -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_odd_powers_l455_45592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_purchase_price_l455_45583

/-- Calculates the original purchase price of a house given various expenses and fees --/
theorem original_purchase_price
  (commission_rate : ℝ)
  (commission_amount : ℝ)
  (improvements : ℝ)
  (appreciation_rate : ℝ)
  (appreciation_years : ℕ)
  (transfer_tax_rate : ℝ)
  (closing_costs : ℝ)
  (legal_fees : ℝ)
  (h1 : commission_rate = 0.06)
  (h2 : commission_amount = 8880)
  (h3 : improvements = 20000)
  (h4 : appreciation_rate = 0.02)
  (h5 : appreciation_years = 3)
  (h6 : transfer_tax_rate = 0.02)
  (h7 : closing_costs = 3000)
  (h8 : legal_fees = 1200) :
  ∃ (selling_price : ℝ),
    selling_price * commission_rate = commission_amount ∧
    selling_price - (commission_amount +
                     improvements +
                     (selling_price * appreciation_rate * appreciation_years) +
                     (selling_price * transfer_tax_rate) +
                     closing_costs +
                     legal_fees) = 103080 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_purchase_price_l455_45583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l455_45549

/-- Calculates the speed of a train given its length and time to cross a fixed point. -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  length / time

/-- Theorem: A train 1200 m long that crosses an electric pole in 15 seconds has a speed of 80 m/s. -/
theorem train_speed_theorem :
  train_speed 1200 15 = 80 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l455_45549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_position_l455_45589

noncomputable def vertex_x (a b : ℝ) : ℝ := -b / (2 * a)

noncomputable def vertex_y (a b c : ℝ) : ℝ := c - (b^2) / (4 * a)

theorem parabola_position (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) 
  (h1 : a₁ = 1 ∧ b₁ = -2 ∧ c₁ = 5)  -- Coefficients of y = x^2 - 2x + 5
  (h2 : a₂ = 1 ∧ b₂ = 2 ∧ c₂ = 3)   -- Coefficients of y = x^2 + 2x + 3
  (h3 : a₁ = a₂)                    -- Same shape condition
  : vertex_x a₁ b₁ > vertex_x a₂ b₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_position_l455_45589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_f_equals_g_l455_45518

noncomputable def f (x : ℝ) := Real.sin (2 * x - Real.pi / 3)

def shift_left (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := fun x ↦ f (x + a)

def compress_horizontal (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := fun x ↦ f (k * x)

noncomputable def g (x : ℝ) := Real.sin (4 * x + Real.pi / 3)

theorem transform_f_equals_g :
  compress_horizontal (shift_left f (Real.pi / 3)) 2 = g := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_f_equals_g_l455_45518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_product_inequality_l455_45541

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- Checks if a point is outside the ellipse -/
def is_outside (e : Ellipse) (p : Point) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) > 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For a point P outside an ellipse, |PF|^2 > |QF| · |RF| -/
theorem tangent_product_inequality
  (e : Ellipse) (p f q r : Point)
  (h_outside : is_outside e p)
  (h_focus : f.x = e.a * Real.sqrt (1 - e.b^2 / e.a^2) ∧ f.y = 0)
  (h_tangent : ∃ (t : ℝ), (q.x^2 / e.a^2) + (q.y^2 / e.b^2) = 1 ∧
                          (r.x^2 / e.a^2) + (r.y^2 / e.b^2) = 1 ∧
                          (p.x - q.x) * q.x / e.a^2 + (p.y - q.y) * q.y / e.b^2 = 0 ∧
                          (p.x - r.x) * r.x / e.a^2 + (p.y - r.y) * r.y / e.b^2 = 0) :
  (distance p f)^2 > (distance q f) * (distance r f) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_product_inequality_l455_45541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_negative_two_l455_45555

theorem tan_alpha_negative_two (α : ℝ) (h : Real.tan α = -2) :
  (Real.sin α + 5 * Real.cos α) / (-2 * Real.cos α + Real.sin α) = -3/4 ∧
  Real.sin (α - 5 * Real.pi) * Real.sin (3 * Real.pi / 2 - α) = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_negative_two_l455_45555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_length_l455_45573

theorem triangle_third_side_length (a b c : ℝ) (θ : ℝ) : 
  a = 5 → b = 12 → θ = 150 * Real.pi / 180 → c^2 = a^2 + b^2 - 2*a*b*(Real.cos θ) →
  c = Real.sqrt (169 + 60 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_length_l455_45573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l455_45565

theorem negation_of_proposition :
  (¬(∀ x : ℝ, x^2 - x + 1/4 ≥ 0)) ↔ (∃ x : ℝ, x^2 - x + 1/4 < 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l455_45565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l455_45545

noncomputable section

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the ellipse E
def ellipse_E (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

-- Define a point on ellipse C
def point_on_C (P : ℝ × ℝ) : Prop := ellipse_C P.1 P.2

-- Define a point on ellipse E
def point_on_E (Q : ℝ × ℝ) : Prop := ellipse_E Q.1 Q.2

-- Define the ratio of distances
noncomputable def distance_ratio (O P Q : ℝ × ℝ) : ℝ := 
  Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) / Real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2)

-- Define the area of a triangle
noncomputable def triangle_area (A B Q : ℝ × ℝ) : ℝ := 
  |((B.1 - A.1) * (Q.2 - A.2) - (Q.1 - A.1) * (B.2 - A.2))| / 2

theorem ellipse_properties :
  -- The eccentricity of C is √3/2
  ∃ (c : ℝ), c / 2 = Real.sqrt 3 / 2 ∧
  -- The distance between foci is 4
  c = 2 →
  -- 1. The equation of C is x^2/4 + y^2 = 1
  (∀ x y : ℝ, ellipse_C x y ↔ x^2 / 4 + y^2 = 1) ∧
  -- 2. For any point P on C and Q on E, |OP|/|OQ| = 2
  (∀ O P Q : ℝ × ℝ, point_on_C P → point_on_E Q → distance_ratio O P Q = 2) ∧
  -- 3. The maximum area of ΔABQ is 6√3
  (∃ (max_area : ℝ), max_area = 6 * Real.sqrt 3 ∧
    ∀ A B Q : ℝ × ℝ, point_on_E A → point_on_E B → point_on_E Q →
    triangle_area A B Q ≤ max_area) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l455_45545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l455_45524

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1

-- State the theorem
theorem tangent_line_at_one :
  ∃ (m b : ℝ), 
    (∀ x, m * x + b = f 1 + f' 1 * (x - 1)) ∧
    m = 1 ∧ 
    b = -1 := by
  -- Existence of m and b
  use 1, -1
  constructor
  · -- First part: ∀ x, m * x + b = f 1 + f' 1 * (x - 1)
    intro x
    simp [f, f']
    ring
  · -- Second part: m = 1 ∧ b = -1
    simp


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l455_45524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_question_1_question_2_question_3_l455_45582

/-- The function f(x) defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^2 + m * x - 1

/-- The function g(x) defined in the problem -/
def g (m : ℝ) (a : ℝ) (x : ℝ) : ℝ := |x - a| - x^2 - m * x

/-- The set A defined in the problem -/
def set_A (m : ℝ) : Set ℝ := {y | ∃ x, y = f m x ∧ x ∈ Set.Icc (-Real.sqrt 2 / 2) (Real.sqrt 2 / 2)}

/-- The theorem corresponding to the first question -/
theorem question_1 (m : ℝ) (h : ∀ x, f m x = f m (2 - x)) : 
  set_A m = Set.Icc (-2 * Real.sqrt 2) (2 * Real.sqrt 2) := by
  sorry

/-- The theorem corresponding to the second question -/
theorem question_2 : 
  {m : ℝ | Set.inter (Set.Iio 0) {y | ∃ x, y = f m x + 2} = ∅} = 
  Set.Icc (-2 * Real.sqrt 2) (2 * Real.sqrt 2) := by
  sorry

/-- The theorem corresponding to the third question -/
theorem question_3 (m : ℝ) (a : ℝ) : 
  ∃ min_value : ℝ, ∀ x, f m x + g m a x ≥ min_value := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_question_1_question_2_question_3_l455_45582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_l455_45504

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 1)

-- State the theorem
theorem inverse_function_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a (-1) = 1) :
  Function.invFun (f a) 1 = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_l455_45504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_curve_to_line_l455_45542

-- Define the curve
noncomputable def curve (x : ℝ) : ℝ := x^2 - Real.log x

-- Define the line
def line (x y : ℝ) : Prop := x - y - 2 = 0

-- State the theorem
theorem shortest_distance_curve_to_line :
  ∃ (p : ℝ × ℝ), p.1 > 0 ∧ p.2 = curve p.1 ∧
  (∀ (q : ℝ × ℝ), q.1 > 0 → q.2 = curve q.1 →
    (|p.1 - p.2 - 2| / Real.sqrt 2 ≤ |q.1 - q.2 - 2| / Real.sqrt 2)) ∧
  |p.1 - p.2 - 2| / Real.sqrt 2 = Real.sqrt 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_curve_to_line_l455_45542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_profit_is_4000_l455_45550

/-- Represents the profit distribution in a partnership business --/
structure PartnershipProfit where
  investment_A : ℚ
  investment_B : ℚ
  period_A : ℚ
  period_B : ℚ
  total_profit : ℚ

/-- Calculates B's profit share in the partnership --/
def calculate_B_profit (p : PartnershipProfit) : ℚ :=
  let ratio_B := p.investment_B * p.period_B
  let ratio_A := p.investment_A * p.period_A
  let total_ratio := ratio_A + ratio_B
  (ratio_B / total_ratio) * p.total_profit

/-- Theorem stating B's profit given the conditions --/
theorem B_profit_is_4000 (p : PartnershipProfit) 
  (h1 : p.investment_A = 3 * p.investment_B)
  (h2 : p.period_A = 2 * p.period_B)
  (h3 : p.total_profit = 28000) :
  calculate_B_profit p = 4000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_profit_is_4000_l455_45550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l455_45537

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {x : ℕ | x ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l455_45537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_arrangement_count_l455_45564

/-- The number of ways to arrange n distinct objects. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to choose k objects from n distinct objects, where order matters. -/
def permutations (n k : ℕ) : ℕ := 
  if k ≤ n then Nat.factorial n / Nat.factorial (n - k) else 0

theorem photo_arrangement_count : 
  ∀ (num_students num_teachers : ℕ), 
    num_students = 4 → 
    num_teachers = 3 → 
    (permutations (num_students + 1) num_teachers) * (arrangements num_students) = 
      permutations 5 3 * arrangements 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_arrangement_count_l455_45564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l455_45535

/-- Represents the compound interest calculation for a half-yearly compounded investment over one year -/
noncomputable def compound_interest (principal : ℝ) (annual_rate : ℝ) : ℝ :=
  principal * (1 + annual_rate / 2) ^ 2

/-- Theorem stating that an investment of 12000 at 10% annual interest compounded half-yearly will result in 13230 after one year -/
theorem investment_growth (initial_investment : ℝ) 
  (h1 : initial_investment > 0)
  (h2 : compound_interest initial_investment 0.1 = 13230) :
  initial_investment = 12000 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval compound_interest 12000 0.1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l455_45535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_seven_solutions_l455_45552

-- Define the equation
noncomputable def f (x : ℝ) : ℝ := 10 * Real.sin (x + Real.pi / 6) - x

-- State the theorem
theorem equation_has_seven_solutions :
  ∃! (s : Finset ℝ), s.card = 7 ∧ ∀ x ∈ s, f x = 0 := by
  sorry

#check equation_has_seven_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_seven_solutions_l455_45552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_150th_term_l455_45500

def nextTerm (n : ℕ) : ℕ :=
  if n < 15 then n * 7
  else if n % 2 = 0 then n / 2
  else n - 7

def sequenceJerry (n : ℕ) : ℕ :=
  match n with
  | 0 => 63
  | n + 1 => nextTerm (sequenceJerry n)

theorem sequence_150th_term :
  sequenceJerry 149 = 98 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_150th_term_l455_45500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_seven_twelfths_l455_45579

noncomputable def g (a b : ℝ) : ℝ :=
  if a + b ≤ 4 then
    (2 * a * b - a + 3) / (3 * a)
  else
    (a * b - b - 1) / (-3 * b)

theorem g_sum_equals_seven_twelfths :
  g 2 1 + g 2 4 = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_seven_twelfths_l455_45579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_log_expression_l455_45502

theorem min_value_of_log_expression (a : ℝ) (h : a > 1) :
  Real.log 16 / Real.log a + 2 * Real.log a / Real.log 4 ≥ 4 ∧
  (Real.log 16 / Real.log a + 2 * Real.log a / Real.log 4 = 4 ↔ a = 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_log_expression_l455_45502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_rate_calculation_l455_45569

/-- The rate of a man rowing in still water, given his speeds with and against the stream. -/
noncomputable def mans_rate (speed_with_stream speed_against_stream : ℝ) : ℝ :=
  (speed_with_stream + speed_against_stream) / 2

/-- Theorem stating that given a man who can row with the stream at 20 km/h
    and against the stream at 4 km/h, his rate in still water is 12 km/h. -/
theorem mans_rate_calculation :
  mans_rate 20 4 = 12 := by
  unfold mans_rate
  simp [add_div]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_rate_calculation_l455_45569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_printer_y_time_l455_45509

/-- Represents the time (in hours) it takes for a printer to complete the job alone -/
structure PrinterTime where
  hours : ℝ
  positive : hours > 0

/-- The printing job -/
def job : Type := Unit

/-- Time it takes printer X to complete the job -/
def x : PrinterTime := ⟨12, by norm_num⟩

/-- Time it takes printer Z to complete the job -/
def z : PrinterTime := ⟨20, by norm_num⟩

/-- The ratio of printer X's time to the combined time of Y and Z -/
def ratio : ℝ := 1.8000000000000003

theorem printer_y_time :
    ∃ y : PrinterTime, 
      y.hours = 10 ∧ 
      (x.hours / (1 / (1 / y.hours + 1 / z.hours))) = ratio := by
  -- Define y
  let y : PrinterTime := ⟨10, by norm_num⟩
  
  -- Prove existence
  use y
  
  -- Prove the two conjuncts
  constructor
  
  -- First conjunct: y.hours = 10
  · rfl
  
  -- Second conjunct: (x.hours / (1 / (1 / y.hours + 1 / z.hours))) = ratio
  · sorry  -- This part requires more detailed calculation


end NUMINAMATH_CALUDE_ERRORFEEDBACK_printer_y_time_l455_45509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dr_jones_remaining_money_l455_45567

/-- Calculates the remaining money for Dr. Jones after paying his bills -/
def remaining_money (
  monthly_income : ℕ)
  (rent : ℕ)
  (food_expense : ℕ)
  (electric_water_ratio : ℚ)
  (insurance_ratio : ℚ)
  : ℕ :=
  monthly_income
  - (rent + food_expense)
  - Int.toNat (Int.floor (↑monthly_income * electric_water_ratio))
  - Int.toNat (Int.floor (↑monthly_income * insurance_ratio))

/-- Proves that Dr. Jones has $2280 left after paying his bills -/
theorem dr_jones_remaining_money :
  remaining_money 6000 640 380 (1/4) (1/5) = 2280 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dr_jones_remaining_money_l455_45567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_a_range_of_m_inequality_for_natural_numbers_l455_45597

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * (x + 1)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + a / Real.exp x

-- Statement 1
theorem max_value_of_a (a : ℝ) (h1 : a > 0) (h2 : ∀ x : ℝ, f a x ≥ 0) :
  a ≤ 1 := by sorry

-- Statement 2
theorem range_of_m (a : ℝ) (h : a ≤ -1) (m : ℝ)
  (h2 : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (g a x₂ - g a x₁) / (x₂ - x₁) > m) :
  m ≤ 3 := by sorry

-- Statement 3
theorem inequality_for_natural_numbers (n : ℕ) :
  2 * (Real.exp n - 1) / (Real.exp 1 - 1) ≥ n * (n + 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_a_range_of_m_inequality_for_natural_numbers_l455_45597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_for_circumscribed_trapezoid_l455_45590

/-- An isosceles trapezoid circumscribed around a circle -/
structure CircumscribedTrapezoid where
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The distance between the points where the circle touches the non-parallel sides -/
  tangent_distance : ℝ
  /-- The condition that the tangent distance is √3 times the radius -/
  tangent_condition : tangent_distance = radius * Real.sqrt 3

/-- The ratio of the area of the trapezoid to the area of the circle -/
noncomputable def area_ratio (t : CircumscribedTrapezoid) : ℝ :=
  (8 * Real.sqrt 3) / (3 * Real.pi)

/-- The main theorem stating the area ratio for the given conditions -/
theorem area_ratio_for_circumscribed_trapezoid (t : CircumscribedTrapezoid) :
  area_ratio t = (8 * Real.sqrt 3) / (3 * Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_for_circumscribed_trapezoid_l455_45590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_enclosing_sphere_radius_l455_45505

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : Fin 3 → ℝ
  radius : ℝ

/-- Checks if a sphere is unit and tangent to an axis at distance 1 from origin -/
def is_valid_sphere (s : Sphere) : Prop :=
  s.radius = 1 ∧
  (s.center 0 = 1 ∨ s.center 0 = -1 ∨ 
   s.center 1 = 1 ∨ s.center 1 = -1 ∨
   s.center 2 = 1 ∨ s.center 2 = -1) ∧
  (abs (s.center 0) ≥ 1 ∧ abs (s.center 1) ≥ 1 ∧ abs (s.center 2) ≥ 1)

/-- The theorem to be proven -/
theorem smallest_enclosing_sphere_radius 
  (spheres : Finset Sphere) 
  (h1 : spheres.card = 8)
  (h2 : ∀ s ∈ spheres, is_valid_sphere s) :
  ∃ (r : ℝ), r = Real.sqrt 6 ∧ 
    (∀ s ∈ spheres, 
      Real.sqrt ((s.center 0)^2 + (s.center 1)^2 + (s.center 2)^2) + s.radius ≤ r) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_enclosing_sphere_radius_l455_45505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_half_plus_two_theta_l455_45503

theorem cos_pi_half_plus_two_theta 
  (θ : ℝ) 
  (h1 : Real.cos θ = 1/3) 
  (h2 : θ ∈ Set.Ioo 0 π) : 
  Real.cos (π/2 + 2*θ) = -4*Real.sqrt 2/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_half_plus_two_theta_l455_45503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l455_45532

theorem simplify_expression : 
  (Real.rpow 32 (1/3) - Real.sqrt (4 + 1/4))^2 = ((8 * Real.rpow 2 (1/3) - Real.sqrt 17)^2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l455_45532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_special_case_l455_45547

-- Define an ellipse structure
structure Ellipse where
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis
  c : ℝ  -- focal distance
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_relation : a^2 = b^2 + c^2

-- Define eccentricity
noncomputable def eccentricity (e : Ellipse) : ℝ := e.c / e.a

-- Theorem statement
theorem ellipse_eccentricity_special_case (e : Ellipse) 
  (h : e.c = e.b) : eccentricity e = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_special_case_l455_45547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_theorem_l455_45540

-- Define the angle α
variable (α : Real)

-- Define the x-coordinate of point P
variable (x : Real)

-- Define the conditions
def terminal_side_condition (α x : Real) : Prop := 
  ∃ (y : Real), y = -3 ∧ Real.cos α = x / Real.sqrt (x^2 + y^2)

def cos_alpha_condition (α : Real) : Prop := 
  Real.cos α = -Real.sqrt 3 / 2

-- State the theorem
theorem x_value_theorem (h1 : terminal_side_condition α x) (h2 : cos_alpha_condition α) : 
  x = -3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_theorem_l455_45540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_theorem_l455_45533

theorem binomial_expansion_theorem (a : ℝ) : 
  (∃ x y : ℝ, 
    (Nat.choose 7 2 : ℝ) * x^2 * (a*y)^5 = 6 * (2/9) * a^5 ∧ 
    (Nat.choose 7 4 : ℝ) * x^4 * (a*y)^3 = 52 * (1/2) * a^3) → 
  (∃ x y : ℝ, x = 3/2 ∧ y = 2/3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_theorem_l455_45533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_curve_area_l455_45515

/-- Represents a projectile motion with initial velocity v, angle θ, and additional vertical velocity u -/
structure ProjectileMotion where
  v : ℝ
  u : ℝ
  g : ℝ
  θ : ℝ

/-- The x-coordinate of the projectile at time t -/
noncomputable def x (p : ProjectileMotion) (t : ℝ) : ℝ := p.v * t * Real.cos p.θ

/-- The y-coordinate of the projectile at time t -/
noncomputable def y (p : ProjectileMotion) (t : ℝ) : ℝ := 
  (p.v * t * Real.sin p.θ + p.u * t) - (1/2) * p.g * t^2

/-- The time at which the projectile reaches its highest point -/
noncomputable def peak_time (p : ProjectileMotion) : ℝ := (p.v * Real.sin p.θ + p.u) / p.g

/-- The x-coordinate of the highest point of the projectile's trajectory -/
noncomputable def peak_x (p : ProjectileMotion) : ℝ := x p (peak_time p)

/-- The y-coordinate of the highest point of the projectile's trajectory -/
noncomputable def peak_y (p : ProjectileMotion) : ℝ := y p (peak_time p)

/-- The area of the closed curve traced by the highest points of the projectile trajectories -/
noncomputable def curve_area (p : ProjectileMotion) : ℝ := Real.pi * (p.v^2 + p.u^2)^2 / (8 * p.g^2)

theorem projectile_curve_area (p : ProjectileMotion) 
  (h₁ : 0 ≤ p.θ) (h₂ : p.θ ≤ Real.pi) :
  curve_area p = Real.pi * (p.v^2 + p.u^2)^2 / (8 * p.g^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_curve_area_l455_45515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_derivative_plus_three_halves_l455_45534

/-- The function f(x) = x - ln x + (2x - 1) / x² -/
noncomputable def f (x : ℝ) : ℝ := x - Real.log x + (2 * x - 1) / (x ^ 2)

/-- The derivative of f(x) -/
noncomputable def f_derivative (x : ℝ) : ℝ := 1 - 1 / x + (2 * x ^ 2 - (2 * x - 1) * 2 * x) / (x ^ 4)

theorem f_greater_than_derivative_plus_three_halves 
  (x : ℝ) (hx : x ∈ Set.Icc 1 2) : f x > f_derivative x + 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_derivative_plus_three_halves_l455_45534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_greater_than_100_l455_45570

/-- Represents a point on a circle -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a runner on the circle -/
structure Runner where
  speed : ℝ
  clockwise : Bool

/-- Represents the circle and the runners' situation -/
structure CircleScenario where
  circumference : ℝ
  point_p : Point
  runner_a : Runner
  runner_b : Runner

/-- Helper function to calculate distance between two points on the circle -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

/-- Helper function to determine a runner's position after running a certain distance -/
def position_after_run (scenario : CircleScenario) (runner : Runner) (dist : ℝ) : Point :=
  sorry

/-- The main theorem about the circle's circumference -/
theorem circle_circumference_greater_than_100 (scenario : CircleScenario) 
  (h1 : scenario.runner_a.clockwise = true)
  (h2 : scenario.runner_b.clockwise = false)
  (h3 : scenario.runner_a.speed ≠ scenario.runner_b.speed)
  (h4 : ∃ (q : Point), distance scenario.point_p q = 500 ∧ 
        ∃ (r : Point), distance q r = 400 ∧
        distance scenario.point_p r = 100)
  (h5 : distance scenario.point_p (position_after_run scenario scenario.runner_b scenario.circumference) < scenario.circumference) :
  scenario.circumference > 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_greater_than_100_l455_45570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_inequality_l455_45572

theorem increasing_function_inequality (f : ℝ → ℝ) (hf : DifferentiableOn ℝ f (Set.Icc 0 1))
  (h : ∀ x ∈ Set.Icc 0 1, deriv f x > 0) : f 1 > f 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_inequality_l455_45572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_thousand_sixth_wisdom_number_l455_45563

/-- A Wisdom Number is a positive integer that can be expressed as the difference of squares of two positive integers. -/
def WisdomNumber (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ n = a^2 - b^2

/-- The sequence of Wisdom Numbers in ascending order -/
def WisdomNumberSequence : ℕ → ℕ := sorry

/-- The 2006th Wisdom Number is 2677 -/
theorem two_thousand_sixth_wisdom_number :
  WisdomNumberSequence 2006 = 2677 := by
  sorry

#check two_thousand_sixth_wisdom_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_thousand_sixth_wisdom_number_l455_45563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l455_45599

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := 1 - 1 / (x^2 + 1)

-- State the theorem
theorem f_range : Set.range f = Set.Ico 0 1 := by
  sorry -- Proof is omitted for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l455_45599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l455_45576

theorem trig_identities (θ : Real) (h : Real.sin θ + Real.cos θ = -Real.sqrt 10 / 5) :
  (1 / Real.sin θ + 1 / Real.cos θ = 2 * Real.sqrt 10 / 3) ∧ (Real.tan θ = -Real.sqrt 11 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l455_45576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_to_line_l455_45512

/-- The curve y = e^x + 1 is tangent to the line y = ax + 2 if and only if a = 1 -/
theorem curve_tangent_to_line (a : ℝ) : 
  (∃ x : ℝ, Real.exp x + 1 = a * x + 2 ∧ Real.exp x = a) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_to_line_l455_45512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_rectangular_equivalence_l455_45585

noncomputable def spherical_to_rectangular (ρ θ φ : Real) : Real × Real × Real :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_rectangular_equivalence :
  let ρ : Real := 3
  let θ : Real := 7 * π / 12
  let φ : Real := π / 4
  let (x, y, z) := spherical_to_rectangular ρ θ φ
  x = (3 * Real.sqrt 2 / 2) * Real.cos (7 * π / 12) ∧
  y = (3 * Real.sqrt 2 / 2) * Real.sin (7 * π / 12) ∧
  z = 3 * Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_rectangular_equivalence_l455_45585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_of_data_set_l455_45539

noncomputable def data_set : List ℝ := [5, 7, 7, 8, 10, 11]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let μ := mean xs
  (xs.map (fun x => (x - μ)^2)).sum / xs.length

noncomputable def standard_deviation (xs : List ℝ) : ℝ :=
  Real.sqrt (variance xs)

theorem standard_deviation_of_data_set :
  standard_deviation data_set = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_of_data_set_l455_45539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_coefficient_product_l455_45536

/-- Given a cubic equation with specific properties, prove that the absolute product of its coefficients is 21 -/
theorem cubic_equation_coefficient_product (p q : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) : 
  (∃ r s : ℕ+, 
    (r : ℤ)^3 + p*(r : ℤ)^2 + q*(r : ℤ) - 15*p = 0 ∧
    (s : ℤ)^3 + p*(s : ℤ)^2 + q*(s : ℤ) - 15*p = 0 ∧
    (X - r)^2 * (X - s) = X^3 + p*X^2 + q*X - 15*p) →
  |p*q| = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_coefficient_product_l455_45536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_equilateral_triangle_on_hyperbola_l455_45543

noncomputable section

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

/-- Point A is the left vertex of the hyperbola -/
def left_vertex (A : ℝ × ℝ) : Prop := 
  A.1 = -1 ∧ A.2 = 0 ∧ hyperbola A.1 A.2

/-- A point is on the right branch of the hyperbola -/
def on_right_branch (P : ℝ × ℝ) : Prop :=
  P.1 > 0 ∧ hyperbola P.1 P.2

/-- Triangle ABC is equilateral -/
def is_equilateral (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2

/-- Calculate the area of a triangle given its vertices -/
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

/-- The main theorem -/
theorem area_of_equilateral_triangle_on_hyperbola (A B C : ℝ × ℝ) :
  left_vertex A → on_right_branch B → on_right_branch C → is_equilateral A B C →
  triangle_area A B C = 3 * Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_equilateral_triangle_on_hyperbola_l455_45543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_of_complex_roots_l455_45522

noncomputable def area_of_rhombus (a b c d : ℂ) : ℝ :=
  2 * (Complex.abs ((a + Complex.I) * (b + Complex.I)))

theorem rhombus_area_of_complex_roots : ∃ (a b c d : ℂ),
  (a^4 + 4*Complex.I*a^3 + (-5 + 5*Complex.I)*a^2 + (-10 - Complex.I)*a + (1 - 6*Complex.I) = 0) ∧
  (b^4 + 4*Complex.I*b^3 + (-5 + 5*Complex.I)*b^2 + (-10 - Complex.I)*b + (1 - 6*Complex.I) = 0) ∧
  (c^4 + 4*Complex.I*c^3 + (-5 + 5*Complex.I)*c^2 + (-10 - Complex.I)*c + (1 - 6*Complex.I) = 0) ∧
  (d^4 + 4*Complex.I*d^3 + (-5 + 5*Complex.I)*d^2 + (-10 - Complex.I)*d + (1 - 6*Complex.I) = 0) ∧
  (a + b + c + d = -4*Complex.I) ∧
  (area_of_rhombus a b c d = 10) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_of_complex_roots_l455_45522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_N_l455_45556

def N : ℕ := 1000^2 - 950^2

theorem largest_prime_factor_of_N : 
  (Nat.factors N).maximum? = some 13 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_N_l455_45556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l455_45595

/-- Calculates the present value of an investment -/
noncomputable def present_value (future_value : ℝ) (interest_rate : ℝ) (years : ℕ) : ℝ :=
  future_value / (1 + interest_rate) ^ years

/-- The investment problem -/
theorem investment_problem (ε : ℝ) (ε_pos : ε > 0) :
  ∃ (pv : ℝ), 
    abs (pv - present_value 1000000 0.08 20) < ε ∧ 
    abs (pv - 214548.56) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l455_45595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pollutant_concentration_l455_45598

/-- The pollutant concentration function -/
noncomputable def p (p₀ : ℝ) (t : ℝ) : ℝ := p₀ * 2^(-t/30)

/-- The derivative of the pollutant concentration function -/
noncomputable def dp (p₀ : ℝ) (t : ℝ) : ℝ := -p₀ * (Real.log 2 / 30) * 2^(-t/30)

theorem pollutant_concentration (p₀ : ℝ) :
  (∀ t ∈ Set.Icc 0 30, dp p₀ t = -10 * Real.log 2) →
  p p₀ 60 = 75 * Real.log 2 := by
  sorry

#check pollutant_concentration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pollutant_concentration_l455_45598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_discount_percentage_l455_45519

def normal_price : ℝ := 199.99999999999997
def first_discount : ℝ := 0.1
def final_price : ℝ := 144

def price_after_first_discount : ℝ := normal_price * (1 - first_discount)

theorem second_discount_percentage :
  ∃ (second_discount : ℝ),
    (price_after_first_discount * (1 - second_discount) = final_price) ∧
    (abs (second_discount - 0.2) < 0.00001) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_discount_percentage_l455_45519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squared_residuals_equals_0_03_l455_45587

def regression_equation (x : ℝ) : ℝ := 2 * x + 1

def data_points : List (ℝ × ℝ) := [(2, 5.1), (3, 6.9), (4, 9.1)]

def calculate_residual (point : ℝ × ℝ) : ℝ :=
  point.2 - regression_equation point.1

def sum_squared_residuals (points : List (ℝ × ℝ)) : ℝ :=
  (points.map (λ p => (calculate_residual p) ^ 2)).sum

theorem sum_squared_residuals_equals_0_03 :
  sum_squared_residuals data_points = 0.03 := by
  -- Proof goes here
  sorry

#eval sum_squared_residuals data_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squared_residuals_equals_0_03_l455_45587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_positive_reals_f_greatest_lower_bound_on_interval_l455_45530

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x - 2/x

-- Theorem for monotonicity
theorem f_increasing_on_positive_reals :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ < f x₂ :=
by
  sorry

-- Theorem for the greatest lower bound on [2,6]
theorem f_greatest_lower_bound_on_interval :
  ∃ a : ℝ, a = 1 ∧ (∀ x : ℝ, 2 ≤ x → x ≤ 6 → a ≤ f x) ∧
  (∀ b : ℝ, (∀ x : ℝ, 2 ≤ x → x ≤ 6 → b ≤ f x) → b ≤ a) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_positive_reals_f_greatest_lower_bound_on_interval_l455_45530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_pairs_existence_l455_45580

/-- Represents an arithmetic sequence with first term a and common difference d -/
def ArithmeticSequence (a : ℝ) (d : ℝ) : ℕ → ℝ := λ n ↦ a + (n - 1 : ℝ) * d

/-- Represents a geometric sequence with first term a and common ratio r -/
def GeometricSequence (a : ℝ) (r : ℝ) : ℕ → ℝ := λ n ↦ a * r ^ (n - 1)

/-- The theorem stating that there are exactly two pairs of arithmetic and geometric sequences
    satisfying the given conditions -/
theorem sequence_pairs_existence :
  ∃! (a₁ d₁ a₂ d₂ : ℝ) (r₁ r₂ : ℝ),
    (a₁ = 5 ∧ a₂ = 5) ∧  -- First term of both sequences is 5
    (ArithmeticSequence a₁ d₁ 2 = GeometricSequence a₁ r₁ 2 - 2) ∧  -- Second term condition
    (ArithmeticSequence a₂ d₂ 2 = GeometricSequence a₂ r₂ 2 - 2) ∧  -- Second term condition
    (GeometricSequence a₁ r₁ 3 = ArithmeticSequence a₁ d₁ 6) ∧  -- Third term of geometric = sixth term of arithmetic
    (GeometricSequence a₂ r₂ 3 = ArithmeticSequence a₂ d₂ 6) ∧  -- Third term of geometric = sixth term of arithmetic
    ((a₁, d₁, r₁) ≠ (a₂, d₂, r₂)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_pairs_existence_l455_45580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_range_l455_45554

/-- The function f(x) = xe^x -/
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

/-- The point P through which the tangent lines pass -/
def P (m : ℝ) : ℝ × ℝ := (1, m)

/-- The condition for a line to be tangent to f at point (x₀, f x₀) and pass through P(1,m) -/
noncomputable def is_tangent (m : ℝ) (x₀ : ℝ) : Prop :=
  m = (-(x₀^2) + x₀ + 1) * Real.exp x₀

/-- The theorem stating the range of m for which there are exactly 3 tangent lines -/
theorem tangent_lines_range :
  ∃ a b : ℝ, a = -5 / Real.exp 2 ∧ b = 0 ∧
  (∀ m : ℝ, (∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x₀ ∈ s, is_tangent m x₀) ↔ a < m ∧ m < b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_range_l455_45554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_line_l455_45546

-- Define the function we want to minimize
noncomputable def f (x y : ℝ) : ℝ := (2:ℝ)^x + (4:ℝ)^y

-- State the theorem
theorem min_value_on_line :
  ∃ (min : ℝ), min = 4 * Real.sqrt 2 ∧
  ∀ (x y : ℝ), x + 2*y = 3 → f x y ≥ min := by
  sorry

#check min_value_on_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_line_l455_45546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_above_line_l455_45508

noncomputable section

/-- Square with vertices (4,0), (10,0), (10,6), and (4,6) -/
def square : Set (ℝ × ℝ) :=
  {p | 4 ≤ p.1 ∧ p.1 ≤ 10 ∧ 0 ≤ p.2 ∧ p.2 ≤ 6}

/-- Line passing through (4,2) and (10,1) -/
def line (x : ℝ) : ℝ := -1/6 * x + 14/3

/-- Area of the square -/
def square_area : ℝ := 36

/-- Area of the triangle below the line within the square -/
def triangle_area : ℝ := 4.5

/-- Theorem: The fraction of the square's area above the line is 7/8 -/
theorem fraction_above_line :
  (square_area - triangle_area) / square_area = 7/8 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_above_line_l455_45508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_hexagon_chord_sum_l455_45557

/-- Represents a hexagon inscribed in a circle -/
structure InscribedHexagon where
  side1 : ℝ
  side2 : ℝ
  chord : ℚ

/-- Theorem stating the properties of the inscribed hexagon and its chord -/
theorem inscribed_hexagon_chord_sum (h : InscribedHexagon) :
  h.side1 = 3 ∧ h.side2 = 5 ∧ 
  (∃ m n : ℕ, h.chord = m / n ∧ Nat.Coprime m n) →
  (∃ m n : ℕ, h.chord = m / n ∧ m + n = 409) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_hexagon_chord_sum_l455_45557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_inequality_l455_45525

variable (n : ℕ)
variable (x y : ℕ → ℝ)
variable (z : ℕ → ℂ)

noncomputable def r (n : ℕ) (z : ℕ → ℂ) : ℝ := 
  Real.sqrt (Finset.sum (Finset.range n) (λ k => Complex.abs (z k) ^ 2))

theorem complex_sum_inequality (n : ℕ) (x y : ℕ → ℝ) (z : ℕ → ℂ) 
  (h : ∀ k, z k = Complex.mk (x k) (y k)) :
  r n z ≤ Finset.sum (Finset.range n) (λ k => |x k|) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_inequality_l455_45525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l455_45593

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = -12x -/
def is_parabola (p : Point) : Prop :=
  p.y^2 = -12 * p.x

/-- The focal distance of the parabola -/
def focal_distance : ℝ := 6

/-- Checks if a line passes through the focus of the parabola -/
def passes_through_focus (A B : Point) : Prop :=
  ∃ (t : ℝ), A.x + t * (B.x - A.x) = focal_distance ∧ A.y + t * (B.y - A.y) = 0

/-- The x-coordinate of the midpoint of AB -/
noncomputable def midpoint_x (A B : Point) : ℝ :=
  (A.x + B.x) / 2

/-- The length of segment AB -/
noncomputable def segment_length (A B : Point) : ℝ :=
  A.x + B.x + focal_distance

/-- Main theorem -/
theorem parabola_chord_length 
  (A B : Point) 
  (h1 : is_parabola A) 
  (h2 : is_parabola B) 
  (h3 : passes_through_focus A B) 
  (h4 : midpoint_x A B = -9) : 
  segment_length A B = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l455_45593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_filter_kit_percentage_difference_l455_45513

/-- Represents the prices of individual filters and the kit -/
structure FilterPrices where
  price1 : ℚ  -- Price of first two filters
  price2 : ℚ  -- Price of second two filters
  price3 : ℚ  -- Price of fifth filter
  kit_price : ℚ  -- Price of the kit

/-- Calculates the percentage difference between kit price and total individual price -/
def percentage_difference (prices : FilterPrices) : ℚ :=
  let total_individual := 2 * prices.price1 + 2 * prices.price2 + prices.price3
  let difference := total_individual - prices.kit_price
  (difference / total_individual) * 100

/-- Theorem stating the percentage difference for the given prices -/
theorem filter_kit_percentage_difference :
  let prices : FilterPrices := {
    price1 := 1245/100,
    price2 := 1405/100,
    price3 := 1150/100,
    kit_price := 7250/100
  }
  percentage_difference prices = -31/250 := by
  sorry

#eval percentage_difference {
  price1 := 1245/100,
  price2 := 1405/100,
  price3 := 1150/100,
  kit_price := 7250/100
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_filter_kit_percentage_difference_l455_45513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l455_45510

theorem trigonometric_equation_solution (x : ℝ) : 
  (Real.sin (2025 * x))^4 + (Real.cos (2016 * x))^2019 * (Real.cos (2025 * x))^2018 = 1 ↔ 
  (∃ n : ℤ, x = Real.pi / 4050 + Real.pi * ↑n / 2025) ∨ (∃ k : ℤ, x = Real.pi * ↑k / 9) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l455_45510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_bailing_rate_l455_45584

/-- Minimum bailing rate problem -/
theorem minimum_bailing_rate
  (distance : ℝ)
  (water_intake_rate : ℝ)
  (boat_capacity : ℝ)
  (rowing_speed : ℝ)
  (h1 : distance = 3)
  (h2 : water_intake_rate = 14)
  (h3 : boat_capacity = 40)
  (h4 : rowing_speed = 3)
  : ∃ (bailing_rate : ℝ),
    (bailing_rate ≥ 13.3 ∧ bailing_rate < 13.4) ∧
    bailing_rate * (distance / rowing_speed * 60) ≥
    water_intake_rate * (distance / rowing_speed * 60) - boat_capacity :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_bailing_rate_l455_45584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sally_savings_l455_45575

noncomputable def trip_cost (saved parking_fee entrance_fee meal_pass distance car_efficiency gas_price : ℚ) : ℚ :=
  let gas_needed := distance / car_efficiency
  let gas_cost := gas_needed * gas_price
  let total_cost := parking_fee + entrance_fee + meal_pass + gas_cost
  total_cost - saved

theorem sally_savings : 
  trip_cost 28 10 55 25 165 30 3 = 78.5 := by
  -- Unfold the definition of trip_cost
  unfold trip_cost
  -- Perform the calculation
  norm_num
  -- Close the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sally_savings_l455_45575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_score_is_125_l455_45581

/-- Represents a batsman's score -/
structure BatsmanScore where
  boundaries : ℕ
  sixes : ℕ
  runningPercentage : ℚ

/-- Calculates the total score of a batsman -/
def totalScore (score : BatsmanScore) : ℕ :=
  let boundaryRuns := score.boundaries * 4
  let sixRuns := score.sixes * 6
  let nonRunningRuns := boundaryRuns + sixRuns
  ↑((nonRunningRuns * 100) / (100 - score.runningPercentage.num.toNat))

/-- Theorem: Given the specified conditions, the batsman's total score is 125 runs -/
theorem batsman_score_is_125 (score : BatsmanScore) 
  (h1 : score.boundaries = 5)
  (h2 : score.sixes = 5)
  (h3 : score.runningPercentage = 60 / 100) :
  totalScore score = 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_score_is_125_l455_45581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_properties_l455_45523

/-- The function f(x) = x^2 - (a+1)x - 4(a+5) -/
def f (a x : ℝ) : ℝ := x^2 - (a+1)*x - 4*(a+5)

/-- The function g(x) = ax^2 - x + 5 -/
def g (a x : ℝ) : ℝ := a*x^2 - x + 5

/-- The set of values of a for which f and g have common roots -/
def common_root_values : Set ℝ := {-9/16, -6, -4, 0}

/-- The maximum value of n -/
def max_n : ℕ := 4

/-- The range of a when n is at its maximum value -/
def a_range : Set ℝ := {a | -1 ≤ a ∧ a ≤ -2/9}

theorem functions_properties (a : ℝ) :
  ((∃ x, f a x = 0 ∧ g a x = 0) ↔ a ∈ common_root_values) ∧
  ((∃ m n : ℕ, m < n ∧ n ≤ max_n ∧
    ∃ x, m < x ∧ x < n ∧ f a x < 0 ∧ g a x < 0) ↔ a ∈ a_range) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_properties_l455_45523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lightest_bag_over_2kg_l455_45551

def bag_weights : List Float := [1.2, 3.1, 2.4, 3.0, 1.8]

def bags_over_2kg (weights : List Float) : List Float :=
  weights.filter (λ w => w > 2)

theorem lightest_bag_over_2kg :
  (bags_over_2kg bag_weights).minimum? = some 2.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lightest_bag_over_2kg_l455_45551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_32_2_equals_one_fifth_l455_45594

-- Define the change of base formula
noncomputable def change_of_base (a b x : ℝ) : ℝ := (Real.log x) / (Real.log b)

-- State the theorem
theorem log_32_2_equals_one_fifth :
  change_of_base 32 2 2 = 1/5 := by
  -- Unfold the definition of change_of_base
  unfold change_of_base
  -- Simplify the expression
  simp [Real.log_div]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_32_2_equals_one_fifth_l455_45594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_forms_quadrilateral_l455_45558

-- Define the lines
def line1 (x y : ℝ) : Prop := y = 2 * x + 3
def line2 (x y : ℝ) : Prop := y = -2 * x + 1
def line3 (x y : ℝ) : Prop := y = -1
def line4 (x y : ℝ) : Prop := x = 1

-- Define the set of intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧
    ((line1 x y ∧ line2 x y) ∨
     (line1 x y ∧ line3 x y) ∨
     (line1 x y ∧ line4 x y) ∨
     (line2 x y ∧ line3 x y) ∨
     (line2 x y ∧ line4 x y) ∨
     (line3 x y ∧ line4 x y))}

-- Theorem statement
theorem intersection_forms_quadrilateral :
  ∃ (s : Finset (ℝ × ℝ)), s.card = 4 ∧ (∀ p ∈ s, p ∈ intersection_points) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_forms_quadrilateral_l455_45558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_t_l455_45578

-- Define the function t as noncomputable
noncomputable def t (x y : ℝ) : ℝ := min (2*x + y) (2*y / (x^2 + 2*y^2))

-- State the theorem
theorem max_value_of_t :
  ∃ (M : ℝ), M = Real.sqrt 2 ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → t x y ≤ M) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ t x y = M) := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_t_l455_45578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_linear_combination_l455_45577

/-- Given two planar vectors a and b, prove that if the magnitude of their linear combination
    with scalar λ is 2, then λ is -1. -/
theorem vector_linear_combination (a b : ℝ × ℝ) (l : ℝ) 
    (ha : a = (0, 1)) (hb : b = (2, 1)) : 
    ‖l • a + b‖ = 2 → l = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_linear_combination_l455_45577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_six_theta_l455_45528

theorem cos_six_theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (6*θ) = -3224/4096 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_six_theta_l455_45528
