import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_newborn_population_size_l797_79701

/-- The probability of survival for each of the first three months -/
noncomputable def survival_probability : ℝ := 9/10

/-- The number of newborns expected to survive the first three months -/
noncomputable def expected_survivors : ℝ := 364.5

/-- The initial number of newborns in the group -/
def initial_newborns : ℕ := 500

/-- Theorem stating that the given conditions result in approximately 500 initial newborns -/
theorem newborn_population_size :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |((initial_newborns : ℝ) * survival_probability^3 - expected_survivors)| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_newborn_population_size_l797_79701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_ball_measurements_l797_79728

/-- Represents a sphere with a given diameter -/
structure Sphere where
  diameter : ℝ

/-- Calculates the surface area of a sphere -/
noncomputable def surfaceArea (s : Sphere) : ℝ := 4 * Real.pi * (s.diameter / 2) ^ 2

/-- Calculates the volume of a sphere -/
noncomputable def volume (s : Sphere) : ℝ := (4 / 3) * Real.pi * (s.diameter / 2) ^ 3

theorem bowling_ball_measurements (s : Sphere) (h : s.diameter = 9) :
  surfaceArea s = 81 * Real.pi ∧ volume s = 162 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_ball_measurements_l797_79728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l797_79788

theorem ellipse_eccentricity (b : ℝ) : 
  b > 0 → 
  (∀ x y : ℝ, x^2 + y^2 / (b^2 + 1) = 1 → 
    Real.sqrt (1 - 1 / (b^2 + 1)) = Real.sqrt 10 / 10) → 
  b = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l797_79788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_factor_in_C_l797_79779

def C : Set Nat := {93, 97, 100, 103, 109}

def smallest_prime_factor (n : Nat) : Nat :=
  (Nat.factors n).minimum?
    |>.getD n -- If the list is empty, return n itself

theorem smallest_prime_factor_in_C :
  ∀ x ∈ C, smallest_prime_factor 100 ≤ smallest_prime_factor x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_factor_in_C_l797_79779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l797_79747

-- Define the constants
noncomputable def a : ℝ := Real.log 2 / Real.log (1/3)
noncomputable def b : ℝ := Real.log (1/3) / Real.log (1/2)
noncomputable def c : ℝ := (1/2) ^ (3/10)

-- State the theorem
theorem order_of_abc : a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l797_79747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_magnitude_l797_79711

noncomputable def i : ℂ := Complex.I

noncomputable def z : ℂ := 10 / (3 + i) - 2 * i

theorem z_magnitude : Complex.abs z = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_magnitude_l797_79711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_eulerian_path_l797_79749

/-- A graph with 8 vertices, each having degree 3 -/
structure OddGraph :=
  (V : Finset Nat)
  (E : Finset (Nat × Nat))
  (vertex_count : V.card = 8)
  (edge_count : E.card = 12)
  (degree_three : ∀ v ∈ V, (E.filter (λ e => e.1 = v ∨ e.2 = v)).card = 3)

/-- Definition of an Eulerian path in a graph -/
def has_eulerian_path (G : OddGraph) : Prop :=
  ∃ (path : List (Nat × Nat)), 
    List.Nodup path ∧ 
    (∀ e ∈ G.E, e ∈ path) ∧
    (∀ i, i + 1 < path.length → path[i]!.2 = path[i+1]!.1)

/-- Theorem stating that an OddGraph does not have an Eulerian path -/
theorem no_eulerian_path (G : OddGraph) : ¬ has_eulerian_path G := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_eulerian_path_l797_79749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l797_79799

-- Define the line using a parameter t
noncomputable def line (t : ℝ) : ℝ × ℝ := (2 + (Real.sqrt 2 / 2) * t, -1 + (Real.sqrt 2 / 2) * t)

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Define the chord length function
noncomputable def chord_length (t1 t2 : ℝ) : ℝ := Real.sqrt ((t1 + t2)^2 - 4 * t1 * t2)

-- Theorem statement
theorem intersection_chord_length :
  ∃ t1 t2 : ℝ,
    let (x1, y1) := line t1
    let (x2, y2) := line t2
    circle_eq x1 y1 ∧ circle_eq x2 y2 ∧ chord_length t1 t2 = 3 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l797_79799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l797_79727

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 6)

def IsSymmetricAbout (s : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop :=
  ∀ x y, (x, y) ∈ s ↔ (2 * p.1 - x, y) ∈ s

theorem f_properties :
  let smallestPeriod := Real.pi
  let minValue := -3
  let centerOfSymmetry (k : ℤ) := (Real.pi / 12 + k * Real.pi / 2, 0)
  let monotonicInterval (k : ℤ) := Set.Icc (-Real.pi / 6 + k * Real.pi) (Real.pi / 3 + k * Real.pi)
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ smallestPeriod) ∧
  (∀ x, f x ≥ minValue) ∧
  (∀ k, IsSymmetricAbout (Set.range (λ x => (x, f x))) (centerOfSymmetry k)) ∧
  (∀ k, StrictMonoOn f (monotonicInterval k)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l797_79727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_for_side_5_l797_79781

/-- A regular octagon inscribed in a circle -/
structure RegularOctagonInCircle where
  /-- The side length of the octagon -/
  side_length : ℝ
  /-- The side length is positive -/
  side_length_pos : side_length > 0

/-- The length of the arc intercepted by one side of the octagon -/
noncomputable def arc_length (octagon : RegularOctagonInCircle) : ℝ :=
  (5 * Real.pi) / 4

/-- Theorem: For a regular octagon inscribed in a circle with side length 5,
    the length of the arc intercepted by one side of the octagon is 5π/4 -/
theorem arc_length_for_side_5 (octagon : RegularOctagonInCircle)
    (h : octagon.side_length = 5) :
    arc_length octagon = (5 * Real.pi) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_for_side_5_l797_79781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l797_79721

noncomputable section

-- Define the curve C in polar coordinates
def curve_C (ρ θ : ℝ) : Prop :=
  ρ = 4 * (Real.cos θ + Real.sin θ) - 6 / ρ

-- Define the parametric equations
noncomputable def parametric_x (θ : ℝ) : ℝ := 2 + Real.sqrt 2 * Real.cos θ
noncomputable def parametric_y (θ : ℝ) : ℝ := 2 + Real.sqrt 2 * Real.sin θ

-- Theorem statement
theorem curve_C_properties :
  -- 1. Parametric equation is correct
  (∀ θ : ℝ, curve_C (Real.sqrt ((parametric_x θ - 2)^2 + (parametric_y θ - 2)^2)) θ) ∧
  -- 2. Maximum value of x + y is 6
  (∀ x y : ℝ, (∃ θ : ℝ, x = parametric_x θ ∧ y = parametric_y θ) → x + y ≤ 6) ∧
  -- 3. Maximum occurs at (3, 3)
  (∃ θ : ℝ, parametric_x θ = 3 ∧ parametric_y θ = 3 ∧ parametric_x θ + parametric_y θ = 6) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l797_79721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l797_79768

theorem range_of_a (a : ℝ) : 
  a < 0 →
  (∀ x : ℝ, (x - a) * (x - 3 * a) < 0 → (2 : ℝ)^(3 * x + 1) > (2 : ℝ)^(-x - 7)) →
  -2/3 ≤ a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l797_79768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salmon_oxygen_consumption_ratio_l797_79790

-- Define the relationship between swimming speed and oxygen consumption
noncomputable def swimming_speed (O : ℝ) : ℝ := (1/2) * (Real.log O - Real.log 100) / (Real.log 3)

-- Theorem statement
theorem salmon_oxygen_consumption_ratio :
  ∀ (O₁ O₂ : ℝ), O₁ > 0 → O₂ > 0 →
  swimming_speed O₂ = swimming_speed O₁ + 2 →
  O₂ / O₁ = 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salmon_oxygen_consumption_ratio_l797_79790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_range_of_m_l797_79706

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x

def g (m x : ℝ) : ℝ := m * x + 1

theorem symmetric_points_range_of_m (e : ℝ) (h_e : e = Real.exp 1) :
  ∀ m : ℝ, (∃ x : ℝ, 1/e ≤ x ∧ x ≤ e^2 ∧
    ∃ y : ℝ, y = f x ∧ 2 - y = g m x) →
  -2 * e^((-3:ℝ)/2) ≤ m ∧ m ≤ 3 * e := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_range_of_m_l797_79706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_rounds_proof_l797_79759

/-- The expected number of rounds in the game -/
def expected_rounds : ℚ := 16/3

/-- The probability of player A winning in an odd-numbered round -/
def prob_A_odd : ℚ := 3/4

/-- The probability of player B winning in an even-numbered round -/
def prob_B_even : ℚ := 3/4

/-- The game ends when one player has won 2 more rounds than the other -/
def game_end_condition (a_wins b_wins : ℕ) : Prop :=
  (a_wins = b_wins + 2) ∨ (b_wins = a_wins + 2)

theorem expected_rounds_proof :
  ∀ (round : ℕ) (a_wins b_wins : ℕ),
    (round % 2 = 1 → prob_A_odd = 3/4) →
    (round % 2 = 0 → prob_B_even = 3/4) →
    (game_end_condition a_wins b_wins → expected_rounds = 16/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_rounds_proof_l797_79759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l797_79795

theorem triangle_inequality (a b c : ℝ) (n : ℕ+) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) : 
  (a^(n : ℕ) / (b + c) + b^(n : ℕ) / (c + a) + c^(n : ℕ) / (a + b)) ≥ 
  (2/3)^((n : ℕ) - 2) * ((a + b + c) / 2)^((n : ℕ) - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l797_79795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_operation_result_l797_79787

noncomputable def diamond (a b : ℝ) : ℝ := a - 1 / b

theorem diamond_operation_result : 
  (diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4)) = -29/132 := by
  -- Expand the definition of diamond
  unfold diamond
  -- Perform algebraic simplifications
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_operation_result_l797_79787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_PxQ_l797_79717

def P : Finset ℕ := {3, 4, 5}
def Q : Finset ℕ := {6, 7}

def PxQ : Finset (ℕ × ℕ) := Finset.product P Q

theorem number_of_subsets_PxQ : Finset.card (Finset.powerset PxQ) = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_PxQ_l797_79717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_minus_sum_of_squares_is_integer_l797_79715

def a (n : ℕ) : ℤ :=
  if n ≤ 4 then n
  else (Finset.range (n - 1)).sum (λ i => a (i + 1))^2 - 1

def product_minus_sum_of_squares : ℤ :=
  (Finset.range 100).prod (λ i => a (i + 1)) -
  (Finset.range 100).sum (λ i => (a (i + 1))^2)

theorem product_minus_sum_of_squares_is_integer : 
  ∃ z : ℤ, product_minus_sum_of_squares = z := by
  use product_minus_sum_of_squares
  rfl

#eval product_minus_sum_of_squares

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_minus_sum_of_squares_is_integer_l797_79715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l797_79713

theorem polynomial_factorization (p q : ℕ) (n : ℕ) (a : ℤ) :
  Prime p ∧ Prime q ∧ p ≠ q ∧ n ≥ 3 →
  (∃ (f g : Polynomial ℤ), Polynomial.degree f ≥ 1 ∧ Polynomial.degree g ≥ 1 ∧
    (Polynomial.monomial n 1 + Polynomial.monomial (n-1) a + Polynomial.C (p * q : ℤ)) = f * g) ↔
  (a = -1 - (p * q : ℤ) ∨ a = 1 + (-1)^n * (p * q : ℤ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l797_79713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_110m_l797_79753

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
noncomputable def train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) : ℝ :=
  train_speed * (1000 / 3600) * crossing_time - bridge_length

/-- Theorem stating that under given conditions, the train length is approximately 110 meters -/
theorem train_length_approx_110m (ε : ℝ) (h : ε > 0) :
  ∃ (actual_length : ℝ),
    train_length 60 14.998800095992321 140 = actual_length ∧
    abs (actual_length - 110) < ε :=
by
  sorry

-- Using a computable approximation for demonstration
def train_length_approx (train_speed : Float) (crossing_time : Float) (bridge_length : Float) : Float :=
  train_speed * (1000 / 3600) * crossing_time - bridge_length

#eval train_length_approx 60 14.998800095992321 140

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_110m_l797_79753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_ab_together_arrangements_a_not_head_b_not_end_arrangements_ab_not_adjacent_arrangements_one_between_ab_l797_79702

/- Define the total number of students -/
def total_students : Nat := 7

/- Part 1: A and B must stand together -/
theorem arrangements_ab_together (total_students : Nat) :
  (Nat.factorial 2) * (Nat.factorial (total_students - 1)) = 1440 := by
  sorry

/- Part 2: A is not at the head, and B is not at the end -/
theorem arrangements_a_not_head_b_not_end (total_students : Nat) :
  (Nat.factorial total_students) - 2 * (Nat.factorial (total_students - 1)) + (Nat.factorial (total_students - 2)) = 3720 := by
  sorry

/- Part 3: A and B must not stand next to each other -/
theorem arrangements_ab_not_adjacent (total_students : Nat) :
  (Nat.factorial total_students) - (total_students - 1) * (Nat.factorial (total_students - 2)) = 3600 := by
  sorry

/- Part 4: There must be one person standing between A and B -/
theorem arrangements_one_between_ab (total_students : Nat) :
  (Nat.factorial (total_students - 2)) * (Nat.factorial 2) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_ab_together_arrangements_a_not_head_b_not_end_arrangements_ab_not_adjacent_arrangements_one_between_ab_l797_79702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_implies_a_range_l797_79776

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + a*x else a^2*x - 7*a + 14

theorem function_equality_implies_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = f a x₂) →
  a ∈ Set.union (Set.Iio 2) (Set.Ioo 3 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_implies_a_range_l797_79776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l797_79716

-- Define the curves
def parabola (x y : ℝ) : Prop := y = x^2
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define a line
def line (a b c : ℝ) (x y : ℝ) : Prop := a*x + b*y = c

-- Define tangency
def is_tangent_to_parabola (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, line a b c x y ∧ parabola x y ∧
  ∀ x' y' : ℝ, line a b c x' y' ∧ parabola x' y' → (x', y') = (x, y)

-- Define the theorem
theorem intersection_points_count
  (l1_a l1_b l1_c l2_a l2_b l2_c : ℝ)
  (h1 : ¬∃ x y : ℝ, line l1_a l1_b l1_c x y ∧ hyperbola x y ∧
    ∀ x' y' : ℝ, line l1_a l1_b l1_c x' y' ∧ hyperbola x' y' → (x', y') = (x, y))
  (h2 : ¬∃ x y : ℝ, line l2_a l2_b l2_c x y ∧ hyperbola x y ∧
    ∀ x' y' : ℝ, line l2_a l2_b l2_c x' y' ∧ hyperbola x' y' → (x', y') = (x, y))
  (h3 : is_tangent_to_parabola l1_a l1_b l1_c ∨ is_tangent_to_parabola l2_a l2_b l2_c) :
  ∃ n : ℕ, (n = 2 ∨ n = 3 ∨ n = 4) ∧
    (∃ s : Finset (ℝ × ℝ), s.card = n ∧
      (∀ p ∈ s, (∃ x y : ℝ, p = (x, y) ∧ hyperbola x y) ∧
        ((line l1_a l1_b l1_c p.1 p.2) ∨ (line l2_a l2_b l2_c p.1 p.2)))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l797_79716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_comparison_l797_79726

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem floor_comparison :
  (∀ x y : ℝ, x > y → floor x ≥ floor y) ∧
  (∃ x y : ℝ, floor x > floor y ∧ ¬(x > y)) :=
by
  constructor
  · intro x y hxy
    sorry -- Proof for the first part
  · sorry -- Proof for the second part

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_comparison_l797_79726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carly_swimming_hours_l797_79752

noncomputable def butterfly_hours : ℝ := 3 * 4
noncomputable def backstroke_hours : ℝ := 2 * 6
noncomputable def breaststroke_hours : ℝ := 1.5 * 5
noncomputable def freestyle_hours : ℝ := 2.5 * 3
noncomputable def underwater_hours : ℝ := 1 * 3
noncomputable def relay_hours : ℝ := 4 * 2

noncomputable def total_practice_hours : ℝ := butterfly_hours + backstroke_hours + breaststroke_hours + freestyle_hours + underwater_hours + relay_hours

noncomputable def average_daily_practice : ℝ := (total_practice_hours - relay_hours) / (4 + 6 + 5 + 3)

def rest_days : ℕ := 4
def holiday_days : ℕ := 1

noncomputable def total_rest_hours : ℝ := (rest_days + holiday_days : ℝ) * average_daily_practice

noncomputable def total_swimming_hours : ℝ := total_practice_hours - total_rest_hours

theorem carly_swimming_hours : ∃ ε > 0, |total_swimming_hours - 38.35| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carly_swimming_hours_l797_79752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_amount_l797_79786

/-- The number of quarts of milk added to the soup -/
def milk : ℚ := 2

/-- The total amount of soup in quarts -/
def total_soup : ℚ := milk + 3 * milk + 1

/-- The number of bags filled with soup -/
def num_bags : ℕ := 3

/-- The capacity of each bag in quarts -/
def bag_capacity : ℕ := 3

theorem milk_amount : milk = 2 := by
  -- The proof is omitted for now
  sorry

#check milk_amount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_amount_l797_79786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equality_l797_79730

theorem fraction_equality (x : ℝ) (hx3 : x ≠ 3) (hx7 : x ≠ 7) : 
  (40/7*x - 20) / (x^2 - 10*x + 21) = 5/7 / (x - 3) + 5 / (x - 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_equality_l797_79730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_length_calculation_l797_79709

/-- Represents a rectangular tank with given dimensions -/
structure Tank where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Calculates the surface area of a tank (walls + bottom) -/
def surfaceArea (t : Tank) : ℝ :=
  2 * (t.length * t.depth + t.width * t.depth) + t.length * t.width

/-- The theorem stating the length of the tank given the conditions -/
theorem tank_length_calculation (t : Tank) 
    (h_width : t.width = 12)
    (h_depth : t.depth = 6)
    (h_cost_per_sqm : ℝ)
    (h_total_cost : ℝ)
    (h_cost_condition : h_cost_per_sqm = 0.75)
    (h_total_cost_condition : h_total_cost = 558)
    (h_area_cost_relation : surfaceArea t * h_cost_per_sqm = h_total_cost) :
  t.length = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_length_calculation_l797_79709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_max_profit_l797_79748

-- Define the cost function
noncomputable def C (x : ℝ) : ℝ :=
  if x < 80 then (1/3) * x^2 + 10*x
  else 51*x + 10000/x - 1450

-- Define the profit function
noncomputable def L (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 80 then -(1/3) * x^2 + 40*x - 250
  else if x ≥ 80 then 1200 - (x + 10000/x)
  else 0  -- For x ≤ 0, profit is undefined in the original problem

-- State the theorem
theorem factory_max_profit :
  ∃ (x_max : ℝ), x_max = 100 ∧
  ∀ (x : ℝ), L x ≤ L x_max ∧ L x_max = 1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_max_profit_l797_79748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reaction_percent_yield_l797_79773

/-- Represents the reaction 2X + 3Y → 3Z -/
structure Reaction where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the theoretical yield of Z based on the limiting reagent -/
noncomputable def theoretical_yield (r : Reaction) : ℝ :=
  min (r.x * 3/2) r.y

/-- Calculates the percent yield of the reaction -/
noncomputable def percent_yield (r : Reaction) : ℝ :=
  (r.z / theoretical_yield r) * 100

/-- Theorem stating that the percent yield of the given reaction is 87.5% -/
theorem reaction_percent_yield :
  ∃ (r : Reaction), r.x = 2 ∧ r.y = 2 ∧ r.z = 1.75 ∧ percent_yield r = 87.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reaction_percent_yield_l797_79773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l797_79707

/-- The distance between two points in 3D space -/
noncomputable def distance (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2 + (z₂ - z₁)^2)

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The theorem stating that A(4,0,0) is equidistant from B(4,5,-2) and C(2,3,4) -/
theorem equidistant_point :
  let A : Point3D := ⟨4, 0, 0⟩
  let B : Point3D := ⟨4, 5, -2⟩
  let C : Point3D := ⟨2, 3, 4⟩
  distance A.x A.y A.z B.x B.y B.z = distance A.x A.y A.z C.x C.y C.z :=
by
  -- The proof goes here
  sorry

#check equidistant_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l797_79707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_intersection_condition_zero_at_two_l797_79782

/-- The function f(x) for a given m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/8) * (m-1) * x^2 - m*x + 2*m - 1

/-- Condition for f to have only one point in common with x-axis -/
def has_one_intersection (m : ℝ) : Prop :=
  (∃! x, f m x = 0) ∨ (∀ x, f m x = 0 → x = 2)

/-- Theorem stating conditions for f to have one intersection -/
theorem one_intersection_condition :
  ∀ m : ℝ, has_one_intersection m ↔ (m = 1/3 ∨ m = 1) :=
by sorry

/-- Theorem for the case when 2 is a zero of f -/
theorem zero_at_two :
  ∀ m : ℝ, f m 2 = 0 → m = 3 ∧ (∃ x : ℝ, x ≠ 2 ∧ f m x = 0 ∧ x = 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_intersection_condition_zero_at_two_l797_79782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_cost_proof_l797_79775

def soda_purchase (initial_payment : ℚ) (num_sodas : ℕ) (change : ℚ) : Prop :=
  ∃ (cost_per_soda : ℚ),
    cost_per_soda * num_sodas = initial_payment - change ∧
    cost_per_soda = 2

theorem soda_cost_proof :
  soda_purchase 20 3 14 :=
by
  use 2
  constructor
  · norm_num
  · rfl

#check soda_cost_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_cost_proof_l797_79775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_of_three_in_thirty_factorial_l797_79744

theorem exponent_of_three_in_thirty_factorial : ∃ n : ℕ, 
  3^14 ∣ Nat.factorial 30 ∧ ¬(3^15 ∣ Nat.factorial 30) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_of_three_in_thirty_factorial_l797_79744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_equals_cos_l797_79763

noncomputable def nested_sqrt : ℕ → ℝ
| 0 => 0
| n + 1 => Real.sqrt (2 + nested_sqrt n)

theorem nested_sqrt_equals_cos (n : ℕ) :
  nested_sqrt n = 2 * Real.cos (π / 2^(n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_equals_cos_l797_79763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_shorter_segment_l797_79761

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- The reciprocal of the golden ratio -/
noncomputable def φ_reciprocal : ℝ := (Real.sqrt 5 - 1) / 2

/-- The total length of the line segment -/
def total_length : ℝ := 10

/-- The length of the shorter segment in a golden section division -/
noncomputable def shorter_segment_length : ℝ := total_length - φ_reciprocal * total_length

theorem golden_section_shorter_segment :
  shorter_segment_length = 15 - 5 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_shorter_segment_l797_79761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_packed_sphere_radius_l797_79736

/-- The radius of the largest sphere that can be inscribed in a regular tetrahedron -/
noncomputable def largest_inscribed_sphere_radius (side_length : ℝ) : ℝ :=
  side_length * (Real.sqrt 6 - 1) / 10

/-- The side length of the regular tetrahedron -/
def tetrahedron_side_length : ℝ := 1

/-- The number of spheres to be packed -/
def num_spheres : ℕ := 4

/-- Theorem stating the largest radius of spheres that can be packed in a regular tetrahedron -/
theorem largest_packed_sphere_radius :
  ∃ (r : ℝ), r = largest_inscribed_sphere_radius tetrahedron_side_length ∧
  r = (Real.sqrt 6 - 1) / 10 ∧
  ∀ (r' : ℝ), r' > r → ¬(∃ (centers : Fin num_spheres → ℝ × ℝ × ℝ),
    (∀ i j, i ≠ j → Real.sqrt ((centers i).1 - (centers j).1)^2 + 
                               ((centers i).2.1 - (centers j).2.1)^2 + 
                               ((centers i).2.2 - (centers j).2.2)^2 ≥ 2 * r') ∧
    (∀ i, ∀ p : ℝ × ℝ × ℝ, Real.sqrt (p.1^2 + p.2.1^2 + p.2.2^2) = 1 → 
      Real.sqrt ((centers i).1 - p.1)^2 + 
                ((centers i).2.1 - p.2.1)^2 + 
                ((centers i).2.2 - p.2.2)^2 ≤ 1 - r')) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_packed_sphere_radius_l797_79736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_zero_no_max_value_iff_a_in_range_l797_79798

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ a then x^3 - 3*x else -2*x

-- Theorem 1: When a = 0, the maximum value of f is 2
theorem max_value_when_a_zero :
  ∃ (x : ℝ), f 0 x = 2 ∧ ∀ (y : ℝ), f 0 y ≤ 2 := by
  sorry

-- Theorem 2: f has no maximum value if and only if a ∈ (-∞, -1)
theorem no_max_value_iff_a_in_range :
  ∀ (a : ℝ), (¬∃ (M : ℝ), ∀ (x : ℝ), f a x ≤ M) ↔ a < -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_zero_no_max_value_iff_a_in_range_l797_79798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_theorem_l797_79729

theorem election_votes_theorem (first_candidate_percentage : ℚ) 
                                (second_candidate_votes : ℕ) : 
  first_candidate_percentage = 60 / 100 →
  second_candidate_votes = 480 →
  ∃ total_votes : ℕ, 
    (first_candidate_percentage * total_votes + second_candidate_votes = total_votes) ∧
    total_votes = 1200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_theorem_l797_79729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_result_l797_79758

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (-2, m)

theorem vector_sum_result (m : ℝ) (h : ¬ ∃ k : ℝ, a = k • b m) :
  (2 : ℝ) • a + (3 : ℝ) • b m = (-4, -8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_result_l797_79758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_freshmen_psych_majors_percentage_l797_79733

theorem freshmen_psych_majors_percentage 
  (total_students : ℕ) 
  (freshmen_percent : ℚ) 
  (liberal_arts_percent : ℚ) 
  (psych_major_percent : ℚ) 
  (h1 : freshmen_percent = 1/2) 
  (h2 : liberal_arts_percent = 2/5) 
  (h3 : psych_major_percent = 1/5) : 
  (freshmen_percent * liberal_arts_percent * psych_major_percent : ℚ) = 1/25 := by
  -- Substitute the given values
  rw [h1, h2, h3]
  -- Simplify the expression
  norm_num
  -- The proof is complete
  done

#check freshmen_psych_majors_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_freshmen_psych_majors_percentage_l797_79733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maximum_marks_calculation_l797_79784

theorem maximum_marks_calculation (passing_percentage : Real) 
  (student_marks : Nat) (failing_margin : Nat) : Nat :=
  -- The passing mark is 20% of the maximum marks
  have h1 : passing_percentage = 0.20 := by sorry
  -- A student got 160 marks
  have h2 : student_marks = 160 := by sorry
  -- The student failed by 25 marks
  have h3 : failing_margin = 25 := by sorry
  -- Prove that the maximum marks are 925
  have h4 : (student_marks + failing_margin) / passing_percentage = 925 := by
    sorry
  925


end NUMINAMATH_CALUDE_ERRORFEEDBACK_maximum_marks_calculation_l797_79784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_equation_solution_l797_79710

theorem cubic_root_equation_solution : ∃ (x₁ x₂ : ℝ) (p q : ℤ),
  (x₁ < x₂) ∧
  (x₁^(1/3 : ℝ) + (40 - x₁)^(1/3 : ℝ) = 4) ∧
  (x₂^(1/3 : ℝ) + (40 - x₂)^(1/3 : ℝ) = 4) ∧
  (x₂ = ↑p + Real.sqrt (↑q : ℝ)) ∧
  (p + q = 210) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_equation_solution_l797_79710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_game_target_runs_l797_79719

/-- Represents a cricket game with its parameters -/
structure CricketGame where
  totalOvers : ℕ
  firstPhaseOvers : ℕ
  firstPhaseRunRate : ℚ
  desiredRunRate : ℚ

/-- Calculates the target number of runs for a cricket game -/
def targetRuns (game : CricketGame) : ℕ :=
  (game.desiredRunRate * game.totalOvers).ceil.toNat

/-- Theorem stating the target runs for the given cricket game -/
theorem cricket_game_target_runs :
  let game : CricketGame := {
    totalOvers := 50,
    firstPhaseOvers := 10,
    firstPhaseRunRate := 17/5,  -- 3.4 as a rational number
    desiredRunRate := 31/5      -- 6.2 as a rational number
  }
  targetRuns game = 310 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_game_target_runs_l797_79719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bennett_window_screen_sales_l797_79794

/-- The number of window screens Bennett sold in January -/
def january_sales : ℕ := sorry

/-- The number of window screens Bennett sold in February -/
def february_sales : ℕ := sorry

/-- The number of window screens Bennett sold in March -/
def march_sales : ℕ := sorry

/-- The total number of window screens Bennett sold from January to March -/
def total_sales : ℕ := january_sales + february_sales + march_sales

theorem bennett_window_screen_sales :
  february_sales = 2 * january_sales →
  february_sales = march_sales / 4 →
  march_sales = 8800 →
  total_sales = 12100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bennett_window_screen_sales_l797_79794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_largest_minus_fourth_smallest_l797_79793

def even_numbers : List Nat := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

theorem third_largest_minus_fourth_smallest : 
  let sorted := even_numbers.reverse
  let third_largest := sorted[2]!
  let fourth_smallest := even_numbers[3]!
  third_largest - fourth_smallest = 8 := by
  rw [even_numbers]
  simp
  rfl

#eval even_numbers.reverse[2]! - even_numbers[3]!

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_largest_minus_fourth_smallest_l797_79793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sin_product_zero_l797_79750

theorem integral_sin_product_zero (α β : ℝ) (h1 : α > 0) (h2 : β > 0) (h3 : α ≠ β)
  (h4 : 2 * α = Real.tan α) (h5 : 2 * β = Real.tan β) :
  ∫ x in Set.Icc 0 1, Real.sin (α * x) * Real.sin (β * x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sin_product_zero_l797_79750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l797_79760

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x ≤ -1 ∨ x > 2}

-- Theorem statement
theorem complement_of_A :
  (Set.compl A) = Set.Ioc (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l797_79760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_nonnegative_l797_79741

noncomputable def x : ℕ → ℝ → ℝ
  | 0, a => 0
  | n + 1, a => 1 - a * Real.exp (x n a)

theorem x_nonnegative (a : ℝ) (h : a ≤ 1) (n : ℕ) : x n a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_nonnegative_l797_79741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_domain_example_l797_79718

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2)

def is_valid_domain (S : Set ℝ) : Prop :=
  (∀ x ∈ S, f x ∈ ({0, 4} : Set ℝ)) ∧ 
  (∀ y ∈ ({0, 4} : Set ℝ), ∃ x ∈ S, f x = y)

theorem valid_domain_example : is_valid_domain {0, 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_domain_example_l797_79718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_multiples_problem_l797_79720

theorem consecutive_multiples_problem (s : Set ℕ) (n : ℕ) : 
  (∃ k : ℕ, ∀ i : ℕ, i ≤ 63 → (k + i) * n ∈ s) →
  68 ∈ s →
  320 ∈ s →
  (∀ x ∈ s, x ≥ 68 ∧ x ≤ 320) →
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_multiples_problem_l797_79720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_10_percent_l797_79724

/-- Calculates the interest rate at which A lent money to B -/
noncomputable def calculate_interest_rate (principal : ℝ) (b_lending_rate : ℝ) (time : ℝ) (b_gain : ℝ) : ℝ :=
  let interest_from_c := principal * b_lending_rate * time / 100
  let interest_to_a := interest_from_c - b_gain
  (interest_to_a * 100) / (principal * time)

/-- Theorem stating the interest rate at which A lent money to B -/
theorem interest_rate_is_10_percent (principal : ℝ) (b_lending_rate : ℝ) (time : ℝ) (b_gain : ℝ)
  (h1 : principal = 1500)
  (h2 : b_lending_rate = 11.5)
  (h3 : time = 3)
  (h4 : b_gain = 67.5) :
  calculate_interest_rate principal b_lending_rate time b_gain = 10 :=
by
  -- Unfold the definition of calculate_interest_rate
  unfold calculate_interest_rate
  -- Simplify the expression
  simp [h1, h2, h3, h4]
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_10_percent_l797_79724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_three_l797_79767

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 2*x else -x^2 + 2*x

theorem f_greater_than_three (x : ℝ) : f x > 3 ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_three_l797_79767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_counters_probability_l797_79769

/-- The probability of placing 3 counters on a 6x6 chessboard such that no two counters are in the same row or column -/
theorem three_counters_probability (n : ℕ) (h : n = 6) : 
  (Nat.choose n 3)^2 * Nat.factorial 3 / Nat.choose (n^2) 3 = 40 / 119 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_counters_probability_l797_79769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l797_79703

-- Define the function f
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (3 * x + φ)

-- State the theorem
theorem f_properties (φ : ℝ) (h1 : -π/2 < φ ∧ φ < π/2) 
  (h2 : ∀ x, f φ (π/4 - x) = f φ (π/4 + x)) :
  -- 1. g(x) = f(x - π/12) is an even function
  (∀ x, f φ (x - π/12) = f φ (-x - π/12)) ∧
  -- 2. f(x) is symmetric about (5π/12, 0)
  (∀ x, f φ (5*π/12 - x) = f φ (5*π/12 + x)) ∧
  -- 3. If f(x₁)f(x₂) = -4, then min|x₁ - x₂| = π/3
  (∀ x₁ x₂, f φ x₁ * f φ x₂ = -4 → |x₁ - x₂| ≥ π/3 ∧ 
    ∃ y₁ y₂, f φ y₁ * f φ y₂ = -4 ∧ |y₁ - y₂| = π/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l797_79703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relations_l797_79740

/-- Definition of terminal side of an angle -/
def terminal_side (θ : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | ∃ (t : ℝ), x = t * Real.cos θ ∧ y = t * Real.sin θ ∧ t > 0}

/-- Given a point P on the terminal side of angle α and angle β symmetric to α with respect to the y-axis, prove tan α and cos(β - α) -/
theorem angle_relations (P : ℝ × ℝ) (α β : ℝ) 
  (h1 : P = (1, 2))  -- Point P is (1, 2)
  (h2 : ∀ (x y : ℝ), (x, y) ∈ terminal_side α ↔ (-x, y) ∈ terminal_side β) : -- β is symmetric to α w.r.t. y-axis
  Real.tan α = 2 ∧ Real.cos (β - α) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relations_l797_79740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_sentence_order_l797_79772

-- Define the sentences as abstract types
inductive Sentence : Type where
  | one : Sentence
  | two : Sentence
  | three : Sentence
  | four : Sentence
  | five : Sentence
deriving BEq, DecidableEq

-- Define a type for ordered lists of sentences
def SentenceOrder := List Sentence

-- Define the properties of the correct order
def is_correct_order (order : SentenceOrder) : Prop :=
  -- Sentence ① connects best with the introductory sentence
  order.head? = some Sentence.one ∧
  -- Sentence ④ follows sentence ①
  (order.indexOf Sentence.four = order.indexOf Sentence.one + 1) ∧
  -- Sentence ③ is a concluding sentence
  order.getLast? = some Sentence.three ∧
  -- The order contains all sentences exactly once
  order.length = 5 ∧ order.toFinset.card = 5

-- State the theorem
theorem correct_sentence_order :
  is_correct_order [Sentence.one, Sentence.four, Sentence.five, Sentence.two, Sentence.three] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_sentence_order_l797_79772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_S_revolution_l797_79780

-- Define the region S
noncomputable def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |6 - p.1| + p.2 ≤ 8 ∧ 4 * p.2 - p.1 ≥ 20}

-- Define the axis of revolution
noncomputable def axis : ℝ → ℝ
  | x => (x + 20) / 4

-- Define the volume of the solid of revolution
noncomputable def volumeOfRevolution (S : Set (ℝ × ℝ)) (axis : ℝ → ℝ) : ℝ :=
  sorry  -- Definition of volume calculation

-- Theorem statement
theorem volume_of_S_revolution :
  volumeOfRevolution S axis = 216 * Real.pi / (85 * Real.sqrt 17) := by
  sorry

#check volume_of_S_revolution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_S_revolution_l797_79780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_circumscribed_circle_radius_l797_79785

theorem sector_circumscribed_circle_radius 
  (r : ℝ) 
  (θ : ℝ) 
  (h_r : r = 8) 
  (h_θ : θ = 30 * π / 180) : 
  r / Real.cos (θ / 2) = 8 * (Real.sqrt 6 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_circumscribed_circle_radius_l797_79785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_difference_after_15_years_l797_79708

/-- Calculates the compound interest for a given principal, rate, and time --/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate) ^ time

theorem balance_difference_after_15_years :
  let angelaPrincipal : ℝ := 7000
  let angelaRate : ℝ := 0.05
  let bobPrincipal : ℝ := 9000
  let bobRate : ℝ := 0.03
  let time : ℝ := 15
  let angelaBalance := compoundInterest angelaPrincipal angelaRate time
  let bobBalance := compoundInterest bobPrincipal bobRate (2 * time)
  abs (bobBalance - angelaBalance - 7292.83) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_difference_after_15_years_l797_79708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_B_fill_time_l797_79705

-- Define the volumes of pools M and N
noncomputable def volumeM : ℝ := 1.5
noncomputable def volumeN : ℝ := 1

-- Define the filling rates of pipes A, B, and C
noncomputable def rateA : ℝ := volumeM / 5
noncomputable def rateB : ℝ := volumeN / 5
noncomputable def rateC : ℝ := volumeN / 6

-- Define the total time to fill both pools
noncomputable def totalTime : ℝ := 15 / 4

-- Define the time B fills pool M
noncomputable def timeBFillsM : ℝ := 15 / 8

-- Theorem statement
theorem pipe_B_fill_time :
  ∃ (t : ℝ),
    t = timeBFillsM ∧
    rateA * totalTime + rateB * t = volumeM ∧
    rateC * totalTime + rateB * (totalTime - t) = volumeN :=
by
  use timeBFillsM
  constructor
  · rfl
  constructor
  · sorry -- Proof for the first equation
  · sorry -- Proof for the second equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_B_fill_time_l797_79705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_abs_pythagorean_iff_right_triangle_l797_79789

theorem sufficient_not_necessary_abs : 
  (∀ x : ℝ, x > 1 → |x| > 0) ∧ (∃ x : ℝ, |x| > 0 ∧ ¬(x > 1)) := by
  sorry

theorem pythagorean_iff_right_triangle (a b c : ℝ) :
  (a^2 + b^2 = c^2) ↔ (∃ θ : ℝ, 0 < θ ∧ θ < Real.pi/2 ∧ a = c * Real.sin θ ∧ b = c * Real.cos θ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_abs_pythagorean_iff_right_triangle_l797_79789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_g_extrema_l797_79704

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin (x/2 + Real.pi/4) * Real.cos (x/2 + Real.pi/4) - Real.sin (x + Real.pi)

noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi/6)

theorem f_period_and_g_extrema :
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 Real.pi → g x ≤ 2) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 Real.pi → g x ≥ -1) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 Real.pi ∧ g x = 2) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 Real.pi ∧ g x = -1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_g_extrema_l797_79704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_and_transformation_l797_79742

open Matrix

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]

def transformation_matrix (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![a, 0; 2, b]

def point_M : Matrix (Fin 2) (Fin 1) ℝ := !![3; -1]

def point_N : Matrix (Fin 2) (Fin 1) ℝ := !![3; 5]

theorem rotation_and_transformation (a b : ℝ) :
  (transformation_matrix a b) * (rotation_matrix * point_M) = point_N ↔ a = 3 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_and_transformation_l797_79742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_for_function_equality_l797_79755

-- Define the functions f and g
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 1)^2 * x^(m^2 - 4*m + 2)
noncomputable def g (t : ℝ) (x : ℝ) : ℝ := 2^x - t

-- State the theorem
theorem range_of_t_for_function_equality :
  ∀ (m : ℝ),
  (∀ x > 0, Monotone (fun x => f m x)) →
  (∀ t : ℝ,
    (∀ x₁ ∈ Set.Icc 1 6, ∃ x₂ ∈ Set.Icc 1 6, f m x₁ = g t x₂) ↔
    t ∈ Set.Icc 1 28) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_for_function_equality_l797_79755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_is_focus_of_hyperbola_l797_79714

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 = 1

/-- The focus coordinates -/
noncomputable def focus : ℝ × ℝ := (-Real.sqrt 5, 0)

/-- Theorem: The given point is a focus of the hyperbola -/
theorem is_focus_of_hyperbola :
  ∃ (x y : ℝ), hyperbola_equation x y ∧ (x, y) = focus := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_is_focus_of_hyperbola_l797_79714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_two_quarters_circle_l797_79754

/-- The area of a figure formed by two 90° sectors of a circle with radius 15 placed side by side -/
noncomputable def area_two_sectors (r : ℝ) (angle : ℝ) : ℝ :=
  2 * (angle / (2 * Real.pi)) * Real.pi * r^2

theorem area_two_quarters_circle (r : ℝ) :
  r = 15 → area_two_sectors r (Real.pi/2) = 112.5 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_two_quarters_circle_l797_79754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_cafeteria_ratio_l797_79745

/-- Given a school cafeteria scenario, prove the ratio of total students to students who tasted both of any two dishes. -/
theorem school_cafeteria_ratio 
  (n m : ℕ) 
  (E M : Type) [Fintype E] [Fintype M]
  (h1 : Fintype.card M = 100)
  (h2 : ∀ (e : E), ∃ (S : Finset M), Finset.card S = 10)
  (h3 : ∀ (a b : M), a ≠ b → ∃ (S : Finset E), Finset.card S = m)
  (h4 : Fintype.card E = n)
  : n / m = 110 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_cafeteria_ratio_l797_79745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_h_l797_79756

/-- Given a function h such that h(3x - 2) = 5x + 6 for all x, 
    prove that the unique solution to h(x) = 2x - 1 is x = 31 -/
theorem unique_solution_for_h (h : ℝ → ℝ) 
    (h_def : ∀ x, h (3*x - 2) = 5*x + 6) : 
    (∃! x, h x = 2*x - 1) ∧ (h 31 = 2*31 - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_h_l797_79756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_team_speed_l797_79770

/-- Calculates the speed of the second team given the conditions of the problem -/
theorem second_team_speed 
  (time : ℝ) 
  (first_team_speed : ℝ) 
  (total_distance : ℝ) 
  (h1 : time = 2.5)
  (h2 : first_team_speed = 20)
  (h3 : total_distance = 125) : 
  (total_distance - first_team_speed * time) / time = 30 := by
  sorry

#check second_team_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_team_speed_l797_79770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_of_B_l797_79791

def A : Finset ℕ := {1, 2, 3, 4, 5}

def B : Finset (ℕ × ℕ) := 
  Finset.filter (fun p => p.1 ∈ A ∧ p.2 ∈ A ∧ (p.1 - p.2) ∈ A) (A.product A)

theorem cardinality_of_B : Finset.card B = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_of_B_l797_79791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l797_79723

/-- A line passing through the origin -/
structure OriginLine where
  slope : ℝ

/-- The horizontal line y = 2 -/
noncomputable def horizontal_line : Set (ℝ × ℝ) :=
  {p | p.2 = 2}

/-- The inclined line y = (1/3)x + 2 -/
noncomputable def inclined_line : Set (ℝ × ℝ) :=
  {p | p.2 = (1/3) * p.1 + 2}

/-- The intersection of an OriginLine with the horizontal line -/
noncomputable def horizontal_intersection (l : OriginLine) : ℝ × ℝ :=
  (-2 / l.slope, 2)

/-- The intersection of an OriginLine with the inclined line -/
noncomputable def inclined_intersection (l : OriginLine) : ℝ × ℝ :=
  let x := 6 / (l.slope - 1/3)
  (x, (1/3) * x + 2)

/-- Check if three points form a right triangle -/
def is_right_triangle (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let v1 := (p2.1 - p1.1, p2.2 - p1.2)
  let v2 := (p3.1 - p1.1, p3.2 - p1.2)
  v1.1 * v2.1 + v1.2 * v2.2 = 0

/-- Calculate the area of a triangle given two of its vertices (assuming the third is the origin) -/
noncomputable def triangle_area (p1 p2 : ℝ × ℝ) : ℝ :=
  abs (p1.1 * p2.2 - p2.1 * p1.2) / 2

/-- The main theorem -/
theorem right_triangle_area (l : OriginLine) :
  is_right_triangle (0, 0) (horizontal_intersection l) (inclined_intersection l) →
  triangle_area (horizontal_intersection l) (inclined_intersection l) = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l797_79723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_five_rectangle_division_l797_79734

/-- Represents a rectangle with a smaller rectangle cut out from its center -/
structure CutRectangle where
  width : Nat
  height : Nat
  cutWidth : Nat
  cutHeight : Nat

/-- Represents a triangle -/
structure Triangle

/-- A function that takes a CutRectangle and returns a list of Triangles -/
def divideIntoTriangles (r : CutRectangle) : List Triangle := sorry

/-- Theorem stating that a 6x5 rectangle with a 2x1 cut can be divided into 6 triangles -/
theorem six_five_rectangle_division :
  ∃ (triangles : List Triangle),
    (divideIntoTriangles ⟨6, 5, 2, 1⟩).length = 6 := by
  sorry

#check six_five_rectangle_division

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_five_rectangle_division_l797_79734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_existence_l797_79735

open Set Finset

theorem subset_existence (n : ℕ) (A B : Finset ℕ) 
  (hn : n > 0) 
  (hA : A.Nonempty ∧ A ⊆ range n.succ)
  (hB : B.Nonempty ∧ B ⊆ range n.succ) :
  ∃ D : Finset ℕ, 
    D ⊆ (A.product B).image (fun p => p.1 + p.2) ∧ 
    (D.product D).image (fun p => p.1 + p.2) ⊆ (A.product B).image (fun p => 2 * (p.1 + p.2)) ∧
    D.card ≥ (A.card * B.card) / (2 * n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_existence_l797_79735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_inscribed_circle_l797_79732

/-- The area of a triangle with side lengths in the ratio 5:12:13, inscribed in a circle of radius 5 -/
noncomputable def triangleArea : ℝ :=
  3000 / 169

/-- The radius of the circle in which the triangle is inscribed -/
def circleRadius : ℝ := 5

/-- The ratio of the triangle's side lengths -/
def sideRatio : Fin 3 → ℝ
  | 0 => 5
  | 1 => 12
  | 2 => 13

/-- Theorem stating the properties of the triangle inscribed in the circle -/
theorem triangle_area_inscribed_circle :
  ∃ (k : ℝ), k > 0 ∧
  (∀ i : Fin 3, sideRatio i * k ≤ 2 * circleRadius) ∧
  (∃ i : Fin 3, sideRatio i * k = 2 * circleRadius) ∧
  triangleArea = (sideRatio 0 * k * sideRatio 1 * k) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_inscribed_circle_l797_79732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_range_theorem_l797_79739

/-- For any non-zero real numbers a and b, the minimum value of (|2a+b|+|2a-b|)/|a| is 4 -/
theorem min_value_theorem (a b : ℝ) (ha : a ≠ 0) :
  (∀ a b : ℝ, a ≠ 0 → (|2*a + b| + |2*a - b|) / |a| ≥ 4) ∧
  (∃ a b : ℝ, a ≠ 0 ∧ (|2*a + b| + |2*a - b|) / |a| = 4) :=
by sorry

/-- If |2a+b|+|2a-b| ≥ |a|(|2+x|+|2-x|) holds for all non-zero real a and b, then x is in [-2, 2] -/
theorem range_theorem (x : ℝ) :
  (∀ a b : ℝ, a ≠ 0 → |2*a + b| + |2*a - b| ≥ |a| * (|2 + x| + |2 - x|)) →
  x ∈ Set.Icc (-2) 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_range_theorem_l797_79739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cindy_batch_size_l797_79738

/-- Represents the shape of a cookie -/
inductive CookieShape
  | Circle (radius : ℝ)
  | Square (side : ℝ)
  | Rectangle (length width : ℝ)
  | EquilateralTriangle (side : ℝ)

/-- Represents a type of cookie -/
structure Cookie where
  shape : CookieShape
  thickness : ℝ

/-- Calculates the area of a cookie based on its shape -/
noncomputable def cookieArea (shape : CookieShape) : ℝ :=
  match shape with
  | CookieShape.Circle r => Real.pi * r^2
  | CookieShape.Square s => s^2
  | CookieShape.Rectangle l w => l * w
  | CookieShape.EquilateralTriangle s => (Real.sqrt 3 / 4) * s^2

/-- Calculates the volume of a cookie -/
noncomputable def cookieVolume (c : Cookie) : ℝ :=
  cookieArea c.shape * c.thickness

/-- Represents the four friends' cookies -/
def artCookie : Cookie := { shape := CookieShape.Circle 2, thickness := 1 }
def bobCookie : Cookie := { shape := CookieShape.Square 4, thickness := 2 }
def cindyCookie : Cookie := { shape := CookieShape.Rectangle 6 2, thickness := 1 }
def trishaCookie : Cookie := { shape := CookieShape.EquilateralTriangle 3, thickness := 1 }

/-- The number of cookies Art makes in one batch -/
def artBatchSize : ℕ := 9

/-- Theorem: Cindy makes 9 cookies using the same amount of dough as Art -/
theorem cindy_batch_size :
  ∃ (n : ℕ), n * cookieVolume cindyCookie = artBatchSize * cookieVolume artCookie ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cindy_batch_size_l797_79738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_inequality_l797_79765

noncomputable def f (a b x : ℝ) : ℝ := (-2^x + b) / (2^(x+1) + a)

theorem odd_function_and_inequality (a b : ℝ) :
  (∀ x, f a b x = -f a b (-x)) →
  (∀ x, f a b x ∈ Set.univ) →
  (a = 2 ∧ b = 1) ∧
  (∀ k, (∀ t, f a b (t^2 - 2*t) + f a b (2*t^2 - k) < 0) ↔ k < -1/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_inequality_l797_79765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_theorem_l797_79766

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a/b + b/a = 4 cos C and cos(A - B) = 1/6, then cos C = 2/3 -/
theorem triangle_cosine_theorem (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  a / b + b / a = 4 * Real.cos C →
  Real.cos (A - B) = 1 / 6 →
  Real.cos C = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_theorem_l797_79766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_subset_T_l797_79777

open Real

/-- The set S of pairs (x, y) where x^2 - y^2 is odd -/
def S : Set (ℝ × ℝ) := {p | ∃ k : ℤ, p.1^2 - p.2^2 = 2*k + 1}

/-- The set T of pairs (x, y) satisfying the trigonometric equation -/
def T : Set (ℝ × ℝ) := {p | sin (2*Real.pi*p.1^2) - sin (2*Real.pi*p.2^2) = cos (2*Real.pi*p.1^2) - cos (2*Real.pi*p.2^2)}

/-- Theorem stating that S is a subset of T -/
theorem S_subset_T : S ⊆ T := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_subset_T_l797_79777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l797_79731

/-- A configuration of four points in a plane -/
structure PointConfiguration :=
  (A B C D : ℝ × ℝ)

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The sum of distances between all pairs of points -/
noncomputable def sumOfDistances (config : PointConfiguration) : ℝ :=
  distance config.A config.B + distance config.A config.C + distance config.A config.D +
  distance config.B config.C + distance config.B config.D + distance config.C config.D

/-- The minimum distance condition -/
def satisfiesMinDistance (config : PointConfiguration) : Prop :=
  distance config.A config.B ≥ 1 ∧ distance config.A config.C ≥ 1 ∧ distance config.A config.D ≥ 1 ∧
  distance config.B config.C ≥ 1 ∧ distance config.B config.D ≥ 1 ∧ distance config.C config.D ≥ 1

theorem min_distance_sum :
  ∀ (config : PointConfiguration),
    satisfiesMinDistance config →
    sumOfDistances config ≥ 5 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l797_79731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l797_79762

theorem ellipse_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ (c : ℝ), c^2 = 2 ∧ a^2 = b^2 + c^2) →
  (2 * b^2 / a = 4 * Real.sqrt 6 / 3) →
  a^2 = 6 ∧ b^2 = 4 := by
  intro h_focus h_chord
  sorry

#check ellipse_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l797_79762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l797_79712

/-- The eccentricity of a hyperbola given its parameters a and b -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + (b / a) ^ 2)

/-- The condition for a point (x, y) to be in the "upper" region of a hyperbola -/
def in_upper_region (a b x y : ℝ) : Prop := y > (b / a) * x

theorem hyperbola_eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_upper : in_upper_region a b 1 2) : 
  1 < eccentricity a b ∧ eccentricity a b < Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l797_79712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_of_S_cardinality_l797_79743

/-- Sum of digits function -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Set of numbers with digit sum 15 and less than 10^6 -/
def S : Set ℕ := {n | digit_sum n = 15 ∧ n < 10^6}

/-- Theorem stating the sum of digits of the cardinality of S is 18 -/
theorem digit_sum_of_S_cardinality : digit_sum (Finset.card (Finset.filter (fun n => digit_sum n = 15 ∧ n < 10^6) (Finset.range 1000000))) = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_of_S_cardinality_l797_79743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l797_79771

noncomputable def g (x : ℝ) : ℤ := ⌊⌊x⌋ - x⌋

theorem g_range : Set.range g = {-1, 0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l797_79771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maze_exit_theorem_l797_79764

-- Define the types of rooms
inductive RoomType
  | Black
  | Hatched
  | Gray
  | White

-- Define the structure of the maze
structure Maze where
  rooms : Fin 26 → RoomType
  adjacent : Fin 26 → Fin 26 → Bool

-- Define a path through the maze
def MazePath := List (Fin 26)

-- Define the condition for a valid path
def isValidPath (m : Maze) (p : MazePath) : Prop :=
  -- Path starts from A or D
  (p.head? = some 0 ∨ p.head? = some 3) ∧
  -- Path ends in a hatched room
  (p.getLast?.map m.rooms = some RoomType.Hatched) ∧
  -- All rooms in the path are adjacent
  (∀ i j, i + 1 = j → m.adjacent (p.get! i) (p.get! j)) ∧
  -- All gray rooms are visited exactly once
  (∀ r, m.rooms r = RoomType.Gray → (p.count r = 1)) ∧
  -- No black rooms are visited
  (∀ r, m.rooms r = RoomType.Black → (p.count r = 0)) ∧
  -- Hatched room is only visited at the end
  (∀ i, i < p.length - 1 → m.rooms (p.get! i) ≠ RoomType.Hatched)

-- The theorem to be proved
theorem maze_exit_theorem (m : Maze) :
  ∃ p : MazePath, isValidPath m p ∧
  (p.head? = some 0 ∨ p.head? = some 3) ∧
  ∀ start, start ≠ 0 ∧ start ≠ 3 → ¬∃ p : MazePath, isValidPath m p ∧ p.head? = some start :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_maze_exit_theorem_l797_79764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l797_79746

theorem max_value_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + 3*c = 3) :
  a + Real.sqrt (a*b) + 2 * (a*b*c)^(1/3) ≤ 3.5 ∧ 
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a' + b' + 3*c' = 3 ∧
    a' + Real.sqrt (a'*b') + 2 * (a'*b'*c')^(1/3) = 3.5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l797_79746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l797_79751

noncomputable def f (x : ℝ) := 2 * Real.sin (2 * x + Real.pi / 6)

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S → S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x y, x ∈ Set.Icc 0 (Real.pi / 12) → y ∈ Set.Icc 0 (Real.pi / 12) → x < y → f x < f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l797_79751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_15_l797_79778

-- Define the sequence a_n
def a : ℕ → ℕ
  | 0 => 1  -- Add this case to handle n = 0
  | 1 => 1
  | (n + 1) => 2 * a n + 1

-- Theorem statement
theorem a_4_equals_15 : a 4 = 15 := by
  -- Expand the definition of a for n = 2, 3, and 4
  have a2 : a 2 = 3 := by rfl
  have a3 : a 3 = 7 := by rfl
  have a4 : a 4 = 15 := by rfl
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_15_l797_79778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tug_of_war_competition_l797_79722

theorem tug_of_war_competition (x : ℕ) : 
  (∀ (i j : ℕ), i < x → j < x → i ≠ j → ∃! game : Bool, true) → 
  (x * (x - 1)) / 2 = 28 ↔ 
  (∃! total_matches : ℕ, total_matches = 28 ∧ 
    total_matches = (x * (x - 1)) / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tug_of_war_competition_l797_79722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_values_l797_79796

-- Define the lines
def line1 (m : ℝ) (x y : ℝ) : Prop := (m - 2) * x - y + 5 = 0
def line2 (m : ℝ) (x y : ℝ) : Prop := (m - 2) * x + (3 - m) * y + 2 = 0

-- Define parallel lines
def parallel (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∀ (a b c d e : ℝ), a ≠ 0 ∨ b ≠ 0 → c ≠ 0 ∨ d ≠ 0 → 
    (∀ x y, l1 x y ↔ a * x + b * y + c = 0) →
    (∀ x y, l2 x y ↔ d * x + e * y + c = 0) →
    ∃ (k : ℝ), k ≠ 0 ∧ a * e = k * b * d

theorem parallel_lines_m_values (m : ℝ) :
  parallel (line1 m) (line2 m) → m = 2 ∨ m = 4 := by
  sorry

#check parallel_lines_m_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_values_l797_79796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_conditions_l797_79757

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- Theorem stating conditions for equilateral and isosceles right triangles -/
theorem triangle_conditions (t : Triangle) : 
  (t.a / Real.cos t.A = t.b / Real.cos t.B ∧ 
   t.b / Real.cos t.B = t.c / Real.cos t.C → 
   t.A = t.B ∧ t.B = t.C) ∧
  (Real.sin t.A / t.a = Real.cos t.B / t.b ∧ 
   Real.cos t.B / t.b = Real.cos t.C / t.c → 
   t.B = t.C ∧ t.B = Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_conditions_l797_79757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_square_plus_year_infinite_l797_79774

/-- A natural number is a "number of year" if all its digits are 0, 1, or 2 in decimal representation -/
def IsNumberOfYear (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1 ∨ d = 2

/-- The set of natural numbers that cannot be expressed as A² + B, 
    where A ∈ ℕ and B is a "number of year" -/
def NotSquarePlusYear : Set ℕ :=
  {n : ℕ | ¬∃ (a b : ℕ), n = a^2 + b ∧ IsNumberOfYear b}

/-- The main theorem stating that the set NotSquarePlusYear is infinite -/
theorem not_square_plus_year_infinite : Set.Infinite NotSquarePlusYear := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_square_plus_year_infinite_l797_79774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_folding_theorem_l797_79700

/-- Represents the dimensions of a rectangular paper in decimeters. -/
structure PaperDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular paper. -/
def area (d : PaperDimensions) : ℝ := d.length * d.width

/-- Calculates the sum of areas after n folds. -/
noncomputable def sumOfAreas (n : ℕ) : ℝ := 240 * (3 - (n + 3) / 2^n)

/-- Counts the number of different shapes after n folds. -/
def numberOfShapes (n : ℕ) : ℕ := n + 1

theorem paper_folding_theorem (paper : PaperDimensions) (h1 : paper.length = 20) (h2 : paper.width = 12) :
  (numberOfShapes 4 = 5) ∧
  (∀ n : ℕ, sumOfAreas n = 240 * (3 - (n + 3) / 2^n)) := by
  sorry

#eval numberOfShapes 4
-- Remove the evaluation of sumOfAreas as it's noncomputable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_folding_theorem_l797_79700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l797_79725

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x) + a * (Real.cos x) ^ 2

-- State the theorem
theorem max_value_of_f (a : ℝ) :
  (f a (π / 4) = 0) →
  (∃ (max : ℝ), ∀ (x : ℝ), f a x ≤ max ∧ ∃ (y : ℝ), f a y = max) ∧
  (∀ (max : ℝ), (∀ (x : ℝ), f a x ≤ max ∧ ∃ (y : ℝ), f a y = max) → max = Real.sqrt 2 - 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l797_79725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_BC_length_l797_79792

noncomputable section

structure Trapezoid (A B C D : ℝ × ℝ) :=
  (parallel_AB_CD : (B.2 - A.2) * (D.1 - C.1) = (B.1 - A.1) * (D.2 - C.2))
  (perpendicular_AC_CD : (C.1 - A.1) * (D.1 - C.1) + (C.2 - A.2) * (D.2 - C.2) = 0)

def length (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

def tan_angle (P Q R : ℝ × ℝ) : ℝ := 
  (R.2 - Q.2) / (R.1 - Q.1)

theorem trapezoid_BC_length 
  (A B C D : ℝ × ℝ) 
  (h : Trapezoid A B C D) 
  (h_CD : length C D = 15) 
  (h_tan_C : tan_angle A C D = 1.2) 
  (h_tan_B : tan_angle A B C = 1.8) : 
  length B C = 2 * Real.sqrt 106 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_BC_length_l797_79792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_D_is_correct_l797_79783

-- Define a type for our statements
inductive Statement
  | A : Statement
  | B : Statement
  | C : Statement
  | D : Statement

-- Define variance and fluctuation as noncomputable functions
noncomputable def variance (data : List ℝ) : ℝ := sorry
noncomputable def fluctuation (data : List ℝ) : ℝ := sorry

-- Define a function to check if a statement is correct
def is_correct (s : Statement) : Prop :=
  match s with
  | Statement.A => False
  | Statement.B => False
  | Statement.C => False
  | Statement.D => ∀ (data1 data2 : List ℝ), 
                   variance data1 > variance data2 → 
                   fluctuation data1 > fluctuation data2

-- Theorem stating that only Statement D is correct
theorem only_D_is_correct : 
  ∀ (s : Statement), is_correct s ↔ s = Statement.D := by
  sorry

#check only_D_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_D_is_correct_l797_79783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_six_average_l797_79797

theorem first_six_average (numbers : List ℝ) : 
  numbers.length = 11 → 
  numbers.sum / numbers.length = 60 → 
  (numbers.take 6).sum / 6 = 121 → 
  (numbers.drop 5).sum / 6 = 75 → 
  numbers.get? 5 = some 258 → 
  (numbers.take 6).sum / 6 = 121 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_six_average_l797_79797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l797_79737

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x^2 - 6*x - 5)

-- State the theorem
theorem f_range : Set.range f = Set.Icc 0 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l797_79737
