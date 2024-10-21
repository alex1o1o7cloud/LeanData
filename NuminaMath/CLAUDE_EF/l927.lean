import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l927_92799

/-- Represents an ellipse with foci on the y-axis and center at the origin -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- The standard equation of an ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop := y^2 / e.a^2 + x^2 / e.b^2 = 1

/-- The area of a triangle formed by a point on the ellipse and its foci -/
noncomputable def Ellipse.triangleArea (e : Ellipse) (x y : ℝ) : ℝ := 
  Real.sqrt (e.a^2 - e.b^2) * y

/-- Represents a line through two points -/
def line_through (p1 p2 : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p = (p1.1 + t * (p2.1 - p1.1), p1.2 + t * (p2.2 - p1.2))}

theorem ellipse_properties (e : Ellipse) 
  (h1 : e.a = 2)
  (h2 : ∀ x y, e.equation x y → e.triangleArea x y ≤ Real.sqrt 3)
  (h3 : e.eccentricity < Real.sqrt 2 / 2) :
  (∀ x y, e.equation x y ↔ y^2 / 4 + x^2 / 3 = 1) ∧
  ∃ A B M N : ℝ × ℝ, 
    e.equation A.1 A.2 ∧ 
    e.equation B.1 B.2 ∧ 
    (B.2 - A.2) / (B.1 - A.1) * 0 + A.2 = e.a / 2 ∧
    M.2 = 4 ∧ N.2 = 4 ∧
    (0, 8/5) ∈ line_through A M ∧ 
    (0, 8/5) ∈ line_through B N :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l927_92799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2008_eq_neg_2_l927_92756

-- Define the real-valued functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- f is an even function
axiom f_even : ∀ x : ℝ, f (-x) = f x

-- g is an odd function
axiom g_odd : ∀ x : ℝ, g (-x) = -g x

-- Relationship between f and g
axiom fg_relation : ∀ x : ℝ, g x = f (x - 1)

-- Given value for g(-1)
axiom g_neg_one : g (-1) = 2

theorem f_2008_eq_neg_2 : f 2008 = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2008_eq_neg_2_l927_92756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_of_first_fifteen_integers_l927_92716

def first_fifteen_integers : List ℕ := List.range 15 |>.map (· + 1)

theorem median_of_first_fifteen_integers :
  let sorted_list := first_fifteen_integers.toArray.qsort (·≤·)
  sorted_list[(sorted_list.size - 1) / 2]! = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_of_first_fifteen_integers_l927_92716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_trapezoid_area_l927_92723

/-- A trapezoid with mutually perpendicular diagonals -/
structure Trapezoid where
  /-- Length of one diagonal -/
  diagonal : ℝ
  /-- Height of the trapezoid -/
  height : ℝ
  /-- The diagonals are mutually perpendicular -/
  perpendicular_diagonals : True

/-- The area of a trapezoid with mutually perpendicular diagonals -/
noncomputable def trapezoid_area (t : Trapezoid) : ℝ :=
  (t.height * Real.sqrt (t.diagonal ^ 2 - t.height ^ 2)) / 2

/-- Theorem: The area of a trapezoid with mutually perpendicular diagonals,
    where one diagonal is 17 units long and the height is 15 units,
    is equal to 4335/16 square units -/
theorem special_trapezoid_area :
  ∃ (t : Trapezoid), t.diagonal = 17 ∧ t.height = 15 ∧ trapezoid_area t = 4335 / 16 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_trapezoid_area_l927_92723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_sqrt_l927_92729

theorem negation_of_forall_sqrt :
  (¬ ∀ x : ℝ, x > 1 → Real.sqrt x > 1) ↔ (∃ x₀ : ℝ, x₀ > 1 ∧ Real.sqrt x₀ ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_sqrt_l927_92729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l927_92720

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (5 * x^2 - 17 * x - 6)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | x ∈ Set.Iic (-2/5) ∪ Set.Ici 3} = {x : ℝ | ∃ y, f x = y} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l927_92720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l927_92790

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

-- Theorem statement
theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 3/2) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ 0) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 3/2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l927_92790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_parallel_to_a_l927_92709

noncomputable def a : ℝ × ℝ := (3, 4)

noncomputable def unit_vector : ℝ × ℝ := (3/5, 4/5)

theorem unit_vector_parallel_to_a :
  (∃ k : ℝ, unit_vector = (k * a.1, k * a.2)) ∧
  Real.sqrt (unit_vector.1^2 + unit_vector.2^2) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_parallel_to_a_l927_92709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_valid_m_values_valid_m_values_l927_92783

/-- A right triangle in the coordinate plane with legs parallel to x and y axes -/
structure RightTriangle where
  a : ℝ  -- x-coordinate of the vertex
  b : ℝ  -- y-coordinate of the vertex
  c : ℝ  -- half the length of the vertical leg
  d : ℝ  -- half the length of the horizontal leg

/-- The slopes of the medians of a right triangle -/
noncomputable def median_slopes (t : RightTriangle) : ℝ × ℝ :=
  (t.c / (2 * t.d), 2 * t.c / t.d)

/-- Predicate for a valid right triangle with medians on given lines -/
def is_valid_triangle (t : RightTriangle) (m : ℝ) : Prop :=
  let (slope1, slope2) := median_slopes t
  (slope1 = 5 ∧ slope2 = m) ∨ (slope1 = m ∧ slope2 = 5)

/-- The theorem stating that there are exactly two values of m for valid triangles -/
theorem two_valid_m_values :
  ∃ m₁ m₂ : ℝ, m₁ ≠ m₂ ∧
  (∀ m : ℝ, (∃ t : RightTriangle, is_valid_triangle t m) ↔ (m = m₁ ∨ m = m₂)) := by
  sorry

/-- The specific values of m that allow valid triangles -/
theorem valid_m_values :
  ∃ t₁ t₂ : RightTriangle, is_valid_triangle t₁ 5 ∧ is_valid_triangle t₂ (5/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_valid_m_values_valid_m_values_l927_92783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_subsets_count_l927_92745

def A : Finset Nat := {1, 2, 3, 4, 5}
def B : Finset Nat := {1, 3, 5, 7, 9}

theorem intersection_subsets_count :
  Finset.card (Finset.powerset (A ∩ B)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_subsets_count_l927_92745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_is_negative_85_l927_92742

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

def equation (x : ℤ) : Prop :=
  floor (x / 2 : ℝ) + floor (x / 3 : ℝ) + floor (x / 7 : ℝ) = x

theorem smallest_solution_is_negative_85 :
  ∀ x : ℤ, equation x → x ≥ -85 ∧ equation (-85) := by
  sorry

#check smallest_solution_is_negative_85

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_is_negative_85_l927_92742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_has_related_sequence_l927_92781

def original_sequence : List ℕ := [1, 5, 9, 13, 17]

def related_sequence_term (a : List ℕ) (n : ℕ) : ℚ :=
  (a.take n.succ).sum / n - a.get! n / n

def is_related_sequence (a b : List ℕ) : Prop :=
  a.length = b.length ∧
  ∀ n, n < a.length → related_sequence_term a n = b.get! n

theorem original_has_related_sequence :
  is_related_sequence original_sequence [11, 10, 9, 8, 7] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_has_related_sequence_l927_92781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gordonia_population_ratio_l927_92725

/-- The population of the three towns -/
def total_population : ℕ := 80000

/-- The population of Lake Bright -/
def lake_bright_population : ℕ := 16000

/-- The ratio of Toadon's population to Gordonia's population -/
def toadon_gordonia_ratio : ℚ := 3/5

theorem gordonia_population_ratio :
  ∃ (gordonia_population : ℕ),
    gordonia_population + (toadon_gordonia_ratio * gordonia_population).floor + lake_bright_population = total_population ∧
    (gordonia_population : ℚ) / total_population = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gordonia_population_ratio_l927_92725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_one_third_value_l927_92766

-- Define a power function
noncomputable def PowerFunction (α : ℝ) : ℝ → ℝ := fun x ↦ x^α

-- Main theorem
theorem power_function_one_third_value 
  (f : ℝ → ℝ) 
  (h1 : ∃ α : ℝ, f = PowerFunction α) 
  (h2 : f = fun x ↦ 8 * f x) : 
  f (1/3) = 1/27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_one_third_value_l927_92766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_limit_l927_92741

open Real
open BigOperators

/-- The area of rectangles formed by consecutive points on y = 1/x approaches 1 as n approaches infinity --/
theorem rectangle_area_limit : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
  |1 - (∑ i in Finset.range n, (1 / (i : ℝ) - 1 / ((i : ℝ) + 1)) + 1 / (n : ℝ))| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_limit_l927_92741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l927_92789

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- A point on an ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The focal distance of an ellipse -/
noncomputable def focalDistance (e : Ellipse) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2)

/-- The left focus of an ellipse -/
noncomputable def leftFocus (e : Ellipse) : ℝ × ℝ :=
  (-focalDistance e, 0)

/-- The right focus of an ellipse -/
noncomputable def rightFocus (e : Ellipse) : ℝ × ℝ :=
  (focalDistance e, 0)

/-- The dot product of vectors from a point on the ellipse to its foci -/
noncomputable def fociDotProduct (e : Ellipse) (p : PointOnEllipse e) : ℝ :=
  let (x₁, y₁) := leftFocus e
  let (x₂, y₂) := rightFocus e
  (p.x - x₁) * (p.x - x₂) + (p.y - y₁) * (p.y - y₂)

theorem ellipse_eccentricity_range (e : Ellipse) :
  (∀ p : PointOnEllipse e, fociDotProduct e p ≤ e.a * focalDistance e) →
  (Real.sqrt 5 - 1) / 2 ≤ eccentricity e ∧ eccentricity e < 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l927_92789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_t_1_l927_92710

-- Define the position function
noncomputable def s (t : ℝ) : ℝ := 2 * t^2

-- Define the velocity function as the derivative of position
noncomputable def v (t : ℝ) : ℝ := deriv s t

-- Theorem statement
theorem instantaneous_velocity_at_t_1 :
  v 1 = 4 := by
  -- Unfold the definitions of v and s
  unfold v s
  -- Simplify the derivative
  simp [deriv]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_t_1_l927_92710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_quadrant_trig_l927_92757

theorem fourth_quadrant_trig (θ : Real) 
  (h1 : θ ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi))  -- θ is in the fourth quadrant
  (h2 : Real.sin θ = -1/3) :                          -- sin θ = -1/3
  Real.cos θ = 2 * Real.sqrt 2 / 3 ∧                  -- cos θ = 2√2/3
  Real.sin (2 * θ) = -4 * Real.sqrt 2 / 9 :=          -- sin 2θ = -4√2/9
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_quadrant_trig_l927_92757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_2008_is_six_l927_92715

/-- Represents a 2009-digit integer as a list of digits -/
def Digit2009 := List Nat

/-- Checks if a number has exactly 3 distinct prime factors -/
def has_three_distinct_prime_factors (n : Nat) : Prop := sorry

/-- Constructs a 2-digit number from two consecutive digits -/
def two_digit_number (a b : Nat) : Nat := 10 * a + b

/-- Defines the property that each pair of consecutive digits forms a 2-digit number with 3 distinct prime factors -/
def valid_digit_pairs (d : Digit2009) : Prop :=
  ∀ i, i < 2007 → has_three_distinct_prime_factors (two_digit_number (d.get! i) (d.get! (i+1)))

theorem digit_2008_is_six (d : Digit2009) (h : valid_digit_pairs d) : d.get! 2007 = 6 := by
  sorry

#check digit_2008_is_six

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_2008_is_six_l927_92715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l927_92791

theorem power_equality (x y : ℝ) (h1 : (2 : ℝ)^x = 5) (h2 : (4 : ℝ)^y = 3) : 
  (2 : ℝ)^(x + 2*y) = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l927_92791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_and_variance_of_X_l927_92744

-- Define the probability density function
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 ∨ x ≥ 8 then 0
  else if 2 < x ∧ x ≤ 4 then (1/6) * (x - 2)
  else if 4 < x ∧ x ≤ 8 then -(1/12) * (x - 8)
  else 0

-- State the theorem
theorem expected_value_and_variance_of_X :
  let E := ∫ x in Set.Icc 2 8, x * f x
  let D := ∫ x in Set.Icc 2 8, (x - E)^2 * f x
  E = 14/3 ∧ D = 14/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_and_variance_of_X_l927_92744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_step_height_correct_l927_92761

/-- Represents the height of water in the pool at a given time. -/
noncomputable def water_height (t : ℝ) : ℝ :=
  if t ≤ 18 then 2.5 * t else 45 + 2 * (t - 18)

/-- The step height in the pool. -/
def step_height : ℝ := 45

theorem step_height_correct :
  (water_height 8 = 20) ∧
  (water_height 23 = 55) ∧
  (water_height 35.5 = 80) ∧
  (∀ t : ℝ, 8 < t → t < 23 → water_height t > step_height) →
  step_height = 45 := by
  sorry

#check step_height_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_step_height_correct_l927_92761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l927_92776

/-- The time (in seconds) it takes for a train to cross a bridge -/
noncomputable def time_to_cross (train_length : ℝ) (train_speed_kmph : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  total_distance / train_speed_mps

/-- Theorem stating that a train of length 110 m, traveling at 60 kmph, 
    takes approximately 21 seconds to cross a bridge of length 240 m -/
theorem train_bridge_crossing_time :
  ∃ ε > 0, |time_to_cross 110 60 240 - 21| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l927_92776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_perimeter_l927_92772

/-- A point in a 2D plane --/
structure Point where
  x : Int
  y : Int

/-- A parallelogram defined by four points --/
structure Parallelogram where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- The distance between two points --/
noncomputable def distance (p1 p2 : Point) : Real :=
  Real.sqrt (((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2) : Real)

/-- The perimeter of a parallelogram --/
noncomputable def perimeter (p : Parallelogram) : Real :=
  distance p.P p.Q + distance p.Q p.R + distance p.R p.S + distance p.S p.P

/-- Theorem: The perimeter of the given parallelogram on a 6x6 grid is 16 --/
theorem parallelogram_perimeter :
  let pqrs : Parallelogram := {
    P := { x := 3, y := 4 }
    Q := { x := 0, y := 0 }
    R := { x := 3, y := 0 }
    S := { x := 0, y := 4 }
  }
  perimeter pqrs = 16 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_perimeter_l927_92772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_values_of_f_l927_92788

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x - Real.pi / 6)

theorem min_values_of_f (ω : ℝ) (h_ω_pos : ω > 0) (h_period : (2 * Real.pi) / ω = 4 * Real.pi) :
  ∃ (min_val : ℝ), ∀ (x : ℝ), f ω x ≥ min_val ∧
  (f ω x = min_val ↔ ∃ (k : ℤ), x = 4 * k * Real.pi - 2 * Real.pi / 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_values_of_f_l927_92788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_wins_iff_sum_of_odd_powers_of_2_l927_92767

/-- Represents the game state -/
inductive GameState
| player_a
| player_b

/-- Defines a valid move in the game -/
def valid_move (n : ℕ) (N : ℕ) : Prop :=
  n + 1 ≤ N ∨ 2 * n ≤ N

/-- Defines the winning condition -/
def is_winning_move (n : ℕ) (N : ℕ) : Prop :=
  n = N

/-- Defines the sum of distinct odd powers of 2 -/
def is_sum_of_odd_powers_of_2 (N : ℕ) : Prop :=
  ∃ (S : Finset ℕ), (∀ n ∈ S, ∃ k : ℕ, n = 2^(2*k+1)) ∧ (N = S.sum id)

/-- Main theorem: Bob wins if and only if N is the sum of distinct odd powers of 2 -/
theorem bob_wins_iff_sum_of_odd_powers_of_2 (N : ℕ) :
  (∃ (strategy : ℕ → ℕ), 
    (∀ n < N, valid_move n N → valid_move (strategy n) N) ∧
    (∀ n < N, is_winning_move (strategy n) N → GameState.player_b = GameState.player_b) ∧
    (∀ n < N, ¬is_winning_move n N → 
      ∃ m, valid_move m N ∧ ¬is_winning_move m N ∧ strategy m = N))
  ↔ 
  is_sum_of_odd_powers_of_2 N :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_wins_iff_sum_of_odd_powers_of_2_l927_92767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_x_squared_plus_reciprocal_l927_92773

theorem min_value_x_squared_plus_reciprocal (x : ℝ) (h : x ≠ 0) :
  x^2 + (1/x)^2 ≥ 2 ∧ (x^2 + (1/x)^2 = 2 ↔ x = 1 ∨ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_x_squared_plus_reciprocal_l927_92773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_solution_l927_92713

theorem sqrt_equation_solution :
  let f : ℝ → ℝ := λ x => Real.sqrt (5 * x - 4) + 15 / Real.sqrt (5 * x - 4)
  ∃ (s : Set ℝ), s = {29/5, 13/5} ∧ ∀ x ∈ s, f x = 8 ∧ ∀ y : ℝ, f y = 8 → y ∈ s :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_solution_l927_92713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_square_relation_l927_92732

/-- A function representing the inverse square relationship between x and y -/
noncomputable def inverse_square (k : ℝ) (y : ℝ) : ℝ := k / (y^2)

theorem inverse_square_relation (k : ℝ) :
  (inverse_square k 3 = 1) → (inverse_square k 6 = 1/4) := by
  intro h
  -- Proof steps would go here
  sorry

#check inverse_square_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_square_relation_l927_92732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_from_triangle_l927_92792

/-- Triangle with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Rectangle with width w and length l -/
structure Rectangle where
  w : ℝ
  l : ℝ

/-- The area of a triangle -/
noncomputable def Triangle.area (t : Triangle) : ℝ :=
  (t.a * t.b) / 2

/-- The area of a rectangle -/
noncomputable def Rectangle.area (r : Rectangle) : ℝ :=
  r.w * r.l

/-- The perimeter of a rectangle -/
noncomputable def Rectangle.perimeter (r : Rectangle) : ℝ :=
  2 * (r.w + r.l)

theorem rectangle_perimeter_from_triangle (t : Triangle) (r : Rectangle) :
  t.a = 9 ∧ t.b = 12 ∧ t.c = 15 ∧ r.w = 6 ∧ Triangle.area t = Rectangle.area r →
  Rectangle.perimeter r = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_from_triangle_l927_92792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_local_values_2345_l927_92712

/-- The sum of local values of digits in 2345 equals 2345 -/
theorem sum_local_values_2345 :
  2000 + 300 + 40 + 5 = 2345 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_local_values_2345_l927_92712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_system_solution_l927_92721

open Real

theorem trigonometric_system_solution :
  let S : Set (ℝ × ℝ) := {(x, y) | Real.cos x + Real.cos y = Real.cos (x + y) ∧ Real.sin x + Real.sin y = Real.sin (x + y)}
  S = {(x, y) | ∃ m n : ℤ, (x = π/3 + 2*π*↑m ∧ y = -π/3 + 2*π*↑n) ∨ (x = -π/3 + 2*π*↑m ∧ y = π/3 + 2*π*↑n)} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_system_solution_l927_92721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_increasing_condition_minimum_value_existence_minimum_value_a_l927_92794

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + 1/2) + 2 / (2 * x + 1)

-- Theorem 1: Monotonically increasing condition
theorem monotonically_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Ioi 0, Monotone (f a)) → a ≥ 2 :=
sorry

-- Theorem 2: Existence of minimum value and its corresponding a
theorem minimum_value_existence :
  ∃ a : ℝ, ∀ x ∈ Set.Ioi 0, f a x ≥ 1 ∧ (∃ y ∈ Set.Ioi 0, f a y = 1) :=
sorry

-- Theorem 3: Value of a for minimum
theorem minimum_value_a :
  ∃! a : ℝ, (∀ x ∈ Set.Ioi 0, f a x ≥ 1) ∧ (∃ y ∈ Set.Ioi 0, f a y = 1) ∧ a = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_increasing_condition_minimum_value_existence_minimum_value_a_l927_92794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_survey_l927_92719

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / (x - 4) + 10 * (x - 7)^2

-- Define the profit function h(x)
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := (x - 4) * f a x

-- State the theorem
theorem sales_survey (a : ℝ) :
  (∀ x, 4 < x → x < 7 → f a x = f a x) →  -- Domain condition
  f a 6 = 15 →                           -- Given condition
  a = 10 ∧                               -- Part 1: Value of a
  ∃ x, 4 < x ∧ x < 7 ∧                   -- Part 2: Profit maximization
    x = 5 ∧
    ∀ y, 4 < y → y < 7 → h a y ≤ h a x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_survey_l927_92719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harolds_wrapping_problem_l927_92738

def wrapping_problem (shirt_boxes_per_roll : ℕ) (xl_boxes_per_roll : ℕ) 
  (shirt_boxes_to_wrap : ℕ) (cost_per_roll : ℚ) (total_cost : ℚ) : Prop :=
  let total_rolls : ℕ := (total_cost / cost_per_roll).floor.toNat
  let rolls_for_shirt_boxes : ℕ := (shirt_boxes_to_wrap + shirt_boxes_per_roll - 1) / shirt_boxes_per_roll
  let remaining_rolls : ℕ := total_rolls - rolls_for_shirt_boxes
  let xl_boxes_to_wrap : ℕ := remaining_rolls * xl_boxes_per_roll
  xl_boxes_to_wrap = 12

theorem harolds_wrapping_problem :
  wrapping_problem 5 3 20 4 32 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harolds_wrapping_problem_l927_92738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_functions_theorem_l927_92708

/-- The function f(x) = 2^x -/
noncomputable def f (x : ℝ) : ℝ := 2^x

/-- The function h(x) symmetric to f(x) with respect to (log₂3, 3a/2) -/
noncomputable def h (a x : ℝ) : ℝ := 3*a - 9/(2^x)

/-- The statement to be proved -/
theorem symmetric_functions_theorem (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = h a x₁ ∧ f x₂ = h a x₂ ∧ |x₁ - x₂| = 2) →
  (∀ x : ℝ, h a x = 3*a - 9/(2^x)) ∧ a = 5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_functions_theorem_l927_92708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contest_schools_count_l927_92730

/-- Represents a school in the math contest -/
structure School where
  students : Fin 3 → ℕ

/-- Represents the math contest -/
structure MathContest where
  schools : List School
  andrea_rank : ℕ
  beth_rank : ℕ
  carla_rank : ℕ

/-- Properties of the math contest -/
def ValidContest (contest : MathContest) : Prop :=
  let total_participants := contest.schools.length * 3
  70 < total_participants
  ∧ total_participants < 120
  ∧ contest.andrea_rank = (total_participants + 1) / 2  -- median
  ∧ contest.andrea_rank < contest.beth_rank
  ∧ contest.andrea_rank < contest.carla_rank
  ∧ contest.beth_rank = 42
  ∧ contest.carla_rank = 57

/-- Theorem stating that a valid contest has 33 schools -/
theorem contest_schools_count (contest : MathContest) 
  (h : ValidContest contest) : contest.schools.length = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contest_schools_count_l927_92730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_product_60_prime_factors_l927_92711

def divisors_product (n : ℕ) : ℕ := (List.filter (λ d ↦ n % d = 0) (List.range (n + 1))).prod

def distinct_prime_factors (n : ℕ) : ℕ := (Nat.factors n).eraseDups.length

theorem divisors_product_60_prime_factors :
  distinct_prime_factors (divisors_product 60) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_product_60_prime_factors_l927_92711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_interval_l927_92795

-- Define d as a real number
variable (d : ℝ)

-- Dave's incorrect statement
axiom dave_incorrect : ¬(d ≥ 10)

-- Elena's incorrect statement
axiom elena_incorrect : ¬(d ≤ 9)

-- Fiona's incorrect statement
axiom fiona_incorrect : d ≠ 8

-- Theorem to prove
theorem distance_interval : d ∈ Set.Ioo 9 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_interval_l927_92795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_condition_l927_92717

theorem sufficient_condition (P M N : ℝ) :
  (P > N → M > N) → (P > N → M > N) :=
by
  intro h
  exact h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_condition_l927_92717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l927_92762

/-- A polynomial in x and y with parameter m -/
def polynomial (m : ℤ) (x y : ℤ) : ℤ := x^2 + 4*x*y - x + 2*m*y + m

/-- Predicate to check if a polynomial can be factored into two linear factors with integer coefficients -/
def can_factor_into_linear (p : ℤ → ℤ → ℤ) : Prop :=
  ∃ (a b c d e f : ℤ), ∀ x y, p x y = (a*x + b*y + c) * (d*x + e*y + f)

theorem polynomial_factorization (m : ℤ) :
  can_factor_into_linear (polynomial m) ↔ m = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l927_92762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_folding_properties_l927_92703

/-- Represents the dimensions of a rectangular paper in decimeters -/
structure PaperDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the number of different shapes obtained by folding a paper n times -/
def num_shapes (n : ℕ) : ℕ := n + 1

/-- Calculates the sum of areas of shapes obtained by folding a paper n times -/
noncomputable def sum_areas (n : ℕ) : ℝ := 240 * (3 - (n + 3) / (2^n))

/-- Theorem stating the properties of paper folding -/
theorem paper_folding_properties (paper : PaperDimensions) 
  (h1 : paper.length = 20)
  (h2 : paper.width = 12) :
  (num_shapes 4 = 5) ∧ 
  (∀ n : ℕ, sum_areas n = 240 * (3 - (n + 3) / (2^n))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_folding_properties_l927_92703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_fg_one_root_condition_inequality_condition_l927_92769

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1
def g (x : ℝ) : ℝ := Real.exp x

theorem max_value_fg (a : ℝ) (h : a = -1) :
  ∃ M, M = (3 : ℝ) * Real.exp 2 ∧ 
  ∀ x, x ∈ Set.Icc (-1 : ℝ) 2 → f a x * g x ≤ M :=
sorry

theorem one_root_condition (a : ℝ) (h : a = -1) :
  ∀ k : ℝ, (∃! x, f a x = k * g x) ↔ 
  (k > 3 / Real.exp 2 ∨ (0 < k ∧ k < 1 / Real.exp 1)) :=
sorry

theorem inequality_condition :
  ∀ a : ℝ, (∀ x₁ x₂, x₁ ∈ Set.Icc 0 2 → x₂ ∈ Set.Icc 0 2 → x₁ ≠ x₂ → 
    |f a x₁ - f a x₂| < |g x₁ - g x₂|) ↔ 
  (-1 ≤ a ∧ a ≤ 2 - 2 * Real.log 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_fg_one_root_condition_inequality_condition_l927_92769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_6_to_7_l927_92765

-- Define a normally distributed random variable X
noncomputable def X : Real → ℝ := sorry

-- Define the probability measure
noncomputable def P : Set ℝ → ℝ := sorry

-- Given conditions
def mean : ℝ := 5
def std_dev : ℝ := 1

-- We'll use a simple axiom for normal distribution instead of NormalDistribution
axiom normal_dist : ∀ (a b : ℝ), a < b → P {x | a < X x ∧ X x < b} = sorry

-- Reference values
axiom ref_value_1σ : P {x | mean - std_dev < X x ∧ X x ≤ mean + std_dev} = 0.6826
axiom ref_value_2σ : P {x | mean - 2 * std_dev < X x ∧ X x ≤ mean + 2 * std_dev} = 0.9544
axiom ref_value_3σ : P {x | mean - 3 * std_dev < X x ∧ X x ≤ mean + 3 * std_dev} = 0.9974

-- Theorem to prove
theorem prob_6_to_7 : P {x | 6 < X x ∧ X x < 7} = 0.1359 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_6_to_7_l927_92765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_calculation_l927_92786

-- Define the ⊗ operation
noncomputable def otimes (a b c : ℝ) : ℝ := a / (b - c)

-- Theorem statement
theorem otimes_calculation :
  otimes (otimes (otimes 2 5 3) 4 (otimes 5 1 2))
         (otimes (otimes 3 7 2) 8 (otimes 4 9 5))
         (otimes (otimes 1 6 5) (otimes 6 3 7) 2) = 5/17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_calculation_l927_92786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_is_systematic_l927_92726

/-- Represents a sampling method -/
inductive SamplingMethod
| StratifiedSampling
| LotteryMethod
| RandomNumberTableMethod
| SystematicSampling

/-- Represents a class of students -/
structure StudentClass where
  students : Finset ℕ
  size : students.card = 50
  numbering : students = Finset.range 50

/-- Represents a grade with multiple classes -/
structure Grade where
  classes : Finset StudentClass
  size : classes.card = 12

/-- Represents a sampling strategy -/
def sampling_strategy (g : Grade) : Finset ℕ :=
  g.classes.image (λ _ => 40)

/-- Theorem stating that the described sampling method is Systematic Sampling -/
theorem sampling_is_systematic (g : Grade) : 
  sampling_strategy g = g.classes.image (λ _ => 40) → 
  SamplingMethod.SystematicSampling = SamplingMethod.SystematicSampling := by
  intro h
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_is_systematic_l927_92726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_decomposition_theorem_l927_92728

theorem integer_decomposition_theorem :
  ∃ (N : ℕ → ℕ), ∀ (k : ℕ), k ≥ 2 →
    ∀ (n : ℕ), n ≥ N k →
      ∃ (a : ℕ → ℕ), 
        (n = Finset.sum (Finset.range k) a) ∧
        (∀ i ∈ Finset.range (k - 1), a i < a (i + 1)) ∧
        (∀ i ∈ Finset.range (k - 1), a i ∣ a (i + 1)) ∧
        (a 0 ≥ 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_decomposition_theorem_l927_92728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_is_zero_l927_92771

-- Define the circle centers and their projections on line l
structure CircleConfig where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  P' : ℝ × ℝ
  Q' : ℝ × ℝ
  R' : ℝ × ℝ

-- Define the properties of the configuration
def validConfig (c : CircleConfig) : Prop :=
  let (px, py) := c.P
  let (qx, qy) := c.Q
  let (rx, ry) := c.R
  let (p'x, p'y) := c.P'
  let (q'x, q'y) := c.Q'
  let (r'x, r'y) := c.R'
  -- Circle radii
  ((px - p'x)^2 + (py - p'y)^2 = 9) ∧
  ((qx - q'x)^2 + (qy - q'y)^2 = 4) ∧
  ((rx - r'x)^2 + (ry - r'y)^2 = 16) ∧
  -- Q' between P' and R'
  (p'x ≤ q'x ∧ q'x ≤ r'x) ∧
  -- External tangency
  ((px - qx)^2 + (py - qy)^2 = 25) ∧
  ((qx - rx)^2 + (qy - ry)^2 = 36)

-- Theorem statement
theorem area_of_triangle_PQR_is_zero (c : CircleConfig) (h : validConfig c) :
  let (px, py) := c.P
  let (qx, qy) := c.Q
  let (rx, ry) := c.R
  abs ((px * (qy - ry) + qx * (ry - py) + rx * (py - qy)) / 2) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_is_zero_l927_92771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_and_inequality_l927_92735

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log (1 / x) else 1 / x

theorem f_composition_and_inequality :
  (f (f (Real.exp 1)) = -1) ∧
  (∀ x : ℝ, f x > -1 ↔ x ∈ Set.Ioi (-1) ∪ Set.Ioo 0 (Real.exp 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_and_inequality_l927_92735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_vertex_l927_92700

/-- The quadratic function f(x) = 1/2 * (x - 4)^2 + 5 has vertex (4, 5) -/
theorem quadratic_vertex :
  let f : ℝ → ℝ := λ x ↦ (1/2) * (x - 4)^2 + 5
  (4, 5) = (4, f 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_vertex_l927_92700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_25_l927_92702

/-- The acute angle between clock hands at a given time -/
noncomputable def clockAngle (hours : ℕ) (minutes : ℕ) : ℝ :=
  let hourAngle : ℝ := (hours % 12 + minutes / 60 : ℝ) * 30
  let minuteAngle : ℝ := minutes * 6
  min (abs (hourAngle - minuteAngle)) (360 - abs (hourAngle - minuteAngle))

/-- Theorem: The acute angle between the minute and hour hands on a standard clock at 3:25 is 47.5° -/
theorem clock_angle_at_3_25 : clockAngle 3 25 = 47.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_25_l927_92702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_and_modular_arithmetic_modular_arithmetic_divisibility_l927_92777

theorem polynomial_divisibility_and_modular_arithmetic (n : ℕ) :
  (∀ x : ℤ, (x^2 + x + 1) ∣ (x^(2*n) + x^n + 1)) ↔ n % 3 = 0 :=
by sorry

theorem modular_arithmetic_divisibility (n : ℕ) :
  37 ∣ (10^(2*n + 1) + 10^(n + 1) + 1) ↔ n % 3 = 0 ∨ n % 3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_and_modular_arithmetic_modular_arithmetic_divisibility_l927_92777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wedding_gift_cost_l927_92770

def steak_knife_set_price : ℚ := 80
def steak_knife_sets : ℚ := 2
def dinnerware_set_price : ℚ := 200
def fancy_napkins_price : ℚ := 45
def wine_glasses_price : ℚ := 100
def discount_steak_dinnerware : ℚ := 1/10
def discount_napkins : ℚ := 1/5
def sales_tax : ℚ := 1/20

def total_before_discounts : ℚ := steak_knife_set_price * steak_knife_sets + dinnerware_set_price + fancy_napkins_price + wine_glasses_price

def discount_amount : ℚ := (steak_knife_set_price * steak_knife_sets + dinnerware_set_price) * discount_steak_dinnerware + fancy_napkins_price * discount_napkins

def total_after_discounts : ℚ := total_before_discounts - discount_amount

theorem wedding_gift_cost (total_cost : ℚ) (savings : ℚ) : 
  total_cost = total_after_discounts * (1 + sales_tax) ∧ 
  savings = discount_amount ∧
  total_cost = 55860/100 ∧
  savings = 63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wedding_gift_cost_l927_92770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_bead_bracelet_arrangements_l927_92705

/-- The number of distinct beads on the bracelet -/
def n : ℕ := 8

/-- The number of ways to arrange n distinct beads on a bracelet,
    considering rotations and reflections as the same arrangement,
    where two specific beads must be adjacent -/
def bracelet_arrangements (n : ℕ) : ℕ :=
  (Nat.factorial (n - 1)) / (2 * n)

/-- Theorem stating that the number of arrangements for 8 beads
    with the given constraints is 315 -/
theorem eight_bead_bracelet_arrangements :
  bracelet_arrangements n = 315 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_bead_bracelet_arrangements_l927_92705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tangent_lines_l927_92759

/-- A line in 2D space --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- The parabola y^2 = x --/
def Parabola : Set Point :=
  {p : Point | p.y^2 = p.x}

/-- The given point P --/
def P : Point :=
  ⟨-1, 0⟩

/-- A line passes through a point --/
def passes_through (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- A line has exactly one point in common with the parabola --/
def has_one_common_point (l : Line) : Prop :=
  ∃! p : Point, p ∈ Parabola ∧ passes_through l p

/-- The set of lines passing through P and having exactly one point in common with the parabola --/
def TangentLines : Set Line :=
  {l : Line | passes_through l P ∧ has_one_common_point l}

/-- The main theorem --/
theorem three_tangent_lines : ∃ (S : Finset Line), S.card = 3 ∧ ∀ l : Line, l ∈ S ↔ l ∈ TangentLines :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tangent_lines_l927_92759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_amount_l927_92797

/-- Simple interest calculation -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  (principal * rate * time) / 100

/-- Compound interest calculation -/
noncomputable def compound_interest (principal rate time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

theorem compound_interest_amount (simple_principal simple_rate simple_time
                                  compound_principal compound_rate compound_time : ℝ) :
  simple_principal = 1750 →
  simple_rate = 8 →
  simple_time = 3 →
  compound_rate = 10 →
  compound_time = 2 →
  simple_interest simple_principal simple_rate simple_time = 
    (1/2) * compound_interest compound_principal compound_rate compound_time →
  compound_principal = 4000 := by
  sorry

#check compound_interest_amount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_amount_l927_92797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_term_coefficient_l927_92722

/-- The coefficient of the linear term in the polynomial x^2 - 2x - 3 is -2 -/
theorem linear_term_coefficient : 
  let p : ℝ → ℝ := λ x => x^2 - 2*x - 3
  (deriv p 0 - deriv (deriv p) 0 * 0) / 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_term_coefficient_l927_92722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounded_by_parabola_and_line_l927_92739

-- Define the parabola
def C (x : ℝ) : ℝ := x^2 - x - 2

-- Define the points P and Q on the parabola
def P : ℝ × ℝ := (-2, C (-2))
def Q : ℝ × ℝ := (1, C 1)

-- Define the line PQ
def line_PQ (x : ℝ) : ℝ := -2 * x

-- Theorem for part (c)
theorem area_bounded_by_parabola_and_line :
  ∫ x in Set.Icc (-2) 1, C x - line_PQ x = 9/2 := by sorry

-- Additional conditions
axiom x_P_less_than_x_Q : P.1 < Q.1

-- Condition that O divides PQ internally in ratio 2:1
axiom O_divides_PQ : (0, 0) = ((2 * Q.1 + P.1) / 3, (2 * Q.2 + P.2) / 3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounded_by_parabola_and_line_l927_92739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_on_domain_f_symmetric_about_origin_l927_92727

-- Define the function f(x) = log((2/(x+1)) - 1)
noncomputable def f (x : ℝ) : ℝ := Real.log ((2 / (x + 1)) - 1)

-- Define the domain of the function
def domain : Set ℝ := { x | -1 < x ∧ x < 1 }

-- Theorem statement
theorem f_is_odd_on_domain : 
  ∀ x ∈ domain, f (-x) = -f x :=
by
  sorry

-- Theorem stating that the graph of f is symmetric with respect to the origin
theorem f_symmetric_about_origin :
  ∀ x ∈ domain, ∃ y, f x = y ∧ f (-x) = -y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_on_domain_f_symmetric_about_origin_l927_92727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corrected_mean_calculation_l927_92751

/-- Calculates the corrected mean of a set of observations after adjusting for errors -/
theorem corrected_mean_calculation (n : ℕ) (original_mean : ℚ) 
  (incorrect_value1 incorrect_value2 correct_value1 correct_value2 : ℚ) :
  n = 50 ∧ 
  original_mean = 36 ∧
  incorrect_value1 = 23 ∧
  incorrect_value2 = 55 ∧
  correct_value1 = 34 ∧
  correct_value2 = 45 →
  (n * original_mean + (correct_value1 - incorrect_value1) + (correct_value2 - incorrect_value2)) / n = 36.02 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_corrected_mean_calculation_l927_92751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solutions_l927_92740

theorem problem_solutions :
  ((-1 : ℝ)^3 + |1 - Real.sqrt 2| + 8^(1/3 : ℝ) = Real.sqrt 2) ∧
  ((-2 : ℝ)^3 + Real.sqrt ((-3)^2) + 3 * (1/27)^(1/3 : ℝ) + |Real.sqrt 3 - 4| = -Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solutions_l927_92740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machines_joint_time_l927_92736

/-- The time taken for two machines to complete a job together, given their individual completion times -/
noncomputable def joint_completion_time (time_a : ℝ) (time_b : ℝ) : ℝ :=
  1 / (1 / time_a + 1 / time_b)

/-- Theorem: Two machines with individual completion times of 4 and 12 hours will complete the job in 3 hours when working together -/
theorem machines_joint_time :
  joint_completion_time 4 12 = 3 := by
  -- Unfold the definition of joint_completion_time
  unfold joint_completion_time
  -- Simplify the expression
  simp
  -- The proof is completed using 'sorry'
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_machines_joint_time_l927_92736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_values_in_fourth_quadrant_l927_92787

theorem trig_values_in_fourth_quadrant 
  (x : Real) 
  (h1 : Real.tan x = -8/15) 
  (h2 : 3*Real.pi/2 < x ∧ x < 2*Real.pi) : 
  Real.sin x = -8/17 ∧ 
  Real.cos x = 15/17 ∧ 
  1 / Real.tan x = -15/8 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_values_in_fourth_quadrant_l927_92787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_portion_proof_l927_92704

variable (P : ℕ) -- number of people
variable (W : ℝ) -- total work

-- The original group can complete the work in 16 days
noncomputable def original_rate (P : ℕ) (W : ℝ) : ℝ := W / 16

-- The doubled group works for 4 days
def doubled_time : ℝ := 4

-- The portion of work completed by the doubled group in 4 days
noncomputable def portion_completed (P : ℕ) (W : ℝ) : ℝ := 
  2 * (original_rate P W) * doubled_time

theorem work_portion_proof (P : ℕ) (W : ℝ) :
  portion_completed P W = W / 2 := by
  sorry

#check work_portion_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_portion_proof_l927_92704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l927_92724

def mySequence (n : ℕ) : ℕ := n

def mySequenceSum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sequence_general_term (n : ℕ) (h : n > 0) : 
  mySequence n = n ∧ n * mySequence (n + 1) = 2 * mySequenceSum n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l927_92724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_container_fill_times_l927_92768

noncomputable section

-- Define the reference container and rain rate
def reference_volume : ℝ := 10 * 10 * 30
def rain_rate : ℝ := reference_volume / 1

-- Define the containers
def container1_volume : ℝ := 10 * 10 * 30
def container2_volume : ℝ := 10 * 10 * 20 + 10 * 10 * 10
def container3_volume : ℝ := Real.pi * 1 * 1 * 20

-- Define the time to fill each container
def time_container1 : ℝ := container1_volume / rain_rate
def time_container2 : ℝ := container2_volume / rain_rate
def time_container3 : ℝ := container3_volume / (Real.pi * 1 * 1 * rain_rate)

end noncomputable section

-- Theorem statement
theorem container_fill_times :
  time_container1 = 3 ∧ time_container2 = 1.5 ∧ time_container3 = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_container_fill_times_l927_92768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_inequality_positive_numbers_l927_92748

-- Problem 1
theorem sqrt_inequality : Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 := by sorry

-- Problem 2
theorem positive_numbers (a b c : ℝ) 
  (sum_pos : a + b + c > 0) 
  (sum_products_pos : a * b + b * c + c * a > 0) 
  (product_pos : a * b * c > 0) : 
  a > 0 ∧ b > 0 ∧ c > 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_inequality_positive_numbers_l927_92748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l927_92755

noncomputable def projection (v : ℝ × ℝ) (w : ℝ × ℝ) : ℝ × ℝ :=
  let dot := v.1 * w.1 + v.2 * w.2
  let norm_squared := w.1 * w.1 + w.2 * w.2
  (dot / norm_squared * w.1, dot / norm_squared * w.2)

theorem projection_problem (P : (ℝ × ℝ) → (ℝ × ℝ)) :
  P (6, 2) = (36/5, 12/5) →
  P (2, -4) = (3/5, 1/5) :=
by
  intro h
  sorry

#check projection_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l927_92755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_analysis_l927_92760

-- Define propositions p and q
def p : Prop := ∃ x : ℝ, Real.sin x = Real.sqrt 5 / 2
def q : Prop := ∀ x : ℝ, x^2 + x + 1 > 0

-- State the theorem
theorem proposition_analysis :
  (¬p) ∧ q ∧ ¬(p ∧ q) ∧ ¬(p ∧ ¬q) ∧ (¬p ∨ q) ∧ ¬(¬p ∨ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_analysis_l927_92760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_expression_l927_92749

-- Define the function (noncomputable due to use of Real.sqrt)
noncomputable def f (x : ℝ) : ℝ := 
  Real.sqrt (x + 2 * Real.sqrt (x - 1)) + Real.sqrt (x - 2 * Real.sqrt (x - 1))

-- State the theorem
theorem simplify_sqrt_expression (x : ℝ) (h : x ≥ 1) :
  f x = if x < 2 then 2 else 2 * Real.sqrt (x - 1) := by
  sorry

#check simplify_sqrt_expression

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_expression_l927_92749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_g_not_increasing_l927_92763

noncomputable def f (x : ℝ) : ℝ := (2017^x - 1) / (2017^x + 1)

def g (x : ℝ) : ℝ := 1 - x^2

def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def isMonotonicallyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

theorem f_is_odd_and_g_not_increasing :
  isOdd f ∧ ¬ isMonotonicallyIncreasing g 0 Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_g_not_increasing_l927_92763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oranges_saved_is_correct_l927_92775

/-- The number of oranges saved when making orange juice with a decreasing requirement -/
noncomputable def oranges_saved : ℝ :=
  let initial_ratio : ℝ := 30 / 50  -- oranges per liter
  let initial_oranges : ℝ := initial_ratio * 20  -- oranges for 20 liters
  let first_half : ℝ := initial_oranges / 2  -- oranges for first 10 liters
  let second_half : ℝ := first_half * 0.9  -- oranges for second 10 liters (10% decrease)
  let actual_oranges : ℝ := first_half + second_half
  initial_oranges - actual_oranges

/-- Theorem stating that the number of oranges saved is 0.6 -/
theorem oranges_saved_is_correct : oranges_saved = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oranges_saved_is_correct_l927_92775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_zero_l927_92784

-- Define the vertices of the triangle
def u : Fin 3 → ℝ := ![4, 3, 2]
def v : Fin 3 → ℝ := ![2, 1, 0]
def w : Fin 3 → ℝ := ![10, 7, 4]

-- Define the function to calculate the area of a triangle given three points
noncomputable def triangleArea (p q r : Fin 3 → ℝ) : ℝ :=
  let a := q - p
  let b := r - p
  (1/2) * (Real.sqrt ((a 1 * b 2 - a 2 * b 1)^2 + 
                      (a 2 * b 0 - a 0 * b 2)^2 + 
                      (a 0 * b 1 - a 1 * b 0)^2))

-- Theorem statement
theorem triangle_area_zero : triangleArea u v w = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_zero_l927_92784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solution_range_l927_92750

theorem inequality_system_solution_range (a : ℝ) : 
  (∃! (s : Finset ℤ), s.card = 3 ∧ 
    (∀ x ∈ s, (x : ℝ) / 2 - 1 < (2 - 3 * (x : ℝ)) / 3 ∧ 
               a - 3 < 4 * (x : ℝ) - 2)) →
  -7 ≤ a ∧ a < -3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solution_range_l927_92750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_loci_l927_92753

/-- Right triangle ABC with hypotenuse AB, where A(-1,0) and B(3,0) -/
structure RightTriangle where
  A : ℝ × ℝ := (-1, 0)
  B : ℝ × ℝ := (3, 0)
  C : ℝ × ℝ
  h_right_angle : (C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 0

/-- The locus of the right angle vertex C -/
def locus_C (t : RightTriangle) : Prop :=
  (t.C.1 - 1)^2 + t.C.2^2 = 4 ∧ t.C.1 ≠ 3 ∧ t.C.1 ≠ -1

/-- The midpoint M of BC -/
noncomputable def midpoint_M (t : RightTriangle) : ℝ × ℝ :=
  ((t.B.1 + t.C.1) / 2, (t.B.2 + t.C.2) / 2)

/-- The locus of the midpoint M of BC -/
def locus_M (t : RightTriangle) : Prop :=
  let M := midpoint_M t
  (M.1 - 2)^2 + M.2^2 = 1 ∧ M.1 ≠ 3 ∧ M.1 ≠ 1

theorem right_triangle_loci (t : RightTriangle) :
  locus_C t ∧ locus_M t := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_loci_l927_92753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_squared_plus_sin_x_l927_92796

theorem integral_x_squared_plus_sin_x : ∫ (x : ℝ) in (-1)..(1), (x^2 + Real.sin x) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_squared_plus_sin_x_l927_92796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_minus_c_value_l927_92746

theorem a_minus_c_value (a b c : ℕ) 
  (h1 : a > b)
  (h2 : a > 0)
  (h3 : b > 0)
  (h4 : c > 0)
  (h5 : a^2 - a * c + b * c = 7) : 
  a - c = 0 ∨ a - c = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_minus_c_value_l927_92746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l927_92780

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3^(-x) - 3^x - x

-- State the theorem
theorem range_of_a (a : ℝ) :
  f (2*a + 3) + f (3 - a) > 0 → a < -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l927_92780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_after_trisection_planes_l927_92737

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ
  vertex : Point3D

noncomputable def trisectionPoint (a b : Point3D) (n : ℕ) : Point3D :=
  { x := a.x + n * (b.x - a.x) / 3
  , y := a.y + n * (b.y - a.y) / 3
  , z := a.z + n * (b.z - a.z) / 3 }

noncomputable def volumeAfterPlanes (c : Cube) (p1 p2 : Plane) : ℝ :=
  sorry -- Definition of volume calculation after plane cuts

theorem cube_volume_after_trisection_planes (c : Cube) 
  (h1 : c.sideLength = 6)
  (a e : Point3D)
  (h2 : a.x = c.vertex.x ∧ a.y = c.vertex.y ∧ a.z = c.vertex.z)
  (h3 : e.x = a.x + c.sideLength ∧ e.y = a.y ∧ e.z = a.z)
  (k l : Point3D)
  (h4 : k = trisectionPoint a e 1)
  (h5 : l = trisectionPoint a e 2)
  (p1 p2 : Plane)
  (h6 : p1 = Plane.mk l.x l.y l.z 0) -- Plane LHG
  (h7 : p2 = Plane.mk k.x k.y k.z 0) -- Plane KFG
  : volumeAfterPlanes c p1 p2 = 174 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_after_trisection_planes_l927_92737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_range_l927_92754

/-- The function f(x) = kx --/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x

/-- The function g(x) = ln(x) / x --/
noncomputable def g (x : ℝ) : ℝ := Real.log x / x

/-- The interval [1/e, e] --/
def interval : Set ℝ := Set.Icc (1 / Real.exp 1) (Real.exp 1)

/-- The theorem statement --/
theorem two_solutions_range (k : ℝ) :
  (∃ x y, x ∈ interval ∧ y ∈ interval ∧ x ≠ y ∧ f k x = g x ∧ f k y = g y) →
  k ∈ Set.Icc (1 / (Real.exp 1)^2) (1 / (2 * Real.exp 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_range_l927_92754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_of_products_l927_92714

def numbers : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]

def is_consecutive (a b : Nat) : Bool :=
  (a + 1 = b) ∨ (b + 1 = a)

def satisfies_consecutive_condition (group1 group2 : List Nat) : Prop :=
  ∀ (row : Nat), row < group1.length → 
    ∃ (col : Nat), col < group2.length ∧ is_consecutive (group1.get! row) (group2.get! col)

def sum_of_products (group1 group2 : List Nat) : Nat :=
  (group1.sum) * (group2.sum)

theorem largest_sum_of_products : 
  ∀ (group1 group2 : List Nat),
    group1.length = 4 ∧ 
    group2.length = 4 ∧ 
    (∀ x, x ∈ group1 → x ∈ numbers) ∧ 
    (∀ x, x ∈ group2 → x ∈ numbers) ∧
    satisfies_consecutive_condition group1 group2 →
    sum_of_products group1 group2 ≤ 1440 := by
  sorry

#check largest_sum_of_products

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_of_products_l927_92714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowed_amount_proof_l927_92778

/-- Calculates the principal amount given the total amount, interest rate, and time period. -/
noncomputable def calculate_principal (total_amount : ℝ) (interest_rate : ℝ) (time : ℝ) : ℝ :=
  total_amount / (1 + (interest_rate * time) / 100)

/-- Theorem stating that given the specified conditions, the borrowed amount is approximately 5396.10 -/
theorem borrowed_amount_proof :
  let total_amount : ℝ := 8310
  let interest_rate : ℝ := 6
  let time : ℝ := 9
  let principal := calculate_principal total_amount interest_rate time
  ∃ ε > 0, |principal - 5396.10| < ε :=
by
  sorry

/-- Compute an approximation of the principal amount -/
def approximate_principal : Float :=
  (8310 : Float) / (1 + (6 * 9) / 100)

#eval approximate_principal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowed_amount_proof_l927_92778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_at_sqrt_2_l927_92793

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (2 * x - x^2) * Real.exp x

-- State the theorem
theorem f_max_at_sqrt_2 :
  ∃ (c : ℝ), c = Real.sqrt 2 ∧
  (∀ x : ℝ, f x ≤ f c) ∧
  (∀ ε > 0, ∃ x : ℝ, |x - c| < ε ∧ f x < f c) :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_at_sqrt_2_l927_92793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_MN_l927_92752

/-- Circle C: x^2 + y^2 - 2x = 0 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

/-- Line l: y = x + 2 -/
def line_l (x y : ℝ) : Prop := y = x + 2

/-- Point M is on line l -/
def point_M (x y : ℝ) : Prop := line_l x y

/-- Point N is on circle C -/
def point_N (x y : ℝ) : Prop := circle_C x y

/-- Distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Tangent line from M to C touches at N -/
def is_tangent (xm ym xn yn : ℝ) : Prop :=
  point_M xm ym ∧ point_N xn yn ∧
  ∃ k, (yn - ym) = k * (xn - xm) ∧
       2 * (xn - 1) * (xm - xn) + 2 * yn * (ym - yn) = 0

/-- The minimum distance between M and N is √14/2 -/
theorem min_distance_MN :
  ∃ xm ym xn yn : ℝ,
    is_tangent xm ym xn yn ∧
    (∀ xm' ym' xn' yn' : ℝ, is_tangent xm' ym' xn' yn' →
      distance xm ym xn yn ≤ distance xm' ym' xn' yn') ∧
    distance xm ym xn yn = Real.sqrt 14 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_MN_l927_92752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mindys_tax_rate_l927_92758

theorem mindys_tax_rate (morks_income : ℝ) (morks_tax_rate : ℝ) (mindys_income_multiplier : ℝ) (combined_tax_rate : ℝ) :
  morks_tax_rate = 0.10 →
  mindys_income_multiplier = 3 →
  combined_tax_rate = 0.175 →
  let mindys_income := mindys_income_multiplier * morks_income
  let total_income := morks_income + mindys_income
  let mindys_tax_rate := (combined_tax_rate * total_income - morks_tax_rate * morks_income) / mindys_income
  mindys_tax_rate = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mindys_tax_rate_l927_92758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_circle_area_is_correct_l927_92779

/-- Given an equilateral triangle with area Y, MaxCircleArea Y is the maximum area Z
    such that three non-overlapping circles of radius √Z can fit inside the triangle. -/
noncomputable def MaxCircleArea (Y : ℝ) : ℝ :=
  (2 * Real.sqrt 3 - 3) * Y / 6

/-- Predicate stating that three circles of given radii can fit
    inside an equilateral triangle of area Y without overlapping. -/
def three_circles_fit_in_triangle (Y r₁ r₂ r₃ : ℝ) : Prop :=
  sorry -- Definition of this predicate is omitted for brevity

/-- Theorem stating that MaxCircleArea Y is indeed the maximum area for three
    non-overlapping circles that can fit inside an equilateral triangle of area Y. -/
theorem max_circle_area_is_correct (Y : ℝ) (h : Y > 0) :
  ∀ Z : ℝ, (∃ r₁ r₂ r₃ : ℝ,
    r₁ * r₁ = Z ∧ r₂ * r₂ = Z ∧ r₃ * r₃ = Z ∧
    three_circles_fit_in_triangle Y r₁ r₂ r₃) →
  Z ≤ MaxCircleArea Y :=
by
  sorry -- Proof is omitted for brevity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_circle_area_is_correct_l927_92779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_f_eq_10_l927_92718

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 1 else -2*x

-- State the theorem
theorem unique_solution_f_eq_10 :
  ∃! x : ℝ, f x = 10 ∧ x = -3 := by
  sorry

#check unique_solution_f_eq_10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_f_eq_10_l927_92718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_bound_l927_92733

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  vertices : List (ℝ × ℝ)
  convex : ∀ (a b : ℝ × ℝ), a ∈ vertices → b ∈ vertices → ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → 
    (1 - t) • a + t • b ∈ vertices

/-- A line segment in 2D space -/
structure LineSegment where
  start : ℝ × ℝ
  stop : ℝ × ℝ

/-- The length of a line segment -/
noncomputable def length (s : LineSegment) : ℝ :=
  Real.sqrt ((s.stop.1 - s.start.1)^2 + (s.stop.2 - s.start.2)^2)

/-- Check if a point is inside a convex polygon -/
def isInside (p : ℝ × ℝ) (poly : ConvexPolygon) : Prop := sorry

/-- The maximum length of sides and diagonals of a convex polygon -/
noncomputable def maxSideOrDiagonalLength (poly : ConvexPolygon) : ℝ := sorry

/-- Main theorem: The length of any line segment inside a convex polygon
    does not exceed the maximum length of the polygon's sides and diagonals -/
theorem segment_length_bound (poly : ConvexPolygon) (seg : LineSegment)
    (h1 : isInside seg.start poly) (h2 : isInside seg.stop poly) :
    length seg ≤ maxSideOrDiagonalLength poly := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_bound_l927_92733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equals_100i_l927_92731

-- Define complex numbers Q, E, D and real number R
def Q : ℂ := 3 + 4*Complex.I
def E : ℂ := 2*Complex.I
def D : ℂ := 3 - 4*Complex.I
def R : ℝ := 2

-- State the theorem
theorem product_equals_100i : Q * E * D * R = 100 * Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equals_100i_l927_92731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roses_more_expensive_than_three_carnations_l927_92785

/-- The price of a single rose -/
def x : ℝ := sorry

/-- The price of a single carnation -/
def y : ℝ := sorry

/-- The total price of 6 roses and 3 carnations is more than 24 yuan -/
axiom condition1 : 6 * x + 3 * y > 24

/-- The total price of 4 roses and 5 carnations is less than 22 yuan -/
axiom condition2 : 4 * x + 5 * y < 22

/-- Theorem: The price of 2 roses is higher than the price of 3 carnations -/
theorem two_roses_more_expensive_than_three_carnations :
  2 * x > 3 * y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roses_more_expensive_than_three_carnations_l927_92785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_count_l927_92734

-- Define the circles
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
def C₂ : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 4)^2 = 16}

-- Define the centers and radii
def center₁ : ℝ × ℝ := (0, 0)
def center₂ : ℝ × ℝ := (3, 4)
def radius₁ : ℝ := 1
def radius₂ : ℝ := 4

-- Define a tangent line
def IsTangentLine (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop :=
  ∃ p, p ∈ c ∧ p ∈ l ∧ ∀ q ∈ c, q ≠ p → q ∉ l

-- Theorem statement
theorem common_tangents_count :
  ∃ (tangents : Finset (Set (ℝ × ℝ))),
    (∀ t ∈ tangents, IsTangentLine t C₁ ∧ IsTangentLine t C₂) ∧
    Finset.card tangents = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_count_l927_92734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l927_92774

-- Define the sequences
noncomputable def a (n : ℕ) (t : ℝ) : ℝ := -n + t
noncomputable def b (n : ℕ) : ℝ := 3^(n - 3)
noncomputable def c (n : ℕ) (t : ℝ) : ℝ := (a n t + b n)/2 + |a n t - b n|/2

-- State the theorem
theorem sequence_inequality (t : ℝ) : 
  (∀ n : ℕ, n ≥ 1 → c n t ≥ c 3 t) ↔ 3 ≤ t ∧ t ≤ 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l927_92774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_formula_hexagon_area_pos_l927_92782

/-- The area of a hexagon formed by connecting the outer vertices of squares
    constructed on the sides of an equilateral triangle with side length a. -/
noncomputable def hexagon_area (a : ℝ) : ℝ :=
  a^2 * (3 + Real.sqrt 3)

/-- Theorem stating that the area of the hexagon described above
    is equal to a² (3 + √3) -/
theorem hexagon_area_formula (a : ℝ) (h : a > 0) :
  hexagon_area a = a^2 * (3 + Real.sqrt 3) := by
  -- Unfold the definition of hexagon_area
  unfold hexagon_area
  -- The equality holds by definition
  rfl

/-- Auxiliary theorem: The area of the hexagon is positive for positive side length -/
theorem hexagon_area_pos (a : ℝ) (h : a > 0) :
  hexagon_area a > 0 := by
  unfold hexagon_area
  have h1 : 3 + Real.sqrt 3 > 0 := by
    apply add_pos
    · exact three_pos
    · exact Real.sqrt_pos.mpr three_pos
  exact mul_pos (pow_pos h 2) h1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_formula_hexagon_area_pos_l927_92782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bee_path_theorem_l927_92743

/-- The total distance traveled by a bee on two concentric circles -/
noncomputable def bee_path_distance (r₁ r₂ : ℝ) : ℝ :=
  (1/8) * 2 * Real.pi * r₂ + 2 * r₂

/-- Theorem stating the total distance traveled by the bee -/
theorem bee_path_theorem :
  bee_path_distance 15 25 = (25 * Real.pi / 4) + 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bee_path_theorem_l927_92743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l927_92747

open Real

/-- A function satisfying the given conditions -/
def SatisfiesConditions (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f ∧ 
  (∀ x, x ≠ 1 → (((deriv (deriv f)) x - f x) / (x - 1) > 0)) ∧
  (∀ x, f (2 - x) = f x * Real.exp (2 - 2*x))

/-- The main theorem to prove -/
theorem function_inequality {f : ℝ → ℝ} (hf : SatisfiesConditions f) : 
  f 3 > Real.exp 3 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l927_92747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clash_of_minds_impossibility_l927_92798

theorem clash_of_minds_impossibility (scores : Fin 10 → ℕ) 
  (h_distinct : ∀ i j, i ≠ j → scores i % 10 ≠ scores j % 10) :
  ∃ k, (2 * (Finset.sum Finset.univ scores)) % 10 ≠ scores k % 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clash_of_minds_impossibility_l927_92798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_A_is_HIO₃_l927_92764

/-- Represents a chemical compound --/
structure Compound where
  formula : String
  molarMass : ℚ

/-- Represents a chemical reaction --/
structure Reaction where
  reactants : List Compound
  products : List Compound

/-- The mass of the precipitate formed --/
def precipitateMass : ℚ := 48/100

/-- The mass of compound A used --/
def compoundAMass : ℚ := 88/100

/-- The molar mass of sulfur --/
def sulfurMolarMass : ℚ := 32

/-- Function to calculate the number of moles --/
def calculateMoles (mass : ℚ) (molarMass : ℚ) : ℚ :=
  mass / molarMass

/-- Function to check if a compound matches the given conditions --/
def isCompoundA (c : Compound) : Prop :=
  let molesIodine := calculateMoles precipitateMass sulfurMolarMass
  let molesCompoundA := calculateMoles compoundAMass c.molarMass
  molesIodine = molesCompoundA ∧ c.molarMass = 17591/100

/-- Theorem stating that HIO₃ is compound A --/
theorem compound_A_is_HIO₃ :
  let HIO₃ : Compound := { formula := "HIO₃", molarMass := 17591/100 }
  isCompoundA HIO₃ := by
  sorry

#eval calculateMoles precipitateMass sulfurMolarMass
#eval calculateMoles compoundAMass (17591/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_A_is_HIO₃_l927_92764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_payment_correct_l927_92707

/-- The interest rate as a percentage -/
noncomputable def p : ℝ := 4

/-- The interest factor -/
noncomputable def e : ℝ := 1 + p / 100

/-- The number of years of payments -/
def payment_years : ℕ := 10

/-- The number of years between last payment and first receipt -/
def gap_years : ℕ := 5

/-- The number of years of receipts -/
def receipt_years : ℕ := 15

/-- The annual receipt amount -/
noncomputable def annual_receipt : ℝ := 1500

/-- The annual payment amount -/
noncomputable def annual_payment : ℝ := 
  (annual_receipt * (e^receipt_years - 1)) / (e^(payment_years + gap_years + receipt_years - 1) * (e^payment_years - 1))

theorem annual_payment_correct :
  annual_payment = (annual_receipt * (e^receipt_years - 1)) / (e^(payment_years + gap_years + receipt_years - 1) * (e^payment_years - 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_payment_correct_l927_92707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_l927_92701

-- Define the space and points
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
variable (A B C O D : V)

-- Define the conditions
variable (h1 : D = (B + C) / 2)
variable (h2 : 4 • (O - A) + (O - B) + (O - C) = 0)

-- State the theorem
theorem vector_equality : 2 • (A - O) = O - D := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_l927_92701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_not_even_l927_92706

theorem percentage_not_even (A : Finset ℕ) 
  (h1 : (A.filter (λ x => x % 2 = 0 ∧ x % 3 = 0)).card = (36 : ℕ) * A.card / 100)
  (h2 : (A.filter (λ x => x % 2 = 0 ∧ x % 3 ≠ 0)).card = (40 : ℕ) * (A.filter (λ x => x % 2 = 0)).card / 100) :
  (A.filter (λ x => x % 2 ≠ 0)).card = (40 : ℕ) * A.card / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_not_even_l927_92706
