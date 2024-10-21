import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2008_minus_one_l555_55545

def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b

def f_n (a b : ℝ) : ℕ → (ℝ → ℝ)
  | 0 => λ x => x  -- Base case for n = 0
  | 1 => f a b
  | n + 1 => f a b ∘ f_n a b n

theorem f_2008_minus_one (a b : ℝ) (h : f_n a b 5 = λ x ↦ 32 * x + 31) :
  f_n a b 2008 (-1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2008_minus_one_l555_55545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_partition_exists_l555_55515

/-- Represents a grid square partition --/
structure GridPartition where
  size : Nat
  squares : List (Nat × Nat)

/-- Checks if all squares in the partition have the same size --/
def allSameSize (partition : GridPartition) : Prop :=
  ∀ s₁ s₂, s₁ ∈ partition.squares → s₂ ∈ partition.squares → s₁ = s₂

/-- Checks if the number of squares of each size is the same --/
def equalCountSizes (partition : GridPartition) : Prop :=
  ∀ s₁ s₂, s₁ ∈ partition.squares → s₂ ∈ partition.squares →
    (partition.squares.filter (· = s₁)).length = (partition.squares.filter (· = s₂)).length

/-- The main theorem --/
theorem grid_partition_exists : ∃ (partition : GridPartition),
  partition.size = 8 ∧
  ¬allSameSize partition ∧
  equalCountSizes partition ∧
  partition.squares = [(4, 4), (4, 4), (4, 4), (4, 4)] := by
  sorry

#check grid_partition_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_partition_exists_l555_55515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_implies_a_value_l555_55524

noncomputable def f (x a : ℝ) : ℝ := Real.sin (2*x + Real.pi/6) + Real.sin (2*x - Real.pi/6) + Real.cos (2*x) + a

theorem f_minimum_implies_a_value :
  ∀ a : ℝ, 
  (∀ x ∈ Set.Icc 0 (Real.pi/2), f x a ≥ -2) ∧ 
  (∃ x ∈ Set.Icc 0 (Real.pi/2), f x a = -2) →
  a = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_implies_a_value_l555_55524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_scaled_vectors_l555_55517

noncomputable def angle_between (v w : ℝ × ℝ × ℝ) : ℝ := sorry

theorem angle_between_scaled_vectors 
  (c d : ℝ × ℝ × ℝ) (h : angle_between c d = 60) : 
  angle_between (-2 • c) (3 • d) = 120 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_scaled_vectors_l555_55517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_surface_area_l555_55509

/-- The volume of a regular tetrahedron in cubic decimeters -/
def tetrahedron_volume : ℝ := 9

/-- The surface area of a regular tetrahedron in square decimeters -/
noncomputable def tetrahedron_surface_area : ℝ := 18 * Real.sqrt 3

/-- Theorem: The surface area of a regular tetrahedron with volume 9 dm³ is 18√3 dm² -/
theorem regular_tetrahedron_surface_area :
  tetrahedron_surface_area = 18 * Real.sqrt 3 := by
  -- The proof goes here
  sorry

#eval tetrahedron_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_surface_area_l555_55509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l555_55592

theorem intersection_distance (m n : ℕ) (hm : m > 0) (hn : n > 0) (hcoprime : Nat.Coprime m n) :
  (∃ A B : ℝ × ℝ,
    (A.2 = 3 ∧ A.2 = 4 * A.1^2 + A.1 - 1) ∧
    (B.2 = 3 ∧ B.2 = 4 * B.1^2 + B.1 - 1) ∧
    (B.1 - A.1)^2 = m / (n^2 : ℝ)) →
  m - n = 61 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l555_55592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_fraction_is_23_30_l555_55514

def known_fractions : List ℚ := [1/3, -5/6, 1/5, 1/4, -9/20, -2/15]
def total_sum : ℚ := 13333333333333333 / 100000000000000000

theorem missing_fraction_is_23_30 :
  ∃ x : ℚ, x + (known_fractions.foldl (· + ·) 0) = total_sum ∧ x = 23/30 := by
  sorry

#eval known_fractions.foldl (· + ·) 0
#eval total_sum - known_fractions.foldl (· + ·) 0

end NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_fraction_is_23_30_l555_55514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anniversary_probability_l555_55512

-- Define the type for days of the week
inductive DayOfWeek
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

-- Define a function to check if a year is a leap year
def isLeapYear (year : Nat) : Bool :=
  year % 4 = 0

-- Define a function to calculate the day of the week after n years
def dayAfterNYears (startDay : DayOfWeek) (startYear : Nat) (n : Nat) : DayOfWeek :=
  sorry

-- Theorem statement
theorem anniversary_probability :
  ∀ (startYear : Nat),
    startYear ≥ 1668 → startYear ≤ 1671 →
    ∃ (fridayProb thursdayProb : ℚ),
      fridayProb = 3/4 ∧
      thursdayProb = 1/4 ∧
      (∀ (day : DayOfWeek),
        day ≠ DayOfWeek.Friday ∧ day ≠ DayOfWeek.Thursday →
        (dayAfterNYears DayOfWeek.Friday startYear 11 = day → False)) ∧
      (dayAfterNYears DayOfWeek.Friday startYear 11 = DayOfWeek.Friday →
        fridayProb = 3/4) ∧
      (dayAfterNYears DayOfWeek.Friday startYear 11 = DayOfWeek.Thursday →
        thursdayProb = 1/4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_anniversary_probability_l555_55512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_condition_l555_55542

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if (b-a)(sin B + sin A) = c(√3 sin B - sin C), then A = π/6 -/
theorem triangle_special_condition (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  ((b - a) * (Real.sin B + Real.sin A) = c * (Real.sqrt 3 * Real.sin B - Real.sin C)) →
  A = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_condition_l555_55542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_l555_55582

/-- The function y = 3^x + a - 1 has a zero point -/
def has_zero_point (a : ℝ) : Prop :=
  ∃ x : ℝ, 3^x + a - 1 = 0

/-- The function y = log_a x is a decreasing function on (0, +∞) -/
def is_decreasing_log (a : ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → Real.log x / Real.log a > Real.log y / Real.log a

theorem necessary_but_not_sufficient :
  (∀ a : ℝ, is_decreasing_log a → has_zero_point a) ∧
  (∃ a : ℝ, has_zero_point a ∧ ¬is_decreasing_log a) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_l555_55582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_specific_line_l555_55540

/-- The distance from a point (x₀, y₀) to a line ax + by + c = 0 is |ax₀ + by₀ + c| / √(a² + b²) -/
noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  abs (a * x₀ + b * y₀ + c) / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance from the point (1, 2) to the line y = 2x + 1 is √5/5 -/
theorem distance_point_to_specific_line :
  distance_point_to_line 1 2 2 (-1) 1 = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_specific_line_l555_55540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_plus_cd_composite_l555_55579

theorem ab_plus_cd_composite
  (a b c d : ℕ)
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_order : a > b ∧ b > c ∧ c > d)
  (h_equation : a * c + b * d = (b + c + d - a) * (a + b - c + d)) :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ x * y = a * b + c * d :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_plus_cd_composite_l555_55579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l555_55557

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x * Real.exp (x - 1)

-- State the theorem
theorem tangent_slope_at_one : 
  (deriv f) 1 = 2 := by
  -- Differentiate f
  have h1 : deriv f = fun x => Real.exp (x - 1) + x * Real.exp (x - 1) := by
    sorry -- Proof of differentiation
  
  -- Simplify the derivative at x = 1
  have h2 : (fun x => Real.exp (x - 1) + x * Real.exp (x - 1)) 1 = 2 := by
    sorry -- Proof of simplification
  
  -- Combine the steps
  rw [h1]
  exact h2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l555_55557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_f_g_l555_55536

open Real

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := exp x
noncomputable def g (x : ℝ) : ℝ := log (abs x)

-- Define the interval
def I : Set ℝ := Set.Iio 0  -- (-∞, 0)

-- State the theorem
theorem monotonicity_f_g :
  (∀ x y, x ∈ I → y ∈ I → x < y → f x < f y) ∧
  (∀ x y, x ∈ I → y ∈ I → x < y → g x > g y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_f_g_l555_55536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_theorem_l555_55504

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The left focus of the ellipse -/
noncomputable def leftFocus (e : Ellipse) : Point :=
  { x := -Real.sqrt (e.a^2 - e.b^2), y := 0 }

/-- The right focus of the ellipse -/
noncomputable def rightFocus (e : Ellipse) : Point :=
  { x := Real.sqrt (e.a^2 - e.b^2), y := 0 }

/-- Check if a point is on the ellipse -/
def onEllipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Perimeter of a triangle given its three vertices -/
noncomputable def trianglePerimeter (p1 p2 p3 : Point) : ℝ :=
  distance p1 p2 + distance p2 p3 + distance p3 p1

/-- Symmetric point of p with respect to x-axis -/
def symmetricPointX (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- Main theorem about the ellipse and the fixed point -/
theorem ellipse_fixed_point_theorem (e : Ellipse) 
  (h_min_AB : ∀ A B : Point, onEllipse e A → onEllipse e B → 
    distance (leftFocus e) A + distance (leftFocus e) B = 2 * e.a → distance A B ≥ 3)
  (h_perimeter : ∀ A B : Point, onEllipse e A → onEllipse e B → 
    distance (leftFocus e) A + distance (leftFocus e) B = 2 * e.a → 
    trianglePerimeter A B (rightFocus e) = 8) :
  (e.a = 2 ∧ e.b = Real.sqrt 3) ∧
  (∀ A B : Point, onEllipse e A → onEllipse e B → 
    distance (leftFocus e) A + distance (leftFocus e) B = 2 * e.a →
    A.x ≠ B.x → ∃ k : ℝ, 
      (symmetricPointX A).y - B.y = k * ((symmetricPointX A).x - B.x) ∧
      0 = k * (-4 - B.x) + B.y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_theorem_l555_55504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_objective_function_l555_55526

/-- An arithmetic sequence with n+2 terms -/
structure ArithmeticSequence (n : ℕ) where
  l : ℝ
  a : ℝ
  b : ℝ
  sum_ab : a + b = 18
  last_term : l + (n + 1) * ((17 - l) / (n + 1)) = 17

/-- The function to be minimized -/
noncomputable def objective_function (a b : ℝ) : ℝ := 1 / a + 25 / b

/-- The theorem statement -/
theorem minimize_objective_function :
  ∃ (n : ℕ), ∀ (seq : ArithmeticSequence n),
    (∀ (m : ℕ) (seq' : ArithmeticSequence m),
      objective_function seq'.a seq'.b ≤ objective_function seq.a seq.b) →
    n = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_objective_function_l555_55526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l555_55550

/-- The minimum value of 1/a + 4/b given the conditions -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line : a + b = 1) 
  (h_center : a * 1 + b * 1 = 1) : 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → a' + b' = 1 → 1 / a' + 4 / b' ≥ 1 / a + 4 / b) ∧ 
  1 / a + 4 / b = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l555_55550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_l555_55539

/-- The radius of circle A -/
def radiusA : ℝ := 10

/-- The radius of circle B -/
def radiusB : ℝ := 3

/-- The radius of circle C and D -/
def radiusCD : ℝ := 2

/-- The side length of the equilateral triangle T -/
noncomputable def sideT : ℝ := radiusA * Real.sqrt 3

/-- The radius of circle E -/
def radiusE : ℚ := 27 / 5

/-- The numerator of radiusE -/
def m : ℕ := 27

/-- The denominator of radiusE -/
def n : ℕ := 5

theorem circle_tangency (radiusA radiusB radiusCD : ℝ) (radiusE : ℚ) (m n : ℕ) 
  (h1 : radiusA = 10)
  (h2 : radiusB = 3)
  (h3 : radiusCD = 2)
  (h4 : radiusE = m / n)
  (h5 : Nat.Coprime m n) :
  m + n = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_l555_55539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_series_sum_l555_55576

noncomputable def f (x : ℝ) : ℝ := 
  Real.pi / 2 - (4 / Real.pi) * (∑' n, Real.cos ((2 * n + 1) * x) / (2 * n + 1)^2)

axiom f_eq_x : ∀ x ∈ Set.Icc 0 Real.pi, f x = x

theorem f_properties :
  (∀ x, f (Real.pi - x) + f x = Real.pi) ∧
  f (15 * Real.pi / 4) = Real.pi / 4 ∧
  f 0 = 0 :=
sorry

-- Additional theorem for statement D
theorem series_sum :
  (∑' n, 1 / (2 * n + 1)^2) = Real.pi^2 / 8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_series_sum_l555_55576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_equality_l555_55552

theorem shaded_areas_equality (φ : Real) (s : Real) 
  (h1 : 0 < φ) (h2 : φ < π / 4) (h3 : s > 0) :
  (φ * s^2 / 2 - s^2 * Real.tan (2 * φ) / 2 = φ * s^2 / 2) ↔ 
  (Real.tan (2 * φ) = 2 * φ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_equality_l555_55552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_special_set_l555_55547

noncomputable def three_number_set (a b c : ℝ) : Prop :=
  a ≤ b ∧ b ≤ c

noncomputable def mean (a b c : ℝ) : ℝ :=
  (a + b + c) / 3

noncomputable def median (a b c : ℝ) : ℝ :=
  b

noncomputable def range (a b c : ℝ) : ℝ :=
  c - a

theorem range_of_special_set (a b c : ℝ) 
  (h_set : three_number_set a b c)
  (h_mean : mean a b c = 4)
  (h_median : median a b c = 4)
  (h_smallest : a = 1) :
  range a b c = 6 := by
  sorry

#check range_of_special_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_special_set_l555_55547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_range_l555_55525

-- Define the quadratic function
def f (a x : ℝ) : ℝ := x^2 - a*x - 6*a

-- Define the solution interval length
noncomputable def solution_interval_length (a : ℝ) : ℝ := 
  Real.sqrt (a^2 + 24*a)

-- Theorem statement
theorem quadratic_inequality_solution_range :
  ∀ a : ℝ, (∀ x : ℝ, f a x < 0 → solution_interval_length a ≤ 5) ↔ 
  ((-25 ≤ a ∧ a < -24) ∨ (0 < a ∧ a ≤ 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_range_l555_55525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sqrt_50_squared_l555_55565

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sqrt_50_squared_l555_55565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_composition_parity_invariant_l555_55581

/-- A reflection about a line in a plane. -/
structure Reflection where
  line : Real → Real → Bool  -- Simplified representation of a line

/-- A composition of reflections. -/
structure ReflectionComposition where
  reflections : List Reflection

/-- The parity of a natural number. -/
inductive Parity
  | Even
  | Odd

/-- Get the parity of the number of reflections in a composition. -/
def ReflectionComposition.parity (comp : ReflectionComposition) : Parity :=
  if comp.reflections.length % 2 == 0 then Parity.Even else Parity.Odd

/-- Apply a reflection to a point. -/
def applyReflection (r : Reflection) (p : Real × Real) : Real × Real :=
  sorry  -- Implementation details omitted for simplicity

/-- Apply a composition of reflections to a point. -/
def applyComposition (comp : ReflectionComposition) (p : Real × Real) : Real × Real :=
  comp.reflections.foldl (fun p r => applyReflection r p) p

/-- 
Theorem: The parity of the number of reflections in any composition of 
reflections about lines in a plane is invariant.
-/
theorem reflection_composition_parity_invariant 
  (comp1 comp2 : ReflectionComposition) 
  (h : ∀ p, applyComposition comp1 p = applyComposition comp2 p) : 
  comp1.parity = comp2.parity :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_composition_parity_invariant_l555_55581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sheep_can_always_evade_l555_55523

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the state of the game -/
structure GameState where
  wolf : Point
  sheep : List Point

/-- Defines a valid move (distance ≤ 1) -/
def validMove (p1 p2 : Point) : Prop :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 ≤ 1

/-- The sheep strategy function type -/
def SheepStrategy := GameState → Nat → Point

/-- The wolf strategy function type -/
def WolfStrategy := GameState → Point

/-- Checks if the wolf has caught a sheep -/
def wolfCaught (state : GameState) : Prop :=
  ∃ s ∈ state.sheep, s = state.wolf

/-- Simulates the game for n moves -/
def iterateGame (n : Nat) (initialState : GameState) (wolfStrategy : WolfStrategy) (sheepStrategy : SheepStrategy) : GameState :=
  sorry

/-- Main theorem: There exists a sheep strategy that always evades the wolf -/
theorem sheep_can_always_evade :
  ∃ (sheepStrategy : SheepStrategy),
    ∀ (initialState : GameState),
      initialState.sheep.length = 100 →
        ∀ (wolfStrategy : WolfStrategy),
          ∀ (n : Nat),
            let gameAfterNMoves := iterateGame n initialState wolfStrategy sheepStrategy
            ¬(wolfCaught gameAfterNMoves) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sheep_can_always_evade_l555_55523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_inverse_relation_l555_55587

def B (e : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 7, e]

theorem matrix_inverse_relation (e m : ℝ) :
  (B e)⁻¹ = m • (B e) ^ 2 → e = -2 ∧ m = 1/11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_inverse_relation_l555_55587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_cost_sharing_l555_55564

/-- The total amount contributed by all three people -/
noncomputable def total_contribution : ℝ := 150 + 90 + 210

/-- The equal share each person should contribute -/
noncomputable def equal_share : ℝ := total_contribution / 3

/-- The amount Mike contributed -/
def mike_contribution : ℝ := 150

/-- The amount Jane contributed -/
def jane_contribution : ℝ := 90

/-- The amount Casey contributed -/
def casey_contribution : ℝ := 210

/-- The amount Mike gave to Casey -/
def m : ℝ := 0

/-- The amount Jane gave to Casey -/
noncomputable def j : ℝ := equal_share - jane_contribution

theorem project_cost_sharing :
  m - j = -60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_cost_sharing_l555_55564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_explicit_f_range_of_t_l555_55511

-- Define the function f with parameters b and c
def f (b c x : ℝ) : ℝ := 2 * x^2 + b * x + c

-- Define the solution set of f(x) < 0
def solution_set : Set ℝ := { x | 0 < x ∧ x < 1 }

-- Define the constraint for t
def t_constraint (b c t : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f b c x + t ≤ 2

-- Theorem 1: Prove the explicit form of f
theorem explicit_f : 
  ∃ b c : ℝ, (∀ x, x ∈ solution_set ↔ f b c x < 0) ∧ 
             (∀ x, f b c x = 2 * x^2 - 2 * x) :=
sorry

-- Theorem 2: Prove the range of t
theorem range_of_t : 
  ∀ t : ℝ, (∃ b c : ℝ, t_constraint b c t) → t ≤ -2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_explicit_f_range_of_t_l555_55511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l555_55568

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * Real.log x

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x > 1, f a x > 0) → a < Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l555_55568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l555_55578

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, (0 < x ∧ x < 5) → (|x - 2| < 3)) ∧ 
  (∃ x : ℝ, |x - 2| < 3 ∧ ¬(0 < x ∧ x < 5)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l555_55578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l555_55538

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define O as the circumcenter
variable (O : EuclideanSpace ℝ (Fin 2))

-- Define that the triangle is acute and A is the largest angle
variable (h_acute : IsAcute A B C)
variable (h_A_largest : IsLargestAngle A B C)

-- Define the vector equation
variable (m : ℝ)
variable (h_vector_eq : (Real.cos (angle B) / Real.sin (angle C)) • (B - A) + 
                        (Real.cos (angle C) / Real.sin (angle B)) • (C - A) = 
                        m • (O - A))

-- State the theorem
theorem range_of_m :
  m ∈ Set.Icc (Real.sqrt 3) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l555_55538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_vs_2sin_x_l555_55516

theorem sin_2x_vs_2sin_x (x : ℝ) :
  (∃ n : ℤ, 2 * Real.pi * n - Real.pi ≤ x ∧ x ≤ 2 * Real.pi * n → Real.sin (2 * x) ≥ 2 * Real.sin x) ∧
  (∃ n : ℤ, 2 * Real.pi * n ≤ x ∧ x ≤ Real.pi + 2 * Real.pi * n → Real.sin (2 * x) ≤ 2 * Real.sin x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_vs_2sin_x_l555_55516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_square_l555_55528

noncomputable def a (n : ℕ) : ℝ := (1 / Real.sqrt 5) * (((1 + Real.sqrt 5) / 2) ^ n - ((1 - Real.sqrt 5) / 2) ^ n)

theorem fibonacci_square (m : ℕ) (h : Odd m) : 
  a (m + 4) * a m - 1 = (a (m + 2))^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_square_l555_55528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C_inverse_exists_and_equals_l555_55534

def C : Matrix (Fin 3) (Fin 3) ℚ :=
  !![1, 2, 1;
     3, -5, 3;
     2, 7, -1]

def C_inv : Matrix (Fin 3) (Fin 3) ℚ :=
  !![(-16:ℚ)/33, 9/33, 11/33;
     9/33, (-3:ℚ)/33, 0;
     31/33, (-3:ℚ)/33, (-11:ℚ)/33]

theorem C_inverse_exists_and_equals :
  IsUnit (Matrix.det C) ∧ C⁻¹ = C_inv := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C_inverse_exists_and_equals_l555_55534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_complex_expression_l555_55543

theorem max_value_complex_expression :
  ∃ (a₀ b₀ : ℝ), 
  let z₀ : ℂ := Complex.mk a₀ b₀
  Complex.abs z₀ = 2 ∧ 
  ∀ (a b : ℝ), 
  let z : ℂ := Complex.mk a b
  Complex.abs z = 2 → Complex.abs ((z - 1) * (z + 1)^2) ≤ Complex.abs ((z₀ - 1) * (z₀ + 1)^2) ∧
  Complex.abs ((z₀ - 1) * (z₀ + 1)^2) = 9 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_complex_expression_l555_55543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_in_similar_triangles_l555_55588

-- Define the triangle ABC
structure Triangle (α : Type*) [Field α] where
  A : α × α
  B : α × α
  C : α × α
  isIsosceles : A.1 = B.1 ∧ A.2 = C.2

-- Define the trapezoid DBCE
structure Trapezoid (α : Type*) [Field α] where
  D : α × α
  B : α × α
  C : α × α
  E : α × α

-- Define the area function for the trapezoid
noncomputable def Trapezoid.area {α : Type*} [Field α] (DBCE : Trapezoid α) : α :=
  sorry -- The actual calculation of the area would go here

-- Define the theorem
theorem trapezoid_area_in_similar_triangles 
  {α : Type*} [Field α] (ABC : Triangle α) (DBCE : Trapezoid α) 
  (smallest_triangle_area : α) (num_smallest_triangles : ℕ) (ABC_area : α) :
  smallest_triangle_area = 2 →
  num_smallest_triangles = 9 →
  ABC_area = 72 →
  (DBCE.area : α) = 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_in_similar_triangles_l555_55588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_section_area_l555_55502

/-- 
Represents a parallelepiped with edges a ≤ b ≤ c.
-/
structure Parallelepiped where
  a : ℝ
  b : ℝ
  c : ℝ
  h_ab : a ≤ b
  h_bc : b ≤ c

/-- 
The area of a section passing through a diagonal of the parallelepiped.
-/
noncomputable def sectionArea (p : Parallelepiped) (x y : ℝ) : ℝ :=
  x * Real.sqrt (y^2 + p.c^2)

/-- 
Theorem: The maximum area of a section passing through a diagonal 
of a parallelepiped with edges a ≤ b ≤ c is c√(a² + b²).
-/
theorem max_section_area (p : Parallelepiped) :
  ∀ x y, x ≤ p.c ∧ y ≤ p.c → sectionArea p x y ≤ p.c * Real.sqrt (p.a^2 + p.b^2) := by
  sorry

#check max_section_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_section_area_l555_55502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_exp_linear_l555_55535

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.exp (-5 * x + 2)

-- State the theorem
theorem derivative_of_exp_linear (x : ℝ) :
  deriv f x = -5 * Real.exp (-5 * x + 2) := by
  -- The proof is skipped using 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_exp_linear_l555_55535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_function_period_l555_55570

-- Define the cosine function with parameters
noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ := a * Real.cos (b * x + c) + d

-- Define the period of the function
noncomputable def period (a b c d : ℝ) : ℝ := 2 * Real.pi / b

-- Theorem statement
theorem cosine_function_period 
  (a b c d : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
  (h5 : period a b c d = Real.pi) : 
  b = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_function_period_l555_55570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_a_and_b_l555_55537

noncomputable def f (a b x : ℝ) : ℝ := x / (a * x + b)

def passes_through_point (a b : ℝ) : Prop :=
  f a b (-4) = 4

def symmetric_about_neg_x (a b : ℝ) : Prop :=
  ∀ x y : ℝ, f a b x = y → f a b (-y) = -x

theorem sum_of_a_and_b (a b : ℝ) (ha : a ≠ 0) 
  (h1 : passes_through_point a b) 
  (h2 : symmetric_about_neg_x a b) : 
  a + b = 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_a_and_b_l555_55537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_polygon_sum_equals_134_l555_55549

/-- The area of a 24-sided polygon formed by three overlapping squares --/
noncomputable def overlapping_squares_area (side_length : ℝ) : ℝ :=
  3 * side_length^2 - 2 * (2 * side_length^2 + 2 * side_length^2 * Real.sqrt 2) / 2 - 
  2 * side_length^2 - side_length^2 * Real.sqrt 2

/-- Theorem stating the area of the specific 24-sided polygon --/
theorem area_of_specific_polygon : 
  overlapping_squares_area 6 = 108 - 24 * Real.sqrt 2 := by
  sorry

/-- Compute the sum a + b + c --/
def compute_sum (a b c : ℕ) : ℕ := a + b + c

/-- Theorem stating the sum of a, b, and c --/
theorem sum_equals_134 : 
  compute_sum 108 24 2 = 134 := by
  rfl

#eval compute_sum 108 24 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_polygon_sum_equals_134_l555_55549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l555_55559

open Set

-- Define the function h as noncomputable
noncomputable def h (x : ℝ) : ℝ := (2 * x - 3) / (x - 5)

-- State the theorem about the domain of h
theorem domain_of_h :
  {x : ℝ | ∃ y, h x = y} = (Iio 5) ∪ (Ioi 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l555_55559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_calculation_l555_55510

-- Define the ⊗ operation
noncomputable def otimes (x y z : ℝ) : ℝ := x / (y - z)

-- Theorem statement
theorem otimes_calculation :
  otimes (otimes 2 5 3) (otimes 5 3 2) (otimes 3 2 5) = 1/6 :=
by
  -- Unfold the definition of otimes
  unfold otimes
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_calculation_l555_55510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l555_55589

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line with slope k
def line (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem parabola_line_intersection
  (A B : ℝ × ℝ)  -- Points of intersection
  (k : ℝ)        -- Slope of the line
  (h1 : parabola A.1 A.2)
  (h2 : parabola B.1 B.2)
  (h3 : line k A.1 A.2)
  (h4 : line k B.1 B.2)
  (h5 : distance A focus = 4 * distance B focus)
  : k = 4/3 ∨ k = -4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l555_55589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_positive_numbers_l555_55505

def list_D : List ℚ := List.range 20 |>.map (λ i => -15/2 + i)

def positive_numbers (L : List ℚ) : List ℚ := L.filter (λ x => x > 0)

theorem range_of_positive_numbers :
  (positive_numbers list_D).maximum?.getD 0 - (positive_numbers list_D).minimum?.getD 0 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_positive_numbers_l555_55505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maximal_area_quadrilateral_in_circle_l555_55548

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle with center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Helper function to calculate the area of a quadrilateral -/
noncomputable def area_quadrilateral (A B C D : Point) : ℝ :=
  sorry  -- Implementation details omitted for brevity

/-- Theorem: Maximal Area Quadrilateral in a Circle -/
theorem maximal_area_quadrilateral_in_circle 
  (circle : Circle) 
  (A : Point) 
  (h_A_inside : (A.x - circle.center.x)^2 + (A.y - circle.center.y)^2 < circle.radius^2) 
  (h_A_not_center : A ≠ circle.center) :
  ∃ (B C D : Point),
    (B.x - circle.center.x)^2 + (B.y - circle.center.y)^2 = circle.radius^2 ∧
    (C.x - circle.center.x)^2 + (C.y - circle.center.y)^2 = circle.radius^2 ∧
    (D.x - circle.center.x)^2 + (D.y - circle.center.y)^2 = circle.radius^2 ∧
    (B.x - D.x)^2 + (B.y - D.y)^2 = (2 * circle.radius)^2 ∧
    ((B.x - D.x) * (A.x - circle.center.x) + (B.y - D.y) * (A.y - circle.center.y) = 0) ∧
    (C.x - circle.center.x) * (A.x - circle.center.x) + (C.y - circle.center.y) * (A.y - circle.center.y) = 0 ∧
    ∀ (B' C' D' : Point),
      (B'.x - circle.center.x)^2 + (B'.y - circle.center.y)^2 = circle.radius^2 →
      (C'.x - circle.center.x)^2 + (C'.y - circle.center.y)^2 = circle.radius^2 →
      (D'.x - circle.center.x)^2 + (D'.y - circle.center.y)^2 = circle.radius^2 →
      area_quadrilateral A B C D ≥ area_quadrilateral A B' C' D' :=
by
  sorry  -- Proof details omitted for brevity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maximal_area_quadrilateral_in_circle_l555_55548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_five_l555_55566

theorem opposite_of_five : ∃ x : ℤ, x + 5 = 0 ∧ x = -5 := by
  use -5
  constructor
  · ring
  · rfl

#check opposite_of_five

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_five_l555_55566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_third_l555_55520

theorem tan_alpha_plus_pi_third (α β : ℝ) 
  (h1 : Real.tan (α + β) = 3 / 5)
  (h2 : Real.tan (β - π / 3) = 1 / 4) :
  Real.tan (α + π / 3) = 7 / 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_third_l555_55520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersecting_lines_l555_55558

/-- Represents a line in the form y = kx + b -/
structure Line where
  k : ℝ
  b : ℝ

/-- Represents the family of curves -/
def family_curve (m : ℝ) (x y : ℝ) : Prop :=
  4 * x^2 + 5 * y^2 - 8 * m * x - 20 * m * y + 24 * m^2 - 20 = 0

/-- The length of the intercepted segment -/
noncomputable def segment_length : ℝ := 5 / 3 * Real.sqrt 5

/-- Checks if a line intersects all curves in the family with the required segment length -/
def intersects_family (l : Line) : Prop :=
  ∀ m : ℝ, ∃ x1 x2 : ℝ,
    family_curve m x1 (l.k * x1 + l.b) ∧
    family_curve m x2 (l.k * x2 + l.b) ∧
    (x2 - x1)^2 + (l.k * x2 + l.b - (l.k * x1 + l.b))^2 = segment_length^2

/-- The main theorem stating that only two specific lines satisfy the condition -/
theorem unique_intersecting_lines :
  ∀ l : Line, intersects_family l ↔ (l = ⟨2, 2⟩ ∨ l = ⟨2, -2⟩) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersecting_lines_l555_55558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_is_negative_fourteen_l555_55567

-- Define the binomial expansion
noncomputable def binomial_expansion (x : ℝ) : ℝ := (x^(1/3) - 2/x)^7

-- Define the coefficient of x in the expansion
noncomputable def coefficient_of_x (expansion : ℝ → ℝ) : ℝ :=
  sorry  -- The actual computation of the coefficient

-- Theorem statement
theorem coefficient_is_negative_fourteen :
  coefficient_of_x binomial_expansion = -14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_is_negative_fourteen_l555_55567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_sum_l555_55529

theorem log_equation_sum (A B C : ℕ+) : 
  Nat.Coprime A.val (Nat.gcd B.val C.val) →
  A * Real.log 5 / Real.log 50 + B * Real.log 2 / Real.log 50 = C →
  A + B + C = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_sum_l555_55529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_range_l555_55593

-- Define the given equation
def given_equation (x y : ℝ) : Prop :=
  3 * Real.sin y = 2 * (Real.sin y)^2 + (Real.sin x)^2

-- Define the expression we're interested in
noncomputable def expression (x y : ℝ) : ℝ :=
  Real.cos (2 * x) - 2 * Real.sin y

-- State the theorem
theorem expression_range :
  ∃ (S : Set ℝ), S = Set.Icc (-2) 1 ∪ {-3} ∧
  (∀ z ∈ S, ∃ x y, given_equation x y ∧ expression x y = z) ∧
  (∀ x y, given_equation x y → expression x y ∈ S) := by
  sorry

#check expression_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_range_l555_55593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_line_l555_55556

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line
def my_line (x y a t : ℝ) : Prop := x = a + t ∧ y = 2*t

-- Define the tangency condition
def is_tangent (a : ℝ) : Prop :=
  ∃ (x y t : ℝ), my_circle x y ∧ my_line x y a t

-- Theorem statement
theorem circle_tangent_line (a : ℝ) :
  is_tangent a → (a = Real.sqrt (5/4) ∨ a = -Real.sqrt (5/4)) :=
by
  sorry

#check circle_tangent_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_line_l555_55556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bijection_and_inverse_l555_55590

-- Define the set S₀
def S₀ : Set ℂ := {z : ℂ | Complex.abs z = 1 ∧ z ≠ -1}

-- Define the function f
noncomputable def f (z : ℂ) : ℝ := z.im / (1 + z.re)

-- Define the inverse function f⁻¹
noncomputable def f_inv (y : ℝ) : ℂ := Complex.mk ((1 - y^2) / (1 + y^2)) ((2*y) / (1 + y^2))

-- Theorem statement
theorem f_bijection_and_inverse :
  (Function.Bijective f) ∧ (∀ y : ℝ, f (f_inv y) = y) ∧ (∀ z ∈ S₀, f_inv (f z) = z) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bijection_and_inverse_l555_55590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_represents_circle_l555_55546

/-- The equation r = 4 cot θ csc θ represents a circle -/
theorem equation_represents_circle :
  ∀ (r θ x y : ℝ), r > 0 → θ ∈ Set.Ioo 0 π →
  (r = 4 * (Real.cos θ / Real.sin θ) * (1 / Real.sin θ)) →
  (x = r * Real.cos θ) →
  (y = r * Real.sin θ) →
  ∃ (c : ℝ), c > 0 ∧ x^2 + y^2 = c^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_represents_circle_l555_55546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_countable_set_with_uncountable_family_l555_55585

-- Define the properties we need
def IsCountable (α : Type*) : Prop := Countable α

def HasFiniteIntersections (α : Type*) (F : Set (Set α)) : Prop :=
  ∀ A B, A ∈ F → B ∈ F → A ≠ B → Set.Finite (A ∩ B)

-- State the theorem
theorem exists_countable_set_with_uncountable_family :
  ∃ (Y : Type*) (F : Set (Set Y)),
    IsCountable Y ∧
    ¬ Countable F ∧
    HasFiniteIntersections Y F := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_countable_set_with_uncountable_family_l555_55585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolas_intersection_l555_55571

-- Define the two parabolas
def f (x : ℝ) : ℝ := 3 * x^2 - 15 * x - 15
def g (x : ℝ) : ℝ := x^2 - 5 * x + 10

-- Define the intersection points
def x₁ : ℝ := 0.67
def x₂ : ℝ := 9.33

-- Theorem statement
theorem parabolas_intersection :
  ∃ (ε : ℝ), ε > 0 ∧ 
  (∀ x : ℝ, f x = g x → (|x - x₁| < ε ∨ |x - x₂| < ε)) ∧
  (|f x₁ - g x₁| < 0.01) ∧ (|f x₂ - g x₂| < 0.01) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolas_intersection_l555_55571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l555_55544

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = -4 * x

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x + 1/2)^2 + (y - Real.sqrt 2)^2 = 9/4

-- Define the left focus F₁
def F₁ : ℝ × ℝ := (-1, 0)

-- Define the right focus F₂
def F₂ : ℝ × ℝ := (1, 0)

-- Define the dot product of vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem ellipse_properties :
  -- 1. The equation of the ellipse is correct
  (∀ x y, ellipse x y ↔ x^2 / 2 + y^2 = 1) ∧
  -- 2. The equation of the circle is correct
  (∀ x y, my_circle x y ↔ (x + 1/2)^2 + (y - Real.sqrt 2)^2 = 9/4) ∧
  -- 3. The maximum and minimum values of F₂A ⋅ F₂B are correct
  (∃ A B : ℝ × ℝ, ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧
    dot_product (A.1 - F₂.1, A.2 - F₂.2) (B.1 - F₂.1, B.2 - F₂.2) = 7/2) ∧
  (∃ A B : ℝ × ℝ, ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧
    dot_product (A.1 - F₂.1, A.2 - F₂.2) (B.1 - F₂.1, B.2 - F₂.2) = -1) ∧
  (∀ A B : ℝ × ℝ, ellipse A.1 A.2 → ellipse B.1 B.2 →
    dot_product (A.1 - F₂.1, A.2 - F₂.2) (B.1 - F₂.1, B.2 - F₂.2) ≤ 7/2 ∧
    dot_product (A.1 - F₂.1, A.2 - F₂.2) (B.1 - F₂.1, B.2 - F₂.2) ≥ -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l555_55544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_b_coordinates_l555_55563

noncomputable def vector_a : ℝ × ℝ := (-3, 4)

def opposite_direction (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k < 0 ∧ w = (k * v.1, k * v.2)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

theorem vector_b_coordinates :
  ∃ (b : ℝ × ℝ), opposite_direction vector_a b ∧ magnitude b = 10 → b = (6, -8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_b_coordinates_l555_55563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_plays_required_l555_55596

/-- Represents a theater troupe with actors and plays. -/
structure TheaterTroupe where
  actors : Finset ℕ
  plays : Finset ℕ
  actor_in_play : ℕ → ℕ → Bool

/-- The number of actors in the troupe. -/
def num_actors (t : TheaterTroupe) : ℕ := t.actors.card

/-- The number of plays produced by the troupe. -/
def num_plays (t : TheaterTroupe) : ℕ := t.plays.card

/-- Predicate to check if two actors have performed together in at least one play. -/
def performed_together (t : TheaterTroupe) (a b : ℕ) : Prop :=
  ∃ p ∈ t.plays, t.actor_in_play a p ∧ t.actor_in_play b p

/-- Predicate to check if a play has no more than 30 actors. -/
def play_size_limit (t : TheaterTroupe) (p : ℕ) : Prop :=
  (t.actors.filter (λ a => t.actor_in_play a p)).card ≤ 30

/-- The main theorem stating the minimum number of plays required. -/
theorem min_plays_required (t : TheaterTroupe) :
  num_actors t = 60 ∧
  (∀ a b, a ∈ t.actors → b ∈ t.actors → a ≠ b → performed_together t a b) ∧
  (∀ p, p ∈ t.plays → play_size_limit t p) →
  num_plays t ≥ 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_plays_required_l555_55596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_sum_less_than_25_l555_55522

/-- A triangle with perimeter 100 cm and area 100 cm² -/
structure Triangle where
  perimeter : ℝ
  area : ℝ
  perimeter_eq : perimeter = 100
  area_eq : area = 100

/-- Parallel lines drawn 1 cm from each side of the triangle -/
structure ParallelLines where
  distance : ℝ
  distance_eq : distance = 1

/-- Parallelograms formed by the parallel lines -/
structure Parallelograms where
  count : ℕ
  count_eq : count = 3

/-- Sum of parallelogram areas function (placeholder) -/
def sum_of_parallelogram_areas (t : Triangle) (pl : ParallelLines) (p : Parallelograms) : ℝ :=
  sorry

/-- Theorem: The sum of areas of parallelograms is less than 25 cm² -/
theorem parallelogram_area_sum_less_than_25 
  (t : Triangle) (pl : ParallelLines) (p : Parallelograms) : 
  ∃ (area_sum : ℝ), area_sum < 25 ∧ 
  (area_sum = sum_of_parallelogram_areas t pl p) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_sum_less_than_25_l555_55522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_no_car_percent_l555_55555

noncomputable section

structure Ship where
  round_trip_percent : ℝ
  car_percent : ℝ

noncomputable def Ship.no_car_percent (s : Ship) : ℝ :=
  s.round_trip_percent * (1 - s.car_percent / 100)

def ship_A : Ship := ⟨30, 25⟩
def ship_B : Ship := ⟨50, 15⟩
def ship_C : Ship := ⟨20, 35⟩

theorem highest_no_car_percent :
  (ship_B.no_car_percent > ship_A.no_car_percent) ∧
  (ship_B.no_car_percent > ship_C.no_car_percent) ∧
  (ship_B.no_car_percent = 42.5) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_no_car_percent_l555_55555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_inscribed_circle_l555_55553

/-- A convex n-gon -/
structure ConvexNGon (n : ℕ) where
  -- Assume n ≥ 3
  n_ge_three : n ≥ 3
  -- Vertices of the n-gon
  vertices : Fin n → ℝ × ℝ
  -- Convexity condition (simplified)
  convex : ∀ i j k : Fin n, i ≠ j → j ≠ k → i ≠ k → 
    (vertices j).1 - (vertices i).1 * (vertices k).2 - (vertices j).2 ≥
    (vertices j).2 - (vertices i).2 * (vertices k).1 - (vertices j).1

/-- The perimeter of an n-gon -/
noncomputable def perimeter (n : ℕ) (ngon : ConvexNGon n) : ℝ :=
  sorry

/-- The area of an n-gon -/
noncomputable def area (n : ℕ) (ngon : ConvexNGon n) : ℝ :=
  sorry

/-- Predicate to check if a circle can be inscribed in an n-gon -/
def has_inscribed_circle (n : ℕ) (ngon : ConvexNGon n) : Prop :=
  sorry

/-- Calculate the angle at a vertex of an n-gon -/
noncomputable def angle_at (n : ℕ) (ngon : ConvexNGon n) (i : Fin n) : ℝ :=
  sorry

/-- The main theorem -/
theorem max_area_inscribed_circle (n : ℕ) (angles : Fin n → ℝ) (p : ℝ) :
  ∀ (ngon1 ngon2 : ConvexNGon n),
    (∀ i : Fin n, angle_at n ngon1 i = angles i) →
    (∀ i : Fin n, angle_at n ngon2 i = angles i) →
    perimeter n ngon1 = p →
    perimeter n ngon2 = p →
    has_inscribed_circle n ngon1 →
    area n ngon1 ≥ area n ngon2 :=
  by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_inscribed_circle_l555_55553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_tilt_height_l555_55531

-- Define the rectangle dimensions
def width : ℝ := 100
def length : ℝ := 150  -- Changed from 'height' to 'length' to avoid naming conflict

-- Define the height of the lower corner
def lower_corner_height : ℝ := 20

-- Theorem statement
theorem rectangle_tilt_height :
  let diagonal := Real.sqrt (width^2 + length^2)
  let angle := Real.arcsin (lower_corner_height / width)
  let upper_corner_height := length * Real.sin angle + lower_corner_height
  ⌊upper_corner_height⌋ = 167 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_tilt_height_l555_55531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l555_55541

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x
def g (a : ℝ) (x : ℝ) : ℝ := -(x + 1)^2 + a

-- State the theorem
theorem min_value_of_a (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, f x₁ ≤ g a x₂) → a ≥ -(Real.exp 1)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l555_55541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_subset_l555_55577

theorem divisibility_in_subset (S : Finset ℕ) (h1 : ∀ x ∈ S, x ∈ Finset.range 201) (h2 : Finset.card S = 101) :
  ∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a ∣ b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_subset_l555_55577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_equals_sqrt5_over_2_l555_55506

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the complex number z
noncomputable def z : ℂ := (2 + i) / ((1 + i)^2)

-- State the theorem
theorem abs_z_equals_sqrt5_over_2 : Complex.abs z = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_equals_sqrt5_over_2_l555_55506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_intervals_f_max_min_on_interval_l555_55561

noncomputable section

open Real

-- Define the function f(x)
def f (x : ℝ) : ℝ := 4 * sin x * cos (x + π/6)

-- Theorem for the decreasing intervals
theorem f_decreasing_intervals (k : ℤ) :
  ∀ x ∈ Set.Icc (k * π + π/6) (k * π + 2*π/3),
    ∀ y ∈ Set.Icc (k * π + π/6) (k * π + 2*π/3),
      x < y → f y < f x :=
by sorry

-- Theorem for the maximum and minimum values on [0, π/2]
theorem f_max_min_on_interval :
  (∀ x ∈ Set.Icc 0 (π/2), f x ≤ 1) ∧
  (∃ x ∈ Set.Icc 0 (π/2), f x = 1) ∧
  (∀ x ∈ Set.Icc 0 (π/2), f x ≥ -2) ∧
  (∃ x ∈ Set.Icc 0 (π/2), f x = -2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_intervals_f_max_min_on_interval_l555_55561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_50_between_consecutive_integers_l555_55598

theorem sqrt_50_between_consecutive_integers : 
  ∃ (n : ℕ), n > 0 ∧ n^2 < 50 ∧ 50 < (n+1)^2 ∧ n * (n + 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_50_between_consecutive_integers_l555_55598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l555_55507

theorem divisibility_condition (k n : ℕ) (hk : k > 0) (hn : n > 0) :
  (∃ m : ℕ, 2^n - 1 = m * 3^k) ↔ (∃ p : ℕ, p > 0 ∧ n = 2 * 3^(k-1) * p) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l555_55507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_sphere_ratio_l555_55580

theorem cylinder_sphere_ratio (r R : ℝ) (h : r > 0) :
  2 * π * r * (4 * r) = 4 * π * R^2 → R / r = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_sphere_ratio_l555_55580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perp_distance_product_bound_l555_55572

/-- Triangle ABC with point P inside --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  P : ℝ × ℝ
  inside_P : Prop  -- We'll use a proposition instead of a function

/-- Perpendicular distance from a point to a line --/
noncomputable def perpDistance (P : ℝ × ℝ) (L : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := sorry

/-- Area of a triangle --/
noncomputable def triangleArea (A B C : ℝ × ℝ) : ℝ := sorry

/-- Length of a line segment --/
noncomputable def lineLength (A B : ℝ × ℝ) : ℝ := sorry

/-- Theorem: Upper bound for the product of perpendicular distances --/
theorem perp_distance_product_bound (t : Triangle) : 
  let p := (perpDistance t.P (t.B, t.C)) * (perpDistance t.P (t.C, t.A)) * (perpDistance t.P (t.A, t.B))
  let S := triangleArea t.A t.B t.C
  let a := lineLength t.B t.C
  let b := lineLength t.C t.A
  let c := lineLength t.A t.B
  p ≤ (8 * S^3) / (27 * a * b * c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perp_distance_product_bound_l555_55572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_range_l555_55594

-- Define a regular n-hedron
structure RegularNHedron where
  n : ℕ
  n_ge_3 : n ≥ 3

-- Define the dihedral angle between two adjacent faces
noncomputable def DihedralAngle (nh : RegularNHedron) : ℝ := sorry

-- Theorem stating the range of the dihedral angle
theorem dihedral_angle_range (nh : RegularNHedron) :
  (nh.n - 2 : ℝ) / nh.n * Real.pi < DihedralAngle nh ∧ DihedralAngle nh < Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_range_l555_55594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_count_l555_55575

/-- Circle represented by its equation coefficients -/
structure Circle where
  a : ℝ  -- coefficient of x²
  b : ℝ  -- coefficient of y²
  c : ℝ  -- coefficient of x
  d : ℝ  -- coefficient of y
  e : ℝ  -- constant term

/-- Calculate the center coordinates of a circle -/
noncomputable def Circle.center (c : Circle) : ℝ × ℝ :=
  (- c.c / (2 * c.a), - c.d / (2 * c.b))

/-- Calculate the radius of a circle -/
noncomputable def Circle.radius (c : Circle) : ℝ :=
  Real.sqrt ((c.c ^ 2 + c.d ^ 2) / (4 * c.a * c.b) - c.e / (c.a * c.b))

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Count the number of tangent lines between two circles -/
noncomputable def count_tangent_lines (c1 c2 : Circle) : ℕ :=
  if distance c1.center c2.center == c1.radius + c2.radius then 3 else 0

/-- Main theorem: The number of tangent lines shared by the given circles is 3 -/
theorem tangent_lines_count :
  let c1 : Circle := { a := 1, b := 1, c := 4, d := -4, e := 7 }
  let c2 : Circle := { a := 1, b := 1, c := -4, d := -10, e := 13 }
  count_tangent_lines c1 c2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_count_l555_55575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sale_price_l555_55532

/-- Calculates the final selling price of a car after two sales with given percentages of loss and gain. -/
noncomputable def finalSellingPrice (originalPrice : ℝ) (lossPercentage : ℝ) (gainPercentage : ℝ) : ℝ :=
  let firstSalePrice := originalPrice * (1 - lossPercentage / 100)
  let finalSalePrice := firstSalePrice * (1 + gainPercentage / 100)
  finalSalePrice

/-- Theorem stating that given the specific conditions, the final selling price is 54000. -/
theorem car_sale_price : 
  finalSellingPrice 52325.58 14 20 = 54000 := by
  -- Expand the definition of finalSellingPrice
  unfold finalSellingPrice
  -- Perform the calculation
  norm_num
  -- Close the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sale_price_l555_55532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_after_transform_l555_55569

/-- Represents a pyramid with rectangular base -/
structure RectangularPyramid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculate the volume of a rectangular pyramid -/
noncomputable def volume (p : RectangularPyramid) : ℝ := (1/3) * p.length * p.width * p.height

/-- Transform a pyramid by tripling length, quadrupling width, and doubling height -/
def transform (p : RectangularPyramid) : RectangularPyramid :=
  { length := 3 * p.length
    width := 4 * p.width
    height := 2 * p.height }

theorem volume_after_transform (p : RectangularPyramid) :
  volume (transform p) = 24 * volume p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_after_transform_l555_55569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l555_55501

/-- The function we're analyzing -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi/4) * Real.cos (Real.pi/4 - x)

/-- Definition of a symmetry axis for a function -/
def is_symmetry_axis (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

/-- Theorem stating that x = π/4 is a symmetry axis of the function f -/
theorem symmetry_axis_of_f : is_symmetry_axis f (Real.pi/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l555_55501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terms_added_in_sum_l555_55586

theorem terms_added_in_sum (k : ℕ) : 
  (Finset.range (2^(k+1) - 1)).card - (Finset.range (2^k - 1)).card = 2^k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terms_added_in_sum_l555_55586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_difference_equation_l555_55513

/-- The price difference between commodities X and Y at time t with inflation rate r -/
noncomputable def price_difference (r : ℝ) (t : ℕ) : ℝ :=
  4.20 * (1 + (2*r + 10)/100)^(t - 2001) - 4.40 * (1 + (r + 15)/100)^(t - 2001)

/-- Theorem stating the price difference equation -/
theorem price_difference_equation (r : ℝ) :
  ∃ t : ℕ, price_difference r t = 0.90 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_difference_equation_l555_55513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_sum_zero_l555_55551

-- Define the function f
noncomputable def f (x b : ℝ) : ℝ := x^3 - Real.sin x + b + 2

-- State the theorem
theorem odd_function_sum_zero 
  (a b : ℝ) 
  (h_odd : ∀ x, f x b = -f (-x) b) 
  (h_domain : Set.Icc (a - 4) (2*a - 2) = Set.Icc (-a + 2) (a - 2)) :
  f a b + f b b = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_sum_zero_l555_55551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_perpendicular_integral_l555_55584

/-- The integral of the squared perpendicular length from origin to tangent of an ellipse -/
theorem ellipse_perpendicular_integral (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let r : ℝ → ℝ := λ θ => 
    Real.sqrt ((b^2 * (Real.tan θ)^2 + a^2) / ((Real.cos θ)^2)⁻¹)
  ∫ θ in (0)..(2*Real.pi), r θ^2 = Real.pi * (a^2 + b^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_perpendicular_integral_l555_55584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_problem_l555_55508

noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / 100) ^ time

noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem interest_problem : 
  let principalSI := 3225
  let rateSI := 8
  let principalCI := 8000
  let rateCI := 15
  let timeCI := 2
  ∃ t : ℝ, 
    simpleInterest principalSI rateSI t = 
      (compoundInterest principalCI rateCI timeCI - principalCI) / 2 ∧
    t = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_problem_l555_55508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_line_plane_relationship_l555_55527

-- Define the types for point, line, and plane
variable (Point Line Plane : Type)

-- Define the relationships
variable (on_line : Point → Line → Prop)
variable (in_plane : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (mem : Point → Line → Prop)

-- Define the specific entities
variable (B : Point) (b : Line) (β : Plane)

-- State the theorem
theorem point_line_plane_relationship 
  (h1 : on_line B b) 
  (h2 : in_plane b β) :
  mem B b ∧ subset b β := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_line_plane_relationship_l555_55527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l555_55573

noncomputable def f (x : ℝ) := Real.cos x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x + 1

theorem f_properties :
  -- Smallest positive period is π
  (∀ x, f (x + π) = f x) ∧
  (∀ p, 0 < p → p < π → ∃ x, f (x + p) ≠ f x) ∧
  -- Maximum value is 5/2
  (∀ x, f x ≤ 5/2) ∧
  (∃ x, f x = 5/2) ∧
  -- Minimum value is 1/2
  (∀ x, 1/2 ≤ f x) ∧
  (∃ x, f x = 1/2) ∧
  -- Monotonicity on (0, π)
  (∀ x y, 0 < x → x < y → y ≤ π/6 → f x ≤ f y) ∧
  (∀ x y, π/6 ≤ x → x < y → y ≤ 2*π/3 → f x ≥ f y) ∧
  (∀ x y, 2*π/3 ≤ x → x < y → y < π → f x ≤ f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l555_55573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distance_to_x_axis_l555_55599

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the foci
def F1 : ℝ × ℝ := (-5, 0)
def F2 : ℝ × ℝ := (5, 0)

-- Define a point on the hyperbola
variable (M : ℝ × ℝ)

-- Define the perpendicularity condition
def perpendicular (M F1 F2 : ℝ × ℝ) : Prop :=
  (M.1 - F1.1) * (M.1 - F2.1) + (M.2 - F1.2) * (M.2 - F2.2) = 0

-- Define the distance from a point to the x-axis
def distance_to_x_axis (p : ℝ × ℝ) : ℝ := |p.2|

-- State the theorem
theorem hyperbola_distance_to_x_axis :
  hyperbola M.1 M.2 →
  perpendicular M F1 F2 →
  distance_to_x_axis M = 16/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distance_to_x_axis_l555_55599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangements_correct_l555_55503

def n : ℕ := 10
def k : ℕ := 4

/-- The number of ways to arrange n people in a row where k specific people are not in k consecutive seats -/
def seating_arrangements (n k : ℕ) : ℕ :=
  n.factorial - (n - k + 1).factorial * k.factorial

theorem seating_arrangements_correct (n k : ℕ) (h1 : n ≥ k) (h2 : k ≥ 2) :
  seating_arrangements n k =
    (n.factorial - (n - k + 1).factorial * k.factorial) :=
by
  -- The proof goes here
  sorry

#eval seating_arrangements 10 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangements_correct_l555_55503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_and_projection_l555_55574

/-- Given vectors BA and BC, prove the magnitude of angle BAC and the projection of BA onto AC -/
theorem vector_angle_and_projection (BA BC : ℝ × ℝ) : 
  BA = (1, Real.sqrt 3) → BC = (2, 0) → 
  ∃ (angle : ℝ) (proj : ℝ),
    angle = π / 3 ∧ 
    proj = -1 ∧
    angle = Real.arccos ((BA.1 * (-BA.1 + BC.1) + BA.2 * (-BA.2 + BC.2)) / 
      (Real.sqrt (BA.1^2 + BA.2^2) * Real.sqrt ((-BA.1 + BC.1)^2 + (-BA.2 + BC.2)^2))) ∧
    proj = (BA.1 * (-BA.1 + BC.1) + BA.2 * (-BA.2 + BC.2)) / 
      Real.sqrt ((-BA.1 + BC.1)^2 + (-BA.2 + BC.2)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_and_projection_l555_55574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_digit_divisibility_l555_55583

/-- Represents a two-digit number -/
def TwoDigitNumber : Type := { n : ℕ // n ≥ 10 ∧ n ≤ 99 }

/-- Converts a two-digit number to a six-digit number by repeating it three times -/
def repeatThreeTimes (n : TwoDigitNumber) : ℕ :=
  100000 * n.val + 1000 * n.val + n.val

theorem six_digit_divisibility (n : TwoDigitNumber) :
  (repeatThreeTimes n) % 10101 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_digit_divisibility_l555_55583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_a_n_l555_55591

-- Define the sequence a_n
def a (n : ℕ) : ℚ := 1 - 2 / (2 * (n : ℚ) - 17)

-- Define the product of the first n terms
def T (n : ℕ) : ℚ := 1 - (2 / 15) * (n : ℚ)

-- State the theorem
theorem sum_of_max_min_a_n : 
  ∃ (max min : ℚ), (∀ n, a n ≤ max ∧ min ≤ a n) ∧ max + min = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_a_n_l555_55591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_symmetry_cos_square_plus_sin_min_l555_55500

-- Define the tangent function
noncomputable def tan (x : ℝ) := Real.tan x

-- Statement 1: Symmetry of tan x about (kπ/2, 0)
theorem tan_symmetry (k : ℤ) (x : ℝ) : 
  tan (k * π / 2 + x) = -tan (k * π / 2 - x) := by sorry

-- Statement 2: Minimum value of cos²x + sin x
theorem cos_square_plus_sin_min :
  ∃ x : ℝ, ∀ y : ℝ, Real.cos y ^ 2 + Real.sin y ≥ Real.cos x ^ 2 + Real.sin x ∧ 
  Real.cos x ^ 2 + Real.sin x = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_symmetry_cos_square_plus_sin_min_l555_55500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_water_heated_to_boiling_l555_55519

/-- Represents the heat produced by burning fuel over time -/
noncomputable def heatProduced (t : ℕ) : ℝ :=
  480 * (3/4)^t

/-- Calculates the total heat produced over an infinite time -/
noncomputable def totalHeatProduced : ℝ :=
  480 / (1 - 3/4)

/-- Calculates the heat required to boil water -/
noncomputable def heatRequired (m : ℝ) : ℝ :=
  m * 4.2 * (100 - 20)

/-- Theorem: The maximum integer number of liters of water that can be heated to boiling is 5 -/
theorem max_water_heated_to_boiling :
  ∃ (n : ℕ), n = 5 ∧ 
    (∀ m : ℕ, m > n → heatRequired (m : ℝ) > totalHeatProduced) ∧
    (heatRequired (n : ℝ) ≤ totalHeatProduced) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_water_heated_to_boiling_l555_55519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_net_gain_percent_is_approximately_14_61_l555_55518

/-- Represents the financial details of an item --/
structure Item where
  cost : ℚ
  discount_rate : ℚ
  tax_rate : ℚ
  sell_price : ℚ

/-- Calculates the actual cost price after discount and tax --/
def actual_cost (item : Item) : ℚ :=
  item.cost * (1 - item.discount_rate) * (1 + item.tax_rate)

/-- Calculates the profit for an item --/
def profit (item : Item) : ℚ :=
  item.sell_price - actual_cost item

/-- Calculates the overall net gain percent --/
def net_gain_percent (items : List Item) : ℚ :=
  let total_profit := (items.map profit).sum
  let total_cost := (items.map actual_cost).sum
  (total_profit / total_cost) * 100

/-- The main theorem stating the overall net gain percent --/
theorem overall_net_gain_percent_is_approximately_14_61 :
  let items : List Item := [
    { cost := 100, discount_rate := 0, tax_rate := 0, sell_price := 120 },
    { cost := 150, discount_rate := 1/10, tax_rate := 0, sell_price := 180 },
    { cost := 200, discount_rate := 0, tax_rate := 1/20, sell_price := 210 }
  ]
  abs (net_gain_percent items - 14621/1000) < 1/100 := by
  sorry

#eval net_gain_percent [
  { cost := 100, discount_rate := 0, tax_rate := 0, sell_price := 120 },
  { cost := 150, discount_rate := 1/10, tax_rate := 0, sell_price := 180 },
  { cost := 200, discount_rate := 0, tax_rate := 1/20, sell_price := 210 }
]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_net_gain_percent_is_approximately_14_61_l555_55518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_four_lt_b_seven_l555_55562

def b (n : ℕ) (α : ℕ → ℕ+) : ℚ :=
  match n with
  | 0 => 0
  | 1 => 1 + 1 / (α 1).val
  | k + 1 => 1 + 1 / (b k α + 1 / (α (k + 1)).val)

theorem b_four_lt_b_seven (α : ℕ → ℕ+) : b 4 α < b 7 α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_four_lt_b_seven_l555_55562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l555_55521

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given conditions
noncomputable def given_triangle : Triangle where
  A := Real.arccos (4/5)
  B := Real.arccos (-Real.sqrt 2/10)
  C := Real.pi/4  -- This is what we want to prove
  a := 3 * Real.sqrt 2  -- This is derived in the solution, but not given
  b := Real.sqrt ((3 * Real.sqrt 2)^2 + 5^2 - 2*(3 * Real.sqrt 2)*5*(-Real.sqrt 2/10))  -- Using law of cosines
  c := 5

-- Theorem to prove
theorem triangle_properties (t : Triangle) (h1 : t = given_triangle) :
  t.C = Real.pi/4 ∧ 
  (1/2 * t.a * t.c * Real.sin t.B) = 21/2 := by
  sorry

-- Note: The actual proof is omitted and replaced with 'sorry'

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l555_55521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_plus_pi_half_l555_55597

theorem sin_double_angle_plus_pi_half (θ : ℝ) (h : Real.cos θ = -1/3) :
  Real.sin (2 * θ + Real.pi/2) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_plus_pi_half_l555_55597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_squared_l555_55533

/-- Two circles in the xy-plane:
    Circle 1 centered at (1, 3) with radius 5
    Circle 2 centered at (1, -2) with radius 6 -/
def circle1 : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 3)^2 = 25}
def circle2 : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 + 2)^2 = 36}

/-- The intersection points of the two circles -/
def intersection : Set (ℝ × ℝ) := circle1 ∩ circle2

/-- The squared distance between two points -/
def squaredDistance (p q : ℝ × ℝ) : ℝ := (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- The theorem stating that the squared distance between 
    the intersection points of the circles is 92.16 -/
theorem intersection_distance_squared :
  ∀ (A B : ℝ × ℝ), A ∈ intersection → B ∈ intersection → A ≠ B → squaredDistance A B = 92.16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_squared_l555_55533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curvature_cubic_exists_constant_curvature_curvature_parabola_bound_curvature_exp_bound_l555_55530

-- Define the curvature function φ
noncomputable def φ (f : ℝ → ℝ) (x₁ y₁ x₂ y₂ kₐ k_B : ℝ) : ℝ :=
  |kₐ - k_B| / Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Statement 1
theorem curvature_cubic (x₁ x₂ : ℝ) (h : x₁ = 1 ∧ x₂ = -1) :
  φ (λ x => x^3) x₁ (x₁^3) x₂ (x₂^3) (3*x₁^2) (3*x₂^2) = 0 := by sorry

-- Statement 2
theorem exists_constant_curvature :
  ∃ f : ℝ → ℝ, ∀ x₁ y₁ x₂ y₂ kₐ k_B : ℝ,
    x₁ ≠ x₂ → φ f x₁ y₁ x₂ y₂ kₐ k_B = φ f 0 1 1 2 0 0 := by sorry

-- Statement 3
theorem curvature_parabola_bound (x₁ x₂ : ℝ) (h : x₁ ≠ x₂) :
  φ (λ x => x^2 + 1) x₁ (x₁^2 + 1) x₂ (x₂^2 + 1) (2*x₁) (2*x₂) ≤ 2 := by sorry

-- Statement 4
theorem curvature_exp_bound (x₁ x₂ : ℝ) (h : x₁ ≠ x₂) :
  φ Real.exp x₁ (Real.exp x₁) x₂ (Real.exp x₂) (Real.exp x₁) (Real.exp x₂) < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curvature_cubic_exists_constant_curvature_curvature_parabola_bound_curvature_exp_bound_l555_55530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_proof_l555_55560

noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 27 + y^2 / 36 = 1

noncomputable def point_on_hyperbola : ℝ × ℝ :=
  (Real.sqrt 15, 4)

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 4 - x^2 / 5 = 1

theorem hyperbola_equation_proof :
  ∀ (x y : ℝ),
  (∃ (a b : ℝ), ellipse_equation a b ∧ 
    (x - a)^2 + (y - b)^2 = (x + a)^2 + (y + b)^2) →
  hyperbola_equation point_on_hyperbola.1 point_on_hyperbola.2 →
  hyperbola_equation x y :=
by
  sorry

#check hyperbola_equation_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_proof_l555_55560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_theorem_l555_55595

noncomputable def rational_man_path (t : ℝ) : ℝ × ℝ := (Real.cos t, Real.sin t)

noncomputable def mathematic_man_path (t : ℝ) : ℝ × ℝ := (3 + 2 * Real.cos (t / 2), 4 * Real.sin (t / 2))

noncomputable def distance_function (t τ : ℝ) : ℝ :=
  Real.sqrt ((3 + 2 * Real.cos τ - Real.cos t)^2 + (4 * Real.sin τ - Real.sin t)^2)

theorem minimum_distance_theorem :
  ∃ (min_dist : ℝ),
    (∀ (A : ℝ × ℝ) (M : ℝ × ℝ),
      (A.1^2 + A.2^2 = 1) →
      ((M.1 - 3)^2 / 4 + M.2^2 / 16 = 1) →
      Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) ≥ min_dist) ∧
    (∃ (t τ : ℝ), t ∈ Set.Icc 0 (2 * Real.pi) ∧ τ ∈ Set.Icc 0 (2 * Real.pi) ∧
      distance_function t τ = min_dist) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_theorem_l555_55595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_l555_55554

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x - 2 * (Real.sin x)^2 + 1

theorem max_area_triangle (a b c A B C : ℝ) : 
  f A = 2 * Real.sin A * Real.cos A - 2 * (Real.sin A)^2 + 1 →
  a = Real.sqrt 3 →
  0 < A → A < Real.pi / 2 →
  f (A + Real.pi / 8) = Real.sqrt 2 / 3 →
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A →
  (1/2) * b * c * Real.sin A ≤ 3 * (Real.sqrt 3 + Real.sqrt 2) / 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_l555_55554
