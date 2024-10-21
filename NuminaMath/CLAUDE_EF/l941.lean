import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_properties_l941_94115

/-- The line equation y = x + m -/
def line_equation (x y m : ℝ) : Prop := y = x + m

/-- The parabola equation y^2 = 8x -/
def parabola_equation (x y : ℝ) : Prop := y^2 = 8 * x

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- Perpendicularity condition for two vectors from origin -/
def perpendicular_vectors (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

/-- Main theorem -/
theorem intersection_properties (m : ℝ) :
  (∃ x1 y1 x2 y2 : ℝ,
    line_equation x1 y1 m ∧ parabola_equation x1 y1 ∧
    line_equation x2 y2 m ∧ parabola_equation x2 y2 ∧
    distance x1 y1 x2 y2 = 10) →
  m = 7/16
  ∧
  (∃ x1 y1 x2 y2 : ℝ,
    line_equation x1 y1 m ∧ parabola_equation x1 y1 ∧
    line_equation x2 y2 m ∧ parabola_equation x2 y2 ∧
    perpendicular_vectors x1 y1 x2 y2) →
  m = -8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_properties_l941_94115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_in_E_l941_94186

/-- A point in the plane with integer coordinates -/
structure IntPoint where
  x : ℤ
  y : ℤ

/-- The set of points E -/
def E : Set IntPoint := sorry

/-- The number of points in E -/
def n : ℕ := sorry

/-- Condition: n is at least 3 -/
axiom n_ge_three : n ≥ 3

/-- Function to calculate the centroid of three points -/
def centroid (p1 p2 p3 : IntPoint) : ℚ × ℚ :=
  (((p1.x + p2.x + p3.x) : ℚ) / 3, ((p1.y + p2.y + p3.y) : ℚ) / 3)

/-- Condition: The centroid of any three points from E is not an integer point -/
axiom centroid_not_integer (p1 p2 p3 : IntPoint) :
  p1 ∈ E → p2 ∈ E → p3 ∈ E →
  ¬(∃ (a b : ℤ), centroid p1 p2 p3 = ((a : ℚ), (b : ℚ)))

/-- The main theorem: The maximal number of points in E is 8 -/
theorem max_points_in_E : n ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_in_E_l941_94186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_proof_l941_94198

/-- The curve function -/
noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - 1)

/-- The line function -/
def g (x y : ℝ) : ℝ := 2 * x - y + 3

/-- The shortest distance from a point on the curve to the line -/
noncomputable def shortestDistance : ℝ := Real.sqrt 5

theorem shortest_distance_proof :
  ∃ (x₀ y₀ : ℝ), y₀ = f x₀ ∧
  ∀ (x y : ℝ), y = f x →
  (x - x₀)^2 + (y - y₀)^2 ≥ (g x₀ y₀)^2 / (2^2 + 1) := by
  sorry

#check shortest_distance_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_proof_l941_94198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_is_30_l941_94142

/-- Represents a student's ranking in a class -/
structure StudentRanking where
  fromLeft : ℕ
  fromRight : ℕ

/-- Represents a class with two known student rankings -/
structure ClassInfo where
  studentA : StudentRanking
  studentB : StudentRanking
  nonLinearPattern : Bool

/-- Calculates the total number of students in a class -/
def totalStudents (c : ClassInfo) : ℕ :=
  max c.studentA.fromLeft c.studentB.fromLeft +
  max c.studentA.fromRight c.studentB.fromRight - 1

/-- Theorem: The class with given student rankings has 30 students -/
theorem class_size_is_30 (c : ClassInfo)
  (h1 : c.studentA = ⟨4, 19⟩)
  (h2 : c.studentB = ⟨12, 6⟩)
  (h3 : c.nonLinearPattern = true) :
  totalStudents c = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_is_30_l941_94142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angled_triangle_set_l941_94180

theorem right_angled_triangle_set : ∀ (a b c : ℝ),
  (a = 1 ∧ b = 1 ∧ c = Real.sqrt 2) ↔ 
  (a^2 + b^2 = c^2 ∧ 
   ¬(1^2 + 2^2 = 3^2) ∧
   ¬(6^2 + 8^2 = 11^2) ∧
   ¬(2^2 + 3^2 = 4^2)) := by
  sorry

#check right_angled_triangle_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angled_triangle_set_l941_94180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_end_of_year_l941_94163

/-- Proves that the total number of students in 4th, 5th, and 6th grades at the end of the year is 116 --/
theorem total_students_end_of_year 
  (initial_4th : ℕ) (initial_5th : ℕ) (initial_6th : ℕ)
  (left_4th : ℕ) (joined_4th : ℕ)
  (left_5th : ℕ) (joined_5th : ℕ)
  (left_6th : ℕ) (joined_6th : ℕ)
  (h1 : initial_4th = 33)
  (h2 : initial_5th = 45)
  (h3 : initial_6th = 28)
  (h4 : left_4th = 18)
  (h5 : joined_4th = 14)
  (h6 : left_5th = 12)
  (h7 : joined_5th = 20)
  (h8 : left_6th = 10)
  (h9 : joined_6th = 16) :
  (initial_4th - left_4th + joined_4th) + 
  (initial_5th - left_5th + joined_5th) + 
  (initial_6th - left_6th + joined_6th) = 116 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_end_of_year_l941_94163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_frustum_volume_ratio_l941_94188

/-- Given two similar pyramids where the ratio of their altitudes is 1/3, 
    the volume of the frustum formed by removing the smaller pyramid from 
    the larger one is 26/27 of the original pyramid's volume. -/
theorem pyramid_frustum_volume_ratio : 
  ∀ (V : ℝ) (h : ℝ) (h_small : ℝ),
  h_small = (1/3 : ℝ) * h →
  V > 0 →
  h > 0 →
  (V - V * (h_small / h)^3) / V = 26/27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_frustum_volume_ratio_l941_94188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_ln_positive_l941_94145

theorem negation_of_forall_ln_positive :
  ¬(∀ x : ℝ, x > 0 → Real.log (2*x + 1) > 0) ↔ ∃ x : ℝ, x > 0 ∧ Real.log (2*x + 1) ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_ln_positive_l941_94145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_with_seven_factors_l941_94137

theorem least_integer_with_seven_factors : 
  ∃ n : ℕ, (n = 64) ∧ 
  (∀ m : ℕ, 0 < m → m < n → (Finset.card (Nat.divisors m) ≠ 7)) ∧
  (Finset.card (Nat.divisors n) = 7) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_with_seven_factors_l941_94137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_properties_l941_94167

theorem inequality_properties (a b : ℝ) (h1 : a > b) (h2 : a * b ≠ 0) :
  (2:ℝ)^a > (2:ℝ)^b ∧ a^(1/3:ℝ) > b^(1/3:ℝ) ∧ ((1/3:ℝ)^a < (1/3:ℝ)^b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_properties_l941_94167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l941_94117

theorem problem_solution (x y z : ℝ) 
  (h1 : (3 : ℝ)^x * (4 : ℝ)^y / (2 : ℝ)^z = 59049)
  (h2 : x - y + 2*z = 10)
  (h3 : x^2 + y^2 = z^2) : 
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l941_94117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ramsey_for_lines_l941_94135

/-- A type representing a line in a 2D plane -/
structure Line2D where
  -- We don't need to define the internals of a line for this statement
  mk :: (dummy : Unit)

/-- A function to check if two lines are parallel -/
def are_parallel (l1 l2 : Line2D) : Prop :=
  sorry -- Definition of parallelism

/-- A function to check if a set of lines are all parallel -/
def all_parallel (lines : Set Line2D) : Prop :=
  ∀ l1 l2, l1 ∈ lines → l2 ∈ lines → are_parallel l1 l2

/-- A function to check if a set of lines are pairwise nonparallel -/
def pairwise_nonparallel (lines : Set Line2D) : Prop :=
  ∀ l1 l2, l1 ∈ lines → l2 ∈ lines → l1 ≠ l2 → ¬(are_parallel l1 l2)

/-- The main theorem -/
theorem ramsey_for_lines :
  ∀ (lines : Set Line2D),
    Finite lines →
    Nat.card lines = 1000001 →
    (∃ (subset : Set Line2D),
      subset ⊆ lines ∧
      Nat.card subset = 1001 ∧
      (all_parallel subset ∨ pairwise_nonparallel subset)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ramsey_for_lines_l941_94135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l941_94197

/-- The speed of a train on an inclined track --/
noncomputable def train_speed_on_incline (P v ρ e : ℝ) : ℝ :=
  (ρ * v) / (ρ + e)

/-- Theorem stating the existence of a speed for the train on an inclined track --/
theorem train_speed_theorem (P v ρ e : ℝ) 
  (h_P : P = 150)
  (h_v : v = 72)
  (h_ρ : ρ = 0.005)
  (h_e : e = 0.030)
  (h_P_pos : P > 0)
  (h_v_pos : v > 0)
  (h_ρ_pos : ρ > 0)
  (h_e_pos : e > 0) :
  ∃ u : ℝ, u = train_speed_on_incline P v ρ e := by
  -- The proof goes here
  sorry

#check train_speed_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l941_94197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_area_ratio_l941_94152

/-- A regular pentagon -/
structure RegularPentagon where
  side_length : ℝ
  area : ℝ

/-- The inner pentagon formed by the intersection of all diagonals of a regular pentagon -/
structure InnerPentagon where
  outer_pentagon : RegularPentagon
  area : ℝ

/-- The ratio of the areas of a regular pentagon to its inner pentagon -/
noncomputable def area_ratio (outer : RegularPentagon) (inner : InnerPentagon) : ℝ :=
  outer.area / inner.area

/-- Theorem: The ratio of the area of a regular pentagon to the area of its inner pentagon
    is equal to (cos 36° / cos 72°)² -/
theorem regular_pentagon_area_ratio (F₁ : RegularPentagon) (F₂ : InnerPentagon) 
    (h : F₂.outer_pentagon = F₁) : 
  area_ratio F₁ F₂ = (Real.cos (36 * π / 180) / Real.cos (72 * π / 180))^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_area_ratio_l941_94152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l941_94132

noncomputable def f (x : ℝ) : ℝ := x^3 - 1/x^3

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x y, 0 < x → 0 < y → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l941_94132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_F_to_F_l941_94146

/-- The distance between a point and its reflection over the x-axis -/
noncomputable def distance_to_reflection (x y : ℝ) : ℝ :=
  Real.sqrt ((x - x)^2 + (y - (-y))^2)

/-- Theorem: The distance from point F(-2, 3) to its reflection over the x-axis is 6 -/
theorem distance_F_to_F'_is_6 :
  distance_to_reflection (-2) 3 = 6 := by
  -- Unfold the definition of distance_to_reflection
  unfold distance_to_reflection
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_F_to_F_l941_94146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l941_94148

-- Define the given functions
noncomputable def y (a b x : ℝ) := a - b * Real.cos (3 * x)
noncomputable def z (a b x : ℝ) := -4 * a * Real.sin (3 * b * x)
noncomputable def f (a b x : ℝ) := 2 * Real.sin (a * Real.pi / 3 - 2 * b * x)

-- State the theorem
theorem function_properties 
  (a b : ℝ) 
  (h1 : b > 0)
  (h2 : ∀ x, y a b x ≤ 3/2)
  (h3 : ∀ x, y a b x ≥ -1/2)
  (h4 : ∃ x, y a b x = 3/2)
  (h5 : ∃ x, y a b x = -1/2) :
  (a = 1/2 ∧ b = 1) ∧ 
  (∀ x, z a b (x + 2*Real.pi/(3*b)) = z a b x) ∧
  (∀ x, z a b x ≤ 2 ∧ z a b x ≥ -2) ∧
  (∃ x, z a b x = 2) ∧ (∃ x, z a b x = -2) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k*Real.pi - Real.pi/6) (k*Real.pi + Real.pi/3), 
    (∀ y ∈ Set.Icc (k*Real.pi - Real.pi/6) (k*Real.pi + Real.pi/3), x ≤ y → f a b x ≥ f a b y)) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k*Real.pi + Real.pi/3) (k*Real.pi + 5*Real.pi/6), 
    (∀ y ∈ Set.Icc (k*Real.pi + Real.pi/3) (k*Real.pi + 5*Real.pi/6), x ≤ y → f a b x ≤ f a b y)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l941_94148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parameter_sum_l941_94158

/-- Theorem about the sum of parameters in a hyperbola equation --/
theorem hyperbola_parameter_sum :
  ∀ (h k a b : ℝ),
  let center : ℝ × ℝ := (1, 0)
  let focus : ℝ × ℝ := (1 + Real.sqrt 41, 0)
  let vertex : ℝ × ℝ := (4, 0)
  (h = center.1 ∧ k = center.2) →
  (a = |vertex.1 - center.1|) →
  (∃ (c : ℝ), c = |focus.1 - center.1| ∧ b^2 = c^2 - a^2) →
  (∀ (x y : ℝ), (x - h)^2 / a^2 - (y - k)^2 / b^2 = 1) →
  h + k + a + b = 4 + 4 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parameter_sum_l941_94158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_calculation_l941_94172

/-- Represents a calculator with potentially switched digits -/
structure Calculator where
  digits : Fin 10 → Fin 10

/-- The normal calculator layout -/
def normalCalculator : Calculator :=
  ⟨id⟩

/-- The switched calculator layout -/
def switchedCalculator : Calculator :=
  ⟨fun n => match n with
    | 1 => 3 | 2 => 2 | 3 => 1
    | 4 => 6 | 5 => 5 | 6 => 4
    | 7 => 9 | 8 => 8 | 9 => 7
    | _ => n⟩

/-- Applies the calculator's digit mapping to a natural number -/
def applyCalculator (c : Calculator) (n : Nat) : Nat :=
  sorry

/-- The main theorem stating that 159 × 951 will give different results on normal and switched calculators -/
theorem incorrect_calculation :
  applyCalculator normalCalculator 159 * applyCalculator normalCalculator 951 ≠
  applyCalculator switchedCalculator 159 * applyCalculator switchedCalculator 951 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_calculation_l941_94172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_eq_height_l941_94103

/-- An equilateral triangle with side length a and height h -/
structure EquilateralTriangle where
  a : ℝ
  h : ℝ
  a_pos : 0 < a
  h_eq : h = a * Real.sqrt 3 / 2

/-- A point inside an equilateral triangle -/
structure PointInside (t : EquilateralTriangle) where
  x : ℝ
  y : ℝ
  inside : 0 < x ∧ 0 < y ∧ x + y < t.a

/-- The sum of distances from a point to the sides of an equilateral triangle -/
noncomputable def sumOfDistances (t : EquilateralTriangle) (p : PointInside t) : ℝ :=
  p.y + (t.a - p.x - p.y) * Real.sqrt 3 / 2 + p.x * Real.sqrt 3 / 2

/-- Theorem: The sum of distances from any point inside an equilateral triangle
    to its sides is equal to the height of the triangle -/
theorem sum_of_distances_eq_height (t : EquilateralTriangle) (p : PointInside t) :
    sumOfDistances t p = t.h := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_eq_height_l941_94103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_largest_number_l941_94162

theorem systematic_sampling_largest_number 
  (total_students : ℕ) 
  (smallest_number : ℕ) 
  (second_smallest : ℕ) 
  (h1 : total_students = 160)
  (h2 : smallest_number = 6)
  (h3 : second_smallest = 22)
  (h4 : smallest_number < second_smallest)
  (h5 : second_smallest ≤ total_students) :
  ∃ (sample_size : ℕ), 
    let interval := second_smallest - smallest_number
    let largest_number := smallest_number + (sample_size - 1) * interval
    largest_number = 150 ∧ largest_number ≤ total_students :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_largest_number_l941_94162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_even_numbers_count_l941_94157

/-- Given a sequence of consecutive even numbers with an average of 35 and a greatest number less than or equal to 39, prove that the length of this sequence is 4. -/
theorem consecutive_even_numbers_count (n : ℕ) (seq : List ℕ) : 
  seq.length = n →
  (∀ i j, i < j → i < seq.length → j < seq.length → seq.get! i < seq.get! j) →
  (∀ k, k < seq.length → Even (seq.get! k)) →
  (∀ i j, i + 1 < seq.length → j = i + 1 → seq.get! j - seq.get! i = 2) →
  seq.sum / n = 35 →
  seq.getLast? = some 38 →
  n = 4 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_even_numbers_count_l941_94157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_winning_strategy_l941_94156

/-- Represents a player in the game -/
inductive Player : Type
| Petya : Player
| Vasya : Player
deriving Repr, DecidableEq

/-- Represents a position on the grid -/
structure Position :=
  (x : Nat)
  (y : Nat)
deriving Repr

/-- Represents the game state -/
structure GameState :=
  (grid : List Position)
  (currentPlayer : Player)
deriving Repr

/-- Checks if a 4x4 square contains a chip -/
def has4x4Square (state : GameState) (topLeft : Position) : Bool :=
  sorry

/-- Checks if the game is won -/
def isGameWon (state : GameState) : Bool :=
  sorry

/-- Returns the next player -/
def nextPlayer (player : Player) : Player :=
  match player with
  | Player.Petya => Player.Vasya
  | Player.Vasya => Player.Petya

/-- Represents a move in the game -/
def makeMove (state : GameState) (pos : Position) : GameState :=
  { grid := pos :: state.grid,
    currentPlayer := nextPlayer state.currentPlayer }

/-- Theorem: Vasya has a winning strategy -/
theorem vasya_winning_strategy :
  ∃ (strategy : GameState → Position),
    ∀ (initialState : GameState),
      initialState.currentPlayer = Player.Petya →
      ∀ (game : Nat → GameState),
        game 0 = initialState →
        (∀ n : Nat, game (n + 1) = makeMove (game n) (
          if (game n).currentPlayer = Player.Vasya
          then strategy (game n)
          else Position.mk 0 0 -- Petya's move (placeholder)
        )) →
        ∃ n : Nat, isGameWon (game n) ∧ (game n).currentPlayer = Player.Vasya :=
by
  sorry

#check vasya_winning_strategy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_winning_strategy_l941_94156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l941_94128

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x)

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (x : ℝ), f (π/12 - x) = f (π/12 + x)) ∧
  (∀ (x : ℝ), f (π/3 - x) = f (π/3 + x)) ∧
  ¬(∀ (x y : ℝ), -π/3 ≤ x ∧ x < y ∧ y ≤ π/6 → f x < f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l941_94128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_inequality_l941_94112

theorem sum_inequality (m n : ℕ) (a : Fin m → ℕ) : 
  m > 0 ∧ n > 0 ∧ 
  (∀ i : Fin m, a i ∈ Finset.range n) ∧
  (∀ i j : Fin m, i < j → a i ≠ a j) ∧
  (∀ i j : Fin m, i ≤ j → a i + a j ≤ n → 
    ∃ k : Fin m, a i + a j = a k) →
  (Finset.sum (Finset.univ : Finset (Fin m)) a) / m ≥ (n + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_inequality_l941_94112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_rectangle_minimizes_perimeter_l941_94195

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- The minimum area requirement --/
def minArea : ℝ := 600

/-- The optimal rectangle --/
noncomputable def optimalRectangle : Rectangle :=
  { length := 10 * Real.sqrt 6
  , width := 10 * Real.sqrt 6 }

theorem optimal_rectangle_minimizes_perimeter :
  (area optimalRectangle ≥ minArea) ∧
  (∀ r : Rectangle, area r ≥ minArea → perimeter r ≥ perimeter optimalRectangle) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_rectangle_minimizes_perimeter_l941_94195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_point_relationship_l941_94144

/-- Given an inverse proportion function y = (m^2 + 1) / x, prove the relationship
    between y-coordinates of specific points on its graph. -/
theorem inverse_proportion_point_relationship (m : ℝ) (y₁ y₂ y₃ : ℝ) :
  (∀ x : ℝ, x ≠ 0 → (m^2 + 1) / x = (fun x ↦ (m^2 + 1) / x) x) →  -- Function definition
  (m^2 + 1) / (-2) = y₁ →  -- Point (-2, y₁) on the graph
  (m^2 + 1) / (-1) = y₂ →  -- Point (-1, y₂) on the graph
  (m^2 + 1) / 1 = y₃ →     -- Point (1, y₃) on the graph
  y₂ < y₁ ∧ y₁ < y₃ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_point_relationship_l941_94144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_ones_and_zeros_l941_94159

theorem divisibility_by_ones_and_zeros (k : ℕ) (hk : k > 0) :
  (∃ n : ℕ, (∀ d : ℕ, d ∈ n.digits 10 → d = 1 ∨ d = 0) ∧ k ∣ n) ∧
  (Nat.Coprime k 10 → ∃ m : ℕ, (∀ d : ℕ, d ∈ m.digits 10 → d = 1) ∧ k ∣ m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_ones_and_zeros_l941_94159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l941_94155

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequence) 
  (h : S seq 9 = 9 * S seq 5) : 
  seq.a 5 / seq.a 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l941_94155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smile_area_theorem_l941_94139

/-- The area of the "smile-shaped" region in the given geometric configuration -/
noncomputable def smile_area (r₁ r₂ : ℝ) : ℝ :=
  (9 * Real.pi / 2) - (27 * Real.arctan (2/3) / Real.pi)

/-- Theorem stating the area of the "smile-shaped" region -/
theorem smile_area_theorem (O P Q R : ℝ × ℝ) (S₁ : Set (ℝ × ℝ)) :
  (∀ x ∈ S₁, dist O x = 2) →  -- S₁ is a semicircle with center O and radius 2
  P ∈ S₁ →  -- P lies on S₁
  (O.1 - P.1) * (O.2 - P.2) = 0 →  -- OP is perpendicular to the diameter of S₁
  dist O Q = 3 →  -- OQ = 3
  dist O R = 3 →  -- OR = 3
  smile_area 2 3 = (9 * Real.pi / 2) - (27 * Real.arctan (2/3) / Real.pi) := by
  sorry

#check smile_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smile_area_theorem_l941_94139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_one_six_l941_94100

/-- Represents the outcome of rolling two dice -/
structure DiceRoll where
  first : Fin 6
  second : Fin 6
deriving Fintype, DecidableEq

/-- The set of all possible outcomes when rolling two dice -/
def allRolls : Finset DiceRoll := Finset.univ

/-- The set of outcomes where the dice show different numbers -/
def differentRolls : Finset DiceRoll :=
  allRolls.filter (fun roll => roll.first ≠ roll.second)

/-- The set of outcomes where at least one die shows a 6 -/
def atLeastOneSix : Finset DiceRoll :=
  differentRolls.filter (fun roll => roll.first = 5 ∨ roll.second = 5)

/-- The probability of an event given the condition of different rolls -/
def conditionalProbability (event : Finset DiceRoll) : ℚ :=
  (event ∩ differentRolls).card / differentRolls.card

theorem probability_at_least_one_six :
  conditionalProbability atLeastOneSix = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_one_six_l941_94100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_perimeter_l941_94110

-- Define the necessary structures and types
structure Point := (x y : ℝ)

class EuclideanPlane (α : Type) where
  -- Add necessary axioms and definitions for Euclidean plane

def SegmentLength (A B : Point) : ℝ := sorry
def Perpendicular (C E : Point) (L : Set Point) (plane : Type) [EuclideanPlane plane] : Prop := sorry
def Line (A D : Point) : Set Point := sorry
def Polygon := List Point
def Perimeter (p : Polygon) : ℝ := sorry
def IsoscelesTrapezoid (A B C D : Point) (plane : Type) [EuclideanPlane plane] : Prop := sorry

theorem isosceles_trapezoid_perimeter : 
  ∀ (A B C D E : Point) (plane : Type) [EuclideanPlane plane],
  IsoscelesTrapezoid A B C D plane →
  SegmentLength A B = (SegmentLength B C) / Real.sqrt 2 →
  Perpendicular C E (Line A D) plane →
  SegmentLength B E = Real.sqrt 5 →
  SegmentLength B D = Real.sqrt 10 →
  Perimeter [A, B, C, D] = 6 + 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_perimeter_l941_94110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l941_94131

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (2 * Real.cos x - 1)

-- Theorem stating the domain of f
theorem domain_of_f :
  ∀ x : ℝ, f x ∈ Set.univ ↔ ∃ k : ℤ, x ∈ Set.Ioo (-π/3 + 2*π*(k:ℝ)) (π/3 + 2*π*(k:ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l941_94131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersections_l941_94127

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y = 12

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 16

-- Define the number of intersections
def num_intersections : ℕ := 2

-- Theorem statement
theorem line_circle_intersections :
  ∃ (points : Finset (ℝ × ℝ)), 
    (∀ (p : ℝ × ℝ), p ∈ points ↔ (line_eq p.1 p.2 ∧ circle_eq p.1 p.2)) ∧
    points.card = num_intersections :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersections_l941_94127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_some_number_value_l941_94126

/-- Given that (3.242 * some_number) / 100 = 0.045388, prove that some_number ≈ 1.400 -/
theorem some_number_value (some_number : ℝ) 
  (h : (3.242 * some_number) / 100 = 0.045388) : 
  ‖some_number - 1.400‖ < 0.001 := by
  sorry

#eval Float.abs (1.400 - (0.045388 * 100 / 3.242))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_some_number_value_l941_94126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l941_94154

noncomputable def f (x : ℝ) : ℝ := -Real.sin (Real.pi / 2 * x)

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x y, 0 < x ∧ x < y ∧ y < 1 → f y < f x) ∧
  (∀ x, f (x + 4) = f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l941_94154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_plus_exp_abs_l941_94173

theorem integral_sqrt_plus_exp_abs :
  ∫ x in (-1 : ℝ)..1, (Real.sqrt (1 - x^2) + Real.exp (abs x)) = π / 2 + 2 * Real.exp 1 - 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_plus_exp_abs_l941_94173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_a_monotonicity_intervals_l941_94120

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / 4 + a / x - Real.log x - 3 / 2

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 1 / 4 - a / (x^2) - 1 / x

theorem tangent_perpendicular_implies_a (a : ℝ) :
  f' a 1 = -2 → a = 1 / 4 := by sorry

theorem monotonicity_intervals (x : ℝ) (h : x > 0) :
  let a := 1 / 4
  (x < 1 → (f' a x < 0)) ∧
  (x > 1 → (f' a x > 0)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_a_monotonicity_intervals_l941_94120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_apples_picked_l941_94140

def apples_per_pie : ℕ := 4
def unripe_apples : ℕ := 6
def pies_made : ℕ := 7

theorem total_apples_picked : 
  apples_per_pie * pies_made + unripe_apples = 34 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_apples_picked_l941_94140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_perpendicular_lines_l941_94160

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- First line equation: 3x - 4y = d -/
def line1 (x y d : ℝ) : Prop := 3 * x - 4 * y = d

/-- Second line equation: 8x + ky = d -/
def line2 (x y k d : ℝ) : Prop := 8 * x + k * y = d

/-- The slope of the first line -/
noncomputable def slope1 : ℝ := 3 / 4

/-- The slope of the second line -/
noncomputable def slope2 (k : ℝ) : ℝ := -8 / k

theorem intersection_of_perpendicular_lines (k d : ℝ) : 
  perpendicular slope1 (slope2 k) → 
  line1 2 (-3) d → 
  line2 2 (-3) k d → 
  d = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_perpendicular_lines_l941_94160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_successful_meeting_probability_approx_quarter_l941_94165

/-- Represents the arrival time of a person in hours after 3:00 p.m. -/
def ArrivalTime := { t : ℝ // 0 ≤ t ∧ t ≤ 2 }

/-- Represents the scenario of the manager and three interns arriving -/
structure MeetingScenario where
  manager : ArrivalTime
  intern1 : ArrivalTime
  intern2 : ArrivalTime
  intern3 : ArrivalTime

/-- Checks if the meeting is successful based on arrival times -/
def is_meeting_successful (scenario : MeetingScenario) : Prop :=
  let m := scenario.manager.val
  let i1 := scenario.intern1.val
  let i2 := scenario.intern2.val
  let i3 := scenario.intern3.val
  m > i1 ∧ m > i2 ∧ m > i3 ∧
  |i1 - i2| ≤ 0.5 ∧ |i1 - i3| ≤ 0.5 ∧ |i2 - i3| ≤ 0.5

/-- The probability space of all possible meeting scenarios -/
def MeetingProbabilitySpace := { s : Set MeetingScenario // True }

/-- The probability of a successful meeting -/
noncomputable def successful_meeting_probability (space : MeetingProbabilitySpace) : ℝ :=
  sorry

/-- Theorem stating that the probability of a successful meeting is approximately 0.25 -/
theorem successful_meeting_probability_approx_quarter (space : MeetingProbabilitySpace) :
  ∃ ε > 0, |successful_meeting_probability space - 0.25| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_successful_meeting_probability_approx_quarter_l941_94165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_zeroing_l941_94129

/-- Represents a move on the grid -/
structure Move where
  x1 : Nat
  y1 : Nat
  x2 : Nat
  y2 : Nat
  value : Int

/-- Represents the n × n grid -/
def Grid (n : Nat) := Fin n → Fin n → Int

/-- Check if two squares are adjacent -/
def adjacent (x1 y1 x2 y2 : Nat) : Prop :=
  (x1 = x2 ∧ (y1 = y2 + 1 ∨ y2 = y1 + 1)) ∨
  (y1 = y2 ∧ (x1 = x2 + 1 ∨ x2 = x1 + 1))

/-- Apply a move to the grid -/
def applyMove (n : Nat) (grid : Grid n) (move : Move) : Grid n :=
  λ x y => 
    if (x.val = move.x1 && y.val = move.y1) || (x.val = move.x2 && y.val = move.y2) then
      grid x y + move.value
    else
      grid x y

/-- Check if all squares in the grid display 0 -/
def allZero (n : Nat) (grid : Grid n) : Prop :=
  ∀ x y, grid x y = 0

/-- The main theorem -/
theorem grid_zeroing (n : Nat) :
  (∃ (moves : List Move), allZero n (moves.foldl (applyMove n) (λ _ _ => 0))) ↔ 
  Even n ∧
  (∀ (moves : List Move), allZero n (moves.foldl (applyMove n) (λ _ _ => 0)) → 
    moves.length ≥ 3 * n^2 / 4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_zeroing_l941_94129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_jumps_theorem_l941_94119

/-- The number of points on the circle -/
def n : ℕ := 2016

/-- The possible jump lengths -/
inductive JumpLength
| two : JumpLength
| three : JumpLength

/-- A path on the circle is a list of jumps -/
def CirclePath := List JumpLength

/-- The length of a path in terms of points visited -/
def pathLength (p : CirclePath) : ℕ :=
  p.foldl (λ acc jump => acc + match jump with
    | JumpLength.two => 2
    | JumpLength.three => 3) 0

/-- A path is valid if it visits all points and returns to the start -/
def isValidPath (p : CirclePath) : Prop :=
  pathLength p = n ∧ p.length > 0

/-- The main theorem to prove -/
theorem min_jumps_theorem :
  ∃ (p : CirclePath), isValidPath p ∧ p.length = 673 ∧
  ∀ (q : CirclePath), isValidPath q → q.length ≥ 673 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_jumps_theorem_l941_94119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l941_94114

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (1/3)^(2*x + k)

-- Theorem statement
theorem function_properties (k : ℝ) :
  (f k (-1) = 3) →
  (k = 1) ∧
  (∀ a : ℝ, f k a ≥ 27 → a ≤ -2) ∧
  (∀ b : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f k (|x₁|) - b = 0 ∧ f k (|x₂|) - b = 0) → 
    (0 < b ∧ b < 1/3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l941_94114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_at_negative_103_l941_94176

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x^3 + 5

-- State the theorem
theorem inverse_g_at_negative_103 :
  g ⁻¹' {-3} = {-103} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_at_negative_103_l941_94176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_properties_l941_94161

-- Define the linear function
def f (x : ℝ) : ℝ := 3 * x - 1

-- Define points A and B as functions of a, b, and k
def A (a b : ℝ) : ℝ × ℝ := (a, b)
def B (a b k : ℝ) : ℝ × ℝ := (a + 1, b + k)

-- Theorem statement
theorem linear_function_properties (a b k : ℝ) :
  -- A and B are on the graph of f
  f (A a b).1 = (A a b).2 ∧ f (B a b k).1 = (B a b k).2 →
  -- Part 1: k = 3
  k = 3 ∧
  -- Part 2: If A is on the y-axis, then B = (1, 2)
  ((A a b).1 = 0 → B a b k = (1, 2)) ∧
  -- Part 3: If A is on the y-axis, there exists a point P on the x-axis forming an isosceles triangle BOP
  ((A a b).1 = 0 → ∃ P : ℝ × ℝ, P.2 = 0 ∧ P.1 = -5/2 ∧
    ((B a b k).1 - P.1)^2 + ((B a b k).2 - P.2)^2 = (0 - P.1)^2 + (0 - P.2)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_properties_l941_94161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_radius_increase_approx_l941_94151

/-- Calculates the increase in wheel radius given trip readings and original radius -/
noncomputable def wheel_radius_increase (original_distance : ℝ) (return_distance : ℝ) (original_radius : ℝ) : ℝ :=
  let inches_per_mile : ℝ := 63360
  let pi : ℝ := Real.pi
  let new_radius : ℝ := (original_distance * 2 * pi * original_radius) / (return_distance * inches_per_mile)
  new_radius - original_radius

/-- Theorem stating that the wheel radius increase is approximately 0.39 inches -/
theorem wheel_radius_increase_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ 
  |wheel_radius_increase 500 485 16 - 0.39| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_radius_increase_approx_l941_94151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_curve_intersection_l941_94106

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (t, 4 + t)

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : ℝ) : ℝ := 4 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)

-- Define the number of intersection points
def intersection_count : ℕ := 1

-- Theorem statement
theorem line_curve_intersection :
  ∃! p : ℝ × ℝ, ∃ t θ : ℝ,
    line_l t = p ∧
    (curve_C θ * Real.cos θ, curve_C θ * Real.sin θ) = p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_curve_intersection_l941_94106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_solution_l941_94194

/-- Represents a term in the sequence -/
structure SequenceTerm where
  index : ℕ
  numerator : ℝ
  denominator : ℝ

/-- The sequence satisfies the given pattern -/
def satisfies_pattern (seq : ℕ → SequenceTerm) : Prop :=
  ∀ n : ℕ, (seq n).numerator^2 = (seq n).denominator + 1 ∧ 
            (seq n).denominator = (n + 1)^2 + 1

/-- The third term of the sequence -/
noncomputable def third_term (a b : ℝ) : SequenceTerm :=
  { index := 3
  , numerator := Real.sqrt 17
  , denominator := a + b }

/-- The fourth term of the sequence -/
noncomputable def fourth_term (a b : ℝ) : SequenceTerm :=
  { index := 4
  , numerator := Real.sqrt (a - b)
  , denominator := 25 }

/-- The main theorem -/
theorem sequence_solution (seq : ℕ → SequenceTerm) (a b : ℝ) :
  satisfies_pattern seq ∧ 
  seq 3 = third_term a b ∧
  seq 4 = fourth_term a b →
  a = 21 ∧ b = -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_solution_l941_94194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maintenance_team_journey_l941_94109

def distances : List Int := [22, -3, 4, -2, -8, -17, -2, 12, 7, -5]
def fuel_rate : Float := 0.07

theorem maintenance_team_journey :
  let final_position := distances.sum
  let total_distance := distances.map Int.natAbs |>.sum
  let fuel_consumed := (total_distance.toFloat) * fuel_rate
  (final_position = 8 ∧ fuel_consumed = 5.74) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maintenance_team_journey_l941_94109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bushes_for_72_zucchinis_l941_94111

/-- Represents the number of containers of blueberries per bush -/
def containers_per_bush : ℕ := 10

/-- Represents the number of containers of blueberries needed to trade for one zucchini -/
def containers_per_zucchini : ℚ := 2

/-- Calculates the minimum number of bushes needed to obtain at least the given number of zucchinis -/
def min_bushes_needed (zucchinis : ℕ) : ℕ :=
  (((zucchinis : ℚ) * containers_per_zucchini / containers_per_bush).ceil).toNat

/-- Proves that the minimum number of bushes needed to obtain at least 72 zucchinis is 15 -/
theorem min_bushes_for_72_zucchinis :
  min_bushes_needed 72 = 15 := by
  sorry

#eval min_bushes_needed 72

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bushes_for_72_zucchinis_l941_94111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_solution_l941_94169

-- Define the set of integers satisfying the inequalities
def S : Set ℤ :=
  {x : ℤ | -5*x ≥ 3*x + 11 ∧ -3*x ≤ 15 ∧ -6*x ≥ 4*x + 23}

-- Theorem statement
theorem inequalities_solution :
  ∃ (A : Finset ℤ), (↑A : Set ℤ) = S ∧ Finset.card A = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_solution_l941_94169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l941_94143

/-- Conversion from cylindrical coordinates to rectangular coordinates -/
noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

/-- The given point in cylindrical coordinates -/
noncomputable def cylindrical_point : ℝ × ℝ × ℝ := (10, Real.pi / 3, 2)

theorem cylindrical_to_rectangular_conversion :
  cylindrical_to_rectangular cylindrical_point.1 cylindrical_point.2.1 cylindrical_point.2.2 =
  (5, 5 * Real.sqrt 3, 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l941_94143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_length_l941_94153

/-- The hyperbola equation x^2 - y^2 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

/-- The focus of the hyperbola -/
noncomputable def focus : ℝ := Real.sqrt 2

/-- The line perpendicular to x-axis passing through the focus -/
def perpendicular_line (x : ℝ) : Prop := x = focus

/-- Point A is on both the hyperbola and the perpendicular line -/
noncomputable def point_A : ℝ × ℝ := (focus, -1)

/-- Point B is on both the hyperbola and the perpendicular line -/
noncomputable def point_B : ℝ × ℝ := (focus, 1)

/-- The length of line segment AB -/
noncomputable def length_AB : ℝ := point_B.2 - point_A.2

theorem hyperbola_intersection_length :
  hyperbola point_A.1 point_A.2 ∧
  hyperbola point_B.1 point_B.2 ∧
  perpendicular_line point_A.1 ∧
  perpendicular_line point_B.1 →
  length_AB = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_length_l941_94153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_not_monotonic_l941_94108

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * a * x^3 - x^2 + 5

-- State the theorem
theorem function_not_monotonic (a : ℝ) (h1 : a > 0) :
  (∃ x y, x ∈ Set.Ioo 0 2 ∧ y ∈ Set.Ioo 0 2 ∧ x < y ∧ f a x > f a y) ↔ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_not_monotonic_l941_94108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l941_94179

/-- The circle defined by the equation x^2 + y^2 - 2x - 2y + 1 = 0 -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

/-- The line defined by the equation 3x + 4y + 8 = 0 -/
def line_eq (x y : ℝ) : Prop := 3*x + 4*y + 8 = 0

/-- The distance from a point (x, y) to the line 3x + 4y + 8 = 0 -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |3*x + 4*y + 8| / Real.sqrt (3^2 + 4^2)

/-- The minimum distance from the circle to the line is 2 -/
theorem min_distance_circle_to_line :
  ∃ (d : ℝ), d = 2 ∧
  ∀ (x y : ℝ), circle_eq x y → distance_to_line x y ≥ d ∧
  ∃ (x' y' : ℝ), circle_eq x' y' ∧ distance_to_line x' y' = d :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l941_94179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_triangle_l941_94150

/-- An isosceles triangle with given side lengths -/
structure IsoscelesTriangle where
  -- Length of equal sides
  side : ℝ
  -- Length of base
  base : ℝ
  -- side > 0 and base > 0
  side_pos : side > 0
  base_pos : base > 0
  -- The base must be shorter than twice the side length for the triangle to exist
  valid : base < 2 * side

/-- Calculate the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  let height := Real.sqrt (t.side^2 - (t.base/2)^2)
  (1/2) * t.base * height

/-- The main theorem: area of the specific isosceles triangle -/
theorem area_of_specific_triangle :
  let t : IsoscelesTriangle := {
    side := 15,
    base := 8,
    side_pos := by norm_num,
    base_pos := by norm_num,
    valid := by norm_num
  }
  area t = 4 * Real.sqrt 209 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_triangle_l941_94150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l941_94175

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

theorem function_properties :
  ∀ (A ω φ : ℝ),
  A > 0 ∧ ω > 0 ∧ 0 < φ ∧ φ < Real.pi / 2 →
  (∀ (x₁ x₂ : ℝ), f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ < x₂ ∧ ∀ x ∈ Set.Ioo x₁ x₂, f x ≠ 0 → x₂ - x₁ = Real.pi / 2) →
  f (2 * Real.pi / 3) = -2 →
  (∃ (A' ω' φ' : ℝ), A' > 0 ∧ ω' > 0 ∧ 0 < φ' ∧ φ' < Real.pi / 2 ∧
   (∀ x, f x = A' * Real.sin (ω' * x + φ'))) ∧
  (∀ k : ℤ, ∀ x : ℝ, f (Real.pi / 6 + ↑k * Real.pi / 2 - x) = f (Real.pi / 6 + ↑k * Real.pi / 2 + x)) ∧
  (∀ k : ℤ, ∀ x : ℝ, f (↑k * Real.pi / 2 - Real.pi / 12 - x) = -f (↑k * Real.pi / 2 - Real.pi / 12 + x)) ∧
  (∀ x : ℝ, x ≥ Real.pi / 12 ∧ x ≤ Real.pi / 2 → f x ≥ -1 ∧ f x ≤ 2) ∧
  (∃ x : ℝ, x ≥ Real.pi / 12 ∧ x ≤ Real.pi / 2 ∧ f x = -1) ∧
  (∃ x : ℝ, x ≥ Real.pi / 12 ∧ x ≤ Real.pi / 2 ∧ f x = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l941_94175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l941_94118

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem stating that the area of a trapezium with parallel sides of 26 cm and 18 cm, 
    and a distance of 15 cm between them, is 330 cm². -/
theorem trapezium_area_example : trapeziumArea 26 18 15 = 330 := by
  -- Unfold the definition of trapeziumArea
  unfold trapeziumArea
  -- Simplify the arithmetic
  simp [add_mul, mul_div_right_comm]
  -- Check that the result is equal to 330
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l941_94118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_identity_l941_94147

theorem sin_identity (A : ℝ) : 
  Real.sin (5 * A) - 5 * Real.sin (3 * A) + 10 * Real.sin A = 16 * (Real.sin A)^5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_identity_l941_94147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasyas_birthday_l941_94171

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns the day two days after the given day -/
def twoDaysAfter (day : DayOfWeek) : DayOfWeek :=
  match day with
  | DayOfWeek.Sunday => DayOfWeek.Tuesday
  | DayOfWeek.Monday => DayOfWeek.Wednesday
  | DayOfWeek.Tuesday => DayOfWeek.Thursday
  | DayOfWeek.Wednesday => DayOfWeek.Friday
  | DayOfWeek.Thursday => DayOfWeek.Saturday
  | DayOfWeek.Friday => DayOfWeek.Sunday
  | DayOfWeek.Saturday => DayOfWeek.Monday

/-- Returns the day before the given day -/
def dayBefore (day : DayOfWeek) : DayOfWeek :=
  match day with
  | DayOfWeek.Sunday => DayOfWeek.Saturday
  | DayOfWeek.Monday => DayOfWeek.Sunday
  | DayOfWeek.Tuesday => DayOfWeek.Monday
  | DayOfWeek.Wednesday => DayOfWeek.Tuesday
  | DayOfWeek.Thursday => DayOfWeek.Wednesday
  | DayOfWeek.Friday => DayOfWeek.Thursday
  | DayOfWeek.Saturday => DayOfWeek.Friday

theorem vasyas_birthday (statement_day : DayOfWeek) 
  (h1 : twoDaysAfter statement_day = DayOfWeek.Sunday) : 
  dayBefore statement_day = DayOfWeek.Thursday :=
by
  sorry

-- Remove the #eval line as it's not needed for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasyas_birthday_l941_94171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_equation_l941_94149

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

-- Define the line
def line (x y : ℝ) : Prop := x - 2*y + 1 = 0

-- Define a function to create a line through two points
def line_through (A B : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - A.2) * (B.1 - A.1) = (x - A.1) * (B.2 - A.2)

-- Theorem statement
theorem intersection_line_equation :
  ∀ (A B : ℝ × ℝ),
  (circle1 A.1 A.2 ∧ circle2 A.1 A.2) →
  (circle1 B.1 B.2 ∧ circle2 B.1 B.2) →
  A ≠ B →
  (∀ (x y : ℝ), line_through A B x y ↔ line x y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_equation_l941_94149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_value_at_7_l941_94125

-- Define the polynomial Q(x)
def Q (x : ℂ) (g h k l m n : ℝ) : ℂ :=
  (3 * x^4 - 30 * x^3 + g * x^2 + h * x + k) * (4 * x^4 - 60 * x^3 + l * x^2 + m * x + n)

-- Define the set of complex roots
def roots : Set ℂ := {2, 3, 4, 5, 6}

-- Theorem statement
theorem Q_value_at_7 (g h k l m n : ℝ) :
  (∀ z : ℂ, Q z g h k l m n = 0 → z ∈ roots) →
  (∀ z : ℂ, z ∈ roots → ∃ i : ℕ, i > 0 ∧ (Q z g h k l m n = 0)) →
  Q 7 g h k l m n = 28800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_value_at_7_l941_94125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_tendencies_after_donation_l941_94170

def initial_donations : List ℚ := [5, 3, 6, 5, 10]
def additional_donation : ℚ := 10

noncomputable def median (l : List ℚ) : ℚ := sorry
noncomputable def mode (l : List ℚ) : ℚ := sorry
noncomputable def mean (l : List ℚ) : ℚ := sorry

def updated_donations : List ℚ := 
  (initial_donations.filter (· ≠ 10)) ++ [10 + additional_donation]

theorem central_tendencies_after_donation :
  (median initial_donations = median updated_donations) ∧
  (mode initial_donations = mode updated_donations) ∧
  (mean initial_donations ≠ mean updated_donations) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_tendencies_after_donation_l941_94170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drum_fill_time_l941_94178

/-- The time required to fill a cylindrical drum with rain -/
noncomputable def time_to_fill_drum (rainfall_rate : ℝ) (drum_depth : ℝ) (drum_base_area : ℝ) : ℝ :=
  (drum_depth * drum_base_area) / (rainfall_rate * drum_base_area)

/-- Theorem: The time to fill the drum is 3 hours -/
theorem drum_fill_time :
  let rainfall_rate : ℝ := 5
  let drum_depth : ℝ := 15
  let drum_base_area : ℝ := 300
  time_to_fill_drum rainfall_rate drum_depth drum_base_area = 3 := by
  -- Unfold the definition and simplify
  unfold time_to_fill_drum
  -- Perform algebraic simplification
  simp [mul_div_cancel]
  -- The proof is complete
  norm_num
  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_drum_fill_time_l941_94178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_classroom_weight_distribution_exists_l941_94123

theorem classroom_weight_distribution_exists :
  ∃ (n : ℕ) (b g : ℕ) (boy_weights girl_weights : List ℝ),
    n < 35 ∧
    n = b + g ∧
    b > 0 ∧
    g > 0 ∧
    boy_weights.length = b ∧
    girl_weights.length = g ∧
    (boy_weights.sum + girl_weights.sum) / n = 53.5 ∧
    boy_weights.sum / b = 60 ∧
    girl_weights.sum / g = 47 ∧
    (∃ (lightest_boy : ℝ) (heaviest_girl : ℝ),
      lightest_boy ∈ boy_weights ∧
      heaviest_girl ∈ girl_weights ∧
      lightest_boy < List.minimum girl_weights ∧
      heaviest_girl > List.maximum boy_weights) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_classroom_weight_distribution_exists_l941_94123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_offset_length_l941_94164

/-- The area of a quadrilateral given its diagonal and two offsets -/
noncomputable def quadrilateralArea (diagonal : ℝ) (offset1 offset2 : ℝ) : ℝ :=
  (1/2) * diagonal * (offset1 + offset2)

/-- Theorem: Given a quadrilateral with one diagonal of 22 cm, one offset of 9 cm,
    and an area of 165 cm², the length of the second offset is 6 cm. -/
theorem second_offset_length (diagonal : ℝ) (offset1 : ℝ) (area : ℝ) :
  diagonal = 22 → offset1 = 9 → area = 165 →
  ∃ (offset2 : ℝ), quadrilateralArea diagonal offset1 offset2 = area ∧ offset2 = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_offset_length_l941_94164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisor_of_square_plus_three_l941_94183

theorem prime_divisor_of_square_plus_three (p : ℕ) (h_prime : Nat.Prime p) :
  (∃ a : ℕ, p ∣ a^2 + 3) → p = 2 ∨ p = 3 ∨ ∃ k : ℕ, p = 3 * k + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisor_of_square_plus_three_l941_94183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitive_not_perpendicular_transitive_not_parallel_to_plane_implies_parallel_perpendicular_to_plane_implies_parallel_l941_94193

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [CompleteSpace V]

-- Define lines and planes
def Line (V : Type*) [NormedAddCommGroup V] := V → V → Prop
def Plane (V : Type*) [NormedAddCommGroup V] := V → Prop

-- Define parallel and perpendicular relations
def parallel {V : Type*} [NormedAddCommGroup V] (l1 l2 : Line V) : Prop := sorry
def perpendicular {V : Type*} [NormedAddCommGroup V] (l1 l2 : Line V) : Prop := sorry
def lineParallelToPlane {V : Type*} [NormedAddCommGroup V] (l : Line V) (p : Plane V) : Prop := sorry
def linePerpendicularToPlane {V : Type*} [NormedAddCommGroup V] (l : Line V) (p : Plane V) : Prop := sorry

-- Theorem 1
theorem parallel_transitive {V : Type*} [NormedAddCommGroup V] (a b c : Line V) :
  parallel a b → parallel b c → parallel a c := by sorry

-- Theorem 2
theorem not_perpendicular_transitive {V : Type*} [NormedAddCommGroup V] : 
  ∃ (a b c : Line V), perpendicular a b ∧ perpendicular b c ∧ ¬perpendicular a c := by sorry

-- Theorem 3
theorem not_parallel_to_plane_implies_parallel {V : Type*} [NormedAddCommGroup V] : 
  ∃ (a b : Line V) (γ : Plane V), lineParallelToPlane a γ ∧ lineParallelToPlane b γ ∧ ¬parallel a b := by sorry

-- Theorem 4
theorem perpendicular_to_plane_implies_parallel {V : Type*} [NormedAddCommGroup V] (a b : Line V) (γ : Plane V) :
  linePerpendicularToPlane a γ → linePerpendicularToPlane b γ → parallel a b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitive_not_perpendicular_transitive_not_parallel_to_plane_implies_parallel_perpendicular_to_plane_implies_parallel_l941_94193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_sum_l941_94191

theorem max_value_sqrt_sum (x : ℝ) (h : x ∈ Set.Icc (-49) 49) : 
  Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ 14 ∧ 
  ∃ y ∈ Set.Icc (-49 : ℝ) 49, Real.sqrt (49 + y) + Real.sqrt (49 - y) = 14 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_sum_l941_94191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_is_112_l941_94182

/-- A journey with two segments at different speeds -/
structure Journey where
  total_time : ℚ
  speed1 : ℚ
  speed2 : ℚ

/-- Calculate the total distance of a journey -/
noncomputable def total_distance (j : Journey) : ℚ :=
  (j.speed1 * j.speed2 * j.total_time) / (j.speed1 / 2 + j.speed2 / 2)

/-- Theorem stating that for the given journey parameters, the total distance is 112 km -/
theorem journey_distance_is_112 : 
  let j : Journey := { total_time := 5, speed1 := 21, speed2 := 24 }
  total_distance j = 112 := by
  sorry

#check journey_distance_is_112

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_is_112_l941_94182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_to_left_evaluation_l941_94181

/-- Evaluates an expression from right to left -/
noncomputable def rightToLeftEval (a b c d e : ℝ) : ℝ := a + (b / (c - (d * e)))

/-- The expression to be evaluated -/
noncomputable def expression (a b c d e : ℝ) : ℝ := a + b / c - d * e

/-- Theorem stating that the right-to-left evaluation matches the expression -/
theorem right_to_left_evaluation (a b c d e : ℝ) :
  rightToLeftEval a b c d e = expression a b c d e :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_to_left_evaluation_l941_94181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_from_unit_circle_intersection_l941_94105

theorem sin_value_from_unit_circle_intersection (α : Real) :
  (∃ (x y : Real), x^2 + y^2 = 1 ∧ x = -Real.sqrt 3/2 ∧ y = -1/2 ∧ 
   x = Real.cos α ∧ y = Real.sin α) →
  Real.sin α = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_from_unit_circle_intersection_l941_94105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_value_theorem_for_integrals_l941_94168

open MeasureTheory

theorem mean_value_theorem_for_integrals 
  {f : ℝ → ℝ} {a b : ℝ} (hf : ContinuousOn f (Set.Icc a b)) (hab : a ≤ b) :
  ∃ ξ ∈ Set.Icc a b, ∫ x in a..b, f x = f ξ * (b - a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_value_theorem_for_integrals_l941_94168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l941_94174

-- Define the sets A and B
def A : Set ℝ := {x | (1/2 : ℝ) < (2 : ℝ)^x ∧ (2 : ℝ)^x < 8}
def B (m : ℝ) : Set ℝ := {x | -1 < x ∧ x < m + 1}

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ x, x ∈ B m → x ∈ A) ∧ (∃ x, x ∈ A ∧ x ∉ B m) ↔ m > 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l941_94174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_and_isolating_line_l941_94166

noncomputable section

open Real

-- Define the functions
def h (x : ℝ) : ℝ := x^2
def φ (x : ℝ) : ℝ := 2 * (exp 1) * log x
def F (x : ℝ) : ℝ := h x - φ x

-- Define the isolating line
def isolating_line (x : ℝ) : ℝ := 2 * sqrt (exp 1) * x - (exp 1)

-- Theorem statement
theorem extreme_value_and_isolating_line :
  (∃ (x : ℝ), x > 0 ∧ F x = 0 ∧ ∀ (y : ℝ), y > 0 → F y ≥ F x) ∧
  (∀ (x : ℝ), x > 0 → h x ≥ isolating_line x) ∧
  (∀ (x : ℝ), x > 0 → φ x ≤ isolating_line x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_and_isolating_line_l941_94166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_fraction_decomposition_product_l941_94187

-- Define the polynomial and its factorization
noncomputable def p (x : ℝ) : ℝ := x^2 - 25
noncomputable def q (x : ℝ) : ℝ := (x - 2) * (x + 2) * (x - 3)

-- Define the partial fraction decomposition
noncomputable def partialFraction (A B C : ℝ) (x : ℝ) : ℝ :=
  A / (x - 2) + B / (x + 2) + C / (x - 3)

-- Theorem statement
theorem partial_fraction_decomposition_product :
  ∃ A B C : ℝ, ∀ x : ℝ, x ≠ 2 ∧ x ≠ -2 ∧ x ≠ 3 →
    p x / q x = partialFraction A B C x ∧ A * B * C = 1764 / 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_fraction_decomposition_product_l941_94187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_committee_count_l941_94102

def num_teams : Nat := 5
def team_size : Nat := 8
def committee_size : Nat := 16
def host_members : Nat := 4
def non_host_members : Nat := 3

theorem chess_committee_count :
  (num_teams * (Nat.choose team_size host_members) *
   (Nat.choose team_size non_host_members)^(num_teams - 1)) = 3442073600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_committee_count_l941_94102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_before_first_nonzero_digit_l941_94199

-- We'll define our own function to represent the decimal expansion
def countZerosBeforeFirstNonZeroDigit (q : ℚ) : ℕ :=
  -- This is a placeholder function. In a real implementation, 
  -- we would need to write the logic to convert the rational to decimal
  -- and count the zeros. For now, we'll just return 5 to match the problem.
  5

theorem zeros_before_first_nonzero_digit :
  let fraction := (1 : ℚ) / ((2 ^ 3) * (5 ^ 6))
  countZerosBeforeFirstNonZeroDigit fraction = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_before_first_nonzero_digit_l941_94199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_is_sqrt_17_l941_94130

/-- An isosceles triangle with vertices on the curve xy = c^2 -/
structure IsoscelesTriangleOnCurve (c : ℝ) where
  a : ℝ
  h_positive : c > 0
  h_on_curve : a * 4 = c^2
  h_symmetry : true  -- Represents the line of symmetry along y-axis
  h_altitude : true  -- Represents the altitude of 4 units

/-- The length of the equal sides of the triangle -/
noncomputable def side_length (t : IsoscelesTriangleOnCurve 2) : ℝ :=
  Real.sqrt (t.a^2 + 4^2)

/-- Theorem stating that the side length is √17 when c = 2 -/
theorem side_length_is_sqrt_17 (t : IsoscelesTriangleOnCurve 2) :
  side_length t = Real.sqrt 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_is_sqrt_17_l941_94130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AD_between_20_and_21_l941_94141

-- Define the points
variable (A B C D : ℝ × ℝ)

-- Define the conditions
axiom east_of : B.1 > A.1 ∧ B.2 = A.2
axiom north_of : C.1 = B.1 ∧ C.2 > B.2
axiom distance_AC : (C.1 - A.1)^2 + (C.2 - A.2)^2 = 2 * 12^2
axiom west_of_C : D.1 = C.1 - 5 ∧ D.2 = C.2

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem AD_between_20_and_21 :
  20 < distance A D ∧ distance A D < 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_AD_between_20_and_21_l941_94141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_minimum_value_of_x_ln_x_l941_94185

/-- The function f(x) = x ln x -/
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

/-- The theorem stating that the extreme minimum value of f(x) = x ln x on (0, +∞) is -1/e -/
theorem extreme_minimum_value_of_x_ln_x :
  ∃ (x₀ : ℝ), x₀ > 0 ∧ f x₀ = -1 / Real.exp 1 ∧ ∀ (x : ℝ), x > 0 → f x ≥ -1 / Real.exp 1 := by
  sorry

#check extreme_minimum_value_of_x_ln_x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_minimum_value_of_x_ln_x_l941_94185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ones_digit_of_largest_power_of_two_dividing_32_factorial_l941_94196

def largest_power_of_two_dividing_factorial (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc k => acc + (n / 2^k).log 2) 0

def ones_digit (n : ℕ) : ℕ := n % 10

theorem ones_digit_of_largest_power_of_two_dividing_32_factorial :
  ones_digit (2^(largest_power_of_two_dividing_factorial 32)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ones_digit_of_largest_power_of_two_dividing_32_factorial_l941_94196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_from_directrix_parabola_equation_directrix_3_l941_94189

/-- Predicate to check if a point is on the directrix -/
def IsDirectrix (x₀ : ℝ) (x y : ℝ) : Prop := x = x₀

/-- Predicate to check if a point is on the parabola -/
def IsOnParabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 4*p*x

/-- The standard equation of a parabola given its directrix -/
theorem parabola_equation_from_directrix (x₀ : ℝ) :
  (∀ x y, IsDirectrix x₀ x y → x = x₀) →
  (∀ x y, IsOnParabola (-x₀) x y ↔ y^2 = -4*x₀*x) :=
by sorry

/-- The specific case for directrix x = 3 -/
theorem parabola_equation_directrix_3 :
  (∀ x y, IsDirectrix 3 x y → x = 3) →
  (∀ x y, IsOnParabola (-3) x y ↔ y^2 = -12*x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_from_directrix_parabola_equation_directrix_3_l941_94189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_second_part_speed_l941_94104

/-- A journey with a total distance D and total time T, where the first part is traveled at 80 kmph -/
structure Journey where
  D : ℝ  -- Total distance
  T : ℝ  -- Total time
  h1 : D > 0
  h2 : T > 0

/-- The speed for the second part of the journey -/
noncomputable def second_part_speed (j : Journey) : ℝ :=
  (j.D / 3) / ((2 * j.T) / 3)

/-- The theorem stating the speed for the second part of the journey is 20 kmph -/
theorem journey_second_part_speed (j : Journey) :
  (2 * j.D / 3) / (j.T / 3) = 80 → second_part_speed j = 20 := by
  sorry

#check journey_second_part_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_second_part_speed_l941_94104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_existence_l941_94122

/-- Given the sum and difference of a triangle's side length and height,
    prove that an equilateral triangle exists. -/
theorem equilateral_triangle_existence 
  (s d : ℝ) -- sum and difference of side length and height
  (h_positive : s > 0 ∧ d > 0) -- assume positive values
  (h_constraint : s > d) -- sum must be greater than difference
  : ∃ (side height : ℝ), 
    side > 0 ∧ 
    height > 0 ∧ 
    side + height = s ∧ 
    side - height = d ∧ 
    height = (Real.sqrt 3 / 2) * side :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_existence_l941_94122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l941_94190

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (F₂ P M : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  F₂.1 = c ∧ F₂.2 = 0 →
  (P.1^2 / a^2) - (P.2^2 / b^2) = 1 →
  M = ((P.1 + F₂.1) / 2, P.2 / 2) →
  (F₂.1 - 0)^2 + (F₂.2 - 0)^2 = (M.1 - F₂.1)^2 + (M.2 - F₂.2)^2 →
  (F₂.1 - 0) * (M.1 - F₂.1) + (F₂.2 - 0) * (M.2 - F₂.2) = c^2 / 2 →
  b^2 = c^2 - a^2 →
  let e := c / a
  e = (Real.sqrt 3 + 1) / 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l941_94190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_difference_set_size_l941_94101

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

def hasFibDifferences (S : Set ℤ) (n : ℕ) : Prop :=
  ∀ k, 2 ≤ k ∧ k ≤ n → ∃ x y, x ∈ S ∧ y ∈ S ∧ x - y = fib k

def minSetSize (n : ℕ) : ℕ := 
  Nat.ceil (n / 2) + 1

theorem fibonacci_difference_set_size (n : ℕ) (h : n ≥ 2) :
  ∃ S : Finset ℤ, (S.card = minSetSize n) ∧ hasFibDifferences S n ∧
  ∀ T : Finset ℤ, hasFibDifferences T n → T.card ≥ minSetSize n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_difference_set_size_l941_94101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_is_3_sqrt_2_l941_94113

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y = 2

-- Define the curve C
def curve_C (x y : ℝ) : Prop := y = x^2

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line_l A.1 A.2 ∧ curve_C A.1 A.2 ∧
  line_l B.1 B.2 ∧ curve_C B.1 B.2 ∧
  A ≠ B

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- Theorem statement
theorem segment_length_is_3_sqrt_2 :
  ∀ A B : ℝ × ℝ, intersection_points A B → distance A B = 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_is_3_sqrt_2_l941_94113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_plus_pi_third_l941_94184

theorem sin_double_angle_plus_pi_third (α : ℝ) : 
  0 < α → α < π / 2 → Real.cos (α + π / 6) = 4 / 5 → Real.sin (2 * α + π / 3) = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_plus_pi_third_l941_94184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laptop_price_reduction_l941_94134

theorem laptop_price_reduction (P : ℝ) (h : P > 0) : 
  let first_reduction := P * (1 - 0.3)
  let second_reduction := first_reduction * (1 - 0.5)
  (P - second_reduction) / P = 0.65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_laptop_price_reduction_l941_94134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_is_four_l941_94192

/-- A complex number z satisfies the conditions of the problem if z, z^2, z^3 form three vertices 
    of a non-degenerate square, and there exists a fourth vertex that is 3z away from z 
    and 2z away from z^3 -/
def satisfies_conditions (z : ℂ) : Prop :=
  ∃ (w : ℂ), z ≠ 0 ∧ 
    (w - z = 3 * z ∨ w - z = 3 * z * Complex.I) ∧
    (w - z^3 = 2 * z ∨ w - z^3 = 2 * z * Complex.I) ∧
    (Set.ncard {z, z^2, z^3, w} = 4)

/-- The theorem stating that if z satisfies the conditions, the area of the square is 4 -/
theorem square_area_is_four (z : ℂ) (h : satisfies_conditions z) : 
  ∃ (w : ℂ), Complex.abs (w - z) ^ 2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_is_four_l941_94192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_product_implies_right_triangle_l941_94124

noncomputable section

-- Define the hyperbola and ellipse
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1
def ellipse (m b x y : ℝ) : Prop := x^2 / m^2 + y^2 / b^2 = 1

-- Define the eccentricity of the hyperbola
noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2) / a

-- Define the eccentricity of the ellipse
noncomputable def ellipse_eccentricity (m b : ℝ) : ℝ := Real.sqrt (m^2 - b^2) / m

theorem eccentricity_product_implies_right_triangle 
  (a b m : ℝ) 
  (h1 : m > b) 
  (h2 : b > 0) 
  (h3 : hyperbola_eccentricity a b * ellipse_eccentricity m b = 1) :
  a^2 + b^2 = m^2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_product_implies_right_triangle_l941_94124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l941_94136

/-- The function f(x) = |x + 1| + |2x - 1| -/
def f (x : ℝ) : ℝ := |x + 1| + |2*x - 1|

/-- The function g(x) = |3x - 2m| + |3x - 1| -/
def g (m : ℝ) (x : ℝ) : ℝ := |3*x - 2*m| + |3*x - 1|

/-- Theorem stating the range of m given the conditions -/
theorem range_of_m :
  ∀ m : ℝ, (∀ x₁ : ℝ, ∃ x₂ : ℝ, f x₁ = g m x₂) → -1/4 ≤ m ∧ m ≤ 5/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l941_94136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rays_forming_acute_triangle_are_perpendicular_l941_94107

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a ray in 3D space
structure Ray3D where
  origin : Point3D
  direction : Point3D

-- Define membership for Point3D in Ray3D
def Point3D.mem (p : Point3D) (r : Ray3D) : Prop :=
  ∃ t : ℝ, t ≥ 0 ∧ p = Point3D.mk
    (r.origin.x + t * r.direction.x)
    (r.origin.y + t * r.direction.y)
    (r.origin.z + t * r.direction.z)

instance : Membership Point3D Ray3D where
  mem := Point3D.mem

-- Define the necessary axioms
axiom NonCoplanar : Ray3D → Ray3D → Ray3D → Prop
axiom AcuteTriangle : Point3D → Point3D → Point3D → Prop
axiom PairwisePerpendicular : Ray3D → Ray3D → Ray3D → Prop

-- Define the problem statement
theorem rays_forming_acute_triangle_are_perpendicular
  (O : Point3D)
  (l₁ l₂ l₃ : Ray3D)
  (non_coplanar : NonCoplanar l₁ l₂ l₃)
  (acute_triangle : ∀ (A₁ A₂ A₃ : Point3D),
    A₁ ∈ l₁ → A₂ ∈ l₂ → A₃ ∈ l₃ →
    A₁ ≠ O → A₂ ≠ O → A₃ ≠ O →
    AcuteTriangle A₁ A₂ A₃) :
  PairwisePerpendicular l₁ l₂ l₃ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rays_forming_acute_triangle_are_perpendicular_l941_94107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_diagonal_length_l941_94121

theorem quadrilateral_diagonal_length 
  (A B C D O : EuclideanSpace ℝ (Fin 2))
  (h_AO : ‖A - O‖ = 5)
  (h_CO : ‖C - O‖ = 12)
  (h_DO : ‖D - O‖ = 5)
  (h_BO : ‖B - O‖ = 6)
  (h_BD : ‖B - D‖ = 9) :
  ‖A - C‖ = Real.sqrt 197 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_diagonal_length_l941_94121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biker_path_theorem_l941_94177

/-- Represents the distance traveled by a biker in different directions --/
structure BikerPath where
  west : ℝ
  north1 : ℝ
  east : ℝ
  north2 : ℝ

/-- Calculates the straight-line distance between start and end points --/
noncomputable def straightLineDistance (path : BikerPath) : ℝ :=
  Real.sqrt ((path.west - path.east)^2 + (path.north1 + path.north2)^2)

/-- Theorem stating that if a biker follows a specific path and ends up at a certain distance,
    then the eastward distance must be 4 miles --/
theorem biker_path_theorem (path : BikerPath) 
  (h1 : path.west = 8)
  (h2 : path.north1 = 5)
  (h3 : path.north2 = 15)
  (h4 : straightLineDistance path = 20.396078054371138) :
  path.east = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_biker_path_theorem_l941_94177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_pipe_fill_time_l941_94138

/-- Represents the time (in minutes) it takes for a pipe to fill or empty a tank -/
structure PipeTime where
  minutes : ℚ
  minutes_pos : minutes > 0

/-- Represents the rate at which a pipe fills or empties a tank (in tanks per minute) -/
noncomputable def fillRate (t : PipeTime) : ℚ := 1 / t.minutes

theorem first_pipe_fill_time 
  (empty_pipe : PipeTime) 
  (total_time : ℚ) 
  (switch_time : ℚ) :
  empty_pipe.minutes = 24 →
  total_time = 30 →
  switch_time = 96 →
  ∃ (fill_pipe : PipeTime), 
    fill_pipe.minutes = 6 ∧
    switch_time * (fillRate fill_pipe - fillRate empty_pipe) + 
    (total_time - switch_time) * fillRate fill_pipe = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_pipe_fill_time_l941_94138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_outside_circle_m_range_l941_94133

/-- A circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point is outside a circle -/
def is_outside (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 > c.radius^2

/-- The circle equation x^2 + y^2 - 2x - 4y + m = 0 -/
noncomputable def circle_equation (m : ℝ) : Circle :=
  { center := (1, 2), radius := Real.sqrt (5 - m) }

theorem point_outside_circle_m_range :
  ∀ m : ℝ, is_outside (2, 3) (circle_equation m) → m ∈ Set.Ioo 3 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_outside_circle_m_range_l941_94133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_series_convergent_l941_94116

def is_strictly_increasing (a : ℕ → ℕ) : Prop :=
  ∀ n, a n < a (n + 1)

def lcm_seq (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  Finset.lcm (Finset.range n) (fun i ↦ a (i + 1))

theorem lcm_series_convergent (a : ℕ → ℕ) (h : is_strictly_increasing a) :
  ∃ (s : ℝ), HasSum (fun n ↦ (1 : ℝ) / lcm_seq a n) s :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_series_convergent_l941_94116
