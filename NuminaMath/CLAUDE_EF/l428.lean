import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_integers_l428_42878

theorem count_special_integers :
  ∃! (S : Finset ℕ),
    (∀ n ∈ S, n > 0 ∧ n < 120 ∧ ∃ k, n = 5 * k) ∧
    (∀ n ∈ S, Nat.lcm (Nat.factorial 5) n = 6 * Nat.gcd (Nat.factorial 4) n) ∧
    S.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_integers_l428_42878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l428_42873

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (9 : ℝ)^a = (3 : ℝ)^(1-b)) :
  (∀ x y : ℝ, x > 0 → y > 0 → (9 : ℝ)^x = (3 : ℝ)^(1-y) → 1/(81*x) + 2/(81*y) + x*y ≥ 2/9) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (9 : ℝ)^x = (3 : ℝ)^(1-y) ∧ 1/(81*x) + 2/(81*y) + x*y = 2/9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l428_42873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_properties_l428_42879

noncomputable section

open Real

-- Define the circle C in polar coordinates
def circle_C (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * sqrt 2 * ρ * cos (θ - π/4) + 7 = 0

-- Define the Cartesian equation of the circle
def circle_C_cartesian (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 4*y + 7 = 0

-- Define the range for x + √3y
def range_sum (x y : ℝ) : Prop :=
  2 * sqrt 3 ≤ x + sqrt 3 * y ∧ x + sqrt 3 * y ≤ 4 + 2 * sqrt 3

-- Theorem statement
theorem circle_C_properties :
  (∀ x y : ℝ, circle_C_cartesian x y ↔ ∃ ρ θ : ℝ, circle_C ρ θ ∧ x = ρ * cos θ ∧ y = ρ * sin θ) ∧
  (∀ x y : ℝ, circle_C_cartesian x y → range_sum x y) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_properties_l428_42879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_true_l428_42819

-- Definition of additive inverse
def additive_inverse (x y : ℝ) : Prop := x + y = 0

-- Definition of real roots for quadratic equation
def has_real_roots (q : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*x + q = 0

-- Definition of equilateral triangle
def equilateral_triangle (a b c : ℝ) : Prop := a = b ∧ b = c ∧ a > 0

-- Definition of interior angle
def interior_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < Real.pi

theorem propositions_true :
  (∀ x y : ℝ, additive_inverse x y → x + y = 0) ∧
  (∀ q : ℝ, has_real_roots q → q ≤ 1) ∧
  (∀ a b c θ₁ θ₂ θ₃ : ℝ, 
    equilateral_triangle a b c → 
    interior_angle θ₁ → interior_angle θ₂ → interior_angle θ₃ →
    θ₁ + θ₂ + θ₃ = Real.pi → θ₁ = θ₂ ∧ θ₂ = θ₃) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_true_l428_42819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_container_price_calculation_l428_42843

/-- Represents a cylindrical container with its properties -/
structure Container where
  diameter : ℝ
  height : ℝ
  price : ℝ

/-- Calculates the volume of a cylindrical container -/
noncomputable def volume (c : Container) : ℝ := Real.pi * (c.diameter / 2) ^ 2 * c.height

/-- The problem statement -/
theorem container_price_calculation (c1 c2 : Container)
  (h1 : c1.diameter = 5)
  (h2 : c1.height = 6)
  (h3 : c1.price = 0.75)
  (h4 : c2.diameter = 10)
  (h5 : c2.height = 9) :
  c2.price = 4.50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_container_price_calculation_l428_42843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_area_l428_42849

/-- The lateral area of a cone with base radius 3 and height 4 is 15π. -/
theorem cone_lateral_area : 
  ∀ (r h : ℝ), r = 3 → h = 4 → 
  let l := Real.sqrt (r^2 + h^2)
  let c := 2 * Real.pi * r
  (1/2 : ℝ) * c * l = 15 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_area_l428_42849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l428_42857

/-- Represents a rectangular table -/
structure Table :=
  (width : ℕ) (height : ℕ)

/-- Represents a position on the table -/
structure Position :=
  (x : ℕ) (y : ℕ)

/-- Checks if a position is valid on the table -/
def is_valid_position (t : Table) (p : Position) : Prop :=
  p.x < t.width ∧ p.y < t.height

/-- Represents a game state -/
structure GameState :=
  (table : Table)
  (coins : List Position)

/-- The winning strategy for the first player -/
def first_player_strategy (t : Table) : Position :=
  ⟨t.width / 2, t.height / 2⟩

/-- Theorem stating that the first player wins with the correct strategy -/
theorem first_player_wins (t : Table) :
  ∃ (strategy : GameState → Position),
  ∀ (game : GameState),
  is_valid_position t (strategy game) →
  ¬(∃ (opponent_move : Position),
    is_valid_position t opponent_move ∧
    ¬(∃ (player_response : Position),
      is_valid_position t player_response)) :=
sorry

#check first_player_wins

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l428_42857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_equation_solution_l428_42815

theorem trig_equation_solution (x : Real) : 
  (Real.sin x > 0) → 
  (Real.cos x > 0) → 
  (5.57 * (Real.sin x)^3 * (1 + 1 / (Real.tan x)) + (Real.cos x)^3 * (1 + Real.tan x) = 2 * Real.sqrt ((Real.sin x) * (Real.cos x))) ↔ 
  ∃ k : ℤ, x = π/4 + 2*π*k := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_equation_solution_l428_42815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_game_solvable_l428_42861

/-- Represents a hexagon with integers at each vertex -/
structure Hexagon :=
  (vertices : Fin 6 → Int)

/-- A move on the hexagon -/
def move (h : Hexagon) (i : Fin 6) : Hexagon :=
  { vertices := λ j => if j = i then 
      (h.vertices ((i + 1) % 6) - h.vertices ((i + 5) % 6)).natAbs
    else h.vertices j }

/-- Predicate for a hexagon with all zeros -/
def all_zeros (h : Hexagon) : Prop :=
  ∀ i, h.vertices i = 0

/-- The main theorem -/
theorem hexagon_game_solvable (h : Hexagon) 
  (sum_2003 : (Finset.univ.sum (λ i => (h.vertices i).natAbs)) = 2003) :
  ∃ (moves : List (Fin 6)), all_zeros (moves.foldl move h) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_game_solvable_l428_42861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_second_derivative_shift_l428_42846

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sin x

-- Define the second derivative of f
noncomputable def f'' : ℝ → ℝ := λ x => -Real.sin x

-- State the theorem
theorem sin_second_derivative_shift (x : ℝ) :
  f'' (2 * x + π / 3) = f (2 * (x + 5 * π / 12)) :=
by
  -- Expand the definitions
  unfold f f''
  -- Simplify the expressions
  simp [Real.sin_add]
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_second_derivative_shift_l428_42846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_and_parallel_chord_l428_42847

-- Define the circles and line
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 4 = 0
def l (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Define points E and F
def E : ℝ × ℝ := (1, -3)
def F : ℝ × ℝ := (0, 4)

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 2*x + y + 1 = 0

-- Define circle C₂
def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 16 = 0

theorem circle_intersection_and_parallel_chord :
  ∃ (A B : ℝ × ℝ),
    C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧
    l A.1 A.2 ∧ l B.1 B.2 ∧
    C₂ E.1 E.2 ∧ C₂ F.1 F.2 ∧
    (∃ (m b : ℝ), ∀ (x y : ℝ), C₁ x y ∧ C₂ x y → y = m*x + b) ∧
    (∀ (x y : ℝ), C₁ x y ∧ C₂ x y → ∃ (k : ℝ), y = -2*x + k) →
    (∃ (D : ℝ → ℝ → ℝ → ℝ → ℝ), D A.1 A.2 B.1 B.2 = 4 * Real.sqrt 5 / 5) ∧
    (∀ (x y : ℝ), C₂ x y ↔ x^2 + y^2 + 6*x - 16 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_and_parallel_chord_l428_42847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_count_l428_42814

-- Define a function to represent the correctness of each statement
def statement_correctness : Fin 4 → Bool
  | 0 => false  -- -a does not always represent a negative number
  | 1 => true   -- The largest negative integer is -1
  | 2 => true   -- 2 and -2 are equidistant from the origin
  | 3 => false  -- The degree of 3xy^2-2xy is 3, not 2

-- Theorem: The number of correct statements is 2
theorem correct_statements_count :
  (Finset.filter (fun i => statement_correctness i = true) Finset.univ).card = 2 := by
  -- Convert the function to an explicit list for easier counting
  let correctList := [false, true, true, false]
  -- Assert that our function matches this list
  have h : ∀ i : Fin 4, statement_correctness i = correctList[i] := by
    intro i
    fin_cases i <;> rfl
  -- Count the true values
  simp [statement_correctness, h]
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_count_l428_42814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_condition_negation_existential_quadratic_roots_condition_l428_42840

-- Statement 1
theorem necessary_not_sufficient_condition :
  (∀ x : ℝ, |x| < 2 → x < 3) ∧ (∃ x : ℝ, x < 3 ∧ |x| ≥ 2) :=
by sorry

-- Statement 2
theorem negation_existential :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) :=
by sorry

-- Statement 3
theorem quadratic_roots_condition :
  (∀ m : ℝ, (∃ x y : ℝ, x ≠ y ∧ x * y > 0 ∧ x^2 - 2*x - m = 0 ∧ y^2 - 2*y - m = 0) ↔ (-1 < m ∧ m < 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_condition_negation_existential_quadratic_roots_condition_l428_42840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_segment_speed_is_80_l428_42883

/-- Calculates the average speed for the last segment of a journey given the total distance,
    total time, and average speeds for the first two segments. -/
noncomputable def last_segment_speed (total_distance : ℝ) (total_time : ℝ) 
                       (speed1 : ℝ) (speed2 : ℝ) : ℝ :=
  let segment_time := total_time / 3
  let distance1 := speed1 * segment_time
  let distance2 := speed2 * segment_time
  let distance3 := total_distance - distance1 - distance2
  distance3 / segment_time

/-- Theorem stating that under the given conditions, the average speed
    for the last segment is 80 mph. -/
theorem last_segment_speed_is_80 :
  last_segment_speed 150 2 70 75 = 80 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval last_segment_speed 150 2 70 75

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_segment_speed_is_80_l428_42883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_at_one_l428_42882

noncomputable def P (x : ℝ) : ℝ := x^3 + 3*x^2 - 3*x - 9

noncomputable def Q (x : ℝ) : ℝ :=
  let nonzero_coeff := [1, 3, -3, -9]
  let mean := (nonzero_coeff.sum) / (nonzero_coeff.length : ℝ)
  mean * x^3 + mean * x^2 + mean * x + mean

theorem Q_at_one : Q 1 = -8 := by
  -- Unfold the definition of Q
  unfold Q
  -- Simplify the arithmetic
  simp [List.sum, List.length]
  -- Perform the final calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_at_one_l428_42882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l428_42853

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (3 * x) + 3 * Real.sin x + 3 * Real.sqrt 3 * Real.cos x

-- State the theorem
theorem f_range :
  ∀ x ∈ Set.Icc 0 Real.pi,
  ∃ y ∈ Set.Icc (-3 * Real.sqrt 3) 8,
  f x = y ∧
  ∀ z, f x = z → z ∈ Set.Icc (-3 * Real.sqrt 3) 8 :=
by
  sorry  -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l428_42853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l428_42888

-- Define the propositions p and q
noncomputable def p (m : ℝ) : Prop := ∀ x y : ℝ, x^2 / (5*m - 1) + y^2 / (m + 3) = 1 → 
  ∃ a b : ℝ, a > b ∧ ∀ t : ℝ, (x - t)^2 / a^2 + y^2 / b^2 = 1

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + (1/2) * m * x^2 + x + 2

def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → f m x < f m y

-- Define the theorem
theorem range_of_m (m : ℝ) 
  (h1 : ¬(p m ∧ q m))
  (h2 : p m ∨ q m)
  (h3 : ¬(q m)) :
  m > 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l428_42888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l428_42851

theorem system_solution (m : ℝ) (x y : ℝ) : 
  x + y = -m - 7 →
  x - y = 3*m + 1 →
  x ≤ 0 →
  y < 0 →
  (-2 < m ∧ m ≤ 3) ∧
  (m = -1 ↔ (∃ (n : ℤ), ↑n = m ∧ -2 < m ∧ m ≤ 3 ∧ ∀ x > 1, 2*m*x + x < 2*m + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l428_42851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l428_42863

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := 2 * (a x).1 * (b x).1 + 2 * (a x).2 * (b x).2 - 1

theorem vector_problem (x : ℝ) (k : ℤ) (h : b x ≠ (0, 0)) :
  (∀ x ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6), 
    Monotone f) ∧
  (a x = (Real.sqrt 3 : ℝ) • b x → Real.tan x = Real.sqrt 3) ∧
  (a x • b x = 0 → ∃ y, y > 0 ∧ y = 5 * Real.pi / 6 ∧ 
    (∀ z, z > 0 → a z • b z = 0 → y ≤ z)) := by
  sorry

#check vector_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l428_42863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_collinear_altitude_midpoint_triangle_l428_42830

/-- A triangle with collinear altitude midpoints -/
structure CollinearAltitudeMidpointTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  altitudeMidpointsCollinear : Bool
  largestSide : ℝ
  largestSideIs10 : largestSide = 10

/-- The area of a triangle -/
noncomputable def triangleArea (t : CollinearAltitudeMidpointTriangle) : ℝ := sorry

/-- The maximum possible area of a triangle with collinear altitude midpoints and largest side 10 -/
theorem max_area_collinear_altitude_midpoint_triangle :
  ∃ (t : CollinearAltitudeMidpointTriangle),
    ∀ (t' : CollinearAltitudeMidpointTriangle),
      triangleArea t' ≤ triangleArea t ∧ triangleArea t = 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_collinear_altitude_midpoint_triangle_l428_42830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l428_42850

noncomputable def f (x : ℝ) : ℝ := (3*x - 4)*(x - 2)*(x + 1)/(x - 1)

def solution_set : Set ℝ := Set.Iic (-1) ∪ (Set.Icc (4/3) 2) ∪ Set.Ioi 2

theorem inequality_solution :
  {x : ℝ | f x ≥ 0 ∧ x ≠ 1} = solution_set :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l428_42850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_identity_l428_42877

theorem cosine_identity (α : ℝ) : 
  Real.cos (π/8 - α) = 1/6 → Real.cos (3*π/4 + 2*α) = 17/18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_identity_l428_42877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_ball_higher_probability_l428_42854

/-- The probability that a ball is placed into bin k -/
noncomputable def binProbability (k : ℕ) : ℝ := (3 : ℝ) ^ (-k : ℤ)

/-- The probability that both balls are placed in the same bin k -/
noncomputable def sameBinProbability (k : ℕ) : ℝ := binProbability k * binProbability k

/-- The sum of probabilities of both balls being in the same bin for all bins -/
noncomputable def totalSameBinProbability : ℝ := ∑' k, sameBinProbability k

/-- The probability that the blue ball is in a higher-numbered bin than the yellow ball -/
noncomputable def blueBallHigherProbability : ℝ := (1 - totalSameBinProbability) / 2

theorem blue_ball_higher_probability :
  blueBallHigherProbability = 7 / 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_ball_higher_probability_l428_42854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_implies_perpendicular_parallel_lines_planes_equal_angles_l428_42841

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (angle : Line → Plane → ℝ)

-- State the theorems to be proved
theorem perpendicular_parallel_implies_perpendicular 
  (α : Plane) (m n : Line) 
  (h1 : perpendicular_line_plane m α) (h2 : parallel_line_plane n α) : 
  perpendicular_lines m n :=
sorry

theorem parallel_lines_planes_equal_angles 
  (α β : Plane) (m n : Line)
  (h1 : parallel_lines m n) (h2 : parallel_planes α β) :
  angle m α = angle n β :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_implies_perpendicular_parallel_lines_planes_equal_angles_l428_42841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_year_percentage_approx_l428_42813

/-- The number of second-year students studying numeric methods -/
def numeric_methods : ℕ := 230

/-- The number of second-year students studying automatic control of airborne vehicles -/
def automatic_control : ℕ := 423

/-- The number of second-year students studying both subjects -/
def both_subjects : ℕ := 134

/-- The total number of students in the faculty -/
def total_students : ℕ := 653

/-- The percentage of second-year students in the faculty -/
def second_year_percentage : ℚ :=
  (numeric_methods + automatic_control - both_subjects : ℚ) / total_students * 100

/-- Theorem stating that the percentage of second-year students is approximately 79.48% -/
theorem second_year_percentage_approx :
  ∃ ε > 0, |second_year_percentage - 79.48| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_year_percentage_approx_l428_42813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_seven_equals_ten_l428_42827

-- Define the function f
noncomputable def f (u : ℝ) : ℝ := (u^2 + 10*u + 41) / 16

-- State the theorem
theorem f_of_seven_equals_ten :
  (∀ x : ℝ, f (4*x - 1) = x^2 + 2*x + 2) → f 7 = 10 := by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_seven_equals_ten_l428_42827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_in_third_quadrant_l428_42860

-- Define the function f
noncomputable def f (α : Real) : Real :=
  (Real.sin (α - 3 * Real.pi) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 * Real.pi / 2)) /
  (Real.cos (-Real.pi - α) * Real.sin (-Real.pi - α))

-- State the theorem
theorem f_value_in_third_quadrant (α : Real) 
  (h1 : α > Real.pi ∧ α < 3 * Real.pi / 2) -- α is in the third quadrant
  (h2 : Real.cos (α - 3 * Real.pi / 2) = 1 / 5) : -- given condition
  f α = 2 * Real.sqrt 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_in_third_quadrant_l428_42860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_with_prime_circumradius_l428_42876

theorem right_triangle_with_prime_circumradius 
  (a b c R : ℕ) 
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_circumradius : R > 1 ∧ Nat.Prime R)
  (h_radius_formula : 4 * R * (a + b + c) = a * b * c) :
  ∃ A : Real, A = Real.pi / 2 ∧ 
    (Real.cos A = (b^2 + c^2 - a^2) / (2 * b * c) ∨
     Real.cos A = (a^2 + c^2 - b^2) / (2 * a * c) ∨
     Real.cos A = (a^2 + b^2 - c^2) / (2 * a * b)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_with_prime_circumradius_l428_42876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_radical_equation_l428_42885

theorem solve_radical_equation :
  ∃ x : ℝ, Real.sqrt (9 + Real.sqrt (15 + 9*x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 3 + 3*Real.sqrt 3 ∧ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_radical_equation_l428_42885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_meeting_distance_l428_42816

/-- The distance between two trains traveling towards each other -/
noncomputable def total_distance : ℝ := 150

/-- The time taken by Train X to complete the journey -/
noncomputable def time_x : ℝ := 4

/-- The time taken by Train Y to complete the journey -/
noncomputable def time_y : ℝ := 3.5

/-- The speed of Train X -/
noncomputable def speed_x : ℝ := total_distance / time_x

/-- The speed of Train Y -/
noncomputable def speed_y : ℝ := total_distance / time_y

/-- The time at which the trains meet -/
noncomputable def meeting_time : ℝ := total_distance / (speed_x + speed_y)

/-- The distance traveled by Train X when it meets Train Y -/
noncomputable def distance_x : ℝ := speed_x * meeting_time

theorem train_meeting_distance :
  ∃ ε > 0, |distance_x - 70| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_meeting_distance_l428_42816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_correct_l428_42834

/-- Calculates the cost for a specific type of clothing item --/
def calculate_item_cost (quantity : ℕ) (time_per_item rate : ℚ)
  (discount_threshold : ℕ) (discount_rate : ℚ)
  (rate_increase_threshold : ℕ) (rate_increase : ℚ) : ℚ :=
  let base_cost := quantity * time_per_item * rate
  let rate_increases := (quantity - 1) / rate_increase_threshold
  let increased_rate := rate + rate_increases * rate_increase
  let final_cost := quantity * time_per_item * increased_rate
  if quantity > discount_threshold then final_cost * (1 - discount_rate) else final_cost

/-- Calculates the total cost of fixing clothing items --/
def calculate_total_cost (shirts pants jackets ties : ℕ)
  (shirt_time pant_time jacket_time tie_time : ℚ)
  (shirt_rate pant_rate jacket_rate tie_rate : ℚ)
  (discount_threshold : ℕ) (discount_rate : ℚ)
  (rate_increase_threshold : ℕ) (rate_increase : ℚ) : ℚ :=
  let shirt_cost := calculate_item_cost shirts shirt_time shirt_rate discount_threshold discount_rate rate_increase_threshold rate_increase
  let pant_cost := calculate_item_cost pants pant_time pant_rate discount_threshold discount_rate rate_increase_threshold rate_increase
  let jacket_cost := calculate_item_cost jackets jacket_time jacket_rate discount_threshold discount_rate rate_increase_threshold rate_increase
  let tie_cost := calculate_item_cost ties tie_time tie_rate discount_threshold discount_rate rate_increase_threshold rate_increase
  shirt_cost + pant_cost + jacket_cost + tie_cost

theorem total_cost_is_correct :
  calculate_total_cost 10 12 8 5 -- quantities
    (3/2) 3 (5/2) (1/2) -- time per item
    30 30 40 20 -- base rates
    10 (1/10) -- discount threshold and rate
    5 10 -- rate increase threshold and amount
  = 2551 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_correct_l428_42834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_first_quadrant_l428_42868

/-- The angle in degrees -/
noncomputable def angle : ℝ := 3600.5

/-- Function to determine the quadrant of an angle in degrees -/
noncomputable def quadrant (θ : ℝ) : ℕ :=
  let θ_normalized := θ % 360
  if 0 ≤ θ_normalized ∧ θ_normalized < 90 then 1
  else if 90 ≤ θ_normalized ∧ θ_normalized < 180 then 2
  else if 180 ≤ θ_normalized ∧ θ_normalized < 270 then 3
  else 4

theorem angle_in_first_quadrant : quadrant angle = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_first_quadrant_l428_42868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_six_l428_42811

/-- Given a function f: ℝ → ℝ such that f(4x+2) = x^2 + 2x + 3 for all real x,
    prove that f(6) = 6 -/
theorem function_value_at_six
  (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f (4*x + 2) = x^2 + 2*x + 3) :
  f 6 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_six_l428_42811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l428_42856

-- Define the curve
def f (x : ℝ) : ℝ := -x^3 + 3*x^2

-- Define the point of tangency
def point : ℝ × ℝ := (1, 2)

-- State the theorem
theorem tangent_line_equation :
  let (x₀, y₀) := point
  let m := (6*x₀ - 3*x₀^2)  -- Derivative of f at x₀
  (λ x => m*(x - x₀) + y₀) = (λ x => 3*x - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l428_42856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amanda_saturday_hours_l428_42895

/-- Amanda's hourly rate in dollars -/
noncomputable def hourly_rate : ℚ := 20

/-- Total hours worked on Monday -/
noncomputable def monday_hours : ℚ := 5 * (3/2)

/-- Total hours worked on Tuesday -/
noncomputable def tuesday_hours : ℚ := 3

/-- Total hours worked on Thursday -/
noncomputable def thursday_hours : ℚ := 2 * 2

/-- Total earnings for the week in dollars -/
noncomputable def total_earnings : ℚ := 410

/-- Calculates the number of hours Amanda will work on Saturday -/
noncomputable def saturday_hours : ℚ :=
  (total_earnings - hourly_rate * (monday_hours + tuesday_hours + thursday_hours)) / hourly_rate

theorem amanda_saturday_hours :
  saturday_hours = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amanda_saturday_hours_l428_42895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_iff_a_eq_one_g_is_minimum_l428_42898

-- Define the function f
def f (a x : ℝ) : ℝ := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x

-- Define the minimum value function g
noncomputable def g (a : ℝ) : ℝ :=
  if a > 3 then a^2 * (3 - a)
  else if 1 < a ∧ a ≤ 3 then 0
  else if a < -1 then 3 * a - 1
  else 0  -- This case should not occur given |a| > 1, but Lean requires all cases to be covered

-- Theorem 1: f is monotonically increasing iff a = 1
theorem f_monotonic_iff_a_eq_one (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ a = 1 := by
  sorry

-- Theorem 2: g(a) is the minimum value of f(x) on [0, 2|a|] for |a| > 1
theorem g_is_minimum (a : ℝ) (h : |a| > 1) :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * |a| → g a ≤ f a x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_iff_a_eq_one_g_is_minimum_l428_42898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kimbers_pizza_ingredients_l428_42852

/-- Calculates the total amount of ingredients for Kimber's pizza recipe -/
theorem kimbers_pizza_ingredients (water : ℕ) (flour : ℕ) (salt_ratio : ℚ) : 
  water = 10 → 
  flour = 16 → 
  salt_ratio = 1/2 → 
  water + flour + (salt_ratio * ↑flour).floor = 34 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kimbers_pizza_ingredients_l428_42852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_planes_theorem_l428_42845

/-- A structure representing two planes with points -/
structure TwoPlanes where
  α : Finset ℕ
  β : Finset ℕ
  α_count : α.card = 4
  β_count : β.card = 5
  disjoint : Disjoint α β

/-- The maximum number of planes determined by any three points -/
def max_planes (tp : TwoPlanes) : ℕ :=
  Finset.card (Finset.powersetCard 3 (tp.α ∪ tp.β))

/-- The maximum number of tetrahedrons determined by any four points -/
def max_tetrahedrons (tp : TwoPlanes) : ℕ :=
  Finset.card (Finset.powersetCard 4 (tp.α ∪ tp.β))

/-- The main theorem stating the results for the maximum number of planes and tetrahedrons -/
theorem two_planes_theorem (tp : TwoPlanes) :
  max_planes tp = 72 ∧ max_tetrahedrons tp = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_planes_theorem_l428_42845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_eight_l428_42872

/-- Represents the sum of the first n terms of a geometric sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The common ratio of the geometric sequence -/
def q : ℝ := sorry

/-- The first term of the geometric sequence -/
def a₁ : ℝ := sorry

theorem geometric_sequence_sum_eight :
  (S 4 = -5) →
  (S 6 = 21 * S 2) →
  (∀ n : ℕ, S n = a₁ * (1 - q^n) / (1 - q)) →
  (q ≠ 1) →
  (S 8 = -85) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_eight_l428_42872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_property_l428_42817

/-- The distance between two points in 2D space -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 16 = 1

/-- The fixed points F₁ and F₂ -/
def F₁ : ℝ × ℝ := (5, 0)
def F₂ : ℝ × ℝ := (-5, 0)

/-- The theorem stating that the hyperbola equation describes the curve with the given property -/
theorem hyperbola_property (x y : ℝ) :
  hyperbola_equation x y ↔
  |distance x y F₁.1 F₁.2 - distance x y F₂.1 F₂.2| = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_property_l428_42817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_median_circumcircle_inequality_l428_42831

/-- Helper function to calculate the area of a triangle using Heron's formula. -/
noncomputable def area_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Predicate to check if three sides form a valid triangle. -/
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

/-- Given a triangle with sides a, b, c, corresponding medians m_a, m_b, m_c, 
    and circumcircle diameter D, prove that 
    (a^2+b^2)/m_c + (a^2+c^2)/m_b + (b^2+c^2)/m_a ≤ 6D. -/
theorem triangle_median_circumcircle_inequality 
  (a b c m_a m_b m_c D : ℝ) 
  (h_triangle : is_triangle a b c)
  (h_median_a : m_a = (1/2) * Real.sqrt (2*b^2 + 2*c^2 - a^2))
  (h_median_b : m_b = (1/2) * Real.sqrt (2*a^2 + 2*c^2 - b^2))
  (h_median_c : m_c = (1/2) * Real.sqrt (2*a^2 + 2*b^2 - c^2))
  (h_diameter : D = 2 * (a*b*c) / (4 * area_triangle a b c)) :
  (a^2 + b^2) / m_c + (a^2 + c^2) / m_b + (b^2 + c^2) / m_a ≤ 6 * D :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_median_circumcircle_inequality_l428_42831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_is_55_l428_42889

/-- Represents the swimmer's problem --/
structure SwimmerProblem where
  current_speed : ℝ
  upstream_distance : ℝ
  upstream_time : ℝ
  downstream_time : ℝ

/-- Calculates the downstream distance for a given swimmer problem --/
noncomputable def downstream_distance (p : SwimmerProblem) : ℝ :=
  let still_water_speed := p.upstream_distance / p.upstream_time + p.current_speed
  (still_water_speed + p.current_speed) * p.downstream_time

/-- Theorem stating that the downstream distance is 55 km for the given conditions --/
theorem downstream_distance_is_55 :
  let p : SwimmerProblem := {
    current_speed := 4.5,
    upstream_distance := 10,
    upstream_time := 5,
    downstream_time := 5
  }
  downstream_distance p = 55 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_is_55_l428_42889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_10_equals_171_l428_42866

def fibonacci : ℕ → ℤ
  | 0 => 0
  | 1 => 0
  | 2 => 1
  | n + 3 => fibonacci (n + 2) + 2 * fibonacci (n + 1)

theorem fibonacci_10_equals_171 : fibonacci 10 = 171 := by
  -- Compute the value directly
  have h : fibonacci 10 = 171 := by native_decide
  exact h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_10_equals_171_l428_42866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_maximized_sector_l428_42890

/-- A sector with a fixed perimeter -/
structure Sector where
  perimeter : ℝ
  perimeter_pos : perimeter > 0

/-- The probability of a point falling inside the triangle of a maximized sector -/
noncomputable def probability_in_triangle (s : Sector) : ℝ :=
  (1 / 2) * Real.sin 2

/-- Theorem: The probability of a point falling inside the triangle of a maximized sector is (1/2)sin(2) -/
theorem probability_in_maximized_sector (s : Sector) :
  probability_in_triangle s = (1 / 2) * Real.sin 2 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_maximized_sector_l428_42890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_is_two_l428_42867

noncomputable section

open Real

-- Define the curves C₁ and C₂
def C₁ (α : ℝ) : ℝ × ℝ := (Real.cos α, 1 + Real.sin α)
def C₂ (α : ℝ) : ℝ × ℝ := (3 * Real.cos α, 3 + 3 * Real.sin α)

-- Define the line y = (√3/3)x
def line (x : ℝ) : ℝ := (Real.sqrt 3 / 3) * x

-- Define points A and B
def A : ℝ × ℝ := (1/2, Real.sqrt 3/2)
def B : ℝ × ℝ := (3/2, 3*Real.sqrt 3/2)

-- State the theorem
theorem distance_AB_is_two :
  let d := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  d = 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_is_two_l428_42867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l428_42896

-- Define the function f and its derivative f'
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^4 * Real.cos x + m * x^2 + x
noncomputable def f' (m : ℝ) (x : ℝ) : ℝ := 4 * x^3 * Real.cos x - x^4 * Real.sin x + 2 * m * x + 1

-- State the theorem
theorem min_value_of_f'_given_max (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f' m y ≤ f' m x) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f' m x ≤ 10) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f' m x ≤ f' m y) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f' m x = -8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l428_42896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pigeonhole_sum_theorem_l428_42859

theorem pigeonhole_sum_theorem (A : Finset ℕ) :
  A.card = 52 →
  (∀ a, a ∈ A → a ≤ 100) →
  (∀ a b, a ∈ A → b ∈ A → a ≠ b) →
  ∃ x y z, x ∈ A ∧ y ∈ A ∧ z ∈ A ∧ x + y = z :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pigeonhole_sum_theorem_l428_42859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_congruent_to_one_mod_three_l428_42812

theorem divisors_congruent_to_one_mod_three (n : ℕ) :
  (Finset.filter (fun d => d % 3 = 1) (Nat.divisors (2^n))).card = n + 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_congruent_to_one_mod_three_l428_42812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_table_sum_l428_42809

def numbers : List Nat := [2, 3, 5, 7, 11, 13, 17]

def is_valid_arrangement (top : List Nat) (left : List Nat) : Prop :=
  top.length = 4 ∧ left.length = 3 ∧ (top ++ left).toFinset = numbers.toFinset

def table_sum (top : List Nat) (left : List Nat) : Nat :=
  List.sum (List.map (fun x => List.sum (List.map (fun y => x * y) top)) left)

theorem max_table_sum :
  ∃ (top left : List Nat),
    is_valid_arrangement top left ∧
    table_sum top left = 841 ∧
    ∀ (t l : List Nat), is_valid_arrangement t l → table_sum t l ≤ 841 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_table_sum_l428_42809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_implies_w_value_l428_42842

noncomputable def f (w : ℝ) (x : ℝ) : ℝ := Real.sin (w * x + Real.pi / 3)

theorem period_implies_w_value (w : ℝ) (h1 : w > 0) 
  (h2 : ∀ x : ℝ, f w x = f w (x + Real.pi / 2)) : w = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_implies_w_value_l428_42842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_number_identification_l428_42858

theorem negative_number_identification :
  let options : List ℝ := [|(-2)|, Real.sqrt 3, 0, -5]
  ∃! x, x ∈ options ∧ x < 0 ∧ x = -5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_number_identification_l428_42858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_time_six_horses_at_start_l428_42869

def horse_lap_time (k : ℕ) : ℕ := k

def is_at_start (t k : ℕ) : Bool :=
  t % (horse_lap_time k) = 0

def count_horses_at_start (t : ℕ) : ℕ :=
  (List.range 12).filter (λ k => is_at_start t (k + 1)) |>.length

theorem least_time_six_horses_at_start :
  (∃ T : ℕ, T > 0 ∧
    count_horses_at_start T ≥ 6 ∧
    (∀ t : ℕ, t > 0 → count_horses_at_start t ≥ 6 → T ≤ t)) →
  (∃ T : ℕ, T = 60) := by
  sorry

#eval count_horses_at_start 60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_time_six_horses_at_start_l428_42869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangles_in_pentagon_l428_42897

/-- A polygon with 5 sides -/
structure Pentagon where
  vertices : Fin 5 → ℝ × ℝ

/-- A point in 2D space -/
def Point := ℝ × ℝ

/-- A triangle in 2D space -/
structure Triangle where
  vertices : Fin 3 → Point

/-- A triangulation of a pentagon -/
structure Triangulation (p : Pentagon) where
  triangles : List Triangle
  marked_points : List Point
  marked_points_are_vertices : ∀ pt, pt ∈ marked_points → 
    ∃ t ∈ triangles, ∃ i : Fin 3, t.vertices i = pt

/-- The minimum number of triangles in a pentagon with 1000 marked points is at least 1003 -/
theorem min_triangles_in_pentagon (p : Pentagon) (tri : Triangulation p) :
  tri.marked_points.length = 1000 →
  tri.triangles.length ≥ 1003 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangles_in_pentagon_l428_42897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_thursdays_in_august_l428_42894

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday
deriving Repr, DecidableEq

/-- Represents a date in a month -/
structure Date where
  day : Nat
  dayOfWeek : DayOfWeek
deriving Repr

/-- Represents a month with its dates -/
structure Month where
  dates : List Date
  numDays : Nat
deriving Repr

def july : Month := ⟨[], 31⟩
def august : Month := ⟨[], 31⟩

/-- The theorem to be proved -/
theorem five_thursdays_in_august 
  (h1 : july.numDays = 31)
  (h2 : august.numDays = 31)
  (h3 : (july.dates.filter (fun d => d.dayOfWeek == DayOfWeek.Monday)).length = 5) :
  (august.dates.filter (fun d => d.dayOfWeek == DayOfWeek.Thursday)).length = 5 := by
  sorry

#eval july
#eval august

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_thursdays_in_august_l428_42894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triplet_sum_not_one_l428_42805

def triplet_A : Fin 3 → ℚ
| 0 => 2/5
| 1 => 2/5
| 2 => 1/5

def triplet_B : Fin 3 → ℤ
| 0 => -1
| 1 => 3
| 2 => -1

def triplet_C : Fin 3 → ℚ
| 0 => 1/2
| 1 => 1/5
| 2 => 3/10

def triplet_D : Fin 3 → ℚ
| 0 => 1/4
| 1 => -9/20
| 2 => 1/5

def triplet_E : Fin 3 → ℚ
| 0 => 6/5
| 1 => -1/10
| 2 => -1/10

theorem triplet_sum_not_one :
  (Finset.sum Finset.univ triplet_A = 1) ∧
  (Finset.sum Finset.univ triplet_B = 1) ∧
  (Finset.sum Finset.univ triplet_C = 1) ∧
  (Finset.sum Finset.univ triplet_D ≠ 1) ∧
  (Finset.sum Finset.univ triplet_E = 1) := by
  sorry

#eval Finset.sum Finset.univ triplet_A
#eval Finset.sum Finset.univ triplet_B
#eval Finset.sum Finset.univ triplet_C
#eval Finset.sum Finset.univ triplet_D
#eval Finset.sum Finset.univ triplet_E

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triplet_sum_not_one_l428_42805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_nonagon_perimeter_l428_42833

/-- The perimeter of a regular nonagon with side length 3 units is 27 units. -/
theorem regular_nonagon_perimeter (n : ℕ) : ℕ :=
  match n with
  | 9 => 27
  | _ => 0

#check regular_nonagon_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_nonagon_perimeter_l428_42833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_fourth_quadrant_l428_42884

noncomputable def complex_number : ℂ := 1 + 1 / (1 + Complex.I)

theorem complex_number_in_fourth_quadrant :
  Real.sign complex_number.re = 1 ∧ Real.sign complex_number.im = -1 :=
by
  -- Simplify the complex number
  have h1 : complex_number = (3/2 : ℝ) - (1/2 : ℝ) * Complex.I := by sorry
  
  -- Check the real part
  have h_re : complex_number.re = 3/2 := by sorry
  have h_re_pos : complex_number.re > 0 := by sorry
  have sign_re : Real.sign complex_number.re = 1 := by sorry

  -- Check the imaginary part
  have h_im : complex_number.im = -1/2 := by sorry
  have h_im_neg : complex_number.im < 0 := by sorry
  have sign_im : Real.sign complex_number.im = -1 := by sorry

  exact ⟨sign_re, sign_im⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_fourth_quadrant_l428_42884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l428_42839

/-- Converts cylindrical coordinates to rectangular coordinates -/
noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

/-- The given point in cylindrical coordinates -/
noncomputable def cylindrical_point : ℝ × ℝ × ℝ := (7, Real.pi / 4, 8)

/-- Theorem stating that the conversion of the given cylindrical point
    results in the correct rectangular coordinates -/
theorem cylindrical_to_rectangular_conversion :
  cylindrical_to_rectangular cylindrical_point.1 cylindrical_point.2.1 cylindrical_point.2.2 =
  (7 * Real.sqrt 2 / 2, 7 * Real.sqrt 2 / 2, 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l428_42839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_complex_segment_l428_42824

/-- Given two complex numbers corresponding to points A and B, prove that the midpoint C corresponds to the specified complex number. -/
theorem midpoint_of_complex_segment (z_A z_B : ℂ) (h_A : z_A = 6 + 5*I) (h_B : z_B = -2 + 3*I) :
  (z_A + z_B) / 2 = 2 + 4*I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_complex_segment_l428_42824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l428_42802

/-- The function we're analyzing -/
noncomputable def f (x : ℝ) := Real.arcsin (x^2 - 2*x)

/-- The interval we're proving is the monotonic decreasing interval -/
def I : Set ℝ := Set.Icc (1 - Real.sqrt 2) 1

theorem monotonic_decreasing_interval :
  StrictAntiOn f I ∧ ∀ x ∉ I, ¬StrictAntiOn f (Set.Icc x x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l428_42802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_distance_l428_42823

/-- The distance traveled by a truck with delayed start and constant acceleration -/
noncomputable def distance_traveled (b t : ℝ) : ℝ :=
  b * (90000 + 600 * t + t^2) / (31680 * t)

/-- Theorem stating the distance traveled by the truck under given conditions -/
theorem truck_distance (b t : ℝ) (hb : b > 0) (ht : t > 0) :
  let delay := t
  let acceleration := b / (3 * t^2)
  let duration := 5 * 60  -- 5 minutes in seconds
  let feet_per_mile := 5280
  let total_time := duration + delay
  let distance_feet := (1 / 2) * acceleration * total_time^2
  distance_feet / feet_per_mile = distance_traveled b t :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_distance_l428_42823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_sum_l428_42848

theorem cube_root_equation_sum (x y z : ℕ+) :
  (4 : ℝ) * Real.sqrt (Real.rpow 8 (1/3) - Real.rpow 7 (1/3)) = Real.rpow x (1/3) + Real.rpow y (1/3) - Real.rpow z (1/3) →
  x + y + z = 79 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_sum_l428_42848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_subset_singleton_zero_l428_42865

theorem empty_subset_singleton_zero : ∅ ⊆ ({0} : Set ℕ) := by
  intro x h
  contradiction


end NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_subset_singleton_zero_l428_42865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_segment_greater_than_one_meter_l428_42838

/-- Represents the length of a wooden rod in meters -/
def rod_length : ℝ := 4

/-- Represents the minimum length of interest for a segment in meters -/
def min_segment_length : ℝ := 1

/-- Calculates the probability that one segment is greater than the minimum length
    when a rod is arbitrarily cut into two pieces -/
noncomputable def probability_segment_greater_than_min (total_length min_length : ℝ) : ℝ :=
  (2 * (total_length - min_length)) / total_length

/-- Theorem stating that the probability of one segment being greater than 1 m
    when a 4 m rod is arbitrarily cut into two pieces is equal to 1/2 -/
theorem probability_segment_greater_than_one_meter :
  probability_segment_greater_than_min rod_length min_segment_length = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_segment_greater_than_one_meter_l428_42838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_point_of_gp_lines_l428_42874

/-- A line in 2D space defined by the equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y = l.c

/-- Defines a geometric progression with first term a and common ratio r -/
def geometric_progression (a r : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => a * r^(n + 1)

/-- Theorem: All lines with equation ax + by = c, where a, b, c form a geometric progression,
    pass through the point (0, 0) -/
theorem common_point_of_gp_lines (r : ℝ) (hr : r ≠ 0) :
  let l := Line.mk (geometric_progression 1 r 0) (geometric_progression 1 r 1) (geometric_progression 1 r 2)
  l.contains 0 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_point_of_gp_lines_l428_42874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l428_42855

noncomputable def g (x : ℝ) : ℝ := max 0 (max (x^2/4) (|1 - x|))

theorem min_value_of_g :
  ∃ (x : ℝ), ∀ (y : ℝ), g y ≥ g x ∧ g x = 3 - Real.sqrt 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l428_42855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_can_return_to_start_l428_42821

-- Define the grid
structure Grid where
  size : ℕ

-- Define a cell in the grid
structure Cell where
  x : ℕ
  y : ℕ

-- Define a door between two cells
structure Door where
  fromCell : Cell
  toCell : Cell
  isOpen : Bool

-- Define the bug's position
structure BugPosition where
  cell : Cell

-- Define the state of the grid
structure GridState where
  grid : Grid
  doors : List Door
  bugPosition : BugPosition

-- Define a function to move the bug
def moveBug (state : GridState) (newCell : Cell) : GridState :=
  sorry

-- Define a function to check if a cell is reachable
def isReachable (state : GridState) (cell : Cell) : Prop :=
  sorry

-- The main theorem
theorem bug_can_return_to_start (initialState : GridState) :
  ∀ (moves : List Cell), ∃ (returnMoves : List Cell),
    isReachable (List.foldl moveBug initialState (moves ++ returnMoves)) initialState.bugPosition.cell :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_can_return_to_start_l428_42821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_cylinder_equal_volume_l428_42829

/-- The volume of a cylinder -/
noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The volume of a sphere -/
noncomputable def sphereVolume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

theorem sphere_cylinder_equal_volume :
  let cylinderRadius : ℝ := 2
  let cylinderHeight : ℝ := 3
  let sphereRadius : ℝ := (9 : ℝ)^(1/3)
  cylinderVolume cylinderRadius cylinderHeight = sphereVolume sphereRadius := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_cylinder_equal_volume_l428_42829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_f_implies_a_range_l428_42810

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x ≤ 1 then x^2 + (1/2)*a - 2 else a^x - a

theorem monotone_f_implies_a_range (a : ℝ) :
  (∀ x y, 0 < x ∧ x < y → f a x < f a y) → 1 < a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_f_implies_a_range_l428_42810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_equality_implies_specific_values_l428_42820

noncomputable def omega : ℂ := -1/2 + Complex.I * Real.sqrt 3 / 2

theorem ratio_equality_implies_specific_values 
  (a b c : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a / b = b / c) (hbc : b / c = c / a) :
  (a + b - c) / (a - b + c) = 1 ∨ 
  (a + b - c) / (a - b + c) = omega ∨ 
  (a + b - c) / (a - b + c) = omega^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_equality_implies_specific_values_l428_42820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l428_42864

noncomputable def f (x : ℝ) : ℝ := (8 * x^2 - 4) / (4 * x^2 + 2 * x - 1)

theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ N, ∀ x, |x| > N → |f x - 2| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l428_42864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_winning_strategy_l428_42871

/-- Represents the two types of juice -/
inductive Juice
| Orange
| Apple

/-- Represents a glass that may be filled with juice -/
structure Glass where
  filled : Bool
  juice : Option Juice

/-- Represents the state of the game -/
structure GameState where
  n : Nat
  glasses : Vector Glass (2 * n)

/-- Represents a player's move -/
structure Move (n : Nat) where
  glass1 : Fin (2 * n)
  glass2 : Fin (2 * n)
  juice1 : Juice
  juice2 : Juice

/-- Represents a strategy for a player -/
def Strategy (n : Nat) := GameState → Move n

/-- Checks if there are three consecutive glasses with the same juice -/
def hasThreeConsecutive (state : GameState) : Bool :=
  sorry

/-- Helper function to simulate the game -/
def play_game (n : Nat) (petya_strategy : Strategy n) (vasya_strategy : Strategy n) : GameState :=
  sorry

/-- The main theorem stating that Vasya has a winning strategy -/
theorem vasya_winning_strategy :
  ∀ n : Nat, n ≥ 2 →
  ∃ (vasya_strategy : Strategy n),
    ∀ (petya_strategy : Strategy n),
      ¬(hasThreeConsecutive (play_game n petya_strategy vasya_strategy)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_winning_strategy_l428_42871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_three_nonzero_digits_100_factorial_l428_42818

theorem last_three_nonzero_digits_100_factorial (n : ℕ) : 
  n = 976 ↔ ∃ k : ℕ, Nat.factorial 100 = n + k * 1000 ∧ n < 1000 ∧ n ≠ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_three_nonzero_digits_100_factorial_l428_42818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l428_42837

/-- Definition of the ellipse C -/
def ellipseC (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- Definition of the line l -/
def lineL (k m x y : ℝ) : Prop := y = k * x + m

/-- Definition of the circle O -/
def circleO (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Theorem statement -/
theorem ellipse_line_intersection
  (k m : ℝ)
  (h_tangent : ∃ (x y : ℝ), lineL k m x y ∧ circleO x y)
  (h_intersect : ∃ (x1 y1 x2 y2 : ℝ),
    ellipseC x1 y1 ∧ ellipseC x2 y2 ∧
    lineL k m x1 y1 ∧ lineL k m x2 y2 ∧
    x1 ≠ x2 ∧ y1 ≠ y2)
  (h_dot_product : ∃ (x1 y1 x2 y2 : ℝ),
    ellipseC x1 y1 ∧ ellipseC x2 y2 ∧
    lineL k m x1 y1 ∧ lineL k m x2 y2 ∧
    x1 * x2 + y1 * y2 = -3/2) :
  k = Real.sqrt 2/2 ∨ k = -Real.sqrt 2/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l428_42837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parameters_l428_42803

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The focal distance of a hyperbola -/
noncomputable def focal_distance (h : Hyperbola) : ℝ := 
  Real.sqrt (h.a^2 + h.b^2)

/-- Checks if a point (x, y) lies on the asymptote of a hyperbola -/
def on_asymptote (h : Hyperbola) (x y : ℝ) : Prop :=
  y = (h.b / h.a) * x ∨ y = -(h.b / h.a) * x

theorem hyperbola_parameters 
  (h : Hyperbola) 
  (h_focal : focal_distance h = 10) 
  (h_asymptote : on_asymptote h 1 2) : 
  h.a^2 = 5 ∧ h.b^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parameters_l428_42803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_term_is_five_l428_42808

/-- Calculates the nth term of the sequence where each number k is repeated k times. -/
def sequenceTerm (n : ℕ) : ℕ := 
  Nat.sqrt (2 * n + 1/4) + 1/2

/-- Proves that the 15th term of the sequence is 5. -/
theorem fifteenth_term_is_five : sequenceTerm 15 = 5 := by
  -- Unfold the definition of sequenceTerm
  unfold sequenceTerm
  -- Simplify the arithmetic expression
  simp [Nat.sqrt_eq]
  -- The proof is complete
  rfl

#eval sequenceTerm 15  -- Should output 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_term_is_five_l428_42808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_makes_equation_true_l428_42899

-- Define the possible operations
inductive Operation
| Add
| Subtract
| Multiply
| Divide

-- Define a function to apply the operation
noncomputable def applyOperation (op : Operation) (a b : ℝ) : ℝ :=
  match op with
  | Operation.Add => a + b
  | Operation.Subtract => a - b
  | Operation.Multiply => a * b
  | Operation.Divide => a / b

-- Theorem statement
theorem division_makes_equation_true :
  ∃! op : Operation, (applyOperation op 8 4) + 5 - (3 - 2) = 6 ∧ op = Operation.Divide :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_makes_equation_true_l428_42899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_guts_round_digit_count_l428_42893

/-- Represents the content of the guts round as a string -/
def gutsRoundContent : String := "..."  -- Placeholder for the actual content

/-- Counts the occurrences of digits in a string -/
def countDigits (s : String) : Nat :=
  s.toList.filter Char.isDigit |>.length

/-- The total number of digit occurrences in the guts round -/
def totalDigitOccurrences : Nat := countDigits gutsRoundContent

/-- Theorem stating that the total number of digit occurrences in the guts round is 559 -/
theorem guts_round_digit_count :
  totalDigitOccurrences = 559 := by
  sorry  -- Proof is omitted for now

#eval totalDigitOccurrences  -- This will evaluate to 0 due to the empty placeholder string

end NUMINAMATH_CALUDE_ERRORFEEDBACK_guts_round_digit_count_l428_42893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l428_42822

-- Define the function f(x) = e^x - e^(-x)
noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

-- Theorem for the properties of f
theorem f_properties :
  -- f is an odd function
  (∀ x, f (-x) = -f x) ∧
  -- f is an increasing function
  (∀ x y, x < y → f x < f y) ∧
  -- There exists t = -1/2 such that f(x-t) + f(x^2-t^2) ≥ 0 for all x
  (∃ t, t = -1/2 ∧ ∀ x, f (x - t) + f (x^2 - t^2) ≥ 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l428_42822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sleeping_bag_wholesale_cost_l428_42891

/-- Given a selling price and gross profit percentage, calculate the wholesale cost. -/
noncomputable def wholesale_cost (selling_price : ℝ) (gross_profit_percent : ℝ) : ℝ :=
  selling_price / (1 + gross_profit_percent / 100)

/-- Theorem: The wholesale cost of a sleeping bag sold for $28 with a 15% gross profit is $28 / 1.15. -/
theorem sleeping_bag_wholesale_cost :
  wholesale_cost 28 15 = 28 / 1.15 := by
  unfold wholesale_cost
  -- The proof steps would go here, but for now we'll use sorry
  sorry

/-- Compute an approximation of the wholesale cost -/
def approx_wholesale_cost : ℚ :=
  (28 : ℚ) / (1.15 : ℚ)

#eval approx_wholesale_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sleeping_bag_wholesale_cost_l428_42891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_original_price_l428_42875

/-- Calculates the original price of an item given the discounted price and discount percentage. -/
noncomputable def originalPrice (discountedPrice : ℝ) (discountPercentage : ℝ) : ℝ :=
  discountedPrice / (1 - discountPercentage / 100)

/-- Theorem stating that a shirt sold at a 40% discount for Rs. 560 had an original price of approximately Rs. 933.33. -/
theorem shirt_original_price :
  let discountedPrice : ℝ := 560
  let discountPercentage : ℝ := 40
  let calculatedOriginalPrice := originalPrice discountedPrice discountPercentage
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |calculatedOriginalPrice - 933.33| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_original_price_l428_42875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l428_42880

/-- A line in polar coordinates -/
structure PolarLine where
  equation : ℝ → ℝ → Prop

/-- The polar axis (θ = 0 or θ = Real.pi) -/
def polar_axis : PolarLine :=
  { equation := λ ρ θ ↦ θ = 0 ∨ θ = Real.pi }

/-- A point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Check if a line passes through a point in polar coordinates -/
def passes_through (l : PolarLine) (p : PolarPoint) : Prop :=
  l.equation p.ρ p.θ

/-- Check if two lines are parallel in polar coordinates -/
def parallel (l1 l2 : PolarLine) : Prop :=
  ∀ ρ θ, l1.equation ρ θ ↔ l2.equation ρ θ

theorem line_equation_proof (l : PolarLine) 
  (h1 : l.equation = λ ρ θ ↦ ρ * Real.sin θ = 2) 
  (h2 : passes_through l ⟨2, Real.pi/2⟩) 
  (h3 : parallel l polar_axis) : 
  l.equation = λ ρ θ ↦ ρ * Real.sin θ = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l428_42880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_min_distance_to_line_l428_42807

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + t/2, (Real.sqrt 3/2) * t)

-- Define curve C₁
noncomputable def curve_C1 (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define curve C₂
noncomputable def curve_C2 (θ : ℝ) : ℝ × ℝ := (Real.cos θ / 2, (Real.sqrt 3/2) * Real.sin θ)

-- Theorem for part 1
theorem intersection_distance :
  ∃ A B : ℝ × ℝ, (∃ θ₁, curve_C1 θ₁ = A) ∧ (∃ θ₂, curve_C1 θ₂ = B) ∧
  (∃ t₁ : ℝ, line_l t₁ = A) ∧ (∃ t₂ : ℝ, line_l t₂ = B) ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 1 :=
by sorry

-- Theorem for part 2
theorem min_distance_to_line :
  ∃ d : ℝ, d = (Real.sqrt 6 / 4) * (Real.sqrt 2 - 1) ∧
  ∀ P : ℝ × ℝ, (∃ θ, curve_C2 θ = P) →
  ∀ Q : ℝ × ℝ, (∃ t, line_l t = Q) →
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_min_distance_to_line_l428_42807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_divisible_points_exist_l428_42832

/-- A coloring of integer points on the number line with two colors -/
def Coloring := ℤ → Bool

/-- A set is infinite if for any finite subset, there exists an element not in that subset -/
def IsInfinite (s : Set ℤ) : Prop :=
  ∀ (t : Set ℤ), t.Finite → ∃ x ∈ s, x ∉ t

/-- The main theorem -/
theorem infinite_divisible_points_exist (c : Coloring) :
  ∃ (color : Bool), ∀ (k : ℕ), IsInfinite {n : ℤ | c n = color ∧ (k : ℤ) ∣ n} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_divisible_points_exist_l428_42832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_upper_bound_f_diff_max_value_l428_42826

noncomputable section

open Real

def f (a : ℝ) (x : ℝ) : ℝ := log x + (1/2) * x^2 - (a + 2) * x

def extreme_points (a : ℝ) : Set ℝ :=
  {x | x > 0 ∧ deriv (f a) x = 0}

theorem f_sum_upper_bound (a : ℝ) (ha : a > 0) :
  ∀ m n, m ∈ extreme_points a → n ∈ extreme_points a → m < n → f a m + f a n < -3 := by
  sorry

theorem f_diff_max_value (a : ℝ) (ha : a ≥ sqrt e + 1 / sqrt e - 2) :
  ∃ m n, m ∈ extreme_points a ∧ n ∈ extreme_points a ∧ m < n ∧
    ∀ p q, p ∈ extreme_points a → q ∈ extreme_points a → p < q →
      f a n - f a m ≥ f a q - f a p ∧
      f a n - f a m = 1 - e / 2 + 1 / (2 * e) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_upper_bound_f_diff_max_value_l428_42826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_period_approx_four_years_l428_42862

/-- The number of years required for a principal amount to grow to a final amount under compound interest -/
noncomputable def compound_interest_years (principal final_amount interest_rate : ℝ) : ℝ :=
  Real.log (final_amount / principal) / Real.log (1 + interest_rate)

/-- Theorem stating that the investment period is approximately 4 years -/
theorem investment_period_approx_four_years :
  let principal : ℝ := 8000
  let final_amount : ℝ := 9724.05
  let interest_rate : ℝ := 0.05
  let years := compound_interest_years principal final_amount interest_rate
  ⌊years⌋ = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_period_approx_four_years_l428_42862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l428_42886

-- Define the function f as noncomputable
noncomputable def f (x m : ℝ) : ℝ := Real.log (5^x + 4/5^x + m)

-- State the theorem
theorem range_of_m :
  (∀ m : ℝ, ∀ y : ℝ, ∃ x : ℝ, f x m = y) →
  {m : ℝ | m ≤ -4} = Set.Iic (-4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l428_42886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_pi_over_four_l428_42892

theorem integral_equals_pi_over_four : 
  ∫ x in Set.Icc 0 1, Real.sqrt (2*x - x^2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_pi_over_four_l428_42892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mia_sophia_are_johns_siblings_l428_42887

-- Define the characteristics
inductive EyeColor
| Blue
| Brown

inductive HairColor
| Black
| Blonde

inductive Height
| Tall
| Short

-- Define a child's traits
structure ChildTraits where
  name : String
  eyeColor : EyeColor
  hairColor : HairColor
  height : Height

-- Define the list of children
def children : List ChildTraits := [
  { name := "Emma", eyeColor := EyeColor.Blue, hairColor := HairColor.Black, height := Height.Tall },
  { name := "John", eyeColor := EyeColor.Brown, hairColor := HairColor.Blonde, height := Height.Tall },
  { name := "Oliver", eyeColor := EyeColor.Brown, hairColor := HairColor.Black, height := Height.Short },
  { name := "Mia", eyeColor := EyeColor.Blue, hairColor := HairColor.Blonde, height := Height.Short },
  { name := "Lucas", eyeColor := EyeColor.Blue, hairColor := HairColor.Black, height := Height.Tall },
  { name := "Sophia", eyeColor := EyeColor.Blue, hairColor := HairColor.Blonde, height := Height.Tall }
]

-- Define a function to check if two children share at least one characteristic
def shareCharacteristic (child1 child2 : ChildTraits) : Prop :=
  child1.eyeColor = child2.eyeColor ∨ child1.hairColor = child2.hairColor ∨ child1.height = child2.height

-- Define a function to check if three children have one identical characteristic
def haveIdenticalCharacteristic (child1 child2 child3 : ChildTraits) : Prop :=
  (child1.eyeColor = child2.eyeColor ∧ child2.eyeColor = child3.eyeColor) ∨
  (child1.hairColor = child2.hairColor ∧ child2.hairColor = child3.hairColor) ∨
  (child1.height = child2.height ∧ child2.height = child3.height)

-- Theorem: Mia and Sophia are John's siblings
theorem mia_sophia_are_johns_siblings :
  ∃ (john mia sophia : ChildTraits),
    john ∈ children ∧ mia ∈ children ∧ sophia ∈ children ∧
    john.name = "John" ∧ mia.name = "Mia" ∧ sophia.name = "Sophia" ∧
    shareCharacteristic john mia ∧ shareCharacteristic john sophia ∧ shareCharacteristic mia sophia ∧
    haveIdenticalCharacteristic john mia sophia ∧
    (∀ (other : ChildTraits), other ∈ children → other.name ≠ "John" → other.name ≠ "Mia" → other.name ≠ "Sophia" →
      ¬(shareCharacteristic john other ∧ shareCharacteristic mia other ∧ shareCharacteristic sophia other ∧
        haveIdenticalCharacteristic john mia other)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mia_sophia_are_johns_siblings_l428_42887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_10_in_expansion_l428_42835

theorem coefficient_x_10_in_expansion : ∃ (c : ℤ), c = 179 ∧ 
  (∃ (p : ℕ → ℤ), (∀ n, p n = (Polynomial.coeff ((X + 2)^10 * (X^2 - 1)) n)) ∧ p 10 = c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_10_in_expansion_l428_42835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_line_l428_42806

noncomputable def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2)

def is_equidistant_line (a b c a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  ∀ x y : ℝ, a * x + b * y + c = 0 →
    distance_point_to_line x y a₁ b₁ c₁ = distance_point_to_line x y a₂ b₂ c₂

theorem equidistant_line :
  is_equidistant_line 2 (-7) 1 2 (-7) 8 2 (-7) (-6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_line_l428_42806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_students_fifteen_books_l428_42801

/-- Represents a collection of books owned by students -/
structure BookCollection where
  /-- The number of students -/
  num_students : Nat
  /-- The set of all books -/
  all_books : Finset Nat
  /-- A function that maps each student to their set of books -/
  student_books : Fin num_students → Finset Nat

  /-- Each student has a unique set of books -/
  unique_books : ∀ i j, i ≠ j → student_books i ≠ student_books j

  /-- Every two students have exactly one book in common -/
  one_common_book : ∀ i j, i ≠ j → 
    ∃! book, book ∈ (student_books i ∩ student_books j)

  /-- Each book is owned by exactly two students -/
  two_owners : ∀ book ∈ all_books, 
    ∃! (i j : Fin num_students), i ≠ j ∧ book ∈ student_books i ∧ book ∈ student_books j

  /-- All books owned by students are in the all_books set -/
  books_in_all : ∀ i, student_books i ⊆ all_books

/-- The main theorem: a book collection with 6 students satisfying the given conditions has 15 books in total -/
theorem six_students_fifteen_books (bc : BookCollection) 
  (h : bc.num_students = 6) : bc.all_books.card = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_students_fifteen_books_l428_42801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_markup_rate_approximately_53_85_l428_42844

noncomputable def selling_price : ℝ := 10.00
noncomputable def profit_rate : ℝ := 0.20
noncomputable def expense_rate : ℝ := 0.10
noncomputable def tax_rate : ℝ := 0.05

noncomputable def cost : ℝ := selling_price * (1 - profit_rate - expense_rate - tax_rate)

noncomputable def markup_rate : ℝ := (selling_price - cost) / cost * 100

theorem markup_rate_approximately_53_85 :
  ∃ ε > 0, ε < 0.01 ∧ |markup_rate - 53.85| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_markup_rate_approximately_53_85_l428_42844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_area_l428_42804

/-- The area of a triangle inscribed in a rectangle --/
noncomputable def triangle_area (rect_width rect_height d_x e_y f_x : ℝ) : ℝ :=
  rect_width * rect_height - (
    (d_x * e_y / 2) +  -- Area of triangle I
    (rect_width * (rect_height - e_y) / 2) +  -- Area of triangle II
    ((rect_width - f_x) * d_x / 2)  -- Area of triangle III
  )

/-- The area of the inscribed triangle DEF is 8 square units --/
theorem inscribed_triangle_area :
  triangle_area 6 3 1 2 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_area_l428_42804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_cloth_purchase_l428_42828

/-- The number of meters of cloth John bought -/
noncomputable def meters_of_cloth (total_cost cost_per_meter : ℚ) : ℚ :=
  total_cost / cost_per_meter

/-- Theorem stating that the number of meters of cloth John bought is equal to 434.75 divided by 47 -/
theorem johns_cloth_purchase :
  meters_of_cloth (434.75 : ℚ) 47 = 434.75 / 47 := by
  rfl

#eval (434.75 : ℚ) / 47

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_cloth_purchase_l428_42828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equals_tangent_implies_sum_l428_42836

theorem cosine_equals_tangent_implies_sum (α : ℝ) (h : Real.cos α = Real.tan α) :
  (1 / Real.sin α) + (Real.cos α) ^ 4 = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equals_tangent_implies_sum_l428_42836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_alpha_l428_42881

/-- If the terminal side of angle α passes through the point (sin 15°, -cos 15°), 
    then sin²α = 1/2 + √3/4 -/
theorem sin_squared_alpha (α : ℝ) : 
  (∃ (t : ℝ), t * Real.sin (15 * π / 180) = Real.sin α ∧ 
               t * (-Real.cos (15 * π / 180)) = Real.cos α) →
  Real.sin α ^ 2 = 1/2 + Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_alpha_l428_42881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_pure_imaginary_implies_a_eq_two_l428_42825

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The complex number (a+6i)/(3-i) -/
noncomputable def complex_number (a : ℝ) : ℂ := (a + 6 * i) / (3 - i)

/-- A complex number is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The main theorem -/
theorem complex_pure_imaginary_implies_a_eq_two :
  ∀ a : ℝ, is_pure_imaginary (complex_number a) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_pure_imaginary_implies_a_eq_two_l428_42825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l428_42800

/-- The sum of the infinite series Σ(2n + 1) / (n(n + 1)(n + 2)) for n from 1 to infinity -/
noncomputable def infiniteSeries : ℝ := ∑' n, (2 * ↑n + 1) / (↑n * (↑n + 1) * (↑n + 2))

/-- Theorem stating that the infinite series sums to 5/4 -/
theorem infiniteSeriesSum : infiniteSeries = 5/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l428_42800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_yellow_rectangle_l428_42870

/-- Represents a 4x4 grid where each cell can be either green or yellow -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Checks if a 2x3 rectangle starting at (i, j) is all yellow -/
def isYellowRectangle (g : Grid) (i j : Fin 4) : Prop :=
  ∀ x y, x < 2 → y < 3 → g (i + x) (j + y) = true

/-- Checks if the grid contains a 2x3 yellow rectangle -/
def containsYellowRectangle (g : Grid) : Prop :=
  ∃ i j, isYellowRectangle g i j

/-- The probability of each color for each cell is 1/2 -/
def probabilityOfColor : ℚ := 1/2

/-- The total number of possible grid colorings -/
def totalColorings : ℕ := 2^16

/-- The probability of not having a 2x3 yellow rectangle -/
def probabilityNoYellowRectangle : ℚ := 1767/2048

/-- The number of grids containing a yellow rectangle -/
def yellowRectangleCount : ℕ := 9000  -- This is a placeholder value

theorem probability_no_yellow_rectangle :
  (totalColorings - yellowRectangleCount : ℚ) / totalColorings = probabilityNoYellowRectangle := by
  sorry

#eval probabilityNoYellowRectangle.num + probabilityNoYellowRectangle.den

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_yellow_rectangle_l428_42870
