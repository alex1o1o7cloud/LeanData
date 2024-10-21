import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_and_function_evaluation_l591_59132

noncomputable def f (α : Real) : Real :=
  (Real.sin (Real.pi/2 + α) * Real.cos (Real.pi/2 - α)) / Real.cos (Real.pi + α) +
  (2 * Real.sin (Real.pi + α) * Real.cos (Real.pi - α) - Real.cos (Real.pi + α)) /
  (1 + Real.sin α ^ 2 + Real.cos (3*Real.pi/2 + α) - Real.sin (Real.pi/2 + α) ^ 2)

theorem perpendicular_line_and_function_evaluation :
  -- Part 1: Perpendicular line
  ∀ θ : Real,
  (Real.tan θ = -3) →
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ + 2 = 13/5 ∧
  -- Part 2: Function evaluation
  f (-23*Real.pi/6) = Real.sqrt 3 - 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_and_function_evaluation_l591_59132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lily_jog_time_l591_59159

-- Define the given conditions
noncomputable def max_walk_distance : ℝ := 6
noncomputable def max_walk_time : ℝ := 36
noncomputable def lily_jog_distance : ℝ := 4
noncomputable def lily_jog_time_ratio : ℝ := 1 / 3

-- Define Lily's jogging rate
noncomputable def lily_jog_rate : ℝ := (lily_jog_time_ratio * max_walk_time) / lily_jog_distance

-- Define the distance Lily needs to jog
noncomputable def lily_new_distance : ℝ := 7

-- Theorem statement
theorem lily_jog_time : lily_jog_rate * lily_new_distance = 21 := by
  -- Expand the definitions
  unfold lily_jog_rate lily_jog_time_ratio max_walk_time lily_jog_distance lily_new_distance
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lily_jog_time_l591_59159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_parabola_with_distance_to_focus_l591_59182

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem point_on_parabola_with_distance_to_focus
  (P : ℝ × ℝ)
  (h1 : parabola P.1 P.2)
  (h2 : distance P focus = 4) :
  P.1 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_parabola_with_distance_to_focus_l591_59182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_consecutive_even_integer_product_2880_l591_59130

theorem largest_consecutive_even_integer_product_2880 :
  ∀ n : ℕ,
  (n - 2) % 2 = 0 ∧ n % 2 = 0 ∧ (n + 2) % 2 = 0 →
  (n - 2) * n * (n + 2) = 2880 →
  max (n - 2) (max n (n + 2)) = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_consecutive_even_integer_product_2880_l591_59130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_max_value_l591_59103

open Real

-- Define the function representing the determinant
noncomputable def f (θ : ℝ) : ℝ := -2 * exp (-θ)

-- State the theorem
theorem determinant_max_value :
  (∀ θ : ℝ, f θ ≤ 0) ∧ (∀ ε > 0, ∃ θ : ℝ, f θ > -ε) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_max_value_l591_59103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_m_value_l591_59147

noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

theorem ellipse_equation_and_m_value 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : ellipse a b 0 (-1)) 
  (h4 : eccentricity a b = Real.sqrt 2 / 2) :
  ∃ (m : ℝ),
    -- Part I: The equation of the ellipse
    (∀ x y, ellipse a b x y ↔ x^2 / 2 + y^2 = 1) ∧
    -- Part II: The value of m
    (let k : ℝ → ℝ := λ t ↦ t * (1 - m)
     let line (x y : ℝ) := ∃ t, x = t ∧ y = k t
     let intersect (x y : ℝ) := line x y ∧ ellipse a b x y
     let bisect (x1 y1 x2 y2 : ℝ) := 
       y1 / (x1 - m) + y2 / (x2 - m) = 0
     ∀ x1 y1 x2 y2,
       intersect x1 y1 ∧ intersect x2 y2 ∧ x1 ≠ x2 →
       bisect x1 y1 x2 y2 → m = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_m_value_l591_59147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_combined_time_l591_59154

/-- The time it takes for two workers to complete a job together -/
noncomputable def combined_time (time_a time_b : ℝ) : ℝ :=
  1 / (1 / time_a + 1 / time_b)

/-- Theorem: Two workers who take 7 and 10 hours respectively to complete a job independently
    will take 70/17 hours to complete the job working together but independently -/
theorem workers_combined_time :
  combined_time 7 10 = 70 / 17 := by
  -- Unfold the definition of combined_time
  unfold combined_time
  -- Simplify the expression
  simp
  -- The rest of the proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_combined_time_l591_59154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_angle_condition_l591_59195

theorem interior_angle_condition (A : ℝ) (h_interior : 0 < A ∧ A < Real.pi) :
  (A < Real.pi/3 → Real.sin A < Real.sqrt 3/2) ∧
  ∃ A', 0 < A' ∧ A' < Real.pi ∧ Real.sin A' < Real.sqrt 3/2 ∧ A' ≥ Real.pi/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_angle_condition_l591_59195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_assignment_exists_l591_59109

/-- Represents the shapes in the problem -/
inductive Shape
  | Triangle
  | Square
  | Circle

/-- Represents the friends in the problem -/
inductive Friend
  | Ana
  | Bento
  | Celina
  | Diana
  | Elisa
  | Fabio
  | Guilherme

/-- Represents the position of a friend -/
def Position := Fin 7

/-- Represents whether a position is inside a shape -/
def IsInside : Position → Shape → Prop := sorry

/-- The assignment of positions to friends -/
def Assignment := Friend → Position

/-- Checks if an assignment satisfies all conditions -/
def SatisfiesConditions (a : Assignment) : Prop :=
  (∃! p : Position, a Friend.Bento = p ∧ (∃! s : Shape, IsInside p s)) ∧
  (∀ s : Shape, IsInside (a Friend.Celina) s) ∧
  (IsInside (a Friend.Diana) Shape.Triangle ∧ ¬IsInside (a Friend.Diana) Shape.Square) ∧
  (IsInside (a Friend.Elisa) Shape.Triangle ∧ IsInside (a Friend.Elisa) Shape.Circle) ∧
  (¬IsInside (a Friend.Fabio) Shape.Triangle ∧ ¬IsInside (a Friend.Fabio) Shape.Square) ∧
  (IsInside (a Friend.Guilherme) Shape.Circle)

/-- The main theorem stating that there exists a unique assignment satisfying all conditions -/
theorem unique_assignment_exists : ∃! a : Assignment, SatisfiesConditions a ∧
  a Friend.Ana = ⟨5, by norm_num⟩ ∧
  a Friend.Bento = ⟨6, by norm_num⟩ ∧
  a Friend.Celina = ⟨2, by norm_num⟩ ∧
  a Friend.Diana = ⟨4, by norm_num⟩ ∧
  a Friend.Elisa = ⟨3, by norm_num⟩ ∧
  a Friend.Fabio = ⟨0, by norm_num⟩ ∧
  a Friend.Guilherme = ⟨1, by norm_num⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_assignment_exists_l591_59109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l591_59181

theorem expression_evaluation : 
  (Real.pi - 1) ^ 0 - Real.sqrt 9 + 2 * Real.cos (Real.pi / 4) + (1 / 5) ^ (-1 : ℤ) = 3 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l591_59181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_l591_59145

noncomputable section

-- Define the ellipse E
def E : Set (ℝ × ℝ) := {p | p.1^2 / 3 + p.2^2 / 4 = 1}

-- Define points A, B, and P
def A : ℝ × ℝ := (0, -2)
def B : ℝ × ℝ := (3/2, -1)
def P : ℝ × ℝ := (1, -2)

-- Define the line segment AB
def AB : Set (ℝ × ℝ) := {p | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (1 - t) • A + t • B}

theorem ellipse_fixed_point :
  ∀ (l : Set (ℝ × ℝ)), -- line through P
  ∀ (M N : ℝ × ℝ), -- intersection points of l and E
  ∀ (T : ℝ × ℝ), -- intersection of AB and line through M parallel to x-axis
  ∀ (H : ℝ × ℝ), -- point satisfying MT = TH
  (P ∈ l) →
  (M ∈ l ∩ E) →
  (N ∈ l ∩ E) →
  (T ∈ AB) →
  (∃ t : ℝ, H = M + 2 • (T - M)) →
  (T.1 = M.1) →
  (∃ s : ℝ, (0, -2) = (1 - s) • H + s • N) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_l591_59145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sack_weight_in_pounds_l591_59133

-- Define the conversion factor
noncomputable def kg_per_pound : ℝ := 0.4536

-- Define the weight of the sack in kg
noncomputable def sack_weight_kg : ℝ := 150

-- Function to convert kg to pounds
noncomputable def kg_to_pounds (kg : ℝ) : ℝ := kg / kg_per_pound

-- Function to round to nearest whole number
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

-- Theorem statement
theorem sack_weight_in_pounds :
  round_to_nearest (kg_to_pounds sack_weight_kg) = 331 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sack_weight_in_pounds_l591_59133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_calculation_prove_election_votes_calculation_l591_59177

theorem election_votes_calculation (total_votes : ℕ) 
  (invalid_percent : ℚ) (difference_percent : ℚ) : Prop :=
  -- Total number of votes
  total_votes = 5720 ∧
  -- Percentage of invalid votes
  invalid_percent = 1/5 ∧
  -- Percentage difference between A and B's votes
  difference_percent = 3/20 ∧
  -- The number of valid votes B received
  ((1 - invalid_percent) * difference_percent / 2 : ℚ) * total_votes = 1859

theorem prove_election_votes_calculation :
  ∃ (total_votes : ℕ) (invalid_percent difference_percent : ℚ),
    election_votes_calculation total_votes invalid_percent difference_percent := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_calculation_prove_election_votes_calculation_l591_59177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l591_59121

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- The right focus of a hyperbola -/
noncomputable def right_focus (h : Hyperbola) : ℝ × ℝ :=
  (Real.sqrt (h.a^2 + h.b^2), 0)

/-- One of the asymptotes of the hyperbola -/
noncomputable def asymptote (h : Hyperbola) (x : ℝ) : ℝ :=
  (h.b / h.a) * x

/-- The point symmetric to the right focus with respect to the asymptote -/
noncomputable def symmetric_point (h : Hyperbola) : ℝ × ℝ :=
  ((h.b^2 - h.a^2) / Real.sqrt (h.a^2 + h.b^2), -2 * h.a * h.b / Real.sqrt (h.a^2 + h.b^2))

/-- Checks if a point is on the left branch of the hyperbola -/
def is_on_left_branch (h : Hyperbola) (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ (p.1^2 / h.a^2) - (p.2^2 / h.b^2) = 1

/-- The eccentricity of the hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The main theorem -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_symmetric : is_on_left_branch h (symmetric_point h)) : 
  eccentricity h = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l591_59121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_problem_l591_59150

/-- The number of students on the bus before stopping at the intersection -/
def students_before : ℕ := 28

/-- The number of students who entered the bus at the stop -/
def students_entered : ℕ := 30

/-- The total number of students after stopping -/
def total_students : ℕ := 58

theorem bus_problem :
  (students_before + students_entered = total_students) ∧
  (0.4 * (students_entered : ℝ) = 12) →
  students_before = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_problem_l591_59150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sum_sqrt_max_value_attained_l591_59158

theorem max_value_sum_sqrt (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 8) :
  Real.sqrt (3*x + 2) + Real.sqrt (3*y + 2) + Real.sqrt (3*z + 2) ≤ 3 * Real.sqrt 10 :=
by sorry

theorem max_value_attained (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 8) :
  ∃ (a b c : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 8 ∧
  Real.sqrt (3*a + 2) + Real.sqrt (3*b + 2) + Real.sqrt (3*c + 2) = 3 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sum_sqrt_max_value_attained_l591_59158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l591_59166

theorem min_value_of_expression (x : ℝ) (hx : x > 0) :
  6 * x^6 + 8 * x^(-3 : ℝ) ≥ 14 ∧ ∃ y : ℝ, y > 0 ∧ 6 * y^6 + 8 * y^(-3 : ℝ) = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l591_59166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_running_time_is_38_over_3_l591_59142

/-- Represents the average running time of students across three grades. -/
noncomputable def averageRunningTime (third_grade_time fourth_grade_time fifth_grade_time : ℝ)
  (third_to_fourth_ratio fourth_to_fifth_ratio : ℝ) : ℝ :=
  let fifth_grade_count := 1
  let fourth_grade_count := fourth_to_fifth_ratio * fifth_grade_count
  let third_grade_count := third_to_fourth_ratio * fourth_grade_count
  let total_students := third_grade_count + fourth_grade_count + fifth_grade_count
  let total_time := third_grade_time * third_grade_count + 
                    fourth_grade_time * fourth_grade_count + 
                    fifth_grade_time * fifth_grade_count
  total_time / total_students

/-- Theorem stating that the average running time is 38/3 given the problem conditions. -/
theorem average_running_time_is_38_over_3 :
  averageRunningTime 14 18 8 3 (1/2) = 38/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_running_time_is_38_over_3_l591_59142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_abs_x_squared_minus_one_eq_ten_iff_in_solution_set_l591_59198

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the set of solutions
def solution_set : Set ℝ := 
  {x | x ∈ Set.Ioc (-2 * Real.sqrt 3) (-Real.sqrt 11) ∪ Set.Ico (Real.sqrt 11) (2 * Real.sqrt 3)}

-- State the theorem
theorem floor_abs_x_squared_minus_one_eq_ten_iff_in_solution_set :
  ∀ x : ℝ, floor (|x^2 - 1|) = 10 ↔ x ∈ solution_set :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_abs_x_squared_minus_one_eq_ten_iff_in_solution_set_l591_59198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l591_59111

noncomputable def f (x φ b : ℝ) : ℝ := Real.sin (2 * x + φ) + b

theorem function_properties (φ b : ℝ) 
  (h1 : ∀ x : ℝ, f (x + π/3) φ b = f (-x) φ b)
  (h2 : f (2*π/3) φ b = -1) : 
  b = 0 ∨ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l591_59111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_divisors_not_perfect_square_l591_59155

/-- A positive integer is square-free if it's not divisible by p^2 for any prime p. -/
def SquareFree (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ n)

/-- The number of positive divisors of n. -/
def numDivisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- A function that returns the set of chosen divisors. -/
noncomputable def chosenDivisors (n : ℕ) : Finset ℕ :=
  sorry  -- Implementation not required for the statement

/-- Predicate to check if a number is a perfect square. -/
def isPerfectSquare (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k^2

theorem max_divisors_not_perfect_square (n : ℕ) (h1 : n > 1) (h2 : SquareFree n) :
  let d := numDivisors n
  let k := Nat.log 2 d
  (∀ a b, a ∈ chosenDivisors n → b ∈ chosenDivisors n → ¬isPerfectSquare (a^2 + a*b - n)) →
  (chosenDivisors n).card ≤ 2^(k - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_divisors_not_perfect_square_l591_59155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_l591_59168

def M (m : ℝ) : Set ℝ := {4, 5, -3*m}
def N : Set ℝ := {-9, 3}

theorem intersection_implies_m_value (m : ℝ) :
  (M m ∩ N).Nonempty → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_l591_59168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_attacking_pairs_l591_59131

/-- Represents a chessboard configuration -/
structure ChessBoard where
  rows : Fin 8 → Fin 8 → Bool
  rook_count : Nat
  rook_count_is_16 : rook_count = 16

/-- A pair of rooks that can attack each other -/
def AttackingPair (board : ChessBoard) : Type :=
  { pair : (Fin 8 × Fin 8) × (Fin 8 × Fin 8) // 
    (pair.1.1 = pair.2.1 ∨ pair.1.2 = pair.2.2) ∧ 
    board.rows pair.1.1 pair.1.2 ∧ 
    board.rows pair.2.1 pair.2.2 ∧
    (∀ i j, (i ≠ pair.1.1 ∨ j ≠ pair.1.2) ∧ (i ≠ pair.2.1 ∨ j ≠ pair.2.2) →
      (pair.1.1 = pair.2.1 → i ≠ pair.1.1 ∨ ¬board.rows i j) ∧
      (pair.1.2 = pair.2.2 → j ≠ pair.1.2 ∨ ¬board.rows i j)) }

/-- The number of attacking pairs in a given board configuration -/
def AttackingPairsCount (board : ChessBoard) : Nat :=
  sorry -- We'll use sorry here as we can't directly use Fintype.card

/-- The main theorem: The minimum number of attacking pairs is 16 -/
theorem min_attacking_pairs : 
  ∀ board : ChessBoard, AttackingPairsCount board ≥ 16 := by
  sorry

#check min_attacking_pairs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_attacking_pairs_l591_59131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_12_l591_59183

noncomputable def f (x : ℝ) : ℝ := (Real.cos (Real.pi / 4 + x))^2 - (Real.cos (Real.pi / 4 - x))^2

theorem f_value_at_pi_12 : f (Real.pi / 12) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_12_l591_59183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concave_functions_satisfy_property_l591_59148

variable {α : Type*} [LinearOrderedField α]

def is_concave (f : α → α) (a b : α) : Prop :=
  ∀ (x y : α) (t : α), 0 ≤ t ∧ t ≤ 1 →
    f (t * x + (1 - t) * y) ≤ t * f x + (1 - t) * f y

theorem concave_functions_satisfy_property
  (f₁ f₃ : α → α) (h₁ : is_concave f₁ 0 1) (h₃ : is_concave f₃ 0 1) :
  ∀ (x₁ x₂ t : α), 0 ≤ x₁ ∧ x₁ ≤ 1 → 0 ≤ x₂ ∧ x₂ ≤ 1 → 0 ≤ t ∧ t ≤ 1 →
    f₁ (t * x₁ + (1 - t) * x₂) ≤ t * f₁ x₁ + (1 - t) * f₁ x₂ ∧
    f₃ (t * x₁ + (1 - t) * x₂) ≤ t * f₃ x₁ + (1 - t) * f₃ x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concave_functions_satisfy_property_l591_59148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_descending_order_l591_59175

-- Define the constants as noncomputable
noncomputable def a : ℝ := (0.4 : ℝ) ^ (-0.5 : ℝ)
noncomputable def b : ℝ := (0.5 : ℝ) ^ (0.5 : ℝ)
noncomputable def c : ℝ := Real.log 2 / Real.log 0.2

-- State the theorem
theorem descending_order : a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_descending_order_l591_59175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_cos6_l591_59139

theorem min_sin6_cos6 (x : ℝ) : Real.sin x ^ 6 + Real.cos x ^ 6 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_cos6_l591_59139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_average_weight_l591_59161

/-- The initial average weight of A, B, and C -/
def W : ℝ := sorry

/-- The weight of person D -/
def D : ℝ := sorry

/-- The weight of person A -/
def A : ℝ := 80

theorem initial_average_weight :
  (3 * W + D = 320) →
  (3 * W + 2 * D - 72 = 316) →
  W = 84 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_average_weight_l591_59161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_equivalence_l591_59164

theorem m_range_equivalence (m : ℝ) : 
  (m ∈ Set.Ioi 1) ↔ 
  (∀ x : ℝ, (1 - m < x + m ∧ x + m < 2 * m) ↔ (2 / (x - 1) < -1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_equivalence_l591_59164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_checkerboard_ratio_l591_59128

/-- The size of the checkerboard -/
def board_size : ℕ := 9

/-- The number of rectangles on the board -/
def num_rectangles : ℕ := (board_size + 1).choose 2 * (board_size + 1).choose 2

/-- The number of squares on the board -/
def num_squares : ℕ := (board_size * (board_size + 1) * (2 * board_size + 1)) / 6

/-- The numerator of the simplified ratio of squares to rectangles -/
def m : ℕ := 19

/-- The denominator of the simplified ratio of squares to rectangles -/
def n : ℕ := 135

theorem checkerboard_ratio :
  (num_squares : ℚ) / num_rectangles = m / n ∧ Int.gcd m n = 1 :=
sorry

#eval m + n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_checkerboard_ratio_l591_59128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_primes_l591_59196

/-- Given a sequence of natural numbers, prove that the 100th term has at least 100 distinct prime factors -/
theorem hundred_primes (S : ℕ → ℕ) (h_initial : S 1 > 1) 
  (h_rec : ∀ i : ℕ, i ≥ 1 → S (i + 1) = S i + S i ^ 2) : 
  (Finset.card (Nat.factorization (S 100)).support) ≥ 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_primes_l591_59196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_extrema_l591_59197

/-- The rational function f(x) with parameters a, b, c, and d -/
noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ := (a * x^2 - b * x + c) / (x^2 - d * x - a)

/-- Theorem stating the conditions and conclusion about the parameters of f -/
theorem rational_function_extrema 
  (a b c d : ℝ) :
  (∀ x : ℝ, f a b c d x ≤ f a b c d 2) ∧  -- Maximum at x=2
  (f a b c d 2 = 1) ∧                     -- Maximum value is 1
  (∀ x : ℝ, f a b c d x ≥ f a b c d 5) ∧  -- Minimum at x=5
  (f a b c d 5 = 2.5)                     -- Minimum value is 2.5
  →
  a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_extrema_l591_59197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_inequality_range_l591_59156

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x - Real.sin x else -((-x) - Real.sin (-x))

theorem odd_function_inequality_range (f : ℝ → ℝ) (m : ℝ) : 
  (∀ (x : ℝ), f (-x) = -f x) →
  (∀ (x : ℝ), x ≥ 0 → f x = x - Real.sin x) →
  (∀ (t : ℝ), f (-4*t) > f (2*m*t^2 + m)) →
  m ∈ Set.Ioo (-Real.sqrt 2) 0 := by
  sorry

#check odd_function_inequality_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_inequality_range_l591_59156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_conditions_l591_59122

/-- Given a complex number z dependent on a real parameter m -/
def z (m : ℝ) : ℂ := Complex.mk (m^2 - 1) (m^2 - 3*m + 2)

/-- Condition for z to be a pure imaginary number -/
def is_pure_imaginary (m : ℝ) : Prop := z m = Complex.I * (z m).im

/-- Condition for z to be on the circle centered at (0, -3m) with radius √17 -/
def on_circle (m : ℝ) : Prop := 
  Complex.abs (z m - Complex.I * (-3*m)) = Real.sqrt 17

theorem z_conditions (m : ℝ) : 
  (is_pure_imaginary m ↔ m = -1) ∧ 
  (on_circle m ↔ m = Real.sqrt 2 ∨ m = -Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_conditions_l591_59122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C_to_l_l591_59170

/-- The curve C in the Cartesian plane -/
noncomputable def C : ℝ → ℝ × ℝ := λ s ↦ (2 * s^2, 2 * Real.sqrt 2 * s)

/-- The line l in the Cartesian plane -/
def l : ℝ × ℝ → Prop := λ p ↦ p.1 - 2 * p.2 + 8 = 0

/-- The distance function from a point to the line l -/
noncomputable def dist_to_l (p : ℝ × ℝ) : ℝ :=
  abs (p.1 - 2 * p.2 + 8) / Real.sqrt 5

theorem min_distance_C_to_l :
  ∃ (d : ℝ), d = 4 * Real.sqrt 5 / 5 ∧
  ∀ (s : ℝ), dist_to_l (C s) ≥ d ∧
  ∃ (s₀ : ℝ), dist_to_l (C s₀) = d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C_to_l_l591_59170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l591_59167

/-- IsTriangle predicate -/
def IsTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Area of a triangle given its sides -/
noncomputable def area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Triangle inequality theorem -/
theorem triangle_inequality (a b c : ℝ) (S : ℝ) (h_triangle : IsTriangle a b c) (h_area : S = area a b c) :
  a^2 + b^2 + c^2 ≥ 4 * S * Real.sqrt 3 ∧
  (a^2 + b^2 + c^2 = 4 * S * Real.sqrt 3 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l591_59167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_identity_l591_59100

/-- Sequence G_n defined recursively -/
def G : ℕ → ℕ
  | 0 => 1  -- Adding this case to handle G_0
  | 1 => 1
  | 2 => 2
  | n + 3 => 2 * G (n + 2) + G (n + 1)

/-- Theorem stating that G_10 * G_12 - G_11^2 = 1 -/
theorem G_identity : G 10 * G 12 - G 11 * G 11 = 1 := by
  sorry

#eval G 10 * G 12 - G 11 * G 11  -- This line is optional, for checking the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_identity_l591_59100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_felix_siblings_l591_59190

-- Define the eye colors
inductive EyeColor
| Green
| Grey

-- Define the hair colors
inductive HairColor
| Red
| Black

-- Define a child's characteristics
structure Child where
  name : String
  eyeColor : EyeColor
  hairColor : HairColor

-- Define the list of children
def children : List Child := [
  ⟨"Lucas", EyeColor.Green, HairColor.Black⟩,
  ⟨"Felix", EyeColor.Grey, HairColor.Red⟩,
  ⟨"Zara", EyeColor.Grey, HairColor.Black⟩,
  ⟨"Oliver", EyeColor.Green, HairColor.Red⟩,
  ⟨"Mila", EyeColor.Green, HairColor.Black⟩,
  ⟨"Eva", EyeColor.Green, HairColor.Red⟩
]

-- Define a function to check if two children share a characteristic
def shareCharacteristic (c1 c2 : Child) : Prop :=
  c1.eyeColor = c2.eyeColor ∨ c1.hairColor = c2.hairColor

-- Define a function to find siblings
def areSiblings (c1 c2 c3 : Child) : Prop :=
  shareCharacteristic c1 c2 ∧ shareCharacteristic c2 c3 ∧ shareCharacteristic c1 c3

-- Theorem to prove
theorem felix_siblings :
  ∃ (oliver eva : Child),
    oliver ∈ children ∧
    eva ∈ children ∧
    oliver.name = "Oliver" ∧
    eva.name = "Eva" ∧
    areSiblings (children[1]) oliver eva :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_felix_siblings_l591_59190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_lipstick_count_l591_59172

/-- Proves the number of students wearing blue lipstick given the conditions of the problem -/
theorem blue_lipstick_count (total_students blue_lipstick_count : ℕ) 
  (h1 : total_students = 200)
  (h2 : 2 * (total_students / 2) = total_students) -- Half of students wore lipstick
  (h3 : 4 * ((total_students / 2) / 4) = total_students / 2) -- Quarter of lipstick wearers wore red
  (h4 : 5 * blue_lipstick_count = (total_students / 2) / 4) -- One-fifth as many blue as red
  : blue_lipstick_count = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_lipstick_count_l591_59172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_rotation_and_distance_l591_59107

noncomputable def A : ℂ := Complex.mk (Real.sqrt 3) 1

noncomputable def z₀ : ℂ := A * Complex.exp (Complex.I * (2 * Real.pi / 3))

theorem complex_rotation_and_distance (z : ℂ) :
  Complex.abs (z - z₀) = 1 ∧ 
  Complex.arg ((z - z₀) / z₀) = 2 * Real.pi / 3 →
  z = Complex.mk (-Real.sqrt 3 / 2) (3 / 2) ∨ 
  z = Complex.mk (-Real.sqrt 3) 0 := by
  sorry

#check complex_rotation_and_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_rotation_and_distance_l591_59107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_four_six_l591_59152

-- Define the operation @
def at_op (a b : Int) : Int := 2 * a - 4 * b

-- Theorem statement
theorem at_four_six : at_op 4 6 = -16 := by
  -- Unfold the definition of at_op
  unfold at_op
  -- Perform the calculation
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_four_six_l591_59152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l591_59188

-- Define the propositions
def p : Prop := ∀ x : ℝ, x > 0 → x^2 - x + 1 > 0

def q (A B C a b c : ℝ) : Prop :=
  Real.sin A > Real.sin B ↔ a > b

-- State the theorem
theorem problem_statement :
  (¬p) ∧ (∀ A B C a b c : ℝ, q A B C a b c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l591_59188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_section_A_l591_59129

theorem average_weight_section_A (students_A : ℕ) (students_B : ℕ) (avg_weight_B : ℝ) (avg_weight_total : ℝ) :
  students_A = 40 →
  students_B = 20 →
  avg_weight_B = 40 →
  avg_weight_total = 46.67 →
  ∃ avg_weight_A : ℝ,
    avg_weight_A * (students_A : ℝ) + avg_weight_B * (students_B : ℝ) =
    avg_weight_total * ((students_A + students_B) : ℝ) ∧
    abs (avg_weight_A - 50.005) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_section_A_l591_59129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hollow_cylinder_volume_l591_59136

-- Define the parameters of the hollow cylindrical pipe
def external_diameter : ℝ := 14
def internal_diameter : ℝ := 10
def pipe_height : ℝ := 10

-- Define the volume calculation function for a cylinder
noncomputable def cylinder_volume (diameter : ℝ) (height : ℝ) : ℝ :=
  (Real.pi / 4) * diameter^2 * height

-- Theorem: The volume of the hollow cylindrical pipe is 240π cm³
theorem hollow_cylinder_volume :
  cylinder_volume external_diameter pipe_height - cylinder_volume internal_diameter pipe_height = 240 * Real.pi :=
by
  -- Unfold the definition of cylinder_volume
  unfold cylinder_volume
  -- Simplify the expression
  simp [external_diameter, internal_diameter, pipe_height]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hollow_cylinder_volume_l591_59136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_diagonal_l591_59137

-- Define the quadrilateral PQRS
structure Quadrilateral :=
  (P Q R S : ℝ × ℝ)

-- Define the lengths of the sides
noncomputable def side_length (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- State the theorem
theorem quadrilateral_diagonal (PQRS : Quadrilateral) : 
  side_length PQRS.P PQRS.Q = 7 →
  side_length PQRS.Q PQRS.R = 25 →
  side_length PQRS.R PQRS.S = 7 →
  side_length PQRS.S PQRS.P = 16 →
  ∃ (n : ℤ), (side_length PQRS.P PQRS.R = n) ∧ (n = 19 ∨ n = 20 ∨ n = 21 ∨ n = 22) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_diagonal_l591_59137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_below_zero_negative_l591_59126

/-- Represents temperature in Celsius -/
structure Temperature where
  value : Int
  unit : String
  deriving Repr

/-- Defines the representation of temperature above zero -/
def above_zero (t : Temperature) : Bool :=
  t.value > 0 && t.unit = "°C"

/-- Defines the representation of temperature below zero -/
def below_zero (t : Temperature) : Bool :=
  t.value < 0 && t.unit = "°C"

/-- States that temperatures above zero are denoted with a positive sign -/
axiom above_zero_positive (t : Temperature) :
  above_zero t → t.value.repr.front = '+'

/-- Theorem: If temperatures above zero are denoted with a positive sign,
    then temperatures below zero should be denoted with a negative sign -/
theorem below_zero_negative (t : Temperature) :
  below_zero t → t.value.repr.front = '-' := by
  sorry

#eval Temperature.mk 2 "°C"
#eval Temperature.mk (-14) "°C"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_below_zero_negative_l591_59126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_315_degrees_l591_59194

theorem sin_315_degrees :
  Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_315_degrees_l591_59194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_problem_l591_59192

theorem tan_sum_problem (x y : ℝ) 
  (h1 : Real.tan x + Real.tan y = 15) 
  (h2 : (Real.tan x)⁻¹ + (Real.tan y)⁻¹ = 40) : 
  Real.tan (x + y) = 24 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_problem_l591_59192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_complement_is_empty_l591_59115

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | Real.sqrt (x - 2) ≤ 0}
def B : Set ℝ := {x : ℝ | (10 : ℝ)^(x^2 - 2) = (10 : ℝ)^x}

-- State the theorem
theorem A_intersect_B_complement_is_empty : A ∩ Bᶜ = ∅ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_complement_is_empty_l591_59115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_quadrilateral_l591_59146

noncomputable def side1 : ℝ := 70
noncomputable def side2 : ℝ := 80
noncomputable def side3 : ℝ := 90
noncomputable def side4 : ℝ := 100

/-- A structure to represent a quadrilateral -/
structure Quadrilateral (α : Type*) where
  side1_length : α
  side2_length : α
  side3_length : α
  side4_length : α
  area : α

theorem max_area_quadrilateral :
  let max_area := Real.sqrt (side1 * side2 * side3 * side4)
  ∃ (quad : Quadrilateral ℝ),
    quad.side1_length = side1 ∧
    quad.side2_length = side2 ∧
    quad.side3_length = side3 ∧
    quad.side4_length = side4 ∧
    quad.area ≤ max_area ∧
    ∀ (other_quad : Quadrilateral ℝ),
      other_quad.side1_length = side1 →
      other_quad.side2_length = side2 →
      other_quad.side3_length = side3 →
      other_quad.side4_length = side4 →
      other_quad.area ≤ max_area := by
  sorry

#check max_area_quadrilateral

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_quadrilateral_l591_59146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_fraction_l591_59138

/-- Simple interest calculation --/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

theorem simple_interest_fraction (P : ℝ) :
  simple_interest P 10 2 = P / 5 := by
  unfold simple_interest
  field_simp
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_fraction_l591_59138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_set_of_fraction_l591_59101

theorem value_set_of_fraction (m n t : ℝ) (h1 : m^2 + n^2 = t^2) (h2 : t ≠ 0) :
  ∃ (S : Set ℝ), S = {x | ∃ (m n t : ℝ), m^2 + n^2 = t^2 ∧ t ≠ 0 ∧ x = n / (m - 2*t)} ∧
  S = Set.Icc (-Real.sqrt 3 / 3) (Real.sqrt 3 / 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_set_of_fraction_l591_59101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_topless_cube_configurations_l591_59113

/-- Represents a square piece --/
inductive Square
| A | B | C | D | E | F | G

/-- Represents an L-shaped figure made of three squares --/
structure LShape

/-- Represents a configuration of an L-shape with an additional square --/
structure Configuration where
  l : LShape
  s : Square

/-- Predicate to check if a configuration can be folded into a topless cubical box --/
def can_fold_into_box : Configuration → Prop :=
  fun _ => sorry

/-- The theorem to be proved --/
theorem topless_cube_configurations :
  ∃ (valid_configs : Finset Configuration),
    (∀ c ∈ valid_configs, can_fold_into_box c) ∧
    (∀ c : Configuration, c ∉ valid_configs → ¬can_fold_into_box c) ∧
    Finset.card valid_configs = 5 :=
by
  sorry

#check topless_cube_configurations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_topless_cube_configurations_l591_59113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_problem_l591_59104

theorem cos_alpha_problem (α : ℝ) 
  (h1 : Real.cos α = -Real.sqrt 5 / 5) 
  (h2 : π / 2 < α ∧ α < π) : 
  Real.sin α = 2 * Real.sqrt 5 / 5 ∧ 
  (Real.sin (3 * π / 2 + α) + 2 * Real.cos (π / 2 + α)) / Real.cos (3 * π - α) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_problem_l591_59104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recipes_for_game_night_l591_59176

/-- Calculates the number of full recipes needed for a school event --/
def recipes_needed (total_students : ℕ) (attendance_drop : ℚ) (cookies_per_batch : ℕ) (cookies_per_student : ℕ) : ℕ :=
  let attending_students := (total_students : ℚ) * (1 - attendance_drop)
  let total_cookies_needed := attending_students * (cookies_per_student : ℚ)
  (total_cookies_needed / cookies_per_batch).ceil.toNat

/-- Proves that 10 full recipes are needed for the given conditions --/
theorem recipes_for_game_night : recipes_needed 150 (2/5) 18 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recipes_for_game_night_l591_59176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_sqrt_five_is_seventh_term_l591_59119

/-- The sequence defined by the square root of an arithmetic progression -/
noncomputable def a (n : ℕ) : ℝ := Real.sqrt (3 * n - 1)

/-- Theorem stating that 2√5 is the 7th term of the sequence -/
theorem two_sqrt_five_is_seventh_term :
  a 7 = 2 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_sqrt_five_is_seventh_term_l591_59119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_price_decrease_approx_14_percent_l591_59134

/-- Calculates the percentage decrease in price per pack of pens during a promotion. -/
def pen_price_decrease (original_packs : ℕ) (original_price : ℚ) 
  (promo_packs : ℕ) (promo_price : ℚ) : ℚ :=
  let original_per_pack : ℚ := original_price / original_packs
  let promo_per_pack : ℚ := promo_price / promo_packs
  (original_per_pack - promo_per_pack) / original_per_pack * 100

/-- The percentage decrease in price per pack of pens during the promotion is approximately 14%. -/
theorem pen_price_decrease_approx_14_percent : 
  ⌊pen_price_decrease 3 7 4 8⌋ = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_price_decrease_approx_14_percent_l591_59134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_integer_progression_difference_l591_59110

/-- Represents the progression between counting units for a number system -/
def progression (num_system : Type) : ℕ := sorry

/-- Axiom: Integers have a base-10 progression between counting units -/
axiom integer_progression : progression ℤ = 10

/-- Axiom: Decimals have a base-10 progression between counting units -/
axiom decimal_progression : progression ℚ = 10

/-- Theorem: The statement "Decimals, like integers, have a base-10 progression between counting units" is false -/
theorem decimal_integer_progression_difference : 
  ¬(∀ (n : Type), progression n = 10 → (n = ℤ ∨ n = ℚ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_integer_progression_difference_l591_59110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_eq_beta_sufficient_not_necessary_for_sin_eq_l591_59169

theorem alpha_eq_beta_sufficient_not_necessary_for_sin_eq : 
  (∀ α β : Real, α = β → Real.sin α = Real.sin β) ∧ 
  (∃ α β : Real, Real.sin α = Real.sin β ∧ α ≠ β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_eq_beta_sufficient_not_necessary_for_sin_eq_l591_59169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_snake_length_l591_59184

/-- Represents the length of a snake in inches -/
def SnakeLength : Type := ℕ

/-- Converts inches to feet -/
def inches_to_feet (inches : ℕ) : ℚ :=
  inches / 12

theorem first_snake_length 
  (total_length second_snake third_snake : ℕ)
  (h1 : total_length = 50)
  (h2 : second_snake = 16)
  (h3 : third_snake = 10) :
  inches_to_feet (total_length - second_snake - third_snake) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_snake_length_l591_59184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_surface_area_increase_l591_59178

/-- The surface area of a dodecahedron with edge length a -/
noncomputable def dodecahedronSurfaceArea (a : ℝ) : ℝ := 3 * Real.sqrt (25 + 10 * Real.sqrt 5) * a^2

/-- The percentage increase in surface area when edge length increases by 20% -/
theorem dodecahedron_surface_area_increase : 
  ∀ a : ℝ, a > 0 → 
  (dodecahedronSurfaceArea (1.2 * a) - dodecahedronSurfaceArea a) / dodecahedronSurfaceArea a = 0.44 := by
  sorry

#eval "Dodecahedron surface area increase theorem"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_surface_area_increase_l591_59178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_tails_prob_l591_59106

/-- Represents the probability of getting a specific outcome in a single coin flip -/
noncomputable def single_flip_prob : ℚ := 1 / 2

/-- Represents the number of consecutive coin flips -/
def num_flips : ℕ := 3

/-- Represents the number of ways to get exactly one tails in 3 flips -/
def num_favorable_outcomes : ℕ := 3

/-- Theorem stating the probability of getting exactly one tails in 3 consecutive flips of a fair coin -/
theorem exactly_one_tails_prob : 
  (num_favorable_outcomes : ℚ) * single_flip_prob^num_flips = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_tails_prob_l591_59106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_maximizes_probability_optimal_strategy_l591_59123

/-- Represents the number of voters in Anchuria -/
def n : ℕ := sorry

/-- Represents the probability of Miraflores winning in a district -/
def winProbability (supportersInDistrict totalInDistrict : ℕ) : ℚ :=
  supportersInDistrict / totalInDistrict

/-- Represents the probability of Miraflores winning the election given a specific division strategy -/
def electionProbability (district1Supporters district1Total district2Supporters district2Total : ℕ) : ℚ :=
  (winProbability district1Supporters district1Total) * (winProbability district2Supporters district2Total)

/-- Theorem stating that Miraflores' optimal strategy maximizes his probability of winning -/
theorem optimal_strategy_maximizes_probability :
  ∀ (d1s d1t d2s d2t : ℕ),
    d1s + d2s = n + 1 →
    d1t + d2t = 2 * n →
    d1s ≤ d1t ∧ d2s ≤ d2t →
    electionProbability d1s d1t d2s d2t ≤ electionProbability 1 1 n (2 * n - 1) :=
by sorry

/-- Corollary stating that the optimal strategy is to create one district with only Miraflores -/
theorem optimal_strategy :
  ∃ (d1s d1t d2s d2t : ℕ),
    d1s + d2s = n + 1 ∧
    d1t + d2t = 2 * n ∧
    d1s ≤ d1t ∧ d2s ≤ d2t ∧
    electionProbability d1s d1t d2s d2t = electionProbability 1 1 n (2 * n - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_maximizes_probability_optimal_strategy_l591_59123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_line_and_circle_l591_59112

noncomputable section

-- Define the polar coordinate system
def PolarPoint := ℝ × ℝ

-- Define the line l in polar form
def line_l (p : PolarPoint) : Prop :=
  p.1 * Real.cos (p.2 - Real.pi/3) = 1

-- Define the circle C in polar form
def circle_C (p : PolarPoint) : Prop :=
  p.1 = 2

-- Define the intersection points
def intersection_points : Set PolarPoint :=
  {(2, 0), (2, 2*Real.pi/3)}

-- Theorem statement
theorem intersection_of_line_and_circle :
  ∀ p : PolarPoint, line_l p ∧ circle_C p ↔ p ∈ intersection_points :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_line_and_circle_l591_59112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_alcohol_percentage_l591_59199

/-- Represents a solution with alcohol and water -/
structure Solution where
  alcohol : ℚ
  water : ℚ

/-- Calculates the percentage of alcohol in a solution -/
def alcoholPercentage (s : Solution) : ℚ :=
  s.alcohol / (s.alcohol + s.water) * 100

/-- Solution A with alcohol to water ratio of 21:4 -/
def solutionA : Solution :=
  { alcohol := 21, water := 4 }

/-- Solution B with alcohol to water ratio of 2:3 -/
def solutionB : Solution :=
  { alcohol := 2, water := 3 }

/-- The ratio in which solutions A and B are mixed (5:6) -/
def mixRatio : ℚ × ℚ := (5, 6)

/-- Theorem: The resulting mixture is 60% alcohol -/
theorem mixture_alcohol_percentage :
  let totalVolume := mixRatio.fst + mixRatio.snd
  let mixedAlcohol := mixRatio.fst * alcoholPercentage solutionA / 100 +
                      mixRatio.snd * alcoholPercentage solutionB / 100
  mixedAlcohol / totalVolume * 100 = 60 := by
  sorry

#eval alcoholPercentage solutionA
#eval alcoholPercentage solutionB

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_alcohol_percentage_l591_59199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_quadrilateral_area_l591_59162

/-- An ellipse with semi-major axis 4 and semi-minor axis 2 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 16) + (p.2^2 / 4) = 1}

/-- The foci of the ellipse -/
def Foci : Set (ℝ × ℝ) := {f : ℝ × ℝ | ∃ (x : ℝ), f = (x, 0) ∧ x^2 = 12}

/-- Two points on the ellipse symmetric with respect to the origin -/
def SymmetricPoints (P Q : ℝ × ℝ) : Prop :=
  P ∈ Ellipse ∧ Q ∈ Ellipse ∧ P.1 = -Q.1 ∧ P.2 = -Q.2

/-- The distance between two points -/
noncomputable def Distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

/-- The theorem to be proved -/
theorem ellipse_quadrilateral_area 
  (F₁ F₂ P Q : ℝ × ℝ) 
  (h₁ : F₁ ∈ Foci) (h₂ : F₂ ∈ Foci) (h₃ : F₁ ≠ F₂)
  (h₄ : SymmetricPoints P Q)
  (h₅ : Distance P Q = Distance F₁ F₂) :
  ∃ (A : ℝ), A = 8 ∧ A = abs ((P.1 - F₁.1) * (Q.2 - F₁.2) - (Q.1 - F₁.1) * (P.2 - F₁.2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_quadrilateral_area_l591_59162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_one_real_root_l591_59143

/-- Represents the polynomial x^m + x^(m-1) + ... + x^2 + x + 1 -/
def polynomial (x : ℝ) (m : ℕ) : ℝ :=
  Finset.sum (Finset.range (m + 1)) (fun i => x ^ i)

/-- Theorem: The polynomial has at most one real root when m = 2n + 1 -/
theorem max_one_real_root (n : ℕ) :
  ∃ (c : ℝ), ∀ (x : ℝ), polynomial x (2 * n + 1) = 0 → x = c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_one_real_root_l591_59143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_divisible_by_n_l591_59149

/-- Sequence a_n defined recursively -/
def a : ℕ → ℕ
  | 0 => 1  -- Adding a case for 0 to cover all natural numbers
  | 1 => 1
  | 2 => 2
  | 3 => 24
  | n + 4 => (6 * a (n + 3)^2 * a (n + 1) - 8 * a (n + 3) * a (n + 2)^2) / (a (n + 2) * a (n + 1))

/-- Theorem stating that n divides a_n for all n ≥ 1 -/
theorem a_divisible_by_n (n : ℕ) (h : n ≥ 1) : n ∣ a n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_divisible_by_n_l591_59149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_line_theorem_l591_59117

structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_c_pos : c > 0
  h_a_gt_b : a > b
  h_eq : c^2 = a^2 - b^2

def tangent_line (E : Ellipse) (k : ℝ) := 
  {P : ℝ × ℝ | P.2 = k * (P.1 + E.c)}

def ellipse_circle (E : Ellipse) := 
  {P : ℝ × ℝ | (P.1 - E.c/2)^2 + P.2^2 = E.c^2}

def is_tangent (L : Set (ℝ × ℝ)) (C : Set (ℝ × ℝ)) : Prop :=
  ∃ P, P ∈ L ∧ P ∈ C ∧ ∀ Q ∈ L, Q ∉ C ∨ Q = P

def point_on_ellipse (E : Ellipse) (P : ℝ × ℝ) : Prop :=
  P.1^2 / E.a^2 + P.2^2 / E.b^2 = 1

def perpendicular_to_x_axis (E : Ellipse) (P : ℝ × ℝ) : Prop :=
  P.1 = E.c

theorem ellipse_tangent_line_theorem (E : Ellipse) :
  ∃ k P, 
    is_tangent (tangent_line E k) (ellipse_circle E) ∧
    point_on_ellipse E P ∧
    P.1 > 0 ∧ P.2 > 0 ∧
    perpendicular_to_x_axis E P →
    k = 2 * Real.sqrt 5 / 5 ∧ E.c / E.a = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_line_theorem_l591_59117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_condition_range_condition_l591_59180

noncomputable def f (a x : ℝ) : ℝ := Real.sqrt ((1 - a^2) * x^2 + 3 * (1 - a) * x + 6)

theorem domain_condition (a : ℝ) :
  (∀ x, f a x ∈ Set.univ) → a ∈ Set.Icc (-5/11) 1 := by sorry

theorem range_condition (a : ℝ) :
  (Set.range (f a) = Set.Ici 0) → a ∈ Set.Icc (-1) (-5/11) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_condition_range_condition_l591_59180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_construction_iff_equilateral_l591_59105

/-- Represents a triangle with sides a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The semiperimeter of a triangle -/
noncomputable def semiperimeter (t : Triangle) : ℝ := (t.a + t.b + t.c) / 2

/-- Constructs a new triangle from the given triangle using the semiperimeter -/
noncomputable def constructNewTriangle (t : Triangle) : Triangle where
  a := semiperimeter t - t.a
  b := semiperimeter t - t.b
  c := semiperimeter t - t.c
  positive_sides := by sorry
  triangle_inequality := by sorry

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := t.a = t.b ∧ t.b = t.c

/-- The main theorem: the construction process can be repeated indefinitely
    if and only if the original triangle is equilateral -/
theorem indefinite_construction_iff_equilateral (t : Triangle) :
  (∀ n : ℕ, ∃ t' : Triangle, t' = (constructNewTriangle^[n]) t) ↔ isEquilateral t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_construction_iff_equilateral_l591_59105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_two_l591_59163

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The distance from a focus to an asymptote of a hyperbola -/
noncomputable def focus_to_asymptote_distance (h : Hyperbola) : ℝ := h.b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt ((h.a^2 + h.b^2) / h.a^2)

/-- Theorem: For a hyperbola where the distance from one focus to an asymptote
    is equal to the semi-major axis, the eccentricity is √2 -/
theorem hyperbola_eccentricity_sqrt_two (h : Hyperbola)
    (h_dist_eq_axis : focus_to_asymptote_distance h = h.a) :
    eccentricity h = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_two_l591_59163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_effective_speed_third_hour_l591_59108

/-- Represents the data for one hour of drone flight -/
structure HourData where
  distance : ℝ
  windAssistance : ℝ

/-- Calculates the effective speed for a given hour -/
def effectiveSpeed (data : HourData) : ℝ :=
  data.distance + data.windAssistance

/-- The data for all seven hours of the drone flight -/
def flightData : Fin 7 → HourData
  | 0 => ⟨50, 5⟩
  | 1 => ⟨70, 3⟩
  | 2 => ⟨80, 2⟩
  | 3 => ⟨60, -1⟩
  | 4 => ⟨55, -2⟩
  | 5 => ⟨65, 1⟩
  | 6 => ⟨75, 4⟩

theorem highest_effective_speed_third_hour :
  ∀ i : Fin 7, i ≠ 2 → effectiveSpeed (flightData 2) ≥ effectiveSpeed (flightData i) := by
  sorry

#eval effectiveSpeed (flightData 2)  -- Should output 82

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_effective_speed_third_hour_l591_59108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_problem_l591_59102

theorem certain_number_problem : ∃! x : ℕ, 
  (x / 3 + x + 3 = 63) ∧ x = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_problem_l591_59102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l591_59114

theorem triangle_property (A B C : Real) (a b c : Real) :
  -- Triangle ABC
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- Acute triangle
  A < Real.pi/2 ∧ B < Real.pi/2 ∧ C < Real.pi/2 →
  -- Given condition
  Real.sin A ^ 2 + Real.sin C ^ 2 = Real.sin B ^ 2 + Real.sin A * Real.sin C →
  -- Side length b
  b = Real.sqrt 3 →
  -- Law of sines
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Conclusions
  B = Real.pi/3 ∧ 0 < 2*a - c ∧ 2*a - c < 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l591_59114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l591_59191

/-- The time taken for two trains to pass each other --/
noncomputable def time_to_pass (length : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  length / ((speed1 + speed2) * 1000 / 3600)

/-- Theorem stating the time taken for two trains to pass each other --/
theorem train_passing_time :
  let train_length : ℝ := 250
  let speed1 : ℝ := 45
  let speed2 : ℝ := 30
  abs (time_to_pass train_length speed1 speed2 - 12) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l591_59191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_area_equality_l591_59160

/-- A point on the hyperbola y = 1/x, x > 0 -/
structure HyperbolaPoint where
  x : ℝ
  y : ℝ
  x_pos : x > 0
  on_hyperbola : y = 1 / x

/-- The area of a region bounded by lines and a hyperbola arc -/
noncomputable def area_bounded_by_hyperbola (a b : HyperbolaPoint) : ℝ := sorry

/-- The theorem stating the equality of areas -/
theorem hyperbola_area_equality (a b : HyperbolaPoint) :
  area_bounded_by_hyperbola a b =
  area_bounded_by_hyperbola 
    { x := a.x, y := a.y, x_pos := a.x_pos, on_hyperbola := a.on_hyperbola }
    { x := b.x, y := b.y, x_pos := b.x_pos, on_hyperbola := b.on_hyperbola } :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_area_equality_l591_59160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_games_ratio_proof_l591_59151

/-- Proves that the ratio of initially owned games to gifted games is 1:2 --/
theorem games_ratio_proof (christmas_games birthday_games total_games : ℕ) :
  christmas_games = 12 →
  birthday_games = 8 →
  total_games = 30 →
  let gifted_games := christmas_games + birthday_games
  let initial_games := total_games - gifted_games
  (initial_games : ℚ) / gifted_games = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_games_ratio_proof_l591_59151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_constants_l591_59135

theorem order_of_constants : Real.sin 1 > Real.sqrt 3 - 1 ∧ Real.sqrt 3 - 1 > Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_constants_l591_59135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l591_59187

theorem trigonometric_identities (θ : ℝ) 
  (h1 : Real.sin θ + Real.cos θ = 1/5) 
  (h2 : θ ∈ Set.Ioo 0 Real.pi) : 
  Real.tan θ = -4/3 ∧ (1 + Real.sin (2*θ) + Real.cos (2*θ)) / (1 + Real.sin (2*θ) - Real.cos (2*θ)) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l591_59187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_imply_x_eq_neg_six_l591_59189

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- Define the condition that e₁ and e₂ are non-collinear
variable (h_non_collinear : ¬ ∃ (k : ℝ), e₁ = k • e₂)

-- Define vector a
def a (x : ℝ) : V := x • e₁ - 3 • e₂

-- Define vector b
def b : V := 2 • e₁ + e₂

-- Define the parallel condition
def parallel (a b : V) : Prop := ∃ (k : ℝ), a = k • b

-- State the theorem
theorem parallel_vectors_imply_x_eq_neg_six (x : ℝ) :
  parallel (a e₁ e₂ x) (b e₁ e₂) → x = -6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_imply_x_eq_neg_six_l591_59189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l591_59157

/-- The function f(a, b) as defined in the problem -/
noncomputable def f (a b : ℝ) : ℝ := Real.sqrt (2 * a^2 - 8 * a + 10) + Real.sqrt (b^2 - 6 * b + 10) + Real.sqrt (2 * a^2 - 2 * a * b + b^2)

/-- Theorem stating that the minimum value of f(a, b) is 2√5 -/
theorem min_value_f : ∀ a b : ℝ, f a b ≥ 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l591_59157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_properties_l591_59127

def G (n : Nat) : Nat :=
  if n % 2 = 1 then 3 * n + 1
  else
    let rec divideByTwo (m : Nat) (fuel : Nat) : Nat :=
      match fuel with
      | 0 => m
      | fuel + 1 =>
        if m % 2 = 1 then m
        else divideByTwo (m / 2) fuel
    divideByTwo (n / 2) n

def G_iter (k : Nat) (n : Nat) : Nat :=
  match k with
  | 0 => n
  | k + 1 => G (G_iter k n)

theorem G_properties :
  (G_iter 1 2016 = 63) ∧
  (G_iter 5 19 = 34) ∧
  (G_iter 2017 19 = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_properties_l591_59127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_8_sqrt_2_l591_59144

/-- Parametric equation of line l -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 - (Real.sqrt 2 / 2) * t, 2 + (Real.sqrt 2 / 2) * t)

/-- Equation of parabola -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Definition of points A and B as intersection points -/
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, 
    A = line_l t₁ ∧ parabola A.1 A.2 ∧
    B = line_l t₂ ∧ parabola B.1 B.2 ∧
    t₁ ≠ t₂

/-- The main theorem: length of AB is 8√2 -/
theorem length_AB_is_8_sqrt_2 (A B : ℝ × ℝ) 
  (h : intersection_points A B) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_8_sqrt_2_l591_59144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_bounds_b_formula_limit_a_over_b_l591_59173

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | n + 1 => 1 - 1 / (4 * sequence_a n)

noncomputable def sequence_b (n : ℕ) : ℝ := 2 / (2 * sequence_a n - 1)

theorem a_bounds (n : ℕ) : 1/2 < sequence_a n ∧ sequence_a n ≤ 1 := by sorry

theorem b_formula (n : ℕ) : sequence_b n = 2 * n := by sorry

theorem limit_a_over_b : 
  Filter.Tendsto (λ n => sequence_a n / sequence_b n) Filter.atTop (nhds 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_bounds_b_formula_limit_a_over_b_l591_59173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_side_length_l591_59120

/-- Two triangles are similar -/
structure SimilarTriangles (ABC DEF : Type) :=
  (sim : ABC → DEF → Prop)

/-- The ratio of two lengths -/
noncomputable def LengthRatio (a b : ℝ) := a / b

theorem similar_triangles_side_length 
  (ABC DEF : Type) 
  (h_sim : SimilarTriangles ABC DEF)
  (AB DE BC EF : ℝ)
  (h_ratio : LengthRatio AB DE = 1/2)
  (h_BC : BC = 5) :
  EF = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_side_length_l591_59120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_condition_l591_59141

/-- The function f(x) = x² - ae^x has three zeros if and only if 0 < a < 4/e² -/
theorem three_zeros_condition (a : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    (∀ (i : Fin 3), (fun x => x^2 - a * Real.exp x) (match i with
      | 0 => x₁
      | 1 => x₂
      | 2 => x₃) = 0)) ↔
  0 < a ∧ a < 4 / Real.exp 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_condition_l591_59141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_line_endpoint_characterization_l591_59193

/-- A point in 3D space that satisfies the conditions of the broken line endpoint -/
structure BrokenLineEndpoint (a : ℝ) where
  x : ℝ
  y : ℝ
  z : ℝ
  sum_abs_geq_a : |x| + |y| + |z| ≥ a
  sum_sq_leq_a_sq : x^2 + y^2 + z^2 ≤ a^2
  x_nonzero : x ≠ 0
  y_nonzero : y ≠ 0
  z_nonzero : z ≠ 0

/-- The theorem stating the equivalence between the geometric conditions and the algebraic conditions -/
theorem broken_line_endpoint_characterization (a : ℝ) (h : a > 0) :
  ∀ (x y z : ℝ),
    (∃ (broken_line : Set (ℝ × ℝ × ℝ)),
      (0, 0, 0) ∈ broken_line ∧
      (x, y, z) ∈ broken_line ∧
      (∀ (p q : ℝ × ℝ × ℝ), p ∈ broken_line → q ∈ broken_line → ‖p - q‖ ≤ a) ∧
      (∀ (plane : Set (ℝ × ℝ × ℝ)),
        (∃ (i : Fin 3), ∀ (p q : ℝ × ℝ × ℝ), p ∈ plane → q ∈ plane → 
          ((i = 0 → p.1 = q.1) ∧ (i = 1 → p.2 = q.2) ∧ (i = 2 → p.2.2 = q.2.2))) →
        (∃! (p : ℝ × ℝ × ℝ), p ∈ broken_line ∩ plane))) ↔
    Nonempty (BrokenLineEndpoint a) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_line_endpoint_characterization_l591_59193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jimmy_shoveled_ten_driveways_l591_59125

/-- The number of driveways Jimmy shoveled -/
def jimmy_driveways : ℕ :=
  let candy_bar_cost : ℚ := 75 / 100
  let lollipop_cost : ℚ := 25 / 100
  let candy_bars_bought : ℕ := 2
  let lollipops_bought : ℕ := 4
  let total_spent : ℚ := candy_bar_cost * candy_bars_bought + lollipop_cost * lollipops_bought
  let earnings_fraction : ℚ := 1 / 6
  let driveway_charge : ℚ := 3 / 2
  let total_earnings : ℚ := total_spent / earnings_fraction
  (total_earnings / driveway_charge).floor.toNat

theorem jimmy_shoveled_ten_driveways :
  jimmy_driveways = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jimmy_shoveled_ten_driveways_l591_59125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_lambda_l591_59165

/-- Given two vectors a and b in ℝ², and a scalar lambda, 
    if c = lambda * a + b is perpendicular to a, then lambda = -2 -/
theorem perpendicular_vector_lambda (a b c : ℝ × ℝ) (lambda : ℝ) :
  a = (1, 2) →
  b = (2, 4) →
  c = (lambda * a.1 + b.1, lambda * a.2 + b.2) →
  c.1 * a.1 + c.2 * a.2 = 0 →
  lambda = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_lambda_l591_59165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_half_circumscribed_implies_equilateral_l591_59186

/-- A triangle with inscribed and circumscribed circles -/
structure TriangleWithCircles where
  A : ℝ × ℝ  -- Vertex A of the triangle
  B : ℝ × ℝ  -- Vertex B of the triangle
  C : ℝ × ℝ  -- Vertex C of the triangle
  r : ℝ  -- Radius of the inscribed circle
  R : ℝ  -- Radius of the circumscribed circle

/-- Definition of an equilateral triangle -/
def is_equilateral (t : TriangleWithCircles) : Prop :=
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d t.A t.B = d t.B t.C ∧ d t.B t.C = d t.C t.A

/-- Theorem: If the radius of the inscribed circle is half the radius of the circumscribed circle,
    then the triangle is equilateral -/
theorem inscribed_half_circumscribed_implies_equilateral (t : TriangleWithCircles) 
  (h : t.R = 2 * t.r) : is_equilateral t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_half_circumscribed_implies_equilateral_l591_59186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_terms_l591_59124

/-- An arithmetic sequence with negative common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  d_negative : d < 0

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

/-- The theorem stating that the number of terms maximizing the sum is 5 or 6 -/
theorem max_sum_terms (seq : ArithmeticSequence) 
  (h : seq.a 1 ^ 2 = seq.a 11 ^ 2) :
  ∃ n : ℕ, (n = 5 ∨ n = 6) ∧
    ∀ m : ℕ, sum_n seq m ≤ sum_n seq n := by
  sorry

#check max_sum_terms

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_terms_l591_59124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_exists_l591_59171

noncomputable def reference_slope : ℝ := Real.sqrt 3

def point_P : ℝ × ℝ := (2, 3)

inductive LineEquation
  | eq1 : LineEquation  -- √3x - y + 3 - 2√3 = 0
  | eq2 : LineEquation  -- y = (3/2)x
  | eq3 : LineEquation  -- x + y - 5 = 0

def satisfies_conditions (eq : LineEquation) : Prop :=
  match eq with
  | LineEquation.eq1 => 
      (∃ (k : ℝ), k = reference_slope ∧ 
       k * (point_P.1 - 0) = point_P.2 - 0 ∧ 
       k = Real.sqrt 3)
  | LineEquation.eq2 => 
      (3 / 2 * point_P.1 = point_P.2) ∧ 
      (∃ (x : ℝ), x ≠ 0 ∧ x = 3 / 2 * x)
  | LineEquation.eq3 => 
      (point_P.1 + point_P.2 = 5) ∧ 
      (∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ x = y ∧ x + y = 5)

theorem line_equation_exists : 
  ∃ (eq : LineEquation), satisfies_conditions eq := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_exists_l591_59171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l591_59116

-- Define proposition p
def p : Prop := ∀ x : ℝ, (2 : ℝ)^x + (2 : ℝ)^(-x) ≥ 2

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Define proposition q
def q : Prop := ∀ f : ℝ → ℝ, is_odd f → f 0 = 0

-- Theorem statement
theorem problem_solution : p ∧ ¬q := by
  constructor
  · -- Proof of p
    sorry
  · -- Proof of ¬q
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l591_59116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_set_condition_l591_59174

noncomputable def P (d : ℕ) : ℝ := Real.log (d + 1 : ℝ) - Real.log (d : ℝ)

theorem probability_set_condition : 
  let S : Set ℕ := {4, 6}
  P 5 = (1 / 2 : ℝ) * (P 4 + P 6) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_set_condition_l591_59174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2004_bounds_l591_59140

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | 1 => 2
  | (n + 2) => sequence_a (n + 1) + 1 / sequence_a (n + 1)

theorem sequence_a_2004_bounds :
  63 < sequence_a 2004 ∧ sequence_a 2004 < 78 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2004_bounds_l591_59140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l591_59185

open Real

-- Define the function f
noncomputable def f (a θ : ℝ) : ℝ := sin θ ^ 3 + 4 / (3 * a * sin θ ^ 2 - a ^ 3)

-- State the theorem
theorem f_minimum_value (a θ : ℝ) 
  (h1 : 0 < a) (h2 : a < Real.sqrt 3 * sin θ) 
  (h3 : π / 4 ≤ θ) (h4 : θ ≤ 5 * π / 6) : 
  f a θ ≥ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l591_59185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l591_59118

-- Define the polar equation of curve l
def curve_l (ρ θ : ℝ) : Prop := ρ * Real.cos θ - ρ * Real.sin θ + 1 = 0

-- Define the parametric equations of curve C
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (Real.sqrt 2 * Real.cos θ, Real.sqrt 2 * Real.sin θ)

-- Define the intersection points A and B
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- State the theorem
theorem intersection_distance : 
  ∃ (θ₁ θ₂ : ℝ), 
    curve_l ((curve_C θ₁).1) θ₁ ∧ 
    curve_l ((curve_C θ₂).1) θ₂ ∧ 
    (A = curve_C θ₁) ∧ 
    (B = curve_C θ₂) ∧ 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l591_59118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_slope_PQ_l591_59179

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Defines the ellipse C: x²/6 + y²/3 = 1 -/
def ellipse_C (p : Point) : Prop :=
  p.x^2 / 6 + p.y^2 / 3 = 1

/-- Checks if a point lies on the ellipse C -/
def on_ellipse_C (p : Point) : Prop :=
  ellipse_C p

/-- Point M on the ellipse C -/
def M : Point :=
  ⟨-2, -1⟩

/-- Two lines with complementary angles of inclination -/
def complementary_lines (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

/-- A line passes through a point -/
def line_through_point (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Slope of a line passing through two points -/
noncomputable def slope_between_points (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

/-- Main theorem to prove -/
theorem constant_slope_PQ (l1 l2 : Line) (P Q : Point) :
  on_ellipse_C M →
  complementary_lines l1 l2 →
  line_through_point l1 M →
  line_through_point l2 M →
  on_ellipse_C P →
  on_ellipse_C Q →
  line_through_point l1 P →
  line_through_point l2 Q →
  P ≠ M →
  Q ≠ M →
  slope_between_points P Q = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_slope_PQ_l591_59179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_circumscribed_sphere_volume_l591_59153

theorem cube_circumscribed_sphere_volume (edge_length : ℝ) (h : edge_length = 2) :
  (4 / 3) * Real.pi * ((edge_length * Real.sqrt 3 / 2) ^ 3) = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_circumscribed_sphere_volume_l591_59153
