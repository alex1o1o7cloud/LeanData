import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_equal_division_l1096_109617

/-- Represents a tetrahedron with an inscribed sphere -/
structure Tetrahedron where
  volume : ℝ
  surfaceArea : ℝ
  inscribedSphereCenter : Fin 3 → ℝ
  inscribedSphereRadius : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Fin 3 → ℝ
  point : Fin 3 → ℝ

/-- Represents the two parts of a divided tetrahedron -/
structure DividedTetrahedron where
  part1Volume : ℝ
  part1SurfaceArea : ℝ
  part2Volume : ℝ
  part2SurfaceArea : ℝ

/-- States that there exists a plane dividing a tetrahedron into equal parts -/
theorem tetrahedron_equal_division (t : Tetrahedron) :
  ∃ (p : Plane), p.point = t.inscribedSphereCenter ∧
  ∃ (d : DividedTetrahedron),
    d.part1Volume = d.part2Volume ∧
    d.part1SurfaceArea = d.part2SurfaceArea := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_equal_division_l1096_109617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_solution_set_l1096_109648

noncomputable def f (a b x : ℝ) : ℝ := Real.log (a^x - b^x) / Real.log 10

theorem f_positive_solution_set (a b : ℝ) 
  (h1 : a > 1) (h2 : 1 > b) (h3 : b > 0) (h4 : a^2 = b^2 + 1) :
  {x : ℝ | f a b x > 0} = Set.Ioi 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_solution_set_l1096_109648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1096_109682

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 - x^2)

def domain_of_f : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }

theorem f_domain : 
  ∀ x : ℝ, x ∈ domain_of_f ↔ ∃ y : ℝ, f x = y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1096_109682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_freelance_increase_l1096_109692

/-- Calculates the percentage increase between two values -/
noncomputable def percentageIncrease (initial : ℝ) (final : ℝ) : ℝ :=
  (final - initial) / initial * 100

theorem johns_freelance_increase :
  let initialPartTimeEarnings : ℝ := 65
  let finalPartTimeEarnings : ℝ := 72
  let initialFreelanceEarnings : ℝ := 45
  let finalFreelanceEarnings : ℝ := finalPartTimeEarnings
  percentageIncrease initialFreelanceEarnings finalFreelanceEarnings = 60 := by
  -- Unfold the definition of percentageIncrease
  unfold percentageIncrease
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_freelance_increase_l1096_109692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1096_109661

noncomputable section

variable (f : ℝ → ℝ)

axiom f_domain : ∀ x : ℝ, x > 0 → ∃ y, f x = y

axiom f_neg_for_gt_one : ∀ x > 1, f x < 0
axiom f_half_eq_one : f (1/2) = 1
axiom f_product : ∀ x y : ℝ, x > 0 → y > 0 → f (x * y) = f x + f y

theorem f_properties :
  (∀ x : ℝ, x > 0 → f (1/x) = -f x) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → f x > f y) ∧
  (Set.Icc 3 5 = {x : ℝ | f 2 + f (5-x) ≥ -2}) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1096_109661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_theorem_l1096_109611

-- Define the cone properties
noncomputable def slant_height : ℝ := 2
noncomputable def base_area : ℝ := Real.pi

-- Define the volume function for a cone
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

-- Theorem statement
theorem cone_volume_theorem : 
  ∃ (r h : ℝ), 
    r^2 * Real.pi = base_area ∧ 
    r^2 + h^2 = slant_height^2 ∧
    cone_volume r h = (Real.sqrt 3 * Real.pi) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_theorem_l1096_109611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_l_tiling_existence_l1096_109640

/-- An L-shaped tile covers exactly three unit squares in a grid. -/
structure LTile where
  squares : Finset (ℕ × ℕ)
  size : squares.card = 3

/-- A tiling of a grid with L-shaped tiles. -/
structure Tiling (m n : ℕ) where
  tiles : Set LTile
  covers : ∀ i j, i < m ∧ j < n → ∃! t : LTile, t ∈ tiles ∧ (i, j) ∈ t.squares

/-- The statement of the theorem. -/
theorem l_tiling_existence (n : ℕ) (h1 : n > 0) (h2 : ¬ 3 ∣ n) :
  ∀ i j, i < 2*n ∧ j < 2*n →
    ∃ (tiling : Tiling (2*n) (2*n)) (removed : ℕ × ℕ),
      removed.1 < 2*n ∧ removed.2 < 2*n ∧
      ∀ x y, (x, y) ≠ removed → (x < 2*n ∧ y < 2*n → 
        ∃ t : LTile, t ∈ tiling.tiles ∧ (x, y) ∈ t.squares) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_l_tiling_existence_l1096_109640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_b_more_cost_effective_l1096_109619

/-- Represents the cost of bananas at Store A -/
def cost_a (x : ℝ) : ℝ := 4 * x

/-- Represents the cost of bananas at Store B -/
noncomputable def cost_b (x : ℝ) : ℝ := if x ≤ 6 then 5 * x else 3.5 * x + 9

/-- Theorem stating that Store B is more cost-effective for 50 kg of bananas -/
theorem store_b_more_cost_effective : cost_b 50 < cost_a 50 := by
  -- Unfold the definitions
  unfold cost_a cost_b
  -- Simplify the if-then-else expression
  simp
  -- Perform numerical calculations
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_b_more_cost_effective_l1096_109619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l1096_109609

noncomputable def curve (x : ℝ) : ℝ := 12 / (x^2 + 9)
noncomputable def line (x : ℝ) : ℝ := 3 - x

theorem intersection_points :
  {x : ℝ | curve x = line x} = {-1, 3, -5} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l1096_109609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_points_equation_l1096_109607

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define points
variable (O A B C D : V)

-- Define the origin
def is_origin (O : V) : Prop := O = 0

-- Define coplanarity
def coplanar (A B C D : V) : Prop :=
  ∃ (a b c d : ℝ), a • (B - A) + b • (C - A) + c • (D - A) = 0 ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0)

-- State the theorem
theorem coplanar_points_equation (h_origin : is_origin O)
  (h_equation : ∃ m : ℝ, 4 • (A - O) - 3 • (B - O) + 6 • (C - O) + m • (D - O) = 0)
  (h_coplanar : coplanar A B C D) :
  ∃ m : ℝ, 4 • (A - O) - 3 • (B - O) + 6 • (C - O) + m • (D - O) = 0 ∧ m = -7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_points_equation_l1096_109607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_y_squared_over_eight_l1096_109681

/-- An isosceles right-angled triangle with hypotenuse length y -/
structure IsoscelesRightTriangle where
  y : ℝ
  y_pos : y > 0

/-- The shaded area within the triangle -/
noncomputable def shadedArea (t : IsoscelesRightTriangle) : ℝ := t.y^2 / 8

/-- Theorem stating that the shaded area is y²/8 -/
theorem shaded_area_is_y_squared_over_eight (t : IsoscelesRightTriangle) :
  shadedArea t = t.y^2 / 8 := by
  -- Unfold the definition of shadedArea
  unfold shadedArea
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_y_squared_over_eight_l1096_109681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angle_corridor_wire_theorem_l1096_109655

theorem right_angle_corridor_wire_theorem :
  ∃ (wire_length : ℝ), 
    let corridor_width : ℝ := 1
    let wire_ends_distance : ℝ := 2 + 2 * Real.sqrt 2
    wire_length > wire_ends_distance ∧
    wire_ends_distance > 4 ∧
    (∃ (wire_path : ℝ → ℝ × ℝ),
      (∀ t, 0 ≤ t ∧ t ≤ 1 →
        (wire_path t).1 ≥ 0 ∧ (wire_path t).2 ≥ 0 ∧
        ((wire_path t).1 ≤ corridor_width ∨ (wire_path t).2 ≤ corridor_width)) ∧
      (wire_path 0).1 = 0 ∧
      (wire_path 1).2 = 0 ∧
      (∀ t, 0 ≤ t ∧ t ≤ 1 →
        (wire_path t).1^2 + (wire_path t).2^2 ≤ wire_length^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angle_corridor_wire_theorem_l1096_109655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1096_109697

theorem triangle_problem (A B C a b c : ℝ) : 
  -- Given conditions
  Real.sin (A + C) = 8 * (Real.sin (B / 2))^2 →
  A + B + C = Real.pi →
  a + c = 6 →
  (1/2) * a * c * Real.sin B = 2 →
  -- Conclusion
  Real.cos B = 15/17 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1096_109697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_B_wins_one_A_wins_match_prob_distribution_X_expected_value_X_l1096_109644

/-- Represents the outcome of a single game -/
inductive GameOutcome
| A_Wins
| B_Wins

/-- Represents the state of the match -/
structure MatchState where
  games_played : ℕ
  consecutive_wins : ℕ
  leader : GameOutcome

/-- Probability of player A winning a single game -/
noncomputable def prob_A_wins : ℝ := 2/3

/-- Probability of player B winning a single game -/
noncomputable def prob_B_wins : ℝ := 1/3

/-- The maximum number of games that can be played in a match -/
def max_games : ℕ := 5

/-- Determines if the match is over given the current state -/
def is_match_over (state : MatchState) : Bool :=
  state.consecutive_wins = 2 || state.games_played = max_games

/-- The random variable representing the number of games played in a match -/
noncomputable def X : ℕ → ℝ 
| 2 => 5/9
| 3 => 2/9
| 4 => 10/81
| 5 => 8/81
| _ => 0

/-- Theorem stating the probability of B winning exactly one game and A winning the match -/
theorem prob_B_wins_one_A_wins_match : 
  (X 3 * 1/3 * (2/3)^2) + (X 4 * 2/3 * 1/3 * (2/3)^2) = 20/81 := by sorry

/-- Theorem stating the probability distribution of X -/
theorem prob_distribution_X : 
  (X 2 = 5/9) ∧ (X 3 = 2/9) ∧ (X 4 = 10/81) ∧ (X 5 = 8/81) := by sorry

/-- Theorem stating the expected value of X -/
theorem expected_value_X : 
  2 * X 2 + 3 * X 3 + 4 * X 4 + 5 * X 5 = 224/81 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_B_wins_one_A_wins_match_prob_distribution_X_expected_value_X_l1096_109644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1096_109621

-- Define variables as parameters
variable (a b : ℚ) (c : ℤ)

-- Condition 1: (3a-2b-1)^2 = 3^2
axiom cond1 : (3*a - 2*b - 1)^2 = 3^2

-- Condition 2: a+2b = -8
axiom cond2 : a + 2*b = -8

-- Condition 3: c is the integer part of 2+√7
axiom cond3 : c = Int.floor (2 + Real.sqrt 7)

theorem problem_solution :
  (a = 2 ∧ b = -2 ∧ c = 4) ∧
  (∃ (x : ℝ), x^2 = a - b + c ∧ (x = 2 * Real.sqrt 2 ∨ x = -2 * Real.sqrt 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1096_109621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_triangle_areas_area_between_curves_l1096_109604

noncomputable section

-- Define the function C
noncomputable def C (x : ℝ) : ℝ := (1/2) * x + Real.sqrt ((1/4) * x^2 + 2)

-- Define the points P1 and P2 on C
noncomputable def P1 (x₁ : ℝ) : ℝ × ℝ := (x₁, C x₁)
noncomputable def P2 (x₂ : ℝ) : ℝ × ℝ := (x₂, C x₂)

-- Define H1 and H2
noncomputable def H1 (x₁ : ℝ) : ℝ × ℝ := (C x₁, C x₁)
noncomputable def H2 (x₂ : ℝ) : ℝ × ℝ := (C x₂, C x₂)

-- Define the area of triangle OPH
noncomputable def triangleArea (P : ℝ × ℝ) (H : ℝ × ℝ) : ℝ :=
  (1/2) * abs (P.1 * H.2 - H.1 * P.2)

theorem equal_triangle_areas (x₁ x₂ : ℝ) :
  triangleArea (P1 x₁) (H1 x₁) = triangleArea (P2 x₂) (H2 x₂) := by
  sorry

theorem area_between_curves (x₁ x₂ : ℝ) (h : x₁ < x₂) :
  ∃ (area : ℝ), area = 2 * Real.log ((C x₂) / (C x₁)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_triangle_areas_area_between_curves_l1096_109604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_line_equations_l1096_109615

-- Define the points
def P : ℝ × ℝ := (26, 1)
def Q : ℝ × ℝ := (2, 1)
def N : ℝ × ℝ := (-2, 3)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the circle
def is_on_circle (p : ℝ × ℝ) : Prop :=
  (p.1 - 1)^2 + (p.2 - 1)^2 = 25

-- Define the lines
def line1 (p : ℝ × ℝ) : Prop := p.1 = -2
def line2 (p : ℝ × ℝ) : Prop := 5*p.1 - 12*p.2 + 46 = 0

-- Theorem statement
theorem trajectory_and_line_equations 
  (M : ℝ × ℝ) 
  (h1 : distance M P = 5 * distance M Q) :
  (is_on_circle M) ∧ 
  (∃ (l : (ℝ × ℝ) → Prop), 
    (l N) ∧ 
    (∀ (p : ℝ × ℝ), l p ↔ (line1 p ∨ line2 p)) ∧
    (∃ (A B : ℝ × ℝ), 
      is_on_circle A ∧ 
      is_on_circle B ∧ 
      l A ∧ 
      l B ∧ 
      distance A B = 8)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_line_equations_l1096_109615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l1096_109653

theorem sin_double_angle (x : ℝ) (h : Real.sin (π / 4 - x) = 3 / 5) : 
  Real.sin (2 * x) = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l1096_109653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_elephant_drinking_time_l1096_109600

/-- The number of days it takes for one elephant to drink a lake -/
def days_for_one_elephant : ℕ := 365

/-- Theorem stating that one elephant will take 365 days to drink the lake -/
theorem one_elephant_drinking_time :
  days_for_one_elephant = 365 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_elephant_drinking_time_l1096_109600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l1096_109698

/-- A power function with an integer exponent -/
noncomputable def f (m : ℤ) : ℝ → ℝ := fun x ↦ x^(-m^2 + 2*m + 3)

/-- The square root of f plus a linear term -/
noncomputable def g (m : ℤ) (c : ℝ) : ℝ → ℝ := fun x ↦ Real.sqrt (f m x) + 2*x + c

/-- The theorem stating the properties of f and g -/
theorem f_and_g_properties :
  ∃ (m : ℤ),
    (∀ x, f m x = f m (-x)) ∧  -- f is even
    (∀ x y, 0 < x ∧ x < y → f m x < f m y) ∧  -- f is monotonically increasing on (0, +∞)
    (∀ x, g m 3 x > 2) →  -- g(x) > 2 for all x when c = 3
  (∀ x, f m x = x^4) ∧ (∀ c, (∀ x, g m c x > 2) → c > 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l1096_109698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_crosses_at_point_l1096_109614

/-- The x-coordinate of a point on the curve as a function of t -/
def x (t : ℝ) : ℝ := t^3 - 3*t^2 - 2

/-- The y-coordinate of a point on the curve as a function of t -/
def y (t : ℝ) : ℝ := t^4 - 12*t^2 + 8

/-- The point where the curve crosses itself -/
noncomputable def crossingPoint : ℝ × ℝ := (3 * Real.sqrt 3 - 11, -19)

/-- Theorem stating that the curve crosses itself at the specified point -/
theorem curve_crosses_at_point :
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
  (x t₁, y t₁) = crossingPoint ∧ 
  (x t₂, y t₂) = crossingPoint := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_crosses_at_point_l1096_109614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1096_109635

noncomputable def f (x : ℝ) := 2 * (Real.sin x)^2 + Real.sin x * Real.cos x + (Real.cos x)^2

theorem f_properties :
  (f (π / 12) = 7 / 4 - Real.sqrt 3 / 4) ∧
  (∀ x : ℝ, f x ≥ 3 / 2 - Real.sqrt 2 / 2) ∧
  (∀ k : ℤ, f (k * π - π / 8) = 3 / 2 - Real.sqrt 2 / 2) ∧
  (∀ k : ℤ, ∀ x : ℝ, -π / 8 + k * π ≤ x ∧ x ≤ 3 * π / 8 + k * π →
    (∀ y : ℝ, -π / 8 + k * π ≤ y ∧ y ≤ x → f y ≤ f x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1096_109635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_planes_l1096_109699

/-- Represents a plane in 3D space -/
structure Plane where
  normal : ℝ × ℝ × ℝ
  d : ℝ

/-- Calculates the distance between two parallel planes -/
noncomputable def distance_between_parallel_planes (p1 p2 : Plane) : ℝ :=
  let (a, b, c) := p1.normal
  |p2.d - p1.d| / Real.sqrt (a^2 + b^2 + c^2)

/-- Theorem: The distance between two specific parallel planes is 1/√3 -/
theorem distance_specific_planes :
  let plane_A := Plane.mk (1, 1, 1) 1
  let plane_B := Plane.mk (1, 1, 1) 2
  distance_between_parallel_planes plane_A plane_B = 1 / Real.sqrt 3 := by
  sorry

#check distance_specific_planes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_planes_l1096_109699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l1096_109630

-- Define the universal set U
def U : Set ℝ := Set.Ioo (-4) 4

-- Define set A
def A : Set ℝ := {x : ℝ | 0 < -x ∧ -x ≤ 2}

-- Theorem statement
theorem complement_of_A : Set.compl A ∩ U = Set.union (Set.Ioo (-4) (-2)) (Set.Ico 0 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l1096_109630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_generating_function_binet_formula_l1096_109643

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2
noncomputable def φ_hat : ℝ := (1 - Real.sqrt 5) / 2

noncomputable def generating_function (x : ℝ) : ℝ := ∑' n, (fibonacci n : ℝ) * x^n

theorem fibonacci_generating_function (x : ℝ) (hx : |x| < 1) :
  generating_function x = x / (1 - x - x^2) ∧
  generating_function x = (1 / Real.sqrt 5) * (1 / (1 - φ * x) - 1 / (1 - φ_hat * x)) :=
sorry

theorem binet_formula (n : ℕ) :
  (fibonacci n : ℝ) = (φ^n - φ_hat^n) / Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_generating_function_binet_formula_l1096_109643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_special_case_l1096_109674

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle given two sides and the included angle -/
noncomputable def triangleArea (s1 s2 angle : ℝ) : ℝ := (1/2) * s1 * s2 * Real.sin angle

/-- Theorem stating the area of triangle ABC under given conditions -/
theorem triangle_area_special_case (t : Triangle)
    (h1 : t.c * Real.sin t.A = Real.sqrt 3 * t.a * Real.cos t.C)
    (h2 : t.c = Real.sqrt 31)
    (h3 : t.a + t.b = 7) :
    triangleArea t.a t.b t.C = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_special_case_l1096_109674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_pointed_star_sum_l1096_109685

-- Define the number of points on the circle
def n : ℕ := 9

-- Define the measure of each small arc between adjacent points
noncomputable def small_arc_measure : ℝ := 360 / n

-- Define the measure of the arc intercepted by each tip of the star
noncomputable def intercepted_arc_measure : ℝ := 4 * small_arc_measure

-- Define the measure of the angle at each tip of the star
noncomputable def tip_angle_measure : ℝ := intercepted_arc_measure / 2

-- Theorem statement
theorem nine_pointed_star_sum :
  n * tip_angle_measure = 720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_pointed_star_sum_l1096_109685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_A_minus_B_l1096_109664

def A : ℤ := 2 * 3 + 4 * 5 + 6 * 7 + 8 * 9 + 10 * 11 + 12 * 13 + 14 * 15 + 16 * 17 + 18 * 19 + 20 * 21 + 22 * 23 + 24 * 25 + 26 * 27 + 28 * 29 + 30 * 31 + 32 * 33 + 34 * 35 + 36 * 37 + 38 * 39 + 40

def B : ℤ := 2 + 3 * 4 + 5 * 6 + 7 * 8 + 9 * 10 + 11 * 12 + 13 * 14 + 15 * 16 + 17 * 18 + 19 * 20 + 21 * 22 + 23 * 24 + 25 * 26 + 27 * 28 + 29 * 30 + 31 * 32 + 33 * 34 + 35 * 36 + 37 * 38 + 39 * 40

theorem abs_A_minus_B : |A - B| = 1159 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_A_minus_B_l1096_109664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1096_109626

noncomputable def f (x : ℝ) := Real.exp x + 2 * x - 3

theorem root_in_interval :
  ∃ c ∈ Set.Ioo 0 1, f c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1096_109626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coe_has_least_money_l1096_109638

structure Person where
  name : String
  money : ℕ

def anne : Person := ⟨"Anne", 0⟩
def bo : Person := ⟨"Bo", 0⟩
def coe : Person := ⟨"Coe", 0⟩
def dan : Person := ⟨"Dan", 0⟩
def el : Person := ⟨"El", 0⟩

axiom different_amounts : anne.money ≠ bo.money ∧ anne.money ≠ coe.money ∧ anne.money ≠ dan.money ∧ anne.money ≠ el.money ∧
                          bo.money ≠ coe.money ∧ bo.money ≠ dan.money ∧ bo.money ≠ el.money ∧
                          coe.money ≠ dan.money ∧ coe.money ≠ el.money ∧
                          dan.money ≠ el.money

axiom bo_more_than_anne_dan : bo.money > anne.money ∧ bo.money > dan.money

axiom anne_el_more_than_coe : anne.money > coe.money ∧ el.money > coe.money

axiom dan_between_coe_anne : dan.money > coe.money ∧ dan.money < anne.money

theorem coe_has_least_money : ∀ p : Person, p ≠ coe → coe.money < p.money := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coe_has_least_money_l1096_109638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l1096_109675

/-- The infinite series sum from n=3 to infinity of (n^4+4n^2+10n+10) / (3^n * (n^4+4)) -/
noncomputable def infiniteSeries : ℝ := ∑' (n : ℕ), if n ≥ 3 then (n^4 + 4*n^2 + 10*n + 10) / (3^n * (n^4 + 4)) else 0

/-- Theorem stating that the infinite series sum is equal to 3 -/
theorem infiniteSeriesSum : infiniteSeries = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l1096_109675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remove_all_cards_iff_not_both_even_l1096_109620

/-- Represents a card on the board -/
inductive Card
  | White
  | Black

/-- Represents the board configuration -/
def Board (m n : Nat) := Fin m → Fin n → Card

/-- Initial board setup -/
def initialBoard (m n : Nat) : Board m n :=
  fun i j => if i.val = 0 ∧ j.val = 0 then Card.Black else Card.White

/-- Defines a valid move on the board -/
def isValidMove (m n : Nat) (b : Board m n) (i : Fin m) (j : Fin n) : Prop :=
  b i j = Card.Black

/-- Defines the result of a move on the board -/
def makeMove (m n : Nat) (b : Board m n) (i : Fin m) (j : Fin n) : Board m n :=
  fun x y => 
    if x = i ∧ y = j then
      Card.White  -- Remove the card
    else if (x = i ∧ (y.val + 1 = j.val ∨ y.val = j.val + 1)) ∨ 
            (y = j ∧ (x.val + 1 = i.val ∨ x.val = i.val + 1)) then
      match b x y with
      | Card.White => Card.Black
      | Card.Black => Card.White
    else
      b x y

/-- Defines if all cards can be removed from the board -/
def canRemoveAllCards (m n : Nat) : Prop :=
  ∃ (moves : List (Fin m × Fin n)), 
    let finalBoard := moves.foldl (fun b (i, j) => makeMove m n b i j) (initialBoard m n)
    ∀ i j, finalBoard i j = Card.White

/-- Main theorem: All cards can be removed if and only if m and n are not both even -/
theorem remove_all_cards_iff_not_both_even (m n : Nat) :
  canRemoveAllCards m n ↔ ¬(Even m ∧ Even n) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remove_all_cards_iff_not_both_even_l1096_109620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_arithmetic_l1096_109679

def A : ℂ := 6 - 2 * Complex.I
def M : ℂ := -5 + 3 * Complex.I
def S : ℂ := 2 * Complex.I
def P : ℂ := 3

theorem complex_arithmetic : A - M + S - P = 8 - 3 * Complex.I := by
  -- Expand the definitions
  simp [A, M, S, P]
  -- Simplify the complex arithmetic
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_arithmetic_l1096_109679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_constant_proof_l1096_109676

theorem fraction_constant_proof (p q : ℚ) (h : p / q = 4 / 5) : 
  ∃ C : ℚ, C + ((2 * q - p) / (2 * q + p)) = 571428571428571 / 1000000000000000 ∧ 
  C = 14285714285714 / 100000000000000 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_constant_proof_l1096_109676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_equals_4_l1096_109649

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
axiom perpendicular_lines_slope_product {m₁ m₂ : ℝ} : 
  m₁ * m₂ = -1 ↔ ∃ (a b c d e f : ℝ), 
    (∀ x y : ℝ, (a * x + b * y + c = 0 ∧ d * x + e * y + f = 0) →
      (x, y) ∈ {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0} ∩ {p : ℝ × ℝ | d * p.1 + e * p.2 + f = 0}) ∧
    m₁ = -a / b ∧ m₂ = -d / e

/-- The theorem to be proved -/
theorem perpendicular_lines_a_equals_4 : 
  ∃ a : ℝ, (∀ x y : ℝ, a * x + 2 * y - 1 = 0 ∧ 3 * x - 6 * y - 1 = 0 →
    (x, y) ∈ {p : ℝ × ℝ | a * p.1 + 2 * p.2 - 1 = 0} ∩ {p : ℝ × ℝ | 3 * p.1 - 6 * p.2 - 1 = 0}) ∧
  a = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_equals_4_l1096_109649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_problem_l1096_109636

/-- The time taken for a person to cover the entire length of an escalator -/
noncomputable def escalatorTime (escalatorSpeed : ℝ) (personSpeed : ℝ) (escalatorLength : ℝ) : ℝ :=
  escalatorLength / (escalatorSpeed + personSpeed)

/-- Proof that the time taken is 14 seconds given the specified conditions -/
theorem escalator_problem :
  let escalatorSpeed : ℝ := 12
  let personSpeed : ℝ := 2
  let escalatorLength : ℝ := 196
  escalatorTime escalatorSpeed personSpeed escalatorLength = 14 := by
  -- Unfold the definition of escalatorTime
  unfold escalatorTime
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_problem_l1096_109636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_one_fourth_l1096_109696

-- Define the function g as noncomputable
noncomputable def g (a b c : ℝ) : ℝ :=
  if a + b + c ≤ 5 then
    (a * b - a + c) / (2 * a)
  else
    (a * b - b - c) / (-2 * b)

-- Theorem to prove
theorem g_sum_equals_one_fourth :
  g 1 2 1 + g 3 2 1 = 1/4 := by
  -- Expand the definition of g for both cases
  have h1 : g 1 2 1 = 1 := by
    -- Prove that 1 + 2 + 1 ≤ 5
    have : 1 + 2 + 1 ≤ 5 := by norm_num
    -- Simplify g 1 2 1
    simp [g, this]
    norm_num
  
  have h2 : g 3 2 1 = -3/4 := by
    -- Prove that 3 + 2 + 1 > 5
    have : ¬(3 + 2 + 1 ≤ 5) := by norm_num
    -- Simplify g 3 2 1
    simp [g, this]
    norm_num
  
  -- Combine the results
  calc
    g 1 2 1 + g 3 2 1 = 1 + (-3/4) := by rw [h1, h2]
    _ = 1/4 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_one_fourth_l1096_109696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_power_difference_l1096_109646

theorem tan_power_difference (x : ℝ) : 
  (2 * Real.sin x = Real.cos x + Real.cos x / Real.sin x) → 
  (Real.tan x ^ 6 - Real.tan x ^ 2 = -15/64 ∨ Real.tan x ^ 6 - Real.tan x ^ 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_power_difference_l1096_109646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_in_first_quadrant_l1096_109624

-- Define the concept of an angle being in a specific quadrant
def InSecondQuadrant (θ : Real) : Prop := Real.pi / 2 < θ ∧ θ < Real.pi
def InFirstQuadrant (θ : Real) : Prop := 0 < θ ∧ θ < Real.pi / 2

-- State the theorem
theorem half_angle_in_first_quadrant (θ : Real) 
  (h1 : InSecondQuadrant θ) 
  (h2 : Real.cos (θ / 2) - Real.sin (θ / 2) = Real.sqrt (1 - Real.sin θ)) : 
  InFirstQuadrant (θ / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_in_first_quadrant_l1096_109624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1096_109602

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : Real.sqrt 3 * t.c = Real.sqrt 3 * t.a * Real.cos t.B - t.a * Real.sin t.B)
  (h2 : ∃ (D : ℝ), D = 3 ∧ D * (t.b + t.c) = 2 * t.b * t.c) :
  t.A = 2 * Real.pi / 3 ∧ 
  ∀ (area : ℝ), area = 1/2 * t.b * t.c * Real.sin t.A → area ≥ 9 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1096_109602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_to_polar_l1096_109623

theorem complex_sum_to_polar : 
  12 * Complex.exp (Complex.I * (4 * Real.pi / 13)) + 12 * Complex.exp (Complex.I * (9 * Real.pi / 26)) = 
  24 * Real.cos (Real.pi / 26) * Complex.exp (Complex.I * (17 * Real.pi / 52)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_to_polar_l1096_109623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_37_l1096_109691

/-- The number of hour marks on a clock face -/
def clockHourMarks : ℕ := 12

/-- The number of degrees in a full circle -/
noncomputable def fullCircleDegrees : ℝ := 360

/-- The number of degrees between each hour mark -/
noncomputable def hourMarkDegrees : ℝ := fullCircleDegrees / clockHourMarks

/-- The number of minutes in an hour -/
def minutesPerHour : ℕ := 60

/-- Calculate the angle of the minute hand at a given minute -/
noncomputable def minuteHandAngle (minute : ℕ) : ℝ :=
  (minute : ℝ) / minutesPerHour * fullCircleDegrees

/-- Calculate the angle of the hour hand at a given hour and minute -/
noncomputable def hourHandAngle (hour : ℕ) (minute : ℕ) : ℝ :=
  (hour : ℝ) * hourMarkDegrees + (minute : ℝ) / minutesPerHour * hourMarkDegrees

/-- Calculate the acute angle between the hour and minute hands -/
noncomputable def acuteAngleBetweenHands (hour : ℕ) (minute : ℕ) : ℝ :=
  let angle := abs (minuteHandAngle minute - hourHandAngle hour minute)
  min angle (fullCircleDegrees - angle)

/-- The acute angle between the hour and minute hands at 3:37 is 113.5° -/
theorem clock_angle_at_3_37 : 
  acuteAngleBetweenHands 3 37 = 113.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_37_l1096_109691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_surface_area_ratio_l1096_109628

/-- A truncated cone inscribed around a sphere -/
structure TruncatedCone (r : ℝ) where
  R₁ : ℝ
  R₂ : ℝ
  h : ℝ
  a : ℝ
  h_eq : h = 2 * r
  a_eq : a = R₁ + R₂

/-- The volume of a truncated cone -/
noncomputable def volume (tc : TruncatedCone r) : ℝ :=
  (tc.R₁^2 + tc.R₂^2 + tc.R₁ * tc.R₂) * tc.h * (Real.pi / 3)

/-- The surface area of a truncated cone -/
noncomputable def surfaceArea (tc : TruncatedCone r) : ℝ :=
  2 * (tc.R₁^2 + tc.R₂^2 + tc.R₁ * tc.R₂) * Real.pi

/-- The ratio of volume to surface area of a truncated cone -/
noncomputable def volumeSurfaceAreaRatio (tc : TruncatedCone r) : ℝ :=
  volume tc / surfaceArea tc

/-- Theorem: The maximum ratio of volume to surface area for a truncated cone
    inscribed around a sphere is r/3 -/
theorem max_volume_surface_area_ratio (r : ℝ) (hr : r > 0) :
  ∃ (tc : TruncatedCone r), ∀ (tc' : TruncatedCone r),
    volumeSurfaceAreaRatio tc' ≤ volumeSurfaceAreaRatio tc ∧
    volumeSurfaceAreaRatio tc = r / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_surface_area_ratio_l1096_109628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_fourth_term_l1096_109601

theorem binomial_expansion_fourth_term 
  (x : ℝ) (x_neq_zero : x ≠ 0) : 
  let expansion := (x^2 - 1/(2*x))^9;
  let fourth_term := Finset.sum (Finset.range 10) (λ k ↦ 
    if k = 3 then (-(1/2))^k * Nat.choose 9 k * x^(18-3*k) else 0)
  ∃ (c : ℝ), fourth_term = c * x^9 ∧ Nat.choose 9 3 = 84 ∧ c = -21/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_fourth_term_l1096_109601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l1096_109652

theorem complex_number_properties (z : ℂ) (ω : ℝ) (h1 : ω = z + z⁻¹) (h2 : -1 < ω) (h3 : ω < 2) :
  (Complex.abs z = 1) ∧
  (-1/2 < z.re ∧ z.re < 1) ∧
  (∃ (y : ℝ), (1 - z) / (1 + z) = Complex.I * y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l1096_109652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_appearance_in_prime_power_l1096_109671

theorem digit_appearance_in_prime_power (p n : ℕ) : 
  Prime p → p > 3 → n > 0 → (Nat.digits 10 (p^n)).length = 20 →
  ¬ (∀ d : Fin 10, (Nat.digits 10 (p^n)).count d.val = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_appearance_in_prime_power_l1096_109671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_relations_l1096_109684

theorem circle_relations (X Y Z : Set (ℝ × ℝ)) (center_X center_Y center_Z : ℝ × ℝ) 
  (radius_X radius_Y radius_Z : ℝ) :
  (X = {p : ℝ × ℝ | (p.1 - center_X.1)^2 + (p.2 - center_X.2)^2 = radius_X^2}) →
  (Y = {p : ℝ × ℝ | (p.1 - center_Y.1)^2 + (p.2 - center_Y.2)^2 = radius_Y^2}) →
  (Z = {p : ℝ × ℝ | (p.1 - center_Z.1)^2 + (p.2 - center_Z.2)^2 = radius_Z^2}) →
  (π * radius_X^2 = π * radius_Y^2) →
  (2 * π * radius_X = 18 * π) →
  (π * radius_Z^2 = 4 * π * radius_X^2) →
  (radius_Y = radius_Z / 2) ∧ (2 * radius_Z = 4 * radius_X) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_relations_l1096_109684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_in_interval_l1096_109625

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) - 1 / x

theorem f_has_zero_in_interval :
  ∃ c ∈ Set.Ioo 1 2, f c = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_in_interval_l1096_109625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_equals_target_l1096_109654

noncomputable section

-- Define the radii of the two smaller circles
def r₁ : ℝ := 3
def r₂ : ℝ := 5

-- Define the radius of the larger circle
noncomputable def R : ℝ := (r₁ + r₂ + Real.sqrt ((r₁ + r₂)^2 + (r₂ - r₁)^2)) / 2

-- Define the shaded area
noncomputable def shaded_area : ℝ := Real.pi * R^2 - Real.pi * r₁^2 - Real.pi * r₂^2

-- Theorem statement
theorem shaded_area_equals_target : shaded_area = 14 * Real.pi + 32 * Real.sqrt 2 * Real.pi := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_equals_target_l1096_109654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_entire_function_l1096_109660

open Complex

/-- A function is entire if it is complex differentiable everywhere. -/
def Entire (f : ℂ → ℂ) : Prop := ∀ z : ℂ, DifferentiableAt ℂ f z

/-- The theorem statement -/
theorem exists_special_entire_function : ∃ F : ℂ → ℂ, 
  Entire F ∧ 
  (∃ z : ℂ, F z ≠ 0) ∧
  (∀ z : ℂ, Complex.abs (F z) ≤ Real.exp (Complex.abs z)) ∧
  (∀ y : ℝ, Complex.abs (F (I * y)) ≤ 1) ∧
  Set.Infinite {x : ℝ | F x = 0} :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_entire_function_l1096_109660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circles_l1096_109693

/-- Circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  distance c.center (l.slope * c.center.1 + l.intercept, c.center.2) = c.radius

theorem tangent_line_to_circles :
  let c1 : Circle := { center := (1, 2), radius := Real.sqrt 2 }
  let c2 : Circle := { center := (3, 4), radius := Real.sqrt 2 }
  let l : Line := { slope := -1, intercept := 5 }
  isTangent l c1 ∧ isTangent l c2 := by
  sorry

#check tangent_line_to_circles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circles_l1096_109693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1096_109673

noncomputable def f (a θ : ℝ) : ℝ := Real.sin θ ^ 3 + 4 / (3 * a * Real.sin θ ^ 2 - a ^ 3)

theorem f_minimum_value (a θ : ℝ) 
  (h1 : 0 < a) 
  (h2 : a < Real.sqrt 3 * Real.sin θ)
  (h3 : π / 6 ≤ θ)
  (h4 : θ ≤ Real.arcsin ((3 : ℝ) ^ (1/3) / 2)) :
  f a θ ≥ 137 / 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1096_109673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_triangle_l1096_109662

/-- Represents a right trapezoid --/
structure RightTrapezoid where
  shortBase : ℝ
  longBase : ℝ
  height : ℝ

/-- Represents a triangle --/
structure Triangle where
  base : ℝ
  height : ℝ

/-- Calculates the area of a right trapezoid --/
noncomputable def areaRightTrapezoid (t : RightTrapezoid) : ℝ :=
  (t.shortBase + t.longBase) * t.height / 2

/-- Calculates the area of a triangle --/
noncomputable def areaTriangle (t : Triangle) : ℝ :=
  t.base * t.height / 2

/-- The main theorem --/
theorem probability_in_triangle (trap : RightTrapezoid) (tri : Triangle) :
  trap.shortBase = 10 →
  trap.longBase = 20 →
  trap.height = 10 →
  tri.base = 8 →
  tri.height = 5 →
  areaTriangle tri / areaRightTrapezoid trap = 2 / 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_triangle_l1096_109662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersection_concyclic_l1096_109613

-- Define the types for points and circles
variable (Point Circle : Type*) 

-- Define the basic geometric operations
variable (center : Circle → Point)
variable (on_circle : Point → Circle → Prop)
variable (intersect : Circle → Circle → Set Point)
variable (angle : Point → Point → Point → ℝ)
variable (line_intersect : Point → Point → Circle → Point)
variable (circumcircle : Point → Point → Point → Circle)

-- Define the problem setup
variable (Γ₁ Γ₂ : Circle)
variable (O₁ O₂ A B C D E F : Point)

-- State the theorem
theorem circles_intersection_concyclic 
  (h1 : center Γ₁ = O₁)
  (h2 : center Γ₂ = O₂)
  (h3 : A ∈ intersect Γ₁ Γ₂)
  (h4 : B ∈ intersect Γ₁ Γ₂)
  (h5 : A ≠ B)
  (h6 : angle O₁ A O₂ > π / 2)
  (h7 : C ∈ intersect (circumcircle O₁ A O₂) Γ₁)
  (h8 : D ∈ intersect (circumcircle O₁ A O₂) Γ₂)
  (h9 : C ≠ A)
  (h10 : D ≠ A)
  (h11 : E = line_intersect C B Γ₂)
  (h12 : F = line_intersect D B Γ₁)
  : ∃ (Γ : Circle), on_circle C Γ ∧ on_circle D Γ ∧ on_circle E Γ ∧ on_circle F Γ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersection_concyclic_l1096_109613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_exists_l1096_109658

noncomputable def triangleArea (z1 z2 z3 : ℂ) : ℝ :=
  (1/2) * abs (z1.re * z2.im + z2.re * z3.im + z3.re * z1.im - 
               z1.im * z2.re - z2.im * z3.re - z3.im * z1.re)

def firstTriangleCondition (n : ℕ+) : Prop :=
  triangleArea (n + Complex.I) ((n + Complex.I)^2) ((n + Complex.I)^4) > 5000

def secondTriangleCondition (n : ℕ+) : Prop :=
  triangleArea (n + Complex.I) ((n + Complex.I)^3) ((n + Complex.I)^5) > 3000

theorem smallest_n_exists : ∃ (n : ℕ+), 
  (∀ (m : ℕ+), m < n → ¬(firstTriangleCondition m ∧ secondTriangleCondition m)) ∧
  (firstTriangleCondition n ∧ secondTriangleCondition n) := by
  sorry

#check smallest_n_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_exists_l1096_109658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_sum_of_self_inverses_l1096_109680

theorem square_sum_of_self_inverses (a b n : ℕ) (hn : n > 1) 
  (ha : a * a ≡ 1 [MOD n]) (hb : b * b ≡ 1 [MOD n]) : 
  a^2 + b^2 ≡ 2 [MOD n] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_sum_of_self_inverses_l1096_109680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_u_l1096_109683

-- Define the function u as noncomputable
noncomputable def u (x y : ℝ) : ℝ := Real.log x / Real.log 10 + Real.log y / Real.log 10

-- State the theorem
theorem max_value_of_u :
  ∀ x y : ℝ, x > 0 → y > 0 → x + y = 8 →
  u x y ≤ 4 * (Real.log 2 / Real.log 10) :=
by
  -- The proof is skipped using sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_u_l1096_109683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_sum_l1096_109665

theorem cubic_root_sum (x : ℝ) :
  27 * x^3 - 4 * x^2 - 4 * x - 1 = 0 →
  ∃ (p q r : ℕ), 
    p > 0 ∧ q > 0 ∧ r > 0 ∧
    x = (p^(1/3 : ℝ) + q^(1/3 : ℝ) + 1) / r ∧
    p + q + r = 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_sum_l1096_109665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_relation_triangle_specific_angle_l1096_109610

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  hA : 0 < A ∧ A < π
  hB : 0 < B ∧ B < π
  hC : 0 < C ∧ C < π
  sum_angles : A + B + C = π
  law_of_sines : a / Real.sin A = b / Real.sin B
  law_of_cosines : a^2 = b^2 + c^2 - 2*b*c*(Real.cos A)

theorem triangle_angle_relation (t : Triangle) 
    (h1 : t.a = 2 * t.b * Real.cos t.B) 
    (h2 : t.b ≠ t.c) : 
  t.A = 2 * t.B := by sorry

theorem triangle_specific_angle (t : Triangle) 
    (h1 : t.a = 2 * t.b * Real.cos t.B) 
    (h2 : t.b ≠ t.c) 
    (h3 : t.a^2 + t.c^2 = t.b^2 + 2*t.a*t.c*Real.sin t.C) : 
  t.A = π / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_relation_triangle_specific_angle_l1096_109610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l1096_109687

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/169 + y^2/144 = 1

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2/9 - y^2/16 = 1

-- Define the circle (renamed to avoid conflict)
def target_circle (x y : ℝ) : Prop := x^2 + y^2 - 10*x + 9 = 0

-- Define the right focus of the ellipse
def right_focus (x y : ℝ) : Prop := ellipse x y ∧ x > 0 ∧ y = 0

-- Define the asymptotes of the hyperbola
def asymptote (x y : ℝ) : Prop := hyperbola x y ∧ (y = 4/3*x ∨ y = -4/3*x)

-- Theorem statement
theorem circle_properties :
  ∃ (cx cy : ℝ), 
    (right_focus cx cy) ∧ 
    (∀ (x y : ℝ), asymptote x y → 
      ∃ (px py : ℝ), target_circle px py ∧ 
        ((px - cx)^2 + (py - cy)^2 = (x - cx)^2 + (y - cy)^2)) ∧
    (∀ (x y : ℝ), target_circle x y ↔ (x - cx)^2 + (y - cy)^2 = 4^2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l1096_109687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_diagonals_in_rectangle_l1096_109678

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square grid within a rectangle -/
structure SquareGrid where
  rectangle : Rectangle
  squareSize : ℕ

/-- Represents a diagonal in a square -/
structure Diagonal where
  startX : ℕ
  startY : ℕ
  endX : ℕ
  endY : ℕ

/-- Checks if two diagonals share endpoints -/
def sharesEndpoints (d1 d2 : Diagonal) : Prop :=
  (d1.startX = d2.startX ∧ d1.startY = d2.startY) ∨
  (d1.startX = d2.endX ∧ d1.startY = d2.endY) ∨
  (d1.endX = d2.startX ∧ d1.endY = d2.startY) ∨
  (d1.endX = d2.endX ∧ d1.endY = d2.endY)

/-- The theorem to be proved -/
theorem max_diagonals_in_rectangle (grid : SquareGrid) 
  (h1 : grid.rectangle.width = 100)
  (h2 : grid.rectangle.height = 3)
  (h3 : grid.squareSize = 1)
  (h4 : grid.rectangle.width * grid.rectangle.height = 300) :
  ∃ (diagonals : List Diagonal), 
    (∀ d1 d2, d1 ∈ diagonals → d2 ∈ diagonals → d1 ≠ d2 → ¬sharesEndpoints d1 d2) ∧ 
    diagonals.length = 200 ∧
    (∀ diagonals' : List Diagonal, 
      (∀ d1 d2, d1 ∈ diagonals' → d2 ∈ diagonals' → d1 ≠ d2 → ¬sharesEndpoints d1 d2) →
      diagonals'.length ≤ 200) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_diagonals_in_rectangle_l1096_109678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_points_l1096_109656

/-- The base function f₁ -/
def f₁ (x : ℝ) : ℝ := 4 * (x - x^2)

/-- The recursive definition of fₙ -/
def f : ℕ → (ℝ → ℝ)
| 0 => f₁
| 1 => f₁
| n+2 => fun x => f (n+1) (f₁ x)

/-- The number of maximum points for fₙ in [0,1] -/
def a (n : ℕ) : ℕ := 2^(n-1)

/-- The number of minimum points for fₙ in [0,1] -/
def b (n : ℕ) : ℕ := 2^(n-1) + 1

/-- The main theorem stating the number of maximum and minimum points for fₙ -/
theorem max_min_points (n : ℕ) (h : n ≥ 1) :
  (∃ S : Set ℝ, S.Finite ∧ S.ncard = a n ∧ ∀ x ∈ S, x ∈ Set.Icc 0 1 ∧ IsLocalMax (f n) x) ∧
  (∃ T : Set ℝ, T.Finite ∧ T.ncard = b n ∧ ∀ x ∈ T, x ∈ Set.Icc 0 1 ∧ IsLocalMin (f n) x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_points_l1096_109656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1096_109666

open Set
open Function
open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 - 1/x

-- State the theorem
theorem range_of_m (a b : ℝ) (h_ab : a < b) :
  (∃ m : ℝ, Ioo a b = {x | x > 0 ∧ f x ∈ Ioo (m * a) (m * b)}) →
  (∃ m : ℝ, 0 < m ∧ m < 1/4) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1096_109666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1096_109627

theorem equation_solution :
  ∃! x : ℚ, (16 : ℝ) ^ ((3 : ℝ) * x - 5) = (1 / 4 : ℝ) ^ ((2 : ℝ) * x + 6) :=
by
  use (-1/2 : ℚ)
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1096_109627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1096_109637

noncomputable def f (x : ℝ) : ℝ := 1 / (2^x + 1) - 1/2

theorem solution_set (a : ℝ) : 
  f (a + 1) + f (a^2 - 1) > 0 ↔ -1 < a ∧ a < 0 := by
  sorry

#check solution_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1096_109637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_proof_l1096_109639

theorem triangle_cosine_proof (A B C : ℝ) (a b c : ℝ) :
  (A + B + C = Real.pi) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (b * Real.cos C + c * Real.cos B = Real.sqrt 3 * a * Real.cos B) →
  Real.cos B = Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_proof_l1096_109639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_d_equals_one_l1096_109686

-- Define the line
def line (x : ℝ) (d : ℝ) : ℝ := 3 * x + d

-- Define the parabola (marked as noncomputable due to sqrt)
noncomputable def parabola (x : ℝ) : ℝ := Real.sqrt (12 * x)

-- Define the condition for the line to be tangent to the parabola
def is_tangent (d : ℝ) : Prop :=
  ∃ x : ℝ, line x d = parabola x ∧
    ∀ y : ℝ, y ≠ x → line y d ≠ parabola y

-- Theorem statement
theorem tangent_line_d_equals_one :
  ∀ d : ℝ, is_tangent d ↔ d = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_d_equals_one_l1096_109686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_l1096_109650

open Real

/-- Given that f is an even function and φ is in the open interval (0, π/2),
    prove that g(x) = cos(2x - φ) is equivalent to f(x + π/3) --/
theorem function_equivalence (φ : ℝ) 
  (hφ : φ ∈ Set.Ioo 0 (π/2))
  (f : ℝ → ℝ)
  (hf : ∀ x, f x = 1 + 2 * cos x * cos (x + 3*φ))
  (hf_even : ∀ x, f (-x) = f x)
  (g : ℝ → ℝ)
  (hg : ∀ x, g x = cos (2*x - φ)) :
  ∀ x, g x = f (x + π/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_l1096_109650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1096_109616

-- Define the domain M
def M : Set ℝ := {x : ℝ | x < 1 ∨ x > 3}

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (4 : ℝ)^x - 2^(x + 1)

-- Theorem statement
theorem f_range :
  {y : ℝ | ∃ x ∈ M, f x = y} = Set.Icc (-1) 0 ∪ Set.Ioi 48 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1096_109616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_subset_complement_implies_m_range_l1096_109641

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (3 + 2*x - x^2)

-- Define the domain A of f
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- Define the set B
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 9 ≤ 0}

-- Theorem 1
theorem intersection_implies_m_value (m : ℝ) :
  A ∩ B m = Set.Icc 2 3 → m = 5 := by sorry

-- Theorem 2
theorem subset_complement_implies_m_range (m : ℝ) :
  A ⊆ (Set.univ \ B m) → m < -4 ∨ m > 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_subset_complement_implies_m_range_l1096_109641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_subset_exists_l1096_109631

theorem perfect_square_subset_exists (nums : Finset ℕ) (primes : Finset ℕ) : 
  nums.card = 2021 →
  (∀ n ∈ nums, ∀ m ∈ nums, n ≠ m → n ≠ m) →
  primes.card = 2020 →
  (∀ p ∈ primes, Nat.Prime p) →
  (∀ p ∈ primes, p ∣ (nums.prod id)) →
  (∀ q : ℕ, Nat.Prime q → q ∣ (nums.prod id) → q ∈ primes) →
  ∃ subset : Finset ℕ, subset.Nonempty ∧ subset ⊆ nums ∧ ∃ k : ℕ, (subset.prod id) = k^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_subset_exists_l1096_109631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_root_of_equation_l1096_109612

theorem smaller_root_of_equation :
  ∃ (x : ℝ), (x - 3/4) * (x - 3/4) + (x - 3/4) * (x - 1/4) = 0 ∧
  x = 1/2 ∧
  ∀ y, (y - 3/4) * (y - 3/4) + (y - 3/4) * (y - 1/4) = 0 → y ≥ 1/2 := by
  sorry

#check smaller_root_of_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_root_of_equation_l1096_109612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_volumes_l1096_109622

/-- The volume of a sphere given its radius -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

/-- The volume of a cylinder given its base radius and height -/
noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The volume of a cone given its base radius and height -/
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The main theorem stating the volume of the space inside the sphere and outside both the cylinder and cone -/
theorem inscribed_volumes (sphere_radius cylinder_base_radius cylinder_height : ℝ) 
  (h1 : sphere_radius = 6)
  (h2 : cylinder_base_radius = 4)
  (h3 : cylinder_height = 10) :
  sphere_volume sphere_radius - 
  cylinder_volume cylinder_base_radius cylinder_height - 
  cone_volume cylinder_base_radius cylinder_height = 
  (224/3) * Real.pi := by
  sorry

#check inscribed_volumes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_volumes_l1096_109622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_november_rainfall_l1096_109663

/-- Represents the daily rainfall pattern for the first 15 days -/
def first_15_days_pattern : List ℚ := [2, 4, 6, 8, 10, 2, 4, 6, 8, 10, 2, 4, 6, 8, 10]

/-- Calculates the total rainfall for the first 15 days -/
def total_first_15_days : ℚ := first_15_days_pattern.sum

/-- Calculates the average daily rainfall for the first 15 days -/
noncomputable def avg_first_15_days : ℚ := total_first_15_days / 15

/-- Calculates the new average daily rainfall for the next 10 days -/
noncomputable def new_avg_daily_rainfall : ℚ := 2 * avg_first_15_days

/-- Generates the rainfall pattern for the last 10 days -/
noncomputable def last_10_days_pattern : List ℚ := 
  List.map (fun i => if i % 2 = 1 then new_avg_daily_rainfall - 2 else new_avg_daily_rainfall + 2) (List.range 10)

/-- Calculates the total rainfall for the last 10 days -/
noncomputable def total_last_10_days : ℚ := last_10_days_pattern.sum

/-- Theorem: The total rainfall for November is 210 inches -/
theorem total_november_rainfall : total_first_15_days + total_last_10_days = 210 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_november_rainfall_l1096_109663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_second_quadrant_l1096_109645

noncomputable def complex_number : ℂ := (5 * Complex.I) / (2 - Complex.I)

theorem complex_number_in_second_quadrant :
  complex_number.re < 0 ∧ complex_number.im > 0 := by
  -- Simplify the complex number
  have h : complex_number = -1 + 2 * Complex.I := by
    -- Proof of simplification goes here
    sorry
  
  -- Show that the real part is negative
  have h_re : complex_number.re = -1 := by
    rw [h]
    simp
  
  -- Show that the imaginary part is positive
  have h_im : complex_number.im = 2 := by
    rw [h]
    simp
  
  -- Conclude that the complex number is in the second quadrant
  exact ⟨by linarith, by linarith⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_second_quadrant_l1096_109645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_four_nine_equals_five_l1096_109694

theorem repeating_decimal_four_nine_equals_five :
  ∀ (x : ℚ), (∃ (n : ℕ), x = 4 + 9 * (1 / 10^n) / (1 - 1 / 10)) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_four_nine_equals_five_l1096_109694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l1096_109667

noncomputable def diamond (a b : ℝ) : ℝ := (a^2 + b^2) / (a - b)

theorem diamond_calculation : diamond (diamond 2 3) 4 = -185/17 := by
  -- Unfold the definition of diamond
  unfold diamond
  -- Simplify the expression
  simp [pow_two]
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l1096_109667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1096_109629

/-- Given a hyperbola E and a line y = 2a that intersects E, forming a right triangle MON with O as the origin,
    prove that the eccentricity of E is √(21)/3 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let E := {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}
  let line := {p : ℝ × ℝ | p.2 = 2 * a}
  let O := (0, 0)
  ∃ (M N : ℝ × ℝ), M ∈ E ∧ N ∈ E ∧ M ∈ line ∧ N ∈ line ∧
    (O.1 - M.1)^2 + (O.2 - M.2)^2 + (O.1 - N.1)^2 + (O.2 - N.2)^2 = (M.1 - N.1)^2 + (M.2 - N.2)^2 →
    Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 7 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1096_109629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_x_floor_one_over_x_l1096_109647

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define the limit function
noncomputable def f (x : ℝ) : ℝ :=
  x * (floor (1 / x))

-- State the theorem
theorem limit_x_floor_one_over_x :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ → |f x - 1| < ε :=
by
  sorry

#check limit_x_floor_one_over_x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_x_floor_one_over_x_l1096_109647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_complex_square_root_l1096_109633

theorem modulus_of_complex_square_root (w : ℂ) (h : w^2 = -48 + 14*I) : Complex.abs w = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_complex_square_root_l1096_109633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_difference_is_104_l1096_109670

/-- Represents the outcome of rolling a die --/
inductive DieOutcome
  | Prime
  | Composite
  | Reroll

/-- Represents Alice's drink choice --/
inductive Drink
  | Tea
  | Coffee

/-- The number of days in a non-leap year --/
def daysInYear : Nat := 365

/-- The probability of rolling a prime number on the effective 7-sided die --/
noncomputable def probPrime : ℝ := 4/7

/-- The probability of rolling a composite number on the effective 7-sided die --/
noncomputable def probComposite : ℝ := 2/7

/-- Maps the outcome of a die roll to Alice's drink choice --/
def outcomeTodrink (outcome : DieOutcome) : Option Drink :=
  match outcome with
  | DieOutcome.Prime => some Drink.Tea
  | DieOutcome.Composite => some Drink.Coffee
  | DieOutcome.Reroll => none

/-- The expected difference between days drinking tea and coffee in a year --/
noncomputable def expectedDifference : ℝ := daysInYear * (probPrime - probComposite)

theorem expected_difference_is_104 :
  expectedDifference = 104 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_difference_is_104_l1096_109670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1096_109690

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to angle A
  b : ℝ  -- Side opposite to angle B
  c : ℝ  -- Side opposite to angle C

-- Define the properties of the triangle
def triangle_properties (t : Triangle) : Prop :=
  t.A + t.B + t.C = Real.pi ∧  -- Sum of angles is π radians (180°)
  Real.cos t.B = 1/3 ∧
  t.b = 4 ∧
  1/2 * t.a * t.c * Real.sin t.B = 2 * Real.sqrt 2  -- Area formula

-- Theorem statement
theorem triangle_theorem (t : Triangle) 
  (h : triangle_properties t) : 
  (Real.cos (t.B/2))^2 + (Real.tan ((t.A + t.C)/2))^2 = 8/3 ∧
  (t.c = Real.sqrt 2 ∨ t.c = 3 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1096_109690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_voltage_properties_l1096_109689

noncomputable def voltage (t : ℝ) : ℝ := 120 * Real.sqrt 2 * Real.sin (100 * Real.pi * t - Real.pi / 6)

def time_domain (t : ℝ) : Prop := t ≥ 0

def neon_lamp_on (v : ℝ) : Prop := v > 84

def sqrt2_approx : ℝ := 1.4

theorem voltage_properties :
  ∃ (T f A : ℝ),
    (∀ t, time_domain t → voltage (t + T) = voltage t) ∧  -- Period
    (f = 1 / T) ∧  -- Frequency
    (∀ t, time_domain t → voltage t ≤ A) ∧  -- Amplitude
    (T = 1 / 50) ∧
    (f = 50) ∧
    (A = 120 * Real.sqrt 2) ∧
    (∃ t_on, t_on = 1 / 150 ∧
      ∀ t, time_domain t →
        (0 ≤ t ∧ t ≤ T / 2) →
        (neon_lamp_on (voltage t) ↔ (t_on / 2 ≤ t ∧ t ≤ T / 2 - t_on / 2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_voltage_properties_l1096_109689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_in_cube_l1096_109672

/-- A sphere inside a cube, touching all six faces -/
structure SphereCube where
  radius : ℝ
  touches_all_faces : True

/-- The volume of a sphere -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

/-- The edge length of a cube containing a sphere touching all faces -/
def cube_edge_length (sc : SphereCube) : ℝ := 2 * sc.radius

theorem sphere_in_cube (sc : SphereCube) (h : sc.radius = 5) :
  sphere_volume sc.radius = (500/3) * Real.pi ∧ cube_edge_length sc = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_in_cube_l1096_109672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fumigant_disinfection_l1096_109651

-- Define the concentration function
noncomputable def concentration (x : ℝ) : ℝ :=
  if x ≤ 8 then (3/4) * x else 48 / x

-- Define the conditions
theorem fumigant_disinfection 
  (h1 : ∀ x, x ≤ 8 → concentration x = (3/4) * x)
  (h2 : ∀ x, x > 8 → concentration x = 48 / x)
  (h3 : concentration 8 = 6)
  (h4 : ∀ x, x > 0 → concentration x > 0) :
  -- 1. Concentration function is correct
  (∀ x, x ≥ 0 → concentration x = if x ≤ 8 then (3/4) * x else 48 / x) ∧
  -- 2. Students can enter after 30 minutes
  (∀ x, x ≥ 30 → concentration x < 1.6) ∧
  -- 3. Disinfection is effective
  (∃ t, t ≥ 10 ∧ ∀ x, 8 ≤ x ∧ x ≤ 8 + t → concentration x ≥ 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fumigant_disinfection_l1096_109651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_repayment_theorem_l1096_109659

/-- Represents the loan and profit scenario for a company --/
structure LoanProfitScenario where
  initialLoan : ℝ
  interestRate : ℝ
  annualProfit : ℝ
  profitStartYear : ℕ

/-- Calculates the total borrowed amount at the beginning of the nth year --/
noncomputable def totalBorrowed (scenario : LoanProfitScenario) (n : ℕ) : ℝ :=
  scenario.initialLoan * (1 + scenario.interestRate) ^ n

/-- Calculates the total profit at the beginning of the nth year --/
noncomputable def totalProfit (scenario : LoanProfitScenario) (n : ℕ) : ℝ :=
  if n < scenario.profitStartYear then 0
  else scenario.annualProfit * ((1 + scenario.interestRate) ^ (n - scenario.profitStartYear) - 1) / scenario.interestRate

/-- The main theorem to be proved --/
theorem loan_repayment_theorem (scenario : LoanProfitScenario) :
  scenario.initialLoan = 10000 ∧
  scenario.interestRate = 0.1 ∧
  scenario.annualProfit = 3000 ∧
  scenario.profitStartYear = 2 →
  (∀ n : ℕ, n ≥ scenario.profitStartYear → totalProfit scenario n = 3000 * ((1 + 0.1) ^ (n - 1) - 1)) ∧
  (totalProfit scenario 6 ≥ totalBorrowed scenario 6 ∧
   totalProfit scenario 5 < totalBorrowed scenario 5) := by
  sorry

#check loan_repayment_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_repayment_theorem_l1096_109659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_for_z_in_second_quadrant_l1096_109618

-- Define the complex number z
def z (m : ℝ) : ℂ := Complex.mk (m - 1) (m + 2)

-- Define what it means for a complex number to be in the second quadrant
def in_second_quadrant (w : ℂ) : Prop := w.re < 0 ∧ w.im > 0

-- State the theorem
theorem m_range_for_z_in_second_quadrant :
  ∀ m : ℝ, in_second_quadrant (z m) ↔ -2 < m ∧ m < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_for_z_in_second_quadrant_l1096_109618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_index_correspondence_l1096_109642

-- Define the set E
def E : Finset ℕ := Finset.range 10

-- Define the function that maps a subset to its index
def subsetToIndex (s : Finset ℕ) : ℕ :=
  s.sum (fun i => 2^(i-1))

-- Define the function that maps an index to its corresponding subset
def indexToSubset (k : ℕ) : Finset ℕ :=
  Finset.filter (fun i => k.testBit (i-1)) E

theorem subset_index_correspondence :
  (subsetToIndex {1, 3} = 5) ∧
  (indexToSubset 211 = {1, 2, 5, 7, 8}) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_index_correspondence_l1096_109642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1096_109632

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 4 * x / (3 * x^2 + 3)
noncomputable def g (a x : ℝ) : ℝ := (1/3) * a * x^3 - a^2 * x

-- Define the theorem
theorem range_of_a :
  ∀ (a : ℝ), a ≠ 0 →
  (∀ (x₁ : ℝ), 0 ≤ x₁ ∧ x₁ ≤ 2 →
    ∃ (x₂ : ℝ), 0 ≤ x₂ ∧ x₂ ≤ 2 ∧ f x₁ = g a x₂) →
  1/3 ≤ a ∧ a ≤ 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1096_109632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tabby_average_speed_l1096_109605

/-- Calculates the average speed for two events with equal time spent on each -/
noncomputable def averageSpeed (speed1 : ℝ) (speed2 : ℝ) : ℝ :=
  (speed1 + speed2) / 2

theorem tabby_average_speed :
  let swimmingSpeed : ℝ := 1
  let runningSpeed : ℝ := 8
  averageSpeed swimmingSpeed runningSpeed = (4.5 : ℝ) := by
  unfold averageSpeed
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tabby_average_speed_l1096_109605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_half_cistern_l1096_109606

/-- The rate at which the cistern is filled, measured in fraction per minute -/
def fill_rate (t : ℝ) : ℝ := sorry

/-- The amount of the cistern filled after t minutes -/
def fill_amount (t : ℝ) : ℝ := sorry

/-- Given a fill pipe that can fill 1/2 of a cistern in 15 minutes,
    prove that the time to fill 1/2 of the cistern is 15 minutes. -/
theorem fill_time_half_cistern :
  (∀ t : ℝ, t > 0 → fill_rate t = (1/2) / 15) →
  (∃ t : ℝ, t > 0 ∧ fill_amount t = 1/2) →
  ∃ t : ℝ, t = 15 ∧ fill_amount t = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_half_cistern_l1096_109606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1096_109668

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : even_function f)
  (h2 : increasing_on f (Set.Iic 0))
  (h3 : f a ≤ f 2) :
  a ≤ -2 ∨ a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1096_109668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reading_time_difference_l1096_109608

/-- Proves that Molly takes 135 minutes longer than Xanthia to read a 225-page book -/
theorem reading_time_difference 
  (xanthia_speed : ℝ) 
  (molly_speed : ℝ) 
  (book_pages : ℝ) 
  (h1 : xanthia_speed = 100) 
  (h2 : molly_speed = 50) 
  (h3 : book_pages = 225) : 
  (book_pages / molly_speed - book_pages / xanthia_speed) * 60 = 135 := by
  sorry

#check reading_time_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reading_time_difference_l1096_109608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_checkerboard_swap_l1096_109669

/-- Represents the color of a cell -/
inductive Color
  | White
  | Black
  | Green

/-- Represents a grid cell -/
structure Cell :=
  (row : Nat)
  (col : Nat)
  (color : Color)

/-- Represents the n × n grid -/
def Grid (n : Nat) := List Cell

/-- The recoloring rule for a 2×2 square -/
def recolor (cells : List Cell) : List Cell :=
  cells.map fun c =>
    match c.color with
    | Color.White => { c with color := Color.Black }
    | Color.Black => { c with color := Color.Green }
    | Color.Green => { c with color := Color.White }

/-- Checks if the grid has a checkerboard pattern -/
def isCheckerboard (n : Nat) (grid : Grid n) : Prop := sorry

/-- Checks if at least one corner cell is black -/
def hasBlackCorner (n : Nat) (grid : Grid n) : Prop := sorry

/-- Checks if the colors of black and white cells are swapped -/
def isSwapped (n : Nat) (original : Grid n) (current : Grid n) : Prop := sorry

/-- The main theorem -/
theorem checkerboard_swap (n : Nat) :
  (∃ (grid : Grid n), isCheckerboard n grid ∧ hasBlackCorner n grid ∧
    ∃ (moves : Nat), ∃ (final : Grid n), isSwapped n grid final) ↔ 
  ∃ (k : Nat), n = 3 * k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_checkerboard_swap_l1096_109669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rook_polynomial_specific_config_l1096_109634

/-- Represents a chessboard configuration --/
structure ChessboardConfig where
  representation : String

/-- Calculates the rook polynomial for a given chessboard configuration --/
noncomputable def rookPolynomial (config : ChessboardConfig) : Polynomial ℚ :=
  sorry -- Implementation details omitted for brevity

/-- The specific chessboard configuration for this problem --/
def specificConfig : ChessboardConfig :=
  { representation := "× × × × × ×" }

/-- Theorem stating that the rook polynomial for the specific configuration
    is equal to the given polynomial --/
theorem rook_polynomial_specific_config :
  rookPolynomial specificConfig =
    1 + 6 * X + 11 * X^2 + 6 * X^3 + X^4 := by
  sorry -- Proof omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rook_polynomial_specific_config_l1096_109634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_fourth_minus_alpha_l1096_109695

theorem tan_pi_fourth_minus_alpha (α : ℝ) (h1 : α ∈ Set.Ioo π (3 * π / 2)) (h2 : Real.cos α = -4/5) :
  Real.tan (π/4 - α) = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_fourth_minus_alpha_l1096_109695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaolin_final_score_l1096_109677

/-- Calculates the final score in physical education given the scores and weights for sit-ups and 800 meters --/
noncomputable def final_score (sit_ups_score : ℝ) (meters_800_score : ℝ) (sit_ups_weight : ℝ) (meters_800_weight : ℝ) : ℝ :=
  (sit_ups_score * sit_ups_weight + meters_800_score * meters_800_weight) / (sit_ups_weight + meters_800_weight)

/-- Xiaolin's final score in physical education is 86 points --/
theorem xiaolin_final_score :
  let sit_ups_score : ℝ := 80
  let meters_800_score : ℝ := 90
  let sit_ups_weight : ℝ := 4
  let meters_800_weight : ℝ := 6
  final_score sit_ups_score meters_800_score sit_ups_weight meters_800_weight = 86 := by
  unfold final_score
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaolin_final_score_l1096_109677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_equal_42_l1096_109657

/-- The number of people available for selection -/
def num_people : ℕ := 5

/-- The number of positions to be filled -/
def num_positions : ℕ := 4

/-- The number of positions where person A can be placed -/
def positions_for_A : ℕ := 3

/-- Calculates the number of arrangements when A is selected -/
def arrangements_with_A : ℕ :=
  positions_for_A * Nat.factorial (num_positions - 1)

/-- Calculates the number of arrangements when A is not selected -/
def arrangements_without_A : ℕ :=
  Nat.factorial (num_people - 1)

/-- The total number of possible arrangements -/
def total_arrangements : ℕ :=
  arrangements_with_A + arrangements_without_A

/-- Theorem stating that the total number of arrangements is 42 -/
theorem arrangements_equal_42 : total_arrangements = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_equal_42_l1096_109657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1096_109688

noncomputable section

-- Define the function f
noncomputable def f (x φ : Real) : Real :=
  (1/2) * Real.sin (2*x) * Real.sin φ + Real.cos x^2 * Real.cos φ - (1/2) * Real.sin (Real.pi/2 + φ)

-- Define the function g
noncomputable def g (x : Real) : Real :=
  (1/2) * Real.sin (x + Real.pi/6)

-- State the theorem
theorem function_properties :
  ∃ (φ : Real), 0 < φ ∧ φ < Real.pi ∧
  f (Real.pi/6) φ = 1/2 ∧
  φ = Real.pi/3 ∧
  (∀ x, g x = (1/2) * Real.sin (x + Real.pi/6)) ∧
  (∃ x₀ ∈ Set.Icc 0 (Real.pi/4), ∀ x ∈ Set.Icc 0 (Real.pi/4), g x ≤ g x₀ ∧ g x₀ = 1/2) ∧
  (¬∃ x₀ ∈ Set.Icc 0 (Real.pi/4), ∀ x ∈ Set.Icc 0 (Real.pi/4), g x ≥ g x₀) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1096_109688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l1096_109603

/-- Circle C₁ with equation x² + y² = b² -/
def Circle (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = b^2}

/-- Ellipse C₂ with equation x²/a² + y²/b² = 1 -/
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}

/-- Eccentricity of an ellipse -/
noncomputable def Eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- Tangent line to a circle -/
def TangentLine (b : ℝ) (p : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- Two lines are perpendicular -/
def Perpendicular (l1 l2 : Set (ℝ × ℝ)) : Prop := sorry

theorem eccentricity_range (a b : ℝ) (h1 : 0 < b) (h2 : b < a) :
  ∃ p ∈ Ellipse a b,
    ∃ t1 t2 : Set (ℝ × ℝ),
      t1 = TangentLine b p ∧
      t2 = TangentLine b p ∧
      Perpendicular t1 t2 →
      Real.sqrt 2 / 2 ≤ Eccentricity a b ∧ Eccentricity a b < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l1096_109603
