import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l666_66652

noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 2) + Real.exp (-x * Real.log 2)
noncomputable def g (x : ℝ) : ℝ := Real.exp (x * Real.log 2) - Real.exp (-x * Real.log 2)

theorem min_m_value (m : ℝ) :
  (∀ x : ℝ, f x = f (-x)) →  -- f is even
  (∀ x : ℝ, g (-x) = -g x) →  -- g is odd
  (∀ x : ℝ, f x - g x = Real.exp ((1 - x) * Real.log 2)) →  -- given equation
  (∃ x : ℝ, m * f x = (g x)^2 + 2*m + 9) →  -- equation has a solution
  m ≥ 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l666_66652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_abs_inequality_l666_66698

theorem contrapositive_abs_inequality :
  (∀ a b : ℝ, a > b → abs a > abs b) ↔ (∀ a b : ℝ, abs a ≤ abs b → a ≤ b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_abs_inequality_l666_66698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_theorem_l666_66608

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on a circle
structure PointOnCircle (γ : Circle) where
  point : ℝ × ℝ
  on_circle : (point.1 - γ.center.1)^2 + (point.2 - γ.center.2)^2 = γ.radius^2

-- Define a convex hexagon
structure ConvexHexagon (γ : Circle) where
  A : PointOnCircle γ
  B : PointOnCircle γ
  C : PointOnCircle γ
  D : PointOnCircle γ
  E : PointOnCircle γ
  F : PointOnCircle γ
  is_convex : Bool  -- Assume there's a way to check convexity

-- Define a line through two points
def line (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r : ℝ × ℝ | ∃ t : ℝ, r = (1 - t) • p + t • q}

-- Define concurrent lines
def concurrent (l1 l2 l3 : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ l1 ∧ p ∈ l2 ∧ p ∈ l3

-- Define distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem hexagon_theorem (γ : Circle) (hex : ConvexHexagon γ) 
  (h_concurrent : concurrent 
    (line hex.A.point hex.D.point) 
    (line hex.B.point hex.E.point) 
    (line hex.C.point hex.F.point)) :
  distance hex.A.point hex.B.point * 
  distance hex.C.point hex.D.point * 
  distance hex.E.point hex.F.point = 
  distance hex.B.point hex.C.point * 
  distance hex.D.point hex.E.point * 
  distance hex.F.point hex.A.point := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_theorem_l666_66608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_length_l666_66646

/-- The length of a rectangular box given specific conditions -/
theorem box_length
  (max_cubes : ℝ)
  (cube_volume : ℝ)
  (box_width : ℝ)
  (box_height : ℝ)
  (box_length : ℝ) :
  max_cubes = 24 →
  cube_volume = 27 →
  box_width = 9 →
  box_height = 12 →
  box_length * box_width * box_height = max_cubes * cube_volume →
  box_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_length_l666_66646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_eq_one_l666_66634

theorem sin_plus_cos_eq_one (x : ℝ) :
  0 ≤ x ∧ x < 2 * Real.pi → (Real.sin x + Real.cos x = 1 ↔ x = 0 ∨ x = Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_eq_one_l666_66634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_inequality_l666_66676

/-- A function satisfying the given condition -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (x₂ * f x₁ - x₁ * f x₂) / (x₁ - x₂) > 0

theorem special_function_inequality (f : ℝ → ℝ) (h : SpecialFunction f) :
  f (Real.log 5 / Real.log 2) / (Real.log 5 / Real.log 2) <
  f (3 ^ (1/5 : ℝ)) / (3 ^ (1/5 : ℝ)) ∧
  f (3 ^ (1/5 : ℝ)) / (3 ^ (1/5 : ℝ)) <
  f ((3/10 : ℝ) ^ 2) / ((3/10 : ℝ) ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_inequality_l666_66676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l666_66601

noncomputable section

/-- The main function f(x) -/
def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + (1/2) * x^2 - 2*x + a

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := x^2 + x - 2

/-- The function g(x) -/
def g (a m : ℝ) (x : ℝ) : ℝ := f a x + (1/2) * (m-1) * x^2 - (2*m^2 - 2) * x - 1

/-- The derivative of g(x) -/
def g' (m : ℝ) (x : ℝ) : ℝ := (x + 2*m) * (x - m)

theorem problem_solution :
  ∃ (a b : ℝ),
    (∀ x, f a x = 0 → x = 0) ∧  -- f intersects y-axis
    (f' 0 = b) ∧  -- tangent line at y-intercept
    (f a 0 = 1) ∧  -- y-intercept is (0, 1)
    (a = 1 ∧ b = 2) ∧  -- part (I) solution
    (∃ m : ℝ, (m = -1 ∨ m = 3980/7) ∧
      (∀ x, g a m x ≥ -10/3) ∧
      (∃ x, g a m x = -10/3)) ∧  -- part (II) solution
    (∀ t : ℝ, t ≤ 2 →
      ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-1 : ℝ) 0 → x₂ ∈ Set.Icc (-1 : ℝ) 0 → x₁ ≠ x₂ →
        |f a x₁ - f a x₂| ≥ t * |x₁ - x₂|) -- part (III) solution
  := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l666_66601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l666_66691

/-- The average speed of a car given its speeds in two consecutive hours. -/
noncomputable def average_speed (speed1 : ℝ) (speed2 : ℝ) : ℝ :=
  (speed1 + speed2) / 2

/-- Theorem: The average speed of a car traveling 90 km in the first hour
    and 75 km in the second hour is 82.5 km/h. -/
theorem car_average_speed :
  average_speed 90 75 = 82.5 := by
  unfold average_speed
  norm_num
  
-- We can't use #eval with noncomputable functions, so let's use #check instead
#check average_speed 90 75

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l666_66691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l666_66692

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + Real.sin x

-- State the theorem
theorem min_value_of_f :
  ∃ (min_val : ℝ), min_val = (1 - Real.sqrt 2) / 2 ∧
  ∀ (x : ℝ), |x| ≤ π/4 → f x ≥ min_val :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l666_66692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_leg_time_l666_66637

/-- Represents a two-leg relay race -/
structure RelayRace where
  leg1_time : ℚ
  leg2_time : ℚ

/-- The average time for a leg in a relay race -/
def average_time (race : RelayRace) : ℚ :=
  (race.leg1_time + race.leg2_time) / 2

theorem first_leg_time (race : RelayRace) 
  (h1 : race.leg2_time = 7)
  (h2 : average_time race = 22.5) : 
  race.leg1_time = 38 := by
  sorry

#eval average_time ⟨38, 7⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_leg_time_l666_66637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l666_66624

/-- The function g(x) defined as ax^2 + bx + c + d*sin(x) -/
noncomputable def g (a b c d x : ℝ) : ℝ := a * x^2 + b * x + c + d * Real.sin x

/-- Theorem stating the minimum value of g(x) -/
theorem min_value_of_g (a b c d : ℝ) (ha : a > 0) (hd : d > 0) :
  ∃ (m : ℝ), ∀ (x : ℝ), g a b c d x ≥ m ∧ m = -b^2 / (4 * a) + c - d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l666_66624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l666_66677

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := x + 1/x + 1/(x^2 + 1/x^2)

/-- Theorem stating the minimum value of f(x) for x > 0 -/
theorem min_value_of_f :
  (∀ x > 0, f x ≥ 2.5) ∧ (∃ x > 0, f x = 2.5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l666_66677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_perpendicular_lines_l666_66666

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The point of intersection between two lines -/
noncomputable def intersection (l1 l2 : Line) : ℝ × ℝ :=
  let x := (l2.intercept - l1.intercept) / (l1.slope - l2.slope)
  let y := l1.slope * x + l1.intercept
  (x, y)

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

theorem intersection_of_perpendicular_lines :
  let l1 : Line := { slope := 3, intercept := 4 }
  let l2 : Line := { slope := -1/3, intercept := 4 }
  perpendicular l1 l2 ∧
  l2.slope * 3 + l2.intercept = 3 →
  intersection l1 l2 = (0, 4) := by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_perpendicular_lines_l666_66666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_price_proof_l666_66668

noncomputable def original_price (final_price : ℝ) (first_reduction : ℝ) (second_reduction : ℝ) : ℝ :=
  final_price / ((1 - first_reduction) * (1 - second_reduction))

theorem car_price_proof (final_price : ℝ) (first_reduction : ℝ) (second_reduction : ℝ)
    (h1 : final_price = 15000)
    (h2 : first_reduction = 0.3)
    (h3 : second_reduction = 0.4) :
  Int.floor (original_price final_price first_reduction second_reduction) = 35714 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_price_proof_l666_66668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volleyball_teams_max_astonishing_l666_66684

theorem volleyball_teams_max_astonishing (n : ℕ) (hn : n = 30) : 
  ∃ (t : ℕ), t ≤ 9 ∧ 
  (∀ (k : ℕ), k > t → 
    (t * (n - t) + (3 * t - 3) * t / 2 + (2 * t - 1) * (2 * t - 2) + 
     (n - t) * (n - t - 1) / 2 > n * (n - 1) / 2)) := by
  sorry

#check volleyball_teams_max_astonishing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volleyball_teams_max_astonishing_l666_66684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_competition_count_l666_66658

/-- The number of ways students can sign up for sports competitions -/
def signUpWays (numStudents : ℕ) (numCompetitions : ℕ) : ℕ :=
  numCompetitions ^ numStudents

/-- The number of possibilities for winning the championship -/
def championshipPossibilities (numStudents : ℕ) (numCompetitions : ℕ) : ℕ :=
  numStudents ^ numCompetitions

/-- Theorem stating the correct number of ways to sign up and championship possibilities -/
theorem sports_competition_count :
  signUpWays 5 4 = 4^5 ∧ championshipPossibilities 5 4 = 5^4 := by
  sorry

-- Commented out #eval statements
/-
#eval signUpWays 5 4  -- Should output 1024
#eval championshipPossibilities 5 4  -- Should output 625
-/

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_competition_count_l666_66658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_selection_count_l666_66685

/-- The number of members in the committee -/
def committee_size : ℕ := 5

/-- The number of roles to be filled -/
def roles_to_fill : ℕ := 3

/-- The number of members who cannot be selected for a specific role -/
def restricted_members : ℕ := 2

/-- The number of ways to select members for the committee roles -/
def selection_ways : ℕ := 36

/-- Theorem stating the number of ways to select committee members -/
theorem committee_selection_count :
  (committee_size - restricted_members).choose 1 * (committee_size - 1).choose 2 * 2 = selection_ways :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_selection_count_l666_66685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l666_66614

/-- Given vectors in ℝ² -/
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-2, 3]
def c (m : ℝ) : Fin 2 → ℝ := ![-2, m]

/-- The dot product of two vectors -/
def dot_product (u v : Fin 2 → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1)

/-- The magnitude of a vector -/
noncomputable def magnitude (v : Fin 2 → ℝ) : ℝ :=
  Real.sqrt ((v 0)^2 + (v 1)^2)

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (u v : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), v = fun i => k * (u i)

theorem vector_problem :
  (∃ m : ℝ, dot_product a (fun i => b i + c m i) = 0 → magnitude (c m) = Real.sqrt 5) ∧
  (∃ k : ℝ, collinear (fun i => k * a i + b i) (fun i => 2 * a i - b i) → k = -2) := by
  sorry

#check vector_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l666_66614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_22_terms_l666_66615

def a : ℕ → ℚ
  | 0 => 5/2
  | n + 1 => 2 / (2 - a n)

def S (n : ℕ) : ℚ :=
  Finset.sum (Finset.range n) (λ i => a i)

theorem sum_22_terms : S 22 = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_22_terms_l666_66615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l666_66636

theorem trig_identity (α β γ : ℝ) :
  (Real.sin α)^3 * (Real.sin (β - γ))^3 + (Real.sin β)^3 * (Real.sin (γ - α))^3 + (Real.sin γ)^3 * (Real.sin (α - β))^3 =
  3 * Real.sin α * Real.sin β * Real.sin γ * Real.sin (α - β) * Real.sin (β - γ) * Real.sin (γ - α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l666_66636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_f_is_four_l666_66643

noncomputable def f (x : ℝ) : ℝ := 2 - Real.log x / Real.log 2

theorem root_of_f_is_four (a : ℝ) (h : f a = 0) : a = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_f_is_four_l666_66643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_point_sum_l666_66695

-- Define the triangle vertices and point P
def D : ℝ × ℝ := (0, 0)
def E : ℝ × ℝ := (8, 0)
def F : ℝ × ℝ := (5, 7)
def P : ℝ × ℝ := (5, 3)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem fermat_point_sum :
  ∃ (p q : ℕ), 
    distance D P + distance E P + distance F P = Real.sqrt 34 + 3 * Real.sqrt 2 + 4 ∧
    p + q = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_point_sum_l666_66695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rect_conversion_l666_66606

noncomputable section

-- Define the polar coordinates
def r : ℝ := 4
def θ : ℝ := 7 * Real.pi / 6

-- Define the conversion functions
def polar_to_rect_x (r θ : ℝ) : ℝ := r * Real.cos θ
def polar_to_rect_y (r θ : ℝ) : ℝ := r * Real.sin θ

-- State the theorem
theorem polar_to_rect_conversion :
  (polar_to_rect_x r θ, polar_to_rect_y r θ) = (-2 * Real.sqrt 3, -2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rect_conversion_l666_66606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_30min_angle_l666_66630

/-- The angle swept by the minute hand of a clock in a given time period -/
noncomputable def angle_swept (total_minutes : ℕ) : ℝ :=
  (total_minutes : ℝ) * 360 / 60

/-- Theorem: The angle swept by the minute hand in 30 minutes is 180° -/
theorem minute_hand_30min_angle : angle_swept 30 = 180 := by
  -- Unfold the definition of angle_swept
  unfold angle_swept
  -- Simplify the arithmetic
  simp [Nat.cast_ofNat]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_30min_angle_l666_66630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_desargues_collinearity_l666_66699

-- Define the points in a 2D plane
variable (A B C A₁ B₁ C₁ : EuclideanSpace ℝ (Fin 2))

-- Define the condition that points lie on one line
def collinear (P Q R : EuclideanSpace ℝ (Fin 2)) : Prop := 
  ∃ (t : ℝ), R - P = t • (Q - P)

-- Define parallel lines
def parallel (P Q R S : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ (t : ℝ), Q - P = t • (S - R)

-- State the theorem
theorem desargues_collinearity
  (A B C A₁ B₁ C₁ : EuclideanSpace ℝ (Fin 2)) :
  collinear A B C ∧ 
  parallel A B₁ B A₁ ∧
  parallel A C₁ C A₁ ∧
  parallel B C₁ C B₁ →
  collinear A₁ B₁ C₁ := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_desargues_collinearity_l666_66699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_relation_l666_66613

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), a = (k * b.1, k * b.2.1, k * b.2.2)

/-- Given two collinear vectors (1, -2, m) and (n, 4, 6), prove that m - 2n = 1 -/
theorem collinear_vectors_relation (m n : ℝ) :
  collinear (1, -2, m) (n, 4, 6) → m - 2*n = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_relation_l666_66613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_contrapositive_true_l666_66650

-- Statement 1
theorem negation_equivalence :
  (¬ ∀ x : ℝ, Real.sin x ≠ 1) ↔ (∃ x : ℝ, Real.sin x = 1) :=
sorry

-- Statement 2
theorem contrapositive_true :
  (∀ x y : ℝ, Real.sin x ≠ Real.sin y → x ≠ y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_contrapositive_true_l666_66650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l666_66681

noncomputable def sequence_a (n : ℕ+) : ℝ := n

noncomputable def S (n : ℕ+) : ℝ := ((n : ℝ) + 1) * sequence_a n / 2

noncomputable def sequence_b (n : ℕ+) : ℝ := Real.log (sequence_a n)

theorem sequence_properties :
  (∀ n : ℕ+, S n = ((n : ℝ) + 1) * sequence_a n / 2) ∧
  sequence_a 1 = 1 →
  (∀ n : ℕ+, sequence_a n = n) ∧
  ¬ ∃ k : ℕ+, k ≥ 2 ∧
    (sequence_b k * sequence_b (k + 2) = sequence_b (k + 1) ^ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l666_66681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l666_66678

/-- The number of days A takes to complete the work alone -/
noncomputable def a_days : ℝ := 45

/-- The ratio of work capacity of A to B -/
noncomputable def capacity_ratio : ℝ := 3 / 2

/-- The number of days A and B take to complete the work together -/
noncomputable def combined_days : ℝ := 27

/-- Theorem stating that given the work capacity ratio and A's individual completion time,
    A and B together will take 27 days to complete the work -/
theorem work_completion_time :
  capacity_ratio = 3 / 2 →
  a_days = 45 →
  combined_days = 27 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l666_66678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l666_66680

/-- The complex number z -/
noncomputable def z : ℂ := (3 + Complex.I) / (1 - 3 * Complex.I) + 2

/-- A complex number is in the first quadrant if its real and imaginary parts are both positive -/
def is_in_first_quadrant (w : ℂ) : Prop := 0 < w.re ∧ 0 < w.im

/-- Theorem stating that z is in the first quadrant -/
theorem z_in_first_quadrant : is_in_first_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l666_66680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scalar_cross_product_theorem_l666_66673

variable (V : Type*) [AddCommGroup V] [Module ℝ V]

-- Define the cross product operation
variable (cross : V → V → V)

-- State the properties of cross product
axiom cross_anticommutative (a b : V) : cross a b = - cross b a
axiom cross_self_zero (a : V) : cross a a = 0

-- Define the main theorem
theorem scalar_cross_product_theorem : 
  ∃! k : ℝ, ∀ u v w : V, u + v + w = 0 → 
    k • (cross v u) + cross v w + cross w u = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scalar_cross_product_theorem_l666_66673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemon_juice_volume_approx_l666_66644

/-- Represents a cylindrical glass -/
structure CylindricalGlass where
  height : ℝ
  diameter : ℝ

/-- Represents the lemonade mixture -/
structure LemonadeMixture where
  lemonJuiceRatio : ℝ
  waterRatio : ℝ

/-- Calculates the volume of lemon juice in a half-filled cylindrical glass -/
noncomputable def lemonJuiceVolume (glass : CylindricalGlass) (mixture : LemonadeMixture) : ℝ :=
  let radius := glass.diameter / 2
  let lemonadeHeight := glass.height / 2
  let lemonadeVolume := Real.pi * radius^2 * lemonadeHeight
  let totalRatio := mixture.lemonJuiceRatio + mixture.waterRatio
  lemonadeVolume * (mixture.lemonJuiceRatio / totalRatio)

/-- The main theorem stating the volume of lemon juice in the glass -/
theorem lemon_juice_volume_approx :
  let glass : CylindricalGlass := { height := 8, diameter := 3 }
  let mixture : LemonadeMixture := { lemonJuiceRatio := 1, waterRatio := 5 }
  abs (lemonJuiceVolume glass mixture - 4.71) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemon_juice_volume_approx_l666_66644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_inequality_part2_k_range_l666_66631

/-- Given functions f and g -/
noncomputable def f (a : ℝ) (x : ℝ) := x^2 - a*x
noncomputable def g (x : ℝ) := Real.log x

/-- Part 1: Prove inequality when a = 1 -/
theorem part1_inequality : ∀ x > 0, f 1 x ≥ x * g x := by sorry

/-- Function r for part 2 -/
noncomputable def r (a : ℝ) (x : ℝ) := f a x + g ((1 + a*x)/2)

/-- Part 2: Prove the range of k -/
theorem part2_k_range : 
  (∀ a ∈ Set.Ioo 1 2, ∃ x₀ ∈ Set.Icc (1/2) 1, r a x₀ > k*(1 - a^2)) ↔ k ≥ 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_inequality_part2_k_range_l666_66631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_race_distance_l666_66621

/-- The minimum distance a runner must travel in a race with given constraints -/
theorem minimum_race_distance : ℕ := by
  let wall_length : ℝ := 800
  let point_A : ℝ × ℝ := (0, -200)
  let point_B : ℝ × ℝ := (400, 600)
  let point_B_reflected : ℝ × ℝ := (400, -600)
  let distance := Real.sqrt ((point_B_reflected.1 - point_A.1)^2 + (point_B_reflected.2 - point_A.2)^2)
  have h : ⌊distance + 0.5⌋ = 566 := by sorry
  exact 566


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_race_distance_l666_66621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_n_l666_66665

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1/2 then x + 1/2 else x^2

def seq (f : ℝ → ℝ) (a₀ : ℝ) : ℕ → ℝ
| 0 => a₀
| n + 1 => f (seq f a₀ n)

theorem existence_of_n (a b : ℝ) (ha : 0 < a) (hb : a < b) (hb1 : b < 1) :
  ∃ n : ℕ, (seq f a (n + 1) - seq f a n) * (seq f b (n + 1) - seq f b n) < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_n_l666_66665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_clock_angle_4_to_5_45_l666_66629

/-- Represents a clock with a given number of hours and divisions between hours. -/
structure Clock where
  hours : ℕ
  divisions : ℕ

/-- Calculates the angle turned by the hour hand of a clock. -/
def angleTurned (c : Clock) (start : ℕ) (end_hour : ℕ) (end_minutes : ℕ) : ℚ :=
  let totalDivisions := c.hours * c.divisions
  let anglePerDivision : ℚ := 360 / totalDivisions
  let startDivisions := start * c.divisions
  let endDivisions := end_hour * c.divisions + (end_minutes * c.divisions / 60)
  (endDivisions - startDivisions) * anglePerDivision

/-- Theorem stating that for a standard clock, when the hour hand moves from 4 o'clock to 5:45, it turns through an angle of 52.5 degrees. -/
theorem standard_clock_angle_4_to_5_45 :
  let c : Clock := { hours := 12, divisions := 5 }
  angleTurned c 4 5 45 = 52.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_clock_angle_4_to_5_45_l666_66629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_darryl_revenue_l666_66696

/-- Represents the types of melons Darryl sells -/
inductive MelonType
  | Cantaloupe
  | Honeydew
  | Watermelon
deriving Repr, DecidableEq

/-- Represents a customer's purchase -/
structure Purchase where
  melonType : MelonType
  quantity : ℕ

def melonPrice (m : MelonType) : ℕ :=
  match m with
  | MelonType.Cantaloupe => 2
  | MelonType.Honeydew => 3
  | MelonType.Watermelon => 4

def initialStock : MelonType → ℕ
  | MelonType.Cantaloupe => 30
  | MelonType.Honeydew => 27
  | MelonType.Watermelon => 20

def finalStock : MelonType → ℕ
  | MelonType.Cantaloupe => 8
  | MelonType.Honeydew => 9
  | MelonType.Watermelon => 7

def damagedMelons : MelonType → ℕ
  | MelonType.Cantaloupe => 3
  | MelonType.Honeydew => 4
  | MelonType.Watermelon => 0

def watermelonDiscount (quantity : ℕ) : ℕ :=
  if quantity > 2 then 2 else 0

def bulkDiscount (totalMelons : ℕ) : ℚ :=
  if totalMelons > 5 then 1/10 else 0

theorem darryl_revenue (purchases : List Purchase) 
  (h1 : ∃ p ∈ purchases, p.melonType = MelonType.Watermelon ∧ p.quantity = 3)
  (h2 : (purchases.filter (fun p => p.quantity > 5)).length = 5) :
  let soldMelons (m : MelonType) := initialStock m - finalStock m - damagedMelons m
  let revenue := purchases.foldl (fun acc p =>
    acc + p.quantity * melonPrice p.melonType - 
    (if p.melonType = MelonType.Watermelon then watermelonDiscount p.quantity else 0)
  ) 0
  let totalMelons := purchases.foldl (fun acc p => acc + p.quantity) 0
  let discountedRevenue := ↑revenue * (1 - bulkDiscount totalMelons)
  discountedRevenue = 124 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_darryl_revenue_l666_66696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_rate_is_correct_l666_66647

/-- The compound interest rate that satisfies the given conditions -/
noncomputable def compound_interest_rate : ℝ := 10

/-- The principal amount for simple interest -/
noncomputable def simple_interest_principal : ℝ := 5250

/-- The principal amount for compound interest -/
noncomputable def compound_interest_principal : ℝ := 4000

/-- The time period in years -/
noncomputable def time : ℝ := 2

/-- The simple interest rate per annum -/
noncomputable def simple_interest_rate : ℝ := 4

/-- Calculate simple interest -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  (principal * rate * time) / 100

/-- Calculate compound interest -/
noncomputable def compound_interest (principal rate time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

theorem compound_interest_rate_is_correct :
  simple_interest simple_interest_principal simple_interest_rate time = 
  (1 / 2) * compound_interest compound_interest_principal compound_interest_rate time :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_rate_is_correct_l666_66647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_A_and_B_l666_66645

open Set

def U : Set ℕ := {x | x ≤ 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {4, 5, 6}

theorem intersection_complement_A_and_B :
  (U \ A) ∩ B = {4, 6} := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_A_and_B_l666_66645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_symmetry_and_decreasing_l666_66657

noncomputable def f (m : ℤ) (x : ℝ) : ℝ := x^(m^2 - 4*m)

theorem power_function_symmetry_and_decreasing (m : ℤ) :
  (∀ x : ℝ, f m x = f m (-x)) →  -- Symmetry about y-axis
  (∀ x y : ℝ, 0 < x → x < y → f m y < f m x) →  -- Decreasing on (0, +∞)
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_symmetry_and_decreasing_l666_66657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hdha_ratio_is_zero_l666_66653

-- Define a triangle with sides 12, 13, and 17
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 12
  hb : b = 13
  hc : c = 17

-- Helper function to calculate area using Heron's formula
noncomputable def area (t : Triangle) : ℝ :=
  let s := (t.a + t.b + t.c) / 2
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

-- Define the altitude AD and point H
noncomputable def Triangle.altitude (t : Triangle) : ℝ := 
  2 * (area t) / t.b

noncomputable def Triangle.H (t : Triangle) : ℝ × ℝ := sorry

-- Define the ratio HD:HA
noncomputable def Triangle.hdha_ratio (t : Triangle) : ℝ := 
  let A : ℝ × ℝ := sorry
  let D : ℝ × ℝ := sorry
  let H : ℝ × ℝ := t.H
  let HD : ℝ := sorry
  let HA : ℝ := sorry
  HD / HA

-- State the theorem
theorem hdha_ratio_is_zero (t : Triangle) : t.hdha_ratio = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hdha_ratio_is_zero_l666_66653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2018_equals_5_l666_66679

def sequence_a : ℕ → ℚ
  | 0 => -1/4  -- Define for 0 to cover all natural numbers
  | 1 => -1/4
  | n + 2 => 1 - 1 / sequence_a (n + 1)

theorem a_2018_equals_5 : sequence_a 2018 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2018_equals_5_l666_66679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_framing_for_specific_picture_l666_66649

/-- Calculates the minimum number of linear feet of framing needed for a picture with given dimensions and border width. -/
def min_framing_feet (original_width : ℚ) (original_height : ℚ) (enlargement_factor : ℚ) (border_width : ℚ) : ℕ :=
  let enlarged_width := original_width * enlargement_factor
  let enlarged_height := original_height * enlargement_factor
  let total_width := enlarged_width + 2 * border_width
  let total_height := enlarged_height + 2 * border_width
  let perimeter := 2 * (total_width + total_height)
  (Nat.ceil (perimeter / 12) : ℕ)

/-- Theorem stating that for a 4-inch by 6-inch picture enlarged 4 times with a 3-inch border, 
    the minimum framing needed is 9 feet. -/
theorem framing_for_specific_picture : 
  min_framing_feet 4 6 4 3 = 9 := by
  sorry

#eval min_framing_feet 4 6 4 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_framing_for_specific_picture_l666_66649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_50_equals_sqrt_3_l666_66611

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 0  -- Add this case for 0
  | 1 => 0
  | (n + 2) => (sequence_a (n + 1) + Real.sqrt 3) / (1 - Real.sqrt 3 * sequence_a (n + 1))

theorem a_50_equals_sqrt_3 : sequence_a 50 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_50_equals_sqrt_3_l666_66611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l666_66625

-- Define the ⊗ operation
noncomputable def otimes (a b : ℝ) : ℝ := if a - b ≤ 2 then a else b

-- Define function f
noncomputable def f (x : ℝ) : ℝ := otimes (3^(x+1)) (1-x)

-- Define function g
def g (x : ℝ) : ℝ := x^2 - 6*x

-- Theorem statement
theorem m_range (m : ℝ) :
  (∀ x y, m < x ∧ x < y ∧ y < m + 1 → f x > f y) ∧
  (∀ x y, m < x ∧ x < y ∧ y < m + 1 → g x > g y) →
  0 ≤ m ∧ m ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l666_66625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_decreasing_l666_66654

/-- Given a > 1, define the function f(x) = (1/a)^x - a^x --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/a)^x - a^x

/-- Theorem stating that f is an odd function and decreasing on ℝ --/
theorem f_odd_and_decreasing (a : ℝ) (h : a > 1) :
  (∀ x, f a (-x) = -(f a x)) ∧ 
  (∀ x y, x < y → f a x > f a y) := by
  sorry

#check f_odd_and_decreasing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_decreasing_l666_66654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_cosine_l666_66674

theorem smallest_angle_cosine (A B C : ℝ) (hABC : A + B + C = Real.pi) 
  (hSin : ∃ k : ℝ, k > 0 ∧ Real.sin A = 3 * k ∧ Real.sin B = 5 * k ∧ Real.sin C = 7 * k) :
  ∃ θ : ℝ, θ = min A (min B C) ∧ Real.cos θ = 13/14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_cosine_l666_66674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_elements_l666_66622

def M : Set ℝ := {x : ℝ | x^2 - x - 2 < 0}

def P : Set ℤ := {x : ℤ | |x - 1| ≤ 3}

def Q : Set ℤ := {x : ℤ | x ∈ P ∧ ¬(↑x ∈ M)}

theorem Q_elements : Q = {-2, -1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_elements_l666_66622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sum_is_14_l666_66687

/-- The sum of integers from a to b, inclusive -/
def sum_integers (a b : ℤ) : ℤ := (b - a + 1) * (a + b) / 2

/-- A 4x4 matrix of integers -/
def Matrix4x4 := Fin 4 → Fin 4 → ℤ

/-- The property that all rows, columns, and main diagonals have the same sum -/
def has_equal_sums (m : Matrix4x4) (s : ℤ) : Prop :=
  (∀ i : Fin 4, (Finset.univ.sum (λ j ↦ m i j)) = s) ∧
  (∀ j : Fin 4, (Finset.univ.sum (λ i ↦ m i j)) = s) ∧
  ((Finset.univ.sum (λ i ↦ m i i)) = s) ∧
  ((Finset.univ.sum (λ i ↦ m i (3 - i))) = s)

theorem equal_sum_is_14 :
  ∃ (m : Matrix4x4),
    (∀ i j, m i j ∈ Finset.Icc (-4 : ℤ) 11) ∧
    (Finset.sum (Finset.Icc (-4 : ℤ) 11) id = sum_integers (-4) 11) ∧
    has_equal_sums m 14 := by
  sorry

#check equal_sum_is_14

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sum_is_14_l666_66687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l666_66642

-- Define the probability of success for group A
noncomputable def P1 : ℝ := 2/3

-- Define the probability of success for group B as a parameter
variable (P2 : ℝ)

-- Define the probability of achieving "Advanced Harmonious Laboratory" in one project
noncomputable def advanced_harmonious_prob (P2 : ℝ) : ℝ :=
  2 * P1 * (1 - P1) * P2 * (1 - P2) + P1^2 * P2^2

-- Define the expected number of "Advanced Harmonious Laboratory" awards in 6 projects
noncomputable def expected_awards (P2 : ℝ) : ℝ :=
  6 * advanced_harmonious_prob P2

-- Theorem for Part I
theorem part_one : advanced_harmonious_prob (1/2) = 1/6 := by sorry

-- Theorem for Part II
theorem part_two : ∀ P2 ∈ Set.Icc (3/4 : ℝ) 1,
  expected_awards P2 ≥ 2.5 ∧ ∀ P2' ∉ Set.Icc (3/4 : ℝ) 1, expected_awards P2' < 2.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l666_66642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_trapezoid_DBCE_l666_66639

-- Define the triangle ABC
structure Triangle (α : Type*) [LinearOrderedField α] where
  A : α × α
  B : α × α
  C : α × α
  isIsosceles : A.1 = C.1 ∧ A.2 = C.2

-- Define the properties of the diagram
structure DiagramProperties (α : Type*) [LinearOrderedField α] where
  ABC : Triangle α
  smallestTriangleArea : α
  areaABC : α
  areaADE : α
  smallestTriangleAreaPositive : smallestTriangleArea > 0
  areaABCValue : areaABC = 40
  areaADEValue : areaADE = 6 * smallestTriangleArea
  smallestTrianglesInABC : 7 * smallestTriangleArea ≤ areaABC

-- State the theorem
theorem area_of_trapezoid_DBCE 
  {α : Type*} [LinearOrderedField α] 
  (diagram : DiagramProperties α) : 
  ∃ (areaTrapezoidDBCE : α), areaTrapezoidDBCE = 34 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_trapezoid_DBCE_l666_66639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiring_decision_l666_66661

structure Candidate where
  writtenScore : ℝ
  interviewScore : ℝ

noncomputable def equalWeightedScore (c : Candidate) : ℝ :=
  (c.writtenScore + c.interviewScore) / 2

noncomputable def weightedScore (c : Candidate) (writtenWeight interviewWeight : ℝ) : ℝ :=
  (c.writtenScore * writtenWeight + c.interviewScore * interviewWeight) / (writtenWeight + interviewWeight)

theorem hiring_decision (candidateA candidateB : Candidate)
  (hA : candidateA = { writtenScore := 85, interviewScore := 75 })
  (hB : candidateB = { writtenScore := 60, interviewScore := 95 }) :
  (equalWeightedScore candidateA > equalWeightedScore candidateB) ∧
  (weightedScore candidateA 0.4 0.6 < weightedScore candidateB 0.4 0.6) := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiring_decision_l666_66661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_touching_planes_l666_66638

/-- A sphere resting on a horizontal plane -/
structure Sphere where
  R : ℝ  -- radius of the sphere
  center : ℝ × ℝ × ℝ := (0, 0, R)

/-- A point in 3D space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The equation of planes touching a sphere at points on a circle -/
def touching_plane_equation (S : Sphere) (P : Point) (d : ℝ) : ℝ → ℝ → ℝ → Prop :=
  fun x y z ↦ x * P.x + y * P.y - d * P.z + d * S.R = S.R^2

/-- Theorem: The equation of planes passing through a point and touching the sphere at points on a circle -/
theorem sphere_touching_planes (S : Sphere) (P : Point) (d : ℝ) :
  ∀ x y z, touching_plane_equation S P d x y z ↔
    (x^2 + y^2 + (z - S.R)^2 = S.R^2 ∧  -- sphere equation
     ∃ x' y', x'^2 + y'^2 = d * (2 * S.R - d) ∧  -- circle equation
     x * x' + y * y' + (z - S.R) * (S.R - d) = S.R^2)  -- tangent plane equation
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_touching_planes_l666_66638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_correct_l666_66610

/-- The function for which we're finding the axis of symmetry -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

/-- The proposed axis of symmetry -/
noncomputable def axis_of_symmetry : ℝ := Real.pi / 12

theorem axis_of_symmetry_correct :
  ∀ x : ℝ, f (axis_of_symmetry - x) = f (axis_of_symmetry + x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_correct_l666_66610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_implies_a_range_l666_66602

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (x^2 - 2*a*x + 3)

-- State the theorem
theorem monotonic_f_implies_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, -1 < x ∧ x < y ∧ y < 1 → f a x ≤ f a y) →
  -2 ≤ a ∧ a ≤ -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_implies_a_range_l666_66602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_complex_quotient_l666_66675

theorem symmetric_complex_quotient :
  ∀ (z₁ z₂ : ℂ),
  (z₁.re = -z₂.re) →              -- symmetry with respect to imaginary axis
  (z₁.im = z₂.im) →               -- symmetry with respect to imaginary axis
  (z₁ = -1 + Complex.I) →         -- given condition for z₁
  (z₁ / z₂ = Complex.I) :=        -- prove that z₁ / z₂ = i
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_complex_quotient_l666_66675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l666_66627

variable (a c : ℝ)
variable (ha : 0 < a)
variable (hc : 0 < c)

noncomputable def f (x : ℝ) : ℝ := 2 * (a - x) * (x + Real.sqrt (x^2 + c^2))

theorem max_value_of_f (a c : ℝ) (ha : 0 < a) (hc : 0 < c) :
  ∃ (M : ℝ), (∀ (x : ℝ), f a c x ≤ M) ∧ (M = a^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l666_66627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_x_for_perfect_square_l666_66672

theorem largest_x_for_perfect_square : 
  ∀ x : ℕ, (∃ y : ℕ, (4:ℕ)^27 + (4:ℕ)^1000 + (4:ℕ)^x = y^2) → x ≤ 1972 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_x_for_perfect_square_l666_66672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_51_l666_66655

def G : ℕ → ℚ
  | 0 => 3  -- Add a case for 0 to cover all natural numbers
  | 1 => 3
  | (n + 2) => (3 * G (n + 1) + 2) / 3

theorem G_51 : G 51 = 109 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_51_l666_66655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vegetable_pricing_theorem_l666_66648

/-- Represents the price reduction scenario and discount options --/
structure VegetablePricing where
  original_price : ℝ
  final_price : ℝ
  quantity : ℝ
  discount_percentage : ℝ
  cash_discount_per_ton : ℝ

/-- Calculates the average percentage reduction --/
noncomputable def average_reduction (p : VegetablePricing) : ℝ :=
  1 - (p.final_price / p.original_price) ^ (1/2)

/-- Calculates the total cost for option one (percentage discount) --/
noncomputable def option_one_cost (p : VegetablePricing) : ℝ :=
  p.final_price * (1 - p.discount_percentage) * p.quantity

/-- Calculates the total cost for option two (cash discount) --/
noncomputable def option_two_cost (p : VegetablePricing) : ℝ :=
  p.final_price * p.quantity - p.cash_discount_per_ton * (p.quantity / 1000)

/-- The main theorem to be proved --/
theorem vegetable_pricing_theorem (p : VegetablePricing) 
  (h1 : p.original_price = 10)
  (h2 : p.final_price = 6.4)
  (h3 : p.quantity = 2000)
  (h4 : p.discount_percentage = 0.2)
  (h5 : p.cash_discount_per_ton = 1000) :
  average_reduction p = 0.2 ∧ 
  option_one_cost p < option_two_cost p := by
  sorry

#eval "Vegetable Pricing Theorem"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vegetable_pricing_theorem_l666_66648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_PX_l666_66663

-- Define the points in the Euclidean plane
variable (C D W X P : EuclideanSpace ℝ (Fin 2))

-- Define the given conditions
variable (h_parallel : Parallel (Line.throughPts C D) (Line.throughPts W X))
variable (h_CX : dist C X = 60)
variable (h_DP : dist D P = 20)
variable (h_PW : dist P W = 40)

-- State the theorem
theorem segment_length_PX : dist P X = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_PX_l666_66663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l666_66626

theorem angle_in_second_quadrant (α : ℝ) 
  (h1 : Real.sin α = 3/5) 
  (h2 : π/2 < α ∧ α < π) : 
  Real.tan α = -3/4 ∧ (2*Real.sin α + 3*Real.cos α) / (Real.cos α - Real.sin α) = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l666_66626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_mpg_l666_66693

/-- Calculates the average miles per gallon for a round trip -/
noncomputable def average_mpg (total_distance : ℝ) (mpg_first_half : ℝ) (mpg_second_half : ℝ) : ℝ :=
  total_distance / ((total_distance / 2 / mpg_first_half) + (total_distance / 2 / mpg_second_half))

/-- Theorem: The average mpg for the given round trip is 18.75 -/
theorem round_trip_mpg :
  let total_distance : ℝ := 300
  let mpg_first_half : ℝ := 25
  let mpg_second_half : ℝ := 15
  average_mpg total_distance mpg_first_half mpg_second_half = 18.75 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval average_mpg 300 25 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_mpg_l666_66693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ibrahim_lacks_129_euros_l666_66617

/-- Calculates the amount of money Ibrahim lacks to buy all items --/
noncomputable def money_lacking (mp3_price cd_price headphones_price case_price savings father_contribution discount_threshold discount_rate : ℚ) : ℚ :=
  let total_price := mp3_price + cd_price + headphones_price + case_price
  let discounted_price := if total_price > discount_threshold
                          then total_price * (1 - discount_rate)
                          else total_price
  let available_money := savings + father_contribution
  discounted_price - available_money

/-- Theorem stating that Ibrahim lacks 129 euros to buy all items --/
theorem ibrahim_lacks_129_euros :
  money_lacking 135 25 50 30 55 20 150 (15/100) = 129 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ibrahim_lacks_129_euros_l666_66617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_consecutive_numbers_simplification_l666_66605

theorem product_of_consecutive_numbers_simplification :
  2 * (((2007 * 2009 * 2011 * 2013 : ℝ) + 10 * 2010 * 2010 - 9) ^ (1/4)) - 4000 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_consecutive_numbers_simplification_l666_66605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_zeros_iff_omega_in_range_l666_66618

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then Real.exp x + x
  else if 0 ≤ x ∧ x ≤ Real.pi then Real.sin (ω * x - Real.pi / 3)
  else 0  -- undefined for x > π, but we need to cover all reals

def has_four_zeros (ω : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧
    f ω x₁ = 0 ∧ f ω x₂ = 0 ∧ f ω x₃ = 0 ∧ f ω x₄ = 0

theorem four_zeros_iff_omega_in_range (ω : ℝ) :
  has_four_zeros ω ↔ (7 / 3 ≤ ω ∧ ω < 10 / 3) :=
sorry

#check four_zeros_iff_omega_in_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_zeros_iff_omega_in_range_l666_66618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_projection_area_ratio_l666_66603

/-- Represents a triangle in a 2D plane -/
structure Triangle where
  base : ℝ
  height : ℝ

/-- Represents the oblique projection method -/
noncomputable def obliqueProjection (t : Triangle) : Triangle :=
  { base := t.base,  -- x-axis lengths unchanged
    height := t.height / 2 }  -- y-axis lengths halved

/-- The ratio of areas between the projected and original triangles -/
noncomputable def areaRatio (original : Triangle) (projected : Triangle) : ℝ :=
  (projected.base * projected.height) / (original.base * original.height)

theorem oblique_projection_area_ratio (t : Triangle) : 
  areaRatio t (obliqueProjection t) = Real.sqrt 2 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_projection_area_ratio_l666_66603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_best_restaurant_l666_66686

/-- Represents the comparison result between two restaurants in a category -/
inductive Comparison
| Better
| Worse
| Equal

/-- Represents a restaurant -/
structure Restaurant :=
  (id : ℕ)

/-- Compares two restaurants in a given category -/
def compare (category : ℕ) (r1 r2 : Restaurant) : Comparison :=
  sorry

/-- The set of all restaurants -/
def Restaurants : Finset Restaurant :=
  sorry

variable (n : ℕ)

/-- The number of restaurants is positive -/
axiom positive_n : 0 < n

/-- The number of restaurants matches the set size -/
axiom restaurant_count : Finset.card Restaurants = n

/-- Comparison is transitive -/
axiom transitive_comparison (category : ℕ) (r1 r2 r3 : Restaurant) :
  compare category r1 r2 = Comparison.Better →
  compare category r2 r3 = Comparison.Better →
  compare category r1 r3 = Comparison.Better

/-- At least one category can be compared for each pair -/
axiom comparable_category (r1 r2 : Restaurant) :
  (compare 0 r1 r2 ≠ Comparison.Equal) ∨ (compare 1 r1 r2 ≠ Comparison.Equal)

/-- There exists a restaurant that is not worse than any other in at least one category -/
theorem exists_best_restaurant :
  ∃ (R : Restaurant), ∀ (S : Restaurant),
    S ≠ R → (compare 0 R S = Comparison.Better) ∨ (compare 1 R S = Comparison.Better) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_best_restaurant_l666_66686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_intervals_sin_alpha_value_l666_66633

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 - 1/2

theorem f_monotone_intervals (k : ℤ) :
  MonotoneOn f (Set.Icc (-(π/3) + k * π) ((π/6) + k * π)) :=
sorry

theorem sin_alpha_value (α : ℝ) (h1 : f (α/2) = -3/5) (h2 : α ∈ Set.Ioo (-π/2) 0) :
  Real.sin α = -3 * Real.sqrt 3 / 10 - 2/5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_intervals_sin_alpha_value_l666_66633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_can_win_l666_66609

def game_state := ℚ × ℚ

def initial_state : game_state := (1/2009, 1/2008)

def increase_first (s : game_state) (x : ℚ) : game_state :=
  (s.1 + x, s.2)

def increase_second (s : game_state) (x : ℚ) : game_state :=
  (s.1, s.2 + x)

def is_winning_state (s : game_state) : Prop :=
  s.1 = 1 ∨ s.2 = 1

theorem vasya_can_win :
  ∃ (strategy : ℕ → ℚ),
    ∀ (player_choice : ℕ → Bool),
      ∃ (n : ℕ),
        let final_state := (List.foldl 
          (λ s i ↦ if player_choice i
            then increase_first s (strategy i)
            else increase_second s (strategy i))
          initial_state
          (List.range n))
        is_winning_state final_state := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_can_win_l666_66609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_approximation_l666_66667

/-- The volume of a cone given its radius and height -/
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The height of a cone given its radius and volume -/
noncomputable def cone_height (r v : ℝ) : ℝ := (3 * v) / (Real.pi * r^2)

theorem cone_height_approximation (r v : ℝ) (h : ℝ) :
  r = 10 →
  v = 2199.114857512855 →
  h = cone_height r v →
  ∃ ε > 0, |h - 21| < ε :=
by
  intros hr hv hh
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_approximation_l666_66667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_120_equals_1_l666_66620

def b : ℕ → ℚ
  | 0 => 1  -- Adding this case to handle Nat.zero
  | 1 => 1
  | 2 => -1
  | n+3 => (1 - b (n+2) - 3 * b (n+1)) / (2 * b (n+1))

theorem b_120_equals_1 : b 120 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_120_equals_1_l666_66620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_complex_l666_66690

theorem square_of_complex (i : ℂ) (h : i^2 = -1) :
  (5 + 3*i)^2 = 16 + 30*i := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_complex_l666_66690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_paths_to_5_5_l666_66632

/-- Represents a point in the Cartesian plane -/
structure Point where
  x : Nat
  y : Nat

/-- Defines the valid moves for the particle -/
inductive Move
  | Right
  | Up
  | Diagonal

/-- A path is a list of moves -/
def ParticlePath := List Move

/-- Checks if a path is valid (no right angle turns) -/
def isValidPath (path : ParticlePath) : Bool :=
  sorry

/-- Counts the number of valid paths from (0,0) to a given point -/
def countValidPaths (p : Point) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem valid_paths_to_5_5 :
  countValidPaths ⟨5, 5⟩ = 252 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_paths_to_5_5_l666_66632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_roots_l666_66671

theorem cubic_equation_roots (a : ℝ) (x₁ x₂ x₃ : ℝ) : 
  (x₁^3 - (a+2)*x₁^2 + (2*a+1)*x₁ - a = 0) ∧ 
  (x₂^3 - (a+2)*x₂^2 + (2*a+1)*x₂ - a = 0) ∧ 
  (x₃^3 - (a+2)*x₃^2 + (2*a+1)*x₃ - a = 0) ∧
  (2/x₁ + 2/x₂ = 3/x₃) →
  (a = 2 ∧ x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 2) ∨ 
  (a = 3/4 ∧ x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 3/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_roots_l666_66671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_period_and_max_value_l666_66607

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x) + 4 * Real.cos (2 * x)

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  (∃ (M : ℝ), (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x : ℝ), f x = M)) :=
by sorry

theorem f_period_and_max_value :
  (∃ (p : ℝ), p = Real.pi ∧ p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  (∃ (M : ℝ), M = 5 ∧ (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x : ℝ), f x = M)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_period_and_max_value_l666_66607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_perfect_square_factorial_product_l666_66616

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

def factorial_product (n : ℕ) : ℕ := Nat.factorial n * Nat.factorial (n + 1)

theorem unique_perfect_square_factorial_product :
  ∃! n : ℕ, n ≥ 14 ∧ n ≤ 18 ∧ is_perfect_square (factorial_product n / 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_perfect_square_factorial_product_l666_66616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_f_neg_eight_eq_neg_one_l666_66689

-- Define the logarithm with base 1/3
noncomputable def log_one_third (x : ℝ) : ℝ := Real.log x / Real.log (1/3)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then log_one_third (x + 1) else log_one_third (-x + 1)

-- Define the function φ
noncomputable def φ (x : ℝ) : ℝ := log_one_third (-x + 1)

-- State the theorem
theorem phi_f_neg_eight_eq_neg_one
  (h_even : ∀ x, f x = f (-x)) -- f is even
  (h_f_nonneg : ∀ x ≥ 0, f x = log_one_third (x + 1)) -- definition of f for x ≥ 0
  (h_f_neg : ∀ x < 0, f x = φ x) -- definition of f for x < 0
  : φ (f (-8)) = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_f_neg_eight_eq_neg_one_l666_66689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_l666_66664

/-- Represents the lengths of edges in the pyramid --/
inductive EdgeLength
  | thirteen
  | twentyfour
  | thirtyseven

/-- Represents a triangular face of the pyramid --/
structure TriangularFace where
  edge1 : EdgeLength
  edge2 : EdgeLength
  edge3 : EdgeLength

/-- Represents the pyramid DABC --/
structure Pyramid where
  faces : List TriangularFace
  edge_length_condition : ∀ f ∈ faces, f.edge1 ∈ [EdgeLength.thirteen, EdgeLength.twentyfour, EdgeLength.thirtyseven] ∧
                                       f.edge2 ∈ [EdgeLength.thirteen, EdgeLength.twentyfour, EdgeLength.thirtyseven] ∧
                                       f.edge3 ∈ [EdgeLength.thirteen, EdgeLength.twentyfour, EdgeLength.thirtyseven]
  not_equilateral : ∀ f ∈ faces, f.edge1 ≠ f.edge2 ∨ f.edge1 ≠ f.edge3 ∨ f.edge2 ≠ f.edge3

/-- Calculates the surface area of a pyramid --/
noncomputable def surface_area (p : Pyramid) : ℝ :=
  sorry

/-- Theorem stating that the surface area of the pyramid DABC is approximately 600.6 --/
theorem pyramid_surface_area (p : Pyramid) : ∃ (ε : ℝ), ε > 0 ∧ |surface_area p - 600.6| < ε :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_l666_66664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jake_vodka_shots_l666_66670

/-- Proves that Jake split 8 shots of vodka with his friend -/
theorem jake_vodka_shots : 
  ∃ (shot_size alcohol_percentage jake_pure_alcohol : ℝ) (total_shots : ℕ),
    shot_size = 1.5 ∧
    alcohol_percentage = 0.5 ∧
    jake_pure_alcohol = 3 ∧
    total_shots = 8 ∧
    (jake_pure_alcohol / (shot_size * alcohol_percentage)) * 2 = total_shots := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jake_vodka_shots_l666_66670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_heights_eq_three_l666_66662

/-- Represents a triangle with its side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- Calculates the sum of medians of a triangle -/
noncomputable def sumOfMedians (t : Triangle) : ℝ :=
  1/2 * Real.sqrt (((t.a + t.b)^2 + (t.b + t.c)^2 + (t.c + t.a)^2 - t.a^2 - t.b^2 - t.c^2) / 2)

/-- Calculates the sum of heights of a triangle -/
noncomputable def sumOfHeights (t : Triangle) : ℝ :=
  2 * t.a * t.b * t.c / Real.sqrt ((t.a + t.b + t.c) * (-t.a + t.b + t.c) * (t.a - t.b + t.c) * (t.a + t.b - t.c))

/-- Theorem: The maximum sum of heights for a triangle with sum of medians equal to 3 is 3 -/
theorem max_sum_of_heights_eq_three (t : Triangle) (h : sumOfMedians t = 3) :
  sumOfHeights t ≤ 3 ∧ ∃ (t_eq : Triangle), sumOfMedians t_eq = 3 ∧ sumOfHeights t_eq = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_heights_eq_three_l666_66662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_and_sum_l666_66688

-- Define the arithmetic sequence a_n
def a : ℕ+ → ℚ := sorry

-- Define the sequence c_n
def c (n : ℕ+) : ℚ := 2 / (a (n + 1) * a (n + 2))

-- Define the sum of the first n terms of c_n
def T (n : ℕ+) : ℚ := (Finset.range n).sum (λ i => c ⟨i + 1, Nat.succ_pos i⟩)

theorem arithmetic_sequence_and_sum :
  (∀ n : ℕ+, ∃ d : ℚ, a (n + 1) = a n + d) →  -- a_n is an arithmetic sequence
  a 1 + a 2 + a 3 = 6 →                       -- First condition
  a 5 = 5 →                                   -- Second condition
  (∀ n : ℕ+, a n = n) ∧                       -- Prove a_n = n
  (∀ n : ℕ+, T n = n / (n + 2)) :=            -- Prove T_n = n / (n + 2)
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_and_sum_l666_66688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_angle_bisector_theorem_l666_66659

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line segment between two points -/
def LineSegment (A B : Point) := {P : Point | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P.x = A.x + t * (B.x - A.x) ∧ P.y = A.y + t * (B.y - A.y)}

/-- A square in a 2D plane -/
structure Square where
  A : Point
  B : Point
  C : Point
  D : Point

/-- The perimeter of a square -/
def Perimeter (s : Square) := 
  LineSegment s.A s.B ∪ LineSegment s.B s.C ∪ LineSegment s.C s.D ∪ LineSegment s.D s.A

/-- An angle bisector of an angle formed by three points -/
def AngleBisector (A B C : Point) := {P : Point | ∃ t : ℝ, 0 < t ∧ P.x = B.x + t * ((A.x - B.x) + (C.x - B.x)) ∧ P.y = B.y + t * ((A.y - B.y) + (C.y - B.y))}

/-- The length of a line segment -/
noncomputable def Length (A B : Point) : ℝ := Real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2)

/-- The main theorem -/
theorem square_angle_bisector_theorem (s : Square) (M N : Point) :
  M ∈ Perimeter s →
  M ≠ s.C →
  (M ∈ LineSegment s.C s.D ∧ N ∈ AngleBisector s.B s.A M ∩ Perimeter s) ∨
  (M ∈ LineSegment s.B s.C ∧ N ∈ AngleBisector s.D s.A M ∩ Perimeter s) →
  (Length s.B N + Length s.D M = Length s.A M) ∨
  (Length s.D N + Length s.B M = Length s.A M) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_angle_bisector_theorem_l666_66659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asian_games_driver_proof_l666_66641

def driving_distances : List Int := [6, -4, 2, -3, 7, -3, -5, 5, 6, -8]

def calculate_fare (distance : Int) : Int :=
  if distance ≤ 3 then 8 else 8 + 2 * (distance - 3)

def total_fare (distances : List Int) : Int :=
  distances.map (fun d => calculate_fare (Int.natAbs d)) |>.sum

theorem asian_games_driver_proof :
  (driving_distances.take 7).sum = 0 ∧
  driving_distances.sum = 3 ∧
  total_fare driving_distances = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asian_games_driver_proof_l666_66641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_increases_l666_66694

noncomputable def inverse_proportion (x : ℝ) : ℝ := -2 / x

theorem inverse_proportion_increases (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a < b) :
  inverse_proportion a > inverse_proportion b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_increases_l666_66694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l666_66640

theorem sin_cos_difference (θ : Real) (h1 : θ ∈ Set.Ioo 0 (π/4)) 
  (h2 : Real.sin θ + Real.cos θ = 4/3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l666_66640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l666_66682

/-- Calculates the speed of a train in km/h given its length and time to cross a fixed point. -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Proves that a train with given length and crossing time has a specific speed. -/
theorem train_speed_theorem (length : ℝ) (time : ℝ) 
  (h1 : length = 100) 
  (h2 : time = 20) : 
  train_speed length time = 18 := by
  sorry

/-- Evaluates the train speed for the given problem. -/
def evaluate_train_speed : ℚ :=
  (100 / 20) * (18 / 5)

#eval evaluate_train_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l666_66682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_positive_integer_solution_l666_66651

theorem no_positive_integer_solution :
  ¬ ∃ (x : ℕ+), |((x : ℝ) + 4)| < (x : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_positive_integer_solution_l666_66651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_sevens_up_to_80_l666_66660

/-- Count the number of 7s in a single digit -/
def countSevensInDigit (d : ℕ) : ℕ :=
  if d = 7 then 1 else 0

/-- Count the number of 7s in a two-digit number -/
def countSevensInTwoDigitNumber (n : ℕ) : ℕ :=
  countSevensInDigit (n / 10) + countSevensInDigit (n % 10)

/-- Count the number of 7s in house numbers from 1 to n -/
def countSevensUpTo (n : ℕ) : ℕ :=
  (List.range n).map (fun i => countSevensInTwoDigitNumber (i + 1)) |>.sum

theorem count_sevens_up_to_80 : countSevensUpTo 80 = 9 := by
  sorry

#eval countSevensUpTo 80

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_sevens_up_to_80_l666_66660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_sixth_plus_two_x_l666_66635

theorem sin_pi_sixth_plus_two_x (x : ℝ) :
  Real.sin (π / 6 - x) = 4 / 5 → Real.sin (π / 6 + 2 * x) = -7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_sixth_plus_two_x_l666_66635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_telegraph_post_l666_66623

/-- Calculates the time (in seconds) for a train to pass a stationary point -/
noncomputable def train_passing_time (length : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms : ℝ := speed_kmh * (1000 / 3600)
  length / speed_ms

/-- Theorem: A train 40 meters long, moving at 36 km/h, takes 4 seconds to pass a stationary point -/
theorem train_passing_telegraph_post :
  train_passing_time 40 36 = 4 := by
  -- Unfold the definition of train_passing_time
  unfold train_passing_time
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_telegraph_post_l666_66623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l666_66628

theorem inequality_solution_set (x : ℝ) :
  (3 * x / (2 * x - 1) ≤ 2) ↔ (x ∈ Set.Iic (1/2) ∪ Set.Ici 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l666_66628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_prime_not_dividing_power_minus_prime_l666_66683

theorem exists_prime_not_dividing_power_minus_prime (p : ℕ) (hp : Nat.Prime p) :
  ∃ q : ℕ, Nat.Prime q ∧ ∀ n : ℕ, ¬(q ∣ n^p - p) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_prime_not_dividing_power_minus_prime_l666_66683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisibility_condition_l666_66600

theorem prime_divisibility_condition (a b : ℕ) :
  let p := a^2 + b + 1
  Prime p ∧
  p ∣ (b^2 - a^3 - 1) ∧
  ¬(p ∣ (a + b - 1)^2) →
  ∃ x : ℕ, x ≥ 2 ∧ a = 2^x ∧ b = 2^(2*x) - 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisibility_condition_l666_66600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l666_66697

/-- Represents a right circular cone --/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a spherical marble --/
structure Marble where
  radius : ℝ

/-- Calculate the volume of a cone given its base radius and height --/
noncomputable def coneVolume (r : ℝ) (h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- Calculate the volume of a sphere given its radius --/
noncomputable def sphereVolume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

/-- Theorem statement --/
theorem liquid_rise_ratio 
  (cone1 cone2 : Cone) 
  (marble : Marble) 
  (h1 h2 : ℝ) :
  cone1.baseRadius = 4 →
  cone2.baseRadius = 8 →
  marble.radius = 1 →
  coneVolume cone1.baseRadius h1 = coneVolume cone2.baseRadius h2 →
  (let rise1 := sphereVolume marble.radius / (Real.pi * cone1.baseRadius^2)
   let rise2 := sphereVolume marble.radius / (Real.pi * cone2.baseRadius^2)
   rise1 / rise2 = 4) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l666_66697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_shift_equivalence_l666_66656

noncomputable section

/-- The given function -/
def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x)

/-- The reference function -/
def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

/-- The shifted reference function -/
def h (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

theorem function_equivalence (x : ℝ) : f x = h x := by
  sorry

theorem shift_equivalence (x : ℝ) : h x = g (x - Real.pi / 6) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_shift_equivalence_l666_66656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_two_solutions_l666_66604

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def validTriangle (t : Triangle) : Prop :=
  t.a = 2 ∧ t.B = Real.pi / 4

-- Theorem 1: Exactly one solution when b = √2
theorem unique_solution (t : Triangle) (h : validTriangle t) :
  t.b = Real.sqrt 2 → ∃! t', validTriangle t' ∧ t'.b = Real.sqrt 2 :=
sorry

-- Theorem 2: Two solutions when b is in (√2, 2)
theorem two_solutions (t : Triangle) (h : validTriangle t) :
  (∃ t₁ t₂, t₁ ≠ t₂ ∧ validTriangle t₁ ∧ validTriangle t₂ ∧ t₁.b = t.b ∧ t₂.b = t.b) ↔
  Real.sqrt 2 < t.b ∧ t.b < 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_two_solutions_l666_66604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l666_66612

/-- The length of a train that crosses two platforms of different lengths in given times. -/
theorem train_length (platform1_length platform2_length time1 time2 : ℝ)
                     (h1 : platform1_length = 140)
                     (h2 : platform2_length = 250)
                     (h3 : time1 = 15)
                     (h4 : time2 = 20)
                     (speed : ℝ)
                     (train_length : ℝ)
                     (h5 : platform1_length + train_length = time1 * speed)
                     (h6 : platform2_length + train_length = time2 * speed) : 
                     train_length = 190 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l666_66612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l666_66669

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  angleSum : A + B + C = π
  positiveAngles : 0 < A ∧ 0 < B ∧ 0 < C

-- Define the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : (1 + Real.sin abc.A) / Real.cos abc.A = Real.sin (2 * abc.B) / (1 - Real.cos (2 * abc.B)))
  (h2 : ∃ (x : ℝ), 0 < x ∧ x < 1 ∧ 
    x^2 * 3 + 2 * x * (1 - x) * Real.sqrt 3 * (Real.sqrt 3 / 2) + (1 - x)^2 = 2) :
  Real.sin (abc.C - abc.B) = 1 ∧ 
  ∃ (x : ℝ), x = (Real.sqrt 5 - 1) / 2 ∧ 
    x^2 * 3 + 2 * x * (1 - x) * Real.sqrt 3 * (Real.sqrt 3 / 2) + (1 - x)^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l666_66669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_failure_rate_l666_66619

theorem exam_failure_rate 
  (total_candidates : ℕ)
  (num_girls : ℕ)
  (boys_pass_rate : ℚ)
  (girls_pass_rate : ℚ)
  (h1 : total_candidates = 2000)
  (h2 : num_girls = 900)
  (h3 : boys_pass_rate = 28 / 100)
  (h4 : girls_pass_rate = 32 / 100) : 
  (total_candidates - (((total_candidates - num_girls : ℚ) * boys_pass_rate).floor + 
  (num_girls * girls_pass_rate).floor : ℚ)) / total_candidates * 100 = 702 / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_failure_rate_l666_66619
