import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_perfect_square_on_12_sided_die_l775_77539

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def die_sides : ℕ := 12

def perfect_squares_on_die : Finset ℕ := 
  {1, 4, 9}

theorem probability_of_perfect_square_on_12_sided_die : 
  (perfect_squares_on_die.card : ℚ) / die_sides = 1 / 4 := by
  simp [perfect_squares_on_die, die_sides]
  norm_num

#eval perfect_squares_on_die.card
#eval die_sides

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_perfect_square_on_12_sided_die_l775_77539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_image_is_parabolic_segment_and_curve_l775_77595

-- Define the triangle OAB
def triangle_OAB : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p = (0, 0) ∨ p = (1, 0) ∨ p = (0, 1) ∨
    (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (p = (t, 0) ∨ p = (0, t) ∨ p = (t, 1 - t)))}

-- Define the transformation
noncomputable def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (x^2 * Real.cos y, x * Real.sin y)

-- Define the image of triangle OAB under the transformation
def image_triangle_OAB : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | ∃ p ∈ triangle_OAB, q = transform p}

-- Theorem statement
theorem image_is_parabolic_segment_and_curve :
  ∃ f g : ℝ → ℝ,
    (∀ t ∈ Set.Icc 0 1, (f t, 0) ∈ image_triangle_OAB) ∧
    (∀ t ∈ Set.Icc 0 1, (g t, t) ∈ image_triangle_OAB) ∧
    f 0 = 0 ∧ f 1 = 1 ∧ g 0 = 0 ∧ g 1 = 0 ∧
    (∀ q ∈ image_triangle_OAB, ∃ t ∈ Set.Icc 0 1, q = (f t, 0) ∨ q = (g t, t)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_image_is_parabolic_segment_and_curve_l775_77595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l775_77533

/-- Represents a train with its length and speed. -/
structure Train where
  length : ℝ
  speed : ℝ

/-- The time it takes for a train to pass an object of a given length. -/
noncomputable def passingTime (train : Train) (objectLength : ℝ) : ℝ :=
  (train.length + objectLength) / train.speed

theorem train_length_calculation (train : Train) 
  (h1 : passingTime train 450 = 45)
  (h2 : passingTime train 0 = 18) : 
  train.length = 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l775_77533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sequence_finitely_many_primes_l775_77525

/-- A sequence of natural numbers -/
def SequenceOfNaturals := ℕ → ℕ

/-- Predicate to check if a number is prime -/
def IsPrime (n : ℕ) : Prop := Nat.Prime n

/-- Predicate to check if only finitely many elements of a sequence are prime -/
def FinitelyManyPrimes (s : SequenceOfNaturals) : Prop :=
  ∀ k : ℤ, ∃ N : ℕ, ∀ n ≥ N, ¬IsPrime (s n + k.natAbs)

/-- Theorem stating the existence of an infinite sequence of natural numbers
    such that for any integer k, only finitely many elements of (aₙ + k) are prime -/
theorem exists_sequence_finitely_many_primes :
  ∃ a : SequenceOfNaturals, FinitelyManyPrimes a ∧ Function.Injective a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sequence_finitely_many_primes_l775_77525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_cookie_price_theorem_l775_77538

noncomputable def jane_cookie_radius : ℝ := 4
noncomputable def bob_cookie_side : ℝ := 6
def jane_cookie_count : ℕ := 18
noncomputable def jane_cookie_price : ℝ := 0.50

noncomputable def jane_total_dough : ℝ := jane_cookie_count * Real.pi * jane_cookie_radius^2
noncomputable def bob_cookie_area : ℝ := bob_cookie_side^2
noncomputable def bob_cookie_count : ℝ := jane_total_dough / bob_cookie_area

theorem bob_cookie_price_theorem :
  (jane_cookie_count * jane_cookie_price * 100) / bob_cookie_count = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_cookie_price_theorem_l775_77538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l775_77519

/-- The sum of the infinite series ∑_{n=1}^∞ (n^2 + 3n - 2) / (n+3)! is equal to 1/2. -/
theorem infinite_series_sum : 
  ∑' (n : ℕ), (((n^2 : ℝ) + 3*n - 2) / (Nat.factorial (n+3) : ℝ)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l775_77519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_simplification_l775_77554

theorem cube_root_sum_simplification :
  (512 : ℝ) ^ (1/3) / (216 : ℝ) ^ (1/3) + (343 : ℝ) ^ (1/3) / (125 : ℝ) ^ (1/3) = 41 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_simplification_l775_77554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l775_77568

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ :=
  Real.sqrt 3 * (Real.cos (ω * x + φ))^2 - 
  Real.cos (ω * x + φ) * Real.sin (ω * x + φ + Real.pi/3) - 
  Real.sqrt 3 / 4

noncomputable def g (f : ℝ → ℝ) (m : ℝ) (x : ℝ) : ℝ :=
  (f (x - 5/6))^2 + 1/4 * f (x - 1/3) + m

theorem function_properties 
  (ω φ : ℝ) 
  (hω : ω > 0) 
  (hφ : 0 < φ ∧ φ < Real.pi/2) :
  -- Extremal points and adjacent symmetry centers form an isosceles right triangle
  -- (2/3, 0) is a symmetry center of f
  -- These conditions are assumed but not formalized due to complexity
  
  -- 1. ω = π/2 and φ = π/3
  (ω = Real.pi/2 ∧ φ = Real.pi/3) ∧
  
  -- 2. f(x) = 1/2 * cos(πx + 5π/6)
  (∀ x, f ω φ x = 1/2 * Real.cos (Real.pi * x + 5 * Real.pi/6)) ∧
  
  -- 3. f(x) is monotonically decreasing on [0, 1/6] ∪ [5/6, 2] for x ∈ [0, 2]
  (∀ x₁ x₂, ((0 ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ 1/6) ∨ (5/6 ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ 2)) →
    f ω φ x₂ ≤ f ω φ x₁) ∧
  
  -- 4. For g(x) = f²(x - 5/6) + 1/4 * f(x - 1/3) + m, m ∈ [-17/64, -1/8] when g(x) has a root for x ∈ [5/6, 3/2]
  (∀ m, (∃ x, 5/6 ≤ x ∧ x ≤ 3/2 ∧ g (f ω φ) m x = 0) →
    -17/64 ≤ m ∧ m ≤ -1/8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l775_77568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_spherical_correct_l775_77507

noncomputable def rectangular_to_spherical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2 + z^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 then Real.pi + Real.arctan (y / x)
           else if y ≥ 0 then Real.pi / 2
           else 3 * Real.pi / 2
  let φ := Real.arccos (z / ρ)
  (ρ, θ, φ)

theorem rectangular_to_spherical_correct :
  let (ρ, θ, φ) := rectangular_to_spherical 1 (-4) (2 * Real.sqrt 3)
  ρ = Real.sqrt 29 ∧
  θ = Real.pi - Real.arctan 4 ∧
  φ = Real.arccos ((2 * Real.sqrt 3) / Real.sqrt 29) ∧
  ρ > 0 ∧
  0 ≤ θ ∧ θ < 2 * Real.pi ∧
  0 ≤ φ ∧ φ ≤ Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_spherical_correct_l775_77507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_set_of_m_l775_77522

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := a * x^2 + Real.log x + 1

-- State the theorem
theorem value_set_of_m (m : ℝ) :
  (∀ a x : ℝ, -2 < a ∧ a < -1 ∧ 1 ≤ x ∧ x ≤ 2 → m * a - f a x > a^2) →
  m ≤ -3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_set_of_m_l775_77522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_Q_l775_77597

-- Define the ellipse C
def ellipse_C : ℝ × ℝ → Prop := λ p => p.1^2/24 + p.2^2/16 = 1

-- Define the line l
def line_l : ℝ × ℝ → Prop := λ p => p.1/12 + p.2/8 = 1

-- Define a point on a line
def point_on_line (P : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop :=
  l P

-- Define the intersection of a ray with an ellipse
def ray_intersects_ellipse (O R : ℝ × ℝ) (C : ℝ × ℝ → Prop) : Prop :=
  ∃ t : ℝ, t > 0 ∧ C (⟨O.1 + t * (R.1 - O.1), O.2 + t * (R.2 - O.2)⟩)

-- Define the relationship between OQ, OP, and OR
def point_Q_relation (O P Q R : ℝ × ℝ) : Prop :=
  ((Q.1 - O.1)^2 + (Q.2 - O.2)^2) * 
  ((P.1 - O.1)^2 + (P.2 - O.2)^2) = 
  ((R.1 - O.1)^2 + (R.2 - O.2)^2)^2

-- Theorem statement
theorem locus_of_Q (O P Q R : ℝ × ℝ) :
  point_on_line P line_l →
  ray_intersects_ellipse O R ellipse_C →
  point_Q_relation O P Q R →
  ∃ x y : ℝ, Q = (x, y) ∧ ((x-1)^2)/(5/2) + ((y-1)^2)/(5/3) = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_Q_l775_77597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_f_equation_l775_77566

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then x^2 + 2 else 2*x

-- State the theorem
theorem solve_f_equation :
  ∀ x₀ : ℝ, f x₀ = 8 ↔ x₀ = 4 ∨ x₀ = -Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_f_equation_l775_77566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_result_l775_77581

theorem game_result (initial_a initial_b initial_c : ℤ) 
  (h_initial_a : initial_a = 10)
  (h_initial_b : initial_b = 57)
  (h_initial_c : initial_c = 29)
  (final_total : ℤ) 
  (h_final_total : final_total = initial_a + initial_b + initial_c)
  (x : ℤ) -- Amount A won
  (h_final_a : initial_a + x = (initial_a + x))
  (h_final_b : initial_b - 2*(initial_a + x) = (initial_a + x))
  (h_final_c : initial_c + (3*x - initial_c) = 3*x) :
  3*x - initial_c = -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_result_l775_77581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_count_l775_77543

def rod_lengths : Finset ℕ := Finset.range 40

theorem quadrilateral_count (a b c : ℕ) (ha : a ∈ rod_lengths) (hb : b ∈ rod_lengths) (hc : c ∈ rod_lengths)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) (h_order : a < b ∧ b < c) :
  (rod_lengths.filter (fun d => d ≠ a ∧ d ≠ b ∧ d ≠ c ∧
    d + a + b > c ∧ d + a + c > b ∧ d + b + c > a ∧ a + b + c > d)).card = 30 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_count_l775_77543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sequence_divisibility_l775_77547

-- Define the sequence a_n
def a (p : ℕ) : ℕ → ℕ
  | 0 => 2  -- Add case for 0
  | n+1 => a p n + (Int.ceil ((p * a p n : ℚ) / (n+1 : ℚ))).toNat

-- State the theorem
theorem prime_sequence_divisibility 
  (p : ℕ) 
  (hp : Nat.Prime p) 
  (hp2 : Nat.Prime (p + 2)) 
  (hp_gt_3 : p > 3) :
  ∀ n, 3 ≤ n → n < p → n ∣ (p * a p (n-1) + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sequence_divisibility_l775_77547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_l775_77510

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 4*x + 6 else x + 6

-- Define the solution set
def solution_set : Set ℝ :=
  {x : ℝ | -3 < x ∧ x < 1} ∪ {x : ℝ | x > 3}

-- Theorem statement
theorem f_inequality_solution_set :
  {x : ℝ | f x > f 1} = solution_set := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_l775_77510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l775_77536

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  (a - b + c) / c = b / (a + b - c) →
  a = 2 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a = 2 * Real.sin B * Real.sin C / Real.sin A →
  b = 2 * Real.sin C * Real.sin A / Real.sin B →
  c = 2 * Real.sin A * Real.sin B / Real.sin C →
  ∃ (area : ℝ), area ≤ Real.sqrt 3 ∧
    (∀ (area' : ℝ), 
      area' = 1/2 * a * b * Real.sin C → 
      area' ≤ area) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l775_77536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charlie_july_earnings_l775_77504

/-- Charlie's work hours and earnings in July --/
structure CharlieWork where
  first_week_hours : ℕ
  second_week_hours : ℕ
  wage_difference : ℚ
  hourly_wage : ℚ

/-- Calculate total earnings for two weeks --/
def total_earnings (work : CharlieWork) : ℚ :=
  work.hourly_wage * (work.first_week_hours + work.second_week_hours)

/-- Theorem stating Charlie's total earnings for the first two weeks of July --/
theorem charlie_july_earnings :
  ∀ (work : CharlieWork),
  work.first_week_hours = 20 →
  work.second_week_hours = 30 →
  work.wage_difference = 70 →
  work.hourly_wage * (work.second_week_hours - work.first_week_hours) = work.wage_difference →
  total_earnings work = 350 := by
  intro work h1 h2 h3 h4
  sorry

#check charlie_july_earnings

end NUMINAMATH_CALUDE_ERRORFEEDBACK_charlie_july_earnings_l775_77504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_sum_l775_77542

theorem cubic_root_sum (ω : ℂ) : 
  ω^3 = 1 ∧ ω.im ≠ 0 → (1 - ω + ω^2)^4 + (1 + ω - ω^2)^4 = -16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_sum_l775_77542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_volumes_and_ratio_l775_77559

/-- Regular tetrahedron with base edge length a and height m -/
structure RegularTetrahedron where
  a : ℝ
  m : ℝ
  h_positive : 0 < a ∧ 0 < m

/-- Volume of the inscribed cube with vertices on side edges -/
noncomputable def volume_cube_side_edges (t : RegularTetrahedron) : ℝ :=
  (t.a * t.m / (t.a + t.m)) ^ 3

/-- Volume of the inscribed cube with vertices on heights of side faces -/
noncomputable def volume_cube_side_heights (t : RegularTetrahedron) : ℝ :=
  (t.a * t.m / (t.a + Real.sqrt 2 * t.m)) ^ 3

/-- Theorem stating the volumes of inscribed cubes and their ratio when m = a -/
theorem inscribed_cube_volumes_and_ratio (t : RegularTetrahedron) :
  volume_cube_side_edges t = (t.a * t.m / (t.a + t.m)) ^ 3 ∧
  volume_cube_side_heights t = (t.a * t.m / (t.a + Real.sqrt 2 * t.m)) ^ 3 ∧
  (t.m = t.a → volume_cube_side_edges t / volume_cube_side_heights t = (7 + 5 * Real.sqrt 2) / 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_volumes_and_ratio_l775_77559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroids_not_collinear_l775_77526

/-- Represents a convex pentagon in 2D space -/
structure ConvexPentagon where
  vertices : Fin 5 → ℝ × ℝ
  convex : Convex ℝ (Set.range vertices)

/-- Represents a triangle formed by three points in 2D space -/
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

/-- Calculates the centroid of a triangle -/
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  (((t.a).1 + (t.b).1 + (t.c).1) / 3, ((t.a).2 + (t.b).2 + (t.c).2) / 3)

/-- Checks if three points are collinear -/
def collinear (p q r : ℝ × ℝ) : Prop :=
  (r.2 - p.2) * (q.1 - p.1) = (q.2 - p.2) * (r.1 - p.1)

/-- Divides a convex pentagon into three triangles using non-intersecting diagonals -/
noncomputable def divideIntoTriangles (p : ConvexPentagon) : Fin 3 → Triangle :=
  sorry

theorem centroids_not_collinear (p : ConvexPentagon) :
  let triangles := divideIntoTriangles p
  let centroids := λ i => centroid (triangles i)
  ¬ collinear (centroids 0) (centroids 1) (centroids 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroids_not_collinear_l775_77526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_changes_result_l775_77505

/-- The result of a series of percentage changes applied to an initial value. -/
noncomputable def percentage_changes (initial : ℝ) : ℝ :=
  let step1 := initial * (1 + 0.40)
  let step2 := step1 * (1 - Real.sqrt 0.25)
  let step3 := step2 * (1 + 0.15^2)
  step3

/-- Theorem stating that the series of percentage changes applied to 150 results in 107.3625. -/
theorem percentage_changes_result :
  percentage_changes 150 = 107.3625 := by
  sorry

-- Remove the #eval line as it's not computable
-- #eval percentage_changes 150

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_changes_result_l775_77505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hole_empty_time_is_60_l775_77549

/-- Represents the time (in hours) it takes for a pipe to fill a tank -/
noncomputable def pipeFillTime : ℝ := 15

/-- Represents the time (in hours) it takes for a pipe to fill a tank with a hole -/
noncomputable def pipeFillTimeWithHole : ℝ := 20

/-- Calculates the time (in hours) it takes for the hole to empty the full tank -/
noncomputable def holeEmptyTime : ℝ :=
  1 / (1 / pipeFillTime - 1 / pipeFillTimeWithHole)

theorem hole_empty_time_is_60 : holeEmptyTime = 60 := by
  -- Unfold the definition of holeEmptyTime
  unfold holeEmptyTime
  -- Simplify the expression
  simp [pipeFillTime, pipeFillTimeWithHole]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hole_empty_time_is_60_l775_77549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_exponential_and_not_odd_cosine_l775_77530

noncomputable def f (x : ℝ) := (2 : ℝ) ^ x
noncomputable def g (x : ℝ) := Real.cos x

def p : Prop := ∀ x y : ℝ, x < y → f x < f y
def q : Prop := ∀ x : ℝ, g (-x) = -g x

theorem increasing_exponential_and_not_odd_cosine : p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_exponential_and_not_odd_cosine_l775_77530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_weekly_revenue_loss_l775_77516

/-- Represents the daily tire production and sales scenario --/
structure TireProduction where
  daily_production : ℕ
  production_cost : ℕ
  selling_price_multiplier : ℚ
  potential_daily_sales : ℕ

/-- Calculates the weekly revenue loss due to production limitations --/
def weekly_revenue_loss (tp : TireProduction) : ℕ :=
  let selling_price := tp.production_cost * tp.selling_price_multiplier
  let additional_tires := tp.potential_daily_sales - tp.daily_production
  let daily_loss := (additional_tires : ℚ) * selling_price
  (7 * daily_loss).floor.toNat

/-- Theorem stating the weekly revenue loss for John's tire production --/
theorem john_weekly_revenue_loss :
  let john_production : TireProduction := {
    daily_production := 1000,
    production_cost := 250,
    selling_price_multiplier := 3/2,
    potential_daily_sales := 1200
  }
  weekly_revenue_loss john_production = 525000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_weekly_revenue_loss_l775_77516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeterAreaRatio_max_value_l775_77591

/-- An isosceles right triangle with area 50 square units -/
structure IsoscelesRightTriangle where
  a : ℝ
  area_eq : a ^ 2 / 2 = 50

/-- The ratio of perimeter to area for an isosceles right triangle -/
noncomputable def perimeterAreaRatio (t : IsoscelesRightTriangle) : ℝ :=
  (2 * t.a + t.a * Real.sqrt 2) / (2 * 50)

/-- The maximum value of the perimeter to area ratio is 0.4 + 0.2 * √2 -/
theorem perimeterAreaRatio_max_value :
  ∀ t : IsoscelesRightTriangle, perimeterAreaRatio t ≤ 0.4 + 0.2 * Real.sqrt 2 := by
  sorry

#check perimeterAreaRatio_max_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeterAreaRatio_max_value_l775_77591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_decreasing_interval_l775_77576

open Set Real

-- Define the interval where both sine and cosine are decreasing
def decreasing_interval (k : ℤ) : Set ℝ := Icc (2 * k * π + π / 2) (2 * k * π + π)

-- Define the property of a function being decreasing on an interval
def is_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x ≤ y → f y ≤ f x

-- Theorem statement
theorem sine_cosine_decreasing_interval (k : ℤ) :
  is_decreasing_on sin (decreasing_interval k) ∧
  is_decreasing_on cos (decreasing_interval k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_decreasing_interval_l775_77576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_roots_of_cubic_l775_77513

theorem integer_roots_of_cubic (a : ℤ) : 
  (∃ x : ℤ, x^3 + 2*x^2 + a*x + 11 = 0) ↔ a ∈ ({-158, -84, -14, 12} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_roots_of_cubic_l775_77513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moli_bought_seven_clips_l775_77585

/-- The cost of one ribbon -/
def R : ℚ := sorry

/-- The cost of one clip -/
def C : ℚ := sorry

/-- The cost of one soap -/
def S : ℚ := sorry

/-- The number of clips Moli bought initially -/
def x : ℕ := sorry

/-- First purchase equation -/
axiom first_purchase : 3 * R + x * C + S = 120

/-- Second purchase equation -/
axiom second_purchase : 4 * R + 10 * C + S = 164

/-- Individual costs equation -/
axiom individual_costs : R + C + S = 32

/-- Theorem: Moli bought 7 clips initially -/
theorem moli_bought_seven_clips : x = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_moli_bought_seven_clips_l775_77585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_arithmetic_1_complex_arithmetic_2_l775_77582

-- Define complex numbers
def z₁ : ℂ := -2 - 4*Complex.I
def z₂ : ℂ := 7 - 5*Complex.I
def z₃ : ℂ := 1 + 7*Complex.I

def w₁ : ℂ := 1 + Complex.I
def w₂ : ℂ := 2 + Complex.I
def w₃ : ℂ := 5 + Complex.I
def w₄ : ℂ := 1 - Complex.I

-- Theorem 1
theorem complex_arithmetic_1 : z₁ - z₂ + z₃ = -8 + 8*Complex.I := by sorry

-- Theorem 2
theorem complex_arithmetic_2 : w₁ * w₂ + w₃ / w₄ + w₄^2 = 3 + 4*Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_arithmetic_1_complex_arithmetic_2_l775_77582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l775_77561

-- Define sets A and B
def A : Set ℝ := {x | -3 < x ∧ x < 2}
def B : Set ℝ := {x | x^2 + 4*x - 5 ≤ 0}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = Set.Ioo (-3 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l775_77561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_triangle_l775_77574

theorem min_perimeter_triangle (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  A + B + C = Real.pi →                             -- Sum of angles in a triangle is π radians
  Real.sin A + Real.sin C = (Real.cos A + Real.cos C) * Real.sin B → -- Given trigonometric relation
  (1 / 2) * a * b * Real.sin C = 4 →                -- Area of the triangle is 4
  a > 0 → b > 0 → c > 0 →                           -- Positive side lengths
  c^2 = a^2 + b^2 →                                 -- Pythagorean theorem
  -- Conclusion
  a + b + c ≥ 4 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_triangle_l775_77574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_b_magnitude_l775_77563

-- Define the vector a
def a : ℝ × ℝ := (2, 4)

-- Define vector b as a variable
variable (b : ℝ × ℝ)

-- Theorem statement
theorem vector_b_magnitude
  (proj_a_on_b : ℝ)
  (dist_a_minus_b : ℝ)
  (h1 : proj_a_on_b = 3)
  (h2 : dist_a_minus_b = 3 * Real.sqrt 3)
  (h3 : proj_a_on_b = (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2))
  (h4 : dist_a_minus_b^2 = (a.1 - b.1)^2 + (a.2 - b.2)^2) :
  Real.sqrt (b.1^2 + b.2^2) = 7 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_b_magnitude_l775_77563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slip_4_5_in_cup_B_l775_77541

def slips : List ℝ := [1.5, 2, 2, 2.5, 2.5, 3, 3, 3.5, 3.5, 4, 4, 4.5, 5, 5.5, 6]

inductive Cup
| A | B | C | D | E
deriving Repr, DecidableEq

def cup_sums : List ℤ := [8, 9, 10, 11, 14]

def slip_in_cup (slip : ℝ) (cup : Cup) : Prop :=
  match cup with
  | Cup.A => slip = 2
  | Cup.C => slip = 3.5
  | _ => True

def Cup.toNat : Cup → Nat
  | Cup.A => 0
  | Cup.B => 1
  | Cup.C => 2
  | Cup.D => 3
  | Cup.E => 4

theorem slip_4_5_in_cup_B :
  ∀ (distribution : Cup → List ℝ),
    (∀ cup, List.sum (distribution cup) = cup_sums[Cup.toNat cup]!) →
    (∀ slip cup, slip ∈ distribution cup → slip ∈ slips) →
    (∀ slip cup, slip_in_cup slip cup → slip ∈ distribution cup) →
    (∀ slip, slip ∈ slips → ∃! cup, slip ∈ distribution cup) →
    4.5 ∈ distribution Cup.B :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slip_4_5_in_cup_B_l775_77541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l775_77521

/-- A function f with a cubic polynomial form -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 - a*x + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 4*x - a

/-- The property of f having exactly one extremum point in (-1, 1) -/
def has_one_extremum (a : ℝ) : Prop :=
  ∃! x, x ∈ Set.Ioo (-1) 1 ∧ f' a x = 0

/-- The theorem stating the range of a given the properties of f -/
theorem a_range (a : ℝ) (h : has_one_extremum a) : a ∈ Set.Ioo 0 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l775_77521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_shift_l775_77528

noncomputable def average (xs : List ℝ) : ℝ := xs.sum / xs.length

theorem average_shift (n : ℕ) (xs : List ℝ) (h : average xs = 2) :
  average (xs.map (· + 2)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_shift_l775_77528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_zero_sum_thirty_l775_77527

theorem determinant_zero_sum_thirty (x y : ℝ) : 
  x ≠ y →
  Matrix.det !![2, 5, 10; 4, x, y; 4, y, x] = 0 →
  x + y = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_zero_sum_thirty_l775_77527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_telephone_number_theorem_l775_77512

structure TelephoneNumber where
  A : Nat
  B : Nat
  C : Nat
  D : Nat
  E : Nat
  F : Nat
  G : Nat
  H : Nat
  I : Nat
  J : Nat

def valid_telephone_number (t : TelephoneNumber) : Prop :=
  -- Each letter corresponds to a unique digit
  t.A ≠ t.B ∧ t.A ≠ t.C ∧ t.A ≠ t.D ∧ t.A ≠ t.E ∧ t.A ≠ t.F ∧ t.A ≠ t.G ∧ t.A ≠ t.H ∧ t.A ≠ t.I ∧ t.A ≠ t.J ∧
  t.B ≠ t.C ∧ t.B ≠ t.D ∧ t.B ≠ t.E ∧ t.B ≠ t.F ∧ t.B ≠ t.G ∧ t.B ≠ t.H ∧ t.B ≠ t.I ∧ t.B ≠ t.J ∧
  t.C ≠ t.D ∧ t.C ≠ t.E ∧ t.C ≠ t.F ∧ t.C ≠ t.G ∧ t.C ≠ t.H ∧ t.C ≠ t.I ∧ t.C ≠ t.J ∧
  t.D ≠ t.E ∧ t.D ≠ t.F ∧ t.D ≠ t.G ∧ t.D ≠ t.H ∧ t.D ≠ t.I ∧ t.D ≠ t.J ∧
  t.E ≠ t.F ∧ t.E ≠ t.G ∧ t.E ≠ t.H ∧ t.E ≠ t.I ∧ t.E ≠ t.J ∧
  t.F ≠ t.G ∧ t.F ≠ t.H ∧ t.F ≠ t.I ∧ t.F ≠ t.J ∧
  t.G ≠ t.H ∧ t.G ≠ t.I ∧ t.G ≠ t.J ∧
  t.H ≠ t.I ∧ t.H ≠ t.J ∧
  t.I ≠ t.J ∧

  -- Digits arranged in decreasing order
  t.A > t.B ∧ t.B > t.C ∧
  t.D > t.E ∧ t.E > t.F ∧
  t.G > t.H ∧ t.H > t.I ∧ t.I > t.J ∧

  -- D, E, F are consecutive even digits
  t.D % 2 = 0 ∧ t.E % 2 = 0 ∧ t.F % 2 = 0 ∧
  t.E = t.D - 2 ∧ t.F = t.E - 2 ∧

  -- G, H, I, J are consecutive odd digits
  t.G % 2 = 1 ∧ t.H % 2 = 1 ∧ t.I % 2 = 1 ∧ t.J % 2 = 1 ∧
  t.H = t.G - 2 ∧ t.I = t.H - 2 ∧ t.J = t.I - 2 ∧

  -- Sum of digits in the first part
  t.A + t.B + t.C = 10

theorem telephone_number_theorem (t : TelephoneNumber) :
  valid_telephone_number t → t.A = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_telephone_number_theorem_l775_77512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l775_77589

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.cos ((21 * Real.pi / 4) - 2 * x)

theorem f_properties :
  (∀ x, f (5 * Real.pi / 8 + x) = f (5 * Real.pi / 8 - x)) ∧
  (∀ x, f (7 * Real.pi / 8 + x) = -f (7 * Real.pi / 8 - x)) ∧
  (Set.Icc (-Real.sqrt 2 / 4) (1/2) = Set.range f ∩ Set.Ioc (-Real.pi / 2) 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l775_77589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_time_theorem_l775_77540

/-- The time required for two people to complete a job together -/
noncomputable def combined_time (time_p time_q : ℝ) : ℝ :=
  1 / (1 / time_p + 1 / time_q)

/-- Theorem: Given two people who can complete a job in 4 and 6 hours respectively,
    the time required for them to complete the job together is 2.4 hours -/
theorem combined_time_theorem (time_p time_q : ℝ) 
    (h_p : time_p = 4)
    (h_q : time_q = 6) :
  combined_time time_p time_q = 2.4 := by
  sorry

/-- Compute the combined time for 4 and 6 hours -/
def example_combined_time : ℚ :=
  12/5

#eval example_combined_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_time_theorem_l775_77540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_arithmetic_inequality_l775_77599

theorem geometric_arithmetic_inequality (a b : ℕ → ℝ)
  (h_a_pos : ∀ n, a n > 0)
  (h_a_geo : ∀ n, a (n + 1) / a n = a 2 / a 1)
  (h_b_arith : ∀ n, b (n + 1) - b n = b 2 - b 1)
  (h_eq : a 6 = b 8) :
  a 3 + a 9 ≥ b 9 + b 7 := by
  sorry

#check geometric_arithmetic_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_arithmetic_inequality_l775_77599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perimeter_of_special_isosceles_triangle_l775_77520

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if a triangle is isosceles with AB = AC -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.b = t.c

/-- Calculates the semiperimeter of a triangle -/
def Triangle.semiperimeter (t : Triangle) : ℚ :=
  (t.a + t.b + t.c) / 2

/-- Calculates the inradius of a triangle -/
noncomputable def Triangle.inradius (t : Triangle) : ℝ :=
  let s := t.semiperimeter
  Real.sqrt ((s - t.a) * (s - t.b) * (s - t.c) / s)

/-- Calculates the radius of an excircle opposite to side a -/
def Triangle.exradius_a (t : Triangle) : ℚ :=
  let s := t.semiperimeter
  (s * (s - t.b) * (s - t.c)) / (s - t.a)

/-- Checks if the incircle is internally tangent to all excircles -/
noncomputable def Triangle.incircle_tangent_to_excircles (t : Triangle) : Prop :=
  let r := t.inradius
  let s := t.semiperimeter
  (r : ℝ) + (s - t.b) = (t.exradius_a : ℝ) ∧
  (r : ℝ) + (s - t.a) = ((s * (s - t.a) * (s - t.c)) / (s - t.b) : ℝ) ∧
  (r : ℝ) + (s - t.a) = ((s * (s - t.a) * (s - t.b)) / (s - t.c) : ℝ)

/-- The main theorem statement -/
theorem smallest_perimeter_of_special_isosceles_triangle :
  ∃ (t : Triangle), 
    t.isIsosceles ∧ 
    t.incircle_tangent_to_excircles ∧
    (∀ (t' : Triangle), t'.isIsosceles → t'.incircle_tangent_to_excircles → 
      t.a + t.b + t.c ≤ t'.a + t'.b + t'.c) ∧
    t.a + t.b + t.c = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perimeter_of_special_isosceles_triangle_l775_77520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_and_line_l775_77517

/-- An ellipse with the given properties -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0
  right_triangle : ∃ c : ℝ, b^2 = 3 * c^2 ∧ a^2 = b^2 + c^2
  point_on_ellipse : 1^2 / a^2 + (3/2)^2 / b^2 = 1

/-- The line intersecting the ellipse -/
structure IntersectingLine (E : SpecialEllipse) where
  k : ℝ
  m : ℝ
  intersects_ellipse : ∃ x y : ℝ, y = k * x + m ∧ x^2 / E.a^2 + y^2 / E.b^2 = 1
  pm_equals_mn : ∃ x y : ℝ, y = k * x + m ∧ x^2 / E.a^2 + y^2 / E.b^2 = 1 ∧
    (x - 0)^2 + (y - m)^2 = (0 - (-m/k))^2 + (m - 0)^2

/-- The theorem stating the properties of the ellipse and the line -/
theorem special_ellipse_and_line (E : SpecialEllipse) :
  E.a^2 = 4 ∧ E.b^2 = 3 ∧
  ∃ l : IntersectingLine E, (l.k = 1/2 ∨ l.k = -1/2) ∧ (l.m = Real.sqrt 21/7 ∨ l.m = -Real.sqrt 21/7) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_and_line_l775_77517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_l775_77506

/-- The function f(x) = (2x-3)/(7x+4) -/
noncomputable def f (x : ℝ) : ℝ := (2*x - 3) / (7*x + 4)

/-- The x-coordinate of the vertical asymptote -/
noncomputable def asymptote : ℝ := -4/7

theorem vertical_asymptote : 
  ∀ ε > (0 : ℝ), ∃ δ > (0 : ℝ), ∀ x : ℝ, 0 < |x - asymptote| ∧ |x - asymptote| < δ → |f x| > 1/ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_l775_77506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_purchase_theorem_l775_77548

/-- Represents the solution space for the goods purchase problem -/
inductive SolutionSpace
  | TwoSolutions
  | OneSolution
  | NoSolution

/-- Determines the solution space based on the value of 'a' -/
noncomputable def determineSolutionSpace (a : ℝ) : SolutionSpace :=
  if 0 < a ∧ a < 5 then SolutionSpace.TwoSolutions
  else if a = 5 then SolutionSpace.OneSolution
  else SolutionSpace.NoSolution

/-- Theorem stating the relationship between 'a' and the solution space -/
theorem goods_purchase_theorem (a : ℝ) (x y : ℝ) 
  (h1 : x - y = 1)
  (h2 : 45 / x - 20 / y = a)
  (h3 : 0 < a ∧ a ≤ 45) :
  determineSolutionSpace a = 
    if 0 < a ∧ a < 5 then SolutionSpace.TwoSolutions
    else if a = 5 then SolutionSpace.OneSolution
    else SolutionSpace.NoSolution :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_purchase_theorem_l775_77548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_eight_factors_twenty_four_has_eight_factors_twenty_four_is_smallest_l775_77503

def has_eight_factors (n : ℕ) : Prop := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 8

theorem smallest_with_eight_factors : 
  ∀ m : ℕ, m > 0 → has_eight_factors m → m ≥ 24 :=
by sorry

theorem twenty_four_has_eight_factors : has_eight_factors 24 :=
by sorry

theorem twenty_four_is_smallest : 
  ∀ m : ℕ, m > 0 → has_eight_factors m → m = 24 ∨ m > 24 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_eight_factors_twenty_four_has_eight_factors_twenty_four_is_smallest_l775_77503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l775_77567

/-- The time taken for a train to cross a telegraph post -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed : ℝ) : ℝ :=
  train_length / train_speed

/-- Theorem: A train of length 100 m traveling at 30.000000000000004 m/s takes approximately 3.33 seconds to cross a telegraph post -/
theorem train_crossing_theorem :
  let train_length : ℝ := 100
  let train_speed : ℝ := 30.000000000000004
  let crossing_time := train_crossing_time train_length train_speed
  abs (crossing_time - 3.33) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l775_77567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximized_at_90_l775_77537

/-- The profit function of a manufacturer -/
noncomputable def profit (x : ℝ) : ℝ := -1/3 * x^3 + 81 * x - 234

/-- The annual output that maximizes profit -/
def max_output : ℝ := 90

theorem profit_maximized_at_90 :
  ∀ x : ℝ, profit x ≤ profit max_output :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximized_at_90_l775_77537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_days_passed_is_five_l775_77546

/-- Represents the road construction project -/
structure RoadProject where
  totalLength : ℚ
  totalDays : ℚ
  initialWorkers : ℚ
  completedLength : ℚ
  additionalWorkers : ℚ

/-- Calculates the number of days passed when the engineer realized the progress -/
def daysPassed (project : RoadProject) : ℚ :=
  project.completedLength * project.totalDays / project.totalLength

/-- Theorem stating that the number of days passed is 5 for the given project conditions -/
theorem days_passed_is_five (project : RoadProject) 
  (h1 : project.totalLength = 10)
  (h2 : project.totalDays = 15)
  (h3 : project.initialWorkers = 30)
  (h4 : project.completedLength = 2)
  (h5 : project.additionalWorkers = 30) :
  daysPassed project = 5 := by
  sorry

def example_project : RoadProject := {
  totalLength := 10,
  totalDays := 15,
  initialWorkers := 30,
  completedLength := 2,
  additionalWorkers := 30
}

#eval daysPassed example_project

end NUMINAMATH_CALUDE_ERRORFEEDBACK_days_passed_is_five_l775_77546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_consecutive_heads_l775_77532

-- Define a fair coin
noncomputable def fairCoin : ℝ := 1 / 2

-- Define the number of flips
def numFlips : ℕ := 3

-- Theorem statement
theorem probability_three_consecutive_heads :
  (fairCoin ^ numFlips : ℝ) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_consecutive_heads_l775_77532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_with_parallel_centroid_incenter_line_equation_for_given_slopes_l775_77529

noncomputable section

-- Define the hyperbola C
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Define the foci
def left_focus : ℝ × ℝ := (-3, 0)
def right_focus : ℝ × ℝ := (3, 0)

-- Define the centroid of a triangle
def centroid (p1 p2 p3 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1 + p3.1) / 3, (p1.2 + p2.2 + p3.2) / 3)

-- Define the incenter of a triangle (simplified for this problem)
noncomputable def incenter (p1 p2 p3 : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define parallel lines
def parallel (p1 p2 q1 q2 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (q2.1 - q1.1) = (p2.1 - p1.1) * (q2.2 - q1.2)

-- Theorem 1
theorem exists_point_with_parallel_centroid_incenter :
  ∃ p : ℝ × ℝ, hyperbola p.1 p.2 ∧ p.2 > 0 ∧
  parallel (centroid p left_focus right_focus) (incenter p left_focus right_focus) left_focus right_focus :=
sorry

-- Define a line through a point with a given slope
def line_equation (p : ℝ × ℝ) (k : ℝ) (x : ℝ) : ℝ := k * (x - p.1) + p.2

-- Theorem 2
theorem line_equation_for_given_slopes :
  ∀ m n : ℝ × ℝ,
  hyperbola m.1 m.2 ∧ hyperbola n.1 n.2 ∧
  (m.2 / (m.1 + 2) + n.2 / (n.1 + 2) = -1/2) →
  ∃ k : ℝ, line_equation right_focus k = line_equation right_focus (-2) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_with_parallel_centroid_incenter_line_equation_for_given_slopes_l775_77529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l775_77571

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C := {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}
  let F := (Real.sqrt (a^2 + b^2), 0)
  let B := (0, b)
  let asymptote_slope := b / a
  (∃ (m : ℝ), (F.1 - B.1) / (F.2 - B.2) = -1 / m ∧ m = asymptote_slope) →
  Real.sqrt (F.1^2 + F.2^2) / a = (Real.sqrt 5 + 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l775_77571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_proof_l775_77587

/-- Represents the compound interest calculation for a half-yearly compounded investment --/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  principal * (1 + rate / 2) ^ (2 * periods)

/-- Proves that an investment of 3000 at 10% annual interest, compounded half-yearly, results in 3307.5 after one year --/
theorem investment_proof :
  let principal := 3000
  let annual_rate := 0.1
  let periods := 1
  let final_amount := 3307.5
  compound_interest principal annual_rate periods = final_amount := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_proof_l775_77587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l775_77577

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1

-- State the theorem
theorem function_property (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ 2 ∧ 0 ≤ x₂ ∧ x₂ ≤ 2 ∧
   f a x₁ = f a x₂ ∧ |x₁ - x₂| ≥ 1) →
  Real.exp 1 - 1 < a ∧ a < Real.exp 2 - Real.exp 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l775_77577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_to_rectangle_width_ratio_l775_77565

/-- The perimeter of both shapes -/
noncomputable def perimeter : ℝ := 60

/-- The side length of the equilateral triangle -/
noncomputable def triangle_side : ℝ := perimeter / 3

/-- The width of the rectangle -/
noncomputable def rectangle_width : ℝ := perimeter / 6

/-- The length of the rectangle -/
noncomputable def rectangle_length : ℝ := 2 * rectangle_width

/-- Theorem stating that the ratio of the triangle side to the rectangle width is 2 -/
theorem triangle_side_to_rectangle_width_ratio :
  triangle_side / rectangle_width = 2 := by
  -- Expand definitions
  unfold triangle_side rectangle_width
  -- Simplify the fraction
  simp [perimeter]
  -- The result follows from arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_to_rectangle_width_ratio_l775_77565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spatial_relationships_l775_77579

-- Define the types for lines and planes
structure Line : Type
structure Plane : Type

-- Define the relationships between lines and planes
axiom perpendicular : Line → Plane → Prop
axiom parallel : Line → Plane → Prop
axiom contained_in : Line → Plane → Prop
axiom perpendicular_planes : Plane → Plane → Prop
axiom parallel_planes : Plane → Plane → Prop
axiom perpendicular_lines : Line → Line → Prop

-- Define non-coincidence for lines and planes
axiom non_coincident_lines : Line → Line → Prop
axiom non_coincident_planes : Plane → Plane → Prop

-- Theorem statement
theorem spatial_relationships 
  (m n : Line) (α β : Plane) 
  (h_lines : non_coincident_lines m n)
  (h_planes : non_coincident_planes α β) :
  (∃ m α β, perpendicular_planes α β ∧ perpendicular m α ∧ ¬(parallel m β)) ∧
  (∀ m α β, perpendicular m α → perpendicular m β → parallel_planes α β) ∧
  (∀ m n α, parallel m α → perpendicular n α → perpendicular_lines m n) ∧
  (∃ m α β, parallel m α ∧ contained_in m β ∧ ¬(parallel_planes α β)) := by
  sorry

#check spatial_relationships

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spatial_relationships_l775_77579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_standard_equation_l775_77557

/-- A parabola with axis of symmetry y = -1, focus on the positive semi-axis of y,
    and distance 2 from the focus to the axis of symmetry has the standard equation x^2 = 4y -/
theorem parabola_standard_equation (p : Set (ℝ × ℝ)) 
  (h1 : ∀ (x y : ℝ), (x, y) ∈ p → y = -1 → x = 0)  -- axis of symmetry is y = -1
  (h2 : ∃ (a : ℝ), a > -1 ∧ (0, a) ∈ p)  -- focus is on positive semi-axis of y
  (h3 : ∃ (a : ℝ), a > -1 ∧ (0, a) ∈ p ∧ a - (-1) = 2)  -- distance from focus to axis of symmetry is 2
  : ∀ (x y : ℝ), (x, y) ∈ p ↔ x^2 = 4*y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_standard_equation_l775_77557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_factor_proof_l775_77560

theorem lcm_factor_proof (A B : ℕ) (X : ℕ) 
  (h1 : Nat.gcd A B = 15)
  (h2 : Nat.lcm A B = Nat.gcd A B * 11 * X)
  (h3 : A = 225) :
  X = 15 := by
  sorry

#check lcm_factor_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_factor_proof_l775_77560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_on_circle_l775_77596

/-- The point of intersection of two lines -/
noncomputable def intersection_point (a1 b1 c1 a2 b2 c2 : ℝ) : ℝ × ℝ :=
  let x := (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1)
  let y := (a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)
  (x, y)

/-- Distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: The intersection point of the given lines lies on the given circle -/
theorem intersection_on_circle : 
  let P := intersection_point 2 (-3) 4 3 (-2) 1
  let C := (2, 4)
  let R := Real.sqrt 5
  distance P C = R := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_on_circle_l775_77596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_hemisphere_volume_ratio_l775_77551

/-- The volume of a sphere with radius r -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The volume of a hemisphere with radius r -/
noncomputable def hemisphere_volume (r : ℝ) : ℝ := (1 / 2) * sphere_volume r

/-- The ratio of the volume of a sphere with radius 4a to the volume of a hemisphere with radius 3a -/
noncomputable def volume_ratio (a : ℝ) : ℝ := sphere_volume (4 * a) / hemisphere_volume (3 * a)

theorem sphere_hemisphere_volume_ratio (a : ℝ) : volume_ratio a = 128 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_hemisphere_volume_ratio_l775_77551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l775_77578

-- Define the type for a point in 2D space
def Point := ℝ × ℝ

-- Define the given endpoints
def endpoints : List Point := [(1, 6), (4, -3), (11, 6)]

-- Define the function to calculate distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the ellipse type
structure Ellipse where
  center : Point
  semi_major_axis : ℝ
  semi_minor_axis : ℝ

-- Define the function to construct an ellipse from the given endpoints
noncomputable def constructEllipse (eps : List Point) : Ellipse :=
  sorry -- Implementation details omitted

-- Define the function to calculate the distance between foci
noncomputable def focalDistance (e : Ellipse) : ℝ :=
  2 * Real.sqrt (e.semi_major_axis^2 - e.semi_minor_axis^2)

-- Theorem statement
theorem ellipse_focal_distance :
  let e := constructEllipse endpoints
  focalDistance e = 4 * Real.sqrt 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l775_77578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integer_sets_l775_77508

def is_valid_set (a n : ℕ) : Prop :=
  n ≥ 3 ∧ (n * (2 * a + n - 1)) / 2 = 150

theorem consecutive_integer_sets :
  ∃! k : ℕ, k > 0 ∧ 
  (∃ S : Finset (ℕ × ℕ), 
    S.card = k ∧ 
    (∀ p : ℕ × ℕ, p ∈ S → is_valid_set p.1 p.2) ∧
    (∀ a n, is_valid_set a n → (a, n) ∈ S)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integer_sets_l775_77508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_home_to_grocery_l775_77588

/-- The distance from Angelina's home to the grocery, in meters. -/
noncomputable def D : ℝ := sorry

/-- Angelina's speed from home to grocery, in meters per second. -/
noncomputable def v : ℝ := sorry

/-- The time Angelina spent walking from home to grocery, in seconds. -/
noncomputable def time_home_to_grocery : ℝ := D / v

/-- The time Angelina spent walking from grocery to gym, in seconds. -/
noncomputable def time_grocery_to_gym : ℝ := 300 / (2 * v)

/-- The theorem stating the distance from Angelina's home to the grocery. -/
theorem distance_home_to_grocery :
  (time_home_to_grocery = time_grocery_to_gym + 50) ∧
  (2 * v = 2) →
  D = 200 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_home_to_grocery_l775_77588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_complex_l775_77583

theorem modulus_of_complex : Complex.abs (3/4 - 3*Complex.I) = Real.sqrt 153 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_complex_l775_77583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_value_a_range_l775_77573

-- Define the equation
def equation (x a : Real) : Prop :=
  (Real.cos (x + Real.pi))^2 - Real.sin x + a = 0

-- Theorem 1: When x = 5π/6, a = -7/4
theorem solution_value :
  equation (5*Real.pi/6) (-7/4) := by sorry

-- Theorem 2: Range of a
theorem a_range (x a : Real) :
  equation x a → -5/4 ≤ a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_value_a_range_l775_77573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_deriv_2008_l775_77552

open Real

/-- Recursive definition of the nth derivative of cosine -/
noncomputable def f (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => cos
  | n + 1 => deriv (f n)

/-- Theorem: The 2008th derivative of cosine is equal to cosine -/
theorem cos_deriv_2008 : f 2008 = cos := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_deriv_2008_l775_77552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_l775_77570

/-- Given vectors a and b in ℝ², find lambda such that a + lambda*b is perpendicular to a -/
theorem perpendicular_vector (a b : ℝ × ℝ) (h1 : a = (3, -2)) (h2 : b = (1, 2)) :
  ∃ lambda : ℝ, (a.1 + lambda * b.1, a.2 + lambda * b.2) • a = 0 ∧ lambda = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_l775_77570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l775_77524

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x)

-- State the theorem
theorem inequality_theorem (n k : ℕ) (hn : n ≥ 2) (hk : k ≥ 1 ∧ k < n) :
  (∀ x ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioi 0, f x < x) ∧
  (Finset.sum (Finset.range k) (λ i ↦ (1 / n : ℝ) * f ((i + 1 : ℕ) / n)) <
   (1 + (k + 1) / n) * f ((k + 1) / n) - (k + 1) / n) ∧
  ((1 + (k + 1) / n) * f ((k + 1) / n) - (k + 1) / n ≤ 2 * Real.log 2 - 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l775_77524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l775_77580

/-- Definition of a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ
  equation : ℝ → ℝ → Prop

/-- Given two lines l₁ and l₂, prove that l₂ has the equation 4x - 3y + 9 = 0 -/
theorem line_equation_proof (l₁ l₂ : Line) : 
  (∃ θ : ℝ, l₁.slope = θ ∧ l₁.equation = fun x y ↦ x - 2*y - 2 = 0) →
  (l₂.slope = 2 * l₁.slope ∧ l₂.yIntercept = 3) →
  l₂.equation = fun x y ↦ 4*x - 3*y + 9 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l775_77580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_speed_minimizes_cost_l775_77545

/-- The daily cost function for a ship's operation -/
noncomputable def daily_cost (a k v : ℝ) : ℝ := a + k * v^3

/-- The distance covered by the ship in a day -/
noncomputable def daily_distance (v : ℝ) : ℝ := 24 * v

/-- The cost per kilometer function -/
noncomputable def cost_per_km (a k v : ℝ) : ℝ := daily_cost a k v / daily_distance v

/-- The most economical speed for ship operation -/
noncomputable def optimal_speed (a k : ℝ) : ℝ := (a / (2 * k))^(1/3)

theorem optimal_speed_minimizes_cost (a k : ℝ) (ha : a > 0) (hk : k > 0) :
  let v := optimal_speed a k
  ∀ u > 0, cost_per_km a k v ≤ cost_per_km a k u := by
  sorry

#check optimal_speed_minimizes_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_speed_minimizes_cost_l775_77545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l775_77553

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x + 2)^2 + (y - 3)^2 = 4

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-2, 3)

-- Theorem stating that the defined circle has the correct center and is tangent to the y-axis
theorem circle_properties :
  (∀ x y, circle_eq x y ↔ ((x - circle_center.1)^2 + (y - circle_center.2)^2 = 4)) ∧
  (∃ y, circle_eq 0 y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l775_77553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_perimeters_implies_regular_l775_77514

/-- A tetrahedron is a polyhedron with four faces. -/
structure Tetrahedron where
  edges : Fin 6 → ℝ
  faces : Fin 4 → Fin 3 → Fin 6

/-- The perimeter of a face is the sum of its edge lengths. -/
def face_perimeter (t : Tetrahedron) (f : Fin 4) : ℝ :=
  Finset.sum (Finset.univ : Finset (Fin 3)) (λ i => t.edges (t.faces f i))

/-- A tetrahedron has equal face perimeters if all face perimeters are equal. -/
def has_equal_face_perimeters (t : Tetrahedron) : Prop :=
  ∀ f₁ f₂ : Fin 4, face_perimeter t f₁ = face_perimeter t f₂

/-- A tetrahedron is regular if all its edges have the same length. -/
def is_regular (t : Tetrahedron) : Prop :=
  ∀ e₁ e₂ : Fin 6, t.edges e₁ = t.edges e₂

/-- 
Theorem: If a tetrahedron has equal face perimeters, then it is regular.
-/
theorem equal_perimeters_implies_regular (t : Tetrahedron) 
  (h : has_equal_face_perimeters t) : is_regular t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_perimeters_implies_regular_l775_77514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_handshake_count_e_handshakes_determined_l775_77562

/-- Represents a person in the group --/
inductive Person : Type
| a | b | c | d | e

/-- The number of handshakes for each person --/
def handshakes : Person → Nat
| Person.a => 4
| Person.b => 1
| Person.c => 3
| Person.d => 2
| Person.e => 2  -- This is what we want to prove

/-- The total number of handshakes in the group --/
def total_handshakes : Nat :=
  handshakes Person.a + handshakes Person.b + handshakes Person.c + handshakes Person.d + handshakes Person.e

/-- Each handshake is counted twice in the total --/
theorem handshake_count : total_handshakes = 12 := by sorry

/-- The number of handshakes for person e is uniquely determined --/
theorem e_handshakes_determined : 
  ∃! n : Nat, handshakes Person.e = n ∧ 
  total_handshakes = (handshakes Person.a + handshakes Person.b + 
                      handshakes Person.c + handshakes Person.d + n) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_handshake_count_e_handshakes_determined_l775_77562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_triangle_area_l775_77501

open Real

-- Define the triangle ABC and its orthocenter H
variable (A B C H : EuclideanSpace ℝ (Fin 2))

-- Define the distances from orthocenter to vertices
def AH_distance : ℝ := 2
def BH_distance : ℝ := 12
def CH_distance : ℝ := 9

-- Define the properties of the triangle
def is_acute_triangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry
def is_orthocenter (H A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the area of a triangle
noncomputable def triangle_area (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- Theorem statement
theorem orthocenter_triangle_area 
  (h_acute : is_acute_triangle A B C)
  (h_orthocenter : is_orthocenter H A B C)
  (h_AH : ‖A - H‖ = AH_distance)
  (h_BH : ‖B - H‖ = BH_distance)
  (h_CH : ‖C - H‖ = CH_distance) :
  triangle_area A B C = 7 * sqrt 63 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_triangle_area_l775_77501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_a_l775_77592

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x)

theorem triangle_side_a (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  f (A / 2) = 2 →
  b = 1 →
  c = 2 →
  a = Real.sqrt 7 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_a_l775_77592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_theorem_l775_77594

-- Define the given parameters
variable (l : ℝ) -- length of the diagonal
variable (α : ℝ) -- angle between the plane and axial section
variable (β : ℝ) -- angle between diagonal and base plane

-- Define the volume of the cylinder
noncomputable def cylinder_volume (l α β : ℝ) : ℝ :=
  (Real.pi * l^3 * (Real.cos β)^2 * Real.sin β) / (4 * (Real.cos α)^2)

-- State the theorem
theorem cylinder_volume_theorem (l α β : ℝ) 
  (h1 : 0 < l) 
  (h2 : 0 < α ∧ α < Real.pi/2) 
  (h3 : 0 < β ∧ β < Real.pi/2) : 
  ∃ V, V = cylinder_volume l α β := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_theorem_l775_77594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_range_a_l775_77590

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x - 1 else Real.log x / Real.log a

-- State the theorem
theorem monotonic_f_range_a :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) → 2 < a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_range_a_l775_77590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_sheet_distance_l775_77564

/-- Represents a rectangular sheet of paper with a white front and black back -/
structure Sheet where
  width : ℝ
  length : ℝ
  area : ℝ

/-- The distance of point A from its original position when the sheet is folded -/
noncomputable def foldedDistance (s : Sheet) : ℝ :=
  Real.sqrt 14

/-- Theorem stating the properties of the folded sheet -/
theorem folded_sheet_distance (s : Sheet) 
  (area_cond : s.area = 12)
  (length_cond : s.length = 2 * s.width)
  (fold_cond : ∃ x : ℝ, (1/2) * x^2 = s.area - (1/2) * x^2) :
  foldedDistance s = Real.sqrt 14 := by
  sorry

#check folded_sheet_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_sheet_distance_l775_77564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mikes_salary_l775_77509

/-- Calculates Mike's annual salary given the house cost, downpayment percentage, 
    savings percentage, and years to save. -/
noncomputable def calculate_salary (house_cost : ℝ) (downpayment_percent : ℝ) 
                     (savings_percent : ℝ) (years : ℝ) : ℝ :=
  (house_cost * downpayment_percent) / (years * savings_percent)

/-- Theorem stating that Mike's annual salary is $150,000 given the problem conditions. -/
theorem mikes_salary : 
  let house_cost : ℝ := 450000
  let downpayment_percent : ℝ := 0.20
  let savings_percent : ℝ := 0.10
  let years : ℝ := 6
  calculate_salary house_cost downpayment_percent savings_percent years = 150000 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_salary 450000 0.20 0.10 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mikes_salary_l775_77509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_sum_simplification_l775_77500

open Real

theorem trig_sum_simplification : 
  (sin (10 * π / 180) + sin (20 * π / 180) + sin (30 * π / 180) + sin (40 * π / 180) + 
   sin (50 * π / 180) + sin (60 * π / 180) + sin (70 * π / 180) + sin (80 * π / 180)) / 
  (cos (5 * π / 180) * cos (10 * π / 180) * cos (20 * π / 180)) = 4 * sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_sum_simplification_l775_77500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_assignRoles_eq_twentyfour_l775_77523

/-- The number of ways to assign 4 distinct roles to 4 distinct people -/
def assignRoles : ℕ :=
  Fintype.card (Equiv.Perm (Fin 4))

/-- Theorem stating that the number of ways to assign 4 distinct roles to 4 distinct people is 24 -/
theorem assignRoles_eq_twentyfour : assignRoles = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_assignRoles_eq_twentyfour_l775_77523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_permutation_decimals_l775_77558

-- Define auxiliary functions
def period_length (x : ℚ) : ℕ := sorry

def decimal_expansion (x : ℚ) : List ℕ := sorry

def cyclic_permutation (k : ℕ) (l : List ℕ) : List ℕ := sorry

-- Main theorem
theorem cyclic_permutation_decimals (p : ℕ) (h_prime : Prime p) 
  (h_period : period_length (1 / p) = p - 1) :
  ∀ a : ℕ, 1 ≤ a → a < p → 
    ∃ k : ℕ, decimal_expansion (a / p) = cyclic_permutation k (decimal_expansion (1 / p)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_permutation_decimals_l775_77558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_pairs_l775_77584

def interior_angle (n : ℕ) : ℚ := 180 - 360 / n

theorem regular_polygon_pairs : 
  ∃! (pairs : List (ℕ × ℕ)), 
    (∀ p ∈ pairs, p.1 > 5 ∧ p.2 > 5) ∧
    (∀ p ∈ pairs, interior_angle p.1 / interior_angle p.2 = 4 / 3) ∧
    pairs.length = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_pairs_l775_77584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l775_77534

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1  -- Add a case for 0 to make the function total
  | 1 => 1
  | n + 2 => 2/3 * sequence_a (n + 1) + 1

theorem sequence_a_formula (n : ℕ) (h : n ≥ 1) : 
  sequence_a n = 3 - 2 * (2/3)^(n-1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l775_77534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_repeating_digits_of_seven_thirteenths_l775_77572

theorem least_repeating_digits_of_seven_thirteenths :
  let f : ℚ := 7 / 13
  ∃ (n : ℕ), n > 0 ∧ (∀ m : ℕ, m > 0 → (10^n * f - ⌊10^n * f⌋ = 10^(n+m) * f - ⌊10^(n+m) * f⌋)) ∧
    (∀ k : ℕ, 0 < k ∧ k < n → (10^k * f - ⌊10^k * f⌋ ≠ 10^n * f - ⌊10^n * f⌋)) ∧
    n = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_repeating_digits_of_seven_thirteenths_l775_77572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_two_l775_77586

/-- The function f(x) = ln(√(1+x²) - x) + 1 -/
noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x^2) - x) + 1

/-- lg3 is the logarithm base 10 of 3 -/
noncomputable def lg3 : ℝ := Real.log 3 / Real.log 10

/-- The statement to prove -/
theorem f_sum_equals_two : f lg3 + f (-lg3) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_two_l775_77586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l775_77531

noncomputable section

variable (a : ℝ)
variable (m : ℝ)

def f (x : ℝ) : ℝ := x^2 * (x - a)

def g (x : ℝ) : ℝ := f a x + m / (x - 1)

def tangent_parallel_to_x_axis (a : ℝ) : Prop :=
  (deriv (f a)) 2 = 0

def g_increasing_on_3_to_inf (a m : ℝ) : Prop :=
  ∀ x y, 3 ≤ x ∧ x < y → g a m x < g a m y

theorem problem_solution :
  a > 0 →
  tangent_parallel_to_x_axis a →
  g_increasing_on_3_to_inf a m →
  (a = 3 ∧
   (∀ x ∈ Set.Icc 0 2, f a x ≥ (if 0 < a ∧ a < 3 then -4/27 * a^3 else 8 - 4*a)) ∧
   m ≤ 36) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l775_77531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_implies_relation_l775_77535

/-- The set of positive real numbers -/
def PositiveReals : Type := {x : ℝ // x > 0}

/-- The functional equation condition -/
def SatisfiesEquation (f g : PositiveReals → PositiveReals) : Prop :=
  ∀ x y : PositiveReals, f ⟨(g x).val * y.val + (f x).val, sorry⟩ = ⟨(y.val + 2015) * (f x).val, sorry⟩

/-- The theorem to be proved -/
theorem functional_equation_implies_relation
  (f g : PositiveReals → PositiveReals)
  (h : SatisfiesEquation f g) :
  ∀ x : PositiveReals, f x = ⟨2015 * (g x).val, sorry⟩ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_implies_relation_l775_77535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_domain_equivalent_to_quadratic_positive_l775_77515

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x + 6) / Real.sqrt (x^2 - 5*x + 6)

-- Define IsValidArg for real-valued functions
def IsValidArg (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ y, f x = y

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | IsValidArg f x} = {x : ℝ | x < 2 ∨ x > 3} := by sorry

-- State that the domain is equivalent to the set where x^2 - 5x + 6 > 0
theorem domain_equivalent_to_quadratic_positive :
  {x : ℝ | IsValidArg f x} = {x : ℝ | x^2 - 5*x + 6 > 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_domain_equivalent_to_quadratic_positive_l775_77515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_time_in_storm_car_storm_average_time_l775_77518

/-- Represents the position of an object in 2D space -/
structure Position where
  x : ℝ
  y : ℝ

/-- Represents a moving object -/
structure MovingObject where
  initialPosition : Position
  velocity : Position

/-- Represents a circular storm -/
structure Storm extends MovingObject where
  radius : ℝ

/-- Calculates the position of a moving object at time t -/
def position (obj : MovingObject) (t : ℝ) : Position :=
  { x := obj.initialPosition.x + obj.velocity.x * t,
    y := obj.initialPosition.y + obj.velocity.y * t }

/-- Calculates the distance between two positions -/
noncomputable def distance (p1 p2 : Position) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Checks if a position is inside the storm -/
def isInsideStorm (stormPosition : Position) (carPosition : Position) (stormRadius : ℝ) : Prop :=
  distance stormPosition carPosition ≤ stormRadius

/-- Theorem: The average time between when the car enters and exits the storm is 90 minutes -/
theorem average_time_in_storm (car : MovingObject) (storm : Storm) : ℝ := by
  -- Assuming car starts at (0, 0) and moves east at 1 mile per minute
  have h1 : car.initialPosition = { x := 0, y := 0 } := by sorry
  have h2 : car.velocity = { x := 1, y := 0 } := by sorry
  
  -- Assuming storm starts at (0, 90) and moves southeast at 1 mile per minute
  have h3 : storm.initialPosition = { x := 0, y := 90 } := by sorry
  have h4 : storm.velocity = { x := 1, y := -1 } := by sorry
  have h5 : storm.radius = 30 := by sorry

  -- Prove that the average time is 90 minutes
  sorry

/-- The main theorem stating the result -/
theorem car_storm_average_time : ℝ := by
  -- Define the car and storm
  let car : MovingObject := {
    initialPosition := { x := 0, y := 0 },
    velocity := { x := 1, y := 0 }
  }
  let storm : Storm := {
    initialPosition := { x := 0, y := 90 },
    velocity := { x := 1, y := -1 },
    radius := 30
  }
  
  -- Apply the average_time_in_storm theorem
  exact average_time_in_storm car storm

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_time_in_storm_car_storm_average_time_l775_77518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_and_sum_l775_77598

-- Define the arithmetic sequence a_n
noncomputable def a (n : ℕ) : ℝ := 2 * n + 10

-- Define the sequence b_n
noncomputable def b (n : ℕ) : ℝ := 2^(a n - 10)

-- Define the sum of the first n terms of b_n
noncomputable def T (n : ℕ) : ℝ := 4 * (4^n - 1) / 3

theorem arithmetic_sequence_and_sum :
  (a 10 = 30) ∧ (a 20 = 50) →
  (∀ n : ℕ, a n = 2 * n + 10) ∧
  (∀ n : ℕ, T n = 4 * (4^n - 1) / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_and_sum_l775_77598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_APBQ_l775_77575

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9
def C₂ (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1

-- Define the trajectory E
def E (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1 ∧ x ≠ -2

-- Define the fixed points P and Q
def P : ℝ × ℝ := (-2, 0)
def Q : ℝ × ℝ := (2, 0)

-- Define the line l passing through (1,0)
def l (m : ℝ) (x y : ℝ) : Prop := x = m * y + 1

-- Helper function to calculate the area of a quadrilateral
-- This is a placeholder and should be properly defined
noncomputable def area_quadrilateral (P Q A B : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem max_area_APBQ : 
  ∀ m : ℝ, 
  let A := λ (x y : ℝ) ↦ E x y ∧ l m x y
  let B := λ (x y : ℝ) ↦ E x y ∧ l m x y ∧ (x, y) ≠ (m + 1, 0)
  (∃ x₁ y₁, A x₁ y₁) → 
  (∃ x₂ y₂, B x₂ y₂) → 
  (∀ x y, E x y → (∃ r : ℝ, r > 0 ∧ 
    (∀ x' y', C₁ x' y' → (x - x')^2 + (y - y')^2 = (3 - r)^2) ∧
    (∀ x' y', C₂ x' y' → (x - x')^2 + (y - y')^2 = (1 + r)^2))) →
  (∀ S : ℝ, S = area_quadrilateral P Q (m + 1, 0) (m + 1, 0) → S ≤ 6) ∧ 
  (∃ S : ℝ, S = area_quadrilateral P Q (m + 1, 0) (m + 1, 0) ∧ S = 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_APBQ_l775_77575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_power_four_l775_77550

open Matrix

variable {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)

theorem det_power_four (h : Matrix.det A = 3) : Matrix.det (A^4) = 81 := by
  rw [Matrix.det_pow]
  rw [h]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_power_four_l775_77550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_l775_77569

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r + r)^2

theorem circle_tangency (r : ℝ) (hr : r > 0) :
  externally_tangent (0, 0) (3, -1) r → r = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_l775_77569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l775_77555

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^4 + b^4 + (a + b)^(-4 : ℤ) ≥ Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l775_77555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_monotone_iff_a_ge_one_l775_77502

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (3 * Real.log (x + 2) - Real.log (x - 2)) / 2

-- Define the function F
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := a * Real.log (x - 1) - f x

-- State the theorem
theorem F_monotone_iff_a_ge_one :
  ∀ a : ℝ, (∀ x y : ℝ, 2 < x ∧ x < y → F a x ≤ F a y) ↔ a ≥ 1 := by
  sorry

#check F_monotone_iff_a_ge_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_monotone_iff_a_ge_one_l775_77502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_inequality_solution_set_l775_77556

open Set
open Real

theorem tan_inequality_solution_set :
  let S := {x : ℝ | tan x > -1}
  ∀ k : ℤ, ∃ A : Set ℝ,
    A = Ioo (k * π - π/4) (k * π + π/2) ∧
    S = ⋃ (k : ℤ), Ioo (k * π - π/4) (k * π + π/2) :=
by
  intro S k
  use Ioo (k * π - π/4) (k * π + π/2)
  constructor
  · rfl
  · sorry  -- Proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_inequality_solution_set_l775_77556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_size_equals_selected_l775_77593

/-- Represents the sample size of a study -/
def sample_size : ℕ := sorry

/-- Represents the number of students selected for the study -/
def selected_students : ℕ := sorry

/-- The number of students selected for this study is 100 -/
axiom students_count : selected_students = 100

/-- The sample size is equal to the number of selected students -/
theorem sample_size_equals_selected : sample_size = selected_students := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_size_equals_selected_l775_77593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l775_77544

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 * x - 4)

-- State the theorem
theorem tangent_line_equation :
  let x₀ : ℝ := 4
  let y₀ : ℝ := f x₀
  let m : ℝ := 1 / Real.sqrt (2 * x₀ - 4)
  (λ x y => y = m * (x - x₀) + y₀) = (λ x y => y = 1/2 * x - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l775_77544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_height_l775_77511

-- Define the volume in cubic centimeters
noncomputable def volume_cm : ℝ := 1380000000

-- Define the base area in square meters
noncomputable def base_area_m : ℝ := 115

-- Define the conversion factor from cubic centimeters to cubic meters
noncomputable def cm3_to_m3 : ℝ := 1 / 1000000

-- Theorem statement
theorem cuboid_height :
  let volume_m := volume_cm * cm3_to_m3
  let height := volume_m / base_area_m
  height = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_height_l775_77511
