import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l534_53459

noncomputable def f (x : ℝ) := Real.sin (2 * x) - 2 * (Real.sin x) ^ 2

theorem f_properties :
  ∃ (T : ℝ),
    (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), 0 < T' ∧ T' < T → ∃ (x : ℝ), f (x + T') ≠ f x) ∧
    T = Real.pi ∧
    (∀ (x : ℝ), f x ≤ 0) ∧
    (∀ (k : ℤ), f (Real.pi / 4 + k * Real.pi) = 0) ∧
    (∀ (x : ℝ), f x = 0 → ∃ (k : ℤ), x = Real.pi / 4 + k * Real.pi) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l534_53459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_man_time_l534_53447

/-- Calculates the time (in seconds) for a train to pass a person moving in the opposite direction. -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed : ℝ) (person_speed : ℝ) : ℝ :=
  let relative_speed := train_speed + person_speed
  let relative_speed_mps := relative_speed * (1000 / 3600)
  train_length / relative_speed_mps

/-- The time for a 110-meter long train moving at 60 kmph to pass a man moving at 6 kmph
    in the opposite direction is approximately 6 seconds. -/
theorem train_passing_man_time :
  ∃ ε > 0, |train_passing_time 110 60 6 - 6| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_man_time_l534_53447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_curve_l534_53472

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 + (y + 2)^2 = 1

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem min_distance_to_curve (P : ℝ × ℝ) :
  ∃ (min_dist : ℝ), ∀ (Q : ℝ × ℝ), curve Q.1 Q.2 → distance P Q ≥ min_dist := by
  sorry

#check min_distance_to_curve

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_curve_l534_53472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2α_minus_π_4_l534_53432

theorem cos_2α_minus_π_4 (α : Real) (h1 : Real.tan α + 1 / Real.tan α = 5/2) (h2 : 0 < α ∧ α < π/4) :
  Real.cos (2 * α - π/4) = 7 * Real.sqrt 2 / 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2α_minus_π_4_l534_53432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_curve_intersection_l534_53448

-- Define the curve C
noncomputable def C (t : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos (2 * t), 2 * Real.sin t)

-- Define the line l in polar form
def l (ρ θ m : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 3) + m = 0

-- Theorem: Cartesian equation of l and intersection condition
theorem line_and_curve_intersection (m : ℝ) :
  (∀ x y, (∃ θ ρ, l ρ θ m ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ↔ Real.sqrt 3 * x + y + 2 * m = 0) ∧
  (∃ t, ∃ x y, C t = (x, y) ∧ Real.sqrt 3 * x + y + 2 * m = 0) ↔ 
  (m ≥ -19/12 ∧ m ≤ 5/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_curve_intersection_l534_53448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l534_53444

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin (Real.pi + x) - Real.sqrt 3 * Real.cos x * Real.sin (2 * x)) / (2 * Real.cos (Real.pi - x)) - 1/2

theorem f_properties :
  ∃ (T : ℝ),
    (∀ x, f (x + T) = f x) ∧
    (T > 0) ∧
    (∀ S, S > 0 → (∀ x, f (x + S) = f x) → T ≤ S) ∧
    (StrictMonoOn f (Set.Icc (Real.pi/3) (Real.pi/2))) ∧
    (StrictMonoOn f (Set.Icc (Real.pi/2) (5*Real.pi/6))) ∧
    (∀ x ∈ Set.Ioo 0 (Real.pi/2), f x ≤ 1) ∧
    (f (Real.pi/3) = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l534_53444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_theorem_l534_53405

/-- The number of days X needs to finish the remaining work -/
noncomputable def days_needed_by_x (x_days : ℝ) (y_days : ℝ) (y_worked : ℝ) : ℝ :=
  (x_days * (y_days - y_worked)) / y_days

theorem work_completion_theorem (x_days y_days y_worked : ℝ) 
  (hx : x_days = 18)
  (hy : y_days = 15)
  (hw : y_worked = 10) :
  days_needed_by_x x_days y_days y_worked = 6 := by
  sorry

#check work_completion_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_theorem_l534_53405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_max_volume_l534_53497

/-- The height that maximizes the volume of a cone with slant height 30 cm -/
noncomputable def max_volume_height : ℝ := 10 * Real.sqrt 3

/-- The slant height of the cone -/
def slant_height : ℝ := 30

/-- Theorem stating that the volume is maximized at the calculated height -/
theorem cone_max_volume :
  ∀ h : ℝ, h > 0 → h ≤ slant_height →
  (1/3 * Real.pi * (slant_height^2 - h^2) * h) ≤ (1/3 * Real.pi * (slant_height^2 - max_volume_height^2) * max_volume_height) :=
by
  sorry

#check cone_max_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_max_volume_l534_53497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_problem_solution_l534_53488

/-- Represents the number of days it takes for a worker to complete the entire job. -/
abbrev WorkDays := ℚ

/-- Represents the fraction of work completed in one day. -/
abbrev WorkRate := ℚ

/-- The problem setup -/
structure WorkProblem where
  a_total_days : WorkDays  -- Days for A to complete the entire work
  b_total_days : WorkDays  -- Days for B to complete the entire work
  b_actual_days : WorkDays -- Days B actually worked to complete the remaining work
  total_work : ℚ           -- Represents the entire work as 1

/-- Calculate the work rate given the total days to complete the job -/
def calculate_work_rate (days : WorkDays) : WorkRate :=
  1 / days

/-- The main theorem to prove -/
theorem work_problem_solution (p : WorkProblem) 
  (h1 : p.a_total_days = 15)
  (h2 : p.b_total_days = 9)
  (h3 : p.b_actual_days = 6)
  (h4 : p.total_work = 1) :
  ∃ (x : ℚ), 
    x * calculate_work_rate p.a_total_days + 
    p.b_actual_days * calculate_work_rate p.b_total_days = p.total_work ∧ 
    x = 5 := by
  sorry

#eval calculate_work_rate 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_problem_solution_l534_53488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_proof_l534_53400

-- Define the circle
def myCircle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

-- Define the line
def myLine (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x + Real.sqrt 3

-- Theorem statement
theorem tangent_line_proof :
  ∃ (x₀ y₀ : ℝ),
    -- The line passes through (0, √3)
    myLine 0 (Real.sqrt 3) ∧
    -- The line touches the circle at exactly one point (x₀, y₀)
    myCircle x₀ y₀ ∧
    myLine x₀ y₀ ∧
    -- For any other point on the line, it's outside the circle
    ∀ (x y : ℝ), x ≠ x₀ ∧ myLine x y → ¬ myCircle x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_proof_l534_53400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_in_specific_prism_l534_53454

/-- The maximum radius of a sphere that can be placed inside a right triangular prism container -/
noncomputable def max_sphere_radius (h : ℝ) (base_edge : ℝ) : ℝ :=
  base_edge / (2 * Real.sqrt 3)

/-- Theorem: The maximum radius of a sphere in a right triangular prism with height 5 and base edge 4√3 is 2 -/
theorem max_sphere_radius_in_specific_prism :
  max_sphere_radius 5 (4 * Real.sqrt 3) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_in_specific_prism_l534_53454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_intersection_points_l534_53441

/-- The circle equation -/
def circleEq (x y a : ℝ) : Prop := x^2 + y^2 = a^2

/-- The ellipse equation -/
def ellipseEq (x y a : ℝ) : Prop := y = x^2 / 2 - a

/-- The number of intersection points between the circle and ellipse -/
noncomputable def intersectionPoints (a : ℝ) : ℕ := sorry

/-- Theorem stating the condition for exactly four intersection points -/
theorem four_intersection_points (a : ℝ) :
  intersectionPoints a = 4 ↔ a > 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_intersection_points_l534_53441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_variance_specific_set_l534_53481

noncomputable def sineVariance (a : List ℝ) (a₀ : ℝ) : ℝ :=
  (a.map (fun x => Real.sin (x - a₀) ^ 2)).sum / a.length

theorem sine_variance_specific_set (a₀ : ℝ) :
  sineVariance [π / 2, 5 * π / 6, 7 * π / 6] a₀ = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_variance_specific_set_l534_53481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_verbose_island_words_l534_53460

def alphabet_size : ℕ := 26
def max_word_length : ℕ := 5

def words_with_a (n : ℕ) : ℕ :=
  alphabet_size^n - (alphabet_size - 1)^n

def total_words_with_a : ℕ :=
  (List.range max_word_length).map (λ i ↦ words_with_a (i + 1)) |>.sum

theorem verbose_island_words :
  total_words_with_a = 2202115 := by
  sorry

#eval total_words_with_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_verbose_island_words_l534_53460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l534_53483

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

-- State the theorem
theorem tangent_line_at_zero (x y : ℝ) : 
  (∀ ε > 0, ∃ δ > 0, ∀ h, -δ < h ∧ h < δ → 
    |f h - (f 0 + h)| ≤ ε * |h|) →
  y = x ↔ y - f 0 = (x - 0) * (Real.exp 0 * (Real.sin 0 + Real.cos 0)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l534_53483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l534_53442

noncomputable def g (x : ℝ) : ℝ := 
  (Real.arcsin (x / 3))^2 - Real.pi * Real.arccos (x / 3) + (Real.arccos (x / 3))^2 + (Real.pi^2 / 18) * (x^2 - 3*x + 9)

theorem g_range : 
  ∀ x ∈ Set.Icc (-3 : ℝ) 3, g x ∈ Set.Icc (Real.pi^2 / 4) ((3 * Real.pi^2) / 2) ∧
  ∀ y ∈ Set.Icc (Real.pi^2 / 4) ((3 * Real.pi^2) / 2), ∃ x ∈ Set.Icc (-3 : ℝ) 3, g x = y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l534_53442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_3cos_values_l534_53427

theorem sin_plus_3cos_values (x : ℝ) (h : Real.cos x - 3 * Real.sin x = 2) :
  (Real.sin x + 3 * Real.cos x = (2 * Real.sqrt 6 - 3) / 5) ∨
  (Real.sin x + 3 * Real.cos x = -(2 * Real.sqrt 6 + 3) / 5) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_3cos_values_l534_53427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_m_range_l534_53451

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then -x^2 + 2*m*x - 2 else 1 + Real.log x

-- State the theorem
theorem increasing_function_m_range (m : ℝ) :
  (∀ x y : ℝ, x < y → f m x < f m y) →
  1 ≤ m ∧ m ≤ 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_m_range_l534_53451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_distribution_theorem_l534_53445

/-- Represents the state of card distribution -/
structure CardState (n : ℕ) where
  points : Fin n → ℕ
  center : ℕ

/-- Operation A: Move cards from a point to adjacent points and center -/
def operationA (n : ℕ) (state : CardState n) (i : Fin n) : CardState n :=
  sorry

/-- Operation B: Move cards from center to all points -/
def operationB (n : ℕ) (state : CardState n) : CardState n :=
  sorry

/-- Check if all points have at least n+1 cards -/
def allPointsHaveEnoughCards (n : ℕ) (state : CardState n) : Prop :=
  ∀ i : Fin n, state.points i ≥ n + 1 ∧ state.center ≥ n + 1

/-- Sum of cards in the state -/
def totalCards (n : ℕ) (state : CardState n) : ℕ :=
  (Finset.univ.sum (state.points)) + state.center

/-- The main theorem -/
theorem card_distribution_theorem (n : ℕ) (h : n ≥ 3) 
  (initial_state : CardState n) 
  (total_cards : totalCards n initial_state ≥ n^2 + 3*n + 1) :
  ∃ (final_state : CardState n), allPointsHaveEnoughCards n final_state := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_distribution_theorem_l534_53445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l534_53425

theorem trigonometric_problem (α β : Real) 
  (h1 : 0 < β) (h2 : β < α) (h3 : α < π / 2)
  (h4 : Real.tan α = 4 * Real.sqrt 3)
  (h5 : Real.cos (α - β) = 13 / 14) :
  Real.sin (2 * α) = 8 * Real.sqrt 3 / 49 ∧ β = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l534_53425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_surface_area_from_volume_l534_53415

/-- The volume of a regular tetrahedron -/
noncomputable def tetrahedron_volume (a : ℝ) : ℝ := (Real.sqrt 2 / 12) * a^3

/-- The surface area of a regular tetrahedron -/
noncomputable def tetrahedron_surface_area (a : ℝ) : ℝ := Real.sqrt 3 * a^2

/-- Theorem: If the volume of a regular tetrahedron is 9 dm³, then its surface area is 18√3 dm² -/
theorem tetrahedron_surface_area_from_volume :
  ∃ (a : ℝ), tetrahedron_volume a = 9 ∧ tetrahedron_surface_area a = 18 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_surface_area_from_volume_l534_53415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cole_drive_time_to_work_l534_53456

noncomputable def drive_time_to_work (speed_to_work : ℝ) (speed_back_home : ℝ) (total_round_trip_time : ℝ) : ℝ :=
  let distance := (total_round_trip_time * speed_to_work * speed_back_home) / (speed_to_work + speed_back_home)
  distance / speed_to_work * 60

theorem cole_drive_time_to_work :
  drive_time_to_work 30 90 2 = 90 := by
  sorry

-- Remove the #eval line as it's not computable
-- #eval drive_time_to_work 30 90 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cole_drive_time_to_work_l534_53456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l534_53406

theorem unique_solution_exponential_equation :
  ∃! p : ℝ × ℝ, (32 : ℝ)^(p.1^2 + p.2) + (32 : ℝ)^(p.1 + p.2^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l534_53406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cloud_height_theorem_l534_53429

/-- The height of a cloud above an observer, given tower height, elevation angle, and depression angle -/
noncomputable def cloud_height (m : ℝ) (α β : ℝ) : ℝ :=
  m + (2 * m * Real.cos β * Real.sin α) / Real.sin (β - α)

/-- Theorem stating the height of the cloud above the observer -/
theorem cloud_height_theorem (m : ℝ) (α β : ℝ)
  (h_m : m > 0)
  (h_α : 0 < α ∧ α < π/2)
  (h_β : 0 < β ∧ β < π/2)
  (h_αβ : α < β) :
  ∃ (h : ℝ), h > m ∧ 
  (Real.tan α) / (Real.tan β) = (h - m) / (h + m) ∧
  h = cloud_height m α β :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cloud_height_theorem_l534_53429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l534_53407

/-- Calculates the time for two trains to cross each other on a bridge. -/
theorem train_crossing_time (bridge_length : ℝ) (train_a_length train_a_speed train_b_length train_b_speed : ℝ)
  (h1 : bridge_length = 1200)
  (h2 : train_a_length = 300)
  (h3 : train_a_speed = 54)
  (h4 : train_b_length = 250)
  (h5 : train_b_speed = 45) :
  let total_distance := bridge_length + train_a_length + train_b_length
  let speed_a_ms := train_a_speed * (1000 / 3600)
  let speed_b_ms := train_b_speed * (1000 / 3600)
  let relative_speed := speed_a_ms + speed_b_ms
  let crossing_time := total_distance / relative_speed
  ∃ (ε : ℝ), ε > 0 ∧ |crossing_time - 63.64| < ε := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l534_53407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_k_value_l534_53433

/-- Given three points A, B, and C in the plane, this function checks if they are collinear -/
def are_collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, (B.1 - A.1, B.2 - A.2) = t • (C.1 - A.1, C.2 - A.2)

/-- Theorem stating that if A(3,1), B(-2,k), and C(8,11) are collinear, then k = -9 -/
theorem collinear_points_k_value :
  ∀ k : ℝ, are_collinear (3, 1) (-2, k) (8, 11) → k = -9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_k_value_l534_53433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_elements_sum_of_A_intersect_B_l534_53404

def A : Set ℕ := {n : ℕ | ∃ k : ℕ, n = k * (k + 1) ∧ k > 0}

def B : Set ℕ := {n : ℕ | ∃ m : ℕ, n = m * (m + 1) * (m + 2) ∧ m > 0}

theorem smallest_elements_sum_of_A_intersect_B : 
  ∃ (x y : ℕ), x ∈ A ∩ B ∧ y ∈ A ∩ B ∧ x < y ∧
  ∀ z ∈ A ∩ B, z = x ∨ z ≥ y ∧
  x + y = 216 := by
  sorry

#check smallest_elements_sum_of_A_intersect_B

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_elements_sum_of_A_intersect_B_l534_53404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_books_available_end_of_week_l534_53424

/-- Represents the daily changes in book inventory --/
structure DailyChanges where
  checkedOut : ℕ
  returned : ℕ
  newBooks : ℕ
  damaged : ℕ
  misplaced : ℕ

/-- Calculates the number of books available at the end of the week --/
def booksAvailableAtEndOfWeek (initialBooks : ℕ) (monday : DailyChanges) (tuesday : DailyChanges) 
  (wednesday : DailyChanges) (thursday : DailyChanges) (friday : DailyChanges) : ℕ :=
  let mondayEnd := initialBooks - monday.checkedOut + monday.returned
  let tuesdayEnd := mondayEnd - tuesday.checkedOut + tuesday.returned + tuesday.newBooks
  let wednesdayEnd := tuesdayEnd - wednesday.checkedOut + wednesday.returned - wednesday.damaged
  let thursdayEnd := wednesdayEnd - thursday.checkedOut + thursday.returned
  thursdayEnd - friday.checkedOut + friday.returned - friday.misplaced

/-- Theorem stating the number of books available at the end of the week --/
theorem books_available_end_of_week :
  booksAvailableAtEndOfWeek 98
    { checkedOut := 43, returned := 23, newBooks := 0, damaged := 0, misplaced := 0 }
    { checkedOut := 28, returned := 0, newBooks := 35, damaged := 0, misplaced := 0 }
    { checkedOut := 0, returned := 15, newBooks := 0, damaged := 3, misplaced := 0 }
    { checkedOut := 37, returned := 8, newBooks := 0, damaged := 0, misplaced := 0 }
    { checkedOut := 29, returned := 7, newBooks := 0, damaged := 0, misplaced := 4 } = 42 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_books_available_end_of_week_l534_53424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_max_area_l534_53493

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the point A
def point_A (a : ℝ) : ℝ × ℝ := (0, a)

-- Define the condition for point P
def point_P_condition (B C P : ℝ × ℝ) (lambda : ℝ) : Prop :=
  ∃ (xB yB xC yC x y : ℝ),
    B = (xB, yB) ∧ C = (xC, yC) ∧ P = (x, y) ∧
    x - xB = lambda * (xC - x) ∧
    y - yB = lambda * (yC - y)

-- Define the arithmetic sequence condition for reciprocals of x-coordinates
def arithmetic_sequence_condition (xB xP xC : ℝ) : Prop :=
  2 / xP = 1 / xB + 1 / xC

-- Main theorem
theorem trajectory_and_max_area 
  (a : ℝ) 
  (B C P : ℝ × ℝ) 
  (lambda : ℝ) 
  (h1 : circle_M B.1 B.2)
  (h2 : circle_M C.1 C.2)
  (h3 : point_P_condition B C P lambda)
  (h4 : arithmetic_sequence_condition B.1 P.1 C.1) :
  (∃ (x y : ℝ), P = (x, y) ∧ 2*x - a*y - 3 = 0) ∧
  (∃ (S : ℝ), S = Real.sqrt 3/4 ∧ 
    ∀ (R : ℝ), R ≤ S ∧ 
    (R = S → a = 0) ∧
    (∃ (x1 y1 x2 y2 : ℝ),
      circle_M x1 y1 ∧ circle_M x2 y2 ∧
      2*x1 - a*y1 - 3 = 0 ∧ 2*x2 - a*y2 - 3 = 0 ∧
      R = (1/2) * Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_max_area_l534_53493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_intersection_and_point_l534_53491

noncomputable def intersection_point (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : ℝ × ℝ :=
  let x := (b₁ * c₂ - b₂ * c₁) / (a₁ * b₂ - a₂ * b₁)
  let y := (a₂ * c₁ - a₁ * c₂) / (a₁ * b₂ - a₂ * b₁)
  (x, y)

def point_on_line (x y a b c : ℝ) : Prop :=
  a * x + b * y + c = 0

def line_equation (x₁ y₁ x₂ y₂ : ℝ) : ℝ → ℝ → ℝ → Prop :=
  λ a b c ↦ a * (y₂ - y₁) = b * (x₁ - x₂) ∧ c = a * x₁ + b * y₁

theorem line_through_intersection_and_point :
  let (x₀, y₀) := intersection_point 7 5 (-24) 1 (-1) 0
  line_equation x₀ y₀ 5 1 1 3 (-8) ∧
  point_on_line 5 1 1 3 (-8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_intersection_and_point_l534_53491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_implies_a_range_l534_53422

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then (x - a)^2
  else x + 1/x + a + 4

-- State the theorem
theorem f_minimum_implies_a_range (a : ℝ) :
  (∀ x : ℝ, f a x ≥ f a 0) → 0 ≤ a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_implies_a_range_l534_53422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_range_of_a_l534_53428

-- Define the sets A, B, and M
def A : Set ℝ := {x | 4 ≤ (2 : ℝ)^x ∧ (2 : ℝ)^x < 128}
def B : Set ℝ := {x | 1 < x ∧ x ≤ 6}
def M (a : ℝ) : Set ℝ := {x | a - 3 < x ∧ x < a + 3}

-- Statement 1: A ∩ ¬B = {x | 6 < x < 7}
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x : ℝ | 6 < x ∧ x < 7} := by sorry

-- Statement 2: If M ∪ ¬B = ℝ, then 3 < a ≤ 4
theorem range_of_a (a : ℝ) :
  M a ∪ (Set.univ \ B) = Set.univ → 3 < a ∧ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_range_of_a_l534_53428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_theorem_max_distance_theorem_l534_53461

-- Define the circles
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the tangent line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 4)

-- Define the distance between points A and B
noncomputable def distance_AB (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem circle_tangency_theorem :
  ∀ (x y : ℝ),
  (∃ (R : ℝ), R > 0 ∧
    (∀ (xp yp : ℝ), (xp - x)^2 + (yp - y)^2 = R^2 →
      (∃ (xm ym : ℝ), circle_M xm ym ∧ (xp - xm)^2 + (yp - ym)^2 = (R + 1)^2) ∧
      (∃ (xn yn : ℝ), circle_N xn yn ∧ (xp - xn)^2 + (yp - yn)^2 = (3 - R)^2))) →
  curve_C x y :=
by sorry

theorem max_distance_theorem :
  ∃ (k x1 y1 x2 y2 : ℝ),
  line_l k x1 y1 ∧ line_l k x2 y2 ∧
  curve_C x1 y1 ∧ curve_C x2 y2 ∧
  (∀ (x y R : ℝ), (x - 2)^2 + y^2 = 4 → R ≤ 2) →
  distance_AB x1 y1 x2 y2 = 18 / 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_theorem_max_distance_theorem_l534_53461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_covered_l534_53479

/-- The velocity function of the object (in m/s) -/
def velocity (t : ℝ) : ℝ := 3 * t^2 + t

/-- The theorem stating that the distance covered is 72 meters -/
theorem distance_covered : ∫ t in (0 : ℝ)..(4 : ℝ), velocity t = 72 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_covered_l534_53479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_sequence_anomaly_l534_53499

def cubic_sequence : List ℤ := [1, 9, 35, 99, 225, 441, 784, 1296]

def is_cubic_progression (seq : List ℤ) : Prop :=
  seq.length ≥ 4 ∧
  ∃ (a b c d : ℚ), ∀ (i : Fin seq.length),
    seq.get i = (a * (i.val : ℚ)^3 + b * (i.val : ℚ)^2 + c * (i.val : ℚ) + d).floor

theorem cubic_sequence_anomaly :
  ¬ is_cubic_progression cubic_sequence := by
  sorry

#eval cubic_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_sequence_anomaly_l534_53499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l534_53498

theorem problem_solution (x : ℝ) : 
  let p := x^2 - 3*x - 4 ≠ 0
  let q := ∃ (n : ℕ), x = n ∧ n > 0
  (¬(p ∧ q)) → (¬¬q) → x = 4 := by
  intro hp hq
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l534_53498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_symmetry_square_symmetry_angle_symmetry_l534_53412

-- Define the concept of an axis of symmetry
def axis_of_symmetry : Type → ℕ := λ _ => 0

-- Define isosceles trapezoid
structure IsoscelesTrapezoid :=
  (axes : ℕ)

-- Define square
structure Square :=
  (axes : ℕ)

-- Define angle
structure Angle :=
  (bisector : ℕ)
  (axis : ℕ)

-- Theorem statements
theorem isosceles_trapezoid_symmetry (t : IsoscelesTrapezoid) : t.axes = 1 := by sorry

theorem square_symmetry (s : Square) : s.axes = 4 := by sorry

theorem angle_symmetry (a : Angle) : a.axis = a.bisector := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_symmetry_square_symmetry_angle_symmetry_l534_53412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_21_l534_53453

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The sum of even-indexed terms from 2 to 20 is 10. -/
def sum_even_terms_10 (a : ℕ → ℝ) : Prop :=
  (Finset.range 10).sum (λ i => a (2 * i + 2)) = 10

/-- The sum of the first 21 terms of the sequence. -/
def sum_21_terms (a : ℕ → ℝ) : ℝ :=
  (Finset.range 21).sum a

theorem arithmetic_sequence_sum_21 (a : ℕ → ℝ) 
  (h1 : is_arithmetic_sequence a) 
  (h2 : sum_even_terms_10 a) : 
  sum_21_terms a = 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_21_l534_53453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l534_53421

/-- A structure representing a triangle in a 2D space --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  noncollinear : ¬(∃ (t : ℝ), (1 - t) • A + t • B = C)

/-- The distance between two points in 2D space --/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The triangle inequality theorem --/
theorem triangle_inequality (t : Triangle) :
  distance t.A t.B + distance t.B t.C > distance t.A t.C ∧
  distance t.A t.B + distance t.A t.C > distance t.B t.C ∧
  distance t.B t.C + distance t.A t.C > distance t.A t.B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l534_53421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_is_pi_over_six_l534_53478

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the line l
noncomputable def line_l (x₀ θ t : ℝ) : ℝ × ℝ := (x₀ + t * Real.cos θ, t * Real.sin θ)

-- Define the left focus of the ellipse
noncomputable def left_focus : ℝ × ℝ := (-Real.sqrt 2, 0)

-- Define point C on the y-axis
noncomputable def point_C (θ : ℝ) : ℝ × ℝ := (0, Real.sqrt 2 / Real.cos θ * Real.sin θ)

-- State the theorem
theorem line_slope_angle_is_pi_over_six :
  ∃ (x₀ θ : ℝ) (A B : ℝ × ℝ),
    -- Line l passes through the left focus
    line_l x₀ θ (Real.sqrt 2 / Real.cos θ) = left_focus
    -- A and B are on the ellipse and line l
    ∧ ellipse A.1 A.2
    ∧ ellipse B.1 B.2
    ∧ (∃ (t_A t_B : ℝ), line_l x₀ θ t_A = A ∧ line_l x₀ θ t_B = B)
    -- A is above C
    ∧ A.2 > (point_C θ).2
    -- |F₁B| = |AC|
    ∧ Real.sqrt ((B.1 - left_focus.1)^2 + (B.2 - left_focus.2)^2) =
       Real.sqrt ((A.1 - (point_C θ).1)^2 + (A.2 - (point_C θ).2)^2)
    -- The slope angle is π/6
    → θ = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_is_pi_over_six_l534_53478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_value_l534_53474

theorem definite_integral_value : ∫ x in (0:ℝ)..(1:ℝ), (2 * x + Real.exp x) = Real.exp 1 - 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_value_l534_53474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_monotone_decreasing_omega_range_l534_53471

theorem sin_monotone_decreasing_omega_range (ω : ℝ) (h1 : ω > 0) :
  (∀ x ∈ Set.Icc (π / 3) (π / 2), 
    ∀ y ∈ Set.Icc (π / 3) (π / 2), 
    x ≤ y → Real.sin (ω * x) ≥ Real.sin (ω * y)) →
  3 / 2 ≤ ω ∧ ω ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_monotone_decreasing_omega_range_l534_53471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moving_circle_trajectory_l534_53484

-- Define the circles O and C
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 1

-- Define the moving circle M
structure MovingCircle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangency conditions
def externally_tangent (M : MovingCircle) : Prop :=
  let (x, y) := M.center
  Real.sqrt (x^2 + y^2) = M.radius + 1

def internally_tangent (M : MovingCircle) : Prop :=
  let (x, y) := M.center
  Real.sqrt ((x - 3)^2 + y^2) = M.radius - 1

-- Theorem statement
theorem moving_circle_trajectory :
  ∀ (M : MovingCircle),
    externally_tangent M →
    internally_tangent M →
    ∃ (a b : ℝ), 
      let (x, y) := M.center
      (x^2 / a^2) - (y^2 / b^2) = 1 ∧ x > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_moving_circle_trajectory_l534_53484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_a_value_l534_53450

noncomputable section

variable (f g : ℝ → ℝ) (a : ℝ)

theorem prove_a_value (h1 : ∀ x, f x = a^x * g x)
                      (h2 : a > 0)
                      (h3 : a ≠ 1)
                      (h4 : ∀ x, g x ≠ 0)
                      (h5 : ∀ x, (deriv f x) * g x < f x * (deriv g x))
                      (h6 : f 1 / g 1 + f (-1) / g (-1) = 5/2) :
  a = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_a_value_l534_53450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_quadrilateral_area_l534_53431

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : (a^2 - b^2) / a^2 = 1/2  -- eccentricity² = 1/2

/-- A point on the ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The quadrilateral formed by points M, B₂, N, B₁ -/
def Quadrilateral (e : Ellipse) (m : PointOnEllipse e) :=
  {n : ℝ × ℝ // 
    (n.1 + m.x) * m.y = 0 ∧  -- NB₁ ⊥ MB₁
    (n.1 - m.x) * m.y = 0    -- NB₂ ⊥ MB₂
  }

/-- The area of the quadrilateral -/
noncomputable def QuadrilateralArea (e : Ellipse) (m : PointOnEllipse e) : ℝ :=
  3 * (|m.x| + |(m.y^2 - 9) / m.x|)

/-- The theorem statement -/
theorem max_quadrilateral_area (e : Ellipse) :
  e.b = 3 →
  ∃ (m : PointOnEllipse e), ∀ (m' : PointOnEllipse e),
    QuadrilateralArea e m ≥ QuadrilateralArea e m' ∧
    QuadrilateralArea e m = 27 * Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_quadrilateral_area_l534_53431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_formulas_l534_53417

/-- Given a triangle ABC with sides a, b, c, semi-perimeter p, circumradius R,
    and angles α, β, γ opposite to sides a, b, c respectively,
    the length of the angle bisector l_a of angle A can be expressed
    by four equivalent formulas. -/
theorem angle_bisector_formulas (a b c p R α β γ : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_tri : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_p : p = (a + b + c) / 2)
  (h_angles : α + β + γ = π)
  (h_sine_law : R = a / (2 * Real.sin α)) : 
  let l_a := Real.sqrt (4 * p * (p - a) * b * c / ((b + c)^2))
  l_a = 2 * b * c * Real.cos (α / 2) / (b + c) ∧
  l_a = 2 * R * Real.sin β * Real.sin γ / Real.cos ((β - γ) / 2) ∧
  l_a = 4 * p * Real.sin (β / 2) * Real.sin (γ / 2) / (Real.sin β + Real.sin γ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_formulas_l534_53417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_speed_interval_l534_53466

-- Define the time intervals and distances
def intervals : List (Nat × Nat) := [(0, 30), (30, 60), (60, 90), (90, 120)]
def distances : List ℝ := [45, 135, 255, 325]

-- Function to calculate average speed for an interval
noncomputable def averageSpeed (start_distance : ℝ) (end_distance : ℝ) (time_interval : ℝ) : ℝ :=
  (end_distance - start_distance) / time_interval

-- Theorem stating that the 60-90 minute interval has the highest average speed
theorem highest_speed_interval :
  let speeds := List.zipWith (λ i d ↦ averageSpeed (List.get! distances i) (List.get! distances (i + 1)) 30)
                 (List.range 3)
                 (List.take 3 distances)
  ∀ i ∈ List.range 4, i ≠ 2 → List.get! speeds 2 > List.get! speeds i :=
by sorry

#check highest_speed_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_speed_interval_l534_53466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_less_than_10_l534_53443

-- Define the set of positive factors of 90
def factors_of_90 : Set Nat := {n : Nat | n > 0 ∧ 90 % n = 0}

-- Define the set of factors less than 10
def factors_less_than_10 : Set Nat := {n : Nat | n ∈ factors_of_90 ∧ n < 10}

-- State the theorem
theorem probability_factor_less_than_10 : 
  (Finset.card (Finset.filter (λ n => n < 10) (Finset.filter (λ n => 90 % n = 0) (Finset.range 91)))) / 
  (Finset.card (Finset.filter (λ n => 90 % n = 0) (Finset.range 91))) = 1 / 2 := by
  sorry

#eval Finset.card (Finset.filter (λ n => n < 10) (Finset.filter (λ n => 90 % n = 0) (Finset.range 91)))
#eval Finset.card (Finset.filter (λ n => 90 % n = 0) (Finset.range 91))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_less_than_10_l534_53443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l534_53408

def mySequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (a 1 = 1) ∧ 
  (∀ n, a n ^ 2 - (2 * a (n + 1) - 1) * a n - 2 * a (n + 1) = 0)

theorem sequence_formula (a : ℕ → ℝ) (h : mySequence a) : 
  ∀ n : ℕ, a n = (1 : ℝ) / (2 ^ (n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l534_53408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_parabola_intersection_l534_53480

/-- The circle equation x^2 + y^2 = 4 -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The parabola equation y = x^2 - bx -/
def parabola_eq (x y b : ℝ) : Prop := y = x^2 - b*x

/-- The number of intersection points between the circle and parabola -/
noncomputable def intersection_count (b : ℝ) : ℕ := sorry

/-- The theorem stating that the circle and parabola intersect at exactly two points if and only if b = 2 -/
theorem circle_parabola_intersection :
  ∀ b : ℝ, intersection_count b = 2 ↔ b = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_parabola_intersection_l534_53480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_diameter_l534_53465

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  area : ℝ

-- Define the given conditions
noncomputable def triangle_ABC : Triangle where
  a := 1
  B := Real.pi / 4  -- 45° in radians
  area := 2
  b := 5  -- We now know this value from the solution
  c := 4 * Real.sqrt 2  -- We now know this value from the solution
  A := sorry  -- We don't know these values, but they're part of the structure
  C := sorry

-- Theorem statement
theorem circumcircle_diameter (t : Triangle) (h1 : t = triangle_ABC) :
  2 * t.area / (t.a * Real.sin t.B) = 5 * Real.sqrt 2 := by
  sorry

-- Note: The formula for the diameter of the circumcircle is 2R, where R is the radius.
-- The radius can be calculated as abc / (4S), where S is the area.
-- This is equivalent to the formula used in the solution: b / sin B.

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_diameter_l534_53465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hotel_charges_difference_l534_53455

noncomputable section

-- Define room charges relative to Hotel R
def standard_R : ℝ := 100
def deluxe_R : ℝ := 100
def premium_R : ℝ := 100

-- Define room charges for Hotel P relative to Hotel R
def standard_P : ℝ := standard_R * 0.3
def deluxe_P : ℝ := deluxe_R * 0.4
def premium_P : ℝ := premium_R * 0.5

-- Define room charges for Hotel G relative to Hotel P
def standard_G : ℝ := standard_P / 0.9
def deluxe_G : ℝ := deluxe_P / 0.8
def premium_G : ℝ := premium_P / 0.7

-- Define discounts and taxes
def discount_P : ℝ := 0.15
def discount_R : ℝ := 0.10
def tax_G : ℝ := 0.05

-- Calculate total charges for Hotel R with discounts
def total_R : ℝ := 
  standard_R * (1 - discount_R) + 
  deluxe_R * (1 - discount_R) + 
  premium_R

-- Calculate total charges for Hotel G with taxes
def total_G : ℝ := 
  (standard_G + deluxe_G + premium_G) * (1 + tax_G)

-- Calculate percent difference
def percent_difference : ℝ := 
  (abs (total_R - total_G) / ((total_R + total_G) / 2)) * 100

theorem hotel_charges_difference : 
  abs (percent_difference - 58.49) < 0.01 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hotel_charges_difference_l534_53455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_after_nine_years_l534_53467

/-- Represents the price of a product that decreases every 3 years -/
noncomputable def price_after_years (initial_price : ℝ) (years : ℕ) : ℝ :=
  initial_price * (1/3)^(years / 3)

/-- Theorem stating that the price after 9 years will be 2400 yuan -/
theorem price_after_nine_years (initial_price : ℝ) (h : initial_price = 8100) :
  price_after_years initial_price 9 = 2400 := by
  sorry

#check price_after_nine_years

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_after_nine_years_l534_53467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olympiad_team_partition_l534_53469

/-- Represents a school in the olympiad -/
structure School where
  id : Nat

/-- Represents a student participating in the olympiad -/
structure Student where
  id : Nat
  school : School

/-- Represents a team of students -/
structure Team where
  students : Finset Student
  valid : students.card = 3

/-- Predicate to check if a team is valid according to the problem statement -/
def ValidTeam (t : Team) : Prop :=
  (∀ s₁ s₂, s₁ ∈ t.students → s₂ ∈ t.students → s₁.school = s₂.school) ∨ 
  (∀ s₁ s₂, s₁ ∈ t.students → s₂ ∈ t.students → s₁ ≠ s₂ → s₁.school ≠ s₂.school)

/-- The main theorem statement -/
theorem olympiad_team_partition :
  ∀ (students : Finset Student) (schools : Finset School),
  students.card = 300 →
  schools.card ≥ 4 →
  (∀ s, s ∈ students → ∃ sc, sc ∈ schools ∧ s.school = sc) →
  ∃ (teams : Finset Team),
    (∀ t, t ∈ teams → ValidTeam t) ∧
    (∀ s, s ∈ students → ∃! t, t ∈ teams ∧ s ∈ t.students) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_olympiad_team_partition_l534_53469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_final_direction_l534_53434

-- Define the directions
inductive Direction
| North
| East
| South
| West

-- Define the rotation
def rotate (d : Direction) (revolutions : ℚ) : Direction :=
  match (revolutions % 1 : ℚ).num.toNat with
  | 0 => d
  | 1 => match d with
    | Direction.North => Direction.East
    | Direction.East => Direction.South
    | Direction.South => Direction.West
    | Direction.West => Direction.North
  | 2 => match d with
    | Direction.North => Direction.South
    | Direction.East => Direction.West
    | Direction.South => Direction.North
    | Direction.West => Direction.East
  | 3 => match d with
    | Direction.North => Direction.West
    | Direction.East => Direction.North
    | Direction.South => Direction.East
    | Direction.West => Direction.South
  | _ => d  -- This case should not occur in our problem

-- Define the theorem
theorem spinner_final_direction :
  let initial_direction := Direction.South
  let clockwise_revolutions : ℚ := 7/2
  let counterclockwise_revolutions : ℚ := 25/4
  let final_direction := rotate (rotate initial_direction clockwise_revolutions) (-counterclockwise_revolutions)
  final_direction = Direction.West := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_final_direction_l534_53434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sue_payment_l534_53487

noncomputable def total_cost : ℝ := 2100
def sister_days : ℕ := 4
def days_in_week : ℕ := 7

def sue_days : ℕ := days_in_week - sister_days

noncomputable def sue_percentage : ℝ := (sue_days : ℝ) / (days_in_week : ℝ)

noncomputable def sue_cost : ℝ := total_cost * sue_percentage

theorem sue_payment : sue_cost = 900 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sue_payment_l534_53487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_with_two_rational_points_l534_53418

-- Define a rational point
def rational_point (p : ℚ × ℚ) : Prop := True

-- Define a line passing through a point
def line_through_point (p : ℝ × ℝ) : Type := 
  {l : Set (ℝ × ℝ) // p ∈ l ∧ ∃ (m b : ℝ), ∀ (x y : ℝ), (x, y) ∈ l ↔ y = m * x + b}

-- State the theorem
theorem unique_line_with_two_rational_points (a : ℝ) (h : ¬ ∃ (q : ℚ), (q : ℝ) = a) :
  ∃! (l : line_through_point (a, 0)), ∃ (p q : ℚ × ℚ), p ≠ q ∧ 
    ((↑p.1 : ℝ), (↑p.2 : ℝ)) ∈ l.val ∧ ((↑q.1 : ℝ), (↑q.2 : ℝ)) ∈ l.val ∧ 
    rational_point p ∧ rational_point q :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_with_two_rational_points_l534_53418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_exponential_inequality_l534_53419

theorem solution_set_exponential_inequality :
  {x : ℝ | (2 : ℝ)^(x^2 - 2*x - 2) < 2} = {x : ℝ | -1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_exponential_inequality_l534_53419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_remaining_area_l534_53468

/-- Represents a rectangle with side lengths a and b -/
structure Rectangle where
  a : ℝ
  b : ℝ

/-- Calculates the area of a rectangle -/
noncomputable def Rectangle.area (r : Rectangle) : ℝ := r.a * r.b

/-- Represents an isosceles right triangle -/
structure IsoscelesRightTriangle where
  leg : ℝ

/-- Calculates the area of an isosceles right triangle -/
noncomputable def IsoscelesRightTriangle.area (t : IsoscelesRightTriangle) : ℝ := t.leg * t.leg / 2

/-- The main theorem -/
theorem minimum_remaining_area (r : Rectangle) 
  (h1 : r.a = 4) 
  (h2 : r.b = 6) : 
  ∃ (t1 t2 t3 : IsoscelesRightTriangle), 
    r.area - (t1.area + t2.area + t3.area) = 2.5 ∧ 
    ∀ (s1 s2 s3 : IsoscelesRightTriangle), 
      r.area - (s1.area + s2.area + s3.area) ≥ 2.5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_remaining_area_l534_53468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_cuboid_intersection_quadrilateral_l534_53411

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular cuboid -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Calculate the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Calculate the area of a quadrilateral using Bretschneider's formula -/
noncomputable def bretschneiderArea (quad : Quadrilateral) : ℝ :=
  let a := distance quad.A quad.B
  let b := distance quad.B quad.C
  let c := distance quad.C quad.D
  let d := distance quad.D quad.A
  let s := (a + b + c + d) / 2
  let p := distance quad.A quad.C
  let q := distance quad.B quad.D
  Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d) - (a * b * c * d * (1 + ((p^2 + q^2 - a^2 - c^2) * (p^2 + q^2 - b^2 - d^2)) / (4 * p^2 * q^2))))

/-- The theorem to be proved -/
theorem area_of_cuboid_intersection_quadrilateral 
  (cuboid : Cuboid) 
  (quad : Quadrilateral) 
  (h1 : cuboid.length = 2 ∧ cuboid.width = 1 ∧ cuboid.height = 1)
  (h2 : quad.A = ⟨0, 0, 0⟩)
  (h3 : quad.B = ⟨1, 0, 0⟩)
  (h4 : quad.C = ⟨2, 0.5, 0⟩)
  (h5 : quad.D = ⟨0, 1, 0.5⟩) :
  ∃ (area : ℝ), area = bretschneiderArea quad :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_cuboid_intersection_quadrilateral_l534_53411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_100_equals_2_l534_53414

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Define a as a positive real number
noncomputable def a : ℝ := sorry

-- State the conditions
axiom inverse_functions : Function.LeftInverse f g ∧ Function.RightInverse f g
axiom g_def : ∀ x, g x = a ^ x
axiom f_point : f 10 = 1
axiom a_positive : a > 0

-- State the theorem to be proved
theorem f_100_equals_2 : f 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_100_equals_2_l534_53414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximizing_rate_l534_53464

/-- A bank's deposit and profit model -/
structure BankModel where
  k : ℝ
  h_k_pos : k > 0

variable (b : BankModel)

/-- The deposit volume as a function of interest rate -/
def deposit_volume (x : ℝ) : ℝ := b.k * x

/-- The interest paid to depositors as a function of interest rate -/
def interest_paid (x : ℝ) : ℝ := b.k * x^2

/-- The bank's profit as a function of interest rate -/
def profit (x : ℝ) : ℝ := 0.06 * b.k * x - b.k * x^2

/-- Theorem stating that the profit-maximizing deposit rate is 0.03 -/
theorem profit_maximizing_rate (b : BankModel) : 
  ∀ x, x ∈ Set.Ioo (0 : ℝ) (0.06 : ℝ) → profit b (0.03 : ℝ) ≥ profit b x := by
  sorry

#check profit_maximizing_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximizing_rate_l534_53464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_is_identity_l534_53449

/-- A function satisfying specific properties -/
def special_function (f : ℕ+ → ℕ+) : Prop :=
  (f 2 = 2) ∧
  (∀ m n : ℕ+, f (m * n) = f m * f n) ∧
  (∀ m n : ℕ+, m > n → f m > f n)

/-- Theorem stating that any function satisfying the special properties is the identity function on positive integers -/
theorem special_function_is_identity (f : ℕ+ → ℕ+) (hf : special_function f) : 
  ∀ n : ℕ+, f n = n := by
  sorry

#check special_function_is_identity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_is_identity_l534_53449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_price_correct_l534_53486

-- Define the ticket price function
noncomputable def ticket_price (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 100 then 0.5 * x
  else if x > 100 then 0.4 * x + 10
  else 0

-- Theorem statement
theorem ticket_price_correct (x : ℝ) : 
  ticket_price x = if 0 < x ∧ x ≤ 100 then 0.5 * x
                   else if x > 100 then 0.4 * x + 10
                   else 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_price_correct_l534_53486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_nine_fourths_l534_53490

theorem sqrt_nine_fourths :
  Real.sqrt (9 / 4) = 3 / 2 ∨ Real.sqrt (9 / 4) = -3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_nine_fourths_l534_53490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_4x_eq_cos_x_solutions_l534_53458

theorem tan_4x_eq_cos_x_solutions : ∃! (n : ℕ), n = 9 ∧
  ∃ (S : Finset ℝ), S.card = n ∧
  (∀ x ∈ S, x ∈ Set.Icc 0 (2 * Real.pi) ∧ Real.tan (4 * x) = Real.cos x) ∧
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), Real.tan (4 * x) = Real.cos x → x ∈ S) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_4x_eq_cos_x_solutions_l534_53458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_target_set_count_l534_53477

-- Define the upper bound
def upper_bound : ℕ := 499

-- Define the set of numbers we're interested in
def target_set : Set ℕ := {n : ℕ | n > 0 ∧ n ≤ upper_bound ∧ n % 5 ≠ 0 ∧ n % 7 ≠ 0}

-- The theorem to prove
theorem target_set_count : Finset.card (Finset.filter (λ n => n > 0 ∧ n ≤ upper_bound ∧ n % 5 ≠ 0 ∧ n % 7 ≠ 0) (Finset.range (upper_bound + 1))) = 343 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_target_set_count_l534_53477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_six_nineteenths_l534_53457

-- Define the triangle PQR
structure Triangle where
  PQ : ℝ
  QR : ℝ
  PR : ℝ

-- Define points S and T
structure Points where
  PS : ℝ
  PT : ℝ

-- Define the ratio of areas
noncomputable def areaRatio (t : Triangle) (p : Points) : ℝ :=
  6 / 19

-- Theorem statement
theorem area_ratio_is_six_nineteenths (t : Triangle) (p : Points) 
  (h1 : t.PQ = 30) (h2 : t.QR = 50) (h3 : t.PR = 54) 
  (h4 : p.PS = 18) (h5 : p.PT = 24) :
  areaRatio t p = 6 / 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_six_nineteenths_l534_53457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_distribution_maximizes_triples_l534_53489

/-- Represents a distribution of points into groups -/
structure PointDistribution where
  sizes : List Nat
  unequal_sizes : ∀ i j, i ≠ j → sizes.get? i ≠ sizes.get? j
  total_points : sizes.sum = 1989
  group_count : sizes.length = 30

/-- Counts the number of valid triples for a given distribution -/
def count_valid_triples (d : PointDistribution) : Nat :=
  sorry

/-- The optimal distribution of points -/
def optimal_distribution : PointDistribution :=
  { sizes := (List.range 31).map (fun i => if i < 6 then i + 51 else i + 52)
    unequal_sizes := by sorry
    total_points := by sorry
    group_count := by sorry }

/-- Theorem stating that the optimal distribution maximizes valid triples -/
theorem optimal_distribution_maximizes_triples :
  ∀ d : PointDistribution, count_valid_triples d ≤ count_valid_triples optimal_distribution :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_distribution_maximizes_triples_l534_53489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_three_digit_numbers_l534_53462

/-- Represents a three-digit number --/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to a natural number --/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Converts a ThreeDigitNumber to a real number with decimal point after first digit --/
noncomputable def ThreeDigitNumber.toRealAfterFirst (n : ThreeDigitNumber) : Real :=
  n.hundreds + (n.tens : Real) / 10 + (n.ones : Real) / 100

/-- Converts a ThreeDigitNumber to a real number with decimal point after second digit --/
noncomputable def ThreeDigitNumber.toRealAfterSecond (n : ThreeDigitNumber) : Real :=
  10 * n.hundreds + n.tens + (n.ones : Real) / 10

theorem sum_of_three_digit_numbers (a b : ThreeDigitNumber) 
    (h1 : a.toRealAfterFirst + b.toRealAfterSecond = 50.13)
    (h2 : a.toRealAfterSecond + b.toRealAfterFirst = 34.02) :
    a.toNat + b.toNat = 765 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_three_digit_numbers_l534_53462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_arithmetic_sequence_l534_53438

theorem cos_arithmetic_sequence (a : Real) :
  0 < a ∧ a < 2 * Real.pi ∧
  (∃ r : Real, Real.cos a - Real.cos (2 * a) = Real.cos (2 * a) - Real.cos (4 * a) ∧ r = Real.cos a - Real.cos (2 * a)) →
  a = Real.pi / 3 ∨ a = 2 * Real.pi / 3 ∨ a = 4 * Real.pi / 3 ∨ a = 5 * Real.pi / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_arithmetic_sequence_l534_53438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_line_equations_l534_53401

/-- Triangle ABC with given conditions -/
structure Triangle where
  /-- Vertex B coordinates -/
  B : ℝ × ℝ
  /-- Equation of altitude on side AB: ax + by + c = 0 -/
  altitude : ℝ × ℝ × ℝ
  /-- Equation of angle bisector of angle A: px + qy + r = 0 -/
  angle_bisector : ℝ × ℝ × ℝ

/-- Helper function to represent line AB -/
def line_AB (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- Helper function to represent line AC -/
def line_AC (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- Theorem stating the equations of lines AB and AC given the conditions -/
theorem triangle_line_equations (t : Triangle) 
  (h1 : t.B = (-2, 0))
  (h2 : t.altitude = (1, 3, -26))
  (h3 : t.angle_bisector = (1, 1, -2)) :
  (∃ (a b c : ℝ), a = 3 ∧ b = -1 ∧ c = 6 ∧ 
    (∀ (x y : ℝ), a*x + b*y + c = 0 ↔ (x, y) ∈ line_AB t)) ∧ 
  (∃ (p q r : ℝ), p = 1 ∧ q = -3 ∧ r = 10 ∧ 
    (∀ (x y : ℝ), p*x + q*y + r = 0 ↔ (x, y) ∈ line_AC t)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_line_equations_l534_53401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polyhedra_count_l534_53485

/-- A regular polyhedron with n-sided faces and m-sided polyhedral angles -/
structure RegularPolyhedron where
  n : ℕ
  m : ℕ
  f : ℕ
  n_ge_3 : n ≥ 3
  m_ge_3 : m ≥ 3
  f_pos : f > 0
  dihedral_angle_eq : (2 : ℝ) / m = (4 + n * f - 2 * f) / (n * f)

/-- The set of all possible regular polyhedra -/
def RegularPolyhedra : Set RegularPolyhedron :=
  {p : RegularPolyhedron | ∃ (n m f : ℕ), p.n = n ∧ p.m = m ∧ p.f = f}

theorem regular_polyhedra_count :
  ∃ (S : Finset RegularPolyhedron), ↑S = RegularPolyhedra ∧ Finset.card S = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polyhedra_count_l534_53485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l534_53413

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 + Real.cos (2 * x)

theorem f_properties :
  ∃ (T : ℝ), T > 0 ∧ 
  (∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = π ∧
  (∀ x ∈ Set.Icc 0 (π / 2), f x ≤ Real.sqrt 2 + 1) ∧
  (∃ x ∈ Set.Icc 0 (π / 2), f x = Real.sqrt 2 + 1) ∧
  (∀ x ∈ Set.Icc 0 (π / 2), f x ≥ 0) ∧
  (∃ x ∈ Set.Icc 0 (π / 2), f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l534_53413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_x_given_condition_l534_53475

theorem max_cos_x_given_condition (x y : ℝ) :
  Real.cos (2 * x + y) = Real.cos x + Real.cos y →
  ∀ z, Real.cos x ≤ Real.cos z →
  Real.cos x ≤ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_x_given_condition_l534_53475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_necessary_not_sufficient_l534_53470

theorem sin_half_necessary_not_sufficient :
  (∃ θ : Real, Real.sin θ = 1/2 ∧ θ ≠ 30 * Real.pi / 180) ∧
  (∀ θ : Real, θ = 30 * Real.pi / 180 → Real.sin θ = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_necessary_not_sufficient_l534_53470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_section_proof_l534_53476

/-- Represents a cone with equal base radius and height -/
structure EqualRadiusHeightCone where
  radius : ℝ
  lateral_area : ℝ
  section_area : ℝ

/-- The distance from the center of the base to the section -/
noncomputable def distance_to_section (cone : EqualRadiusHeightCone) : ℝ :=
  2 * Real.sqrt 3 / 3

/-- Theorem stating the distance from the center of the base to the section -/
theorem distance_to_section_proof (cone : EqualRadiusHeightCone) 
  (h1 : cone.lateral_area = 4 * Real.sqrt 2 * Real.pi)
  (h2 : cone.section_area = 2 * Real.sqrt 3) : 
  distance_to_section cone = 2 * Real.sqrt 3 / 3 := by
  sorry

#check distance_to_section_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_section_proof_l534_53476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_third_vertex_l534_53492

/-- The y-coordinate of the third vertex of an equilateral triangle -/
noncomputable def third_vertex_y_coordinate : ℝ := 3 + 4 * Real.sqrt 3

/-- Theorem: The y-coordinate of the third vertex of an equilateral triangle with two vertices
    at (2,3) and (10,3), and the third vertex in the first quadrant, is 3 + 4√3 -/
theorem equilateral_triangle_third_vertex :
  let vertex1 : ℝ × ℝ := (2, 3)
  let vertex2 : ℝ × ℝ := (10, 3)
  ∀ (vertex3 : ℝ × ℝ),
  (vertex3.1 > 0 ∧ vertex3.2 > 0) →  -- Third vertex in first quadrant
  (vertex1.1 - vertex3.1)^2 + (vertex1.2 - vertex3.2)^2 = 64 →  -- Distance from vertex1 to vertex3 is 8
  (vertex2.1 - vertex3.1)^2 + (vertex2.2 - vertex3.2)^2 = 64 →  -- Distance from vertex2 to vertex3 is 8
  vertex3.2 = third_vertex_y_coordinate :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_third_vertex_l534_53492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_length_is_505_l534_53410

/-- The number of terms in an arithmetic sequence with first term 5,
    common difference 4, and last term less than or equal to 2021 -/
def arithmetic_sequence_length : ℕ :=
  (2021 - 5) / 4 + 1

theorem arithmetic_sequence_length_is_505 : arithmetic_sequence_length = 505 := by
  -- Unfold the definition of arithmetic_sequence_length
  unfold arithmetic_sequence_length
  -- Evaluate the arithmetic expression
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_length_is_505_l534_53410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_150_degrees_l534_53446

/-- A line in 2D space defined by parametric equations. -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The angle of inclination of a line, measured in radians. -/
noncomputable def angleOfInclination (l : ParametricLine) : ℝ := sorry

/-- Converts degrees to radians. -/
noncomputable def degToRad (deg : ℝ) : ℝ := deg * (Real.pi / 180)

theorem line_inclination_150_degrees :
  let l : ParametricLine := {
    x := λ t => 1 + 3 * t,
    y := λ t => 2 - Real.sqrt 3 * t
  }
  let α := angleOfInclination l
  0 ≤ α ∧ α < degToRad 180 ∧ α = degToRad 150 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_150_degrees_l534_53446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_width_approx_6_l534_53435

/-- The width of a cistern with given dimensions and wet surface area -/
noncomputable def cistern_width (length : ℝ) (water_depth : ℝ) (wet_surface_area : ℝ) : ℝ :=
  (wet_surface_area - 2 * length * water_depth) / (length + 2 * water_depth)

/-- Theorem stating that the width of the cistern is approximately 6 meters -/
theorem cistern_width_approx_6 :
  let length := (8 : ℝ)
  let water_depth := (1.85 : ℝ)
  let wet_surface_area := (99.8 : ℝ)
  abs (cistern_width length water_depth wet_surface_area - 6) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_width_approx_6_l534_53435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_constraint_l534_53463

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (a * 9^x + 3^x + 1)

-- State the theorem
theorem domain_constraint (a : ℝ) :
  (∀ x ≤ 1, f a x ∈ Set.Ioi 0) ↔ a = -4/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_constraint_l534_53463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beautiful_sum_l534_53402

-- Define beautiful numbers
def IsBeautiful (x : ℕ) : Prop :=
  ∃ (a n : ℕ), x = a^n ∧ a ∈ ({3, 4, 5, 6} : Finset ℕ) ∧ n > 0

-- Define the theorem
theorem beautiful_sum (n : ℕ) (h : n ≥ 3) :
  ∃ (S : Finset ℕ), (∀ x ∈ S, IsBeautiful x) ∧ (S.sum id = n) ∧ S.card = S.toSet.toFinset.card :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beautiful_sum_l534_53402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventeen_one_dollar_coins_needed_l534_53496

/-- The minimum number of one-dollar coins needed to buy a sweater -/
def min_one_dollar_coins (sweater_price : ℚ) (five_dollar_bills : ℕ) (nickels : ℕ) : ℕ :=
  let five_dollar_amount : ℚ := (5 : ℚ) * five_dollar_bills
  let nickel_amount : ℚ := (5 : ℚ) / 100 * nickels
  (sweater_price - five_dollar_amount - nickel_amount).ceil.toNat

/-- Proof that 17 one-dollar coins are needed for the given scenario -/
theorem seventeen_one_dollar_coins_needed :
  min_one_dollar_coins 37.5 4 10 = 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventeen_one_dollar_coins_needed_l534_53496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_status_l534_53423

/-- Represents the external diameter of a part -/
def ExternalDiameter : Type := ℝ

/-- Represents whether a production is normal or abnormal -/
inductive ProductionStatus
| Normal
| Abnormal

/-- The mean of the normal distribution for external diameters -/
def mean : ℝ := 10

/-- The standard deviation of the normal distribution for external diameters -/
def std_dev : ℝ := 0.2  -- sqrt(0.04)

/-- The lower bound for normal production (3 standard deviations below the mean) -/
def lower_bound : ℝ := mean - 3 * std_dev

/-- The upper bound for normal production (3 standard deviations above the mean) -/
def upper_bound : ℝ := mean + 3 * std_dev

/-- Determines if a given diameter is within normal range -/
def is_normal (d : ℝ) : Prop :=
  lower_bound ≤ d ∧ d ≤ upper_bound

/-- The diameter of the part from morning production -/
def morning_diameter : ℝ := 9.9

/-- The diameter of the part from afternoon production -/
def afternoon_diameter : ℝ := 9.3

/-- Theorem stating that the morning production is normal and the afternoon production is abnormal -/
theorem production_status :
  (is_normal morning_diameter) ∧ ¬(is_normal afternoon_diameter) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_status_l534_53423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_infinite_power_arithmetic_progression_l534_53482

theorem no_infinite_power_arithmetic_progression :
  ¬ ∃ (a₁ d : ℕ) (f : ℕ → ℕ),
    (∀ n, ∃ (a b : ℕ), b ≥ 2 ∧ f n = a ^ b) ∧
    (∀ n, f (n + 1) = f n + d) ∧
    (d ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_infinite_power_arithmetic_progression_l534_53482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_mean_median_relation_l534_53494

theorem set_mean_median_relation (x : ℤ) (a : ℤ) :
  let s : Finset ℤ := {x, x + a, x + 4, x + 7, x + 27}
  let median := x + 4
  let mean := (x + (x + a) + (x + 4) + (x + 7) + (x + 27)) / 5
  (∀ i ∈ s, i > 0) → (mean = median + 4) → a = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_mean_median_relation_l534_53494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_expression_l534_53439

theorem max_value_trig_expression :
  (∃ (θ₁ θ₂ θ₃ θ₄ θ₅ : ℝ), 
    (Real.cos θ₁)^2 * (Real.sin θ₂)^2 + (Real.cos θ₂)^2 * (Real.sin θ₃)^2 + (Real.cos θ₃)^2 * (Real.sin θ₄)^2 + 
    (Real.cos θ₄)^2 * (Real.sin θ₅)^2 + (Real.cos θ₅)^2 * (Real.sin θ₁)^2 = 5/4) ∧
  (∀ (θ₁ θ₂ θ₃ θ₄ θ₅ : ℝ), 
    (Real.cos θ₁)^2 * (Real.sin θ₂)^2 + (Real.cos θ₂)^2 * (Real.sin θ₃)^2 + (Real.cos θ₃)^2 * (Real.sin θ₄)^2 + 
    (Real.cos θ₄)^2 * (Real.sin θ₅)^2 + (Real.cos θ₅)^2 * (Real.sin θ₁)^2 ≤ 5/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_expression_l534_53439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octahedron_faces_l534_53426

/-- A regular octahedron is a three-dimensional geometric shape -/
structure RegularOctahedron where
  -- No specific properties needed for this problem

/-- The number of faces in a geometric shape -/
def num_faces (shape : Type) : ℕ := sorry

/-- Theorem: A regular octahedron has 8 faces -/
theorem regular_octahedron_faces :
  ∀ (o : RegularOctahedron), num_faces RegularOctahedron = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octahedron_faces_l534_53426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_square_properties_l534_53436

/-- Represents a 4x4 magic square -/
structure MagicSquare where
  board : Matrix (Fin 4) (Fin 4) ℕ
  is_valid : ∀ i j, board i j ∈ Finset.range 16
  is_magic : ∀ i, (Finset.sum Finset.univ (λ j ↦ board i j)) = 
                   (Finset.sum Finset.univ (λ j ↦ board j i)) ∧
              (Finset.sum Finset.univ (λ i ↦ board i i)) = 
              (Finset.sum Finset.univ (λ i ↦ board i (3 - i)))

/-- The Magic Sum of a 4x4 magic square -/
def magicSum (ms : MagicSquare) : ℕ := Finset.sum Finset.univ (λ j ↦ ms.board 0 j)

/-- Sum of the four corner cells -/
def cornerSum (ms : MagicSquare) : ℕ := 
  ms.board 0 0 + ms.board 0 3 + ms.board 3 0 + ms.board 3 3

/-- Sum of the four center cells -/
def centerSum (ms : MagicSquare) : ℕ := 
  ms.board 1 1 + ms.board 1 2 + ms.board 2 1 + ms.board 2 2

/-- Theorem stating the main properties of the magic square -/
theorem magic_square_properties (ms : MagicSquare) : 
  (magicSum ms = 34) ∧ 
  (centerSum ms = 34 → cornerSum ms = 34) ∧
  (∃ k : ℕ, (∀ i j, ms.board i j ∈ Finset.range 16 ∧ 
             ms.board i j ≥ k ∧ ms.board i j < k + 16) ∧ 
   magicSum ms = 50 → k = 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_square_properties_l534_53436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_domain_l534_53403

-- Define the expression
noncomputable def f (x : ℝ) := Real.sqrt (x + 2) + 1 / x

-- Define the domain of the expression
def domain (x : ℝ) : Prop := x + 2 ≥ 0 ∧ x ≠ 0

-- Theorem statement
theorem expression_domain :
  {x : ℝ | domain x} = {x : ℝ | x ≥ -2 ∧ x ≠ 0} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_domain_l534_53403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_d_inequality_l534_53416

theorem smallest_d_inequality (d : ℝ) : 
  (d = 1) ↔ 
  (d > 0 ∧ 
   ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → 
   Real.exp (x * y) + d * abs (x^2 - y^2) ≥ Real.exp ((x + y) / 2)) ∧
  (∀ d' : ℝ, 0 < d' → d' < d → 
   ∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ 
   Real.exp (x * y) + d' * abs (x^2 - y^2) < Real.exp ((x + y) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_d_inequality_l534_53416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_multiplications_l534_53420

/-- Represents a polynomial of degree n -/
def MyPolynomial (n : ℕ) := Fin (n + 1) → ℝ

/-- The degree of the polynomial -/
def degree : ℕ := 6

/-- The polynomial f(x) = 5x^6 - 7x^5 + 2x^4 - 8x^3 + 3x^2 - 9x + 1 -/
def f : MyPolynomial degree := fun i => 
  match i with
  | 0 => 1
  | 1 => -9
  | 2 => 3
  | 3 => -8
  | 4 => 2
  | 5 => -7
  | 6 => 5

/-- The number of multiplications required by Horner's method for a polynomial of degree n -/
def horner_multiplications (n : ℕ) : ℕ := n

theorem horner_method_multiplications : 
  horner_multiplications degree = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_multiplications_l534_53420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emma_finishes_first_l534_53473

/-- Represents the area of a lawn -/
structure LawnArea where
  size : ℝ
  size_pos : size > 0

/-- Represents the mowing speed of a lawn mower -/
structure MowingSpeed where
  rate : ℝ
  rate_pos : rate > 0

/-- Calculates the time needed to mow a lawn -/
noncomputable def mowing_time (area : LawnArea) (speed : MowingSpeed) : ℝ :=
  area.size / speed.rate

theorem emma_finishes_first 
  (daniel_lawn : LawnArea)
  (emma_lawn : LawnArea)
  (fiona_lawn : LawnArea)
  (daniel_speed : MowingSpeed)
  (emma_speed : MowingSpeed)
  (fiona_speed : MowingSpeed)
  (h1 : daniel_lawn.size = 3 * emma_lawn.size)
  (h2 : daniel_lawn.size = 4 * fiona_lawn.size)
  (h3 : fiona_speed.rate = 1/2 * emma_speed.rate)
  (h4 : fiona_speed.rate = 1/4 * daniel_speed.rate) :
  mowing_time emma_lawn emma_speed < min (mowing_time daniel_lawn daniel_speed) (mowing_time fiona_lawn fiona_speed) :=
by
  sorry

#check emma_finishes_first

end NUMINAMATH_CALUDE_ERRORFEEDBACK_emma_finishes_first_l534_53473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chips_to_win_l534_53409

/-- Represents a 5x5 game board -/
def Board := Fin 5 → Fin 5 → Bool

/-- An L-shaped tile covering 3 cells -/
structure LTile :=
  (x y : Fin 5)
  (orientation : Fin 4)

/-- Check if an L-tile is valid (within board boundaries) -/
def LTile.isValid (t : LTile) : Bool :=
  sorry

/-- Check if two L-tiles overlap -/
def LTile.overlap (t1 t2 : LTile) : Bool :=
  sorry

/-- Check if an L-tile covers a specific cell -/
def LTile.covers (t : LTile) (x y : Fin 5) : Bool :=
  sorry

/-- Check if a set of L-tiles covers all chips on the board -/
def coversAllChips (tiles : List LTile) (board : Board) : Bool :=
  sorry

/-- The main theorem stating that 9 is the minimum number of chips needed -/
theorem min_chips_to_win (board : Board) :
  (∃ (chips : Finset (Fin 5 × Fin 5)), chips.card = 9 ∧
    ∀ (tiles : List LTile),
      (∀ t1 t2, t1 ∈ tiles → t2 ∈ tiles → t1 ≠ t2 → ¬ t1.overlap t2) →
      (∀ t, t ∈ tiles → t.isValid) →
      ¬ coversAllChips tiles board) ∧
  (∀ (chips : Finset (Fin 5 × Fin 5)), chips.card < 9 →
    ∃ (tiles : List LTile),
      (∀ t1 t2, t1 ∈ tiles → t2 ∈ tiles → t1 ≠ t2 → ¬ t1.overlap t2) ∧
      (∀ t, t ∈ tiles → t.isValid) ∧
      coversAllChips tiles board) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chips_to_win_l534_53409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_lines_through_point_l534_53452

/-- Given points P, A, and B, there exist two lines passing through P
    such that the distance from A to each line is equal to the distance from B to that line -/
theorem equidistant_lines_through_point (P A B : ℝ × ℝ) 
  (hP : P = (3, -1)) (hA : A = (2, -3)) (hB : B = (-4, 5)) : 
  ∃ (l₁ l₂ : ℝ → ℝ → Prop),
    (∀ x y, l₁ x y ↔ 4 * x + 3 * y - 9 = 0) ∧
    (∀ x y, l₂ x y ↔ x + 2 * y - 1 = 0) ∧
    (∀ x y, l₁ x y → l₁ 3 (-1)) ∧
    (∀ x y, l₂ x y → l₂ 3 (-1)) ∧
    (∀ x y, l₁ x y → dist A (x, y) = dist B (x, y)) ∧
    (∀ x y, l₂ x y → dist A (x, y) = dist B (x, y)) :=
by sorry

/-- The distance function between two points in ℝ² -/
noncomputable def dist (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_lines_through_point_l534_53452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_special_bisector_l534_53437

noncomputable def AngleBisector (A B C : ℝ × ℝ) : Prop := sorry
noncomputable def Altitude (A B C : ℝ × ℝ) : Prop := sorry
noncomputable def RightTriangle (A B C : ℝ × ℝ) : Prop := sorry
noncomputable def AngleBisectorPoint (A B C : ℝ × ℝ) : ℝ × ℝ := sorry
noncomputable def AltitudePoint (A B C : ℝ × ℝ) : ℝ × ℝ := sorry
noncomputable def PointOnLine (B C : ℝ × ℝ) : ℝ × ℝ := sorry
noncomputable def SegmentRatio (P Q R : ℝ × ℝ) : ℝ := sorry
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry

theorem right_triangle_special_bisector (A B C : ℝ × ℝ) 
  (H : RightTriangle A B C) 
  (h_bisector : AngleBisector A B C)
  (h_altitude : Altitude A B C)
  (h_ratio : SegmentRatio (AngleBisectorPoint A B C) (AltitudePoint A B C) (PointOnLine B C) = 1 + Real.sqrt 2) :
  angle B A C = π/4 ∧ angle A B C = π/4 := by
  sorry

#check right_triangle_special_bisector

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_special_bisector_l534_53437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_equals_five_l534_53430

-- Define the coordinates of points A and B
def A : ℝ × ℝ := (-1, -2)
def B : ℝ → ℝ × ℝ := λ a ↦ (2, a)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem distance_equals_five (a : ℝ) :
  distance A (B a) = 5 ↔ a = 2 ∨ a = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_equals_five_l534_53430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_distance_l534_53440

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Checks if a point is on the ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / (2*e.a)^2 + p.y^2 / (2*e.b)^2 = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Main theorem -/
theorem ellipse_intersection_distance 
  (e : Ellipse) 
  (F1 F2 A B : Point) : 
  e.a^2 = 5 ∧ e.b^2 = 1 →
  isOnEllipse A e ∧ isOnEllipse B e →
  distance F1 A + distance F1 B = 5 * Real.sqrt 5 →
  distance A B = 3 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_distance_l534_53440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lino_shell_count_l534_53495

/-- Calculates the final number of shells Lino has after collecting, discarding broken ones,
    gifting some, finding more, and putting some back. -/
def final_shell_count (initial_shells : ℕ) (broken_percent : ℚ) (gifted_shells : ℕ)
                      (additional_shells : ℕ) (put_back_percent : ℚ) : ℕ :=
  let remaining_after_broken := initial_shells - (↑initial_shells * broken_percent).ceil.toNat
  let remaining_after_gifting := remaining_after_broken - gifted_shells
  let total_before_putting_back := remaining_after_gifting + additional_shells
  let put_back_shells := (↑additional_shells * put_back_percent).floor.toNat
  total_before_putting_back - put_back_shells

/-- Theorem stating that given the initial conditions, Lino ends up with 367 shells. -/
theorem lino_shell_count :
  final_shell_count 324 (15 / 100) 25 292 (60 / 100) = 367 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lino_shell_count_l534_53495
