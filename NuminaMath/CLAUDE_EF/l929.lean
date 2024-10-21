import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_schedules_count_lower_bound_l929_92914

/-- Represents a degustation schedule for Viktor and Natalia. -/
def DegustationSchedule := List (Fin 2020 × Fin 2020)

/-- Checks if a schedule is fair. -/
def isFair (schedule : DegustationSchedule) : Bool :=
  sorry

/-- Counts the number of fair schedules. -/
def countFairSchedules : Nat :=
  sorry

/-- The main theorem to be proved. -/
theorem fair_schedules_count_lower_bound :
  countFairSchedules > 2020 * (2^1010 + 1010 * 1010) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_schedules_count_lower_bound_l929_92914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alec_election_ratio_l929_92981

/-- Proves that given the conditions of Alec's election campaign, the ratio of his goal votes to total students is 3:4 -/
theorem alec_election_ratio (total_students : ℕ) (considering_voters : ℕ) (additional_votes_needed : ℕ) :
  total_students = 60 →
  considering_voters = 5 →
  additional_votes_needed = 5 →
  (((total_students / 2 + considering_voters + (total_students - total_students / 2 - considering_voters) / 5 + additional_votes_needed) : ℚ) / total_students) = 3 / 4 :=
by
  intro h1 h2 h3
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alec_election_ratio_l929_92981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_second_side_length_l929_92926

/-- The length of the second parallel side of a trapezium -/
noncomputable def second_side_length (a b h : ℝ) : ℝ := 2 * a / h - b

/-- Theorem: Given a trapezium with one parallel side of 20 cm, 
    a distance of 14 cm between the parallel sides, 
    and an area of 245 square centimeters, 
    the length of the second parallel side is 15 cm. -/
theorem trapezium_second_side_length : 
  second_side_length 245 20 14 = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_second_side_length_l929_92926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_dot_product_l929_92934

-- Define the line l
noncomputable def line_l (α : ℝ) (t : ℝ) : ℝ × ℝ :=
  (1 + t * Real.cos α, t * Real.sin α)

-- Define the curve C
noncomputable def curve_C (θ : ℝ) : ℝ :=
  4 * Real.cos θ / (1 - Real.cos θ ^ 2)

-- State the theorem
theorem intersection_dot_product
  (α : ℝ)
  (h_α : 0 < α ∧ α < Real.pi / 2)
  (A B : ℝ × ℝ)
  (h_A : ∃ t₁ θ₁, line_l α t₁ = A ∧ curve_C θ₁ = Real.sqrt (A.1^2 + A.2^2))
  (h_B : ∃ t₂ θ₂, line_l α t₂ = B ∧ curve_C θ₂ = Real.sqrt (B.1^2 + B.2^2))
  : A.1 * B.1 + A.2 * B.2 = -3 := by
  sorry

#check intersection_dot_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_dot_product_l929_92934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_path_turns_l929_92920

/-- The expected number of turns for a fly moving on an n × n grid -/
noncomputable def expected_turns (n : ℕ) : ℝ :=
  n + 1/2 - (n - 1/2) / Real.sqrt (Real.pi * (n - 1))

/-- Theorem stating the expected number of turns for a fly on an n × n grid -/
theorem fly_path_turns (n : ℕ) (hn : n > 1) :
  ∃ (ε : ℝ) (hε : ε > 0),
  |expected_turns n - (n + 1/2 - (n - 1/2) * (Nat.choose (2*n - 2) (n - 1) / 2^(2*n - 2)))| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_path_turns_l929_92920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fiftieth_digit_of_five_fourteenths_l929_92962

theorem fiftieth_digit_of_five_fourteenths (n : ℕ) : n = 50 → 
  (∃ (a b : ℕ) (d : Fin 10), 
    (5 : ℚ) / 14 = ↑a + ↑b * (10 : ℚ)^(-(n : ℤ)) + (↑d : ℚ) * (10 : ℚ)^(-(n + 1 : ℤ)) ∧ 
    d = 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fiftieth_digit_of_five_fourteenths_l929_92962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flash_catch_up_theorem_l929_92994

/-- The distance Flash must run to catch up to Ace -/
noncomputable def flash_catch_up_distance (v x z k : ℝ) : ℝ :=
  (x * (z + v * k)) / (x - 1)

/-- Theorem stating the distance Flash must run to catch up to Ace -/
theorem flash_catch_up_theorem (v x z k : ℝ) 
  (hv : v > 0) (hx : x > 1) (hz : z ≥ 0) (hk : k ≥ 0) :
  flash_catch_up_distance v x z k = (x * (z + v * k)) / (x - 1) := by
  -- Proof goes here
  sorry

#check flash_catch_up_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flash_catch_up_theorem_l929_92994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l929_92958

-- Define the given conditions
noncomputable def α : Real := Real.arctan (-1/3)
noncomputable def β : Real := Real.arccos (Real.sqrt 5 / 5)

-- Define the function f
noncomputable def f (x : Real) : Real := Real.sqrt 2 * Real.sin (x - α) + Real.cos (x + β)

-- Theorem statement
theorem problem_solution :
  (0 < α ∧ α < Real.pi) ∧ 
  (0 < β ∧ β < Real.pi) ∧ 
  Real.tan α = -1/3 ∧ 
  Real.cos β = Real.sqrt 5 / 5 →
  Real.tan (α + β) = 1 ∧
  (∀ x, f x ≤ Real.sqrt 5) ∧
  (∀ x, f x ≥ -Real.sqrt 5) ∧
  (∃ x, f x = Real.sqrt 5) ∧
  (∃ x, f x = -Real.sqrt 5) := by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l929_92958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_parallel_projection_l929_92923

-- Define a 2D vector type
def Vector2D := ℝ × ℝ

-- Define vector addition
noncomputable def vector_add (v1 v2 : Vector2D) : Vector2D :=
  (v1.1 + v2.1, v1.2 + v2.2)

-- Define vector projection
noncomputable def vector_proj (a b : Vector2D) : Vector2D :=
  let scalar := (a.1 * b.1 + a.2 * b.2) / (b.1 * b.1 + b.2 * b.2)
  (scalar * b.1, scalar * b.2)

-- Define parallel vectors
def parallel (a b : Vector2D) : Prop :=
  ∃ (k : ℝ), a = (k * b.1, k * b.2)

theorem vector_equality (AB MB BC OM CO : Vector2D) :
  vector_add (vector_add (vector_add (vector_add AB MB) BC) OM) CO = AB := by
  sorry

theorem parallel_projection (a b : Vector2D) :
  parallel a b → vector_proj a b = a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_parallel_projection_l929_92923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_l929_92900

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 5

-- Define the point P
def P : ℝ × ℝ := (2, 4)

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := x + 2*y - 10 = 0

-- Theorem statement
theorem tangent_line_at_P : 
  ∀ (x y : ℝ), my_circle x y → tangent_line x y ↔ 
  ∃ (t : ℝ), (x, y) = (2 + t, 4 - t/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_l929_92900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_bounds_l929_92984

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1  -- Define for 0 to cover all natural numbers
  | 1 => 1
  | n + 2 => sequence_a (n + 1) - (1/3) * (sequence_a (n + 1))^2

theorem sequence_a_bounds : 5/2 < 100 * sequence_a 100 ∧ 100 * sequence_a 100 < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_bounds_l929_92984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_meeting_on_7th_l929_92964

/-- Represents a day in January --/
def Day := Fin 31

/-- Represents a visiting schedule --/
structure Schedule where
  start : Day
  interval : Nat

/-- Xiaoming's schedule --/
def xiaoming : Schedule :=
  { start := ⟨3, by norm_num⟩,  -- First Wednesday of January
    interval := 4 }

/-- Xiaoqiang's schedule --/
def xiaoqiang : Schedule :=
  { start := ⟨4, by norm_num⟩,  -- First Thursday of January
    interval := 3 }

/-- Check if a given day is in the schedule --/
def inSchedule (day : Day) (schedule : Schedule) : Prop :=
  ∃ k : Nat, day.val = (schedule.start.val + k * schedule.interval) % 31 + 1

/-- The day they meet in January --/
def meetingDay : Day := ⟨7, by norm_num⟩

/-- Main theorem: Xiaoming and Xiaoqiang meet only once in January, on the 7th --/
theorem unique_meeting_on_7th :
  (inSchedule meetingDay xiaoming ∧ inSchedule meetingDay xiaoqiang) ∧
  (∀ d : Day, d ≠ meetingDay →
    ¬(inSchedule d xiaoming ∧ inSchedule d xiaoqiang)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_meeting_on_7th_l929_92964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nadines_cleaning_time_l929_92983

/-- Represents the cleaning process of a muddy dog -/
structure DogCleaning where
  initial_mud : ℝ
  hosing_time : ℕ
  hosing_removal : ℝ
  shampoo1_time : ℕ
  shampoo1_removal : ℝ
  shampoo2_time : ℕ
  shampoo2_removal : ℝ
  shampoo3_time : ℕ
  shampoo3_removal : ℝ
  drying_base_time : ℕ
  drying_extra_time : ℕ
  drying_threshold : ℝ
  brushing_normal_time : ℕ
  brushing_quick_time : ℕ
  brushing_threshold : ℝ

/-- Calculates the total cleaning time for a given DogCleaning process -/
def total_cleaning_time (dc : DogCleaning) : ℕ := sorry

/-- The specific cleaning process described in the problem -/
def nadines_cleaning : DogCleaning :=
  { initial_mud := 100
  , hosing_time := 10
  , hosing_removal := 50
  , shampoo1_time := 15
  , shampoo1_removal := 30
  , shampoo2_time := 12
  , shampoo2_removal := 15
  , shampoo3_time := 10
  , shampoo3_removal := 5
  , drying_base_time := 20
  , drying_extra_time := 5
  , drying_threshold := 10
  , brushing_normal_time := 25
  , brushing_quick_time := 20
  , brushing_threshold := 90
  }

/-- Theorem stating that Nadine's cleaning process takes 97 minutes -/
theorem nadines_cleaning_time : total_cleaning_time nadines_cleaning = 97 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nadines_cleaning_time_l929_92983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_l929_92910

theorem sequence_limit : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
    |((3 * (n : ℝ)^2 - 6 * n + 7) / (3 * n^2 + 20 * n - 1))^(-(n : ℝ) + 1) - Real.exp (26/3)| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_l929_92910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_sum_equals_one_l929_92907

noncomputable def probability_distribution (a : ℝ) (k : ℕ) : ℝ :=
  a * (1/3)^k

theorem distribution_sum_equals_one (a : ℝ) :
  (probability_distribution a 1 + probability_distribution a 2 + probability_distribution a 3 = 1) →
  (a = 27/13) :=
by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check distribution_sum_equals_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_sum_equals_one_l929_92907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_coordinates_l929_92947

/-- The circle in polar coordinates --/
def my_circle (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

/-- The line parallel to l --/
def parallel_line (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 3

/-- The line l --/
def line_l (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 1

/-- The center of the circle --/
def circle_center : ℝ × ℝ := (1, 0)

/-- The polar axis --/
def polar_axis (θ : ℝ) : Prop := θ = 0

theorem intersection_point_coordinates :
  ∀ (ρ θ : ℝ),
  (∃ (ρ₀ θ₀ : ℝ), my_circle ρ₀ θ₀ ∧ line_l ρ₀ θ₀) →  -- l passes through the center of the circle
  (∀ (ρ₁ θ₁ : ℝ), line_l ρ₁ θ₁ ↔ parallel_line (ρ₁ - 1) θ₁) →  -- l is parallel to ρcosθ=3
  polar_axis θ →  -- Intersection with polar axis
  line_l ρ θ →  -- Point satisfies the equation of line l
  ρ = 1 ∧ θ = 0  -- The intersection point is (1,0)
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_coordinates_l929_92947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_has_six_sides_number_of_cube_sides_l929_92955

/-- The number of orientations a cube can have -/
def cube_orientations : ℕ := 24

/-- The number of distinct ways to paint the cube -/
def distinct_paintings : ℕ := 30

/-- The number of available colors -/
def available_colors : ℕ := 6

/-- Theorem stating that a cube has 6 sides given the painting conditions -/
theorem cube_has_six_sides :
  (Nat.factorial available_colors) / cube_orientations = distinct_paintings →
  available_colors = 6 := by
  sorry

/-- Main theorem proving the number of sides on the cube -/
theorem number_of_cube_sides : ℕ := by
  exact available_colors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_has_six_sides_number_of_cube_sides_l929_92955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_cubed_l929_92902

def expansion (x : ℝ) := (x^2 - x + 1)^5

theorem coefficient_of_x_cubed : 
  (∃ a b c d e f : ℝ, expansion x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) →
  (∃ c : ℝ, c = -30 ∧ 
    ∃ a b d e f : ℝ, expansion x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_cubed_l929_92902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_in_classroom_l929_92982

theorem boys_in_classroom (total_children : ℕ) (girls_fraction : ℚ) (boys : ℕ) : 
  total_children = 45 →
  girls_fraction = 1/3 →
  boys = total_children - (girls_fraction * ↑total_children).floor →
  boys = 30 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_in_classroom_l929_92982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_e_l929_92975

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

noncomputable def x₀ : ℝ := Real.exp 1

-- State the theorem
theorem tangent_line_at_e :
  let y₀ := f x₀
  let m := deriv f x₀
  ∀ x : ℝ, (x - x₀) * m + y₀ = 1 / Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_e_l929_92975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_different_classes_three_two_l929_92941

/-- Represents a class of students -/
structure StudentClass where
  size : Nat

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : Nat) : Nat :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculates the probability of selecting students from different classes -/
def probDifferentClasses (c1 c2 : StudentClass) : Rat :=
  let totalStudents := c1.size + c2.size
  let totalWays := choose totalStudents 2
  let sameClassWays := choose c1.size 2 + choose c2.size 2
  1 - (sameClassWays : Rat) / totalWays

theorem prob_different_classes_three_two :
  let classA : StudentClass := ⟨3⟩
  let classB : StudentClass := ⟨2⟩
  probDifferentClasses classA classB = 3/5 := by
  sorry

#eval probDifferentClasses ⟨3⟩ ⟨2⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_different_classes_three_two_l929_92941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l929_92965

theorem log_equation_solution (b : ℝ) (h : Real.logb b 625 = -4/2) : b = 1/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l929_92965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_path_length_is_optimal_l929_92905

/-- A cube with side length 2 meters -/
structure Cube where
  side_length : ℝ
  side_length_eq : side_length = 2

/-- A path in the cube consisting of straight lines between corners -/
structure CubePath (cube : Cube) where
  length : ℝ
  visits_all_corners_twice : Prop
  returns_to_start : Prop
  different_sequences : Prop

/-- The maximum possible length of a valid path in the cube -/
noncomputable def max_path_length (cube : Cube) : ℝ :=
  16 * Real.sqrt 3 + 8 * Real.sqrt 2

/-- Theorem stating that the defined max_path_length is optimal -/
theorem max_path_length_is_optimal (cube : Cube) :
  ∀ (path : CubePath cube), path.length ≤ max_path_length cube :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_path_length_is_optimal_l929_92905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_three_equals_fifteen_l929_92988

/-- A quadratic function f(x) = ax^2 + bx + c with specific properties -/
noncomputable def f (x : ℝ) : ℝ := 
  let a : ℝ := 0
  let b : ℝ := 4
  let c : ℝ := 3
  a * x^2 + b * x + c

/-- Theorem stating that f(3) = 15 given the conditions -/
theorem f_of_three_equals_fifteen :
  (f 1 = 7) ∧ (f 2 = 11) → f 3 = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_three_equals_fifteen_l929_92988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l929_92961

-- Define the parabola
noncomputable def is_on_parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define the focus of the parabola
noncomputable def focus : ℝ × ℝ := (1/2, 0)

-- Define the property of points intersecting with the focus
noncomputable def intersects_focus (x y : ℝ) : Prop :=
  is_on_parabola x y ∧ (x = focus.1 ∨ y = focus.2)

theorem parabola_intersection_length 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : intersects_focus x₁ y₁) 
  (h₂ : intersects_focus x₂ y₂) 
  (h₃ : x₁ + x₂ = 3) :
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 4 := by
  sorry

#check parabola_intersection_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l929_92961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_triangle_area_theorem_l929_92937

/-- A line in 2D space represented by the equation x - 2y + 2k = 0 --/
structure Line (k : ℝ) where
  equation : ℝ → ℝ → Prop
  equation_def : equation = λ x y ↦ x - 2 * y + 2 * k = 0

/-- The area of a triangle formed by a line and the coordinate axes --/
noncomputable def triangle_area (k : ℝ) (l : Line k) : ℝ := 
  let x_intercept := -2 * k
  let y_intercept := k
  (1/2) * |x_intercept| * |y_intercept|

/-- Main theorem: If a line x - 2y + 2k = 0 forms a triangle with area 1 with the coordinate axes, then k = 1 or k = -1 --/
theorem line_triangle_area_theorem (k : ℝ) (l : Line k) : 
  triangle_area k l = 1 → k = 1 ∨ k = -1 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_triangle_area_theorem_l929_92937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l929_92967

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 3*x + 2) / Real.log 0.2

-- State the theorem
theorem f_increasing_on_interval :
  StrictMonoOn f (Set.Iio 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l929_92967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equality_l929_92980

noncomputable section

-- Define the rectangle ABCD
def A : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (3, 4)
def B : ℝ × ℝ := (C.1, A.2)
def D : ℝ × ℝ := (A.1, C.2)

-- Define point T
def T : ℝ × ℝ := (0, 10)

-- Define area functions
def rectangleArea (p q : ℝ × ℝ) : ℝ := |p.1 - q.1| * |p.2 - q.2|

def triangleArea (p q r : ℝ × ℝ) : ℝ :=
  |(p.1 * (q.2 - r.2) + q.1 * (r.2 - p.2) + r.1 * (p.2 - q.2))| / 2

-- Theorem statement
theorem area_equality :
  triangleArea B D T = rectangleArea A C := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equality_l929_92980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l929_92995

-- Define the points in the Cartesian coordinate system
def A : ℝ × ℝ := (1, 4)
def B : ℝ × ℝ := (-2, 3)
def C : ℝ × ℝ := (2, -1)

-- Define vectors
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def OC : ℝ × ℝ := C

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector addition
def vector_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Define vector magnitude
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define scalar multiplication
def scalar_mul (t : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (t * v.1, t * v.2)

-- Define the theorem
theorem vector_properties :
  dot_product AB AC = 2 ∧
  magnitude (vector_add AB AC) = 2 * Real.sqrt 10 ∧
  ∃ t : ℝ, t = -1 ∧ dot_product (vector_add AB (scalar_mul (-t) OC)) OC = 0 := by
  sorry

#check vector_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l929_92995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_elements_not_divisible_by_three_l929_92939

theorem max_elements_not_divisible_by_three (n : ℕ) (h : n = 2003) :
  (∃ (S : Finset ℕ), 
    (∀ x, x ∈ S → x ∈ Finset.range (n + 1) \ {0}) ∧
    (∀ x y, x ∈ S → y ∈ S → x ≠ y → (x + y) % 3 ≠ 0) ∧
    (∀ T : Finset ℕ, 
      (∀ x, x ∈ T → x ∈ Finset.range (n + 1) \ {0}) →
      (∀ x y, x ∈ T → y ∈ T → x ≠ y → (x + y) % 3 ≠ 0) →
      T.card ≤ S.card)) →
  (∃ (S : Finset ℕ), 
    (∀ x, x ∈ S → x ∈ Finset.range (n + 1) \ {0}) ∧
    (∀ x y, x ∈ S → y ∈ S → x ≠ y → (x + y) % 3 ≠ 0) ∧
    S.card = 669) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_elements_not_divisible_by_three_l929_92939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_bounds_and_limit_l929_92932

noncomputable def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => a n + 1 / (2 * a n)

theorem a_bounds_and_limit :
  (∀ n : ℕ, n ≥ 1 → (n : ℝ) ≤ (a n)^2 ∧ (a n)^2 < (n : ℝ) + (n : ℝ)^(1/3)) ∧
  Filter.Tendsto (λ n => a n - Real.sqrt n) Filter.atTop (nhds 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_bounds_and_limit_l929_92932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_main_result_l929_92993

def numerator_sequence (n : ℕ) : ℕ := 3 * n
def denominator_sequence (n : ℕ) : ℕ := 5 * n

def numerator_sum : ℕ := (Finset.range 15).sum numerator_sequence
def denominator_sum : ℕ := (Finset.range 10).sum denominator_sequence

theorem expression_value (x : ℝ) :
  (x ^ numerator_sum) / (x ^ denominator_sum) = x ^ (numerator_sum - denominator_sum) :=
by sorry

theorem main_result :
  (3 : ℝ) ^ numerator_sum / (3 : ℝ) ^ denominator_sum = 3 ^ (-295 : ℤ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_main_result_l929_92993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boats_initial_distance_l929_92973

/-- The initial distance between two boats --/
noncomputable def initial_distance (boat1_speed boat2_speed : ℝ) (distance_before_collision : ℝ) : ℝ :=
  distance_before_collision + (boat1_speed + boat2_speed) / 60

/-- Theorem: The initial distance between the boats is approximately 0.8666666666666666 miles --/
theorem boats_initial_distance :
  let boat1_speed : ℝ := 5
  let boat2_speed : ℝ := 21
  let distance_before_collision : ℝ := 0.43333333333333335
  ∃ ε > 0, |initial_distance boat1_speed boat2_speed distance_before_collision - 0.8666666666666666| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boats_initial_distance_l929_92973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_partition_theorem_all_partitions_valid_l929_92991

/-- Represents a sequence of white square counts in rectangles --/
def WhiteSquareSequence := List Nat

/-- Checks if a sequence is strictly increasing --/
def isStrictlyIncreasing (seq : WhiteSquareSequence) : Prop :=
  ∀ i j, i < j → i < seq.length → j < seq.length → seq.get! i < seq.get! j

/-- Checks if a sequence sums to half the chessboard squares --/
def sumsToHalfChessboard (seq : WhiteSquareSequence) : Prop :=
  seq.sum = 32

/-- Checks if each element in the sequence is at least its index (1-based) --/
def eachElementAtLeastIndex (seq : WhiteSquareSequence) : Prop :=
  ∀ i, i < seq.length → seq.get! i ≥ i + 1

/-- Represents a valid partition of the chessboard --/
structure ValidPartition where
  sequence : WhiteSquareSequence
  isValid : isStrictlyIncreasing sequence ∧
            sumsToHalfChessboard sequence ∧
            eachElementAtLeastIndex sequence

/-- The main theorem statement --/
theorem chessboard_partition_theorem :
  (∃ (p : Nat), p = 7 ∧
    (∀ (partition : ValidPartition), partition.sequence.length ≤ p) ∧
    (∃ (partitions : List ValidPartition),
      partitions.length = 5 ∧
      (∀ (partition : ValidPartition),
        partition.sequence.length = p →
        partition ∈ partitions))) := by sorry

-- Example valid partitions
def partition1 : ValidPartition := {
  sequence := [1, 2, 3, 4, 5, 6, 11],
  isValid := by sorry
}

def partition2 : ValidPartition := {
  sequence := [1, 2, 3, 4, 5, 7, 10],
  isValid := by sorry
}

def partition3 : ValidPartition := {
  sequence := [1, 2, 3, 4, 5, 8, 9],
  isValid := by sorry
}

def partition4 : ValidPartition := {
  sequence := [1, 2, 3, 4, 6, 7, 9],
  isValid := by sorry
}

def partition5 : ValidPartition := {
  sequence := [1, 2, 3, 5, 6, 7, 8],
  isValid := by sorry
}

-- List of all valid partitions
def allValidPartitions : List ValidPartition :=
  [partition1, partition2, partition3, partition4, partition5]

-- Proof that the list contains all valid partitions
theorem all_partitions_valid :
  ∀ (partition : ValidPartition),
    partition.sequence.length = 7 →
    partition ∈ allValidPartitions := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_partition_theorem_all_partitions_valid_l929_92991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_all_rectangles_equal_diagonals_l929_92974

-- Define a rectangle type
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define a function to calculate the length of a diagonal
noncomputable def diagonalLength (r : Rectangle) : ℝ :=
  Real.sqrt (r.width ^ 2 + r.height ^ 2)

-- State the theorem
theorem negation_of_all_rectangles_equal_diagonals :
  (¬ ∀ r : Rectangle, diagonalLength r = diagonalLength r) ↔
  (∃ r : Rectangle, diagonalLength r ≠ diagonalLength r) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_all_rectangles_equal_diagonals_l929_92974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_values_l929_92989

noncomputable section

-- Define the ellipse equation
def ellipse_equation (k x y : ℝ) : Prop :=
  x^2 / (k + 8) + y^2 / 9 = 1

-- Define the eccentricity
def eccentricity : ℝ := 1/2

-- Theorem statement
theorem ellipse_k_values :
  ∀ k : ℝ, (∀ x y : ℝ, ellipse_equation k x y) →
  (k = 4 ∨ k = -5/4) ↔ 
  ((∃ a b : ℝ, a > b ∧ a^2 - b^2 = (eccentricity * a)^2) ∧
   ((k + 8 = a^2 ∧ 9 = b^2) ∨ (9 = a^2 ∧ k + 8 = b^2))) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_values_l929_92989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_myCircle_composition_l929_92909

/-- The ⊕ operation -/
def myCircle (x y : ℚ) : ℚ := x * y^2 - x

/-- Theorem stating that p ⊕ (p ⊕ p) = p⁷ - 2p⁵ + p³ - p -/
theorem myCircle_composition (p : ℚ) : myCircle p (myCircle p p) = p^7 - 2*p^5 + p^3 - p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_myCircle_composition_l929_92909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l929_92908

/-- The n-th term of the series -/
def a (n : ℕ) : ℚ :=
  match n % 3 with
  | 0 => 1 / (3 * 4^(n / 3))
  | 1 => -1 / (6 * 4^(n / 3))
  | _ => -1 / (12 * 4^(n / 3))

/-- The partial sum of the first n terms of the series -/
def partial_sum (n : ℕ) : ℚ :=
  (Finset.range n).sum a

/-- The statement that the infinite series converges to 1/72 -/
theorem series_sum :
  Filter.Tendsto partial_sum Filter.atTop (nhds (1/72)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l929_92908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthographic_projection_area_l929_92933

/-- The area of the orthographic projection of an equilateral triangle --/
theorem orthographic_projection_area (side_length : ℝ) (h : side_length = 2) :
  let original_area := (Real.sqrt 3 / 4) * side_length^2
  let projection_area := (Real.sqrt 2 / 4) * original_area
  projection_area = Real.sqrt 6 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthographic_projection_area_l929_92933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_values_l929_92986

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the first line ax - y + 2a = 0 -/
noncomputable def slope1 (a : ℝ) : ℝ := a

/-- The slope of the second line (2a - 1)x + ay + a = 0 -/
noncomputable def slope2 (a : ℝ) : ℝ := -(2*a - 1) / a

theorem perpendicular_lines_a_values :
  ∀ a : ℝ, perpendicular (slope1 a) (slope2 a) → a = 1 ∨ a = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_values_l929_92986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l929_92953

theorem problem_statement (x y : ℝ) (m n : ℝ) : 
  x = 1 / (3 + 2 * Real.sqrt 2) →
  y = 1 / (3 - 2 * Real.sqrt 2) →
  m = x - ⌊x⌋ →
  n = y - ⌊y⌋ →
  (x^2 + y^2 + x*y = 35) ∧ 
  ((m + n)^2023 - (m - n)^(1/3 : ℝ) = -4 + 4 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l929_92953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l929_92917

/-- An ellipse with semi-major axis length a, semi-minor axis length b, and longest chord length 20 -/
structure Ellipse where
  a : ℝ
  b : ℝ
  longest_chord : ℝ
  chord_constraint : longest_chord = 20

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

theorem ellipse_properties (e : Ellipse) : e.a = 10 ∧ eccentricity e = Real.sqrt (1 - e.b^2 / 100) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l929_92917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l929_92950

/-- The area of a triangle given its side lengths using Heron's formula -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: The area of a triangle with sides 26, 24, and 15 is approximately 175.8 -/
theorem triangle_area_approx :
  ∃ ε > 0, |triangleArea 26 24 15 - 175.8| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l929_92950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l929_92946

theorem arithmetic_sequence_ratio (n : ℕ) (S T : ℕ → ℚ) (a b : ℕ → ℚ) 
  (h : ∀ n, S n / T n = (7 * n + 2 : ℚ) / (n + 3 : ℚ))
  (h_S : ∀ n, S n = (n : ℚ) * (a 1 + a n) / 2)
  (h_T : ∀ n, T n = (n : ℚ) * (b 1 + b n) / 2) :
  a 5 / b 5 = 65 / 12 := by
  -- We want to show that a₅/b₅ = S₉/T₉
  have h1 : a 5 / b 5 = S 9 / T 9 := by
    -- This step requires a more detailed proof about arithmetic sequences
    sorry
  
  -- Now we can use the given condition
  have h2 : S 9 / T 9 = (7 * 9 + 2 : ℚ) / (9 + 3 : ℚ) := h 9
  
  -- Combine the steps
  rw [h1, h2]
  
  -- Simplify the fraction
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l929_92946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l929_92927

theorem exponential_equation_solution :
  let f (x : ℝ) := (3 : ℝ)^(4*x^2 - 9*x + 3)
  let g (x : ℝ) := (3 : ℝ)^(-4*x^2 + 15*x - 11)
  ∀ x : ℝ, f x = g x ↔ x = (3 + Real.sqrt 2) / 2 ∨ x = (3 - Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l929_92927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l929_92918

theorem sin_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : Real.tan (π - α) + 3 = 0) :
  Real.sin α = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l929_92918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l929_92940

-- Define the functions f and g
noncomputable def f (a x : ℝ) : ℝ := x - (a + 1) * Real.log x - a / x

noncomputable def g (x : ℝ) : ℝ := (1/2) * x^2 + Real.exp x - x * Real.exp x

-- State the theorem
theorem function_inequality (a : ℝ) :
  (a < 1) →
  (∃ x₁ ∈ Set.Icc (Real.exp 1) (Real.exp 2), ∀ x₂ ∈ Set.Icc (-2) 0, f a x₁ < g x₂) →
  a ∈ Set.Ioo ((Real.exp 2 - 2*Real.exp 1) / (Real.exp 1 + 1)) 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l929_92940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_ratio_l929_92966

/-- The ratio of speeds between two trains -/
theorem train_speed_ratio :
  ∀ (speed1 speed2 : ℚ),
    speed1 = 87.5 →
    speed2 = 400 / 4 →
    speed1 / speed2 = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_ratio_l929_92966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_top_radius_larger_than_base_diameter_l929_92901

/-- Represents a truncated cone (frustum) shaped glass -/
structure Frustum where
  R : ℝ  -- radius of the top circle
  r : ℝ  -- radius of the base circle
  h : ℝ  -- height of the frustum

/-- Volume of a frustum -/
noncomputable def volume (f : Frustum) : ℝ :=
  (f.h * Real.pi / 3) * (f.R^2 + f.R * f.r + f.r^2)

/-- Condition that the volume at half height is one-third of the full volume -/
def volume_condition (f : Frustum) : Prop :=
  volume { R := (f.R + f.r) / 2, r := f.r, h := f.h / 2 } = (1/3) * volume f

/-- Theorem stating that for a frustum satisfying the volume condition, 
    the top radius is larger than the base diameter -/
theorem frustum_top_radius_larger_than_base_diameter (f : Frustum) 
  (h : volume_condition f) : f.R > 2 * f.r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_top_radius_larger_than_base_diameter_l929_92901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_A_l929_92968

def M : Finset ℕ := Finset.filter (λ n => 1 ≤ n ∧ n ≤ 10) (Finset.range 11)

def A : Finset (ℕ × ℕ × ℕ) := 
  Finset.filter (λ t : ℕ × ℕ × ℕ => t.1 ∈ M ∧ t.2.1 ∈ M ∧ t.2.2 ∈ M ∧ 9 ∣ (t.1^3 + t.2.1^3 + t.2.2^3))
    (Finset.product M (Finset.product M M))

theorem count_A : Finset.card A = 243 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_A_l929_92968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l929_92938

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- The statement to be proved -/
theorem equidistant_point (y : ℝ) : 
  let A := Point3D.mk 0 y 0
  let B := Point3D.mk 0 (-2) 4
  let C := Point3D.mk (-4) 0 4
  distance A B = distance A C ↔ y = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l929_92938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l929_92949

noncomputable def a (n : ℕ) : ℝ := 2 * n

noncomputable def S (n : ℕ) : ℝ := n^2 + n

noncomputable def b (n : ℕ) : ℝ := 2^(a n / 2)

noncomputable def S' (n : ℕ) : ℝ := 2 * (2^n - 1)

noncomputable def c (n : ℕ) : ℝ := 2 / n

noncomputable def T (n : ℕ) : ℝ := 2 * (1 + 1/2 - 1/(n+1) - 1/(n+2))

theorem arithmetic_sequence_properties 
  (h1 : a 1 = 2) 
  (h2 : ∃ r : ℝ, a 1 * r = a 4 ∧ a 4 * r = S 5 + 2) :
  (∀ n : ℕ, a n = 2 * n) ∧
  (∀ n : ℕ, S n = n^2 + n) ∧
  (∀ n : ℕ, S' n * S' (n+2) ≤ (S' (n+1))^2) ∧
  (∀ n : ℕ, n ≥ 1 → 4/3 ≤ T n ∧ T n < 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l929_92949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_BC_l929_92929

-- Define the efficiencies of workers A, B, and C
noncomputable def efficiency_A : ℝ := 1 / 104
noncomputable def efficiency_B : ℝ := 1 / 52
noncomputable def efficiency_C : ℝ := 1 / 104

-- Define the relationships between efficiencies
axiom A_half_B : efficiency_A = efficiency_B / 2
axiom B_twice_C : efficiency_B = 2 * efficiency_C

-- Define the time taken by different combinations
axiom time_ABC : 1 / (efficiency_A + efficiency_B + efficiency_C) = 26
axiom time_AB : 1 / (efficiency_A + efficiency_B) = 13
axiom time_AC : 1 / (efficiency_A + efficiency_C) = 39

-- Define constant work rate
axiom constant_rate : ∀ t > 0, (efficiency_A + efficiency_B + efficiency_C) * t = 1

-- Theorem to prove
theorem time_BC : 1 / (efficiency_B + efficiency_C) = 104 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_BC_l929_92929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_inequality_l929_92931

theorem triangle_sine_inequality (a b c : ℝ) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : a + b > c ∧ b + c > a ∧ a + c > b)
  (h3 : a + b + c ≤ 2 * Real.pi) :
  Real.sin a + Real.sin b > Real.sin c ∧
  Real.sin b + Real.sin c > Real.sin a ∧
  Real.sin a + Real.sin c > Real.sin b := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_inequality_l929_92931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_fourth_quadrant_l929_92922

theorem tan_value_fourth_quadrant (x : ℝ) :
  Real.cos x = 12/13 → x ∈ Set.Icc (3*Real.pi/2) (2*Real.pi) → Real.tan x = -5/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_fourth_quadrant_l929_92922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cooling_constant_approx_room_temp_approx_l929_92952

-- Define the water temperature function
noncomputable def water_temp (θ₀ θ₁ k t : ℝ) : ℝ := θ₀ + (θ₁ - θ₀) * Real.exp (-k * t)

-- Theorem for part 1
theorem cooling_constant_approx :
  ∃ k : ℝ, k > 0 ∧ 
  abs (k - 0.029) < 0.001 ∧
  abs (water_temp 20 98 k 60 - 71.2) < 0.001 := by
  sorry

-- Theorem for part 2
theorem room_temp_approx :
  ∃ θ₀ : ℝ, 
  abs (θ₀ - 20.0) < 0.1 ∧
  abs (water_temp θ₀ 100 0.01 150 - 40) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cooling_constant_approx_room_temp_approx_l929_92952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sausage_consumption_l929_92976

theorem sausage_consumption (total : ℚ) (remaining : ℚ) : 
  total = 600 →
  remaining = total - 2/5 * total - (total - 2/5 * total) / 2 →
  (3/4 : ℚ) = (remaining - 45) / remaining :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sausage_consumption_l929_92976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_monotonicity_l929_92972

-- Define the function f(x) = a^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- State the theorem
theorem exponential_function_monotonicity (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a (-2) < f a (-3) → 0 < a ∧ a < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_monotonicity_l929_92972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_polygons_comparison_l929_92919

/-- Two polygons inscribed in the same circle with specific side length conditions -/
structure InscribedPolygons where
  K : Set (ℝ × ℝ)
  M₁ : Set (ℝ × ℝ)
  M₂ : Set (ℝ × ℝ)
  inscribed_M₁ : M₁ ⊆ K
  inscribed_M₂ : M₂ ⊆ K
  longest_side_M₁ : ℝ
  shortest_side_M₂ : ℝ
  side_condition : longest_side_M₁ < shortest_side_M₂

/-- Perimeter of a polygon -/
noncomputable def perimeter (M : Set (ℝ × ℝ)) : ℝ := sorry

/-- Area of a polygon -/
noncomputable def area (M : Set (ℝ × ℝ)) : ℝ := sorry

/-- Main theorem: Perimeter and Area comparison of inscribed polygons -/
theorem inscribed_polygons_comparison (h : InscribedPolygons) :
  (perimeter h.M₁ > perimeter h.M₂) ∧ (area h.M₁ > area h.M₂) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_polygons_comparison_l929_92919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_length_theorem_l929_92960

/-- Represents the properties of a uniform steel rod -/
structure SteelRod where
  /-- Weight per unit length (kg/m) -/
  weightPerMeter : ℝ
  /-- Ensures the weight per meter is positive -/
  weightPerMeter_pos : weightPerMeter > 0

/-- Calculates the length of a steel rod given its weight -/
noncomputable def rodLength (rod : SteelRod) (weight : ℝ) : ℝ :=
  weight / rod.weightPerMeter

/-- Theorem stating the length of the rod weighing 42.75 kg is 11.25 m -/
theorem rod_length_theorem (rod : SteelRod) 
  (h : rodLength rod 22.8 = 6) : rodLength rod 42.75 = 11.25 := by
  sorry

#check rod_length_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_length_theorem_l929_92960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_time_l929_92916

-- Define the parameters of the journey
noncomputable def distance1 : ℝ := 80  -- miles
noncomputable def speed1 : ℝ := 50     -- mph
noncomputable def distance2 : ℝ := 100 -- miles
noncomputable def speed2 : ℝ := 75     -- mph
noncomputable def stop_time : ℝ := 0.25 -- hours (15 minutes)

-- Define the total travel time
noncomputable def total_time : ℝ := distance1 / speed1 + distance2 / speed2 + stop_time

-- Theorem statement
theorem train_journey_time : ∃ ε > 0, |total_time - 3.183| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_time_l929_92916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_division_theorem_l929_92999

/-- Given points A, B, C, and P in a vector space, where B is between A and C,
    and P extends BC such that BP:PC = 7:3, prove that P = -3/4 • B + 7/4 • C -/
theorem point_division_theorem 
  {V : Type*} [AddCommGroup V] [Module ℚ V] 
  (A B C P : V) 
  (h_between : ∃ t : ℚ, 0 < t ∧ t < 1 ∧ B = t • A + (1 - t) • C)
  (h_ratio : ∃ s : ℚ, s > 1 ∧ P = s • C + (1 - s) • B ∧ (s - 1) / 1 = 7 / 3) :
  P = (-3/4 : ℚ) • B + (7/4 : ℚ) • C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_division_theorem_l929_92999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2010_is_4_l929_92998

def customSequence : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | n + 2 => (customSequence n * customSequence (n + 1)) % 10

theorem sequence_2010_is_4 : customSequence 2009 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2010_is_4_l929_92998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_can_equalize_second_player_can_unequalize_l929_92904

structure XGame where
  cells : Fin 5 → Nat
  sum_left_diagonal : Nat
  sum_right_diagonal : Nat

def valid_game (g : XGame) : Prop :=
  (∀ i : Fin 5, g.cells i ∈ ({1, 2, 3, 4, 5} : Set Nat)) ∧
  (∀ i j : Fin 5, i ≠ j → g.cells i ≠ g.cells j) ∧
  (g.sum_left_diagonal = g.cells 0 + g.cells 2 + g.cells 4) ∧
  (g.sum_right_diagonal = g.cells 1 + g.cells 2 + g.cells 3)

theorem first_player_can_equalize :
  ∃ (first_move : Fin 5 → Nat),
    ∀ (g : XGame),
      valid_game { cells := first_move, sum_left_diagonal := g.sum_left_diagonal, sum_right_diagonal := g.sum_right_diagonal } →
      g.sum_left_diagonal = g.sum_right_diagonal :=
sorry

theorem second_player_can_unequalize :
  ∀ (first_move : Fin 5 → Nat),
    ∃ (second_move : Fin 5 → Nat),
      ∀ (g : XGame),
        valid_game { cells := first_move, sum_left_diagonal := g.sum_left_diagonal, sum_right_diagonal := g.sum_right_diagonal } →
        valid_game { cells := second_move, sum_left_diagonal := g.sum_left_diagonal, sum_right_diagonal := g.sum_right_diagonal } ∧
        g.sum_left_diagonal ≠ g.sum_right_diagonal :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_can_equalize_second_player_can_unequalize_l929_92904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l929_92913

noncomputable def f (x : ℝ) := Real.log (x^2 + 1)
noncomputable def g (x m : ℝ) := (1/2)^x - m

theorem m_range (m : ℝ) : 
  (∀ x₁ ∈ Set.Icc 0 3, ∃ x₂ ∈ Set.Icc 1 2, f x₁ ≥ g x₂ m) → m ≥ 1/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l929_92913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_in_expansion_l929_92971

/-- The coefficient of x in the expansion of (2x - 1/x)^5 -/
def coefficient_of_x : ℤ := 80

/-- The binomial expansion of (2x - 1/x)^5 -/
noncomputable def expansion (x : ℝ) : ℝ := (2*x - 1/x)^5

/-- Theorem stating that the coefficient of x in the expansion is correct -/
theorem coefficient_of_x_in_expansion :
  ∃ (a b c d f : ℝ), expansion = λ x ↦ a*x^5 + b*x^4 + c*x^3 + d*x^2 + (coefficient_of_x : ℝ)*x + f :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_in_expansion_l929_92971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l929_92951

/-- Given a right prism with a right triangular base where:
    - The hypotenuse of the base triangle has length c
    - One acute angle of the base triangle is 30°
    - A plane is drawn through the hypotenuse of the lower base and the right angle vertex of the upper base
    - This plane forms an angle of 45° with the plane of the base
    Then the volume of the triangular pyramid cut off from the prism is c³/32 -/
theorem pyramid_volume (c : ℝ) (c_pos : c > 0) : 
  (c^3 / 32 : ℝ) = c^3 / 32 :=
by
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l929_92951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l929_92957

noncomputable def projection (v : ℝ × ℝ) (u : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * u.1 + v.2 * u.2
  let magnitude_squared := u.1 * u.1 + u.2 * u.2
  (dot_product / magnitude_squared * u.1, dot_product / magnitude_squared * u.2)

theorem projection_theorem : 
  ∃ (u : ℝ × ℝ), projection (3, -3) u = (27/10, -9/10) → 
  projection (1, -1) u = (6/5, -2/5) :=
by
  sorry

#check projection_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l929_92957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_composite_sum_l929_92906

theorem infinite_composite_sum (a b c d : ℕ) :
  Set.Infinite {n : ℕ | ∃ (k : ℕ), k > 1 ∧ k ∣ (a^n + b^n + c^n + d^n)} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_composite_sum_l929_92906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complete_graph_has_hamiltonian_path_minimal_flights_in_complete_graph_l929_92987

/-- A complete graph with n vertices -/
structure CompleteGraph (n : ℕ) where
  vertices : Finset (Fin n)
  edges : Set (Fin n × Fin n)
  complete : ∀ (i j : Fin n), i ≠ j → (i, j) ∈ edges

/-- A path in a graph is a sequence of vertices where each consecutive pair is connected by an edge -/
def IsPath {n : ℕ} (G : CompleteGraph n) (path : List (Fin n)) : Prop :=
  path.Nodup ∧ 
  ∀ (i : Fin (path.length - 1)), (path[i.val], path[i.val + 1]) ∈ G.edges

/-- A Hamiltonian path visits each vertex exactly once -/
def IsHamiltonianPath {n : ℕ} (G : CompleteGraph n) (path : List (Fin n)) : Prop :=
  IsPath G path ∧ path.length = n

/-- In a complete graph with n vertices, there exists a Hamiltonian path -/
theorem complete_graph_has_hamiltonian_path (n : ℕ) (G : CompleteGraph n) :
  ∃ (path : List (Fin n)), IsHamiltonianPath G path := by
  sorry

/-- The minimal number of flights needed in a complete graph is zero -/
theorem minimal_flights_in_complete_graph (n : ℕ) (G : CompleteGraph n) :
  ∃ (path : List (Fin n)), IsHamiltonianPath G path ∧ (∀ i : Fin (path.length - 1), (path[i.val], path[i.val + 1]) ∈ G.edges) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complete_graph_has_hamiltonian_path_minimal_flights_in_complete_graph_l929_92987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l929_92942

-- Define the function f
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - x + c

-- Theorem statement
theorem function_properties (c : ℝ) :
  -- Part 1: Maximum and minimum values on [0, 1]
  (∀ x, x ∈ Set.Icc 0 1 → f c x ≤ c) ∧
  (∃ x, x ∈ Set.Icc 0 1 ∧ f c x = c) ∧
  (∀ x, x ∈ Set.Icc 0 1 → f c x ≥ c - 1/4) ∧
  (∃ x, x ∈ Set.Icc 0 1 ∧ f c x = c - 1/4) ∧
  -- Part 2: Bound on the difference of function values
  (∀ x₁ x₂, x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 → |f c x₁ - f c x₂| ≤ 1/4) ∧
  -- Part 3: Range of c when f has 2 zeros in [0, 2]
  ((∃ x₁ x₂, x₁ ∈ Set.Icc 0 2 ∧ x₂ ∈ Set.Icc 0 2 ∧ x₁ ≠ x₂ ∧ f c x₁ = 0 ∧ f c x₂ = 0) → 0 ≤ c ∧ c < 1/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l929_92942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sin_cos_equality_l929_92911

open Real

theorem tan_sin_cos_equality : 
  tan (40 * π / 180) + 3 * sin (40 * π / 180) + 2 * cos (20 * π / 180) = 
  4 * sin (55 * π / 180) * cos (15 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sin_cos_equality_l929_92911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_and_intersection_l929_92963

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log ((x - 1) / x)}
def N : Set ℝ := {y | ∃ x, y = x^2 + 2*x + 2}

-- State the theorem
theorem set_equality_and_intersection :
  (M = (Set.Ioi 1 ∪ Set.Iio 0)) ∧
  ((Set.univ \ M) ∩ N = {1}) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_and_intersection_l929_92963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_l929_92997

-- Define the line and circle equations
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y = 12
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Theorem: The line and circle have no intersection points
theorem no_intersection :
  ¬∃ (x y : ℝ), line_eq x y ∧ circle_eq x y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_l929_92997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_l929_92978

noncomputable def f (x : ℝ) := -1/x - 1

theorem f_increasing : 
  ∀ x₁ x₂ : ℝ, 0 < x₂ → x₂ < x₁ → f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_l929_92978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l929_92912

/-- The function g(x) = (x^3 - 2x) / (x^2 - 2x + 2) -/
noncomputable def g (x : ℝ) : ℝ := (x^3 - 2*x) / (x^2 - 2*x + 2)

/-- The range of g is all real numbers -/
theorem range_of_g : ∀ z : ℝ, ∃ x : ℝ, g x = z := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l929_92912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_7_area_l929_92959

/-- The side length of the nth square in the sequence -/
noncomputable def square_side_length (n : ℕ) : ℝ :=
  4 * (1 / Real.sqrt 2) ^ (n - 1)

/-- The radius of the circle passing through the vertices of the nth square -/
noncomputable def circle_radius (n : ℕ) : ℝ :=
  square_side_length n / Real.sqrt 2

/-- The area of the circle passing through the vertices of the nth square -/
noncomputable def circle_area (n : ℕ) : ℝ :=
  Real.pi * (circle_radius n) ^ 2

theorem circle_7_area :
  circle_area 7 = Real.pi / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_7_area_l929_92959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_fourth_equals_twenty_ninths_l929_92925

-- Define the functions g and f
noncomputable def g (x : ℝ) : ℝ := 1 - x^2

noncomputable def f (x : ℝ) : ℝ :=
  if x = 0 then 0  -- Arbitrary value for x = 0
  else (2 - (Real.sqrt (1 - x))^2) / (Real.sqrt (1 - x))^4

-- State the theorem
theorem f_one_fourth_equals_twenty_ninths :
  f (1/4 : ℝ) = 20/9 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_fourth_equals_twenty_ninths_l929_92925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_example_l929_92930

/-- Dilation of a complex number -/
noncomputable def dilation (center : ℂ) (scale : ℝ) (z : ℂ) : ℂ :=
  center + scale • (z - center)

theorem dilation_example : dilation (1 - 2*I) 3 (-1 + I) = -5 + 7*I := by
  -- Unfold the definition of dilation
  unfold dilation
  -- Simplify the expression
  simp [Complex.I]
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_example_l929_92930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_truck_ratio_l929_92903

/-- Given a road with 4 lanes, 60 trucks in each lane, and a total of 2160 vehicles,
    prove that the ratio of cars in each lane to trucks in all lanes is 2:1 -/
theorem car_truck_ratio (num_lanes : ℕ) (trucks_per_lane : ℕ) (total_vehicles : ℕ)
  (h_lanes : num_lanes = 4)
  (h_trucks : trucks_per_lane = 60)
  (h_total : total_vehicles = 2160) :
  (total_vehicles - num_lanes * trucks_per_lane) / num_lanes = 2 * trucks_per_lane := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_truck_ratio_l929_92903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_g_simplification_g_min_on_interval_g_max_on_interval_l929_92936

noncomputable def f (x : ℝ) : ℝ := 1 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 2 * (Real.sin x) ^ 2

noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi / 6)

theorem f_simplification (x : ℝ) : f x = 2 * Real.sin (2 * x + Real.pi / 6) := by sorry

theorem g_simplification (x : ℝ) : g x = 2 * Real.sin (2 * x - Real.pi / 6) := by sorry

theorem g_min_on_interval : 
  ∃ x₀ ∈ Set.Icc (-Real.pi / 2) 0, ∀ x ∈ Set.Icc (-Real.pi / 2) 0, g x₀ ≤ g x ∧ g x₀ = -2 := by sorry

theorem g_max_on_interval : 
  ∃ x₀ ∈ Set.Icc (-Real.pi / 2) 0, ∀ x ∈ Set.Icc (-Real.pi / 2) 0, g x ≤ g x₀ ∧ g x₀ = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_g_simplification_g_min_on_interval_g_max_on_interval_l929_92936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_extrema_l929_92944

theorem sum_of_extrema :
  ∃ (a b : ℝ), 
    (∀ (x y : ℝ), 1 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 4 → x^2 - x*y + y^2 ≥ a) ∧
    (∀ (x y : ℝ), 1 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 4 → x^2 - x*y + y^2 ≤ b) ∧
    (a + b = 13/2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_extrema_l929_92944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifi_closet_hangers_l929_92924

def total_hangers (pink green blue yellow orange purple red brown gray : ℕ) : ℕ :=
  pink + green + blue + yellow + orange + purple + red + brown + gray

theorem fifi_closet_hangers : ∃ (pink green blue yellow orange purple red brown gray : ℕ),
  pink = 7 ∧
  green = 4 ∧
  blue = green - 1 ∧
  yellow = blue - 1 ∧
  orange = 2 * (pink + green) ∧
  purple = (blue - yellow) + 3 ∧
  red = Int.toNat (Int.floor ((pink + green + blue : ℚ) / 3 + 1/2)) ∧
  brown = 3 * red + 1 ∧
  gray = Int.toNat (Int.floor ((3 : ℚ) / 5 * purple + 1/2)) ∧
  total_hangers pink green blue yellow orange purple red brown gray = 65 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifi_closet_hangers_l929_92924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gravel_and_tank_volume_l929_92954

noncomputable def cone_volume (r h : ℝ) : ℝ := (Real.pi * r^2 * h) / 3

noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

theorem gravel_and_tank_volume :
  let cone_diameter : ℝ := 12
  let cone_height : ℝ := 0.6 * cone_diameter
  let cone_radius : ℝ := cone_diameter / 2
  let cylinder_height : ℝ := 2 * cone_height
  let total_volume : ℝ := cone_volume cone_radius cone_height + cylinder_volume cone_radius cylinder_height
  total_volume = 604.8 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gravel_and_tank_volume_l929_92954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seaplane_trip_average_speed_l929_92956

/-- Represents a flight with speed and duration -/
structure Flight where
  speed : ℝ
  duration : ℝ

/-- Calculates the average speed given a list of flights and the total distance -/
noncomputable def averageSpeed (flights : List Flight) (totalDistance : ℝ) : ℝ :=
  totalDistance / (flights.map (λ f => f.duration)).sum

/-- The problem statement -/
theorem seaplane_trip_average_speed :
  let outboundFlight1 : Flight := { speed := 140, duration := 2 }
  let outboundFlight2 : Flight := { speed := 88, duration := 1.5 }
  let returnFlight : Flight := { speed := 73, duration := 3.5 }
  let flights := [outboundFlight1, outboundFlight2, returnFlight]
  let totalDistance := 667.5
  abs (averageSpeed flights totalDistance - 95.36) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seaplane_trip_average_speed_l929_92956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_link_hinge_distance_bounds_l929_92943

/-- A two-link hinge with given lengths -/
structure TwoLinkHinge where
  length1 : ℝ
  length2 : ℝ

/-- The possible distance between the free ends of a two-link hinge -/
noncomputable def distance (h : TwoLinkHinge) : Set ℝ :=
  {d | ∃ θ : ℝ, d = Real.sqrt ((h.length1 + h.length2 * Real.cos θ)^2 + (h.length2 * Real.sin θ)^2)}

/-- Theorem stating the minimum and maximum distances for a specific two-link hinge -/
theorem two_link_hinge_distance_bounds :
  let h : TwoLinkHinge := ⟨5, 3⟩
  (∀ d ∈ distance h, 2 ≤ d ∧ d ≤ 8) ∧
  (2 ∈ distance h) ∧
  (8 ∈ distance h) := by
  sorry

#check two_link_hinge_distance_bounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_link_hinge_distance_bounds_l929_92943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_chocolates_sold_l929_92948

-- Define the variables
variable (C : ℝ) -- Cost price of one chocolate
variable (S : ℝ) -- Selling price of one chocolate
variable (n : ℝ) -- Number of chocolates sold at selling price

-- Define the conditions
def condition1 (C S n : ℝ) : Prop := 165 * C = n * S
def condition2 (C S : ℝ) : Prop := S = 1.1 * C

-- Theorem statement
theorem number_of_chocolates_sold (h1 : condition1 C S n) (h2 : condition2 C S) : n = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_chocolates_sold_l929_92948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_intervals_l929_92928

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x^2 - 5*x + 2*Real.log x

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := (2*x^2 - 5*x + 2) / x

-- Theorem statement
theorem f_monotone_increasing_intervals :
  ∀ x : ℝ, x > 0 →
    (f_derivative x > 0 ↔ (x ∈ Set.Ioo 0 (1/2) ∨ x ∈ Set.Ioi 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_intervals_l929_92928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiling_with_square_and_pentagon_l929_92979

/-- The number of sides of the third regular polygon when tiling with a square and a regular pentagon -/
def third_polygon_sides : ℕ := 20

/-- The interior angle of a regular polygon with n sides -/
noncomputable def interior_angle (n : ℕ) : ℝ := (n - 2 : ℝ) * 180 / n

/-- The sum of interior angles at a vertex when tiling with three regular polygons -/
noncomputable def angle_sum (n : ℕ) : ℝ := interior_angle 4 + interior_angle 5 + interior_angle n

/-- Theorem stating that the sum of interior angles at a vertex is 360° when tiling with a square, 
    a regular pentagon, and a regular polygon with third_polygon_sides sides -/
theorem tiling_with_square_and_pentagon :
  angle_sum third_polygon_sides = 360 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiling_with_square_and_pentagon_l929_92979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_rational_non_integer_solution_l929_92992

/-- The radius of the cylindrical box -/
noncomputable def R : ℝ := 8

/-- The height of the cylindrical box -/
noncomputable def H : ℝ := 3

/-- The volume of a cylinder given radius r and height h -/
noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- Theorem stating that there exists a unique rational, non-integer x > 0 satisfying the volume increase condition -/
theorem unique_rational_non_integer_solution :
  ∃! x : ℚ, x > 0 ∧ ¬(∃ n : ℤ, (x : ℝ) = n) ∧
    cylinderVolume (R + x) H - cylinderVolume R H = 
    cylinderVolume R (H + x) - cylinderVolume R H :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_rational_non_integer_solution_l929_92992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rational_points_on_circle_l929_92969

-- Define the circle C
def C (r : ℝ) := {p : ℝ × ℝ | (p.1 - Real.sqrt 2)^2 + (p.2 - Real.sqrt 3)^2 = r^2}

-- Define a rational point
def is_rational_point (p : ℝ × ℝ) : Prop := ∃ (q1 q2 : ℚ), p = (↑q1, ↑q2)

-- Theorem statement
theorem max_rational_points_on_circle (r : ℝ) (hr : r > 0) :
  ∃ (n : ℕ), n ≤ 1 ∧ ∀ (S : Finset (ℝ × ℝ)), 
    (∀ p ∈ S, p ∈ C r ∧ is_rational_point p) → Finset.card S ≤ n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rational_points_on_circle_l929_92969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_percent_calculation_l929_92996

theorem gain_percent_calculation (MP : ℝ) (MP_pos : MP > 0) : 
  let CP := 0.55 * MP
  let discount := 0.15 * MP
  let SP := MP - discount
  let gain := SP - CP
  let gain_percent := (gain / CP) * 100
  ∃ ε > 0, |gain_percent - 54.55| < ε := by
  sorry

#check gain_percent_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_percent_calculation_l929_92996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_rotations_approx_1020_l929_92935

/-- The number of complete rotations of a bicycle wheel -/
noncomputable def wheel_rotations (wheel_diameter : ℝ) (cycling_time_minutes : ℝ) (speed_km_per_hour : ℝ) : ℝ :=
  let cycling_time_hours := cycling_time_minutes / 60
  let distance_km := speed_km_per_hour * cycling_time_hours
  let distance_m := distance_km * 1000
  let wheel_circumference := Real.pi * wheel_diameter
  distance_m / wheel_circumference

/-- Theorem stating that the number of wheel rotations is approximately 1020 -/
theorem wheel_rotations_approx_1020 :
  ∃ ε > 0, |wheel_rotations 0.75 6 24 - 1020| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_rotations_approx_1020_l929_92935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_four_l929_92977

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2:ℝ)^(-x) * (1 - a^x)

-- State the theorem
theorem odd_function_implies_a_equals_four (a : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x, f a x = -f a (-x)) : a = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_four_l929_92977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hari_joined_after_five_months_l929_92921

/-- Represents the business partnership details -/
structure BusinessPartnership where
  praveen_investment : ℚ
  hari_investment : ℚ
  profit_ratio_praveen : ℚ
  profit_ratio_hari : ℚ
  total_duration : ℚ

/-- Calculates the number of months after which Hari joined the business -/
noncomputable def months_until_hari_joined (bp : BusinessPartnership) : ℚ :=
  bp.total_duration - (bp.profit_ratio_hari * bp.praveen_investment * bp.total_duration) / 
    (bp.profit_ratio_praveen * bp.hari_investment)

/-- Theorem stating that Hari joined approximately 5 months after Praveen started -/
theorem hari_joined_after_five_months (bp : BusinessPartnership) 
    (h1 : bp.praveen_investment = 3920)
    (h2 : bp.hari_investment = 10080)
    (h3 : bp.profit_ratio_praveen = 2)
    (h4 : bp.profit_ratio_hari = 3)
    (h5 : bp.total_duration = 12) :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/10 ∧ |months_until_hari_joined bp - 5| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hari_joined_after_five_months_l929_92921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_possible_values_l929_92990

/-- An arithmetic progression of integers -/
def ArithmeticProgression (a₁ : ℤ) (d : ℤ) : ℕ → ℤ
  | 0 => a₁
  | n + 1 => ArithmeticProgression a₁ d n + d

/-- Sum of the first n terms of an arithmetic progression -/
def SumArithmeticProgression (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_progression_possible_values (a₁ : ℤ) (d : ℤ) :
  d > 0 →
  (let S := SumArithmeticProgression a₁ d 15
   let a₇ := ArithmeticProgression a₁ d 6
   let a₁₁ := ArithmeticProgression a₁ d 10
   let a₁₂ := ArithmeticProgression a₁ d 11
   let a₁₆ := ArithmeticProgression a₁ d 15
   a₇ * a₁₆ > S - 24 ∧ a₁₁ * a₁₂ < S + 4) →
  a₁ ∈ ({-5, -4, -2, -1} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_possible_values_l929_92990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l929_92970

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := λ x => 
  if x ≥ 0 then x^2 - 6*x + a else -((-x)^2 - 6*(-x) + a)

-- State the theorem
theorem solution_set_of_inequality (a : ℝ) :
  (∀ x, f a (-x) = -(f a x)) →  -- f is odd
  (∃ y, y ≥ 0 ∧ f a y = y^2 - 6*y + a) →  -- definition of f for x ≥ 0
  {x : ℝ | f a x < |x|} = Set.Ioi (-5) ∪ Set.Ioo 0 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l929_92970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_minimum_value_l929_92915

noncomputable def a (n : ℕ) : ℚ :=
  if n = 0 then 0
  else if n = 1 then 60
  else a (n - 1) + 2 * (n - 1)

theorem sequence_minimum_value :
  ∃ (m : ℕ), m > 0 ∧ ∀ (n : ℕ), n > 0 → a n / n ≥ 29 / 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_minimum_value_l929_92915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_saree_final_price_l929_92985

/-- Calculate the final price after successive discounts -/
noncomputable def final_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  let price_after_first_discount := original_price * (1 - discount1 / 100)
  price_after_first_discount * (1 - discount2 / 100)

/-- Theorem stating the final price of the saree after discounts -/
theorem saree_final_price :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |final_price 298 12 15 - 223| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_saree_final_price_l929_92985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crow_eating_time_l929_92945

/-- Represents the time taken for a crow to eat a portion of nuts -/
noncomputable def time_to_eat (total_nuts : ℝ) (eating_rate : ℝ) (portion : ℝ) : ℝ :=
  (portion * total_nuts) / eating_rate

/-- Theorem stating the time taken for a crow to eat a portion of nuts -/
theorem crow_eating_time (total_nuts : ℝ) (portion : ℝ) 
  (h1 : total_nuts > 0) (h2 : 0 ≤ portion ∧ portion ≤ 1) :
  time_to_eat total_nuts ((1/5) * total_nuts / 8) portion = 40 * portion := by
  sorry

#check crow_eating_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crow_eating_time_l929_92945
