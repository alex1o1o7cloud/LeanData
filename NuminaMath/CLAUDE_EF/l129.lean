import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l129_12977

-- Define the function as noncomputable due to its dependence on Real.log
noncomputable def f (x : ℝ) : ℝ := (Real.log (x + 1)) / (x - 1) + 2^(x - 2)

-- Define the domain
def domain : Set ℝ := {x | x > -1 ∧ x ≠ 1}

-- Theorem statement
theorem f_domain : 
  ∀ x, x ∈ domain ↔ (∃ y, f x = y) :=
by
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l129_12977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_approximation_w_l129_12952

theorem closest_approximation_w : 
  ∃ (w : ℝ), (∀ (x : ℝ), abs ((69.28 * 0.004) / 0.03 - w) ≤ abs ((69.28 * 0.004) / 0.03 - x)) ∧ 
  (w = 9.24) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_approximation_w_l129_12952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_difference_l129_12967

/-- The speed of the first train in km/hr -/
noncomputable def speed1 : ℝ := 60

/-- The time taken by the first train to cross the pole in seconds -/
noncomputable def time1 : ℝ := 3

/-- The speed of the second train in km/hr -/
noncomputable def speed2 : ℝ := 90

/-- The time taken by the second train to cross the pole in seconds -/
noncomputable def time2 : ℝ := 2

/-- Conversion factor from km/hr to m/s -/
noncomputable def km_hr_to_m_s : ℝ := 5 / 18

theorem train_length_difference :
  ∃ (length1 length2 : ℝ),
    length1 = speed1 * km_hr_to_m_s * time1 ∧
    length2 = speed2 * km_hr_to_m_s * time2 ∧
    |length2 - length1| = 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_difference_l129_12967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cylinder_formulas_l129_12950

/-- Represents a truncated cylinder --/
structure TruncatedCylinder where
  r : ℝ  -- base radius
  a : ℝ  -- shortest side length
  b : ℝ  -- longest side length
  h_positive : r > 0 ∧ a > 0 ∧ b > 0  -- ensure positive dimensions
  h_a_le_b : a ≤ b  -- ensure a is the shortest and b is the longest

/-- The volume of a truncated cylinder --/
noncomputable def volume (c : TruncatedCylinder) : ℝ :=
  c.r^2 * Real.pi * (c.a + c.b) / 2

/-- The lateral surface area of a truncated cylinder --/
noncomputable def lateralSurfaceArea (c : TruncatedCylinder) : ℝ :=
  Real.pi * c.r * (c.a + c.b)

/-- Theorem stating the correctness of volume and lateral surface area formulas --/
theorem truncated_cylinder_formulas (c : TruncatedCylinder) :
  volume c = c.r^2 * Real.pi * (c.a + c.b) / 2 ∧
  lateralSurfaceArea c = Real.pi * c.r * (c.a + c.b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cylinder_formulas_l129_12950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hilt_travel_theorem_l129_12974

/-- Calculates the miles traveled per book given the total distance and number of books read. -/
noncomputable def miles_per_book (total_distance : ℝ) (num_books : ℕ) : ℝ :=
  total_distance / (num_books : ℝ)

/-- Proves that Mrs. Hilt traveled approximately 450.67 miles per book. -/
theorem hilt_travel_theorem :
  let total_distance : ℝ := 6760
  let num_books : ℕ := 15
  let result := miles_per_book total_distance num_books
  ‖result - 450.67‖ < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hilt_travel_theorem_l129_12974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_top_circle_values_l129_12902

def is_valid_arrangement (a b c d e f : ℕ) : Prop :=
  a ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧
  b ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧
  c ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧
  d ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧
  e ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧
  f ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f ∧
  a = Int.natAbs (b - c) ∧
  b = Int.natAbs (d - e) ∧
  c = Int.natAbs (e - f)

theorem top_circle_values :
  ∀ a b c d e f : ℕ, is_valid_arrangement a b c d e f → a ∈ ({1, 2, 3} : Finset ℕ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_top_circle_values_l129_12902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_length_l129_12982

/-- 
Given an arithmetic sequence with:
- First term a₁ = -6
- Last term aₙ = 39
- Common difference d = 5
Prove that the sequence contains exactly 10 terms.
-/
theorem arithmetic_sequence_length : 
  ∀ (n : ℕ), n > 0 → 
  (let a : ℕ → ℤ := λ i => -6 + (i - 1) * 5
   a 1 = -6 ∧ a n = 39) → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_length_l129_12982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_problem_solution_l129_12995

/-- Represents the speed of a train in kilometers per hour. -/
def Speed := ℝ

/-- Represents time in hours. -/
def Time := ℝ

/-- Represents distance in kilometers. -/
def Distance := ℝ

/-- The problem setup for two trains meeting and reaching their destinations. -/
structure TrainProblem where
  speed_A : Speed
  speed_B : Speed
  time_B_after_meeting : Time
  time_A_after_meeting : Time

/-- The solution to the train problem. -/
def solve_train_problem (p : TrainProblem) : Prop :=
  p.speed_A = (60 : ℝ) ∧
  p.speed_B = (90 : ℝ) ∧
  p.time_B_after_meeting = (4 : ℝ) ∧
  p.time_A_after_meeting = (16 : ℝ)

theorem train_problem_solution (p : TrainProblem) :
  solve_train_problem p →
  p.time_A_after_meeting = (16 : ℝ) := by
  intro h
  exact h.right.right.right


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_problem_solution_l129_12995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l129_12954

/-- An ellipse centered at the origin with major axis along the x-axis -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  h : a > b ∧ b > 0

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- A point on an ellipse -/
noncomputable def Ellipse.point (e : Ellipse) (t : ℝ) : ℝ × ℝ :=
  (e.a * Real.cos t, e.b * Real.sin t)

/-- The foci of an ellipse -/
noncomputable def Ellipse.foci (e : Ellipse) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := Real.sqrt (e.a^2 - e.b^2)
  ((-c, 0), (c, 0))

/-- The area of a triangle given its three vertices -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The vector subtraction of two 2D points -/
def vector_sub (A B : ℝ × ℝ) : ℝ × ℝ := (A.1 - B.1, A.2 - B.2)

theorem ellipse_properties (e : Ellipse) 
  (h1 : e.point (3*π/2) = (0, -Real.sqrt 5))
  (h2 : e.eccentricity = Real.sqrt 6 / 6) :
  (∀ x y, x^2/6 + y^2/5 = 1 ↔ e.point (Real.arccos (x/e.a)) = (x, y)) ∧ 
  (∀ M : ℝ × ℝ, (∃ t, e.point t = M) → 
    area_triangle M e.foci.1 e.foci.2 ≤ Real.sqrt 5) ∧
  (∀ P : ℝ × ℝ, (∃ t, e.point t = P) → 
    dot_product (vector_sub P e.foci.1) (vector_sub P e.foci.2) ≠ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l129_12954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculator_sum_approximation_l129_12901

/-- The number of participants in the circle -/
def num_participants : ℕ := 50

/-- The initial value of the first calculator -/
def calc1_initial : ℤ := 2

/-- The initial value of the second calculator -/
def calc2_initial : ℤ := -2

/-- The initial value of the third calculator -/
def calc3_initial : ℤ := 0

/-- The operation performed on the first calculator -/
def calc1_op (n : ℤ) : ℤ := 2 * n

/-- The operation performed on the second calculator -/
def calc2_op (n : ℤ) : ℤ := n * n

/-- The operation performed on the third calculator -/
def calc3_op (n : ℤ) : ℤ := n - 1

/-- The final value of the first calculator after 50 operations -/
def calc1_final : ℤ := calc1_initial * (2 ^ num_participants)

/-- The final value of the second calculator after 50 operations -/
noncomputable def calc2_final : ℤ := calc2_initial ^ (2 ^ num_participants)

/-- The final value of the third calculator after 50 operations -/
def calc3_final : ℤ := calc3_initial - num_participants

/-- The theorem stating that the sum of the final calculator values is approximately 2^(2^50) -/
theorem calculator_sum_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |((calc1_final : ℝ) + (calc2_final : ℝ) + (calc3_final : ℝ)) - (2 : ℝ) ^ (2 ^ num_participants)| < ε * (2 : ℝ) ^ (2 ^ num_participants) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculator_sum_approximation_l129_12901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_reading_time_l129_12955

theorem book_reading_time (chapters : ℕ) (pages_per_chapter : ℕ) (illustrations_per_chapter : ℕ)
  (minutes_per_page : ℕ) (minutes_per_illustration : ℕ) 
  (h1 : chapters = 25)
  (h2 : pages_per_chapter = 12)
  (h3 : illustrations_per_chapter = 3)
  (h4 : minutes_per_page = 10)
  (h5 : minutes_per_illustration = 5) :
  chapters * pages_per_chapter * minutes_per_page +
  chapters * illustrations_per_chapter * minutes_per_illustration = 3375 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_reading_time_l129_12955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_B_empty_time_l129_12943

-- Define the fill rate of pipe A
noncomputable def fill_rate_A : ℝ := 1 / 6

-- Define the time when pipe B is closed
def close_time_B : ℝ := 96

-- Define the total time to fill the tank
def total_fill_time : ℝ := 30

-- Define the unknown empty rate of pipe B
noncomputable def empty_rate_B : ℝ := 1 / 24

-- Theorem statement
theorem pipe_B_empty_time : 
  (fill_rate_A - empty_rate_B) * close_time_B + fill_rate_A * (total_fill_time - close_time_B) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_B_empty_time_l129_12943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_three_match_winners_200_l129_12917

/-- Represents a single-elimination tournament -/
structure Tournament where
  participants : ℕ

/-- The number of losers in a tournament is one less than the number of participants -/
def losers_count (t : Tournament) : ℕ := t.participants - 1

/-- The maximum number of participants who can win at least three matches -/
def max_three_match_winners (t : Tournament) : ℕ :=
  (losers_count t) / 3

/-- Theorem: In a tournament with 200 participants, the maximum number of participants
    who can win at least three matches is 66 -/
theorem max_three_match_winners_200 (t : Tournament) (h : t.participants = 200) :
  max_three_match_winners t = 66 := by
  -- Proof goes here
  sorry

#eval max_three_match_winners ⟨200⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_three_match_winners_200_l129_12917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_ray_equation_l129_12936

/-- Helper function to compute the reflection of a point across a line -/
def reflection_point (P : ℝ × ℝ) (line : Set (ℝ × ℝ)) : ℝ × ℝ :=
sorry

/-- Given a ray of light from point (2, 3) reflected off the line x + y = -1
    and passing through point (1, 1), the equation of the reflected ray is 4x - 5y + 1 = 0 -/
theorem reflected_ray_equation :
  ∀ (x y : ℝ),
  let P : ℝ × ℝ := (2, 3)
  let Q : ℝ × ℝ := (1, 1)
  let reflecting_line : Set (ℝ × ℝ) := {(a, b) | a + b = -1}
  let reflected_point := reflection_point P reflecting_line
  let reflected_ray : Set (ℝ × ℝ) := {(a, b) | ∃ t : ℝ, (a, b) = Q + t • (Q - reflected_point)}
  (x, y) ∈ reflected_ray ↔ 4*x - 5*y + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_ray_equation_l129_12936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_over_2_l129_12913

noncomputable section

/-- The function f(x) with parameters ω and b -/
def f (ω b : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4) + b

/-- The period of the function f -/
def T (ω : ℝ) : ℝ := 2 * Real.pi / ω

theorem f_value_at_pi_over_2 (ω b : ℝ) (h1 : ω > 0) 
  (h2 : 2 * Real.pi / 3 < T ω) (h3 : T ω < Real.pi)
  (h4 : f ω b (3 * Real.pi / 2) = 2) 
  (h5 : ∀ x, f ω b (3 * Real.pi - x) = f ω b x) : 
  f ω b (Real.pi / 2) = 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_over_2_l129_12913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l129_12972

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a point is outside a circle -/
def isOutside (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 > c.radius^2

/-- Predicate to check if a line intersects a circle -/
def intersects (l : Line) (c : Circle) : Prop :=
  let d := |l.a * c.center.x + l.b * c.center.y + l.c| / Real.sqrt (l.a^2 + l.b^2)
  d < c.radius

/-- Theorem: If a point P(a,b) is outside the unit circle centered at origin,
    then the line ax + by + 1 = 0 intersects the circle -/
theorem line_intersects_circle (a b : ℝ) :
  let p : Point := ⟨a, b⟩
  let c : Circle := ⟨⟨0, 0⟩, 1⟩
  let l : Line := ⟨a, b, 1⟩
  isOutside p c → intersects l c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l129_12972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonious_range_l129_12925

-- Define harmonious functions
def harmonious (f g : ℝ → ℝ) : Prop :=
  ∀ x : ℕ+, f (x : ℝ) * g (x : ℝ) ≥ 0

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - 20
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem harmonious_range (a : ℝ) :
  (harmonious (f a) (g a)) ↔ (4 ≤ a ∧ a ≤ 5) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonious_range_l129_12925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l129_12938

/-- An arithmetic sequence with first term 2 and non-zero common difference -/
def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  a 1 = 2 ∧ d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℚ) : Prop :=
  y^2 = x * z

/-- Sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

theorem arithmetic_sequence_sum (a : ℕ → ℚ) (d : ℚ) :
  arithmetic_sequence a d →
  geometric_sequence (a 1) (a 3) (a 6) →
  arithmetic_sum a 9 = 36 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l129_12938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_comparison_of_logarithmic_and_root_expressions_l129_12933

theorem comparison_of_logarithmic_and_root_expressions :
  2 + Real.log 6 / Real.log 2 > 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_comparison_of_logarithmic_and_root_expressions_l129_12933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinning_class_frequency_l129_12971

/-- Represents the number of spinning classes per week -/
noncomputable def classes_per_week (calories_per_minute : ℝ) (class_duration_hours : ℝ) (total_calories_per_week : ℝ) : ℝ :=
  total_calories_per_week / (calories_per_minute * class_duration_hours * 60)

theorem spinning_class_frequency 
  (calories_per_minute : ℝ) 
  (class_duration_hours : ℝ) 
  (total_calories_per_week : ℝ) 
  (h1 : calories_per_minute = 7)
  (h2 : class_duration_hours = 1.5)
  (h3 : total_calories_per_week = 1890) :
  classes_per_week calories_per_minute class_duration_hours total_calories_per_week = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinning_class_frequency_l129_12971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l2_equation_l129_12923

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the angle of inclination of a line --/
noncomputable def angle_of_inclination (l : Line) : ℝ := Real.arctan (l.a / l.b)

/-- Checks if a point lies on a line --/
def point_on_line (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Theorem stating the equation of line l2 given the conditions --/
theorem line_l2_equation (l1 l2 : Line) :
  l1.a = 2 ∧ l1.b = -1 ∧ l1.c = 1 →
  point_on_line l2 1 1 →
  angle_of_inclination l2 = 2 * angle_of_inclination l1 →
  l2.a = 4 ∧ l2.b = 3 ∧ l2.c = -7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l2_equation_l129_12923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fabric_weighted_average_profit_margin_l129_12940

/-- Calculates the weighted average profit margin for fabric sales -/
theorem fabric_weighted_average_profit_margin
  (profit_margin_A profit_margin_B profit_margin_C : ℝ)
  (quantity_A quantity_B quantity_C : ℝ)
  (h_profit_A : profit_margin_A = 0.20)
  (h_profit_B : profit_margin_B = 0.25)
  (h_profit_C : profit_margin_C = 0.30)
  (h_quantity_A : quantity_A = 10)
  (h_quantity_B : quantity_B = 10)
  (h_quantity_C : quantity_C = 10) :
  (profit_margin_A * quantity_A + profit_margin_B * quantity_B + profit_margin_C * quantity_C) /
  (quantity_A + quantity_B + quantity_C) = 0.25 := by
  sorry

#check fabric_weighted_average_profit_margin

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fabric_weighted_average_profit_margin_l129_12940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotonic_iff_a_in_range_l129_12962

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 2 then 2 * x^2 + 10
  else (3 - a) * 3^x

-- State the theorem
theorem function_monotonic_iff_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ 1 ≤ a ∧ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotonic_iff_a_in_range_l129_12962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_l129_12990

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def x (n : ℕ+) : ℕ := (floor (n / 5 : ℝ)).toNat

theorem sum_of_sequence (n : ℕ) : 
  (Finset.range (5 * n)).sum (λ i => x ⟨i + 1, Nat.succ_pos i⟩) = 
    (5 * n ^ 2 - 3 * n) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_l129_12990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_alternate_angles_equal_implies_parallel_l129_12983

/-- Two lines in a plane -/
structure Line where

/-- A transversal line crossing two other lines -/
structure Transversal where

/-- An angle formed by two lines -/
structure Angle where

/-- Represents two interior alternate angles formed by a transversal crossing two lines -/
structure InteriorAlternateAngles (l1 l2 : Line) (t : Transversal) where
  angle1 : Angle
  angle2 : Angle

/-- Predicate to check if two angles are equal -/
def angles_equal (a1 a2 : Angle) : Prop := sorry

/-- Predicate to check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop := sorry

/-- Theorem: If interior alternate angles formed by a transversal crossing two lines are equal,
    then the two lines are parallel -/
theorem interior_alternate_angles_equal_implies_parallel
  (l1 l2 : Line) (t : Transversal) (ias : InteriorAlternateAngles l1 l2 t)
  (h : angles_equal ias.angle1 ias.angle2) :
  are_parallel l1 l2 := by
  sorry

#check interior_alternate_angles_equal_implies_parallel

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_alternate_angles_equal_implies_parallel_l129_12983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_c_l129_12991

theorem triangle_sin_c (A B C : ℝ) : 
  A + B + C = Real.pi →
  Real.sin A = 4/5 →
  Real.cos B = 12/13 →
  Real.tan A = 4/3 →
  Real.sin C = 63/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_c_l129_12991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sticker_counting_possible_sticker_counting_possible_proof_l129_12993

/-- Represents a pack of stickers -/
structure StickerPack where
  total : ℕ
  remaining : ℕ
  inv : remaining ≤ total

/-- Represents the seller's inventory and counting process -/
structure SellerInventory where
  packs : List StickerPack
  time_elapsed : ℕ

/-- Represents a customer's sticker request -/
structure CustomerRequest where
  amount : ℕ

/-- Function to count stickers for a customer -/
def count_stickers (inventory : SellerInventory) (request : CustomerRequest) : Option SellerInventory :=
  sorry

/-- The main theorem to prove -/
theorem sticker_counting_possible (initial_inventory : SellerInventory) 
  (requests : List CustomerRequest) : Prop :=
  ∃ (final_inventory : SellerInventory),
    (initial_inventory.packs.length = 3) ∧
    (∀ pack ∈ initial_inventory.packs, pack.total = 100) ∧
    (requests.length = 3) ∧
    (requests.get? 0 = some ⟨70⟩) ∧
    (requests.get? 1 = some ⟨60⟩) ∧
    (requests.get? 2 = some ⟨60⟩) ∧
    (final_inventory.time_elapsed ≤ 70) ∧
    (∀ request ∈ requests, ∃ (intermediate_inventory : SellerInventory),
      count_stickers intermediate_inventory request = some final_inventory)

/-- Proof of the theorem -/
theorem sticker_counting_possible_proof (initial_inventory : SellerInventory) 
  (requests : List CustomerRequest) : sticker_counting_possible initial_inventory requests :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sticker_counting_possible_sticker_counting_possible_proof_l129_12993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l129_12922

/-- The area of a triangle whose intuitive diagram is an equilateral triangle with side length 1 -/
noncomputable def triangle_area : ℝ := Real.sqrt 6 / 2

/-- The side length of the equilateral triangle in the intuitive diagram -/
def intuitive_side_length : ℝ := 1

/-- The area of an equilateral triangle with side length s -/
noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

/-- The relationship between the area of the original triangle and its intuitive diagram -/
def area_relationship (intuitive_area original_area : ℝ) : Prop :=
  original_area = intuitive_area * 2 * Real.sqrt 2

theorem triangle_area_proof :
  area_relationship (equilateral_triangle_area intuitive_side_length) triangle_area :=
by
  sorry

#check triangle_area_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l129_12922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charge_configuration_energy_change_l129_12908

noncomputable def energy_between_charges (distance : ℝ) (base_energy : ℝ) (base_distance : ℝ) : ℝ :=
  base_energy * base_distance / distance

noncomputable def total_energy (e1 e2 e3 : ℝ) : ℝ := e1 + e2 + e3

theorem charge_configuration_energy_change 
  (original_energy : ℝ) 
  (side_length : ℝ) :
  original_energy = 18 →
  let pair_energy := original_energy / 3
  let new_energy1 := energy_between_charges (side_length / 2) pair_energy side_length
  let new_energy2 := energy_between_charges (side_length / 3) pair_energy side_length
  let new_energy3 := pair_energy
  total_energy new_energy1 new_energy2 new_energy3 = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_charge_configuration_energy_change_l129_12908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_convergence_l129_12984

theorem geometric_series_convergence :
  let a : ℝ := 3
  let r : ℝ := 1/3
  let S := (∑' n, a * r^n : ℝ)
  S = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_convergence_l129_12984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l129_12980

open Set
open Real

/-- The function f(x) = (x+1)/(x^2+1) -/
noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 + 1)

/-- The range of f -/
def range_f : Set ℝ := { y | ∃ x, f x = y }

/-- Theorem stating the range of f is [0, 0.6] -/
theorem range_of_f : range_f = Icc 0 (3/5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l129_12980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_gas_tank_capacity_l129_12998

/-- Proves that the car's gas tank capacity is 12 gallons given the problem conditions -/
theorem car_gas_tank_capacity 
  (truck_capacity : ℝ) 
  (truck_initial_fill : ℝ) 
  (car_initial_fill_fraction : ℝ) 
  (total_gas_added : ℝ) 
  (h1 : truck_capacity = 20)
  (h2 : truck_initial_fill = truck_capacity / 2)
  (h3 : car_initial_fill_fraction = 1 / 3)
  (h4 : total_gas_added = 18)
  : (total_gas_added - (truck_capacity - truck_initial_fill)) / (1 - car_initial_fill_fraction) = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_gas_tank_capacity_l129_12998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l129_12986

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 4*x else -x^2 + 4*x

-- State the theorem
theorem f_properties :
  -- f is an odd function
  (∀ x, f (-x) = -f x) ∧
  -- f(x) = -x² + 4x for x > 0
  (∀ x > 0, f x = -x^2 + 4*x) ∧
  -- Analytical expression of f
  (∀ x, f x = if x ≤ 0 then x^2 + 4*x else -x^2 + 4*x) ∧
  -- Minimum value on [-2, a] where a > -2
  (∀ a > -2,
    (a ≤ 2 + 2*Real.sqrt 2 → ∀ x ∈ Set.Icc (-2) a, f x ≥ -4) ∧
    (a > 2 + 2*Real.sqrt 2 → ∀ x ∈ Set.Icc (-2) a, f x ≥ f a)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l129_12986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l129_12988

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x + 2 * x

-- State the theorem
theorem range_of_a (a : ℝ) : (f (1 - a) + f (2 * a) < 0) → a < -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l129_12988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l129_12919

noncomputable def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

def minor_axis_length (b : ℝ) : ℝ := 2 * b

def line_slope_1 (x y m : ℝ) : Prop := y = x + m

def circle_through_origin (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

theorem ellipse_properties (a b : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : eccentricity a b = Real.sqrt 2 / 2) 
  (h4 : minor_axis_length b = 6) :
  (∃ x y, ellipse x y (3 * Real.sqrt 2) 3) ∧ 
  (∃ m x1 y1 x2 y2, 
    line_slope_1 x1 y1 m ∧ 
    line_slope_1 x2 y2 m ∧ 
    ellipse x1 y1 (3 * Real.sqrt 2) 3 ∧ 
    ellipse x2 y2 (3 * Real.sqrt 2) 3 ∧ 
    circle_through_origin x1 y1 x2 y2 ∧ 
    (m = 2 * Real.sqrt 3 ∨ m = -2 * Real.sqrt 3)) :=
by sorry

#check ellipse_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l129_12919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l129_12928

/-- Function representing the number of ways to express n as a sum of positive integers -/
def f (n : ℕ) : ℕ := sorry

/-- Theorem stating the inequality for f(n) -/
theorem f_inequality (n : ℕ) (hn : n ≥ 1) : 
  f (n + 1) ≤ (f n + f (n + 2)) / 2 := by
  sorry

#check f_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l129_12928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_abundant_not_multiple_of_five_l129_12968

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def proper_divisors (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (λ d => d > 0 ∧ n % d = 0)

def sum_proper_divisors (n : ℕ) : ℕ :=
  (proper_divisors n).sum id

def is_abundant (n : ℕ) : Prop :=
  n > 1 ∧ ¬(is_prime n) ∧ sum_proper_divisors n ≥ n

def not_multiple_of_five (n : ℕ) : Prop :=
  ¬(n % 5 = 0)

theorem smallest_abundant_not_multiple_of_five :
  ∀ n : ℕ, n < 12 → ¬(is_abundant n ∧ not_multiple_of_five n) ∧
  (is_abundant 12 ∧ not_multiple_of_five 12) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_abundant_not_multiple_of_five_l129_12968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_tripler_l129_12926

theorem matrix_tripler (a b c d : ℝ) :
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![3, 0; 0, 3]
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  N • A = !![3*a, 3*b; 3*c, 3*d] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_tripler_l129_12926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l129_12960

/-- The function f(x) defined in the problem -/
noncomputable def f (a b x : ℝ) : ℝ := (2^x + b) / (2^x + a)

/-- The function g(x) defined in the problem -/
noncomputable def g (a b x : ℝ) : ℝ := f a b x + x

/-- Main theorem encapsulating the problem -/
theorem problem_solution :
  ∃ (a b : ℝ),
    f a b 0 = 0 ∧
    f a b 1 = 1/3 ∧
    a = 1 ∧
    b = -1 ∧
    (∀ x ∈ Set.Icc 0 1, g a b x ∈ Set.Icc 0 (4/3)) ∧
    g a b 0 = 0 ∧
    g a b 1 = 4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l129_12960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_proof_l129_12958

-- Define a type for angles in degrees, minutes, and seconds
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)
  (seconds : ℕ)

-- Define a function to convert an Angle to a real number (in degrees)
noncomputable def angleToReal (a : Angle) : ℝ :=
  (a.degrees : ℝ) + (a.minutes : ℝ) / 60 + (a.seconds : ℝ) / 3600

-- Define the given angle
def givenAngle : Angle := ⟨18, 24, 36⟩

-- Define the angle we want to prove
def targetAngle : Angle := ⟨47, 43, 36⟩

-- Theorem statement
theorem angle_proof (x : Angle) : 
  angleToReal x - (180 - angleToReal x) / 2 = -angleToReal givenAngle → 
  x = targetAngle :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_proof_l129_12958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_nondivisible_l129_12987

theorem infinitely_many_nondivisible (a b : ℕ+) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, ¬((n ^ b.val + 1) ∣ (a.val ^ n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_nondivisible_l129_12987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_abc_product_l129_12937

-- Define the expression
noncomputable def original_expression : ℝ := (2 + Real.sqrt 5) / (2 - Real.sqrt 5)

-- Define the rationalized form
noncomputable def rationalized_form : ℝ := -9 - 4 * Real.sqrt 5

-- Theorem statement
theorem rationalize_denominator :
  ∃ (A B : ℤ) (C : ℕ), (A = -9 ∧ B = -4 ∧ C = 5) ∧
  (original_expression = A + B * Real.sqrt C) := by
  -- The proof goes here
  sorry

-- Verify that ABC = 180
theorem abc_product :
  let A : ℤ := -9
  let B : ℤ := -4
  let C : ℕ := 5
  A * B * C = 180 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_abc_product_l129_12937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_2020_value_l129_12947

def t : ℕ → ℚ
  | 0 => 20  -- We add this case to cover Nat.zero
  | 1 => 20
  | 2 => 21
  | n+3 => (5 * t (n+2) + 1) / (25 * t (n+1))

theorem t_2020_value : t 2020 = 101 / 525 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_2020_value_l129_12947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_height_angle_formula_permissible_alpha_values_l129_12914

/-- Regular triangular pyramid with lateral edge angle α -/
structure RegularTriangularPyramid where
  α : Real
  h_angle : 0 < α ∧ α < π / 2

/-- The angle between the lateral edge and the height of the pyramid -/
noncomputable def lateral_height_angle (p : RegularTriangularPyramid) : Real :=
  Real.arcsin ((2 * Real.cos p.α) / Real.sqrt 3)

theorem lateral_height_angle_formula (p : RegularTriangularPyramid) :
  lateral_height_angle p = Real.arcsin ((2 * Real.cos p.α) / Real.sqrt 3) :=
by sorry

theorem permissible_alpha_values (p : RegularTriangularPyramid) :
  π / 6 < p.α ∧ p.α < π / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_height_angle_formula_permissible_alpha_values_l129_12914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_ratio_in_combined_bags_yellow_ratio_percentage_l129_12910

/-- Represents a bag of jelly beans -/
structure JellyBeanBag where
  total : ℕ
  yellowRatio : ℚ

/-- Calculates the number of yellow jelly beans in a bag -/
def yellowCount (bag : JellyBeanBag) : ℚ :=
  bag.total * bag.yellowRatio

/-- Theorem: The ratio of yellow jelly beans to all beans when four bags are combined -/
theorem yellow_ratio_in_combined_bags 
  (bag1 bag2 bag3 bag4 : JellyBeanBag)
  (h1 : bag1 = ⟨20, 2/5⟩)
  (h2 : bag2 = ⟨25, 3/10⟩)
  (h3 : bag3 = ⟨35, 1/4⟩)
  (h4 : bag4 = ⟨40, 1/10⟩) :
  let totalYellow := yellowCount bag1 + yellowCount bag2 + yellowCount bag3 + yellowCount bag4
  let totalBeans := bag1.total + bag2.total + bag3.total + bag4.total
  (totalYellow / totalBeans) = 29/120 := by
  -- Proof steps would go here
  sorry

/-- Approximate equality for rationals -/
def approx_equal (x y : ℚ) (ε : ℚ) : Prop :=
  abs (x - y) < ε

notation:50 x " ≈ " y => approx_equal x y (1/1000)

/-- Theorem: The ratio of yellow jelly beans to all beans is approximately 24.17% -/
theorem yellow_ratio_percentage 
  (bag1 bag2 bag3 bag4 : JellyBeanBag)
  (h1 : bag1 = ⟨20, 2/5⟩)
  (h2 : bag2 = ⟨25, 3/10⟩)
  (h3 : bag3 = ⟨35, 1/4⟩)
  (h4 : bag4 = ⟨40, 1/10⟩) :
  let totalYellow := yellowCount bag1 + yellowCount bag2 + yellowCount bag3 + yellowCount bag4
  let totalBeans := bag1.total + bag2.total + bag3.total + bag4.total
  ((totalYellow / totalBeans) * 100) ≈ 24.17 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_ratio_in_combined_bags_yellow_ratio_percentage_l129_12910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_binary_numbers_with_blocks_l129_12918

def number_of_n_digit_binary_numbers_with_m_blocks (n m : ℕ) : ℕ :=
  Nat.choose (n + 1) (2 * m + 1)

theorem count_binary_numbers_with_blocks (n m : ℕ) : 
  (number_of_n_digit_binary_numbers_with_m_blocks n m) = Nat.choose (n + 1) (2 * m + 1) :=
by
  -- Unfold the definition
  unfold number_of_n_digit_binary_numbers_with_m_blocks
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_binary_numbers_with_blocks_l129_12918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_row_sum_is_528_l129_12997

/-- Represents a position in the grid -/
structure Position where
  row : ℕ
  col : ℕ

/-- Represents the grid and its properties -/
structure Grid where
  size : ℕ
  start : Position
  numbers : Finset ℕ

/-- Defines the spiral pattern -/
def spiral_pattern (g : Grid) : List Position := sorry

/-- Finds numbers in a specific row -/
def numbers_in_row (g : Grid) (row : ℕ) : List ℕ := sorry

/-- Theorem: Sum of max and min numbers in third row is 528 -/
theorem third_row_sum_is_528 (g : Grid) : 
  g.size = 17 ∧ 
  g.start = ⟨9, 9⟩ ∧ 
  g.numbers = Finset.range 289 →
  let third_row_nums := numbers_in_row g 3
  (List.maximum third_row_nums).getD 0 + (List.minimum third_row_nums).getD 0 = 528 := by
  sorry

#check third_row_sum_is_528

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_row_sum_is_528_l129_12997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_AP_l129_12944

noncomputable section

-- Define the ellipse C₁
def C₁ (x y : ℝ) : Prop := x^2 + (4/3) * y^2 = 1

-- Define the parabola C₂
def C₂ (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the focus F of C₁
def F : ℝ × ℝ := (-1/2, 0)

-- Define the right vertex A of C₁
def A : ℝ × ℝ := (1, 0)

-- Define the directrix l of C₂
noncomputable def l : ℝ → ℝ := λ _ => -1

-- Define a point P on l
noncomputable def P (m : ℝ) : ℝ × ℝ := (-1, -2/m)

-- Define a point Q on l, symmetric to P about x-axis
noncomputable def Q (m : ℝ) : ℝ × ℝ := (-1, 2/m)

-- Define point B as the intersection of AP and C₁
noncomputable def B (m : ℝ) : ℝ × ℝ := ((-3*m^2 + 4)/(3*m^2 + 4), (-6*m)/(3*m^2 + 4))

-- Define point D as the intersection of BQ and x-axis
noncomputable def D (m : ℝ) : ℝ × ℝ := ((2 - 3*m^2)/(3*m^2 + 2), 0)

-- State the theorem
theorem slope_of_AP (m : ℝ) :
  (∀ x y, C₁ x y → x^2/1^2 + y^2/(3/4) = 1) →
  (∀ x y, C₂ x y → y^2 = 4*x) →
  (F.1 = -1/2 ∧ F.2 = 0) →
  (A.1 = 1 ∧ A.2 = 0) →
  (l 0 = -1) →
  (P m = (-1, -2/m)) →
  (Q m = (-1, 2/m)) →
  (B m = ((-3*m^2 + 4)/(3*m^2 + 4), (-6*m)/(3*m^2 + 4))) →
  (D m = ((2 - 3*m^2)/(3*m^2 + 2), 0)) →
  (1/2 * |A.1 - (D m).1| * |2/m| = Real.sqrt 6/2) →
  m = Real.sqrt 6/2 ∨ m = -Real.sqrt 6/2 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_AP_l129_12944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_cardinality_bound_l129_12951

theorem set_cardinality_bound (n : ℕ) (m : ℕ) (F : Fin m → Finset (Fin n)) :
  (∀ i : Fin m, F i ⊆ Finset.univ) →
  (∀ i j : Fin m, i < j → min (Finset.card (F i \ F j)) (Finset.card (F j \ F i)) = 1) →
  m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_cardinality_bound_l129_12951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l129_12963

noncomputable section

theorem triangle_properties (A B C : Real) (a b c : Real) :
  Real.cos A = Real.sqrt 6 / 3 →
  Real.cos B = 2 * Real.sqrt 2 / 3 →
  c = 2 * Real.sqrt 2 →
  Real.tan (2 * A) = 2 * Real.sqrt 2 ∧
  (1/2) * a * c * Real.sin B = 2 * Real.sqrt 2 / 3 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l129_12963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l129_12969

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    prove that if b^2 + c^2 - a^2 = bc and sin^2(A) + sin^2(B) = sin^2(C),
    then A = π/3 and B = π/6 -/
theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  b^2 + c^2 - a^2 = b * c →
  Real.sin A ^ 2 + Real.sin B ^ 2 = Real.sin C ^ 2 →
  A = π/3 ∧ B = π/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l129_12969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l129_12953

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if certain conditions are met, then angle A and side a have specific values. -/
theorem triangle_problem (a b c A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →  -- Valid angle ranges
  Real.cos A * Real.sin C = 2 * Real.sin A * Real.sin B →  -- Given condition
  c = 2 * b →  -- Given condition
  (1/2) * b * c * Real.sin A = 2 * Real.sqrt 2 →  -- Area condition
  A = π/4 ∧ a^2 = 20 - 8 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l129_12953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_ratio_l129_12909

-- Define the square pyramid
noncomputable def square_pyramid (base_edge : ℝ) (altitude : ℝ) : ℝ := 
  (1/3) * base_edge^2 * altitude

-- Define the theorem
theorem frustum_volume_ratio : 
  let original_base_edge : ℝ := 24
  let original_altitude : ℝ := 18
  let smaller_altitude : ℝ := original_altitude / 3
  let original_volume := square_pyramid original_base_edge original_altitude
  let smaller_volume := square_pyramid (original_base_edge * (smaller_altitude / original_altitude)) smaller_altitude
  let frustum_volume := original_volume - smaller_volume
  (frustum_volume / original_volume) = 26/27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_ratio_l129_12909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_not_equal_area_division_l129_12949

/-- A triangle with vertices A, B, and C. -/
structure Triangle where
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  C : EuclideanSpace ℝ (Fin 2)

/-- An angle bisector of a triangle. -/
def AngleBisector (t : Triangle) := Set (EuclideanSpace ℝ (Fin 2))

/-- The area of a region. -/
noncomputable def Area (r : Set (EuclideanSpace ℝ (Fin 2))) : ℝ := sorry

/-- Theorem: Angle bisectors do not always divide a triangle into two parts with equal areas. -/
theorem angle_bisector_not_equal_area_division (t : Triangle) :
  ¬ ∀ (ab : AngleBisector t), ∃ (r1 r2 : Set (EuclideanSpace ℝ (Fin 2))),
    r1 ∪ r2 = {p : EuclideanSpace ℝ (Fin 2) | p = t.A ∨ p = t.B ∨ p = t.C} ∧
    r1 ∩ r2 ⊆ ab ∧
    Area r1 = Area r2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_not_equal_area_division_l129_12949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l129_12916

theorem equation_solution (x y z u : ℤ) :
  (4 : ℝ)^x + (4 : ℝ)^y + (4 : ℝ)^z = (u : ℝ)^2 ↔ 
    y = (x + z + 1) / 2 ∧ 
    u = (2 : ℝ)^x + (2 : ℝ)^z ∧ 
    x ≥ y ∧ 
    y ≥ z :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l129_12916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_less_cos_sin_l129_12976

-- Define the third quadrant
def third_quadrant (α : ℝ) : Prop :=
  Real.sin α < 0 ∧ Real.cos α < 0

-- State the theorem
theorem sin_cos_less_cos_sin (α : ℝ) (h : third_quadrant α) :
  Real.sin (Real.cos α) < Real.cos (Real.sin α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_less_cos_sin_l129_12976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l129_12966

/-- Given a hyperbola with equation x²/4 - y²/m² = 1 where m > 0,
    if the eccentricity is √3, then m = 2√2 -/
theorem hyperbola_eccentricity (m : ℝ) (h1 : m > 0) :
  (∀ x y : ℝ, x^2 / 4 - y^2 / m^2 = 1) →
  (Real.sqrt (4 + m^2) / 2 = Real.sqrt 3) →
  m = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l129_12966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_equals_sqrt_two_l129_12931

-- Define the complex number z
noncomputable def z : ℂ := (1 + 3 * Complex.I) / (2 - Complex.I)

-- State the theorem
theorem abs_z_equals_sqrt_two : Complex.abs z = Real.sqrt 2 := by
  -- The proof steps would go here, but we'll use 'sorry' for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_equals_sqrt_two_l129_12931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_positive_power_of_two_greater_than_one_l129_12989

theorem negation_of_forall_positive_power_of_two_greater_than_one :
  (¬ (∀ x : ℝ, x > 0 → (2 : ℝ)^x > 1)) ↔ (∃ x : ℝ, x > 0 ∧ (2 : ℝ)^x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_positive_power_of_two_greater_than_one_l129_12989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conical_hat_area_l129_12929

/-- The lateral surface area of a conical paper hat -/
noncomputable def lateral_surface_area (slant_height : ℝ) (base_diameter : ℝ) : ℝ :=
  Real.pi * (base_diameter / 2) * slant_height

/-- Theorem: The lateral surface area of a conical paper hat with slant height 6 and base diameter 4 is 12π -/
theorem conical_hat_area :
  lateral_surface_area 6 4 = 12 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conical_hat_area_l129_12929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_reciprocal_sum_l129_12978

/-- The ellipse in polar form -/
def ellipse (ρ θ : ℝ) : Prop :=
  (ρ * Real.cos θ)^2 / 16 + (ρ * Real.sin θ)^2 / 4 = 1

/-- Theorem: For any two points on the ellipse separated by π/2 in polar angle,
    the sum of the reciprocals of their squared radii is 5/16 -/
theorem ellipse_reciprocal_sum (ρ₁ ρ₂ θ : ℝ) :
  ellipse ρ₁ θ → ellipse ρ₂ (θ + π/2) → 1/ρ₁^2 + 1/ρ₂^2 = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_reciprocal_sum_l129_12978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_triangle_xyz_l129_12964

/-- Right prism with equilateral triangular bases -/
structure RightPrism :=
  (base_side : ℝ)
  (height : ℝ)

/-- Midpoint of an edge -/
structure Midpoint :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

/-- Triangle XYZ formed by midpoints -/
def TriangleXYZ (p : RightPrism) (x y z : Midpoint) : Prop :=
  x.z = y.z ∧ x.z = p.height / 2 ∧
  x.x = y.y ∧ x.x = p.base_side / 2 ∧
  z.x = p.base_side / 2 ∧ z.y = p.base_side / 2

/-- Perimeter of triangle XYZ -/
noncomputable def perimeterXYZ (p : RightPrism) (x y z : Midpoint) : ℝ :=
  let xy := p.base_side / 2
  let xz := Real.sqrt ((p.base_side / 2)^2 + (p.height / 2)^2)
  xy + 2 * xz

theorem perimeter_triangle_xyz 
  (p : RightPrism) 
  (x y z : Midpoint) 
  (h : TriangleXYZ p x y z) 
  (h_base : p.base_side = 14) 
  (h_height : p.height = 20) : 
  perimeterXYZ p x y z = 7 + 2 * Real.sqrt 149 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_triangle_xyz_l129_12964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_density_of_distance_to_origin_l129_12981

-- Define the square K
def K : Set (ℝ × ℝ) := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the random variable ξ as the distance from a point to the origin
noncomputable def ξ (p : ℝ × ℝ) : ℝ := Real.sqrt (p.1^2 + p.2^2)

-- Define the probability density function
noncomputable def p (x : ℝ) : ℝ :=
  if x > 0 ∧ x < 1 then
    Real.pi * x / 2
  else if x ≥ 1 ∧ x < Real.sqrt 2 then
    Real.pi * x / 2 - 2 * x * Real.arccos (1 / x)
  else
    0

-- State the theorem
theorem probability_density_of_distance_to_origin :
  ∀ x : ℝ, p x = 
    if x > 0 ∧ x < 1 then
      Real.pi * x / 2
    else if x ≥ 1 ∧ x < Real.sqrt 2 then
      Real.pi * x / 2 - 2 * x * Real.arccos (1 / x)
    else
      0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_density_of_distance_to_origin_l129_12981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l129_12959

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  2 * t.a * Real.cos t.C + 2 * t.c * Real.cos t.A = t.a + t.c ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  0 < t.A ∧ t.A < Real.pi ∧
  0 < t.B ∧ t.B < Real.pi ∧
  0 < t.C ∧ t.C < Real.pi

-- Part 1
theorem part_one (t : Triangle) (h : triangle_conditions t) 
  (h_sin : 4 * Real.sin t.A = 3 * Real.sin t.B) : 
  t.c / t.b = 5 / 4 := by sorry

-- Part 2
theorem part_two (t : Triangle) (h : triangle_conditions t) 
  (h_C : t.C = 2 * Real.pi / 3) (h_diff : t.c - t.a = 8) : 
  (1 / 2) * t.a * t.b * Real.sin t.C = 15 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l129_12959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_of_children_l129_12975

theorem average_weight_of_children (num_boys num_girls : ℕ) 
  (avg_weight_boys avg_weight_girls : ℚ) :
  num_boys = 8 →
  num_girls = 3 →
  avg_weight_boys = 155 →
  avg_weight_girls = 115 →
  (num_boys * avg_weight_boys + num_girls * avg_weight_girls) / (num_boys + num_girls) = 144 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_of_children_l129_12975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_samias_journey_l129_12939

/-- The problem of Samia's journey --/
theorem samias_journey 
  (total_time : ℚ) 
  (bike_speed : ℚ) 
  (walk_speed : ℚ) 
  (h1 : total_time = 44 / 60) 
  (h2 : bike_speed = 17) 
  (h3 : walk_speed = 5) : 
  ∃ (distance : ℚ), 
    (distance / bike_speed + distance / walk_speed = total_time) ∧ 
    (Int.floor (distance * 10 + 1/2) / 10 : ℚ) = 28/10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_samias_journey_l129_12939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_percentage_is_42_86_l129_12906

/-- Define the ratio of boys to girls -/
def boys_to_girls_ratio : ℚ := 3 / 4

/-- Define the total number of students -/
def total_students : ℕ := 42

/-- Define the function to calculate the percentage of boys -/
def percentage_of_boys : ℚ :=
  (boys_to_girls_ratio * total_students) / (boys_to_girls_ratio * total_students + total_students) * 100

/-- Theorem statement -/
theorem boys_percentage_is_42_86 :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |percentage_of_boys - 42.86| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_percentage_is_42_86_l129_12906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_tangent_implies_a_value_l129_12965

/-- The circle equation -/
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 + 2*a*x + a*y + 2*a^2 + a - 1 = 0

/-- The point P -/
def P : ℝ × ℝ := (2, 1)

/-- The condition for exactly one tangent line -/
def exactly_one_tangent (a : ℝ) : Prop :=
  ∃! l : Set (ℝ × ℝ), (P ∈ l) ∧ 
    (∀ p, p ∈ l → circle_equation p.1 p.2 a → 
      (∃! q, q ∈ l ∧ q ≠ p ∧ circle_equation q.1 q.2 a))

/-- The theorem statement -/
theorem unique_tangent_implies_a_value :
  ∀ a : ℝ, exactly_one_tangent a → a = -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_tangent_implies_a_value_l129_12965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_block_distance_increase_block_distance_increase_proof_l129_12985

/-- The increase in distance between two blocks after cutting a string -/
theorem block_distance_increase
  (m : Real) -- Mass of each block
  (μ : Real) -- Coefficient of friction
  (g : Real) -- Acceleration due to gravity
  (P : Real) -- Initial potential energy of the compressed spring
  (h : m > 0) -- Mass is positive
  (h1 : μ > 0) -- Coefficient of friction is positive
  (h2 : g > 0) -- Acceleration due to gravity is positive
  (h3 : P > 0) -- Initial potential energy is positive
  : Real :=
  P / (μ * m * g)

/-- The theorem states that the increase in distance is equal to P / (μ * m * g) -/
theorem block_distance_increase_proof
  (m : Real) -- Mass of each block
  (μ : Real) -- Coefficient of friction
  (g : Real) -- Acceleration due to gravity
  (P : Real) -- Initial potential energy of the compressed spring
  (h : m > 0) -- Mass is positive
  (h1 : μ > 0) -- Coefficient of friction is positive
  (h2 : g > 0) -- Acceleration due to gravity is positive
  (h3 : P > 0) -- Initial potential energy is positive
  : block_distance_increase m μ g P h h1 h2 h3 = P / (μ * m * g) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_block_distance_increase_block_distance_increase_proof_l129_12985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_10_50_l129_12946

theorem sin_cos_sum_10_50 :
  Real.sin (10 * π / 180) * Real.cos (50 * π / 180) +
  Real.cos (10 * π / 180) * Real.sin (50 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_10_50_l129_12946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mountain_ocean_depth_l129_12912

/-- Represents a cone-shaped mountain -/
structure Mountain where
  height : ℝ
  volumeAboveWater : ℝ

/-- The depth of the ocean at the base of the mountain -/
noncomputable def oceanDepth (m : Mountain) : ℝ :=
  m.height - (m.height * (m.volumeAboveWater ^ (1/3 : ℝ)))

/-- Theorem stating the depth of the ocean for a specific mountain -/
theorem mountain_ocean_depth :
  ∀ m : Mountain,
    m.height = 8000 →
    m.volumeAboveWater = 1/8 →
    oceanDepth m = 4000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mountain_ocean_depth_l129_12912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l129_12994

theorem simplify_expression (n : ℤ) :
  (2 : ℝ)^(-(3*n+1)) + (2 : ℝ)^(-(3*n-2)) - 3 * (2 : ℝ)^(-3*n) = (3/2) * (2 : ℝ)^(-3*n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l129_12994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_l129_12948

-- Define the complex number type
variable (z : ℂ)

-- Define the imaginary unit
def i : ℂ := Complex.I

-- State the theorem
theorem complex_magnitude : i * z = 1 + 2 * i → Complex.abs (z - 1) = Real.sqrt 2 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_l129_12948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_fraction_at_sports_meet_l129_12961

/-- Represents a school with a given number of students and ratio of boys to girls -/
structure School where
  total_students : ℕ
  boys_ratio : ℕ
  girls_ratio : ℕ

/-- Calculates the number of girls in a school -/
def num_girls (s : School) : ℕ :=
  s.total_students * s.girls_ratio / (s.boys_ratio + s.girls_ratio)

/-- Theorem: The fraction of girls at the combined sports meet is 1/2 -/
theorem girls_fraction_at_sports_meet (maplewood riverview : School)
  (h_maplewood : maplewood = ⟨300, 3, 2⟩)
  (h_riverview : riverview = ⟨240, 3, 5⟩) :
  (num_girls maplewood + num_girls riverview) * 2 =
  maplewood.total_students + riverview.total_students := by
  sorry

#eval num_girls ⟨300, 3, 2⟩ -- Expected: 120
#eval num_girls ⟨240, 3, 5⟩ -- Expected: 150

end NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_fraction_at_sports_meet_l129_12961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equal_angles_after_cut_l129_12932

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  angle_sum : (angles 0) + (angles 1) + (angles 2) = 180
  positive_angles : ∀ i, angles i > 0

-- Define a function that represents cutting a triangle into two triangles
def cut_triangle (t : Triangle) : Triangle × Triangle :=
  sorry

-- Define a function to count equal angles
def count_equal_angles (t1 t2 : Triangle) : ℕ :=
  sorry

-- Theorem statement
theorem max_equal_angles_after_cut :
  (∀ t : Triangle, let (t1, t2) := cut_triangle t
    count_equal_angles t1 t2 ≤ 4) ∧
  (∃ t : Triangle, let (t1, t2) := cut_triangle t
    count_equal_angles t1 t2 = 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equal_angles_after_cut_l129_12932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_expansion_theorem_l129_12973

noncomputable def coefficientOfX3 : (ℝ → ℝ) → ℝ := 
  λ f ↦ sorry -- This function returns the coefficient of x^3 in the expansion of f

theorem coefficient_expansion_theorem (a : ℝ) :
  (∃ c : ℝ, c = 20 ∧ 
   c = coefficientOfX3 ((λ x ↦ x + a) * (λ x ↦ 1/x + 2*x)^5)) →
  a = 1/4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_expansion_theorem_l129_12973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l129_12905

/-- For positive real numbers x and a, the expression (x^2 + 2a - √(x^4 + 4a^2)) / x 
    is less than or equal to 2√(2a) - 2a, with equality when x = √(2a). -/
theorem max_value_expression (x a : ℝ) (hx : x > 0) (ha : a > 0) :
  (x^2 + 2*a - Real.sqrt (x^4 + 4*a^2)) / x ≤ 2 * Real.sqrt (2*a) - 2*a ∧
  (x^2 + 2*a - Real.sqrt (x^4 + 4*a^2)) / x = 2 * Real.sqrt (2*a) - 2*a ↔ x = Real.sqrt (2*a) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l129_12905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integral_area_of_circle_l129_12934

open Real

theorem smallest_integral_area_of_circle : 
  ∃ (A : ℕ), A = 29 ∧ 
  (∀ (r : ℝ), r > 0 → π * r^2 > 2 * π * r → ↑A > π * r^2) ∧
  (∀ (B : ℕ), B < A → ∃ (r : ℝ), r > 0 ∧ π * r^2 > 2 * π * r ∧ ↑B ≤ π * r^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integral_area_of_circle_l129_12934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l129_12921

noncomputable section

open Real

structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ

def Triangle.law_of_sines (t : Triangle) : ℝ :=
  t.a / sin t.A - t.b / sin t.B

def Triangle.law_of_cosines (t : Triangle) : ℝ :=
  t.a^2 - (t.b^2 + t.c^2 - 2 * t.b * t.c * cos t.A)

theorem triangle_problem (t : Triangle) 
  (h1 : t.b * cos t.A + t.a * sin t.B = 0)
  (h2 : t.b + t.c = 2 + Real.sqrt 2)
  (h3 : t.area = 1) :
  t.A = 3 * π / 4 ∧ t.a = Real.sqrt 10 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l129_12921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_property_l129_12999

/-- The fractional part of a real number -/
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

/-- Theorem statement -/
theorem golden_ratio_property (a : ℝ) (h1 : a > 0) (h2 : frac (a⁻¹) = frac (a^2)) (h3 : 2 < a^2) (h4 : a^2 < 3) :
  a^12 - 144 * a⁻¹ = 233 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_property_l129_12999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_schedule_meets_target_l129_12957

/-- Represents the weekly work schedule and earnings for an internship --/
structure WorkSchedule where
  weeksWorked : ℕ
  hoursPerWeek : ℚ
  baseRate : ℚ
  overtimeRate : ℚ
  baseHours : ℕ
  targetAmount : ℚ

/-- Calculates the total earnings based on the work schedule --/
noncomputable def totalEarnings (schedule : WorkSchedule) : ℚ :=
  let baseEarnings := schedule.baseRate * (min schedule.hoursPerWeek (schedule.baseHours : ℚ))
  let overtimeHours := max (schedule.hoursPerWeek - (schedule.baseHours : ℚ)) 0
  let overtimeEarnings := schedule.overtimeRate * overtimeHours
  (baseEarnings + overtimeEarnings) * (schedule.weeksWorked : ℚ)

/-- Theorem stating that the calculated work schedule meets the target amount --/
theorem work_schedule_meets_target (schedule : WorkSchedule) 
    (h1 : schedule.weeksWorked = 7)
    (h2 : schedule.hoursPerWeek = 27140 / 1000)
    (h3 : schedule.baseRate = 10)
    (h4 : schedule.overtimeRate = 15)
    (h5 : schedule.baseHours = 10)
    (h6 : schedule.targetAmount = 2500) :
    totalEarnings schedule = schedule.targetAmount := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_schedule_meets_target_l129_12957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_A₁A₂_when_circle_passes_F₁_l129_12979

-- Define the points and shapes
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

def circle_F₁ (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 16

def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 1)

-- Define the intersection points
noncomputable def A₁ : ℝ × ℝ := sorry
noncomputable def A₂ : ℝ × ℝ := sorry
noncomputable def B₁ : ℝ × ℝ := sorry
noncomputable def B₂ : ℝ × ℝ := sorry

-- State the theorem
theorem length_A₁A₂_when_circle_passes_F₁ :
  ∀ k : ℝ,
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    B₁ = (x₁, y₁) ∧ B₂ = (x₂, y₂) ∧
    ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
    line_l k x₁ y₁ ∧ line_l k x₂ y₂) →
  (∃ x₃ y₃ x₄ y₄ : ℝ,
    A₁ = (x₃, y₃) ∧ A₂ = (x₄, y₄) ∧
    parabola x₃ y₃ ∧ parabola x₄ y₄ ∧
    line_l k x₃ y₃ ∧ line_l k x₄ y₄) →
  ((F₁.1 - B₁.1) * (F₁.1 - B₂.1) + (F₁.2 - B₁.2) * (F₁.2 - B₂.2) = 0) →
  ‖A₁ - A₂‖ = 64/9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_A₁A₂_when_circle_passes_F₁_l129_12979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_range_l129_12904

theorem sin_cos_range (x y : ℝ) (h : Real.sin x + Real.sin y = 1/3) :
  ∃ (z : ℝ), Real.sin y - Real.cos x ^ 2 = z ∧ -11/12 ≤ z ∧ z ≤ 4/9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_range_l129_12904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_l129_12900

def a : Fin 3 → ℝ := ![(-2 : ℝ), 3, -1]
def b : Fin 3 → ℝ := ![(1 : ℝ), -2, 3]

theorem vector_subtraction : (a - 5 • b) = ![(-7 : ℝ), 13, -16] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_l129_12900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_8_256_l129_12903

theorem log_8_256 : Real.log 256 / Real.log 8 = 8 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_8_256_l129_12903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_summer_special_savings_l129_12941

/-- Represents the "summer special" deal at the 2023 Lakeview County Fair --/
structure SummerSpecial where
  regular_price : ℚ
  second_hat_discount : ℚ
  other_hats_discount : ℚ

/-- Calculates the total cost for a given number of hats under the summer special deal --/
def total_cost (deal : SummerSpecial) (num_hats : ℕ) : ℚ :=
  if num_hats = 0 then 0
  else if num_hats = 1 then deal.regular_price
  else if num_hats = 2 then deal.regular_price + deal.regular_price * (1 - deal.second_hat_discount)
  else deal.regular_price + 
       deal.regular_price * (1 - deal.second_hat_discount) + 
       deal.regular_price * (1 - deal.other_hats_discount) * (num_hats - 2 : ℚ)

/-- Calculates the percentage saved for a given number of hats under the summer special deal --/
def percentage_saved (deal : SummerSpecial) (num_hats : ℕ) : ℚ :=
  (1 - total_cost deal num_hats / (deal.regular_price * num_hats)) * 100

/-- Theorem stating that the percentage saved when buying four hats under the given summer special deal is 27.5% --/
theorem summer_special_savings : 
  let deal : SummerSpecial := { 
    regular_price := 30,
    second_hat_discount := 1/2,
    other_hats_discount := 3/10
  }
  percentage_saved deal 4 = 55/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_summer_special_savings_l129_12941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maxDistanceToLine_l129_12992

-- Define the circle
def circleEq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 7 = 0

-- Define the line
def lineEq (x y : ℝ) : Prop := x - y - 5 = 0

-- Define the distance function from a point to the line
noncomputable def distToLine (x y : ℝ) : ℝ := |x - y - 5| / Real.sqrt 2

-- Theorem statement
theorem maxDistanceToLine :
  ∃ (maxDist : ℝ), maxDist = (5 * Real.sqrt 2) / 2 + 1 ∧
  ∀ (x y : ℝ), circleEq x y → distToLine x y ≤ maxDist :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_maxDistanceToLine_l129_12992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intersection_l129_12930

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x

theorem tangent_line_intersection :
  let x₀ : ℝ := Real.exp 1
  let y₀ : ℝ := f x₀
  let m : ℝ := (deriv f) x₀
  let tangent_line (x : ℝ) : ℝ := m * (x - x₀) + y₀
  tangent_line 0 = 0 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intersection_l129_12930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l129_12996

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a x - x

-- Define the function h
def h (a : ℝ) : ℝ := (a + 1) * (1 - Real.log (a + 1))

-- Define IsTagentLine
def IsTagentLine (f : ℝ → ℝ) (x₀ y₀ : ℝ) (l : ℝ → ℝ) : Prop :=
  (f x₀ = y₀) ∧ (HasDerivAt f (l x₀ - y₀) x₀)

theorem function_properties (a : ℝ) (h_a : a > -1) :
  -- Part 1: Tangent line equation when a = 1
  (a = 1 → ∀ y, y = 1 ↔ IsTagentLine (f 1) 0 1 (λ _ ↦ 1)) ∧
  -- Part 2: Monotonicity of f
  ((-1 < a ∧ a ≤ 0 → ∀ x, HasDerivAt (f a) (Real.exp x - a) x ∧ Real.exp x - a > 0) ∧
   (0 < a → ∀ x, HasDerivAt (f a) (Real.exp x - a) x ∧
     (x < Real.log a → Real.exp x - a < 0) ∧
     (x > Real.log a → Real.exp x - a > 0))) ∧
  -- Part 3: Maximum value of h
  (∀ a, h a ≤ 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l129_12996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_is_45_l129_12945

/-- Represents the swimming scenario with given conditions -/
structure SwimmingScenario where
  downstream_time : ℝ
  upstream_time : ℝ
  upstream_distance : ℝ
  current_speed : ℝ

/-- Calculates the downstream distance given a swimming scenario -/
noncomputable def downstream_distance (s : SwimmingScenario) : ℝ :=
  let still_water_speed := s.upstream_distance / s.upstream_time + s.current_speed
  still_water_speed * s.downstream_time - s.current_speed * s.downstream_time

/-- Theorem stating that under the given conditions, the downstream distance is 45 km -/
theorem downstream_distance_is_45 (s : SwimmingScenario) 
  (h1 : s.downstream_time = 5)
  (h2 : s.upstream_time = 5)
  (h3 : s.upstream_distance = 25)
  (h4 : s.current_speed = 2) :
  downstream_distance s = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_is_45_l129_12945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_l129_12915

theorem triangle_angle (a b c : ℝ) (h : b^2 + c^2 = b*c + a^2) :
  Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_l129_12915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_a_and_ω_l129_12970

/-- A function f that satisfies the given conditions -/
noncomputable def f (ω a : ℝ) : ℝ → ℝ := λ x ↦ Real.sin (ω * x) + a * Real.cos (ω * x)

/-- Theorem stating the existence of a and ω satisfying the conditions -/
theorem exists_a_and_ω : ∃ (a ω : ℝ),
  ω > 0 ∧
  (∀ x, f ω a (2*π/3 - x) = f ω a (π/3 + x)) ∧
  (∀ x, f ω a (π/6) ≤ f ω a x) ∧
  a + ω = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_a_and_ω_l129_12970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_z_first_quadrant_condition_l129_12942

-- Part 1: Complex fraction simplification
theorem complex_fraction_simplification :
  (Complex.I - 3) / (2 - 4 * Complex.I) = -1/2 - Complex.I/2 := by sorry

-- Part 2: Range of m for z in first quadrant
def z (m : ℝ) : ℂ := Complex.mk (m + 2) (m^2 - m - 2)

theorem z_first_quadrant_condition (m : ℝ) :
  (z m).re > 0 ∧ (z m).im > 0 ↔ m > 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_z_first_quadrant_condition_l129_12942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_l129_12924

-- Define the function f(x) = 4x^2 + 1/x
noncomputable def f (x : ℝ) : ℝ := 4 * x^2 + 1/x

-- State the theorem
theorem f_monotonic_increasing :
  ∀ x₁ x₂ : ℝ, x₁ > (1/2 : ℝ) → x₂ > (1/2 : ℝ) → x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_l129_12924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipses_properties_l129_12920

/-- Defines an ellipse M with equation x²/9 + y²/b² = 1 where b > 0 -/
def ellipse_M (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 9 + p.2^2 / b^2 = 1 ∧ b > 0}

/-- Defines an ellipse N that passes through (√2/2, √3) -/
def ellipse_N : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 / 6 = 1}

/-- Defines the line y = x - 2 -/
def line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 - 2}

/-- States that one focus of M is at (2, 0) -/
axiom focus_M : ∃ (b : ℝ), (2, 0) ∈ ellipse_M b

/-- States that N has its focus at the vertex on the minor axis of M -/
axiom focus_N : ∃ (b : ℝ), (0, b) ∈ ellipse_M b ∧ (0, b) ∈ ellipse_N

/-- Theorem stating the properties of ellipses M and N -/
theorem ellipses_properties :
  (Real.sqrt 2 / 2, Real.sqrt 3) ∈ ellipse_N ∧
  (∃ (A B : ℝ × ℝ), A ∈ ellipse_N ∧ B ∈ ellipse_N ∧ A ∈ line ∧ B ∈ line ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 12/7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipses_properties_l129_12920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_minimum_f_l129_12935

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := 
  (1 + 1 / Real.log (Real.sqrt (x^2 + 10) + x)) * 
  (1 + 2 / Real.log (Real.sqrt (x^2 + 10) - x))

/-- The theorem stating that f(x) has no minimum value in the given interval -/
theorem no_minimum_f : 
  ¬ ∃ (m : ℝ), ∀ (x : ℝ), 0 < x → x < 4.5 → f x ≥ m ∧ ∃ (y : ℝ), 0 < y ∧ y < 4.5 ∧ f y = m :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_minimum_f_l129_12935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l129_12907

/-- The eccentricity of an ellipse with the given properties -/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  ∃ (c : ℝ), c > 0 ∧ c^2 = a^2 - b^2 ∧
  (∃ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 ∧ 
   (x - 2*c)^2 + y^2 = (x + c)^2 + y^2 ∧
   Real.sqrt 3 * (x + c)/2 + y/2 = 0) →
  c/a = Real.sqrt 3 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l129_12907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_track_circumference_l129_12911

/-- Represents the circumference of the circular track -/
noncomputable def C : ℝ := sorry

/-- A and B start from diametrically opposite points -/
noncomputable def start_distance : ℝ := C / 2

/-- Distance traveled by A at first meeting -/
def first_meeting_A : ℝ := 120

/-- Distance traveled by A at second meeting -/
noncomputable def second_meeting_A : ℝ := C - 80

/-- Distance traveled by B at third meeting -/
noncomputable def third_meeting_B : ℝ := C + 240

/-- Uniform speed ratio between A and B -/
noncomputable def speed_ratio : ℝ := first_meeting_A / (start_distance - first_meeting_A)

theorem track_circumference : C = 520 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_track_circumference_l129_12911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_music_students_l129_12927

theorem music_students (total art both neither music : ℕ) 
  (h1 : total = 500)
  (h2 : art = 20)
  (h3 : both = 10)
  (h4 : neither = 460) :
  total = (art + music - both) + neither → music = 30 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_music_students_l129_12927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pq_length_sum_l129_12956

def S : ℝ × ℝ := (10, 8)

def line1 (x y : ℝ) : Prop := 5 * y = 12 * x

def line2 (x y : ℝ) : Prop := 15 * y = 4 * x

def is_midpoint (P Q M : ℝ × ℝ) : Prop :=
  M.1 = (P.1 + Q.1) / 2 ∧ M.2 = (P.2 + Q.2) / 2

def on_line (P : ℝ × ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  line P.1 P.2

theorem pq_length_sum (P Q : ℝ × ℝ) (m n : ℕ) :
  on_line P line1 →
  on_line Q line2 →
  is_midpoint P Q S →
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (m / n : ℝ)^2 →
  Nat.Coprime m n →
  m + n = 337 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pq_length_sum_l129_12956
