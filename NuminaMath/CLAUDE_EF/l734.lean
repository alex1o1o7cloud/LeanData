import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_of_triangle_l734_73425

/-- The longest side of a triangle with vertices at (1,3), (4,8), and (8,3) has a length of √41 units. -/
theorem longest_side_of_triangle : ∃ (a b c : ℝ × ℝ), 
  a = (1, 3) ∧ b = (4, 8) ∧ c = (8, 3) ∧
  let sides := [dist a b, dist b c, dist c a]
  (sides.maximum? : Option ℝ) = some (Real.sqrt 41) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_of_triangle_l734_73425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_distance_l734_73412

/-- Calculates the distance traveled given time in minutes and speed in km/hr -/
noncomputable def distance_traveled (time_minutes : ℕ) (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (time_minutes : ℝ) / 60

/-- Proves that walking for 72 minutes at 10 km/hr results in a distance of 12 km -/
theorem walking_distance : distance_traveled 72 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_distance_l734_73412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l734_73495

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) * Real.tan x + Real.cos (2 * x - Real.pi / 3) - 1

theorem f_monotone_decreasing (k : ℤ) :
  StrictMonoOn f (Set.Icc (k * Real.pi + Real.pi / 3) (k * Real.pi + Real.pi / 2)) ∧
  StrictMonoOn f (Set.Ioc (k * Real.pi + Real.pi / 2) (k * Real.pi + 5 * Real.pi / 6)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l734_73495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_minimum_triangle_area_line_equation_at_minimum_area_l734_73434

noncomputable def line_l (k : ℝ) (x y : ℝ) : Prop := k * x + y + 1 + 2 * k = 0

def fixed_point : ℝ × ℝ := (-2, -1)

noncomputable def triangle_area (k : ℝ) : ℝ := 
  (1 + 2 * k)^2 / (2 * k)

theorem line_passes_through_fixed_point :
  ∀ k : ℝ, line_l k (fixed_point.1) (fixed_point.2) := by
  intro k
  simp [line_l, fixed_point]
  ring

theorem minimum_triangle_area :
  ∃ k : ℝ, k > 0 ∧ 
    (∀ k' : ℝ, k' > 0 → triangle_area k ≤ triangle_area k') ∧
    triangle_area k = 4 := by
  sorry

theorem line_equation_at_minimum_area :
  ∃ k : ℝ, k > 0 ∧ 
    (∀ k' : ℝ, k' > 0 → triangle_area k ≤ triangle_area k') ∧
    (∀ x y : ℝ, line_l k x y ↔ x + 2 * y + 4 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_minimum_triangle_area_line_equation_at_minimum_area_l734_73434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_points_covered_l734_73462

/-- A point in a plane represented by its coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- A set of points can be covered by a circle of radius r if there exists a point
    such that the distance from this point to any point in the set is at most r -/
def canBeCovered (points : Set Point) (r : ℝ) : Prop :=
  ∃ center : Point, ∀ p ∈ points, distance center p ≤ r

/-- Main theorem: If any three points can be covered by a circle of radius 1,
    then all points can be covered by a circle of radius 1 -/
theorem all_points_covered (p : ℕ) (points : Set Point) 
    (h : ∀ (a b c : Point), a ∈ points → b ∈ points → c ∈ points → 
      canBeCovered {a, b, c} 1) :
  canBeCovered points 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_points_covered_l734_73462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C₁_to_C₂_area_triangle_C₁_C₃_l734_73414

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := (x + 2)^2 + (y + 1)^2 = 1

-- Define the line C₂
def C₂ (x : ℝ) : Prop := x = 3

-- Define the line C₃
def C₃ (x y : ℝ) : Prop := y = x

-- Theorem for the maximum distance
theorem max_distance_C₁_to_C₂ :
  ∃ (d : ℝ), d = 6 ∧ 
  ∀ (P : ℝ × ℝ), C₁ P.1 P.2 → 
  ∀ (Q : ℝ × ℝ), C₂ Q.1 → 
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ d :=
sorry

-- Theorem for the area of the triangle
theorem area_triangle_C₁_C₃ :
  ∃ (A B : ℝ × ℝ), C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧ C₃ A.1 A.2 ∧ C₃ B.1 B.2 ∧
  let C := (-2, -1)  -- Center of C₁
  let area := (1/2) * Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) * 
              (abs (C.1 - A.1) / Real.sqrt 2)
  area = 1/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C₁_to_C₂_area_triangle_C₁_C₃_l734_73414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_substitute_teacher_turnover_l734_73411

/-- Given the initial number of substitute teachers, the percentage who walk out after 1 hour,
    and the number remaining after lunch, calculate the percentage of remaining teachers
    who quit before lunch. -/
theorem substitute_teacher_turnover
  (initial_teachers : ℕ)
  (walkout_percentage : ℚ)
  (remaining_after_lunch : ℕ)
  (h1 : initial_teachers = 60)
  (h2 : walkout_percentage = 1/2)
  (h3 : remaining_after_lunch = 21) :
  let remaining_after_hour := initial_teachers - (walkout_percentage * initial_teachers).floor
  let quit_before_lunch := remaining_after_hour - remaining_after_lunch
  (quit_before_lunch : ℚ) / remaining_after_hour = 3/10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_substitute_teacher_turnover_l734_73411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evelyn_paper_usage_l734_73445

/-- Calculates the average number of sheets used per working day, rounded to the nearest integer -/
def average_sheets_per_day (sheets_per_pad : ℕ) (working_days_per_week : ℕ) 
  (sheets_day1 sheets_day2 sheets_day3 : ℕ) (weeks_per_year : ℕ) (additional_days_off : ℕ) : ℕ :=
  let total_sheets_per_week := sheets_day1 + sheets_day2 + sheets_day3
  let total_working_days := working_days_per_week * weeks_per_year - additional_days_off
  let total_sheets_per_year := total_sheets_per_week * weeks_per_year
  (total_sheets_per_year + total_working_days / 2) / total_working_days

theorem evelyn_paper_usage :
  average_sheets_per_day 60 3 2 4 6 48 8 = 4 := by
  rfl

#eval average_sheets_per_day 60 3 2 4 6 48 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evelyn_paper_usage_l734_73445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_solution_l734_73455

/-- The smallest positive angle (in radians) satisfying the given trigonometric equation --/
noncomputable def smallest_angle : ℝ :=
  (10.45 * Real.pi) / 180

/-- The given trigonometric equation --/
def trig_equation (x : ℝ) : Prop :=
  9 * Real.sin x * (Real.cos x)^3 - 9 * (Real.sin x)^3 * Real.cos x = 3/2

theorem smallest_angle_solution :
  trig_equation smallest_angle ∧ 
  (∀ y, 0 < y ∧ y < smallest_angle → ¬ trig_equation y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_solution_l734_73455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrationality_check_l734_73457

theorem irrationality_check :
  ¬ (∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 5 = (p : ℚ) / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 4 = (p : ℚ) / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ 22 / 7 = (p : ℚ) / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (1.414 : ℚ) = (p : ℚ) / q) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrationality_check_l734_73457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_f_range_of_a_l734_73437

/-- The function f(x) defined in the problem -/
noncomputable def f (a b x : ℝ) : ℝ := x - a * Real.log x - b / x - 2

/-- Theorem for part I of the problem -/
theorem monotonicity_of_f (a b : ℝ) (h1 : a - b = 1) (h2 : a > 1) :
  ∃ (x1 x2 : ℝ), x1 < x2 ∧ 
  (∀ x ∈ Set.Ioo 0 x1, Monotone (fun x => f a b x)) ∧
  (∀ x ∈ Set.Ioo x1 x2, Antitone (fun x => f a b x)) ∧
  (∀ x ∈ Set.Ioi x2, Monotone (fun x => f a b x)) := by
  sorry

/-- Theorem for part II of the problem -/
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 2 4, f a (-1) x < -3 / x) ↔ a ∈ Set.Ioc (2 / Real.log 2) 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_f_range_of_a_l734_73437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_formula_line_slope_through_given_points_l734_73421

/-- The slope of a line passing through two points (x₁, y₁) and (x₂, y₂) is (y₂ - y₁) / (x₂ - x₁) -/
theorem slope_formula (x₁ y₁ x₂ y₂ : ℚ) (h : x₂ ≠ x₁) :
  (y₂ - y₁) / (x₂ - x₁) = (y₂ - y₁) / (x₂ - x₁) := by rfl

/-- Theorem: The slope of the line passing through points (-3, 3/2) and (4, -7/2) is -5/7 -/
theorem line_slope_through_given_points :
  ((-7/2) - (3/2)) / (4 - (-3)) = -5/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_formula_line_slope_through_given_points_l734_73421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_l734_73473

theorem min_value_trig_expression (x : Real) (h : 0 < x ∧ x < Real.pi / 2) :
  (2 * Real.sin x + 1 / Real.sin x)^2 + (2 * Real.cos x + 1 / Real.cos x)^2 ≥ 28 ∧
  ∃ y, 0 < y ∧ y < Real.pi / 2 ∧
    (2 * Real.sin y + 1 / Real.sin y)^2 + (2 * Real.cos y + 1 / Real.cos y)^2 = 28 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_l734_73473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_double_application_function_l734_73481

theorem no_double_application_function :
  ¬ ∃ (f : ℝ → ℝ), ∀ (x : ℝ), f (f x) = x^2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_double_application_function_l734_73481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_minimization_l734_73453

/-- Given ten distinct natural numbers, prove the minimum value of N / a₆ and the values of a₆ that minimize N / a₁ --/
theorem lcm_minimization (a : Fin 10 → ℕ) (h_distinct : Function.Injective a) 
  (h_ordered : ∀ i j, i < j → a i < a j) (h_a6_range : a 5 ∈ Set.Icc 1 2000) :
  let N := Nat.lcm (a 0) (Nat.lcm (a 1) (Nat.lcm (a 2) (Nat.lcm (a 3) (Nat.lcm (a 4) 
           (Nat.lcm (a 5) (Nat.lcm (a 6) (Nat.lcm (a 7) (Nat.lcm (a 8) (a 9)))))))))
  ∃ (min_N_div_a6 : N / a 5 ≥ 420), 
  {x : ℕ | x ∈ Set.Icc 1 2000 ∧ 
    ∃ (a' : Fin 10 → ℕ), Function.Injective a' ∧ (∀ i j, i < j → a' i < a' j) ∧ 
    a' 0 = 1 ∧ a' 5 = x ∧ 
    Nat.lcm (a' 0) (Nat.lcm (a' 1) (Nat.lcm (a' 2) (Nat.lcm (a' 3) (Nat.lcm (a' 4) 
    (Nat.lcm (a' 5) (Nat.lcm (a' 6) (Nat.lcm (a' 7) (Nat.lcm (a' 8) (a' 9))))))))) / a' 0 
    = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 8 
    (Nat.lcm 9 (Nat.lcm 10 (Nat.lcm 14 15)))))))
  } = {504, 1008, 1512} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_minimization_l734_73453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l734_73456

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := ((x - 1) / (x + 1)) ^ 2

-- Define the inverse function of f
noncomputable def f_inv (x : ℝ) : ℝ := (1 + Real.sqrt x) / (1 - Real.sqrt x)

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc (1/16) (1/4), (1 - Real.sqrt x) * f_inv x > m * (m - Real.sqrt x)) ↔
  -1 < m ∧ m < 5/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l734_73456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_a_range_l734_73427

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x < 1 then (3*a - 1)*x + 4*a else a^x

theorem decreasing_function_a_range :
  ∀ a : ℝ, 
  (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (1/6 : ℝ) (1/3 : ℝ) ∧ a ≠ 1/3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_a_range_l734_73427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_implies_a_range_l734_73406

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^3 - a*x) / Real.log a

-- State the theorem
theorem monotonic_f_implies_a_range (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : StrictMonoOn (f a) (Set.Ioo (-1/3 : ℝ) 0)) :
  a ∈ Set.Icc (1/3 : ℝ) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_implies_a_range_l734_73406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l734_73471

-- Define the equation of the region
def region_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 18*y = -45

-- Define the area of the region
noncomputable def region_area : ℝ := 52 * Real.pi

-- Theorem statement
theorem area_of_region :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, region_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    region_area = Real.pi * radius^2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l734_73471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l734_73470

/-- An arithmetic sequence with given conditions -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (using rationals instead of reals)
  h1 : a 10 = 30  -- a_10 = 30
  h2 : a 20 = 50  -- a_20 = 50

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = 2 * n + 10) ∧
  (∃ n : ℕ, sum_n seq n = 242 ∧ n = 11) :=
by
  sorry

#eval sum_n {a := λ n => 2 * n + 10, h1 := by norm_num, h2 := by norm_num} 11

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l734_73470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l734_73469

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.b = Real.sqrt 2 * Real.sin t.B ∧
  Real.tan t.A + Real.tan t.C = (2 * Real.sin t.B) / Real.cos t.A

-- Helper function for triangle area
noncomputable def area (t : Triangle) : Real :=
  (1 / 2) * t.a * t.b * Real.sin t.C

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : satisfies_conditions t) :
  t.C = π / 3 ∧
  t.c = Real.sqrt 6 / 2 ∧
  (∀ t' : Triangle, satisfies_conditions t' → 
    area t' ≤ 3 * Real.sqrt 3 / 8) ∧
  (∃ t' : Triangle, satisfies_conditions t' ∧ 
    area t' = 3 * Real.sqrt 3 / 8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l734_73469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largeCircleRadius_satisfies_equation_nestedSmallerCircleRadius_satisfies_equation_l734_73419

/-- The radius of the large circles in a configuration where:
    - There are 7 small circles with unit radius.
    - The centers of 6 small circles form a regular hexagon, with the 7th circle at the center.
    - The large circles are tangent to each other and to the small circles. -/
noncomputable def largeCircleRadius : ℝ := 2 * Real.sqrt 3

theorem largeCircleRadius_satisfies_equation :
  largeCircleRadius^2 - (4 * Real.sqrt 3 / 3) * largeCircleRadius + 1 = 0 := by
  sorry

/-- The other root of the quadratic equation represents the radius of a nested smaller circle
    when the large circle configuration undergoes a specific transformation. -/
noncomputable def nestedSmallerCircleRadius : ℝ := (2 * Real.sqrt 3) / 3

theorem nestedSmallerCircleRadius_satisfies_equation :
  nestedSmallerCircleRadius^2 - (4 * Real.sqrt 3 / 3) * nestedSmallerCircleRadius + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largeCircleRadius_satisfies_equation_nestedSmallerCircleRadius_satisfies_equation_l734_73419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_before_incline_l734_73435

/-- The original speed of a train before an incline, given certain conditions -/
theorem train_speed_before_incline (train_length : ℝ) (crossing_time : ℝ) (man_speed_kmph : ℝ) (incline_percent : ℝ) (speed_decrease_percent : ℝ) :
  train_length = 1000 →
  crossing_time = 15 →
  man_speed_kmph = 10 →
  incline_percent = 5 →
  speed_decrease_percent = 10 →
  ∃ (train_speed : ℝ), (train_speed ≥ 255.9 ∧ train_speed ≤ 256.1) ∧ 
    (train_length / crossing_time = (1 - speed_decrease_percent / 100) * train_speed + man_speed_kmph * (1000 / 3600)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_before_incline_l734_73435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_between_spheres_l734_73451

noncomputable def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem volume_between_spheres :
  let r₁ : ℝ := 4
  let r₂ : ℝ := 7
  volume_sphere r₂ - volume_sphere r₁ = 372 * Real.pi :=
by
  -- Unfold the definitions
  unfold volume_sphere
  -- Simplify the expressions
  simp [Real.pi]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_between_spheres_l734_73451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_range_l734_73490

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x < 0 then a^x else (3-a)*x + (1/2)*a

theorem increasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → 2 ≤ a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_range_l734_73490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l734_73429

-- Define proposition p
def p : Prop := ∀ a > 0, a ≠ 1 → (∀ x y, x < y → (fun x => a^x) x < (fun x => a^x) y)

-- Define proposition q
def q : Prop := ∀ x, x > Real.pi/4 ∧ x < 5*Real.pi/4 → Real.sin x > Real.cos x

-- Theorem to prove
theorem problem_solution : ¬p ∧ q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l734_73429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_proof_l734_73468

/-- The quadratic function g(x) = x^2 - 7x + k -/
def g (k : ℝ) (x : ℝ) : ℝ := x^2 - 7*x + k

/-- 3 is in the range of g(x) -/
def three_in_range (k : ℝ) : Prop := ∃ x, g k x = 3

/-- The smallest value of k such that 3 is in the range of g(x) -/
noncomputable def smallest_k : ℝ := 61/4

theorem smallest_k_proof :
  (∀ k < smallest_k, ¬(three_in_range k)) ∧ 
  (three_in_range smallest_k) := by
  sorry

#check smallest_k_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_proof_l734_73468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_ln_three_halves_l734_73484

noncomputable def f (x : ℝ) : ℝ := 2 / (x^2 - 1)

def lower_bound : ℝ := 2
def upper_bound : ℝ := 3

theorem area_equals_ln_three_halves :
  ∫ x in lower_bound..upper_bound, f x = Real.log (3/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_ln_three_halves_l734_73484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_x_eq_tan_x_squared_solutions_l734_73483

theorem tan_x_eq_tan_x_squared_solutions :
  ∃ (S : Finset ℝ), S.card = 3 ∧
  (∀ x ∈ S, 0 ≤ x ∧ x ≤ Real.arctan 1000 ∧ Real.tan x = Real.tan (x^2)) ∧
  (∀ x, 0 ≤ x ∧ x ≤ Real.arctan 1000 ∧ Real.tan x = Real.tan (x^2) → x ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_x_eq_tan_x_squared_solutions_l734_73483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_function_min_value_achieved_l734_73467

theorem min_value_of_function (x : ℝ) : 
  (x^2 + 2) / Real.sqrt (x^2 + 1) ≥ 2 := by sorry

theorem min_value_achieved : 
  ∃ x₀ : ℝ, (x₀^2 + 2) / Real.sqrt (x₀^2 + 1) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_function_min_value_achieved_l734_73467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_inequality_solution_set_l734_73415

theorem sin_inequality_solution_set (x : ℝ) :
  (Real.sqrt 2 + 2 * Real.sin x < 0) ↔ 
  (∃ k : ℤ, 2 * k * Real.pi + 5 * Real.pi / 4 < x ∧ x < 2 * k * Real.pi + 7 * Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_inequality_solution_set_l734_73415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dartboard_section_angle_l734_73459

/-- The measure in degrees of the central angle of a circular section, given the probability of a dart landing in that section -/
noncomputable def central_angle_measure (probability : ℝ) : ℝ := 360 * probability

/-- Theorem: If the probability of a dart landing in a section of a circular dartboard is 1/6, 
    then the measure of the central angle of this section is 60 degrees -/
theorem dartboard_section_angle (probability : ℝ) (h : probability = 1/6) : 
  central_angle_measure probability = 60 := by
  -- Unfold the definition of central_angle_measure
  unfold central_angle_measure
  -- Substitute the value of probability
  rw [h]
  -- Simplify the arithmetic
  norm_num

-- This line is removed as it's not necessary for the proof
-- #eval central_angle_measure (1/6)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dartboard_section_angle_l734_73459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_center_is_homothety_center_l734_73436

/-- Point in a geometric space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Sphere in a geometric space -/
structure Sphere where
  center : Point
  radius : ℝ

/-- Inversion is a geometric transformation that maps points from a sphere to another sphere -/
structure Inversion where
  center : Point
  source : Sphere
  target : Sphere

/-- Homothety is a geometric transformation that scales a figure with respect to a fixed point -/
structure Homothety where
  center : Point
  source : Sphere
  target : Sphere

/-- Given an inversion mapping sphere S to S*, its center is also the center of a homothety mapping S to S* -/
theorem inversion_center_is_homothety_center (i : Inversion) :
  ∃ h : Homothety, h.center = i.center ∧ h.source = i.source ∧ h.target = i.target := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_center_is_homothety_center_l734_73436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_coordinates_l734_73478

noncomputable section

-- Define points A and B
def A : ℝ × ℝ := (3, 1)
def B : ℝ × ℝ := (9, 7)

-- Define vector AB
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define point C
def C : ℝ × ℝ := (B.1 + AB.1 / 2, B.2 + AB.2 / 2)

-- Theorem statement
theorem point_C_coordinates :
  C = (12, 10) := by
  -- Proof goes here
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_coordinates_l734_73478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_pairs_properties_l734_73424

noncomputable def symmetric_pairs (a b : ℝ) := ((1 / Real.sqrt a, Real.sqrt b), (Real.sqrt b, 1 / Real.sqrt a))

theorem symmetric_pairs_properties :
  (∀ a b : ℝ, a > 0 → b > 0 → 
    symmetric_pairs a b = ((1 / Real.sqrt a, Real.sqrt b), (Real.sqrt b, 1 / Real.sqrt a))) ∧
  (symmetric_pairs 9 3 = ((1/3, Real.sqrt 3), (Real.sqrt 3, 1/3))) ∧
  (∀ y : ℝ, y > 0 → (symmetric_pairs 3 y).1 = (symmetric_pairs 3 y).2 → y = 1/3) ∧
  (∀ x : ℝ, x > 0 → (symmetric_pairs x 2).1 = (Real.sqrt 2, 1) → x = 1) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → 
    (symmetric_pairs a b).1 = (Real.sqrt 3, 3 * Real.sqrt 2) → 
    ((a = 1/3 ∧ b = 18) ∨ (a = 1/18 ∧ b = 3))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_pairs_properties_l734_73424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_line_equation_l734_73433

noncomputable def O : ℝ × ℝ := (0, 0)

noncomputable def A (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, 2 * Real.sin α)

noncomputable def B (α : ℝ) : ℝ × ℝ := (Real.cos α - Real.sqrt 3 * Real.sin α, Real.sin α + Real.sqrt 3 * Real.cos α)

noncomputable def P (α : ℝ) : ℝ × ℝ := (3 * Real.cos α - Real.sqrt 3 * Real.sin α, 3 * Real.sin α + Real.sqrt 3 * Real.cos α)

noncomputable def M : ℝ × ℝ := (4, 0)

theorem max_angle_line_equation (α : ℝ) :
  let (x, y) := P α
  (Real.sqrt 3 * x - y - 4 * Real.sqrt 3 = 0) ∨ (Real.sqrt 3 * x + y - 4 * Real.sqrt 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_line_equation_l734_73433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_isosceles_coincide_l734_73418

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define an isosceles triangle
structure IsoscelesTriangle where
  A : Point
  B : Point
  C : Point
  isIsosceles : (A.x - B.x)^2 + (A.y - B.y)^2 = (A.x - C.x)^2 + (A.y - C.y)^2

-- Define altitude, median, and angle bisector
noncomputable def altitude (t : IsoscelesTriangle) (p : Point) : Line :=
  sorry

noncomputable def median (t : IsoscelesTriangle) (p : Point) : Line :=
  sorry

noncomputable def angleBisector (t : IsoscelesTriangle) (p : Point) : Line :=
  sorry

-- Theorem statement
theorem not_all_isosceles_coincide :
  ¬ ∀ (t : IsoscelesTriangle) (p : Point),
    altitude t p = median t p ∧ median t p = angleBisector t p :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_isosceles_coincide_l734_73418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_and_angle_l734_73441

noncomputable def a : ℝ × ℝ := (3, -2)
noncomputable def b : ℝ × ℝ := (4, -6)
noncomputable def d : ℝ × ℝ := (3, -1)

noncomputable def line1 (t : ℝ) : ℝ × ℝ := (a.1 + t * d.1, a.2 + t * d.2)
noncomputable def line2 (s : ℝ) : ℝ × ℝ := (b.1 + s * d.1, b.2 + s * d.2)

noncomputable def distance_between_lines : ℝ := (11 * Real.sqrt 10) / 10

noncomputable def angle_between_vectors : ℝ := Real.arccos (11 / Real.sqrt 130)

theorem parallel_lines_distance_and_angle :
  (∃ (dist : ℝ), dist = distance_between_lines ∧
    dist = Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)) ∧
  (∃ (angle : ℝ), angle = angle_between_vectors ∧
    angle = Real.arccos ((d.1 * a.1 + d.2 * a.2) /
      (Real.sqrt (d.1^2 + d.2^2) * Real.sqrt (a.1^2 + a.2^2)))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_and_angle_l734_73441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l734_73438

-- Define set A
def A : Set ℝ := {1, 2, 3}

-- Define set B
def B : Set ℝ := {x : ℝ | x < 3}

-- State the theorem
theorem union_of_A_and_B :
  A ∪ B = Set.Iic 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l734_73438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_condition_neither_necessary_nor_sufficient_l734_73494

/-- Predicate to determine if an equation represents an ellipse -/
def IsEllipse (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ x y : ℝ, f x y ↔ (x^2 / a^2) + (y^2 / b^2) = 1

/-- The equation x^2 + y^2 * cos(α) = 1 -/
def EllipseEquation (α : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 * Real.cos α = 1

theorem ellipse_condition_neither_necessary_nor_sufficient :
  ¬(∀ α : ℝ, (0 < α ∧ α < π) ↔ IsEllipse (EllipseEquation α)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_condition_neither_necessary_nor_sufficient_l734_73494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_polar_points_l734_73404

noncomputable def polar_to_cartesian (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_between_polar_points :
  let A := polar_to_cartesian 1 (π/6)
  let B := polar_to_cartesian 2 (-π/2)
  distance A B = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_polar_points_l734_73404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_order_l734_73479

noncomputable def x₁ : ℝ := Real.log 2 / Real.log 3

noncomputable def x₂ : ℝ := Real.sqrt (1 / 2)

noncomputable def x₃ : ℝ := Real.exp (Real.log 3 * Real.log 3)

theorem x_order : x₁ < x₂ ∧ x₂ < x₃ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_order_l734_73479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dish_price_l734_73474

/-- Proves that the original price of a dish is $42 given the specified conditions --/
theorem dish_price (P : ℝ) 
  (johns_payment : ℝ → ℝ := λ x ↦ 0.9 * x + 0.15 * x)
  (janes_payment : ℝ → ℝ := λ x ↦ 0.9 * x + 0.15 * (0.9 * x))
  (payment_difference : johns_payment P = janes_payment P + 0.63) :
  P = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dish_price_l734_73474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_g_monotone_decreasing_positive_g_monotone_decreasing_negative_l734_73405

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 1 - 3 * x
noncomputable def g (x : ℝ) : ℝ := 1 / x + 2

-- Theorem for f
theorem f_monotone_decreasing :
  ∀ x y : ℝ, x < y → f x > f y :=
by sorry

-- Theorem for g on (0, +∞)
theorem g_monotone_decreasing_positive :
  ∀ x y : ℝ, 0 < x ∧ x < y → g x > g y :=
by sorry

-- Theorem for g on (-∞, 0)
theorem g_monotone_decreasing_negative :
  ∀ x y : ℝ, x < y ∧ y < 0 → g x > g y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_g_monotone_decreasing_positive_g_monotone_decreasing_negative_l734_73405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_proof_l734_73491

noncomputable def line1 (x y : ℝ) : Prop := 5 * x - 2 * y = 20
noncomputable def line2 (x y : ℝ) : Prop := 5 * x - 2 * y = -40
noncomputable def line3 (x y : ℝ) : Prop := 3 * x + y = 0

noncomputable def center_point : ℝ × ℝ := (-10/11, 30/11)

theorem circle_center_proof :
  let (x, y) := center_point
  5 * x - 2 * y = -10 ∧ 3 * x + y = 0 :=
by
  -- Unfold the definition of center_point
  unfold center_point
  -- Split the goal into two parts
  apply And.intro
  -- Prove 5 * x - 2 * y = -10
  · norm_num
  -- Prove 3 * x + y = 0
  · norm_num

#check circle_center_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_proof_l734_73491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_exp_satisfies_condition_l734_73417

-- Define the four functions
def f1 (x : ℝ) : ℝ := x^2
noncomputable def f2 (x : ℝ) : ℝ := Real.exp x
noncomputable def f3 (x : ℝ) : ℝ := Real.log x
noncomputable def f4 (x : ℝ) : ℝ := Real.cos x

-- Define the condition that needs to be satisfied
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x₁, ∃! x₂, f x₁ * f x₂ = 1

-- Theorem statement
theorem only_exp_satisfies_condition :
  satisfies_condition f2 ∧
  ¬satisfies_condition f1 ∧
  ¬satisfies_condition f3 ∧
  ¬satisfies_condition f4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_exp_satisfies_condition_l734_73417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_problem_l734_73496

/-- Calculates the time taken to cover the entire length of an escalator -/
noncomputable def escalatorTime (escalatorSpeed : ℝ) (escalatorLength : ℝ) (personSpeed : ℝ) : ℝ :=
  escalatorLength / (escalatorSpeed + personSpeed)

/-- Theorem: Given an escalator with speed 10 ft/sec and length 112 feet, 
    and a person walking at 4 ft/sec in the same direction, 
    the time taken to cover the entire length is 8 seconds. -/
theorem escalator_problem : 
  let escalatorSpeed : ℝ := 10
  let escalatorLength : ℝ := 112
  let personSpeed : ℝ := 4
  escalatorTime escalatorSpeed escalatorLength personSpeed = 8 := by
  -- Unfold the definition of escalatorTime
  unfold escalatorTime
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_problem_l734_73496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_addition_l734_73498

theorem perfect_square_addition (a : ℤ) :
  ∀ x y : ℤ, x^2 + a = y^2 ↔ ∃ α β : ℤ, a = α * β ∧ x = (β - α) / 2 ∧ y = (β + α) / 2 :=
by
  intro x y
  constructor
  · intro h
    sorry -- Proof of forward direction
  · intro ⟨α, β, h_a, h_x, h_y⟩
    sorry -- Proof of backward direction

#check perfect_square_addition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_addition_l734_73498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_arithmetic_sequence_l734_73465

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) : ℕ → ℤ
  | 0 => a₁
  | n + 1 => arithmetic_sequence a₁ d n + d

def sum_arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  (n : ℤ) * (2 * a₁ + (n - 1) * d) / 2

theorem min_sum_arithmetic_sequence :
  let a₁ := -28
  let d := 4
  ∀ k : ℕ, k ≥ 1 →
    sum_arithmetic_sequence a₁ d k ≥ sum_arithmetic_sequence a₁ d 7 ∧
    sum_arithmetic_sequence a₁ d k ≥ sum_arithmetic_sequence a₁ d 8 :=
by sorry

#check min_sum_arithmetic_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_arithmetic_sequence_l734_73465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_n_range_given_f_inequality_l734_73447

/-- The function f(x) = e^x + (1/2)x^2 - x -/
noncomputable def f (x : ℝ) : ℝ := Real.exp x + (1/2) * x^2 - x

/-- Theorem stating the range of n given the conditions -/
theorem n_range_given_f_inequality :
  ∀ n : ℝ, (∃ m : ℝ, f m ≤ 2 * n^2 - n) →
  n ∈ Set.Iic (-1/2) ∪ Set.Ici 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_n_range_given_f_inequality_l734_73447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_forming_function_range_l734_73480

/-- A function that can form a triangle --/
noncomputable def CanFormTriangle (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ x₃ : ℝ, ∃ (a b c : ℝ), a + b > c ∧ b + c > a ∧ c + a > b ∧
    a = f x₁ ∧ b = f x₂ ∧ c = f x₃

/-- The given function --/
noncomputable def f (t : ℝ) : ℝ → ℝ := λ x => (x^2 + t) / (x^2 + 1)

/-- The theorem to be proved --/
theorem triangle_forming_function_range :
  {t : ℝ | CanFormTriangle (f t)} = Set.Icc (1/2) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_forming_function_range_l734_73480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_power_of_two_permutation_l734_73448

/-- Represents a number as a list of its digits. -/
def digits (n : ℕ) : List ℕ := sorry

/-- Constructs a number from a list of digits. -/
def fromDigits (l : List ℕ) : ℕ := sorry

/-- Given natural numbers k and n where k > 3 and n > k, there does not exist a 
    permutation of the digits of 2^k that equals 2^n. -/
theorem no_power_of_two_permutation (k n : ℕ) (hk : k > 3) (hn : n > k) : 
  ¬ ∃ (perm : List ℕ), 
    perm.Perm (digits (2^k)) ∧ 
    fromDigits perm = 2^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_power_of_two_permutation_l734_73448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_on_circle_l734_73410

/-- The length of a chord intercepted on a circle by a line passing through the origin -/
theorem chord_length_on_circle : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + (y - Real.sqrt 3)^2 = 7}
  let line := {(x, y) : ℝ × ℝ | Real.sqrt 6 * x - Real.sqrt 3 * y = 0}
  let chord_length := 2 * Real.sqrt ((Real.sqrt 7)^2 - 1^2)
  chord_length = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_on_circle_l734_73410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_l734_73407

noncomputable def given_line (x : ℝ) : ℝ := 3/4 * x + 6

noncomputable def line_L (x c : ℝ) : ℝ := 3/4 * x + c

noncomputable def distance_parallel_lines (c₁ c₂ : ℝ) : ℝ := 
  |c₂ - c₁| / Real.sqrt ((3/4)^2 + 1)

theorem parallel_line_equation :
  ∃ (c : ℝ), (distance_parallel_lines 6 c = 5) ∧
  (c = 12.25 ∨ c = -0.25) := by
  sorry

#check parallel_line_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_l734_73407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_inequality_l734_73443

theorem negation_of_cosine_inequality :
  (¬ ∀ x : ℝ, Real.cos x ≤ 1) ↔ (∃ x₀ : ℝ, Real.cos x₀ > 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_inequality_l734_73443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_characterization_l734_73463

/-- A triangle with integer sides where one angle is twice another -/
structure SpecialTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  angle_condition : ∃ (α : ℝ), 0 < α ∧ α < π ∧ 
    (Real.sin α / Real.sin (2*α) = a.val / b.val) ∧ 
    (Real.sin α / Real.sin (π - 3*α) = c.val / b.val)

/-- The characterization of special triangles -/
theorem special_triangle_characterization (t : SpecialTriangle) :
  ∃ (k A B : ℕ+), 
    t.a = k * A^2 ∧
    t.b = k * A * B ∧
    t.c = k * (B^2 - A^2) ∧
    A < B ∧ B < 2 * A := by
  sorry

#check special_triangle_characterization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_characterization_l734_73463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_perimeter_52_l734_73446

/-- Calculates the perimeter of a rhombus given its diagonals -/
noncomputable def rhombusPerimeter (d1 d2 : ℝ) : ℝ :=
  4 * ((d1 / 2) ^ 2 + (d2 / 2) ^ 2).sqrt

/-- Theorem: A rhombus with diagonals of 10 inches and 24 inches has a perimeter of 52 inches -/
theorem rhombus_perimeter_52 : rhombusPerimeter 10 24 = 52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_perimeter_52_l734_73446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l734_73442

-- Define the hyperbola C
def C (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the asymptotes
def asymptote1 (x y : ℝ) : Prop := y = Real.sqrt 3 / 3 * x
def asymptote2 (x y : ℝ) : Prop := y = -Real.sqrt 3 / 3 * x

-- Define the points
structure Point where
  x : ℝ
  y : ℝ

-- Define the conditions
def conditions (A B P : Point) (l : ℝ) : Prop :=
  C P.x P.y ∧
  asymptote1 A.x A.y ∧
  asymptote2 B.x B.y ∧
  A.x > 0 ∧ B.x > 0 ∧ B.y < 0 ∧
  l ≥ 1/3 ∧ l ≤ 2 ∧
  (P.x - A.x, P.y - A.y) = l • (B.x - P.x, B.y - P.y)

-- Define the area of triangle AOB
noncomputable def area_AOB (A B : Point) : ℝ :=
  Real.sqrt 3 / 4 * (A.x * B.x)

-- Define the perimeter of triangle GQF₂
noncomputable def perimeter_GQF₂ (G Q : Point) : ℝ :=
  let F₂ : Point := ⟨2, 0⟩
  Real.sqrt ((G.x - F₂.x)^2 + G.y^2) +
  Real.sqrt ((Q.x - F₂.x)^2 + Q.y^2) +
  Real.sqrt ((G.x - Q.x)^2 + (G.y - Q.y)^2)

-- State the theorem
theorem hyperbola_theorem (A B P G Q : Point) (l : ℝ) :
  conditions A B P l →
  (∃ (A' B' P' : Point) (l' : ℝ), conditions A' B' P' l' ∧
    area_AOB A' B' = 4 * Real.sqrt 3 / 3) ∧
  (∃ (G' Q' : Point), C G'.x G'.y ∧ C Q'.x Q'.y ∧ G'.x < 0 ∧ Q'.x < 0 ∧
    perimeter_GQF₂ G' Q' = 16 * Real.sqrt 3 / 3) ∧
  (∀ (G'' Q'' : Point), C G''.x G''.y ∧ C Q''.x Q''.y ∧ G''.x < 0 ∧ Q''.x < 0 →
    perimeter_GQF₂ G'' Q'' ≥ 16 * Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l734_73442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_coordinates_l734_73440

def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (4, 0)

def on_extension (A B P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ P = (t * B.1 + (1 - t) * A.1, t * B.2 + (1 - t) * A.2)

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem point_P_coordinates :
  ∀ P : ℝ × ℝ,
  on_extension A B P →
  distance A P = 3 * distance P B →
  P = (11/2, -1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_coordinates_l734_73440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l734_73466

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (5 * x + 2) / Real.sqrt (2 * x - 10)

-- Define the domain of g
def domain_g : Set ℝ := {x | x > 5}

-- Theorem statement
theorem domain_of_g : 
  ∀ x : ℝ, x ∈ domain_g ↔ g x ∈ Set.univ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l734_73466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagonal_pyramid_surface_area_l734_73432

/-- The surface area of a pentagonal pyramid formed by slicing a regular pentagonal right prism -/
theorem pentagonal_pyramid_surface_area 
  (base_side_length : ℝ) 
  (prism_height : ℝ) 
  (base_area : ℝ) 
  (h1 : base_side_length = 10)
  (h2 : prism_height = 20)
  (h3 : base_area = (1/4) * Real.sqrt (5 * (5 + 2 * Real.sqrt 5)) * base_side_length^2) :
  let slice_height := prism_height / 3
  let slice_side_length := (2/3) * base_side_length
  let triangle_area := (Real.sqrt 3 / 4) * slice_side_length^2
  let pyramid_surface_area := base_area + 3 * triangle_area
  pyramid_surface_area = base_area + 300 * Real.sqrt 3 / 9 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagonal_pyramid_surface_area_l734_73432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_C_to_D_theorem_l734_73416

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a polygon with 7 vertices -/
structure Polygon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point
  G : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculates the area of a polygon -/
noncomputable def area (p : Polygon) : ℝ := sorry

/-- Checks if a polygon has square corners -/
def hasSquareCorners (p : Polygon) : Prop := sorry

/-- Theorem about the distance from C to D in a specific polygon -/
theorem distance_C_to_D_theorem (p : Polygon) 
  (h1 : hasSquareCorners p)
  (h2 : distance p.E p.F = 20)
  (h3 : distance p.A p.B = 10)
  (h4 : distance p.A p.G = distance p.G p.F)
  (h5 : area p = 280)
  (h6 : ∃ D' : Point, distance p.A D' + distance D' p.D = distance p.A p.D ∧ 
                      area { p with D := D' } = area { p with A := D' } ) :
  distance p.C p.D = 40 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_C_to_D_theorem_l734_73416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_point_lambda_l734_73431

-- Define the space
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V]

-- Define the points
variable (A B C O P : V)

-- Define the conditions
variable (h1 : A ≠ B ∧ A ≠ C ∧ B ≠ C)
variable (h2 : O ∉ affineSpan ℝ {A, B, C})
variable (h3 : ∃ (l : ℝ), P - O = (1/5 : ℝ) • (A - O) + (2/3 : ℝ) • (B - O) + l • (C - O))
variable (h4 : P ∈ affineSpan ℝ {A, B, C})

-- State the theorem
theorem coplanar_point_lambda :
  ∃ (l : ℝ), P - O = (1/5 : ℝ) • (A - O) + (2/3 : ℝ) • (B - O) + l • (C - O) ∧ l = 2/15 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_point_lambda_l734_73431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_difference_approx_fifty_percent_l734_73439

/-- The side length of the square field -/
noncomputable def side_length : ℝ := 2

/-- Jerry's path length -/
noncomputable def jerry_path : ℝ := side_length + 1 + (side_length + 1 / Real.sqrt 2)

/-- Silvia's path length (diagonal of the square) -/
noncomputable def silvia_path : ℝ := side_length * Real.sqrt 2

/-- The percentage difference between Jerry's and Silvia's paths -/
noncomputable def path_difference_percentage : ℝ := 
  (jerry_path - silvia_path) / jerry_path * 100

theorem path_difference_approx_fifty_percent :
  |path_difference_percentage - 50| < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_difference_approx_fifty_percent_l734_73439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_trigonometric_powers_l734_73492

open Real

theorem order_of_trigonometric_powers (a b : ℝ) 
  (h1 : 0 < b) (h2 : b < 1) (h3 : 0 < a) (h4 : a < π/4) :
  (sin a) ^ (log (sin a) / log b) < (sin a) ^ (log (cos a) / log b) ∧ 
  (sin a) ^ (log (cos a) / log b) < (cos a) ^ (log (cos a) / log b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_trigonometric_powers_l734_73492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_puzzle_no_solution_l734_73475

/-- Represents a number formed by permuting a set of digits -/
structure PermutedNumber where
  value : Nat

/-- The difference of two numbers formed by permuting the same set of digits -/
def permuted_difference (a b : PermutedNumber) : Nat := 
  if a.value ≥ b.value then a.value - b.value else b.value - a.value

/-- Axiom: The difference of two numbers formed by permuting the same set of digits is always divisible by 9 -/
axiom permuted_difference_div_by_9 (a b : PermutedNumber) : 
  (permuted_difference a b) % 9 = 0

/-- The product of 2012 and 2013 -/
def product_2012_2013 : Nat := 2012 * 2013

/-- Theorem: The puzzle APELSIN - SPANIEL = 2012 * 2013 has no solution -/
theorem puzzle_no_solution (apelsin spaniel : PermutedNumber) : 
  permuted_difference apelsin spaniel ≠ product_2012_2013 := by
  sorry

#eval product_2012_2013

end NUMINAMATH_CALUDE_ERRORFEEDBACK_puzzle_no_solution_l734_73475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_perpendicular_tangent_l734_73472

-- Define the function f(x) = ln x + x^2
noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := 1 / x + 2 * x

-- Define the slope of the tangent line
noncomputable def tangent_slope (x : ℝ) : ℝ := f_derivative x

-- Define the slope of the perpendicular line
noncomputable def perpendicular_slope (x : ℝ) : ℝ := -1 / tangent_slope x

-- Theorem statement
theorem exists_perpendicular_tangent :
  ∃ x : ℝ, x > 0 ∧ perpendicular_slope x = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_perpendicular_tangent_l734_73472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_error_percent_l734_73430

-- Define the actual length and width of the rectangle
variable (L W : ℝ)

-- Define the measured length (6% in excess)
def measured_length (L : ℝ) : ℝ := 1.06 * L

-- Define the measured width (5% in deficit)
def measured_width (W : ℝ) : ℝ := 0.95 * W

-- Define the actual area
def actual_area (L W : ℝ) : ℝ := L * W

-- Define the calculated area based on measurements
def calculated_area (L W : ℝ) : ℝ := measured_length L * measured_width W

-- Define the error in area
def area_error (L W : ℝ) : ℝ := calculated_area L W - actual_area L W

-- Define the error percent in area
noncomputable def error_percent (L W : ℝ) : ℝ := (area_error L W / actual_area L W) * 100

-- Theorem statement
theorem area_error_percent (L W : ℝ) :
  error_percent L W = 0.7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_error_percent_l734_73430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_function_l734_73402

-- Define the theorem
theorem min_value_of_function (x y : ℝ) : 
  -2 < x ∧ x < 2 → 
  -2 < y ∧ y < 2 → 
  x * y = -1 → 
  (∀ a b : ℝ, -2 < a ∧ a < 2 → -2 < b ∧ b < 2 → a * b = -1 → 
    4 / (4 - x^2) + 9 / (9 - y^2) ≤ 4 / (4 - a^2) + 9 / (9 - b^2)) → 
  4 / (4 - x^2) + 9 / (9 - y^2) = 12 / 5 := by
  sorry

-- Check the theorem
#check min_value_of_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_function_l734_73402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_x_plus_2y_l734_73408

theorem max_value_x_plus_2y (x y : ℝ) (h : (2 : ℝ)^x + (4 : ℝ)^y = 1) : 
  ∀ z, x + 2*y ≤ z → z ≤ -2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_x_plus_2y_l734_73408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_length_l734_73426

-- Define a right triangle with one leg of 15 inches and an angle of 45° opposite that leg
def RightTriangle45 : Set (ℝ × ℝ × ℝ) := {t | ∃ (a b c : ℝ), 
  a^2 + b^2 = c^2 ∧  -- Pythagorean theorem for right triangle
  a = 15 ∧           -- One leg is 15 inches
  Real.cos (Real.pi/4) = b / c ∧  -- Angle opposite to leg a is 45° (π/4 radians)
  t = (a, b, c)
}

-- Theorem statement
theorem hypotenuse_length : 
  ∃ (t : ℝ × ℝ × ℝ), t ∈ RightTriangle45 ∧ t.2.2 = 15 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_length_l734_73426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_and_division_problem_l734_73403

theorem remainder_and_division_problem (x y : ℕ) 
  (hx : x > 0)
  (hy : y > 0)
  (h1 : x % y = 8)
  (h2 : (x : ℝ) / (y : ℝ) = 76.4) : 
  y = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_and_division_problem_l734_73403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l734_73422

-- Define the inequality function
noncomputable def f (x m : ℝ) := 2 * x + m + 8 / (x - 1)

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x > 1 → f x m > 0) ↔ m > -10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l734_73422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_decrease_percentage_l734_73499

/-- The percentage decrease in revenue from old_revenue to new_revenue -/
noncomputable def percentage_decrease (old_revenue new_revenue : ℝ) : ℝ :=
  ((old_revenue - new_revenue) / old_revenue) * 100

/-- Theorem stating that the percentage decrease in revenue from $85.0 billion to $48.0 billion is approximately 43.53% -/
theorem revenue_decrease_percentage :
  abs (percentage_decrease 85.0 48.0 - 43.53) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_decrease_percentage_l734_73499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_questions_to_determine_pair_l734_73449

def is_valid_pair (x y : ℕ) : Prop := 0 < x ∧ x ≤ 20 ∧ 0 < y ∧ y ≤ 23

def question (a b x y : ℕ) : Prop := x ≤ a ∧ y ≤ b

def strategy (N : ℕ) : Prop :=
  ∃ (f : ℕ → ℕ × ℕ),
    ∀ (x y : ℕ), is_valid_pair x y →
      ∃ (n : ℕ), n ≤ N ∧
        ∀ (m : ℕ), m < n →
          let (a, b) := f m;
          question a b x y ∨ ¬question a b x y

theorem min_questions_to_determine_pair :
  ∃ (N : ℕ), strategy N ∧ ∀ (M : ℕ), M < N → ¬strategy M :=
sorry

#check min_questions_to_determine_pair

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_questions_to_determine_pair_l734_73449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spaces_to_win_l734_73428

def board_length : ℕ := 48

def first_move : ℤ := 8
def second_move : ℤ := 2 - 5
def third_move : ℤ := 6

def total_moved : ℤ := first_move + second_move + third_move

theorem spaces_to_win : (↑board_length : ℤ) - total_moved = 37 := by
  -- Convert board_length to ℤ for consistent subtraction
  have h1 : (↑board_length : ℤ) = 48 := rfl
  have h2 : total_moved = 11 := by
    unfold total_moved first_move second_move third_move
    ring
  rw [h1, h2]
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spaces_to_win_l734_73428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_morse_high_school_seniors_l734_73477

theorem morse_high_school_seniors (total_students : ℝ) 
  (senior_count : ℝ) (non_senior_count : ℝ) :
  non_senior_count = 1500 →
  total_students = senior_count + non_senior_count →
  (0.4 * senior_count + 0.1 * non_senior_count) / total_students = 0.15 →
  senior_count = 300 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_morse_high_school_seniors_l734_73477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_two_l734_73450

def sequenceterm : ℕ → ℕ
  | 0 => 20
  | n + 1 => 
    let t := sequenceterm n
    if t % 2 = 0 then t / 2 else 3 * t + 1

theorem tenth_term_is_two :
  sequenceterm 9 = 2 := by
  -- Compute the sequence terms
  have t1 : sequenceterm 0 = 20 := rfl
  have t2 : sequenceterm 1 = 10 := by simp [sequenceterm]
  have t3 : sequenceterm 2 = 5 := by simp [sequenceterm]
  have t4 : sequenceterm 3 = 16 := by simp [sequenceterm]
  have t5 : sequenceterm 4 = 8 := by simp [sequenceterm]
  have t6 : sequenceterm 5 = 4 := by simp [sequenceterm]
  have t7 : sequenceterm 6 = 2 := by simp [sequenceterm]
  have t8 : sequenceterm 7 = 1 := by simp [sequenceterm]
  have t9 : sequenceterm 8 = 4 := by simp [sequenceterm]
  have t10 : sequenceterm 9 = 2 := by simp [sequenceterm]
  exact t10

#eval sequenceterm 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_two_l734_73450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_is_one_fifth_l734_73454

/-- The distance from a point to a line in 2D space -/
noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  abs (a * x₀ + b * y₀ + c) / Real.sqrt (a^2 + b^2)

/-- The point P -/
def P : ℝ × ℝ := (-1, 1)

/-- The line l: y = -3/4x in standard form Ax + By + C = 0 -/
def line_coefficients : ℝ × ℝ × ℝ := (3, 4, 0)

theorem distance_point_to_line_is_one_fifth :
  distance_point_to_line P.1 P.2 line_coefficients.1 line_coefficients.2.1 line_coefficients.2.2 = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_is_one_fifth_l734_73454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_angle_l734_73493

open Real

-- Define the equation
noncomputable def equation (x : ℝ) : Prop :=
  tan (6 * x) = (sin x - cos x) / (sin x + cos x)

-- Convert degrees to radians
noncomputable def deg_to_rad (d : ℝ) : ℝ := d * π / 180

-- State the theorem
theorem smallest_positive_angle :
  ∃ (x : ℝ), x > 0 ∧ x < deg_to_rad 360 ∧ equation x ∧
  (∀ (y : ℝ), 0 < y ∧ y < x → ¬equation y) ∧
  x = deg_to_rad 7.5 := by
  sorry

#check smallest_positive_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_angle_l734_73493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_foci_coincide_l734_73460

/-- The squared semi-major axis of the ellipse -/
noncomputable def a_ellipse_squared : ℝ := 25

/-- The squared semi-major axis of the hyperbola -/
noncomputable def a_hyperbola_squared : ℝ := 196 / 49

/-- The squared semi-minor axis of the hyperbola -/
noncomputable def b_hyperbola_squared : ℝ := 121 / 49

/-- The equation of the ellipse -/
def ellipse_equation (x y b : ℝ) : Prop :=
  x^2 / a_ellipse_squared + y^2 / b^2 = 1

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / a_hyperbola_squared - y^2 / b_hyperbola_squared = 1

/-- The theorem stating that if the foci of the ellipse and hyperbola coincide,
    then the squared semi-minor axis of the ellipse is 908/49 -/
theorem ellipse_hyperbola_foci_coincide :
  ∃ b : ℝ, (∀ x y : ℝ, ellipse_equation x y b ↔ hyperbola_equation x y) →
  b^2 = 908 / 49 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_foci_coincide_l734_73460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walkers_speed_to_work_l734_73464

/-- Calculates the speed of the outbound trip given the total distance, total time, and return speed --/
noncomputable def outbound_speed (total_distance : ℝ) (total_time : ℝ) (return_speed : ℝ) : ℝ :=
  let return_time := total_distance / (2 * return_speed)
  let outbound_time := total_time - return_time
  let outbound_distance := total_distance / 2
  outbound_distance / outbound_time

theorem walkers_speed_to_work :
  outbound_speed 48 1 40 = 60 := by
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walkers_speed_to_work_l734_73464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_non_range_value_l734_73476

/-- The function g defined as (ax + b) / (cx + d) -/
noncomputable def g (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

/-- Theorem stating that under given conditions, 22 is the unique number not in the range of g -/
theorem unique_non_range_value (a b c d : ℝ) :
  a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 →
  g a b c d 13 = 13 →
  g a b c d 31 = 31 →
  (∀ x : ℝ, x ≠ -d/c → g a b c d (g a b c d x) = x) →
  ∃! y : ℝ, (∀ x : ℝ, g a b c d x ≠ y) ∧ y = 22 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_non_range_value_l734_73476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l734_73486

/-- The function f(x) defined for x ≥ 3 -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (Real.pi / 4 * Real.sin (Real.sqrt (x - 3) + 2 * x + 2))

/-- Theorem stating the range of f(x) -/
theorem f_range :
  ∀ x : ℝ, x ≥ 3 → Real.sqrt 2 ≤ f x ∧ f x ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l734_73486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l734_73461

theorem inequality_solution : 
  {x : ℤ | x > 0 ∧ -3 ≤ 5 - 2*x ∧ 5 - 2*x < 3} = {2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l734_73461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l734_73444

noncomputable def f (x y : ℝ) : ℝ := (x * y) / (x^2 - y^2)

theorem max_value_of_f :
  ∃ (max : ℝ), max = 3/8 ∧
  ∀ (x y : ℝ), 0.1 ≤ x ∧ x ≤ 0.6 ∧ 0.2 ≤ y ∧ y ≤ 0.5 ∧ x^2 ≠ y^2 →
  f x y ≤ max :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l734_73444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l734_73458

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2 * x * (1 - x)
  else if x < 0 then 2 * x * (1 + x)
  else 0  -- Define f(0) = 0 to make it a total function

-- State the theorem
theorem odd_function_property :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x : ℝ, x > 0 → f x = 2 * x * (1 - x)) ∧
  (∀ x : ℝ, x < 0 → f x = 2 * x * (1 + x)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l734_73458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_cities_l734_73452

theorem distance_between_cities (S : ℕ) : S > 0 ∧ 
  (∀ x : ℕ, x ≤ S → Nat.gcd x (S - x) ∈ ({1, 3, 13} : Set ℕ)) ∧
  (∀ T : ℕ, T > 0 → (∀ y : ℕ, y ≤ T → Nat.gcd y (T - y) ∈ ({1, 3, 13} : Set ℕ)) → S ≤ T) →
  S = 39 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_cities_l734_73452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_A_counterexample_statement_B_statement_C_statement_D_l734_73413

-- Statement A (incorrect)
theorem statement_A_counterexample : ∃ a b c : ℝ, a > b ∧ c < 0 ∧ a^2 * c ≥ b^2 * c := by sorry

-- Statement B (correct)
theorem statement_B : ∀ a b c : ℝ, a > b → c < 0 → a^3 * c < b^3 * c := by sorry

-- Statement C (correct)
theorem statement_C : ∀ a b : ℝ, a < b → b < 0 → a^2 > a * b ∧ a * b > b^2 := by sorry

-- Statement D (incorrect)
noncomputable def f (x : ℝ) : ℝ := (x^2 + 5) / Real.sqrt (x^2 + 4)

theorem statement_D : ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x : ℝ, f x = m) ∧ m = 5/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_A_counterexample_statement_B_statement_C_statement_D_l734_73413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_product_reciprocal_exponents_l734_73482

theorem power_product_reciprocal_exponents :
  let r : ℚ := 5 / 6
  (r ^ 4) * (r ^ (-4 : ℤ)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_product_reciprocal_exponents_l734_73482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_irreducible_l734_73487

theorem fraction_irreducible (n : ℤ) : Int.gcd (2*n + 1) (3*n + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_irreducible_l734_73487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rooks_correct_l734_73400

/-- The maximum number of rooks that can be placed on a k × n chessboard
    such that each rook attacks exactly one other rook -/
def max_rooks (k n : ℕ) : ℕ :=
  if n > 2 * k then 2 * k
  else 2 * ((k + n) / 3)

theorem max_rooks_correct (k n : ℕ) (h : k ≤ n) :
  (max_rooks k n = 
    if n > 2 * k 
    then 2 * k
    else 2 * ((k + n) / 3)) ∧
  max_rooks k n ≤ 2 * k ∧
  (∀ m : ℕ, m > max_rooks k n → 
    ¬ ∃ (placement : Fin m → Fin k × Fin n),
      (∀ i j : Fin m, i ≠ j → 
        (placement i).1 = (placement j).1 ∨ (placement i).2 = (placement j).2) ∧
      (∀ i : Fin m, ∃! j : Fin m, i ≠ j ∧
        ((placement i).1 = (placement j).1 ∨ (placement i).2 = (placement j).2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rooks_correct_l734_73400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_x_sqrt_5x_eq_3x_l734_73401

theorem largest_x_sqrt_5x_eq_3x : 
  ∃ (x : ℝ), x = 5/9 ∧ 
  (∀ (y : ℝ), Real.sqrt (5*y) = 3*y → y ≤ x) ∧
  Real.sqrt (5*x) = 3*x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_x_sqrt_5x_eq_3x_l734_73401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_in_interval_l734_73409

noncomputable def f (x : ℝ) : ℝ := 6 / x - x^2

theorem f_has_zero_in_interval :
  ∃ c ∈ Set.Ioo 1 2, f c = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_in_interval_l734_73409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_roots_of_unity_l734_73423

/-- The polynomial z^6 - z^3 + 1 -/
def f (z : ℂ) : ℂ := z^6 - z^3 + 1

/-- n^th root of unity -/
def is_nth_root_of_unity (z : ℂ) (n : ℕ) : Prop := z^n = 1

/-- All roots of f are n^th roots of unity -/
def all_roots_are_nth_roots_of_unity (n : ℕ) : Prop :=
  ∀ z : ℂ, f z = 0 → is_nth_root_of_unity z n

/-- Predicate for the smallest positive n where all roots are n^th roots of unity -/
def is_smallest_n (n : ℕ) : Prop :=
  all_roots_are_nth_roots_of_unity n ∧ n > 0 ∧
  ∀ m, m < n → ¬(all_roots_are_nth_roots_of_unity m ∧ m > 0)

theorem smallest_n_for_roots_of_unity :
  is_smallest_n 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_roots_of_unity_l734_73423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_pile_volume_calculation_l734_73489

noncomputable def sand_pile_volume (diameter : ℝ) (height_ratio : ℝ) : ℝ :=
  let radius := diameter / 2
  let height := height_ratio * diameter
  (1 / 3) * Real.pi * radius^2 * height

theorem sand_pile_volume_calculation :
  sand_pile_volume 12 0.6 = 86.4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_pile_volume_calculation_l734_73489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solution_set_l734_73485

theorem integer_solution_set : 
  {(x, y, z) : ℤ × ℤ × ℤ | x - y * z = 1 ∧ x * z + y = 2} = 
  {(1, 0, 2), (1, 2, 0)} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solution_set_l734_73485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l734_73420

noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

def equation (x : ℝ) : Prop :=
  3 * x - 2 * (floor x) + 4 = 0

theorem equation_solutions :
  {x : ℝ | equation x} = {-4, -14/3, -16/3} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l734_73420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_ceiling_sqrt_10_to_50_l734_73497

-- Define the ceiling function
noncomputable def ceiling (x : ℝ) : ℤ := Int.ceil x

-- Define the sum of ceiling of square roots from 10 to 50
noncomputable def sum_ceiling_sqrt : ℕ → ℤ
  | 0 => 0
  | n+1 => if n+1 ≥ 10 ∧ n+1 ≤ 50 then
             ceiling (Real.sqrt (n+1 : ℝ)) + sum_ceiling_sqrt n
           else
             sum_ceiling_sqrt n

-- Theorem statement
theorem sum_ceiling_sqrt_10_to_50 :
  sum_ceiling_sqrt 50 = 238 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_ceiling_sqrt_10_to_50_l734_73497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_cubic_polynomial_unique_l734_73488

/-- A monic cubic polynomial with real coefficients -/
def MonicCubicPolynomial (a b c : ℝ) : ℂ → ℂ := fun x ↦ x^3 + a*x^2 + b*x + c

theorem monic_cubic_polynomial_unique 
  (q : ℂ → ℂ) 
  (h_monic : ∀ x, q x = x^3 + (q 1 - 1) * x^2 + (q 1 - 3*q 0 - 1) * x + q 0) 
  (h_root : q (5 - 3*Complex.I) = 0) 
  (h_const : q 0 = -40) :
  ∀ x, q x = x^3 - (190/17)*x^2 + (778/17)*x - 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_cubic_polynomial_unique_l734_73488
