import Mathlib

namespace NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l2885_288520

theorem sin_cos_difference_equals_half : 
  Real.sin (43 * π / 180) * Real.cos (13 * π / 180) - 
  Real.cos (43 * π / 180) * Real.sin (13 * π / 180) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l2885_288520


namespace NUMINAMATH_CALUDE_max_saturdays_is_five_l2885_288515

/-- Represents the possible number of days in a month -/
inductive MonthLength
  | Days28
  | Days29
  | Days30
  | Days31

/-- Represents the day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Calculates the number of Saturdays in a month -/
def saturdays_in_month (length : MonthLength) (start : DayOfWeek) : Nat :=
  sorry

/-- The maximum number of Saturdays in any month -/
def max_saturdays : Nat := 5

/-- Theorem: The maximum number of Saturdays in any month is 5 -/
theorem max_saturdays_is_five :
  ∀ (length : MonthLength) (start : DayOfWeek),
    saturdays_in_month length start ≤ max_saturdays :=
  sorry

end NUMINAMATH_CALUDE_max_saturdays_is_five_l2885_288515


namespace NUMINAMATH_CALUDE_binomial_9_choose_5_l2885_288525

theorem binomial_9_choose_5 : Nat.choose 9 5 = 126 := by sorry

end NUMINAMATH_CALUDE_binomial_9_choose_5_l2885_288525


namespace NUMINAMATH_CALUDE_inequality_range_l2885_288527

theorem inequality_range (x : ℝ) : 
  (∀ p : ℝ, 0 ≤ p ∧ p ≤ 4 → x^2 + p*x > 4*x + p - 3) ↔ (x > 3 ∨ x < -1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l2885_288527


namespace NUMINAMATH_CALUDE_forty_ab_value_l2885_288512

theorem forty_ab_value (a b : ℝ) (h : 4 * a = 5 * b ∧ 5 * b = 30) : 40 * a * b = 1800 := by
  sorry

end NUMINAMATH_CALUDE_forty_ab_value_l2885_288512


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2885_288591

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I : ℂ) * (((a - Complex.I) / (1 - Complex.I)).im) = ((a - Complex.I) / (1 - Complex.I)) → 
  a = -1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2885_288591


namespace NUMINAMATH_CALUDE_curve_W_and_rectangle_perimeter_l2885_288546

-- Define the curve W
def W (x y : ℝ) : Prop := |y| = Real.sqrt (x^2 + (y - 1/2)^2)

-- Define a rectangle with three vertices on W
def RectangleOnW (A B C D : ℝ × ℝ) : Prop :=
  W A.1 A.2 ∧ W B.1 B.2 ∧ W C.1 C.2 ∧
  (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0 ∧
  (A.1 - D.1) * (C.1 - D.1) + (A.2 - D.2) * (C.2 - D.2) = 0

-- Theorem statement
theorem curve_W_and_rectangle_perimeter 
  (A B C D : ℝ × ℝ) (h : RectangleOnW A B C D) :
  (∀ x y : ℝ, W x y ↔ y = x^2 + 1/4) ∧ 
  2 * (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) + 
       Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)) > 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_curve_W_and_rectangle_perimeter_l2885_288546


namespace NUMINAMATH_CALUDE_max_t_value_l2885_288501

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x * |x - a| - x

-- State the theorem
theorem max_t_value (a : ℝ) (h : a ≤ 1) :
  (∃ t : ℝ, t = 1 + Real.sqrt 7 ∧
   (∀ x : ℝ, x ∈ Set.Icc 0 t → -1 ≤ f a x ∧ f a x ≤ 6) ∧
   (∀ t' : ℝ, t' > t →
     ∃ x : ℝ, x ∈ Set.Icc 0 t' ∧ (f a x < -1 ∨ f a x > 6))) ∧
  (∀ a' : ℝ, a' ≤ 1 →
    ∀ t : ℝ, (∀ x : ℝ, x ∈ Set.Icc 0 t → -1 ≤ f a' x ∧ f a' x ≤ 6) →
      t ≤ 1 + Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_max_t_value_l2885_288501


namespace NUMINAMATH_CALUDE_exists_integer_divisible_by_15_with_sqrt_between_30_and_30_5_l2885_288554

theorem exists_integer_divisible_by_15_with_sqrt_between_30_and_30_5 :
  ∃ n : ℕ+, 15 ∣ n ∧ 30 ≤ (n : ℝ).sqrt ∧ (n : ℝ).sqrt < 30.5 := by
  sorry

end NUMINAMATH_CALUDE_exists_integer_divisible_by_15_with_sqrt_between_30_and_30_5_l2885_288554


namespace NUMINAMATH_CALUDE_divisibility_implication_l2885_288547

theorem divisibility_implication (m n : ℤ) : 
  (11 ∣ (5*m + 3*n)) → (11 ∣ (9*m + n)) := by
sorry

end NUMINAMATH_CALUDE_divisibility_implication_l2885_288547


namespace NUMINAMATH_CALUDE_parametric_eq_normal_l2885_288549

/-- The parametric equation of a plane -/
def plane_parametric (s t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 2*s - 3*t, 4 - s + 2*t, 1 - 3*s - t)

/-- The normal form equation of a plane -/
def plane_normal (x y z : ℝ) : Prop :=
  5*x + 11*y + 7*z - 61 = 0

/-- Theorem stating that the parametric and normal form equations represent the same plane -/
theorem parametric_eq_normal :
  ∀ (x y z : ℝ), (∃ (s t : ℝ), plane_parametric s t = (x, y, z)) ↔ plane_normal x y z :=
by sorry

end NUMINAMATH_CALUDE_parametric_eq_normal_l2885_288549


namespace NUMINAMATH_CALUDE_maximum_marks_correct_l2885_288598

/-- The maximum marks in an exam where:
    1. The passing threshold is 33% of the maximum marks.
    2. A student got 92 marks.
    3. The student failed by 40 marks (i.e., needed 40 more marks to pass). -/
def maximum_marks : ℕ := 400

/-- The passing threshold as a fraction of the maximum marks -/
def passing_threshold : ℚ := 33 / 100

/-- The marks obtained by the student -/
def obtained_marks : ℕ := 92

/-- The additional marks needed to pass -/
def additional_marks_needed : ℕ := 40

theorem maximum_marks_correct :
  maximum_marks * (passing_threshold : ℚ) = obtained_marks + additional_marks_needed := by
  sorry

end NUMINAMATH_CALUDE_maximum_marks_correct_l2885_288598


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l2885_288570

theorem pure_imaginary_product (x : ℝ) : 
  (∃ b : ℝ, (x + 2*I)*((x + 2) + 2*I)*((x + 4) + 2*I) = b*I) ↔ (x = -4 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l2885_288570


namespace NUMINAMATH_CALUDE_original_class_size_l2885_288596

theorem original_class_size (original_avg : ℝ) (new_students : ℕ) (new_avg : ℝ) (avg_decrease : ℝ) : 
  original_avg = 50 →
  new_students = 12 →
  new_avg = 32 →
  avg_decrease = 4 →
  ∃ (original_size : ℕ), 
    (original_size : ℝ) * original_avg + (new_students : ℝ) * new_avg = 
    (original_size + new_students : ℝ) * (original_avg - avg_decrease) ∧
    original_size = 42 :=
by sorry


end NUMINAMATH_CALUDE_original_class_size_l2885_288596


namespace NUMINAMATH_CALUDE_only_sunrise_certain_l2885_288507

-- Define the type for events
inductive Event
  | MovieTicket
  | TVAdvertisement
  | Rain
  | Sunrise

-- Define what it means for an event to be certain
def is_certain (e : Event) : Prop :=
  match e with
  | Event.Sunrise => true
  | _ => false

-- Theorem stating that only the sunrise event is certain
theorem only_sunrise_certain :
  ∀ (e : Event), is_certain e ↔ e = Event.Sunrise :=
by
  sorry

end NUMINAMATH_CALUDE_only_sunrise_certain_l2885_288507


namespace NUMINAMATH_CALUDE_number_problem_l2885_288562

theorem number_problem : ∃ x : ℚ, x / 3 = x - 30 ∧ x = 45 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2885_288562


namespace NUMINAMATH_CALUDE_quadratic_root_l2885_288518

theorem quadratic_root (m : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - (m + 3) * x + m = 0 ∧ x = 1) → 
  (∃ y : ℝ, 2 * y^2 - (m + 3) * y + m = 0 ∧ y = (-m - 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_l2885_288518


namespace NUMINAMATH_CALUDE_a_range_l2885_288555

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x + 1 else a^x

/-- The condition for the function -/
def strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) > 0

/-- The theorem stating the range of a -/
theorem a_range (a : ℝ) :
  (strictly_increasing (f a)) ↔ (3/2 ≤ a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_a_range_l2885_288555


namespace NUMINAMATH_CALUDE_minimized_surface_area_sum_l2885_288548

/-- Represents a tetrahedron with specific properties -/
structure Tetrahedron where
  /-- One face has sides of length 3, 4, and 5 -/
  base_sides : Fin 3 → ℝ
  base_sides_values : base_sides = ![3, 4, 5]
  /-- The volume of the tetrahedron is 24 -/
  volume : ℝ
  volume_value : volume = 24

/-- Represents the surface area of the tetrahedron in the form a√b + c -/
structure SurfaceArea where
  a : ℕ
  b : ℕ
  c : ℕ
  /-- b is not divisible by the square of any prime -/
  b_squarefree : ∀ p : ℕ, Prime p → ¬(p^2 ∣ b)

/-- The main theorem stating the sum of a, b, and c for the minimized surface area -/
theorem minimized_surface_area_sum (t : Tetrahedron) :
  ∃ (sa : SurfaceArea), (∀ other_sa : SurfaceArea, 
    sa.a * Real.sqrt sa.b + sa.c ≤ other_sa.a * Real.sqrt other_sa.b + other_sa.c) → 
    sa.a + sa.b + sa.c = 157 := by
  sorry

end NUMINAMATH_CALUDE_minimized_surface_area_sum_l2885_288548


namespace NUMINAMATH_CALUDE_divisibility_implies_sum_ten_l2885_288503

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def number (C F : ℕ) : ℕ := C * 1000000 + 854000 + F * 100 + 72

theorem divisibility_implies_sum_ten (C F : ℕ) 
  (h_C : is_digit C) (h_F : is_digit F) 
  (h_div_8 : number C F % 8 = 0) 
  (h_div_9 : number C F % 9 = 0) : 
  C + F = 10 := by
sorry

end NUMINAMATH_CALUDE_divisibility_implies_sum_ten_l2885_288503


namespace NUMINAMATH_CALUDE_jeffrey_steps_to_mailbox_l2885_288578

/-- Represents Jeffrey's walking pattern -/
structure WalkingPattern where
  forward : ℕ
  backward : ℕ

/-- Calculates the total steps taken given a walking pattern and distance -/
def totalSteps (pattern : WalkingPattern) (distance : ℕ) : ℕ :=
  let effectiveStep := pattern.forward - pattern.backward
  let cycles := distance / effectiveStep
  cycles * (pattern.forward + pattern.backward)

/-- Theorem: Jeffrey's total steps to the mailbox -/
theorem jeffrey_steps_to_mailbox :
  let pattern : WalkingPattern := { forward := 3, backward := 2 }
  let distance : ℕ := 66
  totalSteps pattern distance = 330 := by
  sorry


end NUMINAMATH_CALUDE_jeffrey_steps_to_mailbox_l2885_288578


namespace NUMINAMATH_CALUDE_tangent_line_sum_l2885_288519

/-- Given a function f: ℝ → ℝ with a tangent line y = -x + 6 at x=2, 
    prove that f(2) + f'(2) = 3 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : ∀ x, f 2 + (deriv f 2) * (x - 2) = -x + 6) : 
    f 2 + deriv f 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l2885_288519


namespace NUMINAMATH_CALUDE_perpendicular_implies_parallel_skew_perpendicular_parallel_implies_perpendicular_l2885_288510

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Define the property of skew lines
variable (skew : Line → Line → Prop)

-- Theorem 1: If a line is perpendicular to two planes, then those planes are parallel
theorem perpendicular_implies_parallel
  (m : Line) (α β : Plane)
  (h1 : perpendicular m α)
  (h2 : perpendicular m β) :
  plane_parallel α β :=
sorry

-- Theorem 2: If two skew lines are each perpendicular to one plane and parallel to the other, 
-- then the planes are perpendicular
theorem skew_perpendicular_parallel_implies_perpendicular
  (m n : Line) (α β : Plane)
  (h1 : skew m n)
  (h2 : perpendicular m α)
  (h3 : parallel m β)
  (h4 : perpendicular n β)
  (h5 : parallel n α) :
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_implies_parallel_skew_perpendicular_parallel_implies_perpendicular_l2885_288510


namespace NUMINAMATH_CALUDE_square_difference_l2885_288545

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2885_288545


namespace NUMINAMATH_CALUDE_largest_negative_root_l2885_288543

noncomputable def α : ℝ := Real.arctan (4 / 13)
noncomputable def β : ℝ := Real.arctan (8 / 11)

def equation (x : ℝ) : Prop :=
  4 * Real.sin (3 * x) + 13 * Real.cos (3 * x) = 8 * Real.sin x + 11 * Real.cos x

theorem largest_negative_root :
  ∃ (x : ℝ), x < 0 ∧ equation x ∧ 
  ∀ (y : ℝ), y < 0 → equation y → y ≤ x ∧
  x = (α - β) / 2 :=
sorry

end NUMINAMATH_CALUDE_largest_negative_root_l2885_288543


namespace NUMINAMATH_CALUDE_july_birth_percentage_l2885_288552

def total_people : ℕ := 100
def born_in_july : ℕ := 13

theorem july_birth_percentage :
  (born_in_july : ℚ) / total_people * 100 = 13 := by
  sorry

end NUMINAMATH_CALUDE_july_birth_percentage_l2885_288552


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2885_288516

theorem quadratic_factorization (x : ℝ) : 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2885_288516


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2885_288584

/-- An arithmetic sequence with sum S_n of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) (m : ℕ) 
  (h_m : m > 1)
  (h_condition : seq.a (m - 1) + seq.a (m + 1) - (seq.a m)^2 = 0)
  (h_sum : seq.S (2 * m - 1) = 38) :
  m = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2885_288584


namespace NUMINAMATH_CALUDE_find_divisor_l2885_288597

theorem find_divisor (N : ℝ) (D : ℝ) (h1 : (N - 6) / D = 2) (h2 : N = 32) : D = 13 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l2885_288597


namespace NUMINAMATH_CALUDE_intersection_point_solution_l2885_288593

/-- Given two lines y = x + 1 and y = mx + n that intersect at point (1,b),
    prove that the solution to the system of equations { x + 1 = y, y - mx = n }
    is x = 1 and y = 2. -/
theorem intersection_point_solution (m n b : ℝ) :
  (∃ x y : ℝ, x + 1 = y ∧ y - m*x = n) →
  (1 + 1 = b) →
  (∀ x y : ℝ, x + 1 = y ∧ y - m*x = n → x = 1 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_solution_l2885_288593


namespace NUMINAMATH_CALUDE_total_blocks_is_55_l2885_288514

/-- Calculates the total number of blocks in Thomas's stacks --/
def total_blocks : ℕ :=
  let first_stack := 7
  let second_stack := first_stack + 3
  let third_stack := second_stack - 6
  let fourth_stack := third_stack + 10
  let fifth_stack := 2 * second_stack
  first_stack + second_stack + third_stack + fourth_stack + fifth_stack

/-- Theorem stating that the total number of blocks is 55 --/
theorem total_blocks_is_55 : total_blocks = 55 := by
  sorry

end NUMINAMATH_CALUDE_total_blocks_is_55_l2885_288514


namespace NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l2885_288504

-- Define sets M and N
def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

-- Theorem stating that "a ∈ M" is a necessary but not sufficient condition for "a ∈ N"
theorem a_in_M_necessary_not_sufficient_for_a_in_N :
  (∀ a, a ∈ N → a ∈ M) ∧ (∃ a, a ∈ M ∧ a ∉ N) := by
  sorry


end NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l2885_288504


namespace NUMINAMATH_CALUDE_regular_polygon_30_degree_exterior_angle_l2885_288535

/-- A regular polygon with exterior angles measuring 30° has 12 sides -/
theorem regular_polygon_30_degree_exterior_angle (n : ℕ) :
  (n > 0) →
  (360 / n = 30) →
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_30_degree_exterior_angle_l2885_288535


namespace NUMINAMATH_CALUDE_problem_solution_l2885_288561

theorem problem_solution : (-1/2)⁻¹ - 4 * Real.cos (30 * π / 180) - (π + 2013)^0 + Real.sqrt 12 = -3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2885_288561


namespace NUMINAMATH_CALUDE_min_ratio_folded_to_total_area_ratio_two_thirds_achievable_min_ratio_is_two_thirds_l2885_288539

/-- Represents a point on the square tablecloth -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the square tablecloth with dark spots -/
structure Tablecloth where
  side_length : ℝ
  spots : Set Point
  total_area : ℝ
  folded_area : ℝ

/-- The ratio of folded visible area to total area is at least 2/3 -/
theorem min_ratio_folded_to_total_area (t : Tablecloth) : 
  t.folded_area / t.total_area ≥ 2/3 := by
  sorry

/-- The ratio of 2/3 is achievable -/
theorem ratio_two_thirds_achievable : 
  ∃ t : Tablecloth, t.folded_area / t.total_area = 2/3 := by
  sorry

/-- The minimum ratio of folded visible area to total area is exactly 2/3 -/
theorem min_ratio_is_two_thirds : 
  (∀ t : Tablecloth, t.folded_area / t.total_area ≥ 2/3) ∧
  (∃ t : Tablecloth, t.folded_area / t.total_area = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_min_ratio_folded_to_total_area_ratio_two_thirds_achievable_min_ratio_is_two_thirds_l2885_288539


namespace NUMINAMATH_CALUDE_phone_plan_cost_difference_l2885_288582

/-- Calculates the cost difference between Darnell's current phone plan and an alternative plan -/
theorem phone_plan_cost_difference :
  let current_plan_cost : ℚ := 12
  let texts_per_month : ℕ := 60
  let call_minutes_per_month : ℕ := 60
  let alt_plan_text_cost : ℚ := 1
  let alt_plan_text_limit : ℕ := 30
  let alt_plan_call_cost : ℚ := 3
  let alt_plan_call_limit : ℕ := 20
  let alt_plan_text_total : ℚ := (texts_per_month : ℚ) / alt_plan_text_limit * alt_plan_text_cost
  let alt_plan_call_total : ℚ := (call_minutes_per_month : ℚ) / alt_plan_call_limit * alt_plan_call_cost
  let alt_plan_total_cost : ℚ := alt_plan_text_total + alt_plan_call_total
  current_plan_cost - alt_plan_total_cost = 1 :=
by sorry

end NUMINAMATH_CALUDE_phone_plan_cost_difference_l2885_288582


namespace NUMINAMATH_CALUDE_sum_of_nth_row_sum_of_100th_row_l2885_288505

/-- The sum of numbers in the nth row of the triangular array -/
def f (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else 2 * f (n - 1) + 2 * n

/-- Theorem: The closed form of the sum of numbers in the nth row -/
theorem sum_of_nth_row (n : ℕ) : f n = 3 * 2^(n-1) - 2 * n := by
  sorry

/-- Corollary: The sum of numbers in the 100th row -/
theorem sum_of_100th_row : f 100 = 3 * 2^99 - 200 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_nth_row_sum_of_100th_row_l2885_288505


namespace NUMINAMATH_CALUDE_prime_condition_theorem_l2885_288532

def satisfies_condition (p : ℕ) : Prop :=
  Nat.Prime p ∧
  ∀ q : ℕ, Nat.Prime q → q < p →
    ∀ k r : ℕ, p = k * q + r → 0 ≤ r → r < q →
      ∀ a : ℕ, a > 1 → ¬(a^2 ∣ r)

theorem prime_condition_theorem :
  {p : ℕ | satisfies_condition p} = {2, 3, 5, 7, 13} :=
sorry

end NUMINAMATH_CALUDE_prime_condition_theorem_l2885_288532


namespace NUMINAMATH_CALUDE_nth_root_sum_theorem_l2885_288579

theorem nth_root_sum_theorem (a : ℝ) (n : ℕ) (hn : n > 1) :
  let f : ℝ → ℝ := λ x => (x^n - a^n)^(1/n) + (2*a^n - x^n)^(1/n)
  ∀ x, f x = a ↔ 
    (a ≠ 0 ∧ 
      ((n % 2 = 1 ∧ (x = a * (2^(1/n)) ∨ x = a)) ∨ 
       (n % 2 = 0 ∧ a > 0 ∧ (x = a * (2^(1/n)) ∨ x = -a * (2^(1/n)) ∨ x = a ∨ x = -a)))) ∨
    (a = 0 ∧ 
      ((n % 2 = 1 ∧ true) ∨ 
       (n % 2 = 0 ∧ x = 0))) :=
by sorry


end NUMINAMATH_CALUDE_nth_root_sum_theorem_l2885_288579


namespace NUMINAMATH_CALUDE_email_count_theorem_l2885_288560

/-- Calculates the total number of emails received in a month with changing email rates --/
def total_emails (days : ℕ) (initial_rate : ℕ) (increase : ℕ) : ℕ :=
  let half_days := days / 2
  let first_half := initial_rate * half_days
  let second_half := (initial_rate + increase) * (days - half_days)
  first_half + second_half

/-- Theorem stating that given the conditions, the total emails received is 675 --/
theorem email_count_theorem :
  total_emails 30 20 5 = 675 := by
  sorry

end NUMINAMATH_CALUDE_email_count_theorem_l2885_288560


namespace NUMINAMATH_CALUDE_dhoni_rent_percentage_l2885_288529

theorem dhoni_rent_percentage (rent_percentage : ℝ) 
  (h1 : rent_percentage > 0)
  (h2 : rent_percentage < 100)
  (h3 : rent_percentage + (rent_percentage - 10) + 52.5 = 100) :
  rent_percentage = 28.75 := by
sorry

end NUMINAMATH_CALUDE_dhoni_rent_percentage_l2885_288529


namespace NUMINAMATH_CALUDE_condition_for_equation_l2885_288528

theorem condition_for_equation (x y z : ℤ) : x = y ∧ y = z → x * (x - y) + y * (y - z) + z * (z - x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_condition_for_equation_l2885_288528


namespace NUMINAMATH_CALUDE_investment_interest_rate_l2885_288559

/-- Proves that given the specified conditions, the interest rate for the second part of an investment is 5% -/
theorem investment_interest_rate 
  (total_investment : ℕ)
  (first_part : ℕ)
  (first_rate : ℚ)
  (total_interest : ℕ)
  (h1 : total_investment = 3400)
  (h2 : first_part = 1300)
  (h3 : first_rate = 3 / 100)
  (h4 : total_interest = 144) :
  let second_part := total_investment - first_part
  let first_interest := (first_part : ℚ) * first_rate
  let second_interest := (total_interest : ℚ) - first_interest
  let second_rate := second_interest / (second_part : ℚ)
  second_rate = 5 / 100 := by
sorry

end NUMINAMATH_CALUDE_investment_interest_rate_l2885_288559


namespace NUMINAMATH_CALUDE_minimum_marbles_to_add_proof_minimum_marbles_l2885_288536

theorem minimum_marbles_to_add (initial_marbles : Nat) (people : Nat) : Nat :=
  let additional_marbles := people - initial_marbles % people
  if additional_marbles = people then 0 else additional_marbles

theorem proof_minimum_marbles :
  minimum_marbles_to_add 62 8 = 2 ∧
  (62 + minimum_marbles_to_add 62 8) % 8 = 0 ∧
  ∀ x : Nat, x < minimum_marbles_to_add 62 8 → (62 + x) % 8 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_minimum_marbles_to_add_proof_minimum_marbles_l2885_288536


namespace NUMINAMATH_CALUDE_isosceles_triangle_sides_isosceles_triangle_4_exists_l2885_288523

/-- An isosceles triangle with perimeter 18 and legs twice the base length -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  perimeter_eq : base + 2 * leg = 18
  leg_eq : leg = 2 * base

/-- An isosceles triangle with one side 4 -/
structure IsoscelesTriangle4 where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  perimeter_eq : side1 + side2 + side3 = 18
  isosceles_eq : side2 = side3
  one_side_4 : side1 = 4 ∨ side2 = 4

theorem isosceles_triangle_sides (t : IsoscelesTriangle) :
  t.base = 18 / 5 ∧ t.leg = 36 / 5 := by sorry

theorem isosceles_triangle_4_exists :
  ∃ (t : IsoscelesTriangle4), t.side2 = 7 ∧ t.side3 = 7 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_sides_isosceles_triangle_4_exists_l2885_288523


namespace NUMINAMATH_CALUDE_tree_height_after_two_years_l2885_288534

def tree_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (3 ^ years)

theorem tree_height_after_two_years 
  (h : tree_height (tree_height h0 2) 2 = 81) : tree_height h0 2 = 9 :=
by
  sorry

#check tree_height_after_two_years

end NUMINAMATH_CALUDE_tree_height_after_two_years_l2885_288534


namespace NUMINAMATH_CALUDE_race_symmetry_l2885_288500

/-- Represents a car in the race -/
structure Car where
  speed : ℝ
  direction : Bool -- true for clockwise, false for counterclockwise

/-- Represents the race scenario -/
structure RaceScenario where
  A : Car
  B : Car
  C : Car
  D : Car
  track_length : ℝ
  first_AC_meet_time : ℝ
  first_BD_meet_time : ℝ
  first_AB_meet_time : ℝ

/-- The main theorem statement -/
theorem race_symmetry (race : RaceScenario) :
  race.A.direction = true ∧
  race.B.direction = true ∧
  race.C.direction = false ∧
  race.D.direction = false ∧
  race.A.speed ≠ race.B.speed ∧
  race.A.speed ≠ race.C.speed ∧
  race.A.speed ≠ race.D.speed ∧
  race.B.speed ≠ race.C.speed ∧
  race.B.speed ≠ race.D.speed ∧
  race.C.speed ≠ race.D.speed ∧
  race.first_AC_meet_time = 7 ∧
  race.first_BD_meet_time = 7 ∧
  race.first_AB_meet_time = 53 →
  ∃ (first_CD_meet_time : ℝ), first_CD_meet_time = race.first_AB_meet_time :=
by
  sorry

end NUMINAMATH_CALUDE_race_symmetry_l2885_288500


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_is_one_max_value_of_m_max_value_of_m_achievable_l2885_288511

-- Define the function f
def f (x a b : ℝ) : ℝ := |x - a| + 2 * |x + b|

-- Theorem 1
theorem sum_of_a_and_b_is_one 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hmin : ∀ x, f x a b ≥ 1) 
  (hmin_exists : ∃ x, f x a b = 1) : 
  a + b = 1 := 
sorry

-- Theorem 2
theorem max_value_of_m 
  (a b m : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a + b = 1) 
  (hm : m ≤ 1/a + 2/b) : 
  m ≤ 3 + 2 * Real.sqrt 2 := 
sorry

-- The maximum value is achievable
theorem max_value_of_m_achievable :
  ∃ m, m = 3 + 2 * Real.sqrt 2 ∧ 
  ∃ a b, a > 0 ∧ b > 0 ∧ a + b = 1 ∧ m ≤ 1/a + 2/b :=
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_is_one_max_value_of_m_max_value_of_m_achievable_l2885_288511


namespace NUMINAMATH_CALUDE_gcd_problem_l2885_288574

theorem gcd_problem : Nat.gcd 7260 540 - 12 + 5 = 53 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l2885_288574


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2885_288533

theorem solution_set_inequality (x : ℝ) : 
  x * (x + 3) ≥ 0 ↔ x ≥ 0 ∨ x ≤ -3 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2885_288533


namespace NUMINAMATH_CALUDE_andrew_fruit_purchase_l2885_288586

/-- The total cost of a fruit purchase, including tax -/
def totalCost (grapeQuantity mangoQuantity grapePrice mangoPrice grapeTaxRate mangoTaxRate : ℚ) : ℚ :=
  let grapeCost := grapeQuantity * grapePrice
  let mangoCost := mangoQuantity * mangoPrice
  let grapeTax := grapeCost * grapeTaxRate
  let mangoTax := mangoCost * mangoTaxRate
  grapeCost + mangoCost + grapeTax + mangoTax

/-- The theorem stating the total cost of Andrew's fruit purchase -/
theorem andrew_fruit_purchase :
  totalCost 8 9 70 55 (8/100) (11/100) = 1154.25 := by
  sorry

end NUMINAMATH_CALUDE_andrew_fruit_purchase_l2885_288586


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2885_288542

def A : Set ℕ := {x | (x + 1) * (x - 2) = 0}
def B : Set ℕ := {2, 4, 5}

theorem union_of_A_and_B : A ∪ B = {2, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2885_288542


namespace NUMINAMATH_CALUDE_hexagon_sixth_angle_l2885_288583

/-- The sum of interior angles of a hexagon is 720 degrees -/
def hexagon_angle_sum : ℝ := 720

/-- The given angles of the hexagon -/
def given_angles : List ℝ := [135, 105, 87, 120, 78]

/-- Theorem: In a hexagon where five of the interior angles measure 135°, 105°, 87°, 120°, and 78°, the sixth angle measures 195°. -/
theorem hexagon_sixth_angle : 
  List.sum given_angles + 195 = hexagon_angle_sum := by
  sorry

end NUMINAMATH_CALUDE_hexagon_sixth_angle_l2885_288583


namespace NUMINAMATH_CALUDE_angle_halving_l2885_288592

-- Define what it means for an angle to be in the fourth quadrant
def in_fourth_quadrant (θ : Real) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi - Real.pi / 2 < θ ∧ θ < 2 * k * Real.pi

-- Define what it means for an angle to be in the first or third quadrant
def in_first_or_third_quadrant (θ : Real) : Prop :=
  ∃ k : ℤ, k * Real.pi < θ ∧ θ < k * Real.pi + Real.pi / 2

theorem angle_halving (θ : Real) :
  in_fourth_quadrant θ → in_first_or_third_quadrant (-θ/2) :=
by sorry

end NUMINAMATH_CALUDE_angle_halving_l2885_288592


namespace NUMINAMATH_CALUDE_root_sum_squares_l2885_288517

theorem root_sum_squares (p q r : ℝ) : 
  (p^3 - 24*p^2 + 50*p - 8 = 0) →
  (q^3 - 24*q^2 + 50*q - 8 = 0) →
  (r^3 - 24*r^2 + 50*r - 8 = 0) →
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 1052 := by
sorry

end NUMINAMATH_CALUDE_root_sum_squares_l2885_288517


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2885_288544

theorem perfect_square_condition (k : ℤ) : 
  (∀ x : ℤ, ∃ y : ℤ, x^2 - 2*(k+1)*x + 4 = y^2) → (k = -3 ∨ k = 1) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2885_288544


namespace NUMINAMATH_CALUDE_friendship_theorem_l2885_288580

-- Define a type for people
def Person : Type := Nat

-- Define the friendship relation
def IsFriend (p q : Person) : Prop := sorry

-- State the theorem
theorem friendship_theorem :
  ∀ (group : Finset Person),
  (Finset.card group = 12) →
  ∃ (A B : Person),
    A ∈ group ∧ B ∈ group ∧ A ≠ B ∧
    ∃ (C D E F G : Person),
      C ∈ group ∧ D ∈ group ∧ E ∈ group ∧ F ∈ group ∧ G ∈ group ∧
      C ≠ A ∧ C ≠ B ∧ D ≠ A ∧ D ≠ B ∧ E ≠ A ∧ E ≠ B ∧ F ≠ A ∧ F ≠ B ∧ G ≠ A ∧ G ≠ B ∧
      C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ E ≠ F ∧ E ≠ G ∧ F ≠ G ∧
      ((IsFriend C A ∧ IsFriend C B) ∨ (¬IsFriend C A ∧ ¬IsFriend C B)) ∧
      ((IsFriend D A ∧ IsFriend D B) ∨ (¬IsFriend D A ∧ ¬IsFriend D B)) ∧
      ((IsFriend E A ∧ IsFriend E B) ∨ (¬IsFriend E A ∧ ¬IsFriend E B)) ∧
      ((IsFriend F A ∧ IsFriend F B) ∨ (¬IsFriend F A ∧ ¬IsFriend F B)) ∧
      ((IsFriend G A ∧ IsFriend G B) ∨ (¬IsFriend G A ∧ ¬IsFriend G B)) :=
by
  sorry


end NUMINAMATH_CALUDE_friendship_theorem_l2885_288580


namespace NUMINAMATH_CALUDE_probability_of_science_second_draw_l2885_288506

/-- Represents the type of questions --/
inductive QuestionType
| Science
| LiberalArts

/-- Represents the state of the questions after the first draw --/
structure QuestionState :=
  (total : Nat)
  (science : Nat)
  (liberal_arts : Nat)

/-- The initial state of questions --/
def initial_state : QuestionState :=
  ⟨5, 3, 2⟩

/-- The state after drawing a science question --/
def after_first_draw (s : QuestionState) : QuestionState :=
  ⟨s.total - 1, s.science - 1, s.liberal_arts⟩

/-- The probability of drawing a science question on the second draw --/
def prob_science_second_draw (s : QuestionState) : Rat :=
  s.science / s.total

theorem probability_of_science_second_draw :
  prob_science_second_draw (after_first_draw initial_state) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_science_second_draw_l2885_288506


namespace NUMINAMATH_CALUDE_minimal_fraction_sum_l2885_288590

theorem minimal_fraction_sum (a b : ℕ+) (h : (9:ℚ)/22 < (a:ℚ)/b ∧ (a:ℚ)/b < 5/11) :
  (∃ (c d : ℕ+), (9:ℚ)/22 < (c:ℚ)/d ∧ (c:ℚ)/d < 5/11 ∧ c.val + d.val < a.val + b.val) ∨ (a = 3 ∧ b = 7) :=
sorry

end NUMINAMATH_CALUDE_minimal_fraction_sum_l2885_288590


namespace NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l2885_288575

/-- The coefficient of x^2 in the expansion of (1/x - √x)^10 is 45 -/
theorem coefficient_x_squared_expansion (x : ℝ) : 
  (Finset.range 11).sum (fun k => (-1)^k * (Nat.choose 10 k : ℝ) * x^((3*k:ℤ)/2 - 5)) = 45 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l2885_288575


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2885_288521

/-- Given vectors a and b in ℝ², if a + k * b is perpendicular to a - b, then k = 11/20 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (2, -1)) 
  (h2 : b = (-1, 4)) 
  (h3 : (a.1 + k * b.1, a.2 + k * b.2) • (a.1 - b.1, a.2 - b.2) = 0) : 
  k = 11/20 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2885_288521


namespace NUMINAMATH_CALUDE_ab_over_c_equals_two_l2885_288537

theorem ab_over_c_equals_two 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h_eq1 : a * b - c = 3) 
  (h_eq2 : a * b * c = 18) : 
  a * b / c = 2 := by
sorry

end NUMINAMATH_CALUDE_ab_over_c_equals_two_l2885_288537


namespace NUMINAMATH_CALUDE_prob_two_heads_with_second_tail_l2885_288587

/-- A fair coin flip sequence that ends with either two heads or two tails in a row -/
inductive CoinFlipSequence
| TH : CoinFlipSequence → CoinFlipSequence
| TT : CoinFlipSequence
| HH : CoinFlipSequence

/-- The probability of a specific coin flip sequence -/
def probability (seq : CoinFlipSequence) : ℚ :=
  match seq with
  | CoinFlipSequence.TH s => (1/2) * probability s
  | CoinFlipSequence.TT => (1/2) * (1/2)
  | CoinFlipSequence.HH => (1/2) * (1/2)

/-- The probability of getting two heads in a row while seeing a second tail before seeing a second head -/
def probTwoHeadsWithSecondTail : ℚ :=
  (1/2) * (1/2) * (1/2) * (1/3)

theorem prob_two_heads_with_second_tail :
  probTwoHeadsWithSecondTail = 1/24 :=
sorry

end NUMINAMATH_CALUDE_prob_two_heads_with_second_tail_l2885_288587


namespace NUMINAMATH_CALUDE_shirt_price_calculation_l2885_288557

theorem shirt_price_calculation (total_cost sweater_price shirt_price : ℝ) :
  total_cost = 80.34 →
  sweater_price - shirt_price = 7.43 →
  total_cost = sweater_price + shirt_price →
  shirt_price = 36.455 := by
sorry

end NUMINAMATH_CALUDE_shirt_price_calculation_l2885_288557


namespace NUMINAMATH_CALUDE_distribute_fraction_over_parentheses_l2885_288508

theorem distribute_fraction_over_parentheses (x : ℝ) : (1 / 3) * (6 * x - 3) = 2 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_distribute_fraction_over_parentheses_l2885_288508


namespace NUMINAMATH_CALUDE_blue_tetrahedron_volume_l2885_288589

/-- The volume of a tetrahedron formed by alternately colored vertices of a cube -/
theorem blue_tetrahedron_volume (s : ℝ) (h : s = 8) :
  let cube_volume := s^3
  let small_tetrahedron_volume := (1/6) * s^3
  cube_volume - 4 * small_tetrahedron_volume = (512:ℝ)/3 :=
by
  sorry

end NUMINAMATH_CALUDE_blue_tetrahedron_volume_l2885_288589


namespace NUMINAMATH_CALUDE_actual_time_greater_than_planned_l2885_288576

/-- Represents the running competition scenario -/
structure RunningCompetition where
  V : ℝ  -- Planned constant speed
  D : ℝ  -- Total distance
  V1 : ℝ := 1.25 * V  -- Increased speed for first half
  V2 : ℝ := 0.80 * V  -- Decreased speed for second half

/-- Theorem stating that the actual time is greater than the planned time -/
theorem actual_time_greater_than_planned (rc : RunningCompetition) 
  (h_positive_speed : rc.V > 0) (h_positive_distance : rc.D > 0) : 
  (rc.D / (2 * rc.V1) + rc.D / (2 * rc.V2)) > (rc.D / rc.V) :=
by sorry

end NUMINAMATH_CALUDE_actual_time_greater_than_planned_l2885_288576


namespace NUMINAMATH_CALUDE_area_between_circles_l2885_288565

/-- The area of the region between two concentric circles with given radii and a tangent chord --/
theorem area_between_circles (R r c : ℝ) (hR : R = 60) (hr : r = 40) (hc : c = 100)
  (h_concentric : R > r) (h_tangent : c^2 = 4 * (R^2 - r^2)) :
  π * (R^2 - r^2) = 2000 * π := by
sorry

end NUMINAMATH_CALUDE_area_between_circles_l2885_288565


namespace NUMINAMATH_CALUDE_locus_of_vertex_A_l2885_288568

def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  B = (-6, 0) ∧ C = (6, 0)

def angle_condition (A B C : ℝ) : Prop :=
  Real.sin B - Real.sin C = (1/2) * Real.sin A

def locus_equation (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 27 = 1 ∧ x < -3

theorem locus_of_vertex_A (A B C : ℝ × ℝ) (angleA angleB angleC : ℝ) :
  triangle_ABC A B C →
  angle_condition angleA angleB angleC →
  locus_equation A.1 A.2 :=
sorry

end NUMINAMATH_CALUDE_locus_of_vertex_A_l2885_288568


namespace NUMINAMATH_CALUDE_min_value_of_function_l2885_288502

theorem min_value_of_function (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + y = 2) :
  (2 / x + 1 / y) ≥ 3 / 2 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2885_288502


namespace NUMINAMATH_CALUDE_solution_set_f_geq_1_range_of_a_l2885_288558

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 1| - 2

-- Theorem for the solution set of f(x) ≥ 1
theorem solution_set_f_geq_1 :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} := by sorry

-- Theorem for the range of a where f(x) ≥ a^2 - a - 2 for all x in ℝ
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x ≥ a^2 - a - 2) ↔ -1 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_1_range_of_a_l2885_288558


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2885_288567

theorem completing_square_equivalence (x : ℝ) :
  (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2885_288567


namespace NUMINAMATH_CALUDE_line_slope_proof_l2885_288556

theorem line_slope_proof (x y : ℝ) : 
  (((Real.sqrt 3) / 3) * x + y - 7 = 0) → 
  (∃ m : ℝ, m = -(Real.sqrt 3) / 3 ∧ y = m * x + 7) := by
  sorry

end NUMINAMATH_CALUDE_line_slope_proof_l2885_288556


namespace NUMINAMATH_CALUDE_gcd_1554_2405_l2885_288550

theorem gcd_1554_2405 : Nat.gcd 1554 2405 = 37 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1554_2405_l2885_288550


namespace NUMINAMATH_CALUDE_letters_with_dot_only_in_given_alphabet_l2885_288541

/-- Represents an alphabet with letters containing dots and/or straight lines -/
structure Alphabet where
  total : ℕ
  both : ℕ
  straight_only : ℕ
  dot_only : ℕ
  all_contain : both + straight_only + dot_only = total

/-- The number of letters containing only a dot in a specific alphabet -/
def letters_with_dot_only (a : Alphabet) : ℕ := a.dot_only

/-- Theorem stating the number of letters with only a dot in the given alphabet -/
theorem letters_with_dot_only_in_given_alphabet :
  ∃ (a : Alphabet), a.total = 60 ∧ a.both = 20 ∧ a.straight_only = 36 ∧ letters_with_dot_only a = 4 := by
  sorry

end NUMINAMATH_CALUDE_letters_with_dot_only_in_given_alphabet_l2885_288541


namespace NUMINAMATH_CALUDE_semicircle_inscriptions_l2885_288553

theorem semicircle_inscriptions (D : ℝ) (N : ℕ) (h : N > 0) : 
  let r := D / (2 * N)
  let R := N * r
  let A := N * (π * r^2 / 2)
  let B := π * R^2 / 2 - A
  A / B = 2 / 25 → N = 14 := by
sorry

end NUMINAMATH_CALUDE_semicircle_inscriptions_l2885_288553


namespace NUMINAMATH_CALUDE_count_D_eq_3_is_33_l2885_288538

/-- D(n) is the number of pairs of different adjacent digits in the binary representation of n -/
def D (n : ℕ) : ℕ := sorry

/-- Count of positive integers n ≤ 200 for which D(n) = 3 -/
def count_D_eq_3 : ℕ := sorry

theorem count_D_eq_3_is_33 : count_D_eq_3 = 33 := by sorry

end NUMINAMATH_CALUDE_count_D_eq_3_is_33_l2885_288538


namespace NUMINAMATH_CALUDE_project_completion_days_l2885_288530

/-- Represents the work rates and schedule for a project completed by three persons. -/
structure ProjectSchedule where
  rate_A : ℚ  -- Work rate of person A (fraction of work completed per day)
  rate_B : ℚ  -- Work rate of person B
  rate_C : ℚ  -- Work rate of person C
  days_A : ℕ  -- Number of days A works alone
  days_BC : ℕ  -- Number of days B and C work together

/-- Calculates the total number of days needed to complete the project. -/
def totalDays (p : ProjectSchedule) : ℚ :=
  let work_A := p.rate_A * p.days_A
  let rate_BC := p.rate_B + p.rate_C
  let work_BC := rate_BC * p.days_BC
  let remaining_work := 1 - (work_A + work_BC)
  p.days_A + p.days_BC + remaining_work / p.rate_C

/-- Theorem stating that for the given project schedule, the total days needed is 9. -/
theorem project_completion_days :
  let p := ProjectSchedule.mk (1/10) (1/12) (1/15) 2 4
  totalDays p = 9 := by sorry

end NUMINAMATH_CALUDE_project_completion_days_l2885_288530


namespace NUMINAMATH_CALUDE_exists_divisible_figure_l2885_288585

/-- Represents a geometric shape --/
structure Shape :=
  (area : ℝ)

/-- Represents a T-shaped piece --/
def T_shape : Shape :=
  { area := 3 }

/-- Represents the set of five specific pieces --/
def five_pieces : Finset Shape :=
  sorry

/-- A figure that can be divided into different sets of pieces --/
structure DivisibleFigure :=
  (total_area : ℝ)
  (can_divide_into_four_T : Prop)
  (can_divide_into_five_pieces : Prop)

/-- The existence of a figure that satisfies both division conditions --/
theorem exists_divisible_figure : 
  ∃ (fig : DivisibleFigure), 
    fig.can_divide_into_four_T ∧ 
    fig.can_divide_into_five_pieces :=
sorry

end NUMINAMATH_CALUDE_exists_divisible_figure_l2885_288585


namespace NUMINAMATH_CALUDE_ball_max_height_l2885_288563

theorem ball_max_height :
  let f : ℝ → ℝ := fun t ↦ -5 * t^2 + 20 * t + 10
  ∃ (max : ℝ), max = 30 ∧ ∀ t, f t ≤ max :=
by
  sorry

end NUMINAMATH_CALUDE_ball_max_height_l2885_288563


namespace NUMINAMATH_CALUDE_expression_value_l2885_288569

theorem expression_value :
  let x : ℕ := 3
  5^3 - 2^x * 3 + 4^2 = 117 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2885_288569


namespace NUMINAMATH_CALUDE_circle_plus_two_four_l2885_288531

-- Define the operation ⊕
def circle_plus (a b : ℝ) : ℝ := 5 * a + 2 * b

-- Theorem statement
theorem circle_plus_two_four : circle_plus 2 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_circle_plus_two_four_l2885_288531


namespace NUMINAMATH_CALUDE_electric_bus_pricing_and_optimal_plan_l2885_288594

/-- Represents the unit price of a type A electric bus in million yuan -/
def type_a_price : ℝ := 36

/-- Represents the unit price of a type B electric bus in million yuan -/
def type_b_price : ℝ := 40

/-- Represents the number of type A buses in the optimal plan -/
def optimal_type_a : ℕ := 20

/-- Represents the number of type B buses in the optimal plan -/
def optimal_type_b : ℕ := 10

/-- Represents the total cost of the optimal plan in million yuan -/
def optimal_total_cost : ℝ := 1120

theorem electric_bus_pricing_and_optimal_plan :
  (type_b_price = type_a_price + 4) ∧
  (720 / type_a_price = 800 / type_b_price) ∧
  (optimal_type_a + optimal_type_b = 30) ∧
  (optimal_type_a ≥ 10) ∧
  (optimal_type_a ≤ 2 * optimal_type_b) ∧
  (∀ m n : ℕ, m + n = 30 → m ≥ 10 → m ≤ 2 * n →
    type_a_price * m + type_b_price * n ≥ optimal_total_cost) ∧
  (optimal_total_cost = type_a_price * optimal_type_a + type_b_price * optimal_type_b) :=
by sorry


end NUMINAMATH_CALUDE_electric_bus_pricing_and_optimal_plan_l2885_288594


namespace NUMINAMATH_CALUDE_divisibility_condition_l2885_288572

theorem divisibility_condition (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  n ∣ (1 + m^(3^n) + m^(2*3^n)) ↔ ∃ t : ℕ+, n = 3 ∧ m = 3 * t - 2 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2885_288572


namespace NUMINAMATH_CALUDE_function_equality_up_to_constant_l2885_288566

theorem function_equality_up_to_constant 
  (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) 
  (hg : Differentiable ℝ g) 
  (h : ∀ x, deriv f x = deriv g x) : 
  ∃ C, ∀ x, f x = g x + C :=
sorry

end NUMINAMATH_CALUDE_function_equality_up_to_constant_l2885_288566


namespace NUMINAMATH_CALUDE_inverse_difference_simplification_l2885_288522

theorem inverse_difference_simplification (x y : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : 3 * x - y / 3 ≠ 0) :
  (3 * x - y / 3)⁻¹ * ((3 * x)⁻¹ - (y / 3)⁻¹) = -(x * y)⁻¹ :=
by sorry

end NUMINAMATH_CALUDE_inverse_difference_simplification_l2885_288522


namespace NUMINAMATH_CALUDE_expression_equals_zero_l2885_288581

theorem expression_equals_zero (θ : Real) (h : Real.tan θ = 5) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 := by
sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l2885_288581


namespace NUMINAMATH_CALUDE_giants_playoff_wins_l2885_288509

theorem giants_playoff_wins (total_games : ℕ) (games_to_win : ℕ) (more_wins_needed : ℕ) : 
  total_games = 30 →
  games_to_win = (2 * total_games) / 3 →
  more_wins_needed = 8 →
  games_to_win - more_wins_needed = 12 :=
by sorry

end NUMINAMATH_CALUDE_giants_playoff_wins_l2885_288509


namespace NUMINAMATH_CALUDE_cricket_bat_cost_price_l2885_288526

theorem cricket_bat_cost_price (profit_A_to_B : Real) (profit_B_to_C : Real) (price_C : Real) :
  profit_A_to_B = 0.20 →
  profit_B_to_C = 0.25 →
  price_C = 231 →
  ∃ (cost_price_A : Real), cost_price_A = 154 ∧
    price_C = cost_price_A * (1 + profit_A_to_B) * (1 + profit_B_to_C) := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_cost_price_l2885_288526


namespace NUMINAMATH_CALUDE_equal_roots_condition_l2885_288524

theorem equal_roots_condition (m : ℝ) : 
  (∀ x, x ≠ -3 ∧ m ≠ -1 ∧ m ≠ 0 → 
    (x * (x + 3) - (m - 3)) / ((x + 3) * (m + 1)) = x / m) →
  (∃! r, ∀ x, x ≠ -3 ∧ m ≠ -1 ∧ m ≠ 0 → 
    (x * (x + 3) - (m - 3)) / ((x + 3) * (m + 1)) = x / m → x = r) ↔ 
  m = 3/2 :=
sorry

end NUMINAMATH_CALUDE_equal_roots_condition_l2885_288524


namespace NUMINAMATH_CALUDE_remainder_3_2015_mod_13_l2885_288577

theorem remainder_3_2015_mod_13 : ∃ k : ℤ, 3^2015 = 13 * k + 9 :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_3_2015_mod_13_l2885_288577


namespace NUMINAMATH_CALUDE_distance_between_points_l2885_288540

def point1 : ℝ × ℝ := (2, -2)
def point2 : ℝ × ℝ := (8, 9)

theorem distance_between_points : 
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = Real.sqrt 157 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2885_288540


namespace NUMINAMATH_CALUDE_calculate_expression_l2885_288513

theorem calculate_expression : -Real.sqrt 4 + |Real.sqrt 2 - 2| - 202 * 3^0 = -Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2885_288513


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2885_288573

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, (a - b) * a^2 < 0 → a < b) ∧
  (∃ a b : ℝ, a < b ∧ ¬((a - b) * a^2 < 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2885_288573


namespace NUMINAMATH_CALUDE_triangle_area_l2885_288599

theorem triangle_area (a b c : ℝ) (h1 : a = 14) (h2 : b = 48) (h3 : c = 50) :
  (1/2) * a * b = 336 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2885_288599


namespace NUMINAMATH_CALUDE_leo_current_weight_l2885_288571

/-- Leo's current weight in pounds -/
def leo_weight : ℝ := 80

/-- Kendra's current weight in pounds -/
def kendra_weight : ℝ := 140 - leo_weight

/-- The combined weight of Leo and Kendra in pounds -/
def combined_weight : ℝ := 140

theorem leo_current_weight :
  (leo_weight + 10 = 1.5 * kendra_weight) ∧
  (leo_weight + kendra_weight = combined_weight) →
  leo_weight = 80 := by
sorry

end NUMINAMATH_CALUDE_leo_current_weight_l2885_288571


namespace NUMINAMATH_CALUDE_decimal_multiplication_l2885_288588

theorem decimal_multiplication : (0.25 : ℝ) * 0.75 * 0.1 = 0.01875 := by sorry

end NUMINAMATH_CALUDE_decimal_multiplication_l2885_288588


namespace NUMINAMATH_CALUDE_expression_evaluation_l2885_288551

theorem expression_evaluation : (-1)^3 + 4 * (-2) - 3 / (-3) = -8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2885_288551


namespace NUMINAMATH_CALUDE_donut_distribution_l2885_288564

/-- The number of ways to distribute n identical items among k distinct groups,
    where each group must receive at least one item -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

/-- The problem statement -/
theorem donut_distribution : distribute 4 5 = 70 := by
  sorry

end NUMINAMATH_CALUDE_donut_distribution_l2885_288564


namespace NUMINAMATH_CALUDE_sin_2theta_third_quadrant_l2885_288595

theorem sin_2theta_third_quadrant (θ : Real) :
  (π < θ ∧ θ < 3*π/2) →  -- θ is in the third quadrant
  (Real.sin θ)^4 + (Real.cos θ)^4 = 5/9 →
  Real.sin (2*θ) = -2*Real.sqrt 2/3 :=
by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_third_quadrant_l2885_288595
