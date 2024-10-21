import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vessel_width_is_15_l608_60840

/-- Represents the dimensions and properties of a cube immersed in a rectangular vessel --/
structure CubeImmersion where
  cube_edge : ℝ
  vessel_length : ℝ
  water_rise : ℝ

/-- Calculates the width of the vessel's base given a cube immersion scenario --/
noncomputable def vessel_base_width (ci : CubeImmersion) : ℝ :=
  (ci.cube_edge ^ 3) / (ci.vessel_length * ci.water_rise)

/-- Theorem stating that for the given dimensions, the vessel base width is 15 cm --/
theorem vessel_width_is_15 :
  let ci : CubeImmersion := ⟨15, 20, 11.25⟩
  vessel_base_width ci = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vessel_width_is_15_l608_60840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_best_fitting_model_l608_60828

def r_squared_values : List ℝ := [0.62, 0.63, 0.68, 0.65]

noncomputable def best_fit (values : List ℝ) : ℝ := 
  (List.maximum values).getD 0

theorem best_fitting_model (values : List ℝ) (h : values = r_squared_values) :
  best_fit values = 0.68 := by
  sorry

#check best_fitting_model

end NUMINAMATH_CALUDE_ERRORFEEDBACK_best_fitting_model_l608_60828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l608_60874

/-- If the slope angle of one of the asymptotes of the hyperbola x²/a - y²/9 = 1 is 60°, then a = 3 -/
theorem hyperbola_asymptote (a : ℝ) :
  (∃ θ : ℝ, θ = 60 * π / 180 ∧
    ∃ m : ℝ, m = Real.tan θ ∧
    m^2 = 3 / a) →
  a = 3 :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l608_60874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_geq_neg_three_l608_60827

/-- The function f(x) defined on the real numbers. -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x - 1/x

/-- Theorem: If f(x) is increasing on (1/2, +∞), then a ≥ -3 -/
theorem f_increasing_implies_a_geq_neg_three (a : ℝ) :
  (∀ x y : ℝ, 1/2 < x ∧ x < y → f a x < f a y) →
  a ≥ -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_geq_neg_three_l608_60827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l608_60843

theorem triangle_side_length (A : Real) (a b c : Real) :
  Real.cos A = 7/8 →
  c - a = 2 →
  b = 3 →
  a = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l608_60843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_and_hypotenuse_l608_60848

theorem right_triangle_area_and_hypotenuse (a b : ℝ) (h1 : a = 48) (h2 : b = 55) :
  (1 / 2) * a * b = 1320 ∧ Real.sqrt (a^2 + b^2) = 73 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_and_hypotenuse_l608_60848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_per_normal_class_correct_l608_60858

/-- Calculates the number of students in each normal class at the new school --/
def students_per_normal_class
  (total_students : ℕ)
  (moving_percentage : ℚ)
  (grade_levels : ℕ)
  (advanced_class_size : ℕ)
  (normal_classes_per_grade : ℕ) : ℕ :=
  let moving_students := (total_students : ℚ) * moving_percentage
  let students_per_grade := moving_students / grade_levels
  let normal_students_per_grade := students_per_grade - advanced_class_size
  (normal_students_per_grade / normal_classes_per_grade).floor.toNat

theorem students_per_normal_class_correct
  (total_students : ℕ)
  (moving_percentage : ℚ)
  (grade_levels : ℕ)
  (advanced_class_size : ℕ)
  (normal_classes_per_grade : ℕ)
  (h1 : total_students = 1590)
  (h2 : moving_percentage = 2/5)
  (h3 : grade_levels = 3)
  (h4 : advanced_class_size = 20)
  (h5 : normal_classes_per_grade = 6)
  : students_per_normal_class total_students moving_percentage grade_levels advanced_class_size normal_classes_per_grade = 32 :=
by
  sorry

#eval students_per_normal_class 1590 (2/5) 3 20 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_per_normal_class_correct_l608_60858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_board_game_theorem_l608_60892

def board_operation (a b c : ℤ) : ℤ × ℤ × ℤ :=
  (b, c, a + b - c)

def is_valid_board (x y z : ℤ) : Prop :=
  (y = x + 6 ∧ z = y + 6) ∨ (x = y + 6 ∧ z = x + 6) ∨ (x = z + 6 ∧ y = x + 6)

theorem board_game_theorem :
  ∀ (a b c : ℤ),
  is_valid_board 3 9 15 →
  (∀ (x y z : ℤ), is_valid_board x y z → 
    is_valid_board (board_operation x y z).1 (board_operation x y z).2.1 (board_operation x y z).2.2) →
  (∃ (n : ℕ) (x y z : ℤ), 
    ((x, y, z) = Nat.iterate (λ t => board_operation t.1 t.2.1 t.2.2) n (3, 9, 15)) ∧ 
    is_valid_board x y z ∧ 
    x = 2013 ∧ x < y ∧ y < z) →
  ∃ (y z : ℤ), y = 2019 ∧ z = 2025 := by
  sorry

#check board_game_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_board_game_theorem_l608_60892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l608_60872

/-- The focal length of a hyperbola -/
noncomputable def focal_length (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

/-- The angle between the asymptotes of a hyperbola -/
noncomputable def angle_between_asymptotes (a b : ℝ) : ℝ := 
  2 * Real.arctan (b / a)

/-- The distance from the focus to an asymptote of a hyperbola -/
noncomputable def distance_focus_to_asymptote (a b : ℝ) : ℝ := 
  b * Real.sqrt (a^2 + b^2) / (a^2 + b^2)

/-- Theorem about the focal length of a specific hyperbola -/
theorem hyperbola_focal_length 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : Real.sin (angle_between_asymptotes a b) = 4/5) 
  (h4 : distance_focus_to_asymptote a b = 1) :
  focal_length a b = Real.sqrt 5 ∨ focal_length a b = 2 * Real.sqrt 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l608_60872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_space_shuttle_speed_l608_60866

/-- The speed of a space shuttle in kilometers per hour -/
noncomputable def speed_km_per_hour : ℝ := 32400

/-- The number of seconds in an hour -/
noncomputable def seconds_per_hour : ℝ := 3600

/-- The speed of a space shuttle in kilometers per second -/
noncomputable def speed_km_per_second : ℝ := speed_km_per_hour / seconds_per_hour

/-- Theorem stating that the speed in km/s is 9 -/
theorem space_shuttle_speed : speed_km_per_second = 9 := by
  -- Unfold the definitions
  unfold speed_km_per_second speed_km_per_hour seconds_per_hour
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_space_shuttle_speed_l608_60866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_of_N_l608_60852

def M : Finset ℕ := {1, 2, 3}

def N : Finset ℕ := Finset.biUnion M (λ x => Finset.image (λ y => x + y) M)

theorem cardinality_of_N : Finset.card N = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_of_N_l608_60852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_l608_60829

/-- Two lines in 3D space -/
structure Line3D where
  direction : Fin 3 → ℝ

/-- The dot product of two 3D vectors -/
def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

/-- Two lines are perpendicular if their direction vectors' dot product is zero -/
def perpendicular (l1 l2 : Line3D) : Prop :=
  dot_product l1.direction l2.direction = 0

/-- The main theorem: given two lines with specific direction vectors, prove they are perpendicular -/
theorem lines_perpendicular : 
  let l1 : Line3D := ⟨fun i => [1, 2, -2].get i⟩
  let l2 : Line3D := ⟨fun i => [-2, 3, 2].get i⟩
  perpendicular l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_l608_60829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_body_with_disk_sections_is_ball_l608_60842

/-- A three-dimensional body. -/
structure Body3D where
  -- Add necessary fields here

/-- A plane in three-dimensional space. -/
structure Plane where
  -- Add necessary fields here

/-- A disk in three-dimensional space. -/
structure Disk where
  -- Add necessary fields here

/-- A ball in three-dimensional space. -/
structure Ball where
  -- Add necessary fields here

/-- A point in three-dimensional space. -/
structure Point where
  -- Add necessary fields here

/-- The intersection of a body and a plane. -/
def planarSection (B : Body3D) (P : Plane) : Set Point := sorry

/-- Convert a Ball to a Body3D -/
def ballToBody3D (S : Ball) : Body3D := sorry

/-- Convert a Disk to a Set Point -/
def diskToSetPoint (D : Disk) : Set Point := sorry

/-- The theorem stating that if every planar section of a body is a disk, then the body is a ball. -/
theorem body_with_disk_sections_is_ball (B : Body3D) :
  (∀ P : Plane, ∃ D : Disk, planarSection B P = diskToSetPoint D) →
  ∃ S : Ball, B = ballToBody3D S :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_body_with_disk_sections_is_ball_l608_60842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_391_l608_60864

theorem greatest_prime_factor_of_391 : 
  (Nat.factors 391).maximum? = some 23 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_391_l608_60864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_specific_l608_60890

/-- Given an angle θ with vertex at the origin, initial side along the positive x-axis,
    and terminal side passing through (1,-2), prove that sin 2θ = -4/5 -/
theorem sin_double_angle_specific (θ : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ r^2 = 5 ∧ Real.cos θ = 1/r ∧ Real.sin θ = -2/r) → 
  Real.sin (2*θ) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_specific_l608_60890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_is_one_range_of_a_l608_60801

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x ^ 2 + a * Real.cos x + (5/8) * a - 3/2

-- Part 1: Maximum value when a = 1
theorem max_value_when_a_is_one :
  ∃ (max : ℝ), (∀ x, f 1 x ≤ max) ∧ (∃ x, f 1 x = max) ∧ max = 3/8 := by
  sorry

-- Part 2: Range of a when f(x) ≤ 1 for x ∈ [0, π/2]
theorem range_of_a (a : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi/2 → f a x ≤ 1) ↔ a ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_is_one_range_of_a_l608_60801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_function_value_l608_60885

noncomputable section

-- Define the functions h and j
noncomputable def h (x : ℝ) : ℝ := 2 * x - 3
noncomputable def j (x : ℝ) : ℝ := x / 4

-- Define the inverse functions
noncomputable def h_inv (x : ℝ) : ℝ := (x + 3) / 2
noncomputable def j_inv (x : ℝ) : ℝ := 4 * x

-- State the theorem
theorem composite_function_value :
  h (j_inv (h_inv (h_inv (j (h 12))))) = 25.5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_function_value_l608_60885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mask_pricing_and_quantities_l608_60847

-- Define variables
variable (x y : ℚ) -- Price of medical masks and disinfectant
variable (m n : ℕ) -- Number of N95 masks and disinfectant bottles

-- Define conditions
def total_budget : ℚ := 3500
def condition1 (x y : ℚ) : Prop := 800 * x + 120 * y = 3600
def condition2 (x y : ℚ) : Prop := 1000 * x + 100 * y = 3500
def n95_price : ℚ := 6
def total_masks : ℕ := 1000

-- Theorem to prove
theorem mask_pricing_and_quantities :
  ∀ x y : ℚ, condition1 x y ∧ condition2 x y →
  x = 3/2 ∧ y = 20 ∧
  ∀ m n : ℕ, n = 100 - (9/40) * m ∧
  (m = 120 ∨ m = 160) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mask_pricing_and_quantities_l608_60847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zeros_a_range_l608_60800

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - a
noncomputable def g (x : ℝ) : ℝ := (1/3) * x^3 - 2 * x^2 + 3 * x + 16/3

-- Theorem for the number of zeros of f
theorem f_zeros (a : ℝ) :
  (a < 0 ∨ a = 1 → (∃! x, f a x = 0)) ∧
  (0 ≤ a ∧ a < 1 → ¬∃ x, f a x = 0) ∧
  (a > 1 → ∃ x₁ x₂, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) :=
by sorry

-- Theorem for the range of a
theorem a_range :
  {a : ℝ | ∀ x₁ ∈ Set.Icc (-1) 2, ∃ x₂ ∈ Set.Icc (-1) 2, f a x₁ ≥ g x₂} = Set.Iic 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zeros_a_range_l608_60800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l608_60865

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 1) / x

theorem f_properties :
  ∀ x : ℝ, x ≠ 0 →
    (∀ ε > 0, ε < |x| → (f (x + ε) - f x) / ε > 0) ∧
    (x > 0 → f x > x * Real.log (x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l608_60865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_cones_apex_angle_l608_60824

/-- The apex angle of a cone is the angle between its generatrices in an axial section. -/
noncomputable def apex_angle (cone : Type) : ℝ := sorry

/-- A cone touches another cone externally if they share a common tangent plane at their point of contact. -/
def touches_externally (cone1 cone2 : Type) : Prop := sorry

/-- A cone touches another cone internally if one is contained within the other and they share a common tangent plane at their point of contact. -/
def touches_internally (cone1 cone2 : Type) : Prop := sorry

/-- The theorem stating the relationship between the apex angles of four cones under specific conditions. -/
theorem four_cones_apex_angle 
  (cone1 cone2 cone3 cone4 : Type)
  (h1 : apex_angle cone1 = π/3)
  (h2 : apex_angle cone2 = π/3)
  (h3 : apex_angle cone3 = π/3)
  (h4 : touches_externally cone1 cone2)
  (h5 : touches_externally cone2 cone3)
  (h6 : touches_externally cone3 cone1)
  (h7 : touches_internally cone4 cone1)
  (h8 : touches_internally cone4 cone2)
  (h9 : touches_internally cone4 cone3) :
  apex_angle cone4 = π/3 + 2 * Real.arcsin (1 / Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_cones_apex_angle_l608_60824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_f_14_x_l608_60814

def x : ℕ := (List.range 12).foldl (fun acc i => acc * (2 * (i + 1))) 1

def f (a b : ℕ) : ℕ := a^2 + b

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem greatest_prime_factor_of_f_14_x :
  ∃ (p : ℕ), is_prime p ∧ p ∣ f 14 x ∧ ∀ (q : ℕ), is_prime q → q ∣ f 14 x → q ≤ p ∧ p = 23 := by
  sorry

#eval x
#eval f 14 x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_f_14_x_l608_60814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l608_60877

/-- Given two vectors a and b, prove that a = 1 is a sufficient but not necessary condition for a ⟂ b -/
theorem perpendicular_condition (a : ℝ) : 
  let vec_a : Fin 3 → ℝ := ![1, 2, -3]
  let vec_b : Fin 3 → ℝ := ![2, a - 1, a^2 - 1/3]
  (∃ x : ℝ, x ≠ 1 ∧ (vec_a 0) * (vec_b 0) + (vec_a 1) * (vec_b 1) + (vec_a 2) * (vec_b 2) = 0) ∧
  (a = 1 → (vec_a 0) * (vec_b 0) + (vec_a 1) * (vec_b 1) + (vec_a 2) * (vec_b 2) = 0) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l608_60877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_notebook_page_sum_l608_60809

theorem notebook_page_sum (sheets : Nat) (torn_sheets : Nat) (claimed_sum : Nat) : 
  sheets = 96 → 
  torn_sheets = 25 → 
  claimed_sum = 2002 → 
  ∃ (actual_sum : Nat), 
    actual_sum = torn_sheets * (sheets * 2 + 1) ∧ 
    actual_sum ≠ claimed_sum :=
by
  intro h_sheets h_torn h_claimed
  let actual_sum := torn_sheets * (sheets * 2 + 1)
  exists actual_sum
  constructor
  · rfl
  · sorry  -- The actual proof would go here, but we're using sorry to skip it for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_notebook_page_sum_l608_60809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intercept_sum_l608_60898

/-- A line in the form 4x + 7y + c = 0 -/
structure Line where
  c : ℝ

/-- The x-intercept of the line -/
noncomputable def x_intercept (l : Line) : ℝ := -l.c / 4

/-- The y-intercept of the line -/
noncomputable def y_intercept (l : Line) : ℝ := -l.c / 7

/-- The sum of x-intercept and y-intercept -/
noncomputable def intercept_sum (l : Line) : ℝ := x_intercept l + y_intercept l

theorem line_intercept_sum (l : Line) : intercept_sum l = 22 → l.c = -56 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intercept_sum_l608_60898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_property_l608_60841

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  third_term : a 3 = 3
  fifth_term : a 5 = 6
  geometric_subset : ∃ (m : ℕ) (r : ℝ), a 5 = r * a 3 ∧ a m = r * a 5

/-- The theorem stating that m equals 9 for the given arithmetic sequence -/
theorem arithmetic_geometric_sequence_property (seq : ArithmeticSequence) : 
  ∃ m : ℕ, m = 9 ∧ (∃ r : ℝ, seq.a 5 = r * seq.a 3 ∧ seq.a m = r * seq.a 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_property_l608_60841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_2f_l608_60803

noncomputable def f (n : ℕ) : ℝ :=
  (3 + Real.sqrt 3) / 6 * ((1 + Real.sqrt 3) / 2) ^ n +
  (3 - Real.sqrt 3) / 6 * ((1 - Real.sqrt 3) / 2) ^ n

noncomputable def g (n : ℕ) : ℝ :=
  (2 + Real.sqrt 2) / 4 * ((1 + Real.sqrt 2) / 2) ^ n +
  (2 - Real.sqrt 2) / 4 * ((1 - Real.sqrt 2) / 2) ^ n

theorem g_sum_equals_2f (n : ℕ) : g (n + 1) + g (n - 1) = 2 * f n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_2f_l608_60803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_population_approx_l608_60823

noncomputable def doubling_time : ℝ := 2
noncomputable def growth_time : ℝ := 17.931568569324174
noncomputable def final_population : ℝ := 500000

noncomputable def initial_population : ℝ := final_population / (2 ^ (growth_time / doubling_time))

theorem initial_population_approx :
  Int.floor initial_population = 1010 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_population_approx_l608_60823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorial_divisible_by_9450_l608_60849

theorem smallest_factorial_divisible_by_9450 :
  (∃ n : ℕ, n > 0 ∧ 9450 ∣ n.factorial ∧ ∀ m : ℕ, m > 0 → m < n → ¬(9450 ∣ m.factorial)) ∧
  (∀ n : ℕ, n > 0 ∧ 9450 ∣ n.factorial ∧ ∀ m : ℕ, m > 0 → m < n → ¬(9450 ∣ m.factorial) → n = 10) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorial_divisible_by_9450_l608_60849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_is_50_l608_60860

/-- The area of a parallelogram constructed on two vectors -/
def parallelogramArea (a b : ℝ × ℝ) : ℝ :=
  abs ((a.1 * b.2) - (a.2 * b.1))

theorem parallelogram_area_is_50 
  (p q : ℝ × ℝ) 
  (h_p_norm : Real.sqrt (p.1^2 + p.2^2) = 10)
  (h_q_norm : Real.sqrt (q.1^2 + q.2^2) = 1)
  (h_orthogonal : p.1 * q.1 + p.2 * q.2 = 0)
  (a b : ℝ × ℝ)
  (h_a : a = (3 * p.1 + 2 * q.1, 3 * p.2 + 2 * q.2))
  (h_b : b = (p.1 - q.1, p.2 - q.2)) :
  parallelogramArea a b = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_is_50_l608_60860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_ratio_l608_60886

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line
def line (x y : ℝ) : Prop := x + 2 * Real.sqrt 3 * y - 4 * Real.sqrt 3 = 0

-- Define the foci
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define a point on the line
noncomputable def P : ℝ × ℝ := sorry

-- Define the distance between two points
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- Define the angle between three points
noncomputable def angle (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem max_angle_ratio :
  ∀ P : ℝ × ℝ,
  line P.1 P.2 →
  (∀ Q : ℝ × ℝ, line Q.1 Q.2 → angle F₁ Q F₂ ≤ angle F₁ P F₂) →
  distance P F₁ / distance P F₂ = Real.sqrt 15 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_ratio_l608_60886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_diametrically_opposite_pair_l608_60859

/-- A point on the circumference of a circle -/
structure CirclePoint where
  angle : ℝ
  inv_angle_nonneg : 0 ≤ angle
  inv_angle_lt_two_pi : angle < 2 * Real.pi

/-- An arc on the circumference of a circle -/
structure CircleArc where
  start : CirclePoint
  length : ℕ
  inv_length_pos : 0 < length
  inv_length_le_three : length ≤ 3

/-- A configuration of points and arcs on a circle -/
structure CircleConfiguration (k : ℕ) where
  points : Finset CirclePoint
  arcs : Finset CircleArc
  inv_points_card : points.card = 3 * k
  inv_arcs_card : arcs.card = 3 * k
  inv_arc_lengths : ∀ l, l ∈ ({1, 2, 3} : Finset ℕ) → (arcs.filter (λ a => a.length = l)).card = k

/-- Two points are diametrically opposite if their angles differ by π -/
def diametrically_opposite (p q : CirclePoint) : Prop :=
  abs (p.angle - q.angle) = Real.pi

/-- The main theorem -/
theorem exists_diametrically_opposite_pair (k : ℕ) (config : CircleConfiguration k) :
  ∃ p q, p ∈ config.points ∧ q ∈ config.points ∧ p ≠ q ∧ diametrically_opposite p q :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_diametrically_opposite_pair_l608_60859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l608_60853

-- Define the curves and line
def C₁ (ρ θ : ℝ) : Prop := ρ^2 + 8*ρ*Real.cos θ - 6*ρ*Real.sin θ + 24 = 0

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := (8 * Real.cos θ, 3 * Real.sin θ)

def C₃ (t : ℝ) : ℝ × ℝ := (3 + 2*t, -2 + t)

-- Define point P on C₁
def P : ℝ × ℝ := (-4, 4)

-- Define the midpoint M of PQ
noncomputable def M (θ : ℝ) : ℝ × ℝ := 
  let Q := C₂ θ
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define the distance function from a point to a line
noncomputable def distToLine (p : ℝ × ℝ) : ℝ :=
  |p.1 - 2*p.2 - 7| / Real.sqrt 5

-- Theorem statement
theorem min_distance_to_line :
  ∃ (d : ℝ), d = 8 * Real.sqrt 5 / 5 ∧ 
  ∀ (θ : ℝ), distToLine (M θ) ≥ d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l608_60853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_start_time_l608_60811

/-- The speed of l in km/hr -/
noncomputable def speed_l : ℝ := 50

/-- The speed of k in km/hr -/
noncomputable def speed_k : ℝ := speed_l * 1.5

/-- The total distance between l and k in km -/
noncomputable def total_distance : ℝ := 300

/-- The time l travels in hours -/
noncomputable def travel_time_l : ℝ := 3

/-- The time k travels in hours -/
noncomputable def travel_time_k : ℝ := total_distance / 2 / speed_k

theorem k_start_time (h : travel_time_l - travel_time_k = 1) : 
  travel_time_k = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_start_time_l608_60811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_approx_l608_60818

theorem sqrt_sum_approx : 
  abs ((Real.sqrt 1.5) / (Real.sqrt 0.81) + (Real.sqrt 1.44) / (Real.sqrt 0.49) - 3.075) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_approx_l608_60818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_song_time_is_367_l608_60897

/-- Represents a radio show with its duration and segments -/
structure RadioShow where
  duration : ℕ
  talkingSegments : ℕ
  talkingDuration : ℕ
  adBreaks : ℕ
  adDuration : ℕ

/-- Calculates the time spent on songs for a single radio show -/
def songTime (radioShow : RadioShow) : ℕ :=
  radioShow.duration - (radioShow.talkingSegments * radioShow.talkingDuration) - (radioShow.adBreaks * radioShow.adDuration)

/-- The three radio shows as described in the problem -/
def show1 : RadioShow := { duration := 180, talkingSegments := 3, talkingDuration := 10, adBreaks := 5, adDuration := 5 }
def show2 : RadioShow := { duration := 240, talkingSegments := 4, talkingDuration := 15, adBreaks := 6, adDuration := 4 }
def show3 : RadioShow := { duration := 120, talkingSegments := 2, talkingDuration := 8, adBreaks := 3, adDuration := 6 }

/-- Theorem stating that the total song time for all shows is 367 minutes -/
theorem total_song_time_is_367 : 
  songTime show1 + songTime show2 + songTime show3 = 367 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_song_time_is_367_l608_60897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sin_sum_when_cos_sum_max_l608_60846

theorem triangle_angle_sin_sum_when_cos_sum_max (α β γ : Real) : 
  α + β + γ = Real.pi → 
  (∀ a b c, a + b + c = Real.pi → Real.cos a + Real.cos b + Real.cos c ≤ Real.cos α + Real.cos β + Real.cos γ) →
  Real.sin α + Real.sin β + Real.sin γ = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sin_sum_when_cos_sum_max_l608_60846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C_C₁_no_common_points_C_rect_eq_C₁_is_circle_l608_60806

-- Define the curve C in polar coordinates
noncomputable def C_polar (θ : ℝ) : ℝ × ℝ := 
  (2 * Real.sqrt 2 * Real.cos θ * Real.cos θ, 2 * Real.sqrt 2 * Real.cos θ * Real.sin θ)

-- Define the curve C in rectangular coordinates
def C_rect (x y : ℝ) : Prop := (x - Real.sqrt 2)^2 + y^2 = 2

-- Define point A
def A : ℝ × ℝ := (1, 0)

-- Define the curve C₁ in parametric form
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ := 
  (3 - Real.sqrt 2 + 2 * Real.cos θ, 2 * Real.sin θ)

-- Statement: C and C₁ have no common points
theorem C_C₁_no_common_points : ∀ θ₁ θ₂ : ℝ, C_polar θ₁ ≠ C₁ θ₂ := by
  sorry

-- Additional theorem to show the rectangular form of C
theorem C_rect_eq : ∀ x y : ℝ, C_rect x y ↔ (x - Real.sqrt 2)^2 + y^2 = 2 := by
  sorry

-- Theorem to show that C₁ is a circle
theorem C₁_is_circle : ∀ θ : ℝ, 
  let (x, y) := C₁ θ
  (x - (3 - Real.sqrt 2))^2 + y^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C_C₁_no_common_points_C_rect_eq_C₁_is_circle_l608_60806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_table_filling_theorem_l608_60838

theorem table_filling_theorem (n : ℕ+) : 
  ∃ (table : Fin n → Fin n → ℕ), 
    (∀ i j, table i j ∈ Finset.range (n^2 + 1) \ {0}) ∧ 
    (∀ i j₁ j₂, j₁ ≠ j₂ → table i j₁ ≠ table i j₂) ∧
    (∀ i₁ i₂ j, i₁ ≠ i₂ → table i₁ j ≠ table i₂ j) ∧
    (∀ i, ∃ k : ℕ, (Finset.sum (Finset.range n) (λ j ↦ table i j)) = k * n) ∧
    (∀ j, ∃ k : ℕ, (Finset.sum (Finset.range n) (λ i ↦ table i j)) = k * n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_table_filling_theorem_l608_60838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_around_circle_l608_60839

/-- The shortest path from (0,0) to (15,20) avoiding a circle -/
theorem shortest_path_around_circle :
  let start : ℝ × ℝ := (0, 0)
  let end_point : ℝ × ℝ := (15, 20)
  let circle_center : ℝ × ℝ := (7, 9)
  let circle_radius : ℝ := 6
  ∃ (path : Set (ℝ × ℝ)),
    (start ∈ path) ∧
    (end_point ∈ path) ∧
    (∀ p ∈ path, (p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2 ≥ circle_radius^2) ∧
    (∀ other_path : Set (ℝ × ℝ),
      (start ∈ other_path) ∧
      (end_point ∈ other_path) ∧
      (∀ p ∈ other_path, (p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2 ≥ circle_radius^2) →
      ∃ (length : ℝ), length ≥ Real.sqrt 94 * 2 + 2 * Real.pi) ∧
    ∃ (path_length : ℝ), path_length = Real.sqrt 94 * 2 + 2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_around_circle_l608_60839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomial_and_multiple_l608_60884

theorem gcd_of_polynomial_and_multiple (y : ℤ) (h : ∃ k : ℤ, y = 45678 * k) :
  Nat.gcd ((3 * y + 4) * (8 * y + 3) * (14 * y + 9) * (y + 14)).natAbs y.natAbs = 1512 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomial_and_multiple_l608_60884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_mod_five_l608_60836

def sequenceList : List Nat := [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]

theorem product_remainder_mod_five :
  (sequenceList.prod) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_mod_five_l608_60836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_diophantine_equation_l608_60826

theorem solution_to_diophantine_equation :
  ∀ (x y z p : ℕ),
    (x > 0) → (y > 0) → (z > 0) → (p > 0) →
    (x^p + y^p = p^z) ∧ (Nat.Prime (12 * 148 * p)) →
    (∃ (k : ℕ),
      ((x = 2^k ∧ y = 2^k ∧ z = 2*k + 1 ∧ p = 2) ∨
       (x = 3^k ∧ y = 2 * 3^k ∧ z = 2 + 3*k ∧ p = 3) ∨
       (x = 2 * 3^k ∧ y = 3^k ∧ z = 2 + 3*k ∧ p = 3))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_diophantine_equation_l608_60826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l608_60868

theorem simplify_expression (n : ℕ) : (-2 : ℤ)^(2*n+1) + 2*(-2 : ℤ)^(2*n) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l608_60868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_modulo_and_minimum_value_l608_60817

/-- α is the positive root of x^2 + x = 5 -/
noncomputable def α : ℝ := ((-1 + Real.sqrt 21) / 2)

/-- The sum of c_i * α^i from i = 0 to n equals 2015 -/
def sum_condition (c : ℕ → ℕ) (n : ℕ) : Prop :=
  Finset.sum (Finset.range (n + 1)) (fun i => (c i : ℝ) * α^i) = 2015

theorem sum_modulo_and_minimum_value (c : ℕ → ℕ) (n : ℕ) 
  (h : sum_condition c n) :
  (Finset.sum (Finset.range (n + 1)) c) % 3 = 2 ∧
  29 ≤ Finset.sum (Finset.range (n + 1)) c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_modulo_and_minimum_value_l608_60817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_z_l608_60822

-- Define the constraints
def constraint1 (x y : ℝ) : Prop := x - 4*y ≤ -3
def constraint2 (x y : ℝ) : Prop := 3*x + 5*y ≤ 25
def constraint3 (x : ℝ) : Prop := x ≥ 1

-- Define the function z
noncomputable def z (x y : ℝ) : ℝ := (4:ℝ)^x * (2:ℝ)^y

-- State the theorem
theorem min_value_of_z :
  ∃ (x y : ℝ), constraint1 x y ∧ constraint2 x y ∧ constraint3 x ∧
  (∀ (x' y' : ℝ), constraint1 x' y' → constraint2 x' y' → constraint3 x' → z x y ≤ z x' y') ∧
  z x y = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_z_l608_60822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l608_60804

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis
  c : ℝ  -- focal distance

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := h.c / h.a

/-- Condition that line FB is perpendicular to an asymptote -/
def perpendicular_to_asymptote (h : Hyperbola) : Prop :=
  h.b / h.a = h.c / h.b

theorem hyperbola_eccentricity (h : Hyperbola) 
  (perp : perpendicular_to_asymptote h) :
  eccentricity h = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l608_60804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_at_P_l608_60856

noncomputable section

-- Define the curve
def f (x : ℝ) : ℝ := x^4 - 2*x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 4*x^3 - 2

-- Define the point P
def P : ℝ × ℝ := (1, -1)

-- Define the slope of the line x + 2y + 1 = 0
def m_line : ℝ := -1/2

theorem tangent_perpendicular_at_P :
  let m_tangent := f' P.fst
  m_tangent * m_line = -1 ∧ f P.fst = P.snd := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_at_P_l608_60856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l608_60851

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define the foci of the hyperbola
def left_focus : ℝ × ℝ := (-2, 0)
def right_focus : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem hyperbola_triangle_area (x y : ℝ) :
  hyperbola x y →
  distance x y (left_focus.1) (left_focus.2) + distance x y (right_focus.1) (right_focus.2) = 2 * Real.sqrt 5 →
  (1/2) * distance (left_focus.1) (left_focus.2) (right_focus.1) (right_focus.2) *
    Real.sqrt (1 - (distance (left_focus.1) (left_focus.2) (right_focus.1) (right_focus.2))^2 /
    ((distance x y (left_focus.1) (left_focus.2) + distance x y (right_focus.1) (right_focus.2))^2)) = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l608_60851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_conditions_imply_valid_a_l608_60863

-- Define the propositions p and q as functions of a
def p (a : ℝ) : Prop := ∃! x : ℝ, x > 1 ∧ x < 2 ∧ x^2 - 2*x + a = 0

def q (a : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + (2*a - 3)*x + 1 = 0 ∧ y^2 + (2*a - 3)*y + 1 = 0

-- Define the set of valid a values
def valid_a : Set ℝ := {a | a ≤ 0 ∨ (1/2 ≤ a ∧ a < 1) ∨ a > 5/2}

-- State the theorem
theorem proposition_conditions_imply_valid_a :
  ∀ a : ℝ, (¬(p a ∧ q a) ∧ (p a ∨ q a)) → a ∈ valid_a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_conditions_imply_valid_a_l608_60863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_rainfall_hours_is_45_l608_60825

/-- Represents the rainfall data for a week with two rainstorms. -/
structure RainfallData where
  first_storm_rate : ℚ
  second_storm_rate : ℚ
  first_storm_duration : ℚ
  total_rainfall : ℚ

/-- Calculates the total hours of rainfall given the rainfall data. -/
def total_rainfall_hours (data : RainfallData) : ℚ :=
  let second_storm_duration := (data.total_rainfall - data.first_storm_rate * data.first_storm_duration) / data.second_storm_rate
  data.first_storm_duration + second_storm_duration

/-- Theorem stating that for the given rainfall data, the total hours of rainfall is 45. -/
theorem total_rainfall_hours_is_45 (data : RainfallData) 
  (h1 : data.first_storm_rate = 30)
  (h2 : data.second_storm_rate = 15)
  (h3 : data.first_storm_duration = 20)
  (h4 : data.total_rainfall = 975) :
  total_rainfall_hours data = 45 := by
  sorry

#eval total_rainfall_hours { first_storm_rate := 30, second_storm_rate := 15, first_storm_duration := 20, total_rainfall := 975 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_rainfall_hours_is_45_l608_60825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_tank_used_is_five_twelfths_l608_60861

/-- Represents a car with its characteristics and trip details -/
structure Car where
  speed : ℚ  -- Speed in miles per hour
  fuelEfficiency : ℚ  -- Miles per gallon
  tankCapacity : ℚ  -- Tank capacity in gallons
  travelTime : ℚ  -- Travel time in hours

/-- Calculates the fraction of a full tank used for a given car and trip -/
def fractionOfTankUsed (car : Car) : ℚ :=
  (car.speed * car.travelTime) / (car.fuelEfficiency * car.tankCapacity)

/-- Theorem stating that for the given car specifications, the fraction of tank used is 5/12 -/
theorem fraction_of_tank_used_is_five_twelfths :
  let car : Car := {
    speed := 40,
    fuelEfficiency := 40,
    tankCapacity := 12,
    travelTime := 5
  }
  fractionOfTankUsed car = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_tank_used_is_five_twelfths_l608_60861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l608_60888

/-- A function defined on non-zero real numbers -/
noncomputable def f : ℝ → ℝ := sorry

/-- The derivative of f -/
noncomputable def f' : ℝ → ℝ := sorry

/-- For x > 0, xf'(x) - f(x) < 0 -/
axiom h1 : ∀ x > 0, x * f' x - f x < 0

/-- Definition of a -/
noncomputable def a : ℝ := f (2^(1/5)) / (2^(1/5))

/-- Definition of b -/
noncomputable def b : ℝ := f (0.2^2) / (0.2^2)

/-- Definition of c -/
noncomputable def c : ℝ := f (Real.log 5 / Real.log 2) / (Real.log 5 / Real.log 2)

/-- Theorem to prove -/
theorem order_of_abc : c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l608_60888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_triangle_perimeter_approx_l608_60871

/-- Triangle PQR with given side lengths and parallel lines intersecting its interior --/
structure TriangleWithParallelLines where
  PQ : ℝ
  QR : ℝ
  PR : ℝ
  m_P_length : ℝ
  m_Q_length : ℝ
  m_R_length : ℝ
  h_PQ : PQ = 150
  h_QR : QR = 270
  h_PR : PR = 210
  h_m_P : m_P_length = 60
  h_m_Q : m_Q_length = 35
  h_m_R : m_R_length = 25

/-- The perimeter of the triangle formed by the intersections of parallel lines --/
noncomputable def inner_triangle_perimeter (t : TriangleWithParallelLines) : ℝ :=
  (t.m_P_length / t.PQ) * t.PR +
  (t.m_Q_length / t.QR) * t.PQ +
  (t.m_R_length / t.PR) * t.QR

/-- The theorem stating that the perimeter of the inner triangle is approximately 135.57 --/
theorem inner_triangle_perimeter_approx (t : TriangleWithParallelLines) :
  ∃ ε > 0, abs (inner_triangle_perimeter t - 135.57) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_triangle_perimeter_approx_l608_60871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_cost_per_pound_l608_60850

/-- Calculates the cost of salt per pound based on given conditions --/
theorem salt_cost_per_pound 
  (flour_needed : ℕ)
  (flour_bag_weight : ℕ)
  (flour_bag_cost : ℚ)
  (salt_needed : ℕ)
  (promotion_cost : ℚ)
  (ticket_price : ℚ)
  (tickets_sold : ℕ)
  (profit : ℚ)
  (h1 : flour_needed = 500)
  (h2 : flour_bag_weight = 50)
  (h3 : flour_bag_cost = 20)
  (h4 : salt_needed = 10)
  (h5 : promotion_cost = 1000)
  (h6 : ticket_price = 20)
  (h7 : tickets_sold = 500)
  (h8 : profit = 8798) :
  ∃ (salt_cost_per_pound : ℚ),
    (ticket_price * tickets_sold - (promotion_cost + (flour_needed / flour_bag_weight * flour_bag_cost) + salt_needed * salt_cost_per_pound) = profit) ∧
    salt_cost_per_pound = 120.20 := by
  sorry

#check salt_cost_per_pound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_cost_per_pound_l608_60850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_and_triangle_area_l608_60813

noncomputable def m (ω : ℝ) (x : ℝ) : ℝ × ℝ := 
  (Real.sin (ω * x) + Real.cos (ω * x), Real.sqrt 3 * Real.cos (ω * x))

noncomputable def n (ω : ℝ) (x : ℝ) : ℝ × ℝ := 
  (Real.cos (ω * x) - Real.sin (ω * x), 2 * Real.sin (ω * x))

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  (m ω x).1 * (n ω x).1 + (m ω x).2 * (n ω x).2

theorem max_omega_and_triangle_area (ω : ℝ) (A B C : ℝ) (a b c : ℝ) :
  ω > 0 →
  (∀ x : ℝ, ∃ y : ℝ, y - x ≥ π / 2 ∧ f ω x = f ω y) →
  a = 2 →
  (∃ A : ℝ, f 1 A = 1) →
  (ω ≤ 1 ∧
   ∃ S : ℝ, S = Real.sqrt 3 ∧
   ∀ A B C : ℝ, A + B + C = π →
   ∀ a b c : ℝ, a = 2 → b > 0 → c > 0 →
   a / Real.sin A = b / Real.sin B → b / Real.sin B = c / Real.sin C →
   1 / 2 * b * c * Real.sin A ≤ S) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_and_triangle_area_l608_60813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interview_problem_l608_60805

theorem interview_problem (n : ℕ) : 
  n > 2 →  -- Ensure n is large enough for the problem to make sense
  (Nat.choose n 3 : ℚ) ≠ 0 →  -- Ensure the denominator is not zero
  (Nat.choose 2 2 * Nat.choose (n - 2) 1 : ℚ) / (Nat.choose n 3 : ℚ) = 1 / 70 → 
  n = 21 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interview_problem_l608_60805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l608_60878

open Real

theorem trigonometric_equation_solution (t : ℝ) : 
  (sin (t + π/4))^2 - (sin (t - π/6))^2 - sin (π/12) * cos (2*t + π/12) = 0.5 * sin (6*t) ↔ 
  (∃ k : ℤ, t = k * π/2) ∨ (∃ l : ℤ, t = π/12 + l * π/2) ∨ (∃ m : ℤ, t = -π/12 + m * π/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l608_60878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolas_intersection_l608_60815

noncomputable def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 12 * x - 15
noncomputable def parabola2 (x : ℝ) : ℝ := x^2 - 2 * x + 1

noncomputable def point1 : ℝ × ℝ := ((5 - Real.sqrt 57) / 2, 21 - 2 * Real.sqrt 57)
noncomputable def point2 : ℝ × ℝ := ((5 + Real.sqrt 57) / 2, 21 + 2 * Real.sqrt 57)

theorem parabolas_intersection :
  (∀ x y : ℝ, parabola1 x = parabola2 x → (x, y) = point1 ∨ (x, y) = point2) ∧
  parabola1 point1.1 = parabola2 point1.1 ∧
  parabola1 point2.1 = parabola2 point2.1 := by
  sorry

#check parabolas_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolas_intersection_l608_60815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_a_eq_four_l608_60821

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- State the theorem
theorem floor_of_a_eq_four (a : ℝ) (h : 3^a + a^3 = 123) : floor a = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_a_eq_four_l608_60821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l608_60808

-- Define the vectors a and b
noncomputable def a : ℝ × ℝ := (Real.sqrt 3, 0)
noncomputable def b (x : ℝ) : ℝ × ℝ := (x, -2)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- State the theorem
theorem perpendicular_vectors :
  ∃ x : ℝ, dot_product a (a.1 - 2 * (b x).1, a.2 - 2 * (b x).2) = 0 ∧ x = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l608_60808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l608_60816

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1

-- Define the line l passing through A(0,1) with slope k
def lineL (k x y : ℝ) : Prop := y = k * x + 1

-- Define the dot product of vectors OM and ON
def dotProduct (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

-- Theorem statement
theorem intersection_length (k : ℝ) (x1 y1 x2 y2 : ℝ) :
  circleC x1 y1 ∧ circleC x2 y2 ∧  -- M and N are on the circle
  lineL k x1 y1 ∧ lineL k x2 y2 ∧  -- M and N are on the line
  dotProduct x1 y1 x2 y2 = 12   -- OM · ON = 12
  → 
  (x1 - x2)^2 + (y1 - y2)^2 = 4  -- |MN|^2 = 2^2 = 4
  := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l608_60816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_symmetry_point_l608_60895

/-- Circle C with center on positive x-axis and radius 2 -/
structure CircleC where
  center : ℝ
  center_positive : center > 0
  radius : ℝ
  radius_eq_2 : radius = 2

/-- The chord of circle C cut by the line y = √3x has length √13 -/
def chord_condition (c : CircleC) : Prop :=
  ∃ (x₁ x₂ : ℝ), (x₁ - c.center)^2 + (Real.sqrt 3 * x₁)^2 = c.radius^2 ∧
                 (x₂ - c.center)^2 + (Real.sqrt 3 * x₂)^2 = c.radius^2 ∧
                 (x₂ - x₁)^2 + (Real.sqrt 3 * (x₂ - x₁))^2 = 13

/-- The equation of circle C is (x-1)² + y² = 4 -/
theorem circle_equation (c : CircleC) (h : chord_condition c) :
  c.center = 1 ∧ ∀ (x y : ℝ), (x - c.center)^2 + y^2 = c.radius^2 ↔ (x - 1)^2 + y^2 = 4 := by
  sorry

/-- Point N(5,0) ensures symmetry of AN and BN about x-axis -/
theorem symmetry_point (c : CircleC) (h : chord_condition c) :
  ∃ (N : ℝ × ℝ), N.1 = 5 ∧ N.2 = 0 ∧
    ∀ (k : ℝ), ∀ (A B : ℝ × ℝ),
      (A.1 - c.center)^2 + A.2^2 = c.radius^2 ∧
      (B.1 - c.center)^2 + B.2^2 = c.radius^2 ∧
      A.2 = k * (A.1 - 2) ∧ B.2 = k * (B.1 - 2) →
      (A.2 / (A.1 - N.1)) + (B.2 / (B.1 - N.1)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_symmetry_point_l608_60895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_A_coordinates_l608_60830

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin in the Cartesian coordinate system -/
def origin : Point := ⟨0, 0⟩

/-- Distance between two points in the Cartesian coordinate system -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem: Coordinates of point A given its properties -/
theorem point_A_coordinates :
  ∀ A : Point,
  A.x = 0 →                        -- A lies on the y-axis
  A.y > 0 →                        -- A is above the origin
  distance A origin = 3 →          -- A is at a distance of 3 units from the origin
  A = ⟨0, 3⟩                       -- The coordinates of A are (0, 3)
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_A_coordinates_l608_60830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_achieves_target_profit_l608_60881

/-- Represents the pomegranate sales problem -/
structure PomegranateSales where
  cost : ℚ
  originalPrice : ℚ
  originalSales : ℚ
  priceReductionFactor : ℚ
  salesIncreaseFactor : ℚ
  targetProfit : ℚ

/-- Calculates the daily sales volume based on price reduction -/
def dailySales (p : PomegranateSales) (reduction : ℚ) : ℚ :=
  p.originalSales + (p.salesIncreaseFactor / p.priceReductionFactor) * reduction

/-- Calculates the daily profit based on price reduction -/
def dailyProfit (p : PomegranateSales) (reduction : ℚ) : ℚ :=
  (p.originalPrice - reduction - p.cost) * (dailySales p reduction)

/-- Theorem stating that a price reduction of 1.5 yuan results in the target profit -/
theorem price_reduction_achieves_target_profit (p : PomegranateSales) 
  (h1 : p.cost = 2)
  (h2 : p.originalPrice = 6)
  (h3 : p.originalSales = 150)
  (h4 : p.priceReductionFactor = 1/2)
  (h5 : p.salesIncreaseFactor = 50)
  (h6 : p.targetProfit = 750) :
  dailyProfit p (3/2) = p.targetProfit := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_achieves_target_profit_l608_60881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_proof_l608_60819

theorem problem_proof : ((((3 : ℝ) + 1)⁻¹ + 1)⁻¹ + 1 - 1)⁻¹ + 1 = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_proof_l608_60819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_decreasing_iff_m_range_l608_60880

open Real

/-- A function f is monotonically decreasing on an interval (a, b) if for any x₁, x₂ in (a, b) with x₁ < x₂, we have f(x₁) > f(x₂) -/
def MonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x₁ x₂, a < x₁ → x₁ < x₂ → x₂ < b → f x₁ > f x₂

/-- The function f(x) = (m - 2 sin x) / cos x -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 2 * sin x) / cos x

theorem f_monotonically_decreasing_iff_m_range :
  (∀ m : ℝ, MonotonicallyDecreasing (f m) 0 (π/2)) ↔ (∀ m : ℝ, m ≤ 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_decreasing_iff_m_range_l608_60880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simson_line_angle_theorem_l608_60891

/-- Simson line of a point with respect to a triangle -/
def SimsonLine (P : EuclideanSpace ℝ (Fin 2)) (triangle : Set (EuclideanSpace ℝ (Fin 2))) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

/-- Angle between two lines -/
def angleBetweenLines (l1 l2 : Set (EuclideanSpace ℝ (Fin 2))) : ℝ := sorry

/-- Arc between two points on a circle -/
def arcMeasure (P Q : EuclideanSpace ℝ (Fin 2)) (circle : Set (EuclideanSpace ℝ (Fin 2))) : ℝ := sorry

/-- Theorem: The angle between Simson lines is half the arc measure -/
theorem simson_line_angle_theorem 
  (A B C P Q : EuclideanSpace ℝ (Fin 2)) 
  (triangle : Set (EuclideanSpace ℝ (Fin 2))) 
  (circle : Set (EuclideanSpace ℝ (Fin 2))) :
  triangle = {A, B, C} →
  (∀ x, x ∈ circle ↔ ∃ r > 0, ∀ y ∈ triangle, ‖x - y‖ = r) →
  P ∈ circle →
  Q ∈ circle →
  angleBetweenLines (SimsonLine P triangle) (SimsonLine Q triangle) = 
    (1 / 2) * (arcMeasure P Q circle) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simson_line_angle_theorem_l608_60891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_l608_60870

-- Define area and dist as local variables
variable (area : Set (ℝ × ℝ) → ℝ)
variable (dist : (ℝ × ℝ) → (ℝ × ℝ) → ℝ)

theorem hexagon_area (S : ℝ) (h : S > 0) : ∃ (A : ℝ), A = (2/9) * S ∧ 
  ∃ (T : Set (ℝ × ℝ)) (H : Set (ℝ × ℝ)),
    (∃ (a b c : ℝ × ℝ), T = {a, b, c} ∧ area T = S) ∧
    (∀ (side : Set (ℝ × ℝ)), side ⊆ T → ∃ (p q : ℝ × ℝ), p ∈ side ∧ q ∈ side ∧ 
      ∃ (r : ℝ × ℝ), r ∈ side ∧ dist p r = dist r q ∧ dist p q = 3 * dist p r) ∧
    (∃ (T1 T2 : Set (ℝ × ℝ)), 
      (∀ p : ℝ × ℝ, p ∈ T1 ∨ p ∈ T2 → ∃ (side : Set (ℝ × ℝ)), side ⊆ T ∧ p ∈ side) ∧
      H = T1 ∩ T2) ∧
    area H = A :=
by
  sorry

-- Definitions:
-- area : Set (ℝ × ℝ) → ℝ (area of a set of points in 2D plane)
-- dist : (ℝ × ℝ) → (ℝ × ℝ) → ℝ (distance between two points)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_l608_60870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_matching_cube_l608_60896

/-- Represents a face of a cube --/
inductive Face : Type
| Top
| Bottom
| Front
| Back
| Left
| Right

/-- Represents a net of a cube --/
structure CubeNet where
  faces : List Face
  adjacency : List (Face × Face)

/-- Represents a cube configuration --/
structure Cube where
  faces : List Face
  adjacency : List (Face × Face)

/-- The given net of the cube --/
def givenNet : CubeNet := sorry

/-- The four cube configurations --/
def cubeConfigurations : List Cube := sorry

/-- Function to check if a cube matches a given net --/
def matchesNet (net : CubeNet) (cube : Cube) : Prop := sorry

/-- Theorem stating that only one cube configuration matches the given net --/
theorem unique_matching_cube :
  ∃! c, c ∈ cubeConfigurations ∧ matchesNet givenNet c := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_matching_cube_l608_60896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l608_60876

noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (r^n - 1) / (r - 1)

theorem geometric_series_sum :
  let a : ℝ := -2
  let r : ℝ := 4
  let n : ℕ := 7
  geometric_sum a r n = -10922 := by
  -- Unfold the definition of geometric_sum
  unfold geometric_sum
  -- Simplify the expression
  simp [Real.rpow_nat_cast]
  -- Perform numerical computations
  norm_num
  -- Complete the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l608_60876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_ceiling_product_l608_60882

theorem floor_ceiling_product : 
  Int.floor (-3.5 : ℝ) * Int.ceil (3.5 : ℝ) * 
  Int.floor (-2.5 : ℝ) * Int.ceil (2.5 : ℝ) * 
  Int.floor (-1.5 : ℝ) * Int.ceil (1.5 : ℝ) = -576 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_ceiling_product_l608_60882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_AB_l608_60867

/-- The distance between two points A and B --/
noncomputable def distance_AB : ℝ := 240

/-- The time it takes for persons A and B to meet at the midpoint in the first scenario --/
def meeting_time : ℝ := 6

/-- The additional speed of person A in the second scenario --/
def speed_difference : ℝ := 4

/-- The head start time for person B in the second scenario --/
def head_start : ℝ := 1

/-- The speed of person A in the first scenario --/
noncomputable def speed_A : ℝ := distance_AB / (2 * meeting_time)

/-- The speed of person B in the first scenario --/
noncomputable def speed_B : ℝ := distance_AB / (2 * meeting_time)

theorem distance_between_AB : 
  (meeting_time * speed_A = meeting_time * speed_B) ∧ 
  ((meeting_time - head_start) * (speed_A + speed_difference) = meeting_time * speed_B) →
  distance_AB = 240 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_AB_l608_60867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiger_tooth_aloe_cost_l608_60820

/-- Calculates the cost of each tiger tooth aloe given the conditions of Mike's plant purchase. -/
theorem tiger_tooth_aloe_cost 
  (rose_bush_cost : ℝ)
  (total_spent : ℝ)
  (total_rose_bushes : ℕ)
  (friend_rose_bushes : ℕ)
  (tiger_tooth_aloes : ℝ) :
  rose_bush_cost = 75 →
  total_rose_bushes = 6 →
  friend_rose_bushes = 2 →
  tiger_tooth_aloes = 2 →
  total_spent = 500 →
  (total_spent - rose_bush_cost * (total_rose_bushes - friend_rose_bushes : ℝ)) / tiger_tooth_aloes = 100 := by
  sorry

#check tiger_tooth_aloe_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiger_tooth_aloe_cost_l608_60820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_revenue_decrease_l608_60855

noncomputable def inflation_rate_old : ℝ := 0.25
noncomputable def inflation_rate_new : ℝ := 0.16
noncomputable def tax_rate : ℝ := 0.20

noncomputable def real_value_deposit : ℝ := 100000

noncomputable def nominal_value_old : ℝ := real_value_deposit
noncomputable def nominal_value_new : ℝ := real_value_deposit * (1 + inflation_rate_old)

noncomputable def interest_old : ℝ := nominal_value_old * inflation_rate_old
noncomputable def interest_new : ℝ := nominal_value_new * inflation_rate_new

noncomputable def tax_old : ℝ := interest_old * tax_rate
noncomputable def tax_new : ℝ := interest_new * tax_rate

noncomputable def real_tax_old : ℝ := tax_old / (1 + inflation_rate_old)
noncomputable def real_tax_new : ℝ := tax_new / ((1 + inflation_rate_old) * (1 + inflation_rate_new))

noncomputable def tax_decrease_percentage : ℝ := (real_tax_old - real_tax_new) / real_tax_old * 100

theorem tax_revenue_decrease :
  |tax_decrease_percentage - 31| < 0.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_revenue_decrease_l608_60855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_equation_l608_60832

/-- A parametric representation of a plane -/
structure ParametricPlane where
  v : ℝ → ℝ → ℝ × ℝ × ℝ
  h_v : ∀ s t, v s t = (2 + 2*s - t, 1 - 2*s, 4 - s + 3*t)

/-- The equation of a plane in the form Ax + By + Cz + D = 0 -/
structure PlaneEquation where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ
  h_A_pos : A > 0
  h_gcd : Nat.gcd (Int.natAbs A) (Nat.gcd (Int.natAbs B) (Nat.gcd (Int.natAbs C) (Int.natAbs D))) = 1

/-- Theorem stating that the given parametric plane has the specified equation -/
theorem parametric_to_equation (p : ParametricPlane) :
  ∃ (eq : PlaneEquation), ∀ (x y z : ℝ),
    (∃ s t, p.v s t = (x, y, z)) ↔ eq.A * x + eq.B * y + eq.C * z + eq.D = 0 ∧
    eq.A = 6 ∧ eq.B = 5 ∧ eq.C = 2 ∧ eq.D = -25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_equation_l608_60832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_union_triangle_rotation_l608_60873

/-- Triangle with side lengths a, b, c -/
structure Triangle (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α] where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Area of a triangle using Heron's formula -/
noncomputable def triangle_area (t : Triangle ℝ) : ℝ :=
  let s := (t.a + t.b + t.c) / 2
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

/-- Theorem: Area of union of triangle and its 180° rotation -/
theorem area_union_triangle_rotation
  (t : Triangle ℝ)
  (h1 : t.a = 15)
  (h2 : t.b = 16)
  (h3 : t.c = 17) :
  triangle_area t = 6 * Real.sqrt 1008 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_union_triangle_rotation_l608_60873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l608_60887

/-- Given a line L1 with equation 2x - 3y + 4 = 0 and a point P (-1, 2),
    prove that the line L2 passing through P and perpendicular to L1
    has the equation 3x + 2y - 1 = 0 -/
theorem perpendicular_line_equation :
  ∃ (L1 L2 : Set (ℝ × ℝ)) (P : ℝ × ℝ),
    (L1 = {(x, y) : ℝ × ℝ | 2 * x - 3 * y + 4 = 0}) ∧
    (P = (-1, 2)) ∧
    (P ∈ L2) ∧ 
    (∀ (x y : ℝ), (x, y) ∈ L2 ↔ 3 * x + 2 * y - 1 = 0) ∧
    (∀ (v w p q : ℝ × ℝ), v ∈ L1 → w ∈ L1 → v ≠ w → 
      p ∈ L2 → q ∈ L2 → p ≠ q →
        (v.1 - w.1) * (p.1 - q.1) + (v.2 - w.2) * (p.2 - q.2) = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l608_60887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_percentage_after_addition_l608_60844

/-- Given a mixture of wine and water, calculate the new percentage of water after adding more water -/
theorem water_percentage_after_addition
  (initial_volume : ℝ)
  (initial_water_percentage : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 150)
  (h2 : initial_water_percentage = 10)
  (h3 : added_water = 30) :
  (initial_volume * (initial_water_percentage / 100) + added_water) / (initial_volume + added_water) * 100 = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_percentage_after_addition_l608_60844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_cubic_meters_to_cubic_feet_l608_60834

/-- Conversion factor from meters to feet -/
def meter_to_feet : ℝ := 3.28084

/-- Volume conversion from cubic meters to cubic feet -/
def cubic_meter_to_cubic_feet (v : ℝ) : ℝ := v * (meter_to_feet ^ 3)

/-- Theorem: 5 cubic meters is approximately 176.5735 cubic feet -/
theorem five_cubic_meters_to_cubic_feet :
  ∃ (x : ℝ), abs (cubic_meter_to_cubic_feet 5 - x) < 0.0001 ∧ x = 176.5735 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_cubic_meters_to_cubic_feet_l608_60834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spider_change_is_eight_l608_60875

/-- Represents the change in creature counts after using magic items. -/
structure CreatureChange where
  fleas : Int
  beetles : Int
  spiders : Int
  bugs : Int

/-- Represents the effects of using a magic nut. -/
def magic_nut_effect (flea_to_beetle : Bool) : CreatureChange :=
  if flea_to_beetle then
    { fleas := -1, beetles := 1, spiders := 0, bugs := 0 }
  else
    { fleas := 0, beetles := 0, spiders := -1, bugs := 1 }

/-- Represents the effects of using a magic acorn. -/
def magic_acorn_effect (flea_to_spider : Bool) : CreatureChange :=
  if flea_to_spider then
    { fleas := -1, beetles := 0, spiders := 1, bugs := 0 }
  else
    { fleas := 0, beetles := -1, spiders := 0, bugs := 1 }

/-- Add two CreatureChange structures. -/
def add_change (a b : CreatureChange) : CreatureChange :=
  { fleas := a.fleas + b.fleas
  , beetles := a.beetles + b.beetles
  , spiders := a.spiders + b.spiders
  , bugs := a.bugs + b.bugs }

/-- Theorem stating that the change in the number of spiders is 8. -/
theorem spider_change_is_eight 
  (total_nuts : Nat) 
  (total_acorns : Nat) 
  (beetle_increase : Int) 
  (h_nuts : total_nuts = 20) 
  (h_acorns : total_acorns = 23) 
  (h_beetles : beetle_increase = 5) : 
  ∃ (nut_flea_to_beetle nut_spider_to_bug acorn_flea_to_spider acorn_beetle_to_bug : Nat), 
    let total_change := 
      (List.replicate nut_flea_to_beetle (magic_nut_effect true)).foldl add_change 
      ((List.replicate nut_spider_to_bug (magic_nut_effect false)).foldl add_change 
      ((List.replicate acorn_flea_to_spider (magic_acorn_effect true)).foldl add_change 
      ((List.replicate acorn_beetle_to_bug (magic_acorn_effect false)).foldl add_change 
        { fleas := 0, beetles := 0, spiders := 0, bugs := 0 })))
    nut_flea_to_beetle + nut_spider_to_bug = total_nuts ∧
    acorn_flea_to_spider + acorn_beetle_to_bug = total_acorns ∧
    total_change.beetles = beetle_increase ∧
    total_change.spiders = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spider_change_is_eight_l608_60875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twin_pairs_implies_a_zero_l608_60899

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 2 else -x^3 + 6*x^2 - 9*x + 2 - a

def is_twin_point_pair (a : ℝ) (x y : ℝ) : Prop :=
  f a x = f a (-y) ∧ f a y = f a (-x) ∧ (x ≠ 0 ∨ y ≠ 0)

def has_exactly_two_twin_pairs (a : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
    is_twin_point_pair a x₁ y₁ ∧ 
    is_twin_point_pair a x₂ y₂ ∧ 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    ∀ x y : ℝ, is_twin_point_pair a x y → 
      ((x = x₁ ∧ y = y₁) ∨ (x = y₁ ∧ y = x₁) ∨ 
       (x = x₂ ∧ y = y₂) ∨ (x = y₂ ∧ y = x₂))

theorem twin_pairs_implies_a_zero :
  ∀ a : ℝ, has_exactly_two_twin_pairs a → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twin_pairs_implies_a_zero_l608_60899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_of_sequence_l608_60889

-- Define the sequence a_n
def a : ℕ → ℚ
  | 0 => 1  -- We need to define a value for 0 to avoid the missing case error
  | 1 => 1
  | n + 1 => 2 * a n + 1 / 2

-- State the theorem
theorem general_term_of_sequence (n : ℕ) (h : n ≥ 1) : 
  a n = 3 * 2^(n-2) - 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_of_sequence_l608_60889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l608_60893

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := 3 * x - Real.log x

-- Define the derivative of the curve
noncomputable def f' (x : ℝ) : ℝ := 3 - 1 / x

-- Define the point of tangency
def point : ℝ × ℝ := (1, 3)

-- Define the slope of the tangent line at x = 1
noncomputable def tangent_slope : ℝ := f' 1

-- Theorem: The equation of the tangent line is 2x - y - 1 = 0
theorem tangent_line_equation :
  ∀ (x y : ℝ), (x - point.1) * tangent_slope = y - point.2 ↔ 2 * x - y - 1 = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l608_60893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_value_l608_60833

def y : ℕ → ℕ
  | 0 => 150  -- Added case for 0
  | 1 => 150
  | (n + 2) => y (n + 1) ^ 2 + 2 * y (n + 1)

noncomputable def series_sum : ℝ := ∑' n, (1 : ℝ) / (y n + 1)

theorem series_sum_value : series_sum = 1 / 151 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_value_l608_60833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l608_60837

/-- Circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

/-- Line equation -/
def line_eq (a b x y : ℝ) : Prop := 2*a*x - b*y + 2 = 0

/-- Symmetric point with respect to the line -/
def symmetric_point (a b x y x' y' : ℝ) : Prop :=
  line_eq a b ((x + x')/2) ((y + y')/2) ∧ (x' - x)*(2*a) = (y' - y)*b

/-- Main theorem -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y x' y' : ℝ, circle_eq x y → symmetric_point a b x y x' y' → circle_eq x' y') →
  (∃ m : ℝ, m = 3 + 2*Real.sqrt 2 ∧ ∀ a' b' : ℝ, a' > 0 → b' > 0 → 1/a' + 2/b' ≥ m) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l608_60837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_area_is_12pi_l608_60854

/-- Represents a tetrahedron P-ABC with a circumscribed sphere -/
structure Tetrahedron where
  R : ℝ  -- Radius of the circumscribed sphere
  AB : ℝ  -- Length of side AB
  AC : ℝ  -- Length of side AC
  volume : ℝ  -- Volume of the tetrahedron

/-- The conditions of the tetrahedron and its circumscribed sphere -/
def tetrahedron_conditions (t : Tetrahedron) : Prop :=
  t.AB = 2 * t.R ∧  -- O lies on AB, so AB is a diameter
  t.AC = Real.sqrt 3 * t.R ∧  -- Given: 2AC = √3 * AB
  t.volume = 3 / 2  -- Given volume

/-- The surface area of the circumscribed sphere -/
noncomputable def sphere_surface_area (t : Tetrahedron) : ℝ :=
  4 * Real.pi * t.R^2

/-- Theorem stating that under the given conditions, the surface area of the sphere is 12π -/
theorem sphere_area_is_12pi (t : Tetrahedron) :
    tetrahedron_conditions t → sphere_surface_area t = 12 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_area_is_12pi_l608_60854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_alone_time_l608_60883

/-- The time it takes for worker A to complete the work alone -/
noncomputable def time_A : ℝ := 4

/-- The time it takes for workers B and C together to complete the work -/
noncomputable def time_BC : ℝ := 3

/-- The time it takes for workers A and C together to complete the work -/
noncomputable def time_AC : ℝ := 2

/-- The work rate of worker A -/
noncomputable def rate_A : ℝ := 1 / time_A

/-- The combined work rate of workers B and C -/
noncomputable def rate_BC : ℝ := 1 / time_BC

/-- The combined work rate of workers A and C -/
noncomputable def rate_AC : ℝ := 1 / time_AC

/-- The work rate of worker B -/
noncomputable def rate_B : ℝ := rate_BC - (rate_AC - rate_A)

theorem b_alone_time (h : rate_B > 0) : 1 / rate_B = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_alone_time_l608_60883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mad_cow_variance_l608_60802

/-- The number of cows in the ranch -/
def n : ℕ := 10

/-- The probability of a cow contracting mad cow disease -/
def p : ℝ := 0.02

/-- The random variable representing the number of cows that contract the disease -/
def ξ : ℕ → ℝ := sorry

/-- The variance of the binomial distribution -/
def D (ξ : ℕ → ℝ) : ℝ := n * p * (1 - p)

theorem mad_cow_variance : D ξ = 0.196 := by
  unfold D n p
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mad_cow_variance_l608_60802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_not_on_minimizing_line_l608_60857

/-- The line y = kx that minimizes the sum of squared distances from given points -/
noncomputable def MinimizingLine (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1}

/-- The sum of squared distances from points to the line -/
noncomputable def SumSquaredDistances (k : ℝ) : ℝ :=
  ((-2*k - 1)^2 + (-3*k + 2)^2 + (-k + 3)^2) / (k^2 + 1)

theorem points_not_on_minimizing_line :
  let A : ℝ × ℝ := (-2, 1)
  let B : ℝ × ℝ := (-3, -2)
  let C : ℝ × ℝ := (-1, -3)
  let k_min : ℝ := 1  -- The value of k that minimizes SumSquaredDistances
  ∀ k : ℝ, SumSquaredDistances k ≥ SumSquaredDistances k_min →
    A ∉ MinimizingLine k_min ∧
    B ∉ MinimizingLine k_min ∧
    C ∉ MinimizingLine k_min :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_not_on_minimizing_line_l608_60857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_small_circles_in_large_circle_l608_60845

theorem max_small_circles_in_large_circle :
  let large_radius : ℝ := 11
  let small_radius : ℝ := 1
  let n : ℕ := 31
  (∀ m : ℕ, m > n → ¬ (∃ (centers : Fin m → ℝ × ℝ),
    (∀ i : Fin m, (centers i).1 ^ 2 + (centers i).2 ^ 2 = (large_radius - small_radius) ^ 2) ∧
    (∀ i j : Fin m, i ≠ j → ((centers i).1 - (centers j).1) ^ 2 + ((centers i).2 - (centers j).2) ^ 2 ≥ (2 * small_radius) ^ 2))) ∧
  (∃ (centers : Fin n → ℝ × ℝ),
    (∀ i : Fin n, (centers i).1 ^ 2 + (centers i).2 ^ 2 = (large_radius - small_radius) ^ 2) ∧
    (∀ i j : Fin n, i ≠ j → ((centers i).1 - (centers j).1) ^ 2 + ((centers i).2 - (centers j).2) ^ 2 ≥ (2 * small_radius) ^ 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_small_circles_in_large_circle_l608_60845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l608_60894

-- Define the curve
noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := a * Real.log (x + 1)

-- Define the derivative of the curve
noncomputable def curve_derivative (a : ℝ) (x : ℝ) : ℝ := a / (x + 1)

theorem tangent_line_slope (a : ℝ) :
  (curve a 0 = 0) →
  (curve_derivative a 0 = 2) →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l608_60894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_on_x_axis_l608_60807

/-- The parabola equation -/
noncomputable def parabola (x : ℝ) (c : ℝ) : ℝ := 2 * x^2 - 10 * x + c

/-- The x-coordinate of the vertex -/
noncomputable def vertex_x : ℝ := 5/2

/-- The y-coordinate of the vertex -/
noncomputable def vertex_y (c : ℝ) : ℝ := parabola vertex_x c

/-- Theorem: The vertex is on the x-axis if and only if c = 25/2 -/
theorem vertex_on_x_axis (c : ℝ) : vertex_y c = 0 ↔ c = 25/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_on_x_axis_l608_60807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_perimeter_l608_60831

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - 4*y^2 = 4

-- Define points A, B, F1, and F2
variable (A B F1 F2 : ℝ × ℝ)

-- Define that F1, A, and B are collinear
def collinear (F1 A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, A = F1 + t • (B - F1) ∨ B = F1 + t • (A - F1)

-- Define that A and B are on the left branch of the hyperbola
def on_left_branch (A B : ℝ × ℝ) : Prop :=
  hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ A.1 < 0 ∧ B.1 < 0

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- The main theorem
theorem hyperbola_triangle_perimeter
  (h1 : collinear F1 A B)
  (h2 : on_left_branch A B)
  (h3 : distance A B = 3) :
  distance A F2 + distance B F2 + distance A B = 14 := by
  sorry

#check hyperbola_triangle_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_perimeter_l608_60831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equality_l608_60835

theorem factorial_equality (n : ℕ) : Nat.factorial 5 * Nat.factorial 3 = Nat.factorial n → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equality_l608_60835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_implies_a_range_l608_60862

open Set Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := abs (log x)

-- Define the function g
noncomputable def g (a x : ℝ) : ℝ := f x - a * x

-- Define the interval (0, e^2)
def interval : Set ℝ := Ioo 0 (exp 2)

-- Theorem statement
theorem three_zeros_implies_a_range (a : ℝ) :
  (∃ x y z, x ∈ interval ∧ y ∈ interval ∧ z ∈ interval ∧ 
   x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
   g a x = 0 ∧ g a y = 0 ∧ g a z = 0) →
  a ∈ Ioo (2 / (exp 2)) (1 / exp 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_implies_a_range_l608_60862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_f_monotone_decreasing_on_unit_to_inf_l608_60869

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 - 1)

-- Theorem for the domain of f
theorem domain_of_f :
  {x : ℝ | f x ≠ 0} = {x : ℝ | x ≠ 1 ∧ x ≠ -1} :=
by sorry

-- Theorem for the monotonicity of f on (1, +∞)
theorem f_monotone_decreasing_on_unit_to_inf :
  ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f x₁ > f x₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_f_monotone_decreasing_on_unit_to_inf_l608_60869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gideon_marbles_period_l608_60810

theorem gideon_marbles_period (marbles : ℕ) (period : ℕ) : marbles = period →
  (2 * (marbles / 4 : ℚ)).floor = 45 + 5 →
  period = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gideon_marbles_period_l608_60810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_relations_l608_60812

-- Define the basic types
variable (α β : Type*) -- Planes
variable (l m : Type*) -- Lines

-- Define the relations
def perpendicular (a b : Type*) : Prop := sorry
def parallel (a b : Type*) : Prop := sorry
def contains (a b : Type*) : Prop := sorry

-- State the theorem
theorem perpendicular_parallel_relations 
  (α β l m : Type*)
  (h1 : perpendicular l α) 
  (h2 : contains β m) :
  (parallel α β → perpendicular l m) ∧ 
  (parallel l m → perpendicular α β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_relations_l608_60812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_between_circles_l608_60879

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles :
  ∃ d : ℝ, d = Real.sqrt 130 - (1 + Real.sqrt 13) ∧
    ∀ p1 p2 : ℝ × ℝ,
      (p1.1^2 - 6*p1.1 + p1.2^2 - 8*p1.2 + 16 = 0) →
      (p2.1^2 + 8*p2.1 + p2.2^2 + 10*p2.2 + 28 = 0) →
      d ≤ Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_between_circles_l608_60879
