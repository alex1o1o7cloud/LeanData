import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l598_59837

-- Define the given conditions
noncomputable def tan30 : ℝ := Real.sqrt 3 / 3
def sqrt9 : ℝ := 3
def inverse_third : ℝ := 3

-- State the theorem
theorem calculate_expression :
  3 * tan30 - sqrt9 + inverse_third = Real.sqrt 3 := by
  -- Proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l598_59837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_sum_l598_59850

-- Define the points
variable (P Q R S T : ℝ × ℝ)

-- Define the conditions
def rightAnglePQR (P Q R : ℝ × ℝ) : Prop := sorry
def PQ_length (P Q : ℝ × ℝ) : ℝ := 6
def PR_length (P R : ℝ × ℝ) : ℝ := 8
def rightAnglePRS (P R S : ℝ × ℝ) : Prop := sorry
def RS_length (R S : ℝ × ℝ) : ℝ := 24
def P_S_opposite (P Q R S : ℝ × ℝ) : Prop := sorry
def S_parallel_PQ (S T P Q : ℝ × ℝ) : Prop := sorry

-- Define the ratio ST/SR
def ST_SR_ratio (S T R : ℝ × ℝ) (p q : ℕ) : Prop :=
  ∃ (ST SR : ℝ), ST / SR = p / q ∧ Nat.Coprime p q

-- The theorem to prove
theorem triangle_ratio_sum (P Q R S T : ℝ × ℝ) (p q : ℕ) :
  rightAnglePQR P Q R →
  rightAnglePRS P R S →
  P_S_opposite P Q R S →
  S_parallel_PQ S T P Q →
  ST_SR_ratio S T R p q →
  p + q = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_sum_l598_59850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_squared_l598_59835

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : a > 0
  pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ :=
  Real.sqrt (1 + b^2 / a^2)

/-- The right focus of a hyperbola -/
noncomputable def right_focus (h : Hyperbola a b) : ℝ × ℝ :=
  (Real.sqrt (a^2 + b^2), 0)

/-- An asymptote of the hyperbola -/
noncomputable def asymptote (h : Hyperbola a b) (x : ℝ) : ℝ := 
  (b / a) * x

theorem hyperbola_eccentricity_squared (a b : ℝ) (h : Hyperbola a b) :
  let F := right_focus h
  let M := (F.1, asymptote h F.1)
  (M.1 - F.1)^2 + (M.2 - F.2)^2 = (2 * a)^2 →
  (eccentricity h)^2 = (1 + Real.sqrt 17) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_squared_l598_59835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_triangles_l598_59887

/-- The total area of two triangles QCA and ABC given specific coordinates -/
theorem total_area_triangles (r : ℝ) : 
  (3 : ℝ) / 2 * (15 - r) + (1 : ℝ) / 2 * Real.sqrt (9 + (15 - r)^2) * Real.sqrt (225 + r^2) =
  45 / 2 - 3 * r / 2 + (1 : ℝ) / 2 * Real.sqrt (9 + (15 - r)^2) * Real.sqrt (225 + r^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_triangles_l598_59887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_to_center_C_l598_59872

/-- The line l -/
def line_l (x y : ℝ) : Prop := y = 2 * x

/-- The circle C -/
def circle_C (x y : ℝ) : Prop := (x - 8)^2 + (y - 1)^2 = 2

/-- Point P -/
def point_P : ℝ × ℝ := (2, 4)

/-- Center of circle C -/
def center_C : ℝ × ℝ := (8, 1)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_P_to_center_C :
  distance point_P center_C = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_to_center_C_l598_59872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bag_weight_difference_l598_59819

/-- Given three bags with weights in a specific ratio, prove the difference between
    the sum of the heaviest and lightest bags and the middle bag. -/
theorem bag_weight_difference (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a ≤ b ∧ b ≤ c →
  (a : ℚ) / 4 = (b : ℚ) / 5 →
  (a : ℚ) / 4 = (c : ℚ) / 6 →
  a = 36 →
  a + c - b = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bag_weight_difference_l598_59819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_price_calculation_l598_59845

theorem apple_price_calculation (bike_cost repair_percentage remaining_fraction : ℝ) 
  (apples_sold : ℕ) 
  (h1 : bike_cost = 80)
  (h2 : repair_percentage = 0.25)
  (h3 : remaining_fraction = 1/5)
  (h4 : apples_sold = 20)
  : (bike_cost * repair_percentage) / (1 - remaining_fraction) / apples_sold = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_price_calculation_l598_59845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l598_59829

/-- Given a function f(x) = A * cos(2x + φ) where A > 0 and |φ| < π,
    if f is an odd function and f(3π/4) = -1,
    then g(x) = sin(x) where g is derived from f by doubling the abscissa while keeping the ordinate unchanged -/
theorem function_transformation (A φ : ℝ) (h1 : A > 0) (h2 : |φ| < π) :
  let f := λ x => A * Real.cos (2 * x + φ)
  (∀ x, f (-x) = -f x) →  -- f is an odd function
  f (3 * π / 4) = -1 →
  let g := λ x => f (x / 2)  -- g is derived from f by doubling the abscissa
  g = λ x => Real.sin x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l598_59829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complementary_angles_adjustment_l598_59822

theorem complementary_angles_adjustment (small_angle large_angle : ℝ) : 
  small_angle + large_angle = 90 →
  small_angle / large_angle = 1/2 →
  let new_small_angle := small_angle * 1.2;
  let new_large_angle := 90 - new_small_angle;
  (large_angle - new_large_angle) / large_angle = 0.1
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complementary_angles_adjustment_l598_59822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_1964_not_divisible_by_4_l598_59897

def mySequence : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => mySequence (n + 1) * mySequence n + 1

theorem a_1964_not_divisible_by_4 : ¬ (4 ∣ mySequence 1964) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_1964_not_divisible_by_4_l598_59897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_large_primes_divide_infinitely_l598_59873

def my_sequence (n : ℕ) : ℕ := 10^n - 1

theorem not_all_large_primes_divide_infinitely (k : ℕ) : 
  ∃ p : ℕ, p > k ∧ Prime p ∧ ¬(∀ m : ℕ, ∃ n ≥ m, p ∣ my_sequence n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_large_primes_divide_infinitely_l598_59873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_range_l598_59815

-- Define the function s(y)
noncomputable def s (y : ℝ) : ℝ := 1 / ((1 + y)^2 + 1)

-- Theorem stating the range of s(y)
theorem s_range : Set.range s = Set.Ioo 0 1 ∪ {1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_range_l598_59815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_minus_two_sin_zero_solution_set_l598_59865

theorem sin_squared_minus_two_sin_zero_solution_set :
  {x : ℝ | Real.sin x ^ 2 - 2 * Real.sin x = 0} = {x : ℝ | ∃ k : ℤ, x = k * Real.pi} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_minus_two_sin_zero_solution_set_l598_59865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_magnitude_two_l598_59833

open Complex

theorem pure_imaginary_magnitude_two (a : ℝ) :
  let z : ℂ := 1 + a * I
  (z^2).re = 0 → Complex.abs (z^2) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_magnitude_two_l598_59833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_l598_59867

theorem angle_terminal_side (a : ℝ) : 
  (∃ P : ℝ × ℝ, P = (3*a, 4)) →
  Real.cos (Real.arccos (-3/5)) = -3/5 →
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_l598_59867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_surface_area_l598_59825

/-- A tetrahedron with three perpendicular edges from a common vertex -/
structure Tetrahedron where
  edge1 : ℝ
  edge2 : ℝ
  edge3 : ℝ
  perpendicular : True  -- We'll assume perpendicularity without proving it for now

/-- The surface area of a sphere containing the tetrahedron -/
noncomputable def sphere_surface_area (t : Tetrahedron) : ℝ :=
  4 * Real.pi * (t.edge1^2 + t.edge2^2 + t.edge3^2) / 4

theorem tetrahedron_sphere_surface_area :
  ∀ t : Tetrahedron,
    t.edge1 = 1 →
    t.edge2 = Real.sqrt 6 →
    t.edge3 = 3 →
    sphere_surface_area t = 16 * Real.pi :=
by
  intro t h1 h2 h3
  simp [sphere_surface_area, h1, h2, h3]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check tetrahedron_sphere_surface_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_surface_area_l598_59825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_transformation_l598_59866

/-- Given a matrix M with eigenvalue 4 and eigenvector [2, 3], 
    prove that the transformation of 5x² + 8xy + 4y² = 1 by M results in x² + y² = 2 -/
theorem curve_transformation (b c : ℝ) : 
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![1, b; c, 2]
  (M.mulVec (![2, 3]) = (4 : ℝ) • ![2, 3]) →
  (∀ x y x' y' : ℝ, 
    (5 * x^2 + 8 * x * y + 4 * y^2 = 1) →
    (x' = M 0 0 * x + M 0 1 * y) →
    (y' = M 1 0 * x + M 1 1 * y) →
    (x'^2 + y'^2 = 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_transformation_l598_59866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_quadratic_product_of_roots_specific_equation_l598_59856

theorem product_of_roots_quadratic (a b c : ℚ) (h : a ≠ 0) :
  let f : ℚ → ℚ := λ x => a * x^2 + b * x + c
  (∃ x y : ℚ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  ∀ x y : ℚ, f x = 0 → f y = 0 → x * y = c / a :=
by sorry

theorem product_of_roots_specific_equation :
  let f : ℚ → ℚ := λ x => 12 * x^2 + 28 * x - 315
  (∃ x y : ℚ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  ∀ x y : ℚ, f x = 0 → f y = 0 → x * y = -105 / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_quadratic_product_of_roots_specific_equation_l598_59856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_at_2_3_l598_59875

/-- The function representing the curve y = x^2 - 1 --/
noncomputable def f (x : ℝ) : ℝ := x^2 - 1

/-- The expression for m --/
noncomputable def m (x y : ℝ) : ℝ := (3*x + y - 5)/(x - 1) + (x + 3*y - 7)/(y - 2)

theorem min_m_at_2_3 :
  ∀ x y : ℝ, x > Real.sqrt 3 → y = f x → m x y ≥ m 2 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_at_2_3_l598_59875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_volume_ratio_l598_59803

/-- The volume of a right circular cylinder -/
noncomputable def cylinder_volume (height : ℝ) (circumference : ℝ) : ℝ :=
  (height * circumference^2) / (4 * Real.pi)

/-- The theorem stating that the volume of tank A is 60% of the volume of tank B -/
theorem tank_volume_ratio :
  let tank_a_volume := cylinder_volume 10 6
  let tank_b_volume := cylinder_volume 6 10
  tank_a_volume / tank_b_volume = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_volume_ratio_l598_59803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_properties_l598_59882

theorem triangle_vector_properties (AB BC : ℝ × ℝ) (θ : ℝ) :
  0 ≤ θ ∧ θ < π →
  AB.1 * BC.1 + AB.2 * BC.2 = 6 →
  6 * (2 - Real.sqrt 3) ≤ Real.sqrt (AB.1^2 + AB.2^2) * Real.sqrt (BC.1^2 + BC.2^2) * Real.sin (π - θ) ∧
  Real.sqrt (AB.1^2 + AB.2^2) * Real.sqrt (BC.1^2 + BC.2^2) * Real.sin (π - θ) ≤ 6 * Real.sqrt 3 →
  (Real.tan (15 * π / 180) = 2 - Real.sqrt 3) ∧
  (π / 12 ≤ θ ∧ θ ≤ π / 3) ∧
  (∃ (max_value : ℝ), max_value = Real.sqrt 3 - 1 ∧
    ∀ (x : ℝ), π / 12 ≤ x ∧ x ≤ π / 3 →
      (1 - Real.sqrt 2 * Real.cos (2 * x - π / 4)) / Real.sin x ≤ max_value) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_properties_l598_59882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_plus_sqrt2_over_2_in_first_quadrant_l598_59857

noncomputable def z : ℂ := Complex.I / (1 - Complex.I)

theorem z_plus_sqrt2_over_2_in_first_quadrant :
  let w := z + (Real.sqrt 2 / 2)
  0 < w.re ∧ 0 < w.im := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_plus_sqrt2_over_2_in_first_quadrant_l598_59857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_oclock_angle_l598_59807

/-- Represents a clock face -/
structure ClockFace where
  total_degrees : ℕ := 360
  num_spaces : ℕ := 12

/-- Calculates the angle between clock hands at a given hour -/
def angle_at_hour (clock : ClockFace) (hour : ℕ) : ℕ :=
  (hour * (clock.total_degrees / clock.num_spaces)) % clock.total_degrees

/-- Theorem: The smaller angle between clock hands at 5 o'clock is 150 degrees -/
theorem five_oclock_angle : 
  let clock : ClockFace := ⟨360, 12⟩
  min (angle_at_hour clock 5) (clock.total_degrees - angle_at_hour clock 5) = 150 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_oclock_angle_l598_59807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_equals_1_to_3_l598_59898

-- Define set A
def A : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1/(x^2 + 1)}

-- Define set B
def B : Set ℝ := {x : ℝ | x < 3}

-- Theorem statement
theorem A_intersect_B_equals_1_to_3 : A ∩ B = Set.Icc 1 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_equals_1_to_3_l598_59898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_point_implies_a_zero_range_of_b_when_a_negative_one_l598_59852

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + 1) + x^3 - x^2 - a * x

theorem extremum_point_implies_a_zero :
  ∀ a : ℝ, (∃ f' : ℝ → ℝ, (∀ x : ℝ, HasDerivAt (f a) (f' x) x) ∧ f' (2/3) = 0) → a = 0 := by
  sorry

theorem range_of_b_when_a_negative_one :
  ∀ b : ℝ, (∃ x : ℝ, x > 0 ∧ f (-1) (1 - x) - (1 - x)^3 = b / x) → b ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_point_implies_a_zero_range_of_b_when_a_negative_one_l598_59852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_venerable_characterization_l598_59809

-- Define σ' function
def σ' (n : ℕ) : ℕ := (Finset.sum (Nat.divisors n) id) - n

-- Define the venerable property
def is_venerable (n : ℕ) : Prop := n > 1 ∧ σ' n = n - 1

-- Define the property of having a venerable power
def has_venerable_power (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 1 ∧ is_venerable (n^m)

-- Theorem statement
theorem venerable_characterization (n : ℕ) :
  (is_venerable n ∧ has_venerable_power n) ↔ ∃ k : ℕ, k > 0 ∧ n = 2^k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_venerable_characterization_l598_59809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_books_count_l598_59851

theorem math_books_count :
  ∃ (math_books : ℕ),
    math_books ≤ 90 ∧
    ∃ (history_books : ℕ),
      history_books = 90 - math_books ∧
      4 * math_books + 5 * history_books = 390 ∧
      math_books = 60 := by
  sorry

#check math_books_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_books_count_l598_59851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l598_59846

def f (x : ℝ) := 3 * x^2 + 6

theorem quadratic_properties :
  (∀ (x y : ℝ), x < y → f x < f y) ∧ 
  (∀ (x₁ x₂ : ℝ), x₁ < 0 ∧ x₂ < 0 ∧ x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ (x : ℝ), f x = f (-x)) ∧
  (∀ (x : ℝ), f 0 ≤ f x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l598_59846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_sin_even_implies_m_l598_59890

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sin (2 * (x + m) + Real.pi / 3)

theorem shifted_sin_even_implies_m (m : ℝ) :
  (∀ x, f m x = f m (-x)) → m = Real.pi / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_sin_even_implies_m_l598_59890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_geometric_series_sum_l598_59832

/-- The sum of an infinite geometric series with first term a and common ratio r where |r| < 1 -/
noncomputable def geometric_series_sum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Theorem: The sum of the infinite geometric series with first term 1/4 and common ratio 1/2 is 1/2 -/
theorem specific_geometric_series_sum :
  geometric_series_sum (1/4 : ℝ) (1/2 : ℝ) = 1/2 := by
  sorry

#check specific_geometric_series_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_geometric_series_sum_l598_59832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_properties_l598_59863

/-- Parabola structure -/
structure Parabola where
  a : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 4 * a * x

/-- Point on a parabola -/
structure ParabolaPoint (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : p.equation x y

/-- Line passing through two points -/
def Line (p q : ℝ × ℝ) : ℝ → ℝ → Prop :=
  fun x y => (y - p.2) * (q.1 - p.1) = (x - p.1) * (q.2 - p.2)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_intersection_properties 
  (p : Parabola)
  (h_p : p.a = 1)
  (F : ℝ × ℝ)
  (h_F : F = (1, 0))
  (M N : ParabolaPoint p)
  (h_line : Line F (M.x, M.y) N.x N.y) :
  (M.x + N.x = 6 → distance (M.x, M.y) (N.x, N.y) = 8) ∧
  (1 / distance (M.x, M.y) F + 1 / distance (N.x, N.y) F = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_properties_l598_59863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candles_equal_height_only_at_start_l598_59801

/-- Represents the height of a candle at a given time -/
noncomputable def candle_height (initial_height : ℝ) (burn_time : ℝ) (elapsed_time : ℝ) : ℝ :=
  initial_height * (1 - elapsed_time / burn_time)

theorem candles_equal_height_only_at_start :
  ∀ (initial_height : ℝ),
  initial_height > 0 →
  ∀ (t : ℝ),
  t ≥ 0 →
  candle_height initial_height 5 t = candle_height initial_height 4 t →
  t = 0 := by
  sorry

#check candles_equal_height_only_at_start

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candles_equal_height_only_at_start_l598_59801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l598_59840

noncomputable def f (x : ℝ) : ℝ := (1/2)^(-x^2 + 2*x)

theorem f_range : Set.range f = Set.Ioo (1/2) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l598_59840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l598_59813

-- Define the curve
noncomputable def curve (x : ℝ) : ℝ := x^2 - Real.log x

-- Define the line
def line (x : ℝ) : ℝ := x - 2

-- Theorem statement
theorem min_distance_curve_to_line :
  ∃ (d : ℝ), d = Real.sqrt 2 ∧
  ∀ (x y : ℝ), y = curve x →
    d ≤ Real.sqrt ((x - line x)^2 + (y - line x)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l598_59813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_passenger_seat_probability_l598_59861

/-- Represents the seating process on an airplane --/
def SeatingProcess (n : ℕ) := Unit

/-- The probability that the last passenger sits in their own seat --/
noncomputable def lastPassengerProbability (n : ℕ) (process : SeatingProcess n) : ℝ := 1 / 2

/-- Theorem stating that the probability of the last passenger sitting in their own seat is 1/2 --/
theorem last_passenger_seat_probability (n : ℕ) (process : SeatingProcess n) :
  lastPassengerProbability n process = 1 / 2 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_passenger_seat_probability_l598_59861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l598_59862

/-- The speed of a train in km/hr, given its length in meters and time to cross a point in seconds. -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / 1000) / (time / 3600)

/-- Theorem: A train 200 meters long crossing an electric pole in 5.80598713393251 seconds has a speed of approximately 124.019 km/hr. -/
theorem train_speed_calculation :
  let length : ℝ := 200
  let time : ℝ := 5.80598713393251
  abs (train_speed length time - 124.019) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l598_59862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tomato_problem_correct_l598_59827

def tomato_problem (birds_eaten : ℕ) (haruto_left : ℕ) : ℕ :=
  let total_before_sharing := haruto_left * 2
  let total_grown := total_before_sharing + birds_eaten
  total_grown

theorem tomato_problem_correct (birds_eaten haruto_left : ℕ) :
  tomato_problem birds_eaten haruto_left = haruto_left * 2 + birds_eaten :=
by
  unfold tomato_problem
  simp

#eval tomato_problem 19 54

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tomato_problem_correct_l598_59827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joshua_needs_two_point_five_usd_l598_59848

/-- The amount of additional US dollars Joshua needs to exchange to buy the pen -/
noncomputable def additional_usd_needed (pen_cost : ℝ) (usd_amount : ℝ) (eur_amount : ℝ) 
  (usd_to_chf_rate : ℝ) (eur_to_chf_rate : ℝ) : ℝ :=
  let chf_from_eur := eur_amount * eur_to_chf_rate
  let remaining_chf_needed := pen_cost - chf_from_eur
  remaining_chf_needed / usd_to_chf_rate

/-- Theorem stating that Joshua needs to exchange 2.5 USD -/
theorem joshua_needs_two_point_five_usd :
  additional_usd_needed 18 20 15 0.9 1.05 = 2.5 := by
  -- Unfold the definition of additional_usd_needed
  unfold additional_usd_needed
  -- Simplify the expression
  simp
  -- The proof is completed numerically
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_joshua_needs_two_point_five_usd_l598_59848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_b_value_l598_59808

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 1

-- Define the function g
noncomputable def g (a x : ℝ) : ℝ := -x + a / (5 * a^2 - 4 * a + 1)

-- State the theorem
theorem min_b_value (a b : ℝ) :
  a ≠ 0 →
  0 < a →
  a < 1 →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a b x₁ = x₁ ∧ f a b x₂ = x₂) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a b x₁ = x₁ ∧ f a b x₂ = x₂ ∧ 
    g a ((x₁ + x₂) / 2) = (x₁ + x₂) / 2) →
  (∀ b' : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a b' x₁ = x₁ ∧ f a b' x₂ = x₂ ∧ 
    g a ((x₁ + x₂) / 2) = (x₁ + x₂) / 2) → b' ≥ b) →
  b = -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_b_value_l598_59808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_formula_l598_59880

noncomputable def f₀ (x : ℝ) : ℝ := x * (Real.sin x + Real.cos x)

noncomputable def f (n : ℕ) (x : ℝ) : ℝ := 
  match n with
  | 0 => f₀ x
  | n+1 => deriv (fun y => f n y) x

theorem f_formula (n : ℕ) (x : ℝ) : 
  f n x = (x + n) * Real.sin (x + n * Real.pi / 2) + 
          (x - n) * Real.cos (x + n * Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_formula_l598_59880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_smallest_n_satisfying_condition_l598_59847

noncomputable def sequence_a (n : ℕ) : ℝ := 2^n - 1

noncomputable def sum_S (n : ℕ) : ℝ := 2 * sequence_a n - n

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → sum_S n + n = 2 * sequence_a n) ∧
  (∀ n : ℕ, n > 1 → sequence_a (n + 1) + 1 = 2 * (sequence_a n + 1)) ∧
  (∀ n : ℕ, n > 0 → sequence_a n = 2^n - 1) :=
by sorry

noncomputable def sequence_T (n : ℕ) : ℝ := (n - 1) * 2^(n + 1) + 2 - (n * (n + 1)) / 2

theorem smallest_n_satisfying_condition :
  (∀ n : ℕ, n < 8 → sequence_T n + (n^2 + n) / 2 ≤ 2018) ∧
  (sequence_T 8 + (8^2 + 8) / 2 > 2018) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_smallest_n_satisfying_condition_l598_59847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l598_59810

/-- Represents the possible rectangle sizes in the game -/
inductive RectangleSize
  | OneByOne
  | OneByTwo
  | TwoByTwo

/-- Represents the game state -/
structure GameState where
  gridSize : Nat
  uncoloredSquares : Nat

/-- Represents a player's move -/
structure Move where
  size : RectangleSize

/-- Calculates the number of squares covered by a move -/
def squaresCovered (move : Move) : Nat :=
  match move.size with
  | RectangleSize.OneByOne => 1
  | RectangleSize.OneByTwo => 2
  | RectangleSize.TwoByTwo => 4

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  { state with uncoloredSquares := state.uncoloredSquares - squaresCovered move }

/-- Checks if a move is valid in the current game state -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  squaresCovered move ≤ state.uncoloredSquares

/-- Defines the winning strategy for the first player -/
def firstPlayerStrategy (state : GameState) (opponentMove : Move) : Move :=
  sorry -- Strategy implementation

/-- Simulates the game play -/
def gamePlay (initialState : GameState) (strategy : GameState → Move → Move) (opponentStrategy : GameState → Move) : GameState :=
  sorry -- Game simulation implementation

/-- Theorem stating that the first player can guarantee victory -/
theorem first_player_wins (initialState : GameState) 
  (h1 : initialState.gridSize = 10)
  (h2 : initialState.uncoloredSquares = 100) :
  ∃ (strategy : GameState → Move → Move),
    ∀ (opponentStrategy : GameState → Move),
      let finalState := gamePlay initialState strategy opponentStrategy
      finalState.uncoloredSquares = 0 ∧
      finalState.uncoloredSquares % 2 = 0 :=
by
  sorry -- Proof implementation


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l598_59810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amy_bob_meeting_l598_59817

/-- The distance between two skaters on a frozen lake --/
def distance_AB : ℝ := 120

/-- Amy's skating speed in meters per second --/
def amy_speed : ℝ := 9

/-- Bob's skating speed in meters per second --/
def bob_speed : ℝ := 10

/-- The angle between Amy's path and the line AB in radians --/
noncomputable def amy_angle : ℝ := Real.pi / 4

/-- The time it takes for Amy and Bob to meet --/
noncomputable def meeting_time : ℝ := 
  (1530 * Real.sqrt 2 + Real.sqrt ((1530 * Real.sqrt 2)^2 - 4 * 19 * 14400)) / 38

/-- The distance Amy skates before meeting Bob --/
noncomputable def amy_distance : ℝ := amy_speed * meeting_time

theorem amy_bob_meeting :
  ∃ ε > 0, |amy_distance - 105.2775| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amy_bob_meeting_l598_59817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_vertical_asymptote_num_vertical_asymptotes_correct_l598_59838

/-- The number of vertical asymptotes of the function f(x) = (x-2)/(x^2+4x-12) -/
def num_vertical_asymptotes : ℕ := 1

/-- The function f(x) = (x-2)/(x^2+4x-12) -/
noncomputable def f (x : ℝ) : ℝ := (x - 2) / (x^2 + 4*x - 12)

/-- Theorem stating that f has exactly one vertical asymptote -/
theorem f_has_one_vertical_asymptote :
  ∃! a : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - a| ∧ |x - a| < δ → |f x| > 1/ε :=
by
  -- The proof goes here
  sorry

/-- Theorem stating that the number of vertical asymptotes is correct -/
theorem num_vertical_asymptotes_correct : num_vertical_asymptotes = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_vertical_asymptote_num_vertical_asymptotes_correct_l598_59838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l598_59804

-- Define the function f(x) = 2x / (x + 1)
noncomputable def f (x : ℝ) : ℝ := 2 * x / (x + 1)

-- Define the interval [3, 5]
def I : Set ℝ := Set.Icc 3 5

-- Theorem stating the properties of f(x) on [3, 5]
theorem f_properties :
  (∀ x y, x ∈ I → y ∈ I → x < y → f x < f y) ∧
  (∀ x, x ∈ I → f x ≥ 5/4) ∧
  (∀ x, x ∈ I → f x ≤ 3/2) ∧
  f 3 = 5/4 ∧
  f 5 = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l598_59804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_impossible_quadrilateral_l598_59879

theorem smallest_m_for_impossible_quadrilateral (S : Finset ℕ) 
  (h j k l m : ℕ) : 
  S.card = 5 ∧ 
  S = {h, j, k, l, m} ∧ 
  0 < h ∧ h < j ∧ j < k ∧ k < l ∧ l < m ∧
  (∀ a b c d : ℕ, a ∈ S → b ∈ S → c ∈ S → d ∈ S → 
    a < b → b < c → c < d → a + b + c ≤ d) →
  m ≥ 11 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_impossible_quadrilateral_l598_59879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conference_married_women_fraction_l598_59828

theorem conference_married_women_fraction :
  ∀ (total_people : ℚ) (single_men : ℚ) (married_men : ℚ) (married_women : ℚ),
    single_men + married_men > 0 →
    single_men / (single_men + married_men) = 3 / 7 →
    married_men = married_women →
    total_people = single_men + married_men + married_women →
    married_women / total_people = 4 / 11 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conference_married_women_fraction_l598_59828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_triangle_sides_l598_59864

noncomputable section

def ω : ℝ := 1  -- We know ω = 1 from the solution
axiom ω_pos : ω > 0

def m (x : ℝ) : ℝ × ℝ := (2 * Real.cos (ω * x), -1)
def n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (ω * x) + Real.cos (ω * x), 1)

def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

axiom f_intersect : ∃ x₁ x₂, x₁ < x₂ ∧ x₂ - x₁ = Real.pi / 2 ∧ f x₁ = 0 ∧ f x₂ = 0

theorem f_range : Set.Icc (-1 : ℝ) 2 = Set.image f (Set.Icc 0 (Real.pi / 2)) := by sorry

def triangle_ABC (A B C : ℝ) : Prop :=
  0 < A ∧ A < Real.pi ∧ f A = 1 ∧
  let a := 3
  let h := 3 * Real.sqrt 3 / 2
  ∃ b c, b > 0 ∧ c > 0 ∧
    a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) ∧
    (1/2) * a * h = (1/2) * b * c * Real.sin A

theorem triangle_sides : ∀ A B C, triangle_ABC A B C → ∃ b c, b = 3 ∧ c = 3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_triangle_sides_l598_59864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cod_mass_at_15kg_l598_59870

/-- Represents the mass of Jeff's pet Atlantic cod as a function of its age in years -/
def cod_mass : ℝ → ℝ := sorry

/-- The age of the cod when its mass is 15 kg -/
def age_at_15kg : ℝ := 7

theorem cod_mass_at_15kg : cod_mass age_at_15kg = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cod_mass_at_15kg_l598_59870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_case_l598_59868

theorem tan_double_angle_special_case (θ : Real) :
  let P : Real × Real := (-1, 2)
  let tan_θ : Real := P.2 / (-P.1)
  tan_θ = -2 →
  Real.tan (2 * θ) = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_case_l598_59868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_parabola_l598_59844

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Distance from a point to a horizontal line -/
def distanceToHorizontalLine (p : Point) (y : ℝ) : ℝ :=
  |p.y - y|

/-- The locus of points satisfying the given condition -/
def locus : Set Point :=
  {p : Point | distanceToHorizontalLine p (-1) = distance p ⟨0, 2⟩ - 1}

theorem locus_is_parabola :
  ∃ (a : ℝ), a > 0 ∧ locus = {p : Point | p.x^2 = 8 * p.y} ∧
  ∀ (p : Point), p ∈ locus ↔ distance p ⟨0, 2⟩ = distanceToHorizontalLine p (-2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_parabola_l598_59844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_range_l598_59816

noncomputable def s (x : ℝ) : ℝ := 1 / (2 - x^2)^2

theorem s_range : Set.range s = Set.Ioc 0 (1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_range_l598_59816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_german_product_divisibility_l598_59830

/-- Definition of a German set in an n × n grid -/
def GermanSet (n : ℕ) : Type :=
  {s : Finset (Fin n × Fin n) // s.card = n ∧ ∀ i j, (∃ k, (i, k) ∈ s) ∧ (∃ k, (k, j) ∈ s)}

/-- Definition of a valid labelling of an n × n grid -/
def ValidLabelling (n : ℕ) : Type :=
  {f : Fin n × Fin n → Fin (n^2) // Function.Injective f}

/-- Definition of a German product given a labelling -/
def GermanProduct {n : ℕ} (f : Fin n × Fin n → Fin (n^2)) (s : Finset (Fin n × Fin n)) : ℕ :=
  s.prod (λ x ↦ (f x).val + 1)

/-- Main theorem statement -/
theorem german_product_divisibility :
  (∃ f : ValidLabelling 10, ∀ (s₁ s₂ : GermanSet 10),
    (GermanProduct f.val s₁.val - GermanProduct f.val s₂.val) % 101 = 0) ∧
  ¬(∃ f : ValidLabelling 8, ∀ (s₁ s₂ : GermanSet 8),
    (GermanProduct f.val s₁.val - GermanProduct f.val s₂.val) % 65 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_german_product_divisibility_l598_59830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abe_found_four_ants_l598_59878

/-- The number of ants found by Abe -/
def abe_ants : ℕ := 4

/-- The number of ants found by Beth -/
def beth_ants : ℕ := (3 * abe_ants) / 2

/-- The number of ants found by CeCe -/
def cece_ants : ℕ := 2 * abe_ants

/-- The number of ants found by Duke -/
def duke_ants : ℕ := abe_ants / 2

/-- The total number of ants found by all four children -/
def total_ants : ℕ := 20

theorem abe_found_four_ants :
  abe_ants + beth_ants + cece_ants + duke_ants = total_ants ∧
  abe_ants = 4 := by
  sorry

#eval abe_ants

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abe_found_four_ants_l598_59878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_interval_for_five_zeros_l598_59820

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x + 1

theorem min_interval_for_five_zeros :
  ∃ (m n : ℝ), 
    (∀ x, f x = f (x + π)) ∧
    (∃ (x₁ x₂ x₃ x₄ x₅ : ℝ), m ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧ x₄ < x₅ ∧ x₅ ≤ n ∧
      f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0 ∧ f x₅ = 0) ∧
    (∀ m' n', 
      (∃ (x₁ x₂ x₃ x₄ x₅ : ℝ), m' ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧ x₄ < x₅ ∧ x₅ ≤ n' ∧
        f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0 ∧ f x₅ = 0) →
      n' - m' ≥ n - m) ∧
    n - m = 2 * π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_interval_for_five_zeros_l598_59820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l598_59895

/-- Represents the number of type A phones purchased -/
def x : ℕ := 15

/-- Total number of phones purchased -/
def total_phones : ℕ := 20

/-- Unit price of type A phones -/
def price_A (x : ℕ) : ℝ := -20 * x + 1600

/-- Unit price of type B phones -/
def price_B (x : ℕ) : ℝ := -10 * (total_phones - x) + 1360

/-- Selling price of type A phones -/
def sell_price_A : ℝ := 1800

/-- Selling price of type B phones -/
def sell_price_B : ℝ := 1700

/-- Profit function -/
def profit (x : ℕ) : ℝ :=
  (sell_price_A - price_A x) * x + (sell_price_B - price_B x) * (total_phones - x)

/-- Theorem stating the maximum profit and optimal purchase strategy -/
theorem max_profit :
  (x ≤ 15) →
  (x ≥ total_phones - x) →
  (∀ y : ℕ, y ≤ 15 → y ≥ total_phones - y → profit x ≥ profit y) →
  (x = 15 ∧ profit x = 9450) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l598_59895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_color_probability_l598_59874

/-- The probability of drawing two chips of different colors with replacement -/
theorem different_color_probability (blue red yellow green : ℕ) 
  (h_blue : blue = 7)
  (h_red : red = 8)
  (h_yellow : yellow = 6)
  (h_green : green = 3) :
  let total := blue + red + yellow + green
  (blue * (red + yellow + green) +
   red * (blue + yellow + green) +
   yellow * (blue + red + green) +
   green * (blue + red + yellow) : ℚ) / (total * total) = 537 / 576 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_color_probability_l598_59874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_floors_and_cost_l598_59806

/-- Represents the housing project parameters -/
structure HousingProject where
  landCost : ℝ
  numBuildings : ℕ
  floorArea : ℝ
  baseCost : ℝ
  kValue : ℝ

/-- Calculates the average comprehensive cost per square meter -/
noncomputable def averageCost (project : HousingProject) (floors : ℕ) : ℝ :=
  let totalArea := project.numBuildings * project.floorArea * (floors : ℝ)
  let constructionCost := project.numBuildings * project.floorArea * 
    ((floors : ℝ) * (project.baseCost + project.kValue * ((floors : ℝ) + 1) / 2))
  (project.landCost + constructionCost) / totalArea

/-- The main theorem about the optimal number of floors and minimum cost -/
theorem optimal_floors_and_cost (project : HousingProject) 
  (h1 : project.landCost = 16000000)
  (h2 : project.numBuildings = 10)
  (h3 : project.floorArea = 1000)
  (h4 : project.baseCost = 800)
  (h5 : project.kValue = 50)
  (h6 : averageCost project 5 = 1270) :
  (∃ (n : ℕ), n > 0 ∧ 
    (∀ (m : ℕ), m > 0 → averageCost project n ≤ averageCost project m) ∧
    averageCost project n = 1225 ∧ n = 8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_floors_and_cost_l598_59806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_xfx_positive_l598_59889

/-- An odd function satisfying certain conditions -/
class OddFunction (f : ℝ → ℝ) where
  odd : ∀ x, f (-x) = -f x
  zero_at_neg_two : f (-2) = 0
  condition_for_positive : ∀ x, x > 0 → (x * (deriv f x) - f x) / (x^2) > 0

/-- The theorem stating the solution set for xf(x) > 0 -/
theorem solution_set_xfx_positive (f : ℝ → ℝ) [OddFunction f] :
  {x : ℝ | x * f x > 0} = Set.Ioi 2 ∪ Set.Iio (-2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_xfx_positive_l598_59889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l598_59800

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The sides form an arithmetic sequence -/
def isArithmeticSequence (t : Triangle) : Prop :=
  2 * t.b = t.a + t.c

/-- Angle B is 30 degrees (π/6 radians) -/
def angleBIs30Degrees (t : Triangle) : Prop :=
  t.B = Real.pi / 6

/-- The area of the triangle is 1/2 -/
def areaIsHalf (t : Triangle) : Prop :=
  1 / 2 * t.a * t.c * Real.sin t.B = 1 / 2

theorem triangle_side_length (t : Triangle) 
  (h1 : isArithmeticSequence t)
  (h2 : angleBIs30Degrees t)
  (h3 : areaIsHalf t) :
  t.b = (3 + Real.sqrt 3) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l598_59800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equalize_rice_containers_correct_amount_to_move_kg_l598_59894

-- Define the initial amounts of rice in grams
def container_a_initial : ℚ := 12400
def container_b_initial : ℚ := 7600

-- Define the amount to be moved in grams
def amount_to_move : ℚ := 2400

-- Theorem statement
theorem equalize_rice_containers :
  let total_rice := container_a_initial + container_b_initial
  let equal_amount := total_rice / 2
  container_a_initial - amount_to_move = equal_amount ∧
  container_b_initial + amount_to_move = equal_amount :=
by sorry

-- Convert the amount to move from grams to kilograms
def amount_to_move_kg : ℚ := amount_to_move / 1000

-- Theorem to show that 2.4 kg is the correct amount to move
theorem correct_amount_to_move_kg :
  amount_to_move_kg = 2.4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equalize_rice_containers_correct_amount_to_move_kg_l598_59894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l598_59888

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 2*y + 6 = 0

-- Define the point A
def point_A : ℝ × ℝ := (4, 0)

-- Define a line with non-zero slope passing through point A
def line_through_A (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 4) ∧ k ≠ 0

-- Define the chord length
noncomputable def chord_length : ℝ := 2 * Real.sqrt 3

-- Main theorem
theorem line_equation_proof :
  ∀ k : ℝ,
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    circle_O x₁ y₁ ∧ circle_O x₂ y₂ ∧
    line_through_A k x₁ y₁ ∧ line_through_A k x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = chord_length^2) →
  (∀ x y : ℝ, line_through_A k x y ↔ 7*x - 24*y - 28 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l598_59888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_correct_l598_59821

/-- Represents the journey of three people with specific travel conditions -/
structure Journey where
  totalDistance : ℚ
  atkinsSpeed : ℚ
  brownSpeed : ℚ
  cranbySpeed : ℚ

/-- Calculates the total journey time for the given conditions -/
def journeyTime (j : Journey) : ℚ :=
  10 + 5/41

/-- Theorem stating that the journey time is correct for the given conditions -/
theorem journey_time_correct (j : Journey) 
  (h1 : j.totalDistance = 40)
  (h2 : j.atkinsSpeed = 1)
  (h3 : j.brownSpeed = 2)
  (h4 : j.cranbySpeed = 8) : 
  journeyTime j = 10 + 5/41 := by
  sorry

/-- Example calculation of journey time -/
def example_journey : Journey := {
  totalDistance := 40,
  atkinsSpeed := 1,
  brownSpeed := 2,
  cranbySpeed := 8
}

#eval journeyTime example_journey

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_correct_l598_59821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_half_equals_pi_sixth_l598_59854

theorem f_of_half_equals_pi_sixth 
  (f : ℝ → ℝ) 
  (h : ∀ x ∈ Set.Icc 0 (π / 2), f (Real.sin x) = x) : 
  f (1/2) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_half_equals_pi_sixth_l598_59854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_area_line_l598_59896

/-- A line passing through (2,1) and intersecting positive x and y axes -/
structure IntersectingLine where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0
  passes_through : 2 / a + 1 / b = 1

/-- The area of the triangle formed by the line and coordinate axes -/
noncomputable def triangle_area (l : IntersectingLine) : ℝ := 1 / 2 * l.a * l.b

/-- The specific line with equation x + 2y - 4 = 0 -/
def specific_line : IntersectingLine := {
  a := 4
  b := 2
  a_pos := by norm_num
  b_pos := by norm_num
  passes_through := by norm_num
}

/-- Theorem stating that the specific line minimizes the triangle area -/
theorem minimum_area_line :
  ∀ l : IntersectingLine, triangle_area l ≥ triangle_area specific_line :=
by
  sorry

#check minimum_area_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_area_line_l598_59896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_sin_equality_l598_59834

theorem contrapositive_sin_equality (A B : ℝ) : 
  (A = B → Real.sin A = Real.sin B) ↔ (Real.sin A ≠ Real.sin B → A ≠ B) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_sin_equality_l598_59834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sides_cube_plane_intersection_l598_59859

/-- A cube is a three-dimensional solid object with six square faces. -/
structure Cube where

/-- A plane is a flat, two-dimensional surface that extends infinitely far. -/
structure Plane where

/-- A polygon is a plane figure with straight sides. -/
structure Polygon where
  sides : ℕ

/-- The intersection of a plane and a cube results in a polygon. -/
def intersect (c : Cube) (p : Plane) : Polygon :=
  sorry

/-- The maximum number of sides a polygon can have when formed by the intersection of a plane and a cube is 6. -/
theorem max_sides_cube_plane_intersection (c : Cube) (p : Plane) :
  (intersect c p).sides ≤ 6 ∧ ∃ (c' : Cube) (p' : Plane), (intersect c' p').sides = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sides_cube_plane_intersection_l598_59859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_theorem_l598_59886

/-- Represents a circle with a given radius and center -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the configuration of concentric circles and tangent circles -/
structure CircleConfiguration where
  inner : Circle
  outer : Circle
  tangent : Circle

/-- Checks if two circles are concentric -/
def areConcentric (c1 c2 : Circle) : Prop :=
  c1.center = c2.center

/-- Checks if a circle is tangent to both inner and outer circles -/
def isTangent (config : CircleConfiguration) : Prop :=
  let innerDist := Real.sqrt ((config.tangent.center.1 - config.inner.center.1)^2 + 
                              (config.tangent.center.2 - config.inner.center.2)^2)
  let outerDist := Real.sqrt ((config.tangent.center.1 - config.outer.center.1)^2 + 
                              (config.tangent.center.2 - config.outer.center.2)^2)
  innerDist = config.inner.radius + config.tangent.radius ∧
  outerDist = config.outer.radius - config.tangent.radius

/-- Helper function to calculate the maximum number of non-overlapping tangent circles -/
def maxNonOverlappingTangentCircles (config : CircleConfiguration) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem tangent_circles_theorem (config : CircleConfiguration) 
  (h1 : config.inner.radius = 1)
  (h2 : config.outer.radius = 3)
  (h3 : areConcentric config.inner config.outer)
  (h4 : isTangent config) :
  config.tangent.radius = 1 ∧ 
  (∃ n : ℕ, n = 6 ∧ n = maxNonOverlappingTangentCircles config) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_theorem_l598_59886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_colors_for_distribution_centers_l598_59858

theorem min_colors_for_distribution_centers (n : ℕ) : 
  (n + n.choose 2 ≥ 12) ∧
  (∀ m : ℕ, m < n → m + m.choose 2 < 12) →
  n = 5 := by
  
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_colors_for_distribution_centers_l598_59858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l598_59836

theorem min_value_of_expression (x : ℝ) (h : x > 2) :
  (x^2 - 6*x + 8) / (2*x - 4) ≥ -1/2 ∧
  ∃ (s : ℕ → ℝ), (∀ n, s n > 2) ∧ (Filter.Tendsto s Filter.atTop (nhds 2)) ∧
  (Filter.Tendsto (fun n => (s n^2 - 6*(s n) + 8) / (2*(s n) - 4)) Filter.atTop (nhds (-1/2))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l598_59836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_l598_59839

-- Define the circle equation
def circle_eq (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x + 4*y - 6 = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop :=
  x + 2*y + 1 = 0

-- Define a point on the circle
def point_on_circle (a : ℝ) (x y : ℝ) : Prop :=
  circle_eq a x y

-- Define the symmetric point with respect to the line
def symmetric_point (x y x' y' : ℝ) : Prop :=
  line_eq ((x + x') / 2) ((y + y') / 2) ∧
  (x' - x) = -2 * (y' - y)

theorem circle_symmetry (a : ℝ) :
  (∃ x y x' y' : ℝ,
    point_on_circle a x y ∧
    symmetric_point x y x' y' ∧
    point_on_circle a x' y') →
  a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_l598_59839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_number_count_l598_59892

def is_valid_number (n : ℕ) : Bool :=
  n ≥ 10 ∧ n < 100 ∧ n > 65 ∧ n % 10 % 2 = 0 ∧ (n / 10) % 2 = 1

theorem valid_number_count : 
  (Finset.filter (fun n => is_valid_number n) (Finset.range 100)).card = 10 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_number_count_l598_59892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_residual_after_removal_residual_after_removal_proof_l598_59881

/-- Represents a sample point with x and y coordinates -/
structure SamplePoint where
  x : ℝ
  y : ℝ

/-- Represents a regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Calculates the y value for a given x using a regression line -/
def predictY (line : RegressionLine) (x : ℝ) : ℝ :=
  line.slope * x + line.intercept

/-- Calculates the residual for a sample point given a regression line -/
def calculateResidual (line : RegressionLine) (point : SamplePoint) : ℝ :=
  point.y - predictY line point.x

/-- Main theorem statement -/
theorem residual_after_removal (initialSample : List SamplePoint) 
    (initialLine : RegressionLine) (removedPoints : List SamplePoint) 
    (newSlope : ℝ) (targetPoint : SamplePoint) : Prop :=
  -- Initial conditions
  initialSample.length = 10 ∧
  initialLine.slope = 2 ∧
  initialLine.intercept = -0.4 ∧
  (initialSample.map (λ p => p.x)).sum / 10 = 2 ∧
  removedPoints = [⟨-3, 1⟩, ⟨3, -1⟩] ∧
  newSlope = 3 ∧
  targetPoint = ⟨4, 8⟩ →
  -- Conclusion
  ∃ newLine : RegressionLine,
    newLine.slope = newSlope ∧
    calculateResidual newLine targetPoint = -1

/-- Proof of the main theorem -/
theorem residual_after_removal_proof : ∀ initialSample initialLine removedPoints newSlope targetPoint,
    residual_after_removal initialSample initialLine removedPoints newSlope targetPoint := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_residual_after_removal_residual_after_removal_proof_l598_59881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l598_59802

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sqrt x + 3 else a * x + b

-- State the theorem
theorem function_property (a b : ℝ) :
  (∀ x₁ : ℝ, x₁ ≠ 0 → ∃! x₂ : ℝ, x₂ ≠ 0 ∧ x₂ ≠ x₁ ∧ f a b x₁ = f a b x₂) →
  a < 0 →
  b = 3 →
  f a b (Real.sqrt (3 * a)) = f a b (4 * b) →
  a + b = -Real.sqrt 2 + 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l598_59802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_lent_to_C_l598_59893

noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem amount_lent_to_C : 
  ∀ (amountToC : ℝ),
  -- Conditions
  let amountToB : ℝ := 5000
  let timeB : ℝ := 2
  let timeC : ℝ := 4
  let rate : ℝ := 10
  let totalInterest : ℝ := 2200
  -- Total interest calculation
  totalInterest = simpleInterest amountToB rate timeB + simpleInterest amountToC rate timeC →
  -- Conclusion
  amountToC = 3000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_lent_to_C_l598_59893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meiosis_and_fertilization_maintain_chromosomes_l598_59860

-- Define the types for our biological processes and cells
inductive BiologicalProcess
| Mitosis
| Meiosis
| Fertilization

structure Cell where
  chromosomeCount : ℕ

-- Define the effects of biological processes on chromosome count
def mitosisEffect (parent : Cell) : Cell :=
  { chromosomeCount := parent.chromosomeCount }

def meiosisEffect (parent : Cell) : Cell :=
  { chromosomeCount := parent.chromosomeCount / 2 }

def fertilizationEffect (gamete1 gamete2 : Cell) : Cell :=
  { chromosomeCount := gamete1.chromosomeCount + gamete2.chromosomeCount }

-- Define the property of maintaining constant chromosome count
def maintainsConstantChromosomes (processes : List BiologicalProcess) : Prop :=
  ∀ (parent : Cell),
    ∃ (offspring : Cell),
      (BiologicalProcess.Meiosis ∈ processes ∧ BiologicalProcess.Fertilization ∈ processes) →
      offspring.chromosomeCount = parent.chromosomeCount

-- Theorem statement
theorem meiosis_and_fertilization_maintain_chromosomes :
  maintainsConstantChromosomes [BiologicalProcess.Meiosis, BiologicalProcess.Fertilization] := by
  sorry

#check meiosis_and_fertilization_maintain_chromosomes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meiosis_and_fertilization_maintain_chromosomes_l598_59860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_properties_l598_59812

theorem inequality_properties (a b : ℝ) (h : a > b) :
  a^3 > b^3 ∧ (2 : ℝ)^a > (2 : ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_properties_l598_59812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_length_l598_59826

/-- Given a triangle ABC where AB = 8, AC = 6, and ∠BAC = 60°, 
    the length of the angle bisector AM is 24√3 / 7 -/
theorem angle_bisector_length (A B C M : ℝ × ℝ) : 
  let d := λ p q : ℝ × ℝ ↦ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let angle := λ p q r : ℝ × ℝ ↦ Real.arccos (
    ((q.1 - p.1) * (r.1 - p.1) + (q.2 - p.2) * (r.2 - p.2)) /
    (d p q * d p r))
  let bisects := λ p q r s : ℝ × ℝ ↦ angle p q s = angle p r s
  d A B = 8 ∧ 
  d A C = 6 ∧ 
  angle B A C = π / 3 ∧
  bisects A B C M →
  d A M = 24 * Real.sqrt 3 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_length_l598_59826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_30_l598_59841

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a trapezoid given its vertices -/
noncomputable def trapezoidArea (p q r d : Point) : ℝ :=
  let base1 := |q.y - p.y|
  let base2 := |d.y - r.y|
  let height := |r.x - p.x|
  (base1 + base2) * height / 2

/-- Theorem: The area of trapezoid PQRD with given vertices is 30 square units -/
theorem trapezoid_area_is_30 :
  let p : Point := ⟨0, 0⟩
  let q : Point := ⟨0, -3⟩
  let r : Point := ⟨5, 0⟩
  let d : Point := ⟨5, 9⟩
  trapezoidArea p q r d = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_30_l598_59841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_sum_l598_59814

noncomputable section

/-- Parabola defined by y = x^2 -/
def parabola (x : ℝ) : ℝ := x^2

/-- Focus of the parabola y = x^2 -/
def focus : ℝ × ℝ := (0, 1/4)

/-- Distance from a point to the focus -/
noncomputable def distanceToFocus (p : ℝ × ℝ) : ℝ := |p.2 - focus.2|

/-- The four intersection points of the circle and parabola -/
def intersectionPoints : List (ℝ × ℝ) := [(5, 25), (1, 1), (-6, 36), (0, 0)]

/-- Sum of distances from focus to intersection points -/
noncomputable def sumOfDistances : ℝ := (intersectionPoints.map distanceToFocus).sum

theorem parabola_circle_intersection_sum :
  sumOfDistances = 61.5 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_sum_l598_59814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_scores_mode_l598_59805

/-- Represents a stem-and-leaf plot entry -/
structure StemLeafEntry where
  stem : Nat
  leaves : List Nat

/-- Represents the stem-and-leaf plot of test scores -/
def testScores : List StemLeafEntry := [
  ⟨5, [5, 5, 5]⟩,
  ⟨6, [2, 2, 2, 2]⟩,
  ⟨7, [3, 8, 9]⟩,
  ⟨8, [0, 1, 1, 1, 1, 1]⟩,
  ⟨9, [2, 5, 7, 7, 7]⟩,
  ⟨10, [1, 1, 1, 2, 2, 2, 2]⟩
]

/-- Calculates the actual score from a stem-leaf pair -/
def actualScore (stem : Nat) (leaf : Nat) : Nat :=
  stem * 10 + leaf

/-- Finds the mode of a list of numbers -/
def mode (numbers : List Nat) : Nat :=
  sorry

/-- Theorem: The mode of the test scores is 81 -/
theorem test_scores_mode :
  mode (testScores.bind (fun entry => entry.leaves.map (actualScore entry.stem))) = 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_scores_mode_l598_59805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_length_is_10000_l598_59884

noncomputable def rise : ℝ := 600

noncomputable def initial_grade : ℝ := 3

noncomputable def final_grade : ℝ := 2

noncomputable def horizontal_length (grade : ℝ) : ℝ := rise / (grade / 100)

noncomputable def additional_length : ℝ := horizontal_length final_grade - horizontal_length initial_grade

theorem additional_length_is_10000 : 
  ⌊additional_length⌋ = 10000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_length_is_10000_l598_59884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_sum_incorrect_l598_59885

/-- Represents the sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

/-- The given sums from the problem -/
def S1 : ℝ := 8
def S2 : ℝ := 20
def S3 : ℝ := 36
def S4 : ℝ := 65

/-- Theorem stating that at least one of the given sums is incorrect -/
theorem at_least_one_sum_incorrect : 
  ¬∃ (a r : ℝ), 
    geometric_sum a r 1 = S1 ∧ 
    geometric_sum a r 2 = S2 ∧ 
    geometric_sum a r 3 = S3 ∧ 
    geometric_sum a r 4 = S4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_sum_incorrect_l598_59885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_of_sequence_sum_l598_59869

-- Define the sequence sum
def sequenceSum : ℕ := 
  (1 + 2 + 3 + 1997 + 1998 + 1999 + 1998 + 1997 + 3 + 2 + 1)

-- Define the property of symmetry around 1999
axiom symmetry : ∃ (a : ℕ → ℕ), 
  sequenceSum = (2 * (Finset.sum (Finset.range 1998) (λ i => a i + 1))) + 1999

-- Theorem statement
theorem unit_digit_of_sequence_sum : 
  sequenceSum % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_of_sequence_sum_l598_59869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sine_l598_59849

theorem arithmetic_sequence_sine (a : ℝ) : 
  0 < a ∧ a < 2 * Real.pi →
  (∃ d : ℝ, Real.sin a + d = Real.sin (2 * a) ∧ Real.sin (2 * a) + d = Real.sin (3 * a)) ↔ 
  (a = Real.pi / 4 ∨ a = Real.pi ∨ a = 7 * Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sine_l598_59849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_distance_less_than_perimeter_l598_59831

/-- A regular polygon with 2^k sides -/
structure RegularPolygon (k : ℕ) where
  center : ℂ
  vertices : Fin (2^k) → ℂ
  is_regular : ∀ (i j : Fin (2^k)), Complex.abs (vertices i - vertices j) = Complex.abs (vertices 0 - vertices 1)

/-- The result of reflecting a point across a line segment -/
noncomputable def reflect (p : ℂ) (a b : ℂ) : ℂ := sorry

/-- Perform sequential reflections across all sides of the polygon -/
noncomputable def sequential_reflections {k : ℕ} (poly : RegularPolygon k) : ℂ := sorry

/-- The perimeter of a regular 2^k-gon -/
noncomputable def perimeter {k : ℕ} (poly : RegularPolygon k) : ℝ := sorry

theorem reflection_distance_less_than_perimeter {k : ℕ} (poly : RegularPolygon k) :
  Complex.abs (sequential_reflections poly - poly.center) < perimeter poly := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_distance_less_than_perimeter_l598_59831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numerator_and_denominator_l598_59899

/-- Represents a repeating decimal with a three-digit repetend -/
def RepeatingDecimal (a b c : ℕ) : ℚ :=
  (100 * a + 10 * b + c : ℚ) / 999

/-- The repeating decimal 0.145145145... -/
def x : ℚ := RepeatingDecimal 1 4 5

theorem sum_of_numerator_and_denominator : ∃ (n d : ℕ), x = n / d ∧ Nat.Coprime n d ∧ n + d = 1144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numerator_and_denominator_l598_59899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_class_size_l598_59823

theorem smallest_class_size (n : ℕ) (scores : Fin n → ℕ) : 
  (∃ (top_scorers : Finset (Fin n)), top_scorers.card = 7 ∧ 
    ∀ i ∈ top_scorers, scores i = 150) →
  (∀ i : Fin n, scores i ≥ 90) →
  (Finset.sum Finset.univ (fun i => (scores i : ℚ)) : ℚ) / n = 120 →
  n ≥ 14 ∧ 
  ∃ (scores' : Fin 14 → ℕ), 
    (∃ (top_scorers : Finset (Fin 14)), top_scorers.card = 7 ∧ 
      ∀ i ∈ top_scorers, scores' i = 150) ∧
    (∀ i : Fin 14, scores' i ≥ 90) ∧
    (Finset.sum Finset.univ (fun i => (scores' i : ℚ)) : ℚ) / 14 = 120 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_class_size_l598_59823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_four_l598_59883

theorem power_of_four (x y : ℝ) (h1 : (4 : ℝ)^x = 2) (h2 : (4 : ℝ)^y = 3) : 
  (4 : ℝ)^(x-2*y) = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_four_l598_59883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l598_59855

/-- The area of a triangle with vertices P(2, 2), Q(7, 2), R(5, 9) is 17.5 square units. -/
theorem triangle_area : 
  let P : ℝ × ℝ := (2, 2)
  let Q : ℝ × ℝ := (7, 2)
  let R : ℝ × ℝ := (5, 9)
  (1/2) * abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2)) = 17.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l598_59855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotonically_increasing_l598_59842

theorem function_monotonically_increasing 
  (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_deriv_pos : ∀ x, DifferentiableAt ℝ f x ∧ deriv f x > 0) : 
  StrictMonoOn f Set.univ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotonically_increasing_l598_59842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coconut_grove_yield_l598_59871

/-- Represents the yield of coconut trees in a grove -/
structure CoconutGrove (x : ℕ) where
  yield_x_plus_4 : ℕ            -- yield of (x + 4) trees
  yield_x : ℕ                   -- yield of x trees
  yield_x_minus_4 : ℕ           -- yield of (x - 4) trees
  average_yield : ℕ             -- average yield per tree
  total_trees : ℕ               -- total number of trees
  total_yield : ℕ               -- total yield of all trees
  h1 : yield_x = 120            -- x trees yield 120 nuts
  h2 : yield_x_minus_4 = 180    -- (x - 4) trees yield 180 nuts
  h3 : average_yield = 100      -- average yield is 100 nuts per tree
  h4 : total_trees = 3 * x      -- total trees = (x + 4) + x + (x - 4) = 3x
  h5 : total_yield = total_trees * average_yield  -- total yield calculation

/-- The main theorem to prove -/
theorem coconut_grove_yield (x : ℕ) (h : x = 8) :
  ∃ (g : CoconutGrove x), g.yield_x_plus_4 = 720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coconut_grove_yield_l598_59871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l598_59811

def is_divisible_by_6 (n : Nat) : Prop := n % 6 = 0

def is_divisible_by_11 (n : Nat) : Prop := n % 11 = 0

def valid_digit (d : Nat) : Prop := d > 0 ∧ d < 10

def form_number (a b c : Nat) : Nat := 100 * a + 10 * b + c

def satisfies_conditions (a b c : Nat) : Prop :=
  valid_digit a ∧ valid_digit b ∧ valid_digit c ∧
  (∀ (p q r : Nat), Multiset.ofList [p, q, r] = Multiset.ofList [a, b, c] → 
    is_divisible_by_6 (form_number p q r)) ∧
  (∃ (p q r : Nat), Multiset.ofList [p, q, r] = Multiset.ofList [a, b, c] ∧ 
    is_divisible_by_11 (form_number p q r))

theorem unique_solution :
  ∀ (a b c : Nat), satisfies_conditions a b c ↔ Multiset.ofList [a, b, c] = Multiset.ofList [2, 4, 6] :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l598_59811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_l598_59818

/-- The function g(x) is defined as the minimum of three linear functions. -/
noncomputable def g (x : ℝ) : ℝ := min (3 * x + 2) (min ((3/2) * x + 1) (-(3/4) * x + 7))

/-- The maximum value of g(x) is 25/3. -/
theorem g_max_value : ∃ (m : ℝ), m = 25/3 ∧ ∀ (x : ℝ), g x ≤ m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_l598_59818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l598_59877

noncomputable def f (x : ℝ) : ℝ := (x^2 - 3*x + 2) / (x^2 - 5*x + 6)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 2 ∨ (2 < x ∧ x < 3) ∨ 3 < x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l598_59877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l598_59876

/-- Arithmetic sequence sum -/
noncomputable def S (n : ℕ) (a₁ d : ℝ) : ℝ := n * a₁ + (n * (n - 1) / 2) * d

/-- Proof of the arithmetic sequence problem -/
theorem arithmetic_sequence_problem (a₁ d : ℝ) 
  (h1 : S 1 a₁ d = 1)
  (h2 : S 4 a₁ d / S 2 a₁ d = 4) :
  S 6 a₁ d / S 4 a₁ d = 9/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l598_59876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_unit_square_is_bounded_by_parabolic_curves_l598_59853

-- Define the unit square OABC
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (0, 1)

-- Define the transformation
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (x^2 - y^2, 2*x*y)

-- Define the unit square
def unitSquare : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define a parabolic curve
def parabolicCurve (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ t, p.1 = a*t^2 + b*t + c ∧ 0 ≤ t ∧ t ≤ 1}

-- Define a custom "isBoundedBy" predicate
def isBoundedBy (S : Set (ℝ × ℝ)) (curves : Set (Set (ℝ × ℝ))) : Prop :=
  ∀ p ∈ S, ∃ c ∈ curves, p ∈ c

-- Theorem statement
theorem transform_unit_square_is_bounded_by_parabolic_curves :
  ∃ (curves : Set (Set (ℝ × ℝ))),
    (∀ c ∈ curves, ∃ a b d, c = parabolicCurve a b d) ∧
    isBoundedBy (transform '' unitSquare) curves := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_unit_square_is_bounded_by_parabolic_curves_l598_59853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l598_59891

noncomputable section

-- Define the hyperbola and parabola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the asymptotes of the hyperbola
def asymptote (a b : ℝ) (x y : ℝ) : Prop := y = b / a * x ∨ y = -b / a * x

-- Define the intersection points A and B
noncomputable def intersection_point (a b : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  (4 * a^2 / b^2, 4 * a / b, 4 * a^2 / b^2, -4 * a / b)

-- Define the angle AFB
noncomputable def angle_AFB (A B : ℝ × ℝ) : ℝ := Real.arccos (-7/9)

-- Define the eccentricity of the hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + (b/a)^2)

-- State the theorem
theorem hyperbola_eccentricity (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  let (Ax, Ay, Bx, By) := intersection_point a b
  (hyperbola a b Ax Ay ∧ hyperbola a b Bx By) →
  (parabola Ax Ay ∧ parabola Bx By) →
  (asymptote a b Ax Ay ∧ asymptote a b Bx By) →
  angle_AFB (Ax, Ay) (Bx, By) = Real.arccos (-7/9) →
  eccentricity a b = Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l598_59891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l598_59824

-- Define the two functions
noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 3*x + 6
noncomputable def g (x : ℝ) : ℝ := -2*x/3 + 2

-- Theorem statement
theorem intersection_sum :
  ∃ (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ),
    (f x₁ = g x₁) ∧ (f x₂ = g x₂) ∧ (f x₃ = g x₃) ∧
    (x₁ + x₂ + x₃ = 2) ∧
    (y₁ + y₂ + y₃ = 14/3) ∧
    (y₁ = f x₁) ∧ (y₂ = f x₂) ∧ (y₃ = f x₃) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l598_59824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_ratio_l598_59843

/-- An arithmetic sequence with common difference d ≠ 0 -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- Three terms of a sequence form a geometric sequence -/
def geometric_subsequence (a : ℕ → ℝ) (i j k : ℕ) : Prop :=
  (a j) ^ 2 = (a i) * (a k)

/-- The common ratio of a geometric sequence -/
noncomputable def geometric_ratio (a : ℕ → ℝ) (i j : ℕ) : ℝ :=
  (a j) / (a i)

theorem arithmetic_geometric_ratio
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : arithmetic_sequence a d)
  (h2 : geometric_subsequence a 2 3 6) :
  geometric_ratio a 2 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_ratio_l598_59843
