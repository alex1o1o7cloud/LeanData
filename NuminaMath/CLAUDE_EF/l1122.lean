import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1122_112256

noncomputable def f (x : ℝ) := Real.sin (x + Real.pi/6) + Real.cos x

theorem f_properties :
  ∃ (max_value : ℝ) (max_set : Set ℝ),
    (∀ x, f x ≤ max_value) ∧
    (max_set = {x | f x = max_value}) ∧
    (max_value = Real.sqrt 3) ∧
    (max_set = {x | ∃ k : ℤ, x = Real.pi/6 + 2 * Real.pi * (k : ℝ)}) ∧
    (∀ a : ℝ, a ∈ Set.Ioo 0 (Real.pi/2) →
      f (a + Real.pi/6) = 3 * Real.sqrt 3 / 5 →
      f (2 * a) = (24 * Real.sqrt 3 - 21) / 50) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1122_112256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l1122_112238

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (1, -3)
def center2 : ℝ × ℝ := (2, -1)
def radius1 : ℝ := 2
def radius2 : ℝ := 1

-- Define the distance between the centers
noncomputable def center_distance : ℝ := Real.sqrt 5

-- Theorem stating that the circles are intersecting
theorem circles_intersect :
  (radius1 - radius2 < center_distance) ∧
  (center_distance < radius1 + radius2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l1122_112238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_minimum_perimeter_l1122_112218

theorem equilateral_triangle_minimum_perimeter (a b c : ℝ) (S : ℝ) :
  a > 0 → b > 0 → c > 0 → S > 0 →
  S = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c)) →
  ∀ x y z : ℝ,
    x > 0 → y > 0 → z > 0 →
    S = Real.sqrt ((x + y + z) / 2 * ((x + y + z) / 2 - x) * ((x + y + z) / 2 - y) * ((x + y + z) / 2 - z)) →
    a + b + c ≤ x + y + z →
    a = b ∧ b = c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_minimum_perimeter_l1122_112218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_sum_l1122_112277

/-- Represents a triangle with side lengths 6, 8, and x, where x is variable --/
structure SpecialTriangle where
  x : ℝ
  angles_in_arithmetic_progression : Bool
  satisfies_triangle_inequality : Bool

/-- Represents the sum of possible x values in the form a + √b + c√d --/
structure SumOfPossibleX where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- Function to calculate the sum of possible x values --/
noncomputable def sum_of_possible_x (x : ℝ) : ℝ :=
  3 + Real.sqrt 37 + 2 * Real.sqrt 13

/-- 
  Theorem: For a triangle with angles in arithmetic progression and 
  side lengths 6, 8, and x, where x satisfies the triangle inequality,
  if the sum of possible x values is represented as a + √b + c√d,
  then a + b + c + d = 55
--/
theorem special_triangle_sum (t : SpecialTriangle) (s : SumOfPossibleX) 
  (h1 : t.angles_in_arithmetic_progression = true)
  (h2 : t.satisfies_triangle_inequality = true)
  (h3 : s.a + Real.sqrt s.b + s.c * Real.sqrt s.d = sum_of_possible_x t.x) :
  s.a + s.b + s.c + s.d = 55 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_sum_l1122_112277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_equals_target_l1122_112201

/-- The number of green balls in the bin -/
def green_balls : ℕ := 7

/-- The number of purple balls in the bin -/
def k : ℕ := 4  -- We define k as 4 based on the solution

/-- The total number of balls in the bin -/
def total_balls : ℕ := green_balls + k

/-- The probability of drawing a green ball -/
noncomputable def prob_green : ℚ := green_balls / total_balls

/-- The probability of drawing a purple ball -/
noncomputable def prob_purple : ℚ := k / total_balls

/-- The amount won if a green ball is drawn -/
def green_win : ℚ := 3

/-- The amount lost if a purple ball is drawn -/
def purple_loss : ℚ := 3

/-- The expected value of the game -/
noncomputable def expected_value : ℚ := 
  prob_green * green_win - prob_purple * purple_loss

/-- The target expected value -/
def target_value : ℚ := 3/4

/-- Theorem stating that the expected value equals the target value when k = 4 -/
theorem expected_value_equals_target : 
  expected_value = target_value := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_equals_target_l1122_112201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_covered_by_semicircles_l1122_112254

/-- Represents a triangle with side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a semicircle with a given diameter -/
structure Semicircle where
  diameter : ℝ

/-- Calculate the area of a semicircle given its diameter -/
noncomputable def semicircleArea (s : Semicircle) : ℝ :=
  Real.pi * s.diameter^2 / 8

/-- Calculate the area of a triangle given its side lengths -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  let s := (t.a + t.b + t.c) / 2
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

theorem isosceles_right_triangle_covered_by_semicircles 
  (t : Triangle) 
  (s_large : Semicircle) 
  (s_small1 s_small2 : Semicircle) :
  t.a = t.b ∧ 
  t.c = 16 ∧ 
  t.a^2 + t.b^2 = t.c^2 ∧
  s_large.diameter = t.c ∧
  s_small1.diameter = t.a ∧
  s_small2.diameter = t.b →
  triangleArea t = semicircleArea s_large - semicircleArea s_small1 - semicircleArea s_small2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_covered_by_semicircles_l1122_112254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1122_112253

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

noncomputable def M : ℝ := 2
noncomputable def T : ℝ := Real.pi

theorem function_properties :
  (∀ x, f x ≤ M) ∧
  (∀ x, f (x + T) = f x) ∧
  (∀ t, t > 0 → (∀ x, f (x + t) = f x) → t ≥ T) ∧
  (∃ x_list : List ℝ,
    x_list.length = 10 ∧
    x_list.Nodup ∧
    (∀ x ∈ x_list, 0 < x ∧ x < 10 * Real.pi ∧ f x = M) ∧
    x_list.sum = 140 / 3 * Real.pi) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1122_112253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_shift_shrink_is_sin_l1122_112247

noncomputable def f (x : ℝ) := Real.cos (2 * x)

noncomputable def g (x : ℝ) := f ((2 * x) + Real.pi / 4)

theorem cos_shift_shrink_is_sin :
  ∀ x : ℝ, g x = Real.sin x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_shift_shrink_is_sin_l1122_112247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_range_l1122_112270

theorem complex_modulus_range (Z : ℂ) (a : ℝ) (h1 : 0 < a) (h2 : a < 2) (h3 : Z = Complex.ofReal a + Complex.I) :
  1 < Complex.abs Z ∧ Complex.abs Z < Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_range_l1122_112270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vision_status_study_l1122_112297

structure School where
  total_students : ℕ
  measured_students : ℕ

def Population (s : School) : ℕ := s.total_students

def Individual (s : School) : Type := Unit

def Sample (s : School) : Finset Unit := Finset.univ

def SampleSize (s : School) : ℕ := s.measured_students

theorem vision_status_study (s : School)
  (h1 : s.total_students = 200)
  (h2 : s.measured_students = 50) :
  Population s = 200 ∧
  (∃ (i : Individual s), True) ∧
  (∃ (sample : Finset (Individual s)), sample = Sample s) ∧
  SampleSize s = 50 := by
  constructor
  · exact h1
  constructor
  · exact ⟨(), trivial⟩
  constructor
  · exact ⟨Sample s, rfl⟩
  · exact h2

#check vision_status_study

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vision_status_study_l1122_112297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_simplification_l1122_112279

noncomputable def lhs (x : ℝ) : ℝ := (2*x - 3) / 4 + (5*x + 2) / 5

noncomputable def rhs (x : ℝ) : ℝ := (30*x - 7) / 20

theorem fraction_simplification (x : ℝ) : lhs x = rhs x := by
  -- Expand the definitions of lhs and rhs
  unfold lhs rhs
  -- Normalize the expressions
  norm_num
  -- Perform algebraic simplifications
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_simplification_l1122_112279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_triangle_areas_l1122_112261

-- Define the points on the first line
structure Line1 where
  G : ℝ × ℝ
  H : ℝ × ℝ
  I : ℝ × ℝ
  J : ℝ × ℝ

-- Define the points on the second line
structure Line2 where
  K : ℝ × ℝ
  L : ℝ × ℝ
  M : ℝ × ℝ

-- Define the properties of the points
def valid_configuration (l1 : Line1) (l2 : Line2) : Prop :=
  let (gx, gy) := l1.G
  let (hx, hy) := l1.H
  let (ix, iy) := l1.I
  let (jx, jy) := l1.J
  let (kx, ky) := l2.K
  let (lx, ly) := l2.L
  let (mx, my) := l2.M
  (hx - gx)^2 + (hy - gy)^2 = 4 ∧
  (ix - hx)^2 + (iy - hy)^2 = 9 ∧
  (jx - ix)^2 + (jy - iy)^2 = 16 ∧
  (lx - kx)^2 + (ly - ky)^2 = 4 ∧
  (mx - lx)^2 + (my - ly)^2 = 9 ∧
  ∃ (a b : ℝ), a ≠ 0 ∧ 
    (gy = a * gx + b) ∧ (hy = a * hx + b) ∧ (iy = a * ix + b) ∧ (jy = a * jx + b) ∧
    ∃ (c : ℝ), c ≠ b ∧ (ky = a * kx + c) ∧ (ly = a * lx + c) ∧ (my = a * mx + c)

-- Theorem statement
theorem num_triangle_areas (l1 : Line1) (l2 : Line2) :
  valid_configuration l1 l2 → ∃ (areas : Finset ℝ), areas.card = 6 ∧ 
    ∀ (triangle_area : ℝ), 
      (∃ (p1 p2 p3 : ℝ × ℝ), 
        (p1 ∈ ({l1.G, l1.H, l1.I, l1.J, l2.K, l2.L, l2.M} : Set (ℝ × ℝ)) ∧
         p2 ∈ ({l1.G, l1.H, l1.I, l1.J, l2.K, l2.L, l2.M} : Set (ℝ × ℝ)) ∧
         p3 ∈ ({l1.G, l1.H, l1.I, l1.J, l2.K, l2.L, l2.M} : Set (ℝ × ℝ)) ∧
         p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
         triangle_area = abs ((p1.1 - p3.1) * (p2.2 - p3.2) - (p2.1 - p3.1) * (p1.2 - p3.2)) / 2)) ↔
      triangle_area ∈ areas := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_triangle_areas_l1122_112261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_function_value_l1122_112292

noncomputable def f (A ω x : ℝ) : ℝ := 2 * Real.sin (ω * x) * Real.cos (ω * x) + 2 * A * (Real.cos (ω * x))^2 - A

theorem function_properties (A ω : ℝ) (h1 : A > 0) (h2 : ω > 0)
  (h3 : ∀ x, f A ω (x + π) = f A ω x)  -- minimum positive period is π
  (h4 : ∀ x, f A ω x ≤ 2)  -- maximum value is 2
  (h5 : ∃ x, f A ω x = 2)  -- maximum value is achieved
  : A = Real.sqrt 3 ∧ ω = 1 := by
  sorry

theorem function_value (A ω θ : ℝ) (h1 : A > 0) (h2 : ω > 0)
  (h3 : ∀ x, f A ω (x + π) = f A ω x)  -- minimum positive period is π
  (h4 : ∀ x, f A ω x ≤ 2)  -- maximum value is 2
  (h5 : ∃ x, f A ω x = 2)  -- maximum value is achieved
  (h6 : π / 6 < θ ∧ θ < π / 3)
  (h7 : f A ω θ = 2 / 3)
  : f A ω (π / 3 - θ) = (1 + 2 * Real.sqrt 6) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_function_value_l1122_112292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2014_eq_one_fifth_l1122_112227

/-- The sequence a_n defined by the given recurrence relation -/
def a : ℕ → ℚ
  | 0 => 3/5  -- Add this case for n = 0
  | 1 => 3/5
  | n + 2 => if a (n + 1) < 1/2 then 2 * a (n + 1) else 2 * a (n + 1) - 1

/-- The theorem stating that the 2014th term of the sequence is 1/5 -/
theorem a_2014_eq_one_fifth : a 2014 = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2014_eq_one_fifth_l1122_112227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_curved_surface_area_l1122_112290

/-- The curved surface area of a cone -/
noncomputable def curved_surface_area (r l : ℝ) : ℝ := Real.pi * r * l

/-- Theorem: The curved surface area of a cone with slant height 18 cm and base radius 8 cm is 144π cm² -/
theorem cone_curved_surface_area : 
  curved_surface_area 8 18 = 144 * Real.pi := by
  unfold curved_surface_area
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_curved_surface_area_l1122_112290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l1122_112269

theorem binomial_expansion_coefficient (a : ℝ) : 
  (∃ k : ℝ, k = 84 ∧ k = 4 * (-a)^5 * (Nat.choose 7 5)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l1122_112269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l1122_112244

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a else Real.log x / Real.log a

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem f_increasing_iff_a_in_range (a : ℝ) :
  (is_increasing (f a)) ↔ (a ∈ Set.Icc (3/2) 3 ∧ a ≠ 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l1122_112244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_quantifier_negation_of_absolute_value_inequality_l1122_112226

theorem negation_of_universal_quantifier {α : Type*} (P : α → Prop) :
  (¬∀ x : α, P x) ↔ (∃ x : α, ¬P x) :=
by sorry

theorem negation_of_absolute_value_inequality {α : Type*} [NormedAddCommGroup α] :
  (¬∀ x : α, ‖x‖ ≠ 2) ↔ (∃ x : α, ‖x‖ = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_quantifier_negation_of_absolute_value_inequality_l1122_112226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_characteristic_l1122_112294

theorem units_digit_characteristic (p : ℕ) (h1 : p % 2 = 0) (h2 : (p + 2) % 10 = 8) :
  p % 10 = 6 ∧ p^3 % 10 = p^2 % 10 := by
  have hp : p % 10 = 6 := by
    have h : (p + 2) % 10 = ((p % 10) + 2) % 10 := by sorry
    rw [h2] at h
    sorry -- Proof that p % 10 = 6
  
  have hp2 : p^2 % 10 = 6 := by
    sorry -- Proof that p^2 % 10 = 6 when p % 10 = 6
  
  have hp3 : p^3 % 10 = 6 := by
    sorry -- Proof that p^3 % 10 = 6 when p % 10 = 6

  constructor
  · exact hp
  · rw [hp2, hp3]

#check units_digit_characteristic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_characteristic_l1122_112294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_years_l1122_112210

/-- Pete's current age -/
def p : ℕ := by sorry

/-- Claire's current age -/
def c : ℕ := by sorry

/-- The number of years until the ratio of Pete's age to Claire's age is 2:1 -/
def x : ℕ := by sorry

/-- Two years ago, Pete was three times as old as Claire -/
axiom two_years_ago : p - 2 = 3 * (c - 2)

/-- Four years ago, Pete was four times as old as Claire -/
axiom four_years_ago : p - 4 = 4 * (c - 4)

/-- The ratio of their ages will be 2:1 after x years -/
axiom future_ratio : (p + x) = 2 * (c + x)

theorem age_ratio_years : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_years_l1122_112210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sum_third_angle_measure_l1122_112249

/-- The sum of interior angles in a triangle is 180 degrees -/
theorem triangle_angle_sum : ∀ (a b c : ℝ), a + b + c = 180 → True := by sorry

/-- Given a triangle with two interior angles of 50° and 80°, prove that the third angle is 50° -/
theorem third_angle_measure (angle1 angle2 angle3 : ℝ) 
  (h1 : angle1 = 50)
  (h2 : angle2 = 80)
  (h3 : angle1 + angle2 + angle3 = 180) :
  angle3 = 50 := by
  sorry

#check third_angle_measure

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sum_third_angle_measure_l1122_112249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_observation_mean_change_l1122_112260

theorem observation_mean_change (n : ℕ) (h : n > 0) : 
  (n : ℝ) * 200 - n * 15 = n * 185 := by
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_observation_mean_change_l1122_112260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_through_M_l1122_112268

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y = 0

-- Define point M
def M : ℝ × ℝ := (1, 1)

-- Define the line
def my_line (x y : ℝ) : Prop := x + 2*y - 3 = 0

-- Theorem statement
theorem shortest_chord_through_M :
  ∀ (l : ℝ → ℝ → Prop),
  (∀ x y, l x y ↔ my_line x y) →
  (∀ c : ℝ → ℝ → Prop, c = my_circle →
    ∀ other_line : ℝ → ℝ → Prop,
    (other_line M.1 M.2) →
    (∃ x1 y1 x2 y2, c x1 y1 ∧ c x2 y2 ∧ other_line x1 y1 ∧ other_line x2 y2) →
    (∃ x1 y1 x2 y2, c x1 y1 ∧ c x2 y2 ∧ l x1 y1 ∧ l x2 y2 ∧
      (x2 - x1)^2 + (y2 - y1)^2 ≤ (x2 - x1)^2 + (y2 - y1)^2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_through_M_l1122_112268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_is_zero_l1122_112275

open Real

/-- The partial fraction decomposition of a rational function -/
def partialFractionDecomposition (x : ℝ) (A B C D E : ℝ) : Prop :=
  1 / ((x - 1) * x * (x + 1) * (x + 2) * (x + 3)) =
  A / (x - 1) + B / x + C / (x + 1) + D / (x + 2) + E / (x + 3)

/-- The theorem stating that the sum of coefficients in the partial fraction decomposition is zero -/
theorem sum_of_coefficients_is_zero :
  ∀ A B C D E : ℝ, (∀ x : ℝ, partialFractionDecomposition x A B C D E) →
  A + B + C + D + E = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_is_zero_l1122_112275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_stashed_10_carrots_l1122_112259

/-- Represents the number of burrows dug by the rabbit -/
def rabbit_burrows : ℕ := sorry

/-- Represents the number of burrows dug by the fox -/
def fox_burrows : ℕ := sorry

/-- The fox needs 3 fewer burrows than the rabbit -/
axiom burrow_difference : fox_burrows = rabbit_burrows - 3

/-- The rabbit places 2 carrots in each burrow -/
def rabbit_carrots : ℕ := 2 * rabbit_burrows

/-- The fox places 5 carrots in each burrow -/
def fox_carrots : ℕ := 5 * fox_burrows

/-- The total number of carrots stashed by both animals is the same -/
axiom equal_carrots : rabbit_carrots = fox_carrots

theorem rabbit_stashed_10_carrots : rabbit_carrots = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_stashed_10_carrots_l1122_112259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_equation_solution_range_l1122_112246

theorem cos_sin_equation_solution_range (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Ioo 0 (Real.pi / 2) ∧ Real.cos x ^ 2 + Real.sin x + a = 0) ↔ 
  a ∈ Set.Icc (-5/4) (-1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_equation_solution_range_l1122_112246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_drawing_probabilities_l1122_112215

/-- Represents the contents of the bag -/
structure Bag where
  red : ℕ
  white : ℕ

/-- Represents the number of balls drawn -/
def drawn : ℕ := 4

/-- Represents the point values of the balls -/
def points : Bag where
  red := 2
  white := 1

/-- The initial bag configuration -/
def initial_bag : Bag where
  red := 6
  white := 4

/-- The probability of drawing more red balls than white balls -/
def prob_more_red : Bag → ℚ
  | _ => 19 / 42

/-- The expected value of the total score -/
def expected_score : Bag → ℚ
  | _ => 32 / 5

/-- Main theorem statement -/
theorem ball_drawing_probabilities :
  prob_more_red initial_bag = 19 / 42 ∧
  expected_score initial_bag = 32 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_drawing_probabilities_l1122_112215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equals_one_l1122_112272

/-- Given a finite set of positive real numbers with a specific parity property,
    prove that the product of all its elements is 1. -/
theorem product_equals_one (A : Finset ℝ) 
    (h_pos : ∀ x ∈ A, x > 0)
    (h_parity : ∀ a : ℝ, a > 0 → 
      (A.filter (fun x => x > a)).card % 2 = 
      (A.filter (fun x => x < 1/a)).card % 2) :
  A.prod (fun x => x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equals_one_l1122_112272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_duration_calculation_l1122_112257

/-- Calculate the investment duration given the principal, interest rate, and final amount -/
noncomputable def investment_duration (principal : ℝ) (rate : ℝ) (final_amount : ℝ) : ℝ :=
  (final_amount - principal) / (principal * rate)

theorem investment_duration_calculation :
  let principal : ℝ := 1100
  let rate : ℝ := 0.05
  let final_amount : ℝ := 1232
  investment_duration principal rate final_amount = 2.4 := by
  -- Unfold the definition of investment_duration
  unfold investment_duration
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_duration_calculation_l1122_112257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_beta_l1122_112228

theorem sin_alpha_plus_beta (α β : Real) 
  (h1 : 0 < β) (h2 : β < α) (h3 : α < π/2)
  (h4 : Real.cos (α - β) = 12/13) (h5 : Real.cos (2*β) = 3/5) : 
  Real.sin (α + β) = 63/65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_beta_l1122_112228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_theorem_l1122_112291

-- Define the constants given in the problem
noncomputable def total_time : ℝ := 10
noncomputable def speed_first_half : ℝ := 25
noncomputable def speed_second_half : ℝ := 40

-- Define the function to calculate the total distance
noncomputable def calculate_distance (t : ℝ) (v1 v2 : ℝ) : ℝ :=
  (t * v1 * v2) / (v1 / 2 + v2 / 2)

-- Theorem statement
theorem journey_distance_theorem :
  ∃ (d : ℝ), abs (d - calculate_distance total_time speed_first_half speed_second_half) < 0.5 ∧ 
             d = 154 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_theorem_l1122_112291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bracelet_pairing_impossibility_l1122_112278

theorem bracelet_pairing_impossibility (n : ℕ) (h : n = 100) :
  ¬ ∃ (arrangement : List (Finset ℕ)),
    (∀ t, t ∈ arrangement → t.card = 3) ∧
    (∀ i j, i < n → j < n → i ≠ j →
      ∃! t, t ∈ arrangement ∧ i ∈ t ∧ j ∈ t) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bracelet_pairing_impossibility_l1122_112278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_iff_lambda_eq_neg_one_l1122_112224

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) (lambda : ℤ) : ℤ := (n + 1)^2 + lambda

/-- The nth term of the sequence -/
def a (n : ℕ) (lambda : ℤ) : ℤ := S n lambda - S (n - 1) lambda

/-- A sequence is arithmetic if the difference between consecutive terms is constant -/
def is_arithmetic_sequence (lambda : ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a (n + 1) lambda - a n lambda = a n lambda - a (n - 1) lambda

theorem arithmetic_sequence_iff_lambda_eq_neg_one (lambda : ℤ) :
  is_arithmetic_sequence lambda ↔ lambda = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_iff_lambda_eq_neg_one_l1122_112224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_sales_l1122_112276

theorem store_sales (price : ℝ) (items_a : ℕ) (items_b : ℕ) : 
  price * (items_a : ℝ) = 7200 →
  0.8 * price * (items_b : ℝ) = 7200 →
  items_b = items_a + 15 →
  items_a = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_sales_l1122_112276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_a_b_l1122_112230

noncomputable section

variables (a b : ℝ)

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sqrt x + 3 else a * x + b

def unique_x₂ (x₁ : ℝ) : Prop :=
  x₁ ≠ 0 → ∃! x₂ : ℝ, x₂ ≠ x₁ ∧ f a b x₁ = f a b x₂

theorem sum_a_b 
  (h1 : ∀ x₁, unique_x₂ a b x₁)
  (h2 : f a b (2 * a) = f a b (3 * b)) :
  a + b = -Real.sqrt 6 / 2 + 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_a_b_l1122_112230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_APB_l1122_112293

/-- The line on which point P moves -/
def line (x y : ℝ) : Prop := x - y = 0

/-- The circle to which tangents are drawn -/
def circle_eq (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 2

/-- The angle APB formed by the tangents -/
noncomputable def angle_APB (P A B : ℝ × ℝ) : ℝ := sorry

theorem max_angle_APB :
  ∀ P A B : ℝ × ℝ,
  line P.1 P.2 →
  circle_eq A.1 A.2 →
  circle_eq B.1 B.2 →
  angle_APB P A B ≤ 60 * Real.pi / 180 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_APB_l1122_112293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pinecone_reduction_exists_l1122_112231

/-- Represents the state of pinecones and environmental factors for a single day -/
structure DailyState where
  pinecones : ℝ
  temperature : ℝ
  daylightHours : ℝ
deriving Inhabited

/-- Calculates the number of pinecones remaining after one day -/
def nextDay (state : DailyState) (a b c : ℝ) : ℝ :=
  let R := a * state.temperature^2 + b * state.temperature
  let S := 2 * R * (1 + c * state.daylightHours)
  let D := 0.3
  state.pinecones * (1 - R - S - D)

/-- Theorem stating the existence of constants a, b, and c that satisfy the problem conditions -/
theorem pinecone_reduction_exists :
  ∃ (a b c : ℝ) (states : List DailyState),
    states.length = 7 ∧
    (states.head!).pinecones = 5000 ∧
    (List.foldl (λ acc s => DailyState.mk (nextDay acc a b c) s.temperature s.daylightHours) 
      (states.head!) (states.tail)).pinecones = 1100 := by
  sorry

#check pinecone_reduction_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pinecone_reduction_exists_l1122_112231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_validity_l1122_112206

-- Define the types for lines and planes
structure Line where

structure Plane where

-- Define the relationships
def parallel_planes (α β : Plane) : Prop := sorry

def parallel_line_plane (m : Line) (α : Plane) : Prop := sorry

def perpendicular_planes (α β : Plane) : Prop := sorry

def perpendicular_line_plane (m : Line) (α : Plane) : Prop := sorry

def parallel_lines (m n : Line) : Prop := sorry

def line_in_plane (m : Line) (α : Plane) : Prop := sorry

-- Define the propositions
def proposition1 (α β γ : Plane) : Prop :=
  parallel_planes α β → parallel_planes α γ → parallel_planes β γ

def proposition2 (α β : Plane) (m : Line) : Prop :=
  perpendicular_planes α β → parallel_line_plane m α → perpendicular_line_plane m β

def proposition3 (α β : Plane) (m : Line) : Prop :=
  perpendicular_line_plane m α → parallel_line_plane m β → perpendicular_planes α β

def proposition4 (α : Plane) (m n : Line) : Prop :=
  parallel_lines m n → line_in_plane n α → parallel_line_plane m α

theorem propositions_validity (α β γ : Plane) (m n : Line) :
  proposition1 α β γ ∧
  ¬(∀ α β m, proposition2 α β m) ∧
  proposition3 α β m ∧
  ¬(∀ α m n, proposition4 α m n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_validity_l1122_112206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_last_three_terms_l1122_112232

def mySequence : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => if n % 2 = 0 then 2 * mySequence (n + 1) else mySequence n + 1

theorem sequence_last_three_terms :
  (mySequence 9 = 31) ∧ (mySequence 10 = 62) ∧ (mySequence 11 = 63) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_last_three_terms_l1122_112232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1122_112217

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the circle O
def circleO (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the conditions
def conditions (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ 2 * (a^2 - b^2).sqrt = 2 * Real.sqrt 2 ∧ 4 * a = 4 * Real.sqrt 3

-- Define the triangle area function
noncomputable def triangle_area (p m n : ℝ × ℝ) : ℝ :=
  let (px, py) := p
  let (mx, my) := m
  let (nx, ny) := n
  abs ((mx - px) * (ny - py) - (nx - px) * (my - py)) / 2

-- State the theorem
theorem max_triangle_area (a b : ℝ) (h : conditions a b) :
  ∃ (p m n : ℝ × ℝ),
    circleO p.1 p.2 ∧
    ellipse a b m.1 m.2 ∧
    ellipse a b n.1 n.2 ∧
    (∀ (p' m' n' : ℝ × ℝ),
      circleO p'.1 p'.2 →
      ellipse a b m'.1 m'.2 →
      ellipse a b n'.1 n'.2 →
      triangle_area p m n ≥ triangle_area p' m' n') ∧
    triangle_area p m n = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1122_112217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_flow_rate_l1122_112255

/-- Given a river with specified dimensions and flow volume, calculate its flow rate. -/
theorem river_flow_rate 
  (depth : ℝ) 
  (width : ℝ) 
  (volume_per_minute : ℝ) 
  (h_depth : depth = 5) 
  (h_width : width = 35) 
  (h_volume : volume_per_minute = 5833.333333333333) : 
  (volume_per_minute / (depth * width) / 1000 * 60) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_flow_rate_l1122_112255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_has_most_symmetry_l1122_112200

/-- Represents the number of lines of symmetry for a shape -/
inductive LinesOfSymmetry
| Finite : ℕ → LinesOfSymmetry
| Infinite : LinesOfSymmetry

/-- Compares two LinesOfSymmetry values -/
def linesOfSymmetryLt : LinesOfSymmetry → LinesOfSymmetry → Prop
| LinesOfSymmetry.Finite n, LinesOfSymmetry.Finite m => n < m
| LinesOfSymmetry.Finite _, LinesOfSymmetry.Infinite => True
| LinesOfSymmetry.Infinite, _ => False

infixl:50 " < " => linesOfSymmetryLt

/-- The number of lines of symmetry for each shape -/
def regularPentagonSymmetry : LinesOfSymmetry := LinesOfSymmetry.Finite 5
def isoscelesTriangleSymmetry : LinesOfSymmetry := LinesOfSymmetry.Finite 1
def parallelogramSymmetry : LinesOfSymmetry := LinesOfSymmetry.Finite 0
def nonEquilateralRhombusSymmetry : LinesOfSymmetry := LinesOfSymmetry.Finite 2
def circleSymmetry : LinesOfSymmetry := LinesOfSymmetry.Infinite

theorem circle_has_most_symmetry :
  regularPentagonSymmetry < circleSymmetry ∧
  isoscelesTriangleSymmetry < circleSymmetry ∧
  parallelogramSymmetry < circleSymmetry ∧
  nonEquilateralRhombusSymmetry < circleSymmetry :=
by
  apply And.intro
  · exact True.intro
  apply And.intro
  · exact True.intro
  apply And.intro
  · exact True.intro
  exact True.intro

#check circle_has_most_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_has_most_symmetry_l1122_112200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_specific_line_l1122_112281

/-- The distance from the origin to a line given by ax + by + c = 0 is |c| / √(a² + b²) -/
noncomputable def distanceFromOriginToLine (a b c : ℝ) : ℝ := |c| / Real.sqrt (a^2 + b^2)

/-- The line equation y = -1/2x + 5/2 can be written as x + 2y - 5 = 0 -/
def lineEquation (x y : ℝ) : Prop := x + 2*y - 5 = 0

theorem distance_to_specific_line :
  distanceFromOriginToLine 1 2 (-5) = Real.sqrt 5 := by
  -- Unfold the definition of distanceFromOriginToLine
  unfold distanceFromOriginToLine
  -- Simplify the expression
  simp [Real.sqrt_sq]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_specific_line_l1122_112281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1122_112243

noncomputable def a (n : ℕ) : ℝ := 3 * n + 1

noncomputable def S (n : ℕ) : ℝ := (n / 2 : ℝ) * (a 1 + a n)

noncomputable def b (n : ℕ) : ℝ := 2^n * (a n)

noncomputable def T (n : ℕ) : ℝ := (3 * n - 2) * 2^(n + 1) + 4

theorem sequence_properties (n : ℕ) :
  (∀ k, a k > 0) ∧
  (∀ k, (a k)^2 + 3 * (a k) = 6 * (S k) + 4) →
  (a n = 3 * n + 1) ∧
  (T n = (3 * n - 2) * 2^(n + 1) + 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1122_112243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_bounds_l1122_112222

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | (n + 1) => 1 + n / sequence_a n

theorem sequence_a_bounds (n : ℕ) : Real.sqrt n ≤ sequence_a n ∧ sequence_a n ≤ Real.sqrt n + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_bounds_l1122_112222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_blocks_fit_l1122_112287

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Represents the larger box -/
def largerBox : BoxDimensions :=
  { length := 4, width := 3, height := 3 }

/-- Represents the smaller block -/
def smallerBlock : BoxDimensions :=
  { length := 3, width := 2, height := 1 }

/-- The theorem stating the maximum number of smaller blocks that can fit in the larger box -/
theorem max_blocks_fit :
  (boxVolume largerBox) / (boxVolume smallerBlock) = 6 ∧
  ∃ (arrangement : List BoxDimensions),
    arrangement.length = 6 ∧
    ∀ block ∈ arrangement, block = smallerBlock ∧
    arrangement.all (fun b => b.length ≤ largerBox.length ∧
                              b.width ≤ largerBox.width ∧
                              b.height ≤ largerBox.height) :=
by sorry

#eval boxVolume largerBox
#eval boxVolume smallerBlock
#eval (boxVolume largerBox) / (boxVolume smallerBlock)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_blocks_fit_l1122_112287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l1122_112285

open Real
open BigOperators
open Matrix

def parallel (v w : Fin 3 → ℝ) : Prop := ∃ t : ℝ, v = fun i => t * w i

theorem vector_decomposition (a b : Fin 3 → ℝ) :
  (fun i => a i + b i) = ![5, 0, -10] →
  parallel a ![2, 2, 2] →
  (Matrix.dotProduct b ![2, 2, 2]) = 0 →
  b = ![25/3, 10/3, -20/3] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l1122_112285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carol_optimal_choice_l1122_112252

/-- Alice's choice is uniformly distributed between 1/4 and 3/4 -/
def alice_choice : Set ℝ := Set.Icc (1/4 : ℝ) (3/4 : ℝ)

/-- Bob's choice is uniformly distributed between 1/3 and 3/4 -/
def bob_choice : Set ℝ := Set.Icc (1/3 : ℝ) (3/4 : ℝ)

/-- Carol's winning condition: her number is between Alice's and Bob's -/
def carol_wins (a b c : ℝ) : Prop :=
  (a < c ∧ c < b) ∨ (b < c ∧ c < a)

/-- The probability of Carol winning given her choice c -/
noncomputable def prob_carol_wins (c : ℝ) : ℝ := sorry

/-- Theorem: Carol's optimal choice is 1/2 -/
theorem carol_optimal_choice :
  ∀ c, c ∈ Set.Icc (0 : ℝ) 1 → prob_carol_wins (1/2) ≥ prob_carol_wins c := by
  sorry

#check carol_optimal_choice

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carol_optimal_choice_l1122_112252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_points_theorem_l1122_112248

structure Point where
  x : ℝ
  y : ℝ

def collinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

def isParallelogram (p q r s : Point) : Prop :=
  (q.x - p.x = s.x - r.x ∧ q.y - p.y = s.y - r.y) ∨
  (r.x - p.x = s.x - q.x ∧ r.y - p.y = s.y - q.y)

theorem parallelogram_points_theorem (n : ℕ) (points : Fin n → Point)
    (h_n : n ≥ 4)
    (h_noncollinear : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → ¬collinear (points i) (points j) (points k))
    (h_parallelogram : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → 
      ∃ l, l ≠ i ∧ l ≠ j ∧ l ≠ k ∧ isParallelogram (points i) (points j) (points k) (points l)) :
  n = 4 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_points_theorem_l1122_112248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1122_112251

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x^3 + 3 * x^2 + 1 else Real.exp (a * x)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 3, f a x ≤ 2) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 3, f a x = 2) →
  a ≤ (1/3) * Real.log 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1122_112251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_value_l1122_112212

theorem trig_expression_value (α : Real) 
  (h1 : 0 < α) (h2 : α < π) (h3 : -Real.sin α = 2 * Real.cos α) : 
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_value_l1122_112212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_zero_value_l1122_112241

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the properties of f and g
axiom f_odd : ∀ x : ℝ, f x + f (-x) = 0
axiom g_periodic : ∀ x : ℝ, g x = g (x + 4)

-- State the given conditions
axiom f_neg_two : f (-2) = 6
axiom g_neg_two : g (-2) = 6
axiom given_equation : f (f 2 + g 2) + g (f (-2) + g (-2)) = -2 + 2 * g 4

-- State the theorem to be proved
theorem g_zero_value : g 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_zero_value_l1122_112241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_2000_l1122_112250

def series_sum (n : ℕ) : ℤ :=
  let groups := n / 4
  let remainder := n % 4
  let full_groups_sum := -4 * (groups : ℤ)
  let last_group_sum := match remainder with
    | 0 => 0
    | 1 => n
    | 2 => n + (n - 1)
    | 3 => n + (n - 1) - (n - 2)
    | _ => 0
  full_groups_sum + (last_group_sum : ℤ)

theorem series_sum_2000 :
  series_sum 2000 = -2000 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_2000_l1122_112250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_over_factorial_eq_power_of_three_l1122_112221

theorem factorial_sum_over_factorial_eq_power_of_three (x y n : ℕ) :
  x > 0 ∧ y > 0 ∧ n > 0 →
  (Nat.factorial x + Nat.factorial y) / Nat.factorial n = 3^n ↔ 
  ((x = 2 ∧ y = 1 ∧ n = 1) ∨ (x = 1 ∧ y = 2 ∧ n = 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_over_factorial_eq_power_of_three_l1122_112221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_c_sequence_l1122_112207

noncomputable section

def sequence_a : ℕ+ → ℝ := sorry
def sequence_b : ℕ+ → ℝ := sorry
def sequence_c : ℕ+ → ℝ := sorry

def S : ℕ+ → ℝ := sorry
def T : ℕ+ → ℝ := sorry

axiom S_def : ∀ n : ℕ+, S n = 2 * sequence_a n - sequence_a 1

axiom b_arithmetic : ∃ d : ℝ, 
  sequence_a 2 - sequence_a 1 = d ∧
  sequence_a 3 - sequence_a 2 = d

axiom c_def : ∀ n : ℕ+, sequence_c n = sequence_a n + sequence_b n

theorem sum_of_c_sequence : 
  ∀ n : ℕ+, T n = 2^((n : ℕ) + 1) - 2 + (n : ℝ)^2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_c_sequence_l1122_112207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_is_12_5_l1122_112239

-- Define the right triangle DEF
def triangle_DEF (D E F : ℝ × ℝ) : Prop :=
  D = (0, 0) ∧ E = (15, 0) ∧ F = (0, 20)

-- Define the median DM
noncomputable def median_DM (D M : ℝ × ℝ) : ℝ :=
  Real.sqrt ((M.1 - D.1)^2 + (M.2 - D.2)^2)

-- Theorem statement
theorem median_length_is_12_5 (D E F M : ℝ × ℝ) :
  triangle_DEF D E F →
  M = ((E.1 + F.1) / 2, (E.2 + F.2) / 2) →
  median_DM D M = 12.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_is_12_5_l1122_112239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_parts_for_99_lines_num_parts_less_than_199_possible_num_parts_l1122_112214

-- Define the number of lines
def num_lines : ℕ := 99

-- Define a function that represents the number of parts created by the lines
def num_parts : ℕ → ℕ 
| n => sorry  -- We'll leave this as sorry for now, as the actual implementation is not provided

-- Theorem stating the possible values of num_parts for 99 lines
theorem num_parts_for_99_lines :
  num_parts num_lines = 100 ∨ num_parts num_lines = 198 := by
  sorry

-- Theorem stating that the number of parts is less than 199
theorem num_parts_less_than_199 :
  num_parts num_lines < 199 := by
  sorry

-- Main theorem combining the above results
theorem possible_num_parts :
  num_parts num_lines = 100 ∨ num_parts num_lines = 198 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_parts_for_99_lines_num_parts_less_than_199_possible_num_parts_l1122_112214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_triangle_area_l1122_112245

-- Define the curves
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 16
def ellipse_equation (x y : ℝ) : Prop := (x-2)^2 + 4*y^2 = 36

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ circle_equation x y ∧ ellipse_equation x y}

-- Define the triangle area function
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

-- Theorem statement
theorem intersection_triangle_area :
  ∃ p1 p2 p3, p1 ∈ intersection_points ∧ 
              p2 ∈ intersection_points ∧ 
              p3 ∈ intersection_points ∧
              triangle_area p1 p2 p3 = 16 * Real.sqrt 80 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_triangle_area_l1122_112245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_from_tan_condition_l1122_112216

theorem sin_value_from_tan_condition (x : Real) :
  Real.tan x = Real.sin (x + π / 2) → Real.sin x = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_from_tan_condition_l1122_112216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1122_112283

noncomputable def f (x : ℝ) : ℝ := abs (Real.sin x) + 1 / Real.sin (abs x)

theorem f_properties :
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, x ∈ Set.Ioo (π/2) π → y ∈ Set.Ioo (π/2) π → x < y → f x < f y) ∧
  (∃ x y, x ∈ Set.Ioo (-π) (4*π) ∧ y ∈ Set.Ioo (-π) (4*π) ∧ x ≠ y ∧ f x = 0 ∧ f y = 0 ∧ 
   ∀ z, z ∈ Set.Ioo (-π) (4*π) → f z = 0 → z = x ∨ z = y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1122_112283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equals_interval_l1122_112295

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 1}
def B : Set ℝ := {x | Real.rpow 2 x > 1}

-- State the theorem
theorem intersection_complement_equals_interval :
  A ∩ (Set.univ \ B) = Set.Ioc (-2) 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equals_interval_l1122_112295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_minimum_perimeter_l1122_112205

theorem triangle_minimum_perimeter (a b c : ℝ) (A B C : ℝ) : 
  C = max A (max B C) →
  Real.sin C = 1 + Real.cos C * Real.cos (A - B) →
  2 / a + 1 / b = 1 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = Real.pi →
  c^2 = a^2 + b^2 →
  (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 → 
    2 / a' + 1 / b' = 1 → c'^2 = a'^2 + b'^2 → a' + b' + c' ≥ 10) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_minimum_perimeter_l1122_112205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_sum_l1122_112223

def A (b : ℤ) : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, 3, b],
    ![0, 1, 5],
    ![0, 0, 1]]

theorem matrix_power_sum (b m : ℤ) :
  (∃ (A_pow_m : Matrix (Fin 3) (Fin 3) ℤ),
    A_pow_m = (A b) ^ m ∧
    A_pow_m = ![![1, 27, 3050],
                ![0, 1, 45],
                ![0, 0, 1]]) →
  b + m = 287 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_sum_l1122_112223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1122_112211

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Distance from a point to a horizontal line -/
def distanceToLine (p : Point) (y : ℝ) : ℝ :=
  |p.y - y|

/-- The focus point of the parabola -/
def focus : Point := ⟨0, -2⟩

/-- The y-coordinate of the directrix -/
def directrix : ℝ := 3

/-- Theorem stating the equation of the parabola -/
theorem parabola_equation (P : Point) : 
  distanceToLine P directrix = distance P focus + 1 →
  P.x^2 = -8 * P.y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1122_112211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_necessarily_perpendicular_planes_l1122_112258

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (perpendicular parallel : Line → Line → Prop)
variable (planePerpendicular planeParallel : Plane → Plane → Prop)
variable (intersect : Line → Line → Line → Prop)

-- Define the given planes and lines
variable (α β r : Plane) (a b c d l : Line)

-- State the theorem
theorem not_necessarily_perpendicular_planes 
  (h1 : subset a α) 
  (h2 : subset b α) 
  (h3 : subset c β) 
  (h4 : subset d β) 
  (h5 : ∃ A, intersect a b A) 
  (h6 : ∃ B, intersect c d B) 
  (h7 : perpendicular a c) 
  (h8 : perpendicular b d) :
  ¬(planePerpendicular α β → True) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_necessarily_perpendicular_planes_l1122_112258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l1122_112289

noncomputable section

/-- The line equation y = (x + 3) / 3 --/
def line_equation (x : ℝ) : ℝ := (x + 3) / 3

/-- The point we want to find the closest point to --/
def target_point : ℝ × ℝ := (3, 2)

/-- The proposed closest point on the line --/
def proposed_closest_point : ℝ × ℝ := (3, 1)

/-- Theorem stating that the proposed point is indeed the closest point on the line to the target point --/
theorem closest_point_on_line :
  ∀ (x : ℝ), 
  (x - target_point.1)^2 + (line_equation x - target_point.2)^2 ≥ 
  (proposed_closest_point.1 - target_point.1)^2 + (proposed_closest_point.2 - target_point.2)^2 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l1122_112289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1122_112274

-- Define the properties of the function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def monotone_increasing_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

def exists_negative (f : ℝ → ℝ) : Prop := ∃ x, f x < 0

-- Define the two specific functions
def f1 (x : ℝ) : ℝ := x^2 - 3

noncomputable def f2 (x : ℝ) : ℝ := Real.log (abs x) / Real.log 2

-- State the theorem
theorem function_properties :
  ∃ f : ℝ → ℝ, is_even f ∧ monotone_increasing_on_positive f ∧ exists_negative f ∧
  (f = f1 ∨ f = f2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1122_112274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_in_still_water_l1122_112203

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water (current_speed downstream_distance downstream_time boat_speed : ℝ) :
  current_speed = 3 ∧
  downstream_distance = 3.6 ∧
  downstream_time = 1/5 →
  boat_speed = 15 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_in_still_water_l1122_112203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bailing_rate_bounds_l1122_112284

/-- Represents the fishing scenario with Amy and Jake --/
structure FishingScenario where
  distance_to_shore : ℝ
  water_intake_rate : ℝ
  max_water_capacity : ℝ
  rowing_speed : ℝ

/-- Calculates the minimum bailing rate required to reach shore without sinking --/
noncomputable def min_bailing_rate (scenario : FishingScenario) : ℝ :=
  let time_to_shore := scenario.distance_to_shore / scenario.rowing_speed
  let total_water_intake := scenario.water_intake_rate * time_to_shore
  let excess_water := total_water_intake - scenario.max_water_capacity
  excess_water / time_to_shore

/-- Theorem stating the minimum bailing rate is between 8 and 9 gallons per minute --/
theorem min_bailing_rate_bounds (scenario : FishingScenario) 
  (h1 : scenario.distance_to_shore = 1.5)
  (h2 : scenario.water_intake_rate = 10)
  (h3 : scenario.max_water_capacity = 40)
  (h4 : scenario.rowing_speed = 3) :
  8 < min_bailing_rate scenario ∧ min_bailing_rate scenario ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bailing_rate_bounds_l1122_112284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_result_l1122_112225

theorem election_result (total_voters : ℝ) (dem_for_x : ℝ) : 
  total_voters > 0 →
  let rep := (3/5) * total_voters
  let dem := (2/5) * total_voters
  let votes_for_x := 0.7 * rep + (dem_for_x / 100) * dem
  votes_for_x / total_voters = 0.539999999999999853 →
  abs (dem_for_x - 30) < 0.00001 := by
  sorry

#eval (30 : Float)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_result_l1122_112225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_CMKD_is_nineteen_fortieths_l1122_112213

/-- Represents a parallelogram ABCD with point M on BC and K on BD -/
structure Parallelogram where
  -- Vectors representing sides of the parallelogram
  b : ℝ × ℝ
  c : ℝ × ℝ
  -- Point M divides BC in ratio 1:3
  m : ℝ × ℝ
  m_def : m = b + (1/4 : ℝ) • (c - b)
  -- Point K is intersection of AM and BD
  k : ℝ × ℝ
  k_def : k = (1/2 : ℝ) • b + (1/8 : ℝ) • c
  -- Area of parallelogram is 1
  area_is_one : abs (b.1 * c.2 - b.2 * c.1) = 1

/-- The area of quadrilateral CMKD in the given parallelogram configuration -/
noncomputable def area_CMKD (p : Parallelogram) : ℝ := 19/40

/-- Theorem stating that the area of CMKD is 19/40 -/
theorem area_CMKD_is_nineteen_fortieths (p : Parallelogram) : 
  area_CMKD p = 19/40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_CMKD_is_nineteen_fortieths_l1122_112213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_sum_mn_equals_two_l1122_112299

def f (x m : ℝ) : ℝ := x^3 + x + m

theorem odd_function_sum_mn_equals_two
  (m n : ℝ)
  (h_odd : ∀ x, x ∈ Set.Icc (-2 - n) (2 * n) → f (-x) m = -f x m)
  (h_domain : Set.Nonempty (Set.Icc (-2 - n) (2 * n))) :
  m + n = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_sum_mn_equals_two_l1122_112299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_perpendicular_lines_l1122_112236

-- Define the points and lines as functions of m and n
def A (m : ℝ) : ℝ × ℝ := (-2, m)
def B (m : ℝ) : ℝ × ℝ := (m, 4)

noncomputable def l1_slope (m : ℝ) : ℝ := (4 - m) / (m + 2)
def l2_slope : ℝ := -2
noncomputable def l3_slope (n : ℝ) : ℝ := -1 / n

-- State the theorem
theorem parallel_perpendicular_lines (m n : ℝ) : 
  l1_slope m = l2_slope ∧ l2_slope * l3_slope n = -1 → m + n = -10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_perpendicular_lines_l1122_112236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1122_112219

-- Define the rational function f
noncomputable def f (x : ℝ) : ℝ := (x^3 - 3*x^2 + 6*x - 8) / (x^2 - 5*x + 6)

-- Define the domain of f
def domain_f : Set ℝ := {x | x ≠ 2 ∧ x ≠ 3}

-- Theorem stating that the domain of f is (-∞, 2) ∪ (2, 3) ∪ (3, ∞)
theorem domain_of_f : domain_f = Set.Iio 2 ∪ Set.Ioo 2 3 ∪ Set.Ioi 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1122_112219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_triangle_area_l1122_112229

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3) + Real.sqrt 3 / 2

-- Theorem for the range of g(x)
theorem g_range :
  ∀ x ∈ Set.Icc 0 (Real.pi / 2),
  g x ∈ Set.Icc 0 (Real.sqrt 3 / 2 + 1) := by
  sorry

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = Real.pi

-- Theorem for the area of triangle ABC
theorem triangle_area (abc : Triangle)
  (h1 : f abc.A = Real.sqrt 3 / 2)
  (h2 : abc.a = 4)
  (h3 : abc.b + abc.c = 5) :
  (abc.b * abc.c * Real.sin abc.A) / 2 = 9 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_triangle_area_l1122_112229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_undefined_at_one_l1122_112263

noncomputable def f (x : ℝ) : ℝ := (x - 5) / (x - 6)

-- State the theorem
theorem inverse_f_undefined_at_one :
  ∀ x : ℝ, x ≠ 6 → (∃ y : ℝ, f y = x) → x ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_undefined_at_one_l1122_112263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l1122_112280

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 1)^2 + (y + 1)^2) - Real.sqrt ((x - 7)^2 + (y + 1)^2) = 4

/-- The positive slope of the asymptote -/
noncomputable def positive_asymptote_slope : ℝ := Real.sqrt 5 / 2

/-- Theorem: The positive slope of the asymptote of the given hyperbola is √5/2 -/
theorem hyperbola_asymptote_slope :
  ∀ x y : ℝ, hyperbola_equation x y →
  positive_asymptote_slope = Real.sqrt 5 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l1122_112280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_typhoon_damage_conversion_l1122_112288

/-- Converts Australian dollars to American dollars based on the given exchange rate -/
noncomputable def convert_to_usd (aud : ℝ) (exchange_rate : ℝ) : ℝ := aud / exchange_rate

/-- Theorem stating the damage conversion from Australian to American dollars -/
theorem typhoon_damage_conversion (damage_aud : ℝ) (exchange_rate : ℝ) 
  (h1 : damage_aud = 45000000)
  (h2 : exchange_rate = 2) :
  convert_to_usd damage_aud exchange_rate = 22500000 := by
  sorry

#check typhoon_damage_conversion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_typhoon_damage_conversion_l1122_112288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_two_zeros_l1122_112237

/-- A function f with the given form -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(x-1) - Real.log x - a

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a_for_two_zeros :
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧
   ∀ z : ℝ, f a z = 0 → z = x ∨ z = y) →
  a > 1 ∧ ∀ b > 1, ∃ x y : ℝ, x ≠ y ∧ f b x = 0 ∧ f b y = 0 ∧
   ∀ z : ℝ, f b z = 0 → z = x ∨ z = y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_two_zeros_l1122_112237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l1122_112286

/-- Curve C in polar coordinates -/
noncomputable def curve_C (θ : ℝ) : ℝ := 2 * Real.sin θ

/-- Line l in rectangular coordinates -/
noncomputable def line_l (x : ℝ) : ℝ := -Real.sqrt 3 * x + 5

/-- Theorem stating the rectangular equation of curve C and the point with shortest distance to line l -/
theorem curve_C_properties :
  (∀ x y : ℝ, x^2 + (y-1)^2 = 1 ↔ ∃ θ : ℝ, 0 ≤ θ ∧ θ < 2*Real.pi ∧ x = curve_C θ * Real.cos θ ∧ y = curve_C θ * Real.sin θ) ∧
  (∀ x y : ℝ, x^2 + (y-1)^2 = 1 → 
    (x - Real.sqrt 3 / 2)^2 + (y - 3/2)^2 ≤ (x - x')^2 + (y - y')^2 ∨ 
    (x' - Real.sqrt 3 / 2)^2 + (y' - 3/2)^2 < (x - x')^2 + (y - y')^2) :=
by
  sorry

where
  x' : ℝ → ℝ := id
  y' : ℝ → ℝ := line_l

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l1122_112286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1122_112264

theorem triangle_problem (A B C : Real) (a b c : Real) :
  0 < C → C < Real.pi / 2 →
  Real.cos (2 * C) = -1 / 4 →
  a = 2 →
  2 * Real.sin A = Real.sin C →
  Real.cos C = Real.sqrt 6 / 4 ∧ c = 4 ∧ b = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1122_112264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_ratio_is_one_l1122_112233

/-- Represents a truncated right circular cone with an inscribed sphere -/
structure TruncatedConeWithSphere where
  R : ℝ  -- radius of the bottom base
  r : ℝ  -- radius of the top base
  s : ℝ  -- radius of the inscribed sphere
  H : ℝ  -- height of the truncated cone

/-- The volume of a sphere -/
noncomputable def sphereVolume (radius : ℝ) : ℝ := (4 / 3) * Real.pi * radius ^ 3

/-- The volume of a truncated cone -/
noncomputable def truncatedConeVolume (cone : TruncatedConeWithSphere) : ℝ :=
  (Real.pi * cone.H / 3) * (cone.R ^ 2 + cone.R * cone.r + cone.r ^ 2)

/-- Theorem stating the ratio of radii is 1 under given conditions -/
theorem truncated_cone_ratio_is_one (cone : TruncatedConeWithSphere) :
    truncatedConeVolume cone = 3 * sphereVolume cone.s →
    cone.H = 4 * cone.s →
    cone.R = cone.r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_ratio_is_one_l1122_112233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_value_theorem_l1122_112267

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^m + Real.log x

theorem m_value_theorem (m : ℝ) :
  (∀ x > 0, f m x = x^m + Real.log x) →
  (∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |((f m (1 + 2*Δx) - f m 1) / Δx) + 2| < ε) →
  m = -2 := by
  sorry

#check m_value_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_value_theorem_l1122_112267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l1122_112296

theorem polynomial_divisibility (a b c d e : ℚ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e)
  (f : ℚ → ℚ) (g : ℚ → ℚ)
  (hf : ∀ x, f x = a * x^2 + b * x + c)
  (hg : ∀ x, g x = d * x + e)
  (h_int : ∀ n : ℕ, ∃ k : ℤ, f n / g n = k) :
  ∃ q : ℚ → ℚ, ∀ x, f x = g x * q x := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l1122_112296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_max_area_l1122_112265

-- Define the circles F₁ and F₂
def F₁ (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 49
def F₂ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 / 9 + y^2 / 5 = 1

-- Define the point Q on curve C not on the x-axis
def Q (x y : ℝ) : Prop := C x y ∧ y ≠ 0

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the center of F₂
def F₂_center : ℝ × ℝ := (2, 0)

-- Define a function to calculate the area of a triangle
noncomputable def area_triangle (p q r : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := p
  let (x₂, y₂) := q
  let (x₃, y₃) := r
  (1/2) * abs ((x₁ - x₃) * (y₂ - y₃) - (x₂ - x₃) * (y₁ - y₃))

theorem curve_C_and_max_area :
  ∃ (P : ℝ × ℝ → Prop) (M N : ℝ × ℝ),
    (∀ x y, P (x, y) → F₁ x y ∨ F₂ x y) ∧  -- P is tangent to F₁ and internally tangent to F₂
    (∀ x y, C x y ↔ ∃ t, P (x, y) = P (t, 0)) ∧  -- C is the trajectory of the center of P
    (∀ x y, Q x y → 
      ∃ m n, C m n ∧ C x n ∧ 
      (n - F₂_center.2) / (m - F₂_center.1) = (y - O.2) / (x - O.1) ∧
      m ≠ x) →  -- Line through F₂ parallel to OQ intersects C at M and N
    (∀ x y, C x y ↔ x^2 / 9 + y^2 / 5 = 1) ∧  -- Equation of curve C
    (∃ S : ℝ, 
      (∀ x y m n, Q x y → C m n → C x n → 
        (n - F₂_center.2) / (m - F₂_center.1) = (y - O.2) / (x - O.1) → 
        m ≠ x → 
        area_triangle (x, y) (m, n) (x, n) ≤ S) ∧
      S = 30 / 9) -- Maximum area of triangle QMN
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_max_area_l1122_112265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_folded_area_l1122_112273

/-- Represents a triangular piece of paper -/
structure Triangle where
  area : ℝ

/-- Represents the result of folding a triangle -/
structure FoldedTriangle where
  original : Triangle
  foldDistance : ℝ  -- Normalized distance from 0 to 1

/-- Calculate the area of the folded triangle -/
noncomputable def foldedArea (ft : FoldedTriangle) : ℝ :=
  if ft.foldDistance ≤ 1/2 then
    1 - ft.foldDistance^2
  else
    3 * ft.foldDistance^2 - 4 * ft.foldDistance + 2

/-- The main theorem stating the minimum area after folding -/
theorem min_folded_area (t : Triangle) (h : t.area = 1) :
  ∃ (minArea : ℝ), minArea = 2/3 ∧
  ∀ (ft : FoldedTriangle), ft.original = t → foldedArea ft ≥ minArea := by
  sorry

#check min_folded_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_folded_area_l1122_112273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1122_112240

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2 / (x - 1)

-- Define the domain
def domain : Set ℝ := Set.Iio 1 ∪ (Set.Ici 2 ∩ Set.Iio 5)

-- State the theorem
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = Set.Iio 0 ∪ Set.Ioo (1/2) 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1122_112240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l1122_112204

def M : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 5}
def N : Set ℝ := {y : ℝ | -5 < y ∧ y < 5}

theorem intersection_of_M_and_N : M ∩ N = Set.Ioo (-3) 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l1122_112204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dap_equivalent_to_dips_l1122_112271

/-- Represents the number of daps -/
def Dap : Type := ℚ

/-- Represents the number of dops -/
def Dop : Type := ℚ

/-- Represents the number of dips -/
def Dip : Type := ℚ

/-- Conversion rate between daps and dops -/
def dap_dop_rate : ℚ := 8 / 6

/-- Conversion rate between dops and dips -/
def dop_dip_rate : ℚ := 3 / 11

/-- The problem statement -/
theorem dap_equivalent_to_dips :
  ∀ (d : ℚ),
  d = 66 →
  d * (dop_dip_rate * dap_dop_rate) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dap_equivalent_to_dips_l1122_112271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l1122_112209

/-- Represents a length in inches as a real number -/
structure RealInches where
  value : ℝ

/-- Represents an angle in radians -/
structure Angle where
  value : ℝ

/-- Represents a right triangle -/
def IsRightTriangle : (Set ℝ × Set ℝ) → Prop := sorry

/-- Calculates the area of a triangle -/
def TriangleArea : (Set ℝ × Set ℝ) → ℝ := sorry

/-- The area of a right triangle with hypotenuse 12 and one angle 30° is 18√3 -/
theorem right_triangle_area (triangle : Set ℝ × Set ℝ) 
  (is_right_triangle : IsRightTriangle triangle)
  (hypotenuse : RealInches)
  (angle : Angle)
  (h1 : hypotenuse.value = 12)
  (h2 : angle.value = 30 * π / 180) : 
  TriangleArea triangle = 18 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l1122_112209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l1122_112298

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x^2 + 1 > 0)) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
by
  apply Iff.intro
  · intro h
    push_neg at h
    exact h
  · intro h
    push_neg
    exact h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l1122_112298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l1122_112266

/-- The circle C: x^2 + y^2 - 4y + 3 = 0 -/
def Circle (x y : ℝ) : Prop := x^2 + y^2 - 4*y + 3 = 0

/-- The line l: x - √3*y + 3*√3 = 0 -/
def Line (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 3 * Real.sqrt 3 = 0

/-- The maximum distance from any point on the circle to the line -/
noncomputable def MaxDistance : ℝ := Real.sqrt 3 / 2 + 1

theorem max_distance_circle_to_line :
  ∀ (x y : ℝ), Circle x y →
  (∃ (d : ℝ), ∀ (x' y' : ℝ), Circle x' y' →
    d ≥ abs (x' - Real.sqrt 3 * y' + 3 * Real.sqrt 3) / Real.sqrt (1 + 3)) ∧
  (∃ (x₀ y₀ : ℝ), Circle x₀ y₀ ∧
    abs (x₀ - Real.sqrt 3 * y₀ + 3 * Real.sqrt 3) / Real.sqrt (1 + 3) = MaxDistance) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l1122_112266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_triangle_abc_l1122_112220

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + Real.sqrt 3 * Real.sin (2 * x) - 2

-- Theorem for the minimum value of f
theorem f_min_value :
  ∃ (m : ℝ), m = -2 ∧
  ∀ (x : ℝ), -π/6 ≤ x ∧ x ≤ π/3 → f x ≥ m :=
by sorry

-- Theorem for triangle ABC
theorem triangle_abc (a b c : ℝ) (A B C : ℝ) :
  f C = 1 →
  c = 1 →
  a * b = 2 * Real.sqrt 3 →
  a > b →
  a = 2 ∧ b = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_triangle_abc_l1122_112220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fodder_consumption_equation_l1122_112208

/-- Represents the daily fodder consumption for big cows -/
def big_cow_fodder : ℝ → Prop := sorry

/-- Represents the daily fodder consumption for small cows -/
def small_cow_fodder : ℝ → Prop := sorry

/-- The total number of big cows -/
def num_big_cows : ℕ := 30

/-- The total number of small cows -/
def num_small_cows : ℕ := 15

/-- The total daily fodder consumption for all cows -/
def total_fodder : ℝ := 675

/-- Theorem stating that if m and n represent the daily fodder consumption
    for big and small cows respectively, they must satisfy the equation -/
theorem fodder_consumption_equation (m n : ℝ) :
  big_cow_fodder m ∧ small_cow_fodder n →
  num_big_cows * m + num_small_cows * n = total_fodder :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fodder_consumption_equation_l1122_112208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_more_wins_than_draws_l1122_112262

/-- Represents a chess player in the tournament -/
structure Player where
  wins : Nat
  draws : Nat
  losses : Nat

/-- Calculates the score of a player -/
def score (p : Player) : Rat :=
  p.wins + p.draws / 2

/-- The chess tournament -/
structure Tournament where
  players : Finset Player
  player_count : players.card = 20
  all_games_played : ∀ p ∈ players, p.wins + p.draws + p.losses = 19
  score_difference : ∀ p q, p ∈ players → q ∈ players → p ≠ q → score p ≠ score q

theorem exists_more_wins_than_draws (t : Tournament) :
  ∃ p, p ∈ t.players ∧ p.wins > p.draws := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_more_wins_than_draws_l1122_112262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1122_112242

/-- Calculates the length of a train given its speed and time to cross a pole. -/
noncomputable def trainLength (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Converts speed from km/hr to m/s. -/
noncomputable def convertSpeed (speed : ℝ) : ℝ :=
  speed * (1000 / 3600)

theorem train_length_calculation (speed : ℝ) (time : ℝ) 
  (h1 : speed = 180) 
  (h2 : time = 18) : 
  trainLength (convertSpeed speed) time = 900 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval trainLength (convertSpeed 180) 18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1122_112242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_N_not_perfect_square_l1122_112202

/-- An integer with 300 ones and the remaining digits as zeros in its decimal representation -/
def N : ℕ := (10^300 - 1) / 9

/-- Theorem: N cannot be a perfect square -/
theorem N_not_perfect_square : ¬ ∃ m : ℕ, N = m^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_N_not_perfect_square_l1122_112202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_difference_l1122_112282

def jo_sum : ℕ := (100 * 101) / 2

def round_to_5 (n : ℕ) : ℕ :=
  if n % 5 < 3 then (n / 5) * 5 else ((n / 5) + 1) * 5

def kate_sum : ℕ := (List.range 100).map (λ i => round_to_5 (i + 1)) |>.sum

theorem sum_difference : |Int.ofNat jo_sum - Int.ofNat kate_sum| = 3550 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_difference_l1122_112282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_properties_l1122_112234

-- Define a right triangle with hypotenuse 3
def RightTriangle (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a^2 + b^2 = 9

-- Define the variance of the side lengths
noncomputable def Variance (a b : ℝ) : ℝ :=
  6 - ((a + b + 3) / 3)^2

-- Define the standard deviation of the side lengths
noncomputable def StandardDeviation (a b : ℝ) : ℝ :=
  Real.sqrt (Variance a b)

theorem right_triangle_properties :
  ∀ a b : ℝ,
  RightTriangle a b →
  (Variance a b < 5) ∧
  (∀ x y : ℝ, RightTriangle x y → StandardDeviation a b ≤ StandardDeviation x y) ∧
  (StandardDeviation a b = Real.sqrt 2 - 1) ∧
  (a = 3 * Real.sqrt 2 / 2) ∧
  (b = 3 * Real.sqrt 2 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_properties_l1122_112234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_n_reachable_l1122_112235

/-- Represents a line segment on a plane --/
structure Segment where
  start : ℕ
  end_ : ℕ

/-- Represents the configuration of line segments on a plane --/
structure SegmentConfiguration where
  n : ℕ
  segments : List Segment
  no_triple_intersection : Bool
  all_pairs_intersect : Bool
  distinct_endpoints : Bool

/-- Represents a direction assignment for each segment --/
def DirectionAssignment := List Bool

/-- Function to check if an endpoint is reachable given a direction assignment --/
def is_reachable (config : SegmentConfiguration) (direction : DirectionAssignment) (endpoint : ℕ) : Bool :=
  sorry

/-- Main theorem: There exists a direction assignment such that exactly n endpoints are reachable --/
theorem exactly_n_reachable (config : SegmentConfiguration) :
  ∃ (direction : DirectionAssignment),
    (List.sum (List.map (fun i => if is_reachable config direction i then 1 else 0) (List.range (2 * config.n)))) = config.n :=
  sorry

#check exactly_n_reachable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_n_reachable_l1122_112235
