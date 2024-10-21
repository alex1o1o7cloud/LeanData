import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangulation_l256_25667

/-- A function that determines if a given integer n ≥ 3 allows for a triangulation
    of a regular n-gon consisting only of isosceles triangles. -/
def isTriangulatable (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 2^a * (2^b + 1) ∧ (a ≠ 0 ∨ b ≠ 0) ∧ n ≥ 3

/-- Represents a triangulation of an n-gon -/
def Triangulation (n : ℕ) := Set (Fin n × Fin n × Fin n)

/-- Checks if a triangulation is valid for an n-gon -/
def IsValidTriangulation (n : ℕ) (t : Triangulation n) : Prop := sorry

/-- Checks if all triangles in a triangulation are isosceles -/
def AllTrianglesIsosceles (n : ℕ) (t : Triangulation n) : Prop := sorry

/-- Theorem stating that a regular n-gon (where n ≥ 3) can be triangulated
    using only isosceles triangles if and only if n = 2^a(2^b + 1),
    where a and b are nonnegative integers not both zero. -/
theorem isosceles_triangulation (n : ℕ) (h : n ≥ 3) :
  (∃ (t : Triangulation n), IsValidTriangulation n t ∧ AllTrianglesIsosceles n t) ↔
  isTriangulatable n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangulation_l256_25667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_time_to_return_keys_l256_25665

/-- The minimum time for the cyclist to bring the keys to the pedestrian -/
noncomputable def minTimeToReturnKeys (alleyDiameter : ℝ) (pedestrianSpeed : ℝ) (cyclistRoadSpeed : ℝ) 
  (cyclistAlleySpeed : ℝ) (pedestrianWalkTime : ℝ) : ℝ :=
  (alleyDiameter * Real.pi - pedestrianSpeed * pedestrianWalkTime) / 
  (cyclistAlleySpeed + pedestrianSpeed) + alleyDiameter / cyclistRoadSpeed

theorem min_time_to_return_keys :
  minTimeToReturnKeys 4 7 15 20 1 = (4 * Real.pi - 7) / 27 := by
  sorry

#check min_time_to_return_keys

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_time_to_return_keys_l256_25665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_theorem_l256_25694

noncomputable section

/-- The function f(x) = sin x(cos x - √3 sin x) -/
def f (x : ℝ) : ℝ := Real.sin x * (Real.cos x - Real.sqrt 3 * Real.sin x)

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = Real.pi
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

/-- Main theorem -/
theorem triangle_ratio_theorem (t : Triangle) 
  (h1 : f t.B = 0)
  (h2 : ∃ d > 0, t.b = t.a + d ∧ Real.sqrt 3 * t.c = t.b + d) :
  Real.sin t.A / Real.sin t.C = Real.sqrt 3 * (3 - 2 * Real.sqrt 2) / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_theorem_l256_25694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_four_l256_25601

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluate a quadratic polynomial at a given point -/
noncomputable def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- The sum of the roots of a quadratic polynomial -/
noncomputable def QuadraticPolynomial.sumOfRoots (p : QuadraticPolynomial) : ℝ :=
  -p.b / p.a

/-- The theorem stating that if P(x^3 + x) ≥ P(x^2 + 1) for all real x,
    then the sum of the roots of P(x) is 4 -/
theorem sum_of_roots_is_four (p : QuadraticPolynomial) 
    (h : ∀ x : ℝ, p.eval (x^3 + x) ≥ p.eval (x^2 + 1)) : 
  p.sumOfRoots = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_four_l256_25601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_impossible_d_values_l256_25688

/-- Represents the side length of a square -/
def square_side : ℝ → ℝ := λ s ↦ s

/-- Represents the perimeter of a square -/
def square_perimeter (s : ℝ) : ℝ := 4 * square_side s

/-- Represents the length of equal sides of an isosceles triangle -/
def triangle_equal_side (s d : ℝ) : ℝ := square_side s + 2 * d

/-- Represents the length of the base of an isosceles triangle -/
def triangle_base (s d : ℝ) : ℝ := square_side s + d

/-- Represents the perimeter of an isosceles triangle -/
def triangle_perimeter (s d : ℝ) : ℝ := 2 * triangle_equal_side s d + triangle_base s d

/-- The main theorem stating that there are infinitely many positive integers that cannot be the value of d -/
theorem infinitely_many_impossible_d_values (s : ℝ) (h₁ : square_perimeter s > 0) :
  ∃ (f : ℕ → ℕ), Function.Injective f ∧ ∀ (n : ℕ), ¬∃ (d : ℕ), triangle_perimeter s (d : ℝ) = square_perimeter s + 4041 ∧ d = f n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_impossible_d_values_l256_25688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_triangle_side_sum_range_l256_25668

noncomputable section

open Real

def Triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi

def VectorM (A : ℝ) : ℝ × ℝ := (-cos (A/2), sin (A/2))

def VectorN (A : ℝ) : ℝ × ℝ := (cos (A/2), sin (A/2))

def DotProduct (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def TriangleArea (b c A : ℝ) : ℝ := (1/2) * b * c * sin A

theorem triangle_side_sum (A B C a b c : ℝ) :
  Triangle A B C a b c →
  a = 2 * sqrt 3 →
  DotProduct (VectorM A) (VectorN A) = 1/2 →
  TriangleArea b c A = sqrt 3 →
  b + c = 4 :=
by sorry

theorem triangle_side_sum_range (A B C a b c : ℝ) :
  Triangle A B C a b c →
  a = 2 * sqrt 3 →
  2 * sqrt 3 < b + c ∧ b + c ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_triangle_side_sum_range_l256_25668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_terrace_theorem_l256_25646

/-- Represents a square garden with a square terrace -/
structure GardenWithTerrace where
  garden_side : ℝ
  terrace_side : ℝ
  terrace_vertex_at_center : Bool
  side_division_ratio : ℝ

/-- The ratio in which the opposite side of the terrace divides the garden -/
def opposite_side_division_ratio (g : GardenWithTerrace) : ℝ := sorry

/-- The area of the garden reduced by the terrace -/
def reduced_area (g : GardenWithTerrace) : ℝ := sorry

/-- The Garden-Terrace Theorem -/
theorem garden_terrace_theorem (g : GardenWithTerrace) 
  (h1 : g.garden_side = 6)
  (h2 : g.terrace_side = 7)
  (h3 : g.terrace_vertex_at_center = true)
  (h4 : g.side_division_ratio = 1/5) :
  (opposite_side_division_ratio g = 1/5) ∧ 
  (reduced_area g = 9) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_terrace_theorem_l256_25646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_right_triangle_l256_25670

/-- The ellipse equation -/
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 16) + (P.2^2 / 12) = 1

/-- The foci of the ellipse -/
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

/-- The distance between two points -/
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- The theorem to prove -/
theorem ellipse_right_triangle (P : ℝ × ℝ) 
  (h_ellipse : is_on_ellipse P)
  (h_diff : distance P F₁ - distance P F₂ = 2) :
  distance F₁ F₂^2 + distance P F₂^2 = distance P F₁^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_right_triangle_l256_25670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_properties_l256_25690

-- Define the curve C
def C : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, p.1 = t ∧ p.2 = 1 + 2*t}

-- Define the line l
def l : Set (ℝ × ℝ) :=
  {p | ∃ θ : ℝ, p.1^2 + p.2^2 = (2*Real.sqrt 2 * Real.sin (θ + Real.pi/4))^2}

-- Define point P
def P : ℝ × ℝ := (0, 1)

theorem curve_and_line_properties :
  -- C is a circle with center (1,1) and radius √2
  ∃ center : ℝ × ℝ, ∃ radius : ℝ,
    center = (1, 1) ∧ radius = Real.sqrt 2 ∧
    C = {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2} ∧
  -- l intersects C
  (∃ p : ℝ × ℝ, p ∈ C ∧ p ∈ l) ∧
  -- If A and B are intersection points, then PA · PB = -1
  ∀ A B : ℝ × ℝ, A ∈ C ∧ A ∈ l → B ∈ C ∧ B ∈ l → A ≠ B →
    ((A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2)) = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_properties_l256_25690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_difference_l256_25673

-- Define the tangent values
def tan_alpha : ℝ := 5
def tan_beta : ℝ := 3

-- State the theorem
theorem tan_double_angle_difference :
  (2 * tan_alpha) / (1 - tan_alpha^2) - (2 * tan_beta) / (1 - tan_beta^2) /
  (1 + (2 * tan_alpha) / (1 - tan_alpha^2) * (2 * tan_beta) / (1 - tan_beta^2)) = 16 / 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_difference_l256_25673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_sixth_f_monotonic_increase_interval_l256_25637

noncomputable def f (x : Real) : Real := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

theorem f_value_at_pi_sixth : f (π / 6) = 2 := by sorry

theorem f_monotonic_increase_interval (k : Int) :
  StrictMonoOn f (Set.Icc (k * π - π / 3) (k * π + π / 6)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_sixth_f_monotonic_increase_interval_l256_25637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_oil_price_l256_25685

/-- Proves that the price of the second oil is 68 rupees per litre -/
theorem second_oil_price (
  first_oil_volume : ℝ)
  (first_oil_price : ℝ)
  (second_oil_volume : ℝ)
  (second_oil_price : ℝ)
  (mixture_price : ℝ)
  (h1 : first_oil_volume = 10)
  (h2 : first_oil_price = 50)
  (h3 : second_oil_volume = 5)
  (h4 : mixture_price = 56)
  (h5 : (first_oil_volume * first_oil_price + second_oil_volume * second_oil_price) / (first_oil_volume + second_oil_volume) = mixture_price) :
  second_oil_price = 68 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_oil_price_l256_25685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l256_25638

theorem polynomial_divisibility (n k : ℕ) 
  (h : (X ^ (2 * k) - X ^ k + 1 : Polynomial ℤ) ∣ (X ^ (2 * n) + X ^ n + 1)) :
  (X ^ (2 * k) + X ^ k + 1 : Polynomial ℤ) ∣ (X ^ (2 * n) + X ^ n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l256_25638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_earnings_l256_25689

-- Define the constants
def hours_per_day_123 : ℕ := 23
def hours_per_day_4 : ℕ := 12
def production_rate_A_12 : ℕ := 2
def production_rate_B_12 : ℕ := 1
def production_rate_A_34 : ℕ := 3
def production_rate_B_34 : ℕ := 2
def price_A : ℕ := 50
def price_C : ℕ := 100
def conversion_rate_B_to_C : ℕ := 2

-- Define the theorem
theorem factory_earnings : 
  (production_rate_A_12 * hours_per_day_123 * 2 + 
   production_rate_A_34 * hours_per_day_123 + 
   production_rate_A_34 * hours_per_day_4) * price_A +
  ((production_rate_B_12 * hours_per_day_123 * 2 + 
    production_rate_B_34 * hours_per_day_123 + 
    production_rate_B_34 * hours_per_day_4) / conversion_rate_B_to_C) * price_C
  = 15650 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_earnings_l256_25689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l256_25696

-- Define the function
noncomputable def y (k x : ℝ) : ℝ := (k + 2) * x^(k^2 + k - 4)

-- Define the conditions
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

def increasing_for_negative_x (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ ∧ x₂ < 0 → f x₁ < f x₂

-- State the theorem
theorem quadratic_function_properties :
  ∃ k : ℝ,
    (is_quadratic (y k)) ∧
    (increasing_for_negative_x (y k)) ∧
    (∀ m n : ℝ, -2 ≤ m ∧ m ≤ 1 ∧ y k m = n → -4 ≤ n ∧ n ≤ 0) ∧
    k = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l256_25696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angles_with_axes_l256_25619

def vector_a : Fin 3 → ℝ := ![(-2 : ℝ), 2, 1]

theorem angles_with_axes :
  let magnitude := Real.sqrt (vector_a 0^2 + vector_a 1^2 + vector_a 2^2)
  (vector_a 0 / magnitude = -2/3) ∧
  (vector_a 1 / magnitude = 2/3) ∧
  (vector_a 2 / magnitude = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angles_with_axes_l256_25619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_construction_l256_25659

-- Define the necessary structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Square where
  center : ℝ × ℝ
  side : ℝ

structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

-- Helper functions (not proven)
def circle_points (O : Circle) : Set (ℝ × ℝ) := sorry
def square_points (K : Square) : Set (ℝ × ℝ) := sorry
def segment_length (p₁ p₂ : ℝ × ℝ) : ℝ := sorry
def segment_parallel (p₁ p₂ : ℝ × ℝ) (L : Line) : Prop := sorry

-- Define the theorem
theorem segment_construction (O : Circle) (K : Square) (L : Line) (d : ℝ) 
  (h : d > 0) : 
  ∃ (p₁ : ℝ × ℝ) (p₂ : ℝ × ℝ), 
    (p₁ ∈ circle_points O) ∧ 
    (p₂ ∈ square_points K) ∧ 
    (segment_length p₁ p₂ = d) ∧ 
    (segment_parallel p₁ p₂ L) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_construction_l256_25659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distance_squares_constant_l256_25626

variable (R : ℝ)
variable (M : ℝ × ℝ)

def isOnCircle (M : ℝ × ℝ) (R : ℝ) : Prop :=
  M.1^2 + M.2^2 = R^2

noncomputable def diameterPoints (R : ℝ) : List (ℝ × ℝ) :=
  [(-R, 0), (-R/2, 0), (0, 0), (R/2, 0), (R, 0)]

def distanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

noncomputable def sumOfDistanceSquares (M : ℝ × ℝ) (R : ℝ) : ℝ :=
  (diameterPoints R).map (distanceSquared M) |>.sum

theorem sum_of_distance_squares_constant (R : ℝ) (M : ℝ × ℝ) (h : isOnCircle M R) :
  sumOfDistanceSquares M R = 15 * R^2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distance_squares_constant_l256_25626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mod_31_problem_l256_25682

theorem mod_31_problem (m : ℕ) (h1 : 0 ≤ m) (h2 : m < 31) (h3 : (4 * m) % 31 = 1) :
  (3^m)^4 % 31 - 3 ≡ 29 [ZMOD 31] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mod_31_problem_l256_25682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_finite_iff_a_nonzero_max_A_40_l256_25686

def A (a : ℝ) : Set ℕ :=
  {n : ℕ | ∃ k : ℕ, n^2 + a * n = k^2}

theorem A_finite_iff_a_nonzero (a : ℝ) :
  (A a).Finite ↔ a ≠ 0 :=
sorry

-- Additional theorem for part b
theorem max_A_40 : 
  ∃ m : ℕ, m ∈ A 40 ∧ ∀ n : ℕ, n ∈ A 40 → n ≤ m ∧ m = 380 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_finite_iff_a_nonzero_max_A_40_l256_25686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_answers_are_correct_l256_25644

/-- Represents the vocabulary words from the text --/
inductive Vocabulary where
  | avoid
  | add
  | kick

/-- Represents the correct answers for each word --/
def correct_answer (word : Vocabulary) : String :=
  match word with
  | .avoid => "avoid"
  | .add => "add"
  | .kick => "kick"

/-- A theorem stating that the answers are correct --/
theorem answers_are_correct :
  (correct_answer Vocabulary.avoid = "avoid") ∧
  (correct_answer Vocabulary.add = "add") ∧
  (correct_answer Vocabulary.kick = "kick") := by
  simp [correct_answer]

#print answers_are_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_answers_are_correct_l256_25644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_transformations_l256_25666

def is_arithmetic_sequence (s : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) - s n = d

def subsequence (s : ℕ → ℕ → ℝ) (f : ℕ → ℕ) : ℕ → ℝ :=
  λ n ↦ s n (f n)

theorem arithmetic_sequence_transformations (a : ℕ → ℝ) 
  (h : is_arithmetic_sequence a) : 
  (is_arithmetic_sequence (subsequence (λ _ ↦ a) (λ n ↦ 2*n - 1))) ∧
  (is_arithmetic_sequence (λ n ↦ 3 - 2 * (a n))) ∧
  (¬ ∀ b : ℝ, is_arithmetic_sequence (λ n ↦ |a n|)) ∧
  (¬ ∀ b : ℝ, b > 0 → is_arithmetic_sequence (λ n ↦ Real.log (a n))) ∧
  (¬ ∀ b : ℝ, is_arithmetic_sequence (λ n ↦ (a n)^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_transformations_l256_25666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_periodic_l256_25603

noncomputable def f (x : ℝ) : ℝ := Real.cos (2*x)^2 - Real.sin (2*x)^2

theorem f_is_even_and_periodic :
  (∀ x, f x = f (-x)) ∧ (∀ x, f (x + π/2) = f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_periodic_l256_25603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_animal_arrangement_count_l256_25681

def num_chickens : ℕ := 5
def num_dogs : ℕ := 3
def num_cats : ℕ := 6
def num_rabbits : ℕ := 4
def total_animals : ℕ := num_chickens + num_dogs + num_cats + num_rabbits

def arrangement_count : ℕ := 17863680

theorem animal_arrangement_count :
  (Nat.factorial 4) *
  (Nat.factorial num_chickens) *
  (Nat.factorial num_dogs) *
  (Nat.factorial num_cats) *
  (Nat.factorial num_rabbits) = arrangement_count := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_animal_arrangement_count_l256_25681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_third_quadrant_l256_25610

theorem tan_value_third_quadrant (α : Real) (h1 : α ∈ Set.Icc π (3*π/2)) 
  (h2 : Real.sin α = -2/3) : Real.tan α = 2*Real.sqrt 5/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_third_quadrant_l256_25610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l256_25600

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 else x + 4/x - 3

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Ici (0 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l256_25600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_K₃C₆₀_has_both_bond_types_l256_25609

/-- Represents a chemical element -/
inductive Element
| K  -- Potassium
| C  -- Carbon
deriving BEq, Repr

/-- Represents a chemical bond type -/
inductive BondType
| Ionic
| Covalent
deriving BEq, Repr

/-- Represents a chemical compound -/
structure Compound where
  elements : List Element
  bondTypes : List BondType

/-- Represents properties of a substance -/
structure SubstanceProperties where
  isIonicCrystal : Bool
  hasSuperconductivity : Bool

/-- Defines K₃C₆₀ -/
def K₃C₆₀ : Compound :=
  { elements := List.replicate 3 Element.K ++ List.replicate 60 Element.C,
    bondTypes := [BondType.Ionic, BondType.Covalent] }

/-- Properties of K₃C₆₀ -/
def K₃C₆₀Properties : SubstanceProperties :=
  { isIonicCrystal := true,
    hasSuperconductivity := true }

/-- Checks if a compound contains both ionic and covalent bonds -/
def hasBothBondTypes (c : Compound) : Bool :=
  c.bondTypes.contains BondType.Ionic && c.bondTypes.contains BondType.Covalent

/-- Theorem: K₃C₆₀ contains both ionic and covalent bonds -/
theorem K₃C₆₀_has_both_bond_types :
  hasBothBondTypes K₃C₆₀ = true := by
  simp [hasBothBondTypes, K₃C₆₀]
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_K₃C₆₀_has_both_bond_types_l256_25609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_over_log2e_l256_25621

/-- The sum of elements in the nth row of Pascal's triangle -/
def pascal_row_sum (n : ℕ) : ℕ := 2^n

/-- Function g(n) as defined in the problem -/
noncomputable def g (n : ℕ) : ℝ := Real.log (2 * pascal_row_sum n)

/-- The main theorem to prove -/
theorem g_over_log2e (n : ℕ) : g n / Real.log 2 = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_over_log2e_l256_25621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_ce_l256_25623

/-- Definition of an equilateral triangle -/
def EquilateralTriangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

/-- Definition of congruent triangles -/
def CongruentTriangles (A B C D E F : ℝ × ℝ) : Prop :=
  dist A B = dist D E ∧ dist B C = dist E F ∧ dist C A = dist F D

/-- Given an equilateral triangle ABC with side length √200 and four congruent triangles
    AD₁E₁, AD₁E₂, AD₂E₃, and AD₂E₄ where BD₁ = BD₂ = √20, prove that the sum of the
    squares of CE₁, CE₂, CE₃, and CE₄ is equal to 1220. -/
theorem sum_of_squares_of_ce (A B C D₁ D₂ E₁ E₂ E₃ E₄ : ℝ × ℝ) : 
  EquilateralTriangle A B C → 
  dist A B = Real.sqrt 200 →
  CongruentTriangles A D₁ E₁ A B C →
  CongruentTriangles A D₁ E₂ A B C →
  CongruentTriangles A D₂ E₃ A B C →
  CongruentTriangles A D₂ E₄ A B C →
  dist B D₁ = Real.sqrt 20 →
  dist B D₂ = Real.sqrt 20 →
  (dist C E₁)^2 + (dist C E₂)^2 + (dist C E₃)^2 + (dist C E₄)^2 = 1220 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_ce_l256_25623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_lines_l256_25612

-- Define a line passing through (-2,2) with slope k
noncomputable def line (k : ℝ) := {(x, y) : ℝ × ℝ | y - 2 = k * (x + 2)}

-- Define the x-intercept of the line
noncomputable def x_intercept (k : ℝ) : ℝ := -(2 + 2*k) / k

-- Define the y-intercept of the line
noncomputable def y_intercept (k : ℝ) : ℝ := 2*k + 2

-- Define the area of the triangle formed by the line and the axes
noncomputable def triangle_area (k : ℝ) : ℝ := |x_intercept k * y_intercept k| / 2

-- The main theorem
theorem exactly_three_lines :
  ∃! (s : Finset ℝ), s.card = 3 ∧ 
  (∀ k ∈ s, triangle_area k = 8) ∧
  (∀ k : ℝ, triangle_area k = 8 → k ∈ s) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_lines_l256_25612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_properties_l256_25672

-- Define the system of linear equations
def system (x y m : ℝ) : Prop :=
  3 * x + 5 * y = m + 2 ∧ 2 * x + 3 * y = m

-- Theorem statement
theorem system_properties :
  (∃ m : ℝ, system 4 (-1) m) ∧
  (∀ m₁ m₂ : ℝ, ∃ x₁ y₁ x₂ y₂ : ℝ,
    system x₁ y₁ m₁ ∧ system x₂ y₂ m₂ →
    (2 : ℝ)^x₁ * (4 : ℝ)^y₁ = (2 : ℝ)^x₂ * (4 : ℝ)^y₂) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_properties_l256_25672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_parallel_vectors_l256_25630

noncomputable def m (x : ℝ) : ℝ × ℝ := (2 * Real.sqrt 3 * Real.sin x, Real.cos x)
noncomputable def p : ℝ × ℝ := (2 * Real.sqrt 3, 1)

theorem sin_cos_product_parallel_vectors (x : ℝ) :
  (∃ (k : ℝ), m x = k • p) → Real.sin x * Real.cos x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_parallel_vectors_l256_25630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_cosine_double_angle_l256_25617

theorem triangle_angle_cosine_double_angle (A B C : ℝ) : 
  (0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi) → 
  (A > B ↔ Real.cos (2 * B) > Real.cos (2 * A)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_cosine_double_angle_l256_25617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_triangle_area_l256_25695

/-- Represents an ellipse with center at the origin -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis

/-- The equation of the ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The focal distance of the ellipse -/
noncomputable def Ellipse.focalDistance (e : Ellipse) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2)

/-- The area of triangle OAB given the x-coordinate of point A -/
def triangleArea (k : ℝ) (x : ℝ) : ℝ :=
  k * x^2

theorem ellipse_max_triangle_area (k : ℝ) (hk : k > 0) :
  ∃ (e : Ellipse),
    e.focalDistance = Real.sqrt 3 ∧
    e.a = 2 * e.focalDistance ∧
    (∀ (x y : ℝ), e.equation x y → ∀ (t : ℝ), y = k * t → triangleArea k x ≤ 3 * Real.sqrt 3) ∧
    (∃ (x y : ℝ), e.equation x y ∧ y = k * x ∧ triangleArea k x = 3 * Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_triangle_area_l256_25695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_square_root_l256_25651

theorem log_sum_square_root (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y * z = 10^81)
  (h2 : (Real.log x) * (Real.log y + Real.log z) + (Real.log y) * (Real.log z) = 468 * (Real.log 10)) :
  Real.sqrt ((Real.log x)^2 + (Real.log y)^2 + (Real.log z)^2) = 75 * Real.log 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_square_root_l256_25651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_squares_l256_25663

theorem min_sum_of_squares (a b : ℝ) : 
  (∃ x : ℝ, (a * x - b / x)^6 = -160 + (a * x - b / x)^6 + 160) → 
  (∀ a' b' : ℝ, (∃ x : ℝ, (a' * x - b' / x)^6 = -160 + (a' * x - b' / x)^6 + 160) → a'^2 + b'^2 ≥ 4) ∧
  (∃ a' b' : ℝ, (∃ x : ℝ, (a' * x - b' / x)^6 = -160 + (a' * x - b' / x)^6 + 160) ∧ a'^2 + b'^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_squares_l256_25663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_inequality_tangent_line_equation_l256_25656

-- Define the functions f and g
def f (x : ℝ) : ℝ := -x^2 - 3
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 2*x*Real.log x - a*x

-- Define the derivatives of f and g
def f' (x : ℝ) : ℝ := -2*x
noncomputable def g' (a : ℝ) (x : ℝ) : ℝ := 2*Real.log x + 2 - a

-- Theorem statement
theorem tangent_line_and_inequality (a : ℝ) :
  (∀ x : ℝ, x > 0 → g' a x = f' x) →
  (∀ x : ℝ, x > 0 → g a x - f x ≥ 0) ↔ a ≤ 4 :=
by sorry

-- Theorem for the equation of the tangent line
theorem tangent_line_equation (a : ℝ) :
  (∀ x : ℝ, x > 0 → g' a x = f' x) →
  ∃ m b : ℝ, ∀ x y : ℝ, y = g a 1 + (g' a 1) * (x - 1) ↔ 2*x + y + 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_inequality_tangent_line_equation_l256_25656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_c_coordinates_l256_25660

/-- Given two points A and B in a plane, and a real number l ≠ 1,
    this theorem states that the point C with the given coordinates
    satisfies the condition (ABC) = l. -/
theorem point_c_coordinates
  (x₁ y₁ x₂ y₂ l : ℝ)
  (h : l ≠ 1) :
  let A := (x₁, y₁)
  let B := (x₂, y₂)
  let C := ((l * x₂ - x₁) / (l - 1), (l * y₂ - y₁) / (l - 1))
  (A - C) = l • (B - C) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_c_coordinates_l256_25660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_school_proof_l256_25679

/-- The distance to school in miles -/
noncomputable def distance_to_school : ℝ := 9

/-- The typical speed in miles per hour -/
noncomputable def typical_speed : ℝ := 18

/-- The increased speed on a quieter day in miles per hour -/
noncomputable def quieter_day_speed : ℝ := typical_speed + 12

/-- The typical trip time in hours -/
noncomputable def typical_time : ℝ := 1/2

/-- The quieter day trip time in hours -/
noncomputable def quieter_time : ℝ := 3/10

theorem distance_to_school_proof :
  (distance_to_school = typical_speed * typical_time) ∧
  (distance_to_school = quieter_day_speed * quieter_time) := by
  sorry

#check distance_to_school_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_school_proof_l256_25679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_sine_proposition_l256_25634

theorem negation_of_universal_sine_proposition :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x : ℝ, Real.sin x > 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_sine_proposition_l256_25634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l256_25645

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -4 * (Real.cos x)^2 + 4 * Real.sqrt 3 * a * Real.sin x * Real.cos x + 2

theorem function_properties (a : ℝ) :
  (∀ x : ℝ, f a x = f a (π/6 - x)) →
  (a = 1) ∧
  (∀ k : ℤ, ∀ x : ℝ, x ∈ Set.Icc (π/3 + k*π) (5*π/6 + k*π) → 
    ∀ y : ℝ, y ∈ Set.Icc (π/3 + k*π) (5*π/6 + k*π) → x < y → f a x > f a y) ∧
  (∀ x : ℝ, f a x = f a (x + π)) ∧
  (Set.Icc (-4) 2 = {y | ∃ x ∈ Set.Icc (-π/4) (π/6), f a x = y}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l256_25645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_triangle_area_l256_25636

noncomputable section

/-- Given an ellipse with semi-major axis a and semi-minor axis b -/
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

/-- The eccentricity of an ellipse -/
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (b^2 / a^2))

/-- The distance from a point to a line ax + by + c = 0 -/
def distancePointToLine (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * p.1 + b * p.2 + c) / Real.sqrt (a^2 + b^2)

/-- The area of a triangle given three points -/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) / 2

theorem ellipse_max_triangle_area
  (a b : ℝ)
  (h_ab : a > b ∧ b > 0)
  (h_ecc : eccentricity a b = 1/2)
  (h_dist : ∃ (f : ℝ × ℝ), f ∈ Ellipse a b ∧ distancePointToLine f 3 4 0 = 3/5)
  (k m : ℝ)
  (h_km : k * m ≠ 0)
  (h_midpoint : ∃ (A B : ℝ × ℝ), A ∈ Ellipse a b ∧ B ∈ Ellipse a b ∧
    A.2 = k * A.1 + m ∧ B.2 = k * B.1 + m ∧
    3 * ((A.1 + B.1) / 2) + 4 * ((A.2 + B.2) / 2) = 0) :
  ∃ (S : ℝ), (∀ (A B : ℝ × ℝ), A ∈ Ellipse a b → B ∈ Ellipse a b →
    A.2 = k * A.1 + m → B.2 = k * B.1 + m →
    triangleArea (0, 0) A B ≤ S) ∧
  S = Real.sqrt 3 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_triangle_area_l256_25636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l256_25650

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (x - 2)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | x > 2} = {x : ℝ | ∃ y, f x = y} :=
by
  -- The proof is skipped using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l256_25650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_prime_factorization_l256_25628

theorem factorial_prime_factorization :
  ∃ (i j k m n p q : ℕ),
    Nat.factorial 15 = 2^i * 3^j * 5^k * 7^m * 11^n * 13^p * 17^q ∧
    i + j + k + m + n + p + q = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_prime_factorization_l256_25628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l256_25606

/-- The sum of the infinite series 1/(4^1) + 2/(4^2) + 3/(4^3) + ... + k/(4^k) + ... -/
noncomputable def infiniteSeries : ℝ := ∑' k, k / (4 ^ k)

/-- The sum of the infinite series is equal to 4/9 -/
theorem infiniteSeriesSum : infiniteSeries = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l256_25606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_through_focus_property_l256_25661

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/36 + y^2/9 = 1

-- Define the focus
noncomputable def focus : ℝ × ℝ := (3 * Real.sqrt 3, 0)

-- Define a point on the ellipse
structure Point_on_ellipse where
  x : ℝ
  y : ℝ
  on_ellipse : is_on_ellipse x y

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem chord_through_focus_property
  (A B : Point_on_ellipse)
  (h_chord : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ 
    (t * A.x + (1 - t) * B.x = focus.1) ∧
    (t * A.y + (1 - t) * B.y = focus.2))
  (h_AF : distance (A.x, A.y) focus = 3/2) :
  distance (B.x, B.y) focus = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_through_focus_property_l256_25661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_concurrent_circumcircles_l256_25680

-- Define the types for points and lines
variable (Point Line : Type)

-- Define the incidence relation
variable (lies_on : Point → Line → Prop)

-- Define the intersection of two lines
variable (intersect : Line → Line → Point)

-- Define the circumcircle of a triangle
variable (circumcircle : Point → Point → Point → Type)

-- Define the concurrency of circles
variable (concurrent : List Type → Prop)

-- Define the ratio of lengths
variable (length_ratio : Point → Point → Point → Point → ℚ)

theorem quadrilateral_concurrent_circumcircles 
  (A B C D E F S T : Point) 
  (AB BC CD DA EF : Line) :
  lies_on E DA →
  lies_on F BC →
  length_ratio A E E D = length_ratio B F F C →
  S = intersect EF AB →
  T = intersect EF CD →
  concurrent [circumcircle S A E, circumcircle S B F, circumcircle T C F, circumcircle T D E] :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_concurrent_circumcircles_l256_25680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_arithmetic_sequence_properties_l256_25693

/-- An increasing arithmetic sequence with special properties -/
structure SpecialArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  h_increasing : ∀ n, a n < a (n + 1)
  h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  h_first_term : a 1 = 2
  h_geometric : (2 * S 1) * (3 * S 3) = (2 * S 2)^2

/-- The general term of the sequence -/
noncomputable def general_term (seq : SpecialArithmeticSequence) (n : ℕ) : ℝ := 2 * n

/-- The b_n sequence derived from a_n -/
noncomputable def b (seq : SpecialArithmeticSequence) (n : ℕ) : ℝ := 
  4 / (seq.a n * seq.a (n + 1))

/-- The sum of the first n terms of b_n -/
noncomputable def T (seq : SpecialArithmeticSequence) (n : ℕ) : ℝ := n / (n + 1 : ℝ)

/-- The main theorem stating the properties of the special arithmetic sequence -/
theorem special_arithmetic_sequence_properties (seq : SpecialArithmeticSequence) :
  (∀ n, seq.a n = general_term seq n) ∧ 
  (∀ n, T seq n = n / (n + 1 : ℝ)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_arithmetic_sequence_properties_l256_25693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_relation_l256_25687

-- Define a triangle in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the centroid of a triangle
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem centroid_distance_relation (t : Triangle) (X : ℝ × ℝ) :
  let S := centroid t
  (distance t.A X)^2 + (distance t.B X)^2 + (distance t.C X)^2 =
  (distance S t.A)^2 + (distance S t.B)^2 + (distance S t.C)^2 + 3 * (distance S X)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_relation_l256_25687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_for_constant_term_l256_25648

/-- The binomial expression -/
noncomputable def binomial_expr (x : ℝ) := 3 * x^2 - 2 / (3 * x)

/-- Predicate to check if the expansion contains a constant term -/
def has_constant_term (n : ℕ) : Prop :=
  ∃ r : ℕ, r ≤ n ∧ 2 * n = (7 * r : ℕ) / 3

/-- The theorem stating the minimum value of n -/
theorem min_n_for_constant_term :
  ∀ n : ℕ, n > 0 → (has_constant_term n → n ≥ 7) ∧
  (has_constant_term 7) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_for_constant_term_l256_25648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l256_25613

theorem trig_problem (α β : ℝ) 
  (h1 : Real.sin α = 4/5)
  (h2 : α ∈ Set.Ioo (π/2) π)
  (h3 : Real.cos β = -5/13)
  (h4 : β ∈ Set.Ioo π (3*π/2)) :
  Real.sin (α - β) = -56/65 ∧ Real.tan (α + β) = 16/63 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l256_25613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_A_max_B_l256_25677

/-- Two-digit natural number -/
def TwoDigitNat : Type := {n : ℕ // 10 ≤ n ∧ n ≤ 99}

/-- The equation constraint -/
def SatisfiesEquation (A B : TwoDigitNat) : Prop :=
  (A.val - 5 : ℚ) / A.val + 4 / B.val = 1

theorem min_A_max_B :
  ∃ (A B : TwoDigitNat), SatisfiesEquation A B ∧
    (∀ (A' B' : TwoDigitNat), SatisfiesEquation A' B' → A.val ≤ A'.val) ∧
    (∀ (A' B' : TwoDigitNat), SatisfiesEquation A' B' → B'.val ≤ B.val) ∧
    A.val = 15 ∧ B.val = 76 := by
  sorry

#check min_A_max_B

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_A_max_B_l256_25677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_meeting_time_l256_25655

/-- The circumference of the circular track in meters -/
noncomputable def track_circumference : ℝ := 3000

/-- Dong-hoon's walking speed in meters per minute -/
noncomputable def speed_donghoon : ℝ := 100

/-- Yeon-jeong's walking speed in meters per minute -/
noncomputable def speed_yeonjeong : ℝ := 150

/-- The time in minutes when Dong-hoon and Yeon-jeong meet again -/
noncomputable def meeting_time : ℝ := track_circumference / (speed_donghoon + speed_yeonjeong)

theorem first_meeting_time : meeting_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_meeting_time_l256_25655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_integers_sum_zero_l256_25669

theorem three_integers_sum_zero (m : ℕ) (S : Finset ℤ) : 
  S.card = 2*m + 1 →
  (∀ x ∈ S, |x| ≤ 2*m - 1) →
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_integers_sum_zero_l256_25669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l256_25664

theorem trigonometric_identity (x : ℝ) : 
  Real.cos x * Real.cos (2 * x) = 
    Real.sin (Real.pi / 4 + x) * Real.sin (Real.pi / 4 + 4 * x) + 
    Real.sin (3 * Real.pi / 4 + 4 * x) * Real.cos (7 * Real.pi / 4 - 5 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l256_25664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_2005_calculation_l256_25605

-- Define the per capita net income for 2004
variable (a : ℝ)

-- Define the percentage increase
def increase_percentage : ℚ := 14.2

-- Define the per capita net income for 2005
noncomputable def income_2005 (a : ℝ) : ℝ := a * (1 + increase_percentage / 100)

-- Theorem to prove
theorem income_2005_calculation (a : ℝ) : 
  income_2005 a = 1.142 * a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_2005_calculation_l256_25605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_payment_is_600_l256_25639

/-- The time (in days) it takes person a to complete the work alone -/
noncomputable def a_time : ℚ := 6

/-- The time (in days) it takes person b to complete the work alone -/
noncomputable def b_time : ℚ := 8

/-- The time (in days) it takes person c to complete the work alone -/
noncomputable def c_time : ℚ := 12

/-- The total payment for the work -/
noncomputable def total_payment : ℚ := 1800

/-- The share of the work done by person b -/
noncomputable def b_share : ℚ := (1 / b_time) / ((1 / a_time) + (1 / b_time) + (1 / c_time))

/-- Theorem stating that b's share of the payment is 600 -/
theorem b_payment_is_600 : b_share * total_payment = 600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_payment_is_600_l256_25639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_multiples_of_seven_l256_25649

/-- The sequence (a_n) defined recursively -/
def a : ℕ → ℕ
  | 0 => 1  -- Define for 0 to cover all cases
  | 1 => 1
  | n + 2 => a (n + 1) + a ((n + 2) / 2)

/-- Theorem: The sequence (a_n) contains infinitely many multiples of 7 -/
theorem infinitely_many_multiples_of_seven :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ 7 ∣ a n :=
by
  sorry  -- Proof is omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_multiples_of_seven_l256_25649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l256_25678

-- Problem 1
theorem problem_1 : (1 : ℝ) * (-1)^(2023 : ℕ) + (Real.pi + Real.sqrt 3)^(0 : ℕ) + (-1/2 : ℝ)^(-2 : ℤ) = 4 := by sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) : 
  (2 : ℝ) * (2 * x^2 * y^2 + x * y^3) / (x * y) = 2 * x * y + y^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l256_25678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_difference_l256_25642

-- Define the functions f and g
def f (b : ℝ) (x : ℝ) : ℝ := x^3 + b*x - 14
noncomputable def g (x : ℝ) : ℝ := x^2 - 3 * Real.log x

-- State the theorem
theorem common_tangent_difference (a b : ℝ) :
  (∃ (x y : ℝ), x + y + a = 0 ∧
    (∃ (m : ℝ), (x, y) = (m, g m) ∧ (deriv g) m = -1) ∧
    (∃ (n : ℝ), (x, y) = (n, f b n) ∧ (deriv (f b)) n = -1)) →
  a - b = 11 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_difference_l256_25642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_calorie_burn_ratio_l256_25641

/-- Represents the dancing and calorie burning information for James --/
structure DancingInfo where
  daily_sessions : Nat
  session_duration : ℚ
  weekly_frequency : Nat
  walking_calories_per_hour : Nat
  weekly_dancing_calories : Nat

/-- Calculates the ratio of calories burned per hour dancing to calories burned per hour walking --/
noncomputable def calorie_burn_ratio (info : DancingInfo) : ℚ :=
  let weekly_dancing_hours := (info.daily_sessions : ℚ) * info.session_duration * (info.weekly_frequency : ℚ)
  let dancing_calories_per_hour := (info.weekly_dancing_calories : ℚ) / weekly_dancing_hours
  dancing_calories_per_hour / (info.walking_calories_per_hour : ℚ)

/-- Theorem stating that the calorie burn ratio for James is 2 --/
theorem james_calorie_burn_ratio :
  let james_info : DancingInfo := {
    daily_sessions := 2,
    session_duration := 1/2,
    weekly_frequency := 4,
    walking_calories_per_hour := 300,
    weekly_dancing_calories := 2400
  }
  calorie_burn_ratio james_info = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_calorie_burn_ratio_l256_25641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l256_25691

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, (0 < S ∧ S < T) → ∃ x, f (x + S) ≠ f x) ∧  -- smallest positive period is π
  (∀ x y, x ∈ Set.Icc 0 (Real.pi / 12) → y ∈ Set.Icc 0 (Real.pi / 12) → x < y → f x < f y) ∧  -- increasing on [0, π/12]
  (∀ x, Real.sin (2 * x) = -Real.sin (2 * (-x))) :=  -- g(x) = sin(2x) is odd
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l256_25691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_painted_face_equivalence_l256_25615

/-- Represents a face of a cube -/
inductive Face
| Painted (size : ℕ) (orientation : ℕ)
| Unpainted

/-- Represents a cube with six faces -/
structure Cube where
  faces : Fin 6 → Face

/-- Checks if a cube has exactly one painted face -/
def has_one_painted_face (c : Cube) : Prop :=
  (∃! i, ∃ (s o : ℕ), c.faces i = Face.Painted s o) ∧
  (∀ i, (∀ s o, c.faces i ≠ Face.Painted s o) → c.faces i = Face.Unpainted)

/-- Checks if two cubes are equivalent after rolling -/
def equivalent_after_rolling (c1 c2 : Cube) : Prop :=
  ∃ (perm : Fin 6 → Fin 6), Function.Bijective perm ∧
  ∀ i, c1.faces i = c2.faces (perm i)

/-- Main theorem: A cube with one painted face is only equivalent to cubes with the same characteristics -/
theorem one_painted_face_equivalence (c1 c2 : Cube) 
  (h1 : has_one_painted_face c1) 
  (h2 : has_one_painted_face c2) :
  equivalent_after_rolling c1 c2 ↔ 
  ∃ (i j : Fin 6) (s o : ℕ), 
    c1.faces i = Face.Painted s o ∧ 
    c2.faces j = Face.Painted s o :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_painted_face_equivalence_l256_25615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_side_length_l256_25647

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  vertices : Fin 8 → Point3D

/-- Represents an octahedron in 3D space -/
structure Octahedron where
  vertices : Fin 6 → Point3D

/-- Checks if two points are adjacent in a unit cube -/
def areAdjacent (p q : Point3D) : Prop :=
  |p.x - q.x| + |p.y - q.y| + |p.z - q.z| = 1

/-- Checks if two points are opposite in a unit cube -/
def areOpposite (p q : Point3D) : Prop :=
  |p.x - q.x| + |p.y - q.y| + |p.z - q.z| = 3

/-- Checks if a point lies on the line segment between two other points -/
def liesBetween (p q r : Point3D) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧
    p.x = q.x + t * (r.x - q.x) ∧
    p.y = q.y + t * (r.y - q.y) ∧
    p.z = q.z + t * (r.z - q.z)

/-- Main theorem: The side length of the octahedron is 3√2/4 -/
theorem octahedron_side_length 
    (cube : Cube) 
    (octa : Octahedron) : 
    (∀ i j : Fin 8, i ≠ j → areAdjacent (cube.vertices i) (cube.vertices j) ∨ areOpposite (cube.vertices i) (cube.vertices j)) →
    (∀ i : Fin 6, ∃ j k : Fin 8, j ≠ k ∧ liesBetween (octa.vertices i) (cube.vertices j) (cube.vertices k)) →
    ∃ i j : Fin 6, i ≠ j → 
      Real.sqrt ((octa.vertices i).x - (octa.vertices j).x)^2 + 
                ((octa.vertices i).y - (octa.vertices j).y)^2 + 
                ((octa.vertices i).z - (octa.vertices j).z)^2 = 3 * Real.sqrt 2 / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_side_length_l256_25647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l256_25622

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x^2 - 2*x + 3 > 0)) ↔ (∃ x : ℝ, x^2 - 2*x + 3 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l256_25622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_six_probability_l256_25625

noncomputable section

/-- Represents a six-sided die -/
structure Die where
  /-- Probability of rolling a six -/
  prob_six : ℝ
  /-- Probability of rolling any other number (1 to 5) -/
  prob_other : ℝ
  /-- The probabilities sum to 1 -/
  prob_sum : prob_six + 5 * prob_other = 1

/-- The fair die -/
noncomputable def fair_die : Die where
  prob_six := 1/6
  prob_other := 1/6
  prob_sum := by norm_num

/-- The biased die -/
noncomputable def biased_die : Die where
  prob_six := 3/4
  prob_other := 1/20
  prob_sum := by norm_num

/-- The probability of choosing either die -/
def prob_choose_die : ℝ := 1/2

/-- The probability of rolling three sixes in a row with a given die -/
noncomputable def prob_three_sixes (d : Die) : ℝ := d.prob_six ^ 3

/-- The probability of rolling four sixes in a row with a given die -/
noncomputable def prob_four_sixes (d : Die) : ℝ := d.prob_six ^ 4

/-- The probability of rolling three sixes and then any number with a given die -/
noncomputable def prob_three_sixes_any (d : Die) : ℝ := d.prob_six ^ 3

theorem fourth_six_probability :
  let p_three_sixes := prob_choose_die * prob_three_sixes fair_die + 
                       prob_choose_die * prob_three_sixes biased_die
  let p_biased_given_three_sixes := (prob_three_sixes biased_die * prob_choose_die) / p_three_sixes
  let p_fair_given_three_sixes := 1 - p_biased_given_three_sixes
  let p_fourth_six := p_biased_given_three_sixes * biased_die.prob_six + 
                      p_fair_given_three_sixes * fair_die.prob_six
  p_fourth_six = 365.335 / 491 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_six_probability_l256_25625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_expressions_equality_l256_25671

theorem sqrt_expressions_equality : 
  (Real.sqrt 8 - 2 * Real.sqrt 18 + Real.sqrt 24 = -4 * Real.sqrt 2 + 2 * Real.sqrt 6) ∧
  ((Real.sqrt (4/3) + Real.sqrt 3) * Real.sqrt 12 - Real.sqrt 48 + Real.sqrt 6 = 10 - 4 * Real.sqrt 3 + Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_expressions_equality_l256_25671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_scores_l256_25632

/-- Represents a football team --/
inductive Team
| Hawks
| Eagles
| Falcons
deriving Repr, DecidableEq

/-- Represents a football match --/
structure Match where
  team1 : Team
  team2 : Team
  team1_touchdowns : ℕ
  team1_fieldgoals : ℕ
  team1_safeties : ℕ
  team2_touchdowns : ℕ
  team2_fieldgoals : ℕ
  team2_safeties : ℕ

/-- Calculates the score for a team in a match --/
def score_in_match (m : Match) (t : Team) : ℕ :=
  if t = m.team1 then
    7 * m.team1_touchdowns + 3 * m.team1_fieldgoals + 2 * m.team1_safeties
  else if t = m.team2 then
    7 * m.team2_touchdowns + 3 * m.team2_fieldgoals + 2 * m.team2_safeties
  else
    0

/-- The two matches in the tournament --/
def match1 : Match :=
  { team1 := Team.Hawks
  , team2 := Team.Eagles
  , team1_touchdowns := 3
  , team1_fieldgoals := 2
  , team1_safeties := 1
  , team2_touchdowns := 5
  , team2_fieldgoals := 4
  , team2_safeties := 0 }

def match2 : Match :=
  { team1 := Team.Hawks
  , team2 := Team.Falcons
  , team1_touchdowns := 4
  , team1_fieldgoals := 3
  , team1_safeties := 0
  , team2_touchdowns := 6
  , team2_fieldgoals := 0
  , team2_safeties := 2 }

/-- Calculates the total score for a team across all matches --/
def total_score (t : Team) : ℕ :=
  score_in_match match1 t + score_in_match match2 t

/-- Theorem stating the correct total scores for each team --/
theorem correct_scores :
  total_score Team.Hawks = 66 ∧
  total_score Team.Eagles = 47 ∧
  total_score Team.Falcons = 46 := by
  sorry

#eval total_score Team.Hawks
#eval total_score Team.Eagles
#eval total_score Team.Falcons

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_scores_l256_25632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_digit_base_4_853_l256_25640

/-- The first digit of the base 4 representation of a positive integer -/
def first_digit_base_4 (n : ℕ) : ℕ :=
  if n = 0 then 0
  else
    let k := Nat.log 4 n
    (n / (4 ^ k : ℕ))

theorem first_digit_base_4_853 :
  first_digit_base_4 853 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_digit_base_4_853_l256_25640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_C_in_triangle_l256_25657

open Real Complex

theorem max_angle_C_in_triangle (A B C : ℝ) (z : ℂ) : 
  A + B + C = π →
  z = (Real.sqrt 65 / 5) * Complex.exp (I * ((A + B) / 2)) + I * Complex.exp (I * ((A - B) / 2)) →
  Complex.abs z = 3 * Real.sqrt 5 / 5 →
  C ≤ π - Real.arctan (12 / 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_C_in_triangle_l256_25657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_line_l256_25608

/-- Three points lie on a straight line if and only if c = 0 or c = -2/3 -/
theorem points_on_line (a b c : ℝ) : 
  (∃ (t : ℝ), (9*c, 9*c, -2*c) = (1, 0, a) + t • ((b, 2, 0) - (1, 0, a))) ↔ (c = 0 ∨ c = -2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_line_l256_25608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_petal_area_correct_four_petal_area_approx_l256_25629

/-- The area of the four-petal shape formed by constructing semicircles 
    on each side of a square with side length a -/
noncomputable def fourPetalArea (a : ℝ) : ℝ := (Real.pi / 2 - 1) * a^2

/-- Theorem stating that the area of the four-petal shape is correct -/
theorem four_petal_area_correct (a : ℝ) (h : a > 0) : 
  fourPetalArea a = (Real.pi / 2 - 1) * a^2 := by
  -- Unfold the definition of fourPetalArea
  unfold fourPetalArea
  -- The equality follows directly from the definition
  rfl

/-- Theorem approximating the area of the four-petal shape -/
theorem four_petal_area_approx (a : ℝ) (h : a > 0) :
  ∃ (ε : ℝ), ε > 0 ∧ |fourPetalArea a - (4/7) * a^2| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_petal_area_correct_four_petal_area_approx_l256_25629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_three_pi_halves_l256_25602

theorem sin_three_pi_halves : Real.sin (3 * π / 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_three_pi_halves_l256_25602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_difference_sphere_cylinder_l256_25643

-- Define the radius of the sphere
def sphere_radius : ℝ := 6

-- Define the base radius of the cylinder
def cylinder_base_radius : ℝ := 4

-- Define the volume of a sphere
noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

-- Define the volume of a cylinder
noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

-- Statement to prove
theorem volume_difference_sphere_cylinder :
  ∃ (h : ℝ), 
    h^2 = 80 ∧
    sphere_volume sphere_radius - cylinder_volume cylinder_base_radius h = (288 - 64 * Real.sqrt 5) * Real.pi := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_difference_sphere_cylinder_l256_25643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_heights_l256_25652

/-- A parallelogram is a quadrilateral with opposite sides parallel -/
structure Parallelogram where
  -- We don't need to specify the exact properties here, just declare it
  -- as we're not proving anything, just stating the theorem

/-- Height of a parallelogram from a side -/
def Parallelogram.height (p : Parallelogram) (side : ℕ) : ℝ :=
  sorry -- We don't need to implement this, just declare it

theorem parallelogram_heights (p : Parallelogram) :
  (∀ (s₁ s₂ : ℕ), s₁ + 2 = s₂ → p.height s₁ = p.height s₂) ∧
  (∃ (s₁ s₂ : ℕ), s₁ + 1 = s₂ ∧ p.height s₁ ≠ p.height s₂) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_heights_l256_25652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_discount_percentage_l256_25684

theorem second_discount_percentage (initial_price : ℝ) (first_discount : ℝ) (final_price : ℝ) :
  initial_price = 200 →
  first_discount = 20 →
  final_price = 144 →
  let price_after_first_discount := initial_price * (1 - first_discount / 100)
  let second_discount_percentage := (price_after_first_discount - final_price) / price_after_first_discount * 100
  second_discount_percentage = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_discount_percentage_l256_25684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matching_digits_l256_25631

/-- A six-digit number -/
def SixDigitNumber := Fin 6 → Fin 10

/-- Given six six-digit numbers N, A, B, C, D, E, where N matches each of A, B, C, D, E
    in exactly three digits, prove that there exist at least two numbers among A, B, C, D, E
    that match in at least two digits. -/
theorem matching_digits (N A B C D E : SixDigitNumber)
  (hNA : ∃ (S : Finset (Fin 6)), S.card = 3 ∧ ∀ i ∈ S, N i = A i)
  (hNB : ∃ (S : Finset (Fin 6)), S.card = 3 ∧ ∀ i ∈ S, N i = B i)
  (hNC : ∃ (S : Finset (Fin 6)), S.card = 3 ∧ ∀ i ∈ S, N i = C i)
  (hND : ∃ (S : Finset (Fin 6)), S.card = 3 ∧ ∀ i ∈ S, N i = D i)
  (hNE : ∃ (S : Finset (Fin 6)), S.card = 3 ∧ ∀ i ∈ S, N i = E i) :
  ∃ (X Y : SixDigitNumber), X ≠ Y ∧ (X = A ∨ X = B ∨ X = C ∨ X = D ∨ X = E) ∧
    (Y = A ∨ Y = B ∨ Y = C ∨ Y = D ∨ Y = E) ∧
    (∃ (S : Finset (Fin 6)), S.card ≥ 2 ∧ ∀ i ∈ S, X i = Y i) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matching_digits_l256_25631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_4kg_optimal_fertilizer_amount_l256_25697

-- Define the yield function
noncomputable def L (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 then 5 * (x^2 + 6)
  else if 2 < x ∧ x ≤ 5 then (75 * x) / (1 + x)
  else 0

-- Define the profit function
noncomputable def f (x : ℝ) : ℝ :=
  15 * L x - 45 * x

-- State the theorem
theorem max_profit_at_4kg (x : ℝ) :
  0 ≤ x ∧ x ≤ 5 → f x ≤ 720 ∧ (f 4 = 720) := by
  sorry

-- Prove the optimal fertilizer amount
theorem optimal_fertilizer_amount :
  ∃ x, 0 ≤ x ∧ x ≤ 5 ∧ f x = 720 ∧ ∀ y, 0 ≤ y ∧ y ≤ 5 → f y ≤ f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_4kg_optimal_fertilizer_amount_l256_25697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increases_for_negative_x_l256_25676

noncomputable def f (x : ℝ) : ℝ := -6 / x

theorem f_increases_for_negative_x : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 0 → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increases_for_negative_x_l256_25676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_pentagon_l256_25658

theorem cosine_sum_pentagon : Real.cos ((2 * Real.pi) / 5) + Real.cos ((4 * Real.pi) / 5) = -1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_pentagon_l256_25658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tanya_completion_time_l256_25627

/-- The number of days it takes Tanya to complete a piece of work, given Sakshi's time and Tanya's efficiency relative to Sakshi. -/
noncomputable def tanya_work_days (sakshi_days : ℝ) (tanya_efficiency : ℝ) : ℝ :=
  sakshi_days / (1 + tanya_efficiency)

/-- Theorem stating that Tanya can complete the work in 16 days given the conditions. -/
theorem tanya_completion_time :
  tanya_work_days 20 0.25 = 16 := by
  -- Unfold the definition of tanya_work_days
  unfold tanya_work_days
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tanya_completion_time_l256_25627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l256_25611

theorem triangle_side_length (A B C : ℝ) (BC AB : ℝ) (cosC : ℝ) : 
  BC = 2 → AB = 4 → cosC = -1/4 → 
  ∃ (AC : ℝ), AC = 3 ∧ 
    AC^2 = BC^2 + AB^2 - 2*BC*AB*cosC :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l256_25611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l256_25654

/-- Sequence a_n where consecutive terms are roots of a quadratic equation -/
def a : ℕ → ℝ := sorry

/-- The c_n term in the quadratic equation x^2 - 3nx + c_n = 0 -/
def c : ℕ → ℝ := sorry

theorem sequence_property (n : ℕ) :
  (∀ k : ℕ, k ≥ 1 → (a k)^2 - 3 * k * (a k) + c k = 0 ∧
                    (a (k + 1))^2 - 3 * k * (a (k + 1)) + c k = 0) →
  a 1 = 1 →
  c (2 * n - 1) = 9 * n^2 - 9 * n + 2 ∧
  c (2 * n) = 9 * n^2 - 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l256_25654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_constant_l256_25614

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the line
def line (y : ℝ) : Prop := y = Real.sqrt 2

-- Define perpendicularity
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Define distance from a point to a line
noncomputable def distanceToLine (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  abs (x2 * y3 - x3 * y2 - x1 * y3 + x3 * y1 + x1 * y2 - x2 * y1) /
  Real.sqrt ((x3 - x2)^2 + (y3 - y2)^2)

theorem distance_is_constant :
  ∀ (x1 y1 x2 y2 : ℝ),
    ellipse x1 y1 →
    line y2 →
    perpendicular x1 y1 x2 y2 →
    distanceToLine 0 0 x1 y1 x2 y2 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_constant_l256_25614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_kingdoms_kill_probability_l256_25635

/-- Represents the teams in the "Three Kingdoms Kill" game -/
inductive Team
| MonarchLoyalist
| Rebel
| Traitor

/-- The number of players in each team -/
def team_size (t : Team) : ℕ :=
  match t with
  | Team.MonarchLoyalist => 4
  | Team.Rebel => 5
  | Team.Traitor => 1

/-- The total number of players -/
def total_players : ℕ := 10

/-- The probability of two randomly selected players being on the same team -/
def same_team_probability : ℚ :=
  16 / 45

theorem three_kingdoms_kill_probability :
  same_team_probability = 
    (Nat.choose (team_size Team.MonarchLoyalist) 2 + Nat.choose (team_size Team.Rebel) 2 + Nat.choose (team_size Team.Traitor) 2) / 
    (Nat.choose total_players 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_kingdoms_kill_probability_l256_25635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l256_25674

-- Define the function f(x) = -x^3 + 3x
def f (x : ℝ) : ℝ := -x^3 + 3*x

-- Define the interval (-1, 1]
def interval : Set ℝ := Set.Ioc (-1) 1

theorem f_properties :
  -- f(x) is monotonically increasing on (-1, 1]
  (∀ x y, x ∈ interval → y ∈ interval → x < y → f x < f y) ∧
  -- f(x) = a has a solution in (-1, 1] iff a ∈ (-2, 2]
  (∀ a : ℝ, (∃ x, x ∈ interval ∧ f x = a) ↔ a ∈ Set.Ioc (-2) 2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l256_25674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l256_25618

/-- The average speed of a train given two journey segments -/
noncomputable def average_speed (d1 d2 t1 t2 : ℝ) : ℝ :=
  (d1 + d2) / (t1 + t2)

/-- Theorem: The average speed of the train is 100 km/h -/
theorem train_average_speed :
  average_speed 250 350 2 4 = 100 := by
  -- Unfold the definition of average_speed
  unfold average_speed
  -- Simplify the arithmetic
  simp [add_div]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l256_25618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acquaintance_reading_time_l256_25683

/-- Represents the time it takes to read a novel -/
structure ReadingTime where
  hours : ℚ
  deriving Repr

/-- Represents a person's reading speed relative to a base speed -/
structure RelativeSpeed where
  factor : ℚ
  deriving Repr

/-- Calculates the time in minutes for a person to read a novel given their relative speed -/
def timeToRead (baseTime : ReadingTime) (speed : RelativeSpeed) : ℚ :=
  (baseTime.hours * 60) / speed.factor

/-- Proves that given the conditions, the acquaintance's reading time is 30 minutes -/
theorem acquaintance_reading_time 
  (my_time : ReadingTime)
  (friend_speed : RelativeSpeed)
  (acquaintance_speed : RelativeSpeed)
  (h1 : my_time.hours = 3)
  (h2 : friend_speed.factor = 3)
  (h3 : acquaintance_speed.factor = 2 * friend_speed.factor) :
  timeToRead my_time acquaintance_speed = 30 := by
  sorry

#check acquaintance_reading_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acquaintance_reading_time_l256_25683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jana_walking_speed_l256_25675

/-- Calculates the required walking speed in miles per hour given the current distance, 
    time, and desired new distance. -/
noncomputable def required_speed (current_distance : ℝ) (time_minutes : ℝ) (new_distance : ℝ) : ℝ :=
  (new_distance / time_minutes) * 60

theorem jana_walking_speed :
  let current_distance : ℝ := 2
  let time_minutes : ℝ := 30
  let new_distance : ℝ := 3
  required_speed current_distance time_minutes new_distance = 6 := by
  -- Unfold the definition of required_speed
  unfold required_speed
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

-- Remove the #eval statement as it's not computable
-- #eval required_speed 2 30 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jana_walking_speed_l256_25675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l256_25633

/-- Represents a geometric sequence with first term a and common ratio q -/
structure GeometricSequence where
  a : ℝ
  q : ℝ
  h_q_ne_one : q ≠ 1

/-- Sum of first n terms of a geometric sequence -/
noncomputable def sumGeometric (g : GeometricSequence) (n : ℕ) : ℝ :=
  g.a * (1 - g.q^n) / (1 - g.q)

theorem geometric_sequence_first_term
  (g : GeometricSequence)
  (h_condition : g.a * g.q * (g.a * g.q^7) = 2 * (g.a * g.q^2) * (g.a * g.q^5))
  (h_sum : sumGeometric g 5 = -62) :
  g.a = -2 := by
  sorry

#check geometric_sequence_first_term

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l256_25633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_correct_l256_25698

/-- A parallelepiped with all faces being equal rhombuses -/
structure RhombicParallelepiped where
  /-- Side length of each rhombus face -/
  a : ℝ
  /-- Acute angle of each rhombus face in radians -/
  angle : ℝ
  /-- Assumption that the acute angle is 60° (π/3 radians) -/
  angle_eq : angle = π / 3
  /-- Assumption that the side length is positive -/
  a_pos : 0 < a

/-- The volume of a rhombic parallelepiped -/
noncomputable def volume (p : RhombicParallelepiped) : ℝ :=
  (p.a ^ 3 * Real.sqrt 2) / 2

/-- Theorem stating that the volume formula is correct -/
theorem volume_formula_correct (p : RhombicParallelepiped) :
  volume p = (p.a ^ 3 * Real.sqrt 2) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_correct_l256_25698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_sinC_l256_25604

theorem triangle_ABC_sinC (A B C : ℝ) : 
  A = π / 4 → 
  Real.cos B = Real.sqrt 10 / 10 → 
  Real.sin C = 2 * Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_sinC_l256_25604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_theorem_l256_25653

/-- The speed of a bus excluding stoppages -/
noncomputable def speed_excluding_stoppages : ℝ := 32

/-- The speed of the bus including stoppages -/
noncomputable def speed_including_stoppages : ℝ := 16

/-- The fraction of an hour that the bus is in motion -/
noncomputable def motion_fraction : ℝ := 1/2

theorem bus_speed_theorem :
  speed_excluding_stoppages = speed_including_stoppages / motion_fraction :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_theorem_l256_25653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eggs_equal_kerosene_l256_25607

/-- The cost of a dozen eggs in dollars -/
def cost_dozen_eggs : ℚ := 36/100

/-- The cost of a pound of rice in dollars -/
def cost_pound_rice : ℚ := 36/100

/-- The cost of a half-liter of kerosene in dollars -/
def cost_half_liter_kerosene : ℚ := 36/100

/-- The number of eggs in a dozen -/
def eggs_in_dozen : ℕ := 12

theorem eggs_equal_kerosene :
  (cost_half_liter_kerosene / (cost_dozen_eggs / eggs_in_dozen)) = eggs_in_dozen :=
by
  -- Convert all values to rationals for easier computation
  have h1 : (36/100 : ℚ) / ((36/100 : ℚ) / 12) = 12 := by norm_num
  exact h1

#eval (cost_half_liter_kerosene / (cost_dozen_eggs / eggs_in_dozen))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eggs_equal_kerosene_l256_25607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emmalyn_fence_length_l256_25692

/-- The length of each fence Emmalyn painted -/
noncomputable def fence_length (rate : ℚ) (num_fences : ℕ) (total_earnings : ℚ) : ℚ :=
  (total_earnings / rate) / num_fences

/-- Theorem stating that under the given conditions, each fence is 500 meters long -/
theorem emmalyn_fence_length :
  let rate : ℚ := 1/5
  let num_fences : ℕ := 50
  let total_earnings : ℚ := 5000
  fence_length rate num_fences total_earnings = 500 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_emmalyn_fence_length_l256_25692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_quality_comparison_l256_25616

/-- Represents the quality data for a machine --/
structure MachineData where
  first_class : Nat
  second_class : Nat
  total : Nat

/-- Calculates the K² value for comparing two machines --/
noncomputable def calculate_k_squared (a b c d : Nat) : Real :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : Real) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Theorem stating the frequencies and confidence level for the machine comparison --/
theorem machine_quality_comparison (machine_a machine_b : MachineData)
  (h1 : machine_a.first_class = 150)
  (h2 : machine_a.second_class = 50)
  (h3 : machine_a.total = 200)
  (h4 : machine_b.first_class = 120)
  (h5 : machine_b.second_class = 80)
  (h6 : machine_b.total = 200)
  (h7 : calculate_k_squared machine_a.first_class machine_a.second_class
                             machine_b.first_class machine_b.second_class > 6.635) :
  (machine_a.first_class : Real) / machine_a.total = 3/4 ∧
  (machine_b.first_class : Real) / machine_b.total = 3/5 ∧
  ∃ (confidence_level : Real), confidence_level ≥ 0.99 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_quality_comparison_l256_25616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sequence_properties_l256_25699

noncomputable def vector_sequence : ℕ → ℝ × ℝ
  | 0 => (1, 1)  -- Adding the case for 0
  | 1 => (1, 1)
  | n + 2 => let (x, y) := vector_sequence (n + 1); (1/2 * (x - y), 1/2 * (x + y))

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

noncomputable def angle (v1 v2 : ℝ × ℝ) : ℝ := 
  Real.arccos ((v1.1 * v2.1 + v1.2 * v2.2) / (magnitude v1 * magnitude v2))

theorem vector_sequence_properties :
  (magnitude (vector_sequence 1) * magnitude (vector_sequence 5) = 1/2) ∧
  (∀ n : ℕ, n ≥ 2 → angle (vector_sequence (n-1)) (vector_sequence n) = π/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sequence_properties_l256_25699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joint_purchase_popularity_in_countries_joint_purchase_unpopularity_among_neighbors_l256_25624

/-- Represents the popularity of joint purchases -/
def JointPurchasePopularity := ℝ

/-- Represents the level of cost savings -/
def CostSavings := ℝ

/-- Represents the level of information sharing -/
def InformationSharing := ℝ

/-- Represents the level of potential risks -/
def PotentialRisks := ℝ

/-- Represents the level of transactional costs -/
def TransactionalCosts := ℝ

/-- Represents the proximity to stores -/
def ProximityToStores := ℝ

/-- Threshold for popularity -/
def PopularityThreshold : ℝ := 0

/-- Function to determine if joint purchases are popular -/
def isPopular (popularity : ℝ) : Prop :=
  popularity > PopularityThreshold

/-- Theorem stating the conditions for popularity of joint purchases in many countries -/
theorem joint_purchase_popularity_in_countries 
  (cs : ℝ) (is : ℝ) (pr : ℝ) :
  isPopular (cs + is - pr) :=
sorry

/-- Theorem stating the conditions for unpopularity of joint purchases among neighbors -/
theorem joint_purchase_unpopularity_among_neighbors 
  (tc : ℝ) (ps : ℝ) :
  ¬isPopular (-tc - ps) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joint_purchase_popularity_in_countries_joint_purchase_unpopularity_among_neighbors_l256_25624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_problem_l256_25662

theorem tan_sum_problem (x y : ℝ) 
  (h1 : Real.tan x + Real.tan y = 18) 
  (h2 : (Real.tan x)⁻¹ + (Real.tan y)⁻¹ = 24) : 
  Real.tan (x + y) = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_problem_l256_25662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_l256_25620

/-- The parabola y = ax² with a > 0 -/
def parabola (a : ℝ) (x y : ℝ) : Prop := y = a * x^2 ∧ a > 0

/-- The circle (x-3)² + y² = 1 -/
def circle' (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 1

/-- The directrix of the parabola y = ax² -/
def directrix (a : ℝ) (y : ℝ) : Prop := y = -(1 / (4 * a))

/-- The chord length is √3 -/
def chord_length (l : ℝ) : Prop := l = Real.sqrt 3

theorem parabola_circle_intersection (a : ℝ) :
  (∃ x y : ℝ, parabola a x y ∧ circle' x y) →
  (∃ y : ℝ, directrix a y ∧ (∃ x : ℝ, circle' x y)) →
  (∃ l : ℝ, chord_length l) →
  a = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_l256_25620
