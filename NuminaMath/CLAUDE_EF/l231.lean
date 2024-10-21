import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_result_l231_23122

/-- Represents a class of students with their corresponding numbers. -/
structure StudentClass where
  size : ℕ
  student_numbers : Finset ℕ
  h_size : student_numbers.card = size
  h_range : ∀ n ∈ student_numbers, 1 ≤ n ∧ n ≤ size

/-- Represents a systematic sampling of students from a class. -/
structure SystematicSampling (c : StudentClass) where
  sample_size : ℕ
  h_sample_size : sample_size > 0 ∧ sample_size ≤ c.size
  interval : ℕ
  h_interval : interval = c.size / sample_size
  selected : Finset ℕ
  h_selected : selected.card = sample_size
  h_systematic : ∀ n ∈ selected, ∃ k : ℕ, n = k * interval ∧ n ∈ c.student_numbers

/-- The theorem stating that the systematic sampling of 5 students from a class of 60 
    results in the selection of students numbered 6, 18, 30, 42, and 54. -/
theorem systematic_sampling_result (c : StudentClass) 
  (h_class_size : c.size = 60)
  (s : SystematicSampling c) 
  (h_sample_size : s.sample_size = 5) :
  s.selected = {6, 18, 30, 42, 54} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_result_l231_23122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unloaded_widgets_correct_l231_23167

def unloaded_widgets (total_product : Nat) 
  (doohicky_weight geegaw_weight widget_weight yamyam_weight : Nat) : Nat :=
  if total_product = 104350400 ∧ 
     doohicky_weight = 2 ∧ 
     geegaw_weight = 11 ∧ 
     widget_weight = 5 ∧ 
     yamyam_weight = 7 
  then 2
  else 0

#eval unloaded_widgets 104350400 2 11 5 7

theorem unloaded_widgets_correct (total_product : Nat) 
  (doohicky_weight geegaw_weight widget_weight yamyam_weight : Nat) :
  unloaded_widgets total_product doohicky_weight geegaw_weight widget_weight yamyam_weight = 2 :=
by
  unfold unloaded_widgets
  simp
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unloaded_widgets_correct_l231_23167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_approx_2_5_l231_23144

theorem equation_solution_approx_2_5 :
  ∃ X : ℝ, 
    (1.5 * ((3.6 * 0.48 * X) / (0.12 * 0.09 * 0.5)) = 1200.0000000000002) ∧ 
    (abs (X - 2.5) < 0.0000000000000005) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_approx_2_5_l231_23144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l231_23160

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 8 * Real.sin x - Real.tan x

-- State the theorem
theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 (Real.pi / 2) ∧
  (∀ (y : ℝ), y ∈ Set.Ioo 0 (Real.pi / 2) → f y ≤ f x) ∧
  f x = 3 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l231_23160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_condition_l231_23198

-- Define Necessary and Sufficient before using them in the theorem
def Necessary (P Q : Prop) : Prop :=
  Q → P

def Sufficient (P Q : Prop) : Prop :=
  P → Q

theorem necessary_not_sufficient_condition (a b : ℝ) :
  (Necessary ((a ≠ 1) ∨ (b ≠ 2)) (a + b ≠ 3)) ∧
  (¬Sufficient ((a ≠ 1) ∨ (b ≠ 2)) (a + b ≠ 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_condition_l231_23198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_problem_l231_23150

-- Define a power function
noncomputable def PowerFunction (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

-- State the theorem
theorem power_function_problem :
  ∀ f : ℝ → ℝ,
  (∃ α : ℝ, ∀ x : ℝ, f x = PowerFunction α x) →  -- f is a power function
  f 2 = 1/4 →                                    -- f passes through (2, 1/4)
  f 4 = 1/16 :=                                  -- conclusion: f(4) = 1/16
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_problem_l231_23150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l231_23138

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.log x - k * x

-- State the theorem
theorem f_properties (k : ℝ) :
  -- Part 1: Tangent line equation when k = 2
  (k = 2 → ∃ m b : ℝ, ∀ x y : ℝ, y = f 2 x → (x = 1 ∧ y = -2) ∨ y = m * x + b) ∧
  -- Part 2: Condition for f to have no zeros
  (∀ x > 0, f k x ≠ 0 ↔ k > Real.exp (-1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l231_23138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l231_23199

theorem equation_solution : ∃ (x y : ℝ), (16 : ℝ)^(x^2 + y) + (16 : ℝ)^(y^2 + x) = 1 ∧ x = -0.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l231_23199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_circle_centers_is_parabola_l231_23187

structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ → ℝ

def is_parabola_locus (S : Set (ℝ × ℝ)) (P : Parabola) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ S ↔ 
    dist (x, y) P.focus = |y - P.directrix x|

theorem locus_of_circle_centers_is_parabola 
  (fixed_circle : Set (ℝ × ℝ))
  (fixed_line : Set ℝ)
  (C : Set (ℝ × ℝ)) :
  (fixed_circle = {(x, y) | x^2 + (y-3)^2 = 1}) →
  (fixed_line = {y | y = 0}) →
  (∀ (c : ℝ × ℝ), c ∈ C ↔ 
    (∃ r > 0, (∀ (p : ℝ × ℝ), p ∈ fixed_circle → dist c p = r + 1) ∧
              (∀ y, y ∈ fixed_line → c.2 = r))) →
  ∃ P : Parabola, is_parabola_locus C P := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_circle_centers_is_parabola_l231_23187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_for_specific_pyramid_l231_23110

/-- A regular hexagonal pyramid inscribed in a sphere -/
structure HexagonalPyramid where
  height : ℝ
  outer_sphere_radius : ℝ

/-- The radius of the sphere inscribed in the hexagonal pyramid -/
noncomputable def inscribed_sphere_radius (p : HexagonalPyramid) : ℝ :=
  3 * (Real.sqrt 5 - 1) / 2

/-- Theorem stating the radius of the inscribed sphere for a specific pyramid -/
theorem inscribed_sphere_radius_for_specific_pyramid :
  let p : HexagonalPyramid := { height := 6, outer_sphere_radius := 4 }
  inscribed_sphere_radius p = 3 * (Real.sqrt 5 - 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_for_specific_pyramid_l231_23110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_l231_23133

/-- Given a hyperbola and a parabola with specific properties, prove that p = 2 -/
theorem hyperbola_parabola_intersection (a b p : ℝ) (ha : 0 < a) (hb : 0 < b) (hp : 0 < p) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → y^2 = 2*p*x) →  -- Hyperbola and parabola equations
  (b^2 / a^2 = 3) →  -- Eccentricity of hyperbola is 2
  (p^2 * Real.sqrt 3 / 2 = Real.sqrt 3) →  -- Area of triangle AOB is √3
  p = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_l231_23133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_in_range_l231_23181

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (6 - a * x) / Real.log a

-- State the theorem
theorem decreasing_f_implies_a_in_range :
  ∀ a : ℝ, (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 2 → f a x > f a y) → 1 < a ∧ a < 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_in_range_l231_23181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amber_max_ounces_l231_23153

def amber_money : ℚ := 7
def candy_price : ℚ := 1.25
def candy_ounces : ℕ := 12
def candy_stock : ℕ := 5
def chips_price : ℚ := 1.4
def chips_ounces : ℕ := 17
def chips_stock : ℕ := 4

def max_candy_bags : ℕ := Int.toNat (min (⌊amber_money / candy_price⌋) candy_stock)
def max_chips_bags : ℕ := Int.toNat (min (⌊amber_money / chips_price⌋) chips_stock)

def candy_total_ounces : ℕ := max_candy_bags * candy_ounces
def chips_total_ounces : ℕ := max_chips_bags * chips_ounces

theorem amber_max_ounces :
  max candy_total_ounces chips_total_ounces = 68 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amber_max_ounces_l231_23153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_f_l231_23125

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 3 then 2*x - x^2
  else if -2 ≤ x ∧ x < 0 then x^2 + 6*x
  else 0  -- This else case is added to make the function total

-- State the theorem about the domain of f
theorem domain_of_f :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ -2 ≤ x ∧ x ≤ 3 := by
  sorry

-- State the theorem about the range of f
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ -8 ≤ y ∧ y ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_f_l231_23125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_day_speed_is_four_l231_23120

/-- Represents the climbing problem with given conditions -/
structure ClimbingProblem where
  total_time : ℝ
  total_distance : ℝ
  speed_difference : ℝ
  time_difference : ℝ

/-- Calculates the speed on the second day given the climbing problem conditions -/
noncomputable def second_day_speed (problem : ClimbingProblem) : ℝ :=
  let first_day_time := (problem.total_time + problem.time_difference) / 2
  let second_day_time := problem.total_time - first_day_time
  let first_day_speed := (problem.total_distance - problem.speed_difference * second_day_time) / problem.total_time
  first_day_speed + problem.speed_difference

/-- Theorem stating that the speed on the second day is 4 km/h -/
theorem second_day_speed_is_four (problem : ClimbingProblem) 
    (h1 : problem.total_time = 14)
    (h2 : problem.total_distance = 52)
    (h3 : problem.speed_difference = 0.5)
    (h4 : problem.time_difference = 2) : 
  second_day_speed problem = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_day_speed_is_four_l231_23120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_nine_terms_l231_23121

/-- An arithmetic sequence with properties matching the problem -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (changed to ℚ for computability)
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  root_property : (a 4)^2 - 18*(a 4) + (a 4)*(a 6) = 0 ∧ (a 6)^2 - 18*(a 6) + (a 4)*(a 6) = 0

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The main theorem -/
theorem sum_of_nine_terms (seq : ArithmeticSequence) : S seq 9 = 81 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_nine_terms_l231_23121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l231_23117

-- Define the triangle
def Triangle (A B C a b c : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = Real.pi ∧
  0 < a ∧ 0 < b ∧ 0 < c

-- State the theorem
theorem triangle_properties 
  (A B C a b c : ℝ)
  (h_triangle : Triangle A B C a b c)
  (h_sine_ratio : Real.sin (A - B) / Real.cos B = Real.sin (A - C) / Real.cos C)
  (h_A : A = Real.pi / 3)
  (h_a_sin_C : a * Real.sin C = 1) :
  B = Real.pi / 3 ∧ 
  (∃ (x : ℝ), x = 1 / a^2 + 1 / b^2 ∧ x ≤ 25 / 16 ∧ 
    ∀ (y : ℝ), y = 1 / a^2 + 1 / b^2 → y ≤ 25 / 16) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l231_23117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l231_23105

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos (3 * x / 2), Real.sin (3 * x / 2))

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos (x / 2), -Real.sin (x / 2))

noncomputable def f (x : ℝ) : ℝ := 
  let a_vec := a x
  let b_vec := b x
  (a_vec.1 * b_vec.1 + a_vec.2 * b_vec.2) - 
  Real.sqrt ((a_vec.1 + b_vec.1)^2 + (a_vec.2 + b_vec.2)^2)

theorem f_max_min :
  ∀ x ∈ Set.Icc (-π/3) (π/4),
    f x ≤ -1 ∧ f x ≥ -3/2 ∧
    (∃ x₁ ∈ Set.Icc (-π/3) (π/4), f x₁ = -1) ∧
    (∃ x₂ ∈ Set.Icc (-π/3) (π/4), f x₂ = -3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l231_23105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_gt_c_gt_b_l231_23188

noncomputable section

variable (f : ℝ → ℝ)

-- f is an odd function
axiom f_odd : ∀ x, f (-x) = -f x

-- For x < 0, f(x) + xf'(x) < 0
axiom f_condition : ∀ x, x < 0 → f x + x * (deriv f x) < 0

-- Define a, b, and c
def a : ℝ := 3 * f 3
def b : ℝ := (Real.log 3 / Real.log Real.pi) * f (Real.log 3 / Real.log Real.pi)
def c : ℝ := -2 * f (-2)

theorem a_gt_c_gt_b : a > c ∧ c > b := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_gt_c_gt_b_l231_23188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_a_neg_one_range_of_a_non_negative_l231_23139

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * (1 - 1 / x)

-- Theorem for the minimum value when a = -1
theorem min_value_when_a_neg_one :
  ∀ x > 0, f (-1) x ≥ 0 ∧ f (-1) 1 = 0 := by sorry

-- Theorem for the range of a
theorem range_of_a_non_negative :
  (∀ x ≥ 1, f a x ≥ 0) ↔ a ≥ -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_a_neg_one_range_of_a_non_negative_l231_23139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_approx_l231_23103

/-- Represents the distance between points A and B in kilometers. -/
def distance : ℝ := sorry

/-- Represents the time taken for the entire journey in hours. -/
def total_time : ℝ := 19

/-- Represents the speed of the boat in still water in km/h. -/
def boat_speed : ℝ := 14

/-- Represents the stream velocity at point A in km/h. -/
def stream_velocity_A : ℝ := 4

/-- Represents the stream velocity at point B in km/h. -/
def stream_velocity_B : ℝ := 8

/-- Represents the stream velocity at point C (midway) in km/h. -/
def stream_velocity_C : ℝ := 6

/-- Theorem stating that the distance between points A and B is approximately 111 km. -/
theorem distance_AB_approx : ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ distance = 111 + ε := by
  sorry

#check distance_AB_approx

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_approx_l231_23103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l231_23148

noncomputable def f (x : ℝ) := Real.sqrt (2 * x - 1) / (x^2 - 1)

theorem domain_of_f :
  {x : ℝ | 2 * x - 1 ≥ 0 ∧ x^2 ≠ 1} = {x : ℝ | x ∈ Set.Icc (1/2) 1 ∪ Set.Ioi 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l231_23148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_to_midpoints_sum_is_zero_l231_23155

/-- Given a triangle ABC with centroid M and midpoints A₁, B₁, C₁ of sides BC, AC, and AB respectively,
    prove that the sum of vectors from M to each midpoint is the zero vector. -/
theorem centroid_to_midpoints_sum_is_zero (A B C M A₁ B₁ C₁ : ℝ × ℝ) : 
  A₁ = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) →
  B₁ = ((A.1 + C.1) / 2, (A.2 + C.2) / 2) →
  C₁ = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  M = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3) →
  (A₁.1 - M.1, A₁.2 - M.2) + (B₁.1 - M.1, B₁.2 - M.2) + (C₁.1 - M.1, C₁.2 - M.2) = (0, 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_to_midpoints_sum_is_zero_l231_23155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_equivalent_option_l231_23163

noncomputable section

-- Define the target value
def target : ℚ := 0.000045

-- Define the options
def option_a : ℚ := 45 / 10000
def option_b : ℚ := 45 / 1000000
def option_c : ℚ := 9 / (2 * 100000)
def option_d : ℚ := 9 / (200 * 10000)
def option_e : ℚ := 45 / 1000000

theorem not_equivalent_option : 
  (option_a = target) ∧
  (option_b = target) ∧
  (option_c = target) ∧
  (option_d ≠ target) ∧
  (option_e = target) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_equivalent_option_l231_23163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_state_a_selection_percentage_main_theorem_l231_23136

theorem state_a_selection_percentage 
  (total_candidates : ℕ) 
  (state_b_percentage : ℚ) 
  (difference : ℕ) : ℚ :=
  let state_b_selected := (state_b_percentage * total_candidates) / 100
  let state_a_selected := state_b_selected - difference
  (state_a_selected * 100) / total_candidates

#check @state_a_selection_percentage

theorem main_theorem : 
  state_a_selection_percentage 8400 7 84 = 6 := by
  unfold state_a_selection_percentage
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_state_a_selection_percentage_main_theorem_l231_23136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_abcd_l231_23107

/-- The side length of the original square -/
noncomputable def original_side_length : ℝ := 6

/-- The radius of the semicircles -/
noncomputable def semicircle_radius : ℝ := original_side_length / 2

/-- The side length of square ABCD -/
noncomputable def abcd_side_length : ℝ := original_side_length * Real.sqrt 2

/-- Square ABCD has its vertices tangent to the semicircles -/
axiom vertices_tangent_to_semicircles : True

/-- The sides of square ABCD are parallel to the original square -/
axiom sides_parallel : True

/-- The area of square ABCD -/
noncomputable def abcd_area : ℝ := abcd_side_length ^ 2

theorem area_of_abcd : abcd_area = 72 := by
  -- Expand the definition of abcd_area
  unfold abcd_area
  -- Expand the definition of abcd_side_length
  unfold abcd_side_length
  -- Simplify the expression
  simp [original_side_length]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_abcd_l231_23107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_properties_l231_23189

noncomputable def y (x : ℝ) : ℝ := Real.sin x + (Real.sin x)^2

theorem y_properties :
  (∀ x, -1/4 ≤ y x ∧ y x ≤ 2) ∧
  (∃ x, y x = 2) ∧
  (∃ x, y x = -1/4) ∧
  (∀ x, y (x + 2 * Real.pi) = y x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_properties_l231_23189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l231_23156

/-- The eccentricity of an ellipse with given conditions is between 1/2 and 1 -/
theorem ellipse_eccentricity_range (a b x₁ y₁ x₂ y₂ : ℝ) :
  a > b ∧ b > 0 ∧ x₁ ≠ x₂ ∧
  x₁^2 / a^2 + y₁^2 / b^2 = 1 ∧
  x₂^2 / a^2 + y₂^2 / b^2 = 1 ∧
  (x₁ - a/4)^2 + y₁^2 = (x₂ - a/4)^2 + y₂^2 →
  let e := Real.sqrt (1 - b^2/a^2)
  1/2 < e ∧ e < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l231_23156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l231_23192

-- Define the triangle
structure Triangle :=
  (A B C : Real)  -- angles
  (a b c : Real)  -- sides
  (m n : Real × Real)  -- vectors

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  t.m = (-Real.cos (t.A/2), Real.sin (t.A/2)) ∧
  t.n = (Real.cos (t.A/2), Real.sin (t.A/2)) ∧
  t.m.1 * t.n.1 + t.m.2 * t.n.2 = 1/2 ∧
  t.a = 2 * Real.sqrt 3 ∧
  1/2 * t.b * t.c * Real.sin t.A = Real.sqrt 3

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : triangle_conditions t) : 
  t.A = 2*Real.pi/3 ∧ t.b + t.c = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l231_23192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_line_equation_l231_23146

/-- Given a triangle ABC with vertex A at (-4, 2) and two medians on specific lines,
    prove that the equation of line BC is 2x + y - 8 = 0 -/
theorem triangle_line_equation (B C : ℝ × ℝ) : 
  let A : ℝ × ℝ := (-4, 2)
  let median1 : ℝ → ℝ → Prop := λ x y ↦ 3*x - 2*y + 2 = 0
  let median2 : ℝ → ℝ → Prop := λ x y ↦ 3*x + 5*y - 12 = 0
  let midpoint1 : ℝ × ℝ := ((A.1 + B.1)/2, (A.2 + B.2)/2)
  let midpoint2 : ℝ × ℝ := ((A.1 + C.1)/2, (A.2 + C.2)/2)
  median1 midpoint1.1 midpoint1.2 ∧ 
  median2 midpoint2.1 midpoint2.2 →
  ∀ (x y : ℝ), 2*x + y - 8 = 0 ↔ (∃ t : ℝ, x = B.1 + t*(C.1 - B.1) ∧ y = B.2 + t*(C.2 - B.2))
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_line_equation_l231_23146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_implies_a_range_no_parallel_tangents_l231_23178

noncomputable def f (a b x : ℝ) : ℝ := (1/2) * a * x^2 + b * x
noncomputable def g (x : ℝ) : ℝ := 1 + Real.log x

-- Part I
theorem monotonic_decreasing_interval_implies_a_range (a : ℝ) :
  a ≠ 0 →
  (∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ 
    ∀ (x : ℝ), x₁ < x ∧ x < x₂ → 
      (g x - f a 1 x) > (g x₂ - f a 1 x₂)) →
  a ∈ Set.Ioo (-1/4 : ℝ) 0 ∪ Set.Ioi 0 := by sorry

-- Part II
theorem no_parallel_tangents (a b : ℝ) :
  a ≠ 0 →
  ¬∃ (x₁ x₂ T : ℝ), 
    0 < x₁ ∧ x₁ < x₂ ∧
    g x₁ = f a b x₁ ∧
    g x₂ = f a b x₂ ∧
    T = (x₁ + x₂) / 2 ∧
    (1 / T) = (a * T + b) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_implies_a_range_no_parallel_tangents_l231_23178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_relationship_l231_23186

noncomputable def a : ℝ := Real.log 2 / 2
noncomputable def b : ℝ := Real.log 3 / 3
noncomputable def c : ℝ := Real.log 5 / 5

theorem abc_relationship : c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_relationship_l231_23186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l231_23157

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (Real.pi / 4 * Real.sin (x^2 + 2*x + 2 + Real.cos x))

theorem f_range :
  ∀ x : ℝ, Real.sqrt 2 ≤ f x ∧ f x ≤ 2 ∧
  (∃ y : ℝ, f y = Real.sqrt 2) ∧
  (∃ z : ℝ, f z = 2) := by
  sorry

#check f_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l231_23157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_lands_l231_23143

-- Define the propositions
variable (p q : Prop)

-- Define the meaning of p and q
def p_meaning (p : Prop) : Prop := p ↔ (∃ x, x = "Person A lands in the designated area")
def q_meaning (q : Prop) : Prop := q ↔ (∃ x, x = "Person B lands in the designated area")

-- Theorem to prove
theorem at_least_one_lands (p q : Prop) : 
  (∃ x, x = "At least one person lands in the designated area") ↔ (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_lands_l231_23143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l231_23140

/-- A hyperbola with center at the origin and foci on the y-axis -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The slope of an asymptote of a hyperbola -/
noncomputable def asymptote_slope (h : Hyperbola) : ℝ :=
  h.a / h.b

theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_asymptote : asymptote_slope h = Real.sqrt 3) :
  eccentricity h = 2 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l231_23140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_fourth_l231_23170

theorem cos_alpha_minus_pi_fourth (α : Real) 
  (h1 : α > 0) (h2 : α < π/2) (h3 : Real.tan α = 2) : 
  Real.cos (α - π/4) = (3 * Real.sqrt 10) / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_fourth_l231_23170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blocks_dont_fit_l231_23151

/-- Represents a rectangular block with integer dimensions -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ
deriving Inhabited

/-- Represents a rectangular box with integer dimensions -/
structure Box where
  length : ℕ
  width : ℕ
  height : ℕ
deriving Inhabited

/-- Calculates the volume of a block -/
def blockVolume (b : Block) : ℕ := b.length * b.width * b.height

/-- Calculates the volume of a box -/
def boxVolume (b : Box) : ℕ := b.length * b.width * b.height

/-- Checks if the total volume of blocks equals the volume of the box -/
def volumesMatch (blocks : List Block) (box : Box) : Prop :=
  (blocks.map blockVolume).sum = boxVolume box

/-- Checks if blocks can fit in the box based on face dimensions -/
def canFitInBox (blocks : List Block) (box : Box) : Prop :=
  blocks.length > 0 →
    (box.length * box.width) % blocks.head!.length = 0 ∧
    (box.length * box.height) % blocks.head!.length = 0 ∧
    (box.width * box.height) % blocks.head!.length = 0

/-- The main theorem stating that the blocks cannot fit in the box -/
theorem blocks_dont_fit (blocks : List Block) (box : Box) :
  blocks.length = 77 ∧
  (∀ b ∈ blocks, b.length = 3 ∧ b.width = 3 ∧ b.height = 1) ∧
  box.length = 7 ∧ box.width = 9 ∧ box.height = 11 ∧
  volumesMatch blocks box →
  ¬ canFitInBox blocks box := by
  sorry

#check blocks_dont_fit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blocks_dont_fit_l231_23151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_ball_surface_area_bowling_ball_volume_l231_23132

-- Define the diameter of the bowling ball
noncomputable def bowling_ball_diameter : ℝ := 9

-- Define the radius of the bowling ball
noncomputable def bowling_ball_radius : ℝ := bowling_ball_diameter / 2

-- Theorem for the surface area of the bowling ball
theorem bowling_ball_surface_area :
  4 * Real.pi * bowling_ball_radius ^ 2 = 81 * Real.pi := by
  sorry

-- Theorem for the volume of the bowling ball
theorem bowling_ball_volume :
  (4 / 3) * Real.pi * bowling_ball_radius ^ 3 = (729 * Real.pi) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_ball_surface_area_bowling_ball_volume_l231_23132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_inequality_l231_23173

-- Define the conditions
noncomputable def m (a b : ℝ) : ℝ := a + 1 / ((a - b) * b)

-- State the theorem
theorem min_value_and_inequality :
  (∃ (t : ℝ), t = 3 ∧ ∀ a b : ℝ, a > b → b > 0 → m a b ≥ t) ∧
  (∀ x y z : ℝ, x^2 + 4*y^2 + z^2 = 3 → |x + 2*y + z| ≤ 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_inequality_l231_23173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_half_l231_23185

/-- Right triangle ABC with AB = 10, BC = 6, and C as the right angle -/
structure RightTriangle where
  AB : ℝ
  BC : ℝ
  (AB_eq : AB = 10)
  (BC_eq : BC = 6)

/-- Point P within the triangle -/
structure PointInTriangle where
  P : ℝ × ℝ

/-- Area of a triangle given base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (1 / 2) * base * height

/-- Distance of a point from a line -/
noncomputable def distanceFromLine (P : ℝ × ℝ) (line : ℝ × ℝ → ℝ) : ℝ := sorry

/-- Probability of a point satisfying both conditions -/
noncomputable def probabilityOfConditions (t : RightTriangle) (p : PointInTriangle) : ℝ := sorry

/-- Theorem: The probability of a randomly placed point P within the right triangle ABC
    satisfying both conditions is equal to 1/2 -/
theorem probability_is_half (t : RightTriangle) (p : PointInTriangle) :
  probabilityOfConditions t p = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_half_l231_23185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_fruit_transport_l231_23161

/-- Represents the fruit transportation problem --/
structure FruitTransport where
  total_capacity : ℚ
  apple_capacity : ℚ
  orange_capacity : ℚ
  apple_profit : ℚ
  orange_profit : ℚ

/-- Calculates the number of orange trucks given the number of apple trucks --/
def orange_trucks (ft : FruitTransport) (x : ℚ) : ℚ :=
  (ft.total_capacity - ft.apple_capacity * x) / ft.orange_capacity

/-- Calculates the total profit given the number of apple trucks --/
def total_profit (ft : FruitTransport) (x : ℚ) : ℚ :=
  ft.apple_profit * ft.apple_capacity * x + ft.orange_profit * ft.orange_capacity * (orange_trucks ft x)

/-- Theorem stating the maximum profit for the fruit transportation problem --/
theorem max_profit_fruit_transport (ft : FruitTransport) 
  (h1 : ft.total_capacity = 60)
  (h2 : ft.apple_capacity = 4)
  (h3 : ft.orange_capacity = 6)
  (h4 : ft.apple_profit = 1200)
  (h5 : ft.orange_profit = 1500) :
  ∃ (x : ℚ), x ≥ orange_trucks ft x ∧ 
             total_profit ft x = 82800 ∧ 
             ∀ (y : ℚ), y ≥ orange_trucks ft y → total_profit ft y ≤ 82800 := by
  sorry

#eval let ft : FruitTransport := {
  total_capacity := 60,
  apple_capacity := 4,
  orange_capacity := 6,
  apple_profit := 1200,
  orange_profit := 1500
}
orange_trucks ft 6  -- Expected: 6

#eval let ft : FruitTransport := {
  total_capacity := 60,
  apple_capacity := 4,
  orange_capacity := 6,
  apple_profit := 1200,
  orange_profit := 1500
}
total_profit ft 6  -- Expected: 82800

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_fruit_transport_l231_23161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l231_23197

-- Define the power function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 1)^2 * x^(m^2 - 4*m + 2)

-- Define the function g
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := 2^x - k

-- Define the set A (range of f when x ∈ [1, 2])
def A (m : ℝ) : Set ℝ := { y | ∃ x ∈ Set.Icc 1 2, f m x = y }

-- Define the set B (range of g when x ∈ [1, 2])
def B (k : ℝ) : Set ℝ := { y | ∃ x ∈ Set.Icc 1 2, g k x = y }

-- Theorem for part I
theorem part_one : 
  (∀ x > 0, StrictMono (f m)) → m = 0 := by sorry

-- Theorem for part II
theorem part_two (m : ℝ) (h : m = 0) : 
  (∀ x > 0, StrictMono (f m)) → 
  (B k ⊆ A m) → 
  0 ≤ k ∧ k ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l231_23197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_sum_implies_values_l231_23145

-- Define the problem parameters
variable (m n : ℝ)
variable (a b : ℕ)
variable (x y : ℝ)

-- Define the condition that the sum is a monomial
def is_monomial_sum (m n : ℝ) (a b : ℕ) (x y : ℝ) : Prop :=
  ∃ (k : ℝ) (p q : ℕ), 2 * m * x^3 * y^b + (-5 * n * x^(2*a-3) * y) = k * x^p * y^q

-- State the theorem
theorem monomial_sum_implies_values (m n : ℝ) (a b : ℕ) (x y : ℝ) :
  is_monomial_sum m n a b x y →
  (a = 3 ∧ b = 1) ∧ (7 * a - 22)^2024 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_sum_implies_values_l231_23145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_not_always_true_l231_23142

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V]

-- Define lines and planes
def Line (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] := Set V
def Plane (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] := Set V

-- Define subset relation
def subset {α : Type*} (s t : Set α) := ∀ x, x ∈ s → x ∈ t

-- Define perpendicular relation (placeholder)
def perp (A B : Set V) : Prop := sorry

-- Define parallel relation (placeholder)
def parallel (A B : Set V) : Prop := sorry

-- Define the main theorem
theorem converse_not_always_true :
  ¬(∀ (b : Line V) (α β : Plane V), 
    (subset b α ∧ perp α β) → perp b β) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_not_always_true_l231_23142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_difference_zero_l231_23184

theorem factorial_difference_zero : Nat.factorial 7 - 6 * Nat.factorial 6 - Nat.factorial 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_difference_zero_l231_23184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_cans_total_liters_l231_23183

theorem oil_cans_total_liters : ∃ (total_liters : ℕ), total_liters = 290 := by
  -- Define the given values
  let total_cans : ℕ := 24
  let cans_with_8 : ℕ := 10
  let liters_in_8 : ℕ := 8
  let liters_in_15 : ℕ := 15

  -- Calculate total liters
  let total_liters : ℕ := cans_with_8 * liters_in_8 + (total_cans - cans_with_8) * liters_in_15

  -- Prove the theorem
  use total_liters
  show total_liters = 290
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_cans_total_liters_l231_23183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_union_B_equals_expected_result_l231_23109

-- Define sets A and B
def A : Set ℝ := {x | Real.log (x - 1) ≤ 0}
def B : Set ℝ := {x | x + 1/x ≤ 2}

-- Define the union of A and B
def AUnionB : Set ℝ := A ∪ B

-- Define the complement of A ∪ B in ℝ
def complementAUnionB : Set ℝ := Set.univ \ AUnionB

-- Define the expected result using Set.Icc and Set.Ioi
def expectedResult : Set ℝ := Set.Icc 0 1 ∪ Set.Ioi 2

-- Theorem statement
theorem complement_A_union_B_equals_expected_result :
  complementAUnionB = expectedResult := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_union_B_equals_expected_result_l231_23109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_isosceles_views_is_regular_triangular_pyramid_l231_23191

/-- A polygon representation -/
structure Polygon

/-- Predicate to check if a polygon is an isosceles triangle -/
def IsIsoscelesTriangle (p : Polygon) : Prop := sorry

/-- A solid with three views that are all isosceles triangles -/
structure ThreeIsoscelesViewSolid where
  /-- The solid has three views -/
  views : Fin 3 → Polygon
  /-- All three views are isosceles triangles -/
  all_isosceles : ∀ i, IsIsoscelesTriangle (views i)

/-- A regular triangular pyramid -/
structure RegularTriangularPyramid

/-- Theorem: A solid with three isosceles triangle views is a regular triangular pyramid -/
theorem three_isosceles_views_is_regular_triangular_pyramid :
  ThreeIsoscelesViewSolid → RegularTriangularPyramid := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_isosceles_views_is_regular_triangular_pyramid_l231_23191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_function_l231_23158

theorem min_value_of_function (x y : ℝ) (hx : 1/4 ≤ x ∧ x ≤ 3/5) (hy : 1/5 ≤ y ∧ y ≤ 1/2) :
  ∃ (min_val : ℝ), min_val = Real.sqrt 2 / 4 ∧
    ∀ (a b : ℝ), 1/4 ≤ a ∧ a ≤ 3/5 → 1/5 ≤ b ∧ b ≤ 1/2 →
      min_val ≤ (a * b) / (a^2 + 2 * b^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_function_l231_23158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_continuous_function_with_specific_periodic_orbits_l231_23154

/-- A continuous function on the unit interval -/
def ContinuousUnitInterval := {f : ℝ → ℝ // Continuous f ∧ ∀ x, x ∈ Set.Icc 0 1 → f x ∈ Set.Icc 0 1}

/-- A point x is periodic of order n for a function f -/
def IsPeriodic (f : ℝ → ℝ) (x : ℝ) (n : ℕ) : Prop :=
  (f^[n] x = x) ∧ ∀ m < n, f^[m] x ≠ x

/-- A function f has a periodic orbit of order n -/
def HasPeriodicOrbit (f : ℝ → ℝ) (n : ℕ) : Prop :=
  ∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧ IsPeriodic f x n

/-- Main theorem: There exists a continuous function on [0,1] with a periodic orbit of order 5 but no periodic orbit of order 3 -/
theorem exists_continuous_function_with_specific_periodic_orbits :
  ∃ (f : ContinuousUnitInterval), HasPeriodicOrbit f.val 5 ∧ ¬HasPeriodicOrbit f.val 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_continuous_function_with_specific_periodic_orbits_l231_23154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_in_triangle_l231_23195

open Real

theorem angle_measure_in_triangle (a b c : ℝ) (h : b^2 + c^2 - a^2 = Real.sqrt 3 * b * c) :
  Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_in_triangle_l231_23195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_log_divisors_10_to_11_l231_23165

-- Define the sum of logarithms of divisors function
noncomputable def sum_log_divisors (n : ℕ) : ℝ :=
  (n * (n + 1)^2) / 2

-- Theorem statement
theorem sum_log_divisors_10_to_11 :
  sum_log_divisors 11 = 792 := by
  -- Unfold the definition of sum_log_divisors
  unfold sum_log_divisors
  -- Simplify the expression
  simp [Nat.cast_mul, Nat.cast_add, Nat.cast_one]
  -- Check that the equation holds
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_log_divisors_10_to_11_l231_23165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_good_segments_correct_l231_23174

def min_good_segments (n : ℕ) : ℚ :=
  match n % 4 with
  | 0 => (n * (n - 4)) / 8
  | 1 => ((n - 1) * (n - 3)) / 8
  | 2 => ((n - 2) ^ 2) / 8
  | 3 => ((n - 1) * (n - 3)) / 8
  | _ => 0 -- This case is unreachable, but Lean requires it for completeness

-- We assume this function exists elsewhere in the codebase
noncomputable def min_number_of_good_segments (n : ℕ) : ℚ :=
  sorry -- Placeholder for the actual implementation

theorem min_good_segments_correct (n : ℕ) :
  min_good_segments n = min_number_of_good_segments n :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_good_segments_correct_l231_23174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_surrounding_coins_four_surrounding_coins_exist_l231_23147

/-- Represents a square coin on a table -/
structure SquareCoin where
  -- We don't need to define any specific properties for this problem

/-- Represents an arrangement of square coins around a central coin -/
structure CoinArrangement where
  centralCoin : SquareCoin
  surroundingCoins : List SquareCoin

/-- Represents that a surrounding coin touches the central coin -/
def touchesCentralCoin (coin : SquareCoin) (centralCoin : SquareCoin) : Prop :=
  sorry -- Definition not provided, as it's not essential for the proof

/-- Represents that a surrounding coin touches two other surrounding coins -/
def touchesTwoOthers (coin : SquareCoin) (surroundingCoins : List SquareCoin) : Prop :=
  sorry -- Definition not provided, as it's not essential for the proof

/-- Checks if a coin arrangement is valid according to the problem conditions -/
def isValidArrangement (arrangement : CoinArrangement) : Prop :=
  ∀ coin ∈ arrangement.surroundingCoins,
    (touchesCentralCoin coin arrangement.centralCoin) ∧
    (touchesTwoOthers coin arrangement.surroundingCoins)

/-- The main theorem stating that the maximum number of surrounding coins is 4 -/
theorem max_surrounding_coins :
  ∀ arrangement : CoinArrangement,
    isValidArrangement arrangement →
    arrangement.surroundingCoins.length ≤ 4 :=
by
  sorry

/-- The theorem stating that an arrangement with 4 surrounding coins exists -/
theorem four_surrounding_coins_exist :
  ∃ arrangement : CoinArrangement,
    isValidArrangement arrangement ∧
    arrangement.surroundingCoins.length = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_surrounding_coins_four_surrounding_coins_exist_l231_23147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l231_23193

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x + Real.cos x ^ 2 - 1/2

noncomputable def g (x : ℝ) : ℝ := f (x + 5 * Real.pi / 12)

theorem range_of_g :
  Set.range (fun x => g x) ∩ Set.Icc (-Real.pi/12) (Real.pi/3) = Set.Icc (-1) (1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l231_23193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_more_pups_than_adults_l231_23166

/-- Proves that the difference between the total number of pups and the total number of adult dogs is 30 -/
theorem more_pups_than_adults
  (num_huskies : ℕ)
  (num_pitbulls : ℕ)
  (num_golden_retrievers : ℕ)
  (husky_pups : ℕ)
  (golden_retriever_pups : ℕ)
  (h1 : num_huskies = 5)
  (h2 : num_pitbulls = 2)
  (h3 : num_golden_retrievers = 4)
  (h4 : husky_pups = 3)
  (h5 : golden_retriever_pups = husky_pups + 2) :
  (num_huskies * husky_pups + num_pitbulls * husky_pups + num_golden_retrievers * golden_retriever_pups) -
  (num_huskies + num_pitbulls + num_golden_retrievers) = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_more_pups_than_adults_l231_23166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_display_signal_count_l231_23152

/-- Represents a display with 4 parallel holes -/
def Display := Fin 4 → Bool

/-- Checks if two holes are adjacent -/
def adjacent (a b : Fin 4) : Bool :=
  (a.val + 1 = b.val) ∨ (b.val + 1 = a.val)

/-- A valid signal on the display -/
def ValidSignal (d : Display) : Prop :=
  (∃ (a b : Fin 4), a ≠ b ∧ ¬adjacent a b ∧ d a ∧ d b) ∧
  (∀ (x : Fin 4), d x → ∃ (y : Fin 4), x ≠ y ∧ d y) ∧
  (∀ (x y : Fin 4), d x → d y → adjacent x y → x = y)

/-- The set of all valid signals -/
def ValidSignals : Set Display :=
  {d | ValidSignal d}

/-- Instance to show that ValidSignals is finite -/
instance : Fintype ValidSignals := by
  sorry

theorem display_signal_count : Fintype.card ValidSignals = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_display_signal_count_l231_23152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_can_mark_all_rationals_l231_23118

/-- A function that represents the ability to mark points on a segment -/
def CanMark : (ℚ → Prop) → Prop :=
  λ f => ∀ q : ℚ, 0 < q → q < 1 → f q

/-- The set of ratios that Petya can mark -/
def PetyaMarkSet (q : ℚ) : Prop :=
  q = 1/2 ∨ ∃ n : ℕ, q = n / (n + 1)

/-- Theorem stating that Petya's marking ability is sufficient to mark any rational ratio -/
theorem petya_can_mark_all_rationals (h : CanMark PetyaMarkSet) :
  CanMark (λ q => q ∈ Set.Ioo (0 : ℚ) 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_can_mark_all_rationals_l231_23118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_mixture_percentage_l231_23179

/-- Represents the composition of an acid-water mixture -/
structure Mixture where
  acid : ℝ
  water : ℝ

/-- The percentage of acid in a mixture -/
noncomputable def acid_percentage (m : Mixture) : ℝ :=
  m.acid / (m.acid + m.water) * 100

theorem original_mixture_percentage (original : Mixture)
  (h1 : acid_percentage ⟨original.acid, original.water + 2⟩ = 25)
  (h2 : acid_percentage ⟨original.acid + 2, original.water + 4⟩ = 40) :
  acid_percentage original = 100/3 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_mixture_percentage_l231_23179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_proof_l231_23112

noncomputable def f (x : ℝ) : ℝ := Real.exp x

def tangent_line (x : ℝ) : ℝ := x + 1

theorem tangent_line_proof :
  -- The tangent line passes through (-1, 0)
  tangent_line (-1) = 0 ∧
  -- The tangent line touches f(x) at some point
  ∃ a : ℝ, f a = tangent_line a ∧
  -- The slope of the tangent line equals the derivative of f at the point of tangency
  (∀ x : ℝ, deriv f x = f x) ∧
  ∃ a : ℝ, deriv f a = 1 := by
  sorry

#check tangent_line_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_proof_l231_23112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_trajectory_l231_23135

noncomputable def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

noncomputable def Eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (b^2 / a^2))

theorem ellipse_equation_and_trajectory 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : (0, 4) ∈ Ellipse a b) 
  (h4 : Eccentricity a b = 3/5) :
  (∃ C : Set (ℝ × ℝ), C = Ellipse 5 4) ∧ 
  (∃ T : Set (ℝ × ℝ), T = {p : ℝ × ℝ | 16 * p.1^2 + 25 * p.2^2 - 48 * p.1 = 0}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_trajectory_l231_23135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_m_range_l231_23131

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin x * (Real.sin ((Real.pi + 2*x)/4))^2 - (Real.sin x)^2 + (Real.cos x)^2

-- Define the domain
def domain : Set ℝ := {x | 0 ≤ x ∧ x < 2*Real.pi}

-- Define set A
def A : Set ℝ := {x | Real.pi/6 ≤ x ∧ x ≤ 2*Real.pi/3}

-- Define set B
def B (m : ℝ) : Set ℝ := {x | -2 < f x - m ∧ f x - m < 2}

-- Theorem statement
theorem f_monotonicity_and_m_range :
  ∀ m : ℝ, (A ⊆ B m) → m ∈ Set.Ioo 1 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_m_range_l231_23131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_problem_l231_23106

theorem quadratic_roots_problem (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁^2 - 2*(m+1)*x₁ + m^2 + 2 = 0 ∧
   x₂^2 - 2*(m+1)*x₂ + m^2 + 2 = 0 ∧
   (x₁ + 1)*(x₂ + 1) = 8) → 
  m = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_problem_l231_23106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_four_squared_l231_23130

-- Define the square root function for non-negative real numbers
noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

-- State the theorem
theorem sqrt_four_squared : (sqrt 4)^2 = 4 := by
  -- Expand the definition of sqrt
  unfold sqrt
  -- Use the property of Real.sqrt
  rw [Real.sq_sqrt]
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_four_squared_l231_23130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_energy_correct_l231_23129

/-- Represents the energy stored between two charges -/
noncomputable def energy (distance : ℝ) (charge1 : ℝ) (charge2 : ℝ) : ℝ :=
  (charge1 * charge2) / distance

/-- The configuration of charges in a square -/
structure ChargeConfiguration where
  sideLength : ℝ
  charge : ℝ
  initialEnergy : ℝ

/-- Calculates the additional energy when one charge is moved to the center -/
noncomputable def additionalEnergy (config : ChargeConfiguration) : ℝ :=
  20 * (Real.sqrt 2 - 1)

theorem additional_energy_correct (config : ChargeConfiguration) 
  (h1 : config.sideLength > 0)
  (h2 : config.charge > 0)
  (h3 : config.initialEnergy = 20) :
  additionalEnergy config = 20 * (Real.sqrt 2 - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_energy_correct_l231_23129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l231_23102

-- Define the function f(x) = ln(x) + x
noncomputable def f (x : ℝ) : ℝ := Real.log x + x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 1 / x + 1

-- Theorem statement
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ m * x - y + b = 0) ∧
    (f' 1 = m) ∧
    (f 1 = 1 * m + b) ∧
    (m = 2) ∧
    (b = -1) := by
  sorry

#check tangent_line_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l231_23102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x1_for_2020_l231_23128

def sequence_rule (x : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 
    (x n % 2 = 0 → x (n + 1) = x n / 2) ∧ 
    (x n % 2 ≠ 0 → ∃ k : ℕ, 2^(k-1) ≤ x n ∧ x n < 2^k ∧ x (n + 1) = (x n - 1) / 2 + 2^(k-1))

def contains_2020 (x : ℕ → ℕ) : Prop :=
  ∃ n : ℕ, x n = 2020

theorem smallest_x1_for_2020 :
  ∀ x : ℕ → ℕ, sequence_rule x → contains_2020 x → x 1 ≥ 1183 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x1_for_2020_l231_23128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l231_23171

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 12 * x

/-- The circle equation -/
def circleEq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 6 * y = 0

/-- The intersection points of the parabola and the circle -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | parabola p.1 p.2 ∧ circleEq p.1 p.2}

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem stating that the distance between the intersection points is 6√3 -/
theorem intersection_distance :
  ∃ (p q : ℝ × ℝ), p ∈ intersection_points ∧ q ∈ intersection_points ∧ p ≠ q ∧
  distance p q = 6 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l231_23171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_median_ratio_l231_23175

theorem right_triangle_median_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  ∃ m : ℝ, m = Real.sqrt ((x^2 / 4 + y^2) / (y^2 / 4 + x^2)) ∧ (1 / 2 : ℝ) < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_median_ratio_l231_23175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cheese_equalization_l231_23137

def CheeseWeights := List Nat

def cut (weights : CheeseWeights) (i j : Nat) : CheeseWeights :=
  weights.mapIdx (fun k w => if k = i ∨ k = j then w - 1 else w)

def isEqual (weights : CheeseWeights) : Prop :=
  weights.all (· = weights.head!)

def canEqualize (initial : CheeseWeights) : Prop :=
  ∃ (n : Nat), ∃ (cuts : List (Nat × Nat)),
    cuts.length = n ∧
    (cuts.foldl (fun acc (i, j) => cut acc i j) initial).all (· > 0) ∧
    isEqual (cuts.foldl (fun acc (i, j) => cut acc i j) initial)

theorem cheese_equalization :
  canEqualize [5, 8, 11] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cheese_equalization_l231_23137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_commuting_polynomial_l231_23113

-- Define the type of polynomials with real coefficients
def RealPolynomial := Polynomial ℝ

-- Define the property that f(g(x)) = g(f(x)) for all polynomials g
def CommutesWithAll (f : RealPolynomial) : Prop :=
  ∀ g : RealPolynomial, Polynomial.comp f g = Polynomial.comp g f

-- Theorem statement
theorem unique_commuting_polynomial :
  ∃! f : RealPolynomial, CommutesWithAll f ∧ f = Polynomial.X :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_commuting_polynomial_l231_23113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaoliang_payment_l231_23124

/-- Calculates the discounted price based on the given rules --/
noncomputable def discountedPrice (x : ℝ) : ℝ :=
  if x < 200 then x
  else if x < 500 then 0.9 * x
  else 450 + 0.8 * (x - 500)

/-- Xiaoming's first purchase amount --/
def purchase1 : ℝ := 198

/-- Xiaoming's second purchase amount --/
def purchase2 : ℝ := 554

/-- Calculates the original price for Xiaoming's second purchase --/
noncomputable def originalPrice2 : ℝ := (554 - 450) / 0.8 + 500

/-- Theorem stating Xiaoliang's payment options --/
theorem xiaoliang_payment :
  let totalOriginalPrice1 := purchase1 + originalPrice2
  let totalOriginalPrice2 := purchase1 / 0.9 + originalPrice2
  discountedPrice totalOriginalPrice1 = 712.4 ∧
  discountedPrice totalOriginalPrice2 = 730 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaoliang_payment_l231_23124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_min_distance_ratio_l231_23149

/-- Parabola type -/
structure Parabola where
  f : ℝ → ℝ
  focus : ℝ × ℝ

/-- Point on a parabola -/
structure ParabolaPoint (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4 * x

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Ratio of distances -/
noncomputable def distance_ratio (p : Parabola) (pt : ParabolaPoint p) : ℝ :=
  distance (pt.x, pt.y) p.focus / distance (pt.x, pt.y) (-1, 0)

/-- Line equation type -/
inductive LineEquation where
  | pos : LineEquation
  | neg : LineEquation

theorem parabola_min_distance_ratio (p : Parabola) 
  (h1 : p.focus = (1, 0)) 
  (h2 : p.f = fun x ↦ Real.sqrt (4 * x)) : 
  ∃ (l : LineEquation), 
    (∀ (pt : ParabolaPoint p), distance_ratio p pt ≥ Real.sqrt (3/5)) ∧
    (∃ (pt : ParabolaPoint p), 
      distance_ratio p pt = Real.sqrt (3/5) ∧
      ((l = LineEquation.pos ∧ pt.x + pt.y + 1 = 0) ∨
       (l = LineEquation.neg ∧ pt.x - pt.y + 1 = 0))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_min_distance_ratio_l231_23149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_failing_both_tests_l231_23162

/-- The probability of students failing both tests -/
theorem probability_failing_both_tests
  (total : ℕ)
  (passed_first : ℕ)
  (passed_second : ℕ)
  (passed_both : ℕ)
  (h_total : total = 100)
  (h_first : passed_first = 60)
  (h_second : passed_second = 40)
  (h_both : passed_both = 20) :
  (total - (passed_first + passed_second - passed_both)) / total = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_failing_both_tests_l231_23162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_cube_volume_specific_l231_23127

/-- The volume of the largest cube that can be cut from a rectangular cuboid. -/
noncomputable def largest_cube_volume (width : ℝ) (length : ℝ) (height : ℝ) : ℝ :=
  (min width (min length height)) ^ 3

/-- Theorem: The volume of the largest cube that can be cut from a 15 cm × 12 cm × 8 cm cuboid is 512 cm³. -/
theorem largest_cube_volume_specific : largest_cube_volume 15 12 8 = 512 := by
  -- Unfold the definition of largest_cube_volume
  unfold largest_cube_volume
  -- Simplify the min expressions
  simp [min]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_cube_volume_specific_l231_23127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_long_rod_weight_l231_23196

/-- Represents the length of a rod in meters -/
def RodLength : Type := ℝ

/-- Represents the weight of a rod in kilograms -/
def RodWeight : Type := ℝ

/-- The weight of a rod is directly proportional to its length -/
axiom weight_proportional_to_length (l1 l2 w1 w2 : ℝ) :
  l1 * w2 = l2 * w1

/-- The length of the shorter rod -/
def short_rod_length : RodLength := (6 : ℝ)

/-- The length of the longer rod -/
def long_rod_length : RodLength := (12 : ℝ)

/-- The weight of the shorter rod -/
def short_rod_weight : RodWeight := (7 : ℝ)

/-- Theorem: The weight of the 12 meters long rod is 14 kg -/
theorem long_rod_weight : RodWeight := (14 : ℝ)

/-- Proof of the theorem -/
lemma prove_long_rod_weight : long_rod_weight = (14 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_long_rod_weight_l231_23196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_condition_l231_23180

variable (a : ℝ)

def f (x : ℝ) : ℝ := a * x^3 + 6 * x^2 + (a - 1) * x - 5

def has_extremum (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, ∀ x : ℝ, (f x ≤ f x₀) ∨ (f x ≥ f x₀)

theorem extremum_condition :
  has_extremum (f a) ↔ -3 < a ∧ a < 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_condition_l231_23180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_r_daily_earnings_l231_23159

/-- Daily earnings of p -/
def P : ℝ := sorry

/-- Daily earnings of q -/
def Q : ℝ := sorry

/-- Daily earnings of r -/
def R : ℝ := sorry

/-- p, q, and r together earn 1800 in 9 days -/
axiom total_earnings : (P + Q + R) * 9 = 1800

/-- p and r can earn 600 in 5 days -/
axiom p_r_earnings : (P + R) * 5 = 600

/-- q and r can earn 910 in 7 days -/
axiom q_r_earnings : (Q + R) * 7 = 910

/-- r's daily earnings are 50 -/
theorem r_daily_earnings : R = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_r_daily_earnings_l231_23159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_m_range_l231_23108

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 3

-- Theorem for monotonicity intervals
theorem monotonicity_intervals :
  (∀ x < -1, (f' x > 0)) ∧
  (∀ x ∈ Set.Ioo (-1) 1, (f' x < 0)) ∧
  (∀ x > 1, (f' x > 0)) :=
sorry

-- Theorem for the range of m
theorem m_range :
  ∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, 
    x₁ ∈ Set.Icc (-3/2) 3 ∧ 
    x₂ ∈ Set.Icc (-3/2) 3 ∧ 
    x₃ ∈ Set.Icc (-3/2) 3 ∧ 
    x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    f x₁ = m ∧ f x₂ = m ∧ f x₃ = m) ↔
  m ∈ Set.Ico (9/8) 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_m_range_l231_23108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_without_discount_theorem_l231_23119

/-- Calculates the profit percentage without discount, given the discount percentage and profit percentage with discount. -/
noncomputable def profit_without_discount (discount_percent : ℝ) (profit_with_discount_percent : ℝ) : ℝ :=
  let cost_price := 100
  let selling_price_with_discount := cost_price * (1 - discount_percent / 100)
  let profit_amount := cost_price * profit_with_discount_percent / 100
  let selling_price_without_discount := cost_price + profit_amount
  (selling_price_without_discount - cost_price) / cost_price * 100

/-- Theorem stating that given a 5% discount and a 19.7% profit, the profit percentage without discount would be 19.7%. -/
theorem profit_without_discount_theorem :
  profit_without_discount 5 19.7 = 19.7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_without_discount_theorem_l231_23119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l231_23190

/-- The area of a triangle with vertices (2, 3), (-3, -7), and (4, -2) is 22.5 square units. -/
theorem triangle_area : ∃ (area : ℝ), area = 22.5 ∧ area = abs ((2 - 4) * (-7 - (-2)) - (3 - (-2)) * (-3 - 4)) / 2 := by
  let A : ℝ × ℝ := (2, 3)
  let B : ℝ × ℝ := (-3, -7)
  let C : ℝ × ℝ := (4, -2)
  let area := abs ((A.1 - C.1) * (B.2 - C.2) - (A.2 - C.2) * (B.1 - C.1)) / 2
  use area
  constructor
  · sorry  -- Proof that area = 22.5
  · sorry  -- Proof that area equals the given formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l231_23190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_immovable_placement_l231_23182

/-- Represents a tile placement on the board -/
structure TilePlacement where
  tiles : Nat
  emptySquares : Nat

/-- Checks if a tile placement is valid -/
def isValidPlacement (p : TilePlacement) : Prop :=
  p.tiles > 0 ∧ 
  p.tiles * 2 + p.emptySquares = 64 ∧ 
  p.emptySquares ≤ 8

/-- Represents an empty square on the board -/
def isEmptySquare (p : TilePlacement) (i j : Nat) : Prop :=
  i < 8 ∧ j < 8 ∧ (i * 8 + j < p.emptySquares)

/-- Checks if a tile placement is immovable -/
def isImmovable (p : TilePlacement) : Prop :=
  p.emptySquares ≤ 8 ∧ 
  ∀ (i j : Nat), i < 8 ∧ j < 8 → 
    (i + 1 < 8 → ¬(isEmptySquare p i j ∧ isEmptySquare p (i + 1) j)) ∧
    (j + 1 < 8 → ¬(isEmptySquare p i j ∧ isEmptySquare p i (j + 1)))

/-- The theorem to be proved -/
theorem smallest_immovable_placement :
  ∃ (p : TilePlacement), 
    isValidPlacement p ∧ 
    isImmovable p ∧ 
    p.tiles = 28 ∧
    ∀ (q : TilePlacement), 
      isValidPlacement q ∧ isImmovable q → 
      p.tiles ≤ q.tiles :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_immovable_placement_l231_23182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_passenger_A_stop_A2_prob_different_stops_correct_l231_23172

-- Define the bus stops
inductive BusStop
| A0 | A1 | A2 | A3 | A4 | A5
deriving DecidableEq

-- Define a passenger
structure Passenger :=
  (stop : BusStop)

-- Define the probability of a passenger getting off at a specific stop
def prob_get_off (p : Passenger) (s : BusStop) : ℚ :=
  if s = BusStop.A0 then 0 else 1/5

-- Define the probability of two passengers getting off at different stops
def prob_different_stops : ℚ := 4/5

-- Theorem statements
theorem prob_passenger_A_stop_A2 :
  prob_get_off ⟨BusStop.A2⟩ BusStop.A2 = 1/5 := by
  rfl

theorem prob_different_stops_correct :
  prob_different_stops = 4/5 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_passenger_A_stop_A2_prob_different_stops_correct_l231_23172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equals_piecewise_l231_23111

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  if x ∈ Set.Icc (-4) (-1) then 3 * x + 2
  else if x ∈ Set.Icc (-1) 3 then -Real.sqrt (9 - (x - 1)^2) + 2
  else if x ∈ Set.Icc 3 5 then -3 * (x - 3) + 1
  else 0  -- Default value for x outside the defined ranges

-- Define the function h(x) = -2g(x) - 1
noncomputable def h (x : ℝ) : ℝ := -2 * g x - 1

-- Theorem statement
theorem h_equals_piecewise (x : ℝ) :
  (x ∈ Set.Icc (-4) (-1) → h x = -6 * x - 5) ∧
  (x ∈ Set.Icc (-1) 3 → h x = 2 * Real.sqrt (9 - (x - 1)^2) - 5) ∧
  (x ∈ Set.Icc 3 5 → h x = 6 * x - 21) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equals_piecewise_l231_23111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_triangle_area_l231_23169

-- Define the ellipse C
def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Define the circle
def circle_eq (b x y : ℝ) : Prop := x^2 + y^2 = b^2

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - y + Real.sqrt 6 = 0

-- Define the intersecting line
def intersecting_line (k m x y : ℝ) : Prop := y = k * x + m

-- Define the slope product condition
def slope_product (x₁ y₁ x₂ y₂ : ℝ) : Prop := (y₁ / x₁) * (y₂ / x₂) = -3/4

-- Define the area of triangle AOB
noncomputable def triangle_area (x₁ y₁ x₂ y₂ : ℝ) : ℝ := abs (x₁ * y₂ - x₂ * y₁) / 2

-- State the theorem
theorem constant_triangle_area 
  (a b k m x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : a > b) (h₂ : b > 0) 
  (h₃ : eccentricity a b = 1/2) 
  (h₄ : ∃ (x y : ℝ), circle_eq b x y ∧ tangent_line x y) 
  (h₅ : ellipse a b x₁ y₁ ∧ ellipse a b x₂ y₂) 
  (h₆ : intersecting_line k m x₁ y₁ ∧ intersecting_line k m x₂ y₂) 
  (h₇ : slope_product x₁ y₁ x₂ y₂) :
  triangle_area x₁ y₁ x₂ y₂ = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_triangle_area_l231_23169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_discount_is_twenty_percent_l231_23194

/-- Calculates the first discount percentage given the original price, final price, and second discount percentage. -/
noncomputable def calculateFirstDiscount (originalPrice finalPrice secondDiscountPercent : ℚ) : ℚ :=
  let priceAfterFirstDiscount := finalPrice / (1 - secondDiscountPercent / 100)
  (originalPrice - priceAfterFirstDiscount) / originalPrice * 100

/-- Theorem stating that for a saree with original price 200 rupees, 
    if it undergoes two successive discounts where the second discount is 10%, 
    and the final price is 144 rupees, then the first discount must have been 20%. -/
theorem first_discount_is_twenty_percent :
  calculateFirstDiscount 200 144 10 = 20 := by
  sorry

-- Remove the #eval statement as it's not necessary for building
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_discount_is_twenty_percent_l231_23194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_function_properties_l231_23123

/-- Given a function f(x) = x³ - ax where x=1 is an extremum point, 
    this theorem proves the explicit formula of f(x) and its maximum 
    and minimum values on the interval [0, 2]. -/
theorem extremum_function_properties (a : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = x^3 - a*x)
    (h2 : HasDerivAt f 0 1) :
  (∀ x, f x = x^3 - 3*x) ∧ 
  (∃ x ∈ Set.Icc 0 2, ∀ y ∈ Set.Icc 0 2, f y ≤ f x ∧ f x = 2) ∧
  (∃ x ∈ Set.Icc 0 2, ∀ y ∈ Set.Icc 0 2, f x ≤ f y ∧ f x = -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_function_properties_l231_23123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ammonium_chloride_formation_l231_23116

/-- Represents the number of moles of a substance -/
structure Moles where
  value : ℝ

/-- The number of moles of Hydrochloric acid available -/
def hcl_moles : Moles := ⟨3⟩

/-- The number of moles of Ammonium chloride formed -/
def nh4cl_moles : Moles := ⟨3⟩

/-- Theorem stating that the number of moles of Ammonium chloride formed
    is equal to the number of moles of Hydrochloric acid available -/
theorem ammonium_chloride_formation (hcl : Moles) (nh4cl : Moles)
    (h1 : hcl = hcl_moles)
    (h2 : nh4cl = nh4cl_moles)
    (h3 : hcl_moles = nh4cl_moles) :
    hcl = nh4cl := by
  rw [h1, h2, h3]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ammonium_chloride_formation_l231_23116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_between_two_and_three_l231_23168

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the theorem
theorem root_between_two_and_three (x : ℝ) : x^(floor x) = 8 → 2 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_between_two_and_three_l231_23168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_of_square_from_quartic_roots_l231_23126

noncomputable def is_square_vertices (a b c d : ℂ) : Prop :=
  ∃ (center : ℂ) (side_length : ℝ), 
    Complex.abs (a - center) = side_length ∧
    Complex.abs (b - center) = side_length ∧
    Complex.abs (c - center) = side_length ∧
    Complex.abs (d - center) = side_length ∧
    Complex.abs (b - a) = Complex.abs (c - b) ∧
    Complex.abs (c - b) = Complex.abs (d - c) ∧
    Complex.abs (d - c) = Complex.abs (a - d)

noncomputable def square_area (a b c d : ℂ) : ℝ :=
  let side := Complex.abs (b - a)
  side * side

theorem min_area_of_square_from_quartic_roots (p q r s : ℤ) : 
  let f : ℂ → ℂ := λ x => x^4 + p*x^3 + q*x^2 + r*x + s
  let roots := {z : ℂ | f z = 0}
  (∃ (a b c d : ℂ), roots = {a, b, c, d} ∧ is_square_vertices a b c d) →
  (∀ (a' b' c' d' : ℂ), roots = {a', b', c', d'} ∧ is_square_vertices a' b' c' d' → 
    square_area a' b' c' d' ≥ 2)
  :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_of_square_from_quartic_roots_l231_23126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_condition_implies_a_range_l231_23114

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (x + 1)^2

-- State the theorem
theorem function_condition_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ → f a x₁ - f a x₂ ≥ 4 * (x₁ - x₂)) ↔ a ≥ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_condition_implies_a_range_l231_23114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l231_23104

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 2) + 1 / (x - 1)

-- Define the domain of f
def domain_f : Set ℝ := {x | x ≥ -2 ∧ x ≠ 1}

-- Theorem statement
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = domain_f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l231_23104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_al_oh_3_concentration_l231_23141

/-- Molar concentration of a solute in a solution -/
noncomputable def molarConcentration (moles : ℝ) (volume : ℝ) : ℝ :=
  moles / volume

/-- The problem statement -/
theorem al_oh_3_concentration :
  let moles : ℝ := 1.2
  let volume : ℝ := 3.5
  let concentration := molarConcentration moles volume
  ∃ ε > 0, |concentration - 0.3429| < ε ∧ ε < 0.00005 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_al_oh_3_concentration_l231_23141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l231_23115

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi - x) * Real.sin (Real.pi / 2 - x) + Real.cos x ^ 2

-- State the theorem
theorem f_properties :
  -- The smallest positive period is π
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  -- The function is monotonically increasing on [-π/8, π/8] when x ∈ [-π/8, 3π/8]
  (∀ (x y : ℝ), -Real.pi/8 ≤ x ∧ x ≤ y ∧ y ≤ Real.pi/8 → f x ≤ f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l231_23115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trigonometric_expression_l231_23134

theorem min_value_trigonometric_expression (θ : Real) (h : 0 < θ ∧ θ < Real.pi / 2) :
  3 * Real.cos θ + 2 / Real.sin θ + 2 * Real.sqrt 3 * (Real.cos θ / Real.sin θ) ≥ 6 * (2 * Real.sqrt 3) ^ (1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trigonometric_expression_l231_23134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l231_23177

noncomputable def f (x : ℝ) := (x - 3) / (x^2 - 4*x + 13)

theorem solution_set_of_inequality :
  {x : ℝ | f x ≥ 0} = {x : ℝ | x ≥ 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l231_23177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_fourth_quadrant_l231_23176

theorem sin_double_angle_fourth_quadrant (θ : ℝ) :
  θ ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi) →  -- fourth quadrant
  Real.cos θ = 4/5 →
  Real.sin (2 * θ) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_fourth_quadrant_l231_23176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l231_23164

-- Define the conditions
def P (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0 ∧ a > 0

def Q (x : ℝ) : Prop := 8 < (2 : ℝ)^(x + 1) ∧ (2 : ℝ)^(x + 1) ≤ 16

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (P Q : ℝ → Prop) : Prop :=
  (∀ x, Q x → P x) ∧ ¬(∀ x, P x → Q x)

-- State the theorem
theorem range_of_a (a : ℝ) :
  sufficient_not_necessary (P · a) Q →
  (1 < a ∧ a ≤ 2) ↔ (∃ x, P x a) ∧ (∀ x, Q x → P x a) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l231_23164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_third_quadrant_l231_23100

-- Define the complex number z
variable (z : ℂ)

-- Define the condition
def condition (z : ℂ) : Prop := (1 + Complex.I) = (1 - 3 * Complex.I) / (2 * z)

-- Theorem statement
theorem z_in_third_quadrant (h : condition z) : 
  z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_third_quadrant_l231_23100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_phi_equals_sqrt19_div_10_l231_23101

-- Define the square ABCD
def Square (A B C D : ℝ × ℝ) : Prop :=
  A = (0, 0) ∧ B = (0, 4) ∧ C = (4, 4) ∧ D = (4, 0)

-- Define point P on BC
def P : ℝ × ℝ := (3, 4)

-- Define point Q on CD
def Q : ℝ × ℝ := (4, 2)

-- Define the angle φ between AP and AQ
noncomputable def φ (A P Q : ℝ × ℝ) : ℝ := Real.arccos (
  ((P.1 - A.1)*(Q.1 - A.1) + (P.2 - A.2)*(Q.2 - A.2)) /
  (Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) * Real.sqrt ((Q.1 - A.1)^2 + (Q.2 - A.2)^2))
)

theorem sin_phi_equals_sqrt19_div_10 (A B C D : ℝ × ℝ) :
  Square A B C D →
  Real.sin (φ A P Q) = Real.sqrt 19 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_phi_equals_sqrt19_div_10_l231_23101
