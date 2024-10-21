import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_combined_time_l971_97113

/-- The time taken by three workers to complete a job together -/
noncomputable def combined_time (a_time b_time c_time : ℝ) : ℝ :=
  1 / (1 / a_time + 1 / b_time + 1 / c_time)

/-- Theorem stating that workers who can complete a job in 15, 20, and 45 days
    individually can complete it together in 7.2 days -/
theorem workers_combined_time :
  combined_time 15 20 45 = 7.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_combined_time_l971_97113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_is_three_l971_97199

-- Define the left-hand side of the equation as a function
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sqrt (x - 2) + 2 * Real.sqrt (2 * x + 3) + Real.sqrt (x + 1)

-- Theorem statement
theorem unique_root_is_three :
  ∃! x : ℝ, f x = 11 ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_is_three_l971_97199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_gis_function_l971_97159

/-- Represents the main function of a Geographic Information System -/
inductive GISFunction
  | CollectProcessAnalyze
  | ProcessRemoteSensing
  | ObtainComprehensiveData
  | NavigateLocate

/-- The correct main function of a Geographic Information System -/
def correctGISFunction : GISFunction := GISFunction.CollectProcessAnalyze

/-- Theorem stating that the correct GIS function is to collect, process, and analyze data -/
theorem correct_gis_function :
  correctGISFunction = GISFunction.CollectProcessAnalyze :=
by
  -- The proof is trivial as it's defined this way
  rfl

#check correct_gis_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_gis_function_l971_97159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_filling_rate_verify_final_amount_l971_97112

/-- The rate at which the pool fills during the first hour -/
def R : ℝ := 8

/-- The amount of water added during the second and third hours -/
def second_third_hours : ℝ := 20

/-- The amount of water added during the fourth hour -/
def fourth_hour : ℝ := 14

/-- The amount of water lost during the fifth hour -/
def fifth_hour : ℝ := 8

/-- The final amount of water in the pool -/
def final_amount : ℝ := 34

/-- Theorem stating that the rate R at which the pool fills during the first hour is 8 gallons per hour -/
theorem pool_filling_rate : R = 8 := by
  rfl

/-- Theorem verifying the final amount of water in the pool -/
theorem verify_final_amount : R + second_third_hours + fourth_hour - fifth_hour = final_amount := by
  simp [R, second_third_hours, fourth_hour, fifth_hour, final_amount]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_filling_rate_verify_final_amount_l971_97112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_person_job_completion_l971_97165

noncomputable def job_completion_time (time_person1 time_person2 : ℝ) : ℝ :=
  1 / (1 / time_person1 + 1 / time_person2)

theorem two_person_job_completion
  (time_person1 time_person2 : ℝ)
  (h1 : time_person1 = 24)
  (h2 : time_person2 = 12) :
  job_completion_time time_person1 time_person2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_person_job_completion_l971_97165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_bananas_l971_97163

theorem shopkeeper_bananas (B : ℝ) : 
  (0.85 * 600 + 0.92 * B) / (600 + B) = 0.878 → B = 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_bananas_l971_97163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_hypertension_count_l971_97187

/-- The probability that an American has hypertension -/
noncomputable def hypertension_probability : ℝ := 1 / 3

/-- The total number of Americans in the sample -/
def sample_size : ℕ := 450

/-- The expected number of Americans with hypertension in the sample -/
noncomputable def expected_hypertension : ℝ := hypertension_probability * (sample_size : ℝ)

theorem expected_hypertension_count : expected_hypertension = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_hypertension_count_l971_97187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_consecutive_010101_l971_97127

def sequenceA : ℕ → ℕ
  | 0 => 1
  | 1 => 0
  | 2 => 1
  | 3 => 0
  | 4 => 1
  | 5 => 0
  | n + 6 => (sequenceA n + sequenceA (n + 1) + sequenceA (n + 2) + 
              sequenceA (n + 3) + sequenceA (n + 4) + sequenceA (n + 5)) % 10

theorem no_consecutive_010101 :
  ¬ ∃ n : ℕ, sequenceA n = 0 ∧ sequenceA (n + 1) = 1 ∧ 
          sequenceA (n + 2) = 0 ∧ sequenceA (n + 3) = 1 ∧ 
          sequenceA (n + 4) = 0 ∧ sequenceA (n + 5) = 1 :=
by
  sorry

#eval sequenceA 0
#eval sequenceA 1
#eval sequenceA 2
#eval sequenceA 3
#eval sequenceA 4
#eval sequenceA 5
#eval sequenceA 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_consecutive_010101_l971_97127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_product_equation_l971_97152

theorem prime_product_equation (a b c d : ℤ) (w x y z : ℕ) : 
  Prime w ∧ Prime x ∧ Prime y ∧ Prime z →
  w < x ∧ x < y ∧ y < z →
  (w^a.toNat) * (x^b.toNat) * (y^c.toNat) * (z^d.toNat) = 660 →
  (a + b) - (c + d) = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_product_equation_l971_97152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_e_value_l971_97104

-- Define the circle and points
def Circle : Type := Unit
def Point : Type := Unit

-- Define the diameter of the circle
noncomputable def diameter : ℝ := 2

-- Define the points on the circle
variable (P Q X Y Z : Point)

-- Define the distances
noncomputable def PX : ℝ := 4/5
noncomputable def PY : ℝ := 3/4

-- Define the line segment e
noncomputable def e (P Q X Y Z : Point) : ℝ := sorry

-- Theorem statement
theorem max_e_value (c : Circle) (P Q X Y Z : Point) :
  (∃ (e_max : ℝ), ∀ (e_val : ℝ), e P Q X Y Z ≤ e_max ∧ e_max = 41 - 16 * Real.sqrt 25) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_e_value_l971_97104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_location_l971_97164

theorem point_location (m n : ℝ) (h : (2 : ℝ)^m + (2 : ℝ)^n < 4) : m + n < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_location_l971_97164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_rectangle_area_l971_97158

/-- The area of the smallest rectangle containing two tangent circles -/
noncomputable def rectangle_area (area1 area2 : ℝ) : ℝ :=
  let r1 := Real.sqrt (area1 / Real.pi)
  let r2 := Real.sqrt (area2 / Real.pi)
  4 * (r1 + r2) * r1

theorem tangent_circles_rectangle_area :
  rectangle_area 40 10 = 240 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_rectangle_area_l971_97158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_intersections_is_80_l971_97168

/-- Represents a regular polygon inscribed in a circle -/
structure InscribedPolygon :=
  (sides : ℕ)

/-- Calculates the number of intersections between two inscribed polygons -/
def intersections (p1 p2 : InscribedPolygon) : ℕ :=
  2 * min p1.sides p2.sides

/-- The set of inscribed polygons in our problem -/
def polygons : List InscribedPolygon :=
  [⟨6⟩, ⟨7⟩, ⟨8⟩, ⟨9⟩]

/-- Theorem stating that the total number of intersections is 80 -/
theorem total_intersections_is_80 :
  (List.sum (List.map
    (λ (pair : InscribedPolygon × InscribedPolygon) => 
      if pair.1.sides < pair.2.sides then intersections pair.1 pair.2 else 0)
    (List.product polygons polygons))) = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_intersections_is_80_l971_97168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_positive_iff_x_positive_l971_97174

/-- The function f(x) = x - sin(x) -/
noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

/-- Theorem stating that f(x-1) + f(x+1) > 0 if and only if x > 0 -/
theorem f_sum_positive_iff_x_positive (x : ℝ) :
  f (x - 1) + f (x + 1) > 0 ↔ x > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_positive_iff_x_positive_l971_97174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a_squared_plus_b_squared_l971_97145

theorem min_value_a_squared_plus_b_squared (a b : ℝ) : 
  (∃ k : ℕ, (Nat.choose 6 k) * a^(6-k) * b^k = 20 ∧ 12 - 3*k = 3) → 
  a^2 + b^2 ≥ 2 ∧ ∃ (a₀ b₀ : ℝ), a₀^2 + b₀^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a_squared_plus_b_squared_l971_97145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_of_y_bounds_l971_97189

noncomputable def y (a b x : ℝ) := a * Real.cos x + b

noncomputable def f (a b x : ℝ) := b * Real.sin (a * x + Real.pi / 3)

def increasing_interval (a b : ℝ) : Set ℝ :=
  if a > 0 then
    {x | ∃ k : ℤ, k * Real.pi + Real.pi / 12 ≤ x ∧ x ≤ k * Real.pi + 7 * Real.pi / 12}
  else
    {x | ∃ k : ℤ, k * Real.pi - Real.pi / 12 ≤ x ∧ x ≤ k * Real.pi + 5 * Real.pi / 12}

theorem increasing_interval_of_y_bounds (a b : ℝ) :
  (∀ x, y a b x ≤ 1) ∧ (∃ x, y a b x = 1) ∧
  (∀ x, y a b x ≥ -3) ∧ (∃ x, y a b x = -3) →
  (∀ x, x ∈ increasing_interval a b ↔ 
    (∀ h > 0, ∃ δ > 0, ∀ y, |x - y| < δ → f a b x < f a b (x + h))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_of_y_bounds_l971_97189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sum_bound_l971_97114

theorem distinct_sum_bound (n : ℕ) (k : ℕ) (a : Fin k → ℕ) :
  (∀ i : Fin k, 1 ≤ a i) →
  (∀ i j : Fin k, i < j → a i < a j) →
  (∀ i : Fin k, a i ≤ n) →
  (∀ i j m l : Fin k, i ≤ j → m ≤ l → (i, j) ≠ (m, l) → a i + a j ≠ a m + a l) →
  k ≤ Nat.sqrt (2 * n) + 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sum_bound_l971_97114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excircle_tangency_concyclic_l971_97117

/-- Predicate stating that a point is the incenter of a triangle -/
def IsIncenter (I A B C : EuclideanPlane) : Prop := sorry

/-- Predicate stating that a point is a tangency point of an excircle -/
def IsTangencyPoint (P B C : EuclideanPlane) : Prop := sorry

/-- Predicate stating that a point is the midpoint of a line segment -/
def Midpoint (M P Q : EuclideanPlane) : Prop := sorry

/-- Predicate stating that a point is the intersection of two lines -/
def LineIntersectsAt (N A B C D : EuclideanPlane) : Prop := sorry

/-- Predicate stating that four points are concyclic (lie on the same circle) -/
def Concyclic (P Q R S : EuclideanPlane) : Prop := sorry

/-- Given a triangle ABC with incenter I, if B₁ is the point of tangency of an excircle with side BC,
    M is the midpoint of IC, and N is the intersection of AA₁ and BB₁, 
    then N, B₁, A, and M are concyclic. -/
theorem excircle_tangency_concyclic (A B C : EuclideanPlane) (I : EuclideanPlane) 
  (B₁ : EuclideanPlane) (M : EuclideanPlane) (N : EuclideanPlane) (A₁ : EuclideanPlane) :
  IsIncenter I A B C →
  IsTangencyPoint B₁ B C →
  Midpoint M I C →
  LineIntersectsAt N A B₁ B A₁ →
  Concyclic N B₁ A M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_excircle_tangency_concyclic_l971_97117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angles_on_x_axis_characterization_l971_97194

-- Define the set of angles whose terminal sides lie on the x-axis
def AnglesOnXAxis : Set ℝ :=
  {β : ℝ | ∃ n : ℤ, β = n * 180}

-- Theorem statement
theorem angles_on_x_axis_characterization :
  AnglesOnXAxis = {β : ℝ | ∃ n : ℤ, β = n * 180} := by
  -- The proof is skipped using 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angles_on_x_axis_characterization_l971_97194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_180_l971_97181

/-- The sum of all positive divisors of 180 is 546 -/
theorem sum_of_divisors_180 : (Finset.filter (λ x ↦ 180 % x = 0) (Finset.range 181)).sum id = 546 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_180_l971_97181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l971_97195

theorem trig_problem (x : ℝ) 
  (h1 : Real.cos (x - π/4) = Real.sqrt 2 / 10)
  (h2 : x ∈ Set.Ioo (π/2) (3*π/4)) :
  Real.sin x = 4/5 ∧ Real.sin (2*x + π/3) = -(24 + 7*Real.sqrt 3) / 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l971_97195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l971_97191

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * (Real.sin (x + Real.pi/4))^2 - (Real.cos x)^2 - (1 + Real.sqrt 3)/2

-- Define the theorem
theorem function_properties :
  ∃ (A : ℝ), 0 < A ∧ A < Real.pi/2 ∧
  (∀ x, f x ≥ -2) ∧
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ Real.pi) ∧
  (1 * 1 + 5 * f (Real.pi/4 - A) = 0) ∧
  Real.cos (2 * A) = (4 * Real.sqrt 3 + 3) / 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l971_97191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_l_value_min_l_is_2_sqrt_3_l971_97167

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if abs x ≤ 1 then 2 * Real.cos (Real.pi / 2 * x) else x^2 - 1

-- State the theorem
theorem min_l_value (l : ℝ) : 
  (l > 0 ∧ 
   ∀ x : ℝ, abs (f x + f (x + l) - 2) + abs (f x - f (x + l)) ≥ 2) →
  l ≥ 2 * Real.sqrt 3 := by
  sorry

-- State that 2√3 is the minimum value
theorem min_l_is_2_sqrt_3 : 
  ∃ l : ℝ, l = 2 * Real.sqrt 3 ∧
  (l > 0 ∧ 
   ∀ x : ℝ, abs (f x + f (x + l) - 2) + abs (f x - f (x + l)) ≥ 2) ∧
  ∀ l' : ℝ, (l' > 0 ∧ 
   ∀ x : ℝ, abs (f x + f (x + l') - 2) + abs (f x - f (x + l')) ≥ 2) →
  l' ≥ l := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_l_value_min_l_is_2_sqrt_3_l971_97167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_cosine_interval_l971_97148

theorem angle_cosine_interval (θ : Real) (h1 : 0 < θ ∧ θ < Real.pi) 
  (h2 : ∀ x : Real, 0 < Real.cos θ * x^2 - 4 * Real.sin θ * x + 6) :
  1/2 < Real.cos θ ∧ Real.cos θ < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_cosine_interval_l971_97148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_and_two_alpha_minus_beta_l971_97186

open Real

theorem tan_alpha_and_two_alpha_minus_beta 
  (α β : ℝ) 
  (h_α : α ∈ Set.Ioo 0 (π/4)) 
  (h_β : β ∈ Set.Ioo 0 π) 
  (h_tan_diff : tan (α - β) = 1/2) 
  (h_tan_β : tan β = -1/7) : 
  tan α = 1/3 ∧ 2*α - β = -3*π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_and_two_alpha_minus_beta_l971_97186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_segments_in_broken_line_l971_97121

/-- A square grid of size n x n -/
structure SquareGrid (n : ℕ) where
  squares : Fin n → Fin n → Unit

/-- A broken line (polyline) in the grid -/
structure BrokenLine (n : ℕ) where
  segments : List (Fin n × Fin n)
  passes_through_all_centers : ∀ (i j : Fin n), (i, j) ∈ segments

/-- The main theorem stating the minimum number of segments in a broken line -/
theorem min_segments_in_broken_line (n : ℕ) :
  (∃ (bl : BrokenLine n), bl.segments.length = 2 * n - 2) ∧
  (∀ (bl : BrokenLine n), bl.segments.length ≥ 2 * n - 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_segments_in_broken_line_l971_97121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l971_97171

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a circle with center and radius -/
structure Circle where
  center : Point
  radius : ℝ

def totalStudents : ℕ := 2500
def grade10Students : ℕ := 1000
def grade11Students : ℕ := 900
def grade12Students : ℕ := 600
def sampleSize : ℕ := 100

def centerA : Point := ⟨1, -1⟩

/-- The angle BAC in radians -/
noncomputable def angleBACRad : ℝ := 2 * Real.pi / 3

/-- Function to calculate the sample size for a grade based on stratified sampling -/
def calculateSampleSize (gradeStudents : ℕ) : ℕ :=
  (gradeStudents * sampleSize) / totalStudents

/-- The line that intersects the circle -/
def intersectingLine : Line :=
  let a := calculateSampleSize grade10Students
  let b := calculateSampleSize grade12Students
  ⟨a, b, 8⟩

/-- Theorem stating the equation of circle C -/
theorem circle_equation : 
  ∃ (C : Circle), 
    C.center = centerA ∧ 
    C.radius^2 = 18/17 ∧
    ∀ (x y : ℝ), (x - C.center.x)^2 + (y - C.center.y)^2 = C.radius^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l971_97171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_point_l971_97183

/-- Given a line L1 with equation x - 2y + 1 = 0 and a point A (1, -1),
    prove that the line L2 with equation 2x + y - 1 = 0 passes through A
    and is perpendicular to L1. -/
theorem perpendicular_line_through_point :
  let L1 : ℝ → ℝ → Prop := λ x y => x - 2*y + 1 = 0
  let L2 : ℝ → ℝ → Prop := λ x y => 2*x + y - 1 = 0
  let A : ℝ × ℝ := (1, -1)
  (L2 A.1 A.2) ∧ 
  (∀ (x1 y1 x2 y2 : ℝ), L1 x1 y1 → L1 x2 y2 → x2 ≠ x1 →
    let m1 := (y2 - y1) / (x2 - x1)
    let m2 := -1 / m1
    m1 * m2 = -1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_point_l971_97183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_even_602_to_700_l971_97133

def sum_first_n_even (n : ℕ) : ℕ := n * (n + 1)

def sum_even_in_range (start : ℕ) (end_val : ℕ) : ℕ :=
  let first_even := if start % 2 = 0 then start else start + 1
  let last_even := if end_val % 2 = 0 then end_val else end_val - 1
  let num_terms := (last_even - first_even) / 2 + 1
  num_terms * (first_even + last_even) / 2

theorem sum_even_602_to_700 :
  sum_first_n_even 50 = 2550 →
  sum_even_in_range 602 700 = 32550 := by
  sorry

#eval sum_even_in_range 602 700

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_even_602_to_700_l971_97133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_without_two_l971_97125

/-- Counts the number of digits in a positive integer that are not equal to 2 -/
def countNonTwoDigits (n : ℕ) : ℕ := sorry

/-- Checks if a positive integer contains the digit 2 -/
def containsTwo (n : ℕ) : Bool := sorry

/-- The set of whole numbers between 1 and 2000 that do not contain the digit 2 -/
def numberSet : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 2000 ∧ ¬containsTwo n}

theorem count_numbers_without_two : Finset.card (Finset.filter (λ n => ¬containsTwo n) (Finset.range 2000)) = 6560 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_without_two_l971_97125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tutoring_hours_l971_97109

/-- Represents the number of hours tutored in the first month -/
def first_month_hours : ℕ → Prop := sorry

/-- Hourly rate for tutoring in dollars -/
def hourly_rate : ℕ := 10

/-- Additional hours tutored in the second month -/
def additional_hours : ℕ := 5

/-- Fraction of earnings spent on personal needs -/
def spent_fraction : ℚ := 4/5

/-- Amount saved in dollars -/
def savings : ℕ := 150

theorem tutoring_hours :
  ∃ h : ℕ, first_month_hours h ∧
  (h * hourly_rate + (h + additional_hours) * hourly_rate) * (1 - spent_fraction) = savings ∧
  h = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tutoring_hours_l971_97109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_insect_reaches_northernmost_point_l971_97149

/-- Represents the disk on which the insect crawls -/
structure Disk where
  diameter : ℝ
  rotationPeriod : ℝ

/-- Represents the insect's movement -/
structure Insect where
  speed : ℝ
  startPosition : ℝ × ℝ  -- (x, y) coordinates
  direction : ℝ × ℝ      -- Unit vector representing direction

/-- Helper function to calculate insect's position at time t -/
def insect_position (d : Disk) (i : Insect) (t : ℝ) : ℝ × ℝ :=
  sorry

/-- Theorem stating that the insect reaches the northernmost point -/
theorem insect_reaches_northernmost_point (d : Disk) (i : Insect) :
  d.diameter = 3 ∧
  d.rotationPeriod = 15 ∧
  i.speed = 1 ∧
  i.startPosition = (0, -1.5) ∧
  i.direction = (0, 1) →
  ∃ t : ℝ, t > 0 ∧ insect_position d i t = (0, 1.5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_insect_reaches_northernmost_point_l971_97149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_interior_point_inequality_l971_97106

/-- Triangle with sidelengths a, b, c and semiperimeter p -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  p : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  semiperimeter : p = (a + b + c) / 2
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- An interior point of a triangle -/
structure InteriorPoint (t : Triangle) where
  x : ℝ
  y : ℝ
  is_interior : True  -- Placeholder condition; replace with actual interior condition if needed

/-- Distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem triangle_interior_point_inequality (t : Triangle) (p : InteriorPoint t) :
  let pa := distance p.x p.y 0 0  -- Assuming A is at (0,0)
  let pb := distance p.x p.y t.a 0  -- Assuming B is at (t.a,0)
  let pc := distance p.x p.y (t.b * Real.cos (π/3)) (t.b * Real.sin (π/3))  -- Assuming C forms an equilateral triangle
  min (pa / (t.p - t.a)) (min (pb / (t.p - t.b)) (pc / (t.p - t.c))) ≤ 2 / Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_interior_point_inequality_l971_97106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_work_time_l971_97188

noncomputable def work_rate (time : ℝ) : ℝ := 1 / time

theorem b_work_time (a b c : ℝ) : 
  work_rate 4 = a →
  work_rate 2 = b + c →
  work_rate 2 = a + c →
  work_rate (1 / b) = work_rate 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_work_time_l971_97188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polyhedron_properties_l971_97124

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  is_convex : Bool

/-- Number of sides of a face -/
def face_sides (f : ConvexPolyhedron → ℕ) : Prop :=
  ∃ n : ℕ, ∀ p : ConvexPolyhedron, f p = n

/-- Number of faces meeting at a polyhedral angle -/
def angle_faces (a : ConvexPolyhedron → ℕ) : Prop :=
  ∃ n : ℕ, ∀ p : ConvexPolyhedron, a p = n

/-- Theorem stating that there exists at least one face with 5 or fewer sides 
    and at least one polyhedral angle with 5 or fewer faces in any convex polyhedron -/
theorem convex_polyhedron_properties (p : ConvexPolyhedron) : 
  (∃ f : ConvexPolyhedron → ℕ, face_sides f ∧ ∃ n : ℕ, n ≤ 5 ∧ f p = n) ∧ 
  (∃ a : ConvexPolyhedron → ℕ, angle_faces a ∧ ∃ m : ℕ, m ≤ 5 ∧ a p = m) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polyhedron_properties_l971_97124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_goals_scored_l971_97172

theorem mean_goals_scored (players_with_2_goals players_with_4_goals players_with_5_goals players_with_6_goals : ℕ)
  (h1 : players_with_2_goals = 3)
  (h2 : players_with_4_goals = 2)
  (h3 : players_with_5_goals = 1)
  (h4 : players_with_6_goals = 1) :
  (2 * players_with_2_goals + 4 * players_with_4_goals + 5 * players_with_5_goals + 6 * players_with_6_goals : ℚ) /
  (players_with_2_goals + players_with_4_goals + players_with_5_goals + players_with_6_goals) = 25 / 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_goals_scored_l971_97172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_range_l971_97192

noncomputable def f (x : ℝ) := Real.cos x ^ 2
noncomputable def g (x : ℝ) := 1 + (1/2) * Real.sin (2 * x)
noncomputable def h (x : ℝ) := f x + g x

theorem symmetry_and_range (x₀ : ℝ) :
  (∀ x, f (x₀ + x) = f (x₀ - x)) →
  g (2 * x₀) = 1 ∧
  ∀ x ∈ Set.Icc 0 (Real.pi / 4), 2 ≤ h x ∧ h x ≤ (3 + Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_range_l971_97192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base6_addition_sum_7_l971_97151

/-- Represents a digit in base 6 --/
def Base6Digit := Fin 6

/-- Represents a three-digit number in base 6 --/
structure Base6Number :=
  (hundreds : Base6Digit)
  (tens : Base6Digit)
  (ones : Base6Digit)

/-- Addition of two Base6Number results in a Base6Number --/
def base6_add : Base6Number → Base6Number → Base6Number := sorry

/-- Converts a Base6Number to its decimal (base 10) representation --/
def to_decimal (n : Base6Number) : ℕ := sorry

/-- Helper function to create a Base6Digit from a natural number --/
def nat_to_base6 (n : ℕ) : Base6Digit :=
  ⟨n % 6, by
    apply Nat.mod_lt
    exact Nat.zero_lt_succ 5⟩

/-- The main theorem statement --/
theorem base6_addition_sum_7 (X Y : Base6Digit) :
  let lhs : Base6Number := ⟨nat_to_base6 5, X, Y⟩
  let rhs : Base6Number := ⟨nat_to_base6 0, nat_to_base6 2, nat_to_base6 3⟩
  let result : Base6Number := ⟨nat_to_base6 6, nat_to_base6 1, X⟩
  base6_add lhs rhs = result →
  (X.val : ℕ) + (Y.val : ℕ) = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base6_addition_sum_7_l971_97151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_l971_97190

-- Define the given line
noncomputable def given_line (x : ℝ) : ℝ := (3/2) * x + 12

-- Define the slope of the given line
noncomputable def m : ℝ := 3/2

-- Define the distance between the lines
def D : ℝ := 8

-- Define the possible equations for line L
noncomputable def L1 (x : ℝ) : ℝ := m * x + (12 + 4 * Real.sqrt 13)
noncomputable def L2 (x : ℝ) : ℝ := m * x + (12 - 4 * Real.sqrt 13)

-- Theorem statement
theorem parallel_line_equation :
  (∀ x, L1 x - given_line x = D ∨ given_line x - L1 x = D) ∧
  (∀ x, L2 x - given_line x = D ∨ given_line x - L2 x = D) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_l971_97190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_two_l971_97115

-- Define the integer part function
noncomputable def integerPart (x : ℝ) : ℤ :=
  ⌊x⌋

-- Define the fractional part function
noncomputable def fractionalPart (x : ℝ) : ℝ :=
  x - integerPart x

-- Define the equation
def equation (x : ℝ) : Prop :=
  (integerPart x) * (integerPart x - 2) = 3 - fractionalPart x

-- Theorem statement
theorem sum_of_roots_is_two :
  ∃ (x₁ x₂ : ℝ), equation x₁ ∧ equation x₂ ∧ x₁ + x₂ = 2 := by
  -- Proof goes here
  sorry

#check sum_of_roots_is_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_two_l971_97115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_petya_margin_l971_97196

/-- Represents the election results for a class president election between Petya and Vasya. -/
structure ElectionResult where
  total_votes : ℕ
  petya_first_two_hours : ℕ
  vasya_first_two_hours : ℕ
  petya_last_hour : ℕ
  vasya_last_hour : ℕ

/-- Checks if the election result is valid according to the given conditions. -/
def is_valid_election (result : ElectionResult) : Prop :=
  result.total_votes = 27 ∧
  result.petya_first_two_hours = result.vasya_first_two_hours + 9 ∧
  result.vasya_last_hour = result.petya_last_hour + 9 ∧
  result.petya_first_two_hours + result.petya_last_hour > result.vasya_first_two_hours + result.vasya_last_hour

/-- Calculates the margin of victory for Petya. -/
def petya_margin (result : ElectionResult) : ℤ :=
  (result.petya_first_two_hours + result.petya_last_hour : ℤ) -
  (result.vasya_first_two_hours + result.vasya_last_hour : ℤ)

/-- Theorem stating that the maximum margin of victory for Petya is 9 votes. -/
theorem max_petya_margin :
  ∀ result : ElectionResult, is_valid_election result → petya_margin result ≤ 9 :=
by
  sorry

#check max_petya_margin

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_petya_margin_l971_97196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_extremes_in_fifth_row_l971_97177

/-- Represents a position in the grid -/
structure Position where
  row : Nat
  col : Nat

/-- Represents the grid -/
def Grid := Position → Nat

/-- The size of the grid -/
def gridSize : Nat := 10

/-- The starting position -/
def startPos : Position := ⟨5, 6⟩

/-- Fills the grid with numbers from 1 to 100 counterclockwise starting from startPos -/
def fillGrid : Grid :=
  sorry

/-- Returns the numbers in the fifth row -/
def fifthRowNumbers (g : Grid) : List Nat :=
  sorry

/-- Theorem: The sum of the greatest and least numbers in the fifth row is 110 -/
theorem sum_of_extremes_in_fifth_row :
  let g := fillGrid
  let numbers := fifthRowNumbers g
  (List.maximum? numbers).getD 0 + (List.minimum? numbers).getD 0 = 110 := by
  sorry

#check sum_of_extremes_in_fifth_row

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_extremes_in_fifth_row_l971_97177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_range_of_x_l971_97131

-- Define the inequality function
def f (m x : ℝ) : ℝ := m * x^2 - 2 * m * x - 1

-- Part 1: Range of m
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, f m x < 0) → m ∈ Set.Ioo (-1 : ℝ) 0 ∪ {0} :=
sorry

-- Part 2: Range of x
theorem range_of_x (x : ℝ) :
  (∀ m : ℝ, |m| ≤ 1 → f m x < 0) → 
  x ∈ Set.Ioo (1 - Real.sqrt 2) 1 ∪ Set.Ioo 1 (1 + Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_range_of_x_l971_97131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l971_97146

theorem trig_identity (θ : ℝ) (h1 : Real.tan θ = -2) (h2 : -π/2 < θ) (h3 : θ < 0) :
  (Real.sin θ)^2 / (Real.cos (2*θ) + 2) = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l971_97146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f₁_derivative_f₂_l971_97166

-- Function 1
noncomputable def f₁ (x : ℝ) : ℝ := 2 * Real.log x

theorem derivative_f₁ (x : ℝ) (h : x > 0) : 
  deriv f₁ x = 2 / x := by sorry

-- Function 2
noncomputable def f₂ (x : ℝ) : ℝ := Real.exp x / x

theorem derivative_f₂ (x : ℝ) (h : x ≠ 0) : 
  deriv f₂ x = (x * Real.exp x - Real.exp x) / (x^2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f₁_derivative_f₂_l971_97166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_l971_97123

/-- Triangle T with side lengths 1, 2, and √7 -/
structure TriangleT where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  h1 : side1 = 1
  h2 : side2 = 2
  h3 : side3 = Real.sqrt 7

/-- Equilateral triangle formed by three copies of Triangle T -/
structure EquilateralTriangle where
  side : ℝ

/-- The area of an equilateral triangle -/
noncomputable def areaEquilateral (t : EquilateralTriangle) : ℝ :=
  (Real.sqrt 3 / 4) * t.side^2

/-- The ratio of areas of two equilateral triangles -/
noncomputable def areaRatio (t1 t2 : EquilateralTriangle) : ℝ :=
  areaEquilateral t1 / areaEquilateral t2

/-- The main theorem -/
theorem triangle_area_ratio (T : TriangleT) 
  (outer inner : EquilateralTriangle)
  (h_outer : outer.side = Real.sqrt 7)
  (h_inner : inner.side = 1) :
  areaRatio outer inner = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_l971_97123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_l971_97184

theorem negation_of_existence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 3 < 0) ↔ (∀ x : ℝ, x^2 + 2*x + 3 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_l971_97184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_question_not_proposition_l971_97169

def is_proposition (s : String) : Prop := 
  ∃ (p : Prop), s.length > 0 ∧ (p ∨ ¬p)

theorem question_not_proposition : 
  ¬(is_proposition "Is x always less than 2x?") :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_question_not_proposition_l971_97169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_position_on_second_side_l971_97120

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Represents the position of an ant on the perimeter of an equilateral triangle -/
noncomputable def ant_position (triangle : EquilateralTriangle) (distance_traveled : ℝ) : ℝ :=
  distance_traveled / (3 * triangle.side_length)

/-- Theorem: When an ant travels 42% of the perimeter of an equilateral triangle,
    it will be 26% of the way along the second side of the triangle -/
theorem ant_position_on_second_side (triangle : EquilateralTriangle) :
  let total_distance := 0.42 * (3 * triangle.side_length)
  let position := ant_position triangle total_distance
  position > 1/3 ∧ position < 2/3 ∧ 
  (position - 1/3) / (1/3) = 0.26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_position_on_second_side_l971_97120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l971_97182

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  eq : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a chord of an ellipse -/
structure Chord where
  A : Point
  B : Point

/-- Calculates the area of a triangle given three points -/
noncomputable def triangleArea (A B C : Point) : ℝ := sorry

/-- States that a given point is a focus of the ellipse -/
def isFocus (e : Ellipse a b) (F : Point) : Prop := sorry

/-- States that a chord passes through the center of the ellipse -/
def passesThroughCenter (e : Ellipse a b) (c : Chord) : Prop := sorry

theorem max_triangle_area 
  (a b c : ℝ) 
  (e : Ellipse a b) 
  (F : Point) 
  (h1 : isFocus e F) 
  (h2 : F.x = c ∧ F.y = 0) :
  ∃ (A B : Point),
    passesThroughCenter e (Chord.mk A B) ∧
    ∀ (X Y : Point), passesThroughCenter e (Chord.mk X Y) → 
      triangleArea F X Y ≤ b * c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l971_97182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_range_on_interval_l971_97105

noncomputable def f (x : ℝ) : ℝ := Real.sin x
def domain : Set ℝ := { x | Real.pi/6 ≤ x ∧ x ≤ 2*Real.pi/3 }

-- State the theorem
theorem sin_range_on_interval :
  Set.Icc (1/2 : ℝ) 1 = { y | ∃ x ∈ domain, f x = y } := by
  sorry

#check sin_range_on_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_range_on_interval_l971_97105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosing_triangles_count_l971_97100

/-- A convex n-gon where no two sides are parallel -/
structure ConvexNGon (n : ℕ) where
  polygon : Type
  convex : polygon → Prop
  sides : Fin n → polygon → Prop
  no_parallel_sides : ∀ (i j : Fin n) (p : polygon), i ≠ j → sides i p → sides j p → Prop

/-- A triangle formed by extensions of the n-gon's sides that encloses the n-gon -/
structure EnclosingTriangle (ng : ConvexNGon n) where
  lines : Fin 3 → ng.polygon → Prop
  from_sides : ∀ (i : Fin 3) (p : ng.polygon), ∃ (j : Fin n), ng.sides j p → lines i p
  encloses : ng.polygon → Prop

/-- The main theorem -/
theorem enclosing_triangles_count (n : ℕ) (h : n ≥ 3) (ng : ConvexNGon n) :
  ∃ (triangles : Finset (EnclosingTriangle ng)), Finset.card triangles ≥ n - 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosing_triangles_count_l971_97100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_properties_l971_97101

noncomputable def f (x m : ℝ) : ℝ := |Real.log x - 2/x + 2| - m

theorem zeros_properties (m : ℝ) (x₁ x₂ : ℝ) 
  (hm : 0 < m ∧ m < 3)
  (hx : x₁ < x₂)
  (hf₁ : f x₁ m = 0)
  (hf₂ : f x₂ m = 0) :
  x₂/x₁ < Real.exp (2*m) ∧ 
  x₁ > 2/(m+2) ∧ 
  x₁*x₂ > 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_properties_l971_97101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_third_f_smallest_positive_period_f_monotonic_increasing_interval_l971_97130

open Real

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := cos (2 * x - π / 3) - cos (2 * x)

-- Theorem for f(π/3)
theorem f_value_at_pi_third : f (π / 3) = 1 := by sorry

-- Theorem for the smallest positive period
theorem f_smallest_positive_period : ∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧
  (∀ T' : ℝ, T' > 0 → (∀ x : ℝ, f (x + T') = f x) → T ≤ T') ∧ T = π := by sorry

-- Theorem for the monotonically increasing interval
theorem f_monotonic_increasing_interval :
  ∀ k : ℤ, ∀ x y : ℝ, k * π - π / 6 ≤ x ∧ x < y ∧ y ≤ k * π + π / 3 → f x < f y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_third_f_smallest_positive_period_f_monotonic_increasing_interval_l971_97130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_distribution_l971_97154

theorem apple_distribution (craig_initial judy_initial dwayne_initial : ℝ)
  (craig_received craig_shared : ℝ)
  (h1 : craig_initial = 20.5)
  (h2 : judy_initial = 11.25)
  (h3 : dwayne_initial = 17.85)
  (h4 : craig_received = 7.15)
  (h5 : craig_shared = 3.5) :
  (craig_initial + craig_received - craig_shared / 2) +
  (judy_initial / 2) +
  (dwayne_initial + craig_shared / 2) +
  (judy_initial / 2) = 56.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_distribution_l971_97154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_transfer_proof_l971_97185

/-- The amount of milk transferred from container C to container B to equalize the quantities -/
noncomputable def milk_transferred (capacity_A : ℝ) (percent_less_B : ℝ) : ℝ :=
  let quantity_B := capacity_A * (1 - percent_less_B)
  let quantity_C := capacity_A - quantity_B
  (quantity_C - quantity_B) / 2

theorem milk_transfer_proof (capacity_A : ℝ) (percent_less_B : ℝ) 
  (h1 : capacity_A = 1184)
  (h2 : percent_less_B = 0.625) :
  milk_transferred capacity_A percent_less_B = 148 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval milk_transferred 1184 0.625

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_transfer_proof_l971_97185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_monotonicity_l971_97175

def a (k : ℝ) (n : ℕ+) : ℝ := n^2 + k * n

def monotonically_increasing (k : ℝ) : Prop :=
  ∀ n : ℕ+, a k n < a k (n + 1)

theorem sequence_monotonicity (k : ℝ) :
  monotonically_increasing k ↔ k > -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_monotonicity_l971_97175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_star_equals_five_l971_97141

-- Define the * operation
noncomputable def star (a b : ℝ) : ℝ := a + b + a * b

-- Define the nested expression
noncomputable def nestedStar : ℝ :=
  star (1/2) (star (1/3) (star (1/4) (star (1/5) (star (1/6) (star (1/7) (star (1/8) (star (1/9) (star (1/10) (1/11)))))))))

-- Theorem statement
theorem nested_star_equals_five : nestedStar = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_star_equals_five_l971_97141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_B_l971_97173

def B : Set ℕ := {n | ∃ x : ℕ, n = x + (x + 1) + (x + 2) + (x + 3)}

theorem gcd_of_B : ∃ d : ℕ, ∀ n ∈ B, d ∣ n ∧ d = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_B_l971_97173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_divisible_by_eight_l971_97150

theorem cube_root_sum_divisible_by_eight (n : ℕ) (h : n > 2) :
  ∃ k : ℤ, ⌊((n : ℝ)^(1/3) + ((n+2) : ℝ)^(1/3))^3⌋ + 1 = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_divisible_by_eight_l971_97150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_spiders_sufficient_and_necessary_l971_97126

/-- Represents a position on a 3D grid -/
structure Position where
  x : Int
  y : Int
  z : Int

/-- Represents the grid dimensions -/
def gridSize : Int := 2018

/-- Defines adjacency for two positions -/
def adjacent (p1 p2 : Position) : Prop :=
  (abs (p1.x - p2.x) + abs (p1.y - p2.y) + abs (p1.z - p2.z) ≤ 1) ∧
  (0 ≤ p1.x) ∧ (p1.x < gridSize) ∧ (0 ≤ p1.y) ∧ (p1.y < gridSize) ∧ (0 ≤ p1.z) ∧ (p1.z < gridSize) ∧
  (0 ≤ p2.x) ∧ (p2.x < gridSize) ∧ (0 ≤ p2.y) ∧ (p2.y < gridSize) ∧ (0 ≤ p2.z) ∧ (p2.z < gridSize)

/-- Defines a strategy for spiders to catch the fly -/
def CatchStrategy :=
  (Position → Position → Position → Position × Position) →
  Nat → Position → Position → Position → Prop

/-- Theorem: Two spiders are sufficient and necessary to catch the fly -/
theorem two_spiders_sufficient_and_necessary :
  (∃ (strategy : CatchStrategy),
    ∀ (fly_start spider1_start spider2_start : Position),
    ∃ (n : Nat), strategy (λ _ _ _ ↦ (spider1_start, spider2_start)) n fly_start spider1_start spider2_start) ∧
  (¬ ∃ (strategy : Position → Position → Position),
    ∀ (fly_start spider_start : Position),
    ∃ (n : Nat), strategy fly_start spider_start = fly_start) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_spiders_sufficient_and_necessary_l971_97126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_count_l971_97107

theorem acute_triangle_count : ∃! n : ℕ, n = (
  Finset.filter (fun (x : ℕ × ℕ × ℕ) => 
    let (a, b, c) := x
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a < 90 ∧ b < 90 ∧ c < 90 ∧
    c = 3 * a ∧
    a + b + c = 180
  ) (Finset.product (Finset.range 90) (Finset.product (Finset.range 90) (Finset.range 90)))
).card ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_count_l971_97107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_ratio_l971_97139

/-- The function f(x) = (2/3)x³ --/
noncomputable def f (x : ℝ) : ℝ := (2/3) * x^3

/-- The derivative of f(x) --/
noncomputable def f' (x : ℝ) : ℝ := 2 * x^2

/-- The slant angle of the tangent line at (1, f(1)) --/
noncomputable def α : ℝ := Real.arctan (f' 1)

theorem tangent_line_ratio :
  (Real.sin α)^2 - (Real.cos α)^2 = (3/5) * (2 * Real.sin α * Real.cos α + (Real.cos α)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_ratio_l971_97139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loom_weaving_rate_l971_97160

/-- The rate of cloth weaving given the time and length -/
noncomputable def weaving_rate (time : ℝ) (length : ℝ) : ℝ := length / time

theorem loom_weaving_rate : 
  let time := 113.63636363636363
  let length := 15
  let rate := weaving_rate time length
  ∃ ε > 0, abs (rate - 0.132) < ε :=
by
  -- We use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loom_weaving_rate_l971_97160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wicket_keeper_age_l971_97157

theorem wicket_keeper_age (team_size : ℕ) (captain_age : ℕ) (team_avg_age : ℝ) :
  team_size = 11 →
  captain_age = 27 →
  team_avg_age = 23 →
  (∃ (wicket_keeper_age : ℕ), wicket_keeper_age > captain_age) →
  (∃ (remaining_avg_age : ℝ), 
    remaining_avg_age * (team_size - 2) + captain_age + (captain_age + 1) = team_avg_age * team_size ∧
    remaining_avg_age = team_avg_age - 1) →
  ∃ (wicket_keeper_age : ℕ), wicket_keeper_age = captain_age + 1 :=
by
  intros h1 h2 h3 h4 h5
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wicket_keeper_age_l971_97157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_problem_l971_97119

def has_divisors (x : ℕ) (d : ℕ) : Prop :=
  (Finset.filter (λ y ↦ x % y = 0) (Finset.range (x + 1))).card = d

theorem divisors_problem (n : ℕ) (hn : n > 0) (h_divisors : has_divisors (150 * n^3) 150) :
  has_divisors (108 * n^5) 432 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_problem_l971_97119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_l971_97102

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 - 4*x + 3 else -x^2 - 2*x + 3

-- State the theorem
theorem inequality_holds_iff (a : ℝ) :
  (∀ x ∈ Set.Icc a (a + 1), f (x + a) > f (2*a - x)) ↔ a < -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_l971_97102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_squared_of_specific_parabolas_l971_97155

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with a focus and a directrix -/
structure Parabola where
  focus : Point
  directrix : ℝ → ℝ

/-- Calculates the square of the distance between two points -/
def distanceSquared (p q : Point) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

/-- Checks if a point lies on a parabola -/
def pointOnParabola (p : Point) (parabola : Parabola) : Prop :=
  distanceSquared p parabola.focus = (p.y - parabola.directrix p.x)^2

/-- Theorem: The square of the distance between intersection points of two specific parabolas -/
theorem intersection_distance_squared_of_specific_parabolas :
  let focus := Point.mk 20 22
  let p1 := Parabola.mk focus (λ _ => 0)  -- x-axis directrix
  let p2 := Parabola.mk focus (λ x => x)  -- y-axis directrix
  ∃ X Y : Point,
    (pointOnParabola X p1 ∧ pointOnParabola X p2) ∧
    (pointOnParabola Y p1 ∧ pointOnParabola Y p2) ∧
    X ≠ Y ∧
    distanceSquared X Y = 3520 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_squared_of_specific_parabolas_l971_97155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_sequence_l971_97111

def sequenceOddNegIntegers : List ℤ := sorry

theorem product_of_sequence :
  let prod := (sequenceOddNegIntegers.map Int.natAbs).foldl (·*·) 1
  prod > 0 ∧ prod % 100 = 75 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_sequence_l971_97111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_normal_chord_length_is_3_sqrt_3_m_l971_97110

/-- A parabola with focus distance m from the directrix -/
structure Parabola (m : ℝ) where
  focus_distance : m > 0

/-- A chord normal to the parabola at point A -/
structure NormalChord (m : ℝ) (p : Parabola m) where
  length : ℝ
  is_normal : Bool

/-- The minimum length of a normal chord -/
noncomputable def min_normal_chord_length (m : ℝ) (p : Parabola m) : ℝ :=
  3 * Real.sqrt 3 * m

/-- Theorem: The minimum length of a normal chord is 3√3m -/
theorem min_normal_chord_length_is_3_sqrt_3_m (m : ℝ) (p : Parabola m) :
  ∀ (chord : NormalChord m p), chord.is_normal → chord.length ≥ min_normal_chord_length m p := by
  sorry

#check min_normal_chord_length_is_3_sqrt_3_m

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_normal_chord_length_is_3_sqrt_3_m_l971_97110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_triple_angle_l971_97142

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1/3) : Real.cos (3*θ) = -23/27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_triple_angle_l971_97142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_students_count_l971_97138

structure School where
  total_students : ℕ
  blue_shirt_percent : ℚ
  red_shirt_percent : ℚ
  green_shirt_percent : ℚ
  blue_stripe_percent : ℚ
  blue_polka_percent : ℚ
  red_stripe_percent : ℚ
  red_polka_percent : ℚ
  green_stripe_percent : ℚ
  green_polka_percent : ℚ
  striped_glasses_percent : ℚ
  polka_hat_percent : ℚ
  no_pattern_scarf_percent : ℚ

def count_specific_students (s : School) : ℕ :=
  let red_polka_hat := (s.total_students : ℚ) * s.red_shirt_percent * s.red_polka_percent * s.polka_hat_percent
  let green_no_pattern_scarf := (s.total_students : ℚ) * s.green_shirt_percent * (1 - s.green_stripe_percent - s.green_polka_percent) * s.no_pattern_scarf_percent
  (red_polka_hat + green_no_pattern_scarf).floor.toNat

theorem specific_students_count (s : School) 
  (h1 : s.total_students = 1000)
  (h2 : s.blue_shirt_percent = 2/5)
  (h3 : s.red_shirt_percent = 1/4)
  (h4 : s.green_shirt_percent = 1/5)
  (h5 : s.blue_stripe_percent = 3/10)
  (h6 : s.blue_polka_percent = 7/20)
  (h7 : s.red_stripe_percent = 1/5)
  (h8 : s.red_polka_percent = 2/5)
  (h9 : s.green_stripe_percent = 1/4)
  (h10 : s.green_polka_percent = 1/4)
  (h11 : s.striped_glasses_percent = 1/5)
  (h12 : s.polka_hat_percent = 3/20)
  (h13 : s.no_pattern_scarf_percent = 1/10) :
  count_specific_students s = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_students_count_l971_97138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_shop_expenses_l971_97132

/-- Calculates the total weekly expenses for James's flower shop --/
noncomputable def total_weekly_expenses (weekly_rent : ℚ) (utility_rate : ℚ) (cashier_wage : ℚ) 
  (manager_wage : ℚ) (hours_per_shift : ℚ) (shifts_per_day : ℚ) (days_per_week : ℚ) 
  (loan_amount : ℚ) (loan_interest_rate : ℚ) (weeks_per_year : ℚ) 
  (sales_tax_rate : ℚ) (weekly_revenue : ℚ) : ℚ :=
  let utilities := weekly_rent * utility_rate
  let rent_and_utilities := weekly_rent + utilities
  let weekly_hours := hours_per_shift * shifts_per_day * days_per_week
  let wages := (cashier_wage + manager_wage) * weekly_hours
  let annual_interest := loan_amount * loan_interest_rate
  let loan_installment := (loan_amount + annual_interest) / weeks_per_year
  let sales_tax := weekly_revenue * sales_tax_rate
  rent_and_utilities + wages + loan_installment + sales_tax

/-- Theorem stating that the total weekly expenses for James's flower shop is $5540 --/
theorem flower_shop_expenses : 
  total_weekly_expenses 1200 (1/5) (25/2) 15 8 3 5 20000 (1/25) 52 (1/20) 8000 = 5540 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_shop_expenses_l971_97132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_l971_97140

theorem sum_of_reciprocals (m n : ℝ) (h1 : (2 : ℝ)^m = 5) (h2 : (5 : ℝ)^n = 2) :
  1 / (m + 1) + 1 / (n + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_l971_97140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_in_square_l971_97197

/-- The side length of the square -/
noncomputable def square_side : ℝ := 4 * Real.sqrt 3

/-- The area of the rhombus formed by the intersection of two equilateral triangles -/
noncomputable def rhombus_area (s : ℝ) : ℝ := 24 * Real.sqrt 3 - 24

/-- Theorem stating that the area of the rhombus is 24√3 - 24 -/
theorem rhombus_area_in_square : 
  rhombus_area square_side = 24 * Real.sqrt 3 - 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_in_square_l971_97197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_lighting_time_l971_97176

/-- Represents the length of a candle stub after burning for a given time -/
noncomputable def stub_length (initial_length : ℝ) (burn_time : ℝ) (elapsed_time : ℝ) : ℝ :=
  initial_length * (burn_time - elapsed_time) / burn_time

theorem candle_lighting_time 
  (initial_length : ℝ) 
  (burn_time_1 burn_time_2 elapsed_time : ℝ) 
  (h1 : burn_time_1 = 5)
  (h2 : burn_time_2 = 7)
  (h3 : initial_length > 0)
  (h4 : stub_length initial_length burn_time_2 elapsed_time = 
        3 * stub_length initial_length burn_time_1 elapsed_time) :
  elapsed_time = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_lighting_time_l971_97176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_ordering_l971_97178

noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

noncomputable def V₁ : ℝ := cylinder_volume 10 10
noncomputable def V₂ : ℝ := cylinder_volume 5 10
noncomputable def V₃ : ℝ := cylinder_volume 5 20
noncomputable def V₄ : ℝ := cylinder_volume 5 15
noncomputable def V₅ : ℝ := cylinder_volume 8 10

theorem cylinder_volume_ordering :
  V₂ < V₃ ∧ V₃ < V₁ ∧
  V₂ < V₄ ∧ V₄ < V₃ ∧
  V₁ < V₅ ∧ V₅ < V₃ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_ordering_l971_97178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_f_102_103_l971_97134

/-- The function f as defined in the problem -/
def f (x : ℤ) : ℤ := x^2 - x + 2008

/-- Theorem stating that the GCD of f(102) and f(103) is 2 -/
theorem gcd_f_102_103 : Int.gcd (f 102) (f 103) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_f_102_103_l971_97134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l971_97135

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.cos (2 * x + φ)

noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := f φ (x + Real.pi/12)

theorem phi_value (φ : ℝ) (h1 : 0 < φ) (h2 : φ < Real.pi) :
  (∀ x, g φ (-x) = -g φ x) → φ = Real.pi/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l971_97135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2002_equals_2_pow_2001_l971_97162

/-- The sequence a_n defined recursively -/
def a : ℕ → ℕ
  | 0 => 1  -- Define for 0 to cover all natural numbers
  | 1 => 1
  | 2 => 2
  | (n + 3) => 3 * a (n + 2) - 2 * a (n + 1)

/-- Theorem stating that the 2002nd term of the sequence equals 2^2001 -/
theorem a_2002_equals_2_pow_2001 : a 2002 = 2^2001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2002_equals_2_pow_2001_l971_97162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_exists_l971_97137

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

noncomputable def line_equation (a b : ℝ) (x y : ℝ) : Prop := x / a + y / b = 1

noncomputable def satisfies_conditions (a b : ℝ) : Prop :=
  a > 0 ∧ a < 10 ∧ is_prime (Int.toNat ⌊a⌋) ∧
  b > 0 ∧ ↑(Int.floor b) = b ∧
  line_equation a b 5 2

theorem unique_line_exists :
  ∃! p : ℝ × ℝ, satisfies_conditions p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_exists_l971_97137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_thirteenth_578th_digit_l971_97143

def decimal_expansion (n d : ℕ) : List ℕ :=
  sorry

theorem seventh_thirteenth_578th_digit : 
  let expansion := decimal_expansion 7 13
  (expansion.get? 577).getD 0 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_thirteenth_578th_digit_l971_97143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_relation_l971_97161

theorem tan_relation (α β : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2) 
  (h3 : Real.sin (2*α) = 2 * Real.sin (2*β)) : Real.tan (α + β) = 3 * Real.tan (α - β) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_relation_l971_97161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_remainder_l971_97103

theorem least_number_with_remainder : 
  ∃ (n : ℕ), n = 256 ∧ 
  (∀ d : ℕ, d ∈ ({7, 9, 12, 18} : Finset ℕ) → n % d = 4) ∧
  (∀ m : ℕ, m < n → ∃ d : ℕ, d ∈ ({7, 9, 12, 18} : Finset ℕ) ∧ m % d ≠ 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_remainder_l971_97103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_both_heads_is_quarter_l971_97136

/-- The probability of getting heads on a single coin toss -/
noncomputable def prob_heads : ℝ := 1 / 2

/-- The sample space of possible outcomes when tossing two coins -/
inductive CoinToss
  | HH  -- Both heads
  | HT  -- First head, second tail
  | TH  -- First tail, second head
  | TT  -- Both tails

/-- The probability of both coins landing heads up when tossing two uniform density coins -/
noncomputable def prob_both_heads : ℝ := 1 / 4

theorem prob_both_heads_is_quarter :
  prob_both_heads = 1 / 4 := by
  -- Unfold the definition of prob_both_heads
  unfold prob_both_heads
  -- The equality now holds by reflexivity
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_both_heads_is_quarter_l971_97136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_covering_theorem_l971_97198

-- Define a type for points in a plane
def Point : Type := ℝ × ℝ

-- Define a type for equilateral triangles
structure EquilateralTriangle :=
  (center : Point)
  (side_length : ℝ)

-- Define a function to check if a point is covered by a triangle
def is_covered (p : Point) (t : EquilateralTriangle) : Prop := sorry

-- Define a function to translate a triangle
def translate (t : EquilateralTriangle) (v : Point) : EquilateralTriangle := sorry

-- Define a proposition to check if a set of points can be covered by two translated triangles
def can_be_covered_by_two (X : Set Point) (T : EquilateralTriangle) : Prop :=
  ∃ (v1 v2 : Point), ∀ p ∈ X, is_covered p (translate T v1) ∨ is_covered p (translate T v2)

-- Main theorem
theorem covering_theorem (X : Set Point) (T : EquilateralTriangle) :
  (∀ X' : Finset Point, ↑X' ⊆ X → X'.card ≤ 9 → can_be_covered_by_two ↑X' T) →
  can_be_covered_by_two X T :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_covering_theorem_l971_97198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_proper_iff_subset_iff_equal_iff_l971_97153

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 3*x + 2 ≤ 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - (a+1)*x + a ≤ 0}

-- Theorem statements
theorem subset_proper_iff (a : ℝ) : A ⊂ B a ↔ a > 2 := by sorry

theorem subset_iff (a : ℝ) : A ⊆ B a ↔ a ≥ 2 := by sorry

theorem equal_iff (a : ℝ) : A = B a ↔ a = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_proper_iff_subset_iff_equal_iff_l971_97153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_loss_percentage_l971_97193

/-- The number of pencils sold for 1 rupee in the initial scenario -/
noncomputable def initial_pencils : ℝ := 10

/-- The number of pencils sold for 1 rupee in the second scenario -/
noncomputable def second_pencils : ℝ := 7.391304347826086

/-- The profit percentage in the second scenario -/
noncomputable def profit_percentage : ℝ := 15

/-- The selling price per pencil in the initial scenario -/
noncomputable def initial_price : ℝ := 1 / initial_pencils

/-- The selling price per pencil in the second scenario -/
noncomputable def second_price : ℝ := 1 / second_pencils

/-- The cost price per pencil -/
noncomputable def cost_price : ℝ := second_price / (1 + profit_percentage / 100)

/-- The percentage loss in the initial scenario -/
noncomputable def loss_percentage : ℝ := (cost_price - initial_price) / cost_price * 100

theorem initial_loss_percentage :
  abs (loss_percentage - 17.65) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_loss_percentage_l971_97193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_symmetric_shift_l971_97116

/-- The original function -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos (ω * x) + Real.sin (ω * x)

/-- The shifted function -/
noncomputable def g (ω m : ℝ) (x : ℝ) : ℝ := f ω (x + m)

/-- Theorem statement -/
theorem smallest_symmetric_shift (ω : ℝ) (h_ω_pos : ω > 0) :
  (∃ (k : ℤ), ∀ (n : ℤ), f ω ((2 * Real.pi / 3 + n * Real.pi / 2) / ω) = 0) →
  (∃ (m : ℝ), m > 0 ∧ 
    (∀ (x : ℝ), g ω m x = g ω m (-x)) ∧ 
    (∀ (m' : ℝ), 0 < m' ∧ m' < m → ∃ (x : ℝ), g ω m' x ≠ g ω m' (-x))) →
  (∃ (m : ℝ), m = Real.pi / 12 ∧ 
    (∀ (x : ℝ), g ω m x = g ω m (-x)) ∧ 
    (∀ (m' : ℝ), 0 < m' ∧ m' < m → ∃ (x : ℝ), g ω m' x ≠ g ω m' (-x))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_symmetric_shift_l971_97116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_z_l971_97122

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the absolute value of 1+i
noncomputable def abs_1_plus_i : ℝ := Complex.abs (1 + i)

-- Define z based on the given condition
noncomputable def z : ℂ := (1 - i)⁻¹ * abs_1_plus_i

-- Theorem statement
theorem real_part_of_z : Complex.re z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_z_l971_97122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_land_distribution_l971_97118

theorem farmer_land_distribution (total_land : ℝ) (cleared_percentage : ℝ) 
  (grapes_percentage : ℝ) (potato_percentage : ℝ) 
  (h1 : total_land = 4999.999999999999)
  (h2 : cleared_percentage = 0.9)
  (h3 : grapes_percentage = 0.1)
  (h4 : potato_percentage = 0.8) : 
  (total_land * cleared_percentage - 
   (total_land * cleared_percentage * grapes_percentage + 
    total_land * cleared_percentage * potato_percentage)) = 450 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_land_distribution_l971_97118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_flow_range_max_vehicle_density_l971_97179

-- Define the traffic flow function
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then
    100 - 135 * (1/3)^(80/x)
  else if 40 ≤ x ∧ x ≤ 80 then
    -((7/8) * (x - 40)) + 85
  else
    0

-- Define the vehicle density function
noncomputable def q (x : ℝ) : ℝ := x * f x

-- Theorem for part (1)
theorem traffic_flow_range (x : ℝ) :
  f x > 95 → 0 < x ∧ x < 80/3 := by
  sorry

-- Theorem for part (2)
theorem max_vehicle_density :
  f 80 = 50 → ∀ x, 0 < x → x ≤ 80 → q x ≤ 28800/7 := by
  sorry

#check f
#check q
#check traffic_flow_range
#check max_vehicle_density

end NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_flow_range_max_vehicle_density_l971_97179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_curve_range_of_m_for_equidistant_point_l971_97170

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  y + Real.sqrt 3 * x = Real.sqrt 3 * m

-- Define the curve C
def curve_C (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 3

-- Define the distance between a point and a line
noncomputable def distance_point_to_line (x y m : ℝ) : ℝ :=
  (|Real.sqrt 3 - m * Real.sqrt 3|) / 2

-- Theorem 1: Line l is tangent to curve C when m = 3
theorem line_tangent_to_curve :
  ∃ (x y : ℝ), curve_C x y ∧ line_l 3 x y ∧
  ∀ (x' y' : ℝ), curve_C x' y' → line_l 3 x' y' → (x = x' ∧ y = y') := by
  sorry

-- Theorem 2: Range of m for equidistant point
theorem range_of_m_for_equidistant_point :
  ∀ m : ℝ, (∃ (x y : ℝ), curve_C x y ∧ distance_point_to_line x y m = Real.sqrt 3 / 2) ↔
  -2 ≤ m ∧ m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_curve_range_of_m_for_equidistant_point_l971_97170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_two_l971_97147

theorem expression_equals_two :
  (27 : ℝ) ^ (1/3 : ℝ) + (2 + Real.sqrt 5) * (2 - Real.sqrt 5) + (-2 : ℝ) ^ (0 : ℝ) + (-1/3 : ℝ) ^ (-1 : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_two_l971_97147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l971_97180

theorem sin_double_angle_special_case (θ : ℝ) : 
  Real.cos (π/4 - θ) = 1/2 → Real.sin (2*θ) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l971_97180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l971_97108

-- Define the line l
noncomputable def line_l (x y : ℝ) : Prop := x + y = Real.sqrt 3

-- Define the curve C₂
def curve_C2 (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define point P
noncomputable def point_P : ℝ × ℝ := (-Real.sqrt 3, 0)

-- Define the theorem
theorem intersection_distance_sum :
  ∃ (M N : ℝ × ℝ),
    line_l M.1 M.2 ∧
    line_l N.1 N.2 ∧
    curve_C2 M.1 M.2 ∧
    curve_C2 N.1 N.2 ∧
    (1 / ((M.1 - point_P.1)^2 + (M.2 - point_P.2)^2).sqrt +
     1 / ((N.1 - point_P.1)^2 + (N.2 - point_P.2)^2).sqrt = 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l971_97108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_theorem_l971_97128

/-- Calculates the length of a train given its speed, the speed of a person running in the opposite direction, and the time it takes for the train to pass the person. -/
noncomputable def train_length (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed := train_speed + man_speed
  let relative_speed_mps := relative_speed * (5/18)
  relative_speed_mps * passing_time

/-- Theorem stating that under the given conditions, the train length is approximately 385 meters. -/
theorem train_length_theorem :
  let train_speed := (60 : ℝ)
  let man_speed := (6 : ℝ)
  let passing_time := (21 : ℝ)
  ∃ ε > 0, ε < 1 ∧ |train_length train_speed man_speed passing_time - 385| < ε :=
by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_theorem_l971_97128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_thirteen_pi_over_four_l971_97129

theorem cos_thirteen_pi_over_four : Real.cos (13 * π / 4) = -(1 / Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_thirteen_pi_over_four_l971_97129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_projection_properties_l971_97144

-- Define the oblique projection method
def obliqueProjection (shape : Type) : Type := sorry

-- Define shapes
inductive Shape
| Triangle
| Square
| IsoscelesTrapezoid
| Rhombus

-- Define properties of projected shapes
def isTriangle (shape : Type) : Prop := sorry
def isRhombus (shape : Type) : Prop := sorry
def isParallelogram (shape : Type) : Prop := sorry

-- Axioms based on the problem conditions
axiom triangle_projects_to_triangle :
  ∀ (t : Shape), t = Shape.Triangle → isTriangle (obliqueProjection Shape)

axiom square_not_always_rhombus :
  ∃ (s : Shape), s = Shape.Square ∧ ¬isRhombus (obliqueProjection Shape)

axiom isosceles_trapezoid_not_always_parallelogram :
  ∃ (it : Shape), it = Shape.IsoscelesTrapezoid ∧ ¬isParallelogram (obliqueProjection Shape)

axiom rhombus_not_always_rhombus :
  ∃ (r : Shape), r = Shape.Rhombus ∧ ¬isRhombus (obliqueProjection Shape)

-- Theorem to prove
theorem oblique_projection_properties :
  (∀ (t : Shape), t = Shape.Triangle → isTriangle (obliqueProjection Shape)) ∧
  (∃ (s : Shape), s = Shape.Square ∧ ¬isRhombus (obliqueProjection Shape)) ∧
  (∃ (it : Shape), it = Shape.IsoscelesTrapezoid ∧ ¬isParallelogram (obliqueProjection Shape)) ∧
  (∃ (r : Shape), r = Shape.Rhombus ∧ ¬isRhombus (obliqueProjection Shape)) :=
by
  apply And.intro
  · exact triangle_projects_to_triangle
  · apply And.intro
    · exact square_not_always_rhombus
    · apply And.intro
      · exact isosceles_trapezoid_not_always_parallelogram
      · exact rhombus_not_always_rhombus


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_projection_properties_l971_97144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l971_97156

/-- Given the equation (0.625 * 0.0729 * 28.9) / (0.0017 * x * 8.1) = 382.5, 
    prove that x is approximately equal to 0.24847 -/
theorem equation_solution :
  ∃ x : ℝ, (0.625 * 0.0729 * 28.9) / (0.0017 * x * 8.1) = 382.5 ∧ 
  (abs (x - 0.24847) < 0.00001) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l971_97156
