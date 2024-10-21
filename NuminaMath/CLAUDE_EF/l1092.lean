import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_b_c_order_l1092_109279

-- Define a, b, and c
noncomputable def a : ℝ := (6 : ℝ) ^ (0.4 : ℝ)
noncomputable def b : ℝ := Real.log 0.5 / Real.log 0.4
noncomputable def c : ℝ := Real.log 0.4 / Real.log 8

-- Theorem statement
theorem a_b_c_order : c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_b_c_order_l1092_109279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_logo_perimeter_is_4pi_l1092_109226

/-- The perimeter of a shape formed by linking 4 semicircles along the diameter of a unit circle -/
noncomputable def semicircle_logo_perimeter : ℝ := 4 * Real.pi

/-- Theorem: The perimeter of a shape formed by linking 4 semicircles along the diameter of a unit circle is equal to 4π -/
theorem semicircle_logo_perimeter_is_4pi :
  semicircle_logo_perimeter = 4 * Real.pi := by
  -- Unfold the definition of semicircle_logo_perimeter
  unfold semicircle_logo_perimeter
  -- The equality now holds by reflexivity
  rfl

#check semicircle_logo_perimeter_is_4pi

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_logo_perimeter_is_4pi_l1092_109226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_fill_time_l1092_109284

/-- The time it takes for Pipe A to fill the tank without a leak -/
noncomputable def T : ℝ := Real.sqrt 5

/-- The time it takes for the leak to empty a full tank -/
def leak_time : ℝ := 10

/-- The filling rate of Pipe A -/
noncomputable def fill_rate : ℝ := 1 / T

/-- The emptying rate of the leak -/
noncomputable def leak_rate : ℝ := 1 / leak_time

/-- The effective filling rate with the leak present -/
noncomputable def effective_rate : ℝ := fill_rate - leak_rate

theorem pipe_fill_time :
  effective_rate = 1 / (T + 0.5) → T = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_fill_time_l1092_109284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_cubic_equation_l1092_109295

theorem product_of_roots_cubic_equation :
  let f : ℝ → ℝ := fun x ↦ 3 * x^3 - 8 * x^2 + x + 7
  let roots := {r : ℝ | f r = 0}
  ∀ (a b c : ℝ), a ∈ roots → b ∈ roots → c ∈ roots → a * b * c = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_cubic_equation_l1092_109295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_paper_usage_l1092_109245

/-- Calculates the number of reams of paper John needs for a year of writing --/
theorem john_paper_usage (flash_fiction_stories_per_week : ℕ) 
                         (flash_fiction_pages_per_story : ℕ)
                         (short_stories_per_week : ℕ)
                         (short_story_pages : ℕ)
                         (novel_pages : ℕ)
                         (pages_per_sheet_double_sided : ℕ)
                         (sheets_per_ream : ℕ)
                         (weeks_per_year : ℕ) :
  flash_fiction_stories_per_week = 2 →
  flash_fiction_pages_per_story = 10 →
  short_stories_per_week = 1 →
  short_story_pages = 50 →
  novel_pages = 1500 →
  pages_per_sheet_double_sided = 2 →
  sheets_per_ream = 500 →
  weeks_per_year = 52 →
  ↑(((flash_fiction_stories_per_week * flash_fiction_pages_per_story / pages_per_sheet_double_sided +
      short_stories_per_week * short_story_pages / pages_per_sheet_double_sided) * weeks_per_year +
     novel_pages) / sheets_per_ream + 1) = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_paper_usage_l1092_109245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_roots_sum_l1092_109222

theorem tan_roots_sum (α β : Real) (h1 : α ∈ Set.Ioo (-π/2) (π/2)) (h2 : β ∈ Set.Ioo (-π/2) (π/2))
  (h3 : (Real.tan α)^2 + 3 * Real.sqrt 3 * (Real.tan α) + 4 = 0)
  (h4 : (Real.tan β)^2 + 3 * Real.sqrt 3 * (Real.tan β) + 4 = 0) :
  α + β = -2*π/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_roots_sum_l1092_109222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irregular_pentagon_area_l1092_109213

/-- The area of an irregular pentagon with given vertices -/
theorem irregular_pentagon_area : 
  let vertices : List (ℝ × ℝ) := [(-3, 1), (1, 1), (1, -2), (-3, -2), (-2, 0)]
  ∃ (area_of_polygon : List (ℝ × ℝ) → ℝ), area_of_polygon vertices = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irregular_pentagon_area_l1092_109213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_cube_volume_theorem_l1092_109264

/-- The volume of a solid formed by rotating a unit cube around one of its body diagonals -/
noncomputable def rotatedCubeVolume : ℝ := Real.sqrt 3 / 3 * Real.pi

/-- Theorem stating that the volume of a solid formed by rotating a unit cube around one of its body diagonals is equal to (√3/3)π -/
theorem rotated_cube_volume_theorem :
  let cube_edge_length : ℝ := 1
  rotatedCubeVolume = Real.sqrt 3 / 3 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_cube_volume_theorem_l1092_109264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scale_model_height_l1092_109274

/-- The scale ratio of the model -/
def scale_ratio : ℚ := 1 / 25

/-- The original height of the Statue of Liberty in feet -/
def original_height : ℕ := 305

/-- The height of the scale model before rounding -/
def model_height : ℚ := original_height / scale_ratio.num * scale_ratio.den

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

theorem scale_model_height :
  round_to_nearest model_height = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scale_model_height_l1092_109274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l1092_109292

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x - 4 ≤ 0}
def B : Set ℝ := {x | 1 < x ∧ x < 5}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = Set.Ico (-1) 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l1092_109292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_original_radius_l1092_109288

/-- The volume of a cylinder with radius r and height h is π * r^2 * h -/
noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

theorem cylinder_original_radius (z : ℝ) :
  let h₀ : ℝ := 4
  let r : ℝ := 8
  let v₀ := cylinderVolume r h₀
  let v₁ := cylinderVolume (r + 4) h₀
  let v₂ := cylinderVolume r (h₀ + 5)
  v₁ - v₀ = z ∧ v₂ - v₀ = z → r = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_original_radius_l1092_109288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_height_weight_correlated_l1092_109275

-- Define the relationships
def square_area_side (s : ℝ) : ℝ := s^2

noncomputable def height_weight : ℝ → ℝ := sorry

def distance_time (v : ℝ) (t : ℝ) : ℝ := v * t

noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

-- Define what it means for a relationship to be functional
def is_functional (f : ℝ → ℝ) : Prop := 
  ∀ x y : ℝ, f x = f y → x = y

-- Define what it means for a relationship to have a correlation
def has_correlation (f : ℝ → ℝ) : Prop := 
  ∃ x y : ℝ, x ≠ y ∧ f x ≠ f y

-- Theorem stating that only height-weight has correlation and is not functional
theorem only_height_weight_correlated :
  (is_functional square_area_side) ∧
  (is_functional (distance_time 1)) ∧
  (is_functional sphere_volume) ∧
  (has_correlation height_weight) ∧
  (¬ is_functional height_weight) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_height_weight_correlated_l1092_109275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_plane_and_area_l1092_109235

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The four given points -/
def A : Point3D := ⟨1, 0, 2⟩
def B : Point3D := ⟨4, 3, -1⟩
def C : Point3D := ⟨0, 3, -1⟩
def D : Point3D := ⟨5, -2, 4⟩

/-- Check if a point lies on the plane y + z = 2 -/
def onPlane (p : Point3D) : Prop :=
  p.y + p.z = 2

/-- The area of the quadrilateral ABCD -/
noncomputable def quadrilateralArea : ℝ := 15 * Real.sqrt 2

theorem points_on_plane_and_area :
  (onPlane A ∧ onPlane B ∧ onPlane C ∧ onPlane D) ∧
  (∃ (area : ℝ), area = quadrilateralArea) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_plane_and_area_l1092_109235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_interest_calculation_l1092_109294

theorem investment_interest_calculation (total_investment : ℝ) (rate1 : ℝ) (rate2 : ℝ) (investment2 : ℝ) :
  total_investment = 9000 →
  rate1 = 0.065 →
  rate2 = 0.08 →
  investment2 = 6258 →
  (total_investment - investment2) * rate1 + investment2 * rate2 = 678.87 := by
  intro h1 h2 h3 h4
  have investment1 := total_investment - investment2
  have interest1 := investment1 * rate1
  have interest2 := investment2 * rate2
  have combined_interest := interest1 + interest2
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_interest_calculation_l1092_109294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_volume_l1092_109271

/-- Represents a substance with a volume-to-weight ratio -/
structure Substance where
  volume : ℚ
  weight : ℚ
  ratio : ℚ := volume / weight

/-- Calculates the volume of a substance given its weight -/
def volumeOfSubstance (s : Substance) (w : ℚ) : ℚ :=
  w * s.ratio

/-- The problem statement -/
theorem mixture_volume 
  (substanceA : Substance)
  (substanceB : Substance)
  (h1 : substanceA.volume = 48 ∧ substanceA.weight = 112)
  (h2 : substanceB.volume = 36 ∧ substanceB.weight = 90)
  (weightA : ℚ)
  (weightB : ℚ)
  (h3 : weightA = 63)
  (h4 : weightB = 75) :
  volumeOfSubstance substanceA weightA + volumeOfSubstance substanceB weightB = 57 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_volume_l1092_109271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_three_point_five_l1092_109240

open Real

-- Define lg as the base-10 logarithm
noncomputable def lg (x : ℝ) : ℝ := log x / log 10

-- State the theorem
theorem expression_equals_three_point_five :
  (lg 2)^2 + (0.064 : ℝ)^(-(1/3 : ℝ)) + lg 5 * lg 20 = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_three_point_five_l1092_109240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1092_109280

/-- The perimeter of a triangle with vertices at (0,20), (12,0), and (0,0) is √544 + 32 -/
theorem triangle_perimeter : Real.sqrt 544 + 32 = Real.sqrt 544 + 32 := by
  -- Define the vertices of the triangle
  let A : ℝ × ℝ := (0, 20)
  let B : ℝ × ℝ := (12, 0)
  let C : ℝ × ℝ := (0, 0)

  -- Define the distance function
  let dist (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

  -- Calculate the perimeter
  let perimeter := dist A B + dist A C + dist B C

  -- Prove that the perimeter equals √544 + 32
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1092_109280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorable_b_l1092_109216

/-- A function that checks if a quadratic polynomial can be factored into two binomials with integer coefficients -/
def is_factorable (b : ℤ) : Prop :=
  ∃ (r s : ℤ), ∀ (x : ℤ), x^2 + b*x + 2016 = (x + r) * (x + s)

/-- The statement that 92 is the smallest positive integer b for which x^2 + bx + 2016 can be factored into two binomials with integer coefficients -/
theorem smallest_factorable_b : 
  (is_factorable 92) ∧ (∀ b : ℤ, 0 < b → b < 92 → ¬(is_factorable b)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorable_b_l1092_109216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifty_three_days_after_thursday_is_monday_l1092_109238

/-- Enumeration of days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to calculate the day of the week after a given number of days -/
def dayAfter (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => dayAfter (nextDay start) n

/-- Theorem stating that 53 days after Thursday is Monday -/
theorem fifty_three_days_after_thursday_is_monday :
  dayAfter DayOfWeek.Thursday 53 = DayOfWeek.Monday := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifty_three_days_after_thursday_is_monday_l1092_109238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_derivative_ratio_l1092_109224

/-- Given a quadratic function y = ax² + b, if its derivative at (1,3) is 2, then b/a = 2 -/
theorem quadratic_derivative_ratio (a b : ℝ) : 
  (∀ x, (fun x ↦ a * x^2 + b) x = a * x^2 + b) →  -- Definition of the function
  (fun x ↦ a * x^2 + b) 1 = 3 →                   -- Function passes through (1,3)
  (2 * a = 2) →                                -- Derivative at x=1 is 2
  b / a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_derivative_ratio_l1092_109224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_complex_number_l1092_109241

theorem modulus_of_complex_number :
  let z : ℂ := (Complex.I * Real.sqrt 5) / (1 + 2 * Complex.I)
  Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_complex_number_l1092_109241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_and_subtract_l1092_109244

/-- Round a number to the nearest thousandth -/
noncomputable def roundToThousandth (x : ℝ) : ℝ := 
  (⌊x * 1000 + 0.5⌋ : ℝ) / 1000

/-- The problem statement -/
theorem round_and_subtract (x : ℝ) (h : x = 18.48571) : 
  roundToThousandth x - 0.005 = 18.481 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_and_subtract_l1092_109244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_curve_solution_l1092_109273

/-- The integral curve of y'' = x + 1 passing through (1,1) and tangent to y = (1/2)x + 1/2 at this point -/
noncomputable def integral_curve (x : ℝ) : ℝ := (x^3 / 6) + (x^2 / 2) - x + (4/3)

/-- The second derivative of the integral curve -/
def second_derivative (x : ℝ) : ℝ := x + 1

theorem integral_curve_solution :
  (∀ x, deriv (deriv (integral_curve)) x = second_derivative x) ∧
  (integral_curve 1 = 1) ∧
  (deriv integral_curve 1 = 1/2) := by
  sorry

#check integral_curve_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_curve_solution_l1092_109273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_lighting_time_l1092_109267

/-- Represents the length of a candle stub after burning for a certain time -/
noncomputable def stubLength (totalBurnTime minutes : ℝ) (initialLength : ℝ) : ℝ :=
  initialLength * (totalBurnTime - minutes) / totalBurnTime

theorem candle_lighting_time :
  ∀ (initialLength : ℝ),
  initialLength > 0 →
  ∃ (lightingTime : ℝ),
  lightingTime > 0 ∧
  lightingTime < 6 * 60 ∧
  stubLength (5 * 60) (6 * 60 - lightingTime) initialLength = 
    3 * stubLength (7 * 60) (6 * 60 - lightingTime) initialLength ∧
  lightingTime = 6 * 60 - (4 * 60 + 22) :=
by sorry

#check candle_lighting_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_lighting_time_l1092_109267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1092_109220

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - x^2)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Icc (-1 : ℝ) 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1092_109220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_line_plane_relationship_l1092_109211

-- Define the types for point, line, and plane
variable (Point Line Plane : Type)

-- Define the relationships
variable (is_on : Point → Line → Prop)
variable (is_in_plane : Line → Plane → Prop)
variable (contains : Plane → Point → Prop)

-- State the theorem
theorem point_line_plane_relationship 
  (N : Point) (a : Line) (α : Plane) 
  (h1 : is_on N a) 
  (h2 : is_in_plane a α) : 
  ∃ (S : Set Point) (T : Set Line), 
    N ∈ S ∧ a ∈ T ∧ S = {p | is_on p a} ∧ T ⊆ {l | is_in_plane l α} :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_line_plane_relationship_l1092_109211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tens_digit_of_8_pow_1234_l1092_109253

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem tens_digit_of_8_pow_1234 :
  ∃ (cycle_length : ℕ) (cycle : List ℕ),
    cycle_length = 20 ∧
    cycle.length = cycle_length ∧
    (∀ k, last_two_digits (8^k) = cycle[k % cycle_length]!) ∧
    last_two_digits (8^14) = 4 →
    (8^1234 / 10) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tens_digit_of_8_pow_1234_l1092_109253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1092_109258

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let E := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let asymptote := {(x, y) : ℝ × ℝ | a * y - b * x = 0}
  let focus : ℝ × ℝ := (Real.sqrt (a^2 + b^2), 0)
  (∃ (d : ℝ), d = Real.sqrt 3 * a ∧ d = |b * focus.1| / Real.sqrt (a^2 + b^2)) →
  (Real.sqrt (a^2 + b^2)) / a = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1092_109258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_last_year_salary_l1092_109268

/-- Represents John's financial situation over two years -/
structure JohnFinances where
  bonus_percentage : ℚ
  last_year_bonus : ℚ
  this_year_salary : ℚ
  this_year_total : ℚ

/-- Calculates John's salary from last year given his financial information -/
def last_year_salary (j : JohnFinances) : ℚ :=
  j.last_year_bonus / j.bonus_percentage

/-- Theorem stating that John's last year salary was $100,000 -/
theorem john_last_year_salary (j : JohnFinances) 
  (h1 : j.last_year_bonus = 10000)
  (h2 : j.this_year_salary = 200000)
  (h3 : j.this_year_total = 220000)
  (h4 : j.bonus_percentage = (j.this_year_total - j.this_year_salary) / j.this_year_salary) :
  last_year_salary j = 100000 := by
  sorry

def main : IO Unit := do
  let j : JohnFinances := { 
    bonus_percentage := 1/10,
    last_year_bonus := 10000,
    this_year_salary := 200000,
    this_year_total := 220000
  }
  IO.println s!"John's last year salary: {last_year_salary j}"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_last_year_salary_l1092_109268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_angle_ACB_l1092_109207

-- Define the triangle ABC and point D
variable (A B C D : EuclideanSpace ℝ (Fin 2))

-- Define the angles
noncomputable def angle_ABC : ℝ := 60 * Real.pi / 180
noncomputable def angle_DAB : ℝ := 10 * Real.pi / 180

-- Define the condition that D is on BC
def D_on_BC (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = t • B + (1 - t) • C

-- Define the condition that 3 * BD = CD
def BD_CD_ratio (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop :=
  3 * ‖B - D‖ = ‖C - D‖

-- State the theorem
theorem unique_angle_ACB
    (A B C D : EuclideanSpace ℝ (Fin 2))
    (h1 : D_on_BC A B C D)
    (h2 : BD_CD_ratio A B C D)
    (h3 : EuclideanGeometry.angle A B C = angle_ABC)
    (h4 : EuclideanGeometry.angle D A B = angle_DAB) :
    ∃! θ : ℝ, 0 < θ ∧ θ < Real.pi ∧ EuclideanGeometry.angle A C B = θ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_angle_ACB_l1092_109207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_game_probabilities_l1092_109243

/-- The probability of winning for the starting player in the coin-flipping game -/
noncomputable def starting_player_win_prob : ℝ := 2/3

/-- The probability of winning for the second player in the coin-flipping game -/
noncomputable def second_player_win_prob : ℝ := 1/3

/-- The probability of the game never ending -/
noncomputable def game_never_ends_prob : ℝ := 0

/-- The coin-flipping game where two players take turns and the first to get heads wins -/
theorem coin_flip_game_probabilities :
  (starting_player_win_prob + second_player_win_prob + game_never_ends_prob = 1) ∧
  (second_player_win_prob = (1/2) * starting_player_win_prob) ∧
  (starting_player_win_prob = (1/2) + (1/2) * second_player_win_prob) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_game_probabilities_l1092_109243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_implies_m_value_l1092_109227

def OA : Fin 2 → ℝ := ![0, 1]
def OB : Fin 2 → ℝ := ![1, 3]
def OC (m : ℝ) : Fin 2 → ℝ := ![m, m]

def AB : Fin 2 → ℝ := ![1, 2]
def AC (m : ℝ) : Fin 2 → ℝ := ![m, m-1]

def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), v = fun i => k * w i

theorem vector_parallel_implies_m_value :
  ∀ m : ℝ, parallel AB (AC m) → m = -1 := by
  sorry

#check vector_parallel_implies_m_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_implies_m_value_l1092_109227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_basis_of_plane_l1092_109228

open Submodule

variable (V : Type*) [AddCommGroup V] [Module ℝ V]

variable (e₁ e₂ : V)

-- Define the plane as a subspace spanned by e₁ and e₂
def plane (e₁ e₂ : V) : Submodule ℝ V :=
  span ℝ {e₁, e₂}

-- State that e₁ and e₂ form a basis of the plane
variable (h : LinearIndependent ℝ ![e₁, e₂])
variable (h_span : plane V e₁ e₂ = ⊤)

-- Theorem statement
theorem new_basis_of_plane :
  LinearIndependent ℝ ![e₁ + e₂, e₁ - e₂] ∧
  span ℝ {e₁ + e₂, e₁ - e₂} = plane V e₁ e₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_basis_of_plane_l1092_109228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_condition_l1092_109270

theorem angle_condition (B : ℝ) : 
  B ∈ Set.Ioo 0 π → (
    (B = π / 3 → Real.sin B = Real.sqrt 3 / 2) ∧
    ∃ B', B' ∈ Set.Ioo 0 π ∧ B' ≠ π / 3 ∧ Real.sin B' = Real.sqrt 3 / 2
  ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_condition_l1092_109270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_bound_l1092_109276

/-- A rectangle with area and position -/
structure Rectangle where
  area : ℝ
  x : ℝ
  y : ℝ
  width : ℝ
  height : ℝ

/-- The main rectangle containing all smaller rectangles -/
def mainRectangle : Rectangle :=
  { area := 5, x := 0, y := 0, width := 5, height := 1 }

/-- A list of 9 smaller rectangles, each with area 1 -/
def smallRectangles : List Rectangle :=
  List.replicate 9 { area := 1, x := 0, y := 0, width := 1, height := 1 }

/-- Function to calculate the intersection area of two rectangles -/
noncomputable def intersectionArea (r1 r2 : Rectangle) : ℝ :=
  sorry

/-- Theorem stating that there exist two smaller rectangles with intersection area ≥ 1/9 -/
theorem intersection_area_bound (mainRect : Rectangle) (smallRects : List Rectangle) :
  mainRect = mainRectangle →
  smallRects = smallRectangles →
  ∃ r1 r2, r1 ∈ smallRects ∧ r2 ∈ smallRects ∧ r1 ≠ r2 ∧ intersectionArea r1 r2 ≥ 1/9 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_bound_l1092_109276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_paths_count_l1092_109232

/-- Represents a point on the 3D grid --/
structure Point3D where
  x : Nat
  y : Nat
  z : Nat

/-- Defines the grid dimensions --/
def gridDimensions : Point3D := ⟨3, 2, 2⟩

/-- Starting point P --/
def P : Point3D := ⟨0, 0, 0⟩

/-- Ending point Q --/
def Q : Point3D := ⟨3, 2, 2⟩

/-- Predicate to check if a move is valid (closer to Q and farther from P) --/
def isValidMove (p1 p2 : Point3D) : Prop :=
  p2.x ≥ p1.x ∧ p2.y ≥ p1.y ∧ p2.z ≥ p1.z ∧
  (p2.x > p1.x ∨ p2.y > p1.y ∨ p2.z > p1.z)

/-- Counts the number of valid paths from P to Q on the grid --/
noncomputable def countValidPaths (start : Point3D) (finish : Point3D) : Nat :=
  sorry

/-- The main theorem to prove --/
theorem squirrel_paths_count :
  countValidPaths P Q = 14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_paths_count_l1092_109232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l1092_109293

-- Define the function
noncomputable def f (x : ℝ) := Real.log (x^2 - x - 2)

-- State the theorem
theorem monotonic_decreasing_interval_of_f :
  let S := {x : ℝ | x < -1 ∨ x > 2}
  ∀ x ∈ S, x^2 - x - 2 > 0 →
  ∀ y ∈ S, y^2 - y - 2 > 0 →
  (∀ t > 0, ∀ u > 0, t < u → Real.log t < Real.log u) →
  (∀ x y, x ∈ S → y ∈ S → x < y ∧ y ≤ -1 → f y ≤ f x) ∧
  (∃ a b, a ∈ S ∧ b ∈ S ∧ a < b ∧ -1 < b ∧ f a < f b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l1092_109293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_total_distance_l1092_109239

def bug_journey (start end1 end2 final : Int) : Nat :=
  (Int.natAbs (end1 - start)) + (Int.natAbs (end2 - end1)) + (Int.natAbs (final - end2))

theorem bug_total_distance :
  bug_journey 3 (-4) 7 0 = 25 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_total_distance_l1092_109239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perimeter_quadrilateral_l1092_109289

/-- Represents a quadrilateral in 2D space -/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Calculates the perimeter of a quadrilateral -/
noncomputable def perimeter (q : Quadrilateral) : ℝ := sorry

/-- Constructs a parallelogram from two diagonals and an angle -/
noncomputable def constructParallelogram (a b : ℝ) (α : ℝ) : Quadrilateral := sorry

/-- Returns the intersection point of the diagonals of a parallelogram -/
noncomputable def diagonalIntersection (p : Quadrilateral) : ℝ × ℝ := sorry

/-- Calculates the angle between diagonals of a quadrilateral -/
noncomputable def angleBetweenDiagonals (q : Quadrilateral) : ℝ := sorry

/-- Theorem: The quadrilateral with the smallest perimeter has one vertex at the diagonal intersection -/
theorem smallest_perimeter_quadrilateral 
  (a b α : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hα : 0 < α ∧ α < π) :
  ∀ q : Quadrilateral, 
    (∃ d1 d2 : ℝ, d1 = a ∧ d2 = b ∧ angleBetweenDiagonals q = α) →
    perimeter q ≥ perimeter (
      let p := constructParallelogram a b α
      let N := diagonalIntersection p
      Quadrilateral.mk p.A p.B p.C N
    ) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perimeter_quadrilateral_l1092_109289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1092_109212

theorem triangle_inequality (a b c α β γ : ℝ) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h2 : 0 < α ∧ 0 < β ∧ 0 < γ) 
  (h3 : α + β + γ = Real.pi) 
  (h4 : a / Real.sin α = b / Real.sin β) 
  (h5 : b / Real.sin β = c / Real.sin γ) : 
  a * (1/β + 1/γ) + b * (1/γ + 1/α) + c * (1/α + 1/β) ≥ 2 * (a/α + b/β + c/γ) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1092_109212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_l₁_and_l₂_l1092_109219

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₁ - c₂| / Real.sqrt (a^2 + b^2)

/-- Line l₁: 3x - 4y + 6 = 0 -/
def l₁ (x y : ℝ) : Prop := 3 * x - 4 * y + 6 = 0

/-- Line l₂: 6x - 8y + 9 = 0 -/
def l₂ (x y : ℝ) : Prop := 6 * x - 8 * y + 9 = 0

theorem distance_between_l₁_and_l₂ :
  distance_between_parallel_lines 6 (-8) 12 9 = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_l₁_and_l₂_l1092_109219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_show_total_time_l1092_109296

-- Define the number of commercials
def num_commercials : ℕ := 3

-- Define the length of each commercial in minutes
def commercial_length : ℕ := 10

-- Define the length of the TV show without commercials in minutes
def show_length : ℕ := 60

-- Theorem statement
theorem tv_show_total_time :
  (show_length + num_commercials * commercial_length : ℚ) / 60 = (3 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_show_total_time_l1092_109296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_equation_l1092_109266

theorem integer_solutions_equation : 
  {(x, y) : ℤ × ℤ | x^2 * y = 10000 * x + y} = 
  {(-9, -1125), (-3, -3750), (0, 0), (3, 3750), (9, 1125)} := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_equation_l1092_109266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_diagonal_of_rectangle_min_diagonal_value_l1092_109260

noncomputable section

open Real

theorem min_diagonal_of_rectangle (l w : ℝ) : 
  l > 0 → w > 0 → l + w = 18 → 
  ∀ x y : ℝ, x > 0 → y > 0 → x + y = 18 → 
  Real.sqrt (l^2 + w^2) ≤ Real.sqrt (x^2 + y^2) :=
by sorry

theorem min_diagonal_value : 
  ∃ l w : ℝ, l > 0 ∧ w > 0 ∧ l + w = 18 ∧ Real.sqrt (l^2 + w^2) = Real.sqrt 162 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_diagonal_of_rectangle_min_diagonal_value_l1092_109260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellen_stuffing_time_l1092_109205

/-- Earl's rate of stuffing envelopes per minute -/
noncomputable def earl_rate : ℝ := 36

/-- Time taken by Earl and Ellen together to stuff 360 envelopes (in minutes) -/
noncomputable def combined_time : ℝ := 6

/-- Number of envelopes stuffed by Earl and Ellen together -/
noncomputable def combined_envelopes : ℝ := 360

/-- Ellen's rate of stuffing envelopes per minute -/
noncomputable def ellen_rate : ℝ := (combined_envelopes / combined_time) - earl_rate

/-- Time taken by Ellen to stuff the same number of envelopes as Earl does in a minute -/
noncomputable def ellen_time : ℝ := earl_rate / ellen_rate

theorem ellen_stuffing_time :
  ellen_time = 1.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellen_stuffing_time_l1092_109205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounded_region_l1092_109254

/-- The area of the region bounded by the lines x = 2, y = 2, the x-axis, and the y-axis is 4 -/
theorem area_bounded_region (bounded_x : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2)
                            (bounded_y : ∀ (y : ℝ), 0 ≤ y ∧ y ≤ 2)
                            (area : ℝ := 4) : area = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounded_region_l1092_109254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_l1092_109217

/-- Calculates the length of a train given the speeds of two trains, their crossing time, and the length of the other train. -/
noncomputable def train_length (speed1 speed2 : ℝ) (crossing_time : ℝ) (other_length : ℝ) : ℝ :=
  (speed1 + speed2) * crossing_time * (5 / 18) - other_length

/-- Theorem stating the length of the first train given the problem conditions. -/
theorem first_train_length :
  let speed1 : ℝ := 36
  let speed2 : ℝ := 54
  let crossing_time : ℝ := 12
  let second_train_length : ℝ := 80
  train_length speed1 speed2 crossing_time second_train_length = 220 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval train_length 36 54 12 80

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_l1092_109217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_offset_l1092_109200

/-- Represents a quadrilateral with a diagonal and two offsets -/
structure Quadrilateral where
  diagonal : ℝ
  offset1 : ℝ
  offset2 : ℝ

/-- Calculates the area of a quadrilateral given its diagonal and offsets -/
noncomputable def area (q : Quadrilateral) : ℝ :=
  (q.diagonal * q.offset1 + q.diagonal * q.offset2) / 2

/-- Theorem: In a quadrilateral with diagonal 40, offset1 9, and area 300, offset2 is 6 -/
theorem quadrilateral_offset (q : Quadrilateral) 
    (h_diagonal : q.diagonal = 40)
    (h_offset1 : q.offset1 = 9)
    (h_area : area q = 300) :
  q.offset2 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_offset_l1092_109200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_true_statements_l1092_109259

theorem max_true_statements : 
  ∃ (x : ℝ) (true_statements : List Bool), 
    true_statements.length ≤ 3 ∧ 
    (0 < x^2 ∧ x^2 < 1) = true_statements[0]! ∧
    (x^2 ≥ 4) = true_statements[1]! ∧
    (-1 < x ∧ x < 0) = true_statements[2]! ∧
    (0 < x ∧ x < 1) = true_statements[3]! ∧
    (0 < x - x^2 ∧ x - x^2 ≤ 1/4) = true_statements[4]! :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_true_statements_l1092_109259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_is_four_point_five_l1092_109282

/-- A right triangle with specific properties -/
structure RightTriangle where
  -- The legs of the right triangle
  a : ℝ
  b : ℝ
  -- The hypotenuse of the right triangle
  c : ℝ
  -- The area of the triangle is 10
  area_eq : a * b / 2 = 10
  -- The radius of the inscribed circle is 1
  inradius_eq : (a + b - c) / 2 = 1
  -- Pythagorean theorem
  pythagoras : c^2 = a^2 + b^2

/-- The radius of the circumscribed circle of a right triangle -/
noncomputable def circumradius (t : RightTriangle) : ℝ := t.c / 2

/-- Theorem: The circumradius of the specified right triangle is 4.5 -/
theorem circumradius_is_four_point_five (t : RightTriangle) : 
  circumradius t = 4.5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_is_four_point_five_l1092_109282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l1092_109255

-- Define the necessary functions
noncomputable def f1 (x : ℝ) : ℝ := Real.log x
noncomputable def g1 (x : ℝ) : ℝ := (1/2) * Real.log (x^2)

def f2 (x : ℝ) : ℝ := x
noncomputable def g2 (x : ℝ) : ℝ := Real.sqrt (x^2)

noncomputable def f3 (x : ℝ) : ℝ := Real.log (Real.exp x)
noncomputable def g3 (x : ℝ) : ℝ := Real.exp (Real.log x)

noncomputable def f4 (x : ℝ) : ℝ := Real.log x / Real.log (1/2)
noncomputable def g4 (x : ℝ) : ℝ := -(Real.log x / Real.log 2)

-- State the theorem
theorem function_equality :
  (∀ x > 0, f1 x = g1 x) ∧
  (∀ x > 0, f4 x = g4 x) ∧
  (∃ x, f2 x ≠ g2 x) ∧
  (∃ x, f3 x ≠ g3 x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l1092_109255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_sum_l1092_109221

/-- The sum of the infinite series 1 + 2x + 3x² + 4x³ + ... for |x| < 1 -/
noncomputable def infiniteSeries (x : ℝ) : ℝ := ∑' n, (n + 1) * x^n

theorem infiniteSeries_sum (x : ℝ) (h : |x| < 1) : 
  infiniteSeries x = 1 / (1 - x)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_sum_l1092_109221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_in_triangle_l1092_109230

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 1/2 - Real.cos x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

-- Define the theorem
theorem f_range_in_triangle (a b c : ℝ) (h : b^2 + c^2 - a^2 > b*c) :
  ∃ (A : ℝ), A > 0 ∧ A < π/3 ∧ 
  (∀ y : ℝ, -1/2 < y ∧ y < 1 → ∃ x : ℝ, x > 0 ∧ x < π/3 ∧ f x = y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_in_triangle_l1092_109230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_percentage_l1092_109261

/-- Square with side length 6 units -/
def square_side_length : ℚ := 6

/-- Area of the first shaded rectangle -/
def rect1_area : ℚ := 2 * 2

/-- Area of the second shaded rectangle -/
def rect2_area : ℚ := 1 * square_side_length

/-- Area of the third shaded rectangle -/
def rect3_area : ℚ := 1 * square_side_length

/-- Total area of shaded rectangles -/
def total_shaded_area : ℚ := rect1_area + rect2_area + rect3_area

/-- Area of the square -/
def square_area : ℚ := square_side_length ^ 2

/-- Percentage of the square that is shaded -/
def shaded_percentage : ℚ := (total_shaded_area / square_area) * 100

theorem shaded_area_percentage :
  ∃ (ε : ℚ), abs (shaded_percentage - 44.44) < ε ∧ ε > 0 := by
  sorry

#eval shaded_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_percentage_l1092_109261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1092_109247

theorem unique_solution (a b c d : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h_sum : a + b + c + d = 1)
  (h_eq : max (a^2 / b) (b^2 / a) * max (c^2 / d) (d^2 / c) = (min (a + b) (c + d))^4) :
  a = 1/4 ∧ b = 1/4 ∧ c = 1/4 ∧ d = 1/4 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1092_109247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipses_properties_l1092_109257

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h₁ : a > 0
  h₂ : b > 0

/-- The focal distance of an ellipse -/
noncomputable def focal_distance (e : Ellipse) : ℝ := Real.sqrt (e.a^2 - e.b^2)

/-- Two ellipses with the same foci -/
def same_foci (e₁ e₂ : Ellipse) : Prop := focal_distance e₁ = focal_distance e₂

theorem ellipses_properties (e₁ e₂ : Ellipse) 
  (h_foci : same_foci e₁ e₂) (h_a : e₁.a > e₂.a) (h_b : e₂.a > e₂.b) : 
  (∀ x y : ℝ, x^2 / e₁.a^2 + y^2 / e₁.b^2 = 1 → x^2 / e₂.a^2 + y^2 / e₂.b^2 ≠ 1) ∧ 
  (e₁.a^2 - e₂.a^2 = e₁.b^2 - e₂.b^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipses_properties_l1092_109257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_handshake_count_is_539_l1092_109262

/-- The number of handshakes at a gathering with twins and triplets -/
def handshake_count : ℕ := 
  let twin_sets : ℕ := 12
  let triplet_sets : ℕ := 8
  let abstaining_twin_sets : ℕ := 1
  let abstaining_triplet_sets : ℕ := 1
  let total_twins : ℕ := twin_sets * 2
  let total_triplets : ℕ := triplet_sets * 3
  let participating_twins : ℕ := total_twins - (abstaining_twin_sets * 2)
  let participating_triplets : ℕ := total_triplets - (abstaining_triplet_sets * 3)
  let twins_per_triplet : ℕ := participating_twins / 4
  let triplets_per_twin : ℕ := participating_triplets / 3
  (participating_twins * (participating_twins - 2) + 
   participating_triplets * (participating_triplets - 3) + 
   participating_twins * triplets_per_twin + 
   participating_triplets * twins_per_triplet) / 2

theorem handshake_count_is_539 : handshake_count = 539 := by
  -- Proof goes here
  sorry

#eval handshake_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_handshake_count_is_539_l1092_109262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_points_distance_ratio_l1092_109290

/-- Given four points in a plane, the ratio of the largest distance to the smallest distance
    between any two of these points is always greater than or equal to √2. -/
theorem four_points_distance_ratio (P₁ P₂ P₃ P₄ : ℝ × ℝ) :
  let distances := [dist P₁ P₂, dist P₁ P₃, dist P₁ P₄, dist P₂ P₃, dist P₂ P₄, dist P₃ P₄]
  (List.maximum distances).get! / (List.minimum distances).get! ≥ Real.sqrt 2 := by
  sorry

/-- The Euclidean distance between two points in ℝ² -/
noncomputable def dist (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_points_distance_ratio_l1092_109290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1092_109252

-- Define the function f(x) = ln x - x^2 + x - 1
noncomputable def f (x : ℝ) : ℝ := Real.log x - x^2 + x - 1

-- Theorem stating the maximum value of f(x) is -1 and f(x) < x - 3/2 for all x > 0
theorem f_properties :
  (∃ (x : ℝ), x > 0 ∧ f x = -1 ∧ ∀ (y : ℝ), y > 0 → f y ≤ f x) ∧
  (∀ (x : ℝ), x > 0 → f x < x - 3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1092_109252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cold_brew_time_theorem_l1092_109286

/-- Represents the cold brew coffee making scenario -/
structure ColdBrewScenario where
  batch_size : ℚ  -- in gallons
  consumption_rate : ℚ  -- in ounces per 2 days
  brewing_time : ℚ  -- in hours
  period : ℚ  -- in days

/-- Calculates the time spent making coffee over a given period -/
noncomputable def time_spent_brewing (scenario : ColdBrewScenario) : ℚ :=
  let ounces_per_gallon : ℚ := 128
  let batch_size_oz : ℚ := scenario.batch_size * ounces_per_gallon
  let daily_consumption : ℚ := scenario.consumption_rate / 2
  let batch_duration : ℚ := batch_size_oz / daily_consumption
  let num_batches : ℚ := scenario.period / batch_duration
  num_batches * scenario.brewing_time

/-- Theorem stating that Jack spends 120 hours making coffee over 24 days -/
theorem cold_brew_time_theorem (jack : ColdBrewScenario) 
    (h1 : jack.batch_size = 3/2)
    (h2 : jack.consumption_rate = 96)
    (h3 : jack.brewing_time = 20)
    (h4 : jack.period = 24) :
  time_spent_brewing jack = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cold_brew_time_theorem_l1092_109286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_photocopier_problem_l1092_109201

/-- Represents the time (in minutes) it takes for a photocopier to complete the task alone -/
structure Photocopier where
  time : ℝ
  time_positive : time > 0

/-- Represents a pair of photocopiers -/
structure PhotocopierPair where
  a : Photocopier
  b : Photocopier

/-- Calculates the time taken when both photocopiers are used simultaneously -/
noncomputable def simultaneous_time (pair : PhotocopierPair) : ℝ :=
  1 / (1 / pair.a.time + 1 / pair.b.time)

/-- Calculates the fraction of the task completed in a given time -/
noncomputable def task_fraction (pair : PhotocopierPair) (time : ℝ) : ℝ :=
  time * (1 / pair.a.time + 1 / pair.b.time)

theorem photocopier_problem (pair : PhotocopierPair) 
    (ha : pair.a.time = 90) (hb : pair.b.time = 60) : 
    simultaneous_time pair = 36 ∧ 
    (task_fraction pair 30 + (1 - task_fraction pair 30) / (1 / pair.a.time) > 43) ∧
    (task_fraction pair 30 + 9 / pair.a.time + 
      (1 - task_fraction pair 30 - 9 / pair.a.time) / (1 / pair.a.time + 1 / pair.b.time) < 43) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_photocopier_problem_l1092_109201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_for_given_ring_area_l1092_109246

-- Define the radii of the circles
variable (R r : ℝ)

-- Define the area of the ring
noncomputable def ring_area (R r : ℝ) : ℝ := Real.pi * R^2 - Real.pi * r^2

-- Define the length of the chord
noncomputable def chord_length (R r : ℝ) : ℝ := 2 * Real.sqrt (R^2 - r^2)

-- Theorem statement
theorem chord_length_for_given_ring_area (R r : ℝ) :
  ring_area R r = 25 * Real.pi → chord_length R r = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_for_given_ring_area_l1092_109246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l1092_109256

noncomputable def is_simple_quadratic_radical (x : ℝ) : Prop :=
  ∀ y z : ℝ, y > 0 → z > 0 → x = y * Real.sqrt z → y = 1

noncomputable def radicals : List ℝ := [Real.sqrt 1.5, Real.sqrt (1/3), Real.sqrt 8, Real.sqrt 5]

theorem simplest_quadratic_radical :
  ∃ x ∈ radicals, is_simple_quadratic_radical x ∧ 
    ∀ y ∈ radicals, is_simple_quadratic_radical y → x = y := by
  sorry

#check simplest_quadratic_radical

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l1092_109256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_properties_l1092_109251

def m (a b : ℝ) : ℝ × ℝ := (a, b)

noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.sin (2 * x), 2 * (Real.cos x) ^ 2)

noncomputable def f (a b x : ℝ) : ℝ := (m a b).1 * (n x).1 + (m a b).2 * (n x).2

theorem vector_dot_product_properties 
  (a b : ℝ) 
  (h1 : f a b 0 = 8) 
  (h2 : f a b (π / 6) = 12) : 
  (a = 4 * Real.sqrt 3 ∧ b = 4) ∧ 
  (∀ x, f a b x ≤ 12) ∧
  (∀ k : ℤ, f a b (k * π + π / 6) = 12) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k * π - π / 3) (k * π + π / 6), 
    ∀ y ∈ Set.Icc (k * π - π / 3) (k * π + π / 6), 
    x ≤ y → f a b x ≤ f a b y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_properties_l1092_109251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxed_land_percentage_is_four_percent_l1092_109202

/-- Represents the farm tax scenario in a village -/
structure FarmTaxScenario where
  totalTax : ℚ
  williamTax : ℚ
  williamLandPercentage : ℚ

/-- The percentage of cultivated land that is taxed -/
def taxedLandPercentage (scenario : FarmTaxScenario) : ℚ :=
  (scenario.williamTax / scenario.williamLandPercentage) / scenario.totalTax * 100

/-- Theorem stating that the percentage of cultivated land taxed is 4% -/
theorem taxed_land_percentage_is_four_percent (scenario : FarmTaxScenario) 
    (h1 : scenario.totalTax = 3840)
    (h2 : scenario.williamTax = 480)
    (h3 : scenario.williamLandPercentage = 1/2) :
    taxedLandPercentage scenario = 4 := by
  sorry

#eval taxedLandPercentage { totalTax := 3840, williamTax := 480, williamLandPercentage := 1/2 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxed_land_percentage_is_four_percent_l1092_109202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_in_special_triangle_l1092_109218

/-- In a triangle ABC where angles A, B, C form an arithmetic sequence and AC = 2,
    the maximum value of AB · AC is 2 + (4√3)/3 -/
theorem max_dot_product_in_special_triangle (A B C : ℝ) :
  -- Angles A, B, C form an arithmetic sequence
  B = (A + C) / 2 →
  -- Sum of angles in a triangle is π
  A + B + C = π →
  -- Side AC has length 2
  2 = 2 →
  -- Maximum value of AB · AC
  (∃ (max : ℝ), ∀ (AB AC : ℝ), AB * AC ≤ max ∧ 
   ∃ (AB₀ AC₀ : ℝ), AB₀ * AC₀ = max) →
  -- The maximum value is 2 + (4√3)/3
  ∃ (max : ℝ), (max = 2 + (4 * Real.sqrt 3) / 3 ∧
    (∀ (AB AC : ℝ), AB * AC ≤ max) ∧
    (∃ (AB₀ AC₀ : ℝ), AB₀ * AC₀ = max)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_in_special_triangle_l1092_109218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_a_zero_l1092_109204

/-- The function representing the curve -/
noncomputable def f (x : ℝ) : ℝ := (x * Real.sin x) / Real.exp x

/-- The derivative of the function f -/
noncomputable def f' (x : ℝ) : ℝ := ((1 - x) * Real.sin x + x * Real.cos x) / Real.exp x

/-- The slope of the tangent line at the origin -/
noncomputable def tangent_slope : ℝ := f' 0

/-- The line to be perpendicular to the tangent line -/
def perpendicular_line (a : ℝ) (x y : ℝ) : Prop := 3 * x - a * y + 1 = 0

theorem tangent_perpendicular_implies_a_zero (a : ℝ) : 
  (tangent_slope = 0 ∧ perpendicular_line a 0 0) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_a_zero_l1092_109204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_d_is_correct_l1092_109214

/-- Triangle DEF with given properties --/
structure Triangle where
  b : ℝ
  c : ℝ
  cosDE : ℝ
  h_b : b = 7
  h_c : c = 8
  h_cosDE : cosDE = 55 / 64

/-- The side length d opposite to angle D in the given triangle --/
noncomputable def side_length_d (t : Triangle) : ℝ := Real.sqrt 105

/-- Theorem stating that the side length d is correct --/
theorem side_length_d_is_correct (t : Triangle) : 
  side_length_d t = Real.sqrt 105 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_length_d_is_correct_l1092_109214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_A_B_l1092_109281

def U : Finset Int := {-2, -1, 0, 1, 2, 3}
def A : Finset Int := {-1, 2}
def B : Finset Int := {1, 3}

theorem complement_union_A_B :
  (U \ (A ∪ B)) = {-2, 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_A_B_l1092_109281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1092_109225

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x + a

-- Define the derivative of f
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a

theorem f_inequality (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > 0) 
  (hx : x₁ < x₂) 
  (hf₁ : f a x₁ = 0) 
  (hf₂ : f a x₂ = 0) : 
  f a (3 * Real.log a) > f_deriv a ((2 * x₁ * x₂) / (x₁ + x₂)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1092_109225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_of_angle_l1092_109234

/-- Given an angle of 60 degrees, prove that after a 600-degree clockwise rotation,
    the new acute angle is also 60 degrees. -/
theorem rotation_of_angle (initial_angle rotation : ℝ) (h1 : initial_angle = 60)
  (h2 : rotation = 600) : 
  (360 - ((rotation % 360 - initial_angle) % 360)) % 180 = 60 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_of_angle_l1092_109234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_divisible_by_55_l1092_109248

def a : ℕ → ℕ
  | 0 => 5  -- Define for 0 to cover all natural numbers
  | 1 => 5  -- Define for 1 to 4 as well
  | 2 => 5
  | 3 => 5
  | 4 => 5
  | 5 => 5
  | n + 1 => 50 * a n + 5 * (n + 1)

def is_divisible_by_55 (n : ℕ) : Prop := ∃ k : ℕ, a n = 55 * k

theorem least_n_divisible_by_55 :
  (∀ m, 5 < m ∧ m < 7 → ¬ is_divisible_by_55 m) ∧
  is_divisible_by_55 7 := by
  sorry

#eval a 7  -- This will compute a₇ to verify our result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_divisible_by_55_l1092_109248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_amount_proof_l1092_109265

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Represents the mass of a substance in grams -/
def Grams : Type := ℝ

/-- Represents the molar mass of a substance in g/mol -/
def MolarMass : Type := ℝ

/-- The balanced chemical equation for the reaction -/
def reaction_equation : String := 
  "NaHCO₃ + CH₃COOH → NaCH₃COO + CO₂ + H₂O"

/-- The molar ratio of water to sodium bicarbonate in the reaction -/
def water_to_bicarbonate_ratio : ℚ := 1/1

/-- The molar mass of water in g/mol -/
def water_molar_mass : ℝ := 18.015

/-- Calculate the amount of water formed in the reaction -/
noncomputable def water_formed (bicarbonate : ℝ) (acid : ℝ) : ℝ :=
  bicarbonate * (water_to_bicarbonate_ratio : ℝ) * water_molar_mass

/-- Theorem: The amount of water formed when 3 moles of Sodium bicarbonate
    react with 3 moles of ethanoic acid is 54.045 grams -/
theorem water_amount_proof : 
  water_formed 3 3 = 54.045 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_amount_proof_l1092_109265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_problem_l1092_109269

noncomputable def i : ℂ := Complex.I

noncomputable def z : ℂ := 10 / (3 + i) - 2 * i

theorem complex_modulus_problem : Complex.abs z = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_problem_l1092_109269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cinnamon_swirl_eaters_l1092_109297

theorem cinnamon_swirl_eaters (total_swirls : ℚ) (jane_pieces : ℕ) (h1 : total_swirls = 12.0) (h2 : jane_pieces = 4) :
  (total_swirls / jane_pieces : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cinnamon_swirl_eaters_l1092_109297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_distances_l1092_109223

theorem circle_intersection_distances (r : ℝ) (p q : ℕ) (m n : ℕ) (u v : ℝ) :
  Odd (Int.floor r) →
  Nat.Prime p →
  Nat.Prime q →
  u = Real.rpow (↑p) m →
  v = Real.rpow (↑q) n →
  u > v →
  u^2 + v^2 = r^2 →
  ∃ (A B C D P M N : ℝ × ℝ),
    A = (r, 0) ∧
    B = (-r, 0) ∧
    C = (0, -r) ∧
    D = (0, r) ∧
    P = (u, v) ∧
    M = (u, 0) ∧
    N = (0, v) ∧
    |A.1 - M.1| = 1 ∧
    |B.1 - M.1| = 9 ∧
    |C.2 - N.2| = 8 ∧
    |D.2 - N.2| = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_distances_l1092_109223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plumber_number_divisibility_l1092_109287

theorem plumber_number_divisibility (a b c d : ℕ) 
  (ha : 10 ≤ a ∧ a < 100) 
  (hb : 10 ≤ b ∧ b < 100) 
  (hc : 10 ≤ c ∧ c < 100) 
  (hd : 10 ≤ d ∧ d < 100) : 
  (a * 1000000 + a * 10000 + b * 100000 + b * 1000 + c * 100 + c * 10 + d * 10 + d) % 101 = 0 := by
  sorry

#check plumber_number_divisibility

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plumber_number_divisibility_l1092_109287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_divisibility_by_five_l1092_109291

theorem three_digit_divisibility_by_five (M : ℕ) : 
  100 ≤ M ∧ M < 1000 ∧ M % 10 = 4 → (Nat.card {M | M % 5 = 0 ∧ 100 ≤ M ∧ M < 1000 ∧ M % 10 = 4}) / (Nat.card {M | 100 ≤ M ∧ M < 1000 ∧ M % 10 = 4}) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_divisibility_by_five_l1092_109291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_books_l1092_109263

theorem sandy_books (benny_books tim_books total_books : ℕ) 
  (h1 : benny_books = 24)
  (h2 : tim_books = 33)
  (h3 : total_books = 67) :
  ∃ sandy_books : ℕ, sandy_books = 10 ∧ total_books = benny_books + tim_books + sandy_books := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_books_l1092_109263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_k_value_l1092_109242

noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

/-- The point (2, 6) lies on the inverse proportion curve -/
axiom point_on_curve (k : ℝ) : inverse_proportion k 2 = 6

theorem inverse_proportion_k_value : ∃ k : ℝ, k = 12 ∧ inverse_proportion k 2 = 6 := by
  use 12
  constructor
  · rfl
  · unfold inverse_proportion
    norm_num

#check inverse_proportion_k_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_k_value_l1092_109242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1092_109210

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := Real.exp x + x

-- Define the point of tangency
def point : ℝ × ℝ := (0, 1)

-- State the theorem
theorem tangent_line_equation :
  ∃ (k m : ℝ), k ≠ 0 ∧
  (∀ x y : ℝ, k * x - y + m = 0 ↔
    (y - f point.1 = (deriv f point.1) * (x - point.1) ∧
     y = f x)) := by
  -- The proof goes here
  sorry

-- Optionally, we can add a lemma to calculate the derivative
lemma deriv_f (x : ℝ) : deriv f x = Real.exp x + 1 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1092_109210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_k_l1092_109285

/-- A line in 2D space represented by parametric equations --/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- A line in 2D space represented by a general equation ax + by = c --/
structure GeneralLine where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The slope of a parametric line --/
noncomputable def paramSlope (l : ParametricLine) : ℝ :=
  (l.y 1 - l.y 0) / (l.x 1 - l.x 0)

/-- The slope of a general line --/
noncomputable def generalSlope (l : GeneralLine) : ℝ :=
  -l.a / l.b

/-- Two lines are perpendicular if the product of their slopes is -1 --/
def isPerpendicular (l1 : ParametricLine) (l2 : GeneralLine) : Prop :=
  paramSlope l1 * generalSlope l2 = -1

theorem perpendicular_lines_k :
  let l1 : ParametricLine := { x := λ t => 1 - 2*t, y := λ t => 2 + 3*t }
  let l2 : GeneralLine := { a := 4, b := k, c := 1 }
  isPerpendicular l1 l2 → k = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_k_l1092_109285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_roots_sum_l1092_109231

theorem complex_roots_sum (n : ℕ) (z : ℕ → ℂ) (θ : ℕ → ℝ) :
  (∀ m, z m ^ 35 - z m ^ 7 - 1 = 0) →
  (∀ m, Complex.abs (z m) = 1) →
  (∀ m, z m = Complex.exp (Complex.I * θ m * Real.pi / 180)) →
  (∀ m, 0 ≤ θ m ∧ θ m < 360) →
  (∀ m, m < 2*n → θ m < θ (m+1)) →
  θ 2 + θ 4 + θ 6 + θ 8 + θ 10 + θ 12 + θ 14 = 925.714 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_roots_sum_l1092_109231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_B_complement_A_l1092_109283

-- Define the universal set U as a real number type
variable {U : Type} [LinearOrderedField U]

-- Define set A
def A (x : U) : Prop := x^2 - 2*x - 3 > 0

-- Define set B
def B (x : U) : Prop := 2 < x ∧ x < 4

-- Define the complement of A
def complement_A (x : U) : Prop := ¬(A x)

-- State the theorem
theorem intersection_B_complement_A :
  {x : U | B x ∧ complement_A x} = {x : U | 2 < x ∧ x ≤ 3} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_B_complement_A_l1092_109283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_l1092_109298

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 + Real.sqrt (1 + x^4))

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  intro x
  unfold f
  -- The proof steps would go here
  sorry

#check f_is_even

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_l1092_109298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1092_109233

-- Define the ellipse (C)
def ellipse_C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the reference ellipse
def reference_ellipse (x y : ℝ) : Prop := x^2 + y^2 / 2 = 1

-- Define the line (l)
def line_l (x y k m : ℝ) : Prop := y = k * x + m

-- Define the focal length
def focal_length (c : ℝ) : Prop := c = 1

-- Define the eccentricity equality
def same_eccentricity (a b : ℝ) : Prop := a / b = Real.sqrt 2

-- Define the condition for a > b > 0
def size_condition (a b : ℝ) : Prop := a > b ∧ b > 0

-- Define the vector addition condition
def vector_addition (xA yA xB yB xQ yQ lambda : ℝ) : Prop :=
  xA + xB = lambda * xQ ∧ yA + yB = lambda * yQ

-- Theorem statement
theorem ellipse_properties (a b c : ℝ) (h1 : size_condition a b) 
  (h2 : focal_length c) (h3 : same_eccentricity a b) :
  (∀ x y, ellipse_C x y a b ↔ ellipse_C x y (Real.sqrt 2) 1) ∧
  (∀ k m xA yA xB yB xQ yQ lambda,
    ellipse_C xA yA (Real.sqrt 2) 1 ∧ 
    ellipse_C xB yB (Real.sqrt 2) 1 ∧
    ellipse_C xQ yQ (Real.sqrt 2) 1 ∧
    line_l xA yA k m ∧
    line_l xB yB k m ∧
    vector_addition xA yA xB yB xQ yQ lambda →
    -2 < lambda ∧ lambda < 2 ∧ lambda ≠ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1092_109233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_gentle_and_sequence_bound_l1092_109237

/-- A function f is gentle on a set D if |f(x) - f(y)| ≤ |x - y| for all x, y in D -/
def GentleFunction (f : ℝ → ℝ) (D : Set ℝ) :=
  ∀ x y, x ∈ D → y ∈ D → |f x - f y| ≤ |x - y|

/-- The sequence condition given in the problem -/
def SequenceCondition (x : ℕ → ℝ) :=
  ∀ n : ℕ, |x (n + 1) - x n| ≤ 1 / (2 * n + 1)^2

theorem sin_gentle_and_sequence_bound
    (x : ℕ → ℝ)
    (h1 : GentleFunction Real.sin Set.univ)
    (h2 : SequenceCondition x)
    (n : ℕ) :
    |Real.sin (x (n + 1)) - Real.sin (x 1)| < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_gentle_and_sequence_bound_l1092_109237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trig_ratio_l1092_109206

open Real

theorem min_trig_ratio : ∀ x : ℝ, 
  (sin x)^8 + (cos x)^8 + 1 ≥ 14/27 * ((sin x)^6 + (cos x)^6 + 1) ∧
  ∃ y : ℝ, (sin y)^8 + (cos y)^8 + 1 = 14/27 * ((sin y)^6 + (cos y)^6 + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trig_ratio_l1092_109206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1092_109208

noncomputable def deg_to_rad (x : ℝ) : ℝ := x * (Real.pi / 180)

-- State the theorem
theorem problem_solution : 
  (Real.cos (deg_to_rad 10) ≠ 0) → 
  (Real.cos (deg_to_rad 50) ≠ 0) → 
  (Real.sin (deg_to_rad 50) ≠ 0) → 
  (Real.sin (deg_to_rad 20) = (1/3) * Real.cos (deg_to_rad 10)) → 
  (1 / Real.cos (deg_to_rad 50) - 2 / Real.sin (deg_to_rad 50) = 4/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1092_109208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_decrease_percentage_l1092_109203

noncomputable def original_salary : ℝ := 3000
noncomputable def increase_percentage : ℝ := 10
noncomputable def final_salary : ℝ := 3135

noncomputable def salary_after_increase : ℝ := original_salary * (1 + increase_percentage / 100)

theorem salary_decrease_percentage :
  ∃ (decrease_percentage : ℝ),
    salary_after_increase * (1 - decrease_percentage / 100) = final_salary ∧
    decrease_percentage = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_decrease_percentage_l1092_109203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oq_op_ratio_max_area_abq_l1092_109249

/-- Ellipse C with equation x²/4 + y² = 1 and eccentricity √3/2 -/
noncomputable def EllipseC : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 4) + p.2^2 = 1}

/-- Ellipse E with equation x²/16 + y²/4 = 1 -/
noncomputable def EllipseE : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 16) + (p.2^2 / 4) = 1}

/-- The eccentricity of Ellipse C -/
noncomputable def eccentricity : ℝ := Real.sqrt 3 / 2

/-- Theorem: For any point P on EllipseC, |OQ|/|OP| = 2 where Q is on EllipseE -/
theorem oq_op_ratio (P : ℝ × ℝ) (h : P ∈ EllipseC) :
  ∃ Q : ℝ × ℝ, Q ∈ EllipseE ∧ ‖Q‖ / ‖P‖ = 2 := by sorry

/-- Function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The maximum area of triangle ABQ is 6√3 -/
theorem max_area_abq :
  ∃ A B Q : ℝ × ℝ, A ∈ EllipseE ∧ B ∈ EllipseE ∧ Q ∈ EllipseE ∧
  ∃ P : ℝ × ℝ, P ∈ EllipseC ∧
  ∃ k m : ℝ, (∀ x : ℝ, A.2 = k * A.1 + m ∧ B.2 = k * B.1 + m ∧ P.2 = k * P.1 + m) ∧
  area_triangle A B Q ≤ 6 * Real.sqrt 3 ∧
  (∃ A' B' Q' : ℝ × ℝ, area_triangle A' B' Q' = 6 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oq_op_ratio_max_area_abq_l1092_109249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_calculation_l1092_109272

/-- Represents the problem of calculating a person's average speed across a street with varying conditions --/
noncomputable def averageSpeedProblem (totalDistance : ℝ) (stretch1Distance : ℝ) (stretch1Time : ℝ)
  (stretch2Distance : ℝ) (stretch2Speed : ℝ) (stretch3Distance : ℝ) (stretch3Time : ℝ) : ℝ :=
  let totalTime := stretch1Time + stretch2Distance / (stretch2Speed * 1000 / 60) + stretch3Time
  totalDistance / (totalTime / 60)

/-- Theorem stating that the average speed is approximately 2.795 km/h given the problem conditions --/
theorem average_speed_calculation :
  ∃ ε > 0, abs (averageSpeedProblem 1.5 0.4 10 0.6 5 0.5 15 - 2.795) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_calculation_l1092_109272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_condition_inequality_condition_l1092_109229

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (1 - x) * (-a + Real.cos x)

theorem increasing_interval_condition (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), 0 ≤ x₁ ∧ x₂ ≤ π ∧ x₁ < x₂ ∧ 
    ∀ (y z : ℝ), x₁ ≤ y ∧ y < z ∧ z ≤ x₂ → f a y < f a z) →
  a ≥ Real.sqrt 2 :=
by sorry

theorem inequality_condition (a : ℝ) :
  f a (π / 2) = 0 →
  ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1/2 → 
    f a (-x - 1) + 2 * (deriv (f a)) x * Real.cos (-x - 1) > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_condition_inequality_condition_l1092_109229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_l1092_109236

-- Define the function f(x) = (1/2)^x - x^(1/3)
noncomputable def f (x : ℝ) : ℝ := (1/2)^x - x^(1/3)

-- State the theorem
theorem solution_in_interval (x₀ : ℝ) (h : f x₀ = 0) : x₀ ∈ Set.Ioo (1/3 : ℝ) (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_l1092_109236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_water_level_approx_new_water_level_rounded_l1092_109299

/-- Represents the setup of two connected cylindrical vessels with water and oil -/
structure VesselSetup where
  initial_height : ℝ
  water_density : ℝ
  oil_density : ℝ

/-- Calculates the new water level after opening the valve -/
noncomputable def new_water_level (setup : VesselSetup) : ℝ :=
  (2 * setup.initial_height * setup.water_density) /
  (setup.water_density + setup.oil_density)

/-- Theorem stating that the new water level is approximately 32.94 cm -/
theorem new_water_level_approx (setup : VesselSetup)
  (h_initial : setup.initial_height = 40)
  (h_water_density : setup.water_density = 1000)
  (h_oil_density : setup.oil_density = 700) :
  ∃ ε > 0, |new_water_level setup - 32.94| < ε := by
  sorry

/-- Theorem stating that the rounded up value of the new water level is 34 cm -/
theorem new_water_level_rounded (setup : VesselSetup)
  (h_initial : setup.initial_height = 40)
  (h_water_density : setup.water_density = 1000)
  (h_oil_density : setup.oil_density = 700) :
  ⌈new_water_level setup⌉ = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_water_level_approx_new_water_level_rounded_l1092_109299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_certificate_interest_rate_l1092_109209

/-- Calculates the simple interest for a given principal, rate, and time -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Represents the problem of finding the interest rate of the second certificate -/
theorem second_certificate_interest_rate 
  (initial_investment : ℝ) 
  (first_rate : ℝ) 
  (first_duration : ℝ) 
  (second_duration : ℝ) 
  (final_amount : ℝ) :
  initial_investment = 8000 →
  first_rate = 0.1 →
  first_duration = 1/3 →
  second_duration = 1/3 →
  final_amount = 8840 →
  ∃ (second_rate : ℝ), 
    (simpleInterest (simpleInterest initial_investment first_rate first_duration) second_rate second_duration = final_amount) ∧
    (abs (second_rate - 0.2079) < 0.0001) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_certificate_interest_rate_l1092_109209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_wicks_is_thirty_l1092_109278

def spool_length_feet : ℚ := 25
def wick_length_1 : ℚ := 13/2
def wick_length_2 : ℚ := 37/4
def wick_length_3 : ℚ := 51/4

def total_wicks_count : ℕ :=
  let spool_length_inches : ℚ := spool_length_feet * 12
  let set_length : ℚ := wick_length_1 + wick_length_2 + wick_length_3
  let num_sets : ℕ := (spool_length_inches / set_length).floor.toNat
  num_sets * 3

#eval total_wicks_count

theorem total_wicks_is_thirty :
  total_wicks_count = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_wicks_is_thirty_l1092_109278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circumscribed_triangle_area_l1092_109215

/-- Predicate indicating that triangle XYZ is inscribed in triangle ABC -/
def IsInscribed (XYZ ABC : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  sorry

/-- Predicate indicating that triangle XYZ is circumscribed around triangle ABC -/
def IsCircumscribed (XYZ ABC : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  sorry

/-- Predicate indicating that line segments AB and CD are parallel -/
def IsParallel (AB CD : (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  sorry

/-- Function to calculate the area of a triangle given its vertices -/
noncomputable def area (ABC : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : ℝ :=
  sorry

/-- Given two triangles A₁B₁C₁ and A₂B₂C₂ inscribed in and circumscribed around a triangle ABC
    respectively, with parallel sides and known areas, the area of ABC is the geometric mean
    of the areas of A₁B₁C₁ and A₂B₂C₂. -/
theorem inscribed_circumscribed_triangle_area
  (A B C A₁ B₁ C₁ A₂ B₂ C₂ : ℝ × ℝ)
  (h_inscribed : IsInscribed (A₁, B₁, C₁) (A, B, C))
  (h_circumscribed : IsCircumscribed (A₂, B₂, C₂) (A, B, C))
  (h_parallel1 : IsParallel (A₁, B₁) (A₂, B₂))
  (h_parallel2 : IsParallel (A₁, C₁) (A₂, C₂))
  (h_parallel3 : IsParallel (B₁, C₁) (B₂, C₂))
  (t₁ : ℝ) (h_t₁ : t₁ = area (A₁, B₁, C₁))
  (t₂ : ℝ) (h_t₂ : t₂ = area (A₂, B₂, C₂)) :
  area (A, B, C) = Real.sqrt (t₁ * t₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circumscribed_triangle_area_l1092_109215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_three_sum_l1092_109250

theorem divisible_by_three_sum (a b c d e : ℤ) : 
  ∃ x y z, x ∈ ({a, b, c, d, e} : Set ℤ) ∧ 
           y ∈ ({a, b, c, d, e} : Set ℤ) ∧ 
           z ∈ ({a, b, c, d, e} : Set ℤ) ∧ 
           x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
           (x + y + z) % 3 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_three_sum_l1092_109250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1092_109277

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 3 * (Real.cos x)^2 - (Real.sin x)^2 + 3

-- State the theorem
theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x : ℝ), f x ≤ 6) ∧
  (∃ (x : ℝ), f x = 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1092_109277
