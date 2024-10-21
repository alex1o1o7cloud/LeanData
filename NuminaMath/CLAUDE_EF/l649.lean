import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l649_64932

-- Define the sequence
def a : ℕ → ℚ
| 0 => 2  -- Add this case for n = 0
| n + 1 => 10 * a n + 2

-- State the theorem
theorem a_formula (n : ℕ) : a n = (2 * (10^n - 1)) / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l649_64932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_four_equals_thirtytwo_l649_64981

theorem power_of_four_equals_thirtytwo (x : ℝ) : (4 : ℝ)^x = 32 → x = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_four_equals_thirtytwo_l649_64981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclined_triangular_prism_surface_area_l649_64960

/-- Represents an inclined triangular prism -/
structure InclinedTriangularPrism where
  a : ℝ  -- distance between lateral edges
  l : ℝ  -- length of lateral edge
  incline_angle : ℝ  -- angle of inclination of lateral edge to base plane

/-- Calculates the total surface area of an inclined triangular prism -/
noncomputable def total_surface_area (prism : InclinedTriangularPrism) : ℝ :=
  prism.a * (3 * prism.l + prism.a * Real.sqrt 3 / 2)

/-- Theorem stating the total surface area of an inclined triangular prism -/
theorem inclined_triangular_prism_surface_area 
  (prism : InclinedTriangularPrism) 
  (h : prism.incline_angle = π / 3) : 
  total_surface_area prism = prism.a * (3 * prism.l + prism.a * Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclined_triangular_prism_surface_area_l649_64960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_range_implies_x_coordinate_range_l649_64997

theorem tangent_angle_range_implies_x_coordinate_range :
  ∀ (x y : ℝ), 
    y = x^2 + 2*x + 3 →
    (∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ π/4 ∧ (2*x + 2) = Real.tan θ) →
    -1 ≤ x ∧ x ≤ -1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_range_implies_x_coordinate_range_l649_64997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_roots_of_seven_l649_64916

theorem square_roots_of_seven :
  ∀ x : ℝ, x^2 = 7 ↔ x = Real.sqrt 7 ∨ x = -Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_roots_of_seven_l649_64916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_trig_function_l649_64978

theorem max_value_of_trig_function :
  ∀ x : ℝ, Real.cos (2 * x) + 6 * Real.cos (π / 2 - x) ≤ 5 ∧
  ∃ x₀ : ℝ, Real.cos (2 * x₀) + 6 * Real.cos (π / 2 - x₀) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_trig_function_l649_64978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_R_cycle_l649_64951

noncomputable def angle_of_reflection (θ : ℝ) (α : ℝ) : ℝ := 2 * α - θ

noncomputable def R (θ : ℝ) : ℝ := angle_of_reflection (angle_of_reflection θ (Real.pi / 50)) (Real.pi / 45)

noncomputable def R_n : ℕ → ℝ → ℝ 
  | 0, θ => θ
  | n + 1, θ => R (R_n n θ)

theorem smallest_n_for_R_cycle (l : ℝ) : 
  (∃ n : ℕ, n > 0 ∧ R_n n l = R l) → 
  (∃ n : ℕ, n > 0 ∧ R_n n l = R l ∧ ∀ m : ℕ, 0 < m → m < n → R_n m l ≠ R l) →
  (let θ := Real.arctan (21 / 88);
   ∃ n : ℕ, n > 0 ∧ R_n n θ = R θ ∧ ∀ m : ℕ, 0 < m → m < n → R_n m θ ≠ R θ) →
  (∃ n : ℕ, n = 15 ∧ R_n n (Real.arctan (21 / 88)) = R (Real.arctan (21 / 88)) ∧ 
    ∀ m : ℕ, 0 < m → m < 15 → R_n m (Real.arctan (21 / 88)) ≠ R (Real.arctan (21 / 88))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_R_cycle_l649_64951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_of_hyperbola_l649_64911

/-- The equation of the asymptotes of a hyperbola -/
noncomputable def asymptote_equation (a b : ℝ) : ℝ → ℝ := fun x => (a / b) * x

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : ℝ := 16 * x^2 - 9 * y^2 + 144

/-- Theorem: The equation of the asymptotes of the given hyperbola is y = ±(4/3)x -/
theorem asymptotes_of_hyperbola :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), hyperbola_equation x y = 0 →
    (y = asymptote_equation a b x ∨ y = -asymptote_equation a b x)) ∧
  a / b = 4 / 3 := by
  sorry

#check asymptotes_of_hyperbola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_of_hyperbola_l649_64911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_target_l649_64929

noncomputable def options : List ℝ := [1500, 2000, 2500, 3000, 3500]

noncomputable def target : ℝ := 504 / 0.252

theorem closest_to_target :
  ∀ x ∈ options, |2000 - target| ≤ |x - target| := by
  sorry

#check closest_to_target

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_target_l649_64929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l649_64908

noncomputable def f (x : ℝ) : ℝ := Real.sin (-x/2 + Real.pi/4)

theorem smallest_positive_period_of_f : 
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = 4 * Real.pi := by
  sorry

#check smallest_positive_period_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l649_64908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l649_64922

/-- The circle C defined by the equation x^2 + y^2 - 2x + 4y + 4 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y + 4 = 0

/-- Point P with coordinates (0, -1) -/
def point_P : ℝ × ℝ := (0, -1)

/-- A line is tangent to the circle if it touches the circle at exactly one point -/
def is_tangent_line (m b : ℝ) : Prop :=
  ∃! (x y : ℝ), circle_equation x y ∧ y = m * x + b

/-- Function to represent a vertical line -/
def vertical_line (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ x = a

theorem tangent_lines_to_circle :
  (is_tangent_line 0 (-1) ∧ (λ x y ↦ y = -1) point_P.1 point_P.2) ∨
  (is_tangent_line (1/0) 0 ∧ vertical_line 0 point_P.1 point_P.2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l649_64922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sample_correct_l649_64986

/-- Represents the number of students in each year and the sample size -/
structure SchoolData where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ
  sample_size : ℕ

/-- Represents the number of students to be sampled from each year -/
structure SampleDistribution where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ

/-- Calculates the stratified sample distribution given school data -/
def stratifiedSample (data : SchoolData) : SampleDistribution :=
  let total := data.first_year + data.second_year + data.third_year
  let proportion := data.sample_size / total
  { first_year := (data.first_year * proportion),
    second_year := (data.second_year * proportion),
    third_year := (data.third_year * proportion) }

/-- Theorem stating that the stratified sample for the given school data results in the expected distribution -/
theorem stratified_sample_correct (data : SchoolData)
  (h1 : data.first_year = 650)
  (h2 : data.second_year = 550)
  (h3 : data.third_year = 500)
  (h4 : data.sample_size = 68) :
  let result := stratifiedSample data
  result.first_year = 26 ∧ result.second_year = 22 ∧ result.third_year = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sample_correct_l649_64986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_equals_two_l649_64972

theorem tan_alpha_equals_two (α : Real) 
  (h1 : α > 0) (h2 : α < π / 2) 
  (h3 : Real.sin (2 * α) = Real.sin α ^ 2) : 
  Real.tan α = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_equals_two_l649_64972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_selling_price_l649_64900

/-- Calculates the selling price of an article given its cost price and the profit/loss percentage -/
noncomputable def sellingPrice (costPrice : ℝ) (profitPercentage : ℝ) : ℝ :=
  costPrice * (1 + profitPercentage / 100)

/-- Represents the problem of calculating the combined selling price of three articles -/
theorem combined_selling_price (costPrice1 costPrice2 costPrice3 : ℝ) :
  costPrice1 = 70 →
  costPrice2 = 120 →
  costPrice3 = 150 →
  ∃ (sellingPrice1 : ℝ),
    (2/3 * sellingPrice1 = 0.85 * costPrice1) ∧
    (sellingPrice1 + sellingPrice costPrice2 30 + sellingPrice costPrice3 (-20) = 365.25) :=
by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_selling_price_l649_64900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l649_64915

-- Define the function f(x) = ln x + x - 4
noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 4

-- Theorem statement
theorem zero_in_interval :
  ∃! x : ℝ, x ∈ Set.Ioo 2 3 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l649_64915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l649_64988

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus and directrix intersection point
def F : ℝ × ℝ := (2, 0)
def K : ℝ × ℝ := (-2, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

theorem parabola_triangle_area :
  ∀ (x y : ℝ),
    parabola x y →
    distance (x, y) K = Real.sqrt 2 * distance (x, y) F →
    triangleArea (x, y) K F = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l649_64988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_three_or_five_count_l649_64906

theorem divisible_by_three_or_five_count : ℕ := by
  -- Define the set of numbers from 1 to 60
  let S : Finset ℕ := Finset.range 60

  -- Define the property of being divisible by 3 or 5
  let divisible_by_three_or_five (n : ℕ) : Prop := n % 3 = 0 ∨ n % 5 = 0

  -- Count the numbers in S that satisfy the property
  let count := S.filter divisible_by_three_or_five |>.card

  -- Assert that this count is equal to 28
  have h : count = 28 := by sorry

  -- Return the result
  exact count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_three_or_five_count_l649_64906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l649_64979

noncomputable def f (x : ℝ) : ℝ := Real.log x / x

theorem tangent_line_at_one : 
  ∃ (m b : ℝ), (∀ x y : ℝ, y = m * x + b ↔ y = f 1 + (deriv f 1) * (x - 1)) ∧ 
  (m * 1 + b = f 1) ∧ 
  (∀ x y : ℝ, y = m * x + b ↔ x - y - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l649_64979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_mix_solution_l649_64982

/-- Represents the coffee mix problem -/
structure CoffeeMix where
  total_mix : ℚ
  brazilian_price : ℚ
  final_mix_price : ℚ
  columbian_amount : ℚ

/-- The solution to the coffee mix problem -/
def solve_coffee_mix (mix : CoffeeMix) : ℚ :=
  (mix.total_mix * mix.final_mix_price - 
   (mix.total_mix - mix.columbian_amount) * mix.brazilian_price) / mix.columbian_amount

/-- Theorem stating the solution to the specific coffee mix problem -/
theorem coffee_mix_solution :
  let mix : CoffeeMix := {
    total_mix := 100
    brazilian_price := 375/100
    final_mix_price := 635/100
    columbian_amount := 52
  }
  solve_coffee_mix mix = 875/100 := by
  sorry

#eval solve_coffee_mix {
  total_mix := 100
  brazilian_price := 375/100
  final_mix_price := 635/100
  columbian_amount := 52
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_mix_solution_l649_64982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oncoming_train_speed_l649_64969

/-- The speed of the trains departing from Station A -/
noncomputable def departure_speed : ℝ := 50

/-- The time interval between the departure of the two trains from Station A in hours -/
noncomputable def departure_interval : ℝ := 12 / 60

/-- The time interval between the oncoming train's encounters with the two trains in hours -/
noncomputable def encounter_interval : ℝ := 5 / 60

/-- The speed of the oncoming train -/
noncomputable def oncoming_speed : ℝ := 70

theorem oncoming_train_speed :
  let distance_between_trains := departure_speed * departure_interval
  let time_between_encounters := distance_between_trains / (oncoming_speed + departure_speed)
  time_between_encounters = encounter_interval := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oncoming_train_speed_l649_64969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l649_64975

-- Define the point P
def P : ℝ × ℝ := (-1, -1)

-- Define the angle α
noncomputable def α : ℝ := Real.arctan (P.2 / P.1) + Real.pi

-- Theorem statement
theorem sin_alpha_value :
  Real.sin α = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l649_64975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_root_l649_64942

theorem quadratic_polynomial_root : 
  ∃ p : ℂ → ℂ, 
    p = (fun z => z^2 + 6*z + 25) ∧
    p (-3 - 4*Complex.I) = 0 ∧ 
    ∀ a b c : ℝ, (fun z => a*z^2 + b*z + c : ℂ → ℂ) = p → b = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_root_l649_64942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_inequality_l649_64959

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (-2^x + a) / (2^(x+1) + 2)

theorem odd_function_and_inequality (a k : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (∃ θ ∈ Set.Icc (-π/4) 0, f a (Real.sqrt 3 * Real.sin θ * Real.cos θ) + f a (k - Real.cos θ * Real.cos θ) > 0) →
  a = 1 ∧ k < 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_inequality_l649_64959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_from_directrix_and_eccentricity_l649_64965

/-- Represents an ellipse with center at the origin -/
structure Ellipse where
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis
  c : ℝ  -- half of the focal length
  h_positive_a : 0 < a
  h_positive_b : 0 < b
  h_positive_c : 0 < c
  h_a_ge_b : a ≥ b
  h_c_lt_a : c < a
  h_pythagorean : a^2 = b^2 + c^2

/-- The equation of the ellipse -/
noncomputable def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The eccentricity of the ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  e.c / e.a

/-- The equation of the directrix -/
noncomputable def Ellipse.directrix (e : Ellipse) : ℝ :=
  e.a^2 / e.c

theorem ellipse_equation_from_directrix_and_eccentricity 
  (e : Ellipse) 
  (h_directrix : e.directrix = 4)
  (h_eccentricity : e.eccentricity = 1/2) :
  e.equation = λ x y ↦ x^2/4 + y^2/3 = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_from_directrix_and_eccentricity_l649_64965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l649_64917

/-- The time (in seconds) it takes for a train to cross another train moving in the same direction -/
noncomputable def crossing_time (length1 length2 : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  (length1 + length2) / ((speed1 - speed2) * (5 / 18))

/-- Theorem stating that the crossing time for the given train scenario is 64 seconds -/
theorem train_crossing_time :
  crossing_time 280 360 72 36 = 64 := by
  -- Unfold the definition of crossing_time
  unfold crossing_time
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l649_64917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_range_l649_64958

open Set Real

noncomputable def angle_of_inclination (θ : ℝ) : Set ℝ :=
  let A := (cos θ, sin θ ^ 2)
  let B := (0, 1)
  let slope := (1 - sin θ ^ 2) / (-cos θ)
  { α | α ∈ Ioo 0 (π / 4) ∪ Icc (3 * π / 4) π ∧ tan α = slope }

theorem angle_of_inclination_range (θ : ℝ) (hθ : (cos θ, sin θ ^ 2) ≠ (0, 1)) :
  angle_of_inclination θ = Ioo 0 (π / 4) ∪ Icc (3 * π / 4) π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_range_l649_64958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_sum_l649_64987

noncomputable section

-- Define the function f
def f (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

-- State the theorem
theorem inverse_function_sum (a b c d : ℝ) :
  a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 →
  (∀ x, f a b c d (f a b c d x) = x) →
  a + d = 0 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_sum_l649_64987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_triangle_area_l649_64971

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 9 + y^2 / 3 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4 * Real.sqrt 6 * x

-- Define the line l₁
def line_l1 (x y t : ℝ) : Prop := x - 2*y + t = 0

-- Define a perpendicular line to l₁
def perp_line (x y m : ℝ) : Prop := 2*x + y + m = 0

-- Define the triangle area function
noncomputable def triangle_area (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  (1/2) * abs (x₁*y₂ - x₂*y₁)

-- State the theorem
theorem ellipse_max_triangle_area :
  ∃ (F : ℝ × ℝ),
    (∀ x y, ellipse_C x y ↔ parabola x y) →
    (∀ t m x₁ y₁ x₂ y₂,
      ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
      perp_line x₁ y₁ m ∧ perp_line x₂ y₂ m ∧
      line_l1 x₁ y₁ t ∧ line_l1 x₂ y₂ t →
      triangle_area x₁ y₁ x₂ y₂ ≤ 3*Real.sqrt 3 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_triangle_area_l649_64971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_a_100_divisible_by_15_l649_64954

def a : ℕ → ℕ
| 0 => 1  -- We define a₁ = 1 to match the problem statement
| n + 1 => 2 * a n + 1

theorem a_formula (n : ℕ) : a n = 2^n - 1 := by
  sorry

theorem a_100_divisible_by_15 : 15 ∣ a 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_a_100_divisible_by_15_l649_64954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_sevens_theorem_l649_64993

-- Define a type for expressions using sevens
inductive SevenExpr
  | Const : ℕ → SevenExpr
  | Add : SevenExpr → SevenExpr → SevenExpr
  | Sub : SevenExpr → SevenExpr → SevenExpr
  | Mul : SevenExpr → SevenExpr → SevenExpr
  | Div : SevenExpr → SevenExpr → SevenExpr
  | Exp : SevenExpr → SevenExpr → SevenExpr

-- Function to count the number of sevens used in an expression
def countSevens : SevenExpr → ℕ
  | SevenExpr.Const n => if n = 7 then 1 else 0
  | SevenExpr.Add e1 e2 => countSevens e1 + countSevens e2
  | SevenExpr.Sub e1 e2 => countSevens e1 + countSevens e2
  | SevenExpr.Mul e1 e2 => countSevens e1 + countSevens e2
  | SevenExpr.Div e1 e2 => countSevens e1 + countSevens e2
  | SevenExpr.Exp e1 e2 => countSevens e1 + countSevens e2

-- Function to evaluate a SevenExpr to a rational number
noncomputable def evaluate : SevenExpr → ℚ
  | SevenExpr.Const n => n
  | SevenExpr.Add e1 e2 => evaluate e1 + evaluate e2
  | SevenExpr.Sub e1 e2 => evaluate e1 - evaluate e2
  | SevenExpr.Mul e1 e2 => evaluate e1 * evaluate e2
  | SevenExpr.Div e1 e2 => evaluate e1 / evaluate e2
  | SevenExpr.Exp e1 e2 => (evaluate e1) ^ (Int.floor (evaluate e2)).toNat

-- Theorem stating that all numbers from 1 to 22 can be formed using five sevens
theorem five_sevens_theorem :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 22 →
    ∃ expr : SevenExpr, countSevens expr = 5 ∧ evaluate expr = n :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_sevens_theorem_l649_64993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spatial_relationships_option_A_incorrect_l649_64927

-- Define the types for planes and lines
variable (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P]
variable (α β : Subspace ℝ P)
variable (l : AffineSubspace ℝ P)

-- Define the relationships
def parallel (s t : Subspace ℝ P) : Prop := sorry
def perpendicular (s t : Subspace ℝ P) : Prop := sorry
def line_parallel_plane (l : AffineSubspace ℝ P) (p : Subspace ℝ P) : Prop := sorry
def line_perpendicular_plane (l : AffineSubspace ℝ P) (p : Subspace ℝ P) : Prop := sorry
def line_not_subset_plane (l : AffineSubspace ℝ P) (p : Subspace ℝ P) : Prop := sorry

-- State the theorem
theorem spatial_relationships
  (h_diff : α ≠ β)
  (h_not_subset : line_not_subset_plane P l β) :
  (line_parallel_plane P l α ∧ parallel P α β → line_parallel_plane P l β) ∧
  (line_perpendicular_plane P l α ∧ parallel P α β → line_perpendicular_plane P l β) ∧
  (line_perpendicular_plane P l α ∧ perpendicular P α β → line_parallel_plane P l β) :=
by
  sorry

-- Statement for option A (the incorrect one)
theorem option_A_incorrect
  (h_parallel : line_parallel_plane P l α)
  (h_perp : perpendicular P α β) :
  ¬(line_perpendicular_plane P l β) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spatial_relationships_option_A_incorrect_l649_64927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l649_64933

noncomputable def f (x : ℝ) := Real.sin x ^ 2 - Real.sin (x - Real.pi / 6) ^ 2

theorem f_properties :
  ∃ (p : ℝ),
    (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
    p = Real.pi ∧
    (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/3) (Real.pi/4) → f x ≤ Real.sqrt 3 / 4) ∧
    (∃ (x : ℝ), x ∈ Set.Icc (-Real.pi/3) (Real.pi/4) ∧ f x = Real.sqrt 3 / 4) ∧
    (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/3) (Real.pi/4) → f x ≥ -1/2) ∧
    (∃ (x : ℝ), x ∈ Set.Icc (-Real.pi/3) (Real.pi/4) ∧ f x = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l649_64933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_hexagon_angle_l649_64938

/-- Represents the angles of a hexagon in a given ratio --/
structure HexagonAngles where
  ratio : Fin 6 → ℕ
  is_valid_ratio : ratio 0 = 3 ∧ ratio 1 = 3 ∧ ratio 2 = 3 ∧ ratio 3 = 4 ∧ ratio 4 = 5 ∧ ratio 5 = 6

/-- The sum of angles in a hexagon is 720° --/
axiom hexagon_angle_sum : (720 : ℝ) = 720

/-- Theorem: The largest angle in a hexagon with angles in the ratio 3:3:3:4:5:6 is 180° --/
theorem largest_hexagon_angle (h : HexagonAngles) : (180 : ℝ) = 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_hexagon_angle_l649_64938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_r_composition_six_times_l649_64920

noncomputable def r (θ : ℝ) : ℝ := 1 / (2 - θ)

theorem r_composition_six_times (x : ℝ) : r (r (r (r (r (r x))))) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_r_composition_six_times_l649_64920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_ideal_speed_l649_64991

/-- The speed at which Alice should drive to arrive exactly on time -/
noncomputable def ideal_speed (d : ℝ) (t : ℝ) : ℝ := d / t

theorem alice_ideal_speed (d : ℝ) (t : ℝ) 
  (h1 : d = 50 * (t + 1/15))  -- Equation for 50 mph
  (h2 : d = 70 * (t - 1/30))  -- Equation for 70 mph
  : ∃ (speed : ℝ), abs (speed - 57) < 1 ∧ speed = ideal_speed d t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_ideal_speed_l649_64991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_product_l649_64964

variable {n : ℕ}
variable (C D : Matrix (Fin n) (Fin n) ℝ)

theorem det_product (hC : Matrix.det C = 5) (hD : Matrix.det D = 7) : 
  Matrix.det (C * D) = 35 := by
  have h1 : Matrix.det (C * D) = Matrix.det C * Matrix.det D := by
    exact Matrix.det_mul C D
  rw [h1, hC, hD]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_product_l649_64964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l649_64940

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (10 : ℝ)^4 * (100 : ℝ)^x = (1000 : ℝ)^6 ∧ x = 7 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l649_64940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_vector_sum_l649_64962

/-- Regular octagon centered at the origin -/
structure RegularOctagon where
  vertices : Fin 8 → ℂ
  center_at_origin : vertices 0 = 0
  regular : ∀ i : Fin 8, Complex.abs (vertices i) = 1
  octagon : ∀ i : Fin 8, vertices ((i + 1) % 8) = vertices i * Complex.exp (Complex.I * Real.pi / 4)

/-- Vector from one vertex to the next -/
def a (oct : RegularOctagon) (i : Fin 8) : ℂ :=
  oct.vertices ((i + 1) % 8) - oct.vertices i

/-- Vector from origin to a vertex -/
def b (oct : RegularOctagon) (j : Fin 8) : ℂ :=
  oct.vertices j

/-- The main theorem -/
theorem octagon_vector_sum (oct : RegularOctagon) :
  a oct 2 + a oct 5 + b oct 2 + b oct 5 + b oct 7 = b oct 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_vector_sum_l649_64962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_for_factorial_inequality_l649_64955

def is_composite (k : ℕ) : Prop :=
  k > 1 ∧ ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ k = a * b

theorem no_solutions_for_factorial_inequality :
  ¬∃ (n k : ℕ), n ≥ 2 ∧ is_composite k ∧ n ≤ Nat.factorial n - k^n ∧ Nat.factorial n - k^n ≤ k * n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_for_factorial_inequality_l649_64955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_planes_theorem_l649_64966

/-- The surface equation -/
def surface (x y z : ℝ) : Prop := x^2 + 4*y^2 + 9*z^2 = 1

/-- The parallel plane equation -/
def parallel_plane (x y z : ℝ) : Prop := x + y + 2*z = 1

/-- The tangent plane equations -/
def tangent_plane (x y z : ℝ) (k : ℝ) : Prop := x + y + 2*z = k

/-- The specific k values for the tangent planes -/
noncomputable def k₁ : ℝ := 109 / (6 * Real.sqrt 61)
noncomputable def k₂ : ℝ := -109 / (6 * Real.sqrt 61)

/-- The theorem stating that the found planes are tangent to the surface and parallel to the given plane -/
theorem tangent_planes_theorem :
  ∃ (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ),
    surface x₁ y₁ z₁ ∧ surface x₂ y₂ z₂ ∧
    tangent_plane x₁ y₁ z₁ k₁ ∧ tangent_plane x₂ y₂ z₂ k₂ ∧
    (∀ (x y z : ℝ), parallel_plane x y z ↔ ∃ (t : ℝ), x = t ∧ y = t ∧ z = 2*t) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_planes_theorem_l649_64966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toothbrush_theorem_l649_64919

def toothbrush_problem (total : ℕ) (jan feb mar : ℕ) : Prop :=
  let remaining := total - (jan + feb + mar)
  let apr_may := remaining / 2
  let months := [jan, feb, mar, apr_may, apr_may]
  (List.maximum? months).isSome ∧ 
  (List.minimum? months).isSome ∧ 
  (List.maximum? months).get! - (List.minimum? months).get! = 36

theorem toothbrush_theorem : 
  toothbrush_problem 330 53 67 46 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toothbrush_theorem_l649_64919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l649_64999

open Set
open Function
open Real

noncomputable def f (x : ℝ) : ℝ := (1 - Real.sin x) / (3 * Real.sin x + 2)

theorem f_range : range f = Iic (-2) ∪ Ici 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l649_64999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_scores_l649_64980

/-- The set of possible scores for participant B in a mathematics test -/
def possible_scores_B (
  total_questions : ℕ)
  (options_per_question : ℕ)
  (points_per_correct : ℕ)
  (A_score : ℕ)
  (differences : ℕ) : Set ℕ :=
{ score | ∃ (B_correct : ℕ),
    B_correct ≤ total_questions ∧
    score = B_correct * points_per_correct ∧
    (A_score / points_per_correct).sub B_correct ≤ differences ∧
    B_correct.sub (A_score / points_per_correct) ≤ differences }

theorem test_scores 
  (total_questions : ℕ)
  (options_per_question : ℕ)
  (points_per_correct : ℕ)
  (A_score : ℕ)
  (differences : ℕ)
  (h1 : total_questions = 10)
  (h2 : options_per_question = 4)
  (h3 : points_per_correct = 3)
  (h4 : A_score = 27)
  (h5 : differences = 1) :
  possible_scores_B total_questions options_per_question points_per_correct A_score differences = {24, 27, 30} :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_scores_l649_64980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l649_64984

theorem trigonometric_problem (θ : Real) 
  (h1 : θ ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi)) -- θ is in the fourth quadrant
  (h2 : Real.sin (θ + Real.pi / 4) = 3 / 5) : 
  Real.sin θ = -Real.sqrt 2 / 10 ∧ Real.tan (θ - Real.pi / 4) = -4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l649_64984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boulder_weight_in_pounds_l649_64945

/-- Conversion factor from kilograms to pounds -/
noncomputable def kg_to_pound : ℝ := 1 / 0.4536

/-- Weight of the boulder in kilograms -/
def boulder_weight_kg : ℝ := 350

/-- Function to round a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

/-- Theorem stating that a 350 kg boulder weighs 772 pounds to the nearest whole pound -/
theorem boulder_weight_in_pounds :
  round_to_nearest (boulder_weight_kg * kg_to_pound) = 772 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boulder_weight_in_pounds_l649_64945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_valid_number_l649_64909

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  (∀ d, d ∈ (Nat.digits 10 n) → d ≠ 0) ∧
  (Nat.digits 10 n).length = 4 ∧
  (Nat.digits 10 n).toFinset.card = 4 ∧
  (∀ d, d ∈ (Nat.digits 10 n) → n % d = 0)

theorem greatest_valid_number :
  is_valid_number 9864 ∧
  ∀ m, is_valid_number m → m ≤ 9864 :=
by sorry

#check greatest_valid_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_valid_number_l649_64909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_investment_is_960_l649_64943

/-- Represents the cost and quantity of stationery items. -/
structure Stationery where
  costA : ℚ  -- Cost of one item of type A
  costB : ℚ  -- Cost of one item of type B
  totalItems : ℕ  -- Total number of items to be purchased
  minInvestment : ℚ  -- Minimum total investment
  maxInvestment : ℚ  -- Maximum total investment

/-- Calculates the minimum investment for a given Stationery configuration. -/
def minInvestmentCalc (s : Stationery) : ℚ :=
  let x : ℕ := Int.toNat (Int.ceil ((s.minInvestment - 5 * s.totalItems) / (s.costA - s.costB)))
  (s.costA - s.costB) * x + s.costB * s.totalItems

/-- Theorem stating the minimum investment for the given problem. -/
theorem min_investment_is_960 (s : Stationery) 
  (h1 : 2 * s.costA + s.costB = 35)
  (h2 : s.costA + 3 * s.costB = 30)
  (h3 : s.totalItems = 120)
  (h4 : s.minInvestment = 955)
  (h5 : s.maxInvestment = 1000) :
  minInvestmentCalc s = 960 := by
  sorry

#eval minInvestmentCalc 
  { costA := 15, costB := 5, totalItems := 120, minInvestment := 955, maxInvestment := 1000 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_investment_is_960_l649_64943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_subtraction_example_l649_64921

/-- Represents a number in base 8 (octal) --/
structure Octal where
  value : Nat

/-- Converts an Octal to its decimal (base 10) representation --/
def octal_to_decimal (n : Octal) : Nat := sorry

/-- Converts a decimal (base 10) number to its Octal representation --/
def decimal_to_octal (n : Nat) : Octal := sorry

/-- Subtracts two Octal numbers --/
def octal_subtract (a b : Octal) : Octal :=
  decimal_to_octal (octal_to_decimal a - octal_to_decimal b)

/-- Allows creation of Octal numbers from Nat literals --/
instance : OfNat Octal n where
  ofNat := Octal.mk n

theorem octal_subtraction_example : octal_subtract 652 274 = 356 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_subtraction_example_l649_64921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_butterfly_catching_l649_64935

/-- The number of students who caught 1 butterfly -/
def x : ℕ := sorry

/-- The number of students who caught 2 butterflies -/
def y : ℕ := sorry

/-- The number of students who caught 3 butterflies -/
def z : ℕ := sorry

/-- The number of students who did not catch any butterflies -/
def w : ℕ := sorry

theorem butterfly_catching (h1 : x + y + z + w = 18)
                           (h2 : x + 2*y + 3*z = 32)
                           (h3 : x = y + 5)
                           (h4 : x = z + 2) :
  w = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_butterfly_catching_l649_64935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l649_64918

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x, 2^x * (deriv f x) - 2^x * f x * log 2 > 0) →
  2 * f (-2) < f (-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l649_64918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l649_64989

/-- The number of integer values for the side length x in a non-degenerate triangle with other side lengths 14 and 38. -/
theorem triangle_side_count : ∃ (n : ℕ), n = 27 ∧ 
  (∀ x : ℤ, (x > 24 ∧ x < 52) ↔ 
    (x + 14 > 38 ∧ x + 38 > 14 ∧ 14 + 38 > x ∧ x > 0)) ∧
  n = Finset.card (Finset.Icc 25 51) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l649_64989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_lattice_equilateral_triangles_l649_64956

/-- Represents a point in the hexagonal lattice -/
structure LatticePoint where
  x : ℝ
  y : ℝ

/-- Represents the hexagonal lattice -/
structure HexagonalLattice where
  center : LatticePoint
  inner_ring : List LatticePoint
  outer_ring : List LatticePoint

/-- Distance between two lattice points -/
noncomputable def distance (p q : LatticePoint) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- An equilateral triangle in the lattice -/
structure EquilateralTriangle where
  p1 : LatticePoint
  p2 : LatticePoint
  p3 : LatticePoint

/-- Predicate to check if a triangle is equilateral -/
def is_equilateral (t : EquilateralTriangle) : Prop :=
  distance t.p1 t.p2 = distance t.p2 t.p3 ∧ 
  distance t.p2 t.p3 = distance t.p3 t.p1

/-- The main theorem -/
theorem hexagonal_lattice_equilateral_triangles 
  (lattice : HexagonalLattice)
  (h1 : ∀ p, p ∈ lattice.inner_ring → distance lattice.center p = 1)
  (h2 : ∀ p, p ∈ lattice.outer_ring → distance lattice.center p = 2)
  (h3 : ∀ p q, p ∈ lattice.inner_ring → q ∈ lattice.inner_ring → p ≠ q → distance p q = 1 ∨ distance p q = Real.sqrt 3)
  (h4 : ∀ p q, p ∈ lattice.outer_ring → q ∈ lattice.outer_ring → p ≠ q → distance p q = 1 ∨ distance p q = Real.sqrt 3 ∨ distance p q = 2)
  (h5 : lattice.inner_ring.length = 6)
  (h6 : lattice.outer_ring.length = 12) :
  ∃ triangles : List EquilateralTriangle, 
    (∀ t, t ∈ triangles → is_equilateral t) ∧ 
    (triangles.length = 12) ∧
    (∀ t : EquilateralTriangle, 
      (t.p1 ∈ lattice.center :: lattice.inner_ring ++ lattice.outer_ring) ∧
      (t.p2 ∈ lattice.center :: lattice.inner_ring ++ lattice.outer_ring) ∧
      (t.p3 ∈ lattice.center :: lattice.inner_ring ++ lattice.outer_ring) →
      t ∈ triangles) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_lattice_equilateral_triangles_l649_64956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_fraction_is_one_third_l649_64931

/-- Represents the contents of a cup --/
structure CupContents where
  coffee : ℚ
  milk : ℚ

/-- Represents the state of both cups --/
structure CupState where
  cup1 : CupContents
  cup2 : CupContents

/-- Perform the first transfer: 1/3 from Cup 1 to Cup 2 --/
def transfer1 (state : CupState) : CupState := 
  let transferAmount := state.cup1.coffee / 3
  { cup1 := { coffee := state.cup1.coffee - transferAmount, milk := state.cup1.milk },
    cup2 := { coffee := state.cup2.coffee + transferAmount, milk := state.cup2.milk } }

/-- Perform the second transfer: 1/4 from Cup 2 to Cup 1 --/
def transfer2 (state : CupState) : CupState := 
  let totalInCup2 := state.cup2.coffee + state.cup2.milk
  let transferAmount := totalInCup2 / 4
  let coffeeFraction := state.cup2.coffee / totalInCup2
  { cup1 := { coffee := state.cup1.coffee + transferAmount * coffeeFraction, 
              milk := state.cup1.milk + transferAmount * (1 - coffeeFraction) },
    cup2 := { coffee := state.cup2.coffee - transferAmount * coffeeFraction, 
              milk := state.cup2.milk - transferAmount * (1 - coffeeFraction) } }

/-- Perform the third transfer: 1/5 from Cup 1 to Cup 2 --/
def transfer3 (state : CupState) : CupState := 
  let totalInCup1 := state.cup1.coffee + state.cup1.milk
  let transferAmount := totalInCup1 / 5
  let coffeeFraction := state.cup1.coffee / totalInCup1
  { cup1 := { coffee := state.cup1.coffee - transferAmount * coffeeFraction, 
              milk := state.cup1.milk - transferAmount * (1 - coffeeFraction) },
    cup2 := { coffee := state.cup2.coffee + transferAmount * coffeeFraction, 
              milk := state.cup2.milk + transferAmount * (1 - coffeeFraction) } }

/-- Perform all three transfers --/
def performAllTransfers (initialState : CupState) : CupState :=
  transfer3 (transfer2 (transfer1 initialState))

/-- The main theorem to prove --/
theorem coffee_fraction_is_one_third : 
  let initialState : CupState := { cup1 := { coffee := 3, milk := 0 }, 
                                   cup2 := { coffee := 0, milk := 3 } }
  let finalState := performAllTransfers initialState
  let totalInCup2 := finalState.cup2.coffee + finalState.cup2.milk
  finalState.cup2.coffee / totalInCup2 = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_fraction_is_one_third_l649_64931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_correct_l649_64983

/-- The circle in the problem -/
def circle_eq (x y : ℝ) : Prop := (x - 1/2)^2 + (y + 1/2)^2 = 1/2

/-- The line in the problem -/
def line_eq (x y : ℝ) : Prop := 3*x + 4*y + 1 = 0

/-- The distance between the center of the circle and the line -/
noncomputable def distance_center_to_line : ℝ := 1/10

/-- The radius of the circle -/
noncomputable def radius : ℝ := Real.sqrt 2 / 2

/-- The length of the chord -/
noncomputable def chord_length : ℝ := 7/5

theorem chord_length_is_correct :
  2 * Real.sqrt (radius^2 - distance_center_to_line^2) = chord_length := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_correct_l649_64983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_from_ellipse_l649_64994

/-- Given ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

/-- Hyperbola equation to be proved -/
def hyperbola_equation (x y : ℝ) : Prop := x^2/7 - y^2/9 = 1

/-- Foci of the ellipse -/
def ellipse_foci : Set (ℝ × ℝ) := {(Real.sqrt 7, 0), (-Real.sqrt 7, 0)}

/-- Vertices of the ellipse -/
def ellipse_vertices : Set (ℝ × ℝ) := {(4, 0), (-4, 0), (0, 3), (0, -3)}

/-- Theorem stating the relationship between the given ellipse and the resulting hyperbola -/
theorem hyperbola_from_ellipse :
  (∀ x y, ellipse_equation x y → 
    (Set.image Prod.fst ellipse_foci = Set.image Prod.fst ellipse_vertices)) →
  (∀ x y, hyperbola_equation x y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_from_ellipse_l649_64994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_3_squared_l649_64992

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- State the conditions
axiom fg_condition : ∀ x, x ≥ 1 → f (g x) = x^3
axiom gf_condition : ∀ x, x ≥ 1 → g (f x) = x^2
axiom g_27 : g 27 = 27

-- State the theorem to be proved
theorem g_3_squared : (g 3)^2 = 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_3_squared_l649_64992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matematika_encoding_l649_64995

/-- Represents the encoding of a letter --/
inductive Encoding
| Single : Nat → Encoding
| Double : Nat → Nat → Encoding

/-- The set of valid digits for encoding --/
def ValidDigits : Set Nat := {1, 2, 3}

/-- A function that maps letters to their encodings --/
def LetterEncoding : Char → Encoding := sorry

/-- Assertion that different letters have different encodings --/
axiom encoding_injective : ∀ c1 c2 : Char, c1 ≠ c2 → LetterEncoding c1 ≠ LetterEncoding c2

/-- Assertion that ROBOT is encoded as 3112131233 --/
axiom robot_encoding : 
  (LetterEncoding 'R') = Encoding.Double 3 1 ∧
  (LetterEncoding 'O') = Encoding.Double 1 2 ∧
  (LetterEncoding 'B') = Encoding.Double 1 3 ∧
  (LetterEncoding 'T') = Encoding.Double 3 3

/-- Assertion that KROKODIL and BEGEMOT have identical encodings --/
axiom krokodil_begemot_same : 
  (List.map LetterEncoding ['K', 'R', 'O', 'K', 'O', 'D', 'I', 'L']) =
  (List.map LetterEncoding ['B', 'E', 'G', 'E', 'M', 'O', 'T'])

/-- The main theorem to prove --/
theorem matematika_encoding :
  (List.map LetterEncoding ['M', 'A', 'T', 'E', 'M', 'A', 'T', 'I', 'K', 'A']) = 
  [Encoding.Double 2 2, Encoding.Double 3 2, Encoding.Double 3 3, 
   Encoding.Double 1 1, Encoding.Double 2 2, Encoding.Double 3 2, 
   Encoding.Double 3 3, Encoding.Double 2 3, Encoding.Single 1, 
   Encoding.Double 3 2] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matematika_encoding_l649_64995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_property_l649_64928

/-- Given a triangle PQR with centroid G, if GP^2 + GQ^2 + GR^2 = 22, 
    then PQ^2 + PR^2 + QR^2 = 66 -/
theorem triangle_centroid_property (P Q R G : EuclideanSpace ℝ (Fin 3)) 
  (h_centroid : G = (1/3 : ℝ) • (P + Q + R))
  (h_sum_squares : ‖G - P‖^2 + ‖G - Q‖^2 + ‖G - R‖^2 = 22) :
  ‖P - Q‖^2 + ‖P - R‖^2 + ‖Q - R‖^2 = 66 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_property_l649_64928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l649_64934

def M : Set ℕ := {1, 3, 4}

def N : Set ℕ := {x : ℕ | x^2 - 4*x + 3 = 0}

theorem intersection_M_N : M ∩ N = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l649_64934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_removed_tetrahedra_for_cube_side_2_l649_64973

/-- The volume of tetrahedra removed from a cube's corners --/
noncomputable def volume_removed_tetrahedra (cube_side_length : ℝ) : ℝ :=
  let x : ℝ := 2 * (Real.sqrt 2 - 1)
  let height : ℝ := cube_side_length - x / Real.sqrt 2
  let base_area : ℝ := (1 / 2) * (cube_side_length - cube_side_length / Real.sqrt 2) ^ 2
  let tetrahedron_volume : ℝ := (1 / 3) * base_area * height
  8 * tetrahedron_volume

/-- Theorem stating the volume of removed tetrahedra for a cube with side length 2 --/
theorem volume_removed_tetrahedra_for_cube_side_2 :
  volume_removed_tetrahedra 2 = (104 - 72 * Real.sqrt 2) / 3 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval volume_removed_tetrahedra 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_removed_tetrahedra_for_cube_side_2_l649_64973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_property_l649_64907

/-- Given a quadratic function f(x) = ax^2 + bx + c, 
    if f(x₁) = f(x₂) and x₁ ≠ x₂, then f(x₁ + x₂) = c -/
theorem quadratic_function_property 
  (a b c : ℝ) 
  (f : ℝ → ℝ)
  (hf : ∀ x, f x = a * x^2 + b * x + c) 
  (x₁ x₂ : ℝ)
  (h_eq : f x₁ = f x₂) 
  (h_neq : x₁ ≠ x₂) : 
  f (x₁ + x₂) = c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_property_l649_64907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C_at_point_P_l649_64996

-- Define the type for people
inductive Person : Type
  | A | B | C | D

-- Define the type for positions
inductive Position : Type
  | TopLeft | TopRight | BottomLeft | BottomRight

-- Define a function to represent who can see whom
def canSee : Person → Person → Prop := sorry

-- Define the visibility conditions
axiom A_sees_nobody : ∀ p, ¬(canSee Person.A p)
axiom B_sees_only_C : ∀ p, canSee Person.B p ↔ p = Person.C
axiom C_sees_B_and_D : (canSee Person.C Person.B) ∧ (canSee Person.C Person.D)
axiom D_sees_only_C : ∀ p, canSee Person.D p ↔ p = Person.C

-- Define a function to represent the position of each person
def position : Person → Position := sorry

-- Define point P
def P : Position := Position.BottomLeft

-- State the theorem
theorem C_at_point_P : position Person.C = P := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C_at_point_P_l649_64996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_20_l649_64913

def h : ℕ → ℕ
  | 0 => 2  -- Added case for 0
  | 1 => 2
  | 2 => 3
  | n+3 => h (n+2) + h (n+1) + 2*(n+3)

theorem h_20 : h 20 = 1020 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_20_l649_64913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_fraction_above_line_l649_64976

/-- The fraction of a square's area above a line -/
theorem square_area_fraction_above_line (square_vertices : Fin 4 → ℝ × ℝ)
  (line_point1 line_point2 : ℝ × ℝ) : ℚ :=
  by
  have h1 : square_vertices 0 = (2, 0) := by sorry
  have h2 : square_vertices 1 = (5, 0) := by sorry
  have h3 : square_vertices 2 = (5, 3) := by sorry
  have h4 : square_vertices 3 = (2, 3) := by sorry
  have h5 : line_point1 = (2, 3) := by sorry
  have h6 : line_point2 = (5, 0) := by sorry
  
  -- The actual proof would go here
  exact 1/2

#check square_area_fraction_above_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_fraction_above_line_l649_64976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_ratio_sum_theorem_l649_64977

theorem lcm_ratio_sum_theorem (a b : ℕ) : 
  Nat.lcm a b = 48 →
  a = 2 * (b / 3) →
  a + b = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_ratio_sum_theorem_l649_64977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dinner_seating_arrangements_l649_64902

theorem dinner_seating_arrangements (n m : ℕ) (hn : n = 8) (hm : m = 7) :
  (n.choose m) * (Nat.factorial (m - 1)) = 5760 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dinner_seating_arrangements_l649_64902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_segments_line_exists_l649_64930

/-- A line in a plane --/
structure Line where
  -- Define a line (this is a simplified representation)
  mk :: -- Add a constructor

/-- A point in a plane --/
structure Point where
  -- Define a point (this is a simplified representation)
  mk :: -- Add a constructor

/-- Represents the intersection of two lines --/
def intersect (l1 l2 : Line) : Point :=
  sorry

/-- Checks if a point lies on a line --/
def on_line (p : Point) (l : Line) : Prop :=
  sorry

/-- Represents a segment between two points --/
def line_segment (p1 p2 : Point) : Set Point :=
  sorry

/-- Checks if two segments are equal --/
def segment_equal (s1 s2 : Set Point) : Prop :=
  sorry

/-- Main theorem --/
theorem equal_segments_line_exists (P : Point) (a b c : Line) :
  ∃ (l : Line), 
    on_line P l ∧ 
    ∃ (X Y Z : Point), 
      on_line X a ∧ on_line X l ∧
      on_line Y b ∧ on_line Y l ∧
      on_line Z c ∧ on_line Z l ∧
      segment_equal (line_segment X Y) (line_segment Y Z) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_segments_line_exists_l649_64930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_to_sine_curve_l649_64953

/-- Represents a point on the surface of a cylinder --/
structure CylinderPoint where
  θ : Real  -- angle around the base circle
  z : Real  -- height along the axis

/-- Represents the intersection of a plane with a cylinder --/
def PlaneIntersection (α : Real) : Set CylinderPoint :=
  {p : CylinderPoint | p.z = Real.tan α * Real.sin p.θ}

/-- Function that maps points on the cylinder to the unrolled plane --/
def UnrollCylinder (p : CylinderPoint) : Real × Real :=
  (p.θ, p.z)

/-- The main theorem --/
theorem ellipse_to_sine_curve :
  ∀ (α : Real),
  α = π/4 →
  ∃ (f : Real → Real),
  (∀ x, f x = Real.sin x) ∧
  (∀ p ∈ PlaneIntersection α,
    ∃ x, UnrollCylinder p = (x, f x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_to_sine_curve_l649_64953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_k_l649_64990

/-- The polynomial z^12 + z^11 + z^8 + z^7 + z^5 + z^3 + 1 -/
def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^5 + z^3 + 1

/-- Theorem stating that 120 is the smallest positive integer k such that
    z^12 + z^11 + z^8 + z^7 + z^5 + z^3 + 1 divides z^k - 1 -/
theorem smallest_divisible_k : 
  ∀ k : ℕ, (∀ z : ℂ, f z = 0 → z^k = 1) ↔ k ≥ 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_k_l649_64990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_circle_area_ratio_l649_64970

/-- The radius of the circle -/
noncomputable def radius : ℝ := 2

/-- The area of the original circle -/
noncomputable def circle_area : ℝ := Real.pi * radius^2

/-- The side length of the square surrounding the star figure -/
noncomputable def square_side : ℝ := 2 * radius

/-- The area of the square surrounding the star figure -/
noncomputable def square_area : ℝ := square_side^2

/-- The area of one quarter-circle cutout -/
noncomputable def quarter_circle_area : ℝ := (Real.pi * radius^2) / 4

/-- The area of the star figure -/
noncomputable def star_area : ℝ := square_area - 4 * quarter_circle_area

/-- The ratio of the star area to the circle area -/
noncomputable def area_ratio : ℝ := star_area / circle_area

theorem star_circle_area_ratio :
  area_ratio = (4 - Real.pi) / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_circle_area_ratio_l649_64970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_neg_three_l649_64985

/-- The function f(x) = (x^2 + 4x + 5) / (x + 3) has a vertical asymptote at x = -3 -/
theorem vertical_asymptote_at_neg_three :
  ∃ (f : ℝ → ℝ), (∀ x : ℝ, x ≠ -3 → f x = (x^2 + 4*x + 5) / (x + 3)) ∧
  (∀ ε : ℝ, ε > 0 → ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, 0 < |x + 3| ∧ |x + 3| < δ → |f x| > 1/ε) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_neg_three_l649_64985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_of_i_l649_64967

theorem sum_of_powers_of_i (i : ℂ) (h : i^2 = -1) : 
  Finset.sum (Finset.range 604) (λ n => i^n) = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_of_i_l649_64967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vegetable_vendor_problem_l649_64944

/-- Vegetable vendor problem -/
theorem vegetable_vendor_problem 
  (total_weight : ℝ) 
  (total_cost : ℝ) 
  (cabbage_purchase_price : ℝ) 
  (broccoli_purchase_price : ℝ) 
  (cabbage_selling_price_yesterday : ℝ) 
  (broccoli_selling_price : ℝ) 
  (cabbage_damage_rate : ℝ) 
  (h_total_weight : total_weight = 200) 
  (h_total_cost : total_cost = 600) 
  (h_cabbage_purchase : cabbage_purchase_price = 2.8) 
  (h_broccoli_purchase : broccoli_purchase_price = 3.2) 
  (h_cabbage_sell_yesterday : cabbage_selling_price_yesterday = 4) 
  (h_broccoli_sell : broccoli_selling_price = 4.5) 
  (h_damage : cabbage_damage_rate = 0.1) :
  ∃ (min_cabbage_price_today : ℝ), 
    min_cabbage_price_today ≥ 4.1 ∧ 
    ∀ (cabbage_price_today : ℝ), 
      cabbage_price_today ≥ min_cabbage_price_today → 
      (let cabbage_amount := (total_cost - broccoli_purchase_price * total_weight) / (cabbage_purchase_price - broccoli_purchase_price)
       let broccoli_amount := total_weight - cabbage_amount
       let yesterday_profit := (cabbage_selling_price_yesterday - cabbage_purchase_price) * cabbage_amount + 
                               (broccoli_selling_price - broccoli_purchase_price) * broccoli_amount
       let today_profit := (cabbage_price_today - cabbage_purchase_price) * cabbage_amount * (1 - cabbage_damage_rate) + 
                           (broccoli_selling_price - broccoli_purchase_price) * broccoli_amount
       today_profit ≥ yesterday_profit) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vegetable_vendor_problem_l649_64944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_implies_m_less_than_one_l649_64924

-- Define the power function as noncomputable
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^(m-1)

-- State the theorem
theorem power_function_decreasing_implies_m_less_than_one :
  ∀ m : ℝ, (∀ x y : ℝ, 0 < x ∧ x < y → f m y < f m x) → m < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_implies_m_less_than_one_l649_64924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_x_coordinate_range_l649_64923

-- Define the line l: y = 2x - 4
def line_l (x : ℝ) : ℝ := 2 * x - 4

-- Define the point A
def point_A : ℝ × ℝ := (0, 3)

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the condition that the center of C is on line l
def center_on_line (C : Circle) : Prop :=
  C.center.2 = line_l C.center.1

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the condition |MA| = 2|MO|
def distance_condition (M : ℝ × ℝ) : Prop :=
  distance M point_A = 2 * distance M (0, 0)

-- Define the theorem
theorem center_x_coordinate_range (C : Circle) :
  C.radius = 1 ∧
  center_on_line C ∧
  (∃ M : ℝ × ℝ, distance M C.center = C.radius ∧ distance_condition M) →
  0 ≤ C.center.1 ∧ C.center.1 ≤ 12/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_x_coordinate_range_l649_64923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_is_40_l649_64963

/-- Represents a cyclic quadrilateral with angles in arithmetic sequence -/
structure CyclicQuadrilateral where
  /-- The smallest angle of the quadrilateral -/
  a : ℝ
  /-- The common difference in the arithmetic sequence of angles -/
  d : ℝ
  /-- The angles form an arithmetic sequence -/
  angle_sequence : List ℝ := [a, a + d, a + 2*d, a + 3*d]
  /-- The sum of opposite angles in a cyclic quadrilateral is 180° -/
  opposite_angles_sum : a + (a + 3*d) = 180 ∧ (a + d) + (a + 2*d) = 180
  /-- The largest angle is 140° -/
  largest_angle : a + 3*d = 140

/-- Theorem: In a cyclic quadrilateral where the angles form an arithmetic sequence
    and the largest angle is 140°, the smallest angle is 40° -/
theorem smallest_angle_is_40 (q : CyclicQuadrilateral) : q.a = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_is_40_l649_64963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_l649_64901

-- Define the parabola
noncomputable def parabola (x : ℝ) : ℝ := x^2 - 4*x + 4

-- Define the line
noncomputable def line (x : ℝ) : ℝ := 2*x - 6

-- Define the distance function between a point on the parabola and the line
noncomputable def distance (c : ℝ) : ℝ := |2*c - (parabola c) - 6| / Real.sqrt 5

-- Theorem statement
theorem shortest_distance :
  ∃ (min_dist : ℝ), min_dist = 1 / Real.sqrt 5 ∧
  ∀ (c : ℝ), distance c ≥ min_dist := by
  sorry

#check shortest_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_l649_64901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_periodicity_l649_64952

/-- Sequence definition -/
def seq (x : ℕ → ℚ) : Prop :=
  ∀ n, x (n + 1) = 1 - |1 - 2 * x n|

/-- Initial condition -/
def initial_condition (x : ℕ → ℚ) : Prop :=
  0 ≤ x 1 ∧ x 1 ≤ 1

/-- Periodicity definition -/
def is_periodic (x : ℕ → ℚ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ ∀ n, x (n + k) = x n

/-- Main theorem -/
theorem sequence_periodicity (x : ℕ → ℚ) :
  seq x → initial_condition x → (is_periodic x ↔ True) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_periodicity_l649_64952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_in_domain_of_f_composed_l649_64950

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x - 2) ^ (1/3)

-- Theorem statement
theorem smallest_x_in_domain_of_f_composed (x : ℝ) : 
  (∀ y : ℝ, f (f y) ≥ 0 → x ≤ y) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_in_domain_of_f_composed_l649_64950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_160_meters_l649_64957

-- Define the speed of the train in km/h
noncomputable def train_speed : ℝ := 72

-- Define the time taken to cross in seconds
noncomputable def crossing_time : ℝ := 8

-- Define the conversion factor from km/h to m/s
noncomputable def km_h_to_m_s : ℝ := 5 / 18

-- Theorem statement
theorem train_length_is_160_meters :
  let speed_m_s := train_speed * km_h_to_m_s
  let length := speed_m_s * crossing_time
  length = 160 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_160_meters_l649_64957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_xz_length_l649_64974

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  right_angle : (Q.1 - P.1) * (R.1 - Q.1) + (Q.2 - P.2) * (R.2 - Q.2) = 0
  pq_length : Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) = 3
  qr_length : Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2) = 8

-- Define the variable point X on PQ
noncomputable def X (t : ℝ) (P Q : ℝ × ℝ) : ℝ × ℝ :=
  (P.1 + t * (Q.1 - P.1), P.2 + t * (Q.2 - P.2))

-- Define the intersection point Y
noncomputable def Y (t : ℝ) (P Q R : ℝ × ℝ) : ℝ × ℝ :=
  let X := X t P Q
  let m := (R.2 - Q.2) / (R.1 - Q.1)
  let b := X.2 - m * X.1
  let x := (R.2 - m * R.1 - b) / (m - (R.2 - P.2) / (R.1 - P.1))
  (x, m * x + b)

-- Define the intersection point Z
noncomputable def Z (t : ℝ) (P Q R : ℝ × ℝ) : ℝ × ℝ :=
  let Y := Y t P Q R
  (Q.1 + (Y.1 - P.1), Q.2)

-- Define the length of XZ
noncomputable def XZ_length (t : ℝ) (P Q R : ℝ × ℝ) : ℝ :=
  let X := X t P Q
  let Z := Z t P Q R
  Real.sqrt ((Z.1 - X.1)^2 + (Z.2 - X.2)^2)

-- Theorem statement
theorem least_xz_length (P Q R : ℝ × ℝ) (h : Triangle P Q R) :
  ∃ t : ℝ, t ≥ 0 ∧ t ≤ 1 ∧ XZ_length t P Q R = 0 ∧
  ∀ s, s ≥ 0 → s ≤ 1 → XZ_length s P Q R ≥ XZ_length t P Q R := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_xz_length_l649_64974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcyclist_speed_l649_64937

/-- Represents the average speed of a motorcyclist's journey -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

theorem motorcyclist_speed 
  (distance_AB : ℝ)
  (distance_BC : ℝ)
  (time_AB : ℝ)
  (time_BC : ℝ)
  (total_average_speed : ℝ)
  (h1 : distance_AB = 120)
  (h2 : distance_BC = distance_AB / 2)
  (h3 : time_AB = 3 * time_BC)
  (h4 : average_speed (distance_AB + distance_BC) (time_AB + time_BC) = total_average_speed)
  (h5 : total_average_speed = 45)
  : average_speed distance_BC time_BC = 60 := by
  sorry

#check motorcyclist_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcyclist_speed_l649_64937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_candies_l649_64905

noncomputable def basket_of_eggs : Type := Unit

noncomputable def blue_eggs (b : basket_of_eggs) : ℝ := 4/10
noncomputable def purple_eggs (b : basket_of_eggs) : ℝ := 3/10
noncomputable def red_eggs (b : basket_of_eggs) : ℝ := 2/10
noncomputable def green_eggs (b : basket_of_eggs) : ℝ := 1/10

noncomputable def purple_eggs_with_five_candies (b : basket_of_eggs) : ℝ := 1/2
noncomputable def blue_eggs_with_three_candies (b : basket_of_eggs) : ℝ := 1/3
noncomputable def blue_eggs_with_two_candies (b : basket_of_eggs) : ℝ := 1/2

noncomputable def red_eggs_with_one_candy (b : basket_of_eggs) : ℝ := 3/4
noncomputable def red_eggs_with_four_candies (b : basket_of_eggs) : ℝ := 1/4

noncomputable def green_eggs_with_six_candies (b : basket_of_eggs) : ℝ := 1/2
noncomputable def green_eggs_with_eight_candies (b : basket_of_eggs) : ℝ := 1/2

theorem expected_candies (b : basket_of_eggs) :
  blue_eggs b * (blue_eggs_with_three_candies b * 3 + blue_eggs_with_two_candies b * 2) +
  purple_eggs b * (purple_eggs_with_five_candies b * 5) +
  red_eggs b * (red_eggs_with_one_candy b * 1 + red_eggs_with_four_candies b * 4) +
  green_eggs b * (green_eggs_with_six_candies b * 6 + green_eggs_with_eight_candies b * 8) = 2.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_candies_l649_64905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_transformation_l649_64939

/-- Given a quadratic equation ax^2 + bx + c = 0 with roots r and s,
    prove that ar + 2b and as + 2b are roots of x^2 - 3bx + 2b^2 + ac = 0 --/
theorem quadratic_root_transformation (a b c r s : ℝ) 
  (h1 : a * r^2 + b * r + c = 0)
  (h2 : a * s^2 + b * s + c = 0) :
  let f : ℝ → ℝ := λ x ↦ x^2 - 3*b*x + 2*b^2 + a*c
  (f (a*r + 2*b) = 0) ∧ (f (a*s + 2*b) = 0) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_transformation_l649_64939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_identity_l649_64925

/-- Regular pentagon ABCDE with angles A, B, C, D, E -/
structure RegularPentagon where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  sum_angles : A + B + C + D + E = 540
  equal_sides : True  -- Representing that all sides are equal

theorem regular_pentagon_identity (p : RegularPentagon) 
  (hA : p.A = 95)
  (hB : p.B = 105) :
  Real.cos p.A = Real.cos p.C + Real.cos p.D - Real.cos (p.C + p.D) - 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_identity_l649_64925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_journey_distance_l649_64949

/-- Proves that given a journey of 250 km with two parts at speeds of 40 kmph and 60 kmph,
    and a total time of 5.5 hours, the distance covered at 40 kmph is 160 km. -/
theorem bus_journey_distance (x : ℝ) : 
  x ≥ 0 ∧ x ≤ 250 ∧  -- x is the distance at 40 kmph, which must be non-negative and not exceed total distance
  (x / 40 + (250 - x) / 60 = 5.5) →  -- total time equation
  x = 160 := by 
  intro h
  -- The proof steps would go here
  sorry

#check bus_journey_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_journey_distance_l649_64949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l649_64947

/-- A circle in a plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a plane. -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ  -- represents the line ax + by + c = 0

/-- The number of intersection points between a circle and a line. -/
def intersectionPointsCircleLine (c : Circle) (l : Line) : ℕ := sorry

/-- The number of intersection points between two lines. -/
def intersectionPointsLines (l1 l2 : Line) : ℕ := sorry

/-- Two lines are distinct if they are not identical. -/
def distinctLines (l1 l2 : Line) : Prop := l1 ≠ l2

/-- The theorem stating the maximum number of intersection points. -/
theorem max_intersection_points (c : Circle) (l1 l2 : Line) 
  (h : distinctLines l1 l2) : 
  intersectionPointsCircleLine c l1 + 
  intersectionPointsCircleLine c l2 + 
  intersectionPointsLines l1 l2 ≤ 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l649_64947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l649_64926

/-- The volume of a pyramid with a square base of side length 1 and equilateral triangular faces -/
theorem pyramid_volume : 
  ∀ (pyramid : Real → Real → Real → Real),
    (∀ x y z, pyramid x y z ≥ 0) →  -- non-negative volume
    (∀ x y, pyramid x y 0 = 0) →  -- zero volume at height 0
    (∀ x y h, pyramid x y h = (1/3) * x * y * h) →  -- volume formula for pyramid
    (∀ s, pyramid 1 1 (Real.sqrt 3 / 2 * s) = Real.sqrt 3 / 6 * s^3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l649_64926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l649_64948

-- Problem 1
theorem problem_1 : 
  2 * (7^(2/3 : ℝ)) + 2 * ((Real.exp 1 - 1)^0) + 1 / (Real.sqrt 5 + 2) - (16^(1/4 : ℝ)) + ((3 - Real.pi)^4)^(1/4 : ℝ) = 4 + Real.pi + Real.sqrt 5 := by sorry

-- Problem 2
theorem problem_2 : 
  (Real.log 3 / Real.log 2 + Real.log 3 / Real.log 5) * (Real.log 5 / Real.log 3 + Real.log 5 / Real.log 9) * Real.log 2 = 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l649_64948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l649_64914

-- Define the function f
noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

-- Define the function g
noncomputable def g (x φ : ℝ) : ℝ := Real.sin (2 * x + 2 * Real.pi / 3 + φ)

-- Theorem statement
theorem max_value_of_f (φ : ℝ) 
  (h1 : |φ| < Real.pi / 2)
  (h2 : ∀ x, g x φ = g (-x) φ) -- Symmetry of g about the origin
  : ∃ x ∈ Set.Icc 0 (Real.pi / 2), ∀ y ∈ Set.Icc 0 (Real.pi / 2), f x φ ≥ f y φ ∧ f x φ = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l649_64914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l649_64903

/-- The function f(x) as described in the problem -/
noncomputable def f (φ : Real) (x : Real) : Real := 2 * Real.sin (2 * x + Real.pi / 3 + φ)

/-- The function g(x) which is f(x) shifted left by π/2 -/
noncomputable def g (φ : Real) (x : Real) : Real := f φ (x + Real.pi / 2)

theorem function_symmetry (φ : Real) 
  (h1 : |φ| < Real.pi / 2) 
  (h2 : ∀ x, g φ x = g φ (-x)) : 
  φ = Real.pi / 6 ∧ ∀ x, f φ x = 2 * Real.cos (2 * x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l649_64903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_x_distance_at_meeting_l649_64968

/-- Represents a train with its starting point, total distance, and travel time. -/
structure Train where
  start : ℝ
  distance : ℝ
  time : ℝ

/-- The setup of the train problem -/
structure TrainProblem where
  totalRoute : ℝ
  trainX : Train
  trainY : Train
  trainZ : Train

/-- The given problem setup -/
def givenProblem : TrainProblem :=
  { totalRoute := 300
  , trainX := { start := 0, distance := 300, time := 6 }
  , trainY := { start := 150, distance := 150, time := 3.5 }
  , trainZ := { start := 100, distance := 100, time := 2.5 }
  }

/-- Calculate the speed of a train -/
noncomputable def speed (t : Train) : ℝ := t.distance / t.time

/-- Theorem stating the distance Train X travels when it meets Train Y -/
theorem train_x_distance_at_meeting (p : TrainProblem) :
  let tX := p.trainX
  let tY := p.trainY
  let tZ := p.trainZ
  let sX := speed tX
  let sY := speed tY
  let sZ := speed tZ
  let t := (tY.start - tX.start) / (sX + sY)
  abs (sX * t - 80.8) < 0.1 := by sorry

#check train_x_distance_at_meeting givenProblem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_x_distance_at_meeting_l649_64968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_areas_l649_64912

-- Define the points A, B, C, D, E, F, G
variable (A B C D E F G : ℝ × ℝ)

-- Define the length of CF
def CF_length : ℝ := 16

-- Define the right angle FAC
def is_right_angle (A C F : ℝ × ℝ) : Prop :=
  (C.1 - A.1) * (F.1 - A.1) + (C.2 - A.2) * (F.2 - A.2) = 0

-- Define the area of a square given its side length
def square_area (side : ℝ) : ℝ := side ^ 2

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- State the theorem
theorem sum_of_squares_areas (h1 : is_right_angle A C F) (h2 : distance C F = CF_length) :
  square_area (distance A C) + square_area (distance A F) = 256 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_areas_l649_64912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_investment_profit_range_l649_64946

-- Define the profit functions and constraints
noncomputable def y1 (t : ℝ) : ℝ := (1/5) * Real.sqrt t
noncomputable def y2 (a t : ℝ) : ℝ := (1/5) * a * t

-- Define the total profit function
noncomputable def total_profit (a x : ℝ) : ℝ := y1 x + y2 a (5 - x)

-- Part 1: Optimal investment when a = 1/3
theorem optimal_investment :
  ∀ x : ℝ, 1 ≤ x → x ≤ 4 →
  total_profit (1/3) (9/4) ≥ total_profit (1/3) x := by
  sorry

-- Part 2: Range of a for which total profit equals (-4a+3)/5
theorem profit_range :
  {a : ℝ | a > 0 ∧ ∃ x : ℝ, 1 ≤ x ∧ x ≤ 4 ∧ total_profit a x = (-4*a+3)/5} =
  {a : ℝ | 1/5 ≤ a ∧ a ≤ 1/4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_investment_profit_range_l649_64946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l649_64904

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 3 * Real.sin (ω * x + φ)

theorem axis_of_symmetry 
  (ω φ : ℝ) 
  (h1 : |φ| < π/2)
  (h2 : ∀ x, f ω φ (x + π/ω) = f ω φ x)  -- minimum positive period is π
  (h3 : f ω φ (-π/6) = 0) :
  ∃ k : ℤ, f ω φ (-5*π/12 + x) = f ω φ (-5*π/12 - x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l649_64904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blackboard_operation_l649_64998

theorem blackboard_operation (r k : ℝ) (hr : r > 0) (hk : k > 0) :
  ∃ (operation : ℕ → ℝ → ℝ × ℝ) 
    (final_numbers : Finset ℝ),
    (∀ x, x > 0 → 
      let (a, b) := operation 0 x
      2 * x^2 = a * b ∧ a > 0 ∧ b > 0) ∧
    (final_numbers.card = k^2) ∧
    (∃ s ∈ final_numbers, s ≤ k * r) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blackboard_operation_l649_64998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_line_l649_64910

/-- The circle with center (3, -1) and radius 2 -/
def myCircle (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 4

/-- The vertical line x = -3 -/
def verticalLine (x : ℝ) : Prop := x = -3

/-- The distance between two points in ℝ² -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- The minimum distance between a point on the circle and a point on the vertical line -/
theorem min_distance_circle_line : 
  ∃ (min_dist : ℝ), min_dist = 4 ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ), myCircle x₁ y₁ → verticalLine x₂ → 
  distance x₁ y₁ x₂ y₂ ≥ min_dist :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_line_l649_64910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_bound_l649_64941

/-- Given a sequence {a_n} with sum S_n defined by an integral, 
    prove that 1/5 is the maximum value of m for which 
    a_n^2 + (S_n^2/n^2) ≥ ma_1^2 holds for all n. -/
theorem sequence_inequality_bound 
  (a b : ℝ) 
  (a_n : ℕ → ℝ) 
  (S_n : ℕ → ℝ) 
  (h_S_n : ∀ n : ℕ, S_n n = ∫ x in (0 : ℝ)..(n : ℝ), (2*a*x + b)) 
  (h_inequality : ∀ n : ℕ, n > 0 → (a_n n)^2 + (S_n n)^2 / (n : ℝ)^2 ≥ (1/5) * (a_n 1)^2) :
  ∀ m : ℝ, (∀ n : ℕ, n > 0 → (a_n n)^2 + (S_n n)^2 / (n : ℝ)^2 ≥ m * (a_n 1)^2) → m ≤ 1/5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_bound_l649_64941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_is_pi_over_six_area_case1_area_case2_l649_64936

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
noncomputable def satisfies_condition (t : Triangle) : Prop :=
  (t.b - t.a) * (Real.sin t.B + Real.sin t.A) = t.c * (Real.sqrt 3 * Real.sin t.B - Real.sin t.C)

-- Define the area of a triangle
noncomputable def area (t : Triangle) : ℝ := 
  1/2 * t.a * t.c * Real.sin t.B

-- Theorem 1: If the condition is satisfied, then A = π/6
theorem angle_A_is_pi_over_six (t : Triangle) (h : satisfies_condition t) : 
  t.A = Real.pi / 6 := by sorry

-- Theorem 2: If a = 2 and B = π/4, then the area is √3 + 1
theorem area_case1 (t : Triangle) (h : satisfies_condition t) 
  (h1 : t.a = 2) (h2 : t.B = Real.pi / 4) : 
  area t = Real.sqrt 3 + 1 := by sorry

-- Theorem 3: If a = 2 and c = √3b, then the area is √3
theorem area_case2 (t : Triangle) (h : satisfies_condition t) 
  (h1 : t.a = 2) (h2 : t.c = Real.sqrt 3 * t.b) : 
  area t = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_is_pi_over_six_area_case1_area_case2_l649_64936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l649_64961

/-- The line l: x + √3y - 4 = 0 is tangent to the circle C: x^2 + y^2 = 4 -/
theorem line_tangent_to_circle :
  let l : ℝ × ℝ → Prop := λ p ↦ p.1 + Real.sqrt 3 * p.2 - 4 = 0
  let C : ℝ × ℝ → Prop := λ p ↦ p.1^2 + p.2^2 = 4
  let center : ℝ × ℝ := (0, 0)
  let radius : ℝ := 2
  ∃ (p : ℝ × ℝ), l p ∧ C p ∧
    ∀ (q : ℝ × ℝ), l q → C q → q = p :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l649_64961
