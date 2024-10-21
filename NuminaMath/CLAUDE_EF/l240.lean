import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_cosine_range_l240_24002

theorem monotonic_cosine_range (ω : ℝ) (h_pos : ω > 0) :
  (∀ x ∈ Set.Ioo (π/2) π, StrictMono (λ x => Real.cos (ω * x + π/4))) →
  3/2 ≤ ω ∧ ω ≤ 7/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_cosine_range_l240_24002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solve_complex_problem_l240_24080

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the complex number a + bi
def complex_num (a b : ℝ) : ℂ := a + b * i

-- Define the equation for a + bi
theorem complex_equation (a b : ℝ) : 
  complex_num a b = (1 - i)^2 + 2*(5 + i) / (3 + i) :=
sorry

-- Define z
def z (y : ℝ) : ℂ := -1 + y * i

-- Define the condition for the bisector of first and third quadrants
def on_bisector (w : ℂ) : Prop := w.re = w.im

-- Main theorem
theorem solve_complex_problem :
  ∃ (a b y : ℝ),
    (complex_num a b = (1 - i)^2 + 2*(5 + i) / (3 + i)) ∧
    on_bisector ((complex_num a b) * (z y)) ∧
    a = 3 ∧ b = -1 ∧ y = -2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solve_complex_problem_l240_24080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_edge_angle_l240_24070

/-- Represents a rectangular prism -/
structure RectangularPrism where
  /-- The angle between the lateral edge and a side of the base -/
  lateral_edge_base_angle : ℝ
  /-- The inclination angle of the lateral edge to the base plane -/
  lateral_edge_inclination : ℝ
  /-- Property that the lateral edge forms equal angles with the sides of the base -/
  lateral_edge_equal_angles : Prop

/-- 
Given a rectangular prism where:
- The lateral edge forms equal angles with the sides of the base
- The lateral edge is inclined to the plane of the base at an angle α
This theorem states that the angle θ between the lateral edge and a side of the base
is given by θ = arccos(√2 * cos(α) / 2)
-/
theorem lateral_edge_angle (α : ℝ) :
  let θ := Real.arccos ((Real.sqrt 2 * Real.cos α) / 2)
  ∃ (prism : RectangularPrism), 
    prism.lateral_edge_equal_angles ∧ 
    prism.lateral_edge_inclination = α ∧
    prism.lateral_edge_base_angle = θ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_edge_angle_l240_24070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_on_interval_l240_24074

open Real

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x + π / 6)

-- Define the transformed function g
noncomputable def g (x : ℝ) : ℝ := 2 * sin (x + π / 3)

-- State the theorem
theorem range_of_g_on_interval :
  ∀ y ∈ Set.range (fun x => g x) ∩ Set.Icc 0 (5 * π / 6),
  -1 ≤ y ∧ y ≤ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_on_interval_l240_24074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relation_l240_24021

theorem sin_cos_relation :
  (∀ α β : Real, Real.sin α + Real.cos β = 0 → (Real.sin α)^2 + (Real.sin β)^2 = 1) ∧
  (∃ α β : Real, (Real.sin α)^2 + (Real.sin β)^2 = 1 ∧ Real.sin α + Real.cos β ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relation_l240_24021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_uniqueness_l240_24037

noncomputable def q (x : ℂ) : ℂ := x^3 + (27/4 : ℝ)*x^2 - 23*x - (351/4 : ℝ)

theorem cubic_polynomial_uniqueness :
  (∀ x, q x = x^3 + (a : ℝ)*x^2 + (b : ℝ)*x + (c : ℝ) → a = 27/4 ∧ b = -23 ∧ c = -351/4) ∧
  q (2 - 3*Complex.I) = 0 ∧
  q 0 = 40 ∧
  (deriv q) 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_uniqueness_l240_24037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equal_consecutive_digits_is_five_l240_24055

/-- The maximum number of consecutive digits that can be underlined
    to form equal numbers in the sequence of three-digit numbers from 100 to 999 -/
def max_equal_consecutive_digits : ℕ := 5

/-- Theorem stating that 5 is the maximum number of consecutive digits
    that can be underlined to form equal numbers in the sequence -/
theorem max_equal_consecutive_digits_is_five :
  max_equal_consecutive_digits = 5 := by
  -- The proof would go here
  sorry

#check max_equal_consecutive_digits_is_five

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equal_consecutive_digits_is_five_l240_24055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_value_l240_24007

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 2 then x - 1
  else if -2 ≤ x ∧ x ≤ 0 then -1
  else 0  -- undefined for x outside [-2, 2]

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f x + a * x

-- State the theorem
theorem even_function_implies_a_value :
  ∀ a : ℝ, (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → g a x = g a (-x)) → a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_value_l240_24007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_problem_l240_24036

-- Define the circles and points
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*y + 8 = 0
def point_M : ℝ × ℝ := (0, 2)
def point_N : ℝ × ℝ := (2, 0)

-- Define circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define line L
def line_L (x y k : ℝ) : Prop := y = k*x - (k + 1)

-- Helper function for arc length ratio (not implemented)
noncomputable def arc_length_ratio (circle : (ℝ → ℝ → Prop)) (p1 p2 : ℝ × ℝ) : ℝ := 
  sorry

-- State the theorem
theorem circle_and_line_problem :
  -- Circle O is tangent to circle C at point M
  (∀ x y, circle_C x y ↔ circle_O x y) →
  -- Circle O passes through point N
  circle_O point_N.1 point_N.2 →
  -- Line L intercepts two arc lengths on circle O with ratio 3:1
  (∃ k : ℝ, ∀ x y, line_L x y k → 
    (∃ a b : ℝ, circle_O a b ∧ line_L a b k ∧ 
      (∃ c d : ℝ, circle_O c d ∧ line_L c d k ∧ 
        arc_length_ratio circle_O (a, b) (c, d) = 3))) →
  -- Conclusion: The equation of circle O is x²+y²=4 and k = 1
  (∀ x y, circle_O x y ↔ x^2 + y^2 = 4) ∧ (∃ k : ℝ, k = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_problem_l240_24036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fiftieth_number_is_21_l240_24031

def row_length (n : ℕ) : ℕ := 2 * n

def row_value (n : ℕ) : ℕ := 3 * n

def cumulative_length (n : ℕ) : ℕ := 
  (List.range n).foldl (λ acc i => acc + row_length (i + 1)) 0

def value_at_position (pos : ℕ) : ℕ :=
  match (List.range pos).findSome? (λ n => if cumulative_length (n + 1) ≥ pos then some (n + 1) else none) with
  | some n => row_value n
  | none => 0

theorem fiftieth_number_is_21 : value_at_position 50 = 21 := by
  sorry

#eval value_at_position 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fiftieth_number_is_21_l240_24031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_sum_l240_24016

open InnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem magnitude_of_vector_sum (a b : V) 
  (h1 : ‖a‖ = 2) 
  (h2 : ‖b‖ = 3) 
  (h3 : inner a b = ‖a‖ * ‖b‖ * (1/2)) : 
  ‖a + b‖ = Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_sum_l240_24016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l240_24050

-- Define sets A and B
def A : Set ℝ := {x | Real.exp (x * Real.log 2) > 4}
def B : Set ℝ := {x | |x - 1| < 3}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l240_24050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_of_specific_angle_l240_24086

/-- An angle is represented by its terminal point, assuming the initial side is on the positive x-axis -/
structure Angle where
  x : ℝ
  y : ℝ

/-- The tangent of an angle -/
noncomputable def tan (α : Angle) : ℝ := α.y / α.x

theorem tan_of_specific_angle :
  let α : Angle := { x := -2, y := 1 }
  tan α = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_of_specific_angle_l240_24086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_eccentricity_l240_24061

-- Define the geometric sequence
def is_geometric_sequence (a b c : ℝ) : Prop :=
  b ^ 2 = a * c

-- Define the conic section
def conic_section (m : ℝ) : (ℝ × ℝ) → Prop :=
  λ (x, y) ↦ x^2 / m + y^2 = 1

-- Define eccentricity for a conic section
noncomputable def eccentricity (m : ℝ) : ℝ :=
  Real.sqrt (1 - m)

-- Theorem statement
theorem conic_eccentricity (m : ℝ) :
  is_geometric_sequence m 6 (-9) →
  eccentricity m = Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_eccentricity_l240_24061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_percentage_is_fifty_percent_l240_24015

/-- Represents the financial situation of a person over two years -/
structure FinancialSituation where
  income1 : ℝ  -- Income in the first year
  savings1 : ℝ  -- Savings in the first year
  income2 : ℝ  -- Income in the second year
  savings2 : ℝ  -- Savings in the second year

/-- The financial situation satisfies the given conditions -/
def SatisfiesConditions (fs : FinancialSituation) : Prop :=
  fs.income1 > 0 ∧
  fs.savings1 > 0 ∧
  fs.income2 = 1.5 * fs.income1 ∧
  fs.savings2 = 2 * fs.savings1 ∧
  (fs.income1 - fs.savings1) + (fs.income2 - fs.savings2) = 2 * (fs.income1 - fs.savings1)

/-- The percentage of income saved in the first year -/
noncomputable def SavingsPercentage (fs : FinancialSituation) : ℝ :=
  (fs.savings1 / fs.income1) * 100

/-- Theorem: Given the conditions, the savings percentage in the first year is 50% -/
theorem savings_percentage_is_fifty_percent (fs : FinancialSituation) 
  (h : SatisfiesConditions fs) : SavingsPercentage fs = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_percentage_is_fifty_percent_l240_24015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_l240_24044

theorem sum_of_angles (α β : Real) (h1 : Real.tan α + Real.tan β - Real.tan α * Real.tan β + 1 = 0)
  (h2 : π / 2 < α) (h3 : α < π) (h4 : π / 2 < β) (h5 : β < π) :
  α + β = 7 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_l240_24044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_from_origin_l240_24028

-- Define the post position
def post : ℝ × ℝ := (3, 4)

-- Define the rope length
def rope_length : ℝ := 15

-- Define the wall
def wall_x : ℝ := 5
def wall_y_min : ℝ := 4
def wall_y_max : ℝ := 9

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the constraint for points within rope length of the post
def within_rope (p : ℝ × ℝ) : Prop :=
  distance p post ≤ rope_length

-- Define the constraint for points not beyond the wall
def not_beyond_wall (p : ℝ × ℝ) : Prop :=
  p.1 ≤ wall_x ∨ p.2 < wall_y_min ∨ p.2 > wall_y_max

-- Theorem statement
theorem max_distance_from_origin :
  ∃ (max_dist : ℝ), max_dist = 20 ∧
  ∀ (p : ℝ × ℝ), within_rope p → not_beyond_wall p →
  distance p (0, 0) ≤ max_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_from_origin_l240_24028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_greater_than_one_f_satisfies_properties_l240_24059

/-- A function that satisfies the given properties -/
noncomputable def f (x : ℝ) : ℝ := 1 + Real.exp x

/-- Theorem stating that f is an increasing function on ℝ -/
theorem f_increasing : StrictMono f := by sorry

/-- Theorem stating that f(x) > 1 for all x ∈ ℝ -/
theorem f_greater_than_one : ∀ x : ℝ, f x > 1 := by sorry

/-- Main theorem combining both properties -/
theorem f_satisfies_properties : 
  (StrictMono f) ∧ (∀ x : ℝ, f x > 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_greater_than_one_f_satisfies_properties_l240_24059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sock_draw_probability_l240_24033

/-- The number of socks -/
def total_socks : ℕ := 12

/-- The number of colors -/
def num_colors : ℕ := 6

/-- The number of socks per color -/
def socks_per_color : ℕ := 2

/-- The number of socks drawn -/
def drawn_socks : ℕ := 6

/-- The probability of drawing exactly one pair of socks with the same color -/
def prob_one_pair : ℚ := 20 / 77

theorem sock_draw_probability :
  (Nat.choose total_socks drawn_socks *
   (Nat.choose num_colors 3 * Nat.choose 3 1 * socks_per_color * socks_per_color)) /
  (Nat.choose total_socks drawn_socks) = prob_one_pair := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sock_draw_probability_l240_24033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_p_coordinates_l240_24013

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 4)^2 + (y - 2)^2 = 9
def C₂ (x y : ℝ) : Prop := (x - 5)^2 + (y - 6)^2 = 9

-- Define the condition for point P
def satisfies_condition (P : ℝ × ℝ) : Prop :=
  ∀ k : ℝ, k ≠ 0 →
    let (a, b) := P
    let l₁ := fun (x y : ℝ) => y - b = k * (x - a)
    let l₂ := fun (x y : ℝ) => y - b = (-1/k) * (x - a)
    ∃ (x₁ y₁ x₂ y₂ : ℝ),
      C₁ x₁ y₁ ∧ C₂ x₂ y₂ ∧
      l₁ x₁ y₁ ∧ l₂ x₂ y₂ ∧
      (x₁ - a)^2 + (y₁ - b)^2 = (x₂ - a)^2 + (y₂ - b)^2

-- The main theorem
theorem point_p_coordinates :
  ∀ P : ℝ × ℝ, satisfies_condition P →
    (P = (-3/2, 17/2) ∨ P = (5/2, -1/2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_p_coordinates_l240_24013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_equation_satisfies_general_l240_24042

-- Define the parametric equations
noncomputable def x (θ : ℝ) : ℝ := 2 + 2 * Real.sin θ
noncomputable def y (θ : ℝ) : ℝ := 2 * Real.cos θ

-- State the theorem
theorem parametric_equation_satisfies_general (θ : ℝ) :
  (x θ)^2 + (y θ)^2 - 4*(x θ) = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_equation_satisfies_general_l240_24042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_pattern_l240_24051

theorem equation_pattern (a b : ℝ) (n : ℕ+) (h1 : b ≠ 1) (h2 : a * b = 1) :
  (1 + a^(n : ℝ)) / (1 + b^(n : ℝ)) = ((1 + a) / (1 + b))^(n : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_pattern_l240_24051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_equality_l240_24064

noncomputable def f (x : ℝ) : ℝ := x - 1

noncomputable def g (x : ℝ) : ℝ := 2 * x

noncomputable def f_inv (x : ℝ) : ℝ := x + 1

noncomputable def g_inv (x : ℝ) : ℝ := x / 2

theorem composition_equality :
  f (g_inv (f_inv (f_inv (g (f 10))))) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_equality_l240_24064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_grade3_percent_l240_24092

/-- Represents a school with a total number of students and a percentage of grade 3 students -/
structure School where
  total_students : ℕ
  grade3_percent : ℚ

/-- Calculates the number of grade 3 students in a school -/
def grade3_students (s : School) : ℕ :=
  (s.total_students : ℚ) * s.grade3_percent |>.floor.toNat

/-- Theorem stating that the percentage of grade 3 students in the combined population is 17% -/
theorem combined_grade3_percent
  (maplewood : School)
  (brookside : School)
  (h_maplewood_total : maplewood.total_students = 150)
  (h_brookside_total : brookside.total_students = 250)
  (h_maplewood_grade3 : maplewood.grade3_percent = 15 / 100)
  (h_brookside_grade3 : brookside.grade3_percent = 18 / 100) :
  (grade3_students maplewood + grade3_students brookside : ℚ) /
  (maplewood.total_students + brookside.total_students : ℚ) = 17 / 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_grade3_percent_l240_24092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_remain_parallel_l240_24009

-- Define the concept of a line
def Line : Type := Unit

-- Define the concept of an affine transformation
def AffineTransformation : Type := Unit

-- Define what it means for two lines to be parallel
def Parallel (L₁ L₂ : Line) : Prop := True

-- Define the property that affine transformations map lines to lines
def affine_maps_lines (T : AffineTransformation) (L : Line) : Line := L

-- Define the property that affine transformations preserve collinearity
axiom affine_preserves_collinearity (T : AffineTransformation) : True

-- Define the property that affine transformations are one-to-one
axiom affine_one_to_one (T : AffineTransformation) : Function.Injective (λ _ : Unit => ())

-- State the theorem
theorem parallel_lines_remain_parallel 
  (L₁ L₂ : Line) (T : AffineTransformation) :
  Parallel L₁ L₂ → Parallel (affine_maps_lines T L₁) (affine_maps_lines T L₂) :=
by
  intro h
  exact h


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_remain_parallel_l240_24009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l240_24041

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  S_5_eq : S 5 = 35
  a_5_7_mean : (a 5 + a 7) / 2 = 13

/-- The nth term of the sequence -/
noncomputable def a_n (n : ℕ) : ℝ := 2 * n + 1

/-- The sum of the first n terms -/
noncomputable def S_n (n : ℕ) : ℝ := n^2 + 2 * n

/-- The nth term of the derived sequence b_n -/
noncomputable def b_n (n : ℕ) : ℝ := 4 / ((a_n n)^2 - 1)

/-- The sum of the first n terms of the derived sequence -/
noncomputable def T_n (n : ℕ) : ℝ := n / (n + 1)

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = a_n n) ∧
  (∀ n : ℕ, seq.S n = S_n n) ∧
  (∀ n : ℕ, T_n n = n / (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l240_24041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moving_time_calculation_l240_24034

theorem moving_time_calculation (filling_time driving_time num_trips : ℕ) :
  filling_time = 15 ∧ driving_time = 30 ∧ num_trips = 6 →
  (filling_time + driving_time) * num_trips / 60 = (9 : ℚ) / 2 := by
  sorry

#check moving_time_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_moving_time_calculation_l240_24034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_samuel_car_efficiency_l240_24047

/-- Represents the fuel efficiency problem for Samuel's car. -/
structure CarEfficiency where
  highway_mpg : ℚ
  total_miles : ℚ
  total_gallons : ℚ
  highway_city_diff : ℚ

/-- Calculates the city miles per gallon given the car efficiency data. -/
noncomputable def city_mpg (ce : CarEfficiency) : ℚ :=
  let city_miles := (ce.total_miles - ce.highway_city_diff) / 2
  let highway_miles := city_miles + ce.highway_city_diff
  let highway_gallons := highway_miles / ce.highway_mpg
  let city_gallons := ce.total_gallons - highway_gallons
  city_miles / city_gallons

/-- Theorem stating that given the problem conditions, the city mpg is 30. -/
theorem samuel_car_efficiency :
  let ce : CarEfficiency := {
    highway_mpg := 37,
    total_miles := 365,
    total_gallons := 11,
    highway_city_diff := 5
  }
  city_mpg ce = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_samuel_car_efficiency_l240_24047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l240_24018

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := -1/2 * x^2 + m * Real.log x

-- Define the property of f being decreasing on (1, +∞)
def is_decreasing_on_interval (m : ℝ) : Prop :=
  ∀ x y, 1 < x → x < y → f m x > f m y

-- State the theorem
theorem range_of_m :
  {m : ℝ | is_decreasing_on_interval m} = Set.Iic 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l240_24018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_equal_distance_to_axes_l240_24027

/-- A point with coordinates (2x-2, -x+4) has equal distance to both coordinate axes
    if and only if its coordinates are (2, 2) or (-6, 6) -/
theorem point_equal_distance_to_axes :
  ∀ x : ℝ, 
  let P : ℝ × ℝ := (2*x - 2, -x + 4)
  (abs (2*x - 2) = abs (-x + 4)) ↔ (P = (2, 2) ∨ P = (-6, 6)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_equal_distance_to_axes_l240_24027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_two_digit_prime_factor_l240_24088

def product : ℕ := Finset.prod (Finset.range 49) (fun i => i + 101)

theorem largest_two_digit_prime_factor :
  (∃ p : ℕ, Nat.Prime p ∧ p < 100 ∧ p > 9 ∧ p ∣ product ∧
    ∀ q : ℕ, Nat.Prime q → q < 100 → q > 9 → q ∣ product → q ≤ p) ∧
  (73 ∣ product) ∧ Nat.Prime 73 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_two_digit_prime_factor_l240_24088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_for_given_fare_l240_24063

/-- Represents the taxi fare structure in a city -/
structure TaxiFare where
  startingPrice : ℝ
  initialDistance : ℝ
  additionalFarePerKm : ℝ

/-- Calculates the total fare for a given distance -/
noncomputable def calculateFare (fare : TaxiFare) (distance : ℝ) : ℝ :=
  if distance ≤ fare.initialDistance then
    fare.startingPrice
  else
    fare.startingPrice + (distance - fare.initialDistance) * fare.additionalFarePerKm

/-- Theorem: Given the specific taxi fare structure and a total fare of 20 yuan, 
    the distance traveled is 9 km -/
theorem distance_for_given_fare (fare : TaxiFare) 
    (h1 : fare.startingPrice = 8)
    (h2 : fare.initialDistance = 3)
    (h3 : fare.additionalFarePerKm = 2)
    (h4 : calculateFare fare 9 = 20) :
    ∃ (distance : ℝ), calculateFare fare distance = 20 ∧ distance = 9 := by
  sorry

#check distance_for_given_fare

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_for_given_fare_l240_24063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_50_value_l240_24067

def b : ℕ → ℕ
  | 0 => 3  -- Adding the base case for 0
  | 1 => 3
  | n + 1 => b n + 3 * n

theorem b_50_value : b 50 = 3678 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_50_value_l240_24067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_origin_l240_24089

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.cos x * Real.sin x + x - (a - 1) * x^2

theorem tangent_slope_at_origin (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (deriv (f a)) 0 = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_origin_l240_24089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l240_24066

theorem product_remainder (a b c : ℕ) : 
  a % 7 = 2 → b % 7 = 3 → c % 7 = 5 → (a * b * c) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l240_24066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_squared_l240_24058

open Real

noncomputable def f (x : ℝ) := -x^2 + 3 * log x
def g (x : ℝ) := x + 2

theorem min_distance_squared :
  ∃ (a b c d : ℝ), 0 < a ∧ b = f a ∧ d = g c ∧ 
  (∀ (x y : ℝ), (x - c)^2 + (y - d)^2 ≥ (a - c)^2 + (b - d)^2) ∧
  (a - c)^2 + (b - d)^2 = 8 := by
  sorry

#check min_distance_squared

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_squared_l240_24058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l240_24054

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : (Real.cos t.A) / (1 + Real.sin t.A) = (Real.sin (2 * t.B)) / (1 + Real.cos (2 * t.B)))
  (h2 : t.C = 2 * Real.pi / 3)
  (h3 : t.A + t.B + t.C = Real.pi)
  (h4 : Real.sin t.A / t.a = Real.sin t.B / t.b)
  (h5 : Real.sin t.A / t.a = Real.sin t.C / t.c) : 
  (t.B = Real.pi / 6) ∧ 
  (∃ (x : ℝ), x = (t.a^2 + t.b^2) / t.c^2 ∧ 
   ∀ (y : ℝ), y = (t.a^2 + t.b^2) / t.c^2 → x ≤ y) ∧
  ((t.a^2 + t.b^2) / t.c^2 ≥ 4 * Real.sqrt 2 - 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l240_24054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_job_theorem_l240_24014

noncomputable def team_a_time : ℝ := 2.5
noncomputable def team_b_time : ℝ := 75 / 60
noncomputable def total_time : ℝ := 1.5

theorem painting_job_theorem (m n : ℕ) (h_coprime : Nat.Coprime m n) :
  (m : ℝ) / n * team_a_time + (1 - (m : ℝ) / n) * team_b_time = total_time →
  m + n = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_job_theorem_l240_24014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l240_24069

noncomputable def f (x : ℝ) := Real.sqrt ((3 * x + 6) / (1 - x))

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f x = y} = {x : ℝ | -2 ≤ x ∧ x < 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l240_24069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_triangle_perimeter_l240_24081

/-- Represents a triangle in the sequence --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Generates the next triangle in the sequence --/
noncomputable def nextTriangle (t : Triangle) : Triangle :=
  { a := (t.b + t.c - t.a) / 2,
    b := (t.a + t.c - t.b) / 2,
    c := (t.a + t.b - t.c) / 2 }

/-- Checks if a triangle is valid (satisfies triangle inequality) --/
def isValidTriangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- Calculates the perimeter of a triangle --/
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- The initial triangle in the sequence --/
def T₁ : Triangle :=
  { a := 2021, b := 2022, c := 2023 }

/-- Generates the nth triangle in the sequence --/
noncomputable def Tₙ : ℕ → Triangle
  | 0 => T₁
  | n + 1 => nextTriangle (Tₙ n)

/-- The main theorem to be proved --/
theorem last_triangle_perimeter :
  ∃ n : ℕ, (isValidTriangle (Tₙ n) ∧
            ¬isValidTriangle (Tₙ (n + 1)) ∧
            perimeter (Tₙ n) = 1516.5 / 256) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_triangle_perimeter_l240_24081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_angle_l240_24060

-- Define the curve
noncomputable def curve (x : ℝ) : ℝ := x^3

-- Define point B on the curve
structure PointB where
  x : ℝ
  y : ℝ
  on_curve : y = curve x

-- Define the tangent line
structure TangentLine where
  B : PointB
  slope : ℝ
  is_tangent : slope = 3 * B.x^2  -- Derivative of x^3 is 3x^2

-- Define point A where the tangent line intersects x-axis
noncomputable def point_A (l : TangentLine) : ℝ × ℝ :=
  (l.B.x - l.B.y / l.slope, 0)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem tangent_line_angle (B : PointB) (l : TangentLine) 
  (h_tangent : l.B = B)
  (h_isosceles : (point_A l).1^2 + (point_A l).2^2 = B.x^2 + B.y^2) :
  Real.arctan l.slope = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_angle_l240_24060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l240_24098

noncomputable def f (x : ℝ) : ℝ := (1/3)^x + 1

noncomputable def g (x : ℝ) : ℝ := f x - 1

theorem f_satisfies_conditions :
  (∃ a : ℝ, ∀ x : ℝ, g x = a^x) ∧
  (∀ x y : ℝ, x < y → f x > f y) ∧
  (f (-1) > 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l240_24098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pq_ab_ratio_l240_24071

/-- Given a triangle ABC and points P and Q inside it, if certain vector equations hold,
    then the ratio of PQ to AB is 1/30. -/
theorem pq_ab_ratio (A B C P Q : ℝ × ℝ) : 
  (P - A) + 2 • (P - B) + 3 • (P - C) = (0, 0) →
  2 • (Q - A) + 3 • (Q - B) + 5 • (Q - C) = (0, 0) →
  ‖P - Q‖ / ‖A - B‖ = 1 / 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pq_ab_ratio_l240_24071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_circle_l240_24057

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 1

-- Define the tangent condition
def is_tangent (x y : ℝ) : Prop := x - 2*y + 2 = 0

-- Define the distance function
noncomputable def distance (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

-- Theorem statement
theorem min_distance_to_circle :
  ∃ (x y : ℝ), 
    my_circle x y ∧ 
    is_tangent x y ∧ 
    (∀ (x' y' : ℝ), my_circle x' y' ∧ is_tangent x' y' → distance x y ≤ distance x' y') ∧
    distance x y = 2 * Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_circle_l240_24057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_equivalence_l240_24030

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x) - Real.cos (2 * x)

theorem translation_equivalence :
  ∀ x : ℝ, f (x + π/4) = g x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_equivalence_l240_24030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l240_24084

-- Define the vectors
noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.sin x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, Real.sin x)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- State the theorem
theorem triangle_area (ABC : Triangle) (h1 : f ABC.A = 3/2) (h2 : ABC.b + ABC.c = 4) (h3 : ABC.a = Real.sqrt 7) :
  (1/2) * ABC.b * ABC.c * Real.sin ABC.A = (3 * Real.sqrt 3) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l240_24084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_sum_l240_24087

theorem max_value_and_sum (x y z v w : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) (pos_v : v > 0) (pos_w : w > 0)
  (sum_squares : x^2 + y^2 + z^2 + v^2 + w^2 = 2025) :
  let expr := x*z + 2*y*z + 4*z*v + 8*z*w
  ∃ (x_M y_M z_M v_M w_M : ℝ),
    (∀ a b c d e : ℝ, a > 0 → b > 0 → c > 0 → d > 0 → e > 0 → 
      a^2 + b^2 + c^2 + d^2 + e^2 = 2025 → 
      a*c + 2*b*c + 4*c*d + 8*c*e ≤ expr) ∧
    expr = 3075 * Real.sqrt 17 ∧
    expr + x_M + y_M + z_M + v_M + w_M = 3075 * Real.sqrt 17 + 5 * Real.sqrt 1012.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_sum_l240_24087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l240_24019

-- Define the sets P and Q
def P : Finset ℕ := {1, 3, 4}
def Q : Finset ℕ := {1, 2, 3, 6}

-- Define the universal set U
def U : Finset ℕ := P ∪ Q

-- Theorem statement
theorem problem_solution :
  (P ∩ Q = {1, 3}) ∧ (Finset.card U = 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l240_24019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_immersion_water_rise_l240_24072

/-- Represents the rise in water level when a cube is immersed in a rectangular vessel. -/
noncomputable def water_rise (cube_edge : ℝ) (vessel_length : ℝ) (vessel_width : ℝ) : ℝ :=
  (cube_edge ^ 3) / (vessel_length * vessel_width)

/-- Theorem stating that a 12 cm cube immersed in a 20 cm × 15 cm vessel raises the water level by 5.76 cm. -/
theorem cube_immersion_water_rise :
  water_rise 12 20 15 = 5.76 := by
  -- Expand the definition of water_rise
  unfold water_rise
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_immersion_water_rise_l240_24072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_multiples_of_seven_l240_24017

theorem three_digit_multiples_of_seven : 
  (Finset.filter (fun n => n % 7 = 0) (Finset.range 900)).card = 128 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_multiples_of_seven_l240_24017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l240_24094

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

noncomputable def sum_of_arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (a₁ + arithmetic_sequence a₁ d n) / 2

theorem arithmetic_sequence_properties :
  let a₁ : ℝ := 20
  let d : ℝ := -2
  ∀ n : ℕ+,
    (arithmetic_sequence a₁ d n = -2 * n + 22) ∧
    (∃ max_sum : ℝ, max_sum = 110 ∧
      ∀ k : ℕ+, sum_of_arithmetic_sequence a₁ d k ≤ max_sum) := by
  sorry

#check arithmetic_sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l240_24094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_minus_cot_value_l240_24079

theorem tan_minus_cot_value (θ : Real) 
  (h1 : Real.sin θ + Real.cos θ = Real.sqrt 2 / 3)
  (h2 : π / 2 < θ)
  (h3 : θ < π) : 
  Real.tan θ - (1 / Real.tan θ) = -8 * Real.sqrt 2 / 7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_minus_cot_value_l240_24079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l240_24046

-- Define the line segments using real numbers
def segment1 : (ℝ × ℝ) × (ℝ × ℝ) := ((-4, -5), (-2, 0))
def segment2 : (ℝ × ℝ) × (ℝ × ℝ) := ((-2, 0), (-1, -1))
def segment3 : (ℝ × ℝ) × (ℝ × ℝ) := ((-1, -1), (1, 3))
def segment4 : (ℝ × ℝ) × (ℝ × ℝ) := ((1, 3), (2, 2))
def segment5 : (ℝ × ℝ) × (ℝ × ℝ) := ((2, 2), (4, 6))

-- Define the y-coordinate of the intersection line
def y_intersect : ℝ := 1.5

-- Function to calculate the x-coordinate of intersection
noncomputable def intersect_x (seg : (ℝ × ℝ) × (ℝ × ℝ)) (y : ℝ) : ℝ :=
  let ((x1, y1), (x2, y2)) := seg
  let m := (y2 - y1) / (x2 - x1)
  (y - y1) / m + x1

-- Theorem statement
theorem intersection_sum :
  (intersect_x segment1 y_intersect) +
  (intersect_x segment2 y_intersect) +
  (intersect_x segment3 y_intersect) +
  (intersect_x segment4 y_intersect) +
  (intersect_x segment5 y_intersect) = 2.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l240_24046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l240_24056

/-- The function f(x) = x(|x|-2) -/
noncomputable def f (x : ℝ) : ℝ := x * (|x| - 2)

/-- The function g(x) = 4x/(x+1) -/
noncomputable def g (x : ℝ) : ℝ := 4 * x / (x + 1)

/-- The theorem statement -/
theorem function_inequality (a : ℝ) : 
  (∀ x₁ ∈ Set.Ioo (-1 : ℝ) a, ∃ x₂ ∈ Set.Ioo (-1 : ℝ) a, f x₁ ≤ g x₂) ↔ 
  a ∈ Set.Icc (1/3 : ℝ) 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l240_24056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_limits_l240_24035

/-- An isosceles triangle with base b, equal sides s, and height h -/
structure IsoscelesTriangle where
  b : ℝ
  s : ℝ
  h : ℝ
  b_pos : 0 < b
  s_pos : 0 < s
  h_pos : 0 < h

/-- The sequence of perimeters of triangles formed by repeatedly joining midpoints -/
noncomputable def perimeterSequence (t : IsoscelesTriangle) : ℕ → ℝ
  | 0 => t.b + 2 * t.s
  | n + 1 => (t.b + 2 * t.s) / (2^(n + 1))

/-- The sequence of areas of triangles formed by repeatedly joining midpoints -/
noncomputable def areaSequence (t : IsoscelesTriangle) : ℕ → ℝ
  | 0 => (1/2) * t.b * t.h
  | n + 1 => ((1/2) * t.b * t.h) / (4^(n + 1))

/-- The theorem stating the limits of the sums of perimeters and areas -/
theorem isosceles_triangle_limits (t : IsoscelesTriangle) : 
  (∑' n, perimeterSequence t n) = 2 * t.b + 4 * t.s ∧ 
  (∑' n, areaSequence t n) = (2/3) * t.b * t.h := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_limits_l240_24035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chicken_feed_cost_l240_24043

/-- Represents the cost and profit structure of a chicken farm --/
structure ChickenFarm where
  chicken_price : ℚ
  feed_bag_weight : ℚ
  feed_per_chicken : ℚ
  num_chickens : ℕ
  profit : ℚ

/-- Calculates the cost of a bag of chicken feed --/
noncomputable def bag_cost (farm : ChickenFarm) : ℚ :=
  let total_revenue := farm.chicken_price * farm.num_chickens
  let total_feed_weight := farm.feed_per_chicken * farm.num_chickens
  let num_bags := total_feed_weight / farm.feed_bag_weight
  let total_feed_cost := total_revenue - farm.profit
  total_feed_cost / num_bags

/-- Theorem stating that the cost of a bag of chicken feed is $2 --/
theorem chicken_feed_cost (farm : ChickenFarm) 
  (h1 : farm.chicken_price = 3/2)
  (h2 : farm.feed_bag_weight = 20)
  (h3 : farm.feed_per_chicken = 2)
  (h4 : farm.num_chickens = 50)
  (h5 : farm.profit = 65) :
  bag_cost farm = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chicken_feed_cost_l240_24043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deck_size_rank_count_suit_count_draw_is_unfair_l240_24076

/-- Represents the rank of a card -/
inductive Rank
| Six | Seven | Eight | Nine | Ten | Jack | Queen | King | Ace
deriving Fintype, Repr

/-- Represents the suit of a card -/
inductive Suit
| Hearts | Diamonds | Clubs | Spades
deriving Fintype, Repr

/-- Represents a playing card -/
structure Card where
  rank : Rank
  suit : Suit
deriving Fintype, Repr

/-- The deck of cards -/
def deck : Finset Card := Finset.univ

/-- The number of cards in the deck -/
theorem deck_size : deck.card = 36 := by
  simp [deck]
  rfl

/-- The number of ranks -/
theorem rank_count : Fintype.card Rank = 9 := by
  simp [Fintype.card]
  rfl

/-- The number of suits -/
theorem suit_count : Fintype.card Suit = 4 := by
  simp [Fintype.card]
  rfl

/-- Volodya's probability of winning -/
noncomputable def volodya_win_prob : ℝ := sorry

/-- Masha's probability of winning -/
noncomputable def masha_win_prob : ℝ := sorry

/-- The draw is unfair (biased towards Volodya) -/
theorem draw_is_unfair : volodya_win_prob > masha_win_prob := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_deck_size_rank_count_suit_count_draw_is_unfair_l240_24076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_edges_sum_is_eight_l240_24083

/-- Represents a rectangular picture frame -/
structure Frame where
  width : ℝ  -- Width of the wood used for the frame
  outer_edge : ℝ  -- Length of one outer edge
  frame_area : ℝ  -- Area of the frame

/-- Calculates the sum of the lengths of the four interior edges of the frame -/
noncomputable def interior_edges_sum (f : Frame) : ℝ :=
  let interior_width := f.outer_edge - 2 * f.width
  let interior_height := (f.frame_area - (f.outer_edge * f.width - interior_width * f.width)) / (2 * f.width)
  2 * (interior_width + interior_height)

/-- Theorem stating that for a frame with given properties, the sum of interior edges is 8 inches -/
theorem interior_edges_sum_is_eight (f : Frame) 
  (h1 : f.width = 2) 
  (h2 : f.outer_edge = 7) 
  (h3 : f.frame_area = 32) : 
  interior_edges_sum f = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_edges_sum_is_eight_l240_24083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_maximizes_chord_length_l240_24038

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y + 3)^2 = 3

-- Define the point M
def M : ℝ × ℝ := (1, 2)

-- Define the center of the circle
def center : ℝ × ℝ := (2, -3)

-- Define the line passing through M and the center
def line_equation (x y : ℝ) : Prop := 5*x + y - 7 = 0

-- Theorem statement
theorem line_maximizes_chord_length :
  (∀ x y, line_equation x y → (x, y) ≠ M → (x, y) ≠ center → circle_equation x y) ∧
  (∀ k : ℝ, k ≠ -5 → ∃ x y, y - 2 = k*(x - 1) ∧ (x, y) ≠ M ∧ (x, y) ≠ center ∧ circle_equation x y ∧
    ∃ x' y', y' - 2 = k*(x' - 1) ∧ (x', y') ≠ (x, y) ∧ circle_equation x' y' ∧
    (x' - x)^2 + (y' - y)^2 < (center.1 - M.1)^2 + (center.2 - M.2)^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_maximizes_chord_length_l240_24038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_of_f_l240_24075

noncomputable def f (x : ℝ) := |Real.tan (2 * x)|

theorem min_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_of_f_l240_24075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jordan_rice_purchase_l240_24045

theorem jordan_rice_purchase 
  (rice_price : ℝ) 
  (lentil_price : ℝ) 
  (total_pounds : ℝ) 
  (total_cost : ℝ) 
  (h1 : rice_price = 1.10)
  (h2 : lentil_price = 0.55)
  (h3 : total_pounds = 30)
  (h4 : total_cost = 22.65) :
  ∃ (rice_pounds : ℝ), 
    rice_pounds * rice_price + (total_pounds - rice_pounds) * lentil_price = total_cost ∧ 
    |rice_pounds - 11.2| < 0.05 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jordan_rice_purchase_l240_24045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_is_systematic_l240_24011

/-- Represents a sampling method -/
inductive SamplingMethod
| Stratified
| Lottery
| Random
| Systematic

/-- Represents a class in the freshman year -/
structure FreshmanClass where
  students : Fin 56

/-- Represents the freshman year -/
def FreshmanYear := Fin 35 → FreshmanClass

/-- The sampling function that selects student number 14 from each class -/
def samplingFunction (year : FreshmanYear) : Fin 35 → Fin 56 :=
  λ _ ↦ 14

/-- Theorem stating that the given sampling method is systematic sampling -/
theorem sampling_is_systematic (year : FreshmanYear) :
  ∃ (method : SamplingMethod), method = SamplingMethod.Systematic ∧
  (∀ (c : Fin 35), (samplingFunction year c) = 14) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_is_systematic_l240_24011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_flavors_l240_24026

/-- Theorem: The number of ways to distribute n indistinguishable objects into k+1 distinguishable boxes
    is equal to the number of ways to choose k positions from n+k positions. -/
theorem ice_cream_flavors (n : ℕ) (k : ℕ) :
  Nat.choose (n + k) k = Nat.choose (n + k) k :=
by
  -- The proof is trivial as we're asserting equality with itself
  rfl

/-- Verification for the specific case of 5 scoops and 3 flavors (2 dividers) -/
example : Nat.choose (5 + 2) 2 = 21 :=
by
  -- Evaluate the left-hand side
  rfl

#eval Nat.choose (5 + 2) 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_flavors_l240_24026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_one_multiple_of_4_l240_24005

def range : Finset ℕ := Finset.filter (λ n => 1 ≤ n ∧ n ≤ 60) (Finset.range 61)

def multiples_of_4 : Finset ℕ := Finset.filter (λ n => n % 4 = 0) range

def prob_not_multiple_of_4 : ℚ := (range.card - multiples_of_4.card : ℚ) / (range.card : ℚ)

theorem probability_at_least_one_multiple_of_4 :
  1 - prob_not_multiple_of_4^3 = 37/64 := by
  sorry

#eval prob_not_multiple_of_4
#eval 1 - prob_not_multiple_of_4^3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_one_multiple_of_4_l240_24005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_degree_for_horizontal_asymptote_l240_24090

/-- The numerator of the rational function -/
noncomputable def numerator (x : ℝ) : ℝ := 3*x^8 - 2*x^3 + x - 5

/-- The rational function -/
noncomputable def rationalFunction (p : ℝ → ℝ) (x : ℝ) : ℝ := numerator x / p x

/-- A function has a horizontal asymptote if it converges to a finite value as x approaches infinity -/
def hasHorizontalAsymptote (f : ℝ → ℝ) : Prop :=
  ∃ L : ℝ, ∀ ε > 0, ∃ M : ℝ, ∀ x > M, |f x - L| < ε

/-- The degree of a polynomial -/
def degree (p : ℝ → ℝ) : ℕ := sorry

/-- The main theorem -/
theorem smallest_degree_for_horizontal_asymptote :
  ∀ p : ℝ → ℝ, hasHorizontalAsymptote (rationalFunction p) →
  ∃ q : ℝ → ℝ, degree q = 8 ∧ hasHorizontalAsymptote (rationalFunction q) ∧
  ∀ r : ℝ → ℝ, degree r < 8 → ¬hasHorizontalAsymptote (rationalFunction r) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_degree_for_horizontal_asymptote_l240_24090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empirical_regression_properties_l240_24006

/-- Sample data point -/
structure SamplePoint where
  x : ℝ
  y : ℝ

/-- Linear function -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Calculate the residual for a sample point given a linear function -/
def calculateResidual (point : SamplePoint) (f : LinearFunction) : ℝ :=
  point.y - (f.slope * point.x + f.intercept)

/-- Sum of squared residuals for a list of sample points and a linear function -/
def sumSquaredResiduals (points : List SamplePoint) (f : LinearFunction) : ℝ :=
  (points.map (λ p => (calculateResidual p f)^2)).sum

/-- Empirical regression equation obtained by the method of least squares -/
def empiricalRegression : LinearFunction :=
  { slope := 1.6, intercept := -0.5 }  -- We now know the value of 'a' is -0.5

/-- Sample data points -/
def sampleData : List SamplePoint :=
  [{ x := 1, y := 1 }, { x := 2, y := 3 }, { x := 2.5, y := 3.5 }, { x := 3, y := 4 }, { x := 4, y := 6 }]

theorem empirical_regression_properties :
  (empiricalRegression.slope > 0) ∧
  (∀ (otherFunction : LinearFunction),
    sumSquaredResiduals sampleData empiricalRegression ≤ sumSquaredResiduals sampleData otherFunction) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_empirical_regression_properties_l240_24006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_d_coordinates_l240_24099

/-- A quadrilateral in a 2D coordinate system -/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Two line segments are parallel -/
def parallel (seg1 : (ℝ × ℝ) × (ℝ × ℝ)) (seg2 : (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  let (a1, a2) := seg1
  let (b1, b2) := seg2
  (a2.1 - a1.1) * (b2.2 - b1.2) = (a2.2 - a1.2) * (b2.1 - b1.1)

/-- The length of a line segment -/
noncomputable def segmentLength (seg : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ :=
  let (a, b) := seg
  Real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

/-- The main theorem -/
theorem quadrilateral_d_coordinates (q : Quadrilateral) :
  q.A = (1, 0) →
  q.B = (3, 0) →
  q.C = (2, 2) →
  parallel (q.A, q.B) (q.C, q.D) →
  segmentLength (q.A, q.B) = segmentLength (q.C, q.D) →
  q.D = (0, 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_d_coordinates_l240_24099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_binary_div_8_l240_24032

/-- The binary representation of the number --/
def binary_num : List Bool := [true, false, true, false, true, true, true, false, false, true, false, true]

/-- Convert a binary number to decimal --/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Get the last n bits of a binary number --/
def last_n_bits (bits : List Bool) (n : ℕ) : List Bool :=
  bits.reverse.take n

theorem remainder_of_binary_div_8 :
  binary_to_decimal (last_n_bits binary_num 3) % 8 = 5 := by
  sorry

#eval binary_to_decimal (last_n_bits binary_num 3) % 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_binary_div_8_l240_24032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_term_correct_l240_24048

def sequence_term (n : ℕ) : ℚ := 2 / (n * (n + 1))

theorem sequence_term_correct (a : ℕ → ℚ) (S : ℕ → ℚ) (h : ∀ n, S n - n^2 * a n = S 1 - a 1) :
  a 1 = 1 → (∀ n, a n = sequence_term n) := by
  intro h1
  sorry

#check sequence_term_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_term_correct_l240_24048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l240_24096

theorem evaluate_expression : 
  ∀ (x : ℝ), x > 0 → (3 * x^(1/4 : ℝ)) / x^(1/3 : ℝ) = 3 * x^(-(1/12 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l240_24096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carousel_horses_count_l240_24062

theorem carousel_horses_count (blue purple green gold : ℕ) :
  blue = 3 →
  purple = 3 * blue →
  green = 2 * purple →
  gold = green / 6 →
  blue + purple + green + gold = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carousel_horses_count_l240_24062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_circle_intersection_angle_sum_l240_24078

-- Define a circle with a center and radius
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on a plane
def Point : Type := ℝ × ℝ

-- Define the central angle of an arc on a circle
noncomputable def centralAngle (c : Circle) (p q : Point) : ℝ := sorry

-- Define a predicate for a point being on a circle
def onCircle (p : Point) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define the theorem
theorem three_circle_intersection_angle_sum 
  (O₁ O₂ O₃ : Circle) 
  (A B C D E F : Point) :
  O₁.radius = O₂.radius ∧ O₂.radius = O₃.radius →
  (onCircle A O₁ ∧ onCircle A O₂) →
  (onCircle B O₁ ∧ onCircle B O₃) →
  (onCircle C O₂ ∧ onCircle C O₃) →
  (onCircle D O₁ ∧ onCircle D O₂) →
  (onCircle E O₂ ∧ onCircle E O₃) →
  (onCircle F O₁ ∧ onCircle F O₃) →
  centralAngle O₁ A B + centralAngle O₂ C D + centralAngle O₃ E F = 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_circle_intersection_angle_sum_l240_24078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_value_f_leq_g_implies_a_range_l240_24077

open Real

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := log x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x + a / x
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := f x + g a x - x

-- Part 1
theorem min_value_implies_a_value (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (exp 1), F a x ≥ 3/2) ∧ 
  (∃ x ∈ Set.Icc 1 (exp 1), F a x = 3/2) →
  a = sqrt (exp 1) := by
sorry

-- Part 2
theorem f_leq_g_implies_a_range (a : ℝ) :
  (∀ x ≥ 1, f x ≤ g a x) →
  a ≥ -1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_value_f_leq_g_implies_a_range_l240_24077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_athlete_triple_jump_l240_24010

/-- Represents an athlete's jump performance -/
structure AthleteJumps where
  long_jump : ℝ
  triple_jump : ℝ
  high_jump : ℝ

/-- Calculates the average jump distance for an athlete -/
noncomputable def average_jump (jumps : AthleteJumps) : ℝ :=
  (jumps.long_jump + jumps.triple_jump + jumps.high_jump) / 3

/-- The main theorem to prove -/
theorem second_athlete_triple_jump
  (first_athlete : AthleteJumps)
  (second_athlete : AthleteJumps)
  (h1 : first_athlete.long_jump = 26)
  (h2 : first_athlete.triple_jump = 30)
  (h3 : first_athlete.high_jump = 7)
  (h4 : second_athlete.long_jump = 24)
  (h5 : second_athlete.high_jump = 8)
  (h6 : max (average_jump first_athlete) (average_jump second_athlete) = 22) :
  second_athlete.triple_jump = 34 := by
  sorry

#check second_athlete_triple_jump

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_athlete_triple_jump_l240_24010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_N2O3_approx_l240_24097

/-- The molar mass of nitrogen in g/mol -/
noncomputable def molar_mass_N : ℝ := 14.01

/-- The molar mass of oxygen in g/mol -/
noncomputable def molar_mass_O : ℝ := 16.00

/-- The number of nitrogen atoms in N2O3 -/
def num_N : ℕ := 2

/-- The number of oxygen atoms in N2O3 -/
def num_O : ℕ := 3

/-- The molar mass of N2O3 in g/mol -/
noncomputable def molar_mass_N2O3 : ℝ := num_N * molar_mass_N + num_O * molar_mass_O

/-- The mass of oxygen in N2O3 in g/mol -/
noncomputable def mass_O_in_N2O3 : ℝ := num_O * molar_mass_O

/-- The mass percentage of oxygen in N2O3 -/
noncomputable def mass_percentage_O_in_N2O3 : ℝ := (mass_O_in_N2O3 / molar_mass_N2O3) * 100

/-- Theorem: The mass percentage of oxygen in N2O3 is approximately 63.15% -/
theorem mass_percentage_O_in_N2O3_approx :
  |mass_percentage_O_in_N2O3 - 63.15| < 0.01 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_N2O3_approx_l240_24097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_configuration_exists_l240_24065

/-- Represents the state of a lamp (on or off) -/
inductive LampState
| On : LampState
| Off : LampState

/-- Represents a configuration of lamps -/
def LampConfiguration := List LampState

/-- Function to update the lamp configuration according to the rules -/
def updateConfiguration (config : LampConfiguration) : LampConfiguration :=
  sorry

/-- Predicate to check if a configuration has at least one lamp on -/
def hasLampOn (config : LampConfiguration) : Prop :=
  sorry

/-- Theorem stating that for all n except 1 and 3, there exists a valid initial configuration -/
theorem valid_configuration_exists (n : Nat) : 
  n ≠ 1 ∧ n ≠ 3 → 
  ∃ (initial : LampConfiguration), 
    initial.length = n ∧ 
    ∀ (t : Nat), hasLampOn (Nat.rec initial (fun _ => updateConfiguration) t) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_configuration_exists_l240_24065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_division_areas_l240_24008

/-- Given a parallelogram ABCD with area 1, sides AB and CD divided into n equal parts,
    and sides AD and BC divided into m equal parts. -/
def Parallelogram (m n : ℕ) : Prop :=
  ∃ (ABCD : Set (ℝ × ℝ)) (area : Set (ℝ × ℝ) → ℝ),
    area ABCD = 1 ∧
    (∃ (divAB divCD : Fin n → Set (ℝ × ℝ)) (divAD divBC : Fin m → Set (ℝ × ℝ)),
      (∀ i, divAB i ⊆ ABCD ∧ divCD i ⊆ ABCD) ∧
      (∀ j, divAD j ⊆ ABCD ∧ divBC j ⊆ ABCD))

/-- The area of each small parallelogram in configuration (a) -/
noncomputable def AreaConfigA (m n : ℕ) : ℝ := 1 / (m * n + 1)

/-- The area of each small parallelogram in configuration (b) -/
noncomputable def AreaConfigB (m n : ℕ) : ℝ := 1 / (m * n - 1)

/-- Theorem stating the areas of small parallelograms in both configurations -/
theorem parallelogram_division_areas (m n : ℕ) (h : m * n > 1) :
  Parallelogram m n →
  ∃ (areaA areaB : Set (ℝ × ℝ) → ℝ),
    (∀ smallA, areaA smallA = AreaConfigA m n) ∧
    (∀ smallB, areaB smallB = AreaConfigB m n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_division_areas_l240_24008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_beta_max_value_l240_24012

open Real

theorem tan_beta_max_value (α β : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : β ∈ Set.Ioo 0 (π/2)) 
  (h3 : sin (2*α + β) = 2 * sin β) : 
  ∃ (β_max : ℝ), β_max ∈ Set.Ioo 0 (π/2) ∧ 
    (∀ β' ∈ Set.Ioo 0 (π/2), tan β' ≤ tan β_max) ∧
    tan β_max = sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_beta_max_value_l240_24012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_Y_l240_24029

/-- Represents a right circular cylinder -/
structure Cylinder where
  height : ℝ
  radius : ℝ

/-- Calculates the volume of a cylinder -/
noncomputable def volume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

/-- Represents the oil cans X, Y, and Z -/
def oil_cans : Fin 3 → Cylinder
  | 0 => ⟨1, 1⟩  -- Can X (arbitrary height and radius)
  | 1 => ⟨4, 4⟩  -- Can Y (4 times height and radius of X)
  | 2 => ⟨2, 2⟩  -- Can Z (2 times height and radius of X)

/-- Price of oil in can X -/
def price_X : ℝ := 2

/-- Price of oil in can Z -/
def price_Z : ℝ := 5

/-- Calculates the price of oil for a given volume -/
noncomputable def oil_price (v : ℝ) : ℝ := price_X * v / volume (oil_cans 0)

theorem oil_price_Y (h : ℝ) : 
  oil_price (volume ⟨h, (oil_cans 1).radius⟩) = 64 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_Y_l240_24029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bag_of_chips_cost_l240_24073

/-- Represents the cost of chips in dollars -/
def ChipsCost (cost : ℝ) : Prop := cost ≥ 0

/-- The number of calories in one chip -/
def calories_per_chip : ℕ := 10

/-- The number of chips in one bag -/
def chips_per_bag : ℕ := 24

/-- The total calories Peter wants to eat -/
def total_calories : ℕ := 480

/-- The amount Peter spends to eat the total calories -/
def total_spent : ℝ := 4

theorem bag_of_chips_cost :
  ChipsCost 2 := by
  -- Proof goes here
  sorry

#check bag_of_chips_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bag_of_chips_cost_l240_24073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l240_24049

theorem problem_statement (a b : ℕ) (h : (18 : ℕ) ^ a * 9 ^ (3 * a - 1) = (2 : ℕ) ^ 7 * 3 ^ b) : a = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l240_24049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l240_24001

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  let E := {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1}
  ∃ (A B : ℝ × ℝ), A ∈ E ∧ B ∈ E ∧ 
    (∃ (k : ℝ), k = Real.tan (135 : ℝ) * π / 180 ∧ (B.2 - A.2) = k * (B.1 - A.1)) ∧
    ((A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = 1) →
  let c := Real.sqrt (a^2 - b^2)
  let e := c / a
  e = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l240_24001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_intercepts_not_in_third_quadrant_l240_24004

/-- The equation of line l is (a+1)x + y - 3 + a = 0, where a ∈ ℝ -/
def line_equation (a x y : ℝ) : Prop := (a + 1) * x + y - 3 + a = 0

/-- The x-intercept of line l -/
noncomputable def x_intercept (a : ℝ) : ℝ := (3 - a) / (a + 1)

/-- The y-intercept of line l -/
def y_intercept (a : ℝ) : ℝ := 3 - a

/-- Theorem: The intercepts of l on the two coordinate axes are equal if and only if a = 0 or a = 3 -/
theorem equal_intercepts (a : ℝ) : 
  (x_intercept a = y_intercept a) ↔ (a = 0 ∨ a = 3) := by
  sorry

/-- Theorem: Line l does not pass through the third quadrant if and only if -1 ≤ a ≤ 3 -/
theorem not_in_third_quadrant (a : ℝ) :
  (∀ x y : ℝ, line_equation a x y → (x < 0 ∧ y < 0 → False)) ↔ (-1 ≤ a ∧ a ≤ 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_intercepts_not_in_third_quadrant_l240_24004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_abc_l240_24068

/-- Triangle ABC with side AB = 16 and ratio BC:AC = 3:4 -/
structure Triangle where
  BC : ℝ
  AC : ℝ
  h_ratio : BC / AC = 3 / 4
  h_positive : BC > 0 ∧ AC > 0

/-- The area of a triangle given its side lengths -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- The area of our specific triangle -/
noncomputable def triangleABC (t : Triangle) : ℝ :=
  triangleArea 16 t.BC t.AC

theorem max_area_triangle_abc :
  ∃ (t : Triangle), ∀ (t' : Triangle), triangleABC t ≥ triangleABC t' ∧
  triangleABC t = 128 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_abc_l240_24068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l240_24039

theorem ellipse_properties (a b : ℝ) (k m lambda : ℝ) :
  a > b ∧ b > 0 →
  ((-1)^2 / a^2) + ((Real.sqrt 2 / 2)^2 / b^2) = 1 →
  Real.sqrt (a^2 - b^2) / a = Real.sqrt 2 / 2 →
  m^2 = k^2 + 1 →
  (1 + 2*k^2) * (-1)^2 + 4*k*m*(-1) + 2*m^2 - 2 > 0 →
  (2/3 : ℝ) ≤ lambda ∧ lambda ≤ (3/4 : ℝ) →
  lambda = (1 + k^2) / (1 + 2*k^2) →
  (∃ (S : ℝ), (Real.sqrt 6 / 4 ≤ S ∧ S ≤ 2/3) ∧
    S = Real.sqrt ((2*(k^4 + k^2)) / (4*(k^4 + k^2) + 1))) ∧
  a^2 = 2 ∧ b^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l240_24039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_monotonicity_of_f_range_of_a_l240_24000

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - 1 - a * Real.log x

-- Statement 1: Tangent line equation when a = 2
theorem tangent_line_at_one (x y : ℝ) :
  HasDerivAt (f 2) (-1) 1 → x + y - 1 = 0 := by sorry

-- Statement 2: Monotonicity of f
theorem monotonicity_of_f (a : ℝ) :
  (a ≤ 0 → ∀ x > 0, MonotoneOn (f a) (Set.Ioi 0)) ∧
  (a > 0 → ∀ x, x ∈ Set.Ioo 0 a → StrictMonoOn (f a) (Set.Ioo 0 a)) ∧
  (a > 0 → ∀ x, x ∈ Set.Ioi a → MonotoneOn (f a) (Set.Ioi a)) := by sorry

-- Statement 3: Range of a for the given condition
theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂, x₁ ∈ Set.Ioc 0 1 → x₂ ∈ Set.Ioc 0 1 → 
    |f a x₁ - f a x₂| ≤ 4 * |1/x₁ - 1/x₂|) ↔
  -3 ≤ a ∧ a < 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_monotonicity_of_f_range_of_a_l240_24000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_equidistant_point_plane_l240_24095

/-- The locus of points equidistant from a point and a plane -/
theorem locus_equidistant_point_plane (x y z : ℝ) :
  let A : ℝ × ℝ × ℝ := (-2, 0, 0)
  let P : ℝ × ℝ × ℝ := (x, y, z)
  let dist_to_A := Real.sqrt ((x + 2)^2 + y^2 + z^2)
  let dist_to_plane := |x - 2|
  dist_to_A = dist_to_plane → x = -(y^2 + z^2) / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_equidistant_point_plane_l240_24095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_solution_eq_211_935_l240_24003

noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

theorem largest_solution_eq_211_935 :
  ∃ (x : ℝ), ⌊x⌋ = 12 + 200 * (frac x) ∧
  ∀ (y : ℝ), ⌊y⌋ = 12 + 200 * (frac y) → y ≤ x ∧
  x = 211.935 := by
  sorry

#check largest_solution_eq_211_935

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_solution_eq_211_935_l240_24003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l240_24025

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 9 = 0

-- Define the line
def line_eq (m x y : ℝ) : Prop := m*x + y + m - 2 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, 0)

-- Define the radius of the circle
noncomputable def circle_radius : ℝ := Real.sqrt 10

-- Define the point M that the line passes through
def point_M : ℝ × ℝ := (-1, 2)

-- Theorem statement
theorem line_intersects_circle (m : ℝ) : 
  ∃ (x y : ℝ), circle_eq x y ∧ line_eq m x y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l240_24025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_to_equation_unique_solutions_l240_24040

/-- The function representing the left-hand side of the equation -/
noncomputable def f (x : ℝ) : ℝ :=
  5 / (Real.sqrt (x - 9) - 8) - 2 / (Real.sqrt (x - 9) - 5) + 
  6 / (Real.sqrt (x - 9) + 5) - 9 / (Real.sqrt (x - 9) + 8)

/-- The first solution to the equation -/
def x₁ : ℝ := 19.2917

/-- The second solution to the equation -/
def x₂ : ℝ := 8.9167

/-- Theorem stating that x₁ and x₂ are solutions to the equation -/
theorem solutions_to_equation : f x₁ = 0 ∧ f x₂ = 0 := by
  sorry

/-- Theorem stating that x₁ and x₂ are the only solutions to the equation -/
theorem unique_solutions : ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_to_equation_unique_solutions_l240_24040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_distance_centers_value_l240_24085

/-- A right triangle with legs 3 and 4 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  is_right : a = 3 ∧ b = 4

/-- The square of the distance between the centers of inscribed and circumscribed circles -/
noncomputable def square_distance_centers (t : RightTriangle) : ℝ := 
  let c := Real.sqrt (t.a ^ 2 + t.b ^ 2)
  let s := (t.a + t.b + c) / 2
  let r := (t.a * t.b) / (2 * s)
  (r - c / 2) ^ 2 + r ^ 2

/-- Theorem: The square of the distance between the centers of inscribed and circumscribed circles is 1.25 -/
theorem square_distance_centers_value (t : RightTriangle) : 
  square_distance_centers t = 1.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_distance_centers_value_l240_24085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l240_24020

-- Define the power function
noncomputable def powerFunction (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

-- State the theorem
theorem power_function_through_point (f : ℝ → ℝ) :
  (∃ α : ℝ, f = powerFunction α) →  -- f is a power function
  f 9 = 1/3 →                       -- f passes through (9, 1/3)
  f 25 = 1/5 :=                     -- f(25) = 1/5
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l240_24020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l240_24052

/-- Given a hyperbola and a parabola with specific properties, prove the equation of the hyperbola. -/
theorem hyperbola_equation :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) ∧  -- Hyperbola equation
  (a / Real.sqrt (a^2 - b^2) = Real.sqrt 3) ∧  -- Eccentricity is √3
  (∃ (x y : ℝ), y^2 = 12 * x) ∧  -- Parabola equation
  (∃ (f : ℝ × ℝ), f.1 = 3 ∧ f.2 = 0) →  -- Focus of parabola at (3, 0)
  (∃ (x y : ℝ), x^2 / 3 - y^2 / 6 = 1) :=  -- Conclusion: equation of hyperbola
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l240_24052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_divisibility_by_11_l240_24024

def u : ℕ → ℤ
  | 0 => 1  -- We need to define u for 0 to cover all natural numbers
  | 1 => 1
  | 2 => 3
  | n+3 => (n+4) * u (n+2) - (n+3) * u (n+1)

theorem u_divisibility_by_11 (n : ℕ) :
  11 ∣ u n ↔ (n ≥ 10 ∧ n ∉ ({1, 2, 3, 5, 6, 7, 9} : Finset ℕ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_divisibility_by_11_l240_24024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_odd_coefficients_sum_even_coefficients_l240_24091

-- Define the polynomial coefficients
def a : Fin 12 → ℤ := sorry

-- Define the equation
def equation (x : ℝ) : Prop :=
  (1 + x)^6 * (1 - 2*x)^5 = (Finset.range 12).sum (λ i => a i * x^(i : ℕ))

-- Theorem for the first part
theorem sum_odd_coefficients : 
  equation 1 → equation 0 → (Finset.range 11).sum (λ i => a (i + 1)) = -65 := by sorry

-- Theorem for the second part
theorem sum_even_coefficients : 
  equation 1 → equation (-1) → (Finset.range 6).sum (λ i => a (2 * i)) = -32 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_odd_coefficients_sum_even_coefficients_l240_24091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_mul_comm_custom_mul_assoc_l240_24093

/-- The custom multiplication operation for positive real numbers -/
noncomputable def custom_mul (x y : ℝ) : ℝ := (x * y) / (x + y)

/-- Commutativity of the custom multiplication operation -/
theorem custom_mul_comm (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  custom_mul x y = custom_mul y x := by
  -- Proof goes here
  sorry

/-- Associativity of the custom multiplication operation -/
theorem custom_mul_assoc (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  custom_mul (custom_mul x y) z = custom_mul x (custom_mul y z) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_mul_comm_custom_mul_assoc_l240_24093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_average_speed_l240_24082

/-- Calculates the average speed of a round trip with a pause in between -/
noncomputable def average_speed (distance_one_way : ℝ) (time_to : ℝ) (time_pause : ℝ) (time_from : ℝ) : ℝ :=
  (2 * distance_one_way) / (time_to + time_pause + time_from)

theorem johns_average_speed :
  let distance_to_library : ℝ := 2  -- in km
  let time_to_library : ℝ := 40/60  -- in hours
  let time_reading : ℝ := 20/60     -- in hours
  let time_from_library : ℝ := 20/60  -- in hours
  average_speed distance_to_library time_to_library time_reading time_from_library = 3 := by
  sorry

#check johns_average_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_average_speed_l240_24082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_rental_miles_driven_proof_l240_24022

/-- Calculates the number of miles driven given car rental conditions --/
def car_rental_miles_driven 
  (daily_rate : ℝ) 
  (mileage_charge : ℝ) 
  (discount_rate : ℝ) 
  (insurance_rate : ℝ) 
  (rental_days : ℕ) 
  (total_invoice : ℝ) : ℕ :=
  517

/-- Proof of the theorem --/
theorem car_rental_miles_driven_proof : 
  ∃ (miles : ℕ), car_rental_miles_driven 35 0.09 0.1 5 4 192.50 = miles ∧ miles = 517 := by
  use 517
  constructor
  · rfl
  · rfl

#eval car_rental_miles_driven 35 0.09 0.1 5 4 192.50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_rental_miles_driven_proof_l240_24022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_implies_sum_of_powers_l240_24053

noncomputable def f (a b x : ℝ) : ℝ := (x + a) / (x + b)

theorem function_property_implies_sum_of_powers (a b : ℝ) :
  (∀ x : ℝ, x ≠ -b → f a b x + f a b (1/x) = 0) →
  a^2018 + b^2018 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_implies_sum_of_powers_l240_24053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_graph_most_suitable_for_height_change_l240_24023

/-- Represents different types of graphs --/
inductive GraphType
  | BarGraph
  | LineGraph
  | PieChart

/-- Characteristics of different graph types --/
def graphCharacteristics (g : GraphType) : String :=
  match g with
  | .BarGraph => "compares quantities among different groups at a specific time point"
  | .LineGraph => "shows trends or changes over time"
  | .PieChart => "shows a part-to-whole relationship at a single point in time"

/-- Represents a time period --/
structure TimePeriod where
  start : String
  finish : String

/-- Represents a measurement that changes over time --/
structure TimeSeriesData where
  measurement : String
  period : TimePeriod

/-- Determines if a graph type is suitable for displaying time series data --/
def isSuitableForTimeSeries (g : GraphType) : Prop :=
  graphCharacteristics g = "shows trends or changes over time"

theorem line_graph_most_suitable_for_height_change 
  (data : TimeSeriesData) 
  (h : data.measurement = "height" ∧ 
       data.period.start = "elementary school" ∧ 
       data.period.finish = "junior high school") : 
  isSuitableForTimeSeries GraphType.LineGraph ∧ 
  ¬isSuitableForTimeSeries GraphType.BarGraph ∧ 
  ¬isSuitableForTimeSeries GraphType.PieChart := by
  sorry

#check line_graph_most_suitable_for_height_change

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_graph_most_suitable_for_height_change_l240_24023
