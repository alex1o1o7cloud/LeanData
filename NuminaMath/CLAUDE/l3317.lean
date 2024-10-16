import Mathlib

namespace NUMINAMATH_CALUDE_solve_system_l3317_331760

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 10) 
  (eq2 : 6 * p + 5 * q = 17) : 
  q = -25 / 11 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l3317_331760


namespace NUMINAMATH_CALUDE_town_population_problem_l3317_331747

theorem town_population_problem (original_population : ℕ) : 
  (((original_population + 1500) * 85 / 100 : ℕ) = original_population - 45) → 
  original_population = 8800 := by
  sorry

end NUMINAMATH_CALUDE_town_population_problem_l3317_331747


namespace NUMINAMATH_CALUDE_largest_n_for_equation_l3317_331780

theorem largest_n_for_equation : ∃ (x y z : ℕ+), 
  (100 : ℤ) = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 12 ∧ 
  ∀ (n : ℕ+), n > 10 → ¬∃ (a b c : ℕ+), 
    (n^2 : ℤ) = a^2 + b^2 + c^2 + 2*a*b + 2*b*c + 2*c*a + 5*a + 5*b + 5*c - 12 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_equation_l3317_331780


namespace NUMINAMATH_CALUDE_rectangle_cutting_l3317_331735

theorem rectangle_cutting (large_width large_height small_width small_height : ℝ) 
  (hw : large_width = 50)
  (hh : large_height = 90)
  (hsw : small_width = 1)
  (hsh : small_height = 10 * Real.sqrt 2) :
  ⌊(large_width * large_height) / (small_width * small_height)⌋ = 318 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_cutting_l3317_331735


namespace NUMINAMATH_CALUDE_monotonic_function_constraint_l3317_331730

theorem monotonic_function_constraint (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x ∈ Set.Icc (-1) 2, Monotone (fun x => -1/3 * x^3 + a * x^2 + b * x)) →
  a + b ≥ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_function_constraint_l3317_331730


namespace NUMINAMATH_CALUDE_trajectory_is_ellipse_chord_length_l3317_331793

-- Define the trajectory C
def trajectory_C (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y + Real.sqrt 3)^2) + Real.sqrt (x^2 + (y - Real.sqrt 3)^2) = 4

-- Define the equation of the ellipse
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 + y^2/4 = 1

-- Define the line y = 1/2x
def line (x y : ℝ) : Prop :=
  y = 1/2 * x

-- Theorem 1: The trajectory C is equivalent to the ellipse equation
theorem trajectory_is_ellipse :
  ∀ x y : ℝ, trajectory_C x y ↔ ellipse_equation x y :=
sorry

-- Theorem 2: The length of the chord AB is 4
theorem chord_length :
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse_equation x₁ y₁ ∧
    ellipse_equation x₂ y₂ ∧
    line x₁ y₁ ∧
    line x₂ y₂ ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 4 :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_ellipse_chord_length_l3317_331793


namespace NUMINAMATH_CALUDE_unique_prime_for_equiangular_polygons_l3317_331726

theorem unique_prime_for_equiangular_polygons :
  ∃! k : ℕ, 
    Prime k ∧ 
    k > 1 ∧
    ∃ (x n₁ n₂ : ℕ),
      -- Angle formula for P1
      x = 180 - 360 / n₁ ∧ 
      -- Angle formula for P2
      k * x = 180 - 360 / n₂ ∧ 
      -- Angles must be positive and less than 180°
      0 < x ∧ x < 180 ∧
      0 < k * x ∧ k * x < 180 ∧
      -- Number of sides must be at least 3
      n₁ ≥ 3 ∧ n₂ ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_for_equiangular_polygons_l3317_331726


namespace NUMINAMATH_CALUDE_range_of_x_inequality_l3317_331712

theorem range_of_x_inequality (x : ℝ) : 
  (∀ (a b : ℝ), a ≠ 0 → |a + b| + |a - b| ≥ |a| * |x - 2|) ↔ 0 ≤ x ∧ x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_inequality_l3317_331712


namespace NUMINAMATH_CALUDE_image_of_line_under_T_l3317_331772

/-- The transformation T on the plane -/
noncomputable def T (x y : ℝ) : ℝ × ℝ :=
  if |x| ≠ |y| then (x / (x^2 - y^2), y / (x^2 - y^2)) else (x, y)

/-- A line in the plane represented by Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ
  nonzero : A^2 + B^2 ≠ 0

/-- The image of a line under transformation T -/
def imageUnderT (l : Line) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), l.A * x + l.B * y + l.C = 0 ∧ p = T x y}

/-- Theorem stating the properties of the image of a line under T -/
theorem image_of_line_under_T (l : Line) :
  (l.C = 0 → imageUnderT l = {p | l.A * p.1 + l.B * p.2 = 0}) ∧
  (l.C ≠ 0 → |l.A| ≠ |l.B| → ∃ (a b c : ℝ), imageUnderT l = {p | (p.1 - a)^2 - (p.2 - b)^2 = c}) ∧
  (l.C ≠ 0 → |l.A| = |l.B| → ∃ (a b : ℝ), imageUnderT l = {p | (p.1 - a) * (p.1 - b) = 0}) :=
sorry

end NUMINAMATH_CALUDE_image_of_line_under_T_l3317_331772


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_80_l3317_331788

theorem thirty_percent_less_than_80 : 
  80 * (1 - 0.3) = (224 / 5) * (1 + 1 / 4) := by sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_80_l3317_331788


namespace NUMINAMATH_CALUDE_earth_inhabitable_fraction_l3317_331710

theorem earth_inhabitable_fraction :
  let water_fraction : ℚ := 2/3
  let land_fraction : ℚ := 1 - water_fraction
  let inhabitable_land_fraction : ℚ := 1/3
  (1 - water_fraction) * inhabitable_land_fraction = 1/9 :=
by
  sorry

end NUMINAMATH_CALUDE_earth_inhabitable_fraction_l3317_331710


namespace NUMINAMATH_CALUDE_olympic_triathlon_distance_l3317_331752

theorem olympic_triathlon_distance :
  ∀ (cycling running swimming : ℝ),
  cycling = 4 * running →
  swimming = (3 / 80) * cycling →
  running - swimming = 8.5 →
  cycling + running + swimming = 51.5 := by
sorry

end NUMINAMATH_CALUDE_olympic_triathlon_distance_l3317_331752


namespace NUMINAMATH_CALUDE_tangent_line_implies_a_and_b_l3317_331728

/-- Given a function f(x) = x^3 - 3ax^2 + b, prove that if the curve y = f(x) is tangent
    to the line y = 8 at the point (2, f(2)), then a = 1 and b = 12. -/
theorem tangent_line_implies_a_and_b (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 - 3*a*x^2 + b
  (f 2 = 8) ∧ (deriv f 2 = 0) → a = 1 ∧ b = 12 := by
  sorry

#check tangent_line_implies_a_and_b

end NUMINAMATH_CALUDE_tangent_line_implies_a_and_b_l3317_331728


namespace NUMINAMATH_CALUDE_middle_number_proof_l3317_331745

theorem middle_number_proof (x y z : ℕ) : 
  x < y ∧ y < z ∧ 
  x + y = 22 ∧ 
  x + z = 29 ∧ 
  y + z = 31 ∧ 
  x = 10 → 
  y = 12 := by
sorry

end NUMINAMATH_CALUDE_middle_number_proof_l3317_331745


namespace NUMINAMATH_CALUDE_real_part_of_inverse_l3317_331751

theorem real_part_of_inverse (z : ℂ) (h1 : z ≠ 0) (h2 : z.im ≠ 0) (h3 : Complex.abs z = 2) :
  (1 / (2 - z)).re = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_inverse_l3317_331751


namespace NUMINAMATH_CALUDE_profit_percentage_is_fifty_percent_l3317_331790

/-- Calculates the profit percentage given the costs and selling price -/
def profit_percentage (purchase_price repair_cost transport_cost selling_price : ℚ) : ℚ :=
  let total_cost := purchase_price + repair_cost + transport_cost
  let profit := selling_price - total_cost
  (profit / total_cost) * 100

/-- Theorem: The profit percentage is 50% given the specific costs and selling price -/
theorem profit_percentage_is_fifty_percent :
  profit_percentage 10000 5000 1000 24000 = 50 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_is_fifty_percent_l3317_331790


namespace NUMINAMATH_CALUDE_rotation_of_point_N_l3317_331791

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Rotates a point 180 degrees around the origin -/
def rotate180 (p : Point) : Point :=
  ⟨-p.x, -p.y⟩

theorem rotation_of_point_N : 
  let N : Point := ⟨-1, -2⟩
  rotate180 N = ⟨1, 2⟩ := by
  sorry

end NUMINAMATH_CALUDE_rotation_of_point_N_l3317_331791


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3317_331720

-- Define the sets A and B
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {y | ∃ x, y = 1 - x^2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3317_331720


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3317_331743

-- Define set A
def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3317_331743


namespace NUMINAMATH_CALUDE_conditional_structure_correctness_l3317_331782

-- Define a conditional structure
structure ConditionalStructure where
  hasTwoExits : Bool
  hasOneEffectiveExit : Bool

-- Define the properties of conditional structures
def conditionalStructureProperties : ConditionalStructure where
  hasTwoExits := true
  hasOneEffectiveExit := true

-- Theorem to prove
theorem conditional_structure_correctness :
  (conditionalStructureProperties.hasTwoExits = true) ∧
  (conditionalStructureProperties.hasOneEffectiveExit = true) := by
  sorry

#check conditional_structure_correctness

end NUMINAMATH_CALUDE_conditional_structure_correctness_l3317_331782


namespace NUMINAMATH_CALUDE_complex_number_real_l3317_331700

theorem complex_number_real (m : ℝ) :
  (m ≠ -5) →
  (∃ (z : ℂ), z = (m + 5)⁻¹ + (m^2 + 2*m - 15)*I ∧ z.im = 0) →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_number_real_l3317_331700


namespace NUMINAMATH_CALUDE_difference_twice_x_and_three_less_than_zero_l3317_331719

theorem difference_twice_x_and_three_less_than_zero (x : ℝ) :
  (2 * x - 3 < 0) ↔ (∃ y, y = 2 * x ∧ y - 3 < 0) :=
by sorry

end NUMINAMATH_CALUDE_difference_twice_x_and_three_less_than_zero_l3317_331719


namespace NUMINAMATH_CALUDE_amare_fabric_needed_l3317_331785

/-- The amount of fabric Amare needs for the dresses -/
def fabric_needed (fabric_per_dress : ℝ) (num_dresses : ℕ) (fabric_owned : ℝ) : ℝ :=
  fabric_per_dress * num_dresses * 3 - fabric_owned

/-- Theorem stating the amount of fabric Amare needs -/
theorem amare_fabric_needed :
  fabric_needed 5.5 4 7 = 59 := by
  sorry

end NUMINAMATH_CALUDE_amare_fabric_needed_l3317_331785


namespace NUMINAMATH_CALUDE_exists_x0_sin_minus_tan_negative_l3317_331702

open Real

theorem exists_x0_sin_minus_tan_negative :
  ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ < π/2 ∧ sin x₀ - tan x₀ < 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_x0_sin_minus_tan_negative_l3317_331702


namespace NUMINAMATH_CALUDE_probability_different_colors_is_three_fifths_l3317_331770

def num_red_balls : ℕ := 3
def num_white_balls : ℕ := 2
def total_balls : ℕ := num_red_balls + num_white_balls

def probability_different_colors : ℚ :=
  (num_red_balls * num_white_balls : ℚ) / ((total_balls * (total_balls - 1)) / 2 : ℚ)

theorem probability_different_colors_is_three_fifths :
  probability_different_colors = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_different_colors_is_three_fifths_l3317_331770


namespace NUMINAMATH_CALUDE_cylinder_intersection_area_l3317_331727

/-- Represents a cylinder --/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Represents the area of a surface formed by intersecting a cylinder with a plane --/
def intersectionArea (c : Cylinder) (arcAngle : ℝ) : ℝ := sorry

theorem cylinder_intersection_area :
  let c : Cylinder := { radius := 7, height := 9 }
  let arcAngle : ℝ := 150 * (π / 180)  -- Convert degrees to radians
  intersectionArea c arcAngle = 62.4 * π + 112 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_cylinder_intersection_area_l3317_331727


namespace NUMINAMATH_CALUDE_frank_payment_l3317_331776

/-- The amount of money Frank handed to the cashier -/
def amount_handed (chocolate_bars : ℕ) (chips : ℕ) (chocolate_price : ℕ) (chips_price : ℕ) (change : ℕ) : ℕ :=
  chocolate_bars * chocolate_price + chips * chips_price + change

/-- Proof that Frank handed $20 to the cashier -/
theorem frank_payment : amount_handed 5 2 2 3 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_frank_payment_l3317_331776


namespace NUMINAMATH_CALUDE_angle_MDN_is_acute_l3317_331738

/-- The parabola y^2 = 2x -/
def parabola (x y : ℝ) : Prop := y^2 = 2*x

/-- A line passing through point (2,0) -/
def line_through_P (k : ℝ) (x y : ℝ) : Prop := x = k*y + 2

/-- The vertical line x = -1/2 -/
def vertical_line (x : ℝ) : Prop := x = -1/2

/-- The dot product of two 2D vectors -/
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1*x2 + y1*y2

theorem angle_MDN_is_acute (k t : ℝ) (xM yM xN yN : ℝ) :
  parabola xM yM →
  parabola xN yN →
  line_through_P k xM yM →
  line_through_P k xN yN →
  vertical_line (-1/2) →
  xM ≠ xN ∨ yM ≠ yN →
  dot_product (xM + 1/2) (yM - t) (xN + 1/2) (yN - t) > 0 :=
sorry

end NUMINAMATH_CALUDE_angle_MDN_is_acute_l3317_331738


namespace NUMINAMATH_CALUDE_inequality_theorem_l3317_331753

-- Define the inequality and its solution set
def inequality (m : ℝ) (x : ℝ) : Prop := m - |x - 2| ≥ 1
def solution_set (m : ℝ) : Set ℝ := {x : ℝ | inequality m x}

-- Define the theorem
theorem inequality_theorem (m : ℝ) 
  (h1 : solution_set m = Set.Icc 0 4) 
  (a b : ℝ) 
  (h2 : a > 0) 
  (h3 : b > 0) 
  (h4 : a + b = m) : 
  m = 3 ∧ ∃ (min : ℝ), min = 9/2 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a + b = m → a^2 + b^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3317_331753


namespace NUMINAMATH_CALUDE_hyperbola_focus_asymptote_distance_l3317_331777

/-- The distance from the focus of a hyperbola to its asymptote -/
def distance_focus_to_asymptote (b : ℝ) : ℝ := 
  sorry

/-- The theorem stating the distance from the focus to the asymptote for a specific hyperbola -/
theorem hyperbola_focus_asymptote_distance : 
  ∀ b : ℝ, b > 0 → 
  (∀ x y : ℝ, x^2 / 4 - y^2 / b^2 = 1) → 
  (∃ x : ℝ, x^2 / 4 + b^2 = 9) →
  (∀ x y : ℝ, y^2 = 12*x) →
  distance_focus_to_asymptote b = Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_asymptote_distance_l3317_331777


namespace NUMINAMATH_CALUDE_parabola_circle_tangency_l3317_331757

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = x

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define a point on the parabola
def point_on_parabola (p : ℝ × ℝ) : Prop := parabola_C p.1 p.2

-- Define tangency of a line to the circle
def line_tangent_to_circle (p q : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), circle_M (p.1 + t * (q.1 - p.1)) (p.2 + t * (q.2 - p.2)) ∧
             ∀ (s : ℝ), s ≠ t → ¬circle_M (p.1 + s * (q.1 - p.1)) (p.2 + s * (q.2 - p.2))

theorem parabola_circle_tangency 
  (A₁ A₂ A₃ : ℝ × ℝ) 
  (h₁ : point_on_parabola A₁) 
  (h₂ : point_on_parabola A₂) 
  (h₃ : point_on_parabola A₃) 
  (h₄ : line_tangent_to_circle A₁ A₂) 
  (h₅ : line_tangent_to_circle A₁ A₃) : 
  line_tangent_to_circle A₂ A₃ := by
  sorry

end NUMINAMATH_CALUDE_parabola_circle_tangency_l3317_331757


namespace NUMINAMATH_CALUDE_ceiling_sqrt_165_l3317_331779

theorem ceiling_sqrt_165 : ⌈Real.sqrt 165⌉ = 13 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_165_l3317_331779


namespace NUMINAMATH_CALUDE_inequality_chain_l3317_331773

theorem inequality_chain (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a + b = 1) :
  2 * a * b < (a + b) / 2 ∧ (a + b) / 2 < (a^4 - b^4) / (a - b) ∧ (a^4 - b^4) / (a - b) < b := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l3317_331773


namespace NUMINAMATH_CALUDE_ned_video_game_earnings_l3317_331758

/-- Calculates the total money earned from selling video games --/
def totalEarnings (totalGames : ℕ) (nonWorkingGames : ℕ) 
                  (firstGroupSize : ℕ) (firstGroupPrice : ℕ)
                  (secondGroupSize : ℕ) (secondGroupPrice : ℕ)
                  (remainingPrice : ℕ) : ℕ :=
  let workingGames := totalGames - nonWorkingGames
  let remainingGames := workingGames - firstGroupSize - secondGroupSize
  firstGroupSize * firstGroupPrice + 
  secondGroupSize * secondGroupPrice + 
  remainingGames * remainingPrice

/-- Theorem stating the total earnings from selling the working games --/
theorem ned_video_game_earnings : 
  totalEarnings 25 8 5 9 7 12 15 = 204 := by
  sorry

end NUMINAMATH_CALUDE_ned_video_game_earnings_l3317_331758


namespace NUMINAMATH_CALUDE_duplicate_page_number_l3317_331737

/-- The largest positive integer n such that n(n+1)/2 < 2550 -/
def n : ℕ := 70

/-- The theorem stating the existence and uniqueness of the duplicated page number -/
theorem duplicate_page_number :
  ∃! x : ℕ, x ≤ n ∧ (n * (n + 1)) / 2 + x = 2550 := by sorry

end NUMINAMATH_CALUDE_duplicate_page_number_l3317_331737


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l3317_331764

theorem largest_angle_in_triangle (a b c : ℝ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a + b = 105 →      -- Sum of two angles is 7/6 of a right angle (90° * 7/6 = 105°)
  b = a + 40 →       -- One angle is 40° larger than the other
  max a (max b c) = 75 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l3317_331764


namespace NUMINAMATH_CALUDE_quadratic_coefficients_l3317_331729

/-- A quadratic function with vertex (4, -1) passing through (0, 7) has coefficients a = 1/2, b = -4, and c = 7. -/
theorem quadratic_coefficients :
  ∀ (f : ℝ → ℝ) (a b c : ℝ),
    (∀ x, f x = a * x^2 + b * x + c) →
    (∀ x, f x = f (8 - x)) →
    f 4 = -1 →
    f 0 = 7 →
    a = (1/2 : ℝ) ∧ b = -4 ∧ c = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_l3317_331729


namespace NUMINAMATH_CALUDE_quadrilateral_area_l3317_331759

theorem quadrilateral_area (d h₁ h₂ : ℝ) (hd : d = 26) (hh₁ : h₁ = 9) (hh₂ : h₂ = 6) :
  (1/2) * d * (h₁ + h₂) = 195 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l3317_331759


namespace NUMINAMATH_CALUDE_line_parameterization_l3317_331742

/-- Given a line y = -3x + 2 parameterized as [x; y] = [5; r] + t[k; 8], prove r = -13 and k = -4 -/
theorem line_parameterization (r k : ℝ) : 
  (∀ x y t : ℝ, y = -3 * x + 2 ↔ ∃ t, (x, y) = (5 + t * k, r + t * 8)) →
  r = -13 ∧ k = -4 := by
  sorry

end NUMINAMATH_CALUDE_line_parameterization_l3317_331742


namespace NUMINAMATH_CALUDE_smallest_base_for_100_l3317_331709

theorem smallest_base_for_100 : 
  ∃ (b : ℕ), b = 5 ∧ 
  (∀ (x : ℕ), x < b → ¬(x^2 ≤ 100 ∧ 100 < x^3)) ∧
  (5^2 ≤ 100 ∧ 100 < 5^3) := by
  sorry

end NUMINAMATH_CALUDE_smallest_base_for_100_l3317_331709


namespace NUMINAMATH_CALUDE_gcd_factorial_8_and_factorial_11_times_9_squared_l3317_331762

theorem gcd_factorial_8_and_factorial_11_times_9_squared :
  Nat.gcd (Nat.factorial 8) (Nat.factorial 11 * 9^2) = Nat.factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_8_and_factorial_11_times_9_squared_l3317_331762


namespace NUMINAMATH_CALUDE_area_of_region_R_l3317_331799

/-- A square with side length 3 -/
structure Square :=
  (side_length : ℝ)
  (is_three : side_length = 3)

/-- The region R in the square -/
def region_R (s : Square) :=
  {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1^2 + p.2^2 ≤ (3*Real.sqrt 2/2)^2}

/-- The area of a region -/
noncomputable def area (r : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem to be proved -/
theorem area_of_region_R (s : Square) : area (region_R s) = 9 * Real.pi / 8 := by
  sorry

end NUMINAMATH_CALUDE_area_of_region_R_l3317_331799


namespace NUMINAMATH_CALUDE_tic_tac_toe_strategy_l3317_331765

/-- Represents a 10x10 tic-tac-toe board -/
def Board := Fin 10 → Fin 10 → Bool

/-- Counts the number of sets of five consecutive marks for a player -/
def count_sets (b : Board) (player : Bool) : ℕ := sorry

/-- Calculates the score for the first player (X) -/
def score (b : Board) : ℤ :=
  (count_sets b true : ℤ) - (count_sets b false : ℤ)

/-- A strategy for a player -/
def Strategy := Board → Fin 10 × Fin 10

/-- Applies a strategy to a board, returning the updated board -/
def apply_strategy (b : Board) (s : Strategy) (player : Bool) : Board := sorry

/-- Represents a full game play -/
def play_game (s1 s2 : Strategy) : Board := sorry

theorem tic_tac_toe_strategy :
  (∃ (s : Strategy), ∀ (s2 : Strategy), score (play_game s s2) ≥ 0) ∧
  (¬ ∃ (s : Strategy), ∀ (s2 : Strategy), score (play_game s s2) > 0) :=
sorry

end NUMINAMATH_CALUDE_tic_tac_toe_strategy_l3317_331765


namespace NUMINAMATH_CALUDE_grid_game_winner_parity_second_player_wins_when_even_first_player_wins_when_odd_l3317_331775

/-- Represents the outcome of the grid game -/
inductive GameOutcome
  | FirstPlayerWins
  | SecondPlayerWins

/-- Determines the winner of the grid game based on the dimensions of the grid -/
def gridGameWinner (m n : ℕ) : GameOutcome :=
  if (m + n) % 2 = 0 then
    GameOutcome.SecondPlayerWins
  else
    GameOutcome.FirstPlayerWins

/-- Theorem stating the winning condition for the grid game -/
theorem grid_game_winner_parity (m n : ℕ) :
  gridGameWinner m n = 
    if (m + n) % 2 = 0 then 
      GameOutcome.SecondPlayerWins
    else 
      GameOutcome.FirstPlayerWins := by
  sorry

/-- Corollary: The second player wins when m + n is even -/
theorem second_player_wins_when_even (m n : ℕ) (h : (m + n) % 2 = 0) :
  gridGameWinner m n = GameOutcome.SecondPlayerWins := by
  sorry

/-- Corollary: The first player wins when m + n is odd -/
theorem first_player_wins_when_odd (m n : ℕ) (h : (m + n) % 2 ≠ 0) :
  gridGameWinner m n = GameOutcome.FirstPlayerWins := by
  sorry

end NUMINAMATH_CALUDE_grid_game_winner_parity_second_player_wins_when_even_first_player_wins_when_odd_l3317_331775


namespace NUMINAMATH_CALUDE_x_minus_y_squared_l3317_331744

theorem x_minus_y_squared (x y : ℝ) : 
  y = Real.sqrt (2 * x - 3) + Real.sqrt (3 - 2 * x) - 4 →
  x = 3 / 2 →
  x - y^2 = -29 / 2 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_squared_l3317_331744


namespace NUMINAMATH_CALUDE_village_population_equality_l3317_331755

/-- The initial population of Village X -/
def initial_population_X : ℕ := 78000

/-- The yearly decrease in population of Village X -/
def decrease_rate_X : ℕ := 1200

/-- The yearly increase in population of Village Y -/
def increase_rate_Y : ℕ := 800

/-- The number of years after which the populations will be equal -/
def years : ℕ := 18

/-- The initial population of Village Y -/
def initial_population_Y : ℕ := 42000

theorem village_population_equality :
  initial_population_X - decrease_rate_X * years = 
  initial_population_Y + increase_rate_Y * years :=
by sorry

#check village_population_equality

end NUMINAMATH_CALUDE_village_population_equality_l3317_331755


namespace NUMINAMATH_CALUDE_percentage_problem_l3317_331792

theorem percentage_problem (P : ℝ) : 
  (100 : ℝ) = (P / 100) * 100 + 84 → P = 16 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3317_331792


namespace NUMINAMATH_CALUDE_no_natural_solutions_l3317_331716

theorem no_natural_solutions (x y : ℕ) : 
  (1 : ℚ) / (x^2 : ℚ) + (1 : ℚ) / ((x * y) : ℚ) + (1 : ℚ) / (y^2 : ℚ) ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solutions_l3317_331716


namespace NUMINAMATH_CALUDE_dans_seashells_l3317_331725

def seashells_problem (initial_seashells : ℕ) (remaining_seashells : ℕ) : Prop :=
  initial_seashells ≥ remaining_seashells →
  ∃ (given_seashells : ℕ), given_seashells = initial_seashells - remaining_seashells

theorem dans_seashells : seashells_problem 56 22 := by
  sorry

end NUMINAMATH_CALUDE_dans_seashells_l3317_331725


namespace NUMINAMATH_CALUDE_no_solution_condition_l3317_331741

theorem no_solution_condition (k : ℝ) : 
  (∀ x : ℝ, x ≠ 3 ∧ x ≠ 4 → (x - 1) / (x - 3) ≠ (x - k) / (x - 4)) ↔ k = 2 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_condition_l3317_331741


namespace NUMINAMATH_CALUDE_figure_to_square_approximation_l3317_331715

/-- A figure on a grid of squares -/
structure GridFigure where
  area : ℕ
  is_on_grid : Bool

/-- Represents a division of a figure into parts -/
structure FigureDivision where
  parts : ℕ
  can_rearrange_to_square : Bool

/-- Theorem: A figure with 18 unit squares can be divided into three parts and rearranged to approximate a square -/
theorem figure_to_square_approximation (f : GridFigure) (d : FigureDivision) :
  f.area = 18 ∧ f.is_on_grid = true ∧ d.parts = 3 → d.can_rearrange_to_square = true := by
  sorry

end NUMINAMATH_CALUDE_figure_to_square_approximation_l3317_331715


namespace NUMINAMATH_CALUDE_multiplication_subtraction_equality_l3317_331733

theorem multiplication_subtraction_equality : 72 * 989 - 12 * 989 = 59340 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_subtraction_equality_l3317_331733


namespace NUMINAMATH_CALUDE_intercept_ratio_l3317_331714

/-- Given two lines with the same y-intercept (0, b) where b ≠ 0,
    if the first line has slope 12 and x-intercept (s, 0),
    and the second line has slope 8 and x-intercept (t, 0),
    then s/t = 2/3 -/
theorem intercept_ratio (b s t : ℝ) (hb : b ≠ 0)
  (h1 : 0 = 12 * s + b) (h2 : 0 = 8 * t + b) : s / t = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_intercept_ratio_l3317_331714


namespace NUMINAMATH_CALUDE_equal_parts_complex_l3317_331769

/-- A complex number is an "equal parts complex number" if its real and imaginary parts are equal -/
def is_equal_parts (z : ℂ) : Prop := z.re = z.im

/-- Given that Z = (1+ai)i is an "equal parts complex number", prove that a = -1 -/
theorem equal_parts_complex (a : ℝ) :
  is_equal_parts ((1 + a * Complex.I) * Complex.I) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_equal_parts_complex_l3317_331769


namespace NUMINAMATH_CALUDE_rachels_father_age_at_25_is_60_l3317_331734

/-- Calculates the age of Rachel's father when Rachel is 25 years old -/
def rachels_father_age_at_25 (rachel_current_age : ℕ) (grandfather_age_multiplier : ℕ) (father_age_difference : ℕ) : ℕ :=
  let grandfather_age := rachel_current_age * grandfather_age_multiplier
  let mother_age := grandfather_age / 2
  let father_current_age := mother_age + father_age_difference
  let years_until_25 := 25 - rachel_current_age
  father_current_age + years_until_25

/-- Theorem stating that Rachel's father will be 60 years old when Rachel is 25 -/
theorem rachels_father_age_at_25_is_60 :
  rachels_father_age_at_25 12 7 5 = 60 := by
  sorry

#eval rachels_father_age_at_25 12 7 5

end NUMINAMATH_CALUDE_rachels_father_age_at_25_is_60_l3317_331734


namespace NUMINAMATH_CALUDE_odd_function_with_period_4_sum_l3317_331708

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_function_with_period_4_sum (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) (h_period : has_period f 4) :
  f 2005 + f 2006 + f 2007 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_with_period_4_sum_l3317_331708


namespace NUMINAMATH_CALUDE_sparrow_percentage_among_non_owls_l3317_331736

theorem sparrow_percentage_among_non_owls (total : ℝ) (total_pos : 0 < total) :
  let sparrows := 0.4 * total
  let owls := 0.2 * total
  let pigeons := 0.1 * total
  let finches := 0.2 * total
  let robins := total - (sparrows + owls + pigeons + finches)
  let non_owls := total - owls
  (sparrows / non_owls) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_sparrow_percentage_among_non_owls_l3317_331736


namespace NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l3317_331750

theorem arithmetic_sequence_seventh_term
  (a : ℚ) -- First term of the sequence
  (d : ℚ) -- Common difference of the sequence
  (h1 : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 20) -- Sum of first five terms
  (h2 : a + 5*d = 8) -- Sixth term
  : a + 6*d = 28/3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l3317_331750


namespace NUMINAMATH_CALUDE_pen_purchase_cost_l3317_331787

/-- The cost of a single brand X pen -/
def brand_x_cost : ℚ := 4

/-- The cost of a single brand Y pen -/
def brand_y_cost : ℚ := 14/5

/-- The number of brand X pens purchased -/
def num_brand_x : ℕ := 8

/-- The total number of pens purchased -/
def total_pens : ℕ := 12

/-- The number of brand Y pens purchased -/
def num_brand_y : ℕ := total_pens - num_brand_x

/-- The total cost of all pens purchased -/
def total_cost : ℚ := num_brand_x * brand_x_cost + num_brand_y * brand_y_cost

theorem pen_purchase_cost : total_cost = 216/5 := by sorry

end NUMINAMATH_CALUDE_pen_purchase_cost_l3317_331787


namespace NUMINAMATH_CALUDE_division_problem_l3317_331796

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 52 → 
  divisor = 3 → 
  remainder = 4 → 
  dividend = divisor * quotient + remainder →
  quotient = 16 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3317_331796


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3317_331701

theorem complex_magnitude_problem (z : ℂ) : z = 2 / (1 - I) + I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3317_331701


namespace NUMINAMATH_CALUDE_smallest_angle_proof_l3317_331713

def AP : ℝ := 2

noncomputable def smallest_angle (x : ℝ) : ℝ :=
  Real.arctan (Real.sqrt 2 / 4)

theorem smallest_angle_proof (x : ℝ) : 
  smallest_angle x = Real.arctan (Real.sqrt 2 / 4) :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_proof_l3317_331713


namespace NUMINAMATH_CALUDE_pop_albums_count_l3317_331771

def country_albums : ℕ := 2
def songs_per_album : ℕ := 6
def total_songs : ℕ := 30

theorem pop_albums_count : 
  ∃ (pop_albums : ℕ), 
    country_albums * songs_per_album + pop_albums * songs_per_album = total_songs ∧ 
    pop_albums = 3 := by
  sorry

end NUMINAMATH_CALUDE_pop_albums_count_l3317_331771


namespace NUMINAMATH_CALUDE_vector_properties_l3317_331739

/-- Given points A and B in a 2D Cartesian coordinate system, prove properties of vectors AB and OA·OB --/
theorem vector_properties (A B : ℝ × ℝ) (h1 : A = (-3, -4)) (h2 : B = (5, -12)) :
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let OA : ℝ × ℝ := A
  let OB : ℝ × ℝ := B
  (AB = (8, -8)) ∧
  (Real.sqrt ((AB.1)^2 + (AB.2)^2) = 8 * Real.sqrt 2) ∧
  (OA.1 * OB.1 + OA.2 * OB.2 = 33) := by
sorry

end NUMINAMATH_CALUDE_vector_properties_l3317_331739


namespace NUMINAMATH_CALUDE_smallest_shift_for_scaled_function_l3317_331783

-- Define a periodic function with period 30
def isPeriodic30 (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x - 30) = g x

-- Define the property we're looking for
def hasProperty (g : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x, g ((x - b) / 3) = g (x / 3)

theorem smallest_shift_for_scaled_function (g : ℝ → ℝ) (h : isPeriodic30 g) :
  ∃ b : ℝ, b > 0 ∧ hasProperty g b ∧ ∀ b' : ℝ, b' > 0 → hasProperty g b' → b ≤ b' :=
sorry

end NUMINAMATH_CALUDE_smallest_shift_for_scaled_function_l3317_331783


namespace NUMINAMATH_CALUDE_least_common_period_l3317_331761

-- Define the property that f satisfies the given functional equation
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 6) + f (x - 6) = f x

-- Define the property of being periodic with period p
def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

-- State the theorem
theorem least_common_period :
  ∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f →
    (∃ p : ℝ, p > 0 ∧ IsPeriodic f p) →
    (∀ q : ℝ, q > 0 → IsPeriodic f q → q ≥ 36) :=
  sorry

end NUMINAMATH_CALUDE_least_common_period_l3317_331761


namespace NUMINAMATH_CALUDE_decreasing_function_on_positive_reals_l3317_331723

/-- The function f(x) = -x(x+2) is decreasing on the interval (0, +∞) -/
theorem decreasing_function_on_positive_reals :
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → -x * (x + 2) > -y * (y + 2) := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_on_positive_reals_l3317_331723


namespace NUMINAMATH_CALUDE_quadratic_extrema_l3317_331763

-- Define the function
def f (x : ℝ) : ℝ := (x - 3)^2 - 1

-- Define the domain
def domain : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}

theorem quadratic_extrema :
  ∃ (min max : ℝ), 
    (∀ x ∈ domain, f x ≥ min) ∧
    (∃ x ∈ domain, f x = min) ∧
    (∀ x ∈ domain, f x ≤ max) ∧
    (∃ x ∈ domain, f x = max) ∧
    min = -1 ∧ max = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_extrema_l3317_331763


namespace NUMINAMATH_CALUDE_smallest_prime_twelve_less_prime_square_l3317_331781

theorem smallest_prime_twelve_less_prime_square : ∃ (p n : ℕ), 
  p = 13 ∧ 
  Nat.Prime p ∧ 
  Nat.Prime n ∧ 
  p = n^2 - 12 ∧
  ∀ (q m : ℕ), Nat.Prime q ∧ Nat.Prime m ∧ q = m^2 - 12 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_twelve_less_prime_square_l3317_331781


namespace NUMINAMATH_CALUDE_diagonals_15_gon_l3317_331766

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a convex 15-gon is 90 -/
theorem diagonals_15_gon : num_diagonals 15 = 90 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_15_gon_l3317_331766


namespace NUMINAMATH_CALUDE_unique_zero_of_f_l3317_331795

noncomputable def f (x : ℝ) := 2^x + x^3 - 2

theorem unique_zero_of_f :
  ∃! x : ℝ, f x = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_zero_of_f_l3317_331795


namespace NUMINAMATH_CALUDE_largest_prime_value_l3317_331774

theorem largest_prime_value (p x y : ℕ) : 
  p.Prime → 
  x > 0 → 
  y > 0 → 
  x^3 + y^3 - 3*x*y = p - 1 → 
  p ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_largest_prime_value_l3317_331774


namespace NUMINAMATH_CALUDE_circle_kinetic_energy_l3317_331717

/-- 
Given a circle with radius R and a point P on its diameter AB, where PC is a semicord perpendicular to AB,
if three unit masses move along PA, PB, and PC with constant velocities reaching A, B, and C respectively in one unit of time,
and the total kinetic energy expended is a^2 units, then:
1. The distance of P from A is R ± √(2a^2 - 3R^2)
2. The value of a^2 must satisfy 3/2 * R^2 ≤ a^2 < 2R^2
-/
theorem circle_kinetic_energy (R a : ℝ) (h : R > 0) :
  let PA : ℝ → ℝ := λ x => x
  let PB : ℝ → ℝ := λ x => 2 * R - x
  let PC : ℝ → ℝ := λ x => Real.sqrt (x * (2 * R - x))
  let kinetic_energy : ℝ → ℝ := λ x => (PA x)^2 / 2 + (PB x)^2 / 2 + (PC x)^2 / 2
  ∃ x : ℝ, 0 < x ∧ x < 2 * R ∧ kinetic_energy x = a^2 →
    (x = R + Real.sqrt (2 * a^2 - 3 * R^2) ∨ x = R - Real.sqrt (2 * a^2 - 3 * R^2)) ∧
    3 / 2 * R^2 ≤ a^2 ∧ a^2 < 2 * R^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_kinetic_energy_l3317_331717


namespace NUMINAMATH_CALUDE_bella_steps_l3317_331786

/-- The distance between Bella's and Ella's houses in feet -/
def distance : ℝ := 10560

/-- The ratio of Ella's speed to Bella's speed -/
def speed_ratio : ℝ := 3

/-- The length of Bella's step in feet -/
def step_length : ℝ := 3

/-- The number of steps Bella takes before meeting Ella -/
def steps : ℕ := 880

theorem bella_steps :
  (distance / (1 + speed_ratio)) / step_length = steps := by
  sorry

end NUMINAMATH_CALUDE_bella_steps_l3317_331786


namespace NUMINAMATH_CALUDE_equation_represents_three_lines_l3317_331767

-- Define the equation
def equation (x y : ℝ) : Prop := (x + y)^3 = x^3 + y^3

-- Define what it means for a point to be on a line
def on_line (x y a b c : ℝ) : Prop := a*x + b*y + c = 0

-- Define the three lines we expect
def line1 (x y : ℝ) : Prop := on_line x y 1 1 0  -- x + y = 0
def line2 (x y : ℝ) : Prop := on_line x y 1 0 0  -- x = 0
def line3 (x y : ℝ) : Prop := on_line x y 0 1 0  -- y = 0

-- Theorem statement
theorem equation_represents_three_lines :
  ∀ x y : ℝ, equation x y ↔ (line1 x y ∨ line2 x y ∨ line3 x y) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_three_lines_l3317_331767


namespace NUMINAMATH_CALUDE_remaining_cooking_time_l3317_331722

def total_potatoes : ℕ := 16
def cooked_potatoes : ℕ := 7
def cooking_time_per_potato : ℕ := 5

theorem remaining_cooking_time : 
  (total_potatoes - cooked_potatoes) * cooking_time_per_potato = 45 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cooking_time_l3317_331722


namespace NUMINAMATH_CALUDE_F_r_properties_l3317_331731

/-- Represents a point in the cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the polygon F_r -/
def F_r (r : ℝ) : Set Point :=
  {p : Point | p.x^2 + p.y^2 = r^2 ∧ (p.x * p.y)^2 = 1}

/-- The area of the polygon F_r as a function of r -/
noncomputable def area (r : ℝ) : ℝ :=
  sorry

/-- Predicate to check if a polygon is regular -/
def is_regular (s : Set Point) : Prop :=
  sorry

theorem F_r_properties :
  ∃ (A : ℝ → ℝ),
    (∀ r, A r = area r) ∧
    is_regular (F_r 1) ∧
    ∀ r > 1, is_regular (F_r r) := by
  sorry

end NUMINAMATH_CALUDE_F_r_properties_l3317_331731


namespace NUMINAMATH_CALUDE_angle_measure_proof_l3317_331778

theorem angle_measure_proof (x : ℝ) : 
  (180 - x = 6 * (90 - x)) → x = 72 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l3317_331778


namespace NUMINAMATH_CALUDE_intersection_A_B_l3317_331703

def A : Set ℝ := {-1, 0, 2, 3, 5}

def B : Set ℝ := {x | -1 < x ∧ x < 3}

theorem intersection_A_B : A ∩ B = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3317_331703


namespace NUMINAMATH_CALUDE_equation_solutions_l3317_331748

theorem equation_solutions :
  (∃ x : ℚ, 4 * (x + 3) = 25 ∧ x = 13 / 4) ∧
  (∃ x₁ x₂ : ℚ, 5 * x₁^2 - 3 * x₁ = x₁ + 1 ∧ x₁ = -1 / 5 ∧
               5 * x₂^2 - 3 * x₂ = x₂ + 1 ∧ x₂ = 1) ∧
  (∃ x₁ x₂ : ℚ, 2 * (x₁ - 2)^2 - (x₁ - 2) = 0 ∧ x₁ = 2 ∧
               2 * (x₂ - 2)^2 - (x₂ - 2) = 0 ∧ x₂ = 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3317_331748


namespace NUMINAMATH_CALUDE_max_intersections_nested_polygons_l3317_331789

/-- Represents a convex polygon -/
structure ConvexPolygon where
  sides : ℕ
  convex : Bool

/-- Represents the configuration of two nested convex polygons -/
structure NestedPolygons where
  inner : ConvexPolygon
  outer : ConvexPolygon
  nested : Bool
  no_shared_segments : Bool

/-- Calculates the maximum number of intersection points between two nested convex polygons -/
def max_intersections (np : NestedPolygons) : ℕ :=
  np.inner.sides * np.outer.sides

/-- Theorem stating the maximum number of intersections for the given configuration -/
theorem max_intersections_nested_polygons :
  ∀ (np : NestedPolygons),
    np.inner.sides = 5 →
    np.outer.sides = 8 →
    np.inner.convex = true →
    np.outer.convex = true →
    np.nested = true →
    np.no_shared_segments = true →
    max_intersections np = 40 :=
by sorry

end NUMINAMATH_CALUDE_max_intersections_nested_polygons_l3317_331789


namespace NUMINAMATH_CALUDE_sports_equipment_pricing_and_purchasing_l3317_331754

theorem sports_equipment_pricing_and_purchasing (x y a b : ℤ) : 
  (2 * x + y = 330) →
  (5 * x + 2 * y = 780) →
  (120 * a + 90 * b = 810) →
  (x = 120 ∧ y = 90) ∧ (a = 3 ∧ b = 5) :=
by sorry

end NUMINAMATH_CALUDE_sports_equipment_pricing_and_purchasing_l3317_331754


namespace NUMINAMATH_CALUDE_sum_removal_proof_l3317_331707

theorem sum_removal_proof : 
  let original_sum := (1/2 : ℚ) + 1/3 + 1/4 + 1/6 + 1/8 + 1/9 + 1/12
  let removed_terms := (1/8 : ℚ) + 1/9
  original_sum - removed_terms = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_removal_proof_l3317_331707


namespace NUMINAMATH_CALUDE_intersection_distance_l3317_331706

/-- Given ω > 0, if the distance between the two closest intersection points 
    of y = 4sin(ωx) and y = 4cos(ωx) is 6, then ω = π/2 -/
theorem intersection_distance (ω : Real) (h1 : ω > 0) : 
  (∃ x₁ x₂ : Real, 
    x₁ ≠ x₂ ∧ 
    4 * Real.sin (ω * x₁) = 4 * Real.cos (ω * x₁) ∧
    4 * Real.sin (ω * x₂) = 4 * Real.cos (ω * x₂) ∧
    ∀ x : Real, 4 * Real.sin (ω * x) = 4 * Real.cos (ω * x) → 
      (x = x₁ ∨ x = x₂ ∨ |x - x₁| ≥ |x₁ - x₂| ∧ |x - x₂| ≥ |x₁ - x₂|) ∧
    (x₁ - x₂)^2 = 36) →
  ω = π / 2 := by sorry

end NUMINAMATH_CALUDE_intersection_distance_l3317_331706


namespace NUMINAMATH_CALUDE_quadratic_real_roots_k_range_l3317_331768

theorem quadratic_real_roots_k_range (k : ℝ) :
  (∃ x : ℝ, 2 * x^2 + 4 * x + k - 1 = 0) →
  k ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_k_range_l3317_331768


namespace NUMINAMATH_CALUDE_tallest_player_height_l3317_331798

theorem tallest_player_height (shortest_height : ℝ) (height_difference : ℝ) 
  (h1 : shortest_height = 68.25)
  (h2 : height_difference = 9.5) :
  shortest_height + height_difference = 77.75 := by
  sorry

end NUMINAMATH_CALUDE_tallest_player_height_l3317_331798


namespace NUMINAMATH_CALUDE_sqrt_8_times_sqrt_18_l3317_331721

theorem sqrt_8_times_sqrt_18 : Real.sqrt 8 * Real.sqrt 18 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_8_times_sqrt_18_l3317_331721


namespace NUMINAMATH_CALUDE_binary_octal_conversion_l3317_331749

/-- Converts a binary number (represented as a list of 0s and 1s) to decimal -/
def binary_to_decimal (binary : List Nat) : Nat :=
  binary.reverse.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- Converts a decimal number to octal (represented as a list of digits) -/
def decimal_to_octal (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else aux (m / 8) ((m % 8) :: acc)
  aux n []

theorem binary_octal_conversion :
  let binary : List Nat := [1, 0, 1, 1, 1, 1]
  let decimal : Nat := binary_to_decimal binary
  let octal : List Nat := decimal_to_octal decimal
  decimal = 47 ∧ octal = [5, 7] := by sorry

end NUMINAMATH_CALUDE_binary_octal_conversion_l3317_331749


namespace NUMINAMATH_CALUDE_equation_has_real_root_l3317_331794

theorem equation_has_real_root (K : ℝ) : ∃ x : ℝ, x = K^3 * (x - 1) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_has_real_root_l3317_331794


namespace NUMINAMATH_CALUDE_find_number_l3317_331711

theorem find_number : ∃ x : ℝ, (0.4 * x - 30 = 50) ∧ (x = 200) := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3317_331711


namespace NUMINAMATH_CALUDE_complex_square_equation_l3317_331724

theorem complex_square_equation (a b : ℕ+) :
  (a + b * Complex.I) ^ 2 = 7 + 24 * Complex.I →
  a + b * Complex.I = 4 + 3 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_square_equation_l3317_331724


namespace NUMINAMATH_CALUDE_fraction_division_problem_l3317_331705

theorem fraction_division_problem : (3/7 + 1/3) / (2/5) = 40/21 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_problem_l3317_331705


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3317_331740

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    one of its asymptotes is perpendicular to the line l: x - 2y - 5 = 0,
    and one of its foci lies on line l,
    prove that the equation of the hyperbola is x²/5 - y²/20 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_asymptote : ∃ (m : ℝ), m * (1/2) = -1 ∧ m = b/a)
  (h_focus : ∃ (x y : ℝ), x - 2*y - 5 = 0 ∧ x^2/a^2 - y^2/b^2 = 1 ∧ x^2 - (a^2 + b^2) = 0) :
  a^2 = 5 ∧ b^2 = 20 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3317_331740


namespace NUMINAMATH_CALUDE_principal_amount_is_16065_l3317_331797

/-- Calculates the principal amount given simple interest, rate, and time -/
def calculate_principal (simple_interest : ℚ) (rate : ℚ) (time : ℕ) : ℚ :=
  (simple_interest * 100) / (rate * time)

/-- Theorem: Given the specified conditions, the principal amount is 16065 -/
theorem principal_amount_is_16065 :
  let simple_interest : ℚ := 4016.25
  let rate : ℚ := 5
  let time : ℕ := 5
  calculate_principal simple_interest rate time = 16065 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_is_16065_l3317_331797


namespace NUMINAMATH_CALUDE_age_diff_ratio_is_two_to_one_l3317_331732

-- Define the current ages of Roy, Julia, and Kelly
def roy_age : ℕ := sorry
def julia_age : ℕ := sorry
def kelly_age : ℕ := sorry

-- Roy is 8 years older than Julia
axiom roy_julia_diff : roy_age = julia_age + 8

-- In 4 years, Roy will be twice as old as Julia
axiom roy_julia_future : roy_age + 4 = 2 * (julia_age + 4)

-- In 4 years, Roy's age multiplied by Kelly's age is 192
axiom roy_kelly_product : (roy_age + 4) * (kelly_age + 4) = 192

-- The ratio we want to prove
def age_diff_ratio : ℚ := (roy_age - julia_age) / (roy_age - kelly_age)

-- Theorem to prove
theorem age_diff_ratio_is_two_to_one : age_diff_ratio = 2 / 1 := by sorry

end NUMINAMATH_CALUDE_age_diff_ratio_is_two_to_one_l3317_331732


namespace NUMINAMATH_CALUDE_condition_one_condition_two_l3317_331718

-- Define the lines l₁ and l₂
def l₁ (a b : ℝ) (x y : ℝ) : Prop := a * x - b * y + 4 = 0
def l₂ (a b : ℝ) (x y : ℝ) : Prop := (a - 1) * x + y + b = 0

-- Define perpendicularity condition
def perpendicular (a b : ℝ) : Prop := a * (a - 1) - b = 0

-- Define parallel condition
def parallel (a b : ℝ) : Prop := a * (a - 1) + b = 0

-- Define the condition that l₁ passes through (-3, -1)
def passes_through (a b : ℝ) : Prop := l₁ a b (-3) (-1)

-- Define the condition that intercepts are equal
def equal_intercepts (a b : ℝ) : Prop := b = -a

theorem condition_one (a b : ℝ) :
  perpendicular a b ∧ passes_through a b → a = 2 ∧ b = 2 :=
by sorry

theorem condition_two (a b : ℝ) :
  parallel a b ∧ equal_intercepts a b → a = 2 ∧ b = -2 :=
by sorry

end NUMINAMATH_CALUDE_condition_one_condition_two_l3317_331718


namespace NUMINAMATH_CALUDE_multiples_of_five_l3317_331704

theorem multiples_of_five (a b : ℤ) (ha : 5 ∣ a) (hb : 10 ∣ b) : 5 ∣ b ∧ 5 ∣ (a - b) := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_five_l3317_331704


namespace NUMINAMATH_CALUDE_parabola_intersections_and_point_position_l3317_331746

/-- Represents a parabola of the form y = x^2 + px + q -/
structure Parabola where
  p : ℝ
  q : ℝ

/-- A point on the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem about parabola intersections and point position -/
theorem parabola_intersections_and_point_position 
  (parabola : Parabola) 
  (M : Point) 
  (h_below_x_axis : M.y < 0) :
  ∃ (x₁ x₂ : ℝ), 
    (x₁^2 + parabola.p * x₁ + parabola.q = 0) ∧ 
    (x₂^2 + parabola.p * x₂ + parabola.q = 0) ∧ 
    (x₁ < x₂) ∧
    (x₁ < M.x) ∧ (M.x < x₂) := by
  sorry


end NUMINAMATH_CALUDE_parabola_intersections_and_point_position_l3317_331746


namespace NUMINAMATH_CALUDE_cats_puppies_weight_difference_l3317_331784

/-- The number of puppies Hartley has -/
def num_puppies : ℕ := 4

/-- The weight of each puppy in kilograms -/
def puppy_weight : ℚ := 7.5

/-- The number of cats at the rescue center -/
def num_cats : ℕ := 14

/-- The weight of each cat in kilograms -/
def cat_weight : ℚ := 2.5

/-- The total weight of the puppies in kilograms -/
def total_puppy_weight : ℚ := num_puppies * puppy_weight

/-- The total weight of the cats in kilograms -/
def total_cat_weight : ℚ := num_cats * cat_weight

theorem cats_puppies_weight_difference :
  total_cat_weight - total_puppy_weight = 5 := by
  sorry

end NUMINAMATH_CALUDE_cats_puppies_weight_difference_l3317_331784


namespace NUMINAMATH_CALUDE_e_neg_4i_in_second_quadrant_l3317_331756

-- Define the complex exponential function
noncomputable def cexp (z : ℂ) : ℂ := Real.exp z.re * (Complex.cos z.im + Complex.I * Complex.sin z.im)

-- Define the quadrants of the complex plane
def in_second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

-- Theorem statement
theorem e_neg_4i_in_second_quadrant : 
  in_second_quadrant (cexp (-4 * Complex.I)) :=
sorry

end NUMINAMATH_CALUDE_e_neg_4i_in_second_quadrant_l3317_331756
