import Mathlib

namespace NUMINAMATH_CALUDE_minimum_points_tenth_game_l1478_147842

def first_five_games : List ℕ := [18, 15, 13, 17, 19]
def next_four_games : List ℕ := [14, 20, 12, 21]

def total_points_nine_games : ℕ := (first_five_games.sum + next_four_games.sum)

theorem minimum_points_tenth_game :
  ∀ x : ℕ, 
    (((total_points_nine_games + x) : ℚ) / 10 > 17) ∧ 
    (∀ y : ℕ, y < x → ((total_points_nine_games + y : ℚ) / 10 ≤ 17)) → 
    x = 22 :=
by sorry

end NUMINAMATH_CALUDE_minimum_points_tenth_game_l1478_147842


namespace NUMINAMATH_CALUDE_sum_of_ages_is_48_l1478_147825

/-- The sum of ages of 4 children with a 4-year age difference -/
def sum_of_ages (eldest_age : ℕ) : ℕ :=
  eldest_age + (eldest_age - 4) + (eldest_age - 8) + (eldest_age - 12)

/-- Theorem: The sum of ages of 4 children, where each child is born 4 years apart
    and the eldest is 18 years old, is equal to 48 years. -/
theorem sum_of_ages_is_48 : sum_of_ages 18 = 48 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_is_48_l1478_147825


namespace NUMINAMATH_CALUDE_root_difference_cubic_equation_l1478_147810

theorem root_difference_cubic_equation :
  ∃ (α β γ : ℝ),
    (81 * α^3 - 162 * α^2 + 90 * α - 10 = 0) ∧
    (81 * β^3 - 162 * β^2 + 90 * β - 10 = 0) ∧
    (81 * γ^3 - 162 * γ^2 + 90 * γ - 10 = 0) ∧
    (β = 2 * α ∨ γ = 2 * α ∨ γ = 2 * β) ∧
    (max α (max β γ) - min α (min β γ) = 1) :=
sorry

end NUMINAMATH_CALUDE_root_difference_cubic_equation_l1478_147810


namespace NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_plus_x_l1478_147826

theorem integral_sqrt_one_minus_x_squared_plus_x : 
  ∫ x in (-1)..1, (Real.sqrt (1 - x^2) + x) = π / 2 := by sorry

end NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_plus_x_l1478_147826


namespace NUMINAMATH_CALUDE_downward_parabola_m_range_l1478_147865

/-- A parabola that opens downwards -/
structure DownwardParabola where
  m : ℝ
  eq : ℝ → ℝ := fun x ↦ (m + 3) * x^2 + 1
  opens_downward : m + 3 < 0

/-- The range of m for a downward opening parabola -/
theorem downward_parabola_m_range (p : DownwardParabola) : p.m < -3 := by
  sorry

end NUMINAMATH_CALUDE_downward_parabola_m_range_l1478_147865


namespace NUMINAMATH_CALUDE_joes_lift_l1478_147891

theorem joes_lift (first_lift second_lift : ℝ)
  (h1 : first_lift + second_lift = 1800)
  (h2 : 2 * first_lift = second_lift + 300) :
  first_lift = 700 := by
sorry

end NUMINAMATH_CALUDE_joes_lift_l1478_147891


namespace NUMINAMATH_CALUDE_no_real_roots_l1478_147889

-- Define an arithmetic sequence
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + m) - a n = m * (a 1 - a 0)

-- Define the problem statement
theorem no_real_roots
  (a : ℕ → ℝ)
  (h_arithmetic : isArithmeticSequence a)
  (h_sum : a 2 + a 5 + a 8 = 9) :
  ∀ x : ℝ, x^2 + (a 4 + a 6) * x + 10 ≠ 0 :=
by
  sorry


end NUMINAMATH_CALUDE_no_real_roots_l1478_147889


namespace NUMINAMATH_CALUDE_tetrahedron_edges_form_triangles_l1478_147850

/-- Represents a tetrahedron with edge lengths a, b, c, d, e, f -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  d_pos : 0 < d
  e_pos : 0 < e
  f_pos : 0 < f
  vertex_sum_equal : a + b + c = b + d + f ∧ a + b + c = c + d + e ∧ a + b + c = a + e + f

theorem tetrahedron_edges_form_triangles (t : Tetrahedron) :
  (t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b) ∧
  (t.b + t.d > t.f ∧ t.d + t.f > t.b ∧ t.f + t.b > t.d) ∧
  (t.c + t.d > t.e ∧ t.d + t.e > t.c ∧ t.e + t.c > t.d) ∧
  (t.a + t.e > t.f ∧ t.e + t.f > t.a ∧ t.f + t.a > t.e) := by
  sorry


end NUMINAMATH_CALUDE_tetrahedron_edges_form_triangles_l1478_147850


namespace NUMINAMATH_CALUDE_log_product_equals_three_fourths_l1478_147849

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_product_equals_three_fourths :
  log 4 3 * log 9 8 = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_log_product_equals_three_fourths_l1478_147849


namespace NUMINAMATH_CALUDE_percentage_subtraction_l1478_147868

theorem percentage_subtraction (total : ℝ) (difference : ℝ) : 
  total = 8000 → 
  difference = 796 → 
  ∃ (P : ℝ), (1/10 * total) - (P/100 * total) = difference ∧ P = 5 := by
sorry

end NUMINAMATH_CALUDE_percentage_subtraction_l1478_147868


namespace NUMINAMATH_CALUDE_multiple_of_six_l1478_147841

theorem multiple_of_six (n : ℤ) : 
  (∃ k : ℤ, n = 6 * k) → (∃ m : ℤ, n = 2 * m) ∧ (∃ p : ℤ, n = 3 * p) := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_six_l1478_147841


namespace NUMINAMATH_CALUDE_plane_contains_line_and_point_l1478_147832

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a line in 3D space using parametric equations -/
structure ParametricLine3D where
  x : ℝ → ℝ
  y : ℝ → ℝ
  z : ℝ → ℝ

/-- Represents a plane in 3D space using the general equation Ax + By + Cz + D = 0 -/
structure Plane3D where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point lies on a plane -/
def pointOnPlane (plane : Plane3D) (point : Point3D) : Prop :=
  plane.A * point.x + plane.B * point.y + plane.C * point.z + plane.D = 0

/-- Check if a line lies on a plane -/
def lineOnPlane (plane : Plane3D) (line : ParametricLine3D) : Prop :=
  ∀ t : ℝ, plane.A * line.x t + plane.B * line.y t + plane.C * line.z t + plane.D = 0

/-- The given line -/
def givenLine : ParametricLine3D :=
  { x := λ t => 4 * t + 2
    y := λ t => -t - 1
    z := λ t => 5 * t + 2 }

/-- The given point -/
def givenPoint : Point3D :=
  { x := 2, y := -3, z := 3 }

/-- The plane to be proven -/
def resultPlane : Plane3D :=
  { A := 1, B := 14, C := 1, D := 18 }

theorem plane_contains_line_and_point :
  lineOnPlane resultPlane givenLine ∧
  pointOnPlane resultPlane givenPoint ∧
  resultPlane.A > 0 ∧
  Nat.gcd (Int.natAbs resultPlane.A) (Int.natAbs resultPlane.B) = 1 ∧
  Nat.gcd (Int.natAbs resultPlane.C) (Int.natAbs resultPlane.D) = 1 :=
sorry

end NUMINAMATH_CALUDE_plane_contains_line_and_point_l1478_147832


namespace NUMINAMATH_CALUDE_bond_interest_rate_proof_l1478_147802

/-- Proves that the interest rate of a bond is 5.75% given specific investment conditions -/
theorem bond_interest_rate_proof (total_investment : ℝ) (unknown_bond_investment : ℝ) 
  (known_bond_investment : ℝ) (known_interest_rate : ℝ) (desired_interest_income : ℝ) :
  total_investment = 32000 →
  unknown_bond_investment = 20000 →
  known_bond_investment = 12000 →
  known_interest_rate = 0.0625 →
  desired_interest_income = 1900 →
  ∃ unknown_interest_rate : ℝ,
    unknown_interest_rate = 0.0575 ∧
    desired_interest_income = unknown_bond_investment * unknown_interest_rate + 
                              known_bond_investment * known_interest_rate :=
by sorry

end NUMINAMATH_CALUDE_bond_interest_rate_proof_l1478_147802


namespace NUMINAMATH_CALUDE_angle_C_in_right_triangle_l1478_147822

-- Define a right triangle ABC
structure RightTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  right_angle : A = 90
  angle_sum : A + B + C = 180

-- Theorem statement
theorem angle_C_in_right_triangle (t : RightTriangle) (h : t.B = 50) : t.C = 40 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_in_right_triangle_l1478_147822


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1478_147871

theorem polynomial_divisibility (r s : ℝ) : 
  (∀ x : ℝ, (x - 2) * (x + 1) ∣ (x^6 - x^5 + 3*x^4 - r*x^3 + s*x^2 + 3*x - 7)) ↔ 
  (r = 33/4 ∧ s = -13/4) := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1478_147871


namespace NUMINAMATH_CALUDE_age_difference_proof_l1478_147895

/-- Represents a person with an age -/
structure Person where
  age : ℕ

/-- Calculates the age of a person after a given number of years -/
def age_after (p : Person) (years : ℕ) : ℕ := p.age + years

/-- Calculates the age of a person before a given number of years -/
def age_before (p : Person) (years : ℕ) : ℕ := p.age - years

/-- The problem statement -/
theorem age_difference_proof (john james james_brother : Person) 
    (h1 : john.age = 39)
    (h2 : age_before john 3 = 2 * age_after james 6)
    (h3 : james_brother.age = 16) :
  james_brother.age - james.age = 4 := by
  sorry


end NUMINAMATH_CALUDE_age_difference_proof_l1478_147895


namespace NUMINAMATH_CALUDE_quadratic_equation_conditions_l1478_147808

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 + a*x + 2 = 0
def q (a : ℝ) : Prop := ∀ x : ℝ, 0 < x → x < 1 → x^2 - a < 0

-- Define the set of real numbers that satisfy the conditions for p
def S₁ : Set ℝ := {a : ℝ | a ≤ -2 * Real.sqrt 2 ∨ a ≥ 2 * Real.sqrt 2}

-- Define the set of real numbers that satisfy the conditions for exactly one of p or q
def S₂ : Set ℝ := {a : ℝ | a ≤ -2 * Real.sqrt 2 ∨ (1 ≤ a ∧ a < 2 * Real.sqrt 2)}

-- State the theorem
theorem quadratic_equation_conditions (a : ℝ) :
  (p a ↔ a ∈ S₁) ∧
  ((p a ∧ ¬q a) ∨ (¬p a ∧ q a) ↔ a ∈ S₂) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_conditions_l1478_147808


namespace NUMINAMATH_CALUDE_value_of_3b_plus_4c_l1478_147885

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x + 2

-- Define the function f
def f (b c x : ℝ) : ℝ := b * x + c

-- State the theorem
theorem value_of_3b_plus_4c (b c : ℝ) :
  (∃ f_inv : ℝ → ℝ, 
    (∀ x, f b c (f_inv x) = x ∧ f_inv (f b c x) = x) ∧ 
    (∀ x, g x = 2 * f_inv x + 4)) →
  3 * b + 4 * c = 14/3 :=
by sorry

end NUMINAMATH_CALUDE_value_of_3b_plus_4c_l1478_147885


namespace NUMINAMATH_CALUDE_power_equation_l1478_147839

theorem power_equation (y : ℕ) : (2^13 : ℕ) - 2^y = 3 * 2^11 → y = 11 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_l1478_147839


namespace NUMINAMATH_CALUDE_min_cubes_for_specific_box_l1478_147855

/-- Calculates the minimum number of cubes required to build a box -/
def min_cubes_for_box (length width height cube_volume : ℕ) : ℕ :=
  (length * width * height + cube_volume - 1) / cube_volume

/-- Theorem stating the minimum number of cubes required for the specific box -/
theorem min_cubes_for_specific_box :
  min_cubes_for_box 10 18 4 12 = 60 := by
  sorry

#eval min_cubes_for_box 10 18 4 12

end NUMINAMATH_CALUDE_min_cubes_for_specific_box_l1478_147855


namespace NUMINAMATH_CALUDE_combined_research_degrees_l1478_147870

def total_percentage : ℝ := 100
def microphotonics_percentage : ℝ := 10
def home_electronics_percentage : ℝ := 24
def food_additives_percentage : ℝ := 15
def genetically_modified_microorganisms_percentage : ℝ := 29
def industrial_lubricants_percentage : ℝ := 8
def nanotechnology_percentage : ℝ := 7

def basic_astrophysics_percentage : ℝ :=
  total_percentage - (microphotonics_percentage + home_electronics_percentage + 
  food_additives_percentage + genetically_modified_microorganisms_percentage + 
  industrial_lubricants_percentage + nanotechnology_percentage)

def combined_percentage : ℝ := basic_astrophysics_percentage + nanotechnology_percentage

def degrees_in_circle : ℝ := 360

theorem combined_research_degrees :
  combined_percentage * (degrees_in_circle / total_percentage) = 50.4 := by
  sorry

end NUMINAMATH_CALUDE_combined_research_degrees_l1478_147870


namespace NUMINAMATH_CALUDE_expand_product_l1478_147875

theorem expand_product (x : ℝ) : 2 * (x - 3) * (x + 7) = 2 * x^2 + 8 * x - 42 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1478_147875


namespace NUMINAMATH_CALUDE_ab_value_l1478_147807

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 33) : a * b = 12 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1478_147807


namespace NUMINAMATH_CALUDE_probability_is_half_l1478_147860

/-- A circular field with six equally spaced radial roads -/
structure CircularField :=
  (radius : ℝ)
  (num_roads : ℕ)
  (h_num_roads : num_roads = 6)

/-- A geologist traveling on one of the roads -/
structure Geologist :=
  (speed : ℝ)
  (road : ℕ)
  (h_speed : speed = 5)
  (h_road : road ∈ Finset.range 6)

/-- The distance between two geologists after one hour -/
def distance (field : CircularField) (g1 g2 : Geologist) : ℝ :=
  sorry

/-- The probability of two geologists being more than 8 km apart -/
def probability (field : CircularField) : ℝ :=
  sorry

/-- Main theorem: The probability is 0.5 -/
theorem probability_is_half (field : CircularField) :
  probability field = 0.5 :=
sorry

end NUMINAMATH_CALUDE_probability_is_half_l1478_147860


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l1478_147840

theorem complex_modulus_equality (x : ℝ) (h : x > 0) :
  Complex.abs (6 + x * Complex.I) = 15 * Real.sqrt 2 ↔ x = Real.sqrt 414 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l1478_147840


namespace NUMINAMATH_CALUDE_largest_integer_in_interval_l1478_147817

theorem largest_integer_in_interval : 
  ∃ (x : ℤ), (1/4 : ℚ) < (x : ℚ)/7 ∧ (x : ℚ)/7 < 7/11 ∧ 
  ∀ (y : ℤ), ((1/4 : ℚ) < (y : ℚ)/7 ∧ (y : ℚ)/7 < 7/11) → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_in_interval_l1478_147817


namespace NUMINAMATH_CALUDE_triangle_incenter_properties_l1478_147890

/-- 
Given a right-angled triangle ABC with angle A = 90°, sides BC = a, AC = b, AB = c,
and a line d passing through the incenter intersecting AB at P and AC at Q.
-/
theorem triangle_incenter_properties 
  (a b c : ℝ) 
  (h_right_angle : a^2 = b^2 + c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (P Q : ℝ × ℝ) 
  (h_P_on_AB : P.1 ≥ 0 ∧ P.1 ≤ c ∧ P.2 = 0)
  (h_Q_on_AC : Q.1 = 0 ∧ Q.2 ≥ 0 ∧ Q.2 ≤ b)
  (h_PQ_through_incenter : ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ 
    P = (t * c, 0) ∧ 
    Q = (0, (1 - t) * b) ∧ 
    t * c / (a + b + c) = (1 - t) * b / (a + b + c)) :
  (b * (c - P.1) / P.1 + c * (b - Q.2) / Q.2 = a) ∧
  (∃ (m : ℝ), ∀ (x y : ℝ), 
    x ≥ 0 ∧ x ≤ c ∧ y ≥ 0 ∧ y ≤ b →
    ((c - x) / x)^2 + ((b - y) / y)^2 ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_incenter_properties_l1478_147890


namespace NUMINAMATH_CALUDE_triangle_third_side_minimum_l1478_147880

theorem triangle_third_side_minimum (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  (a - b = 5 ∨ b - a = 5) →
  Even (a + b + c) →
  c ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_triangle_third_side_minimum_l1478_147880


namespace NUMINAMATH_CALUDE_min_value_abc_min_value_abc_achieved_l1478_147819

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : 1/a + 1/b + 1/c = 9) :
  a^3 * b^2 * c ≥ 64/729 := by
  sorry

theorem min_value_abc_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  1/a + 1/b + 1/c = 9 ∧
  a^3 * b^2 * c < 64/729 + ε := by
  sorry

end NUMINAMATH_CALUDE_min_value_abc_min_value_abc_achieved_l1478_147819


namespace NUMINAMATH_CALUDE_dice_roll_sums_theorem_l1478_147866

/-- Represents the possible movements based on dice rolls -/
inductive Movement
  | West : Movement
  | East : Movement
  | North : Movement
  | South : Movement
  | Stay : Movement
  | NorthThree : Movement

/-- Converts a die roll to a movement -/
def rollToMovement (roll : Nat) : Movement :=
  match roll with
  | 1 => Movement.West
  | 2 => Movement.East
  | 3 => Movement.North
  | 4 => Movement.South
  | 5 => Movement.Stay
  | 6 => Movement.NorthThree
  | _ => Movement.Stay  -- Default case

/-- Represents a position in 2D space -/
structure Position where
  x : Int
  y : Int

/-- Updates the position based on a movement -/
def updatePosition (pos : Position) (mov : Movement) : Position :=
  match mov with
  | Movement.West => ⟨pos.x - 1, pos.y⟩
  | Movement.East => ⟨pos.x + 1, pos.y⟩
  | Movement.North => ⟨pos.x, pos.y + 1⟩
  | Movement.South => ⟨pos.x, pos.y - 1⟩
  | Movement.Stay => pos
  | Movement.NorthThree => ⟨pos.x, pos.y + 3⟩

/-- Calculates the final position after a sequence of rolls -/
def finalPosition (rolls : List Nat) : Position :=
  rolls.foldl (fun pos roll => updatePosition pos (rollToMovement roll)) ⟨0, 0⟩

/-- Theorem: Given the movement rules and final position of 1 km east,
    the possible sums of five dice rolls are 12, 15, 18, 22, and 25 -/
theorem dice_roll_sums_theorem (rolls : List Nat) :
  rolls.length = 5 ∧ 
  (finalPosition rolls).x = 1 ∧ 
  (finalPosition rolls).y = 0 →
  rolls.sum ∈ [12, 15, 18, 22, 25] :=
sorry


end NUMINAMATH_CALUDE_dice_roll_sums_theorem_l1478_147866


namespace NUMINAMATH_CALUDE_katie_cupcakes_made_l1478_147806

/-- The number of cupcakes Katie made after selling the first batch -/
def cupcakes_made_after (initial sold final : ℕ) : ℕ :=
  final - (initial - sold)

/-- Theorem: Katie made 20 cupcakes after selling the first batch -/
theorem katie_cupcakes_made :
  cupcakes_made_after 26 20 26 = 20 := by
  sorry

end NUMINAMATH_CALUDE_katie_cupcakes_made_l1478_147806


namespace NUMINAMATH_CALUDE_number_of_men_l1478_147844

/-- Proves that the number of men is 15 given the specified conditions -/
theorem number_of_men (men women boys : ℕ) (total_earnings men_wage : ℕ) : 
  men = women ∧ women = 8 ∧ 
  total_earnings = 120 ∧
  men_wage = 8 ∧
  total_earnings = men_wage * men →
  men = 15 := by
sorry

end NUMINAMATH_CALUDE_number_of_men_l1478_147844


namespace NUMINAMATH_CALUDE_number_of_children_l1478_147892

theorem number_of_children (pencils_per_child : ℕ) (total_pencils : ℕ) (h1 : pencils_per_child = 2) (h2 : total_pencils = 22) :
  total_pencils / pencils_per_child = 11 := by
  sorry

end NUMINAMATH_CALUDE_number_of_children_l1478_147892


namespace NUMINAMATH_CALUDE_sheridan_cats_l1478_147830

/-- The number of cats Mrs. Sheridan has after giving some away -/
def remaining_cats (initial : Float) (given_away : Float) : Float :=
  initial - given_away

/-- Theorem: Mrs. Sheridan has 3.0 cats after giving away 14.0 cats from her initial 17.0 cats -/
theorem sheridan_cats : remaining_cats 17.0 14.0 = 3.0 := by
  sorry

end NUMINAMATH_CALUDE_sheridan_cats_l1478_147830


namespace NUMINAMATH_CALUDE_inequality_proof_l1478_147818

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) : 
  a + b + Real.sqrt 2 * c ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1478_147818


namespace NUMINAMATH_CALUDE_conic_is_ellipse_l1478_147824

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  x^2 + 3*y^2 - 6*x - 12*y + 9 = 0

/-- The standard form of an ellipse equation -/
def is_ellipse (h k a b : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

/-- Theorem stating that the given equation represents an ellipse -/
theorem conic_is_ellipse :
  ∃ (h k a b : ℝ), ∀ (x y : ℝ),
    conic_equation x y ↔ is_ellipse h k a b x y :=
sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_l1478_147824


namespace NUMINAMATH_CALUDE_modulus_of_complex_expression_l1478_147876

theorem modulus_of_complex_expression :
  let z : ℂ := (1 : ℂ) / (1 + Complex.I) + Complex.I
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_expression_l1478_147876


namespace NUMINAMATH_CALUDE_f_properties_l1478_147809

noncomputable def f (x : ℝ) : ℝ := (Real.sin (2 * x) - Real.cos (2 * x) + 1) / (2 * Real.sin x)

theorem f_properties :
  (∃ (S : Set ℝ), S = {x : ℝ | ∀ k : ℤ, x ≠ k * Real.pi} ∧ (∀ x : ℝ, x ∈ S ↔ f x ≠ 0)) ∧
  (Set.range f = Set.Icc (-Real.sqrt 2) (-1) ∪ Set.Ioo (-1) 1 ∪ Set.Icc 1 (Real.sqrt 2)) ∧
  (∀ α : ℝ, 0 < α ∧ α < Real.pi / 2 → Real.tan (α / 2) = 1 / 2 → f α = 7 / 5) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1478_147809


namespace NUMINAMATH_CALUDE_tangent_and_decreasing_interval_l1478_147805

-- Define the function f
def f (m n : ℝ) (x : ℝ) : ℝ := m * x^3 + n * x^2

-- Define the derivative of f
def f_derivative (m n : ℝ) (x : ℝ) : ℝ := 3 * m * x^2 + 2 * n * x

-- Theorem statement
theorem tangent_and_decreasing_interval 
  (m n : ℝ) 
  (h1 : f m n (-1) = 2)
  (h2 : f_derivative m n (-1) = -3)
  (h3 : ∀ t : ℝ, ∀ x ∈ Set.Icc t (t + 1), 
        f_derivative m n x ≤ 0 → 
        -2 ≤ t ∧ t ≤ -1) :
  ∀ t : ℝ, (∀ x ∈ Set.Icc t (t + 1), f_derivative m n x ≤ 0) → 
    t ∈ Set.Icc (-2) (-1) :=
sorry

end NUMINAMATH_CALUDE_tangent_and_decreasing_interval_l1478_147805


namespace NUMINAMATH_CALUDE_simplify_tan_product_l1478_147803

theorem simplify_tan_product (tan30 tan15 : ℝ) : 
  tan30 + tan15 = 1 - tan30 * tan15 → (1 + tan30) * (1 + tan15) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_tan_product_l1478_147803


namespace NUMINAMATH_CALUDE_max_volume_triangular_pyramid_l1478_147893

/-- A triangular pyramid with vertex S and base ABC -/
structure TriangularPyramid where
  SA : ℝ
  SB : ℝ
  SC : ℝ
  AB : ℝ
  BC : ℝ
  AC : ℝ

/-- The volume of a triangular pyramid -/
def volume (t : TriangularPyramid) : ℝ := sorry

/-- The conditions given in the problem -/
def satisfiesConditions (t : TriangularPyramid) : Prop :=
  t.SA = 4 ∧
  t.SB ≥ 7 ∧
  t.SC ≥ 9 ∧
  t.AB = 5 ∧
  t.BC ≤ 6 ∧
  t.AC ≤ 8

/-- The theorem stating the maximum volume of the triangular pyramid -/
theorem max_volume_triangular_pyramid :
  ∀ t : TriangularPyramid, satisfiesConditions t → volume t ≤ 8 * Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_max_volume_triangular_pyramid_l1478_147893


namespace NUMINAMATH_CALUDE_correlation_identification_l1478_147836

-- Define the relationships
inductive Relationship
| AgeAndFat
| CurvePoints
| FruitProduction
| StudentAndID

-- Define the property of having a correlation
def HasCorrelation : Relationship → Prop :=
  fun r => match r with
  | Relationship.AgeAndFat => true
  | Relationship.CurvePoints => false
  | Relationship.FruitProduction => true
  | Relationship.StudentAndID => false

-- Define the property of being a functional relationship
def IsFunctionalRelationship : Relationship → Prop :=
  fun r => match r with
  | Relationship.AgeAndFat => false
  | Relationship.CurvePoints => true
  | Relationship.FruitProduction => false
  | Relationship.StudentAndID => true

-- Theorem statement
theorem correlation_identification :
  (∀ r : Relationship, HasCorrelation r ↔ ¬(IsFunctionalRelationship r)) ∧
  (HasCorrelation Relationship.AgeAndFat ∧ HasCorrelation Relationship.FruitProduction) ∧
  (¬HasCorrelation Relationship.CurvePoints ∧ ¬HasCorrelation Relationship.StudentAndID) :=
by sorry

end NUMINAMATH_CALUDE_correlation_identification_l1478_147836


namespace NUMINAMATH_CALUDE_find_n_l1478_147815

def is_valid_n (n : ℕ) : Prop :=
  n > 0 ∧
  ∃ r : ℕ, r > 0 ∧ 
    (2287 % n = r) ∧ 
    (2028 % n = r) ∧ 
    (1806 % n = r)

theorem find_n : 
  (∀ m : ℕ, is_valid_n m → m ≤ 37) ∧ 
  is_valid_n 37 :=
sorry

end NUMINAMATH_CALUDE_find_n_l1478_147815


namespace NUMINAMATH_CALUDE_min_value_2x_plus_y_l1478_147878

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + 2*x*y - 3 = 0) :
  2*x + y ≥ 3 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀^2 + 2*x₀*y₀ - 3 = 0 ∧ 2*x₀ + y₀ = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_2x_plus_y_l1478_147878


namespace NUMINAMATH_CALUDE_smallest_y_for_inequality_l1478_147863

theorem smallest_y_for_inequality : ∃ (y : ℕ), y > 0 ∧ (y^6 : ℚ) / (y^3 : ℚ) > 80 ∧ ∀ (z : ℕ), z > 0 → (z^6 : ℚ) / (z^3 : ℚ) > 80 → y ≤ z :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_for_inequality_l1478_147863


namespace NUMINAMATH_CALUDE_exists_universal_friend_l1478_147888

-- Define a type for people
variable {Person : Type}

-- Define the friendship relation
variable (friends : Person → Person → Prop)

-- Define the property that every two people have exactly one friend in common
def one_common_friend (friends : Person → Person → Prop) : Prop :=
  ∀ a b : Person, a ≠ b →
    ∃! c : Person, friends a c ∧ friends b c

-- State the theorem
theorem exists_universal_friend
  [Finite Person]
  (h : one_common_friend friends) :
  ∃ x : Person, ∀ y : Person, y ≠ x → friends x y :=
sorry

end NUMINAMATH_CALUDE_exists_universal_friend_l1478_147888


namespace NUMINAMATH_CALUDE_rectangle_area_with_circles_l1478_147838

/-- The area of a rectangle with specific circle arrangement -/
theorem rectangle_area_with_circles (d : ℝ) (w l : ℝ) : 
  d = 6 →                    -- diameter of each circle
  w = 3 * d →                -- width equals total diameter of three circles
  l = 2 * w →                -- length is twice the width
  w * l = 648 := by           -- area of the rectangle
  sorry

#check rectangle_area_with_circles

end NUMINAMATH_CALUDE_rectangle_area_with_circles_l1478_147838


namespace NUMINAMATH_CALUDE_sin_alpha_value_l1478_147896

theorem sin_alpha_value (α : Real) 
  (h1 : 2 * Real.tan α * Real.sin α = 3)
  (h2 : -π/2 < α ∧ α < 0) : 
  Real.sin α = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l1478_147896


namespace NUMINAMATH_CALUDE_triangle_area_l1478_147843

theorem triangle_area (a b c : ℝ) (h1 : a = 14) (h2 : b = 48) (h3 : c = 50) :
  (1/2) * a * b = 336 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1478_147843


namespace NUMINAMATH_CALUDE_log_equality_l1478_147862

theorem log_equality (y : ℝ) (m : ℝ) : 
  (Real.log 5 / Real.log 8 = y) → 
  (Real.log 125 / Real.log 2 = m * y) → 
  m = 9 := by
sorry

end NUMINAMATH_CALUDE_log_equality_l1478_147862


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1478_147813

/-- An isosceles triangle with sides of 4 cm and 7 cm has a perimeter of either 15 cm or 18 cm. -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a = 4 ∧ b = 7 ∧ 
  ((a = b ∧ c = 7) ∨ (a = c ∧ b = 7) ∨ (b = c ∧ a = 4)) → 
  (a + b + c = 15 ∨ a + b + c = 18) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1478_147813


namespace NUMINAMATH_CALUDE_value_of_expression_l1478_147864

theorem value_of_expression (m n : ℤ) (h : m - n = 1) : (m - n)^2 - 2*m + 2*n = -1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1478_147864


namespace NUMINAMATH_CALUDE_base_conversion_l1478_147872

theorem base_conversion (b : ℕ) (h1 : b > 0) : 
  (5 * 6 + 2 = b * b + b + 1) → b = 5 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_l1478_147872


namespace NUMINAMATH_CALUDE_min_weighings_eq_counterfeit_problem_weighings_l1478_147848

/-- Represents a coin collection with genuine and counterfeit coins. -/
structure CoinCollection where
  total : ℕ
  genuine : ℕ
  counterfeit : ℕ
  genuine_weight : ℝ
  counterfeit_weights : Finset ℝ
  h_total : total = genuine + counterfeit
  h_genuine_lt_counterfeit : genuine < counterfeit
  h_counterfeit_heavier : ∀ w ∈ counterfeit_weights, w > genuine_weight
  h_counterfeit_distinct : counterfeit_weights.card = counterfeit

/-- Represents a weighing on a balance scale. -/
def Weighing (c : CoinCollection) := Finset (Fin c.total)

/-- The result of a weighing is either balanced or unbalanced. -/
inductive WeighingResult
  | balanced
  | unbalanced

/-- Performs a weighing and returns the result. -/
def performWeighing (c : CoinCollection) (w : Weighing c) : WeighingResult :=
  sorry

/-- Theorem: The minimum number of weighings needed to guarantee finding a genuine coin is equal to the number of counterfeit coins. -/
theorem min_weighings_eq_counterfeit (c : CoinCollection) :
  (∃ n : ℕ, ∀ m : ℕ, (∃ (weighings : Fin m → Weighing c), 
    ∃ (i : Fin m), performWeighing c (weighings i) = WeighingResult.balanced) ↔ m ≥ n) ∧ 
  (∀ k : ℕ, k < c.counterfeit → 
    ∃ (weighings : Fin k → Weighing c), ∀ i : Fin k, 
      performWeighing c (weighings i) = WeighingResult.unbalanced) :=
  sorry

/-- The specific coin collection from the problem. -/
def problemCollection : CoinCollection where
  total := 100
  genuine := 30
  counterfeit := 70
  genuine_weight := 1
  counterfeit_weights := sorry
  h_total := rfl
  h_genuine_lt_counterfeit := by norm_num
  h_counterfeit_heavier := sorry
  h_counterfeit_distinct := sorry

/-- The main theorem: 70 weighings are needed for the problem collection. -/
theorem problem_weighings :
  (∃ n : ℕ, ∀ m : ℕ, (∃ (weighings : Fin m → Weighing problemCollection), 
    ∃ (i : Fin m), performWeighing problemCollection (weighings i) = WeighingResult.balanced) ↔ m ≥ n) ∧
  n = 70 :=
  sorry

end NUMINAMATH_CALUDE_min_weighings_eq_counterfeit_problem_weighings_l1478_147848


namespace NUMINAMATH_CALUDE_classroom_shirts_and_shorts_l1478_147831

theorem classroom_shirts_and_shorts (total_students : ℕ) 
  (h1 : total_students = 81)
  (h2 : ∃ striped_shirts : ℕ, striped_shirts = (2 * total_students) / 3)
  (h3 : ∃ checkered_shirts : ℕ, checkered_shirts = total_students - striped_shirts)
  (h4 : ∃ shorts : ℕ, shorts > checkered_shirts)
  (h5 : ∃ striped_shirts shorts : ℕ, striped_shirts = shorts + 8) :
  ∃ shorts checkered_shirts : ℕ, shorts = checkered_shirts + 19 := by
  sorry

end NUMINAMATH_CALUDE_classroom_shirts_and_shorts_l1478_147831


namespace NUMINAMATH_CALUDE_trig_identity_l1478_147854

theorem trig_identity (α : Real) (h : (1 + Real.tan α) / (1 - Real.tan α) = 2012) :
  1 / Real.cos (2 * α) + Real.tan (2 * α) = 2012 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1478_147854


namespace NUMINAMATH_CALUDE_expression_values_l1478_147828

theorem expression_values (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ (x : ℝ), x ∈ ({-4, 0, 4} : Set ℝ) ∧
  x = a / |a| + b / |b| + c / |c| + (a * b * c) / |a * b * c| :=
by sorry

end NUMINAMATH_CALUDE_expression_values_l1478_147828


namespace NUMINAMATH_CALUDE_symmetric_points_a_value_l1478_147852

/-- Two points are symmetric with respect to the origin if their coordinates sum to zero -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 + q.1 = 0 ∧ p.2 + q.2 = 0

/-- Given that point A(a,1) is symmetric to point B(-3,-1) with respect to the origin, prove that a = 3 -/
theorem symmetric_points_a_value :
  ∀ a : ℝ, symmetric_wrt_origin (a, 1) (-3, -1) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_a_value_l1478_147852


namespace NUMINAMATH_CALUDE_triangle_special_condition_l1478_147847

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and area S, if 4√3S = (a+b)² - c², then the measure of angle C is π/3 -/
theorem triangle_special_condition (a b c S : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (area_eq : 4 * Real.sqrt 3 * S = (a + b)^2 - c^2) :
  ∃ (A B C : ℝ), 
    0 < A ∧ 0 < B ∧ 0 < C ∧ 
    A + B + C = π ∧
    S = 1/2 * a * b * Real.sin C ∧
    c^2 = a^2 + b^2 - 2*a*b*Real.cos C ∧
    C = π/3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_special_condition_l1478_147847


namespace NUMINAMATH_CALUDE_car_cost_proof_l1478_147887

/-- The cost of Gary's used car -/
def car_cost : ℝ := 6000

/-- The monthly payment difference between 2-year and 5-year loans -/
def monthly_difference : ℝ := 150

/-- The number of months in 2 years -/
def months_in_2_years : ℝ := 2 * 12

/-- The number of months in 5 years -/
def months_in_5_years : ℝ := 5 * 12

theorem car_cost_proof :
  (car_cost / months_in_2_years) - (car_cost / months_in_5_years) = monthly_difference :=
sorry

end NUMINAMATH_CALUDE_car_cost_proof_l1478_147887


namespace NUMINAMATH_CALUDE_triangle_theorem_l1478_147873

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a^2 + t.c^2 - t.b^2 = Real.sqrt 3 * t.a * t.c)
  (h2 : 2 * t.b * Real.cos t.A = Real.sqrt 3 * (t.c * Real.cos t.A + t.a * Real.cos t.C))
  (h3 : (t.a^2 + t.b^2 + t.c^2 - (t.b^2 + t.c^2 - t.a^2) / 2) / 4 = 7) :
  t.B = π / 6 ∧ t.a * t.b * Real.sin t.C / 2 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1478_147873


namespace NUMINAMATH_CALUDE_katies_new_games_l1478_147897

/-- Katie's new games problem -/
theorem katies_new_games :
  ∀ (k : ℕ),  -- k represents Katie's new games
  (k + 8 = 92) →  -- Total new games between Katie and her friends is 92
  (k = 84)  -- Katie has 84 new games
  := by sorry

end NUMINAMATH_CALUDE_katies_new_games_l1478_147897


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l1478_147881

theorem fixed_point_on_line (m : ℝ) : 
  (3*m + 4) * (-1) + (5 - 2*m) * 2 + 7*m - 6 = 0 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l1478_147881


namespace NUMINAMATH_CALUDE_cookie_consumption_l1478_147801

theorem cookie_consumption (total cookies_left father_ate : ℕ) 
  (h1 : total = 30)
  (h2 : cookies_left = 8)
  (h3 : father_ate = 10) :
  let mother_ate := father_ate / 2
  let total_eaten := total - cookies_left
  let brother_ate := total_eaten - (father_ate + mother_ate)
  brother_ate - mother_ate = 2 := by sorry

end NUMINAMATH_CALUDE_cookie_consumption_l1478_147801


namespace NUMINAMATH_CALUDE_negation_of_universal_quadrilateral_circumcircle_l1478_147821

-- Define the type for quadrilaterals
variable (Quadrilateral : Type)

-- Define the property of having a circumcircle
variable (has_circumcircle : Quadrilateral → Prop)

-- Theorem stating the negation of "Every quadrilateral has a circumcircle"
-- is equivalent to "Some quadrilaterals do not have a circumcircle"
theorem negation_of_universal_quadrilateral_circumcircle :
  ¬(∀ q : Quadrilateral, has_circumcircle q) ↔ ∃ q : Quadrilateral, ¬(has_circumcircle q) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_quadrilateral_circumcircle_l1478_147821


namespace NUMINAMATH_CALUDE_remainder_problem_l1478_147874

theorem remainder_problem (n : ℕ) 
  (h1 : n^2 % 5 = 4)
  (h2 : n^3 % 5 = 2) : 
  n % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1478_147874


namespace NUMINAMATH_CALUDE_sum_of_squares_over_factorial_l1478_147804

theorem sum_of_squares_over_factorial : (1^2 + 2^2 + 3^2 + 4^2) / (1 * 2 * 3) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_over_factorial_l1478_147804


namespace NUMINAMATH_CALUDE_bird_island_injured_parrots_l1478_147867

/-- The number of parrots on Bird Island -/
def total_parrots : ℕ := 105

/-- The fraction of parrots that are green -/
def green_fraction : ℚ := 5/7

/-- The percentage of green parrots that are injured -/
def injured_percentage : ℚ := 3/100

/-- The number of injured green parrots -/
def injured_green_parrots : ℕ := 2

theorem bird_island_injured_parrots :
  ⌊(total_parrots : ℚ) * green_fraction * injured_percentage⌋ = injured_green_parrots := by
  sorry

end NUMINAMATH_CALUDE_bird_island_injured_parrots_l1478_147867


namespace NUMINAMATH_CALUDE_lines_cannot_form_triangle_iff_m_in_set_l1478_147814

/-- A line in the plane, represented by its equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if three lines form a triangle -/
def form_triangle (l₁ l₂ l₃ : Line) : Prop :=
  sorry

/-- The set of m values for which the lines cannot form a triangle -/
def invalid_m_values : Set ℝ :=
  {4, -1/6, -1, 2/3}

theorem lines_cannot_form_triangle_iff_m_in_set (m : ℝ) :
  let l₁ : Line := ⟨4, 1, 4⟩
  let l₂ : Line := ⟨m, 1, 0⟩
  let l₃ : Line := ⟨2, -3*m, 4⟩
  ¬(form_triangle l₁ l₂ l₃) ↔ m ∈ invalid_m_values :=
by sorry

end NUMINAMATH_CALUDE_lines_cannot_form_triangle_iff_m_in_set_l1478_147814


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1478_147894

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- Positive terms
  (q ≠ 1) →  -- Common ratio not equal to 1
  (∀ n, a (n + 1) = q * a n) →  -- Geometric sequence definition
  (a 2 - (1/2 * a 3) = (1/2 * a 3) - a 1) →  -- Arithmetic sequence condition
  ((a 4 + a 5) / (a 3 + a 4) = (1 + Real.sqrt 5) / 2) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1478_147894


namespace NUMINAMATH_CALUDE_range_of_a_l1478_147899

theorem range_of_a (x a : ℝ) : 
  (∀ x, (1/2 ≤ x ∧ x ≤ 1) → (a ≤ x ∧ x ≤ a + 1)) ∧
  (∃ x, (1/2 ≤ x ∧ x ≤ 1) ∧ ¬(a ≤ x ∧ x ≤ a + 1)) →
  (0 ≤ a ∧ a ≤ 1/2) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1478_147899


namespace NUMINAMATH_CALUDE_breakfast_calories_proof_l1478_147853

/-- Calculates the breakfast calories given the daily calorie limit, remaining calories, dinner calories, and lunch calories. -/
def breakfast_calories (daily_limit : ℕ) (remaining : ℕ) (dinner : ℕ) (lunch : ℕ) : ℕ :=
  daily_limit - remaining - (dinner + lunch)

/-- Proves that given the specific calorie values, the breakfast calories are 560. -/
theorem breakfast_calories_proof :
  breakfast_calories 2500 525 635 780 = 560 := by
  sorry

end NUMINAMATH_CALUDE_breakfast_calories_proof_l1478_147853


namespace NUMINAMATH_CALUDE_factorization_of_difference_of_squares_not_factorization_expansion_not_complete_factorization_not_factorization_expansion_2_l1478_147823

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 4 = (x - 2) * (x + 2) :=
by sorry

theorem not_factorization_expansion (a : ℝ) :
  (a - 1)^2 = a^2 - 2*a + 1 :=
by sorry

theorem not_complete_factorization (x : ℝ) :
  x^2 - 2*x - 6 = x*(x - 2) - 6 :=
by sorry

theorem not_factorization_expansion_2 (x : ℝ) :
  x*(x - 1) = x^2 - x :=
by sorry

end NUMINAMATH_CALUDE_factorization_of_difference_of_squares_not_factorization_expansion_not_complete_factorization_not_factorization_expansion_2_l1478_147823


namespace NUMINAMATH_CALUDE_linear_equation_solution_l1478_147857

theorem linear_equation_solution (x y : ℝ) : 5 * x + y = 4 → y = 4 - 5 * x := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l1478_147857


namespace NUMINAMATH_CALUDE_alice_wins_second_attempt_prob_l1478_147859

-- Define the number of cards in the deck
def deckSize : ℕ := 20

-- Define the probability of a correct guess in each turn
def probFirst : ℚ := 1 / deckSize
def probSecond : ℚ := 1 / (deckSize - 1)
def probThird : ℚ := 1 / (deckSize - 2)

-- Define the probability of Alice winning on her second attempt
def aliceWinsSecondAttempt : ℚ := (1 - probFirst) * (1 - probSecond) * probThird

-- Theorem to prove
theorem alice_wins_second_attempt_prob :
  aliceWinsSecondAttempt = 1 / deckSize := by
  sorry


end NUMINAMATH_CALUDE_alice_wins_second_attempt_prob_l1478_147859


namespace NUMINAMATH_CALUDE_sqrt_sum_diff_equals_fifteen_halves_l1478_147837

theorem sqrt_sum_diff_equals_fifteen_halves :
  Real.sqrt 9 + Real.sqrt 25 - Real.sqrt (1/4) = 15/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_diff_equals_fifteen_halves_l1478_147837


namespace NUMINAMATH_CALUDE_max_sum_of_integers_l1478_147845

theorem max_sum_of_integers (a c d : ℤ) (b : ℕ+) 
  (eq1 : a + b = c) 
  (eq2 : b + c = d) 
  (eq3 : c + d = a) : 
  a + b + c + d ≤ -5 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_integers_l1478_147845


namespace NUMINAMATH_CALUDE_f_nonnegative_range_l1478_147812

def f (a x : ℝ) : ℝ := a * x^2 - (a + 1) * x + 2

theorem f_nonnegative_range (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 3, f a x ≥ 0) →
  1/6 ≤ a ∧ a ≤ 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_f_nonnegative_range_l1478_147812


namespace NUMINAMATH_CALUDE_negation_equivalence_exists_false_conjunction_true_component_negation_implication_l1478_147877

-- Define the propositions
def p : Prop := ∃ x : ℝ, x^2 + x - 1 < 0
def q : Prop := ∃ x : ℝ, x^2 - 3*x + 2 = 0

-- Statement 1
theorem negation_equivalence : (¬p) ↔ (∀ x : ℝ, x^2 + x - 1 ≥ 0) := by sorry

-- Statement 2
theorem exists_false_conjunction_true_component :
  ∃ (p q : Prop), ¬(p ∧ q) ∧ (p ∨ q) := by sorry

-- Statement 3
theorem negation_implication :
  ¬(q → ∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x = 2) ≠
  (q → ∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x ≠ 2) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_exists_false_conjunction_true_component_negation_implication_l1478_147877


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l1478_147833

theorem gcd_lcm_sum : Nat.gcd 48 70 + Nat.lcm 18 45 = 92 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l1478_147833


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1478_147883

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, (m + 1) * x^2 - m * x + (m - 1) ≥ 0) → m ≥ 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1478_147883


namespace NUMINAMATH_CALUDE_soda_count_l1478_147834

/-- Proves that given 2 sandwiches at $2.49 each and some sodas at $1.87 each,
    if the total cost is $12.46, then the number of sodas purchased is 4. -/
theorem soda_count (sandwich_cost soda_cost total_cost : ℚ) (sandwich_count : ℕ) :
  sandwich_cost = 249/100 →
  soda_cost = 187/100 →
  total_cost = 1246/100 →
  sandwich_count = 2 →
  ∃ (soda_count : ℕ), soda_count = 4 ∧
    sandwich_count * sandwich_cost + soda_count * soda_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_soda_count_l1478_147834


namespace NUMINAMATH_CALUDE_quadratic_factor_value_l1478_147861

-- Define the polynomials
def f (x : ℝ) : ℝ := x^4 + 8*x^3 + 18*x^2 + 8*x + 35
def g (x : ℝ) : ℝ := 2*x^4 - 4*x^3 + x^2 + 26*x + 10

-- Define the quadratic polynomial q
def q (d e : ℤ) (x : ℝ) : ℝ := x^2 + d*x + e

-- State the theorem
theorem quadratic_factor_value (d e : ℤ) :
  (∃ (p₁ p₂ : ℝ → ℝ), f = q d e * p₁ ∧ g = q d e * p₂) →
  q d e 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factor_value_l1478_147861


namespace NUMINAMATH_CALUDE_repetend_of_five_elevenths_l1478_147829

/-- The decimal representation of 5/11 has a repetend of 45. -/
theorem repetend_of_five_elevenths : ∃ (a b : ℕ), 
  (5 : ℚ) / 11 = (a : ℚ) / 100 + (b : ℚ) / 99 * (1 / 100) ∧ b = 45 := by
  sorry

end NUMINAMATH_CALUDE_repetend_of_five_elevenths_l1478_147829


namespace NUMINAMATH_CALUDE_race_distance_l1478_147827

-- Define the race distance
variable (d : ℝ)

-- Define the speeds of A, B, and C
variable (a b c : ℝ)

-- Define the conditions of the race
variable (h1 : d / a = (d - 30) / b)
variable (h2 : d / b = (d - 15) / c)
variable (h3 : d / a = (d - 40) / c)

-- The theorem to prove
theorem race_distance : d = 90 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_l1478_147827


namespace NUMINAMATH_CALUDE_alternating_arrangements_adjacent_ab_arrangements_l1478_147846

/-- Represents the number of male students -/
def num_male : Nat := 2

/-- Represents the number of female students -/
def num_female : Nat := 3

/-- Represents the total number of students -/
def total_students : Nat := num_male + num_female

/-- Calculates the number of ways to arrange n distinct objects -/
def arrangements (n : Nat) : Nat := Nat.factorial n

/-- Theorem stating the number of alternating arrangements -/
theorem alternating_arrangements : 
  arrangements num_male * arrangements num_female = 12 := by sorry

/-- Theorem stating the number of arrangements with A and B adjacent -/
theorem adjacent_ab_arrangements : 
  arrangements (total_students - 1) * 2 = 48 := by sorry

end NUMINAMATH_CALUDE_alternating_arrangements_adjacent_ab_arrangements_l1478_147846


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l1478_147835

theorem system_of_equations_solution :
  ∃! (x y : ℝ), (6 * x - 3 * y = -3) ∧ (5 * x - 9 * y = -35) ∧ x = 2 ∧ y = 5 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l1478_147835


namespace NUMINAMATH_CALUDE_return_percentage_is_80_percent_l1478_147879

/-- Represents the library book collection and loan statistics --/
structure LibraryStats where
  initial_books : ℕ
  final_books : ℕ
  loaned_books : ℕ

/-- Calculates the percentage of loaned books that were returned --/
def return_percentage (stats : LibraryStats) : ℚ :=
  ((stats.final_books - (stats.initial_books - stats.loaned_books)) / stats.loaned_books) * 100

/-- Theorem stating that the return percentage is 80% for the given statistics --/
theorem return_percentage_is_80_percent (stats : LibraryStats) 
  (h1 : stats.initial_books = 75)
  (h2 : stats.final_books = 64)
  (h3 : stats.loaned_books = 55) : 
  return_percentage stats = 80 := by
  sorry

end NUMINAMATH_CALUDE_return_percentage_is_80_percent_l1478_147879


namespace NUMINAMATH_CALUDE_anthony_balloons_l1478_147858

theorem anthony_balloons (tom_balloons luke_balloons anthony_balloons : ℕ) :
  tom_balloons = 3 * luke_balloons →
  luke_balloons = anthony_balloons / 4 →
  tom_balloons = 33 →
  anthony_balloons = 44 :=
by sorry

end NUMINAMATH_CALUDE_anthony_balloons_l1478_147858


namespace NUMINAMATH_CALUDE_smallest_k_for_integer_product_l1478_147816

def a : ℕ → ℝ
  | 0 => 1
  | 1 => 3 ^ (1 / 17)
  | (n + 2) => a (n + 1) * (a n) ^ 2

def product_up_to (k : ℕ) : ℝ :=
  (List.range k).foldl (λ acc i => acc * a (i + 1)) 1

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

theorem smallest_k_for_integer_product :
  (∀ k < 11, ¬ is_integer (product_up_to k)) ∧
  is_integer (product_up_to 11) := by sorry

end NUMINAMATH_CALUDE_smallest_k_for_integer_product_l1478_147816


namespace NUMINAMATH_CALUDE_random_events_identification_l1478_147820

structure Event where
  description : String
  is_random : Bool

def event1 : Event := { 
  description := "An object will fall freely under the influence of gravity alone",
  is_random := false 
}

def event2 : Event := { 
  description := "The equation x^2 + 2x + 8 = 0 has two real roots",
  is_random := false 
}

def event3 : Event := { 
  description := "A certain information desk receives more than 10 requests for information consultation during a certain period of the day",
  is_random := true 
}

def event4 : Event := { 
  description := "It will rain next Saturday",
  is_random := true 
}

def events : List Event := [event1, event2, event3, event4]

theorem random_events_identification : 
  (events.filter (λ e => e.is_random)).map (λ e => e.description) = 
  [event3.description, event4.description] := by sorry

end NUMINAMATH_CALUDE_random_events_identification_l1478_147820


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l1478_147811

theorem max_value_trig_expression (x y z : ℝ) :
  (Real.sin (2 * x) + Real.sin y + Real.sin (3 * z)) *
  (Real.cos (2 * x) + Real.cos y + Real.cos (3 * z)) ≤ 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l1478_147811


namespace NUMINAMATH_CALUDE_megans_books_l1478_147800

theorem megans_books (books_per_shelf : ℕ) (mystery_shelves : ℕ) (picture_shelves : ℕ) 
  (h1 : books_per_shelf = 7)
  (h2 : mystery_shelves = 8)
  (h3 : picture_shelves = 2) :
  books_per_shelf * (mystery_shelves + picture_shelves) = 70 :=
by sorry

end NUMINAMATH_CALUDE_megans_books_l1478_147800


namespace NUMINAMATH_CALUDE_harmonious_example_harmonious_rational_sum_harmonious_rational_ratio_l1478_147884

/-- A pair of real numbers (a, b) is harmonious if a^2 + b and a + b^2 are both rational. -/
def Harmonious (a b : ℝ) : Prop :=
  (∃ q₁ : ℚ, a^2 + b = q₁) ∧ (∃ q₂ : ℚ, a + b^2 = q₂)

theorem harmonious_example :
  Harmonious (Real.sqrt 2 + 1/2) (1/2 - Real.sqrt 2) := by sorry

theorem harmonious_rational_sum {a b : ℝ} (h : Harmonious a b) (hs : ∃ q : ℚ, a + b = q) (hne : a + b ≠ 1) :
  ∃ (q₁ q₂ : ℚ), a = q₁ ∧ b = q₂ := by sorry

theorem harmonious_rational_ratio {a b : ℝ} (h : Harmonious a b) (hr : ∃ q : ℚ, a = q * b) :
  ∃ (q₁ q₂ : ℚ), a = q₁ ∧ b = q₂ := by sorry

end NUMINAMATH_CALUDE_harmonious_example_harmonious_rational_sum_harmonious_rational_ratio_l1478_147884


namespace NUMINAMATH_CALUDE_expression_simplification_l1478_147869

theorem expression_simplification (x y : ℝ) (h : (x + 2)^3 - (y - 2)^3 ≠ 0) :
  ((x + 2)^3 + (y + x)^3) / ((x + 2)^3 - (y - 2)^3) = (2*x + y + 2) / (x - y + 4) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1478_147869


namespace NUMINAMATH_CALUDE_seashell_ratio_l1478_147898

theorem seashell_ratio (monday_shells : ℕ) (price_per_shell : ℚ) (total_revenue : ℚ) :
  monday_shells = 30 →
  price_per_shell = 6/5 →
  total_revenue = 54 →
  ∃ (tuesday_shells : ℕ), 
    (tuesday_shells : ℚ) / monday_shells = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_seashell_ratio_l1478_147898


namespace NUMINAMATH_CALUDE_min_cost_2009_l1478_147851

/-- Represents the available coin denominations in rubles -/
inductive Coin : Type
  | one : Coin
  | two : Coin
  | five : Coin
  | ten : Coin

/-- The value of a coin in rubles -/
def coinValue : Coin → Nat
  | Coin.one => 1
  | Coin.two => 2
  | Coin.five => 5
  | Coin.ten => 10

/-- An arithmetic expression using coins and operations -/
inductive Expr : Type
  | coin : Coin → Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluates an expression to its numerical value -/
def eval : Expr → Int
  | Expr.coin c => coinValue c
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- Calculates the cost of an expression in rubles -/
def cost : Expr → Nat
  | Expr.coin c => coinValue c
  | Expr.add e1 e2 => cost e1 + cost e2
  | Expr.sub e1 e2 => cost e1 + cost e2
  | Expr.mul e1 e2 => cost e1 + cost e2
  | Expr.div e1 e2 => cost e1 + cost e2

/-- Theorem: The minimum cost to represent 2009 is 23 rubles -/
theorem min_cost_2009 : 
  (∃ e : Expr, eval e = 2009 ∧ cost e = 23) ∧
  (∀ e : Expr, eval e = 2009 → cost e ≥ 23) := by sorry

end NUMINAMATH_CALUDE_min_cost_2009_l1478_147851


namespace NUMINAMATH_CALUDE_APMS_is_parallelogram_l1478_147882

-- Define the points
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the quadrilateral APMS
def APMS (P Q M S A : Point2D) : Prop :=
  M.x = (P.x + Q.x) / 2 ∧
  M.y = (P.y + Q.y) / 2 ∧
  S.x = M.x ∧
  S.y ≠ M.y

-- Define what it means for a quadrilateral to be a parallelogram
def IsParallelogram (A P M S : Point2D) : Prop :=
  (P.x - A.x = M.x - S.x ∧ P.y - A.y = M.y - S.y) ∧
  (M.x - A.x = S.x - P.x ∧ M.y - A.y = S.y - P.y)

-- Theorem statement
theorem APMS_is_parallelogram 
  (P Q M S A : Point2D) 
  (h_distinct : P ≠ Q) 
  (h_APMS : APMS P Q M S A) : 
  IsParallelogram A P M S :=
sorry

end NUMINAMATH_CALUDE_APMS_is_parallelogram_l1478_147882


namespace NUMINAMATH_CALUDE_root_equation_solution_l1478_147886

theorem root_equation_solution (a b c : ℕ) (ha : a > 1) (hb : b > 1) (hc : c > 1)
  (h : ∀ (N : ℝ), N ≠ 1 → (N^2 * (N^3 * N^(4/c))^(1/b))^(1/a) = N^(17/24)) :
  b = 4 := by sorry

end NUMINAMATH_CALUDE_root_equation_solution_l1478_147886


namespace NUMINAMATH_CALUDE_smiley_face_tulips_l1478_147856

theorem smiley_face_tulips : 
  let red_tulips_per_eye : ℕ := 8
  let red_tulips_for_smile : ℕ := 18
  let yellow_tulips_multiplier : ℕ := 9
  let number_of_eyes : ℕ := 2

  let total_red_tulips : ℕ := red_tulips_per_eye * number_of_eyes + red_tulips_for_smile
  let total_yellow_tulips : ℕ := yellow_tulips_multiplier * red_tulips_for_smile
  let total_tulips : ℕ := total_red_tulips + total_yellow_tulips

  total_tulips = 196 := by sorry

end NUMINAMATH_CALUDE_smiley_face_tulips_l1478_147856
