import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l1095_109537

noncomputable section

-- Define the points A, B, and C
def A : ℝ × ℝ := (5, -2)
def B : ℝ × ℝ := (7, 3)
def C : ℝ × ℝ := (-5, -3)

-- Define the midpoints M and N
noncomputable def M : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
noncomputable def N : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the conditions for M and N
def M_on_y_axis : Prop := M.1 = 0
def N_on_x_axis : Prop := N.2 = 0

-- Define the line MN
def line_MN (x y : ℝ) : Prop := 5 * x - 2 * y - 5 = 0

-- Define the area of the triangle formed by line AB and coordinate axes
noncomputable def triangle_area : ℝ := 841 / 20

theorem triangle_ABC_properties :
  M_on_y_axis ∧ N_on_x_axis →
  (C = (-5, -3)) ∧
  (∀ x y, line_MN x y ↔ 5 * x - 2 * y - 5 = 0) ∧
  (triangle_area = 841 / 20) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l1095_109537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_minimum_at_critical_point_l1095_109587

-- Define the function F(t)
noncomputable def F (t : ℝ) : ℝ :=
  ∫ (x : ℝ) in (t - 3)..(2 * t), (2 : ℝ) ^ (x^2)

-- Define the critical point
noncomputable def critical_point : ℝ := -1 + Real.sqrt 33 / 3

-- Theorem statement
theorem F_minimum_at_critical_point :
  ∀ t ∈ Set.Icc (0 : ℝ) 2,
    F t ≥ F critical_point :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_minimum_at_critical_point_l1095_109587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_can_be_any_nonneg_real_l1095_109574

/-- Represents the area of a trapezoid with bases and altitude in geometric progression -/
noncomputable def trapezoidArea (a : ℝ) (r : ℝ) : ℝ :=
  (1/2) * a * r^3 * (1 + r)

/-- Theorem stating that the trapezoid area can be any non-negative real number -/
theorem trapezoid_area_can_be_any_nonneg_real :
  ∀ (K : ℝ), K ≥ 0 → ∃ (a : ℝ) (r : ℝ), a > 0 ∧ r > 0 ∧ trapezoidArea a r = K := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_can_be_any_nonneg_real_l1095_109574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_without_A_at_head_l1095_109538

theorem arrangements_without_A_at_head (n : ℕ) (h : n = 5) : 
  Nat.factorial n - Nat.factorial (n - 1) = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_without_A_at_head_l1095_109538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_rationalize_l1095_109540

theorem simplify_and_rationalize (a b c d e f : ℝ) :
  a > 0 → b > 0 → c > 0 → d > 0 → e > 0 → f > 0 →
  a = 3 → b = 5 → c = 6 → d = 7 → e = 11 → f = 13 →
  (Real.sqrt a / Real.sqrt d) * (Real.sqrt b / Real.sqrt e) * (Real.sqrt c / Real.sqrt f) = (3 * Real.sqrt 10010) / 1001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_rationalize_l1095_109540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1095_109568

noncomputable def ArithmeticSequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

noncomputable def SumArithmeticSequence (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum (a₁ : ℝ) (S : ℕ → ℝ) :
  (∃ d : ℝ, ∀ n : ℕ, S n = SumArithmeticSequence a₁ d n) →
  a₁ = -2011 →
  S 2010 / 2010 - S 2008 / 2008 = 2 →
  S 2011 = -2011 := by
  sorry

#check arithmetic_sequence_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1095_109568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_point_with_slope_angle_l1095_109508

-- Define the point and slope angle
def point : ℝ × ℝ := (-3, 2)
noncomputable def slope_angle : ℝ := 60 * Real.pi / 180

-- Define the equation of the line
def line_equation (x y : ℝ) : Prop := y - 2 = Real.sqrt 3 * (x + 3)

-- Theorem statement
theorem line_passes_through_point_with_slope_angle :
  line_equation point.1 point.2 ∧
  Real.tan slope_angle = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_point_with_slope_angle_l1095_109508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_darker_green_yellow_percentage_l1095_109521

/-- Proves that the percentage of yellow paint in the darker green paint is 40% -/
theorem darker_green_yellow_percentage
  (light_green_volume : ℝ)
  (light_green_yellow_percentage : ℝ)
  (darker_green_volume : ℝ)
  (final_yellow_percentage : ℝ)
  (darker_green_yellow_percentage : ℝ)
  (h1 : light_green_volume = 5)
  (h2 : light_green_yellow_percentage = 0.2)
  (h3 : darker_green_volume = 1.66666666667)
  (h4 : final_yellow_percentage = 0.25)
  (h5 : light_green_volume * light_green_yellow_percentage +
        darker_green_volume * darker_green_yellow_percentage =
        final_yellow_percentage * (light_green_volume + darker_green_volume)) :
  darker_green_yellow_percentage = 0.4 := by
  sorry

#check darker_green_yellow_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_darker_green_yellow_percentage_l1095_109521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_6_8_l1095_109536

/-- The area of a rhombus with given diagonal lengths -/
noncomputable def rhombusArea (d1 d2 : ℝ) : ℝ := (1/2) * d1 * d2

/-- Theorem: The area of a rhombus with diagonals of lengths 6 and 8 is 24 -/
theorem rhombus_area_6_8 : rhombusArea 6 8 = 24 := by
  -- Unfold the definition of rhombusArea
  unfold rhombusArea
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_6_8_l1095_109536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfies_equation_l1095_109562

theorem function_satisfies_equation (k : ℤ) :
  ∀ n : ℤ, n ≤ -2 →
    (λ m : ℤ ↦ (m - k)^2) n * (λ m : ℤ ↦ (m - k)^2) (n + 1) = 
    ((λ m : ℤ ↦ (m - k)^2) n + n - k)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfies_equation_l1095_109562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_b_completion_time_l1095_109596

/-- The number of days it takes worker B to complete a work, given that:
    - Worker A can complete the work in 12 days
    - Worker B is 20% less efficient than worker A
-/
theorem worker_b_completion_time : ℝ := by
  -- Define A's work rate
  let a_rate : ℝ := 1 / 12

  -- Define B's efficiency relative to A
  let b_efficiency : ℝ := 1 - 0.2

  -- Define B's work rate
  let b_rate : ℝ := b_efficiency * a_rate

  -- Calculate the number of days B needs to complete the work
  have h : (1 : ℝ) / b_rate = 15 := by
    -- Proof steps would go here
    sorry

  -- Conclude that B takes 15 days to complete the work
  exact 15

-- This line is not necessary in a theorem, so I've commented it out
-- #eval worker_b_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_b_completion_time_l1095_109596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_c_and_f_is_seven_l1095_109558

/-- Given a list of positive integers [a, 8, b, c, d, e, f, g, 2] where the sum
of any four consecutive terms is 17, prove that c + f = 7. -/
theorem sum_of_c_and_f_is_seven 
  (a b c d e f g : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (he : e > 0) (hf : f > 0) (hg : g > 0)
  (sum_property : ∀ x y z w, x + y + z + w = 17 → 
    (x = a ∧ y = 8 ∧ z = b ∧ w = c) ∨
    (x = 8 ∧ y = b ∧ z = c ∧ w = d) ∨
    (x = b ∧ y = c ∧ z = d ∧ w = e) ∨
    (x = c ∧ y = d ∧ z = e ∧ w = f) ∨
    (x = d ∧ y = e ∧ z = f ∧ w = g) ∨
    (x = e ∧ y = f ∧ z = g ∧ w = 2)) :
  c + f = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_c_and_f_is_seven_l1095_109558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1095_109516

noncomputable section

open Real

theorem trigonometric_identities :
  (∀ α : ℝ,
    Real.sin (420 * π / 180) * Real.cos (330 * π / 180) + Real.sin (-690 * π / 180) * Real.cos (-660 * π / 180) = 1) ∧
  (∀ α : ℝ,
    (Real.sin (π / 2 + α) * Real.cos (π / 2 - α)) / Real.cos (π + α) + 
    (Real.sin (π - α) * Real.cos (π / 2 + α)) / Real.sin (π + α) = 0) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1095_109516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_drink_volume_l1095_109518

/-- Represents a fruit drink composition -/
structure FruitDrink where
  orange_percent : ℚ
  watermelon_percent : ℚ
  grape_ounces : ℚ

/-- Calculates the total volume of a fruit drink -/
noncomputable def total_volume (drink : FruitDrink) : ℚ :=
  drink.grape_ounces / (1 - drink.orange_percent - drink.watermelon_percent)

/-- Theorem: The total volume of the specific fruit drink is 150 ounces -/
theorem fruit_drink_volume :
  let drink : FruitDrink := {
    orange_percent := 35/100,
    watermelon_percent := 35/100,
    grape_ounces := 45
  }
  total_volume drink = 150 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_drink_volume_l1095_109518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_code_l1095_109575

/-- Checks if a six-digit number satisfies all the given conditions --/
def is_valid_code (n : ℕ) : Prop :=
  -- The number is six digits long
  100000 ≤ n ∧ n < 1000000 ∧
  -- No digit is repeated
  (∀ i j, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10) ∧
  -- The code includes 0, but 0 is not in the penultimate position
  (∃ i, i ≠ 1 ∧ (n / 10^i) % 10 = 0) ∧
  -- No two consecutive digits are both odd or both even
  (∀ i, i < 5 → (n / 10^i) % 2 ≠ (n / 10^(i+1)) % 2) ∧
  -- Neighboring single-digit numbers differ by at least 3
  (∀ i, i < 5 → ((n / 10^i) % 10).sub ((n / 10^(i+1)) % 10) ≥ 3 ∨ 
                ((n / 10^(i+1)) % 10).sub ((n / 10^i) % 10) ≥ 3) ∧
  -- The numbers formed by the first two and second two digits are both multiples of the number formed by the last two digits
  (n / 10000) % ((n / 100) % 100) = 0 ∧
  (n / 100) % 100 % ((n / 100) % 100) = 0

/-- The theorem stating that 903618 is the only valid code --/
theorem unique_valid_code : 
  (is_valid_code 903618) ∧ (∀ n : ℕ, is_valid_code n → n = 903618) := by
  sorry

#check unique_valid_code

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_code_l1095_109575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_outside_circle_l1095_109565

/-- A point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- The circle ρ = 2cos θ -/
def circleEquation (p : PolarPoint) : Prop :=
  p.r = 2 * Real.cos p.θ

/-- A point is outside the circle if its distance from the origin is greater than 2cos θ -/
def outsideCircle (p : PolarPoint) : Prop :=
  p.r > 2 * Real.cos p.θ

theorem point_outside_circle (m : ℝ) (h1 : m > 0) 
    (h2 : outsideCircle ⟨m, π/3⟩) : m > 1 := by
  sorry

#check point_outside_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_outside_circle_l1095_109565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1095_109594

theorem trigonometric_identity (α : ℝ) (h1 : Real.sin α ≠ 0) (h2 : Real.cos α ≠ 0) :
  (Real.sin α + (1 / Real.sin α) + (Real.sin α / Real.cos α))^2 + 
  (Real.cos α + (1 / Real.cos α) + (Real.cos α / Real.sin α))^2 = 
  11 + 2 * (Real.sin α / Real.cos α)^2 + 2 * (Real.cos α / Real.sin α)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1095_109594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_acute_triangle_l1095_109583

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = Real.pi

-- Define what it means for a triangle to be acute
def is_acute (t : Triangle) : Prop :=
  t.A < Real.pi/2 ∧ t.B < Real.pi/2 ∧ t.C < Real.pi/2

-- State the theorem
theorem not_acute_triangle (t : Triangle) 
  (h : Real.cos t.B * Real.cos t.C - Real.sin t.B * Real.sin t.C ≥ 0) : 
  ¬(is_acute t) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_acute_triangle_l1095_109583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_area_and_volume_l1095_109515

/-- Frustum of a right circular cone -/
structure Frustum where
  R : ℝ  -- Lower base radius
  r : ℝ  -- Upper base radius
  h : ℝ  -- Height

/-- Lateral surface area of a frustum -/
noncomputable def lateralSurfaceArea (f : Frustum) : ℝ :=
  Real.pi * (f.R + f.r) * Real.sqrt (f.h^2 + (f.R - f.r)^2)

/-- Volume of a frustum -/
noncomputable def volume (f : Frustum) : ℝ :=
  (1/3) * Real.pi * f.h * (f.R^2 + f.R * f.r + f.r^2)

/-- Theorem stating the lateral surface area and volume of a specific frustum -/
theorem frustum_area_and_volume :
  let f : Frustum := ⟨8, 4, 5⟩
  lateralSurfaceArea f = 12 * Real.pi * Real.sqrt 41 ∧ volume f = 560 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_area_and_volume_l1095_109515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_b1_b2_l1095_109579

def sequence_b (b₁ b₂ : ℕ+) : ℕ → ℕ+
  | 0 => b₁
  | 1 => b₂
  | (n + 2) => ⟨(sequence_b b₁ b₂ n + 2021) / (1 + sequence_b b₁ b₂ (n + 1)), sorry⟩

theorem min_sum_b1_b2 :
  ∃ b₁ b₂ : ℕ+, (∀ n : ℕ, (sequence_b b₁ b₂ n).val ∈ Set.univ) ∧
    ∀ c₁ c₂ : ℕ+, (∀ n : ℕ, (sequence_b c₁ c₂ n).val ∈ Set.univ) →
      b₁.val + b₂.val ≤ c₁.val + c₂.val ∧
      b₁.val + b₂.val = 90 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_b1_b2_l1095_109579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_f_increasing_g_range_l1095_109559

-- Define the power function f
def f : ℝ → ℝ := sorry

-- Define the even function g
def g : ℝ → ℝ := sorry

-- Axioms
axiom f_is_power_function : ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α
axiom f_passes_through_point : f 2 = Real.sqrt 2
axiom g_is_even : ∀ x : ℝ, g (-x) = g x
axiom g_equals_f_nonneg : ∀ x : ℝ, x ≥ 0 → g x = f x

-- Theorem statements
theorem f_expression : ∀ x : ℝ, x ≥ 0 → f x = Real.sqrt x := by sorry

theorem f_increasing : StrictMono f := by sorry

theorem g_range : Set.Icc (-4 : ℝ) 6 = {m : ℝ | g (1 - m) ≤ Real.sqrt 5} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_f_increasing_g_range_l1095_109559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_is_two_l1095_109586

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The distance from a focus to an asymptote of the hyperbola -/
noncomputable def focus_to_asymptote_distance (h : Hyperbola) : ℝ :=
  h.a * h.b * eccentricity h / Real.sqrt (h.a^2 + h.b^2)

/-- The length of the line segment intercepted by the asymptotes on the given line -/
noncomputable def intercepted_segment_length (h : Hyperbola) : ℝ :=
  2 * h.b^3 / (h.a * Real.sqrt (h.a^2 + h.b^2))

/-- The main theorem -/
theorem eccentricity_is_two (h : Hyperbola) 
    (h_equal : focus_to_asymptote_distance h = intercepted_segment_length h) :
    eccentricity h = 2 := by
  sorry

#check eccentricity_is_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_is_two_l1095_109586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_city_mpg_l1095_109528

-- Define the properties of the car
noncomputable def highway_miles_per_tankful : ℝ := 462
noncomputable def city_miles_per_tankful : ℝ := 336
noncomputable def highway_city_mpg_difference : ℝ := 18

-- Define the function to calculate city miles per gallon
noncomputable def city_mpg (tank_capacity : ℝ) : ℝ :=
  city_miles_per_tankful / tank_capacity

-- Define the function to calculate highway miles per gallon
noncomputable def highway_mpg (tank_capacity : ℝ) : ℝ :=
  highway_miles_per_tankful / tank_capacity

-- Theorem statement
theorem car_city_mpg :
  ∃ (tank_capacity : ℝ),
    tank_capacity > 0 ∧
    highway_mpg tank_capacity = city_mpg tank_capacity + highway_city_mpg_difference ∧
    city_mpg tank_capacity = 48 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_city_mpg_l1095_109528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_different_colors_is_seven_ninths_l1095_109554

/-- Represents the possible colors for shorts -/
inductive ShortsColor
| Black
| Gold
| Blue
deriving Fintype

/-- Represents the possible colors for jerseys -/
inductive JerseyColor
| Black
| White
| Gold
deriving Fintype

/-- Calculates the probability of selecting different colors for shorts and jersey -/
def prob_different_colors : ℚ :=
  let total_combinations := (Fintype.card ShortsColor) * (Fintype.card JerseyColor)
  let matching_combinations := 2  -- Black-Black and Gold-Gold
  let mismatching_combinations := total_combinations - matching_combinations
  mismatching_combinations / total_combinations

/-- Theorem stating that the probability of selecting different colors for shorts and jersey is 7/9 -/
theorem prob_different_colors_is_seven_ninths :
  prob_different_colors = 7 / 9 := by
  sorry

#eval prob_different_colors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_different_colors_is_seven_ninths_l1095_109554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sum_squared_ge_200_l1095_109512

-- Define the circle
def Circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 100}

-- Define the points P and Q on the circle
def P : ℝ × ℝ := (-10, 0)
def Q : ℝ × ℝ := (10, 0)

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem triangle_sum_squared_ge_200 :
  ∀ R : ℝ × ℝ, R ∈ Circle → R ≠ P → R ≠ Q →
  (distance P R + distance Q R)^2 ≥ 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sum_squared_ge_200_l1095_109512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_theorem_l1095_109551

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ -1 then (2 : ℝ)^(-2*x) else 2*x + 2

theorem f_range_theorem : 
  {a : ℝ | f a ≥ 2} = Set.Iic (-1) ∪ Set.Ici 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_theorem_l1095_109551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oliver_bag_fraction_is_one_sixth_l1095_109530

/-- The fraction of the weight of each of Oliver's bags compared to James's bag -/
noncomputable def oliverBagFraction (jamesWeight : ℝ) (oliverTotalWeight : ℝ) : ℝ :=
  oliverTotalWeight / (2 * jamesWeight)

/-- Proof that Oliver's bag fraction is 1/6 given the conditions -/
theorem oliver_bag_fraction_is_one_sixth :
  let jamesWeight : ℝ := 18
  let oliverTotalWeight : ℝ := 6
  oliverBagFraction jamesWeight oliverTotalWeight = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oliver_bag_fraction_is_one_sixth_l1095_109530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perfect_square_over_50_with_odd_factors_l1095_109549

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_odd_factors (n : ℕ) : Prop := Odd (Finset.card (Nat.divisors n))

theorem smallest_perfect_square_over_50_with_odd_factors :
  (∀ m : ℕ, m > 50 ∧ is_perfect_square m ∧ has_odd_factors m → m ≥ 64) ∧
  (64 > 50 ∧ is_perfect_square 64 ∧ has_odd_factors 64) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perfect_square_over_50_with_odd_factors_l1095_109549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ananthu_work_days_l1095_109582

/-- Represents the number of days it takes for a person to complete the work alone -/
structure WorkDays where
  days : ℚ
  days_positive : days > 0

/-- Represents the portion of work completed -/
def work_completed (w : WorkDays) (days : ℚ) : ℚ := days / w.days

theorem ananthu_work_days 
  (amit : WorkDays)
  (total_days : ℚ)
  (amit_worked : ℚ)
  (h_amit : amit.days = 15)
  (h_total : total_days = 39)
  (h_amit_worked : amit_worked = 3)
  (h_total_positive : total_days > 0)
  (h_amit_worked_positive : amit_worked > 0)
  (h_amit_worked_le_total : amit_worked ≤ total_days) :
  ∃ (ananthu : WorkDays), ananthu.days = 45 ∧ 
    work_completed amit amit_worked + 
    work_completed ananthu (total_days - amit_worked) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ananthu_work_days_l1095_109582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_inscribed_square_l1095_109519

/-- Given a circle and a square centered at the origin, where the circle's equation is x^2 + y^2 = r^2
    and the square has side length s, prove that r = (s√2)/2 when the square's vertices touch the circle. -/
theorem circle_inscribed_square (r s : ℝ) : r = s * Real.sqrt 2 / 2 ↔ 
  ∃ (circle : Set (ℝ × ℝ)) (square : Set (ℝ × ℝ)),
    (∀ p : ℝ × ℝ, p ∈ circle ↔ p.1^2 + p.2^2 = r^2) ∧
    (∀ p : ℝ × ℝ, p ∈ square ↔ max (|p.1|) (|p.2|) = s/2) ∧
    (∀ p : ℝ × ℝ, p ∈ square → p.1^2 + p.2^2 ≤ r^2) ∧
    (∃ p : ℝ × ℝ, p ∈ square ∧ p.1^2 + p.2^2 = r^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_inscribed_square_l1095_109519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equals_sqrt13_l1095_109577

/-- The projection of vector b in the direction of vector a -/
noncomputable def vector_projection (a b : ℝ × ℝ) : ℝ :=
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (a.1^2 + a.2^2)

/-- Theorem: The projection of vector b = (-4, 7) in the direction of vector a = (2, 3) is √13 -/
theorem projection_equals_sqrt13 :
  vector_projection (2, 3) (-4, 7) = Real.sqrt 13 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equals_sqrt13_l1095_109577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1095_109502

/-- The set M defined as {y | y = x^2} -/
def M : Set ℝ := {y | ∃ x, y = x^2}

/-- The set N defined as {x | x^2/2 + y^2 <= 1} -/
def N : Set ℝ := {x | ∃ y, x^2/2 + y^2 ≤ 1}

/-- The intersection of sets M and N is equal to the interval [0, √2] -/
theorem intersection_M_N : M ∩ N = Set.Icc 0 (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1095_109502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_hit_six_l1095_109572

/-- Represents the score of a single dart throw -/
def DartScore := Fin 10

/-- Represents a pair of dart throws -/
structure ThrowPair where
  first : DartScore
  second : DartScore
  different : first ≠ second

/-- Represents a player in the dart game -/
inductive Player
| Alice | Ben | Cindy | Dave | Ellen

/-- The total score for each player -/
def playerScore : Player → Nat
| Player.Alice => 16
| Player.Ben => 4
| Player.Cindy => 7
| Player.Dave => 11
| Player.Ellen => 17

/-- The throws made by each player -/
def playerThrows : Player → ThrowPair := sorry

/-- All throws are unique across all players -/
axiom throws_unique : ∀ p1 p2 : Player, p1 ≠ p2 →
  (playerThrows p1).first ≠ (playerThrows p2).first ∧
  (playerThrows p1).first ≠ (playerThrows p2).second ∧
  (playerThrows p1).second ≠ (playerThrows p2).first ∧
  (playerThrows p1).second ≠ (playerThrows p2).second

/-- The sum of a player's throws equals their score -/
axiom score_sum : ∀ p : Player,
  (playerThrows p).first.val + (playerThrows p).second.val + 2 = playerScore p

/-- Alice hit the region worth 6 points -/
theorem alice_hit_six : ∃ x : DartScore, x.val = 6 ∧ ((playerThrows Player.Alice).first = x ∨ (playerThrows Player.Alice).second = x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_hit_six_l1095_109572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sum_of_squares_and_divisibility_l1095_109580

theorem prime_sum_of_squares_and_divisibility (p : ℕ) : 
  Prime p → 
  (∃ m n : ℤ, (p : ℤ) = m^2 + n^2 ∧ (p : ℤ) ∣ m^3 + n^3 - 4) → 
  p = 2 ∨ p = 5 ∨ p = 13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sum_of_squares_and_divisibility_l1095_109580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_implies_a_nonnegative_l1095_109511

noncomputable section

/-- The function f(x) = x^2 + 2/x + a*ln(x) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2/x + a * Real.log x

/-- f is monotonically increasing on [1,+∞) -/
def is_monotone_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 1 ≤ x → x ≤ y → f x ≤ f y

theorem monotone_increasing_implies_a_nonnegative (a : ℝ) :
  is_monotone_increasing (f a) → a ≥ 0 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_implies_a_nonnegative_l1095_109511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_other_diagonal_l1095_109505

/-- The area of a rhombus given its diagonals -/
noncomputable def rhombusArea (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

theorem rhombus_other_diagonal 
  (d1 : ℝ) (area : ℝ) (h1 : d1 = 14) (h2 : area = 126) :
  ∃ d2 : ℝ, d2 = 18 ∧ rhombusArea d1 d2 = area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_other_diagonal_l1095_109505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_jump_distances_l1095_109522

structure Animal where
  name : String
  initialJump : ℝ
  jumpPattern : ℕ → ℝ → ℝ

def grasshopper : Animal :=
  { name := "Grasshopper"
  , initialJump := 25
  , jumpPattern := λ _ d => d + 10 }

def frog : Animal :=
  { name := "Frog"
  , initialJump := 50
  , jumpPattern := λ _ d => d - 5 }

def mouse : Animal :=
  { name := "Mouse"
  , initialJump := 25
  , jumpPattern := λ n d => if n % 2 = 0 then d + 3 else d - 2 }

def kangaroo : Animal :=
  { name := "Kangaroo"
  , initialJump := 75
  , jumpPattern := λ _ d => d * 2 }

def rabbit : Animal :=
  { name := "Rabbit"
  , initialJump := 187.5
  , jumpPattern := λ n d => d * (2.5 - 0.5 * (n : ℝ)) }

def jumpSequence (animal : Animal) : ℕ → ℝ
  | 0 => animal.initialJump
  | n + 1 => animal.jumpPattern n (jumpSequence animal n)

theorem last_jump_distances (n : ℕ) (h : n = 3) :
  (jumpSequence grasshopper n = 55) ∧
  (jumpSequence frog n = 35) ∧
  (jumpSequence mouse n = 29) ∧
  (jumpSequence kangaroo n = 600) ∧
  (jumpSequence rabbit n = 600) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_jump_distances_l1095_109522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_roots_range_f_plus_g_positive_l1095_109573

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.log x - 1
noncomputable def g (x : ℝ) : ℝ := x / Real.exp x

-- Theorem for the range of a
theorem g_roots_range (a : ℝ) :
  (∃ x y, 0 < x ∧ x < 2 ∧ 0 < y ∧ y < 2 ∧ x ≠ y ∧ g x = a ∧ g y = a) ↔
  (2 / Real.exp 2 < a ∧ a < 1 / Real.exp 1) :=
sorry

-- Theorem for the positivity of f(x) + 2/(e * g(x))
theorem f_plus_g_positive (x : ℝ) (hx : x > 0) :
  f x + 2 / (Real.exp 1 * g x) > 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_roots_range_f_plus_g_positive_l1095_109573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_a_on_b_l1095_109545

noncomputable def a : Fin 2 → ℝ := ![1, 2]
noncomputable def b : Fin 2 → ℝ := ![-1, 2]

noncomputable def projection_vector (u v : Fin 2 → ℝ) : Fin 2 → ℝ :=
  (((u • v) / (v • v)) • v)

theorem projection_of_a_on_b :
  projection_vector a b = ![-(3/5), 6/5] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_a_on_b_l1095_109545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_concentration_l1095_109584

/-- Represents a mixture of acid and water -/
structure Mixture where
  acid : ℝ
  water : ℝ

/-- Calculates the acid percentage in a mixture -/
noncomputable def acid_percentage (m : Mixture) : ℝ :=
  m.acid / (m.acid + m.water) * 100

/-- Theorem stating the properties of the mixture -/
theorem mixture_concentration (m : Mixture) : 
  acid_percentage { acid := m.acid, water := m.water + 2 } = 18 →
  acid_percentage { acid := m.acid + 2, water := m.water + 2 } = 36 →
  ∃ ε > 0, |acid_percentage m - 19| < ε := by
  sorry

#check mixture_concentration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_concentration_l1095_109584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l1095_109500

open Real

/-- The original function f(x) = √3 * cos(x) + sin(x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * cos x + sin x

/-- The translated function g(x) = f(x + m) -/
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f (x + m)

/-- A function is symmetric about the origin if f(x) = -f(-x) for all x -/
def symmetric_about_origin (h : ℝ → ℝ) : Prop :=
  ∀ x, h x = -h (-x)

/-- Theorem stating the existence of a minimum positive translation for symmetry -/
theorem min_translation_for_symmetry :
  ∃ m : ℝ, m > 0 ∧ symmetric_about_origin (g m) ∧
  ∀ m' : ℝ, m' > 0 ∧ symmetric_about_origin (g m') → m ≤ m' := by
  sorry

#check min_translation_for_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l1095_109500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1095_109543

open Set

noncomputable def f (x : ℝ) : ℝ := (10 * x^2 + 20 * x - 68) / ((2 * x - 3) * (x + 4) * (x - 2))

def solution_set : Set ℝ := {x | f x < 3 ∧ (2 * x - 3) * (x + 4) * (x - 2) ≠ 0}

theorem inequality_solution :
  solution_set = Ioo (-4 : ℝ) (-2) ∪ Ioo (-1/3) (3/2) := by
  sorry

#check inequality_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1095_109543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_diff_is_own_rationalized_factor_l1095_109567

theorem sqrt_diff_is_own_rationalized_factor (x y : ℝ) : 
  (x ≥ y) → Real.sqrt (x - y) = Real.sqrt (x - y) :=
by
  intro h
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_diff_is_own_rationalized_factor_l1095_109567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1095_109563

theorem problem_statement : (∀ x : ℝ, (2 : ℝ)^x > 0) ∨ (∃ x : ℝ, Real.sin x = 2) := by
  left
  intro x
  exact Real.rpow_pos_of_pos (by norm_num) x


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1095_109563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_max_k_l1095_109588

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x

-- State the theorem
theorem tangent_line_and_max_k :
  -- Part 1: Tangent line equation
  (∀ x y : ℝ, (x, y) ∈ {(x, y) | 2*x - y - 1 = 0} ↔ 
    ∃ h : ℝ, h ≠ 0 ∧ (f (1 + h) - f 1) / h = 2) ∧
  -- Part 2: Maximum value of k
  (∀ k : ℤ, (∀ x : ℝ, x > 1 → ↑k * (x - 1) < f x) → k ≤ 3) ∧
  (∃ k : ℤ, k = 3 ∧ ∀ x : ℝ, x > 1 → ↑k * (x - 1) < f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_max_k_l1095_109588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_circle_l1095_109507

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a plane -/
def Point := ℝ × ℝ

/-- The distance between two points in a plane -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- A circle passes through a point if the distance from its center to the point equals its radius -/
def passes_through (c : Circle) (p : Point) : Prop :=
  distance c.center p = c.radius

/-- The locus of centers of circles with radius a passing through a fixed point is a circle -/
theorem locus_is_circle (a : ℝ) (fixed_point : Point) :
  ∃ (locus : Circle), ∀ (c : Circle),
    c.radius = a ∧ passes_through c fixed_point ↔ passes_through locus c.center := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_circle_l1095_109507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1095_109585

noncomputable def a : ℝ := Real.cos (12 * Real.pi / 180) ^ 2 - Real.sin (12 * Real.pi / 180) ^ 2

noncomputable def b : ℝ := (2 * Real.tan (12 * Real.pi / 180)) / (1 - Real.tan (12 * Real.pi / 180) ^ 2)

noncomputable def c : ℝ := ((1 - Real.cos (48 * Real.pi / 180)) / 2) ^ (1/2 : ℝ)

theorem relationship_abc : c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1095_109585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_negative_at_midpoint_l1095_109509

open Real

-- Define the function f(x) = e^x - ax
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

-- Define the derivative of f
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a

theorem derivative_negative_at_midpoint (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > 0) 
  (hx : x₁ < x₂) 
  (hf₁ : f a x₁ = 0) 
  (hf₂ : f a x₂ = 0) : 
  f_deriv a ((x₁ + x₂) / 2) < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_negative_at_midpoint_l1095_109509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_square_arrangement_l1095_109564

/-- Represents the arrangement of numbers in a square -/
structure SquareArrangement where
  top_left : ℕ
  top_right : ℕ
  bottom_left : ℕ
  bottom_right : ℕ
  center : ℕ

/-- Checks if two numbers are connected in the square arrangement -/
def isConnected (a b : ℕ) (arr : SquareArrangement) : Prop :=
  (a = arr.center ∧ (b = arr.top_left ∨ b = arr.top_right ∨ b = arr.bottom_left ∨ b = arr.bottom_right)) ∨
  (b = arr.center ∧ (a = arr.top_left ∨ a = arr.top_right ∨ a = arr.bottom_left ∨ a = arr.bottom_right))

/-- Helper function to get the set of all numbers in the arrangement -/
def SquareArrangement.toSet (arr : SquareArrangement) : Set ℕ :=
  {arr.top_left, arr.top_right, arr.bottom_left, arr.bottom_right, arr.center}

/-- The main theorem stating the existence of a valid square arrangement -/
theorem exists_valid_square_arrangement :
  ∃ (arr : SquareArrangement),
    (∀ a b : ℕ, a ∈ arr.toSet → b ∈ arr.toSet → a ≠ b →
      (isConnected a b arr → ∃ d : ℕ, d > 1 ∧ d ∣ a ∧ d ∣ b) ∧
      (¬isConnected a b arr → Nat.gcd a b = 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_square_arrangement_l1095_109564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2014_value_l1095_109591

def sequence_a : ℕ → ℚ
  | 0 => 3  -- Add this case for 0
  | 1 => 3
  | n + 2 => 1 / (sequence_a (n + 1) - 1) + 1

theorem a_2014_value : sequence_a 2014 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2014_value_l1095_109591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_modulus_l1095_109503

theorem complex_power_modulus : Complex.abs ((2 + Complex.I) ^ 8) = 625 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_modulus_l1095_109503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_k_l1095_109524

-- Define the inequality
def inequality (k : ℝ) (x : ℝ) : Prop := 1 + k / (x - 1) ≤ 0

-- Define the solution set
def solution_set (k : ℝ) : Set ℝ := {x : ℝ | inequality k x}

-- Theorem statement
theorem determine_k : 
  ∀ k : ℝ, solution_set k = Set.Icc (-2 : ℝ) 1 → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_k_l1095_109524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_coefficient_l1095_109597

theorem constant_term_coefficient (a : ℝ) : 
  (∃ k : ℕ, k = (6 : ℕ).choose 3 ∧ k * a^3 = 20) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_coefficient_l1095_109597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_path_exists_l1095_109561

/-- Represents a cube with marked vertices and face centers -/
structure MarkedCube where
  vertices : Finset (Fin 8)
  faceCenters : Finset (Fin 6)

/-- Represents a path on the cube using face diagonals -/
def CubePath (cube : MarkedCube) := List (Fin 14)

/-- Checks if a path is valid (alternates between vertices and face centers) -/
def isValidPath (cube : MarkedCube) (path : CubePath cube) : Prop :=
  sorry

/-- Checks if a path visits all marked points exactly once -/
def visitsAllPointsOnce (cube : MarkedCube) (path : CubePath cube) : Prop :=
  sorry

/-- Theorem stating that no valid path exists that visits all points exactly once -/
theorem no_valid_path_exists (cube : MarkedCube) :
  ¬∃ (path : CubePath cube), isValidPath cube path ∧ visitsAllPointsOnce cube path :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_path_exists_l1095_109561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wide_flag_width_is_five_l1095_109517

/-- Calculates the width of wide rectangular flags given the following conditions:
    * Total fabric: 1000 square feet
    * Square flags: 4 feet by 4 feet, 16 made
    * Wide rectangular flags: unknown width, 3 feet height, 20 made
    * Tall rectangular flags: 3 feet by 5 feet, 10 made
    * Leftover fabric: 294 square feet
-/
noncomputable def wide_flag_width (total_fabric : ℝ) (square_flag_side : ℝ) (square_flag_count : ℕ)
  (wide_flag_height : ℝ) (wide_flag_count : ℕ)
  (tall_flag_width : ℝ) (tall_flag_height : ℝ) (tall_flag_count : ℕ)
  (leftover_fabric : ℝ) : ℝ :=
  let square_flag_area := square_flag_side * square_flag_side
  let tall_flag_area := tall_flag_width * tall_flag_height
  let used_fabric := square_flag_area * (square_flag_count : ℝ) + tall_flag_area * (tall_flag_count : ℝ)
  let wide_flags_total_area := total_fabric - leftover_fabric - used_fabric
  wide_flags_total_area / ((wide_flag_count : ℝ) * wide_flag_height)

/-- Theorem stating that the width of the wide rectangular flags is 5 feet. -/
theorem wide_flag_width_is_five :
  wide_flag_width 1000 4 16 3 20 3 5 10 294 = 5 := by
  -- Unfold the definition of wide_flag_width
  unfold wide_flag_width
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wide_flag_width_is_five_l1095_109517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kubik_family_seating_l1095_109578

/-- The number of family members -/
def n : ℕ := 7

/-- The number of distinct seating arrangements for n people at a round table,
    where one specific person cannot sit next to another specific person -/
def seating_arrangements (n : ℕ) : ℕ :=
  (n - 1).factorial - 2 * (n - 2).factorial

/-- Theorem stating that the number of seating arrangements for 7 people
    under the given conditions is 480 -/
theorem kubik_family_seating :
  seating_arrangements n = 480 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kubik_family_seating_l1095_109578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1095_109504

noncomputable def f (x : ℝ) : ℝ := 
  (Real.sin x ^ 3 + 8 * Real.sin x ^ 2 + Real.sin x + 3 * Real.cos x ^ 2 - 9) / (Real.sin x - 1)

theorem f_range : 
  ∀ y : ℝ, (∃ x : ℝ, Real.sin x ≠ 1 ∧ f x = y) ↔ 2 ≤ y ∧ y < 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1095_109504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_sum_l1095_109523

/-- The equation of the ellipse -/
noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  3 * x^2 + 2 * x * y + 4 * y^2 - 15 * x - 25 * y + 55 = 0

/-- The set of all points (x,y) on the ellipse -/
noncomputable def ellipse_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ellipse_equation p.1 p.2}

/-- The ratio y/x for a point (x,y) -/
noncomputable def ratio (p : ℝ × ℝ) : ℝ := p.2 / p.1

/-- The theorem statement -/
theorem ellipse_ratio_sum :
  ∃ (max_ratio min_ratio : ℝ),
    (∀ p ∈ ellipse_points, ratio p ≤ max_ratio) ∧
    (∀ p ∈ ellipse_points, ratio p ≥ min_ratio) ∧
    (max_ratio + min_ratio = 26 / 51) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_sum_l1095_109523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_fee_correct_l1095_109539

-- Define the water fee function
noncomputable def water_fee (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 5 then 1.2 * x
  else if 5 < x ∧ x ≤ 6 then 3.6 * x - 12
  else if 6 < x ∧ x ≤ 7 then 6 * x - 26.4
  else 0

-- Theorem: The water fee function is correct and f(6.5) = 12.6
theorem water_fee_correct :
  (∀ x : ℝ, 0 < x → x ≤ 7 →
    (x ≤ 5 → water_fee x = 1.2 * x) ∧
    (5 < x → x ≤ 6 → water_fee x = 3.6 * x - 12) ∧
    (6 < x → x ≤ 7 → water_fee x = 6 * x - 26.4)) ∧
  water_fee 6.5 = 12.6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_fee_correct_l1095_109539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sally_change_l1095_109557

def frame_cost : ℝ := 3
def frame_quantity : ℕ := 3
def candle_cost : ℝ := 4
def candle_quantity : ℕ := 2
def bowl_cost : ℝ := 7
def discount_rate : ℝ := 0.1
def paid_amount : ℝ := 50

def total_cost : ℝ := frame_cost * frame_quantity + candle_cost * candle_quantity + bowl_cost

def discounted_cost : ℝ := total_cost * (1 - discount_rate)

theorem sally_change : 
  paid_amount - discounted_cost = 28.4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sally_change_l1095_109557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_graph_transformation_l1095_109520

theorem sine_graph_transformation (x : ℝ) :
  let original_func := λ x => Real.sin (x - Real.pi / 6)
  let translated_func := λ x => original_func (x - Real.pi / 4)
  let final_func := λ x => translated_func (x / 2)
  final_func x = Real.sin (x / 2 - 5 * Real.pi / 12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_graph_transformation_l1095_109520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_110_l1095_109535

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  first_term : ℚ
  common_diff : ℚ

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.first_term + (n - 1 : ℚ) * seq.common_diff)

theorem arithmetic_sequence_sum_110 (seq : ArithmeticSequence) :
  sum_n_terms seq 10 = 100 →
  sum_n_terms seq 100 = 10 →
  sum_n_terms seq 110 = -110 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_110_l1095_109535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_value_height_value_l1095_109547

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesCondition1 (t : Triangle) : Prop :=
  t.a * Real.cos t.B - t.c = t.b / 2

def satisfiesCondition2 (t : Triangle) : Prop :=
  t.b - t.c = Real.sqrt 6 ∧ t.a = 3 + Real.sqrt 3

-- Define the height function
noncomputable def triangleHeight (t : Triangle) : ℝ :=
  2 * (t.a * t.b * Real.sin t.C) / (t.a + t.b + t.c)

-- Theorem 1
theorem angle_A_value (t : Triangle) (h : satisfiesCondition1 t) :
  t.A = 2 * Real.pi / 3 := by
  sorry

-- Theorem 2
theorem height_value (t : Triangle) (h1 : satisfiesCondition1 t) (h2 : satisfiesCondition2 t) :
  triangleHeight t = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_value_height_value_l1095_109547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l1095_109527

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space defined by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Calculate the y-coordinate of a point on a line given its x-coordinate -/
def Line.yAt (l : Line) (x : ℝ) : ℝ :=
  l.slope * x + l.yIntercept

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : Point) : Prop :=
  p.y = l.yAt p.x

/-- Calculate the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

theorem quadrilateral_area : 
  let O : Point := ⟨0, 0⟩
  let C : Point := ⟨6, 0⟩
  let E : Point := ⟨3, 3⟩
  let line1 : Line := ⟨-3, 12⟩  -- y = -3x + 12
  let line2 : Line := ⟨-1, 6⟩  -- y = -x + 6
  let B : Point := ⟨0, line1.yAt 0⟩
  line1.contains E ∧ line2.contains E ∧ line2.contains C →
  triangleArea O B C + triangleArea B E C = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l1095_109527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_l1095_109532

/-- Two blocks of cheese with given dimensional relationships -/
structure CheeseBlocks where
  -- Dimensions of the second block
  a : ℝ  -- length
  b : ℝ  -- width
  c : ℝ  -- height
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- Volume of the first block -/
noncomputable def volume_first (blocks : CheeseBlocks) : ℝ :=
  (3/2 * blocks.a) * (4/5 * blocks.b) * (7/10 * blocks.c)

/-- Volume of the second block -/
noncomputable def volume_second (blocks : CheeseBlocks) : ℝ :=
  blocks.a * blocks.b * blocks.c

/-- Theorem stating the volume relationship between the two blocks -/
theorem volume_ratio (blocks : CheeseBlocks) :
  volume_second blocks = (25/21) * volume_first blocks := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_l1095_109532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_reciprocal_l1095_109533

theorem tan_alpha_reciprocal (α : ℝ) (h : Real.tan α = -1/2) :
  1 / (Real.sin α ^ 2 - Real.sin α * Real.cos α - 2 * Real.cos α ^ 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_reciprocal_l1095_109533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1095_109552

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * Real.sin (x + Real.pi/3) - Real.sqrt 3 * (Real.sin x)^2 + Real.sin x * Real.cos x

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ ∃ (x₀ : ℝ), f x₀ = m) ∧
  (∃ (k : ℤ), f (k * Real.pi - 5 * Real.pi / 12) = -2 ∧ ∀ (x : ℝ), f x ≥ -2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1095_109552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angles_satisfy_conditions_l1095_109550

noncomputable def α₁ (k : ℤ) : ℝ := Real.arctan (24 / 7) + 2 * Real.pi * (k : ℝ)
noncomputable def α₂ (k : ℤ) : ℝ := Real.pi + 2 * Real.pi * (k : ℝ)
noncomputable def α₃ (k : ℤ) : ℝ := Real.arctan ((4 + 3 * Real.sqrt 24) / (4 * Real.sqrt 24 - 3)) + 2 * Real.pi * (k : ℝ)
noncomputable def α₄ (k : ℤ) : ℝ := Real.arctan ((3 * Real.sqrt 24 - 4) / (4 * Real.sqrt 24 + 3)) + 2 * Real.pi * (k : ℝ)

theorem angles_satisfy_conditions (k : ℤ) :
  (∃ (x y : ℝ), x * Real.cos (α₁ k) + y * Real.sin (α₁ k) = 2 ∧ x^2 + y^2 = 4) ∧
  (∃ (x y : ℝ), x * Real.cos (α₂ k) + y * Real.sin (α₂ k) = 2 ∧ x^2 + y^2 = 4) ∧
  (∃ (x y : ℝ), x * Real.cos (α₃ k) + y * Real.sin (α₃ k) = 2 ∧ x^2 + y^2 = 4) ∧
  (∃ (x y : ℝ), x * Real.cos (α₄ k) + y * Real.sin (α₄ k) = 2 ∧ x^2 + y^2 = 4) ∧
  (∃ (x y : ℝ), x * Real.cos (α₁ k) + y * Real.sin (α₁ k) = 2 ∧ (x + 3)^2 + (y - 4)^2 = 1) ∧
  (∃ (x y : ℝ), x * Real.cos (α₂ k) + y * Real.sin (α₂ k) = 2 ∧ (x + 3)^2 + (y - 4)^2 = 1) ∧
  (∃ (x y : ℝ), x * Real.cos (α₃ k) + y * Real.sin (α₃ k) = 2 ∧ (x + 3)^2 + (y - 4)^2 = 1) ∧
  (∃ (x y : ℝ), x * Real.cos (α₄ k) + y * Real.sin (α₄ k) = 2 ∧ (x + 3)^2 + (y - 4)^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angles_satisfy_conditions_l1095_109550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_K2_confidence_statement_false_l1095_109556

/-- Represents the observed value of the K^2 random variable -/
def K2_observed_value : ℝ → ℝ := sorry

/-- Represents the confidence level in judging that X is related to Y -/
def confidence_level : ℝ → ℝ := sorry

/-- Axiom: The relationship between K^2 observed value and confidence level -/
axiom K2_confidence_relation : ∀ k1 k2 : ℝ, 
  k1 < k2 → confidence_level k1 < confidence_level k2

/-- Theorem: The statement about K^2 and confidence is false -/
theorem K2_confidence_statement_false : 
  ¬(∀ k1 k2 : ℝ, k1 < k2 → confidence_level k1 > confidence_level k2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_K2_confidence_statement_false_l1095_109556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_direction_of_cd_l1095_109569

/-- A conic section in the plane -/
structure ConicSection where
  a : ℝ
  b : ℝ
  c : ℝ
  e : ℝ
  f : ℝ
  g : ℝ
  a_nonzero : a ≠ 0

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in the plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- A line in the plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The direction of a line, represented by its slope -/
def direction (l : Line) : ℝ := l.slope

/-- Predicate to check if a point lies on a conic section -/
def on_conic_section (p : Point) (cs : ConicSection) : Prop :=
  cs.a * p.x^2 + cs.b * p.x * p.y + cs.c * p.y^2 + cs.e * p.x + cs.f * p.y + cs.g = 0

/-- Predicate to check if a point lies on a circle -/
def on_circle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- The line passing through two points -/
noncomputable def line_through_points (p1 p2 : Point) : Line :=
  { slope := (p2.y - p1.y) / (p2.x - p1.x),
    intercept := p1.y - (p2.y - p1.y) / (p2.x - p1.x) * p1.x }

/-- The main theorem -/
theorem fixed_direction_of_cd 
  (cs : ConicSection) 
  (A B : Point) 
  (h_A : on_conic_section A cs) 
  (h_B : on_conic_section B cs) :
  ∃ d : ℝ, ∀ (circ : Circle) (C D : Point),
    on_circle A circ → on_circle B circ →
    on_conic_section C cs → on_conic_section D cs →
    on_circle C circ → on_circle D circ →
    C ≠ A → C ≠ B → D ≠ A → D ≠ B →
    direction (line_through_points C D) = d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_direction_of_cd_l1095_109569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_x_value_l1095_109542

/-- If two 2D vectors (2, -1) and (1, x) are parallel, then x = -1/2 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![2, -1]
  let b : Fin 2 → ℝ := ![1, x]
  (∃ (k : ℝ), ∀ i, a i = k * b i) →
  x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_x_value_l1095_109542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_sqrt2_over_2_l1095_109514

theorem arcsin_sqrt2_over_2 : Real.arcsin (Real.sqrt 2 / 2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_sqrt2_over_2_l1095_109514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_dot_product_l1095_109595

-- Define the ellipse
noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

-- Define points C, D, O
def C : ℝ × ℝ := (-2, 0)
def D : ℝ × ℝ := (2, 0)
def O : ℝ × ℝ := (0, 0)

-- Define the moving point M
noncomputable def M (k : ℝ) : ℝ × ℝ := (2, 4*k)

-- Define the intersection point P
noncomputable def P (k : ℝ) : ℝ × ℝ := ((2 - 4*k^2) / (1 + 2*k^2), 4*k / (1 + 2*k^2))

-- Define the dot product of two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1) + (v1.2 * v2.2)

-- Define vector subtraction
def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

-- Theorem statement
theorem constant_dot_product (k : ℝ) :
  ellipse (P k).1 (P k).2 ∧
  dot_product (vector_sub (M k) D) (vector_sub D C) = 0 →
  dot_product (vector_sub (M k) O) (vector_sub (P k) O) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_dot_product_l1095_109595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_angle_measure_l1095_109599

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define points on the circle
def Point (c : Circle) := { p : ℝ × ℝ // (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 }

-- Define the arc measure
noncomputable def arc_measure (c : Circle) (a b : Point c) : ℝ := sorry

-- Define the inscribed angle
noncomputable def inscribed_angle (c : Circle) (a b d : Point c) : ℝ := sorry

-- State the theorem
theorem inscribed_triangle_angle_measure 
  (ω : Circle) (D E F : Point ω) (x : ℝ) :
  arc_measure ω D E = x + 90 ∧ 
  arc_measure ω E F = 2*x + 50 ∧ 
  arc_measure ω F D = 3*x - 40 →
  inscribed_angle ω E F D = 68 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_angle_measure_l1095_109599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_removal_l1095_109525

theorem average_after_removal (numbers : Finset ℝ) (sum : ℝ) : 
  numbers.card = 12 →
  sum = numbers.sum id →
  sum / 12 = 90 →
  65 ∈ numbers →
  85 ∈ numbers →
  ((sum - 65 - 85) / 10 : ℝ) = 93 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_removal_l1095_109525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equality_condition_l1095_109576

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 0 then -x else 2*x - 43

-- State the theorem
theorem g_equality_condition (a : ℝ) :
  a < 0 → (g (g (g 13)) = g (g (g a)) ↔ a = -30) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equality_condition_l1095_109576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_rate_calculation_l1095_109541

noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  (principal * rate * (time : ℝ)) / 100

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate / 100) ^ (time : ℝ) - 1)

theorem compound_interest_rate_calculation 
  (principal_simple : ℝ) 
  (rate_simple : ℝ) 
  (time_simple : ℕ) 
  (principal_compound : ℝ) 
  (time_compound : ℕ) : 
  principal_simple = 1272.000000000001 →
  rate_simple = 10 →
  time_simple = 5 →
  principal_compound = 5000 →
  time_compound = 2 →
  simple_interest principal_simple rate_simple time_simple = 
    (1/2) * compound_interest principal_compound 12 time_compound :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_rate_calculation_l1095_109541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_area_theorem_l1095_109589

/-- Represents a square flag with a symmetric cross -/
structure FlagWithCross where
  side : ℝ
  cross_area_percent : ℝ
  orange_area_percent : ℝ

/-- Conditions for our specific flag -/
def flag_conditions (f : FlagWithCross) : Prop :=
  f.side > 0 ∧ 
  f.cross_area_percent = 49 ∧
  f.orange_area_percent > 0 ∧
  f.orange_area_percent < f.cross_area_percent

/-- Theorem stating that if the cross occupies 49% of the flag, 
    then the orange center occupies 2.45% of the flag -/
theorem orange_area_theorem (f : FlagWithCross) 
  (h : flag_conditions f) : 
  f.orange_area_percent = 2.45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_area_theorem_l1095_109589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_equation_of_l_intersection_range_l1095_109513

noncomputable def C (t : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos (2 * t), 2 * Real.sin t)

def l (ρ θ m : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 3) + m = 0

theorem cartesian_equation_of_l (x y m : ℝ) :
  (∃ ρ θ, l ρ θ m ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ↔ 
  Real.sqrt 3 * x + y + 2 * m = 0 :=
sorry

theorem intersection_range (m : ℝ) :
  (∃ t, (∃ x y, C t = (x, y) ∧ Real.sqrt 3 * x + y + 2 * m = 0)) ↔ 
  -19/12 ≤ m ∧ m ≤ 5/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_equation_of_l_intersection_range_l1095_109513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_disk_in_intersection_l1095_109546

/-- The set M in ℝ² -/
def M : Set (ℝ × ℝ) := {p | p.2 ≥ (1/4) * p.1^2}

/-- The set N in ℝ² -/
def N : Set (ℝ × ℝ) := {p | p.2 ≤ -(1/4) * p.1^2 + p.1 + 7}

/-- The closed disk with center (x₀, y₀) and radius r in ℝ² -/
def D (x₀ y₀ r : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - x₀)^2 + (p.2 - y₀)^2 ≤ r^2}

/-- The largest radius of a disk centered at (1, 4) contained in M ∩ N -/
noncomputable def largest_radius : ℝ := Real.sqrt ((25 - 5 * Real.sqrt 5) / 2)

theorem largest_disk_in_intersection :
  ∀ r, D 1 4 r ⊆ M ∩ N ↔ r ≤ largest_radius :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_disk_in_intersection_l1095_109546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_l1095_109592

/-- Converts meters to kilometers -/
noncomputable def meters_to_km (m : ℝ) : ℝ := m / 1000

/-- Converts minutes and seconds to hours -/
noncomputable def min_sec_to_hours (min : ℝ) (sec : ℝ) : ℝ := (min * 60 + sec) / 3600

/-- Calculates speed in km/hr given distance in meters and time in minutes and seconds -/
noncomputable def speed_km_hr (distance_m : ℝ) (time_min : ℝ) (time_sec : ℝ) : ℝ :=
  let distance_km := meters_to_km distance_m
  let time_hr := min_sec_to_hours time_min time_sec
  distance_km / time_hr

/-- Theorem stating that a cyclist covering 750 m in 2 min 30 sec has a speed of 18 km/hr -/
theorem cyclist_speed : 
  ∀ (ε : ℝ), ε > 0 → |speed_km_hr 750 2 30 - 18| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_l1095_109592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_23_l1095_109590

def mySequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 3
  | n + 1 => mySequence n + 2 * (n + 1)

theorem fifth_term_is_23 : mySequence 4 = 23 := by
  rw [mySequence]
  simp
  norm_num
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_23_l1095_109590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lifting_the_exponent_l1095_109534

theorem lifting_the_exponent (a m : ℕ) (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) 
  (h1 : p^m ∣ (a - 1)) (h2 : ¬(p^(m+1) ∣ (a - 1))) :
  (∀ n : ℕ, p^(m+n) ∣ (a^(p^n) - 1)) ∧ 
  (∀ n : ℕ, ¬(p^(m+n+1) ∣ (a^(p^n) - 1))) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lifting_the_exponent_l1095_109534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convenient_sampling_l1095_109506

/-- Represents the total number of staff members -/
def total_staff : ℕ := 624

/-- Represents the desired sample size as a percentage -/
def sample_percentage : ℚ := 1 / 10

/-- Represents the number of people to be eliminated for convenient sampling -/
def elimination_count : ℕ := 4

/-- Theorem stating that eliminating 4 people results in a convenient sample size -/
theorem convenient_sampling :
  (total_staff - elimination_count) * sample_percentage.num = 
  sample_percentage.den * ((total_staff - elimination_count) / 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convenient_sampling_l1095_109506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1095_109526

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x + 4/x
noncomputable def g (x a : ℝ) : ℝ := 2^x + a

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x₁ ∈ Set.Icc (1/2 : ℝ) 3, ∃ x₂ ∈ Set.Icc 2 3, f x₁ ≥ g x₂ a) →
  a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1095_109526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_author_hardcover_percentage_l1095_109560

/-- Calculates the percentage an author receives from hardcover book sales --/
theorem author_hardcover_percentage
  (paper_copies : ℕ)
  (paper_price : ℚ)
  (paper_percentage : ℚ)
  (hardcover_copies : ℕ)
  (hardcover_price : ℚ)
  (total_earnings : ℚ)
  (h1 : paper_copies = 32000)
  (h2 : paper_price = 1/5)  -- Changed to a rational number representation
  (h3 : paper_percentage = 3/50)  -- Changed to a rational number representation
  (h4 : hardcover_copies = 15000)
  (h5 : hardcover_price = 2/5)  -- Changed to a rational number representation
  (h6 : total_earnings = 1104) :
  (total_earnings - paper_copies * paper_price * paper_percentage) / (hardcover_copies * hardcover_price) = 3/25 := by  -- Changed to a rational number representation
  sorry

-- Remove the #eval line as it's not necessary for building the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_author_hardcover_percentage_l1095_109560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jake_has_nine_peaches_l1095_109531

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 16

/-- The difference in peaches between Steven and Jake -/
def difference : ℕ := 7

/-- The number of peaches Jake has -/
def jake_peaches : ℕ := steven_peaches - difference

/-- Jake has some more peaches than Jill -/
def jake_more_than_jill : Prop := ∃ (jill_peaches : ℕ), jake_peaches > jill_peaches

theorem jake_has_nine_peaches : jake_peaches = 9 := by
  rfl

#eval jake_peaches

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jake_has_nine_peaches_l1095_109531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_latus_rectum_of_parabola_l1095_109544

/-- Predicate to determine if a point (x, y) lies on the latus rectum of the parabola y^2 = 2x -/
def IsLatusRectum (x y : ℝ) : Prop :=
  y^2 = 2*x ∧ x = -1/2

/-- Given a parabola with equation y^2 = 2x, its latus rectum has the equation x = -1/2 -/
theorem latus_rectum_of_parabola (x y : ℝ) :
  y^2 = 2*x → (x = -1/2 ↔ IsLatusRectum x y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_latus_rectum_of_parabola_l1095_109544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_count_l1095_109581

/-- Represents a statement about regression analysis and model fitting -/
inductive RegressionStatement
  | residualPlot
  | correlationIndex
  | sumSquaredResiduals

/-- Determines if a given statement is correct -/
def isCorrectStatement (s : RegressionStatement) : Bool :=
  match s with
  | .residualPlot => false
  | .correlationIndex => true
  | .sumSquaredResiduals => true

/-- The list of all statements to be evaluated -/
def allStatements : List RegressionStatement :=
  [.residualPlot, .correlationIndex, .sumSquaredResiduals]

/-- Counts the number of correct statements in a list -/
def countCorrectStatements (statements : List RegressionStatement) : Nat :=
  statements.filter isCorrectStatement |>.length

theorem correct_statements_count :
  countCorrectStatements allStatements = 2 := by
  -- Proof goes here
  sorry

#eval countCorrectStatements allStatements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_count_l1095_109581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sum_of_five_or_less_squares_l1095_109566

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

def sum_of_squares (n : ℕ) (k : ℕ) : Prop :=
  ∃ (squares : List ℕ), (∀ s ∈ squares, is_perfect_square s) ∧
                        squares.length ≤ k ∧
                        squares.sum = n

def unique_sum_of_squares (n : ℕ) (k : ℕ) : Prop :=
  ∃! (squares : List ℕ), (∀ s ∈ squares, is_perfect_square s) ∧
                         squares.length ≤ k ∧
                         squares.sum = n

theorem unique_sum_of_five_or_less_squares :
  ∀ n : ℕ, unique_sum_of_squares n 5 ↔ n ∈ ({1, 2, 3, 6, 7, 15} : Set ℕ) :=
by
  sorry

#check unique_sum_of_five_or_less_squares

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sum_of_five_or_less_squares_l1095_109566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_l1095_109555

-- Use the built-in complex number type
open Complex

-- Define what it means for a complex number to be pure imaginary
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- State the theorem
theorem pure_imaginary_condition (b : ℝ) :
  is_pure_imaginary (I * (2 + b * I)) → b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_l1095_109555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_lights_on_l1095_109570

-- Define the type for light states
inductive LightState
| On
| Off

-- Define the infinite sequence of lights
def lights : ℕ → LightState := sorry

-- Define the rules for the lights
axiom rule1 : ∀ k : ℕ, lights k = LightState.On → lights (2 * k) = LightState.On ∧ lights (2 * k + 1) = LightState.On
axiom rule2 : ∀ k : ℕ, lights k = LightState.Off → lights (4 * k + 1) = LightState.Off ∧ lights (4 * k + 3) = LightState.Off

-- Given condition: Light 2023 is on
axiom light_2023_on : lights 2023 = LightState.On

-- Theorem to prove
theorem all_lights_on : ∀ n : ℕ, n ≤ 2023 → lights n = LightState.On := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_lights_on_l1095_109570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_holes_needed_l1095_109598

/-- The circumference of the circular flower bed in meters -/
def circumference : ℕ := 300

/-- The initial interval between holes in meters -/
def initial_interval : ℕ := 3

/-- The number of holes initially dug -/
def initial_holes : ℕ := 30

/-- The new interval between holes in meters -/
def new_interval : ℕ := 5

/-- The theorem stating the number of additional holes needed -/
theorem additional_holes_needed : 
  (circumference / new_interval) - (Nat.lcm initial_interval new_interval / initial_interval) = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_holes_needed_l1095_109598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_values_l1095_109529

theorem expression_values :
  ∃ (S : Finset ℤ), (∀ y ∈ S, ∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ y = 10 - 10 * |2*x - 3|) ∧ S.card = 11 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_values_l1095_109529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_correct_l1095_109510

/-- The plane equation coefficients -/
def A : ℤ := 6
def B : ℤ := -6
def C : ℤ := -5
def D : ℤ := -3

/-- The three points on the plane -/
def p₁ : ℝ × ℝ × ℝ := (2, -1, 3)
def p₂ : ℝ × ℝ × ℝ := (-1, 5, 0)
def p₃ : ℝ × ℝ × ℝ := (4, 0, -1)

/-- The plane equation -/
def plane_equation (x y z : ℝ) : Prop :=
  A * x + B * y + C * z + D = 0

theorem plane_equation_correct :
  (A > 0) ∧
  (Nat.gcd (Int.natAbs A) (Int.natAbs B) = Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Int.natAbs C) ∧
   Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Int.natAbs C)) (Int.natAbs D) = 1) ∧
  plane_equation p₁.1 p₁.2.1 p₁.2.2 ∧
  plane_equation p₂.1 p₂.2.1 p₂.2.2 ∧
  plane_equation p₃.1 p₃.2.1 p₃.2.2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_correct_l1095_109510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_arrangement_sum_not_80_l1095_109553

/-- Represents a cube with numbered faces -/
structure Cube where
  faces : Fin 6 → Nat
  valid_faces : ∀ i : Fin 6, faces i ∈ ({1, 2, 3, 4, 5, 6} : Set Nat)

/-- Represents an arrangement of 8 cubes in a 2 × 2 × 2 formation -/
def CubeArrangement := Fin 8 → Cube

/-- The sum of visible faces in a 2 × 2 × 2 cube arrangement -/
def visible_sum (arrangement : CubeArrangement) : Nat :=
  sorry

theorem cube_arrangement_sum_not_80 :
  ∀ (arrangement : CubeArrangement),
    visible_sum arrangement ≠ 80 := by
  sorry

#check cube_arrangement_sum_not_80

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_arrangement_sum_not_80_l1095_109553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolas_intersection_l1095_109593

-- Define the two parabolas
def f (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 1
def g (x : ℝ) : ℝ := 2 * x^2 - 5 * x + 3

-- Define the intersection points
noncomputable def p1 : ℝ × ℝ := (2 + Real.sqrt 6, 13 + 3 * Real.sqrt 6)
noncomputable def p2 : ℝ × ℝ := (2 - Real.sqrt 6, 13 - 3 * Real.sqrt 6)

theorem parabolas_intersection :
  (∀ x y : ℝ, f x = g x ∧ y = f x → (x, y) = p1 ∨ (x, y) = p2) ∧
  f p1.1 = g p1.1 ∧ f p2.1 = g p2.1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolas_intersection_l1095_109593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_twelve_percent_l1095_109571

/-- Calculates the interest rate given the principal, time, and total interest paid -/
noncomputable def calculate_interest_rate (principal : ℝ) (time : ℝ) (total_interest : ℝ) : ℝ :=
  total_interest / (principal * time)

/-- Proves that the interest rate is 0.12 given the problem conditions -/
theorem interest_rate_is_twelve_percent 
  (principal : ℝ) 
  (time : ℝ) 
  (total_interest : ℝ) 
  (h1 : principal = 10000)
  (h2 : time = 3)
  (h3 : total_interest = 3600) :
  calculate_interest_rate principal time total_interest = 0.12 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_interest_rate 10000 3 3600

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_twelve_percent_l1095_109571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_f_l1095_109548

noncomputable def f (x φ : ℝ) : ℝ := -2 * Real.tan (2 * x + φ)

theorem decreasing_interval_of_f (φ : ℝ) (h1 : |φ| < π) (h2 : f (π/16) φ = -2) :
  ∃ (a b : ℝ), a = 3*π/16 ∧ b = 11*π/16 ∧ 
  ∀ (x y : ℝ), a < x ∧ x < y ∧ y < b → f x φ > f y φ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_f_l1095_109548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l1095_109501

/-- The equation (z+6)^8 = 81 where z is a complex number --/
def equation (z : ℂ) : Prop := (z + 6) ^ 8 = 81

/-- The set of solutions to the equation --/
def solutions : Set ℂ := {z : ℂ | equation z}

/-- The solutions form a regular octagon in the complex plane --/
axiom solutions_form_octagon : ∃ (center : ℂ) (radius : ℝ), 
  ∀ z ∈ solutions, ∃ k : Fin 8, z = center + radius * Complex.exp (2 * Real.pi * Complex.I * (k : ℝ) / 8)

/-- The area of a triangle formed by three points in the complex plane --/
noncomputable def triangle_area (a b c : ℂ) : ℝ := 
  abs ((b.re - a.re) * (c.im - a.im) - (c.re - a.re) * (b.im - a.im)) / 2

/-- The theorem stating the minimum area of a triangle formed by any three vertices of the octagon --/
theorem min_triangle_area : 
  ∃ (a b c : ℂ), a ∈ solutions ∧ b ∈ solutions ∧ c ∈ solutions ∧
    (∀ (x y z : ℂ), x ∈ solutions → y ∈ solutions → z ∈ solutions →
      triangle_area a b c ≤ triangle_area x y z) ∧
    triangle_area a b c = (3 * Real.sqrt 2 - 3) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l1095_109501
