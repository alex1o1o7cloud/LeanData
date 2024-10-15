import Mathlib

namespace NUMINAMATH_CALUDE_inverse_sum_reciprocal_l2767_276745

theorem inverse_sum_reciprocal (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x⁻¹ + y⁻¹)⁻¹ = (x * y) / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_reciprocal_l2767_276745


namespace NUMINAMATH_CALUDE_function_relationship_l2767_276749

theorem function_relationship (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → a^x > a^y) →
  (∀ x y : ℝ, x < y → (2 - a) * x^3 < (2 - a) * y^3) ∧
  ¬(∀ x y : ℝ, x < y → (2 - a) * x^3 < (2 - a) * y^3 →
    ∀ x y : ℝ, x < y → a^x > a^y) :=
by sorry

end NUMINAMATH_CALUDE_function_relationship_l2767_276749


namespace NUMINAMATH_CALUDE_sin_double_angle_special_point_l2767_276726

/-- Given an angle θ in standard position with its terminal side passing through the point (1, -2),
    prove that sin(2θ) = -4/5 -/
theorem sin_double_angle_special_point :
  ∀ θ : Real,
  (∃ (r : Real), r > 0 ∧ r * Real.cos θ = 1 ∧ r * Real.sin θ = -2) →
  Real.sin (2 * θ) = -4/5 := by
sorry

end NUMINAMATH_CALUDE_sin_double_angle_special_point_l2767_276726


namespace NUMINAMATH_CALUDE_upstream_speed_calculation_l2767_276748

/-- Represents the speed of a man rowing in different water conditions -/
structure RowingSpeed where
  stillWater : ℝ
  downstream : ℝ

/-- Calculates the speed of the man rowing upstream given his rowing speeds in still water and downstream -/
def upstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.downstream

/-- Theorem stating that given the man's speed in still water is 20 kmph and downstream is 33 kmph, his upstream speed is 7 kmph -/
theorem upstream_speed_calculation (s : RowingSpeed) 
  (h1 : s.stillWater = 20) 
  (h2 : s.downstream = 33) : 
  upstreamSpeed s = 7 := by
  sorry

#eval upstreamSpeed { stillWater := 20, downstream := 33 }

end NUMINAMATH_CALUDE_upstream_speed_calculation_l2767_276748


namespace NUMINAMATH_CALUDE_sum_of_odd_numbers_sum_of_cubes_l2767_276771

def u (n : ℕ) : ℕ := (2 * n - 1) + if n = 0 then 0 else u (n - 1)

def S (n : ℕ) : ℕ := n^3 + if n = 0 then 0 else S (n - 1)

theorem sum_of_odd_numbers (n : ℕ) : u n = n^2 := by
  sorry

theorem sum_of_cubes (n : ℕ) : S n = (n * (n + 1) / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_odd_numbers_sum_of_cubes_l2767_276771


namespace NUMINAMATH_CALUDE_line_intersects_circle_midpoint_trajectory_line_equations_with_ratio_l2767_276783

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

-- Define the line L
def line_L (m x y : ℝ) : Prop := m * x - y + 1 - m = 0

-- Define the fixed point P
def point_P : ℝ × ℝ := (1, 1)

-- Theorem 1: Line L always intersects circle C at two distinct points
theorem line_intersects_circle (m : ℝ) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_L m x₁ y₁ ∧ line_L m x₂ y₂ :=
sorry

-- Theorem 2: Trajectory of midpoint M
theorem midpoint_trajectory (x y : ℝ) :
  (∃ (m : ℝ), ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_L m x₁ y₁ ∧ line_L m x₂ y₂ ∧
    x = (x₁ + x₂) / 2 ∧ y = (y₁ + y₂) / 2) ↔
  x^2 + y^2 - x - 2*y + 1 = 0 :=
sorry

-- Theorem 3: Equations of line L when P divides AB in 1:2 ratio
theorem line_equations_with_ratio :
  ∃ (m : ℝ), ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_L m x₁ y₁ ∧ line_L m x₂ y₂ ∧
    2 * (point_P.1 - x₁) = x₂ - point_P.1 ∧
    2 * (point_P.2 - y₁) = y₂ - point_P.2 ↔
  (∀ x y, x - y = 0 ∨ x + y - 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_midpoint_trajectory_line_equations_with_ratio_l2767_276783


namespace NUMINAMATH_CALUDE_tan_five_pi_over_four_l2767_276723

theorem tan_five_pi_over_four : Real.tan (5 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_five_pi_over_four_l2767_276723


namespace NUMINAMATH_CALUDE_right_isosceles_triangle_special_property_l2767_276703

/-- A right isosceles triangle with the given property has 45° acute angles -/
theorem right_isosceles_triangle_special_property (a h : ℝ) (θ : ℝ) : 
  a > 0 → -- The leg length is positive
  h > 0 → -- The hypotenuse length is positive
  h = a * Real.sqrt 2 → -- Right isosceles triangle property
  h^2 = 3 * a * Real.sin θ → -- Given special property
  θ = π/4 := by -- Conclusion: acute angle is 45° (π/4 radians)
sorry

end NUMINAMATH_CALUDE_right_isosceles_triangle_special_property_l2767_276703


namespace NUMINAMATH_CALUDE_completing_square_result_l2767_276725

/-- Represents the completing the square method applied to a quadratic equation -/
def completing_square (a b c : ℝ) : ℝ × ℝ := sorry

theorem completing_square_result :
  let (p, q) := completing_square 1 4 3
  p = 2 ∧ q = 1 := by sorry

end NUMINAMATH_CALUDE_completing_square_result_l2767_276725


namespace NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l2767_276797

/-- Given vectors a and b in ℝ², if a ⊥ b, then |a| = 2 -/
theorem perpendicular_vectors_magnitude (x : ℝ) :
  let a : ℝ × ℝ := (x, Real.sqrt 3)
  let b : ℝ × ℝ := (3, -Real.sqrt 3)
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- a ⊥ b condition
  Real.sqrt (a.1^2 + a.2^2) = 2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l2767_276797


namespace NUMINAMATH_CALUDE_belinda_age_l2767_276790

theorem belinda_age (tony_age belinda_age : ℕ) : 
  tony_age + belinda_age = 56 →
  belinda_age = 2 * tony_age + 8 →
  tony_age = 16 →
  belinda_age = 40 := by
sorry

end NUMINAMATH_CALUDE_belinda_age_l2767_276790


namespace NUMINAMATH_CALUDE_line_on_plane_perp_other_plane_implies_planes_perp_l2767_276742

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- The line lies on the plane -/
def lies_on (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- The line is perpendicular to the plane -/
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Two planes are perpendicular -/
def perpendicular_planes (p1 p2 : Plane3D) : Prop :=
  sorry

theorem line_on_plane_perp_other_plane_implies_planes_perp
  (l : Line3D) (α β : Plane3D) :
  lies_on l α → perpendicular_line_plane l β → perpendicular_planes α β :=
by sorry

end NUMINAMATH_CALUDE_line_on_plane_perp_other_plane_implies_planes_perp_l2767_276742


namespace NUMINAMATH_CALUDE_rotated_angle_measure_l2767_276755

/-- Given an angle of 60 degrees rotated 540 degrees clockwise, 
    the resulting new acute angle is also 60 degrees. -/
theorem rotated_angle_measure (initial_angle rotation : ℝ) : 
  initial_angle = 60 → 
  rotation = 540 → 
  (rotation % 360 - initial_angle) % 180 = 60 := by
sorry

end NUMINAMATH_CALUDE_rotated_angle_measure_l2767_276755


namespace NUMINAMATH_CALUDE_shoes_count_l2767_276717

/-- The total number of pairs of shoes Ellie and Riley have together -/
def total_shoes (ellie_shoes : ℕ) (riley_difference : ℕ) : ℕ :=
  ellie_shoes + (ellie_shoes - riley_difference)

/-- Theorem stating that given Ellie has 8 pairs of shoes and Riley has 3 fewer pairs,
    they have 13 pairs of shoes in total -/
theorem shoes_count : total_shoes 8 3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_shoes_count_l2767_276717


namespace NUMINAMATH_CALUDE_rhombus_area_l2767_276733

/-- Theorem: Area of a rhombus with given side and diagonal lengths -/
theorem rhombus_area (side : ℝ) (diagonal1 : ℝ) (area : ℝ) : 
  side = 26 → diagonal1 = 20 → area = 480 → 
  ∃ (diagonal2 : ℝ), 
    diagonal2 ^ 2 = 4 * (side ^ 2 - (diagonal1 / 2) ^ 2) ∧ 
    area = (diagonal1 * diagonal2) / 2 := by
  sorry


end NUMINAMATH_CALUDE_rhombus_area_l2767_276733


namespace NUMINAMATH_CALUDE_locus_P_is_correct_l2767_276709

/-- The locus of points P that are the second intersection of line OM and circle OAN,
    where O is the center of a circle with radius r, A(c, 0) is a point on its diameter,
    and M and N are symmetrical points on the circle with respect to OA. -/
def locus_P (r c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; (x^2 + y^2 - 2*c*x)^2 - r^2*(x^2 + y^2) = 0}

/-- Theorem stating that the locus_P is the correct description of the geometric locus. -/
theorem locus_P_is_correct (r c : ℝ) (hr : r > 0) (hc : c ≠ 0) :
  ∀ p : ℝ × ℝ, p ∈ locus_P r c ↔ 
    ∃ (m n : ℝ × ℝ),
      (∀ x y, (x, y) = m → x^2 + y^2 = r^2) ∧
      (∀ x y, (x, y) = n → x^2 + y^2 = r^2) ∧
      (∃ t, m.1 = t * n.1 ∧ m.2 = -t * n.2) ∧
      (∃ s, p = (s * m.1, s * m.2)) ∧
      (∃ u v, p.1^2 + p.2^2 + 2*u*p.1 + 2*v*p.2 = 0 ∧
              c^2 + 2*u*c = 0 ∧
              0^2 + 0^2 + 2*u*0 + 2*v*0 = 0) :=
by sorry

end NUMINAMATH_CALUDE_locus_P_is_correct_l2767_276709


namespace NUMINAMATH_CALUDE_gcd_12a_18b_min_l2767_276773

theorem gcd_12a_18b_min (a b : ℕ+) (h : Nat.gcd a b = 9) :
  (∃ (a' b' : ℕ+), Nat.gcd a' b' = 9 ∧ Nat.gcd (12 * a') (18 * b') = 54) ∧
  (Nat.gcd (12 * a) (18 * b) ≥ 54) :=
sorry

end NUMINAMATH_CALUDE_gcd_12a_18b_min_l2767_276773


namespace NUMINAMATH_CALUDE_circle_diameter_problem_l2767_276753

/-- Given two circles A and C inside a larger circle B, prove the diameter of A -/
theorem circle_diameter_problem (R B r : ℝ) : 
  R = 10 → -- Radius of circle B is 10 cm (half the diameter of 20 cm)
  100 * Real.pi - 2 * Real.pi * r^2 = 5 * (Real.pi * r^2) → -- Ratio of shaded area to area of A is 5:1
  (2 * r : ℝ) = 2 * Real.sqrt (100 / 7) := by
  sorry

#check circle_diameter_problem

end NUMINAMATH_CALUDE_circle_diameter_problem_l2767_276753


namespace NUMINAMATH_CALUDE_parallelogram_height_l2767_276732

/-- Given a parallelogram with area 180 square centimeters and base 18 cm, its height is 10 cm. -/
theorem parallelogram_height (area : ℝ) (base : ℝ) (height : ℝ) : 
  area = 180 → base = 18 → area = base * height → height = 10 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l2767_276732


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2767_276702

-- Problem 1
theorem problem_1 (a b : ℝ) : (a * b) ^ 6 / (a * b) ^ 2 * (a * b) ^ 4 = a ^ 8 * b ^ 8 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) : (3 * x ^ 3) ^ 2 * x ^ 5 - (-x ^ 2) ^ 6 / x = 8 * x ^ 11 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2767_276702


namespace NUMINAMATH_CALUDE_keatons_annual_profit_l2767_276743

/-- Represents a fruit type with its harvest frequency, selling price, and cost --/
structure Fruit where
  harvestFrequency : Nat
  sellingPrice : Nat
  harvestCost : Nat

/-- Calculates the annual profit for a single fruit type --/
def annualProfit (fruit : Fruit) : Nat :=
  let harvestsPerYear := 12 / fruit.harvestFrequency
  let profitPerHarvest := fruit.sellingPrice - fruit.harvestCost
  harvestsPerYear * profitPerHarvest

/-- Keaton's farm setup --/
def oranges : Fruit := ⟨2, 50, 20⟩
def apples : Fruit := ⟨3, 30, 15⟩
def peaches : Fruit := ⟨4, 45, 25⟩
def blackberries : Fruit := ⟨6, 70, 30⟩

/-- Theorem: Keaton's total annual profit is $380 --/
theorem keatons_annual_profit :
  annualProfit oranges + annualProfit apples + annualProfit peaches + annualProfit blackberries = 380 := by
  sorry

end NUMINAMATH_CALUDE_keatons_annual_profit_l2767_276743


namespace NUMINAMATH_CALUDE_pencil_division_l2767_276782

theorem pencil_division (num_students num_pencils : ℕ) 
  (h1 : num_students = 2) 
  (h2 : num_pencils = 18) : 
  num_pencils / num_students = 9 := by
sorry

end NUMINAMATH_CALUDE_pencil_division_l2767_276782


namespace NUMINAMATH_CALUDE_locus_C_is_ellipse_l2767_276738

/-- Circle O₁ with equation (x-1)² + y² = 1 -/
def circle_O₁ : Set (ℝ × ℝ) :=
  {p | (p.1 - 1)^2 + p.2^2 = 1}

/-- Circle O₂ with equation (x+1)² + y² = 16 -/
def circle_O₂ : Set (ℝ × ℝ) :=
  {p | (p.1 + 1)^2 + p.2^2 = 16}

/-- The set of points P(x, y) that represent the center of circle C -/
def locus_C : Set (ℝ × ℝ) :=
  {p | ∃ r > 0,
    (∀ q ∈ circle_O₁, (p.1 - q.1)^2 + (p.2 - q.2)^2 = (r + 1)^2) ∧
    (∀ q ∈ circle_O₂, (p.1 - q.1)^2 + (p.2 - q.2)^2 = (4 - r)^2)}

/-- Theorem stating that the locus of the center of circle C is an ellipse -/
theorem locus_C_is_ellipse : ∃ a b c d e f : ℝ,
  a > 0 ∧ b^2 < 4 * a * c ∧
  locus_C = {p | a * p.1^2 + b * p.1 * p.2 + c * p.2^2 + d * p.1 + e * p.2 + f = 0} :=
sorry

end NUMINAMATH_CALUDE_locus_C_is_ellipse_l2767_276738


namespace NUMINAMATH_CALUDE_article_gain_percentage_l2767_276705

/-- Calculates the percentage gain when selling an article -/
def percentageGain (costPrice sellingPrice : ℚ) : ℚ :=
  (sellingPrice - costPrice) / costPrice * 100

/-- Calculates the cost price given a selling price and loss percentage -/
def calculateCostPrice (sellingPrice : ℚ) (lossPercentage : ℚ) : ℚ :=
  sellingPrice / (1 - lossPercentage / 100)

theorem article_gain_percentage :
  let lossPrice : ℚ := 102
  let gainPrice : ℚ := 144
  let lossPercentage : ℚ := 15
  let costPrice := calculateCostPrice lossPrice lossPercentage
  percentageGain costPrice gainPrice = 20 := by sorry

end NUMINAMATH_CALUDE_article_gain_percentage_l2767_276705


namespace NUMINAMATH_CALUDE_deductive_reasoning_form_not_sufficient_l2767_276770

/-- A structure representing a deductive argument --/
structure DeductiveArgument where
  premises : List Prop
  conclusion : Prop
  form_correct : Bool

/-- A predicate that determines if a deductive argument is valid --/
def is_valid (arg : DeductiveArgument) : Prop :=
  arg.form_correct ∧ (∀ p ∈ arg.premises, p) → arg.conclusion

/-- Theorem stating that conforming to the form of deductive reasoning alone
    is not sufficient to guarantee the correctness of the conclusion --/
theorem deductive_reasoning_form_not_sufficient :
  ∃ (arg : DeductiveArgument), arg.form_correct ∧ ¬arg.conclusion :=
sorry

end NUMINAMATH_CALUDE_deductive_reasoning_form_not_sufficient_l2767_276770


namespace NUMINAMATH_CALUDE_special_hexagon_side_length_l2767_276756

/-- An equilateral hexagon with specific properties -/
structure SpecialHexagon where
  -- Side length of the hexagon
  side_length : ℝ
  -- Three nonadjacent acute interior angles measure 45°
  has_45_degree_angles : Prop
  -- The enclosed area of the hexagon
  area : ℝ
  -- The hexagon is equilateral
  is_equilateral : Prop
  -- The area is 9√2
  area_is_9_sqrt_2 : area = 9 * Real.sqrt 2

/-- Theorem stating that a hexagon with the given properties has a side length of 2√3 -/
theorem special_hexagon_side_length 
  (h : SpecialHexagon) : h.side_length = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_special_hexagon_side_length_l2767_276756


namespace NUMINAMATH_CALUDE_minimum_fuse_length_l2767_276701

theorem minimum_fuse_length (safe_distance : ℝ) (personnel_speed : ℝ) (fuse_burn_speed : ℝ) :
  safe_distance = 70 →
  personnel_speed = 7 →
  fuse_burn_speed = 10.3 →
  ∃ x : ℝ, x > 103 ∧ x / fuse_burn_speed > safe_distance / personnel_speed :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_fuse_length_l2767_276701


namespace NUMINAMATH_CALUDE_parabola_point_focus_distance_l2767_276722

theorem parabola_point_focus_distance (p m : ℝ) : 
  p > 0 → 
  m^2 = 2*p*4 → 
  (4 + p/2)^2 + m^2 = (17/4)^2 → 
  p = 1/2 ∧ (m = 2 ∨ m = -2) := by
sorry

end NUMINAMATH_CALUDE_parabola_point_focus_distance_l2767_276722


namespace NUMINAMATH_CALUDE_rectangle_width_l2767_276754

theorem rectangle_width (a w l d : ℝ) : 
  a > 0 → w > 0 → l > 0 → d > 0 →
  a = w * l →
  d^2 = w^2 + l^2 →
  a = 12 →
  d = 5 →
  w = 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_l2767_276754


namespace NUMINAMATH_CALUDE_four_possible_d_values_l2767_276791

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the addition of two 5-digit numbers resulting in another 5-digit number -/
def ValidAddition (a b c d : Digit) : Prop :=
  ∃ (n : ℕ), n < 100000 ∧
  10000 * a.val + 1000 * b.val + 100 * c.val + 10 * d.val + a.val +
  10000 * c.val + 1000 * b.val + 100 * a.val + 10 * d.val + d.val =
  10000 * d.val + 1000 * d.val + 100 * d.val + 10 * c.val + b.val

/-- The main theorem stating that there are exactly 4 possible values for D -/
theorem four_possible_d_values :
  ∃! (s : Finset Digit), s.card = 4 ∧
  ∀ d : Digit, d ∈ s ↔
    ∃ (a b c : Digit), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ d ∧ a ≠ c ∧ b ≠ d ∧
    ValidAddition a b c d :=
sorry

end NUMINAMATH_CALUDE_four_possible_d_values_l2767_276791


namespace NUMINAMATH_CALUDE_line_in_quadrants_l2767_276789

-- Define a line y = kx + b
structure Line where
  k : ℝ
  b : ℝ

-- Define quadrants
inductive Quadrant
  | first
  | second
  | third
  | fourth

-- Define a function to check if a line passes through a quadrant
def passesThrough (l : Line) (q : Quadrant) : Prop := sorry

-- Theorem statement
theorem line_in_quadrants (l : Line) :
  passesThrough l Quadrant.first ∧ 
  passesThrough l Quadrant.third ∧ 
  passesThrough l Quadrant.fourth →
  l.k > 0 := by sorry

end NUMINAMATH_CALUDE_line_in_quadrants_l2767_276789


namespace NUMINAMATH_CALUDE_smallest_five_digit_multiple_of_3_and_4_l2767_276708

theorem smallest_five_digit_multiple_of_3_and_4 : ∃ n : ℕ, 
  (n ≥ 10000 ∧ n < 100000) ∧ 
  3 ∣ n ∧ 
  4 ∣ n ∧ 
  (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) → 3 ∣ m → 4 ∣ m → m ≥ n) ∧
  n = 10008 :=
sorry

end NUMINAMATH_CALUDE_smallest_five_digit_multiple_of_3_and_4_l2767_276708


namespace NUMINAMATH_CALUDE_equation_system_solutions_l2767_276794

/-- The system of equations has two types of solutions:
    1. (3, 5, 7, 9)
    2. (t, -t, t, -t) for any real t -/
theorem equation_system_solutions :
  ∀ (a b c d : ℝ),
    (a * b + a * c = 3 * b + 3 * c) ∧
    (b * c + b * d = 5 * c + 5 * d) ∧
    (a * c + c * d = 7 * a + 7 * d) ∧
    (a * d + b * d = 9 * a + 9 * b) →
    ((a = 3 ∧ b = 5 ∧ c = 7 ∧ d = 9) ∨
     (∃ t : ℝ, a = t ∧ b = -t ∧ c = t ∧ d = -t)) :=
by sorry

end NUMINAMATH_CALUDE_equation_system_solutions_l2767_276794


namespace NUMINAMATH_CALUDE_driver_speed_proof_l2767_276774

theorem driver_speed_proof (v : ℝ) : v > 0 → v / (v + 12) = 2/3 → v = 24 := by
  sorry

end NUMINAMATH_CALUDE_driver_speed_proof_l2767_276774


namespace NUMINAMATH_CALUDE_quadratic_minimum_ratio_bound_l2767_276728

-- Define a quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the derivative of the quadratic function
def quadratic_derivative (a b : ℝ) (x : ℝ) : ℝ := 2 * a * x + b

-- Define the second derivative of the quadratic function
def quadratic_second_derivative (a : ℝ) : ℝ := 2 * a

theorem quadratic_minimum_ratio_bound (a b c : ℝ) :
  a > 0 →  -- Ensures the function is concave up
  quadratic_derivative a b 0 > 0 →  -- f'(0) > 0
  (∀ x : ℝ, quadratic a b c x ≥ 0) →  -- f(x) ≥ 0 for all real x
  (quadratic a b c 1) / (quadratic_second_derivative a) ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_ratio_bound_l2767_276728


namespace NUMINAMATH_CALUDE_computer_price_increase_l2767_276778

theorem computer_price_increase (d : ℝ) (h1 : 2 * d = 580) : 
  d * (1 + 0.3) = 377 := by sorry

end NUMINAMATH_CALUDE_computer_price_increase_l2767_276778


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2767_276721

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -1)

theorem perpendicular_vectors (x : ℝ) :
  (vector_a.1 * (vector_b x).1 + vector_a.2 * (vector_b x).2 = 0) →
  x = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2767_276721


namespace NUMINAMATH_CALUDE_order_of_abc_l2767_276707

theorem order_of_abc : 
  let a : ℝ := 2017^0
  let b : ℝ := 2015 * 2017 - 2016^2
  let c : ℝ := (-2/3)^2016 * (3/2)^2017
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l2767_276707


namespace NUMINAMATH_CALUDE_total_cash_realized_l2767_276793

/-- Calculates the cash realized from selling a stock -/
def cashRealized (value : ℝ) (returnRate : ℝ) (brokerageFeeRate : ℝ) : ℝ :=
  let grossValue := value * (1 + returnRate)
  grossValue * (1 - brokerageFeeRate)

/-- Theorem: The total cash realized from selling all three stocks is $65,120.75 -/
theorem total_cash_realized :
  let stockA := cashRealized 10000 0.14 0.0025
  let stockB := cashRealized 20000 0.10 0.005
  let stockC := cashRealized 30000 0.07 0.0075
  stockA + stockB + stockC = 65120.75 := by
  sorry

end NUMINAMATH_CALUDE_total_cash_realized_l2767_276793


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_on_circle_l2767_276780

theorem sum_of_x_and_y_on_circle (x y : ℝ) (h : x^2 + y^2 = 12*x - 8*y - 48) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_on_circle_l2767_276780


namespace NUMINAMATH_CALUDE_additional_distance_for_average_speed_l2767_276711

def initial_distance : ℝ := 20
def initial_speed : ℝ := 25
def second_speed : ℝ := 40
def desired_average_speed : ℝ := 35

theorem additional_distance_for_average_speed :
  ∃ (additional_distance : ℝ),
    (initial_distance + additional_distance) / ((initial_distance / initial_speed) + (additional_distance / second_speed)) = desired_average_speed ∧
    additional_distance = 64 := by
  sorry

end NUMINAMATH_CALUDE_additional_distance_for_average_speed_l2767_276711


namespace NUMINAMATH_CALUDE_fraction_value_l2767_276729

theorem fraction_value (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (x^2 - 2*x + 1) / (x^2 - 1) = 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2767_276729


namespace NUMINAMATH_CALUDE_reading_time_reduction_xiao_yu_reading_time_l2767_276704

/-- Represents the number of days to read a book given the pages per day -/
def days_to_read (total_pages : ℕ) (pages_per_day : ℕ) : ℕ :=
  total_pages / pages_per_day

/-- The theorem stating the relationship between reading rates and days to finish the book -/
theorem reading_time_reduction (initial_pages_per_day : ℕ) (initial_days : ℕ) (additional_pages : ℕ) :
  initial_pages_per_day > 0 →
  initial_days > 0 →
  additional_pages > 0 →
  days_to_read (initial_pages_per_day * initial_days) (initial_pages_per_day + additional_pages) =
    initial_days * initial_pages_per_day / (initial_pages_per_day + additional_pages) :=
by
  sorry

/-- The specific instance of the theorem for the given problem -/
theorem xiao_yu_reading_time :
  days_to_read (15 * 24) (15 + 3) = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_reading_time_reduction_xiao_yu_reading_time_l2767_276704


namespace NUMINAMATH_CALUDE_equation_solution_l2767_276713

theorem equation_solution : ∃! x : ℚ, (3 * x - 15) / 4 = (x + 9) / 5 ∧ x = 10 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2767_276713


namespace NUMINAMATH_CALUDE_at_least_one_not_greater_than_third_l2767_276765

theorem at_least_one_not_greater_than_third (a b c : ℝ) (h : a + b + c = 1) :
  min a (min b c) ≤ 1/3 := by sorry

end NUMINAMATH_CALUDE_at_least_one_not_greater_than_third_l2767_276765


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2767_276776

def set_A : Set Int := {x | |x| < 3}
def set_B : Set Int := {x | |x| > 1}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {-2, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2767_276776


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2767_276734

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ a b c d : ℝ, a > 0 → b > 0 → c > 0 → d > 0 → a * b * c * d = 1 →
    (f a + f b) * (f c + f d) = (a + b) * (c + d)

/-- The main theorem stating that any function satisfying the equation
    must be either the identity function or its reciprocal -/
theorem functional_equation_solution (f : ℝ → ℝ) 
    (hf : ∀ x : ℝ, x > 0 → f x > 0) 
    (heq : SatisfiesEquation f) :
    (∀ x : ℝ, x > 0 → f x = x) ∨ (∀ x : ℝ, x > 0 → f x = 1 / x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2767_276734


namespace NUMINAMATH_CALUDE_people_off_first_stop_l2767_276714

/-- Represents the number of people who got off at the first stop -/
def first_stop_off : ℕ := sorry

/-- The initial number of people on the bus -/
def initial_people : ℕ := 50

/-- The number of people who got off at the second stop -/
def second_stop_off : ℕ := 8

/-- The number of people who got on at the second stop -/
def second_stop_on : ℕ := 2

/-- The number of people who got off at the third stop -/
def third_stop_off : ℕ := 4

/-- The number of people who got on at the third stop -/
def third_stop_on : ℕ := 3

/-- The final number of people on the bus after the third stop -/
def final_people : ℕ := 28

theorem people_off_first_stop :
  initial_people - first_stop_off - (second_stop_off - second_stop_on) - (third_stop_off - third_stop_on) = final_people ∧
  first_stop_off = 15 := by sorry

end NUMINAMATH_CALUDE_people_off_first_stop_l2767_276714


namespace NUMINAMATH_CALUDE_semicircle_radius_in_isosceles_triangle_exists_isosceles_triangle_with_inscribed_semicircle_l2767_276777

/-- An isosceles triangle with a semicircle inscribed along its base -/
structure IsoscelesTriangleWithInscribedSemicircle where
  /-- The base of the isosceles triangle -/
  base : ℝ
  /-- The height of the isosceles triangle -/
  height : ℝ
  /-- The radius of the inscribed semicircle -/
  radius : ℝ

/-- Theorem stating the relationship between the triangle's dimensions and the semicircle's radius -/
theorem semicircle_radius_in_isosceles_triangle 
  (triangle : IsoscelesTriangleWithInscribedSemicircle) 
  (h_base : triangle.base = 20) 
  (h_height : triangle.height = 18) : 
  triangle.radius = 90 / Real.sqrt 106 := by
  sorry

/-- Existence of the isosceles triangle with the given properties -/
theorem exists_isosceles_triangle_with_inscribed_semicircle :
  ∃ (triangle : IsoscelesTriangleWithInscribedSemicircle), 
    triangle.base = 20 ∧ 
    triangle.height = 18 ∧ 
    triangle.radius = 90 / Real.sqrt 106 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_in_isosceles_triangle_exists_isosceles_triangle_with_inscribed_semicircle_l2767_276777


namespace NUMINAMATH_CALUDE_genevieve_coffee_consumption_l2767_276736

/-- Proves that Genevieve drank 6 pints of coffee given the conditions -/
theorem genevieve_coffee_consumption 
  (total_coffee : ℚ) 
  (num_thermoses : ℕ) 
  (genevieve_thermoses : ℕ) 
  (h1 : total_coffee = 4.5) 
  (h2 : num_thermoses = 18) 
  (h3 : genevieve_thermoses = 3) 
  (h4 : ∀ g : ℚ, g * 8 = g * (8 : ℚ)) -- Conversion from gallons to pints
  : (total_coffee * 8 * genevieve_thermoses) / num_thermoses = 6 := by
  sorry

end NUMINAMATH_CALUDE_genevieve_coffee_consumption_l2767_276736


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l2767_276737

theorem complex_number_in_third_quadrant : 
  let z : ℂ := (1 - Complex.I)^2 / (1 + Complex.I)
  (z.re < 0) ∧ (z.im < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l2767_276737


namespace NUMINAMATH_CALUDE_number_puzzle_l2767_276746

theorem number_puzzle : ∃ x : ℝ, 3 * (x + 2) = 24 + x ∧ x = 9 := by sorry

end NUMINAMATH_CALUDE_number_puzzle_l2767_276746


namespace NUMINAMATH_CALUDE_new_premium_calculation_l2767_276763

def calculate_new_premium (initial_premium : ℝ) (accident_increase_percent : ℝ) 
  (ticket_increase : ℝ) (late_payment_increase : ℝ) (num_accidents : ℕ) 
  (num_tickets : ℕ) (num_late_payments : ℕ) : ℝ :=
  initial_premium + 
  (initial_premium * accident_increase_percent * num_accidents : ℝ) +
  (ticket_increase * num_tickets) +
  (late_payment_increase * num_late_payments)

theorem new_premium_calculation :
  calculate_new_premium 125 0.12 7 15 2 4 3 = 228 := by
  sorry

end NUMINAMATH_CALUDE_new_premium_calculation_l2767_276763


namespace NUMINAMATH_CALUDE_max_gemstone_value_is_72_l2767_276712

/-- Represents a type of gemstone with its weight and value --/
structure Gemstone where
  weight : ℕ
  value : ℕ

/-- The problem setup --/
def treasureHuntProblem :=
  let sapphire : Gemstone := ⟨6, 15⟩
  let ruby : Gemstone := ⟨3, 9⟩
  let diamond : Gemstone := ⟨2, 5⟩
  let maxWeight : ℕ := 24
  let minEachType : ℕ := 10
  (sapphire, ruby, diamond, maxWeight, minEachType)

/-- The maximum value of gemstones that can be carried --/
def maxGemstoneValue (problem : Gemstone × Gemstone × Gemstone × ℕ × ℕ) : ℕ :=
  sorry

/-- Theorem stating the maximum value is 72 --/
theorem max_gemstone_value_is_72 :
  maxGemstoneValue treasureHuntProblem = 72 := by
  sorry

end NUMINAMATH_CALUDE_max_gemstone_value_is_72_l2767_276712


namespace NUMINAMATH_CALUDE_ellipse_midpoint_property_l2767_276741

noncomputable section

-- Define the ellipse C
def C : Set (ℝ × ℝ) := {p | p.1^2 / 3 + p.2^2 = 1}

-- Define vertices A₁ and A₂
def A₁ : ℝ × ℝ := (-Real.sqrt 3, 0)
def A₂ : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define the line x = -2√3
def line_P : Set (ℝ × ℝ) := {p | p.1 = -2 * Real.sqrt 3}

-- Main theorem
theorem ellipse_midpoint_property 
  (P : ℝ × ℝ) 
  (h_P : P ∈ line_P ∧ P.2 ≠ 0) 
  (M N : ℝ × ℝ) 
  (h_M : M ∈ C ∧ ∃ t : ℝ, M = (1 - t) • P + t • A₂) 
  (h_N : N ∈ C ∧ ∃ s : ℝ, N = (1 - s) • P + s • A₁) 
  (Q : ℝ × ℝ) 
  (h_Q : Q = (M + N) / 2) : 
  2 * dist A₁ Q = dist M N :=
sorry

end

end NUMINAMATH_CALUDE_ellipse_midpoint_property_l2767_276741


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2767_276786

theorem polynomial_factorization (x y : ℝ) :
  -6 * x^2 * y + 12 * x * y^2 - 3 * x * y = -3 * x * y * (2 * x - 4 * y + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2767_276786


namespace NUMINAMATH_CALUDE_cuboid_sphere_surface_area_l2767_276758

-- Define the cuboid
structure Cuboid where
  face_area1 : ℝ
  face_area2 : ℝ
  face_area3 : ℝ
  vertices_on_sphere : Bool

-- Define the theorem
theorem cuboid_sphere_surface_area 
  (c : Cuboid) 
  (h1 : c.face_area1 = 12) 
  (h2 : c.face_area2 = 15) 
  (h3 : c.face_area3 = 20) 
  (h4 : c.vertices_on_sphere = true) : 
  ∃ (sphere_surface_area : ℝ), sphere_surface_area = 50 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_cuboid_sphere_surface_area_l2767_276758


namespace NUMINAMATH_CALUDE_decrease_xyz_squared_l2767_276795

theorem decrease_xyz_squared (x y z : ℝ) :
  let x' := 0.6 * x
  let y' := 0.6 * y
  let z' := 0.6 * z
  x' * y' * z' ^ 2 = 0.1296 * x * y * z ^ 2 := by
sorry

end NUMINAMATH_CALUDE_decrease_xyz_squared_l2767_276795


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2767_276744

theorem arithmetic_sequence_problem (a d : ℤ) :
  let seq := [a, a + d, a + 2*d, a + 3*d, a + 4*d]
  (a^3 + (a + d)^3 + (a + 2*d)^3 + (a + 3*d)^3 = 16 * (a + (a + d) + (a + 2*d) + (a + 3*d))^2) ∧
  ((a + d)^3 + (a + 2*d)^3 + (a + 3*d)^3 + (a + 4*d)^3 = 16 * ((a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d))^2) →
  seq = [0, 16, 32, 48, 64] :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2767_276744


namespace NUMINAMATH_CALUDE_absolute_value_of_two_is_not_negative_two_l2767_276700

theorem absolute_value_of_two_is_not_negative_two : ¬(|2| = -2) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_two_is_not_negative_two_l2767_276700


namespace NUMINAMATH_CALUDE_product_statistics_l2767_276762

def product_ratings : List ℝ := [9.6, 10.1, 9.7, 9.8, 10.0, 9.7, 10.0, 9.8, 10.1, 10.2]

def sum_of_squares : ℝ := 98.048

def improvement : ℝ := 0.2

def is_first_class (rating : ℝ) : Prop := rating ≥ 10

theorem product_statistics :
  let n : ℕ := product_ratings.length
  let mean : ℝ := (product_ratings.sum) / n
  let variance : ℝ := sum_of_squares / n - mean ^ 2
  let new_mean : ℝ := mean + improvement
  let new_variance : ℝ := variance
  (mean = 9.9) ∧
  (variance = 0.038) ∧
  (new_mean = 10.1) ∧
  (new_variance = 0.038) :=
sorry

end NUMINAMATH_CALUDE_product_statistics_l2767_276762


namespace NUMINAMATH_CALUDE_sector_area_l2767_276735

/-- Given a circular sector with perimeter 10 and central angle 3 radians, its area is 6 -/
theorem sector_area (r : ℝ) (perimeter : ℝ) (central_angle : ℝ) : 
  perimeter = 10 → central_angle = 3 → perimeter = 2 * r + central_angle * r → 
  (1/2) * r^2 * central_angle = 6 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2767_276735


namespace NUMINAMATH_CALUDE_probability_correct_l2767_276716

-- Define the number of red and blue marbles
def red_marbles : ℕ := 15
def blue_marbles : ℕ := 9

-- Define the total number of marbles
def total_marbles : ℕ := red_marbles + blue_marbles

-- Define the number of marbles to be selected
def selected_marbles : ℕ := 4

-- Define the probability of selecting 2 red and 2 blue marbles
def probability_two_red_two_blue : ℚ := 4 / 27

-- Theorem statement
theorem probability_correct : 
  (Nat.choose red_marbles 2 * Nat.choose blue_marbles 2) / Nat.choose total_marbles selected_marbles = probability_two_red_two_blue := by
  sorry

end NUMINAMATH_CALUDE_probability_correct_l2767_276716


namespace NUMINAMATH_CALUDE_lower_interest_rate_l2767_276799

/-- Calculates simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem lower_interest_rate 
  (principal : ℝ) 
  (high_rate low_rate : ℝ) 
  (time : ℝ) 
  (interest_difference : ℝ) :
  principal = 12000 →
  high_rate = 0.15 →
  time = 2 →
  interest_difference = 720 →
  simple_interest principal high_rate time - simple_interest principal low_rate time = interest_difference →
  low_rate = 0.12 := by
sorry

end NUMINAMATH_CALUDE_lower_interest_rate_l2767_276799


namespace NUMINAMATH_CALUDE_chicken_selling_price_l2767_276788

/-- Represents the problem of determining the selling price of chickens --/
theorem chicken_selling_price 
  (num_chickens : ℕ) 
  (profit : ℚ) 
  (feed_per_chicken : ℚ) 
  (feed_bag_weight : ℚ) 
  (feed_bag_cost : ℚ) :
  num_chickens = 50 →
  profit = 65 →
  feed_per_chicken = 2 →
  feed_bag_weight = 20 →
  feed_bag_cost = 2 →
  ∃ (selling_price : ℚ), selling_price = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_chicken_selling_price_l2767_276788


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l2767_276757

theorem ratio_x_to_y (x y : ℝ) (h : (15 * x - 4 * y) / (18 * x - 3 * y) = 4 / 7) :
  x / y = 16 / 33 := by
sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l2767_276757


namespace NUMINAMATH_CALUDE_gcd_consecutive_fib_46368_75025_l2767_276787

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- Theorem: The GCD of two consecutive Fibonacci numbers 46368 and 75025 is 1 -/
theorem gcd_consecutive_fib_46368_75025 :
  ∃ n : ℕ, fib n = 46368 ∧ fib (n + 1) = 75025 ∧ Nat.gcd 46368 75025 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_consecutive_fib_46368_75025_l2767_276787


namespace NUMINAMATH_CALUDE_pine_cones_on_roof_l2767_276718

/-- Calculates the weight of pine cones on a roof given the number of trees, 
    pine cones per tree, percentage on roof, and weight per pine cone. -/
theorem pine_cones_on_roof 
  (num_trees : ℕ) 
  (cones_per_tree : ℕ) 
  (percent_on_roof : ℚ) 
  (weight_per_cone : ℕ) 
  (h1 : num_trees = 8)
  (h2 : cones_per_tree = 200)
  (h3 : percent_on_roof = 30 / 100)
  (h4 : weight_per_cone = 4) :
  (num_trees * cones_per_tree : ℚ) * percent_on_roof * weight_per_cone = 1920 :=
sorry

end NUMINAMATH_CALUDE_pine_cones_on_roof_l2767_276718


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2767_276779

theorem purely_imaginary_complex_number (m : ℝ) :
  let z : ℂ := Complex.mk (m^2 - 2*m - 3) (m^2 - 1)
  (z.re = 0 ∧ z.im ≠ 0) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2767_276779


namespace NUMINAMATH_CALUDE_simple_interest_rate_percent_l2767_276766

/-- Given an amount of simple interest, time period, and rate, prove that the rate percent is correct. -/
theorem simple_interest_rate_percent 
  (principal : ℝ) 
  (interest : ℝ) 
  (time : ℝ) 
  (rate : ℝ) 
  (h1 : interest = 400) 
  (h2 : time = 4) 
  (h3 : rate = 0.1) : 
  rate * 100 = 10 := by
sorry


end NUMINAMATH_CALUDE_simple_interest_rate_percent_l2767_276766


namespace NUMINAMATH_CALUDE_collective_purchase_equation_l2767_276761

theorem collective_purchase_equation (x y : ℤ) : 
  (8 * x - 3 = y) → (7 * x + 4 = y) := by
  sorry

end NUMINAMATH_CALUDE_collective_purchase_equation_l2767_276761


namespace NUMINAMATH_CALUDE_add_1850_minutes_to_3_15pm_l2767_276768

/-- Represents a time of day in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  hLt24 : hours < 24
  mLt60 : minutes < 60

/-- Adds minutes to a given time and wraps around to the next day if necessary -/
def addMinutes (t : Time) (m : Nat) : Time :=
  sorry

/-- Converts a number of minutes to days, hours, and minutes -/
def minutesToDHM (m : Nat) : (Nat × Nat × Nat) :=
  sorry

theorem add_1850_minutes_to_3_15pm (start : Time) (h : start.hours = 15 ∧ start.minutes = 15) :
  let end_time := addMinutes start 1850
  end_time.hours = 22 ∧ end_time.minutes = 5 ∧ (minutesToDHM 1850).1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_add_1850_minutes_to_3_15pm_l2767_276768


namespace NUMINAMATH_CALUDE_max_value_x2_plus_y2_l2767_276720

theorem max_value_x2_plus_y2 (x y : ℝ) (h : 3 * x^2 + 2 * y^2 = 2 * x) :
  ∃ (M : ℝ), M = 4/9 ∧ x^2 + y^2 ≤ M ∧ ∃ (x₀ y₀ : ℝ), 3 * x₀^2 + 2 * y₀^2 = 2 * x₀ ∧ x₀^2 + y₀^2 = M :=
sorry

end NUMINAMATH_CALUDE_max_value_x2_plus_y2_l2767_276720


namespace NUMINAMATH_CALUDE_derivative_zero_at_origin_l2767_276724

theorem derivative_zero_at_origin (f : ℝ → ℝ) (h1 : Differentiable ℝ f) 
  (h2 : ∀ x, f (-x) = f x) : 
  deriv f 0 = 0 := by
sorry

end NUMINAMATH_CALUDE_derivative_zero_at_origin_l2767_276724


namespace NUMINAMATH_CALUDE_sum_of_odd_periodic_function_l2767_276767

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period_4 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 4) = f x

theorem sum_of_odd_periodic_function 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_periodic : has_period_4 f) 
  (h_f1 : f 1 = -1) : 
  f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 = -1 :=
sorry

end NUMINAMATH_CALUDE_sum_of_odd_periodic_function_l2767_276767


namespace NUMINAMATH_CALUDE_probability_factor_less_than_10_of_90_l2767_276775

def factors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ x => n % x = 0) (Finset.range (n + 1))

theorem probability_factor_less_than_10_of_90 :
  let all_factors := factors 90
  let factors_less_than_10 := all_factors.filter (λ x => x < 10)
  (factors_less_than_10.card : ℚ) / all_factors.card = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_factor_less_than_10_of_90_l2767_276775


namespace NUMINAMATH_CALUDE_factorial_ratio_equals_twelve_l2767_276759

theorem factorial_ratio_equals_twelve : (Nat.factorial 10 * Nat.factorial 4 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 5) = 12 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_equals_twelve_l2767_276759


namespace NUMINAMATH_CALUDE_max_faces_limited_neighbor_tri_neighbor_is_tetrahedron_l2767_276764

/-- A convex polyhedron. -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  euler_formula : vertices - edges + faces = 2
  edge_face_relation : edges = 2 * faces

/-- A convex polyhedron where each face has at most 4 neighboring faces. -/
structure LimitedNeighborPolyhedron extends ConvexPolyhedron where
  max_neighbors : edges ≤ 2 * faces

/-- A convex polyhedron where each face has exactly 3 neighboring faces. -/
structure TriNeighborPolyhedron extends ConvexPolyhedron where
  tri_neighbors : edges = 3 * faces / 2

/-- Theorem: The maximum number of faces in a LimitedNeighborPolyhedron is 6. -/
theorem max_faces_limited_neighbor (P : LimitedNeighborPolyhedron) : P.faces ≤ 6 := by
  sorry

/-- Theorem: A TriNeighborPolyhedron must be a tetrahedron (4 faces). -/
theorem tri_neighbor_is_tetrahedron (P : TriNeighborPolyhedron) : P.faces = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_faces_limited_neighbor_tri_neighbor_is_tetrahedron_l2767_276764


namespace NUMINAMATH_CALUDE_tangent_line_slope_intersection_line_equation_l2767_276706

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

-- Define point P
def P : ℝ × ℝ := (1, 2)

-- Define point Q
def Q : ℝ × ℝ := (0, -2)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Theorem for part 1
theorem tangent_line_slope :
  ∃ m : ℝ, m = -3/4 ∧
  (∀ x y : ℝ, y - P.2 = m * (x - P.1) → 
   (∃ t : ℝ, x = t ∧ y = t ∧ circle_C x y)) ∧
  (∀ x y : ℝ, circle_C x y → (y - P.2 ≠ m * (x - P.1) ∨ (x = P.1 ∧ y = P.2))) :=
sorry

-- Theorem for part 2
theorem intersection_line_equation :
  ∃ k : ℝ, (k = 5/3 ∨ k = 1) ∧
  (∀ x y : ℝ, y = k*x - 2 →
   (∃ A B : ℝ × ℝ,
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    y = k*x - 2 ∧
    (A.2 / A.1) * (B.2 / B.1) = -1/7)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_intersection_line_equation_l2767_276706


namespace NUMINAMATH_CALUDE_village_population_calculation_l2767_276785

def initial_population : ℕ := 3161
def death_rate : ℚ := 5 / 100
def leaving_rate : ℚ := 15 / 100

theorem village_population_calculation :
  let remaining_after_deaths := initial_population - Int.floor (↑initial_population * death_rate)
  let final_population := remaining_after_deaths - Int.floor (↑remaining_after_deaths * leaving_rate)
  final_population = 2553 := by
  sorry

end NUMINAMATH_CALUDE_village_population_calculation_l2767_276785


namespace NUMINAMATH_CALUDE_amit_left_after_three_days_l2767_276752

/-- The number of days Amit takes to complete the work alone -/
def amit_days : ℕ := 15

/-- The number of days Ananthu takes to complete the work alone -/
def ananthu_days : ℕ := 45

/-- The total number of days taken to complete the work -/
def total_days : ℕ := 39

/-- The number of days Amit worked before leaving -/
def amit_worked_days : ℕ := 3

theorem amit_left_after_three_days :
  ∃ (w : ℝ), w > 0 ∧
  amit_worked_days * (w / amit_days) + (total_days - amit_worked_days) * (w / ananthu_days) = w :=
sorry

end NUMINAMATH_CALUDE_amit_left_after_three_days_l2767_276752


namespace NUMINAMATH_CALUDE_tom_spent_seven_tickets_on_hat_l2767_276715

/-- The number of tickets Tom spent on the hat -/
def tickets_spent_on_hat (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (tickets_left : ℕ) : ℕ :=
  whack_a_mole_tickets + skee_ball_tickets - tickets_left

/-- Theorem stating that Tom spent 7 tickets on the hat -/
theorem tom_spent_seven_tickets_on_hat :
  tickets_spent_on_hat 32 25 50 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tom_spent_seven_tickets_on_hat_l2767_276715


namespace NUMINAMATH_CALUDE_base_85_congruence_l2767_276727

theorem base_85_congruence (b : ℤ) (h1 : 0 ≤ b) (h2 : b ≤ 20) 
  (h3 : (74639281 : ℤ) - b ≡ 0 [ZMOD 17]) : b = 1 := by
  sorry

end NUMINAMATH_CALUDE_base_85_congruence_l2767_276727


namespace NUMINAMATH_CALUDE_other_diagonal_length_l2767_276798

/-- Represents a rhombus with given properties -/
structure Rhombus where
  d1 : ℝ  -- Length of one diagonal
  d2 : ℝ  -- Length of the other diagonal
  area : ℝ -- Area of the rhombus

/-- The area of a rhombus is half the product of its diagonals -/
axiom rhombus_area (r : Rhombus) : r.area = (r.d1 * r.d2) / 2

/-- Given a rhombus with one diagonal of 16 cm and an area of 88 cm², 
    the length of the other diagonal is 11 cm -/
theorem other_diagonal_length (r : Rhombus) 
    (h1 : r.d2 = 16) 
    (h2 : r.area = 88) : 
    r.d1 = 11 := by
  sorry


end NUMINAMATH_CALUDE_other_diagonal_length_l2767_276798


namespace NUMINAMATH_CALUDE_mothers_daughters_ages_l2767_276710

theorem mothers_daughters_ages (mother_age daughter_age : ℕ) : 
  mother_age = 40 →
  daughter_age + 2 * mother_age = 95 →
  mother_age + 2 * daughter_age = 70 := by
  sorry

end NUMINAMATH_CALUDE_mothers_daughters_ages_l2767_276710


namespace NUMINAMATH_CALUDE_log_equation_holds_l2767_276751

theorem log_equation_holds (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 4) * (Real.log 7 / Real.log x) = Real.log 7 / Real.log 4 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_holds_l2767_276751


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l2767_276731

/-- Given an isosceles right triangle with perimeter 2p, its area is (3-2√2)p² -/
theorem isosceles_right_triangle_area (p : ℝ) (h : p > 0) : 
  ∃ (x : ℝ), 
    x > 0 ∧ 
    (2 * x + x * Real.sqrt 2 = 2 * p) ∧ 
    ((1 / 2) * x * x = (3 - 2 * Real.sqrt 2) * p^2) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l2767_276731


namespace NUMINAMATH_CALUDE_repeating_decimal_length_seven_twelfths_l2767_276740

theorem repeating_decimal_length_seven_twelfths :
  ∃ (d : ℕ) (n : ℕ), 
    7 * (10^n) ≡ d [MOD 12] ∧ 
    7 * (10^(n+1)) ≡ d [MOD 12] ∧ 
    0 < d ∧ d < 12 ∧
    n = 1 :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_length_seven_twelfths_l2767_276740


namespace NUMINAMATH_CALUDE_ios_department_larger_l2767_276796

/-- Represents the number of developers in the Android department -/
def android_devs : ℕ := sorry

/-- Represents the number of developers in the iOS department -/
def ios_devs : ℕ := sorry

/-- The total number of messages sent equals the total number of messages received -/
axiom message_balance : 7 * android_devs + 15 * ios_devs = 15 * android_devs + 9 * ios_devs

theorem ios_department_larger : ios_devs > android_devs := by
  sorry

end NUMINAMATH_CALUDE_ios_department_larger_l2767_276796


namespace NUMINAMATH_CALUDE_guitar_purchase_savings_l2767_276792

/-- Proves that the difference in cost between Guitar Center and Sweetwater is $50 --/
theorem guitar_purchase_savings (retail_price : ℝ) 
  (gc_discount_rate : ℝ) (gc_shipping_fee : ℝ) 
  (sw_discount_rate : ℝ) :
  retail_price = 1000 →
  gc_discount_rate = 0.15 →
  gc_shipping_fee = 100 →
  sw_discount_rate = 0.10 →
  (retail_price * (1 - gc_discount_rate) + gc_shipping_fee) -
  (retail_price * (1 - sw_discount_rate)) = 50 := by
sorry

end NUMINAMATH_CALUDE_guitar_purchase_savings_l2767_276792


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l2767_276760

theorem quadratic_roots_properties (x₁ x₂ : ℝ) 
  (h₁ : x₁^2 - 5*x₁ - 3 = 0) 
  (h₂ : x₂^2 - 5*x₂ - 3 = 0) :
  (x₁^2 + x₂^2 = 31) ∧ (1/x₁ - 1/x₂ = Real.sqrt 37 / 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l2767_276760


namespace NUMINAMATH_CALUDE_probability_at_least_one_cherry_plum_probability_at_least_one_cherry_plum_proof_l2767_276747

/-- The probability of selecting at least one cherry plum cutting -/
theorem probability_at_least_one_cherry_plum 
  (total_cuttings : ℕ) 
  (cherry_plum_cuttings : ℕ) 
  (plum_cuttings : ℕ) 
  (selected_cuttings : ℕ)
  (h1 : total_cuttings = 20)
  (h2 : cherry_plum_cuttings = 8)
  (h3 : plum_cuttings = 12)
  (h4 : selected_cuttings = 3)
  (h5 : total_cuttings = cherry_plum_cuttings + plum_cuttings) : 
  ℚ :=
46/57

theorem probability_at_least_one_cherry_plum_proof 
  (total_cuttings : ℕ) 
  (cherry_plum_cuttings : ℕ) 
  (plum_cuttings : ℕ) 
  (selected_cuttings : ℕ)
  (h1 : total_cuttings = 20)
  (h2 : cherry_plum_cuttings = 8)
  (h3 : plum_cuttings = 12)
  (h4 : selected_cuttings = 3)
  (h5 : total_cuttings = cherry_plum_cuttings + plum_cuttings) : 
  probability_at_least_one_cherry_plum total_cuttings cherry_plum_cuttings plum_cuttings selected_cuttings h1 h2 h3 h4 h5 = 46/57 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_cherry_plum_probability_at_least_one_cherry_plum_proof_l2767_276747


namespace NUMINAMATH_CALUDE_calculate_interest_rate_l2767_276781

/-- Given a total sum and two parts with specific interest conditions, 
    calculate the interest rate for the second part. -/
theorem calculate_interest_rate 
  (total_sum : ℚ) 
  (second_part : ℚ) 
  (first_part_years : ℚ) 
  (first_part_rate : ℚ) 
  (second_part_years : ℚ) 
  (h1 : total_sum = 2730) 
  (h2 : second_part = 1680) 
  (h3 : first_part_years = 8) 
  (h4 : first_part_rate = 3 / 100) 
  (h5 : second_part_years = 3) 
  (h6 : (total_sum - second_part) * first_part_rate * first_part_years = 
        second_part * (second_part_years * x) ) :
  x = 5 / 100 := by
  sorry

end NUMINAMATH_CALUDE_calculate_interest_rate_l2767_276781


namespace NUMINAMATH_CALUDE_stadium_seats_problem_l2767_276750

/-- Represents the number of seats in the n-th row -/
def a (n : ℕ) : ℕ := n + 1

/-- The total number of seats in the first n rows -/
def total_seats (n : ℕ) : ℕ := n * (n + 3) / 2

/-- The sum of the first n terms of the sequence a_n / (n(n+1)^2) -/
def S (n : ℕ) : ℚ := n / (n + 1)

theorem stadium_seats_problem :
  (total_seats 20 = 230) ∧ (S 20 = 20 / 21) := by
  sorry

end NUMINAMATH_CALUDE_stadium_seats_problem_l2767_276750


namespace NUMINAMATH_CALUDE_elevator_problem_l2767_276739

theorem elevator_problem :
  let num_elevators : ℕ := 4
  let num_people : ℕ := 3
  let num_same_elevator : ℕ := 2
  (Nat.choose num_people num_same_elevator) * (num_elevators * (num_elevators - 1)) = 36
  := by sorry

end NUMINAMATH_CALUDE_elevator_problem_l2767_276739


namespace NUMINAMATH_CALUDE_equation_solutions_l2767_276784

theorem equation_solutions :
  (∃ x : ℚ, (x + 1) / 2 - (x - 3) / 6 = (5 * x + 1) / 3 + 1 ∧ x = -1/4) ∧
  (∃ x : ℚ, (x - 4) / (1/5) - 1 = (x - 3) / (1/2) ∧ x = 5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2767_276784


namespace NUMINAMATH_CALUDE_crease_length_in_folded_rectangle_l2767_276769

/-- Represents a folded rectangle with given dimensions and fold properties -/
structure FoldedRectangle where
  width : ℝ
  fold_distance : ℝ
  crease_length : ℝ
  fold_angle : ℝ

/-- Theorem stating the crease length in a specific folded rectangle configuration -/
theorem crease_length_in_folded_rectangle (r : FoldedRectangle) 
  (h1 : r.width = 8)
  (h2 : r.fold_distance = 2)
  (h3 : Real.tan r.fold_angle = 3) : 
  r.crease_length = 2/3 := by
  sorry

#check crease_length_in_folded_rectangle

end NUMINAMATH_CALUDE_crease_length_in_folded_rectangle_l2767_276769


namespace NUMINAMATH_CALUDE_tetrahedron_edge_relation_l2767_276772

/-- Given a tetrahedron ABCD with edge lengths and angles, prove that among t₁, t₂, t₃,
    there is at least one number equal to the sum of the other two. -/
theorem tetrahedron_edge_relation (a₁ a₂ a₃ b₁ b₂ b₃ θ₁ θ₂ θ₃ : ℝ) 
  (h_pos : a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 ∧ b₁ > 0 ∧ b₂ > 0 ∧ b₃ > 0)
  (h_angle : 0 < θ₁ ∧ θ₁ < π ∧ 0 < θ₂ ∧ θ₂ < π ∧ 0 < θ₃ ∧ θ₃ < π) :
  ∃ (i j k : Fin 3), i ≠ j ∧ i ≠ k ∧ j ≠ k ∧
    let t : Fin 3 → ℝ := λ n => match n with
      | 0 => a₁ * b₁ * Real.cos θ₁
      | 1 => a₂ * b₂ * Real.cos θ₂
      | 2 => a₃ * b₃ * Real.cos θ₃
    t i = t j + t k :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_edge_relation_l2767_276772


namespace NUMINAMATH_CALUDE_tshirts_bought_l2767_276719

/-- Given the price conditions for pants and t-shirts, 
    prove the number of t-shirts that can be bought with 800 Rs. -/
theorem tshirts_bought (pants_price t_shirt_price : ℕ) : 
  (3 * pants_price + 6 * t_shirt_price = 1500) →
  (pants_price + 12 * t_shirt_price = 1500) →
  (800 / t_shirt_price = 8) := by
  sorry

end NUMINAMATH_CALUDE_tshirts_bought_l2767_276719


namespace NUMINAMATH_CALUDE_expression_approximation_l2767_276730

theorem expression_approximation : 
  let x := Real.sqrt 1.1 / Real.sqrt 0.81 + Real.sqrt 1.44 / Real.sqrt 0.49
  ∃ ε > 0, ε < 0.00005 ∧ |x - 2.8793| < ε :=
by sorry

end NUMINAMATH_CALUDE_expression_approximation_l2767_276730
