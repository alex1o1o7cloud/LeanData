import Mathlib

namespace NUMINAMATH_CALUDE_line_at_distance_iff_tangent_to_cylinder_l4082_408242

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by two points -/
structure Line3D where
  a : Point3D
  b : Point3D

/-- A cylinder in 3D space defined by its axis (a line) and radius -/
structure Cylinder where
  axis : Line3D
  radius : ℝ

/-- Distance between a point and a line in 3D space -/
def distance_point_to_line (p : Point3D) (l : Line3D) : ℝ := sorry

/-- Check if a line is tangent to a cylinder -/
def is_tangent_to_cylinder (l : Line3D) (c : Cylinder) : Prop := sorry

/-- Check if a line passes through a point -/
def line_passes_through_point (l : Line3D) (p : Point3D) : Prop := sorry

/-- Main theorem: A line passing through M is at distance d from AB iff it's tangent to the cylinder -/
theorem line_at_distance_iff_tangent_to_cylinder 
  (M : Point3D) (AB : Line3D) (d : ℝ) (l : Line3D) : 
  (line_passes_through_point l M ∧ distance_point_to_line M AB = d) ↔ 
  is_tangent_to_cylinder l (Cylinder.mk AB d) :=
sorry

end NUMINAMATH_CALUDE_line_at_distance_iff_tangent_to_cylinder_l4082_408242


namespace NUMINAMATH_CALUDE_fixed_point_on_curve_circle_center_on_line_l4082_408265

-- Define the curve C
def C (a x y : ℝ) : Prop := x^2 + y^2 - 4*a*x + 2*a*y - 20 + 20*a = 0

-- Theorem 1: The point (4, -2) always lies on C for any value of a
theorem fixed_point_on_curve (a : ℝ) : C a 4 (-2) := by sorry

-- Theorem 2: When a ≠ 2, C is a circle and its center lies on the line x + 2y = 0
theorem circle_center_on_line (a : ℝ) (h : a ≠ 2) :
  ∃ (x y : ℝ), C a x y ∧ (∀ (x' y' : ℝ), C a x' y' → (x' - x)^2 + (y' - y)^2 = (x - x')^2 + (y - y')^2) ∧ x + 2*y = 0 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_curve_circle_center_on_line_l4082_408265


namespace NUMINAMATH_CALUDE_isosceles_from_cosine_relation_l4082_408284

/-- A triangle with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  angle_bounds : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π

/-- A triangle is isosceles if it has at least two equal sides -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

/-- The main theorem: if a = 2b cos C, then the triangle is isosceles -/
theorem isosceles_from_cosine_relation (t : Triangle) (h : t.a = 2 * t.b * Real.cos t.C) :
  t.isIsosceles := by
  sorry

end NUMINAMATH_CALUDE_isosceles_from_cosine_relation_l4082_408284


namespace NUMINAMATH_CALUDE_c_range_l4082_408264

def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem c_range (a b c : ℝ) :
  (0 < f a b c (-1) ∧ f a b c (-1) = f a b c (-2) ∧ f a b c (-2) = f a b c (-3) ∧ f a b c (-3) ≤ 3) →
  (6 < c ∧ c ≤ 9) := by
  sorry

end NUMINAMATH_CALUDE_c_range_l4082_408264


namespace NUMINAMATH_CALUDE_unique_pair_for_squared_difference_l4082_408259

theorem unique_pair_for_squared_difference : 
  ∃! (a b : ℕ), a^2 - b^2 = 25 ∧ a = 13 ∧ b = 12 :=
by sorry

end NUMINAMATH_CALUDE_unique_pair_for_squared_difference_l4082_408259


namespace NUMINAMATH_CALUDE_distance_to_symmetry_axis_range_l4082_408269

variable (a b c : ℝ)
variable (f : ℝ → ℝ)

theorem distance_to_symmetry_axis_range 
  (ha : a > 0)
  (hf : f = fun x ↦ a * x^2 + b * x + c)
  (htangent : ∀ x₀, 0 ≤ (2 * a * x₀ + b) ∧ (2 * a * x₀ + b) ≤ 1) :
  ∃ d : Set ℝ, d = {x | 0 ≤ x ∧ x ≤ 1 / (2 * a)} ∧
    ∀ x₀, Set.Mem (|x₀ + b / (2 * a)|) d :=
by sorry

end NUMINAMATH_CALUDE_distance_to_symmetry_axis_range_l4082_408269


namespace NUMINAMATH_CALUDE_count_integer_lengths_specific_triangle_l4082_408292

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  a : ℕ  -- length of first leg
  b : ℕ  -- length of second leg
  c : ℕ  -- length of hypotenuse
  right_angle : c^2 = a^2 + b^2  -- Pythagorean theorem

/-- Counts the number of distinct integer lengths of line segments
    that can be drawn from a vertex to the opposite side -/
def count_integer_lengths (t : RightTriangle) : ℕ :=
  -- Implementation details omitted
  sorry

/-- The main theorem -/
theorem count_integer_lengths_specific_triangle :
  ∃ t : RightTriangle, t.a = 15 ∧ t.b = 20 ∧ count_integer_lengths t = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_count_integer_lengths_specific_triangle_l4082_408292


namespace NUMINAMATH_CALUDE_hyperbola_n_range_l4082_408246

/-- Represents a hyperbola with parameters m and n -/
structure Hyperbola (m n : ℝ) where
  equation : ∀ x y : ℝ, x^2 / (m^2 + n) - y^2 / (3 * m^2 - n) = 1

/-- The distance between the foci of a hyperbola -/
def focal_distance (h : Hyperbola m n) : ℝ := 4

/-- Theorem stating the range of n for a hyperbola with given properties -/
theorem hyperbola_n_range (m n : ℝ) (h : Hyperbola m n) :
  focal_distance h = 4 → -1 < n ∧ n < 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_n_range_l4082_408246


namespace NUMINAMATH_CALUDE_max_sections_five_lines_l4082_408275

/-- The number of sections created by n line segments in a rectangle -/
def sections (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else sections (n - 1) + (n - 1)

/-- The maximum number of sections created by 5 line segments in a rectangle -/
theorem max_sections_five_lines :
  sections 5 = 12 :=
sorry

end NUMINAMATH_CALUDE_max_sections_five_lines_l4082_408275


namespace NUMINAMATH_CALUDE_perpendicular_bisector_b_l4082_408299

/-- The value of b for which the line x + y = b is the perpendicular bisector
    of the line segment connecting (1,4) and (7,10) -/
theorem perpendicular_bisector_b : ∃ b : ℝ,
  (∀ x y : ℝ, x + y = b ↔ 
    ((x - 4)^2 + (y - 7)^2 = 9) ∧ 
    ((x - 1) * (7 - 1) + (y - 4) * (10 - 4) = 0)) ∧
  b = 11 := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_bisector_b_l4082_408299


namespace NUMINAMATH_CALUDE_clay_molding_minimum_operations_l4082_408210

/-- Represents a clay molding operation -/
structure ClayOperation where
  groups : List (List Nat)
  deriving Repr

/-- The result of applying a clay molding operation -/
def applyOperation (pieces : List Nat) (op : ClayOperation) : List Nat :=
  sorry

/-- Checks if all elements in a list are distinct -/
def allDistinct (l : List Nat) : Prop :=
  sorry

/-- The main theorem stating that 2 operations are sufficient and minimal -/
theorem clay_molding_minimum_operations :
  ∃ (op1 op2 : ClayOperation),
    let initial_pieces := List.replicate 111 1
    let after_op1 := applyOperation initial_pieces op1
    let final_pieces := applyOperation after_op1 op2
    (final_pieces.length = 11) ∧
    (allDistinct final_pieces) ∧
    (∀ (op1' op2' : ClayOperation),
      let after_op1' := applyOperation initial_pieces op1'
      let final_pieces' := applyOperation after_op1' op2'
      (final_pieces'.length = 11 ∧ allDistinct final_pieces') →
      ¬∃ (single_op : ClayOperation),
        let result := applyOperation initial_pieces single_op
        (result.length = 11 ∧ allDistinct result)) :=
  sorry

end NUMINAMATH_CALUDE_clay_molding_minimum_operations_l4082_408210


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l4082_408260

def containsOnly3And4 (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 3 ∨ d = 4

def contains3And4 (n : ℕ) : Prop :=
  3 ∈ n.digits 10 ∧ 4 ∈ n.digits 10

def isMultipleOf3And4 (n : ℕ) : Prop :=
  n % 3 = 0 ∧ n % 4 = 0

theorem smallest_number_with_conditions :
  ∀ n : ℕ, 
    containsOnly3And4 n ∧ 
    contains3And4 n ∧ 
    isMultipleOf3And4 n →
    n ≥ 3444 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l4082_408260


namespace NUMINAMATH_CALUDE_odd_square_minus_one_div_eight_l4082_408285

theorem odd_square_minus_one_div_eight (a : ℕ) (h1 : a > 0) (h2 : Odd a) :
  ∃ k : ℤ, a^2 - 1 = 8 * k :=
sorry

end NUMINAMATH_CALUDE_odd_square_minus_one_div_eight_l4082_408285


namespace NUMINAMATH_CALUDE_total_carrots_grown_l4082_408226

theorem total_carrots_grown (sandy_carrots sam_carrots : ℕ) 
  (h1 : sandy_carrots = 6) 
  (h2 : sam_carrots = 3) : 
  sandy_carrots + sam_carrots = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_carrots_grown_l4082_408226


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l4082_408250

/-- Given two points on a quadratic function, prove the value of b -/
theorem quadratic_coefficient (a c y₁ y₂ : ℝ) :
  y₁ = a * 2^2 + b * 2 + c →
  y₂ = a * (-2)^2 + b * (-2) + c →
  y₁ - y₂ = -12 →
  b = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l4082_408250


namespace NUMINAMATH_CALUDE_celeste_candy_theorem_l4082_408224

/-- Represents the state of candies on the table -/
structure CandyState (n : ℕ+) where
  counts : Fin n → ℕ

/-- Represents the operations that can be performed on the candy state -/
inductive Operation (n : ℕ+)
  | split : Fin n → Operation n
  | take : Fin n → Operation n

/-- Applies an operation to a candy state -/
def apply_operation {n : ℕ+} (state : CandyState n) (op : Operation n) : CandyState n :=
  sorry

/-- Checks if a candy state is empty -/
def is_empty {n : ℕ+} (state : CandyState n) : Prop :=
  ∀ i, state.counts i = 0

/-- Main theorem: Celeste can empty the table for any initial configuration
    if and only if n is not divisible by 3 -/
theorem celeste_candy_theorem (n : ℕ+) :
  (∀ (m : ℕ+) (initial_state : CandyState n),
    ∃ (ops : List (Operation n)), is_empty (ops.foldl apply_operation initial_state))
  ↔ ¬(n : ℕ) % 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_celeste_candy_theorem_l4082_408224


namespace NUMINAMATH_CALUDE_f_value_at_5pi_3_l4082_408243

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  is_periodic f p ∧ ∀ q, 0 < q ∧ q < p → ¬ is_periodic f q

theorem f_value_at_5pi_3 (f : ℝ → ℝ) (h1 : is_even f)
  (h2 : smallest_positive_period f π)
  (h3 : ∀ x ∈ Set.Icc 0 (π/2), f x = Real.cos x) :
  f (5*π/3) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_5pi_3_l4082_408243


namespace NUMINAMATH_CALUDE_power_calculation_l4082_408262

theorem power_calculation : 16^4 * 8^2 / 4^12 = (1 : ℚ) / 4 := by sorry

end NUMINAMATH_CALUDE_power_calculation_l4082_408262


namespace NUMINAMATH_CALUDE_cubic_expansion_result_l4082_408295

theorem cubic_expansion_result (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (Real.sqrt 3 * x - Real.sqrt 2)^3 = a₀ * x^3 + a₁ * x^2 + a₂ * x + a₃) →
  (a₀ + a₂)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expansion_result_l4082_408295


namespace NUMINAMATH_CALUDE_hockey_league_games_l4082_408223

/-- The total number of games played in a hockey league season -/
def total_games (n : ℕ) (g : ℕ) : ℕ :=
  n * (n - 1) * g / 2

/-- Theorem: In a league with 12 teams, where each team plays 4 games with each other team,
    the total number of games played is 264 -/
theorem hockey_league_games :
  total_games 12 4 = 264 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_games_l4082_408223


namespace NUMINAMATH_CALUDE_other_sales_percentage_l4082_408241

/-- The Paper Boutique's sales percentages -/
structure SalesPercentages where
  pens : ℝ
  pencils : ℝ
  notebooks : ℝ
  total : ℝ
  pens_percent : pens = 25
  pencils_percent : pencils = 30
  notebooks_percent : notebooks = 20
  total_sum : total = 100

/-- Theorem: The percentage of sales that are neither pens, pencils, nor notebooks is 25% -/
theorem other_sales_percentage (s : SalesPercentages) : 
  s.total - (s.pens + s.pencils + s.notebooks) = 25 := by
  sorry

end NUMINAMATH_CALUDE_other_sales_percentage_l4082_408241


namespace NUMINAMATH_CALUDE_fifth_element_row_20_l4082_408255

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The fifth element in a row of Pascal's triangle -/
def fifthElement (row : ℕ) : ℕ := binomial row 4

theorem fifth_element_row_20 : fifthElement 20 = 4845 := by
  sorry

end NUMINAMATH_CALUDE_fifth_element_row_20_l4082_408255


namespace NUMINAMATH_CALUDE_min_photos_theorem_l4082_408257

theorem min_photos_theorem (n_girls n_boys : ℕ) (h_girls : n_girls = 4) (h_boys : n_boys = 8) :
  ∃ (min_photos : ℕ), min_photos = n_girls * n_boys + 1 ∧
  (∀ (num_photos : ℕ), num_photos ≥ min_photos →
    (∃ (photo : Fin num_photos → Fin (n_girls + n_boys) × Fin (n_girls + n_boys)),
      (∃ (i : Fin num_photos), (photo i).1 ≥ n_girls ∧ (photo i).2 ≥ n_girls) ∨
      (∃ (i : Fin num_photos), (photo i).1 < n_girls ∧ (photo i).2 < n_girls) ∨
      (∃ (i j : Fin num_photos), i ≠ j ∧ photo i = photo j))) :=
by sorry

end NUMINAMATH_CALUDE_min_photos_theorem_l4082_408257


namespace NUMINAMATH_CALUDE_sin_negative_thirty_degrees_l4082_408271

theorem sin_negative_thirty_degrees : Real.sin (-(30 * π / 180)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_thirty_degrees_l4082_408271


namespace NUMINAMATH_CALUDE_investment_ratio_equals_return_ratio_l4082_408206

/-- Given three investors with investments in some ratio, prove that their investment ratio
    is the same as their return ratio under certain conditions. -/
theorem investment_ratio_equals_return_ratio
  (a b c : ℕ) -- investments of A, B, and C
  (ra rb rc : ℕ) -- returns of A, B, and C
  (h1 : ra = 6 * k ∧ rb = 5 * k ∧ rc = 4 * k) -- return ratio condition
  (h2 : rb = ra + 250) -- B earns 250 more than A
  (h3 : ra + rb + rc = 7250) -- total earnings
  : ∃ (m : ℕ), a = 6 * m ∧ b = 5 * m ∧ c = 4 * m := by
  sorry


end NUMINAMATH_CALUDE_investment_ratio_equals_return_ratio_l4082_408206


namespace NUMINAMATH_CALUDE_TUVW_product_l4082_408274

def letter_value (c : Char) : ℕ :=
  c.toNat - 'A'.toNat + 1

theorem TUVW_product : 
  (letter_value 'T') * (letter_value 'U') * (letter_value 'V') * (letter_value 'W') = 
  2^3 * 3 * 5 * 7 * 11 * 23 := by
  sorry

end NUMINAMATH_CALUDE_TUVW_product_l4082_408274


namespace NUMINAMATH_CALUDE_triangle_side_ratio_range_l4082_408232

theorem triangle_side_ratio_range (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  A + B + C = π →
  S > 0 →
  2 * S = a^2 - (b - c)^2 →
  3 / 5 < b / c ∧ b / c < 5 / 3 := by
sorry


end NUMINAMATH_CALUDE_triangle_side_ratio_range_l4082_408232


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l4082_408204

/-- If the simplest quadratic radical √a is of the same type as √27, then a = 3 -/
theorem simplest_quadratic_radical (a : ℝ) : (∃ k : ℕ+, a = 27 * k^2) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l4082_408204


namespace NUMINAMATH_CALUDE_power_of_power_l4082_408201

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l4082_408201


namespace NUMINAMATH_CALUDE_unique_x_intercept_l4082_408277

/-- The parabola equation: x = -3y^2 + 2y + 3 -/
def parabola (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 3

/-- X-intercept occurs when y = 0 -/
def x_intercept : ℝ := parabola 0

/-- Theorem: The parabola has exactly one x-intercept -/
theorem unique_x_intercept : ∃! x : ℝ, ∃ y : ℝ, parabola y = x ∧ y = 0 := by sorry

end NUMINAMATH_CALUDE_unique_x_intercept_l4082_408277


namespace NUMINAMATH_CALUDE_man_against_stream_speed_l4082_408278

/-- Represents the speed of a man rowing a boat in different conditions -/
structure BoatSpeed where
  stillWaterRate : ℝ
  withStreamSpeed : ℝ

/-- Calculates the speed of the boat against the stream -/
def againstStreamSpeed (bs : BoatSpeed) : ℝ :=
  abs (2 * bs.stillWaterRate - bs.withStreamSpeed)

/-- Theorem: Given the man's rate in still water and speed with the stream,
    prove that his speed against the stream is 12 km/h -/
theorem man_against_stream_speed (bs : BoatSpeed)
    (h1 : bs.stillWaterRate = 7)
    (h2 : bs.withStreamSpeed = 26) :
    againstStreamSpeed bs = 12 := by
  sorry

end NUMINAMATH_CALUDE_man_against_stream_speed_l4082_408278


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l4082_408281

/-- The probability of two randomly chosen diagonals intersecting in a regular nonagon -/
theorem nonagon_diagonal_intersection_probability :
  let n : ℕ := 9  -- number of sides in a nonagon
  let total_diagonals : ℕ := n.choose 2 - n
  let diagonal_pairs : ℕ := total_diagonals.choose 2
  let intersecting_pairs : ℕ := n.choose 4
  (intersecting_pairs : ℚ) / diagonal_pairs = 14 / 39 :=
by sorry


end NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l4082_408281


namespace NUMINAMATH_CALUDE_divisibility_condition_l4082_408289

theorem divisibility_condition (a b : ℕ+) :
  (a.val * b.val^2 + b.val + 7) ∣ (a.val^2 * b.val + a.val + b.val) ↔
  (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ (∃ k : ℕ+, a = 7 * k.val^2 ∧ b = 7 * k.val) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l4082_408289


namespace NUMINAMATH_CALUDE_square_sum_of_product_and_sum_l4082_408215

theorem square_sum_of_product_and_sum (p q : ℝ) 
  (h1 : p * q = 12) 
  (h2 : p + q = 8) : 
  p^2 + q^2 = 40 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_product_and_sum_l4082_408215


namespace NUMINAMATH_CALUDE_A_suff_not_nec_D_l4082_408225

-- Define the propositions
variable (A B C D : Prop)

-- Define the relationships between the propositions
axiom A_suff_not_nec_B : (A → B) ∧ ¬(B → A)
axiom B_nec_and_suff_C : (B ↔ C)
axiom D_nec_not_suff_C : (C → D) ∧ ¬(D → C)

-- Theorem to prove
theorem A_suff_not_nec_D : (A → D) ∧ ¬(D → A) :=
sorry

end NUMINAMATH_CALUDE_A_suff_not_nec_D_l4082_408225


namespace NUMINAMATH_CALUDE_moon_speed_mph_approx_l4082_408228

/-- Conversion factor from kilometers to miles -/
def km_to_miles : ℝ := 0.621371

/-- Conversion factor from seconds to hours -/
def seconds_to_hours : ℝ := 3600

/-- The moon's speed in kilometers per second -/
def moon_speed_km_s : ℝ := 1.02

/-- Converts a speed from kilometers per second to miles per hour -/
def convert_km_s_to_mph (speed_km_s : ℝ) : ℝ :=
  speed_km_s * km_to_miles * seconds_to_hours

/-- Theorem stating that the moon's speed in miles per hour is approximately 2281.34 -/
theorem moon_speed_mph_approx :
  ∃ ε > 0, |convert_km_s_to_mph moon_speed_km_s - 2281.34| < ε :=
sorry

end NUMINAMATH_CALUDE_moon_speed_mph_approx_l4082_408228


namespace NUMINAMATH_CALUDE_sum_product_range_l4082_408253

theorem sum_product_range (a b c : ℝ) (h : a + b + c = 0) :
  (∀ x : ℝ, x ≤ 0 → ∃ a b c : ℝ, a + b + c = 0 ∧ a * b + a * c + b * c = x) ∧
  (∀ a b c : ℝ, a + b + c = 0 → a * b + a * c + b * c ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_sum_product_range_l4082_408253


namespace NUMINAMATH_CALUDE_linear_function_properties_l4082_408244

/-- Linear function defined as f(x) = -2x + 4 -/
def f (x : ℝ) : ℝ := -2 * x + 4

theorem linear_function_properties :
  /- Property 1: For any two points on the graph, if x₁ < x₂, then f(x₁) > f(x₂) -/
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂) ∧
  /- Property 2: The graph does not pass through the third quadrant -/
  (∀ x y : ℝ, f x = y → (x ≤ 0 → y ≥ 0) ∧ (y ≤ 0 → x ≥ 0)) ∧
  /- Property 3: Shifting the graph down by 4 units results in y = -2x -/
  (∀ x : ℝ, f x - 4 = -2 * x) ∧
  /- Property 4: The x-intercept is at (2, 0) -/
  (f 2 = 0 ∧ ∀ x : ℝ, f x = 0 → x = 2) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l4082_408244


namespace NUMINAMATH_CALUDE_triangle_properties_l4082_408245

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition -/
def condition (t : Triangle) : Prop :=
  (2 * t.a + t.b) * Real.cos t.C + t.c * Real.cos t.B = 0

theorem triangle_properties (t : Triangle) 
  (h : condition t) : 
  t.C = 2 * Real.pi / 3 ∧ 
  (t.c = 6 → ∃ (max_area : ℝ), max_area = 3 * Real.sqrt 3 ∧ 
    ∀ (area : ℝ), area ≤ max_area) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l4082_408245


namespace NUMINAMATH_CALUDE_wickets_before_last_match_is_55_l4082_408298

/-- Represents a bowler's statistics -/
structure BowlerStats where
  initialAverage : ℝ
  lastMatchWickets : ℕ
  lastMatchRuns : ℕ
  averageDecrease : ℝ

/-- Calculates the number of wickets taken before the last match -/
def wicketsBeforeLastMatch (stats : BowlerStats) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions, the bowler took 55 wickets before the last match -/
theorem wickets_before_last_match_is_55 (stats : BowlerStats)
  (h1 : stats.initialAverage = 12.4)
  (h2 : stats.lastMatchWickets = 4)
  (h3 : stats.lastMatchRuns = 26)
  (h4 : stats.averageDecrease = 0.4) :
  wicketsBeforeLastMatch stats = 55 :=
  sorry

end NUMINAMATH_CALUDE_wickets_before_last_match_is_55_l4082_408298


namespace NUMINAMATH_CALUDE_product_units_digit_base8_l4082_408249

theorem product_units_digit_base8 : ∃ (n : ℕ), 
  (505 * 71) % 8 = n ∧ n = ((505 % 8) * (71 % 8)) % 8 := by
  sorry

end NUMINAMATH_CALUDE_product_units_digit_base8_l4082_408249


namespace NUMINAMATH_CALUDE_total_employees_after_increase_l4082_408233

-- Define the initial conditions
def initial_total : ℕ := 1200
def initial_production : ℕ := 800
def initial_admin : ℕ := 400
def production_increase : ℚ := 35 / 100
def admin_increase : ℚ := 3 / 5

-- Define the theorem
theorem total_employees_after_increase : 
  initial_production * (1 + production_increase) + initial_admin * (1 + admin_increase) = 1720 := by
  sorry

end NUMINAMATH_CALUDE_total_employees_after_increase_l4082_408233


namespace NUMINAMATH_CALUDE_marble_problem_l4082_408237

/-- The number of white marbles in the bag -/
def white_marbles : ℕ := 3

/-- The probability that all 3 girls select the same colored marble -/
def same_color_prob : ℚ := 1/10

/-- The number of black marbles in the bag -/
def black_marbles : ℕ := 3

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := white_marbles + black_marbles

/-- The probability of all girls selecting white marbles -/
def all_white_prob : ℚ := (white_marbles / total_marbles) * 
                          ((white_marbles - 1) / (total_marbles - 1)) * 
                          ((white_marbles - 2) / (total_marbles - 2))

/-- The probability of all girls selecting black marbles -/
def all_black_prob : ℚ := (black_marbles / total_marbles) * 
                          ((black_marbles - 1) / (total_marbles - 1)) * 
                          ((black_marbles - 2) / (total_marbles - 2))

theorem marble_problem : 
  all_white_prob + all_black_prob = same_color_prob :=
by sorry

end NUMINAMATH_CALUDE_marble_problem_l4082_408237


namespace NUMINAMATH_CALUDE_angle_of_inclination_slope_one_l4082_408286

/-- The angle of inclination of a line with slope 1 in the Cartesian coordinate system is π/4 -/
theorem angle_of_inclination_slope_one :
  let line := {(x, y) : ℝ × ℝ | x - y - 3 = 0}
  let slope : ℝ := 1
  let angle_of_inclination := Real.arctan slope
  angle_of_inclination = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_of_inclination_slope_one_l4082_408286


namespace NUMINAMATH_CALUDE_smallest_angle_through_point_l4082_408258

theorem smallest_angle_through_point (α : Real) : 
  (∃ k : ℤ, α = 11 * Real.pi / 6 + 2 * Real.pi * k) ∧ 
  (∀ β : Real, β > 0 → 
    (Real.sin β = Real.sin (2 * Real.pi / 3) ∧ 
     Real.cos β = Real.cos (2 * Real.pi / 3)) → 
    α ≤ β) ↔ 
  (Real.sin α = Real.sin (2 * Real.pi / 3) ∧ 
   Real.cos α = Real.cos (2 * Real.pi / 3) ∧ 
   α > 0 ∧ 
   α < 2 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_through_point_l4082_408258


namespace NUMINAMATH_CALUDE_sin_75_degrees_l4082_408261

theorem sin_75_degrees : Real.sin (75 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_75_degrees_l4082_408261


namespace NUMINAMATH_CALUDE_range_of_2x_plus_y_min_value_of_c_l4082_408282

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 2*y

-- Theorem 1: Range of 2x + y
theorem range_of_2x_plus_y :
  ∀ x y : ℝ, Circle x y → 1 - Real.sqrt 2 ≤ 2*x + y ∧ 2*x + y ≤ 1 + Real.sqrt 2 :=
sorry

-- Theorem 2: Minimum value of c
theorem min_value_of_c :
  (∃ c : ℝ, ∀ x y : ℝ, Circle x y → x + y + c > 0) ∧
  (∀ c' : ℝ, (∀ x y : ℝ, Circle x y → x + y + c' > 0) → c' ≥ -1) :=
sorry

end NUMINAMATH_CALUDE_range_of_2x_plus_y_min_value_of_c_l4082_408282


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4082_408213

theorem sqrt_equation_solution :
  ∃ s : ℝ, (Real.sqrt (3 * Real.sqrt (s - 1)) = (9 - s) ^ (1/4)) ∧ s = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4082_408213


namespace NUMINAMATH_CALUDE_car_trading_profit_l4082_408218

theorem car_trading_profit (P : ℝ) (h : P > 0) : 
  let discount_rate : ℝ := 0.3
  let increase_rate : ℝ := 0.7
  let buying_price : ℝ := P * (1 - discount_rate)
  let selling_price : ℝ := buying_price * (1 + increase_rate)
  let profit : ℝ := selling_price - P
  profit / P = 0.19 := by sorry

end NUMINAMATH_CALUDE_car_trading_profit_l4082_408218


namespace NUMINAMATH_CALUDE_balloon_arrangements_l4082_408200

theorem balloon_arrangements (n : ℕ) (n1 n2 n3 n4 n5 : ℕ) : 
  n = 7 → 
  n1 = 2 → 
  n2 = 2 → 
  n3 = 1 → 
  n4 = 1 → 
  n5 = 1 → 
  (n.factorial) / (n1.factorial * n2.factorial * n3.factorial * n4.factorial * n5.factorial) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_balloon_arrangements_l4082_408200


namespace NUMINAMATH_CALUDE_sphere_radius_equal_volume_cone_l4082_408268

/-- The radius of a sphere with the same volume as a cone -/
theorem sphere_radius_equal_volume_cone (r h : ℝ) (hr : r = 2) (hh : h = 8) :
  ∃ (r_sphere : ℝ), (1/3 * π * r^2 * h) = (4/3 * π * r_sphere^3) ∧ r_sphere = 2 * (2 : ℝ)^(1/3) :=
sorry

end NUMINAMATH_CALUDE_sphere_radius_equal_volume_cone_l4082_408268


namespace NUMINAMATH_CALUDE_initial_amount_of_liquid_A_l4082_408239

/-- Given a mixture of liquids A and B with an initial ratio of 4:1, prove that the initial amount
of liquid A is 16 liters when 10 L of the mixture is replaced with liquid B, resulting in a new
ratio of 2:3. -/
theorem initial_amount_of_liquid_A (x : ℝ) : 
  (4 * x) / x = 4 / 1 →  -- Initial ratio of A to B is 4:1
  ((4 * x - 8) / (x + 8) = 2 / 3) →  -- New ratio after replacement is 2:3
  4 * x = 16 :=  -- Initial amount of liquid A is 16 liters
by sorry

#check initial_amount_of_liquid_A

end NUMINAMATH_CALUDE_initial_amount_of_liquid_A_l4082_408239


namespace NUMINAMATH_CALUDE_total_ladybugs_count_l4082_408221

/-- The number of ladybugs with spots -/
def ladybugs_with_spots : ℕ := 12170

/-- The number of ladybugs without spots -/
def ladybugs_without_spots : ℕ := 54912

/-- The total number of ladybugs -/
def total_ladybugs : ℕ := ladybugs_with_spots + ladybugs_without_spots

theorem total_ladybugs_count : total_ladybugs = 67082 := by
  sorry

end NUMINAMATH_CALUDE_total_ladybugs_count_l4082_408221


namespace NUMINAMATH_CALUDE_room_length_is_twenty_l4082_408273

/-- Represents the dimensions and tiling of a rectangular room. -/
structure Room where
  length : ℝ
  breadth : ℝ
  tileSize : ℝ
  blackTileWidth : ℝ
  blueTileCount : ℕ

/-- Theorem stating the length of the room given specific conditions. -/
theorem room_length_is_twenty (r : Room) : 
  r.breadth = 10 ∧ 
  r.tileSize = 2 ∧ 
  r.blackTileWidth = 2 ∧ 
  r.blueTileCount = 16 ∧
  (r.length - 2 * r.blackTileWidth) * (r.breadth - 2 * r.blackTileWidth) * (2/3) = 
    (r.blueTileCount : ℝ) * r.tileSize * r.tileSize →
  r.length = 20 := by
  sorry

#check room_length_is_twenty

end NUMINAMATH_CALUDE_room_length_is_twenty_l4082_408273


namespace NUMINAMATH_CALUDE_radical_simplification_l4082_408267

theorem radical_simplification (a : ℝ) (ha : a > 0) :
  Real.sqrt (50 * a^3) * Real.sqrt (18 * a^2) * Real.sqrt (98 * a^5) = 42 * a^5 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l4082_408267


namespace NUMINAMATH_CALUDE_exactly_six_valid_tuples_l4082_408294

def is_valid_tuple (t : Fin 4 → Fin 4) : Prop :=
  (∃ (σ : Equiv (Fin 4) (Fin 4)), ∀ i, t i = σ i) ∧
  (t 0 = 1 ∨ t 1 ≠ 1 ∨ t 2 = 2 ∨ t 3 ≠ 4) ∧
  ¬(t 0 = 1 ∧ t 1 ≠ 1) ∧
  ¬(t 0 = 1 ∧ t 2 = 2) ∧
  ¬(t 0 = 1 ∧ t 3 ≠ 4) ∧
  ¬(t 1 ≠ 1 ∧ t 2 = 2) ∧
  ¬(t 1 ≠ 1 ∧ t 3 ≠ 4) ∧
  ¬(t 2 = 2 ∧ t 3 ≠ 4)

theorem exactly_six_valid_tuples :
  ∃! (s : Finset (Fin 4 → Fin 4)), s.card = 6 ∧ ∀ t, t ∈ s ↔ is_valid_tuple t :=
sorry

end NUMINAMATH_CALUDE_exactly_six_valid_tuples_l4082_408294


namespace NUMINAMATH_CALUDE_min_intercept_sum_l4082_408229

/-- Given a line passing through (1, 2) with equation x/a + y/b = 1 where a > 0 and b > 0,
    the minimum value of a + b is 3 + 2√2 -/
theorem min_intercept_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_line : 1 / a + 2 / b = 1) : 
  ∀ (x y : ℝ), x / a + y / b = 1 → x + y ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_intercept_sum_l4082_408229


namespace NUMINAMATH_CALUDE_map_area_calculation_l4082_408288

/-- Given a map scale and an area on the map, calculate the actual area -/
theorem map_area_calculation (scale : ℝ) (map_area : ℝ) (actual_area : ℝ) :
  scale = 1 / 50000 →
  map_area = 100 →
  actual_area = 2.5 * 10^7 →
  map_area / actual_area = scale^2 :=
by sorry

end NUMINAMATH_CALUDE_map_area_calculation_l4082_408288


namespace NUMINAMATH_CALUDE_equation_equivalence_l4082_408236

theorem equation_equivalence :
  ∀ x : ℝ, (x - 1) / 0.2 - x / 0.5 = 1 ↔ 3 * x = 6 := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l4082_408236


namespace NUMINAMATH_CALUDE_white_then_red_probability_l4082_408216

/-- The probability of drawing a white marble first and a red marble second from a bag with 4 red and 6 white marbles -/
theorem white_then_red_probability : 
  let total_marbles : ℕ := 4 + 6
  let red_marbles : ℕ := 4
  let white_marbles : ℕ := 6
  let prob_white_first : ℚ := white_marbles / total_marbles
  let prob_red_second : ℚ := red_marbles / (total_marbles - 1)
  prob_white_first * prob_red_second = 4 / 15 :=
by sorry

end NUMINAMATH_CALUDE_white_then_red_probability_l4082_408216


namespace NUMINAMATH_CALUDE_square_equation_proof_l4082_408263

theorem square_equation_proof (h1 : 3 > 1) (h2 : 1 > 1) : (3 * (1^3 + 3))^2 = 8339 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_proof_l4082_408263


namespace NUMINAMATH_CALUDE_modulo_eleven_residue_l4082_408256

theorem modulo_eleven_residue : (341 + 6 * 50 + 4 * 156 + 3 * 12^2) % 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_modulo_eleven_residue_l4082_408256


namespace NUMINAMATH_CALUDE_square_plus_self_even_l4082_408219

theorem square_plus_self_even (n : ℤ) : ∃ k : ℤ, n^2 + n = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_square_plus_self_even_l4082_408219


namespace NUMINAMATH_CALUDE_chromium_content_bounds_l4082_408227

/-- Represents the chromium content in an alloy mixture -/
structure ChromiumAlloy where
  x : ℝ  -- Relative mass of 1st alloy
  y : ℝ  -- Relative mass of 2nd alloy
  z : ℝ  -- Relative mass of 3rd alloy
  k : ℝ  -- Chromium content

/-- Conditions for a valid ChromiumAlloy -/
def is_valid_alloy (a : ChromiumAlloy) : Prop :=
  a.x ≥ 0 ∧ a.y ≥ 0 ∧ a.z ≥ 0 ∧
  a.x + a.y + a.z = 1 ∧
  0.9 * a.x + 0.3 * a.z = 0.45 ∧
  0.4 * a.x + 0.1 * a.y + 0.5 * a.z = a.k

theorem chromium_content_bounds (a : ChromiumAlloy) 
  (h : is_valid_alloy a) : 
  a.k ≥ 0.25 ∧ a.k ≤ 0.4 := by
  sorry

end NUMINAMATH_CALUDE_chromium_content_bounds_l4082_408227


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l4082_408276

theorem lcm_gcf_problem (n : ℕ) : 
  Nat.lcm n 12 = 48 → Nat.gcd n 12 = 8 → n = 32 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l4082_408276


namespace NUMINAMATH_CALUDE_function_range_l4082_408205

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem function_range :
  ∀ y ∈ Set.Icc 2 6, ∃ x ∈ Set.Icc (-1) 2, f x = y ∧
  ∀ x ∈ Set.Icc (-1) 2, f x ∈ Set.Icc 2 6 :=
by sorry

end NUMINAMATH_CALUDE_function_range_l4082_408205


namespace NUMINAMATH_CALUDE_wire_cut_square_octagon_ratio_l4082_408202

theorem wire_cut_square_octagon_ratio (a b : ℝ) (h_positive_a : 0 < a) (h_positive_b : 0 < b) :
  (a^2 / 16 = b^2 * (1 + Real.sqrt 2) / 32) → a / b = Real.sqrt ((2 + Real.sqrt 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_wire_cut_square_octagon_ratio_l4082_408202


namespace NUMINAMATH_CALUDE_unique_solution_exists_l4082_408293

theorem unique_solution_exists (y : ℝ) (h : y > 0) :
  ∃! x : ℝ, (2 ^ (4 * x + 2)) * (4 ^ (2 * x + 3)) = 8 ^ (3 * x + 4) * y :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l4082_408293


namespace NUMINAMATH_CALUDE_min_vector_difference_l4082_408220

/-- Given planar vectors a, b, c satisfying the conditions, 
    the minimum value of |a - b| is 6 -/
theorem min_vector_difference (a b c : ℝ × ℝ) 
    (h1 : a • b = 0)
    (h2 : ‖c‖ = 1)
    (h3 : ‖a - c‖ = 5)
    (h4 : ‖b - c‖ = 5) :
    6 ≤ ‖a - b‖ ∧ ∃ (a' b' c' : ℝ × ℝ), 
      a' • b' = 0 ∧ 
      ‖c'‖ = 1 ∧ 
      ‖a' - c'‖ = 5 ∧ 
      ‖b' - c'‖ = 5 ∧
      ‖a' - b'‖ = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_vector_difference_l4082_408220


namespace NUMINAMATH_CALUDE_f_sum_difference_equals_two_l4082_408203

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem f_sum_difference_equals_two :
  f 2016 + (deriv f) 2016 + f (-2016) - (deriv f) (-2016) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_difference_equals_two_l4082_408203


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_perpendicular_to_parallel_planes_l4082_408252

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the basic relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- Theorem 1: If two lines are perpendicular to the same plane, then they are parallel
theorem perpendicular_lines_parallel 
  (m n : Line) (α : Plane) 
  (h1 : perpendicular m α) (h2 : perpendicular n α) : 
  parallel m n :=
sorry

-- Theorem 2: If three planes are parallel and a line is perpendicular to one of them, 
-- then it is perpendicular to all of them
theorem perpendicular_to_parallel_planes 
  (m : Line) (α β γ : Plane)
  (h1 : plane_parallel α β) (h2 : plane_parallel β γ) 
  (h3 : perpendicular m α) :
  perpendicular m γ :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_perpendicular_to_parallel_planes_l4082_408252


namespace NUMINAMATH_CALUDE_sum_is_negative_l4082_408247

theorem sum_is_negative (x y : ℝ) (hx : x > 0) (hy : y < 0) (hxy : |x| < |y|) : x + y < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_is_negative_l4082_408247


namespace NUMINAMATH_CALUDE_biancas_birthday_money_l4082_408238

theorem biancas_birthday_money (amount_per_friend : ℕ) (total_amount : ℕ) : 
  amount_per_friend = 6 → total_amount = 30 → total_amount / amount_per_friend = 5 := by
  sorry

end NUMINAMATH_CALUDE_biancas_birthday_money_l4082_408238


namespace NUMINAMATH_CALUDE_positions_after_631_moves_l4082_408222

/-- Represents the possible positions of the dog on the hexagon -/
inductive DogPosition
  | Top
  | TopRight
  | BottomRight
  | Bottom
  | BottomLeft
  | TopLeft

/-- Represents the possible positions of the rabbit on the hexagon -/
inductive RabbitPosition
  | TopCenter
  | TopRight
  | RightUpper
  | RightLower
  | BottomRight
  | BottomCenter
  | BottomLeft
  | LeftLower
  | LeftUpper
  | TopLeft
  | LeftCenter
  | RightCenter

/-- Calculates the position of the dog after a given number of moves -/
def dogPositionAfterMoves (moves : Nat) : DogPosition :=
  match moves % 6 with
  | 0 => DogPosition.TopLeft
  | 1 => DogPosition.Top
  | 2 => DogPosition.TopRight
  | 3 => DogPosition.BottomRight
  | 4 => DogPosition.Bottom
  | 5 => DogPosition.BottomLeft
  | _ => DogPosition.Top  -- This case is unreachable, but needed for exhaustiveness

/-- Calculates the position of the rabbit after a given number of moves -/
def rabbitPositionAfterMoves (moves : Nat) : RabbitPosition :=
  match moves % 12 with
  | 0 => RabbitPosition.RightCenter
  | 1 => RabbitPosition.TopCenter
  | 2 => RabbitPosition.TopRight
  | 3 => RabbitPosition.RightUpper
  | 4 => RabbitPosition.RightLower
  | 5 => RabbitPosition.BottomRight
  | 6 => RabbitPosition.BottomCenter
  | 7 => RabbitPosition.BottomLeft
  | 8 => RabbitPosition.LeftLower
  | 9 => RabbitPosition.LeftUpper
  | 10 => RabbitPosition.TopLeft
  | 11 => RabbitPosition.LeftCenter
  | _ => RabbitPosition.TopCenter  -- This case is unreachable, but needed for exhaustiveness

theorem positions_after_631_moves :
  dogPositionAfterMoves 631 = DogPosition.Top ∧
  rabbitPositionAfterMoves 631 = RabbitPosition.BottomLeft :=
by sorry

end NUMINAMATH_CALUDE_positions_after_631_moves_l4082_408222


namespace NUMINAMATH_CALUDE_farm_leg_count_l4082_408283

def farm_animals : ℕ := 13
def chickens : ℕ := 4
def chicken_legs : ℕ := 2
def buffalo_legs : ℕ := 4

theorem farm_leg_count : 
  (chickens * chicken_legs) + ((farm_animals - chickens) * buffalo_legs) = 44 := by
  sorry

end NUMINAMATH_CALUDE_farm_leg_count_l4082_408283


namespace NUMINAMATH_CALUDE_twentieth_term_of_sequence_l4082_408217

/-- An arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

/-- The 20th term of the arithmetic sequence 8, 5, 2, ... -/
theorem twentieth_term_of_sequence : arithmeticSequence 8 (-3) 20 = -49 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_of_sequence_l4082_408217


namespace NUMINAMATH_CALUDE_average_score_is_correct_l4082_408251

def total_students : ℕ := 120

-- Define the score distribution
def score_distribution : List (ℕ × ℕ) := [
  (95, 12),
  (85, 24),
  (75, 30),
  (65, 20),
  (55, 18),
  (45, 10),
  (35, 6)
]

-- Calculate the average score
def average_score : ℚ :=
  let total_score : ℕ := (score_distribution.map (λ (score, count) => score * count)).sum
  (total_score : ℚ) / total_students

-- Theorem to prove
theorem average_score_is_correct :
  average_score = 8380 / 120 := by sorry

end NUMINAMATH_CALUDE_average_score_is_correct_l4082_408251


namespace NUMINAMATH_CALUDE_milo_run_distance_l4082_408280

/-- Milo's running speed in miles per hour -/
def milo_run_speed : ℝ := 3

/-- Milo's skateboard rolling speed in miles per hour -/
def milo_roll_speed : ℝ := milo_run_speed * 2

/-- Cory's wheelchair driving speed in miles per hour -/
def cory_drive_speed : ℝ := 12

/-- Time Milo runs in hours -/
def run_time : ℝ := 2

theorem milo_run_distance : 
  (milo_roll_speed = milo_run_speed * 2) →
  (cory_drive_speed = milo_roll_speed * 2) →
  (cory_drive_speed = 12) →
  (milo_run_speed * run_time = 6) :=
by
  sorry


end NUMINAMATH_CALUDE_milo_run_distance_l4082_408280


namespace NUMINAMATH_CALUDE_smallest_number_l4082_408248

theorem smallest_number : ∀ (a b c d : ℚ), 
  a = 1 → b = -2 → c = 0 → d = -1/2 → 
  b ≤ a ∧ b ≤ c ∧ b ≤ d := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l4082_408248


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4082_408287

/-- A geometric sequence with the given properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a → a 3 + a 5 = 20 → a 4 = 8 → a 2 + a 6 = 34 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4082_408287


namespace NUMINAMATH_CALUDE_intersection_equals_open_closed_interval_l4082_408212

-- Define the sets M and N
def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 4}

-- State the theorem
theorem intersection_equals_open_closed_interval : M ∩ N = Set.Ioc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_open_closed_interval_l4082_408212


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l4082_408230

/-- A pyramid with a regular hexagonal base and isosceles triangular lateral faces -/
structure HexagonalPyramid where
  base_side_length : ℝ
  lateral_face_base_length : ℝ
  lateral_face_height : ℝ

/-- A cube inscribed in a hexagonal pyramid -/
structure InscribedCube where
  pyramid : HexagonalPyramid
  edge_length : ℝ

/-- The volume of a cube -/
def cube_volume (c : InscribedCube) : ℝ := c.edge_length ^ 3

/-- Conditions for the specific pyramid and inscribed cube -/
def specific_pyramid_and_cube : Prop :=
  ∃ (p : HexagonalPyramid) (c : InscribedCube),
    p.base_side_length = 1 ∧
    p.lateral_face_base_length = 1 ∧
    c.pyramid = p ∧
    c.edge_length = 1

/-- Theorem stating that the volume of the inscribed cube is 1 -/
theorem inscribed_cube_volume :
  specific_pyramid_and_cube →
  ∃ (c : InscribedCube), cube_volume c = 1 :=
sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l4082_408230


namespace NUMINAMATH_CALUDE_handshake_count_l4082_408266

theorem handshake_count (n : ℕ) (h : n = 8) :
  let pairs := n / 2
  let handshakes_per_person := n - 2
  (n * handshakes_per_person) / 2 = 24 :=
by sorry

end NUMINAMATH_CALUDE_handshake_count_l4082_408266


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l4082_408208

def is_arithmetic_sequence (x y z : ℝ) : Prop :=
  y - x = z - y

def is_geometric_sequence (x y z w v : ℝ) : Prop :=
  y / x = z / y ∧ z / y = w / z ∧ w / z = v / w

theorem arithmetic_geometric_sequence_property :
  ∀ (a b m n : ℝ),
  is_arithmetic_sequence (-9) a (-1) →
  is_geometric_sequence (-9) m b n (-1) →
  a * b = 15 := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l4082_408208


namespace NUMINAMATH_CALUDE_sphere_volume_equals_cone_cylinder_volume_l4082_408207

/-- Given a cone with height 6 and radius 1.5, and a cylinder with the same height and volume as the cone,
    prove that a sphere with radius 1.5 has the same volume as both the cone and cylinder. -/
theorem sphere_volume_equals_cone_cylinder_volume :
  let cone_height : ℝ := 6
  let cone_radius : ℝ := 1.5
  let cylinder_height : ℝ := cone_height
  let cone_volume : ℝ := (1 / 3) * Real.pi * cone_radius^2 * cone_height
  let cylinder_volume : ℝ := cone_volume
  let sphere_radius : ℝ := 1.5
  let sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius^3
  sphere_volume = cone_volume := by
  sorry


end NUMINAMATH_CALUDE_sphere_volume_equals_cone_cylinder_volume_l4082_408207


namespace NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l4082_408270

/-- Given a line ax - by - 2 = 0 and a curve y = x^3 intersecting at point P(1, 1),
    if the tangent lines at P are perpendicular, then a/b = -1/3 -/
theorem perpendicular_tangents_ratio (a b : ℝ) : 
  (∀ x y, a * x - b * y - 2 = 0 → y = x^3) →  -- Line and curve equations
  (a * 1 - b * 1 - 2 = 0) →                   -- Point P(1, 1) satisfies line equation
  (1 = 1^3) →                                 -- Point P(1, 1) satisfies curve equation
  (∃ k₁ k₂ : ℝ, k₁ * k₂ = -1 ∧                -- Perpendicular tangent lines condition
              k₁ = a / b ∧                    -- Slope of line
              k₂ = 3 * 1^2) →                 -- Slope of curve at P(1, 1)
  a / b = -1 / 3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l4082_408270


namespace NUMINAMATH_CALUDE_scientific_notation_502000_l4082_408209

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coefficient : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_502000 :
  toScientificNotation 502000 = ScientificNotation.mk 5.02 5 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_502000_l4082_408209


namespace NUMINAMATH_CALUDE_incircle_and_inscribed_circles_inequality_l4082_408296

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a circle
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the theorem
theorem incircle_and_inscribed_circles_inequality 
  (triangle : Triangle) 
  (incircle : Circle) 
  (inscribed_circle1 inscribed_circle2 inscribed_circle3 : Circle) :
  -- Conditions
  (incircle.radius > 0) →
  (inscribed_circle1.radius > 0) →
  (inscribed_circle2.radius > 0) →
  (inscribed_circle3.radius > 0) →
  (inscribed_circle1.radius < incircle.radius) →
  (inscribed_circle2.radius < incircle.radius) →
  (inscribed_circle3.radius < incircle.radius) →
  -- Theorem statement
  inscribed_circle1.radius + inscribed_circle2.radius + inscribed_circle3.radius ≥ incircle.radius :=
by
  sorry

end NUMINAMATH_CALUDE_incircle_and_inscribed_circles_inequality_l4082_408296


namespace NUMINAMATH_CALUDE_min_value_product_l4082_408290

theorem min_value_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_abc : a * b * c = 8) :
  (2 * a + 3 * b) * (2 * b + 3 * c) * (2 * c + 3 * a) ≥ 288 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_l4082_408290


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l4082_408235

/-- An isosceles triangle with side lengths that are roots of x^2 - 4x + 3 = 0 -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  is_root : base^2 - 4*base + 3 = 0 ∧ leg^2 - 4*leg + 3 = 0
  is_isosceles : base ≠ leg
  triangle_inequality : base < 2*leg

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.base + 2*t.leg

/-- Theorem: The perimeter of the isosceles triangle is 7 -/
theorem isosceles_triangle_perimeter : 
  ∀ t : IsoscelesTriangle, perimeter t = 7 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l4082_408235


namespace NUMINAMATH_CALUDE_periodic_even_function_value_l4082_408211

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem periodic_even_function_value 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h_even : is_even f)
  (h_periodic : has_period f 6)
  (h_interval : ∀ x ∈ Set.Icc (-3) 3, f x = (x + 1) * (x - a)) :
  f (-6) = -1 :=
sorry

end NUMINAMATH_CALUDE_periodic_even_function_value_l4082_408211


namespace NUMINAMATH_CALUDE_seeds_in_small_gardens_l4082_408272

theorem seeds_in_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (num_small_gardens : ℕ) :
  total_seeds = 42 →
  big_garden_seeds = 36 →
  num_small_gardens = 3 →
  num_small_gardens > 0 →
  total_seeds ≥ big_garden_seeds →
  (total_seeds - big_garden_seeds) % num_small_gardens = 0 →
  (total_seeds - big_garden_seeds) / num_small_gardens = 2 := by
sorry

end NUMINAMATH_CALUDE_seeds_in_small_gardens_l4082_408272


namespace NUMINAMATH_CALUDE_count_valid_triples_l4082_408297

def valid_triple (x y z : ℕ+) : Prop :=
  Nat.lcm x.val y.val = 48 ∧
  Nat.lcm x.val z.val = 450 ∧
  Nat.lcm y.val z.val = 600

theorem count_valid_triples :
  ∃! (n : ℕ), ∃ (S : Finset (ℕ+ × ℕ+ × ℕ+)),
    S.card = n ∧
    (∀ (t : ℕ+ × ℕ+ × ℕ+), t ∈ S ↔ valid_triple t.1 t.2.1 t.2.2) ∧
    n = 5 :=
sorry

end NUMINAMATH_CALUDE_count_valid_triples_l4082_408297


namespace NUMINAMATH_CALUDE_slide_total_boys_l4082_408291

theorem slide_total_boys (initial : ℕ) (second : ℕ) (third : ℕ) 
  (h1 : initial = 87) 
  (h2 : second = 46) 
  (h3 : third = 29) : 
  initial + second + third = 162 := by
  sorry

end NUMINAMATH_CALUDE_slide_total_boys_l4082_408291


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l4082_408231

theorem complex_fraction_equals_i : (1 + Complex.I) / (1 - Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l4082_408231


namespace NUMINAMATH_CALUDE_unique_four_digit_consecutive_square_swap_l4082_408234

def is_consecutive_digits (n : ℕ) : Prop :=
  ∃ x : ℕ, x ≤ 6 ∧ 
    n = 1000 * x + 100 * (x + 1) + 10 * (x + 2) + (x + 3)

def swap_thousands_hundreds (n : ℕ) : ℕ :=
  let thousands := n / 1000
  let hundreds := (n / 100) % 10
  let tens := (n / 10) % 10
  let ones := n % 10
  1000 * hundreds + 100 * thousands + 10 * tens + ones

theorem unique_four_digit_consecutive_square_swap :
  ∃! n : ℕ, 1000 ≤ n ∧ n < 10000 ∧
    is_consecutive_digits n ∧
    ∃ m : ℕ, swap_thousands_hundreds n = m * m :=
by
  use 3456
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_consecutive_square_swap_l4082_408234


namespace NUMINAMATH_CALUDE_parallelogram_area_l4082_408240

theorem parallelogram_area (base height : ℝ) (h1 : base = 12) (h2 : height = 18) :
  base * height = 216 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_area_l4082_408240


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4082_408279

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4082_408279


namespace NUMINAMATH_CALUDE_unique_prime_triplet_l4082_408214

theorem unique_prime_triplet : ∃! p : ℕ, Prime p ∧ Prime (p + 2) ∧ Prime (p + 4) ∧ p = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_triplet_l4082_408214


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l4082_408254

/-- Given a hyperbola with equation x²/a² - y²/2 = 1 (a > 0) and eccentricity √3,
    prove that its asymptotes are y = ±√2 x -/
theorem hyperbola_asymptotes (a : ℝ) (h1 : a > 0) :
  let hyperbola := λ (x y : ℝ) => x^2 / a^2 - y^2 / 2 = 1
  let eccentricity := Real.sqrt 3
  let asymptotes := λ (x y : ℝ) => y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x
  ∀ (x y : ℝ), hyperbola x y ∧ eccentricity = Real.sqrt 3 → asymptotes x y :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l4082_408254
