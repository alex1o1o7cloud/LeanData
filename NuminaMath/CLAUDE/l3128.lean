import Mathlib

namespace NUMINAMATH_CALUDE_smallest_max_sum_l3128_312852

theorem smallest_max_sum (p q r s t : ℕ+) (h : p + q + r + s + t = 2025) :
  let N := max (p + q) (max (q + r) (max (r + s) (s + t)))
  ∀ m : ℕ, (∃ p' q' r' s' t' : ℕ+, p' + q' + r' + s' + t' = 2025 ∧ 
    max (p' + q') (max (q' + r') (max (r' + s') (s' + t'))) < m) → m > 676 := by
  sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l3128_312852


namespace NUMINAMATH_CALUDE_base6_sum_is_6_l3128_312854

/-- Represents a single digit in base 6 -/
def Base6Digit := Fin 6

/-- The addition problem in base 6 -/
def base6_addition (X Y : Base6Digit) : Prop :=
  ∃ (carry : Nat),
    (3 * 6^2 + X.val * 6 + Y.val) + 24 = 
    6 * 6^2 + carry * 6 + X.val

/-- The main theorem to prove -/
theorem base6_sum_is_6 :
  ∀ X Y : Base6Digit,
    base6_addition X Y →
    (X.val : ℕ) + (Y.val : ℕ) = 6 := by sorry

end NUMINAMATH_CALUDE_base6_sum_is_6_l3128_312854


namespace NUMINAMATH_CALUDE_starting_lineup_count_l3128_312802

-- Define the total number of players
def total_players : ℕ := 12

-- Define the number of twins
def num_twins : ℕ := 2

-- Define the size of the starting lineup
def lineup_size : ℕ := 5

-- Theorem statement
theorem starting_lineup_count :
  (num_twins * Nat.choose (total_players - 1) (lineup_size - 1)) = 660 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_count_l3128_312802


namespace NUMINAMATH_CALUDE_inequality_proof_l3128_312813

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2 + y^2 + z^2 + x*y + y*z + z*x ≤ 1) :
  (1/x - 1) * (1/y - 1) * (1/z - 1) ≥ 9 * Real.sqrt 6 - 19 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3128_312813


namespace NUMINAMATH_CALUDE_range_of_a_l3128_312807

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) ↔ a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3128_312807


namespace NUMINAMATH_CALUDE_prob_12th_last_value_l3128_312879

/-- Probability of getting a different roll on a four-sided die -/
def p_different : ℚ := 3 / 4

/-- Probability of getting the same roll on a four-sided die -/
def p_same : ℚ := 1 / 4

/-- Number of rolls before the final roll -/
def n : ℕ := 11

/-- Probability of the 12th roll being the last roll -/
def prob_12th_last : ℚ := p_different ^ n * p_same

theorem prob_12th_last_value : 
  prob_12th_last = (3 ^ 10 : ℚ) / (4 ^ 11 : ℚ) := by sorry

end NUMINAMATH_CALUDE_prob_12th_last_value_l3128_312879


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l3128_312855

def p (x : ℝ) : ℝ := 2 * x^3 + 5 * x^2 - 3 * x
def q (x : ℝ) : ℝ := 3 * x^2 - 4 * x - 5

theorem coefficient_of_x_squared :
  ∃ (a b c d e : ℝ), p x * q x = a * x^5 + b * x^4 + c * x^3 - 37 * x^2 + d * x + e :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l3128_312855


namespace NUMINAMATH_CALUDE_equation_solution_l3128_312895

theorem equation_solution : 
  ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3128_312895


namespace NUMINAMATH_CALUDE_fifteen_percent_of_800_is_120_l3128_312828

theorem fifteen_percent_of_800_is_120 :
  ∀ x : ℝ, (15 / 100) * x = 120 → x = 800 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_percent_of_800_is_120_l3128_312828


namespace NUMINAMATH_CALUDE_cubic_polynomial_sum_of_coefficients_l3128_312899

theorem cubic_polynomial_sum_of_coefficients 
  (A B C : ℝ) (v : ℂ) :
  let Q : ℂ → ℂ := λ z ↦ z^3 + A*z^2 + B*z + C
  (∀ z : ℂ, Q z = 0 ↔ z = v - 2*I ∨ z = v + 7*I ∨ z = 3*v + 5) →
  A + B + C = Q 1 - 1 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_sum_of_coefficients_l3128_312899


namespace NUMINAMATH_CALUDE_binary_addition_theorem_l3128_312809

/-- Converts a binary number (represented as a list of bits) to decimal -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to binary (represented as a list of bits) -/
def decimal_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
    let rec to_binary_aux (m : Nat) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
    to_binary_aux n

/-- The main theorem: 1010₂ + 10₂ = 1100₂ -/
theorem binary_addition_theorem : 
  decimal_to_binary (binary_to_decimal [false, true, false, true] + 
                     binary_to_decimal [false, true]) =
  [false, false, true, true] := by sorry

end NUMINAMATH_CALUDE_binary_addition_theorem_l3128_312809


namespace NUMINAMATH_CALUDE_regular_18gon_symmetry_sum_l3128_312824

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- Add any necessary fields here

/-- The number of lines of symmetry for a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := sorry

/-- The smallest positive angle of rotational symmetry in degrees for a regular polygon -/
def smallestRotationalSymmetryAngle (p : RegularPolygon n) : ℝ := sorry

theorem regular_18gon_symmetry_sum :
  ∀ (p : RegularPolygon 18),
    (linesOfSymmetry p : ℝ) + smallestRotationalSymmetryAngle p = 38 := by sorry

end NUMINAMATH_CALUDE_regular_18gon_symmetry_sum_l3128_312824


namespace NUMINAMATH_CALUDE_train_length_l3128_312885

/-- Given a train that crosses a tree in 100 seconds and takes 150 seconds to pass a platform 700 m long, prove that the length of the train is 1400 meters. -/
theorem train_length (tree_crossing_time platform_crossing_time platform_length : ℝ) 
  (h1 : tree_crossing_time = 100)
  (h2 : platform_crossing_time = 150)
  (h3 : platform_length = 700) : 
  ∃ train_length : ℝ, train_length = 1400 ∧ 
    train_length / tree_crossing_time = (train_length + platform_length) / platform_crossing_time :=
by
  sorry


end NUMINAMATH_CALUDE_train_length_l3128_312885


namespace NUMINAMATH_CALUDE_function_passes_through_point_l3128_312898

-- Define the function
def f (x : ℝ) : ℝ := 3 * x - 2

-- State the theorem
theorem function_passes_through_point :
  f (-1) = -5 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l3128_312898


namespace NUMINAMATH_CALUDE_external_angle_c_l3128_312866

theorem external_angle_c (A B C : ℝ) : 
  A = 40 → B = 2 * A → A + B + C = 180 → 180 - C = 120 := by sorry

end NUMINAMATH_CALUDE_external_angle_c_l3128_312866


namespace NUMINAMATH_CALUDE_distance_between_places_l3128_312882

/-- The distance between two places given speed changes and time differences --/
theorem distance_between_places (x : ℝ) (y : ℝ) : 
  ((x + 6) * (y - 5/60) = x * y) →
  ((x - 5) * (y + 6/60) = x * y) →
  x * y = 15 := by
sorry

end NUMINAMATH_CALUDE_distance_between_places_l3128_312882


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3128_312803

/-- Arithmetic sequence sum -/
def S (n : ℕ) : ℕ := n^2

/-- Theorem: For an arithmetic sequence with a_1 = 1 and d = 2,
    if S_{k+2} - S_k = 24, then k = 5 -/
theorem arithmetic_sequence_sum (k : ℕ) :
  S (k + 2) - S k = 24 → k = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3128_312803


namespace NUMINAMATH_CALUDE_axis_of_symmetry_y₂_greater_y₁_l3128_312817

/-- A parabola in the Cartesian coordinate system -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  m : ℝ
  t : ℝ
  y₁ : ℝ
  y₂ : ℝ
  h_a_pos : a > 0
  h_point : m = a * 2^2 + b * 2 + c
  h_axis : t = -b / (2 * a)
  h_y₁ : y₁ = a * (-1)^2 + b * (-1) + c
  h_y₂ : y₂ = a * 3^2 + b * 3 + c

/-- When m = c, the axis of symmetry is at x = 1 -/
theorem axis_of_symmetry (p : Parabola) (h : p.m = p.c) : p.t = 1 := by sorry

/-- When c < m, y₂ > y₁ -/
theorem y₂_greater_y₁ (p : Parabola) (h : p.c < p.m) : p.y₂ > p.y₁ := by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_y₂_greater_y₁_l3128_312817


namespace NUMINAMATH_CALUDE_base5_division_l3128_312874

/-- Converts a base 5 number to base 10 -/
def toBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a base 10 number to base 5 -/
def toBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

theorem base5_division (dividend : List Nat) (divisor : List Nat) :
  dividend = [2, 0, 1, 3] ∧ divisor = [3, 2] →
  toBase5 (toBase10 dividend / toBase10 divisor) = [0, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_base5_division_l3128_312874


namespace NUMINAMATH_CALUDE_log_difference_negative_l3128_312894

theorem log_difference_negative (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  Real.log (b - a) < 0 := by
  sorry

end NUMINAMATH_CALUDE_log_difference_negative_l3128_312894


namespace NUMINAMATH_CALUDE_logarithm_properties_l3128_312840

noncomputable def log (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem logarithm_properties (b : ℝ) (h1 : b > 0) (h2 : b ≠ 1) :
  (log b 1 = 0) ∧
  (log b b = 1) ∧
  (log b (1/b) = -1) ∧
  (∀ x : ℝ, 0 < x → x < 1 → log b x < 0) :=
by sorry

end NUMINAMATH_CALUDE_logarithm_properties_l3128_312840


namespace NUMINAMATH_CALUDE_point_not_in_third_quadrant_or_origin_l3128_312822

theorem point_not_in_third_quadrant_or_origin (n : ℝ) : 
  ¬(n ≤ 0 ∧ 1 - n ≤ 0) ∧ ¬(n = 0 ∧ 1 - n = 0) := by
  sorry

end NUMINAMATH_CALUDE_point_not_in_third_quadrant_or_origin_l3128_312822


namespace NUMINAMATH_CALUDE_remainder_of_sum_divided_by_eight_l3128_312853

theorem remainder_of_sum_divided_by_eight :
  (2356789 + 211) % 8 = 0 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_sum_divided_by_eight_l3128_312853


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l3128_312889

theorem triangle_third_side_length 
  (a b c : ℝ) 
  (angle_cos : ℝ) 
  (h1 : a = 4) 
  (h2 : b = 5) 
  (h3 : 2 * angle_cos^2 + 3 * angle_cos - 2 = 0) 
  (h4 : c^2 = a^2 + b^2 - 2*a*b*angle_cos) : 
  c = Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l3128_312889


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3128_312842

-- Define the complex number z
variable (z : ℂ)

-- Define the equation z ⋅ (1+2i)² = 3+4i
def equation (z : ℂ) : Prop := z * (1 + 2*Complex.I)^2 = 3 + 4*Complex.I

-- Theorem statement
theorem z_in_fourth_quadrant (h : equation z) : 
  0 < z.re ∧ z.im < 0 := by sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l3128_312842


namespace NUMINAMATH_CALUDE_m_less_than_one_necessary_l3128_312856

/-- A function f(x) = x + mx + m has a root. -/
def has_root (m : ℝ) : Prop :=
  ∃ x : ℝ, x + m * x + m = 0

/-- "m < 1" is a necessary condition for f(x) = x + mx + m to have a root. -/
theorem m_less_than_one_necessary (m : ℝ) :
  has_root m → m < 1 := by sorry

end NUMINAMATH_CALUDE_m_less_than_one_necessary_l3128_312856


namespace NUMINAMATH_CALUDE_min_m_plus_n_l3128_312886

theorem min_m_plus_n (m n : ℕ+) (h : 108 * m = n ^ 3) : 
  ∀ (m' n' : ℕ+), 108 * m' = n' ^ 3 → m + n ≤ m' + n' := by
  sorry

end NUMINAMATH_CALUDE_min_m_plus_n_l3128_312886


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l3128_312841

/-- The cubic polynomial q(x) that satisfies specific conditions -/
def q (x : ℚ) : ℚ := (51/13) * x^3 - (31/13) * x^2 + (16/13) * x + (3/13)

/-- Theorem stating that q(x) satisfies the given conditions -/
theorem q_satisfies_conditions :
  q 1 = 3 ∧ q 2 = 23 ∧ q 3 = 81 ∧ q 5 = 399 := by
  sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_l3128_312841


namespace NUMINAMATH_CALUDE_tax_reduction_theorem_l3128_312861

theorem tax_reduction_theorem (T C : ℝ) (x : ℝ) 
  (h1 : T > 0) (h2 : C > 0) 
  (h3 : (T * (1 - x / 100)) * (C * 1.1) = T * C * 0.88) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_tax_reduction_theorem_l3128_312861


namespace NUMINAMATH_CALUDE_smallest_n_multiple_of_3_and_3n_multiple_of_5_l3128_312867

theorem smallest_n_multiple_of_3_and_3n_multiple_of_5 :
  ∃ n : ℕ, n > 0 ∧ 3 ∣ n ∧ 5 ∣ (3 * n) ∧
  ∀ m : ℕ, m > 0 → 3 ∣ m → 5 ∣ (3 * m) → n ≤ m :=
by
  -- The proof goes here
  sorry

#check smallest_n_multiple_of_3_and_3n_multiple_of_5

end NUMINAMATH_CALUDE_smallest_n_multiple_of_3_and_3n_multiple_of_5_l3128_312867


namespace NUMINAMATH_CALUDE_base14_remainder_theorem_l3128_312897

-- Define a function to convert a base-14 integer to decimal
def base14ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (14 ^ i)) 0

-- Define our specific base-14 number
def ourNumber : List Nat := [1, 4, 6, 2]

-- Theorem statement
theorem base14_remainder_theorem :
  (base14ToDecimal ourNumber) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_base14_remainder_theorem_l3128_312897


namespace NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l3128_312820

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A convex polygon with 9 sides has 27 diagonals -/
theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 := by sorry

end NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l3128_312820


namespace NUMINAMATH_CALUDE_trig_identities_l3128_312857

theorem trig_identities (α : Real) (h : Real.tan α = 3) : 
  ((Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2) ∧
  (1 / (Real.sin α ^ 2 - Real.sin α * Real.cos α - 2 * Real.cos α ^ 2) = 2) := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l3128_312857


namespace NUMINAMATH_CALUDE_elberta_has_35_5_l3128_312872

/-- The amount of money Granny Smith has -/
def granny_smith_amount : ℚ := 81

/-- The amount of money Anjou has -/
def anjou_amount : ℚ := granny_smith_amount / 4

/-- The amount of money Elberta has -/
def elberta_amount : ℚ := 2 * anjou_amount - 5

/-- Theorem stating that Elberta has $35.5 -/
theorem elberta_has_35_5 : elberta_amount = 35.5 := by
  sorry

end NUMINAMATH_CALUDE_elberta_has_35_5_l3128_312872


namespace NUMINAMATH_CALUDE_sharp_four_times_100_l3128_312848

-- Define the # function
def sharp (N : ℝ) : ℝ := 0.7 * N + 5

-- State the theorem
theorem sharp_four_times_100 : sharp (sharp (sharp (sharp 100))) = 36.675 := by
  sorry

end NUMINAMATH_CALUDE_sharp_four_times_100_l3128_312848


namespace NUMINAMATH_CALUDE_cosine_angle_C_l3128_312844

/-- Given a triangle ABC with side lengths and angle relation, prove the cosine of angle C -/
theorem cosine_angle_C (A B C : ℝ) (BC AC : ℝ) (h1 : BC = 5) (h2 : AC = 4) 
  (h3 : Real.cos (A - B) = 7/8) : Real.cos C = 9/16 := by
  sorry

end NUMINAMATH_CALUDE_cosine_angle_C_l3128_312844


namespace NUMINAMATH_CALUDE_smallest_lucky_integer_l3128_312805

/-- An integer is lucky if there exist several consecutive integers, including itself, that add up to 2023. -/
def IsLucky (n : ℤ) : Prop :=
  ∃ k : ℕ, ∃ m : ℤ, (m + k : ℤ) = n ∧ (k + 1) * (2 * m + k) / 2 = 2023

/-- The smallest lucky integer -/
def SmallestLuckyInteger : ℤ := -2022

theorem smallest_lucky_integer :
  IsLucky SmallestLuckyInteger ∧
  ∀ n : ℤ, n < SmallestLuckyInteger → ¬IsLucky n :=
by sorry

end NUMINAMATH_CALUDE_smallest_lucky_integer_l3128_312805


namespace NUMINAMATH_CALUDE_age_difference_l3128_312858

/-- Given the age ratios and sum of ages, prove the difference between Patrick's and Monica's ages --/
theorem age_difference (patrick_age michael_age monica_age : ℕ) : 
  patrick_age * 5 = michael_age * 3 →
  michael_age * 5 = monica_age * 3 →
  patrick_age + michael_age + monica_age = 147 →
  monica_age - patrick_age = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3128_312858


namespace NUMINAMATH_CALUDE_largest_three_digit_number_with_conditions_l3128_312810

theorem largest_three_digit_number_with_conditions :
  ∃ n : ℕ,
    n = 960 ∧
    100 ≤ n ∧ n ≤ 999 ∧
    ∃ k : ℕ, n = 7 * k + 1 ∧
    ∃ m : ℕ, n = 8 * m + 4 ∧
    ∀ x : ℕ,
      (100 ≤ x ∧ x ≤ 999 ∧
       ∃ k' : ℕ, x = 7 * k' + 1 ∧
       ∃ m' : ℕ, x = 8 * m' + 4) →
      x ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_number_with_conditions_l3128_312810


namespace NUMINAMATH_CALUDE_complex_number_problem_l3128_312815

theorem complex_number_problem (a : ℝ) (z : ℂ) : 
  z = (Complex.I * (2 + a * Complex.I)) / (1 - Complex.I) →
  (∃ (b : ℝ), z = b * Complex.I) →
  z = 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3128_312815


namespace NUMINAMATH_CALUDE_point_B_coordinates_l3128_312814

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the problem statement
theorem point_B_coordinates :
  let A : Point2D := ⟨2, -3⟩
  let AB_length : ℝ := 4
  let B_parallel_to_x_axis : ℝ → Prop := λ y => y = A.y
  ∃ (B : Point2D), (B.x = -2 ∨ B.x = 6) ∧ 
                   B_parallel_to_x_axis B.y ∧ 
                   ((B.x - A.x)^2 + (B.y - A.y)^2 = AB_length^2) :=
by sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l3128_312814


namespace NUMINAMATH_CALUDE_triangle_problem_l3128_312862

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The conditions given in the problem -/
def satisfiesConditions (t : Triangle) : Prop :=
  8 * t.a * t.b * Real.sin t.C = 3 * (t.b^2 + t.c^2 - t.a^2) ∧
  t.a = Real.sqrt 10 ∧
  t.c = 5

/-- The theorem to be proved -/
theorem triangle_problem (t : Triangle) (h : satisfiesConditions t) :
  Real.cos t.A = 4/5 ∧
  (t.a * t.b * Real.sin t.C / 2 = 15/2 ∨ t.a * t.b * Real.sin t.C / 2 = 9/2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3128_312862


namespace NUMINAMATH_CALUDE_ribbon_distribution_l3128_312888

/-- Given total ribbon, number of gifts, and leftover ribbon, calculate ribbon per gift --/
def ribbon_per_gift (total_ribbon : ℕ) (num_gifts : ℕ) (leftover : ℕ) : ℚ :=
  (total_ribbon - leftover : ℚ) / num_gifts

theorem ribbon_distribution (total_ribbon num_gifts leftover : ℕ) 
  (h1 : total_ribbon = 18)
  (h2 : num_gifts = 6)
  (h3 : leftover = 6)
  (h4 : num_gifts > 0) :
  ribbon_per_gift total_ribbon num_gifts leftover = 2 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_distribution_l3128_312888


namespace NUMINAMATH_CALUDE_biker_journey_l3128_312860

/-- Proves that given a biker's journey conditions, the distance between towns is 140 km and initial speed is 28 km/hr -/
theorem biker_journey (total_distance : ℝ) (initial_speed : ℝ) : 
  (total_distance / 2 = initial_speed * 2.5) →
  (total_distance / 2 = (initial_speed + 2) * 2.333) →
  (total_distance = 140 ∧ initial_speed = 28) := by
  sorry

end NUMINAMATH_CALUDE_biker_journey_l3128_312860


namespace NUMINAMATH_CALUDE_median_is_39_l3128_312851

/-- Represents the score distribution of students --/
structure ScoreDistribution where
  scores : List Nat
  counts : List Nat
  total_students : Nat

/-- Calculates the median of a score distribution --/
def median (sd : ScoreDistribution) : Rat :=
  sorry

/-- The specific score distribution from the problem --/
def problem_distribution : ScoreDistribution :=
  { scores := [36, 37, 38, 39, 40],
    counts := [1, 2, 1, 4, 2],
    total_students := 10 }

/-- Theorem stating that the median of the given distribution is 39 --/
theorem median_is_39 : median problem_distribution = 39 := by
  sorry

end NUMINAMATH_CALUDE_median_is_39_l3128_312851


namespace NUMINAMATH_CALUDE_complex_power_four_l3128_312892

theorem complex_power_four : (1 + 2 * Complex.I) ^ 4 = -7 - 24 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_power_four_l3128_312892


namespace NUMINAMATH_CALUDE_ricky_magic_box_friday_l3128_312870

/-- Calculates the number of pennies in the magic money box after a given number of days -/
def pennies_after_days (initial_pennies : ℕ) (days : ℕ) : ℕ :=
  initial_pennies * 2^days

/-- Theorem: Ricky's magic money box contains 48 pennies on Friday -/
theorem ricky_magic_box_friday : pennies_after_days 3 4 = 48 := by
  sorry

end NUMINAMATH_CALUDE_ricky_magic_box_friday_l3128_312870


namespace NUMINAMATH_CALUDE_product_from_hcf_lcm_l3128_312836

theorem product_from_hcf_lcm (a b : ℕ+) : 
  Nat.gcd a b = 16 → Nat.lcm a b = 160 → a * b = 2560 := by
  sorry

end NUMINAMATH_CALUDE_product_from_hcf_lcm_l3128_312836


namespace NUMINAMATH_CALUDE_intersection_M_N_l3128_312863

-- Define set M
def M : Set ℝ := {y | ∃ x : ℝ, y = x - |x|}

-- Define set N
def N : Set ℝ := {y | ∃ x : ℝ, y = Real.sqrt x}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3128_312863


namespace NUMINAMATH_CALUDE_nap_hours_in_70_days_l3128_312829

/-- Calculates the total hours of naps taken in a given number of days -/
def total_nap_hours (days : ℕ) (naps_per_week : ℕ) (hours_per_nap : ℕ) : ℕ :=
  let weeks : ℕ := days / 7
  let total_naps : ℕ := weeks * naps_per_week
  total_naps * hours_per_nap

/-- Theorem stating that 70 days of naps results in 60 hours of nap time -/
theorem nap_hours_in_70_days :
  total_nap_hours 70 3 2 = 60 := by
  sorry

#eval total_nap_hours 70 3 2

end NUMINAMATH_CALUDE_nap_hours_in_70_days_l3128_312829


namespace NUMINAMATH_CALUDE_certain_number_problem_l3128_312896

theorem certain_number_problem (x : ℝ) (n : ℝ) : 
  (9 - 4 / x = 7 + n / x) → (x = 6) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3128_312896


namespace NUMINAMATH_CALUDE_prob_king_or_queen_in_special_deck_l3128_312865

structure Deck :=
  (total_cards : ℕ)
  (num_ranks : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)

def probability_king_or_queen (d : Deck) : ℚ :=
  let kings_and_queens := d.num_suits * 2
  kings_and_queens / d.total_cards

theorem prob_king_or_queen_in_special_deck :
  let d : Deck := {
    total_cards := 60,
    num_ranks := 15,
    num_suits := 4,
    cards_per_suit := 15
  }
  probability_king_or_queen d = 2 / 15 := by sorry

end NUMINAMATH_CALUDE_prob_king_or_queen_in_special_deck_l3128_312865


namespace NUMINAMATH_CALUDE_complex_number_coordinates_l3128_312800

theorem complex_number_coordinates : 
  let i : ℂ := Complex.I
  let z : ℂ := i * (1 + i)
  z = -1 + i := by sorry

end NUMINAMATH_CALUDE_complex_number_coordinates_l3128_312800


namespace NUMINAMATH_CALUDE_blue_jellybean_probability_blue_jellybean_probability_is_two_nineteenths_l3128_312869

/-- The probability of drawing 3 blue jellybeans in a row from a bag containing 10 red and 10 blue jellybeans, without replacement. -/
theorem blue_jellybean_probability : ℚ :=
  let total_jellybeans : ℕ := 20
  let blue_jellybeans : ℕ := 10
  let draws : ℕ := 3

  let prob_first : ℚ := blue_jellybeans / total_jellybeans
  let prob_second : ℚ := (blue_jellybeans - 1) / (total_jellybeans - 1)
  let prob_third : ℚ := (blue_jellybeans - 2) / (total_jellybeans - 2)

  prob_first * prob_second * prob_third

/-- Proof that the probability of drawing 3 blue jellybeans in a row is 2/19. -/
theorem blue_jellybean_probability_is_two_nineteenths :
  blue_jellybean_probability = 2 / 19 := by
  sorry

end NUMINAMATH_CALUDE_blue_jellybean_probability_blue_jellybean_probability_is_two_nineteenths_l3128_312869


namespace NUMINAMATH_CALUDE_value_of_lg_ta_ratio_l3128_312880

-- Define the necessary functions
noncomputable def sn (x : ℝ) : ℝ := Real.sin x
noncomputable def si (x : ℝ) : ℝ := Real.sin x
noncomputable def ta (x : ℝ) : ℝ := Real.tan x
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem value_of_lg_ta_ratio (α β : ℝ) 
  (h1 : sn (α + β) = 1/2) 
  (h2 : si (α - β) = 1/3) : 
  lg (5 * (ta α / ta β)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_lg_ta_ratio_l3128_312880


namespace NUMINAMATH_CALUDE_gcd_of_4536_13440_216_l3128_312806

theorem gcd_of_4536_13440_216 : Nat.gcd 4536 (Nat.gcd 13440 216) = 216 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_4536_13440_216_l3128_312806


namespace NUMINAMATH_CALUDE_parabola_properties_l3128_312816

/-- The quadratic function f(x) = x^2 - 8x + 12 -/
def f (x : ℝ) : ℝ := x^2 - 8*x + 12

/-- The x-coordinate of the vertex of f -/
def vertex_x : ℝ := 4

/-- The y-coordinate of the vertex of f -/
def vertex_y : ℝ := -4

theorem parabola_properties :
  (∀ x : ℝ, f x ≥ f vertex_x) ∧ 
  f vertex_x = vertex_y ∧
  f 3 = -3 := by sorry

end NUMINAMATH_CALUDE_parabola_properties_l3128_312816


namespace NUMINAMATH_CALUDE_rug_overlap_problem_l3128_312837

theorem rug_overlap_problem (total_area : ℝ) (covered_area : ℝ) (two_layer_area : ℝ)
  (h1 : total_area = 350)
  (h2 : covered_area = 250)
  (h3 : two_layer_area = 45) :
  total_area = covered_area + two_layer_area + 55 :=
by sorry

end NUMINAMATH_CALUDE_rug_overlap_problem_l3128_312837


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l3128_312893

theorem tan_alpha_plus_pi_fourth (α β : Real) 
  (h1 : Real.tan (α + β) = 3/5)
  (h2 : Real.tan (β - π/4) = 1/4) :
  Real.tan (α + π/4) = 7/23 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l3128_312893


namespace NUMINAMATH_CALUDE_total_instruments_eq_113_l3128_312823

/-- The total number of musical instruments owned by Charlie, Carli, Nick, and Daisy -/
def total_instruments (charlie_flutes charlie_horns charlie_harps charlie_drums : ℕ)
  (carli_flute_ratio carli_horn_ratio carli_drum_ratio : ℕ)
  (nick_flute_offset nick_horn_offset nick_drum_ratio nick_drum_offset : ℕ)
  (daisy_horn_denominator : ℕ) : ℕ :=
  let carli_flutes := charlie_flutes * carli_flute_ratio
  let carli_horns := charlie_horns / carli_horn_ratio
  let carli_drums := charlie_drums * carli_drum_ratio

  let nick_flutes := carli_flutes * 2 - nick_flute_offset
  let nick_horns := charlie_horns + carli_horns
  let nick_drums := carli_drums * nick_drum_ratio - nick_drum_offset

  let daisy_flutes := nick_flutes ^ 2
  let daisy_horns := (nick_horns - carli_horns) / daisy_horn_denominator
  let daisy_harps := charlie_harps
  let daisy_drums := (charlie_drums + carli_drums + nick_drums) / 3

  charlie_flutes + charlie_horns + charlie_harps + charlie_drums +
  carli_flutes + carli_horns + carli_drums +
  nick_flutes + nick_horns + nick_drums +
  daisy_flutes + daisy_horns + daisy_harps + daisy_drums

theorem total_instruments_eq_113 :
  total_instruments 1 2 1 5 3 2 2 1 0 4 2 2 = 113 := by
  sorry

end NUMINAMATH_CALUDE_total_instruments_eq_113_l3128_312823


namespace NUMINAMATH_CALUDE_intersection_with_complement_l3128_312825

def U : Set Nat := {1, 2, 4, 6, 8}
def A : Set Nat := {1, 2, 4}
def B : Set Nat := {2, 4, 6}

theorem intersection_with_complement : A ∩ (U \ B) = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l3128_312825


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3128_312843

theorem sufficient_not_necessary_condition (a b c : ℝ) :
  (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b) ∧
  (∃ a b c : ℝ, a > b ∧ a * c^2 ≤ b * c^2) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3128_312843


namespace NUMINAMATH_CALUDE_unique_solution_l3128_312873

-- Define the system of equations
def system (x y z : ℝ) : Prop :=
  x^3 - 3*x = 4 - y ∧
  2*y^3 - 6*y = 6 - z ∧
  3*z^3 - 9*z = 8 - x

-- Theorem statement
theorem unique_solution :
  ∀ x y z : ℝ, system x y z → (x = 2 ∧ y = 2 ∧ z = 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l3128_312873


namespace NUMINAMATH_CALUDE_intersection_distance_l3128_312804

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  Real.sqrt 3 * x - y + 3 = 0

-- Define the curve C
def curve_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*y - 2*Real.sqrt 3*x = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line_l A.1 A.2 ∧ curve_C A.1 A.2 ∧
  line_l B.1 B.2 ∧ curve_C B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem intersection_distance :
  ∀ A B : ℝ × ℝ, intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 15 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l3128_312804


namespace NUMINAMATH_CALUDE_cousins_age_sum_l3128_312887

theorem cousins_age_sum (ages : Fin 5 → ℕ) 
  (mean_condition : (ages 0 + ages 1 + ages 2 + ages 3 + ages 4) / 5 = 10)
  (median_condition : ages 2 = 7)
  (sorted : ∀ i j, i ≤ j → ages i ≤ ages j) :
  ages 0 + ages 4 = 29 := by
sorry

end NUMINAMATH_CALUDE_cousins_age_sum_l3128_312887


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l3128_312821

/-- A function that returns the digits of a three-digit number -/
def digits (n : ℕ) : Fin 3 → ℕ :=
  fun i => (n / (100 / 10^i.val)) % 10

/-- Check if three numbers form a geometric progression -/
def isGeometricProgression (a b c : ℕ) : Prop :=
  b * b = a * c

/-- Check if three numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℕ) : Prop :=
  2 * b = a + c

/-- The main theorem -/
theorem unique_three_digit_number : ∃! n : ℕ,
  100 ≤ n ∧ n < 1000 ∧
  (let d := digits n
   isGeometricProgression (d 0) (d 1) (d 2) ∧
   d 0 ≠ d 1 ∧ d 1 ≠ d 2 ∧ d 0 ≠ d 2) ∧
  (let m := n - 200
   100 ≤ m ∧ m < 1000 ∧
   let d := digits m
   isArithmeticProgression (d 0) (d 1) (d 2)) ∧
  n = 842 :=
sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l3128_312821


namespace NUMINAMATH_CALUDE_bianca_points_l3128_312850

/-- Calculates the points earned for recycling cans given the total number of bags, 
    number of bags not recycled, and points per bag. -/
def points_earned (total_bags : ℕ) (bags_not_recycled : ℕ) (points_per_bag : ℕ) : ℕ :=
  (total_bags - bags_not_recycled) * points_per_bag

/-- Proves that Bianca earned 45 points for recycling cans. -/
theorem bianca_points : points_earned 17 8 5 = 45 := by
  sorry

end NUMINAMATH_CALUDE_bianca_points_l3128_312850


namespace NUMINAMATH_CALUDE_wendy_packaging_chocolates_l3128_312808

/-- The number of chocolates Wendy can package in one hour -/
def chocolates_per_hour : ℕ := 1152 / 4

/-- The number of chocolates Wendy can package in h hours -/
def chocolates_in_h_hours (h : ℕ) : ℕ := chocolates_per_hour * h

theorem wendy_packaging_chocolates (h : ℕ) : 
  chocolates_in_h_hours h = 288 * h := by
  sorry

#check wendy_packaging_chocolates

end NUMINAMATH_CALUDE_wendy_packaging_chocolates_l3128_312808


namespace NUMINAMATH_CALUDE_adrianna_gum_purchase_l3128_312884

/-- Calculates the number of gum pieces bought at the store -/
def gum_bought_at_store (initial_gum : ℕ) (friends_given : ℕ) (gum_left : ℕ) : ℕ :=
  friends_given + gum_left - initial_gum

/-- Theorem: Given the initial conditions, prove that Adrianna bought 3 pieces of gum at the store -/
theorem adrianna_gum_purchase :
  gum_bought_at_store 10 11 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_adrianna_gum_purchase_l3128_312884


namespace NUMINAMATH_CALUDE_coffee_lasts_13_days_l3128_312864

def coffee_problem (coffee_weight : ℕ) (cups_per_pound : ℕ) 
  (angie_cups : ℕ) (bob_cups : ℕ) (carol_cups : ℕ) : ℕ :=
  let total_cups := coffee_weight * cups_per_pound
  let daily_consumption := angie_cups + bob_cups + carol_cups
  total_cups / daily_consumption

theorem coffee_lasts_13_days :
  coffee_problem 3 40 3 2 4 = 13 := by
  sorry

end NUMINAMATH_CALUDE_coffee_lasts_13_days_l3128_312864


namespace NUMINAMATH_CALUDE_corn_growth_first_week_l3128_312831

/-- Represents the growth of corn over three weeks -/
structure CornGrowth where
  week1 : ℝ
  week2 : ℝ
  week3 : ℝ

/-- The conditions of corn growth as described in the problem -/
def valid_growth (g : CornGrowth) : Prop :=
  g.week2 = 2 * g.week1 ∧
  g.week3 = 4 * g.week2 ∧
  g.week1 + g.week2 + g.week3 = 22

/-- The theorem stating that the corn grew 2 inches in the first week -/
theorem corn_growth_first_week :
  ∀ g : CornGrowth, valid_growth g → g.week1 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_corn_growth_first_week_l3128_312831


namespace NUMINAMATH_CALUDE_mary_max_earnings_l3128_312868

/-- Calculates the maximum weekly earnings for a worker with the given conditions -/
def max_weekly_earnings (max_hours : ℕ) (regular_hours : ℕ) (regular_rate : ℚ) (overtime_rate_increase : ℚ) : ℚ :=
  let overtime_hours := max_hours - regular_hours
  let overtime_rate := regular_rate * (1 + overtime_rate_increase)
  regular_hours * regular_rate + overtime_hours * overtime_rate

/-- Mary's maximum weekly earnings under the given conditions -/
theorem mary_max_earnings :
  max_weekly_earnings 60 20 8 (1/4) = 560 := by
  sorry

#eval max_weekly_earnings 60 20 8 (1/4)

end NUMINAMATH_CALUDE_mary_max_earnings_l3128_312868


namespace NUMINAMATH_CALUDE_substring_012_occurrences_l3128_312832

/-- Base-3 representation of an integer without leading zeroes -/
def base3Repr (n : ℕ) : List ℕ := sorry

/-- Continuous string formed by joining base-3 representations of integers from 1 to 729 -/
def continuousString : List ℕ := sorry

/-- Count occurrences of a substring in a list -/
def countSubstring (list : List ℕ) (substring : List ℕ) : ℕ := sorry

theorem substring_012_occurrences :
  countSubstring continuousString [0, 1, 2] = 148 := by sorry

end NUMINAMATH_CALUDE_substring_012_occurrences_l3128_312832


namespace NUMINAMATH_CALUDE_range_of_g_l3128_312849

theorem range_of_g : ∀ x : ℝ, 
  (3/4 : ℝ) ≤ (Real.cos x)^4 + (Real.sin x)^2 ∧ 
  (Real.cos x)^4 + (Real.sin x)^2 ≤ 1 ∧
  ∃ y z : ℝ, (Real.cos y)^4 + (Real.sin y)^2 = (3/4 : ℝ) ∧
             (Real.cos z)^4 + (Real.sin z)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_g_l3128_312849


namespace NUMINAMATH_CALUDE_converse_propositions_l3128_312818

-- Define the basic concepts
def Point : Type := ℝ × ℝ × ℝ
def Line : Type := Point → Prop

-- Define the relationships
def coplanar (a b c d : Point) : Prop := sorry
def collinear (a b c : Point) : Prop := sorry
def have_common_point (l₁ l₂ : Line) : Prop := sorry
def skew_lines (l₁ l₂ : Line) : Prop := sorry

-- State the theorem
theorem converse_propositions :
  (∀ a b c d : Point, (¬collinear a b c ∧ ¬collinear a b d ∧ ¬collinear a c d ∧ ¬collinear b c d) → ¬coplanar a b c d) = false ∧
  (∀ l₁ l₂ : Line, skew_lines l₁ l₂ → ¬have_common_point l₁ l₂) = true :=
sorry

end NUMINAMATH_CALUDE_converse_propositions_l3128_312818


namespace NUMINAMATH_CALUDE_ages_sum_after_three_years_l3128_312830

theorem ages_sum_after_three_years 
  (ava_age bob_age carlo_age : ℕ) 
  (h : ava_age + bob_age + carlo_age = 31) : 
  (ava_age + 3) + (bob_age + 3) + (carlo_age + 3) = 40 := by
  sorry

end NUMINAMATH_CALUDE_ages_sum_after_three_years_l3128_312830


namespace NUMINAMATH_CALUDE_larger_cuboid_width_l3128_312827

theorem larger_cuboid_width
  (small_length small_width small_height : ℝ)
  (large_length large_height : ℝ)
  (num_small_cuboids : ℕ)
  (h1 : small_length = 5)
  (h2 : small_width = 4)
  (h3 : small_height = 3)
  (h4 : large_length = 16)
  (h5 : large_height = 12)
  (h6 : num_small_cuboids = 32)
  (h7 : small_length * small_width * small_height * num_small_cuboids = large_length * large_height * (large_length * large_height / (small_length * small_width * small_height * num_small_cuboids))) :
  large_length * large_height / (small_length * small_width * small_height * num_small_cuboids) = 10 := by
sorry

end NUMINAMATH_CALUDE_larger_cuboid_width_l3128_312827


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3128_312845

theorem solution_set_inequality (x : ℝ) (h : x ≠ 0) :
  (x + 1) / x ≤ 3 ↔ x ∈ Set.Iio 0 ∪ Set.Ici (1/2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3128_312845


namespace NUMINAMATH_CALUDE_problem_statement_l3128_312891

noncomputable def f₁ (a x : ℝ) : ℝ := Real.exp (abs (x - 2*a + 1))
noncomputable def f₂ (a x : ℝ) : ℝ := Real.exp (abs (x - a) + 1)
noncomputable def f (a x : ℝ) : ℝ := f₁ a x + f₂ a x
noncomputable def g (a x : ℝ) : ℝ := (f₁ a x + f₂ a x) / 2 - abs (f₁ a x - f₂ a x) / 2

theorem problem_statement :
  (∀ x ∈ Set.Icc 2 3, f 2 x ≥ 2 * Real.exp 1) ∧
  (∃ x ∈ Set.Icc 2 3, f 2 x = 2 * Real.exp 1) ∧
  (∀ a, (∀ x ≥ a, f₂ a x ≥ f₁ a x) ↔ 0 ≤ a ∧ a ≤ 2) ∧
  (∀ x ∈ Set.Icc 1 6, 
    g a x ≥ 
      (if 1 ≤ a ∧ a ≤ 7/2 then 1
      else if -2 ≤ a ∧ a ≤ 0 then Real.exp (2 - a)
      else if a < -2 ∨ (0 < a ∧ a < 1) then Real.exp (3 - 2*a)
      else if 7/2 < a ∧ a ≤ 6 then Real.exp 1
      else Real.exp (a - 5))) ∧
  (∃ x ∈ Set.Icc 1 6, 
    g a x = 
      (if 1 ≤ a ∧ a ≤ 7/2 then 1
      else if -2 ≤ a ∧ a ≤ 0 then Real.exp (2 - a)
      else if a < -2 ∨ (0 < a ∧ a < 1) then Real.exp (3 - 2*a)
      else if 7/2 < a ∧ a ≤ 6 then Real.exp 1
      else Real.exp (a - 5))) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3128_312891


namespace NUMINAMATH_CALUDE_infinite_inscribed_rectangles_l3128_312819

/-- A point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A rectangle defined by its four vertices -/
structure Rectangle :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Predicate to check if a point lies on a side of a rectangle -/
def PointOnSide (P : Point) (R : Rectangle) : Prop :=
  (P.x = R.A.x ∧ R.A.y ≤ P.y ∧ P.y ≤ R.B.y) ∨
  (P.y = R.B.y ∧ R.B.x ≤ P.x ∧ P.x ≤ R.C.x) ∨
  (P.x = R.C.x ∧ R.C.y ≥ P.y ∧ P.y ≥ R.D.y) ∨
  (P.y = R.D.y ∧ R.D.x ≥ P.x ∧ P.x ≥ R.A.x)

/-- Predicate to check if four points form a rectangle -/
def IsRectangle (E F G H : Point) : Prop :=
  (E.x - F.x) * (G.x - H.x) + (E.y - F.y) * (G.y - H.y) = 0 ∧
  (E.x - H.x) * (F.x - G.x) + (E.y - H.y) * (F.y - G.y) = 0

theorem infinite_inscribed_rectangles (ABCD : Rectangle) :
  ∃ (S : Set (Point × Point × Point × Point)),
    (∀ (E F G H : Point), (E, F, G, H) ∈ S →
      PointOnSide E ABCD ∧ PointOnSide F ABCD ∧
      PointOnSide G ABCD ∧ PointOnSide H ABCD ∧
      IsRectangle E F G H) ∧
    Set.Infinite S :=
  sorry

end NUMINAMATH_CALUDE_infinite_inscribed_rectangles_l3128_312819


namespace NUMINAMATH_CALUDE_selling_price_calculation_l3128_312826

theorem selling_price_calculation (cost_price : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) :
  cost_price = 540 →
  markup_percentage = 15 →
  discount_percentage = 26.570048309178745 →
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let discount_amount := marked_price * (discount_percentage / 100)
  let selling_price := marked_price - discount_amount
  selling_price = 456 := by
sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l3128_312826


namespace NUMINAMATH_CALUDE_martha_apples_problem_l3128_312875

/-- Given that Martha has 20 apples initially, gives 5 to Jane and 2 more than that to James,
    prove that she needs to give away 4 more apples to be left with exactly 4 apples. -/
theorem martha_apples_problem (initial_apples : ℕ) (jane_apples : ℕ) (james_extra_apples : ℕ) 
  (h1 : initial_apples = 20)
  (h2 : jane_apples = 5)
  (h3 : james_extra_apples = 2) :
  initial_apples - jane_apples - (jane_apples + james_extra_apples) - 4 = 4 := by
  sorry

#check martha_apples_problem

end NUMINAMATH_CALUDE_martha_apples_problem_l3128_312875


namespace NUMINAMATH_CALUDE_odd_binomial_coefficients_count_l3128_312877

theorem odd_binomial_coefficients_count (n : ℕ) : 
  (Finset.sum (Finset.range (2^n)) (λ u => 
    (Finset.sum (Finset.range (u+1)) (λ v => 
      if Nat.choose u v % 2 = 1 then 1 else 0
    ))
  )) = 3^n := by sorry

end NUMINAMATH_CALUDE_odd_binomial_coefficients_count_l3128_312877


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l3128_312890

theorem adult_ticket_cost (student_ticket_cost : ℝ) (total_tickets : ℕ) (total_revenue : ℝ) (adult_tickets : ℕ) (student_tickets : ℕ) :
  student_ticket_cost = 3 →
  total_tickets = 846 →
  total_revenue = 3846 →
  adult_tickets = 410 →
  student_tickets = 436 →
  ∃ (adult_ticket_cost : ℝ), adult_ticket_cost = 6.19 ∧
    adult_ticket_cost * adult_tickets + student_ticket_cost * student_tickets = total_revenue :=
by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l3128_312890


namespace NUMINAMATH_CALUDE_stirring_ensures_representativeness_l3128_312839

/-- Represents the lottery method for sampling -/
structure LotteryMethod where
  /-- The action of stirring the lots -/
  stir : Bool

/-- Represents the representativeness of a sample -/
def representative (method : LotteryMethod) : Prop :=
  method.stir

/-- Theorem stating that stirring evenly is key to representativeness in the lottery method -/
theorem stirring_ensures_representativeness (method : LotteryMethod) :
  representative method ↔ method.stir :=
sorry

end NUMINAMATH_CALUDE_stirring_ensures_representativeness_l3128_312839


namespace NUMINAMATH_CALUDE_mass_CaSO4_formed_l3128_312846

-- Define the molar masses
def molar_mass_Ca : ℝ := 40.08
def molar_mass_S : ℝ := 32.06
def molar_mass_O : ℝ := 16.00

-- Define the molar mass of CaSO₄
def molar_mass_CaSO4 : ℝ := molar_mass_Ca + molar_mass_S + 4 * molar_mass_O

-- Define the number of moles of Ca(OH)₂
def moles_CaOH2 : ℝ := 12

-- Theorem statement
theorem mass_CaSO4_formed (excess_H2SO4 : Prop) (neutralization_reaction : Prop) :
  moles_CaOH2 * molar_mass_CaSO4 = 1633.68 := by
  sorry


end NUMINAMATH_CALUDE_mass_CaSO4_formed_l3128_312846


namespace NUMINAMATH_CALUDE_first_two_satisfying_numbers_l3128_312876

def satisfiesConditions (n : ℕ) : Prop :=
  n % 7 = 3 ∧ n % 9 = 4

theorem first_two_satisfying_numbers :
  ∃ (a b : ℕ), a < b ∧
  satisfiesConditions a ∧
  satisfiesConditions b ∧
  (∀ (x : ℕ), x < a → ¬satisfiesConditions x) ∧
  (∀ (x : ℕ), a < x → x < b → ¬satisfiesConditions x) ∧
  a = 31 ∧ b = 94 := by
  sorry

end NUMINAMATH_CALUDE_first_two_satisfying_numbers_l3128_312876


namespace NUMINAMATH_CALUDE_complete_square_for_given_equation_l3128_312883

/-- Represents a quadratic equation of the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the result of completing the square for a quadratic equation -/
structure CompletedSquareForm where
  r : ℝ
  s : ℝ

/-- Completes the square for a given quadratic equation -/
def completeSquare (eq : QuadraticEquation) : CompletedSquareForm :=
  sorry

theorem complete_square_for_given_equation :
  let eq := QuadraticEquation.mk 9 (-18) (-720)
  let result := completeSquare eq
  result.s = 81 := by sorry

end NUMINAMATH_CALUDE_complete_square_for_given_equation_l3128_312883


namespace NUMINAMATH_CALUDE_janet_ticket_problem_l3128_312834

/-- The number of tickets needed for one ride on the roller coaster -/
def roller_coaster_tickets : ℕ := 5

/-- The total number of tickets needed for 7 rides on the roller coaster and 4 rides on the giant slide -/
def total_tickets : ℕ := 47

/-- The number of roller coaster rides -/
def roller_coaster_rides : ℕ := 7

/-- The number of giant slide rides -/
def giant_slide_rides : ℕ := 4

/-- The number of tickets needed for one ride on the giant slide -/
def giant_slide_tickets : ℕ := 3

theorem janet_ticket_problem :
  roller_coaster_tickets * roller_coaster_rides + giant_slide_tickets * giant_slide_rides = total_tickets :=
sorry

end NUMINAMATH_CALUDE_janet_ticket_problem_l3128_312834


namespace NUMINAMATH_CALUDE_second_discount_percentage_l3128_312878

theorem second_discount_percentage (initial_price : ℝ) (first_discount : ℝ) (final_price : ℝ) :
  initial_price = 600 →
  first_discount = 10 →
  final_price = 513 →
  ∃ (second_discount : ℝ),
    final_price = initial_price * (1 - first_discount / 100) * (1 - second_discount / 100) ∧
    second_discount = 5 := by
  sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l3128_312878


namespace NUMINAMATH_CALUDE_delivery_ratio_l3128_312838

theorem delivery_ratio : 
  let meals : ℕ := 3
  let total : ℕ := 27
  let packages : ℕ := total - meals
  packages / meals = 8 := by
sorry

end NUMINAMATH_CALUDE_delivery_ratio_l3128_312838


namespace NUMINAMATH_CALUDE_total_white_pieces_l3128_312833

/-- The total number of pieces -/
def total_pieces : ℕ := 300

/-- The number of piles -/
def num_piles : ℕ := 100

/-- The number of pieces in each pile -/
def pieces_per_pile : ℕ := 3

/-- The number of piles with exactly one white piece -/
def piles_with_one_white : ℕ := 27

/-- The number of piles with 2 or 3 black pieces -/
def piles_with_two_or_three_black : ℕ := 42

theorem total_white_pieces :
  ∃ (piles_with_three_white : ℕ) 
    (piles_with_two_white : ℕ)
    (total_white : ℕ),
  piles_with_three_white = num_piles - piles_with_one_white - piles_with_two_or_three_black + piles_with_one_white ∧
  piles_with_two_white = num_piles - piles_with_one_white - 2 * piles_with_three_white ∧
  total_white = piles_with_one_white * 1 + piles_with_three_white * 3 + piles_with_two_white * 2 ∧
  total_white = 158 :=
by sorry

end NUMINAMATH_CALUDE_total_white_pieces_l3128_312833


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l3128_312881

/-- Given a circle with circumference 87.98229536926875 cm, its area is approximately 615.75164 square centimeters. -/
theorem circle_area_from_circumference : 
  let circumference : ℝ := 87.98229536926875
  let radius : ℝ := circumference / (2 * Real.pi)
  let area : ℝ := Real.pi * radius ^ 2
  ∃ ε > 0, abs (area - 615.75164) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l3128_312881


namespace NUMINAMATH_CALUDE_lcm_18_27_l3128_312871

theorem lcm_18_27 : Nat.lcm 18 27 = 54 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_27_l3128_312871


namespace NUMINAMATH_CALUDE_sum_of_polynomials_l3128_312801

-- Define the polynomials
def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

-- State the theorem
theorem sum_of_polynomials :
  ∀ x : ℝ, p x + q x + r x = -4 * x^2 + 12 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_polynomials_l3128_312801


namespace NUMINAMATH_CALUDE_diane_gingerbreads_l3128_312812

/-- The number of trays with 25 gingerbreads each -/
def trays_25 : ℕ := 4

/-- The number of gingerbreads in each of the first type of tray -/
def gingerbreads_per_tray_25 : ℕ := 25

/-- The number of trays with 20 gingerbreads each -/
def trays_20 : ℕ := 3

/-- The number of gingerbreads in each of the second type of tray -/
def gingerbreads_per_tray_20 : ℕ := 20

/-- The total number of gingerbreads Diane bakes -/
def total_gingerbreads : ℕ := trays_25 * gingerbreads_per_tray_25 + trays_20 * gingerbreads_per_tray_20

theorem diane_gingerbreads : total_gingerbreads = 160 := by
  sorry

end NUMINAMATH_CALUDE_diane_gingerbreads_l3128_312812


namespace NUMINAMATH_CALUDE_range_of_f_l3128_312847

def f (x : ℝ) : ℝ := |x - 3| - |x + 4|

theorem range_of_f :
  ∀ y ∈ Set.range f, -7 ≤ y ∧ y ≤ 7 ∧
  ∀ z, -7 ≤ z ∧ z ≤ 7 → ∃ x, f x = z :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l3128_312847


namespace NUMINAMATH_CALUDE_tv_selection_problem_l3128_312859

theorem tv_selection_problem (type_a : ℕ) (type_b : ℕ) (total_select : ℕ) : 
  type_a = 4 → type_b = 5 → total_select = 3 →
  (Nat.choose type_a 1 * Nat.choose type_b 2 + Nat.choose type_a 2 * Nat.choose type_b 1) = 70 :=
by sorry

end NUMINAMATH_CALUDE_tv_selection_problem_l3128_312859


namespace NUMINAMATH_CALUDE_fencing_length_l3128_312811

theorem fencing_length (area : ℝ) (uncovered_side : ℝ) (fencing_length : ℝ) : 
  area = 600 →
  uncovered_side = 20 →
  fencing_length = uncovered_side + 2 * (area / uncovered_side) →
  fencing_length = 80 := by
sorry

end NUMINAMATH_CALUDE_fencing_length_l3128_312811


namespace NUMINAMATH_CALUDE_diophantine_equation_solvable_l3128_312835

theorem diophantine_equation_solvable (a : ℕ+) :
  ∃ (x y : ℤ), x^2 - y^2 = (a : ℤ)^3 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solvable_l3128_312835
