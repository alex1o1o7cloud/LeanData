import Mathlib

namespace NUMINAMATH_CALUDE_base7_243_to_base10_l1640_164009

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (d2 d1 d0 : ℕ) : ℕ :=
  d0 + d1 * 7 + d2 * 7^2

/-- The base 10 equivalent of 243 in base 7 --/
theorem base7_243_to_base10 : base7ToBase10 2 4 3 = 129 := by
  sorry

end NUMINAMATH_CALUDE_base7_243_to_base10_l1640_164009


namespace NUMINAMATH_CALUDE_k_value_l1640_164080

theorem k_value (θ : Real) (k : Real) 
  (h1 : k = (3 * Real.sin θ + 5 * Real.cos θ) / (2 * Real.sin θ + Real.cos θ))
  (h2 : Real.tan θ = 3) : k = 2 := by
  sorry

end NUMINAMATH_CALUDE_k_value_l1640_164080


namespace NUMINAMATH_CALUDE_inverse_of_exponential_l1640_164079

noncomputable def g (x : ℝ) : ℝ := 3^x

theorem inverse_of_exponential (f : ℝ → ℝ) :
  (∀ x, f (g x) = x ∧ g (f x) = x) → f 3 = 1 := by sorry

end NUMINAMATH_CALUDE_inverse_of_exponential_l1640_164079


namespace NUMINAMATH_CALUDE_fractional_part_An_bounds_l1640_164035

theorem fractional_part_An_bounds (n : ℕ+) :
  let An := (49 * n.val ^ 2 + 0.35 * n.val : ℝ).sqrt
  0.024 < An - ⌊An⌋ ∧ An - ⌊An⌋ < 0.025 := by
  sorry

end NUMINAMATH_CALUDE_fractional_part_An_bounds_l1640_164035


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1640_164004

/-- The sum of the repeating decimals 0.666... and 0.333... is equal to 1. -/
theorem sum_of_repeating_decimals : 
  (∃ (x y : ℚ), (10 * x - x = 6 ∧ 10 * y - y = 3) → x + y = 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1640_164004


namespace NUMINAMATH_CALUDE_man_walking_rest_distance_l1640_164057

/-- Proves that a man walking at 10 mph, resting for 8 minutes after every d miles,
    and taking 332 minutes to walk 50 miles, rests after every 10 miles. -/
theorem man_walking_rest_distance (d : ℝ) : 
  (10 : ℝ) = d → -- walking speed in mph
  (8 : ℝ) = 8 → -- rest duration in minutes
  (332 : ℝ) = 332 → -- total time in minutes
  (50 : ℝ) = 50 → -- total distance in miles
  (300 : ℝ) + (50 / d - 1) * 8 = 332 → -- time equation
  d = 10 := by
sorry

end NUMINAMATH_CALUDE_man_walking_rest_distance_l1640_164057


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_ab_value_l1640_164028

theorem ellipse_hyperbola_ab_value (a b : ℝ) : 
  (∃ (c : ℝ), c = 5 ∧ c^2 = b^2 - a^2) →
  (∃ (d : ℝ), d = 8 ∧ d^2 = a^2 + b^2) →
  |a * b| = Real.sqrt 3471 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_ab_value_l1640_164028


namespace NUMINAMATH_CALUDE_polynomial_value_at_negative_three_l1640_164087

/-- Given a polynomial ax^5 - bx^3 + cx - 7 that equals 65 when x = 3,
    prove that it equals -79 when x = -3 -/
theorem polynomial_value_at_negative_three 
  (a b c : ℝ) 
  (h : a * 3^5 - b * 3^3 + c * 3 - 7 = 65) :
  a * (-3)^5 - b * (-3)^3 + c * (-3) - 7 = -79 := by
  sorry


end NUMINAMATH_CALUDE_polynomial_value_at_negative_three_l1640_164087


namespace NUMINAMATH_CALUDE_angle_C_is_30_degrees_l1640_164049

-- Define the triangle ABC
structure Triangle :=
  (A B C : Real)  -- Angles
  (a b c : Real)  -- Side lengths

-- Define the properties of the triangle
def IsObtuseTriangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = 180 ∧
  (t.A > 90 ∨ t.B > 90 ∨ t.C > 90)

-- Theorem statement
theorem angle_C_is_30_degrees (t : Triangle) 
  (h1 : IsObtuseTriangle t)
  (h2 : t.a = 4)
  (h3 : t.b = 4 * Real.sqrt 3)
  (h4 : t.A = 30) :
  t.C = 30 :=
sorry

end NUMINAMATH_CALUDE_angle_C_is_30_degrees_l1640_164049


namespace NUMINAMATH_CALUDE_no_intersection_l1640_164097

-- Define the two functions
def f (x : ℝ) : ℝ := |3 * x + 6|
def g (x : ℝ) : ℝ := -|4 * x - 3|

-- Theorem stating that there are no intersection points
theorem no_intersection :
  ¬ ∃ (x y : ℝ), f x = y ∧ g x = y :=
sorry

end NUMINAMATH_CALUDE_no_intersection_l1640_164097


namespace NUMINAMATH_CALUDE_consecutive_numbers_pairing_l1640_164029

theorem consecutive_numbers_pairing (n : ℕ) : 
  ∃ (p₁ p₂ : List (ℕ × ℕ)), 
    p₁ ≠ p₂ ∧ 
    (∀ i ∈ [0, 1, 2, 3, 4], 
      (n + 2*i) ∈ (p₁.map Prod.fst ++ p₁.map Prod.snd) ∧ 
      (n + 2*i + 1) ∈ (p₁.map Prod.fst ++ p₁.map Prod.snd)) ∧
    (∀ i ∈ [0, 1, 2, 3, 4], 
      (n + 2*i) ∈ (p₂.map Prod.fst ++ p₂.map Prod.snd) ∧ 
      (n + 2*i + 1) ∈ (p₂.map Prod.fst ++ p₂.map Prod.snd)) ∧
    p₁.length = 5 ∧ 
    p₂.length = 5 ∧ 
    (p₁.map (λ (a, b) => a * b)).sum = (p₂.map (λ (a, b) => a * b)).sum :=
by
  sorry


end NUMINAMATH_CALUDE_consecutive_numbers_pairing_l1640_164029


namespace NUMINAMATH_CALUDE_cube_and_sphere_problem_l1640_164083

theorem cube_and_sphere_problem (V1 : Real) (h1 : V1 = 8) : ∃ (V2 r : Real),
  let s1 := V1 ^ (1/3)
  let A1 := 6 * s1^2
  let A2 := 3 * A1
  let s2 := (A2 / 6) ^ (1/2)
  V2 = s2^3 ∧ 
  4 * Real.pi * r^2 = A2 ∧
  V2 = 24 * Real.sqrt 3 ∧
  r = Real.sqrt (18 / Real.pi) := by
sorry

end NUMINAMATH_CALUDE_cube_and_sphere_problem_l1640_164083


namespace NUMINAMATH_CALUDE_fun_run_signups_l1640_164044

/-- The number of people who signed up for the Fun Run last year -/
def signups_last_year : ℕ := sorry

/-- The number of people who did not show up to run last year -/
def no_shows : ℕ := 40

/-- The number of people running this year -/
def runners_this_year : ℕ := 320

theorem fun_run_signups :
  signups_last_year = 200 ∧
  runners_this_year = 2 * (signups_last_year - no_shows) :=
sorry

end NUMINAMATH_CALUDE_fun_run_signups_l1640_164044


namespace NUMINAMATH_CALUDE_intersection_of_sets_l1640_164040

theorem intersection_of_sets : 
  let M : Set ℤ := {-1, 1, 3, 5}
  let N : Set ℤ := {-3, 1, 5}
  M ∩ N = {1, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l1640_164040


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1640_164002

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1640_164002


namespace NUMINAMATH_CALUDE_symmetric_distribution_property_l1640_164046

/-- A symmetric distribution about a mean -/
structure SymmetricDistribution where
  /-- The mean of the distribution -/
  mean : ℝ
  /-- The standard deviation of the distribution -/
  std_dev : ℝ
  /-- The cumulative distribution function -/
  cdf : ℝ → ℝ
  /-- The distribution is symmetric about the mean -/
  symmetric : ∀ x, cdf (mean - x) + cdf (mean + x) = 1
  /-- 68% of the distribution lies within one standard deviation of the mean -/
  std_dev_property : cdf (mean + std_dev) - cdf (mean - std_dev) = 0.68

/-- 
Theorem: In a symmetric distribution about the mean m, where 68% of the distribution 
lies within one standard deviation h of the mean, the percentage of the distribution 
less than m + h is 84%.
-/
theorem symmetric_distribution_property (d : SymmetricDistribution) : 
  d.cdf (d.mean + d.std_dev) = 0.84 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_distribution_property_l1640_164046


namespace NUMINAMATH_CALUDE_range_of_m_value_of_m_l1640_164051

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : Prop :=
  x^2 - 4*x + 3 - 2*m = 0

-- Define the condition for distinct real roots
def has_distinct_real_roots (m : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ quadratic_equation x₁ m ∧ quadratic_equation x₂ m

-- Define the additional condition
def additional_condition (x₁ x₂ m : ℝ) : Prop :=
  x₁ * x₂ + x₁ + x₂ - m^2 = 4

-- Theorem 1: Range of m
theorem range_of_m (m : ℝ) :
  has_distinct_real_roots m ↔ m ≥ -1/2 :=
sorry

-- Theorem 2: Value of m
theorem value_of_m (m : ℝ) :
  has_distinct_real_roots m ∧
  (∃ (x₁ x₂ : ℝ), quadratic_equation x₁ m ∧ quadratic_equation x₂ m ∧ additional_condition x₁ x₂ m) →
  m = 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_value_of_m_l1640_164051


namespace NUMINAMATH_CALUDE_basil_sage_ratio_l1640_164062

def herb_ratio (basil sage verbena : ℕ) : Prop :=
  sage = verbena - 5 ∧
  basil = 12 ∧
  basil + sage + verbena = 29

theorem basil_sage_ratio :
  ∀ basil sage verbena : ℕ,
    herb_ratio basil sage verbena →
    (basil : ℚ) / sage = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_basil_sage_ratio_l1640_164062


namespace NUMINAMATH_CALUDE_negation_of_existence_cube_positive_negation_l1640_164007

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬∃ x, P x) ↔ (∀ x, ¬P x) :=
by sorry

theorem cube_positive_negation : 
  (¬∃ x : ℝ, x^3 > 0) ↔ (∀ x : ℝ, x^3 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_cube_positive_negation_l1640_164007


namespace NUMINAMATH_CALUDE_exponential_inequality_l1640_164038

theorem exponential_inequality (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  Real.exp a * Real.exp c > Real.exp b * Real.exp d := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l1640_164038


namespace NUMINAMATH_CALUDE_constant_term_expansion_constant_term_is_three_l1640_164094

/-- The constant term in the expansion of (x^2 + 2)(1/x^2 - 1)^5 -/
theorem constant_term_expansion : ℤ :=
  let expansion := (fun x : ℚ => (x^2 + 2) * (1/x^2 - 1)^5)
  3

/-- Proof that the constant term in the expansion of (x^2 + 2)(1/x^2 - 1)^5 is 3 -/
theorem constant_term_is_three : constant_term_expansion = 3 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_constant_term_is_three_l1640_164094


namespace NUMINAMATH_CALUDE_percentage_equality_l1640_164098

theorem percentage_equality : (0.1 / 100) * 12356 = 12.356000000000002 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l1640_164098


namespace NUMINAMATH_CALUDE_complement_of_union_M_N_l1640_164068

-- Define the sets M and N
def M : Set ℝ := {x | (x + 3) * (x - 1) < 0}
def N : Set ℝ := {x | x ≤ -3}

-- State the theorem
theorem complement_of_union_M_N :
  (M ∪ N)ᶜ = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_M_N_l1640_164068


namespace NUMINAMATH_CALUDE_harry_friday_speed_l1640_164024

-- Define Harry's running speeds throughout the week
def monday_speed : ℝ := 10
def tuesday_speed : ℝ := monday_speed * (1 - 0.3)
def wednesday_speed : ℝ := monday_speed * (1 + 0.5)
def thursday_speed : ℝ := wednesday_speed
def friday_speed : ℝ := thursday_speed * (1 + 0.6)

-- Theorem statement
theorem harry_friday_speed : friday_speed = 24 := by
  sorry

end NUMINAMATH_CALUDE_harry_friday_speed_l1640_164024


namespace NUMINAMATH_CALUDE_calculation_proof_l1640_164066

theorem calculation_proof :
  ((-36) * (-(7/12) - 3/4 + 5/6) = 18) ∧
  (-3^2 / 4 * |-(4/3)| * 6 + (-2)^3 = -26) := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l1640_164066


namespace NUMINAMATH_CALUDE_cups_needed_for_six_cookies_l1640_164069

/-- The number of cups in a quart -/
def cups_per_quart : ℚ := 4

/-- The number of cookies that can be baked with 3 quarts of milk -/
def cookies_per_three_quarts : ℚ := 18

/-- The number of cookies we want to bake -/
def target_cookies : ℚ := 6

/-- The number of cups of milk needed to bake the target number of cookies -/
def cups_needed : ℚ := (3 * cups_per_quart * target_cookies) / cookies_per_three_quarts

theorem cups_needed_for_six_cookies :
  cups_needed = 4 := by sorry

end NUMINAMATH_CALUDE_cups_needed_for_six_cookies_l1640_164069


namespace NUMINAMATH_CALUDE_opposite_face_in_cube_net_l1640_164074

-- Define the faces of the cube
inductive Face : Type
  | x | A | B | C | D | Z

-- Define the cube net structure
structure CubeNet where
  faces : List Face
  surrounding : List Face
  not_connected : Face

-- Define the property of being opposite in a cube
def opposite (f1 f2 : Face) : Prop := sorry

-- Theorem statement
theorem opposite_face_in_cube_net (net : CubeNet) :
  net.faces = [Face.x, Face.A, Face.B, Face.C, Face.D, Face.Z] →
  net.surrounding = [Face.A, Face.B, Face.Z, Face.C] →
  net.not_connected = Face.D →
  opposite Face.x Face.D :=
by sorry

end NUMINAMATH_CALUDE_opposite_face_in_cube_net_l1640_164074


namespace NUMINAMATH_CALUDE_complex_number_coordinates_l1640_164072

theorem complex_number_coordinates (z : ℂ) : z = (2 * Complex.I) / (1 + Complex.I) → z.re = 1 ∧ z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_coordinates_l1640_164072


namespace NUMINAMATH_CALUDE_regular_dodecagon_diagonal_sum_l1640_164096

/-- A regular dodecagon -/
structure RegularDodecagon where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  radius_pos : radius > 0

/-- A diagonal in a regular dodecagon -/
inductive Diagonal
  | D1_7  -- Diagonal from vertex 1 to vertex 7
  | D1_3  -- Diagonal from vertex 1 to vertex 3
  | D1_11 -- Diagonal from vertex 1 to vertex 11

/-- The length of a diagonal in a regular dodecagon -/
def diagonalLength (d : RegularDodecagon) (diag : Diagonal) : ℝ :=
  match diag with
  | Diagonal.D1_7  => 2 * d.radius
  | Diagonal.D1_3  => d.radius
  | Diagonal.D1_11 => d.radius

/-- Theorem: In a regular dodecagon, there exist three diagonals where 
    the length of one diagonal equals the sum of the lengths of the other two -/
theorem regular_dodecagon_diagonal_sum (d : RegularDodecagon) :
  ∃ (d1 d2 d3 : Diagonal), 
    diagonalLength d d1 = diagonalLength d d2 + diagonalLength d d3 :=
sorry


end NUMINAMATH_CALUDE_regular_dodecagon_diagonal_sum_l1640_164096


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l1640_164086

theorem closest_integer_to_cube_root : ∃ (n : ℤ), 
  n = 10 ∧ ∀ (m : ℤ), |m - (7^3 + 9^3)^(1/3)| ≥ |n - (7^3 + 9^3)^(1/3)| :=
by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l1640_164086


namespace NUMINAMATH_CALUDE_liu_hui_perimeter_l1640_164064

-- Define the right triangle
def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Define the parallelogram formed by Liu Hui block puzzle
def liu_hui_parallelogram (a b c : ℝ) : Prop :=
  right_triangle a b c ∧ a > 0 ∧ b > 0 ∧ c > 0

-- Theorem statement
theorem liu_hui_perimeter (a b c : ℝ) :
  liu_hui_parallelogram a b c → a = 3 → b = 4 → 2 * (c + a + b) = 24 := by
  sorry


end NUMINAMATH_CALUDE_liu_hui_perimeter_l1640_164064


namespace NUMINAMATH_CALUDE_linked_rings_distance_l1640_164085

/-- Calculates the total distance of a series of linked rings -/
def total_distance (top_diameter : ℕ) (bottom_diameter : ℕ) (thickness : ℕ) : ℕ :=
  let num_rings := (top_diameter - bottom_diameter) / 2 + 1
  let sum_inside_diameters := num_rings * (top_diameter - thickness + bottom_diameter - thickness) / 2
  sum_inside_diameters + 2 * thickness

/-- Theorem stating the total distance of the linked rings -/
theorem linked_rings_distance :
  total_distance 30 4 2 = 214 := by
  sorry

end NUMINAMATH_CALUDE_linked_rings_distance_l1640_164085


namespace NUMINAMATH_CALUDE_complement_of_A_l1640_164012

-- Define the universal set U
def U : Set Int := {-1, 0, 2}

-- Define set A
def A : Set Int := {-1, 0}

-- Define the complement operation
def complement (S : Set Int) : Set Int :=
  {x | x ∈ U ∧ x ∉ S}

-- Theorem statement
theorem complement_of_A : complement A = {2} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l1640_164012


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1640_164053

/-- A geometric sequence with specified properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  h1 : a 5 - a 3 = 12
  h2 : a 6 - a 4 = 24

/-- Sum of the first n terms of a geometric sequence -/
def S (seq : GeometricSequence) (n : ℕ) : ℝ := sorry

/-- Theorem stating the ratio of S_n to a_n -/
theorem geometric_sequence_ratio (seq : GeometricSequence) (n : ℕ) :
  S seq n / seq.a n = 2 - 2^(1 - n) := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1640_164053


namespace NUMINAMATH_CALUDE_evaluate_expression_l1640_164059

theorem evaluate_expression : 4^3 - 4 * 4^2 + 6 * 4 - 1 = 23 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1640_164059


namespace NUMINAMATH_CALUDE_exists_fib_divisible_by_10_8_l1640_164014

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem exists_fib_divisible_by_10_8 :
  ∃ k : ℕ, k ≤ 10000000000000002 ∧ fib k % (10^8) = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_fib_divisible_by_10_8_l1640_164014


namespace NUMINAMATH_CALUDE_closest_fraction_l1640_164003

def actual_fraction : ℚ := 24 / 150

def candidate_fractions : List ℚ := [1/5, 1/6, 1/7, 1/8, 1/9]

theorem closest_fraction :
  ∃ (closest : ℚ), closest ∈ candidate_fractions ∧
  (∀ (f : ℚ), f ∈ candidate_fractions → |f - actual_fraction| ≥ |closest - actual_fraction|) ∧
  closest = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_closest_fraction_l1640_164003


namespace NUMINAMATH_CALUDE_inequality_solution_l1640_164018

theorem inequality_solution (p q r : ℝ) 
  (h1 : ∀ x : ℝ, (x - p) * (x - q) / (x - r) ≤ 0 ↔ x < -1 ∨ |x - 30| ≤ 2)
  (h2 : p < q) : 
  p + 2*q + 3*r = 89 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1640_164018


namespace NUMINAMATH_CALUDE_arithmetic_less_than_geometric_l1640_164095

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q

/-- Theorem: For arithmetic sequence a and geometric sequence b satisfying given conditions,
    a_n < b_n for all n > 2 -/
theorem arithmetic_less_than_geometric
  (a b : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_eq1 : a 1 = b 1)
  (h_eq2 : a 2 = b 2)
  (h_neq : a 1 ≠ a 2)
  (h_pos : ∀ i : ℕ, a i > 0) :
  ∀ n : ℕ, n > 2 → a n < b n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_less_than_geometric_l1640_164095


namespace NUMINAMATH_CALUDE_fencing_cost_theorem_l1640_164099

/-- A rectangular park with sides in ratio 3:2 and area 5766 sq m -/
structure RectangularPark where
  length : ℝ
  width : ℝ
  ratio_condition : length = 3/2 * width
  area_condition : length * width = 5766

/-- The cost of fencing in paise per meter -/
def fencing_cost_paise : ℝ := 50

/-- Theorem stating the cost of fencing the park -/
theorem fencing_cost_theorem (park : RectangularPark) : 
  (2 * (park.length + park.width) * fencing_cost_paise) / 100 = 155 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_theorem_l1640_164099


namespace NUMINAMATH_CALUDE_f_72_value_l1640_164071

/-- A function satisfying f(ab) = f(a) + f(b) for all a and b -/
def MultitiveFunction (f : ℕ → ℝ) : Prop :=
  ∀ a b : ℕ, f (a * b) = f a + f b

theorem f_72_value (f : ℕ → ℝ) (p q : ℝ) 
    (h_mult : MultitiveFunction f) 
    (h_f2 : f 2 = q) 
    (h_f3 : f 3 = p) : 
  f 72 = 2 * p + 3 * q := by
  sorry

end NUMINAMATH_CALUDE_f_72_value_l1640_164071


namespace NUMINAMATH_CALUDE_equation_equivalence_l1640_164005

theorem equation_equivalence (x y : ℝ) 
  (eq1 : 2 * x^2 + 6 * x + 5 * y + 1 = 0) 
  (eq2 : 2 * x + y + 3 = 0) : 
  y^2 + 10 * y - 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1640_164005


namespace NUMINAMATH_CALUDE_no_valid_arrangement_l1640_164000

/-- A type representing a circular arrangement of 60 numbers -/
def CircularArrangement := Fin 60 → ℕ

/-- Predicate checking if a given arrangement satisfies all conditions -/
def SatisfiesConditions (arr : CircularArrangement) : Prop :=
  (∀ i : Fin 60, (arr i + arr ((i + 2) % 60)) % 2 = 0) ∧
  (∀ i : Fin 60, (arr i + arr ((i + 3) % 60)) % 3 = 0) ∧
  (∀ i : Fin 60, (arr i + arr ((i + 7) % 60)) % 7 = 0)

/-- Predicate checking if an arrangement is a permutation of 1 to 60 -/
def IsValidArrangement (arr : CircularArrangement) : Prop :=
  (∀ n : ℕ, n ∈ Finset.range 60 → ∃ i : Fin 60, arr i = n + 1) ∧
  (∀ i j : Fin 60, arr i = arr j → i = j)

/-- Theorem stating the impossibility of the arrangement -/
theorem no_valid_arrangement :
  ¬ ∃ arr : CircularArrangement, IsValidArrangement arr ∧ SatisfiesConditions arr :=
sorry

end NUMINAMATH_CALUDE_no_valid_arrangement_l1640_164000


namespace NUMINAMATH_CALUDE_regular_pyramid_from_equal_edges_l1640_164050

/-- A pyramid is a polyhedron with a polygonal base and triangular faces meeting at a common point (apex) --/
structure Pyramid where
  base : Set Point
  apex : Point
  lateral_faces : Set (Set Point)

/-- A regular pyramid is a pyramid with a regular polygon base and congruent triangular faces --/
def IsRegularPyramid (p : Pyramid) : Prop := sorry

/-- All edges of a pyramid have equal length --/
def AllEdgesEqual (p : Pyramid) : Prop := sorry

/-- Theorem: If all edges of a pyramid are equal, then it is a regular pyramid --/
theorem regular_pyramid_from_equal_edges (p : Pyramid) :
  AllEdgesEqual p → IsRegularPyramid p := by sorry

end NUMINAMATH_CALUDE_regular_pyramid_from_equal_edges_l1640_164050


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1640_164088

theorem binomial_expansion_coefficient (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x + 1)^5 = a₀ + a₁*(x - 1) + a₂*(x - 1)^2 + a₃*(x - 1)^3 + a₄*(x - 1)^4 + a₅*(x - 1)^5) →
  a₀ = 32 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1640_164088


namespace NUMINAMATH_CALUDE_circle_area_from_PQ_l1640_164013

-- Define the points P and Q
def P : ℝ × ℝ := (1, 3)
def Q : ℝ × ℝ := (5, 8)

-- Define the circle based on the diameter endpoints
def circle_from_diameter (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {x : ℝ × ℝ | ∃ (c : ℝ × ℝ), (x.1 - c.1)^2 + (x.2 - c.2)^2 = ((p.1 - q.1)^2 + (p.2 - q.2)^2) / 4}

-- Define the area of a circle
def circle_area (c : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem circle_area_from_PQ : 
  circle_area (circle_from_diameter P Q) = 41 * π / 4 := by sorry

end NUMINAMATH_CALUDE_circle_area_from_PQ_l1640_164013


namespace NUMINAMATH_CALUDE_remainder_11_pow_2023_mod_5_l1640_164060

theorem remainder_11_pow_2023_mod_5 : 11^2023 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_11_pow_2023_mod_5_l1640_164060


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_equation_l1640_164055

theorem arithmetic_geometric_mean_equation (a b : ℝ) :
  (a + b) / 2 = 10 ∧ Real.sqrt (a * b) = 10 →
  ∀ x, x^2 - 20*x + 100 = 0 ↔ x = a ∨ x = b :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_equation_l1640_164055


namespace NUMINAMATH_CALUDE_equal_prob_conditions_l1640_164045

/-- Represents an urn with two compartments -/
structure Urn :=
  (v₁ f₁ v₂ f₂ : ℕ)

/-- Probability of drawing a red ball from a partitioned urn -/
def probPartitioned (u : Urn) : ℚ :=
  (u.v₁ * (u.v₂ + u.f₂) + u.v₂ * (u.v₁ + u.f₁)) / (2 * (u.v₁ + u.f₁) * (u.v₂ + u.f₂))

/-- Probability of drawing a red ball from a non-partitioned urn -/
def probNonPartitioned (u : Urn) : ℚ :=
  (u.v₁ + u.v₂) / ((u.v₁ + u.f₁) + (u.v₂ + u.f₂))

/-- Theorem stating the conditions for equal probabilities -/
theorem equal_prob_conditions (u : Urn) :
  probPartitioned u = probNonPartitioned u ↔ u.v₁ * u.f₂ = u.v₂ * u.f₁ ∨ u.v₁ + u.f₁ = u.v₂ + u.f₂ :=
sorry

end NUMINAMATH_CALUDE_equal_prob_conditions_l1640_164045


namespace NUMINAMATH_CALUDE_central_cell_is_seven_l1640_164030

-- Define the grid
def Grid := Fin 3 → Fin 3 → Fin 9

-- Define adjacency
def adjacent (a b : Fin 3 × Fin 3) : Prop :=
  (a.1 = b.1 ∧ a.2.succ = b.2) ∨
  (a.1 = b.1 ∧ a.2 = b.2.succ) ∨
  (a.1.succ = b.1 ∧ a.2 = b.2) ∨
  (a.1 = b.1.succ ∧ a.2 = b.2)

-- Define consecutive numbers
def consecutive (a b : Fin 9) : Prop :=
  a.val.succ = b.val ∨ b.val.succ = a.val

-- Define the property of consecutive numbers being adjacent
def consecutiveAdjacent (g : Grid) : Prop :=
  ∀ i j k l : Fin 3, consecutive (g i j) (g k l) → adjacent (i, j) (k, l)

-- Define corner cells
def cornerCells : List (Fin 3 × Fin 3) :=
  [(0, 0), (0, 2), (2, 0), (2, 2)]

-- Define the sum of corner cells
def cornerSum (g : Grid) : Nat :=
  (cornerCells.map (fun (i, j) => (g i j).val)).sum

-- Define central cell
def centralCell : Fin 3 × Fin 3 := (1, 1)

-- Theorem statement
theorem central_cell_is_seven (g : Grid)
  (h1 : consecutiveAdjacent g)
  (h2 : cornerSum g = 18) :
  (g centralCell.1 centralCell.2).val = 7 := by
  sorry

end NUMINAMATH_CALUDE_central_cell_is_seven_l1640_164030


namespace NUMINAMATH_CALUDE_fraction_simplification_l1640_164058

theorem fraction_simplification : 
  (3 / 7 + 5 / 8) / (5 / 12 + 2 / 3) = 59 / 61 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1640_164058


namespace NUMINAMATH_CALUDE_four_digit_integers_with_specific_remainders_l1640_164093

theorem four_digit_integers_with_specific_remainders :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, 1000 ≤ n ∧ n < 10000 ∧ 
      n % 7 = 1 ∧ n % 10 = 3 ∧ n % 13 = 5) ∧
    (∀ n, 1000 ≤ n ∧ n < 10000 ∧ 
      n % 7 = 1 ∧ n % 10 = 3 ∧ n % 13 = 5 → n ∈ s) ∧
    Finset.card s = 6 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_integers_with_specific_remainders_l1640_164093


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1640_164016

/-- Given a geometric sequence {a_n} and its partial sums S_n -/
def geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r ≠ 0 ∧ ∀ n, a (n + 1) = r * a n

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))

theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  geometric_sequence a S →
  32 * a 2 + a 7 = 0 →
  S 5 / S 2 = -11 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1640_164016


namespace NUMINAMATH_CALUDE_marie_sold_700_reading_materials_l1640_164089

/-- The number of magazines Marie sold -/
def magazines : ℕ := 425

/-- The number of newspapers Marie sold -/
def newspapers : ℕ := 275

/-- The total number of reading materials Marie sold -/
def total_reading_materials : ℕ := magazines + newspapers

/-- Proof that Marie sold 700 reading materials -/
theorem marie_sold_700_reading_materials : total_reading_materials = 700 := by
  sorry

end NUMINAMATH_CALUDE_marie_sold_700_reading_materials_l1640_164089


namespace NUMINAMATH_CALUDE_all_expressions_correct_l1640_164033

theorem all_expressions_correct (x y : ℚ) (h : x / y = 5 / 3) :
  (2 * x + y) / y = 13 / 3 ∧
  y / (y - 2 * x) = 3 / (-7) ∧
  (x + y) / x = 8 / 5 ∧
  x / (3 * y) = 5 / 9 ∧
  (x - 2 * y) / y = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_all_expressions_correct_l1640_164033


namespace NUMINAMATH_CALUDE_money_left_after_purchase_l1640_164032

def meat_quantity : ℕ := 2
def meat_price_per_kg : ℕ := 82
def initial_money : ℕ := 180

theorem money_left_after_purchase : 
  initial_money - (meat_quantity * meat_price_per_kg) = 16 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_purchase_l1640_164032


namespace NUMINAMATH_CALUDE_stating_right_triangle_constructible_l1640_164023

/-- Represents a right triangle --/
structure RightTriangle where
  hypotenuse : ℝ
  angle_difference : ℝ
  h_hypotenuse_positive : hypotenuse > 0
  h_angle_difference_range : 0 < angle_difference ∧ angle_difference < 90

/-- 
Theorem stating that a right triangle can be constructed 
given its hypotenuse and the difference of its two acute angles (ε), 
if and only if 0 < ε < 90°
--/
theorem right_triangle_constructible (h : ℝ) (ε : ℝ) :
  (∃ (t : RightTriangle), t.hypotenuse = h ∧ t.angle_difference = ε) ↔ 
  (h > 0 ∧ 0 < ε ∧ ε < 90) :=
sorry

end NUMINAMATH_CALUDE_stating_right_triangle_constructible_l1640_164023


namespace NUMINAMATH_CALUDE_closet_area_l1640_164091

theorem closet_area (diagonal : ℝ) (shorter_side : ℝ)
  (h1 : diagonal = 7)
  (h2 : shorter_side = 4) :
  ∃ (area : ℝ), area = 4 * Real.sqrt 33 ∧ 
  area = shorter_side * Real.sqrt (diagonal^2 - shorter_side^2) :=
by sorry

end NUMINAMATH_CALUDE_closet_area_l1640_164091


namespace NUMINAMATH_CALUDE_polynomial_value_l1640_164047

theorem polynomial_value (x : ℝ) (h : x^2 + 2*x + 1 = 4) : 2*x^2 + 4*x + 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l1640_164047


namespace NUMINAMATH_CALUDE_solution_set_l1640_164067

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + (a - b)*x + 1

-- Define the property of f being even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the domain of f
def domain (a : ℝ) : Set ℝ := Set.Icc (a - 1) (2*a + 4)

-- State the theorem
theorem solution_set 
  (a b : ℝ) 
  (h1 : is_even (f a b))
  (h2 : domain a = Set.Icc (-2) 2) :
  {x | f a b x > f a b b} = 
    (Set.Ioc (-2) (-1) ∪ Set.Ioc 1 2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_l1640_164067


namespace NUMINAMATH_CALUDE_cats_sold_l1640_164054

theorem cats_sold (siamese : ℕ) (house : ℕ) (remaining : ℕ) :
  siamese = 19 →
  house = 45 →
  remaining = 8 →
  siamese + house - remaining = 56 :=
by
  sorry

end NUMINAMATH_CALUDE_cats_sold_l1640_164054


namespace NUMINAMATH_CALUDE_computer_additions_l1640_164075

/-- Represents the number of additions a computer can perform per second. -/
def additions_per_second : ℕ := 15000

/-- Represents the duration in seconds for which we want to calculate the total additions. -/
def duration_in_seconds : ℕ := 2 * 3600 + 30 * 60

/-- Calculates the total number of additions performed by the computer. -/
def total_additions : ℕ := additions_per_second * duration_in_seconds

/-- Theorem stating that the computer performs 135,000,000 additions in the given time. -/
theorem computer_additions : total_additions = 135000000 := by
  sorry

end NUMINAMATH_CALUDE_computer_additions_l1640_164075


namespace NUMINAMATH_CALUDE_fred_bought_two_tickets_l1640_164026

def ticket_cost : ℚ := 592 / 100
def movie_cost : ℚ := 679 / 100
def paid_amount : ℚ := 20
def change_received : ℚ := 137 / 100

theorem fred_bought_two_tickets :
  (paid_amount - change_received - movie_cost) / ticket_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_fred_bought_two_tickets_l1640_164026


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odds_l1640_164043

theorem largest_divisor_of_consecutive_odds (n : ℕ) (h : Even n) (h_pos : 0 < n) :
  (∃ (k : ℕ), (n + 1) * (n + 3) * (n + 7) * (n + 9) * (n + 11) = 15 * k) ∧
  (∀ (m : ℕ), m > 15 → ∃ (n : ℕ), Even n ∧ 0 < n ∧
    ¬(∃ (k : ℕ), (n + 1) * (n + 3) * (n + 7) * (n + 9) * (n + 11) = m * k)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odds_l1640_164043


namespace NUMINAMATH_CALUDE_bus_truck_speed_ratio_l1640_164073

theorem bus_truck_speed_ratio :
  ∀ (distance : ℝ) (bus_time truck_time : ℝ),
    bus_time = 10 →
    truck_time = 15 →
    (distance / bus_time) / (distance / truck_time) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_bus_truck_speed_ratio_l1640_164073


namespace NUMINAMATH_CALUDE_periodic_trig_function_l1640_164034

theorem periodic_trig_function (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (π * x + α) + b * Real.cos (π * x + β)
  f 2015 = -1 → f 2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_periodic_trig_function_l1640_164034


namespace NUMINAMATH_CALUDE_special_triangle_sum_squares_is_square_l1640_164076

/-- A triangle with integer side lengths where one altitude equals the sum of the other two -/
structure SpecialTriangle where
  a : ℤ
  b : ℤ
  c : ℤ
  altitude_sum : ℝ
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ
  side_positive : 0 < a ∧ 0 < b ∧ 0 < c
  altitude_relation : h₁ = h₂ + h₃
  area_equality : (a : ℝ) * h₁ = (b : ℝ) * h₂ ∧ (b : ℝ) * h₂ = (c : ℝ) * h₃

theorem special_triangle_sum_squares_is_square (t : SpecialTriangle) :
  ∃ n : ℤ, t.a^2 + t.b^2 + t.c^2 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_sum_squares_is_square_l1640_164076


namespace NUMINAMATH_CALUDE_rohan_salary_l1640_164082

/-- Rohan's monthly expenses and savings -/
structure RohanFinances where
  salary : ℝ
  food_percent : ℝ
  rent_percent : ℝ
  entertainment_percent : ℝ
  conveyance_percent : ℝ
  education_percent : ℝ
  utilities_percent : ℝ
  savings : ℝ

/-- Theorem stating Rohan's monthly salary given his expenses and savings -/
theorem rohan_salary (r : RohanFinances) : 
  r.food_percent = 0.30 →
  r.rent_percent = 0.20 →
  r.entertainment_percent = r.food_percent / 2 →
  r.conveyance_percent = r.entertainment_percent * 1.25 →
  r.education_percent = 0.05 →
  r.utilities_percent = 0.10 →
  r.savings = 2500 →
  r.salary = 200000 := by
  sorry

end NUMINAMATH_CALUDE_rohan_salary_l1640_164082


namespace NUMINAMATH_CALUDE_inheritance_interest_rate_proof_l1640_164010

theorem inheritance_interest_rate_proof (inheritance : ℝ) (amount_first : ℝ) (rate_first : ℝ) (total_interest : ℝ) :
  inheritance = 12000 →
  amount_first = 5000 →
  rate_first = 0.06 →
  total_interest = 860 →
  let amount_second := inheritance - amount_first
  let interest_first := amount_first * rate_first
  let interest_second := total_interest - interest_first
  let rate_second := interest_second / amount_second
  rate_second = 0.08 := by sorry

end NUMINAMATH_CALUDE_inheritance_interest_rate_proof_l1640_164010


namespace NUMINAMATH_CALUDE_square_of_product_72519_9999_l1640_164020

theorem square_of_product_72519_9999 : (72519 * 9999)^2 = 525545577128752961 := by
  sorry

end NUMINAMATH_CALUDE_square_of_product_72519_9999_l1640_164020


namespace NUMINAMATH_CALUDE_sqrt_2x_minus_4_real_l1640_164052

theorem sqrt_2x_minus_4_real (x : ℝ) : (∃ (y : ℝ), y ^ 2 = 2 * x - 4) ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2x_minus_4_real_l1640_164052


namespace NUMINAMATH_CALUDE_range_of_m_with_two_integer_solutions_l1640_164015

theorem range_of_m_with_two_integer_solutions (m : ℝ) : 
  (∃ (x y : ℤ), x ≠ y ∧ 
    (∀ z : ℤ, (-1 : ℝ) ≤ z ∧ (z : ℝ) < m ↔ z = x ∨ z = y)) →
  0 < m ∧ m ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_with_two_integer_solutions_l1640_164015


namespace NUMINAMATH_CALUDE_constant_sequence_l1640_164084

theorem constant_sequence (a : Fin 2016 → ℕ) 
  (h1 : ∀ i, a i ≤ 2016)
  (h2 : ∀ i j : Fin 2016, (i.val + j.val) ∣ (i.val * a i + j.val * a j)) :
  ∀ i j : Fin 2016, a i = a j :=
by sorry

end NUMINAMATH_CALUDE_constant_sequence_l1640_164084


namespace NUMINAMATH_CALUDE_range_of_m_m_value_for_specific_chord_length_m_value_for_perpendicular_chords_l1640_164081

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Theorem 1: Range of m
theorem range_of_m :
  ∀ m : ℝ, (∃ x y : ℝ, circle_equation x y m) → m < 5 :=
sorry

-- Theorem 2: Value of m when |MN| = 4√5/5
theorem m_value_for_specific_chord_length :
  ∃ m : ℝ, 
    (∃ x₁ y₁ x₂ y₂ : ℝ,
      circle_equation x₁ y₁ m ∧ 
      circle_equation x₂ y₂ m ∧
      line_equation x₁ y₁ ∧ 
      line_equation x₂ y₂ ∧
      (x₁ - x₂)^2 + (y₁ - y₂)^2 = (4*Real.sqrt 5/5)^2) ∧
    m = 4 :=
sorry

-- Theorem 3: Value of m when OM ⊥ ON
theorem m_value_for_perpendicular_chords :
  ∃ m : ℝ, 
    (∃ x₁ y₁ x₂ y₂ : ℝ,
      circle_equation x₁ y₁ m ∧ 
      circle_equation x₂ y₂ m ∧
      line_equation x₁ y₁ ∧ 
      line_equation x₂ y₂ ∧
      x₁*x₂ + y₁*y₂ = 0) ∧
    m = 8/5 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_m_value_for_specific_chord_length_m_value_for_perpendicular_chords_l1640_164081


namespace NUMINAMATH_CALUDE_three_layer_runner_area_l1640_164092

/-- Given three table runners and a table, calculate the area covered by three layers of runner -/
theorem three_layer_runner_area 
  (total_runner_area : ℝ) 
  (table_area : ℝ) 
  (coverage_percent : ℝ) 
  (two_layer_area : ℝ) 
  (h1 : total_runner_area = 224) 
  (h2 : table_area = 175) 
  (h3 : coverage_percent = 0.8) 
  (h4 : two_layer_area = 24) : 
  ∃ (three_layer_area : ℝ), 
    three_layer_area = 12 ∧ 
    total_runner_area = coverage_percent * table_area + two_layer_area + 2 * three_layer_area :=
by sorry

end NUMINAMATH_CALUDE_three_layer_runner_area_l1640_164092


namespace NUMINAMATH_CALUDE_additional_combinations_l1640_164027

theorem additional_combinations (original_set1 original_set3 : ℕ)
  (original_set2 original_set4 : ℕ)
  (added_set1 added_set3 : ℕ) :
  original_set1 = 4 →
  original_set2 = 2 →
  original_set3 = 3 →
  original_set4 = 3 →
  added_set1 = 2 →
  added_set3 = 1 →
  (original_set1 + added_set1) * original_set2 * (original_set3 + added_set3) * original_set4 -
  original_set1 * original_set2 * original_set3 * original_set4 = 72 := by
  sorry

#check additional_combinations

end NUMINAMATH_CALUDE_additional_combinations_l1640_164027


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_AD_l1640_164001

/-- Rectangle ABCD with given properties -/
structure Rectangle where
  AB : ℝ
  AD : ℝ
  E : ℝ
  ac_be_perp : Bool

/-- Conditions for the rectangle -/
def rectangle_conditions (rect : Rectangle) : Prop :=
  rect.AB = 80 ∧
  rect.E = (1/3) * rect.AD ∧
  rect.ac_be_perp = true

/-- Theorem statement -/
theorem greatest_integer_less_than_AD (rect : Rectangle) 
  (h : rectangle_conditions rect) : 
  ⌊rect.AD⌋ = 138 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_AD_l1640_164001


namespace NUMINAMATH_CALUDE_production_days_l1640_164022

/-- Given an initial average production of 50 units over n days, 
    adding 95 units on the next day results in a new average of 55 units,
    prove that n must be 8. -/
theorem production_days (n : ℕ) 
  (h1 : (n * 50 : ℝ) / n = 50)  -- Initial average of 50 units over n days
  (h2 : ((n * 50 + 95 : ℝ) / (n + 1) = 55)) -- New average of 55 units over n+1 days
  : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_production_days_l1640_164022


namespace NUMINAMATH_CALUDE_division_problem_l1640_164037

theorem division_problem (dividend : Nat) (divisor : Nat) (remainder : Nat) (quotient : Nat) :
  dividend = 161 →
  divisor = 16 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  quotient = 10 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1640_164037


namespace NUMINAMATH_CALUDE_cos_triple_angle_l1640_164006

theorem cos_triple_angle (θ : Real) (x : Real) (h : x = Real.cos θ) :
  Real.cos (3 * θ) = 4 * x^3 - 3 * x := by sorry

end NUMINAMATH_CALUDE_cos_triple_angle_l1640_164006


namespace NUMINAMATH_CALUDE_sin_n_equals_cos_276_l1640_164036

theorem sin_n_equals_cos_276 :
  ∃ n : ℤ, -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * π / 180) = Real.cos (276 * π / 180) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_sin_n_equals_cos_276_l1640_164036


namespace NUMINAMATH_CALUDE_cone_max_section_area_cone_max_section_area_condition_l1640_164078

/-- Given a cone with lateral surface that unfolds into a sector with radius 2 and central angle 5π/3,
    the maximum area of any section determined by two generatrices is 2. -/
theorem cone_max_section_area :
  ∀ (r : ℝ) (l : ℝ) (a : ℝ),
  l = 2 →
  2 * π * r = 2 * (5 * π / 3) →
  0 < a →
  a ≤ 2 * r →
  (a / 2) * Real.sqrt (4 - a^2 / 4) ≤ 2 :=
by sorry

/-- The maximum area is achieved when a = 2√2 -/
theorem cone_max_section_area_condition (r : ℝ) (l : ℝ) :
  l = 2 →
  2 * π * r = 2 * (5 * π / 3) →
  (2 * Real.sqrt 2 / 2) * Real.sqrt (4 - (2 * Real.sqrt 2)^2 / 4) = 2 :=
by sorry

end NUMINAMATH_CALUDE_cone_max_section_area_cone_max_section_area_condition_l1640_164078


namespace NUMINAMATH_CALUDE_volume_change_l1640_164056

/-- Represents the dimensions of a rectangular parallelepiped -/
structure Parallelepiped where
  length : ℝ
  breadth : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular parallelepiped -/
def volume (p : Parallelepiped) : ℝ := p.length * p.breadth * p.height

/-- Applies the given changes to the dimensions of a parallelepiped -/
def apply_changes (p : Parallelepiped) : Parallelepiped :=
  { length := p.length * 1.5,
    breadth := p.breadth * 0.7,
    height := p.height * 1.2 }

theorem volume_change (p : Parallelepiped) :
  volume (apply_changes p) = 1.26 * volume p := by
  sorry

end NUMINAMATH_CALUDE_volume_change_l1640_164056


namespace NUMINAMATH_CALUDE_average_of_numbers_l1640_164042

def numbers : List ℕ := [12, 13, 14, 510, 520, 1115, 1120, 1, 1252140, 2345]

theorem average_of_numbers :
  (numbers.sum : ℚ) / numbers.length = 125789 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_l1640_164042


namespace NUMINAMATH_CALUDE_quadratic_equations_properties_l1640_164021

theorem quadratic_equations_properties (b c : ℤ) (x₁ x₂ x₁' x₂' : ℤ) :
  (x₁^2 + b*x₁ + c = 0) →
  (x₂^2 + b*x₂ + c = 0) →
  (x₁'^2 + c*x₁' + b = 0) →
  (x₂'^2 + c*x₂' + b = 0) →
  (x₁ * x₂ > 0) →
  (x₁' * x₂' > 0) →
  (x₁ + x₂ = -b) →
  (x₁ * x₂ = c) →
  (x₁' + x₂' = -c) →
  (x₁' * x₂' = b) →
  (x₁ < 0 ∧ x₂ < 0) ∧
  (b - 1 ≤ c ∧ c ≤ b + 1) ∧
  ((b = 5 ∧ c = 6) ∨ (b = 6 ∧ c = 5) ∨ (b = 4 ∧ c = 4)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_properties_l1640_164021


namespace NUMINAMATH_CALUDE_tailor_trim_problem_l1640_164041

theorem tailor_trim_problem (original_side : ℝ) (trimmed_opposite : ℝ) (remaining_area : ℝ) :
  original_side = 22 →
  trimmed_opposite = 6 →
  remaining_area = 120 →
  ∃ x : ℝ, (original_side - trimmed_opposite - trimmed_opposite) * (original_side - x) = remaining_area ∧ x = 10 :=
by sorry

end NUMINAMATH_CALUDE_tailor_trim_problem_l1640_164041


namespace NUMINAMATH_CALUDE_new_person_weight_l1640_164019

theorem new_person_weight (initial_count : ℕ) (weight_removed : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  weight_removed = 65 →
  avg_increase = 2 →
  ∃ (initial_avg : ℝ) (new_weight : ℝ),
    initial_count * (initial_avg + avg_increase) = initial_count * initial_avg - weight_removed + new_weight →
    new_weight = 81 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1640_164019


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1640_164077

theorem complex_equation_solution (z : ℂ) : (1 + 2*I)*z = 4 + 3*I → z = 2 - I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1640_164077


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1640_164048

theorem arithmetic_calculations : 
  ((1 : ℝ) - 1^2 + 2 * 5 / (1/5) = 49) ∧ 
  (24 * (1/6) + 24 * (-1/8) - (-24) * (1/2) = 13) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1640_164048


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_positive_l1640_164011

theorem quadratic_inequality_always_positive (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 2)*x - k + 8 > 0) ↔ k > -2*Real.sqrt 7 ∧ k < 2*Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_positive_l1640_164011


namespace NUMINAMATH_CALUDE_combine_like_terms_to_zero_l1640_164090

theorem combine_like_terms_to_zero (x y : ℝ) : -2 * x * y^2 + 2 * x * y^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_combine_like_terms_to_zero_l1640_164090


namespace NUMINAMATH_CALUDE_smallest_prime_for_perfect_square_l1640_164017

-- Define a function to check if a number is prime
def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬(n % m = 0)

-- Define a function to check if a number is a perfect square
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

-- Theorem statement
theorem smallest_prime_for_perfect_square :
  ∃ p : Nat,
    isPrime p ∧
    isPerfectSquare (5 * p^2 + p^3) ∧
    (∀ q : Nat, q < p → isPrime q → ¬isPerfectSquare (5 * q^2 + q^3)) ∧
    p = 11 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_for_perfect_square_l1640_164017


namespace NUMINAMATH_CALUDE_bags_weight_after_removal_l1640_164039

/-- 
Given a bag of sugar weighing 16 kg and a bag of salt weighing 30 kg,
if 4 kg is removed from their combined weight, the resulting weight is 42 kg.
-/
theorem bags_weight_after_removal (sugar_weight salt_weight removal_weight : ℕ) 
  (h1 : sugar_weight = 16)
  (h2 : salt_weight = 30)
  (h3 : removal_weight = 4) :
  sugar_weight + salt_weight - removal_weight = 42 :=
by sorry

end NUMINAMATH_CALUDE_bags_weight_after_removal_l1640_164039


namespace NUMINAMATH_CALUDE_at_most_one_integer_solution_l1640_164061

theorem at_most_one_integer_solution (a b : ℤ) :
  ∃! (n : ℕ), ∃ (x : ℤ), (x - a) * (x - b) * (x - 3) + 1 = 0 ∧ n = 1 :=
by sorry

end NUMINAMATH_CALUDE_at_most_one_integer_solution_l1640_164061


namespace NUMINAMATH_CALUDE_total_pencils_is_60_l1640_164070

/-- The total number of pencils owned by 5 children -/
def total_pencils : ℕ :=
  let child1 := 6
  let child2 := 9
  let child3 := 12
  let child4 := 15
  let child5 := 18
  child1 + child2 + child3 + child4 + child5

/-- Theorem stating that the total number of pencils is 60 -/
theorem total_pencils_is_60 : total_pencils = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_is_60_l1640_164070


namespace NUMINAMATH_CALUDE_fifteen_plus_sixteen_l1640_164025

theorem fifteen_plus_sixteen : 15 + 16 = 31 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_plus_sixteen_l1640_164025


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l1640_164008

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 13)
  (sum_products_eq : a * b + a * c + b * c = 32) :
  a^3 + b^3 + c^3 - 3*a*b*c = 949 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l1640_164008


namespace NUMINAMATH_CALUDE_vasyas_birthday_l1640_164065

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to get the day two days after
def twoDaysAfter (d : DayOfWeek) : DayOfWeek :=
  nextDay (nextDay d)

theorem vasyas_birthday (statementDay : DayOfWeek) 
  (h1 : twoDaysAfter statementDay = DayOfWeek.Sunday) :
  nextDay DayOfWeek.Thursday = statementDay :=
by
  sorry


end NUMINAMATH_CALUDE_vasyas_birthday_l1640_164065


namespace NUMINAMATH_CALUDE_pauline_cars_l1640_164031

theorem pauline_cars (total : ℕ) (regular_percent : ℚ) (truck_percent : ℚ) 
  (h_total : total = 125)
  (h_regular : regular_percent = 64/100)
  (h_truck : truck_percent = 8/100) :
  (total : ℚ) * (1 - regular_percent - truck_percent) = 35 := by
  sorry

end NUMINAMATH_CALUDE_pauline_cars_l1640_164031


namespace NUMINAMATH_CALUDE_rect_to_cylindrical_coordinates_l1640_164063

/-- Given a point with rectangular coordinates (1, 1, 1), 
    its cylindrical coordinates are (√2, π/4, 1) -/
theorem rect_to_cylindrical_coordinates : 
  let x : ℝ := 1
  let y : ℝ := 1
  let z : ℝ := 1
  let ρ : ℝ := Real.sqrt 2
  let θ : ℝ := π / 4
  x = ρ * Real.cos θ ∧ 
  y = ρ * Real.sin θ ∧ 
  z = z := by sorry

end NUMINAMATH_CALUDE_rect_to_cylindrical_coordinates_l1640_164063
