import Mathlib

namespace NUMINAMATH_CALUDE_mixture_ratio_l1179_117909

/-- Given two solutions A and B with different alcohol-water ratios, 
    prove that mixing them in a specific ratio results in a mixture with 60% alcohol. -/
theorem mixture_ratio (V_A V_B : ℝ) : 
  V_A > 0 → V_B > 0 →
  (21 / 25 * V_A + 2 / 5 * V_B) / (V_A + V_B) = 3 / 5 →
  V_A / V_B = 5 / 6 := by
sorry

/-- The ratio of Solution A to Solution B in the mixture -/
def solution_ratio : ℚ := 5 / 6

#check mixture_ratio
#check solution_ratio

end NUMINAMATH_CALUDE_mixture_ratio_l1179_117909


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1179_117963

theorem sum_of_three_numbers (a b c : ℝ) : 
  a^2 + b^2 + c^2 = 252 → 
  a*b + b*c + c*a = 116 → 
  a + b + c = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1179_117963


namespace NUMINAMATH_CALUDE_complex_division_result_l1179_117959

theorem complex_division_result : ∃ (i : ℂ), i * i = -1 ∧ (4 * i) / (1 + i) = 2 + 2 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_division_result_l1179_117959


namespace NUMINAMATH_CALUDE_geometric_sequence_a6_l1179_117966

/-- Given a geometric sequence {aₙ} where a₁ = 1 and a₄ = 8, prove that a₆ = 32 -/
theorem geometric_sequence_a6 (a : ℕ → ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) →  -- geometric sequence condition
  a 1 = 1 →                                  -- a₁ = 1
  a 4 = 8 →                                  -- a₄ = 8
  a 6 = 32 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a6_l1179_117966


namespace NUMINAMATH_CALUDE_block_dimension_l1179_117906

/-- The number of positions to place a 2x1x1 block in a layer of 11x10 --/
def positions_in_layer : ℕ := 199

/-- The number of positions to place a 2x1x1 block across two adjacent layers --/
def positions_across_layers : ℕ := 110

/-- The total number of positions to place a 2x1x1 block in a nx11x10 block --/
def total_positions (n : ℕ) : ℕ := n * positions_in_layer + (n - 1) * positions_across_layers

theorem block_dimension (n : ℕ) :
  total_positions n = 2362 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_block_dimension_l1179_117906


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1179_117941

theorem inequality_solution_range (k : ℝ) : 
  (∀ x : ℝ, (k - 1) * x^2 + (k - 1) * x + 2 > 0) ↔ k ∈ Set.Icc 1 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1179_117941


namespace NUMINAMATH_CALUDE_fifth_element_is_35_l1179_117919

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  totalElements : ℕ
  sampleSize : ℕ
  firstElement : ℕ

/-- Calculates the nth element in a systematic sample -/
def nthElement (s : SystematicSampling) (n : ℕ) : ℕ :=
  s.firstElement + (n - 1) * (s.totalElements / s.sampleSize)

theorem fifth_element_is_35 (s : SystematicSampling) 
  (h1 : s.totalElements = 160)
  (h2 : s.sampleSize = 20)
  (h3 : s.firstElement = 3) :
  nthElement s 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_fifth_element_is_35_l1179_117919


namespace NUMINAMATH_CALUDE_franks_books_l1179_117940

theorem franks_books (days_per_book : ℕ) (total_days : ℕ) (h1 : days_per_book = 12) (h2 : total_days = 492) :
  total_days / days_per_book = 41 := by
  sorry

end NUMINAMATH_CALUDE_franks_books_l1179_117940


namespace NUMINAMATH_CALUDE_value_of_expression_l1179_117975

theorem value_of_expression (x : ℤ) (h : x = -4) : 5 * x - 2 = -22 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1179_117975


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1179_117913

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 2 + a^(x - 1)
  f 1 = 3 :=
sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1179_117913


namespace NUMINAMATH_CALUDE_max_value_of_function_l1179_117917

theorem max_value_of_function (x : ℝ) :
  ∃ (M : ℝ), M = Real.sqrt 2 ∧ ∀ y, y = Real.sin (2 * x) - 2 * (Real.sin x)^2 + 1 → y ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_function_l1179_117917


namespace NUMINAMATH_CALUDE_inverse_proportion_k_value_l1179_117947

/-- Given an inverse proportion function y = (k-1)x^(k^2-5) where k is a constant,
    if y decreases as x increases when x > 0, then k = 2 -/
theorem inverse_proportion_k_value (k : ℝ) : 
  (∀ x y : ℝ, x > 0 → y = (k - 1) * x^(k^2 - 5)) →  -- y is a function of x
  (∀ x1 x2 y1 y2 : ℝ, x1 > 0 → x2 > 0 → x1 < x2 → 
    y1 = (k - 1) * x1^(k^2 - 5) → y2 = (k - 1) * x2^(k^2 - 5) → y1 > y2) →  -- y decreases as x increases
  k = 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_value_l1179_117947


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l1179_117973

theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 30)
  (area2 : w * h = 40)
  (area3 : l * h = 12) :
  l * w * h = 120 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l1179_117973


namespace NUMINAMATH_CALUDE_rabbit_chicken_puzzle_l1179_117990

theorem rabbit_chicken_puzzle (total_animals : ℕ) (rabbit_count : ℕ) : 
  total_animals = 40 →
  4 * rabbit_count = 10 * 2 * (total_animals - rabbit_count) + 8 →
  rabbit_count = 33 := by
sorry

end NUMINAMATH_CALUDE_rabbit_chicken_puzzle_l1179_117990


namespace NUMINAMATH_CALUDE_marys_sheep_ratio_l1179_117927

theorem marys_sheep_ratio (initial : ℕ) (remaining : ℕ) : 
  initial = 400 → remaining = 150 → (initial - remaining * 2) / initial = 1 / 4 := by
  sorry

#check marys_sheep_ratio

end NUMINAMATH_CALUDE_marys_sheep_ratio_l1179_117927


namespace NUMINAMATH_CALUDE_expression_evaluation_l1179_117968

theorem expression_evaluation : 
  (0.8 : ℝ)^3 - (0.5 : ℝ)^3 / (0.8 : ℝ)^2 + 0.40 + (0.5 : ℝ)^2 = 0.9666875 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1179_117968


namespace NUMINAMATH_CALUDE_parabola_symmetric_point_l1179_117935

/-- Parabola type -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  h_positive : p > 0
  h_equation : ∀ x y, equation x y ↔ y^2 = 2*p*x
  h_focus : focus = (p/2, 0)

/-- Line type -/
structure Line where
  angle : ℝ
  point : ℝ × ℝ

/-- Symmetric points with respect to a line -/
def symmetric (P Q : ℝ × ℝ) (l : Line) : Prop :=
  sorry

theorem parabola_symmetric_point
  (C : Parabola)
  (l : Line)
  (h_angle : l.angle = π/6)
  (h_passes : l.point = C.focus)
  (P : ℝ × ℝ)
  (h_on_parabola : C.equation P.1 P.2)
  (h_symmetric : symmetric P (5, 0) l) :
  P.1 = 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_symmetric_point_l1179_117935


namespace NUMINAMATH_CALUDE_ant_ratio_l1179_117989

theorem ant_ratio (abe beth cece duke : ℕ) : 
  abe = 4 →
  beth = abe + abe / 2 →
  cece = 2 * abe →
  abe + beth + cece + duke = 20 →
  duke = 2 ∧ duke * 2 = abe := by sorry

end NUMINAMATH_CALUDE_ant_ratio_l1179_117989


namespace NUMINAMATH_CALUDE_responses_needed_l1179_117955

/-- Given a 65% response rate and 461.54 questionnaires mailed, prove that 300 responses are needed -/
theorem responses_needed (response_rate : ℝ) (questionnaires_mailed : ℝ) : 
  response_rate = 0.65 → 
  questionnaires_mailed = 461.54 → 
  ⌊response_rate * questionnaires_mailed⌋ = 300 := by
sorry

end NUMINAMATH_CALUDE_responses_needed_l1179_117955


namespace NUMINAMATH_CALUDE_cubic_equation_root_sum_l1179_117922

/-- Given a cubic equation with roots a, b, c and parameter k, prove that k = 5 -/
theorem cubic_equation_root_sum (k : ℝ) (a b c : ℝ) : 
  (∀ x : ℝ, x^3 - (k+1)*x^2 + k*x + 12 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  (a - 2)^3 + (b - 2)^3 + (c - 2)^3 = -18 →
  k = 5 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_sum_l1179_117922


namespace NUMINAMATH_CALUDE_alloy_price_calculation_l1179_117918

/-- Calculates the price of an alloy per kg given the prices of two metals and their mixing ratio -/
theorem alloy_price_calculation (price_a price_b : ℚ) (ratio : ℚ) :
  price_a = 68 →
  price_b = 96 →
  ratio = 3 →
  (ratio * price_a + price_b) / (ratio + 1) = 75 :=
by sorry

end NUMINAMATH_CALUDE_alloy_price_calculation_l1179_117918


namespace NUMINAMATH_CALUDE_yellow_candy_percentage_l1179_117933

theorem yellow_candy_percentage :
  ∀ (r b y : ℝ),
  r + b + y = 1 →
  y = 1.14 * b →
  r = 0.86 * b →
  y = 0.38 :=
by
  sorry

end NUMINAMATH_CALUDE_yellow_candy_percentage_l1179_117933


namespace NUMINAMATH_CALUDE_lcm_18_30_l1179_117954

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_30_l1179_117954


namespace NUMINAMATH_CALUDE_edge_sum_greater_than_3d_l1179_117984

-- Define a convex polyhedron
structure ConvexPolyhedron where
  vertices : Set (Fin 3 → ℝ)
  edges : Set (Fin 2 → Fin 3 → ℝ)
  is_convex : Bool

-- Define the maximum distance between vertices
def max_distance (p : ConvexPolyhedron) : ℝ :=
  sorry

-- Define the sum of edge lengths
def sum_edge_lengths (p : ConvexPolyhedron) : ℝ :=
  sorry

-- The theorem to prove
theorem edge_sum_greater_than_3d (p : ConvexPolyhedron) :
  p.is_convex → sum_edge_lengths p > 3 * max_distance p :=
by sorry

end NUMINAMATH_CALUDE_edge_sum_greater_than_3d_l1179_117984


namespace NUMINAMATH_CALUDE_equation_solution_polynomial_expansion_l1179_117943

-- Part 1: Equation solution
theorem equation_solution :
  {x : ℝ | 9 * (x - 3)^2 - 121 = 0} = {20/3, -2/3} := by sorry

-- Part 2: Polynomial expansion
theorem polynomial_expansion (x y : ℝ) :
  (x - 2*y) * (x^2 + 2*x*y + 4*y^2) = x^3 - 8*y^3 := by sorry

end NUMINAMATH_CALUDE_equation_solution_polynomial_expansion_l1179_117943


namespace NUMINAMATH_CALUDE_triangle_inequality_l1179_117931

/-- Given an acute triangle ABC with circumradius 1, 
    prove that the sum of the ratios of each side to (1 - sine of its opposite angle) 
    is greater than or equal to 18 + 12√3 -/
theorem triangle_inequality (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  A < π/2 ∧ B < π/2 ∧ C < π/2 ∧
  a = 2 * Real.sin A ∧
  b = 2 * Real.sin B ∧
  c = 2 * Real.sin C →
  (a / (1 - Real.sin A)) + (b / (1 - Real.sin B)) + (c / (1 - Real.sin C)) ≥ 18 + 12 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1179_117931


namespace NUMINAMATH_CALUDE_bicycle_trip_speed_l1179_117970

/-- Proves that given a 12-mile trip divided into three equal parts, each taking 15 minutes,
    with speeds of 16 mph and 12 mph for the first two parts respectively,
    the speed for the last part must be 16 mph. -/
theorem bicycle_trip_speed (total_distance : ℝ) (part_time : ℝ) (speed1 speed2 : ℝ) :
  total_distance = 12 →
  part_time = 0.25 →
  speed1 = 16 →
  speed2 = 12 →
  (speed1 * part_time + speed2 * part_time + 4) = total_distance →
  4 / part_time = 16 := by
  sorry


end NUMINAMATH_CALUDE_bicycle_trip_speed_l1179_117970


namespace NUMINAMATH_CALUDE_problem_statement_l1179_117974

-- Define the propositions
def universal_prop := ∀ x : ℝ, 2*x + 5 > 0
def equation_prop := ∀ x : ℝ, x^2 + 5*x = 6

-- Define logical variables
variable (p q : Prop)

-- Theorem statement
theorem problem_statement :
  (universal_prop) ∧
  (¬(equation_prop) ≠ ∃ x : ℝ, x^2 + 5*x ≠ 6) ∧
  (∃ x y : ℝ, |x| = |y| ∧ x ≠ y) ∧
  ((¬(p ∨ q)) → (¬p ∧ ¬q)) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l1179_117974


namespace NUMINAMATH_CALUDE_q_zero_at_two_two_l1179_117920

def q (b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ : ℝ) (x y : ℝ) : ℝ :=
  b₀ + b₁*x + b₂*y + b₃*x^2 + b₄*x*y + b₅*y^2 + b₆*x^3 + b₇*y^3 + b₈*x^4 + b₉*y^4

theorem q_zero_at_two_two 
  (b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ : ℝ) 
  (h₀ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 0 = 0)
  (h₁ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 1 0 = 0)
  (h₂ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ (-1) 0 = 0)
  (h₃ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 1 = 0)
  (h₄ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 (-1) = 0)
  (h₅ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 1 1 = 0)
  (h₆ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ (-1) (-1) = 0)
  (h₇ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 2 0 = 0)
  (h₈ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 2 = 0) :
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 2 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_q_zero_at_two_two_l1179_117920


namespace NUMINAMATH_CALUDE_juans_speed_l1179_117949

/-- Given a distance of 800 miles traveled in 80.0 hours, prove that the speed is 10 miles per hour -/
theorem juans_speed (distance : ℝ) (time : ℝ) (h1 : distance = 800) (h2 : time = 80) :
  distance / time = 10 := by
  sorry

end NUMINAMATH_CALUDE_juans_speed_l1179_117949


namespace NUMINAMATH_CALUDE_seventh_term_is_five_l1179_117969

/-- A geometric sequence with given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)
  a_3_eq_1 : a 3 = 1
  a_11_eq_25 : a 11 = 25

/-- The 7th term of the geometric sequence is 5 -/
theorem seventh_term_is_five (seq : GeometricSequence) : seq.a 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_five_l1179_117969


namespace NUMINAMATH_CALUDE_sum_vector_magnitude_l1179_117923

/-- Given planar vectors a and b satisfying specific conditions, 
    prove that the magnitude of their sum is 5. -/
theorem sum_vector_magnitude (a b : ℝ × ℝ) : 
  (a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) = 3) →
  (a = (1/2, Real.sqrt 3/2)) →
  (Real.sqrt (b.1^2 + b.2^2) = 2 * Real.sqrt 5) →
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_vector_magnitude_l1179_117923


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1179_117961

theorem inequality_solution_set (x : ℝ) : 
  (2 * x - 2) / (x^2 - 5 * x + 6) ≤ 3 ↔ 
  (5/3 < x ∧ x ≤ 2) ∨ (3 ≤ x ∧ x ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1179_117961


namespace NUMINAMATH_CALUDE_disjunction_truth_l1179_117948

theorem disjunction_truth (p q : Prop) (hp : p) (hq : ¬q) : p ∨ q := by
  sorry

end NUMINAMATH_CALUDE_disjunction_truth_l1179_117948


namespace NUMINAMATH_CALUDE_y_derivative_l1179_117996

noncomputable def y (x : ℝ) : ℝ := Real.cos (Real.log 13) - (1 / 44) * (Real.cos (22 * x))^2 / Real.sin (44 * x)

theorem y_derivative (x : ℝ) (h : Real.sin (22 * x) ≠ 0) : 
  deriv y x = 1 / (4 * (Real.sin (22 * x))^2) := by sorry

end NUMINAMATH_CALUDE_y_derivative_l1179_117996


namespace NUMINAMATH_CALUDE_sarah_apple_slices_l1179_117981

/-- Given a number of boxes of apples, apples per box, and slices per apple,
    calculate the total number of apple slices -/
def total_apple_slices (boxes : ℕ) (apples_per_box : ℕ) (slices_per_apple : ℕ) : ℕ :=
  boxes * apples_per_box * slices_per_apple

/-- Theorem: Sarah has 392 apple slices -/
theorem sarah_apple_slices :
  total_apple_slices 7 7 8 = 392 := by
  sorry

end NUMINAMATH_CALUDE_sarah_apple_slices_l1179_117981


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1179_117912

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n and common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, S n = n * (2 * (a 1) + (n - 1) * d) / 2

/-- The theorem to be proved -/
theorem arithmetic_sequence_difference (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a S d →
  (S 2017 / 2017) - (S 17 / 17) = 100 →
  d = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1179_117912


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l1179_117930

theorem quadratic_root_zero (m : ℝ) :
  (∃ x : ℝ, (m - 1) * x^2 + x + m^2 - 1 = 0) ∧
  ((m - 1) * 0^2 + 0 + m^2 - 1 = 0) →
  m = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l1179_117930


namespace NUMINAMATH_CALUDE_expression_nonnegative_iff_l1179_117950

theorem expression_nonnegative_iff (x : ℝ) : 
  (3*x - 12*x^2 + 48*x^3) / (27 - x^3) ≥ 0 ↔ x ∈ Set.Ici 0 ∩ Set.Iio 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_nonnegative_iff_l1179_117950


namespace NUMINAMATH_CALUDE_a_range_l1179_117994

theorem a_range (a : ℝ) : 
  (∀ x : ℝ, x ∈ {x | x^2 - 2*x + a > 0} ↔ x^2 - 2*x + a > 0) →
  1 ∉ {x : ℝ | x^2 - 2*x + a > 0} →
  a ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_a_range_l1179_117994


namespace NUMINAMATH_CALUDE_factorization_proof_l1179_117999

theorem factorization_proof (a : ℝ) : 
  45 * a^2 + 135 * a + 90 * a^3 = 45 * a * (90 * a^2 + a + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1179_117999


namespace NUMINAMATH_CALUDE_fraction_ordering_l1179_117956

theorem fraction_ordering : 
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 14
  let c := (6 : ℚ) / 17
  let d := b - (1 : ℚ) / 56
  d < c ∧ c < a :=
by sorry

end NUMINAMATH_CALUDE_fraction_ordering_l1179_117956


namespace NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l1179_117972

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Defines that a line is contained in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines that a line is perpendicular to a plane -/
def line_perp_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines that two planes are perpendicular -/
def planes_perpendicular (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Theorem: If a line contained in a plane is perpendicular to another plane,
    then the two planes are perpendicular -/
theorem line_perp_plane_implies_planes_perp
  (l : Line3D) (α β : Plane3D)
  (h1 : line_in_plane l α)
  (h2 : line_perp_plane l β) :
  planes_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l1179_117972


namespace NUMINAMATH_CALUDE_existence_of_factors_l1179_117971

theorem existence_of_factors : ∃ (a b c d : ℕ), 
  (10 ≤ a ∧ a < 100) ∧ 
  (10 ≤ b ∧ b < 100) ∧ 
  (10 ≤ c ∧ c < 100) ∧ 
  (10 ≤ d ∧ d < 100) ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a * a * b * c * d = 2016000 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_factors_l1179_117971


namespace NUMINAMATH_CALUDE_smallest_k_sum_squares_multiple_180_sum_squares_360_multiple_180_smallest_k_is_360_l1179_117987

theorem smallest_k_sum_squares_multiple_180 :
  ∀ k : ℕ+, (k.val * (k.val + 1) * (2 * k.val + 1)) % 1080 = 0 → k.val ≥ 360 :=
by sorry

theorem sum_squares_360_multiple_180 :
  (360 * 361 * 721) % 1080 = 0 :=
by sorry

theorem smallest_k_is_360 :
  ∃! k : ℕ+, k.val = 360 ∧
    (∀ m : ℕ+, (m.val * (m.val + 1) * (2 * m.val + 1)) % 1080 = 0 → k ≤ m) ∧
    (k.val * (k.val + 1) * (2 * k.val + 1)) % 1080 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_sum_squares_multiple_180_sum_squares_360_multiple_180_smallest_k_is_360_l1179_117987


namespace NUMINAMATH_CALUDE_derivative_of_x_plus_exp_l1179_117944

/-- The derivative of f(x) = x + e^x is f'(x) = 1 + e^x -/
theorem derivative_of_x_plus_exp (x : ℝ) :
  deriv (fun x => x + Real.exp x) x = 1 + Real.exp x := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_x_plus_exp_l1179_117944


namespace NUMINAMATH_CALUDE_dragon_lion_equivalence_l1179_117914

theorem dragon_lion_equivalence (P Q : Prop) : 
  ((P → Q) ↔ (¬Q → ¬P)) ∧ ((P → Q) ↔ (¬P ∨ Q)) := by sorry

end NUMINAMATH_CALUDE_dragon_lion_equivalence_l1179_117914


namespace NUMINAMATH_CALUDE_seven_thirteenths_repeating_block_length_l1179_117901

/-- The least number of digits in a repeating block of the decimal expansion of 7/13 -/
def least_repeating_block_length : ℕ := 6

/-- Theorem stating that the least number of digits in a repeating block of 7/13 is 6 -/
theorem seven_thirteenths_repeating_block_length :
  least_repeating_block_length = 6 := by sorry

end NUMINAMATH_CALUDE_seven_thirteenths_repeating_block_length_l1179_117901


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1179_117903

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x + 3

-- State the theorem
theorem quadratic_minimum :
  ∃ (x_min y_min : ℝ), x_min = -1 ∧ y_min = 1 ∧
  ∀ x, f x ≥ f x_min := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1179_117903


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1179_117952

theorem equal_roots_quadratic (q : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + q = 0 ∧ 
   ∀ y : ℝ, y^2 - 3*y + q = 0 → y = x) ↔ 
  q = 9/4 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1179_117952


namespace NUMINAMATH_CALUDE_childrens_home_toddlers_l1179_117962

theorem childrens_home_toddlers (total : ℕ) (newborns : ℕ) :
  total = 40 →
  newborns = 4 →
  ∃ (toddlers teenagers : ℕ),
    toddlers + teenagers + newborns = total ∧
    teenagers = 5 * toddlers ∧
    toddlers = 6 :=
by sorry

end NUMINAMATH_CALUDE_childrens_home_toddlers_l1179_117962


namespace NUMINAMATH_CALUDE_book_division_l1179_117936

theorem book_division (total_books : ℕ) (first_division : ℕ) (second_division : ℕ) (books_per_category : ℕ) 
  (h1 : total_books = 1200)
  (h2 : first_division = 3)
  (h3 : second_division = 4)
  (h4 : books_per_category = 15) :
  (total_books / first_division / second_division / books_per_category) * 
  first_division * second_division = 84 :=
by sorry

end NUMINAMATH_CALUDE_book_division_l1179_117936


namespace NUMINAMATH_CALUDE_impossible_all_white_l1179_117932

-- Define the grid as a function from coordinates to colors
def Grid := Fin 8 → Fin 8 → Bool

-- Define the initial grid configuration
def initial_grid : Grid :=
  fun i j => (i = 0 ∧ j = 0) ∨ (i = 0 ∧ j = 7) ∨ (i = 7 ∧ j = 0) ∨ (i = 7 ∧ j = 7)

-- Define a row flip operation
def flip_row (g : Grid) (row : Fin 8) : Grid :=
  fun i j => if i = row then !g i j else g i j

-- Define a column flip operation
def flip_column (g : Grid) (col : Fin 8) : Grid :=
  fun i j => if j = col then !g i j else g i j

-- Define a predicate for an all-white grid
def all_white (g : Grid) : Prop :=
  ∀ i j, g i j = false

-- Theorem: It's impossible to achieve an all-white configuration
theorem impossible_all_white :
  ¬ ∃ (flips : List (Sum (Fin 8) (Fin 8))),
    all_white (flips.foldl (fun g flip => 
      match flip with
      | Sum.inl row => flip_row g row
      | Sum.inr col => flip_column g col
    ) initial_grid) :=
  sorry


end NUMINAMATH_CALUDE_impossible_all_white_l1179_117932


namespace NUMINAMATH_CALUDE_domain_transformation_l1179_117997

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the domain of f(x+1)
def domain_f_x_plus_1 (f : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | -2 < x ∧ x < -1}

-- Define the domain of f(2x+1)
def domain_f_2x_plus_1 (f : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | -1 < x ∧ x < -1/2}

-- Theorem statement
theorem domain_transformation (f : ℝ → ℝ) :
  (∀ x, x ∈ domain_f_x_plus_1 f ↔ f (x + 1) ∈ Set.univ) →
  (∀ x, x ∈ domain_f_2x_plus_1 f ↔ f (2*x + 1) ∈ Set.univ) :=
sorry

end NUMINAMATH_CALUDE_domain_transformation_l1179_117997


namespace NUMINAMATH_CALUDE_vote_count_proof_l1179_117979

theorem vote_count_proof (total votes_against votes_in_favor : ℕ) 
  (h1 : votes_in_favor = votes_against + 68)
  (h2 : votes_against = (40 : ℕ) * total / 100)
  (h3 : total = votes_in_favor + votes_against) :
  total = 340 :=
sorry

end NUMINAMATH_CALUDE_vote_count_proof_l1179_117979


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_log_sum_l1179_117993

theorem arithmetic_geometric_mean_log_sum (a b c x y z m : ℝ) 
  (hb : b = (a + c) / 2)
  (hy : y^2 = x * z)
  (hx : x > 0)
  (hy_pos : y > 0)
  (hz : z > 0)
  (hm : m > 0 ∧ m ≠ 1) :
  (b - c) * (Real.log x / Real.log m) + 
  (c - a) * (Real.log y / Real.log m) + 
  (a - b) * (Real.log z / Real.log m) = 0 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_log_sum_l1179_117993


namespace NUMINAMATH_CALUDE_fresh_grape_water_content_l1179_117926

/-- The percentage of water in raisins -/
def raisin_water_percentage : ℝ := 25

/-- The weight of fresh grapes used -/
def fresh_grape_weight : ℝ := 100

/-- The weight of raisins produced -/
def raisin_weight : ℝ := 20

/-- The percentage of water in fresh grapes -/
def fresh_grape_water_percentage : ℝ := 85

theorem fresh_grape_water_content :
  fresh_grape_water_percentage = 85 :=
sorry

end NUMINAMATH_CALUDE_fresh_grape_water_content_l1179_117926


namespace NUMINAMATH_CALUDE_three_digit_number_sum_property_l1179_117916

theorem three_digit_number_sum_property :
  ∃! (N : ℕ), ∃ (a b c : ℕ),
    (100 ≤ N) ∧ (N < 1000) ∧
    (1 ≤ a) ∧ (a ≤ 9) ∧
    (0 ≤ b) ∧ (b ≤ 9) ∧
    (0 ≤ c) ∧ (c ≤ 9) ∧
    (N = 100 * a + 10 * b + c) ∧
    (N = 11 * (a + b + c)) ∧
    (N = 198) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_sum_property_l1179_117916


namespace NUMINAMATH_CALUDE_max_erasers_purchase_l1179_117978

def pen_cost : ℕ := 3
def pencil_cost : ℕ := 4
def eraser_cost : ℕ := 8
def total_budget : ℕ := 60

def is_valid_purchase (pens pencils erasers : ℕ) : Prop :=
  pens ≥ 1 ∧ pencils ≥ 1 ∧ erasers ≥ 1 ∧
  pens * pen_cost + pencils * pencil_cost + erasers * eraser_cost = total_budget

theorem max_erasers_purchase :
  ∃ (pens pencils : ℕ), is_valid_purchase pens pencils 5 ∧
  ∀ (p n e : ℕ), is_valid_purchase p n e → e ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_erasers_purchase_l1179_117978


namespace NUMINAMATH_CALUDE_angle_in_linear_pair_l1179_117980

/-- 
Given a line segment AB with three angles:
- ACD = 90°
- ECB = 52°
- DCE = x°
Prove that x = 38°
-/
theorem angle_in_linear_pair (x : ℝ) : 
  90 + x + 52 = 180 → x = 38 := by sorry

end NUMINAMATH_CALUDE_angle_in_linear_pair_l1179_117980


namespace NUMINAMATH_CALUDE_second_team_made_131_pieces_l1179_117934

/-- The number of fish fillet pieces made by the second team -/
def second_team_pieces (total : ℕ) (first_team : ℕ) (third_team : ℕ) : ℕ :=
  total - (first_team + third_team)

/-- Theorem stating that the second team made 131 pieces of fish fillets -/
theorem second_team_made_131_pieces : 
  second_team_pieces 500 189 180 = 131 := by
  sorry

end NUMINAMATH_CALUDE_second_team_made_131_pieces_l1179_117934


namespace NUMINAMATH_CALUDE_expression_value_when_b_is_negative_one_l1179_117991

theorem expression_value_when_b_is_negative_one :
  let b : ℚ := -1
  let expr := (3 * b⁻¹ + (2 * b⁻¹) / 3) / b
  expr = 11 / 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_when_b_is_negative_one_l1179_117991


namespace NUMINAMATH_CALUDE_min_value_F_range_of_m_l1179_117977

noncomputable section

def f (x : ℝ) := x * Real.exp x
def g (x : ℝ) := (1/2) * x^2 + x
def F (x : ℝ) := f x + g x

-- Part 1
theorem min_value_F :
  ∃ (x_min : ℝ), ∀ (x : ℝ), F x_min ≤ F x ∧ F x_min = -1 - 1/Real.exp 1 :=
sorry

-- Part 2
theorem range_of_m (m : ℝ) :
  (∀ (x₁ x₂ : ℝ), -1 ≤ x₂ ∧ x₂ < x₁ →
    m * (f x₁ - f x₂) > g x₁ - g x₂) ↔ m ≥ Real.exp 1 :=
sorry

end

end NUMINAMATH_CALUDE_min_value_F_range_of_m_l1179_117977


namespace NUMINAMATH_CALUDE_square_roots_theorem_l1179_117902

theorem square_roots_theorem (a x : ℝ) : 
  x > 0 ∧ (2*a - 1)^2 = x ∧ (-a + 2)^2 = x → x = 9 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_theorem_l1179_117902


namespace NUMINAMATH_CALUDE_x_twelfth_power_l1179_117937

theorem x_twelfth_power (x : ℝ) (h : x + 1/x = 2 * Real.sqrt 2) : x^12 = 46656 := by
  sorry

end NUMINAMATH_CALUDE_x_twelfth_power_l1179_117937


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1179_117992

theorem sum_of_three_numbers (a b c : ℝ) 
  (eq1 : 2 * a + b = 46)
  (eq2 : b + 2 * c = 53)
  (eq3 : 2 * c + a = 29) :
  a + b + c = 146.5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1179_117992


namespace NUMINAMATH_CALUDE_initial_white_lights_equal_total_colored_lights_l1179_117907

/-- The number of white lights Malcolm had initially -/
def initialWhiteLights : ℕ := sorry

/-- The number of red lights Malcolm bought -/
def redLights : ℕ := 12

/-- The number of blue lights Malcolm bought -/
def blueLights : ℕ := 3 * redLights

/-- The number of green lights Malcolm bought -/
def greenLights : ℕ := 6

/-- The number of colored lights Malcolm still needs to buy -/
def remainingLights : ℕ := 5

/-- Theorem stating that the initial number of white lights equals the total number of colored lights -/
theorem initial_white_lights_equal_total_colored_lights :
  initialWhiteLights = redLights + blueLights + greenLights + remainingLights := by sorry

end NUMINAMATH_CALUDE_initial_white_lights_equal_total_colored_lights_l1179_117907


namespace NUMINAMATH_CALUDE_definite_integral_x_plus_sin_x_l1179_117915

open Real MeasureTheory

theorem definite_integral_x_plus_sin_x : ∫ x in (-1)..1, (x + Real.sin x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_x_plus_sin_x_l1179_117915


namespace NUMINAMATH_CALUDE_water_for_lemonade_l1179_117965

/-- Represents the ratio of water to lemon juice in the lemonade mixture -/
def water_to_juice_ratio : ℚ := 7 / 1

/-- Represents the number of quarts in a gallon -/
def quarts_per_gallon : ℚ := 4

/-- Calculates the amount of water needed to make one gallon of lemonade -/
def water_needed (ratio : ℚ) (quarts_in_gallon : ℚ) : ℚ :=
  (ratio * quarts_in_gallon) / (ratio + 1)

/-- Theorem stating that the amount of water needed for one gallon of lemonade is 7/2 quarts -/
theorem water_for_lemonade :
  water_needed water_to_juice_ratio quarts_per_gallon = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_water_for_lemonade_l1179_117965


namespace NUMINAMATH_CALUDE_probability_x_gt_9y_l1179_117957

/-- The probability that a randomly chosen point (x,y) from a rectangle
    with vertices (0,0), (2017,0), (2017,2018), and (0,2018) satisfies x > 9y -/
theorem probability_x_gt_9y : Real := by
  -- Define the rectangle
  let rectangle_width : ℕ := 2017
  let rectangle_height : ℕ := 2018

  -- Define the condition x > 9y
  let condition (x y : Real) : Prop := x > 9 * y

  -- Define the probability
  let probability : Real := 2017 / 36324

  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_probability_x_gt_9y_l1179_117957


namespace NUMINAMATH_CALUDE_function_domain_implies_k_range_l1179_117986

/-- Given a function f(x) = √(kx² + kx + 3) with domain ℝ, k must be in [0, 12] -/
theorem function_domain_implies_k_range (k : ℝ) : 
  (∀ x, ∃ y, y = Real.sqrt (k * x^2 + k * x + 3)) → 0 ≤ k ∧ k ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_function_domain_implies_k_range_l1179_117986


namespace NUMINAMATH_CALUDE_second_caterer_cheaper_at_34_l1179_117946

def first_caterer_cost (n : ℕ) : ℝ := 50 + 18 * n

def second_caterer_cost (n : ℕ) : ℝ :=
  if n ≥ 30 then 150 + 15 * n else 180 + 15 * n

theorem second_caterer_cheaper_at_34 :
  (∀ n : ℕ, n ≥ 34 → second_caterer_cost n < first_caterer_cost n) ∧
  (∀ n : ℕ, n < 34 → second_caterer_cost n ≥ first_caterer_cost n) :=
sorry

end NUMINAMATH_CALUDE_second_caterer_cheaper_at_34_l1179_117946


namespace NUMINAMATH_CALUDE_series_sum_convergence_l1179_117938

open Real
open BigOperators

/-- The sum of the infinite series ∑(n=1 to ∞) (3n - 2) / (n(n + 1)(n + 2)) converges to 5/6 -/
theorem series_sum_convergence :
  ∑' n : ℕ, (3 * n - 2 : ℝ) / (n * (n + 1) * (n + 2)) = 5/6 := by sorry

end NUMINAMATH_CALUDE_series_sum_convergence_l1179_117938


namespace NUMINAMATH_CALUDE_melanie_initial_plums_l1179_117924

/-- The number of plums Melanie initially picked -/
def initial_plums : ℕ := sorry

/-- The number of plums Melanie gave to Sam -/
def plums_given : ℕ := 3

/-- The number of plums Melanie has left -/
def plums_left : ℕ := 4

/-- Theorem: Melanie initially picked 7 plums -/
theorem melanie_initial_plums : initial_plums = 7 := by
  sorry

end NUMINAMATH_CALUDE_melanie_initial_plums_l1179_117924


namespace NUMINAMATH_CALUDE_no_prime_solution_l1179_117911

def base_p_to_decimal (digits : List Nat) (p : Nat) : Nat :=
  digits.foldr (fun d acc => d + p * acc) 0

theorem no_prime_solution :
  ¬∃ p : Nat, Nat.Prime p ∧
    (base_p_to_decimal [4, 1, 0, 1] p +
     base_p_to_decimal [2, 0, 5] p +
     base_p_to_decimal [7, 1, 2] p +
     base_p_to_decimal [1, 3, 2] p +
     base_p_to_decimal [2, 1] p =
     base_p_to_decimal [4, 5, 2] p +
     base_p_to_decimal [7, 4, 5] p +
     base_p_to_decimal [5, 7, 6] p) :=
by sorry

end NUMINAMATH_CALUDE_no_prime_solution_l1179_117911


namespace NUMINAMATH_CALUDE_eighth_number_is_four_l1179_117925

/-- A sequence of 12 numbers satisfying the given conditions -/
def SpecialSequence : Type := 
  {s : Fin 12 → ℕ // s 0 = 5 ∧ s 11 = 10 ∧ ∀ i, i < 10 → s i + s (i + 1) + s (i + 2) = 19}

/-- The theorem stating that the 8th number (index 7) in the sequence is 4 -/
theorem eighth_number_is_four (s : SpecialSequence) : s.val 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_eighth_number_is_four_l1179_117925


namespace NUMINAMATH_CALUDE_triangle_side_inequality_l1179_117904

/-- For any triangle with side lengths a, b, and c, 
    the sum of squares of the sides is less than 
    twice the sum of the products of pairs of sides. -/
theorem triangle_side_inequality (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a^2 + b^2 + c^2 < 2*(a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_inequality_l1179_117904


namespace NUMINAMATH_CALUDE_increase_by_percentage_seventy_five_increased_by_150_percent_l1179_117910

theorem increase_by_percentage (x : ℝ) (p : ℝ) :
  x + x * (p / 100) = x * (1 + p / 100) := by sorry

theorem seventy_five_increased_by_150_percent :
  75 + 75 * (150 / 100) = 187.5 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_seventy_five_increased_by_150_percent_l1179_117910


namespace NUMINAMATH_CALUDE_color_copies_comparison_l1179_117942

/-- The cost per color copy at print shop X -/
def cost_x : ℚ := 120 / 100

/-- The cost per color copy at print shop Y -/
def cost_y : ℚ := 170 / 100

/-- The difference in total cost between print shops Y and X -/
def cost_difference : ℚ := 35

/-- The number of color copies being compared -/
def n : ℚ := 70

theorem color_copies_comparison :
  cost_y * n = cost_x * n + cost_difference :=
by sorry

end NUMINAMATH_CALUDE_color_copies_comparison_l1179_117942


namespace NUMINAMATH_CALUDE_union_determines_k_l1179_117985

def A (k : ℕ) : Set ℕ := {1, 2, k}
def B : Set ℕ := {2, 5}

theorem union_determines_k (k : ℕ) : A k ∪ B = {1, 2, 3, 5} → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_union_determines_k_l1179_117985


namespace NUMINAMATH_CALUDE_last_two_nonzero_digits_of_80_factorial_l1179_117958

theorem last_two_nonzero_digits_of_80_factorial (n : ℕ) : n = 80 →
  ∃ k : ℕ, n.factorial = 100 * k + 48 ∧ k % 10 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_last_two_nonzero_digits_of_80_factorial_l1179_117958


namespace NUMINAMATH_CALUDE_min_value_of_f_l1179_117928

def f (x : ℝ) : ℝ := x^2 - 6*x + 9

theorem min_value_of_f :
  ∀ x : ℝ, f x ≥ f 3 := by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1179_117928


namespace NUMINAMATH_CALUDE_gcd_sum_and_sum_of_squares_l1179_117921

theorem gcd_sum_and_sum_of_squares (a b : ℤ) : 
  Int.gcd a b = 1 → Int.gcd (a + b) (a^2 + b^2) = 1 ∨ Int.gcd (a + b) (a^2 + b^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_sum_and_sum_of_squares_l1179_117921


namespace NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l1179_117998

theorem smallest_positive_integer_congruence :
  ∃ y : ℕ+, 
    (∀ z : ℕ+, 58 * z + 14 ≡ 4 [ZMOD 36] → y ≤ z) ∧ 
    (58 * y + 14 ≡ 4 [ZMOD 36]) ∧
    y = 26 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l1179_117998


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l1179_117982

theorem equilateral_triangle_area_perimeter_ratio :
  let s : ℝ := 10
  let altitude : ℝ := s * Real.sqrt 3 / 2
  let area : ℝ := s * altitude / 2
  let perimeter : ℝ := 3 * s
  area / perimeter = 5 * Real.sqrt 3 / 6 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l1179_117982


namespace NUMINAMATH_CALUDE_petya_sequence_l1179_117976

theorem petya_sequence (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (eq1 : (a + b) * (a + c) = a)
  (eq2 : (a + b) * (b + c) = b)
  (eq3 : (a + c) * (b + c) = c) :
  a = 1/4 ∧ b = 1/4 ∧ c = 1/4 := by
sorry

end NUMINAMATH_CALUDE_petya_sequence_l1179_117976


namespace NUMINAMATH_CALUDE_equal_points_iff_odd_participants_l1179_117988

/-- A round-robin chess tournament with no draws -/
structure ChessTournament where
  n : ℕ  -- number of participants
  no_draws : Bool

/-- The total number of games played in a round-robin tournament -/
def total_games (t : ChessTournament) : ℕ := t.n * (t.n - 1) / 2

/-- The total number of points scored in the tournament -/
def total_points (t : ChessTournament) : ℕ := total_games t

/-- Whether all participants have the same number of points -/
def all_equal_points (t : ChessTournament) : Prop :=
  ∃ k : ℕ, k * t.n = total_points t

/-- The main theorem: all participants can have equal points iff the number of participants is odd -/
theorem equal_points_iff_odd_participants (t : ChessTournament) (h : t.no_draws = true) :
  all_equal_points t ↔ Odd t.n :=
sorry

end NUMINAMATH_CALUDE_equal_points_iff_odd_participants_l1179_117988


namespace NUMINAMATH_CALUDE_tan_22_5_decomposition_l1179_117995

theorem tan_22_5_decomposition :
  ∃ (a b c : ℕ), 
    (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
    (a ≥ b) ∧ (b ≥ c) ∧
    (Real.tan (22.5 * π / 180) = Real.sqrt a - Real.sqrt b - c) ∧
    (a + b + c = 4) := by
  sorry

end NUMINAMATH_CALUDE_tan_22_5_decomposition_l1179_117995


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1179_117939

/-- Definition of a quadratic function -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function we want to prove is quadratic -/
def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x

/-- Theorem stating that f is a quadratic function -/
theorem f_is_quadratic : is_quadratic f := by sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l1179_117939


namespace NUMINAMATH_CALUDE_correct_mark_l1179_117960

theorem correct_mark (num_pupils : ℕ) (wrong_mark : ℕ) (correct_mark : ℕ) : 
  num_pupils = 44 →
  wrong_mark = 67 →
  (wrong_mark - correct_mark : ℚ) = num_pupils / 2 →
  correct_mark = 45 := by
sorry

end NUMINAMATH_CALUDE_correct_mark_l1179_117960


namespace NUMINAMATH_CALUDE_cloth_selling_amount_l1179_117951

/-- Calculates the total selling amount for cloth given the quantity, cost price, and loss per metre. -/
def totalSellingAmount (quantity : ℕ) (costPrice : ℕ) (lossPerMetre : ℕ) : ℕ :=
  quantity * (costPrice - lossPerMetre)

/-- Proves that the total selling amount for 200 metres of cloth with a cost price of 66 and a loss of 6 per metre is 12000. -/
theorem cloth_selling_amount :
  totalSellingAmount 200 66 6 = 12000 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_amount_l1179_117951


namespace NUMINAMATH_CALUDE_yellow_ball_percentage_l1179_117900

/-- Given the number of yellow and brown balls, calculate the percentage of yellow balls -/
theorem yellow_ball_percentage (yellow_balls brown_balls : ℕ) : 
  yellow_balls = 27 → brown_balls = 33 → 
  (yellow_balls : ℚ) / (yellow_balls + brown_balls : ℚ) * 100 = 45 := by
  sorry

end NUMINAMATH_CALUDE_yellow_ball_percentage_l1179_117900


namespace NUMINAMATH_CALUDE_max_value_of_f_l1179_117945

/-- The function f(x) = x³ - 3x -/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The maximum value of f(x) is 2 -/
theorem max_value_of_f : ∃ (M : ℝ), M = 2 ∧ ∀ (x : ℝ), f x ≤ M :=
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1179_117945


namespace NUMINAMATH_CALUDE_sum_of_bases_is_sixteen_l1179_117905

/-- Represents a repeating decimal in a given base -/
structure RepeatingDecimal (base : ℕ) where
  integerPart : ℕ
  repeatingPart : ℕ

/-- Given two bases and representations of G₁ and G₂ in those bases, proves their sum is 16 -/
theorem sum_of_bases_is_sixteen
  (S₁ S₂ : ℕ)
  (G₁_in_S₁ : RepeatingDecimal S₁)
  (G₂_in_S₁ : RepeatingDecimal S₁)
  (G₁_in_S₂ : RepeatingDecimal S₂)
  (G₂_in_S₂ : RepeatingDecimal S₂)
  (h₁ : G₁_in_S₁ = ⟨0, 45⟩)
  (h₂ : G₂_in_S₁ = ⟨0, 54⟩)
  (h₃ : G₁_in_S₂ = ⟨0, 14⟩)
  (h₄ : G₂_in_S₂ = ⟨0, 41⟩)
  : S₁ + S₂ = 16 :=
sorry

end NUMINAMATH_CALUDE_sum_of_bases_is_sixteen_l1179_117905


namespace NUMINAMATH_CALUDE_optimal_purchase_max_profit_l1179_117908

/-- Represents the types of multimedia --/
inductive MultimediaType
| A
| B

/-- Represents the cost and price of each type of multimedia --/
def cost_price (t : MultimediaType) : ℝ × ℝ :=
  match t with
  | MultimediaType.A => (3, 3.3)
  | MultimediaType.B => (2.4, 2.8)

/-- The total number of sets to purchase --/
def total_sets : ℕ := 50

/-- The total cost in million yuan --/
def total_cost : ℝ := 132

/-- Theorem for part 1 of the problem --/
theorem optimal_purchase :
  ∃ (a b : ℕ),
    a + b = total_sets ∧
    a * (cost_price MultimediaType.A).1 + b * (cost_price MultimediaType.B).1 = total_cost ∧
    a = 20 ∧ b = 30 := by sorry

/-- Function to calculate profit --/
def profit (a : ℕ) : ℝ :=
  let b := total_sets - a
  a * ((cost_price MultimediaType.A).2 - (cost_price MultimediaType.A).1) +
  b * ((cost_price MultimediaType.B).2 - (cost_price MultimediaType.B).1)

/-- Theorem for part 2 of the problem --/
theorem max_profit :
  ∃ (a : ℕ),
    10 < a ∧ a < 20 ∧
    (∀ m, 10 < m → m < 20 → profit m ≤ profit a) ∧
    a = 11 ∧ profit a = 18.9 := by sorry

end NUMINAMATH_CALUDE_optimal_purchase_max_profit_l1179_117908


namespace NUMINAMATH_CALUDE_incorrect_statement_l1179_117953

-- Define the concept of planes
variable (α β : Set (ℝ × ℝ × ℝ))

-- Define perpendicularity between planes
def perpendicular (p q : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define the concept of a line
def Line : Type := Set (ℝ × ℝ × ℝ)

-- Define perpendicularity between a line and a plane
def line_perp_plane (l : Line) (p : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define the intersection line of two planes
def intersection_line (p q : Set (ℝ × ℝ × ℝ)) : Line := sorry

-- Define a function to create a perpendicular line from a point to a line
def perp_line_to_line (point : ℝ × ℝ × ℝ) (l : Line) : Line := sorry

-- Theorem to be disproved
theorem incorrect_statement 
  (h1 : perpendicular α β)
  (point : ℝ × ℝ × ℝ)
  (h2 : point ∈ α) :
  line_perp_plane (perp_line_to_line point (intersection_line α β)) β := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_l1179_117953


namespace NUMINAMATH_CALUDE_unique_solution_l1179_117929

-- Define the equation
def equation (x a : ℝ) : Prop :=
  3 * x^2 + 2 * a * x - a^2 = Real.log ((x - a) / (2 * x))

-- Define the domain conditions
def domain_conditions (x a : ℝ) : Prop :=
  x - a > 0 ∧ 2 * x > 0

-- Theorem statement
theorem unique_solution (a : ℝ) (h : a ≠ 0) :
  ∃! x : ℝ, equation x a ∧ domain_conditions x a :=
by
  -- The unique solution is x = -a
  use -a
  sorry -- Proof omitted

end NUMINAMATH_CALUDE_unique_solution_l1179_117929


namespace NUMINAMATH_CALUDE_opposite_reciprocal_abs_neg_seven_l1179_117983

theorem opposite_reciprocal_abs_neg_seven :
  -(1 / |(-7 : ℤ)|) = -((1 : ℚ) / 7) := by sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_abs_neg_seven_l1179_117983


namespace NUMINAMATH_CALUDE_cubic_factorization_l1179_117967

theorem cubic_factorization (x : ℝ) : x^3 - 4*x = x*(x+2)*(x-2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1179_117967


namespace NUMINAMATH_CALUDE_sequence_is_geometric_progression_l1179_117964

theorem sequence_is_geometric_progression (a : ℕ+ → ℚ) (S : ℕ+ → ℚ) 
  (h : ∀ n : ℕ+, S n = (1 : ℚ) / 3 * (a n - 1)) :
  a 1 = -(1 : ℚ) / 2 ∧ 
  a 2 = (1 : ℚ) / 4 ∧ 
  (∀ n : ℕ+, n > 1 → a n / a (n - 1) = -(1 : ℚ) / 2) := by
  sorry

end NUMINAMATH_CALUDE_sequence_is_geometric_progression_l1179_117964
