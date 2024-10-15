import Mathlib

namespace NUMINAMATH_CALUDE_output_is_76_l3070_307093

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 ≤ 30 then
    (step1 + 10)
  else
    ((step1 - 7) * 2)

theorem output_is_76 : function_machine 15 = 76 := by
  sorry

end NUMINAMATH_CALUDE_output_is_76_l3070_307093


namespace NUMINAMATH_CALUDE_puzzle_solution_l3070_307084

theorem puzzle_solution : 
  ∀ (S T U K : ℕ),
  (S ≠ T ∧ S ≠ U ∧ S ≠ K ∧ T ≠ U ∧ T ≠ K ∧ U ≠ K) →
  (100 ≤ T * 100 + U * 10 + K ∧ T * 100 + U * 10 + K < 1000) →
  (1000 ≤ S * 1000 + T * 100 + U * 10 + K ∧ S * 1000 + T * 100 + U * 10 + K < 10000) →
  (5 * (T * 100 + U * 10 + K) = S * 1000 + T * 100 + U * 10 + K) →
  (T * 100 + U * 10 + K = 250 ∨ T * 100 + U * 10 + K = 750) := by
sorry

end NUMINAMATH_CALUDE_puzzle_solution_l3070_307084


namespace NUMINAMATH_CALUDE_derivative_independent_of_function_value_l3070_307005

variable (f : ℝ → ℝ)
variable (x₀ : ℝ)

theorem derivative_independent_of_function_value :
  ∃ (g : ℝ → ℝ), g x₀ ≠ f x₀ ∧ HasDerivAt g (deriv f x₀) x₀ :=
sorry

end NUMINAMATH_CALUDE_derivative_independent_of_function_value_l3070_307005


namespace NUMINAMATH_CALUDE_total_rectangles_is_176_l3070_307059

/-- The number of gray cells in the picture -/
def total_gray_cells : ℕ := 40

/-- The number of blue cells (more frequent gray cells) -/
def blue_cells : ℕ := 36

/-- The number of red cells (less frequent gray cells) -/
def red_cells : ℕ := 4

/-- The number of rectangles containing each blue cell -/
def rectangles_per_blue_cell : ℕ := 4

/-- The number of rectangles containing each red cell -/
def rectangles_per_red_cell : ℕ := 8

/-- The total number of rectangles containing exactly one gray cell -/
def total_rectangles : ℕ := blue_cells * rectangles_per_blue_cell + red_cells * rectangles_per_red_cell

theorem total_rectangles_is_176 : total_rectangles = 176 := by
  sorry

end NUMINAMATH_CALUDE_total_rectangles_is_176_l3070_307059


namespace NUMINAMATH_CALUDE_positive_root_negative_root_zero_root_l3070_307096

-- Define the equation
def equation (a b x : ℝ) : Prop := b + x = 4 * x + a

-- Theorem for positive root
theorem positive_root (a b : ℝ) : 
  b > a → ∃ x : ℝ, x > 0 ∧ equation a b x := by sorry

-- Theorem for negative root
theorem negative_root (a b : ℝ) : 
  b < a → ∃ x : ℝ, x < 0 ∧ equation a b x := by sorry

-- Theorem for zero root
theorem zero_root (a b : ℝ) : 
  b = a → ∃ x : ℝ, x = 0 ∧ equation a b x := by sorry

end NUMINAMATH_CALUDE_positive_root_negative_root_zero_root_l3070_307096


namespace NUMINAMATH_CALUDE_circle_radius_from_chords_l3070_307083

/-- Given a circle with two chords drawn from a single point, prove that the radius is 85/8 -/
theorem circle_radius_from_chords (chord1 chord2 midpoint_distance : ℝ) 
  (h1 : chord1 = 9)
  (h2 : chord2 = 17)
  (h3 : midpoint_distance = 5) : 
  ∃ (radius : ℝ), radius = 85 / 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_chords_l3070_307083


namespace NUMINAMATH_CALUDE_min_recolor_is_n_minus_one_l3070_307036

/-- A complete graph of order n (≥ 3) with edges colored using three colors. -/
structure ColoredCompleteGraph where
  n : ℕ
  n_ge_3 : n ≥ 3
  colors : Fin 3 → Type
  edge_coloring : Fin n → Fin n → Fin 3
  each_color_used : ∀ c : Fin 3, ∃ i j : Fin n, i ≠ j ∧ edge_coloring i j = c

/-- The minimum number of edges that need to be recolored to make the graph connected by one color. -/
def min_recolor (G : ColoredCompleteGraph) : ℕ := G.n - 1

/-- Theorem stating that the minimum number of edges to recolor is n - 1. -/
theorem min_recolor_is_n_minus_one (G : ColoredCompleteGraph) :
  min_recolor G = G.n - 1 := by sorry

end NUMINAMATH_CALUDE_min_recolor_is_n_minus_one_l3070_307036


namespace NUMINAMATH_CALUDE_dummies_leftover_l3070_307039

theorem dummies_leftover (n : ℕ) (h : n % 10 = 3) : (4 * n) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_dummies_leftover_l3070_307039


namespace NUMINAMATH_CALUDE_exists_irrational_greater_than_neg_three_l3070_307018

theorem exists_irrational_greater_than_neg_three :
  ∃ x : ℝ, Irrational x ∧ x > -3 := by sorry

end NUMINAMATH_CALUDE_exists_irrational_greater_than_neg_three_l3070_307018


namespace NUMINAMATH_CALUDE_simplify_fraction_l3070_307043

theorem simplify_fraction (m : ℝ) (hm : m ≠ 0) :
  (m - 1) / m / ((m - 1) / (m^2)) = m := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3070_307043


namespace NUMINAMATH_CALUDE_radical_equation_condition_l3070_307009

theorem radical_equation_condition (x y : ℝ) : 
  xy ≠ 0 → (Real.sqrt (4 * x^2 * y^3) = -2 * x * y * Real.sqrt y ↔ x < 0 ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_radical_equation_condition_l3070_307009


namespace NUMINAMATH_CALUDE_vector_problem_l3070_307045

/-- Given vectors in R^2 -/
def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![-1, 2]
def c : Fin 2 → ℝ := ![4, 1]

/-- Parallel vectors in R^2 -/
def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (t : ℝ), v 0 * w 1 = t * v 1 * w 0

theorem vector_problem :
  (∃ k : ℝ, parallel (fun i => a i + k * c i) (fun i => 2 * b i + c i) → k = -11/18) ∧
  (∃ m n : ℝ, (∀ i, a i = m * b i - n * c i) → m = 5/9 ∧ n = -8/9) :=
sorry

end NUMINAMATH_CALUDE_vector_problem_l3070_307045


namespace NUMINAMATH_CALUDE_positive_number_equality_l3070_307010

theorem positive_number_equality (x : ℝ) (h1 : x > 0) 
  (h2 : (2/3) * x = (64/216) * (1/x)) : x = (2/9) * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_equality_l3070_307010


namespace NUMINAMATH_CALUDE_tom_helicopter_rental_days_l3070_307000

/-- Calculates the number of days a helicopter was rented given the rental conditions and total payment -/
def helicopter_rental_days (hours_per_day : ℕ) (cost_per_hour : ℕ) (total_paid : ℕ) : ℕ :=
  total_paid / (hours_per_day * cost_per_hour)

/-- Theorem: Given Tom's helicopter rental conditions, he rented it for 3 days -/
theorem tom_helicopter_rental_days :
  helicopter_rental_days 2 75 450 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tom_helicopter_rental_days_l3070_307000


namespace NUMINAMATH_CALUDE_expansion_nonzero_terms_l3070_307053

/-- The number of nonzero terms in the expansion of (x^2+5)(3x^3+2x^2+6)-4(x^4-3x^3+8x^2+1) + 2x^3 -/
theorem expansion_nonzero_terms (x : ℝ) : 
  let expanded := (x^2 + 5) * (3*x^3 + 2*x^2 + 6) - 4*(x^4 - 3*x^3 + 8*x^2 + 1) + 2*x^3
  ∃ (a b c d e : ℝ) (n : ℕ), 
    expanded = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e ∧ 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧
    n = 5 := by
  sorry

end NUMINAMATH_CALUDE_expansion_nonzero_terms_l3070_307053


namespace NUMINAMATH_CALUDE_limit_alternating_log_infinity_l3070_307032

/-- The limit of (-1)^n * log(n) as n approaches infinity is infinity. -/
theorem limit_alternating_log_infinity :
  ∀ M : ℝ, M > 0 → ∃ N : ℕ, ∀ n : ℕ, n ≥ N → |(-1:ℝ)^n * Real.log n| > M :=
sorry

end NUMINAMATH_CALUDE_limit_alternating_log_infinity_l3070_307032


namespace NUMINAMATH_CALUDE_ellipse_properties_l3070_307056

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the semi-focal distance
def semi_focal_distance : ℝ := 1

-- Define the condition that a > b > 0
def size_condition (a b : ℝ) : Prop :=
  a > b ∧ b > 0

-- Define the condition that the circle with diameter F₁F₂ passes through upper and lower vertices
def circle_condition (a b : ℝ) : Prop :=
  2 * semi_focal_distance = a

-- Theorem statement
theorem ellipse_properties (a b : ℝ) 
  (h1 : size_condition a b) 
  (h2 : circle_condition a b) :
  (∀ x y : ℝ, ellipse a b x y ↔ x^2 / 2 + y^2 = 1) ∧
  (∃ k : ℝ, -Real.sqrt 2 / 2 < k ∧ k < 0 ∧
    ∀ x y : ℝ, y = k * (x - 2) → ellipse a b x y → y > 0 → x = 2 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3070_307056


namespace NUMINAMATH_CALUDE_checkers_rectangle_exists_l3070_307003

/-- Represents the color of a checker -/
inductive Color
| White
| Black

/-- Represents a 3x7 grid of checkers -/
def CheckerGrid := Fin 3 → Fin 7 → Color

/-- Checks if four positions form a rectangle in the grid -/
def IsRectangle (a b c d : Fin 3 × Fin 7) : Prop :=
  (a.1 = b.1 ∧ c.1 = d.1 ∧ a.2 ≠ c.2) ∨
  (a.1 = c.1 ∧ b.1 = d.1 ∧ a.2 ≠ b.2)

/-- The main theorem -/
theorem checkers_rectangle_exists (grid : CheckerGrid) :
  ∃ (color : Color) (a b c d : Fin 3 × Fin 7),
    IsRectangle a b c d ∧
    grid a.1 a.2 = color ∧
    grid b.1 b.2 = color ∧
    grid c.1 c.2 = color ∧
    grid d.1 d.2 = color :=
sorry

end NUMINAMATH_CALUDE_checkers_rectangle_exists_l3070_307003


namespace NUMINAMATH_CALUDE_salary_for_may_l3070_307058

/-- Proves that the salary for May is 3600, given the average salaries for two sets of four months and the salary for January. -/
theorem salary_for_may (jan feb mar apr may : ℝ) : 
  (jan + feb + mar + apr) / 4 = 8000 →
  (feb + mar + apr + may) / 4 = 8900 →
  jan = 2900 →
  may = 3600 := by
  sorry

end NUMINAMATH_CALUDE_salary_for_may_l3070_307058


namespace NUMINAMATH_CALUDE_tournament_matches_l3070_307019

/-- Represents the number of matches played by each student -/
structure MatchCounts where
  student1 : Nat
  student2 : Nat
  student3 : Nat
  student4 : Nat
  student5 : Nat
  student6 : Nat

/-- The total number of matches in a tournament with 6 players -/
def totalMatches : Nat := 15

theorem tournament_matches (mc : MatchCounts) : 
  mc.student1 = 5 → 
  mc.student2 = 4 → 
  mc.student3 = 3 → 
  mc.student4 = 2 → 
  mc.student5 = 1 → 
  mc.student1 + mc.student2 + mc.student3 + mc.student4 + mc.student5 + mc.student6 = 2 * totalMatches → 
  mc.student6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tournament_matches_l3070_307019


namespace NUMINAMATH_CALUDE_negative_cube_squared_l3070_307026

theorem negative_cube_squared (x : ℝ) : (-x^3)^2 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_squared_l3070_307026


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l3070_307071

def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (m - 2) * x + 2

theorem quadratic_symmetry (m : ℝ) :
  (∀ x, f m x = f m (-x)) →
  (m = 2 ∧
   (∀ x y, x < y → f m x < f m y) ∧
   (∀ x, x > 0 → f m x > f m 0) ∧
   (f m 0 = 2 ∧ ∀ x, f m x ≥ 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l3070_307071


namespace NUMINAMATH_CALUDE_same_color_config_prob_is_correct_l3070_307029

def total_candies : ℕ := 40
def red_candies : ℕ := 15
def blue_candies : ℕ := 15
def green_candies : ℕ := 10

def same_color_config_prob : ℚ :=
  let prob_both_red := (red_candies * (red_candies - 1) * (red_candies - 2) * (red_candies - 3)) / 
                       (total_candies * (total_candies - 1) * (total_candies - 2) * (total_candies - 3))
  let prob_both_blue := (blue_candies * (blue_candies - 1) * (blue_candies - 2) * (blue_candies - 3)) / 
                        (total_candies * (total_candies - 1) * (total_candies - 2) * (total_candies - 3))
  let prob_both_green := (green_candies * (green_candies - 1) * (green_candies - 2) * (green_candies - 3)) / 
                         (total_candies * (total_candies - 1) * (total_candies - 2) * (total_candies - 3))
  let prob_both_red_blue := (red_candies * blue_candies * (red_candies - 1) * (blue_candies - 1)) / 
                            (total_candies * (total_candies - 1) * (total_candies - 2) * (total_candies - 3))
  2 * prob_both_red + 2 * prob_both_blue + prob_both_green + 2 * prob_both_red_blue

theorem same_color_config_prob_is_correct : same_color_config_prob = 579 / 8686 := by
  sorry

end NUMINAMATH_CALUDE_same_color_config_prob_is_correct_l3070_307029


namespace NUMINAMATH_CALUDE_binary_multiplication_division_equality_l3070_307095

/-- Represents a binary number as a list of booleans, with the least significant bit first. -/
def BinaryNumber := List Bool

/-- Converts a natural number to its binary representation. -/
def toBinary (n : Nat) : BinaryNumber :=
  if n = 0 then [] else (n % 2 = 1) :: toBinary (n / 2)

/-- Converts a binary number to its decimal representation. -/
def toDecimal (b : BinaryNumber) : Nat :=
  b.foldl (fun acc digit => 2 * acc + if digit then 1 else 0) 0

/-- Multiplies two binary numbers. -/
def binaryMultiply (a b : BinaryNumber) : BinaryNumber :=
  toBinary (toDecimal a * toDecimal b)

/-- Divides a binary number by another binary number. -/
def binaryDivide (a b : BinaryNumber) : BinaryNumber :=
  toBinary (toDecimal a / toDecimal b)

theorem binary_multiplication_division_equality :
  let a := [false, true, false, true, true, false, true]  -- 1011010₂
  let b := [false, false, true, false, true, false, true] -- 1010100₂
  let c := [false, true, false, true]                     -- 1010₂
  binaryDivide (binaryMultiply a b) c = 
    [false, false, true, false, false, true, true, true, false, true] -- 1011100100₂
  := by sorry

end NUMINAMATH_CALUDE_binary_multiplication_division_equality_l3070_307095


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3070_307099

theorem quadratic_factorization (y : ℝ) : 16 * y^2 - 48 * y + 36 = (4 * y - 6)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3070_307099


namespace NUMINAMATH_CALUDE_sequence_sum_equals_321_64_l3070_307027

def sequence_term (n : ℕ) : ℚ := (2^n - 1) / 2^n

def sum_of_terms (n : ℕ) : ℚ := n - 1 + 1 / 2^(n+1)

theorem sequence_sum_equals_321_64 :
  ∃ n : ℕ, sum_of_terms n = 321 / 64 ∧ n = 6 := by sorry

end NUMINAMATH_CALUDE_sequence_sum_equals_321_64_l3070_307027


namespace NUMINAMATH_CALUDE_kermit_sleep_positions_l3070_307054

/-- Represents a position on the infinite square grid -/
structure Position :=
  (x : Int) (y : Int)

/-- The number of Joules Kermit starts with -/
def initial_energy : Nat := 100

/-- Calculates the number of unique positions Kermit can reach -/
def unique_positions (energy : Nat) : Nat :=
  (2 * energy + 1) * (2 * energy + 1)

/-- Theorem stating the number of unique positions Kermit can reach -/
theorem kermit_sleep_positions : 
  unique_positions initial_energy = 10201 := by
  sorry

end NUMINAMATH_CALUDE_kermit_sleep_positions_l3070_307054


namespace NUMINAMATH_CALUDE_ab_sufficient_not_necessary_for_a_plus_b_l3070_307042

theorem ab_sufficient_not_necessary_for_a_plus_b (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∀ a b, a > 0 → b > 0 → a * b > 1 → a + b > 2) ∧ 
  (∃ a b, a > 0 ∧ b > 0 ∧ a + b > 2 ∧ a * b ≤ 1) := by
sorry

end NUMINAMATH_CALUDE_ab_sufficient_not_necessary_for_a_plus_b_l3070_307042


namespace NUMINAMATH_CALUDE_problem_solution_l3070_307031

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | x > a}

theorem problem_solution :
  (A ∪ B = {x : ℝ | 1 < x ∧ x ≤ 8}) ∧
  (∀ a : ℝ, A ∩ C a = ∅ ↔ a ≥ 8) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3070_307031


namespace NUMINAMATH_CALUDE_triangle_area_l3070_307008

-- Define the lines
def line1 (x y : ℝ) : Prop := y - 2*x = 3
def line2 (x y : ℝ) : Prop := 2*y - x = 9

-- Define the triangle
def triangle := {(x, y) : ℝ × ℝ | x ≥ 0 ∧ line1 x y ∧ line2 x y}

-- State the theorem
theorem triangle_area : MeasureTheory.volume triangle = 3/4 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l3070_307008


namespace NUMINAMATH_CALUDE_collinear_sufficient_not_necessary_for_coplanar_l3070_307022

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Determines if three points are collinear -/
def collinear (p q r : Point3D) : Prop := sorry

/-- Determines if four points are coplanar -/
def coplanar (p q r s : Point3D) : Prop := sorry

/-- The main theorem stating that three collinear points out of four
    is sufficient but not necessary for four points to be coplanar -/
theorem collinear_sufficient_not_necessary_for_coplanar :
  ∃ (p q r s : Point3D),
    (collinear p q r → coplanar p q r s) ∧
    (coplanar p q r s ∧ ¬(collinear p q r ∨ collinear p q s ∨ collinear p r s ∨ collinear q r s)) := by
  sorry

end NUMINAMATH_CALUDE_collinear_sufficient_not_necessary_for_coplanar_l3070_307022


namespace NUMINAMATH_CALUDE_dividend_calculation_l3070_307052

/-- Calculates the dividend received from an investment in shares -/
theorem dividend_calculation (investment : ℝ) (face_value : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ)
  (h1 : investment = 14400)
  (h2 : face_value = 100)
  (h3 : premium_rate = 0.20)
  (h4 : dividend_rate = 0.07) :
  let price_per_share := face_value * (1 + premium_rate)
  let num_shares := investment / price_per_share
  let dividend_per_share := face_value * dividend_rate
  num_shares * dividend_per_share = 840 := by sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3070_307052


namespace NUMINAMATH_CALUDE_mitchs_weekly_earnings_l3070_307048

/-- Mitch's weekly earnings calculation --/
theorem mitchs_weekly_earnings : 
  let weekday_hours : ℕ := 5
  let weekend_hours : ℕ := 3
  let weekday_rate : ℕ := 3
  let weekend_rate : ℕ := 2 * weekday_rate
  let weekdays : ℕ := 5
  let weekend_days : ℕ := 2

  weekdays * weekday_hours * weekday_rate + 
  weekend_days * weekend_hours * weekend_rate = 111 := by
  sorry

end NUMINAMATH_CALUDE_mitchs_weekly_earnings_l3070_307048


namespace NUMINAMATH_CALUDE_extra_postage_count_l3070_307082

structure Envelope where
  length : Float
  height : Float
  thickness : Float

def requires_extra_postage (e : Envelope) : Bool :=
  let ratio := e.length / e.height
  ratio < 1.2 || ratio > 2.8 || e.thickness > 0.25

def envelopes : List Envelope := [
  { length := 7, height := 5, thickness := 0.2 },
  { length := 10, height := 2, thickness := 0.3 },
  { length := 7, height := 7, thickness := 0.1 },
  { length := 12, height := 4, thickness := 0.26 }
]

theorem extra_postage_count :
  (envelopes.filter requires_extra_postage).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_extra_postage_count_l3070_307082


namespace NUMINAMATH_CALUDE_fraction_inequality_l3070_307073

theorem fraction_inequality (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (hab : a < b) (hcd : c < d) : 
  (a + c) / (b + c) < (a + d) / (b + d) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3070_307073


namespace NUMINAMATH_CALUDE_f_one_equals_one_l3070_307066

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_one_equals_one
  (f : ℝ → ℝ)
  (h : is_odd_function (fun x ↦ f (x + 1) - 1)) :
  f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_one_equals_one_l3070_307066


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3070_307012

theorem intersection_of_sets : 
  let A : Set ℤ := {-1, 1, 2, 4}
  let B : Set ℤ := {-1, 0, 2}
  A ∩ B = {-1, 2} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3070_307012


namespace NUMINAMATH_CALUDE_pencil_sale_ratio_l3070_307094

theorem pencil_sale_ratio :
  ∀ (C S : ℚ),
  C > 0 → S > 0 →
  80 * C = 80 * S + 30 * S →
  (80 * C) / (80 * S) = 11 / 8 := by
sorry

end NUMINAMATH_CALUDE_pencil_sale_ratio_l3070_307094


namespace NUMINAMATH_CALUDE_special_line_properties_l3070_307060

/-- A line passing through (-2, 3) with x-intercept twice the y-intercept -/
def special_line (x y : ℝ) : Prop :=
  x + 2 * y - 4 = 0

theorem special_line_properties :
  (special_line (-2) 3) ∧
  (∃ a : ℝ, a ≠ 0 ∧ special_line (2 * a) 0 ∧ special_line 0 a) :=
by sorry

end NUMINAMATH_CALUDE_special_line_properties_l3070_307060


namespace NUMINAMATH_CALUDE_sqrt_square_of_negative_l3070_307025

theorem sqrt_square_of_negative : Real.sqrt ((-2023)^2) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_of_negative_l3070_307025


namespace NUMINAMATH_CALUDE_initial_average_age_proof_l3070_307075

/-- Proves that the initial average age of a group is 16 years, given the specified conditions. -/
theorem initial_average_age_proof (initial_count : ℕ) (new_count : ℕ) (new_avg_age : ℚ) (final_avg_age : ℚ) :
  initial_count = 20 →
  new_count = 20 →
  new_avg_age = 15 →
  final_avg_age = 15.5 →
  (initial_count * initial_avg_age + new_count * new_avg_age) / (initial_count + new_count) = final_avg_age →
  initial_avg_age = 16 := by
  sorry

#check initial_average_age_proof

end NUMINAMATH_CALUDE_initial_average_age_proof_l3070_307075


namespace NUMINAMATH_CALUDE_product_of_sums_equals_3280_l3070_307063

theorem product_of_sums_equals_3280 :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equals_3280_l3070_307063


namespace NUMINAMATH_CALUDE_distance_PF_is_five_l3070_307044

/-- Parabola structure with focus and directrix -/
structure Parabola :=
  (focus : ℝ × ℝ)
  (directrix : ℝ)

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) :=
  (point : ℝ × ℝ)
  (on_parabola : (point.2)^2 = 4 * point.1)

/-- Given parabola y^2 = 4x -/
def given_parabola : Parabola :=
  { focus := (1, 0),
    directrix := -1 }

/-- Point P on the parabola with x-coordinate 4 -/
def point_P : PointOnParabola given_parabola :=
  { point := (4, 4),
    on_parabola := by sorry }

/-- Theorem: The distance between P and F is 5 -/
theorem distance_PF_is_five :
  let F := given_parabola.focus
  let P := point_P.point
  Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) = 5 := by sorry

end NUMINAMATH_CALUDE_distance_PF_is_five_l3070_307044


namespace NUMINAMATH_CALUDE_root_ordering_l3070_307023

/-- Given a quadratic function f(x) = (x-m)(x-n) + 2 where m < n,
    and α, β are the roots of f(x) = 0 with α < β,
    prove that m < α < β < n -/
theorem root_ordering (m n α β : ℝ) (hm : m < n) (hα : α < β)
  (hf : ∀ x, (x - m) * (x - n) + 2 = 0 ↔ x = α ∨ x = β) :
  m < α ∧ α < β ∧ β < n :=
sorry

end NUMINAMATH_CALUDE_root_ordering_l3070_307023


namespace NUMINAMATH_CALUDE_impossibleTransformation_l3070_307046

-- Define the button colors
inductive Color
| A
| B
| C

-- Define the configuration as a list of colors
def Configuration := List Color

-- Define the card values
inductive CardValue
| One
| NegOne
| Zero

-- Function to calculate the card value between two adjacent colors
def getCardValue (c1 c2 : Color) : CardValue :=
  match c1, c2 with
  | Color.B, Color.A => CardValue.One
  | Color.A, Color.C => CardValue.One
  | Color.A, Color.B => CardValue.NegOne
  | Color.C, Color.A => CardValue.NegOne
  | _, _ => CardValue.Zero

-- Function to calculate the sum of card values for a configuration
def sumCardValues (config : Configuration) : Int :=
  let pairs := List.zip config (config.rotateLeft 1)
  let cardValues := pairs.map (fun (c1, c2) => getCardValue c1 c2)
  cardValues.foldl (fun sum cv => 
    sum + match cv with
    | CardValue.One => 1
    | CardValue.NegOne => -1
    | CardValue.Zero => 0
  ) 0

-- Define the initial and final configurations
def initialConfig : Configuration := [Color.A, Color.C, Color.B, Color.C, Color.B]
def finalConfig : Configuration := [Color.A, Color.B, Color.C, Color.B, Color.C]

-- Theorem: It's impossible to transform the initial configuration to the final configuration
theorem impossibleTransformation : 
  ∀ (swapSequence : List (Configuration → Configuration)),
  (∀ (config : Configuration), sumCardValues config = sumCardValues (swapSequence.foldl (fun c f => f c) config)) →
  swapSequence.foldl (fun c f => f c) initialConfig ≠ finalConfig :=
sorry

end NUMINAMATH_CALUDE_impossibleTransformation_l3070_307046


namespace NUMINAMATH_CALUDE_blueprint_to_actual_length_l3070_307050

/-- Represents the scale of a blueprint in feet per inch -/
def blueprint_scale : ℝ := 500

/-- Represents the length of a line segment on the blueprint in inches -/
def blueprint_length : ℝ := 6.5

/-- Represents the actual length in feet corresponding to the blueprint length -/
def actual_length : ℝ := blueprint_scale * blueprint_length

theorem blueprint_to_actual_length :
  actual_length = 3250 := by sorry

end NUMINAMATH_CALUDE_blueprint_to_actual_length_l3070_307050


namespace NUMINAMATH_CALUDE_math_competition_theorem_l3070_307011

/-- Represents the number of participants who solved both problem i and problem j -/
def p (i j : Fin 6) (n : ℕ) : ℕ := sorry

/-- Represents the number of participants who solved exactly k problems -/
def n_k (k : Fin 7) (n : ℕ) : ℕ := sorry

theorem math_competition_theorem (n : ℕ) :
  (∀ i j : Fin 6, i < j → p i j n > (2 * n) / 5) →
  (n_k 6 n = 0) →
  (n_k 5 n ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_math_competition_theorem_l3070_307011


namespace NUMINAMATH_CALUDE_minimum_guests_l3070_307035

theorem minimum_guests (total_food : ℝ) (max_per_guest : ℝ) (min_guests : ℕ) :
  total_food = 406 →
  max_per_guest = 2.5 →
  min_guests = 163 →
  (↑min_guests : ℝ) * max_per_guest ≥ total_food ∧
  ∀ n : ℕ, (↑n : ℝ) * max_per_guest ≥ total_food → n ≥ min_guests :=
by sorry

end NUMINAMATH_CALUDE_minimum_guests_l3070_307035


namespace NUMINAMATH_CALUDE_fifteen_sided_figure_area_main_theorem_l3070_307085

/-- The area of a fifteen-sided figure created by cutting off three right triangles
    from the corners of a 4 × 5 rectangle --/
theorem fifteen_sided_figure_area : ℝ → Prop :=
  λ area_result : ℝ =>
    let rectangle_width : ℝ := 4
    let rectangle_height : ℝ := 5
    let rectangle_area : ℝ := rectangle_width * rectangle_height
    let triangle_side : ℝ := 1
    let triangle_area : ℝ := (1 / 2) * triangle_side * triangle_side
    let num_triangles : ℕ := 3
    let total_removed_area : ℝ := (triangle_area : ℝ) * num_triangles
    let final_area : ℝ := rectangle_area - total_removed_area
    area_result = final_area ∧ area_result = 18.5

/-- The main theorem stating that the area of the fifteen-sided figure is 18.5 cm² --/
theorem main_theorem : fifteen_sided_figure_area 18.5 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_sided_figure_area_main_theorem_l3070_307085


namespace NUMINAMATH_CALUDE_composition_equation_solution_l3070_307079

theorem composition_equation_solution :
  let δ : ℝ → ℝ := λ x ↦ 4 * x + 9
  let φ : ℝ → ℝ := λ x ↦ 9 * x + 6
  ∃ x : ℝ, δ (φ x) = 3 ∧ x = -5/6 := by
sorry

end NUMINAMATH_CALUDE_composition_equation_solution_l3070_307079


namespace NUMINAMATH_CALUDE_monday_hours_calculation_l3070_307098

def hourly_wage : ℝ := 10
def monday_tips : ℝ := 18
def tuesday_hours : ℝ := 5
def tuesday_tips : ℝ := 12
def wednesday_hours : ℝ := 7
def wednesday_tips : ℝ := 20
def total_earnings : ℝ := 240

theorem monday_hours_calculation (monday_hours : ℝ) :
  hourly_wage * monday_hours + monday_tips +
  hourly_wage * tuesday_hours + tuesday_tips +
  hourly_wage * wednesday_hours + wednesday_tips = total_earnings →
  monday_hours = 7 := by
sorry

end NUMINAMATH_CALUDE_monday_hours_calculation_l3070_307098


namespace NUMINAMATH_CALUDE_vegetable_ghee_weight_l3070_307013

/-- The weight of one liter of brand 'b' vegetable ghee in grams -/
def weight_b : ℝ := 700

/-- The ratio of brand 'a' to brand 'b' in the mixture -/
def ratio_a : ℝ := 3
def ratio_b : ℝ := 2

/-- The total volume of the mixture in liters -/
def total_volume : ℝ := 4

/-- The total weight of the mixture in grams -/
def total_weight : ℝ := 3280

/-- The weight of one liter of brand 'a' vegetable ghee in grams -/
def weight_a : ℝ := 900

theorem vegetable_ghee_weight : 
  (weight_a * (ratio_a / (ratio_a + ratio_b)) * total_volume) + 
  (weight_b * (ratio_b / (ratio_a + ratio_b)) * total_volume) = total_weight :=
by sorry

end NUMINAMATH_CALUDE_vegetable_ghee_weight_l3070_307013


namespace NUMINAMATH_CALUDE_reservoir_fullness_after_storm_l3070_307049

theorem reservoir_fullness_after_storm 
  (original_content : ℝ) 
  (initial_percentage : ℝ) 
  (added_water : ℝ) 
  (h1 : original_content = 220)
  (h2 : initial_percentage = 55.00000000000001)
  (h3 : added_water = 120) : 
  (original_content + added_water) / (original_content / (initial_percentage / 100)) * 100 = 85 := by
sorry

end NUMINAMATH_CALUDE_reservoir_fullness_after_storm_l3070_307049


namespace NUMINAMATH_CALUDE_andrew_payment_l3070_307017

/-- The total amount Andrew paid for grapes and mangoes -/
def total_paid (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Andrew paid 1055 for his purchase -/
theorem andrew_payment : total_paid 8 70 9 55 = 1055 := by
  sorry

end NUMINAMATH_CALUDE_andrew_payment_l3070_307017


namespace NUMINAMATH_CALUDE_least_positive_integer_satisfying_inequality_l3070_307074

theorem least_positive_integer_satisfying_inequality : 
  ∀ n : ℕ+, (1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 15 ↔ n ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_satisfying_inequality_l3070_307074


namespace NUMINAMATH_CALUDE_area_of_region_is_10_625_l3070_307033

/-- The lower boundary function of the region -/
def lower_boundary (x : ℝ) : ℝ := |x - 4|

/-- The upper boundary function of the region -/
def upper_boundary (x : ℝ) : ℝ := 5 - |x - 2|

/-- The region in the xy-plane -/
def region : Set (ℝ × ℝ) :=
  {p | lower_boundary p.1 ≤ p.2 ∧ p.2 ≤ upper_boundary p.1}

/-- The area of the region -/
noncomputable def area_of_region : ℝ := sorry

theorem area_of_region_is_10_625 : area_of_region = 10.625 := by sorry

end NUMINAMATH_CALUDE_area_of_region_is_10_625_l3070_307033


namespace NUMINAMATH_CALUDE_vector_parallel_implies_k_equals_one_l3070_307021

-- Define the vectors
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, 3)
def c (k : ℝ) : ℝ × ℝ := (k, 7)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v.1 * w.2 = t * v.2 * w.1

-- Theorem statement
theorem vector_parallel_implies_k_equals_one (k : ℝ) :
  parallel (a.1 + 2 * (c k).1, a.2 + 2 * (c k).2) b → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_implies_k_equals_one_l3070_307021


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l3070_307057

theorem right_triangle_shorter_leg (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  a ≤ b →            -- a is the shorter leg
  a = 25 :=          -- The shorter leg is 25 units
by sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l3070_307057


namespace NUMINAMATH_CALUDE_negation_of_forall_proposition_l3070_307067

open Set

theorem negation_of_forall_proposition :
  (¬ ∀ x ∈ (Set.Ioo 0 1), x^2 - x < 0) ↔ (∃ x ∈ (Set.Ioo 0 1), x^2 - x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_proposition_l3070_307067


namespace NUMINAMATH_CALUDE_fifth_hexagon_dots_l3070_307065

/-- The number of dots on each side of a hexagon layer -/
def dots_per_side (n : ℕ) : ℕ := n + 2

/-- The total number of dots in a single layer of a hexagon -/
def dots_in_layer (n : ℕ) : ℕ := 6 * (dots_per_side n)

/-- The total number of dots in a hexagon with n layers -/
def total_dots (n : ℕ) : ℕ := 
  if n = 0 then 0
  else total_dots (n - 1) + dots_in_layer n

/-- The fifth hexagon has 150 dots -/
theorem fifth_hexagon_dots : total_dots 5 = 150 := by
  sorry


end NUMINAMATH_CALUDE_fifth_hexagon_dots_l3070_307065


namespace NUMINAMATH_CALUDE_roots_satisfy_conditions_l3070_307015

theorem roots_satisfy_conditions : ∃ (x y : ℝ),
  x + y = 10 ∧
  |x - y| = 12 ∧
  x^2 - 10*x - 22 = 0 ∧
  y^2 - 10*y - 22 = 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_satisfy_conditions_l3070_307015


namespace NUMINAMATH_CALUDE_monotonic_cubic_range_l3070_307081

/-- A cubic function parameterized by b -/
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - b*x^2 + 3*x - 5

/-- The derivative of f with respect to x -/
def f_deriv (b : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*b*x + 3

theorem monotonic_cubic_range (b : ℝ) :
  (∀ x : ℝ, Monotone (f b)) ↔ b ∈ Set.Icc (-3) 3 :=
sorry

end NUMINAMATH_CALUDE_monotonic_cubic_range_l3070_307081


namespace NUMINAMATH_CALUDE_existence_of_twin_primes_l3070_307030

theorem existence_of_twin_primes : ∃ n : ℕ, Prime n ∧ Prime (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_twin_primes_l3070_307030


namespace NUMINAMATH_CALUDE_triangle_inequality_l3070_307068

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the theorem
theorem triangle_inequality (t : Triangle) 
  (h1 : t.a^2 = t.b * (t.b + t.c))  -- Given condition
  (h2 : t.C > Real.pi / 2)          -- Angle C is obtuse
  : t.a < 2 * t.b ∧ 2 * t.b < t.c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3070_307068


namespace NUMINAMATH_CALUDE_connection_duration_l3070_307092

/-- Calculates the number of days a client can be connected to the internet given the specified parameters. -/
def days_connected (initial_balance : ℚ) (payment : ℚ) (daily_cost : ℚ) (discontinuation_threshold : ℚ) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions, the client will be connected for 14 days. -/
theorem connection_duration :
  days_connected 0 7 (1/2) 5 = 14 :=
by sorry

end NUMINAMATH_CALUDE_connection_duration_l3070_307092


namespace NUMINAMATH_CALUDE_rocketry_club_theorem_l3070_307072

theorem rocketry_club_theorem (total_students : ℕ) 
  (nails_neq_bolts : ℕ) (screws_eq_nails : ℕ) :
  total_students = 40 →
  nails_neq_bolts = 15 →
  screws_eq_nails = 10 →
  ∃ (screws_neq_bolts : ℕ), screws_neq_bolts ≥ 15 ∧
    screws_neq_bolts ≤ total_students - screws_eq_nails :=
by sorry

end NUMINAMATH_CALUDE_rocketry_club_theorem_l3070_307072


namespace NUMINAMATH_CALUDE_compound_composition_l3070_307020

/-- Represents the number of atoms of each element in the compound -/
structure CompoundComposition where
  hydrogen : ℕ
  chromium : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given its composition and atomic weights -/
def molecularWeight (comp : CompoundComposition) (h_weight o_weight cr_weight : ℚ) : ℚ :=
  comp.hydrogen * h_weight + comp.chromium * cr_weight + comp.oxygen * o_weight

/-- States that the compound has the given composition and molecular weight -/
theorem compound_composition (h_weight o_weight cr_weight : ℚ) :
  ∃ (comp : CompoundComposition),
    comp.chromium = 1 ∧
    comp.oxygen = 4 ∧
    molecularWeight comp h_weight o_weight cr_weight = 118 ∧
    h_weight = 1 ∧
    o_weight = 16 ∧
    cr_weight = 52 ∧
    comp.hydrogen = 2 := by
  sorry

end NUMINAMATH_CALUDE_compound_composition_l3070_307020


namespace NUMINAMATH_CALUDE_magician_trick_l3070_307070

def is_valid_selection (a d : ℕ) : Prop :=
  2 ≤ a ∧ a ≤ 16 ∧
  2 ≤ d ∧ d ≤ 16 ∧
  a % 2 = 0 ∧ d % 2 = 0 ∧
  a ≠ d ∧
  (d - a) % 16 = 3 ∨ (a - d) % 16 = 3

theorem magician_trick :
  ∃ (a d : ℕ), is_valid_selection a d ∧ a * d = 120 :=
sorry

end NUMINAMATH_CALUDE_magician_trick_l3070_307070


namespace NUMINAMATH_CALUDE_tori_trash_count_l3070_307088

/-- The number of pieces of trash Tori picked up in the classrooms -/
def classroom_trash : ℕ := 344

/-- The number of pieces of trash Tori picked up outside the classrooms -/
def outside_trash : ℕ := 1232

/-- The total number of pieces of trash Tori picked up -/
def total_trash : ℕ := classroom_trash + outside_trash

theorem tori_trash_count : total_trash = 1576 := by
  sorry

end NUMINAMATH_CALUDE_tori_trash_count_l3070_307088


namespace NUMINAMATH_CALUDE_torn_sheets_count_l3070_307040

/-- Represents a book with consecutively numbered pages, two per sheet. -/
structure Book where
  /-- The last page number in the book -/
  last_page : ℕ

/-- Represents a set of consecutively torn-out sheets from a book -/
structure TornSheets where
  /-- The first torn-out page number -/
  first_page : ℕ
  /-- The last torn-out page number -/
  last_page : ℕ

/-- Check if two numbers have the same digits -/
def same_digits (a b : ℕ) : Prop := sorry

/-- Calculate the number of sheets torn out -/
def sheets_torn_out (ts : TornSheets) : ℕ :=
  (ts.last_page - ts.first_page + 1) / 2

/-- Main theorem -/
theorem torn_sheets_count (b : Book) (ts : TornSheets) :
  ts.first_page = 185 →
  same_digits ts.first_page ts.last_page →
  Even ts.last_page →
  ts.last_page > ts.first_page →
  ts.last_page ≤ b.last_page →
  sheets_torn_out ts = 167 := by sorry

end NUMINAMATH_CALUDE_torn_sheets_count_l3070_307040


namespace NUMINAMATH_CALUDE_integral_reciprocal_x_from_one_over_e_to_e_l3070_307002

open Real MeasureTheory

theorem integral_reciprocal_x_from_one_over_e_to_e :
  ∫ x in (1 / Real.exp 1)..(Real.exp 1), (1 / x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_reciprocal_x_from_one_over_e_to_e_l3070_307002


namespace NUMINAMATH_CALUDE_at_most_one_right_or_obtuse_angle_l3070_307016

-- Define a triangle as a structure with three angles
structure Triangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real
  -- Sum of angles in a triangle is 180 degrees
  sum_180 : angle1 + angle2 + angle3 = 180

-- Theorem: At most one angle in a triangle is greater than or equal to 90 degrees
theorem at_most_one_right_or_obtuse_angle (t : Triangle) :
  (t.angle1 ≥ 90 ∧ t.angle2 < 90 ∧ t.angle3 < 90) ∨
  (t.angle1 < 90 ∧ t.angle2 ≥ 90 ∧ t.angle3 < 90) ∨
  (t.angle1 < 90 ∧ t.angle2 < 90 ∧ t.angle3 ≥ 90) :=
by
  sorry


end NUMINAMATH_CALUDE_at_most_one_right_or_obtuse_angle_l3070_307016


namespace NUMINAMATH_CALUDE_min_balls_for_single_color_l3070_307087

theorem min_balls_for_single_color (red green yellow blue white black : ℕ) 
  (h_red : red = 35)
  (h_green : green = 22)
  (h_yellow : yellow = 18)
  (h_blue : blue = 15)
  (h_white : white = 12)
  (h_black : black = 8) :
  let total := red + green + yellow + blue + white + black
  ∀ n : ℕ, n ≥ 87 → 
    ∃ color : ℕ, color ≥ 18 ∧ 
      (color ≤ red ∨ color ≤ green ∨ color ≤ yellow ∨ 
       color ≤ blue ∨ color ≤ white ∨ color ≤ black) ∧
    ∀ m : ℕ, m < 87 → 
      ¬(∃ color : ℕ, color ≥ 18 ∧ 
        (color ≤ red ∨ color ≤ green ∨ color ≤ yellow ∨ 
         color ≤ blue ∨ color ≤ white ∨ color ≤ black)) :=
by sorry

end NUMINAMATH_CALUDE_min_balls_for_single_color_l3070_307087


namespace NUMINAMATH_CALUDE_investment_interest_l3070_307037

/-- Calculates the interest earned on an investment with compound interest -/
def interestEarned (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

/-- Proves that the interest earned on a $5000 investment at 3% annual interest
    compounded annually for 10 years is $1720 (rounded to the nearest dollar) -/
theorem investment_interest : 
  Int.floor (interestEarned 5000 0.03 10) = 1720 := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_l3070_307037


namespace NUMINAMATH_CALUDE_fraction_simplification_l3070_307038

theorem fraction_simplification : (2 : ℚ) / 462 + 29 / 42 = 107 / 154 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3070_307038


namespace NUMINAMATH_CALUDE_prob_at_most_two_cars_is_one_sixth_l3070_307077

/-- The number of cars in the metro train -/
def num_cars : ℕ := 6

/-- The number of deceased passengers -/
def num_deceased : ℕ := 4

/-- The probability that at most two cars have deceased passengers -/
def prob_at_most_two_cars : ℚ := 1 / 6

/-- Theorem stating that the probability of at most two cars having deceased passengers is 1/6 -/
theorem prob_at_most_two_cars_is_one_sixth :
  prob_at_most_two_cars = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_prob_at_most_two_cars_is_one_sixth_l3070_307077


namespace NUMINAMATH_CALUDE_vote_ratio_l3070_307041

/-- Given a total of 60 votes and Ben receiving 24 votes, 
    prove that the ratio of votes received by Ben to votes received by Matt is 2:3 -/
theorem vote_ratio (total_votes : Nat) (ben_votes : Nat) 
    (h1 : total_votes = 60) 
    (h2 : ben_votes = 24) : 
  ∃ (matt_votes : Nat), 
    matt_votes = total_votes - ben_votes ∧ 
    (ben_votes : ℚ) / (matt_votes : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_vote_ratio_l3070_307041


namespace NUMINAMATH_CALUDE_third_term_zero_l3070_307061

/-- 
Given two geometric progressions with first terms u₁ and v₁ and common ratios q and p respectively,
if the sum of their first terms is 0 and the sum of their second terms is 0,
then the sum of their third terms is also 0.
-/
theorem third_term_zero (u₁ v₁ q p : ℝ) 
  (h1 : u₁ + v₁ = 0) 
  (h2 : u₁ * q + v₁ * p = 0) : 
  u₁ * q^2 + v₁ * p^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_third_term_zero_l3070_307061


namespace NUMINAMATH_CALUDE_custom_mult_five_four_l3070_307076

-- Define the custom multiplication operation
def custom_mult (a b : ℤ) : ℤ := a^2 + a*b - b^2

-- Theorem statement
theorem custom_mult_five_four :
  custom_mult 5 4 = 29 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_five_four_l3070_307076


namespace NUMINAMATH_CALUDE_trapezium_height_l3070_307064

theorem trapezium_height (a b h : ℝ) (area : ℝ) : 
  a = 20 → b = 18 → area = 209 → (1/2) * (a + b) * h = area → h = 11 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_height_l3070_307064


namespace NUMINAMATH_CALUDE_remainder_theorem_l3070_307034

theorem remainder_theorem (s : ℤ) : 
  (s^15 - 2) % (s - 3) = 14348905 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3070_307034


namespace NUMINAMATH_CALUDE_no_integer_divisible_by_289_l3070_307055

theorem no_integer_divisible_by_289 :
  ∀ a : ℤ, ¬(289 ∣ (a^2 - 3*a - 19)) := by
sorry

end NUMINAMATH_CALUDE_no_integer_divisible_by_289_l3070_307055


namespace NUMINAMATH_CALUDE_hundred_brick_tower_heights_l3070_307028

/-- Represents the dimensions of a brick in inches -/
structure BrickDimensions where
  length : Nat
  width : Nat
  height : Nat

/-- Calculates the number of different tower heights achievable -/
def towerHeights (brickCount : Nat) (dimensions : BrickDimensions) : Nat :=
  sorry

/-- The main theorem stating the number of different tower heights -/
theorem hundred_brick_tower_heights :
  let brickDims : BrickDimensions := { length := 3, width := 11, height := 18 }
  towerHeights 100 brickDims = 1404 := by
  sorry

end NUMINAMATH_CALUDE_hundred_brick_tower_heights_l3070_307028


namespace NUMINAMATH_CALUDE_division_multiplication_equality_l3070_307004

theorem division_multiplication_equality : (0.24 / 0.006) * 2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_equality_l3070_307004


namespace NUMINAMATH_CALUDE_alcohol_mixture_percentage_l3070_307062

theorem alcohol_mixture_percentage (original_volume : ℝ) (water_added : ℝ) (final_percentage : ℝ) :
  original_volume = 11 →
  water_added = 3 →
  final_percentage = 33 →
  (final_percentage / 100) * (original_volume + water_added) = 
    (42 / 100) * original_volume :=
by sorry

end NUMINAMATH_CALUDE_alcohol_mixture_percentage_l3070_307062


namespace NUMINAMATH_CALUDE_years_since_same_average_l3070_307069

/-- Represents a club with members and their ages -/
structure Club where
  members : Nat
  avgAge : ℝ

/-- Represents the replacement of a member in the club -/
structure Replacement where
  oldMemberAge : ℝ
  newMemberAge : ℝ

/-- Theorem: The number of years since the average age was the same
    is equal to the age difference between the replaced and new member -/
theorem years_since_same_average (c : Club) (r : Replacement) :
  c.members = 5 →
  r.oldMemberAge - r.newMemberAge = 15 →
  c.avgAge * c.members = (c.avgAge * c.members - r.oldMemberAge + r.newMemberAge) →
  (r.oldMemberAge - r.newMemberAge : ℝ) = (c.avgAge * c.members - (c.avgAge * c.members - r.oldMemberAge + r.newMemberAge)) / c.members :=
by
  sorry


end NUMINAMATH_CALUDE_years_since_same_average_l3070_307069


namespace NUMINAMATH_CALUDE_domino_double_cover_l3070_307090

/-- Represents a domino tile placement on a 2×2 square -/
inductive DominoPlacement
  | Horizontal
  | Vertical

/-- Represents a tiling of a 2n × 2m rectangle using 1 × 2 domino tiles -/
def Tiling (n m : ℕ) := Fin n → Fin m → DominoPlacement

/-- Checks if two tilings are complementary (non-overlapping) -/
def complementary (t1 t2 : Tiling n m) : Prop :=
  ∀ i j, t1 i j ≠ t2 i j

theorem domino_double_cover (n m : ℕ) :
  ∃ (t1 t2 : Tiling n m), complementary t1 t2 := by sorry

end NUMINAMATH_CALUDE_domino_double_cover_l3070_307090


namespace NUMINAMATH_CALUDE_max_points_top_four_teams_l3070_307091

/-- Represents a tournament with the given conditions -/
structure Tournament :=
  (num_teams : ℕ)
  (points_for_win : ℕ)
  (points_for_draw : ℕ)
  (points_for_loss : ℕ)

/-- Calculates the total number of games in the tournament -/
def total_games (t : Tournament) : ℕ :=
  t.num_teams * (t.num_teams - 1) / 2

/-- Represents the maximum possible points for top teams -/
def max_points_for_top_teams (t : Tournament) (num_top_teams : ℕ) : ℕ :=
  sorry

/-- Theorem stating the maximum possible points for each of the top four teams -/
theorem max_points_top_four_teams (t : Tournament) :
  t.num_teams = 7 →
  t.points_for_win = 3 →
  t.points_for_draw = 1 →
  t.points_for_loss = 0 →
  max_points_for_top_teams t 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_max_points_top_four_teams_l3070_307091


namespace NUMINAMATH_CALUDE_min_box_height_l3070_307047

/-- The minimum height of a box with a square base, where the height is 5 units more
    than the side length of the base, and the surface area is at least 120 square units. -/
theorem min_box_height (x : ℝ) (h1 : x > 0) : 
  let height := x + 5
  let surface_area := 2 * x^2 + 4 * x * height
  surface_area ≥ 120 → height ≥ 25/3 := by
  sorry

end NUMINAMATH_CALUDE_min_box_height_l3070_307047


namespace NUMINAMATH_CALUDE_min_value_sum_fractions_equality_condition_l3070_307014

theorem min_value_sum_fractions (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y) / z + (x + z) / y + (y + z) / x + 3 ≥ 9 :=
by sorry

theorem equality_condition (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y) / z + (x + z) / y + (y + z) / x + 3 = 9 ↔ x = y ∧ y = z :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_fractions_equality_condition_l3070_307014


namespace NUMINAMATH_CALUDE_range_of_c_l3070_307006

def p (c : ℝ) : Prop := c^2 < c

def q (c : ℝ) : Prop := ∀ x : ℝ, x^2 + 4*c*x + 1 < 0

theorem range_of_c (c : ℝ) (h1 : p c ∨ q c) (h2 : ¬(p c ∧ q c)) :
  c ∈ Set.Icc (1/2) 1 ∪ Set.Ioc (-1/2) 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_c_l3070_307006


namespace NUMINAMATH_CALUDE_sum_of_modified_integers_l3070_307078

theorem sum_of_modified_integers (P : ℤ) (x y : ℤ) (h : x + y = P) :
  3 * (x + 5) + 3 * (y + 5) = 3 * P + 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_modified_integers_l3070_307078


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_seventeen_thirds_l3070_307086

theorem greatest_integer_less_than_negative_seventeen_thirds :
  ⌊-17/3⌋ = -6 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_seventeen_thirds_l3070_307086


namespace NUMINAMATH_CALUDE_sheridan_fish_problem_l3070_307097

/-- The problem of calculating Mrs. Sheridan's initial number of fish -/
theorem sheridan_fish_problem (fish_from_sister fish_total : ℕ) 
  (h1 : fish_from_sister = 47)
  (h2 : fish_total = 69)
  (h3 : fish_total = fish_from_sister + initial_fish) :
  initial_fish = 22 :=
by
  sorry

#check sheridan_fish_problem

end NUMINAMATH_CALUDE_sheridan_fish_problem_l3070_307097


namespace NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l3070_307007

theorem tan_sum_pi_twelfths : Real.tan (π / 12) + Real.tan (5 * π / 12) = 8 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l3070_307007


namespace NUMINAMATH_CALUDE_alexander_shopping_cost_l3070_307001

/-- Calculates the total cost of Alexander's shopping trip -/
def shopping_cost (apple_count : ℕ) (apple_price : ℕ) (orange_count : ℕ) (orange_price : ℕ) : ℕ :=
  apple_count * apple_price + orange_count * orange_price

/-- Theorem: Alexander spends $9 on his shopping trip -/
theorem alexander_shopping_cost :
  shopping_cost 5 1 2 2 = 9 := by
  sorry


end NUMINAMATH_CALUDE_alexander_shopping_cost_l3070_307001


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3070_307080

/-- The range of m for which the quadratic inequality (m-3)x^2 - 2mx - 8 > 0
    has a solution set that is an open interval with length between 1 and 2 -/
theorem quadratic_inequality_range (m : ℝ) : 
  (∃ a b : ℝ, 
    (∀ x : ℝ, (m - 3) * x^2 - 2 * m * x - 8 > 0 ↔ a < x ∧ x < b) ∧ 
    1 ≤ b - a ∧ b - a ≤ 2) ↔ 
  m ≤ -15 ∨ (7/3 ≤ m ∧ m ≤ 33/14) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3070_307080


namespace NUMINAMATH_CALUDE_number_of_pencils_l3070_307024

theorem number_of_pencils (pens pencils : ℕ) : 
  (pens : ℚ) / (pencils : ℚ) = 5 / 6 →
  pencils = pens + 6 →
  pencils = 36 := by
sorry

end NUMINAMATH_CALUDE_number_of_pencils_l3070_307024


namespace NUMINAMATH_CALUDE_geometric_sequence_and_max_function_l3070_307051

/-- Given that real numbers a, b, c, and d form a geometric sequence, 
    and the function y = ln(x + 2) - x attains its maximum value of c when x = b, 
    prove that ad = -1 -/
theorem geometric_sequence_and_max_function (a b c d : ℝ) :
  (∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r) →
  (∀ x : ℝ, Real.log (x + 2) - x ≤ c) →
  (Real.log (b + 2) - b = c) →
  a * d = -1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_and_max_function_l3070_307051


namespace NUMINAMATH_CALUDE_plaster_cost_per_sq_meter_l3070_307089

/-- Represents the dimensions of a rectangular tank -/
structure TankDimensions where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Calculates the total surface area of a rectangular tank that needs to be plastered -/
def totalPlasterArea (d : TankDimensions) : ℝ :=
  2 * (d.length * d.depth + d.width * d.depth) + d.length * d.width

/-- Theorem: Given a rectangular tank with dimensions 25m x 12m x 6m and a total plastering cost of 223.2 paise, 
    the cost per square meter of plastering is 0.3 paise -/
theorem plaster_cost_per_sq_meter (tank : TankDimensions) 
  (h1 : tank.length = 25)
  (h2 : tank.width = 12)
  (h3 : tank.depth = 6)
  (total_cost : ℝ)
  (h4 : total_cost = 223.2) : 
  total_cost / totalPlasterArea tank = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_plaster_cost_per_sq_meter_l3070_307089
