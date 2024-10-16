import Mathlib

namespace NUMINAMATH_CALUDE_common_factor_of_polynomial_l3085_308528

/-- The common factor of each term in the polynomial 2m^3 - 8m is 2m -/
theorem common_factor_of_polynomial (m : ℤ) : ∃ (k₁ k₂ : ℤ), 2 * m^3 - 8 * m = 2 * m * (k₁ * m^2 + k₂) :=
by sorry

end NUMINAMATH_CALUDE_common_factor_of_polynomial_l3085_308528


namespace NUMINAMATH_CALUDE_complex_number_properties_l3085_308550

theorem complex_number_properties (z : ℂ) (h : (z - 2*I) / z = 2 + I) : 
  z.im = -1 ∧ z^6 = -8*I := by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l3085_308550


namespace NUMINAMATH_CALUDE_direct_proportion_function_m_l3085_308581

theorem direct_proportion_function_m (m : ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, (m - 3) * x^(m^2 - 8) = k * x) ↔ m = -3 :=
by sorry

end NUMINAMATH_CALUDE_direct_proportion_function_m_l3085_308581


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3085_308591

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (3*x - 1)^7 = a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₇ + a₆ + a₅ + a₄ + a₃ + a₂ + a₁ + a₀ = 128 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3085_308591


namespace NUMINAMATH_CALUDE_race_result_l3085_308563

-- Define the set of runners
inductive Runner : Type
| P : Runner
| Q : Runner
| R : Runner
| S : Runner
| T : Runner

-- Define the relation "beats" between runners
def beats : Runner → Runner → Prop := sorry

-- Define the relation "finishes_before" between runners
def finishes_before : Runner → Runner → Prop := sorry

-- Define what it means for a runner to finish third
def finishes_third : Runner → Prop := sorry

-- State the theorem
theorem race_result : 
  (beats Runner.P Runner.Q) →
  (beats Runner.P Runner.R) →
  (beats Runner.Q Runner.S) →
  (finishes_before Runner.P Runner.T) →
  (finishes_before Runner.T Runner.Q) →
  (¬ finishes_third Runner.P ∧ ¬ finishes_third Runner.S) ∧
  (∃ (x : Runner), x ≠ Runner.P ∧ x ≠ Runner.S ∧ finishes_third x) :=
by sorry

end NUMINAMATH_CALUDE_race_result_l3085_308563


namespace NUMINAMATH_CALUDE_towels_used_is_285_towels_used_le_total_towels_l3085_308574

/-- Calculates the total number of towels used in a gym over 4 hours -/
def totalTowelsUsed (firstHourGuests : ℕ) : ℕ :=
  let secondHourGuests := firstHourGuests + (firstHourGuests * 20 / 100)
  let thirdHourGuests := secondHourGuests + (secondHourGuests * 25 / 100)
  let fourthHourGuests := thirdHourGuests + (thirdHourGuests * 1 / 3)
  firstHourGuests + secondHourGuests + thirdHourGuests + fourthHourGuests

/-- Theorem stating that the total number of towels used is 285 -/
theorem towels_used_is_285 :
  totalTowelsUsed 50 = 285 := by
  sorry

/-- The number of towels laid out daily -/
def totalTowels : ℕ := 300

/-- Theorem stating that the number of towels used is less than or equal to the total towels -/
theorem towels_used_le_total_towels :
  totalTowelsUsed 50 ≤ totalTowels := by
  sorry

end NUMINAMATH_CALUDE_towels_used_is_285_towels_used_le_total_towels_l3085_308574


namespace NUMINAMATH_CALUDE_wrong_mark_calculation_l3085_308535

theorem wrong_mark_calculation (n : ℕ) (initial_avg correct_avg correct_mark : ℝ) : 
  n = 10 ∧ 
  initial_avg = 100 ∧ 
  correct_avg = 96 ∧ 
  correct_mark = 10 → 
  ∃ wrong_mark : ℝ, 
    wrong_mark = 50 ∧ 
    n * initial_avg = (n - 1) * correct_avg + wrong_mark ∧
    n * correct_avg = (n - 1) * correct_avg + correct_mark :=
by sorry

end NUMINAMATH_CALUDE_wrong_mark_calculation_l3085_308535


namespace NUMINAMATH_CALUDE_tetrahedron_volume_is_sqrt3_over_3_l3085_308514

-- Define the square ABCD
def square_side_length : ℝ := 2

-- Define point E as the midpoint of AB
def E_is_midpoint (A B E : ℝ × ℝ) : Prop :=
  E = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the folding along EC and ED
def folded_square (A B C D E : ℝ × ℝ) : Prop :=
  E_is_midpoint A B E ∧
  (A.1 - E.1)^2 + (A.2 - E.2)^2 = (B.1 - E.1)^2 + (B.2 - E.2)^2

-- Define the tetrahedron CDEA
structure Tetrahedron :=
  (C D E A : ℝ × ℝ)

-- Define the volume of a tetrahedron
def tetrahedron_volume (t : Tetrahedron) : ℝ := sorry

-- Theorem statement
theorem tetrahedron_volume_is_sqrt3_over_3 
  (A B C D E : ℝ × ℝ) 
  (h1 : (C.1 - A.1)^2 + (C.2 - A.2)^2 = square_side_length^2)
  (h2 : (D.1 - B.1)^2 + (D.2 - B.2)^2 = square_side_length^2)
  (h3 : folded_square A B C D E) :
  tetrahedron_volume {C := C, D := D, E := E, A := A} = Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_is_sqrt3_over_3_l3085_308514


namespace NUMINAMATH_CALUDE_fibonacci_arithmetic_sequence_l3085_308572

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Main theorem -/
theorem fibonacci_arithmetic_sequence (a b d : ℕ) : 
  (∀ n ≥ 3, fib n = fib (n - 1) + fib (n - 2)) →  -- Fibonacci recurrence relation
  (fib a < fib b ∧ fib b < fib d) →  -- Increasing sequence
  (fib d - fib b = fib b - fib a) →  -- Arithmetic sequence
  d = b + 2 →  -- Given condition
  a + b + d = 1000 →  -- Given condition
  a = 332 := by
sorry

end NUMINAMATH_CALUDE_fibonacci_arithmetic_sequence_l3085_308572


namespace NUMINAMATH_CALUDE_discount_calculation_l3085_308547

theorem discount_calculation (cost_price : ℝ) (profit_with_discount : ℝ) (profit_without_discount : ℝ) :
  cost_price = 100 ∧ profit_with_discount = 20 ∧ profit_without_discount = 25 →
  (cost_price + cost_price * profit_without_discount / 100) - (cost_price + cost_price * profit_with_discount / 100) = 5 := by
sorry

end NUMINAMATH_CALUDE_discount_calculation_l3085_308547


namespace NUMINAMATH_CALUDE_parallelogram_area_l3085_308571

theorem parallelogram_area (side1 side2 : ℝ) (angle : ℝ) :
  side1 = 7 →
  side2 = 12 →
  angle = Real.pi / 3 →
  side2 * side1 * Real.sin angle = 12 * 7 * Real.sin (Real.pi / 3) :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3085_308571


namespace NUMINAMATH_CALUDE_exists_x_fx_equals_four_l3085_308597

open Real

theorem exists_x_fx_equals_four :
  ∃ x₀ ∈ Set.Ioo 0 (3 * π), 3 + cos (2 * x₀) = 4 := by
  sorry

end NUMINAMATH_CALUDE_exists_x_fx_equals_four_l3085_308597


namespace NUMINAMATH_CALUDE_ordering_abc_l3085_308558

theorem ordering_abc : 
  let a : ℝ := 0.1 * Real.exp 0.1
  let b : ℝ := 1 / 9
  let c : ℝ := -Real.log 0.9
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_ordering_abc_l3085_308558


namespace NUMINAMATH_CALUDE_rearranged_number_bounds_l3085_308580

/-- Given a natural number B, returns the number A obtained by moving the last digit of B to the first position --/
def rearrange_digits (B : ℕ) : ℕ :=
  let b := B % 10
  10^8 * b + (B - b) / 10

/-- Checks if two natural numbers are coprime --/
def are_coprime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

/-- Theorem stating the largest and smallest possible values of A given the conditions on B --/
theorem rearranged_number_bounds :
  ∀ B : ℕ,
  B > 222222222 →
  are_coprime B 18 →
  ∃ A : ℕ,
  A = rearrange_digits B ∧
  A ≤ 999999998 ∧
  A ≥ 122222224 ∧
  (∀ A' : ℕ, A' = rearrange_digits B → A' ≤ 999999998 ∧ A' ≥ 122222224) :=
sorry

end NUMINAMATH_CALUDE_rearranged_number_bounds_l3085_308580


namespace NUMINAMATH_CALUDE_intersection_with_complement_l3085_308536

open Set

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x < 1}

-- State the theorem
theorem intersection_with_complement : 
  A ∩ (𝒰 \ B) = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l3085_308536


namespace NUMINAMATH_CALUDE_exists_polygon_with_n_axes_of_symmetry_l3085_308502

/-- A convex polygon. -/
structure ConvexPolygon where
  -- Add necessary fields here
  -- This is a placeholder definition

/-- The number of axes of symmetry of a convex polygon. -/
def axesOfSymmetry (p : ConvexPolygon) : ℕ :=
  sorry -- Placeholder definition

/-- For any natural number n, there exists a convex polygon with exactly n axes of symmetry. -/
theorem exists_polygon_with_n_axes_of_symmetry :
  ∀ n : ℕ, ∃ p : ConvexPolygon, axesOfSymmetry p = n :=
sorry

end NUMINAMATH_CALUDE_exists_polygon_with_n_axes_of_symmetry_l3085_308502


namespace NUMINAMATH_CALUDE_b_fourth_zero_implies_b_squared_zero_l3085_308534

theorem b_fourth_zero_implies_b_squared_zero 
  (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B ^ 4 = 0) : 
  B ^ 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_b_fourth_zero_implies_b_squared_zero_l3085_308534


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l3085_308512

theorem quadratic_is_square_of_binomial (a : ℚ) : 
  (∃ b : ℚ, ∀ x : ℚ, 4*x^2 + 14*x + a = (2*x + b)^2) → a = 49/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l3085_308512


namespace NUMINAMATH_CALUDE_calculation_proof_l3085_308564

theorem calculation_proof :
  (1 : ℝ) = (1/3)^0 ∧
  3 = Real.sqrt 27 ∧
  3 = |-3| ∧
  1 = Real.tan (π/4) →
  (1/3)^0 + Real.sqrt 27 - |-3| + Real.tan (π/4) = 1 + 3 * Real.sqrt 3 - 2 ∧
  ∀ x : ℝ, (x + 2)^2 - 2*(x - 1) = x^2 + 2*x + 6 :=
by sorry

end NUMINAMATH_CALUDE_calculation_proof_l3085_308564


namespace NUMINAMATH_CALUDE_fifth_term_is_32_l3085_308557

/-- A sequence where the difference between each term and its predecessor increases by 3 each time -/
def special_sequence : ℕ → ℕ
| 0 => 2
| 1 => 5
| n + 2 => special_sequence (n + 1) + 3 * (n + 1)

theorem fifth_term_is_32 : special_sequence 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_32_l3085_308557


namespace NUMINAMATH_CALUDE_graph_below_line_l3085_308508

noncomputable def f (x : ℝ) := x * Real.log x - x^2 - 1

theorem graph_below_line (x : ℝ) (h : x > 0) : Real.log x - Real.exp x + 1 < 0 := by
  sorry

end NUMINAMATH_CALUDE_graph_below_line_l3085_308508


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3085_308559

theorem polynomial_simplification (y : ℝ) :
  (3 * y - 2) * (5 * y^12 + 3 * y^11 + 6 * y^10 + 4 + 2 * y^9) =
  15 * y^13 - y^12 + 12 * y^11 - 6 * y^10 - 4 * y^9 + 12 * y - 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3085_308559


namespace NUMINAMATH_CALUDE_chord_quadrilateral_probability_l3085_308522

/-- Given 7 points on a circle, the probability that 4 randomly selected chords
    form a convex quadrilateral is 1/171. -/
theorem chord_quadrilateral_probability :
  let n : ℕ := 7  -- number of points on the circle
  let k : ℕ := 4  -- number of chords selected
  let total_chords : ℕ := n.choose 2  -- total number of possible chords
  let total_selections : ℕ := total_chords.choose k  -- ways to select k chords
  let convex_quads : ℕ := n.choose k  -- number of convex quadrilaterals
  (convex_quads : ℚ) / total_selections = 1 / 171 := by
sorry

end NUMINAMATH_CALUDE_chord_quadrilateral_probability_l3085_308522


namespace NUMINAMATH_CALUDE_max_notebooks_charlie_can_buy_l3085_308587

theorem max_notebooks_charlie_can_buy (available : ℝ) (cost_per_notebook : ℝ) 
  (h1 : available = 12) (h2 : cost_per_notebook = 1.45) : 
  ⌊available / cost_per_notebook⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_notebooks_charlie_can_buy_l3085_308587


namespace NUMINAMATH_CALUDE_people_born_in_country_l3085_308552

theorem people_born_in_country (immigrants : ℕ) (new_residents : ℕ) 
  (h1 : immigrants = 16320) 
  (h2 : new_residents = 106491) : 
  new_residents - immigrants = 90171 := by
sorry

end NUMINAMATH_CALUDE_people_born_in_country_l3085_308552


namespace NUMINAMATH_CALUDE_marble_count_l3085_308543

/-- The total number of marbles owned by Albert, Angela, Allison, Addison, and Alex -/
def total_marbles (allison angela albert addison alex : ℕ) : ℕ :=
  allison + angela + albert + addison + alex

/-- Theorem stating the total number of marbles given the conditions -/
theorem marble_count :
  ∀ (allison angela albert addison alex : ℕ),
    allison = 28 →
    angela = allison + 8 →
    albert = 3 * angela →
    addison = 2 * albert →
    alex = allison + 5 →
    alex = angela / 2 →
    total_marbles allison angela albert addison alex = 421 := by
  sorry


end NUMINAMATH_CALUDE_marble_count_l3085_308543


namespace NUMINAMATH_CALUDE_cone_central_angle_l3085_308582

/-- Given a cone where the lateral area is twice the area of its base,
    prove that the central angle of the sector of the unfolded side is 180 degrees. -/
theorem cone_central_angle (r R : ℝ) (h : r > 0) (H : R > 0) : 
  π * r * R = 2 * π * r^2 → (180 : ℝ) * (2 * π * r) / (π * R) = 180 :=
by sorry

end NUMINAMATH_CALUDE_cone_central_angle_l3085_308582


namespace NUMINAMATH_CALUDE_trapezoid_area_l3085_308560

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  p : Point
  q : Point
  r : Point
  s : Point
  area : ℝ

/-- Represents a trapezoid -/
structure Trapezoid where
  t : Point
  u : Point
  v : Point
  s : Point

/-- Given a rectangle PQRS and points T, U, V forming a trapezoid TUVS, 
    prove that the area of TUVS is 10 square units -/
theorem trapezoid_area 
  (pqrs : Rectangle)
  (t : Point)
  (u : Point)
  (v : Point)
  (h1 : pqrs.area = 20)
  (h2 : t.x - pqrs.p.x = 2)
  (h3 : t.y = pqrs.p.y)
  (h4 : u.x - pqrs.q.x = 2)
  (h5 : u.y = pqrs.r.y)
  (h6 : v.x = pqrs.r.x)
  (h7 : v.y - t.y = pqrs.r.y - pqrs.p.y)
  : ∃ (tuvs : Trapezoid), tuvs.t = t ∧ tuvs.u = u ∧ tuvs.v = v ∧ tuvs.s = pqrs.s ∧ 
    (tuvs.v.x - tuvs.t.x + tuvs.s.x - tuvs.u.x) * (tuvs.u.y - tuvs.t.y) / 2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l3085_308560


namespace NUMINAMATH_CALUDE_symmetry_implies_axis_l3085_308578

/-- A function g : ℝ → ℝ with the property that g(x) = g(3-x) for all x ∈ ℝ -/
def SymmetricFunction (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g x = g (3 - x)

/-- The line x = 1.5 is an axis of symmetry for g -/
def IsAxisOfSymmetry (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g x = g (3 - x)

theorem symmetry_implies_axis (g : ℝ → ℝ) (h : SymmetricFunction g) :
  IsAxisOfSymmetry g := by sorry

end NUMINAMATH_CALUDE_symmetry_implies_axis_l3085_308578


namespace NUMINAMATH_CALUDE_trivia_team_points_l3085_308533

/-- Calculates the total points scored by a trivia team given the total number of members,
    the number of absent members, and the points scored by each attending member. -/
def total_points (total_members : ℕ) (absent_members : ℕ) (points_per_member : ℕ) : ℕ :=
  (total_members - absent_members) * points_per_member

/-- Proves that a trivia team with 15 total members, 6 absent members, and 3 points per
    attending member scores a total of 27 points. -/
theorem trivia_team_points :
  total_points 15 6 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_points_l3085_308533


namespace NUMINAMATH_CALUDE_triangles_from_circle_points_l3085_308525

def points_on_circle : ℕ := 10

theorem triangles_from_circle_points :
  Nat.choose points_on_circle 3 = 120 :=
by sorry

end NUMINAMATH_CALUDE_triangles_from_circle_points_l3085_308525


namespace NUMINAMATH_CALUDE_matching_color_probability_l3085_308599

def total_jellybeans_ava : ℕ := 4
def total_jellybeans_ben : ℕ := 8

def green_jellybeans_ava : ℕ := 2
def red_jellybeans_ava : ℕ := 2
def green_jellybeans_ben : ℕ := 2
def red_jellybeans_ben : ℕ := 3

theorem matching_color_probability :
  let p_green := (green_jellybeans_ava / total_jellybeans_ava) * (green_jellybeans_ben / total_jellybeans_ben)
  let p_red := (red_jellybeans_ava / total_jellybeans_ava) * (red_jellybeans_ben / total_jellybeans_ben)
  p_green + p_red = 5 / 16 := by
sorry

end NUMINAMATH_CALUDE_matching_color_probability_l3085_308599


namespace NUMINAMATH_CALUDE_smallest_c_value_l3085_308554

theorem smallest_c_value (c d : ℤ) : 
  (∃ (r₁ r₂ r₃ : ℤ), 
    r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧
    ∀ (x : ℤ), x^3 - c*x^2 + d*x - 3990 = (x - r₁) * (x - r₂) * (x - r₃)) →
  c ≥ 56 :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_value_l3085_308554


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_common_difference_l3085_308596

/-- Given a geometric sequence {a_n} where a_1+1, a_3+4, a_5+7 form an arithmetic sequence,
    the common difference of this arithmetic sequence is 3. -/
theorem geometric_arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_geometric : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q)
  (h_arithmetic : ∃ d : ℝ, (a 3 + 4) - (a 1 + 1) = d ∧ (a 5 + 7) - (a 3 + 4) = d) :
  ∃ d : ℝ, (a 3 + 4) - (a 1 + 1) = d ∧ d = 3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_common_difference_l3085_308596


namespace NUMINAMATH_CALUDE_systematic_sampling_third_group_l3085_308583

/-- Systematic sampling function -/
def systematicSample (totalItems : ℕ) (sampleSize : ℕ) (firstSelected : ℕ) : ℕ → ℕ :=
  fun n => firstSelected + (n - 1) * (totalItems / sampleSize)

theorem systematic_sampling_third_group 
  (totalItems : ℕ) 
  (sampleSize : ℕ) 
  (groupSize : ℕ) 
  (numGroups : ℕ) 
  (firstSelected : ℕ) :
  totalItems = 300 →
  sampleSize = 20 →
  groupSize = 20 →
  numGroups = 15 →
  firstSelected = 6 →
  totalItems = groupSize * numGroups →
  systematicSample totalItems sampleSize firstSelected 3 = 36 := by
  sorry

#check systematic_sampling_third_group

end NUMINAMATH_CALUDE_systematic_sampling_third_group_l3085_308583


namespace NUMINAMATH_CALUDE_log_64_4_l3085_308590

theorem log_64_4 : Real.log 4 / Real.log 64 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_64_4_l3085_308590


namespace NUMINAMATH_CALUDE_rotten_apples_l3085_308516

theorem rotten_apples (apples_per_crate : ℕ) (num_crates : ℕ) (boxes : ℕ) (apples_per_box : ℕ) :
  apples_per_crate = 180 →
  num_crates = 12 →
  boxes = 100 →
  apples_per_box = 20 →
  apples_per_crate * num_crates - boxes * apples_per_box = 160 :=
by
  sorry

end NUMINAMATH_CALUDE_rotten_apples_l3085_308516


namespace NUMINAMATH_CALUDE_reflection_coordinate_sum_l3085_308518

/-- Given a point A with coordinates (x, 7), prove that the sum of its coordinates
    and the coordinates of its reflection B over the x-axis is 2x. -/
theorem reflection_coordinate_sum (x : ℝ) : 
  let A : ℝ × ℝ := (x, 7)
  let B : ℝ × ℝ := (x, -7)  -- Reflection of A over x-axis
  (A.1 + A.2 + B.1 + B.2) = 2 * x := by
sorry

end NUMINAMATH_CALUDE_reflection_coordinate_sum_l3085_308518


namespace NUMINAMATH_CALUDE_largest_angle_in_special_quadrilateral_l3085_308504

/-- A quadrilateral with angles in the ratio 3:4:5:6 has its largest angle equal to 120°. -/
theorem largest_angle_in_special_quadrilateral : 
  ∀ (a b c d : ℝ), 
  a > 0 → b > 0 → c > 0 → d > 0 →
  (a + b + c + d = 360) →
  (b = 4/3 * a) → (c = 5/3 * a) → (d = 2 * a) →
  d = 120 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_quadrilateral_l3085_308504


namespace NUMINAMATH_CALUDE_power_equation_solution_l3085_308561

theorem power_equation_solution : ∃ K : ℕ, (4 ^ 5) * (2 ^ 3) = 2 ^ K ∧ K = 13 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l3085_308561


namespace NUMINAMATH_CALUDE_parabola_c_value_l3085_308503

/-- Represents a parabola of the form x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  p.x_coord 1 = -3 →  -- vertex at (-3, 1)
  p.x_coord 3 = -1 →  -- passes through (-1, 3)
  p.c = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l3085_308503


namespace NUMINAMATH_CALUDE_max_viewers_after_T_l3085_308592

/-- Represents a movie rating system --/
structure MovieRating where
  -- Total sum of scores
  scoreSum : ℕ
  -- Number of voters
  voterCount : ℕ
  -- Current rating (sum of scores divided by number of voters)
  rating : ℕ

/-- Theorem: Maximum number of viewers after moment T is 5 --/
theorem max_viewers_after_T (initialRating : MovieRating) : 
  initialRating.rating ≤ 10 →
  initialRating.rating > 0 →
  (∀ newScore : ℕ, newScore ≤ 10 →
    let newRating : MovieRating := {
      scoreSum := initialRating.scoreSum + newScore,
      voterCount := initialRating.voterCount + 1,
      rating := (initialRating.scoreSum + newScore) / (initialRating.voterCount + 1)
    }
    newRating.rating = initialRating.rating - 1) →
  (∃ (n : ℕ), n ≤ 5 ∧ 
    (∀ (m : ℕ), m > n → 
      ∃ (badRating : MovieRating), 
        badRating.rating ≤ 0 ∨ 
        badRating.rating ≥ initialRating.rating)) :=
by sorry


end NUMINAMATH_CALUDE_max_viewers_after_T_l3085_308592


namespace NUMINAMATH_CALUDE_solutions_count_l3085_308523

/-- The number of different integer solutions (x, y) for |x|+|y|=n -/
def num_solutions (n : ℕ) : ℕ := 4 * n

theorem solutions_count :
  (num_solutions 1 = 4) ∧
  (num_solutions 2 = 8) ∧
  (num_solutions 3 = 12) →
  ∀ n : ℕ, num_solutions n = 4 * n :=
by sorry

end NUMINAMATH_CALUDE_solutions_count_l3085_308523


namespace NUMINAMATH_CALUDE_ln_concave_l3085_308540

/-- The natural logarithm function is concave on the positive real numbers. -/
theorem ln_concave : ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 →
  Real.log ((x₁ + x₂) / 2) ≥ (Real.log x₁ + Real.log x₂) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ln_concave_l3085_308540


namespace NUMINAMATH_CALUDE_negation_of_all_odd_double_even_l3085_308509

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

def A : Set ℤ := {n : ℤ | is_odd n}
def B : Set ℤ := {n : ℤ | is_even n}

theorem negation_of_all_odd_double_even :
  (¬ ∀ x ∈ A, (2 * x) ∈ B) ↔ (∃ x ∈ A, (2 * x) ∉ B) :=
sorry

end NUMINAMATH_CALUDE_negation_of_all_odd_double_even_l3085_308509


namespace NUMINAMATH_CALUDE_only_D_positive_l3085_308521

theorem only_D_positive :
  let a := -3 + 7 - 5
  let b := (1 - 2) * 3
  let c := -16 / ((-3)^2)
  let d := -(2^4) * (-6)
  (a ≤ 0 ∧ b ≤ 0 ∧ c ≤ 0 ∧ d > 0) := by sorry

end NUMINAMATH_CALUDE_only_D_positive_l3085_308521


namespace NUMINAMATH_CALUDE_parabola_intercepts_sum_l3085_308511

theorem parabola_intercepts_sum (d e f : ℝ) : 
  (∀ x, 3 * x^2 - 9 * x + 5 = 3 * 0^2 - 9 * 0 + 5 → d = 3 * 0^2 - 9 * 0 + 5) →
  (3 * e^2 - 9 * e + 5 = 0) →
  (3 * f^2 - 9 * f + 5 = 0) →
  d + e + f = 8 := by
sorry

end NUMINAMATH_CALUDE_parabola_intercepts_sum_l3085_308511


namespace NUMINAMATH_CALUDE_sqrt_24_simplification_l3085_308570

theorem sqrt_24_simplification : Real.sqrt 24 = 2 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_24_simplification_l3085_308570


namespace NUMINAMATH_CALUDE_college_student_count_l3085_308539

/-- The total number of students at the college -/
def total_students : ℕ := 880

/-- The percentage of students enrolled in biology classes -/
def biology_enrollment_percentage : ℚ := 47.5 / 100

/-- The number of students not enrolled in biology classes -/
def students_not_in_biology : ℕ := 462

/-- Theorem stating the total number of students at the college -/
theorem college_student_count :
  total_students = students_not_in_biology / (1 - biology_enrollment_percentage) := by
  sorry

end NUMINAMATH_CALUDE_college_student_count_l3085_308539


namespace NUMINAMATH_CALUDE_power_seven_mod_twelve_l3085_308568

theorem power_seven_mod_twelve : 7^253 % 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_mod_twelve_l3085_308568


namespace NUMINAMATH_CALUDE_fraction_product_theorem_l3085_308513

theorem fraction_product_theorem : 
  (7 / 4 : ℚ) * (14 / 49 : ℚ) * (10 / 15 : ℚ) * (12 / 36 : ℚ) * 
  (21 / 14 : ℚ) * (40 / 80 : ℚ) * (33 / 22 : ℚ) * (16 / 64 : ℚ) = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_theorem_l3085_308513


namespace NUMINAMATH_CALUDE_complex_equality_l3085_308532

theorem complex_equality (z : ℂ) : 
  z = -1 + I ↔ Complex.abs (z - 2) = Complex.abs (z + 4) ∧ 
               Complex.abs (z - 2) = Complex.abs (z - 2*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l3085_308532


namespace NUMINAMATH_CALUDE_vector_dot_product_l3085_308520

/-- Given vectors a, b, c in ℝ², if a is parallel to b, then b · c = 10 -/
theorem vector_dot_product (a b c : ℝ × ℝ) : 
  a = (-1, 2) → b.1 = 2 → c = (7, 1) → (∃ k : ℝ, b = k • a) → b.1 * c.1 + b.2 * c.2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_l3085_308520


namespace NUMINAMATH_CALUDE_bus_seating_capacity_l3085_308515

theorem bus_seating_capacity : ∀ (x : ℕ),
  (4 * x + 30 = 5 * x - 10) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_bus_seating_capacity_l3085_308515


namespace NUMINAMATH_CALUDE_amount_left_after_purchases_l3085_308577

def calculate_discounted_price (price : ℚ) (discount_percent : ℚ) : ℚ :=
  price * (1 - discount_percent / 100)

def initial_amount : ℚ := 60

def frame_price : ℚ := 15
def frame_discount : ℚ := 10

def wheel_price : ℚ := 25
def wheel_discount : ℚ := 5

def seat_price : ℚ := 8
def seat_discount : ℚ := 15

def handlebar_price : ℚ := 5
def handlebar_discount : ℚ := 0

def bell_price : ℚ := 3
def bell_discount : ℚ := 0

def hat_price : ℚ := 10
def hat_discount : ℚ := 25

def total_cost : ℚ :=
  calculate_discounted_price frame_price frame_discount +
  calculate_discounted_price wheel_price wheel_discount +
  calculate_discounted_price seat_price seat_discount +
  calculate_discounted_price handlebar_price handlebar_discount +
  calculate_discounted_price bell_price bell_discount +
  calculate_discounted_price hat_price hat_discount

theorem amount_left_after_purchases :
  initial_amount - total_cost = 45 / 100 := by sorry

end NUMINAMATH_CALUDE_amount_left_after_purchases_l3085_308577


namespace NUMINAMATH_CALUDE_sin_30_degrees_l3085_308545

theorem sin_30_degrees : Real.sin (30 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_degrees_l3085_308545


namespace NUMINAMATH_CALUDE_gcd_50403_40302_l3085_308593

theorem gcd_50403_40302 : Nat.gcd 50403 40302 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_50403_40302_l3085_308593


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3085_308505

theorem simplify_sqrt_expression :
  Real.sqrt 5 - Real.sqrt 20 + Real.sqrt 45 - 2 * Real.sqrt 80 = -6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3085_308505


namespace NUMINAMATH_CALUDE_denis_neighbors_l3085_308527

-- Define the set of children
inductive Child : Type
| Anya : Child
| Borya : Child
| Vera : Child
| Gena : Child
| Denis : Child

-- Define the line as a function from position (1 to 5) to Child
def Line := Fin 5 → Child

-- Define the conditions
def is_valid_line (l : Line) : Prop :=
  -- Borya is at the beginning of the line
  l 1 = Child.Borya ∧
  -- Vera is next to Anya but not next to Gena
  (∃ i : Fin 4, (l i = Child.Vera ∧ l (i+1) = Child.Anya) ∨ (l (i+1) = Child.Vera ∧ l i = Child.Anya)) ∧
  (∀ i : Fin 4, ¬(l i = Child.Vera ∧ l (i+1) = Child.Gena) ∧ ¬(l (i+1) = Child.Vera ∧ l i = Child.Gena)) ∧
  -- Among Anya, Borya, and Gena, no two are standing next to each other
  (∀ i : Fin 4, ¬((l i = Child.Anya ∨ l i = Child.Borya ∨ l i = Child.Gena) ∧
                 (l (i+1) = Child.Anya ∨ l (i+1) = Child.Borya ∨ l (i+1) = Child.Gena)))

-- Theorem statement
theorem denis_neighbors (l : Line) (h : is_valid_line l) :
  (∃ i : Fin 4, (l i = Child.Anya ∧ l (i+1) = Child.Denis) ∨ (l (i+1) = Child.Anya ∧ l i = Child.Denis)) ∧
  (∃ j : Fin 4, (l j = Child.Gena ∧ l (j+1) = Child.Denis) ∨ (l (j+1) = Child.Gena ∧ l j = Child.Denis)) :=
by sorry

end NUMINAMATH_CALUDE_denis_neighbors_l3085_308527


namespace NUMINAMATH_CALUDE_f_constant_on_interval_inequality_solution_condition_l3085_308530

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x + 3|

-- Theorem 1: f(x) is constant on the interval [-3, 1]
theorem f_constant_on_interval :
  ∀ x y : ℝ, x ∈ Set.Icc (-3) 1 → y ∈ Set.Icc (-3) 1 → f x = f y :=
sorry

-- Theorem 2: For f(x) - a ≤ 0 to have a solution, a must be ≥ 4
theorem inequality_solution_condition :
  ∀ a : ℝ, (∃ x : ℝ, f x - a ≤ 0) ↔ a ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_f_constant_on_interval_inequality_solution_condition_l3085_308530


namespace NUMINAMATH_CALUDE_preceding_number_in_base_3_l3085_308531

def base_3_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (3^i)) 0

def decimal_to_base_3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 3) ((m % 3) :: acc)
  aux n []

theorem preceding_number_in_base_3 (N : Nat) (h : base_3_to_decimal [2, 1, 0, 1] = N) :
  decimal_to_base_3 (N - 1) = [2, 1, 0, 0] :=
sorry

end NUMINAMATH_CALUDE_preceding_number_in_base_3_l3085_308531


namespace NUMINAMATH_CALUDE_triangle_theorem_l3085_308517

noncomputable section

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.a * sin (t.A - t.C) = t.b * (sin t.A - sin t.B)

/-- The theorem to be proved -/
theorem triangle_theorem (t : Triangle) 
  (h1 : satisfiesCondition t) 
  (h2 : t.c = 4) : 
  t.C = π/3 ∧ 
  (∀ (t' : Triangle), satisfiesCondition t' → t'.c = 4 → 
    t.a + t.b + t.c ≤ 12) :=
sorry

end

end NUMINAMATH_CALUDE_triangle_theorem_l3085_308517


namespace NUMINAMATH_CALUDE_opposite_signs_and_larger_negative_l3085_308551

theorem opposite_signs_and_larger_negative (a b : ℝ) : 
  a + b < 0 → a * b < 0 → 
  ((a < 0 ∧ b > 0 ∧ |a| > |b|) ∨ (a > 0 ∧ b < 0 ∧ |a| < |b|)) := by
  sorry

end NUMINAMATH_CALUDE_opposite_signs_and_larger_negative_l3085_308551


namespace NUMINAMATH_CALUDE_unique_number_l3085_308537

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0

def digits_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem unique_number :
  ∃! n : ℕ, is_two_digit n ∧ is_odd n ∧ is_multiple_of_9 n ∧ is_perfect_square (digits_product n) ∧ n = 99 :=
sorry

end NUMINAMATH_CALUDE_unique_number_l3085_308537


namespace NUMINAMATH_CALUDE_min_magnitude_a_minus_c_l3085_308584

noncomputable section

-- Define the plane vectors
variable (a b c : ℝ × ℝ)

-- Define the conditions
def magnitude_a : ℝ := Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2))
def magnitude_b_minus_c : ℝ := Real.sqrt (((b.1 - c.1) ^ 2) + ((b.2 - c.2) ^ 2))
def angle_between_a_and_b : ℝ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (magnitude_a a * Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2))))

-- State the theorem
theorem min_magnitude_a_minus_c (h1 : magnitude_a a = 2)
                                (h2 : magnitude_b_minus_c b c = 1)
                                (h3 : angle_between_a_and_b a b = π / 3) :
  ∃ (min_value : ℝ), ∀ (a' b' c' : ℝ × ℝ),
    magnitude_a a' = 2 →
    magnitude_b_minus_c b' c' = 1 →
    angle_between_a_and_b a' b' = π / 3 →
    Real.sqrt (((a'.1 - c'.1) ^ 2) + ((a'.2 - c'.2) ^ 2)) ≥ min_value ∧
    min_value = Real.sqrt 3 - 1 :=
  sorry

end

end NUMINAMATH_CALUDE_min_magnitude_a_minus_c_l3085_308584


namespace NUMINAMATH_CALUDE_arrangement_theorem_l3085_308565

/-- The number of ways to arrange 4 different products in a row,
    with both product A and product B placed to the left of product C. -/
def arrangement_count : ℕ := 8

/-- The total number of products to arrange. -/
def total_products : ℕ := 4

/-- Theorem stating that the number of arrangements under the given conditions is 8. -/
theorem arrangement_theorem :
  arrangement_count = 8 ∧ total_products = 4 := by sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l3085_308565


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l3085_308589

theorem rectangular_prism_diagonal (l w h : ℝ) (hl : l = 15) (hw : w = 25) (hh : h = 12) :
  Real.sqrt (l^2 + w^2 + h^2) = Real.sqrt 994 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l3085_308589


namespace NUMINAMATH_CALUDE_star_four_three_l3085_308598

-- Define the star operation
def star (a b : ℝ) : ℝ := a^2 - a*b + b^2 + 2*a*b

-- State the theorem
theorem star_four_three : star 4 3 = 37 := by
  sorry

end NUMINAMATH_CALUDE_star_four_three_l3085_308598


namespace NUMINAMATH_CALUDE_deepak_age_l3085_308585

/-- Given that the ratio of Rahul's age to Deepak's age is 4:3 and 
    Rahul's age after 6 years will be 34 years, 
    prove that Deepak's present age is 21 years. -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 6 = 34 →
  deepak_age = 21 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l3085_308585


namespace NUMINAMATH_CALUDE_roshesmina_piggy_bank_pennies_l3085_308579

/-- Calculates the total number of pennies in a piggy bank -/
def total_pennies (compartments : ℕ) (initial_pennies : ℕ) (added_pennies : ℕ) : ℕ :=
  compartments * (initial_pennies + added_pennies)

/-- Theorem: The total number of pennies in Roshesmina's piggy bank -/
theorem roshesmina_piggy_bank_pennies :
  total_pennies 12 2 6 = 96 := by
  sorry

end NUMINAMATH_CALUDE_roshesmina_piggy_bank_pennies_l3085_308579


namespace NUMINAMATH_CALUDE_abs_neg_one_third_l3085_308594

theorem abs_neg_one_third : |(-1 : ℚ) / 3| = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_one_third_l3085_308594


namespace NUMINAMATH_CALUDE_kola_sugar_percentage_l3085_308507

/-- Calculates the percentage of sugar in a kola solution after adding ingredients -/
theorem kola_sugar_percentage
  (initial_volume : Real)
  (initial_water_percent : Real)
  (initial_kola_percent : Real)
  (added_sugar : Real)
  (added_water : Real)
  (added_kola : Real)
  (h1 : initial_volume = 340)
  (h2 : initial_water_percent = 88)
  (h3 : initial_kola_percent = 5)
  (h4 : added_sugar = 3.2)
  (h5 : added_water = 10)
  (h6 : added_kola = 6.8) :
  let initial_sugar_percent := 100 - initial_water_percent - initial_kola_percent
  let initial_water := initial_volume * initial_water_percent / 100
  let initial_kola := initial_volume * initial_kola_percent / 100
  let initial_sugar := initial_volume * initial_sugar_percent / 100
  let final_water := initial_water + added_water
  let final_kola := initial_kola + added_kola
  let final_sugar := initial_sugar + added_sugar
  let final_volume := final_water + final_kola + final_sugar
  final_sugar / final_volume * 100 = 7.5 := by
  sorry


end NUMINAMATH_CALUDE_kola_sugar_percentage_l3085_308507


namespace NUMINAMATH_CALUDE_power_difference_equality_l3085_308501

theorem power_difference_equality : (3^2)^3 - (2^3)^2 = 665 := by sorry

end NUMINAMATH_CALUDE_power_difference_equality_l3085_308501


namespace NUMINAMATH_CALUDE_integral_proof_l3085_308566

theorem integral_proof (x : ℝ) (h : x ≠ 2 ∧ x ≠ -2) : 
  (deriv (fun x => Real.log (abs (x - 2)) - 3 / (2 * (x + 2)^2))) x = 
  (x^3 + 6*x^2 + 15*x + 2) / ((x - 2) * (x + 2)^3) :=
by sorry

end NUMINAMATH_CALUDE_integral_proof_l3085_308566


namespace NUMINAMATH_CALUDE_train_journey_times_l3085_308556

/-- Proves that given the conditions of two trains running late, their usual journey times are both 2 hours -/
theorem train_journey_times (speed_ratio_A speed_ratio_B : ℚ) (delay_A delay_B : ℚ) 
  (h1 : speed_ratio_A = 4/5)
  (h2 : speed_ratio_B = 3/4)
  (h3 : delay_A = 1/2)  -- 30 minutes in hours
  (h4 : delay_B = 2/3)  -- 40 minutes in hours
  : ∃ (T_A T_B : ℚ), T_A = 2 ∧ T_B = 2 ∧ 
    (1/speed_ratio_A) * T_A = T_A + delay_A ∧
    (1/speed_ratio_B) * T_B = T_B + delay_B :=
by sorry


end NUMINAMATH_CALUDE_train_journey_times_l3085_308556


namespace NUMINAMATH_CALUDE_cuboid_volume_l3085_308506

/-- A cuboid with given height and base area has the specified volume. -/
theorem cuboid_volume (height : ℝ) (base_area : ℝ) :
  height = 13 → base_area = 14 → height * base_area = 182 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volume_l3085_308506


namespace NUMINAMATH_CALUDE_advanced_purchase_ticket_price_l3085_308546

/-- Given information about ticket sales for an art exhibition, prove the price of advanced-purchase tickets. -/
theorem advanced_purchase_ticket_price
  (total_tickets : ℕ)
  (total_revenue : ℚ)
  (door_price : ℚ)
  (advanced_tickets : ℕ)
  (h_total_tickets : total_tickets = 140)
  (h_total_revenue : total_revenue = 1720)
  (h_door_price : door_price = 14)
  (h_advanced_tickets : advanced_tickets = 100) :
  ∃ (advanced_price : ℚ),
    advanced_price * advanced_tickets + door_price * (total_tickets - advanced_tickets) = total_revenue ∧
    advanced_price = 11.60 :=
by sorry

end NUMINAMATH_CALUDE_advanced_purchase_ticket_price_l3085_308546


namespace NUMINAMATH_CALUDE_subtraction_problem_l3085_308573

theorem subtraction_problem : 943 - 87 = 856 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l3085_308573


namespace NUMINAMATH_CALUDE_drain_time_for_specific_pumps_l3085_308575

/-- Represents the time taken to drain a lake with three pumps working together -/
def drain_time (rate1 rate2 rate3 : ℚ) : ℚ :=
  1 / (rate1 + rate2 + rate3)

/-- Theorem stating the time taken to drain a lake with three specific pumps -/
theorem drain_time_for_specific_pumps :
  drain_time (1/9) (1/6) (1/12) = 36/13 := by
  sorry

end NUMINAMATH_CALUDE_drain_time_for_specific_pumps_l3085_308575


namespace NUMINAMATH_CALUDE_quadratic_root_sqrt5_minus3_l3085_308519

theorem quadratic_root_sqrt5_minus3 : ∃ (a b c : ℚ), 
  a = 1 ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = Real.sqrt 5 - 3 ∨ x = -Real.sqrt 5 - 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_sqrt5_minus3_l3085_308519


namespace NUMINAMATH_CALUDE_divisibility_property_l3085_308541

def sequence_a : ℕ → ℕ
  | 0 => 3
  | n + 1 => (2 * (n + 2) * sequence_a n - (n + 1) - 2) / (n + 1)

theorem divisibility_property (p : ℕ) (hp : Nat.Prime p) (hp_odd : Odd p) :
  ∃ m : ℕ, p ∣ sequence_a m ∧ p ∣ sequence_a (m + 1) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_property_l3085_308541


namespace NUMINAMATH_CALUDE_samara_detailing_cost_samara_detailing_cost_proof_l3085_308542

/-- Proves that Samara's spending on detailing equals $79 given the problem conditions -/
theorem samara_detailing_cost : ℕ → Prop :=
  fun (detailing_cost : ℕ) =>
    let alberto_total : ℕ := 2457
    let samara_oil : ℕ := 25
    let samara_tires : ℕ := 467
    let difference : ℕ := 1886
    alberto_total = samara_oil + samara_tires + detailing_cost + difference →
    detailing_cost = 79

/-- The proof of the theorem -/
theorem samara_detailing_cost_proof : samara_detailing_cost 79 := by
  sorry

end NUMINAMATH_CALUDE_samara_detailing_cost_samara_detailing_cost_proof_l3085_308542


namespace NUMINAMATH_CALUDE_jason_music_store_expenditure_l3085_308555

/-- The total cost of Jason's music store purchases --/
def total_cost : ℚ :=
  142.46 + 8.89 + 7.00 + 15.75 + 12.95 + 36.50 + 5.25

/-- Theorem stating that Jason's total music store expenditure is $229.80 --/
theorem jason_music_store_expenditure :
  total_cost = 229.80 := by sorry

end NUMINAMATH_CALUDE_jason_music_store_expenditure_l3085_308555


namespace NUMINAMATH_CALUDE_expression_simplification_l3085_308586

theorem expression_simplification (x : ℝ) :
  x - 3 * (2 + x) + 4 * (2 - x) - 5 * (1 + 3 * x) + 2 * x^2 = 2 * x^2 - 21 * x - 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3085_308586


namespace NUMINAMATH_CALUDE_matrix_not_invertible_sum_l3085_308526

def matrix (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x + y, x, y],
    ![x, y + z, y],
    ![y, x, x + z]]

theorem matrix_not_invertible_sum (x y z : ℝ) :
  ¬(IsUnit (Matrix.det (matrix x y z))) →
  x + y + z = 0 →
  x / (y + z) + y / (x + z) + z / (x + y) = -3 := by
  sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_sum_l3085_308526


namespace NUMINAMATH_CALUDE_quadratic_polynomial_conditions_l3085_308529

theorem quadratic_polynomial_conditions (p : ℝ → ℝ) :
  (∀ x, p x = 2 * x^2 - 3 * x - 1) →
  p (-2) = 13 ∧ p 1 = -2 ∧ p 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_conditions_l3085_308529


namespace NUMINAMATH_CALUDE_twin_brothers_age_l3085_308588

theorem twin_brothers_age :
  ∀ x : ℕ,
  (x + 1) * (x + 1) = x * x + 17 →
  x = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_twin_brothers_age_l3085_308588


namespace NUMINAMATH_CALUDE_inequality_proof_l3085_308549

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  1 / (x^3 * y) + 1 / (y^3 * z) + 1 / (z^3 * x) ≥ x * y + y * z + z * x := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3085_308549


namespace NUMINAMATH_CALUDE_cuboid_dimensions_sum_l3085_308500

theorem cuboid_dimensions_sum (A B C : ℝ) (h1 : A * B = 45) (h2 : B * C = 80) (h3 : C * A = 180) :
  A + B + C = 145 / 9 := by
sorry

end NUMINAMATH_CALUDE_cuboid_dimensions_sum_l3085_308500


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l3085_308569

def A : Set ℝ := {-1, 0, 1}
def B (a : ℝ) : Set ℝ := {a + 1, 2 * a}

theorem intersection_implies_a_value :
  ∀ a : ℝ, (A ∩ B a = {0}) → a = -1 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l3085_308569


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l3085_308538

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_parallel_planes
  (α β : Plane) (m n : Line)
  (h1 : parallel_planes α β)
  (h2 : perpendicular_line_plane m α)
  (h3 : parallel_line_plane n β) :
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l3085_308538


namespace NUMINAMATH_CALUDE_boxes_shipped_this_week_l3085_308567

/-- Represents the number of pomelos in a dozen -/
def dozen : ℕ := 12

/-- Represents the number of boxes shipped last week -/
def last_week_boxes : ℕ := 10

/-- Represents the total number of pomelos shipped last week -/
def last_week_pomelos : ℕ := 240

/-- Represents the number of dozens of pomelos shipped this week -/
def this_week_dozens : ℕ := 60

/-- Calculates the number of boxes shipped this week -/
def boxes_this_week : ℕ :=
  (this_week_dozens * dozen) / (last_week_pomelos / last_week_boxes)

theorem boxes_shipped_this_week :
  boxes_this_week = 30 := by sorry

end NUMINAMATH_CALUDE_boxes_shipped_this_week_l3085_308567


namespace NUMINAMATH_CALUDE_constant_term_expansion_l3085_308595

theorem constant_term_expansion (n : ℕ) : 
  (∃ (k : ℕ), (Nat.choose n (2*n/3 : ℕ)) = 15 ∧ 2*n/3 = k) → n = 6 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l3085_308595


namespace NUMINAMATH_CALUDE_clothing_problem_l3085_308524

/-- Calculates the remaining clothing pieces after donations and discarding --/
def remaining_clothing (initial : ℕ) (donated1 : ℕ) (donated2_multiplier : ℕ) (discarded : ℕ) : ℕ :=
  initial - (donated1 + donated1 * donated2_multiplier) - discarded

/-- Theorem stating that given the specific values in the problem, 
    the remaining clothing pieces is 65 --/
theorem clothing_problem : 
  remaining_clothing 100 5 3 15 = 65 := by
  sorry

end NUMINAMATH_CALUDE_clothing_problem_l3085_308524


namespace NUMINAMATH_CALUDE_factors_of_180_l3085_308510

def number_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem factors_of_180 : number_of_factors 180 = 18 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_180_l3085_308510


namespace NUMINAMATH_CALUDE_green_tiles_in_50th_row_l3085_308548

/-- Represents the number of tiles in a row of the tiling pattern. -/
def num_tiles (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the number of green tiles in a row of the tiling pattern. -/
def num_green_tiles (n : ℕ) : ℕ := (num_tiles n - 1) / 2

theorem green_tiles_in_50th_row :
  num_green_tiles 50 = 49 := by sorry

end NUMINAMATH_CALUDE_green_tiles_in_50th_row_l3085_308548


namespace NUMINAMATH_CALUDE_cos_2alpha_value_l3085_308562

theorem cos_2alpha_value (α : ℝ) (h : (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1/2) :
  Real.cos (2 * α) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_value_l3085_308562


namespace NUMINAMATH_CALUDE_some_number_value_l3085_308576

theorem some_number_value (x y : ℝ) 
  (h1 : x / y = 3 / 2)
  (h2 : (7 * x + y) / (x - y) = 23) :
  y = 1 := by
sorry

end NUMINAMATH_CALUDE_some_number_value_l3085_308576


namespace NUMINAMATH_CALUDE_polygon_perimeter_bounds_l3085_308544

theorem polygon_perimeter_bounds :
  ∃ (m₃ m₄ m₅ m₆ m₇ m₈ m₉ m₁₀ : ℝ),
    (abs m₃ ≤ 3) ∧
    (abs m₄ ≤ 5) ∧
    (abs m₅ ≤ 7) ∧
    (abs m₆ ≤ 9) ∧
    (abs m₇ ≤ 12) ∧
    (abs m₈ ≤ 14) ∧
    (abs m₉ ≤ 16) ∧
    (abs m₁₀ ≤ 19) ∧
    (m₃ ≤ m₄) ∧ (m₄ ≤ m₅) ∧ (m₅ ≤ m₆) ∧ (m₆ ≤ m₇) ∧
    (m₇ ≤ m₈) ∧ (m₈ ≤ m₉) ∧ (m₉ ≤ m₁₀) := by
  sorry


end NUMINAMATH_CALUDE_polygon_perimeter_bounds_l3085_308544


namespace NUMINAMATH_CALUDE_cos_difference_l3085_308553

theorem cos_difference (A B : Real) 
  (h1 : Real.sin A + Real.sin B = 1/2) 
  (h2 : Real.cos A + Real.cos B = 2) : 
  Real.cos (A - B) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_cos_difference_l3085_308553
