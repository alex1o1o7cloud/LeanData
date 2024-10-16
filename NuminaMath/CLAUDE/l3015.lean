import Mathlib

namespace NUMINAMATH_CALUDE_electronic_components_probability_l3015_301577

theorem electronic_components_probability (p : ℝ) 
  (h1 : 0 ≤ p ∧ p ≤ 1) 
  (h2 : 1 - (1 - p)^3 = 0.999) : 
  p = 0.9 := by
sorry

end NUMINAMATH_CALUDE_electronic_components_probability_l3015_301577


namespace NUMINAMATH_CALUDE_overlapping_area_is_75_over_8_l3015_301581

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ

/-- The area of the overlapping region formed by two 30-60-90 triangles -/
def overlapping_area (t1 t2 : Triangle30_60_90) : ℝ :=
  sorry

/-- The theorem stating the area of the overlapping region -/
theorem overlapping_area_is_75_over_8 (t1 t2 : Triangle30_60_90) 
  (h1 : t1.hypotenuse = 10)
  (h2 : t2.hypotenuse = 10)
  (h3 : overlapping_area t1 t2 ≠ 0) : 
  overlapping_area t1 t2 = 75 / 8 := by
  sorry

end NUMINAMATH_CALUDE_overlapping_area_is_75_over_8_l3015_301581


namespace NUMINAMATH_CALUDE_sum_of_squares_l3015_301511

theorem sum_of_squares (a b c : ℝ) 
  (eq1 : a^2 + 3*b = 10)
  (eq2 : b^2 + 5*c = 0)
  (eq3 : c^2 + 7*a = -21) :
  a^2 + b^2 + c^2 = 83/4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3015_301511


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3015_301508

-- Define the sets M and N
def M : Set ℝ := {x | x < 0 ∨ x > 2}
def N : Set ℝ := {x | x^2 - 4*x + 3 < 0}

-- State the theorem
theorem complement_intersection_theorem :
  (N \ (M ∩ N)) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3015_301508


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3015_301557

theorem arithmetic_mean_problem (x : ℝ) : 
  ((x + 10) + 17 + (2 * x) + 15 + (2 * x + 6)) / 5 = 26 → x = 82 / 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3015_301557


namespace NUMINAMATH_CALUDE_sinusoidal_symmetry_center_l3015_301503

/-- Given a sinusoidal function with specific properties, prove that one of its symmetry centers has coordinates (-2π/3, 0) -/
theorem sinusoidal_symmetry_center 
  (f : ℝ → ℝ) 
  (ω φ : ℝ) 
  (h1 : ∀ x, f x = Real.sin (ω * x + φ))
  (h2 : ω > 0)
  (h3 : |φ| < π / 2)
  (h4 : ∀ x, f (x + 4 * π) = f x)
  (h5 : ∀ t, t > 0 → (∀ x, f (x + t) = f x) → t ≥ 4 * π)
  (h6 : f (π / 3) = 1) :
  ∃ k : ℤ, f (x + (-2 * π / 3)) = -f (-x + (-2 * π / 3)) := by
  sorry

end NUMINAMATH_CALUDE_sinusoidal_symmetry_center_l3015_301503


namespace NUMINAMATH_CALUDE_second_derivative_value_l3015_301588

def f (q : ℝ) : ℝ := 3 * q - 3

theorem second_derivative_value (q : ℝ) : f (f q) = 210 → q = 74 / 3 := by
  sorry

end NUMINAMATH_CALUDE_second_derivative_value_l3015_301588


namespace NUMINAMATH_CALUDE_f_properties_l3015_301517

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + (x - 2) / (x + 1)

theorem f_properties (a : ℝ) (h : a > 1) :
  (∀ x y : ℝ, x > -1 → y > -1 → x < y → f a x < f a y) ∧
  (∀ x : ℝ, x < 0 → f a x ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3015_301517


namespace NUMINAMATH_CALUDE_line_and_circle_properties_l3015_301518

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 2 * k = 0

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 8

-- Define the line l₀
def line_l0 (x y : ℝ) : Prop := x - 2 * y + 2 = 0

theorem line_and_circle_properties :
  (∃ k : ℝ, ∀ x y : ℝ, line_l k x y → line_l0 x y → x = y) ∧
  (∀ k : ℝ, ∃ x y : ℝ, line_l k x y ∧ circle_O x y) :=
sorry

end NUMINAMATH_CALUDE_line_and_circle_properties_l3015_301518


namespace NUMINAMATH_CALUDE_circle_area_and_circumference_l3015_301520

theorem circle_area_and_circumference (r : ℝ) (h : r > 0) :
  ∃ (A C : ℝ),
    A = π * r^2 ∧
    C = 2 * π * r :=
by sorry

end NUMINAMATH_CALUDE_circle_area_and_circumference_l3015_301520


namespace NUMINAMATH_CALUDE_team_total_points_l3015_301525

theorem team_total_points (player_points : ℕ) (percentage : ℚ) (h1 : player_points = 35) (h2 : percentage = 1/2) :
  player_points / percentage = 70 := by
  sorry

end NUMINAMATH_CALUDE_team_total_points_l3015_301525


namespace NUMINAMATH_CALUDE_equation_solvable_for_small_primes_l3015_301559

theorem equation_solvable_for_small_primes :
  ∀ p : ℕ, p ≤ 100 → Prime p → ∃ x y : ℕ, y^37 ≡ x^3 + 11 [ZMOD p] :=
by sorry

end NUMINAMATH_CALUDE_equation_solvable_for_small_primes_l3015_301559


namespace NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l3015_301561

theorem midpoint_sum_equals_vertex_sum (a b c : ℝ) (h1 : a + b + c = 15) (h2 : a - b = 3) :
  (a + b) / 2 + (b + c) / 2 + (c + a) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l3015_301561


namespace NUMINAMATH_CALUDE_chef_nuts_total_weight_l3015_301585

theorem chef_nuts_total_weight (almond_weight pecan_weight : Real) 
  (h1 : almond_weight = 0.14)
  (h2 : pecan_weight = 0.38) :
  almond_weight + pecan_weight = 0.52 := by
sorry

end NUMINAMATH_CALUDE_chef_nuts_total_weight_l3015_301585


namespace NUMINAMATH_CALUDE_platform_length_l3015_301599

/-- The length of the platform given a train's characteristics and crossing times. -/
theorem platform_length
  (train_length : ℝ)
  (time_platform : ℝ)
  (time_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_platform = 45)
  (h3 : time_pole = 18) :
  let speed := train_length / time_pole
  let total_distance := speed * time_platform
  train_length + (total_distance - train_length) = 450 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l3015_301599


namespace NUMINAMATH_CALUDE_max_distance_to_complex_point_l3015_301549

open Complex

theorem max_distance_to_complex_point (z : ℂ) :
  let z₁ : ℂ := 2 - 2*I
  (abs z = 1) →
  (∀ w : ℂ, abs w = 1 → abs (w - z₁) ≤ 2*Real.sqrt 2 + 1) ∧
  (∃ w : ℂ, abs w = 1 ∧ abs (w - z₁) = 2*Real.sqrt 2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_max_distance_to_complex_point_l3015_301549


namespace NUMINAMATH_CALUDE_stair_step_24th_row_white_squares_l3015_301538

/-- Represents the number of squares in a row of the stair-step figure -/
def total_squares (n : ℕ) : ℕ := 1 + 2 * (n - 1)

/-- Represents the number of white squares in a row of the stair-step figure -/
def white_squares (n : ℕ) : ℕ := (total_squares n - 2) / 2 + (total_squares n - 2) % 2

/-- Theorem stating that the 24th row of the stair-step figure contains 23 white squares -/
theorem stair_step_24th_row_white_squares :
  white_squares 24 = 23 := by sorry

end NUMINAMATH_CALUDE_stair_step_24th_row_white_squares_l3015_301538


namespace NUMINAMATH_CALUDE_incorrect_multiplication_l3015_301597

theorem incorrect_multiplication : (79133 * 111107) % 9 ≠ 8792240231 % 9 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_multiplication_l3015_301597


namespace NUMINAMATH_CALUDE_handshake_theorem_l3015_301567

def corporate_event (n : ℕ) (completed_handshakes : ℕ) : Prop :=
  let total_handshakes := n * (n - 1) / 2
  total_handshakes - completed_handshakes = 42

theorem handshake_theorem :
  corporate_event 10 3 := by
  sorry

end NUMINAMATH_CALUDE_handshake_theorem_l3015_301567


namespace NUMINAMATH_CALUDE_number_problem_l3015_301556

theorem number_problem (x : ℝ) (h : x - 7 = 9) : 5 * x = 80 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3015_301556


namespace NUMINAMATH_CALUDE_unique_solution_cubic_equation_l3015_301564

theorem unique_solution_cubic_equation :
  ∃! x : ℝ, x ≠ 0 ∧ x ≠ 5 ∧ (3 * x^3 - 15 * x^2) / (x^2 - 5 * x) = x - 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_equation_l3015_301564


namespace NUMINAMATH_CALUDE_holly_initial_amount_l3015_301591

/-- The amount of chocolate milk Holly drinks at breakfast, in ounces. -/
def breakfast_consumption : ℕ := 8

/-- The amount of chocolate milk Holly drinks at lunch, in ounces. -/
def lunch_consumption : ℕ := 8

/-- The amount of chocolate milk Holly drinks at dinner, in ounces. -/
def dinner_consumption : ℕ := 8

/-- The amount of chocolate milk Holly ends the day with, in ounces. -/
def end_of_day_amount : ℕ := 56

/-- The size of the new container Holly buys during lunch, in ounces. -/
def new_container_size : ℕ := 64

/-- Theorem stating that Holly began the day with 80 ounces of chocolate milk. -/
theorem holly_initial_amount :
  breakfast_consumption + lunch_consumption + dinner_consumption + end_of_day_amount = 80 :=
by sorry

end NUMINAMATH_CALUDE_holly_initial_amount_l3015_301591


namespace NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l3015_301550

theorem isosceles_triangle_largest_angle (a b c : ℝ) : 
  -- The triangle is isosceles
  a = b →
  -- One of the angles opposite an equal side is 50°
  c = 50 →
  -- The sum of angles in a triangle is 180°
  a + b + c = 180 →
  -- The largest angle is 80°
  max a (max b c) = 80 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l3015_301550


namespace NUMINAMATH_CALUDE_angle_DAE_measure_l3015_301563

-- Define the points
variable (A B C D E F : Point)

-- Define the shapes
def is_equilateral_triangle (A B C : Point) : Prop := sorry

def is_regular_pentagon (B C D E F : Point) : Prop := sorry

-- Define the shared side
def share_side (A B C D E F : Point) : Prop := sorry

-- Define the angle measurement
def angle_measure (A D E : Point) : ℝ := sorry

-- Theorem statement
theorem angle_DAE_measure 
  (h1 : is_equilateral_triangle A B C) 
  (h2 : is_regular_pentagon B C D E F) 
  (h3 : share_side A B C D E F) : 
  angle_measure A D E = 108 := by sorry

end NUMINAMATH_CALUDE_angle_DAE_measure_l3015_301563


namespace NUMINAMATH_CALUDE_triple_equation_solutions_l3015_301592

theorem triple_equation_solutions :
  ∀ a b c : ℝ, 
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 →
    a^2 + a*b = c ∧ 
    b^2 + b*c = a ∧ 
    c^2 + c*a = b →
    (a = 0 ∧ b = 0 ∧ c = 0) ∨ 
    (a = 1/2 ∧ b = 1/2 ∧ c = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_triple_equation_solutions_l3015_301592


namespace NUMINAMATH_CALUDE_flow_rates_theorem_l3015_301516

/-- Represents an irrigation channel in the system -/
inductive Channel
| AB | BC | CD | DE | BG | GD | GF | FE

/-- Represents a node in the irrigation system -/
inductive Node
| A | B | C | D | E | F | G | H

/-- The flow rate in a channel -/
def flow_rate (c : Channel) : ℝ := sorry

/-- The total input flow rate -/
def q₀ : ℝ := sorry

/-- The irrigation system is symmetric -/
axiom symmetric_system : ∀ c₁ c₂ : Channel, flow_rate c₁ = flow_rate c₂

/-- The sum of flow rates remains constant along any path -/
axiom constant_flow_sum : ∀ path : List Channel, 
  (∀ c ∈ path, c ∈ [Channel.AB, Channel.BC, Channel.CD, Channel.DE, Channel.BG, Channel.GD, Channel.GF, Channel.FE]) →
  (List.sum (path.map flow_rate) = q₀)

/-- Theorem stating the flow rates in channels DE, BC, and GF -/
theorem flow_rates_theorem :
  flow_rate Channel.DE = (4/7) * q₀ ∧
  flow_rate Channel.BC = (2/7) * q₀ ∧
  flow_rate Channel.GF = (3/7) * q₀ := by
  sorry

end NUMINAMATH_CALUDE_flow_rates_theorem_l3015_301516


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3015_301593

theorem fraction_to_decimal :
  (53 : ℚ) / (4 * 5^7) = (1325 : ℚ) / 10^7 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3015_301593


namespace NUMINAMATH_CALUDE_stamps_per_page_l3015_301582

theorem stamps_per_page (a b c : ℕ) (ha : a = 1200) (hb : b = 1800) (hc : c = 2400) :
  Nat.gcd a (Nat.gcd b c) = 600 := by
  sorry

end NUMINAMATH_CALUDE_stamps_per_page_l3015_301582


namespace NUMINAMATH_CALUDE_max_residents_per_apartment_is_four_l3015_301523

/-- Represents a block of flats -/
structure BlockOfFlats where
  floors : ℕ
  apartments_per_floor_type1 : ℕ
  apartments_per_floor_type2 : ℕ
  max_residents : ℕ

/-- Calculates the maximum number of residents per apartment -/
def max_residents_per_apartment (block : BlockOfFlats) : ℕ :=
  block.max_residents / ((block.floors / 2) * block.apartments_per_floor_type1 + 
                         (block.floors / 2) * block.apartments_per_floor_type2)

/-- Theorem stating the maximum number of residents per apartment -/
theorem max_residents_per_apartment_is_four (block : BlockOfFlats) 
  (h1 : block.floors = 12)
  (h2 : block.apartments_per_floor_type1 = 6)
  (h3 : block.apartments_per_floor_type2 = 5)
  (h4 : block.max_residents = 264) :
  max_residents_per_apartment block = 4 := by
  sorry

#eval max_residents_per_apartment { 
  floors := 12, 
  apartments_per_floor_type1 := 6, 
  apartments_per_floor_type2 := 5, 
  max_residents := 264 
}

end NUMINAMATH_CALUDE_max_residents_per_apartment_is_four_l3015_301523


namespace NUMINAMATH_CALUDE_function_equal_to_inverse_is_identity_l3015_301537

-- Define an increasing function from R to R
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- Define the theorem
theorem function_equal_to_inverse_is_identity
  (f : ℝ → ℝ)
  (h_increasing : IncreasingFunction f)
  (h_inverse : ∀ x : ℝ, f x = Function.invFun f x) :
  ∀ x : ℝ, f x = x :=
sorry

end NUMINAMATH_CALUDE_function_equal_to_inverse_is_identity_l3015_301537


namespace NUMINAMATH_CALUDE_min_value_theorem_l3015_301514

/-- Given a function f(x) = (1/3)ax³ + (1/2)bx² - x with a > 0 and b > 0,
    if f has a local minimum at x = 1, then the minimum value of (1/a) + (4/b) is 9 -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f : ℝ → ℝ := λ x ↦ (1/3) * a * x^3 + (1/2) * b * x^2 - x
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≥ f 1) →
  (∀ p q : ℝ, p > 0 → q > 0 → p + q = 1 → (1/p) + (4/q) ≥ 9) ∧
  (∃ p q : ℝ, p > 0 ∧ q > 0 ∧ p + q = 1 ∧ (1/p) + (4/q) = 9) :=
by sorry


end NUMINAMATH_CALUDE_min_value_theorem_l3015_301514


namespace NUMINAMATH_CALUDE_parabola_a_value_l3015_301534

/-- A parabola with equation y = ax^2 + bx + c, vertex at (3, -2), and passing through (0, -50) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ
  vertex_y : ℝ
  point_x : ℝ
  point_y : ℝ
  vertex_condition : vertex_y = a * vertex_x^2 + b * vertex_x + c
  point_condition : point_y = a * point_x^2 + b * point_x + c

/-- The theorem stating that the value of 'a' for the given parabola is -16/3 -/
theorem parabola_a_value (p : Parabola) 
  (h1 : p.vertex_x = 3) 
  (h2 : p.vertex_y = -2) 
  (h3 : p.point_x = 0) 
  (h4 : p.point_y = -50) : 
  p.a = -16/3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_a_value_l3015_301534


namespace NUMINAMATH_CALUDE_eggs_remaining_l3015_301572

/-- Given a box with 47 eggs, if 5 eggs are removed, then 42 eggs remain in the box. -/
theorem eggs_remaining (initial_eggs : Nat) (removed_eggs : Nat) (remaining_eggs : Nat) : 
  initial_eggs = 47 → removed_eggs = 5 → remaining_eggs = initial_eggs - removed_eggs → remaining_eggs = 42 := by
  sorry

end NUMINAMATH_CALUDE_eggs_remaining_l3015_301572


namespace NUMINAMATH_CALUDE_seventh_term_largest_coefficient_l3015_301590

def binomial_expansion (x : ℝ) (n : ℕ) : ℕ → ℝ
  | r => (-1)^r * (Nat.choose n r) * x^(2*n - 3*r)

theorem seventh_term_largest_coefficient :
  ∃ (x : ℝ), ∀ (r : ℕ), r ≠ 6 →
    |binomial_expansion x 11 6| ≥ |binomial_expansion x 11 r| :=
sorry

end NUMINAMATH_CALUDE_seventh_term_largest_coefficient_l3015_301590


namespace NUMINAMATH_CALUDE_temperature_stats_l3015_301512

def temperatures : List ℝ := [12, 9, 10, 6, 11, 12, 17]

def median (l : List ℝ) : ℝ := sorry

def range (l : List ℝ) : ℝ := sorry

theorem temperature_stats :
  median temperatures = 11 ∧ range temperatures = 11 := by sorry

end NUMINAMATH_CALUDE_temperature_stats_l3015_301512


namespace NUMINAMATH_CALUDE_compare_expressions_l3015_301546

theorem compare_expressions : 3 - Real.sqrt 2 > 4 - 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_compare_expressions_l3015_301546


namespace NUMINAMATH_CALUDE_base4_equals_base2_l3015_301541

-- Define a function to convert a number from base 4 to decimal
def base4ToDecimal (n : ℕ) : ℕ := sorry

-- Define a function to convert a number from base 2 to decimal
def base2ToDecimal (n : ℕ) : ℕ := sorry

theorem base4_equals_base2 : base4ToDecimal 1010 = base2ToDecimal 1000100 := by sorry

end NUMINAMATH_CALUDE_base4_equals_base2_l3015_301541


namespace NUMINAMATH_CALUDE_warehouse_bins_count_l3015_301540

/-- Calculates the total number of bins in a warehouse given specific conditions. -/
def totalBins (totalCapacity : ℕ) (twentyTonBins : ℕ) (twentyTonCapacity : ℕ) (fifteenTonCapacity : ℕ) : ℕ :=
  twentyTonBins + (totalCapacity - twentyTonBins * twentyTonCapacity) / fifteenTonCapacity

/-- Theorem stating that under given conditions, the total number of bins is 30. -/
theorem warehouse_bins_count :
  totalBins 510 12 20 15 = 30 := by
  sorry

end NUMINAMATH_CALUDE_warehouse_bins_count_l3015_301540


namespace NUMINAMATH_CALUDE_standing_arrangements_eq_210_l3015_301573

/-- The number of ways to arrange n distinct objects in k positions --/
def arrangement (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose k objects from n distinct objects --/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways 3 people can stand on 6 steps with given conditions --/
def standing_arrangements : ℕ :=
  arrangement 6 3 + choose 3 1 * arrangement 6 2

theorem standing_arrangements_eq_210 : standing_arrangements = 210 := by sorry

end NUMINAMATH_CALUDE_standing_arrangements_eq_210_l3015_301573


namespace NUMINAMATH_CALUDE_subset_implies_t_equals_two_l3015_301554

theorem subset_implies_t_equals_two (t : ℝ) : 
  let A : Set ℝ := {1, t, 2*t}
  let B : Set ℝ := {1, t^2}
  B ⊆ A → t = 2 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_t_equals_two_l3015_301554


namespace NUMINAMATH_CALUDE_determinant_equality_l3015_301558

theorem determinant_equality (p q r s : ℝ) : 
  (p * s - q * r = 7) → ((p + 2 * r) * s - (q + 2 * s) * r = 7) := by
  sorry

end NUMINAMATH_CALUDE_determinant_equality_l3015_301558


namespace NUMINAMATH_CALUDE_fourth_vertex_coordinates_l3015_301504

/-- A regular tetrahedron with integer coordinates -/
structure RegularTetrahedron where
  v1 : ℤ × ℤ × ℤ
  v2 : ℤ × ℤ × ℤ
  v3 : ℤ × ℤ × ℤ
  v4 : ℤ × ℤ × ℤ
  is_regular : True  -- Placeholder for the regularity condition

/-- The fourth vertex of the regular tetrahedron -/
def fourth_vertex (t : RegularTetrahedron) : ℤ × ℤ × ℤ := t.v4

/-- The theorem stating the coordinates of the fourth vertex -/
theorem fourth_vertex_coordinates (t : RegularTetrahedron) 
  (h1 : t.v1 = (0, 1, 2))
  (h2 : t.v2 = (4, 2, 1))
  (h3 : t.v3 = (3, 1, 5)) :
  fourth_vertex t = (3, -2, 2) := by sorry

end NUMINAMATH_CALUDE_fourth_vertex_coordinates_l3015_301504


namespace NUMINAMATH_CALUDE_range_of_m_l3015_301552

-- Define the sets
def set1 (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - 4 ≤ 0 ∧ p.2 ≥ 0 ∧ m * p.1 - p.2 ≥ 0 ∧ m > 0}

def set2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 2)^2 ≤ 8}

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (set1 m ⊆ set2) → (0 < m ∧ m ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3015_301552


namespace NUMINAMATH_CALUDE_stevens_apples_l3015_301519

/-- The number of apples Steven has set aside to meet his seed collection goal. -/
def apples_set_aside : ℕ :=
  let total_seeds_needed : ℕ := 60
  let seeds_per_apple : ℕ := 6
  let seeds_per_pear : ℕ := 2
  let seeds_per_grape : ℕ := 3
  let pears : ℕ := 3
  let grapes : ℕ := 9
  let seeds_short : ℕ := 3

  let seeds_from_pears : ℕ := pears * seeds_per_pear
  let seeds_from_grapes : ℕ := grapes * seeds_per_grape
  let seeds_collected : ℕ := total_seeds_needed - seeds_short
  let seeds_from_apples : ℕ := seeds_collected - seeds_from_pears - seeds_from_grapes

  seeds_from_apples / seeds_per_apple

theorem stevens_apples :
  apples_set_aside = 4 :=
by sorry

end NUMINAMATH_CALUDE_stevens_apples_l3015_301519


namespace NUMINAMATH_CALUDE_min_value_iff_lower_bound_l3015_301583

/-- Given a function f: ℝ → ℝ and a constant M, prove that the following are equivalent:
    1) For all x ∈ ℝ, f(x) ≥ M
    2) M is the minimum value of f -/
theorem min_value_iff_lower_bound (f : ℝ → ℝ) (M : ℝ) :
  (∀ x, f x ≥ M) ↔ (∀ x, f x ≥ M ∧ ∃ y, f y = M) :=
by sorry

end NUMINAMATH_CALUDE_min_value_iff_lower_bound_l3015_301583


namespace NUMINAMATH_CALUDE_a_initial_is_9000_l3015_301529

/-- Represents the initial investment and profit distribution scenario -/
structure BusinessScenario where
  a_initial : ℕ  -- A's initial investment
  b_investment : ℕ  -- B's investment
  b_join_time : ℕ  -- Time when B joined (in months)
  total_time : ℕ  -- Total time of the year (in months)
  profit_ratio : Rat  -- Profit ratio (A:B)

/-- Calculates the initial investment of A given the business scenario -/
def calculate_a_initial (scenario : BusinessScenario) : ℕ :=
  (scenario.b_investment * scenario.b_join_time * 2) / scenario.total_time

/-- Theorem stating that A's initial investment is 9000 given the specific conditions -/
theorem a_initial_is_9000 : 
  let scenario : BusinessScenario := {
    a_initial := 9000,
    b_investment := 27000,
    b_join_time := 2,
    total_time := 12,
    profit_ratio := 2/1
  }
  calculate_a_initial scenario = 9000 := by
  sorry

#eval calculate_a_initial {
  a_initial := 9000,
  b_investment := 27000,
  b_join_time := 2,
  total_time := 12,
  profit_ratio := 2/1
}

end NUMINAMATH_CALUDE_a_initial_is_9000_l3015_301529


namespace NUMINAMATH_CALUDE_problem_statement_l3015_301584

theorem problem_statement (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 3) :
  2 * (Real.cos (π / 6 + α / 2))^2 + 1 = 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3015_301584


namespace NUMINAMATH_CALUDE_sum_of_multiples_of_4_between_34_and_135_l3015_301594

def sumOfMultiplesOf4 (lower upper : ℕ) : ℕ :=
  let firstMultiple := (lower + 3) / 4 * 4
  let lastMultiple := upper / 4 * 4
  let n := (lastMultiple - firstMultiple) / 4 + 1
  n * (firstMultiple + lastMultiple) / 2

theorem sum_of_multiples_of_4_between_34_and_135 :
  sumOfMultiplesOf4 34 135 = 2100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_of_4_between_34_and_135_l3015_301594


namespace NUMINAMATH_CALUDE_trip_time_change_l3015_301580

/-- Calculates the time required for a trip given the original time, original speed, and new speed -/
def new_trip_time (original_time : ℚ) (original_speed : ℚ) (new_speed : ℚ) : ℚ :=
  (original_time * original_speed) / new_speed

theorem trip_time_change (original_time : ℚ) (original_speed : ℚ) (new_speed : ℚ) 
  (h1 : original_time = 16/3)
  (h2 : original_speed = 80)
  (h3 : new_speed = 50) :
  ∃ (ε : ℚ), abs (new_trip_time original_time original_speed new_speed - 853/100) < ε ∧ ε < 1/100 :=
sorry

end NUMINAMATH_CALUDE_trip_time_change_l3015_301580


namespace NUMINAMATH_CALUDE_max_area_is_12_l3015_301522

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the conditions of the quadrilateral
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  let dist := λ p1 p2 : ℝ × ℝ => Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  dist q.A q.B = 5 ∧
  dist q.B q.C = 5 ∧
  dist q.C q.D = 5 ∧
  dist q.D q.A = 3

-- Define the deformation that maximizes ∠ABC
def max_angle_deformation (q : Quadrilateral) : Quadrilateral :=
  sorry

-- Define the area calculation function
def area (q : Quadrilateral) : ℝ :=
  sorry

-- Theorem statement
theorem max_area_is_12 (q : Quadrilateral) (h : is_valid_quadrilateral q) :
  area (max_angle_deformation q) = 12 :=
sorry

end NUMINAMATH_CALUDE_max_area_is_12_l3015_301522


namespace NUMINAMATH_CALUDE_simple_interest_rate_correct_l3015_301575

/-- The simple interest rate that makes a sum of money increase to 7/6 of itself in 6 years -/
def simple_interest_rate : ℚ :=
  100 / 36

/-- The time period in years -/
def time_period : ℕ := 6

/-- The ratio of final amount to initial amount -/
def final_to_initial_ratio : ℚ := 7 / 6

theorem simple_interest_rate_correct : 
  final_to_initial_ratio = 1 + (simple_interest_rate * time_period) / 100 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_correct_l3015_301575


namespace NUMINAMATH_CALUDE_unique_b_values_l3015_301548

theorem unique_b_values : ∃! (b₂ b₃ b₄ b₅ b₆ : ℕ),
  (11 : ℚ) / 15 = b₂ / 2 + b₃ / 6 + b₄ / 24 + b₅ / 120 + b₆ / 720 ∧
  b₂ < 2 ∧ b₃ < 3 ∧ b₄ < 4 ∧ b₅ < 5 ∧ b₆ < 6 ∧
  b₂ + b₃ + b₄ + b₅ + b₆ = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_b_values_l3015_301548


namespace NUMINAMATH_CALUDE_min_colors_correct_l3015_301565

/-- The number of distribution centers to be represented -/
def num_centers : ℕ := 12

/-- Calculates the number of unique representations possible with n colors -/
def num_representations (n : ℕ) : ℕ := n + n.choose 2

/-- Checks if a given number of colors is sufficient to represent all centers -/
def is_sufficient (n : ℕ) : Prop := num_representations n ≥ num_centers

/-- The minimum number of colors needed -/
def min_colors : ℕ := 5

/-- Theorem stating that min_colors is the minimum number of colors needed -/
theorem min_colors_correct :
  is_sufficient min_colors ∧ ∀ k < min_colors, ¬is_sufficient k :=
sorry

end NUMINAMATH_CALUDE_min_colors_correct_l3015_301565


namespace NUMINAMATH_CALUDE_ln_range_is_real_l3015_301536

-- Define the natural logarithm function
noncomputable def ln : ℝ → ℝ := Real.log

-- Statement: The range of the natural logarithm is all real numbers
theorem ln_range_is_real : Set.range ln = Set.univ := by sorry

end NUMINAMATH_CALUDE_ln_range_is_real_l3015_301536


namespace NUMINAMATH_CALUDE_one_valid_x_l3015_301524

def box_volume (x : ℤ) : ℤ := (x + 6) * (x - 6) * (x^2 + 36)

theorem one_valid_x : ∃! x : ℤ, 
  x > 0 ∧ 
  x - 6 > 0 ∧ 
  box_volume x < 800 :=
sorry

end NUMINAMATH_CALUDE_one_valid_x_l3015_301524


namespace NUMINAMATH_CALUDE_root_sum_equation_l3015_301543

theorem root_sum_equation (n m : ℝ) (hn : n ≠ 0) 
  (hroot : n^2 + m*n + 3*n = 0) : m + n = -3 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_equation_l3015_301543


namespace NUMINAMATH_CALUDE_circus_performance_legs_on_ground_l3015_301570

/-- Calculates the total number of legs/paws/hands on the ground in a circus performance --/
def circus_legs_on_ground (total_dogs : ℕ) (total_cats : ℕ) (total_horses : ℕ) (acrobats_one_hand : ℕ) (acrobats_two_hands : ℕ) : ℕ :=
  let dogs_on_back_legs := total_dogs / 2
  let dogs_on_all_fours := total_dogs - dogs_on_back_legs
  let cats_on_back_legs := total_cats / 3
  let cats_on_all_fours := total_cats - cats_on_back_legs
  let horses_on_hind_legs := 2
  let horses_on_all_fours := total_horses - horses_on_hind_legs
  
  let dog_paws := dogs_on_back_legs * 2 + dogs_on_all_fours * 4
  let cat_paws := cats_on_back_legs * 2 + cats_on_all_fours * 4
  let horse_hooves := horses_on_hind_legs * 2 + horses_on_all_fours * 4
  let acrobat_hands := acrobats_one_hand * 1 + acrobats_two_hands * 2
  
  dog_paws + cat_paws + horse_hooves + acrobat_hands

theorem circus_performance_legs_on_ground :
  circus_legs_on_ground 20 10 5 4 2 = 118 := by
  sorry

end NUMINAMATH_CALUDE_circus_performance_legs_on_ground_l3015_301570


namespace NUMINAMATH_CALUDE_inequality_proof_l3015_301528

theorem inequality_proof (a b : ℝ) : (a^2 - 1) * (b^2 - 1) ≤ 0 → a^2 + b^2 - 1 - a^2*b^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3015_301528


namespace NUMINAMATH_CALUDE_trapezium_shorter_side_length_l3015_301596

theorem trapezium_shorter_side_length 
  (longer_side : ℝ) 
  (height : ℝ) 
  (area : ℝ) 
  (h1 : longer_side = 30) 
  (h2 : height = 16) 
  (h3 : area = 336) : 
  ∃ (shorter_side : ℝ), 
    area = (1 / 2) * (shorter_side + longer_side) * height ∧ 
    shorter_side = 12 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_shorter_side_length_l3015_301596


namespace NUMINAMATH_CALUDE_work_completion_proof_l3015_301535

/-- The number of days A takes to complete the work alone -/
def a_days : ℝ := 15

/-- The number of days B takes to complete the work alone -/
def b_days : ℝ := 20

/-- The fraction of work left after A and B work together -/
def work_left : ℝ := 0.41666666666666663

/-- The number of days A and B worked together -/
def days_worked_together : ℝ := 5

theorem work_completion_proof :
  let work_rate_a := 1 / a_days
  let work_rate_b := 1 / b_days
  let combined_rate := work_rate_a + work_rate_b
  combined_rate * days_worked_together = 1 - work_left :=
by sorry

end NUMINAMATH_CALUDE_work_completion_proof_l3015_301535


namespace NUMINAMATH_CALUDE_geometric_progression_in_floor_sqrt2003_l3015_301560

/-- For any positive integers k and m greater than 1, there exists a subsequence
of {⌊n√2003⌋} (n ≥ 1) that forms a geometric progression with m terms and ratio k. -/
theorem geometric_progression_in_floor_sqrt2003 (k m : ℕ) (hk : k > 1) (hm : m > 1) :
  ∃ (n : ℕ), ∀ (i : ℕ), i < m →
    (⌊(k^i * n : ℝ) * Real.sqrt 2003⌋ : ℤ) = k^i * ⌊(n : ℝ) * Real.sqrt 2003⌋ :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_in_floor_sqrt2003_l3015_301560


namespace NUMINAMATH_CALUDE_shifted_parabola_passes_through_point_l3015_301578

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c + v }

/-- Evaluates a parabola at a given x-coordinate -/
def eval_parabola (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem shifted_parabola_passes_through_point :
  let original := Parabola.mk (-1) (-2) 3
  let shifted := shift_parabola original 1 (-2)
  eval_parabola shifted (-1) = 1 := by sorry

end NUMINAMATH_CALUDE_shifted_parabola_passes_through_point_l3015_301578


namespace NUMINAMATH_CALUDE_cone_volume_l3015_301598

/-- Given a cone with base radius 3 and lateral surface area 15π, its volume is 12π. -/
theorem cone_volume (r h : ℝ) : 
  r = 3 → 
  π * r * (r^2 + h^2).sqrt = 15 * π → 
  (1/3) * π * r^2 * h = 12 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l3015_301598


namespace NUMINAMATH_CALUDE_three_colors_sufficient_and_necessary_l3015_301527

/-- A function that returns the minimum number of colors needed to uniquely identify n keys on a single keychain. -/
def min_colors (n : ℕ) : ℕ :=
  if n ≤ 2 then n else 3

/-- Theorem stating that for n ≥ 3 keys on a single keychain, 3 colors are sufficient and necessary to uniquely identify each key. -/
theorem three_colors_sufficient_and_necessary (n : ℕ) (h : n ≥ 3) :
  min_colors n = 3 := by sorry

end NUMINAMATH_CALUDE_three_colors_sufficient_and_necessary_l3015_301527


namespace NUMINAMATH_CALUDE_long_furred_brown_dogs_l3015_301507

theorem long_furred_brown_dogs 
  (total : ℕ) 
  (long_furred : ℕ) 
  (brown : ℕ) 
  (neither : ℕ) 
  (h1 : total = 45)
  (h2 : long_furred = 26)
  (h3 : brown = 30)
  (h4 : neither = 8) :
  long_furred + brown - (total - neither) = 19 := by
sorry

end NUMINAMATH_CALUDE_long_furred_brown_dogs_l3015_301507


namespace NUMINAMATH_CALUDE_latus_rectum_of_parabola_l3015_301530

/-- Given a parabola with equation y² = 4x, its latus rectum has the equation x = -1 -/
theorem latus_rectum_of_parabola :
  ∀ (x y : ℝ), y^2 = 4*x → (∃ (x₀ : ℝ), x₀ = -1 ∧ ∀ (y₀ : ℝ), (y₀^2 = 4*x₀ → x₀ = -1)) :=
by sorry

end NUMINAMATH_CALUDE_latus_rectum_of_parabola_l3015_301530


namespace NUMINAMATH_CALUDE_reflection_creates_symmetry_l3015_301553

/-- Represents a letter in the word --/
inductive Letter
| G | E | O | M | T | R | I | Ya

/-- Represents a position in 2D space --/
structure Position :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a word as a list of letters with their positions --/
def Word := List (Letter × Position)

/-- The original word "ГЕОМЕТРИя" --/
def original_word : Word := sorry

/-- Reflects a position across a vertical axis --/
def reflect_vertical (p : Position) (axis : ℝ) : Position :=
  ⟨2 * axis - p.x, p.y⟩

/-- Reflects a word across a vertical axis --/
def reflect_word_vertical (w : Word) (axis : ℝ) : Word :=
  w.map (fun (l, p) => (l, reflect_vertical p axis))

/-- Checks if a word is symmetrical across a vertical axis --/
def is_symmetrical_vertical (w : Word) (axis : ℝ) : Prop :=
  w = reflect_word_vertical w axis

/-- Theorem: Reflecting the word "ГЕОМЕТРИя" across a vertical axis results in a symmetrical figure --/
theorem reflection_creates_symmetry (axis : ℝ) :
  is_symmetrical_vertical (reflect_word_vertical original_word axis) axis := by
  sorry

end NUMINAMATH_CALUDE_reflection_creates_symmetry_l3015_301553


namespace NUMINAMATH_CALUDE_dorchester_puppies_washed_l3015_301587

/-- Calculates the number of puppies washed given the total earnings, base pay, and rate per puppy -/
def puppies_washed (total_earnings base_pay rate_per_puppy : ℚ) : ℚ :=
  (total_earnings - base_pay) / rate_per_puppy

/-- Proves that Dorchester washed 16 puppies on Wednesday -/
theorem dorchester_puppies_washed :
  puppies_washed 76 40 (9/4) = 16 := by
  sorry

end NUMINAMATH_CALUDE_dorchester_puppies_washed_l3015_301587


namespace NUMINAMATH_CALUDE_curve_intersection_points_l3015_301545

-- Define the parametric equations of the curve
def x (t : ℝ) : ℝ := -2 + 5 * t
def y (t : ℝ) : ℝ := 1 - 2 * t

-- Theorem statement
theorem curve_intersection_points :
  (∃ t : ℝ, x t = 0 ∧ y t = 1/5) ∧
  (∃ t : ℝ, x t = 1/2 ∧ y t = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_curve_intersection_points_l3015_301545


namespace NUMINAMATH_CALUDE_intersection_and_subset_l3015_301568

def set_A : Set ℝ := {x | x^2 - x - 6 ≤ 0}
def set_B (a : ℝ) : Set ℝ := {x | 1 - a < x ∧ x ≤ 3*a + 1}

theorem intersection_and_subset :
  (∀ x : ℝ, x ∈ (set_A ∩ set_B 1) ↔ (0 < x ∧ x ≤ 3)) ∧
  (∀ a : ℝ, set_B a ⊆ set_A ↔ a ≤ 2/3) := by sorry

end NUMINAMATH_CALUDE_intersection_and_subset_l3015_301568


namespace NUMINAMATH_CALUDE_unique_solution_sum_l3015_301531

def star_operation (m n : ℕ) : ℕ := m^n + m*n

theorem unique_solution_sum (m n : ℕ) 
  (hm : m ≥ 2) 
  (hn : n ≥ 2) 
  (h_star : star_operation m n = 64) : 
  m + n = 6 := by sorry

end NUMINAMATH_CALUDE_unique_solution_sum_l3015_301531


namespace NUMINAMATH_CALUDE_extreme_value_and_min_max_l3015_301569

/-- Function f(x) = 2x³ + ax² + bx + 1 -/
def f (a b x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + 1

/-- Derivative of f(x) -/
def f' (a b x : ℝ) : ℝ := 6 * x^2 + 2 * a * x + b

theorem extreme_value_and_min_max (a b : ℝ) : 
  f a b 1 = -6 ∧ f' a b 1 = 0 →
  a = 3 ∧ b = -12 ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f 3 (-12) x ≤ 21) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f 3 (-12) x ≥ -6) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f 3 (-12) x = 21) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f 3 (-12) x = -6) :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_and_min_max_l3015_301569


namespace NUMINAMATH_CALUDE_fraction_problem_l3015_301595

theorem fraction_problem (x y : ℚ) : 
  y / (x - 1) = 1 / 3 → (y + 4) / x = 1 / 2 → y / x = 7 / 22 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3015_301595


namespace NUMINAMATH_CALUDE_solution_set_for_decreasing_function_l3015_301574

/-- A function f is decreasing on its domain -/
def IsDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- The set of x satisfying f(1/x) > f(1) for a decreasing function f -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | f (1/x) > f 1}

theorem solution_set_for_decreasing_function (f : ℝ → ℝ) (h : IsDecreasing f) :
    SolutionSet f = {x | x < 0 ∨ x > 1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_for_decreasing_function_l3015_301574


namespace NUMINAMATH_CALUDE_intersection_point_sum_l3015_301515

/-- The x-coordinate of the intersection point of two lines -/
def a : ℝ := 5.5

/-- The y-coordinate of the intersection point of two lines -/
def b : ℝ := 2.5

/-- The first line equation -/
def line1 (x y : ℝ) : Prop := y = -x + 8

/-- The second line equation -/
def line2 (x y : ℝ) : Prop := 173 * y = -289 * x + 2021

theorem intersection_point_sum :
  line1 a b ∧ line2 a b → a + b = 8 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_sum_l3015_301515


namespace NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l3015_301533

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 13) : 
  x^3 + y^3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l3015_301533


namespace NUMINAMATH_CALUDE_rectangular_paper_area_l3015_301513

/-- The area of a rectangular sheet of paper -/
def paper_area (width length : ℝ) : ℝ := width * length

theorem rectangular_paper_area :
  let width : ℝ := 25
  let length : ℝ := 20
  paper_area width length = 500 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_paper_area_l3015_301513


namespace NUMINAMATH_CALUDE_sum_of_consecutive_odd_primes_has_at_least_four_divisors_l3015_301555

/-- Two natural numbers are consecutive primes if they are both prime and there is no prime between them. -/
def ConsecutivePrimes (p q : ℕ) : Prop :=
  Prime p ∧ Prime q ∧ p < q ∧ ∀ k, p < k → k < q → ¬ Prime k

/-- The number of positive divisors of a natural number n. -/
def numPositiveDivisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range n.succ)).card

theorem sum_of_consecutive_odd_primes_has_at_least_four_divisors
  (p q : ℕ) (h : ConsecutivePrimes p q) :
  4 ≤ numPositiveDivisors (p + q) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_odd_primes_has_at_least_four_divisors_l3015_301555


namespace NUMINAMATH_CALUDE_max_difference_l3015_301589

theorem max_difference (a b : ℝ) : 
  a < 0 → 
  (∀ x, a < x ∧ x < b → (x^2 + 2017*a)*(x + 2016*b) ≥ 0) → 
  b - a ≤ 2017 :=
by sorry

end NUMINAMATH_CALUDE_max_difference_l3015_301589


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l3015_301571

theorem part_to_whole_ratio (N P : ℚ) 
  (h1 : (1/4) * (1/3) * P = 25)
  (h2 : (2/5) * N = 300) : 
  P / N = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l3015_301571


namespace NUMINAMATH_CALUDE_charley_pencils_loss_l3015_301521

theorem charley_pencils_loss (initial_pencils : ℕ) (lost_moving : ℕ) (current_pencils : ℕ)
  (h1 : initial_pencils = 30)
  (h2 : lost_moving = 6)
  (h3 : current_pencils = 16) :
  (initial_pencils - lost_moving - current_pencils : ℚ) / (initial_pencils - lost_moving : ℚ) = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_charley_pencils_loss_l3015_301521


namespace NUMINAMATH_CALUDE_max_students_distribution_l3015_301551

theorem max_students_distribution (pens pencils notebooks erasers : ℕ) 
  (h1 : pens = 891) (h2 : pencils = 810) (h3 : notebooks = 1080) (h4 : erasers = 972) : 
  Nat.gcd pens (Nat.gcd pencils (Nat.gcd notebooks erasers)) = 27 := by
  sorry

end NUMINAMATH_CALUDE_max_students_distribution_l3015_301551


namespace NUMINAMATH_CALUDE_bedroom_curtain_length_l3015_301542

theorem bedroom_curtain_length :
  let total_fabric_area : ℝ := 16 * 12
  let living_room_curtain_area : ℝ := 4 * 6
  let bedroom_curtain_width : ℝ := 2
  let remaining_fabric_area : ℝ := 160
  let bedroom_curtain_area : ℝ := total_fabric_area - living_room_curtain_area - remaining_fabric_area
  bedroom_curtain_area / bedroom_curtain_width = 4 := by
  sorry

end NUMINAMATH_CALUDE_bedroom_curtain_length_l3015_301542


namespace NUMINAMATH_CALUDE_crate_height_difference_l3015_301576

/-- The number of cans in each crate -/
def num_cans : ℕ := 300

/-- The diameter of each can in cm -/
def can_diameter : ℕ := 12

/-- The number of rows in triangular stacking -/
def triangular_rows : ℕ := 24

/-- The number of rows in square stacking -/
def square_rows : ℕ := 18

/-- The height of the triangular stacking in cm -/
def triangular_height : ℕ := triangular_rows * can_diameter

/-- The height of the square stacking in cm -/
def square_height : ℕ := square_rows * can_diameter

theorem crate_height_difference :
  triangular_height - square_height = 72 :=
sorry

end NUMINAMATH_CALUDE_crate_height_difference_l3015_301576


namespace NUMINAMATH_CALUDE_even_monotone_inequality_l3015_301566

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def monotone_increasing_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

-- State the theorem
theorem even_monotone_inequality (h1 : is_even f) (h2 : monotone_increasing_on_positive f) :
  f (-1) < f 2 ∧ f 2 < f (-3) :=
sorry

end NUMINAMATH_CALUDE_even_monotone_inequality_l3015_301566


namespace NUMINAMATH_CALUDE_checkerboard_covering_l3015_301500

/-- Represents a checkerboard -/
structure Checkerboard where
  size : ℕ
  removed_squares : Fin (4 * size * size) × Fin (4 * size * size)

/-- Represents a 2 × 1 domino -/
structure Domino

/-- Predicate to check if two squares are of opposite colors -/
def opposite_colors (c : Checkerboard) (s1 s2 : Fin (4 * c.size * c.size)) : Prop :=
  (s1.val + s2.val) % 2 = 1

/-- Predicate to check if a checkerboard can be covered by dominoes -/
def can_cover (c : Checkerboard) : Prop :=
  ∃ (covering : List (Fin (4 * c.size * c.size) × Fin (4 * c.size * c.size))),
    (∀ (square : Fin (4 * c.size * c.size)), 
      square ≠ c.removed_squares.1 ∧ square ≠ c.removed_squares.2 → 
      ∃ (domino : Fin (4 * c.size * c.size) × Fin (4 * c.size * c.size)), 
        domino ∈ covering ∧ (square = domino.1 ∨ square = domino.2)) ∧
    (∀ (domino : Fin (4 * c.size * c.size) × Fin (4 * c.size * c.size)), 
      domino ∈ covering → 
      (domino.1 ≠ c.removed_squares.1 ∧ domino.1 ≠ c.removed_squares.2) ∧
      (domino.2 ≠ c.removed_squares.1 ∧ domino.2 ≠ c.removed_squares.2) ∧
      (domino.1.val + 1 = domino.2.val ∨ domino.1.val + 2 * c.size = domino.2.val))

/-- Theorem stating that any 2k × 2k checkerboard with two squares of opposite colors removed can be covered by 2 × 1 dominoes -/
theorem checkerboard_covering (k : ℕ) (c : Checkerboard) 
  (h_size : c.size = 2 * k)
  (h_opposite : opposite_colors c c.removed_squares.1 c.removed_squares.2) :
  can_cover c :=
sorry

end NUMINAMATH_CALUDE_checkerboard_covering_l3015_301500


namespace NUMINAMATH_CALUDE_haley_tree_count_l3015_301544

/-- The number of trees Haley has after growing some, losing some to a typhoon, and growing more. -/
def final_tree_count (initial : ℕ) (lost : ℕ) (new : ℕ) : ℕ :=
  initial - lost + new

/-- Theorem stating that with 9 initial trees, 4 lost, and 5 new, the final count is 10. -/
theorem haley_tree_count : final_tree_count 9 4 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_haley_tree_count_l3015_301544


namespace NUMINAMATH_CALUDE_diego_paycheck_l3015_301501

/-- Diego's monthly paycheck problem -/
theorem diego_paycheck (monthly_expenses : ℝ) (annual_savings : ℝ) (h1 : monthly_expenses = 4600) (h2 : annual_savings = 4800) :
  monthly_expenses + annual_savings / 12 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_diego_paycheck_l3015_301501


namespace NUMINAMATH_CALUDE_chefs_and_waiters_arrangements_l3015_301510

/-- The number of ways to arrange chefs and waiters in a row --/
def arrangements (num_chefs num_waiters : ℕ) : ℕ :=
  if num_chefs + num_waiters ≠ 5 then 0
  else if num_chefs ≠ 2 then 0
  else if num_waiters ≠ 3 then 0
  else 36

/-- Theorem stating that the number of arrangements for 2 chefs and 3 waiters is 36 --/
theorem chefs_and_waiters_arrangements :
  arrangements 2 3 = 36 := by sorry

end NUMINAMATH_CALUDE_chefs_and_waiters_arrangements_l3015_301510


namespace NUMINAMATH_CALUDE_rug_inner_length_is_four_l3015_301562

/-- Represents the dimensions of a rectangular region -/
structure RectDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle given its dimensions -/
def rectangleArea (d : RectDimensions) : ℝ := d.length * d.width

/-- Represents the rug with three regions -/
structure Rug where
  innerLength : ℝ
  innerWidth : ℝ := 2
  middleWidth : ℝ := 6
  outerWidth : ℝ := 10

/-- Calculates the areas of the three regions of the rug -/
def rugAreas (r : Rug) : Fin 3 → ℝ
  | 0 => rectangleArea ⟨r.innerLength, r.innerWidth⟩
  | 1 => rectangleArea ⟨r.innerLength + 4, r.middleWidth⟩ - rectangleArea ⟨r.innerLength, r.innerWidth⟩
  | 2 => rectangleArea ⟨r.innerLength + 8, r.outerWidth⟩ - rectangleArea ⟨r.innerLength + 4, r.middleWidth⟩

/-- Checks if three numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℝ) : Prop := b - a = c - b

theorem rug_inner_length_is_four :
  ∀ (r : Rug), isArithmeticProgression (rugAreas r 0) (rugAreas r 1) (rugAreas r 2) →
  r.innerLength = 4 := by
  sorry

end NUMINAMATH_CALUDE_rug_inner_length_is_four_l3015_301562


namespace NUMINAMATH_CALUDE_genetic_material_distribution_l3015_301509

/-- Represents a diploid organism -/
structure DiploidOrganism :=
  (chromosomes : ℕ)
  (is_diploid : chromosomes % 2 = 0)

/-- Represents genetic material in the cytoplasm -/
structure GeneticMaterial :=
  (amount : ℝ)

/-- Represents a cell of a diploid organism -/
structure Cell :=
  (organism : DiploidOrganism)
  (cytoplasm : GeneticMaterial)

/-- Represents the distribution of genetic material during cell division -/
def genetic_distribution (parent : Cell) (daughter1 daughter2 : Cell) : Prop :=
  (daughter1.cytoplasm.amount + daughter2.cytoplasm.amount = parent.cytoplasm.amount) ∧
  (daughter1.cytoplasm.amount ≠ daughter2.cytoplasm.amount)

/-- Theorem stating that genetic material in the cytoplasm is distributed randomly and unequally during cell division -/
theorem genetic_material_distribution 
  (parent : Cell) 
  (daughter1 daughter2 : Cell) :
  genetic_distribution parent daughter1 daughter2 :=
sorry

end NUMINAMATH_CALUDE_genetic_material_distribution_l3015_301509


namespace NUMINAMATH_CALUDE_basketball_team_cutoff_l3015_301505

theorem basketball_team_cutoff (girls boys callback : ℕ) 
  (h_girls : girls = 17)
  (h_boys : boys = 32)
  (h_callback : callback = 10) :
  girls + boys - callback = 39 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_cutoff_l3015_301505


namespace NUMINAMATH_CALUDE_gcd_repeating_six_digit_l3015_301586

def is_repeating_six_digit (n : ℕ) : Prop :=
  ∃ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ n = 1001 * m

theorem gcd_repeating_six_digit :
  ∃ d : ℕ, d > 0 ∧ (∀ n : ℕ, is_repeating_six_digit n → d ∣ n) ∧
  (∀ d' : ℕ, d' > 0 → (∀ n : ℕ, is_repeating_six_digit n → d' ∣ n) → d' ≤ d) ∧
  d = 1001 :=
sorry

end NUMINAMATH_CALUDE_gcd_repeating_six_digit_l3015_301586


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3015_301502

theorem expression_simplification_and_evaluation :
  let x : ℚ := 4
  ((1 / (x + 2) + 1) / ((x^2 + 6*x + 9) / (x^2 - 4))) = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3015_301502


namespace NUMINAMATH_CALUDE_problem_statement_l3015_301506

theorem problem_statement (x y : ℝ) (a : ℝ) :
  (x - a*y) * (x + a*y) = x^2 - 16*y^2 → a = 4 ∨ a = -4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3015_301506


namespace NUMINAMATH_CALUDE_circles_intersect_l3015_301547

/-- Definition of circle C1 -/
def C1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x + 3*y + 2 = 0

/-- Definition of circle C2 -/
def C2 (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 3*y + 1 = 0

/-- Theorem stating that C1 and C2 are intersecting -/
theorem circles_intersect : ∃ (x y : ℝ), C1 x y ∧ C2 x y :=
sorry

end NUMINAMATH_CALUDE_circles_intersect_l3015_301547


namespace NUMINAMATH_CALUDE_round_table_gender_divisibility_l3015_301579

theorem round_table_gender_divisibility (n : ℕ) : 
  (∃ k : ℕ, k = n / 2 ∧ k = n - k) → 
  (∃ m : ℕ, n = 4 * m) :=
by sorry

end NUMINAMATH_CALUDE_round_table_gender_divisibility_l3015_301579


namespace NUMINAMATH_CALUDE_green_yarn_length_l3015_301532

theorem green_yarn_length :
  ∀ (green_length red_length : ℕ),
  red_length = 3 * green_length + 8 →
  green_length + red_length = 632 →
  green_length = 156 :=
by
  sorry

end NUMINAMATH_CALUDE_green_yarn_length_l3015_301532


namespace NUMINAMATH_CALUDE_max_apartments_l3015_301526

/-- Represents an apartment building with specific properties. -/
structure ApartmentBuilding where
  entrances : Nat
  floors : Nat
  apartments_per_floor : Nat
  two_digit_apartments_in_entrance : Nat

/-- The conditions of the apartment building as described in the problem. -/
def building_conditions (b : ApartmentBuilding) : Prop :=
  b.apartments_per_floor = 4 ∧
  b.two_digit_apartments_in_entrance = 10 * b.entrances ∧
  b.two_digit_apartments_in_entrance ≤ 90

/-- The total number of apartments in the building. -/
def total_apartments (b : ApartmentBuilding) : Nat :=
  b.entrances * b.floors * b.apartments_per_floor

/-- Theorem stating the maximum number of apartments in the building. -/
theorem max_apartments (b : ApartmentBuilding) (h : building_conditions b) :
  total_apartments b ≤ 936 := by
  sorry

#check max_apartments

end NUMINAMATH_CALUDE_max_apartments_l3015_301526


namespace NUMINAMATH_CALUDE_deposit_exceeds_target_on_saturday_l3015_301539

def initial_deposit : ℕ := 2
def multiplication_factor : ℕ := 3
def target_amount : ℕ := 500 * 100  -- Convert $500 to cents

def deposit_on_day (n : ℕ) : ℕ :=
  initial_deposit * multiplication_factor ^ n

def total_deposit (n : ℕ) : ℕ :=
  (List.range (n + 1)).map deposit_on_day |>.sum

def days_of_week := ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

theorem deposit_exceeds_target_on_saturday :
  (total_deposit 5 ≤ target_amount) ∧ 
  (total_deposit 6 > target_amount) ∧
  (days_of_week[(6 : ℕ) % 7] = "Saturday") := by
  sorry

end NUMINAMATH_CALUDE_deposit_exceeds_target_on_saturday_l3015_301539
