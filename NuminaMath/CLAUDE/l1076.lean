import Mathlib

namespace NUMINAMATH_CALUDE_solution_set_of_equation_l1076_107622

theorem solution_set_of_equation (x : ℝ) : 
  (16 * Real.sin (π * x) * Real.cos (π * x) = 16 * x + 1 / x) ↔ (x = 1/4 ∨ x = -1/4) :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_equation_l1076_107622


namespace NUMINAMATH_CALUDE_least_positive_y_l1076_107693

theorem least_positive_y (x y : ℤ) : 
  (∃ (k : ℤ), 0 < 24 * x + k * y ∧ ∀ (m : ℤ), 0 < 24 * x + m * y → 24 * x + k * y ≤ 24 * x + m * y) ∧
  (∀ (n : ℤ), 0 < 24 * x + n * y → 4 ≤ 24 * x + n * y) →
  y = 4 ∨ y = -4 :=
sorry

end NUMINAMATH_CALUDE_least_positive_y_l1076_107693


namespace NUMINAMATH_CALUDE_weekend_pie_revenue_l1076_107685

structure PieSlice where
  name : String
  slices_per_pie : ℕ
  price_per_slice : ℕ
  customers : ℕ

def apple_pie : PieSlice := {
  name := "Apple",
  slices_per_pie := 8,
  price_per_slice := 3,
  customers := 88
}

def peach_pie : PieSlice := {
  name := "Peach",
  slices_per_pie := 6,
  price_per_slice := 4,
  customers := 78
}

def cherry_pie : PieSlice := {
  name := "Cherry",
  slices_per_pie := 10,
  price_per_slice := 5,
  customers := 45
}

def revenue (pie : PieSlice) : ℕ :=
  pie.customers * pie.price_per_slice

def total_revenue (pies : List PieSlice) : ℕ :=
  pies.foldl (fun acc pie => acc + revenue pie) 0

theorem weekend_pie_revenue :
  total_revenue [apple_pie, peach_pie, cherry_pie] = 801 := by
  sorry

end NUMINAMATH_CALUDE_weekend_pie_revenue_l1076_107685


namespace NUMINAMATH_CALUDE_optical_mice_ratio_l1076_107624

theorem optical_mice_ratio (total_mice : ℕ) (trackball_mice : ℕ) : 
  total_mice = 80 →
  trackball_mice = 20 →
  (total_mice / 2 : ℚ) = total_mice / 2 →
  (total_mice - total_mice / 2 - trackball_mice : ℚ) / total_mice = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_optical_mice_ratio_l1076_107624


namespace NUMINAMATH_CALUDE_sum_of_even_coefficients_l1076_107611

theorem sum_of_even_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (3*x - 1)^7 = a₀*x^7 + a₁*x^6 + a₂*x^5 + a₃*x^4 + a₄*x^3 + a₅*x^2 + a₆*x + a₇) →
  a₀ + a₂ + a₄ + a₆ = 4128 := by
sorry

end NUMINAMATH_CALUDE_sum_of_even_coefficients_l1076_107611


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l1076_107619

-- Define the propositions p and q
def p (t : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (t + 2) + y^2 / (t - 10) = 1

def q (t m : ℝ) : Prop := -m < t ∧ t < m + 1 ∧ m > 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (p q : Prop) : Prop :=
  (q → p) ∧ ¬(p → q)

-- Theorem statement
theorem hyperbola_m_range :
  (∀ t, sufficient_not_necessary (p t) (∃ m, q t m)) →
  ∀ m, (m > 0 ∧ m ≤ 2) ↔ (∃ t, q t m ∧ p t) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l1076_107619


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l1076_107695

theorem smallest_n_square_and_cube : ∃ (n : ℕ), 
  n > 0 ∧ 
  (∃ (k : ℕ), 4 * n = k^2) ∧ 
  (∃ (m : ℕ), 5 * n = m^3) ∧ 
  (∀ (x : ℕ), x > 0 → (∃ (y : ℕ), 4 * x = y^2) → (∃ (z : ℕ), 5 * x = z^3) → x ≥ n) ∧
  n = 400 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l1076_107695


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1076_107690

theorem tan_alpha_plus_pi_fourth (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - π/4) = 1/4) :
  Real.tan (α + π/4) = 3/22 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1076_107690


namespace NUMINAMATH_CALUDE_total_books_l1076_107610

theorem total_books (keith_books jason_books : ℕ) 
  (h1 : keith_books = 20) 
  (h2 : jason_books = 21) : 
  keith_books + jason_books = 41 := by
sorry

end NUMINAMATH_CALUDE_total_books_l1076_107610


namespace NUMINAMATH_CALUDE_divisibility_by_eighteen_l1076_107630

theorem divisibility_by_eighteen (n : ℕ) : 
  n ≤ 9 → 
  913 * 10 + n ≥ 1000 → 
  913 * 10 + n < 10000 → 
  (913 * 10 + n) % 18 = 0 ↔ n = 8 := by sorry

end NUMINAMATH_CALUDE_divisibility_by_eighteen_l1076_107630


namespace NUMINAMATH_CALUDE_alice_stool_height_l1076_107667

/-- The height of the ceiling above the floor in centimeters -/
def ceiling_height : ℝ := 300

/-- The distance of the light bulb below the ceiling in centimeters -/
def light_bulb_below_ceiling : ℝ := 15

/-- Alice's height in centimeters -/
def alice_height : ℝ := 150

/-- The distance Alice can reach above her head in centimeters -/
def alice_reach : ℝ := 40

/-- The minimum height of the stool Alice needs in centimeters -/
def stool_height : ℝ := 95

theorem alice_stool_height : 
  ceiling_height - light_bulb_below_ceiling = alice_height + alice_reach + stool_height := by
  sorry

end NUMINAMATH_CALUDE_alice_stool_height_l1076_107667


namespace NUMINAMATH_CALUDE_complex_modulus_l1076_107609

theorem complex_modulus (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_l1076_107609


namespace NUMINAMATH_CALUDE_line_division_theorem_l1076_107666

/-- A line in the plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if three lines divide the plane into six parts -/
def divides_into_six_parts (l₁ l₂ l₃ : Line) : Prop :=
  sorry

/-- The set of k values that satisfy the condition -/
def k_values : Set ℝ :=
  {0, -1, -2}

theorem line_division_theorem (k : ℝ) :
  let l₁ : Line := ⟨1, -2, 1⟩  -- x - 2y + 1 = 0
  let l₂ : Line := ⟨1, 0, -1⟩ -- x - 1 = 0
  let l₃ : Line := ⟨1, k, 0⟩  -- x + ky = 0
  divides_into_six_parts l₁ l₂ l₃ → k ∈ k_values := by
  sorry

end NUMINAMATH_CALUDE_line_division_theorem_l1076_107666


namespace NUMINAMATH_CALUDE_soybean_oil_conversion_l1076_107674

/-- Represents the problem of determining the amount of soybeans converted to soybean oil --/
theorem soybean_oil_conversion (total_soybeans : ℝ) (total_revenue : ℝ) 
  (tofu_conversion : ℝ) (oil_conversion : ℝ) (tofu_price : ℝ) (oil_price : ℝ) :
  total_soybeans = 460 ∧ 
  total_revenue = 1800 ∧
  tofu_conversion = 3 ∧
  oil_conversion = 1 / 6 ∧
  tofu_price = 3 ∧
  oil_price = 15 →
  ∃ (x : ℝ), 
    x = 360 ∧ 
    tofu_price * tofu_conversion * (total_soybeans - x) + oil_price * oil_conversion * x = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_soybean_oil_conversion_l1076_107674


namespace NUMINAMATH_CALUDE_higher_profit_percentage_l1076_107660

/-- The profit percentage that results in $72 more profit than 9% on a cost price of $800 is 18% -/
theorem higher_profit_percentage (cost_price : ℝ) (additional_profit : ℝ) :
  cost_price = 800 →
  additional_profit = 72 →
  ∃ (P : ℝ), P * cost_price / 100 = (9 * cost_price / 100) + additional_profit ∧ P = 18 :=
by sorry

end NUMINAMATH_CALUDE_higher_profit_percentage_l1076_107660


namespace NUMINAMATH_CALUDE_edward_final_earnings_l1076_107629

/-- Edward's lawn mowing business earnings and expenses --/
def edward_business (spring_earnings summer_earnings supplies_cost : ℕ) : ℕ :=
  spring_earnings + summer_earnings - supplies_cost

/-- Theorem: Edward's final earnings --/
theorem edward_final_earnings :
  edward_business 2 27 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_edward_final_earnings_l1076_107629


namespace NUMINAMATH_CALUDE_complex_locus_is_ellipse_l1076_107692

theorem complex_locus_is_ellipse (z : ℂ) (h : Complex.abs z = 3) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (w : ℂ), w = 2 * z + 1 / z →
  (w.re / a) ^ 2 + (w.im / b) ^ 2 = 1 :=
sorry

end NUMINAMATH_CALUDE_complex_locus_is_ellipse_l1076_107692


namespace NUMINAMATH_CALUDE_units_digit_of_3_to_1987_l1076_107668

theorem units_digit_of_3_to_1987 : 3^1987 % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_3_to_1987_l1076_107668


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l1076_107651

theorem quadratic_root_implies_k (k : ℝ) : 
  (∃ (x : ℂ), x = (-5 + Complex.I * Real.sqrt 171) / 14 ∧ 
   7 * x^2 + 5 * x + k = 0) → k = 7 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l1076_107651


namespace NUMINAMATH_CALUDE_path_equivalence_arrow_sequence_equivalence_l1076_107621

/-- Represents the cyclic pattern of points in the path -/
def cycle_length : ℕ := 5

/-- Maps a point to its equivalent position in the cycle -/
def cycle_position (n : ℕ) : ℕ := n % cycle_length

/-- Theorem: The path from point 520 to 523 is equivalent to 0 to 3 in the cycle -/
theorem path_equivalence : 
  (cycle_position 520 = cycle_position 0) ∧ 
  (cycle_position 523 = cycle_position 3) := by
  sorry

/-- The sequence of arrows from 520 to 523 is the same as 0 to 3 -/
theorem arrow_sequence_equivalence : 
  ∀ (i : ℕ), i < 3 → 
  cycle_position (520 + i) = cycle_position i := by
  sorry

end NUMINAMATH_CALUDE_path_equivalence_arrow_sequence_equivalence_l1076_107621


namespace NUMINAMATH_CALUDE_farmer_animals_l1076_107681

theorem farmer_animals (goats cows pigs : ℕ) : 
  pigs = 2 * cows ∧ 
  cows = goats + 4 ∧ 
  goats + cows + pigs = 56 →
  goats = 11 := by
sorry

end NUMINAMATH_CALUDE_farmer_animals_l1076_107681


namespace NUMINAMATH_CALUDE_angle_of_inclination_sqrt3_l1076_107617

/-- The angle of inclination of a line with slope √3 is 60°. -/
theorem angle_of_inclination_sqrt3 :
  let slope : ℝ := Real.sqrt 3
  let angle : ℝ := 60 * π / 180  -- Convert 60° to radians
  Real.tan angle = slope := by sorry

end NUMINAMATH_CALUDE_angle_of_inclination_sqrt3_l1076_107617


namespace NUMINAMATH_CALUDE_larger_integer_is_fifteen_l1076_107656

theorem larger_integer_is_fifteen (a b : ℤ) : 
  (a : ℚ) / b = 1 / 3 → 
  (a + 10 : ℚ) / b = 1 → 
  b = 15 := by
sorry

end NUMINAMATH_CALUDE_larger_integer_is_fifteen_l1076_107656


namespace NUMINAMATH_CALUDE_largest_undefined_x_l1076_107657

theorem largest_undefined_x (f : ℝ → ℝ) :
  (∀ x, f x = (x + 2) / (10 * x^2 - 85 * x + 10)) →
  (∃ x, 10 * x^2 - 85 * x + 10 = 0) →
  (∀ x, 10 * x^2 - 85 * x + 10 = 0 → x ≤ 10) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_undefined_x_l1076_107657


namespace NUMINAMATH_CALUDE_parallel_iff_a_eq_neg_one_l1076_107637

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := x - y - 1 = 0
def l₂ (a x y : ℝ) : Prop := x + a*y - 2 = 0

-- Define parallelism for these lines
def parallel (a : ℝ) : Prop := ∀ x y, l₁ x y ↔ ∃ k, l₂ a (x + k) (y + k)

-- State the theorem
theorem parallel_iff_a_eq_neg_one :
  ∀ a : ℝ, parallel a ↔ a = -1 := by sorry

end NUMINAMATH_CALUDE_parallel_iff_a_eq_neg_one_l1076_107637


namespace NUMINAMATH_CALUDE_porche_homework_time_l1076_107604

/-- Proves that given a total time of 3 hours (180 minutes) and homework assignments
    taking 45, 30, 50, and 25 minutes respectively, the remaining time for a special project
    is 30 minutes. -/
theorem porche_homework_time (total_time : ℕ) (math_time english_time science_time history_time : ℕ) :
  total_time = 180 ∧
  math_time = 45 ∧
  english_time = 30 ∧
  science_time = 50 ∧
  history_time = 25 →
  total_time - (math_time + english_time + science_time + history_time) = 30 :=
by sorry

end NUMINAMATH_CALUDE_porche_homework_time_l1076_107604


namespace NUMINAMATH_CALUDE_max_packing_ge_min_covering_l1076_107683

/-- Represents a polygon in 2D space -/
structure Polygon

/-- The largest number of non-overlapping circles with diameter 1 whose centers lie inside the polygon -/
def max_packing (M : Polygon) : ℕ :=
  sorry

/-- The smallest number of circles with radius 1 needed to cover the entire polygon -/
def min_covering (M : Polygon) : ℕ :=
  sorry

/-- Theorem stating that the maximum packing is greater than or equal to the minimum covering -/
theorem max_packing_ge_min_covering (M : Polygon) : max_packing M ≥ min_covering M :=
  sorry

end NUMINAMATH_CALUDE_max_packing_ge_min_covering_l1076_107683


namespace NUMINAMATH_CALUDE_part_one_part_two_l1076_107650

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | 2*m - 1 ≤ x ∧ x ≤ m + 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 2}

-- Theorem for part 1
theorem part_one :
  (A (-1) ∪ B = {x | -3 ≤ x ∧ x ≤ 2}) ∧
  (A (-1) ∩ (Set.univ \ B) = {x | -3 ≤ x ∧ x ≤ -1}) := by
  sorry

-- Theorem for part 2
theorem part_two (m : ℝ) :
  A m ∩ B = ∅ ↔ m ≤ -3 ∨ m > 3/2 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1076_107650


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1076_107638

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, x^2 - a*x - b < 0 ↔ 2 < x ∧ x < 3) →
  a + b = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1076_107638


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1076_107648

-- Define a geometric sequence with common ratio 2
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = 2 * a n

-- Define the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : a 1 + a 2 = 3) : 
  a 3 + a 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1076_107648


namespace NUMINAMATH_CALUDE_prime_sum_difference_l1076_107677

theorem prime_sum_difference (p q : Nat) : 
  Nat.Prime p → Nat.Prime q → p > 0 → q > 0 →
  p + p^2 + p^4 - q - q^2 - q^4 = 83805 →
  p = 17 ∧ q = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_difference_l1076_107677


namespace NUMINAMATH_CALUDE_polyhedron_edge_length_bound_l1076_107614

/-- A polyhedron is represented as a set of points in ℝ³ -/
def Polyhedron : Type := Set (Fin 3 → ℝ)

/-- The edge length of a polyhedron -/
def edgeLength (P : Polyhedron) : ℝ := sorry

/-- The sum of all edge lengths of a polyhedron -/
def sumEdgeLengths (P : Polyhedron) : ℝ := sorry

/-- The distance between two points in ℝ³ -/
def distance (a b : Fin 3 → ℝ) : ℝ := sorry

/-- The maximum distance between any two points in a polyhedron -/
def maxDistance (P : Polyhedron) : ℝ := sorry

/-- Theorem: The sum of edge lengths is at least 3 times the maximum distance -/
theorem polyhedron_edge_length_bound (P : Polyhedron) :
  sumEdgeLengths P ≥ 3 * maxDistance P := by sorry

end NUMINAMATH_CALUDE_polyhedron_edge_length_bound_l1076_107614


namespace NUMINAMATH_CALUDE_intersection_complement_when_m_3_sufficient_necessary_condition_l1076_107664

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 5}

def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2 * m + 1}

theorem intersection_complement_when_m_3 :
  A ∩ (Set.univ \ B 3) = {x | 0 ≤ x ∧ x < 2} := by sorry

theorem sufficient_necessary_condition (m : ℝ) :
  (∀ x, x ∈ B m ↔ x ∈ A) ↔ 1 ≤ m ∧ m ≤ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_when_m_3_sufficient_necessary_condition_l1076_107664


namespace NUMINAMATH_CALUDE_pebbles_on_day_15_l1076_107649

/-- Represents Murtha's pebble collection strategy -/
def pebbleCollection (n : ℕ) : ℕ := 
  if n < 15 then n else 2 * 15

/-- The sum of pebbles collected up to day n -/
def totalPebbles (n : ℕ) : ℕ := 
  (List.range n).map pebbleCollection |>.sum

/-- Theorem stating the total number of pebbles collected by the end of the 15th day -/
theorem pebbles_on_day_15 : totalPebbles 15 = 135 := by
  sorry

end NUMINAMATH_CALUDE_pebbles_on_day_15_l1076_107649


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1076_107626

/-- Represents a quadratic function of the form y = x^2 + bx - c -/
structure QuadraticFunction where
  b : ℝ
  c : ℝ

/-- Represents the roots of a quadratic function -/
structure Roots where
  m : ℝ
  h : m ≠ 0

theorem quadratic_function_properties (f : QuadraticFunction) (r : Roots) :
  (∀ x, f.b * x + x^2 - f.c = 0 ↔ x = r.m ∨ x = -2 * r.m) →
  f.c = 2 * f.b^2 ∧
  (f.b / 2 = -1 → f.b = 2 ∧ f.c = 8) := by
  sorry

#check quadratic_function_properties

end NUMINAMATH_CALUDE_quadratic_function_properties_l1076_107626


namespace NUMINAMATH_CALUDE_f_neg_two_eq_eleven_l1076_107603

/-- The function f(x) = x^2 - 3x + 1 -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 1

/-- Theorem: f(-2) = 11 -/
theorem f_neg_two_eq_eleven : f (-2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_eleven_l1076_107603


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_cubed_l1076_107687

theorem imaginary_part_of_one_minus_i_cubed (i : ℂ) : 
  i^2 = -1 → Complex.im ((1 - i)^3) = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_cubed_l1076_107687


namespace NUMINAMATH_CALUDE_black_area_after_three_cycles_l1076_107659

/-- Represents the fraction of black area remaining after a number of cycles. -/
def blackAreaFraction (cycles : ℕ) : ℚ :=
  (2 / 3) ^ cycles

/-- The number of cycles in the problem. -/
def numCycles : ℕ := 3

/-- Theorem stating that after three cycles, 8/27 of the original area remains black. -/
theorem black_area_after_three_cycles :
  blackAreaFraction numCycles = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_black_area_after_three_cycles_l1076_107659


namespace NUMINAMATH_CALUDE_plywood_cutting_result_l1076_107636

/-- Represents the cutting of a square plywood into smaller squares. -/
structure PlywoodCutting where
  side : ℝ
  small_square_side : ℝ
  large_square_side : ℝ
  total_cut_length : ℝ

/-- Calculates the total number of squares obtained from cutting the plywood. -/
def total_squares (cut : PlywoodCutting) : ℕ :=
  sorry

/-- Theorem stating that for the given plywood cutting specifications, 
    the total number of squares obtained is 16. -/
theorem plywood_cutting_result : 
  let cut := PlywoodCutting.mk 50 10 20 280
  total_squares cut = 16 := by
  sorry

end NUMINAMATH_CALUDE_plywood_cutting_result_l1076_107636


namespace NUMINAMATH_CALUDE_frame_interior_edges_sum_l1076_107691

/-- Represents a rectangular picture frame -/
structure Frame where
  outerLength : ℝ
  outerWidth : ℝ
  frameWidth : ℝ

/-- Calculates the area of the frame -/
def frameArea (f : Frame) : ℝ :=
  f.outerLength * f.outerWidth - (f.outerLength - 2 * f.frameWidth) * (f.outerWidth - 2 * f.frameWidth)

/-- Calculates the sum of the lengths of all four interior edges of the frame -/
def interiorEdgesSum (f : Frame) : ℝ :=
  2 * (f.outerLength - 2 * f.frameWidth) + 2 * (f.outerWidth - 2 * f.frameWidth)

/-- Theorem stating that for a frame with given dimensions, the sum of interior edges is 7 -/
theorem frame_interior_edges_sum :
  ∃ (f : Frame),
    f.outerLength = 7 ∧
    f.frameWidth = 2 ∧
    frameArea f = 30 ∧
    interiorEdgesSum f = 7 := by
  sorry

end NUMINAMATH_CALUDE_frame_interior_edges_sum_l1076_107691


namespace NUMINAMATH_CALUDE_fifth_root_over_sixth_root_of_eleven_l1076_107635

theorem fifth_root_over_sixth_root_of_eleven (x : ℝ) :
  (11 ^ (1/5)) / (11 ^ (1/6)) = 11 ^ (1/30) :=
sorry

end NUMINAMATH_CALUDE_fifth_root_over_sixth_root_of_eleven_l1076_107635


namespace NUMINAMATH_CALUDE_colby_remaining_mangoes_l1076_107605

def total_harvest : ℕ := 60
def sold_to_market : ℕ := 20
def mangoes_per_kg : ℕ := 8

def remaining_after_market : ℕ := total_harvest - sold_to_market

def sold_to_community : ℕ := remaining_after_market / 2

def remaining_kg : ℕ := remaining_after_market - sold_to_community

theorem colby_remaining_mangoes :
  remaining_kg * mangoes_per_kg = 160 := by sorry

end NUMINAMATH_CALUDE_colby_remaining_mangoes_l1076_107605


namespace NUMINAMATH_CALUDE_words_with_vowels_count_l1076_107606

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'I'}
def vowels : Finset Char := {'A', 'E', 'I'}
def consonants : Finset Char := alphabet \ vowels

def word_length : Nat := 5

def total_words : Nat := alphabet.card ^ word_length
def words_without_vowels : Nat := consonants.card ^ word_length

theorem words_with_vowels_count :
  total_words - words_without_vowels = 29643 :=
sorry

end NUMINAMATH_CALUDE_words_with_vowels_count_l1076_107606


namespace NUMINAMATH_CALUDE_no_prime_polynomial_l1076_107662

-- Define a polynomial with integer coefficients
def IntPolynomial := ℕ → ℤ

-- Define what it means for a polynomial to be constant
def IsConstant (P : IntPolynomial) : Prop :=
  ∀ n m : ℕ, P n = P m

-- Define primality
def IsPrime (n : ℤ) : Prop :=
  n > 1 ∧ ∀ m : ℤ, 1 < m → m < n → ¬(n % m = 0)

-- The main theorem
theorem no_prime_polynomial :
  ¬∃ (P : IntPolynomial),
    (¬IsConstant P) ∧
    (∀ n : ℕ, n > 0 → IsPrime (P n)) :=
sorry

end NUMINAMATH_CALUDE_no_prime_polynomial_l1076_107662


namespace NUMINAMATH_CALUDE_result_not_divisible_by_1998_l1076_107676

/-- The operation of multiplying by 2 and adding 1 -/
def operation (n : ℕ) : ℕ := 2 * n + 1

/-- The result of applying the operation k times to n -/
def iterate_operation (n k : ℕ) : ℕ :=
  match k with
  | 0 => n
  | k + 1 => operation (iterate_operation n k)

theorem result_not_divisible_by_1998 (n k : ℕ) :
  ¬(1998 ∣ iterate_operation n k) := by
  sorry

#check result_not_divisible_by_1998

end NUMINAMATH_CALUDE_result_not_divisible_by_1998_l1076_107676


namespace NUMINAMATH_CALUDE_joshua_second_box_toys_l1076_107670

/-- The number of toy cars in Joshua's second box -/
def toysCarsInSecondBox (totalToys : ℕ) (firstBoxToys : ℕ) (thirdBoxToys : ℕ) : ℕ :=
  totalToys - firstBoxToys - thirdBoxToys

theorem joshua_second_box_toys :
  toysCarsInSecondBox 71 21 19 = 31 := by
  sorry

end NUMINAMATH_CALUDE_joshua_second_box_toys_l1076_107670


namespace NUMINAMATH_CALUDE_fraction_sum_equals_61_30_l1076_107615

theorem fraction_sum_equals_61_30 :
  (3 + 6 + 9) / (2 + 5 + 8) + (2 + 5 + 8) / (3 + 6 + 9) = 61 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_61_30_l1076_107615


namespace NUMINAMATH_CALUDE_rectangle_area_change_l1076_107628

/-- Given a rectangle with area 540 square centimeters, if its length is increased by 15% and
    its width is decreased by 15%, the new area will be 527.55 square centimeters. -/
theorem rectangle_area_change (l w : ℝ) (h : l * w = 540) :
  (1.15 * l) * (0.85 * w) = 527.55 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l1076_107628


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1076_107616

-- Problem 1
theorem problem_1 : Real.sqrt 27 / Real.sqrt 3 + Real.sqrt 12 * Real.sqrt (1/3) - Real.sqrt 5 = 5 - Real.sqrt 5 := by
  sorry

-- Problem 2
theorem problem_2 : (Real.sqrt 5 + 2) * (Real.sqrt 5 - 2) + (2 * Real.sqrt 3 + 1)^2 = 14 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1076_107616


namespace NUMINAMATH_CALUDE_expression_simplification_l1076_107620

theorem expression_simplification (x y : ℝ) 
  (hx : x = Real.sqrt 2) 
  (hy : y = 2 * Real.sqrt 2) : 
  (4 * y^2 - x^2) / (x^2 + 2*x*y + y^2) / ((x - 2*y) / (2*x^2 + 2*x*y)) = -10 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1076_107620


namespace NUMINAMATH_CALUDE_final_block_count_l1076_107639

theorem final_block_count :
  let initial_blocks : ℕ := 250
  let added_blocks : ℕ := 13
  let intermediate_blocks : ℕ := initial_blocks + added_blocks
  let doubling_factor : ℕ := 2
  let final_blocks : ℕ := intermediate_blocks * doubling_factor
  final_blocks = 526 := by sorry

end NUMINAMATH_CALUDE_final_block_count_l1076_107639


namespace NUMINAMATH_CALUDE_number_of_girls_l1076_107632

theorem number_of_girls (total_children : Nat) (happy_children : Nat) (sad_children : Nat) 
  (neutral_children : Nat) (boys : Nat) (happy_boys : Nat) (sad_girls : Nat) (neutral_boys : Nat) :
  total_children = 60 →
  happy_children = 30 →
  sad_children = 10 →
  neutral_children = 20 →
  boys = 16 →
  happy_boys = 6 →
  sad_girls = 4 →
  neutral_boys = 4 →
  total_children = happy_children + sad_children + neutral_children →
  total_children - boys = 44 := by
  sorry

#check number_of_girls

end NUMINAMATH_CALUDE_number_of_girls_l1076_107632


namespace NUMINAMATH_CALUDE_range_of_m_l1076_107678

-- Define proposition p
def p (m : ℝ) : Prop :=
  ∃ a b : ℝ, a > b ∧ a^2 = m/2 ∧ b^2 = m/2 - 1

-- Define proposition q
def q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 - 4*m*x + 4*m - 3 ≥ 0

-- Theorem statement
theorem range_of_m :
  ∃ m_min m_max : ℝ,
    m_min = 1 ∧ m_max = 2 ∧
    ∀ m : ℝ, (¬(p m) ∧ q m) ↔ m_min ≤ m ∧ m ≤ m_max :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1076_107678


namespace NUMINAMATH_CALUDE_number_equation_l1076_107642

theorem number_equation (x : ℝ) (n : ℝ) : x = 32 → (35 - (n - (15 - x)) = 12 * 2 / (1 / 2)) ↔ n = -30 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l1076_107642


namespace NUMINAMATH_CALUDE_m_range_characterization_l1076_107688

/-- Proposition p: For all x ∈ ℝ, x² + 2x > m -/
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*x > m

/-- Proposition q: There exists x₀ ∈ ℝ, such that x₀² + 2mx₀ + 2 - m ≤ 0 -/
def q (m : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 2*m*x₀ + 2 - m ≤ 0

/-- The range of values for m -/
def m_range (m : ℝ) : Prop := (m > -2 ∧ m < -1) ∨ m ≥ 1

theorem m_range_characterization (m : ℝ) : 
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m_range m :=
sorry

end NUMINAMATH_CALUDE_m_range_characterization_l1076_107688


namespace NUMINAMATH_CALUDE_min_candies_count_l1076_107684

theorem min_candies_count (c : ℕ) : 
  c % 6 = 5 → 
  c % 8 = 7 → 
  c % 9 = 6 → 
  c % 11 = 0 → 
  (∀ n : ℕ, n < c → 
    (n % 6 = 5 ∧ n % 8 = 7 ∧ n % 9 = 6 ∧ n % 11 = 0) → False) → 
  c = 359 := by
sorry

end NUMINAMATH_CALUDE_min_candies_count_l1076_107684


namespace NUMINAMATH_CALUDE_value_of_t_l1076_107696

theorem value_of_t (u m j : ℝ) (A t : ℝ) (h : A = u^m / (2 + j)^t) :
  t = Real.log (u^m / A) / Real.log (2 + j) := by
  sorry

end NUMINAMATH_CALUDE_value_of_t_l1076_107696


namespace NUMINAMATH_CALUDE_chemical_mixture_problem_l1076_107602

/-- Given two solutions x and y, where:
  - x has A% of chemical a and 90% of chemical b
  - y has 20% of chemical a and 80% of chemical b
  - A mixture of x and y is 12% chemical a
  - The mixture is 80% solution x and 20% solution y
  Prove that A = 10 -/
theorem chemical_mixture_problem (A : ℝ) : 
  A + 90 = 100 →
  0.8 * A + 0.2 * 20 = 12 →
  A = 10 := by sorry

end NUMINAMATH_CALUDE_chemical_mixture_problem_l1076_107602


namespace NUMINAMATH_CALUDE_probability_point_between_C_and_D_l1076_107655

/-- Given points A, B, C, and D on a line segment AB where AB = 4AD and AB = 3BC,
    the probability that a randomly selected point on AB is between C and D is 5/12. -/
theorem probability_point_between_C_and_D 
  (A B C D : ℝ) 
  (h_order : A ≤ C ∧ C ≤ D ∧ D ≤ B) 
  (h_AB_4AD : B - A = 4 * (D - A))
  (h_AB_3BC : B - A = 3 * (B - C)) : 
  (D - C) / (B - A) = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_point_between_C_and_D_l1076_107655


namespace NUMINAMATH_CALUDE_largest_value_l1076_107680

theorem largest_value (x y z w : ℝ) (h : x + 3 = y - 1 ∧ x + 3 = z + 5 ∧ x + 3 = w - 4) :
  w ≥ x ∧ w ≥ y ∧ w ≥ z :=
by sorry

end NUMINAMATH_CALUDE_largest_value_l1076_107680


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1076_107608

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}
def B : Set ℝ := {x : ℝ | x < 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1076_107608


namespace NUMINAMATH_CALUDE_tetrahedron_volume_not_unique_l1076_107618

/-- Represents a tetrahedron with face areas and circumradius -/
structure Tetrahedron where
  face_areas : Fin 4 → ℝ
  circumradius : ℝ

/-- The volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of a tetrahedron is not uniquely determined by its face areas and circumradius -/
theorem tetrahedron_volume_not_unique : ∃ (t1 t2 : Tetrahedron), 
  (∀ i : Fin 4, t1.face_areas i = t2.face_areas i) ∧ 
  t1.circumradius = t2.circumradius ∧ 
  volume t1 ≠ volume t2 :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_not_unique_l1076_107618


namespace NUMINAMATH_CALUDE_concentric_circles_no_common_tangents_l1076_107675

-- Define a circle in a plane
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define concentric circles
def concentric (c1 c2 : Circle) : Prop :=
  c1.center = c2.center ∧ c1.radius ≠ c2.radius

-- Define a tangent line to a circle
def is_tangent_to (line : ℝ × ℝ → ℝ) (c : Circle) : Prop :=
  ∃ (point : ℝ × ℝ), line point = 0 ∧ 
    (point.1 - c.center.1)^2 + (point.2 - c.center.2)^2 = c.radius^2

-- Theorem: Two concentric circles have 0 common tangents
theorem concentric_circles_no_common_tangents (c1 c2 : Circle) 
  (h : concentric c1 c2) : 
  ¬∃ (line : ℝ × ℝ → ℝ), is_tangent_to line c1 ∧ is_tangent_to line c2 :=
sorry

end NUMINAMATH_CALUDE_concentric_circles_no_common_tangents_l1076_107675


namespace NUMINAMATH_CALUDE_log_difference_equals_six_l1076_107601

theorem log_difference_equals_six : 
  ∀ (log₄ : ℝ → ℝ),
  (log₄ 256 = 4) →
  (log₄ (1/16) = -2) →
  (log₄ 256 - log₄ (1/16) = 6) :=
by
  sorry

end NUMINAMATH_CALUDE_log_difference_equals_six_l1076_107601


namespace NUMINAMATH_CALUDE_quilt_shaded_fraction_l1076_107697

/-- Represents a square quilt block -/
structure QuiltBlock where
  total_squares : ℕ
  shaded_squares : ℕ
  half_shaded_squares : ℕ

/-- Calculates the fraction of the quilt block that is shaded -/
def shaded_fraction (q : QuiltBlock) : ℚ :=
  (q.shaded_squares + q.half_shaded_squares / 2) / q.total_squares

/-- Theorem stating that the given quilt block has 3/8 of its area shaded -/
theorem quilt_shaded_fraction :
  let q : QuiltBlock := {
    total_squares := 16,
    shaded_squares := 4,
    half_shaded_squares := 4
  }
  shaded_fraction q = 3 / 8 := by sorry

end NUMINAMATH_CALUDE_quilt_shaded_fraction_l1076_107697


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1076_107600

theorem arithmetic_computation : 12 + 4 * (5 - 9)^2 / 2 = 44 := by sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1076_107600


namespace NUMINAMATH_CALUDE_douglas_weight_is_52_l1076_107689

/-- Anne's weight in pounds -/
def anne_weight : ℕ := 67

/-- The difference in weight between Anne and Douglas in pounds -/
def weight_difference : ℕ := 15

/-- Douglas's weight in pounds -/
def douglas_weight : ℕ := anne_weight - weight_difference

/-- Theorem stating Douglas's weight -/
theorem douglas_weight_is_52 : douglas_weight = 52 := by
  sorry

end NUMINAMATH_CALUDE_douglas_weight_is_52_l1076_107689


namespace NUMINAMATH_CALUDE_smallest_five_digit_multiple_of_3_and_5_l1076_107672

theorem smallest_five_digit_multiple_of_3_and_5 : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- n is a five-digit number
  n % 3 = 0 ∧                 -- n is divisible by 3
  n % 5 = 0 ∧                 -- n is divisible by 5
  (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 3 = 0 ∧ m % 5 = 0 → m ≥ n) ∧  -- n is the smallest such number
  n = 10005                   -- the value of n is 10005
  := by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_multiple_of_3_and_5_l1076_107672


namespace NUMINAMATH_CALUDE_triangle_properties_l1076_107643

/-- Given a triangle ABC with interior angles A, B, and C, prove the magnitude of A and the maximum perimeter. -/
theorem triangle_properties (A B C : Real) (R : Real) : 
  -- Conditions
  A + B + C = π ∧ 
  (Real.cos B * Real.cos C - Real.sin B * Real.sin C = 1/2) ∧
  R = 2 →
  -- Conclusions
  A = 2*π/3 ∧ 
  ∃ (L : Real), L = 2*Real.sqrt 3 + 4 ∧ 
    ∀ (a b c : Real), 
      a / Real.sin A = 2*R → 
      a^2 = b^2 + c^2 - 2*b*c*Real.cos A → 
      a + b + c ≤ L :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1076_107643


namespace NUMINAMATH_CALUDE_second_turkey_weight_proof_l1076_107612

/-- The weight of the second turkey in kilograms -/
def second_turkey_weight : ℝ := 9

/-- The total cost of all turkeys in dollars -/
def total_cost : ℝ := 66

/-- The cost of turkey per kilogram in dollars -/
def cost_per_kg : ℝ := 2

/-- The weight of the first turkey in kilograms -/
def first_turkey_weight : ℝ := 6

theorem second_turkey_weight_proof :
  second_turkey_weight = 9 :=
by
  have h1 : total_cost = (first_turkey_weight + second_turkey_weight + 2 * second_turkey_weight) * cost_per_kg :=
    sorry
  have h2 : total_cost = (6 + 3 * second_turkey_weight) * 2 :=
    sorry
  have h3 : 66 = (6 + 3 * second_turkey_weight) * 2 :=
    sorry
  have h4 : 33 = 6 + 3 * second_turkey_weight :=
    sorry
  have h5 : 27 = 3 * second_turkey_weight :=
    sorry
  sorry

end NUMINAMATH_CALUDE_second_turkey_weight_proof_l1076_107612


namespace NUMINAMATH_CALUDE_rectangle_ratio_l1076_107641

theorem rectangle_ratio (w : ℝ) (h1 : w > 0) (h2 : 2 * w + 2 * 10 = 36) : w / 10 = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l1076_107641


namespace NUMINAMATH_CALUDE_student_number_problem_l1076_107633

theorem student_number_problem (x : ℝ) : 5 * x - 138 = 102 → x = 48 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l1076_107633


namespace NUMINAMATH_CALUDE_rectangular_parallelepiped_surface_area_l1076_107686

theorem rectangular_parallelepiped_surface_area
  (m n : ℤ)
  (h_m_lt_n : m < n)
  (x y z : ℤ)
  (h_x : x = n * (n - m))
  (h_y : y = m * n)
  (h_z : z = m * (n - m)) :
  2 * (x + y) * z = 2 * x * y := by
  sorry

end NUMINAMATH_CALUDE_rectangular_parallelepiped_surface_area_l1076_107686


namespace NUMINAMATH_CALUDE_unique_solution_l1076_107652

theorem unique_solution (x y : ℝ) : 
  |x - 2*y + 1| + (x + y - 5)^2 = 0 → x = 3 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1076_107652


namespace NUMINAMATH_CALUDE_two_intersecting_lines_determine_plane_l1076_107654

-- Define the basic types
def Point : Type := sorry
def Line : Type := sorry
def Plane : Type := sorry

-- Define the axioms of solid geometry (focusing on Axiom 3)
axiom intersecting_lines (l1 l2 : Line) : Prop
axiom determine_plane (l1 l2 : Line) (p : Plane) : Prop

-- Axiom 3: Two intersecting lines determine a plane
axiom axiom_3 (l1 l2 : Line) (p : Plane) : 
  intersecting_lines l1 l2 → determine_plane l1 l2 p

-- Theorem to prove
theorem two_intersecting_lines_determine_plane (l1 l2 : Line) :
  intersecting_lines l1 l2 → ∃ p : Plane, determine_plane l1 l2 p :=
sorry

end NUMINAMATH_CALUDE_two_intersecting_lines_determine_plane_l1076_107654


namespace NUMINAMATH_CALUDE_max_triangle_area_l1076_107665

/-- Ellipse E with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_ecc : (a^2 - b^2) / a^2 = 3/4
  h_point : 1/a^2 + 3/(4*b^2) = 1

/-- Line l intersecting the ellipse -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  m : ℝ
  h_intersect : ∃ (x y : ℝ), x^2/E.a^2 + y^2/E.b^2 = 1 ∧ y = k*x + m

/-- Perpendicular bisector condition -/
def perp_bisector_condition (E : Ellipse) (l : IntersectingLine E) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁^2/E.a^2 + y₁^2/E.b^2 = 1 ∧
    x₂^2/E.a^2 + y₂^2/E.b^2 = 1 ∧
    y₁ = l.k*x₁ + l.m ∧
    y₂ = l.k*x₂ + l.m ∧
    (x₁ + x₂)/2 = 0 ∧
    (y₁ + y₂)/2 = 1/2

/-- Area of triangle AOB -/
def triangle_area (E : Ellipse) (l : IntersectingLine E) : ℝ :=
  sorry

/-- Theorem: Maximum area of triangle AOB is 1 -/
theorem max_triangle_area (E : Ellipse) :
  ∃ (l : IntersectingLine E),
    perp_bisector_condition E l ∧
    triangle_area E l = 1 ∧
    ∀ (l' : IntersectingLine E),
      perp_bisector_condition E l' →
      triangle_area E l' ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_l1076_107665


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1076_107669

theorem x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : 3 * x + y = 18) : 
  x^2 - y^2 = -72 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1076_107669


namespace NUMINAMATH_CALUDE_inequality_solution_l1076_107625

theorem inequality_solution (x : ℝ) : 
  3 - 1 / (4 * x + 6) ≤ 5 ↔ x < -3/2 ∨ x > -1/8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1076_107625


namespace NUMINAMATH_CALUDE_blue_apples_count_l1076_107645

theorem blue_apples_count (b : ℕ) : 
  (3 * b : ℚ) - (3 * b : ℚ) / 5 = 12 → b = 5 := by sorry

end NUMINAMATH_CALUDE_blue_apples_count_l1076_107645


namespace NUMINAMATH_CALUDE_integer_power_sum_l1076_107682

theorem integer_power_sum (x : ℝ) (h1 : x ≠ 0) (h2 : ∃ k : ℤ, x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/(x^n) = m :=
sorry

end NUMINAMATH_CALUDE_integer_power_sum_l1076_107682


namespace NUMINAMATH_CALUDE_total_seeds_in_garden_l1076_107613

/-- Represents the number of beds of each type in the garden -/
def num_beds : ℕ := 2

/-- Represents the number of rows in a top bed -/
def top_rows : ℕ := 4

/-- Represents the number of seeds per row in a top bed -/
def top_seeds_per_row : ℕ := 25

/-- Represents the number of rows in a medium bed -/
def medium_rows : ℕ := 3

/-- Represents the number of seeds per row in a medium bed -/
def medium_seeds_per_row : ℕ := 20

/-- Calculates the total number of seeds that can be planted in Grace's raised bed garden -/
theorem total_seeds_in_garden : 
  num_beds * (top_rows * top_seeds_per_row + medium_rows * medium_seeds_per_row) = 320 := by
  sorry

end NUMINAMATH_CALUDE_total_seeds_in_garden_l1076_107613


namespace NUMINAMATH_CALUDE_f_properties_l1076_107623

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/2) * m * (x - 1)^2 - 2*x + 3 + Real.log x

theorem f_properties (m : ℝ) (h : m ≥ 1) :
  (∃ a b, a > 0 ∧ b > 0 ∧ a < b ∧ ∀ x ∈ Set.Icc a b, (deriv (f m)) x ≤ 0) ∧
  (∃! m, ∀ x, x > 0 → (f m x = -x + 2 → x = 1)) ∧
  (∀ x, x > 0 → (f 1 x = -x + 2 → x = 1)) := by sorry

end NUMINAMATH_CALUDE_f_properties_l1076_107623


namespace NUMINAMATH_CALUDE_set_equality_l1076_107694

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 3}

theorem set_equality : (U \ M) ∩ (U \ N) = {5, 6} := by sorry

end NUMINAMATH_CALUDE_set_equality_l1076_107694


namespace NUMINAMATH_CALUDE_cab_driver_average_income_l1076_107663

def income : List ℝ := [45, 50, 60, 65, 70]

theorem cab_driver_average_income :
  (income.sum / income.length : ℝ) = 58 := by
  sorry

end NUMINAMATH_CALUDE_cab_driver_average_income_l1076_107663


namespace NUMINAMATH_CALUDE_uranium_conductivity_is_deductive_reasoning_l1076_107634

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Metal : U → Prop)
variable (ConductsElectricity : U → Prop)

-- Define uranium as a constant in our universe
variable (uranium : U)

-- Define what deductive reasoning is
def is_deductive_reasoning (premise1 premise2 conclusion : Prop) : Prop :=
  (premise1 ∧ premise2) → conclusion

-- State the theorem
theorem uranium_conductivity_is_deductive_reasoning :
  is_deductive_reasoning
    (∀ x : U, Metal x → ConductsElectricity x)
    (Metal uranium)
    (ConductsElectricity uranium) :=
by
  sorry


end NUMINAMATH_CALUDE_uranium_conductivity_is_deductive_reasoning_l1076_107634


namespace NUMINAMATH_CALUDE_only_prime_perfect_square_l1076_107653

theorem only_prime_perfect_square : 
  ∀ p : ℕ, Prime p → (∃ k : ℕ, 5^p + 12^p = k^2) → p = 2 :=
by sorry

end NUMINAMATH_CALUDE_only_prime_perfect_square_l1076_107653


namespace NUMINAMATH_CALUDE_sum_of_abc_l1076_107607

theorem sum_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 30) (hac : a * c = 60) (hbc : b * c = 90) :
  a + b + c = 11 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_abc_l1076_107607


namespace NUMINAMATH_CALUDE_august_electricity_bill_l1076_107673

-- Define electricity prices for different seasons
def electricity_price (month : Nat) : Real :=
  if month ≤ 3 then 0.12
  else if month ≤ 6 then 0.10
  else if month ≤ 9 then 0.09
  else 0.11

-- Define appliance consumption rates
def oven_consumption : Real := 2.4
def ac_consumption : Real := 1.6
def fridge_consumption : Real := 0.15
def washer_consumption : Real := 0.5

-- Define appliance usage durations
def oven_usage : Nat := 25
def ac_usage : Nat := 150
def fridge_usage : Nat := 720
def washer_usage : Nat := 20

-- Define the month of August
def august : Nat := 8

-- Theorem: Coco's total electricity bill for August is $37.62
theorem august_electricity_bill :
  let price := electricity_price august
  let oven_cost := oven_consumption * oven_usage * price
  let ac_cost := ac_consumption * ac_usage * price
  let fridge_cost := fridge_consumption * fridge_usage * price
  let washer_cost := washer_consumption * washer_usage * price
  oven_cost + ac_cost + fridge_cost + washer_cost = 37.62 := by
  sorry


end NUMINAMATH_CALUDE_august_electricity_bill_l1076_107673


namespace NUMINAMATH_CALUDE_min_value_expression_l1076_107631

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  x^2 + 4*x*y + 9*y^2 + 8*y*z + 3*z^2 ≥ 9^(10/9) ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 ∧
    x₀^2 + 4*x₀*y₀ + 9*y₀^2 + 8*y₀*z₀ + 3*z₀^2 = 9^(10/9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1076_107631


namespace NUMINAMATH_CALUDE_giorgio_class_size_l1076_107671

/-- The number of students in Giorgio's class -/
def num_students : ℕ := sorry

/-- The number of cookies Giorgio makes per student -/
def cookies_per_student : ℕ := 2

/-- The percentage of students who want oatmeal raisin cookies -/
def oatmeal_raisin_percentage : ℚ := 1/10

/-- The number of oatmeal raisin cookies Giorgio makes -/
def num_oatmeal_raisin_cookies : ℕ := 8

/-- Theorem stating that the number of students in Giorgio's class is 40 -/
theorem giorgio_class_size :
  num_students = 40 :=
sorry

end NUMINAMATH_CALUDE_giorgio_class_size_l1076_107671


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_cube_l1076_107640

theorem smallest_multiplier_for_cube (n : ℕ) : 
  (∀ m : ℕ, m < 300 → ¬∃ k : ℕ, 720 * m = k^3) ∧ 
  (∃ k : ℕ, 720 * 300 = k^3) := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_cube_l1076_107640


namespace NUMINAMATH_CALUDE_johns_arcade_spending_l1076_107647

theorem johns_arcade_spending (allowance : ℚ) (arcade_fraction : ℚ) :
  allowance = 3/2 →
  2/3 * (1 - arcade_fraction) * allowance = 2/5 →
  arcade_fraction = 3/5 := by
sorry

end NUMINAMATH_CALUDE_johns_arcade_spending_l1076_107647


namespace NUMINAMATH_CALUDE_leo_current_weight_l1076_107646

/-- Leo's current weight in pounds -/
def leo_weight : ℝ := 98

/-- Kendra's current weight in pounds -/
def kendra_weight : ℝ := 170 - leo_weight

theorem leo_current_weight :
  (leo_weight + 10 = 1.5 * kendra_weight) ∧
  (leo_weight + kendra_weight = 170) →
  leo_weight = 98 := by
sorry

end NUMINAMATH_CALUDE_leo_current_weight_l1076_107646


namespace NUMINAMATH_CALUDE_congruence_and_range_implies_value_l1076_107658

theorem congruence_and_range_implies_value :
  ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -1234 [ZMOD 8] → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_congruence_and_range_implies_value_l1076_107658


namespace NUMINAMATH_CALUDE_fraction_simplification_l1076_107698

theorem fraction_simplification (x : ℝ) : (x - 2) / 4 - (3 * x + 1) / 3 = (-9 * x - 10) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1076_107698


namespace NUMINAMATH_CALUDE_sachin_age_l1076_107679

theorem sachin_age : 
  ∀ (s r : ℕ), 
  r = s + 8 →  -- Sachin is younger than Rahul by 8 years
  s * 9 = r * 7 →  -- The ratio of their ages is 7 : 9
  s = 28 :=  -- Sachin's age is 28 years
by
  sorry

end NUMINAMATH_CALUDE_sachin_age_l1076_107679


namespace NUMINAMATH_CALUDE_sequence_a2_value_l1076_107644

theorem sequence_a2_value (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n, S n = 2 * (a n - 1)) : a 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sequence_a2_value_l1076_107644


namespace NUMINAMATH_CALUDE_fraction_relation_l1076_107699

theorem fraction_relation (x : ℝ) (h : 1 / (x + 3) = 2) : 1 / (x + 5) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_relation_l1076_107699


namespace NUMINAMATH_CALUDE_basketball_shot_expectation_l1076_107627

theorem basketball_shot_expectation (a b : ℝ) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (h_exp : 3 * a + 2 * b = 2) :
  (∀ x y : ℝ, 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ 3 * x + 2 * y = 2 → 
    2 / a + 1 / (3 * b) ≤ 2 / x + 1 / (3 * y)) ∧
  2 / a + 1 / (3 * b) = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_basketball_shot_expectation_l1076_107627


namespace NUMINAMATH_CALUDE_polygon_division_theorem_l1076_107661

/-- A polygon that can be divided into a specific number of rectangles -/
structure DivisiblePolygon where
  vertices : ℕ
  can_divide : ℕ → Prop
  h_100 : can_divide 100
  h_not_99 : ¬ can_divide 99

/-- The main theorem stating that a polygon divisible into 100 rectangles but not 99
    has more than 200 vertices and cannot be divided into 100 triangles -/
theorem polygon_division_theorem (P : DivisiblePolygon) :
  P.vertices > 200 ∧ ¬ ∃ (triangles : ℕ), triangles = 100 ∧ P.can_divide triangles := by
  sorry


end NUMINAMATH_CALUDE_polygon_division_theorem_l1076_107661
