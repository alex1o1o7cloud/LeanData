import Mathlib

namespace NUMINAMATH_CALUDE_parabola_vertex_l2756_275699

/-- The vertex of a parabola defined by y = x^2 + 2x - 3 is (-1, -4) -/
theorem parabola_vertex : 
  let f (x : ℝ) := x^2 + 2*x - 3
  ∃! (a b : ℝ), (∀ x, f x = (x - a)^2 + b) ∧ (a = -1 ∧ b = -4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2756_275699


namespace NUMINAMATH_CALUDE_point_on_direct_proportion_l2756_275683

/-- A direct proportion function passing through two points -/
def DirectProportion (k : ℝ) (x y : ℝ) : Prop := y = k * x

/-- The theorem stating that if A(3,-5) and B(-6,a) lie on a direct proportion function, then a = 10 -/
theorem point_on_direct_proportion (k a : ℝ) :
  DirectProportion k 3 (-5) ∧ DirectProportion k (-6) a → a = 10 := by
  sorry

end NUMINAMATH_CALUDE_point_on_direct_proportion_l2756_275683


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l2756_275615

theorem largest_angle_in_special_triangle (α β γ : Real) : 
  α + β + γ = π ∧ 
  0 < α ∧ 0 < β ∧ 0 < γ ∧
  Real.tan α + Real.tan β + Real.tan γ = 2016 →
  (max α (max β γ)) > π/2 - π/360 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l2756_275615


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l2756_275671

theorem lcm_gcf_problem (n : ℕ+) :
  (Nat.lcm n 12 = 48) → (Nat.gcd n 12 = 8) → n = 32 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l2756_275671


namespace NUMINAMATH_CALUDE_total_seashells_l2756_275629

theorem total_seashells (sally tom jessica alex : ℝ) 
  (h1 : sally = 9.5)
  (h2 : tom = 7.2)
  (h3 : jessica = 5.3)
  (h4 : alex = 12.8) :
  sally + tom + jessica + alex = 34.8 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_l2756_275629


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_l2756_275617

def i : ℂ := Complex.I

theorem pure_imaginary_complex (a : ℝ) : 
  (∃ (b : ℝ), (2 - i) * (a - i) = b * i ∧ b ≠ 0) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_l2756_275617


namespace NUMINAMATH_CALUDE_shaded_area_sum_l2756_275695

def circle_setup (r₁ r₂ r₃ : ℝ) : Prop :=
  r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧
  r₁ * r₁ = 100 ∧
  r₂ = r₁ / 2 ∧
  r₃ = r₂ / 2

theorem shaded_area_sum (r₁ r₂ r₃ : ℝ) 
  (h : circle_setup r₁ r₂ r₃) : 
  (π * r₁ * r₁ / 2) + (π * r₂ * r₂ / 2) + (π * r₃ * r₃ / 2) = 65.625 * π :=
by
  sorry

#check shaded_area_sum

end NUMINAMATH_CALUDE_shaded_area_sum_l2756_275695


namespace NUMINAMATH_CALUDE_fixed_point_on_graph_l2756_275656

theorem fixed_point_on_graph (k : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 9 * x^2 + 3 * k * x - 5 * k
  f 5 = 225 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_graph_l2756_275656


namespace NUMINAMATH_CALUDE_cubic_function_properties_monotonicity_interval_l2756_275659

/-- A cubic function f(x) = ax^3 + bx^2 passing through (1,4) with slope 9 at x = 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x

theorem cubic_function_properties (a b : ℝ) :
  f a b 1 = 4 ∧ f' a b 1 = 9 → a = 1 ∧ b = 3 :=
sorry

theorem monotonicity_interval (a b m : ℝ) :
  (a = 1 ∧ b = 3) →
  (∀ x ∈ Set.Icc m (m + 1), f' a b x ≥ 0) ↔ (m ≥ 0 ∨ m ≤ -3) :=
sorry

end NUMINAMATH_CALUDE_cubic_function_properties_monotonicity_interval_l2756_275659


namespace NUMINAMATH_CALUDE_expression_value_l2756_275653

theorem expression_value (x y z : ℝ) 
  (hx : x = -5/4) 
  (hy : y = -3/2) 
  (hz : z = Real.sqrt 2) : 
  -2 * x^3 - y^2 + Real.sin z = 53/32 + Real.sin (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2756_275653


namespace NUMINAMATH_CALUDE_max_area_rectangle_with_perimeter_60_l2756_275623

/-- The maximum area of a rectangle with perimeter 60 is 225 -/
theorem max_area_rectangle_with_perimeter_60 :
  ∀ a b : ℝ,
  a > 0 → b > 0 →
  2 * a + 2 * b = 60 →
  a * b ≤ 225 :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangle_with_perimeter_60_l2756_275623


namespace NUMINAMATH_CALUDE_specific_pyramid_side_edge_l2756_275651

/-- Regular square pyramid with given base edge length and volume -/
structure RegularSquarePyramid where
  base_edge : ℝ
  volume : ℝ

/-- The side edge length of a regular square pyramid -/
def side_edge_length (p : RegularSquarePyramid) : ℝ :=
  sorry

/-- Theorem stating the side edge length of a specific regular square pyramid -/
theorem specific_pyramid_side_edge :
  let p : RegularSquarePyramid := ⟨4 * Real.sqrt 2, 32⟩
  side_edge_length p = 5 := by
  sorry

end NUMINAMATH_CALUDE_specific_pyramid_side_edge_l2756_275651


namespace NUMINAMATH_CALUDE_tickets_per_box_l2756_275607

theorem tickets_per_box (total_tickets : ℕ) (num_boxes : ℕ) (h1 : total_tickets = 45) (h2 : num_boxes = 9) :
  total_tickets / num_boxes = 5 := by
  sorry

end NUMINAMATH_CALUDE_tickets_per_box_l2756_275607


namespace NUMINAMATH_CALUDE_abs_two_set_l2756_275652

theorem abs_two_set : {x : ℝ | |x| = 2} = {-2, 2} := by
  sorry

end NUMINAMATH_CALUDE_abs_two_set_l2756_275652


namespace NUMINAMATH_CALUDE_direction_vector_of_line_l2756_275626

/-- Given a line l with equation x + y + 1 = 0, prove that (1, -1) is a direction vector of l. -/
theorem direction_vector_of_line (l : Set (ℝ × ℝ)) :
  (∀ p : ℝ × ℝ, p ∈ l ↔ p.1 + p.2 + 1 = 0) →
  ∃ t : ℝ, (1 + t, -1 + t) ∈ l := by sorry

end NUMINAMATH_CALUDE_direction_vector_of_line_l2756_275626


namespace NUMINAMATH_CALUDE_union_equality_intersection_equality_l2756_275684

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 3*x - 10 ≤ 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | m - 4 ≤ x ∧ x ≤ 3*m + 2}

-- Theorem for the first question
theorem union_equality (m : ℝ) : A ∪ B m = B m ↔ m ∈ Set.Icc 1 2 := by sorry

-- Theorem for the second question
theorem intersection_equality (m : ℝ) : A ∩ B m = B m ↔ m < -3 := by sorry

end NUMINAMATH_CALUDE_union_equality_intersection_equality_l2756_275684


namespace NUMINAMATH_CALUDE_surface_area_of_sliced_solid_l2756_275603

/-- A right prism with equilateral triangular bases -/
structure RightPrism where
  height : ℝ
  base_side : ℝ

/-- Midpoints of edges in the prism -/
structure Midpoints where
  X : ℝ × ℝ × ℝ
  Y : ℝ × ℝ × ℝ
  Z : ℝ × ℝ × ℝ

/-- The solid formed by slicing off a part of the prism -/
def SlicedSolid (p : RightPrism) (m : Midpoints) : ℝ × ℝ × ℝ × ℝ := sorry

/-- Surface area of the sliced solid -/
def surfaceArea (s : ℝ × ℝ × ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem surface_area_of_sliced_solid (p : RightPrism) (m : Midpoints) :
  p.height = 20 ∧ p.base_side = 10 →
  surfaceArea (SlicedSolid p m) = 100 + 25 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_sliced_solid_l2756_275603


namespace NUMINAMATH_CALUDE_range_of_H_l2756_275658

-- Define the function H
def H (x : ℝ) : ℝ := |x + 2|^2 - |x - 2|^2

-- State the theorem about the range of H
theorem range_of_H :
  ∀ y : ℝ, ∃ x : ℝ, H x = y :=
sorry

end NUMINAMATH_CALUDE_range_of_H_l2756_275658


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2756_275640

theorem fraction_to_decimal : (47 : ℚ) / (2^2 * 5^4) = 0.0188 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2756_275640


namespace NUMINAMATH_CALUDE_point_A_in_third_quadrant_l2756_275618

/-- A point in the Cartesian coordinate system is in the third quadrant if and only if
    both its x and y coordinates are negative. -/
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- The point A with coordinates (-1, -3) lies in the third quadrant. -/
theorem point_A_in_third_quadrant :
  third_quadrant (-1) (-3) := by
  sorry

end NUMINAMATH_CALUDE_point_A_in_third_quadrant_l2756_275618


namespace NUMINAMATH_CALUDE_fifth_term_is_13_l2756_275664

/-- A sequence where the difference between consecutive terms increases by 1 each time -/
def increasing_diff_seq (a₁ : ℕ) (d₁ : ℕ) : ℕ → ℕ
| 0 => a₁
| n + 1 => increasing_diff_seq a₁ d₁ n + d₁ + n

theorem fifth_term_is_13 (a₁ d₁ : ℕ) :
  a₁ = 3 ∧ d₁ = 1 →
  increasing_diff_seq a₁ d₁ 1 = 4 ∧
  increasing_diff_seq a₁ d₁ 2 = 6 ∧
  increasing_diff_seq a₁ d₁ 3 = 9 →
  increasing_diff_seq a₁ d₁ 4 = 13 := by
  sorry

#eval increasing_diff_seq 3 1 4  -- Should output 13

end NUMINAMATH_CALUDE_fifth_term_is_13_l2756_275664


namespace NUMINAMATH_CALUDE_divisibility_problem_l2756_275622

theorem divisibility_problem (a b c d m : ℤ) 
  (h_m_pos : m > 0)
  (h_ac : m ∣ a * c)
  (h_bd : m ∣ b * d)
  (h_sum : m ∣ b * c + a * d) :
  (m ∣ b * c) ∧ (m ∣ a * d) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l2756_275622


namespace NUMINAMATH_CALUDE_largest_number_problem_l2756_275672

theorem largest_number_problem (a b c : ℝ) : 
  a < b ∧ b < c →
  a + b + c = 77 →
  c - b = 9 →
  b - a = 5 →
  c = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_problem_l2756_275672


namespace NUMINAMATH_CALUDE_train_crossing_time_l2756_275678

theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 90 ∧ train_speed_kmh = 72 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2756_275678


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_equals_one_l2756_275691

/-- Given two lines l₁: ax + (3-a)y + 1 = 0 and l₂: 2x - y = 0,
    if l₁ is perpendicular to l₂, then a = 1 -/
theorem perpendicular_lines_a_equals_one (a : ℝ) :
  (∀ x y : ℝ, a * x + (3 - a) * y + 1 = 0 → 2 * x - y = 0 → 
    (a * 2 + (-1) * (3 - a) = 0)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_equals_one_l2756_275691


namespace NUMINAMATH_CALUDE_gcd_282_470_l2756_275601

theorem gcd_282_470 : Nat.gcd 282 470 = 94 := by sorry

end NUMINAMATH_CALUDE_gcd_282_470_l2756_275601


namespace NUMINAMATH_CALUDE_unit_vector_parallel_to_d_l2756_275689

def vector_d : Fin 2 → ℝ := ![12, 5]

theorem unit_vector_parallel_to_d :
  let magnitude : ℝ := Real.sqrt (12^2 + 5^2)
  let unit_vector_positive : Fin 2 → ℝ := ![12 / magnitude, 5 / magnitude]
  let unit_vector_negative : Fin 2 → ℝ := ![-12 / magnitude, -5 / magnitude]
  (∀ i, vector_d i = magnitude * unit_vector_positive i) ∧
  (∀ i, vector_d i = magnitude * unit_vector_negative i) ∧
  (∀ i, unit_vector_positive i * unit_vector_positive i + 
        unit_vector_negative i * unit_vector_negative i = 2) :=
by sorry

end NUMINAMATH_CALUDE_unit_vector_parallel_to_d_l2756_275689


namespace NUMINAMATH_CALUDE_expression_change_l2756_275690

/-- The change in the expression x³ - 5x + 1 when x changes by b -/
def expressionChange (x b : ℝ) : ℝ :=
  let f := fun t => t^3 - 5*t + 1
  f (x + b) - f x

theorem expression_change (x b : ℝ) (h : b > 0) :
  expressionChange x b = 3*b*x^2 + 3*b^2*x + b^3 - 5*b ∨
  expressionChange x (-b) = -3*b*x^2 + 3*b^2*x - b^3 + 5*b :=
sorry

end NUMINAMATH_CALUDE_expression_change_l2756_275690


namespace NUMINAMATH_CALUDE_cubic_root_sum_squares_over_product_l2756_275633

theorem cubic_root_sum_squares_over_product (k : ℤ) (hk : k ≠ 0) 
  (a b c : ℂ) (h : ∀ x : ℂ, x^3 + 10*x^2 + 5*x - k = 0 ↔ x = a ∨ x = b ∨ x = c) : 
  (a^2 + b^2 + c^2) / (a * b * c) = 90 / k := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_squares_over_product_l2756_275633


namespace NUMINAMATH_CALUDE_temperature_difference_l2756_275697

theorem temperature_difference (highest lowest : Int) 
  (h1 : highest = 8) 
  (h2 : lowest = -2) : 
  highest - lowest = 10 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l2756_275697


namespace NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_factor_l2756_275639

def number : Nat := 15999

-- Define a function to get the greatest prime factor
def greatest_prime_factor (n : Nat) : Nat :=
  sorry

-- Define a function to sum the digits of a number
def sum_of_digits (n : Nat) : Nat :=
  sorry

-- Theorem statement
theorem sum_of_digits_of_greatest_prime_factor :
  sum_of_digits (greatest_prime_factor number) = 17 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_factor_l2756_275639


namespace NUMINAMATH_CALUDE_race_head_start_l2756_275667

/-- Given two runners A and B, where A's speed is 20/19 times B's speed,
    the head start fraction that A should give B for a dead heat is 1/20 of the race length. -/
theorem race_head_start (speedA speedB : ℝ) (length headStart : ℝ) :
  speedA = (20 / 19) * speedB →
  (length / speedA = (length - headStart) / speedB) →
  headStart = (1 / 20) * length :=
by sorry

end NUMINAMATH_CALUDE_race_head_start_l2756_275667


namespace NUMINAMATH_CALUDE_ellipse_tangent_properties_l2756_275636

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the companion circle E
def companion_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define a point on the companion circle
def point_on_circle (P : ℝ × ℝ) : Prop := companion_circle P.1 P.2

-- Define a tangent line to the ellipse
def is_tangent (P A : ℝ × ℝ) : Prop :=
  point_on_circle P ∧ ellipse A.1 A.2 ∧
  ∀ t : ℝ, t ≠ 0 → ¬(ellipse (A.1 + t * (P.1 - A.1)) (A.2 + t * (P.2 - A.2)))

-- Main theorem
theorem ellipse_tangent_properties :
  ∀ P A B Q : ℝ × ℝ,
  point_on_circle P →
  is_tangent P A →
  is_tangent P B →
  companion_circle Q.1 Q.2 →
  (∃ t : ℝ, Q.1 = A.1 + t * (P.1 - A.1) ∧ Q.2 = A.2 + t * (P.2 - A.2)) →
  (A ≠ B) →
  (∀ k₁ k₂ : ℝ,
    (P.1 ≠ 0 ∨ P.2 ≠ 0) →
    (Q.1 ≠ 0 ∨ Q.2 ≠ 0) →
    k₁ = P.2 / P.1 →
    k₂ = Q.2 / Q.1 →
    (((P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0) ∧
     (k₁ * k₂ = -1/3))) :=
sorry

end NUMINAMATH_CALUDE_ellipse_tangent_properties_l2756_275636


namespace NUMINAMATH_CALUDE_impossible_arrangement_l2756_275619

/-- Represents a 3x3 grid filled with digits --/
def Grid := Fin 3 → Fin 3 → Fin 4

/-- The set of available digits --/
def AvailableDigits : Finset (Fin 4) := {0, 1, 2, 3}

/-- Check if a list of cells contains three different digits --/
def hasThreeDifferentDigits (g : Grid) (cells : List (Fin 3 × Fin 3)) : Prop :=
  (cells.map (fun (i, j) => g i j)).toFinset.card = 3

/-- Check if all rows, columns, and diagonals have three different digits --/
def isValidArrangement (g : Grid) : Prop :=
  (∀ i : Fin 3, hasThreeDifferentDigits g [(i, 0), (i, 1), (i, 2)]) ∧
  (∀ j : Fin 3, hasThreeDifferentDigits g [(0, j), (1, j), (2, j)]) ∧
  hasThreeDifferentDigits g [(0, 0), (1, 1), (2, 2)] ∧
  hasThreeDifferentDigits g [(0, 2), (1, 1), (2, 0)]

/-- Main theorem: It's impossible to arrange the digits as described --/
theorem impossible_arrangement : ¬∃ g : Grid, isValidArrangement g := by
  sorry


end NUMINAMATH_CALUDE_impossible_arrangement_l2756_275619


namespace NUMINAMATH_CALUDE_heather_biking_speed_l2756_275657

def total_distance : ℝ := 320
def num_days : ℝ := 8.0

theorem heather_biking_speed : total_distance / num_days = 40 := by
  sorry

end NUMINAMATH_CALUDE_heather_biking_speed_l2756_275657


namespace NUMINAMATH_CALUDE_bryan_skittles_count_l2756_275687

/-- Given that Ben has 20 M&M's and Bryan has 30 more candies than Ben, 
    prove that Bryan has 50 skittles. -/
theorem bryan_skittles_count : 
  ∀ (ben_candies bryan_candies : ℕ),
  ben_candies = 20 →
  bryan_candies = ben_candies + 30 →
  bryan_candies = 50 := by
sorry

end NUMINAMATH_CALUDE_bryan_skittles_count_l2756_275687


namespace NUMINAMATH_CALUDE_log_inequality_l2756_275641

theorem log_inequality (a b c : ℝ) (h1 : a < b) (h2 : 0 < c) (h3 : c < 1) :
  a * Real.log c > b * Real.log c := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l2756_275641


namespace NUMINAMATH_CALUDE_circle_radius_given_area_circumference_ratio_l2756_275666

theorem circle_radius_given_area_circumference_ratio 
  (A C : ℝ) (h1 : A > 0) (h2 : C > 0) (h3 : A / C = 10) : 
  ∃ r : ℝ, r > 0 ∧ A = π * r^2 ∧ C = 2 * π * r ∧ r = 20 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_given_area_circumference_ratio_l2756_275666


namespace NUMINAMATH_CALUDE_brilliant_permutations_l2756_275645

def word := "BRILLIANT"

/-- The number of permutations of the letters in 'BRILLIANT' where no two adjacent letters are the same -/
def valid_permutations : ℕ :=
  Nat.factorial 9 / (Nat.factorial 2 * Nat.factorial 2) -
  (Nat.factorial 8 / Nat.factorial 2 +
   Nat.factorial 8 / Nat.factorial 2 -
   Nat.factorial 7)

theorem brilliant_permutations :
  valid_permutations = 55440 :=
sorry

end NUMINAMATH_CALUDE_brilliant_permutations_l2756_275645


namespace NUMINAMATH_CALUDE_spade_calculation_l2756_275616

/-- The spade operation for real numbers -/
def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

/-- The main theorem -/
theorem spade_calculation : spade 5 (spade 7 8) = -200 := by sorry

end NUMINAMATH_CALUDE_spade_calculation_l2756_275616


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2756_275644

theorem geometric_series_sum : 
  let a : ℝ := 1
  let r : ℝ := 1/4
  let S := ∑' n, a * r^n
  S = 4/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2756_275644


namespace NUMINAMATH_CALUDE_range_of_f_l2756_275660

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 2*x + 3

-- State the theorem
theorem range_of_f :
  {y | ∃ x ≥ 0, f x = y} = {y | y ≥ 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2756_275660


namespace NUMINAMATH_CALUDE_hyperbola_properties_l2756_275627

/-- A hyperbola with asymptotes y = ±2x passing through (1, 0) -/
def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2/4 = 1

theorem hyperbola_properties :
  ∀ (x y : ℝ),
    -- The equation represents a hyperbola
    (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ x^2/a^2 - y^2/b^2 = 1) →
    -- With asymptotes y = ±2x
    (∃ (k : ℝ), k ≠ 0 ∧ (y = 2*x ∨ y = -2*x) → (x^2 - y^2/4 = k)) →
    -- Passing through the point (1, 0)
    hyperbola 1 0 →
    -- Then the hyperbola has the equation x² - y²/4 = 1
    hyperbola x y :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l2756_275627


namespace NUMINAMATH_CALUDE_sum_of_xyz_l2756_275649

theorem sum_of_xyz (p q : ℝ) (x y z : ℤ) : 
  p^2 = 25/50 →
  q^2 = (3 + Real.sqrt 7)^2 / 14 →
  p < 0 →
  q > 0 →
  (p + q)^3 = (x : ℝ) * Real.sqrt (y : ℝ) / (z : ℝ) →
  x + y + z = 177230 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l2756_275649


namespace NUMINAMATH_CALUDE_total_paving_cost_l2756_275600

/-- Represents a section of a room with its dimensions and slab cost -/
structure Section where
  length : ℝ
  width : ℝ
  slabCost : ℝ

/-- Calculates the cost of paving a section -/
def sectionCost (s : Section) : ℝ :=
  s.length * s.width * s.slabCost

/-- The three sections of the room -/
def sectionA : Section := { length := 8, width := 4.75, slabCost := 900 }
def sectionB : Section := { length := 6, width := 3.25, slabCost := 800 }
def sectionC : Section := { length := 5, width := 2.5, slabCost := 1000 }

/-- Theorem stating the total cost of paving the floor for the entire room -/
theorem total_paving_cost :
  sectionCost sectionA + sectionCost sectionB + sectionCost sectionC = 62300 := by
  sorry


end NUMINAMATH_CALUDE_total_paving_cost_l2756_275600


namespace NUMINAMATH_CALUDE_max_x_plus_y_l2756_275648

theorem max_x_plus_y (x y : ℝ) (h : x^2 + 3*y^2 = 1) :
  ∃ (max_x max_y : ℝ), max_x^2 + 3*max_y^2 = 1 ∧
  ∀ (a b : ℝ), a^2 + 3*b^2 = 1 → a + b ≤ max_x + max_y ∧
  max_x = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_max_x_plus_y_l2756_275648


namespace NUMINAMATH_CALUDE_max_profit_theorem_l2756_275602

/-- Represents the unit prices and quantities of exercise books -/
structure BookPrices where
  regular : ℝ
  deluxe : ℝ

/-- Represents the purchase quantities of exercise books -/
structure PurchaseQuantities where
  regular : ℝ
  deluxe : ℝ

/-- Defines the conditions of the problem -/
def problem_conditions (prices : BookPrices) : Prop :=
  150 * prices.regular + 100 * prices.deluxe = 1450 ∧
  200 * prices.regular + 50 * prices.deluxe = 1100

/-- Defines the profit function -/
def profit_function (prices : BookPrices) (quantities : PurchaseQuantities) : ℝ :=
  (prices.regular - 2) * quantities.regular + (prices.deluxe - 7) * quantities.deluxe

/-- Defines the purchase constraints -/
def purchase_constraints (quantities : PurchaseQuantities) : Prop :=
  quantities.regular + quantities.deluxe = 500 ∧
  quantities.regular ≥ 3 * quantities.deluxe

/-- Theorem stating the maximum profit and optimal purchase strategy -/
theorem max_profit_theorem (prices : BookPrices) 
  (h_conditions : problem_conditions prices) :
  ∃ (quantities : PurchaseQuantities),
    purchase_constraints quantities ∧
    profit_function prices quantities = 750 ∧
    ∀ (other_quantities : PurchaseQuantities),
      purchase_constraints other_quantities →
      profit_function prices other_quantities ≤ 750 :=
sorry

end NUMINAMATH_CALUDE_max_profit_theorem_l2756_275602


namespace NUMINAMATH_CALUDE_total_rent_is_7800_l2756_275686

/-- Represents the rent shares of four people renting a house -/
structure RentShares where
  purity : ℝ
  sheila : ℝ
  rose : ℝ
  john : ℝ

/-- Calculates the total rent based on the given rent shares -/
def totalRent (shares : RentShares) : ℝ :=
  shares.purity + shares.sheila + shares.rose + shares.john

/-- Theorem stating that the total rent is $7,800 given the conditions -/
theorem total_rent_is_7800 :
  ∀ (shares : RentShares),
    shares.sheila = 5 * shares.purity →
    shares.rose = 3 * shares.purity →
    shares.john = 4 * shares.purity →
    shares.rose = 1800 →
    totalRent shares = 7800 := by
  sorry

#check total_rent_is_7800

end NUMINAMATH_CALUDE_total_rent_is_7800_l2756_275686


namespace NUMINAMATH_CALUDE_cyclist_average_speed_l2756_275680

/-- The average speed of a cyclist driving four laps of equal distance at different speeds -/
theorem cyclist_average_speed (d : ℝ) (h : d > 0) : 
  let speeds := [6, 12, 18, 24]
  let total_distance := 4 * d
  let total_time := (d / 6 + d / 12 + d / 18 + d / 24)
  total_distance / total_time = 288 / 25 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_average_speed_l2756_275680


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2756_275663

def a : Fin 2 → ℝ := ![1, 2]
def b (x : ℝ) : Fin 2 → ℝ := ![x, -2]

theorem perpendicular_vectors (x : ℝ) : 
  (∀ i : Fin 2, (a i) * ((a i) - (b x i)) = 0) → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2756_275663


namespace NUMINAMATH_CALUDE_total_cost_of_collars_l2756_275669

/-- Represents the material composition and cost of a collar --/
structure Collar :=
  (nylon_inches : ℕ)
  (leather_inches : ℕ)
  (nylon_cost_per_inch : ℕ)
  (leather_cost_per_inch : ℕ)

/-- Calculates the total cost of a single collar --/
def collar_cost (c : Collar) : ℕ :=
  c.nylon_inches * c.nylon_cost_per_inch + c.leather_inches * c.leather_cost_per_inch

/-- Defines a dog collar according to the problem specifications --/
def dog_collar : Collar :=
  { nylon_inches := 18
  , leather_inches := 4
  , nylon_cost_per_inch := 1
  , leather_cost_per_inch := 2 }

/-- Defines a cat collar according to the problem specifications --/
def cat_collar : Collar :=
  { nylon_inches := 10
  , leather_inches := 2
  , nylon_cost_per_inch := 1
  , leather_cost_per_inch := 2 }

/-- Theorem stating the total cost of materials for 9 dog collars and 3 cat collars --/
theorem total_cost_of_collars :
  9 * collar_cost dog_collar + 3 * collar_cost cat_collar = 276 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_collars_l2756_275669


namespace NUMINAMATH_CALUDE_semicircle_in_quarter_circle_l2756_275685

theorem semicircle_in_quarter_circle (r : ℝ) (hr : r > 0) :
  let s := r * Real.sqrt 3
  let quarter_circle_area := π * s^2 / 4
  let semicircle_area := π * r^2 / 2
  semicircle_area / quarter_circle_area = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_semicircle_in_quarter_circle_l2756_275685


namespace NUMINAMATH_CALUDE_peach_difference_l2756_275650

theorem peach_difference (jill steven jake : ℕ) : 
  jill = 12 →
  steven = jill + 15 →
  jake = steven - 16 →
  jill - jake = 1 := by
sorry

end NUMINAMATH_CALUDE_peach_difference_l2756_275650


namespace NUMINAMATH_CALUDE_storage_box_faces_l2756_275693

theorem storage_box_faces : ∃ n : ℕ, n > 0 ∧ Nat.factorial n = 720 := by
  sorry

end NUMINAMATH_CALUDE_storage_box_faces_l2756_275693


namespace NUMINAMATH_CALUDE_min_time_for_all_flashes_l2756_275673

/-- The number of colored lights -/
def num_lights : ℕ := 5

/-- The number of colors available -/
def num_colors : ℕ := 5

/-- The time for one light to shine in seconds -/
def shine_time : ℕ := 1

/-- The interval between two consecutive flashes in seconds -/
def interval_time : ℕ := 5

/-- The number of different possible flashes -/
def num_flashes : ℕ := Nat.factorial num_lights

/-- The minimum time required to achieve all different flashes in seconds -/
def min_time_required : ℕ := 
  (num_flashes * num_lights * shine_time) + ((num_flashes - 1) * interval_time)

theorem min_time_for_all_flashes : min_time_required = 1195 := by
  sorry

end NUMINAMATH_CALUDE_min_time_for_all_flashes_l2756_275673


namespace NUMINAMATH_CALUDE_yangtze_farm_grass_consumption_l2756_275632

/-- Represents the grass consumption scenario on Yangtze Farm -/
structure GrassConsumption where
  /-- The amount of grass one cow eats in one day -/
  b : ℝ
  /-- The initial amount of grass -/
  g : ℝ
  /-- The rate of grass growth per day -/
  r : ℝ

/-- Given the conditions, proves that 36 cows will eat the grass in 3 days -/
theorem yangtze_farm_grass_consumption (gc : GrassConsumption) 
  (h1 : gc.g + 6 * gc.r = 24 * 6 * gc.b)  -- 24 cows eat the grass in 6 days
  (h2 : gc.g + 8 * gc.r = 21 * 8 * gc.b)  -- 21 cows eat the grass in 8 days
  : gc.g + 3 * gc.r = 36 * 3 * gc.b := by
  sorry


end NUMINAMATH_CALUDE_yangtze_farm_grass_consumption_l2756_275632


namespace NUMINAMATH_CALUDE_problem_statements_l2756_275643

theorem problem_statements :
  (∀ x : ℤ, x^2 + 1 > 0) ∧
  (∃ x y : ℝ, x + y > 5 ∧ ¬(x > 2 ∧ y > 3)) ∧
  (∀ x : ℝ, x^2 - x + 1 ≥ 0) ∧
  (∀ y : ℝ, y ≤ 3 → ∃ x : ℝ, y = -x^2 + 2*x + 2) ∧
  (∀ x : ℝ, -x^2 + 2*x + 2 ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l2756_275643


namespace NUMINAMATH_CALUDE_least_grood_number_l2756_275606

theorem least_grood_number (n : ℕ) : n ≥ 10 ↔ (n * (n + 1) : ℚ) / 4 > n^2 := by sorry

end NUMINAMATH_CALUDE_least_grood_number_l2756_275606


namespace NUMINAMATH_CALUDE_marbles_distribution_l2756_275612

/-- The number of marbles distributed per class -/
def marbles_per_class : ℕ := 37

/-- The number of classes -/
def number_of_classes : ℕ := 23

/-- The number of leftover marbles -/
def leftover_marbles : ℕ := 16

/-- The total number of marbles distributed to students -/
def total_marbles : ℕ := marbles_per_class * number_of_classes + leftover_marbles

theorem marbles_distribution :
  total_marbles = 867 := by sorry

end NUMINAMATH_CALUDE_marbles_distribution_l2756_275612


namespace NUMINAMATH_CALUDE_triangle_angle_B_l2756_275611

theorem triangle_angle_B (a b c : ℝ) (A B C : ℝ) :
  c = 2 →
  b = 2 * Real.sqrt 3 →
  C = 30 * π / 180 →
  (B = 60 * π / 180 ∨ B = 120 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l2756_275611


namespace NUMINAMATH_CALUDE_tax_reduction_problem_l2756_275679

theorem tax_reduction_problem (T C : ℝ) (X : ℝ) 
  (h1 : X > 0 ∧ X < 100) -- Ensure X is a valid percentage
  (h2 : T > 0 ∧ C > 0)   -- Ensure initial tax and consumption are positive
  (h3 : T * (1 - X / 100) * C * 1.25 = 0.75 * T * C) -- Revenue equation
  : X = 40 := by
  sorry

end NUMINAMATH_CALUDE_tax_reduction_problem_l2756_275679


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_sum_l2756_275681

-- Define the conditions
def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  ∃ d : ℝ, b - a = c - b ∧ b - a = d ∧ d ≠ 0

def is_geometric_sequence (c a b : ℝ) : Prop :=
  ∃ r : ℝ, a / c = b / a ∧ a / c = r ∧ r ≠ 1

-- State the theorem
theorem arithmetic_geometric_sequence_sum (a b c : ℝ) :
  a ≠ b ∧ b ≠ c ∧ c ≠ a →
  is_arithmetic_sequence a b c →
  is_geometric_sequence c a b →
  a + 3*b + c = 10 →
  a = -4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_sum_l2756_275681


namespace NUMINAMATH_CALUDE_waiter_tips_fraction_l2756_275634

theorem waiter_tips_fraction (salary : ℝ) (tips : ℝ) (income : ℝ) : 
  tips = (5/3) * salary → 
  income = salary + tips → 
  tips / income = 5/8 := by
sorry

end NUMINAMATH_CALUDE_waiter_tips_fraction_l2756_275634


namespace NUMINAMATH_CALUDE_at_op_four_six_l2756_275605

-- Define the @ operation
def at_op (a b : ℤ) : ℤ := 2 * a^2 - 2 * b^2

-- Theorem statement
theorem at_op_four_six : at_op 4 6 = -40 := by sorry

end NUMINAMATH_CALUDE_at_op_four_six_l2756_275605


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2756_275674

theorem min_value_of_expression (a b c : ℤ) (h1 : a > b) (h2 : b > c) :
  let x := (a + b + c) / (a - b - c)
  (x + 1 / x : ℚ) ≥ 2 ∧ ∃ (a' b' c' : ℤ), a' > b' ∧ b' > c' ∧
    let x' := (a' + b' + c' : ℚ) / (a' - b' - c' : ℚ)
    x' + 1 / x' = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2756_275674


namespace NUMINAMATH_CALUDE_line_circle_intersection_l2756_275604

/-- A line in 2D space -/
structure Line where
  k : ℝ
  b : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a line and a circle intersect -/
def intersect (l : Line) (c : Circle) : Prop :=
  ∃ x y : ℝ, y = l.k * x + l.b ∧ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem line_circle_intersection (k : ℝ) :
  (∀ l : Line, l.b = 1 → intersect l (Circle.mk (0, 1) 1)) ∧
  (∃ l : Line, l.b ≠ 1 ∧ intersect l (Circle.mk (0, 1) 1)) :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l2756_275604


namespace NUMINAMATH_CALUDE_zero_in_interval_l2756_275646

def f (a x : ℝ) : ℝ := 3 * a * x - 1 - 2 * a

theorem zero_in_interval (a : ℝ) : 
  (∃ x ∈ Set.Ioo (-1) 1, f a x = 0) → 
  (a < -1/5 ∨ a > 1) := by
sorry

end NUMINAMATH_CALUDE_zero_in_interval_l2756_275646


namespace NUMINAMATH_CALUDE_checkerboard_coverage_unsolvable_boards_l2756_275655

/-- Represents a checkerboard -/
structure Checkerboard :=
  (rows : ℕ)
  (cols : ℕ)
  (removed_squares : ℕ)

/-- Determines if a checkerboard can be completely covered by dominoes -/
def can_cover (board : Checkerboard) : Prop :=
  (board.rows * board.cols - board.removed_squares) % 2 = 0

theorem checkerboard_coverage (board : Checkerboard) :
  can_cover board ↔ (board.rows * board.cols - board.removed_squares) % 2 = 0 := by
  sorry

/-- 5x7 board -/
def board_5x7 : Checkerboard := ⟨5, 7, 0⟩

/-- 7x3 board with two removed squares -/
def board_7x3_modified : Checkerboard := ⟨7, 3, 2⟩

theorem unsolvable_boards :
  ¬(can_cover board_5x7) ∧ ¬(can_cover board_7x3_modified) := by
  sorry

end NUMINAMATH_CALUDE_checkerboard_coverage_unsolvable_boards_l2756_275655


namespace NUMINAMATH_CALUDE_correct_new_balance_l2756_275630

/-- Calculates the new credit card balance after transactions -/
def new_balance (initial_balance groceries_expense towels_return : ℚ) : ℚ :=
  initial_balance + groceries_expense + (groceries_expense / 2) - towels_return

/-- Proves that the new balance is correct given the specified transactions -/
theorem correct_new_balance :
  new_balance 126 60 45 = 171 := by
  sorry

end NUMINAMATH_CALUDE_correct_new_balance_l2756_275630


namespace NUMINAMATH_CALUDE_cosine_inequality_solution_l2756_275638

theorem cosine_inequality_solution (x : ℝ) : 
  0 ≤ x ∧ x ≤ 2 * π →
  (2 * Real.cos x ≤ |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ∧
   |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ≤ Real.sqrt 2) ↔
  (π / 4 ≤ x ∧ x ≤ 7 * π / 4) :=
by sorry

end NUMINAMATH_CALUDE_cosine_inequality_solution_l2756_275638


namespace NUMINAMATH_CALUDE_log_equation_solution_set_l2756_275628

theorem log_equation_solution_set :
  let S : Set ℝ := {x | ∃ k : ℤ, x = 2 * k * Real.pi + 5 * Real.pi / 6}
  ∀ x : ℝ, (Real.log (Real.sqrt 3 * Real.sin x) = Real.log (-Real.cos x)) ↔ x ∈ S :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_set_l2756_275628


namespace NUMINAMATH_CALUDE_impossibility_of_measuring_one_liter_l2756_275620

theorem impossibility_of_measuring_one_liter :
  ¬ ∃ (k l : ℤ), k * Real.sqrt 2 + l * (2 - Real.sqrt 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_impossibility_of_measuring_one_liter_l2756_275620


namespace NUMINAMATH_CALUDE_school_referendum_non_voters_l2756_275631

theorem school_referendum_non_voters (total : ℝ) (yes_votes : ℝ) (no_votes : ℝ)
  (h1 : yes_votes = (3 / 5) * total)
  (h2 : no_votes = 0.28 * total)
  (h3 : total > 0) :
  (total - (yes_votes + no_votes)) / total = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_school_referendum_non_voters_l2756_275631


namespace NUMINAMATH_CALUDE_second_day_travel_l2756_275675

/-- Represents the distance traveled on the second day -/
def second_day_distance : ℝ := 420

/-- Represents the distance traveled on the first day -/
def first_day_distance : ℝ := 240

/-- Represents the average speed on both days -/
def average_speed : ℝ := 60

/-- Represents the time difference between the two trips -/
def time_difference : ℝ := 3

/-- Theorem stating that the distance traveled on the second day is 420 miles -/
theorem second_day_travel :
  second_day_distance = first_day_distance + average_speed * time_difference :=
by sorry

end NUMINAMATH_CALUDE_second_day_travel_l2756_275675


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l2756_275665

theorem quadratic_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ m * x^2 + 2*x + 1 = 0 ∧ m * y^2 + 2*y + 1 = 0) ↔ 
  (m ≤ 1 ∧ m ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l2756_275665


namespace NUMINAMATH_CALUDE_word_to_number_correct_l2756_275609

def word_to_number (s : String) : ℝ :=
  match s with
  | "fifty point zero zero one" => 50.001
  | "seventy-five point zero six" => 75.06
  | _ => 0  -- Default case for other inputs

theorem word_to_number_correct :
  (word_to_number "fifty point zero zero one" = 50.001) ∧
  (word_to_number "seventy-five point zero six" = 75.06) := by
  sorry

end NUMINAMATH_CALUDE_word_to_number_correct_l2756_275609


namespace NUMINAMATH_CALUDE_complement_B_intersection_A_complement_B_l2756_275668

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x < 0}

-- Define set B
def B : Set ℝ := {x | x > 1}

-- Theorem for the complement of B with respect to U
theorem complement_B : Set.compl B = {x : ℝ | x ≤ 1} := by sorry

-- Theorem for the intersection of A and the complement of B
theorem intersection_A_complement_B : A ∩ Set.compl B = {x : ℝ | x < 0} := by sorry

end NUMINAMATH_CALUDE_complement_B_intersection_A_complement_B_l2756_275668


namespace NUMINAMATH_CALUDE_max_tuesdays_in_63_days_l2756_275662

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of days we are considering -/
def total_days : ℕ := 63

/-- Each week has one Tuesday -/
axiom one_tuesday_per_week : ℕ

/-- The maximum number of Tuesdays in the first 63 days of a year -/
def max_tuesdays : ℕ := total_days / days_in_week

theorem max_tuesdays_in_63_days : max_tuesdays = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_tuesdays_in_63_days_l2756_275662


namespace NUMINAMATH_CALUDE_library_wall_length_proof_l2756_275614

/-- The length of the library wall given the specified conditions -/
def library_wall_length : ℝ := 8

/-- Represents the number of desks (which is equal to the number of bookcases) -/
def num_furniture : ℕ := 2

theorem library_wall_length_proof :
  (∃ n : ℕ, 
    n = num_furniture ∧ 
    2 * n + 1.5 * n + 1 = library_wall_length ∧
    ∀ m : ℕ, m > n → 2 * m + 1.5 * m + 1 > library_wall_length) := by
  sorry

#check library_wall_length_proof

end NUMINAMATH_CALUDE_library_wall_length_proof_l2756_275614


namespace NUMINAMATH_CALUDE_rectangles_form_square_l2756_275676

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- The set of given rectangles -/
def rectangles : List Rectangle := [
  ⟨1, 2⟩, ⟨7, 10⟩, ⟨6, 5⟩, ⟨8, 12⟩, ⟨9, 3⟩
]

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.width * r.height

/-- Theorem: The given rectangles can form a square -/
theorem rectangles_form_square : ∃ (s : ℕ), s > 0 ∧ s * s = (rectangles.map area).sum := by
  sorry

end NUMINAMATH_CALUDE_rectangles_form_square_l2756_275676


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l2756_275613

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 4 * x + y = 11) 
  (eq2 : x + 4 * y = 15) : 
  13 * x^2 + 14 * x * y + 13 * y^2 = 275.2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l2756_275613


namespace NUMINAMATH_CALUDE_nickel_count_proof_l2756_275694

/-- Represents the number of nickels in a collection of coins -/
def number_of_nickels (total_value : ℚ) (total_coins : ℕ) : ℕ :=
  2

/-- The value of a dime in dollars -/
def dime_value : ℚ := 1/10

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 1/20

theorem nickel_count_proof (total_value : ℚ) (total_coins : ℕ) 
  (h1 : total_value = 7/10) 
  (h2 : total_coins = 8) :
  number_of_nickels total_value total_coins = 2 ∧ 
  ∃ (d n : ℕ), d + n = total_coins ∧ 
               d * dime_value + n * nickel_value = total_value :=
by
  sorry

#check nickel_count_proof

end NUMINAMATH_CALUDE_nickel_count_proof_l2756_275694


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2756_275698

/-- The number of games in a chess tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1)

/-- The number of players in the tournament -/
def num_players : ℕ := 20

/-- Each game is played twice -/
def games_per_pair : ℕ := 2

theorem chess_tournament_games :
  num_games num_players * games_per_pair = 760 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l2756_275698


namespace NUMINAMATH_CALUDE_floor_area_closest_to_160000_l2756_275696

def hand_length : ℝ := 20

def floor_width (hl : ℝ) : ℝ := 18 * hl
def floor_length (hl : ℝ) : ℝ := 22 * hl

def floor_area (w l : ℝ) : ℝ := w * l

def closest_area : ℝ := 160000

theorem floor_area_closest_to_160000 :
  ∀ (ε : ℝ), ε > 0 →
  |floor_area (floor_width hand_length) (floor_length hand_length) - closest_area| < ε →
  ∀ (other_area : ℝ), other_area ≠ closest_area →
  |floor_area (floor_width hand_length) (floor_length hand_length) - other_area| ≥ ε :=
by sorry

end NUMINAMATH_CALUDE_floor_area_closest_to_160000_l2756_275696


namespace NUMINAMATH_CALUDE_single_elimination_games_l2756_275692

/-- A single-elimination tournament with no ties. -/
structure Tournament :=
  (num_teams : ℕ)
  (no_ties : Bool)

/-- The number of games played in a single-elimination tournament. -/
def games_played (t : Tournament) : ℕ :=
  t.num_teams - 1

theorem single_elimination_games (t : Tournament) 
  (h1 : t.num_teams = 23) 
  (h2 : t.no_ties = true) : 
  games_played t = 22 := by
  sorry

end NUMINAMATH_CALUDE_single_elimination_games_l2756_275692


namespace NUMINAMATH_CALUDE_larger_integer_problem_l2756_275635

theorem larger_integer_problem (x y : ℤ) (h1 : x + y = -9) (h2 : x - y = 1) : x = -4 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_problem_l2756_275635


namespace NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l2756_275642

theorem quadratic_polynomial_with_complex_root :
  ∃ (a b c : ℝ), 
    (a = 3) ∧ 
    (∀ (x : ℂ), a * x^2 + b * x + c = 0 ↔ x = 4 + I ∨ x = 4 - I) ∧
    (a * (4 + I)^2 + b * (4 + I) + c = 0) ∧
    (a = 3 ∧ b = -24 ∧ c = 51) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l2756_275642


namespace NUMINAMATH_CALUDE_a_work_time_l2756_275670

-- Define the work rates for A, B, and C
def work_rate_A : ℚ := 1 / 4
def work_rate_B : ℚ := 1 / 12
def work_rate_C : ℚ := 1 / 4

-- Define the theorem
theorem a_work_time : 
  work_rate_A + work_rate_C = 1 / 2 ∧ 
  work_rate_B + work_rate_C = 1 / 3 ∧ 
  work_rate_B = 1 / 12 →
  1 / work_rate_A = 4 := by
  sorry


end NUMINAMATH_CALUDE_a_work_time_l2756_275670


namespace NUMINAMATH_CALUDE_least_positive_integer_to_make_multiple_of_five_l2756_275610

theorem least_positive_integer_to_make_multiple_of_five (n : ℕ) : 
  (∃ k : ℕ, (789 + n) = 5 * k) ∧ (∀ m : ℕ, m < n → ¬∃ k : ℕ, (789 + m) = 5 * k) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_to_make_multiple_of_five_l2756_275610


namespace NUMINAMATH_CALUDE_fourth_root_sum_of_fourth_powers_l2756_275608

/-- Given segments a and b, there exists a segment x such that x^4 = a^4 + b^4 -/
theorem fourth_root_sum_of_fourth_powers (a b : ℝ) : ∃ x : ℝ, x^4 = a^4 + b^4 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_sum_of_fourth_powers_l2756_275608


namespace NUMINAMATH_CALUDE_triangle_inequality_l2756_275647

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : (a^2 + b^2 + c^2)^2 > 2*(a^4 + b^4 + c^4)) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2756_275647


namespace NUMINAMATH_CALUDE_equation_solution_l2756_275688

theorem equation_solution : ∃! x : ℚ, (9 - x)^2 = (x + 1/2)^2 ∧ x = 323/76 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2756_275688


namespace NUMINAMATH_CALUDE_symbol_values_l2756_275637

theorem symbol_values (triangle star : ℤ) 
  (eq1 : 3 * triangle + 2 * star = 14)
  (eq2 : 2 * star + 5 * triangle = 18) : 
  triangle = 2 ∧ star = 4 := by
sorry

end NUMINAMATH_CALUDE_symbol_values_l2756_275637


namespace NUMINAMATH_CALUDE_regression_line_equation_specific_regression_line_equation_l2756_275621

/-- The regression line equation given the y-intercept and a point it passes through -/
theorem regression_line_equation (a : ℝ) (x_center y_center : ℝ) :
  let b := (y_center - a) / x_center
  (∀ x y, y = b * x + a) → (y_center = b * x_center + a) :=
by
  sorry

/-- The specific regression line equation for the given problem -/
theorem specific_regression_line_equation :
  let a := 0.2
  let x_center := 4
  let y_center := 5
  let b := (y_center - a) / x_center
  (∀ x y, y = b * x + a) ∧ (y_center = b * x_center + a) ∧ (b = 1.2) :=
by
  sorry

end NUMINAMATH_CALUDE_regression_line_equation_specific_regression_line_equation_l2756_275621


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2756_275682

def M : Set ℝ := {x | (x - 1) * (x - 2) > 0}
def N : Set ℝ := {x | x^2 + x < 0}

theorem necessary_but_not_sufficient :
  (∀ x : ℝ, x ∈ N → x ∈ M) ∧
  (∃ x : ℝ, x ∈ M ∧ x ∉ N) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2756_275682


namespace NUMINAMATH_CALUDE_smallest_integer_fraction_l2756_275624

theorem smallest_integer_fraction (y : ℤ) : (7 : ℚ) / 11 < (y : ℚ) / 17 ↔ 11 ≤ y := by sorry

end NUMINAMATH_CALUDE_smallest_integer_fraction_l2756_275624


namespace NUMINAMATH_CALUDE_smallest_n_for_monochromatic_subgraph_l2756_275661

/-- A simple graph with 10 vertices and n edges, where edges are colored in two colors -/
structure ColoredGraph (n : ℕ) :=
  (edges : Fin n → Fin 10 × Fin 10)
  (color : Fin n → Bool)
  (simple : ∀ i : Fin n, (edges i).1 ≠ (edges i).2)

/-- A monochromatic triangle in a colored graph -/
def has_monochromatic_triangle (G : ColoredGraph n) : Prop :=
  ∃ (i j k : Fin n), 
    G.edges i ≠ G.edges j ∧ G.edges i ≠ G.edges k ∧ G.edges j ≠ G.edges k ∧
    G.color i = G.color j ∧ G.color j = G.color k

/-- A monochromatic quadrilateral in a colored graph -/
def has_monochromatic_quadrilateral (G : ColoredGraph n) : Prop :=
  ∃ (i j k l : Fin n), 
    G.edges i ≠ G.edges j ∧ G.edges i ≠ G.edges k ∧ G.edges i ≠ G.edges l ∧ 
    G.edges j ≠ G.edges k ∧ G.edges j ≠ G.edges l ∧ G.edges k ≠ G.edges l ∧
    G.color i = G.color j ∧ G.color j = G.color k ∧ G.color k = G.color l

/-- The main theorem -/
theorem smallest_n_for_monochromatic_subgraph : 
  (∀ G : ColoredGraph 31, has_monochromatic_triangle G ∨ has_monochromatic_quadrilateral G) ∧
  (∃ G : ColoredGraph 30, ¬(has_monochromatic_triangle G ∨ has_monochromatic_quadrilateral G)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_monochromatic_subgraph_l2756_275661


namespace NUMINAMATH_CALUDE_divisibility_problem_l2756_275677

theorem divisibility_problem (d r : ℤ) : 
  d > 1 ∧ 
  (∃ k₁ k₂ k₃ k₄ : ℤ, 
    29 * 11 = k₁ * d + r ∧
    1059 = k₂ * d + r ∧
    1417 = k₃ * d + r ∧
    2312 = k₄ * d + r) →
  d - r = 15 := by
sorry

end NUMINAMATH_CALUDE_divisibility_problem_l2756_275677


namespace NUMINAMATH_CALUDE_sum_of_averages_equals_155_l2756_275654

def even_integers_to_100 : List ℕ := List.range 51 |> List.map (· * 2)
def even_integers_to_50 : List ℕ := List.range 26 |> List.map (· * 2)
def even_perfect_squares_to_250 : List ℕ := [0, 4, 16, 36, 64, 100, 144, 196]

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem sum_of_averages_equals_155 :
  average even_integers_to_100 +
  average even_integers_to_50 +
  average even_perfect_squares_to_250 = 155 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_averages_equals_155_l2756_275654


namespace NUMINAMATH_CALUDE_power_two_geq_double_plus_two_l2756_275625

theorem power_two_geq_double_plus_two (n : ℕ) (h : n ≥ 3) : 2^n ≥ 2*(n+1) := by
  sorry

end NUMINAMATH_CALUDE_power_two_geq_double_plus_two_l2756_275625
