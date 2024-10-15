import Mathlib

namespace NUMINAMATH_CALUDE_circle_radius_from_area_l2846_284603

theorem circle_radius_from_area (A : ℝ) (r : ℝ) (h : A = 121 * Real.pi) :
  A = Real.pi * r^2 → r = 11 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_l2846_284603


namespace NUMINAMATH_CALUDE_time_to_reach_ticket_window_l2846_284606

-- Define Kit's movement and remaining distance
def initial_distance : ℝ := 90 -- feet
def initial_time : ℝ := 30 -- minutes
def remaining_distance : ℝ := 100 -- yards

-- Define conversion factor
def yards_to_feet : ℝ := 3 -- feet per yard

-- Theorem to prove
theorem time_to_reach_ticket_window : 
  (remaining_distance * yards_to_feet) / (initial_distance / initial_time) = 100 := by
  sorry

end NUMINAMATH_CALUDE_time_to_reach_ticket_window_l2846_284606


namespace NUMINAMATH_CALUDE_min_sum_absolute_values_l2846_284686

theorem min_sum_absolute_values :
  ∀ x : ℝ, |x + 3| + |x + 5| + |x + 6| ≥ 5 ∧ ∃ x : ℝ, |x + 3| + |x + 5| + |x + 6| = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_absolute_values_l2846_284686


namespace NUMINAMATH_CALUDE_unique_prime_with_prime_sums_l2846_284688

theorem unique_prime_with_prime_sums : ∃! p : ℕ, 
  Nat.Prime p ∧ Nat.Prime (p + 10) ∧ Nat.Prime (p + 14) := by sorry

end NUMINAMATH_CALUDE_unique_prime_with_prime_sums_l2846_284688


namespace NUMINAMATH_CALUDE_integral_of_f_minus_x_l2846_284638

/-- Given a function f: ℝ → ℝ such that f'(x) = 2x + 1 for all x ∈ ℝ,
    prove that the definite integral of f(-x) from -1 to 3 equals 14/3. -/
theorem integral_of_f_minus_x (f : ℝ → ℝ) (h : ∀ x, deriv f x = 2 * x + 1) :
  ∫ x in (-1)..(3), f (-x) = 14/3 := by
  sorry

end NUMINAMATH_CALUDE_integral_of_f_minus_x_l2846_284638


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l2846_284662

def M (p : ℝ) := {x : ℝ | x^2 - p*x + 6 = 0}
def N (q : ℝ) := {x : ℝ | x^2 + 6*x - q = 0}

theorem intersection_implies_sum (p q : ℝ) :
  M p ∩ N q = {2} → p + q = 21 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l2846_284662


namespace NUMINAMATH_CALUDE_g_composition_value_l2846_284655

def g (y : ℝ) : ℝ := y^3 - 3*y + 1

theorem g_composition_value : g (g (g (-1))) = 6803 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_value_l2846_284655


namespace NUMINAMATH_CALUDE_f_intersects_positive_y_axis_l2846_284626

-- Define the function f(x) = x + 1
def f (x : ℝ) : ℝ := x + 1

-- Theorem stating that f intersects the y-axis at a point with positive y-coordinate
theorem f_intersects_positive_y_axis : ∃ (y : ℝ), y > 0 ∧ f 0 = y := by
  sorry

end NUMINAMATH_CALUDE_f_intersects_positive_y_axis_l2846_284626


namespace NUMINAMATH_CALUDE_tangent_line_distance_l2846_284619

/-- The curve function -/
def f (x : ℝ) : ℝ := -x^3 + 2*x

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := -3*x^2 + 2

/-- The x-coordinate of the tangent point -/
def x₀ : ℝ := -1

/-- The y-coordinate of the tangent point -/
def y₀ : ℝ := f x₀

/-- The slope of the tangent line -/
def m : ℝ := f' x₀

/-- The point we're measuring the distance from -/
def P : ℝ × ℝ := (3, 2)

theorem tangent_line_distance :
  let A : ℝ := 1
  let B : ℝ := 1
  let C : ℝ := -(m * x₀ - y₀)
  (A * P.1 + B * P.2 + C) / Real.sqrt (A^2 + B^2) = 7 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_distance_l2846_284619


namespace NUMINAMATH_CALUDE_sequence_negative_term_l2846_284656

theorem sequence_negative_term
  (k : ℝ) (h_k : 0 < k ∧ k < 1)
  (a : ℕ → ℝ)
  (h_a : ∀ n : ℕ, n ≥ 1 → a (n + 1) ≤ (1 + k / n) * a n - 1) :
  ∃ t : ℕ, a t < 0 := by
sorry

end NUMINAMATH_CALUDE_sequence_negative_term_l2846_284656


namespace NUMINAMATH_CALUDE_set_A_equals_zero_to_three_l2846_284611

def A : Set ℤ := {x | x^2 - 3*x - 4 < 0}

theorem set_A_equals_zero_to_three : A = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_set_A_equals_zero_to_three_l2846_284611


namespace NUMINAMATH_CALUDE_smallest_number_in_sequence_l2846_284646

theorem smallest_number_in_sequence (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive integers
  (a + b + c) / 3 = 30 →  -- arithmetic mean is 30
  b = 28 →  -- median is 28
  c = b + 6 →  -- largest number is 6 more than median
  a ≤ b ∧ a ≤ c →  -- a is the smallest number
  a = 28 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_in_sequence_l2846_284646


namespace NUMINAMATH_CALUDE_nine_not_in_remaining_sums_l2846_284631

/-- Represents a cube with numbered faces -/
structure NumberedCube where
  faces : Fin 6 → Nat
  face_values : ∀ i, faces i ∈ Finset.range 7
  distinct_faces : ∀ i j, i ≠ j → faces i ≠ faces j
  opposite_pair_sum_11 : ∃ i j, i ≠ j ∧ faces i + faces j = 11

/-- The sum of the remaining pairs of opposite faces -/
def remaining_pair_sums (cube : NumberedCube) : Finset Nat :=
  sorry

theorem nine_not_in_remaining_sums (cube : NumberedCube) :
  9 ∉ remaining_pair_sums cube :=
sorry

end NUMINAMATH_CALUDE_nine_not_in_remaining_sums_l2846_284631


namespace NUMINAMATH_CALUDE_seventh_root_of_137858491849_l2846_284605

theorem seventh_root_of_137858491849 : 
  (137858491849 : ℝ) ^ (1/7 : ℝ) = 11 := by sorry

end NUMINAMATH_CALUDE_seventh_root_of_137858491849_l2846_284605


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l2846_284691

/-- Given an equilateral triangle with perimeter 60 and an isosceles triangle with perimeter 50,
    where one side of the equilateral triangle is a side of the isosceles triangle,
    the base of the isosceles triangle is 10 units long. -/
theorem isosceles_triangle_base_length : ℝ → ℝ → ℝ → Prop :=
  fun equilateral_perimeter isosceles_perimeter isosceles_base =>
    equilateral_perimeter = 60 →
    isosceles_perimeter = 50 →
    let equilateral_side := equilateral_perimeter / 3
    let isosceles_side := equilateral_side
    isosceles_perimeter = 2 * isosceles_side + isosceles_base →
    isosceles_base = 10

/-- Proof of the theorem -/
theorem isosceles_triangle_base_length_proof :
  isosceles_triangle_base_length 60 50 10 :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l2846_284691


namespace NUMINAMATH_CALUDE_zeros_in_20_pow_10_eq_11_l2846_284615

/-- The number of zeros in the decimal representation of 20^10 -/
def zeros_in_20_pow_10 : ℕ :=
  let base_20_pow_10 := (20 : ℕ) ^ 10
  let digits := base_20_pow_10.digits 10
  digits.count 0

/-- Theorem stating that the number of zeros in 20^10 is 11 -/
theorem zeros_in_20_pow_10_eq_11 : zeros_in_20_pow_10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_zeros_in_20_pow_10_eq_11_l2846_284615


namespace NUMINAMATH_CALUDE_cosine_increasing_interval_l2846_284661

theorem cosine_increasing_interval (a : Real) : 
  (∀ x₁ x₂ : Real, -π ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ a → Real.cos x₁ < Real.cos x₂) → 
  a ∈ Set.Ioc (-π) 0 := by
sorry

end NUMINAMATH_CALUDE_cosine_increasing_interval_l2846_284661


namespace NUMINAMATH_CALUDE_quadratic_nature_l2846_284601

/-- Given a quadratic function f(x) = ax^2 + bx + b^2 / (3a), prove that:
    1. If a > 0, the graph of y = f(x) has a minimum
    2. If a < 0, the graph of y = f(x) has a maximum -/
theorem quadratic_nature (a b : ℝ) (h : a ≠ 0) :
  let f := fun x : ℝ => a * x^2 + b * x + b^2 / (3 * a)
  (a > 0 → ∃ x₀, ∀ x, f x ≥ f x₀) ∧
  (a < 0 → ∃ x₀, ∀ x, f x ≤ f x₀) :=
sorry

end NUMINAMATH_CALUDE_quadratic_nature_l2846_284601


namespace NUMINAMATH_CALUDE_cube_with_cylindrical_hole_volume_l2846_284639

/-- The volume of a cube with a cylindrical hole -/
theorem cube_with_cylindrical_hole_volume (cube_side : ℝ) (hole_diameter : ℝ) : 
  cube_side = 6 →
  hole_diameter = 3 →
  abs (cube_side ^ 3 - π * (hole_diameter / 2) ^ 2 * cube_side - 173.59) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_cylindrical_hole_volume_l2846_284639


namespace NUMINAMATH_CALUDE_min_value_implies_a_inequality_implies_a_range_function_properties_l2846_284621

-- Define the function f(x)
def f (a x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1: Minimum value of f(x) is 2
theorem min_value_implies_a (a : ℝ) :
  (∀ x, f a x ≥ 2) ∧ (∃ x, f a x = 2) → a = -1 ∨ a = -5 := by sorry

-- Part 2: Inequality holds for x ∈ [0, 1]
theorem inequality_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a x ≤ |5 + x|) → a ∈ Set.Icc (-1) 2 := by sorry

-- Combined theorem
theorem function_properties (a : ℝ) :
  ((∀ x, f a x ≥ 2) ∧ (∃ x, f a x = 2) → a = -1 ∨ a = -5) ∧
  ((∀ x ∈ Set.Icc 0 1, f a x ≤ |5 + x|) → a ∈ Set.Icc (-1) 2) := by sorry

end NUMINAMATH_CALUDE_min_value_implies_a_inequality_implies_a_range_function_properties_l2846_284621


namespace NUMINAMATH_CALUDE_fraction_equality_l2846_284657

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4*x + y) / (x - 4*y) = 3) : 
  (x + 4*y) / (4*x - y) = 9/53 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2846_284657


namespace NUMINAMATH_CALUDE_min_students_all_activities_l2846_284623

theorem min_students_all_activities 
  (total : Nat) 
  (swim : Nat) 
  (cycle : Nat) 
  (tennis : Nat) 
  (h1 : total = 52) 
  (h2 : swim = 30) 
  (h3 : cycle = 35) 
  (h4 : tennis = 42) :
  total - ((total - swim) + (total - cycle) + (total - tennis)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_students_all_activities_l2846_284623


namespace NUMINAMATH_CALUDE_min_intercept_line_l2846_284677

/-- A line that passes through a point and intersects the positive halves of the coordinate axes -/
structure InterceptLine where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : 1 / a + 9 / b = 1

/-- The sum of intercepts of an InterceptLine -/
def sum_of_intercepts (l : InterceptLine) : ℝ := l.a + l.b

/-- The equation of the line with minimum sum of intercepts -/
def min_intercept_line_eq (x y : ℝ) : Prop := 3 * x + y - 12 = 0

theorem min_intercept_line :
  ∃ (l : InterceptLine), ∀ (l' : InterceptLine), 
    sum_of_intercepts l ≤ sum_of_intercepts l' ∧
    min_intercept_line_eq l.a l.b := by sorry

end NUMINAMATH_CALUDE_min_intercept_line_l2846_284677


namespace NUMINAMATH_CALUDE_interest_rate_is_five_percent_l2846_284602

-- Define the principal amount and interest rate
variable (P : ℝ) -- Principal amount
variable (r : ℝ) -- Interest rate (as a decimal)

-- Define the conditions
def condition1 : Prop := P * (1 + 3 * r) = 460
def condition2 : Prop := P * (1 + 8 * r) = 560

-- Theorem to prove
theorem interest_rate_is_five_percent 
  (h1 : condition1 P r) 
  (h2 : condition2 P r) : 
  r = 0.05 := by
  sorry


end NUMINAMATH_CALUDE_interest_rate_is_five_percent_l2846_284602


namespace NUMINAMATH_CALUDE_ashok_subjects_l2846_284679

theorem ashok_subjects (total_average : ℝ) (five_subjects_average : ℝ) (sixth_subject_mark : ℝ) 
  (h1 : total_average = 70)
  (h2 : five_subjects_average = 74)
  (h3 : sixth_subject_mark = 50) :
  ∃ (n : ℕ), n = 6 ∧ n * total_average = 5 * five_subjects_average + sixth_subject_mark :=
by
  sorry

end NUMINAMATH_CALUDE_ashok_subjects_l2846_284679


namespace NUMINAMATH_CALUDE_parallelism_sufficiency_not_necessity_l2846_284628

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m₁ n₁ c₁ m₂ n₂ c₂ : ℝ) : Prop :=
  m₁ * n₂ = m₂ * n₁ ∧ m₁ ≠ 0 ∧ m₂ ≠ 0

/-- The condition for parallelism of the given lines -/
def parallelism_condition (a : ℝ) : Prop :=
  are_parallel 2 a (-2) (a + 1) 1 (-a)

theorem parallelism_sufficiency_not_necessity :
  (∀ a : ℝ, a = 1 → parallelism_condition a) ∧
  ¬(∀ a : ℝ, parallelism_condition a → a = 1) :=
by sorry

end NUMINAMATH_CALUDE_parallelism_sufficiency_not_necessity_l2846_284628


namespace NUMINAMATH_CALUDE_turtle_path_max_entries_l2846_284614

/-- Represents a turtle's path on a square grid -/
structure TurtlePath (n : ℕ) :=
  (grid_size : ℕ := 4*n + 2)
  (start_corner : Bool)
  (visits_all_squares_once : Bool)
  (ends_at_start : Bool)

/-- Represents the number of times a turtle enters a row or column -/
def max_entries (path : TurtlePath n) : ℕ := sorry

/-- Main theorem: There exists a row or column that the turtle enters at least 2n + 2 times -/
theorem turtle_path_max_entries {n : ℕ} (path : TurtlePath n) 
  (h1 : path.start_corner = true)
  (h2 : path.visits_all_squares_once = true)
  (h3 : path.ends_at_start = true) :
  max_entries path ≥ 2*n + 2 :=
sorry

end NUMINAMATH_CALUDE_turtle_path_max_entries_l2846_284614


namespace NUMINAMATH_CALUDE_intersection_and_union_when_a_is_5_intersection_equals_b_iff_a_in_range_l2846_284627

def A : Set ℝ := {x | 1 < x - 1 ∧ x - 1 < 3}
def B (a : ℝ) : Set ℝ := {x | (x - 3) * (x - a) < 0}

theorem intersection_and_union_when_a_is_5 :
  (A ∩ B 5 = {x | 3 < x ∧ x < 4}) ∧
  (A ∪ B 5 = {x | 2 < x ∧ x < 5}) := by sorry

theorem intersection_equals_b_iff_a_in_range :
  ∀ a : ℝ, A ∩ B a = B a ↔ 2 ≤ a ∧ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_a_is_5_intersection_equals_b_iff_a_in_range_l2846_284627


namespace NUMINAMATH_CALUDE_max_rooks_300x300_l2846_284617

/-- Represents a chessboard with side length n -/
structure Chessboard (n : ℕ) where
  size : n > 0

/-- Represents a valid rook placement on a chessboard -/
structure RookPlacement (n : ℕ) where
  board : Chessboard n
  rooks : Finset (ℕ × ℕ)
  valid : ∀ (r1 r2 : ℕ × ℕ), r1 ∈ rooks → r2 ∈ rooks → r1 ≠ r2 →
    (r1.1 = r2.1 ∨ r1.2 = r2.2) → 
    (∀ r3 ∈ rooks, r3 ≠ r1 ∧ r3 ≠ r2 → r3.1 ≠ r1.1 ∧ r3.2 ≠ r1.2 ∧ r3.1 ≠ r2.1 ∧ r3.2 ≠ r2.2)

/-- The maximum number of rooks that can be placed on a 300x300 chessboard
    such that each rook attacks no more than one other rook is 400 -/
theorem max_rooks_300x300 : 
  ∀ (p : RookPlacement 300), Finset.card p.rooks ≤ 400 ∧ 
  ∃ (p' : RookPlacement 300), Finset.card p'.rooks = 400 := by
  sorry

end NUMINAMATH_CALUDE_max_rooks_300x300_l2846_284617


namespace NUMINAMATH_CALUDE_largest_quantity_l2846_284699

theorem largest_quantity : 
  let A := (2010 : ℚ) / 2009 + 2010 / 2011
  let B := (2010 : ℚ) / 2011 + 2012 / 2011
  let C := (2011 : ℚ) / 2010 + 2011 / 2012
  A > B ∧ A > C := by sorry

end NUMINAMATH_CALUDE_largest_quantity_l2846_284699


namespace NUMINAMATH_CALUDE_barbells_bought_l2846_284632

theorem barbells_bought (amount_given : ℕ) (change_received : ℕ) (cost_per_barbell : ℕ) : 
  amount_given = 850 → change_received = 40 → cost_per_barbell = 270 → 
  (amount_given - change_received) / cost_per_barbell = 3 :=
by sorry

end NUMINAMATH_CALUDE_barbells_bought_l2846_284632


namespace NUMINAMATH_CALUDE_largest_root_is_six_l2846_284635

/-- Polynomial P(x) = x^6 - 15x^5 + 74x^4 - 130x^3 + ax^2 + bx -/
def P (a b : ℝ) (x : ℝ) : ℝ := x^6 - 15*x^5 + 74*x^4 - 130*x^3 + a*x^2 + b*x

/-- Line L(x) = cx - 24 -/
def L (c : ℝ) (x : ℝ) : ℝ := c*x - 24

/-- The difference between P(x) and L(x) -/
def D (a b c : ℝ) (x : ℝ) : ℝ := P a b x - L c x

theorem largest_root_is_six (a b c : ℝ) : 
  (∃ p q : ℝ, (∀ x : ℝ, D a b c x = (x - p)^3 * (x - q)^2)) →
  (∀ x : ℝ, D a b c x ≥ 0) →
  (∃ x₁ x₂ x₃ : ℝ, D a b c x₁ = 0 ∧ D a b c x₂ = 0 ∧ D a b c x₃ = 0 ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) →
  (∃ x : ℝ, D a b c x = 0 ∧ x = 6 ∧ ∀ y : ℝ, D a b c y = 0 → y ≤ x) :=
sorry

end NUMINAMATH_CALUDE_largest_root_is_six_l2846_284635


namespace NUMINAMATH_CALUDE_triangle_base_increase_l2846_284643

theorem triangle_base_increase (h b : ℝ) (h_pos : h > 0) (b_pos : b > 0) :
  let new_height := 0.95 * h
  let new_area := 1.045 * (0.5 * b * h)
  ∃ x : ℝ, x > 0 ∧ new_area = 0.5 * (b * (1 + x / 100)) * new_height ∧ x = 10 :=
by sorry

end NUMINAMATH_CALUDE_triangle_base_increase_l2846_284643


namespace NUMINAMATH_CALUDE_man_speed_against_current_and_headwind_l2846_284695

/-- The speed of a man rowing in a river with current and headwind -/
def man_speed (downstream_speed current_speed headwind_reduction : ℝ) : ℝ :=
  downstream_speed - current_speed - current_speed - headwind_reduction

/-- Theorem stating the man's speed against current and headwind -/
theorem man_speed_against_current_and_headwind 
  (downstream_speed : ℝ) 
  (current_speed : ℝ) 
  (headwind_reduction : ℝ) 
  (h1 : downstream_speed = 22) 
  (h2 : current_speed = 4.5) 
  (h3 : headwind_reduction = 1.5) : 
  man_speed downstream_speed current_speed headwind_reduction = 11.5 := by
  sorry

#eval man_speed 22 4.5 1.5

end NUMINAMATH_CALUDE_man_speed_against_current_and_headwind_l2846_284695


namespace NUMINAMATH_CALUDE_smallest_cut_length_five_is_smallest_smallest_integral_cut_l2846_284689

theorem smallest_cut_length (x : ℕ) : x ≥ 5 ↔ ¬(9 - x + 14 - x > 18 - x) :=
  sorry

theorem five_is_smallest : ∀ y : ℕ, y < 5 → (9 - y + 14 - y > 18 - y) :=
  sorry

theorem smallest_integral_cut : 
  ∃ x : ℕ, (x ≥ 5) ∧ (∀ y : ℕ, y < x → (9 - y + 14 - y > 18 - y)) :=
  sorry

end NUMINAMATH_CALUDE_smallest_cut_length_five_is_smallest_smallest_integral_cut_l2846_284689


namespace NUMINAMATH_CALUDE_arithmetic_computation_l2846_284613

theorem arithmetic_computation : -7 * 5 - (-4 * 3) + (-9 * -6) = 31 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l2846_284613


namespace NUMINAMATH_CALUDE_process_time_per_picture_l2846_284644

/-- Given a total number of pictures and total processing time in hours,
    calculate the time required to process each picture in minutes. -/
def time_per_picture (total_pictures : ℕ) (total_hours : ℕ) : ℚ :=
  (total_hours * 60) / total_pictures

/-- Theorem: Given 960 pictures and a total processing time of 32 hours,
    the time required to process each picture is 2 minutes. -/
theorem process_time_per_picture :
  time_per_picture 960 32 = 2 := by
  sorry

end NUMINAMATH_CALUDE_process_time_per_picture_l2846_284644


namespace NUMINAMATH_CALUDE_complex_sum_problem_l2846_284642

theorem complex_sum_problem (a b c d g h : ℂ) : 
  b = 4 →
  g = -a - c →
  a + b * I + c + d * I + g + h * I = 3 * I →
  d + h = -1 := by sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l2846_284642


namespace NUMINAMATH_CALUDE_p_iff_a_in_range_exactly_one_true_iff_a_in_range_l2846_284625

-- Define the propositions and conditions
def p (a : ℝ) : Prop := ∃ x : ℝ, -1 < x ∧ x < 1 ∧ x^2 - (a+2)*x + 2*a = 0

def q (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, 
  x₁^2 - 2*m*x₁ - 3 = 0 ∧ 
  x₂^2 - 2*m*x₂ - 3 = 0 ∧ 
  x₁ ≠ x₂

def inequality_holds (a m : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, q m → a^2 - 3*a ≥ |x₁ - x₂|

-- State the theorems
theorem p_iff_a_in_range (a : ℝ) : 
  p a ↔ -1 < a ∧ a < 1 :=
sorry

theorem exactly_one_true_iff_a_in_range (a : ℝ) : 
  (p a ∧ ∀ m : ℝ, -1 ≤ m ∧ m ≤ 1 → ¬(q m ∧ inequality_holds a m)) ∨
  (¬p a ∧ ∃ m : ℝ, -1 ≤ m ∧ m ≤ 1 ∧ q m ∧ inequality_holds a m)
  ↔ 
  a < 1 ∨ a ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_p_iff_a_in_range_exactly_one_true_iff_a_in_range_l2846_284625


namespace NUMINAMATH_CALUDE_short_sleeve_students_l2846_284694

/-- Proves the number of students wearing short sleeves in a class with given conditions -/
theorem short_sleeve_students (total : ℕ) (difference : ℕ) (short : ℕ) (long : ℕ) : 
  total = 36 →
  long - short = difference →
  difference = 24 →
  short + long = total →
  short = 6 := by sorry

end NUMINAMATH_CALUDE_short_sleeve_students_l2846_284694


namespace NUMINAMATH_CALUDE_projection_a_on_b_l2846_284692

def a : ℝ × ℝ := (-1, 3)
def b : ℝ × ℝ := (3, -4)

theorem projection_a_on_b :
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = -3 := by sorry

end NUMINAMATH_CALUDE_projection_a_on_b_l2846_284692


namespace NUMINAMATH_CALUDE_franks_work_days_l2846_284671

/-- Frank's work schedule problem -/
theorem franks_work_days 
  (hours_per_day : ℕ) 
  (total_hours : ℕ) 
  (h1 : hours_per_day = 8) 
  (h2 : total_hours = 32) : 
  total_hours / hours_per_day = 4 := by
  sorry

end NUMINAMATH_CALUDE_franks_work_days_l2846_284671


namespace NUMINAMATH_CALUDE_max_value_of_f_l2846_284672

/-- The quadratic function f(x) = -2x^2 + 4x + 3 -/
def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 3

/-- The maximum value of f(x) for x ∈ ℝ is 5 -/
theorem max_value_of_f :
  ∃ (M : ℝ), M = 5 ∧ ∀ (x : ℝ), f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2846_284672


namespace NUMINAMATH_CALUDE_line_circle_intersection_l2846_284612

/-- Given a circle with radius 5 and a line at distance k from its center,
    if the equation x^2 - kx + 1 = 0 has equal roots,
    then the line intersects the circle. -/
theorem line_circle_intersection (k : ℝ) : 
  (∃ x : ℝ, x^2 - k*x + 1 = 0 ∧ (∀ y : ℝ, y^2 - k*y + 1 = 0 → y = x)) →
  k < 5 :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l2846_284612


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2846_284678

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2 - x| ≥ 1} = {x : ℝ | x ≤ 1 ∨ x ≥ 3} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2846_284678


namespace NUMINAMATH_CALUDE_problem_solution_l2846_284630

theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (eq1 : a * Real.sqrt a + b * Real.sqrt b = 183)
  (eq2 : a * Real.sqrt b + b * Real.sqrt a = 182) :
  9 / 5 * (a + b) = 657 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2846_284630


namespace NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l2846_284634

theorem smallest_b_in_arithmetic_sequence (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- positive terms
  ∃ d : ℝ, a = b - d ∧ c = b + d →  -- arithmetic sequence
  a * b * c = 216 →  -- product condition
  b ≥ 6 ∧ (∀ x : ℝ, x > 0 ∧ (∃ y z : ℝ, y > 0 ∧ z > 0 ∧ 
    (∃ e : ℝ, y = x - e ∧ z = x + e) ∧ 
    y * x * z = 216) → x ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l2846_284634


namespace NUMINAMATH_CALUDE_limit_example_l2846_284666

/-- The limit of (5x^2 - 4x - 1)/(x - 1) as x approaches 1 is 6 -/
theorem limit_example : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x - 1| → |x - 1| < δ → 
    |(5*x^2 - 4*x - 1)/(x - 1) - 6| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_example_l2846_284666


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l2846_284622

theorem longest_side_of_triangle (x : ℝ) : 
  9 + (x + 5) + (2*x + 3) = 40 →
  max 9 (max (x + 5) (2*x + 3)) = 55/3 :=
by sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l2846_284622


namespace NUMINAMATH_CALUDE_max_power_of_two_product_l2846_284675

open BigOperators

def is_permutation (a : Fin 17 → ℕ) : Prop :=
  ∀ i : Fin 17, ∃ j : Fin 17, a j = i.val + 1

theorem max_power_of_two_product (a : Fin 17 → ℕ) (n : ℕ) 
  (h_perm : is_permutation a) 
  (h_prod : ∏ i : Fin 17, (a i - a (i + 1)) = 2^n) : 
  n ≤ 40 ∧ ∃ a₀ : Fin 17 → ℕ, is_permutation a₀ ∧ ∏ i : Fin 17, (a₀ i - a₀ (i + 1)) = 2^40 :=
sorry

end NUMINAMATH_CALUDE_max_power_of_two_product_l2846_284675


namespace NUMINAMATH_CALUDE_weekend_study_hours_per_day_l2846_284649

-- Define the given conditions
def weekday_study_hours_per_night : ℕ := 2
def weekday_study_nights_per_week : ℕ := 5
def weeks_until_exam : ℕ := 6
def total_study_hours : ℕ := 96

-- Define the number of days in a weekend
def days_per_weekend : ℕ := 2

-- Define the theorem
theorem weekend_study_hours_per_day :
  (total_study_hours - weekday_study_hours_per_night * weekday_study_nights_per_week * weeks_until_exam) / (days_per_weekend * weeks_until_exam) = 3 := by
  sorry

end NUMINAMATH_CALUDE_weekend_study_hours_per_day_l2846_284649


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2846_284653

theorem complex_number_quadrant : 
  let z : ℂ := (2 + Complex.I) * Complex.I
  (z.re < 0 ∧ z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2846_284653


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l2846_284648

/-- Given two rectangles with equal area, where one rectangle measures 5 inches by 24 inches
    and the other is 8 inches long, prove that the width of the second rectangle is 15 inches. -/
theorem equal_area_rectangles (carol_length carol_width jordan_length jordan_width : ℝ) :
  carol_length = 5 →
  carol_width = 24 →
  jordan_length = 8 →
  carol_length * carol_width = jordan_length * jordan_width →
  jordan_width = 15 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l2846_284648


namespace NUMINAMATH_CALUDE_open_spots_difference_is_five_l2846_284650

/-- Represents a parking garage with given characteristics -/
structure ParkingGarage where
  totalLevels : Nat
  spotsPerLevel : Nat
  openSpotsFirstLevel : Nat
  openSpotsSecondLevel : Nat
  openSpotsFourthLevel : Nat
  totalFullSpots : Nat

/-- Calculates the difference between open spots on third and second levels -/
def openSpotsDifference (garage : ParkingGarage) : Int :=
  let totalSpots := garage.totalLevels * garage.spotsPerLevel
  let totalOpenSpots := totalSpots - garage.totalFullSpots
  let openSpotsThirdLevel := totalOpenSpots - garage.openSpotsFirstLevel - garage.openSpotsSecondLevel - garage.openSpotsFourthLevel
  openSpotsThirdLevel - garage.openSpotsSecondLevel

/-- Theorem stating the difference between open spots on third and second levels is 5 -/
theorem open_spots_difference_is_five (garage : ParkingGarage)
  (h1 : garage.totalLevels = 4)
  (h2 : garage.spotsPerLevel = 100)
  (h3 : garage.openSpotsFirstLevel = 58)
  (h4 : garage.openSpotsSecondLevel = garage.openSpotsFirstLevel + 2)
  (h5 : garage.openSpotsFourthLevel = 31)
  (h6 : garage.totalFullSpots = 186) :
  openSpotsDifference garage = 5 := by
  sorry

end NUMINAMATH_CALUDE_open_spots_difference_is_five_l2846_284650


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l2846_284620

theorem volleyball_team_selection (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 16 →
  k = 7 →
  m = 2 →
  (Nat.choose (n - m) k) + (Nat.choose (n - m) (k - m)) = 5434 :=
by sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l2846_284620


namespace NUMINAMATH_CALUDE_greatest_solution_of_equation_l2846_284618

theorem greatest_solution_of_equation (x : Real) : 
  x ∈ Set.Icc 0 (10 * Real.pi) →
  |2 * Real.sin x - 1| + |2 * Real.cos (2 * x) - 1| = 0 →
  x ≤ 61 * Real.pi / 6 ∧ 
  ∃ y ∈ Set.Icc 0 (10 * Real.pi), 
    |2 * Real.sin y - 1| + |2 * Real.cos (2 * y) - 1| = 0 ∧
    y = 61 * Real.pi / 6 :=
by sorry

end NUMINAMATH_CALUDE_greatest_solution_of_equation_l2846_284618


namespace NUMINAMATH_CALUDE_four_fish_guarantee_l2846_284608

/-- A coloring of vertices of a regular polygon -/
def Coloring (n : ℕ) := Fin n → Bool

/-- The number of perpendicular segments with same-colored endpoints for a given diagonal -/
def sameColorSegments (n : ℕ) (c : Coloring n) (d : Fin n) : ℕ :=
  sorry

theorem four_fish_guarantee (c : Coloring 20) :
  ∃ d : Fin 20, sameColorSegments 20 c d ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_four_fish_guarantee_l2846_284608


namespace NUMINAMATH_CALUDE_harry_beach_collection_l2846_284637

/-- The number of items Harry has left after his walk on the beach -/
def items_left (sea_stars seashells snails lost : ℕ) : ℕ :=
  sea_stars + seashells + snails - lost

/-- Theorem stating that Harry has 59 items left after his walk -/
theorem harry_beach_collection : items_left 34 21 29 25 = 59 := by
  sorry

end NUMINAMATH_CALUDE_harry_beach_collection_l2846_284637


namespace NUMINAMATH_CALUDE_tangent_r_values_l2846_284674

-- Define the curves C and C1
def C (x y : ℝ) : Prop := (x - 0)^2 + (y - 2)^2 = 4

def C1 (x y r : ℝ) : Prop := ∃ α, x = 3 + r * Real.cos α ∧ y = -2 + r * Real.sin α

-- Define the tangency condition
def are_tangent (r : ℝ) : Prop :=
  ∃ x y, C x y ∧ C1 x y r

-- Theorem statement
theorem tangent_r_values :
  ∀ r : ℝ, are_tangent r ↔ r = 3 ∨ r = -3 ∨ r = 7 ∨ r = -7 :=
sorry

end NUMINAMATH_CALUDE_tangent_r_values_l2846_284674


namespace NUMINAMATH_CALUDE_pauls_cousin_score_l2846_284636

theorem pauls_cousin_score (paul_score : ℕ) (total_score : ℕ) 
  (h1 : paul_score = 3103)
  (h2 : total_score = 5816) :
  total_score - paul_score = 2713 :=
by sorry

end NUMINAMATH_CALUDE_pauls_cousin_score_l2846_284636


namespace NUMINAMATH_CALUDE_polynomial_equality_l2846_284664

theorem polynomial_equality (a b c d : ℝ) : 
  (∀ x : ℝ, x^4 + 4*x^3 + 3*x^2 + 2*x + 1 = 
    (x+1)^4 + a*(x+1)^3 + b*(x+1)^2 + c*(x+1) + d) → 
  (a = 0 ∧ b = -3 ∧ c = 4 ∧ d = -1) := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2846_284664


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2846_284670

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   x^2 - 2*(m+1)*x + m^2 + 2 = 0 ∧ 
   y^2 - 2*(m+1)*y + m^2 + 2 = 0 ∧ 
   (1/x + 1/y = 1)) → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2846_284670


namespace NUMINAMATH_CALUDE_rtl_grouping_equivalence_l2846_284697

/-- Right-to-left grouping evaluation function -/
noncomputable def rtlEval (a b c d e : ℝ) : ℝ := a / (b * c - (d + e))

/-- Standard algebraic notation representation -/
noncomputable def standardNotation (a b c d e : ℝ) : ℝ := a / (b * c - d - e)

/-- Theorem stating the equivalence of right-to-left grouping and standard notation -/
theorem rtl_grouping_equivalence (a b c d e : ℝ) :
  rtlEval a b c d e = standardNotation a b c d e :=
sorry

end NUMINAMATH_CALUDE_rtl_grouping_equivalence_l2846_284697


namespace NUMINAMATH_CALUDE_sin_15_cos_15_equals_quarter_l2846_284685

theorem sin_15_cos_15_equals_quarter :
  Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_equals_quarter_l2846_284685


namespace NUMINAMATH_CALUDE_find_y_l2846_284651

theorem find_y (x : ℕ) (y : ℕ) (h1 : 2^x - 2^(x-2) = 3 * 2^y) (h2 : x = 12) : y = 10 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l2846_284651


namespace NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l2846_284667

theorem sqrt_mixed_number_simplification :
  Real.sqrt (7 + 9 / 16) = 11 / 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l2846_284667


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2846_284647

theorem sum_of_three_numbers (a b c : ℤ) 
  (h1 : a + b = 31) 
  (h2 : b + c = 47) 
  (h3 : c + a = 54) : 
  a + b + c = 66 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2846_284647


namespace NUMINAMATH_CALUDE_fraction_change_l2846_284659

theorem fraction_change (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (0.6 * x) / (0.4 * y) = 1.5 * (x / y) := by
sorry

end NUMINAMATH_CALUDE_fraction_change_l2846_284659


namespace NUMINAMATH_CALUDE_ball_hit_ground_time_l2846_284616

/-- The time when a ball hits the ground -/
theorem ball_hit_ground_time : ∃ (t : ℚ),
  t = 2313 / 1000 ∧
  0 = -4.9 * t^2 + 7 * t + 10 :=
by sorry

end NUMINAMATH_CALUDE_ball_hit_ground_time_l2846_284616


namespace NUMINAMATH_CALUDE_jacob_number_problem_l2846_284668

theorem jacob_number_problem : 
  ∃! x : ℕ, 10 ≤ x ∧ x < 100 ∧ 
  (∃ a b : ℕ, 4 * x - 8 = 10 * a + b ∧ 
              25 ≤ 10 * b + a ∧ 10 * b + a ≤ 30) ∧
  x = 15 := by
  sorry

end NUMINAMATH_CALUDE_jacob_number_problem_l2846_284668


namespace NUMINAMATH_CALUDE_oil_depth_in_specific_tank_l2846_284641

/-- Represents a horizontally positioned cylindrical tank -/
structure CylindricalTank where
  length : ℝ
  diameter : ℝ

/-- Calculates the depth of oil in a cylindrical tank given the surface area -/
def oilDepth (tank : CylindricalTank) (surfaceArea : ℝ) : ℝ :=
  sorry

theorem oil_depth_in_specific_tank :
  let tank : CylindricalTank := { length := 12, diameter := 8 }
  let surfaceArea : ℝ := 32
  oilDepth tank surfaceArea = 4 := by
  sorry

end NUMINAMATH_CALUDE_oil_depth_in_specific_tank_l2846_284641


namespace NUMINAMATH_CALUDE_modulus_of_z_plus_one_equals_two_l2846_284652

def i : ℂ := Complex.I

theorem modulus_of_z_plus_one_equals_two :
  Complex.abs ((1 - 3 * i) / (1 + i) + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_plus_one_equals_two_l2846_284652


namespace NUMINAMATH_CALUDE_circle_center_center_coordinates_l2846_284600

theorem circle_center (x y : ℝ) : 
  (4 * x^2 - 8 * x + 4 * y^2 + 16 * y + 20 = 0) ↔ 
  ((x - 1)^2 + (y + 2)^2 = 0) :=
sorry

theorem center_coordinates : 
  ∃ (x y : ℝ), (4 * x^2 - 8 * x + 4 * y^2 + 16 * y + 20 = 0) ∧ 
  (x = 1 ∧ y = -2) :=
sorry

end NUMINAMATH_CALUDE_circle_center_center_coordinates_l2846_284600


namespace NUMINAMATH_CALUDE_min_value_theorem_l2846_284690

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  ∀ z : ℝ, z = 1 / x + 4 / y → z ≥ 9 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2846_284690


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_probability_l2846_284665

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculate the area of a triangle -/
def triangleArea (t : Triangle) : ℝ :=
  sorry

/-- Check if a point is inside a triangle -/
def isInside (p : Point) (t : Triangle) : Prop :=
  sorry

/-- Calculate the probability of a point being in a specific region of a triangle -/
def probabilityInRegion (t : Triangle) (condition : Point → Bool) : ℝ :=
  sorry

/-- The main theorem to be proved -/
theorem isosceles_right_triangle_probability :
  let A : Point := ⟨0, 8⟩
  let B : Point := ⟨0, 0⟩
  let C : Point := ⟨8, 0⟩
  let ABC : Triangle := ⟨A, B, C⟩
  probabilityInRegion ABC (fun P => 
    triangleArea ⟨P, B, C⟩ < (1/3) * triangleArea ABC) = 7/32 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_probability_l2846_284665


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_120_l2846_284610

theorem last_three_digits_of_7_to_120 : 7^120 % 1000 = 681 := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_120_l2846_284610


namespace NUMINAMATH_CALUDE_function_composition_property_l2846_284684

theorem function_composition_property (f g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + g y) = 2 * x + y) :
  ∀ x y : ℝ, g (x + f y) = x / 2 + y := by
  sorry

end NUMINAMATH_CALUDE_function_composition_property_l2846_284684


namespace NUMINAMATH_CALUDE_yellow_balls_count_l2846_284629

def total_balls : ℕ := 1500

def red_balls : ℕ := (2 * total_balls) / 7

def remaining_after_red : ℕ := total_balls - red_balls

def blue_balls : ℕ := (3 * remaining_after_red) / 11

def remaining_after_blue : ℕ := remaining_after_red - blue_balls

def green_balls : ℕ := remaining_after_blue / 5

def remaining_after_green : ℕ := remaining_after_blue - green_balls

def orange_balls : ℕ := remaining_after_green / 8

def yellow_balls : ℕ := remaining_after_green - orange_balls

theorem yellow_balls_count : yellow_balls = 546 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l2846_284629


namespace NUMINAMATH_CALUDE_janet_monday_wednesday_hours_l2846_284693

/-- Janet's weekly gym schedule -/
structure GymSchedule where
  total_hours : ℝ
  monday_hours : ℝ
  tuesday_hours : ℝ
  wednesday_hours : ℝ
  friday_hours : ℝ

/-- Janet's gym schedule satisfies the given conditions -/
def janet_schedule (s : GymSchedule) : Prop :=
  s.total_hours = 5 ∧
  s.tuesday_hours = s.friday_hours ∧
  s.friday_hours = 1 ∧
  s.monday_hours = s.wednesday_hours

/-- Theorem: Janet spends 1.5 hours at the gym on Monday and Wednesday each -/
theorem janet_monday_wednesday_hours (s : GymSchedule) 
  (h : janet_schedule s) : s.monday_hours = 1.5 ∧ s.wednesday_hours = 1.5 := by
  sorry


end NUMINAMATH_CALUDE_janet_monday_wednesday_hours_l2846_284693


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l2846_284624

theorem perpendicular_lines_b_value (b : ℝ) : 
  let v1 : Fin 2 → ℝ := ![4, -1]
  let v2 : Fin 2 → ℝ := ![b, 8]
  (∀ i, v1 i * v2 i = 0) → b = 2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l2846_284624


namespace NUMINAMATH_CALUDE_race_speed_ratio_l2846_284683

/-- Proves the speed ratio in a race with given conditions -/
theorem race_speed_ratio (L : ℝ) (h : L > 0) : 
  ∃ R : ℝ, 
    (R > 0) ∧ 
    (0.26 * L = (1 - 0.74) * L) ∧
    (R * L = (1 - 0.60) * L) →
    R = 0.26 := by
  sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l2846_284683


namespace NUMINAMATH_CALUDE_hyogeun_weight_l2846_284676

/-- Given the weights of three people satisfying certain conditions, 
    prove that one person's weight is as specified. -/
theorem hyogeun_weight (H S G : ℝ) : 
  H + S + G = 106.6 →
  G = S - 7.7 →
  S = H - 4.8 →
  H = 41.3 := by
sorry

end NUMINAMATH_CALUDE_hyogeun_weight_l2846_284676


namespace NUMINAMATH_CALUDE_number_problem_l2846_284687

theorem number_problem (x : ℝ) : 42 + 3 * x - 10 = 65 → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2846_284687


namespace NUMINAMATH_CALUDE_quarters_percentage_theorem_l2846_284645

def num_dimes : ℕ := 70
def num_quarters : ℕ := 30
def num_nickels : ℕ := 15

def value_dime : ℕ := 10
def value_quarter : ℕ := 25
def value_nickel : ℕ := 5

def total_value : ℕ := num_dimes * value_dime + num_quarters * value_quarter + num_nickels * value_nickel
def quarters_value : ℕ := num_quarters * value_quarter

theorem quarters_percentage_theorem : 
  (quarters_value : ℚ) / (total_value : ℚ) * 100 = 750 / 1525 * 100 := by
  sorry

end NUMINAMATH_CALUDE_quarters_percentage_theorem_l2846_284645


namespace NUMINAMATH_CALUDE_book_arrangement_count_l2846_284663

def num_arabic : ℕ := 2
def num_german : ℕ := 3
def num_spanish : ℕ := 4
def total_books : ℕ := num_arabic + num_german + num_spanish

def arrangement_count : ℕ := sorry

theorem book_arrangement_count :
  arrangement_count = 3456 := by sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l2846_284663


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2846_284654

theorem fraction_to_decimal : (21 : ℚ) / 40 = 0.525 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2846_284654


namespace NUMINAMATH_CALUDE_digit_105_of_7_19th_l2846_284660

/-- The decimal representation of 7/19 has a repeating cycle of length 18 -/
def decimal_cycle_length : ℕ := 18

/-- The repeating decimal representation of 7/19 -/
def decimal_rep : List ℕ := [3, 6, 8, 4, 2, 1, 0, 5, 2, 6, 3, 1, 5, 7, 8, 9, 4, 7]

/-- The 105th digit after the decimal point in the decimal representation of 7/19 is 7 -/
theorem digit_105_of_7_19th : decimal_rep[(105 - 1) % decimal_cycle_length] = 7 := by
  sorry

end NUMINAMATH_CALUDE_digit_105_of_7_19th_l2846_284660


namespace NUMINAMATH_CALUDE_mystery_book_shelves_l2846_284680

theorem mystery_book_shelves :
  ∀ (books_per_shelf : ℕ) 
    (picture_book_shelves : ℕ) 
    (total_books : ℕ),
  books_per_shelf = 8 →
  picture_book_shelves = 4 →
  total_books = 72 →
  ∃ (mystery_book_shelves : ℕ),
    mystery_book_shelves * books_per_shelf + 
    picture_book_shelves * books_per_shelf = total_books ∧
    mystery_book_shelves = 5 :=
by sorry

end NUMINAMATH_CALUDE_mystery_book_shelves_l2846_284680


namespace NUMINAMATH_CALUDE_square_garden_perimeter_l2846_284607

theorem square_garden_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) :
  area = 200 →
  area = side ^ 2 →
  perimeter = 4 * side →
  perimeter = 40 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_square_garden_perimeter_l2846_284607


namespace NUMINAMATH_CALUDE_two_cos_thirty_degrees_equals_sqrt_three_l2846_284604

theorem two_cos_thirty_degrees_equals_sqrt_three :
  2 * Real.cos (30 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_two_cos_thirty_degrees_equals_sqrt_three_l2846_284604


namespace NUMINAMATH_CALUDE_price_decrease_percentage_l2846_284669

theorem price_decrease_percentage (initial_price : ℝ) (h : initial_price > 0) : 
  let increased_price := initial_price * (1 + 0.25)
  let decrease_percentage := (increased_price - initial_price) / increased_price * 100
  decrease_percentage = 20 := by
sorry

end NUMINAMATH_CALUDE_price_decrease_percentage_l2846_284669


namespace NUMINAMATH_CALUDE_johns_house_nails_l2846_284609

/-- The number of nails needed for a house wall -/
def total_nails (large_planks small_planks large_nails small_nails : ℕ) : ℕ :=
  large_nails + small_nails

/-- Theorem stating the total number of nails needed for John's house wall -/
theorem johns_house_nails :
  total_nails 12 10 15 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_johns_house_nails_l2846_284609


namespace NUMINAMATH_CALUDE_gifts_sent_calculation_l2846_284696

/-- The number of gifts sent to the orphanage given the initial number of gifts and the number of gifts left -/
def gifts_sent_to_orphanage (initial_gifts : ℕ) (gifts_left : ℕ) : ℕ :=
  initial_gifts - gifts_left

/-- Theorem stating that for the given scenario, 66 gifts were sent to the orphanage -/
theorem gifts_sent_calculation :
  gifts_sent_to_orphanage 77 11 = 66 := by
  sorry

end NUMINAMATH_CALUDE_gifts_sent_calculation_l2846_284696


namespace NUMINAMATH_CALUDE_sum_even_integers_100_to_200_l2846_284681

-- Define the sum of the first n positive even integers
def sumFirstNEvenIntegers (n : ℕ) : ℕ := n * (n + 1)

-- Define the sum of even integers from a to b inclusive
def sumEvenIntegersFromTo (a b : ℕ) : ℕ :=
  let n := (b - a) / 2 + 1
  n * (a + b) / 2

-- Theorem statement
theorem sum_even_integers_100_to_200 :
  sumFirstNEvenIntegers 50 = 2550 →
  sumEvenIntegersFromTo 100 200 = 7650 := by
  sorry

end NUMINAMATH_CALUDE_sum_even_integers_100_to_200_l2846_284681


namespace NUMINAMATH_CALUDE_centrifuge_force_scientific_notation_l2846_284658

theorem centrifuge_force_scientific_notation :
  17000 = 1.7 * (10 ^ 4) := by sorry

end NUMINAMATH_CALUDE_centrifuge_force_scientific_notation_l2846_284658


namespace NUMINAMATH_CALUDE_f_monotone_increasing_iff_a_in_range_l2846_284640

open Real

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - (1/3) * sin (2*x) + a * sin x

/-- The derivative of f(x) -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1 - (2/3) * cos (2*x) + a * cos x

/-- Theorem stating the range of 'a' for which f(x) is monotonically increasing -/
theorem f_monotone_increasing_iff_a_in_range :
  ∀ a : ℝ, (∀ x : ℝ, Monotone (f a)) ↔ a ∈ Set.Icc (-1/3) (1/3) :=
by sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_iff_a_in_range_l2846_284640


namespace NUMINAMATH_CALUDE_B_squared_equals_451_l2846_284682

/-- The function g defined as g(x) = √31 + 105/x -/
noncomputable def g (x : ℝ) : ℝ := Real.sqrt 31 + 105 / x

/-- The equation from the problem -/
def problem_equation (x : ℝ) : Prop :=
  x = g (g (g (g (g x))))

/-- The sum of absolute values of roots of the equation -/
noncomputable def B : ℝ :=
  abs ((Real.sqrt 31 + Real.sqrt 451) / 2) +
  abs ((Real.sqrt 31 - Real.sqrt 451) / 2)

/-- Theorem stating that B^2 equals 451 -/
theorem B_squared_equals_451 : B^2 = 451 := by
  sorry

end NUMINAMATH_CALUDE_B_squared_equals_451_l2846_284682


namespace NUMINAMATH_CALUDE_monkeys_eating_bananas_l2846_284698

/-- Given the rate at which monkeys eat bananas, prove that 6 monkeys are needed to eat 18 bananas in 18 minutes -/
theorem monkeys_eating_bananas 
  (initial_monkeys : ℕ) 
  (initial_time : ℕ) 
  (initial_bananas : ℕ) 
  (target_time : ℕ) 
  (target_bananas : ℕ) 
  (h1 : initial_monkeys = 6) 
  (h2 : initial_time = 6) 
  (h3 : initial_bananas = 6) 
  (h4 : target_time = 18) 
  (h5 : target_bananas = 18) : 
  (target_bananas * initial_time * initial_monkeys) / (initial_bananas * target_time) = 6 := by
  sorry

end NUMINAMATH_CALUDE_monkeys_eating_bananas_l2846_284698


namespace NUMINAMATH_CALUDE_count_three_digit_multiples_of_15_l2846_284633

theorem count_three_digit_multiples_of_15 : 
  (Finset.filter (λ x => x % 15 = 0) (Finset.range 900)).card = 60 := by
  sorry

end NUMINAMATH_CALUDE_count_three_digit_multiples_of_15_l2846_284633


namespace NUMINAMATH_CALUDE_carousel_attendance_l2846_284673

/-- The number of children attending a carousel, given:
  * 4 clowns also attend
  * The candy seller initially had 700 candies
  * Each clown and child receives 20 candies
  * The candy seller has 20 candies left after selling
-/
def num_children : ℕ := 30

theorem carousel_attendance : num_children = 30 := by
  sorry

end NUMINAMATH_CALUDE_carousel_attendance_l2846_284673
