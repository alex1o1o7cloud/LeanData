import Mathlib

namespace NUMINAMATH_CALUDE_angle_B_is_60_degrees_l225_22584

-- Define a structure for a triangle with angles A, B, and C
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem angle_B_is_60_degrees (t : Triangle) 
  (h1 : t.B = 2 * t.A)
  (h2 : t.C = 3 * t.A)
  (h3 : t.A + t.B + t.C = 180) : 
  t.B = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_is_60_degrees_l225_22584


namespace NUMINAMATH_CALUDE_perimeter_of_square_C_l225_22564

/-- Given three squares A, B, and C, prove that the perimeter of C is 90 -/
theorem perimeter_of_square_C (a b c : ℝ) : 
  (4 * a = 30) →  -- perimeter of A is 30
  (b = 2 * a) →   -- side of B is twice the side of A
  (c = a + b) →   -- side of C is sum of sides of A and B
  (4 * c = 90) :=  -- perimeter of C is 90
by sorry

end NUMINAMATH_CALUDE_perimeter_of_square_C_l225_22564


namespace NUMINAMATH_CALUDE_problem_solution_l225_22561

theorem problem_solution (n : ℕ) (h1 : n > 30) (h2 : (4 * n - 1) ∣ (2002 * n)) : n = 36 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l225_22561


namespace NUMINAMATH_CALUDE_max_volume_container_l225_22592

/-- Represents the dimensions of a rectangular container --/
structure ContainerDimensions where
  length : Real
  width : Real
  height : Real

/-- Calculates the volume of a rectangular container --/
def volume (d : ContainerDimensions) : Real :=
  d.length * d.width * d.height

/-- Represents the constraint of the total length of the steel bar --/
def totalLength (d : ContainerDimensions) : Real :=
  2 * (d.length + d.width) + 4 * d.height

/-- Theorem stating the maximum volume and corresponding height --/
theorem max_volume_container :
  ∃ (d : ContainerDimensions),
    totalLength d = 14.8 ∧
    d.length = d.width + 0.5 ∧
    d.height = 1.2 ∧
    volume d = 2.2 ∧
    ∀ (d' : ContainerDimensions),
      totalLength d' = 14.8 ∧ d'.length = d'.width + 0.5 →
      volume d' ≤ volume d :=
by sorry

end NUMINAMATH_CALUDE_max_volume_container_l225_22592


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l225_22524

theorem sum_of_reciprocals (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h_sum : x + y + z = 3) :
  1 / (y^2 + z^2 - x^2) + 1 / (x^2 + z^2 - y^2) + 1 / (x^2 + y^2 - z^2) = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l225_22524


namespace NUMINAMATH_CALUDE_wall_length_calculation_l225_22570

theorem wall_length_calculation (mirror_side : ℝ) (wall_width : ℝ) :
  mirror_side = 21 →
  wall_width = 28 →
  (mirror_side ^ 2) * 2 = wall_width * (31.5 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_wall_length_calculation_l225_22570


namespace NUMINAMATH_CALUDE_time_addition_theorem_l225_22544

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds a duration to a given time, wrapping around in 12-hour format -/
def addTime (start : Time) (durationHours durationMinutes durationSeconds : Nat) : Time :=
  sorry

/-- Calculates the sum of hour, minute, and second components of a time -/
def sumComponents (t : Time) : Nat :=
  t.hours + t.minutes + t.seconds

theorem time_addition_theorem :
  let startTime := Time.mk 3 0 0
  let finalTime := addTime startTime 315 58 36
  finalTime = Time.mk 6 58 36 ∧ sumComponents finalTime = 100 := by sorry

end NUMINAMATH_CALUDE_time_addition_theorem_l225_22544


namespace NUMINAMATH_CALUDE_quadratic_substitution_roots_l225_22527

/-- Given a quadratic equation ax^2 + bx + c = 0, this theorem proves the conditions for equal
    product of roots after substitution and the sum of all roots in those cases. -/
theorem quadratic_substitution_roots (a b c : ℝ) (h : a ≠ 0) :
  ∃ k : ℝ, 
    (k = 0 ∨ k = -b/a) ∧ 
    (∀ y : ℝ, c/a = (a*k^2 + b*k + c)/a) ∧
    ((k = 0 → ((-b/a) + (-b/a) = -2*b/a)) ∧ 
     (k = -b/a → ((-b/a) + (b/a) = 0))) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_substitution_roots_l225_22527


namespace NUMINAMATH_CALUDE_g_negative_one_eq_three_l225_22598

/-- A polynomial function of degree 9 -/
noncomputable def g (d e f : ℝ) (x : ℝ) : ℝ := d * x^9 - e * x^5 + f * x + 1

/-- Theorem: If g(1) = -1, then g(-1) = 3 -/
theorem g_negative_one_eq_three {d e f : ℝ} (h : g d e f 1 = -1) : g d e f (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_negative_one_eq_three_l225_22598


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l225_22563

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 : 
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l225_22563


namespace NUMINAMATH_CALUDE_two_players_percentage_of_goals_l225_22585

def total_goals : ℕ := 300
def player_goals : ℕ := 30
def num_players : ℕ := 2

theorem two_players_percentage_of_goals :
  (player_goals * num_players : ℚ) / total_goals * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_two_players_percentage_of_goals_l225_22585


namespace NUMINAMATH_CALUDE_smallest_interesting_rectangle_area_l225_22553

/-- A rectangle is interesting if it has integer side lengths and contains
    exactly four lattice points strictly in its interior. -/
def is_interesting (a b : ℕ) : Prop :=
  (a - 1) * (b - 1) = 4

/-- The area of the smallest interesting rectangle is 10. -/
theorem smallest_interesting_rectangle_area : 
  (∃ a b : ℕ, is_interesting a b ∧ a * b = 10) ∧ 
  (∀ a b : ℕ, is_interesting a b → a * b ≥ 10) :=
sorry

end NUMINAMATH_CALUDE_smallest_interesting_rectangle_area_l225_22553


namespace NUMINAMATH_CALUDE_parabola_reflection_l225_22573

/-- Given a parabola y = x^2 and a line y = x + 2, prove that the reflection of the parabola about the line is x = y^2 - 4y + 2 -/
theorem parabola_reflection (x y : ℝ) :
  (y = x^2) ∧ (∃ (x' y' : ℝ), y' = x' + 2 ∧ 
    ((x' = y - 2 ∧ y' = x + 2) ∨ (x' = x ∧ y' = y))) →
  x = y^2 - 4*y + 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_reflection_l225_22573


namespace NUMINAMATH_CALUDE_b_n_equals_c_1_l225_22501

theorem b_n_equals_c_1 (n : ℕ) (a : ℕ → ℝ) (b c : ℕ → ℝ)
  (h_positive : ∀ i, 1 ≤ i → i ≤ n → 0 < a i)
  (h_b_1 : b 1 = a 1)
  (h_b_2 : b 2 = max (a 1) (a 2))
  (h_b_i : ∀ i, 3 ≤ i → i ≤ n → b i = max (b (i - 1)) (b (i - 2) + a i))
  (h_c_n : c n = a n)
  (h_c_n_1 : c (n - 1) = max (a n) (a (n - 1)))
  (h_c_i : ∀ i, 1 ≤ i → i ≤ n - 2 → c i = max (c (i + 1)) (c (i + 2) + a i)) :
  b n = c 1 := by
  sorry


end NUMINAMATH_CALUDE_b_n_equals_c_1_l225_22501


namespace NUMINAMATH_CALUDE_prob_two_target_rolls_l225_22593

/-- The number of sides on each die -/
def num_sides : ℕ := 7

/-- The sum we're aiming for -/
def target_sum : ℕ := 8

/-- The set of all possible outcomes when rolling two dice -/
def all_outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range num_sides) (Finset.range num_sides)

/-- The set of outcomes that sum to the target -/
def favorable_outcomes : Finset (ℕ × ℕ) :=
  all_outcomes.filter (fun (a, b) => a + b + 2 = target_sum)

/-- The probability of rolling the target sum once -/
def prob_target : ℚ :=
  (favorable_outcomes.card : ℚ) / (all_outcomes.card : ℚ)

theorem prob_two_target_rolls : prob_target * prob_target = 1 / 49 := by
  sorry


end NUMINAMATH_CALUDE_prob_two_target_rolls_l225_22593


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l225_22528

theorem fraction_equals_zero (x : ℝ) :
  (2*x - 4) / (x + 1) = 0 ∧ x + 1 ≠ 0 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l225_22528


namespace NUMINAMATH_CALUDE_total_books_is_42_l225_22540

-- Define the initial number of books on each shelf
def initial_books_shelf1 : ℕ := 9
def initial_books_shelf2 : ℕ := 0
def initial_books_shelf3 : ℕ := initial_books_shelf1 + (initial_books_shelf1 * 3 / 10)
def initial_books_shelf4 : ℕ := initial_books_shelf3 / 2

-- Define the number of books added to each shelf
def added_books_shelf1 : ℕ := 10
def added_books_shelf4 : ℕ := 5

-- Define the total number of books after additions
def total_books : ℕ := 
  (initial_books_shelf1 + added_books_shelf1) +
  initial_books_shelf2 +
  initial_books_shelf3 +
  (initial_books_shelf4 + added_books_shelf4)

-- Theorem statement
theorem total_books_is_42 : total_books = 42 := by
  sorry

end NUMINAMATH_CALUDE_total_books_is_42_l225_22540


namespace NUMINAMATH_CALUDE_digit_sum_property_l225_22569

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_valid_number (A : ℕ) : Prop :=
  10 ≤ A ∧ A ≤ 99 ∧ (sum_of_digits A)^2 = sum_of_digits (A^2)

def solution_set : Finset ℕ := {11, 12, 13, 20, 21, 22, 30, 31, 50}

theorem digit_sum_property :
  ∀ A : ℕ, is_valid_number A ↔ A ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_digit_sum_property_l225_22569


namespace NUMINAMATH_CALUDE_prism_with_21_edges_has_14_vertices_l225_22596

/-- A prism is a polyhedron with two congruent and parallel bases -/
structure Prism where
  base_edges : ℕ
  total_edges : ℕ

/-- The number of edges in a prism is three times the number of edges in its base -/
axiom prism_edge_count (p : Prism) : p.total_edges = 3 * p.base_edges

/-- The number of vertices in a prism is twice the number of edges in its base -/
def prism_vertex_count (p : Prism) : ℕ := 2 * p.base_edges

/-- Theorem: A prism with 21 edges has 14 vertices -/
theorem prism_with_21_edges_has_14_vertices (p : Prism) (h : p.total_edges = 21) : 
  prism_vertex_count p = 14 := by
  sorry

end NUMINAMATH_CALUDE_prism_with_21_edges_has_14_vertices_l225_22596


namespace NUMINAMATH_CALUDE_composite_numbers_equal_if_same_main_divisors_l225_22513

/-- Main divisors of a natural number -/
def main_divisors (n : ℕ) : Set ℕ :=
  {d ∈ Nat.divisors n | d ≠ n ∧ d > 1 ∧ ∀ e ∈ Nat.divisors n, e ≠ n → e ≤ d}

/-- Two largest elements of a finite set of natural numbers -/
def two_largest (s : Set ℕ) : Set ℕ :=
  {x ∈ s | ∀ y ∈ s, y ≤ x ∨ ∃ z ∈ s, z ≠ x ∧ z ≠ y ∧ y ≤ z}

theorem composite_numbers_equal_if_same_main_divisors
  (a b : ℕ) (ha : ¬Nat.Prime a) (hb : ¬Nat.Prime b)
  (h : two_largest (main_divisors a) = two_largest (main_divisors b)) :
  a = b := by
  sorry

end NUMINAMATH_CALUDE_composite_numbers_equal_if_same_main_divisors_l225_22513


namespace NUMINAMATH_CALUDE_inverse_as_linear_combination_l225_22526

def M : Matrix (Fin 2) (Fin 2) ℚ := !![3, 1; 0, 4]

theorem inverse_as_linear_combination :
  ∃ (a b : ℚ), M⁻¹ = a • M + b • (1 : Matrix (Fin 2) (Fin 2) ℚ) ∧ 
  a = -1/12 ∧ b = 7/12 := by
sorry

end NUMINAMATH_CALUDE_inverse_as_linear_combination_l225_22526


namespace NUMINAMATH_CALUDE_circle_equation_l225_22522

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the properties of the circle
def is_tangent_to_y_axis (c : Circle) : Prop :=
  c.center.1 = c.radius

def center_on_line (c : Circle) : Prop :=
  c.center.1 = 3 * c.center.2

def cuts_chord_on_line (c : Circle) (chord_length : ℝ) : Prop :=
  ∃ (p q : ℝ × ℝ),
    p.1 - p.2 = 0 ∧ q.1 - q.2 = 0 ∧
    (p.1 - q.1)^2 + (p.2 - q.2)^2 = chord_length^2 ∧
    (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
    (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 = c.radius^2

-- Theorem statement
theorem circle_equation (c : Circle) :
  is_tangent_to_y_axis c →
  center_on_line c →
  cuts_chord_on_line c (2 * Real.sqrt 7) →
  (∀ x y : ℝ, (x - 3)^2 + (y - 1)^2 = 9 ∨ (x + 3)^2 + (y + 1)^2 = 9 ↔
    (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l225_22522


namespace NUMINAMATH_CALUDE_decrement_calculation_l225_22581

theorem decrement_calculation (n : ℕ) (original_mean updated_mean : ℝ) 
  (h1 : n = 50)
  (h2 : original_mean = 200)
  (h3 : updated_mean = 194) :
  (n : ℝ) * original_mean - n * updated_mean = 6 * n := by
  sorry

end NUMINAMATH_CALUDE_decrement_calculation_l225_22581


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l225_22590

theorem complex_fraction_equality : 
  2013 * (5.7 * 4.2 + (21/5) * 4.3) / ((14/73) * 15 + (5/73) * 177 + 656) = 126 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l225_22590


namespace NUMINAMATH_CALUDE_hierarchy_combinations_l225_22530

def society_size : ℕ := 12
def num_dukes : ℕ := 3
def knights_per_duke : ℕ := 2

def choose_hierarchy : ℕ := 
  society_size * 
  (society_size - 1) * 
  (society_size - 2) * 
  (society_size - 3) * 
  (Nat.choose (society_size - 4) knights_per_duke) * 
  (Nat.choose (society_size - 4 - knights_per_duke) knights_per_duke) * 
  (Nat.choose (society_size - 4 - 2 * knights_per_duke) knights_per_duke)

theorem hierarchy_combinations : 
  choose_hierarchy = 907200 :=
by sorry

end NUMINAMATH_CALUDE_hierarchy_combinations_l225_22530


namespace NUMINAMATH_CALUDE_pizza_slices_left_l225_22591

theorem pizza_slices_left (total_slices : Nat) (john_slices : Nat) (sam_multiplier : Nat) : 
  total_slices = 12 → 
  john_slices = 3 → 
  sam_multiplier = 2 → 
  total_slices - (john_slices + sam_multiplier * john_slices) = 3 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_left_l225_22591


namespace NUMINAMATH_CALUDE_intersection_not_roots_l225_22512

theorem intersection_not_roots : ∀ x : ℝ, 
  (x = x - 3 → x^2 - 3*x ≠ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_not_roots_l225_22512


namespace NUMINAMATH_CALUDE_factor_implies_a_value_l225_22574

theorem factor_implies_a_value (a b : ℝ) :
  (∀ x : ℝ, (x^2 + x - 6) ∣ (2*x^4 + x^3 - a*x^2 + b*x + a + b - 1)) →
  a = 16 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_a_value_l225_22574


namespace NUMINAMATH_CALUDE_cosine_product_equals_half_to_seventh_power_l225_22533

theorem cosine_product_equals_half_to_seventh_power :
  (Real.cos (12 * π / 180)) *
  (Real.cos (24 * π / 180)) *
  (Real.cos (36 * π / 180)) *
  (Real.cos (48 * π / 180)) *
  (Real.cos (60 * π / 180)) *
  (Real.cos (72 * π / 180)) *
  (Real.cos (84 * π / 180)) = (1/2)^7 := by
  sorry

end NUMINAMATH_CALUDE_cosine_product_equals_half_to_seventh_power_l225_22533


namespace NUMINAMATH_CALUDE_unique_two_digit_number_exists_l225_22502

/-- A two-digit number -/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n < 100 }

/-- Get the tens digit of a two-digit number -/
def tens_digit (n : TwoDigitNumber) : ℕ := n.val / 10

/-- Get the units digit of a two-digit number -/
def units_digit (n : TwoDigitNumber) : ℕ := n.val % 10

/-- Reverse the digits of a two-digit number -/
def reverse_digits (n : TwoDigitNumber) : TwoDigitNumber :=
  ⟨10 * (units_digit n) + (tens_digit n), by sorry⟩

theorem unique_two_digit_number_exists :
  ∃! (X : TwoDigitNumber),
    (tens_digit X) * (units_digit X) = 24 ∧
    (reverse_digits X).val = X.val + 18 ∧
    X.val = 46 := by sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_exists_l225_22502


namespace NUMINAMATH_CALUDE_cube_side_ratio_l225_22537

theorem cube_side_ratio (w1 w2 s1 s2 : ℝ) (hw1 : w1 = 6) (hw2 : w2 = 48) 
  (hv : w2 / w1 = (s2 / s1)^3) : s2 / s1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_ratio_l225_22537


namespace NUMINAMATH_CALUDE_rooms_per_hall_first_wing_is_32_l225_22549

/-- Represents a hotel with two wings -/
structure Hotel where
  total_rooms : ℕ
  first_wing_floors : ℕ
  first_wing_halls_per_floor : ℕ
  second_wing_floors : ℕ
  second_wing_halls_per_floor : ℕ
  second_wing_rooms_per_hall : ℕ

/-- Calculates the number of rooms in each hall of the first wing -/
def rooms_per_hall_first_wing (h : Hotel) : ℕ :=
  let second_wing_rooms := h.second_wing_floors * h.second_wing_halls_per_floor * h.second_wing_rooms_per_hall
  let first_wing_rooms := h.total_rooms - second_wing_rooms
  let total_halls_first_wing := h.first_wing_floors * h.first_wing_halls_per_floor
  first_wing_rooms / total_halls_first_wing

/-- Theorem stating that for the given hotel configuration, 
    each hall in the first wing has 32 rooms -/
theorem rooms_per_hall_first_wing_is_32 :
  rooms_per_hall_first_wing {
    total_rooms := 4248,
    first_wing_floors := 9,
    first_wing_halls_per_floor := 6,
    second_wing_floors := 7,
    second_wing_halls_per_floor := 9,
    second_wing_rooms_per_hall := 40
  } = 32 := by
  sorry

end NUMINAMATH_CALUDE_rooms_per_hall_first_wing_is_32_l225_22549


namespace NUMINAMATH_CALUDE_triangle_side_length_l225_22599

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = 2 →
  c = 2 * Real.sqrt 3 →
  Real.cos A = Real.sqrt 3 / 2 →
  b < c →
  b = 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l225_22599


namespace NUMINAMATH_CALUDE_f_properties_l225_22568

noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.sin x, Real.cos x ^ 2)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, 2)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧
  (∃ M, ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ M) ∧
  (∃ m, ∀ x ∈ Set.Icc 0 (Real.pi / 2), m ≤ f x) ∧
  (∃ x₁ ∈ Set.Icc 0 (Real.pi / 2), ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ f x₁) ∧
  (∃ x₂ ∈ Set.Icc 0 (Real.pi / 2), ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x₂ ≤ f x) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l225_22568


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l225_22531

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ r : ℝ, r > 0 ∧ a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 1 * a 2 * a 3 = 5 →
  a 7 * a 8 * a 9 = 10 →
  a 4 * a 5 * a 6 = 5 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_product_l225_22531


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l225_22545

def M : Set ℝ := {x | -1 < x ∧ x < 3}
def N : Set ℝ := {x | -2 < x ∧ x < 1}

theorem intersection_of_M_and_N : M ∩ N = {x | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l225_22545


namespace NUMINAMATH_CALUDE_gcd_n_cube_plus_25_and_n_plus_6_l225_22582

theorem gcd_n_cube_plus_25_and_n_plus_6 (n : ℕ) (h : n > 2^5) :
  Nat.gcd (n^3 + 5^2) (n + 6) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_n_cube_plus_25_and_n_plus_6_l225_22582


namespace NUMINAMATH_CALUDE_hyperbola_focus_l225_22575

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 3 - x^2 / 6 = 1

/-- Definition of a focus of the hyperbola -/
def is_focus (x y : ℝ) : Prop :=
  ∃ (c : ℝ), c^2 = 3 + 6 ∧ (x = 0 ∧ (y = c ∨ y = -c))

/-- Theorem: One focus of the hyperbola has coordinates (0, 3) -/
theorem hyperbola_focus : ∃ (x y : ℝ), hyperbola_equation x y ∧ is_focus x y ∧ x = 0 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focus_l225_22575


namespace NUMINAMATH_CALUDE_total_age_proof_l225_22566

/-- Given three people a, b, and c, where:
  - a is two years older than b
  - b is twice as old as c
  - b is 28 years old
  Prove that the total of their ages is 72 years. -/
theorem total_age_proof (a b c : ℕ) : 
  b = 28 → a = b + 2 → b = 2 * c → a + b + c = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_age_proof_l225_22566


namespace NUMINAMATH_CALUDE_range_of_m_l225_22521

theorem range_of_m (m : ℝ) : 
  (∃ (x : ℝ), x^2 + 2*x - m > 0) ↔ 
  (1^2 + 2*1 - m ≤ 0 ∧ 2^2 + 2*2 - m > 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l225_22521


namespace NUMINAMATH_CALUDE_cnc_processing_time_l225_22510

/-- The time required for one CNC machine to process a given number of parts, 
    given the rate of multiple machines. -/
theorem cnc_processing_time 
  (machines : ℕ) 
  (parts : ℕ) 
  (hours : ℕ) 
  (target_parts : ℕ) : 
  machines > 0 → 
  parts > 0 → 
  hours > 0 → 
  target_parts > 0 → 
  (3 : ℕ) = machines → 
  (960 : ℕ) = parts → 
  (4 : ℕ) = hours → 
  (400 : ℕ) = target_parts → 
  (5 : ℕ) = (target_parts * machines * hours) / parts := by
  sorry


end NUMINAMATH_CALUDE_cnc_processing_time_l225_22510


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l225_22532

theorem inequality_and_equality_condition 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℕ+) :
  let lhs := (a₁ * b₂ + a₁ * b₃ + a₂ * b₁ + a₂ * b₃ + a₃ * b₁ + a₃ * b₂)^2
  let rhs := 4 * (a₁ * a₂ + a₂ * a₃ + a₃ * a₁) * (b₁ * b₂ + b₂ * b₃ + b₃ * b₁)
  (lhs ≥ rhs) ∧ 
  (lhs = rhs ↔ (a₁ : ℚ) / b₁ = (a₂ : ℚ) / b₂ ∧ (a₂ : ℚ) / b₂ = (a₃ : ℚ) / b₃) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l225_22532


namespace NUMINAMATH_CALUDE_fraction_subtraction_equality_l225_22550

theorem fraction_subtraction_equality : 
  -1/8 - (1 + 1/3) - (-5/8) - (4 + 2/3) = -(11/2) := by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_equality_l225_22550


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l225_22554

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 1) :
  (1/x + 1/y) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l225_22554


namespace NUMINAMATH_CALUDE_subset_with_unique_sum_representation_l225_22517

theorem subset_with_unique_sum_representation :
  ∃ (X : Set ℤ), ∀ (n : ℤ), ∃! (p : ℤ × ℤ), p.1 ∈ X ∧ p.2 ∈ X ∧ p.1 + 2 * p.2 = n :=
sorry

end NUMINAMATH_CALUDE_subset_with_unique_sum_representation_l225_22517


namespace NUMINAMATH_CALUDE_ribbon_remaining_length_l225_22520

/-- The length of the original ribbon in meters -/
def original_length : ℝ := 51

/-- The number of pieces cut from the ribbon -/
def num_pieces : ℕ := 100

/-- The length of each piece in centimeters -/
def piece_length_cm : ℝ := 15

/-- Conversion factor from centimeters to meters -/
def cm_to_m : ℝ := 0.01

/-- The remaining length of the ribbon after cutting the pieces -/
def remaining_length : ℝ := original_length - (num_pieces : ℝ) * piece_length_cm * cm_to_m

theorem ribbon_remaining_length :
  remaining_length = 36 := by sorry

end NUMINAMATH_CALUDE_ribbon_remaining_length_l225_22520


namespace NUMINAMATH_CALUDE_pole_height_l225_22500

/-- Represents the geometry of a telephone pole with a supporting cable -/
structure TelephonePole where
  /-- Height of the pole in meters -/
  height : ℝ
  /-- Distance from the base of the pole to where the cable touches the ground, in meters -/
  cable_ground_distance : ℝ
  /-- Height of a person touching the cable, in meters -/
  person_height : ℝ
  /-- Distance from the base of the pole to where the person stands, in meters -/
  person_distance : ℝ

/-- Theorem stating the height of the telephone pole -/
theorem pole_height (pole : TelephonePole) 
  (h1 : pole.cable_ground_distance = 3)
  (h2 : pole.person_height = 1.5)
  (h3 : pole.person_distance = 2.5)
  : pole.height = 9 := by
  sorry

/-- Main statement combining the structure and theorem -/
def main : Prop :=
  ∃ pole : TelephonePole, 
    pole.cable_ground_distance = 3 ∧
    pole.person_height = 1.5 ∧
    pole.person_distance = 2.5 ∧
    pole.height = 9

end NUMINAMATH_CALUDE_pole_height_l225_22500


namespace NUMINAMATH_CALUDE_molecular_weight_CaCO3_is_100_09_l225_22576

/-- The molecular weight of calcium carbonate (CaCO3) -/
def molecular_weight_CaCO3 : ℝ :=
  let calcium_weight : ℝ := 40.08
  let carbon_weight : ℝ := 12.01
  let oxygen_weight : ℝ := 16.00
  calcium_weight + carbon_weight + 3 * oxygen_weight

/-- Theorem stating that the molecular weight of CaCO3 is approximately 100.09 -/
theorem molecular_weight_CaCO3_is_100_09 :
  ∃ ε > 0, |molecular_weight_CaCO3 - 100.09| < ε :=
sorry

end NUMINAMATH_CALUDE_molecular_weight_CaCO3_is_100_09_l225_22576


namespace NUMINAMATH_CALUDE_sine_sqrt_equality_l225_22539

theorem sine_sqrt_equality (a : ℝ) (h1 : a ≥ 0) :
  (∀ x : ℝ, x ≥ 0 → Real.sin (Real.sqrt (x + a)) = Real.sin (Real.sqrt x)) →
  a = 0 := by
  sorry

end NUMINAMATH_CALUDE_sine_sqrt_equality_l225_22539


namespace NUMINAMATH_CALUDE_half_inequality_l225_22529

theorem half_inequality (a b : ℝ) (h : a > b) : a / 2 > b / 2 := by
  sorry

end NUMINAMATH_CALUDE_half_inequality_l225_22529


namespace NUMINAMATH_CALUDE_f_plus_g_at_one_equals_two_l225_22546

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_plus_g_at_one_equals_two
  (f g : ℝ → ℝ)
  (h_even : isEven f)
  (h_odd : isOdd g)
  (h_eq : ∀ x, f x - g x = x^3 + x^2 + 1) :
  f 1 + g 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_f_plus_g_at_one_equals_two_l225_22546


namespace NUMINAMATH_CALUDE_square_difference_equality_l225_22552

theorem square_difference_equality : 1.99^2 - 1.98 * 1.99 + 0.99^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l225_22552


namespace NUMINAMATH_CALUDE_probability_of_green_ball_l225_22577

theorem probability_of_green_ball (total_balls : ℕ) (green_balls : ℕ) (red_balls : ℕ)
  (h1 : total_balls = 10)
  (h2 : green_balls = 7)
  (h3 : red_balls = 3)
  (h4 : total_balls = green_balls + red_balls) :
  (green_balls : ℚ) / total_balls = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_green_ball_l225_22577


namespace NUMINAMATH_CALUDE_parabola_properties_l225_22538

-- Define the parabola and its coefficients
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the points that the parabola passes through
def point_A : ℝ × ℝ := (-1, 0)
def point_C : ℝ × ℝ := (0, 3)
def point_B : ℝ × ℝ := (2, -3)

-- Theorem stating the properties of the parabola
theorem parabola_properties :
  ∃ (a b c : ℝ),
    -- The parabola passes through the given points
    (parabola a b c (point_A.1) = point_A.2) ∧
    (parabola a b c (point_C.1) = point_C.2) ∧
    (parabola a b c (point_B.1) = point_B.2) ∧
    -- The parabola equation is y = -2x² + x + 3
    (a = -2 ∧ b = 1 ∧ c = 3) ∧
    -- The axis of symmetry is x = 1/4
    (- b / (2 * a) = 1 / 4) ∧
    -- The vertex coordinates are (1/4, 25/8)
    (parabola a b c (1 / 4) = 25 / 8) := by
  sorry


end NUMINAMATH_CALUDE_parabola_properties_l225_22538


namespace NUMINAMATH_CALUDE_digit_strike_out_theorem_l225_22525

/-- Represents a positive integer as a list of its digits --/
def DigitList := List Nat

/-- Checks if a number represented as a list of digits is divisible by 9 --/
def isDivisibleBy9 (n : DigitList) : Prop :=
  (n.sum % 9 = 0)

/-- Checks if a number can be obtained by striking out one digit from another number --/
def canBeObtainedByStrikingOut (m n : DigitList) : Prop :=
  ∃ (i : Nat), i < n.length ∧ m = (n.take i ++ n.drop (i+1))

/-- The main theorem --/
theorem digit_strike_out_theorem (N : DigitList) :
  (∃ (M : DigitList), N.sum = 9 * M.sum ∧ 
    canBeObtainedByStrikingOut M N ∧ 
    isDivisibleBy9 M) →
  (∀ (K : DigitList), canBeObtainedByStrikingOut K M → isDivisibleBy9 K) ∧
  (N ∈ [[1,0,1,2,5], [2,0,2,5], [3,0,3,7,5], [4,0,5], [5,0,6,2,5], [6,7,5], [7,0,8,7,5]]) :=
by
  sorry


end NUMINAMATH_CALUDE_digit_strike_out_theorem_l225_22525


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l225_22595

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  d : ℤ      -- Common difference
  h1 : a 3 + a 4 = 15
  h2 : a 2 * a 5 = 54
  h3 : d < 0

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = 11 - n) ∧
  (∃ n : ℕ, sum_n seq n = 55) ∧
  (∀ n : ℕ, sum_n seq n ≤ 55) ∧
  (sum_n seq 11 = 55) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l225_22595


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l225_22578

theorem fraction_sum_equality (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a / (b - c) + b / (c - a) + c / (a - b) = 1) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 1 / (b - c) + 1 / (c - a) + 1 / (a - b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l225_22578


namespace NUMINAMATH_CALUDE_president_vp_advisory_board_selection_l225_22548

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem president_vp_advisory_board_selection (total_people : ℕ) (h : total_people = 10) :
  (total_people) * (total_people - 1) * (choose (total_people - 2) 2) = 2520 :=
sorry

end NUMINAMATH_CALUDE_president_vp_advisory_board_selection_l225_22548


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l225_22507

theorem perfect_square_polynomial (n : ℤ) : 
  (∃ k : ℤ, n^4 + 6*n^3 + 11*n^2 + 3*n + 31 = k^2) ↔ n = 10 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l225_22507


namespace NUMINAMATH_CALUDE_representatives_selection_theorem_l225_22551

def number_of_students : ℕ := 6
def number_of_representatives : ℕ := 3

def select_representatives (n m : ℕ) (at_least_one_from_set : ℕ) : ℕ :=
  sorry

theorem representatives_selection_theorem :
  select_representatives number_of_students number_of_representatives 2 = 96 :=
sorry

end NUMINAMATH_CALUDE_representatives_selection_theorem_l225_22551


namespace NUMINAMATH_CALUDE_triangle_extension_l225_22580

/-- Triangle extension theorem -/
theorem triangle_extension (n : ℕ) (a b c t S : ℝ) 
  (h_n : n > 0)
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_area : t > 0)
  (h_S : S = a^2 + b^2 + c^2)
  (t_i : Fin (n-1) → ℝ)
  (S_i : Fin (n-1) → ℝ)
  (h_t_i : ∀ i, t_i i > 0)
  (h_S_i : ∀ i, S_i i > 0) :
  (∃ k : ℝ, 
    (S + (Finset.sum Finset.univ S_i) = n^3 * S) ∧ 
    (t + (Finset.sum Finset.univ t_i) = n^3 * t) ∧ 
    (∀ i : Fin (n-1), S_i i / t_i i = k) ∧ 
    (S / t = k)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_extension_l225_22580


namespace NUMINAMATH_CALUDE_problem_statement_l225_22523

theorem problem_statement : (-1)^53 + 2^(4^3 + 5^2 - 7^2) = 1099511627775 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l225_22523


namespace NUMINAMATH_CALUDE_one_ounce_in_gallons_l225_22555

/-- The number of ounces in one gallon of water -/
def ounces_per_gallon : ℚ := 128

/-- The number of ounces Jimmy drinks each time -/
def ounces_per_serving : ℚ := 8

/-- The number of times Jimmy drinks water per day -/
def servings_per_day : ℚ := 8

/-- The number of gallons Jimmy prepares for 5 days -/
def gallons_for_five_days : ℚ := 5/2

/-- The number of days Jimmy prepares water for -/
def days_prepared : ℚ := 5

/-- Theorem stating that 1 ounce of water is equal to 1/128 gallons -/
theorem one_ounce_in_gallons :
  1 / ounces_per_gallon = 
    gallons_for_five_days / (ounces_per_serving * servings_per_day * days_prepared) :=
by sorry

end NUMINAMATH_CALUDE_one_ounce_in_gallons_l225_22555


namespace NUMINAMATH_CALUDE_smaller_of_reciprocal_and_sine_interval_length_l225_22560

open Real

theorem smaller_of_reciprocal_and_sine (x : ℝ) :
  (min (1/x) (sin x) > 1/2) ↔ (π/6 < x ∧ x < 5*π/6) :=
sorry

theorem interval_length : 
  (5*π/6 - π/6 : ℝ) = 2*π/3 :=
sorry

end NUMINAMATH_CALUDE_smaller_of_reciprocal_and_sine_interval_length_l225_22560


namespace NUMINAMATH_CALUDE_james_payment_is_six_l225_22556

/-- Calculates James's share of the payment for stickers -/
def jamesPayment (packs : ℕ) (stickersPerPack : ℕ) (stickerCost : ℚ) (friendSharePercent : ℚ) : ℚ :=
  let totalStickers := packs * stickersPerPack
  let totalCost := totalStickers * stickerCost
  totalCost * (1 - friendSharePercent)

/-- Proves that James pays $6 for his share of the stickers -/
theorem james_payment_is_six :
  jamesPayment 4 30 (1/10) (1/2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_james_payment_is_six_l225_22556


namespace NUMINAMATH_CALUDE_stream_to_meadow_distance_l225_22572

/-- Given a hiking trip with known distances, prove the distance between two points -/
theorem stream_to_meadow_distance 
  (total_distance : ℝ)
  (car_to_stream : ℝ)
  (meadow_to_campsite : ℝ)
  (h1 : total_distance = 0.7)
  (h2 : car_to_stream = 0.2)
  (h3 : meadow_to_campsite = 0.1) :
  total_distance - car_to_stream - meadow_to_campsite = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_stream_to_meadow_distance_l225_22572


namespace NUMINAMATH_CALUDE_quadratic_roots_distinct_real_l225_22515

/-- Given a quadratic equation bx^2 - 3x√5 + d = 0 with real constants b and d,
    and a discriminant of 25, the roots are distinct and real. -/
theorem quadratic_roots_distinct_real (b d : ℝ) : 
  let discriminant := (-3 * Real.sqrt 5) ^ 2 - 4 * b * d
  ∀ x : ℝ, (b * x^2 - 3 * x * Real.sqrt 5 + d = 0 ∧ discriminant = 25) →
    ∃ y : ℝ, x ≠ y ∧ b * y^2 - 3 * y * Real.sqrt 5 + d = 0 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_roots_distinct_real_l225_22515


namespace NUMINAMATH_CALUDE_days_to_fulfill_orders_l225_22589

/-- Represents the production details of Wallace's beef jerky company -/
structure JerkyProduction where
  small_batch_time : ℕ
  small_batch_output : ℕ
  large_batch_time : ℕ
  large_batch_output : ℕ
  total_small_bags_ordered : ℕ
  total_large_bags_ordered : ℕ
  small_bags_in_stock : ℕ
  large_bags_in_stock : ℕ
  max_daily_production_hours : ℕ

/-- Calculates the minimum number of days required to fulfill all orders -/
def min_days_to_fulfill_orders (prod : JerkyProduction) : ℕ :=
  let small_bags_to_produce := prod.total_small_bags_ordered - prod.small_bags_in_stock
  let large_bags_to_produce := prod.total_large_bags_ordered - prod.large_bags_in_stock
  let small_batches_needed := (small_bags_to_produce + prod.small_batch_output - 1) / prod.small_batch_output
  let large_batches_needed := (large_bags_to_produce + prod.large_batch_output - 1) / prod.large_batch_output
  let total_hours_needed := small_batches_needed * prod.small_batch_time + large_batches_needed * prod.large_batch_time
  (total_hours_needed + prod.max_daily_production_hours - 1) / prod.max_daily_production_hours

/-- Theorem stating that given the specific conditions, 13 days are required to fulfill all orders -/
theorem days_to_fulfill_orders :
  let prod := JerkyProduction.mk 8 12 12 8 157 97 18 10 18
  min_days_to_fulfill_orders prod = 13 := by
  sorry


end NUMINAMATH_CALUDE_days_to_fulfill_orders_l225_22589


namespace NUMINAMATH_CALUDE_a_value_l225_22571

theorem a_value (a : ℚ) (h : a + a/4 = 10/4) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_a_value_l225_22571


namespace NUMINAMATH_CALUDE_polygon_sides_l225_22509

/-- 
A polygon has n sides. 
The sum of its interior angles is (n - 2) * 180°.
The sum of its exterior angles is 360°.
The sum of its interior angles is three times the sum of its exterior angles.
Prove that n = 8.
-/
theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l225_22509


namespace NUMINAMATH_CALUDE_g_composition_result_l225_22543

/-- Definition of the function g for complex numbers -/
noncomputable def g (z : ℂ) : ℂ :=
  if z.im = 0 then -z^3 - 1 else z^3 + 1

/-- Theorem stating the result of g(g(g(g(2+i)))) -/
theorem g_composition_result :
  g (g (g (g (2 + Complex.I)))) = (-64555 + 70232 * Complex.I)^3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_result_l225_22543


namespace NUMINAMATH_CALUDE_connie_initial_marbles_l225_22588

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℕ := 70

/-- The number of marbles Connie has left -/
def marbles_left : ℕ := 3

/-- The initial number of marbles Connie had -/
def initial_marbles : ℕ := marbles_given + marbles_left

theorem connie_initial_marbles : initial_marbles = 73 := by
  sorry

end NUMINAMATH_CALUDE_connie_initial_marbles_l225_22588


namespace NUMINAMATH_CALUDE_total_age_is_32_l225_22534

-- Define the ages of a, b, and c
def age_b : ℕ := 12
def age_a : ℕ := age_b + 2
def age_c : ℕ := age_b / 2

-- Theorem to prove
theorem total_age_is_32 : age_a + age_b + age_c = 32 := by
  sorry


end NUMINAMATH_CALUDE_total_age_is_32_l225_22534


namespace NUMINAMATH_CALUDE_quadratic_linear_intersection_l225_22562

theorem quadratic_linear_intersection (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 2 * x + 2 = -3 * x - 2) ↔ a = 25 / 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_linear_intersection_l225_22562


namespace NUMINAMATH_CALUDE_circle_chord_intersection_theorem_l225_22597

noncomputable def circle_chord_intersection_problem 
  (O : ℝ × ℝ) 
  (A B C D P : ℝ × ℝ) 
  (radius : ℝ) 
  (chord_AB_length : ℝ) 
  (chord_CD_length : ℝ) 
  (midpoint_distance : ℝ) : Prop :=
  let midpoint_AB := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let midpoint_CD := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  radius = 25 ∧
  chord_AB_length = 30 ∧
  chord_CD_length = 14 ∧
  midpoint_distance = 12 ∧
  (∀ X : ℝ × ℝ, (X.1 - O.1)^2 + (X.2 - O.2)^2 = radius^2 → 
    ((X = A ∨ X = B ∨ X = C ∨ X = D) ∨ 
     ((X.1 - A.1)^2 + (X.2 - A.2)^2) * ((X.1 - B.1)^2 + (X.2 - B.2)^2) > chord_AB_length^2 ∧
     ((X.1 - C.1)^2 + (X.2 - C.2)^2) * ((X.1 - D.1)^2 + (X.2 - D.2)^2) > chord_CD_length^2)) ∧
  (P.1 - A.1) * (B.1 - A.1) + (P.2 - A.2) * (B.2 - A.2) = 
    (P.1 - B.1) * (A.1 - B.1) + (P.2 - B.2) * (A.2 - B.2) ∧
  (P.1 - C.1) * (D.1 - C.1) + (P.2 - C.2) * (D.2 - C.2) = 
    (P.1 - D.1) * (C.1 - D.1) + (P.2 - D.2) * (C.2 - D.2) ∧
  (midpoint_AB.1 - midpoint_CD.1)^2 + (midpoint_AB.2 - midpoint_CD.2)^2 = midpoint_distance^2 →
  (P.1 - O.1)^2 + (P.2 - O.2)^2 = 4050 / 7

theorem circle_chord_intersection_theorem 
  (O : ℝ × ℝ) 
  (A B C D P : ℝ × ℝ) 
  (radius : ℝ) 
  (chord_AB_length : ℝ) 
  (chord_CD_length : ℝ) 
  (midpoint_distance : ℝ) :
  circle_chord_intersection_problem O A B C D P radius chord_AB_length chord_CD_length midpoint_distance :=
by sorry

end NUMINAMATH_CALUDE_circle_chord_intersection_theorem_l225_22597


namespace NUMINAMATH_CALUDE_percentage_of_truth_speakers_l225_22559

theorem percentage_of_truth_speakers (L B : ℝ) (h1 : L = 0.2) (h2 : B = 0.1) 
  (h3 : L + B + (L + B - B) = 0.4) : L + B - B = 0.3 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_truth_speakers_l225_22559


namespace NUMINAMATH_CALUDE_nancy_next_month_games_l225_22583

/-- The number of football games Nancy plans to attend next month -/
def games_next_month (games_this_month games_last_month total_games : ℕ) : ℕ :=
  total_games - (games_this_month + games_last_month)

/-- Proof that Nancy plans to attend 7 games next month -/
theorem nancy_next_month_games :
  games_next_month 9 8 24 = 7 := by
  sorry

end NUMINAMATH_CALUDE_nancy_next_month_games_l225_22583


namespace NUMINAMATH_CALUDE_prime_difference_divisibility_l225_22542

theorem prime_difference_divisibility (n : ℕ) : 
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ n ∣ (p - q) := by
  sorry

end NUMINAMATH_CALUDE_prime_difference_divisibility_l225_22542


namespace NUMINAMATH_CALUDE_cone_base_diameter_l225_22506

/-- For a cone with surface area 3π and lateral surface that unfolds into a semicircle, 
    the diameter of its base is 2. -/
theorem cone_base_diameter (l r : ℝ) 
  (h1 : (1/2) * Real.pi * l^2 + Real.pi * r^2 = 3 * Real.pi) 
  (h2 : Real.pi * l = 2 * Real.pi * r) : 
  2 * r = 2 := by sorry

end NUMINAMATH_CALUDE_cone_base_diameter_l225_22506


namespace NUMINAMATH_CALUDE_no_damaged_pool_floats_l225_22535

/-- Prove that the number of damaged pool floats is 0 given the following conditions:
  - Total donations: 300
  - Basketball hoops: 60
  - Half of basketball hoops came with basketballs
  - Pool floats donated: 120
  - Footballs: 50
  - Tennis balls: 40
  - Remaining donations were basketballs
-/
theorem no_damaged_pool_floats (total_donations : ℕ) (basketball_hoops : ℕ) (pool_floats : ℕ)
  (footballs : ℕ) (tennis_balls : ℕ) (h1 : total_donations = 300)
  (h2 : basketball_hoops = 60) (h3 : pool_floats = 120) (h4 : footballs = 50) (h5 : tennis_balls = 40)
  (h6 : 2 * (basketball_hoops / 2) + pool_floats + footballs + tennis_balls +
    (total_donations - (basketball_hoops + pool_floats + footballs + tennis_balls)) = total_donations) :
  total_donations - (basketball_hoops + pool_floats + footballs + tennis_balls) = pool_floats := by
  sorry

#check no_damaged_pool_floats

end NUMINAMATH_CALUDE_no_damaged_pool_floats_l225_22535


namespace NUMINAMATH_CALUDE_total_area_is_36_l225_22503

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- The size of the square grid -/
def gridSize : ℕ := 6

/-- The center point of the grid -/
def gridCenter : Point := { x := 3, y := 3 }

/-- Calculates the area of a triangle given its three points -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Generates all triangles formed by connecting the center to adjacent perimeter points -/
def perimeterTriangles : List Triangle := sorry

/-- Theorem: The total area of triangles formed by connecting the center of a 6x6 square grid
    to each pair of adjacent vertices along the perimeter is equal to 36 -/
theorem total_area_is_36 : 
  (perimeterTriangles.map triangleArea).sum = 36 := by sorry

end NUMINAMATH_CALUDE_total_area_is_36_l225_22503


namespace NUMINAMATH_CALUDE_church_cookie_baking_l225_22518

theorem church_cookie_baking (members : ℕ) (cookies_per_sheet : ℕ) (total_cookies : ℕ) 
  (h1 : members = 100)
  (h2 : cookies_per_sheet = 16)
  (h3 : total_cookies = 16000) :
  total_cookies / (members * cookies_per_sheet) = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_church_cookie_baking_l225_22518


namespace NUMINAMATH_CALUDE_trajectory_of_P_max_distance_to_L_min_distance_to_L_l225_22586

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point M on circle C
def M (x₀ y₀ : ℝ) : Prop := C x₀ y₀

-- Define point N
def N : ℝ × ℝ := (4, 0)

-- Define point P as midpoint of MN
def P (x y x₀ y₀ : ℝ) : Prop := x = (x₀ + 4) / 2 ∧ y = y₀ / 2

-- Theorem for the trajectory of P
theorem trajectory_of_P (x y : ℝ) : 
  (∃ x₀ y₀, M x₀ y₀ ∧ P x y x₀ y₀) → (x - 2)^2 + y^2 = 1 :=
sorry

-- Define the line L: 3x + 4y - 26 = 0
def L (x y : ℝ) : Prop := 3*x + 4*y - 26 = 0

-- Theorem for maximum distance
theorem max_distance_to_L (x y : ℝ) :
  (∃ x₀ y₀, M x₀ y₀ ∧ P x y x₀ y₀) → 
  (∀ x' y', (∃ x₀' y₀', M x₀' y₀' ∧ P x' y' x₀' y₀') → 
    |3*x + 4*y - 26| / Real.sqrt 25 ≤ 5) :=
sorry

-- Theorem for minimum distance
theorem min_distance_to_L (x y : ℝ) :
  (∃ x₀ y₀, M x₀ y₀ ∧ P x y x₀ y₀) → 
  (∀ x' y', (∃ x₀' y₀', M x₀' y₀' ∧ P x' y' x₀' y₀') → 
    |3*x + 4*y - 26| / Real.sqrt 25 ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_P_max_distance_to_L_min_distance_to_L_l225_22586


namespace NUMINAMATH_CALUDE_ball_radius_from_hole_dimensions_l225_22558

/-- Given a spherical ball partially submerged in a frozen surface,
    if the hole left after removing the ball is 30 cm across and 10 cm deep,
    then the radius of the ball is 16.25 cm. -/
theorem ball_radius_from_hole_dimensions (hole_width : ℝ) (hole_depth : ℝ) (ball_radius : ℝ) :
  hole_width = 30 →
  hole_depth = 10 →
  ball_radius = 16.25 := by
  sorry

end NUMINAMATH_CALUDE_ball_radius_from_hole_dimensions_l225_22558


namespace NUMINAMATH_CALUDE_function_properties_l225_22511

-- Define the function f(x)
def f (x : ℝ) : ℝ := |3*x + 3| - |x - 5|

-- Define the solution set M
def M : Set ℝ := {x | f x > 0}

-- State the theorem
theorem function_properties :
  (M = {x | x < -4 ∨ x > 1/2}) ∧
  (∃ m : ℝ, ∀ x : ℝ, f x ≥ m ∧ ∃ x₀ : ℝ, f x₀ = m) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 6 →
    1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 3/4) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l225_22511


namespace NUMINAMATH_CALUDE_b_completes_in_20_days_l225_22514

/-- The number of days it takes for person A to complete the work alone -/
def days_A : ℝ := 15

/-- The number of days A and B work together -/
def days_together : ℝ := 6

/-- The fraction of work left after A and B work together -/
def work_left : ℝ := 0.3

/-- The number of days it takes for person B to complete the work alone -/
def days_B : ℝ := 20

/-- Theorem stating that given the conditions, B can complete the work alone in 20 days -/
theorem b_completes_in_20_days :
  days_together * (1 / days_A + 1 / days_B) = 1 - work_left :=
sorry

end NUMINAMATH_CALUDE_b_completes_in_20_days_l225_22514


namespace NUMINAMATH_CALUDE_floor_sum_possible_values_l225_22565

theorem floor_sum_possible_values (x y z : ℝ) 
  (hx : ⌊x⌋ = 5) (hy : ⌊y⌋ = -3) (hz : ⌊z⌋ = -2) : 
  ⌊x - y + z⌋ ∈ ({5, 6, 7} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_possible_values_l225_22565


namespace NUMINAMATH_CALUDE_equation_solution_l225_22504

theorem equation_solution (x : ℝ) :
  8.438 * Real.cos (x - π/4) * (1 - 4 * Real.cos (2*x)^2) - 2 * Real.cos (4*x) = 3 →
  ∃ k : ℤ, x = π/4 * (8*k + 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l225_22504


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l225_22557

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - k * x + 4 ≥ 0) ↔ (0 ≤ k ∧ k ≤ 16) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l225_22557


namespace NUMINAMATH_CALUDE_domain_of_g_l225_22516

-- Define the function f with domain [-3, 1]
def f : Set ℝ → Set ℝ := fun D ↦ {x | x ∈ D ∧ -3 ≤ x ∧ x ≤ 1}

-- Define the function g in terms of f
def g (f : Set ℝ → Set ℝ) : Set ℝ → Set ℝ := fun D ↦ {x | (x + 1) ∈ f D}

-- Theorem statement
theorem domain_of_g (D : Set ℝ) :
  g f D = {x : ℝ | -4 ≤ x ∧ x ≤ 0} :=
sorry

end NUMINAMATH_CALUDE_domain_of_g_l225_22516


namespace NUMINAMATH_CALUDE_seven_balls_two_boxes_l225_22541

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes -/
def distributeDistinguishableBalls (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: There are 128 ways to distribute 7 distinguishable balls into 2 distinguishable boxes -/
theorem seven_balls_two_boxes :
  distributeDistinguishableBalls 7 2 = 128 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_two_boxes_l225_22541


namespace NUMINAMATH_CALUDE_second_pump_rate_l225_22547

/-- Proves that the rate of the second pump is 70 gallons per hour given the conditions -/
theorem second_pump_rate (pump1_rate : ℝ) (total_time : ℝ) (total_volume : ℝ) (pump2_time : ℝ)
  (h1 : pump1_rate = 180)
  (h2 : total_time = 6)
  (h3 : total_volume = 1325)
  (h4 : pump2_time = 3.5) :
  (total_volume - pump1_rate * total_time) / pump2_time = 70 := by
  sorry

end NUMINAMATH_CALUDE_second_pump_rate_l225_22547


namespace NUMINAMATH_CALUDE_turban_price_turban_price_is_70_l225_22579

/-- The price of a turban given the following conditions:
  * The total salary for one year is Rs. 90 plus the turban
  * The servant works for 9 months (3/4 of a year)
  * The servant receives Rs. 50 and the turban after 9 months
-/
theorem turban_price : ℝ → Prop :=
  fun price =>
    let yearly_salary := 90 + price
    let worked_fraction := 3 / 4
    let received_salary := 50 + price
    worked_fraction * yearly_salary = received_salary

/-- The price of the turban is 70 rupees -/
theorem turban_price_is_70 : turban_price 70 := by
  sorry

end NUMINAMATH_CALUDE_turban_price_turban_price_is_70_l225_22579


namespace NUMINAMATH_CALUDE_ages_when_violet_reaches_thomas_age_l225_22594

def thomas_age : ℕ := 6
def shay_age : ℕ := thomas_age + 13
def james_age : ℕ := shay_age + 5
def violet_age : ℕ := thomas_age - 3
def emily_age : ℕ := shay_age

def years_until_violet_reaches_thomas_age : ℕ := thomas_age - violet_age

theorem ages_when_violet_reaches_thomas_age :
  james_age + years_until_violet_reaches_thomas_age = 27 ∧
  emily_age + years_until_violet_reaches_thomas_age = 22 :=
by sorry

end NUMINAMATH_CALUDE_ages_when_violet_reaches_thomas_age_l225_22594


namespace NUMINAMATH_CALUDE_positive_integer_pairs_satisfying_equation_l225_22587

theorem positive_integer_pairs_satisfying_equation :
  ∀ x y : ℕ+, 
    (x * y * Nat.gcd x.val y.val = x + y + (Nat.gcd x.val y.val)^2) ↔ 
    ((x = 2 ∧ y = 2) ∨ (x = 2 ∧ y = 3) ∨ (x = 3 ∧ y = 2)) := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_pairs_satisfying_equation_l225_22587


namespace NUMINAMATH_CALUDE_window_installation_time_l225_22508

theorem window_installation_time (total_windows : ℕ) (installed_windows : ℕ) (time_per_window : ℕ) :
  total_windows = 10 →
  installed_windows = 6 →
  time_per_window = 5 →
  (total_windows - installed_windows) * time_per_window = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_window_installation_time_l225_22508


namespace NUMINAMATH_CALUDE_ab_plus_cd_value_l225_22536

theorem ab_plus_cd_value (a b c d : ℝ) 
  (eq1 : a + b + c = 3)
  (eq2 : a + b + d = -1)
  (eq3 : a + c + d = 8)
  (eq4 : b + c + d = 0) :
  a * b + c * d = -127 / 9 := by
sorry

end NUMINAMATH_CALUDE_ab_plus_cd_value_l225_22536


namespace NUMINAMATH_CALUDE_buttons_pattern_l225_22567

/-- Represents the number of buttons in the nth box -/
def buttonsInBox (n : ℕ) : ℕ := 3^(n - 1)

/-- Represents the total number of buttons up to the nth box -/
def totalButtons (n : ℕ) : ℕ := (3^n - 1) / 2

theorem buttons_pattern (n : ℕ) (h : n > 0) :
  (buttonsInBox 1 = 1) ∧
  (buttonsInBox 2 = 3) ∧
  (buttonsInBox 3 = 9) ∧
  (buttonsInBox 4 = 27) ∧
  (buttonsInBox 5 = 81) →
  (∀ k : ℕ, k > 0 → buttonsInBox k = 3^(k - 1)) ∧
  (totalButtons n = (3^n - 1) / 2) :=
sorry

end NUMINAMATH_CALUDE_buttons_pattern_l225_22567


namespace NUMINAMATH_CALUDE_one_fourth_of_eight_point_eight_l225_22519

theorem one_fourth_of_eight_point_eight (x : ℚ) : x = 8.8 → (1 / 4 : ℚ) * x = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_eight_point_eight_l225_22519


namespace NUMINAMATH_CALUDE_team_a_win_probability_l225_22505

/-- Probability of Team A winning a non-fifth set -/
def p : ℚ := 2/3

/-- Probability of Team A winning the fifth set -/
def p_fifth : ℚ := 1/2

/-- The probability of Team A winning the volleyball match -/
theorem team_a_win_probability : 
  (p^3) + (3 * p^2 * (1-p) * p) + (6 * p^2 * (1-p)^2 * p_fifth) = 20/27 := by
  sorry

end NUMINAMATH_CALUDE_team_a_win_probability_l225_22505
