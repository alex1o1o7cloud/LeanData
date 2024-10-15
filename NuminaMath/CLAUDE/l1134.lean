import Mathlib

namespace NUMINAMATH_CALUDE_probability_one_or_two_in_pascal_l1134_113456

/-- The number of elements in the first n rows of Pascal's Triangle -/
def pascal_triangle_elements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of 1's in the first n rows of Pascal's Triangle -/
def ones_in_pascal (n : ℕ) : ℕ := if n = 0 then 0 else 2 * n - 1

/-- The number of 2's in the first n rows of Pascal's Triangle -/
def twos_in_pascal (n : ℕ) : ℕ := if n ≤ 2 then 0 else 2 * (n - 2)

/-- The probability of selecting 1 or 2 from the first 20 rows of Pascal's Triangle -/
theorem probability_one_or_two_in_pascal : 
  (ones_in_pascal 20 + twos_in_pascal 20 : ℚ) / pascal_triangle_elements 20 = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_or_two_in_pascal_l1134_113456


namespace NUMINAMATH_CALUDE_mary_flour_amount_l1134_113454

/-- The amount of flour Mary puts in the cake. -/
def total_flour (recipe_flour extra_flour : ℝ) : ℝ :=
  recipe_flour + extra_flour

/-- Theorem stating the total amount of flour Mary uses. -/
theorem mary_flour_amount :
  let recipe_flour : ℝ := 7.0
  let extra_flour : ℝ := 2.0
  total_flour recipe_flour extra_flour = 9.0 := by
  sorry

end NUMINAMATH_CALUDE_mary_flour_amount_l1134_113454


namespace NUMINAMATH_CALUDE_tank_egg_difference_l1134_113426

/-- The number of eggs Tank gathered in the first round -/
def tank_first : ℕ := 160

/-- The number of eggs Tank gathered in the second round -/
def tank_second : ℕ := 30

/-- The number of eggs Emma gathered in the first round -/
def emma_first : ℕ := tank_first - 10

/-- The number of eggs Emma gathered in the second round -/
def emma_second : ℕ := 60

/-- The total number of eggs collected by all 8 people -/
def total_eggs : ℕ := 400

theorem tank_egg_difference :
  tank_first - tank_second = 130 ∧
  emma_second = 2 * tank_second ∧
  tank_first > tank_second ∧
  tank_first = emma_first + 10 ∧
  tank_first + emma_first + tank_second + emma_second = total_eggs :=
by sorry

end NUMINAMATH_CALUDE_tank_egg_difference_l1134_113426


namespace NUMINAMATH_CALUDE_alices_number_l1134_113481

theorem alices_number (y : ℝ) : 3 * (3 * y + 15) = 135 → y = 10 := by
  sorry

end NUMINAMATH_CALUDE_alices_number_l1134_113481


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l1134_113476

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

theorem collinear_vectors_x_value :
  let a : ℝ × ℝ := (2, 4)
  let b : ℝ × ℝ := (x, 6)
  collinear a b → x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l1134_113476


namespace NUMINAMATH_CALUDE_intersection_implies_m_value_l1134_113447

theorem intersection_implies_m_value (m : ℝ) : 
  let A : Set ℝ := {1, m-2}
  let B : Set ℝ := {2, 3}
  A ∩ B = {2} → m = 4 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_m_value_l1134_113447


namespace NUMINAMATH_CALUDE_cricket_team_size_l1134_113450

theorem cricket_team_size :
  ∀ n : ℕ,
  n > 0 →
  let avg_age : ℚ := 26
  let wicket_keeper_age : ℚ := avg_age + 3
  let remaining_avg_age : ℚ := avg_age - 1
  (n : ℚ) * avg_age = wicket_keeper_age + avg_age + (n - 2 : ℚ) * remaining_avg_age →
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_size_l1134_113450


namespace NUMINAMATH_CALUDE_houses_with_dogs_l1134_113416

/-- Given a group of houses, prove the number of houses with dogs -/
theorem houses_with_dogs 
  (total_houses : ℕ) 
  (houses_with_cats : ℕ) 
  (houses_with_both : ℕ) 
  (h1 : total_houses = 60) 
  (h2 : houses_with_cats = 30) 
  (h3 : houses_with_both = 10) : 
  ∃ (houses_with_dogs : ℕ), houses_with_dogs = 40 ∧ 
    houses_with_dogs + houses_with_cats - houses_with_both ≤ total_houses :=
by sorry

end NUMINAMATH_CALUDE_houses_with_dogs_l1134_113416


namespace NUMINAMATH_CALUDE_binomial_not_perfect_power_l1134_113443

theorem binomial_not_perfect_power (n k l m : ℕ) : 
  l ≥ 2 → 4 ≤ k → k ≤ n - 4 → (n.choose k) ≠ m^l := by
  sorry

end NUMINAMATH_CALUDE_binomial_not_perfect_power_l1134_113443


namespace NUMINAMATH_CALUDE_plane_not_perp_implies_no_perp_line_l1134_113424

-- Define planes and lines
variable (α β : Set (ℝ × ℝ × ℝ))
variable (l : Set (ℝ × ℝ × ℝ))

-- Define perpendicularity for planes and lines
def perpendicular_planes (p q : Set (ℝ × ℝ × ℝ)) : Prop := sorry
def perpendicular_line_plane (l : Set (ℝ × ℝ × ℝ)) (p : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define a line being in a plane
def line_in_plane (l : Set (ℝ × ℝ × ℝ)) (p : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Theorem statement
theorem plane_not_perp_implies_no_perp_line :
  ¬(perpendicular_planes α β) →
  ¬∃ l, line_in_plane l α ∧ perpendicular_line_plane l β :=
sorry

end NUMINAMATH_CALUDE_plane_not_perp_implies_no_perp_line_l1134_113424


namespace NUMINAMATH_CALUDE_ray_walks_to_highschool_l1134_113423

/-- Represents the number of blocks Ray walks to the park -/
def blocks_to_park : ℕ := 4

/-- Represents the number of blocks Ray walks from the high school to home -/
def blocks_from_highschool_to_home : ℕ := 11

/-- Represents the number of times Ray walks his dog each day -/
def walks_per_day : ℕ := 3

/-- Represents the total number of blocks Ray's dog walks each day -/
def total_blocks_per_day : ℕ := 66

/-- Represents the number of blocks Ray walks to the high school -/
def blocks_to_highschool : ℕ := 7

theorem ray_walks_to_highschool :
  blocks_to_highschool = 7 ∧
  walks_per_day * (blocks_to_park + blocks_to_highschool + blocks_from_highschool_to_home) = total_blocks_per_day :=
by sorry

end NUMINAMATH_CALUDE_ray_walks_to_highschool_l1134_113423


namespace NUMINAMATH_CALUDE_cubic_roots_cubed_l1134_113479

/-- Given a cubic equation x³ + ax² + bx + c = 0 with roots α, β, and γ,
    the cubic equation whose roots are α³, β³, and γ³ is
    x³ + (-a³ + 3ab - 3c)x² + (-b³ + 3abc)x + c³ = 0 -/
theorem cubic_roots_cubed (a b c : ℝ) (α β γ : ℝ) : 
  (∀ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ↔ x = α ∨ x = β ∨ x = γ) →
  (∀ x : ℝ, x^3 + (-a^3 + 3*a*b - 3*c)*x^2 + (-b^3 + 3*a*b*c)*x + c^3 = 0 
           ↔ x = α^3 ∨ x = β^3 ∨ x = γ^3) :=
by sorry


end NUMINAMATH_CALUDE_cubic_roots_cubed_l1134_113479


namespace NUMINAMATH_CALUDE_x_plus_y_value_l1134_113466

theorem x_plus_y_value (x y : ℝ) 
  (h1 : |x| - 2*x + y = 1) 
  (h2 : x - |y| + y = 8) : 
  x + y = 17 ∨ x + y = 1 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l1134_113466


namespace NUMINAMATH_CALUDE_distance_difference_l1134_113470

/-- The width of the streets in Longtown -/
def street_width : ℝ := 30

/-- The length of the longer side of the block -/
def block_length : ℝ := 500

/-- The length of the shorter side of the block -/
def block_width : ℝ := 300

/-- The distance Jenny runs around the block -/
def jenny_distance : ℝ := 2 * (block_length + block_width)

/-- The distance Jeremy runs around the block -/
def jeremy_distance : ℝ := 2 * ((block_length + 2 * street_width) + (block_width + 2 * street_width))

/-- Theorem stating the difference in distance run by Jeremy and Jenny -/
theorem distance_difference : jeremy_distance - jenny_distance = 240 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l1134_113470


namespace NUMINAMATH_CALUDE_complex_absolute_value_l1134_113452

theorem complex_absolute_value (z : ℂ) (h : (3 - I) / (z - 3*I) = 1 + I) : 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l1134_113452


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1134_113485

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n, a n > 0) →
  a 1 = 1 →
  a 5 = 16 →
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1134_113485


namespace NUMINAMATH_CALUDE_f_3_range_l1134_113480

theorem f_3_range (a c : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = a * x^2 - c)
  (h1 : -4 ≤ f 1 ∧ f 1 ≤ -1) (h2 : -1 ≤ f 2 ∧ f 2 ≤ 5) :
  -1 ≤ f 3 ∧ f 3 ≤ 20 := by
sorry

end NUMINAMATH_CALUDE_f_3_range_l1134_113480


namespace NUMINAMATH_CALUDE_eulers_pedal_triangle_theorem_l1134_113486

/-- Euler's theorem on the area of pedal triangles -/
theorem eulers_pedal_triangle_theorem (S R d : ℝ) (hR : R > 0) : 
  ∃ (S' : ℝ), S' = (S / 4) * |1 - (d^2 / R^2)| := by
  sorry

end NUMINAMATH_CALUDE_eulers_pedal_triangle_theorem_l1134_113486


namespace NUMINAMATH_CALUDE_eric_pencils_l1134_113422

/-- The number of boxes of pencils Eric has -/
def num_boxes : ℕ := 12

/-- The number of pencils in each box -/
def pencils_per_box : ℕ := 17

/-- The total number of pencils Eric has -/
def total_pencils : ℕ := num_boxes * pencils_per_box

theorem eric_pencils : total_pencils = 204 := by
  sorry

end NUMINAMATH_CALUDE_eric_pencils_l1134_113422


namespace NUMINAMATH_CALUDE_staircase_problem_l1134_113414

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def jumps (step_size : ℕ) (total_steps : ℕ) : ℕ := 
  (total_steps + step_size - 1) / step_size

theorem staircase_problem (n : ℕ) : 
  is_prime n → 
  jumps 3 n - jumps 6 n = 25 → 
  ∃ m : ℕ, is_prime m ∧ 
           jumps 3 m - jumps 6 m = 25 ∧ 
           n + m = 300 :=
sorry

end NUMINAMATH_CALUDE_staircase_problem_l1134_113414


namespace NUMINAMATH_CALUDE_chocolate_bar_breaks_l1134_113419

/-- Represents a rectangular chocolate bar -/
structure ChocolateBar where
  rows : ℕ
  cols : ℕ

/-- Calculates the minimum number of breaks required to separate a chocolate bar into individual pieces -/
def min_breaks (bar : ChocolateBar) : ℕ :=
  (bar.rows - 1) * bar.cols + (bar.cols - 1)

theorem chocolate_bar_breaks (bar : ChocolateBar) (h1 : bar.rows = 5) (h2 : bar.cols = 8) :
  min_breaks bar = 39 := by
  sorry

#eval min_breaks ⟨5, 8⟩

end NUMINAMATH_CALUDE_chocolate_bar_breaks_l1134_113419


namespace NUMINAMATH_CALUDE_mod_equivalence_l1134_113464

theorem mod_equivalence (m : ℕ) : 
  152 * 936 ≡ m [ZMOD 50] → 0 ≤ m → m < 50 → m = 22 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_l1134_113464


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l1134_113472

/-- The repeating decimal 0.565656... expressed as a rational number -/
def repeating_decimal : ℚ := 0.56565656

/-- The fraction 56/99 -/
def fraction : ℚ := 56 / 99

/-- Theorem stating that the repeating decimal 0.565656... is equal to the fraction 56/99 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l1134_113472


namespace NUMINAMATH_CALUDE_trajectory_of_point_l1134_113428

/-- The trajectory of a point M, given specific conditions -/
theorem trajectory_of_point (M : ℝ × ℝ) :
  (∀ (x y : ℝ), M = (x, y) →
    (x^2 + (y + 3)^2)^(1/2) = |y - 3|) →  -- M is equidistant from (0, -3) and y = 3
  (∃ (a b c : ℝ), ∀ (x y : ℝ), M = (x, y) → 
    a*x^2 + b*y + c = 0) →  -- Trajectory of M is a conic section (which includes parabolas)
  ∃ (x y : ℝ), M = (x, y) ∧ x^2 = -12*y :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_point_l1134_113428


namespace NUMINAMATH_CALUDE_word_transformations_l1134_113440

-- Define the alphabet
inductive Letter : Type
| x : Letter
| y : Letter
| t : Letter

-- Define a word as a list of letters
def Word := List Letter

-- Define the transformation rules
inductive Transform : Word → Word → Prop
| xy_yyx : Transform (Letter.x::Letter.y::w) (Letter.y::Letter.y::Letter.x::w)
| xt_ttx : Transform (Letter.x::Letter.t::w) (Letter.t::Letter.t::Letter.x::w)
| yt_ty  : Transform (Letter.y::Letter.t::w) (Letter.t::Letter.y::w)
| refl   : ∀ w, Transform w w
| symm   : ∀ v w, Transform v w → Transform w v
| trans  : ∀ u v w, Transform u v → Transform v w → Transform u w

-- Define the theorem
theorem word_transformations :
  (¬ ∃ (w : Word), Transform [Letter.x, Letter.y] [Letter.x, Letter.t]) ∧
  (¬ ∃ (w : Word), Transform [Letter.x, Letter.y, Letter.t, Letter.x] [Letter.t, Letter.x, Letter.y, Letter.t]) ∧
  (∃ (w : Word), Transform [Letter.x, Letter.t, Letter.x, Letter.y, Letter.y] [Letter.t, Letter.t, Letter.x, Letter.y, Letter.y, Letter.y, Letter.y, Letter.x])
  := by sorry

end NUMINAMATH_CALUDE_word_transformations_l1134_113440


namespace NUMINAMATH_CALUDE_largest_angle_60_degrees_l1134_113403

/-- 
Given a triangle ABC with side lengths a, b, and c satisfying the equation
a^2 + b^2 = c^2 - ab, the largest interior angle of the triangle is 60°.
-/
theorem largest_angle_60_degrees 
  (a b c : ℝ) 
  (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (eq : a^2 + b^2 = c^2 - a*b) : 
  ∃ θ : ℝ, θ ≤ 60 * π / 180 ∧ 
    θ = Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)) ∧
    θ = Real.arccos ((a^2 + c^2 - b^2) / (2*a*c)) ∧
    θ = Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)) :=
sorry

end NUMINAMATH_CALUDE_largest_angle_60_degrees_l1134_113403


namespace NUMINAMATH_CALUDE_units_digit_sum_base_8_l1134_113438

-- Define a function to get the units digit in base 8
def units_digit_base_8 (n : ℕ) : ℕ := n % 8

-- Define the numbers in base 8
def num1 : ℕ := 64
def num2 : ℕ := 34

-- Theorem statement
theorem units_digit_sum_base_8 :
  units_digit_base_8 (num1 + num2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_sum_base_8_l1134_113438


namespace NUMINAMATH_CALUDE_tent_setup_plans_l1134_113462

theorem tent_setup_plans : 
  let total_students : ℕ := 50
  let valid_setup (x y : ℕ) := 3 * x + 2 * y = total_students
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => valid_setup p.1 p.2) (Finset.product (Finset.range (total_students + 1)) (Finset.range (total_students + 1)))).card ∧ n = 8
  := by sorry

end NUMINAMATH_CALUDE_tent_setup_plans_l1134_113462


namespace NUMINAMATH_CALUDE_compare_expressions_l1134_113492

theorem compare_expressions (x : ℝ) (h : x > 1) : x^3 + 6*x > x^2 + 6 := by
  sorry

end NUMINAMATH_CALUDE_compare_expressions_l1134_113492


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l1134_113469

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 27) : 
  r - p = 34 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l1134_113469


namespace NUMINAMATH_CALUDE_train_trip_probability_l1134_113432

theorem train_trip_probability : ∀ (p₁ p₂ p₃ p₄ : ℝ),
  p₁ = 0.3 →
  p₂ = 0.1 →
  p₃ = 0.4 →
  p₁ + p₂ + p₃ + p₄ = 1 →
  p₄ = 0.2 := by
sorry

end NUMINAMATH_CALUDE_train_trip_probability_l1134_113432


namespace NUMINAMATH_CALUDE_cupcake_cost_split_l1134_113429

theorem cupcake_cost_split (num_cupcakes : ℕ) (price_per_cupcake : ℚ) (num_people : ℕ) :
  num_cupcakes = 12 →
  price_per_cupcake = 3/2 →
  num_people = 2 →
  (num_cupcakes : ℚ) * price_per_cupcake / num_people = 9 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_cost_split_l1134_113429


namespace NUMINAMATH_CALUDE_conference_theorem_l1134_113478

/-- Represents the state of knowledge among scientists at a conference --/
structure ConferenceState where
  total_scientists : Nat
  initial_knowers : Nat
  final_knowers : Nat

/-- Calculates the probability of a specific number of scientists knowing the news after pairing --/
noncomputable def probability_of_final_knowers (state : ConferenceState) : ℚ :=
  sorry

/-- Calculates the expected number of scientists knowing the news after pairing --/
noncomputable def expected_final_knowers (state : ConferenceState) : ℚ :=
  sorry

theorem conference_theorem (state : ConferenceState) 
  (h1 : state.total_scientists = 18) 
  (h2 : state.initial_knowers = 10) : 
  (probability_of_final_knowers {total_scientists := 18, initial_knowers := 10, final_knowers := 13} = 0) ∧ 
  (probability_of_final_knowers {total_scientists := 18, initial_knowers := 10, final_knowers := 14} = 1120/2431) ∧
  (expected_final_knowers {total_scientists := 18, initial_knowers := 10, final_knowers := 0} = 14 + 12/17) :=
by sorry

end NUMINAMATH_CALUDE_conference_theorem_l1134_113478


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1134_113417

theorem geometric_series_sum : 
  let a : ℝ := 2/3
  let r : ℝ := 2/3
  let series_sum : ℝ := ∑' i, a * r^(i - 1)
  series_sum = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1134_113417


namespace NUMINAMATH_CALUDE_painted_cells_count_l1134_113405

/-- Represents a rectangular grid with alternating painted rows and columns -/
structure PaintedGrid where
  rows : Nat
  cols : Nat
  unpainted_cells : Nat

/-- Calculates the number of painted cells in the grid -/
def painted_cells (grid : PaintedGrid) : Nat :=
  grid.rows * grid.cols - grid.unpainted_cells

theorem painted_cells_count (grid : PaintedGrid) : 
  grid.rows = 5 ∧ grid.cols = 75 ∧ grid.unpainted_cells = 74 → painted_cells grid = 301 := by
  sorry

#check painted_cells_count

end NUMINAMATH_CALUDE_painted_cells_count_l1134_113405


namespace NUMINAMATH_CALUDE_circle_intersection_range_l1134_113436

theorem circle_intersection_range (a : ℝ) : 
  (∃! (p q : ℝ × ℝ), p ≠ q ∧ 
    ((p.1 - a)^2 + (p.2 - a)^2 = 4) ∧
    ((q.1 - a)^2 + (q.2 - a)^2 = 4) ∧
    (p.1^2 + p.2^2 = 4) ∧
    (q.1^2 + q.2^2 = 4)) →
  (-2 * Real.sqrt 2 < a ∧ a < 2 * Real.sqrt 2 ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l1134_113436


namespace NUMINAMATH_CALUDE_sum_of_numbers_l1134_113408

theorem sum_of_numbers (x y : ℝ) : 
  y = 2 * x - 3 →
  y = 37 →
  x + y = 57 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l1134_113408


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l1134_113413

/-- The equation ({m-2}){x^{m^2-2}}+4x-7=0 is quadratic -/
def is_quadratic (m : ℝ) : Prop :=
  (m^2 - 2 = 2) ∧ (m - 2 ≠ 0)

theorem quadratic_equation_m_value :
  ∀ m : ℝ, is_quadratic m → m = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l1134_113413


namespace NUMINAMATH_CALUDE_min_clicks_to_one_color_l1134_113495

/-- Represents a chessboard -/
def Chessboard := Fin 98 → Fin 98 → Bool

/-- Represents a rectangle on the chessboard -/
structure Rectangle where
  top_left : Fin 98 × Fin 98
  bottom_right : Fin 98 × Fin 98

/-- Applies a click to a rectangle on the chessboard -/
def applyClick (board : Chessboard) (rect : Rectangle) : Chessboard :=
  sorry

/-- Checks if the entire board is one color -/
def isOneColor (board : Chessboard) : Bool :=
  sorry

/-- Initial chessboard with alternating colors -/
def initialBoard : Chessboard :=
  sorry

/-- Theorem: The minimum number of clicks to make the chessboard one color is 98 -/
theorem min_clicks_to_one_color :
  ∀ (clicks : List Rectangle),
    isOneColor (clicks.foldl applyClick initialBoard) →
    clicks.length ≥ 98 :=
  sorry

end NUMINAMATH_CALUDE_min_clicks_to_one_color_l1134_113495


namespace NUMINAMATH_CALUDE_friend_walking_rates_l1134_113453

theorem friend_walking_rates (trail_length : ℝ) (p_distance : ℝ) 
  (hp : trail_length = 33)
  (hpd : p_distance = 18) :
  let q_distance := trail_length - p_distance
  let rate_ratio := p_distance / q_distance
  (rate_ratio - 1) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_friend_walking_rates_l1134_113453


namespace NUMINAMATH_CALUDE_min_time_for_given_problem_l1134_113412

/-- Represents the chef's cooking problem -/
structure ChefProblem where
  total_potatoes : ℕ
  cooked_potatoes : ℕ
  cooking_time_per_potato : ℕ
  salad_prep_time : ℕ

/-- Calculates the minimum time needed to complete the cooking task -/
def min_time_needed (problem : ChefProblem) : ℕ :=
  max problem.salad_prep_time (problem.cooking_time_per_potato)

/-- Theorem stating the minimum time needed for the given problem -/
theorem min_time_for_given_problem :
  let problem : ChefProblem := {
    total_potatoes := 35,
    cooked_potatoes := 11,
    cooking_time_per_potato := 7,
    salad_prep_time := 15
  }
  min_time_needed problem = 15 := by sorry

end NUMINAMATH_CALUDE_min_time_for_given_problem_l1134_113412


namespace NUMINAMATH_CALUDE_equal_sides_implies_rhombus_l1134_113411

/-- A quadrilateral is a polygon with four sides -/
structure Quadrilateral where
  sides : Fin 4 → ℝ
  positive : ∀ i, sides i > 0

/-- A rhombus is a quadrilateral with all sides of equal length -/
def is_rhombus (q : Quadrilateral) : Prop :=
  ∀ i j, q.sides i = q.sides j

/-- Theorem: A quadrilateral with all sides of equal length is a rhombus -/
theorem equal_sides_implies_rhombus (q : Quadrilateral) 
  (h : ∀ i j, q.sides i = q.sides j) : is_rhombus q := by
  sorry

end NUMINAMATH_CALUDE_equal_sides_implies_rhombus_l1134_113411


namespace NUMINAMATH_CALUDE_friend_team_assignment_count_l1134_113407

-- Define the number of friends and teams
def num_friends : ℕ := 6
def num_teams : ℕ := 4

-- Theorem statement
theorem friend_team_assignment_count :
  (num_teams ^ num_friends : ℕ) = 4096 := by
  sorry

end NUMINAMATH_CALUDE_friend_team_assignment_count_l1134_113407


namespace NUMINAMATH_CALUDE_candle_ratio_l1134_113410

/-- Proves that the ratio of candles in Kalani's bedroom to candles in the living room is 2:1 -/
theorem candle_ratio :
  ∀ (bedroom_candles living_room_candles donovan_candles total_candles : ℕ),
    bedroom_candles = 20 →
    donovan_candles = 20 →
    total_candles = 50 →
    bedroom_candles + living_room_candles + donovan_candles = total_candles →
    (bedroom_candles : ℚ) / living_room_candles = 2 := by
  sorry

end NUMINAMATH_CALUDE_candle_ratio_l1134_113410


namespace NUMINAMATH_CALUDE_motorcycle_journey_time_ratio_l1134_113418

/-- Proves that the time taken to travel from A to B is 2 times the time taken to travel from B to C -/
theorem motorcycle_journey_time_ratio :
  ∀ (total_distance AB_distance BC_distance average_speed : ℝ),
  total_distance = 180 →
  AB_distance = 120 →
  BC_distance = 60 →
  average_speed = 20 →
  AB_distance = 2 * BC_distance →
  ∃ (AB_time BC_time : ℝ),
    AB_time > 0 ∧ BC_time > 0 ∧
    AB_time + BC_time = total_distance / average_speed ∧
    AB_time = AB_distance / average_speed ∧
    BC_time = BC_distance / average_speed ∧
    AB_time = 2 * BC_time :=
by sorry

end NUMINAMATH_CALUDE_motorcycle_journey_time_ratio_l1134_113418


namespace NUMINAMATH_CALUDE_sqrt_25_equals_5_l1134_113487

theorem sqrt_25_equals_5 : Real.sqrt 25 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_25_equals_5_l1134_113487


namespace NUMINAMATH_CALUDE_geometric_sequence_iff_square_middle_l1134_113400

/-- A sequence (a, b, c) is geometric if there exists a common ratio r such that b = ar and c = br. -/
def IsGeometricSequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

/-- The main theorem stating that b² = ac is necessary and sufficient for (a, b, c) to form a geometric sequence. -/
theorem geometric_sequence_iff_square_middle (a b c : ℝ) :
  IsGeometricSequence a b c ↔ b^2 = a * c :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_iff_square_middle_l1134_113400


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1134_113471

theorem sum_of_squares_of_roots (a b c : ℝ) : 
  (a^3 - 2*a^2 + 5*a + 7 = 0) → 
  (b^3 - 2*b^2 + 5*b + 7 = 0) → 
  (c^3 - 2*c^2 + 5*c + 7 = 0) → 
  a^2 + b^2 + c^2 = -6 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1134_113471


namespace NUMINAMATH_CALUDE_expand_product_l1134_113421

theorem expand_product (x y : ℝ) : 4 * (x + 3) * (x + 2 + y) = 4 * x^2 + 4 * x * y + 20 * x + 12 * y + 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1134_113421


namespace NUMINAMATH_CALUDE_midpoint_on_number_line_l1134_113458

theorem midpoint_on_number_line (A B C : ℝ) : 
  A = -7 → 
  |B - A| = 5 → 
  C = (A + B) / 2 → 
  C = -9/2 ∨ C = -19/2 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_on_number_line_l1134_113458


namespace NUMINAMATH_CALUDE_z_value_l1134_113439

theorem z_value (x y z : ℝ) (h : (x + 1)⁻¹ + (y + 1)⁻¹ = z⁻¹) : 
  z = ((x + 1) * (y + 1)) / (x + y + 2) := by
  sorry

end NUMINAMATH_CALUDE_z_value_l1134_113439


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l1134_113444

theorem simplify_sqrt_expression : 
  Real.sqrt 5 - Real.sqrt 20 + Real.sqrt 45 = 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l1134_113444


namespace NUMINAMATH_CALUDE_product_zero_implies_factor_zero_l1134_113401

theorem product_zero_implies_factor_zero (a b c : ℝ) : a * b * c = 0 → (a = 0 ∨ b = 0 ∨ c = 0) := by
  sorry

end NUMINAMATH_CALUDE_product_zero_implies_factor_zero_l1134_113401


namespace NUMINAMATH_CALUDE_intersection_area_l1134_113468

/-- A regular cube with side length 2 units -/
structure Cube where
  side_length : ℝ
  is_regular : side_length = 2

/-- A plane that cuts the cube -/
structure IntersectingPlane where
  parallel_to_face : Bool
  at_middle : Bool

/-- The polygon formed by the intersection of the plane and the cube -/
def intersection_polygon (c : Cube) (p : IntersectingPlane) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem intersection_area (c : Cube) (p : IntersectingPlane) :
  p.parallel_to_face ∧ p.at_middle →
  area (intersection_polygon c p) = 4 :=
sorry

end NUMINAMATH_CALUDE_intersection_area_l1134_113468


namespace NUMINAMATH_CALUDE_tan_sqrt3_sin_equality_l1134_113499

theorem tan_sqrt3_sin_equality : (Real.tan (10 * π / 180) - Real.sqrt 3) * Real.sin (40 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sqrt3_sin_equality_l1134_113499


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1134_113483

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (36 - a) + b / (45 - b) + c / (54 - c) = 8) : 
  4 / (36 - a) + 5 / (45 - b) + 6 / (54 - c) = 11 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1134_113483


namespace NUMINAMATH_CALUDE_total_cost_l1134_113441

/-- Represents the price of an enchilada in dollars -/
def enchilada_price : ℚ := sorry

/-- Represents the price of a taco in dollars -/
def taco_price : ℚ := sorry

/-- Represents the price of a drink in dollars -/
def drink_price : ℚ := sorry

/-- The first price condition: one enchilada, two tacos, and a drink cost $3.20 -/
axiom price_condition1 : enchilada_price + 2 * taco_price + drink_price = 32/10

/-- The second price condition: two enchiladas, three tacos, and a drink cost $4.90 -/
axiom price_condition2 : 2 * enchilada_price + 3 * taco_price + drink_price = 49/10

/-- Theorem stating that the cost of four enchiladas, five tacos, and two drinks is $8.30 -/
theorem total_cost : 4 * enchilada_price + 5 * taco_price + 2 * drink_price = 83/10 := by sorry

end NUMINAMATH_CALUDE_total_cost_l1134_113441


namespace NUMINAMATH_CALUDE_price_increase_achieves_target_profit_l1134_113497

/-- Represents the price increase in yuan -/
def price_increase : ℝ := 5

/-- Initial profit per kilogram in yuan -/
def initial_profit_per_kg : ℝ := 10

/-- Initial daily sales volume in kilograms -/
def initial_sales_volume : ℝ := 500

/-- Decrease in sales volume per yuan of price increase -/
def sales_volume_decrease_rate : ℝ := 20

/-- Target daily profit in yuan -/
def target_daily_profit : ℝ := 6000

/-- Theorem stating that the given price increase achieves the target daily profit -/
theorem price_increase_achieves_target_profit :
  (initial_sales_volume - sales_volume_decrease_rate * price_increase) *
  (initial_profit_per_kg + price_increase) = target_daily_profit :=
by sorry

end NUMINAMATH_CALUDE_price_increase_achieves_target_profit_l1134_113497


namespace NUMINAMATH_CALUDE_equation_proof_l1134_113402

theorem equation_proof : Real.sqrt (3^2 + 4^2) / Real.sqrt (25 - 1) = 5 * Real.sqrt 6 / 12 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l1134_113402


namespace NUMINAMATH_CALUDE_solve_equation_l1134_113437

theorem solve_equation (a : ℚ) (h : 3 * a + 6 * a / 4 = 6) : a = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1134_113437


namespace NUMINAMATH_CALUDE_gasoline_expense_gasoline_expense_proof_l1134_113463

/-- Calculates the amount spent on gasoline given the initial amount, known expenses, money received, and the amount left for the return trip. -/
theorem gasoline_expense (initial_amount : ℝ) (lunch_expense : ℝ) (gift_expense : ℝ) 
  (money_from_grandma : ℝ) (return_trip_money : ℝ) : ℝ :=
  let total_amount := initial_amount + money_from_grandma
  let known_expenses := lunch_expense + gift_expense
  let remaining_after_known_expenses := total_amount - known_expenses
  remaining_after_known_expenses - return_trip_money

/-- Proves that the amount spent on gasoline is $8 given the specific values from the problem. -/
theorem gasoline_expense_proof :
  gasoline_expense 50 15.65 10 20 36.35 = 8 := by
  sorry

end NUMINAMATH_CALUDE_gasoline_expense_gasoline_expense_proof_l1134_113463


namespace NUMINAMATH_CALUDE_candy_bar_distribution_l1134_113445

theorem candy_bar_distribution (total_bars : ℕ) (spare_bars : ℕ) (num_friends : ℕ) 
  (h1 : total_bars = 24)
  (h2 : spare_bars = 10)
  (h3 : num_friends = 7)
  : (total_bars - spare_bars) / num_friends = 2 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_distribution_l1134_113445


namespace NUMINAMATH_CALUDE_g_at_negative_three_l1134_113435

def g (x : ℝ) : ℝ := 3 * x^5 - 5 * x^4 + 7 * x^3 - 10 * x^2 - 12 * x + 36

theorem g_at_negative_three : g (-3) = -1341 := by sorry

end NUMINAMATH_CALUDE_g_at_negative_three_l1134_113435


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1134_113498

theorem geometric_series_sum : 
  let a : ℚ := 1/5
  let r : ℚ := -1/3
  let n : ℕ := 7
  let series_sum := a * (1 - r^n) / (1 - r)
  series_sum = 1641/10935 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1134_113498


namespace NUMINAMATH_CALUDE_probability_qualified_product_l1134_113465

/-- The proportion of the first batch in the total mix -/
def batch1_proportion : ℝ := 0.30

/-- The proportion of the second batch in the total mix -/
def batch2_proportion : ℝ := 0.70

/-- The defect rate of the first batch -/
def batch1_defect_rate : ℝ := 0.05

/-- The defect rate of the second batch -/
def batch2_defect_rate : ℝ := 0.04

/-- The probability of selecting a qualified product from the mixed batches -/
theorem probability_qualified_product : 
  batch1_proportion * (1 - batch1_defect_rate) + batch2_proportion * (1 - batch2_defect_rate) = 0.957 := by
  sorry

end NUMINAMATH_CALUDE_probability_qualified_product_l1134_113465


namespace NUMINAMATH_CALUDE_additional_investment_rate_problem_l1134_113491

/-- Calculates the rate of additional investment needed to achieve a target total rate --/
def additional_investment_rate (initial_investment : ℚ) (initial_rate : ℚ) 
  (additional_investment : ℚ) (target_total_rate : ℚ) : ℚ :=
  let total_investment := initial_investment + additional_investment
  let initial_interest := initial_investment * initial_rate
  let total_desired_interest := total_investment * target_total_rate
  let additional_interest_needed := total_desired_interest - initial_interest
  additional_interest_needed / additional_investment

theorem additional_investment_rate_problem 
  (initial_investment : ℚ) 
  (initial_rate : ℚ) 
  (additional_investment : ℚ) 
  (target_total_rate : ℚ) :
  initial_investment = 8000 →
  initial_rate = 5 / 100 →
  additional_investment = 4000 →
  target_total_rate = 6 / 100 →
  additional_investment_rate initial_investment initial_rate additional_investment target_total_rate = 8 / 100 :=
by
  sorry

end NUMINAMATH_CALUDE_additional_investment_rate_problem_l1134_113491


namespace NUMINAMATH_CALUDE_papaya_problem_l1134_113477

def remaining_green_papayas (initial : Nat) (friday_yellow : Nat) : Nat :=
  initial - friday_yellow - (2 * friday_yellow)

theorem papaya_problem :
  remaining_green_papayas 14 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_papaya_problem_l1134_113477


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l1134_113474

theorem sqrt_sum_inequality : Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l1134_113474


namespace NUMINAMATH_CALUDE_divisibility_theorem_l1134_113455

theorem divisibility_theorem (a b c d e f : ℤ) 
  (h : (13 : ℤ) ∣ (a^12 + b^12 + c^12 + d^12 + e^12 + f^12)) : 
  (13^6 : ℤ) ∣ (a * b * c * d * e * f) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l1134_113455


namespace NUMINAMATH_CALUDE_pizza_order_count_l1134_113493

theorem pizza_order_count (slices_per_pizza : ℕ) (total_slices : ℕ) (h1 : slices_per_pizza = 8) (h2 : total_slices = 168) :
  total_slices / slices_per_pizza = 21 := by
  sorry

end NUMINAMATH_CALUDE_pizza_order_count_l1134_113493


namespace NUMINAMATH_CALUDE_sine_angle_plus_pi_half_l1134_113448

theorem sine_angle_plus_pi_half (α : Real) : 
  (∃ r : Real, r > 0 ∧ -1 = r * Real.cos α ∧ Real.sqrt 3 = r * Real.sin α) →
  Real.sin (α + π/2) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_sine_angle_plus_pi_half_l1134_113448


namespace NUMINAMATH_CALUDE_production_increase_l1134_113459

def planned_daily_production : ℕ := 500

def daily_changes : List ℤ := [40, -30, 90, -50, -20, -10, 20]

def actual_daily_production : List ℕ := 
  List.scanl (λ acc change => (acc : ℤ) + change |>.toNat) planned_daily_production daily_changes

def total_actual_production : ℕ := actual_daily_production.sum

def total_planned_production : ℕ := planned_daily_production * 7

theorem production_increase :
  total_actual_production = 3790 ∧ total_actual_production > total_planned_production :=
by sorry

end NUMINAMATH_CALUDE_production_increase_l1134_113459


namespace NUMINAMATH_CALUDE_base_k_equality_l1134_113488

theorem base_k_equality (k : ℕ) : k^2 + 3*k + 2 = 30 → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_k_equality_l1134_113488


namespace NUMINAMATH_CALUDE_fundraising_goal_l1134_113406

/-- Fundraising goal calculation for a school's community outreach program -/
theorem fundraising_goal (families_20 families_10 families_5 : ℕ) 
  (donation_20 donation_10 donation_5 : ℕ) (additional_needed : ℕ) : 
  families_20 = 2 → 
  families_10 = 8 → 
  families_5 = 10 → 
  donation_20 = 20 → 
  donation_10 = 10 → 
  donation_5 = 5 → 
  additional_needed = 30 → 
  families_20 * donation_20 + families_10 * donation_10 + families_5 * donation_5 + additional_needed = 200 := by
sorry

#eval 2 * 20 + 8 * 10 + 10 * 5 + 30

end NUMINAMATH_CALUDE_fundraising_goal_l1134_113406


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_zero_l1134_113484

theorem ceiling_floor_sum_zero : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_zero_l1134_113484


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l1134_113431

theorem system_of_equations_solution (x y z : ℝ) : 
  x + 3*y = 4*y^3 ∧ 
  y + 3*z = 4*z^3 ∧ 
  z + 3*x = 4*x^3 →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = 1 ∧ y = 1 ∧ z = 1) ∨
  (x = -1 ∧ y = -1 ∧ z = -1) ∨
  (x = Real.cos (π/14) ∧ y = -Real.cos (5*π/14) ∧ z = Real.cos (3*π/14)) ∨
  (x = -Real.cos (π/14) ∧ y = Real.cos (5*π/14) ∧ z = -Real.cos (3*π/14)) ∨
  (x = Real.cos (π/7) ∧ y = -Real.cos (2*π/7) ∧ z = Real.cos (3*π/7)) ∨
  (x = -Real.cos (π/7) ∧ y = Real.cos (2*π/7) ∧ z = -Real.cos (3*π/7)) ∨
  (x = Real.cos (π/13) ∧ y = -Real.cos (π/13) ∧ z = Real.cos (3*π/13)) ∨
  (x = -Real.cos (π/13) ∧ y = Real.cos (π/13) ∧ z = -Real.cos (3*π/13)) :=
by sorry


end NUMINAMATH_CALUDE_system_of_equations_solution_l1134_113431


namespace NUMINAMATH_CALUDE_square_difference_l1134_113433

theorem square_difference (x y z w : ℝ) 
  (sum_xy : x + y = 10)
  (diff_xy : x - y = 8)
  (sum_yz : y + z = 15)
  (sum_zw : z + w = 20) :
  x^2 - w^2 = 45 := by sorry

end NUMINAMATH_CALUDE_square_difference_l1134_113433


namespace NUMINAMATH_CALUDE_machine_value_after_two_years_l1134_113457

def initial_value : ℝ := 8000
def depreciation_rate : ℝ := 0.15

def market_value_after_two_years (initial : ℝ) (rate : ℝ) : ℝ :=
  initial * (1 - rate) * (1 - rate)

theorem machine_value_after_two_years :
  market_value_after_two_years initial_value depreciation_rate = 5780 := by
  sorry

end NUMINAMATH_CALUDE_machine_value_after_two_years_l1134_113457


namespace NUMINAMATH_CALUDE_complex_symmetry_product_l1134_113490

theorem complex_symmetry_product (z₁ z₂ : ℂ) : 
  (z₁.re = -z₂.re ∧ z₁.im = z₂.im) →  -- symmetry about imaginary axis
  z₁ = 1 + 2*I →                     -- given value of z₁
  z₁ * z₂ = -5 :=                    -- product equals -5
by
  sorry

end NUMINAMATH_CALUDE_complex_symmetry_product_l1134_113490


namespace NUMINAMATH_CALUDE_estimate_pi_l1134_113460

theorem estimate_pi (total_beans : ℕ) (beans_in_circle : ℕ) 
  (h1 : total_beans = 80) (h2 : beans_in_circle = 64) : 
  (4 * beans_in_circle : ℝ) / total_beans = 3.2 :=
sorry

end NUMINAMATH_CALUDE_estimate_pi_l1134_113460


namespace NUMINAMATH_CALUDE_max_c_magnitude_l1134_113449

theorem max_c_magnitude (a b c : ℝ × ℝ) : 
  (‖a‖ = 1) → 
  (‖b‖ = 1) → 
  (a • b = 1/2) → 
  (‖a - b + c‖ ≤ 1) →
  (∃ (c : ℝ × ℝ), ‖c‖ = 2) ∧ 
  (∀ (c : ℝ × ℝ), ‖a - b + c‖ ≤ 1 → ‖c‖ ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_max_c_magnitude_l1134_113449


namespace NUMINAMATH_CALUDE_cos_360_degrees_l1134_113461

theorem cos_360_degrees : Real.cos (2 * Real.pi) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_360_degrees_l1134_113461


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1134_113494

-- Define the sets A and B
def A : Set ℝ := {x | 3*x^2 - 14*x + 16 ≤ 0}
def B : Set ℝ := {x | (3*x - 7) / x > 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x | 7/3 < x ∧ x ≤ 8/3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1134_113494


namespace NUMINAMATH_CALUDE_max_distance_between_paths_l1134_113404

theorem max_distance_between_paths : 
  ∃ (C : ℝ), C = 3 * Real.sqrt 3 ∧ 
  ∀ (t : ℝ), 
    Real.sqrt ((t - (t - 5))^2 + (Real.sin t - Real.cos (t - 5))^2) ≤ C :=
sorry

end NUMINAMATH_CALUDE_max_distance_between_paths_l1134_113404


namespace NUMINAMATH_CALUDE_point_above_line_l1134_113473

/-- A point in the 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The line y = x -/
def line_y_eq_x : Set Point2D := {p : Point2D | p.y = p.x}

/-- The region above the line y = x -/
def region_above_line : Set Point2D := {p : Point2D | p.y > p.x}

/-- Theorem: Any point M(x, y) where y > x is located in the region above the line y = x -/
theorem point_above_line (M : Point2D) (h : M.y > M.x) : M ∈ region_above_line := by
  sorry

end NUMINAMATH_CALUDE_point_above_line_l1134_113473


namespace NUMINAMATH_CALUDE_inverse_proportion_l1134_113482

/-- Given that α is inversely proportional to β, prove that when α = 5 for β = 10, 
    then α = 25/2 for β = 4 -/
theorem inverse_proportion (α β : ℝ) (k : ℝ) (h1 : α * β = k) 
    (h2 : 5 * 10 = k) : 
  4 * (25/2 : ℝ) = k := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_l1134_113482


namespace NUMINAMATH_CALUDE_smallest_divisible_by_all_is_divisible_by_all_168_smallest_number_of_books_l1134_113420

def is_divisible_by_all (n : ℕ) : Prop :=
  n % 3 = 0 ∧ n % 4 = 0 ∧ n % 6 = 0 ∧ n % 7 = 0 ∧ n % 8 = 0

theorem smallest_divisible_by_all :
  ∀ m : ℕ, m > 0 → is_divisible_by_all m → m ≥ 168 :=
by sorry

theorem is_divisible_by_all_168 : is_divisible_by_all 168 :=
by sorry

theorem smallest_number_of_books : 
  ∃! n : ℕ, n > 0 ∧ is_divisible_by_all n ∧ ∀ m : ℕ, m > 0 → is_divisible_by_all m → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_all_is_divisible_by_all_168_smallest_number_of_books_l1134_113420


namespace NUMINAMATH_CALUDE_f_plus_g_equals_one_l1134_113467

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- State the theorem
theorem f_plus_g_equals_one
  (h1 : is_even f)
  (h2 : is_odd g)
  (h3 : ∀ x, f x - g x = x^3 + x^2 + 1) :
  f 1 + g 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_plus_g_equals_one_l1134_113467


namespace NUMINAMATH_CALUDE_number_of_gyms_l1134_113409

def number_of_bikes_per_gym : ℕ := 10
def number_of_treadmills_per_gym : ℕ := 5
def number_of_ellipticals_per_gym : ℕ := 5

def cost_of_bike : ℕ := 700
def cost_of_treadmill : ℕ := cost_of_bike + cost_of_bike / 2
def cost_of_elliptical : ℕ := 2 * cost_of_treadmill

def total_replacement_cost : ℕ := 455000

def cost_per_gym : ℕ := 
  number_of_bikes_per_gym * cost_of_bike +
  number_of_treadmills_per_gym * cost_of_treadmill +
  number_of_ellipticals_per_gym * cost_of_elliptical

theorem number_of_gyms : 
  total_replacement_cost / cost_per_gym = 20 := by sorry

end NUMINAMATH_CALUDE_number_of_gyms_l1134_113409


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l1134_113496

theorem binomial_coefficient_problem (n : ℕ) (a b : ℝ) :
  (2 * n.choose 1 = 8) →
  n.choose 2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l1134_113496


namespace NUMINAMATH_CALUDE_perpendicular_lines_l1134_113427

def line1 (x y : ℝ) : Prop := 3 * x - y = 6

def line2 (x y : ℝ) : Prop := y = -1/3 * x + 7/3

def perpendicular (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ m1 m2 : ℝ, (∀ x y, f x y ↔ y = m1 * x + 0) ∧ 
              (∀ x y, g x y ↔ y = m2 * x + 0) ∧
              m1 * m2 = -1

theorem perpendicular_lines :
  perpendicular line1 line2 ∧ line2 (-2) 3 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l1134_113427


namespace NUMINAMATH_CALUDE_max_parts_formula_initial_values_correct_l1134_113489

/-- The maximum number of parts that n ellipses can divide a plane into -/
def max_parts (n : ℕ+) : ℕ :=
  2 * n.val * n.val - 2 * n.val + 2

/-- Theorem stating the formula for the maximum number of parts -/
theorem max_parts_formula (n : ℕ+) : max_parts n = 2 * n.val * n.val - 2 * n.val + 2 := by
  sorry

/-- The first few values of the sequence are correct -/
theorem initial_values_correct :
  max_parts 1 = 2 ∧ max_parts 2 = 6 ∧ max_parts 3 = 14 ∧ max_parts 4 = 26 := by
  sorry

end NUMINAMATH_CALUDE_max_parts_formula_initial_values_correct_l1134_113489


namespace NUMINAMATH_CALUDE_no_solution_cube_equation_mod_9_l1134_113475

theorem no_solution_cube_equation_mod_9 :
  ∀ (x y z : ℤ), (x^3 + y^3) % 9 ≠ (z^3 + 4) % 9 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_cube_equation_mod_9_l1134_113475


namespace NUMINAMATH_CALUDE_passes_count_is_32_l1134_113451

/-- Represents a swimmer in the pool --/
structure Swimmer where
  speed : ℝ
  startPosition : ℝ

/-- Represents the pool and swimming scenario --/
structure SwimmingScenario where
  poolLength : ℝ
  swimmer1 : Swimmer
  swimmer2 : Swimmer
  totalTime : ℝ

/-- Calculates the number of times swimmers pass each other --/
def countPasses (scenario : SwimmingScenario) : ℕ :=
  sorry

/-- The specific swimming scenario from the problem --/
def problemScenario : SwimmingScenario :=
  { poolLength := 100
    swimmer1 := { speed := 4, startPosition := 0 }
    swimmer2 := { speed := 3, startPosition := 100 }
    totalTime := 720 }

/-- Theorem stating that the number of passes in the given scenario is 32 --/
theorem passes_count_is_32 : countPasses problemScenario = 32 :=
  sorry

end NUMINAMATH_CALUDE_passes_count_is_32_l1134_113451


namespace NUMINAMATH_CALUDE_second_number_value_l1134_113442

theorem second_number_value (x y z : ℝ) 
  (sum_eq : x + y + z = 120) 
  (ratio_xy : x / y = 3 / 4) 
  (ratio_yz : y / z = 4 / 7) : 
  y = 34 := by sorry

end NUMINAMATH_CALUDE_second_number_value_l1134_113442


namespace NUMINAMATH_CALUDE_parabola_equation_l1134_113434

/-- Theorem: For a parabola y² = 2px where p > 0, if a line passing through its focus
    intersects the parabola at two points P(x₁, y₁) and Q(x₂, y₂) such that x₁ + x₂ = 2
    and |PQ| = 4, then the equation of the parabola is y² = 4x. -/
theorem parabola_equation (p : ℝ) (x₁ x₂ y₁ y₂ : ℝ) :
  p > 0 →
  y₁^2 = 2*p*x₁ →
  y₂^2 = 2*p*x₂ →
  x₁ + x₂ = 2 →
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16 →
  ∀ x y, y^2 = 2*p*x → y^2 = 4*x :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1134_113434


namespace NUMINAMATH_CALUDE_shoes_price_calculation_l1134_113446

/-- The price of shoes after a markup followed by a discount -/
def monday_price (thursday_price : ℝ) (friday_markup : ℝ) (monday_discount : ℝ) : ℝ :=
  thursday_price * (1 + friday_markup) * (1 - monday_discount)

/-- Theorem stating that the Monday price is $50.60 given the specified conditions -/
theorem shoes_price_calculation :
  monday_price 50 0.15 0.12 = 50.60 := by
  sorry

#eval monday_price 50 0.15 0.12

end NUMINAMATH_CALUDE_shoes_price_calculation_l1134_113446


namespace NUMINAMATH_CALUDE_num_correct_statements_is_zero_l1134_113425

/-- Represents a programming statement --/
inductive Statement
  | Input (vars : List String)
  | Output (expr : String)
  | Assignment (lhs : String) (rhs : String)

/-- Checks if an input statement is correct --/
def isValidInput (s : Statement) : Bool :=
  match s with
  | Statement.Input vars => vars.length > 0 && vars.all (fun v => v.length > 0)
  | _ => false

/-- Checks if an output statement is correct --/
def isValidOutput (s : Statement) : Bool :=
  match s with
  | Statement.Output expr => expr.startsWith "PRINT"
  | _ => false

/-- Checks if an assignment statement is correct --/
def isValidAssignment (s : Statement) : Bool :=
  match s with
  | Statement.Assignment lhs rhs => lhs.length > 0 && !lhs.toList.head!.isDigit && !rhs.contains '='
  | _ => false

/-- Checks if a statement is correct --/
def isValidStatement (s : Statement) : Bool :=
  isValidInput s || isValidOutput s || isValidAssignment s

/-- The list of statements to check --/
def statements : List Statement :=
  [Statement.Input ["a;", "b;", "c"],
   Statement.Output "A=4",
   Statement.Assignment "3" "B",
   Statement.Assignment "A" "B=-2"]

/-- Theorem: The number of correct statements is 0 --/
theorem num_correct_statements_is_zero : 
  (statements.filter isValidStatement).length = 0 := by
  sorry


end NUMINAMATH_CALUDE_num_correct_statements_is_zero_l1134_113425


namespace NUMINAMATH_CALUDE_intersection_of_sets_l1134_113415

theorem intersection_of_sets : 
  let A : Set ℕ := {2, 3, 4}
  let B : Set ℕ := {1, 2, 3}
  A ∩ B = {2, 3} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l1134_113415


namespace NUMINAMATH_CALUDE_triple_sum_diverges_l1134_113430

open Real BigOperators

theorem triple_sum_diverges :
  let f (m n k : ℕ) := (1 : ℝ) / (m * (m + n + k) * (n + 1))
  ∃ (S : ℝ), ∀ (M N K : ℕ), (∑ m in Finset.range M, ∑ n in Finset.range N, ∑ k in Finset.range K, f m n k) ≤ S
  → false :=
sorry

end NUMINAMATH_CALUDE_triple_sum_diverges_l1134_113430
