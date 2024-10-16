import Mathlib

namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l730_73046

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

/-- Theorem: A regular dodecagon has 54 diagonals -/
theorem dodecagon_diagonals : num_diagonals dodecagon_sides = 54 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l730_73046


namespace NUMINAMATH_CALUDE_double_length_isosceles_triangle_base_length_l730_73059

/-- A triangle with one side length being twice the length of another side is called a "double-length triangle". -/
def is_double_length_triangle (a b c : ℝ) : Prop :=
  a = 2*b ∨ a = 2*c ∨ b = 2*a ∨ b = 2*c ∨ c = 2*a ∨ c = 2*b

/-- An isosceles triangle is a triangle with at least two equal sides. -/
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ a = c ∨ b = c

theorem double_length_isosceles_triangle_base_length 
  (a b c : ℝ) 
  (h_isosceles : is_isosceles_triangle a b c) 
  (h_double_length : is_double_length_triangle a b c) 
  (h_side_length : a = 6) : 
  (a = b ∧ a = 2*c ∧ c = 3) ∨ (a = c ∧ a = 2*b ∧ b = 3) :=
sorry

end NUMINAMATH_CALUDE_double_length_isosceles_triangle_base_length_l730_73059


namespace NUMINAMATH_CALUDE_simplify_expression_l730_73082

theorem simplify_expression (x y : ℝ) :
  5 * x - 3 * y + 9 * x^2 + 8 - (4 - 5 * x + 3 * y - 9 * x^2) =
  18 * x^2 + 10 * x - 6 * y + 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l730_73082


namespace NUMINAMATH_CALUDE_dice_roll_sum_l730_73038

theorem dice_roll_sum (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 →
  1 ≤ b ∧ b ≤ 6 →
  1 ≤ c ∧ c ≤ 6 →
  1 ≤ d ∧ d ≤ 6 →
  a * b * c * d = 216 →
  a + b + c + d ≠ 18 := by
sorry

end NUMINAMATH_CALUDE_dice_roll_sum_l730_73038


namespace NUMINAMATH_CALUDE_divisibility_of_ones_l730_73086

theorem divisibility_of_ones (p : ℕ) (h_prime : Nat.Prime p) (h_ge_7 : p ≥ 7) :
  ∃ k : ℤ, (10^(p-1) - 1) / 9 = k * p := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_ones_l730_73086


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l730_73044

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2 ≥ 0}
def B : Set ℝ := {x | x^2 - 4*x + 3 ≤ 0}

-- State the theorem
theorem union_of_A_and_B :
  A ∪ B = {x | x ≤ -Real.sqrt 2 ∨ x ≥ 1} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l730_73044


namespace NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l730_73035

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular and subset relations
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_perp_plane_implies_planes_perp
  (α β : Plane) (l : Line)
  (h1 : α ≠ β)
  (h2 : subset l α)
  (h3 : perpendicular_line_plane l β) :
  perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l730_73035


namespace NUMINAMATH_CALUDE_top_square_after_folds_l730_73063

/-- Represents a position on the 5x5 grid -/
structure Position :=
  (row : Fin 5)
  (col : Fin 5)

/-- Represents the state of the grid after folding -/
structure FoldedGrid :=
  (top_square : ℕ)
  (visible_squares : List ℕ)

/-- Initial numbering of the grid in row-major order -/
def initial_grid : Position → ℕ
  | ⟨r, c⟩ => r.val * 5 + c.val + 1

/-- Fold along the diagonal from bottom left to top right -/
def fold_diagonal (grid : Position → ℕ) : FoldedGrid :=
  sorry

/-- Fold the left half over the right half -/
def fold_left_to_right (fg : FoldedGrid) : FoldedGrid :=
  sorry

/-- Fold the top half over the bottom half -/
def fold_top_to_bottom (fg : FoldedGrid) : FoldedGrid :=
  sorry

/-- Fold the bottom half over the top half -/
def fold_bottom_to_top (fg : FoldedGrid) : FoldedGrid :=
  sorry

/-- Apply all folding steps -/
def apply_all_folds (grid : Position → ℕ) : FoldedGrid :=
  fold_bottom_to_top (fold_top_to_bottom (fold_left_to_right (fold_diagonal grid)))

theorem top_square_after_folds :
  (apply_all_folds initial_grid).top_square = 13 := by
  sorry

end NUMINAMATH_CALUDE_top_square_after_folds_l730_73063


namespace NUMINAMATH_CALUDE_right_triangle_tan_l730_73020

theorem right_triangle_tan (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : a = 40) (h3 : c = 41) :
  b / a = 9 / 40 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_tan_l730_73020


namespace NUMINAMATH_CALUDE_lee_ribbons_left_l730_73093

/-- The number of ribbons Mr. Lee had left after giving away ribbons in the morning and afternoon -/
def ribbons_left (initial : ℕ) (morning : ℕ) (afternoon : ℕ) : ℕ :=
  initial - (morning + afternoon)

/-- Theorem stating that Mr. Lee had 8 ribbons left -/
theorem lee_ribbons_left : ribbons_left 38 14 16 = 8 := by
  sorry

end NUMINAMATH_CALUDE_lee_ribbons_left_l730_73093


namespace NUMINAMATH_CALUDE_cubic_function_tangent_and_minimum_l730_73055

/-- A cubic function with parameters m and n -/
def f (m n : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + n*x + 1

/-- The derivative of f -/
def f' (m n : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*m*x + n

theorem cubic_function_tangent_and_minimum (m n : ℝ) :
  (∃ x : ℝ, x ≠ 0 ∧ f m n x = 1 ∧ f' m n x = 0) →
  (∃ x : ℝ, ∀ y : ℝ, f m n y ≥ f m n x ∧ f m n x = -31) →
  m = 12 ∧ n = 36 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_tangent_and_minimum_l730_73055


namespace NUMINAMATH_CALUDE_pencil_average_cost_l730_73003

/-- The average cost per pencil, including shipping -/
def average_cost (num_pencils : ℕ) (pencil_cost shipping_cost : ℚ) : ℚ :=
  (pencil_cost + shipping_cost) / num_pencils

/-- Theorem stating the average cost per pencil for the given problem -/
theorem pencil_average_cost :
  average_cost 150 (24.75 : ℚ) (8.50 : ℚ) = (33.25 : ℚ) / 150 :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_average_cost_l730_73003


namespace NUMINAMATH_CALUDE_phyllis_garden_problem_l730_73007

/-- The number of plants in Phyllis's first garden -/
def plants_in_first_garden : ℕ := 20

/-- The number of plants in Phyllis's second garden -/
def plants_in_second_garden : ℕ := 15

/-- The fraction of tomato plants in the first garden -/
def tomato_fraction_first : ℚ := 1/10

/-- The fraction of tomato plants in the second garden -/
def tomato_fraction_second : ℚ := 1/3

/-- The fraction of tomato plants in both gardens combined -/
def total_tomato_fraction : ℚ := 1/5

theorem phyllis_garden_problem :
  (plants_in_first_garden : ℚ) * tomato_fraction_first +
  (plants_in_second_garden : ℚ) * tomato_fraction_second =
  ((plants_in_first_garden + plants_in_second_garden) : ℚ) * total_tomato_fraction :=
by sorry

end NUMINAMATH_CALUDE_phyllis_garden_problem_l730_73007


namespace NUMINAMATH_CALUDE_difference_of_squares_l730_73078

theorem difference_of_squares (a b : ℕ) (h1 : a + b = 72) (h2 : a - b = 16) : a^2 - b^2 = 1152 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l730_73078


namespace NUMINAMATH_CALUDE_forest_tree_count_l730_73098

/-- Calculates the total number of trees in a forest given the side length of a square street,
    the ratio of forest area to street area, and the tree density in the forest. -/
theorem forest_tree_count (street_side : ℝ) (forest_street_ratio : ℝ) (trees_per_sqm : ℝ) : 
  street_side = 100 →
  forest_street_ratio = 3 →
  trees_per_sqm = 4 →
  (street_side^2 * forest_street_ratio * trees_per_sqm : ℝ) = 120000 := by
  sorry

end NUMINAMATH_CALUDE_forest_tree_count_l730_73098


namespace NUMINAMATH_CALUDE_tower_remainder_l730_73058

/-- Represents the number of towers that can be built with cubes of sizes 1 to n -/
def T : ℕ → ℕ
| 0 => 1
| 1 => 1
| 2 => 6
| 3 => 18
| n+4 => 4 * T (n+3)

/-- The main theorem stating the result for 9 cubes -/
theorem tower_remainder : T 9 % 1000 = 296 := by
  sorry


end NUMINAMATH_CALUDE_tower_remainder_l730_73058


namespace NUMINAMATH_CALUDE_subset_implies_complement_subset_l730_73099

theorem subset_implies_complement_subset (P Q : Set α) 
  (h_nonempty_P : P.Nonempty) (h_nonempty_Q : Q.Nonempty) 
  (h_intersection : P ∩ Q = P) : 
  ∀ x, x ∉ Q → x ∉ P := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_complement_subset_l730_73099


namespace NUMINAMATH_CALUDE_least_months_to_triple_l730_73048

def interest_factor : ℝ := 1.06

def exceeds_triple (t : ℕ) : Prop :=
  interest_factor ^ t > 3

theorem least_months_to_triple : ∃ (t : ℕ), t = 20 ∧ exceeds_triple t ∧ ∀ (k : ℕ), k < t → ¬exceeds_triple k :=
sorry

end NUMINAMATH_CALUDE_least_months_to_triple_l730_73048


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l730_73034

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 1 - a 9 + a 17 = 7 →
  a 3 + a 15 = 14 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l730_73034


namespace NUMINAMATH_CALUDE_article_cost_l730_73076

/-- Proves that the cost of an article is 120, given the selling prices and gain difference --/
theorem article_cost (sp1 sp2 : ℕ) (gain_diff : ℚ) :
  sp1 = 380 →
  sp2 = 420 →
  gain_diff = 8 / 100 →
  sp2 - (sp1 - (sp2 - sp1)) = 120 := by
sorry

end NUMINAMATH_CALUDE_article_cost_l730_73076


namespace NUMINAMATH_CALUDE_tangent_line_x_intercept_l730_73080

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 4*x + 5

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 + 4

theorem tangent_line_x_intercept :
  let slope := f' 1
  let point := (1, f 1)
  let m := slope
  let b := point.2 - m * point.1
  (0 - b) / m = -3/7 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_x_intercept_l730_73080


namespace NUMINAMATH_CALUDE_inverse_f_sum_l730_73075

-- Define the function f
def f (x : ℝ) : ℝ := x^2 * abs x

-- State the theorem
theorem inverse_f_sum : (∃ y₁ y₂ : ℝ, f y₁ = 8 ∧ f y₂ = -27 ∧ y₁ + y₂ = -1) := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_sum_l730_73075


namespace NUMINAMATH_CALUDE_statement_a_statement_d_l730_73070

-- Statement A
theorem statement_a (a b c : ℝ) (h1 : c ≠ 0) (h2 : a * c^2 > b * c^2) : a > b := by
  sorry

-- Statement D
theorem statement_d (a b : ℝ) (h : a > b ∧ b > 0) : a + 1/b > b + 1/a := by
  sorry

end NUMINAMATH_CALUDE_statement_a_statement_d_l730_73070


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l730_73049

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be converted -/
def original_number : ℕ := 2270000

/-- Converts a natural number to scientific notation -/
def to_scientific_notation (n : ℕ) : ScientificNotation := sorry

theorem scientific_notation_correct :
  let sn := to_scientific_notation original_number
  sn.coefficient = 2.27 ∧ sn.exponent = 6 ∧ original_number = sn.coefficient * (10 ^ sn.exponent) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l730_73049


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l730_73012

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 + x - 12 ≥ 0 ↔ x ≤ -4 ∨ x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l730_73012


namespace NUMINAMATH_CALUDE_quadratic_max_abs_value_bound_l730_73019

/-- For any quadratic function f(x) = x^2 + px + q, 
    the maximum absolute value of f(1), f(2), and f(3) 
    is greater than or equal to 1/2. -/
theorem quadratic_max_abs_value_bound (p q : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + p*x + q
  ∃ i : Fin 3, |f (i.val + 1)| ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_max_abs_value_bound_l730_73019


namespace NUMINAMATH_CALUDE_base_seven_digits_of_1234_digits_in_base_seven_1234_l730_73004

theorem base_seven_digits_of_1234 : ∃ n : ℕ, n > 0 ∧ 7^(n-1) ≤ 1234 ∧ 1234 < 7^n :=
by
  -- The proof would go here
  sorry

theorem digits_in_base_seven_1234 : ∃ n : ℕ, n > 0 ∧ 7^(n-1) ≤ 1234 ∧ 1234 < 7^n ∧ n = 4 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_base_seven_digits_of_1234_digits_in_base_seven_1234_l730_73004


namespace NUMINAMATH_CALUDE_inverse_of_B_cubed_l730_73022

def B_inv : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; -2, -3]

theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = B_inv) : 
  (B^3)⁻¹ = B⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_B_cubed_l730_73022


namespace NUMINAMATH_CALUDE_least_sum_of_four_primes_l730_73025

def is_sum_of_four_primes (n : ℕ) : Prop :=
  ∃ p₁ p₂ p₃ p₄ : ℕ, 
    p₁.Prime ∧ p₂.Prime ∧ p₃.Prime ∧ p₄.Prime ∧
    p₁ > 10 ∧ p₂ > 10 ∧ p₃ > 10 ∧ p₄ > 10 ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    n = p₁ + p₂ + p₃ + p₄

theorem least_sum_of_four_primes : 
  (is_sum_of_four_primes 60) ∧ (∀ m < 60, ¬(is_sum_of_four_primes m)) :=
by sorry

end NUMINAMATH_CALUDE_least_sum_of_four_primes_l730_73025


namespace NUMINAMATH_CALUDE_isosceles_triangle_l730_73043

/-- If in a triangle with sides a and b, and their opposite angles α and β, 
    the equation a / cos(α) = b / cos(β) holds, then a = b. -/
theorem isosceles_triangle (a b α β : Real) : 
  0 < a ∧ 0 < b ∧ 0 < α ∧ α < π ∧ 0 < β ∧ β < π →  -- Ensuring valid triangle
  a / Real.cos α = b / Real.cos β →
  a = b :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l730_73043


namespace NUMINAMATH_CALUDE_harriet_ran_approximately_45_miles_l730_73096

/-- The total distance run by six runners -/
def total_distance : ℝ := 378.5

/-- The distance run by Katarina -/
def katarina_distance : ℝ := 47.5

/-- The distance run by Adriana -/
def adriana_distance : ℝ := 83.25

/-- The distance run by Jeremy -/
def jeremy_distance : ℝ := 92.75

/-- The difference in distance between Tomas, Tyler, and Harriet -/
def difference : ℝ := 6.5

/-- Harriet's approximate distance -/
def harriet_distance : ℕ := 45

theorem harriet_ran_approximately_45_miles :
  ∃ (tomas_distance tyler_distance harriet_exact_distance : ℝ),
    tomas_distance ≠ tyler_distance ∧
    tyler_distance ≠ harriet_exact_distance ∧
    tomas_distance ≠ harriet_exact_distance ∧
    (tomas_distance = tyler_distance + difference ∨ tyler_distance = tomas_distance + difference) ∧
    (tyler_distance = harriet_exact_distance + difference ∨ harriet_exact_distance = tyler_distance + difference) ∧
    tomas_distance + tyler_distance + harriet_exact_distance + katarina_distance + adriana_distance + jeremy_distance = total_distance ∧
    harriet_distance = round harriet_exact_distance :=
by
  sorry

end NUMINAMATH_CALUDE_harriet_ran_approximately_45_miles_l730_73096


namespace NUMINAMATH_CALUDE_work_completion_time_l730_73037

theorem work_completion_time (aarti_rate ramesh_rate : ℚ) 
  (h1 : aarti_rate = 1 / 6)
  (h2 : ramesh_rate = 1 / 8)
  (h3 : (aarti_rate + ramesh_rate) * 3 = 1) :
  3 = 3 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l730_73037


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l730_73032

/-- Parabola P with equation y = x^2 + 3x + 1 -/
def P : ℝ → ℝ := λ x => x^2 + 3*x + 1

/-- Point Q -/
def Q : ℝ × ℝ := (10, 50)

/-- Line through Q with slope m -/
def line (m : ℝ) : ℝ → ℝ := λ x => m*(x - Q.1) + Q.2

/-- Condition for line not intersecting parabola -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x, P x ≠ line m x

/-- Main theorem -/
theorem parabola_line_intersection :
  ∃! (r s : ℝ), (∀ m, no_intersection m ↔ r < m ∧ m < s) ∧ r + s = 46 := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l730_73032


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l730_73097

theorem polynomial_division_remainder : 
  ∃ q : Polynomial ℚ, 
    (3 * X^5 + 16 * X^4 - 17 * X^3 - 100 * X^2 + 32 * X + 90 : Polynomial ℚ) = 
    (X^3 + 8 * X^2 - X - 6) * q + (422 * X^2 + 48 * X - 294) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l730_73097


namespace NUMINAMATH_CALUDE_f_12_equals_190_l730_73085

def f (n : ℤ) : ℤ := n^2 + 2*n + 22

theorem f_12_equals_190 : f 12 = 190 := by
  sorry

end NUMINAMATH_CALUDE_f_12_equals_190_l730_73085


namespace NUMINAMATH_CALUDE_extreme_values_cubic_l730_73010

/-- Given a cubic function with extreme values at x=1 and x=2, prove that b=4 -/
theorem extreme_values_cubic (a b : ℝ) : 
  let f := fun x : ℝ => 2 * x^3 + 3 * a * x^2 + 3 * b * x
  let f' := fun x : ℝ => 6 * x^2 + 6 * a * x + 3 * b
  (f' 1 = 0 ∧ f' 2 = 0) → b = 4 := by
sorry

end NUMINAMATH_CALUDE_extreme_values_cubic_l730_73010


namespace NUMINAMATH_CALUDE_car_travel_time_l730_73066

/-- Proves that a car with given specifications travels for 5 hours -/
theorem car_travel_time (speed : ℝ) (fuel_efficiency : ℝ) (tank_capacity : ℝ) (fuel_used_ratio : ℝ) :
  speed = 50 →
  fuel_efficiency = 30 →
  tank_capacity = 10 →
  fuel_used_ratio = 0.8333333333333334 →
  (fuel_used_ratio * tank_capacity * fuel_efficiency) / speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_time_l730_73066


namespace NUMINAMATH_CALUDE_shirts_washed_l730_73017

-- Define the given quantities
def short_sleeve_shirts : ℕ := 39
def long_sleeve_shirts : ℕ := 47
def unwashed_shirts : ℕ := 66

-- Define the total number of shirts
def total_shirts : ℕ := short_sleeve_shirts + long_sleeve_shirts

-- Theorem: The number of shirts Oliver washed is 20
theorem shirts_washed : total_shirts - unwashed_shirts = 20 := by
  sorry

end NUMINAMATH_CALUDE_shirts_washed_l730_73017


namespace NUMINAMATH_CALUDE_A_difference_max_l730_73083

def A (a b : ℕ+) : ℚ := (a + 3) / 12

theorem A_difference_max :
  (∃ a₁ b₁ a₂ b₂ : ℕ+, 
    A a₁ b₁ = 15 / (26 - b₁) ∧
    A a₂ b₂ = 15 / (26 - b₂) ∧
    ∀ a b : ℕ+, A a b = 15 / (26 - b) → 
      A a₁ b₁ ≤ A a b ∧ A a b ≤ A a₂ b₂) →
  A a₂ b₂ - A a₁ b₁ = 57 / 4 :=
sorry

end NUMINAMATH_CALUDE_A_difference_max_l730_73083


namespace NUMINAMATH_CALUDE_solution_set_for_a_3_f_geq_1_iff_a_leq_1_or_geq_3_l730_73033

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := |x| + 2*|x + 2 - a|

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := g a (x - 2)

-- Theorem for part (1)
theorem solution_set_for_a_3 :
  {x : ℝ | g 3 x ≤ 4} = Set.Icc (-2/3) 2 := by sorry

-- Theorem for part (2)
theorem f_geq_1_iff_a_leq_1_or_geq_3 :
  (∀ x, f a x ≥ 1) ↔ (a ≤ 1 ∨ a ≥ 3) := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_3_f_geq_1_iff_a_leq_1_or_geq_3_l730_73033


namespace NUMINAMATH_CALUDE_parallel_vectors_angle_l730_73005

theorem parallel_vectors_angle (θ : Real) 
  (h_acute : 0 < θ ∧ θ < π / 2)
  (h_parallel : ∃ (k : Real), k ≠ 0 ∧ (1 - Real.sin θ, 1) = k • (1/2, 1 + Real.sin θ)) :
  θ = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_angle_l730_73005


namespace NUMINAMATH_CALUDE_factorization_problems_l730_73036

theorem factorization_problems (x : ℝ) : 
  (2 * x^2 - 8 = 2 * (x + 2) * (x - 2)) ∧ 
  (2 * x^2 + 2 * x + (1/2) = 2 * (x + 1/2)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l730_73036


namespace NUMINAMATH_CALUDE_senior_student_count_l730_73001

theorem senior_student_count 
  (total_students : ℕ) 
  (freshmen : ℕ) 
  (sophomore_prob : ℚ) 
  (h1 : total_students = 2000)
  (h2 : freshmen = 650)
  (h3 : sophomore_prob = 0.40)
  : ∃ (seniors : ℕ), seniors = 550 :=
by sorry

end NUMINAMATH_CALUDE_senior_student_count_l730_73001


namespace NUMINAMATH_CALUDE_smallest_n_for_g_nine_l730_73072

/-- Sum of digits in base 5 representation -/
def f (n : ℕ) : ℕ := sorry

/-- Sum of digits in base 9 representation of f(n) -/
def g (n : ℕ) : ℕ := sorry

/-- The smallest positive integer n such that g(n) = 9 -/
theorem smallest_n_for_g_nine : 
  (∀ m : ℕ, m > 0 ∧ m < 344 → g m ≠ 9) ∧ g 344 = 9 := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_g_nine_l730_73072


namespace NUMINAMATH_CALUDE_tshirt_pricing_l730_73057

def first_batch_cost : ℝ := 4000
def second_batch_cost : ℝ := 8800
def cost_difference : ℝ := 4
def discounted_quantity : ℕ := 40
def discount_rate : ℝ := 0.3
def min_profit_margin : ℝ := 0.8

def cost_price_first_batch : ℝ := 40
def cost_price_second_batch : ℝ := 44
def min_retail_price : ℝ := 80

theorem tshirt_pricing :
  let first_quantity := first_batch_cost / cost_price_first_batch
  let second_quantity := second_batch_cost / cost_price_second_batch
  let total_quantity := first_quantity + second_quantity
  (2 * first_quantity = second_quantity) ∧
  (cost_price_second_batch = cost_price_first_batch + cost_difference) ∧
  (min_retail_price * (total_quantity - discounted_quantity) +
   min_retail_price * (1 - discount_rate) * discounted_quantity ≥
   (first_batch_cost + second_batch_cost) * (1 + min_profit_margin)) :=
by sorry

end NUMINAMATH_CALUDE_tshirt_pricing_l730_73057


namespace NUMINAMATH_CALUDE_unique_solution_for_P_squared_prime_l730_73069

/-- The polynomial P(n) = n^3 - n^2 - 5n + 2 -/
def P (n : ℤ) : ℤ := n^3 - n^2 - 5*n + 2

/-- A predicate to check if a number is prime -/
def isPrime (p : ℤ) : Prop := Nat.Prime p.natAbs

theorem unique_solution_for_P_squared_prime :
  ∃! n : ℤ, ∃ p : ℤ, isPrime p ∧ (P n)^2 = p^2 ∧ n = -3 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_for_P_squared_prime_l730_73069


namespace NUMINAMATH_CALUDE_meaningful_fraction_condition_l730_73045

theorem meaningful_fraction_condition (x : ℝ) :
  (∃ y, y = (x - 1) / (x + 1)) ↔ x ≠ -1 :=
sorry

end NUMINAMATH_CALUDE_meaningful_fraction_condition_l730_73045


namespace NUMINAMATH_CALUDE_range_of_a_l730_73089

/-- Proposition p: The real number x satisfies x^2 - 4ax + 3a^2 < 0 -/
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

/-- Proposition q: The real number x satisfies x^2 - x - 6 ≤ 0 and x^2 + 2x - 8 > 0 -/
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

/-- The set of x satisfying proposition p -/
def A (a : ℝ) : Set ℝ := {x | p a x}

/-- The set of x satisfying proposition q -/
def B : Set ℝ := {x | q x}

theorem range_of_a (a : ℝ) :
  a > 0 ∧ 
  (∀ x, ¬(q x) → ¬(p a x)) ∧
  (∃ x, ¬(q x) ∧ p a x) →
  1 < a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l730_73089


namespace NUMINAMATH_CALUDE_wrapping_paper_area_theorem_l730_73009

/-- Represents a box with a square base -/
structure Box where
  side : ℝ
  height : ℝ

/-- Represents the wrapping paper -/
structure WrappingPaper where
  side : ℝ

/-- Calculates the area of the wrapping paper needed to wrap the box -/
def wrappingPaperArea (b : Box) (w : WrappingPaper) : ℝ :=
  w.side * w.side

/-- Theorem stating the area of wrapping paper needed -/
theorem wrapping_paper_area_theorem (s : ℝ) (h : s > 0) :
  let b : Box := { side := 2 * s, height := 3 * s }
  let w : WrappingPaper := { side := 4 * s }
  wrappingPaperArea b w = 24 * s^2 := by
  sorry

#check wrapping_paper_area_theorem

end NUMINAMATH_CALUDE_wrapping_paper_area_theorem_l730_73009


namespace NUMINAMATH_CALUDE_medication_forgotten_days_l730_73091

theorem medication_forgotten_days (total_days : ℕ) (taken_days : ℕ) : 
  total_days = 31 → taken_days = 29 → total_days - taken_days = 2 := by
  sorry

end NUMINAMATH_CALUDE_medication_forgotten_days_l730_73091


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l730_73011

theorem triangle_angle_measure (y : ℝ) : 
  y > 0 ∧ 
  y < 180 ∧ 
  3*y > 0 ∧ 
  3*y < 180 ∧
  y + 3*y + 40 = 180 → 
  y = 35 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l730_73011


namespace NUMINAMATH_CALUDE_f_n_has_real_root_l730_73000

def f (x : ℝ) : ℝ := x^2 + 2007*x + 1

def f_n (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n + 1 => f (f_n n x)

theorem f_n_has_real_root (n : ℕ+) : ∃ x : ℝ, f_n n x = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_n_has_real_root_l730_73000


namespace NUMINAMATH_CALUDE_smallest_tripling_period_l730_73013

-- Define the annual interest rate
def r : ℝ := 0.3334

-- Define the function that calculates the investment value after n years
def investment_value (n : ℕ) : ℝ := (1 + r) ^ n

-- Theorem statement
theorem smallest_tripling_period :
  ∀ n : ℕ, (investment_value n > 3 ∧ ∀ m : ℕ, m < n → investment_value m ≤ 3) → n = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_tripling_period_l730_73013


namespace NUMINAMATH_CALUDE_pretzel_ratio_l730_73039

/-- The number of pretzels bought by Angie -/
def angie_pretzels : ℕ := 18

/-- The number of pretzels bought by Barry -/
def barry_pretzels : ℕ := 12

/-- The number of pretzels bought by Shelly -/
def shelly_pretzels : ℕ := angie_pretzels / 3

/-- Theorem stating the ratio of pretzels Shelly bought to pretzels Barry bought -/
theorem pretzel_ratio : 
  (shelly_pretzels : ℚ) / barry_pretzels = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_pretzel_ratio_l730_73039


namespace NUMINAMATH_CALUDE_restaurant_bill_division_l730_73006

/-- Given a group of friends dividing a restaurant bill evenly, this theorem proves
    the number of friends in the group based on the total bill and individual payment. -/
theorem restaurant_bill_division (total_bill : ℕ) (individual_payment : ℕ) 
    (h1 : total_bill = 135)
    (h2 : individual_payment = 45) :
    total_bill / individual_payment = 3 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_division_l730_73006


namespace NUMINAMATH_CALUDE_edward_book_spending_l730_73071

/-- Given Edward's initial amount, amount spent on pens, and remaining amount,
    prove that the amount spent on books is $6. -/
theorem edward_book_spending (initial : ℕ) (spent_on_pens : ℕ) (remaining : ℕ) 
    (h1 : initial = 41)
    (h2 : spent_on_pens = 16)
    (h3 : remaining = 19) :
    initial - remaining - spent_on_pens = 6 := by
  sorry

end NUMINAMATH_CALUDE_edward_book_spending_l730_73071


namespace NUMINAMATH_CALUDE_exists_initial_order_l730_73053

/-- Represents a playing card suit -/
inductive Suit
| Diamonds
| Hearts
| Spades
| Clubs

/-- Represents a playing card rank -/
inductive Rank
| Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- Represents a playing card -/
structure Card :=
  (suit : Suit)
  (rank : Rank)

/-- The number of letters in the name of a card's rank -/
def letterCount : Rank → Nat
| Rank.Ace => 3
| Rank.Two => 3
| Rank.Three => 5
| Rank.Four => 4
| Rank.Five => 4
| Rank.Six => 3
| Rank.Seven => 5
| Rank.Eight => 5
| Rank.Nine => 4
| Rank.Ten => 3
| Rank.Jack => 4
| Rank.Queen => 5
| Rank.King => 4

/-- Applies the card moving rule to a deck -/
def applyRule (deck : List Card) : List Card :=
  sorry

/-- The final order of cards after applying the rule -/
def finalOrder : List Card :=
  sorry

/-- Theorem: There exists an initial deck ordering that results in the specified final order -/
theorem exists_initial_order :
  ∃ (initialDeck : List Card),
    initialDeck.length = 52 ∧
    (∀ s : Suit, ∀ r : Rank, ∃ c ∈ initialDeck, c.suit = s ∧ c.rank = r) ∧
    applyRule initialDeck = finalOrder :=
  sorry

end NUMINAMATH_CALUDE_exists_initial_order_l730_73053


namespace NUMINAMATH_CALUDE_doughnut_cost_theorem_l730_73084

def total_cost (chocolate_count : ℕ) (glazed_count : ℕ) (maple_count : ℕ) (strawberry_count : ℕ)
               (chocolate_price : ℚ) (glazed_price : ℚ) (maple_price : ℚ) (strawberry_price : ℚ)
               (chocolate_discount : ℚ) (maple_discount : ℚ) (free_glazed : ℕ) : ℚ :=
  let chocolate_cost := (chocolate_count : ℚ) * chocolate_price * (1 - chocolate_discount)
  let glazed_cost := (glazed_count : ℚ) * glazed_price
  let maple_cost := (maple_count : ℚ) * maple_price * (1 - maple_discount)
  let strawberry_cost := (strawberry_count : ℚ) * strawberry_price
  let free_glazed_savings := (free_glazed : ℚ) * glazed_price
  chocolate_cost + glazed_cost + maple_cost + strawberry_cost - free_glazed_savings

theorem doughnut_cost_theorem :
  total_cost 10 8 5 2 2 1 (3/2) (5/2) (15/100) (1/10) 1 = 143/4 := by
  sorry

end NUMINAMATH_CALUDE_doughnut_cost_theorem_l730_73084


namespace NUMINAMATH_CALUDE_slower_speed_calculation_l730_73060

/-- Given a person who walks a certain distance at two different speeds, 
    prove that the slower speed is 10 km/hr. -/
theorem slower_speed_calculation 
  (actual_distance : ℝ) 
  (faster_speed : ℝ) 
  (additional_distance : ℝ) 
  (h1 : actual_distance = 50) 
  (h2 : faster_speed = 14) 
  (h3 : additional_distance = 20) :
  let total_distance := actual_distance + additional_distance
  let time := total_distance / faster_speed
  let slower_speed := actual_distance / time
  slower_speed = 10 := by
sorry

end NUMINAMATH_CALUDE_slower_speed_calculation_l730_73060


namespace NUMINAMATH_CALUDE_square_triangle_equal_area_l730_73087

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) (x : ℝ) : 
  square_perimeter = 48 →
  triangle_height = 48 →
  (square_perimeter / 4)^2 = (1/2) * x * triangle_height →
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_equal_area_l730_73087


namespace NUMINAMATH_CALUDE_inequality_solution_l730_73056

theorem inequality_solution (x y : ℝ) : 
  Real.sqrt 3 * Real.tan x - (Real.sin y) ^ (1/4) - 
  Real.sqrt ((3 / (Real.cos x)^2) + Real.sqrt (Real.sin y) - 6) ≥ Real.sqrt 3 ↔ 
  ∃ (n k : ℤ), x = π/4 + n*π ∧ y = k*π :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l730_73056


namespace NUMINAMATH_CALUDE_walters_coins_value_l730_73023

/-- Represents the value of a coin in cents -/
def coin_value : String → Nat
  | "penny" => 1
  | "nickel" => 5
  | "dime" => 10
  | "half_dollar" => 50
  | _ => 0

/-- Calculates the total value of coins in cents -/
def total_value (pennies nickels dimes half_dollars : Nat) : Nat :=
  pennies * coin_value "penny" +
  nickels * coin_value "nickel" +
  dimes * coin_value "dime" +
  half_dollars * coin_value "half_dollar"

/-- Converts a number of cents to a percentage of a dollar -/
def cents_to_percentage (cents : Nat) : Nat :=
  cents

theorem walters_coins_value :
  total_value 2 1 2 1 = 77 ∧ cents_to_percentage (total_value 2 1 2 1) = 77 := by
  sorry

end NUMINAMATH_CALUDE_walters_coins_value_l730_73023


namespace NUMINAMATH_CALUDE_corrected_mean_problem_l730_73077

/-- Calculates the corrected mean of a set of observations after fixing an error -/
def corrected_mean (n : ℕ) (initial_mean : ℚ) (wrong_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n * initial_mean + correct_value - wrong_value) / n

/-- Theorem stating that the corrected mean is 36.14 given the problem conditions -/
theorem corrected_mean_problem :
  let n : ℕ := 50
  let initial_mean : ℚ := 36
  let wrong_value : ℚ := 23
  let correct_value : ℚ := 30
  corrected_mean n initial_mean wrong_value correct_value = 36.14 := by
sorry

#eval corrected_mean 50 36 23 30

end NUMINAMATH_CALUDE_corrected_mean_problem_l730_73077


namespace NUMINAMATH_CALUDE_stock_price_increase_l730_73021

/-- Given a stock price that decreased by 8% in the first year and had a net percentage change of 1.20% over two years, the percentage increase in the second year was 10%. -/
theorem stock_price_increase (initial_price : ℝ) (h : initial_price > 0) :
  let price_after_decrease := initial_price * (1 - 0.08)
  let final_price := initial_price * (1 + 0.012)
  let increase_percentage := (final_price / price_after_decrease - 1) * 100
  increase_percentage = 10 := by sorry

end NUMINAMATH_CALUDE_stock_price_increase_l730_73021


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l730_73052

theorem hyperbola_focal_length (a : ℝ) (h1 : a > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / 20 = 1 → y = 2*x) → 
  let c := Real.sqrt (a^2 + 20)
  2 * c = 10 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l730_73052


namespace NUMINAMATH_CALUDE_cara_age_is_40_l730_73061

-- Define the ages as natural numbers
def grandmother_age : ℕ := 75
def mom_age : ℕ := grandmother_age - 15
def cara_age : ℕ := mom_age - 20

-- Theorem statement
theorem cara_age_is_40 : cara_age = 40 := by
  sorry

end NUMINAMATH_CALUDE_cara_age_is_40_l730_73061


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l730_73047

theorem lcm_gcf_problem (n m : ℕ) (h1 : Nat.lcm n m = 48) (h2 : Nat.gcd n m = 18) (h3 : m = 16) : n = 54 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l730_73047


namespace NUMINAMATH_CALUDE_points_below_line_l730_73008

def arithmetic_sequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def geometric_sequence (a b c d : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c

theorem points_below_line (x₁ x₂ y₁ y₂ : ℝ) :
  arithmetic_sequence 1 x₁ x₂ 2 →
  geometric_sequence 1 y₁ y₂ 2 →
  x₁ > y₁ ∧ x₂ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_points_below_line_l730_73008


namespace NUMINAMATH_CALUDE_unit_fraction_decomposition_l730_73018

theorem unit_fraction_decomposition (n : ℕ) (hn : n > 0) :
  (1 : ℚ) / n = 1 / (2 * n) + 1 / (3 * n) + 1 / (6 * n) := by
  sorry

end NUMINAMATH_CALUDE_unit_fraction_decomposition_l730_73018


namespace NUMINAMATH_CALUDE_rain_both_days_no_snow_l730_73095

theorem rain_both_days_no_snow (rain_sat rain_sun snow_sat : ℝ) 
  (h_rain_sat : rain_sat = 0.7)
  (h_rain_sun : rain_sun = 0.5)
  (h_snow_sat : snow_sat = 0.2)
  (h_independence : True) -- Assumption of independence
  : rain_sat * rain_sun * (1 - snow_sat) = 0.28 := by
  sorry

end NUMINAMATH_CALUDE_rain_both_days_no_snow_l730_73095


namespace NUMINAMATH_CALUDE_prob_at_least_one_boy_and_girl_l730_73024

def prob_boy_or_girl : ℚ := 1 / 2

def family_size : ℕ := 4

theorem prob_at_least_one_boy_and_girl :
  (1 : ℚ) - (prob_boy_or_girl ^ family_size + prob_boy_or_girl ^ family_size) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_boy_and_girl_l730_73024


namespace NUMINAMATH_CALUDE_count_pairs_sum_squares_less_than_50_l730_73029

theorem count_pairs_sum_squares_less_than_50 :
  (Finset.filter (fun p : ℕ × ℕ => p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 + p.2^2 < 50)
    (Finset.product (Finset.range 50) (Finset.range 50))).card = 32 :=
by sorry

end NUMINAMATH_CALUDE_count_pairs_sum_squares_less_than_50_l730_73029


namespace NUMINAMATH_CALUDE_triangle_median_similarity_exists_l730_73041

/-- 
Given a triangle with sides a, b, c (where a < b < c), we define the following:
1) The triangle formed by the medians is similar to the original triangle.
2) The relationship between sides and medians is given by:
   4sa² = -a² + 2b² + 2c²
   4sb² = 2a² - b² + 2c²
   4sc² = 2a² + 2b² - c²
   where sa, sb, sc are the medians opposite to sides a, b, c respectively.
3) The sides satisfy the equation: b² = (a² + c²) / 2

This theorem states that there exists a triplet of natural numbers (a, b, c) 
that satisfies all these conditions, with a < b < c.
-/
theorem triangle_median_similarity_exists : 
  ∃ (a b c : ℕ), a < b ∧ b < c ∧ 
  (b * b : ℚ) = (a * a + c * c) / 2 ∧
  (∃ (sa sb sc : ℚ), 
    4 * sa * sa = -a * a + 2 * b * b + 2 * c * c ∧
    4 * sb * sb = 2 * a * a - b * b + 2 * c * c ∧
    4 * sc * sc = 2 * a * a + 2 * b * b - c * c ∧
    (a : ℚ) / sc = (b : ℚ) / sb ∧ (b : ℚ) / sb = (c : ℚ) / sa) :=
by sorry

end NUMINAMATH_CALUDE_triangle_median_similarity_exists_l730_73041


namespace NUMINAMATH_CALUDE_large_cube_volume_l730_73081

theorem large_cube_volume (die_surface_area : ℝ) (h : die_surface_area = 96) :
  let die_face_area := die_surface_area / 6
  let large_cube_face_area := 4 * die_face_area
  let large_cube_side_length := Real.sqrt large_cube_face_area
  large_cube_side_length ^ 3 = 512 := by
  sorry

end NUMINAMATH_CALUDE_large_cube_volume_l730_73081


namespace NUMINAMATH_CALUDE_zhang_ning_match_results_l730_73015

/-- Represents the outcome of a badminton match --/
inductive MatchOutcome
  | WinTwoZero
  | WinTwoOne
  | LoseTwoOne
  | LoseTwoZero

/-- Probability of Xie Xingfang winning a single set in the first two sets --/
def p_xie : ℝ := 0.6

/-- Probability of Zhang Ning winning the third set if the score reaches 1:1 --/
def p_zhang_third : ℝ := 0.6

/-- Calculates the probability of Zhang Ning winning with a score of 2:1 --/
def prob_zhang_win_two_one : ℝ :=
  2 * (1 - p_xie) * p_xie * p_zhang_third

/-- Calculates the expected value of Zhang Ning's net winning sets --/
def expected_net_wins : ℝ :=
  -2 * (p_xie * p_xie) +
  -1 * (2 * (1 - p_xie) * p_xie * (1 - p_zhang_third)) +
  1 * prob_zhang_win_two_one +
  2 * ((1 - p_xie) * (1 - p_xie))

/-- Theorem stating the probability of Zhang Ning winning 2:1 and her expected net winning sets --/
theorem zhang_ning_match_results :
  prob_zhang_win_two_one = 0.288 ∧ expected_net_wins = 0.496 := by
  sorry


end NUMINAMATH_CALUDE_zhang_ning_match_results_l730_73015


namespace NUMINAMATH_CALUDE_exists_committees_with_common_members_l730_73073

/-- Represents a committee system with members and committees. -/
structure CommitteeSystem where
  members : Finset ℕ
  committees : Finset (Finset ℕ)
  member_count : members.card = 1600
  committee_count : committees.card = 16000
  committee_size : ∀ c ∈ committees, c.card = 80

/-- Theorem stating that in a committee system satisfying the given conditions,
    there exist at least two committees sharing at least 4 members. -/
theorem exists_committees_with_common_members (cs : CommitteeSystem) :
  ∃ (c1 c2 : Finset ℕ), c1 ∈ cs.committees ∧ c2 ∈ cs.committees ∧ c1 ≠ c2 ∧
  (c1 ∩ c2).card ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_exists_committees_with_common_members_l730_73073


namespace NUMINAMATH_CALUDE_train_speed_l730_73051

/-- The speed of a train given its length and time to cross an electric pole. -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 3500) (h2 : time = 80) :
  length / time = 43.75 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l730_73051


namespace NUMINAMATH_CALUDE_additive_multiplicative_inverses_l730_73026

theorem additive_multiplicative_inverses 
  (x y p q : ℝ) 
  (h1 : x + y = 0)  -- x and y are additive inverses
  (h2 : p * q = 1)  -- p and q are multiplicative inverses
  : (x + y) - 2 * p * q = -2 := by
sorry

end NUMINAMATH_CALUDE_additive_multiplicative_inverses_l730_73026


namespace NUMINAMATH_CALUDE_joe_not_eating_pizza_probability_l730_73054

theorem joe_not_eating_pizza_probability 
  (p_eat : ℚ) 
  (h_eat : p_eat = 5/8) : 
  1 - p_eat = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_joe_not_eating_pizza_probability_l730_73054


namespace NUMINAMATH_CALUDE_marla_errand_time_l730_73040

/-- Calculates the total time Marla spends on her errand to her son's school -/
def total_errand_time (one_way_drive_time parent_teacher_time : ℕ) : ℕ :=
  2 * one_way_drive_time + parent_teacher_time

/-- Proves that Marla spends 110 minutes on her errand -/
theorem marla_errand_time :
  total_errand_time 20 70 = 110 := by
  sorry

end NUMINAMATH_CALUDE_marla_errand_time_l730_73040


namespace NUMINAMATH_CALUDE_sum_ways_2002_l730_73031

/-- The number of ways to express 2002 as the sum of 3 positive integers, without considering order -/
def ways_to_sum_2002 : ℕ := 334000

/-- A function that counts the number of ways to express a given natural number as the sum of 3 positive integers, without considering order -/
def count_sum_ways (n : ℕ) : ℕ :=
  sorry

theorem sum_ways_2002 : count_sum_ways 2002 = ways_to_sum_2002 := by
  sorry

end NUMINAMATH_CALUDE_sum_ways_2002_l730_73031


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_twenty_five_sixths_l730_73014

theorem greatest_integer_less_than_negative_twenty_five_sixths :
  Int.floor (-25 / 6 : ℚ) = -5 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_twenty_five_sixths_l730_73014


namespace NUMINAMATH_CALUDE_two_digit_divisible_by_digit_product_l730_73068

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def tens_digit (n : ℕ) : ℕ := n / 10

def ones_digit (n : ℕ) : ℕ := n % 10

def divisible_by_digit_product (n : ℕ) : Prop :=
  is_two_digit n ∧ n % (tens_digit n * ones_digit n) = 0

theorem two_digit_divisible_by_digit_product :
  {n : ℕ | divisible_by_digit_product n} = {11, 12, 24, 36, 15} :=
by sorry

end NUMINAMATH_CALUDE_two_digit_divisible_by_digit_product_l730_73068


namespace NUMINAMATH_CALUDE_exponential_function_passes_through_point_zero_one_l730_73050

theorem exponential_function_passes_through_point_zero_one
  (a : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x
  f 0 = 1 := by
sorry

end NUMINAMATH_CALUDE_exponential_function_passes_through_point_zero_one_l730_73050


namespace NUMINAMATH_CALUDE_determine_h_of_x_l730_73074

theorem determine_h_of_x (x : ℝ) (h : ℝ → ℝ) : 
  (4 * x^4 + 5 * x^2 - 2 * x + 1 + h x = 6 * x^3 - 4 * x^2 + 7 * x - 5) → 
  (h x = -4 * x^4 + 6 * x^3 - 9 * x^2 + 9 * x - 6) := by
  sorry

end NUMINAMATH_CALUDE_determine_h_of_x_l730_73074


namespace NUMINAMATH_CALUDE_prime_sum_47_l730_73027

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define the property we want to prove
def no_prime_sum_47 : Prop :=
  ∀ p q : ℕ, is_prime p → is_prime q → p + q ≠ 47

-- State the theorem
theorem prime_sum_47 : no_prime_sum_47 :=
sorry

end NUMINAMATH_CALUDE_prime_sum_47_l730_73027


namespace NUMINAMATH_CALUDE_exists_cut_sequence_for_1003_l730_73067

/-- Represents the number of pieces selected for cutting at each step -/
def CutSequence := List Nat

/-- Calculates the number of pieces after a sequence of cuts -/
def numPieces (cuts : CutSequence) : Nat :=
  3 * (cuts.sum + 1) + 1

/-- Theorem: It's possible to obtain 1003 pieces through the cutting process -/
theorem exists_cut_sequence_for_1003 : ∃ (cuts : CutSequence), numPieces cuts = 1003 := by
  sorry

end NUMINAMATH_CALUDE_exists_cut_sequence_for_1003_l730_73067


namespace NUMINAMATH_CALUDE_wall_length_approximation_l730_73065

/-- Given a square mirror and a rectangular wall, if the mirror's area is exactly half the wall's area,
    prove that the length of the wall is approximately 43 inches. -/
theorem wall_length_approximation (mirror_side : ℝ) (wall_width : ℝ) (wall_length : ℝ) : 
  mirror_side = 34 →
  wall_width = 54 →
  mirror_side ^ 2 = (wall_width * wall_length) / 2 →
  ∃ ε > 0, |wall_length - 43| < ε :=
by sorry

end NUMINAMATH_CALUDE_wall_length_approximation_l730_73065


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l730_73028

theorem least_common_multiple_first_ten : ∃ n : ℕ+, 
  (∀ k : ℕ+, k ≤ 10 → k ∣ n) ∧ 
  (∀ m : ℕ+, (∀ k : ℕ+, k ≤ 10 → k ∣ m) → n ≤ m) ∧
  n = 2520 := by
sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l730_73028


namespace NUMINAMATH_CALUDE_yellow_parrots_count_l730_73002

theorem yellow_parrots_count (total : ℕ) (red_fraction : ℚ) (h1 : total = 108) (h2 : red_fraction = 5/6) :
  total * (1 - red_fraction) = 18 := by
  sorry

end NUMINAMATH_CALUDE_yellow_parrots_count_l730_73002


namespace NUMINAMATH_CALUDE_unique_g_2_l730_73016

/-- The functional equation for g -/
def functional_equation (g : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → g x * g y = g (x * y) + 2010 * (1 / x + 1 / y + 2009)

/-- The main theorem -/
theorem unique_g_2 (g : ℝ → ℝ) (h : functional_equation g) :
    ∃! v, g 2 = v ∧ v = 1 / 2 + 2010 := by sorry

end NUMINAMATH_CALUDE_unique_g_2_l730_73016


namespace NUMINAMATH_CALUDE_m_range_l730_73090

theorem m_range (m : ℝ) : 
  (∃ x₀ : ℝ, x₀^2 + m ≤ 0) ∧ 
  (∀ x : ℝ, x^2 + m*x + 1 > 0) ∧ 
  ¬((¬∃ x₀ : ℝ, x₀^2 + m ≤ 0) ∨ (∀ x : ℝ, x^2 + m*x + 1 > 0)) → 
  m ≤ -2 := by
sorry

end NUMINAMATH_CALUDE_m_range_l730_73090


namespace NUMINAMATH_CALUDE_solution_count_l730_73092

/-- The number of positive integer solutions to the equation 3x + 4y = 1024 -/
def num_solutions : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 3 * p.1 + 4 * p.2 = 1024 ∧ p.1 > 0 ∧ p.2 > 0)
    (Finset.product (Finset.range 1025) (Finset.range 1025))).card

theorem solution_count : num_solutions = 85 := by
  sorry

end NUMINAMATH_CALUDE_solution_count_l730_73092


namespace NUMINAMATH_CALUDE_min_score_game_12_is_42_l730_73094

/-- Represents the scores of a football player over a season -/
structure FootballScores where
  first_seven : ℕ  -- Total score for first 7 games
  game_8 : ℕ := 18
  game_9 : ℕ := 25
  game_10 : ℕ := 10
  game_11 : ℕ := 22
  game_12 : ℕ

/-- The minimum score for game 12 that satisfies all conditions -/
def min_score_game_12 (scores : FootballScores) : Prop :=
  let total_8_to_11 := scores.game_8 + scores.game_9 + scores.game_10 + scores.game_11
  let avg_8_to_11 : ℚ := total_8_to_11 / 4
  let total_12_games := scores.first_seven + total_8_to_11 + scores.game_12
  (scores.first_seven / 7 : ℚ) < (total_12_games - scores.game_12) / 11 ∧ 
  (total_12_games : ℚ) / 12 > 20 ∧
  scores.game_12 = 42 ∧
  ∀ x : ℕ, x < 42 → 
    let total_with_x := scores.first_seven + total_8_to_11 + x
    (total_with_x : ℚ) / 12 ≤ 20 ∨ (scores.first_seven / 7 : ℚ) ≥ (total_with_x - x) / 11

theorem min_score_game_12_is_42 (scores : FootballScores) :
  min_score_game_12 scores := by sorry

end NUMINAMATH_CALUDE_min_score_game_12_is_42_l730_73094


namespace NUMINAMATH_CALUDE_iron_conducts_electricity_is_deductive_reasoning_l730_73042

-- Define the set of all objects
variable (Object : Type)

-- Define the property of being a metal
variable (is_metal : Object → Prop)

-- Define the property of conducting electricity
variable (conducts_electricity : Object → Prop)

-- Define iron as an object
variable (iron : Object)

-- Theorem statement
theorem iron_conducts_electricity_is_deductive_reasoning
  (h1 : ∀ x, is_metal x → conducts_electricity x)  -- All metals conduct electricity
  (h2 : is_metal iron)                             -- Iron is a metal
  : conducts_electricity iron                      -- Therefore, iron conducts electricity
  := by sorry

-- The fact that this can be proved using only the given premises
-- demonstrates that this is deductive reasoning

end NUMINAMATH_CALUDE_iron_conducts_electricity_is_deductive_reasoning_l730_73042


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l730_73062

/-- The capacity of a water tank in liters -/
def tank_capacity : ℝ := 72

/-- The amount of water in the tank when it's 40% full -/
def water_at_40_percent : ℝ := 0.4 * tank_capacity

/-- The amount of water in the tank when it's 10% empty (90% full) -/
def water_at_90_percent : ℝ := 0.9 * tank_capacity

/-- Theorem stating the tank capacity based on the given condition -/
theorem tank_capacity_proof :
  water_at_90_percent - water_at_40_percent = 36 ∧
  tank_capacity = 72 :=
sorry

end NUMINAMATH_CALUDE_tank_capacity_proof_l730_73062


namespace NUMINAMATH_CALUDE_heart_then_ten_probability_l730_73064

/-- The number of cards in a double standard deck -/
def deck_size : ℕ := 104

/-- The number of hearts in a double standard deck -/
def num_hearts : ℕ := 26

/-- The number of 10s in a double standard deck -/
def num_tens : ℕ := 8

/-- The number of 10 of hearts in a double standard deck -/
def num_ten_hearts : ℕ := 2

/-- The probability of drawing a heart as the first card and a 10 as the second card -/
def prob_heart_then_ten : ℚ := 47 / 2678

theorem heart_then_ten_probability :
  prob_heart_then_ten = 
    (num_hearts - num_ten_hearts) / deck_size * num_tens / (deck_size - 1) +
    num_ten_hearts / deck_size * (num_tens - num_ten_hearts) / (deck_size - 1) :=
by sorry

end NUMINAMATH_CALUDE_heart_then_ten_probability_l730_73064


namespace NUMINAMATH_CALUDE_min_value_expression_l730_73030

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : y > 2 * x) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (z : ℝ), z = (y^2 - 2*x*y + x^2) / (x*y - 2*x^2) → z ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l730_73030


namespace NUMINAMATH_CALUDE_inverse_of_A_squared_l730_73079

theorem inverse_of_A_squared (A : Matrix (Fin 2) (Fin 2) ℝ) : 
  A⁻¹ = !![3, 8; -2, -5] → (A^2)⁻¹ = !![(-7), (-16); 4, 9] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_A_squared_l730_73079


namespace NUMINAMATH_CALUDE_valid_marking_exists_l730_73088

/-- Represents a marking of cells in a 9x9 table -/
def Marking := Fin 9 → Fin 9 → Bool

/-- Checks if two adjacent rows have at least 6 marked cells -/
def validRows (m : Marking) : Prop :=
  ∀ i : Fin 8, (Finset.sum (Finset.univ.filter (λ j => m i j || m (i + 1) j)) (λ _ => 1) : ℕ) ≥ 6

/-- Checks if two adjacent columns have at most 5 marked cells -/
def validColumns (m : Marking) : Prop :=
  ∀ j : Fin 8, (Finset.sum (Finset.univ.filter (λ i => m i j || m i (j + 1))) (λ _ => 1) : ℕ) ≤ 5

/-- Theorem stating that a valid marking exists -/
theorem valid_marking_exists : ∃ m : Marking, validRows m ∧ validColumns m := by
  sorry

end NUMINAMATH_CALUDE_valid_marking_exists_l730_73088
