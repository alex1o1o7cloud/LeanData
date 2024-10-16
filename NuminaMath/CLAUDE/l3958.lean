import Mathlib

namespace NUMINAMATH_CALUDE_inequality1_solution_inequality2_solution_inequality3_solution_l3958_395843

-- Define the inequalities
def inequality1 (x : ℝ) := 2 * x^2 - 3 * x + 1 ≥ 0
def inequality2 (x : ℝ) := x^2 - 2 * x - 3 < 0
def inequality3 (x : ℝ) := -3 * x^2 + 5 * x - 2 > 0

-- Define the solution sets
def solution1 : Set ℝ := {x | x ≤ 1/2 ∨ x ≥ 1}
def solution2 : Set ℝ := {x | -1 < x ∧ x < 3}
def solution3 : Set ℝ := {x | 2/3 < x ∧ x < 1}

-- Theorem statements
theorem inequality1_solution : ∀ x : ℝ, x ∈ solution1 ↔ inequality1 x := by sorry

theorem inequality2_solution : ∀ x : ℝ, x ∈ solution2 ↔ inequality2 x := by sorry

theorem inequality3_solution : ∀ x : ℝ, x ∈ solution3 ↔ inequality3 x := by sorry

end NUMINAMATH_CALUDE_inequality1_solution_inequality2_solution_inequality3_solution_l3958_395843


namespace NUMINAMATH_CALUDE_solve_for_y_l3958_395868

theorem solve_for_y (x y : ℝ) : 4 * x + y = 9 → y = 9 - 4 * x := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3958_395868


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l3958_395862

-- Definition of opposite equations
def are_opposite_equations (a b : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ ∃ x y : ℝ, a * x - b = 0 ∧ b * y - a = 0

-- Part (1)
theorem part_one : 
  are_opposite_equations 4 3 → are_opposite_equations 3 c → c = 4 :=
sorry

-- Part (2)
theorem part_two :
  are_opposite_equations 4 (-3 * m - 1) → are_opposite_equations 5 (n - 2) → m / n = -1/3 :=
sorry

-- Part (3)
theorem part_three :
  (∃ x : ℤ, 3 * x - c = 0) → (∃ y : ℤ, c * y - 3 = 0) → c = 3 ∨ c = -3 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l3958_395862


namespace NUMINAMATH_CALUDE_solution_set_f_positive_max_m_inequality_l3958_395873

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 4|

-- Theorem for part I
theorem solution_set_f_positive :
  {x : ℝ | f x > 0} = {x : ℝ | x > 1 ∨ x < -5} :=
sorry

-- Theorem for part II
theorem max_m_inequality (m : ℝ) :
  (∀ x : ℝ, f x + 3*|x - 4| > m) ↔ m < 9 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_positive_max_m_inequality_l3958_395873


namespace NUMINAMATH_CALUDE_segment_rectangle_configurations_l3958_395852

/-- Represents a rectangle made of segments -/
structure SegmentRectangle where
  m : ℕ  -- number of segments on one side
  n : ℕ  -- number of segments on the other side

/-- The total number of segments in a rectangle -/
def total_segments (rect : SegmentRectangle) : ℕ :=
  rect.m * (rect.n + 1) + rect.n * (rect.m + 1)

/-- Possible configurations of a rectangle with 1997 segments -/
def is_valid_configuration (rect : SegmentRectangle) : Prop :=
  total_segments rect = 1997 ∧
  (rect.m = 2 ∧ rect.n = 399) ∨
  (rect.m = 8 ∧ rect.n = 117) ∨
  (rect.m = 23 ∧ rect.n = 42)

/-- Main theorem: The only valid configurations are 2×399, 8×117, and 23×42 -/
theorem segment_rectangle_configurations :
  ∀ rect : SegmentRectangle, total_segments rect = 1997 → is_valid_configuration rect :=
by sorry

end NUMINAMATH_CALUDE_segment_rectangle_configurations_l3958_395852


namespace NUMINAMATH_CALUDE_no_real_roots_for_nonzero_k_l3958_395802

theorem no_real_roots_for_nonzero_k (k : ℝ) (h : k ≠ 0) :
  ∀ x : ℝ, x^2 + 2*k*x + 3*k^2 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_for_nonzero_k_l3958_395802


namespace NUMINAMATH_CALUDE_ceiling_minus_x_l3958_395869

theorem ceiling_minus_x (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 2) : ⌈x⌉ - x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_minus_x_l3958_395869


namespace NUMINAMATH_CALUDE_least_coins_coins_exist_l3958_395893

theorem least_coins (n : ℕ) : 
  (n % 6 = 3) ∧ (n % 4 = 1) ∧ (n % 7 = 2) → n ≥ 9 :=
by sorry

theorem coins_exist : 
  ∃ n : ℕ, (n % 6 = 3) ∧ (n % 4 = 1) ∧ (n % 7 = 2) ∧ n = 9 :=
by sorry

end NUMINAMATH_CALUDE_least_coins_coins_exist_l3958_395893


namespace NUMINAMATH_CALUDE_honey_production_l3958_395890

theorem honey_production (bees : ℕ) (days : ℕ) (honey_per_bee : ℝ) :
  bees = 70 → days = 70 → honey_per_bee = 1 →
  bees * honey_per_bee = 70 := by
sorry

end NUMINAMATH_CALUDE_honey_production_l3958_395890


namespace NUMINAMATH_CALUDE_log_xy_value_l3958_395859

open Real

theorem log_xy_value (x y : ℝ) (h1 : log (x * y^4) = 1) (h2 : log (x^3 * y) = 1) : 
  log (x * y) = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_log_xy_value_l3958_395859


namespace NUMINAMATH_CALUDE_parabola_position_l3958_395815

/-- Represents a quadratic function of the form ax^2 + bx + c --/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem stating properties of a specific type of parabola --/
theorem parabola_position (f : QuadraticFunction) 
  (ha : f.a > 0) (hb : f.b > 0) (hc : f.c < 0) : 
  f.c < 0 ∧ -f.b / (2 * f.a) < 0 := by
  sorry

#check parabola_position

end NUMINAMATH_CALUDE_parabola_position_l3958_395815


namespace NUMINAMATH_CALUDE_equation_satisfied_for_all_x_l3958_395892

theorem equation_satisfied_for_all_x (a b c x : ℝ) 
  (h : a / b = 2 ∧ b / c = 3/4) : 
  (a + b) * (c - x) / a^2 - (b + c) * (x - 2*c) / (b*c) - 
  (c + a) * (c - 2*x) / (a*c) = (a + b) * c / (a*b) + 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfied_for_all_x_l3958_395892


namespace NUMINAMATH_CALUDE_r_profit_share_l3958_395899

/-- Represents a partner in the business partnership --/
inductive Partner
| P
| Q
| R

/-- Represents the initial share ratio of each partner --/
def initial_share_ratio (p : Partner) : Rat :=
  match p with
  | Partner.P => 1/2
  | Partner.Q => 1/3
  | Partner.R => 1/4

/-- The number of months after which P withdraws half of their capital --/
def withdrawal_month : Nat := 2

/-- The total number of months for the profit calculation --/
def total_months : Nat := 12

/-- The total profit to be divided --/
def total_profit : ℚ := 378

/-- Calculates the effective share ratio for a partner over the entire period --/
def effective_share_ratio (p : Partner) : Rat :=
  match p with
  | Partner.P => (initial_share_ratio Partner.P * withdrawal_month + initial_share_ratio Partner.P / 2 * (total_months - withdrawal_month)) / total_months
  | _ => initial_share_ratio p

/-- Calculates a partner's share of the profit --/
def profit_share (p : Partner) : ℚ :=
  (effective_share_ratio p / (effective_share_ratio Partner.P + effective_share_ratio Partner.Q + effective_share_ratio Partner.R)) * total_profit

/-- The main theorem stating R's share of the profit --/
theorem r_profit_share : profit_share Partner.R = 108 := by
  sorry


end NUMINAMATH_CALUDE_r_profit_share_l3958_395899


namespace NUMINAMATH_CALUDE_subset_of_complement_iff_a_in_range_l3958_395856

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M
def M (a : ℝ) : Set ℝ := {x | 3 * a - 1 < x ∧ x < 2 * a}

-- Define set N
def N : Set ℝ := {x | -1 < x ∧ x < 3}

-- State the theorem
theorem subset_of_complement_iff_a_in_range (a : ℝ) :
  N ⊆ (U \ M a) ↔ a ≤ -1/2 ∨ a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_subset_of_complement_iff_a_in_range_l3958_395856


namespace NUMINAMATH_CALUDE_product_of_three_consecutive_integers_divisible_by_six_l3958_395895

theorem product_of_three_consecutive_integers_divisible_by_six (k : ℤ) :
  ∃ m : ℤ, k * (k + 1) * (k + 2) = 6 * m :=
by sorry

end NUMINAMATH_CALUDE_product_of_three_consecutive_integers_divisible_by_six_l3958_395895


namespace NUMINAMATH_CALUDE_triangle_area_l3958_395883

def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (3, 5)

theorem triangle_area : 
  (1/2 : ℝ) * |a.1 * b.2 - a.2 * b.1| = 23/2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l3958_395883


namespace NUMINAMATH_CALUDE_solar_systems_per_planet_l3958_395872

theorem solar_systems_per_planet (total_bodies : ℕ) (planets : ℕ) : 
  total_bodies = 200 → planets = 20 → (total_bodies - planets) / planets = 9 := by
sorry

end NUMINAMATH_CALUDE_solar_systems_per_planet_l3958_395872


namespace NUMINAMATH_CALUDE_shark_teeth_problem_l3958_395845

theorem shark_teeth_problem (S : ℚ) 
  (hammerhead : ℚ → ℚ)
  (great_white : ℚ → ℚ)
  (h1 : hammerhead S = S / 6)
  (h2 : great_white S = 2 * (S + hammerhead S))
  (h3 : great_white S = 420) : 
  S = 180 := by sorry

end NUMINAMATH_CALUDE_shark_teeth_problem_l3958_395845


namespace NUMINAMATH_CALUDE_positive_integer_solution_iff_n_eq_three_l3958_395863

theorem positive_integer_solution_iff_n_eq_three (n : ℕ) :
  (∃ (x y z : ℕ+), x^2 + y^2 + z^2 = n * x * y * z) ↔ n = 3 :=
sorry

end NUMINAMATH_CALUDE_positive_integer_solution_iff_n_eq_three_l3958_395863


namespace NUMINAMATH_CALUDE_small_square_side_length_wire_cut_lengths_l3958_395857

/-- The total length of the wire in centimeters -/
def total_wire_length : ℝ := 64

/-- Theorem for the first part of the problem -/
theorem small_square_side_length
  (small_side : ℝ)
  (large_side : ℝ)
  (h1 : small_side > 0)
  (h2 : large_side > 0)
  (h3 : 4 * small_side + 4 * large_side = total_wire_length)
  (h4 : large_side^2 = 2.25 * small_side^2) :
  small_side = 6.4 := by sorry

/-- Theorem for the second part of the problem -/
theorem wire_cut_lengths
  (small_side : ℝ)
  (large_side : ℝ)
  (h1 : small_side > 0)
  (h2 : large_side > 0)
  (h3 : 4 * small_side + 4 * large_side = total_wire_length)
  (h4 : small_side^2 + large_side^2 = 160) :
  (4 * small_side = 16 ∧ 4 * large_side = 48) ∨
  (4 * small_side = 48 ∧ 4 * large_side = 16) := by sorry

end NUMINAMATH_CALUDE_small_square_side_length_wire_cut_lengths_l3958_395857


namespace NUMINAMATH_CALUDE_smallest_non_odd_unit_proof_l3958_395816

/-- The set of possible units digits for odd numbers -/
def odd_units : Set Nat := {1, 3, 5, 7, 9}

/-- A number is odd if and only if its units digit is in the odd_units set -/
def is_odd (n : Nat) : Prop := n % 10 ∈ odd_units

/-- The smallest digit not in the units place of an odd number -/
def smallest_non_odd_unit : Nat := 0

theorem smallest_non_odd_unit_proof :
  (∀ n : Nat, is_odd n → smallest_non_odd_unit ≠ n % 10) ∧
  (∀ d : Nat, d < smallest_non_odd_unit → ∃ n : Nat, is_odd n ∧ d = n % 10) :=
sorry

end NUMINAMATH_CALUDE_smallest_non_odd_unit_proof_l3958_395816


namespace NUMINAMATH_CALUDE_logs_per_tree_l3958_395896

theorem logs_per_tree (pieces_per_log : ℕ) (total_pieces : ℕ) (total_trees : ℕ)
  (h1 : pieces_per_log = 5)
  (h2 : total_pieces = 500)
  (h3 : total_trees = 25) :
  total_pieces / (pieces_per_log * total_trees) = 4 :=
by sorry

end NUMINAMATH_CALUDE_logs_per_tree_l3958_395896


namespace NUMINAMATH_CALUDE_perpendicular_tangent_line_l3958_395827

/-- Given a line L1 with equation x + 3y - 10 = 0 and a circle C with equation x^2 + y^2 = 4,
    prove that a line L2 perpendicular to L1 and tangent to C has the equation 3x - y ± 2√10 = 0 -/
theorem perpendicular_tangent_line 
  (L1 : ℝ → ℝ → Prop) 
  (C : ℝ → ℝ → Prop)
  (h1 : ∀ x y, L1 x y ↔ x + 3*y - 10 = 0)
  (h2 : ∀ x y, C x y ↔ x^2 + y^2 = 4) :
  ∃ L2 : ℝ → ℝ → Prop,
    (∀ x y, L2 x y ↔ (3*x - y = 2*Real.sqrt 10 ∨ 3*x - y = -2*Real.sqrt 10)) ∧
    (∀ x y, L1 x y → ∀ u v, L2 u v → (x - u) * (3 * (y - v)) = -(y - v) * (x - u)) ∧
    (∃ p q, L2 p q ∧ C p q ∧ ∀ x y, C x y → (x - p)^2 + (y - q)^2 ≥ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_tangent_line_l3958_395827


namespace NUMINAMATH_CALUDE_trapezoid_minimum_distance_l3958_395853

-- Define the trapezoid ABCD
def Trapezoid (A B C D : ℝ × ℝ) : Prop :=
  A.1 = 0 ∧ A.2 = 0 ∧
  B.1 = 0 ∧ B.2 = 12 ∧
  C.1 = 10 ∧ C.2 = 12 ∧
  D.1 = 10 ∧ D.2 = 6

-- Define the circle centered at C with radius 8
def Circle (C F : ℝ × ℝ) : Prop :=
  (F.1 - C.1)^2 + (F.2 - C.2)^2 = 64

-- Define point E on AB
def PointOnAB (A B E : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))

-- Define the theorem
theorem trapezoid_minimum_distance (A B C D E F : ℝ × ℝ) :
  Trapezoid A B C D →
  Circle C F →
  PointOnAB A B E →
  (∀ E' F', PointOnAB A B E' → Circle C F' →
    Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2) + Real.sqrt ((E.1 - F.1)^2 + (E.2 - F.2)^2) ≤
    Real.sqrt ((D.1 - E'.1)^2 + (D.2 - E'.2)^2) + Real.sqrt ((E'.1 - F'.1)^2 + (E'.2 - F'.2)^2)) →
  E.2 - A.2 = 4.5 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_minimum_distance_l3958_395853


namespace NUMINAMATH_CALUDE_train_speed_problem_l3958_395826

/-- Given two trains A and B with lengths 225 m and 150 m respectively,
    if it takes 15 seconds for train A to completely cross train B,
    then the speed of train A is 90 km/hr. -/
theorem train_speed_problem (length_A length_B time_to_cross : ℝ) :
  length_A = 225 →
  length_B = 150 →
  time_to_cross = 15 →
  (length_A + length_B) / time_to_cross * 3.6 = 90 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l3958_395826


namespace NUMINAMATH_CALUDE_x_value_proof_l3958_395822

theorem x_value_proof (x : ℝ) (h : 9 / (x^2) = x / 36) : x = (324 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3958_395822


namespace NUMINAMATH_CALUDE_nancy_carrots_l3958_395807

/-- The number of carrots Nancy picked the next day -/
def carrots_picked_next_day (initial_carrots : ℕ) (thrown_out : ℕ) (total_carrots : ℕ) : ℕ :=
  total_carrots - (initial_carrots - thrown_out)

/-- Proof that Nancy picked 21 carrots the next day -/
theorem nancy_carrots :
  carrots_picked_next_day 12 2 31 = 21 := by
  sorry

end NUMINAMATH_CALUDE_nancy_carrots_l3958_395807


namespace NUMINAMATH_CALUDE_range_of_a_l3958_395808

-- Define propositions p and q
def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0
def q (x a : ℝ) : Prop := x > a

-- Define the property that ¬q is sufficient but not necessary for ¬p
def not_q_sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, ¬(q x a) → ¬(p x)) ∧ ¬(∀ x, ¬(p x) → ¬(q x a))

-- Theorem statement
theorem range_of_a (a : ℝ) (h : not_q_sufficient_not_necessary a) : a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3958_395808


namespace NUMINAMATH_CALUDE_implication_equivalences_l3958_395835

variable (p q : Prop)

theorem implication_equivalences (h : p → q) :
  (∃ (f : p → q), True) ∧
  (p → q) ∧
  (¬q → ¬p) ∧
  ((p → q) ∧ (¬p ∨ q)) :=
by sorry

end NUMINAMATH_CALUDE_implication_equivalences_l3958_395835


namespace NUMINAMATH_CALUDE_twenty_four_game_solution_l3958_395879

theorem twenty_four_game_solution : 
  let a : ℝ := 5
  let b : ℝ := 5
  let c : ℝ := 5
  let d : ℝ := 1
  (a - d / b) * c = 24 := by
  sorry

end NUMINAMATH_CALUDE_twenty_four_game_solution_l3958_395879


namespace NUMINAMATH_CALUDE_intersection_angle_l3958_395809

theorem intersection_angle (φ : Real) 
  (h1 : 0 ≤ φ) (h2 : φ < π)
  (h3 : 2 * Real.cos (π/3) = 2 * Real.sin (2 * (π/3) + φ)) :
  φ = π/6 := by sorry

end NUMINAMATH_CALUDE_intersection_angle_l3958_395809


namespace NUMINAMATH_CALUDE_middle_angle_range_l3958_395889

theorem middle_angle_range (β : Real) (h1 : 0 < β) (h2 : β < 90) : 
  ∃ (α γ : Real), 
    0 < α ∧ 0 < γ ∧ 
    α + β + γ = 180 ∧ 
    α ≤ β ∧ β ≤ γ :=
by sorry

end NUMINAMATH_CALUDE_middle_angle_range_l3958_395889


namespace NUMINAMATH_CALUDE_workforce_from_company_a_l3958_395874

/-- Represents the workforce composition of a company -/
structure WorkforceComposition where
  managers : Real
  software_engineers : Real
  marketing : Real
  human_resources : Real
  support_staff : Real

/-- The workforce composition of Company A -/
def company_a : WorkforceComposition := {
  managers := 0.10,
  software_engineers := 0.70,
  marketing := 0.15,
  human_resources := 0.05,
  support_staff := 0
}

/-- The workforce composition of Company B -/
def company_b : WorkforceComposition := {
  managers := 0.25,
  software_engineers := 0.10,
  marketing := 0.15,
  human_resources := 0.05,
  support_staff := 0.45
}

/-- The workforce composition of the merged company -/
def merged_company : WorkforceComposition := {
  managers := 0.18,
  software_engineers := 0,
  marketing := 0,
  human_resources := 0.10,
  support_staff := 0.50
}

/-- The theorem stating the percentage of workforce from Company A in the merged company -/
theorem workforce_from_company_a : 
  ∃ (total_a total_b : Real), 
    total_a > 0 ∧ total_b > 0 ∧
    company_a.managers * total_a + company_b.managers * total_b = merged_company.managers * (total_a + total_b) ∧
    total_a / (total_a + total_b) = 7 / 15 := by
  sorry

#check workforce_from_company_a

end NUMINAMATH_CALUDE_workforce_from_company_a_l3958_395874


namespace NUMINAMATH_CALUDE_fish_price_proof_l3958_395870

theorem fish_price_proof (discount_rate : ℝ) (discounted_price : ℝ) (package_weight : ℝ) :
  discount_rate = 0.4 →
  discounted_price = 2 →
  package_weight = 1/4 →
  (1 - discount_rate) * (1 / package_weight) * discounted_price = 40/3 := by
  sorry

end NUMINAMATH_CALUDE_fish_price_proof_l3958_395870


namespace NUMINAMATH_CALUDE_triangle_area_proof_l3958_395839

/-- The area of the triangle formed by the lines x = 4, y = 3x, and the positive x-axis -/
def triangle_area : ℝ := 24

/-- The x-coordinate of the vertical line -/
def vertical_line : ℝ := 4

/-- The slope of the diagonal line -/
def diagonal_slope : ℝ := 3

theorem triangle_area_proof :
  let x_intercept : ℝ := vertical_line
  let y_intercept : ℝ := diagonal_slope * vertical_line
  let base : ℝ := x_intercept
  let height : ℝ := y_intercept
  (1/2 : ℝ) * base * height = triangle_area := by sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l3958_395839


namespace NUMINAMATH_CALUDE_integer_fractions_l3958_395884

theorem integer_fractions (x : ℤ) : 
  (∃ k : ℤ, (5 * x^3 - x + 17) = 15 * k) ∧ 
  (∃ m : ℤ, (2 * x^2 + x - 3) = 7 * m) ↔ 
  (∃ t : ℤ, x = 105 * t + 22 ∨ x = 105 * t + 37) :=
sorry

end NUMINAMATH_CALUDE_integer_fractions_l3958_395884


namespace NUMINAMATH_CALUDE_ashas_borrowed_amount_l3958_395841

theorem ashas_borrowed_amount (brother mother granny savings spent_fraction remaining : ℚ)
  (h1 : brother = 20)
  (h2 : mother = 30)
  (h3 : granny = 70)
  (h4 : savings = 100)
  (h5 : spent_fraction = 3/4)
  (h6 : remaining = 65)
  (h7 : (1 - spent_fraction) * (brother + mother + granny + savings + father) = remaining) :
  father = 40 := by
  sorry

end NUMINAMATH_CALUDE_ashas_borrowed_amount_l3958_395841


namespace NUMINAMATH_CALUDE_fourth_student_added_25_l3958_395877

/-- The number of jellybeans added by the fourth student to the average of the first three guesses -/
def jellybeans_added (first_guess : ℕ) (fourth_guess : ℕ) : ℕ :=
  let second_guess := 8 * first_guess
  let third_guess := second_guess - 200
  let average := (first_guess + second_guess + third_guess) / 3
  fourth_guess - average

/-- Theorem stating that given the conditions in the problem, the fourth student added 25 jellybeans -/
theorem fourth_student_added_25 :
  jellybeans_added 100 525 = 25 := by
  sorry

#eval jellybeans_added 100 525

end NUMINAMATH_CALUDE_fourth_student_added_25_l3958_395877


namespace NUMINAMATH_CALUDE_egyptian_fraction_identity_l3958_395818

theorem egyptian_fraction_identity (n : ℕ+) :
  (2 : ℚ) / (2 * n + 1) = 1 / (n + 1) + 1 / ((n + 1) * (2 * n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_egyptian_fraction_identity_l3958_395818


namespace NUMINAMATH_CALUDE_M_intersect_N_l3958_395829

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem M_intersect_N : M ∩ N = {0, 2} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_l3958_395829


namespace NUMINAMATH_CALUDE_wire_cutting_l3958_395847

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_length : ℝ) : 
  total_length = 70 →
  ratio = 2 / 5 →
  shorter_length + (shorter_length / ratio) = total_length →
  shorter_length = 20 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l3958_395847


namespace NUMINAMATH_CALUDE_probability_human_given_id_as_human_l3958_395864

-- Define the total population
def total_population : ℝ := 1000

-- Define the proportion of vampires and humans
def vampire_proportion : ℝ := 0.99
def human_proportion : ℝ := 1 - vampire_proportion

-- Define the correct identification rates
def vampire_correct_id_rate : ℝ := 0.9
def human_correct_id_rate : ℝ := 0.9

-- Define the number of vampires and humans
def num_vampires : ℝ := vampire_proportion * total_population
def num_humans : ℝ := human_proportion * total_population

-- Define the number of correctly and incorrectly identified vampires and humans
def vampires_id_as_vampires : ℝ := vampire_correct_id_rate * num_vampires
def vampires_id_as_humans : ℝ := (1 - vampire_correct_id_rate) * num_vampires
def humans_id_as_humans : ℝ := human_correct_id_rate * num_humans
def humans_id_as_vampires : ℝ := (1 - human_correct_id_rate) * num_humans

-- Define the total number of individuals identified as humans
def total_id_as_humans : ℝ := vampires_id_as_humans + humans_id_as_humans

-- Theorem statement
theorem probability_human_given_id_as_human :
  humans_id_as_humans / total_id_as_humans = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_human_given_id_as_human_l3958_395864


namespace NUMINAMATH_CALUDE_inheritance_calculation_l3958_395837

theorem inheritance_calculation (x : ℝ) 
  (h1 : x > 0)
  (h2 : 0.3 * x + 0.12 * (0.7 * x) + 0.05 * (0.7 * x - 0.12 * (0.7 * x)) = 16800) :
  x = 40500 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l3958_395837


namespace NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circles_l3958_395803

/-- The area of a rectangle with two circles of radius 7 cm inscribed in opposite corners is 196 cm². -/
theorem rectangle_area_with_inscribed_circles (r : ℝ) (h : r = 7) :
  let diameter := 2 * r
  let length := diameter
  let width := diameter
  let area := length * width
  area = 196 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circles_l3958_395803


namespace NUMINAMATH_CALUDE_fraction_equalities_l3958_395825

theorem fraction_equalities (x y : ℚ) (h : x / y = 5 / 6) : 
  ((3 * x + 2 * y) / y = 9 / 2) ∧ 
  (y / (2 * x - y) = 3 / 2) ∧ 
  ((x - 3 * y) / y = -13 / 6) ∧ 
  ((2 * x) / (3 * y) = 5 / 9) ∧ 
  ((x + y) / (2 * y) = 11 / 12) := by
  sorry


end NUMINAMATH_CALUDE_fraction_equalities_l3958_395825


namespace NUMINAMATH_CALUDE_garage_sale_earnings_l3958_395811

/-- The total earnings from selling necklaces at a garage sale -/
def total_earnings (bead_count gemstone_count crystal_count wooden_count : ℕ)
                   (bead_price gemstone_price crystal_price wooden_price : ℕ) : ℕ :=
  bead_count * bead_price + 
  gemstone_count * gemstone_price + 
  crystal_count * crystal_price + 
  wooden_count * wooden_price

/-- Theorem stating that the total earnings from selling the specified necklaces is $53 -/
theorem garage_sale_earnings : 
  total_earnings 4 3 2 5 3 7 5 2 = 53 := by
  sorry

end NUMINAMATH_CALUDE_garage_sale_earnings_l3958_395811


namespace NUMINAMATH_CALUDE_perimeter_calculation_l3958_395876

theorem perimeter_calculation : 
  let segments : List ℕ := [2, 3, 2, 6, 2, 4, 3]
  segments.sum = 22 := by sorry

end NUMINAMATH_CALUDE_perimeter_calculation_l3958_395876


namespace NUMINAMATH_CALUDE_reduced_rate_start_time_l3958_395823

/-- The fraction of a week during which reduced rates apply -/
def reduced_rate_fraction : ℝ := 0.6428571428571429

/-- The number of hours in a week -/
def hours_per_week : ℕ := 24 * 7

/-- The number of weekend days (Saturday and Sunday) -/
def weekend_days : ℕ := 2

/-- The time (in hours) when reduced rates end on weekdays -/
def reduced_rate_end : ℕ := 8

theorem reduced_rate_start_time :
  ∃ (start_time : ℕ),
    start_time = 20 ∧  -- 8 p.m. is 20 in 24-hour format
    (1 - reduced_rate_fraction) * hours_per_week =
      (5 * (reduced_rate_end + (24 - start_time))) +
      (weekend_days * 24) :=
by sorry

end NUMINAMATH_CALUDE_reduced_rate_start_time_l3958_395823


namespace NUMINAMATH_CALUDE_divisibility_condition_l3958_395881

theorem divisibility_condition (n : ℕ+) : (n + 1) ∣ (n^2 + 1) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3958_395881


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3958_395820

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ (x - 6) / 16 = 6 / (x - 16) := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l3958_395820


namespace NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_a_l3958_395830

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Theorem for the solution set of f(x) ≥ 3
theorem solution_set_f_geq_3 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ -3/2} ∪ {x : ℝ | x ≥ 3/2} := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∀ x, f x > a^2 - x^2 + 2*x} = {a : ℝ | -Real.sqrt 5 < a ∧ a < Real.sqrt 5} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_a_l3958_395830


namespace NUMINAMATH_CALUDE_parabola_c_value_l3958_395887

/-- Given a parabola y = ax^2 + bx + c with vertex (3, -5) passing through (1, -3),
    prove that c = -0.5 -/
theorem parabola_c_value (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →   -- Parabola equation
  (3, -5) = (3, a * 3^2 + b * 3 + c) →     -- Vertex condition
  -3 = a * 1^2 + b * 1 + c →               -- Point condition
  c = -0.5 := by
sorry


end NUMINAMATH_CALUDE_parabola_c_value_l3958_395887


namespace NUMINAMATH_CALUDE_initial_alcohol_percentage_l3958_395878

/-- Proves that the initial alcohol percentage in a mixture is 20% given the conditions --/
theorem initial_alcohol_percentage 
  (initial_volume : ℝ) 
  (added_water : ℝ) 
  (final_percentage : ℝ) : ℝ :=
  by
  have h1 : initial_volume = 18 := by sorry
  have h2 : added_water = 3 := by sorry
  have h3 : final_percentage = 17.14285714285715 := by sorry
  
  let final_volume : ℝ := initial_volume + added_water
  let initial_percentage : ℝ := (final_percentage * final_volume) / initial_volume
  
  have h4 : initial_percentage = 20 := by sorry
  
  exact initial_percentage

end NUMINAMATH_CALUDE_initial_alcohol_percentage_l3958_395878


namespace NUMINAMATH_CALUDE_eight_steps_result_l3958_395886

def alternate_divide_multiply (n : ℕ) : ℕ → ℕ
  | 0 => n
  | i + 1 => if i % 2 = 0 then (alternate_divide_multiply n i) / 2 else (alternate_divide_multiply n i) * 3

theorem eight_steps_result :
  alternate_divide_multiply 10000000 8 = 2^3 * 3^4 * 5^7 := by
  sorry

end NUMINAMATH_CALUDE_eight_steps_result_l3958_395886


namespace NUMINAMATH_CALUDE_impossibleTransformation_l3958_395855

/-- Represents the state of a cell in the table -/
inductive CellState
  | Zero
  | One

/-- Represents the n × n table -/
def Table (n : ℕ) := Fin n → Fin n → CellState

/-- The initial table state with n-1 ones and the rest zeros -/
def initialTable (n : ℕ) : Table n := sorry

/-- The operation of choosing a cell, subtracting one from it,
    and adding one to all other numbers in the same row or column -/
def applyOperation (t : Table n) (row col : Fin n) : Table n := sorry

/-- Checks if all cells in the table have the same value -/
def allEqual (t : Table n) : Prop := sorry

/-- The main theorem stating that it's impossible to transform the initial table
    into a table with all equal numbers using the given operations -/
theorem impossibleTransformation (n : ℕ) :
  ¬ ∃ (ops : List (Fin n × Fin n)), allEqual (ops.foldl (λ t (rc : Fin n × Fin n) => applyOperation t rc.1 rc.2) (initialTable n)) :=
sorry

end NUMINAMATH_CALUDE_impossibleTransformation_l3958_395855


namespace NUMINAMATH_CALUDE_logarithm_expression_equality_l3958_395810

theorem logarithm_expression_equality : 
  (Real.log 243 / Real.log 3) / (Real.log 81 / Real.log 3) - 
  (Real.log 729 / Real.log 3) / (Real.log 27 / Real.log 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equality_l3958_395810


namespace NUMINAMATH_CALUDE_complex_product_sum_equality_l3958_395838

/-- Given n complex numbers with modulus a > 0, f(n, m) is the sum of all products of m of these numbers -/
def f (n m : ℕ) (z : ℕ → ℂ) : ℂ := sorry

/-- The theorem states that |f(n,m)| / a^m = |f(n,n-m)| / a^(n-m) -/
theorem complex_product_sum_equality (n m : ℕ) (a : ℝ) (z : ℕ → ℂ) 
  (ha : a > 0) (hz : ∀ i, Complex.abs (z i) = a) :
  Complex.abs (f n m z) / a^m = Complex.abs (f n (n-m) z) / a^(n-m) := by
  sorry

end NUMINAMATH_CALUDE_complex_product_sum_equality_l3958_395838


namespace NUMINAMATH_CALUDE_glycol_concentration_mixture_l3958_395836

/-- Proves that mixing 16 gallons of 40% glycol concentration with 8 gallons of 10% glycol concentration 
    results in a 30% glycol concentration in the final 24-gallon mixture. -/
theorem glycol_concentration_mixture 
  (total_volume : ℝ) 
  (volume_40_percent : ℝ)
  (volume_10_percent : ℝ)
  (concentration_40_percent : ℝ)
  (concentration_10_percent : ℝ)
  (h1 : total_volume = 24)
  (h2 : volume_40_percent = 16)
  (h3 : volume_10_percent = 8)
  (h4 : concentration_40_percent = 0.4)
  (h5 : concentration_10_percent = 0.1)
  (h6 : volume_40_percent + volume_10_percent = total_volume) :
  (volume_40_percent * concentration_40_percent + volume_10_percent * concentration_10_percent) / total_volume = 0.3 := by
  sorry


end NUMINAMATH_CALUDE_glycol_concentration_mixture_l3958_395836


namespace NUMINAMATH_CALUDE_total_cost_is_48_l3958_395897

/-- The cost of a single sandwich in dollars -/
def sandwich_cost : ℕ := 4

/-- The cost of a single soda in dollars -/
def soda_cost : ℕ := 3

/-- The cost of a single cookie in dollars -/
def cookie_cost : ℕ := 2

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 4

/-- The number of sodas purchased -/
def num_sodas : ℕ := 6

/-- The number of cookies purchased -/
def num_cookies : ℕ := 7

/-- Theorem stating that the total cost of the purchase is 48 dollars -/
theorem total_cost_is_48 : 
  num_sandwiches * sandwich_cost + num_sodas * soda_cost + num_cookies * cookie_cost = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_48_l3958_395897


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l3958_395842

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_middle_term
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_positive : ∀ n, a n > 0)
  (h_a2 : a 2 = 2)
  (h_a8 : a 8 = 32) :
  a 5 = 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l3958_395842


namespace NUMINAMATH_CALUDE_base_27_number_divisibility_l3958_395867

theorem base_27_number_divisibility (n : ℕ) : 
  (∃ (a b c d e f g h i j k l m o p q r s t u v w x y z : ℕ),
    (∀ digit ∈ [a, b, c, d, e, f, g, h, i, j, k, l, m, o, p, q, r, s, t, u, v, w, x, y, z], 
      1 ≤ digit ∧ digit ≤ 26) ∧
    n = a * 27^25 + b * 27^24 + c * 27^23 + d * 27^22 + e * 27^21 + f * 27^20 + 
        g * 27^19 + h * 27^18 + i * 27^17 + j * 27^16 + k * 27^15 + l * 27^14 + 
        m * 27^13 + o * 27^12 + p * 27^11 + q * 27^10 + r * 27^9 + s * 27^8 + 
        t * 27^7 + u * 27^6 + v * 27^5 + w * 27^4 + x * 27^3 + y * 27^2 + z * 27^1 + 26) →
  n % 100 = 0 := by
sorry

end NUMINAMATH_CALUDE_base_27_number_divisibility_l3958_395867


namespace NUMINAMATH_CALUDE_green_ball_count_l3958_395831

theorem green_ball_count (blue_count : ℕ) (ratio_blue : ℕ) (ratio_green : ℕ) 
  (h1 : blue_count = 16)
  (h2 : ratio_blue = 4)
  (h3 : ratio_green = 3) :
  (blue_count * ratio_green) / ratio_blue = 12 :=
by sorry

end NUMINAMATH_CALUDE_green_ball_count_l3958_395831


namespace NUMINAMATH_CALUDE_jack_payment_l3958_395821

/-- The amount Jack paid for sandwiches -/
def amount_paid : ℕ := sorry

/-- The number of sandwiches Jack ordered -/
def num_sandwiches : ℕ := 3

/-- The cost of each sandwich in dollars -/
def cost_per_sandwich : ℕ := 5

/-- The amount of change Jack received in dollars -/
def change_received : ℕ := 5

/-- Theorem stating that the amount Jack paid is $20 -/
theorem jack_payment : amount_paid = 20 := by
  sorry

end NUMINAMATH_CALUDE_jack_payment_l3958_395821


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3958_395848

/-- The trajectory of a point P satisfying |PF₂| - |PF₁| = 4, where F₁(-4, 0) and F₂(4, 0) are fixed points -/
def hyperbola_trajectory (P : ℝ × ℝ) : Prop :=
  let F₁ : ℝ × ℝ := (-4, 0)
  let F₂ : ℝ × ℝ := (4, 0)
  let d (A B : ℝ × ℝ) := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  d P F₂ - d P F₁ = 4 →
  P.1^2 / 4 - P.2^2 / 12 = 1 ∧ P.1 ≤ -2

theorem hyperbola_equation : 
  ∀ P : ℝ × ℝ, hyperbola_trajectory P :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3958_395848


namespace NUMINAMATH_CALUDE_editing_posting_time_is_zero_l3958_395834

/-- Represents the time in hours for various activities in video production -/
structure VideoProductionTime where
  setup : ℝ
  painting : ℝ
  cleanup : ℝ
  total : ℝ

/-- The time spent on editing and posting each video -/
def editingPostingTime (t : VideoProductionTime) : ℝ :=
  t.total - (t.setup + t.painting + t.cleanup)

/-- Theorem stating that the editing and posting time is 0 hours -/
theorem editing_posting_time_is_zero (t : VideoProductionTime)
  (h_setup : t.setup = 1)
  (h_painting : t.painting = 1)
  (h_cleanup : t.cleanup = 1)
  (h_total : t.total = 3) :
  editingPostingTime t = 0 := by
  sorry

end NUMINAMATH_CALUDE_editing_posting_time_is_zero_l3958_395834


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3958_395801

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 1 / b) ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3958_395801


namespace NUMINAMATH_CALUDE_second_polygon_sides_l3958_395840

theorem second_polygon_sides (n₁ n₂ : ℕ) (s₁ s₂ : ℝ) :
  n₁ = 50 →
  s₁ = 3 * s₂ →
  n₁ * s₁ = n₂ * s₂ →
  n₂ = 150 := by
sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l3958_395840


namespace NUMINAMATH_CALUDE_shipping_cost_formula_l3958_395854

/-- The cost function for shipping a package -/
def shippingCost (P : ℕ) : ℕ :=
  if P ≤ 5 then 5 * P + 10 else 5 * P + 5

theorem shipping_cost_formula (P : ℕ) (h : P ≥ 1) :
  (P ≤ 5 → shippingCost P = 5 * P + 10) ∧
  (P > 5 → shippingCost P = 5 * P + 5) := by
  sorry

end NUMINAMATH_CALUDE_shipping_cost_formula_l3958_395854


namespace NUMINAMATH_CALUDE_sum_of_roots_eq_two_l3958_395865

theorem sum_of_roots_eq_two :
  let f : ℝ → ℝ := λ x => (x + 3) * (x - 5) - 19
  (∃ a b : ℝ, (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b) ∧ a + b = 2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_eq_two_l3958_395865


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3958_395860

theorem unique_positive_solution : ∃! x : ℚ, x > 0 ∧ 3 * x^2 + 7 * x - 20 = 0 :=
by
  -- The unique positive solution is 5/3
  use 5/3
  constructor
  · -- Prove that 5/3 satisfies the conditions
    constructor
    · -- Prove 5/3 > 0
      sorry
    · -- Prove 3 * (5/3)^2 + 7 * (5/3) - 20 = 0
      sorry
  · -- Prove uniqueness
    sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l3958_395860


namespace NUMINAMATH_CALUDE_ellipse_condition_equiv_k_range_ellipse_standard_equation_l3958_395824

-- Define the curve C
def curve_C (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (4 - k) - y^2 / (1 - k) = 1

-- Define the condition for an ellipse with foci on the x-axis
def is_ellipse_x_axis (k : ℝ) : Prop :=
  4 - k > 0 ∧ k - 1 > 0 ∧ 4 - k > k - 1

-- Define the range of k
def k_range (k : ℝ) : Prop :=
  1 < k ∧ k < 5/2

-- Define the ellipse passing through (√6, 2) with foci at (-2,0) and (2,0)
def ellipse_through_point (x y : ℝ) : Prop :=
  x^2 / 12 + y^2 / 8 = 1

-- Theorem 1: Equivalence of ellipse condition and k range
theorem ellipse_condition_equiv_k_range (k : ℝ) :
  is_ellipse_x_axis k ↔ k_range k :=
sorry

-- Theorem 2: Standard equation of the ellipse
theorem ellipse_standard_equation :
  ellipse_through_point (Real.sqrt 6) 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_equiv_k_range_ellipse_standard_equation_l3958_395824


namespace NUMINAMATH_CALUDE_max_area_triangle_abc_l3958_395832

theorem max_area_triangle_abc (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) : 
  let angle_C : ℝ := π / 3
  let area := (1 / 2) * a * b * Real.sin angle_C
  3 * a * b = 25 - c^2 →
  ∀ (a' b' c' : ℝ), 
    a' > 0 → b' > 0 → c' > 0 →
    3 * a' * b' = 25 - c'^2 →
    area ≤ ((25 : ℝ) / 16) * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_max_area_triangle_abc_l3958_395832


namespace NUMINAMATH_CALUDE_rectangle_area_l3958_395800

/-- The area of a rectangle with perimeter 90 feet and length three times the width is 380.15625 square feet. -/
theorem rectangle_area (w : ℝ) (l : ℝ) (h1 : 2 * l + 2 * w = 90) (h2 : l = 3 * w) :
  l * w = 380.15625 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3958_395800


namespace NUMINAMATH_CALUDE_min_value_expression_l3958_395833

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  6 * Real.sqrt (a * b) + 3 / a + 3 / b ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3958_395833


namespace NUMINAMATH_CALUDE_geometric_subseq_implies_arithmetic_indices_l3958_395880

/-- Given a geometric sequence with common ratio q ≠ 1, if three terms form a geometric sequence,
    then their indices form an arithmetic sequence. -/
theorem geometric_subseq_implies_arithmetic_indices
  (a : ℕ → ℝ) (q : ℝ) (m n p : ℕ) (hq : q ≠ 1)
  (h_geom : ∀ k, a (k + 1) = q * a k)
  (h_subseq : (a n)^2 = a m * a p) :
  2 * n = m + p :=
sorry

end NUMINAMATH_CALUDE_geometric_subseq_implies_arithmetic_indices_l3958_395880


namespace NUMINAMATH_CALUDE_students_not_finding_parents_funny_l3958_395875

theorem students_not_finding_parents_funny 
  (total : ℕ) 
  (funny_dad : ℕ) 
  (funny_mom : ℕ) 
  (funny_both : ℕ) 
  (h1 : total = 50) 
  (h2 : funny_dad = 25) 
  (h3 : funny_mom = 30) 
  (h4 : funny_both = 18) : 
  total - (funny_dad + funny_mom - funny_both) = 13 := by
  sorry

end NUMINAMATH_CALUDE_students_not_finding_parents_funny_l3958_395875


namespace NUMINAMATH_CALUDE_fischer_random_chess_positions_l3958_395898

/-- Represents the number of squares on one row of a chessboard -/
def boardSize : Nat := 8

/-- Represents the number of dark (or light) squares on one row -/
def darkSquares : Nat := boardSize / 2

/-- Represents the number of squares available for queen and knights after placing bishops -/
def remainingSquares : Nat := boardSize - 2

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : Nat) : Nat :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Represents the number of ways to arrange bishops on opposite colors -/
def bishopArrangements : Nat := darkSquares * darkSquares

/-- Represents the number of ways to choose positions for queen and knights -/
def queenKnightPositions : Nat := choose remainingSquares 3

/-- Represents the number of permutations for queen and knights -/
def queenKnightPermutations : Nat := Nat.factorial 3

/-- Represents the total number of ways to arrange queen and knights -/
def queenKnightArrangements : Nat := queenKnightPositions * queenKnightPermutations

/-- Represents the number of ways to arrange king between rooks -/
def kingRookArrangements : Nat := 1

/-- The main theorem stating the number of starting positions in Fischer Random Chess -/
theorem fischer_random_chess_positions :
  bishopArrangements * queenKnightArrangements * kingRookArrangements = 1920 := by
  sorry


end NUMINAMATH_CALUDE_fischer_random_chess_positions_l3958_395898


namespace NUMINAMATH_CALUDE_last_three_digits_of_5_to_9000_l3958_395844

theorem last_three_digits_of_5_to_9000 (h : 5^300 ≡ 1 [ZMOD 1250]) :
  5^9000 ≡ 1 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_5_to_9000_l3958_395844


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3958_395813

/-- Given an arithmetic sequence {a_n} where a₁ + 3a₈ + a₁₅ = 120, prove that a₂ + a₁₄ = 48. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence condition
  a 1 + 3 * a 8 + a 15 = 120 →                      -- given condition
  a 2 + a 14 = 48 := by                             -- conclusion to prove
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3958_395813


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3958_395828

theorem no_integer_solutions (c : ℕ) (hc_pos : c > 0) (hc_odd : Odd c) :
  ¬∃ (x y : ℤ), x^2 - y^3 = (2*c)^3 - 1 :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3958_395828


namespace NUMINAMATH_CALUDE_surface_area_is_64_l3958_395891

/-- The surface area of a figure assembled from four identical blocks -/
def figure_surface_area (block_area : ℝ) (lost_area : ℝ) : ℝ :=
  4 * (block_area - lost_area)

/-- Theorem: The surface area of the assembled figure is 64 cm² -/
theorem surface_area_is_64 :
  figure_surface_area 18 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_is_64_l3958_395891


namespace NUMINAMATH_CALUDE_unique_set_A_l3958_395812

def A : Finset ℕ := {2, 3, 4, 5}

def B : Finset ℕ := {24, 30, 40, 60}

def three_products (S : Finset ℕ) : Finset ℕ :=
  S.powerset.filter (λ s => s.card = 3) |>.image (λ s => s.prod id)

theorem unique_set_A : 
  ∀ S : Finset ℕ, S.card = 4 → three_products S = B → S = A := by
  sorry

end NUMINAMATH_CALUDE_unique_set_A_l3958_395812


namespace NUMINAMATH_CALUDE_value_of_k_l3958_395871

theorem value_of_k (k : ℝ) (h : 16 / k = 4) : k = 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_k_l3958_395871


namespace NUMINAMATH_CALUDE_squirrel_climb_time_l3958_395861

/-- Represents the climbing behavior of a squirrel -/
structure SquirrelClimb where
  climb_rate : ℕ  -- metres climbed in odd minutes
  slip_rate : ℕ   -- metres slipped in even minutes
  total_height : ℕ -- total height of the pole to climb

/-- Calculates the time taken for a squirrel to climb a pole -/
def climb_time (s : SquirrelClimb) : ℕ :=
  sorry

/-- Theorem: A squirrel with given climbing behavior takes 17 minutes to climb 26 metres -/
theorem squirrel_climb_time :
  let s : SquirrelClimb := { climb_rate := 5, slip_rate := 2, total_height := 26 }
  climb_time s = 17 :=
by sorry

end NUMINAMATH_CALUDE_squirrel_climb_time_l3958_395861


namespace NUMINAMATH_CALUDE_abs_ratio_eq_sqrt_five_half_l3958_395866

theorem abs_ratio_eq_sqrt_five_half (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + y^2 = 18*x*y) : 
  |((x+y)/(x-y))| = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_ratio_eq_sqrt_five_half_l3958_395866


namespace NUMINAMATH_CALUDE_solution_set_of_f_minimum_value_of_sum_l3958_395888

-- Part I
def f (x : ℝ) : ℝ := 4 - |x| - |x - 3|

theorem solution_set_of_f (x : ℝ) : f (x + 3/2) ≥ 0 ↔ x ∈ Set.Icc (-2 : ℝ) (2 : ℝ) := by sorry

-- Part II
theorem minimum_value_of_sum (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (h : 1/(3*p) + 1/(2*q) + 1/r = 4) : 
  3*p + 2*q + r ≥ 9/4 ∧ ∃ (p' q' r' : ℝ), p' > 0 ∧ q' > 0 ∧ r' > 0 ∧ 
    1/(3*p') + 1/(2*q') + 1/r' = 4 ∧ 3*p' + 2*q' + r' = 9/4 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_f_minimum_value_of_sum_l3958_395888


namespace NUMINAMATH_CALUDE_root_sum_fraction_l3958_395858

theorem root_sum_fraction (r₁ r₂ r₃ r₄ : ℂ) : 
  (r₁ * r₁ + r₂ * r₂ + r₃ * r₃ + r₄ * r₄ = 0) →
  (r₁ + r₂ + r₃ + r₄ = 4) →
  (r₁ * r₂ + r₁ * r₃ + r₁ * r₄ + r₂ * r₃ + r₂ * r₄ + r₃ * r₄ = 8) →
  (r₁^4 - 4*r₁^3 + 8*r₁^2 - 7*r₁ + 3 = 0) →
  (r₂^4 - 4*r₂^3 + 8*r₂^2 - 7*r₂ + 3 = 0) →
  (r₃^4 - 4*r₃^3 + 8*r₃^2 - 7*r₃ + 3 = 0) →
  (r₄^4 - 4*r₄^3 + 8*r₄^2 - 7*r₄ + 3 = 0) →
  (r₁^2 / (r₂^2 + r₃^2 + r₄^2) + r₂^2 / (r₁^2 + r₃^2 + r₄^2) + 
   r₃^2 / (r₁^2 + r₂^2 + r₄^2) + r₄^2 / (r₁^2 + r₂^2 + r₃^2) = -4) := by
sorry

end NUMINAMATH_CALUDE_root_sum_fraction_l3958_395858


namespace NUMINAMATH_CALUDE_max_value_x_sqrt_3_minus_x_squared_l3958_395851

theorem max_value_x_sqrt_3_minus_x_squared :
  ∀ x : ℝ, 0 < x → x < Real.sqrt 3 →
    x * Real.sqrt (3 - x^2) ≤ 9/4 ∧
    ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ < Real.sqrt 3 ∧ x₀ * Real.sqrt (3 - x₀^2) = 9/4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_sqrt_3_minus_x_squared_l3958_395851


namespace NUMINAMATH_CALUDE_tangent_angle_at_origin_l3958_395850

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_angle_at_origin :
  let x₀ : ℝ := 0
  let y₀ : ℝ := f x₀
  let slope : ℝ := f' x₀
  Real.arctan slope = π / 4 := by sorry

end NUMINAMATH_CALUDE_tangent_angle_at_origin_l3958_395850


namespace NUMINAMATH_CALUDE_distance_between_given_planes_l3958_395882

-- Define the planes
def plane1 (x y z : ℝ) : Prop := 3*x - 2*y + 4*z = 12
def plane2 (x y z : ℝ) : Prop := 6*x - 4*y + 8*z = 5

-- Define the distance function between two planes
noncomputable def distance_between_planes : ℝ := sorry

-- Theorem statement
theorem distance_between_given_planes :
  distance_between_planes = 7 * Real.sqrt 29 / 29 := by sorry

end NUMINAMATH_CALUDE_distance_between_given_planes_l3958_395882


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3958_395819

theorem quadratic_inequality_solution (x : ℝ) : 
  -3 * x^2 + 9 * x + 6 < 0 ↔ -2/3 < x ∧ x < 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3958_395819


namespace NUMINAMATH_CALUDE_total_cards_l3958_395806

theorem total_cards (mao_cards : ℕ) (li_cards : ℕ) 
  (h1 : mao_cards = 23) (h2 : li_cards = 20) : 
  mao_cards + li_cards = 43 := by
  sorry

end NUMINAMATH_CALUDE_total_cards_l3958_395806


namespace NUMINAMATH_CALUDE_electric_bicycle_sales_l3958_395804

theorem electric_bicycle_sales (sales_A_Q1 : ℝ) (sales_BC_Q1 : ℝ) (a : ℝ) :
  sales_A_Q1 = 0.56 ∧
  sales_BC_Q1 = 1 - sales_A_Q1 ∧
  sales_A_Q1 * 1.23 + sales_BC_Q1 * (1 - a / 100) = 1.12 →
  a = 2 :=
by sorry

end NUMINAMATH_CALUDE_electric_bicycle_sales_l3958_395804


namespace NUMINAMATH_CALUDE_data_average_is_three_l3958_395894

def data : List ℝ := [2, 3, 2, 2, 3, 6]

def is_mode (x : ℝ) (l : List ℝ) : Prop :=
  ∀ y ∈ l, (l.count x ≥ l.count y)

theorem data_average_is_three :
  is_mode 2 data →
  (data.sum / data.length : ℝ) = 3 := by
sorry

end NUMINAMATH_CALUDE_data_average_is_three_l3958_395894


namespace NUMINAMATH_CALUDE_range_of_a_l3958_395849

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*a*x + 4 > 0) ∧
  (∀ x y : ℝ, x < y → (3 - 2*a)^x < (3 - 2*a)^y) →
  a > -2 ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3958_395849


namespace NUMINAMATH_CALUDE_new_employee_age_l3958_395885

theorem new_employee_age 
  (initial_employees : ℕ) 
  (initial_avg_age : ℝ) 
  (final_employees : ℕ) 
  (final_avg_age : ℝ) : 
  initial_employees = 13 → 
  initial_avg_age = 35 → 
  final_employees = initial_employees + 1 → 
  final_avg_age = 34 → 
  (final_employees * final_avg_age - initial_employees * initial_avg_age : ℝ) = 21 := by
  sorry

end NUMINAMATH_CALUDE_new_employee_age_l3958_395885


namespace NUMINAMATH_CALUDE_cookie_boxes_problem_l3958_395846

theorem cookie_boxes_problem (n : ℕ) : 
  n - 7 ≥ 1 → 
  n - 2 ≥ 1 → 
  (n - 7) + (n - 2) < n → 
  n = 8 :=
by sorry

end NUMINAMATH_CALUDE_cookie_boxes_problem_l3958_395846


namespace NUMINAMATH_CALUDE_problem_1_l3958_395805

theorem problem_1 (a b : ℝ) (h : a ≠ 0) (h' : b ≠ 0) :
  (-3*b/(2*a)) * (6*a/b^3) = -9/b^2 :=
sorry

end NUMINAMATH_CALUDE_problem_1_l3958_395805


namespace NUMINAMATH_CALUDE_modulus_of_z_l3958_395817

theorem modulus_of_z (z : ℂ) (h : z / (1 + 2 * I) = 1) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3958_395817


namespace NUMINAMATH_CALUDE_same_last_digit_l3958_395814

theorem same_last_digit (a b : ℕ) : 
  (2 * a + b) % 10 = (2 * b + a) % 10 → a % 10 = b % 10 := by
  sorry

end NUMINAMATH_CALUDE_same_last_digit_l3958_395814
