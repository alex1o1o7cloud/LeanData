import Mathlib

namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2589_258905

def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {x | x^2 - 3*x - 4 < 0}

theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {x : ℝ | 1 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2589_258905


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l2589_258963

theorem largest_divisor_of_expression (y : ℤ) (h : Even y) :
  (∃ (k : ℤ), (8*y+4)*(8*y+8)*(4*y+6)*(4*y+2) = 96 * k) ∧
  (∀ (n : ℤ), n > 96 → ¬(∀ (y : ℤ), Even y → ∃ (k : ℤ), (8*y+4)*(8*y+8)*(4*y+6)*(4*y+2) = n * k)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l2589_258963


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l2589_258961

theorem sum_of_reciprocals_of_roots (x : ℝ) : 
  x^2 - 17*x + 8 = 0 → 
  ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ x^2 - 17*x + 8 = (x - r₁) * (x - r₂) ∧ 
  (1 / r₁ + 1 / r₂ : ℝ) = 17 / 8 :=
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l2589_258961


namespace NUMINAMATH_CALUDE_xyz_mod_9_l2589_258982

theorem xyz_mod_9 (x y z : ℕ) : 
  x < 9 → y < 9 → z < 9 →
  (x + 3*y + 2*z) % 9 = 0 →
  (3*x + 2*y + z) % 9 = 5 →
  (2*x + y + 3*z) % 9 = 5 →
  (x*y*z) % 9 = 0 := by
sorry

end NUMINAMATH_CALUDE_xyz_mod_9_l2589_258982


namespace NUMINAMATH_CALUDE_friction_force_on_rotated_board_l2589_258915

/-- The friction force on a block on a rotated rectangular board -/
theorem friction_force_on_rotated_board 
  (m g : ℝ) 
  (α β : ℝ) 
  (h_α_acute : 0 < α ∧ α < π / 2) 
  (h_β_acute : 0 < β ∧ β < π / 2) :
  ∃ F : ℝ, F = m * g * Real.sqrt (1 - Real.cos α ^ 2 * Real.cos β ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_friction_force_on_rotated_board_l2589_258915


namespace NUMINAMATH_CALUDE_average_temperature_l2589_258937

def temperature_day1 : ℤ := -14
def temperature_day2 : ℤ := -8
def temperature_day3 : ℤ := 1
def num_days : ℕ := 3

theorem average_temperature :
  (temperature_day1 + temperature_day2 + temperature_day3) / num_days = -7 :=
by sorry

end NUMINAMATH_CALUDE_average_temperature_l2589_258937


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2018_l2589_258996

def last_four_digits (n : ℕ) : ℕ := n % 10000

def cycle : List ℕ := [3125, 5625, 8125, 0625]

theorem last_four_digits_of_5_pow_2018 :
  last_four_digits (5^2018) = 5625 := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2018_l2589_258996


namespace NUMINAMATH_CALUDE_heap_sheet_count_l2589_258938

/-- The number of bundles of colored paper -/
def colored_bundles : ℕ := 3

/-- The number of bunches of white paper -/
def white_bunches : ℕ := 2

/-- The number of heaps of scrap paper -/
def scrap_heaps : ℕ := 5

/-- The number of sheets in a bunch -/
def sheets_per_bunch : ℕ := 4

/-- The number of sheets in a bundle -/
def sheets_per_bundle : ℕ := 2

/-- The total number of sheets removed -/
def total_sheets_removed : ℕ := 114

/-- The number of sheets in a heap -/
def sheets_per_heap : ℕ := 20

theorem heap_sheet_count :
  sheets_per_heap = 
    (total_sheets_removed - 
      (colored_bundles * sheets_per_bundle + 
       white_bunches * sheets_per_bunch)) / scrap_heaps :=
by sorry

end NUMINAMATH_CALUDE_heap_sheet_count_l2589_258938


namespace NUMINAMATH_CALUDE_hexadecagon_triangles_l2589_258975

/-- The number of vertices in a regular hexadecagon -/
def n : ℕ := 16

/-- Represents that no three vertices are collinear in a regular hexadecagon -/
axiom no_collinear_vertices : True

/-- The number of triangles formed by choosing 3 vertices from n vertices -/
def num_triangles : ℕ := Nat.choose n 3

theorem hexadecagon_triangles : num_triangles = 560 := by
  sorry

end NUMINAMATH_CALUDE_hexadecagon_triangles_l2589_258975


namespace NUMINAMATH_CALUDE_intersection_sum_l2589_258944

/-- Two functions f and g that intersect at given points -/
def f (a b x : ℝ) : ℝ := -2 * abs (x - a) + b
def g (c d x : ℝ) : ℝ := 2 * abs (x - c) + d

/-- Theorem stating that for functions f and g intersecting at (1, 7) and (11, -1), a + c = 12 -/
theorem intersection_sum (a b c d : ℝ) 
  (h1 : f a b 1 = g c d 1 ∧ f a b 1 = 7)
  (h2 : f a b 11 = g c d 11 ∧ f a b 11 = -1) :
  a + c = 12 := by
  sorry


end NUMINAMATH_CALUDE_intersection_sum_l2589_258944


namespace NUMINAMATH_CALUDE_quadratic_set_theorem_l2589_258949

theorem quadratic_set_theorem (a : ℝ) : 
  ({x : ℝ | x^2 + a*x = 0} = {0, 1}) → a = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_set_theorem_l2589_258949


namespace NUMINAMATH_CALUDE_circle_equation_l2589_258945

/-- Given a circle with center (2, -3) intercepted by the line 2x + 3y - 8 = 0
    with a chord length of 4√3, prove that its standard equation is (x-2)² + (y+3)² = 25 -/
theorem circle_equation (x y : ℝ) : 
  let center : ℝ × ℝ := (2, -3)
  let line (x y : ℝ) : ℝ := 2*x + 3*y - 8
  let chord_length : ℝ := 4 * Real.sqrt 3
  ∃ (r : ℝ), r > 0 ∧ 
    (∀ (p : ℝ × ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2 ↔ 
      ((p.1 - 2)^2 + (p.2 + 3)^2 = 25 ∧ 
       ∃ (q : ℝ × ℝ), line q.1 q.2 = 0 ∧ 
         (q.1 - p.1)^2 + (q.2 - p.2)^2 ≤ chord_length^2)) := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l2589_258945


namespace NUMINAMATH_CALUDE_trapezoid_long_side_is_correct_l2589_258956

/-- A rectangle with given dimensions divided into three equal-area shapes -/
structure DividedRectangle where
  length : ℝ
  width : ℝ
  trapezoid_long_side : ℝ
  is_valid : 
    length = 3 ∧ 
    width = 1 ∧
    0 < trapezoid_long_side ∧ 
    trapezoid_long_side < length

/-- The area of each shape is one-third of the rectangle's area -/
def equal_area_condition (r : DividedRectangle) : Prop :=
  let rectangle_area := r.length * r.width
  let trapezoid_area := (r.trapezoid_long_side + r.length / 2) * r.width / 2
  trapezoid_area = rectangle_area / 3

/-- The main theorem: the longer side of the trapezoid is 1.25 -/
theorem trapezoid_long_side_is_correct (r : DividedRectangle) 
  (h : equal_area_condition r) : r.trapezoid_long_side = 1.25 := by
  sorry

#check trapezoid_long_side_is_correct

end NUMINAMATH_CALUDE_trapezoid_long_side_is_correct_l2589_258956


namespace NUMINAMATH_CALUDE_gas_pressure_change_l2589_258985

/-- Represents the state of a gas with pressure and volume -/
structure GasState where
  pressure : ℝ
  volume : ℝ

/-- The constant of proportionality for the gas -/
def gasConstant (state : GasState) : ℝ := state.pressure * state.volume

theorem gas_pressure_change 
  (initial : GasState) 
  (final : GasState) 
  (h1 : initial.pressure = 8) 
  (h2 : initial.volume = 3.5)
  (h3 : final.volume = 10.5)
  (h4 : gasConstant initial = gasConstant final) : 
  final.pressure = 8/3 := by
  sorry

#check gas_pressure_change

end NUMINAMATH_CALUDE_gas_pressure_change_l2589_258985


namespace NUMINAMATH_CALUDE_home_electronics_budget_allocation_l2589_258993

theorem home_electronics_budget_allocation 
  (total_budget : ℝ)
  (microphotonics : ℝ)
  (food_additives : ℝ)
  (genetically_modified_microorganisms : ℝ)
  (industrial_lubricants : ℝ)
  (basic_astrophysics_degrees : ℝ)
  (h1 : total_budget = 100)
  (h2 : microphotonics = 14)
  (h3 : food_additives = 20)
  (h4 : genetically_modified_microorganisms = 29)
  (h5 : industrial_lubricants = 8)
  (h6 : basic_astrophysics_degrees = 18)
  (h7 : (basic_astrophysics_degrees / 360) * 100 + microphotonics + food_additives + genetically_modified_microorganisms + industrial_lubricants + home_electronics = total_budget) :
  home_electronics = 24 := by
  sorry

end NUMINAMATH_CALUDE_home_electronics_budget_allocation_l2589_258993


namespace NUMINAMATH_CALUDE_consecutive_primes_as_greatest_divisors_l2589_258935

theorem consecutive_primes_as_greatest_divisors (p q : ℕ) 
  (hp : Prime p) (hq : Prime q) (hpq : p < q) (hqp : q < 2 * p) :
  ∃ n : ℕ, 
    (∃ k : ℕ+, n = k * p ∧ ∀ m : ℕ, m > p → m.Prime → ¬(m ∣ n)) ∧
    (∃ l : ℕ+, n + 1 = l * q ∧ ∀ m : ℕ, m > q → m.Prime → ¬(m ∣ (n + 1))) ∨
    (∃ k : ℕ+, n = k * q ∧ ∀ m : ℕ, m > q → m.Prime → ¬(m ∣ n)) ∧
    (∃ l : ℕ+, n + 1 = l * p ∧ ∀ m : ℕ, m > p → m.Prime → ¬(m ∣ (n + 1))) :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_primes_as_greatest_divisors_l2589_258935


namespace NUMINAMATH_CALUDE_distribute_5_balls_4_boxes_l2589_258917

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_5_balls_4_boxes : distribute_balls 5 4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_distribute_5_balls_4_boxes_l2589_258917


namespace NUMINAMATH_CALUDE_range_of_f_l2589_258965

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := arctan x + arctan ((x - 2) / (x + 2))

-- Theorem statement
theorem range_of_f :
  ∃ (S : Set ℝ), S = Set.range f ∧ S = {-π/4, arctan 2} :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l2589_258965


namespace NUMINAMATH_CALUDE_complement_of_union_l2589_258907

def U : Set ℕ := {x | x > 0 ∧ x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

theorem complement_of_union : 
  (A ∪ B)ᶜ = {2, 4} :=
sorry

end NUMINAMATH_CALUDE_complement_of_union_l2589_258907


namespace NUMINAMATH_CALUDE_max_y_coordinate_polar_curve_l2589_258994

theorem max_y_coordinate_polar_curve (θ : Real) :
  let r := λ θ : Real => Real.cos (2 * θ)
  let x := λ θ : Real => (r θ) * Real.cos θ
  let y := λ θ : Real => (r θ) * Real.sin θ
  (∀ θ', |y θ'| ≤ |y θ|) → y θ = Real.sqrt (30 * Real.sqrt 6) / 9 :=
by sorry

end NUMINAMATH_CALUDE_max_y_coordinate_polar_curve_l2589_258994


namespace NUMINAMATH_CALUDE_rectangle_semicircle_ratio_l2589_258954

theorem rectangle_semicircle_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  a * b = π * b^2 → a / b = π := by
  sorry

end NUMINAMATH_CALUDE_rectangle_semicircle_ratio_l2589_258954


namespace NUMINAMATH_CALUDE_ten_thousand_squared_l2589_258952

theorem ten_thousand_squared : (10000 : ℕ) * 10000 = 100000000 := by
  sorry

end NUMINAMATH_CALUDE_ten_thousand_squared_l2589_258952


namespace NUMINAMATH_CALUDE_favorite_fruit_apples_l2589_258909

theorem favorite_fruit_apples (total students_oranges students_pears students_strawberries : ℕ) 
  (h1 : total = 450)
  (h2 : students_oranges = 70)
  (h3 : students_pears = 120)
  (h4 : students_strawberries = 113) :
  total - (students_oranges + students_pears + students_strawberries) = 147 := by
  sorry

end NUMINAMATH_CALUDE_favorite_fruit_apples_l2589_258909


namespace NUMINAMATH_CALUDE_givenPoint_in_first_quadrant_l2589_258922

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def isInFirstQuadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- The given point -/
def givenPoint : Point2D :=
  { x := 6, y := 2 }

/-- Theorem stating that the given point is in the first quadrant -/
theorem givenPoint_in_first_quadrant :
  isInFirstQuadrant givenPoint :=
by
  sorry

end NUMINAMATH_CALUDE_givenPoint_in_first_quadrant_l2589_258922


namespace NUMINAMATH_CALUDE_special_square_smallest_area_l2589_258995

/-- A square with specific properties -/
structure SpecialSquare where
  /-- Two vertices lie on the line y = 2x + 3 -/
  vertices_on_line : ℝ → ℝ → Prop
  /-- Two vertices lie on the parabola y = -x^2 + 4x + 5 -/
  vertices_on_parabola : ℝ → ℝ → Prop
  /-- One vertex lies on the origin (0, 0) -/
  vertex_on_origin : Prop

/-- The smallest possible area of a SpecialSquare -/
def smallest_area (s : SpecialSquare) : ℝ := 580

/-- Theorem stating the smallest possible area of a SpecialSquare -/
theorem special_square_smallest_area (s : SpecialSquare) :
  smallest_area s = 580 := by sorry

end NUMINAMATH_CALUDE_special_square_smallest_area_l2589_258995


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2589_258960

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, 1 < x ∧ x < 2 → (x - 2)^2 < 1) ∧
  (∃ x, (x - 2)^2 < 1 ∧ ¬(1 < x ∧ x < 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2589_258960


namespace NUMINAMATH_CALUDE_catering_cost_comparison_l2589_258992

def cost_caterer1 (x : ℕ) : ℚ := 150 + 18 * x
def cost_caterer2 (x : ℕ) : ℚ := 250 + 15 * x

theorem catering_cost_comparison :
  (∀ x : ℕ, x < 34 → cost_caterer1 x ≤ cost_caterer2 x) ∧
  (∀ x : ℕ, x ≥ 34 → cost_caterer1 x > cost_caterer2 x) :=
by sorry

end NUMINAMATH_CALUDE_catering_cost_comparison_l2589_258992


namespace NUMINAMATH_CALUDE_square_minus_equal_two_implies_sum_equal_one_l2589_258924

theorem square_minus_equal_two_implies_sum_equal_one (m : ℝ) 
  (h : m^2 - m = 2) : 
  (m - 1)^2 + (m + 2) * (m - 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_equal_two_implies_sum_equal_one_l2589_258924


namespace NUMINAMATH_CALUDE_missing_number_l2589_258920

theorem missing_number (n : ℕ) : 
  (∀ k : ℕ, k < n → k * (k + 1) / 2 ≤ 575) ∧ 
  (n * (n + 1) / 2 > 575) → 
  n * (n + 1) / 2 - 575 = 20 := by
sorry

end NUMINAMATH_CALUDE_missing_number_l2589_258920


namespace NUMINAMATH_CALUDE_balloon_distribution_l2589_258977

theorem balloon_distribution (yellow_balloons : ℕ) (blue_balloons : ℕ) (black_extra : ℕ) (schools : ℕ) :
  yellow_balloons = 3414 →
  blue_balloons = 5238 →
  black_extra = 1762 →
  schools = 15 →
  ((yellow_balloons + blue_balloons + (yellow_balloons + black_extra)) / schools : ℕ) = 921 :=
by sorry

end NUMINAMATH_CALUDE_balloon_distribution_l2589_258977


namespace NUMINAMATH_CALUDE_slightly_used_crayons_l2589_258902

theorem slightly_used_crayons (total : ℕ) (new : ℕ) (broken : ℕ) (slightly_used : ℕ) : 
  total = 120 →
  new = total / 3 →
  broken = total / 5 →
  slightly_used = total - new - broken →
  slightly_used = 56 := by
sorry

end NUMINAMATH_CALUDE_slightly_used_crayons_l2589_258902


namespace NUMINAMATH_CALUDE_inequality_proof_l2589_258983

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^3 + b^3)^(1/3) < (a^2 + b^2)^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2589_258983


namespace NUMINAMATH_CALUDE_inequality_proof_l2589_258936

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a^2 + b^2 = 1/2) :
  1/(1-a) + 1/(1-b) ≥ 4 ∧ (1/(1-a) + 1/(1-b) = 4 ↔ a = 1/2 ∧ b = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2589_258936


namespace NUMINAMATH_CALUDE_shifted_quadratic_sum_l2589_258941

/-- The original quadratic function -/
def f (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 5

/-- The shifted quadratic function -/
def g (x : ℝ) : ℝ := f (x - 3)

/-- The coefficients of the shifted function -/
def a : ℝ := 3
def b : ℝ := -20
def c : ℝ := 38

theorem shifted_quadratic_sum :
  g x = a * x^2 + b * x + c ∧ a + b + c = 21 := by sorry

end NUMINAMATH_CALUDE_shifted_quadratic_sum_l2589_258941


namespace NUMINAMATH_CALUDE_carlotta_time_theorem_l2589_258914

def singing_time : ℕ := 6

def practice_time (n : ℕ) : ℕ := 2 * n

def tantrum_time (n : ℕ) : ℕ := 3 * n + 1

def total_time (singing : ℕ) : ℕ :=
  singing +
  singing * practice_time singing +
  singing * tantrum_time singing

theorem carlotta_time_theorem :
  total_time singing_time = 192 := by sorry

end NUMINAMATH_CALUDE_carlotta_time_theorem_l2589_258914


namespace NUMINAMATH_CALUDE_complement_A_intersection_nonempty_union_equals_B_l2589_258948

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- Theorem for the complement of A
theorem complement_A : (Set.univ \ A) = {x : ℝ | x ≤ -1 ∨ x > 2} := by sorry

-- Theorem for the range of a when A ∩ B ≠ ∅
theorem intersection_nonempty (a : ℝ) : (A ∩ B a).Nonempty → a > -1 := by sorry

-- Theorem for the range of a when A ∪ B = B
theorem union_equals_B (a : ℝ) : A ∪ B a = B a → a > 2 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersection_nonempty_union_equals_B_l2589_258948


namespace NUMINAMATH_CALUDE_eq2_eq3_same_graph_eq1_different_graph_l2589_258955

-- Define the three equations
def eq1 (x y : ℝ) : Prop := y = x + 3
def eq2 (x y : ℝ) : Prop := y = (x^2 - 1) / (x - 1)
def eq3 (x y : ℝ) : Prop := (x - 1) * y = x^2 - 1

-- Define the concept of having the same graph
def same_graph (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, x ≠ 1 → (f x y ↔ g x y)

-- Theorem stating that eq2 and eq3 have the same graph
theorem eq2_eq3_same_graph : same_graph eq2 eq3 := by sorry

-- Theorem stating that eq1 has a different graph from eq2 and eq3
theorem eq1_different_graph :
  ¬(same_graph eq1 eq2) ∧ ¬(same_graph eq1 eq3) := by sorry

end NUMINAMATH_CALUDE_eq2_eq3_same_graph_eq1_different_graph_l2589_258955


namespace NUMINAMATH_CALUDE_unique_solution_iff_p_eq_neg_four_thirds_l2589_258951

/-- The equation has exactly one solution if and only if p = -4/3 -/
theorem unique_solution_iff_p_eq_neg_four_thirds :
  (∃! x : ℝ, (2 * x + 3) / (p * x - 2) = x) ↔ p = -4/3 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_iff_p_eq_neg_four_thirds_l2589_258951


namespace NUMINAMATH_CALUDE_least_addend_for_divisibility_least_addend_for_1156_and_97_l2589_258912

theorem least_addend_for_divisibility (n m : ℕ) (h : n > 0) : 
  ∃ (x : ℕ), (n + x) % m = 0 ∧ ∀ (y : ℕ), y < x → (n + y) % m ≠ 0 :=
sorry

theorem least_addend_for_1156_and_97 : 
  ∃ (x : ℕ), (1156 + x) % 97 = 0 ∧ ∀ (y : ℕ), y < x → (1156 + y) % 97 ≠ 0 ∧ x = 8 :=
sorry

end NUMINAMATH_CALUDE_least_addend_for_divisibility_least_addend_for_1156_and_97_l2589_258912


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2589_258940

def P : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def Q : Set (ℝ × ℝ) := {q | q.1 - q.2 = 4}

theorem intersection_of_P_and_Q : P ∩ Q = {(3, -1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2589_258940


namespace NUMINAMATH_CALUDE_calculate_x_l2589_258950

theorem calculate_x : ∀ (w y z x : ℕ),
  w = 90 →
  z = w + 25 →
  y = z + 15 →
  x = y + 7 →
  x = 137 := by
  sorry

end NUMINAMATH_CALUDE_calculate_x_l2589_258950


namespace NUMINAMATH_CALUDE_line_polar_equation_l2589_258939

-- Define the line in Cartesian coordinates
def line (x y : ℝ) : Prop := (Real.sqrt 3 / 3) * x - y = 0

-- Define the polar coordinates
def polar_coords (ρ θ : ℝ) (x y : ℝ) : Prop :=
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ ρ ≥ 0

-- State the theorem
theorem line_polar_equation :
  ∀ ρ θ x y : ℝ,
  polar_coords ρ θ x y →
  line x y →
  (θ = π / 6 ∨ θ = 7 * π / 6) :=
sorry

end NUMINAMATH_CALUDE_line_polar_equation_l2589_258939


namespace NUMINAMATH_CALUDE_min_third_side_right_triangle_l2589_258932

theorem min_third_side_right_triangle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  (a = 7 ∨ b = 7 ∨ c = 7) → 
  (a = 24 ∨ b = 24 ∨ c = 24) → 
  a^2 + b^2 = c^2 → 
  min a (min b c) ≥ Real.sqrt 527 :=
by sorry

end NUMINAMATH_CALUDE_min_third_side_right_triangle_l2589_258932


namespace NUMINAMATH_CALUDE_mahogany_count_l2589_258980

/-- The number of initially planted Mahogany trees -/
def initial_mahogany : ℕ := sorry

/-- The number of initially planted Narra trees -/
def initial_narra : ℕ := 30

/-- The total number of trees that fell -/
def total_fallen : ℕ := 5

/-- The number of Mahogany trees that fell -/
def mahogany_fallen : ℕ := sorry

/-- The number of Narra trees that fell -/
def narra_fallen : ℕ := sorry

/-- The number of new Mahogany trees planted after the typhoon -/
def new_mahogany : ℕ := sorry

/-- The number of new Narra trees planted after the typhoon -/
def new_narra : ℕ := sorry

/-- The total number of trees after replanting -/
def total_trees : ℕ := 88

theorem mahogany_count : initial_mahogany = 50 :=
  by sorry

end NUMINAMATH_CALUDE_mahogany_count_l2589_258980


namespace NUMINAMATH_CALUDE_shortest_side_is_15_l2589_258974

/-- A right triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- The length of the first segment of the hypotenuse -/
  segment1 : ℝ
  /-- The length of the second segment of the hypotenuse -/
  segment2 : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- Assumption that segment1 is positive -/
  segment1_pos : segment1 > 0
  /-- Assumption that segment2 is positive -/
  segment2_pos : segment2 > 0
  /-- Assumption that radius is positive -/
  radius_pos : radius > 0

/-- The length of the shortest side in a right triangle with an inscribed circle -/
def shortest_side (t : RightTriangleWithInscribedCircle) : ℝ :=
  sorry

/-- Theorem stating that the shortest side is 15 units under given conditions -/
theorem shortest_side_is_15 (t : RightTriangleWithInscribedCircle) 
  (h1 : t.segment1 = 7) 
  (h2 : t.segment2 = 9) 
  (h3 : t.radius = 5) : 
  shortest_side t = 15 := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_is_15_l2589_258974


namespace NUMINAMATH_CALUDE_triangle_problem_l2589_258966

theorem triangle_problem (AB : ℝ) (sinA sinC : ℝ) :
  AB = 30 →
  sinA = 4/5 →
  sinC = 1/4 →
  ∃ (DC : ℝ), DC = 24 * Real.sqrt 15 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2589_258966


namespace NUMINAMATH_CALUDE_factor_of_valid_Z_l2589_258988

def is_valid_Z (n : ℕ) : Prop :=
  10000000 ≤ n ∧ n < 100000000 ∧
  ∃ (a b c d : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    n = 10000000 * a + 1000000 * b + 100000 * c + 10000 * d +
        1000 * a + 100 * b + 10 * c + d

theorem factor_of_valid_Z (Z : ℕ) (h : is_valid_Z Z) : 
  10001 ∣ Z :=
sorry

end NUMINAMATH_CALUDE_factor_of_valid_Z_l2589_258988


namespace NUMINAMATH_CALUDE_point_coordinates_l2589_258934

/-- A point in the Cartesian coordinate system -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the third quadrant -/
def in_third_quadrant (p : CartesianPoint) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The distance from a point to the x-axis -/
def distance_to_x_axis (p : CartesianPoint) : ℝ :=
  |p.y|

/-- The distance from a point to the y-axis -/
def distance_to_y_axis (p : CartesianPoint) : ℝ :=
  |p.x|

theorem point_coordinates
  (p : CartesianPoint)
  (h1 : in_third_quadrant p)
  (h2 : distance_to_x_axis p = 5)
  (h3 : distance_to_y_axis p = 6) :
  p.x = -6 ∧ p.y = -5 :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_l2589_258934


namespace NUMINAMATH_CALUDE_bakery_pastries_and_bagels_l2589_258976

/-- Proves that the total number of pastries and bagels is 474 given the bakery conditions -/
theorem bakery_pastries_and_bagels :
  let total_items : ℕ := 720
  let bread_rolls : ℕ := 240
  let croissants : ℕ := 75
  let muffins : ℕ := 145
  let cinnamon_rolls : ℕ := 110
  let pastries : ℕ := croissants + muffins + cinnamon_rolls
  let bagels : ℕ := total_items - (bread_rolls + pastries)
  let pastries_per_bread_roll : ℚ := 2.5
  let bagels_per_5_bread_rolls : ℕ := 3

  (pastries : ℚ) / bread_rolls = pastries_per_bread_roll ∧
  (bagels : ℚ) / bread_rolls = (bagels_per_5_bread_rolls : ℚ) / 5 →
  pastries + bagels = 474 := by
sorry

end NUMINAMATH_CALUDE_bakery_pastries_and_bagels_l2589_258976


namespace NUMINAMATH_CALUDE_largest_unattainable_integer_l2589_258925

/-- Given positive integers a, b, c with no pairwise common divisor greater than 1,
    2abc-ab-bc-ca is the largest integer that cannot be expressed as xbc+yca+zab
    for non-negative integers x, y, z -/
theorem largest_unattainable_integer (a b c : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : Nat.gcd a b = 1) (hbc : Nat.gcd b c = 1) (hac : Nat.gcd a c = 1) :
  (∀ x y z : ℕ, x * b * c + y * c * a + z * a * b ≠ 2 * a * b * c - a * b - b * c - c * a) ∧
  (∀ n : ℕ, n > 2 * a * b * c - a * b - b * c - c * a →
    ∃ x y z : ℕ, x * b * c + y * c * a + z * a * b = n) :=
by sorry

end NUMINAMATH_CALUDE_largest_unattainable_integer_l2589_258925


namespace NUMINAMATH_CALUDE_composite_power_sum_l2589_258967

theorem composite_power_sum (n : ℕ) (h : n > 1) :
  ∃ (k : ℕ), k > 1 ∧ k ∣ ((2^(2^(n+1)) + 2^(2^n) + 1) / 3) := by
  sorry

end NUMINAMATH_CALUDE_composite_power_sum_l2589_258967


namespace NUMINAMATH_CALUDE_least_divisible_n_divisors_l2589_258942

theorem least_divisible_n_divisors (n : ℕ) : 
  (∀ k < n, ¬(3^3 * 5^5 * 7^7 ∣ (149^k - 2^k))) →
  (3^3 * 5^5 * 7^7 ∣ (149^n - 2^n)) →
  (∀ m : ℕ, m > n → ¬(3^3 * 5^5 * 7^7 ∣ (149^m - 2^m))) →
  Nat.card {d : ℕ | d ∣ n} = 270 :=
sorry

end NUMINAMATH_CALUDE_least_divisible_n_divisors_l2589_258942


namespace NUMINAMATH_CALUDE_intersection_point_property_l2589_258978

theorem intersection_point_property (n : ℕ) (x₀ y₀ : ℝ) (hn : n ≥ 2) 
  (h1 : y₀^2 = n * x₀ - 1) (h2 : y₀ = x₀) :
  ∀ m : ℕ, m > 0 → ∃ k : ℕ, k ≥ 2 ∧ (x₀^m)^2 = k * (x₀^m) - 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_property_l2589_258978


namespace NUMINAMATH_CALUDE_calculation_result_l2589_258929

theorem calculation_result : (481 + 426)^2 - 4 * 481 * 426 = 3505 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l2589_258929


namespace NUMINAMATH_CALUDE_translation_result_l2589_258933

-- Define the points A, B, and C
def A : ℝ × ℝ := (-2, 5)
def B : ℝ × ℝ := (-3, 0)
def C : ℝ × ℝ := (3, 8)

-- Define the translation vector
def translation_vector : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)

-- Define point D as the result of translating B
def D : ℝ × ℝ := (B.1 + translation_vector.1, B.2 + translation_vector.2)

-- Theorem statement
theorem translation_result : D = (2, 3) := by
  sorry

end NUMINAMATH_CALUDE_translation_result_l2589_258933


namespace NUMINAMATH_CALUDE_cindy_travel_time_l2589_258913

/-- Calculates the total time for Cindy to travel 1 mile -/
theorem cindy_travel_time (run_speed walk_speed run_distance walk_distance : ℝ) :
  run_speed = 3 →
  walk_speed = 1 →
  run_distance = 0.5 →
  walk_distance = 0.5 →
  run_distance + walk_distance = 1 →
  (run_distance / run_speed + walk_distance / walk_speed) * 60 = 40 :=
by sorry

end NUMINAMATH_CALUDE_cindy_travel_time_l2589_258913


namespace NUMINAMATH_CALUDE_thirteenth_term_is_30_l2589_258916

/-- An arithmetic sequence with specified terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m
  a5_eq_6 : a 5 = 6
  a8_eq_15 : a 8 = 15

/-- The 13th term of the arithmetic sequence is 30 -/
theorem thirteenth_term_is_30 (seq : ArithmeticSequence) : seq.a 13 = 30 := by
  sorry

end NUMINAMATH_CALUDE_thirteenth_term_is_30_l2589_258916


namespace NUMINAMATH_CALUDE_binomial_17_4_l2589_258981

theorem binomial_17_4 : Nat.choose 17 4 = 2380 := by
  sorry

end NUMINAMATH_CALUDE_binomial_17_4_l2589_258981


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2589_258987

theorem sum_of_three_numbers (a b c : ℝ) 
  (sum_of_squares : a^2 + b^2 + c^2 = 267)
  (sum_of_products : a*b + b*c + c*a = 131) :
  a + b + c = 23 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2589_258987


namespace NUMINAMATH_CALUDE_abs_neg_three_eq_three_l2589_258970

theorem abs_neg_three_eq_three : |(-3 : ℝ)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_eq_three_l2589_258970


namespace NUMINAMATH_CALUDE_complex_reciprocal_sum_l2589_258964

theorem complex_reciprocal_sum (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 5) :
  Complex.abs (1 / z + 1 / w) = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_reciprocal_sum_l2589_258964


namespace NUMINAMATH_CALUDE_most_reasonable_estimate_l2589_258910

/-- Represents the total number of female students in the first year -/
def total_female : ℕ := 504

/-- Represents the total number of male students in the first year -/
def total_male : ℕ := 596

/-- Represents the total number of students in the first year -/
def total_students : ℕ := total_female + total_male

/-- Represents the average weight of sampled female students -/
def avg_weight_female : ℝ := 49

/-- Represents the average weight of sampled male students -/
def avg_weight_male : ℝ := 57

/-- Theorem stating that the most reasonable estimate for the average weight
    of all first-year students is (504/1100) * 49 + (596/1100) * 57 -/
theorem most_reasonable_estimate :
  (total_female : ℝ) / total_students * avg_weight_female +
  (total_male : ℝ) / total_students * avg_weight_male =
  (504 : ℝ) / 1100 * 49 + (596 : ℝ) / 1100 * 57 := by
  sorry

end NUMINAMATH_CALUDE_most_reasonable_estimate_l2589_258910


namespace NUMINAMATH_CALUDE_range_of_a_l2589_258953

/-- The function f defined on positive real numbers. -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.log x)^2 - Real.log x

/-- The function h defined on positive real numbers. -/
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := (f a x + 1 - a) * (Real.log x)⁻¹

/-- The theorem stating the range of a given the conditions. -/
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (Real.exp (-3)) (Real.exp (-1)) →
                x₂ ∈ Set.Icc (Real.exp (-3)) (Real.exp (-1)) →
                |h a x₁ - h a x₂| ≤ a + 1/3) →
  a ∈ Set.Icc (1/11) (3/5) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2589_258953


namespace NUMINAMATH_CALUDE_bunny_burrow_exits_l2589_258923

/-- The number of times a bunny comes out of its burrow per minute -/
def bunny_rate : ℕ := 3

/-- The number of bunnies -/
def num_bunnies : ℕ := 20

/-- The time period in hours -/
def time_period : ℕ := 10

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

theorem bunny_burrow_exits :
  bunny_rate * minutes_per_hour * time_period * num_bunnies = 36000 := by
  sorry

end NUMINAMATH_CALUDE_bunny_burrow_exits_l2589_258923


namespace NUMINAMATH_CALUDE_arcs_not_exceeding_120_degrees_l2589_258968

/-- Given 21 points on a circle, the number of arcs with these points as endpoints
    that have a measure of no more than 120° is equal to 100. -/
theorem arcs_not_exceeding_120_degrees (n : ℕ) (h : n = 21) : 
  (n.choose 2) - (n - 1) * (n / 2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_arcs_not_exceeding_120_degrees_l2589_258968


namespace NUMINAMATH_CALUDE_mia_wins_two_l2589_258946

/-- Represents a player in the chess tournament -/
inductive Player : Type
  | Sarah : Player
  | Ryan : Player
  | Mia : Player

/-- Represents the number of games won by a player -/
def wins : Player → ℕ
  | Player.Sarah => 5
  | Player.Ryan => 2
  | Player.Mia => 2  -- This is what we want to prove

/-- Represents the number of games lost by a player -/
def losses : Player → ℕ
  | Player.Sarah => 1
  | Player.Ryan => 4
  | Player.Mia => 4

/-- The total number of games played in the tournament -/
def total_games : ℕ := 6

theorem mia_wins_two : wins Player.Mia = 2 := by
  sorry

#check mia_wins_two

end NUMINAMATH_CALUDE_mia_wins_two_l2589_258946


namespace NUMINAMATH_CALUDE_sichuan_peppercorn_transport_l2589_258990

/-- Represents the capacity of a truck type -/
structure TruckCapacity where
  a : ℕ
  b : ℕ
  h : a = b + 20

/-- Represents the number of trucks needed for each type -/
structure TruckCount where
  a : ℕ
  b : ℕ

theorem sichuan_peppercorn_transport 
  (cap : TruckCapacity) 
  (h1 : 1000 / cap.a = 800 / cap.b)
  (count : TruckCount)
  (h2 : count.a + count.b = 18)
  (h3 : cap.a * count.a + cap.b * (count.b - 1) + 65 = 1625) :
  cap.a = 100 ∧ cap.b = 80 ∧ count.a = 10 ∧ count.b = 8 := by
  sorry

#check sichuan_peppercorn_transport

end NUMINAMATH_CALUDE_sichuan_peppercorn_transport_l2589_258990


namespace NUMINAMATH_CALUDE_rectangle_area_y_value_l2589_258971

theorem rectangle_area_y_value 
  (y : ℝ) 
  (h1 : y > 0) 
  (h2 : (5 - (-3)) * (y - (-1)) = 48) : 
  y = 5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_y_value_l2589_258971


namespace NUMINAMATH_CALUDE_a_share_is_4800_l2589_258957

/-- Calculates the share of profit for a partner in a business partnership --/
def calculate_share (contribution_a : ℕ) (months_a : ℕ) (contribution_b : ℕ) (months_b : ℕ) (total_profit : ℕ) : ℕ :=
  let money_months_a := contribution_a * months_a
  let money_months_b := contribution_b * months_b
  let total_money_months := money_months_a + money_months_b
  (money_months_a * total_profit) / total_money_months

/-- Theorem stating that A's share of the profit is 4800 given the problem conditions --/
theorem a_share_is_4800 :
  calculate_share 5000 8 6000 5 8400 = 4800 := by
  sorry

end NUMINAMATH_CALUDE_a_share_is_4800_l2589_258957


namespace NUMINAMATH_CALUDE_apple_division_problem_l2589_258973

/-- Calculates the minimal number of pieces needed to evenly divide apples among students -/
def minimalPieces (apples : ℕ) (students : ℕ) : ℕ :=
  let components := apples.gcd students
  let applesPerComponent := apples / components
  let studentsPerComponent := students / components
  components * (applesPerComponent + studentsPerComponent - 1)

/-- Proves that the minimal number of pieces to evenly divide 221 apples among 403 students is 611 -/
theorem apple_division_problem :
  minimalPieces 221 403 = 611 := by
  sorry

#eval minimalPieces 221 403

end NUMINAMATH_CALUDE_apple_division_problem_l2589_258973


namespace NUMINAMATH_CALUDE_cubic_inequality_l2589_258969

theorem cubic_inequality (x y : ℝ) (h : x > y) : ¬(x^3 < y^3 ∨ x^3 = y^3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2589_258969


namespace NUMINAMATH_CALUDE_perimeter_of_modified_square_l2589_258930

/-- The perimeter of a figure formed by cutting an equilateral triangle from a square and
    translating it to the right side. -/
theorem perimeter_of_modified_square (square_perimeter : ℝ) (h : square_perimeter = 40) :
  let side_length := square_perimeter / 4
  let triangle_side_length := side_length
  let new_perimeter := 2 * side_length + 4 * triangle_side_length
  new_perimeter = 60 := by sorry

end NUMINAMATH_CALUDE_perimeter_of_modified_square_l2589_258930


namespace NUMINAMATH_CALUDE_fuel_mixture_problem_l2589_258959

/-- Proves that the amount of fuel A added to the tank is 106 gallons -/
theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_a : ℝ) (ethanol_b : ℝ) (total_ethanol : ℝ) 
  (h1 : tank_capacity = 214)
  (h2 : ethanol_a = 0.12)
  (h3 : ethanol_b = 0.16)
  (h4 : total_ethanol = 30) :
  ∃ (fuel_a : ℝ), fuel_a = 106 ∧ 
    ethanol_a * fuel_a + ethanol_b * (tank_capacity - fuel_a) = total_ethanol :=
by sorry

end NUMINAMATH_CALUDE_fuel_mixture_problem_l2589_258959


namespace NUMINAMATH_CALUDE_max_k_value_l2589_258919

open Real

noncomputable def f (x : ℝ) := exp x - x - 2

theorem max_k_value :
  ∃ (k : ℤ), k = 2 ∧
  (∀ (x : ℝ), x > 0 → (x - ↑k) * (exp x - 1) + x + 1 > 0) ∧
  (∀ (m : ℤ), m > 2 → ∃ (y : ℝ), y > 0 ∧ (y - ↑m) * (exp y - 1) + y + 1 ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l2589_258919


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l2589_258989

theorem log_sum_equals_two :
  2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l2589_258989


namespace NUMINAMATH_CALUDE_value_of_a_l2589_258979

def A (a : ℝ) : Set ℝ := {0, 2, a^2}
def B (a : ℝ) : Set ℝ := {1, a}

theorem value_of_a : ∀ a : ℝ, A a ∪ B a = {0, 1, 2, 4} → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2589_258979


namespace NUMINAMATH_CALUDE_hotel_loss_calculation_l2589_258984

def hotel_loss (expenses : ℝ) (payment_ratio : ℝ) : ℝ :=
  expenses - (payment_ratio * expenses)

theorem hotel_loss_calculation (expenses : ℝ) (payment_ratio : ℝ) 
  (h1 : expenses = 100)
  (h2 : payment_ratio = 3/4) :
  hotel_loss expenses payment_ratio = 25 := by
  sorry

end NUMINAMATH_CALUDE_hotel_loss_calculation_l2589_258984


namespace NUMINAMATH_CALUDE_log_stack_sum_l2589_258908

theorem log_stack_sum (a l n : ℕ) (h1 : a = 5) (h2 : l = 15) (h3 : n = 11) :
  n * (a + l) / 2 = 110 := by
  sorry

end NUMINAMATH_CALUDE_log_stack_sum_l2589_258908


namespace NUMINAMATH_CALUDE_infinite_solutions_exist_l2589_258906

theorem infinite_solutions_exist (a b c d : ℝ) : 
  ((2*a + 16*b) + (3*c - 8*d)) / 2 = 74 →
  4*a + 6*b = 9*c - 12*d →
  ∃ (f : ℝ → ℝ → ℝ), b = f a d ∧ f a d = -a/21 - 2*d/7 :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_exist_l2589_258906


namespace NUMINAMATH_CALUDE_abs_sqrt3_minus_2_l2589_258999

theorem abs_sqrt3_minus_2 : |Real.sqrt 3 - 2| = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_sqrt3_minus_2_l2589_258999


namespace NUMINAMATH_CALUDE_tan_210_degrees_l2589_258998

theorem tan_210_degrees : Real.tan (210 * π / 180) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_210_degrees_l2589_258998


namespace NUMINAMATH_CALUDE_money_constraints_l2589_258904

theorem money_constraints (a b : ℝ) 
  (eq_constraint : 5 * a - b = 60)
  (ineq_constraint : 6 * a + b < 90) :
  a < 13.64 ∧ b < 8.18 := by
sorry

end NUMINAMATH_CALUDE_money_constraints_l2589_258904


namespace NUMINAMATH_CALUDE_power_of_three_mod_five_l2589_258931

theorem power_of_three_mod_five : 3^2023 % 5 = 2 := by sorry

end NUMINAMATH_CALUDE_power_of_three_mod_five_l2589_258931


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2589_258943

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ((x^3 + 4*x^2 + 2*x + 1) * (y^3 + 4*y^2 + 2*y + 1) * (z^3 + 4*z^2 + 2*z + 1)) / (x*y*z) ≥ 1331 :=
by sorry

theorem min_value_achievable :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  ((x^3 + 4*x^2 + 2*x + 1) * (y^3 + 4*y^2 + 2*y + 1) * (z^3 + 4*z^2 + 2*z + 1)) / (x*y*z) = 1331 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2589_258943


namespace NUMINAMATH_CALUDE_train_length_l2589_258927

/-- The length of a train given specific crossing times -/
theorem train_length (bridge_length : ℝ) (bridge_time : ℝ) (post_time : ℝ) :
  bridge_length = 200 ∧ bridge_time = 10 ∧ post_time = 5 →
  ∃ train_length : ℝ, train_length = 200 ∧
    train_length / post_time = (train_length + bridge_length) / bridge_time :=
by sorry

end NUMINAMATH_CALUDE_train_length_l2589_258927


namespace NUMINAMATH_CALUDE_percentage_difference_l2589_258986

theorem percentage_difference (x y : ℝ) : 
  3 = 0.15 * x → 3 = 0.30 * y → x - y = 10 := by sorry

end NUMINAMATH_CALUDE_percentage_difference_l2589_258986


namespace NUMINAMATH_CALUDE_songs_learned_correct_l2589_258921

/-- The number of songs Vincent knew before summer camp -/
def songs_before : ℕ := 56

/-- The number of songs Vincent knows after summer camp -/
def songs_after : ℕ := 74

/-- The number of songs Vincent learned at summer camp -/
def songs_learned : ℕ := songs_after - songs_before

theorem songs_learned_correct : songs_learned = 18 := by sorry

end NUMINAMATH_CALUDE_songs_learned_correct_l2589_258921


namespace NUMINAMATH_CALUDE_total_seashells_eq_sum_l2589_258947

/-- The number of seashells Dan found on the beach -/
def total_seashells : ℕ := 56

/-- The number of seashells Dan gave to Jessica -/
def seashells_given : ℕ := 34

/-- The number of seashells Dan has left -/
def seashells_left : ℕ := 22

/-- Theorem stating that the total number of seashells is equal to
    the sum of seashells given away and seashells left -/
theorem total_seashells_eq_sum :
  total_seashells = seashells_given + seashells_left := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_eq_sum_l2589_258947


namespace NUMINAMATH_CALUDE_florist_roses_l2589_258928

/-- 
Given a florist who:
- Sells 15 roses
- Picks 21 more roses
- Ends up with 56 roses
Prove that she must have started with 50 roses
-/
theorem florist_roses (initial : ℕ) : 
  initial - 15 + 21 = 56 → initial = 50 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_l2589_258928


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2589_258962

/-- The distance between the vertices of the hyperbola y^2/45 - x^2/20 = 1 is 6√5 -/
theorem hyperbola_vertex_distance : 
  let a := Real.sqrt 45
  let vertex_distance := 2 * a
  vertex_distance = 6 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2589_258962


namespace NUMINAMATH_CALUDE_meeting_point_theorem_l2589_258903

/-- The meeting point of two people, given their positions and the fraction of the distance between them -/
def meetingPoint (x₁ y₁ x₂ y₂ t : ℝ) : ℝ × ℝ :=
  ((1 - t) * x₁ + t * x₂, (1 - t) * y₁ + t * y₂)

/-- Theorem stating that the meeting point one-third of the way from (2, 3) to (8, -5) is (4, 1/3) -/
theorem meeting_point_theorem :
  let mark_pos : ℝ × ℝ := (2, 3)
  let sandy_pos : ℝ × ℝ := (8, -5)
  let t : ℝ := 1/3
  meetingPoint mark_pos.1 mark_pos.2 sandy_pos.1 sandy_pos.2 t = (4, 1/3) := by
  sorry

end NUMINAMATH_CALUDE_meeting_point_theorem_l2589_258903


namespace NUMINAMATH_CALUDE_rectangle_perimeter_in_square_l2589_258911

theorem rectangle_perimeter_in_square (d : ℝ) (h : d = 6) : 
  ∃ (s : ℝ), s > 0 ∧ s * Real.sqrt 2 = d ∧
  ∃ (rect_side : ℝ), rect_side = s / Real.sqrt 2 ∧
  4 * rect_side = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_in_square_l2589_258911


namespace NUMINAMATH_CALUDE_darias_piggy_bank_problem_l2589_258997

/-- The problem of calculating Daria's initial piggy bank balance. -/
theorem darias_piggy_bank_problem
  (vacuum_cost : ℕ)
  (weekly_savings : ℕ)
  (weeks_to_save : ℕ)
  (h1 : vacuum_cost = 120)
  (h2 : weekly_savings = 10)
  (h3 : weeks_to_save = 10)
  (h4 : vacuum_cost = weekly_savings * weeks_to_save + initial_balance) :
  initial_balance = 20 :=
by
  sorry

#check darias_piggy_bank_problem

end NUMINAMATH_CALUDE_darias_piggy_bank_problem_l2589_258997


namespace NUMINAMATH_CALUDE_xyz_value_l2589_258926

theorem xyz_value (a b c x y z : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : a = (b + c) / (x - 3))
  (eq2 : b = (a + c) / (y - 3))
  (eq3 : c = (a + b) / (z - 3))
  (eq4 : x * y + x * z + y * z = 8)
  (eq5 : x + y + z = 4) :
  x * y * z = 10 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l2589_258926


namespace NUMINAMATH_CALUDE_coin_division_sum_25_l2589_258972

/-- Represents the sum of products for coin divisions -/
def sum_of_products (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

/-- Theorem: The sum of products for 25 coins is 300 -/
theorem coin_division_sum_25 :
  sum_of_products 25 = 300 := by
  sorry

#eval sum_of_products 25  -- Should output 300

end NUMINAMATH_CALUDE_coin_division_sum_25_l2589_258972


namespace NUMINAMATH_CALUDE_expression_simplification_l2589_258900

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (1 - x / (x + 1)) / ((x^2 - 1) / (x^2 + 2*x + 1)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2589_258900


namespace NUMINAMATH_CALUDE_reflection_matrix_condition_l2589_258901

def reflection_matrix (a b : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![a, b; -3/2, 1/2]

theorem reflection_matrix_condition (a b : ℚ) :
  (reflection_matrix a b) ^ 2 = 1 ↔ a = -1/2 ∧ b = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_reflection_matrix_condition_l2589_258901


namespace NUMINAMATH_CALUDE_function_derivative_equality_l2589_258958

theorem function_derivative_equality (f : ℝ → ℝ) (x : ℝ) : 
  (∀ x, f x = x^2 * (x - 1)) → 
  (deriv f) x = x → 
  x = 0 ∨ x = 1 := by
sorry

end NUMINAMATH_CALUDE_function_derivative_equality_l2589_258958


namespace NUMINAMATH_CALUDE_mans_rate_in_still_water_l2589_258918

theorem mans_rate_in_still_water 
  (speed_with_stream : ℝ) 
  (speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 6)
  (h2 : speed_against_stream = 2) : 
  (speed_with_stream + speed_against_stream) / 2 = 4 := by
  sorry

#check mans_rate_in_still_water

end NUMINAMATH_CALUDE_mans_rate_in_still_water_l2589_258918


namespace NUMINAMATH_CALUDE_softball_team_size_l2589_258991

theorem softball_team_size :
  ∀ (men women : ℕ),
  women = men + 4 →
  (men : ℚ) / (women : ℚ) = 7/11 →
  men + women = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_softball_team_size_l2589_258991
