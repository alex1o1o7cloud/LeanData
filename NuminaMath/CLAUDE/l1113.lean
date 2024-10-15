import Mathlib

namespace NUMINAMATH_CALUDE_product_divisible_by_14_l1113_111391

theorem product_divisible_by_14 (a b c d : ℤ) (h : 7*a + 8*b = 14*c + 28*d) : 
  14 ∣ (a * b) := by
sorry

end NUMINAMATH_CALUDE_product_divisible_by_14_l1113_111391


namespace NUMINAMATH_CALUDE_f_properties_l1113_111373

noncomputable def f (x : ℝ) := Real.cos (2 * x) + 2 * Real.sin x * Real.sin x

theorem f_properties :
  (∃ p : ℝ, p > 0 ∧ ∀ x, f (x + p) = f x ∧ ∀ q, q > 0 ∧ (∀ x, f (x + q) = f x) → p ≤ q) ∧
  (∃ M : ℝ, ∀ x, f x ≤ M ∧ ∃ x, f x = M) ∧
  (∀ k : ℤ, f (k * Real.pi) = 2) ∧
  (∀ A : ℝ, A > 0 ∧ A < Real.pi / 2 →
    f A = 0 →
    ∀ b a : ℝ, b = 5 ∧ a = 7 →
    ∃ c : ℝ, c > 0 ∧
    (1/2) * b * c * Real.sin A = 10) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1113_111373


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1113_111304

def A : Set ℝ := {-2, -1, 0, 1}
def B : Set ℝ := {x : ℝ | x^2 - 1 ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1113_111304


namespace NUMINAMATH_CALUDE_square_difference_301_299_l1113_111324

theorem square_difference_301_299 : 301^2 - 299^2 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_301_299_l1113_111324


namespace NUMINAMATH_CALUDE_max_type_c_tubes_exists_solution_with_73_type_c_l1113_111323

/-- Represents the types of test tubes -/
inductive TubeType
  | A
  | B
  | C

/-- Represents a solution of test tubes -/
structure Solution where
  a : ℕ  -- number of type A tubes
  b : ℕ  -- number of type B tubes
  c : ℕ  -- number of type C tubes

/-- The concentration of the solution in each type of tube -/
def concentration : TubeType → ℚ
  | TubeType.A => 1/10
  | TubeType.B => 1/5
  | TubeType.C => 9/10

/-- The total number of tubes used -/
def Solution.total (s : Solution) : ℕ := s.a + s.b + s.c

/-- The average concentration of the final solution -/
def Solution.averageConcentration (s : Solution) : ℚ :=
  (s.a * concentration TubeType.A + s.b * concentration TubeType.B + s.c * concentration TubeType.C) / s.total

/-- Predicate to check if the solution satisfies the conditions -/
def Solution.isValid (s : Solution) : Prop :=
  s.averageConcentration = 20.17/100 ∧
  s.total ≥ 3 ∧
  s.a > 0 ∧ s.b > 0 ∧ s.c > 0

theorem max_type_c_tubes (s : Solution) (h : s.isValid) :
  s.c ≤ 73 :=
sorry

theorem exists_solution_with_73_type_c :
  ∃ s : Solution, s.isValid ∧ s.c = 73 :=
sorry

end NUMINAMATH_CALUDE_max_type_c_tubes_exists_solution_with_73_type_c_l1113_111323


namespace NUMINAMATH_CALUDE_sum_of_y_coords_on_y_axis_l1113_111318

-- Define the circle
def circle_center : ℝ × ℝ := (-6, 2)
def circle_radius : ℝ := 10

-- Define a point on the circle
def point_on_circle (p : ℝ × ℝ) : Prop :=
  (p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2 = circle_radius^2

-- Define a point on the y-axis
def point_on_y_axis (p : ℝ × ℝ) : Prop :=
  p.1 = 0

-- Theorem statement
theorem sum_of_y_coords_on_y_axis :
  ∃ (p1 p2 : ℝ × ℝ),
    point_on_circle p1 ∧ point_on_y_axis p1 ∧
    point_on_circle p2 ∧ point_on_y_axis p2 ∧
    p1 ≠ p2 ∧
    p1.2 + p2.2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_y_coords_on_y_axis_l1113_111318


namespace NUMINAMATH_CALUDE_smallest_fourth_number_l1113_111327

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def digits_sum (n : ℕ) : ℕ :=
  let d₁ := n / 10
  let d₂ := n % 10
  d₁ + d₂

def strictly_increasing_digits (n : ℕ) : Prop :=
  let d₁ := n / 10
  let d₂ := n % 10
  d₁ < d₂

theorem smallest_fourth_number :
  ∃ (n : ℕ),
    is_two_digit n ∧
    strictly_increasing_digits n ∧
    (∀ m, is_two_digit m → strictly_increasing_digits m →
      digits_sum 34 + digits_sum 18 + digits_sum 73 + digits_sum n +
      digits_sum m = (34 + 18 + 73 + n + m) / 6 →
      n ≤ m) ∧
    digits_sum 34 + digits_sum 18 + digits_sum 73 + digits_sum n =
      (34 + 18 + 73 + n) / 6 ∧
    n = 29 :=
by sorry

end NUMINAMATH_CALUDE_smallest_fourth_number_l1113_111327


namespace NUMINAMATH_CALUDE_least_number_divisible_l1113_111306

theorem least_number_divisible (n : ℕ) : n = 861 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 24 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 32 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 36 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 54 * k)) ∧
  (∃ k1 k2 k3 k4 : ℕ, (n + 3) = 24 * k1 ∧ (n + 3) = 32 * k2 ∧ (n + 3) = 36 * k3 ∧ (n + 3) = 54 * k4) :=
by sorry

#check least_number_divisible

end NUMINAMATH_CALUDE_least_number_divisible_l1113_111306


namespace NUMINAMATH_CALUDE_fraction_equality_l1113_111353

theorem fraction_equality (m n : ℚ) (h : m / n = 3 / 4) : 
  (m + n) / n = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1113_111353


namespace NUMINAMATH_CALUDE_probability_theorem_l1113_111367

def shirts : ℕ := 6
def shorts : ℕ := 8
def socks : ℕ := 7
def total_items : ℕ := shirts + shorts + socks
def items_chosen : ℕ := 4

def probability_specific_combination : ℚ :=
  (Nat.choose shirts 1 * Nat.choose shorts 2 * Nat.choose socks 1) /
  Nat.choose total_items items_chosen

theorem probability_theorem :
  probability_specific_combination = 392 / 1995 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l1113_111367


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1113_111302

theorem contrapositive_equivalence (a b : ℝ) : 
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔ (a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1113_111302


namespace NUMINAMATH_CALUDE_remainder_of_product_with_modular_inverse_l1113_111312

theorem remainder_of_product_with_modular_inverse (n a b : ℤ) : 
  n > 0 → (a * b) % n = 1 % n → (a * b) % n = 1 :=
by sorry

end NUMINAMATH_CALUDE_remainder_of_product_with_modular_inverse_l1113_111312


namespace NUMINAMATH_CALUDE_water_remaining_l1113_111377

/-- Given 3 gallons of water and using 5/4 gallons in an experiment, 
    prove that the remaining amount is 7/4 gallons. -/
theorem water_remaining (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → used = 5/4 → remaining = initial - used → remaining = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_water_remaining_l1113_111377


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_l1113_111384

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (line_perpendicular : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane 
  (m : Line) (α β : Plane) 
  (h1 : perpendicular α β) 
  (h2 : line_perpendicular m β) 
  (h3 : ¬ line_in_plane m α) : 
  line_parallel m α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_l1113_111384


namespace NUMINAMATH_CALUDE_opposite_numbers_equation_product_l1113_111319

theorem opposite_numbers_equation_product : ∀ x : ℤ, 
  (3 * x - 2 * (-x) = 30) → (x * (-x) = -36) := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_equation_product_l1113_111319


namespace NUMINAMATH_CALUDE_expression_evaluation_l1113_111355

/-- Given x = 3, y = 2, and z = 4, prove that 3 * x - 2 * y + 4 * z = 21 -/
theorem expression_evaluation (x y z : ℕ) (hx : x = 3) (hy : y = 2) (hz : z = 4) :
  3 * x - 2 * y + 4 * z = 21 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1113_111355


namespace NUMINAMATH_CALUDE_output_increase_l1113_111300

theorem output_increase (production_increase : Real) (hours_decrease : Real) : 
  production_increase = 0.8 →
  hours_decrease = 0.1 →
  ((1 + production_increase) / (1 - hours_decrease) - 1) * 100 = 100 := by
sorry

end NUMINAMATH_CALUDE_output_increase_l1113_111300


namespace NUMINAMATH_CALUDE_apples_left_after_pie_l1113_111338

theorem apples_left_after_pie (initial_apples : Real) (anita_contribution : Real) (pie_requirement : Real) :
  initial_apples = 10.0 →
  anita_contribution = 5.0 →
  pie_requirement = 4.0 →
  initial_apples + anita_contribution - pie_requirement = 11.0 := by
  sorry

end NUMINAMATH_CALUDE_apples_left_after_pie_l1113_111338


namespace NUMINAMATH_CALUDE_smallest_x_for_equation_l1113_111381

theorem smallest_x_for_equation : 
  ∃ (x : ℕ), x > 0 ∧ 
  (∃ (y : ℕ), y > 0 ∧ (0.8 : ℚ) = y / (196 + x)) ∧
  (∀ (x' : ℕ), x' > 0 → x' < x → 
    ¬∃ (y : ℕ), y > 0 ∧ (0.8 : ℚ) = y / (196 + x')) ∧
  x = 49 := by
sorry

end NUMINAMATH_CALUDE_smallest_x_for_equation_l1113_111381


namespace NUMINAMATH_CALUDE_expression_evaluation_l1113_111369

theorem expression_evaluation :
  let x : ℝ := 3 + Real.sqrt 2
  (1 - 5 / (x + 2)) / ((x^2 - 6*x + 9) / (x + 2)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1113_111369


namespace NUMINAMATH_CALUDE_min_value_of_function_l1113_111315

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  x + 3 / (4 * x) ≥ Real.sqrt 3 ∧ ∃ y > 0, y + 3 / (4 * y) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1113_111315


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1113_111342

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  (∀ n, S n = (a 1 + a n) * n / 2) →   -- sum formula for arithmetic sequence
  (a 2 - 1)^3 + 2014 * (a 2 - 1) = Real.sin (2011 * Real.pi / 3) →
  (a 2013 - 1)^3 + 2014 * (a 2013 - 1) = Real.cos (2011 * Real.pi / 6) →
  S 2014 = 2014 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1113_111342


namespace NUMINAMATH_CALUDE_smallest_multiple_of_seven_l1113_111307

def is_valid_abc (a b c : ℕ) : Prop :=
  a > 3 ∧ b > 3 ∧ c > 3 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

def form_number (a b c : ℕ) : ℕ :=
  100000 * a + 10000 * b + 1000 * c + 321

theorem smallest_multiple_of_seven :
  ∀ a b c : ℕ,
    is_valid_abc a b c →
    form_number a b c ≥ 468321 ∨ ¬(form_number a b c % 7 = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_seven_l1113_111307


namespace NUMINAMATH_CALUDE_friends_receiving_pens_correct_l1113_111341

/-- Calculate the number of friends who will receive pens --/
def friends_receiving_pens (kendra_packs tony_packs maria_packs : ℕ)
                           (kendra_pens_per_pack tony_pens_per_pack maria_pens_per_pack : ℕ)
                           (pens_kept_per_person : ℕ) : ℕ :=
  let kendra_total := kendra_packs * kendra_pens_per_pack
  let tony_total := tony_packs * tony_pens_per_pack
  let maria_total := maria_packs * maria_pens_per_pack
  let total_pens := kendra_total + tony_total + maria_total
  let total_kept := 3 * pens_kept_per_person
  total_pens - total_kept

theorem friends_receiving_pens_correct :
  friends_receiving_pens 7 5 9 4 6 5 3 = 94 := by
  sorry

end NUMINAMATH_CALUDE_friends_receiving_pens_correct_l1113_111341


namespace NUMINAMATH_CALUDE_unique_triple_solution_l1113_111372

theorem unique_triple_solution :
  ∀ a b c : ℕ+,
    (∃ k₁ : ℕ, a * b + 1 = k₁ * c) ∧
    (∃ k₂ : ℕ, a * c + 1 = k₂ * b) ∧
    (∃ k₃ : ℕ, b * c + 1 = k₃ * a) →
    a = 1 ∧ b = 1 ∧ c = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l1113_111372


namespace NUMINAMATH_CALUDE_point_distance_theorem_l1113_111379

/-- Given a point P with coordinates (x^2 - k, -5), where k is a positive constant,
    if the distance from P to the x-axis is half the distance from P to the y-axis,
    and the total distance from P to both axes is 15 units,
    then k = x^2 - 10. -/
theorem point_distance_theorem (x k : ℝ) (h1 : k > 0) :
  let P : ℝ × ℝ := (x^2 - k, -5)
  abs P.2 = (1/2) * abs P.1 →
  abs P.2 + abs P.1 = 15 →
  k = x^2 - 10 := by
sorry

end NUMINAMATH_CALUDE_point_distance_theorem_l1113_111379


namespace NUMINAMATH_CALUDE_consecutive_integers_right_triangle_l1113_111378

theorem consecutive_integers_right_triangle (m n : ℕ) (h : n^2 = 2*m + 1) :
  n^2 + m^2 = (m + 1)^2 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_right_triangle_l1113_111378


namespace NUMINAMATH_CALUDE_rectangle_area_stage_8_l1113_111337

/-- The area of a rectangle formed by adding n squares of side length s --/
def rectangleArea (n : ℕ) (s : ℝ) : ℝ := n * (s * s)

/-- Theorem: The area of a rectangle formed by adding 8 squares, each 4 inches by 4 inches, is 128 square inches --/
theorem rectangle_area_stage_8 : rectangleArea 8 4 = 128 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_stage_8_l1113_111337


namespace NUMINAMATH_CALUDE_binomial_30_choose_3_l1113_111330

theorem binomial_30_choose_3 : Nat.choose 30 3 = 4060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_30_choose_3_l1113_111330


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_21_19_l1113_111376

theorem half_abs_diff_squares_21_19 : 
  (1 / 2 : ℝ) * |21^2 - 19^2| = 40 := by sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_21_19_l1113_111376


namespace NUMINAMATH_CALUDE_average_speed_calculation_l1113_111351

def initial_reading : ℕ := 2552
def final_reading : ℕ := 2772
def total_time : ℕ := 9

theorem average_speed_calculation :
  (final_reading - initial_reading : ℚ) / total_time = 220 / 9 := by sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l1113_111351


namespace NUMINAMATH_CALUDE_simplify_expression_l1113_111392

theorem simplify_expression (a b : ℝ) : 
  (15*a + 45*b) + (20*a + 35*b) - (25*a + 55*b) + (30*a - 5*b) = 40*a + 20*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1113_111392


namespace NUMINAMATH_CALUDE_cauchy_functional_equation_l1113_111356

theorem cauchy_functional_equation 
  (f : ℚ → ℚ) 
  (h : ∀ x y : ℚ, f (x + y) = f x + f y) : 
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x :=
by sorry

end NUMINAMATH_CALUDE_cauchy_functional_equation_l1113_111356


namespace NUMINAMATH_CALUDE_stationery_cost_is_52_66_l1113_111321

/-- Represents the cost calculation for a set of stationery items -/
def stationery_cost (usd_to_cad_rate : ℝ) : ℝ := by
  -- Define the base costs
  let pencil_cost : ℝ := 2
  let pen_cost : ℝ := pencil_cost + 9
  let notebook_cost : ℝ := 2 * pen_cost

  -- Apply discounts
  let discounted_notebook_cost : ℝ := notebook_cost * 0.85
  let discounted_pen_cost : ℝ := pen_cost * 0.8

  -- Calculate total cost in USD before tax
  let total_usd_before_tax : ℝ := pencil_cost + 2 * discounted_pen_cost + discounted_notebook_cost

  -- Apply tax
  let total_usd_with_tax : ℝ := total_usd_before_tax * 1.1

  -- Convert to CAD
  exact total_usd_with_tax * usd_to_cad_rate

/-- Theorem stating that the total cost of the stationery items is $52.66 CAD -/
theorem stationery_cost_is_52_66 :
  stationery_cost 1.25 = 52.66 := by
  sorry

end NUMINAMATH_CALUDE_stationery_cost_is_52_66_l1113_111321


namespace NUMINAMATH_CALUDE_smallest_among_four_numbers_l1113_111375

theorem smallest_among_four_numbers :
  let a : ℝ := -Real.sqrt 3
  let b : ℝ := 0
  let c : ℝ := 2
  let d : ℝ := -3
  d < a ∧ d < b ∧ d < c := by sorry

end NUMINAMATH_CALUDE_smallest_among_four_numbers_l1113_111375


namespace NUMINAMATH_CALUDE_negative_integer_solution_of_inequality_l1113_111362

theorem negative_integer_solution_of_inequality :
  ∀ x : ℤ, x < 0 →
    (((2 * x - 1 : ℚ) / 3) - ((5 * x + 1 : ℚ) / 2) ≤ 1) ↔ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_negative_integer_solution_of_inequality_l1113_111362


namespace NUMINAMATH_CALUDE_sally_pokemon_cards_l1113_111360

theorem sally_pokemon_cards (initial : ℕ) (new : ℕ) (lost : ℕ) : 
  initial = 27 → new = 41 → lost = 20 → initial + new - lost = 48 := by
  sorry

end NUMINAMATH_CALUDE_sally_pokemon_cards_l1113_111360


namespace NUMINAMATH_CALUDE_xyz_sum_equals_96_l1113_111320

theorem xyz_sum_equals_96 
  (x y z : ℝ) 
  (hpos_x : x > 0) 
  (hpos_y : y > 0) 
  (hpos_z : z > 0)
  (eq1 : x^2 + x*y + y^2 = 108)
  (eq2 : y^2 + y*z + z^2 = 64)
  (eq3 : z^2 + x*z + x^2 = 172) : 
  x*y + y*z + x*z = 96 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_equals_96_l1113_111320


namespace NUMINAMATH_CALUDE_blocks_differing_in_two_ways_l1113_111388

def num_materials : Nat := 3
def num_sizes : Nat := 2
def num_colors : Nat := 5
def num_shapes : Nat := 4

def count_different_blocks : Nat :=
  (num_materials - 1) * 1 + -- Material and Size
  (num_materials - 1) * (num_colors - 1) + -- Material and Color
  (num_materials - 1) * (num_shapes - 1) + -- Material and Shape
  1 * (num_colors - 1) + -- Size and Color
  1 * (num_shapes - 1) + -- Size and Shape
  (num_colors - 1) * (num_shapes - 1) -- Color and Shape

theorem blocks_differing_in_two_ways :
  count_different_blocks = 35 := by
  sorry

end NUMINAMATH_CALUDE_blocks_differing_in_two_ways_l1113_111388


namespace NUMINAMATH_CALUDE_fuel_left_in_tank_l1113_111332

/-- Calculates the remaining fuel in a plane's tank given the fuel consumption rate and remaining flight time. -/
def remaining_fuel (fuel_rate : ℝ) (flight_time : ℝ) : ℝ :=
  fuel_rate * flight_time

/-- Proves that given a plane using fuel at a rate of 9.5 gallons per hour and can continue flying for 0.6667 hours, the amount of fuel left in the tank is approximately 6.33365 gallons. -/
theorem fuel_left_in_tank : 
  let fuel_rate := 9.5
  let flight_time := 0.6667
  abs (remaining_fuel fuel_rate flight_time - 6.33365) < 0.00001 := by
sorry

end NUMINAMATH_CALUDE_fuel_left_in_tank_l1113_111332


namespace NUMINAMATH_CALUDE_units_digit_not_zero_l1113_111365

theorem units_digit_not_zero (a b : Nat) (ha : a ∈ Finset.range 100) (hb : b ∈ Finset.range 100) :
  (5^a + 6^b) % 10 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_units_digit_not_zero_l1113_111365


namespace NUMINAMATH_CALUDE_square_formation_proof_l1113_111343

def is_perfect_square (n : Nat) : Prop := ∃ m : Nat, n = m * m

def piece_sizes : List Nat := [4, 5, 6, 7, 8]

def total_squares : Nat := piece_sizes.sum

theorem square_formation_proof :
  ∃ (removed_piece : Nat),
    removed_piece ∈ piece_sizes ∧
    is_perfect_square (total_squares - removed_piece) ∧
    removed_piece = 5 :=
  sorry

end NUMINAMATH_CALUDE_square_formation_proof_l1113_111343


namespace NUMINAMATH_CALUDE_cross_quadrilateral_area_l1113_111352

/-- Given two rectangles ABCD and EFGH forming a cross shape, 
    prove that the area of quadrilateral AFCH is 52.5 -/
theorem cross_quadrilateral_area 
  (AB BC EF FG : ℝ) 
  (h_AB : AB = 9) 
  (h_BC : BC = 5) 
  (h_EF : EF = 3) 
  (h_FG : FG = 10) : 
  Real.sqrt ((AB * FG / 2 + BC * EF / 2) ^ 2 + (AB * BC + EF * FG - BC * EF) ^ 2) = 52.5 := by
  sorry

end NUMINAMATH_CALUDE_cross_quadrilateral_area_l1113_111352


namespace NUMINAMATH_CALUDE_unreachable_from_2_2_2_reachable_from_3_3_3_l1113_111397

/-- The operation that replaces one number with the difference between the sum of the other two and 1 -/
def operation (x y z : ℤ) : ℤ × ℤ × ℤ → Prop :=
  fun w => (w = (y + z - 1, y, z)) ∨ (w = (x, x + z - 1, z)) ∨ (w = (x, y, x + y - 1))

/-- The relation that represents the repeated application of the operation -/
inductive reachable : ℤ × ℤ × ℤ → ℤ × ℤ × ℤ → Prop
  | refl {x} : reachable x x
  | step {x y z} (h : reachable x y) (o : operation y.1 y.2.1 y.2.2 z) : reachable x z

theorem unreachable_from_2_2_2 :
  ¬ reachable (2, 2, 2) (17, 1999, 2105) :=
sorry

theorem reachable_from_3_3_3 :
  reachable (3, 3, 3) (17, 1999, 2105) :=
sorry

end NUMINAMATH_CALUDE_unreachable_from_2_2_2_reachable_from_3_3_3_l1113_111397


namespace NUMINAMATH_CALUDE_inequality_and_minimum_l1113_111387

theorem inequality_and_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (Real.sqrt (a + 1/2) + Real.sqrt (b + 1/2) ≤ 2) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 1/y + 1/(x*y) ≥ 8) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_minimum_l1113_111387


namespace NUMINAMATH_CALUDE_three_pairs_same_difference_l1113_111344

theorem three_pairs_same_difference (X : Finset ℕ) 
  (h1 : X ⊆ Finset.range 18 \ {0})
  (h2 : X.card = 8) : 
  ∃ (a b c d e f : ℕ), a ∈ X ∧ b ∈ X ∧ c ∈ X ∧ d ∈ X ∧ e ∈ X ∧ f ∈ X ∧ 
  a ≠ b ∧ c ≠ d ∧ e ≠ f ∧
  (a - b : ℤ) = (c - d : ℤ) ∧ (c - d : ℤ) = (e - f : ℤ) :=
by sorry

end NUMINAMATH_CALUDE_three_pairs_same_difference_l1113_111344


namespace NUMINAMATH_CALUDE_total_tiles_is_183_l1113_111325

/-- Calculates the number of tiles needed for a room with given dimensions and tile specifications. -/
def calculate_tiles (room_length room_width border_width : ℕ) 
  (border_tile_size inner_tile_size : ℕ) : ℕ :=
  let inner_length := room_length - 2 * border_width
  let inner_width := room_width - 2 * border_width
  let border_tiles := 2 * (room_length + room_width) * (border_width / border_tile_size) +
                      4 * (border_width / border_tile_size) ^ 2
  let inner_tiles := (inner_length * inner_width) / (inner_tile_size ^ 2)
  border_tiles + inner_tiles

/-- Theorem stating that the total number of tiles for the given room specifications is 183. -/
theorem total_tiles_is_183 :
  calculate_tiles 24 18 2 1 3 = 183 := by sorry

end NUMINAMATH_CALUDE_total_tiles_is_183_l1113_111325


namespace NUMINAMATH_CALUDE_triangle_angle_difference_l1113_111316

theorem triangle_angle_difference (a b c : ℝ) : 
  a = 64 ∧ b = 64 ∧ c < a ∧ a + b + c = 180 → a - c = 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_difference_l1113_111316


namespace NUMINAMATH_CALUDE_count_five_divisors_l1113_111390

theorem count_five_divisors (n : ℕ) (h : n = 50000) : 
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125) + (n / 15625) = 12499 :=
by sorry

end NUMINAMATH_CALUDE_count_five_divisors_l1113_111390


namespace NUMINAMATH_CALUDE_smallest_four_digit_solution_l1113_111358

theorem smallest_four_digit_solution (x : ℕ) : x = 1053 ↔ 
  (x ≥ 1000 ∧ x < 10000) ∧ 
  (∀ y : ℕ, y ≥ 1000 ∧ y < 10000 →
    (9 * y ≡ 27 [ZMOD 15] ∧
     3 * y + 15 ≡ 21 [ZMOD 8] ∧
     -3 * y + 4 ≡ 2 * y + 5 [ZMOD 16]) →
    x ≤ y) ∧
  (9 * x ≡ 27 [ZMOD 15]) ∧
  (3 * x + 15 ≡ 21 [ZMOD 8]) ∧
  (-3 * x + 4 ≡ 2 * x + 5 [ZMOD 16]) :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_solution_l1113_111358


namespace NUMINAMATH_CALUDE_exactly_two_balls_distribution_l1113_111317

-- Define the number of balls and boxes
def num_balls : ℕ := 5
def num_boxes : ℕ := 3

-- Define the function to calculate the number of ways to distribute balls
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  (n.choose 2) * k * (k ^ (n - 2))

-- Theorem statement
theorem exactly_two_balls_distribution :
  distribute_balls num_balls num_boxes = 810 :=
sorry

end NUMINAMATH_CALUDE_exactly_two_balls_distribution_l1113_111317


namespace NUMINAMATH_CALUDE_ceiling_square_count_l1113_111333

theorem ceiling_square_count (x : ℝ) (h : ⌈x⌉ = 15) : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, ∃ y : ℝ, ⌈y⌉ = 15 ∧ ⌈y^2⌉ = n) ∧ S.card = 29 :=
sorry

end NUMINAMATH_CALUDE_ceiling_square_count_l1113_111333


namespace NUMINAMATH_CALUDE_min_value_of_f_l1113_111382

/-- The function we want to minimize -/
def f (x y : ℝ) : ℝ := 3*x^2 + 4*x*y + 2*y^2 - 6*x - 8*y + 6

theorem min_value_of_f :
  (∀ x y : ℝ, f x y ≥ 0) ∧ f (-1) 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1113_111382


namespace NUMINAMATH_CALUDE_coefficient_of_x_l1113_111303

theorem coefficient_of_x (x : ℝ) : 
  let expression := 5*(x - 6) + 3*(9 - 3*x^2 + 2*x) - 10*(3*x - 2)
  ∃ (a b c : ℝ), expression = a*x^2 + (-19)*x + c :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_x_l1113_111303


namespace NUMINAMATH_CALUDE_sum_trailing_zeros_15_factorial_l1113_111366

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def trailingZerosBase10 (n : ℕ) : ℕ := 
  (List.range 5).foldl (fun acc i => acc + n / (5 ^ (i + 1))) 0

def trailingZerosBase12 (n : ℕ) : ℕ := 
  min 
    ((List.range 2).foldl (fun acc i => acc + n / (3 ^ (i + 1))) 0)
    ((List.range 3).foldl (fun acc i => acc + n / (2 ^ (i + 1))) 0 / 2)

theorem sum_trailing_zeros_15_factorial : 
  trailingZerosBase12 (factorial 15) + trailingZerosBase10 (factorial 15) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_trailing_zeros_15_factorial_l1113_111366


namespace NUMINAMATH_CALUDE_resulting_polygon_sides_bound_resulting_polygon_sides_bound_even_l1113_111393

/-- Represents a convex n-gon with all diagonals drawn --/
structure ConvexNGonWithDiagonals (n : ℕ) where
  -- Add necessary fields here

/-- Represents a polygon resulting from the division of the n-gon by its diagonals --/
structure ResultingPolygon (n : ℕ) where
  -- Add necessary fields here

/-- The number of sides of a resulting polygon --/
def num_sides (p : ResultingPolygon n) : ℕ := sorry

theorem resulting_polygon_sides_bound (n : ℕ) (ngon : ConvexNGonWithDiagonals n) 
  (p : ResultingPolygon n) : num_sides p ≤ n := by sorry

theorem resulting_polygon_sides_bound_even (n : ℕ) (ngon : ConvexNGonWithDiagonals n) 
  (p : ResultingPolygon n) (h : Even n) : num_sides p ≤ n - 1 := by sorry

end NUMINAMATH_CALUDE_resulting_polygon_sides_bound_resulting_polygon_sides_bound_even_l1113_111393


namespace NUMINAMATH_CALUDE_troy_beef_purchase_l1113_111335

/-- Represents the problem of determining the amount of beef Troy buys -/
theorem troy_beef_purchase 
  (veg_pounds : ℝ) 
  (veg_price : ℝ) 
  (beef_price_multiplier : ℝ) 
  (total_cost : ℝ) 
  (h1 : veg_pounds = 6)
  (h2 : veg_price = 2)
  (h3 : beef_price_multiplier = 3)
  (h4 : total_cost = 36) :
  ∃ (beef_pounds : ℝ), 
    beef_pounds * (veg_price * beef_price_multiplier) + veg_pounds * veg_price = total_cost ∧ 
    beef_pounds = 4 := by
  sorry

end NUMINAMATH_CALUDE_troy_beef_purchase_l1113_111335


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l1113_111308

theorem rectangular_box_volume (x : ℕ) (h : x > 0) :
  let volume := 10 * x^3
  (volume = 60 ∨ volume = 80 ∨ volume = 100 ∨ volume = 120 ∨ volume = 200) →
  volume = 80 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l1113_111308


namespace NUMINAMATH_CALUDE_circle_diameter_relation_l1113_111339

theorem circle_diameter_relation (R S : Real) (h : R > 0 ∧ S > 0) :
  (R * R) / (S * S) = 0.16 → R / S = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_relation_l1113_111339


namespace NUMINAMATH_CALUDE_train_crossing_time_l1113_111346

/-- Given two trains of equal length, prove the time taken by one train to cross a telegraph post. -/
theorem train_crossing_time (train_length : ℝ) (time_second_train : ℝ) (time_crossing_each_other : ℝ) :
  train_length = 120 →
  time_second_train = 15 →
  time_crossing_each_other = 12 →
  ∃ (time_first_train : ℝ),
    time_first_train = 10 ∧
    train_length / time_first_train + train_length / time_second_train =
      2 * train_length / time_crossing_each_other :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1113_111346


namespace NUMINAMATH_CALUDE_sum_inequality_l1113_111371

theorem sum_inequality (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l1113_111371


namespace NUMINAMATH_CALUDE_intersection_equality_implies_a_range_l1113_111361

def A : Set ℝ := {x | (1/2 : ℝ) ≤ x ∧ x ≤ 3}
def B (a : ℝ) : Set ℝ := {x | x^2 + a < 0}

theorem intersection_equality_implies_a_range (a : ℝ) :
  (Aᶜ ∩ B a = B a) → a ≥ -1/4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_a_range_l1113_111361


namespace NUMINAMATH_CALUDE_specific_right_triangle_with_square_l1113_111385

/-- Represents a right triangle with a square inscribed on its hypotenuse -/
structure RightTriangleWithSquare where
  /-- Length of one leg of the right triangle -/
  leg1 : ℝ
  /-- Length of the other leg of the right triangle -/
  leg2 : ℝ
  /-- Side length of the inscribed square -/
  square_side : ℝ
  /-- Distance from the right angle vertex to the side of the square on the hypotenuse -/
  distance_to_square : ℝ

/-- Theorem stating the properties of the specific right triangle with inscribed square -/
theorem specific_right_triangle_with_square :
  ∃ (t : RightTriangleWithSquare),
    t.leg1 = 9 ∧
    t.leg2 = 12 ∧
    t.square_side = 75 / 7 ∧
    t.distance_to_square = 36 / 5 := by
  sorry

end NUMINAMATH_CALUDE_specific_right_triangle_with_square_l1113_111385


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l1113_111357

theorem min_value_expression (x : ℝ) (hx : x > 0) : 2 * Real.sqrt x + 1 / x + x^2 ≥ 4 :=
by sorry

theorem equality_condition : 2 * Real.sqrt 1 + 1 / 1 + 1^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l1113_111357


namespace NUMINAMATH_CALUDE_complex_equation_result_l1113_111345

theorem complex_equation_result (z : ℂ) 
  (h : 15 * Complex.normSq z = 3 * Complex.normSq (z + 3) + Complex.normSq (z^2 + 4) + 25) : 
  z + 8 / z = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_result_l1113_111345


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1113_111374

theorem inequality_solution_set (a : ℝ) :
  (∀ x, (x - a) * (x + a - 1) > 0 ↔ 
    (a = 1/2 ∧ x ≠ 1/2) ∨
    (a < 1/2 ∧ (x > 1 - a ∨ x < a)) ∨
    (a > 1/2 ∧ (x > a ∨ x < 1 - a))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1113_111374


namespace NUMINAMATH_CALUDE_rachel_painting_time_l1113_111398

/-- Prove that Rachel's painting time is 13 hours -/
theorem rachel_painting_time :
  let matt_time : ℕ := 12
  let patty_time : ℕ := matt_time / 3
  let rachel_time : ℕ := 2 * patty_time + 5
  rachel_time = 13 := by
  sorry

end NUMINAMATH_CALUDE_rachel_painting_time_l1113_111398


namespace NUMINAMATH_CALUDE_no_max_value_cubic_l1113_111364

/-- The function f(x) = 3x^2 + 6x^3 + 27x + 100 has no maximum value over the real numbers -/
theorem no_max_value_cubic (x : ℝ) : 
  ¬∃ (M : ℝ), ∀ (x : ℝ), 3*x^2 + 6*x^3 + 27*x + 100 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_no_max_value_cubic_l1113_111364


namespace NUMINAMATH_CALUDE_probability_one_red_ball_l1113_111334

def total_balls : ℕ := 12
def red_balls : ℕ := 3
def black_balls : ℕ := 4
def white_balls : ℕ := 5
def drawn_balls : ℕ := 2

theorem probability_one_red_ball :
  (red_balls * (black_balls + white_balls)) / (total_balls.choose drawn_balls) = 9 / 22 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_red_ball_l1113_111334


namespace NUMINAMATH_CALUDE_conference_handshakes_l1113_111310

theorem conference_handshakes (n : ℕ) (h : n = 30) :
  (n * (n - 1)) / 2 = 435 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l1113_111310


namespace NUMINAMATH_CALUDE_triangle_max_area_l1113_111336

theorem triangle_max_area (a b c : ℝ) (A : ℝ) (h_a : a = 4) (h_A : A = π/3) :
  (∃ (S : ℝ), S = (1/2) * b * c * Real.sin A ∧ 
    ∀ (S' : ℝ), S' = (1/2) * b * c * Real.sin A → S' ≤ 4 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l1113_111336


namespace NUMINAMATH_CALUDE_sqrt_calculation_l1113_111368

theorem sqrt_calculation : 
  Real.sqrt 2 * Real.sqrt 6 - 4 * Real.sqrt (1/2) - (1 - Real.sqrt 3)^2 = 
  4 * Real.sqrt 3 - 2 * Real.sqrt 2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculation_l1113_111368


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1113_111396

theorem cubic_equation_solution :
  ∃ y : ℝ, y > 0 ∧ 5 * y^(1/3) - 3 * (y / y^(2/3)) = 10 + y^(1/3) ∧ y = 1000 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1113_111396


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1113_111329

theorem polynomial_simplification (x : ℝ) :
  2 * x^2 * (4 * x^3 - 3 * x + 1) - 7 * (x^3 - 3 * x^2 + 2 * x - 8) =
  8 * x^5 - 13 * x^3 + 23 * x^2 - 14 * x + 56 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1113_111329


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_implies_x_geq_one_l1113_111326

theorem sqrt_x_minus_one_real_implies_x_geq_one (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x - 1) → x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_implies_x_geq_one_l1113_111326


namespace NUMINAMATH_CALUDE_shaded_fraction_is_one_eighth_l1113_111389

/-- Represents a rectangle with given dimensions -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.length * r.width

/-- Represents the shaded area within a rectangle -/
structure ShadedRectangle where
  rectangle : Rectangle
  shaded_area : ℝ

theorem shaded_fraction_is_one_eighth 
  (r : Rectangle)
  (sr : ShadedRectangle)
  (h1 : r.length = 15)
  (h2 : r.width = 24)
  (h3 : sr.rectangle = r)
  (h4 : sr.shaded_area = (1 / 4) * (1 / 2) * r.area) :
  sr.shaded_area / r.area = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_shaded_fraction_is_one_eighth_l1113_111389


namespace NUMINAMATH_CALUDE_second_concert_attendance_l1113_111301

theorem second_concert_attendance 
  (first_concert : ℕ) 
  (additional_attendees : ℕ) 
  (h1 : first_concert = 65899)
  (h2 : additional_attendees = 119) :
  first_concert + additional_attendees = 66018 := by
sorry

end NUMINAMATH_CALUDE_second_concert_attendance_l1113_111301


namespace NUMINAMATH_CALUDE_circle_polar_to_cartesian_l1113_111340

/-- Given a circle with polar equation ρ = 2cos θ, its Cartesian equation is (x-1)^2 + y^2 = 1 -/
theorem circle_polar_to_cartesian :
  ∀ (x y ρ θ : ℝ),
  (ρ = 2 * Real.cos θ) →
  (x = ρ * Real.cos θ) →
  (y = ρ * Real.sin θ) →
  ((x - 1)^2 + y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_polar_to_cartesian_l1113_111340


namespace NUMINAMATH_CALUDE_mechanics_total_charge_l1113_111313

/-- Calculates the total amount charged by two mechanics working on a car. -/
theorem mechanics_total_charge
  (hours1 : ℕ)  -- Hours worked by the first mechanic
  (hours2 : ℕ)  -- Hours worked by the second mechanic
  (rate : ℕ)    -- Combined hourly rate in dollars
  (h1 : hours1 = 10)  -- First mechanic worked for 10 hours
  (h2 : hours2 = 5)   -- Second mechanic worked for 5 hours
  (h3 : rate = 160)   -- Combined hourly rate is $160
  : (hours1 + hours2) * rate = 2400 := by
  sorry


end NUMINAMATH_CALUDE_mechanics_total_charge_l1113_111313


namespace NUMINAMATH_CALUDE_difference_in_sums_l1113_111394

def star_list : List Nat := List.range 50 |>.map (· + 1)

def replace_three_with_two (n : Nat) : Nat :=
  let s := toString n
  (s.replace "3" "2").toNat!

def emilio_list : List Nat :=
  star_list.map replace_three_with_two

theorem difference_in_sums : 
  star_list.sum - emilio_list.sum = 105 := by
  sorry

end NUMINAMATH_CALUDE_difference_in_sums_l1113_111394


namespace NUMINAMATH_CALUDE_inequality_proof_l1113_111359

open Real BigOperators Finset

theorem inequality_proof (n : ℕ) (x : Fin n → ℝ) (σ : Equiv.Perm (Fin n)) 
  (h : ∀ i, 0 < x i ∧ x i < 1) : 
  ∑ i, (1 / (1 - x i)) ≥ 
  (1 + (1 / n) * ∑ i, x i) * ∑ i, (1 / (1 - x i * x (σ i))) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1113_111359


namespace NUMINAMATH_CALUDE_triangle_side_product_greater_than_circle_diameters_l1113_111331

theorem triangle_side_product_greater_than_circle_diameters 
  (a b c r R : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ R > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_inradius : r = (a + b - c) * (b + c - a) * (c + a - b) / (4 * (a + b + c)))
  (h_circumradius : R = a * b * c / (4 * (a + b - c) * (b + c - a) * (c + a - b))) :
  a * b > 4 * r * R :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_product_greater_than_circle_diameters_l1113_111331


namespace NUMINAMATH_CALUDE_gcf_seven_eight_factorial_l1113_111350

theorem gcf_seven_eight_factorial :
  Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcf_seven_eight_factorial_l1113_111350


namespace NUMINAMATH_CALUDE_multiples_of_6_factors_of_72_l1113_111348

def is_multiple_of_6 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k

def is_factor_of_72 (n : ℕ) : Prop := 72 % n = 0

def solution_set : Set ℕ := {6, 12, 18, 24, 36, 72}

theorem multiples_of_6_factors_of_72 :
  ∀ n : ℕ, (is_multiple_of_6 n ∧ is_factor_of_72 n) ↔ n ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_multiples_of_6_factors_of_72_l1113_111348


namespace NUMINAMATH_CALUDE_crayons_left_l1113_111349

theorem crayons_left (initial_crayons lost_crayons : ℕ) 
  (h1 : initial_crayons = 253)
  (h2 : lost_crayons = 70) :
  initial_crayons - lost_crayons = 183 := by
sorry

end NUMINAMATH_CALUDE_crayons_left_l1113_111349


namespace NUMINAMATH_CALUDE_football_season_length_l1113_111380

/-- The number of months in a football season -/
def season_length (total_games : ℕ) (games_per_month : ℕ) : ℕ :=
  total_games / games_per_month

/-- Proof that the football season lasts 17 months -/
theorem football_season_length : season_length 323 19 = 17 := by
  sorry

end NUMINAMATH_CALUDE_football_season_length_l1113_111380


namespace NUMINAMATH_CALUDE_happy_children_count_l1113_111383

theorem happy_children_count (total_children : ℕ) 
                              (sad_children : ℕ) 
                              (neutral_children : ℕ) 
                              (total_boys : ℕ) 
                              (total_girls : ℕ) 
                              (happy_boys : ℕ) 
                              (sad_girls : ℕ) :
  total_children = 60 →
  sad_children = 10 →
  neutral_children = 20 →
  total_boys = 18 →
  total_girls = 42 →
  happy_boys = 6 →
  sad_girls = 4 →
  ∃ (happy_children : ℕ), 
    happy_children = 30 ∧
    happy_children + sad_children + neutral_children = total_children ∧
    happy_boys + (sad_children - sad_girls) + (neutral_children - (neutral_children - (total_boys - happy_boys - (sad_children - sad_girls)))) = total_boys ∧
    (happy_children - happy_boys) + sad_girls + (neutral_children - (total_boys - happy_boys - (sad_children - sad_girls))) = total_girls :=
by
  sorry


end NUMINAMATH_CALUDE_happy_children_count_l1113_111383


namespace NUMINAMATH_CALUDE_jasmine_buys_six_bags_l1113_111370

/-- The number of bags of chips Jasmine buys -/
def bags_of_chips : ℕ := sorry

/-- The weight of one bag of chips in ounces -/
def chips_weight : ℕ := 20

/-- The weight of one tin of cookies in ounces -/
def cookies_weight : ℕ := 9

/-- The total weight Jasmine carries in ounces -/
def total_weight : ℕ := 21 * 16

theorem jasmine_buys_six_bags :
  bags_of_chips = 6 ∧
  chips_weight * bags_of_chips + cookies_weight * (4 * bags_of_chips) = total_weight :=
by sorry

end NUMINAMATH_CALUDE_jasmine_buys_six_bags_l1113_111370


namespace NUMINAMATH_CALUDE_specific_child_group_size_l1113_111314

/-- Represents a group of children with specific age characteristics -/
structure ChildGroup where
  sum_of_ages : ℕ
  age_difference : ℕ
  eldest_age : ℕ

/-- Calculates the number of children in a ChildGroup -/
def number_of_children (group : ChildGroup) : ℕ :=
  sorry

/-- Theorem stating that for a specific ChildGroup, the number of children is 10 -/
theorem specific_child_group_size :
  let group : ChildGroup := {
    sum_of_ages := 50,
    age_difference := 2,
    eldest_age := 14
  }
  number_of_children group = 10 := by
  sorry

end NUMINAMATH_CALUDE_specific_child_group_size_l1113_111314


namespace NUMINAMATH_CALUDE_equal_variance_sequence_properties_l1113_111363

/-- Definition of an equal variance sequence -/
def is_equal_variance_sequence (a : ℕ+ → ℝ) (p : ℝ) :=
  ∀ n : ℕ+, a n ^ 2 - a (n + 1) ^ 2 = p

theorem equal_variance_sequence_properties
  (a : ℕ+ → ℝ) (p : ℝ) (h : is_equal_variance_sequence a p) :
  (∀ n : ℕ+, ∃ d : ℝ, a (n + 1) ^ 2 - a n ^ 2 = d) ∧
  is_equal_variance_sequence (fun n ↦ (-1) ^ (n : ℕ)) 0 ∧
  (∀ k : ℕ+, is_equal_variance_sequence (fun n ↦ a (k * n)) (k * p)) :=
by sorry

end NUMINAMATH_CALUDE_equal_variance_sequence_properties_l1113_111363


namespace NUMINAMATH_CALUDE_remaining_average_l1113_111309

theorem remaining_average (total : ℝ) (group1 : ℝ) (group2 : ℝ) :
  total = 6 * 2.8 ∧ group1 = 2 * 2.4 ∧ group2 = 2 * 2.3 →
  (total - group1 - group2) / 2 = 3.7 := by
sorry

end NUMINAMATH_CALUDE_remaining_average_l1113_111309


namespace NUMINAMATH_CALUDE_train_crossing_time_l1113_111395

/-- A train crosses a platform in a certain time -/
theorem train_crossing_time 
  (train_speed : ℝ) 
  (pole_crossing_time : ℝ) 
  (platform_crossing_time : ℝ) : 
  train_speed = 36 → 
  pole_crossing_time = 12 → 
  platform_crossing_time = 49.996960243180546 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1113_111395


namespace NUMINAMATH_CALUDE_transformation_maps_points_l1113_111354

/-- Represents a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Scales a point by a factor about the origin -/
def scale (p : Point) (factor : ℝ) : Point :=
  { x := p.x * factor, y := p.y * factor }

/-- Reflects a point across the x-axis -/
def reflectX (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- Applies scaling followed by reflection across x-axis -/
def scaleAndReflect (p : Point) (factor : ℝ) : Point :=
  reflectX (scale p factor)

theorem transformation_maps_points :
  let C : Point := { x := -5, y := 2 }
  let D : Point := { x := 0, y := 3 }
  let C' : Point := { x := 10, y := -4 }
  let D' : Point := { x := 0, y := -6 }
  (scaleAndReflect C 2 = C') ∧ (scaleAndReflect D 2 = D') := by
  sorry

end NUMINAMATH_CALUDE_transformation_maps_points_l1113_111354


namespace NUMINAMATH_CALUDE_monday_pages_proof_l1113_111311

def total_pages : ℕ := 158
def tuesday_pages : ℕ := 38
def wednesday_pages : ℕ := 61
def thursday_pages : ℕ := 12
def friday_pages : ℕ := 2 * thursday_pages

theorem monday_pages_proof :
  total_pages - (tuesday_pages + wednesday_pages + thursday_pages + friday_pages) = 23 := by
  sorry

end NUMINAMATH_CALUDE_monday_pages_proof_l1113_111311


namespace NUMINAMATH_CALUDE_remainder_after_adding_2025_l1113_111322

theorem remainder_after_adding_2025 (m : ℤ) : 
  m % 9 = 4 → (m + 2025) % 9 = 4 := by
sorry

end NUMINAMATH_CALUDE_remainder_after_adding_2025_l1113_111322


namespace NUMINAMATH_CALUDE_car_owners_without_motorcycle_l1113_111399

theorem car_owners_without_motorcycle (total_adults : ℕ) (car_owners : ℕ) (motorcycle_owners : ℕ) (no_vehicle_owners : ℕ) 
  (h1 : total_adults = 560)
  (h2 : car_owners = 520)
  (h3 : motorcycle_owners = 80)
  (h4 : no_vehicle_owners = 10) :
  car_owners - (total_adults - no_vehicle_owners - (car_owners + motorcycle_owners - (total_adults - no_vehicle_owners))) = 470 := by
  sorry

end NUMINAMATH_CALUDE_car_owners_without_motorcycle_l1113_111399


namespace NUMINAMATH_CALUDE_sum_of_cubes_equation_l1113_111305

theorem sum_of_cubes_equation (x y : ℝ) :
  x^3 + 21*x*y + y^3 = 343 → x + y = 7 ∨ x + y = -14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_equation_l1113_111305


namespace NUMINAMATH_CALUDE_min_jellybeans_correct_l1113_111386

/-- The smallest number of jellybeans Alex should buy -/
def min_jellybeans : ℕ := 134

/-- Theorem stating that min_jellybeans is the smallest number satisfying the conditions -/
theorem min_jellybeans_correct :
  (min_jellybeans ≥ 120) ∧
  (min_jellybeans % 15 = 14) ∧
  (∀ n : ℕ, n ≥ 120 → n % 15 = 14 → n ≥ min_jellybeans) :=
by sorry

end NUMINAMATH_CALUDE_min_jellybeans_correct_l1113_111386


namespace NUMINAMATH_CALUDE_certain_number_problem_l1113_111347

theorem certain_number_problem : ∃ x : ℝ, x * 12 = 0.60 * 900 ∧ x = 45 := by sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1113_111347


namespace NUMINAMATH_CALUDE_jake_second_test_difference_l1113_111328

def jake_test_scores (test1 test2 test3 test4 : ℕ) : Prop :=
  test1 = 80 ∧ 
  test3 = 65 ∧ 
  test3 = test4 ∧ 
  (test1 + test2 + test3 + test4) / 4 = 75

theorem jake_second_test_difference :
  ∀ test1 test2 test3 test4 : ℕ,
    jake_test_scores test1 test2 test3 test4 →
    test2 - test1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_jake_second_test_difference_l1113_111328
