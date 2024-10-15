import Mathlib

namespace NUMINAMATH_CALUDE_toms_robot_collection_l2928_292863

/-- Represents the number of robots of each type for a person -/
structure RobotCollection where
  animal : ℕ
  humanoid : ℕ
  vehicle : ℕ

/-- Given the conditions of the problem, prove that Tom's robot collection matches the expected values -/
theorem toms_robot_collection (michael : RobotCollection) (tom : RobotCollection) : 
  michael.animal = 8 ∧ 
  michael.humanoid = 12 ∧ 
  michael.vehicle = 20 ∧
  tom.animal = 2 * michael.animal ∧
  tom.humanoid = (3 : ℕ) / 2 * michael.humanoid ∧
  michael.vehicle = (5 : ℕ) / 4 * tom.vehicle →
  tom.animal = 16 ∧ tom.humanoid = 18 ∧ tom.vehicle = 16 := by
  sorry

end NUMINAMATH_CALUDE_toms_robot_collection_l2928_292863


namespace NUMINAMATH_CALUDE_existence_of_set_B_l2928_292811

theorem existence_of_set_B : ∃ (a : ℝ), 
  let A : Set ℝ := {1, 3, a^2 + 3*a - 4}
  let B : Set ℝ := {0, 6, a^2 + 4*a - 2, a + 3}
  (A ∩ B = {3}) ∧ (a = 0) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_set_B_l2928_292811


namespace NUMINAMATH_CALUDE_passing_methods_after_six_passes_l2928_292850

/-- The number of ways the ball can be passed back to player A after n passes -/
def passing_methods (n : ℕ) : ℕ :=
  if n < 2 then 0
  else if n = 2 then 2
  else 2^(n-1) - passing_methods (n-1)

/-- The theorem stating that there are 22 different passing methods after 6 passes -/
theorem passing_methods_after_six_passes :
  passing_methods 6 = 22 := by
  sorry

end NUMINAMATH_CALUDE_passing_methods_after_six_passes_l2928_292850


namespace NUMINAMATH_CALUDE_sock_selection_theorem_l2928_292864

/-- Represents the number of socks of a given color -/
def num_socks : Fin 3 → Nat
  | 0 => 5  -- white
  | 1 => 5  -- brown
  | 2 => 3  -- blue

/-- Calculates the number of socks in odd positions for a given color -/
def odd_positions (color : Fin 3) : Nat :=
  (num_socks color + 1) / 2

/-- Calculates the number of socks in even positions for a given color -/
def even_positions (color : Fin 3) : Nat :=
  num_socks color / 2

/-- Calculates the number of ways to select a pair of socks of different colors from either odd or even positions -/
def select_pair_ways : Nat :=
  let white := 0
  let brown := 1
  let blue := 2
  (odd_positions white * odd_positions brown + even_positions white * even_positions brown) +
  (odd_positions brown * odd_positions blue + even_positions brown * even_positions blue) +
  (odd_positions white * odd_positions blue + even_positions white * even_positions blue)

/-- The main theorem stating that the number of ways to select a pair of socks is 29 -/
theorem sock_selection_theorem : select_pair_ways = 29 := by
  sorry

end NUMINAMATH_CALUDE_sock_selection_theorem_l2928_292864


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2928_292873

theorem partial_fraction_decomposition :
  ∃! (A B C : ℚ),
    ∀ (x : ℚ), x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 →
      (x^2 - 5*x + 6) / ((x - 1)*(x - 4)*(x - 6)) =
      A / (x - 1) + B / (x - 4) + C / (x - 6) ∧
      A = 2/15 ∧ B = -1/3 ∧ C = 3/5 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2928_292873


namespace NUMINAMATH_CALUDE_infinite_solutions_condition_l2928_292848

theorem infinite_solutions_condition (a : ℝ) :
  (∀ x : ℝ, 3 * (2 * x - a) = 2 * (3 * x + 12)) ↔ a = -8 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_condition_l2928_292848


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2928_292824

theorem trigonometric_equation_solution (x : ℝ) :
  (Real.sin x)^3 + 6 * (Real.cos x)^3 + (1 / Real.sqrt 2) * Real.sin (2 * x) * Real.sin (x + π / 4) = 0 →
  ∃ n : ℤ, x = -Real.arctan 2 + n * π :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2928_292824


namespace NUMINAMATH_CALUDE_consecutive_product_and_fourth_power_properties_l2928_292803

theorem consecutive_product_and_fourth_power_properties (c d m n : ℕ) : 
  (c * (c + 1) ≠ d * (d + 2)) ∧ 
  (m^4 + (m + 1)^4 ≠ n^2 + (n + 1)^2) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_and_fourth_power_properties_l2928_292803


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2928_292874

theorem partial_fraction_decomposition :
  let f (x : ℚ) := (7 * x - 4) / (x^2 - 9*x - 18)
  let g (x : ℚ) := 59 / (11 * (x - 9)) + 18 / (11 * (x + 2))
  ∀ x, x ≠ 9 ∧ x ≠ -2 → f x = g x :=
by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2928_292874


namespace NUMINAMATH_CALUDE_mod_eleven_problem_l2928_292846

theorem mod_eleven_problem : ∃ n : ℕ, 0 ≤ n ∧ n < 11 ∧ 1234 % 11 = n := by
  sorry

end NUMINAMATH_CALUDE_mod_eleven_problem_l2928_292846


namespace NUMINAMATH_CALUDE_work_completion_time_l2928_292885

/-- Given that A can do a work in 6 days and A and B together can finish the work in 4 days,
    prove that B can do the work alone in 12 days. -/
theorem work_completion_time (a b : ℝ) 
  (ha : a = 6)  -- A can do the work in 6 days
  (hab : 1 / a + 1 / b = 1 / 4)  -- A and B together can finish the work in 4 days
  : b = 12 := by  -- B can do the work alone in 12 days
sorry

end NUMINAMATH_CALUDE_work_completion_time_l2928_292885


namespace NUMINAMATH_CALUDE_symmetry_of_lines_l2928_292827

/-- Given two lines in a 2D plane represented by their equations,
    this function returns true if they are symmetric with respect to the line x+y=0 -/
def are_symmetric_lines (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, l1 x y ↔ l2 (-y) (-x)

/-- The equation of the original line -/
def original_line (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0

/-- The equation of the supposedly symmetric line -/
def symmetric_line (x y : ℝ) : Prop := 4 * x - 3 * y - 5 = 0

/-- Theorem stating that the symmetric_line is indeed symmetric to the original_line
    with respect to the line x+y=0 -/
theorem symmetry_of_lines : are_symmetric_lines original_line symmetric_line :=
sorry

end NUMINAMATH_CALUDE_symmetry_of_lines_l2928_292827


namespace NUMINAMATH_CALUDE_arithmetic_sequence_reciprocal_S_general_term_formula_l2928_292856

def sequence_a (n : ℕ) : ℚ := sorry

def sum_S (n : ℕ) : ℚ := sorry

axiom a_1 : sequence_a 1 = 3

axiom relation_a_S (n : ℕ) : n ≥ 2 → 2 * sequence_a n = sum_S n * sum_S (n - 1)

theorem arithmetic_sequence_reciprocal_S :
  ∀ n : ℕ, n ≥ 2 → (1 / sum_S n - 1 / sum_S (n - 1) = -1 / 2) :=
sorry

theorem general_term_formula :
  ∀ n : ℕ, n ≥ 2 →
    sequence_a n = 18 / ((8 - 3 * n) * (5 - 3 * n)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_reciprocal_S_general_term_formula_l2928_292856


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l2928_292849

theorem simultaneous_equations_solution (k : ℝ) :
  (k ≠ 1) ↔ (∃ x y : ℝ, (y = k * x + 2) ∧ (y = (3 * k - 2) * x + 5)) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l2928_292849


namespace NUMINAMATH_CALUDE_combined_savings_equal_individual_savings_problem_specific_savings_l2928_292820

/-- Represents the store's window offer -/
structure WindowOffer where
  normalPrice : ℕ
  freeWindowsPer : ℕ

/-- Calculates the cost of purchasing windows under the offer -/
def costUnderOffer (offer : WindowOffer) (windowsNeeded : ℕ) : ℕ :=
  let paidWindows := windowsNeeded - (windowsNeeded / (offer.freeWindowsPer + 1))
  paidWindows * offer.normalPrice

/-- Calculates the savings for a given number of windows -/
def savings (offer : WindowOffer) (windowsNeeded : ℕ) : ℕ :=
  windowsNeeded * offer.normalPrice - costUnderOffer offer windowsNeeded

/-- Theorem: The combined savings equal the sum of individual savings -/
theorem combined_savings_equal_individual_savings 
  (offer : WindowOffer) 
  (daveWindows : ℕ) 
  (dougWindows : ℕ) : 
  savings offer (daveWindows + dougWindows) = 
    savings offer daveWindows + savings offer dougWindows :=
by sorry

/-- The specific offer in the problem -/
def storeOffer : WindowOffer := { normalPrice := 100, freeWindowsPer := 3 }

/-- Theorem: For Dave (9 windows) and Doug (6 windows), 
    the combined savings equal the sum of their individual savings -/
theorem problem_specific_savings : 
  savings storeOffer (9 + 6) = savings storeOffer 9 + savings storeOffer 6 :=
by sorry

end NUMINAMATH_CALUDE_combined_savings_equal_individual_savings_problem_specific_savings_l2928_292820


namespace NUMINAMATH_CALUDE_cats_remaining_after_sale_l2928_292877

/-- The number of cats remaining after a sale at a pet store. -/
theorem cats_remaining_after_sale (siamese : ℕ) (house : ℕ) (sold : ℕ) : 
  siamese = 19 → house = 45 → sold = 56 → siamese + house - sold = 8 := by
  sorry

end NUMINAMATH_CALUDE_cats_remaining_after_sale_l2928_292877


namespace NUMINAMATH_CALUDE_sum_of_20th_and_30th_triangular_l2928_292867

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: The sum of the 20th and 30th triangular numbers is 675 -/
theorem sum_of_20th_and_30th_triangular : triangular_number 20 + triangular_number 30 = 675 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_20th_and_30th_triangular_l2928_292867


namespace NUMINAMATH_CALUDE_sphere_diameter_from_cylinder_l2928_292806

noncomputable def cylinder_volume (d h : ℝ) : ℝ := Real.pi * (d / 2)^2 * h

noncomputable def sphere_volume (d : ℝ) : ℝ := (4 / 3) * Real.pi * (d / 2)^3

theorem sphere_diameter_from_cylinder (cylinder_diameter cylinder_height : ℝ) :
  let total_volume := cylinder_volume cylinder_diameter cylinder_height
  let sphere_count := 9
  let individual_sphere_volume := total_volume / sphere_count
  let sphere_diameter := (6 * individual_sphere_volume / Real.pi)^(1/3)
  cylinder_diameter = 16 ∧ cylinder_height = 12 →
  sphere_diameter = 8 := by
  sorry

#check sphere_diameter_from_cylinder

end NUMINAMATH_CALUDE_sphere_diameter_from_cylinder_l2928_292806


namespace NUMINAMATH_CALUDE_triple_hash_90_l2928_292881

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.3 * N + 2

-- State the theorem
theorem triple_hash_90 : hash (hash (hash 90)) = 5.21 := by
  sorry

end NUMINAMATH_CALUDE_triple_hash_90_l2928_292881


namespace NUMINAMATH_CALUDE_investment_return_rate_l2928_292810

theorem investment_return_rate 
  (total_investment : ℝ) 
  (total_interest : ℝ) 
  (known_rate : ℝ) 
  (known_investment : ℝ) 
  (h1 : total_investment = 33000)
  (h2 : total_interest = 970)
  (h3 : known_rate = 0.0225)
  (h4 : known_investment = 13000)
  : ∃ r : ℝ, 
    r * known_investment + known_rate * (total_investment - known_investment) = total_interest ∧ 
    r = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_investment_return_rate_l2928_292810


namespace NUMINAMATH_CALUDE_oil_purchase_calculation_l2928_292842

theorem oil_purchase_calculation (tank_capacity : ℕ) (tanks_needed : ℕ) (total_oil : ℕ) : 
  tank_capacity = 32 → tanks_needed = 23 → total_oil = tank_capacity * tanks_needed → total_oil = 736 :=
by sorry

end NUMINAMATH_CALUDE_oil_purchase_calculation_l2928_292842


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l2928_292821

theorem geometric_sequence_middle_term (χ : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ -1 * r = χ ∧ χ * r = -4) → χ = 2 ∨ χ = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l2928_292821


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l2928_292834

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_ninth_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 5 + a 7 = 16) 
  (h_third : a 3 = 4) : 
  a 9 = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l2928_292834


namespace NUMINAMATH_CALUDE_no_brownies_left_l2928_292876

/-- Represents the number of brownies left after consumption --/
def brownies_left (total : ℚ) (tina_lunch : ℚ) (tina_dinner : ℚ) (husband : ℚ) (guests : ℚ) (daughter : ℚ) : ℚ :=
  total - (5 * (tina_lunch + tina_dinner) + 5 * husband + 2 * guests + 3 * daughter)

/-- Theorem stating that no brownies are left after consumption --/
theorem no_brownies_left : 
  brownies_left 24 1.5 0.5 0.75 2.5 2 = 0 := by
  sorry

#eval brownies_left 24 1.5 0.5 0.75 2.5 2

end NUMINAMATH_CALUDE_no_brownies_left_l2928_292876


namespace NUMINAMATH_CALUDE_parabola_y_intercept_l2928_292853

/-- A parabola passing through two given points has a specific y-intercept -/
theorem parabola_y_intercept (a b : ℝ) : 
  (6 = 2^2 + 2*a + b) ∧ (-14 = (-2)^2 + (-2)*a + b) → b = -8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_y_intercept_l2928_292853


namespace NUMINAMATH_CALUDE_problem_solid_surface_area_l2928_292814

/-- Represents a solid formed by unit cubes -/
structure CubeSolid where
  base_length : ℕ
  base_width : ℕ
  top_length : ℕ
  top_width : ℕ
  base_height : ℕ := 1
  top_height : ℕ := 1

/-- Calculates the surface area of the CubeSolid -/
def surface_area (solid : CubeSolid) : ℕ :=
  let base_area := solid.base_length * solid.base_width
  let top_area := solid.top_length * solid.top_width
  let exposed_base := 2 * base_area
  let exposed_sides := 2 * (solid.base_length * solid.base_height + solid.base_width * solid.base_height)
  let exposed_top := base_area - top_area + top_area
  let exposed_top_sides := 2 * (solid.top_length * solid.top_height + solid.top_width * solid.top_height)
  exposed_base + exposed_sides + exposed_top + exposed_top_sides

/-- The specific solid described in the problem -/
def problem_solid : CubeSolid := {
  base_length := 4
  base_width := 2
  top_length := 2
  top_width := 2
}

theorem problem_solid_surface_area :
  surface_area problem_solid = 36 := by
  sorry

end NUMINAMATH_CALUDE_problem_solid_surface_area_l2928_292814


namespace NUMINAMATH_CALUDE_sam_study_time_l2928_292823

theorem sam_study_time (total_hours : ℕ) (science_minutes : ℕ) (literature_minutes : ℕ) 
  (h1 : total_hours = 3)
  (h2 : science_minutes = 60)
  (h3 : literature_minutes = 40) :
  total_hours * 60 - (science_minutes + literature_minutes) = 80 :=
by sorry

end NUMINAMATH_CALUDE_sam_study_time_l2928_292823


namespace NUMINAMATH_CALUDE_quadratic_solutions_fractional_solution_l2928_292831

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 + 3*x + 1 = 0

-- Define the fractional equation
def fractional_equation (x : ℝ) : Prop := (4*x)/(x-2) - 1 = 3/(2-x)

-- Theorem for the quadratic equation solutions
theorem quadratic_solutions :
  ∃ x1 x2 : ℝ, 
    quadratic_equation x1 ∧ 
    quadratic_equation x2 ∧ 
    x1 = (-3 + Real.sqrt 5) / 2 ∧ 
    x2 = (-3 - Real.sqrt 5) / 2 :=
sorry

-- Theorem for the fractional equation solution
theorem fractional_solution :
  ∃ x : ℝ, fractional_equation x ∧ x = -5/3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_solutions_fractional_solution_l2928_292831


namespace NUMINAMATH_CALUDE_brick_wall_pattern_l2928_292845

/-- Represents a brick wall with a given number of rows and bricks -/
structure BrickWall where
  rows : ℕ
  total_bricks : ℕ
  bottom_row_bricks : ℕ

/-- Calculates the number of bricks in a given row -/
def bricks_in_row (wall : BrickWall) (row : ℕ) : ℕ :=
  wall.bottom_row_bricks - (row - 1)

theorem brick_wall_pattern (wall : BrickWall) 
  (h1 : wall.rows = 5)
  (h2 : wall.total_bricks = 50)
  (h3 : wall.bottom_row_bricks = 8) :
  ∀ row : ℕ, 1 < row → row ≤ wall.rows → 
    bricks_in_row wall row = bricks_in_row wall (row - 1) - 1 :=
by sorry

end NUMINAMATH_CALUDE_brick_wall_pattern_l2928_292845


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2928_292890

/-- Given a geometric sequence {a_n} with a_1 = 1/2 and a_4 = -4, prove that the common ratio q is -2. -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (h_geometric : ∀ n, a (n + 1) = a n * q) 
  (h_a1 : a 1 = 1/2) 
  (h_a4 : a 4 = -4) 
  : q = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2928_292890


namespace NUMINAMATH_CALUDE_wall_height_calculation_l2928_292868

/-- Given a brick and wall with specified dimensions, prove the height of the wall --/
theorem wall_height_calculation (brick_length brick_width brick_height : Real)
  (wall_length wall_width : Real) (num_bricks : Nat) :
  brick_length = 0.20 →
  brick_width = 0.10 →
  brick_height = 0.075 →
  wall_length = 25 →
  wall_width = 0.75 →
  num_bricks = 25000 →
  ∃ (wall_height : Real),
    wall_height = 2 ∧
    num_bricks * (brick_length * brick_width * brick_height) = wall_length * wall_width * wall_height :=
by sorry

end NUMINAMATH_CALUDE_wall_height_calculation_l2928_292868


namespace NUMINAMATH_CALUDE_min_value_problem_l2928_292826

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b - a - 2 * b = 0) :
  ∃ (min : ℝ), min = 7 ∧ 
  (∀ (x y : ℝ), x > 0 → y > 0 → x * y - x - 2 * y = 0 → 
    x^2 / 4 - 2 / x + y^2 - 1 / y ≥ min) ∧
  (a^2 / 4 - 2 / a + b^2 - 1 / b = min) :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l2928_292826


namespace NUMINAMATH_CALUDE_alloy_ratio_theorem_l2928_292841

/-- Represents an alloy with zinc and copper -/
structure Alloy where
  zinc : ℚ
  copper : ℚ

/-- The first alloy with zinc:copper ratio of 1:2 -/
def alloy1 : Alloy := { zinc := 1, copper := 2 }

/-- The second alloy with zinc:copper ratio of 2:3 -/
def alloy2 : Alloy := { zinc := 2, copper := 3 }

/-- The desired third alloy with zinc:copper ratio of 17:27 -/
def alloy3 : Alloy := { zinc := 17, copper := 27 }

/-- Theorem stating the ratio of alloys needed to create the third alloy -/
theorem alloy_ratio_theorem :
  ∃ (x y : ℚ),
    x > 0 ∧ y > 0 ∧
    (x * alloy1.zinc + y * alloy2.zinc) / (x * alloy1.copper + y * alloy2.copper) = alloy3.zinc / alloy3.copper ∧
    x / y = 9 / 35 := by
  sorry


end NUMINAMATH_CALUDE_alloy_ratio_theorem_l2928_292841


namespace NUMINAMATH_CALUDE_deductive_reasoning_example_l2928_292819

-- Define the property of conducting electricity
def conducts_electricity (x : Type) : Prop := sorry

-- Define the concept of metal
def is_metal (x : Type) : Prop := sorry

-- Define deductive reasoning
def is_deductive_reasoning (premise1 premise2 conclusion : Prop) : Prop := sorry

-- Define the specific metals
def gold : Type := sorry
def silver : Type := sorry
def copper : Type := sorry

-- Theorem statement
theorem deductive_reasoning_example :
  let premise1 : Prop := ∀ x, is_metal x → conducts_electricity x
  let premise2 : Prop := is_metal gold ∧ is_metal silver ∧ is_metal copper
  let conclusion : Prop := conducts_electricity gold ∧ conducts_electricity silver ∧ conducts_electricity copper
  is_deductive_reasoning premise1 premise2 conclusion := by sorry

end NUMINAMATH_CALUDE_deductive_reasoning_example_l2928_292819


namespace NUMINAMATH_CALUDE_ticket_identification_operations_l2928_292847

/-- The maximum ticket number --/
def max_ticket : Nat := 30

/-- The number of operations needed to identify all ticket numbers --/
def num_operations : Nat := 5

/-- Function to calculate the number of binary digits needed to represent a number --/
def binary_digits (n : Nat) : Nat :=
  if n = 0 then 1 else Nat.log2 n + 1

theorem ticket_identification_operations :
  binary_digits max_ticket = num_operations :=
by sorry

end NUMINAMATH_CALUDE_ticket_identification_operations_l2928_292847


namespace NUMINAMATH_CALUDE_triangle_angle_and_parameter_l2928_292851

/-- Given a triangle ABC where tan A and tan B are real roots of a quadratic equation,
    prove that angle C is 60° and find the value of p. -/
theorem triangle_angle_and_parameter
  (A B C : ℝ) (p : ℝ) (AB AC : ℝ)
  (h_triangle : A + B + C = Real.pi)
  (h_roots : ∃ (x y : ℝ), x^2 + Real.sqrt 3 * p * x - p + 1 = 0 ∧
                          y^2 + Real.sqrt 3 * p * y - p + 1 = 0 ∧
                          x = Real.tan A ∧ y = Real.tan B)
  (h_AB : AB = 3)
  (h_AC : AC = Real.sqrt 6) :
  C = Real.pi / 3 ∧ p = -1 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_and_parameter_l2928_292851


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_eighty_l2928_292812

theorem thirty_percent_less_than_eighty (x : ℚ) : x + x / 2 = 80 - 80 * 3 / 10 → x = 112 / 3 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_eighty_l2928_292812


namespace NUMINAMATH_CALUDE_grade_assignment_count_l2928_292896

theorem grade_assignment_count (num_students : ℕ) (num_grades : ℕ) :
  num_students = 12 → num_grades = 4 →
  (num_grades : ℕ) ^ num_students = 16777216 :=
by
  sorry

end NUMINAMATH_CALUDE_grade_assignment_count_l2928_292896


namespace NUMINAMATH_CALUDE_double_inequality_solution_l2928_292891

theorem double_inequality_solution (x : ℝ) : 
  (4 * x + 2 > (x - 1)^2 ∧ (x - 1)^2 > 3 * x + 6) ↔ 
  (x > 3 + 2 * Real.sqrt 10 ∧ x < (5 + 3 * Real.sqrt 5) / 2) :=
sorry

end NUMINAMATH_CALUDE_double_inequality_solution_l2928_292891


namespace NUMINAMATH_CALUDE_sequence_sum_l2928_292825

theorem sequence_sum (P Q R S T U V : ℝ) : 
  R = 7 ∧
  P + Q + R = 36 ∧
  Q + R + S = 36 ∧
  R + S + T = 36 ∧
  S + T + U = 36 ∧
  T + U + V = 36 →
  P + V = 29 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l2928_292825


namespace NUMINAMATH_CALUDE_waiter_customers_waiter_customers_proof_l2928_292802

theorem waiter_customers : ℕ → Prop :=
  fun initial_customers =>
    initial_customers - 14 + 36 = 41 →
    initial_customers = 19

-- The proof of the theorem
theorem waiter_customers_proof : ∃ x : ℕ, waiter_customers x :=
  sorry

end NUMINAMATH_CALUDE_waiter_customers_waiter_customers_proof_l2928_292802


namespace NUMINAMATH_CALUDE_inequality_proof_l2928_292836

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (2 * x^2) / (y + z) + (2 * y^2) / (z + x) + (2 * z^2) / (x + y) ≥ x + y + z := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2928_292836


namespace NUMINAMATH_CALUDE_algebra_test_male_students_l2928_292838

theorem algebra_test_male_students (M : ℕ) : 
  (90 * (M + 32) = 82 * M + 92 * 32) → M = 8 := by
sorry

end NUMINAMATH_CALUDE_algebra_test_male_students_l2928_292838


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l2928_292837

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l2928_292837


namespace NUMINAMATH_CALUDE_secret_spread_day_l2928_292828

/-- The number of people who know the secret on day n -/
def secret_spread (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

/-- The day when the secret is known by 6560 people -/
def target_day : ℕ := 8

theorem secret_spread_day : secret_spread target_day = 6560 := by
  sorry

#eval secret_spread target_day

end NUMINAMATH_CALUDE_secret_spread_day_l2928_292828


namespace NUMINAMATH_CALUDE_student_claim_incorrect_l2928_292880

theorem student_claim_incorrect :
  ¬ ∃ (m n : ℤ), 
    n > 0 ∧ 
    n ≤ 100 ∧ 
    ∃ (a : ℕ → ℕ), (m : ℚ) / n = 0.167 + ∑' i, (a i : ℚ) / 10^(i+3) :=
by sorry

end NUMINAMATH_CALUDE_student_claim_incorrect_l2928_292880


namespace NUMINAMATH_CALUDE_number_of_days_function_l2928_292883

/-- The "number of days function" for given day points and a point on its graph -/
theorem number_of_days_function (k : ℝ) :
  (∀ x : ℝ, ∃ y₁ y₂ : ℝ, y₁ = k * x + 4 ∧ y₂ = 2 * x) →
  (∃ y : ℝ → ℝ, y 2 = 3 ∧ ∀ x : ℝ, y x = (k * x + 4) - (2 * x)) →
  (∃ y : ℝ → ℝ, ∀ x : ℝ, y x = -1/2 * x + 4) :=
by sorry

end NUMINAMATH_CALUDE_number_of_days_function_l2928_292883


namespace NUMINAMATH_CALUDE_beautiful_point_coordinates_l2928_292816

/-- A point (x, y) is "beautiful" if x + y = x * y -/
def is_beautiful_point (x y : ℝ) : Prop := x + y = x * y

/-- The distance of a point (x, y) from the y-axis is |x| -/
def distance_from_y_axis (x : ℝ) : ℝ := |x|

theorem beautiful_point_coordinates :
  ∀ x y : ℝ, is_beautiful_point x y → distance_from_y_axis x = 2 →
  ((x = 2 ∧ y = 2) ∨ (x = -2 ∧ y = 2/3)) :=
by sorry

end NUMINAMATH_CALUDE_beautiful_point_coordinates_l2928_292816


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_N_l2928_292878

def M : Set ℝ := {x | |x| ≥ 3}

def N : Set ℝ := {y | ∃ x ∈ M, y = x^2}

theorem M_intersect_N_eq_N : M ∩ N = N := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_N_l2928_292878


namespace NUMINAMATH_CALUDE_sqrt_sum_problem_l2928_292860

theorem sqrt_sum_problem (y : ℝ) (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) :
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_problem_l2928_292860


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2928_292840

theorem complex_number_in_first_quadrant : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (2 - I) / (1 - 3*I) = a + b*I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2928_292840


namespace NUMINAMATH_CALUDE_food_boxes_l2928_292822

theorem food_boxes (total_food : ℝ) (food_per_box : ℝ) (h1 : total_food = 777.5) (h2 : food_per_box = 2.25) :
  ⌊total_food / food_per_box⌋ = 345 := by
  sorry

end NUMINAMATH_CALUDE_food_boxes_l2928_292822


namespace NUMINAMATH_CALUDE_period_is_24_hours_period_in_hours_is_24_l2928_292898

/-- Represents the period in seconds --/
def period (birth_rate : ℚ) (death_rate : ℚ) (net_increase : ℕ) : ℚ :=
  net_increase / (birth_rate / 2 - death_rate / 2)

/-- Theorem stating that the period is 24 hours given the problem conditions --/
theorem period_is_24_hours :
  let birth_rate : ℚ := 10
  let death_rate : ℚ := 2
  let net_increase : ℕ := 345600
  period birth_rate death_rate net_increase = 86400 := by
  sorry

/-- Converts seconds to hours --/
def seconds_to_hours (seconds : ℚ) : ℚ :=
  seconds / 3600

/-- Theorem stating that 86400 seconds is equal to 24 hours --/
theorem period_in_hours_is_24 :
  seconds_to_hours 86400 = 24 := by
  sorry

end NUMINAMATH_CALUDE_period_is_24_hours_period_in_hours_is_24_l2928_292898


namespace NUMINAMATH_CALUDE_book_arrangements_count_l2928_292843

/-- The number of ways to arrange 8 different books (3 math, 3 foreign language, 2 literature)
    such that all math books are together and all foreign language books are together. -/
def book_arrangements : ℕ :=
  let total_books : ℕ := 8
  let math_books : ℕ := 3
  let foreign_books : ℕ := 3
  let literature_books : ℕ := 2
  sorry

/-- Theorem stating that the number of book arrangements is 864. -/
theorem book_arrangements_count : book_arrangements = 864 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangements_count_l2928_292843


namespace NUMINAMATH_CALUDE_square_circle_union_area_l2928_292817

theorem square_circle_union_area (s : Real) (r : Real) : 
  s = 12 → r = 12 → (s ^ 2 + π * r ^ 2 - s ^ 2) = 144 * π := by
  sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l2928_292817


namespace NUMINAMATH_CALUDE_area_of_curve_l2928_292879

/-- The curve defined by x^2 + y^2 = |x| + 2|y| -/
def curve (x y : ℝ) : Prop := x^2 + y^2 = |x| + 2 * |y|

/-- The area enclosed by the curve -/
noncomputable def enclosed_area : ℝ := sorry

theorem area_of_curve : enclosed_area = (5 * π) / 4 := by sorry

end NUMINAMATH_CALUDE_area_of_curve_l2928_292879


namespace NUMINAMATH_CALUDE_cos_sum_17th_roots_unity_l2928_292865

theorem cos_sum_17th_roots_unity :
  Real.cos (2 * Real.pi / 17) + Real.cos (8 * Real.pi / 17) + Real.cos (14 * Real.pi / 17) = (Real.sqrt 17 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_17th_roots_unity_l2928_292865


namespace NUMINAMATH_CALUDE_all_statements_false_l2928_292884

theorem all_statements_false : ∀ (a b c d : ℝ),
  (¬((a ≠ b ∧ c ≠ d) → (a + c ≠ b + d))) ∧
  (¬((a + c ≠ b + d) → (a ≠ b ∧ c ≠ d))) ∧
  (¬(a = b ∧ c = d ∧ a + c ≠ b + d)) ∧
  (¬((a + c = b + d) → (a = b ∨ c = d))) :=
by sorry

end NUMINAMATH_CALUDE_all_statements_false_l2928_292884


namespace NUMINAMATH_CALUDE_dog_human_age_difference_l2928_292844

/-- The ratio of dog years to human years -/
def dogYearRatio : ℕ := 7

/-- Calculates the age difference in dog years between a dog and a human of the same age in human years -/
def ageDifferenceInDogYears (humanAge : ℕ) : ℕ :=
  humanAge * dogYearRatio - humanAge

/-- Theorem stating that for a 3-year-old human and their 3-year-old dog, 
    the dog will be 18 years older in dog years -/
theorem dog_human_age_difference : ageDifferenceInDogYears 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_dog_human_age_difference_l2928_292844


namespace NUMINAMATH_CALUDE_stick_cutting_probability_l2928_292808

theorem stick_cutting_probability (stick_length : Real) (mark_position : Real) 
  (h1 : stick_length = 2)
  (h2 : mark_position = 0.6)
  (h3 : 0 < mark_position ∧ mark_position < stick_length) :
  let cut_range := stick_length - mark_position
  let valid_cut := min (stick_length / 4) cut_range
  (valid_cut / cut_range) = 5/14 := by
  sorry


end NUMINAMATH_CALUDE_stick_cutting_probability_l2928_292808


namespace NUMINAMATH_CALUDE_chord_length_l2928_292888

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def are_externally_tangent (c1 c2 : Circle) : Prop := sorry

def is_internally_tangent (c1 c2 : Circle) : Prop := sorry

def are_collinear (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

def is_common_external_tangent (c1 c2 c3 : Circle) (chord : ℝ × ℝ → ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem chord_length 
  (c1 c2 c3 : Circle)
  (chord : ℝ × ℝ → ℝ × ℝ)
  (h1 : are_externally_tangent c1 c2)
  (h2 : is_internally_tangent c1 c3)
  (h3 : is_internally_tangent c2 c3)
  (h4 : c1.radius = 3)
  (h5 : c2.radius = 9)
  (h6 : are_collinear c1.center c2.center c3.center)
  (h7 : is_common_external_tangent c1 c2 c3 chord) :
  ∃ (a b : ℝ × ℝ), chord a = b ∧ Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 18 :=
sorry

end NUMINAMATH_CALUDE_chord_length_l2928_292888


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l2928_292833

/-- A circle with radius 2, center on the positive x-axis, and tangent to the y-axis -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_x_axis : center.2 = 0
  positive_x : center.1 > 0
  radius_is_two : radius = 2
  tangent_to_y : center.1 = radius

/-- The equation of the circle is x² + y² - 4x = 0 -/
theorem tangent_circle_equation (c : TangentCircle) :
  ∀ x y : ℝ, (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 ↔ x^2 + y^2 - 4*x = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l2928_292833


namespace NUMINAMATH_CALUDE_polynomial_inequality_polynomial_inequality_equality_condition_l2928_292809

/-- A polynomial of degree 3 with roots in (0, 1) -/
structure PolynomialWithRootsInUnitInterval where
  b : ℝ
  c : ℝ
  root_property : ∃ (x₁ x₂ x₃ : ℝ), 0 < x₁ ∧ x₁ < 1 ∧
                                    0 < x₂ ∧ x₂ < 1 ∧
                                    0 < x₃ ∧ x₃ < 1 ∧
                                    x₁ + x₂ + x₃ = 2 ∧
                                    x₁ * x₂ + x₂ * x₃ + x₃ * x₁ = b ∧
                                    x₁ * x₂ * x₃ = -c

/-- The main theorem stating the inequality for polynomials with roots in (0, 1) -/
theorem polynomial_inequality (P : PolynomialWithRootsInUnitInterval) :
  8 * P.b + 9 * P.c ≤ 8 := by
  sorry

/-- Conditions for equality in the polynomial inequality -/
theorem polynomial_inequality_equality_condition (P : PolynomialWithRootsInUnitInterval) :
  (8 * P.b + 9 * P.c = 8) ↔ 
  (∃ (x : ℝ), x = 2/3 ∧ 
   ∃ (x₁ x₂ x₃ : ℝ), x₁ = x ∧ x₂ = x ∧ x₃ = x ∧
                     0 < x₁ ∧ x₁ < 1 ∧
                     0 < x₂ ∧ x₂ < 1 ∧
                     0 < x₃ ∧ x₃ < 1 ∧
                     x₁ + x₂ + x₃ = 2 ∧
                     x₁ * x₂ + x₂ * x₃ + x₃ * x₁ = P.b ∧
                     x₁ * x₂ * x₃ = -P.c) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_polynomial_inequality_equality_condition_l2928_292809


namespace NUMINAMATH_CALUDE_hex_to_binary_digits_l2928_292897

theorem hex_to_binary_digits : ∃ (n : ℕ), n = 20 ∧ 
  (∀ (m : ℕ), 2^m ≤ (11 * 16^4 + 1 * 16^3 + 2 * 16^2 + 3 * 16^1 + 4 * 16^0) → m ≤ n) ∧
  (2^n > 11 * 16^4 + 1 * 16^3 + 2 * 16^2 + 3 * 16^1 + 4 * 16^0) :=
by sorry

end NUMINAMATH_CALUDE_hex_to_binary_digits_l2928_292897


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2928_292830

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  (X^5 - 1) * (X^3 - 1) = (X^3 + X^2 + X + 1) * q + (2*X + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2928_292830


namespace NUMINAMATH_CALUDE_tractor_oil_theorem_l2928_292882

/-- Represents the remaining oil in liters after t hours of work -/
def remaining_oil (initial_oil : ℝ) (consumption_rate : ℝ) (t : ℝ) : ℝ :=
  initial_oil - consumption_rate * t

theorem tractor_oil_theorem (initial_oil : ℝ) (consumption_rate : ℝ) (t : ℝ) :
  initial_oil = 50 → consumption_rate = 8 →
  (∀ t, remaining_oil initial_oil consumption_rate t = 50 - 8 * t) ∧
  (remaining_oil initial_oil consumption_rate 4 = 18) := by
  sorry


end NUMINAMATH_CALUDE_tractor_oil_theorem_l2928_292882


namespace NUMINAMATH_CALUDE_min_organizer_handshakes_l2928_292854

/-- The number of handshakes between players in a chess tournament where each player plays against every other player exactly once -/
def player_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The total number of handshakes including those of the organizer -/
def total_handshakes (n : ℕ) (k : ℕ) : ℕ := player_handshakes n + k

/-- Theorem stating that the minimum number of organizer handshakes is 0 given 406 total handshakes -/
theorem min_organizer_handshakes :
  ∃ (n : ℕ), total_handshakes n 0 = 406 ∧ 
  ∀ (m k : ℕ), total_handshakes m k = 406 → k ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_min_organizer_handshakes_l2928_292854


namespace NUMINAMATH_CALUDE_f_is_even_and_decreasing_l2928_292857

-- Define the function f(x) = -x²
def f (x : ℝ) : ℝ := -x^2

-- State the theorem
theorem f_is_even_and_decreasing :
  (∀ x : ℝ, f (-x) = f x) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_and_decreasing_l2928_292857


namespace NUMINAMATH_CALUDE_complex_division_l2928_292801

/-- Given complex numbers z₁ and z₂ corresponding to points (1, -1) and (-2, 1) in the complex plane,
    prove that z₂/z₁ = -3/2 - 1/2i. -/
theorem complex_division (z₁ z₂ : ℂ) (h₁ : z₁ = Complex.mk 1 (-1)) (h₂ : z₂ = Complex.mk (-2) 1) :
  z₂ / z₁ = Complex.mk (-3/2) (-1/2) := by
  sorry

end NUMINAMATH_CALUDE_complex_division_l2928_292801


namespace NUMINAMATH_CALUDE_equation_value_l2928_292858

theorem equation_value (x y z w : ℝ) 
  (h1 : 4 * x * z + y * w = 3)
  (h2 : (2 * x + y) * (2 * z + w) = 15) :
  x * w + y * z = 6 := by
sorry

end NUMINAMATH_CALUDE_equation_value_l2928_292858


namespace NUMINAMATH_CALUDE_sandcastle_heights_sum_l2928_292800

/-- Represents the height of a sandcastle in feet and fractions of a foot -/
structure SandcastleHeight where
  whole : ℕ
  numerator : ℕ
  denominator : ℕ

/-- Calculates the total height of four sandcastles -/
def total_height (janet : SandcastleHeight) (sister : SandcastleHeight) 
                 (tom : SandcastleHeight) (lucy : SandcastleHeight) : ℚ :=
  (janet.whole : ℚ) + (janet.numerator : ℚ) / (janet.denominator : ℚ) +
  (sister.whole : ℚ) + (sister.numerator : ℚ) / (sister.denominator : ℚ) +
  (tom.whole : ℚ) + (tom.numerator : ℚ) / (tom.denominator : ℚ) +
  (lucy.whole : ℚ) + (lucy.numerator : ℚ) / (lucy.denominator : ℚ)

theorem sandcastle_heights_sum :
  let janet := SandcastleHeight.mk 3 5 6
  let sister := SandcastleHeight.mk 2 7 12
  let tom := SandcastleHeight.mk 1 11 20
  let lucy := SandcastleHeight.mk 2 13 24
  total_height janet sister tom lucy = 10 + 61 / 120 := by sorry

end NUMINAMATH_CALUDE_sandcastle_heights_sum_l2928_292800


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_range_l2928_292818

theorem function_inequality_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 2 * 9^x - 3^x + a^2 - a - 3 > 0) → 
  (a > 2 ∨ a < -1) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_range_l2928_292818


namespace NUMINAMATH_CALUDE_scaling_property_l2928_292892

theorem scaling_property (x y z : ℝ) (h : 2994 * x * 14.5 = 173) : 29.94 * x * 1.45 = 1.73 := by
  sorry

end NUMINAMATH_CALUDE_scaling_property_l2928_292892


namespace NUMINAMATH_CALUDE_mersenne_prime_implies_prime_exponent_l2928_292861

theorem mersenne_prime_implies_prime_exponent (n : ℕ) :
  Nat.Prime (2^n - 1) → Nat.Prime n := by
sorry

end NUMINAMATH_CALUDE_mersenne_prime_implies_prime_exponent_l2928_292861


namespace NUMINAMATH_CALUDE_inequality_proof_l2928_292804

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 6) :
  (1 / (a * (1 + b))) + (1 / (b * (1 + c))) + (1 / (c * (1 + a))) ≥ 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2928_292804


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2928_292886

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 3)^2 - 3*(a 3) - 5 = 0 →
  (a 10)^2 - 3*(a 10) - 5 = 0 →
  a 5 + a 8 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2928_292886


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_l2928_292829

theorem smallest_base_perfect_square : 
  ∃ (b : ℕ), b > 3 ∧ b = 5 ∧ ∀ (x : ℕ), x > 3 ∧ x < b → ¬∃ (y : ℕ), 4 * x + 5 = y ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_l2928_292829


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2928_292899

theorem cube_root_equation_solution :
  ∃ y : ℝ, (5 + 2/y)^(1/3 : ℝ) = 3 ↔ y = 1/11 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2928_292899


namespace NUMINAMATH_CALUDE_x_value_when_y_is_negative_four_l2928_292859

theorem x_value_when_y_is_negative_four :
  ∀ x y : ℝ, 16 * (3 : ℝ)^x = 7^(y + 4) → y = -4 → x = -4 * (Real.log 2 / Real.log 3) := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_negative_four_l2928_292859


namespace NUMINAMATH_CALUDE_remainder_of_2468135792_div_101_l2928_292889

theorem remainder_of_2468135792_div_101 :
  (2468135792 : ℕ) % 101 = 52 := by sorry

end NUMINAMATH_CALUDE_remainder_of_2468135792_div_101_l2928_292889


namespace NUMINAMATH_CALUDE_cistern_fill_fraction_l2928_292855

/-- Represents the time in minutes it takes to fill a portion of the cistern -/
def fill_time : ℝ := 35

/-- Represents the fraction of the cistern filled in the given time -/
def fraction_filled : ℝ := 1

/-- Proves that the fraction of the cistern filled is 1 given the conditions -/
theorem cistern_fill_fraction :
  (fill_time = 35) → fraction_filled = 1 := by
  sorry

end NUMINAMATH_CALUDE_cistern_fill_fraction_l2928_292855


namespace NUMINAMATH_CALUDE_derivative_at_two_l2928_292894

/-- Given a function f with the property that f(x) = 2xf'(2) + x^3 for all x,
    prove that f'(2) = -12 -/
theorem derivative_at_two (f : ℝ → ℝ) (h : ∀ x, f x = 2 * x * (deriv f 2) + x^3) :
  deriv f 2 = -12 := by sorry

end NUMINAMATH_CALUDE_derivative_at_two_l2928_292894


namespace NUMINAMATH_CALUDE_fractional_equation_root_l2928_292870

theorem fractional_equation_root (m : ℝ) : 
  (∃ x : ℝ, x ≠ 2 ∧ x ≠ 2 ∧ (3 / (x - 2) + 1 = m / (4 - 2*x))) → m = -6 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_root_l2928_292870


namespace NUMINAMATH_CALUDE_max_min_value_sqrt_three_l2928_292887

theorem max_min_value_sqrt_three : 
  ∃ (M : ℝ), M > 0 ∧ 
  (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → 
    min (min (min (1/a) (1/(b^2))) (1/(c^3))) (a + b^2 + c^3) ≤ M) ∧
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    min (min (min (1/a) (1/(b^2))) (1/(c^3))) (a + b^2 + c^3) = M) ∧
  M = Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_max_min_value_sqrt_three_l2928_292887


namespace NUMINAMATH_CALUDE_smallest_factor_of_36_l2928_292813

theorem smallest_factor_of_36 (a b c : ℤ) 
  (h1 : a * b * c = 36) 
  (h2 : a + b + c = 4) : 
  min a (min b c) = -4 := by
sorry

end NUMINAMATH_CALUDE_smallest_factor_of_36_l2928_292813


namespace NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_6_divisible_by_5_l2928_292852

theorem smallest_positive_integer_ending_in_6_divisible_by_5 :
  ∃ (n : ℕ), n > 0 ∧ n % 10 = 6 ∧ n % 5 = 0 ∧
  ∀ (m : ℕ), m > 0 → m % 10 = 6 → m % 5 = 0 → m ≥ n :=
by
  use 46
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_6_divisible_by_5_l2928_292852


namespace NUMINAMATH_CALUDE_strawberry_area_l2928_292866

theorem strawberry_area (garden_size : ℝ) (fruit_ratio : ℝ) (strawberry_ratio : ℝ) : 
  garden_size = 64 → 
  fruit_ratio = 1/2 → 
  strawberry_ratio = 1/4 → 
  garden_size * fruit_ratio * strawberry_ratio = 8 := by
sorry

end NUMINAMATH_CALUDE_strawberry_area_l2928_292866


namespace NUMINAMATH_CALUDE_gcd_6363_1923_l2928_292815

theorem gcd_6363_1923 : Nat.gcd 6363 1923 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_6363_1923_l2928_292815


namespace NUMINAMATH_CALUDE_max_jogs_is_seven_l2928_292895

/-- Represents the number of items Bill buys -/
structure BillsPurchase where
  jags : Nat
  jigs : Nat
  jogs : Nat
  jugs : Nat

/-- Calculates the total cost of Bill's purchase -/
def totalCost (p : BillsPurchase) : Nat :=
  2 * p.jags + 3 * p.jigs + 8 * p.jogs + 5 * p.jugs

/-- Represents a valid purchase satisfying all conditions -/
def isValidPurchase (p : BillsPurchase) : Prop :=
  p.jags ≥ 1 ∧ p.jigs ≥ 1 ∧ p.jogs ≥ 1 ∧ p.jugs ≥ 1 ∧ totalCost p = 72

theorem max_jogs_is_seven :
  ∀ p : BillsPurchase, isValidPurchase p → p.jogs ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_max_jogs_is_seven_l2928_292895


namespace NUMINAMATH_CALUDE_min_value_of_function_l2928_292875

theorem min_value_of_function (x : ℝ) (h : x > 5/4) :
  ∃ (y : ℝ), y = 4*x - 1 + 1/(4*x - 5) ∧ y ≥ 6 ∧ (∃ (x₀ : ℝ), x₀ > 5/4 ∧ 4*x₀ - 1 + 1/(4*x₀ - 5) = 6) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2928_292875


namespace NUMINAMATH_CALUDE_average_birds_seen_l2928_292805

def marcus_birds : ℕ := 7
def humphrey_birds : ℕ := 11
def darrel_birds : ℕ := 9
def total_watchers : ℕ := 3

theorem average_birds_seen :
  (marcus_birds + humphrey_birds + darrel_birds) / total_watchers = 9 := by
  sorry

end NUMINAMATH_CALUDE_average_birds_seen_l2928_292805


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2928_292839

theorem arithmetic_sequence_problem (a : ℕ → ℤ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 4 - a 2 = -2 →                                      -- given condition
  a 7 = -3 →                                            -- given condition
  a 9 = -5 := by                                        -- conclusion to prove
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2928_292839


namespace NUMINAMATH_CALUDE_total_money_together_l2928_292893

def henry_initial_money : ℕ := 5
def henry_earned_money : ℕ := 2
def friend_money : ℕ := 13

theorem total_money_together : 
  henry_initial_money + henry_earned_money + friend_money = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_money_together_l2928_292893


namespace NUMINAMATH_CALUDE_monotonic_power_function_l2928_292835

theorem monotonic_power_function (m : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → (m^2 - 5*m + 7) * x₁^(m^2 - 6) < (m^2 - 5*m + 7) * x₂^(m^2 - 6)) →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_monotonic_power_function_l2928_292835


namespace NUMINAMATH_CALUDE_park_outer_boundary_diameter_l2928_292832

/-- The diameter of the outer boundary of a circular park structure -/
def outer_boundary_diameter (pond_diameter : ℝ) (picnic_width : ℝ) (track_width : ℝ) : ℝ :=
  pond_diameter + 2 * (picnic_width + track_width)

/-- Theorem stating the diameter of the outer boundary of the cycling track -/
theorem park_outer_boundary_diameter :
  outer_boundary_diameter 16 10 4 = 44 := by
  sorry

end NUMINAMATH_CALUDE_park_outer_boundary_diameter_l2928_292832


namespace NUMINAMATH_CALUDE_sixth_quiz_score_l2928_292871

theorem sixth_quiz_score (scores : List ℕ) (target_mean : ℕ) : 
  scores = [86, 90, 82, 84, 95] →
  target_mean = 95 →
  ∃ (sixth_score : ℕ), 
    sixth_score = 133 ∧ 
    (scores.sum + sixth_score) / 6 = target_mean :=
by sorry

end NUMINAMATH_CALUDE_sixth_quiz_score_l2928_292871


namespace NUMINAMATH_CALUDE_three_digit_number_proof_l2928_292872

/-- A three-digit number is between 100 and 999 inclusive -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem three_digit_number_proof :
  ∃ (x : ℕ), is_three_digit x ∧ (7000 + x) - (10 * x + 7) = 3555 ∧ x = 382 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_proof_l2928_292872


namespace NUMINAMATH_CALUDE_fraction_expressions_l2928_292807

theorem fraction_expressions (x z : ℚ) (h : x / z = 5 / 6) :
  ((x + 3 * z) / z = 23 / 6) ∧
  (z / (x - z) = -6) ∧
  ((2 * x + z) / z = 8 / 3) ∧
  (3 * x / (4 * z) = 5 / 8) ∧
  ((x - 2 * z) / z = -7 / 6) := by
  sorry

end NUMINAMATH_CALUDE_fraction_expressions_l2928_292807


namespace NUMINAMATH_CALUDE_unique_three_digit_square_l2928_292869

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define a function to get the first two digits of a three-digit number
def first_two_digits (n : ℕ) : ℕ :=
  n / 10

-- Define a function to get the last digit of a three-digit number
def last_digit (n : ℕ) : ℕ :=
  n % 10

-- Define the main theorem
theorem unique_three_digit_square : ∃! n : ℕ, 
  100 ≤ n ∧ n ≤ 999 ∧ 
  is_perfect_square n ∧ 
  is_perfect_square (first_two_digits n / last_digit n) ∧
  n = 361 :=
sorry

end NUMINAMATH_CALUDE_unique_three_digit_square_l2928_292869


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l2928_292862

theorem quadratic_is_square_of_binomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 + 12*x + a = (3*x + b)^2) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l2928_292862
