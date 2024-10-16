import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l2703_270334

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) :
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2703_270334


namespace NUMINAMATH_CALUDE_catering_company_comparison_l2703_270303

/-- Represents the cost function for a catering company -/
structure CateringCompany where
  basicFee : ℕ
  perPersonFee : ℕ

/-- Calculates the total cost for a given number of people -/
def totalCost (company : CateringCompany) (people : ℕ) : ℕ :=
  company.basicFee + company.perPersonFee * people

/-- The problem statement -/
theorem catering_company_comparison :
  let company1 : CateringCompany := ⟨120, 18⟩
  let company2 : CateringCompany := ⟨250, 15⟩
  ∀ n : ℕ, n < 44 → totalCost company1 n ≤ totalCost company2 n ∧
  totalCost company2 44 < totalCost company1 44 :=
by sorry

end NUMINAMATH_CALUDE_catering_company_comparison_l2703_270303


namespace NUMINAMATH_CALUDE_total_apples_l2703_270381

def pinky_apples : ℕ := 36
def danny_apples : ℕ := 73

theorem total_apples : pinky_apples + danny_apples = 109 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_l2703_270381


namespace NUMINAMATH_CALUDE_smallest_s_for_E_l2703_270371

/-- Definition of the function E --/
def E (a b c : ℕ) : ℕ := a * b^c

/-- The smallest positive integer s that satisfies E(s, s, 4) = 2401 is 7 --/
theorem smallest_s_for_E : (∃ s : ℕ, s > 0 ∧ E s s 4 = 2401 ∧ ∀ t : ℕ, t > 0 → E t t 4 = 2401 → s ≤ t) ∧ 
                           (∃ s : ℕ, s > 0 ∧ E s s 4 = 2401 ∧ s = 7) := by
  sorry

end NUMINAMATH_CALUDE_smallest_s_for_E_l2703_270371


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2703_270390

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^9 = a₉*x^9 + a₈*x^8 + a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2703_270390


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l2703_270387

theorem sqrt_x_div_sqrt_y (x y : ℝ) (h : (1/3)^2 + (1/4)^2 = ((1/5)^2 + (1/6)^2 + 1/600) * (25*x)/(73*y)) :
  Real.sqrt x / Real.sqrt y = 147/43 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l2703_270387


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2703_270372

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a1 : a 1 = 3)
  (h_arith : arithmetic_sequence (λ n => match n with
    | 1 => 4 * (a 1)
    | 2 => 2 * (a 2)
    | 3 => a 3
    | _ => 0
  )) :
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2703_270372


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2703_270344

theorem negation_of_universal_proposition (f : ℕ+ → ℝ) :
  (¬ ∀ n : ℕ+, f n ≤ n) ↔ (∃ n : ℕ+, f n > n) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2703_270344


namespace NUMINAMATH_CALUDE_exponent_division_l2703_270383

theorem exponent_division (x : ℝ) (hx : x ≠ 0) : x^3 / x^2 = x := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2703_270383


namespace NUMINAMATH_CALUDE_product_of_valid_m_l2703_270376

theorem product_of_valid_m : ∃ (S : Finset ℤ), 
  (∀ m ∈ S, m ≥ 1 ∧ 
    ∃ y : ℤ, y ≠ 2 ∧ m * y / (y - 2) + 1 = -3 * y / (2 - y)) ∧ 
  (∀ m : ℤ, m ≥ 1 → 
    (∃ y : ℤ, y ≠ 2 ∧ m * y / (y - 2) + 1 = -3 * y / (2 - y)) → 
    m ∈ S) ∧
  S.prod id = 4 :=
sorry

end NUMINAMATH_CALUDE_product_of_valid_m_l2703_270376


namespace NUMINAMATH_CALUDE_total_age_problem_l2703_270343

/-- Given three people a, b, and c, where a is two years older than b, 
    b is twice as old as c, and b is 10 years old, 
    prove that the total of their ages is 27 years. -/
theorem total_age_problem (a b c : ℕ) : 
  b = 10 → a = b + 2 → b = 2 * c → a + b + c = 27 := by
sorry

end NUMINAMATH_CALUDE_total_age_problem_l2703_270343


namespace NUMINAMATH_CALUDE_problem_statement_l2703_270359

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + y^2 = x + y) :
  (∀ a b : ℝ, a > 0 → b > 0 → a^2 + b^2 = a + b → 1/a + 1/b ≥ 2) ∧
  (1/x + 1/y = 2 ↔ x = 1 ∧ y = 1) ∧
  ¬∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = a + b ∧ (a + 1) * (b + 1) = 5 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2703_270359


namespace NUMINAMATH_CALUDE_remainder_2503_div_28_l2703_270317

theorem remainder_2503_div_28 : 2503 % 28 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2503_div_28_l2703_270317


namespace NUMINAMATH_CALUDE_initial_trees_count_l2703_270306

/-- The number of walnut trees to be removed from the park -/
def trees_removed : ℕ := 4

/-- The number of walnut trees remaining after removal -/
def trees_remaining : ℕ := 2

/-- The initial number of walnut trees in the park -/
def initial_trees : ℕ := trees_removed + trees_remaining

theorem initial_trees_count : initial_trees = 6 := by sorry

end NUMINAMATH_CALUDE_initial_trees_count_l2703_270306


namespace NUMINAMATH_CALUDE_unique_cube_constructions_l2703_270397

/-- The order of the rotational group for a 3x3x3 cube -/
def rotational_group_order : ℕ := 26

/-- The total number of unit cubes in a 3x3x3 cube -/
def total_cubes : ℕ := 27

/-- The number of white unit cubes -/
def white_cubes : ℕ := 13

/-- The number of blue unit cubes -/
def blue_cubes : ℕ := 14

/-- The number of configurations for the identity rotation -/
def identity_configurations : ℕ := Nat.choose total_cubes white_cubes

/-- The number of configurations for face rotations -/
def face_rotation_configurations : ℕ := 2

/-- The number of configurations for vertex and edge rotations -/
def vertex_edge_rotation_configurations : ℕ := 1

/-- The total number of configurations -/
def total_configurations : ℕ :=
  identity_configurations + face_rotation_configurations + vertex_edge_rotation_configurations

/-- The theorem stating the number of unique ways to construct the cube -/
theorem unique_cube_constructions :
  (total_configurations / rotational_group_order : ℚ) = 89754 := by sorry

end NUMINAMATH_CALUDE_unique_cube_constructions_l2703_270397


namespace NUMINAMATH_CALUDE_parallel_line_length_l2703_270339

/-- Given a triangle with base 20 inches and a parallel line dividing it into two parts
    where the upper part has 3/4 of the total area, the length of this parallel line is 10 inches. -/
theorem parallel_line_length (base : ℝ) (parallel_line : ℝ) : 
  base = 20 →
  (parallel_line / base) ^ 2 = 1 / 4 →
  parallel_line = 10 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_length_l2703_270339


namespace NUMINAMATH_CALUDE_cafeteria_apples_l2703_270366

def apples_handed_out (initial_apples : ℕ) (pies_made : ℕ) (apples_per_pie : ℕ) : ℕ :=
  initial_apples - pies_made * apples_per_pie

theorem cafeteria_apples 
  (initial_apples : ℕ) 
  (pies_made : ℕ) 
  (apples_per_pie : ℕ) 
  (h1 : initial_apples = 50) 
  (h2 : pies_made = 9) 
  (h3 : apples_per_pie = 5) :
  apples_handed_out initial_apples pies_made apples_per_pie = 5 := by
sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l2703_270366


namespace NUMINAMATH_CALUDE_total_cost_calculation_total_cost_proof_l2703_270312

/-- Given the price of tomatoes and cabbage per kilogram, calculate the total cost of purchasing 20 kg of tomatoes and 30 kg of cabbage. -/
theorem total_cost_calculation (a b : ℝ) : ℝ :=
  let tomato_price_per_kg := a
  let cabbage_price_per_kg := b
  let tomato_quantity := 20
  let cabbage_quantity := 30
  tomato_price_per_kg * tomato_quantity + cabbage_price_per_kg * cabbage_quantity

#check total_cost_calculation

theorem total_cost_proof (a b : ℝ) :
  total_cost_calculation a b = 20 * a + 30 * b := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_total_cost_proof_l2703_270312


namespace NUMINAMATH_CALUDE_sequence_difference_l2703_270386

theorem sequence_difference (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n, S n = n^2) : 
  a 3 - a 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_difference_l2703_270386


namespace NUMINAMATH_CALUDE_towels_per_load_l2703_270391

theorem towels_per_load (total_towels : ℕ) (num_loads : ℕ) (h1 : total_towels = 42) (h2 : num_loads = 6) :
  total_towels / num_loads = 7 := by
  sorry

end NUMINAMATH_CALUDE_towels_per_load_l2703_270391


namespace NUMINAMATH_CALUDE_pulley_centers_distance_l2703_270360

theorem pulley_centers_distance 
  (r1 r2 contact_distance : ℝ) 
  (h1 : r1 = 18)
  (h2 : r2 = 16)
  (h3 : contact_distance = 40) :
  let center_distance := Real.sqrt (contact_distance^2 + (r1 - r2)^2)
  center_distance = Real.sqrt 1604 := by
  sorry

end NUMINAMATH_CALUDE_pulley_centers_distance_l2703_270360


namespace NUMINAMATH_CALUDE_mittens_per_box_l2703_270347

theorem mittens_per_box (boxes : ℕ) (scarves_per_box : ℕ) (total_clothing : ℕ)
  (h1 : boxes = 7)
  (h2 : scarves_per_box = 3)
  (h3 : total_clothing = 49) :
  (total_clothing - boxes * scarves_per_box) / boxes = 4 := by
  sorry

end NUMINAMATH_CALUDE_mittens_per_box_l2703_270347


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l2703_270320

/-- The length of the longer diagonal of a rhombus given its side length and shorter diagonal -/
theorem rhombus_longer_diagonal (side : ℝ) (shorter_diagonal : ℝ) (h1 : side = 65) (h2 : shorter_diagonal = 56) :
  ∃ longer_diagonal : ℝ, longer_diagonal = 2 * Real.sqrt 3441 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l2703_270320


namespace NUMINAMATH_CALUDE_cube_vertical_faces_same_color_prob_l2703_270358

/-- Represents the probability of painting a face blue -/
def blue_prob : ℚ := 1/3

/-- Represents the probability of painting a face red -/
def red_prob : ℚ := 2/3

/-- Represents the number of faces on a cube -/
def num_faces : ℕ := 6

/-- Represents the number of vertical faces when a cube is placed on a horizontal surface -/
def num_vertical_faces : ℕ := 4

/-- Calculates the probability of all faces being the same color -/
def all_same_color_prob : ℚ := red_prob^num_faces + blue_prob^num_faces

/-- Calculates the probability of vertical faces being one color and top/bottom being another -/
def mixed_color_prob : ℚ := 3 * (red_prob^num_vertical_faces * blue_prob^(num_faces - num_vertical_faces) +
                                 blue_prob^num_vertical_faces * red_prob^(num_faces - num_vertical_faces))

/-- The main theorem stating the probability of the cube having all four vertical faces
    the same color when placed on a horizontal surface -/
theorem cube_vertical_faces_same_color_prob :
  all_same_color_prob + mixed_color_prob = 789/6561 := by sorry

end NUMINAMATH_CALUDE_cube_vertical_faces_same_color_prob_l2703_270358


namespace NUMINAMATH_CALUDE_expression_simplification_l2703_270365

theorem expression_simplification (x y : ℝ) : 
  3 * x + 5 * x^2 - 4 * y - (6 - 3 * x - 5 * x^2 + 2 * y) - (4 * y^2 - 8 + 2 * x^2 - y) = 
  8 * x^2 - 4 * y^2 + 6 * x - 5 * y + 2 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l2703_270365


namespace NUMINAMATH_CALUDE_max_operations_l2703_270368

/-- A coin configuration is a list of booleans, where true represents H and false represents T. -/
def CoinConfiguration := List Bool

/-- The number of operations required to turn all coins to T (false) -/
def operations (c : CoinConfiguration) : ℕ := sorry

/-- The theorem states that for any configuration of n coins, the maximum number of operations
    required to turn all coins to T is ⌈n/2⌉. -/
theorem max_operations (n : ℕ) :
  ∀ c : CoinConfiguration, c.length = n →
  operations c ≤ (n + 1) / 2 ∧
  ∃ c : CoinConfiguration, c.length = n ∧ operations c = (n + 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_operations_l2703_270368


namespace NUMINAMATH_CALUDE_product_from_lcm_hcf_l2703_270356

theorem product_from_lcm_hcf (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 72) 
  (h_hcf : Nat.gcd a b = 6) : 
  a * b = 432 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_hcf_l2703_270356


namespace NUMINAMATH_CALUDE_problem_statement_l2703_270301

theorem problem_statement (x y : ℝ) :
  |x - 8*y| + (4*y - 1)^2 = 0 → (x + 2*y)^3 = 125/8 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2703_270301


namespace NUMINAMATH_CALUDE_power_tower_mod_500_l2703_270307

theorem power_tower_mod_500 : 5^(5^(5^5)) ≡ 125 [ZMOD 500] := by sorry

end NUMINAMATH_CALUDE_power_tower_mod_500_l2703_270307


namespace NUMINAMATH_CALUDE_square_51_and_39_l2703_270399

theorem square_51_and_39 : 51^2 = 2601 ∧ 39^2 = 1521 := by
  -- Given: (a ± b)² = a² ± 2ab + b²
  sorry


end NUMINAMATH_CALUDE_square_51_and_39_l2703_270399


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2703_270395

/-- Two vectors are parallel if their corresponding components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ b.1 = k * a.1 ∧ b.2 = k * a.2

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, -4)
  are_parallel a b → x = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2703_270395


namespace NUMINAMATH_CALUDE_polynomial_equality_l2703_270367

theorem polynomial_equality (a b c m n : ℝ) : 
  (∀ x : ℝ, m * x^2 - n * x + 3 = a * (x - 1)^2 + b * (x - 1) + c) →
  a - b + c = 3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2703_270367


namespace NUMINAMATH_CALUDE_tank_capacity_l2703_270364

theorem tank_capacity (y : ℝ) 
  (h1 : (7/8) * y - 20 = (1/4) * y) : y = 32 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l2703_270364


namespace NUMINAMATH_CALUDE_meet_once_l2703_270313

/-- Represents the meeting scenario between Michael and the garbage truck --/
structure MeetingScenario where
  michael_speed : ℝ
  truck_speed : ℝ
  pail_distance : ℝ
  truck_stop_time : ℝ
  initial_distance : ℝ

/-- Calculates the number of meetings between Michael and the truck --/
def number_of_meetings (scenario : MeetingScenario) : ℕ :=
  sorry

/-- Theorem stating that Michael and the truck meet exactly once --/
theorem meet_once (scenario : MeetingScenario) : 
  scenario.michael_speed = 6 ∧ 
  scenario.truck_speed = 10 ∧ 
  scenario.pail_distance = 200 ∧ 
  scenario.truck_stop_time = 30 ∧
  scenario.initial_distance = 200 →
  number_of_meetings scenario = 1 :=
sorry

end NUMINAMATH_CALUDE_meet_once_l2703_270313


namespace NUMINAMATH_CALUDE_a_plus_b_value_l2703_270323

-- Define the functions f and h
def f (a b x : ℝ) : ℝ := a * x + b
def h (x : ℝ) : ℝ := 3 * x + 1

-- State the theorem
theorem a_plus_b_value (a b : ℝ) : 
  (∀ x, h (f a b x) = 5 * x - 8) → a + b = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l2703_270323


namespace NUMINAMATH_CALUDE_sum_of_arithmetic_sequence_l2703_270327

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

-- State the theorem
theorem sum_of_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a5 : a 5 = 3) 
  (h_a6 : a 6 = -2) : 
  (a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 3) :=
sorry

end NUMINAMATH_CALUDE_sum_of_arithmetic_sequence_l2703_270327


namespace NUMINAMATH_CALUDE_solve_for_m_l2703_270379

-- Define the custom operation
def customOp (m n : ℕ) : ℕ := n^2 - m

-- State the theorem
theorem solve_for_m :
  (∀ m n, customOp m n = n^2 - m) →
  (∃ m, customOp m 3 = 3) →
  (∃ m, m = 6) :=
by sorry

end NUMINAMATH_CALUDE_solve_for_m_l2703_270379


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l2703_270377

theorem geometric_series_ratio (a r : ℝ) (h : r ≠ 1) :
  (a * r^4 / (1 - r)) = (a / (1 - r)) / 81 → r = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l2703_270377


namespace NUMINAMATH_CALUDE_horner_method_operations_l2703_270369

def horner_polynomial (x : ℝ) : ℝ := ((((((9 * x + 12) * x + 7) * x + 54) * x + 34) * x + 9) * x + 1)

theorem horner_method_operations :
  let f := λ (x : ℝ) => 9 * x^6 + 12 * x^5 + 7 * x^4 + 54 * x^3 + 34 * x^2 + 9 * x + 1
  ∃ (mult_ops add_ops : ℕ), 
    (∀ x : ℝ, f x = horner_polynomial x) ∧
    mult_ops = 6 ∧
    add_ops = 6 :=
sorry

end NUMINAMATH_CALUDE_horner_method_operations_l2703_270369


namespace NUMINAMATH_CALUDE_number_of_parents_at_park_parents_at_park_l2703_270315

/-- Given a group of people at a park, prove the number of parents. -/
theorem number_of_parents_at_park (num_girls : ℕ) (num_boys : ℕ) (num_groups : ℕ) (group_size : ℕ) : ℕ :=
  let total_people := num_groups * group_size
  let total_children := num_girls + num_boys
  total_people - total_children

/-- Prove that there are 50 parents at the park given the specified conditions. -/
theorem parents_at_park : number_of_parents_at_park 14 11 3 25 = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_of_parents_at_park_parents_at_park_l2703_270315


namespace NUMINAMATH_CALUDE_cleaning_time_proof_l2703_270342

def grove_width : ℕ := 4
def grove_length : ℕ := 5
def cleaning_time_per_tree : ℕ := 6
def minutes_per_hour : ℕ := 60

theorem cleaning_time_proof :
  let total_trees := grove_width * grove_length
  let total_cleaning_time := total_trees * cleaning_time_per_tree
  let cleaning_time_hours := total_cleaning_time / minutes_per_hour
  let actual_cleaning_time := cleaning_time_hours / 2
  actual_cleaning_time = 1 := by sorry

end NUMINAMATH_CALUDE_cleaning_time_proof_l2703_270342


namespace NUMINAMATH_CALUDE_unknown_towel_rate_unknown_towel_rate_solution_l2703_270311

/-- Proves that the unknown rate of two towels is 300, given the conditions of the problem -/
theorem unknown_towel_rate : ℕ → Prop :=
  fun (x : ℕ) ↦
    let total_towels : ℕ := 3 + 5 + 2
    let known_cost : ℕ := 3 * 100 + 5 * 150
    let total_cost : ℕ := 165 * total_towels
    (known_cost + 2 * x = total_cost) → (x = 300)

/-- Solution to the unknown_towel_rate theorem -/
theorem unknown_towel_rate_solution : unknown_towel_rate 300 := by
  sorry

end NUMINAMATH_CALUDE_unknown_towel_rate_unknown_towel_rate_solution_l2703_270311


namespace NUMINAMATH_CALUDE_triangle_radius_equations_l2703_270384

/-- Given a triangle ABC with angles 2α, 2β, and 2γ, prove two equations involving inradius, exradii, and side lengths. -/
theorem triangle_radius_equations (R α β γ : ℝ) (r r_a r_b r_c a b c : ℝ) 
  (h_r : r = 4 * R * Real.sin α * Real.sin β * Real.sin γ)
  (h_ra : r_a = 4 * R * Real.sin α * Real.cos β * Real.cos γ)
  (h_rb : r_b = 4 * R * Real.cos α * Real.sin β * Real.cos γ)
  (h_rc : r_c = 4 * R * Real.cos α * Real.cos β * Real.sin γ)
  (h_a : a = 4 * R * Real.sin α * Real.cos α)
  (h_bc : b + c = 4 * R * Real.sin (β + γ) * Real.cos (β - γ)) :
  (a * (b + c) = (r + r_a) * (4 * R + r - r_a)) ∧ 
  (a * (b - c) = (r_b - r_c) * (4 * R - r_b - r_c)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_radius_equations_l2703_270384


namespace NUMINAMATH_CALUDE_train_speed_calculation_train_speed_is_36_l2703_270336

/-- Calculates the speed of a train given the following conditions:
  * A jogger is running at 9 kmph
  * The jogger is 270 meters ahead of the train's engine
  * The train is 120 meters long
  * The train takes 39 seconds to pass the jogger
-/
theorem train_speed_calculation (jogger_speed : ℝ) (initial_distance : ℝ) 
  (train_length : ℝ) (passing_time : ℝ) : ℝ :=
  let total_distance := initial_distance + train_length
  let train_speed := (total_distance / 1000) / (passing_time / 3600)
  by
    sorry

/-- The main theorem stating that under the given conditions, 
    the train's speed is 36 kmph -/
theorem train_speed_is_36 : 
  train_speed_calculation 9 270 120 39 = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_train_speed_is_36_l2703_270336


namespace NUMINAMATH_CALUDE_max_distance_ellipse_point_l2703_270316

/-- The maximum distance between any point on the ellipse x²/36 + y²/27 = 1 and the point (3,0) is 9 -/
theorem max_distance_ellipse_point : 
  ∃ (M : ℝ × ℝ), 
    (M.1^2 / 36 + M.2^2 / 27 = 1) ∧ 
    (∀ (N : ℝ × ℝ), (N.1^2 / 36 + N.2^2 / 27 = 1) → 
      ((N.1 - 3)^2 + N.2^2)^(1/2) ≤ ((M.1 - 3)^2 + M.2^2)^(1/2)) ∧
    ((M.1 - 3)^2 + M.2^2)^(1/2) = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_ellipse_point_l2703_270316


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l2703_270352

-- Define the circle equation
def circle_equation (x y : ℝ) : ℝ := x^2 + y^2 - 8*x + 6*y

-- Define the points
def point_A : ℝ × ℝ := (0, 0)
def point_B : ℝ × ℝ := (1, 1)
def point_C : ℝ × ℝ := (4, 2)

-- Theorem statement
theorem circle_passes_through_points :
  circle_equation point_A.1 point_A.2 = 0 ∧
  circle_equation point_B.1 point_B.2 = 0 ∧
  circle_equation point_C.1 point_C.2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l2703_270352


namespace NUMINAMATH_CALUDE_sum_of_seven_angles_l2703_270338

-- Define the angles
variable (angle1 angle2 angle3 angle4 angle5 angle6 angle7 angle8 angle9 angle10 : ℝ)

-- State the theorem
theorem sum_of_seven_angles :
  (angle5 + angle6 + angle7 + angle8 = 360) →
  (angle2 + angle3 + angle4 + (180 - angle9) = 360) →
  (angle9 = angle10) →
  (angle8 = angle10 + angle1) →
  (angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + angle7 = 540) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_seven_angles_l2703_270338


namespace NUMINAMATH_CALUDE_distinct_strings_equal_fibonacci_l2703_270329

/-- Represents the possible operations on a string --/
inductive Operation
  | replaceH
  | replaceMM
  | replaceT

/-- Defines a valid string after operations --/
def ValidString : Type := List Char

/-- Applies an operation to a valid string --/
def applyOperation (s : ValidString) (op : Operation) : ValidString :=
  sorry

/-- Counts the number of distinct strings after n operations --/
def countDistinctStrings (n : Nat) : Nat :=
  sorry

/-- Computes the nth Fibonacci number (starting with F(1) = 2, F(2) = 3) --/
def fibonacci (n : Nat) : Nat :=
  sorry

/-- The main theorem: number of distinct strings after 10 operations equals 10th Fibonacci number --/
theorem distinct_strings_equal_fibonacci :
  countDistinctStrings 10 = fibonacci 10 := by
  sorry

end NUMINAMATH_CALUDE_distinct_strings_equal_fibonacci_l2703_270329


namespace NUMINAMATH_CALUDE_quadratic_always_has_real_roots_k_range_for_positive_root_less_than_one_l2703_270309

variable (k : ℝ)

def quadratic_equation (x : ℝ) : Prop :=
  x^2 - (k+3)*x + 2*k + 2 = 0

theorem quadratic_always_has_real_roots :
  ∃ x₁ x₂ : ℝ, quadratic_equation k x₁ ∧ quadratic_equation k x₂ :=
sorry

theorem k_range_for_positive_root_less_than_one :
  (∃ x : ℝ, quadratic_equation k x ∧ 0 < x ∧ x < 1) → -1 < k ∧ k < 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_has_real_roots_k_range_for_positive_root_less_than_one_l2703_270309


namespace NUMINAMATH_CALUDE_correct_calculation_l2703_270308

theorem correct_calculation (x : ℤ) : 63 - x = 70 → 36 + x = 29 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2703_270308


namespace NUMINAMATH_CALUDE_number_times_one_fourth_squared_equals_four_cubed_l2703_270357

theorem number_times_one_fourth_squared_equals_four_cubed (x : ℝ) : 
  x * (1/4)^2 = 4^3 ↔ x = 1024 := by
  sorry

end NUMINAMATH_CALUDE_number_times_one_fourth_squared_equals_four_cubed_l2703_270357


namespace NUMINAMATH_CALUDE_parabola_circle_triangle_l2703_270325

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A parabola defined by x^2 = 2py -/
def Parabola (p : ℝ) : Set Point :=
  {pt : Point | pt.x^2 = 2 * p * pt.y}

/-- Check if three points form an equilateral triangle -/
def isEquilateralTriangle (a b c : Point) : Prop :=
  (a.x - b.x)^2 + (a.y - b.y)^2 = 
  (b.x - c.x)^2 + (b.y - c.y)^2 ∧
  (b.x - c.x)^2 + (b.y - c.y)^2 = 
  (c.x - a.x)^2 + (c.y - a.y)^2

/-- The origin point -/
def O : Point := ⟨0, 0⟩

/-- The given point M -/
def M : Point := ⟨0, 9⟩

theorem parabola_circle_triangle (p : ℝ) 
  (h_p_pos : p > 0)
  (A : Point)
  (h_A_on_parabola : A ∈ Parabola p)
  (B : Point)
  (h_B_on_parabola : B ∈ Parabola p)
  (h_circle : (A.x - M.x)^2 + (A.y - M.y)^2 = A.x^2 + A.y^2)
  (h_equilateral : isEquilateralTriangle A B O) :
  p = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_parabola_circle_triangle_l2703_270325


namespace NUMINAMATH_CALUDE_simplify_expression_l2703_270345

theorem simplify_expression (x y : ℝ) : (5 - 4*x) - (7 + 5*x) + 2*y = -2 - 9*x + 2*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2703_270345


namespace NUMINAMATH_CALUDE_coaching_fee_calculation_l2703_270351

/-- Calculates the number of days from January 1 to a given date in a non-leap year -/
def daysFromNewYear (month : Nat) (day : Nat) : Nat :=
  match month with
  | 1 => day
  | 2 => 31 + day
  | 3 => 59 + day
  | 4 => 90 + day
  | 5 => 120 + day
  | 6 => 151 + day
  | 7 => 181 + day
  | 8 => 212 + day
  | 9 => 243 + day
  | 10 => 273 + day
  | 11 => 304 + day
  | 12 => 334 + day
  | _ => 0

/-- Daily coaching charge in dollars -/
def dailyCharge : Nat := 39

/-- Calculates the total coaching fee -/
def totalCoachingFee (startMonth : Nat) (startDay : Nat) (endMonth : Nat) (endDay : Nat) : Nat :=
  let totalDays := daysFromNewYear endMonth endDay - daysFromNewYear startMonth startDay + 1
  totalDays * dailyCharge

theorem coaching_fee_calculation :
  totalCoachingFee 1 1 11 3 = 11934 := by
  sorry

#eval totalCoachingFee 1 1 11 3

end NUMINAMATH_CALUDE_coaching_fee_calculation_l2703_270351


namespace NUMINAMATH_CALUDE_protons_equal_atomic_number_oxygen16_protons_l2703_270361

/-- Represents an atom with mass number and atomic number -/
structure Atom where
  mass_number : ℕ
  atomic_number : ℕ

/-- The oxygen-16 atom -/
def oxygen16 : Atom := { mass_number := 16, atomic_number := 8 }

/-- The number of protons in an atom is equal to its atomic number -/
theorem protons_equal_atomic_number (a : Atom) : a.atomic_number = a.atomic_number := by sorry

theorem oxygen16_protons : oxygen16.atomic_number = 8 := by sorry

end NUMINAMATH_CALUDE_protons_equal_atomic_number_oxygen16_protons_l2703_270361


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_l2703_270394

/-- Proves that given an income of 20000 and savings of 5000, the ratio of income to expenditure is 4:3 -/
theorem income_expenditure_ratio (income : ℕ) (savings : ℕ) (expenditure : ℕ) :
  income = 20000 →
  savings = 5000 →
  expenditure = income - savings →
  (income : ℚ) / expenditure = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_income_expenditure_ratio_l2703_270394


namespace NUMINAMATH_CALUDE_pears_picked_total_l2703_270331

/-- The number of pears Sara picked -/
def sara_pears : ℕ := 6

/-- The number of pears Tim picked -/
def tim_pears : ℕ := 5

/-- The total number of pears picked -/
def total_pears : ℕ := sara_pears + tim_pears

theorem pears_picked_total : total_pears = 11 := by
  sorry

end NUMINAMATH_CALUDE_pears_picked_total_l2703_270331


namespace NUMINAMATH_CALUDE_prime_factorization_problem_l2703_270392

theorem prime_factorization_problem :
  2006^2 * 2262 - 669^2 * 3599 + 1593^2 * 1337 = 2 * 3 * 7 * 13 * 29 * 59 * 61 * 191 := by
  sorry

end NUMINAMATH_CALUDE_prime_factorization_problem_l2703_270392


namespace NUMINAMATH_CALUDE_viaduct_laying_speed_l2703_270346

/-- Proves that the original daily laying length of a viaduct is 300 meters given specific conditions -/
theorem viaduct_laying_speed (total_length : ℝ) (total_days : ℝ) (initial_length : ℝ) 
  (h1 : total_length = 4800)
  (h2 : total_days = 9)
  (h3 : initial_length = 600) : 
  ∃ (original_speed : ℝ), 
    initial_length / original_speed + (total_length - initial_length) / (2 * original_speed) = total_days ∧ 
    original_speed = 300 := by
sorry

end NUMINAMATH_CALUDE_viaduct_laying_speed_l2703_270346


namespace NUMINAMATH_CALUDE_only_two_special_triples_l2703_270349

/-- A structure representing a triple of positive integers (a, b, c) satisfying certain conditions. -/
structure SpecialTriple where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  h1 : a ≥ b
  h2 : b ≥ c
  h3 : ∃ x : ℕ, a^2 + 3*b = x^2
  h4 : ∃ y : ℕ, b^2 + 3*c = y^2
  h5 : ∃ z : ℕ, c^2 + 3*a = z^2

/-- The theorem stating that there are only two SpecialTriples. -/
theorem only_two_special_triples :
  {t : SpecialTriple | t.a = 1 ∧ t.b = 1 ∧ t.c = 1} ∪
  {t : SpecialTriple | t.a = 37 ∧ t.b = 25 ∧ t.c = 17} =
  {t : SpecialTriple | True} :=
sorry

end NUMINAMATH_CALUDE_only_two_special_triples_l2703_270349


namespace NUMINAMATH_CALUDE_money_division_l2703_270332

theorem money_division (total : ℕ) (p q r : ℕ) : 
  p + q + r = total →
  p * 7 = q * 3 →
  q * 12 = r * 7 →
  q - p = 4000 →
  r - q = 5000 := by
sorry

end NUMINAMATH_CALUDE_money_division_l2703_270332


namespace NUMINAMATH_CALUDE_intersecting_digit_is_three_l2703_270328

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def powers_of_three : Set ℕ := {n | ∃ m : ℕ, n = 3^m ∧ is_three_digit n}
def powers_of_seven : Set ℕ := {n | ∃ m : ℕ, n = 7^m ∧ is_three_digit n}

theorem intersecting_digit_is_three :
  ∃! d : ℕ, d < 10 ∧ 
  (∃ n ∈ powers_of_three, ∃ i : ℕ, n / 10^i % 10 = d) ∧
  (∃ n ∈ powers_of_seven, ∃ i : ℕ, n / 10^i % 10 = d) :=
by sorry

end NUMINAMATH_CALUDE_intersecting_digit_is_three_l2703_270328


namespace NUMINAMATH_CALUDE_college_student_count_l2703_270380

/-- Represents the number of students in a college -/
structure College where
  boys : ℕ
  girls : ℕ

/-- The total number of students in the college -/
def College.total (c : College) : ℕ := c.boys + c.girls

/-- Theorem: In a college where the ratio of boys to girls is 8:5 and there are 190 girls, 
    the total number of students is 494 -/
theorem college_student_count (c : College) 
    (h1 : c.boys * 5 = c.girls * 8) 
    (h2 : c.girls = 190) : 
  c.total = 494 := by
  sorry

end NUMINAMATH_CALUDE_college_student_count_l2703_270380


namespace NUMINAMATH_CALUDE_valid_selections_count_l2703_270318

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → Point

/-- Represents a 3600-gon -/
def BigPolygon : RegularPolygon 3600 := sorry

/-- Represents a 72-gon formed by red vertices -/
def RedPolygon : RegularPolygon 72 := sorry

/-- Predicate to check if a vertex is red -/
def isRed (v : Fin 3600) : Prop := sorry

/-- Represents a selection of 40 vertices -/
def Selection : Finset (Fin 3600) := sorry

/-- Predicate to check if a selection forms a regular 40-gon -/
def isRegular40gon (s : Finset (Fin 3600)) : Prop := sorry

/-- The number of ways to select 40 non-red vertices forming a regular 40-gon -/
def validSelections : ℕ := sorry

theorem valid_selections_count : validSelections = 81 := by sorry

end NUMINAMATH_CALUDE_valid_selections_count_l2703_270318


namespace NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l2703_270310

/-- Acme T-Shirt Company's setup fee -/
def acme_setup : ℕ := 70

/-- Acme T-Shirt Company's per-shirt cost -/
def acme_per_shirt : ℕ := 11

/-- Beta T-Shirt Company's setup fee -/
def beta_setup : ℕ := 10

/-- Beta T-Shirt Company's per-shirt cost -/
def beta_per_shirt : ℕ := 15

/-- The minimum number of shirts for which Acme is cheaper than Beta -/
def min_shirts_for_acme : ℕ := 16

theorem acme_cheaper_at_min_shirts :
  acme_setup + acme_per_shirt * min_shirts_for_acme < 
  beta_setup + beta_per_shirt * min_shirts_for_acme ∧
  ∀ n : ℕ, n < min_shirts_for_acme → 
    acme_setup + acme_per_shirt * n ≥ beta_setup + beta_per_shirt * n :=
by sorry

end NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l2703_270310


namespace NUMINAMATH_CALUDE_power_fraction_equality_l2703_270362

theorem power_fraction_equality : (2^2015 + 2^2011) / (2^2015 - 2^2011) = 17/15 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l2703_270362


namespace NUMINAMATH_CALUDE_cone_surface_area_and_volume_l2703_270304

/-- Represents a cone with given height and sector angle -/
structure Cone where
  height : ℝ
  sectorAngle : ℝ

/-- Calculates the surface area of a cone -/
def surfaceArea (c : Cone) : ℝ := sorry

/-- Calculates the volume of a cone -/
def volume (c : Cone) : ℝ := sorry

/-- Theorem stating the surface area and volume of a specific cone -/
theorem cone_surface_area_and_volume :
  let c : Cone := { height := 12, sectorAngle := 100.8 * π / 180 }
  surfaceArea c = 56 * π ∧ volume c = 49 * π := by sorry

end NUMINAMATH_CALUDE_cone_surface_area_and_volume_l2703_270304


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l2703_270350

theorem smallest_common_multiple_of_6_and_15 : 
  ∃ b : ℕ+, (∀ m : ℕ+, (6 ∣ m) ∧ (15 ∣ m) → b ≤ m) ∧ (6 ∣ b) ∧ (15 ∣ b) ∧ b = 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l2703_270350


namespace NUMINAMATH_CALUDE_maya_total_pages_l2703_270302

/-- The number of books Maya read in the first week -/
def first_week_books : ℕ := 5

/-- The number of pages in each book Maya read in the first week -/
def first_week_pages_per_book : ℕ := 300

/-- The number of pages in each book Maya read in the second week -/
def second_week_pages_per_book : ℕ := 350

/-- The number of pages in each book Maya read in the third week -/
def third_week_pages_per_book : ℕ := 400

/-- The total number of pages Maya read over three weeks -/
def total_pages : ℕ :=
  (first_week_books * first_week_pages_per_book) +
  (2 * first_week_books * second_week_pages_per_book) +
  (3 * first_week_books * third_week_pages_per_book)

theorem maya_total_pages : total_pages = 11000 := by
  sorry

end NUMINAMATH_CALUDE_maya_total_pages_l2703_270302


namespace NUMINAMATH_CALUDE_evaluate_expression_l2703_270363

theorem evaluate_expression : -(18 / 3 * 7^2 - 80 + 4 * 7) = -242 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2703_270363


namespace NUMINAMATH_CALUDE_cost_of_goods_l2703_270330

/-- The cost of goods A, B, and C given certain conditions -/
theorem cost_of_goods (x y z : ℚ) 
  (h1 : 2*x + 4*y + z = 90)
  (h2 : 4*x + 10*y + z = 110) : 
  x + y + z = 80 := by
sorry

end NUMINAMATH_CALUDE_cost_of_goods_l2703_270330


namespace NUMINAMATH_CALUDE_complement_of_A_l2703_270355

def I : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 5}

theorem complement_of_A : (I \ A) = {2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2703_270355


namespace NUMINAMATH_CALUDE_shelbys_drive_l2703_270337

/-- Represents the weather conditions during Shelby's drive --/
inductive Weather
  | Sunny
  | Rainy
  | Foggy

/-- Shelby's driving scenario --/
structure DrivingScenario where
  speed : Weather → ℝ
  total_distance : ℝ
  total_time : ℝ
  time_in_weather : Weather → ℝ

/-- The theorem statement for Shelby's driving problem --/
theorem shelbys_drive (scenario : DrivingScenario) : 
  scenario.speed Weather.Sunny = 35 ∧ 
  scenario.speed Weather.Rainy = 25 ∧ 
  scenario.speed Weather.Foggy = 15 ∧ 
  scenario.total_distance = 19.5 ∧ 
  scenario.total_time = 45 ∧ 
  (scenario.time_in_weather Weather.Sunny + 
   scenario.time_in_weather Weather.Rainy + 
   scenario.time_in_weather Weather.Foggy = scenario.total_time) ∧
  (scenario.speed Weather.Sunny * scenario.time_in_weather Weather.Sunny / 60 +
   scenario.speed Weather.Rainy * scenario.time_in_weather Weather.Rainy / 60 +
   scenario.speed Weather.Foggy * scenario.time_in_weather Weather.Foggy / 60 = 
   scenario.total_distance) →
  scenario.time_in_weather Weather.Foggy = 10.25 := by
  sorry

end NUMINAMATH_CALUDE_shelbys_drive_l2703_270337


namespace NUMINAMATH_CALUDE_parabola_vertex_sum_max_l2703_270348

theorem parabola_vertex_sum_max (a T : ℤ) (h_T : T ≠ 0) : 
  let parabola (x : ℝ) := a * x * (x - 2 * T)
  let N := T - a * T^2
  (parabola 0 = 0) → 
  (parabola (2 * T) = 0) → 
  (parabola (T + 2) = 36) → 
  (∀ (a' T' : ℤ), T' ≠ 0 → 
    let parabola' (x : ℝ) := a' * x * (x - 2 * T')
    let N' := T' - a' * T'^2
    (parabola' 0 = 0) → 
    (parabola' (2 * T') = 0) → 
    (parabola' (T' + 2) = 36) → 
    N ≥ N') → 
  N = 37 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_sum_max_l2703_270348


namespace NUMINAMATH_CALUDE_multiplication_formula_l2703_270326

theorem multiplication_formula (x y z : ℝ) :
  (2*x + y + z) * (2*x - y - z) = 4*x^2 - y^2 - 2*y*z - z^2 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_formula_l2703_270326


namespace NUMINAMATH_CALUDE_circle_point_range_l2703_270319

theorem circle_point_range (a : ℝ) : 
  ((1 - a)^2 + (1 + a)^2 < 4) → (-1 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_point_range_l2703_270319


namespace NUMINAMATH_CALUDE_tangent_line_at_one_zero_l2703_270388

/-- The equation of the tangent line to y = x^3 - 2x + 1 at (1, 0) is y = x - 1 -/
theorem tangent_line_at_one_zero (x y : ℝ) : 
  (y = x^3 - 2*x + 1) → -- curve equation
  (1^3 - 2*1 + 1 = 0) → -- point (1, 0) lies on the curve
  (∀ t, (t - 1) * (3*1^2 - 2) = y - 0) → -- point-slope form of tangent line
  (y = x - 1) -- equation of tangent line
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_zero_l2703_270388


namespace NUMINAMATH_CALUDE_problem_solution_l2703_270322

-- Define the function f
def f (a b x : ℝ) : ℝ := |x - a| - |x + b|

-- Define the function g
def g (a b x : ℝ) : ℝ := -x^2 - a*x - b

theorem problem_solution (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (hmax : ∀ x, f a b x ≤ 3) 
  (hmax_achieved : ∃ x, f a b x = 3) 
  (hg_less_f : ∀ x ≥ a, g a b x < f a b x) :
  (a + b = 3) ∧ (1/2 < a ∧ a < 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2703_270322


namespace NUMINAMATH_CALUDE_right_isosceles_triangle_projection_l2703_270333

/-- Represents a right isosceles triangle -/
structure RightIsoscelesTriangle where
  side : ℝ
  right_angle : Bool
  isosceles : Bool

/-- Represents the projection of a triangle -/
def project (t : RightIsoscelesTriangle) (parallel : Bool) : RightIsoscelesTriangle :=
  if parallel then t else sorry

theorem right_isosceles_triangle_projection
  (t : RightIsoscelesTriangle)
  (h_side : t.side = 6)
  (h_right : t.right_angle = true)
  (h_isosceles : t.isosceles = true)
  (h_parallel : parallel = true) :
  let projected := project t parallel
  projected.side = 6 ∧
  projected.right_angle = true ∧
  projected.isosceles = true ∧
  Real.sqrt (2 * projected.side ^ 2) = 6 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_right_isosceles_triangle_projection_l2703_270333


namespace NUMINAMATH_CALUDE_snow_probability_first_week_february_l2703_270396

theorem snow_probability_first_week_february : 
  let prob_snow_first_three_days : ℚ := 1/4
  let prob_snow_next_four_days : ℚ := 1/3
  let days_in_week : ℕ := 7
  let first_period : ℕ := 3
  let second_period : ℕ := 4
  
  first_period + second_period = days_in_week →
  
  (1 - (1 - prob_snow_first_three_days)^first_period * 
       (1 - prob_snow_next_four_days)^second_period) = 11/12 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_first_week_february_l2703_270396


namespace NUMINAMATH_CALUDE_intersection_and_union_when_a_is_two_union_with_complement_equals_reals_iff_l2703_270393

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | x < a}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

-- Theorem for part (1)
theorem intersection_and_union_when_a_is_two :
  (A 2 ∩ B = {x : ℝ | 1 < x ∧ x < 2}) ∧
  (A 2 ∪ B = {x : ℝ | x < 3}) := by
sorry

-- Theorem for part (2)
theorem union_with_complement_equals_reals_iff (a : ℝ) :
  (A a ∪ (Set.univ \ B) = Set.univ) ↔ a ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_a_is_two_union_with_complement_equals_reals_iff_l2703_270393


namespace NUMINAMATH_CALUDE_peter_winning_strategy_l2703_270378

open Set

/-- Represents a point on a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle defined by three points -/
structure Triangle :=
  (a : Point)
  (b : Point)
  (c : Point)

/-- Represents a color (red or blue) -/
inductive Color
  | Red
  | Blue

/-- Function to check if two triangles are similar -/
def are_similar (t1 t2 : Triangle) : Prop :=
  sorry

/-- Function to check if all points in a set have the same color -/
def all_same_color (points : Set Point) (coloring : Point → Color) : Prop :=
  sorry

/-- Theorem stating that two points are sufficient for Peter's winning strategy -/
theorem peter_winning_strategy (original : Triangle) :
  ∃ (p1 p2 : Point), ∀ (coloring : Point → Color),
    ∃ (t : Triangle), are_similar t original ∧
      all_same_color {t.a, t.b, t.c} coloring :=
sorry

end NUMINAMATH_CALUDE_peter_winning_strategy_l2703_270378


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l2703_270370

def A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 3, -1; 1, -2, 5; 0, 6, 1]
def B : Matrix (Fin 3) (Fin 3) ℤ := !![1, 0, 4; 3, 2, -1; 0, 4, -2]
def C : Matrix (Fin 3) (Fin 3) ℤ := !![11, 2, 7; -5, 16, -4; 18, 16, -8]

theorem matrix_multiplication_result : A * B = C := by
  sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l2703_270370


namespace NUMINAMATH_CALUDE_go_and_chess_problem_l2703_270340

theorem go_and_chess_problem (x y z : ℝ) : 
  (3 * x + 5 * y = 98) →
  (8 * x + 3 * y = 158) →
  (z + (40 - z) = 40) →
  (16 * z + 10 * (40 - z) ≤ 550) →
  (x = 16 ∧ y = 10 ∧ z ≤ 25) := by
  sorry

end NUMINAMATH_CALUDE_go_and_chess_problem_l2703_270340


namespace NUMINAMATH_CALUDE_square_area_decrease_l2703_270374

theorem square_area_decrease (initial_side : ℝ) (decrease_percent : ℝ) : 
  initial_side = 9 ∧ decrease_percent = 20 →
  let new_side := initial_side * (1 - decrease_percent / 100)
  let initial_area := initial_side ^ 2
  let new_area := new_side ^ 2
  let area_decrease_percent := (initial_area - new_area) / initial_area * 100
  area_decrease_percent = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_decrease_l2703_270374


namespace NUMINAMATH_CALUDE_second_person_share_l2703_270375

/-- Represents the share of money for each person -/
structure Shares :=
  (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ)

/-- Theorem: Given a sum of money distributed among four people in the proportion 6:3:5:4,
    where the third person gets 1000 more than the fourth, the second person's share is 3000 -/
theorem second_person_share
  (shares : Shares)
  (h1 : shares.a = 6 * shares.d)
  (h2 : shares.b = 3 * shares.d)
  (h3 : shares.c = 5 * shares.d)
  (h4 : shares.c = shares.d + 1000) :
  shares.b = 3000 :=
by
  sorry

end NUMINAMATH_CALUDE_second_person_share_l2703_270375


namespace NUMINAMATH_CALUDE_min_value_of_z_l2703_270385

/-- Given a system of linear inequalities, prove that the minimum value of z = 2x + y is 4 -/
theorem min_value_of_z (x y : ℝ) 
  (h1 : 2 * x - y ≥ 0) 
  (h2 : x + y - 3 ≥ 0) 
  (h3 : y - x ≥ 0) : 
  ∃ (z : ℝ), z = 2 * x + y ∧ z ≥ 4 ∧ ∀ (w : ℝ), w = 2 * x + y → w ≥ z :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_z_l2703_270385


namespace NUMINAMATH_CALUDE_triangle_properties_l2703_270321

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  (t.a + t.c) / (t.a + t.b) = (t.b - t.a) / t.c ∧
  t.b = Real.sqrt 14 ∧
  Real.sin t.A = 2 * Real.sin t.C

-- State the theorem
theorem triangle_properties (t : Triangle) (h : satisfiesConditions t) :
  t.B = 2 * Real.pi / 3 ∧ min t.a (min t.b t.c) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2703_270321


namespace NUMINAMATH_CALUDE_club_membership_theorem_l2703_270314

theorem club_membership_theorem :
  ∃ n : ℕ, n ≥ 300 ∧ n % 8 = 0 ∧ n % 9 = 0 ∧ n % 11 = 0 ∧
  ∀ m : ℕ, m ≥ 300 ∧ m % 8 = 0 ∧ m % 9 = 0 ∧ m % 11 = 0 → m ≥ n :=
by
  use 792
  sorry

end NUMINAMATH_CALUDE_club_membership_theorem_l2703_270314


namespace NUMINAMATH_CALUDE_divided_volumes_theorem_l2703_270354

/-- Regular triangular prism with base side length 2√14 -/
structure RegularTriangularPrism where
  base_side : ℝ
  height : ℝ
  base_side_eq : base_side = 2 * Real.sqrt 14

/-- Plane dividing the prism -/
structure DividingPlane where
  prism : RegularTriangularPrism
  parallel_to_diagonal : Bool
  passes_through_vertex : Bool
  passes_through_center : Bool
  cross_section_area : ℝ
  cross_section_area_eq : cross_section_area = 21

/-- Volumes of the parts created by the dividing plane -/
def divided_volumes (p : RegularTriangularPrism) (d : DividingPlane) : (ℝ × ℝ) := sorry

/-- Theorem stating the volumes of the divided parts -/
theorem divided_volumes_theorem (p : RegularTriangularPrism) (d : DividingPlane) :
  d.prism = p → divided_volumes p d = (112/3, 154/3) := by sorry

end NUMINAMATH_CALUDE_divided_volumes_theorem_l2703_270354


namespace NUMINAMATH_CALUDE_cubic_equation_with_double_root_l2703_270335

/-- Given a cubic equation 2x^3 + 9x^2 - 117x + k = 0 where two roots are equal and k is positive,
    prove that k = 47050/216 -/
theorem cubic_equation_with_double_root (k : ℝ) : 
  (∃ x y : ℝ, (2 * x^3 + 9 * x^2 - 117 * x + k = 0) ∧ 
               (2 * y^3 + 9 * y^2 - 117 * y + k = 0) ∧
               (x ≠ y)) ∧
  (∃ z : ℝ, (2 * z^3 + 9 * z^2 - 117 * z + k = 0) ∧
            (∃ w : ℝ, w ≠ z ∧ 2 * w^3 + 9 * w^2 - 117 * w + k = 0)) ∧
  (k > 0) →
  k = 47050 / 216 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_with_double_root_l2703_270335


namespace NUMINAMATH_CALUDE_saras_sister_notebooks_l2703_270373

/-- Calculates the final number of notebooks given initial, ordered, and lost quantities. -/
def final_notebooks (initial ordered lost : ℕ) : ℕ :=
  initial + ordered - lost

/-- Theorem stating that Sara's sister's final number of notebooks is 8. -/
theorem saras_sister_notebooks : final_notebooks 4 6 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_saras_sister_notebooks_l2703_270373


namespace NUMINAMATH_CALUDE_amy_height_l2703_270398

def angela_height : ℕ := 157
def angela_helen_diff : ℕ := 4
def helen_amy_diff : ℕ := 3

theorem amy_height :
  ∃ (helen_height amy_height : ℕ),
    helen_height = angela_height - angela_helen_diff ∧
    amy_height = helen_height - helen_amy_diff ∧
    amy_height = 150 := by
  sorry

end NUMINAMATH_CALUDE_amy_height_l2703_270398


namespace NUMINAMATH_CALUDE_factorial_difference_l2703_270389

theorem factorial_difference : Nat.factorial 9 - Nat.factorial 8 = 322560 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l2703_270389


namespace NUMINAMATH_CALUDE_asha_win_probability_l2703_270305

theorem asha_win_probability (lose_prob : ℚ) (win_prob : ℚ) : 
  lose_prob = 7/12 → win_prob + lose_prob = 1 → win_prob = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_asha_win_probability_l2703_270305


namespace NUMINAMATH_CALUDE_max_value_2x_plus_y_l2703_270382

theorem max_value_2x_plus_y (x y : ℝ) (h1 : x + 2*y ≤ 3) (h2 : x ≥ 0) (h3 : y ≥ 0) :
  ∃ (max : ℝ), max = 6 ∧ ∀ (x' y' : ℝ), x' + 2*y' ≤ 3 → x' ≥ 0 → y' ≥ 0 → 2*x' + y' ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_2x_plus_y_l2703_270382


namespace NUMINAMATH_CALUDE_tangent_slope_acute_implies_a_equals_one_l2703_270324

/-- Given a curve C: y = x^3 - 2ax^2 + 2ax, if the slope of the tangent line
    at any point on the curve is acute, then a = 1, where a is an integer. -/
theorem tangent_slope_acute_implies_a_equals_one (a : ℤ) : 
  (∀ x : ℝ, 0 < 3*x^2 - 4*a*x + 2*a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_acute_implies_a_equals_one_l2703_270324


namespace NUMINAMATH_CALUDE_scientific_notation_of_error_l2703_270341

theorem scientific_notation_of_error : ∃ (a : ℝ) (n : ℤ), 
  0.0000003 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3 ∧ n = -7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_error_l2703_270341


namespace NUMINAMATH_CALUDE_f_inverse_a_eq_28_l2703_270300

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then x^(1/3)
  else if x ≥ 1 then 4*(x-1)
  else 0  -- undefined for x ≤ 0

theorem f_inverse_a_eq_28 (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : f a = f (a + 1)) :
  f (1 / a) = 28 := by
  sorry

end NUMINAMATH_CALUDE_f_inverse_a_eq_28_l2703_270300


namespace NUMINAMATH_CALUDE_ellipse_focus_distance_l2703_270353

/-- An ellipse in the first quadrant tangent to both axes with foci at (4,8) and (d,8) -/
structure Ellipse where
  d : ℝ
  tangent_to_axes : Bool
  in_first_quadrant : Bool
  focus1 : ℝ × ℝ := (4, 8)
  focus2 : ℝ × ℝ := (d, 8)

/-- The theorem stating that d = 30 for the given ellipse -/
theorem ellipse_focus_distance (e : Ellipse) (h1 : e.tangent_to_axes) (h2 : e.in_first_quadrant) :
  e.d = 30 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_focus_distance_l2703_270353
