import Mathlib

namespace NUMINAMATH_CALUDE_distance_to_karasuk_proof_l1124_112453

/-- The distance from Novosibirsk to Karasuk -/
def distance_to_karasuk : ℝ := 140

/-- The speed of the bus -/
def bus_speed : ℝ := 1

/-- The speed of the car -/
def car_speed : ℝ := 2 * bus_speed

/-- The initial distance the bus traveled before the car started -/
def initial_bus_distance : ℝ := 70

/-- The distance the bus traveled after Karasuk -/
def bus_distance_after_karasuk : ℝ := 20

/-- The distance the car traveled after Karasuk -/
def car_distance_after_karasuk : ℝ := 40

theorem distance_to_karasuk_proof :
  distance_to_karasuk = initial_bus_distance + 
    (car_distance_after_karasuk * bus_speed / car_speed) :=
by sorry

end NUMINAMATH_CALUDE_distance_to_karasuk_proof_l1124_112453


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1124_112405

theorem geometric_sequence_problem :
  ∀ (a r : ℝ),
    (a * r^2 - a = 48) →
    ((a^2 * r^4 - a^2) / (a^2 + a^2 * r^2 + a^2 * r^4) = 208/217) →
    ((a = 2 ∧ r = 5) ∨
     (a = 2 ∧ r = -5) ∨
     (a = -216/13 ∧ r = Complex.I * (Real.sqrt 17 / 3)) ∨
     (a = -216/13 ∧ r = -Complex.I * (Real.sqrt 17 / 3))) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1124_112405


namespace NUMINAMATH_CALUDE_work_completion_time_l1124_112473

theorem work_completion_time (b a_and_b : ℝ) (hb : b = 8) (hab : a_and_b = 4.8) :
  let a := (1 / a_and_b - 1 / b)⁻¹
  a = 12 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1124_112473


namespace NUMINAMATH_CALUDE_ellie_wide_reflections_count_l1124_112481

/-- The number of times Sarah sees her reflection in tall mirror rooms -/
def sarah_tall_reflections : ℕ := 10

/-- The number of times Sarah sees her reflection in wide mirror rooms -/
def sarah_wide_reflections : ℕ := 5

/-- The number of times Ellie sees her reflection in tall mirror rooms -/
def ellie_tall_reflections : ℕ := 6

/-- The number of times both Sarah and Ellie passed through tall mirror rooms -/
def tall_room_visits : ℕ := 3

/-- The number of times both Sarah and Ellie passed through wide mirror rooms -/
def wide_room_visits : ℕ := 5

/-- The total number of reflections for both Sarah and Ellie -/
def total_reflections : ℕ := 88

/-- The number of times Ellie sees her reflection in wide mirror rooms -/
def ellie_wide_reflections : ℕ := 3

theorem ellie_wide_reflections_count :
  sarah_tall_reflections * tall_room_visits +
  sarah_wide_reflections * wide_room_visits +
  ellie_tall_reflections * tall_room_visits +
  ellie_wide_reflections * wide_room_visits = total_reflections :=
by sorry

end NUMINAMATH_CALUDE_ellie_wide_reflections_count_l1124_112481


namespace NUMINAMATH_CALUDE_excess_purchase_l1124_112436

/-- Calculates the excess amount of Chinese herbal medicine purchased given the planned amount and completion percentages -/
theorem excess_purchase (planned_amount : ℝ) (first_half_percent : ℝ) (second_half_percent : ℝ) 
  (h1 : planned_amount = 1500)
  (h2 : first_half_percent = 55)
  (h3 : second_half_percent = 65) :
  (first_half_percent + second_half_percent - 100) / 100 * planned_amount = 300 := by
  sorry

end NUMINAMATH_CALUDE_excess_purchase_l1124_112436


namespace NUMINAMATH_CALUDE_symmetric_function_product_l1124_112493

/-- A function f(x) that is symmetric about the line x = 2 -/
def f (a b : ℝ) (x : ℝ) : ℝ := (x^2 - 1) * (-x^2 + a*x - b)

/-- The symmetry condition for f(x) about x = 2 -/
def is_symmetric (a b : ℝ) : Prop :=
  ∀ x : ℝ, f a b (2 - x) = f a b (2 + x)

/-- Theorem: If f(x) is symmetric about x = 2, then ab = 120 -/
theorem symmetric_function_product (a b : ℝ) :
  is_symmetric a b → a * b = 120 := by sorry

end NUMINAMATH_CALUDE_symmetric_function_product_l1124_112493


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_six_l1124_112448

theorem at_least_one_not_less_than_six (a b : ℝ) (h : a + b = 12) : max a b ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_six_l1124_112448


namespace NUMINAMATH_CALUDE_john_roommates_l1124_112420

theorem john_roommates (bob_roommates : ℕ) (h1 : bob_roommates = 10) :
  let john_roommates := 2 * bob_roommates + 5
  john_roommates = 25 := by sorry

end NUMINAMATH_CALUDE_john_roommates_l1124_112420


namespace NUMINAMATH_CALUDE_sum_denominator_power_of_two_l1124_112447

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

def sum_series : ℕ → ℚ
  | 0 => 0
  | n + 1 => sum_series n + (double_factorial (2 * (n + 1) - 1) : ℚ) / (double_factorial (2 * (n + 1)) : ℚ)

theorem sum_denominator_power_of_two : 
  ∃ (numerator : ℕ), sum_series 11 = (numerator : ℚ) / 2^8 := by sorry

end NUMINAMATH_CALUDE_sum_denominator_power_of_two_l1124_112447


namespace NUMINAMATH_CALUDE_buying_more_can_cost_less_buying_101_is_cheaper_l1124_112435

/-- The cost function for notebooks -/
def notebook_cost (n : ℕ) : ℝ :=
  if n ≤ 100 then 2.3 * n else 2.2 * n

theorem buying_more_can_cost_less :
  ∃ (n₁ n₂ : ℕ), n₁ < n₂ ∧ notebook_cost n₁ > notebook_cost n₂ :=
sorry

theorem buying_101_is_cheaper :
  notebook_cost 101 < notebook_cost 100 :=
sorry

end NUMINAMATH_CALUDE_buying_more_can_cost_less_buying_101_is_cheaper_l1124_112435


namespace NUMINAMATH_CALUDE_younger_person_age_l1124_112464

/-- Proves that the younger person's age is 8 years, given the conditions of the problem. -/
theorem younger_person_age (y e : ℕ) : 
  e = y + 12 →  -- The elder person's age is 12 years more than the younger person's
  e - 5 = 5 * (y - 5) →  -- Five years ago, the elder was 5 times as old as the younger
  y = 8 :=  -- The younger person's present age is 8 years
by sorry

end NUMINAMATH_CALUDE_younger_person_age_l1124_112464


namespace NUMINAMATH_CALUDE_problem_solution_l1124_112412

-- Define the function f
def f (a x : ℝ) : ℝ := |a - x|

-- Define the set A
def A : Set ℝ := {x | f (3/2) (2*x - 3/2) > 2 * f (3/2) (x + 2) + 2}

theorem problem_solution :
  (A = Set.Iio 0) ∧
  (∀ x₀ ∈ A, ∀ x : ℝ, f (3/2) (x₀ * x) ≥ x₀ * f (3/2) x + f (3/2) ((3/2) * x₀)) := by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l1124_112412


namespace NUMINAMATH_CALUDE_min_positive_temperatures_l1124_112439

theorem min_positive_temperatures (x : ℕ) (y : ℕ) : 
  x * (x - 1) = 110 → 
  y * (y - 1) + (x - y) * (x - 1 - y) = 50 → 
  y ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_min_positive_temperatures_l1124_112439


namespace NUMINAMATH_CALUDE_system_solutions_l1124_112426

-- Define the system of equations
def system (t x y z : ℝ) : Prop :=
  t * (x + y + z) = 0 ∧ t * (x + y) + z = 1 ∧ t * x + y + z = 2

-- State the theorem
theorem system_solutions :
  ∀ t x y z : ℝ,
    (t = 0 → system t x y z ↔ y = 1 ∧ z = 1) ∧
    (t ≠ 0 ∧ t ≠ 1 → system t x y z ↔ x = 2 / (t - 1) ∧ y = -1 / (t - 1) ∧ z = -1 / (t - 1)) ∧
    (t = 1 → ¬∃ x y z, system t x y z) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l1124_112426


namespace NUMINAMATH_CALUDE_specific_gold_cube_profit_l1124_112476

/-- Calculates the profit from selling a gold cube -/
def gold_cube_profit (side_length : ℝ) (density : ℝ) (purchase_price : ℝ) (markup : ℝ) : ℝ :=
  let volume := side_length ^ 3
  let mass := density * volume
  let cost := mass * purchase_price
  let selling_price := cost * markup
  selling_price - cost

/-- Theorem stating the profit for a specific gold cube -/
theorem specific_gold_cube_profit :
  gold_cube_profit 6 19 60 1.5 = 123120 := by
  sorry

end NUMINAMATH_CALUDE_specific_gold_cube_profit_l1124_112476


namespace NUMINAMATH_CALUDE_simultaneous_integers_l1124_112421

theorem simultaneous_integers (x : ℤ) :
  (∃ y z u : ℤ, (x - 3) = 7 * y ∧ (x - 2) = 5 * z ∧ (x - 4) = 3 * u) ↔
  (∃ t : ℤ, x = 105 * t + 52) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_integers_l1124_112421


namespace NUMINAMATH_CALUDE_distribution_count_theorem_l1124_112450

/-- Represents a boat with its capacity -/
structure Boat where
  capacity : Nat

/-- Represents the distribution of people on boats -/
structure Distribution where
  adults : Nat
  children : Nat

/-- Checks if a distribution is valid (i.e., has an adult if there's a child) -/
def is_valid_distribution (d : Distribution) : Bool :=
  d.children > 0 → d.adults > 0

/-- Counts the number of valid ways to distribute people on boats -/
def count_valid_distributions (boats : List Boat) (total_adults total_children : Nat) : Nat :=
  sorry -- The actual implementation would go here

/-- The main theorem to prove -/
theorem distribution_count_theorem :
  let boats := [Boat.mk 3, Boat.mk 2, Boat.mk 1]
  count_valid_distributions boats 3 2 = 33 := by
  sorry

#check distribution_count_theorem

end NUMINAMATH_CALUDE_distribution_count_theorem_l1124_112450


namespace NUMINAMATH_CALUDE_locus_and_line_equations_l1124_112461

-- Define the circles F₁ and F₂
def F₁ (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4
def F₂ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 36

-- Define the locus C
def C (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1 ∧ x ≠ -4

-- Define the line l
def l (x y : ℝ) : Prop := 3 * x - 2 * y - 8 = 0

-- Define the point Q
def Q : ℝ × ℝ := (2, -1)

-- Theorem statement
theorem locus_and_line_equations :
  ∃ (M : ℝ × ℝ → Prop),
    (∀ x y, M (x, y) → F₁ x y) ∧
    (∀ x y, M (x, y) → F₂ x y) ∧
    (∀ x y, C x y ↔ ∃ r > 0, M (x, y) ∧ r = 2) ∧
    (∃ A B : ℝ × ℝ, C A.1 A.2 ∧ C B.1 B.2 ∧ l A.1 A.2 ∧ l B.1 B.2 ∧ Q = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :=
sorry

end NUMINAMATH_CALUDE_locus_and_line_equations_l1124_112461


namespace NUMINAMATH_CALUDE_triangle_area_l1124_112443

theorem triangle_area (A B C : ℝ) (a b c : ℝ) : 
  b * Real.cos C = 3 * a * Real.cos B - c * Real.cos B →
  a * c * Real.cos B = 2 →
  (1/2) * a * c * Real.sin B = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l1124_112443


namespace NUMINAMATH_CALUDE_sqrt_square_abs_l1124_112484

theorem sqrt_square_abs (x : ℝ) : Real.sqrt (x ^ 2) = |x| := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_abs_l1124_112484


namespace NUMINAMATH_CALUDE_problem_statement_l1124_112424

theorem problem_statement : (-12 : ℚ) * ((2 : ℚ) / 3 - (1 : ℚ) / 4 + (1 : ℚ) / 6) = -7 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1124_112424


namespace NUMINAMATH_CALUDE_cookout_2006_l1124_112417

/-- The number of kids at the cookout in 2004 -/
def kids_2004 : ℕ := 60

/-- The number of kids at the cookout in 2005 -/
def kids_2005 : ℕ := kids_2004 / 2

/-- The number of kids at the cookout in 2006 -/
def kids_2006 : ℕ := kids_2005 * 2 / 3

/-- Theorem stating that the number of kids at the cookout in 2006 is 20 -/
theorem cookout_2006 : kids_2006 = 20 := by
  sorry

end NUMINAMATH_CALUDE_cookout_2006_l1124_112417


namespace NUMINAMATH_CALUDE_difference_of_squares_l1124_112487

theorem difference_of_squares (x y : ℝ) 
  (h1 : x + y = 20) 
  (h2 : x - y = 6) : 
  x^2 - y^2 = 120 := by
sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1124_112487


namespace NUMINAMATH_CALUDE_smallest_valid_N_exists_l1124_112477

def is_valid_configuration (N : ℕ) (c₁ c₂ c₃ c₄ c₅ c₆ : ℕ) : Prop :=
  c₁ ≤ N ∧ c₂ ≤ N ∧ c₃ ≤ N ∧ c₄ ≤ N ∧ c₅ ≤ N ∧ c₆ ≤ N ∧
  c₁ = 6 * c₂ - 1 ∧
  N + c₂ = 6 * c₃ - 2 ∧
  2 * N + c₃ = 6 * c₄ - 3 ∧
  3 * N + c₄ = 6 * c₅ - 4 ∧
  4 * N + c₅ = 6 * c₆ - 5 ∧
  5 * N + c₆ = 6 * c₁

theorem smallest_valid_N_exists :
  ∃ N : ℕ, N > 0 ∧ 
  (∃ c₁ c₂ c₃ c₄ c₅ c₆ : ℕ, is_valid_configuration N c₁ c₂ c₃ c₄ c₅ c₆) ∧
  (∀ M : ℕ, M < N → ¬∃ c₁ c₂ c₃ c₄ c₅ c₆ : ℕ, is_valid_configuration M c₁ c₂ c₃ c₄ c₅ c₆) :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_N_exists_l1124_112477


namespace NUMINAMATH_CALUDE_trapezoid_garden_bases_l1124_112437

theorem trapezoid_garden_bases :
  let area : ℕ := 1350
  let altitude : ℕ := 45
  let valid_pair (b₁ b₂ : ℕ) : Prop :=
    area = (altitude * (b₁ + b₂)) / 2 ∧
    b₁ % 9 = 0 ∧
    b₂ % 9 = 0 ∧
    b₁ > 0 ∧
    b₂ > 0
  ∃! (pairs : Finset (ℕ × ℕ)), pairs.card = 3 ∧ ∀ (p : ℕ × ℕ), p ∈ pairs ↔ valid_pair p.1 p.2 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_garden_bases_l1124_112437


namespace NUMINAMATH_CALUDE_range_of_fraction_l1124_112459

theorem range_of_fraction (a b : ℝ) 
  (ha : 0 < a ∧ a ≤ 2) 
  (hb : b ≥ 1)
  (hba : b ≤ a^2) :
  ∃ (t : ℝ), t = b / a ∧ 1/2 ≤ t ∧ t ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_fraction_l1124_112459


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1124_112492

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 5 + 2) :
  (1 / (a + 2)) / ((a^2 - 4*a + 4) / (a^2 - 4)) - 2 / (a - 2) = -(Real.sqrt 5 / 5) := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1124_112492


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l1124_112428

theorem trig_expression_equals_one :
  let numerator := Real.sin (18 * π / 180) * Real.cos (12 * π / 180) + 
                   Real.cos (162 * π / 180) * Real.cos (102 * π / 180)
  let denominator := Real.sin (22 * π / 180) * Real.cos (8 * π / 180) + 
                     Real.cos (158 * π / 180) * Real.cos (98 * π / 180)
  numerator / denominator = 1 := by
sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l1124_112428


namespace NUMINAMATH_CALUDE_average_age_of_population_l1124_112483

/-- The average age of a population given the ratio of men to women and their respective average ages -/
theorem average_age_of_population 
  (ratio_men_to_women : ℚ) 
  (avg_age_men : ℝ) 
  (avg_age_women : ℝ) :
  ratio_men_to_women = 2/3 →
  avg_age_men = 37 →
  avg_age_women = 42 →
  let total_population := ratio_men_to_women + 1
  let weighted_age_men := ratio_men_to_women * avg_age_men
  let weighted_age_women := 1 * avg_age_women
  (weighted_age_men + weighted_age_women) / total_population = 40 :=
by sorry


end NUMINAMATH_CALUDE_average_age_of_population_l1124_112483


namespace NUMINAMATH_CALUDE_cut_position_exists_carpenter_board_cut_position_l1124_112407

/-- Represents a linearly tapering board -/
structure TaperingBoard where
  length : ℝ
  widthA : ℝ
  widthB : ℝ

/-- The equation that determines the cut position for equal areas -/
def cutEquation (board : TaperingBoard) (x : ℝ) : Prop :=
  x^2 - (4 * board.length * x) + (board.length^2 * (2 - (board.widthA / board.widthB))) = 0

/-- Theorem stating the existence of a solution for the cut equation -/
theorem cut_position_exists (board : TaperingBoard) 
    (h1 : board.length > 0)
    (h2 : board.widthA > 0)
    (h3 : board.widthB > 0)
    (h4 : board.widthA < board.widthB) :
    ∃ x : ℝ, 0 < x ∧ x < board.length ∧ cutEquation board x := by
  sorry

/-- The specific board from the problem -/
def carpenterBoard : TaperingBoard :=
  { length := 120
  , widthA := 6
  , widthB := 12 }

/-- Theorem for the specific board in the problem -/
theorem carpenter_board_cut_position :
    ∃ x : ℝ, 0 < x ∧ x < 120 ∧ cutEquation carpenterBoard x := by
  sorry

end NUMINAMATH_CALUDE_cut_position_exists_carpenter_board_cut_position_l1124_112407


namespace NUMINAMATH_CALUDE_fourth_row_middle_cells_l1124_112438

/-- Represents a letter in the grid -/
inductive Letter : Type
| A | B | C | D | E | F

/-- Represents a position in the grid -/
structure Position :=
  (row : Fin 6)
  (col : Fin 6)

/-- Represents the 6x6 grid -/
def Grid := Position → Letter

/-- Checks if a 2x3 rectangle is valid (no repeats) -/
def validRectangle (g : Grid) (topLeft : Position) : Prop :=
  ∀ (i j : Fin 2) (k : Fin 3),
    g ⟨topLeft.row + i, topLeft.col + k⟩ ≠ g ⟨topLeft.row + j, topLeft.col + k⟩ ∨ i = j

/-- Checks if the entire grid is valid -/
def validGrid (g : Grid) : Prop :=
  (∀ r : Fin 6, ∀ i j : Fin 6, g ⟨r, i⟩ ≠ g ⟨r, j⟩ ∨ i = j) ∧  -- No repeats in rows
  (∀ c : Fin 6, ∀ i j : Fin 6, g ⟨i, c⟩ ≠ g ⟨j, c⟩ ∨ i = j) ∧  -- No repeats in columns
  (∀ r c : Fin 2, validRectangle g ⟨3*r, 3*c⟩)                 -- Valid 2x3 rectangles

/-- The main theorem -/
theorem fourth_row_middle_cells (g : Grid) (h : validGrid g) :
  g ⟨3, 1⟩ = Letter.E ∧
  g ⟨3, 2⟩ = Letter.D ∧
  g ⟨3, 3⟩ = Letter.C ∧
  g ⟨3, 4⟩ = Letter.F :=
by sorry

end NUMINAMATH_CALUDE_fourth_row_middle_cells_l1124_112438


namespace NUMINAMATH_CALUDE_frequency_count_theorem_l1124_112498

theorem frequency_count_theorem (sample_size : ℕ) (relative_frequency : ℝ) 
  (h1 : sample_size = 100) 
  (h2 : relative_frequency = 0.2) :
  (sample_size : ℝ) * relative_frequency = 20 := by
  sorry

end NUMINAMATH_CALUDE_frequency_count_theorem_l1124_112498


namespace NUMINAMATH_CALUDE_largest_even_digit_multiple_of_5_proof_l1124_112452

def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

def largest_even_digit_multiple_of_5 : ℕ := 8800

theorem largest_even_digit_multiple_of_5_proof :
  (has_only_even_digits largest_even_digit_multiple_of_5) ∧
  (largest_even_digit_multiple_of_5 < 10000) ∧
  (largest_even_digit_multiple_of_5 % 5 = 0) ∧
  (∀ n : ℕ, n > largest_even_digit_multiple_of_5 →
    ¬(has_only_even_digits n ∧ n < 10000 ∧ n % 5 = 0)) :=
by sorry

#check largest_even_digit_multiple_of_5_proof

end NUMINAMATH_CALUDE_largest_even_digit_multiple_of_5_proof_l1124_112452


namespace NUMINAMATH_CALUDE_expression_evaluation_l1124_112427

theorem expression_evaluation :
  (2 ^ 2010 * 3 ^ 2012 * 5 ^ 2) / 6 ^ 2011 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1124_112427


namespace NUMINAMATH_CALUDE_binary_to_hex_l1124_112431

-- Define the binary number
def binary_num : ℕ := 1011001

-- Define the hexadecimal number
def hex_num : ℕ := 0x59

-- Theorem stating that the binary number is equal to the hexadecimal number
theorem binary_to_hex : binary_num = hex_num := by
  sorry

end NUMINAMATH_CALUDE_binary_to_hex_l1124_112431


namespace NUMINAMATH_CALUDE_other_vehicle_wheels_l1124_112495

theorem other_vehicle_wheels (total_wheels : Nat) (four_wheelers : Nat) (h1 : total_wheels = 58) (h2 : four_wheelers = 14) :
  ∃ (other_wheels : Nat), other_wheels = 2 ∧ total_wheels = four_wheelers * 4 + other_wheels := by
sorry

end NUMINAMATH_CALUDE_other_vehicle_wheels_l1124_112495


namespace NUMINAMATH_CALUDE_circle_intersection_range_l1124_112469

-- Define the circles
def circle1 (x y m : ℝ) : Prop := x^2 + y^2 = m
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 8*y - 11 = 0

-- Define the intersection of the circles
def circles_intersect (m : ℝ) : Prop :=
  ∃ x y : ℝ, circle1 x y m ∧ circle2 x y

-- Theorem statement
theorem circle_intersection_range (m : ℝ) :
  circles_intersect m ↔ 1 < m ∧ m < 121 :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l1124_112469


namespace NUMINAMATH_CALUDE_range_of_a_l1124_112485

/-- Given two statements p and q, where p: x^2 - 8x - 20 < 0 and q: x^2 - 2x + 1 - a^2 ≤ 0 with a > 0,
    and ¬p is a necessary but not sufficient condition for ¬q,
    prove that the range of values for the real number a is [9, +∞). -/
theorem range_of_a (p q : ℝ → Prop) (a : ℝ) 
    (hp : ∀ x, p x ↔ x^2 - 8*x - 20 < 0)
    (hq : ∀ x, q x ↔ x^2 - 2*x + 1 - a^2 ≤ 0)
    (ha : a > 0)
    (hnec : ∀ x, ¬(p x) → ¬(q x))
    (hnsuff : ∃ x, ¬(q x) ∧ p x) :
  a ≥ 9 := by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_range_of_a_l1124_112485


namespace NUMINAMATH_CALUDE_quad_pair_f_one_l1124_112422

/-- Two quadratic polynomials satisfying specific conditions -/
structure QuadraticPair :=
  (f g : ℝ → ℝ)
  (quad_f : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)
  (quad_g : ∃ a b c : ℝ, ∀ x, g x = a * x^2 + b * x + c)
  (h1 : f 2 = 2 ∧ f 3 = 2)
  (h2 : g 2 = 2 ∧ g 3 = 2)
  (h3 : g 1 = 3)
  (h4 : f 4 = 7)
  (h5 : g 4 = 4)

/-- The main theorem stating that f(1) = 7 for the given conditions -/
theorem quad_pair_f_one (qp : QuadraticPair) : qp.f 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_quad_pair_f_one_l1124_112422


namespace NUMINAMATH_CALUDE_trig_identities_l1124_112444

theorem trig_identities (α β γ : Real) (h : α + β + γ = Real.pi) :
  (Real.sin α)^2 + (Real.sin β)^2 - (Real.sin γ)^2 = 2 * Real.sin α * Real.sin β * Real.cos γ ∧
  (Real.cos α)^2 + (Real.cos β)^2 - (Real.cos γ)^2 = 1 - 2 * Real.sin α * Real.sin β * Real.cos γ := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l1124_112444


namespace NUMINAMATH_CALUDE_matrix_inverse_and_transformation_l1124_112414

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 1, 2]

theorem matrix_inverse_and_transformation :
  let A_inv : Matrix (Fin 2) (Fin 2) ℝ := !![2, -3; -1, 2]
  let P : Fin 2 → ℝ := ![3, -1]
  (A⁻¹ = A_inv) ∧ (A.mulVec P = ![3, 1]) := by sorry

end NUMINAMATH_CALUDE_matrix_inverse_and_transformation_l1124_112414


namespace NUMINAMATH_CALUDE_rotten_eggs_probability_l1124_112457

/-- The probability of selecting 2 rotten eggs from a pack of 36 eggs containing 3 rotten eggs -/
theorem rotten_eggs_probability (total_eggs : ℕ) (rotten_eggs : ℕ) (selected_eggs : ℕ) : 
  total_eggs = 36 → rotten_eggs = 3 → selected_eggs = 2 →
  (Nat.choose rotten_eggs selected_eggs : ℚ) / (Nat.choose total_eggs selected_eggs) = 1 / 420 :=
by sorry

end NUMINAMATH_CALUDE_rotten_eggs_probability_l1124_112457


namespace NUMINAMATH_CALUDE_regular_decagon_angles_l1124_112451

/-- Properties of a regular decagon -/
theorem regular_decagon_angles :
  let n : ℕ := 10  -- number of sides in a decagon
  let exterior_angle : ℝ := 360 / n
  let interior_angle : ℝ := (n - 2) * 180 / n
  exterior_angle = 36 ∧ interior_angle = 144 := by
  sorry

end NUMINAMATH_CALUDE_regular_decagon_angles_l1124_112451


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l1124_112468

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 6) (h2 : x * y = -2) :
  1 / x + 1 / y = -3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l1124_112468


namespace NUMINAMATH_CALUDE_jill_shopping_tax_percentage_l1124_112423

/-- Calculates the total tax percentage given spending percentages and tax rates -/
def total_tax_percentage (clothing_percent : ℝ) (food_percent : ℝ) (other_percent : ℝ)
                         (clothing_tax_rate : ℝ) (food_tax_rate : ℝ) (other_tax_rate : ℝ) : ℝ :=
  (clothing_percent * clothing_tax_rate + food_percent * food_tax_rate + other_percent * other_tax_rate) * 100

/-- The total tax percentage for Jill's shopping trip -/
theorem jill_shopping_tax_percentage :
  total_tax_percentage 0.50 0.10 0.40 0.04 0 0.08 = 5.20 := by
  sorry

#eval total_tax_percentage 0.50 0.10 0.40 0.04 0 0.08

end NUMINAMATH_CALUDE_jill_shopping_tax_percentage_l1124_112423


namespace NUMINAMATH_CALUDE_stating_downstream_speed_l1124_112474

/-- Represents the rowing speeds of a man in different conditions. -/
structure RowingSpeeds where
  upstream : ℝ
  still_water : ℝ
  downstream : ℝ

/-- 
Theorem stating that given a man's upstream rowing speed and still water speed,
we can determine his downstream speed.
-/
theorem downstream_speed (speeds : RowingSpeeds) 
  (h_upstream : speeds.upstream = 7)
  (h_still_water : speeds.still_water = 20)
  (h_average : speeds.still_water = (speeds.upstream + speeds.downstream) / 2) :
  speeds.downstream = 33 := by
  sorry

#check downstream_speed

end NUMINAMATH_CALUDE_stating_downstream_speed_l1124_112474


namespace NUMINAMATH_CALUDE_perp_to_countless_lines_necessary_not_sufficient_l1124_112404

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Perpendicularity between a line and a plane -/
def perp_line_plane (l : Line3D) (α : Plane3D) : Prop := sorry

/-- A line is perpendicular to countless lines within a plane -/
def perp_to_countless_lines (l : Line3D) (α : Plane3D) : Prop := sorry

/-- Main theorem: The statement "Line l is perpendicular to countless lines within plane α" 
    is a necessary but not sufficient condition for "l ⊥ α" -/
theorem perp_to_countless_lines_necessary_not_sufficient (l : Line3D) (α : Plane3D) :
  (perp_line_plane l α → perp_to_countless_lines l α) ∧
  ∃ l' α', perp_to_countless_lines l' α' ∧ ¬perp_line_plane l' α' := by
  sorry

end NUMINAMATH_CALUDE_perp_to_countless_lines_necessary_not_sufficient_l1124_112404


namespace NUMINAMATH_CALUDE_friendly_group_has_complete_subgroup_l1124_112479

/-- Represents the property of two people knowing each other -/
def knows (people : Type) : people → people → Prop := sorry

/-- A group of people satisfying the condition that among any three, two know each other -/
structure FriendlyGroup (people : Type) where
  size : Nat
  members : Finset people
  size_eq : members.card = size
  friendly : ∀ (a b c : people), a ∈ members → b ∈ members → c ∈ members →
    a ≠ b → b ≠ c → a ≠ c → (knows people a b ∨ knows people b c ∨ knows people a c)

/-- A complete subgroup where every pair knows each other -/
def CompleteSubgroup {people : Type} (group : FriendlyGroup people) (subgroup : Finset people) : Prop :=
  subgroup ⊆ group.members ∧ ∀ (a b : people), a ∈ subgroup → b ∈ subgroup → a ≠ b → knows people a b

/-- The main theorem: In a group of 9 people satisfying the friendly condition,
    there exists a complete subgroup of 4 people -/
theorem friendly_group_has_complete_subgroup 
  {people : Type} (group : FriendlyGroup people) (h : group.size = 9) :
  ∃ (subgroup : Finset people), subgroup.card = 4 ∧ CompleteSubgroup group subgroup := by
  sorry

end NUMINAMATH_CALUDE_friendly_group_has_complete_subgroup_l1124_112479


namespace NUMINAMATH_CALUDE_mismatched_boots_count_l1124_112425

/-- The number of ways to select k items from n items --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to arrange n distinct items --/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of pairs of boots --/
def num_pairs : ℕ := 6

/-- The number of ways two people can wear mismatched boots --/
def mismatched_boots_ways : ℕ :=
  -- Case 1: Using boots from two pairs
  choose num_pairs 2 * 4 +
  -- Case 2: Using boots from three pairs
  choose num_pairs 3 * 4 * 4 +
  -- Case 3: Using boots from four pairs
  choose num_pairs 4 * factorial 4

theorem mismatched_boots_count :
  mismatched_boots_ways = 740 := by sorry

end NUMINAMATH_CALUDE_mismatched_boots_count_l1124_112425


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1124_112456

theorem trigonometric_equation_solution (x : ℝ) 
  (h_eq : 8.459 * (Real.cos (x^2))^2 * (Real.tan (x^2) + 2 * Real.tan x) + 
          (Real.tan x)^3 * (1 - (Real.sin (x^2))^2) * (2 - Real.tan x * Real.tan (x^2)) = 0)
  (h_cos : Real.cos x ≠ 0)
  (h_x_sq : ∀ n : ℤ, x^2 ≠ Real.pi/2 + Real.pi * n)
  (h_x_1 : ∀ m : ℤ, x ≠ Real.pi/4 + Real.pi * m/2)
  (h_x_2 : ∀ l : ℤ, x ≠ Real.pi/2 + Real.pi * l) :
  ∃ k : ℕ, x = -1 + Real.sqrt (Real.pi * k + 1) ∨ x = -1 - Real.sqrt (Real.pi * k + 1) :=
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1124_112456


namespace NUMINAMATH_CALUDE_percentage_commutation_l1124_112411

theorem percentage_commutation (x : ℝ) : 
  (0.4 * (0.3 * x) = 24) → (0.3 * (0.4 * x) = 24) := by
  sorry

end NUMINAMATH_CALUDE_percentage_commutation_l1124_112411


namespace NUMINAMATH_CALUDE_jack_bike_percentage_l1124_112478

def original_paycheck : ℝ := 125
def tax_rate : ℝ := 0.20
def savings_amount : ℝ := 20

theorem jack_bike_percentage :
  let after_tax := original_paycheck * (1 - tax_rate)
  let remaining := after_tax - savings_amount
  let bike_percentage := (remaining / after_tax) * 100
  bike_percentage = 80 := by sorry

end NUMINAMATH_CALUDE_jack_bike_percentage_l1124_112478


namespace NUMINAMATH_CALUDE_negative_one_third_m_meets_requirements_l1124_112409

/-- Represents an algebraic expression -/
inductive AlgebraicExpression
  | Fraction : ℚ → String → AlgebraicExpression
  | Mixed : ℕ → ℚ → String → AlgebraicExpression
  | Division : String → String → AlgebraicExpression
  | Multiplication : String → ℕ → AlgebraicExpression

/-- Checks if an algebraic expression meets the writing requirements -/
def meetsWritingRequirements (expr : AlgebraicExpression) : Prop :=
  match expr with
  | AlgebraicExpression.Fraction _ _ => true
  | _ => false

/-- The theorem stating that -1/3m meets the writing requirements -/
theorem negative_one_third_m_meets_requirements :
  meetsWritingRequirements (AlgebraicExpression.Fraction (-1/3) "m") :=
by sorry

end NUMINAMATH_CALUDE_negative_one_third_m_meets_requirements_l1124_112409


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1124_112486

theorem rationalize_denominator : 
  7 / Real.sqrt 75 = (7 * Real.sqrt 3) / 15 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1124_112486


namespace NUMINAMATH_CALUDE_repeating_decimal_division_l1124_112470

theorem repeating_decimal_division :
  let a : ℚ := 54 / 99
  let b : ℚ := 18 / 99
  a / b = 3 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_division_l1124_112470


namespace NUMINAMATH_CALUDE_smallest_number_divisible_after_subtraction_l1124_112401

theorem smallest_number_divisible_after_subtraction : ∃ (n : ℕ), 
  (∀ (m : ℕ), m > 0 → (m - 8 : ℤ) % 9 = 0 ∧ (m - 8 : ℤ) % 6 = 0 ∧ 
   (m - 8 : ℤ) % 12 = 0 ∧ (m - 8 : ℤ) % 18 = 0 → m ≥ n) ∧
  (n - 8 : ℤ) % 9 = 0 ∧ (n - 8 : ℤ) % 6 = 0 ∧ 
  (n - 8 : ℤ) % 12 = 0 ∧ (n - 8 : ℤ) % 18 = 0 ∧
  n = 44 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_after_subtraction_l1124_112401


namespace NUMINAMATH_CALUDE_system_solution_l1124_112471

theorem system_solution (x y z t : ℝ) : 
  (x^2 - 9*y^2 = 0 ∧ x + y + z = 0) ↔ 
  ((x = 3*t ∧ y = t ∧ z = -4*t) ∨ (x = -3*t ∧ y = t ∧ z = 2*t)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1124_112471


namespace NUMINAMATH_CALUDE_frank_planted_two_seeds_per_orange_l1124_112458

/-- The number of oranges Betty picked -/
def betty_oranges : ℕ := 15

/-- The number of oranges Bill picked -/
def bill_oranges : ℕ := 12

/-- The number of oranges Frank picked -/
def frank_oranges : ℕ := 3 * (betty_oranges + bill_oranges)

/-- The number of oranges each tree contains -/
def oranges_per_tree : ℕ := 5

/-- The total number of oranges Philip can pick -/
def philip_total_oranges : ℕ := 810

/-- The number of seeds Frank planted from each of his oranges -/
def seeds_per_orange : ℕ := philip_total_oranges / oranges_per_tree / frank_oranges

theorem frank_planted_two_seeds_per_orange : seeds_per_orange = 2 := by
  sorry

end NUMINAMATH_CALUDE_frank_planted_two_seeds_per_orange_l1124_112458


namespace NUMINAMATH_CALUDE_pencil_length_l1124_112449

/-- The total length of a pencil with given colored sections -/
theorem pencil_length (purple_length black_length blue_length : ℝ) 
  (h_purple : purple_length = 1.5)
  (h_black : black_length = 0.5)
  (h_blue : blue_length = 2) :
  purple_length + black_length + blue_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_pencil_length_l1124_112449


namespace NUMINAMATH_CALUDE_decimal_111_to_base5_l1124_112480

/-- Converts a natural number to its base-5 representation as a list of digits -/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Theorem: The base-5 representation of 111 (decimal) is [2, 3, 3] -/
theorem decimal_111_to_base5 :
  toBase5 111 = [2, 3, 3] := by
  sorry

#eval toBase5 111  -- This will output [2, 3, 3]

end NUMINAMATH_CALUDE_decimal_111_to_base5_l1124_112480


namespace NUMINAMATH_CALUDE_flour_cost_for_cheapest_pie_l1124_112489

/-- The cost of flour for the cheapest pie -/
def flour_cost : ℝ := 2

/-- The cost of sugar for both pies -/
def sugar_cost : ℝ := 1

/-- The cost of eggs and butter for both pies -/
def eggs_butter_cost : ℝ := 1.5

/-- The weight of blueberries needed for the blueberry pie in pounds -/
def blueberry_weight : ℝ := 3

/-- The weight of a container of blueberries in ounces -/
def blueberry_container_weight : ℝ := 8

/-- The cost of a container of blueberries -/
def blueberry_container_cost : ℝ := 2.25

/-- The weight of cherries needed for the cherry pie in pounds -/
def cherry_weight : ℝ := 4

/-- The cost of a four-pound bag of cherries -/
def cherry_bag_cost : ℝ := 14

/-- The total price to make the cheapest pie -/
def cheapest_pie_cost : ℝ := 18

theorem flour_cost_for_cheapest_pie :
  flour_cost = cheapest_pie_cost - min
    (sugar_cost + eggs_butter_cost + (blueberry_weight * 16 / blueberry_container_weight) * blueberry_container_cost)
    (sugar_cost + eggs_butter_cost + cherry_bag_cost) :=
by sorry

end NUMINAMATH_CALUDE_flour_cost_for_cheapest_pie_l1124_112489


namespace NUMINAMATH_CALUDE_third_packing_number_l1124_112400

theorem third_packing_number (N : ℕ) (h1 : N = 301) (h2 : N % 3 = 1) (h3 : N % 4 = 1) (h4 : N % 7 = 0) :
  ∃ x : ℕ, x ≠ 3 ∧ x ≠ 4 ∧ x > 4 ∧ N % x = 1 ∧ (∀ y : ℕ, y ≠ 3 ∧ y ≠ 4 ∧ y < x ∧ y > 4 → N % y ≠ 1) ∧ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_third_packing_number_l1124_112400


namespace NUMINAMATH_CALUDE_postcard_selling_price_l1124_112408

/-- Proves that the selling price per postcard is $10 --/
theorem postcard_selling_price 
  (initial_postcards : ℕ)
  (sold_postcards : ℕ)
  (new_postcard_price : ℚ)
  (final_postcard_count : ℕ)
  (h1 : initial_postcards = 18)
  (h2 : sold_postcards = initial_postcards / 2)
  (h3 : new_postcard_price = 5)
  (h4 : final_postcard_count = 36)
  : (sold_postcards : ℚ) * (final_postcard_count - initial_postcards) * new_postcard_price / sold_postcards = 10 := by
  sorry

end NUMINAMATH_CALUDE_postcard_selling_price_l1124_112408


namespace NUMINAMATH_CALUDE_subtraction_result_l1124_112463

theorem subtraction_result : 6102 - 2016 = 4086 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l1124_112463


namespace NUMINAMATH_CALUDE_relay_team_permutations_l1124_112460

def team_size : ℕ := 4
def fixed_positions : ℕ := 1
def remaining_positions : ℕ := team_size - fixed_positions

theorem relay_team_permutations :
  Nat.factorial remaining_positions = 6 :=
by sorry

end NUMINAMATH_CALUDE_relay_team_permutations_l1124_112460


namespace NUMINAMATH_CALUDE_prime_sum_squares_l1124_112402

theorem prime_sum_squares (p q : ℕ) : 
  Prime p ∧ Prime q ∧ Prime (2^2 + p^2 + q^2) ↔ (p = 3 ∧ q = 2) ∨ (p = 2 ∧ q = 3) :=
sorry

end NUMINAMATH_CALUDE_prime_sum_squares_l1124_112402


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_minimum_a_for_always_greater_than_three_l1124_112445

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + x + a

-- Theorem 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≥ 3} = {x : ℝ | x ≥ 1 ∨ x ≤ -2} := by sorry

-- Theorem 2
theorem minimum_a_for_always_greater_than_three :
  (∀ x : ℝ, f a x ≥ 3) ↔ a ≥ 13/4 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_minimum_a_for_always_greater_than_three_l1124_112445


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l1124_112490

theorem cylinder_surface_area (r l : ℝ) : 
  r = 1 → l = 2*r → 2*π*r*(r + l) = 6*π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l1124_112490


namespace NUMINAMATH_CALUDE_max_inverse_sum_l1124_112418

theorem max_inverse_sum (x y a b : ℝ) 
  (ha : a > 1) 
  (hb : b > 1) 
  (hax : a^x = 2) 
  (hby : b^y = 2) 
  (hab : 2*a + b = 8) : 
  (∀ w z : ℝ, a^w = 2 → b^z = 2 → 1/w + 1/z ≤ 3) ∧ 
  (∃ w z : ℝ, a^w = 2 ∧ b^z = 2 ∧ 1/w + 1/z = 3) :=
sorry

end NUMINAMATH_CALUDE_max_inverse_sum_l1124_112418


namespace NUMINAMATH_CALUDE_coefficient_x4_in_binomial_expansion_l1124_112499

theorem coefficient_x4_in_binomial_expansion :
  (Finset.range 11).sum (fun k => (Nat.choose 10 k) * (1^(10 - k)) * (1^k)) = 210 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_in_binomial_expansion_l1124_112499


namespace NUMINAMATH_CALUDE_symmetrical_parabola_directrix_l1124_112454

/-- Given a parabola y = 2x², prove that the equation of the directrix of the parabola
    symmetrical to it with respect to the line y = x is x = -1/8 -/
theorem symmetrical_parabola_directrix (x y : ℝ) :
  (y = 2 * x^2) →  -- Original parabola
  ∃ (x₀ : ℝ), 
    (∀ (x' y' : ℝ), y'^2 = (1/2) * x' ↔ (y = x ∧ x' = y ∧ y' = x)) →  -- Symmetry condition
    (x₀ = -1/8 ∧ ∀ (x' y' : ℝ), y'^2 = (1/2) * x' → |x' - x₀| = (1/4)) :=  -- Directrix equation
sorry

end NUMINAMATH_CALUDE_symmetrical_parabola_directrix_l1124_112454


namespace NUMINAMATH_CALUDE_equation_one_real_solution_l1124_112441

theorem equation_one_real_solution :
  ∃! x : ℝ, (3 * x) / (x^2 + 2 * x + 4) + (4 * x) / (x^2 - 4 * x + 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_one_real_solution_l1124_112441


namespace NUMINAMATH_CALUDE_number_divided_by_five_equals_number_plus_three_l1124_112465

theorem number_divided_by_five_equals_number_plus_three : 
  ∃ x : ℚ, x / 5 = x + 3 ∧ x = -15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_five_equals_number_plus_three_l1124_112465


namespace NUMINAMATH_CALUDE_linear_equation_implies_mn_one_l1124_112482

/-- If (m+2)x^(|m|-1) + y^(2n) = 5 is a linear equation in x and y, where m and n are real numbers, then mn = 1 -/
theorem linear_equation_implies_mn_one (m n : ℝ) : 
  (∃ a b c : ℝ, ∀ x y : ℝ, (m + 2) * x^(|m| - 1) + y^(2*n) = 5 ↔ a*x + b*y = c) → 
  m * n = 1 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_implies_mn_one_l1124_112482


namespace NUMINAMATH_CALUDE_symmetric_difference_equality_l1124_112491

open Set

theorem symmetric_difference_equality (A B K : Set α) : 
  symmDiff A K = symmDiff B K → A = B :=
by sorry

end NUMINAMATH_CALUDE_symmetric_difference_equality_l1124_112491


namespace NUMINAMATH_CALUDE_fraction_sum_equals_one_l1124_112419

theorem fraction_sum_equals_one : 3/5 - 1/10 + 1/2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_one_l1124_112419


namespace NUMINAMATH_CALUDE_slices_per_pie_is_four_l1124_112462

/-- The number of slices in a whole pie at a pie shop -/
def slices_per_pie : ℕ := sorry

/-- The price of a single slice of pie in dollars -/
def price_per_slice : ℕ := 5

/-- The number of whole pies sold -/
def pies_sold : ℕ := 9

/-- The total revenue in dollars from selling all pies -/
def total_revenue : ℕ := 180

/-- Theorem stating that the number of slices per pie is 4 -/
theorem slices_per_pie_is_four :
  slices_per_pie = 4 :=
by sorry

end NUMINAMATH_CALUDE_slices_per_pie_is_four_l1124_112462


namespace NUMINAMATH_CALUDE_parabola_translation_l1124_112466

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola vertically -/
def translateVertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + v }

/-- Translates a parabola horizontally -/
def translateHorizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a, b := 2 * p.a * h + p.b, c := p.a * h^2 + p.b * h + p.c }

/-- The main theorem stating that translating y = 3x² upwards by 3 and left by 2 results in y = 3(x+2)² + 3 -/
theorem parabola_translation (original : Parabola) 
  (h : original = { a := 3, b := 0, c := 0 }) : 
  translateHorizontal (translateVertical original 3) 2 = { a := 3, b := 12, c := 15 } := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l1124_112466


namespace NUMINAMATH_CALUDE_parallel_vectors_tan_double_angle_l1124_112433

/-- Given two vectors a and b in R², where a is parallel to b, 
    prove that tan(2θ) = -4/3 -/
theorem parallel_vectors_tan_double_angle (θ : ℝ) 
  (a b : ℝ × ℝ) 
  (ha : a = (Real.sin θ, 2)) 
  (hb : b = (Real.cos θ, 1)) 
  (hparallel : ∃ (k : ℝ), a = k • b) : 
  Real.tan (2 * θ) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_tan_double_angle_l1124_112433


namespace NUMINAMATH_CALUDE_painters_work_days_l1124_112403

/-- Given that 5 painters take 1.8 work-days to finish a job, prove that 4 painters
    working at the same rate will take 2.25 work-days to finish the same job. -/
theorem painters_work_days (initial_painters : ℕ) (initial_days : ℝ) 
  (new_painters : ℕ) (new_days : ℝ) :
  initial_painters = 5 →
  initial_days = 1.8 →
  new_painters = 4 →
  (initial_painters : ℝ) * initial_days = (new_painters : ℝ) * new_days →
  new_days = 2.25 := by
sorry

end NUMINAMATH_CALUDE_painters_work_days_l1124_112403


namespace NUMINAMATH_CALUDE_zero_in_interval_l1124_112410

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem zero_in_interval :
  ∃ x₀ : ℝ, f x₀ = 0 ∧ x₀ ∈ Set.Ico 2 3 :=
sorry

end NUMINAMATH_CALUDE_zero_in_interval_l1124_112410


namespace NUMINAMATH_CALUDE_percentage_of_x_pay_to_y_l1124_112413

/-- The percentage of X's pay compared to Y's, given their total pay and Y's pay -/
theorem percentage_of_x_pay_to_y (total_pay y_pay x_pay : ℚ) : 
  total_pay = 528 →
  y_pay = 240 →
  x_pay + y_pay = total_pay →
  (x_pay / y_pay) * 100 = 120 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_x_pay_to_y_l1124_112413


namespace NUMINAMATH_CALUDE_system_range_of_a_l1124_112496

/-- Given a system of linear equations in x and y, prove the range of a -/
theorem system_range_of_a (x y a : ℝ) 
  (eq1 : x + 3*y = 2 + a) 
  (eq2 : 3*x + y = -4*a) 
  (h : x + y > 2) : 
  a < -2 := by
sorry

end NUMINAMATH_CALUDE_system_range_of_a_l1124_112496


namespace NUMINAMATH_CALUDE_quadratic_coefficient_sum_l1124_112472

theorem quadratic_coefficient_sum : ∃ (coeff_sum : ℤ),
  (∀ a : ℤ, 
    (∃ r s : ℤ, r < 0 ∧ s < 0 ∧ r ≠ s ∧ r * s = 24 ∧ r + s = a) →
    (∃ x y : ℤ, x^2 + a*x + 24 = 0 ∧ y^2 + a*y + 24 = 0 ∧ x ≠ y ∧ x < 0 ∧ y < 0)) ∧
  (∀ a : ℤ,
    (∃ x y : ℤ, x^2 + a*x + 24 = 0 ∧ y^2 + a*y + 24 = 0 ∧ x ≠ y ∧ x < 0 ∧ y < 0) →
    (∃ r s : ℤ, r < 0 ∧ s < 0 ∧ r ≠ s ∧ r * s = 24 ∧ r + s = a)) ∧
  coeff_sum = -60 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_sum_l1124_112472


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1124_112488

theorem complex_equation_solution :
  ∃ (z : ℂ), z = -3/4 * I ∧ (2 : ℂ) - I * z = -1 + 3 * I * z :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1124_112488


namespace NUMINAMATH_CALUDE_cubic_km_to_m_strip_l1124_112455

/-- The length of a strip formed by cutting a cubic kilometer into cubic meters and laying them out in a single line -/
def strip_length : ℝ := 1000000

/-- Conversion factor from kilometers to meters -/
def km_to_m : ℝ := 1000

theorem cubic_km_to_m_strip : 
  strip_length = (km_to_m ^ 3) / km_to_m := by sorry

end NUMINAMATH_CALUDE_cubic_km_to_m_strip_l1124_112455


namespace NUMINAMATH_CALUDE_prob_even_heads_is_17_25_l1124_112440

/-- Represents an unfair coin where the probability of heads is 4 times the probability of tails -/
structure UnfairCoin where
  p_tails : ℝ
  p_heads : ℝ
  p_tails_pos : 0 < p_tails
  p_heads_pos : 0 < p_heads
  p_sum_one : p_tails + p_heads = 1
  p_heads_four_times : p_heads = 4 * p_tails

/-- The probability of getting an even number of heads when flipping the unfair coin twice -/
def prob_even_heads (c : UnfairCoin) : ℝ :=
  c.p_tails^2 + c.p_heads^2

/-- Theorem stating that the probability of getting an even number of heads
    when flipping the unfair coin twice is 17/25 -/
theorem prob_even_heads_is_17_25 (c : UnfairCoin) :
  prob_even_heads c = 17/25 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_heads_is_17_25_l1124_112440


namespace NUMINAMATH_CALUDE_free_throw_contest_total_l1124_112475

/-- Given a free throw contest where:
  * Alex made 8 baskets
  * Sandra made three times as many baskets as Alex
  * Hector made two times the number of baskets that Sandra made
  Prove that the total number of baskets made by all three is 80. -/
theorem free_throw_contest_total (alex_baskets : ℕ) (sandra_baskets : ℕ) (hector_baskets : ℕ) 
  (h1 : alex_baskets = 8)
  (h2 : sandra_baskets = 3 * alex_baskets)
  (h3 : hector_baskets = 2 * sandra_baskets) :
  alex_baskets + sandra_baskets + hector_baskets = 80 := by
  sorry

#check free_throw_contest_total

end NUMINAMATH_CALUDE_free_throw_contest_total_l1124_112475


namespace NUMINAMATH_CALUDE_john_average_increase_l1124_112429

def john_scores : List ℝ := [90, 85, 92, 95]

theorem john_average_increase :
  let initial_average := (john_scores.take 3).sum / 3
  let new_average := john_scores.sum / 4
  new_average - initial_average = 1.5 := by sorry

end NUMINAMATH_CALUDE_john_average_increase_l1124_112429


namespace NUMINAMATH_CALUDE_youngest_age_l1124_112497

/-- Proves the age of the youngest person given the conditions of the problem -/
theorem youngest_age (n : ℕ) (current_avg : ℚ) (birth_avg : ℚ) 
  (h1 : n = 7)
  (h2 : current_avg = 30)
  (h3 : birth_avg = 22) :
  (n * current_avg - (n - 1) * birth_avg) / n = 78 / 7 := by
  sorry

end NUMINAMATH_CALUDE_youngest_age_l1124_112497


namespace NUMINAMATH_CALUDE_tom_stock_profit_l1124_112415

/-- Calculate Tom's overall profit from stock transactions -/
theorem tom_stock_profit : 
  let stock_a_initial_shares : ℕ := 20
  let stock_a_initial_price : ℚ := 3
  let stock_b_initial_shares : ℕ := 30
  let stock_b_initial_price : ℚ := 5
  let stock_c_initial_shares : ℕ := 15
  let stock_c_initial_price : ℚ := 10
  let commission_rate : ℚ := 2 / 100
  let stock_a_sold_shares : ℕ := 10
  let stock_a_sell_price : ℚ := 4
  let stock_b_sold_shares : ℕ := 20
  let stock_b_sell_price : ℚ := 7
  let stock_c_sold_shares : ℕ := 5
  let stock_c_sell_price : ℚ := 12
  let stock_a_value_increase : ℚ := 2
  let stock_b_value_increase : ℚ := 1.2
  let stock_c_value_decrease : ℚ := 0.9

  let initial_cost := (stock_a_initial_shares * stock_a_initial_price + 
                       stock_b_initial_shares * stock_b_initial_price + 
                       stock_c_initial_shares * stock_c_initial_price) * (1 + commission_rate)

  let sales_revenue := (stock_a_sold_shares * stock_a_sell_price + 
                        stock_b_sold_shares * stock_b_sell_price + 
                        stock_c_sold_shares * stock_c_sell_price) * (1 - commission_rate)

  let remaining_value := (stock_a_initial_shares - stock_a_sold_shares) * stock_a_initial_price * stock_a_value_increase + 
                         (stock_b_initial_shares - stock_b_sold_shares) * stock_b_initial_price * stock_b_value_increase + 
                         (stock_c_initial_shares - stock_c_sold_shares) * stock_c_initial_price * stock_c_value_decrease

  let profit := sales_revenue + remaining_value - initial_cost

  profit = 78
  := by sorry

end NUMINAMATH_CALUDE_tom_stock_profit_l1124_112415


namespace NUMINAMATH_CALUDE_min_value_theorem_l1124_112430

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f (x : ℝ) := x^2 - 2*x + 2
  let g (x : ℝ) := -x^2 + a*x + b
  let f' (x : ℝ) := 2*x - 2
  let g' (x : ℝ) := -2*x + a
  ∃ x₀ : ℝ, f x₀ = g x₀ ∧ f' x₀ * g' x₀ = -1 →
  (1/a + 4/b) ≥ 18/5 ∧ ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 1/a₀ + 4/b₀ = 18/5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1124_112430


namespace NUMINAMATH_CALUDE_inscribed_triangle_ratio_l1124_112406

/-- An ellipse with semi-major axis 4 and semi-minor axis 3 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 16) + (p.2^2 / 9) = 1}

/-- The foci of the ellipse -/
def Foci : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p ∈ Ellipse ∧ ∃ q ∈ Ellipse, p ≠ q ∧ ∀ r ∈ Ellipse, dist p r + dist q r = 8}

/-- An equilateral triangle inscribed in the ellipse -/
structure InscribedTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  A_in_ellipse : A ∈ Ellipse
  B_in_ellipse : B ∈ Ellipse
  C_in_ellipse : C ∈ Ellipse
  B_at_origin : B = (0, 3)
  AC_parallel_x : A.2 = C.2
  equilateral : dist A B = dist B C ∧ dist B C = dist C A
  foci_at_AC : A ∈ Foci ∧ C ∈ Foci

theorem inscribed_triangle_ratio 
  (t : InscribedTriangle) : 
  dist t.A t.B / dist t.A t.C = Real.sqrt 21 / 7 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_ratio_l1124_112406


namespace NUMINAMATH_CALUDE_power_division_nineteen_l1124_112446

theorem power_division_nineteen : 19^12 / 19^10 = 361 := by
  sorry

end NUMINAMATH_CALUDE_power_division_nineteen_l1124_112446


namespace NUMINAMATH_CALUDE_arrow_connections_theorem_l1124_112432

/-- The number of ways to connect 2n points on a circle with n arrows -/
def arrow_connections (n : ℕ) : ℕ := Nat.choose (2 * n) n

/-- Theorem statement for the arrow connection problem -/
theorem arrow_connections_theorem (n : ℕ) (h : n > 0) :
  arrow_connections n = Nat.choose (2 * n) n :=
by sorry

end NUMINAMATH_CALUDE_arrow_connections_theorem_l1124_112432


namespace NUMINAMATH_CALUDE_modular_inverse_13_mod_1200_l1124_112467

theorem modular_inverse_13_mod_1200 : ∃ x : ℕ, x < 1200 ∧ (13 * x) % 1200 = 1 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_13_mod_1200_l1124_112467


namespace NUMINAMATH_CALUDE_eight_b_equals_sixteen_l1124_112416

theorem eight_b_equals_sixteen
  (h1 : 6 * a + 3 * b = 0)
  (h2 : b - 3 = a)
  (h3 : b + c = 5)
  : 8 * b = 16 := by
  sorry

end NUMINAMATH_CALUDE_eight_b_equals_sixteen_l1124_112416


namespace NUMINAMATH_CALUDE_work_increase_percentage_l1124_112494

/-- Proves that when 1/5 of the members in an office are absent, 
    the percentage increase in work for each remaining person is 25% -/
theorem work_increase_percentage (p : ℝ) (W : ℝ) (h1 : p > 0) (h2 : W > 0) : 
  let original_work_per_person := W / p
  let remaining_persons := p * (4/5)
  let new_work_per_person := W / remaining_persons
  let increase_percentage := (new_work_per_person - original_work_per_person) / original_work_per_person * 100
  increase_percentage = 25 := by sorry

end NUMINAMATH_CALUDE_work_increase_percentage_l1124_112494


namespace NUMINAMATH_CALUDE_eraser_cost_l1124_112434

def total_money : ℕ := 100
def heaven_spent : ℕ := 30
def brother_highlighters : ℕ := 30
def num_erasers : ℕ := 10

theorem eraser_cost :
  (total_money - heaven_spent - brother_highlighters) / num_erasers = 4 := by
  sorry

end NUMINAMATH_CALUDE_eraser_cost_l1124_112434


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1124_112442

theorem max_value_of_expression (t : ℝ) : 
  ∃ (max : ℝ), max = (1 / 16) ∧ ∀ t, ((3^t - 4*t) * t) / (9^t) ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1124_112442
