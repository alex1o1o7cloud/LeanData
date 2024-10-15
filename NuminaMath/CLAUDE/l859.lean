import Mathlib

namespace NUMINAMATH_CALUDE_smallest_n_with_seven_in_squares_l859_85991

def contains_seven (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 10 * a + 7 * b ∧ b ≤ 9

theorem smallest_n_with_seven_in_squares : 
  (∀ m : ℕ, m < 26 → ¬(contains_seven (m^2) ∧ contains_seven ((m+1)^2))) ∧
  (contains_seven (26^2) ∧ contains_seven (27^2)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_seven_in_squares_l859_85991


namespace NUMINAMATH_CALUDE_player_a_advantage_l859_85950

/-- Represents the outcome of a roll of two dice -/
structure DiceRoll :=
  (sum : Nat)
  (probability : Rat)

/-- Calculates the expected value for a player given a list of dice rolls -/
def expectedValue (rolls : List DiceRoll) : Rat :=
  rolls.foldl (fun acc roll => acc + roll.sum * roll.probability) 0

/-- Represents the game rules -/
def gameRules (roll : DiceRoll) : Rat :=
  if roll.sum % 2 = 1 then roll.sum * roll.probability
  else if roll.sum = 2 then 0
  else -roll.sum * roll.probability

/-- The list of all possible dice rolls and their probabilities -/
def allRolls : List DiceRoll := [
  ⟨2, 1/36⟩, ⟨3, 1/18⟩, ⟨4, 1/12⟩, ⟨5, 1/9⟩, ⟨6, 5/36⟩, 
  ⟨7, 1/6⟩, ⟨8, 5/36⟩, ⟨9, 1/9⟩, ⟨10, 1/12⟩, ⟨11, 1/18⟩, ⟨12, 1/36⟩
]

/-- The expected value for player A per roll -/
def expectedValueA : Rat := allRolls.foldl (fun acc roll => acc + gameRules roll) 0

theorem player_a_advantage : 
  expectedValueA > 0 ∧ 36 * expectedValueA = 2 := by sorry


end NUMINAMATH_CALUDE_player_a_advantage_l859_85950


namespace NUMINAMATH_CALUDE_sin_A_value_l859_85928

theorem sin_A_value (A : Real) (h1 : 0 < A) (h2 : A < Real.pi / 2) (h3 : Real.tan A = Real.sqrt 2 / 3) : 
  Real.sin A = Real.sqrt 22 / 11 := by
  sorry

end NUMINAMATH_CALUDE_sin_A_value_l859_85928


namespace NUMINAMATH_CALUDE_min_max_values_l859_85958

theorem min_max_values (a b : ℝ) (h1 : a + b = 1) (h2 : a > 0) (h3 : b > 0) :
  (∀ x y, x + y = 1 ∧ x > 0 ∧ y > 0 → a^2 + b^2 ≤ x^2 + y^2) ∧
  (∀ x y, x + y = 1 ∧ x > 0 ∧ y > 0 → Real.sqrt a + Real.sqrt b ≥ Real.sqrt x + Real.sqrt y) ∧
  (∀ x y, x + y = 1 ∧ x > 0 ∧ y > 0 → 1 / (a + 2*b) + 1 / (2*a + b) ≤ 1 / (x + 2*y) + 1 / (2*x + y)) ∧
  a^2 + b^2 = 1/2 ∧
  Real.sqrt a + Real.sqrt b = Real.sqrt 2 ∧
  1 / (a + 2*b) + 1 / (2*a + b) = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_min_max_values_l859_85958


namespace NUMINAMATH_CALUDE_smallest_number_l859_85907

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

def binary : List Nat := [1, 0, 1, 0, 1, 1]
def ternary : List Nat := [1, 2, 1, 0]
def octal : List Nat := [1, 1, 0]
def duodecimal : List Nat := [6, 8]

theorem smallest_number : 
  to_decimal ternary 3 ≤ to_decimal binary 2 ∧
  to_decimal ternary 3 ≤ to_decimal octal 8 ∧
  to_decimal ternary 3 ≤ to_decimal duodecimal 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l859_85907


namespace NUMINAMATH_CALUDE_david_profit_l859_85939

/-- Represents the discount percentage based on the number of sacks bought -/
def discount_percentage (num_sacks : ℕ) : ℚ :=
  if num_sacks ≤ 10 then 2/100
  else if num_sacks ≤ 20 then 4/100
  else 5/100

/-- Calculates the total cost of buying sacks with discount -/
def total_cost (num_sacks : ℕ) (price_per_sack : ℚ) : ℚ :=
  num_sacks * price_per_sack * (1 - discount_percentage num_sacks)

/-- Calculates the total selling price for a given number of days and price per kg -/
def selling_price (kg_per_day : ℚ) (price_per_kg : ℚ) (num_days : ℕ) : ℚ :=
  kg_per_day * price_per_kg * num_days

/-- Calculates the total selling price for the week -/
def total_selling_price (kg_per_day : ℚ) : ℚ :=
  selling_price kg_per_day 1.20 3 +
  selling_price kg_per_day 1.30 2 +
  selling_price kg_per_day 1.25 2

/-- Calculates the profit after tax -/
def profit_after_tax (total_selling : ℚ) (total_cost : ℚ) (tax_rate : ℚ) : ℚ :=
  total_selling * (1 - tax_rate) - total_cost

/-- Theorem stating David's profit for the week -/
theorem david_profit :
  let num_sacks : ℕ := 25
  let price_per_sack : ℚ := 50
  let sack_weight : ℚ := 50
  let total_kg : ℚ := num_sacks * sack_weight
  let kg_per_day : ℚ := total_kg / 7
  let tax_rate : ℚ := 12/100
  profit_after_tax
    (total_selling_price kg_per_day)
    (total_cost num_sacks price_per_sack)
    tax_rate = 179.62 := by
  sorry


end NUMINAMATH_CALUDE_david_profit_l859_85939


namespace NUMINAMATH_CALUDE_dilation_problem_l859_85997

def dilation (center scale : ℂ) (z : ℂ) : ℂ :=
  center + scale * (z - center)

theorem dilation_problem : 
  let center := (0 : ℂ) + 5*I
  let scale := (3 : ℂ)
  let z := (3 : ℂ) + 2*I
  dilation center scale z = (9 : ℂ) - 4*I := by
  sorry

end NUMINAMATH_CALUDE_dilation_problem_l859_85997


namespace NUMINAMATH_CALUDE_disney_banquet_residents_l859_85949

theorem disney_banquet_residents (total_attendees : ℕ) (resident_price non_resident_price : ℚ) (total_revenue : ℚ) :
  total_attendees = 586 →
  resident_price = 12.95 →
  non_resident_price = 17.95 →
  total_revenue = 9423.70 →
  ∃ (residents non_residents : ℕ),
    residents + non_residents = total_attendees ∧
    residents * resident_price + non_residents * non_resident_price = total_revenue ∧
    residents = 220 :=
by sorry

end NUMINAMATH_CALUDE_disney_banquet_residents_l859_85949


namespace NUMINAMATH_CALUDE_no_triangle_from_tangent_line_l859_85942

/-- Given a line ax + by + c = 0 (where a, b, and c are positive) tangent to the circle x^2 + y^2 = 2,
    there does not exist a triangle with side lengths a, b, and c. -/
theorem no_triangle_from_tangent_line (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
    (h_tangent : ∀ x y : ℝ, a * x + b * y + c = 0 → x^2 + y^2 = 2) :
  ¬ ∃ (A B C : ℝ × ℝ), 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = c ∧
    Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = a ∧
    Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = b :=
by sorry

end NUMINAMATH_CALUDE_no_triangle_from_tangent_line_l859_85942


namespace NUMINAMATH_CALUDE_reservoir_shortage_l859_85978

/-- Represents a water reservoir with its capacity and current amount --/
structure Reservoir where
  capacity : ℝ
  current_amount : ℝ
  normal_level : ℝ
  h1 : current_amount = 14
  h2 : current_amount = 2 * normal_level
  h3 : current_amount = 0.7 * capacity

/-- The difference between the total capacity and the normal level is 13 million gallons --/
theorem reservoir_shortage (r : Reservoir) : r.capacity - r.normal_level = 13 := by
  sorry

#check reservoir_shortage

end NUMINAMATH_CALUDE_reservoir_shortage_l859_85978


namespace NUMINAMATH_CALUDE_total_balloons_l859_85956

/-- Given a set of balloons divided into 7 equal groups with 5 balloons in each group,
    the total number of balloons is 35. -/
theorem total_balloons (num_groups : ℕ) (balloons_per_group : ℕ) 
  (h1 : num_groups = 7) (h2 : balloons_per_group = 5) : 
  num_groups * balloons_per_group = 35 := by
  sorry

end NUMINAMATH_CALUDE_total_balloons_l859_85956


namespace NUMINAMATH_CALUDE_solution_exists_l859_85981

-- Define the logarithm function (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the system of equations
def equation_system (x y : ℝ) : Prop :=
  x ≠ 0 ∧ y > 0 ∧ log10 (x^2 / y^3) = 1 ∧ log10 (x^2 * y^3) = 7

-- Theorem statement
theorem solution_exists :
  ∃ x y : ℝ, equation_system x y ∧ (x = 100 ∨ x = -100) ∧ y = 10 :=
by sorry

end NUMINAMATH_CALUDE_solution_exists_l859_85981


namespace NUMINAMATH_CALUDE_max_inscribed_triangles_count_l859_85996

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) :=
  (h_pos : 0 < b ∧ b < a)

/-- A right-angled isosceles triangle inscribed in an ellipse -/
structure InscribedTriangle (e : Ellipse a b) :=
  (vertex : ℝ × ℝ)
  (h_on_ellipse : (vertex.1^2 / a^2) + (vertex.2^2 / b^2) = 1)
  (h_right_angled : True)  -- Placeholder for the right-angled condition
  (h_isosceles : True)     -- Placeholder for the isosceles condition
  (h_vertex_b : vertex.1 = 0 ∧ vertex.2 = b)

/-- The maximum number of right-angled isosceles triangles inscribed in an ellipse -/
def max_inscribed_triangles (e : Ellipse a b) : ℕ :=
  3

theorem max_inscribed_triangles_count (a b : ℝ) (e : Ellipse a b) :
  ∃ (n : ℕ), n ≤ max_inscribed_triangles e ∧
  ∀ (m : ℕ), (∃ (triangles : Fin m → InscribedTriangle e), 
    ∀ (i j : Fin m), i ≠ j → triangles i ≠ triangles j) → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_max_inscribed_triangles_count_l859_85996


namespace NUMINAMATH_CALUDE_variance_transformation_l859_85954

variable {n : ℕ}
variable (x : Fin n → ℝ)
variable (a b : ℝ)

def variance (y : Fin n → ℝ) : ℝ := sorry

theorem variance_transformation (h1 : variance x = 3) 
  (h2 : variance (fun i => a * x i + b) = 12) : a = 2 ∨ a = -2 := by sorry

end NUMINAMATH_CALUDE_variance_transformation_l859_85954


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l859_85968

theorem inequality_not_always_true (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a > b) (hc : c ≠ 0) :
  ¬ (∀ (a b c : ℝ), a > 0 → b > 0 → a > b → c ≠ 0 → a / c > b / c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l859_85968


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l859_85941

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Define the complement of A in U
def complementA : Set ℝ := {x | x < -1 ∨ x > 3}

-- Theorem statement
theorem complement_of_A_in_U : Set.compl A = complementA := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l859_85941


namespace NUMINAMATH_CALUDE_inscribed_rectangle_sides_l859_85986

/-- A triangle with sides 3, 4, and 5 -/
structure Triangle345 where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 3
  hb : b = 4
  hc : c = 5

/-- A rectangle inscribed in a Triangle345 -/
structure InscribedRectangle (t : Triangle345) where
  short_side : ℝ
  long_side : ℝ
  h_double : long_side = 2 * short_side
  h_inscribed : short_side > 0 ∧ long_side > 0 ∧ long_side ≤ t.c

theorem inscribed_rectangle_sides (t : Triangle345) (r : InscribedRectangle t) :
  r.short_side = 48 / 67 ∧ r.long_side = 96 / 67 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_sides_l859_85986


namespace NUMINAMATH_CALUDE_find_n_l859_85910

theorem find_n : ∃ n : ℤ, n + (n + 1) + (n + 2) = 9 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l859_85910


namespace NUMINAMATH_CALUDE_difference_c_minus_a_l859_85908

theorem difference_c_minus_a (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45)
  (h2 : (b + c) / 2 = 70) : 
  c - a = 50 := by
sorry

end NUMINAMATH_CALUDE_difference_c_minus_a_l859_85908


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_sqrt_3_sqrt_difference_times_sqrt_3_equals_neg_sqrt_6_l859_85947

-- Part 1
theorem sqrt_expression_equals_sqrt_3 :
  Real.sqrt 12 - Real.sqrt 48 + 9 * Real.sqrt (1/3) = Real.sqrt 3 := by sorry

-- Part 2
theorem sqrt_difference_times_sqrt_3_equals_neg_sqrt_6 :
  (Real.sqrt 8 - Real.sqrt 18) * Real.sqrt 3 = -Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_sqrt_3_sqrt_difference_times_sqrt_3_equals_neg_sqrt_6_l859_85947


namespace NUMINAMATH_CALUDE_binomial_expansion_arithmetic_sequence_l859_85932

theorem binomial_expansion_arithmetic_sequence (n : ℕ) : 
  (∃ d : ℚ, 1 + d = n / 2 ∧ n / 2 + d = n * (n - 1) / 8) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_arithmetic_sequence_l859_85932


namespace NUMINAMATH_CALUDE_triangle_side_length_l859_85938

theorem triangle_side_length (a c area : ℝ) (ha : a = 1) (hc : c = 7) (harea : area = 5) :
  let h := 2 * area / c
  let b := Real.sqrt ((a^2 + h^2) : ℝ)
  b = Real.sqrt 149 / 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l859_85938


namespace NUMINAMATH_CALUDE_cos_equality_problem_l859_85926

theorem cos_equality_problem (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (812 * π / 180) → n = 88 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_problem_l859_85926


namespace NUMINAMATH_CALUDE_quartic_roots_difference_l859_85951

/-- A quartic polynomial with roots forming an arithmetic sequence -/
def quartic_with_arithmetic_roots (a : ℝ) (x : ℝ) : ℝ := 
  a * (x^4 - 10*x^2 + 9)

/-- The derivative of the quartic polynomial -/
def quartic_derivative (a : ℝ) (x : ℝ) : ℝ := 
  4 * a * x * (x^2 - 5)

theorem quartic_roots_difference (a : ℝ) (h : a ≠ 0) :
  let f := quartic_with_arithmetic_roots a
  let f' := quartic_derivative a
  let max_root := Real.sqrt 5
  let min_root := -Real.sqrt 5
  (∀ x, f' x = 0 → x ≤ max_root) ∧ 
  (∀ x, f' x = 0 → x ≥ min_root) ∧
  (max_root - min_root = 2 * Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_quartic_roots_difference_l859_85951


namespace NUMINAMATH_CALUDE_impossibleToGetAllPlus_l859_85935

/-- Represents a 4x4 grid of signs -/
def Grid := Matrix (Fin 4) (Fin 4) Bool

/-- Flips all signs in a given row -/
def flipRow (g : Grid) (row : Fin 4) : Grid := sorry

/-- Flips all signs in a given column -/
def flipColumn (g : Grid) (col : Fin 4) : Grid := sorry

/-- The initial grid configuration -/
def initialGrid : Grid := 
  ![![true,  false, true,  true],
    ![true,  true,  true,  true],
    ![true,  true,  true,  true],
    ![true,  false, true,  true]]

/-- Checks if all cells in the grid are true ("+") -/
def allPlus (g : Grid) : Prop := ∀ i j, g i j = true

/-- Represents a sequence of row and column flipping operations -/
inductive FlipSequence : Type
  | empty : FlipSequence
  | flipRow : FlipSequence → Fin 4 → FlipSequence
  | flipColumn : FlipSequence → Fin 4 → FlipSequence

/-- Applies a sequence of flipping operations to a grid -/
def applyFlips : Grid → FlipSequence → Grid
  | g, FlipSequence.empty => g
  | g, FlipSequence.flipRow s i => applyFlips (flipRow g i) s
  | g, FlipSequence.flipColumn s j => applyFlips (flipColumn g j) s

theorem impossibleToGetAllPlus : 
  ¬∃ (s : FlipSequence), allPlus (applyFlips initialGrid s) := by
  sorry

end NUMINAMATH_CALUDE_impossibleToGetAllPlus_l859_85935


namespace NUMINAMATH_CALUDE_poultry_farm_solution_l859_85992

/-- Represents the poultry farm problem --/
def poultry_farm_problem (initial_chickens initial_guinea_fowls : ℕ)
  (daily_loss_chickens daily_loss_turkeys daily_loss_guinea_fowls : ℕ)
  (days : ℕ) (total_birds_left : ℕ) : Prop :=
  let initial_turkeys := 200
  let total_initial_birds := initial_chickens + initial_turkeys + initial_guinea_fowls
  let total_loss := (daily_loss_chickens + daily_loss_turkeys + daily_loss_guinea_fowls) * days
  total_initial_birds - total_loss = total_birds_left

/-- Theorem stating the solution to the poultry farm problem --/
theorem poultry_farm_solution :
  poultry_farm_problem 300 80 20 8 5 7 349 := by
  sorry

#check poultry_farm_solution

end NUMINAMATH_CALUDE_poultry_farm_solution_l859_85992


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l859_85989

theorem fixed_point_on_line (a b : ℝ) : (2 * a + b) * (-2) + (a + b) * 3 + a - b = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l859_85989


namespace NUMINAMATH_CALUDE_min_b_value_l859_85919

theorem min_b_value (a c b : ℕ+) (h1 : a < c) (h2 : c < b)
  (h3 : ∃! p : ℝ × ℝ, 3 * p.1 + p.2 = 3000 ∧ 
    p.2 = |p.1 - a.val| + |p.1 - c.val| + |p.1 - b.val|) :
  ∀ b' : ℕ+, (∃ a' c' : ℕ+, a' < c' ∧ c' < b' ∧
    ∃! p : ℝ × ℝ, 3 * p.1 + p.2 = 3000 ∧ 
    p.2 = |p.1 - a'.val| + |p.1 - c'.val| + |p.1 - b'.val|) → 
  9 ≤ b'.val := by
  sorry

end NUMINAMATH_CALUDE_min_b_value_l859_85919


namespace NUMINAMATH_CALUDE_smallest_AAB_l859_85936

theorem smallest_AAB : ∃ (A B : ℕ),
  A ≠ B ∧
  A ∈ Finset.range 10 ∧
  B ∈ Finset.range 10 ∧
  A ≠ 0 ∧
  (10 * A + B) = (110 * A + B) / 8 ∧
  ∀ (A' B' : ℕ),
    A' ≠ B' →
    A' ∈ Finset.range 10 →
    B' ∈ Finset.range 10 →
    A' ≠ 0 →
    (10 * A' + B') = (110 * A' + B') / 8 →
    110 * A + B ≤ 110 * A' + B' ∧
    110 * A + B = 773 :=
by sorry

end NUMINAMATH_CALUDE_smallest_AAB_l859_85936


namespace NUMINAMATH_CALUDE_intersection_one_element_l859_85988

theorem intersection_one_element (a : ℝ) : 
  let A : Set ℝ := {1, a, 5}
  let B : Set ℝ := {2, a^2 + 1}
  (∃! x, x ∈ A ∩ B) → a = 0 ∨ a = -2 := by
sorry

end NUMINAMATH_CALUDE_intersection_one_element_l859_85988


namespace NUMINAMATH_CALUDE_smallest_solution_abs_equation_l859_85965

theorem smallest_solution_abs_equation :
  ∀ x : ℝ, x * |x| = 3 * x + 4 → x ≥ 4 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_equation_l859_85965


namespace NUMINAMATH_CALUDE_least_value_property_l859_85917

/-- Represents a 3-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  is_three_digit : hundreds ≥ 1 ∧ hundreds ≤ 9

/-- The value of a 3-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- The sum of digits of a 3-digit number -/
def ThreeDigitNumber.digit_sum (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.units

/-- Predicate for the difference between hundreds and tens being 8 -/
def digit_difference_eight (n : ThreeDigitNumber) : Prop :=
  n.tens - n.hundreds = 8 ∨ n.hundreds - n.tens = 8

theorem least_value_property (k : ThreeDigitNumber) 
  (h : digit_difference_eight k) :
  ∃ (min_k : ThreeDigitNumber), 
    digit_difference_eight min_k ∧
    ∀ (k' : ThreeDigitNumber), digit_difference_eight k' → 
      min_k.value ≤ k'.value ∧
      min_k.value = 19 * min_k.digit_sum :=
  sorry

end NUMINAMATH_CALUDE_least_value_property_l859_85917


namespace NUMINAMATH_CALUDE_baker_cakes_l859_85985

/-- Calculates the remaining number of cakes after buying and selling -/
def remaining_cakes (initial bought sold : ℕ) : ℕ :=
  initial + bought - sold

/-- Theorem: The number of cakes Baker still has is 190 -/
theorem baker_cakes : remaining_cakes 173 103 86 = 190 := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_l859_85985


namespace NUMINAMATH_CALUDE_parabola_vertex_specific_parabola_vertex_l859_85931

/-- The vertex of a parabola in the form y = a(x-h)^2 + k is (h, k) --/
theorem parabola_vertex (a h k : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * (x - h)^2 + k
  ∃! (x₀ y₀ : ℝ), (∀ x, f x ≥ f x₀) ∧ f x₀ = y₀ ∧ (x₀, y₀) = (h, k) :=
sorry

/-- The vertex of the parabola y = -2(x-3)^2 - 2 is (3, -2) --/
theorem specific_parabola_vertex :
  let f : ℝ → ℝ := λ x ↦ -2 * (x - 3)^2 - 2
  ∃! (x₀ y₀ : ℝ), (∀ x, f x ≥ f x₀) ∧ f x₀ = y₀ ∧ (x₀, y₀) = (3, -2) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_specific_parabola_vertex_l859_85931


namespace NUMINAMATH_CALUDE_arithmetic_equality_l859_85969

theorem arithmetic_equality : 1234562 - 12 * 3 * 2 = 1234490 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l859_85969


namespace NUMINAMATH_CALUDE_lines_perpendicular_l859_85915

/-- A line passing through a point (1, 1) with equation 2x - ay - 1 = 0 -/
def line_l1 (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1 - a * p.2 - 1 = 0 ∧ p = (1, 1)}

/-- A line with equation x + 2y = 0 -/
def line_l2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + 2 * p.2 = 0}

/-- Two lines are perpendicular if their slopes multiply to -1 -/
def perpendicular (l1 l2 : Set (ℝ × ℝ)) : Prop :=
  ∃ m1 m2 : ℝ, (∀ p ∈ l1, ∀ q ∈ l1, p ≠ q → (p.2 - q.2) = m1 * (p.1 - q.1)) ∧
                (∀ p ∈ l2, ∀ q ∈ l2, p ≠ q → (p.2 - q.2) = m2 * (p.1 - q.1)) ∧
                m1 * m2 = -1

theorem lines_perpendicular :
  ∃ a : ℝ, perpendicular (line_l1 a) line_l2 :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_l859_85915


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l859_85980

theorem geometric_sequence_fourth_term 
  (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ ≠ 0)
  (h₂ : a₂ = 3 * a₁ + 3)
  (h₃ : a₃ = 6 * a₁ + 6)
  (h₄ : a₂^2 = a₁ * a₃)  -- Condition for geometric sequence
  : ∃ (r : ℝ), r ≠ 0 ∧ a₂ = r * a₁ ∧ a₃ = r * a₂ ∧ r * a₃ = -24 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l859_85980


namespace NUMINAMATH_CALUDE_total_rooms_to_paint_l859_85921

/-- Proves that the total number of rooms to be painted is 9 -/
theorem total_rooms_to_paint (hours_per_room : ℕ) (rooms_painted : ℕ) (hours_remaining : ℕ) : 
  hours_per_room = 8 → rooms_painted = 5 → hours_remaining = 32 →
  rooms_painted + (hours_remaining / hours_per_room) = 9 :=
by
  sorry

#check total_rooms_to_paint

end NUMINAMATH_CALUDE_total_rooms_to_paint_l859_85921


namespace NUMINAMATH_CALUDE_square_rhombus_diagonal_distinction_l859_85972

/-- A quadrilateral with four equal sides -/
structure Rhombus :=
  (side_length : ℝ)
  (diagonal1 : ℝ)
  (diagonal2 : ℝ)

/-- A square is a rhombus with equal diagonals -/
structure Square extends Rhombus :=
  (diagonals_equal : diagonal1 = diagonal2)

/-- Theorem stating that squares have equal diagonals, but rhombuses don't necessarily have this property -/
theorem square_rhombus_diagonal_distinction :
  ∃ (s : Square) (r : Rhombus), s.diagonal1 = s.diagonal2 ∧ r.diagonal1 ≠ r.diagonal2 :=
sorry

end NUMINAMATH_CALUDE_square_rhombus_diagonal_distinction_l859_85972


namespace NUMINAMATH_CALUDE_minimum_reciprocal_sum_l859_85900

theorem minimum_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 20) :
  (1 / x + 1 / y) ≥ 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_minimum_reciprocal_sum_l859_85900


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l859_85964

/-- Calculate the average speed for a round trip given specific segments and speeds -/
theorem round_trip_average_speed 
  (total_distance : ℝ)
  (train_distance train_speed : ℝ)
  (car_to_y_distance car_to_y_speed : ℝ)
  (bus_distance bus_speed : ℝ)
  (car_return_distance car_return_speed : ℝ)
  (plane_speed : ℝ)
  (h1 : total_distance = 1500)
  (h2 : train_distance = 500)
  (h3 : train_speed = 60)
  (h4 : car_to_y_distance = 700)
  (h5 : car_to_y_speed = 50)
  (h6 : bus_distance = 300)
  (h7 : bus_speed = 40)
  (h8 : car_return_distance = 600)
  (h9 : car_return_speed = 60)
  (h10 : plane_speed = 500)
  : ∃ (average_speed : ℝ), abs (average_speed - 72.03) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l859_85964


namespace NUMINAMATH_CALUDE_shirts_before_buying_l859_85976

/-- Given that Sarah bought new shirts and now has a total number of shirts,
    prove that the number of shirts she had before is the difference between
    the total and the new shirts. -/
theorem shirts_before_buying (total : ℕ) (new : ℕ) (before : ℕ) 
    (h1 : total = before + new) : before = total - new := by
  sorry

end NUMINAMATH_CALUDE_shirts_before_buying_l859_85976


namespace NUMINAMATH_CALUDE_integer_difference_l859_85914

theorem integer_difference (x y : ℤ) : 
  x = 32 → y = 5 * x + 2 → y - x = 130 := by
  sorry

end NUMINAMATH_CALUDE_integer_difference_l859_85914


namespace NUMINAMATH_CALUDE_equation_solution_l859_85952

theorem equation_solution (x : ℝ) : x ≠ -2 → (-2 * x^2 = (4 * x + 2) / (x + 2)) ↔ (x = -1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l859_85952


namespace NUMINAMATH_CALUDE_percentage_problem_l859_85966

theorem percentage_problem (N P : ℝ) : 
  N = 150 → N = (P / 100) * N + 126 → P = 16 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l859_85966


namespace NUMINAMATH_CALUDE_inverse_f_at_seven_l859_85979

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 + 3

-- State the theorem
theorem inverse_f_at_seven (x : ℝ) : f x = 7 → x = 101 := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_at_seven_l859_85979


namespace NUMINAMATH_CALUDE_flower_town_coin_impossibility_l859_85903

/-- Represents the number of inhabitants in Flower Town -/
def num_inhabitants : ℕ := 1990

/-- Represents the number of coins each inhabitant must give -/
def coins_per_inhabitant : ℕ := 10

/-- Represents a meeting between two inhabitants -/
structure Meeting where
  giver : Fin num_inhabitants
  receiver : Fin num_inhabitants
  giver_gives_10 : Bool

/-- The main theorem stating the impossibility of the scenario -/
theorem flower_town_coin_impossibility :
  ¬ ∃ (meetings : List Meeting),
    (∀ i : Fin num_inhabitants, 
      (meetings.filter (λ m => m.giver = i ∨ m.receiver = i)).length = coins_per_inhabitant) ∧
    (∀ m : Meeting, m ∈ meetings → m.giver ≠ m.receiver) :=
by
  sorry


end NUMINAMATH_CALUDE_flower_town_coin_impossibility_l859_85903


namespace NUMINAMATH_CALUDE_cards_kept_away_is_two_l859_85946

/-- The number of cards in a standard deck of playing cards. -/
def standard_deck_size : ℕ := 52

/-- The number of cards used for playing. -/
def cards_used : ℕ := 50

/-- The number of cards kept away. -/
def cards_kept_away : ℕ := standard_deck_size - cards_used

theorem cards_kept_away_is_two : cards_kept_away = 2 := by
  sorry

end NUMINAMATH_CALUDE_cards_kept_away_is_two_l859_85946


namespace NUMINAMATH_CALUDE_coordinate_sum_theorem_l859_85920

theorem coordinate_sum_theorem (g : ℝ → ℝ) (h : g 4 = 7) :
  ∃ (x y : ℝ), 3 * y = 2 * g (3 * x) + 6 ∧ x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_coordinate_sum_theorem_l859_85920


namespace NUMINAMATH_CALUDE_triangle_inradius_l859_85913

/-- Given a triangle with perimeter 48 cm and area 60 cm², its inradius is 2.5 cm. -/
theorem triangle_inradius (p : ℝ) (A : ℝ) (r : ℝ) 
  (h1 : p = 48) 
  (h2 : A = 60) 
  (h3 : A = r * p / 2) : 
  r = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l859_85913


namespace NUMINAMATH_CALUDE_no_inscribed_circle_l859_85962

/-- A pentagon is represented by a list of its side lengths -/
def Pentagon := List ℝ

/-- Check if a list represents a valid pentagon with sides 1, 2, 5, 6, 7 -/
def isValidPentagon (p : Pentagon) : Prop :=
  p.length = 5 ∧ p.toFinset = {1, 2, 5, 6, 7}

/-- Sum of three elements in a list -/
def sumThree (l : List ℝ) (i j k : ℕ) : ℝ :=
  (l.get? i).getD 0 + (l.get? j).getD 0 + (l.get? k).getD 0

/-- Check if the sum of two non-adjacent sides is greater than or equal to
    the sum of the remaining three sides -/
def hasInvalidPair (p : Pentagon) : Prop :=
  (p.get? 0).getD 0 + (p.get? 2).getD 0 ≥ sumThree p 1 3 4 ∨
  (p.get? 0).getD 0 + (p.get? 3).getD 0 ≥ sumThree p 1 2 4 ∨
  (p.get? 1).getD 0 + (p.get? 3).getD 0 ≥ sumThree p 0 2 4 ∨
  (p.get? 1).getD 0 + (p.get? 4).getD 0 ≥ sumThree p 0 2 3 ∨
  (p.get? 2).getD 0 + (p.get? 4).getD 0 ≥ sumThree p 0 1 3

theorem no_inscribed_circle (p : Pentagon) (h : isValidPentagon p) :
  hasInvalidPair p := by
  sorry


end NUMINAMATH_CALUDE_no_inscribed_circle_l859_85962


namespace NUMINAMATH_CALUDE_compound_interest_rate_l859_85912

theorem compound_interest_rate (P r : ℝ) (h1 : P * (1 + r / 100) ^ 2 = 3650) (h2 : P * (1 + r / 100) ^ 3 = 4015) : r = 10 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l859_85912


namespace NUMINAMATH_CALUDE_range_of_a_l859_85918

noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x + Real.exp x - 1 / Real.exp x

theorem range_of_a (a : ℝ) (h : f (a - 1) + f (2 * a^2) ≤ 0) : -1 ≤ a ∧ a ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l859_85918


namespace NUMINAMATH_CALUDE_sample_size_equals_surveyed_parents_l859_85934

/-- Represents a school survey about students' daily activities -/
structure SchoolSurvey where
  total_students : ℕ
  surveyed_parents : ℕ
  sleep_6_to_7_hours_percentage : ℚ
  homework_3_to_4_hours_percentage : ℚ

/-- The sample size of a school survey is equal to the number of surveyed parents -/
theorem sample_size_equals_surveyed_parents (survey : SchoolSurvey) 
  (h1 : survey.total_students = 1800)
  (h2 : survey.surveyed_parents = 1000)
  (h3 : survey.sleep_6_to_7_hours_percentage = 70/100)
  (h4 : survey.homework_3_to_4_hours_percentage = 28/100) :
  survey.surveyed_parents = 1000 := by
  sorry

#check sample_size_equals_surveyed_parents

end NUMINAMATH_CALUDE_sample_size_equals_surveyed_parents_l859_85934


namespace NUMINAMATH_CALUDE_candy_bar_cost_l859_85971

theorem candy_bar_cost (candy_bars : ℕ) (lollipops : ℕ) (lollipop_cost : ℚ)
  (snow_shoveling_fraction : ℚ) (driveway_charge : ℚ) (driveways : ℕ) :
  candy_bars = 2 →
  lollipops = 4 →
  lollipop_cost = 1/4 →
  snow_shoveling_fraction = 1/6 →
  driveway_charge = 3/2 →
  driveways = 10 →
  (driveway_charge * driveways * snow_shoveling_fraction - lollipops * lollipop_cost) / candy_bars = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l859_85971


namespace NUMINAMATH_CALUDE_max_k_value_l859_85904

theorem max_k_value (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  (∀ k : ℝ, (4 / (a - b) + 1 / (b - c) + k / (c - a) ≥ 0) → k ≤ 9) ∧ 
  (∃ k : ℝ, k = 9 ∧ 4 / (a - b) + 1 / (b - c) + k / (c - a) ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l859_85904


namespace NUMINAMATH_CALUDE_set_intersection_empty_l859_85959

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- State the theorem
theorem set_intersection_empty (a : ℝ) : 
  (A a ∩ B = ∅) ↔ ((1/2 ≤ a ∧ a ≤ 2) ∨ a > 3) := by sorry

end NUMINAMATH_CALUDE_set_intersection_empty_l859_85959


namespace NUMINAMATH_CALUDE_halfway_between_one_third_and_one_fifth_l859_85930

theorem halfway_between_one_third_and_one_fifth : 
  (1 / 3 + 1 / 5) / 2 = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_halfway_between_one_third_and_one_fifth_l859_85930


namespace NUMINAMATH_CALUDE_equation_solution_l859_85967

theorem equation_solution :
  ∃ x : ℝ, -((1 : ℝ) / 3) * x - 5 = 4 ∧ x = -27 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l859_85967


namespace NUMINAMATH_CALUDE_abs_equation_solution_l859_85948

theorem abs_equation_solution : ∃! x : ℝ, |x - 3| = 5 - 2*x := by
  sorry

end NUMINAMATH_CALUDE_abs_equation_solution_l859_85948


namespace NUMINAMATH_CALUDE_profit_percentage_doubling_l859_85955

theorem profit_percentage_doubling (cost_price : ℝ) (original_profit_percentage : ℝ) 
  (h1 : original_profit_percentage = 60) :
  let original_selling_price := cost_price * (1 + original_profit_percentage / 100)
  let new_selling_price := 2 * original_selling_price
  let new_profit := new_selling_price - cost_price
  let new_profit_percentage := (new_profit / cost_price) * 100
  new_profit_percentage = 220 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_doubling_l859_85955


namespace NUMINAMATH_CALUDE_solution_value_l859_85925

theorem solution_value (x a : ℝ) (h : 2 * 2 + a = 3) : a = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l859_85925


namespace NUMINAMATH_CALUDE_dianes_honey_harvest_l859_85922

/-- Diane's honey harvest calculation -/
theorem dianes_honey_harvest 
  (last_year_harvest : ℕ) 
  (harvest_increase : ℕ) 
  (h1 : last_year_harvest = 2479)
  (h2 : harvest_increase = 6085) : 
  last_year_harvest + harvest_increase = 8564 := by
  sorry

end NUMINAMATH_CALUDE_dianes_honey_harvest_l859_85922


namespace NUMINAMATH_CALUDE_clothes_batch_size_l859_85923

/-- Proves that the number of sets of clothes in a batch is 30, given the production rates of two workers and their time difference. -/
theorem clothes_batch_size :
  let wang_rate : ℚ := 3  -- Wang's production rate (sets per day)
  let li_rate : ℚ := 5    -- Li's production rate (sets per day)
  let time_diff : ℚ := 4  -- Time difference in days
  let batch_size : ℚ := (wang_rate * li_rate * time_diff) / (li_rate - wang_rate)
  batch_size = 30 := by
  sorry


end NUMINAMATH_CALUDE_clothes_batch_size_l859_85923


namespace NUMINAMATH_CALUDE_present_ages_of_deepak_and_rajat_l859_85924

-- Define the present ages as variables
variable (R D Ra : ℕ)

-- Define the conditions
def present_age_ratio : Prop := R / D = 4 / 3 ∧ Ra / D = 5 / 3
def rahul_future_age : Prop := R + 4 = 32
def rajat_future_age : Prop := Ra + 7 = 50

-- State the theorem
theorem present_ages_of_deepak_and_rajat 
  (h1 : present_age_ratio R D Ra)
  (h2 : rahul_future_age R)
  (h3 : rajat_future_age Ra) :
  D = 21 ∧ Ra = 43 := by
  sorry

end NUMINAMATH_CALUDE_present_ages_of_deepak_and_rajat_l859_85924


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_18_l859_85983

theorem factorization_of_2x_squared_minus_18 (x : ℝ) : 2 * x^2 - 18 = 2 * (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_18_l859_85983


namespace NUMINAMATH_CALUDE_crow_probability_l859_85998

/-- Represents the number of crows of each color on each tree -/
structure CrowDistribution where
  birch_white : ℕ
  birch_black : ℕ
  oak_white : ℕ
  oak_black : ℕ

/-- The probability that the number of white crows on the birch tree remains the same -/
def prob_same (d : CrowDistribution) : ℚ :=
  (d.birch_black * (d.oak_black + 1) + d.birch_white * (d.oak_white + 1)) / (50 * 51)

/-- The probability that the number of white crows on the birch tree changes -/
def prob_change (d : CrowDistribution) : ℚ :=
  (d.birch_black * d.oak_white + d.birch_white * d.oak_black) / (50 * 51)

theorem crow_probability (d : CrowDistribution) 
  (h1 : d.birch_white + d.birch_black = 50)
  (h2 : d.oak_white + d.oak_black = 50)
  (h3 : d.birch_white > 0)
  (h4 : d.oak_white > 0)
  (h5 : d.birch_black ≥ d.birch_white)
  (h6 : d.oak_black ≥ d.oak_white ∨ d.oak_black + 1 = d.oak_white) :
  prob_same d > prob_change d := by
  sorry

end NUMINAMATH_CALUDE_crow_probability_l859_85998


namespace NUMINAMATH_CALUDE_taxi_fare_for_100_miles_l859_85905

/-- Represents the taxi fare system -/
structure TaxiFare where
  fixedCharge : ℝ
  fixedDistance : ℝ
  proportionalRate : ℝ

/-- Calculates the fare for a given distance -/
def calculateFare (tf : TaxiFare) (distance : ℝ) : ℝ :=
  tf.fixedCharge + tf.proportionalRate * (distance - tf.fixedDistance)

theorem taxi_fare_for_100_miles 
  (tf : TaxiFare)
  (h1 : tf.fixedCharge = 20)
  (h2 : tf.fixedDistance = 10)
  (h3 : calculateFare tf 80 = 160) :
  calculateFare tf 100 = 200 := by
  sorry


end NUMINAMATH_CALUDE_taxi_fare_for_100_miles_l859_85905


namespace NUMINAMATH_CALUDE_no_primes_divisible_by_25_l859_85993

theorem no_primes_divisible_by_25 : ∀ p : ℕ, Nat.Prime p → ¬(25 ∣ p) := by
  sorry

end NUMINAMATH_CALUDE_no_primes_divisible_by_25_l859_85993


namespace NUMINAMATH_CALUDE_union_A_B_when_m_2_intersection_A_B_empty_iff_l859_85994

-- Define sets A and B
def A : Set ℝ := {x | (4 : ℝ) / (x + 1) > 1}
def B (m : ℝ) : Set ℝ := {x | (x - m - 4) * (x - m + 1) > 0}

-- Part 1
theorem union_A_B_when_m_2 : A ∪ B 2 = {x : ℝ | x < 3 ∨ x > 6} := by sorry

-- Part 2
theorem intersection_A_B_empty_iff (m : ℝ) : A ∩ B m = ∅ ↔ -1 ≤ m ∧ m ≤ 0 := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_m_2_intersection_A_B_empty_iff_l859_85994


namespace NUMINAMATH_CALUDE_class_ratio_proof_l859_85927

/-- Given a class of students, prove that the ratio of boys in Grade A to girls in Grade B is 2:9. -/
theorem class_ratio_proof (S : ℚ) (G : ℚ) (B : ℚ) 
  (h1 : (1/3) * G = (1/4) * S) 
  (h2 : S = B + G) 
  (h3 : (2/5) * B > 0) 
  (h4 : (3/5) * G > 0) : 
  ((2/5) * B) / ((3/5) * G) = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_class_ratio_proof_l859_85927


namespace NUMINAMATH_CALUDE_division_multiplication_error_percentage_l859_85973

theorem division_multiplication_error_percentage (x : ℝ) (h : x > 0) :
  ∃ (ε : ℝ), ε ≥ 0 ∧ ε < 0.5 ∧
  (|(x / 8 - 8 * x)| / (8 * x)) * 100 = 98 + ε := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_error_percentage_l859_85973


namespace NUMINAMATH_CALUDE_average_age_proof_l859_85987

def john_age (mary_age : ℕ) : ℕ := 2 * mary_age

def tonya_age : ℕ := 60

theorem average_age_proof (mary_age : ℕ) (h1 : john_age mary_age = tonya_age / 2) :
  (mary_age + john_age mary_age + tonya_age) / 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_average_age_proof_l859_85987


namespace NUMINAMATH_CALUDE_initial_amount_calculation_l859_85937

/-- Given a person receives additional money and the difference between their
initial amount and the received amount is known, calculate their initial amount. -/
theorem initial_amount_calculation (received : ℕ) (difference : ℕ) : 
  received = 13 → difference = 11 → received + difference = 24 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_calculation_l859_85937


namespace NUMINAMATH_CALUDE_tan_three_pi_fourth_l859_85901

theorem tan_three_pi_fourth : Real.tan (3 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_three_pi_fourth_l859_85901


namespace NUMINAMATH_CALUDE_sin_difference_monotone_increasing_l859_85929

/-- The function f(x) = sin(2x - π/3) - sin(2x) is monotonically increasing on [π/12, 7π/12] -/
theorem sin_difference_monotone_increasing :
  let f : ℝ → ℝ := λ x ↦ Real.sin (2 * x - π / 3) - Real.sin (2 * x)
  ∀ x y, π / 12 ≤ x ∧ x < y ∧ y ≤ 7 * π / 12 → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_sin_difference_monotone_increasing_l859_85929


namespace NUMINAMATH_CALUDE_polynomial_must_be_constant_l859_85982

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- Sum of decimal digits of an integer's absolute value -/
def sumDecimalDigits (n : ℤ) : ℕ :=
  sorry

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Predicate for Fibonacci numbers -/
def isFibonacci (n : ℕ) : Prop :=
  ∃ k, fib k = n

theorem polynomial_must_be_constant (P : IntPolynomial) :
  (∀ n : ℕ, n > 0 → ¬isFibonacci (sumDecimalDigits (P.eval n))) →
  P.degree = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_must_be_constant_l859_85982


namespace NUMINAMATH_CALUDE_vector_expression_l859_85940

/-- Given vectors a, b, and c in ℝ², prove that c = a - 2b --/
theorem vector_expression (a b c : ℝ × ℝ) :
  a = (3, -2) → b = (-2, 1) → c = (7, -4) → c = a - 2 • b := by
  sorry

end NUMINAMATH_CALUDE_vector_expression_l859_85940


namespace NUMINAMATH_CALUDE_shoe_price_increase_l859_85933

theorem shoe_price_increase (regular_price : ℝ) (h : regular_price > 0) :
  let sale_price := regular_price * (1 - 0.2)
  (regular_price - sale_price) / sale_price * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_shoe_price_increase_l859_85933


namespace NUMINAMATH_CALUDE_dividing_line_sum_of_squares_l859_85957

/-- A circle in the first quadrant of the coordinate plane -/
structure Circle where
  diameter : ℝ
  center : ℝ × ℝ

/-- The region R formed by the union of ten circles -/
def region_R : Set (ℝ × ℝ) :=
  sorry

/-- The line m with slope -1 that divides region_R into two equal areas -/
structure DividingLine where
  a : ℕ
  b : ℕ
  c : ℕ
  slope_neg_one : a = b
  positive : 0 < a ∧ 0 < b ∧ 0 < c
  coprime : Nat.gcd a (Nat.gcd b c) = 1
  divides_equally : sorry

/-- Theorem stating that for the line m, a^2 + b^2 + c^2 = 6 -/
theorem dividing_line_sum_of_squares (m : DividingLine) :
  m.a^2 + m.b^2 + m.c^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_dividing_line_sum_of_squares_l859_85957


namespace NUMINAMATH_CALUDE_original_rectangle_area_l859_85961

/-- Given a rectangle whose dimensions are doubled to form a new rectangle with an area of 32 square meters, 
    prove that the area of the original rectangle is 8 square meters. -/
theorem original_rectangle_area (original_width original_height : ℝ) : 
  original_width > 0 → 
  original_height > 0 → 
  (2 * original_width) * (2 * original_height) = 32 → 
  original_width * original_height = 8 := by
sorry

end NUMINAMATH_CALUDE_original_rectangle_area_l859_85961


namespace NUMINAMATH_CALUDE_seymour_fertilizer_calculation_l859_85990

/-- Calculates the total fertilizer needed for Seymour's plant shop --/
theorem seymour_fertilizer_calculation : 
  let petunia_flats : ℕ := 4
  let petunias_per_flat : ℕ := 8
  let petunia_fertilizer : ℕ := 8
  let rose_flats : ℕ := 3
  let roses_per_flat : ℕ := 6
  let rose_fertilizer : ℕ := 3
  let sunflower_flats : ℕ := 5
  let sunflowers_per_flat : ℕ := 10
  let sunflower_fertilizer : ℕ := 6
  let orchid_flats : ℕ := 2
  let orchids_per_flat : ℕ := 4
  let orchid_fertilizer : ℕ := 4
  let venus_flytraps : ℕ := 2
  let venus_flytrap_fertilizer : ℕ := 2
  
  petunia_flats * petunias_per_flat * petunia_fertilizer +
  rose_flats * roses_per_flat * rose_fertilizer +
  sunflower_flats * sunflowers_per_flat * sunflower_fertilizer +
  orchid_flats * orchids_per_flat * orchid_fertilizer +
  venus_flytraps * venus_flytrap_fertilizer = 646 := by
  sorry

#check seymour_fertilizer_calculation

end NUMINAMATH_CALUDE_seymour_fertilizer_calculation_l859_85990


namespace NUMINAMATH_CALUDE_unique_a_for_linear_equation_l859_85977

def is_linear_equation (a : ℝ) : Prop :=
  (|a| - 1 = 1) ∧ (a - 2 ≠ 0)

theorem unique_a_for_linear_equation :
  ∃! a : ℝ, is_linear_equation a ∧ a = -2 :=
sorry

end NUMINAMATH_CALUDE_unique_a_for_linear_equation_l859_85977


namespace NUMINAMATH_CALUDE_water_polo_team_selection_l859_85970

theorem water_polo_team_selection (total_members : ℕ) (starting_team_size : ℕ) (goalie_count : ℕ) :
  total_members = 18 →
  starting_team_size = 8 →
  goalie_count = 1 →
  (total_members.choose goalie_count) * ((total_members - goalie_count).choose (starting_team_size - goalie_count)) = 222768 :=
by sorry

end NUMINAMATH_CALUDE_water_polo_team_selection_l859_85970


namespace NUMINAMATH_CALUDE_sphere_radius_from_intersection_l859_85911

theorem sphere_radius_from_intersection (r h : ℝ) : 
  r > 0 → h > 0 → r^2 + h^2 = (r + h)^2 → r = 12 → h = 8 → r + h = 13 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_from_intersection_l859_85911


namespace NUMINAMATH_CALUDE_incorrect_inequality_l859_85906

theorem incorrect_inequality (a b : ℝ) (h : a > b) : ¬(-3 * a > -3 * b) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_l859_85906


namespace NUMINAMATH_CALUDE_all_setC_are_polyhedra_setC_consists_entirely_of_polyhedra_l859_85945

-- Define the type for geometric bodies
inductive GeometricBody
  | TriangularPrism
  | QuadrangularPyramid
  | Cube
  | HexagonalPyramid
  | Sphere
  | Cone
  | Frustum
  | Hemisphere

-- Define a predicate for polyhedra
def isPolyhedron : GeometricBody → Prop
  | GeometricBody.TriangularPrism => True
  | GeometricBody.QuadrangularPyramid => True
  | GeometricBody.Cube => True
  | GeometricBody.HexagonalPyramid => True
  | _ => False

-- Define the set of geometric bodies in option C
def setC : List GeometricBody :=
  [GeometricBody.TriangularPrism, GeometricBody.QuadrangularPyramid,
   GeometricBody.Cube, GeometricBody.HexagonalPyramid]

-- Theorem: All elements in setC are polyhedra
theorem all_setC_are_polyhedra : ∀ x ∈ setC, isPolyhedron x := by
  sorry

-- Main theorem: setC consists entirely of polyhedra
theorem setC_consists_entirely_of_polyhedra : 
  (∀ x ∈ setC, isPolyhedron x) ∧ (setC ≠ []) := by
  sorry

end NUMINAMATH_CALUDE_all_setC_are_polyhedra_setC_consists_entirely_of_polyhedra_l859_85945


namespace NUMINAMATH_CALUDE_solution_equality_l859_85943

theorem solution_equality (a b c d : ℝ) 
  (eq1 : a - Real.sqrt (1 - b^2) + Real.sqrt (1 - c^2) = d)
  (eq2 : b - Real.sqrt (1 - c^2) + Real.sqrt (1 - d^2) = a)
  (eq3 : c - Real.sqrt (1 - d^2) + Real.sqrt (1 - a^2) = b)
  (eq4 : d - Real.sqrt (1 - a^2) + Real.sqrt (1 - b^2) = c)
  (nonneg1 : 1 - a^2 ≥ 0)
  (nonneg2 : 1 - b^2 ≥ 0)
  (nonneg3 : 1 - c^2 ≥ 0)
  (nonneg4 : 1 - d^2 ≥ 0) :
  a = b ∧ b = c ∧ c = d := by
  sorry

end NUMINAMATH_CALUDE_solution_equality_l859_85943


namespace NUMINAMATH_CALUDE_sqrt_difference_inequality_l859_85944

theorem sqrt_difference_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  Real.sqrt a - Real.sqrt b < Real.sqrt (a - b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_inequality_l859_85944


namespace NUMINAMATH_CALUDE_simons_treasures_l859_85916

def sand_dollars : ℕ := 10

def sea_glass (sand_dollars : ℕ) : ℕ := 3 * sand_dollars

def seashells (sea_glass : ℕ) : ℕ := 5 * sea_glass

def total_treasures (sand_dollars sea_glass seashells : ℕ) : ℕ :=
  sand_dollars + sea_glass + seashells

theorem simons_treasures :
  total_treasures sand_dollars (sea_glass sand_dollars) (seashells (sea_glass sand_dollars)) = 190 := by
  sorry

end NUMINAMATH_CALUDE_simons_treasures_l859_85916


namespace NUMINAMATH_CALUDE_calculation_one_l859_85999

theorem calculation_one :
  (27 : ℝ) ^ (1/3) + (1/9).sqrt / (-2/3) + |(-(1/2))| = 3 := by sorry

end NUMINAMATH_CALUDE_calculation_one_l859_85999


namespace NUMINAMATH_CALUDE_fourth_year_students_without_glasses_l859_85984

theorem fourth_year_students_without_glasses 
  (total_students : ℕ) 
  (fourth_year_students : ℕ) 
  (students_with_glasses : ℕ) 
  (students_without_glasses : ℕ) :
  total_students = 8 * fourth_year_students - 32 →
  students_with_glasses = students_without_glasses + 10 →
  total_students = 1152 →
  fourth_year_students = students_with_glasses + students_without_glasses →
  students_without_glasses = 69 :=
by sorry

end NUMINAMATH_CALUDE_fourth_year_students_without_glasses_l859_85984


namespace NUMINAMATH_CALUDE_bridge_length_bridge_length_proof_l859_85974

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proof that the bridge length is 215 meters -/
theorem bridge_length_proof :
  bridge_length 160 45 30 = 215 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_bridge_length_proof_l859_85974


namespace NUMINAMATH_CALUDE_simplify_expression_l859_85902

theorem simplify_expression : (5 + 7 + 8) / 3 - 2 / 3 = 6 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l859_85902


namespace NUMINAMATH_CALUDE_project_solution_l859_85960

/-- Represents the time (in days) required for a person to complete the project alone. -/
structure ProjectTime where
  personA : ℝ
  personB : ℝ
  personC : ℝ

/-- Defines the conditions of the engineering project. -/
def ProjectConditions (t : ProjectTime) : Prop :=
  -- Person B works alone for 4 days
  4 / t.personB +
  -- Persons A and C work together for 6 days
  6 * (1 / t.personA + 1 / t.personC) +
  -- Person A completes the remaining work in 9 days
  9 / t.personA = 1 ∧
  -- Work completed by Person B is 1/3 of the work completed by Person A
  t.personB = 3 * t.personA ∧
  -- Work completed by Person C is 2 times the work completed by Person B
  t.personC = t.personB / 2

/-- Theorem stating the solution to the engineering project problem. -/
theorem project_solution :
  ∃ t : ProjectTime, ProjectConditions t ∧ t.personA = 30 ∧ t.personB = 24 ∧ t.personC = 18 :=
by sorry

end NUMINAMATH_CALUDE_project_solution_l859_85960


namespace NUMINAMATH_CALUDE_seating_theorem_standing_theorem_distribution_theorem_l859_85953

/- Problem 1 -/
def seating_arrangements (total_seats : ℕ) (people : ℕ) : ℕ :=
  sorry

theorem seating_theorem : seating_arrangements 8 3 = 24 := by
  sorry

/- Problem 2 -/
def standing_arrangements (total_people : ℕ) (condition : Bool) : ℕ :=
  sorry

theorem standing_theorem : standing_arrangements 5 true = 60 := by
  sorry

/- Problem 3 -/
def distribute_spots (total_spots : ℕ) (schools : ℕ) : ℕ :=
  sorry

theorem distribution_theorem : distribute_spots 10 7 = 84 := by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_standing_theorem_distribution_theorem_l859_85953


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l859_85975

theorem sphere_volume_ratio (S₁ S₂ V₁ V₂ : ℝ) (h_positive : S₁ > 0 ∧ S₂ > 0) (h_surface_ratio : S₁ / S₂ = 1 / 3) : 
  V₁ / V₂ = 1 / (3 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l859_85975


namespace NUMINAMATH_CALUDE_regular_octagon_side_length_l859_85963

/-- A regular octagon is a polygon with 8 sides of equal length and 8 angles of equal measure. -/
structure RegularOctagon where
  sideLength : ℝ
  perimeter : ℝ

/-- The perimeter of a regular octagon is 8 times the length of one side. -/
def RegularOctagon.perimeterFormula (o : RegularOctagon) : ℝ :=
  8 * o.sideLength

theorem regular_octagon_side_length (o : RegularOctagon) 
    (h : o.perimeter = 23.6) : o.sideLength = 2.95 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_side_length_l859_85963


namespace NUMINAMATH_CALUDE_sum_of_m_and_n_l859_85909

theorem sum_of_m_and_n (m n : ℝ) (h : |m - 2| + |n - 6| = 0) : m + n = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_m_and_n_l859_85909


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l859_85995

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  3 * X^2 - 22 * X + 58 = (X - 6) * q + 34 := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l859_85995
