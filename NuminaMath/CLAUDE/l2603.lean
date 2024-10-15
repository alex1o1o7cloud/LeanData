import Mathlib

namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l2603_260344

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the point corresponding to i + i^2
def point : ℂ := i + i^2

-- Theorem stating that the point is in the second quadrant
theorem point_in_second_quadrant : 
  Complex.re point < 0 ∧ Complex.im point > 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l2603_260344


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l2603_260357

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t ≠ 0 ∧ a.1 = t * b.1 ∧ a.2 = t * b.2

/-- Given vectors a and b, if they are parallel, then k = 1/2 -/
theorem parallel_vectors_k_value (k : ℝ) :
  let a : ℝ × ℝ := (1, k)
  let b : ℝ × ℝ := (2, 1)
  are_parallel a b → k = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l2603_260357


namespace NUMINAMATH_CALUDE_consecutive_sum_fifteen_l2603_260343

theorem consecutive_sum_fifteen (n : ℤ) : n + (n + 1) + (n + 2) = 15 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_sum_fifteen_l2603_260343


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2603_260352

theorem decimal_to_fraction : 
  ∃ (n : ℕ) (d : ℕ), d ≠ 0 ∧ (0.4 + (3 : ℚ) / 99) = (n : ℚ) / d := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2603_260352


namespace NUMINAMATH_CALUDE_custom_op_theorem_l2603_260390

/-- Custom operation ⊗ defined as x ⊗ y = x^3 + y^3 -/
def custom_op (x y : ℝ) : ℝ := x^3 + y^3

/-- Theorem stating that h ⊗ (h ⊗ h) = h^3 + 8h^9 -/
theorem custom_op_theorem (h : ℝ) : custom_op h (custom_op h h) = h^3 + 8*h^9 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_theorem_l2603_260390


namespace NUMINAMATH_CALUDE_bazylev_inequality_l2603_260349

theorem bazylev_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^y + y^z + z^x > 1 := by
  sorry

end NUMINAMATH_CALUDE_bazylev_inequality_l2603_260349


namespace NUMINAMATH_CALUDE_logical_implications_l2603_260317

theorem logical_implications (p q : Prop) : 
  (((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q))) ∧
  (((p ∧ q) → ¬(¬p)) ∧ ¬(¬(¬p) → (p ∧ q))) ∧
  ((¬p → ¬(p ∧ q)) ∧ ¬(¬(p ∧ q) → ¬p)) := by
  sorry

end NUMINAMATH_CALUDE_logical_implications_l2603_260317


namespace NUMINAMATH_CALUDE_secret_code_count_l2603_260392

/-- The number of colors available -/
def num_colors : ℕ := 7

/-- The number of slots to fill -/
def num_slots : ℕ := 5

/-- The number of possible secret codes -/
def num_codes : ℕ := 2520

/-- Theorem: The number of ways to arrange 5 colors chosen from 7 distinct colors is 2520 -/
theorem secret_code_count :
  (Finset.card (Finset.range num_colors)).factorial / 
  (Finset.card (Finset.range (num_colors - num_slots))).factorial = num_codes :=
by sorry

end NUMINAMATH_CALUDE_secret_code_count_l2603_260392


namespace NUMINAMATH_CALUDE_tax_calculation_l2603_260334

/-- Given a monthly income and a tax rate, calculates the amount paid in taxes -/
def calculate_tax (income : ℝ) (tax_rate : ℝ) : ℝ :=
  income * tax_rate

/-- Proves that for a monthly income of 2120 dollars and a tax rate of 0.4, 
    the amount paid in taxes is 848 dollars -/
theorem tax_calculation :
  calculate_tax 2120 0.4 = 848 := by
sorry

end NUMINAMATH_CALUDE_tax_calculation_l2603_260334


namespace NUMINAMATH_CALUDE_johns_spending_l2603_260323

/-- Given John's allowance B, prove that he spends 4/13 of B on movie ticket and soda combined -/
theorem johns_spending (B : ℝ) (t d : ℝ) 
  (ht : t = 0.25 * (B - d)) 
  (hd : d = 0.1 * (B - t)) : 
  t + d = (4 / 13) * B := by
sorry

end NUMINAMATH_CALUDE_johns_spending_l2603_260323


namespace NUMINAMATH_CALUDE_tree_planting_equation_l2603_260394

/-- Represents the tree planting scenario -/
structure TreePlanting where
  total_trees : ℕ := 480
  days_saved : ℕ := 4
  new_rate : ℝ
  original_rate : ℝ

/-- The new rate is 1/3 more than the original rate -/
axiom rate_increase {tp : TreePlanting} : tp.new_rate = (4/3) * tp.original_rate

/-- The equation correctly represents the tree planting scenario -/
theorem tree_planting_equation (tp : TreePlanting) :
  (tp.total_trees / (tp.original_rate)) - (tp.total_trees / tp.new_rate) = tp.days_saved := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_equation_l2603_260394


namespace NUMINAMATH_CALUDE_bacteria_growth_proof_l2603_260337

/-- The number of quadrupling cycles in two minutes -/
def quadrupling_cycles : ℕ := 8

/-- The number of bacteria after two minutes -/
def final_bacteria_count : ℕ := 4194304

/-- The growth factor for each cycle -/
def growth_factor : ℕ := 4

/-- The initial number of bacteria -/
def initial_bacteria : ℕ := 64

theorem bacteria_growth_proof :
  initial_bacteria * growth_factor ^ quadrupling_cycles = final_bacteria_count :=
by sorry

end NUMINAMATH_CALUDE_bacteria_growth_proof_l2603_260337


namespace NUMINAMATH_CALUDE_all_positives_can_be_written_l2603_260316

/-- The predicate that determines if a number can be written on the board -/
def CanBeWritten (n : ℕ) : Prop :=
  ∃ (sequence : ℕ → ℕ), sequence 0 = 1 ∧
  (∀ k, ∃ b, (sequence k + b + 1) ∣ (sequence k^2 + b^2 + 1) ∧
            sequence (k + 1) = b)

/-- The main theorem stating that any positive integer can be written on the board -/
theorem all_positives_can_be_written :
  ∀ n : ℕ, n > 0 → CanBeWritten n :=
sorry

end NUMINAMATH_CALUDE_all_positives_can_be_written_l2603_260316


namespace NUMINAMATH_CALUDE_red_balls_count_l2603_260396

/-- Given a bag with red and blue balls, if the total number of balls is 12
    and the probability of drawing two red balls at the same time is 1/18,
    then the number of red balls is 3. -/
theorem red_balls_count (total : ℕ) (red : ℕ) (prob : ℚ) :
  total = 12 →
  prob = 1 / 18 →
  prob = (red / total) * ((red - 1) / (total - 1)) →
  red = 3 :=
sorry

end NUMINAMATH_CALUDE_red_balls_count_l2603_260396


namespace NUMINAMATH_CALUDE_largest_binomial_coefficient_fifth_term_l2603_260373

/-- 
Theorem: There exists a natural number n such that the binomial coefficient 
of the 5th term in the expansion of (x - 2/x)^n is the largest, and n = 7 
is one such value.
-/
theorem largest_binomial_coefficient_fifth_term : 
  ∃ n : ℕ, (
    -- The binomial coefficient of the 5th term is the largest
    ∀ k : ℕ, k ≤ n → (n.choose 4) ≥ (n.choose k)
  ) ∧ 
  -- n = 7 is a valid solution
  (7 : ℕ) ∈ { m : ℕ | ∀ k : ℕ, k ≤ m → (m.choose 4) ≥ (m.choose k) } :=
by sorry


end NUMINAMATH_CALUDE_largest_binomial_coefficient_fifth_term_l2603_260373


namespace NUMINAMATH_CALUDE_total_books_read_is_72cs_l2603_260377

/-- The total number of books read by the entire student body in one year -/
def total_books_read (c s : ℕ) : ℕ :=
  let books_per_month : ℕ := 6
  let months_per_year : ℕ := 12
  let books_per_student_per_year : ℕ := books_per_month * months_per_year
  let total_students : ℕ := c * s
  books_per_student_per_year * total_students

/-- Theorem stating that the total number of books read is 72cs -/
theorem total_books_read_is_72cs (c s : ℕ) :
  total_books_read c s = 72 * c * s := by
  sorry

end NUMINAMATH_CALUDE_total_books_read_is_72cs_l2603_260377


namespace NUMINAMATH_CALUDE_integer_roots_quadratic_l2603_260314

theorem integer_roots_quadratic (n : ℕ+) : 
  (∃ x : ℤ, x^2 - 4*x + n.val = 0) ↔ (n.val = 3 ∨ n.val = 4) := by
  sorry

end NUMINAMATH_CALUDE_integer_roots_quadratic_l2603_260314


namespace NUMINAMATH_CALUDE_hotel_rooms_rented_l2603_260339

theorem hotel_rooms_rented (total_rooms : ℝ) (h1 : total_rooms > 0) : 
  let air_conditioned := (3/5) * total_rooms
  let rented_air_conditioned := (2/3) * air_conditioned
  let not_rented := total_rooms - (rented_air_conditioned + (1/5) * air_conditioned)
  (total_rooms - not_rented) / total_rooms = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_hotel_rooms_rented_l2603_260339


namespace NUMINAMATH_CALUDE_vector_magnitude_sum_l2603_260327

/-- Given two vectors a and b in ℝ², prove that if |a| = 3, |b| = 4, 
    and a - b = (√2, √7), then |a + b| = √41 -/
theorem vector_magnitude_sum (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 3)
  (h2 : ‖b‖ = 4)
  (h3 : a - b = (Real.sqrt 2, Real.sqrt 7)) :
  ‖a + b‖ = Real.sqrt 41 := by
  sorry


end NUMINAMATH_CALUDE_vector_magnitude_sum_l2603_260327


namespace NUMINAMATH_CALUDE_may_scarf_count_l2603_260330

/-- Represents the number of scarves that can be made from one yarn of a given color -/
def scarvesPerYarn (color : String) : ℕ :=
  match color with
  | "red" => 3
  | "blue" => 2
  | "yellow" => 4
  | "green" => 5
  | "purple" => 6
  | _ => 0

/-- Represents the number of yarns May has for each color -/
def yarnCount (color : String) : ℕ :=
  match color with
  | "red" => 1
  | "blue" => 1
  | "yellow" => 1
  | "green" => 3
  | "purple" => 2
  | _ => 0

/-- The list of colors May has yarn for -/
def colors : List String := ["red", "blue", "yellow", "green", "purple"]

/-- The total number of scarves May can make -/
def totalScarves : ℕ := (colors.map (fun c => scarvesPerYarn c * yarnCount c)).sum

theorem may_scarf_count : totalScarves = 36 := by
  sorry

end NUMINAMATH_CALUDE_may_scarf_count_l2603_260330


namespace NUMINAMATH_CALUDE_sqrt_expressions_equality_l2603_260313

theorem sqrt_expressions_equality : 
  (2 * Real.sqrt 3 - 3 * Real.sqrt 12 + 5 * Real.sqrt 27 = 11 * Real.sqrt 3) ∧
  ((1 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 6) - (2 * Real.sqrt 3 - 1)^2 = 
   -2 * Real.sqrt 2 + 4 * Real.sqrt 3 - 13) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expressions_equality_l2603_260313


namespace NUMINAMATH_CALUDE_garage_door_properties_l2603_260347

/-- Represents a garage door mechanism -/
structure GarageDoor where
  AC : ℝ
  BC : ℝ
  CY : ℝ
  AX : ℝ
  BD : ℝ

/-- Properties of the garage door mechanism -/
def isValidGarageDoor (door : GarageDoor) : Prop :=
  door.AC = 0.5 ∧ door.BC = 0.5 ∧ door.CY = 0.5 ∧ door.AX = 1 ∧ door.BD = 2

/-- Calculate CR given XS -/
def calculateCR (door : GarageDoor) (XS : ℝ) : ℝ := sorry

/-- Check if Y's height remains constant -/
def isYHeightConstant (door : GarageDoor) : Prop := sorry

/-- Calculate DT given XT -/
def calculateDT (door : GarageDoor) (XT : ℝ) : ℝ := sorry

/-- Main theorem about the garage door mechanism -/
theorem garage_door_properties (door : GarageDoor) 
  (h : isValidGarageDoor door) : 
  calculateCR door 0.2 = 0.1 ∧ 
  isYHeightConstant door ∧ 
  calculateDT door 0.4 = 0.6 := by sorry

end NUMINAMATH_CALUDE_garage_door_properties_l2603_260347


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2603_260307

/-- Given that x³ + x + m = 7 when x = 1, prove that x³ + x + m = 3 when x = -1. -/
theorem algebraic_expression_value (m : ℝ) 
  (h : 1^3 + 1 + m = 7) : 
  (-1)^3 + (-1) + m = 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2603_260307


namespace NUMINAMATH_CALUDE_simultaneous_truth_probability_l2603_260338

/-- The probability of A telling the truth -/
def prob_A_truth : ℝ := 0.8

/-- The probability of B telling the truth -/
def prob_B_truth : ℝ := 0.6

/-- The probability of A and B telling the truth simultaneously -/
def prob_both_truth : ℝ := prob_A_truth * prob_B_truth

theorem simultaneous_truth_probability :
  prob_both_truth = 0.48 :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_truth_probability_l2603_260338


namespace NUMINAMATH_CALUDE_max_roses_for_680_l2603_260361

/-- Represents the price of roses in different quantities -/
structure RosePrices where
  individual : ℝ
  oneDozen : ℝ
  twoDozen : ℝ

/-- Calculates the maximum number of roses that can be purchased with a given budget -/
def maxRoses (prices : RosePrices) (budget : ℝ) : ℕ :=
  sorry

/-- The theorem stating the maximum number of roses that can be purchased for $680 -/
theorem max_roses_for_680 (prices : RosePrices) 
  (h1 : prices.individual = 4.5)
  (h2 : prices.oneDozen = 36)
  (h3 : prices.twoDozen = 50) :
  maxRoses prices 680 = 318 :=
sorry

end NUMINAMATH_CALUDE_max_roses_for_680_l2603_260361


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l2603_260333

theorem fraction_product_simplification :
  (3 : ℚ) / 4 * (4 : ℚ) / 5 * (5 : ℚ) / 6 * (6 : ℚ) / 7 * (7 : ℚ) / 8 = (3 : ℚ) / 8 :=
by sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l2603_260333


namespace NUMINAMATH_CALUDE_fraction_equality_l2603_260310

theorem fraction_equality (m : ℝ) (h : (m - 1) / m = 3) : (m^2 + 1) / m^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2603_260310


namespace NUMINAMATH_CALUDE_area_pda_equals_sqrt_vw_l2603_260329

-- Define the rectangular pyramid
structure RectangularPyramid where
  -- Lengths of edges
  a : ℝ
  b : ℝ
  h : ℝ
  -- Areas of triangles
  u : ℝ
  v : ℝ
  w : ℝ
  -- Conditions
  pos_a : 0 < a
  pos_b : 0 < b
  pos_h : 0 < h
  area_pab : u = (1/2) * a * h
  area_pbc : v = (1/2) * a * b
  area_pcd : w = (1/2) * b * h

-- Theorem statement
theorem area_pda_equals_sqrt_vw (pyramid : RectangularPyramid) :
  (1/2) * pyramid.b * pyramid.h = Real.sqrt (pyramid.v * pyramid.w) :=
sorry

end NUMINAMATH_CALUDE_area_pda_equals_sqrt_vw_l2603_260329


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l2603_260371

theorem no_solution_for_equation :
  ¬∃ (x : ℝ), x ≠ 1 ∧ x ≠ -1 ∧ (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l2603_260371


namespace NUMINAMATH_CALUDE_quadratic_roots_max_value_l2603_260358

/-- Given a quadratic x^2 - tx + q with roots α and β, where 
    α + β = α^2 + β^2 = α^3 + β^3 = ⋯ = α^2010 + β^2010,
    the maximum value of 1/α^2012 + 1/β^2012 is 2. -/
theorem quadratic_roots_max_value (t q α β : ℝ) : 
  α^2 - t*α + q = 0 →
  β^2 - t*β + q = 0 →
  (∀ n : ℕ, n ≤ 2010 → α^n + β^n = α + β) →
  (∃ M : ℝ, M = 2 ∧ 
    ∀ t' q' α' β' : ℝ, 
      α'^2 - t'*α' + q' = 0 →
      β'^2 - t'*β' + q' = 0 →
      (∀ n : ℕ, n ≤ 2010 → α'^n + β'^n = α' + β') →
      1/α'^2012 + 1/β'^2012 ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_max_value_l2603_260358


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_4_l2603_260348

/-- A geometric sequence with its sum -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum of the first n terms
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1  -- Geometric sequence property

/-- Theorem: For a geometric sequence satisfying given conditions, S_4 = 75 -/
theorem geometric_sequence_sum_4 (seq : GeometricSequence)
  (h1 : seq.a 3 - seq.a 1 = 15)
  (h2 : seq.a 2 - seq.a 1 = 5) :
  seq.S 4 = 75 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_4_l2603_260348


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l2603_260360

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a :=
by sorry

theorem sum_of_roots_specific_quadratic :
  let r₁ := (7 + Real.sqrt (49 - 48)) / 2
  let r₂ := (7 - Real.sqrt (49 - 48)) / 2
  r₁ + r₂ = 7 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l2603_260360


namespace NUMINAMATH_CALUDE_total_outstanding_credit_l2603_260350

/-- The total outstanding consumer installment credit in billions of dollars -/
def total_credit : ℝ := 416.67

/-- The percentage of automobile installment credit in total consumer installment credit -/
def auto_credit_percentage : ℝ := 36

/-- The amount of credit extended by automobile finance companies in billions of dollars -/
def auto_finance_credit : ℝ := 75

/-- Theorem stating the total outstanding consumer installment credit -/
theorem total_outstanding_credit : 
  total_credit = (2 * auto_finance_credit) / (auto_credit_percentage / 100) := by
  sorry

end NUMINAMATH_CALUDE_total_outstanding_credit_l2603_260350


namespace NUMINAMATH_CALUDE_greatest_perfect_square_under_1000_l2603_260309

theorem greatest_perfect_square_under_1000 : 
  ∀ n : ℕ, n < 1000 → n ≤ 961 ∨ ¬∃ m : ℕ, n = m^2 := by
  sorry

end NUMINAMATH_CALUDE_greatest_perfect_square_under_1000_l2603_260309


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l2603_260383

-- Define the variables as positive real numbers
variable (x y z : ℝ) 

-- Define the hypothesis that x, y, and z are positive
variable (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)

-- Define the main theorem
theorem cyclic_sum_inequality :
  let f (a b c : ℝ) := Real.sqrt (a / (b + c)) * Real.sqrt ((a * b + a * c + b^2 + c^2) / (b^2 + c^2))
  f x y z + f y z x + f z x y ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l2603_260383


namespace NUMINAMATH_CALUDE_fraction_not_on_time_l2603_260374

/-- Represents the attendees at the monthly meeting -/
structure Attendees where
  total : ℕ
  males : ℕ
  females : ℕ
  malesOnTime : ℕ
  femalesOnTime : ℕ

/-- The conditions of the problem -/
def meetingConditions (a : Attendees) : Prop :=
  a.males = (2 * a.total) / 3 ∧
  a.females = a.total - a.males ∧
  a.malesOnTime = (3 * a.males) / 4 ∧
  a.femalesOnTime = (5 * a.females) / 6

/-- The theorem to be proved -/
theorem fraction_not_on_time (a : Attendees) 
  (h : meetingConditions a) : 
  (a.total - (a.malesOnTime + a.femalesOnTime)) / a.total = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_fraction_not_on_time_l2603_260374


namespace NUMINAMATH_CALUDE_paths_with_consecutive_right_moves_l2603_260321

/-- The number of paths on a grid with specified conditions -/
def num_paths (horizontal_steps vertical_steps : ℕ) : ℕ :=
  Nat.choose (horizontal_steps + vertical_steps - 1) vertical_steps

/-- The main theorem stating the number of paths under given conditions -/
theorem paths_with_consecutive_right_moves :
  num_paths 7 6 = 924 :=
by
  sorry

end NUMINAMATH_CALUDE_paths_with_consecutive_right_moves_l2603_260321


namespace NUMINAMATH_CALUDE_total_weight_calculation_l2603_260351

/-- The molecular weight of a compound in grams per mole -/
def molecular_weight : ℝ := 1184

/-- The number of moles of the compound -/
def number_of_moles : ℝ := 4

/-- The total weight of the compound in grams -/
def total_weight : ℝ := number_of_moles * molecular_weight

theorem total_weight_calculation : total_weight = 4736 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_calculation_l2603_260351


namespace NUMINAMATH_CALUDE_integer_solution_squared_sum_eq_product_l2603_260359

theorem integer_solution_squared_sum_eq_product (a b c : ℤ) :
  a^2 + b^2 + c^2 = a^2 * b^2 → a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_squared_sum_eq_product_l2603_260359


namespace NUMINAMATH_CALUDE_range_of_m_l2603_260395

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - m*x + m - 1 = 0}

theorem range_of_m : ∀ m : ℝ, (A ∪ B m = A) → m = 3 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l2603_260395


namespace NUMINAMATH_CALUDE_paco_initial_salty_cookies_l2603_260315

/-- The number of salty cookies Paco had initially -/
def initial_salty_cookies : ℕ := sorry

/-- The number of sweet cookies Paco had initially -/
def initial_sweet_cookies : ℕ := 40

/-- The number of salty cookies Paco ate -/
def eaten_salty_cookies : ℕ := 28

/-- The number of sweet cookies Paco ate -/
def eaten_sweet_cookies : ℕ := 15

/-- The difference between salty and sweet cookies eaten -/
def salty_sweet_difference : ℕ := 13

theorem paco_initial_salty_cookies :
  initial_salty_cookies = 56 :=
by sorry

end NUMINAMATH_CALUDE_paco_initial_salty_cookies_l2603_260315


namespace NUMINAMATH_CALUDE_daily_toy_production_l2603_260331

/-- Given a factory that produces toys, this theorem proves the daily production
    when the weekly production and number of working days are known. -/
theorem daily_toy_production
  (weekly_production : ℕ)
  (working_days : ℕ)
  (h_weekly : weekly_production = 6500)
  (h_days : working_days = 5)
  (h_equal_daily : weekly_production % working_days = 0) :
  weekly_production / working_days = 1300 := by
  sorry

#check daily_toy_production

end NUMINAMATH_CALUDE_daily_toy_production_l2603_260331


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2603_260385

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem statement -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 * a 5 = -3/4 →
  a 2 + a 3 + a 4 + a 5 = 5/4 →
  1/a 2 + 1/a 3 + 1/a 4 + 1/a 5 = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2603_260385


namespace NUMINAMATH_CALUDE_prob_different_suits_l2603_260303

/-- Represents a standard 52-card deck -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards in the combined deck -/
def CombinedDeck : ℕ := 2 * StandardDeck

/-- Represents the number of cards of the same suit in the combined deck -/
def SameSuitCards : ℕ := 26

/-- The probability of drawing two cards of different suits from a pile of two shuffled standard 52-card decks -/
theorem prob_different_suits : 
  (CombinedDeck - 1 - SameSuitCards) / (CombinedDeck - 1 : ℚ) = 78 / 103 := by
sorry

end NUMINAMATH_CALUDE_prob_different_suits_l2603_260303


namespace NUMINAMATH_CALUDE_calculation_result_solution_set_l2603_260369

-- Problem 1
theorem calculation_result : (Real.pi - 2023) ^ 0 + |-Real.sqrt 3| - 2 * Real.sin (π / 3) = 1 := by sorry

-- Problem 2
def system_of_inequalities (x : ℝ) : Prop :=
  2 * (x + 3) ≥ 8 ∧ x < (x + 4) / 2

theorem solution_set :
  ∀ x : ℝ, system_of_inequalities x ↔ 1 ≤ x ∧ x < 4 := by sorry

end NUMINAMATH_CALUDE_calculation_result_solution_set_l2603_260369


namespace NUMINAMATH_CALUDE_chord_intersection_probability_l2603_260365

/-- Given 1988 points evenly distributed on a circle, this function represents
    the probability that chord PQ intersects chord RS when selecting four distinct points
    P, Q, R, and S with all quadruples being equally likely. -/
def probability_chords_intersect (n : ℕ) : ℚ :=
  if n = 1988 then 1/3 else 0

/-- Theorem stating that the probability of chord PQ intersecting chord RS
    is 1/3 when selecting 4 points from 1988 evenly distributed points on a circle. -/
theorem chord_intersection_probability :
  probability_chords_intersect 1988 = 1/3 := by sorry

end NUMINAMATH_CALUDE_chord_intersection_probability_l2603_260365


namespace NUMINAMATH_CALUDE_product_of_numbers_l2603_260336

theorem product_of_numbers (x y : ℝ) (sum_eq : x + y = 18) (sum_squares_eq : x^2 + y^2 = 180) : x * y = 72 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2603_260336


namespace NUMINAMATH_CALUDE_shyam_weight_increase_l2603_260320

-- Define the original weight ratio
def weight_ratio : ℚ := 7 / 9

-- Define Ram's weight increase percentage
def ram_increase : ℚ := 12 / 100

-- Define the total new weight
def total_new_weight : ℚ := 165.6

-- Define the total weight increase percentage
def total_increase : ℚ := 20 / 100

-- Theorem to prove
theorem shyam_weight_increase : ∃ (original_ram : ℚ) (original_shyam : ℚ),
  original_shyam = original_ram / weight_ratio ∧
  (original_ram * (1 + ram_increase) + original_shyam * (1 + x)) = total_new_weight ∧
  (original_ram + original_shyam) * (1 + total_increase) = total_new_weight ∧
  abs (x - 26.29 / 100) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_shyam_weight_increase_l2603_260320


namespace NUMINAMATH_CALUDE_max_handshakes_equals_combinations_l2603_260353

/-- The number of men in the group -/
def n : ℕ := 20

/-- The number of men involved in each handshake -/
def k : ℕ := 2

/-- Calculates the number of combinations of k items from n items -/
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem: The maximum number of unique pairwise handshakes among n men is equal to the number of combinations of k=2 men from n men -/
theorem max_handshakes_equals_combinations :
  combinations n k = 190 := by sorry

end NUMINAMATH_CALUDE_max_handshakes_equals_combinations_l2603_260353


namespace NUMINAMATH_CALUDE_martha_cakes_l2603_260306

theorem martha_cakes (num_children : ℕ) (cakes_per_child : ℕ) (h1 : num_children = 3) (h2 : cakes_per_child = 6) :
  num_children * cakes_per_child = 18 :=
by sorry

end NUMINAMATH_CALUDE_martha_cakes_l2603_260306


namespace NUMINAMATH_CALUDE_tan_sin_ratio_equals_three_l2603_260367

theorem tan_sin_ratio_equals_three :
  (Real.tan (20 * π / 180) + 4 * Real.sin (20 * π / 180)) / Real.tan (30 * π / 180) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sin_ratio_equals_three_l2603_260367


namespace NUMINAMATH_CALUDE_intersection_A_B_l2603_260387

def A : Set ℝ := {-2, -1, 1, 2, 4}

def B : Set ℝ := {x : ℝ | (x + 2) * (x - 3) < 0}

theorem intersection_A_B : A ∩ B = {-1, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2603_260387


namespace NUMINAMATH_CALUDE_sterilization_tank_capacity_l2603_260393

/-- The total capacity of the sterilization tank in gallons -/
def tank_capacity : ℝ := 100

/-- The initial concentration of bleach in the tank as a decimal -/
def initial_concentration : ℝ := 0.02

/-- The target concentration of bleach in the tank as a decimal -/
def target_concentration : ℝ := 0.05

/-- The amount of solution drained and replaced with pure bleach in gallons -/
def drained_amount : ℝ := 3.0612244898

theorem sterilization_tank_capacity :
  let initial_bleach := initial_concentration * tank_capacity
  let drained_bleach := initial_concentration * drained_amount
  let added_bleach := drained_amount
  let final_bleach := initial_bleach - drained_bleach + added_bleach
  final_bleach = target_concentration * tank_capacity := by
  sorry

end NUMINAMATH_CALUDE_sterilization_tank_capacity_l2603_260393


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2603_260335

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  right_angle : (R.1 - P.1) * (R.1 - Q.1) + (R.2 - P.2) * (R.2 - Q.2) = 0

-- Define the squares PQXY and PRWZ
structure Square (A B C D : ℝ × ℝ) : Prop where
  side_length_eq : (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2
  right_angle : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0

-- Define points on a circle
def OnCircle (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop :=
  (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2

-- Main theorem
theorem triangle_perimeter 
  (P Q R X Y Z W : ℝ × ℝ) 
  (h_triangle : Triangle P Q R)
  (h_pq_length : (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 100)
  (h_square_pq : Square P Q X Y)
  (h_square_pr : Square P R W Z)
  (h_circle : ∃ (center : ℝ × ℝ) (radius : ℝ), 
    OnCircle center radius X ∧ 
    OnCircle center radius Y ∧ 
    OnCircle center radius Z ∧ 
    OnCircle center radius W) :
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 + 
  (R.1 - P.1)^2 + (R.2 - P.2)^2 + 
  (R.1 - Q.1)^2 + (R.2 - Q.2)^2 = (10 + 10 * Real.sqrt 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2603_260335


namespace NUMINAMATH_CALUDE_AB_BA_parallel_l2603_260341

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = k • w ∨ w = k • v

/-- Vector AB is defined as the difference between points B and A -/
def vector_AB (A B : ℝ × ℝ) : ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2)

/-- Vector BA is defined as the difference between points A and B -/
def vector_BA (A B : ℝ × ℝ) : ℝ × ℝ :=
  (A.1 - B.1, A.2 - B.2)

/-- Theorem: Vectors AB and BA are parallel -/
theorem AB_BA_parallel (A B : ℝ × ℝ) :
  are_parallel (vector_AB A B) (vector_BA A B) := by
  sorry

end NUMINAMATH_CALUDE_AB_BA_parallel_l2603_260341


namespace NUMINAMATH_CALUDE_power_function_even_l2603_260366

theorem power_function_even (α : ℤ) (h1 : 0 ≤ α) (h2 : α ≤ 5) :
  (∀ x : ℝ, (fun x => x^(3 - α)) (-x) = (fun x => x^(3 - α)) x) → α = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_function_even_l2603_260366


namespace NUMINAMATH_CALUDE_interns_escape_probability_l2603_260325

/-- A permutation on n elements -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- The probability that a random permutation on n elements has no cycle longer than k -/
noncomputable def prob_no_long_cycle (n k : ℕ) : ℝ := sorry

/-- The number of interns/drawers -/
def num_interns : ℕ := 44

/-- The maximum allowed cycle length for survival -/
def max_cycle_length : ℕ := 21

/-- The minimum required survival probability -/
def min_survival_prob : ℝ := 0.30

theorem interns_escape_probability :
  prob_no_long_cycle num_interns max_cycle_length > min_survival_prob := by sorry

end NUMINAMATH_CALUDE_interns_escape_probability_l2603_260325


namespace NUMINAMATH_CALUDE_regular_polygon_144_degrees_has_10_sides_l2603_260386

/-- A regular polygon with interior angles of 144 degrees has 10 sides -/
theorem regular_polygon_144_degrees_has_10_sides :
  ∀ (n : ℕ), n > 2 →
  (180 * (n - 2) : ℚ) / n = 144 →
  n = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_144_degrees_has_10_sides_l2603_260386


namespace NUMINAMATH_CALUDE_makenna_larger_garden_l2603_260340

-- Define the dimensions of Karl's garden
def karl_length : ℕ := 30
def karl_width : ℕ := 50

-- Define the dimensions of Makenna's garden
def makenna_length : ℕ := 35
def makenna_width : ℕ := 45

-- Define the area Karl allocates for trees
def karl_tree_area : ℕ := 300

-- Calculate the areas of both gardens
def karl_total_area : ℕ := karl_length * karl_width
def makenna_total_area : ℕ := makenna_length * makenna_width

-- Calculate Karl's vegetable area
def karl_veg_area : ℕ := karl_total_area - karl_tree_area

-- Define the difference between vegetable areas
def veg_area_difference : ℕ := makenna_total_area - karl_veg_area

-- Theorem statement
theorem makenna_larger_garden : veg_area_difference = 375 := by
  sorry

end NUMINAMATH_CALUDE_makenna_larger_garden_l2603_260340


namespace NUMINAMATH_CALUDE_sandy_spending_percentage_l2603_260305

def total_amount : ℝ := 320
def amount_left : ℝ := 224

theorem sandy_spending_percentage :
  (total_amount - amount_left) / total_amount * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sandy_spending_percentage_l2603_260305


namespace NUMINAMATH_CALUDE_b_visited_city_b_l2603_260319

-- Define the types for students and cities
inductive Student : Type
| A : Student
| B : Student
| C : Student

inductive City : Type
| A : City
| B : City
| C : City

-- Define a function to represent whether a student has visited a city
def hasVisited : Student → City → Prop := sorry

-- State the theorem
theorem b_visited_city_b 
  (h1 : ∀ c : City, hasVisited Student.A c → hasVisited Student.B c)
  (h2 : ¬ hasVisited Student.A City.C)
  (h3 : ¬ hasVisited Student.B City.A)
  (h4 : ∃ c : City, hasVisited Student.A c ∧ hasVisited Student.B c ∧ hasVisited Student.C c)
  : hasVisited Student.B City.B :=
sorry

end NUMINAMATH_CALUDE_b_visited_city_b_l2603_260319


namespace NUMINAMATH_CALUDE_computer_store_optimal_solution_l2603_260382

/-- Represents the profit optimization problem for a computer store. -/
def ComputerStoreProblem (total_computers : ℕ) (profit_A profit_B : ℕ) : Prop :=
  ∃ (x : ℕ) (y : ℤ),
    -- Total number of computers is fixed
    x + (total_computers - x) = total_computers ∧
    -- Profit calculation
    y = -100 * x + 50000 ∧
    -- Constraint on type B computers
    (total_computers - x) ≤ 3 * x ∧
    -- x is the optimal number of type A computers
    ∀ (x' : ℕ), x' ≠ x →
      (-100 * x' + 50000 : ℤ) ≤ (-100 * x + 50000 : ℤ) ∧
    -- Maximum profit is achieved
    y = 47500

/-- Theorem stating the existence of an optimal solution for the computer store problem. -/
theorem computer_store_optimal_solution :
  ComputerStoreProblem 100 400 500 :=
sorry

end NUMINAMATH_CALUDE_computer_store_optimal_solution_l2603_260382


namespace NUMINAMATH_CALUDE_scientific_notation_of_ten_million_two_hundred_thousand_l2603_260342

theorem scientific_notation_of_ten_million_two_hundred_thousand :
  ∃ (a : ℝ) (n : ℤ), 10200000 = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 10.2 ∧ n = 7 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_ten_million_two_hundred_thousand_l2603_260342


namespace NUMINAMATH_CALUDE_cube_root_125_times_fourth_root_256_times_fifth_root_32_l2603_260391

theorem cube_root_125_times_fourth_root_256_times_fifth_root_32 :
  (125 : ℝ) ^ (1/3) * (256 : ℝ) ^ (1/4) * (32 : ℝ) ^ (1/5) = 40 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_125_times_fourth_root_256_times_fifth_root_32_l2603_260391


namespace NUMINAMATH_CALUDE_interior_lattice_points_collinear_l2603_260398

/-- A lattice point in the plane -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A triangle in the plane -/
structure Triangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Predicate to check if a point is inside a triangle -/
def isInside (p : LatticePoint) (t : Triangle) : Prop :=
  sorry

/-- Predicate to check if a point is on the boundary of a triangle -/
def isOnBoundary (p : LatticePoint) (t : Triangle) : Prop :=
  sorry

/-- Predicate to check if points are collinear -/
def areCollinear (points : List LatticePoint) : Prop :=
  sorry

/-- The main theorem -/
theorem interior_lattice_points_collinear (t : Triangle) :
  (∀ p : LatticePoint, isOnBoundary p t → (p = t.A ∨ p = t.B ∨ p = t.C)) →
  (∃ (p1 p2 p3 p4 : LatticePoint),
    isInside p1 t ∧ isInside p2 t ∧ isInside p3 t ∧ isInside p4 t ∧
    (∀ q : LatticePoint, isInside q t → (q = p1 ∨ q = p2 ∨ q = p3 ∨ q = p4))) →
  ∃ (p1 p2 p3 p4 : LatticePoint),
    isInside p1 t ∧ isInside p2 t ∧ isInside p3 t ∧ isInside p4 t ∧
    areCollinear [p1, p2, p3, p4] :=
by
  sorry


end NUMINAMATH_CALUDE_interior_lattice_points_collinear_l2603_260398


namespace NUMINAMATH_CALUDE_division_problem_l2603_260324

theorem division_problem : (72 : ℝ) / (6 / (3 / 2)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2603_260324


namespace NUMINAMATH_CALUDE_polynomial_divisibility_theorem_l2603_260379

def is_prime (p : ℕ) : Prop := Nat.Prime p

def divides (p n : ℕ) : Prop := n % p = 0

def polynomial_with_int_coeffs (P : ℕ → ℤ) : Prop :=
  ∃ (coeffs : List ℤ), ∀ x, P x = (coeffs.enum.map (λ (i, a) => a * (x ^ i))).sum

def constant_polynomial (P : ℕ → ℤ) : Prop :=
  ∃ c : ℤ, c ≠ 0 ∧ ∀ x, P x = c

def S (P : ℕ → ℤ) : Set ℕ :=
  {p | is_prime p ∧ ∃ n, divides p (P n).natAbs}

theorem polynomial_divisibility_theorem (P : ℕ → ℤ) 
  (h_poly : polynomial_with_int_coeffs P) :
  (Set.Finite (S P)) ↔ (constant_polynomial P) :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_theorem_l2603_260379


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2603_260388

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₃ + a₈ = 6, prove that 3a₂ + a₁₆ = 12 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arithmetic : arithmetic_sequence a) 
    (h_sum : a 3 + a 8 = 6) : 
  3 * a 2 + a 16 = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2603_260388


namespace NUMINAMATH_CALUDE_no_solution_for_2023_l2603_260322

theorem no_solution_for_2023 : ¬ ∃ (a b : ℤ), a^2 + b^2 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_2023_l2603_260322


namespace NUMINAMATH_CALUDE_pentagon_rods_l2603_260301

theorem pentagon_rods (rods : Finset ℕ) : 
  rods = {4, 9, 18, 25} →
  (∀ e ∈ Finset.range 41 \ rods, 
    (e ≠ 0 ∧ 
     e < 4 + 9 + 18 + 25 ∧
     4 + 9 + 18 + e > 25 ∧
     4 + 9 + 25 + e > 18 ∧
     4 + 18 + 25 + e > 9 ∧
     9 + 18 + 25 + e > 4)) →
  (Finset.range 41 \ rods).card = 51 :=
sorry

end NUMINAMATH_CALUDE_pentagon_rods_l2603_260301


namespace NUMINAMATH_CALUDE_right_triangle_arctan_sum_l2603_260375

/-- Given a right-angled triangle ABC with ∠A = π/2, D is the foot of the altitude from A to BC,
    BD = m, and DC = n. This theorem proves that arctan(b/(m+c)) + arctan(c/(n+b)) = π/4. -/
theorem right_triangle_arctan_sum (a b c m n : ℝ) (h_right : a^2 = b^2 + c^2)
  (h_altitude : m * n = a^2) (h_sum : m * b + c * n = b * n + c * m) :
  Real.arctan (b / (m + c)) + Real.arctan (c / (n + b)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_arctan_sum_l2603_260375


namespace NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l2603_260300

theorem muffin_banana_cost_ratio :
  ∀ (m b : ℝ),
  m > 0 → b > 0 →
  4 * m + 3 * b > 0 →
  2 * (4 * m + 3 * b) = 2 * m + 16 * b →
  m / b = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l2603_260300


namespace NUMINAMATH_CALUDE_characterize_nonnegative_quadratic_function_l2603_260397

/-- A function f: ℝ → ℝ satisfying the given conditions -/
def NonNegativeQuadraticFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x ≥ 0) ∧ 
  (∀ x y, f (x + y) + f (x - y) - 2 * f x - 2 * y^2 = 0)

/-- The theorem stating the form of f -/
theorem characterize_nonnegative_quadratic_function 
  (f : ℝ → ℝ) (h : NonNegativeQuadraticFunction f) : 
  ∃ a c : ℝ, (∀ x, f x = x^2 + a*x + c) ∧ a^2 - 4*c ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_characterize_nonnegative_quadratic_function_l2603_260397


namespace NUMINAMATH_CALUDE_connect_four_ratio_l2603_260311

theorem connect_four_ratio (total_games won_games : ℕ) 
  (h1 : total_games = 30) 
  (h2 : won_games = 18) : 
  (won_games : ℚ) / (total_games - won_games) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_connect_four_ratio_l2603_260311


namespace NUMINAMATH_CALUDE_nora_watch_cost_l2603_260376

/-- The cost of a watch in dollars, given the number of dimes paid and the value of a dime in dollars. -/
def watch_cost (dimes_paid : ℕ) (dime_value : ℚ) : ℚ :=
  (dimes_paid : ℚ) * dime_value

/-- Theorem stating that if Nora paid 90 dimes for a watch, and 1 dime is worth $0.10, the cost of the watch is $9.00. -/
theorem nora_watch_cost :
  let dimes_paid : ℕ := 90
  let dime_value : ℚ := 1/10
  watch_cost dimes_paid dime_value = 9 := by
sorry

end NUMINAMATH_CALUDE_nora_watch_cost_l2603_260376


namespace NUMINAMATH_CALUDE_prob_two_black_is_25_102_l2603_260372

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (total_cards : Nat)
  (black_cards : Nat)
  (h_total : total_cards = 52)
  (h_black : black_cards = 26)

/-- The probability of drawing two black cards from a standard deck -/
def prob_two_black (d : Deck) : Rat :=
  (d.black_cards * (d.black_cards - 1)) / (d.total_cards * (d.total_cards - 1))

/-- Theorem stating the probability of drawing two black cards is 25/102 -/
theorem prob_two_black_is_25_102 (d : Deck) : prob_two_black d = 25 / 102 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_black_is_25_102_l2603_260372


namespace NUMINAMATH_CALUDE_sequence_periodicity_l2603_260355

def M (m : ℕ) : Set ℕ :=
  {x | x ∈ Finset.range m ∨ (x > m ∧ x ≤ 2*m ∧ x % 2 = 1)}

def next_term (m : ℕ) (a : ℕ) : ℕ :=
  if a % 2 = 0 then a / 2 else a + m

def is_periodic (m : ℕ) (a : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, 
    (Nat.iterate (next_term m) k a) = (Nat.iterate (next_term m) n a)

theorem sequence_periodicity (m : ℕ) (h : m > 0) :
  ∀ a : ℕ, is_periodic m a ↔ a ∈ M m :=
sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l2603_260355


namespace NUMINAMATH_CALUDE_hundred_thirteen_in_sequence_l2603_260304

/-- Ewan's sequence starting at 3 and increasing by 11 each time -/
def ewans_sequence (n : ℕ) : ℤ := 11 * n - 8

/-- Theorem stating that 113 is in Ewan's sequence -/
theorem hundred_thirteen_in_sequence : ∃ n : ℕ, ewans_sequence n = 113 := by
  sorry

end NUMINAMATH_CALUDE_hundred_thirteen_in_sequence_l2603_260304


namespace NUMINAMATH_CALUDE_x_plus_y_value_l2603_260346

theorem x_plus_y_value (x y : ℝ) 
  (h1 : |x| + x + y = 10) 
  (h2 : x + |y| - y = 12) : 
  x + y = 3.6 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l2603_260346


namespace NUMINAMATH_CALUDE_function_divisibility_property_l2603_260378

theorem function_divisibility_property (f : ℕ+ → ℕ+) : 
  (∀ a b : ℕ+, ∃ k : ℕ+, a^2 + f a * f b = k * (f a + b)) →
  (∀ n : ℕ+, f n = n) :=
by sorry

end NUMINAMATH_CALUDE_function_divisibility_property_l2603_260378


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l2603_260302

theorem smallest_sum_of_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 20) :
  ∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 20 → (x : ℕ) + (y : ℕ) ≤ (a : ℕ) + (b : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l2603_260302


namespace NUMINAMATH_CALUDE_root_minus_one_l2603_260363

theorem root_minus_one (p : ℝ) (hp : p ≠ 1 ∧ p ≠ -1) : 
  (2 * (1 - p + p^2) / (1 - p^2)) * (-1)^2 + 
  ((2 - p) / (1 + p)) * (-1) - 
  (p / (1 - p)) = 0 := by
sorry

end NUMINAMATH_CALUDE_root_minus_one_l2603_260363


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l2603_260318

theorem nested_fraction_equality : 
  1 + 2 / (3 + 6 / (7 + 8 / 9)) = 409 / 267 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l2603_260318


namespace NUMINAMATH_CALUDE_f_properties_l2603_260389

def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

theorem f_properties :
  (∃ (x : ℝ), -2 < x ∧ x < 2 ∧ f x = 5 ∧ ∀ (y : ℝ), -2 < y ∧ y < 2 → f y ≤ 5) ∧
  (∀ (m : ℝ), ∃ (x : ℝ), -2 < x ∧ x < 2 ∧ f x < m) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2603_260389


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l2603_260370

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 16*y = x*y) :
  ∀ z w : ℝ, z > 0 → w > 0 → z + 16*w = z*w → x + y ≤ z + w ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 16*b = a*b ∧ a + b = 25 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l2603_260370


namespace NUMINAMATH_CALUDE_first_term_of_specific_sequence_l2603_260354

/-- A geometric sequence is defined by its fifth and sixth terms -/
structure GeometricSequence where
  fifth_term : ℚ
  sixth_term : ℚ

/-- The first term of a geometric sequence -/
def first_term (seq : GeometricSequence) : ℚ :=
  256 / 27

/-- Theorem: Given a geometric sequence where the fifth term is 48 and the sixth term is 72, 
    the first term is 256/27 -/
theorem first_term_of_specific_sequence :
  ∀ (seq : GeometricSequence), 
    seq.fifth_term = 48 ∧ seq.sixth_term = 72 → first_term seq = 256 / 27 :=
by
  sorry

end NUMINAMATH_CALUDE_first_term_of_specific_sequence_l2603_260354


namespace NUMINAMATH_CALUDE_cheapest_lamp_cost_l2603_260356

theorem cheapest_lamp_cost (frank_money : ℕ) (remaining : ℕ) (price_ratio : ℕ) : 
  frank_money = 90 →
  remaining = 30 →
  price_ratio = 3 →
  (frank_money - remaining) / price_ratio = 20 :=
by sorry

end NUMINAMATH_CALUDE_cheapest_lamp_cost_l2603_260356


namespace NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l2603_260308

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex 9-sided polygon has 27 diagonals -/
theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l2603_260308


namespace NUMINAMATH_CALUDE_inequality_chain_l2603_260328

theorem inequality_chain (a b : ℝ) (h : a < b ∧ b < 0) : a^2 > a*b ∧ a*b > b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l2603_260328


namespace NUMINAMATH_CALUDE_hyperbola_a_value_l2603_260362

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the focal length
def focal_length : ℝ := 12

-- Define the condition for point M
def point_M_condition (a b c : ℝ) : Prop :=
  b^2 / a = 2 * (a + c)

-- Theorem statement
theorem hyperbola_a_value (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧
  c = focal_length / 2 ∧
  c^2 = a^2 + b^2 ∧
  point_M_condition a b c →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_a_value_l2603_260362


namespace NUMINAMATH_CALUDE_apple_students_l2603_260368

theorem apple_students (bananas apples both one_fruit : ℕ) 
  (h1 : bananas = 8)
  (h2 : one_fruit = 10)
  (h3 : both = 5)
  (h4 : one_fruit = (apples - both) + (bananas - both)) :
  apples = 12 := by
  sorry

end NUMINAMATH_CALUDE_apple_students_l2603_260368


namespace NUMINAMATH_CALUDE_sourball_candies_distribution_l2603_260326

def nellie_limit : ℕ := 12
def jacob_limit : ℕ := nellie_limit / 2
def lana_limit : ℕ := jacob_limit - 3
def total_candies : ℕ := 30
def num_people : ℕ := 3

theorem sourball_candies_distribution :
  (total_candies - (nellie_limit + jacob_limit + lana_limit)) / num_people = 3 := by
  sorry

end NUMINAMATH_CALUDE_sourball_candies_distribution_l2603_260326


namespace NUMINAMATH_CALUDE_namek_clock_overlap_time_l2603_260380

/-- Represents the clock on Namek --/
structure NamekClock where
  minutes_per_hour : ℕ
  hour_hand_rate : ℚ
  minute_hand_rate : ℚ

/-- The time when the hour and minute hands overlap on Namek's clock --/
def overlap_time (clock : NamekClock) : ℚ :=
  360 / (clock.minute_hand_rate - clock.hour_hand_rate)

/-- Theorem stating that the overlap time for Namek's clock is 20/19 hours --/
theorem namek_clock_overlap_time :
  let clock : NamekClock := {
    minutes_per_hour := 100,
    hour_hand_rate := 360 / 20,
    minute_hand_rate := 360 / (100 / 60)
  }
  overlap_time clock = 20 / 19 := by sorry

end NUMINAMATH_CALUDE_namek_clock_overlap_time_l2603_260380


namespace NUMINAMATH_CALUDE_sum_of_exponents_l2603_260399

theorem sum_of_exponents (a b : ℕ) : 
  2^4 + 2^4 = 2^a → 3^5 + 3^5 + 3^5 = 3^b → a + b = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_exponents_l2603_260399


namespace NUMINAMATH_CALUDE_cookie_cost_is_16_l2603_260332

/-- The cost of each cookie Josiah bought in March --/
def cookie_cost (total_spent : ℕ) (days_in_march : ℕ) (cookies_per_day : ℕ) : ℚ :=
  total_spent / (days_in_march * cookies_per_day)

/-- Theorem stating that each cookie costs 16 dollars --/
theorem cookie_cost_is_16 :
  cookie_cost 992 31 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_cookie_cost_is_16_l2603_260332


namespace NUMINAMATH_CALUDE_andrew_work_days_l2603_260312

/-- Given that Andrew worked 2.5 hours each day and 7.5 hours in total on his Science report,
    prove that he spent 3 days working on it. -/
theorem andrew_work_days (hours_per_day : ℝ) (total_hours : ℝ) (h1 : hours_per_day = 2.5) (h2 : total_hours = 7.5) :
  total_hours / hours_per_day = 3 := by
  sorry

end NUMINAMATH_CALUDE_andrew_work_days_l2603_260312


namespace NUMINAMATH_CALUDE_visible_painted_cubes_12_l2603_260364

/-- The number of visible painted unit cubes from a corner of a cube -/
def visiblePaintedCubes (n : ℕ) : ℕ :=
  3 * n^2 - 3 * (n - 1) + 1

/-- Theorem: The number of visible painted unit cubes from a corner of a 12×12×12 cube is 400 -/
theorem visible_painted_cubes_12 :
  visiblePaintedCubes 12 = 400 := by
  sorry

#eval visiblePaintedCubes 12  -- This will evaluate to 400

end NUMINAMATH_CALUDE_visible_painted_cubes_12_l2603_260364


namespace NUMINAMATH_CALUDE_smallest_gcd_multiple_l2603_260384

theorem smallest_gcd_multiple (a b : ℕ+) (h : Nat.gcd a b = 10) :
  (∃ (a' b' : ℕ+), Nat.gcd a' b' = 10 ∧ Nat.gcd (12 * a') (20 * b') = 40) ∧
  (∀ (a'' b'' : ℕ+), Nat.gcd a'' b'' = 10 → Nat.gcd (12 * a'') (20 * b'') ≥ 40) :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_multiple_l2603_260384


namespace NUMINAMATH_CALUDE_moles_of_Na2SO4_formed_l2603_260381

-- Define the reactants and products
structure Compound where
  name : String
  coefficient : ℚ

-- Define the reaction
def reaction : List Compound → List Compound → Prop :=
  λ reactants products => reactants.length = 2 ∧ products.length = 2

-- Define the balanced equation
def balancedEquation : Prop :=
  reaction
    [⟨"H2SO4", 1⟩, ⟨"NaOH", 2⟩]
    [⟨"Na2SO4", 1⟩, ⟨"H2O", 2⟩]

-- Define the given amounts of reactants
def givenReactants : List Compound :=
  [⟨"H2SO4", 1⟩, ⟨"NaOH", 2⟩]

-- Theorem to prove
theorem moles_of_Na2SO4_formed
  (h1 : balancedEquation)
  (h2 : givenReactants = [⟨"H2SO4", 1⟩, ⟨"NaOH", 2⟩]) :
  ∃ (product : Compound),
    product.name = "Na2SO4" ∧ product.coefficient = 1 :=
  sorry

end NUMINAMATH_CALUDE_moles_of_Na2SO4_formed_l2603_260381


namespace NUMINAMATH_CALUDE_point_always_on_line_l2603_260345

theorem point_always_on_line (m b : ℝ) (h : m * b < 0) :
  0 = m * 2003 + b := by sorry

end NUMINAMATH_CALUDE_point_always_on_line_l2603_260345
