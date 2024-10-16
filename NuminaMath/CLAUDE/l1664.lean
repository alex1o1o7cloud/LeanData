import Mathlib

namespace NUMINAMATH_CALUDE_set_equals_interval_l1664_166406

-- Define the set {x|x≥2}
def S : Set ℝ := {x : ℝ | x ≥ 2}

-- Define the interval [2,+∞)
def I : Set ℝ := Set.Ici 2

-- Theorem stating the equivalence of S and I
theorem set_equals_interval : S = I := by sorry

end NUMINAMATH_CALUDE_set_equals_interval_l1664_166406


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_sixty_degrees_l1664_166478

/-- The degree measure of the supplement of the complement of a 60-degree angle is 150°. -/
theorem supplement_of_complement_of_sixty_degrees : 
  let original_angle : ℝ := 60
  let complement := 90 - original_angle
  let supplement := 180 - complement
  supplement = 150 := by sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_sixty_degrees_l1664_166478


namespace NUMINAMATH_CALUDE_equation_represents_parabola_l1664_166427

/-- The equation represents a parabola if it can be transformed into the form y = ax² + bx + c, where a ≠ 0 -/
def is_parabola (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation |y - 3| = √((x+1)² + y²) -/
def given_equation (x y : ℝ) : Prop :=
  |y - 3| = Real.sqrt ((x + 1)^2 + y^2)

theorem equation_represents_parabola :
  ∃ f : ℝ → ℝ, (∀ x y, given_equation x y ↔ y = f x) ∧ is_parabola f :=
sorry

end NUMINAMATH_CALUDE_equation_represents_parabola_l1664_166427


namespace NUMINAMATH_CALUDE_uncles_age_l1664_166471

theorem uncles_age (bud_age uncle_age : ℕ) : 
  bud_age = 8 → 
  3 * bud_age = uncle_age → 
  uncle_age = 24 := by
sorry

end NUMINAMATH_CALUDE_uncles_age_l1664_166471


namespace NUMINAMATH_CALUDE_necklace_length_theorem_l1664_166428

/-- The total length of a necklace made of overlapping paper pieces -/
def necklaceLength (n : ℕ) (pieceLength : ℝ) (overlap : ℝ) : ℝ :=
  n * (pieceLength - overlap)

/-- Theorem: The total length of a necklace made of 16 pieces of colored paper,
    each 10.4 cm long and overlapping by 3.5 cm, is equal to 110.4 cm -/
theorem necklace_length_theorem :
  necklaceLength 16 10.4 3.5 = 110.4 := by
  sorry

end NUMINAMATH_CALUDE_necklace_length_theorem_l1664_166428


namespace NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l1664_166415

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define what it means for a function to be even
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- Define the axis of symmetry for a function
def axis_of_symmetry (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, g (a + x) = g (a - x)

-- State the theorem
theorem symmetry_of_shifted_even_function :
  is_even (λ x => f (x + 1)) → axis_of_symmetry f 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l1664_166415


namespace NUMINAMATH_CALUDE_complex_power_2013_l1664_166449

theorem complex_power_2013 : (((1 : ℂ) + Complex.I) / ((1 : ℂ) - Complex.I)) ^ 2013 = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_power_2013_l1664_166449


namespace NUMINAMATH_CALUDE_randy_house_blocks_l1664_166424

/-- The number of blocks Randy used to build the house -/
def blocks_for_house : ℕ := 20

/-- The total number of blocks Randy has -/
def total_blocks : ℕ := 95

/-- The number of blocks Randy used to build the tower -/
def blocks_for_tower : ℕ := 50

theorem randy_house_blocks :
  blocks_for_house = 20 ∧
  total_blocks = 95 ∧
  blocks_for_tower = 50 ∧
  blocks_for_tower = blocks_for_house + 30 :=
sorry

end NUMINAMATH_CALUDE_randy_house_blocks_l1664_166424


namespace NUMINAMATH_CALUDE_second_number_proof_l1664_166459

theorem second_number_proof (x y z : ℝ) 
  (sum_eq : x + y + z = 120)
  (ratio_xy : x / y = 3 / 4)
  (ratio_yz : y / z = 4 / 7)
  (x_pos : x > 0) (y_pos : y > 0) (z_pos : z > 0) : 
  y = 240 / 7 := by
  sorry

end NUMINAMATH_CALUDE_second_number_proof_l1664_166459


namespace NUMINAMATH_CALUDE_sequence_length_l1664_166469

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

-- Theorem statement
theorem sequence_length :
  let a₁ : ℝ := 2.5
  let d : ℝ := 5
  let aₙ : ℝ := 62.5
  ∃ n : ℕ, n > 0 ∧ arithmetic_sequence a₁ d n = aₙ ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_sequence_length_l1664_166469


namespace NUMINAMATH_CALUDE_problem_statement_l1664_166446

theorem problem_statement (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : a^b = c^d) (h2 : a / (2 * c) = b / d) (h3 : a / (2 * c) = 2) :
  1 / c = 16 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1664_166446


namespace NUMINAMATH_CALUDE_marble_jar_problem_l1664_166443

theorem marble_jar_problem (num_marbles : ℕ) : 
  (∀ (x : ℚ), x = num_marbles / 20 → 
    x - 1 = num_marbles / 22) → 
  num_marbles = 220 := by
  sorry

end NUMINAMATH_CALUDE_marble_jar_problem_l1664_166443


namespace NUMINAMATH_CALUDE_store_A_cheaper_store_A_cheaper_30_l1664_166438

/-- The cost of buying pens at Store A -/
def costA (x : ℝ) : ℝ := 0.9 * x + 6

/-- The cost of buying pens at Store B -/
def costB (x : ℝ) : ℝ := 1.2 * x

/-- Theorem: Store A is cheaper than Store B for 20 or more pens -/
theorem store_A_cheaper (x : ℝ) (h : x ≥ 20) : costA x ≤ costB x := by
  sorry

/-- Corollary: Store A is cheaper for exactly 30 pens -/
theorem store_A_cheaper_30 : costA 30 < costB 30 := by
  sorry

end NUMINAMATH_CALUDE_store_A_cheaper_store_A_cheaper_30_l1664_166438


namespace NUMINAMATH_CALUDE_taco_cost_is_90_cents_l1664_166495

-- Define the cost of a taco and an enchilada
variable (taco_cost enchilada_cost : ℚ)

-- Define the two orders
def order1_cost := 2 * taco_cost + 3 * enchilada_cost
def order2_cost := 3 * taco_cost + 5 * enchilada_cost

-- State the theorem
theorem taco_cost_is_90_cents 
  (h1 : order1_cost = 780/100)
  (h2 : order2_cost = 1270/100) :
  taco_cost = 90/100 := by
  sorry

end NUMINAMATH_CALUDE_taco_cost_is_90_cents_l1664_166495


namespace NUMINAMATH_CALUDE_digit_at_573_l1664_166418

/-- The decimal representation of 11/37 -/
def decimal_rep : ℚ := 11 / 37

/-- The length of the repeating sequence in the decimal representation of 11/37 -/
def period : ℕ := 3

/-- The repeating sequence in the decimal representation of 11/37 -/
def repeating_sequence : Fin 3 → ℕ
| 0 => 2
| 1 => 9
| 2 => 7

/-- The position we're interested in -/
def target_position : ℕ := 573

theorem digit_at_573 : 
  (repeating_sequence (target_position % period : Fin 3)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_digit_at_573_l1664_166418


namespace NUMINAMATH_CALUDE_division_problem_l1664_166404

theorem division_problem (x y : ℕ+) 
  (h1 : (x : ℝ) / (y : ℝ) = 96.12)
  (h2 : (x : ℝ) % (y : ℝ) = 1.44) : 
  y = 12 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1664_166404


namespace NUMINAMATH_CALUDE_max_positive_integers_l1664_166498

/-- A circular arrangement of 100 nonzero integers -/
def CircularArrangement := Fin 100 → ℤ

/-- Predicate to check if an arrangement satisfies the given condition -/
def SatisfiesCondition (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 100, arr i ≠ 0 ∧ arr i > arr ((i + 1) % 100) * arr ((i + 2) % 100)

/-- Count of positive integers in an arrangement -/
def PositiveCount (arr : CircularArrangement) : ℕ :=
  (Finset.univ.filter (fun i => arr i > 0)).card

/-- Theorem stating the maximum number of positive integers possible -/
theorem max_positive_integers (arr : CircularArrangement) 
  (h : SatisfiesCondition arr) : PositiveCount arr ≤ 50 := by
  sorry

end NUMINAMATH_CALUDE_max_positive_integers_l1664_166498


namespace NUMINAMATH_CALUDE_gcd_problem_l1664_166467

theorem gcd_problem (n : ℕ) : 
  30 ≤ n ∧ n ≤ 40 ∧ Nat.gcd 15 n = 5 → n = 35 ∨ n = 40 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1664_166467


namespace NUMINAMATH_CALUDE_tire_usage_calculation_l1664_166499

/-- Calculates the miles each tire is used given the total number of tires, 
    simultaneously used tires, and total miles driven. -/
def miles_per_tire (total_tires : ℕ) (used_tires : ℕ) (total_miles : ℕ) : ℚ :=
  (total_miles * used_tires : ℚ) / total_tires

theorem tire_usage_calculation :
  let total_tires : ℕ := 6
  let used_tires : ℕ := 5
  let total_miles : ℕ := 42000
  miles_per_tire total_tires used_tires total_miles = 35000 := by
  sorry

end NUMINAMATH_CALUDE_tire_usage_calculation_l1664_166499


namespace NUMINAMATH_CALUDE_binary_10101_equals_21_l1664_166492

def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_10101_equals_21 :
  binary_to_decimal [true, false, true, false, true] = 21 := by
  sorry

end NUMINAMATH_CALUDE_binary_10101_equals_21_l1664_166492


namespace NUMINAMATH_CALUDE_chess_board_numbering_specific_cell_number_l1664_166405

/-- Represents the numbering system of an infinite chessboard where each cell
    is assigned the smallest possible number not yet used for numbering any
    preceding cells in the same row or column. -/
noncomputable def chessBoardNumber (row : Nat) (col : Nat) : Nat :=
  sorry

/-- The number assigned to a cell is equal to the XOR of (row - 1) and (col - 1) -/
theorem chess_board_numbering (row col : Nat) :
  chessBoardNumber row col = Nat.xor (row - 1) (col - 1) :=
sorry

/-- The cell at the intersection of the 100th row and the 1000th column
    receives the number 921 -/
theorem specific_cell_number :
  chessBoardNumber 100 1000 = 921 :=
sorry

end NUMINAMATH_CALUDE_chess_board_numbering_specific_cell_number_l1664_166405


namespace NUMINAMATH_CALUDE_parabola_midpoint_trajectory_and_line_l1664_166464

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 16*x

-- Define the trajectory E
def trajectory_E (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y - 1 = 0

-- Define point P
def point_P : ℝ × ℝ := (3, 2)

theorem parabola_midpoint_trajectory_and_line :
  -- Part 1: Prove that the trajectory E is y² = 4x
  (∀ x y : ℝ, (∃ x₀ y₀ : ℝ, parabola x₀ y₀ ∧ x = x₀ ∧ y = y₀/2) → trajectory_E x y) ∧
  -- Part 2: Prove that the line l passing through P and intersecting E at A and B (where P is the midpoint of AB) has the equation x - y - 1 = 0
  (∀ A B : ℝ × ℝ,
    let (x₁, y₁) := A
    let (x₂, y₂) := B
    trajectory_E x₁ y₁ ∧ 
    trajectory_E x₂ y₂ ∧ 
    x₁ + x₂ = 2 * point_P.1 ∧
    y₁ + y₂ = 2 * point_P.2 →
    line_l x₁ y₁ ∧ line_l x₂ y₂) :=
by sorry

end NUMINAMATH_CALUDE_parabola_midpoint_trajectory_and_line_l1664_166464


namespace NUMINAMATH_CALUDE_sum_of_cubes_remainder_l1664_166420

def sum_of_cubes (n : ℕ) : ℕ := (n * (n + 1) / 2) ^ 2

def b : ℕ := 2

theorem sum_of_cubes_remainder (n : ℕ) (h : n = 2010) : 
  sum_of_cubes n % (b ^ 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_remainder_l1664_166420


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1664_166489

/-- Given two vectors a and b in ℝ², where a = (1,2) and b = (2x,-3),
    if a is parallel to b, then x = -3/4 -/
theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (2*x, -3)
  (∃ (k : ℝ), k ≠ 0 ∧ a.1 * k = b.1 ∧ a.2 * k = b.2) →
  x = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1664_166489


namespace NUMINAMATH_CALUDE_exam_time_on_type_A_l1664_166401

/-- Represents the time spent on type A problems in an exam -/
def time_on_type_A (total_time : ℚ) (total_questions : ℕ) (type_A_questions : ℕ) : ℚ :=
  let type_B_questions := total_questions - type_A_questions
  let time_ratio := (2 * type_A_questions + type_B_questions) / total_questions
  (total_time * time_ratio * 2 * type_A_questions) / (2 * type_A_questions + type_B_questions)

/-- Theorem stating the time spent on type A problems in the given exam conditions -/
theorem exam_time_on_type_A :
  time_on_type_A (5/2) 200 10 = 100/7 :=
by sorry

end NUMINAMATH_CALUDE_exam_time_on_type_A_l1664_166401


namespace NUMINAMATH_CALUDE_cube_root_rationality_l1664_166461

theorem cube_root_rationality (a b : ℚ) (ha : 0 < a) (hb : 0 < b) 
  (h : ∃ (s : ℚ), s = (a^(1/3) + b^(1/3))) : 
  ∃ (r₁ r₂ : ℚ), r₁ = a^(1/3) ∧ r₂ = b^(1/3) := by
sorry

end NUMINAMATH_CALUDE_cube_root_rationality_l1664_166461


namespace NUMINAMATH_CALUDE_fraction_simplification_l1664_166433

theorem fraction_simplification :
  (5 : ℝ) / (Real.sqrt 48 + 3 * Real.sqrt 75 + 5 * Real.sqrt 27) = (5 * Real.sqrt 3) / 102 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1664_166433


namespace NUMINAMATH_CALUDE_f_order_l1664_166486

def f (x : ℝ) : ℝ := sorry

axiom f_even : ∀ x, f x = f (-x)
axiom f_periodic : ∀ x, f (x + 2) = f x
axiom f_def : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x^(1/1998)

theorem f_order : f (101/17) < f (98/19) ∧ f (98/19) < f (104/15) := by sorry

end NUMINAMATH_CALUDE_f_order_l1664_166486


namespace NUMINAMATH_CALUDE_average_age_of_nine_students_l1664_166455

theorem average_age_of_nine_students (total_students : ℕ) (total_average : ℝ) 
  (five_students : ℕ) (five_average : ℝ) (seventeenth_age : ℝ) 
  (nine_students : ℕ) (h1 : total_students = 17) 
  (h2 : total_average = 17) (h3 : five_students = 5) 
  (h4 : five_average = 14) (h5 : seventeenth_age = 75) 
  (h6 : nine_students = total_students - five_students - 1) :
  (total_students * total_average - five_students * five_average - seventeenth_age) / nine_students = 16 := by
  sorry

#check average_age_of_nine_students

end NUMINAMATH_CALUDE_average_age_of_nine_students_l1664_166455


namespace NUMINAMATH_CALUDE_three_number_problem_l1664_166409

theorem three_number_problem (a b c : ℚ) : 
  a + b + c = 220 ∧ 
  a = 2 * b ∧ 
  c = (1 / 3) * a →
  b = 60 := by sorry

end NUMINAMATH_CALUDE_three_number_problem_l1664_166409


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l1664_166441

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 9) / (Nat.factorial 4)) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l1664_166441


namespace NUMINAMATH_CALUDE_intersection_point_l1664_166448

-- Define the circle C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 5 ∧ x ≥ 0 ∧ y ≥ 0

-- Define the line C2
def C2 (x y : ℝ) : Prop := y = x - 1

-- Theorem statement
theorem intersection_point : 
  ∃! p : ℝ × ℝ, C1 p.1 p.2 ∧ C2 p.1 p.2 ∧ p = (2, 1) := by sorry

end NUMINAMATH_CALUDE_intersection_point_l1664_166448


namespace NUMINAMATH_CALUDE_statue_selling_price_l1664_166417

/-- The selling price of a statue given its original cost and profit percentage -/
def selling_price (original_cost : ℝ) (profit_percentage : ℝ) : ℝ :=
  original_cost * (1 + profit_percentage)

/-- Theorem: The selling price of the statue is $660 -/
theorem statue_selling_price :
  let original_cost : ℝ := 550
  let profit_percentage : ℝ := 0.20
  selling_price original_cost profit_percentage = 660 := by
  sorry

end NUMINAMATH_CALUDE_statue_selling_price_l1664_166417


namespace NUMINAMATH_CALUDE_max_blocks_fit_l1664_166413

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℕ := d.length * d.width * d.height

/-- Represents the box and block dimensions -/
def box : Dimensions := ⟨40, 60, 80⟩
def block : Dimensions := ⟨20, 30, 40⟩

/-- Calculates the maximum number of blocks that can fit in the box based on volume -/
def max_blocks_by_volume : ℕ := volume box / volume block

/-- Checks if the blocks can be arranged to fit in the box -/
def can_arrange (n : ℕ) : Prop :=
  ∃ (l w h : ℕ), l * block.length ≤ box.length ∧
                 w * block.width ≤ box.width ∧
                 h * block.height ≤ box.height ∧
                 l * w * h = n

/-- The main theorem to prove -/
theorem max_blocks_fit :
  max_blocks_by_volume = 8 ∧ can_arrange 8 ∧
  ∀ n > 8, ¬can_arrange n :=
sorry

end NUMINAMATH_CALUDE_max_blocks_fit_l1664_166413


namespace NUMINAMATH_CALUDE_f_greater_than_one_iff_l1664_166457

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x) - 1 else Real.sqrt x

theorem f_greater_than_one_iff (x₀ : ℝ) :
  f x₀ > 1 ↔ x₀ < -1 ∨ x₀ > 1 := by sorry

end NUMINAMATH_CALUDE_f_greater_than_one_iff_l1664_166457


namespace NUMINAMATH_CALUDE_flying_blade_diameter_l1664_166435

theorem flying_blade_diameter (d : ℝ) (n : ℤ) : 
  d = 0.0000009 → d = 9 * 10^n → n = -7 := by sorry

end NUMINAMATH_CALUDE_flying_blade_diameter_l1664_166435


namespace NUMINAMATH_CALUDE_PQRS_equals_nine_l1664_166458

theorem PQRS_equals_nine :
  let P : ℝ := Real.sqrt 2010 + Real.sqrt 2007
  let Q : ℝ := -Real.sqrt 2010 - Real.sqrt 2007
  let R : ℝ := Real.sqrt 2010 - Real.sqrt 2007
  let S : ℝ := Real.sqrt 2007 - Real.sqrt 2010
  P * Q * R * S = 9 := by
  sorry

end NUMINAMATH_CALUDE_PQRS_equals_nine_l1664_166458


namespace NUMINAMATH_CALUDE_simplify_expression_l1664_166490

theorem simplify_expression : (-5) + (-6) - (-5) + 4 = -5 - 6 + 5 + 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1664_166490


namespace NUMINAMATH_CALUDE_abigail_report_time_l1664_166442

/-- Given a report length, typing speed, and words already written, 
    calculate the time required to finish the report. -/
def time_to_finish_report (total_words : ℕ) (words_per_half_hour : ℕ) (words_written : ℕ) : ℕ :=
  let words_remaining := total_words - words_written
  let minutes_per_word := 30 / words_per_half_hour
  words_remaining * minutes_per_word

/-- Proof that for the given conditions, the time to finish the report is 80 minutes. -/
theorem abigail_report_time : time_to_finish_report 1000 300 200 = 80 := by
  sorry

end NUMINAMATH_CALUDE_abigail_report_time_l1664_166442


namespace NUMINAMATH_CALUDE_inequality_solution_l1664_166479

theorem inequality_solution (x : ℝ) : 
  (x^3 - 3*x^2 + 2*x) / (x^2 - 2*x + 1) ≥ 0 ↔ x ∈ Set.Ici 0 ∩ Set.Iio 1 ∪ Set.Ici 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1664_166479


namespace NUMINAMATH_CALUDE_fencing_required_l1664_166421

theorem fencing_required (area : ℝ) (uncovered_side : ℝ) : 
  area = 560 ∧ uncovered_side = 20 → 
  ∃ (width : ℝ), 
    area = uncovered_side * width ∧ 
    2 * width + uncovered_side = 76 := by
  sorry

end NUMINAMATH_CALUDE_fencing_required_l1664_166421


namespace NUMINAMATH_CALUDE_max_take_home_pay_l1664_166454

-- Define the income function
def income (y : ℝ) : ℝ := 100 * y^2

-- Define the tax function
def tax (y : ℝ) : ℝ := y^3

-- Define the take-home pay function
def takeHomePay (y : ℝ) : ℝ := income y - tax y

-- Theorem statement
theorem max_take_home_pay :
  ∃ y : ℝ, y > 0 ∧ 
    (∀ z : ℝ, z > 0 → takeHomePay z ≤ takeHomePay y) ∧
    income y = 250000 := by sorry

end NUMINAMATH_CALUDE_max_take_home_pay_l1664_166454


namespace NUMINAMATH_CALUDE_kath_movie_cost_l1664_166494

def movie_admission_cost (regular_price : ℚ) (discount_percent : ℚ) (before_6pm : Bool) (num_people : ℕ) : ℚ :=
  let discounted_price := if before_6pm then regular_price * (1 - discount_percent / 100) else regular_price
  discounted_price * num_people

theorem kath_movie_cost :
  let regular_price : ℚ := 8
  let discount_percent : ℚ := 25
  let before_6pm : Bool := true
  let num_people : ℕ := 6
  movie_admission_cost regular_price discount_percent before_6pm num_people = 36 := by
  sorry

end NUMINAMATH_CALUDE_kath_movie_cost_l1664_166494


namespace NUMINAMATH_CALUDE_expression_simplification_l1664_166422

theorem expression_simplification (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 2) (h3 : a ≠ -2) :
  ((a / (a - 2) - a / (a^2 - 2*a)) / (a + 2) * a) = (a^2 - a) / (a^2 - 4) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1664_166422


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1664_166434

/-- A function f satisfying the given conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The coefficients of f in its quadratic form -/
def a : ℝ := sorry
def b : ℝ := sorry
def c : ℝ := sorry

/-- The main theorem stating that a + b + c = 50 -/
theorem sum_of_coefficients :
  (∀ x, f (x + 5) = 5 * x^2 + 9 * x + 6) →
  (∀ x, f x = a * x^2 + b * x + c) →
  a + b + c = 50 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1664_166434


namespace NUMINAMATH_CALUDE_hundred_thousand_eq_scientific_notation_l1664_166475

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coefficient_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Definition of the number 100,000 -/
def hundred_thousand : ℕ := 100000

/-- The scientific notation of 100,000 -/
def hundred_thousand_scientific : ScientificNotation :=
  ⟨1, 5, by {sorry}⟩

/-- Theorem stating that 100,000 is equal to its scientific notation representation -/
theorem hundred_thousand_eq_scientific_notation :
  (hundred_thousand : ℝ) = hundred_thousand_scientific.coefficient * (10 : ℝ) ^ hundred_thousand_scientific.exponent :=
by sorry

end NUMINAMATH_CALUDE_hundred_thousand_eq_scientific_notation_l1664_166475


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1664_166400

theorem simplify_trig_expression (α β : ℝ) : 
  Real.sqrt ((1 - Real.sin α * Real.sin β)^2 - (Real.cos α * Real.cos β)^2) = 
  |Real.sin α - Real.sin β| := by
sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1664_166400


namespace NUMINAMATH_CALUDE_towels_remaining_l1664_166476

/-- The number of green towels Maria bought -/
def green_bought : ℕ := 35

/-- The number of white towels Maria bought -/
def white_bought : ℕ := 21

/-- The number of blue towels Maria bought -/
def blue_bought : ℕ := 15

/-- The number of green towels Maria gave to her mother -/
def green_given : ℕ := 22

/-- The number of white towels Maria gave to her mother -/
def white_given : ℕ := 14

/-- The number of blue towels Maria gave to her mother -/
def blue_given : ℕ := 6

/-- The total number of towels Maria gave to her mother -/
def total_given : ℕ := 42

theorem towels_remaining : 
  (green_bought + white_bought + blue_bought) - total_given = 29 := by
  sorry

end NUMINAMATH_CALUDE_towels_remaining_l1664_166476


namespace NUMINAMATH_CALUDE_solution_in_interval_monotonic_decreasing_range_two_roots_range_l1664_166412

-- Define the function f(x)
def f (x k : ℝ) : ℝ := |x^2 - 1| + x^2 + k*x

-- Theorem 1
theorem solution_in_interval (k : ℝ) :
  ∃ x : ℝ, x ∈ Set.Ioo 0 2 ∧ f x k = k*x + 3 → x = Real.sqrt 2 :=
sorry

-- Theorem 2
theorem monotonic_decreasing_range (k : ℝ) :
  (∀ x y : ℝ, x ∈ Set.Ioo 0 2 → y ∈ Set.Ioo 0 2 → x < y → f x k > f y k) →
  k ∈ Set.Iic (-8) :=
sorry

-- Theorem 3
theorem two_roots_range (k : ℝ) :
  (∃ x y : ℝ, x ∈ Set.Ioo 0 2 ∧ y ∈ Set.Ioo 0 2 ∧ x ≠ y ∧ f x k = 0 ∧ f y k = 0) →
  k ∈ Set.Ioo (-7/2) (-1) :=
sorry

end NUMINAMATH_CALUDE_solution_in_interval_monotonic_decreasing_range_two_roots_range_l1664_166412


namespace NUMINAMATH_CALUDE_tan_neg_585_deg_l1664_166472

theorem tan_neg_585_deg : Real.tan (-585 * π / 180) = -1 := by sorry

end NUMINAMATH_CALUDE_tan_neg_585_deg_l1664_166472


namespace NUMINAMATH_CALUDE_fraction_equivalence_l1664_166439

theorem fraction_equivalence : ∃ n : ℚ, (2 + n) / (7 + n) = 3 / 5 :=
by
  use 11 / 2
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l1664_166439


namespace NUMINAMATH_CALUDE_tangent_point_on_parabola_l1664_166493

theorem tangent_point_on_parabola : ∃ (x y : ℝ), 
  y = x^2 ∧ 
  (2 : ℝ) * x = Real.tan (π / 4) ∧ 
  x = 1 / 2 ∧ 
  y = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_point_on_parabola_l1664_166493


namespace NUMINAMATH_CALUDE_banquet_solution_l1664_166403

def banquet_problem (total_attendees : ℕ) (resident_price : ℚ) (total_revenue : ℚ) (num_residents : ℕ) : ℚ :=
  let non_residents := total_attendees - num_residents
  let resident_revenue := num_residents * resident_price
  let non_resident_revenue := total_revenue - resident_revenue
  non_resident_revenue / non_residents

theorem banquet_solution :
  banquet_problem 586 12.95 9423.70 219 = 17.95 := by
  sorry

end NUMINAMATH_CALUDE_banquet_solution_l1664_166403


namespace NUMINAMATH_CALUDE_area_of_triangle_abc_l1664_166496

noncomputable def triangle_area (a b c : ℝ) (A B C : ℝ) : ℝ :=
  (1/2) * b * c * Real.sin A

theorem area_of_triangle_abc (a b c : ℝ) (A B C : ℝ) 
  (h1 : A = π/4)
  (h2 : b^2 * Real.sin C = 4 * Real.sqrt 2 * Real.sin B) :
  triangle_area a b c A B C = 2 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_abc_l1664_166496


namespace NUMINAMATH_CALUDE_linear_equation_solution_l1664_166410

theorem linear_equation_solution (a : ℝ) :
  (∀ x, ax^2 + 5*x + 14 = 2*x^2 - 2*x + 3*a → x = -8/7) ∧
  (∃ m b, ∀ x, ax^2 + 5*x + 14 = 2*x^2 - 2*x + 3*a ↔ m*x + b = 0) :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l1664_166410


namespace NUMINAMATH_CALUDE_jones_elementary_population_l1664_166470

theorem jones_elementary_population :
  ∀ (total_students : ℕ) (boys_percentage : ℚ),
    (90 : ℚ) = boys_percentage * ((20 : ℚ) / 100) * total_students →
    total_students = 450 := by
  sorry

end NUMINAMATH_CALUDE_jones_elementary_population_l1664_166470


namespace NUMINAMATH_CALUDE_pet_store_combinations_l1664_166480

def num_puppies : ℕ := 20
def num_kittens : ℕ := 4
def num_hamsters : ℕ := 6
def num_rabbits : ℕ := 10
def num_people : ℕ := 4

theorem pet_store_combinations : 
  num_puppies * num_kittens * num_hamsters * num_rabbits * (Nat.factorial num_people) = 115200 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_combinations_l1664_166480


namespace NUMINAMATH_CALUDE_stock_price_theorem_l1664_166436

def stock_price_evolution (initial_price : ℝ) (year1_change : ℝ) (year2_change : ℝ) (year3_change : ℝ) : ℝ :=
  let price_after_year1 := initial_price * (1 + year1_change)
  let price_after_year2 := price_after_year1 * (1 + year2_change)
  let price_after_year3 := price_after_year2 * (1 + year3_change)
  price_after_year3

theorem stock_price_theorem :
  stock_price_evolution 150 0.5 (-0.3) 0.2 = 189 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_theorem_l1664_166436


namespace NUMINAMATH_CALUDE_num_cities_with_protests_is_21_l1664_166487

/-- The number of cities experiencing protests given the specified conditions -/
def num_cities_with_protests : ℕ :=
  let protest_days : ℕ := 30
  let arrests_per_day_per_city : ℕ := 10
  let pre_trial_days : ℕ := 4
  let post_trial_days : ℕ := 7  -- half of 2-week sentence
  let total_jail_weeks : ℕ := 9900

  -- Calculate the number of cities
  21

/-- Theorem stating that the number of cities experiencing protests is 21 -/
theorem num_cities_with_protests_is_21 :
  num_cities_with_protests = 21 := by
  sorry

end NUMINAMATH_CALUDE_num_cities_with_protests_is_21_l1664_166487


namespace NUMINAMATH_CALUDE_sum_75_odd_numbers_l1664_166453

-- Define a function for the sum of first n odd numbers
def sum_odd_numbers (n : ℕ) : ℕ := n^2

-- State the theorem
theorem sum_75_odd_numbers :
  (sum_odd_numbers 50 = 2500) → (sum_odd_numbers 75 = 5625) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_75_odd_numbers_l1664_166453


namespace NUMINAMATH_CALUDE_median_list_i_equals_eight_l1664_166414

def list_i : List ℕ := [9, 2, 4, 7, 10, 11]
def list_ii : List ℕ := [3, 3, 4, 6, 7, 10]

def median (l : List ℕ) : ℚ := sorry
def mode (l : List ℕ) : ℕ := sorry

theorem median_list_i_equals_eight :
  median list_i = 8 :=
by
  have h1 : median list_i = median list_ii + mode list_ii := sorry
  sorry

#check median_list_i_equals_eight

end NUMINAMATH_CALUDE_median_list_i_equals_eight_l1664_166414


namespace NUMINAMATH_CALUDE_investment_growth_l1664_166426

/-- Calculates the final amount after simple interest is applied --/
def final_amount (principal : ℚ) (rate : ℚ) (time : ℕ) : ℚ :=
  principal + principal * rate * time

/-- Proves that an investment of $1000 at 10% simple interest for 3 years results in $1300 --/
theorem investment_growth :
  final_amount 1000 (1/10) 3 = 1300 := by
  sorry

end NUMINAMATH_CALUDE_investment_growth_l1664_166426


namespace NUMINAMATH_CALUDE_x3y2z3_coefficient_in_x_plus_y_plus_z_to_8_l1664_166474

def multinomial_coefficient (n : ℕ) (a b c : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial a * Nat.factorial b * Nat.factorial c)

theorem x3y2z3_coefficient_in_x_plus_y_plus_z_to_8 :
  multinomial_coefficient 8 3 2 3 = 560 := by
  sorry

end NUMINAMATH_CALUDE_x3y2z3_coefficient_in_x_plus_y_plus_z_to_8_l1664_166474


namespace NUMINAMATH_CALUDE_triangle_problem_l1664_166425

-- Define the triangles and their properties
def Triangle (A B C : ℝ × ℝ) := True

def is_45_45_90_triangle (A B D : ℝ × ℝ) : Prop :=
  Triangle A B D ∧ 
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (D.1 - A.1)^2 + (D.2 - A.2)^2 ∧
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (B.1 - D.1)^2 + (B.2 - D.2)^2

def is_30_60_90_triangle (A C D : ℝ × ℝ) : Prop :=
  Triangle A C D ∧ 
  4 * ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 3 * ((D.1 - A.1)^2 + (D.2 - A.2)^2)

-- Define the theorem
theorem triangle_problem (A B C D : ℝ × ℝ) :
  is_45_45_90_triangle A B D →
  is_30_60_90_triangle A C D →
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 36 →
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 24 :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l1664_166425


namespace NUMINAMATH_CALUDE_sandy_puppies_l1664_166484

def puppies_problem (initial_puppies : ℕ) (initial_spotted : ℕ) (new_puppies : ℕ) (new_spotted : ℕ) (given_away : ℕ) : Prop :=
  let initial_non_spotted := initial_puppies - initial_spotted
  let total_spotted := initial_spotted + new_spotted
  let total_non_spotted := initial_non_spotted + (new_puppies - new_spotted) - given_away
  let final_puppies := total_spotted + total_non_spotted
  final_puppies = 9

theorem sandy_puppies : puppies_problem 8 3 4 2 3 :=
by sorry

end NUMINAMATH_CALUDE_sandy_puppies_l1664_166484


namespace NUMINAMATH_CALUDE_part_one_part_two_l1664_166437

-- Define set A
def A : Set ℝ := {x | (x + 1) / (x - 3) ≤ 0}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | x^2 - (m - 1) * x + m - 2 ≤ 0}

-- Statement for part (1)
theorem part_one (a b : ℝ) : A ∪ Set.Icc a b = Set.Icc (-1) 4 → b = 4 ∧ -1 ≤ a ∧ a < 3 := by
  sorry

-- Statement for part (2)
theorem part_two (m : ℝ) : A ∪ B m = A → 1 ≤ m ∧ m < 5 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1664_166437


namespace NUMINAMATH_CALUDE_cone_volume_l1664_166429

/-- The volume of a cone with slant height 15 cm and height 13 cm is (728/3)π cubic centimeters. -/
theorem cone_volume (π : ℝ) (slant_height height : ℝ) 
  (h1 : slant_height = 15)
  (h2 : height = 13) :
  (1/3 : ℝ) * π * (slant_height^2 - height^2) * height = (728/3) * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l1664_166429


namespace NUMINAMATH_CALUDE_equation_solution_l1664_166463

theorem equation_solution : 
  ∃ x : ℚ, (x ≠ -2) ∧ (8 * x / (x + 2) - 5 / (x + 2) = 2 / (x + 2)) ∧ (x = 7 / 8) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1664_166463


namespace NUMINAMATH_CALUDE_sin_cos_sum_bound_l1664_166452

theorem sin_cos_sum_bound (θ : Real) (h1 : π/2 < θ) (h2 : θ < π) (h3 : Real.sin (θ/2) < Real.cos (θ/2)) :
  -Real.sqrt 2 < Real.sin (θ/2) + Real.cos (θ/2) ∧ Real.sin (θ/2) + Real.cos (θ/2) < -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_bound_l1664_166452


namespace NUMINAMATH_CALUDE_meals_without_restrictions_l1664_166483

theorem meals_without_restrictions (total clients vegan kosher gluten_free vegan_kosher vegan_gluten_free kosher_gluten_free vegan_kosher_gluten_free : ℕ) 
  (h1 : total = 50)
  (h2 : vegan = 10)
  (h3 : kosher = 12)
  (h4 : gluten_free = 6)
  (h5 : vegan_kosher = 3)
  (h6 : vegan_gluten_free = 4)
  (h7 : kosher_gluten_free = 2)
  (h8 : vegan_kosher_gluten_free = 1) :
  total - (vegan + kosher + gluten_free - vegan_kosher - vegan_gluten_free - kosher_gluten_free + vegan_kosher_gluten_free) = 30 := by
  sorry

end NUMINAMATH_CALUDE_meals_without_restrictions_l1664_166483


namespace NUMINAMATH_CALUDE_sin_810_degrees_equals_one_l1664_166411

theorem sin_810_degrees_equals_one : Real.sin (810 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_810_degrees_equals_one_l1664_166411


namespace NUMINAMATH_CALUDE_penguin_fish_distribution_penguin_fish_distribution_proof_l1664_166462

theorem penguin_fish_distribution (total_penguins : ℕ) 
  (emperor_ratio adelie_ratio : ℕ) 
  (emperor_fish adelie_fish : ℚ) 
  (fish_constraint : ℕ) : Prop :=
  let emperor_count := (total_penguins * emperor_ratio) / (emperor_ratio + adelie_ratio)
  let adelie_count := (total_penguins * adelie_ratio) / (emperor_ratio + adelie_ratio)
  let total_fish_needed := (emperor_count : ℚ) * emperor_fish + (adelie_count : ℚ) * adelie_fish
  total_penguins = 48 ∧ 
  emperor_ratio = 3 ∧ 
  adelie_ratio = 5 ∧ 
  emperor_fish = 3/2 ∧ 
  adelie_fish = 2 ∧ 
  fish_constraint = 115 →
  total_fish_needed ≤ fish_constraint

-- Proof
theorem penguin_fish_distribution_proof : 
  penguin_fish_distribution 48 3 5 (3/2) 2 115 := by
  sorry

end NUMINAMATH_CALUDE_penguin_fish_distribution_penguin_fish_distribution_proof_l1664_166462


namespace NUMINAMATH_CALUDE_f_4cos2alpha_equals_4_l1664_166481

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function has period T if f(x + T) = f(x) for all x -/
def HasPeriod (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

theorem f_4cos2alpha_equals_4 
  (f : ℝ → ℝ) (α : ℝ) 
  (h_odd : IsOdd f) 
  (h_period : HasPeriod f 5) 
  (h_f_neg3 : f (-3) = 4) 
  (h_sin_alpha : Real.sin α = Real.sqrt 3 / 2) : 
  f (4 * Real.cos (2 * α)) = 4 := by
sorry

end NUMINAMATH_CALUDE_f_4cos2alpha_equals_4_l1664_166481


namespace NUMINAMATH_CALUDE_det_B_is_one_l1664_166419

open Matrix

def B (x y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![x, 2; -3, y]

theorem det_B_is_one (x y : ℝ) (h : B x y + (B x y)⁻¹ = 0) : 
  det (B x y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_det_B_is_one_l1664_166419


namespace NUMINAMATH_CALUDE_jellybean_guess_difference_l1664_166482

/-- The jellybean guessing problem -/
theorem jellybean_guess_difference :
  ∀ (guess1 guess2 guess3 guess4 : ℕ),
  guess1 = 100 →
  guess2 = 8 * guess1 →
  guess3 < guess2 →
  guess4 = (guess1 + guess2 + guess3) / 3 + 25 →
  guess4 = 525 →
  guess2 - guess3 = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_jellybean_guess_difference_l1664_166482


namespace NUMINAMATH_CALUDE_gcf_of_lcm_equals_five_l1664_166477

theorem gcf_of_lcm_equals_five : Nat.gcd (Nat.lcm 9 15) (Nat.lcm 10 21) = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_lcm_equals_five_l1664_166477


namespace NUMINAMATH_CALUDE_smiths_bakery_pies_l1664_166416

theorem smiths_bakery_pies (mcgees_pies : ℕ) (smiths_pies : ℕ) : 
  mcgees_pies = 16 → 
  smiths_pies = 4 * mcgees_pies + 6 → 
  smiths_pies = 70 := by
sorry

end NUMINAMATH_CALUDE_smiths_bakery_pies_l1664_166416


namespace NUMINAMATH_CALUDE_inequality_and_minimum_value_l1664_166423

theorem inequality_and_minimum_value {a b x : ℝ} (ha : a > 0) (hb : b > 0) (hx : 0 < x ∧ x < 1) :
  (a^2 / b + b^2 / a ≥ a + b) ∧
  (∀ y, y = (1 - x)^2 / x + x^2 / (1 - x) → y ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_minimum_value_l1664_166423


namespace NUMINAMATH_CALUDE_julie_order_amount_l1664_166485

/-- The amount of food ordered by Julie -/
def julie_order : ℝ := 10

/-- The amount of food ordered by Letitia -/
def letitia_order : ℝ := 20

/-- The amount of food ordered by Anton -/
def anton_order : ℝ := 30

/-- The tip percentage -/
def tip_percentage : ℝ := 0.20

/-- The individual tip amount paid by each person -/
def individual_tip : ℝ := 4

theorem julie_order_amount :
  julie_order = 10 ∧
  letitia_order = 20 ∧
  anton_order = 30 ∧
  tip_percentage = 0.20 ∧
  individual_tip = 4 →
  tip_percentage * (julie_order + letitia_order + anton_order) = 3 * individual_tip :=
by sorry

end NUMINAMATH_CALUDE_julie_order_amount_l1664_166485


namespace NUMINAMATH_CALUDE_betty_calculation_l1664_166465

theorem betty_calculation : ∀ (x y : ℚ),
  x = 8/100 →
  y = 325/100 →
  (x * y : ℚ) = 26/100 :=
by
  sorry

end NUMINAMATH_CALUDE_betty_calculation_l1664_166465


namespace NUMINAMATH_CALUDE_cube_third_times_eighth_equals_one_over_216_l1664_166497

theorem cube_third_times_eighth_equals_one_over_216 :
  (1 / 3 : ℚ)^3 * (1 / 8 : ℚ) = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_third_times_eighth_equals_one_over_216_l1664_166497


namespace NUMINAMATH_CALUDE_ivan_petrovich_savings_l1664_166432

/-- Simple interest calculation --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Proof of Ivan Petrovich's retirement savings --/
theorem ivan_petrovich_savings : 
  simple_interest 750000 0.08 12 = 1470000 := by
  sorry

end NUMINAMATH_CALUDE_ivan_petrovich_savings_l1664_166432


namespace NUMINAMATH_CALUDE_complete_square_formula_l1664_166431

theorem complete_square_formula (x y : ℝ) : x^2 - 2*x*y + y^2 = (x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_formula_l1664_166431


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l1664_166408

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℤ) :
  (∀ x : ℝ, (2*x - 1)^5 = a₀*x^5 + a₁*x^4 + a₂*x^3 + a₃*x^2 + a₄*x + a₅) →
  a₂ + a₃ = 40 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l1664_166408


namespace NUMINAMATH_CALUDE_problem_statement_l1664_166466

theorem problem_statement (x y : ℝ) 
  (h : |x - 1/2| + Real.sqrt (y^2 - 1) = 0) :
  |x| + |y| = 3/2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1664_166466


namespace NUMINAMATH_CALUDE_three_circles_theorem_l1664_166488

/-- Represents a circle with a center and a radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

/-- The area of the shaded region formed by three externally tangent circles -/
noncomputable def shaded_area (c1 c2 c3 : Circle) : ℝ := sorry

/-- The main theorem -/
theorem three_circles_theorem :
  let c1 : Circle := { center := (0, 0), radius := Real.sqrt 3 - 1 }
  let c2 : Circle := { center := (2, 0), radius := 3 - Real.sqrt 3 }
  let c3 : Circle := { center := (0, 2 * Real.sqrt 3), radius := 1 + Real.sqrt 3 }
  are_externally_tangent c1 c2 ∧
  are_externally_tangent c2 c3 ∧
  are_externally_tangent c3 c1 →
  ∃ (a b c : ℚ),
    shaded_area c1 c2 c3 = a * Real.sqrt 3 + b * Real.pi + c * Real.pi * Real.sqrt 3 ∧
    a + b + c = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_three_circles_theorem_l1664_166488


namespace NUMINAMATH_CALUDE_paul_books_theorem_l1664_166473

/-- The number of books Paul initially had -/
def initial_books : ℕ := 134

/-- The number of books Paul gave to his friend -/
def books_given : ℕ := 39

/-- The number of books Paul sold in the garage sale -/
def books_sold : ℕ := 27

/-- The number of books Paul had left -/
def books_left : ℕ := 68

/-- Theorem stating that the initial number of books equals the sum of books given away, sold, and left -/
theorem paul_books_theorem : initial_books = books_given + books_sold + books_left := by
  sorry

end NUMINAMATH_CALUDE_paul_books_theorem_l1664_166473


namespace NUMINAMATH_CALUDE_probability_white_ball_l1664_166440

theorem probability_white_ball (n : ℕ) : 
  (2 : ℚ) / (n + 2) = 2 / 5 → (n : ℚ) / (n + 2) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_white_ball_l1664_166440


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1664_166460

theorem smallest_n_congruence (n : ℕ) : 
  (∀ k < n, ¬(5^k ≡ k^5 [ZMOD 3])) ∧ (5^n ≡ n^5 [ZMOD 3]) ↔ n = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1664_166460


namespace NUMINAMATH_CALUDE_line_through_point_and_circle_center_l1664_166451

/-- The equation of the line passing through (2,1) and the center of the circle (x-1)^2 + (y+2)^2 = 5 is 3x - y - 5 = 0 -/
theorem line_through_point_and_circle_center :
  let point : ℝ × ℝ := (2, 1)
  let circle_center : ℝ × ℝ := (1, -2)
  let line_equation (x y : ℝ) := 3 * x - y - 5 = 0
  ∀ x y : ℝ, line_equation x y ↔ (y - point.2) / (x - point.1) = (circle_center.2 - point.2) / (circle_center.1 - point.1) :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_and_circle_center_l1664_166451


namespace NUMINAMATH_CALUDE_min_shirts_acme_cheaper_l1664_166407

/-- Represents the cost function for Acme T-Shirt Company -/
def acme_cost (x : ℕ) : ℚ := 30 + 7 * x

/-- Represents the cost function for Gamma T-shirt Company -/
def gamma_cost (x : ℕ) : ℚ := 11 * x

/-- Theorem stating the minimum number of shirts for which Acme is cheaper than Gamma -/
theorem min_shirts_acme_cheaper :
  ∃ n : ℕ, (∀ m : ℕ, m ≥ n → acme_cost m < gamma_cost m) ∧
           (∀ k : ℕ, k < n → acme_cost k ≥ gamma_cost k) ∧
           n = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_shirts_acme_cheaper_l1664_166407


namespace NUMINAMATH_CALUDE_nathan_baseball_weeks_l1664_166468

/-- Nathan's baseball playing problem -/
theorem nathan_baseball_weeks (nathan_daily_hours tobias_daily_hours : ℕ) 
  (total_hours : ℕ) (tobias_weeks : ℕ) :
  nathan_daily_hours = 3 →
  tobias_daily_hours = 5 →
  tobias_weeks = 1 →
  total_hours = 77 →
  ∃ w : ℕ, w * (7 * nathan_daily_hours) + tobias_weeks * (7 * tobias_daily_hours) = total_hours ∧ w = 2 := by
  sorry

#check nathan_baseball_weeks

end NUMINAMATH_CALUDE_nathan_baseball_weeks_l1664_166468


namespace NUMINAMATH_CALUDE_max_prob_with_C_second_l1664_166430

/-- Represents the probability of winning against a player -/
structure WinProbability (α : Type) where
  prob : α → ℝ
  pos : ∀ x, prob x > 0

variable {α : Type}

/-- The players A, B, and C -/
inductive Player : Type where
  | A : Player
  | B : Player
  | C : Player

/-- The probabilities of winning against each player -/
def win_prob (p : WinProbability Player) : Prop :=
  p.prob Player.A < p.prob Player.B ∧ p.prob Player.B < p.prob Player.C

/-- The probability of winning two consecutive games when player x is in the second game -/
def prob_two_consec_wins (p : WinProbability Player) (x : Player) : ℝ :=
  2 * (p.prob Player.A * p.prob x + p.prob Player.B * p.prob x + p.prob Player.C * p.prob x
     - 2 * p.prob Player.A * p.prob Player.B * p.prob Player.C)

/-- The theorem stating that the probability is maximized when C is in the second game -/
theorem max_prob_with_C_second (p : WinProbability Player) (h : win_prob p) :
    ∀ x : Player, prob_two_consec_wins p Player.C ≥ prob_two_consec_wins p x :=
  sorry

end NUMINAMATH_CALUDE_max_prob_with_C_second_l1664_166430


namespace NUMINAMATH_CALUDE_average_cost_theorem_l1664_166491

def iPhone_quantity : ℕ := 100
def iPhone_price : ℝ := 1000
def iPhone_tax_rate : ℝ := 0.1

def iPad_quantity : ℕ := 20
def iPad_price : ℝ := 900
def iPad_discount_rate : ℝ := 0.05

def AppleTV_quantity : ℕ := 80
def AppleTV_price : ℝ := 200
def AppleTV_tax_rate : ℝ := 0.08

def MacBook_quantity : ℕ := 50
def MacBook_price : ℝ := 1500
def MacBook_discount_rate : ℝ := 0.15

def total_quantity : ℕ := iPhone_quantity + iPad_quantity + AppleTV_quantity + MacBook_quantity

def total_cost : ℝ :=
  iPhone_quantity * iPhone_price * (1 + iPhone_tax_rate) +
  iPad_quantity * iPad_price * (1 - iPad_discount_rate) +
  AppleTV_quantity * AppleTV_price * (1 + AppleTV_tax_rate) +
  MacBook_quantity * MacBook_price * (1 - MacBook_discount_rate)

theorem average_cost_theorem :
  total_cost / total_quantity = 832.52 := by sorry

end NUMINAMATH_CALUDE_average_cost_theorem_l1664_166491


namespace NUMINAMATH_CALUDE_total_problems_solved_l1664_166445

/-- The number of problems Seokjin initially solved -/
def initial_problems : ℕ := 12

/-- The number of additional problems Seokjin solved -/
def additional_problems : ℕ := 7

/-- Theorem: The total number of problems Seokjin solved is 19 -/
theorem total_problems_solved : initial_problems + additional_problems = 19 := by
  sorry

end NUMINAMATH_CALUDE_total_problems_solved_l1664_166445


namespace NUMINAMATH_CALUDE_percentage_increase_l1664_166450

theorem percentage_increase (x : ℝ) (h : x = 78.4) : 
  (x - 70) / 70 * 100 = 12 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l1664_166450


namespace NUMINAMATH_CALUDE_two_unique_intersection_lines_l1664_166402

/-- A parabola defined by the equation y^2 = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line on the plane -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Predicate to check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

/-- Predicate to check if a line intersects the parabola at only one point -/
def uniqueIntersection (l : Line) : Prop :=
  ∃! p : Point, pointOnLine p l ∧ parabola p.x p.y

/-- The point (2, 4) -/
def givenPoint : Point := ⟨2, 4⟩

/-- The theorem stating that there are exactly two lines passing through (2, 4) 
    that intersect the parabola y^2 = 8x at only one point -/
theorem two_unique_intersection_lines : 
  ∃! (l1 l2 : Line), 
    pointOnLine givenPoint l1 ∧ 
    pointOnLine givenPoint l2 ∧ 
    uniqueIntersection l1 ∧ 
    uniqueIntersection l2 ∧ 
    l1 ≠ l2 :=
sorry

end NUMINAMATH_CALUDE_two_unique_intersection_lines_l1664_166402


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_by_10_l1664_166456

theorem least_subtraction_for_divisibility_by_10 :
  ∃ (n : ℕ), n = 2 ∧ 
  (427398 - n) % 10 = 0 ∧
  ∀ (m : ℕ), m < n → (427398 - m) % 10 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_by_10_l1664_166456


namespace NUMINAMATH_CALUDE_chord_length_squared_for_specific_circles_exists_configuration_for_specific_circles_l1664_166447

/-- Two circles with given radii and distance between centers -/
structure TwoCircles where
  r1 : ℝ
  r2 : ℝ
  center_distance : ℝ

/-- Represents the configuration of two intersecting circles with equal chords -/
structure IntersectingCirclesWithEqualChords extends TwoCircles where
  chord_length : ℝ
  chord_length_squared : ℝ
  chord_length_squared_eq : chord_length_squared = chord_length ^ 2

/-- The main theorem stating the square of the chord length for the given configuration -/
theorem chord_length_squared_for_specific_circles
  (config : IntersectingCirclesWithEqualChords)
  (h1 : config.r1 = 10)
  (h2 : config.r2 = 7)
  (h3 : config.center_distance = 15) :
  config.chord_length_squared = 245 := by
  sorry

/-- Existence of the configuration for the given circle sizes -/
theorem exists_configuration_for_specific_circles :
  ∃ (config : IntersectingCirclesWithEqualChords),
    config.r1 = 10 ∧ config.r2 = 7 ∧ config.center_distance = 15 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_squared_for_specific_circles_exists_configuration_for_specific_circles_l1664_166447


namespace NUMINAMATH_CALUDE_angle_value_l1664_166444

theorem angle_value (PQR : ℝ) (x : ℝ) : 
  PQR = 90 → 2*x + x = PQR → x = 30 := by sorry

end NUMINAMATH_CALUDE_angle_value_l1664_166444
