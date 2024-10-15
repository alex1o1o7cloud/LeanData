import Mathlib

namespace NUMINAMATH_CALUDE_max_a1_value_l690_69051

/-- A sequence of non-negative real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n ≥ 0) ∧
  (∀ n ≥ 2, a (n + 1) = a n - a (n - 1) + n)

theorem max_a1_value (a : ℕ → ℝ) (h : RecurrenceSequence a) (h2022 : a 2 * a 2022 = 1) :
  ∃ (max_a1 : ℝ), a 1 ≤ max_a1 ∧ max_a1 = 4051 / 2025 :=
sorry

end NUMINAMATH_CALUDE_max_a1_value_l690_69051


namespace NUMINAMATH_CALUDE_expression_evaluation_l690_69058

theorem expression_evaluation (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) :
  let y := 1 / x + z
  (x - 1 / x) * (y + 1 / y) = ((x^2 - 1) * (1 + 2*x*z + x^2*z^2 + x^2)) / (x^2 * (1 + x*z)) :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l690_69058


namespace NUMINAMATH_CALUDE_harrys_seed_purchase_l690_69046

/-- Represents the number of packets of each seed type and the total spent -/
structure SeedPurchase where
  pumpkin : ℕ
  tomato : ℕ
  chili : ℕ
  total_spent : ℚ

/-- Calculates the total cost of a seed purchase -/
def calculate_total_cost (purchase : SeedPurchase) : ℚ :=
  2.5 * purchase.pumpkin + 1.5 * purchase.tomato + 0.9 * purchase.chili

/-- Theorem stating that Harry's purchase of 3 pumpkin, 4 tomato, and 5 chili pepper seed packets
    totaling $18 is correct -/
theorem harrys_seed_purchase :
  ∃ (purchase : SeedPurchase),
    purchase.pumpkin = 3 ∧
    purchase.tomato = 4 ∧
    purchase.chili = 5 ∧
    purchase.total_spent = 18 ∧
    calculate_total_cost purchase = purchase.total_spent :=
  sorry

end NUMINAMATH_CALUDE_harrys_seed_purchase_l690_69046


namespace NUMINAMATH_CALUDE_thomson_incentive_spending_l690_69009

theorem thomson_incentive_spending (incentive : ℝ) (savings : ℝ) (f : ℝ) : 
  incentive = 240 →
  savings = 84 →
  savings = (3/4) * (incentive - f * incentive - (1/5) * incentive) →
  f = 1/3 := by
sorry

end NUMINAMATH_CALUDE_thomson_incentive_spending_l690_69009


namespace NUMINAMATH_CALUDE_min_mutually_visible_pairs_l690_69097

/-- A configuration of birds on a circle. -/
structure BirdConfiguration where
  /-- The total number of birds. -/
  total_birds : ℕ
  /-- The number of points on the circle where birds can sit. -/
  num_points : ℕ
  /-- The distribution of birds across the points. -/
  distribution : Fin num_points → ℕ
  /-- The sum of birds across all points equals the total number of birds. -/
  sum_constraint : (Finset.univ.sum distribution) = total_birds

/-- The number of mutually visible pairs in a given configuration. -/
def mutually_visible_pairs (config : BirdConfiguration) : ℕ :=
  Finset.sum Finset.univ (fun i => config.distribution i * (config.distribution i - 1) / 2)

/-- The theorem stating the minimum number of mutually visible pairs. -/
theorem min_mutually_visible_pairs :
  ∀ (config : BirdConfiguration),
    config.total_birds = 155 →
    mutually_visible_pairs config ≥ 270 :=
  sorry

end NUMINAMATH_CALUDE_min_mutually_visible_pairs_l690_69097


namespace NUMINAMATH_CALUDE_find_first_number_l690_69081

theorem find_first_number (x : ℝ) : 
  let set1 := [20, 40, 60]
  let set2 := [x, 70, 19]
  (set1.sum / set1.length) = (set2.sum / set2.length) + 7 →
  x = 10 := by
sorry

end NUMINAMATH_CALUDE_find_first_number_l690_69081


namespace NUMINAMATH_CALUDE_five_digit_divisibility_l690_69084

/-- A five-digit number -/
def FiveDigitNumber : Type := { n : ℕ // 10000 ≤ n ∧ n < 100000 }

/-- A four-digit number -/
def FourDigitNumber : Type := { n : ℕ // 1000 ≤ n ∧ n < 10000 }

/-- Extract the four-digit number from a five-digit number by removing the middle digit -/
def extractFourDigit (n : FiveDigitNumber) : FourDigitNumber :=
  sorry

theorem five_digit_divisibility (n : FiveDigitNumber) :
  (∃ (m : FourDigitNumber), m = extractFourDigit n ∧ n.val % m.val = 0) ↔ n.val % 1000 = 0 :=
sorry

end NUMINAMATH_CALUDE_five_digit_divisibility_l690_69084


namespace NUMINAMATH_CALUDE_blueberry_count_l690_69052

/-- Represents the number of berries in a box of a specific color -/
structure BerryBox where
  blue : ℕ
  red : ℕ
  green : ℕ

/-- The change in berry counts when replacing boxes -/
structure BerryChange where
  total : ℤ
  difference : ℤ

theorem blueberry_count (box : BerryBox) 
  (replace_blue_with_red : BerryChange)
  (replace_green_with_blue : BerryChange) :
  (replace_blue_with_red.total = 10) →
  (replace_blue_with_red.difference = 50) →
  (replace_green_with_blue.total = -5) →
  (replace_green_with_blue.difference = -30) →
  (box.red - box.blue = replace_blue_with_red.total) →
  (box.blue - box.green = -replace_green_with_blue.total) →
  (box.green - 2 * box.blue = -replace_green_with_blue.difference) →
  box.blue = 35 := by
  sorry

end NUMINAMATH_CALUDE_blueberry_count_l690_69052


namespace NUMINAMATH_CALUDE_first_part_speed_l690_69087

def trip_length : ℝ := 12
def part_time : ℝ := 0.25  -- 15 minutes in hours
def second_part_speed : ℝ := 12
def third_part_speed : ℝ := 20

theorem first_part_speed :
  ∃ (v : ℝ), v * part_time + second_part_speed * part_time + third_part_speed * part_time = trip_length ∧ v = 16 := by
  sorry

end NUMINAMATH_CALUDE_first_part_speed_l690_69087


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l690_69053

/-- A point in the Cartesian plane is in the fourth quadrant if its x-coordinate is positive and its y-coordinate is negative -/
def fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

/-- The point (2, -4) is in the fourth quadrant -/
theorem point_in_fourth_quadrant : fourth_quadrant (2, -4) := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l690_69053


namespace NUMINAMATH_CALUDE_third_square_is_G_l690_69080

-- Define the set of squares
inductive Square : Type
| A | B | C | D | E | F | G | H

-- Define the placement order
def PlacementOrder : List Square := [Square.F, Square.H, Square.G, Square.D, Square.A, Square.B, Square.C, Square.E]

-- Define the size of each small square
def SmallSquareSize : Nat := 2

-- Define the size of the large square
def LargeSquareSize : Nat := 4

-- Define the total number of squares
def TotalSquares : Nat := 8

-- Define the visibility property
def IsFullyVisible (s : Square) : Prop := s = Square.E

-- Define the third placed square
def ThirdPlacedSquare : Square := PlacementOrder[2]

-- Theorem statement
theorem third_square_is_G :
  (∀ s : Square, s ≠ Square.E → ¬IsFullyVisible s) →
  IsFullyVisible Square.E →
  TotalSquares = 8 →
  SmallSquareSize = 2 →
  LargeSquareSize = 4 →
  ThirdPlacedSquare = Square.G :=
by sorry

end NUMINAMATH_CALUDE_third_square_is_G_l690_69080


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l690_69069

theorem right_triangle_third_side : ∀ a b c : ℝ,
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a = 6 ∧ b = 8) ∨ (a = 6 ∧ c = 8) ∨ (b = 6 ∧ c = 8) →
  (a^2 + b^2 = c^2) →
  c = 10 ∨ c = 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l690_69069


namespace NUMINAMATH_CALUDE_retail_price_calculation_l690_69004

/-- The retail price of a machine, given wholesale price, discount, and profit margin -/
theorem retail_price_calculation (wholesale_price discount profit_margin : ℚ) 
  (h1 : wholesale_price = 126)
  (h2 : discount = 10/100)
  (h3 : profit_margin = 20/100)
  (h4 : profit_margin * wholesale_price + wholesale_price = (1 - discount) * retail_price) :
  retail_price = 168 := by
  sorry

#check retail_price_calculation

end NUMINAMATH_CALUDE_retail_price_calculation_l690_69004


namespace NUMINAMATH_CALUDE_equal_implies_parallel_l690_69014

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def parallel (a b : V) : Prop := ∃ (k : ℝ), a = k • b ∨ b = k • a

theorem equal_implies_parallel (a b : V) : a = b → parallel a b := by
  sorry

end NUMINAMATH_CALUDE_equal_implies_parallel_l690_69014


namespace NUMINAMATH_CALUDE_remainder_17_63_mod_7_l690_69064

theorem remainder_17_63_mod_7 :
  ∃ k : ℤ, 17^63 = 7 * k + 6 :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_17_63_mod_7_l690_69064


namespace NUMINAMATH_CALUDE_total_weight_of_aluminum_carbonate_l690_69085

/-- Atomic weight of Aluminum in g/mol -/
def Al_weight : ℝ := 26.98

/-- Atomic weight of Carbon in g/mol -/
def C_weight : ℝ := 12.01

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- Number of Aluminum atoms in Al2(CO3)3 -/
def Al_count : ℕ := 2

/-- Number of Carbon atoms in Al2(CO3)3 -/
def C_count : ℕ := 3

/-- Number of Oxygen atoms in Al2(CO3)3 -/
def O_count : ℕ := 9

/-- Number of moles of Al2(CO3)3 -/
def moles : ℝ := 6

/-- Calculates the molecular weight of Al2(CO3)3 in g/mol -/
def molecular_weight : ℝ :=
  Al_count * Al_weight + C_count * C_weight + O_count * O_weight

/-- Theorem stating the total weight of 6 moles of Al2(CO3)3 -/
theorem total_weight_of_aluminum_carbonate :
  moles * molecular_weight = 1403.94 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_aluminum_carbonate_l690_69085


namespace NUMINAMATH_CALUDE_quadratic_function_with_equal_roots_l690_69094

/-- A quadratic function with specific properties -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_with_equal_roots 
  (f : ℝ → ℝ) 
  (h1 : QuadraticFunction f)
  (h2 : ∀ x, deriv f x = 2 * x + 2)
  (h3 : ∃! r : ℝ, f r = 0 ∧ (deriv f r = 0)) :
  ∀ x, f x = x^2 + 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_with_equal_roots_l690_69094


namespace NUMINAMATH_CALUDE_square_side_length_l690_69032

theorem square_side_length : ∃ (x : ℝ), x > 0 ∧ x^2 = 6^2 + 8^2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_square_side_length_l690_69032


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l690_69037

theorem triangle_abc_proof (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * Real.cos A * Real.cos B - b * Real.sin A * Real.sin A - c * Real.cos A = 2 * b * Real.cos B →
  b = Real.sqrt 7 * a →
  (1/2) * a * c * Real.sin B = 2 * Real.sqrt 3 →
  B = 2 * π / 3 ∧ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_proof_l690_69037


namespace NUMINAMATH_CALUDE_manny_cookie_slices_left_l690_69019

/-- Calculates the number of cookie slices left after distribution --/
def cookie_slices_left (num_pies : ℕ) (slices_per_pie : ℕ) (total_people : ℕ) (half_slice_people : ℕ) : ℕ :=
  let total_slices := num_pies * slices_per_pie
  let full_slice_people := total_people - half_slice_people
  let distributed_slices := full_slice_people + (half_slice_people / 2)
  total_slices - distributed_slices

/-- Theorem stating the number of cookie slices left in Manny's scenario --/
theorem manny_cookie_slices_left : cookie_slices_left 6 12 39 3 = 33 := by
  sorry

#eval cookie_slices_left 6 12 39 3

end NUMINAMATH_CALUDE_manny_cookie_slices_left_l690_69019


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l690_69068

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l690_69068


namespace NUMINAMATH_CALUDE_min_value_expression_l690_69031

theorem min_value_expression (x y : ℝ) : 
  (x * y - 2)^2 + (x + y - 1)^2 ≥ 0 ∧ 
  ∃ (a b : ℝ), (a * b - 2)^2 + (a + b - 1)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l690_69031


namespace NUMINAMATH_CALUDE_complex_equation_solution_l690_69089

theorem complex_equation_solution :
  ∃ (z : ℂ), 2 - (3 + Complex.I) * z = 1 - (3 - Complex.I) * z ∧ z = Complex.I / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l690_69089


namespace NUMINAMATH_CALUDE_final_state_digits_l690_69049

/-- Represents the state of the board as three integers -/
structure BoardState where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Performs one iteration of pairwise sum replacement -/
def iterate (state : BoardState) : BoardState :=
  { a := (state.a + state.b) % 10,
    b := (state.a + state.c) % 10,
    c := (state.b + state.c) % 10 }

/-- Performs n iterations of pairwise sum replacement -/
def iterateN (n : ℕ) (state : BoardState) : BoardState :=
  match n with
  | 0 => state
  | n + 1 => iterate (iterateN n state)

/-- The main theorem to be proved -/
theorem final_state_digits (initialState : BoardState) :
  initialState.a = 1 ∧ initialState.b = 2 ∧ initialState.c = 4 →
  let finalState := iterateN 60 initialState
  (finalState.a = 6 ∧ finalState.b = 7 ∧ finalState.c = 9) ∨
  (finalState.a = 6 ∧ finalState.b = 9 ∧ finalState.c = 7) ∨
  (finalState.a = 7 ∧ finalState.b = 6 ∧ finalState.c = 9) ∨
  (finalState.a = 7 ∧ finalState.b = 9 ∧ finalState.c = 6) ∨
  (finalState.a = 9 ∧ finalState.b = 6 ∧ finalState.c = 7) ∨
  (finalState.a = 9 ∧ finalState.b = 7 ∧ finalState.c = 6) :=
by sorry

end NUMINAMATH_CALUDE_final_state_digits_l690_69049


namespace NUMINAMATH_CALUDE_subset_of_sqrt_eleven_l690_69086

theorem subset_of_sqrt_eleven (h : Real.sqrt 11 < 2 * Real.sqrt 3) :
  {Real.sqrt 11} ⊆ {x : ℝ | |x| ≤ 2 * Real.sqrt 3} := by
  sorry

end NUMINAMATH_CALUDE_subset_of_sqrt_eleven_l690_69086


namespace NUMINAMATH_CALUDE_train_length_calculation_l690_69034

/-- Calculates the length of a train given its speed and time to cross a point. -/
def trainLength (speed : Real) (time : Real) : Real :=
  speed * time

theorem train_length_calculation (speed : Real) (time : Real) 
  (h1 : speed = 90 * 1000 / 3600) -- Speed in m/s
  (h2 : time = 20) : -- Time in seconds
  trainLength speed time = 500 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l690_69034


namespace NUMINAMATH_CALUDE_complex_determinant_equation_l690_69021

-- Define the determinant operation
def det (a b c d : ℂ) : ℂ := a * d - b * c

-- Define the theorem
theorem complex_determinant_equation :
  ∀ z : ℂ, det 1 (-1) z (z * Complex.I) = 2 → z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_determinant_equation_l690_69021


namespace NUMINAMATH_CALUDE_binary_multiplication_theorem_l690_69027

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Represents a binary number as a list of bits (least significant bit first) -/
def binary_1101 : List Bool := [true, false, true, true]
def binary_111 : List Bool := [true, true, true]
def binary_10001111 : List Bool := [true, true, true, true, false, false, false, true]

theorem binary_multiplication_theorem :
  (binary_to_decimal binary_1101) * (binary_to_decimal binary_111) =
  binary_to_decimal binary_10001111 ∧
  (binary_to_decimal binary_1101) * (binary_to_decimal binary_111) = 143 := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_theorem_l690_69027


namespace NUMINAMATH_CALUDE_unique_function_existence_l690_69028

theorem unique_function_existence (g : ℂ → ℂ) (ω a : ℂ) 
  (h1 : ω^3 = 1) (h2 : ω ≠ 1) :
  ∃! f : ℂ → ℂ, ∀ z : ℂ, f z + f (ω * z + a) = g z := by
  sorry

end NUMINAMATH_CALUDE_unique_function_existence_l690_69028


namespace NUMINAMATH_CALUDE_no_integer_distances_point_l690_69070

theorem no_integer_distances_point (a b : ℕ) (ha : Odd a) (hb : Odd b) :
  ¬ ∃ (x y : ℚ), 0 < x ∧ x < b ∧ 0 < y ∧ y < a ∧
    (∀ (i j : ℕ), i ≤ 1 ∧ j ≤ 1 →
      ∃ (n : ℕ), (x - i * b)^2 + (y - j * a)^2 = n^2) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_distances_point_l690_69070


namespace NUMINAMATH_CALUDE_equation_solutions_l690_69048

/-- The integer part of a real number -/
noncomputable def intPart (x : ℝ) : ℤ := Int.floor x

/-- The fractional part of a real number -/
noncomputable def fracPart (x : ℝ) : ℝ := x - Int.floor x

/-- The main theorem stating the solutions to the equation -/
theorem equation_solutions :
  ∀ x : ℝ, 
  (intPart x : ℝ) * fracPart x + x = 2 * fracPart x + 9 →
  (x = 9 ∨ x = 8 + 1/7 ∨ x = 7 + 1/3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l690_69048


namespace NUMINAMATH_CALUDE_gcd_problem_l690_69003

theorem gcd_problem (h : Prime 103) : 
  Nat.gcd (103^7 + 1) (103^7 + 103^5 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l690_69003


namespace NUMINAMATH_CALUDE_factorial_division_l690_69073

theorem factorial_division (h : Nat.factorial 10 = 3628800) :
  Nat.factorial 10 / Nat.factorial 4 = 151200 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l690_69073


namespace NUMINAMATH_CALUDE_vector_problem_l690_69005

def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-1, 3]

def perpendicular (v w : Fin 2 → ℝ) : Prop :=
  (v 0) * (w 0) + (v 1) * (w 1) = 0

def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (c : ℝ), v = fun i => c * (w i)

theorem vector_problem :
  (∃ k : ℝ, perpendicular (fun i => k * (a i) + (b i)) (fun i => (a i) - 3 * (b i)) ∧ k = -2.5) ∧
  (∃ k : ℝ, parallel (fun i => k * (a i) + (b i)) (fun i => (a i) - 3 * (b i)) ∧ k = -1/3) :=
sorry

end NUMINAMATH_CALUDE_vector_problem_l690_69005


namespace NUMINAMATH_CALUDE_smallest_k_inequality_l690_69002

theorem smallest_k_inequality (a b c : ℕ+) : 
  ∃ (k : ℕ+), k = 1297 ∧ 
  (16 * a^2 + 36 * b^2 + 81 * c^2) * (81 * a^2 + 36 * b^2 + 16 * c^2) < k * (a^2 + b^2 + c^2)^2 ∧
  ∀ (m : ℕ+), m < 1297 → 
  (16 * a^2 + 36 * b^2 + 81 * c^2) * (81 * a^2 + 36 * b^2 + 16 * c^2) ≥ m * (a^2 + b^2 + c^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_inequality_l690_69002


namespace NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l690_69026

theorem arithmetic_sequence_20th_term (a : ℕ → ℤ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_sum_odd : a 1 + a 3 + a 5 = 105)
  (h_sum_even : a 2 + a 4 + a 6 = 99) :
  a 20 = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l690_69026


namespace NUMINAMATH_CALUDE_train_crossing_time_l690_69082

theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 300 → 
  train_speed_kmh = 36 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l690_69082


namespace NUMINAMATH_CALUDE_integer_root_of_cubic_l690_69075

theorem integer_root_of_cubic (b c : ℚ) :
  (∃ x : ℝ, x^3 + b*x + c = 0 ∧ x = 2 - Real.sqrt 5) →
  (∃ r : ℤ, r^3 + b*r + c = 0) →
  (∃ r : ℤ, r^3 + b*r + c = 0 ∧ r = -4) :=
by sorry

end NUMINAMATH_CALUDE_integer_root_of_cubic_l690_69075


namespace NUMINAMATH_CALUDE_coefficient_equals_nth_term_l690_69065

def a (n : ℕ) : ℕ := 3 * n - 5

theorem coefficient_equals_nth_term :
  let coefficient : ℕ := (Nat.choose 5 4) + (Nat.choose 6 4) + (Nat.choose 7 4)
  coefficient = a 20 := by sorry

end NUMINAMATH_CALUDE_coefficient_equals_nth_term_l690_69065


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_squared_l690_69067

/-- A circle inscribed in a quadrilateral EFGH -/
structure InscribedCircle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The circle is tangent to EF at R -/
  ER : ℝ
  /-- The circle is tangent to EF at R -/
  RF : ℝ
  /-- The circle is tangent to GH at S -/
  GS : ℝ
  /-- The circle is tangent to GH at S -/
  SH : ℝ

/-- The theorem stating that the square of the radius of the inscribed circle is 1357 -/
theorem inscribed_circle_radius_squared (c : InscribedCircle)
  (h1 : c.ER = 15)
  (h2 : c.RF = 31)
  (h3 : c.GS = 47)
  (h4 : c.SH = 29) :
  c.r ^ 2 = 1357 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_squared_l690_69067


namespace NUMINAMATH_CALUDE_area_of_region_l690_69050

theorem area_of_region (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 20 ∧ 
   A = Real.pi * (Real.sqrt ((x + 5)^2 + (y - 2)^2))^2 ∧
   x^2 + y^2 + 10*x - 4*y + 9 = 0) := by
  sorry

end NUMINAMATH_CALUDE_area_of_region_l690_69050


namespace NUMINAMATH_CALUDE_unique_stamp_value_l690_69054

/-- Given stamps of denominations 6, n, and n+1 cents, 
    this function checks if 115 cents is the greatest 
    postage that cannot be formed -/
def is_valid_stamp_set (n : ℕ) : Prop :=
  n > 0 ∧ 
  (∀ m : ℕ, m > 115 → ∃ a b c : ℕ, m = 6*a + n*b + (n+1)*c) ∧
  ¬(∃ a b c : ℕ, 115 = 6*a + n*b + (n+1)*c)

/-- The theorem stating that 24 is the only value of n 
    that satisfies the stamp condition -/
theorem unique_stamp_value : 
  (∃! n : ℕ, is_valid_stamp_set n) ∧ 
  (∀ n : ℕ, is_valid_stamp_set n → n = 24) :=
sorry

end NUMINAMATH_CALUDE_unique_stamp_value_l690_69054


namespace NUMINAMATH_CALUDE_quadratic_properties_l690_69063

-- Define the quadratic function
def f (x : ℝ) : ℝ := -(x - 1)^2 + 2

-- Theorem stating the properties of the quadratic function
theorem quadratic_properties :
  (∃ (a : ℝ), ∀ (x : ℝ), f x = f (2 - x)) ∧ 
  (∀ (x : ℝ), f x ≤ 2) ∧
  (f 1 = 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l690_69063


namespace NUMINAMATH_CALUDE_simplify_square_roots_l690_69077

theorem simplify_square_roots : Real.sqrt 49 - Real.sqrt 144 + Real.sqrt 9 = -2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l690_69077


namespace NUMINAMATH_CALUDE_sugar_percentage_after_addition_l690_69001

def initial_volume : ℝ := 340
def water_percentage : ℝ := 0.75
def kola_percentage : ℝ := 0.05
def added_sugar : ℝ := 3.2
def added_water : ℝ := 12
def added_kola : ℝ := 6.8

theorem sugar_percentage_after_addition :
  let initial_sugar_percentage : ℝ := 1 - water_percentage - kola_percentage
  let initial_sugar_volume : ℝ := initial_sugar_percentage * initial_volume
  let final_sugar_volume : ℝ := initial_sugar_volume + added_sugar
  let final_volume : ℝ := initial_volume + added_sugar + added_water + added_kola
  let final_sugar_percentage : ℝ := final_sugar_volume / final_volume
  ∃ ε > 0, |final_sugar_percentage - 0.1967| < ε :=
by sorry

end NUMINAMATH_CALUDE_sugar_percentage_after_addition_l690_69001


namespace NUMINAMATH_CALUDE_paper_folding_perimeter_ratio_l690_69006

/-- Given a square piece of paper with side length 8 inches, when folded and cut as described,
    the ratio of the perimeter of the larger rectangle to the perimeter of one of the smaller rectangles is 3/2. -/
theorem paper_folding_perimeter_ratio :
  let initial_side_length : ℝ := 8
  let large_rectangle_length : ℝ := initial_side_length
  let large_rectangle_width : ℝ := initial_side_length / 2
  let small_rectangle_side : ℝ := initial_side_length / 2
  let large_perimeter : ℝ := 2 * (large_rectangle_length + large_rectangle_width)
  let small_perimeter : ℝ := 4 * small_rectangle_side
  large_perimeter / small_perimeter = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_paper_folding_perimeter_ratio_l690_69006


namespace NUMINAMATH_CALUDE_parabola_passes_through_point_l690_69056

/-- The parabola y = (1/2)x^2 - 2 passes through the point (2, 0) -/
theorem parabola_passes_through_point :
  let f : ℝ → ℝ := fun x ↦ (1/2) * x^2 - 2
  f 2 = 0 := by sorry

end NUMINAMATH_CALUDE_parabola_passes_through_point_l690_69056


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l690_69060

theorem pure_imaginary_condition (a : ℝ) : 
  (∃ b : ℝ, (a + 1) * (a - 1 + Complex.I) = Complex.I * b) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l690_69060


namespace NUMINAMATH_CALUDE_correct_yeast_counting_operation_l690_69066

/-- Represents an experimental operation -/
inductive ExperimentalOperation
  | YeastCounting
  | PigmentSeparation
  | AuxinRooting
  | Plasmolysis

/-- Determines if an experimental operation is correct -/
def is_correct_operation (op : ExperimentalOperation) : Prop :=
  match op with
  | ExperimentalOperation.YeastCounting => true
  | _ => false

/-- Theorem stating that shaking the culture solution before yeast counting is the correct operation -/
theorem correct_yeast_counting_operation :
  is_correct_operation ExperimentalOperation.YeastCounting := by
  sorry

end NUMINAMATH_CALUDE_correct_yeast_counting_operation_l690_69066


namespace NUMINAMATH_CALUDE_square_differences_l690_69041

theorem square_differences (n : ℕ) : 
  (n + 1)^2 = n^2 + (2*n + 1) ∧ (n - 1)^2 = n^2 - (2*n - 1) :=
by sorry

end NUMINAMATH_CALUDE_square_differences_l690_69041


namespace NUMINAMATH_CALUDE_average_transformation_l690_69015

theorem average_transformation (x₁ x₂ x₃ : ℝ) (h : (x₁ + x₂ + x₃) / 3 = 2) :
  ((2 * x₁ + 4) + (2 * x₂ + 4) + (2 * x₃ + 4)) / 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_average_transformation_l690_69015


namespace NUMINAMATH_CALUDE_strawberry_sugar_purchase_strategy_l690_69045

theorem strawberry_sugar_purchase_strategy :
  -- Define constants
  let discount_threshold : ℝ := 1000
  let discount_rate : ℝ := 0.5
  let budget : ℝ := 1200
  let strawberry_price : ℝ := 300
  let sugar_price : ℝ := 30
  let strawberry_amount : ℝ := 4
  let sugar_amount : ℝ := 6

  -- Define purchase strategy
  let first_purchase_strawberry : ℝ := 3
  let first_purchase_sugar : ℝ := 4
  let second_purchase_strawberry : ℝ := strawberry_amount - first_purchase_strawberry
  let second_purchase_sugar : ℝ := sugar_amount - first_purchase_sugar

  -- Calculate costs
  let first_purchase_cost : ℝ := first_purchase_strawberry * strawberry_price + first_purchase_sugar * sugar_price
  let second_purchase_full_price : ℝ := second_purchase_strawberry * strawberry_price + second_purchase_sugar * sugar_price
  let second_purchase_discounted : ℝ := second_purchase_full_price * (1 - discount_rate)
  let total_cost : ℝ := first_purchase_cost + second_purchase_discounted

  -- Theorem statement
  (first_purchase_cost ≥ discount_threshold) →
  (total_cost ≤ budget) ∧
  (first_purchase_strawberry + second_purchase_strawberry = strawberry_amount) ∧
  (first_purchase_sugar + second_purchase_sugar = sugar_amount) :=
by sorry

end NUMINAMATH_CALUDE_strawberry_sugar_purchase_strategy_l690_69045


namespace NUMINAMATH_CALUDE_regular_polyhedron_spheres_l690_69008

/-- A regular polyhedron -/
structure RegularPolyhedron where
  -- We don't need to define the internal structure,
  -- as the problem doesn't rely on specific properties

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points -/
def distance (p q : Point3D) : ℝ :=
  sorry

/-- Distance from a point to a face of the polyhedron -/
def distanceToFace (p : Point3D) (poly : RegularPolyhedron) (face : Nat) : ℝ :=
  sorry

/-- Get a vertex of the polyhedron -/
def getVertex (poly : RegularPolyhedron) (v : Nat) : Point3D :=
  sorry

/-- Number of vertices in the polyhedron -/
def numVertices (poly : RegularPolyhedron) : Nat :=
  sorry

/-- Number of faces in the polyhedron -/
def numFaces (poly : RegularPolyhedron) : Nat :=
  sorry

/-- Theorem: For any regular polyhedron, there exists a point O such that
    1) The distance from O to all vertices is constant
    2) The distance from O to all faces is constant -/
theorem regular_polyhedron_spheres (poly : RegularPolyhedron) :
  ∃ (O : Point3D),
    (∀ (i j : Nat), i < numVertices poly → j < numVertices poly →
      distance O (getVertex poly i) = distance O (getVertex poly j)) ∧
    (∀ (i j : Nat), i < numFaces poly → j < numFaces poly →
      distanceToFace O poly i = distanceToFace O poly j) :=
by sorry

end NUMINAMATH_CALUDE_regular_polyhedron_spheres_l690_69008


namespace NUMINAMATH_CALUDE_five_by_five_grid_properties_l690_69038

/-- Represents a square grid --/
structure Grid :=
  (size : ℕ)

/-- Counts the number of squares in a grid --/
def count_squares (g : Grid) : ℕ :=
  sorry

/-- Counts the number of pairs of parallel lines in a grid --/
def count_parallel_pairs (g : Grid) : ℕ :=
  sorry

/-- Counts the number of rectangles in a grid --/
def count_rectangles (g : Grid) : ℕ :=
  sorry

/-- Theorem stating the properties of a 5x5 grid --/
theorem five_by_five_grid_properties :
  let g : Grid := ⟨5⟩
  count_squares g = 55 ∧
  count_parallel_pairs g = 30 ∧
  count_rectangles g = 225 :=
by sorry

end NUMINAMATH_CALUDE_five_by_five_grid_properties_l690_69038


namespace NUMINAMATH_CALUDE_function_existence_l690_69088

theorem function_existence : ∃ f : ℤ → ℤ, ∀ a b : ℤ, f (a + b) + f (a * b - 1) = f a * f b + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_existence_l690_69088


namespace NUMINAMATH_CALUDE_tiffany_lives_l690_69013

theorem tiffany_lives (x : ℕ) : 
  (x - 14 + 27 = 56) → x = 43 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_lives_l690_69013


namespace NUMINAMATH_CALUDE_linear_composition_solution_l690_69099

/-- A linear function f that satisfies f[f(x)] = 4x - 1 -/
def LinearComposition (f : ℝ → ℝ) : Prop :=
  (∃ k b : ℝ, ∀ x, f x = k * x + b) ∧ 
  (∀ x, f (f x) = 4 * x - 1)

/-- The theorem stating that a linear function satisfying the composition condition
    must be one of two specific linear functions -/
theorem linear_composition_solution (f : ℝ → ℝ) (h : LinearComposition f) :
  (∀ x, f x = 2 * x - 1/3) ∨ (∀ x, f x = -2 * x + 1) :=
sorry

end NUMINAMATH_CALUDE_linear_composition_solution_l690_69099


namespace NUMINAMATH_CALUDE_exponential_function_property_l690_69023

theorem exponential_function_property (a : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = a^x) →
  (a > 0) →
  (abs (f 2 - f 1) = a / 2) →
  (a = 1/2 ∨ a = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_exponential_function_property_l690_69023


namespace NUMINAMATH_CALUDE_smallest_angle_satisfies_equation_l690_69025

/-- The smallest positive angle (in degrees) that satisfies the given equation -/
noncomputable def smallest_angle : ℝ :=
  (1 / 4) * Real.arcsin (2 / 9) * (180 / Real.pi)

/-- The equation that the angle must satisfy -/
def equation (x : ℝ) : Prop :=
  9 * Real.sin x * (Real.cos x)^7 - 9 * (Real.sin x)^7 * Real.cos x = 1

theorem smallest_angle_satisfies_equation :
  equation (smallest_angle * (Real.pi / 180)) ∧
  ∀ y, 0 < y ∧ y < smallest_angle → ¬equation (y * (Real.pi / 180)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_satisfies_equation_l690_69025


namespace NUMINAMATH_CALUDE_fourth_buoy_distance_l690_69029

/-- Given buoys placed at even intervals in the ocean, with the third buoy 72 meters from the beach,
    this theorem proves that the fourth buoy is 108 meters from the beach. -/
theorem fourth_buoy_distance (interval : ℝ) (h1 : interval > 0) (h2 : 3 * interval = 72) :
  4 * interval = 108 := by
  sorry

end NUMINAMATH_CALUDE_fourth_buoy_distance_l690_69029


namespace NUMINAMATH_CALUDE_count_m_gons_correct_l690_69007

/-- Given integers m and n where 4 < m < n, and a regular polygon with 2n+1 sides,
    this function computes the number of convex m-gons with vertices from the polygon's vertices
    and exactly two acute interior angles. -/
def count_m_gons (m n : ℕ) : ℕ :=
  (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1))

/-- Theorem stating that count_m_gons correctly computes the number of m-gons
    satisfying the given conditions. -/
theorem count_m_gons_correct (m n : ℕ) (h1 : 4 < m) (h2 : m < n) :
  count_m_gons m n = (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1)) :=
by sorry

end NUMINAMATH_CALUDE_count_m_gons_correct_l690_69007


namespace NUMINAMATH_CALUDE_max_whole_nine_one_number_l690_69059

def is_whole_nine_one_number (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 4 ∧
  1 ≤ b ∧ b ≤ 9 ∧
  1 ≤ c ∧ c ≤ 9 ∧
  1 ≤ d ∧ d ≤ 9 ∧
  a + c = 9 ∧
  b - d = 1 ∧
  ∃ k : ℕ, k * (2 * b + c) = 4 * a + 2 * d

def M (a b c d : ℕ) : ℕ :=
  2000 * a + 100 * b + 10 * c + d

theorem max_whole_nine_one_number :
  ∀ a b c d : ℕ,
    is_whole_nine_one_number a b c d →
    M a b c d ≤ 7524 :=
  sorry

end NUMINAMATH_CALUDE_max_whole_nine_one_number_l690_69059


namespace NUMINAMATH_CALUDE_lemons_for_combined_beverages_l690_69092

/-- The number of lemons needed for a given amount of lemonade and limeade -/
def lemons_needed (lemonade_gallons : ℚ) (limeade_gallons : ℚ) : ℚ :=
  let lemons_per_gallon_lemonade : ℚ := 36 / 48
  let lemons_per_gallon_limeade : ℚ := 2 * lemons_per_gallon_lemonade
  lemonade_gallons * lemons_per_gallon_lemonade + limeade_gallons * lemons_per_gallon_limeade

/-- Theorem stating the number of lemons needed for 18 gallons of combined lemonade and limeade -/
theorem lemons_for_combined_beverages :
  lemons_needed 9 9 = 81/4 := by
  sorry

#eval lemons_needed 9 9

end NUMINAMATH_CALUDE_lemons_for_combined_beverages_l690_69092


namespace NUMINAMATH_CALUDE_milk_production_l690_69043

theorem milk_production (y : ℝ) (h : y > 0) : 
  let initial_production := (y + 2) / (y * (y + 3))
  let new_cows := y + 4
  let new_milk := y + 6
  (new_milk / (new_cows * initial_production)) = (y * (y + 3) * (y + 6)) / ((y + 2) * (y + 4)) :=
by sorry

end NUMINAMATH_CALUDE_milk_production_l690_69043


namespace NUMINAMATH_CALUDE_circle_equation_through_points_l690_69012

theorem circle_equation_through_points : 
  let circle_eq := (fun (x y : ℝ) => x^2 + y^2 - 4*x - 6*y)
  (circle_eq 0 0 = 0) ∧ 
  (circle_eq 4 0 = 0) ∧ 
  (circle_eq (-1) 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_through_points_l690_69012


namespace NUMINAMATH_CALUDE_calculate_markup_l690_69096

/-- Calculates the markup for an article given its purchase price, overhead percentage, and desired net profit. -/
theorem calculate_markup (purchase_price overhead_percent net_profit : ℚ) : 
  purchase_price = 48 →
  overhead_percent = 15 / 100 →
  net_profit = 12 →
  purchase_price + overhead_percent * purchase_price + net_profit - purchase_price = 19.2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_markup_l690_69096


namespace NUMINAMATH_CALUDE_fraction_simplification_l690_69095

theorem fraction_simplification : (5 * 8) / 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l690_69095


namespace NUMINAMATH_CALUDE_twelve_sixteen_twenty_pythagorean_triple_l690_69093

/-- A Pythagorean triple is a set of three positive integers (a, b, c) that satisfy a² + b² = c² -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

/-- The set {12, 16, 20} is a Pythagorean triple -/
theorem twelve_sixteen_twenty_pythagorean_triple :
  isPythagoreanTriple 12 16 20 := by
  sorry

end NUMINAMATH_CALUDE_twelve_sixteen_twenty_pythagorean_triple_l690_69093


namespace NUMINAMATH_CALUDE_square_roots_problem_l690_69022

theorem square_roots_problem (a : ℝ) : 
  (∃ x : ℝ, (3 * a + 2)^2 = x ∧ (a + 14)^2 = x) → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l690_69022


namespace NUMINAMATH_CALUDE_shepherd_problem_l690_69055

def checkpoint (n : ℕ) : ℕ := n / 2 + 1

def process (initial : ℕ) (checkpoints : ℕ) : ℕ :=
  match checkpoints with
  | 0 => initial
  | n + 1 => checkpoint (process initial n)

theorem shepherd_problem (initial : ℕ) (checkpoints : ℕ) :
  initial = 254 ∧ checkpoints = 6 → process initial checkpoints = 2 := by
  sorry

end NUMINAMATH_CALUDE_shepherd_problem_l690_69055


namespace NUMINAMATH_CALUDE_star_value_l690_69039

/-- Operation star defined as a * b = 3a - b^3 -/
def star (a b : ℝ) : ℝ := 3 * a - b^3

/-- Theorem: If a * 3 = 63, then a = 30 -/
theorem star_value (a : ℝ) (h : star a 3 = 63) : a = 30 := by
  sorry

end NUMINAMATH_CALUDE_star_value_l690_69039


namespace NUMINAMATH_CALUDE_odd_function_value_l690_69062

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

theorem odd_function_value (f : ℝ → ℝ) (h1 : is_odd f) (h2 : ∀ x < 0, f x = x^2 + 3*x) :
  f 2 = 2 := by sorry

end NUMINAMATH_CALUDE_odd_function_value_l690_69062


namespace NUMINAMATH_CALUDE_scavenger_hunt_ratio_l690_69036

theorem scavenger_hunt_ratio : 
  ∀ (lewis samantha tanya : ℕ),
  lewis = samantha + 4 →
  ∃ k : ℕ, samantha = k * tanya →
  tanya = 4 →
  lewis = 20 →
  samantha / tanya = 4 := by
sorry

end NUMINAMATH_CALUDE_scavenger_hunt_ratio_l690_69036


namespace NUMINAMATH_CALUDE_hay_in_final_mixture_l690_69071

/-- Represents the composition of a feed mixture -/
structure FeedMixture where
  oats : ℝ  -- Percentage of oats
  corn : ℝ  -- Percentage of corn
  hay : ℝ   -- Percentage of hay
  mass : ℝ  -- Mass of the mixture in kg

/-- Theorem stating the amount of hay in the final mixture -/
theorem hay_in_final_mixture
  (stepan : FeedMixture)
  (pavel : FeedMixture)
  (final : FeedMixture)
  (h1 : stepan.hay = 40)
  (h2 : pavel.oats = 26)
  (h3 : stepan.corn = pavel.corn)
  (h4 : stepan.mass = 150)
  (h5 : pavel.mass = 250)
  (h6 : final.corn = 30)
  (h7 : final.mass = stepan.mass + pavel.mass)
  (h8 : final.corn * final.mass = stepan.corn * stepan.mass + pavel.corn * pavel.mass) :
  final.hay * final.mass = 170 := by
  sorry

end NUMINAMATH_CALUDE_hay_in_final_mixture_l690_69071


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l690_69091

theorem boys_to_girls_ratio :
  ∀ (boys girls : ℕ),
    boys = 80 →
    girls = boys + 128 →
    (boys : ℚ) / girls = 5 / 13 :=
by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l690_69091


namespace NUMINAMATH_CALUDE_square_of_difference_l690_69090

theorem square_of_difference (y : ℝ) (h : y^2 ≥ 49) :
  (7 - Real.sqrt (y^2 - 49))^2 = y^2 - 14 * Real.sqrt (y^2 - 49) := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_l690_69090


namespace NUMINAMATH_CALUDE_tadpoles_kept_l690_69018

theorem tadpoles_kept (total : ℕ) (release_percentage : ℚ) (kept : ℕ) : 
  total = 180 → 
  release_percentage = 75 / 100 → 
  kept = total - (release_percentage * total).floor → 
  kept = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_tadpoles_kept_l690_69018


namespace NUMINAMATH_CALUDE_parallel_planes_equidistant_points_l690_69044

-- Define the concept of a plane
def Plane : Type := sorry

-- Define the concept of a point
def Point : Type := sorry

-- Define what it means for a point to be in a plane
def in_plane (p : Point) (α : Plane) : Prop := sorry

-- Define what it means for three points to be non-collinear
def non_collinear (p q r : Point) : Prop := sorry

-- Define what it means for a point to be equidistant from a plane
def equidistant_from_plane (p : Point) (β : Plane) (d : ℝ) : Prop := sorry

-- Define what it means for two planes to be parallel
def parallel_planes (α β : Plane) : Prop := sorry

-- State the theorem
theorem parallel_planes_equidistant_points (α β : Plane) :
  (∃ (p q r : Point) (d : ℝ), 
    in_plane p α ∧ in_plane q α ∧ in_plane r α ∧
    non_collinear p q r ∧
    equidistant_from_plane p β d ∧
    equidistant_from_plane q β d ∧
    equidistant_from_plane r β d) →
  parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_equidistant_points_l690_69044


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l690_69072

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x | x^2 - 5*x + 4 < 0}

theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l690_69072


namespace NUMINAMATH_CALUDE_probability_all_8_cards_l690_69098

/-- Represents a player in the card game --/
structure Player where
  cards : ℕ

/-- Represents the state of the game --/
structure GameState where
  players : Fin 6 → Player
  cardsDealt : ℕ

/-- The dealing process for a single card --/
def dealCard (state : GameState) : GameState :=
  sorry

/-- The final state after dealing all cards --/
def finalState : GameState :=
  sorry

/-- Checks if all players have exactly 8 cards --/
def allPlayersHave8Cards (state : GameState) : Prop :=
  ∀ i : Fin 6, (state.players i).cards = 8

/-- The probability of all players having 8 cards after dealing --/
def probabilityAllHave8Cards : ℚ :=
  sorry

/-- Theorem stating the probability of all players having 8 cards is 5/6 --/
theorem probability_all_8_cards : probabilityAllHave8Cards = 5/6 :=
  sorry

end NUMINAMATH_CALUDE_probability_all_8_cards_l690_69098


namespace NUMINAMATH_CALUDE_export_probabilities_l690_69035

/-- The number of inspections required for each batch -/
def num_inspections : ℕ := 5

/-- The probability of failing any given inspection -/
def fail_prob : ℝ := 0.2

/-- The probability of passing any given inspection -/
def pass_prob : ℝ := 1 - fail_prob

/-- The probability that a batch cannot be exported -/
def cannot_export_prob : ℝ := 1 - (pass_prob ^ num_inspections + num_inspections * fail_prob * pass_prob ^ (num_inspections - 1))

/-- The probability that all five inspections must be completed -/
def all_inspections_prob : ℝ := (num_inspections - 1) * fail_prob * pass_prob ^ (num_inspections - 2)

theorem export_probabilities :
  (cannot_export_prob = 0.26) ∧ (all_inspections_prob = 0.41) := by
  sorry

end NUMINAMATH_CALUDE_export_probabilities_l690_69035


namespace NUMINAMATH_CALUDE_complex_equality_l690_69024

theorem complex_equality (z : ℂ) : Complex.abs (z + 2) = Complex.abs (z - 3) → z = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l690_69024


namespace NUMINAMATH_CALUDE_smallest_zero_difference_l690_69079

def u (n : ℕ) : ℤ := n^3 - n

def finite_difference (f : ℕ → ℤ) (k : ℕ) : ℕ → ℤ :=
  match k with
  | 0 => f
  | k+1 => λ n => finite_difference f k (n+1) - finite_difference f k n

theorem smallest_zero_difference :
  (∃ k : ℕ, ∀ n : ℕ, finite_difference u k n = 0) ∧
  (∀ k : ℕ, k < 4 → ∃ n : ℕ, finite_difference u k n ≠ 0) ∧
  (∀ n : ℕ, finite_difference u 4 n = 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_zero_difference_l690_69079


namespace NUMINAMATH_CALUDE_equality_implies_product_equality_l690_69076

theorem equality_implies_product_equality (a b c : ℝ) : a = b → a * c = b * c := by
  sorry

end NUMINAMATH_CALUDE_equality_implies_product_equality_l690_69076


namespace NUMINAMATH_CALUDE_y2k_game_second_player_strategy_l690_69042

/-- Represents a player in the Y2K Game -/
inductive Player : Type
  | First : Player
  | Second : Player

/-- Represents a letter that can be placed on the board -/
inductive Letter : Type
  | S : Letter
  | O : Letter

/-- Represents the state of a square on the board -/
inductive Square : Type
  | Empty : Square
  | Filled : Letter → Square

/-- Represents the game board -/
def Board : Type := Fin 2000 → Square

/-- Represents a move in the game -/
structure Move where
  position : Fin 2000
  letter : Letter

/-- Represents the game state -/
structure GameState where
  board : Board
  currentPlayer : Player

/-- Represents a strategy for a player -/
def Strategy : Type := GameState → Move

/-- Checks if a player has won the game -/
def hasWon (board : Board) (player : Player) : Prop := sorry

/-- Checks if the game is a draw -/
def isDraw (board : Board) : Prop := sorry

/-- The Y2K Game theorem -/
theorem y2k_game_second_player_strategy :
  ∃ (strategy : Strategy),
    ∀ (initialState : GameState),
      initialState.currentPlayer = Player.Second →
        (∃ (finalState : GameState),
          (hasWon finalState.board Player.Second ∨ isDraw finalState.board)) :=
sorry

end NUMINAMATH_CALUDE_y2k_game_second_player_strategy_l690_69042


namespace NUMINAMATH_CALUDE_triangle_minimum_product_l690_69017

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if 2c cos B = 2a + b and the area of the triangle is (√3/12)c, then ab ≥ 1/3 -/
theorem triangle_minimum_product (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  2 * c * Real.cos B = 2 * a + b →
  (1 / 2) * a * b * Real.sin C = (Real.sqrt 3 / 12) * c →
  a * b ≥ 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_minimum_product_l690_69017


namespace NUMINAMATH_CALUDE_f_properties_l690_69020

-- Define the function f(x) = x ln|x|
noncomputable def f (x : ℝ) : ℝ := x * Real.log (abs x)

-- Define the function g(x) = f(x) - m
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f x - m

theorem f_properties :
  (∀ x y, x < y ∧ x < -1/Real.exp 1 ∧ y < -1/Real.exp 1 → f x < f y) ∧
  (∀ m : ℝ, ∃ n : ℕ, n ≤ 3 ∧ (∃ s : Finset ℝ, s.card = n ∧ ∀ x ∈ s, g m x = 0) ∧
    ∀ s : Finset ℝ, (∀ x ∈ s, g m x = 0) → s.card ≤ n) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l690_69020


namespace NUMINAMATH_CALUDE_right_handed_players_count_l690_69074

theorem right_handed_players_count (total_players : ℕ) (throwers : ℕ) : 
  total_players = 70 →
  throwers = 31 →
  (total_players - throwers) % 3 = 0 →
  57 = throwers + (total_players - throwers) * 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_handed_players_count_l690_69074


namespace NUMINAMATH_CALUDE_fraction_inequality_l690_69040

theorem fraction_inequality (x : ℝ) : 
  -1 ≤ x ∧ x ≤ 3 →
  (4 * x + 3 > 2 * (8 - 3 * x) ↔ 13 / 10 < x ∧ x ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l690_69040


namespace NUMINAMATH_CALUDE_folding_problem_l690_69061

/-- Represents the folding rate for each type of clothing --/
structure FoldingRate where
  shirts : ℕ
  pants : ℕ
  shorts : ℕ

/-- Represents the number of items for each type of clothing --/
structure ClothingItems where
  shirts : ℕ
  pants : ℕ
  shorts : ℕ

/-- Calculates the remaining items to be folded given the initial conditions --/
def remainingItems (initialItems : ClothingItems) (rate : FoldingRate) (totalTime : ℕ) 
    (shirtFoldTime : ℕ) (pantFoldTime : ℕ) (shirtBreakTime : ℕ) (pantBreakTime : ℕ) : ClothingItems :=
  sorry

/-- The main theorem to be proved --/
theorem folding_problem (initialItems : ClothingItems) (rate : FoldingRate) (totalTime : ℕ) 
    (shirtFoldTime : ℕ) (pantFoldTime : ℕ) (shirtBreakTime : ℕ) (pantBreakTime : ℕ) :
    initialItems = ClothingItems.mk 30 15 20 ∧ 
    rate = FoldingRate.mk 12 8 10 ∧
    totalTime = 120 ∧
    shirtFoldTime = 45 ∧
    pantFoldTime = 30 ∧
    shirtBreakTime = 15 ∧
    pantBreakTime = 10 →
    remainingItems initialItems rate totalTime shirtFoldTime pantFoldTime shirtBreakTime pantBreakTime = 
    ClothingItems.mk 21 11 17 :=
  sorry

end NUMINAMATH_CALUDE_folding_problem_l690_69061


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l690_69047

/-- An ellipse with axes parallel to the coordinate axes -/
structure ParallelAxisEllipse where
  /-- The point where the ellipse is tangent to the x-axis -/
  x_tangent : ℝ × ℝ
  /-- The point where the ellipse is tangent to the y-axis -/
  y_tangent : ℝ × ℝ

/-- The distance between the foci of the ellipse -/
def foci_distance (e : ParallelAxisEllipse) : ℝ :=
  sorry

theorem ellipse_foci_distance :
  let e : ParallelAxisEllipse := ⟨(6, 0), (0, 2)⟩
  foci_distance e = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l690_69047


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l690_69030

/-- 
An isosceles triangle with congruent sides of length 7 cm and perimeter 23 cm 
has a base of length 9 cm.
-/
theorem isosceles_triangle_base_length : 
  ∀ (base : ℝ), 
  base > 0 → 
  7 + 7 + base = 23 → 
  base = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l690_69030


namespace NUMINAMATH_CALUDE_single_point_conic_section_l690_69000

theorem single_point_conic_section (d : ℝ) : 
  (∃! p : ℝ × ℝ, 3 * p.1^2 + p.2^2 + 6 * p.1 - 6 * p.2 + d = 0) → d = 12 := by
  sorry

end NUMINAMATH_CALUDE_single_point_conic_section_l690_69000


namespace NUMINAMATH_CALUDE_pizza_problem_solution_l690_69011

/-- Represents the number of slices in a pizza --/
structure PizzaSlices where
  small : ℕ
  large : ℕ

/-- Represents the number of pizzas purchased --/
structure PizzasPurchased where
  small : ℕ
  large : ℕ

/-- Represents the number of slices eaten by each person --/
structure SlicesEaten where
  george : ℕ
  bob : ℕ
  susie : ℕ
  bill : ℕ
  fred : ℕ
  mark : ℕ

def pizza_problem (slices : PizzaSlices) (purchased : PizzasPurchased) (eaten : SlicesEaten) (leftover : ℕ) : Prop :=
  slices.small = 4 ∧
  slices.large = 8 ∧
  purchased.large = 2 ∧
  eaten.george = 3 ∧
  eaten.bob = eaten.george + 1 ∧
  eaten.susie = eaten.bob / 2 ∧
  eaten.bill = 3 ∧
  eaten.fred = 3 ∧
  eaten.mark = 3 ∧
  leftover = 10 ∧
  purchased.small * slices.small + purchased.large * slices.large =
    eaten.george + eaten.bob + eaten.susie + eaten.bill + eaten.fred + eaten.mark + leftover

theorem pizza_problem_solution 
  (slices : PizzaSlices) (purchased : PizzasPurchased) (eaten : SlicesEaten) (leftover : ℕ) :
  pizza_problem slices purchased eaten leftover → purchased.small = 3 := by
  sorry

end NUMINAMATH_CALUDE_pizza_problem_solution_l690_69011


namespace NUMINAMATH_CALUDE_polynomial_not_equal_77_l690_69010

theorem polynomial_not_equal_77 (x y : ℤ) : 
  x^5 - 4*x^4*y - 5*y^2*x^3 + 20*y^3*x^2 + 4*y^4*x - 16*y^5 ≠ 77 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_not_equal_77_l690_69010


namespace NUMINAMATH_CALUDE_chair_circumference_l690_69078

def parallelogram_circumference (side1 : ℝ) (side2 : ℝ) : ℝ :=
  2 * (side1 + side2)

theorem chair_circumference :
  let side1 := 18
  let side2 := 12
  parallelogram_circumference side1 side2 = 60 := by
sorry

end NUMINAMATH_CALUDE_chair_circumference_l690_69078


namespace NUMINAMATH_CALUDE_intersecting_circles_theorem_l690_69016

/-- Two circles intersecting at two distinct points theorem -/
theorem intersecting_circles_theorem 
  (r a b x₁ y₁ x₂ y₂ : ℝ) 
  (hr : r > 0)
  (hab : a ≠ 0 ∨ b ≠ 0)
  (hC₁_A : x₁^2 + y₁^2 = r^2)
  (hC₂_A : (x₁ + a)^2 + (y₁ + b)^2 = r^2)
  (hC₁_B : x₂^2 + y₂^2 = r^2)
  (hC₂_B : (x₂ + a)^2 + (y₂ + b)^2 = r^2)
  (hAB_distinct : (x₁, y₁) ≠ (x₂, y₂)) :
  (2*a*x₁ + 2*b*y₁ + a^2 + b^2 = 0) ∧ 
  (a*(x₁ - x₂) + b*(y₁ - y₂) = 0) ∧ 
  (x₁ + x₂ = -a ∧ y₁ + y₂ = -b) := by
  sorry

end NUMINAMATH_CALUDE_intersecting_circles_theorem_l690_69016


namespace NUMINAMATH_CALUDE_percentage_of_amount_l690_69033

theorem percentage_of_amount (amount : ℝ) :
  (25 : ℝ) / 100 * amount = 150 → amount = 600 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_amount_l690_69033


namespace NUMINAMATH_CALUDE_batsman_80_run_innings_l690_69083

/-- Represents a batsman's cricket statistics -/
structure BatsmanStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after adding a score -/
def newAverage (stats : BatsmanStats) (score : ℕ) : ℚ :=
  (stats.totalRuns + score) / (stats.innings + 1)

theorem batsman_80_run_innings :
  ∀ (stats : BatsmanStats),
    stats.average = 46 →
    newAverage stats 80 = 48 →
    stats.innings = 16 :=
by sorry

end NUMINAMATH_CALUDE_batsman_80_run_innings_l690_69083


namespace NUMINAMATH_CALUDE_complex_product_theorem_l690_69057

theorem complex_product_theorem (z₁ z₂ : ℂ) :
  z₁ = 4 + I → z₂ = 1 - 2*I → z₁ * z₂ = 6 - 7*I := by
  sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l690_69057
