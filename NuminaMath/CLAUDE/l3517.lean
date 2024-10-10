import Mathlib

namespace hyperbola_eccentricity_l3517_351771

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (P F₁ F₂ : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  P.1 ≥ 0 →
  P.2 ≥ 0 →
  P.1^2 / a^2 - P.2^2 / b^2 = 1 →
  P.1^2 + P.2^2 = a^2 + b^2 →
  F₁.1 < 0 →
  F₂.1 > 0 →
  F₁.2 = 0 →
  F₂.2 = 0 →
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) = 3 * Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) →
  Real.sqrt ((F₂.1 - F₁.1)^2 / (2*a)^2) = Real.sqrt 10 / 2 :=
by sorry

end hyperbola_eccentricity_l3517_351771


namespace pizza_piece_cost_l3517_351727

/-- Given that 4 pizzas cost $80 in total, and each pizza is cut into 5 pieces,
    prove that the cost of each piece of pizza is $4. -/
theorem pizza_piece_cost : 
  (total_cost : ℝ) →
  (num_pizzas : ℕ) →
  (pieces_per_pizza : ℕ) →
  total_cost = 80 →
  num_pizzas = 4 →
  pieces_per_pizza = 5 →
  (total_cost / (num_pizzas * pieces_per_pizza : ℝ)) = 4 :=
by sorry

end pizza_piece_cost_l3517_351727


namespace pool_water_rates_l3517_351746

/-- Represents the water delivery rates for two pools -/
structure PoolRates :=
  (first : ℝ)
  (second : ℝ)

/-- Proves that the water delivery rates for two pools satisfy the given conditions -/
theorem pool_water_rates :
  ∃ (rates : PoolRates),
    rates.first = 90 ∧
    rates.second = 60 ∧
    rates.first = rates.second + 30 ∧
    ∃ (t : ℝ),
      (rates.first * t + rates.second * t = 2 * rates.first * t) ∧
      (rates.first * (t + 8/3) = rates.first * t) ∧
      (rates.second * (t + 10/3) = rates.second * t) :=
by sorry

end pool_water_rates_l3517_351746


namespace impossible_cover_l3517_351702

/-- Represents an L-trimino piece -/
structure LTrimino where
  covers : Nat
  covers_eq : covers = 3

/-- Represents a 3x5 board with special squares -/
structure Board where
  total_squares : Nat
  total_squares_eq : total_squares = 15
  special_squares : Nat
  special_squares_eq : special_squares = 6

/-- States that it's impossible to cover the board with L-triminos -/
theorem impossible_cover (b : Board) (l : LTrimino) : 
  ¬∃ (n : Nat), n * l.covers = b.total_squares ∧ n ≥ b.special_squares :=
sorry

end impossible_cover_l3517_351702


namespace bus_passengers_specific_case_l3517_351730

def passengers (m n : ℕ) : ℕ := m - 12 + n

theorem bus_passengers (m n : ℕ) (h : m ≥ 12) : 
  passengers m n = m - 12 + n :=
by sorry

theorem specific_case : passengers 26 6 = 20 :=
by sorry

end bus_passengers_specific_case_l3517_351730


namespace cubic_foot_to_cubic_inches_l3517_351729

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℝ := 12

/-- Theorem stating that one cubic foot is equal to 1728 cubic inches -/
theorem cubic_foot_to_cubic_inches : (1 : ℝ)^3 * feet_to_inches^3 = 1728 := by
  sorry

end cubic_foot_to_cubic_inches_l3517_351729


namespace at_least_one_passes_l3517_351747

/-- The probability that at least one of three independent events occurs, given their individual probabilities -/
theorem at_least_one_passes (pA pB pC : ℝ) (hA : pA = 0.8) (hB : pB = 0.6) (hC : pC = 0.5) :
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 0.96 := by
  sorry

end at_least_one_passes_l3517_351747


namespace notepad_lasts_four_days_l3517_351780

/-- Represents the number of pieces of letter-size paper used --/
def letter_size_papers : ℕ := 5

/-- Represents the number of times each paper is folded --/
def folds : ℕ := 3

/-- Represents the number of notes written per day --/
def notes_per_day : ℕ := 10

/-- Calculates the number of note-size papers produced from one letter-size paper --/
def note_papers_per_letter_paper : ℕ := 2^folds

/-- Calculates the total number of note-size papers in a notepad --/
def total_note_papers : ℕ := letter_size_papers * note_papers_per_letter_paper

/-- Represents how long a notepad lasts in days --/
def notepad_duration : ℕ := total_note_papers / notes_per_day

theorem notepad_lasts_four_days : notepad_duration = 4 := by
  sorry

end notepad_lasts_four_days_l3517_351780


namespace oxford_high_school_principals_l3517_351782

/-- Oxford High School Problem -/
theorem oxford_high_school_principals 
  (total_people : ℕ) 
  (teachers : ℕ) 
  (classes : ℕ) 
  (students_per_class : ℕ) 
  (h1 : total_people = 349) 
  (h2 : teachers = 48) 
  (h3 : classes = 15) 
  (h4 : students_per_class = 20) :
  total_people - (teachers + classes * students_per_class) = 1 :=
by sorry

end oxford_high_school_principals_l3517_351782


namespace grapes_and_orange_cost_l3517_351786

/-- Represents the prices of items in John's purchase -/
structure Prices where
  peanuts : ℝ
  grapes : ℝ
  orange : ℝ
  chocolates : ℝ

/-- The conditions of John's purchase -/
def purchase_conditions (p : Prices) : Prop :=
  p.peanuts + p.grapes + p.orange + p.chocolates = 25 ∧
  p.chocolates = 2 * p.peanuts ∧
  p.orange = p.peanuts - p.grapes

/-- The theorem stating the cost of grapes and orange -/
theorem grapes_and_orange_cost (p : Prices) 
  (h : purchase_conditions p) : p.grapes + p.orange = 6.25 := by
  sorry

#check grapes_and_orange_cost

end grapes_and_orange_cost_l3517_351786


namespace inequality_theorem_l3517_351711

theorem inequality_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a + b + c ≥ a * b * c) :
  (2 / a + 3 / b + 6 / c ≥ 6 ∧ 2 / b + 3 / c + 6 / a ≥ 6) ∨
  (2 / a + 3 / b + 6 / c ≥ 6 ∧ 2 / c + 3 / a + 6 / b ≥ 6) ∨
  (2 / b + 3 / c + 6 / a ≥ 6 ∧ 2 / c + 3 / a + 6 / b ≥ 6) :=
by sorry

end inequality_theorem_l3517_351711


namespace product_xyz_is_one_ninth_l3517_351784

theorem product_xyz_is_one_ninth 
  (x y z : ℝ) 
  (h1 : x + 1/y = 3) 
  (h2 : y + 1/z = 5) : 
  x * y * z = 1/9 := by
sorry

end product_xyz_is_one_ninth_l3517_351784


namespace specific_log_stack_count_l3517_351760

/-- Represents a stack of logs -/
structure LogStack where
  bottomCount : ℕ
  topCount : ℕ
  decreaseRate : ℕ

/-- Calculates the total number of logs in the stack -/
def totalLogs (stack : LogStack) : ℕ :=
  let rowCount := stack.bottomCount - stack.topCount + 1
  let avgRowCount := (stack.bottomCount + stack.topCount) / 2
  rowCount * avgRowCount

/-- Theorem stating that the specific log stack has 110 logs -/
theorem specific_log_stack_count :
  ∃ (stack : LogStack),
    stack.bottomCount = 15 ∧
    stack.topCount = 5 ∧
    stack.decreaseRate = 1 ∧
    totalLogs stack = 110 := by
  sorry

end specific_log_stack_count_l3517_351760


namespace oreo_distribution_l3517_351799

/-- The number of Oreos Jordan has -/
def jordans_oreos : ℕ := 11

/-- The number of Oreos James has -/
def james_oreos (j : ℕ) : ℕ := 2 * j + 3

/-- The total number of Oreos -/
def total_oreos : ℕ := 36

theorem oreo_distribution : 
  james_oreos jordans_oreos + jordans_oreos = total_oreos :=
by sorry

end oreo_distribution_l3517_351799


namespace ceiling_minus_x_value_l3517_351707

theorem ceiling_minus_x_value (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 1) :
  ∃ δ : ℝ, 0 < δ ∧ δ < 1 ∧ ⌈x⌉ - x = 1 - δ := by
  sorry

end ceiling_minus_x_value_l3517_351707


namespace alien_mineral_conversion_l3517_351728

/-- Converts a three-digit number from base 7 to base 10 -/
def base7ToBase10 (a b c : ℕ) : ℕ :=
  a * 7^2 + b * 7^1 + c * 7^0

/-- The base 7 number 365₇ is equal to 194 in base 10 -/
theorem alien_mineral_conversion :
  base7ToBase10 3 6 5 = 194 := by
  sorry

end alien_mineral_conversion_l3517_351728


namespace cylinder_height_comparison_l3517_351775

theorem cylinder_height_comparison (r₁ r₂ h₁ h₂ : ℝ) :
  r₁ > 0 ∧ r₂ > 0 ∧ h₁ > 0 ∧ h₂ > 0 →
  r₂ = 1.1 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.21 * h₂ :=
by sorry

#check cylinder_height_comparison

end cylinder_height_comparison_l3517_351775


namespace cubic_root_sum_cubes_l3517_351716

theorem cubic_root_sum_cubes (a b c : ℂ) : 
  (5 * a^3 + 2014 * a + 4027 = 0) →
  (5 * b^3 + 2014 * b + 4027 = 0) →
  (5 * c^3 + 2014 * c + 4027 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 2416.2 := by
  sorry

end cubic_root_sum_cubes_l3517_351716


namespace linear_function_proof_l3517_351781

/-- A linear function passing through points (1, 3) and (-2, 12) -/
def f (x : ℝ) : ℝ := -3 * x + 6

theorem linear_function_proof :
  (f 1 = 3 ∧ f (-2) = 12) ∧
  (∀ a : ℝ, f (2 * a) ≠ -6 * a + 8) := by
  sorry

end linear_function_proof_l3517_351781


namespace ratio_problem_l3517_351748

theorem ratio_problem (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + 3 * y + 1) = 4 / 5) :
  x / y = 22 / 7 := by
  sorry

end ratio_problem_l3517_351748


namespace sqrt_difference_equality_l3517_351764

theorem sqrt_difference_equality : 2 * (Real.sqrt (49 + 81) - Real.sqrt (36 - 25)) = 2 * (Real.sqrt 130 - Real.sqrt 11) := by
  sorry

end sqrt_difference_equality_l3517_351764


namespace cryptarithmetic_problem_l3517_351704

theorem cryptarithmetic_problem (A B C D : ℕ) : 
  (A + B + C = 11) →
  (B + A + D = 10) →
  (A + D = 4) →
  (A ≠ B) → (A ≠ C) → (A ≠ D) → (B ≠ C) → (B ≠ D) → (C ≠ D) →
  (A < 10) → (B < 10) → (C < 10) → (D < 10) →
  C = 4 := by
sorry

end cryptarithmetic_problem_l3517_351704


namespace simplify_trigonometric_expression_evaluate_trigonometric_fraction_l3517_351726

-- Part 1
theorem simplify_trigonometric_expression :
  (Real.sqrt (1 - 2 * Real.sin (20 * π / 180) * Real.cos (20 * π / 180))) /
  (Real.sin (160 * π / 180) - Real.sqrt (1 - Real.sin (20 * π / 180) ^ 2)) = -1 := by
  sorry

-- Part 2
theorem evaluate_trigonometric_fraction (α : Real) (h : Real.tan α = 1 / 3) :
  1 / (4 * Real.cos α ^ 2 - 6 * Real.sin α * Real.cos α) = 5 / 9 := by
  sorry

end simplify_trigonometric_expression_evaluate_trigonometric_fraction_l3517_351726


namespace coldness_probability_l3517_351741

def word1 := "CART"
def word2 := "BLEND"
def word3 := "SHOW"
def target_word := "COLDNESS"

def select_letters (word : String) (n : Nat) : Nat := Nat.choose word.length n

theorem coldness_probability :
  let p1 := (1 : ℚ) / select_letters word1 2
  let p2 := (1 : ℚ) / select_letters word2 4
  let p3 := (1 : ℚ) / 2
  p1 * p2 * p3 = 1 / 60 := by sorry

end coldness_probability_l3517_351741


namespace intersection_M_N_l3517_351765

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = 2 * x ∧ x > 0}
def N : Set ℝ := {x | ∃ y, y = Real.log (2 * x - x^2) ∧ x > 0 ∧ x < 2}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioo 1 2 := by
  sorry

end intersection_M_N_l3517_351765


namespace square_plus_one_greater_than_one_l3517_351724

theorem square_plus_one_greater_than_one (a : ℝ) (h : a ≠ 0) : a^2 + 1 > 1 := by
  sorry

end square_plus_one_greater_than_one_l3517_351724


namespace and_sufficient_not_necessary_for_or_l3517_351745

theorem and_sufficient_not_necessary_for_or :
  (∃ p q : Prop, (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)) :=
by sorry

end and_sufficient_not_necessary_for_or_l3517_351745


namespace simplify_square_roots_l3517_351788

theorem simplify_square_roots : 
  (Real.sqrt 392 / Real.sqrt 352) + (Real.sqrt 180 / Real.sqrt 120) = 
  (7 * Real.sqrt 6 + 6 * Real.sqrt 11) / (2 * Real.sqrt 66) := by
  sorry

end simplify_square_roots_l3517_351788


namespace minerals_found_today_l3517_351705

def minerals_yesterday (gemstones_yesterday : ℕ) : ℕ := 2 * gemstones_yesterday

theorem minerals_found_today 
  (gemstones_today minerals_today : ℕ) 
  (h1 : minerals_today = 48) 
  (h2 : gemstones_today = 21) : 
  minerals_today - minerals_yesterday gemstones_today = 6 := by
  sorry

end minerals_found_today_l3517_351705


namespace base6_addition_l3517_351740

/-- Converts a base 6 number represented as a list of digits to its decimal equivalent -/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 6 * acc + d) 0

/-- Converts a decimal number to its base 6 representation as a list of digits -/
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- The main theorem to prove -/
theorem base6_addition :
  base6ToDecimal [4, 5, 3, 5] + base6ToDecimal [2, 3, 2, 4, 3] =
  base6ToDecimal [3, 2, 2, 2, 2] := by
  sorry

end base6_addition_l3517_351740


namespace total_faces_is_198_l3517_351794

/-- The total number of faces on all dice and geometrical shapes -/
def total_faces : ℕ := sorry

/-- Number of six-sided dice -/
def six_sided_dice : ℕ := 4

/-- Number of eight-sided dice -/
def eight_sided_dice : ℕ := 5

/-- Number of twelve-sided dice -/
def twelve_sided_dice : ℕ := 3

/-- Number of twenty-sided dice -/
def twenty_sided_dice : ℕ := 2

/-- Number of cubes -/
def cubes : ℕ := 1

/-- Number of tetrahedrons -/
def tetrahedrons : ℕ := 3

/-- Number of icosahedrons -/
def icosahedrons : ℕ := 2

/-- Theorem stating that the total number of faces is 198 -/
theorem total_faces_is_198 : total_faces = 198 := by sorry

end total_faces_is_198_l3517_351794


namespace orchid_planting_problem_l3517_351768

/-- Calculates the number of orchid bushes to be planted -/
def orchids_to_plant (current : ℕ) (after : ℕ) : ℕ :=
  after - current

theorem orchid_planting_problem :
  let current_orchids : ℕ := 22
  let total_after_planting : ℕ := 35
  orchids_to_plant current_orchids total_after_planting = 13 := by
  sorry

end orchid_planting_problem_l3517_351768


namespace last_digit_of_sum_l3517_351733

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- The value of a three-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Swaps the last two digits of a three-digit number -/
def ThreeDigitNumber.swap_last_two (n : ThreeDigitNumber) : ThreeDigitNumber :=
  { hundreds := n.hundreds
  , tens := n.ones
  , ones := n.tens
  , is_valid := by sorry }

theorem last_digit_of_sum (n : ThreeDigitNumber) :
  (n.value + (n.swap_last_two).value ≥ 1000) →
  (n.value + (n.swap_last_two).value < 2000) →
  (n.value + (n.swap_last_two).value) / 10 = 195 →
  (n.value + (n.swap_last_two).value) % 10 = 4 := by
  sorry


end last_digit_of_sum_l3517_351733


namespace circle_center_sum_l3517_351791

/-- Given a circle with equation x^2 + y^2 = 6x + 4y + 4, prove that the sum of the coordinates of its center is 5. -/
theorem circle_center_sum (h k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 6*x + 4*y + 4 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 4)) → 
  h + k = 5 := by
  sorry

end circle_center_sum_l3517_351791


namespace additional_men_problem_l3517_351715

/-- Calculates the number of additional men given initial conditions and new duration -/
def additional_men (initial_men : ℕ) (initial_days : ℚ) (new_days : ℚ) : ℚ :=
  (initial_men * initial_days / new_days) - initial_men

theorem additional_men_problem :
  let initial_men : ℕ := 1000
  let initial_days : ℚ := 17
  let new_days : ℚ := 11.333333333333334
  additional_men initial_men initial_days new_days = 500 := by
sorry

end additional_men_problem_l3517_351715


namespace symmetric_points_sum_sum_of_symmetric_point_l3517_351739

/-- Given two points A and B in a 2D plane, where A is symmetric to B with respect to the origin,
    prove that the sum of B's coordinates equals the negative sum of A's coordinates. -/
theorem symmetric_points_sum (A B : ℝ × ℝ) (hSymmetric : B = (-A.1, -A.2)) :
  B.1 + B.2 = -(A.1 + A.2) := by
  sorry

/-- Prove that if point A(-2022, -1) is symmetric with respect to the origin O to point B(a, b),
    then a + b = 2023. -/
theorem sum_of_symmetric_point :
  ∃ (a b : ℝ), (a, b) = (-(-2022), -(-1)) → a + b = 2023 := by
  sorry

end symmetric_points_sum_sum_of_symmetric_point_l3517_351739


namespace binary_sequence_eventually_periodic_l3517_351792

/-- A sequence of 0s and 1s -/
def BinarySequence := ℕ → Fin 2

/-- A block of n consecutive terms in a sequence -/
def Block (s : BinarySequence) (n : ℕ) (start : ℕ) : Fin n → Fin 2 :=
  fun i => s (start + i)

/-- A sequence is eventually periodic if there exist positive integers p and N such that
    for all k ≥ N, s(k + p) = s(k) -/
def EventuallyPeriodic (s : BinarySequence) : Prop :=
  ∃ (p N : ℕ), p > 0 ∧ ∀ k ≥ N, s (k + p) = s k

/-- The main theorem: if a binary sequence contains only n different blocks of
    n consecutive terms, where n is a positive integer, then it is eventually periodic -/
theorem binary_sequence_eventually_periodic
  (s : BinarySequence) (n : ℕ) (hn : n > 0)
  (h_blocks : ∃ (blocks : Finset (Fin n → Fin 2)),
    blocks.card = n ∧
    ∀ k, ∃ b ∈ blocks, Block s n k = b) :
  EventuallyPeriodic s :=
sorry

end binary_sequence_eventually_periodic_l3517_351792


namespace complementary_angles_ratio_l3517_351755

theorem complementary_angles_ratio (x y : ℝ) : 
  x + y = 90 →  -- The angles are complementary (sum to 90°)
  x = 4 * y →   -- The ratio of the angles is 4:1
  y = 18 :=     -- The smaller angle is 18°
by sorry

end complementary_angles_ratio_l3517_351755


namespace movie_ticket_price_difference_l3517_351767

theorem movie_ticket_price_difference (regular_price children_price : ℕ) : 
  regular_price = 9 →
  2 * regular_price + 3 * children_price = 39 →
  regular_price - children_price = 2 :=
by sorry

end movie_ticket_price_difference_l3517_351767


namespace sum_three_numbers_l3517_351750

theorem sum_three_numbers (x y z M : ℝ) : 
  x + y + z = 90 ∧ 
  x - 5 = M ∧ 
  y + 5 = M ∧ 
  5 * z = M → 
  M = 450 / 11 := by
sorry

end sum_three_numbers_l3517_351750


namespace hyperbola_m_range_l3517_351770

def is_hyperbola (m : ℝ) : Prop :=
  (16 - m) * (9 - m) < 0

theorem hyperbola_m_range :
  ∀ m : ℝ, is_hyperbola m ↔ 9 < m ∧ m < 16 :=
by sorry

end hyperbola_m_range_l3517_351770


namespace cinema_entry_cost_l3517_351777

def totalEntryCost (totalStudents : ℕ) (regularPrice : ℕ) (discountInterval : ℕ) (freeInterval : ℕ) : ℕ :=
  let discountedStudents := totalStudents / discountInterval
  let freeStudents := totalStudents / freeInterval
  let fullPriceStudents := totalStudents - discountedStudents - freeStudents
  let fullPriceCost := fullPriceStudents * regularPrice
  let discountedCost := discountedStudents * (regularPrice / 2)
  fullPriceCost + discountedCost

theorem cinema_entry_cost :
  totalEntryCost 84 50 12 35 = 3925 := by
  sorry

end cinema_entry_cost_l3517_351777


namespace four_positions_l3517_351732

/-- Represents a cell in the 4x4 grid -/
structure Cell :=
  (row : Fin 4)
  (col : Fin 4)

/-- Represents the value in a cell -/
inductive CellValue
  | One
  | Two
  | Three
  | Four

/-- Represents the 4x4 grid -/
def Grid := Cell → Option CellValue

/-- Check if a 2x2 square is valid (contains 1, 2, 3, 4 exactly once) -/
def isValidSquare (g : Grid) (topLeft : Cell) : Prop := sorry

/-- Check if the entire grid is valid -/
def isValidGrid (g : Grid) : Prop := sorry

/-- The given partial grid -/
def partialGrid : Grid := sorry

/-- Theorem stating the positions of fours in the grid -/
theorem four_positions (g : Grid) 
  (h1 : isValidGrid g) 
  (h2 : g = partialGrid) : 
  g ⟨0, 2⟩ = some CellValue.Four ∧ 
  g ⟨1, 0⟩ = some CellValue.Four ∧ 
  g ⟨2, 1⟩ = some CellValue.Four ∧ 
  g ⟨3, 3⟩ = some CellValue.Four := by
  sorry

end four_positions_l3517_351732


namespace parabola_h_value_l3517_351743

/-- Represents a parabola of the form y = a(x-h)^2 + c -/
structure Parabola where
  a : ℝ
  h : ℝ
  c : ℝ

/-- The y-intercept of a parabola -/
def y_intercept (p : Parabola) : ℝ := p.a * p.h^2 + p.c

/-- Checks if a parabola has two positive integer x-intercepts -/
def has_two_positive_integer_x_intercepts (p : Parabola) : Prop :=
  ∃ x1 x2 : ℤ, x1 > 0 ∧ x2 > 0 ∧ x1 ≠ x2 ∧ 
    p.a * (x1 - p.h)^2 + p.c = 0 ∧ 
    p.a * (x2 - p.h)^2 + p.c = 0

theorem parabola_h_value 
  (p1 p2 : Parabola)
  (h1 : p1.a = 4)
  (h2 : p2.a = 5)
  (h3 : p1.h = p2.h)
  (h4 : y_intercept p1 = 4027)
  (h5 : y_intercept p2 = 4028)
  (h6 : has_two_positive_integer_x_intercepts p1)
  (h7 : has_two_positive_integer_x_intercepts p2) :
  p1.h = 36 := by
  sorry

end parabola_h_value_l3517_351743


namespace solution_value_l3517_351719

theorem solution_value (a : ℝ) (h : a^2 - 2*a - 1 = 0) : a^2 - 2*a + 2022 = 2023 := by
  sorry

end solution_value_l3517_351719


namespace root_equation_solution_l3517_351766

theorem root_equation_solution (y : ℝ) : 
  (y * (y^5)^(1/3))^(1/7) = 4 → y = 2^(21/4) := by
sorry

end root_equation_solution_l3517_351766


namespace transaction_fraction_l3517_351714

theorem transaction_fraction :
  let mabel_transactions : ℕ := 90
  let anthony_transactions : ℕ := mabel_transactions + mabel_transactions / 10
  let jade_transactions : ℕ := 82
  let cal_transactions : ℕ := jade_transactions - 16
  (cal_transactions : ℚ) / (anthony_transactions : ℚ) = 2 / 3 := by
sorry

end transaction_fraction_l3517_351714


namespace circular_arrangement_equality_l3517_351706

/-- Given a circular arrangement of n people numbered 1 to n,
    if the distance from person 31 to person 7 is equal to
    the distance from person 31 to person 14, then n = 41. -/
theorem circular_arrangement_equality (n : ℕ) : n > 30 →
  (n - 31 + 7) % n = (14 - 31 + n) % n →
  n = 41 := by
  sorry


end circular_arrangement_equality_l3517_351706


namespace prob_not_all_same_l3517_351752

-- Define a fair 6-sided die
def fair_die : ℕ := 6

-- Define the number of dice
def num_dice : ℕ := 5

-- Define the probability of all dice showing the same number
def prob_all_same : ℚ := 1 / 1296

-- Theorem statement
theorem prob_not_all_same (d : ℕ) (n : ℕ) (p : ℚ) 
  (hd : d = fair_die) (hn : n = num_dice) (hp : p = prob_all_same) : 
  1 - p = 1295 / 1296 := by
  sorry

end prob_not_all_same_l3517_351752


namespace ratio_of_fractions_l3517_351749

theorem ratio_of_fractions : (1 : ℚ) / 6 / ((5 : ℚ) / 8) = 4 / 15 := by
  sorry

end ratio_of_fractions_l3517_351749


namespace alex_dresses_theorem_l3517_351744

/-- Calculates the maximum number of complete dresses Alex can make --/
def max_dresses (initial_silk initial_satin initial_chiffon : ℕ) 
                (silk_per_dress satin_per_dress chiffon_per_dress : ℕ) 
                (friends : ℕ) (silk_per_friend satin_per_friend chiffon_per_friend : ℕ) : ℕ :=
  let remaining_silk := initial_silk - friends * silk_per_friend
  let remaining_satin := initial_satin - friends * satin_per_friend
  let remaining_chiffon := initial_chiffon - friends * chiffon_per_friend
  min (remaining_silk / silk_per_dress) 
      (min (remaining_satin / satin_per_dress) (remaining_chiffon / chiffon_per_dress))

theorem alex_dresses_theorem : 
  max_dresses 600 400 350 5 3 2 8 15 10 5 = 96 := by
  sorry

end alex_dresses_theorem_l3517_351744


namespace grid_value_l3517_351763

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℝ
  diff : ℝ

/-- Get the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first + seq.diff * (n - 1 : ℝ)

theorem grid_value (row : ArithmeticSequence) (col : ArithmeticSequence) : 
  row.first = 25 ∧ 
  row.nthTerm 4 = 11 ∧ 
  col.nthTerm 2 = 11 ∧ 
  col.nthTerm 3 = 11 ∧
  row.nthTerm 7 = col.nthTerm 1 →
  row.nthTerm 7 = -3 := by
  sorry


end grid_value_l3517_351763


namespace mobile_phone_price_l3517_351703

theorem mobile_phone_price (x : ℝ) : 
  (0.8 * (1.4 * x)) - x = 270 → x = 2250 := by
  sorry

end mobile_phone_price_l3517_351703


namespace inscribed_parallelepiped_surface_area_l3517_351721

/-- A parallelepiped inscribed in a sphere -/
structure InscribedParallelepiped where
  /-- The radius of the circumscribed sphere -/
  sphere_radius : ℝ
  /-- The volume of the parallelepiped -/
  volume : ℝ
  /-- The edges of the parallelepiped -/
  a : ℝ
  b : ℝ
  c : ℝ
  /-- The sphere radius is √3 -/
  sphere_radius_eq : sphere_radius = Real.sqrt 3
  /-- The volume is 8 -/
  volume_eq : volume = 8
  /-- The volume is the product of the edges -/
  volume_product : volume = a * b * c
  /-- The diagonal of the parallelepiped equals the diameter of the sphere -/
  diagonal_eq : a^2 + b^2 + c^2 = 4 * sphere_radius^2

/-- The theorem stating that the surface area of the inscribed parallelepiped is 24 -/
theorem inscribed_parallelepiped_surface_area
  (p : InscribedParallelepiped) : 2 * (p.a * p.b + p.b * p.c + p.c * p.a) = 24 := by
  sorry

end inscribed_parallelepiped_surface_area_l3517_351721


namespace quadratic_function_properties_l3517_351762

-- Define the quadratic function f(x)
def f (x : ℝ) : ℝ := -x^2 + 2*x + 15

-- Define g(x) in terms of f(x)
def g (x : ℝ) : ℝ := f x + (-2)*x

-- Theorem statement
theorem quadratic_function_properties :
  -- Vertex of f(x) is at (1, 16)
  (f 1 = 16) ∧
  -- Roots of f(x) are 8 units apart
  (∃ r₁ r₂ : ℝ, f r₁ = 0 ∧ f r₂ = 0 ∧ r₂ - r₁ = 8) →
  -- Conclusion 1: f(x) = -x^2 + 2x + 15
  (∀ x : ℝ, f x = -x^2 + 2*x + 15) ∧
  -- Conclusion 2: Maximum value of g(x) on [0, 2] is 7
  (∀ x : ℝ, x ≥ 0 ∧ x ≤ 2 → g x ≤ 7) ∧ (∃ x : ℝ, x ≥ 0 ∧ x ≤ 2 ∧ g x = 7) :=
by sorry

end quadratic_function_properties_l3517_351762


namespace value_of_shares_theorem_l3517_351722

/-- Represents the value of shares bought by an investor -/
def value_of_shares (N : ℝ) : ℝ := 0.5 * N * 25

/-- Theorem stating the relationship between the value of shares and the number of shares -/
theorem value_of_shares_theorem (N : ℝ) (dividend_rate : ℝ) (return_rate : ℝ) (share_price : ℝ)
  (h1 : dividend_rate = 0.125)
  (h2 : return_rate = 0.25)
  (h3 : share_price = 25) :
  value_of_shares N = return_rate * (value_of_shares N) / dividend_rate := by
sorry

end value_of_shares_theorem_l3517_351722


namespace association_ticket_sales_l3517_351701

/-- Represents an association with male and female members selling raffle tickets -/
structure Association where
  male_members : ℕ
  female_members : ℕ
  male_avg_tickets : ℝ
  female_avg_tickets : ℝ
  overall_avg_tickets : ℝ

/-- The theorem stating the conditions and the result to be proved -/
theorem association_ticket_sales (a : Association) 
  (h1 : a.female_members = 2 * a.male_members)
  (h2 : a.overall_avg_tickets = 66)
  (h3 : a.male_avg_tickets = 58) :
  a.female_avg_tickets = 70 := by
  sorry


end association_ticket_sales_l3517_351701


namespace student_village_arrangements_l3517_351731

theorem student_village_arrangements :
  let num_students : ℕ := 3
  let num_villages : ℕ := 2
  let arrangements : ℕ := (num_students.choose (num_students - 1)) * (num_villages.factorial)
  arrangements = 6 := by
  sorry

end student_village_arrangements_l3517_351731


namespace right_triangle_angle_calculation_l3517_351736

theorem right_triangle_angle_calculation (A B C : Real) : 
  A = 35 → C = 90 → A + B + C = 180 → B = 55 := by
  sorry

end right_triangle_angle_calculation_l3517_351736


namespace defective_items_count_l3517_351789

def total_products : ℕ := 100
def defective_items : ℕ := 2
def items_to_draw : ℕ := 3

def ways_with_defective : ℕ := Nat.choose total_products items_to_draw - Nat.choose (total_products - defective_items) items_to_draw

theorem defective_items_count : ways_with_defective = 9472 := by
  sorry

end defective_items_count_l3517_351789


namespace sine_inequality_unique_solution_l3517_351779

theorem sine_inequality_unique_solution :
  ∀ y ∈ Set.Icc 0 (Real.pi / 2),
    (∀ x ∈ Set.Icc 0 Real.pi, Real.sin (x + y) < Real.sin x + Real.sin y) ↔
    y = 0 :=
by sorry

end sine_inequality_unique_solution_l3517_351779


namespace hyperbola_equation_l3517_351712

-- Define the hyperbola
def Hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.2^2 / b^2) - (p.1^2 / a^2) = 1}

-- Define the asymptotes
def Asymptotes (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = m * p.1 ∨ p.2 = -m * p.1}

theorem hyperbola_equation 
  (h : (3, 2 * Real.sqrt 2) ∈ Hyperbola 3 2) 
  (a : Asymptotes (2/3) = Asymptotes (2/3)) :
  Hyperbola 3 2 = {p : ℝ × ℝ | (p.2^2 / 4) - (p.1^2 / 9) = 1} :=
by sorry

end hyperbola_equation_l3517_351712


namespace product_real_implies_a_value_l3517_351709

theorem product_real_implies_a_value (z₁ z₂ : ℂ) (a : ℝ) :
  z₁ = 2 + I →
  z₂ = 1 + a * I →
  (z₁ * z₂).im = 0 →
  a = -1/2 := by sorry

end product_real_implies_a_value_l3517_351709


namespace reebok_cost_is_35_l3517_351710

/-- The cost of a pair of Reebok shoes -/
def reebok_cost : ℚ := 35

/-- Alice's sales quota -/
def quota : ℚ := 1000

/-- The cost of a pair of Adidas shoes -/
def adidas_cost : ℚ := 45

/-- The cost of a pair of Nike shoes -/
def nike_cost : ℚ := 60

/-- The number of Nike shoes sold -/
def nike_sold : ℕ := 8

/-- The number of Adidas shoes sold -/
def adidas_sold : ℕ := 6

/-- The number of Reebok shoes sold -/
def reebok_sold : ℕ := 9

/-- The amount by which Alice exceeded her quota -/
def excess : ℚ := 65

theorem reebok_cost_is_35 :
  reebok_cost * reebok_sold + nike_cost * nike_sold + adidas_cost * adidas_sold = quota + excess :=
by sorry

end reebok_cost_is_35_l3517_351710


namespace sugar_price_correct_l3517_351713

/-- The price of a kilogram of sugar -/
def sugar_price : ℝ := 1.50

/-- The price of a kilogram of salt -/
noncomputable def salt_price : ℝ := 5 - 3 * sugar_price

theorem sugar_price_correct : sugar_price = 1.50 := by
  have h1 : 2 * sugar_price + 5 * salt_price = 5.50 := by sorry
  have h2 : 3 * sugar_price + salt_price = 5 := by sorry
  sorry

end sugar_price_correct_l3517_351713


namespace space_diagonals_of_specific_polyhedron_l3517_351783

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  let total_line_segments := Q.vertices.choose 2
  let face_diagonals := Q.quadrilateral_faces * 2
  total_line_segments - Q.edges - face_diagonals

/-- The main theorem stating the number of space diagonals in the given polyhedron -/
theorem space_diagonals_of_specific_polyhedron :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 70,
    faces := 42,
    triangular_faces := 30,
    quadrilateral_faces := 12
  }
  space_diagonals Q = 341 := by sorry

end space_diagonals_of_specific_polyhedron_l3517_351783


namespace monotonic_implies_not_even_but_not_conversely_l3517_351787

-- Define the properties of a function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def IsMonotonic (f : ℝ → ℝ) : Prop := ∀ x y, x ≤ y → f x ≤ f y

-- State the theorem
theorem monotonic_implies_not_even_but_not_conversely :
  (∃ f : ℝ → ℝ, IsMonotonic f → ¬IsEven f) ∧
  (∃ g : ℝ → ℝ, ¬IsEven g ∧ ¬IsMonotonic g) :=
sorry

end monotonic_implies_not_even_but_not_conversely_l3517_351787


namespace polynomial_proofs_l3517_351718

theorem polynomial_proofs (x : ℝ) : 
  (x^2 + 2*x - 3 = (x + 3)*(x - 1)) ∧ 
  (x^2 + 8*x + 7 = (x + 7)*(x + 1)) ∧ 
  (-x^2 + 2/3*x + 1 < 4/3) := by
  sorry

end polynomial_proofs_l3517_351718


namespace parabola_shift_theorem_l3517_351796

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c - v }

/-- The original parabola y = 4x^2 + 1 -/
def original_parabola : Parabola :=
  { a := 4, b := 0, c := 1 }

theorem parabola_shift_theorem :
  let shifted := shift_parabola original_parabola 3 2
  shifted = { a := 4, b := -24, c := 35 } := by sorry

end parabola_shift_theorem_l3517_351796


namespace subset_condition_l3517_351773

def A : Set ℝ := {x | x^2 - 3*x + 2 < 0}
def B (a : ℝ) : Set ℝ := {x | 0 < x ∧ x < a}

theorem subset_condition (a : ℝ) : A ⊆ B a ↔ a ≥ 2 := by
  sorry

end subset_condition_l3517_351773


namespace machine_output_for_26_l3517_351742

def machine_operation (input : ℕ) : ℕ :=
  (input + 15) - 6

theorem machine_output_for_26 :
  machine_operation 26 = 35 := by
  sorry

end machine_output_for_26_l3517_351742


namespace boat_speed_in_still_water_boat_speed_proof_l3517_351723

/-- The speed of a boat in still water, given stream speed and downstream travel information -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : stream_speed = 5)
  (h2 : downstream_distance = 120)
  (h3 : downstream_time = 4)
  : ℝ :=
  let downstream_speed := downstream_distance / downstream_time
  let boat_speed := downstream_speed - stream_speed
  25

/-- Proof that the boat's speed in still water is 25 km/hr -/
theorem boat_speed_proof 
  (stream_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : stream_speed = 5)
  (h2 : downstream_distance = 120)
  (h3 : downstream_time = 4)
  : boat_speed_in_still_water stream_speed downstream_distance downstream_time h1 h2 h3 = 25 := by
  sorry

end boat_speed_in_still_water_boat_speed_proof_l3517_351723


namespace raffle_ticket_cost_l3517_351761

theorem raffle_ticket_cost (total_amount : ℕ) (num_tickets : ℕ) (cost_per_ticket : ℚ) : 
  total_amount = 620 → num_tickets = 155 → cost_per_ticket = 4 → 
  (total_amount : ℚ) / num_tickets = cost_per_ticket :=
by sorry

end raffle_ticket_cost_l3517_351761


namespace juice_bar_problem_l3517_351795

theorem juice_bar_problem (total_spent : ℕ) (mango_juice_cost : ℕ) (other_juice_total : ℕ) (total_people : ℕ) :
  total_spent = 94 →
  mango_juice_cost = 5 →
  other_juice_total = 54 →
  total_people = 17 →
  ∃ (other_juice_cost : ℕ),
    other_juice_cost = 6 ∧
    other_juice_cost * (total_people - (total_spent - other_juice_total) / mango_juice_cost) = other_juice_total :=
by sorry

end juice_bar_problem_l3517_351795


namespace max_value_quadratic_l3517_351753

theorem max_value_quadratic (f : ℝ → ℝ) (h : ∀ x, f x = -x^2 + 2*x + 3) :
  (∀ x ∈ Set.Icc 2 3, f x ≤ 3) ∧ (∃ x ∈ Set.Icc 2 3, f x = 3) := by sorry

end max_value_quadratic_l3517_351753


namespace sequence_sum_equals_642_l3517_351725

def a (n : ℕ) : ℤ := (-2) ^ n
def b (n : ℕ) : ℤ := (-2) ^ n + 2
def c (n : ℕ) : ℚ := ((-2) ^ n : ℚ) / 2

theorem sequence_sum_equals_642 :
  ∃! n : ℕ, (a n : ℚ) + (b n : ℚ) + c n = 642 ∧ n = 8 :=
sorry

end sequence_sum_equals_642_l3517_351725


namespace square_of_five_power_plus_four_l3517_351720

theorem square_of_five_power_plus_four (n : ℕ) : 
  (∃ m : ℕ, 5^n + 4 = m^2) ↔ n = 1 := by sorry

end square_of_five_power_plus_four_l3517_351720


namespace students_taking_both_courses_l3517_351769

/-- Given a class with the following properties:
  * total_students: The total number of students in the class
  * french_students: The number of students taking French
  * german_students: The number of students taking German
  * neither_students: The number of students taking neither French nor German
  
  Prove that the number of students taking both French and German is equal to
  french_students + german_students - (total_students - neither_students) -/
theorem students_taking_both_courses
  (total_students : ℕ)
  (french_students : ℕ)
  (german_students : ℕ)
  (neither_students : ℕ)
  (h1 : total_students = 87)
  (h2 : french_students = 41)
  (h3 : german_students = 22)
  (h4 : neither_students = 33) :
  french_students + german_students - (total_students - neither_students) = 9 := by
  sorry

end students_taking_both_courses_l3517_351769


namespace problem_statement_l3517_351758

theorem problem_statement (m : ℝ) : 
  let U : Set ℝ := Set.univ
  let A : Set ℝ := {x | x^2 + 3*x + 2 = 0}
  let B : Set ℝ := {x | x^2 + (m+1)*x + m = 0}
  (Set.compl A ∩ B = ∅) → (m = 1 ∨ m = 2) := by
  sorry

end problem_statement_l3517_351758


namespace simplify_expression_l3517_351751

theorem simplify_expression (x y : ℝ) (h : x * y ≠ 0) :
  ((x^3 + 2) / x) * ((y^3 + 2) / y) + ((x^3 - 2) / y) * ((y^3 - 2) / x) = 2 * x^2 * y^2 + 8 / (x * y) := by
  sorry

end simplify_expression_l3517_351751


namespace sam_bank_total_l3517_351737

def initial_dimes : ℕ := 9
def initial_quarters : ℕ := 5
def initial_nickels : ℕ := 3

def dad_dimes : ℕ := 7
def dad_quarters : ℕ := 2

def mom_nickels : ℕ := 1
def mom_dimes : ℕ := 2

def grandma_dollars : ℕ := 3

def sister_quarters : ℕ := 4
def sister_nickels : ℕ := 2

def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def nickel_value : ℕ := 5
def dollar_value : ℕ := 100

theorem sam_bank_total :
  (initial_dimes * dime_value +
   initial_quarters * quarter_value +
   initial_nickels * nickel_value +
   dad_dimes * dime_value +
   dad_quarters * quarter_value -
   mom_nickels * nickel_value -
   mom_dimes * dime_value +
   grandma_dollars * dollar_value +
   sister_quarters * quarter_value +
   sister_nickels * nickel_value) = 735 := by
  sorry

end sam_bank_total_l3517_351737


namespace midpoint_polygon_perimeter_bound_l3517_351797

/-- A convex polygon with n sides -/
structure ConvexPolygon where
  n : ℕ
  vertices : Fin n → ℝ × ℝ
  is_convex : sorry

/-- The perimeter of a polygon -/
def perimeter (P : ConvexPolygon) : ℝ :=
  sorry

/-- The polygon formed by connecting the midpoints of sides of another polygon -/
def midpoint_polygon (P : ConvexPolygon) : ConvexPolygon :=
  sorry

/-- Theorem: The perimeter of the midpoint polygon is at least half the perimeter of the original polygon -/
theorem midpoint_polygon_perimeter_bound (P : ConvexPolygon) :
  perimeter (midpoint_polygon P) ≥ (perimeter P) / 2 := by
  sorry

end midpoint_polygon_perimeter_bound_l3517_351797


namespace smallest_four_digit_mod_nine_l3517_351735

theorem smallest_four_digit_mod_nine : ∃ n : ℕ, 
  (n ≥ 1000) ∧ 
  (n < 10000) ∧ 
  (n % 9 = 5) ∧ 
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 9 = 5 → m ≥ n) ∧ 
  (n = 1004) :=
by sorry

end smallest_four_digit_mod_nine_l3517_351735


namespace fourth_term_of_arithmetic_sequence_l3517_351759

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem fourth_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_first : a 0 = 23)
  (h_last : a 5 = 59) :
  a 3 = 41 := by
sorry

end fourth_term_of_arithmetic_sequence_l3517_351759


namespace sonya_fell_six_times_l3517_351774

/-- The number of times Steven fell while ice skating -/
def steven_falls : ℕ := 3

/-- The number of times Stephanie fell while ice skating -/
def stephanie_falls : ℕ := steven_falls + 13

/-- The number of times Sonya fell while ice skating -/
def sonya_falls : ℕ := stephanie_falls / 2 - 2

/-- Theorem stating that Sonya fell 6 times -/
theorem sonya_fell_six_times : sonya_falls = 6 := by sorry

end sonya_fell_six_times_l3517_351774


namespace least_number_with_remainder_l3517_351717

theorem least_number_with_remainder (n : ℕ) : n ≥ 261 ∧ n % 37 = 2 ∧ n % 7 = 2 → n = 261 :=
sorry

end least_number_with_remainder_l3517_351717


namespace geometric_arithmetic_sequence_properties_l3517_351776

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

def arithmetic_sequence (b₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := b₁ + (n - 1) * d

theorem geometric_arithmetic_sequence_properties
  (a₁ : ℝ) (b₁ : ℝ) (q : ℝ) (d : ℝ) 
  (h1 : q = -2/3)
  (h2 : b₁ = 12)
  (h3 : geometric_sequence a₁ q 9 > arithmetic_sequence b₁ d 9)
  (h4 : geometric_sequence a₁ q 10 > arithmetic_sequence b₁ d 10) :
  (geometric_sequence a₁ q 9 * geometric_sequence a₁ q 10 < 0) ∧
  (arithmetic_sequence b₁ d 9 > arithmetic_sequence b₁ d 10) :=
sorry

end geometric_arithmetic_sequence_properties_l3517_351776


namespace solve_for_T_l3517_351738

theorem solve_for_T : ∃ T : ℚ, (3/4 : ℚ) * (1/6 : ℚ) * T = (2/5 : ℚ) * (1/4 : ℚ) * 200 ∧ T = 80 := by
  sorry

end solve_for_T_l3517_351738


namespace no_valid_mapping_divisible_by_1010_l3517_351793

/-- Represents a mapping from letters to digits -/
def LetterToDigitMap := Char → Fin 10

/-- Checks if a mapping is valid for the word INNOPOLIS -/
def is_valid_mapping (m : LetterToDigitMap) : Prop :=
  m 'I' ≠ m 'N' ∧ m 'I' ≠ m 'O' ∧ m 'I' ≠ m 'P' ∧ m 'I' ≠ m 'L' ∧ m 'I' ≠ m 'S' ∧
  m 'N' ≠ m 'O' ∧ m 'N' ≠ m 'P' ∧ m 'N' ≠ m 'L' ∧ m 'N' ≠ m 'S' ∧
  m 'O' ≠ m 'P' ∧ m 'O' ≠ m 'L' ∧ m 'O' ≠ m 'S' ∧
  m 'P' ≠ m 'L' ∧ m 'P' ≠ m 'S' ∧
  m 'L' ≠ m 'S'

/-- Converts the word INNOPOLIS to a number using the given mapping -/
def word_to_number (m : LetterToDigitMap) : ℕ :=
  m 'I' * 100000000 + m 'N' * 10000000 + m 'N' * 1000000 + 
  m 'O' * 100000 + m 'P' * 10000 + m 'O' * 1000 + 
  m 'L' * 100 + m 'I' * 10 + m 'S'

/-- The main theorem stating that no valid mapping exists that makes the number divisible by 1010 -/
theorem no_valid_mapping_divisible_by_1010 :
  ¬ ∃ (m : LetterToDigitMap), is_valid_mapping m ∧ (word_to_number m % 1010 = 0) :=
by sorry

end no_valid_mapping_divisible_by_1010_l3517_351793


namespace count_integers_with_factors_l3517_351790

theorem count_integers_with_factors : 
  ∃! n : ℕ, 200 ≤ n ∧ n ≤ 500 ∧ 22 ∣ n ∧ 16 ∣ n := by
  sorry

end count_integers_with_factors_l3517_351790


namespace problem_solution_l3517_351700

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 2) * x^3 - x^2 + 5*x + (1 - a) * Real.log x

theorem problem_solution :
  (∃ a : ℝ, (∀ x : ℝ, (deriv (f a)) x = 0 ↔ x = 1) ∧ a = 1) ∧
  (∃ x : ℝ, (deriv (f 0)) x = -1) ∧
  (¬∃ x₁ x₂ x₃ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧
    ∃ d : ℝ, x₂ = x₁ + d ∧ x₃ = x₂ + d ∧
    (deriv (f 2)) x₂ = (f 2 x₃ - f 2 x₁) / (x₃ - x₁)) :=
by sorry

end problem_solution_l3517_351700


namespace donation_to_third_home_l3517_351734

/-- Proves that the donation to the third home is $230.00 -/
theorem donation_to_third_home 
  (total_donation : ℝ) 
  (first_home_donation : ℝ) 
  (second_home_donation : ℝ)
  (h1 : total_donation = 700)
  (h2 : first_home_donation = 245)
  (h3 : second_home_donation = 225) :
  total_donation - first_home_donation - second_home_donation = 230 := by
  sorry

end donation_to_third_home_l3517_351734


namespace sixth_graders_percentage_l3517_351708

theorem sixth_graders_percentage (seventh_graders : ℕ) (seventh_graders_percentage : ℚ) (sixth_graders : ℕ) :
  seventh_graders = 64 →
  seventh_graders_percentage = 32 / 100 →
  sixth_graders = 76 →
  (sixth_graders : ℚ) / ((seventh_graders : ℚ) / seventh_graders_percentage) = 38 / 100 := by
  sorry

end sixth_graders_percentage_l3517_351708


namespace quadratic_inequality_equivalence_l3517_351798

theorem quadratic_inequality_equivalence (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 0) ↔ (a > 0 ∧ b^2 - 4*a*c ≤ 0) := by
  sorry

end quadratic_inequality_equivalence_l3517_351798


namespace real_m_values_l3517_351754

theorem real_m_values (m : ℝ) : 
  let z : ℂ := m^2 * (1 + Complex.I) - m * (m + Complex.I)
  Complex.im z = 0 → m = 0 ∨ m = 1 := by
sorry

end real_m_values_l3517_351754


namespace gas_cost_per_gallon_l3517_351772

-- Define the fuel efficiency of the car
def fuel_efficiency : ℝ := 32

-- Define the distance the car can travel
def distance : ℝ := 368

-- Define the total cost of gas
def total_cost : ℝ := 46

-- Theorem to prove the cost of gas per gallon
theorem gas_cost_per_gallon :
  (total_cost / (distance / fuel_efficiency)) = 4 := by
  sorry

end gas_cost_per_gallon_l3517_351772


namespace quadratic_discriminant_l3517_351778

theorem quadratic_discriminant : 
  let a : ℚ := 5
  let b : ℚ := 5 + 1/5
  let c : ℚ := 1/5
  let discriminant := b^2 - 4*a*c
  discriminant = 576/25 := by sorry

end quadratic_discriminant_l3517_351778


namespace extremum_implies_zero_derivative_zero_derivative_not_implies_extremum_l3517_351756

/-- A function f : ℝ → ℝ attains an extremum at x₀ -/
def AttainsExtremum (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  (∀ x, f x ≤ f x₀) ∨ (∀ x, f x ≥ f x₀)

/-- The derivative of f at x₀ is 0 -/
def DerivativeZero (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  deriv f x₀ = 0

theorem extremum_implies_zero_derivative
  (f : ℝ → ℝ) (x₀ : ℝ) (h : Differentiable ℝ f) :
  AttainsExtremum f x₀ → DerivativeZero f x₀ :=
sorry

theorem zero_derivative_not_implies_extremum :
  ∃ (f : ℝ → ℝ) (x₀ : ℝ), Differentiable ℝ f ∧ DerivativeZero f x₀ ∧ ¬AttainsExtremum f x₀ :=
sorry

end extremum_implies_zero_derivative_zero_derivative_not_implies_extremum_l3517_351756


namespace hedge_cost_and_quantity_l3517_351757

/-- Represents the cost and quantity of concrete blocks for a hedge --/
structure HedgeBlocks where
  cost_a : ℕ  -- Cost of Type A blocks
  cost_b : ℕ  -- Cost of Type B blocks
  cost_c : ℕ  -- Cost of Type C blocks
  qty_a : ℕ   -- Quantity of Type A blocks per section
  qty_b : ℕ   -- Quantity of Type B blocks per section
  qty_c : ℕ   -- Quantity of Type C blocks per section
  sections : ℕ -- Number of sections in the hedge

/-- Calculates the total cost and quantity of blocks for the entire hedge --/
def hedge_totals (h : HedgeBlocks) : ℕ × ℕ × ℕ × ℕ :=
  let total_cost := h.sections * (h.cost_a * h.qty_a + h.cost_b * h.qty_b + h.cost_c * h.qty_c)
  let total_a := h.sections * h.qty_a
  let total_b := h.sections * h.qty_b
  let total_c := h.sections * h.qty_c
  (total_cost, total_a, total_b, total_c)

theorem hedge_cost_and_quantity (h : HedgeBlocks) 
  (h_cost_a : h.cost_a = 2)
  (h_cost_b : h.cost_b = 3)
  (h_cost_c : h.cost_c = 4)
  (h_qty_a : h.qty_a = 20)
  (h_qty_b : h.qty_b = 10)
  (h_qty_c : h.qty_c = 5)
  (h_sections : h.sections = 8) :
  hedge_totals h = (720, 160, 80, 40) := by
  sorry

end hedge_cost_and_quantity_l3517_351757


namespace walking_speed_problem_l3517_351785

theorem walking_speed_problem (x : ℝ) (h1 : x > 0) : 
  (100 / x * 12 = 100 + 20) → x = 10 := by
  sorry

end walking_speed_problem_l3517_351785
