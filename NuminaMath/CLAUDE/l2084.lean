import Mathlib

namespace tetrahedron_cross_section_l2084_208460

noncomputable def cross_section_area (V : ℝ) (d : ℝ) : ℝ :=
  3 * V / (5 * d)

theorem tetrahedron_cross_section 
  (V : ℝ) 
  (d : ℝ) 
  (h_V : V = 5) 
  (h_d : d = 1) :
  cross_section_area V d = 3 := by
sorry

end tetrahedron_cross_section_l2084_208460


namespace hundredth_term_is_14_l2084_208413

def S (n : ℕ) : ℕ := n * (n + 1) / 2

def is_nth_term (x n : ℕ) : Prop :=
  ∃ k, k ≤ n ∧ S (x - 1) < k ∧ k ≤ S x

theorem hundredth_term_is_14 : is_nth_term 14 100 := by sorry

end hundredth_term_is_14_l2084_208413


namespace mutual_fund_investment_l2084_208454

theorem mutual_fund_investment
  (total_investment : ℝ)
  (mutual_fund_ratio : ℝ)
  (h1 : total_investment = 250000)
  (h2 : mutual_fund_ratio = 3) :
  let commodity_investment := total_investment / (1 + mutual_fund_ratio)
  let mutual_fund_investment := mutual_fund_ratio * commodity_investment
  mutual_fund_investment = 187500 := by
sorry

end mutual_fund_investment_l2084_208454


namespace exists_pentagon_with_similar_subpentagon_l2084_208457

/-- A convex pentagon with specific angles and side lengths -/
structure ConvexPentagon where
  -- Sides of the pentagon
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  -- Two angles of the pentagon (in radians)
  angle1 : ℝ
  angle2 : ℝ
  -- Convexity condition
  convex : angle1 > 0 ∧ angle2 > 0 ∧ angle1 < π ∧ angle2 < π

/-- Similarity between two pentagons -/
def isSimilar (p1 p2 : ConvexPentagon) : Prop :=
  ∃ k : ℝ, k > 0 ∧
    p2.side1 = k * p1.side1 ∧
    p2.side2 = k * p1.side2 ∧
    p2.side3 = k * p1.side3 ∧
    p2.side4 = k * p1.side4 ∧
    p2.side5 = k * p1.side5 ∧
    p2.angle1 = p1.angle1 ∧
    p2.angle2 = p1.angle2

/-- Theorem stating the existence of a specific convex pentagon with a similar sub-pentagon -/
theorem exists_pentagon_with_similar_subpentagon :
  ∃ (p : ConvexPentagon) (q : ConvexPentagon),
    p.side1 = 2 ∧ p.side2 = 4 ∧ p.side3 = 8 ∧ p.side4 = 6 ∧ p.side5 = 12 ∧
    p.angle1 = π / 3 ∧ p.angle2 = 2 * π / 3 ∧
    isSimilar p q :=
sorry

end exists_pentagon_with_similar_subpentagon_l2084_208457


namespace ninth_term_is_twelve_l2084_208455

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_condition : a 5 + a 7 = 16
  third_term : a 3 = 4

/-- The 9th term of the arithmetic sequence is 12 -/
theorem ninth_term_is_twelve (seq : ArithmeticSequence) : seq.a 9 = 12 := by
  sorry

end ninth_term_is_twelve_l2084_208455


namespace not_all_rationals_repeating_l2084_208486

-- Define rational numbers
def Rational : Type := ℚ

-- Define integers
def Integer : Type := ℤ

-- Define repeating decimal
def RepeatingDecimal (x : ℚ) : Prop := sorry

-- Statement that integers are rational numbers
axiom integer_is_rational : Integer → Rational

-- Statement that not all integers are repeating decimals
axiom not_all_integers_repeating : ∃ (n : Integer), ¬(RepeatingDecimal (integer_is_rational n))

-- Theorem to prove
theorem not_all_rationals_repeating : ¬(∀ (q : Rational), RepeatingDecimal q) := by
  sorry

end not_all_rationals_repeating_l2084_208486


namespace ann_found_blocks_l2084_208471

/-- Given that Ann initially had 9 blocks and ended up with 53 blocks,
    prove that she found 44 blocks. -/
theorem ann_found_blocks (initial_blocks : ℕ) (final_blocks : ℕ) :
  initial_blocks = 9 →
  final_blocks = 53 →
  final_blocks - initial_blocks = 44 := by
  sorry

end ann_found_blocks_l2084_208471


namespace correct_systematic_sample_l2084_208476

def total_missiles : ℕ := 70
def selected_missiles : ℕ := 7

def systematic_sample (start : ℕ) (interval : ℕ) : List ℕ :=
  List.range selected_missiles |>.map (fun i => start + i * interval)

theorem correct_systematic_sample :
  ∃ (start : ℕ), start ≤ total_missiles ∧
  systematic_sample start (total_missiles / selected_missiles) =
    [3, 13, 23, 33, 43, 53, 63] :=
by sorry

end correct_systematic_sample_l2084_208476


namespace square_difference_theorem_l2084_208474

theorem square_difference_theorem (x y : ℝ) 
  (sum_eq : x + y = 8) 
  (diff_eq : x - y = 4) : 
  x^2 - y^2 = 32 := by
sorry

end square_difference_theorem_l2084_208474


namespace nine_hundred_in_column_B_l2084_208478

/-- The column type representing the six columns A, B, C, D, E, F -/
inductive Column
| A | B | C | D | E | F

/-- The function that determines the column for a given positive integer -/
def column_for_number (n : ℕ) : Column :=
  match (n - 3) % 12 with
  | 0 => Column.A
  | 1 => Column.B
  | 2 => Column.C
  | 3 => Column.D
  | 4 => Column.A
  | 5 => Column.F
  | 6 => Column.E
  | 7 => Column.F
  | 8 => Column.D
  | 9 => Column.C
  | 10 => Column.B
  | 11 => Column.A
  | _ => Column.A  -- This case should never occur

theorem nine_hundred_in_column_B :
  column_for_number 900 = Column.B :=
by sorry

end nine_hundred_in_column_B_l2084_208478


namespace function_property_l2084_208496

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem function_property (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f 3)
  (h_f1 : f 1 > 1)
  (h_f2 : f 2 = a) :
  a < -1 := by
  sorry

end function_property_l2084_208496


namespace fourth_term_of_geometric_sequence_l2084_208448

/-- Given a geometric sequence with first term a and common ratio r,
    the nth term is given by a * r^(n-1) -/
def geometric_term (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r^(n-1)

/-- The fourth term of a geometric sequence with first term 3 and second term 1/3 is 1/243 -/
theorem fourth_term_of_geometric_sequence :
  let a := 3
  let a₂ := 1/3
  let r := a₂ / a
  geometric_term a r 4 = 1/243 := by sorry

end fourth_term_of_geometric_sequence_l2084_208448


namespace credit_card_balance_transfer_l2084_208430

theorem credit_card_balance_transfer (G : ℝ) : 
  let gold_limit : ℝ := G
  let platinum_limit : ℝ := 2 * G
  let gold_balance : ℝ := G / 3
  let platinum_balance : ℝ := platinum_limit / 4
  let new_platinum_balance : ℝ := platinum_balance + gold_balance
  let unspent_portion : ℝ := (platinum_limit - new_platinum_balance) / platinum_limit
  unspent_portion = 7 / 12 := by sorry

end credit_card_balance_transfer_l2084_208430


namespace distinct_primes_not_dividing_l2084_208490

/-- A function that pairs the positive divisors of a number -/
def divisor_pairing (n : ℕ+) : Set (ℕ × ℕ) := sorry

/-- Predicate to check if a number is prime -/
def is_prime (p : ℕ) : Prop := Nat.Prime p

/-- The main theorem -/
theorem distinct_primes_not_dividing (n : ℕ+) 
  (h : ∀ (pair : ℕ × ℕ), pair ∈ divisor_pairing n → is_prime (pair.1 + pair.2)) :
  (∀ (p q : ℕ), 
    (∃ (pair1 pair2 : ℕ × ℕ), pair1 ∈ divisor_pairing n ∧ pair2 ∈ divisor_pairing n ∧ 
      p = pair1.1 + pair1.2 ∧ q = pair2.1 + pair2.2 ∧ p ≠ q) →
    (∀ (r : ℕ), (∃ (pair : ℕ × ℕ), pair ∈ divisor_pairing n ∧ r = pair.1 + pair.2) → 
      ¬(r ∣ n))) :=
sorry

end distinct_primes_not_dividing_l2084_208490


namespace consecutive_squares_sum_diff_odd_l2084_208461

theorem consecutive_squares_sum_diff_odd (n : ℕ) : 
  Odd (n^2 + (n+1)^2) ∧ Odd ((n+1)^2 - n^2) := by
  sorry

end consecutive_squares_sum_diff_odd_l2084_208461


namespace decagon_area_decagon_area_specific_l2084_208438

/-- The area of a decagon inscribed in a rectangle with specific properties. -/
theorem decagon_area (perimeter : ℝ) (length_ratio width_ratio : ℕ) : ℝ :=
  let length := (3 * perimeter) / (10 : ℝ)
  let width := (2 * perimeter) / (10 : ℝ)
  let rectangle_area := length * width
  let triangle_area_long := (1 / 2 : ℝ) * (length / 5) * (length / 5)
  let triangle_area_short := (1 / 2 : ℝ) * (width / 5) * (width / 5)
  let total_removed_area := 4 * triangle_area_long + 4 * triangle_area_short
  rectangle_area - total_removed_area

/-- 
  The area of a decagon inscribed in a rectangle is 1984 square centimeters, given:
  - The vertices of the decagon divide the sides of the rectangle into five equal parts
  - The perimeter of the rectangle is 200 centimeters
  - The ratio of length to width of the rectangle is 3:2
-/
theorem decagon_area_specific : decagon_area 200 3 2 = 1984 := by
  sorry

end decagon_area_decagon_area_specific_l2084_208438


namespace line_parabola_intersection_l2084_208475

theorem line_parabola_intersection (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = k * p.1 + 2 ∧ p.2^2 = 8 * p.1) ↔ (k = 0 ∨ k = 1) :=
sorry

end line_parabola_intersection_l2084_208475


namespace sin_cos_sum_equals_negative_one_l2084_208411

theorem sin_cos_sum_equals_negative_one : 
  Real.sin ((11 / 6) * Real.pi) + Real.cos ((10 / 3) * Real.pi) = -1 := by
  sorry

end sin_cos_sum_equals_negative_one_l2084_208411


namespace geometric_with_arithmetic_subsequence_l2084_208435

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

/-- An arithmetic subsequence of a sequence -/
def arithmetic_subsequence (a : ℕ → ℝ) (sub : ℕ → ℕ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (sub (n + 1)) - a (sub n) = d

/-- The main theorem: if a geometric sequence has an infinite arithmetic subsequence,
    then its common ratio is -1 -/
theorem geometric_with_arithmetic_subsequence
  (a : ℕ → ℝ) (q : ℝ) (sub : ℕ → ℕ) (d : ℝ) (h_ne_one : q ≠ 1) :
  geometric_sequence a q →
  (∃ d, arithmetic_subsequence a sub d) →
  q = -1 := by
sorry

end geometric_with_arithmetic_subsequence_l2084_208435


namespace solution_set_implies_sum_l2084_208440

-- Define the inequality solution set
def SolutionSet (a b : ℝ) : Set ℝ :=
  {x | 1 < x ∧ x < 2}

-- State the theorem
theorem solution_set_implies_sum (a b : ℝ) :
  SolutionSet a b = {x | (x - a) * (x - b) < 0} →
  a + b = 3 := by
  sorry

end solution_set_implies_sum_l2084_208440


namespace quadratic_properties_l2084_208441

/-- A quadratic function passing through specific points -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_properties (a b c : ℝ) (h : a ≠ 0) :
  let f := QuadraticFunction a b c
  (f (-3) = 15) ∧ (f (-1) = 3) ∧ (f 0 = 0) ∧ (f 1 = -1) ∧ (f 2 = 0) ∧ (f 4 = 8) →
  (∀ x, f (1 + x) = f (1 - x)) ∧  -- Axis of symmetry at x = 1
  (f (-2) = 8) ∧ (f 3 = 3) ∧      -- Values at x = -2 and x = 3
  (f 0 = 0) ∧ (f 2 = 0)           -- Roots at x = 0 and x = 2
  := by sorry

end quadratic_properties_l2084_208441


namespace min_value_sum_reciprocal_sin_squared_l2084_208445

theorem min_value_sum_reciprocal_sin_squared (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C → -- Angles are positive
  A + B + C = π → -- Sum of angles in a triangle
  C = π / 2 → -- Right angle condition
  (∀ x y : ℝ, 0 < x → 0 < y → x + y + π/2 = π → 4 / (Real.sin x)^2 + 9 / (Real.sin y)^2 ≥ 25) ∧ 
  (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y + π/2 = π ∧ 4 / (Real.sin x)^2 + 9 / (Real.sin y)^2 = 25) :=
by sorry

end min_value_sum_reciprocal_sin_squared_l2084_208445


namespace units_digit_product_l2084_208408

theorem units_digit_product : (5^2 + 1) * (5^3 + 1) * (5^23 + 1) % 10 = 6 := by
  sorry

end units_digit_product_l2084_208408


namespace sum_remainder_mod_11_l2084_208495

theorem sum_remainder_mod_11 : (8735 + 8736 + 8737 + 8738) % 11 = 10 := by
  sorry

end sum_remainder_mod_11_l2084_208495


namespace max_piece_length_l2084_208409

def rope_lengths : List Nat := [48, 72, 120, 144]

def min_pieces : Nat := 5

def is_valid_piece_length (len : Nat) : Bool :=
  rope_lengths.all (fun rope => rope % len = 0 ∧ rope / len ≥ min_pieces)

theorem max_piece_length :
  ∃ (max_len : Nat), max_len = 8 ∧
    is_valid_piece_length max_len ∧
    ∀ (len : Nat), len > max_len → ¬is_valid_piece_length len :=
by sorry

end max_piece_length_l2084_208409


namespace unique_solution_quadratic_l2084_208468

/-- 
For a quadratic equation qx^2 - 20x + 9 = 0 to have exactly one solution,
q must equal 100/9.
-/
theorem unique_solution_quadratic : 
  ∃! q : ℚ, q ≠ 0 ∧ (∃! x : ℝ, q * x^2 - 20 * x + 9 = 0) := by
  sorry

end unique_solution_quadratic_l2084_208468


namespace l_shaped_floor_paving_cost_l2084_208447

/-- Calculates the total cost of paving an L-shaped floor with two types of slabs -/
def total_paving_cost (large_length large_width small_length small_width type_a_cost type_b_cost : ℝ) : ℝ :=
  let large_area := large_length * large_width
  let small_area := small_length * small_width
  let large_cost := large_area * type_a_cost
  let small_cost := small_area * type_b_cost
  large_cost + small_cost

/-- Theorem stating that the total cost of paving the L-shaped floor is Rs. 13,781.25 -/
theorem l_shaped_floor_paving_cost :
  total_paving_cost 5.5 3.75 2.5 1.25 600 450 = 13781.25 := by
  sorry

end l_shaped_floor_paving_cost_l2084_208447


namespace seventh_root_of_unity_product_l2084_208473

theorem seventh_root_of_unity_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 8 := by
  sorry

end seventh_root_of_unity_product_l2084_208473


namespace megan_zoo_pictures_l2084_208424

/-- The number of pictures Megan took at the zoo -/
def zoo_pictures : ℕ := sorry

/-- The number of pictures Megan took at the museum -/
def museum_pictures : ℕ := 18

/-- The number of pictures Megan deleted -/
def deleted_pictures : ℕ := 31

/-- The number of pictures Megan has left after deleting -/
def remaining_pictures : ℕ := 2

/-- Theorem stating that Megan took 15 pictures at the zoo -/
theorem megan_zoo_pictures : 
  zoo_pictures = 15 :=
by
  have h1 : zoo_pictures + museum_pictures - deleted_pictures = remaining_pictures := sorry
  sorry

end megan_zoo_pictures_l2084_208424


namespace total_cost_calculation_l2084_208479

/-- The cost of items in dollars -/
structure ItemCost where
  mango : ℝ
  rice : ℝ
  flour : ℝ

/-- Given conditions and the theorem to prove -/
theorem total_cost_calculation (c : ItemCost) 
  (h1 : 10 * c.mango = 24 * c.rice)
  (h2 : 6 * c.flour = 2 * c.rice)
  (h3 : c.flour = 23) :
  4 * c.mango + 3 * c.rice + 5 * c.flour = 984.4 := by
  sorry


end total_cost_calculation_l2084_208479


namespace involutive_function_property_l2084_208412

/-- A function f that is its own inverse -/
def InvolutiveFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (f x) = x

/-- The main theorem -/
theorem involutive_function_property
  (a b c d : ℝ)
  (hb : b ≠ 0)
  (hd : d ≠ 0)
  (h_c_a : 3 * c^2 = 2 * a^2)
  (f : ℝ → ℝ)
  (hf : ∀ x, f x = (2*a*x + b) / (3*c*x + d))
  (h_involutive : InvolutiveFunction f) :
  2*a + 3*d = -4*a := by
sorry

end involutive_function_property_l2084_208412


namespace base4_calculation_l2084_208414

/-- Converts a number from base 4 to base 10 -/
def base4ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 4 -/
def base10ToBase4 (n : ℕ) : ℕ := sorry

/-- Performs division in base 4 -/
def divBase4 (a b : ℕ) : ℕ := sorry

/-- Performs multiplication in base 4 -/
def mulBase4 (a b : ℕ) : ℕ := sorry

theorem base4_calculation :
  mulBase4 (divBase4 130 3) 14 = 1200 := by sorry

end base4_calculation_l2084_208414


namespace min_value_of_function_l2084_208443

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  9 * x + 1 / (x^3) ≥ 10 ∧ ∃ y > 0, 9 * y + 1 / (y^3) = 10 :=
by sorry

end min_value_of_function_l2084_208443


namespace symmetric_point_correct_l2084_208493

/-- Given a point A and a line l, find the point B symmetric to A about l -/
def symmetricPoint (A : ℝ × ℝ) (l : ℝ × ℝ → Prop) : ℝ × ℝ := sorry

/-- The line x - y - 1 = 0 -/
def line (p : ℝ × ℝ) : Prop :=
  p.1 - p.2 - 1 = 0

theorem symmetric_point_correct :
  let A : ℝ × ℝ := (-1, 1)
  let B : ℝ × ℝ := (2, -2)
  symmetricPoint A line = B := by sorry

end symmetric_point_correct_l2084_208493


namespace exists_divisible_by_digit_sum_in_sequence_l2084_208420

/-- Given a natural number, return the sum of its digits -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Check if a number is divisible by the sum of its digits -/
def is_divisible_by_digit_sum (n : ℕ) : Prop :=
  n % digit_sum n = 0

/-- Theorem: In any sequence of 18 consecutive three-digit numbers, 
    at least one number is divisible by the sum of its digits -/
theorem exists_divisible_by_digit_sum_in_sequence :
  ∀ (start : ℕ), 100 ≤ start → start + 17 < 1000 →
  ∃ (k : ℕ), k ∈ Finset.range 18 ∧ is_divisible_by_digit_sum (start + k) :=
sorry

end exists_divisible_by_digit_sum_in_sequence_l2084_208420


namespace dice_probability_l2084_208462

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The total number of possible outcomes when rolling the dice -/
def totalOutcomes : ℕ := numSides ^ numDice

/-- The number of outcomes where all dice show the same number -/
def allSameOutcomes : ℕ := numSides

/-- The number of possible sequences (e.g., 1-2-3-4-5, 2-3-4-5-6) -/
def numSequences : ℕ := 2

/-- The number of ways to arrange each sequence -/
def sequenceArrangements : ℕ := Nat.factorial numDice

/-- The probability of rolling five fair 6-sided dice where they don't all show
    the same number and the numbers do not form a sequence -/
theorem dice_probability : 
  (totalOutcomes - allSameOutcomes - numSequences * sequenceArrangements) / totalOutcomes = 7530 / 7776 := by
  sorry

end dice_probability_l2084_208462


namespace principal_arg_range_l2084_208427

open Complex

theorem principal_arg_range (z ω : ℂ) 
  (h1 : abs (z - I) = 1)
  (h2 : z ≠ 0)
  (h3 : z ≠ 2 * I)
  (h4 : ∃ (r : ℝ), (ω - 2 * I) / ω * z / (z - 2 * I) = r) :
  ∃ (θ : ℝ), θ ∈ (Set.Ioo 0 π ∪ Set.Ioo π (2 * π)) ∧ arg (ω - 2) = θ :=
sorry

end principal_arg_range_l2084_208427


namespace triangle_area_implies_x_value_l2084_208463

theorem triangle_area_implies_x_value (x : ℝ) (h1 : x > 0) :
  (1/2 : ℝ) * x * (3*x) = 54 → x = 6 := by
  sorry

end triangle_area_implies_x_value_l2084_208463


namespace annual_loss_is_14400_l2084_208418

/-- The number of yellow balls in the box -/
def yellow_balls : ℕ := 3

/-- The number of white balls in the box -/
def white_balls : ℕ := 3

/-- The total number of balls in the box -/
def total_balls : ℕ := yellow_balls + white_balls

/-- The number of balls drawn in each attempt -/
def drawn_balls : ℕ := 3

/-- The reward for drawing 3 balls of the same color (in yuan) -/
def same_color_reward : ℚ := 5

/-- The payment for drawing 3 balls of different colors (in yuan) -/
def diff_color_payment : ℚ := 1

/-- The number of people drawing balls per day -/
def people_per_day : ℕ := 100

/-- The number of days in a year for this calculation -/
def days_per_year : ℕ := 360

/-- The probability of drawing 3 balls of the same color -/
def prob_same_color : ℚ := 1 / 10

/-- The probability of drawing 3 balls of different colors -/
def prob_diff_color : ℚ := 9 / 10

/-- The expected earnings per person (in yuan) -/
def expected_earnings_per_person : ℚ := 
  prob_same_color * same_color_reward - prob_diff_color * diff_color_payment

/-- The daily earnings (in yuan) -/
def daily_earnings : ℚ := expected_earnings_per_person * people_per_day

/-- Theorem: The annual loss is 14400 yuan -/
theorem annual_loss_is_14400 : 
  -daily_earnings * days_per_year = 14400 := by sorry

end annual_loss_is_14400_l2084_208418


namespace painting_price_change_l2084_208403

theorem painting_price_change (P : ℝ) (h : P > 0) : 
  let first_year_price := 1.30 * P
  let final_price := 1.105 * P
  let second_year_decrease := (first_year_price - final_price) / first_year_price
  second_year_decrease = 0.15 := by sorry

end painting_price_change_l2084_208403


namespace consecutive_integers_around_sqrt_13_l2084_208442

theorem consecutive_integers_around_sqrt_13 (a b : ℤ) :
  (b = a + 1) → (a < Real.sqrt 13) → (Real.sqrt 13 < b) → (a + b = 7) := by
  sorry

end consecutive_integers_around_sqrt_13_l2084_208442


namespace largest_fraction_l2084_208421

theorem largest_fraction : 
  let fractions : List ℚ := [2/5, 1/3, 5/15, 4/10, 7/21]
  ∀ x ∈ fractions, x ≤ 2/5 ∧ x ≤ 4/10 :=
by sorry

end largest_fraction_l2084_208421


namespace lattice_points_limit_l2084_208481

/-- The number of lattice points inside a circle of radius r centered at the origin -/
noncomputable def f (r : ℝ) : ℝ := sorry

/-- The difference between f(r) and πr^2 -/
noncomputable def g (r : ℝ) : ℝ := f r - Real.pi * r^2

theorem lattice_points_limit :
  (∀ ε > 0, ∃ R, ∀ r ≥ R, |f r / r^2 - Real.pi| < ε) ∧
  (∀ h < 2, ∀ ε > 0, ∃ R, ∀ r ≥ R, |g r / r^h| < ε) := by
  sorry

end lattice_points_limit_l2084_208481


namespace fraction_zero_solution_l2084_208451

theorem fraction_zero_solution (x : ℝ) : 
  (x^2 - 4*x + 4) / (3*x - 9) = 0 ↔ x = 2 :=
by
  sorry

#check fraction_zero_solution

end fraction_zero_solution_l2084_208451


namespace sphere_radius_l2084_208482

theorem sphere_radius (V : Real) (r : Real) : V = (4 / 3) * Real.pi * r^3 → V = 36 * Real.pi → r = 3 := by
  sorry

end sphere_radius_l2084_208482


namespace circular_tablecloth_radius_increase_l2084_208415

theorem circular_tablecloth_radius_increase :
  let initial_circumference : ℝ := 50
  let final_circumference : ℝ := 64
  let initial_radius : ℝ := initial_circumference / (2 * Real.pi)
  let final_radius : ℝ := final_circumference / (2 * Real.pi)
  final_radius - initial_radius = 7 / Real.pi :=
by sorry

end circular_tablecloth_radius_increase_l2084_208415


namespace hedge_trimming_purpose_l2084_208484

/-- Represents the possible purposes of trimming hedges -/
inductive HedgeTrimPurpose
  | InhibitLateralBuds
  | PromoteLateralBuds
  | InhibitPhototropism
  | InhibitFloweringAndFruiting

/-- Represents the action of trimming hedges -/
structure HedgeTrimming where
  frequency : Nat  -- Represents how often the trimming occurs
  purpose : HedgeTrimPurpose

/-- Represents garden workers -/
structure GardenWorker where
  trims_hedges : Bool

/-- The theorem stating the purpose of hedge trimming -/
theorem hedge_trimming_purpose 
  (workers : List GardenWorker) 
  (trimming : HedgeTrimming) : 
  (∀ w ∈ workers, w.trims_hedges = true) → 
  (trimming.frequency > 0) → 
  (trimming.purpose = HedgeTrimPurpose.PromoteLateralBuds) :=
sorry

end hedge_trimming_purpose_l2084_208484


namespace points_on_parabola_l2084_208405

-- Define the sequence of points
def SequencePoints (x y : ℕ → ℝ) : Prop :=
  ∀ n, Real.sqrt ((x n)^2 + (y n)^2) - y n = 6

-- Define the parabola
def OnParabola (x y : ℝ) : Prop :=
  y = (x^2 / 12) - 3

-- Theorem statement
theorem points_on_parabola 
  (x y : ℕ → ℝ) 
  (h : SequencePoints x y) :
  ∀ n, OnParabola (x n) (y n) := by
sorry

end points_on_parabola_l2084_208405


namespace consecutive_even_numbers_divisible_by_eight_l2084_208400

theorem consecutive_even_numbers_divisible_by_eight (n : ℤ) : 
  ∃ k : ℤ, 4 * n * (n + 1) = 8 * k := by
sorry

end consecutive_even_numbers_divisible_by_eight_l2084_208400


namespace determinant_trigonometric_matrix_l2084_208437

theorem determinant_trigonometric_matrix (α β : Real) :
  let M : Matrix (Fin 3) (Fin 3) Real := ![
    ![Real.sin α * Real.sin β, Real.sin α * Real.cos β, Real.cos α],
    ![Real.cos β, -Real.sin β, 0],
    ![Real.cos α * Real.sin β, Real.cos α * Real.cos β, Real.sin α]
  ]
  Matrix.det M = Real.cos (2 * α) := by
  sorry

end determinant_trigonometric_matrix_l2084_208437


namespace solution_when_a_neg3_m_0_range_of_a_for_real_roots_range_of_m_when_a_0_l2084_208449

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + a + 3
def g (m : ℝ) (x : ℝ) : ℝ := m*x + 5 - 2*m

-- Question 1
theorem solution_when_a_neg3_m_0 :
  {x : ℝ | f (-3) x - g 0 x = 0} = {-1, 5} := by sorry

-- Question 2
theorem range_of_a_for_real_roots :
  {a : ℝ | ∃ x ∈ Set.Icc (-1) 1, f a x = 0} = Set.Icc (-8) 0 := by sorry

-- Question 3
theorem range_of_m_when_a_0 :
  {m : ℝ | ∀ x₁ ∈ Set.Icc 1 4, ∃ x₂ ∈ Set.Icc 1 4, f 0 x₁ = g m x₂} =
  Set.Iic (-3) ∪ Set.Ici 6 := by sorry

end solution_when_a_neg3_m_0_range_of_a_for_real_roots_range_of_m_when_a_0_l2084_208449


namespace expression_simplification_l2084_208497

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 3) :
  (3 / (a - 1) + 1) / ((a^2 + 2*a) / (a^2 - 1)) = (3 + Real.sqrt 3) / 3 := by
  sorry

end expression_simplification_l2084_208497


namespace intersection_sum_l2084_208416

/-- Two lines intersect at a point -/
def intersect_at (m b : ℝ) (x y : ℝ) : Prop :=
  y = m * x + 6 ∧ y = 4 * x + b

/-- The theorem statement -/
theorem intersection_sum (m b : ℝ) :
  intersect_at m b 8 14 → b + m = -17 := by
  sorry

end intersection_sum_l2084_208416


namespace part1_part2_l2084_208489

-- Define the function f
def f (x a b : ℝ) : ℝ := |x - a| + |x - b|

-- Part 1
theorem part1 (a b c : ℝ) (h : |a - b| > c) : ∀ x : ℝ, f x a b > c := by sorry

-- Part 2
theorem part2 (a : ℝ) :
  (∃ x : ℝ, f x a 1 < 2 - |a - 2|) ↔ (1/2 < a ∧ a < 5/2) := by sorry

end part1_part2_l2084_208489


namespace smallest_bottom_right_corner_l2084_208458

/-- Represents a 3x3 grid of natural numbers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Checks if all numbers in the grid are different -/
def all_different (g : Grid) : Prop :=
  ∀ i j k l, g i j = g k l → (i = k ∧ j = l)

/-- Checks if the sum condition is satisfied for rows -/
def row_sum_condition (g : Grid) : Prop :=
  ∀ i, g i 0 + g i 1 = g i 2

/-- Checks if the sum condition is satisfied for columns -/
def col_sum_condition (g : Grid) : Prop :=
  ∀ j, g 0 j + g 1 j = g 2 j

/-- The main theorem stating the smallest possible value for the bottom right corner -/
theorem smallest_bottom_right_corner (g : Grid) 
  (h1 : all_different g) 
  (h2 : row_sum_condition g) 
  (h3 : col_sum_condition g) : 
  g 2 2 ≥ 12 := by
  sorry


end smallest_bottom_right_corner_l2084_208458


namespace sterling_candy_proof_l2084_208446

/-- The number of candy pieces earned for a correct answer -/
def correct_reward : ℕ := 3

/-- The total number of questions answered -/
def total_questions : ℕ := 7

/-- The number of questions answered correctly -/
def correct_answers : ℕ := 7

/-- The number of additional correct answers -/
def additional_correct : ℕ := 2

/-- The total number of candy pieces earned if Sterling answered 2 more questions correctly -/
def total_candy : ℕ := correct_reward * (correct_answers + additional_correct)

theorem sterling_candy_proof :
  total_candy = 27 :=
sorry

end sterling_candy_proof_l2084_208446


namespace alpha_sufficient_not_necessary_for_beta_l2084_208429

theorem alpha_sufficient_not_necessary_for_beta :
  (∀ x : ℝ, x = -1 → x ≤ 0) ∧ 
  (∃ x : ℝ, x ≤ 0 ∧ x ≠ -1) := by
  sorry

end alpha_sufficient_not_necessary_for_beta_l2084_208429


namespace inequality_solution_l2084_208402

def solution_set (a : ℝ) : Set ℝ :=
  if 0 < a ∧ a < 2 then {x | 1 < x ∧ x ≤ 2/a}
  else if a = 2 then ∅
  else if a > 2 then {x | 2/a ≤ x ∧ x < 1}
  else ∅

theorem inequality_solution (a : ℝ) (h : a > 0) :
  {x : ℝ | (a + 2) * x - 4 ≤ 2 * (x - 1)} = solution_set a := by
  sorry

end inequality_solution_l2084_208402


namespace distribution_schemes_count_l2084_208428

/-- The number of ways to distribute students from classes to districts -/
def distribute_students (num_classes : ℕ) (students_per_class : ℕ) (num_districts : ℕ) (students_per_district : ℕ) : ℕ :=
  -- Number of ways to choose 2 classes out of 4
  (num_classes.choose 2) *
  -- Number of ways to choose 2 districts out of 4
  (num_districts.choose 2) *
  -- Number of ways to choose 1 student from each of the remaining 2 classes
  (students_per_class.choose 1) * (students_per_class.choose 1) *
  -- Number of ways to assign these 2 students to the remaining 2 districts
  2

/-- Theorem stating that the number of distribution schemes is 288 -/
theorem distribution_schemes_count :
  distribute_students 4 2 4 2 = 288 := by
  sorry

end distribution_schemes_count_l2084_208428


namespace min_value_theorem_l2084_208453

theorem min_value_theorem (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : 2 * m + n = 2) (h4 : m * n > 0) :
  2 / m + 1 / n ≥ 9 / 2 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 2 * m₀ + n₀ = 2 ∧ m₀ * n₀ > 0 ∧ 2 / m₀ + 1 / n₀ = 9 / 2 :=
by sorry

end min_value_theorem_l2084_208453


namespace quadratic_roots_relation_l2084_208470

theorem quadratic_roots_relation (m n p : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∃ r₁ r₂ : ℝ, (r₁ ≠ r₂) ∧ 
    (∀ x : ℝ, x^2 + p*x + m = 0 ↔ x = r₁ ∨ x = r₂) ∧
    (∀ x : ℝ, x^2 + m*x + n = 0 ↔ x = r₁/2 ∨ x = r₂/2)) →
  n / p = 1 / 8 := by
sorry

end quadratic_roots_relation_l2084_208470


namespace three_intersection_points_l2084_208401

-- Define the three lines
def line1 (x y : ℝ) : Prop := 4 * y - 3 * x = 2
def line2 (x y : ℝ) : Prop := 2 * x + 3 * y = 9
def line3 (x y : ℝ) : Prop := x - y = 1

-- Define an intersection point
def is_intersection (x y : ℝ) : Prop :=
  (line1 x y ∧ line2 x y) ∨ (line2 x y ∧ line3 x y) ∨ (line1 x y ∧ line3 x y)

-- Theorem statement
theorem three_intersection_points :
  ∃ (p1 p2 p3 : ℝ × ℝ),
    is_intersection p1.1 p1.2 ∧
    is_intersection p2.1 p2.2 ∧
    is_intersection p3.1 p3.2 ∧
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    ∀ (x y : ℝ), is_intersection x y → (x, y) = p1 ∨ (x, y) = p2 ∨ (x, y) = p3 :=
sorry

end three_intersection_points_l2084_208401


namespace inverse_function_solution_l2084_208432

/-- Given a function f(x) = 1 / (ax^2 + bx + c), where a, b, and c are nonzero real constants,
    the solutions to f^(-1)(x) = 1 are x = (-b ± √(b^2 - 4a(c-1))) / (2a) -/
theorem inverse_function_solution (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  let f := fun x => 1 / (a * x^2 + b * x + c)
  let sol₁ := (-b + Real.sqrt (b^2 - 4*a*(c-1))) / (2*a)
  let sol₂ := (-b - Real.sqrt (b^2 - 4*a*(c-1))) / (2*a)
  (∀ x, f x = 1 ↔ x = sol₁ ∨ x = sol₂) :=
by sorry

end inverse_function_solution_l2084_208432


namespace special_function_property_l2084_208492

/-- A differentiable function f satisfying f(x) - f''(x) > 0 for all x --/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f ∧ 
  (∀ x : ℝ, Differentiable ℝ (deriv f)) ∧
  (∀ x : ℝ, f x - (deriv (deriv f)) x > 0)

/-- Theorem stating that for a special function f, ef(2015) > f(2016) --/
theorem special_function_property (f : ℝ → ℝ) (hf : SpecialFunction f) : 
  Real.exp 1 * f 2015 > f 2016 := by
  sorry

end special_function_property_l2084_208492


namespace subset_of_any_set_implies_zero_l2084_208426

theorem subset_of_any_set_implies_zero (a : ℝ) : 
  (∀ S : Set ℝ, {x : ℝ | a * x = 1} ⊆ S) → a = 0 := by
  sorry

end subset_of_any_set_implies_zero_l2084_208426


namespace fence_area_calculation_l2084_208491

/-- The time (in hours) it takes the first painter to paint the entire fence alone -/
def painter1_time : ℝ := 12

/-- The time (in hours) it takes the second painter to paint the entire fence alone -/
def painter2_time : ℝ := 15

/-- The reduction in combined painting speed (in square feet per hour) when the painters work together -/
def speed_reduction : ℝ := 5

/-- The time (in hours) it takes both painters to paint the fence together -/
def combined_time : ℝ := 7

/-- The total area of the fence in square feet -/
def fence_area : ℝ := 700

theorem fence_area_calculation :
  (combined_time * (fence_area / painter1_time + fence_area / painter2_time - speed_reduction) = fence_area) :=
sorry

end fence_area_calculation_l2084_208491


namespace urn_probability_theorem_l2084_208452

/-- The number of green balls in the first urn -/
def green1 : ℕ := 6

/-- The number of blue balls in the first urn -/
def blue1 : ℕ := 4

/-- The number of green balls in the second urn -/
def green2 : ℕ := 20

/-- The probability that both drawn balls are of the same color -/
def same_color_prob : ℚ := 65/100

/-- The number of blue balls in the second urn -/
def N : ℕ := 4

/-- The total number of balls in the first urn -/
def total1 : ℕ := green1 + blue1

/-- The total number of balls in the second urn -/
def total2 : ℕ := green2 + N

theorem urn_probability_theorem :
  (green1 : ℚ) / total1 * green2 / total2 + (blue1 : ℚ) / total1 * N / total2 = same_color_prob :=
sorry

end urn_probability_theorem_l2084_208452


namespace arc_length_of_sector_l2084_208422

/-- The arc length of a sector with radius π cm and central angle 120° is 2π²/3 cm. -/
theorem arc_length_of_sector (r : Real) (θ : Real) : 
  r = π → θ = 120 * π / 180 → r * θ = 2 * π^2 / 3 := by
  sorry

end arc_length_of_sector_l2084_208422


namespace complement_subset_l2084_208419

-- Define the set M
def M : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define the set N
def N : Set ℝ := {x | x^2 + x - 6 ≤ 0}

-- Theorem statement
theorem complement_subset : Set.compl N ⊆ Set.compl M := by
  sorry

end complement_subset_l2084_208419


namespace second_number_is_255_l2084_208480

def first_set (x : ℝ) : List ℝ := [28, x, 42, 78, 104]
def second_set (x y : ℝ) : List ℝ := [128, y, 511, 1023, x]

theorem second_number_is_255 
  (x : ℝ)
  (h1 : (first_set x).sum / (first_set x).length = 90)
  (h2 : ∃ y, (second_set x y).sum / (second_set x y).length = 423) :
  ∃ y, (second_set x y).sum / (second_set x y).length = 423 ∧ y = 255 := by
sorry

end second_number_is_255_l2084_208480


namespace andy_wrappers_l2084_208434

theorem andy_wrappers (total : ℕ) (max_wrappers : ℕ) (andy_wrappers : ℕ) :
  total = 49 →
  max_wrappers = 15 →
  total = andy_wrappers + max_wrappers →
  andy_wrappers = 34 := by
sorry

end andy_wrappers_l2084_208434


namespace bernold_can_win_l2084_208464

/-- Represents the game board -/
structure GameBoard :=
  (size : Nat)
  (arnold_moves : Nat → Nat → Bool)
  (bernold_moves : Nat → Nat → Bool)

/-- Defines the game rules -/
def game_rules (board : GameBoard) : Prop :=
  board.size = 2007 ∧
  (∀ x y, board.arnold_moves x y ↔ 
    x + 1 < board.size ∧ y + 1 < board.size ∧ 
    ¬board.bernold_moves x y ∧ ¬board.bernold_moves (x+1) y ∧ 
    ¬board.bernold_moves x (y+1) ∧ ¬board.bernold_moves (x+1) (y+1)) ∧
  (∀ x y, board.bernold_moves x y → x < board.size ∧ y < board.size)

/-- Theorem: Bernold can always win -/
theorem bernold_can_win (board : GameBoard) (h : game_rules board) :
  ∃ (strategy : Nat → Nat → Bool), 
    (∀ x y, strategy x y → board.bernold_moves x y) ∧
    (∀ (arnold_strategy : Nat → Nat → Bool), 
      (∀ x y, arnold_strategy x y → board.arnold_moves x y) →
      (Finset.sum (Finset.product (Finset.range board.size) (Finset.range board.size))
        (fun (x, y) => if arnold_strategy x y then 4 else 0) ≤ 
          (1003 * 1004) / 2)) :=
sorry

end bernold_can_win_l2084_208464


namespace finite_nonempty_set_is_good_l2084_208436

/-- An expression using real numbers, ±, +, ×, and parentheses -/
inductive Expression : Type
| Const : ℝ → Expression
| PlusMinus : Expression → Expression → Expression
| Plus : Expression → Expression → Expression
| Times : Expression → Expression → Expression

/-- The range of an expression -/
def range (e : Expression) : Set ℝ :=
  sorry

/-- A set is good if it's the range of some expression -/
def is_good (S : Set ℝ) : Prop :=
  ∃ e : Expression, range e = S

theorem finite_nonempty_set_is_good (S : Set ℝ) (h₁ : S.Finite) (h₂ : S.Nonempty) :
  is_good S :=
sorry

end finite_nonempty_set_is_good_l2084_208436


namespace mans_speed_in_still_water_l2084_208466

/-- The speed of a man rowing in still water, given his downstream performance and the current speed. -/
theorem mans_speed_in_still_water (current_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  current_speed = 3 →
  distance = 15 / 1000 →
  time = 2.9997600191984644 / 3600 →
  (distance / time) - current_speed = 15 := by
  sorry

end mans_speed_in_still_water_l2084_208466


namespace leahs_birdseed_supply_l2084_208472

/-- Represents the number of weeks Leah can feed her birds without going back to the store -/
def weeks_of_feed (boxes_bought : ℕ) (boxes_in_pantry : ℕ) (parrot_consumption : ℕ) (cockatiel_consumption : ℕ) (grams_per_box : ℕ) : ℕ :=
  let total_boxes := boxes_bought + boxes_in_pantry
  let total_grams := total_boxes * grams_per_box
  let weekly_consumption := parrot_consumption + cockatiel_consumption
  total_grams / weekly_consumption

/-- Theorem stating that Leah can feed her birds for 12 weeks without going back to the store -/
theorem leahs_birdseed_supply : weeks_of_feed 3 5 100 50 225 = 12 := by
  sorry

end leahs_birdseed_supply_l2084_208472


namespace hulk_seventh_jump_exceeds_1km_l2084_208417

def hulk_jump (n : ℕ) : ℝ := 3 * (3 ^ (n - 1))

theorem hulk_seventh_jump_exceeds_1km :
  (∀ k < 7, hulk_jump k ≤ 1000) ∧ hulk_jump 7 > 1000 := by
  sorry

end hulk_seventh_jump_exceeds_1km_l2084_208417


namespace complex_simplification_l2084_208433

theorem complex_simplification (i : ℂ) (h : i * i = -1) : 
  (1 + i) / i = 1 - i := by
  sorry

end complex_simplification_l2084_208433


namespace parrot_days_theorem_l2084_208467

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of phrases the parrot currently knows -/
def current_phrases : ℕ := 17

/-- The number of phrases Georgina teaches the parrot per week -/
def phrases_per_week : ℕ := 2

/-- The number of phrases the parrot knew when Georgina bought it -/
def initial_phrases : ℕ := 3

/-- The number of days Georgina has had the parrot -/
def days_with_parrot : ℕ := 49

theorem parrot_days_theorem :
  (current_phrases - initial_phrases) / phrases_per_week * days_per_week = days_with_parrot := by
  sorry

end parrot_days_theorem_l2084_208467


namespace computer_price_calculation_l2084_208410

theorem computer_price_calculation (P : ℝ) : 
  (P * 1.2 * 0.9 * 1.3 = 351) → P = 250 := by
  sorry

end computer_price_calculation_l2084_208410


namespace product_of_sum_and_sum_of_cubes_l2084_208423

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) 
  (h1 : a + b = 4) 
  (h2 : a^3 + b^3 = 100) : 
  a * b = -3 := by
sorry

end product_of_sum_and_sum_of_cubes_l2084_208423


namespace no_solution_inequalities_l2084_208444

theorem no_solution_inequalities (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ¬∃ x : ℝ, x > a ∧ x < -b := by
sorry

end no_solution_inequalities_l2084_208444


namespace abs_neg_two_eq_two_l2084_208439

theorem abs_neg_two_eq_two : |(-2 : ℤ)| = 2 := by
  sorry

end abs_neg_two_eq_two_l2084_208439


namespace negation_of_proposition_l2084_208425

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) :=
by sorry

end negation_of_proposition_l2084_208425


namespace even_function_implies_a_equals_one_l2084_208494

def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (x - a)

theorem even_function_implies_a_equals_one (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = 1 := by
  sorry

end even_function_implies_a_equals_one_l2084_208494


namespace circles_common_chord_l2084_208488

-- Define the circles
def circle1 (x y a : ℝ) : Prop := (x - a)^2 + (y + 2)^2 = 4
def circle2 (x y b : ℝ) : Prop := (x + b)^2 + (y + 2)^2 = 1

-- Define the condition for intersection
def intersect (a b : ℝ) : Prop := 1 < |a + b| ∧ |a + b| < Real.sqrt 3

-- Define the equation of the common chord
def common_chord (x a b : ℝ) : Prop := (2*a + 2*b)*x + 3 + b^2 - a^2 = 0

-- Theorem statement
theorem circles_common_chord (a b : ℝ) (h : intersect a b) :
  ∀ x y : ℝ, circle1 x y a ∧ circle2 x y b → common_chord x a b :=
sorry

end circles_common_chord_l2084_208488


namespace ken_change_l2084_208459

/-- Represents the grocery purchase and payment scenario --/
def grocery_purchase (steak_price : ℕ) (steak_quantity : ℕ) (eggs_price : ℕ) 
  (milk_price : ℕ) (bagels_price : ℕ) (bill_20 : ℕ) (bill_10 : ℕ) 
  (bill_5 : ℕ) (coin_1 : ℕ) : Prop :=
  let total_cost := steak_price * steak_quantity + eggs_price + milk_price + bagels_price
  let total_paid := 20 * bill_20 + 10 * bill_10 + 5 * bill_5 + coin_1
  total_paid - total_cost = 16

/-- Theorem stating that Ken will receive $16 in change --/
theorem ken_change : grocery_purchase 7 2 3 4 6 1 1 2 3 := by
  sorry

end ken_change_l2084_208459


namespace percentage_boys_playing_soccer_l2084_208483

/-- Proves that the percentage of boys among students playing soccer is 86% -/
theorem percentage_boys_playing_soccer
  (total_students : ℕ)
  (num_boys : ℕ)
  (num_playing_soccer : ℕ)
  (num_girls_not_playing : ℕ)
  (h1 : total_students = 450)
  (h2 : num_boys = 320)
  (h3 : num_playing_soccer = 250)
  (h4 : num_girls_not_playing = 95)
  : (((num_playing_soccer - (total_students - num_boys - num_girls_not_playing)) / num_playing_soccer) : ℚ) = 86 / 100 := by
  sorry


end percentage_boys_playing_soccer_l2084_208483


namespace quartic_roots_sum_l2084_208487

theorem quartic_roots_sum (p q r s : ℂ) : 
  (p^4 = p^2 + p + 2) → 
  (q^4 = q^2 + q + 2) → 
  (r^4 = r^2 + r + 2) → 
  (s^4 = s^2 + s + 2) → 
  p * (q - r)^2 + q * (r - s)^2 + r * (s - p)^2 + s * (p - q)^2 = -6 := by
sorry

end quartic_roots_sum_l2084_208487


namespace parabola_equation_correct_l2084_208465

/-- A parabola with vertex (h, k) and passing through point (x₀, y₀) -/
structure Parabola where
  h : ℝ  -- x-coordinate of vertex
  k : ℝ  -- y-coordinate of vertex
  x₀ : ℝ  -- x-coordinate of point on parabola
  y₀ : ℝ  -- y-coordinate of point on parabola

/-- The equation of a parabola in the form ax^2 + bx + c -/
structure ParabolaEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem: The given parabola equation represents the specified parabola -/
theorem parabola_equation_correct (p : Parabola) (eq : ParabolaEquation) : 
  p.h = 3 ∧ p.k = 5 ∧ p.x₀ = 2 ∧ p.y₀ = 2 ∧
  eq.a = -3 ∧ eq.b = 18 ∧ eq.c = -22 →
  ∀ x y : ℝ, y = eq.a * x^2 + eq.b * x + eq.c ↔ 
    (x = p.h ∧ y = p.k) ∨ 
    (y = eq.a * (x - p.h)^2 + p.k) ∨
    (x = p.x₀ ∧ y = p.y₀) := by
  sorry

end parabola_equation_correct_l2084_208465


namespace set_A_elements_l2084_208450

theorem set_A_elements (A B : Finset ℤ) (h1 : A.card = 4) (h2 : B = {-1, 3, 5, 8}) 
  (h3 : ∀ S : Finset ℤ, S ⊆ A → S.card = 3 → (S.sum id) ∈ B) 
  (h4 : ∀ b ∈ B, ∃ S : Finset ℤ, S ⊆ A ∧ S.card = 3 ∧ S.sum id = b) : 
  A = {-3, 0, 2, 6} := by
sorry

end set_A_elements_l2084_208450


namespace baseball_audience_percentage_l2084_208485

theorem baseball_audience_percentage (total : ℕ) (second_team_percentage : ℚ) (non_supporters : ℕ) :
  total = 50 →
  second_team_percentage = 34 / 100 →
  non_supporters = 3 →
  (total - (total * second_team_percentage).floor - non_supporters : ℚ) / total = 3 / 5 :=
by sorry

end baseball_audience_percentage_l2084_208485


namespace total_family_members_eq_243_l2084_208406

/-- The total number of grandchildren and extended family members for Grandma Olga -/
def total_family_members : ℕ :=
  let daughters := 6
  let sons := 5
  let children_per_daughter := 10 + 9  -- 10 sons + 9 daughters
  let stepchildren_per_daughter := 4
  let children_per_son := 8 + 7  -- 8 daughters + 7 sons
  let inlaws_per_son := 3
  let children_per_inlaw := 2

  daughters * children_per_daughter +
  daughters * stepchildren_per_daughter +
  sons * children_per_son +
  sons * inlaws_per_son * children_per_inlaw

theorem total_family_members_eq_243 :
  total_family_members = 243 := by
  sorry

end total_family_members_eq_243_l2084_208406


namespace series_sum_equals_one_fourth_l2084_208477

/-- The series sum from n=1 to infinity of (3^n) / (1 + 3^n + 3^(n+1) + 3^(2n+1)) equals 1/4 -/
theorem series_sum_equals_one_fourth :
  (∑' n : ℕ, (3 : ℝ)^n / (1 + 3^n + 3^(n+1) + 3^(2*n+1))) = 1 / 4 := by
  sorry

end series_sum_equals_one_fourth_l2084_208477


namespace hiking_campers_l2084_208431

theorem hiking_campers (morning_rowing : ℕ) (afternoon_rowing : ℕ) (total_campers : ℕ)
  (h1 : morning_rowing = 41)
  (h2 : afternoon_rowing = 26)
  (h3 : total_campers = 71)
  : total_campers - (morning_rowing + afternoon_rowing) = 4 := by
  sorry

end hiking_campers_l2084_208431


namespace evaluate_expression_l2084_208456

theorem evaluate_expression (a b : ℤ) (h1 : a = 5) (h2 : b = 3) :
  (a^2 + b)^2 - (a^2 - b)^2 = 300 := by
  sorry

end evaluate_expression_l2084_208456


namespace no_adjacent_standing_probability_l2084_208498

def num_people : ℕ := 10

-- Function to calculate the number of valid arrangements
def valid_arrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => valid_arrangements (n + 1) + valid_arrangements n

def total_outcomes : ℕ := 2^num_people

theorem no_adjacent_standing_probability :
  (valid_arrangements num_people : ℚ) / total_outcomes = 123 / 1024 := by
  sorry

end no_adjacent_standing_probability_l2084_208498


namespace odd_number_of_odd_sided_faces_l2084_208499

-- Define a convex polyhedron
structure ConvexPolyhedron where
  vertices : ℕ
  convex : Bool

-- Define a closed broken line on the polyhedron
structure ClosedBrokenLine where
  polyhedron : ConvexPolyhedron
  passes_all_vertices_once : Bool

-- Define a part of the polyhedron surface
structure SurfacePart where
  polyhedron : ConvexPolyhedron
  broken_line : ClosedBrokenLine
  faces : Finset (Finset ℕ)  -- Each face is represented as a set of its vertices

-- Function to count odd-sided faces in a surface part
def count_odd_sided_faces (part : SurfacePart) : ℕ :=
  (part.faces.filter (λ face => face.card % 2 = 1)).card

-- The main theorem
theorem odd_number_of_odd_sided_faces 
  (poly : ConvexPolyhedron) 
  (line : ClosedBrokenLine) 
  (part : SurfacePart) : 
  poly.vertices = 2003 → 
  poly.convex = true → 
  line.polyhedron = poly → 
  line.passes_all_vertices_once = true → 
  part.polyhedron = poly → 
  part.broken_line = line → 
  count_odd_sided_faces part % 2 = 1 :=
sorry

end odd_number_of_odd_sided_faces_l2084_208499


namespace average_age_increase_l2084_208407

theorem average_age_increase (n : ℕ) (A : ℝ) : 
  n = 10 → 
  ((n * A + 21 + 21 - 10 - 12) / n) - A = 2 :=
by
  sorry

end average_age_increase_l2084_208407


namespace star_composition_l2084_208404

/-- Define the binary operation ★ -/
def star (x y : ℝ) : ℝ := x^2 - 2*y + 1

/-- Theorem: For any real number k, k ★ (k ★ k) = -k^2 + 4k - 1 -/
theorem star_composition (k : ℝ) : star k (star k k) = -k^2 + 4*k - 1 := by
  sorry

end star_composition_l2084_208404


namespace map_length_l2084_208469

theorem map_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 10 → area = 20 → area = width * length → length = 2 := by
sorry

end map_length_l2084_208469
