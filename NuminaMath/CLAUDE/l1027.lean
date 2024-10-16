import Mathlib

namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1027_102765

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 3 - 2) :
  1 - (a - 2) / a / ((a^2 - 4) / (a^2 + a)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1027_102765


namespace NUMINAMATH_CALUDE_horner_rule_evaluation_l1027_102744

def horner_polynomial (x : ℝ) : ℝ :=
  (((((2 * x - 0) * x - 3) * x + 2) * x + 7) * x + 6) * x + 3

theorem horner_rule_evaluation :
  horner_polynomial 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_horner_rule_evaluation_l1027_102744


namespace NUMINAMATH_CALUDE_bob_position_2023_l1027_102701

-- Define the movement pattern
def spiral_move (n : ℕ) : ℤ × ℤ := sorry

-- Define Bob's position after n steps
def bob_position (n : ℕ) : ℤ × ℤ := sorry

-- Theorem statement
theorem bob_position_2023 :
  bob_position 2023 = (0, 43) := sorry

end NUMINAMATH_CALUDE_bob_position_2023_l1027_102701


namespace NUMINAMATH_CALUDE_rectangle_circle_mass_ratio_l1027_102783

/-- Represents the mass of an object -/
structure Mass where
  value : ℝ
  nonneg : 0 ≤ value

/-- Represents an equal-arm scale -/
structure EqualArmScale where
  left : Mass
  right : Mass
  balanced : left.value = right.value

/-- The mass of a rectangle -/
def rectangle_mass : Mass := sorry

/-- The mass of a circle -/
def circle_mass : Mass := sorry

/-- The theorem statement -/
theorem rectangle_circle_mass_ratio 
  (scale : EqualArmScale)
  (h1 : scale.left = Mass.mk (2 * rectangle_mass.value) (by sorry))
  (h2 : scale.right = Mass.mk (6 * circle_mass.value) (by sorry)) :
  rectangle_mass.value = 3 * circle_mass.value :=
sorry

end NUMINAMATH_CALUDE_rectangle_circle_mass_ratio_l1027_102783


namespace NUMINAMATH_CALUDE_royalties_for_420_tax_l1027_102715

/-- Calculates the tax on royalties based on the given rules -/
def calculateTax (royalties : ℕ) : ℚ :=
  if royalties ≤ 800 then 0
  else if royalties ≤ 4000 then (royalties - 800) * 14 / 100
  else royalties * 11 / 100

/-- Theorem stating that 3800 yuan in royalties results in 420 yuan tax -/
theorem royalties_for_420_tax : calculateTax 3800 = 420 := by sorry

end NUMINAMATH_CALUDE_royalties_for_420_tax_l1027_102715


namespace NUMINAMATH_CALUDE_prime_squared_product_l1027_102710

theorem prime_squared_product (p q : ℕ) : 
  Prime p → Prime q → Nat.totient (p^2 * q^2) = 11424 → p^2 * q^2 = 7^2 * 17^2 := by
  sorry

end NUMINAMATH_CALUDE_prime_squared_product_l1027_102710


namespace NUMINAMATH_CALUDE_letter_lock_unsuccessful_attempts_l1027_102728

/-- Represents a letter lock with a given number of rings and letters per ring -/
structure LetterLock where
  num_rings : ℕ
  letters_per_ring : ℕ

/-- Calculates the maximum number of unsuccessful attempts for a given lock -/
def max_unsuccessful_attempts (lock : LetterLock) : ℕ :=
  lock.letters_per_ring ^ lock.num_rings - 1

/-- Theorem stating that a lock with 5 rings and 10 letters per ring has 99,999 unsuccessful attempts -/
theorem letter_lock_unsuccessful_attempts :
  let lock : LetterLock := { num_rings := 5, letters_per_ring := 10 }
  max_unsuccessful_attempts lock = 99999 := by
  sorry

#eval max_unsuccessful_attempts { num_rings := 5, letters_per_ring := 10 }

end NUMINAMATH_CALUDE_letter_lock_unsuccessful_attempts_l1027_102728


namespace NUMINAMATH_CALUDE_set_intersection_equality_l1027_102773

def M : Set ℝ := {x | |x| < 1}
def N : Set ℝ := {x | x^2 - x < 0}

theorem set_intersection_equality : M ∩ N = {x | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l1027_102773


namespace NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l1027_102707

theorem arccos_one_half_equals_pi_third : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l1027_102707


namespace NUMINAMATH_CALUDE_vaccine_development_probabilities_l1027_102736

/-- Success probability of Company A for developing vaccine A -/
def prob_A : ℚ := 2/3

/-- Success probability of Company B for developing vaccine A -/
def prob_B : ℚ := 1/2

/-- The theorem states that given the success probabilities of Company A and Company B,
    1) The probability that both succeed is 1/3
    2) The probability of vaccine A being successfully developed is 5/6 -/
theorem vaccine_development_probabilities :
  (prob_A * prob_B = 1/3) ∧
  (1 - (1 - prob_A) * (1 - prob_B) = 5/6) :=
sorry

end NUMINAMATH_CALUDE_vaccine_development_probabilities_l1027_102736


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l1027_102734

theorem unique_four_digit_number : ∃! n : ℕ,
  1000 ≤ n ∧ n < 10000 ∧
  n % 131 = 112 ∧
  n % 132 = 98 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l1027_102734


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1027_102795

theorem inequality_equivalence (x : ℝ) : (x - 5) / 2 + 1 > x - 3 ↔ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1027_102795


namespace NUMINAMATH_CALUDE_retail_discount_l1027_102798

theorem retail_discount (wholesale_price retail_price : ℝ) 
  (h1 : wholesale_price = 90)
  (h2 : retail_price = 120)
  (h3 : ∃ selling_price, selling_price = wholesale_price * 1.2 ∧ 
                         selling_price = retail_price * (1 - (retail_price - selling_price) / retail_price)) :
  (retail_price - wholesale_price * 1.2) / retail_price = 0.1 := by
sorry

end NUMINAMATH_CALUDE_retail_discount_l1027_102798


namespace NUMINAMATH_CALUDE_unique_solution_l1027_102741

def original_number : Nat := 20222023

theorem unique_solution (n : Nat) :
  (n ≥ 1000000000 ∧ n < 10000000000) ∧  -- 10-digit number
  (∃ (a b : Nat), n = a * 1000000000 + original_number * 10 + b) ∧  -- Formed by adding digits to left and right
  (n % 72 = 0) →  -- Divisible by 72
  n = 3202220232 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1027_102741


namespace NUMINAMATH_CALUDE_repeating_decimal_equation_l1027_102790

/-- A single-digit natural number -/
def SingleDigit (n : ℕ) : Prop := 0 < n ∧ n < 10

/-- Represents a repeating decimal of the form 0.ȳ -/
def RepeatingDecimal (y : ℕ) : ℚ := (y : ℚ) / 9

/-- The main theorem statement -/
theorem repeating_decimal_equation :
  ∀ x y : ℕ, SingleDigit y →
    (x / y + 1 = x + RepeatingDecimal y) ↔ ((x = 1 ∧ y = 3) ∨ (x = 0 ∧ y = 9)) :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_equation_l1027_102790


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1027_102760

theorem sufficient_but_not_necessary (m : ℝ) : 
  (m < -2 → ∀ x : ℝ, x^2 - 2*x - m ≠ 0) ∧
  ¬(∀ m : ℝ, (∀ x : ℝ, x^2 - 2*x - m ≠ 0) → m < -2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1027_102760


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_odd_composite_reverse_l1027_102706

/-- A function that reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- A predicate that checks if a number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- The main theorem statement -/
theorem smallest_two_digit_prime_with_odd_composite_reverse :
  ∃ (n : ℕ), is_two_digit n ∧ Nat.Prime n ∧
  Odd (reverse_digits n) ∧ ¬(Nat.Prime (reverse_digits n)) ∧
  (∀ m : ℕ, is_two_digit m → Nat.Prime m →
    Odd (reverse_digits m) → ¬(Nat.Prime (reverse_digits m)) → n ≤ m) ∧
  n = 19 :=
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_odd_composite_reverse_l1027_102706


namespace NUMINAMATH_CALUDE_existence_of_special_integers_l1027_102755

theorem existence_of_special_integers (k : ℕ+) :
  ∃ x y : ℤ, x % 7 ≠ 0 ∧ y % 7 ≠ 0 ∧ x^2 + 6*y^2 = 7^(k : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_integers_l1027_102755


namespace NUMINAMATH_CALUDE_number_of_arrangements_l1027_102722

/-- Represents a step on the staircase -/
structure Step :=
  (occupants : Finset Char)
  (h : occupants.card ≤ 2)

/-- Represents an arrangement of people on the staircase -/
def Arrangement := Finset Step

/-- The set of all valid arrangements -/
def AllArrangements : Finset Arrangement :=
  sorry

/-- The number of different ways 4 people can stand on 5 steps -/
theorem number_of_arrangements :
  (AllArrangements.filter (fun arr => arr.sum (fun step => step.occupants.card) = 4)).card = 540 :=
sorry

end NUMINAMATH_CALUDE_number_of_arrangements_l1027_102722


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1027_102738

theorem possible_values_of_a (a b c : ℝ) 
  (eq1 : a * b + a + b = c)
  (eq2 : b * c + b + c = a)
  (eq3 : c * a + c + a = b) :
  a = 0 ∨ a = -1 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1027_102738


namespace NUMINAMATH_CALUDE_salem_poem_stanzas_l1027_102747

/-- Represents a poem with a specific structure -/
structure Poem where
  lines_per_stanza : ℕ
  words_per_line : ℕ
  total_words : ℕ

/-- Calculates the number of stanzas in a poem -/
def number_of_stanzas (p : Poem) : ℕ :=
  p.total_words / (p.lines_per_stanza * p.words_per_line)

/-- Theorem: A poem with 10 lines per stanza, 8 words per line, 
    and 1600 total words has 20 stanzas -/
theorem salem_poem_stanzas :
  let p : Poem := ⟨10, 8, 1600⟩
  number_of_stanzas p = 20 := by
  sorry

#check salem_poem_stanzas

end NUMINAMATH_CALUDE_salem_poem_stanzas_l1027_102747


namespace NUMINAMATH_CALUDE_grains_in_gray_parts_grains_in_gray_parts_specific_l1027_102708

/-- Given two circles with the same number of grains in their white parts,
    and their respective total grains, calculate the sum of grains in both gray parts. -/
theorem grains_in_gray_parts
  (white_grains : ℕ)
  (total_grains_1 total_grains_2 : ℕ)
  (h1 : total_grains_1 ≥ white_grains)
  (h2 : total_grains_2 ≥ white_grains) :
  (total_grains_1 - white_grains) + (total_grains_2 - white_grains) = 61 :=
by
  sorry

/-- Specific instance of the theorem with given values -/
theorem grains_in_gray_parts_specific :
  (87 - 68) + (110 - 68) = 61 :=
by
  sorry

end NUMINAMATH_CALUDE_grains_in_gray_parts_grains_in_gray_parts_specific_l1027_102708


namespace NUMINAMATH_CALUDE_andrew_ate_77_donuts_l1027_102784

/-- The number of donuts Andrew ate on Monday -/
def monday_donuts : ℕ := 14

/-- The number of donuts Andrew ate on Tuesday -/
def tuesday_donuts : ℕ := monday_donuts / 2

/-- The number of donuts Andrew ate on Wednesday -/
def wednesday_donuts : ℕ := 4 * monday_donuts

/-- The total number of donuts Andrew ate in three days -/
def total_donuts : ℕ := monday_donuts + tuesday_donuts + wednesday_donuts

/-- Theorem stating that Andrew ate 77 donuts in total -/
theorem andrew_ate_77_donuts : total_donuts = 77 := by
  sorry

end NUMINAMATH_CALUDE_andrew_ate_77_donuts_l1027_102784


namespace NUMINAMATH_CALUDE_tea_canister_production_balance_l1027_102799

/-- Represents the production balance in a factory producing cylindrical tea canisters. -/
theorem tea_canister_production_balance :
  let total_workers : ℕ := 44
  let bodies_per_hour : ℕ := 50
  let bottoms_per_hour : ℕ := 120
  let bottoms_per_body : ℕ := 2
  ∀ x : ℕ,
  x ≤ total_workers →
  (bottoms_per_body * bottoms_per_hour * (total_workers - x) = bodies_per_hour * x) :=
by sorry

end NUMINAMATH_CALUDE_tea_canister_production_balance_l1027_102799


namespace NUMINAMATH_CALUDE_sequence_properties_l1027_102769

def sequence_a (n : ℕ) : ℚ := 1/10 * (3/2)^(n-1) - 2/5 * (-1)^n

def partial_sum (n : ℕ) : ℚ := 3 * sequence_a n + (-1)^n

theorem sequence_properties :
  (sequence_a 1 = 1/2) ∧
  (sequence_a 2 = -1/4) ∧
  (sequence_a 3 = 5/8) ∧
  (∀ n : ℕ, n ≥ 2 → sequence_a n + 2/5 * (-1)^n = 3/2 * (sequence_a (n-1) + 2/5 * (-1)^(n-1))) ∧
  (∀ n : ℕ, partial_sum n = 3 * sequence_a n + (-1)^n) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l1027_102769


namespace NUMINAMATH_CALUDE_cube_roots_of_specific_numbers_l1027_102702

theorem cube_roots_of_specific_numbers :
  (∃ x : ℕ, x^3 = 59319) ∧ (∃ y : ℕ, y^3 = 195112) :=
by
  have h1 : (10 : ℕ)^3 = 1000 := by norm_num
  have h2 : (100 : ℕ)^3 = 1000000 := by norm_num
  sorry

end NUMINAMATH_CALUDE_cube_roots_of_specific_numbers_l1027_102702


namespace NUMINAMATH_CALUDE_problem_solution_l1027_102746

theorem problem_solution (x : ℝ) (h : x = -3007) :
  |(|Real.sqrt ((|x| - x)) - x| - x) - Real.sqrt (|x - x^2|)| = 3084 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1027_102746


namespace NUMINAMATH_CALUDE_existence_of_integers_satisfying_inequality_l1027_102730

theorem existence_of_integers_satisfying_inequality :
  ∃ (a b : ℤ), (2003 : ℝ) < (a : ℝ) + (b : ℝ) * Real.sqrt 2 ∧ 
  (a : ℝ) + (b : ℝ) * Real.sqrt 2 < 2003.01 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_integers_satisfying_inequality_l1027_102730


namespace NUMINAMATH_CALUDE_hiker_problem_l1027_102748

/-- A hiker's walking problem -/
theorem hiker_problem (h : ℕ) : 
  (3 * h) + (4 * (h - 1)) + 15 = 53 → 3 * h = 18 := by
  sorry

end NUMINAMATH_CALUDE_hiker_problem_l1027_102748


namespace NUMINAMATH_CALUDE_intersection_product_l1027_102718

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- Represents a parabola -/
structure Parabola where
  focus : Point

/-- Checks if a point is on the ellipse -/
def on_ellipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Checks if a point is on the parabola -/
def on_parabola (p : Parabola) (pt : Point) : Prop :=
  pt.y^2 = 4 * p.focus.x * pt.x

/-- Theorem statement -/
theorem intersection_product (e : Ellipse) (p : Parabola) 
  (A B P : Point) (h_A : on_ellipse e A ∧ on_parabola p A)
  (h_B : on_ellipse e B ∧ on_parabola p B)
  (h_P : on_ellipse e P)
  (h_quad : A.y > 0 ∧ B.y < 0)
  (h_focus : p.focus.y = 0 ∧ p.focus.x > 0)
  (h_vertex : p.focus.x = e.a^2 / (4 * e.b^2))
  (M N : ℝ) (h_M : ∃ t, A.x + t * (P.x - A.x) = M ∧ A.y + t * (P.y - A.y) = 0)
  (h_N : ∃ t, B.x + t * (P.x - B.x) = N ∧ B.y + t * (P.y - B.y) = 0) :
  M * N = e.a^2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_product_l1027_102718


namespace NUMINAMATH_CALUDE_circle_passes_through_fixed_points_l1027_102739

/-- Quadratic function f(x) = 3x^2 - 4x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := 3 * x^2 - 4 * x + c

/-- Circle equation: x^2 + y^2 + Dx + Ey + F = 0 -/
def circle_equation (D E F x y : ℝ) : Prop :=
  x^2 + y^2 + D*x + E*y + F = 0

/-- Theorem: The circle passing through the intersection points of f(x) with the axes
    also passes through the fixed points (0, 1/3) and (4/3, 1/3) -/
theorem circle_passes_through_fixed_points (c : ℝ) 
  (h1 : 0 < c) (h2 : c < 4/3) : 
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, f c x = 0 ∧ y = 0 → circle_equation D E F x y) ∧ 
    (∀ x y : ℝ, x = 0 ∧ f c 0 = y → circle_equation D E F x y) ∧
    circle_equation D E F 0 (1/3) ∧ 
    circle_equation D E F (4/3) (1/3) := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_fixed_points_l1027_102739


namespace NUMINAMATH_CALUDE_irrational_and_no_negative_square_l1027_102712

-- Define p: 2+√2 is irrational
def p : Prop := Irrational (2 + Real.sqrt 2)

-- Define q: ∃ x ∈ ℝ, x^2 < 0
def q : Prop := ∃ x : ℝ, x^2 < 0

-- Theorem statement
theorem irrational_and_no_negative_square : p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_irrational_and_no_negative_square_l1027_102712


namespace NUMINAMATH_CALUDE_expected_girls_left_of_boys_l1027_102709

theorem expected_girls_left_of_boys (num_boys num_girls : ℕ) 
  (h1 : num_boys = 10) (h2 : num_girls = 7) :
  let total := num_boys + num_girls
  let expected_value := (num_girls : ℚ) / (total + 1 : ℚ)
  expected_value = 7 / 11 := by
  sorry

end NUMINAMATH_CALUDE_expected_girls_left_of_boys_l1027_102709


namespace NUMINAMATH_CALUDE_min_cubes_for_majority_interior_min_total_cubes_l1027_102725

/-- A function that calculates the number of interior cubes in a cube of side length n -/
def interior_cubes (n : ℕ) : ℕ := (n - 2)^3

/-- A function that calculates the total number of unit cubes in a cube of side length n -/
def total_cubes (n : ℕ) : ℕ := n^3

/-- The minimum side length of a cube where more than half of the cubes are interior -/
def min_side_length : ℕ := 10

theorem min_cubes_for_majority_interior :
  (∀ k < min_side_length, 2 * interior_cubes k ≤ total_cubes k) ∧
  2 * interior_cubes min_side_length > total_cubes min_side_length :=
by sorry

theorem min_total_cubes : total_cubes min_side_length = 1000 :=
by sorry

end NUMINAMATH_CALUDE_min_cubes_for_majority_interior_min_total_cubes_l1027_102725


namespace NUMINAMATH_CALUDE_problem_statement_l1027_102775

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a + b = 2 → a * b < m / 2) ↔ m > 2) ∧
  (∀ x : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a + b = 2 → 9 / a + 1 / b ≥ |x - 1| + |x + 2|) ↔ -9/2 ≤ x ∧ x ≤ 7/2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1027_102775


namespace NUMINAMATH_CALUDE_seven_day_payment_possible_l1027_102721

/-- Represents the state of rings at any given time --/
structure RingState :=
  (single : ℕ)    -- number of single rings
  (double : ℕ)    -- number of chains with 2 rings
  (quadruple : ℕ) -- number of chains with 4 rings

/-- Represents a daily transaction --/
inductive Transaction
  | give_single
  | give_double
  | give_quadruple
  | return_single
  | return_double

/-- Applies a transaction to a RingState --/
def apply_transaction (state : RingState) (t : Transaction) : RingState :=
  match t with
  | Transaction.give_single => ⟨state.single - 1, state.double, state.quadruple⟩
  | Transaction.give_double => ⟨state.single, state.double - 1, state.quadruple⟩
  | Transaction.give_quadruple => ⟨state.single, state.double, state.quadruple - 1⟩
  | Transaction.return_single => ⟨state.single + 1, state.double, state.quadruple⟩
  | Transaction.return_double => ⟨state.single, state.double + 1, state.quadruple⟩

/-- Checks if a sequence of transactions is valid for a given initial state --/
def is_valid_sequence (initial : RingState) (transactions : List Transaction) : Prop :=
  ∀ (n : ℕ), n < transactions.length →
    let state := transactions.take n.succ
      |> List.foldl apply_transaction initial
    state.single ≥ 0 ∧ state.double ≥ 0 ∧ state.quadruple ≥ 0

/-- Checks if a sequence of transactions results in a net payment of one ring per day --/
def is_daily_payment (transactions : List Transaction) : Prop :=
  transactions.foldl (λ acc t =>
    match t with
    | Transaction.give_single => acc + 1
    | Transaction.give_double => acc + 2
    | Transaction.give_quadruple => acc + 4
    | Transaction.return_single => acc - 1
    | Transaction.return_double => acc - 2
  ) 0 = 1

/-- The main theorem: it is possible to pay for 7 days using a chain of 7 rings, cutting only one --/
theorem seven_day_payment_possible : ∃ (transactions : List Transaction),
  transactions.length = 7 ∧
  is_valid_sequence ⟨1, 1, 1⟩ transactions ∧
  (∀ (n : ℕ), n < 7 → is_daily_payment (transactions.take (n + 1))) :=
sorry

end NUMINAMATH_CALUDE_seven_day_payment_possible_l1027_102721


namespace NUMINAMATH_CALUDE_angle_complement_l1027_102785

/-- Given an angle α of 63°21', its complement is 26°39' -/
theorem angle_complement (α : Real) : α = 63 + 21 / 60 → 90 - α = 26 + 39 / 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_l1027_102785


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l1027_102720

theorem cost_of_dozen_pens 
  (total_cost : ℕ) 
  (pencil_count : ℕ) 
  (pen_cost : ℕ) 
  (pen_pencil_ratio : ℚ) :
  total_cost = 260 →
  pencil_count = 5 →
  pen_cost = 65 →
  pen_pencil_ratio = 5 / 1 →
  (12 : ℕ) * pen_cost = 780 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l1027_102720


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l1027_102781

theorem smallest_integer_satisfying_inequality :
  ∃ n : ℤ, (∀ m : ℤ, m^2 - 15*m + 56 ≤ 0 → n ≤ m) ∧ (n^2 - 15*n + 56 ≤ 0) ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l1027_102781


namespace NUMINAMATH_CALUDE_tan_equation_solution_l1027_102766

theorem tan_equation_solution (θ : Real) (h1 : 0 < θ) (h2 : θ < Real.pi / 6)
  (h3 : Real.tan θ + Real.tan (2 * θ) + Real.tan (4 * θ) = 0) :
  Real.tan θ = 1 / Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_equation_solution_l1027_102766


namespace NUMINAMATH_CALUDE_prop_1_prop_4_l1027_102786

-- Define the types for lines and planes
axiom Line : Type
axiom Plane : Type

-- Define the relations
axiom perpendicular : Line → Plane → Prop
axiom parallel_line_plane : Line → Plane → Prop
axiom parallel_plane : Plane → Plane → Prop
axiom perpendicular_line : Line → Line → Prop

-- Define variables
variable (m n : Line)
variable (α β γ : Plane)

-- Axiom: m and n are different lines
axiom m_neq_n : m ≠ n

-- Axiom: α, β, and γ are different planes
axiom α_neq_β : α ≠ β
axiom α_neq_γ : α ≠ γ
axiom β_neq_γ : β ≠ γ

-- Proposition 1
theorem prop_1 : perpendicular m α → parallel_line_plane n α → perpendicular_line m n :=
sorry

-- Proposition 4
theorem prop_4 : parallel_plane α β → parallel_plane β γ → perpendicular m α → perpendicular m γ :=
sorry

end NUMINAMATH_CALUDE_prop_1_prop_4_l1027_102786


namespace NUMINAMATH_CALUDE_flower_shop_rearrangement_l1027_102787

theorem flower_shop_rearrangement (initial_bunches : ℕ) (initial_flowers_per_bunch : ℕ) (new_flowers_per_bunch : ℕ) :
  initial_bunches = 8 →
  initial_flowers_per_bunch = 9 →
  new_flowers_per_bunch = 12 →
  (initial_bunches * initial_flowers_per_bunch) / new_flowers_per_bunch = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_flower_shop_rearrangement_l1027_102787


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l1027_102700

theorem cube_sum_and_reciprocal (x R S : ℝ) (hx : x ≠ 0) :
  (x + 1 / x = R) → (x^3 + 1 / x^3 = S) → (S = R^3 - 3 * R) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l1027_102700


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l1027_102753

theorem triangle_third_side_length 
  (a b : ℝ) 
  (γ : ℝ) 
  (ha : a = 10) 
  (hb : b = 15) 
  (hγ : γ = 150 * π / 180) :
  ∃ c : ℝ, c^2 = a^2 + b^2 - 2*a*b*Real.cos γ ∧ c = Real.sqrt (325 + 150 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l1027_102753


namespace NUMINAMATH_CALUDE_max_two_scoop_sundaes_l1027_102737

theorem max_two_scoop_sundaes (n : ℕ) (h : n = 8) : 
  Nat.choose n 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_max_two_scoop_sundaes_l1027_102737


namespace NUMINAMATH_CALUDE_perpendicular_implies_m_eq_half_parallel_implies_m_eq_neg_one_l1027_102793

/-- Two lines in the plane -/
structure Lines (m : ℝ) where
  l1 : ℝ → ℝ → Prop
  l2 : ℝ → ℝ → Prop
  eq1 : ∀ x y, l1 x y ↔ x + m * y + 6 = 0
  eq2 : ∀ x y, l2 x y ↔ (m - 2) * x + 3 * y + 2 * m = 0

/-- The lines are perpendicular -/
def Perpendicular (m : ℝ) (lines : Lines m) : Prop :=
  (-1 / m) * ((m - 2) / 3) = -1

/-- The lines are parallel -/
def Parallel (m : ℝ) (lines : Lines m) : Prop :=
  -1 / m = (m - 2) / 3

theorem perpendicular_implies_m_eq_half (m : ℝ) (lines : Lines m) :
  Perpendicular m lines → m = 1 / 2 := by
  sorry

theorem parallel_implies_m_eq_neg_one (m : ℝ) (lines : Lines m) :
  Parallel m lines → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_implies_m_eq_half_parallel_implies_m_eq_neg_one_l1027_102793


namespace NUMINAMATH_CALUDE_marvelous_class_size_l1027_102751

theorem marvelous_class_size :
  ∀ (girls : ℕ) (boys : ℕ) (jelly_beans : ℕ),
    -- Each girl received twice as many jelly beans as there were girls
    (2 * girls * girls +
    -- Each boy received three times as many jelly beans as there were boys
    3 * boys * boys = 
    -- Total jelly beans given out
    jelly_beans) →
    -- She brought 645 jelly beans and had 3 left
    (jelly_beans = 645 - 3) →
    -- The number of boys was three more than twice the number of girls
    (boys = 2 * girls + 3) →
    -- The total number of students
    (girls + boys = 18) := by
  sorry

end NUMINAMATH_CALUDE_marvelous_class_size_l1027_102751


namespace NUMINAMATH_CALUDE_complex_cube_roots_sum_l1027_102762

theorem complex_cube_roots_sum (a b c : ℂ) 
  (sum_condition : a + b + c = 3)
  (sum_squares_condition : a^2 + b^2 + c^2 = 3)
  (sum_cubes_condition : a^3 + b^3 + c^3 = 6) :
  (a - 1)^2023 + (b - 1)^2023 + (c - 1)^2023 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_roots_sum_l1027_102762


namespace NUMINAMATH_CALUDE_count_subsets_correct_l1027_102771

/-- Given a natural number n, this function returns the number of two-tuples (X, Y) 
    of subsets of {1, 2, ..., n} such that max X > min Y -/
def count_subsets (n : ℕ) : ℕ := 
  2^(2*n) - (n+1) * 2^n

/-- Theorem stating that count_subsets gives the correct number of two-tuples -/
theorem count_subsets_correct (n : ℕ) : 
  count_subsets n = (Finset.powerset (Finset.range n)).card * 
                    (Finset.powerset (Finset.range n)).card - 
                    (Finset.filter (fun p : Finset ℕ × Finset ℕ => 
                      p.1.max ≤ p.2.min) 
                      ((Finset.powerset (Finset.range n)).product 
                       (Finset.powerset (Finset.range n)))).card :=
  sorry

#eval count_subsets 3  -- Example usage

end NUMINAMATH_CALUDE_count_subsets_correct_l1027_102771


namespace NUMINAMATH_CALUDE_paul_takes_remaining_l1027_102788

def initial_sweets : ℕ := 22

def jack_takes (total : ℕ) : ℕ := total / 2 + 4

theorem paul_takes_remaining (paul_takes : ℕ) : 
  paul_takes = initial_sweets - jack_takes initial_sweets := by
  sorry

end NUMINAMATH_CALUDE_paul_takes_remaining_l1027_102788


namespace NUMINAMATH_CALUDE_tape_division_l1027_102759

theorem tape_division (total_tape : ℚ) (num_packages : ℕ) :
  total_tape = 7 / 12 ∧ num_packages = 5 →
  total_tape / num_packages = 7 / 60 := by
  sorry

end NUMINAMATH_CALUDE_tape_division_l1027_102759


namespace NUMINAMATH_CALUDE_max_value_constraint_l1027_102763

theorem max_value_constraint (x y : ℝ) : 
  x^2 + y^2 = 20*x + 9*y + 9 → (4*x + 3*y ≤ 83) ∧ ∃ x y, x^2 + y^2 = 20*x + 9*y + 9 ∧ 4*x + 3*y = 83 := by
  sorry

end NUMINAMATH_CALUDE_max_value_constraint_l1027_102763


namespace NUMINAMATH_CALUDE_china_space_station_orbit_height_scientific_notation_l1027_102731

theorem china_space_station_orbit_height_scientific_notation :
  let orbit_height : ℝ := 400000
  orbit_height = 4 * (10 : ℝ)^5 := by sorry

end NUMINAMATH_CALUDE_china_space_station_orbit_height_scientific_notation_l1027_102731


namespace NUMINAMATH_CALUDE_square_difference_l1027_102724

theorem square_difference (x y k c : ℝ) 
  (h1 : x * y = k) 
  (h2 : 1 / x^2 + 1 / y^2 = c) : 
  (x - y)^2 = c * k^2 - 2 * k := by
sorry

end NUMINAMATH_CALUDE_square_difference_l1027_102724


namespace NUMINAMATH_CALUDE_not_prime_sum_product_l1027_102754

theorem not_prime_sum_product (a b c d : ℤ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d > 0)
  (h5 : a * c + b * d = (b + d + a - c) * (b + d - a + c)) :
  ¬ (Nat.Prime (a * b + c * d).natAbs) := by
sorry

end NUMINAMATH_CALUDE_not_prime_sum_product_l1027_102754


namespace NUMINAMATH_CALUDE_log_base_10_derivative_l1027_102780

theorem log_base_10_derivative (x : ℝ) (h : x > 0) : 
  deriv (fun x => Real.log x / Real.log 10) x = 1 / (x * Real.log 10) := by
sorry

end NUMINAMATH_CALUDE_log_base_10_derivative_l1027_102780


namespace NUMINAMATH_CALUDE_w_squared_value_l1027_102778

theorem w_squared_value (w : ℝ) (h : (w + 13)^2 = (3*w + 7)*(2*w + 4)) : w^2 = 141/5 := by
  sorry

end NUMINAMATH_CALUDE_w_squared_value_l1027_102778


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1027_102705

-- Define the quadratic function
def f (a b x : ℝ) := x^2 - a*x + b

-- Define the solution set
def solution_set (a b : ℝ) := {x : ℝ | f a b x < 0}

-- State the theorem
theorem quadratic_inequality_solution (a b : ℝ) :
  solution_set a b = {x : ℝ | -1 < x ∧ x < 3} →
  a + b = -1 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1027_102705


namespace NUMINAMATH_CALUDE_gcd_property_l1027_102770

theorem gcd_property (a : ℕ) (h : ∀ n : ℤ, (Int.gcd (a * n + 1) (2 * n + 1) = 1)) :
  (∀ n : ℤ, Int.gcd (a - 2) (2 * n + 1) = 1) ∧
  (a = 1 ∨ ∃ m : ℕ, a = 2 + 2^m) := by
  sorry

end NUMINAMATH_CALUDE_gcd_property_l1027_102770


namespace NUMINAMATH_CALUDE_line_proof_l1027_102711

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x + 3*y + 1 = 0
def line2 (x y : ℝ) : Prop := x - 3*y + 4 = 0
def line3 (x y : ℝ) : Prop := 3*x + 4*y - 7 = 0

-- Define the result line
def result_line (x y : ℝ) : Prop := y = (4/3)*x + (1/9)

-- Theorem statement
theorem line_proof :
  ∃ (x₀ y₀ : ℝ),
    (line1 x₀ y₀ ∧ line2 x₀ y₀) ∧
    (result_line x₀ y₀) ∧
    (∀ (x y : ℝ), line3 x y → (y - y₀) = -(3/4) * (x - x₀)) :=
by sorry

end NUMINAMATH_CALUDE_line_proof_l1027_102711


namespace NUMINAMATH_CALUDE_max_students_is_eight_l1027_102740

def knows (n : ℕ) : (Fin n → Fin n → Prop) → Prop :=
  λ f => ∀ (i j : Fin n), i ≠ j → f i j = f j i

def satisfies_conditions (n : ℕ) (f : Fin n → Fin n → Prop) : Prop :=
  knows n f ∧
  (∀ (a b c : Fin n), a ≠ b ∧ b ≠ c ∧ a ≠ c → 
    f a b ∨ f b c ∨ f a c) ∧
  (∀ (a b c d : Fin n), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d → 
    (¬f a b ∧ ¬f c d) ∨ (¬f a c ∧ ¬f b d) ∨ (¬f a d ∧ ¬f b c))

theorem max_students_is_eight :
  (∃ (f : Fin 8 → Fin 8 → Prop), satisfies_conditions 8 f) ∧
  (∀ n > 8, ¬∃ (f : Fin n → Fin n → Prop), satisfies_conditions n f) :=
sorry

end NUMINAMATH_CALUDE_max_students_is_eight_l1027_102740


namespace NUMINAMATH_CALUDE_right_triangle_sides_l1027_102797

-- Define the triangle
structure RightTriangle where
  a : ℝ  -- first leg
  b : ℝ  -- second leg
  c : ℝ  -- hypotenuse
  right_angle : a^2 + b^2 = c^2  -- Pythagorean theorem

-- Define the circumscribed and inscribed circle radii
def circumradius : ℝ := 15
def inradius : ℝ := 6

-- Theorem statement
theorem right_triangle_sides : ∃ (t : RightTriangle),
  t.c = 2 * circumradius ∧
  inradius = (t.a + t.b - t.c) / 2 ∧
  ((t.a = 18 ∧ t.b = 24) ∨ (t.a = 24 ∧ t.b = 18)) ∧
  t.c = 30 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l1027_102797


namespace NUMINAMATH_CALUDE_trig_identity_l1027_102756

theorem trig_identity (α : Real) (h : 2 * Real.sin α + Real.cos α = 0) :
  2 * Real.sin α ^ 2 - 3 * Real.sin α * Real.cos α - 5 * Real.cos α ^ 2 = -12/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1027_102756


namespace NUMINAMATH_CALUDE_johns_average_speed_l1027_102719

/-- Calculates the overall average speed given two activities with their respective durations and speeds. -/
def overall_average_speed (duration1 duration2 : ℚ) (speed1 speed2 : ℚ) : ℚ :=
  (duration1 * speed1 + duration2 * speed2) / (duration1 + duration2)

/-- Proves that John's overall average speed is 11.6 mph given his scooter ride and jog. -/
theorem johns_average_speed :
  let scooter_duration : ℚ := 40 / 60  -- 40 minutes in hours
  let scooter_speed : ℚ := 20  -- 20 mph
  let jog_duration : ℚ := 60 / 60  -- 60 minutes in hours
  let jog_speed : ℚ := 6  -- 6 mph
  overall_average_speed scooter_duration jog_duration scooter_speed jog_speed = 58 / 5 := by
  sorry

#eval (58 : ℚ) / 5  -- Should evaluate to 11.6

end NUMINAMATH_CALUDE_johns_average_speed_l1027_102719


namespace NUMINAMATH_CALUDE_bracelets_count_l1027_102792

def total_stones : ℕ := 140
def stones_per_bracelet : ℕ := 14

theorem bracelets_count : total_stones / stones_per_bracelet = 10 := by
  sorry

end NUMINAMATH_CALUDE_bracelets_count_l1027_102792


namespace NUMINAMATH_CALUDE_roots_equal_opposite_signs_l1027_102782

theorem roots_equal_opposite_signs (a b c m : ℝ) :
  (∃ x y : ℝ, x ≠ 0 ∧ y = -x ∧
    (x^2 - b*x) / (a*x - c) = (m - 1) / (m + 1) ∧
    (y^2 - b*y) / (a*y - c) = (m - 1) / (m + 1)) →
  m = (a - b) / (a + b) := by
sorry

end NUMINAMATH_CALUDE_roots_equal_opposite_signs_l1027_102782


namespace NUMINAMATH_CALUDE_max_term_binomial_expansion_l1027_102772

theorem max_term_binomial_expansion :
  let n : ℕ := 212
  let x : ℝ := Real.sqrt 11
  let term (k : ℕ) : ℝ := (n.choose k) * (x ^ k)
  ∃ k : ℕ, k = 163 ∧ ∀ j : ℕ, j ≠ k → j ≤ n → term k ≥ term j :=
by sorry

end NUMINAMATH_CALUDE_max_term_binomial_expansion_l1027_102772


namespace NUMINAMATH_CALUDE_complex_abs_ratio_bounds_l1027_102794

theorem complex_abs_ratio_bounds (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ (m M : ℝ),
    (∀ z w : ℂ, z ≠ 0 → w ≠ 0 → m ≤ Complex.abs (z + w) / (Complex.abs z + Complex.abs w) ∧
                                 Complex.abs (z + w) / (Complex.abs z + Complex.abs w) ≤ M) ∧
    m = 0 ∧
    M = 1 ∧
    M - m = 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_abs_ratio_bounds_l1027_102794


namespace NUMINAMATH_CALUDE_sin_cos_sum_eighty_forty_l1027_102742

theorem sin_cos_sum_eighty_forty : 
  Real.sin (80 * π / 180) * Real.cos (40 * π / 180) + 
  Real.cos (80 * π / 180) * Real.sin (40 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_cos_sum_eighty_forty_l1027_102742


namespace NUMINAMATH_CALUDE_sqrt_x_plus_inverse_sqrt_x_l1027_102767

theorem sqrt_x_plus_inverse_sqrt_x (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) : 
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_inverse_sqrt_x_l1027_102767


namespace NUMINAMATH_CALUDE_polly_tweets_l1027_102768

/-- Represents the tweet rate per minute for different states of Polly --/
structure TweetRates where
  happy : Nat
  hungry : Nat
  mirror : Nat

/-- Represents the duration in minutes for different activities --/
structure ActivityDurations where
  happy : Nat
  hungry : Nat
  mirror : Nat

/-- Calculates the total number of tweets based on rates and durations --/
def totalTweets (rates : TweetRates) (durations : ActivityDurations) : Nat :=
  rates.happy * durations.happy +
  rates.hungry * durations.hungry +
  rates.mirror * durations.mirror

/-- Theorem stating that Polly's total tweets equal 1340 --/
theorem polly_tweets (rates : TweetRates) (durations : ActivityDurations)
    (h1 : rates.happy = 18)
    (h2 : rates.hungry = 4)
    (h3 : rates.mirror = 45)
    (h4 : durations.happy = 20)
    (h5 : durations.hungry = 20)
    (h6 : durations.mirror = 20) :
    totalTweets rates durations = 1340 := by
  sorry


end NUMINAMATH_CALUDE_polly_tweets_l1027_102768


namespace NUMINAMATH_CALUDE_tangent_point_relation_l1027_102729

-- Define the curve and tangent line
def curve (x a b : ℝ) : ℝ := x^3 + a*x + b
def tangent_line (x k : ℝ) : ℝ := k*x + 1

-- State the theorem
theorem tangent_point_relation (a b k : ℝ) : 
  (∃ x y, x = 1 ∧ y = 3 ∧ 
    curve x a b = y ∧ 
    tangent_line x k = y ∧
    (∀ x', curve x' a b = tangent_line x' k → x' = x)) →
  2*a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_point_relation_l1027_102729


namespace NUMINAMATH_CALUDE_time_to_cover_distance_l1027_102779

/-- Given a constant rate of movement and a remaining distance, prove that the time to cover the remaining distance can be calculated by dividing the remaining distance by the rate. -/
theorem time_to_cover_distance (rate : ℝ) (distance : ℝ) (time : ℝ) : 
  rate > 0 → distance > 0 → time = distance / rate → time * rate = distance := by sorry

end NUMINAMATH_CALUDE_time_to_cover_distance_l1027_102779


namespace NUMINAMATH_CALUDE_simplify_fraction_l1027_102761

theorem simplify_fraction (m : ℝ) (h : m ≠ 3) :
  (m / (m - 3) + 2 / (3 - m)) / ((m - 2) / (m^2 - 6*m + 9)) = m - 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1027_102761


namespace NUMINAMATH_CALUDE_batsman_110_run_inning_l1027_102713

/-- Represents a batsman's scoring history -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  deriving Repr

/-- Calculate the average score of a batsman -/
def average (b : Batsman) : ℚ :=
  b.totalRuns / b.innings

/-- The inning where the batsman scores 110 runs -/
def scoreInning (b : Batsman) : ℕ :=
  b.innings + 1

theorem batsman_110_run_inning (b : Batsman) 
  (h1 : average (⟨b.innings + 1, b.totalRuns + 110⟩ : Batsman) = 60)
  (h2 : average (⟨b.innings + 1, b.totalRuns + 110⟩ : Batsman) - average b = 5) :
  scoreInning b = 11 := by
  sorry

#eval scoreInning ⟨10, 550⟩

end NUMINAMATH_CALUDE_batsman_110_run_inning_l1027_102713


namespace NUMINAMATH_CALUDE_checkerboard_coverage_l1027_102745

/-- Represents a checkerboard -/
structure Checkerboard where
  rows : ℕ
  cols : ℕ

/-- Determines if a checkerboard can be covered by dominoes -/
def can_be_covered (board : Checkerboard) : Prop :=
  Even (board.rows * board.cols)

/-- Theorem stating that a checkerboard can be covered if and only if its area is even -/
theorem checkerboard_coverage (board : Checkerboard) :
  can_be_covered board ↔ Even (board.rows * board.cols) := by sorry

end NUMINAMATH_CALUDE_checkerboard_coverage_l1027_102745


namespace NUMINAMATH_CALUDE_inscribed_trapezoid_theorem_l1027_102704

/-- An isosceles trapezoid inscribed in a circle -/
structure InscribedTrapezoid where
  r : ℝ  -- radius of the circle
  a : ℝ  -- half of the shorter base
  c : ℝ  -- half of the longer base
  h : 0 < r ∧ 0 < a ∧ 0 < c ∧ a < c  -- conditions for a valid trapezoid

/-- Theorem: For an isosceles trapezoid inscribed in a circle, r^2 = ac -/
theorem inscribed_trapezoid_theorem (t : InscribedTrapezoid) : t.r^2 = t.a * t.c := by
  sorry

end NUMINAMATH_CALUDE_inscribed_trapezoid_theorem_l1027_102704


namespace NUMINAMATH_CALUDE_boat_against_stream_distance_l1027_102758

/-- The distance traveled by a boat against a stream in one hour, given its speed in still water and its distance traveled along the stream in one hour. -/
theorem boat_against_stream_distance 
  (boat_speed : ℝ) 
  (along_stream_distance : ℝ) 
  (h1 : boat_speed = 7) 
  (h2 : along_stream_distance = 11) : 
  boat_speed - (along_stream_distance - boat_speed) = 3 := by
sorry

end NUMINAMATH_CALUDE_boat_against_stream_distance_l1027_102758


namespace NUMINAMATH_CALUDE_movie_theater_revenue_is_6810_l1027_102733

/-- Represents the revenue calculation for a movie theater --/
def movie_theater_revenue : ℕ := by
  -- Matinee ticket prices and sales
  let matinee_price : ℕ := 5
  let matinee_early_bird_discount : ℚ := 0.5
  let matinee_early_bird_tickets : ℕ := 20
  let matinee_regular_tickets : ℕ := 180

  -- Evening ticket prices and sales
  let evening_price : ℕ := 12
  let evening_group_discount : ℚ := 0.1
  let evening_student_senior_discount : ℚ := 0.25
  let evening_group_tickets : ℕ := 150
  let evening_student_senior_tickets : ℕ := 75
  let evening_regular_tickets : ℕ := 75

  -- 3D ticket prices and sales
  let threeD_price : ℕ := 20
  let threeD_online_surcharge : ℕ := 3
  let threeD_family_discount : ℚ := 0.15
  let threeD_online_tickets : ℕ := 60
  let threeD_family_tickets : ℕ := 25
  let threeD_regular_tickets : ℕ := 15

  -- Late-night ticket prices and sales
  let late_night_price : ℕ := 10
  let late_night_high_demand_increase : ℚ := 0.2
  let late_night_high_demand_tickets : ℕ := 30
  let late_night_regular_tickets : ℕ := 20

  -- Calculate total revenue
  let total_revenue : ℕ := 6810

  exact total_revenue

/-- Theorem stating that the movie theater's revenue on this day is $6810 --/
theorem movie_theater_revenue_is_6810 : movie_theater_revenue = 6810 := by
  sorry

end NUMINAMATH_CALUDE_movie_theater_revenue_is_6810_l1027_102733


namespace NUMINAMATH_CALUDE_harmonic_sum_identity_l1027_102703

def h (n : ℕ) : ℚ :=
  (Finset.range n).sum (fun i => 1 / (i + 1 : ℚ))

theorem harmonic_sum_identity (n : ℕ) (h_n : n ≥ 2) :
  (n : ℚ) + (Finset.range (n - 1)).sum h = n * h n :=
by sorry

end NUMINAMATH_CALUDE_harmonic_sum_identity_l1027_102703


namespace NUMINAMATH_CALUDE_traffic_class_drunk_drivers_l1027_102789

theorem traffic_class_drunk_drivers :
  ∀ (drunk_drivers speeders seatbelt_violators texting_drivers : ℕ),
    speeders = 7 * drunk_drivers - 3 →
    seatbelt_violators = 2 * drunk_drivers →
    texting_drivers = (speeders / 2) + 5 →
    drunk_drivers + speeders + seatbelt_violators + texting_drivers = 180 →
    drunk_drivers = 13 := by
  sorry

end NUMINAMATH_CALUDE_traffic_class_drunk_drivers_l1027_102789


namespace NUMINAMATH_CALUDE_solve_equation_l1027_102752

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((3 / x) + 3) = 4 / 3 → x = -27 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1027_102752


namespace NUMINAMATH_CALUDE_strawberry_picking_total_l1027_102776

/-- The total number of strawberries picked by a group of friends -/
def total_strawberries (lilibeth_baskets mia_baskets jake_baskets natalie_baskets 
  layla_baskets oliver_baskets ava_baskets : ℕ) 
  (lilibeth_per_basket mia_per_basket jake_per_basket natalie_per_basket 
  layla_per_basket oliver_per_basket ava_per_basket : ℕ) : ℕ :=
  lilibeth_baskets * lilibeth_per_basket +
  mia_baskets * mia_per_basket +
  jake_baskets * jake_per_basket +
  natalie_baskets * natalie_per_basket +
  layla_baskets * layla_per_basket +
  oliver_baskets * oliver_per_basket +
  ava_baskets * ava_per_basket

/-- Theorem stating the total number of strawberries picked by the friends -/
theorem strawberry_picking_total : 
  total_strawberries 6 3 4 5 2 7 6 50 65 45 55 80 40 60 = 1750 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_picking_total_l1027_102776


namespace NUMINAMATH_CALUDE_square_land_area_l1027_102774

/-- Given a square land with perimeter p and area A, prove that A = 81 --/
theorem square_land_area (p A : ℝ) : p = 36 ∧ 5 * A = 10 * p + 45 → A = 81 := by
  sorry

end NUMINAMATH_CALUDE_square_land_area_l1027_102774


namespace NUMINAMATH_CALUDE_total_balloons_l1027_102723

theorem total_balloons (tom_balloons sara_balloons alex_balloons : ℕ) 
  (h1 : tom_balloons = 18) 
  (h2 : sara_balloons = 12) 
  (h3 : alex_balloons = 7) : 
  tom_balloons + sara_balloons + alex_balloons = 37 := by
  sorry

end NUMINAMATH_CALUDE_total_balloons_l1027_102723


namespace NUMINAMATH_CALUDE_no_perfect_cube_solution_l1027_102743

theorem no_perfect_cube_solution : ¬∃ (n : ℕ), n > 0 ∧ ∃ (y : ℕ), 3 * n^2 + 3 * n + 7 = y^3 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_cube_solution_l1027_102743


namespace NUMINAMATH_CALUDE_F_is_even_T_is_even_l1027_102777

variable (f : ℝ → ℝ)

def F (x : ℝ) : ℝ := f x * f (-x)

def T (x : ℝ) : ℝ := f x + f (-x)

theorem F_is_even : ∀ x : ℝ, F f x = F f (-x) := by sorry

theorem T_is_even : ∀ x : ℝ, T f x = T f (-x) := by sorry

end NUMINAMATH_CALUDE_F_is_even_T_is_even_l1027_102777


namespace NUMINAMATH_CALUDE_new_perimeter_after_triangle_rotation_l1027_102764

/-- Given a square with perimeter 48 inches and a right isosceles triangle with legs 12 inches,
    prove that removing the triangle and reattaching it results in a figure with perimeter 36 + 12√2 inches -/
theorem new_perimeter_after_triangle_rotation (square_perimeter : ℝ) (triangle_leg : ℝ) : 
  square_perimeter = 48 → triangle_leg = 12 → 
  36 + 12 * Real.sqrt 2 = square_perimeter - triangle_leg + Real.sqrt (2 * triangle_leg^2) :=
by sorry

end NUMINAMATH_CALUDE_new_perimeter_after_triangle_rotation_l1027_102764


namespace NUMINAMATH_CALUDE_median_of_special_list_l1027_102735

def list_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem median_of_special_list : 
  let total_elements : ℕ := list_sum 100
  let median_position : ℕ := total_elements / 2
  let cumulative_count (k : ℕ) : ℕ := list_sum k
  ∃ n : ℕ, 
    cumulative_count n ≥ median_position ∧ 
    cumulative_count (n-1) < median_position ∧
    n = 71 := by
  sorry

#check median_of_special_list

end NUMINAMATH_CALUDE_median_of_special_list_l1027_102735


namespace NUMINAMATH_CALUDE_fourth_hexagon_dots_l1027_102727

/-- Calculates the number of dots in the nth hexagon of the pattern. -/
def hexagonDots (n : ℕ) : ℕ :=
  if n = 0 then 1
  else hexagonDots (n - 1) + 6 * n

/-- The number of dots in the fourth hexagon is 55. -/
theorem fourth_hexagon_dots : hexagonDots 4 = 55 := by sorry

end NUMINAMATH_CALUDE_fourth_hexagon_dots_l1027_102727


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1027_102717

/-- Theorem: For a triangle with sides in the ratio 5:6:7 and the longest side measuring 280 cm, the perimeter is 720 cm. -/
theorem triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Ensure positive side lengths
  a / b = 5 / 6 →          -- Ratio of first two sides
  b / c = 6 / 7 →          -- Ratio of second two sides
  c = 280 →                -- Length of longest side
  a + b + c = 720 :=       -- Perimeter
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1027_102717


namespace NUMINAMATH_CALUDE_inequality_proof_l1027_102716

theorem inequality_proof (a b c d e f : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) 
  (h_cond : |Real.sqrt (a * b) - Real.sqrt (c * d)| ≤ 2) : 
  (e / a + b / e) * (e / c + d / e) ≥ (f / a - b) * (d - f / c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1027_102716


namespace NUMINAMATH_CALUDE_consecutive_numbers_probability_l1027_102726

def set_size : ℕ := 20
def selection_size : ℕ := 5

def prob_consecutive_numbers : ℚ :=
  1 - (Nat.choose (set_size - selection_size + 1) selection_size : ℚ) / (Nat.choose set_size selection_size : ℚ)

theorem consecutive_numbers_probability :
  prob_consecutive_numbers = 232 / 323 := by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_probability_l1027_102726


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_implies_product_l1027_102749

theorem quadratic_inequality_solution_implies_product (a b : ℝ) :
  (∀ x : ℝ, ax^2 + bx + 1 > 0 ↔ -1 < x ∧ x < 1/3) →
  a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_implies_product_l1027_102749


namespace NUMINAMATH_CALUDE_triangle_area_proof_l1027_102750

/-- The curve function f(x) = (x-5)^2 * (x+3) -/
def f (x : ℝ) : ℝ := (x - 5)^2 * (x + 3)

/-- The area of the triangle bounded by the axes and the curve y = f(x) -/
def triangle_area : ℝ := 300

theorem triangle_area_proof : 
  triangle_area = 300 := by sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l1027_102750


namespace NUMINAMATH_CALUDE_union_complement_problem_l1027_102714

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem union_complement_problem : A ∪ (U \ B) = {0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_complement_problem_l1027_102714


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l1027_102757

theorem system_of_equations_solution :
  ∃! (x y : ℝ), (2 * x - y = 3) ∧ (7 * x - 3 * y = 20) ∧ x = 11 ∧ y = 19 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l1027_102757


namespace NUMINAMATH_CALUDE_max_surface_area_rectangular_solid_l1027_102791

theorem max_surface_area_rectangular_solid (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 + c^2 = 36) : 
  2*a*b + 2*a*c + 2*b*c ≤ 72 :=
sorry

end NUMINAMATH_CALUDE_max_surface_area_rectangular_solid_l1027_102791


namespace NUMINAMATH_CALUDE_sum_of_median_scores_l1027_102796

def median_score (scores : List ℕ) : ℕ := sorry

theorem sum_of_median_scores (scores_A scores_B : List ℕ) 
  (h1 : scores_A.length = 9)
  (h2 : scores_B.length = 9)
  (h3 : median_score scores_A = 28)
  (h4 : median_score scores_B = 36) :
  median_score scores_A + median_score scores_B = 64 := by sorry

end NUMINAMATH_CALUDE_sum_of_median_scores_l1027_102796


namespace NUMINAMATH_CALUDE_coefficient_of_y_squared_l1027_102732

theorem coefficient_of_y_squared (a : ℝ) : 
  (∀ y : ℝ, a * y^2 - 8 * y + 55 = 59) → 
  (∃ y : ℝ, y = 2) → 
  a = 5 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_y_squared_l1027_102732
