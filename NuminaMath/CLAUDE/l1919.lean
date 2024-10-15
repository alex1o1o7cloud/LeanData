import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1919_191900

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + 4 * m * x - 4 < 0) → (-1 < m ∧ m < 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1919_191900


namespace NUMINAMATH_CALUDE_number_puzzle_l1919_191986

theorem number_puzzle : ∃ x : ℝ, 35 + 3 * x = 50 ∧ x - 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1919_191986


namespace NUMINAMATH_CALUDE_fifteen_ways_to_divide_books_l1919_191952

/-- The number of ways to divide 6 different books into 3 groups -/
def divide_books : ℕ :=
  Nat.choose 6 4 * Nat.choose 2 1 * Nat.choose 1 1 / Nat.factorial 2

/-- Theorem stating that there are 15 ways to divide the books -/
theorem fifteen_ways_to_divide_books : divide_books = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_ways_to_divide_books_l1919_191952


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1919_191937

theorem sufficient_but_not_necessary (a : ℝ) :
  (((1 / a) > (1 / 4)) → (∀ x : ℝ, a * x^2 + a * x + 1 > 0)) ∧
  (∃ a : ℝ, (∀ x : ℝ, a * x^2 + a * x + 1 > 0) ∧ ((1 / a) ≤ (1 / 4))) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1919_191937


namespace NUMINAMATH_CALUDE_equation_solutions_range_l1919_191995

theorem equation_solutions_range (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, 4 * Real.cos y - Real.cos y ^ 2 + m - 3 = 0) →
  m ∈ Set.Icc 0 8 :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_range_l1919_191995


namespace NUMINAMATH_CALUDE_glen_village_impossibility_l1919_191996

theorem glen_village_impossibility : ¬ ∃ (h c : ℕ), 21 * h + 6 * c = 96 := by
  sorry

#check glen_village_impossibility

end NUMINAMATH_CALUDE_glen_village_impossibility_l1919_191996


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1919_191951

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ ≠ x₁ ∧ 
   ∀ x : ℝ, x^2 - 6*x + k = 0 ↔ (x = x₁ ∨ x = x₂)) → 
  k = 8 ∧ ∃ x₂ : ℝ, x₂ = 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1919_191951


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l1919_191958

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- Define the theorem
theorem lines_perpendicular_to_plane_are_parallel
  (l m : Line) (α : Plane) :
  l ≠ m →
  perpendicular l α →
  perpendicular m α →
  parallel l m :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l1919_191958


namespace NUMINAMATH_CALUDE_possible_last_ball_A_impossible_last_ball_C_l1919_191949

/-- Represents the types of balls in the simulator -/
inductive BallType
  | A
  | B
  | C

/-- Represents the state of the simulator -/
structure SimulatorState :=
  (countA : Nat)
  (countB : Nat)
  (countC : Nat)

/-- Represents a collision between two ball types -/
def collision (a b : BallType) : BallType :=
  match a, b with
  | BallType.A, BallType.A => BallType.C
  | BallType.B, BallType.B => BallType.C
  | BallType.C, BallType.C => BallType.C
  | BallType.A, BallType.B => BallType.C
  | BallType.B, BallType.A => BallType.C
  | BallType.A, BallType.C => BallType.B
  | BallType.C, BallType.A => BallType.B
  | BallType.B, BallType.C => BallType.A
  | BallType.C, BallType.B => BallType.A

/-- The initial state of the simulator -/
def initialState : SimulatorState :=
  { countA := 12, countB := 9, countC := 10 }

/-- Predicate to check if a state has only one ball left -/
def hasOneBallLeft (state : SimulatorState) : Prop :=
  state.countA + state.countB + state.countC = 1

/-- Theorem stating that it's possible for the last ball to be type A -/
theorem possible_last_ball_A :
  ∃ (finalState : SimulatorState),
    hasOneBallLeft finalState ∧ finalState.countA = 1 :=
sorry

/-- Theorem stating that it's impossible for the last ball to be type C -/
theorem impossible_last_ball_C :
  ∀ (finalState : SimulatorState),
    hasOneBallLeft finalState → finalState.countC = 0 :=
sorry

end NUMINAMATH_CALUDE_possible_last_ball_A_impossible_last_ball_C_l1919_191949


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_three_l1919_191950

theorem fraction_zero_implies_x_negative_three (x : ℝ) :
  (x ≠ 3) → ((x^2 - 9) / (x - 3) = 0) → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_three_l1919_191950


namespace NUMINAMATH_CALUDE_absolute_value_greater_than_one_l1919_191985

theorem absolute_value_greater_than_one (a b : ℝ) 
  (h1 : b * (a + b + 1) < 0) 
  (h2 : b * (a + b - 1) < 0) : 
  |a| > 1 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_greater_than_one_l1919_191985


namespace NUMINAMATH_CALUDE_forall_positive_implies_square_plus_greater_than_one_is_false_l1919_191962

theorem forall_positive_implies_square_plus_greater_than_one_is_false :
  ¬(∀ x : ℝ, x > 0 → x^2 + x > 1) :=
sorry

end NUMINAMATH_CALUDE_forall_positive_implies_square_plus_greater_than_one_is_false_l1919_191962


namespace NUMINAMATH_CALUDE_shares_multiple_l1919_191923

/-- Represents the shares of money for three children -/
structure Shares where
  anusha : ℕ
  babu : ℕ
  esha : ℕ

/-- Theorem stating the conditions and the result to be proved -/
theorem shares_multiple (s : Shares) 
  (h1 : 12 * s.anusha = 8 * s.babu)
  (h2 : ∃ k : ℕ, 8 * s.babu = k * s.esha)
  (h3 : s.anusha + s.babu + s.esha = 378)
  (h4 : s.anusha = 84) :
  ∃ k : ℕ, 8 * s.babu = 6 * s.esha := by
  sorry


end NUMINAMATH_CALUDE_shares_multiple_l1919_191923


namespace NUMINAMATH_CALUDE_max_product_with_digits_1_to_5_l1919_191941

def digit := Fin 5

def valid_number (n : ℕ) : Prop :=
  ∃ (d₁ d₂ d₃ : digit), n = d₁.val + 1 + 10 * (d₂.val + 1) + 100 * (d₃.val + 1)

def valid_product (p : ℕ) : Prop :=
  ∃ (n₁ n₂ : ℕ), valid_number n₁ ∧ valid_number n₂ ∧ p = n₁ * n₂

theorem max_product_with_digits_1_to_5 :
  ∀ p, valid_product p → p ≤ 22412 :=
sorry

end NUMINAMATH_CALUDE_max_product_with_digits_1_to_5_l1919_191941


namespace NUMINAMATH_CALUDE_methods_B_and_D_are_correct_l1919_191976

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 2 * x + 5 * y = 18
def equation2 (x y : ℝ) : Prop := 7 * x + 4 * y = 36

-- Define method B
def methodB (x y : ℝ) : Prop :=
  ∃ (z : ℝ), 9 * x + 9 * y = 54 ∧ z * (9 * x + 9 * y) - (2 * x + 5 * y) = z * 54 - 18

-- Define method D
def methodD (x y : ℝ) : Prop :=
  5 * (7 * x + 4 * y) - 4 * (2 * x + 5 * y) = 5 * 36 - 4 * 18

-- Theorem statement
theorem methods_B_and_D_are_correct :
  ∀ x y : ℝ, equation1 x y ∧ equation2 x y → methodB x y ∧ methodD x y :=
sorry

end NUMINAMATH_CALUDE_methods_B_and_D_are_correct_l1919_191976


namespace NUMINAMATH_CALUDE_inequality_proof_l1919_191948

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1919_191948


namespace NUMINAMATH_CALUDE_triangle_properties_l1919_191926

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define the conditions
axiom side_a : a = 4
axiom side_c : c = Real.sqrt 13
axiom sin_relation : Real.sin A = 4 * Real.sin B

-- State the theorem
theorem triangle_properties : b = 1 ∧ C = Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1919_191926


namespace NUMINAMATH_CALUDE_log_equality_implies_ratio_l1919_191918

theorem log_equality_implies_ratio (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (Real.log p / Real.log 4 = Real.log q / Real.log 18) ∧
  (Real.log p / Real.log 4 = Real.log (p + q) / Real.log 25) →
  q / p = 2 - 2/5 := by
  sorry

end NUMINAMATH_CALUDE_log_equality_implies_ratio_l1919_191918


namespace NUMINAMATH_CALUDE_hash_difference_l1919_191991

def hash (x y : ℤ) : ℤ := 2 * x * y - 3 * x + y

theorem hash_difference : (hash 6 4) - (hash 4 6) = -8 := by
  sorry

end NUMINAMATH_CALUDE_hash_difference_l1919_191991


namespace NUMINAMATH_CALUDE_odd_function_g_l1919_191929

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The function f in terms of g -/
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := g x + x^2

theorem odd_function_g (g : ℝ → ℝ) :
  IsOdd (f g) → (f g 1 = 1) → g = fun x ↦ x^5 - x^2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_g_l1919_191929


namespace NUMINAMATH_CALUDE_small_painting_price_is_80_l1919_191987

/-- The price of a small painting given the conditions of Michael's art sale -/
def small_painting_price (large_price : ℕ) (large_sold small_sold : ℕ) (total_earnings : ℕ) : ℕ :=
  (total_earnings - large_price * large_sold) / small_sold

/-- Theorem stating that the price of a small painting is $80 under the given conditions -/
theorem small_painting_price_is_80 :
  small_painting_price 100 5 8 1140 = 80 := by
  sorry

end NUMINAMATH_CALUDE_small_painting_price_is_80_l1919_191987


namespace NUMINAMATH_CALUDE_max_value_part1_l1919_191989

theorem max_value_part1 (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  (1/2) * x * (1 - 2*x) ≤ 1/16 := by sorry

end NUMINAMATH_CALUDE_max_value_part1_l1919_191989


namespace NUMINAMATH_CALUDE_adjacent_same_tribe_l1919_191939

-- Define the four tribes
inductive Tribe
| Human
| Dwarf
| Elf
| Goblin

-- Define the seating arrangement
def Seating := Fin 33 → Tribe

-- Define the condition that humans cannot sit next to goblins
def NoHumanNextToGoblin (s : Seating) : Prop :=
  ∀ i : Fin 33, (s i = Tribe.Human ∧ s (i + 1) = Tribe.Goblin) ∨
                (s i = Tribe.Goblin ∧ s (i + 1) = Tribe.Human) → False

-- Define the condition that elves cannot sit next to dwarves
def NoElfNextToDwarf (s : Seating) : Prop :=
  ∀ i : Fin 33, (s i = Tribe.Elf ∧ s (i + 1) = Tribe.Dwarf) ∨
                (s i = Tribe.Dwarf ∧ s (i + 1) = Tribe.Elf) → False

-- Define the property of having adjacent same tribe representatives
def HasAdjacentSameTribe (s : Seating) : Prop :=
  ∃ i : Fin 33, s i = s (i + 1)

-- State the theorem
theorem adjacent_same_tribe (s : Seating) 
  (no_human_goblin : NoHumanNextToGoblin s) 
  (no_elf_dwarf : NoElfNextToDwarf s) : 
  HasAdjacentSameTribe s :=
sorry

end NUMINAMATH_CALUDE_adjacent_same_tribe_l1919_191939


namespace NUMINAMATH_CALUDE_rice_grains_difference_l1919_191908

def grains_on_square (k : ℕ) : ℕ := 3^k

def sum_of_grains (n : ℕ) : ℕ := 
  3 * (3^n - 1) / 2

theorem rice_grains_difference : 
  grains_on_square 11 - sum_of_grains 9 = 147624 := by
  sorry

end NUMINAMATH_CALUDE_rice_grains_difference_l1919_191908


namespace NUMINAMATH_CALUDE_container_capacity_increase_l1919_191966

/-- Proves that quadrupling all dimensions of a container increases its capacity by a factor of 64 -/
theorem container_capacity_increase (original_capacity new_capacity : ℝ) : 
  original_capacity = 5 → new_capacity = 320 → new_capacity = original_capacity * 64 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_increase_l1919_191966


namespace NUMINAMATH_CALUDE_probability_A_and_B_selected_l1919_191959

/-- The number of students -/
def total_students : ℕ := 5

/-- The number of students to be selected -/
def selected_students : ℕ := 3

/-- The probability of selecting both A and B when choosing 3 students out of 5 -/
def prob_select_A_and_B : ℚ := 3 / 10

theorem probability_A_and_B_selected :
  (Nat.choose (total_students - 2) (selected_students - 2)) / 
  (Nat.choose total_students selected_students) = prob_select_A_and_B :=
sorry

end NUMINAMATH_CALUDE_probability_A_and_B_selected_l1919_191959


namespace NUMINAMATH_CALUDE_min_circular_arrangement_with_shared_digit_l1919_191932

/-- A function that checks if two natural numbers share a common digit in their decimal representation -/
def share_digit (a b : ℕ) : Prop := sorry

/-- A function that represents a circular arrangement of numbers from 1 to n -/
def circular_arrangement (n : ℕ) : (ℕ → ℕ) := sorry

/-- The main theorem stating that 29 is the smallest number satisfying the conditions -/
theorem min_circular_arrangement_with_shared_digit :
  ∀ n : ℕ, n ≥ 2 →
  (∃ arr : ℕ → ℕ, 
    (∀ i : ℕ, arr i ≤ n) ∧ 
    (∀ i : ℕ, share_digit (arr i) (arr (i + 1))) ∧
    (∀ k : ℕ, k ≤ n → ∃ i : ℕ, arr i = k)) →
  n ≥ 29 :=
sorry

end NUMINAMATH_CALUDE_min_circular_arrangement_with_shared_digit_l1919_191932


namespace NUMINAMATH_CALUDE_fraction_division_subtraction_l1919_191998

theorem fraction_division_subtraction :
  (5 / 6 : ℚ) / (9 / 10 : ℚ) - 1 = -2 / 27 := by sorry

end NUMINAMATH_CALUDE_fraction_division_subtraction_l1919_191998


namespace NUMINAMATH_CALUDE_total_dog_legs_l1919_191954

/-- The standard number of legs for a dog -/
def standard_dog_legs : ℕ := 4

/-- The number of dogs in the park -/
def dogs_in_park : ℕ := 109

/-- Theorem: The total number of dog legs in the park is 436 -/
theorem total_dog_legs : dogs_in_park * standard_dog_legs = 436 := by
  sorry

end NUMINAMATH_CALUDE_total_dog_legs_l1919_191954


namespace NUMINAMATH_CALUDE_unique_pizza_combinations_l1919_191925

/-- The number of available toppings -/
def n : ℕ := 8

/-- The number of toppings on each pizza -/
def k : ℕ := 5

/-- Binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem: The number of unique five-topping pizzas with 8 available toppings is 56 -/
theorem unique_pizza_combinations : binomial n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_unique_pizza_combinations_l1919_191925


namespace NUMINAMATH_CALUDE_cycle_price_problem_l1919_191994

/-- Given a cycle sold at a 25% loss for Rs. 2100, prove that the original price was Rs. 2800. -/
theorem cycle_price_problem (selling_price : ℝ) (loss_percentage : ℝ) 
    (h1 : selling_price = 2100)
    (h2 : loss_percentage = 25) : 
  ∃ original_price : ℝ, 
    original_price * (1 - loss_percentage / 100) = selling_price ∧ 
    original_price = 2800 := by
  sorry

end NUMINAMATH_CALUDE_cycle_price_problem_l1919_191994


namespace NUMINAMATH_CALUDE_students_playing_sports_l1919_191901

theorem students_playing_sports (B C : Finset Nat) : 
  (B.card = 7) → 
  (C.card = 8) → 
  ((B ∩ C).card = 5) → 
  ((B ∪ C).card = 10) := by
sorry

end NUMINAMATH_CALUDE_students_playing_sports_l1919_191901


namespace NUMINAMATH_CALUDE_min_quotient_three_digit_number_l1919_191965

theorem min_quotient_three_digit_number (a : ℕ) :
  a ≠ 0 ∧ a ≠ 7 ∧ a ≠ 8 →
  (∀ x : ℕ, x ≠ 0 ∧ x ≠ 7 ∧ x ≠ 8 →
    (100 * a + 78 : ℚ) / (a + 15) ≤ (100 * x + 78 : ℚ) / (x + 15)) →
  (100 * a + 78 : ℚ) / (a + 15) = 11125 / 1000 :=
sorry

end NUMINAMATH_CALUDE_min_quotient_three_digit_number_l1919_191965


namespace NUMINAMATH_CALUDE_expression_is_integer_l1919_191944

theorem expression_is_integer (n : ℤ) : ∃ k : ℤ, (n / 3 : ℚ) + (n^2 / 2 : ℚ) + (n^3 / 6 : ℚ) = k := by
  sorry

end NUMINAMATH_CALUDE_expression_is_integer_l1919_191944


namespace NUMINAMATH_CALUDE_digit_ratio_l1919_191943

theorem digit_ratio (x : ℕ) (a b c : ℕ) : 
  x ≥ 100 ∧ x < 1000 →  -- x is a 3-digit integer
  a > 0 →               -- a > 0
  x = 100 * a + 10 * b + c →  -- x is composed of digits a, b, c
  (999 : ℕ) - x = 241 →  -- difference between largest possible value and x is 241
  (b : ℚ) / c = 5 / 8 :=  -- ratio of b to c is 5:8
by sorry

end NUMINAMATH_CALUDE_digit_ratio_l1919_191943


namespace NUMINAMATH_CALUDE_two_digit_number_solution_l1919_191968

/-- A two-digit number with unit digit greater than tens digit by 2 and less than 30 -/
def TwoDigitNumber (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧  -- two-digit number
  n % 10 = (n / 10) + 2 ∧  -- unit digit greater than tens digit by 2
  n < 30  -- less than 30

theorem two_digit_number_solution :
  ∀ n : ℕ, TwoDigitNumber n → (n = 13 ∨ n = 24) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_solution_l1919_191968


namespace NUMINAMATH_CALUDE_shiela_painting_distribution_l1919_191970

/-- Given Shiela has 18 paintings and 2 grandmothers, prove that each grandmother
    receives 9 paintings when the paintings are distributed equally. -/
theorem shiela_painting_distribution
  (total_paintings : ℕ)
  (num_grandmothers : ℕ)
  (h1 : total_paintings = 18)
  (h2 : num_grandmothers = 2)
  : total_paintings / num_grandmothers = 9 := by
  sorry

end NUMINAMATH_CALUDE_shiela_painting_distribution_l1919_191970


namespace NUMINAMATH_CALUDE_range_of_a_l1919_191947

def A (a : ℝ) : Set ℝ := {x | a * x < 1}
def B : Set ℝ := {x | |x - 1| < 2}

theorem range_of_a (a : ℝ) (h : A a ∪ B = A a) : a ∈ Set.Icc (-1 : ℝ) (1/3) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1919_191947


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1919_191902

theorem chess_tournament_games (n : ℕ) (total_games : ℕ) : 
  n = 5 → total_games = 20 → (n * (n - 1)) / 2 = total_games → n - 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l1919_191902


namespace NUMINAMATH_CALUDE_tax_difference_equals_0_625_l1919_191917

/-- The price of an item before tax -/
def price : ℝ := 50

/-- The higher tax rate -/
def high_rate : ℝ := 0.075

/-- The lower tax rate -/
def low_rate : ℝ := 0.0625

/-- The difference between the two tax amounts -/
def tax_difference : ℝ := price * high_rate - price * low_rate

theorem tax_difference_equals_0_625 : tax_difference = 0.625 := by
  sorry

end NUMINAMATH_CALUDE_tax_difference_equals_0_625_l1919_191917


namespace NUMINAMATH_CALUDE_fraction_classification_l1919_191904

-- Define a fraction as a pair of integers (numerator, denominator)
def Fraction := ℤ × ℤ

-- Define proper fractions
def ProperFraction (f : Fraction) : Prop := f.1.natAbs < f.2.natAbs ∧ f.2 ≠ 0

-- Define improper fractions
def ImproperFraction (f : Fraction) : Prop := f.1.natAbs ≥ f.2.natAbs ∧ f.2 ≠ 0

-- Theorem stating that all fractions are either proper or improper
theorem fraction_classification (f : Fraction) : f.2 ≠ 0 → ProperFraction f ∨ ImproperFraction f :=
sorry

end NUMINAMATH_CALUDE_fraction_classification_l1919_191904


namespace NUMINAMATH_CALUDE_irrational_sum_of_roots_l1919_191972

theorem irrational_sum_of_roots (n : ℤ) : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt (n - 1) + Real.sqrt (n + 1) = p / q :=
sorry

end NUMINAMATH_CALUDE_irrational_sum_of_roots_l1919_191972


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1919_191984

theorem inequality_solution_set (a b : ℝ) : 
  a > 2 → 
  (∀ x, ax + 3 < 2*x + b ↔ x < 0) → 
  b = 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1919_191984


namespace NUMINAMATH_CALUDE_prob_A_hit_given_target_hit_l1919_191903

/-- The probability of A hitting the target -/
def prob_A_hit : ℚ := 3/5

/-- The probability of B hitting the target -/
def prob_B_hit : ℚ := 4/5

/-- The probability of the target being hit by either A or B -/
def prob_target_hit : ℚ := 1 - (1 - prob_A_hit) * (1 - prob_B_hit)

/-- The probability of A hitting the target (regardless of B) -/
def prob_A_hit_total : ℚ := prob_A_hit * (1 - prob_B_hit) + prob_A_hit * prob_B_hit

theorem prob_A_hit_given_target_hit :
  prob_A_hit_total / prob_target_hit = 15/23 :=
sorry

end NUMINAMATH_CALUDE_prob_A_hit_given_target_hit_l1919_191903


namespace NUMINAMATH_CALUDE_red_markers_count_l1919_191975

theorem red_markers_count (total_markers blue_markers : ℕ) 
  (h1 : total_markers = 105)
  (h2 : blue_markers = 64) :
  total_markers - blue_markers = 41 := by
sorry

end NUMINAMATH_CALUDE_red_markers_count_l1919_191975


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1919_191916

-- Define the set P
def P : Set ℝ := {x | x^2 - 7*x + 10 < 0}

-- Define the set Q
def Q : Set ℝ := {y | ∃ x ∈ P, y = x^2 - 8*x + 19}

-- Theorem statement
theorem intersection_of_P_and_Q : P ∩ Q = Set.Icc 3 5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1919_191916


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l1919_191920

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem: A point M with coordinates (a, b), where a < 0 and b > 0, is in the second quadrant -/
theorem point_in_second_quadrant (a b : ℝ) (ha : a < 0) (hb : b > 0) :
  SecondQuadrant ⟨a, b⟩ := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l1919_191920


namespace NUMINAMATH_CALUDE_world_cup_knowledge_competition_l1919_191977

theorem world_cup_knowledge_competition (p_know : ℝ) (p_guess : ℝ) (num_options : ℕ) :
  p_know = 2/3 →
  p_guess = 1/3 →
  num_options = 4 →
  (p_know * 1 + p_guess * (1 / num_options)) / (p_know + p_guess * (1 / num_options)) = 8/9 :=
by sorry

end NUMINAMATH_CALUDE_world_cup_knowledge_competition_l1919_191977


namespace NUMINAMATH_CALUDE_winning_candidate_vote_percentage_l1919_191955

/-- Given an association with total members, votes cast, and the winning candidate's votes as a percentage of total membership, calculate the percentage of votes cast that the winning candidate received. -/
theorem winning_candidate_vote_percentage
  (total_members : ℕ)
  (votes_cast : ℕ)
  (winning_votes_percentage_of_total : ℚ)
  (h1 : total_members = 1600)
  (h2 : votes_cast = 525)
  (h3 : winning_votes_percentage_of_total = 19.6875 / 100) :
  (winning_votes_percentage_of_total * total_members) / votes_cast = 60 / 100 :=
by sorry

end NUMINAMATH_CALUDE_winning_candidate_vote_percentage_l1919_191955


namespace NUMINAMATH_CALUDE_smallest_k_inequality_half_satisfies_inequality_l1919_191953

theorem smallest_k_inequality (k : ℝ) : 
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → Real.sqrt (x * y) + k * Real.sqrt (|x - y|) ≥ (x + y) / 2) ↔ 
  k ≥ (1 / 2 : ℝ) :=
sorry

theorem half_satisfies_inequality : 
  ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → 
    Real.sqrt (x * y) + (1 / 2 : ℝ) * Real.sqrt (|x - y|) ≥ (x + y) / 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_k_inequality_half_satisfies_inequality_l1919_191953


namespace NUMINAMATH_CALUDE_outfit_choices_l1919_191915

/-- The number of colors available for each clothing item -/
def num_colors : ℕ := 6

/-- The number of shirts available -/
def num_shirts : ℕ := num_colors

/-- The number of pants available -/
def num_pants : ℕ := num_colors

/-- The number of hats available -/
def num_hats : ℕ := num_colors

/-- The total number of possible outfit combinations -/
def total_combinations : ℕ := num_shirts * num_pants * num_hats

/-- The number of outfits where all items are the same color -/
def same_color_outfits : ℕ := num_colors

/-- Theorem: The number of outfit choices where not all items are the same color -/
theorem outfit_choices : 
  total_combinations - same_color_outfits = 210 :=
sorry

end NUMINAMATH_CALUDE_outfit_choices_l1919_191915


namespace NUMINAMATH_CALUDE_jonathan_exercise_distance_l1919_191940

/-- Represents Jonathan's exercise routine for a week -/
structure ExerciseRoutine where
  monday_speed : ℝ
  wednesday_speed : ℝ
  friday_speed : ℝ
  total_time : ℝ

/-- Theorem stating that if Jonathan travels the same distance each day and 
    spends a total of 6 hours exercising in a week, given his speeds on different days, 
    he travels 6 miles on each exercise day. -/
theorem jonathan_exercise_distance (routine : ExerciseRoutine) 
  (h1 : routine.monday_speed = 2)
  (h2 : routine.wednesday_speed = 3)
  (h3 : routine.friday_speed = 6)
  (h4 : routine.total_time = 6)
  (h5 : ∃ d : ℝ, d > 0 ∧ 
    d / routine.monday_speed + 
    d / routine.wednesday_speed + 
    d / routine.friday_speed = routine.total_time) :
  ∃ d : ℝ, d = 6 ∧ 
    d / routine.monday_speed + 
    d / routine.wednesday_speed + 
    d / routine.friday_speed = routine.total_time := by
  sorry

end NUMINAMATH_CALUDE_jonathan_exercise_distance_l1919_191940


namespace NUMINAMATH_CALUDE_sequence_product_l1919_191910

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- The line l passing through the origin with normal vector (3,1) -/
def Line (x y : ℝ) : Prop := 3 * x + y = 0

/-- The sequence {a_n} satisfies the condition that (a_{n+1}, a_n) lies on the line for all n -/
def SequenceOnLine (a : Sequence) : Prop := ∀ n : ℕ, Line (a (n + 1)) (a n)

theorem sequence_product (a : Sequence) (h1 : SequenceOnLine a) (h2 : a 2 = 6) :
  a 1 * a 2 * a 3 * a 4 * a 5 = -32 := by
  sorry

end NUMINAMATH_CALUDE_sequence_product_l1919_191910


namespace NUMINAMATH_CALUDE_division_4073_by_38_l1919_191997

theorem division_4073_by_38 : ∃ (q r : ℕ), 4073 = 38 * q + r ∧ r < 38 ∧ q = 107 ∧ r = 7 := by
  sorry

end NUMINAMATH_CALUDE_division_4073_by_38_l1919_191997


namespace NUMINAMATH_CALUDE_purchasing_ways_l1919_191963

/-- The number of different oreo flavors --/
def oreo_flavors : ℕ := 7

/-- The number of different milk flavors --/
def milk_flavors : ℕ := 4

/-- The total number of products they purchase --/
def total_products : ℕ := 4

/-- Charlie's purchasing strategy: no repeats, can buy both oreos and milk --/
def charlie_strategy (k : ℕ) : ℕ := Nat.choose (oreo_flavors + milk_flavors) k

/-- Delta's purchasing strategy: only oreos, can have repeats --/
def delta_strategy (k : ℕ) : ℕ :=
  if k = 0 then 1
  else if k = 1 then oreo_flavors
  else if k = 2 then Nat.choose oreo_flavors 2 + oreo_flavors
  else if k = 3 then Nat.choose oreo_flavors 3 + oreo_flavors * (oreo_flavors - 1) + oreo_flavors
  else Nat.choose oreo_flavors 4 + oreo_flavors * (oreo_flavors - 1) + 
       (oreo_flavors * (oreo_flavors - 1)) / 2 + oreo_flavors

/-- The total number of ways to purchase 4 products --/
def total_ways : ℕ := 
  (charlie_strategy 4 * delta_strategy 0) +
  (charlie_strategy 3 * delta_strategy 1) +
  (charlie_strategy 2 * delta_strategy 2) +
  (charlie_strategy 1 * delta_strategy 3) +
  (charlie_strategy 0 * delta_strategy 4)

theorem purchasing_ways : total_ways = 4054 := by
  sorry

end NUMINAMATH_CALUDE_purchasing_ways_l1919_191963


namespace NUMINAMATH_CALUDE_sum_consecutive_odd_numbers_remainder_l1919_191961

theorem sum_consecutive_odd_numbers_remainder (start : ℕ) (h : start = 10999) :
  (List.sum (List.map (λ i => start + 2 * i) (List.range 7))) % 14 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_consecutive_odd_numbers_remainder_l1919_191961


namespace NUMINAMATH_CALUDE_two_integers_sum_and_lcm_l1919_191983

theorem two_integers_sum_and_lcm : ∃ (m n : ℕ), 
  m > 0 ∧ n > 0 ∧ m + n = 60 ∧ Nat.lcm m n = 273 ∧ m = 21 ∧ n = 39 := by
  sorry

end NUMINAMATH_CALUDE_two_integers_sum_and_lcm_l1919_191983


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_5_range_of_a_no_solution_l1919_191909

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2| + |2*x + 1|

-- Theorem for the solution of f(x) > 5
theorem solution_set_f_greater_than_5 :
  {x : ℝ | f x > 5} = Set.Iio (-4/3) ∪ Set.Ioi 2 := by sorry

-- Theorem for the range of a when 1/(f(x)-4) = a has no solution
theorem range_of_a_no_solution :
  {a : ℝ | ∀ x, 1/(f x - 4) ≠ a} = Set.Ioo (-2/3) 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_5_range_of_a_no_solution_l1919_191909


namespace NUMINAMATH_CALUDE_area_conversion_time_conversion_l1919_191906

-- Define the conversion factors
def square_meters_per_hectare : ℝ := 10000
def minutes_per_hour : ℝ := 60

-- Define the input values
def area_in_square_meters : ℝ := 123000
def time_in_hours : ℝ := 4.25

-- Theorem for area conversion
theorem area_conversion :
  area_in_square_meters / square_meters_per_hectare = 12.3 := by sorry

-- Theorem for time conversion
theorem time_conversion :
  ∃ (whole_hours minutes : ℕ),
    whole_hours = 4 ∧
    minutes = 15 ∧
    time_in_hours = whole_hours + (minutes : ℝ) / minutes_per_hour := by sorry

end NUMINAMATH_CALUDE_area_conversion_time_conversion_l1919_191906


namespace NUMINAMATH_CALUDE_problem_statement_l1919_191973

theorem problem_statement : (1 / ((-5^2)^3)) * ((-5)^8) * Real.sqrt 5 = 5^(5/2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1919_191973


namespace NUMINAMATH_CALUDE_arithmetic_mean_1_5_l1919_191964

theorem arithmetic_mean_1_5 (m : ℝ) : m = (1 + 5) / 2 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_1_5_l1919_191964


namespace NUMINAMATH_CALUDE_jordyn_zrinka_age_ratio_l1919_191942

/-- Given the ages of Mehki, Jordyn, and Zrinka, prove the ratio of Jordyn's to Zrinka's age -/
theorem jordyn_zrinka_age_ratio :
  ∀ (mehki_age jordyn_age zrinka_age : ℕ),
  mehki_age = 22 →
  zrinka_age = 6 →
  mehki_age = jordyn_age + 10 →
  (jordyn_age : ℚ) / (zrinka_age : ℚ) = 2 := by
sorry

end NUMINAMATH_CALUDE_jordyn_zrinka_age_ratio_l1919_191942


namespace NUMINAMATH_CALUDE_gathering_handshakes_l1919_191980

/-- The number of handshakes in a gathering of elves and dwarves -/
def total_handshakes (num_elves num_dwarves : ℕ) : ℕ :=
  let elf_handshakes := num_elves * (num_elves - 1) / 2
  let elf_dwarf_handshakes := num_elves * num_dwarves
  elf_handshakes + elf_dwarf_handshakes

/-- Theorem stating the total number of handshakes in the gathering -/
theorem gathering_handshakes :
  total_handshakes 25 18 = 750 := by
  sorry

#eval total_handshakes 25 18

end NUMINAMATH_CALUDE_gathering_handshakes_l1919_191980


namespace NUMINAMATH_CALUDE_collinear_points_q_value_l1919_191974

/-- 
If the points (7, q), (5, 3), and (1, -1) are collinear, then q = 5.
-/
theorem collinear_points_q_value (q : ℝ) : 
  (∃ (t : ℝ), (7 - 1) = t * (5 - 1) ∧ (q + 1) = t * (3 + 1)) → q = 5 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_q_value_l1919_191974


namespace NUMINAMATH_CALUDE_sock_pairs_count_l1919_191967

/-- The number of ways to choose a pair of socks of different colors -/
def differentColorPairs (white : Nat) (brown : Nat) (blue : Nat) : Nat :=
  white * brown + brown * blue + white * blue

/-- Theorem: Given 5 white socks, 4 brown socks, and 3 blue socks,
    there are 47 ways to choose a pair of socks of different colors -/
theorem sock_pairs_count :
  differentColorPairs 5 4 3 = 47 := by
  sorry

end NUMINAMATH_CALUDE_sock_pairs_count_l1919_191967


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l1919_191927

theorem trigonometric_expression_equality : 
  (Real.sqrt 3 * Real.tan (12 * π / 180) - 3) / 
  ((4 * (Real.cos (12 * π / 180))^2 - 2) * Real.sin (12 * π / 180)) = 
  -4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l1919_191927


namespace NUMINAMATH_CALUDE_sum_of_squares_l1919_191913

theorem sum_of_squares (a b : ℝ) (h1 : a - b = 6) (h2 : a * b = 7) :
  a^2 + b^2 = 50 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1919_191913


namespace NUMINAMATH_CALUDE_smallest_number_with_remainder_l1919_191930

theorem smallest_number_with_remainder (n : ℕ) : 
  300 % 25 = 0 →
  n > 300 →
  n % 25 = 24 →
  (∀ m : ℕ, m > 300 ∧ m % 25 = 24 → n ≤ m) →
  n = 324 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainder_l1919_191930


namespace NUMINAMATH_CALUDE_area_of_triangle_GAB_l1919_191957

-- Define the curve C
def curve_C (x y : ℝ) : Prop := y^2 = 8*x

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - 2

-- Define points P, Q, and G
def point_P : ℝ × ℝ := (2, 0)
def point_Q : ℝ × ℝ := (0, -2)
def point_G : ℝ × ℝ := (-2, 0)

-- Define the theorem
theorem area_of_triangle_GAB :
  ∃ (A B : ℝ × ℝ),
    curve_C A.1 A.2 ∧
    curve_C B.1 B.2 ∧
    line_l A.1 A.2 ∧
    line_l B.1 B.2 ∧
    line_l point_Q.1 point_Q.2 →
    let area := (1/2) * ‖A - B‖ * (2 * Real.sqrt 2)
    area = 16 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_area_of_triangle_GAB_l1919_191957


namespace NUMINAMATH_CALUDE_percentage_of_sikh_boys_l1919_191956

/-- Proves that the percentage of Sikh boys is 10% given the specified conditions --/
theorem percentage_of_sikh_boys
  (total_boys : ℕ)
  (muslim_percentage : ℚ)
  (hindu_percentage : ℚ)
  (other_boys : ℕ)
  (h1 : total_boys = 850)
  (h2 : muslim_percentage = 44/100)
  (h3 : hindu_percentage = 32/100)
  (h4 : other_boys = 119) :
  (total_boys - (muslim_percentage * total_boys + hindu_percentage * total_boys + other_boys)) / total_boys = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_sikh_boys_l1919_191956


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1919_191933

/-- A geometric sequence -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The main theorem -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a) 
  (h_condition : a 1 * a 13 + 2 * (a 7)^2 = 4 * Real.pi) : 
  Real.tan (a 2 * a 12) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1919_191933


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1919_191919

-- Define the original expression
def original_expr (a b : ℝ) : ℝ := 3*a^2*b - (2*a^2*b - (2*a*b - a^2*b) - 4*a^2) - a*b

-- Define the simplified expression
def simplified_expr (a b : ℝ) : ℝ := a*b + 4*a^2

-- Theorem statement
theorem expression_simplification_and_evaluation :
  (∀ a b : ℝ, original_expr a b = simplified_expr a b) ∧
  (original_expr (-3) (-2) = 22) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1919_191919


namespace NUMINAMATH_CALUDE_gcd_4004_10010_l1919_191922

theorem gcd_4004_10010 : Nat.gcd 4004 10010 = 2002 := by
  sorry

end NUMINAMATH_CALUDE_gcd_4004_10010_l1919_191922


namespace NUMINAMATH_CALUDE_ryan_english_hours_l1919_191981

/-- The number of hours Ryan spends on learning Chinese -/
def chinese_hours : ℕ := 2

/-- The number of hours Ryan spends on learning English -/
def english_hours : ℕ := chinese_hours + 4

/-- Theorem: Ryan spends 6 hours on learning English -/
theorem ryan_english_hours : english_hours = 6 := by
  sorry

end NUMINAMATH_CALUDE_ryan_english_hours_l1919_191981


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l1919_191928

/-- The time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 160)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 215) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 :=
by sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l1919_191928


namespace NUMINAMATH_CALUDE_max_ab_value_l1919_191938

theorem max_ab_value (a b : ℝ) : 
  (∃! x, x^2 - 2*a*x - b^2 + 12 ≤ 0) → 
  ∀ c, a*b ≤ c → c ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_ab_value_l1919_191938


namespace NUMINAMATH_CALUDE_rhinestone_project_l1919_191999

theorem rhinestone_project (total : ℚ) : 
  (1 / 3 : ℚ) * total + (1 / 5 : ℚ) * total + 21 = total → 
  total = 45 := by
sorry

end NUMINAMATH_CALUDE_rhinestone_project_l1919_191999


namespace NUMINAMATH_CALUDE_min_dot_product_l1919_191992

/-- Ellipse C with foci at (0,-√3) and (0,√3) passing through (√3/2, 1) -/
def ellipse_C (x y : ℝ) : Prop :=
  y^2 / 4 + x^2 = 1

/-- Parabola E with vertex at (0,0) and focus at (1,0) -/
def parabola_E (x y : ℝ) : Prop :=
  y^2 = 4 * x

/-- Point on parabola E -/
def point_on_E (x y : ℝ) : Prop :=
  parabola_E x y

/-- Line through focus (1,0) with slope k -/
def line_through_focus (k x y : ℝ) : Prop :=
  y = k * (x - 1)

/-- Perpendicular line through focus (1,0) with slope -1/k -/
def perp_line_through_focus (k x y : ℝ) : Prop :=
  y = -1/k * (x - 1)

/-- Theorem: Minimum value of AG · HB is 16 -/
theorem min_dot_product :
  ∃ (min : ℝ),
    (∀ (k x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
      point_on_E x₁ y₁ ∧ point_on_E x₂ y₂ ∧ point_on_E x₃ y₃ ∧ point_on_E x₄ y₄ ∧
      line_through_focus k x₁ y₁ ∧ line_through_focus k x₂ y₂ ∧
      perp_line_through_focus k x₃ y₃ ∧ perp_line_through_focus k x₄ y₄ →
      ((x₁ - x₃) * (x₄ - x₂) + (y₁ - y₃) * (y₄ - y₂) ≥ min)) ∧
    min = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_dot_product_l1919_191992


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1919_191988

theorem inequality_equivalence (x : ℝ) : 
  3 * x - 6 > 12 - 2 * x + x^2 ↔ -1 < x ∧ x < 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1919_191988


namespace NUMINAMATH_CALUDE_total_cost_is_122_4_l1919_191935

/-- Calculates the total cost of Zoe's app usage over a year -/
def total_cost (initial_app_cost monthly_fee annual_discount in_game_cost upgrade_cost membership_discount : ℝ) : ℝ :=
  let first_two_months := 2 * monthly_fee
  let annual_plan_cost := (12 * monthly_fee) * (1 - annual_discount)
  let discounted_in_game := in_game_cost * (1 - membership_discount)
  let discounted_upgrade := upgrade_cost * (1 - membership_discount)
  initial_app_cost + first_two_months + annual_plan_cost + discounted_in_game + discounted_upgrade

/-- Theorem stating that the total cost is $122.4 given the specified conditions -/
theorem total_cost_is_122_4 :
  total_cost 5 8 0.15 10 12 0.1 = 122.4 := by
  sorry

#eval total_cost 5 8 0.15 10 12 0.1

end NUMINAMATH_CALUDE_total_cost_is_122_4_l1919_191935


namespace NUMINAMATH_CALUDE_cos_105_degrees_l1919_191982

theorem cos_105_degrees : Real.cos (105 * Real.pi / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_105_degrees_l1919_191982


namespace NUMINAMATH_CALUDE_sample_size_is_140_l1919_191979

/-- Represents a school with students and a height measurement study -/
structure School where
  total_students : ℕ
  measured_students : ℕ
  measured_students_le_total : measured_students ≤ total_students

/-- The sample size of a height measurement study in a school -/
def sample_size (s : School) : ℕ := s.measured_students

/-- Theorem stating that for a school with 1740 students and 140 measured students, the sample size is 140 -/
theorem sample_size_is_140 (s : School) 
  (h1 : s.total_students = 1740) 
  (h2 : s.measured_students = 140) : 
  sample_size s = 140 := by sorry

end NUMINAMATH_CALUDE_sample_size_is_140_l1919_191979


namespace NUMINAMATH_CALUDE_freshman_count_l1919_191993

theorem freshman_count (f o j s : ℕ) : 
  f * 4 = o * 5 →
  o * 8 = j * 7 →
  j * 7 = s * 9 →
  f + o + j + s = 2158 →
  f = 630 :=
by sorry

end NUMINAMATH_CALUDE_freshman_count_l1919_191993


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l1919_191971

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (distinct : Plane → Plane → Prop)
variable (distinct_lines : Line → Line → Prop)

-- Theorem statement
theorem line_plane_perpendicularity 
  (α β : Plane) (m n : Line) 
  (h_distinct_planes : distinct α β)
  (h_distinct_lines : distinct_lines m n) :
  (parallel m n ∧ perpendicular m α) → perpendicular n α :=
by sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l1919_191971


namespace NUMINAMATH_CALUDE_sequence_property_l1919_191931

def sequence_sum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun i => a (i + 1))

theorem sequence_property (a : ℕ → ℕ) 
  (h : ∀ n : ℕ, n > 0 → sequence_sum a n = 2 * a n - n) :
  (a 1 = 1 ∧ a 2 = 3 ∧ a 3 = 7) ∧
  (∀ n : ℕ, n > 0 → a n = 2^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l1919_191931


namespace NUMINAMATH_CALUDE_vector_operation_proof_l1919_191978

def vector_operation : Prop :=
  let v1 : Fin 2 → ℝ := ![5, -6]
  let v2 : Fin 2 → ℝ := ![-2, 13]
  let v3 : Fin 2 → ℝ := ![1, -2]
  v1 + v2 - 3 • v3 = ![0, 13]

theorem vector_operation_proof : vector_operation := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_proof_l1919_191978


namespace NUMINAMATH_CALUDE_sixth_term_of_sequence_l1919_191990

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r^(n - 1)

theorem sixth_term_of_sequence (a₁ a₂ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 6) :
  geometric_sequence a₁ (a₂ / a₁) 6 = 96 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_sequence_l1919_191990


namespace NUMINAMATH_CALUDE_unique_fraction_representation_l1919_191924

theorem unique_fraction_representation (p : ℕ) (h_prime : Nat.Prime p) (h_greater_than_two : p > 2) :
  ∃! (x y : ℕ), x ≠ y ∧ x > 0 ∧ y > 0 ∧ (2 : ℚ) / p = 1 / x + 1 / y := by
  sorry

end NUMINAMATH_CALUDE_unique_fraction_representation_l1919_191924


namespace NUMINAMATH_CALUDE_box_volume_l1919_191960

/-- A rectangular box with given face areas and length-height relationship has a volume of 120 cubic inches -/
theorem box_volume (l w h : ℝ) (area1 : l * w = 30) (area2 : w * h = 20) (area3 : l * h = 12) (length_height : l = h + 1) :
  l * w * h = 120 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_l1919_191960


namespace NUMINAMATH_CALUDE_performance_orders_count_l1919_191907

/-- The number of ways to select 4 programs from 8 options -/
def total_options : ℕ := 8

/-- The number of programs to be selected -/
def selected_programs : ℕ := 4

/-- The number of special programs (A and B) -/
def special_programs : ℕ := 2

/-- The number of non-special programs -/
def other_programs : ℕ := total_options - special_programs

/-- Calculates the number of performance orders with only one special program -/
def orders_with_one_special : ℕ :=
  special_programs * (Nat.choose other_programs (selected_programs - 1)) * (Nat.factorial selected_programs)

/-- Calculates the number of performance orders with both special programs -/
def orders_with_both_special : ℕ :=
  (Nat.choose other_programs (selected_programs - 2)) * (Nat.factorial 2) * (Nat.factorial (selected_programs - 2))

/-- The total number of valid performance orders -/
def total_orders : ℕ := orders_with_one_special + orders_with_both_special

theorem performance_orders_count :
  total_orders = 2860 :=
sorry

end NUMINAMATH_CALUDE_performance_orders_count_l1919_191907


namespace NUMINAMATH_CALUDE_guitar_price_l1919_191945

theorem guitar_price (upfront_percentage : ℝ) (upfront_payment : ℝ) (total_price : ℝ) :
  upfront_percentage = 0.20 →
  upfront_payment = 240 →
  upfront_percentage * total_price = upfront_payment →
  total_price = 1200 := by
  sorry

end NUMINAMATH_CALUDE_guitar_price_l1919_191945


namespace NUMINAMATH_CALUDE_valid_representation_characterization_l1919_191969

def is_valid_representation (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 
    a > 1 ∧ 
    a ∣ n ∧ 
    (∀ d : ℕ, d > 1 → d ∣ n → d ≥ a) ∧
    b ∣ n ∧
    n = a^2 + b^2

theorem valid_representation_characterization :
  ∀ n : ℕ, is_valid_representation n ↔ (n = 8 ∨ n = 20) :=
sorry

end NUMINAMATH_CALUDE_valid_representation_characterization_l1919_191969


namespace NUMINAMATH_CALUDE_sum_distances_geq_6r_sum_squared_distances_geq_12r_squared_l1919_191912

-- Define a triangle in a plane
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a point in the plane
def Point : Type := ℝ × ℝ

-- Define the distance function
def distance (p1 p2 : Point) : ℝ := sorry

-- Define the radius of the inscribed circle
def inRadius (t : Triangle) : ℝ := sorry

-- Define Ra, Rb, Rc
def Ra (t : Triangle) (M : Point) : ℝ := distance M t.A
def Rb (t : Triangle) (M : Point) : ℝ := distance M t.B
def Rc (t : Triangle) (M : Point) : ℝ := distance M t.C

-- Theorem 1
theorem sum_distances_geq_6r (t : Triangle) (M : Point) :
  Ra t M + Rb t M + Rc t M ≥ 6 * inRadius t := sorry

-- Theorem 2
theorem sum_squared_distances_geq_12r_squared (t : Triangle) (M : Point) :
  Ra t M ^ 2 + Rb t M ^ 2 + Rc t M ^ 2 ≥ 12 * (inRadius t) ^ 2 := sorry

end NUMINAMATH_CALUDE_sum_distances_geq_6r_sum_squared_distances_geq_12r_squared_l1919_191912


namespace NUMINAMATH_CALUDE_class_average_mark_l1919_191905

theorem class_average_mark (total_students : ℕ) (excluded_students : ℕ) 
  (excluded_avg : ℝ) (remaining_avg : ℝ) : 
  total_students = 10 →
  excluded_students = 5 →
  excluded_avg = 50 →
  remaining_avg = 90 →
  (total_students * (total_students * excluded_avg + (total_students - excluded_students) * remaining_avg) / total_students) / total_students = 70 := by
  sorry

end NUMINAMATH_CALUDE_class_average_mark_l1919_191905


namespace NUMINAMATH_CALUDE_mothers_age_l1919_191946

theorem mothers_age (daughter_age_in_3_years : ℕ) 
  (h1 : daughter_age_in_3_years = 26) 
  (h2 : ∃ (mother_age_5_years_ago daughter_age_5_years_ago : ℕ), 
    mother_age_5_years_ago = 2 * daughter_age_5_years_ago) : 
  ∃ (mother_current_age : ℕ), mother_current_age = 41 := by
sorry

end NUMINAMATH_CALUDE_mothers_age_l1919_191946


namespace NUMINAMATH_CALUDE_rectangular_array_sum_ratio_l1919_191911

theorem rectangular_array_sum_ratio (a : Fin 50 → Fin 40 → ℝ) :
  let row_sum : Fin 50 → ℝ := λ i => (Finset.univ.sum (λ j => a i j))
  let col_sum : Fin 40 → ℝ := λ j => (Finset.univ.sum (λ i => a i j))
  let C : ℝ := (Finset.univ.sum row_sum) / 50
  let D : ℝ := (Finset.univ.sum col_sum) / 40
  C / D = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_rectangular_array_sum_ratio_l1919_191911


namespace NUMINAMATH_CALUDE_min_value_a_l1919_191921

theorem min_value_a (a : ℝ) : 
  (∀ x > a, x + 4 / (x - a) ≥ 5) → a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_min_value_a_l1919_191921


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l1919_191934

/-- The focal length of a hyperbola with equation x² - y² = 1 is 2√2 -/
theorem hyperbola_focal_length :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2 - y^2 = 1
  ∃ (f : ℝ), (f = 2 * Real.sqrt 2 ∧ 
    ∀ (c : ℝ), (c^2 = 2 → f = 2*c) ∧
    ∃ (a b : ℝ), (a^2 = 1 ∧ b^2 = 1 ∧ c^2 = a^2 + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l1919_191934


namespace NUMINAMATH_CALUDE_tiffany_homework_l1919_191914

theorem tiffany_homework (math_pages : ℕ) (problems_per_page : ℕ) (total_problems : ℕ) 
  (h1 : math_pages = 6)
  (h2 : problems_per_page = 3)
  (h3 : total_problems = 30) :
  (total_problems - math_pages * problems_per_page) / problems_per_page = 4 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_homework_l1919_191914


namespace NUMINAMATH_CALUDE_fraction_simplification_l1919_191936

theorem fraction_simplification (x : ℝ) : 
  (x + 2) / 4 + (3 - 4 * x) / 3 = (-13 * x + 18) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1919_191936
