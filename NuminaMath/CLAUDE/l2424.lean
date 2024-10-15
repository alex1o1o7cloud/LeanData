import Mathlib

namespace NUMINAMATH_CALUDE_coloring_book_solution_l2424_242431

/-- Represents the problem of determining the initial stock of coloring books. -/
def ColoringBookProblem (initial_stock acquired_books books_per_shelf total_shelves : ℝ) : Prop :=
  initial_stock + acquired_books = books_per_shelf * total_shelves

/-- The theorem stating the solution to the coloring book problem. -/
theorem coloring_book_solution :
  ∃ (initial_stock : ℝ),
    ColoringBookProblem initial_stock 20 4 15 ∧
    initial_stock = 40 := by
  sorry

end NUMINAMATH_CALUDE_coloring_book_solution_l2424_242431


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l2424_242486

theorem max_sum_of_factors (A B C : ℕ) : 
  A > 0 → B > 0 → C > 0 →
  A ≠ B → B ≠ C → A ≠ C →
  A * B * C = 2023 →
  A + B + C ≤ 297 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l2424_242486


namespace NUMINAMATH_CALUDE_michael_truck_rental_cost_l2424_242490

/-- Calculates the total cost of renting a truck given the rental fee, charge per mile, and miles driven. -/
def truckRentalCost (rentalFee : ℚ) (chargePerMile : ℚ) (milesDriven : ℕ) : ℚ :=
  rentalFee + chargePerMile * milesDriven

/-- Proves that the total cost for Michael's truck rental is $95.74 -/
theorem michael_truck_rental_cost :
  truckRentalCost 20.99 0.25 299 = 95.74 := by
  sorry

end NUMINAMATH_CALUDE_michael_truck_rental_cost_l2424_242490


namespace NUMINAMATH_CALUDE_water_percentage_in_mixture_l2424_242466

/-- Given two liquids with different water percentages, prove the water percentage in their mixture -/
theorem water_percentage_in_mixture 
  (water_percent_1 water_percent_2 : ℝ) 
  (parts_1 parts_2 : ℝ) 
  (h1 : water_percent_1 = 20)
  (h2 : water_percent_2 = 35)
  (h3 : parts_1 = 10)
  (h4 : parts_2 = 4) :
  (water_percent_1 / 100 * parts_1 + water_percent_2 / 100 * parts_2) / (parts_1 + parts_2) * 100 =
  (0.2 * 10 + 0.35 * 4) / (10 + 4) * 100 := by
  sorry

#eval (0.2 * 10 + 0.35 * 4) / (10 + 4) * 100

end NUMINAMATH_CALUDE_water_percentage_in_mixture_l2424_242466


namespace NUMINAMATH_CALUDE_wage_increase_with_productivity_l2424_242438

/-- Represents the linear regression equation for workers' wages as a function of labor productivity -/
def wage_equation (x : ℝ) : ℝ := 50 + 80 * x

/-- Theorem stating that an increase of 1 in labor productivity leads to an increase of 80 in wages -/
theorem wage_increase_with_productivity (x : ℝ) :
  wage_equation (x + 1) - wage_equation x = 80 := by
  sorry

end NUMINAMATH_CALUDE_wage_increase_with_productivity_l2424_242438


namespace NUMINAMATH_CALUDE_complex_square_pure_imaginary_l2424_242481

theorem complex_square_pure_imaginary (a : ℝ) : 
  let z : ℂ := a + 3*I
  (∃ b : ℝ, z^2 = b*I ∧ b ≠ 0) → (a = 3 ∨ a = -3) :=
by sorry

end NUMINAMATH_CALUDE_complex_square_pure_imaginary_l2424_242481


namespace NUMINAMATH_CALUDE_square_area_after_cut_l2424_242492

theorem square_area_after_cut (side : ℝ) (h1 : side > 0) : 
  side * (side - 3) = 40 → side * side = 64 := by sorry

end NUMINAMATH_CALUDE_square_area_after_cut_l2424_242492


namespace NUMINAMATH_CALUDE_profit_maximizing_price_l2424_242437

/-- Represents the profit function for a product -/
def profit_function (x : ℝ) : ℝ :=
  (x - 8) * (100 - 10 * (x - 10))

/-- Theorem stating that the profit-maximizing price is 14 yuan -/
theorem profit_maximizing_price :
  ∃ (x : ℝ), x = 14 ∧ ∀ (y : ℝ), profit_function y ≤ profit_function x :=
sorry

end NUMINAMATH_CALUDE_profit_maximizing_price_l2424_242437


namespace NUMINAMATH_CALUDE_book_sale_loss_percentage_l2424_242488

/-- Proves that the percentage of loss is 10% given the conditions of the problem -/
theorem book_sale_loss_percentage (CP : ℝ) : 
  CP > 720 ∧ 880 = 1.10 * CP → (CP - 720) / CP * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_loss_percentage_l2424_242488


namespace NUMINAMATH_CALUDE_count_possible_denominators_all_denominators_divide_999_fraction_denominator_in_possible_set_seven_possible_denominators_l2424_242465

/-- Represents a three-digit number abc --/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  a_is_digit : a < 10
  b_is_digit : b < 10
  c_is_digit : c < 10
  not_all_nines : ¬(a = 9 ∧ b = 9 ∧ c = 9)
  not_all_zeros : ¬(a = 0 ∧ b = 0 ∧ c = 0)

/-- Converts a ThreeDigitNumber to its decimal value --/
def toDecimal (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- The denominator of the fraction representation of 0.abc̅ --/
def denominator : Nat := 999

/-- The set of possible denominators for 0.abc̅ in lowest terms --/
def possibleDenominators : Finset Nat :=
  {3, 9, 27, 37, 111, 333, 999}

/-- Theorem stating that there are exactly 7 possible denominators --/
theorem count_possible_denominators :
    (possibleDenominators.card : Nat) = 7 := by sorry

/-- Theorem stating that all elements in possibleDenominators are factors of 999 --/
theorem all_denominators_divide_999 :
    ∀ d ∈ possibleDenominators, denominator % d = 0 := by sorry

/-- Theorem stating that for any ThreeDigitNumber, its fraction representation
    has a denominator in possibleDenominators --/
theorem fraction_denominator_in_possible_set (n : ThreeDigitNumber) :
    ∃ d ∈ possibleDenominators,
      (toDecimal n).gcd denominator = (denominator / d) := by sorry

/-- Main theorem proving that there are exactly 7 possible denominators --/
theorem seven_possible_denominators :
    ∃! (s : Finset Nat),
      (∀ n : ThreeDigitNumber,
        ∃ d ∈ s, (toDecimal n).gcd denominator = (denominator / d)) ∧
      s.card = 7 := by sorry

end NUMINAMATH_CALUDE_count_possible_denominators_all_denominators_divide_999_fraction_denominator_in_possible_set_seven_possible_denominators_l2424_242465


namespace NUMINAMATH_CALUDE_game_winner_l2424_242414

/-- Represents the state of the game with three balls -/
structure GameState where
  n : ℕ -- number of empty holes between one outer ball and the middle ball
  k : ℕ -- number of empty holes between the other outer ball and the middle ball

/-- Determines if a player can make a move in the given game state -/
def canMove (state : GameState) : Prop :=
  state.n > 0 ∨ state.k > 0

/-- Determines if the first player wins in the given game state -/
def firstPlayerWins (state : GameState) : Prop :=
  (state.n + state.k) % 2 = 1

theorem game_winner (state : GameState) :
  canMove state → (firstPlayerWins state ↔ ¬firstPlayerWins { n := state.k, k := state.n - 1 }) ∧
                  (¬firstPlayerWins state ↔ ¬firstPlayerWins { n := state.n - 1, k := state.k }) :=
sorry

end NUMINAMATH_CALUDE_game_winner_l2424_242414


namespace NUMINAMATH_CALUDE_division_remainder_proof_l2424_242457

theorem division_remainder_proof :
  ∀ (dividend quotient divisor remainder : ℕ),
    dividend = 144 →
    quotient = 13 →
    divisor = 11 →
    dividend = divisor * quotient + remainder →
    remainder = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l2424_242457


namespace NUMINAMATH_CALUDE_prism_pyramid_sum_l2424_242464

/-- A shape formed by adding a pyramid to one square face of a rectangular prism -/
structure PrismPyramid where
  prism_faces : ℕ
  prism_edges : ℕ
  prism_vertices : ℕ
  pyramid_faces : ℕ
  pyramid_edges : ℕ
  pyramid_vertices : ℕ

/-- The sum of faces, edges, and vertices of the PrismPyramid -/
def total_sum (pp : PrismPyramid) : ℕ :=
  (pp.prism_faces - 1 + pp.pyramid_faces) + 
  (pp.prism_edges + pp.pyramid_edges) + 
  (pp.prism_vertices + pp.pyramid_vertices)

/-- Theorem stating that the total sum is 34 -/
theorem prism_pyramid_sum :
  ∀ (pp : PrismPyramid), 
    pp.prism_faces = 6 ∧ 
    pp.prism_edges = 12 ∧ 
    pp.prism_vertices = 8 ∧
    pp.pyramid_faces = 4 ∧
    pp.pyramid_edges = 4 ∧
    pp.pyramid_vertices = 1 →
    total_sum pp = 34 := by
  sorry

end NUMINAMATH_CALUDE_prism_pyramid_sum_l2424_242464


namespace NUMINAMATH_CALUDE_roots_of_quadratic_equation_l2424_242475

theorem roots_of_quadratic_equation : 
  ∀ x : ℝ, x^2 = 2*x ↔ x = 0 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_equation_l2424_242475


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2424_242472

theorem arithmetic_calculation : 4 * 6 * 8 + 24 / 4 = 198 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2424_242472


namespace NUMINAMATH_CALUDE_fraction_of_y_l2424_242434

theorem fraction_of_y (w x y : ℝ) (hw : w ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (2 / w + 2 / x = 2 / y) → (w * x = y) → ((w + x) / 2 = 0.5) → (2 / y = 2 / y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_y_l2424_242434


namespace NUMINAMATH_CALUDE_jerry_insult_points_l2424_242433

/-- Represents the point system in Mrs. Carlton's class -/
structure PointSystem where
  interrupt_points : ℕ
  throw_points : ℕ
  office_threshold : ℕ

/-- Represents Jerry's behavior -/
structure JerryBehavior where
  interrupts : ℕ
  insults : ℕ
  throws : ℕ

/-- Calculates the points for insults given the point system and Jerry's behavior -/
def insult_points (ps : PointSystem) (jb : JerryBehavior) : ℕ :=
  (ps.office_threshold - (ps.interrupt_points * jb.interrupts + ps.throw_points * jb.throws)) / jb.insults

/-- Theorem stating that Jerry gets 10 points for insulting his classmates -/
theorem jerry_insult_points :
  let ps : PointSystem := { interrupt_points := 5, throw_points := 25, office_threshold := 100 }
  let jb : JerryBehavior := { interrupts := 2, insults := 4, throws := 2 }
  insult_points ps jb = 10 := by
  sorry

end NUMINAMATH_CALUDE_jerry_insult_points_l2424_242433


namespace NUMINAMATH_CALUDE_magician_earnings_l2424_242415

/-- Calculates the money earned by a magician selling card decks --/
def money_earned (price_per_deck : ℕ) (starting_decks : ℕ) (ending_decks : ℕ) : ℕ :=
  (starting_decks - ending_decks) * price_per_deck

/-- Proves that the magician earned 4 dollars --/
theorem magician_earnings : money_earned 2 5 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_magician_earnings_l2424_242415


namespace NUMINAMATH_CALUDE_grid_toothpicks_l2424_242469

/-- Calculates the total number of toothpicks in a grid with diagonals -/
def total_toothpicks (length width : ℕ) : ℕ :=
  let vertical := (length + 1) * width
  let horizontal := (width + 1) * length
  let diagonal := 2 * (length * width)
  vertical + horizontal + diagonal

/-- Theorem stating that a 50x20 grid with diagonals uses 4070 toothpicks -/
theorem grid_toothpicks : total_toothpicks 50 20 = 4070 := by
  sorry

end NUMINAMATH_CALUDE_grid_toothpicks_l2424_242469


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2424_242459

theorem inequality_equivalence (x : ℝ) :
  2 * |x - 2| - |x + 1| > 3 ↔ x < 0 ∨ x > 8 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2424_242459


namespace NUMINAMATH_CALUDE_absolute_value_sum_difference_l2424_242429

theorem absolute_value_sum_difference (x y : ℝ) : 
  (|x| = 3 ∧ |y| = 7) →
  ((x > 0 ∧ y < 0 → x + y = -4) ∧
   (x < y → (x - y = -10 ∨ x - y = -4))) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_difference_l2424_242429


namespace NUMINAMATH_CALUDE_sin_cos_identity_indeterminate_l2424_242446

theorem sin_cos_identity_indeterminate (α : Real) : 
  α ∈ Set.Ioo 0 Real.pi → 
  (Real.sin α)^2 + Real.cos (2 * α) = 1 → 
  ∀ β ∈ Set.Ioo 0 Real.pi, (Real.sin β)^2 + Real.cos (2 * β) = 1 ∧ 
  ¬∃!t, t = Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_indeterminate_l2424_242446


namespace NUMINAMATH_CALUDE_negation_equivalence_l2424_242495

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180

-- Define the original proposition
def has_angle_le_60 (t : Triangle) : Prop :=
  ∃ i : Fin 3, t.angles i ≤ 60

-- Define the negation (assumption for proof by contradiction)
def all_angles_gt_60 (t : Triangle) : Prop :=
  ∀ i : Fin 3, t.angles i > 60

-- The theorem to prove
theorem negation_equivalence :
  ∀ t : Triangle, ¬(has_angle_le_60 t) ↔ all_angles_gt_60 t :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2424_242495


namespace NUMINAMATH_CALUDE_smallest_m_is_170_l2424_242402

/-- The quadratic equation 10x^2 - mx + 660 = 0 has integral solutions -/
def has_integral_solutions (m : ℤ) : Prop :=
  ∃ x : ℤ, 10 * x^2 - m * x + 660 = 0

/-- 170 is a value of m for which the equation has integral solutions -/
axiom solution_exists : has_integral_solutions 170

/-- For any positive integer less than 170, the equation does not have integral solutions -/
axiom no_smaller_solution : ∀ k : ℤ, 0 < k → k < 170 → ¬(has_integral_solutions k)

theorem smallest_m_is_170 : 
  (∃ m : ℤ, 0 < m ∧ has_integral_solutions m) ∧ 
  (∀ k : ℤ, 0 < k ∧ has_integral_solutions k → 170 ≤ k) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_is_170_l2424_242402


namespace NUMINAMATH_CALUDE_pages_read_relationship_l2424_242455

/-- Represents the number of pages read on each night --/
structure PagesRead where
  night1 : ℕ
  night2 : ℕ
  night3 : ℕ

/-- Theorem stating the relationship between pages read on night 3 and the other nights --/
theorem pages_read_relationship (p : PagesRead) (total : ℕ) : 
  p.night1 = 30 →
  p.night2 = 2 * p.night1 - 2 →
  total = p.night1 + p.night2 + p.night3 →
  total = 179 →
  p.night3 = total - (p.night1 + p.night2) := by
  sorry

end NUMINAMATH_CALUDE_pages_read_relationship_l2424_242455


namespace NUMINAMATH_CALUDE_max_weight_proof_l2424_242487

/-- The maximum number of crates the trailer can carry on a single trip -/
def max_crates : ℕ := 5

/-- The minimum weight of each crate in kilograms -/
def min_crate_weight : ℕ := 150

/-- The maximum weight of crates on a single trip in kilograms -/
def max_total_weight : ℕ := max_crates * min_crate_weight

theorem max_weight_proof :
  max_total_weight = 750 := by
  sorry

end NUMINAMATH_CALUDE_max_weight_proof_l2424_242487


namespace NUMINAMATH_CALUDE_subtraction_to_perfect_square_l2424_242443

theorem subtraction_to_perfect_square : ∃ n : ℕ, (92555 : ℕ) - 139 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_to_perfect_square_l2424_242443


namespace NUMINAMATH_CALUDE_expression_value_l2424_242422

theorem expression_value (a b : ℝ) (h : 2 * a - b = -1) : 
  b * 2 - a * 2^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l2424_242422


namespace NUMINAMATH_CALUDE_total_fruit_count_l2424_242426

theorem total_fruit_count (orange_crates : Nat) (oranges_per_crate : Nat)
                          (nectarine_boxes : Nat) (nectarines_per_box : Nat) :
  orange_crates = 12 →
  oranges_per_crate = 150 →
  nectarine_boxes = 16 →
  nectarines_per_box = 30 →
  orange_crates * oranges_per_crate + nectarine_boxes * nectarines_per_box = 2280 := by
  sorry

end NUMINAMATH_CALUDE_total_fruit_count_l2424_242426


namespace NUMINAMATH_CALUDE_cylinder_band_length_l2424_242410

theorem cylinder_band_length (m k n : ℕ) : 
  (m > 0) → (k > 0) → (n > 0) → 
  (∀ (p : ℕ), Prime p → ¬(p^2 ∣ k)) →
  (2 * (24 * Real.sqrt 3 + 28 * Real.pi) = m * Real.sqrt k + n * Real.pi) →
  m + k + n = 107 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_band_length_l2424_242410


namespace NUMINAMATH_CALUDE_minimal_intercept_line_l2424_242407

-- Define a line by its intercepts
structure Line where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0

-- Define the condition that the line passes through (1, 4)
def passesThrough (l : Line) : Prop :=
  1 / l.a + 4 / l.b = 1

-- Define the sum of intercepts
def sumOfIntercepts (l : Line) : ℝ :=
  l.a + l.b

-- State the theorem
theorem minimal_intercept_line :
  ∃ (l : Line),
    passesThrough l ∧
    ∀ (l' : Line), passesThrough l' → sumOfIntercepts l ≤ sumOfIntercepts l' ∧
    2 * (1 : ℝ) + 4 - 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_minimal_intercept_line_l2424_242407


namespace NUMINAMATH_CALUDE_difference_set_Q_P_l2424_242496

-- Define the sets P and Q
def P : Set ℝ := {x | 1 - 2/x < 0}
def Q : Set ℝ := {x | |x - 2| < 1}

-- Define the difference set
def difference_set (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∉ B}

-- Theorem statement
theorem difference_set_Q_P : 
  difference_set Q P = {x | 2 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_difference_set_Q_P_l2424_242496


namespace NUMINAMATH_CALUDE_stating_exist_same_arrangement_l2424_242441

/-- The size of the grid -/
def grid_size : Nat := 25

/-- The size of the sub-squares we're considering -/
def square_size : Nat := 3

/-- The number of possible 3x3 squares in a 25x25 grid -/
def num_squares : Nat := (grid_size - square_size + 1) ^ 2

/-- The number of possible arrangements of plus signs in a 3x3 square -/
def num_arrangements : Nat := 2 ^ (square_size ^ 2)

/-- 
Theorem stating that there exist at least two 3x3 squares 
with the same arrangement of plus signs in a 25x25 grid 
-/
theorem exist_same_arrangement : num_squares > num_arrangements := by sorry

end NUMINAMATH_CALUDE_stating_exist_same_arrangement_l2424_242441


namespace NUMINAMATH_CALUDE_class_average_mark_l2424_242424

theorem class_average_mark (total_students : ℕ) (excluded_students : ℕ) (excluded_avg : ℝ) (remaining_avg : ℝ) :
  total_students = 25 →
  excluded_students = 5 →
  excluded_avg = 20 →
  remaining_avg = 95 →
  (total_students * (total_students * remaining_avg - excluded_students * remaining_avg + excluded_students * excluded_avg)) / (total_students * total_students) = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_class_average_mark_l2424_242424


namespace NUMINAMATH_CALUDE_sum_min_max_value_l2424_242474

theorem sum_min_max_value (a b c d e : ℝ) 
  (sum_condition : a + b + c + d + e = 10)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 + e^2 = 30) : 
  let f := fun (x y z w v : ℝ) => 5 * (x^3 + y^3 + z^3 + w^3 + v^3) - (x^4 + y^4 + z^4 + w^4 + v^4)
  ∃ (m M : ℝ), 
    (∀ x y z w v, f x y z w v ≥ m) ∧ 
    (∃ x y z w v, f x y z w v = m) ∧
    (∀ x y z w v, f x y z w v ≤ M) ∧ 
    (∃ x y z w v, f x y z w v = M) ∧
    m + M = 94 :=
by sorry

end NUMINAMATH_CALUDE_sum_min_max_value_l2424_242474


namespace NUMINAMATH_CALUDE_like_terms_exponents_l2424_242440

theorem like_terms_exponents (a b : ℝ) (x y : ℤ) : 
  (∃ (k : ℝ), -4 * a^(x-y) * b^4 = k * a^2 * b^(x+y)) → 
  (x = 3 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_like_terms_exponents_l2424_242440


namespace NUMINAMATH_CALUDE_determine_m_l2424_242483

-- Define the functions f and g
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + m
def g (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + 6*m

-- State the theorem
theorem determine_m : ∃ m : ℝ, 2 * (f m 3) = 3 * (g m 3) ∧ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_determine_m_l2424_242483


namespace NUMINAMATH_CALUDE_cubic_root_ratio_l2424_242484

theorem cubic_root_ratio (a b c d : ℝ) (h : ∀ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = 4 ∨ x = 5 ∨ x = 6) :
  c / d = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_ratio_l2424_242484


namespace NUMINAMATH_CALUDE_age_equals_birth_year_digit_sum_l2424_242417

theorem age_equals_birth_year_digit_sum :
  ∃! A : ℕ, 0 ≤ A ∧ A ≤ 99 ∧
  (∃ x y : ℕ, 0 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧
    A = 1893 - (1800 + 10 * x + y) ∧
    A = 1 + 8 + x + y) ∧
  A = 24 :=
sorry

end NUMINAMATH_CALUDE_age_equals_birth_year_digit_sum_l2424_242417


namespace NUMINAMATH_CALUDE_sine_graph_horizontal_compression_l2424_242416

/-- Given a function f(x) = 2sin(x + π/3), if we shorten the horizontal coordinates
    of its graph to 1/2 of the original while keeping the vertical coordinates unchanged,
    the resulting function is g(x) = 2sin(2x + π/3) -/
theorem sine_graph_horizontal_compression (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * Real.sin (x + π/3)
  let g : ℝ → ℝ := λ x ↦ 2 * Real.sin (2*x + π/3)
  let h : ℝ → ℝ := λ x ↦ f (x/2)
  h = g :=
by sorry

end NUMINAMATH_CALUDE_sine_graph_horizontal_compression_l2424_242416


namespace NUMINAMATH_CALUDE_fishing_rod_price_theorem_l2424_242403

theorem fishing_rod_price_theorem :
  ∃ (a b c d : ℕ),
    -- Four-digit number condition
    1000 ≤ a * 1000 + b * 100 + c * 10 + d ∧ a * 1000 + b * 100 + c * 10 + d < 10000 ∧
    -- Digit relationships
    a = c + 1 ∧ a = d - 1 ∧
    -- Sum of digits
    a + b + c + d = 6 ∧
    -- Two-digit number difference
    10 * a + b = 10 * c + d + 7 ∧
    -- Product of ages
    a * 1000 + b * 100 + c * 10 + d = 61 * 3 * 11 :=
by sorry

end NUMINAMATH_CALUDE_fishing_rod_price_theorem_l2424_242403


namespace NUMINAMATH_CALUDE_theodore_wooden_statues_l2424_242478

/-- Theodore's monthly statue production and earnings --/
structure StatueProduction where
  stone_statues : ℕ
  wooden_statues : ℕ
  stone_price : ℚ
  wooden_price : ℚ
  tax_rate : ℚ
  total_earnings_after_tax : ℚ

/-- Theorem: Theodore crafts 20 wooden statues per month --/
theorem theodore_wooden_statues (p : StatueProduction) 
  (h1 : p.stone_statues = 10)
  (h2 : p.stone_price = 20)
  (h3 : p.wooden_price = 5)
  (h4 : p.tax_rate = 1/10)
  (h5 : p.total_earnings_after_tax = 270) :
  p.wooden_statues = 20 := by
  sorry

#check theodore_wooden_statues

end NUMINAMATH_CALUDE_theodore_wooden_statues_l2424_242478


namespace NUMINAMATH_CALUDE_age_difference_l2424_242425

theorem age_difference (A B : ℕ) : B = 39 → A + 10 = 2 * (B - 10) → A - B = 9 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2424_242425


namespace NUMINAMATH_CALUDE_measure_all_masses_l2424_242453

-- Define the set of weights
def weights : List ℕ := [1, 3, 9, 27, 81]

-- Define a function to check if a mass can be measured
def can_measure (mass : ℕ) : Prop :=
  ∃ (a b c d e : ℤ), 
    a * 1 + b * 3 + c * 9 + d * 27 + e * 81 = mass ∧ 
    (a ∈ ({-1, 0, 1} : Set ℤ)) ∧ 
    (b ∈ ({-1, 0, 1} : Set ℤ)) ∧ 
    (c ∈ ({-1, 0, 1} : Set ℤ)) ∧ 
    (d ∈ ({-1, 0, 1} : Set ℤ)) ∧ 
    (e ∈ ({-1, 0, 1} : Set ℤ))

-- Theorem statement
theorem measure_all_masses : 
  ∀ m : ℕ, 1 ≤ m ∧ m ≤ 121 → can_measure m :=
by sorry

end NUMINAMATH_CALUDE_measure_all_masses_l2424_242453


namespace NUMINAMATH_CALUDE_partial_fraction_sum_zero_l2424_242497

theorem partial_fraction_sum_zero (x : ℝ) (A B C D E : ℝ) : 
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) = 
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5) →
  A + B + C + D + E = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_zero_l2424_242497


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2424_242419

def f (x : ℝ) := x^12 + 5*x^11 + 20*x^10 + 1300*x^9 - 1105*x^8

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2424_242419


namespace NUMINAMATH_CALUDE_strategy_D_lowest_price_l2424_242405

/-- Represents a pricing strategy with an increase followed by a decrease -/
structure PricingStrategy where
  increase : ℝ
  decrease : ℝ

/-- Calculates the final price factor for a given pricing strategy -/
def finalPriceFactor (strategy : PricingStrategy) : ℝ :=
  (1 + strategy.increase) * (1 - strategy.decrease)

/-- The four pricing strategies -/
def strategyA : PricingStrategy := ⟨0.1, 0.1⟩
def strategyB : PricingStrategy := ⟨-0.1, -0.1⟩
def strategyC : PricingStrategy := ⟨0.2, 0.2⟩
def strategyD : PricingStrategy := ⟨0.3, 0.3⟩

theorem strategy_D_lowest_price :
  finalPriceFactor strategyD ≤ finalPriceFactor strategyA ∧
  finalPriceFactor strategyD ≤ finalPriceFactor strategyB ∧
  finalPriceFactor strategyD ≤ finalPriceFactor strategyC :=
sorry

end NUMINAMATH_CALUDE_strategy_D_lowest_price_l2424_242405


namespace NUMINAMATH_CALUDE_function_proof_l2424_242452

theorem function_proof (f : ℕ → ℕ) 
  (h1 : f 0 = 1)
  (h2 : f 2016 = 2017)
  (h3 : ∀ n, f (f n) + f n = 2 * n + 3) :
  ∀ n, f n = n + 1 := by
sorry

end NUMINAMATH_CALUDE_function_proof_l2424_242452


namespace NUMINAMATH_CALUDE_equivalent_transitive_l2424_242485

def IsGreat (f : ℕ → ℕ → ℤ) : Prop :=
  ∀ m n : ℕ, f (m + 1) (n + 1) * f m n - f (m + 1) n * f m (n + 1) = 1

def Equivalent (A B : ℕ → ℤ) : Prop :=
  ∃ f : ℕ → ℕ → ℤ, IsGreat f ∧ (∀ n : ℕ, f n 0 = A n ∧ f 0 n = B n)

theorem equivalent_transitive :
  ∀ A B C D : ℕ → ℤ,
    Equivalent A B → Equivalent B C → Equivalent C D → Equivalent D A :=
by sorry

end NUMINAMATH_CALUDE_equivalent_transitive_l2424_242485


namespace NUMINAMATH_CALUDE_ball_arrangements_l2424_242420

-- Define the word structure
def Word := String

-- Define a function to count distinct arrangements
def countDistinctArrangements (w : Word) : ℕ := sorry

-- Theorem statement
theorem ball_arrangements :
  let ball : Word := "BALL"
  countDistinctArrangements ball = 12 := by sorry

end NUMINAMATH_CALUDE_ball_arrangements_l2424_242420


namespace NUMINAMATH_CALUDE_expand_and_simplify_polynomial_l2424_242404

theorem expand_and_simplify_polynomial (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_polynomial_l2424_242404


namespace NUMINAMATH_CALUDE_min_gigabytes_plan_y_more_expensive_l2424_242408

/-- Represents the cost of Plan Y in cents for a given number of gigabytes -/
def planYCost (gigabytes : ℕ) : ℕ := 3000 + 200 * gigabytes

/-- Represents the cost of Plan X in cents -/
def planXCost : ℕ := 5000

/-- Theorem stating that 11 gigabytes is the minimum at which Plan Y becomes more expensive than Plan X -/
theorem min_gigabytes_plan_y_more_expensive :
  ∀ g : ℕ, g ≥ 11 ↔ planYCost g > planXCost :=
by sorry

end NUMINAMATH_CALUDE_min_gigabytes_plan_y_more_expensive_l2424_242408


namespace NUMINAMATH_CALUDE_age_difference_l2424_242401

def hiram_age : ℕ := 40
def allyson_age : ℕ := 28

theorem age_difference : 
  (2 * allyson_age) - (hiram_age + 12) = 4 :=
by sorry

end NUMINAMATH_CALUDE_age_difference_l2424_242401


namespace NUMINAMATH_CALUDE_spot_horn_proportion_is_half_l2424_242476

/-- Represents the proportion of spotted females and horned males -/
def spot_horn_proportion (total_cows : ℕ) (female_to_male_ratio : ℕ) (spotted_horned_difference : ℕ) : ℚ :=
  let male_cows := total_cows / (female_to_male_ratio + 1)
  let female_cows := female_to_male_ratio * male_cows
  (spotted_horned_difference : ℚ) / (female_cows - male_cows)

/-- Theorem stating the proportion of spotted females and horned males -/
theorem spot_horn_proportion_is_half :
  spot_horn_proportion 300 2 50 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_spot_horn_proportion_is_half_l2424_242476


namespace NUMINAMATH_CALUDE_ellipse_equation_l2424_242435

/-- The equation of an ellipse given specific conditions -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∃ (c : ℝ), c = 4 ∧ a^2 - b^2 = c^2) →  -- Right focus coincides with parabola focus
  (a / c = 3 / Real.sqrt 6) →             -- Eccentricity condition
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 24 + y^2 / 8 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2424_242435


namespace NUMINAMATH_CALUDE_new_number_correct_l2424_242451

/-- Given a two-digit number with tens' digit t and units' digit u,
    the function calculates the new three-digit number formed by
    reversing the digits and placing 2 after the reversed number. -/
def new_number (t u : ℕ) : ℕ :=
  100 * u + 10 * t + 2

/-- Theorem stating that the new_number function correctly calculates
    the desired three-digit number for any two-digit number. -/
theorem new_number_correct (t u : ℕ) (h1 : t ≥ 1) (h2 : t ≤ 9) (h3 : u ≤ 9) :
  new_number t u = 100 * u + 10 * t + 2 :=
by sorry

end NUMINAMATH_CALUDE_new_number_correct_l2424_242451


namespace NUMINAMATH_CALUDE_correct_calculation_l2424_242406

theorem correct_calculation (x : ℝ) (h : x * 3 - 45 = 159) : (x + 32) * 12 = 1200 := by
  sorry

#check correct_calculation

end NUMINAMATH_CALUDE_correct_calculation_l2424_242406


namespace NUMINAMATH_CALUDE_max_coconuts_count_l2424_242468

/-- Represents the trading ratios and final goat count -/
structure TradingSystem where
  coconuts_per_crab : ℕ
  crabs_per_goat : ℕ
  final_goats : ℕ

/-- Calculates the number of coconuts Max has -/
def coconuts_count (ts : TradingSystem) : ℕ :=
  ts.coconuts_per_crab * ts.crabs_per_goat * ts.final_goats

/-- Theorem stating that Max has 342 coconuts given the trading system -/
theorem max_coconuts_count :
  let ts : TradingSystem := ⟨3, 6, 19⟩
  coconuts_count ts = 342 := by
  sorry


end NUMINAMATH_CALUDE_max_coconuts_count_l2424_242468


namespace NUMINAMATH_CALUDE_min_value_problem_l2424_242449

open Real

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : log 2 * x + log 8 * y = log 2) : 
  (∀ a b : ℝ, a > 0 → b > 0 → log 2 * a + log 8 * b = log 2 → 1/x + 1/y ≤ 1/a + 1/b) ∧ 
  (∃ c d : ℝ, c > 0 ∧ d > 0 ∧ log 2 * c + log 8 * d = log 2 ∧ 1/c + 1/d = 4 + 2 * sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l2424_242449


namespace NUMINAMATH_CALUDE_sqrt_sum_reciprocals_l2424_242498

theorem sqrt_sum_reciprocals : Real.sqrt ((1 : ℝ) / 25 + 1 / 36) = Real.sqrt 61 / 30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_reciprocals_l2424_242498


namespace NUMINAMATH_CALUDE_min_value_ab_l2424_242480

theorem min_value_ab (a b : ℝ) (ha : a > 1) (hb : b > 1) (heq : a * b + 2 = 2 * (a + b)) :
  ∃ (min : ℝ), min = 6 + 4 * Real.sqrt 2 ∧ a * b ≥ min := by
sorry

end NUMINAMATH_CALUDE_min_value_ab_l2424_242480


namespace NUMINAMATH_CALUDE_jane_rejection_rate_l2424_242413

theorem jane_rejection_rate 
  (total_rejection_rate : ℝ) 
  (john_rejection_rate : ℝ) 
  (jane_inspection_fraction : ℝ) 
  (h1 : total_rejection_rate = 0.0075) 
  (h2 : john_rejection_rate = 0.005) 
  (h3 : jane_inspection_fraction = 0.8333333333333333) :
  let john_inspection_fraction := 1 - jane_inspection_fraction
  let jane_rejection_rate := (total_rejection_rate - john_rejection_rate * john_inspection_fraction) / jane_inspection_fraction
  jane_rejection_rate = 0.008 := by
sorry

end NUMINAMATH_CALUDE_jane_rejection_rate_l2424_242413


namespace NUMINAMATH_CALUDE_unique_not_in_range_is_30_l2424_242479

/-- Function f with the given properties -/
noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

/-- Theorem stating that 30 is the unique number not in the range of f -/
theorem unique_not_in_range_is_30
  (a b c d : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : f a b c d 10 = 10)
  (h2 : f a b c d 50 = 50)
  (h3 : ∀ x ≠ -d/c, f a b c d (f a b c d x) = x) :
  ∃! y, ∀ x, f a b c d x ≠ y ∧ y = 30 :=
sorry

end NUMINAMATH_CALUDE_unique_not_in_range_is_30_l2424_242479


namespace NUMINAMATH_CALUDE_sequence_ratio_l2424_242489

/-- Given an arithmetic sequence a and a geometric sequence b with specific conditions,
    prove that the ratio of their second terms is 1. -/
theorem sequence_ratio (a b : ℕ → ℚ) : 
  (∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) →  -- a is arithmetic
  (∀ n : ℕ, b (n + 1) / b n = b 1 / b 0) →  -- b is geometric
  a 0 = -1 →                                -- a₁ = -1
  b 0 = -1 →                                -- b₁ = -1
  a 3 = 8 →                                 -- a₄ = 8
  b 3 = 8 →                                 -- b₄ = 8
  a 1 / b 1 = 1 :=                          -- a₂/b₂ = 1
by sorry

end NUMINAMATH_CALUDE_sequence_ratio_l2424_242489


namespace NUMINAMATH_CALUDE_volleyball_lineup_count_l2424_242462

def choose (n k : ℕ) : ℕ := Nat.choose n k

def volleyball_lineups (total_players triplets : ℕ) (max_triplets : ℕ) : ℕ :=
  let non_triplets := total_players - triplets - 1  -- Subtract 1 for the captain
  let case0 := choose non_triplets 5
  let case1 := triplets * choose non_triplets 4
  let case2 := choose triplets 2 * choose non_triplets 3
  case0 + case1 + case2

theorem volleyball_lineup_count :
  volleyball_lineups 15 4 2 = 1812 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_lineup_count_l2424_242462


namespace NUMINAMATH_CALUDE_equal_part_implies_a_eq_neg_two_l2424_242430

/-- A complex number is an "equal part complex number" if its real and imaginary parts are equal -/
def is_equal_part (z : ℂ) : Prop := z.re = z.im

/-- The complex number z defined in terms of a real number a -/
def z (a : ℝ) : ℂ := Complex.I * (2 + a * Complex.I)

/-- Theorem: If z(a) is an equal part complex number, then a = -2 -/
theorem equal_part_implies_a_eq_neg_two (a : ℝ) :
  is_equal_part (z a) → a = -2 := by sorry

end NUMINAMATH_CALUDE_equal_part_implies_a_eq_neg_two_l2424_242430


namespace NUMINAMATH_CALUDE_sugar_calculation_l2424_242477

/-- The total amount of sugar given the number of packs, weight per pack, and leftover sugar -/
def total_sugar (num_packs : ℕ) (weight_per_pack : ℕ) (leftover : ℕ) : ℕ :=
  num_packs * weight_per_pack + leftover

/-- Theorem: Given 12 packs of sugar weighing 250 grams each and 20 grams of leftover sugar,
    the total amount of sugar is 3020 grams -/
theorem sugar_calculation :
  total_sugar 12 250 20 = 3020 := by
  sorry

end NUMINAMATH_CALUDE_sugar_calculation_l2424_242477


namespace NUMINAMATH_CALUDE_triangle_theorem_l2424_242423

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  cosine_law_a : a^2 = b^2 + c^2 - 2*b*c*Real.cos A
  cosine_law_b : b^2 = a^2 + c^2 - 2*a*c*Real.cos B
  cosine_law_c : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) (h : 2*t.a*Real.cos t.C = 2*t.b - t.c) :
  /- Part 1 -/
  t.A = π/3 ∧
  /- Part 2 -/
  (t.A < π/2 ∧ t.B < π/2 ∧ t.C < π/2 → 
    3/2 < Real.sin t.B + Real.sin t.C ∧ Real.sin t.B + Real.sin t.C ≤ Real.sqrt 3) ∧
  /- Part 3 -/
  (t.a = 2*Real.sqrt 3 ∧ 1/2*t.b*t.c*Real.sin t.A = 2*Real.sqrt 3 →
    Real.cos (2*t.B) + Real.cos (2*t.C) = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2424_242423


namespace NUMINAMATH_CALUDE_quaternary_123_equals_27_l2424_242454

/-- Converts a quaternary (base-4) digit to its decimal value --/
def quaternary_to_decimal (digit : Nat) : Nat :=
  if digit < 4 then digit else 0

/-- Represents the quaternary number 123 --/
def quaternary_123 : List Nat := [1, 2, 3]

/-- Converts a list of quaternary digits to its decimal value --/
def quaternary_list_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + quaternary_to_decimal d * (4 ^ i)) 0

theorem quaternary_123_equals_27 :
  quaternary_list_to_decimal quaternary_123 = 27 := by
  sorry

end NUMINAMATH_CALUDE_quaternary_123_equals_27_l2424_242454


namespace NUMINAMATH_CALUDE_modulo_thirteen_residue_l2424_242463

theorem modulo_thirteen_residue : (247 + 5 * 39 + 7 * 143 + 4 * 15) % 13 = 8 := by
  sorry

end NUMINAMATH_CALUDE_modulo_thirteen_residue_l2424_242463


namespace NUMINAMATH_CALUDE_light_bulb_probability_l2424_242428

/-- The probability of exactly k successes in n independent trials -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The probability that a light bulb lasts more than 1000 hours -/
def p_success : ℝ := 0.2

/-- The number of light bulbs -/
def n : ℕ := 3

/-- The number of light bulbs that fail -/
def k : ℕ := 1

theorem light_bulb_probability : 
  binomial_probability n k p_success = 0.096 := by
  sorry

end NUMINAMATH_CALUDE_light_bulb_probability_l2424_242428


namespace NUMINAMATH_CALUDE_gcd_of_24_and_36_l2424_242400

theorem gcd_of_24_and_36 : Nat.gcd 24 36 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_24_and_36_l2424_242400


namespace NUMINAMATH_CALUDE_jacobStatementsDisproved_l2424_242458

-- Define the type for card sides
inductive CardSide
| Letter : Char → CardSide
| Number : Nat → CardSide

-- Define a card as a pair of sides
def Card := (CardSide × CardSide)

-- Define the properties of cards
def isVowel (c : Char) : Prop := c ∈ ['A', 'E', 'I', 'O', 'U']
def isEven (n : Nat) : Prop := n % 2 = 0
def isPrime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m > 1 → m < n → n % m ≠ 0)

-- Jacob's statements
def jacobStatement1 (card : Card) : Prop :=
  match card with
  | (CardSide.Letter c, CardSide.Number n) => isVowel c → isEven n
  | _ => True

def jacobStatement2 (card : Card) : Prop :=
  match card with
  | (CardSide.Number n, CardSide.Letter c) => isPrime n → isVowel c
  | _ => True

-- Define the set of cards
def cardSet : List Card := [
  (CardSide.Letter 'A', CardSide.Number 8),
  (CardSide.Letter 'R', CardSide.Number 5),
  (CardSide.Letter 'S', CardSide.Number 7),
  (CardSide.Number 1, CardSide.Letter 'R'),
  (CardSide.Number 8, CardSide.Letter 'S'),
  (CardSide.Number 5, CardSide.Letter 'A')
]

-- Theorem: There exist two cards that disprove at least one of Jacob's statements
theorem jacobStatementsDisproved : 
  ∃ (card1 card2 : Card), card1 ∈ cardSet ∧ card2 ∈ cardSet ∧ card1 ≠ card2 ∧
    (¬(jacobStatement1 card1) ∨ ¬(jacobStatement2 card1) ∨
     ¬(jacobStatement1 card2) ∨ ¬(jacobStatement2 card2)) :=
by sorry


end NUMINAMATH_CALUDE_jacobStatementsDisproved_l2424_242458


namespace NUMINAMATH_CALUDE_b_minus_d_squared_l2424_242491

theorem b_minus_d_squared (a b c d e : ℝ) 
  (eq1 : a - b - c + d = 13)
  (eq2 : a + b - c - d = 9)
  (eq3 : a - b + c + e = 11) : 
  (b - d)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_b_minus_d_squared_l2424_242491


namespace NUMINAMATH_CALUDE_determinant_of_specific_matrix_l2424_242473

theorem determinant_of_specific_matrix : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![2, -4, 4; 0, 6, -2; 5, -3, 2]
  Matrix.det A = -68 := by
  sorry

end NUMINAMATH_CALUDE_determinant_of_specific_matrix_l2424_242473


namespace NUMINAMATH_CALUDE_tangent_slope_implies_abscissa_l2424_242450

noncomputable def f (x : ℝ) := Real.exp x + Real.exp (-x)

theorem tangent_slope_implies_abscissa (x : ℝ) :
  (deriv f x = 3/2) → x = Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_abscissa_l2424_242450


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l2424_242499

theorem infinitely_many_solutions (a : ℚ) : 
  (∀ x : ℚ, 4 * (3 * x - 2 * a) = 3 * (4 * x + 18)) ↔ a = -27/4 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l2424_242499


namespace NUMINAMATH_CALUDE_max_value_abc_l2424_242448

theorem max_value_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 2) :
  ∃ (max : ℝ), max = 8 ∧ ∀ (x : ℝ), x = a + b^2 + c^3 → x ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_abc_l2424_242448


namespace NUMINAMATH_CALUDE_tan_double_angle_l2424_242445

theorem tan_double_angle (α : ℝ) (h : Real.tan α = 2) : Real.tan (2 * α) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l2424_242445


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2424_242482

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 3*x - 4 ≤ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2424_242482


namespace NUMINAMATH_CALUDE_ash_cloud_radius_l2424_242418

/-- Calculates the radius of an ash cloud from a volcano eruption -/
theorem ash_cloud_radius 
  (angle : Real) 
  (vertical_distance : Real) 
  (diameter_factor : Real) 
  (h1 : angle = 60) 
  (h2 : vertical_distance = 300) 
  (h3 : diameter_factor = 18) : 
  ∃ (radius : Real), abs (radius - 10228.74) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ash_cloud_radius_l2424_242418


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2424_242436

theorem cubic_equation_solution (x y z n : ℕ+) :
  x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2 ↔ n = 1 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2424_242436


namespace NUMINAMATH_CALUDE_nth_k_gonal_number_l2424_242447

/-- The nth k-gonal number -/
def N (n k : ℕ) : ℚ :=
  ((k - 2) / 2 : ℚ) * n^2 + ((4 - k) / 2 : ℚ) * n

/-- Theorem stating the properties of the nth k-gonal number -/
theorem nth_k_gonal_number (k : ℕ) (h : k ≥ 3) :
  ∀ n : ℕ, N n k = ((k - 2) / 2 : ℚ) * n^2 + ((4 - k) / 2 : ℚ) * n ∧
  N 10 24 = 1000 := by sorry

end NUMINAMATH_CALUDE_nth_k_gonal_number_l2424_242447


namespace NUMINAMATH_CALUDE_inequality_proof_l2424_242470

theorem inequality_proof (x y : ℝ) (hx : |x| ≤ 1) (hy : |y| ≤ 1) : |x + y| ≤ |1 + x * y| := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2424_242470


namespace NUMINAMATH_CALUDE_max_integer_inequality_l2424_242471

theorem max_integer_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 2*a + b = 1) :
  ∀ m : ℤ, (∀ a b, a > 0 → b > 0 → 2*a + b = 1 → 2/a + 1/b ≥ m) → m ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_max_integer_inequality_l2424_242471


namespace NUMINAMATH_CALUDE_replacement_concentration_theorem_l2424_242460

/-- Represents a hydrochloric acid solution --/
structure HClSolution where
  total_mass : ℝ
  concentration : ℝ

/-- Calculates the mass of pure HCl in a solution --/
def pure_hcl_mass (solution : HClSolution) : ℝ :=
  solution.total_mass * solution.concentration

theorem replacement_concentration_theorem 
  (initial_solution : HClSolution)
  (drained_mass : ℝ)
  (final_solution : HClSolution)
  (replacement_solution : HClSolution)
  (h1 : initial_solution.total_mass = 300)
  (h2 : initial_solution.concentration = 0.2)
  (h3 : drained_mass = 25)
  (h4 : final_solution.total_mass = initial_solution.total_mass)
  (h5 : final_solution.concentration = 0.25)
  (h6 : replacement_solution.total_mass = drained_mass)
  (h7 : pure_hcl_mass final_solution = 
        pure_hcl_mass initial_solution - pure_hcl_mass replacement_solution + 
        pure_hcl_mass replacement_solution) :
  replacement_solution.concentration = 0.8 := by
  sorry

#check replacement_concentration_theorem

end NUMINAMATH_CALUDE_replacement_concentration_theorem_l2424_242460


namespace NUMINAMATH_CALUDE_problem_solution_l2424_242493

theorem problem_solution (x n : ℕ) (h1 : x = 9^n - 1) (h2 : Odd n) 
  (h3 : (Nat.factors x).length = 3) (h4 : 61 ∈ Nat.factors x) : x = 59048 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2424_242493


namespace NUMINAMATH_CALUDE_max_a6_value_l2424_242411

theorem max_a6_value (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) 
  (h_order : a₁ ≤ a₂ ∧ a₂ ≤ a₃ ∧ a₃ ≤ a₄ ∧ a₄ ≤ a₅ ∧ a₅ ≤ a₆)
  (h_sum : a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 10)
  (h_sq_dev : (a₁ - 1)^2 + (a₂ - 1)^2 + (a₃ - 1)^2 + (a₄ - 1)^2 + (a₅ - 1)^2 + (a₆ - 1)^2 = 6) :
  a₆ ≤ 10/3 := by
sorry

end NUMINAMATH_CALUDE_max_a6_value_l2424_242411


namespace NUMINAMATH_CALUDE_meeting_time_is_48_minutes_l2424_242432

/-- Represents the cycling scenario between Andrea and Lauren -/
structure CyclingScenario where
  total_distance : ℝ
  andrea_speed_ratio : ℝ
  distance_decrease_rate : ℝ
  andrea_stop_time : ℝ

/-- Calculates the total time for Lauren to meet Andrea -/
def total_meeting_time (scenario : CyclingScenario) : ℝ :=
  sorry

/-- Theorem stating that given the specific conditions, the total meeting time is 48 minutes -/
theorem meeting_time_is_48_minutes 
  (scenario : CyclingScenario)
  (h1 : scenario.total_distance = 30)
  (h2 : scenario.andrea_speed_ratio = 2)
  (h3 : scenario.distance_decrease_rate = 1.5)
  (h4 : scenario.andrea_stop_time = 6) :
  total_meeting_time scenario = 48 :=
sorry

end NUMINAMATH_CALUDE_meeting_time_is_48_minutes_l2424_242432


namespace NUMINAMATH_CALUDE_journey_matches_graph_characteristics_l2424_242442

/-- Represents a point on the speed-time graph -/
structure SpeedTimePoint where
  time : ℝ
  speed : ℝ

/-- Represents a section of the speed-time graph -/
inductive GraphSection
  | Increasing : GraphSection
  | Flat : GraphSection
  | Decreasing : GraphSection

/-- Represents Mike's journey -/
structure Journey where
  cityTraffic : Bool
  highway : Bool
  workplace : Bool
  coffeeBreak : Bool
  workDuration : ℝ
  breakDuration : ℝ

/-- Defines the characteristics of the correct graph -/
def correctGraphCharacteristics : List GraphSection :=
  [GraphSection.Increasing, GraphSection.Flat, GraphSection.Increasing, 
   GraphSection.Flat, GraphSection.Decreasing]

/-- Theorem stating that Mike's journey matches the correct graph characteristics -/
theorem journey_matches_graph_characteristics (j : Journey) :
  j.cityTraffic = true →
  j.highway = true →
  j.workplace = true →
  j.coffeeBreak = true →
  j.workDuration = 2 →
  j.breakDuration = 0.5 →
  ∃ (graph : List GraphSection), graph = correctGraphCharacteristics := by
  sorry

#check journey_matches_graph_characteristics

end NUMINAMATH_CALUDE_journey_matches_graph_characteristics_l2424_242442


namespace NUMINAMATH_CALUDE_steps_to_school_l2424_242494

/-- The number of steps Raine takes walking to and from school in five days -/
def total_steps : ℕ := 1500

/-- The number of days Raine walks to and from school -/
def days : ℕ := 5

/-- Proves that the number of steps Raine takes to walk to school is 150 -/
theorem steps_to_school : (total_steps / (2 * days) : ℕ) = 150 := by
  sorry

end NUMINAMATH_CALUDE_steps_to_school_l2424_242494


namespace NUMINAMATH_CALUDE_sum_in_base5_l2424_242456

/-- Converts a base 5 number to base 10 --/
def base5ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 5 --/
def base10ToBase5 (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of 201₅, 324₅, and 143₅ is equal to 1123₅ in base 5 --/
theorem sum_in_base5 :
  base10ToBase5 (base5ToBase10 201 + base5ToBase10 324 + base5ToBase10 143) = 1123 := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base5_l2424_242456


namespace NUMINAMATH_CALUDE_roots_sum_reciprocal_minus_one_l2424_242467

theorem roots_sum_reciprocal_minus_one (b c : ℝ) : 
  b^2 - b - 1 = 0 → c^2 - c - 1 = 0 → b ≠ c → 1 / (1 - b) + 1 / (1 - c) = -1 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_reciprocal_minus_one_l2424_242467


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2424_242439

/-- The eccentricity of a hyperbola with asymptotes tangent to a specific circle -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1 ∧
   (b * x + a * y = 0 ∨ b * x - a * y = 0) ∧
   (x - Real.sqrt 2)^2 + y^2 = 1) →
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2424_242439


namespace NUMINAMATH_CALUDE_inequality_solutions_l2424_242461

theorem inequality_solutions :
  -- Part 1
  (∀ x : ℝ, (3*x - 2)/(x - 1) > 1 ↔ (x > 1 ∨ x < 1/2)) ∧
  -- Part 2
  (∀ a x : ℝ, 
    (a = 0 → x^2 - a*x - 2*a^2 < 0 ↔ False) ∧
    (a > 0 → (x^2 - a*x - 2*a^2 < 0 ↔ -a < x ∧ x < 2*a)) ∧
    (a < 0 → (x^2 - a*x - 2*a^2 < 0 ↔ 2*a < x ∧ x < -a))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solutions_l2424_242461


namespace NUMINAMATH_CALUDE_chad_savings_l2424_242444

/-- Chad's savings calculation --/
theorem chad_savings (savings_rate : ℚ) (mowing : ℚ) (birthday : ℚ) (video_games : ℚ) (odd_jobs : ℚ) : 
  savings_rate = 2/5 → 
  mowing = 600 → 
  birthday = 250 → 
  video_games = 150 → 
  odd_jobs = 150 → 
  savings_rate * (mowing + birthday + video_games + odd_jobs) = 460 := by
sorry

end NUMINAMATH_CALUDE_chad_savings_l2424_242444


namespace NUMINAMATH_CALUDE_all_statements_incorrect_l2424_242421

/-- Represents a type of reasoning -/
inductive ReasoningType
| Analogical
| Inductive

/-- Represents the direction of reasoning -/
inductive ReasoningDirection
| GeneralToSpecific
| SpecificToGeneral
| SpecificToSpecific

/-- Represents a statement about analogical reasoning -/
structure AnalogicalReasoningStatement where
  always_correct : Bool
  direction : ReasoningDirection
  can_prove_math : Bool
  same_as_inductive : Bool

/-- Definition of analogical reasoning -/
def analogical_reasoning : ReasoningType := ReasoningType.Analogical

/-- Definition of inductive reasoning -/
def inductive_reasoning : ReasoningType := ReasoningType.Inductive

/-- Inductive reasoning is a form of analogical reasoning -/
axiom inductive_is_analogical : inductive_reasoning = analogical_reasoning

/-- The correct properties of analogical reasoning -/
def correct_properties : AnalogicalReasoningStatement :=
  { always_correct := false
  , direction := ReasoningDirection.SpecificToSpecific
  , can_prove_math := false
  , same_as_inductive := false }

/-- Theorem stating that all given statements about analogical reasoning are incorrect -/
theorem all_statements_incorrect (statement : AnalogicalReasoningStatement) :
  statement.always_correct = true ∨
  statement.direction = ReasoningDirection.GeneralToSpecific ∨
  statement.can_prove_math = true ∨
  statement.same_as_inductive = true →
  statement ≠ correct_properties :=
sorry

end NUMINAMATH_CALUDE_all_statements_incorrect_l2424_242421


namespace NUMINAMATH_CALUDE_expression_simplification_l2424_242427

theorem expression_simplification :
  Real.sqrt (1 + 3) * Real.sqrt (4 + Real.sqrt (1 + 3 + 5 + 7 + 9)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2424_242427


namespace NUMINAMATH_CALUDE_pen_cost_l2424_242409

/-- The cost of a pen in cents, given the following conditions:
  * Pencils cost 25 cents each
  * Susan spent 20 dollars in total
  * Susan bought a total of 36 pens and pencils
  * Susan bought 16 pencils
-/
theorem pen_cost (pencil_cost : ℕ) (total_spent : ℕ) (total_items : ℕ) (pencils_bought : ℕ) :
  pencil_cost = 25 →
  total_spent = 2000 →
  total_items = 36 →
  pencils_bought = 16 →
  ∃ (pen_cost : ℕ), pen_cost = 80 :=
by sorry

end NUMINAMATH_CALUDE_pen_cost_l2424_242409


namespace NUMINAMATH_CALUDE_train_crossing_time_l2424_242412

/-- Proves that a train 100 meters long, traveling at 36 km/hr, takes 10 seconds to cross an electric pole -/
theorem train_crossing_time : 
  let train_length : ℝ := 100  -- Length of the train in meters
  let train_speed_kmh : ℝ := 36  -- Speed of the train in km/hr
  let train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)  -- Speed in m/s
  let crossing_time : ℝ := train_length / train_speed_ms  -- Time to cross in seconds
  crossing_time = 10 := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2424_242412
