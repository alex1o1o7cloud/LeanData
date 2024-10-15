import Mathlib

namespace NUMINAMATH_CALUDE_conic_section_equation_l2729_272938

/-- A conic section that satisfies specific conditions -/
structure ConicSection where
  -- The conic section passes through these two points
  point_a : (ℝ × ℝ)
  point_b : (ℝ × ℝ)
  -- The conic section shares a common asymptote with this hyperbola
  asymptote_hyperbola : (ℝ → ℝ → Prop)
  -- The conic section is a hyperbola with this focal length
  focal_length : ℝ

/-- The standard equation of a hyperbola -/
def standard_hyperbola_equation (a b : ℝ) : ℝ → ℝ → Prop :=
  fun x y => x^2 / a^2 - y^2 / b^2 = 1

/-- Theorem stating the standard equation of the conic section -/
theorem conic_section_equation (c : ConicSection)
  (h1 : c.point_a = (2, -Real.sqrt 2 / 2))
  (h2 : c.point_b = (-Real.sqrt 2, -Real.sqrt 3 / 2))
  (h3 : c.asymptote_hyperbola = standard_hyperbola_equation 5 3)
  (h4 : c.focal_length = 8) :
  (standard_hyperbola_equation 10 6 = c.asymptote_hyperbola) ∨
  (standard_hyperbola_equation 6 10 = c.asymptote_hyperbola) :=
sorry

end NUMINAMATH_CALUDE_conic_section_equation_l2729_272938


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l2729_272996

theorem quadratic_root_difference (a b c : ℝ) (h : b^2 - 4*a*c ≥ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a ≠ 0 ∧ a*x^2 + b*x + c = 0 ∧ r₁ * r₂ < 20 → |r₁ - r₂| = 2 :=
by
  sorry

#check quadratic_root_difference 1 (-8) 15

end NUMINAMATH_CALUDE_quadratic_root_difference_l2729_272996


namespace NUMINAMATH_CALUDE_solve_equation_l2729_272988

theorem solve_equation : ∃ x : ℕ, x * 12 = 173 * 240 ∧ x = 3460 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2729_272988


namespace NUMINAMATH_CALUDE_factorial_340_trailing_zeros_l2729_272918

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 340! ends with 83 zeros -/
theorem factorial_340_trailing_zeros :
  trailingZeros 340 = 83 := by
  sorry

end NUMINAMATH_CALUDE_factorial_340_trailing_zeros_l2729_272918


namespace NUMINAMATH_CALUDE_prob_at_least_one_ace_value_l2729_272919

/-- The number of cards in two standard decks -/
def total_cards : ℕ := 104

/-- The number of aces in two standard decks -/
def total_aces : ℕ := 8

/-- The probability of drawing at least one ace when two cards are chosen
    sequentially with replacement from a deck of two standard decks -/
def prob_at_least_one_ace : ℚ :=
  1 - (1 - total_aces / total_cards) ^ 2

theorem prob_at_least_one_ace_value :
  prob_at_least_one_ace = 25 / 169 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_ace_value_l2729_272919


namespace NUMINAMATH_CALUDE_discount_ratio_l2729_272915

/-- Calculates the total discount for a given number of gallons -/
def calculateDiscount (gallons : ℕ) : ℚ :=
  let firstTier := min gallons 10
  let secondTier := min (gallons - 10) 10
  let thirdTier := max (gallons - 20) 0
  (firstTier : ℚ) * (5 / 100) + (secondTier : ℚ) * (10 / 100) + (thirdTier : ℚ) * (15 / 100)

/-- The discount ratio theorem -/
theorem discount_ratio :
  let kimDiscount := calculateDiscount 20
  let isabellaDiscount := calculateDiscount 25
  let elijahDiscount := calculateDiscount 30
  (isabellaDiscount : ℚ) / kimDiscount = 3 / 2 ∧
  (elijahDiscount : ℚ) / kimDiscount = 4 / 2 :=
by sorry

end NUMINAMATH_CALUDE_discount_ratio_l2729_272915


namespace NUMINAMATH_CALUDE_vector_collinearity_l2729_272995

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 * w.2 = k * v.2 * w.1

theorem vector_collinearity (x : ℝ) : 
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 1)
  collinear a (a.1 - b.1, a.2 - b.2) → x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_vector_collinearity_l2729_272995


namespace NUMINAMATH_CALUDE_new_shoes_count_l2729_272913

theorem new_shoes_count (pairs_bought : ℕ) (shoes_per_pair : ℕ) : 
  pairs_bought = 3 → shoes_per_pair = 2 → pairs_bought * shoes_per_pair = 6 := by
  sorry

end NUMINAMATH_CALUDE_new_shoes_count_l2729_272913


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2729_272960

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (λ i => a₁ + d * i)

def count_terms (a₁ : ℤ) (d : ℤ) (aₙ : ℤ) : ℕ :=
  ((aₙ - a₁) / d).toNat + 1

def sum_multiples_of_10 (lst : List ℤ) : ℤ :=
  lst.filter (λ x => x % 10 = 0) |>.sum

theorem arithmetic_sequence_properties :
  let a₁ := -45
  let d := 7
  let aₙ := 98
  let n := count_terms a₁ d aₙ
  let seq := arithmetic_sequence a₁ d n
  n = 21 ∧ sum_multiples_of_10 seq = 60 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2729_272960


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2729_272950

def set_A : Set ℝ := {y | ∃ x, y = Real.log x}
def set_B : Set ℝ := {x | x ≥ 0}

theorem intersection_of_A_and_B : set_A ∩ set_B = Set.Ici (0 : ℝ) := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2729_272950


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l2729_272911

/-- A perfect square is an integer that is the square of another integer. -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- A perfect cube is an integer that is the cube of another integer. -/
def IsPerfectCube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

/-- The smallest positive integer n such that 5n is a perfect square and 3n is a perfect cube. -/
theorem smallest_n_square_and_cube : 
  (∀ m : ℕ, m > 0 ∧ m < 1125 → ¬(IsPerfectSquare (5 * m) ∧ IsPerfectCube (3 * m))) ∧ 
  (IsPerfectSquare (5 * 1125) ∧ IsPerfectCube (3 * 1125)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l2729_272911


namespace NUMINAMATH_CALUDE_writer_productivity_l2729_272975

/-- Given a writer's manuscript details, calculate their writing productivity. -/
theorem writer_productivity (total_words : ℕ) (total_hours : ℕ) (break_hours : ℕ) :
  total_words = 60000 →
  total_hours = 120 →
  break_hours = 20 →
  (total_words : ℝ) / (total_hours - break_hours : ℝ) = 600 := by
  sorry

end NUMINAMATH_CALUDE_writer_productivity_l2729_272975


namespace NUMINAMATH_CALUDE_sin_seven_pi_sixths_l2729_272959

theorem sin_seven_pi_sixths : Real.sin (7 * π / 6) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_seven_pi_sixths_l2729_272959


namespace NUMINAMATH_CALUDE_remainder_problem_l2729_272946

theorem remainder_problem : (((1234567 % 135) * 5) % 27) = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2729_272946


namespace NUMINAMATH_CALUDE_sum_of_A_and_B_l2729_272966

theorem sum_of_A_and_B : ∀ A B : ℚ, 
  (1 / 4 : ℚ) * (1 / 8 : ℚ) = 1 / (4 * A) ∧ 
  (1 / 4 : ℚ) * (1 / 8 : ℚ) = 1 / B → 
  A + B = 40 := by
sorry

end NUMINAMATH_CALUDE_sum_of_A_and_B_l2729_272966


namespace NUMINAMATH_CALUDE_prob_two_even_dice_l2729_272948

/-- The number of sides on each die -/
def sides : ℕ := 8

/-- The set of even numbers on an 8-sided die -/
def even_numbers : Finset ℕ := {2, 4, 6, 8}

/-- The probability of rolling an even number on a single 8-sided die -/
def prob_even : ℚ := (even_numbers.card : ℚ) / sides

/-- The probability of rolling two even numbers on two 8-sided dice -/
theorem prob_two_even_dice : prob_even * prob_even = 1/4 := by sorry

end NUMINAMATH_CALUDE_prob_two_even_dice_l2729_272948


namespace NUMINAMATH_CALUDE_no_three_consecutive_squares_l2729_272901

/-- An arithmetic progression of natural numbers -/
structure ArithmeticProgression where
  terms : ℕ → ℕ
  common_difference : ℕ
  increasing : ∀ n, terms n < terms (n + 1)
  difference_property : ∀ n, terms (n + 1) - terms n = common_difference
  difference_ends_2019 : common_difference % 10000 = 2019

/-- Three consecutive squares in an arithmetic progression -/
def ThreeConsecutiveSquares (ap : ArithmeticProgression) (n : ℕ) : Prop :=
  ∃ a b c : ℕ, 
    ap.terms n = a^2 ∧ 
    ap.terms (n + 1) = b^2 ∧ 
    ap.terms (n + 2) = c^2

theorem no_three_consecutive_squares (ap : ArithmeticProgression) :
  ¬ ∃ n, ThreeConsecutiveSquares ap n :=
sorry

end NUMINAMATH_CALUDE_no_three_consecutive_squares_l2729_272901


namespace NUMINAMATH_CALUDE_ratio_of_segments_l2729_272981

/-- Given four points P, Q, R, and S on a line in that order, 
    with PQ = 3, QR = 7, and PS = 17, the ratio of PR to QS is 10/7. -/
theorem ratio_of_segments (P Q R S : ℝ) : 
  Q < R ∧ R < S ∧ Q - P = 3 ∧ R - Q = 7 ∧ S - P = 17 → 
  (R - P) / (S - Q) = 10 / 7 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_segments_l2729_272981


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l2729_272953

theorem completing_square_quadratic (x : ℝ) : 
  x^2 - 8*x + 6 = 0 ↔ (x - 4)^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l2729_272953


namespace NUMINAMATH_CALUDE_solution_range_for_a_l2729_272999

/-- The system of equations has a solution with distinct real x, y, z if and only if a is in (23/27, 1) -/
theorem solution_range_for_a (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^3 + y^2 + z^2 = a ∧
    x^2 + y^3 + z^2 = a ∧
    x^2 + y^2 + z^3 = a) ↔
  (23/27 < a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_solution_range_for_a_l2729_272999


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2729_272970

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3, 6}
def B : Set Nat := {1, 2}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2729_272970


namespace NUMINAMATH_CALUDE_algebraic_expression_correct_l2729_272930

/-- The algebraic expression for the number that is 2 less than three times the cube of a and b -/
def algebraic_expression (a b : ℝ) : ℝ := 3 * (a^3 + b^3) - 2

/-- Theorem stating that the algebraic expression is correct -/
theorem algebraic_expression_correct (a b : ℝ) :
  algebraic_expression a b = 3 * (a^3 + b^3) - 2 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_correct_l2729_272930


namespace NUMINAMATH_CALUDE_sun_xing_zhe_product_sum_l2729_272997

theorem sun_xing_zhe_product_sum : ∃ (S X Z : ℕ), 
  (S < 10 ∧ X < 10 ∧ Z < 10) ∧ 
  (100 * S + 10 * X + Z) * (100 * Z + 10 * X + S) = 78445 ∧
  (100 * S + 10 * X + Z) + (100 * Z + 10 * X + S) = 1372 := by
  sorry

end NUMINAMATH_CALUDE_sun_xing_zhe_product_sum_l2729_272997


namespace NUMINAMATH_CALUDE_cookies_left_l2729_272952

def cookies_per_tray : ℕ := 12

def daily_trays : List ℕ := [2, 3, 4, 5, 3, 4, 4]

def frank_daily_consumption : ℕ := 2

def ted_consumption : List (ℕ × ℕ) := [(2, 3), (4, 5)]

def jan_consumption : ℕ × ℕ := (3, 5)

def tom_consumption : ℕ × ℕ := (5, 8)

def neighbours_consumption : ℕ × ℕ := (6, 20)

def total_baked (trays : List ℕ) (cookies_per_tray : ℕ) : ℕ :=
  (trays.map (· * cookies_per_tray)).sum

def total_eaten (frank_daily : ℕ) (ted : List (ℕ × ℕ)) (jan : ℕ × ℕ) (tom : ℕ × ℕ) (neighbours : ℕ × ℕ) : ℕ :=
  7 * frank_daily + (ted.map Prod.snd).sum + jan.snd + tom.snd + neighbours.snd

theorem cookies_left : 
  total_baked daily_trays cookies_per_tray - 
  total_eaten frank_daily_consumption ted_consumption jan_consumption tom_consumption neighbours_consumption = 245 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_l2729_272952


namespace NUMINAMATH_CALUDE_h_value_l2729_272916

-- Define polynomials f and h
variable (f h : ℝ → ℝ)

-- Define the conditions
axiom f_def : ∀ x, f x = x^4 - 2*x^3 + x - 1
axiom sum_eq : ∀ x, f x + h x = 3*x^2 + 5*x - 4

-- State the theorem
theorem h_value : ∀ x, h x = -x^4 + 2*x^3 + 3*x^2 + 4*x - 3 :=
sorry

end NUMINAMATH_CALUDE_h_value_l2729_272916


namespace NUMINAMATH_CALUDE_second_train_speed_l2729_272967

/-- Proves that the speed of the second train is 60 km/h given the conditions of the problem -/
theorem second_train_speed
  (first_train_speed : ℝ)
  (time_difference : ℝ)
  (meeting_distance : ℝ)
  (h1 : first_train_speed = 40)
  (h2 : time_difference = 1)
  (h3 : meeting_distance = 120) :
  let second_train_speed := meeting_distance / (meeting_distance / first_train_speed - time_difference)
  second_train_speed = 60 := by
sorry

end NUMINAMATH_CALUDE_second_train_speed_l2729_272967


namespace NUMINAMATH_CALUDE_equation_solution_l2729_272962

theorem equation_solution : ∃ x : ℝ, (x - 6) ^ 4 = (1 / 16)⁻¹ ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2729_272962


namespace NUMINAMATH_CALUDE_notebook_marker_cost_l2729_272921

/-- Given the cost of notebooks and markers, prove the cost of a specific combination -/
theorem notebook_marker_cost (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 7.30)
  (h2 : 5 * x + 3 * y = 11.65) : 
  2 * x + y = 4.35 := by
  sorry

end NUMINAMATH_CALUDE_notebook_marker_cost_l2729_272921


namespace NUMINAMATH_CALUDE_rectangle_area_l2729_272926

/-- Given a rectangle EFGH with vertices E(0, 0), F(0, 6), G(y, 6), and H(y, 0),
    if the area of the rectangle is 42 square units and y > 0, then y = 7. -/
theorem rectangle_area (y : ℝ) : y > 0 → (6 * y = 42) → y = 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2729_272926


namespace NUMINAMATH_CALUDE_splitting_number_345_l2729_272958

/-- The first splitting number for a given base number -/
def first_split (n : ℕ) : ℕ := n * (n - 1) + 1

/-- The property that 345 is one of the splitting numbers of m³ -/
def is_splitting_number (m : ℕ) : Prop :=
  m > 1 ∧ ∃ k, k ≥ 0 ∧ k < m ∧ first_split m + 2 * k = 345

theorem splitting_number_345 (m : ℕ) :
  is_splitting_number m → m = 19 := by
  sorry

end NUMINAMATH_CALUDE_splitting_number_345_l2729_272958


namespace NUMINAMATH_CALUDE_washing_machine_cost_l2729_272939

/-- The cost of a washing machine and dryer, with a discount applied --/
theorem washing_machine_cost 
  (washing_machine_cost : ℝ) 
  (dryer_cost : ℝ) 
  (discount_rate : ℝ) 
  (total_after_discount : ℝ) :
  washing_machine_cost = 100 ∧ 
  dryer_cost = washing_machine_cost - 30 ∧
  discount_rate = 0.1 ∧
  total_after_discount = 153 ∧
  (1 - discount_rate) * (washing_machine_cost + dryer_cost) = total_after_discount →
  washing_machine_cost = 100 := by
sorry

end NUMINAMATH_CALUDE_washing_machine_cost_l2729_272939


namespace NUMINAMATH_CALUDE_equation_linear_iff_k_eq_neg_two_l2729_272908

/-- The equation (k-2)x^(|k|-1) = k+1 is linear in x if and only if k = -2 -/
theorem equation_linear_iff_k_eq_neg_two :
  ∀ k : ℤ, (∃ a b : ℝ, ∀ x : ℝ, (k - 2) * x^(|k| - 1) = a * x + b) ↔ k = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_linear_iff_k_eq_neg_two_l2729_272908


namespace NUMINAMATH_CALUDE_e_is_largest_l2729_272906

-- Define the variables
variable (a b c d e : ℝ)

-- Define the given equation
def equation := (a - 2 = b + 3) ∧ (b + 3 = c - 4) ∧ (c - 4 = d + 5) ∧ (d + 5 = e - 6)

-- Theorem statement
theorem e_is_largest (h : equation a b c d e) : 
  e = max a (max b (max c d)) :=
sorry

end NUMINAMATH_CALUDE_e_is_largest_l2729_272906


namespace NUMINAMATH_CALUDE_parabola_coordinate_shift_l2729_272980

/-- Given a parabola y = 3x² in a Cartesian coordinate system, 
    if the coordinate system is shifted 3 units right and 3 units up,
    then the equation of the parabola in the new coordinate system is y = 3(x+3)² - 3 -/
theorem parabola_coordinate_shift (x y : ℝ) : 
  (y = 3 * x^2) → 
  (∃ x' y', x' = x - 3 ∧ y' = y - 3 ∧ y' = 3 * (x' + 3)^2 - 3) :=
by sorry

end NUMINAMATH_CALUDE_parabola_coordinate_shift_l2729_272980


namespace NUMINAMATH_CALUDE_shaded_probability_is_one_third_l2729_272979

/-- Represents a triangle in the diagram -/
structure Triangle where
  shaded : Bool

/-- Represents the diagram with triangles -/
structure Diagram where
  triangles : List Triangle
  shaded_count : Nat
  h_more_than_five : triangles.length > 5
  h_shaded_count : shaded_count = (triangles.filter (·.shaded)).length

/-- The probability of selecting a shaded triangle -/
def shaded_probability (d : Diagram) : ℚ :=
  d.shaded_count / d.triangles.length

/-- Theorem stating the probability of selecting a shaded triangle is 1/3 -/
theorem shaded_probability_is_one_third (d : Diagram) :
  d.shaded_count = 3 ∧ d.triangles.length = 9 →
  shaded_probability d = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_probability_is_one_third_l2729_272979


namespace NUMINAMATH_CALUDE_variable_prime_count_l2729_272993

/-- The number of primes between n^2 + 1 and n^2 + n is not constant for n > 1 -/
theorem variable_prime_count (n : ℕ) (h : n > 1) :
  ∃ m : ℕ, m > n ∧ 
  (Finset.filter (Nat.Prime) (Finset.range (n^2 + n - (n^2 + 2) + 1))).card ≠
  (Finset.filter (Nat.Prime) (Finset.range (m^2 + m - (m^2 + 2) + 1))).card :=
by sorry

end NUMINAMATH_CALUDE_variable_prime_count_l2729_272993


namespace NUMINAMATH_CALUDE_sunzi_wood_measurement_problem_l2729_272994

/-- Represents the wood measurement problem from "The Mathematical Classic of Sunzi" -/
theorem sunzi_wood_measurement_problem 
  (x y : ℝ) -- x: length of rope, y: length of wood
  (h1 : x - y = 4.5) -- condition: 4.5 feet of rope left when measuring
  (h2 : (1/2) * x + 1 = y) -- condition: 1 foot left when rope is folded in half
  : (x - y = 4.5) ∧ ((1/2) * x + 1 = y) := by
  sorry

end NUMINAMATH_CALUDE_sunzi_wood_measurement_problem_l2729_272994


namespace NUMINAMATH_CALUDE_inverse_composition_nonexistence_l2729_272964

theorem inverse_composition_nonexistence 
  (f h : ℝ → ℝ) 
  (h_def : ∀ x, f⁻¹ (h x) = 7 * x^2 + 4) : 
  ¬∃ y, h⁻¹ (f (-3)) = y :=
sorry

end NUMINAMATH_CALUDE_inverse_composition_nonexistence_l2729_272964


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2729_272907

theorem completing_square_equivalence :
  ∀ x : ℝ, (x^2 - 8*x + 9 = 0) ↔ ((x - 4)^2 = 7) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2729_272907


namespace NUMINAMATH_CALUDE_shell_difference_l2729_272998

theorem shell_difference (perfect_total : ℕ) (broken_total : ℕ)
  (broken_spiral_percent : ℚ) (broken_clam_percent : ℚ)
  (perfect_spiral_percent : ℚ) (perfect_clam_percent : ℚ)
  (h1 : perfect_total = 30)
  (h2 : broken_total = 80)
  (h3 : broken_spiral_percent = 35 / 100)
  (h4 : broken_clam_percent = 40 / 100)
  (h5 : perfect_spiral_percent = 25 / 100)
  (h6 : perfect_clam_percent = 50 / 100) :
  ⌊broken_total * broken_spiral_percent⌋ - ⌊perfect_total * perfect_spiral_percent⌋ = 21 :=
by sorry

end NUMINAMATH_CALUDE_shell_difference_l2729_272998


namespace NUMINAMATH_CALUDE_compound_molar_mass_l2729_272934

/-- Given a compound where 5 moles weigh 1170 grams, prove its molar mass is 234 grams/mole. -/
theorem compound_molar_mass (mass : ℝ) (moles : ℝ) (h1 : mass = 1170) (h2 : moles = 5) :
  mass / moles = 234 := by
sorry

end NUMINAMATH_CALUDE_compound_molar_mass_l2729_272934


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_gt_zero_l2729_272987

theorem x_gt_one_sufficient_not_necessary_for_x_gt_zero :
  (∀ x : ℝ, x > 1 → x > 0) ∧ (∃ x : ℝ, x > 0 ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_gt_zero_l2729_272987


namespace NUMINAMATH_CALUDE_floor_ceil_product_l2729_272990

theorem floor_ceil_product : ⌊(0.998 : ℝ)⌋ * ⌈(1.999 : ℝ)⌉ = 0 := by sorry

end NUMINAMATH_CALUDE_floor_ceil_product_l2729_272990


namespace NUMINAMATH_CALUDE_division_problem_l2729_272925

theorem division_problem (remainder quotient divisor dividend : ℕ) : 
  remainder = 8 →
  divisor = 3 * quotient →
  divisor = 3 * remainder + 3 →
  dividend = divisor * quotient + remainder →
  dividend = 251 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2729_272925


namespace NUMINAMATH_CALUDE_stewart_farm_sheep_count_l2729_272982

theorem stewart_farm_sheep_count :
  ∀ (num_sheep num_horses : ℕ),
    (num_sheep : ℚ) / num_horses = 4 / 7 →
    num_horses * 230 = 12880 →
    num_sheep = 32 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_count_l2729_272982


namespace NUMINAMATH_CALUDE_abc_sum_mod_8_l2729_272905

theorem abc_sum_mod_8 (a b c : ℕ) : 
  0 < a ∧ a < 8 ∧ 
  0 < b ∧ b < 8 ∧ 
  0 < c ∧ c < 8 ∧ 
  (a * b * c) % 8 = 1 ∧ 
  (4 * b * c) % 8 = 3 ∧ 
  (5 * b) % 8 = (3 + b) % 8 
  → (a + b + c) % 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_mod_8_l2729_272905


namespace NUMINAMATH_CALUDE_simplify_expression_l2729_272971

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x + y)⁻¹ * (x⁻¹ + y⁻¹) = x⁻¹ * y⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2729_272971


namespace NUMINAMATH_CALUDE_boys_camp_total_l2729_272924

theorem boys_camp_total (total : ℕ) 
  (h1 : (total : ℚ) * (1/5) = (total : ℚ) * (20/100))  -- 20% of boys are from school A
  (h2 : (total : ℚ) * (1/5) * (3/10) = (total : ℚ) * (1/5) * (30/100))  -- 30% of boys from school A study science
  (h3 : (total : ℚ) * (1/5) * (7/10) = 49)  -- 49 boys are from school A but do not study science
  : total = 350 := by
sorry


end NUMINAMATH_CALUDE_boys_camp_total_l2729_272924


namespace NUMINAMATH_CALUDE_min_cost_45_ropes_l2729_272961

/-- Represents the cost and quantity of ropes --/
structure RopePurchase where
  costA : ℝ  -- Cost of one rope A
  costB : ℝ  -- Cost of one rope B
  quantA : ℕ -- Quantity of rope A
  quantB : ℕ -- Quantity of rope B

/-- Calculates the total cost of a rope purchase --/
def totalCost (p : RopePurchase) : ℝ :=
  p.costA * p.quantA + p.costB * p.quantB

/-- Theorem stating the minimum cost for purchasing 45 ropes --/
theorem min_cost_45_ropes (p : RopePurchase) :
  p.quantA + p.quantB = 45 →
  10 * 10 + 5 * 15 = 175 →
  15 * 10 + 10 * 15 = 300 →
  548 ≤ totalCost p →
  totalCost p ≤ 560 →
  ∃ (q : RopePurchase), 
    q.costA = 10 ∧ 
    q.costB = 15 ∧ 
    q.quantA = 25 ∧ 
    q.quantB = 20 ∧ 
    totalCost q = 550 ∧ 
    totalCost q ≤ totalCost p :=
by
  sorry


end NUMINAMATH_CALUDE_min_cost_45_ropes_l2729_272961


namespace NUMINAMATH_CALUDE_truck_driver_pay_l2729_272963

/-- Calculates the pay for a round trip given the pay rate per mile and one-way distance -/
def round_trip_pay (pay_rate : ℚ) (one_way_distance : ℕ) : ℚ :=
  2 * pay_rate * one_way_distance

/-- Proves that given a pay rate of $0.40 per mile and a one-way trip distance of 400 miles,
    the total pay for a round trip is $320 -/
theorem truck_driver_pay : round_trip_pay (40/100) 400 = 320 := by
  sorry

end NUMINAMATH_CALUDE_truck_driver_pay_l2729_272963


namespace NUMINAMATH_CALUDE_power_of_fraction_five_sevenths_sixth_l2729_272978

theorem power_of_fraction_five_sevenths_sixth : (5 : ℚ) / 7 ^ 6 = 15625 / 117649 := by
  sorry

end NUMINAMATH_CALUDE_power_of_fraction_five_sevenths_sixth_l2729_272978


namespace NUMINAMATH_CALUDE_math_textbooks_in_same_box_l2729_272957

def total_textbooks : ℕ := 13
def math_textbooks : ℕ := 4
def box1_capacity : ℕ := 4
def box2_capacity : ℕ := 4
def box3_capacity : ℕ := 5

def probability_all_math_in_one_box : ℚ := 1 / 4120

theorem math_textbooks_in_same_box :
  let total_arrangements := (total_textbooks.choose box1_capacity) *
                            ((total_textbooks - box1_capacity).choose box2_capacity) *
                            ((total_textbooks - box1_capacity - box2_capacity).choose box3_capacity)
  let favorable_outcomes := (total_textbooks - math_textbooks).choose 1 *
                            ((total_textbooks - math_textbooks - 1).choose box1_capacity) *
                            ((total_textbooks - math_textbooks - 1 - box1_capacity).choose box2_capacity)
  (favorable_outcomes : ℚ) / total_arrangements = probability_all_math_in_one_box :=
sorry

end NUMINAMATH_CALUDE_math_textbooks_in_same_box_l2729_272957


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_is_one_l2729_272932

theorem sum_of_a_and_b_is_one (a b : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : a / (1 - i) = 1 - b * i) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_is_one_l2729_272932


namespace NUMINAMATH_CALUDE_m_geq_two_l2729_272976

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Assume f' is the derivative of f
axiom is_derivative : ∀ x, HasDerivAt f (f' x) x

-- Given condition: f'(x) < x for all x ∈ ℝ
axiom f'_less_than_x : ∀ x, f' x < x

-- Define m as a real number
variable (m : ℝ)

-- Given inequality involving f
axiom f_inequality : f (4 - m) - f m ≥ 8 - 4 * m

-- Theorem to prove
theorem m_geq_two : m ≥ 2 := by sorry

end NUMINAMATH_CALUDE_m_geq_two_l2729_272976


namespace NUMINAMATH_CALUDE_equation_has_real_roots_l2729_272947

theorem equation_has_real_roots (K : ℝ) : ∃ x : ℝ, x = K^2 * (x - 1) * (x - 3) :=
sorry

end NUMINAMATH_CALUDE_equation_has_real_roots_l2729_272947


namespace NUMINAMATH_CALUDE_exists_valid_8_stamp_combination_min_8_stamps_for_valid_combination_min_stamps_is_8_l2729_272929

/-- Represents the number of stamps of each denomination -/
structure StampCombination :=
  (s06 : ℕ)  -- number of 0.6 yuan stamps
  (s08 : ℕ)  -- number of 0.8 yuan stamps
  (s11 : ℕ)  -- number of 1.1 yuan stamps

/-- The total postage value of a stamp combination -/
def postageValue (sc : StampCombination) : ℚ :=
  0.6 * sc.s06 + 0.8 * sc.s08 + 1.1 * sc.s11

/-- The total number of stamps in a combination -/
def totalStamps (sc : StampCombination) : ℕ :=
  sc.s06 + sc.s08 + sc.s11

/-- A stamp combination is valid if it exactly equals the required postage -/
def isValidCombination (sc : StampCombination) : Prop :=
  postageValue sc = 7.5

/-- There exists a valid stamp combination using 8 stamps -/
theorem exists_valid_8_stamp_combination :
  ∃ (sc : StampCombination), isValidCombination sc ∧ totalStamps sc = 8 :=
sorry

/-- Any valid stamp combination uses at least 8 stamps -/
theorem min_8_stamps_for_valid_combination :
  ∀ (sc : StampCombination), isValidCombination sc → totalStamps sc ≥ 8 :=
sorry

/-- The minimum number of stamps required for a valid combination is 8 -/
theorem min_stamps_is_8 :
  (∃ (sc : StampCombination), isValidCombination sc ∧ totalStamps sc = 8) ∧
  (∀ (sc : StampCombination), isValidCombination sc → totalStamps sc ≥ 8) :=
sorry

end NUMINAMATH_CALUDE_exists_valid_8_stamp_combination_min_8_stamps_for_valid_combination_min_stamps_is_8_l2729_272929


namespace NUMINAMATH_CALUDE_circle_radius_given_area_circumference_ratio_l2729_272956

/-- Given a circle with area A and circumference C, if A/C = 15, then the radius is 30 -/
theorem circle_radius_given_area_circumference_ratio (A C : ℝ) (h : A / C = 15) :
  ∃ (r : ℝ), A = π * r^2 ∧ C = 2 * π * r ∧ r = 30 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_given_area_circumference_ratio_l2729_272956


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l2729_272965

theorem matrix_equation_solution :
  let M : ℂ → Matrix (Fin 2) (Fin 2) ℂ := λ x => !![3*x, 3; 2*x, x]
  ∀ x : ℂ, M x = (-6 : ℂ) • (1 : Matrix (Fin 2) (Fin 2) ℂ) ↔ x = 1 + I ∨ x = 1 - I :=
by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l2729_272965


namespace NUMINAMATH_CALUDE_single_point_condition_l2729_272974

/-- The equation of the curve -/
def curve_equation (x y c : ℝ) : Prop :=
  3 * x^2 + y^2 + 6 * x - 12 * y + c = 0

/-- The curve is a single point -/
def is_single_point (c : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, curve_equation p.1 p.2 c

/-- The value of c for which the curve is a single point -/
theorem single_point_condition :
  ∃! c : ℝ, is_single_point c ∧ c = 39 :=
sorry

end NUMINAMATH_CALUDE_single_point_condition_l2729_272974


namespace NUMINAMATH_CALUDE_max_a_for_monotonic_increasing_l2729_272983

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

-- State the theorem
theorem max_a_for_monotonic_increasing (a : ℝ) :
  a > 0 →
  (∀ x ≥ 1, Monotone (fun x => f a x)) →
  a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_a_for_monotonic_increasing_l2729_272983


namespace NUMINAMATH_CALUDE_leastDivisorTheorem_l2729_272943

/-- The least positive integer that divides 16800 to get a number that is both a perfect square and a perfect cube -/
def leastDivisor : ℕ := 8400

/-- 16800 divided by the least divisor is a perfect square -/
def isPerfectSquare : Prop :=
  ∃ m : ℕ, (16800 / leastDivisor) = m * m

/-- 16800 divided by the least divisor is a perfect cube -/
def isPerfectCube : Prop :=
  ∃ m : ℕ, (16800 / leastDivisor) = m * m * m

/-- The main theorem stating that leastDivisor is the smallest positive integer
    that divides 16800 to get both a perfect square and a perfect cube -/
theorem leastDivisorTheorem :
  isPerfectSquare ∧ isPerfectCube ∧
  ∀ n : ℕ, 0 < n ∧ n < leastDivisor →
    ¬(∃ m : ℕ, (16800 / n) = m * m) ∨ ¬(∃ m : ℕ, (16800 / n) = m * m * m) :=
sorry

end NUMINAMATH_CALUDE_leastDivisorTheorem_l2729_272943


namespace NUMINAMATH_CALUDE_hockey_team_starters_l2729_272954

/-- The number of ways to choose starters from a hockey team with quadruplets -/
def chooseStarters (totalPlayers : ℕ) (quadruplets : ℕ) (starters : ℕ) (maxQuadruplets : ℕ) : ℕ :=
  (Nat.choose (totalPlayers - quadruplets) starters) +
  (quadruplets * Nat.choose (totalPlayers - quadruplets) (starters - 1)) +
  (Nat.choose quadruplets 2 * Nat.choose (totalPlayers - quadruplets) (starters - 2))

/-- The theorem stating the correct number of ways to choose starters -/
theorem hockey_team_starters :
  chooseStarters 18 4 7 2 = 27456 := by
  sorry

end NUMINAMATH_CALUDE_hockey_team_starters_l2729_272954


namespace NUMINAMATH_CALUDE_fish_count_after_21_days_l2729_272942

/-- Represents the state of the aquarium --/
structure AquariumState where
  days : ℕ
  fish : ℕ

/-- Calculates the number of fish eaten in a given number of days --/
def fishEaten (days : ℕ) : ℕ :=
  (2 + 3) * days

/-- Calculates the number of fish born in a given number of days --/
def fishBorn (days : ℕ) : ℕ :=
  2 * (days / 3)

/-- Updates the aquarium state for a given number of days --/
def updateAquarium (state : AquariumState) (days : ℕ) : AquariumState :=
  let newFish := max 0 (state.fish - fishEaten days + fishBorn days)
  { days := state.days + days, fish := newFish }

/-- Adds a specified number of fish to the aquarium --/
def addFish (state : AquariumState) (amount : ℕ) : AquariumState :=
  { state with fish := state.fish + amount }

/-- The final state of the aquarium after 21 days --/
def finalState : AquariumState :=
  let initialState : AquariumState := { days := 0, fish := 60 }
  let afterTwoWeeks := updateAquarium initialState 14
  let withAddedFish := addFish afterTwoWeeks 8
  updateAquarium withAddedFish 7

/-- The theorem stating that the number of fish after 21 days is 4 --/
theorem fish_count_after_21_days :
  finalState.fish = 4 := by sorry

end NUMINAMATH_CALUDE_fish_count_after_21_days_l2729_272942


namespace NUMINAMATH_CALUDE_gcd_120_75_l2729_272977

theorem gcd_120_75 : Nat.gcd 120 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_120_75_l2729_272977


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2729_272940

theorem expand_and_simplify (x : ℝ) : (2 * x - 3) * (4 * x + 5) = 8 * x^2 - 2 * x - 15 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2729_272940


namespace NUMINAMATH_CALUDE_cola_sales_count_l2729_272914

/-- Represents the number of bottles sold for each drink type -/
structure DrinkSales where
  cola : ℕ
  juice : ℕ
  water : ℕ

/-- Calculates the total earnings from drink sales -/
def totalEarnings (sales : DrinkSales) : ℚ :=
  3 * sales.cola + 1.5 * sales.juice + 1 * sales.water

/-- Theorem stating that the number of cola bottles sold is 15 -/
theorem cola_sales_count : ∃ (sales : DrinkSales), 
  sales.juice = 12 ∧ 
  sales.water = 25 ∧ 
  totalEarnings sales = 88 ∧ 
  sales.cola = 15 := by
  sorry

end NUMINAMATH_CALUDE_cola_sales_count_l2729_272914


namespace NUMINAMATH_CALUDE_travel_ratio_l2729_272951

-- Define variables for the number of countries each person traveled to
def george_countries : ℕ := 6
def zack_countries : ℕ := 18

-- Define functions for other travelers based on the given conditions
def joseph_countries : ℕ := george_countries / 2
def patrick_countries : ℕ := zack_countries / 2

-- Define the ratio function
def ratio (a b : ℕ) : ℚ := a / b

-- Theorem statement
theorem travel_ratio : ratio patrick_countries joseph_countries = 3 := by sorry

end NUMINAMATH_CALUDE_travel_ratio_l2729_272951


namespace NUMINAMATH_CALUDE_parametric_to_ordinary_equation_l2729_272945

theorem parametric_to_ordinary_equation :
  ∀ (θ : ℝ) (x y : ℝ),
    x = Real.cos θ ^ 2 →
    y = 2 * Real.sin θ ^ 2 →
    2 * x + y - 2 = 0 ∧ x ∈ Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_parametric_to_ordinary_equation_l2729_272945


namespace NUMINAMATH_CALUDE_remainder_14_power_53_mod_7_l2729_272986

theorem remainder_14_power_53_mod_7 : 14^53 % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_14_power_53_mod_7_l2729_272986


namespace NUMINAMATH_CALUDE_composite_sum_l2729_272904

theorem composite_sum (x y : ℕ) (h1 : x > 1) (h2 : y > 1) 
  (h3 : (x^2 + y^2 - 1) % (x + y - 1) = 0) : 
  ¬ Nat.Prime (x + y - 1) := by
sorry

end NUMINAMATH_CALUDE_composite_sum_l2729_272904


namespace NUMINAMATH_CALUDE_folders_needed_l2729_272917

def initial_files : Real := 93.0
def additional_files : Real := 21.0
def files_per_folder : Real := 8.0

theorem folders_needed : 
  ∃ (n : ℕ), n = Int.ceil ((initial_files + additional_files) / files_per_folder) ∧ n = 15 := by
  sorry

end NUMINAMATH_CALUDE_folders_needed_l2729_272917


namespace NUMINAMATH_CALUDE_division_with_maximum_remainder_l2729_272928

theorem division_with_maximum_remainder :
  ∃ (star : ℕ) (triangle : ℕ),
    star / 6 = 102 ∧
    star % 6 = triangle ∧
    triangle ≤ 5 ∧
    (∀ (s t : ℕ), s / 6 = 102 ∧ s % 6 = t → t ≤ triangle) ∧
    triangle = 5 ∧
    star = 617 := by
  sorry

end NUMINAMATH_CALUDE_division_with_maximum_remainder_l2729_272928


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2729_272989

/-- Given that p is inversely proportional to q+2 and p = 1 when q = 4,
    prove that p = 2 when q = 1. -/
theorem inverse_proportion_problem (p q : ℝ) (h : ∃ k : ℝ, ∀ q, p = k / (q + 2)) 
  (h1 : p = 1 → q = 4) : p = 2 → q = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2729_272989


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l2729_272936

theorem smallest_n_for_inequality : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → 3^(3^(m+1)) ≥ 1007 → n ≤ m) ∧
  3^(3^(n+1)) ≥ 1007 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l2729_272936


namespace NUMINAMATH_CALUDE_price_difference_theorem_l2729_272969

-- Define the discounted price
def discounted_price : ℝ := 71.4

-- Define the discount rate
def discount_rate : ℝ := 0.15

-- Define the price increase rate
def increase_rate : ℝ := 0.25

-- Theorem statement
theorem price_difference_theorem :
  let original_price := discounted_price / (1 - discount_rate)
  let final_price := discounted_price * (1 + increase_rate)
  final_price - original_price = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_theorem_l2729_272969


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l2729_272910

-- Define the upstream and downstream speeds
def upstream_speed : ℝ := 6
def downstream_speed : ℝ := 3

-- Define the theorem
theorem round_trip_average_speed :
  let total_distance : ℝ → ℝ := λ d => 2 * d
  let upstream_time : ℝ → ℝ := λ d => d / upstream_speed
  let downstream_time : ℝ → ℝ := λ d => d / downstream_speed
  let total_time : ℝ → ℝ := λ d => upstream_time d + downstream_time d
  let average_speed : ℝ → ℝ := λ d => total_distance d / total_time d
  ∀ d : ℝ, d > 0 → average_speed d = 4 :=
by sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l2729_272910


namespace NUMINAMATH_CALUDE_f_properties_l2729_272949

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3

/-- The theorem stating the properties of function f -/
theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 Real.pi → (f x ≥ 1 ↔ x ∈ Set.Icc 0 (Real.pi / 4) ∪ Set.Icc (11 * Real.pi / 12) Real.pi)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2729_272949


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2729_272973

theorem arithmetic_expression_equality : 5 * 7 - 6 + 2 * 12 + 2 * 6 + 7 * 3 = 86 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2729_272973


namespace NUMINAMATH_CALUDE_ellipse_properties_l2729_272984

/-- An ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  eccentricity : ℝ
  right_focus_to_vertex : ℝ

/-- The standard form of an ellipse equation -/
def standard_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- A line in slope-intercept form -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The dot product of two 2D vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem ellipse_properties (C : Ellipse) 
  (h1 : C.center = (0, 0))
  (h2 : C.foci_on_x_axis = true)
  (h3 : C.eccentricity = 1/2)
  (h4 : C.right_focus_to_vertex = 1) :
  (∃ (x y : ℝ), standard_equation 4 3 x y) ∧
  (∃ (l : Line) (A B : ℝ × ℝ), 
    (standard_equation 4 3 A.1 A.2) ∧
    (standard_equation 4 3 B.1 B.2) ∧
    (A.2 = l.slope * A.1 + l.intercept) ∧
    (B.2 = l.slope * B.1 + l.intercept) ∧
    (dot_product A B = 0)) ∧
  (∀ (m : ℝ), (∃ (k : ℝ), 
    ∃ (A B : ℝ × ℝ),
      (standard_equation 4 3 A.1 A.2) ∧
      (standard_equation 4 3 B.1 B.2) ∧
      (A.2 = k * A.1 + m) ∧
      (B.2 = k * B.1 + m) ∧
      (dot_product A B = 0)) ↔ 
    (m ≤ -2 * Real.sqrt 21 / 7 ∨ m ≥ 2 * Real.sqrt 21 / 7)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2729_272984


namespace NUMINAMATH_CALUDE_largest_divisible_n_l2729_272902

theorem largest_divisible_n : ∃ (n : ℕ), n > 0 ∧ (n + 15) ∣ (n^3 + 150) ∧ ∀ (m : ℕ), m > n → ¬((m + 15) ∣ (m^3 + 150)) :=
by
  -- The largest such n is 2385
  use 2385
  sorry

end NUMINAMATH_CALUDE_largest_divisible_n_l2729_272902


namespace NUMINAMATH_CALUDE_ellipse_properties_l2729_272931

-- Define the ellipse C
def ellipse_C (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / 3 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define points A and B
def A (m : ℝ) : ℝ × ℝ := (-2, m)
def B (n : ℝ) : ℝ × ℝ := (2, n)

-- Define the perpendicular condition
def perpendicular (A B C : ℝ × ℝ) : Prop :=
  (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0

-- Define the isosceles condition
def isosceles (A B C : ℝ × ℝ) : Prop :=
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2

-- Theorem statement
theorem ellipse_properties :
  ∃ (a : ℝ), a > 0 ∧
  (∀ x y, ellipse_C a x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  ∃ m n, perpendicular (A m) (B n) F₁ ∧
         isosceles (A m) (B n) F₁ ∧
         abs ((A m).1 - (B n).1) * abs ((A m).2 - F₁.2) / 2 = 6 * Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2729_272931


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l2729_272955

/-- Given a line with equation x + y + 1 = 0 and a point of symmetry (1, 2),
    the symmetric line has the equation x + y - 7 = 0 -/
theorem symmetric_line_equation :
  let original_line := {(x, y) : ℝ × ℝ | x + y + 1 = 0}
  let symmetry_point := (1, 2)
  let symmetric_line := {(x, y) : ℝ × ℝ | x + y - 7 = 0}
  ∀ (p : ℝ × ℝ), p ∈ symmetric_line ↔
    (2 * symmetry_point.1 - p.1, 2 * symmetry_point.2 - p.2) ∈ original_line :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l2729_272955


namespace NUMINAMATH_CALUDE_problem_statement_l2729_272972

theorem problem_statement (x y z : ℝ) 
  (h1 : x*z/(x+y) + y*z/(y+z) + x*y/(z+x) = -10)
  (h2 : y*z/(x+y) + z*x/(y+z) + x*y/(z+x) = 15) :
  y/(x+y) + z/(y+z) + x/(z+x) = 14 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2729_272972


namespace NUMINAMATH_CALUDE_zero_points_count_midpoint_derivative_negative_l2729_272944

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + 1

-- Define the derivative of f(x)
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1 / x - a

-- Theorem for the number of zero points
theorem zero_points_count (a : ℝ) (h : a > 0) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ ∀ x, f a x = 0 → x = x₁ ∨ x = x₂) ∨
  (∃! x, f a x = 0) ∨
  (∀ x, f a x ≠ 0) :=
sorry

-- Theorem for f'(x₀) < 0
theorem midpoint_derivative_negative (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : a > 0) (h₂ : x₁ < x₂) (h₃ : f a x₁ = 0) (h₄ : f a x₂ = 0) :
  let x₀ := (x₁ + x₂) / 2
  f_deriv a x₀ < 0 :=
sorry

end NUMINAMATH_CALUDE_zero_points_count_midpoint_derivative_negative_l2729_272944


namespace NUMINAMATH_CALUDE_segment_length_product_l2729_272968

theorem segment_length_product (b : ℝ) : 
  (∃ b₁ b₂ : ℝ, 
    (∀ b : ℝ, (((3*b - 7)^2 + (2*b + 1)^2 : ℝ) = 50) ↔ (b = b₁ ∨ b = b₂)) ∧ 
    (b₁ * b₂ = 0)) := by
  sorry

end NUMINAMATH_CALUDE_segment_length_product_l2729_272968


namespace NUMINAMATH_CALUDE_minimum_value_of_f_l2729_272912

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x + a else x^2 - a*x

theorem minimum_value_of_f (a : ℝ) :
  (∀ x, f a x ≥ a) ∧ (∃ x, f a x = a) → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_of_f_l2729_272912


namespace NUMINAMATH_CALUDE_smallest_circle_covering_region_line_intersecting_circle_l2729_272941

-- Define the planar region
def planar_region (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ x + 2*y - 4 ≤ 0

-- Define the circle (C)
def circle_C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 1)^2 = 5

-- Define the line (l)
def line_l (x y : ℝ) : Prop :=
  y = x - 1 + Real.sqrt 5 ∨ y = x - 1 - Real.sqrt 5

-- Theorem for the smallest circle covering the region
theorem smallest_circle_covering_region :
  (∀ x y, planar_region x y → circle_C x y) ∧
  (∀ x' y', (∀ x y, planar_region x y → (x - x')^2 + (y - y')^2 ≤ r'^2) →
    r'^2 ≥ 5) :=
sorry

-- Theorem for the line intersecting the circle
theorem line_intersecting_circle :
  ∃ A B : ℝ × ℝ,
    A ≠ B ∧
    circle_C A.1 A.2 ∧
    circle_C B.1 B.2 ∧
    line_l A.1 A.2 ∧
    line_l B.1 B.2 ∧
    ((A.1 - 2) * (B.1 - 2) + (A.2 - 1) * (B.2 - 1) = 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_circle_covering_region_line_intersecting_circle_l2729_272941


namespace NUMINAMATH_CALUDE_natural_number_equations_l2729_272927

theorem natural_number_equations :
  (∃! (x : ℕ), 2^(x-5) = 2) ∧
  (∃! (x : ℕ), 2^x = 512) ∧
  (∃! (x : ℕ), x^5 = 243) ∧
  (∃! (x : ℕ), x^4 = 625) :=
by
  sorry

end NUMINAMATH_CALUDE_natural_number_equations_l2729_272927


namespace NUMINAMATH_CALUDE_expression_one_equality_l2729_272903

theorem expression_one_equality : 
  4 * Real.sqrt 54 * 3 * Real.sqrt 2 / (-(3/2) * Real.sqrt (1/3)) = -144 := by
sorry

end NUMINAMATH_CALUDE_expression_one_equality_l2729_272903


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l2729_272909

theorem completing_square_quadratic (x : ℝ) : 
  (x^2 + 8*x + 7 = 0) ↔ ((x + 4)^2 = 9) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l2729_272909


namespace NUMINAMATH_CALUDE_rectangle_length_eq_five_l2729_272937

/-- The length of a rectangle with width 20 cm and perimeter equal to that of a regular pentagon with side length 10 cm is 5 cm. -/
theorem rectangle_length_eq_five (width : ℝ) (pentagon_side : ℝ) (length : ℝ) : 
  width = 20 →
  pentagon_side = 10 →
  2 * (length + width) = 5 * pentagon_side →
  length = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_eq_five_l2729_272937


namespace NUMINAMATH_CALUDE_x_eight_plus_x_four_plus_one_eq_zero_l2729_272923

theorem x_eight_plus_x_four_plus_one_eq_zero 
  (x : ℂ) (h : x^2 + x + 1 = 0) : x^8 + x^4 + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_eight_plus_x_four_plus_one_eq_zero_l2729_272923


namespace NUMINAMATH_CALUDE_square_floor_tiles_l2729_272935

/-- Represents a square floor covered with tiles -/
structure TiledFloor where
  side_length : ℕ
  is_even : Even side_length
  diagonal_tiles : ℕ
  h_diagonal : diagonal_tiles = 2 * side_length

/-- The total number of tiles on a square floor -/
def total_tiles (floor : TiledFloor) : ℕ :=
  floor.side_length ^ 2

theorem square_floor_tiles (floor : TiledFloor) 
  (h_diagonal_count : floor.diagonal_tiles = 88) : 
  total_tiles floor = 1936 := by
  sorry

end NUMINAMATH_CALUDE_square_floor_tiles_l2729_272935


namespace NUMINAMATH_CALUDE_initial_girls_count_l2729_272900

theorem initial_girls_count (n : ℕ) : 
  n > 0 →
  (n : ℚ) / 2 - 2 = (2 * n : ℚ) / 5 →
  (n : ℚ) / 2 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_girls_count_l2729_272900


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l2729_272920

def A : Set ℝ := {-1, 0, 1}
def B (a : ℝ) : Set ℝ := {a + 1, 2 * a}

theorem intersection_implies_a_value (a : ℝ) :
  A ∩ B a = {0} → a = -1 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l2729_272920


namespace NUMINAMATH_CALUDE_divisibility_rule_3_decimal_true_divisibility_rule_3_duodecimal_false_l2729_272991

/-- Represents a positional numeral system with a given base. -/
structure NumeralSystem (base : ℕ) where
  (digits : List ℕ)
  (valid_digits : ∀ d ∈ digits, d < base)

/-- The value of a number in a given numeral system. -/
def value (base : ℕ) (num : NumeralSystem base) : ℕ :=
  (num.digits.reverse.enum.map (λ (i, d) => d * base ^ i)).sum

/-- The sum of digits of a number in a given numeral system. -/
def digit_sum (base : ℕ) (num : NumeralSystem base) : ℕ :=
  num.digits.sum

/-- Divisibility rule for 3 in a given numeral system. -/
def divisibility_rule_3 (base : ℕ) : Prop :=
  ∀ (num : NumeralSystem base), 
    (value base num) % 3 = 0 ↔ (digit_sum base num) % 3 = 0

theorem divisibility_rule_3_decimal_true : 
  divisibility_rule_3 10 := by sorry

theorem divisibility_rule_3_duodecimal_false : 
  ¬(divisibility_rule_3 12) := by sorry

end NUMINAMATH_CALUDE_divisibility_rule_3_decimal_true_divisibility_rule_3_duodecimal_false_l2729_272991


namespace NUMINAMATH_CALUDE_calculate_expression_l2729_272985

theorem calculate_expression : (2 - 5 * (-1/2)^2) / (-1/4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2729_272985


namespace NUMINAMATH_CALUDE_batsman_average_l2729_272992

/-- Calculates the overall average runs per match for a batsman -/
def overall_average (matches1 : ℕ) (avg1 : ℚ) (matches2 : ℕ) (avg2 : ℚ) : ℚ :=
  (matches1 * avg1 + matches2 * avg2) / (matches1 + matches2)

/-- The batsman's overall average is approximately 21.43 -/
theorem batsman_average : 
  let matches1 := 15
  let avg1 := 30
  let matches2 := 20
  let avg2 := 15
  abs (overall_average matches1 avg1 matches2 avg2 - 21.43) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_l2729_272992


namespace NUMINAMATH_CALUDE_cube_volumes_theorem_l2729_272933

/-- The edge length of the first cube in centimeters -/
def x : ℝ := 18

/-- The volume of a cube with edge length l -/
def cube_volume (l : ℝ) : ℝ := l^3

/-- The edge length of the second cube in centimeters -/
def second_edge : ℝ := x - 4

/-- The edge length of the third cube in centimeters -/
def third_edge : ℝ := second_edge - 2

/-- The volume of water remaining in the first cube after filling the second -/
def remaining_first : ℝ := cube_volume x - cube_volume second_edge

/-- The volume of water remaining in the second cube after filling the third -/
def remaining_second : ℝ := cube_volume second_edge - cube_volume third_edge

theorem cube_volumes_theorem : 
  remaining_first = 3 * remaining_second + 40 ∧ 
  cube_volume x = 5832 ∧ 
  cube_volume second_edge = 2744 ∧ 
  cube_volume third_edge = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cube_volumes_theorem_l2729_272933


namespace NUMINAMATH_CALUDE_segment_length_given_ratio_points_l2729_272922

/-- Represents a point on a line segment -/
structure PointOnSegment (A B : ℝ) where
  position : ℝ
  h1 : A ≤ position
  h2 : position ≤ B

/-- The length of a line segment -/
def segmentLength (A B : ℝ) : ℝ := B - A

theorem segment_length_given_ratio_points 
  (A B : ℝ) 
  (P Q : PointOnSegment A B)
  (h_order : A < P.position ∧ P.position < Q.position ∧ Q.position < B)
  (h_P_ratio : P.position - A = 3/8 * (B - A))
  (h_Q_ratio : Q.position - A = 2/5 * (B - A))
  (h_PQ_length : Q.position - P.position = 3)
  : segmentLength A B = 120 := by
  sorry

end NUMINAMATH_CALUDE_segment_length_given_ratio_points_l2729_272922
