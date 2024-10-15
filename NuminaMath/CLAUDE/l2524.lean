import Mathlib

namespace NUMINAMATH_CALUDE_power_sum_is_integer_l2524_252442

theorem power_sum_is_integer (x : ℝ) (h1 : x ≠ 0) (h2 : ∃ k : ℤ, x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/(x^n) = m :=
sorry

end NUMINAMATH_CALUDE_power_sum_is_integer_l2524_252442


namespace NUMINAMATH_CALUDE_nancy_soap_packs_l2524_252438

/-- Proves that Nancy bought 6 packs of soap given the conditions -/
theorem nancy_soap_packs : 
  ∀ (bars_per_pack total_bars : ℕ),
    bars_per_pack = 5 →
    total_bars = 30 →
    total_bars / bars_per_pack = 6 := by
  sorry

end NUMINAMATH_CALUDE_nancy_soap_packs_l2524_252438


namespace NUMINAMATH_CALUDE_exists_term_with_nine_l2524_252434

/-- Represents an arithmetic progression with natural number first term and common difference -/
structure ArithmeticProgression where
  first_term : ℕ
  common_difference : ℕ

/-- Predicate to check if a natural number contains the digit 9 -/
def contains_digit_nine (n : ℕ) : Prop := sorry

/-- Theorem stating that there exists a term in the arithmetic progression containing the digit 9 -/
theorem exists_term_with_nine (ap : ArithmeticProgression) : 
  ∃ (k : ℕ), contains_digit_nine (ap.first_term + k * ap.common_difference) := by sorry

end NUMINAMATH_CALUDE_exists_term_with_nine_l2524_252434


namespace NUMINAMATH_CALUDE_expression_value_l2524_252410

theorem expression_value :
  let x : ℤ := -2
  let y : ℤ := 1
  let z : ℤ := 1
  x^2 * y * z - x * y * z^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l2524_252410


namespace NUMINAMATH_CALUDE_max_stick_length_l2524_252422

theorem max_stick_length (a b c : ℕ) (ha : a = 24) (hb : b = 32) (hc : c = 44) :
  Nat.gcd a (Nat.gcd b c) = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_stick_length_l2524_252422


namespace NUMINAMATH_CALUDE_maximize_product_l2524_252451

theorem maximize_product (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : x + y = 50) :
  x^4 * y^3 ≤ (200/7)^4 * (150/7)^3 ∧
  x^4 * y^3 = (200/7)^4 * (150/7)^3 ↔ x = 200/7 ∧ y = 150/7 := by
  sorry

end NUMINAMATH_CALUDE_maximize_product_l2524_252451


namespace NUMINAMATH_CALUDE_unique_p_for_natural_roots_l2524_252488

def cubic_equation (p : ℝ) (x : ℝ) : ℝ :=
  5 * x^3 - 5 * (p + 1) * x^2 + (71 * p - 1) * x + 1 - 66 * p

theorem unique_p_for_natural_roots :
  ∃! p : ℝ, p = 76 ∧
  ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  cubic_equation p x = 0 ∧
  cubic_equation p y = 0 ∧
  cubic_equation p z = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_p_for_natural_roots_l2524_252488


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2524_252416

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 1) :
  (a^2 + 1) / a + (2*b^2 + 1) / b ≥ 4 + 2*Real.sqrt 2 :=
by sorry

theorem min_value_achievable :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 2*b = 1 ∧
  (a^2 + 1) / a + (2*b^2 + 1) / b = 4 + 2*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2524_252416


namespace NUMINAMATH_CALUDE_x3y2z3_coefficient_in_x_plus_y_plus_z_to_8_l2524_252480

def multinomial_coefficient (n : ℕ) (a b c : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial a * Nat.factorial b * Nat.factorial c)

theorem x3y2z3_coefficient_in_x_plus_y_plus_z_to_8 :
  multinomial_coefficient 8 3 2 3 = 560 := by
  sorry

end NUMINAMATH_CALUDE_x3y2z3_coefficient_in_x_plus_y_plus_z_to_8_l2524_252480


namespace NUMINAMATH_CALUDE_concurrent_lines_through_circumcenter_l2524_252400

-- Define the basic structures
structure Point :=
  (x y : ℝ)

structure Triangle :=
  (A B C : Point)

structure Line :=
  (p1 p2 : Point)

-- Define the properties
def isAcuteAngled (t : Triangle) : Prop := sorry

def altitudeFoot (t : Triangle) (v : Point) : Point := sorry

def perpendicularFoot (p : Point) (l : Line) : Point := sorry

def isOn (p : Point) (l : Line) : Prop := sorry

def intersectionPoint (l1 l2 : Line) : Point := sorry

def circumcenter (t : Triangle) : Point := sorry

-- Main theorem
theorem concurrent_lines_through_circumcenter 
  (t : Triangle) 
  (hAcute : isAcuteAngled t)
  (D : Point) (hD : D = altitudeFoot t t.A)
  (E : Point) (hE : E = altitudeFoot t t.B)
  (F : Point) (hF : F = altitudeFoot t t.C)
  (P : Point) (hP : P = perpendicularFoot t.A (Line.mk E F))
  (Q : Point) (hQ : Q = perpendicularFoot t.B (Line.mk F D))
  (R : Point) (hR : R = perpendicularFoot t.C (Line.mk D E)) :
  ∃ O : Point, 
    isOn O (Line.mk t.A P) ∧ 
    isOn O (Line.mk t.B Q) ∧ 
    isOn O (Line.mk t.C R) ∧
    O = circumcenter t :=
sorry

end NUMINAMATH_CALUDE_concurrent_lines_through_circumcenter_l2524_252400


namespace NUMINAMATH_CALUDE_game_score_problem_l2524_252487

theorem game_score_problem (total_questions : ℕ) (correct_points : ℤ) (incorrect_points : ℤ) (total_score : ℤ) :
  total_questions = 30 →
  correct_points = 7 →
  incorrect_points = -12 →
  total_score = 77 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_answers * correct_points + (total_questions - correct_answers) * incorrect_points = total_score ∧
    correct_answers = 23 := by
  sorry

end NUMINAMATH_CALUDE_game_score_problem_l2524_252487


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l2524_252493

theorem area_between_concentric_circles 
  (R r : ℝ) 
  (h_positive_R : R > 0) 
  (h_positive_r : r > 0) 
  (h_R_greater_r : R > r) 
  (h_tangent : r^2 + 5^2 = R^2) : 
  π * (R^2 - r^2) = 25 * π := by
sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l2524_252493


namespace NUMINAMATH_CALUDE_prob_different_colors_bag_l2524_252423

/-- Represents the number of chips of each color in the bag -/
structure ChipCounts where
  blue : ℕ
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the probability of drawing two chips of different colors -/
def probDifferentColors (counts : ChipCounts) : ℚ :=
  let total := counts.blue + counts.red + counts.yellow + counts.green
  let probBlue := counts.blue / total
  let probRed := counts.red / total
  let probYellow := counts.yellow / total
  let probGreen := counts.green / total
  let probDiffAfterBlue := (total - counts.blue) / total
  let probDiffAfterRed := (total - counts.red) / total
  let probDiffAfterYellow := (total - counts.yellow) / total
  let probDiffAfterGreen := (total - counts.green) / total
  probBlue * probDiffAfterBlue + probRed * probDiffAfterRed +
  probYellow * probDiffAfterYellow + probGreen * probDiffAfterGreen

/-- The main theorem stating the probability of drawing two chips of different colors -/
theorem prob_different_colors_bag :
  probDifferentColors { blue := 6, red := 5, yellow := 4, green := 3 } = 119 / 162 := by
  sorry


end NUMINAMATH_CALUDE_prob_different_colors_bag_l2524_252423


namespace NUMINAMATH_CALUDE_black_and_white_cartridge_cost_l2524_252486

/-- The cost of a black-and-white printer cartridge -/
def black_and_white_cost : ℕ := sorry

/-- The cost of a color printer cartridge -/
def color_cost : ℕ := 32

/-- The total cost of printer cartridges -/
def total_cost : ℕ := 123

/-- The number of color cartridges needed -/
def num_color_cartridges : ℕ := 3

/-- The number of black-and-white cartridges needed -/
def num_black_and_white_cartridges : ℕ := 1

theorem black_and_white_cartridge_cost :
  black_and_white_cost = 27 :=
by sorry

end NUMINAMATH_CALUDE_black_and_white_cartridge_cost_l2524_252486


namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_and_twelve_l2524_252447

def repeating_decimal : ℚ := 356 / 999

theorem product_of_repeating_decimal_and_twelve :
  repeating_decimal * 12 = 1424 / 333 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_and_twelve_l2524_252447


namespace NUMINAMATH_CALUDE_amaya_total_score_l2524_252499

/-- Represents the scores in different subjects -/
structure Scores where
  music : ℕ
  social_studies : ℕ
  arts : ℕ
  maths : ℕ

/-- Calculates the total score across all subjects -/
def total_score (s : Scores) : ℕ :=
  s.music + s.social_studies + s.arts + s.maths

/-- Theorem stating the total score given the conditions -/
theorem amaya_total_score :
  ∀ s : Scores,
  s.music = 70 →
  s.social_studies = s.music + 10 →
  s.maths = s.arts - 20 →
  s.maths = (9 * s.arts) / 10 →
  total_score s = 530 := by
  sorry

#check amaya_total_score

end NUMINAMATH_CALUDE_amaya_total_score_l2524_252499


namespace NUMINAMATH_CALUDE_binomial_12_choose_5_l2524_252465

theorem binomial_12_choose_5 : Nat.choose 12 5 = 792 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_5_l2524_252465


namespace NUMINAMATH_CALUDE_smallest_multiple_40_over_100_l2524_252405

theorem smallest_multiple_40_over_100 : ∀ n : ℕ, n > 0 ∧ 40 ∣ n ∧ n > 100 → n ≥ 120 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_40_over_100_l2524_252405


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l2524_252403

theorem fraction_equation_solution (a b : ℝ) (h : a / b = 5 / 4) :
  ∃ x : ℝ, (4 * a + x * b) / (4 * a - x * b) = 4 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l2524_252403


namespace NUMINAMATH_CALUDE_range_of_a_for_inequality_l2524_252435

theorem range_of_a_for_inequality (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ 2^x * (3*x + a) < 1) ↔ a < 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_inequality_l2524_252435


namespace NUMINAMATH_CALUDE_intersection_line_equation_l2524_252421

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 7 = 0
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 - 6*y - 27 = 0

-- Define the line AB
def line_AB (x y : ℝ) : Prop := 3*x - 3*y - 10 = 0

-- Theorem statement
theorem intersection_line_equation :
  ∀ A B : ℝ × ℝ,
  (circle_C1 A.1 A.2 ∧ circle_C2 A.1 A.2) →
  (circle_C1 B.1 B.2 ∧ circle_C2 B.1 B.2) →
  A ≠ B →
  line_AB A.1 A.2 ∧ line_AB B.1 B.2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l2524_252421


namespace NUMINAMATH_CALUDE_min_a_for_quadratic_roots_in_unit_interval_l2524_252445

theorem min_a_for_quadratic_roots_in_unit_interval :
  ∀ (a b c : ℤ) (α β : ℝ),
    a > 0 →
    (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = α ∨ x = β) →
    0 < α →
    α < β →
    β < 1 →
    a ≥ 5 ∧ ∃ (a₀ b₀ c₀ : ℤ) (α₀ β₀ : ℝ),
      a₀ = 5 ∧
      a₀ > 0 ∧
      (∀ x : ℝ, a₀ * x^2 + b₀ * x + c₀ = 0 ↔ x = α₀ ∨ x = β₀) ∧
      0 < α₀ ∧
      α₀ < β₀ ∧
      β₀ < 1 :=
by sorry

end NUMINAMATH_CALUDE_min_a_for_quadratic_roots_in_unit_interval_l2524_252445


namespace NUMINAMATH_CALUDE_grace_earnings_l2524_252409

theorem grace_earnings (weekly_charge : ℕ) (payment_interval : ℕ) (total_weeks : ℕ) (total_earnings : ℕ) : 
  weekly_charge = 300 →
  payment_interval = 2 →
  total_weeks = 6 →
  total_earnings = 1800 →
  total_weeks * weekly_charge = total_earnings :=
by
  sorry

end NUMINAMATH_CALUDE_grace_earnings_l2524_252409


namespace NUMINAMATH_CALUDE_susan_babysitting_earnings_l2524_252414

def susan_earnings (initial : ℝ) : Prop :=
  let after_clothes := initial / 2
  let after_books := after_clothes / 2
  after_books = 150

theorem susan_babysitting_earnings :
  ∃ (initial : ℝ), susan_earnings initial ∧ initial = 600 :=
sorry

end NUMINAMATH_CALUDE_susan_babysitting_earnings_l2524_252414


namespace NUMINAMATH_CALUDE_stamp_collection_value_l2524_252492

theorem stamp_collection_value (partial_value : ℚ) (partial_fraction : ℚ) (total_value : ℚ) : 
  partial_fraction = 4/7 ∧ partial_value = 28 → total_value = 49 :=
by sorry

end NUMINAMATH_CALUDE_stamp_collection_value_l2524_252492


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l2524_252402

/-- Given a complex number z such that z / (1 - z) = 2i, prove that z is in the first quadrant -/
theorem z_in_first_quadrant (z : ℂ) (h : z / (1 - z) = Complex.I * 2) : 
  Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l2524_252402


namespace NUMINAMATH_CALUDE_cafeteria_apples_l2524_252482

theorem cafeteria_apples (initial_apples : ℕ) : 
  (initial_apples - 2 + 23 = 38) → initial_apples = 17 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l2524_252482


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_l2524_252469

theorem quadratic_root_implies_a (a : ℝ) :
  (∃ x : ℝ, x^2 + 3*x + a = 0) ∧ (2^2 + 3*2 + a = 0) → a = -10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_l2524_252469


namespace NUMINAMATH_CALUDE_remainder_polynomial_division_l2524_252413

theorem remainder_polynomial_division (z : ℂ) : 
  ∃ (Q R : ℂ → ℂ), 
    (∀ z, z^2023 - 1 = (z^3 - 1) * (Q z) + R z) ∧ 
    (∃ (a b c : ℂ), ∀ z, R z = a*z^2 + b*z + c) ∧
    R z = z^2 + z - 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_polynomial_division_l2524_252413


namespace NUMINAMATH_CALUDE_multiply_specific_numbers_l2524_252429

theorem multiply_specific_numbers : 469160 * 999999 = 469159530840 := by
  sorry

end NUMINAMATH_CALUDE_multiply_specific_numbers_l2524_252429


namespace NUMINAMATH_CALUDE_paul_books_theorem_l2524_252479

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

end NUMINAMATH_CALUDE_paul_books_theorem_l2524_252479


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l2524_252448

theorem geometric_series_common_ratio :
  let a₁ : ℚ := 8 / 10
  let a₂ : ℚ := -6 / 15
  let a₃ : ℚ := 54 / 225
  let r : ℚ := a₂ / a₁
  (∀ n : ℕ, n ≥ 2 → (a₁ * r ^ (n - 1) = if n % 2 = 0 then -a₁ * (1 / 2) ^ (n - 1) else a₁ * (1 / 2) ^ (n - 1))) →
  r = -1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l2524_252448


namespace NUMINAMATH_CALUDE_sixth_root_equation_l2524_252458

theorem sixth_root_equation (x : ℝ) : 
  (x * (x^4)^(1/3))^(1/6) = 2 → x = 2^(18/7) := by
sorry

end NUMINAMATH_CALUDE_sixth_root_equation_l2524_252458


namespace NUMINAMATH_CALUDE_no_real_sqrt_negative_number_l2524_252474

theorem no_real_sqrt_negative_number (x : ℝ) :
  x = -2.5 ∨ x = 0 ∨ x = 2.1 ∨ x = 6 →
  (∃ y : ℝ, y ^ 2 = x) ↔ x ≠ -2.5 :=
by sorry

end NUMINAMATH_CALUDE_no_real_sqrt_negative_number_l2524_252474


namespace NUMINAMATH_CALUDE_only_valid_pythagorean_triple_l2524_252426

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem only_valid_pythagorean_triple :
  ¬ is_pythagorean_triple 3 4 5 ∧
  ¬ is_pythagorean_triple 2 2 (2 * 2) ∧
  ¬ is_pythagorean_triple 4 5 6 ∧
  is_pythagorean_triple 5 12 13 :=
by sorry

end NUMINAMATH_CALUDE_only_valid_pythagorean_triple_l2524_252426


namespace NUMINAMATH_CALUDE_circle_op_properties_l2524_252452

-- Define the set A as ordered pairs of real numbers
def A : Type := ℝ × ℝ

-- Define the operation ⊙
def circle_op (α β : A) : A :=
  let (a, b) := α
  let (c, d) := β
  (a * d + b * c, b * d - a * c)

-- Theorem statement
theorem circle_op_properties :
  -- Part 1: Specific calculation
  circle_op (2, 3) (-1, 4) = (5, 14) ∧
  -- Part 2: Commutativity
  (∀ α β : A, circle_op α β = circle_op β α) ∧
  -- Part 3: Identity element
  (∃ I : A, ∀ α : A, circle_op I α = α ∧ circle_op α I = α) ∧
  (∀ I : A, (∀ α : A, circle_op I α = α ∧ circle_op α I = α) → I = (0, 1)) :=
by sorry

end NUMINAMATH_CALUDE_circle_op_properties_l2524_252452


namespace NUMINAMATH_CALUDE_line_relationships_l2524_252473

-- Define the slopes of the lines
def slope1 : ℚ := 2
def slope2 : ℚ := 3
def slope3 : ℚ := 2
def slope4 : ℚ := 3/2
def slope5 : ℚ := 1/2

-- Define a function to check if two slopes are parallel
def are_parallel (m1 m2 : ℚ) : Prop := m1 = m2

-- Define a function to check if two slopes are perpendicular
def are_perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

-- Define the list of all slopes
def slopes : List ℚ := [slope1, slope2, slope3, slope4, slope5]

-- Theorem statement
theorem line_relationships :
  (∃! (i j : Fin 5), i < j ∧ are_parallel (slopes.get i) (slopes.get j)) ∧
  (∀ (i j : Fin 5), i < j → ¬are_perpendicular (slopes.get i) (slopes.get j)) :=
sorry

end NUMINAMATH_CALUDE_line_relationships_l2524_252473


namespace NUMINAMATH_CALUDE_new_customers_calculation_l2524_252461

theorem new_customers_calculation (initial_customers final_customers : ℕ) 
  (h1 : initial_customers = 3)
  (h2 : final_customers = 8) :
  final_customers - initial_customers = 5 := by
  sorry

end NUMINAMATH_CALUDE_new_customers_calculation_l2524_252461


namespace NUMINAMATH_CALUDE_russells_earnings_l2524_252475

/-- Proof of Russell's earnings --/
theorem russells_earnings (vika_earnings breanna_earnings saheed_earnings kayla_earnings russell_earnings : ℕ) : 
  vika_earnings = 84 →
  kayla_earnings = vika_earnings - 30 →
  saheed_earnings = 4 * kayla_earnings →
  breanna_earnings = saheed_earnings + (saheed_earnings / 4) →
  russell_earnings = 2 * (breanna_earnings - kayla_earnings) →
  russell_earnings = 432 := by
  sorry

end NUMINAMATH_CALUDE_russells_earnings_l2524_252475


namespace NUMINAMATH_CALUDE_jenny_sweets_problem_l2524_252441

theorem jenny_sweets_problem : ∃ n : ℕ+, 
  5 ∣ n ∧ 6 ∣ n ∧ ¬(12 ∣ n) ∧ n = 90 := by
  sorry

end NUMINAMATH_CALUDE_jenny_sweets_problem_l2524_252441


namespace NUMINAMATH_CALUDE_f_always_positive_l2524_252459

-- Define a triangle with side lengths a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  -- Triangle inequality conditions
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  ineq_ab : a + b > c
  ineq_bc : b + c > a
  ineq_ca : c + a > b

-- Define the function f(x)
def f (t : Triangle) (x : ℝ) : ℝ :=
  t.b^2 * x^2 + (t.b^2 + t.c^2 - t.a^2) * x + t.c^2

-- Theorem statement
theorem f_always_positive (t : Triangle) : ∀ x : ℝ, f t x > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_always_positive_l2524_252459


namespace NUMINAMATH_CALUDE_theater_seats_l2524_252464

/-- Represents a theater with an arithmetic progression of seats per row -/
structure Theater where
  first_row_seats : ℕ
  seat_increase : ℕ
  last_row_seats : ℕ

/-- Calculates the total number of seats in the theater -/
def total_seats (t : Theater) : ℕ :=
  let n := (t.last_row_seats - t.first_row_seats) / t.seat_increase + 1
  n * (t.first_row_seats + t.last_row_seats) / 2

/-- Theorem stating that a theater with given properties has 770 seats -/
theorem theater_seats :
  ∀ t : Theater,
    t.first_row_seats = 14 →
    t.seat_increase = 2 →
    t.last_row_seats = 56 →
    total_seats t = 770 := by
  sorry

#eval total_seats { first_row_seats := 14, seat_increase := 2, last_row_seats := 56 }

end NUMINAMATH_CALUDE_theater_seats_l2524_252464


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2524_252463

theorem partial_fraction_decomposition :
  ∃ (a b c : ℤ), 
    (1 : ℚ) / 2015 = a / 5 + b / 13 + c / 31 ∧
    0 ≤ a ∧ a < 5 ∧
    0 ≤ b ∧ b < 13 ∧
    a + b = 14 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2524_252463


namespace NUMINAMATH_CALUDE_magnitude_b_cos_angle_ab_l2524_252497

-- Define the vectors
def a : ℝ × ℝ := (4, 3)
def b : ℝ × ℝ := (-1, 2)

-- Theorem for the magnitude of vector b
theorem magnitude_b : Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = Real.sqrt 5 := by sorry

-- Theorem for the cosine of the angle between vectors a and b
theorem cos_angle_ab : 
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) * Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2))) 
  = (2 * Real.sqrt 5) / 25 := by sorry

end NUMINAMATH_CALUDE_magnitude_b_cos_angle_ab_l2524_252497


namespace NUMINAMATH_CALUDE_red_balls_count_l2524_252455

/-- Given a jar with white and red balls where the ratio of white to red balls is 3:2,
    and there are 9 white balls, prove that the number of red balls is 6. -/
theorem red_balls_count (white_balls : ℕ) (red_balls : ℕ) : 
  (white_balls : ℚ) / red_balls = 3 / 2 →
  white_balls = 9 →
  red_balls = 6 := by
sorry

end NUMINAMATH_CALUDE_red_balls_count_l2524_252455


namespace NUMINAMATH_CALUDE_frequency_converges_to_half_l2524_252462

/-- A coin toss experiment -/
structure CoinToss where
  /-- The probability of getting heads in a single toss -/
  probHeads : ℝ
  /-- The coin is fair -/
  isFair : probHeads = 0.5

/-- The frequency of heads after n tosses -/
def frequency (c : CoinToss) (n : ℕ) : ℝ :=
  sorry

/-- The theorem stating that the frequency of heads converges to 0.5 as n approaches infinity -/
theorem frequency_converges_to_half (c : CoinToss) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |frequency c n - 0.5| < ε :=
sorry

end NUMINAMATH_CALUDE_frequency_converges_to_half_l2524_252462


namespace NUMINAMATH_CALUDE_circle_diameter_from_viewing_angles_l2524_252450

theorem circle_diameter_from_viewing_angles 
  (r : ℝ) (d α β : ℝ) 
  (h_positive : r > 0 ∧ d > 0)
  (h_angles : 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2) :
  2 * r = (d * Real.sin α * Real.sin β) / (Real.sin ((α + β)/2) * Real.cos ((α - β)/2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_from_viewing_angles_l2524_252450


namespace NUMINAMATH_CALUDE_largest_n_for_trig_inequality_l2524_252477

theorem largest_n_for_trig_inequality : 
  (∃ (n : ℕ), n > 0 ∧ (∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 2/n)) ∧
  (∀ (m : ℕ), m > 6 → ∃ (x : ℝ), (Real.sin x)^m + (Real.cos x)^m < 2/m) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_trig_inequality_l2524_252477


namespace NUMINAMATH_CALUDE_polynomial_M_proof_l2524_252432

-- Define the polynomial M as a function of x and y
def M (x y : ℝ) : ℝ := 2 * x * y - 1

-- Theorem statement
theorem polynomial_M_proof :
  -- Given condition
  (∀ x y : ℝ, M x y + (2 * x^2 * y - 3 * x * y + 1) = 2 * x^2 * y - x * y) →
  -- Conclusion 1: M is correctly defined
  (∀ x y : ℝ, M x y = 2 * x * y - 1) ∧
  -- Conclusion 2: M(-1, 2) = -5
  (M (-1) 2 = -5) := by
sorry

end NUMINAMATH_CALUDE_polynomial_M_proof_l2524_252432


namespace NUMINAMATH_CALUDE_find_a_l2524_252439

-- Define the sets U and A
def U (a : ℝ) : Set ℝ := {2, 3, a^2 + 2*a - 3}
def A (a : ℝ) : Set ℝ := {|2*a - 1|, 2}

-- Define the theorem
theorem find_a : ∃ (a : ℝ), 
  (U a \ A a = {5}) ∧ 
  (A a ⊆ U a) ∧
  (a = 2) := by sorry

end NUMINAMATH_CALUDE_find_a_l2524_252439


namespace NUMINAMATH_CALUDE_sock_order_ratio_l2524_252489

/-- Represents the number of pairs of socks and their prices -/
structure SockOrder where
  grey_pairs : ℕ
  white_pairs : ℕ
  white_price : ℝ

/-- Calculates the total cost of a sock order -/
def total_cost (order : SockOrder) : ℝ :=
  order.grey_pairs * (3 * order.white_price) + order.white_pairs * order.white_price

theorem sock_order_ratio (order : SockOrder) :
  order.grey_pairs = 6 →
  total_cost { grey_pairs := order.white_pairs, white_pairs := order.grey_pairs, white_price := order.white_price } = 1.25 * total_cost order →
  (order.grey_pairs : ℚ) / order.white_pairs = 6 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sock_order_ratio_l2524_252489


namespace NUMINAMATH_CALUDE_books_written_proof_l2524_252412

/-- The number of books written by Zig -/
def zig_books : ℕ := 60

/-- The number of books written by Flo -/
def flo_books : ℕ := zig_books / 4

/-- The number of books written by Tim -/
def tim_books : ℕ := flo_books / 2

/-- The total number of books written by Zig, Flo, and Tim -/
def total_books : ℕ := zig_books + flo_books + tim_books

theorem books_written_proof : total_books = 82 := by
  sorry

end NUMINAMATH_CALUDE_books_written_proof_l2524_252412


namespace NUMINAMATH_CALUDE_felix_lift_problem_l2524_252468

/-- Felix's weight lifting problem -/
theorem felix_lift_problem (felix_weight : ℝ) (felix_brother_weight : ℝ) (felix_brother_lift : ℝ) :
  (felix_brother_weight = 2 * felix_weight) →
  (felix_brother_lift = 3 * felix_brother_weight) →
  (felix_brother_lift = 600) →
  (1.5 * felix_weight = 150) :=
by
  sorry

#check felix_lift_problem

end NUMINAMATH_CALUDE_felix_lift_problem_l2524_252468


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l2524_252470

/-- The area of a square with a diagonal of 3.8 meters is 7.22 square meters. -/
theorem square_area_from_diagonal (d : ℝ) (h : d = 3.8) :
  let s := d / Real.sqrt 2
  s ^ 2 = 7.22 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l2524_252470


namespace NUMINAMATH_CALUDE_shoe_pairs_calculation_shoe_pairs_proof_l2524_252498

/-- Given a total number of shoes and the probability of selecting two shoes of the same color
    without replacement, calculate the number of pairs of shoes. -/
theorem shoe_pairs_calculation (total_shoes : ℕ) (probability : ℚ) : ℕ :=
  let pairs := total_shoes / 2
  let calculated_prob := 1 / (total_shoes - 1 : ℚ)
  if total_shoes = 12 ∧ probability = 1/11 ∧ calculated_prob = probability
  then pairs
  else 0

/-- Prove that given 12 shoes in total and a probability of 1/11 for selecting 2 shoes
    of the same color without replacement, the number of pairs of shoes is 6. -/
theorem shoe_pairs_proof :
  shoe_pairs_calculation 12 (1/11) = 6 := by
  sorry

end NUMINAMATH_CALUDE_shoe_pairs_calculation_shoe_pairs_proof_l2524_252498


namespace NUMINAMATH_CALUDE_closest_integer_to_largest_root_squared_l2524_252443

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 8*x^2 - 2*x + 3

-- State the theorem
theorem closest_integer_to_largest_root_squared : 
  ∃ (a b c : ℝ), 
    (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧ 
    (a > b ∧ a > c) ∧
    (abs (a^2 - 67) < 1) :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_largest_root_squared_l2524_252443


namespace NUMINAMATH_CALUDE_trigonometric_equality_l2524_252446

theorem trigonometric_equality : 
  (2 * Real.sin (47 * π / 180) - Real.sqrt 3 * Real.sin (17 * π / 180)) / Real.cos (17 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l2524_252446


namespace NUMINAMATH_CALUDE_square_minus_product_equals_one_l2524_252490

theorem square_minus_product_equals_one : 2014^2 - 2013 * 2015 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_equals_one_l2524_252490


namespace NUMINAMATH_CALUDE_memory_sequence_increment_prime_or_one_l2524_252456

/-- Sequence representing the memory cell value after each step -/
def memory_sequence : ℕ → ℕ
  | 0 => 6
  | (n + 1) => memory_sequence n + Nat.gcd (memory_sequence n) (n + 1)

/-- Proposition: The difference between consecutive terms is either 1 or prime -/
theorem memory_sequence_increment_prime_or_one :
  ∀ n : ℕ, (memory_sequence (n + 1) - memory_sequence n = 1) ∨ 
    Nat.Prime (memory_sequence (n + 1) - memory_sequence n) :=
by
  sorry


end NUMINAMATH_CALUDE_memory_sequence_increment_prime_or_one_l2524_252456


namespace NUMINAMATH_CALUDE_hyperbola_condition_l2524_252401

/-- For the equation x²/(2+m) - y²/(m+1) = 1 to represent a hyperbola, 
    m must satisfy: m > -1 or m < -2 -/
theorem hyperbola_condition (m : ℝ) : 
  (∃ x y : ℝ, x^2 / (2 + m) - y^2 / (m + 1) = 1 ∧ 
   (2 + m ≠ 0 ∧ m + 1 ≠ 0)) ↔ (m > -1 ∨ m < -2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l2524_252401


namespace NUMINAMATH_CALUDE_choir_arrangement_max_l2524_252430

theorem choir_arrangement_max (n : ℕ) : 
  (∃ k : ℕ, n = k^2 + 11) ∧ 
  (∃ x : ℕ, n = x * (x + 5)) →
  n ≤ 126 :=
sorry

end NUMINAMATH_CALUDE_choir_arrangement_max_l2524_252430


namespace NUMINAMATH_CALUDE_remainder_101_pow_37_mod_100_l2524_252472

theorem remainder_101_pow_37_mod_100 : 101^37 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_101_pow_37_mod_100_l2524_252472


namespace NUMINAMATH_CALUDE_athlete_arrangements_l2524_252420

def male_athletes : ℕ := 7
def female_athletes : ℕ := 3

theorem athlete_arrangements :
  let total_athletes := male_athletes + female_athletes
  let arrangements_case1 := (male_athletes.factorial) * (male_athletes - 1) * (male_athletes - 2) * (male_athletes - 3)
  let arrangements_case2 := 2 * (female_athletes.factorial) * (male_athletes.factorial)
  let arrangements_case3 := (total_athletes + 1).factorial * (female_athletes.factorial)
  (arrangements_case1 = 604800) ∧
  (arrangements_case2 = 60480) ∧
  (arrangements_case3 = 241920) := by
  sorry

#eval male_athletes.factorial * (male_athletes - 1) * (male_athletes - 2) * (male_athletes - 3)
#eval 2 * female_athletes.factorial * male_athletes.factorial
#eval (male_athletes + female_athletes + 1).factorial * female_athletes.factorial

end NUMINAMATH_CALUDE_athlete_arrangements_l2524_252420


namespace NUMINAMATH_CALUDE_f_3_eq_2488_l2524_252454

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 7x^5 + 12x^4 - 5x^3 - 6x^2 + 3x - 5 -/
def f (x : ℝ) : ℝ := 
  horner_eval [7, 12, -5, -6, 3, -5] x

theorem f_3_eq_2488 : f 3 = 2488 := by
  sorry

end NUMINAMATH_CALUDE_f_3_eq_2488_l2524_252454


namespace NUMINAMATH_CALUDE_symmetric_point_is_correct_l2524_252417

/-- The line of symmetry --/
def line_of_symmetry (x y : ℝ) : Prop := x + 2*y - 10 = 0

/-- The original point --/
def original_point : ℝ × ℝ := (1, 2)

/-- The symmetric point --/
def symmetric_point : ℝ × ℝ := (3, 6)

/-- Checks if two points are symmetric with respect to a line --/
def is_symmetric (p1 p2 : ℝ × ℝ) : Prop :=
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  line_of_symmetry midpoint.1 midpoint.2 ∧
  (p2.2 - p1.2) * (1 : ℝ) = (p2.1 - p1.1) * (-2 : ℝ)

theorem symmetric_point_is_correct : 
  is_symmetric original_point symmetric_point :=
sorry

end NUMINAMATH_CALUDE_symmetric_point_is_correct_l2524_252417


namespace NUMINAMATH_CALUDE_remaining_distance_to_grandma_l2524_252433

theorem remaining_distance_to_grandma (total_distance driven_first driven_second : ℕ) 
  (h1 : total_distance = 78)
  (h2 : driven_first = 35)
  (h3 : driven_second = 18) : 
  total_distance - (driven_first + driven_second) = 25 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_to_grandma_l2524_252433


namespace NUMINAMATH_CALUDE_max_value_of_f_l2524_252449

-- Define the function f
def f (x : ℝ) : ℝ := x * (4 - x)

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 4 ∧ ∀ x, x ∈ Set.Ioo 0 4 → f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2524_252449


namespace NUMINAMATH_CALUDE_average_weight_abc_l2524_252424

theorem average_weight_abc (a b c : ℝ) : 
  (a + b) / 2 = 40 →
  (b + c) / 2 = 45 →
  b = 35 →
  (a + b + c) / 3 = 45 := by
sorry

end NUMINAMATH_CALUDE_average_weight_abc_l2524_252424


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2524_252496

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 1 + a 6 = 12) 
  (h_a4 : a 4 = 7) : 
  ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2524_252496


namespace NUMINAMATH_CALUDE_min_value_expression_l2524_252457

theorem min_value_expression (a b c : ℤ) (h1 : c > 0) (h2 : a = b + c) :
  (((a + b : ℚ) / (a - b)) + ((a - b : ℚ) / (a + b))) ≥ 2 ∧
  ∃ (a b : ℤ), ∃ (c : ℤ), c > 0 ∧ a = b + c ∧
    (((a + b : ℚ) / (a - b)) + ((a - b : ℚ) / (a + b))) = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2524_252457


namespace NUMINAMATH_CALUDE_largest_n_for_product_2016_l2524_252407

/-- An arithmetic sequence with integer terms -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem largest_n_for_product_2016 :
  ∀ a b : ℕ → ℤ,
  ArithmeticSequence a →
  ArithmeticSequence b →
  a 1 = 1 →
  b 1 = 1 →
  a 2 ≤ b 2 →
  (∃ n : ℕ, a n * b n = 2016) →
  (∀ m : ℕ, (∃ n : ℕ, n > m ∧ a n * b n = 2016) → m ≤ 32) :=
sorry

end NUMINAMATH_CALUDE_largest_n_for_product_2016_l2524_252407


namespace NUMINAMATH_CALUDE_charity_race_fundraising_l2524_252484

/-- Proves that 30 students, with 10 raising $20 each and the rest raising $30 each, raise a total of $800 --/
theorem charity_race_fundraising (total_students : Nat) (group1_students : Nat) (group1_amount : Nat) (group2_amount : Nat) :
  total_students = 30 →
  group1_students = 10 →
  group1_amount = 20 →
  group2_amount = 30 →
  group1_students * group1_amount + (total_students - group1_students) * group2_amount = 800 := by
  sorry

#check charity_race_fundraising

end NUMINAMATH_CALUDE_charity_race_fundraising_l2524_252484


namespace NUMINAMATH_CALUDE_sum_of_roots_product_polynomials_l2524_252491

theorem sum_of_roots_product_polynomials :
  let p₁ : Polynomial ℝ := 3 * X^3 - 2 * X^2 + 9 * X - 15
  let p₂ : Polynomial ℝ := 4 * X^3 + 8 * X^2 - 4 * X + 24
  let roots := (p₁.roots.toFinset ∪ p₂.roots.toFinset).toList
  List.sum roots = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_product_polynomials_l2524_252491


namespace NUMINAMATH_CALUDE_exponent_calculation_l2524_252453

theorem exponent_calculation : (1 / ((-5^4)^2)) * (-5)^7 = -1/5 := by sorry

end NUMINAMATH_CALUDE_exponent_calculation_l2524_252453


namespace NUMINAMATH_CALUDE_total_coronavirus_cases_l2524_252471

-- Define the number of cases for each state
def new_york_cases : ℕ := 2000
def california_cases : ℕ := new_york_cases / 2
def texas_cases : ℕ := california_cases - 400

-- Theorem to prove
theorem total_coronavirus_cases : 
  new_york_cases + california_cases + texas_cases = 3600 := by
  sorry

end NUMINAMATH_CALUDE_total_coronavirus_cases_l2524_252471


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l2524_252425

theorem trigonometric_simplification (θ : Real) : 
  (Real.sin (2 * Real.pi - θ) * Real.cos (Real.pi + θ) * Real.cos (Real.pi / 2 + θ) * Real.cos (11 * Real.pi / 2 - θ)) / 
  (Real.cos (Real.pi - θ) * Real.sin (3 * Real.pi - θ) * Real.sin (-Real.pi - θ) * Real.sin (9 * Real.pi / 2 + θ)) = 
  -Real.tan θ := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l2524_252425


namespace NUMINAMATH_CALUDE_chocolate_cost_l2524_252404

theorem chocolate_cost (candy_cost : ℕ) (candy_count : ℕ) (chocolate_count : ℕ) (price_difference : ℕ) :
  candy_cost = 530 →
  candy_count = 12 →
  chocolate_count = 8 →
  price_difference = 5400 →
  candy_count * candy_cost = chocolate_count * (candy_count * candy_cost / chocolate_count - price_difference / chocolate_count) + price_difference →
  candy_count * candy_cost / chocolate_count - price_difference / chocolate_count = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_cost_l2524_252404


namespace NUMINAMATH_CALUDE_meaningful_expression_l2524_252440

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x - 3)) ↔ x > 3 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_expression_l2524_252440


namespace NUMINAMATH_CALUDE_space_creature_perimeter_calc_l2524_252483

/-- The perimeter of a space creature, which is a sector of a circle --/
def space_creature_perimeter (r : ℝ) (central_angle : ℝ) : ℝ :=
  r * central_angle + 2 * r

/-- Theorem: The perimeter of the space creature with radius 2 cm and central angle 270° is 3π + 4 cm --/
theorem space_creature_perimeter_calc :
  space_creature_perimeter 2 (3 * π / 2) = 3 * π + 4 := by
  sorry

#check space_creature_perimeter_calc

end NUMINAMATH_CALUDE_space_creature_perimeter_calc_l2524_252483


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l2524_252495

theorem sine_cosine_inequality (α : Real) (h1 : 0 < α) (h2 : α < π) :
  2 * Real.sin (2 * α) ≤ Real.cos (α / 2) ∧
  (2 * Real.sin (2 * α) = Real.cos (α / 2) ↔ α = π / 3) := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l2524_252495


namespace NUMINAMATH_CALUDE_ball_probabilities_l2524_252406

def total_balls : ℕ := 6
def red_balls : ℕ := 3
def white_balls : ℕ := 2
def black_balls : ℕ := 1
def drawn_balls : ℕ := 3

def prob_one_red_one_white : ℚ := 3 / 10
def prob_at_least_two_red : ℚ := 1 / 2
def prob_no_black : ℚ := 1 / 2

theorem ball_probabilities :
  (total_balls = red_balls + white_balls + black_balls) →
  (drawn_balls ≤ total_balls) →
  (prob_one_red_one_white = 3 / 10) ∧
  (prob_at_least_two_red = 1 / 2) ∧
  (prob_no_black = 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ball_probabilities_l2524_252406


namespace NUMINAMATH_CALUDE_x_plus_y_value_l2524_252428

theorem x_plus_y_value (x y : ℝ) 
  (eq1 : x + Real.cos y = 2010)
  (eq2 : x + 2010 * Real.sin y = 2009)
  (y_range : π / 2 ≤ y ∧ y ≤ π) :
  x + y = 2011 + π := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l2524_252428


namespace NUMINAMATH_CALUDE_mod_nine_equivalence_l2524_252419

theorem mod_nine_equivalence :
  ∃! n : ℤ, 0 ≤ n ∧ n < 9 ∧ -2222 ≡ n [ZMOD 9] ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_mod_nine_equivalence_l2524_252419


namespace NUMINAMATH_CALUDE_number_puzzle_l2524_252476

theorem number_puzzle : ∃ x : ℤ, (x - 10 = 15) ∧ (x + 5 = 30) := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2524_252476


namespace NUMINAMATH_CALUDE_group_photo_arrangements_l2524_252436

theorem group_photo_arrangements :
  let total_volunteers : ℕ := 6
  let male_volunteers : ℕ := 4
  let female_volunteers : ℕ := 2
  let elderly_people : ℕ := 2
  let elderly_arrangements : ℕ := 2  -- Number of ways to arrange elderly people
  let female_arrangements : ℕ := 2   -- Number of ways to arrange female volunteers
  let male_arrangements : ℕ := 24    -- Number of ways to arrange male volunteers (4!)
  
  total_volunteers = male_volunteers + female_volunteers + elderly_people →
  (elderly_arrangements * female_arrangements * male_arrangements) = 96 :=
by
  sorry

end NUMINAMATH_CALUDE_group_photo_arrangements_l2524_252436


namespace NUMINAMATH_CALUDE_smallest_divisible_by_15_16_18_l2524_252481

theorem smallest_divisible_by_15_16_18 : 
  ∃ n : ℕ, (n > 0) ∧ 
           (15 ∣ n) ∧ (16 ∣ n) ∧ (18 ∣ n) ∧ 
           (∀ m : ℕ, (m > 0) ∧ (15 ∣ m) ∧ (16 ∣ m) ∧ (18 ∣ m) → n ≤ m) ∧
           n = 720 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_15_16_18_l2524_252481


namespace NUMINAMATH_CALUDE_floor_equation_solution_l2524_252466

theorem floor_equation_solution (x : ℝ) : 
  (⌊⌊3 * x⌋ + 1/3⌋ = ⌊x + 5⌋) ↔ (7/3 ≤ x ∧ x < 3) := by
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l2524_252466


namespace NUMINAMATH_CALUDE_intersection_implies_c_18_l2524_252418

-- Define the functions
def f (x : ℝ) : ℝ := |x - 20| + |x + 18|
def g (c x : ℝ) : ℝ := x + c

-- Define the intersection condition
def unique_intersection (c : ℝ) : Prop :=
  ∃! x, f x = g c x

-- Theorem statement
theorem intersection_implies_c_18 :
  ∀ c : ℝ, unique_intersection c → c = 18 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_c_18_l2524_252418


namespace NUMINAMATH_CALUDE_min_distinct_values_for_given_conditions_l2524_252494

/-- Given a list of positive integers with a unique mode, this function returns the minimum number of distinct values that can occur in the list. -/
def min_distinct_values (list_size : ℕ) (mode_frequency : ℕ) : ℕ :=
  sorry

/-- Theorem stating the minimum number of distinct values for the given conditions -/
theorem min_distinct_values_for_given_conditions :
  min_distinct_values 2057 15 = 147 := by
  sorry

end NUMINAMATH_CALUDE_min_distinct_values_for_given_conditions_l2524_252494


namespace NUMINAMATH_CALUDE_abigail_report_time_l2524_252485

/-- Given a report length, typing speed, and words already written, 
    calculate the time required to finish the report. -/
def time_to_finish_report (total_words : ℕ) (words_per_half_hour : ℕ) (words_written : ℕ) : ℕ :=
  let words_remaining := total_words - words_written
  let minutes_per_word := 30 / words_per_half_hour
  words_remaining * minutes_per_word

/-- Proof that for the given conditions, the time to finish the report is 80 minutes. -/
theorem abigail_report_time : time_to_finish_report 1000 300 200 = 80 := by
  sorry

end NUMINAMATH_CALUDE_abigail_report_time_l2524_252485


namespace NUMINAMATH_CALUDE_bank_deposit_exceeds_400_l2524_252467

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (r^n - 1) / (r - 1)

def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Sunday"
  | 1 => "Monday"
  | 2 => "Tuesday"
  | 3 => "Wednesday"
  | 4 => "Thursday"
  | 5 => "Friday"
  | _ => "Saturday"

theorem bank_deposit_exceeds_400 :
  let a := 2  -- Initial deposit in cents
  let r := 3  -- Common ratio
  let target := 40000  -- Target amount in cents
  ∀ n : ℕ, n < 10 → geometric_sum a r n ≤ target ∧
  geometric_sum a r 10 > target ∧
  day_of_week 10 = "Tuesday" :=
by sorry

end NUMINAMATH_CALUDE_bank_deposit_exceeds_400_l2524_252467


namespace NUMINAMATH_CALUDE_sum_of_remainders_is_93_l2524_252444

def is_valid_number (n : ℕ) : Prop :=
  ∃ a b c d e : ℕ,
    n = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
    a = b + 2 ∧ b = c + 1 ∧ c = d + 1 ∧ d = e + 1 ∧
    0 ≤ e ∧ e < 10 ∧ 2 ≤ a ∧ a ≤ 6

def valid_numbers : List ℕ :=
  [23456, 34567, 45678, 56789, 67890]

theorem sum_of_remainders_is_93 :
  (valid_numbers.map (· % 43)).sum = 93 :=
sorry

end NUMINAMATH_CALUDE_sum_of_remainders_is_93_l2524_252444


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l2524_252427

/-- Given a point A in a Cartesian coordinate system with coordinates (-2, -3),
    its coordinates with respect to the origin are also (-2, -3). -/
theorem point_coordinates_wrt_origin :
  ∀ (A : ℝ × ℝ), A = (-2, -3) → A = (-2, -3) :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l2524_252427


namespace NUMINAMATH_CALUDE_nth_equation_solutions_l2524_252415

theorem nth_equation_solutions (n : ℕ+) :
  let eq := fun x : ℝ => x + (n^2 + n) / x + (2*n + 1)
  eq (-n : ℝ) = 0 ∧ eq (-(n + 1) : ℝ) = 0 :=
by sorry

end NUMINAMATH_CALUDE_nth_equation_solutions_l2524_252415


namespace NUMINAMATH_CALUDE_final_row_ordered_l2524_252408

variable (m n : ℕ)
variable (C : ℕ → ℕ → ℕ)

-- C[i][j] represents the card number at row i and column j
axiom row_ordered : ∀ i j k, j < k → C i j < C i k
axiom col_ordered : ∀ i j k, i < k → C i j < C k j

theorem final_row_ordered :
  ∀ i j k, j < k → C i j < C i k :=
sorry

end NUMINAMATH_CALUDE_final_row_ordered_l2524_252408


namespace NUMINAMATH_CALUDE_right_triangle_legs_l2524_252460

theorem right_triangle_legs (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  ((a = 16 ∧ b = 63) ∨ (a = 63 ∧ b = 16)) →  -- Possible leg lengths
  ∃ (x y : ℕ), x^2 + y^2 = 65^2 ∧ (x = 16 ∧ y = 63) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_legs_l2524_252460


namespace NUMINAMATH_CALUDE_john_weekly_loss_l2524_252431

/-- Represents John's tire production and sales scenario -/
structure TireProduction where
  daily_production : ℕ
  production_cost : ℚ
  selling_price_multiplier : ℚ
  potential_daily_sales : ℕ

/-- Calculates the weekly loss due to production limitations -/
def weekly_loss (t : TireProduction) : ℚ :=
  let profit_per_tire := t.production_cost * (t.selling_price_multiplier - 1)
  let daily_loss := profit_per_tire * (t.potential_daily_sales - t.daily_production)
  7 * daily_loss

/-- Theorem stating that given John's production scenario, the weekly loss is $175,000 -/
theorem john_weekly_loss :
  let john_production : TireProduction := {
    daily_production := 1000,
    production_cost := 250,
    selling_price_multiplier := 1.5,
    potential_daily_sales := 1200
  }
  weekly_loss john_production = 175000 := by
  sorry

end NUMINAMATH_CALUDE_john_weekly_loss_l2524_252431


namespace NUMINAMATH_CALUDE_impossible_mixture_l2524_252411

/-- Represents the properties of an ingredient -/
structure Ingredient :=
  (volume : ℝ)
  (water_content : ℝ)

/-- Proves that it's impossible to create a mixture with exactly 20% water content
    using the given volumes of tomato juice, tomato paste, and secret sauce -/
theorem impossible_mixture
  (tomato_juice : Ingredient)
  (tomato_paste : Ingredient)
  (secret_sauce : Ingredient)
  (h1 : tomato_juice.volume = 40)
  (h2 : tomato_juice.water_content = 0.9)
  (h3 : tomato_paste.volume = 20)
  (h4 : tomato_paste.water_content = 0.45)
  (h5 : secret_sauce.volume = 10)
  (h6 : secret_sauce.water_content = 0.7)
  : ¬ ∃ (x y z : ℝ),
    0 ≤ x ∧ x ≤ tomato_juice.volume ∧
    0 ≤ y ∧ y ≤ tomato_paste.volume ∧
    0 ≤ z ∧ z ≤ secret_sauce.volume ∧
    (x * tomato_juice.water_content + y * tomato_paste.water_content + z * secret_sauce.water_content) / (x + y + z) = 0.2 :=
sorry


end NUMINAMATH_CALUDE_impossible_mixture_l2524_252411


namespace NUMINAMATH_CALUDE_remainder_problem_l2524_252437

theorem remainder_problem (n : ℤ) (h : n % 11 = 3) : (5 * n - 9) % 11 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2524_252437


namespace NUMINAMATH_CALUDE_range_of_a_l2524_252478

theorem range_of_a (x a : ℝ) :
  (∀ x, (-4 < x - a ∧ x - a < 4) ↔ (1 < x ∧ x < 2)) →
  -2 ≤ a ∧ a ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2524_252478
