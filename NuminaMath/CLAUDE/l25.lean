import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_point_Q_l25_2503

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

-- Define points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the condition for point M
def M_condition (M : ℝ × ℝ) : Prop :=
  let (mx, my) := M
  (mx - 2) * (mx + 2) + my^2 = 0  -- MB ⊥ AB

-- Define the condition for point P
def P_condition (P : ℝ × ℝ) : Prop :=
  let (px, py) := P
  ellipse px py  -- P is on the ellipse

-- Define the condition for point Q
def Q_condition (Q : ℝ × ℝ) : Prop :=
  let (qx, qy) := Q
  qy = 0 ∧ qx ≠ -2 ∧ qx ≠ 2  -- Q is on x-axis and distinct from A and B

-- Define the circle condition
def circle_condition (M P Q : ℝ × ℝ) : Prop :=
  let (mx, my) := M
  let (px, py) := P
  let (qx, qy) := Q
  ∃ (I : ℝ × ℝ), 
    (I.1 - px) * (mx - px) + (I.2 - py) * (my - py) = 0 ∧  -- I is on BP
    (I.1 - mx) * (qx - mx) + (I.2 - my) * (qy - my) = 0 ∧  -- I is on MQ
    (I.1 - (mx + px) / 2)^2 + (I.2 - (my + py) / 2)^2 = ((mx - px)^2 + (my - py)^2) / 4  -- I is on the circle

theorem ellipse_point_Q : 
  ∀ (M P Q : ℝ × ℝ),
    M_condition M →
    P_condition P →
    Q_condition Q →
    circle_condition M P Q →
    Q = (0, 0) := by sorry

end NUMINAMATH_CALUDE_ellipse_point_Q_l25_2503


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_l25_2575

theorem negation_of_forall_positive (f : ℝ → ℝ) :
  (¬ ∀ x : ℝ, f x > 0) ↔ (∃ x : ℝ, f x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_l25_2575


namespace NUMINAMATH_CALUDE_sum_of_digits_8_pow_2004_l25_2522

/-- The sum of the tens digit and the units digit in the decimal representation of 8^2004 is 7 -/
theorem sum_of_digits_8_pow_2004 : ∃ (a b : ℕ), 
  8^2004 % 100 = 10 * a + b ∧ 
  a + b = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_8_pow_2004_l25_2522


namespace NUMINAMATH_CALUDE_triangle_inequality_l25_2561

theorem triangle_inequality (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_perimeter : a + b + c = 1) :
  a^2 + b^2 + c^2 + 4*a*b*c < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l25_2561


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l25_2592

theorem complex_fraction_simplification :
  (I : ℂ) / (3 + 4 * I) = (4 : ℂ) / 25 + (3 : ℂ) / 25 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l25_2592


namespace NUMINAMATH_CALUDE_noon_temperature_l25_2556

def morning_temp : ℤ := 4
def temp_drop : ℤ := 10

theorem noon_temperature :
  morning_temp - temp_drop = -6 := by
  sorry

end NUMINAMATH_CALUDE_noon_temperature_l25_2556


namespace NUMINAMATH_CALUDE_girls_in_math_class_l25_2545

theorem girls_in_math_class
  (boy_girl_ratio : ℚ)
  (math_science_ratio : ℚ)
  (science_lit_ratio : ℚ)
  (total_students : ℕ)
  (h1 : boy_girl_ratio = 5 / 8)
  (h2 : math_science_ratio = 7 / 4)
  (h3 : science_lit_ratio = 3 / 5)
  (h4 : total_students = 720) :
  ∃ (girls_math : ℕ), girls_math = 176 :=
by sorry

end NUMINAMATH_CALUDE_girls_in_math_class_l25_2545


namespace NUMINAMATH_CALUDE_first_act_clown_mobiles_l25_2521

/-- The number of clowns in each clown mobile -/
def clowns_per_mobile : ℕ := 28

/-- The total number of clowns in all clown mobiles -/
def total_clowns : ℕ := 140

/-- The number of clown mobiles -/
def num_clown_mobiles : ℕ := total_clowns / clowns_per_mobile

theorem first_act_clown_mobiles : num_clown_mobiles = 5 := by
  sorry

end NUMINAMATH_CALUDE_first_act_clown_mobiles_l25_2521


namespace NUMINAMATH_CALUDE_multiples_difference_cubed_zero_l25_2567

theorem multiples_difference_cubed_zero : 
  let a := (Finset.filter (fun x => x % 12 = 0 ∧ x > 0) (Finset.range 60)).card
  let b := (Finset.filter (fun x => x % 4 = 0 ∧ x % 3 = 0 ∧ x > 0) (Finset.range 60)).card
  (a - b)^3 = 0 := by
sorry

end NUMINAMATH_CALUDE_multiples_difference_cubed_zero_l25_2567


namespace NUMINAMATH_CALUDE_set_equality_l25_2558

theorem set_equality : {x : ℕ | x - 3 < 2} = {0, 1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l25_2558


namespace NUMINAMATH_CALUDE_simplify_expression_l25_2507

theorem simplify_expression : (9 * 10^10) / (3 * 10^3 - 2 * 10^3) = 9 * 10^7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l25_2507


namespace NUMINAMATH_CALUDE_negative_product_expression_B_l25_2548

theorem negative_product_expression_B : 
  let a : ℚ := -9
  let b : ℚ := 1/8
  let c : ℚ := -4/7
  let d : ℚ := 7
  let e : ℚ := -1/3
  a * b * c * d * e < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_product_expression_B_l25_2548


namespace NUMINAMATH_CALUDE_gcd_987654_123456_l25_2535

theorem gcd_987654_123456 : Nat.gcd 987654 123456 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_987654_123456_l25_2535


namespace NUMINAMATH_CALUDE_repeating_decimal_division_l25_2518

theorem repeating_decimal_division :
  let x : ℚ := 63 / 99
  let y : ℚ := 84 / 99
  x / y = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_division_l25_2518


namespace NUMINAMATH_CALUDE_final_catch_up_time_l25_2500

/-- Represents a person's movement --/
structure Person where
  speed : ℝ
  startTime : ℝ

/-- Represents the problem setup --/
structure ProblemSetup where
  personA : Person
  personB : Person
  catchUpTime1 : ℝ
  distanceAB : ℝ

/-- The main theorem --/
theorem final_catch_up_time (setup : ProblemSetup) 
  (h1 : setup.personA.speed * 3 = setup.personB.speed * 2) -- Speed ratio
  (h2 : setup.personA.startTime = 8) -- A starts at 8:00 AM
  (h3 : setup.personB.startTime = 9) -- B starts at 9:00 AM
  (h4 : setup.catchUpTime1 = 11) -- First catch-up at 11:00 AM
  : ∃ (finalTime : ℝ), finalTime = 12 + 48/60 := by
  sorry

#check final_catch_up_time

end NUMINAMATH_CALUDE_final_catch_up_time_l25_2500


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l25_2541

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  (∀ x : ℝ, (3 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ + a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 233 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l25_2541


namespace NUMINAMATH_CALUDE_birthday_celebration_men_count_l25_2515

/-- Proves that the number of men at a birthday celebration was 15 given the specified conditions. -/
theorem birthday_celebration_men_count :
  ∀ (total_guests women men children : ℕ),
    total_guests = 60 →
    women = total_guests / 2 →
    total_guests = women + men + children →
    50 = women + (men - men / 3) + (children - 5) →
    men = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_birthday_celebration_men_count_l25_2515


namespace NUMINAMATH_CALUDE_simplify_expression_l25_2568

theorem simplify_expression : (2^8 + 5^5) * (2^3 - (-2)^3)^7 = 9077567990336 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l25_2568


namespace NUMINAMATH_CALUDE_m_range_theorem_l25_2587

/-- The range of m values satisfying the given conditions -/
def m_range : Set ℝ :=
  Set.Ioc 1 2 ∪ Set.Ici 3

/-- Condition for the first equation to have two distinct negative roots -/
def has_two_negative_roots (m : ℝ) : Prop :=
  ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

/-- Condition for the second equation to have no real roots -/
def has_no_real_roots (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

/-- Main theorem statement -/
theorem m_range_theorem (m : ℝ) :
  (has_two_negative_roots m ∨ has_no_real_roots m) ∧
  ¬(has_two_negative_roots m ∧ has_no_real_roots m) ↔
  m ∈ m_range :=
sorry

end NUMINAMATH_CALUDE_m_range_theorem_l25_2587


namespace NUMINAMATH_CALUDE_twentieth_term_of_sequence_l25_2553

-- Define an arithmetic sequence
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

-- State the theorem
theorem twentieth_term_of_sequence : 
  arithmetic_sequence 2 4 20 = 78 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_of_sequence_l25_2553


namespace NUMINAMATH_CALUDE_three_digit_square_mod_1000_l25_2590

theorem three_digit_square_mod_1000 (n : ℕ) :
  (100 ≤ n ∧ n ≤ 999) → (n^2 ≡ n [ZMOD 1000]) ↔ (n = 376 ∨ n = 625) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_square_mod_1000_l25_2590


namespace NUMINAMATH_CALUDE_exam_question_distribution_l25_2540

theorem exam_question_distribution (total_questions : ℕ) 
  (group_a_marks group_b_marks group_c_marks : ℕ) 
  (group_b_questions : ℕ) :
  total_questions = 100 →
  group_a_marks = 1 →
  group_b_marks = 2 →
  group_c_marks = 3 →
  group_b_questions = 23 →
  (∀ a b c : ℕ, a + b + c = total_questions → a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1) →
  (∀ a b c : ℕ, a + b + c = total_questions → 
    a * group_a_marks ≥ (6 * (a * group_a_marks + b * group_b_marks + c * group_c_marks)) / 10) →
  ∃! c : ℕ, c = 1 ∧ ∃ a : ℕ, a + group_b_questions + c = total_questions :=
by sorry

end NUMINAMATH_CALUDE_exam_question_distribution_l25_2540


namespace NUMINAMATH_CALUDE_rhino_fold_swap_impossible_l25_2584

/-- Represents the number of folds on a rhinoceros -/
structure FoldCount where
  vertical : ℕ
  horizontal : ℕ

/-- Represents the state of folds on both sides of a rhinoceros -/
structure RhinoState where
  left : FoldCount
  right : FoldCount

def total_folds (state : RhinoState) : ℕ :=
  state.left.vertical + state.left.horizontal + state.right.vertical + state.right.horizontal

/-- Represents a single scratch action -/
inductive ScratchAction
  | left_vertical
  | left_horizontal
  | right_vertical
  | right_horizontal

/-- Applies a scratch action to a RhinoState -/
def apply_scratch (state : RhinoState) (action : ScratchAction) : RhinoState :=
  match action with
  | ScratchAction.left_vertical => 
      { left := { vertical := state.left.vertical - 2, horizontal := state.left.horizontal },
        right := { vertical := state.right.vertical + 1, horizontal := state.right.horizontal + 1 } }
  | ScratchAction.left_horizontal => 
      { left := { vertical := state.left.vertical, horizontal := state.left.horizontal - 2 },
        right := { vertical := state.right.vertical + 1, horizontal := state.right.horizontal + 1 } }
  | ScratchAction.right_vertical => 
      { left := { vertical := state.left.vertical + 1, horizontal := state.left.horizontal + 1 },
        right := { vertical := state.right.vertical - 2, horizontal := state.right.horizontal } }
  | ScratchAction.right_horizontal => 
      { left := { vertical := state.left.vertical + 1, horizontal := state.left.horizontal + 1 },
        right := { vertical := state.right.vertical, horizontal := state.right.horizontal - 2 } }

theorem rhino_fold_swap_impossible (initial : RhinoState) 
    (h_total : total_folds initial = 17) :
    ¬∃ (actions : List ScratchAction), 
      let final := actions.foldl apply_scratch initial
      total_folds final = 17 ∧ 
      final.left.vertical = initial.left.horizontal ∧
      final.left.horizontal = initial.left.vertical ∧
      final.right.vertical = initial.right.horizontal ∧
      final.right.horizontal = initial.right.vertical :=
  sorry

end NUMINAMATH_CALUDE_rhino_fold_swap_impossible_l25_2584


namespace NUMINAMATH_CALUDE_max_value_problem_l25_2539

theorem max_value_problem (X Y Z : ℕ) (h : 2 * X + 3 * Y + Z = 18) :
  (∀ X' Y' Z' : ℕ, 2 * X' + 3 * Y' + Z' = 18 →
    X' * Y' * Z' + X' * Y' + Y' * Z' + Z' * X' ≤ X * Y * Z + X * Y + Y * Z + Z * X) →
  X * Y * Z + X * Y + Y * Z + Z * X = 24 :=
sorry

end NUMINAMATH_CALUDE_max_value_problem_l25_2539


namespace NUMINAMATH_CALUDE_stationery_cost_theorem_l25_2546

/-- The cost of stationery items -/
structure StationeryCost where
  pencil : ℝ  -- Cost of one pencil
  pen : ℝ     -- Cost of one pen
  eraser : ℝ  -- Cost of one eraser

/-- Given conditions on stationery costs -/
def stationery_conditions (c : StationeryCost) : Prop :=
  4 * c.pencil + 3 * c.pen + c.eraser = 5.40 ∧
  2 * c.pencil + 2 * c.pen + 2 * c.eraser = 4.60

/-- Theorem stating the cost of 1 pencil, 2 pens, and 3 erasers -/
theorem stationery_cost_theorem (c : StationeryCost) 
  (h : stationery_conditions c) : 
  c.pencil + 2 * c.pen + 3 * c.eraser = 4.60 := by
  sorry

end NUMINAMATH_CALUDE_stationery_cost_theorem_l25_2546


namespace NUMINAMATH_CALUDE_tan_half_product_l25_2526

theorem tan_half_product (a b : Real) :
  3 * (Real.sin a + Real.sin b) + 2 * (Real.sin a * Real.sin b + 1) = 0 →
  Real.tan (a / 2) * Real.tan (b / 2) = -4 ∨ Real.tan (a / 2) * Real.tan (b / 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_product_l25_2526


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_23_19_l25_2544

theorem half_abs_diff_squares_23_19 : (1 / 2 : ℝ) * |23^2 - 19^2| = 84 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_23_19_l25_2544


namespace NUMINAMATH_CALUDE_output_for_input_3_l25_2597

-- Define the function for calculating y
def calculate_y (x : ℝ) : ℝ := 3 * x^2 - 5 * x

-- Define the output function
def output (x y : ℝ) : ℝ × ℝ := (x, y)

-- Theorem statement
theorem output_for_input_3 :
  let x : ℝ := 3
  let y : ℝ := calculate_y x
  output x y = (3, 12) := by sorry

end NUMINAMATH_CALUDE_output_for_input_3_l25_2597


namespace NUMINAMATH_CALUDE_probability_is_three_twentyfifths_l25_2513

/-- A rectangle in the 2D plane --/
structure Rectangle where
  x_max : ℝ
  y_max : ℝ

/-- A point in the 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point satisfies the condition x > 5y --/
def satisfies_condition (p : Point) : Prop :=
  p.x > 5 * p.y

/-- Calculate the probability of a randomly chosen point satisfying the condition --/
def probability_satisfies_condition (r : Rectangle) : ℝ :=
  sorry

/-- The main theorem --/
theorem probability_is_three_twentyfifths :
  let r := Rectangle.mk 3000 2500
  probability_satisfies_condition r = 3 / 25 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_three_twentyfifths_l25_2513


namespace NUMINAMATH_CALUDE_circular_permutation_divisibility_l25_2577

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def circular_permutation (n : ℕ) : Set ℕ :=
  {m : ℕ | ∃ k : ℕ, k < 5 ∧ m = (n * 10^k) % 100000 + n / (100000 / 10^k)}

theorem circular_permutation_divisibility (n : ℕ) (h1 : is_five_digit n) (h2 : n % 41 = 0) :
  ∀ m ∈ circular_permutation n, m % 41 = 0 := by
  sorry

end NUMINAMATH_CALUDE_circular_permutation_divisibility_l25_2577


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l25_2547

theorem partial_fraction_decomposition :
  ∀ x : ℚ, x ≠ 12 → x ≠ -4 →
  (6 * x + 3) / (x^2 - 8 * x - 48) = (75 / 16) / (x - 12) + (21 / 16) / (x + 4) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l25_2547


namespace NUMINAMATH_CALUDE_max_area_rectangle_max_area_achievable_l25_2514

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

/-- The theorem stating the maximum area of a rectangle with given conditions -/
theorem max_area_rectangle (l w : ℕ) : 
  (l + w = 60) →  -- Perimeter condition: 2(l + w) = 120
  (isPrime l ∨ isPrime w) →  -- One dimension is prime
  (l * w ≤ 899) :=  -- The area is at most 899
by sorry

/-- The theorem stating that the maximum area of 899 is achievable -/
theorem max_area_achievable : 
  ∃ l w : ℕ, (l + w = 60) ∧ (isPrime l ∨ isPrime w) ∧ (l * w = 899) :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangle_max_area_achievable_l25_2514


namespace NUMINAMATH_CALUDE_specific_ellipse_area_l25_2582

/-- An ellipse with given major axis endpoints and a point on its curve -/
structure Ellipse where
  major_axis_end1 : ℝ × ℝ
  major_axis_end2 : ℝ × ℝ
  point_on_curve : ℝ × ℝ

/-- Calculate the area of the ellipse -/
def ellipse_area (e : Ellipse) : ℝ := sorry

/-- Theorem: The area of the specific ellipse is 50π -/
theorem specific_ellipse_area :
  let e : Ellipse := {
    major_axis_end1 := (2, -3),
    major_axis_end2 := (22, -3),
    point_on_curve := (20, 0)
  }
  ellipse_area e = 50 * Real.pi := by sorry

end NUMINAMATH_CALUDE_specific_ellipse_area_l25_2582


namespace NUMINAMATH_CALUDE_simpson_paradox_possible_l25_2598

/-- Represents the number of patients and successful treatments for a medication in a hospital -/
structure HospitalData where
  patients : ℕ
  successes : ℕ
  hLe : successes ≤ patients

/-- Calculates the effectiveness rate of a medication -/
def effectivenessRate (data : HospitalData) : ℚ :=
  data.successes / data.patients

theorem simpson_paradox_possible 
  (h1A h1B h2A h2B : HospitalData) 
  (h1_effectiveness : effectivenessRate h1A > effectivenessRate h1B)
  (h2_effectiveness : effectivenessRate h2A > effectivenessRate h2B) :
  ∃ (h1A h1B h2A h2B : HospitalData),
    effectivenessRate h1A > effectivenessRate h1B ∧
    effectivenessRate h2A > effectivenessRate h2B ∧
    effectivenessRate (HospitalData.mk (h1A.patients + h2A.patients) (h1A.successes + h2A.successes) sorry) <
    effectivenessRate (HospitalData.mk (h1B.patients + h2B.patients) (h1B.successes + h2B.successes) sorry) :=
  sorry

end NUMINAMATH_CALUDE_simpson_paradox_possible_l25_2598


namespace NUMINAMATH_CALUDE_find_number_l25_2589

theorem find_number : ∃! x : ℚ, (172 / 4 - 28) * x + 7 = 172 := by sorry

end NUMINAMATH_CALUDE_find_number_l25_2589


namespace NUMINAMATH_CALUDE_subset_M_l25_2501

def M : Set ℝ := {x : ℝ | x > -1}

theorem subset_M : {0} ⊆ M := by sorry

end NUMINAMATH_CALUDE_subset_M_l25_2501


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_less_than_negative_thirty_l25_2525

theorem largest_multiple_of_seven_less_than_negative_thirty :
  ∃ (n : ℤ), n * 7 = -35 ∧ 
  n * 7 < -30 ∧ 
  ∀ (m : ℤ), m * 7 < -30 → m * 7 ≤ -35 := by
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_less_than_negative_thirty_l25_2525


namespace NUMINAMATH_CALUDE_dans_remaining_money_dans_remaining_money_proof_l25_2579

/-- Calculates the remaining money after purchases and tax --/
theorem dans_remaining_money (initial_amount : ℚ) 
  (candy_price : ℚ) (candy_count : ℕ) 
  (gum_price : ℚ) (soda_price : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_before_tax := candy_price * candy_count + gum_price + soda_price
  let total_tax := total_before_tax * tax_rate
  let total_cost := total_before_tax + total_tax
  initial_amount - total_cost

/-- Proves that Dan's remaining money is $40.98 --/
theorem dans_remaining_money_proof :
  dans_remaining_money 50 1.75 3 0.85 2.25 0.08 = 40.98 := by
  sorry

end NUMINAMATH_CALUDE_dans_remaining_money_dans_remaining_money_proof_l25_2579


namespace NUMINAMATH_CALUDE_bailey_towel_discount_percentage_l25_2571

/-- Calculates the discount percentage for Bailey's towel purchase. -/
theorem bailey_towel_discount_percentage : 
  let guest_sets : ℕ := 2
  let master_sets : ℕ := 4
  let guest_price : ℚ := 40
  let master_price : ℚ := 50
  let total_spent : ℚ := 224
  let original_total : ℚ := guest_sets * guest_price + master_sets * master_price
  let discount_amount : ℚ := original_total - total_spent
  let discount_percentage : ℚ := (discount_amount / original_total) * 100
  discount_percentage = 20 := by
  sorry

end NUMINAMATH_CALUDE_bailey_towel_discount_percentage_l25_2571


namespace NUMINAMATH_CALUDE_factor_expression_l25_2594

theorem factor_expression (y : ℝ) : 5 * y * (y - 2) + 9 * (y - 2) = (y - 2) * (5 * y + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l25_2594


namespace NUMINAMATH_CALUDE_total_oranges_in_box_l25_2595

def initial_oranges : ℝ := 55.0
def added_oranges : ℝ := 35.0

theorem total_oranges_in_box : initial_oranges + added_oranges = 90.0 := by
  sorry

end NUMINAMATH_CALUDE_total_oranges_in_box_l25_2595


namespace NUMINAMATH_CALUDE_systematic_sampling_missiles_l25_2543

/-- Represents a systematic sampling sequence -/
def SystematicSample (total : ℕ) (sampleSize : ℕ) (start : ℕ) : List ℕ :=
  List.range sampleSize |>.map (fun i => start + i * (total / sampleSize))

/-- The problem statement -/
theorem systematic_sampling_missiles :
  let total := 50
  let sampleSize := 5
  let start := 3
  SystematicSample total sampleSize start = [3, 13, 23, 33, 43] := by
  sorry

#eval SystematicSample 50 5 3

end NUMINAMATH_CALUDE_systematic_sampling_missiles_l25_2543


namespace NUMINAMATH_CALUDE_two_distinct_roots_l25_2578

/-- The cubic function f(x) = x^3 - 3x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + a

/-- Theorem stating that f(x) has exactly two distinct roots iff a = 2/√3 -/
theorem two_distinct_roots (a : ℝ) (h : a > 0) :
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ ∀ z : ℝ, f a z = 0 → z = x ∨ z = y) ↔
  a = 2 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_two_distinct_roots_l25_2578


namespace NUMINAMATH_CALUDE_outfits_count_l25_2529

/-- The number of different outfits that can be made -/
def num_outfits (num_shirts : ℕ) (num_ties : ℕ) (num_pants : ℕ) (num_belts : ℕ) : ℕ :=
  num_shirts * num_pants * (num_ties + 1) * (num_belts + 1)

/-- Theorem stating that the number of outfits is 360 given the specific conditions -/
theorem outfits_count :
  num_outfits 5 5 4 2 = 360 :=
by sorry

end NUMINAMATH_CALUDE_outfits_count_l25_2529


namespace NUMINAMATH_CALUDE_jason_has_18_books_l25_2565

/-- The number of books Mary has -/
def mary_books : ℕ := 42

/-- The total number of books Jason and Mary have together -/
def total_books : ℕ := 60

/-- The number of books Jason has -/
def jason_books : ℕ := total_books - mary_books

theorem jason_has_18_books : jason_books = 18 := by
  sorry

end NUMINAMATH_CALUDE_jason_has_18_books_l25_2565


namespace NUMINAMATH_CALUDE_remainder_divisibility_l25_2512

theorem remainder_divisibility (n : ℤ) : 
  (2 * n) % 7 = 4 → n % 7 = 2 := by
sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l25_2512


namespace NUMINAMATH_CALUDE_jack_classic_authors_l25_2573

/-- The number of books each classic author has -/
def books_per_author : ℕ := 33

/-- The total number of classic books in Jack's collection -/
def total_classic_books : ℕ := 198

/-- The number of classic authors in Jack's collection -/
def number_of_authors : ℕ := total_classic_books / books_per_author

theorem jack_classic_authors :
  number_of_authors = 6 :=
sorry

end NUMINAMATH_CALUDE_jack_classic_authors_l25_2573


namespace NUMINAMATH_CALUDE_fraction_increase_l25_2585

theorem fraction_increase (m n a : ℝ) (h1 : m > n) (h2 : n > 0) (h3 : a > 0) :
  (n + a) / (m + a) > n / m := by
  sorry

end NUMINAMATH_CALUDE_fraction_increase_l25_2585


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l25_2591

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l25_2591


namespace NUMINAMATH_CALUDE_range_of_product_l25_2530

theorem range_of_product (a b : ℝ) (h1 : |a| ≤ 1) (h2 : |a + b| ≤ 1) :
  ∃ (min max : ℝ), min = -2 ∧ max = 9/4 ∧
  ∀ x, x = (a + 1) * (b + 1) → min ≤ x ∧ x ≤ max :=
sorry

end NUMINAMATH_CALUDE_range_of_product_l25_2530


namespace NUMINAMATH_CALUDE_karen_wrong_answers_l25_2580

/-- Represents the number of wrong answers for each person -/
structure TestResults where
  karen : ℕ
  leo : ℕ
  morgan : ℕ
  nora : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (r : TestResults) : Prop :=
  r.karen + r.leo = r.morgan + r.nora ∧
  r.karen + r.nora = r.leo + r.morgan + 3 ∧
  r.morgan = 6

theorem karen_wrong_answers (r : TestResults) (h : satisfiesConditions r) : r.karen = 6 := by
  sorry

#check karen_wrong_answers

end NUMINAMATH_CALUDE_karen_wrong_answers_l25_2580


namespace NUMINAMATH_CALUDE_total_apple_weight_marta_apple_purchase_l25_2531

/-- The weight of one apple in ounces -/
def apple_weight : ℕ := 4

/-- The weight of one orange in ounces -/
def orange_weight : ℕ := 3

/-- The maximum weight a bag can hold in ounces -/
def bag_capacity : ℕ := 49

/-- The number of bags Marta wants to buy -/
def num_bags : ℕ := 3

/-- The number of apples in one bag -/
def apples_per_bag : ℕ := 7

theorem total_apple_weight :
  apple_weight * (apples_per_bag * num_bags) = 84 :=
by sorry

/-- The main theorem stating the total weight of apples Marta should buy -/
theorem marta_apple_purchase :
  ∃ (x : ℕ), x = apple_weight * (apples_per_bag * num_bags) ∧
  x ≤ bag_capacity * num_bags ∧
  x = 84 :=
by sorry

end NUMINAMATH_CALUDE_total_apple_weight_marta_apple_purchase_l25_2531


namespace NUMINAMATH_CALUDE_marble_ratio_l25_2581

theorem marble_ratio (total : ℕ) (red : ℕ) (yellow : ℕ) 
  (h_total : total = 85)
  (h_red : red = 14)
  (h_yellow : yellow = 29) :
  (total - red - yellow) / red = 3 := by
sorry

end NUMINAMATH_CALUDE_marble_ratio_l25_2581


namespace NUMINAMATH_CALUDE_remainder_theorem_l25_2534

theorem remainder_theorem (n : ℤ) (h : n % 9 = 4) : (5 * n - 12) % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l25_2534


namespace NUMINAMATH_CALUDE_smallest_number_with_properties_l25_2576

def is_sum_of_five_fourth_powers (n : ℕ) : Prop :=
  ∃ (a b c d e : ℕ), a < b ∧ b < c ∧ c < d ∧ d < e ∧ 
    n = a^4 + b^4 + c^4 + d^4 + e^4

def is_sum_of_six_consecutive_integers (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = k + (k+1) + (k+2) + (k+3) + (k+4) + (k+5)

theorem smallest_number_with_properties : 
  ∀ n : ℕ, n < 2019 → 
    ¬(is_sum_of_five_fourth_powers n ∧ is_sum_of_six_consecutive_integers n) :=
by
  sorry

#check smallest_number_with_properties

end NUMINAMATH_CALUDE_smallest_number_with_properties_l25_2576


namespace NUMINAMATH_CALUDE_circle_radius_l25_2549

theorem circle_radius (P Q : ℝ) (h : P / Q = 40 / Real.pi) :
  ∃ (r : ℝ), r > 0 ∧ P = Real.pi * r^2 ∧ Q = 2 * Real.pi * r ∧ r = 80 / Real.pi :=
sorry

end NUMINAMATH_CALUDE_circle_radius_l25_2549


namespace NUMINAMATH_CALUDE_find_b_l25_2502

def gcd_notation (x y : ℕ) : ℕ := x * y

theorem find_b : ∃ b : ℕ, gcd_notation (gcd_notation (16 * b) (18 * 24)) 2 = 2 ∧ b = 1 := by sorry

end NUMINAMATH_CALUDE_find_b_l25_2502


namespace NUMINAMATH_CALUDE_discount_difference_equals_582_l25_2532

def initial_bill : ℝ := 12000

def single_discount_rate : ℝ := 0.45
def successive_discount_rates : List ℝ := [0.30, 0.10, 0.05]

def apply_single_discount (bill : ℝ) (rate : ℝ) : ℝ :=
  bill * (1 - rate)

def apply_successive_discounts (bill : ℝ) (rates : List ℝ) : ℝ :=
  rates.foldl (fun acc rate => acc * (1 - rate)) bill

theorem discount_difference_equals_582 :
  apply_successive_discounts initial_bill successive_discount_rates -
  apply_single_discount initial_bill single_discount_rate = 582 := by
    sorry

end NUMINAMATH_CALUDE_discount_difference_equals_582_l25_2532


namespace NUMINAMATH_CALUDE_xy_not_6_sufficient_not_necessary_l25_2516

theorem xy_not_6_sufficient_not_necessary :
  (∃ x y : ℝ, (x ≠ 2 ∨ y ≠ 3) ∧ x * y = 6) ∧
  (∀ x y : ℝ, x * y ≠ 6 → (x ≠ 2 ∨ y ≠ 3)) :=
by sorry

end NUMINAMATH_CALUDE_xy_not_6_sufficient_not_necessary_l25_2516


namespace NUMINAMATH_CALUDE_exchange_problem_l25_2542

def exchange_rate : ℚ := 11 / 8
def spent_amount : ℕ := 70

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.repr.toList.map (λ c => c.toNat - '0'.toNat)
  digits.sum

theorem exchange_problem (d : ℕ) :
  (exchange_rate * d : ℚ) - spent_amount = d →
  sum_of_digits d = 10 := by
  sorry

end NUMINAMATH_CALUDE_exchange_problem_l25_2542


namespace NUMINAMATH_CALUDE_inequality_solution_l25_2574

def solution_set : Set ℝ := {x : ℝ | x^2 - 3*x - 4 < 0}

theorem inequality_solution : solution_set = Set.Ioo (-1) 4 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l25_2574


namespace NUMINAMATH_CALUDE_circle_tangent_to_y_axis_l25_2524

def circle_equation (x y : ℝ) := (x - 2)^2 + (y - 1)^2 = 4

def is_tangent_to_y_axis (equation : ℝ → ℝ → Prop) : Prop :=
  ∃ y : ℝ, equation 0 y ∧ ∀ x y : ℝ, x ≠ 0 → equation x y → (x - 0)^2 + (y - y)^2 > 0

theorem circle_tangent_to_y_axis :
  is_tangent_to_y_axis circle_equation ∧
  ∀ x y : ℝ, circle_equation x y → (x - 2)^2 + (y - 1)^2 = 4 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_y_axis_l25_2524


namespace NUMINAMATH_CALUDE_hall_length_l25_2551

/-- The length of a rectangular hall given its width, height, and total area to be covered. -/
theorem hall_length (width height total_area : ℝ) (hw : width = 15) (hh : height = 5) 
  (ha : total_area = 950) : 
  ∃ length : ℝ, length = 32 ∧ total_area = length * width + 2 * (height * length + height * width) :=
by sorry

end NUMINAMATH_CALUDE_hall_length_l25_2551


namespace NUMINAMATH_CALUDE_cars_meeting_time_l25_2517

theorem cars_meeting_time (distance : ℝ) (speed1 speed2 : ℝ) (h1 : distance = 60) 
  (h2 : speed1 = 13) (h3 : speed2 = 17) : 
  distance / (speed1 + speed2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cars_meeting_time_l25_2517


namespace NUMINAMATH_CALUDE_skylar_donation_l25_2564

/-- Calculates the total donation amount given starting age, current age, and annual donation. -/
def total_donation (start_age : ℕ) (current_age : ℕ) (annual_donation : ℕ) : ℕ :=
  (current_age - start_age) * annual_donation

/-- Proves that Skylar's total donation is $432,000 given the specified conditions. -/
theorem skylar_donation :
  let start_age : ℕ := 17
  let current_age : ℕ := 71
  let annual_donation : ℕ := 8000
  total_donation start_age current_age annual_donation = 432000 := by
  sorry

end NUMINAMATH_CALUDE_skylar_donation_l25_2564


namespace NUMINAMATH_CALUDE_abs_a_gt_abs_b_l25_2520

theorem abs_a_gt_abs_b (a b : ℝ) (ha : a > 0) (hb : b < 0) (hab : a + b > 0) : |a| > |b| := by
  sorry

end NUMINAMATH_CALUDE_abs_a_gt_abs_b_l25_2520


namespace NUMINAMATH_CALUDE_base4_multiplication_division_l25_2562

-- Define a function to convert from base 4 to base 10
def base4ToBase10 (n : ℕ) : ℕ := sorry

-- Define a function to convert from base 10 to base 4
def base10ToBase4 (n : ℕ) : ℕ := sorry

-- Define the theorem
theorem base4_multiplication_division :
  base10ToBase4 ((base4ToBase10 131 * base4ToBase10 21) / base4ToBase10 3) = 1113 := by sorry

end NUMINAMATH_CALUDE_base4_multiplication_division_l25_2562


namespace NUMINAMATH_CALUDE_candy_distribution_l25_2559

theorem candy_distribution (total : ℕ) (a b c d : ℕ) : 
  total = 2013 →
  a + b + c + d = total →
  a = 2 * b + 10 →
  a = 3 * c + 18 →
  a = 5 * d - 55 →
  a = 990 :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l25_2559


namespace NUMINAMATH_CALUDE_last_digit_of_77_in_binary_l25_2528

theorem last_digit_of_77_in_binary (n : Nat) : n = 77 → n % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_77_in_binary_l25_2528


namespace NUMINAMATH_CALUDE_star_count_l25_2583

theorem star_count (east : ℕ) (west : ℕ) : 
  east = 120 → 
  west = 6 * east → 
  east + west = 840 := by
sorry

end NUMINAMATH_CALUDE_star_count_l25_2583


namespace NUMINAMATH_CALUDE_circle_area_tripled_l25_2527

theorem circle_area_tripled (r n : ℝ) : 
  (π * (r + n)^2 = 3 * π * r^2) → (r = n * (Real.sqrt 3 - 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_area_tripled_l25_2527


namespace NUMINAMATH_CALUDE_product_mod_six_l25_2572

theorem product_mod_six : 2017 * 2018 * 2019 * 2020 ≡ 0 [ZMOD 6] := by
  sorry

end NUMINAMATH_CALUDE_product_mod_six_l25_2572


namespace NUMINAMATH_CALUDE_ellipse_circle_centers_distance_l25_2586

/-- Given an ellipse with center O and semi-axes a and b, and a circle with radius r 
    and center C on the major semi-axis of the ellipse that touches the ellipse at two points, 
    prove that the square of the distance between the centers of the ellipse and the circle 
    is equal to ((a^2 - b^2) * (b^2 - r^2)) / b^2. -/
theorem ellipse_circle_centers_distance 
  (O : ℝ × ℝ) (C : ℝ × ℝ) (a b r : ℝ) : 
  (a > 0) → (b > 0) → (r > 0) → (a ≥ b) →
  (∃ (P Q : ℝ × ℝ), 
    (P.1 - O.1)^2 / a^2 + (P.2 - O.2)^2 / b^2 = 1 ∧
    (Q.1 - O.1)^2 / a^2 + (Q.2 - O.2)^2 / b^2 = 1 ∧
    (P.1 - C.1)^2 + (P.2 - C.2)^2 = r^2 ∧
    (Q.1 - C.1)^2 + (Q.2 - C.2)^2 = r^2 ∧
    C.2 = O.2) →
  (C.1 - O.1)^2 = ((a^2 - b^2) * (b^2 - r^2)) / b^2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_circle_centers_distance_l25_2586


namespace NUMINAMATH_CALUDE_red_balls_count_l25_2537

/-- Given a jar with white and red balls where the ratio of white to red balls is 4:3 
    and there are 12 white balls, prove that there are 9 red balls. -/
theorem red_balls_count (white_balls : ℕ) (red_balls : ℕ) : 
  (white_balls : ℚ) / red_balls = 4 / 3 → white_balls = 12 → red_balls = 9 := by
sorry

end NUMINAMATH_CALUDE_red_balls_count_l25_2537


namespace NUMINAMATH_CALUDE_apple_distribution_l25_2563

theorem apple_distribution (total_apples : ℕ) (alice_min : ℕ) (becky_min : ℕ) (chris_min : ℕ)
  (h1 : total_apples = 30)
  (h2 : alice_min = 3)
  (h3 : becky_min = 2)
  (h4 : chris_min = 2) :
  (Nat.choose (total_apples - alice_min - becky_min - chris_min + 2) 2) = 300 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l25_2563


namespace NUMINAMATH_CALUDE_unknown_number_proof_l25_2533

theorem unknown_number_proof (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 192)
  (h_hcf : Nat.gcd a b = 16)
  (h_a : a = 64) :
  b = 48 := by
sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l25_2533


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_seven_l25_2511

theorem greatest_two_digit_multiple_of_seven : ∃ n : ℕ, n = 98 ∧ 
  (∀ m : ℕ, m < 100 ∧ 7 ∣ m → m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_seven_l25_2511


namespace NUMINAMATH_CALUDE_profit_distribution_l25_2599

theorem profit_distribution (P Q R : ℝ) (total_profit : ℝ) (R_profit : ℝ) (k : ℝ) :
  4 * P = 6 * Q ∧ 
  4 * P = k * R ∧
  total_profit = 4650 ∧
  R_profit = 900 →
  k = 2.4 := by
sorry

end NUMINAMATH_CALUDE_profit_distribution_l25_2599


namespace NUMINAMATH_CALUDE_zero_in_interval_l25_2523

def f (x : ℝ) := x^3 + 2*x - 2

theorem zero_in_interval :
  ∃ c ∈ Set.Icc 0 1, f c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_l25_2523


namespace NUMINAMATH_CALUDE_trig_identity_l25_2552

theorem trig_identity (x y : ℝ) : 
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.sin (x + y) = Real.cos x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l25_2552


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l25_2506

def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 2}

theorem intersection_of_M_and_N : M ∩ N = {(2, 0)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l25_2506


namespace NUMINAMATH_CALUDE_complex_second_quadrant_second_quadrant_implies_neg_real_pos_imag_l25_2504

/-- A complex number is in the second quadrant if its real part is negative and its imaginary part is positive -/
theorem complex_second_quadrant (z : ℂ) :
  (z.re < 0 ∧ z.im > 0) ↔ (z.arg > Real.pi / 2 ∧ z.arg < Real.pi) :=
by sorry

/-- If a complex number is in the second quadrant, then its real part is negative and its imaginary part is positive -/
theorem second_quadrant_implies_neg_real_pos_imag (z : ℂ) 
  (h : z.arg > Real.pi / 2 ∧ z.arg < Real.pi) : 
  z.re < 0 ∧ z.im > 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_second_quadrant_second_quadrant_implies_neg_real_pos_imag_l25_2504


namespace NUMINAMATH_CALUDE_sum_value_l25_2588

def S : ℚ := 3003 + (1/3) * (3002 + (1/6) * (3001 + (1/9) * (3000 + (1/(3*1000)) * 3)))

theorem sum_value : S = 3002.5 := by sorry

end NUMINAMATH_CALUDE_sum_value_l25_2588


namespace NUMINAMATH_CALUDE_triangle_properties_l25_2538

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem to be proved -/
theorem triangle_properties (t : Triangle) 
  (h1 : (t.a * Real.cos t.B + t.b * Real.cos t.A) / t.c = 2 * Real.cos t.C)
  (h2 : (1/2) * t.a * t.b * Real.sin t.C = 2 * Real.sqrt 3)
  (h3 : t.a + t.b = 6) :
  t.C = π/3 ∧ t.c = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l25_2538


namespace NUMINAMATH_CALUDE_second_term_is_seven_l25_2596

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  first_term : ℝ
  common_difference : ℝ
  num_terms : ℕ
  sum_first_eight_eq_last_four : ℝ → Prop

/-- The theorem statement -/
theorem second_term_is_seven
  (seq : ArithmeticSequence)
  (h1 : seq.num_terms = 12)
  (h2 : seq.common_difference = 2)
  (h3 : seq.sum_first_eight_eq_last_four seq.first_term) :
  seq.first_term + seq.common_difference = 7 := by
  sorry

end NUMINAMATH_CALUDE_second_term_is_seven_l25_2596


namespace NUMINAMATH_CALUDE_alcohol_dilution_l25_2555

theorem alcohol_dilution (initial_volume : ℝ) (initial_alcohol_percentage : ℝ) (added_water : ℝ) :
  initial_volume = 15 →
  initial_alcohol_percentage = 20 →
  added_water = 5 →
  let initial_alcohol := initial_volume * (initial_alcohol_percentage / 100)
  let new_volume := initial_volume + added_water
  let new_alcohol_percentage := (initial_alcohol / new_volume) * 100
  new_alcohol_percentage = 15 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_dilution_l25_2555


namespace NUMINAMATH_CALUDE_supplement_of_complementary_l25_2557

/-- Two angles are complementary if their sum is 90 degrees -/
def complementary (α β : ℝ) : Prop := α + β = 90

/-- The supplement of an angle is 180 degrees minus the angle -/
def supplement (θ : ℝ) : ℝ := 180 - θ

/-- 
If two angles α and β are complementary, 
then the supplement of α is 90 degrees greater than β 
-/
theorem supplement_of_complementary (α β : ℝ) 
  (h : complementary α β) : 
  supplement α = β + 90 := by sorry

end NUMINAMATH_CALUDE_supplement_of_complementary_l25_2557


namespace NUMINAMATH_CALUDE_triangle_altitude_and_median_equations_l25_2550

/-- Triangle ABC with vertices A(4,0), B(6,7), and C(0,3) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The given triangle -/
def triangle : Triangle := { A := (4, 0), B := (6, 7), C := (0, 3) }

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The equation of the altitude from B to AC -/
def altitudeEquation : LineEquation := { a := 3, b := 2, c := -12 }

/-- The equation of the median from B to AC -/
def medianEquation : LineEquation := { a := 5, b := 1, c := -20 }

theorem triangle_altitude_and_median_equations :
  let t := triangle
  let alt := altitudeEquation
  let med := medianEquation
  (∀ x y : ℝ, alt.a * x + alt.b * y + alt.c = 0 ↔ 
    (x - t.B.1) * (t.A.1 - t.C.1) + (y - t.B.2) * (t.A.2 - t.C.2) = 0) ∧
  (∀ x y : ℝ, med.a * x + med.b * y + med.c = 0 ↔ 
    2 * (x - t.B.1) = t.A.1 - t.C.1 ∧ 2 * (y - t.B.2) = t.A.2 - t.C.2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_and_median_equations_l25_2550


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_measure_l25_2569

-- Define a regular octagon
structure RegularOctagon where
  -- No specific fields needed for this problem

-- Define the measure of an interior angle of a regular octagon
def interior_angle_measure (o : RegularOctagon) : ℝ := 135

-- Theorem statement
theorem regular_octagon_interior_angle_measure (o : RegularOctagon) :
  interior_angle_measure o = 135 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_measure_l25_2569


namespace NUMINAMATH_CALUDE_no_solution_to_equation_l25_2554

theorem no_solution_to_equation :
  ¬ ∃ (x : ℝ), x ≠ 2 ∧ (1 / (x - 2) = (1 - x) / (2 - x) - 3) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_equation_l25_2554


namespace NUMINAMATH_CALUDE_min_area_hyperbola_triangle_l25_2593

/-- A point on the hyperbola xy = 1 -/
structure HyperbolaPoint where
  x : ℝ
  y : ℝ
  on_hyperbola : x * y = 1

/-- An isosceles right triangle on the hyperbola xy = 1 -/
structure HyperbolaTriangle where
  A : HyperbolaPoint
  B : HyperbolaPoint
  C : HyperbolaPoint
  is_right_angle : (B.x - A.x) * (C.x - A.x) + (B.y - A.y) * (C.y - A.y) = 0
  is_isosceles : (B.x - A.x)^2 + (B.y - A.y)^2 = (C.x - A.x)^2 + (C.y - A.y)^2

/-- The area of a triangle given by three points -/
def triangleArea (A B C : HyperbolaPoint) : ℝ :=
  0.5 * abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

/-- The theorem stating the minimum area of an isosceles right triangle on the hyperbola xy = 1 -/
theorem min_area_hyperbola_triangle :
  ∀ T : HyperbolaTriangle, triangleArea T.A T.B T.C ≥ 3 * Real.sqrt 3 := by
  sorry

#check min_area_hyperbola_triangle

end NUMINAMATH_CALUDE_min_area_hyperbola_triangle_l25_2593


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l25_2570

theorem system_of_equations_solution :
  ∃! (x y z : ℚ),
    x + 2 * y - z = 20 ∧
    y = 5 ∧
    3 * x + 4 * z = 40 ∧
    x = 80 / 7 ∧
    z = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l25_2570


namespace NUMINAMATH_CALUDE_cube_root_unity_product_l25_2508

/-- A complex cube root of unity -/
def ω : ℂ :=
  sorry

/-- The property of ω being a complex cube root of unity -/
axiom ω_cube_root : ω^3 = 1

/-- The sum of powers of ω equals zero -/
axiom ω_sum_zero : 1 + ω + ω^2 = 0

/-- The main theorem -/
theorem cube_root_unity_product (a b c : ℂ) :
  (a + b*ω + c*ω^2) * (a + b*ω^2 + c*ω) = a^2 + b^2 + c^2 - a*b - a*c - b*c :=
sorry

end NUMINAMATH_CALUDE_cube_root_unity_product_l25_2508


namespace NUMINAMATH_CALUDE_geese_percentage_among_non_swans_l25_2519

theorem geese_percentage_among_non_swans 
  (total_percentage : ℝ) 
  (geese_percentage : ℝ) 
  (swan_percentage : ℝ) 
  (h1 : total_percentage = 100) 
  (h2 : geese_percentage = 20) 
  (h3 : swan_percentage = 25) : 
  (geese_percentage / (total_percentage - swan_percentage)) * 100 = 26.67 := by
sorry

end NUMINAMATH_CALUDE_geese_percentage_among_non_swans_l25_2519


namespace NUMINAMATH_CALUDE_min_shift_for_symmetry_l25_2505

theorem min_shift_for_symmetry (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = Real.sqrt 3 * Real.sin (2 * x) + Real.cos (2 * x)) →
  φ > 0 →
  (∀ x, f (x - φ) = f (π / 6 - x)) →
  φ ≥ 5 * π / 12 :=
by sorry

end NUMINAMATH_CALUDE_min_shift_for_symmetry_l25_2505


namespace NUMINAMATH_CALUDE_red_pens_count_l25_2566

/-- The number of red pens in Maria's desk drawer. -/
def red_pens : ℕ := sorry

/-- The number of black pens in Maria's desk drawer. -/
def black_pens : ℕ := red_pens + 10

/-- The number of blue pens in Maria's desk drawer. -/
def blue_pens : ℕ := red_pens + 7

/-- The total number of pens in Maria's desk drawer. -/
def total_pens : ℕ := 41

theorem red_pens_count : red_pens = 8 := by
  sorry

end NUMINAMATH_CALUDE_red_pens_count_l25_2566


namespace NUMINAMATH_CALUDE_migraine_expectation_l25_2510

/-- The fraction of Canadians suffering from migraines -/
def migraine_fraction : ℚ := 2 / 7

/-- The total number of Canadians in the sample -/
def sample_size : ℕ := 350

/-- The expected number of Canadians in the sample suffering from migraines -/
def expected_migraines : ℕ := 100

theorem migraine_expectation :
  (migraine_fraction * sample_size : ℚ) = expected_migraines := by sorry

end NUMINAMATH_CALUDE_migraine_expectation_l25_2510


namespace NUMINAMATH_CALUDE_lines_parallel_iff_m_eq_3_or_neg_1_l25_2536

-- Define the lines
def line1 (m : ℝ) (x y : ℝ) : Prop := x + m * y + 7 = 0
def line2 (m : ℝ) (x y : ℝ) : Prop := (m - 2) * x + 3 * y + 2 * m = 0

-- Define parallel lines
def parallel (m : ℝ) : Prop := ∀ x y, line1 m x y ↔ ∃ k, line2 m (x + k) (y + k)

-- Theorem statement
theorem lines_parallel_iff_m_eq_3_or_neg_1 :
  ∀ m : ℝ, parallel m ↔ m = 3 ∨ m = -1 := by sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_m_eq_3_or_neg_1_l25_2536


namespace NUMINAMATH_CALUDE_salary_change_l25_2509

theorem salary_change (original : ℝ) (h : original > 0) :
  let decreased := original * 0.5
  let increased := decreased * 1.5
  (original - increased) / original = 0.25 := by
sorry

end NUMINAMATH_CALUDE_salary_change_l25_2509


namespace NUMINAMATH_CALUDE_dans_car_fuel_efficiency_l25_2560

/-- Represents the fuel efficiency of Dan's car in miles per gallon. -/
def fuel_efficiency : ℝ := 32

/-- The cost of gas in dollars per gallon. -/
def gas_cost : ℝ := 4

/-- The distance Dan's car can travel on $42 of gas, in miles. -/
def distance : ℝ := 336

/-- The amount spent on gas, in dollars. -/
def gas_spent : ℝ := 42

/-- Theorem stating that Dan's car's fuel efficiency is 32 miles per gallon. -/
theorem dans_car_fuel_efficiency :
  fuel_efficiency = distance / (gas_spent / gas_cost) := by
  sorry

end NUMINAMATH_CALUDE_dans_car_fuel_efficiency_l25_2560
