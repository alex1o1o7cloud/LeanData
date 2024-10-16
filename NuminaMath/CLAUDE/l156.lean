import Mathlib

namespace NUMINAMATH_CALUDE_fraction_simplification_l156_15637

theorem fraction_simplification (x : ℝ) (h : x ≠ 1) :
  (2*x - 1) / (x - 1) + x / (1 - x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l156_15637


namespace NUMINAMATH_CALUDE_triangle_inequality_l156_15686

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (htri : a + b > c ∧ b + c > a ∧ c + a > b) :
  (c + a - b)^4 / (a * (a + b - c)) +
  (a + b - c)^4 / (b * (b + c - a)) +
  (b + c - a)^4 / (c * (c + a - b)) ≥
  a^2 + b^2 + c^2 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l156_15686


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l156_15630

theorem product_of_three_numbers (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_ab : a * b = 45 * Real.rpow 3 (1/3))
  (h_ac : a * c = 75 * Real.rpow 3 (1/3))
  (h_bc : b * c = 27 * Real.rpow 3 (1/3)) :
  a * b * c = 135 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l156_15630


namespace NUMINAMATH_CALUDE_pr_qs_ratio_l156_15689

/-- Given four points P, Q, R, and S on a number line, prove that the ratio of lengths PR:QS is 7:12 -/
theorem pr_qs_ratio (P Q R S : ℝ) (hP : P = 3) (hQ : Q = 5) (hR : R = 10) (hS : S = 17) :
  (R - P) / (S - Q) = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_pr_qs_ratio_l156_15689


namespace NUMINAMATH_CALUDE_hamburgers_left_over_l156_15699

theorem hamburgers_left_over (made served : ℕ) (h1 : made = 9) (h2 : served = 3) :
  made - served = 6 := by
  sorry

end NUMINAMATH_CALUDE_hamburgers_left_over_l156_15699


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l156_15694

/-- Given an arithmetic sequence {a_n} where a_5 + a_7 = 2, 
    prove that a_4 + 2a_6 + a_8 = 4 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) : 
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) → -- arithmetic sequence condition
  (a 5 + a 7 = 2) →                                -- given condition
  (a 4 + 2 * a 6 + a 8 = 4) :=                     -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l156_15694


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l156_15674

/-- Expresses the sum of repeating decimals 0.3̅, 0.07̅, and 0.008̅ as a common fraction -/
theorem repeating_decimal_sum : 
  (1 : ℚ) / 3 + 7 / 99 + 8 / 999 = 418 / 999 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l156_15674


namespace NUMINAMATH_CALUDE_cathy_remaining_money_l156_15644

/-- Calculates the remaining money in Cathy's wallet after all expenditures --/
def remaining_money (initial_amount dad_sent mom_sent_multiplier book_cost cab_ride_percent dinner_percent : ℝ) : ℝ :=
  let total_from_parents := dad_sent + mom_sent_multiplier * dad_sent
  let total_initial := total_from_parents + initial_amount
  let food_budget := 0.4 * total_initial
  let after_book := total_initial - book_cost
  let cab_cost := cab_ride_percent * after_book
  let after_cab := after_book - cab_cost
  let dinner_cost := dinner_percent * food_budget
  after_cab - dinner_cost

/-- Theorem stating that Cathy's remaining money is $52.44 --/
theorem cathy_remaining_money :
  remaining_money 12 25 2 15 0.03 0.5 = 52.44 := by
  sorry

end NUMINAMATH_CALUDE_cathy_remaining_money_l156_15644


namespace NUMINAMATH_CALUDE_min_value_xy_l156_15643

theorem min_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : (2 / x) + (8 / y) = 1) :
  xy ≥ 64 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ (2 / x₀) + (8 / y₀) = 1 ∧ x₀ * y₀ = 64 :=
by sorry

end NUMINAMATH_CALUDE_min_value_xy_l156_15643


namespace NUMINAMATH_CALUDE_p_and_q_true_p_and_not_q_false_l156_15697

-- Define proposition p
def p : Prop := ∀ m : ℝ, ∃ x : ℝ, x^2 - m*x - 1 = 0

-- Define proposition q
def q : Prop := ∃ x₀ : ℕ, x₀^2 - 2*x₀ - 1 ≤ 0

-- Theorem stating that p and q are true
theorem p_and_q_true : p ∧ q := by sorry

-- Theorem stating that p ∧ (¬q) is false
theorem p_and_not_q_false : ¬(p ∧ ¬q) := by sorry

end NUMINAMATH_CALUDE_p_and_q_true_p_and_not_q_false_l156_15697


namespace NUMINAMATH_CALUDE_f_abs_g_is_odd_l156_15655

-- Define the real-valued functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- State the theorem
theorem f_abs_g_is_odd 
  (hf : is_odd f) 
  (hg : is_even g) : 
  is_odd (λ x ↦ f x * |g x|) := by
  sorry

end NUMINAMATH_CALUDE_f_abs_g_is_odd_l156_15655


namespace NUMINAMATH_CALUDE_locus_and_fixed_point_l156_15634

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define point N
def point_N : ℝ × ℝ := (-2, 0)

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Define points A₁ and A₂
def point_A1 : ℝ × ℝ := (-1, 0)
def point_A2 : ℝ × ℝ := (1, 0)

-- Define the line x = 2
def line_x_2 (x y : ℝ) : Prop := x = 2

-- Theorem statement
theorem locus_and_fixed_point :
  ∀ (P : ℝ × ℝ) (Q : ℝ × ℝ) (E : ℝ × ℝ) (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ),
  circle_M P.1 P.2 →
  line_x_2 E.1 E.2 ∧ line_x_2 F.1 F.2 →
  E.2 = -F.2 →
  curve_C Q.1 Q.2 →
  curve_C A.1 A.2 ∧ curve_C B.1 B.2 →
  (∃ (m : ℝ), A.2 - point_A1.2 = m * (A.1 - point_A1.1) ∧
               E.2 - point_A1.2 = m * (E.1 - point_A1.1)) →
  (∃ (n : ℝ), B.2 - point_A2.2 = n * (B.1 - point_A2.1) ∧
               F.2 - point_A2.2 = n * (F.1 - point_A2.1)) →
  (∃ (k : ℝ), B.2 - A.2 = k * (B.1 - A.1) ∧ 0 = k * (2 - A.1) + A.2) :=
sorry

end NUMINAMATH_CALUDE_locus_and_fixed_point_l156_15634


namespace NUMINAMATH_CALUDE_special_triangle_side_length_l156_15601

/-- An equilateral triangle with a point inside satisfying certain distances -/
structure SpecialTriangle where
  /-- Side length of the equilateral triangle -/
  t : ℝ
  /-- Point A of the triangle -/
  A : ℝ × ℝ
  /-- Point B of the triangle -/
  B : ℝ × ℝ
  /-- Point C of the triangle -/
  C : ℝ × ℝ
  /-- Point Q inside the triangle -/
  Q : ℝ × ℝ
  /-- Triangle ABC is equilateral with side length t -/
  equilateral : ‖A - B‖ = t ∧ ‖B - C‖ = t ∧ ‖C - A‖ = t
  /-- Distance AQ is 2 -/
  AQ_dist : ‖A - Q‖ = 2
  /-- Distance BQ is 2√2 -/
  BQ_dist : ‖B - Q‖ = 2 * Real.sqrt 2
  /-- Distance CQ is 3 -/
  CQ_dist : ‖C - Q‖ = 3

/-- Theorem stating that the side length of the special triangle is √15 -/
theorem special_triangle_side_length (tri : SpecialTriangle) : tri.t = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_side_length_l156_15601


namespace NUMINAMATH_CALUDE_average_pen_price_l156_15647

/-- Represents the types of pens --/
inductive PenType
  | A
  | B
  | C
  | D

/-- Given data about pen sales --/
def pen_data : List (PenType × Nat × Nat) :=
  [(PenType.A, 5, 5), (PenType.B, 3, 8), (PenType.C, 2, 27), (PenType.D, 1, 10)]

/-- Total number of pens sold --/
def total_pens : Nat := 50

/-- Theorem stating that the average unit price of pens sold is 2.26元 --/
theorem average_pen_price :
  let total_revenue := (pen_data.map (fun (_, price, quantity) => price * quantity)).sum
  let average_price := (total_revenue : ℚ) / total_pens
  average_price = 226 / 100 := by
  sorry

#check average_pen_price

end NUMINAMATH_CALUDE_average_pen_price_l156_15647


namespace NUMINAMATH_CALUDE_jacob_age_l156_15606

theorem jacob_age (maya drew peter john jacob : ℕ) 
  (h1 : drew = maya + 5)
  (h2 : peter = drew + 4)
  (h3 : john = 30)
  (h4 : john = 2 * maya)
  (h5 : jacob + 2 = (peter + 2) / 2) :
  jacob = 11 := by
  sorry

end NUMINAMATH_CALUDE_jacob_age_l156_15606


namespace NUMINAMATH_CALUDE_y_value_l156_15650

theorem y_value (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l156_15650


namespace NUMINAMATH_CALUDE_sum_of_digits_mod_9_C_mod_9_eq_5_l156_15662

/-- The sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A is the sum of digits of 4568^7777 -/
def A : ℕ := sumOfDigits (4568^7777)

/-- B is the sum of digits of A -/
def B : ℕ := sumOfDigits A

/-- C is the sum of digits of B -/
def C : ℕ := sumOfDigits B

/-- Theorem stating that the sum of digits of a number is congruent to the number modulo 9 -/
theorem sum_of_digits_mod_9 (n : ℕ) : sumOfDigits n ≡ n [MOD 9] := sorry

/-- Main theorem to prove -/
theorem C_mod_9_eq_5 : C ≡ 5 [MOD 9] := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_mod_9_C_mod_9_eq_5_l156_15662


namespace NUMINAMATH_CALUDE_inequality_solution_set_l156_15615

-- Define the inequality
def inequality (x : ℝ) : Prop := (3*x - 1) / (x - 2) ≤ 0

-- Define the solution set
def solution_set : Set ℝ := {x | 1/3 ≤ x ∧ x < 2}

-- Theorem statement
theorem inequality_solution_set :
  ∀ x : ℝ, x ≠ 2 → (x ∈ solution_set ↔ inequality x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l156_15615


namespace NUMINAMATH_CALUDE_integer_less_than_sqrt_10_l156_15635

theorem integer_less_than_sqrt_10 : ∃ n : ℤ, (n : ℝ) < Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_integer_less_than_sqrt_10_l156_15635


namespace NUMINAMATH_CALUDE_combined_average_marks_specific_average_marks_l156_15620

/-- Given two classes of students with their respective sizes and average marks,
    calculate the combined average mark of all students. -/
theorem combined_average_marks (n1 n2 : ℕ) (avg1 avg2 : ℝ) :
  n1 > 0 → n2 > 0 →
  let total_students := n1 + n2
  let total_marks := n1 * avg1 + n2 * avg2
  total_marks / total_students = (n1 * avg1 + n2 * avg2) / (n1 + n2) := by
  sorry

/-- The average mark of all students given the specific class sizes and averages. -/
theorem specific_average_marks :
  let n1 := 24
  let n2 := 50
  let avg1 := 40
  let avg2 := 60
  let total_students := n1 + n2
  let total_marks := n1 * avg1 + n2 * avg2
  total_marks / total_students = (24 * 40 + 50 * 60) / (24 + 50) := by
  sorry

end NUMINAMATH_CALUDE_combined_average_marks_specific_average_marks_l156_15620


namespace NUMINAMATH_CALUDE_shopping_theorem_l156_15631

def shopping_problem (shoe_price : ℝ) (shoe_discount : ℝ) (shirt_price : ℝ) (num_shirts : ℕ) (final_discount : ℝ) : Prop :=
  let discounted_shoe_price := shoe_price * (1 - shoe_discount)
  let total_shirt_price := shirt_price * num_shirts
  let subtotal := discounted_shoe_price + total_shirt_price
  let final_price := subtotal * (1 - final_discount)
  final_price = 285

theorem shopping_theorem :
  shopping_problem 200 0.30 80 2 0.05 := by
  sorry

end NUMINAMATH_CALUDE_shopping_theorem_l156_15631


namespace NUMINAMATH_CALUDE_max_perimeter_after_cut_l156_15691

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem: Maximum perimeter after cutting out a smaller rectangle -/
theorem max_perimeter_after_cut (original : Rectangle) (cutout : Rectangle) :
  original.length = 20 ∧ 
  original.width = 16 ∧ 
  cutout.length = 10 ∧ 
  cutout.width = 5 →
  ∃ (remaining : Rectangle), 
    perimeter remaining = 92 ∧ 
    ∀ (other : Rectangle), perimeter other ≤ perimeter remaining :=
by sorry

end NUMINAMATH_CALUDE_max_perimeter_after_cut_l156_15691


namespace NUMINAMATH_CALUDE_fraction_subtraction_l156_15684

theorem fraction_subtraction : 3 / 5 - (2 / 15 + 1 / 3) = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l156_15684


namespace NUMINAMATH_CALUDE_museum_ticket_problem_l156_15648

/-- Represents the cost calculation for museum tickets with discounts -/
structure TicketCost where
  basePrice : ℕ
  option1Discount : ℚ
  option2Discount : ℚ
  freeTickets : ℕ

/-- Calculates the cost for Option 1 -/
def option1Cost (tc : TicketCost) (students : ℕ) : ℚ :=
  tc.basePrice * (1 - tc.option1Discount) * students

/-- Calculates the cost for Option 2 -/
def option2Cost (tc : TicketCost) (students : ℕ) : ℚ :=
  tc.basePrice * (1 - tc.option2Discount) * (students - tc.freeTickets)

theorem museum_ticket_problem (tc : TicketCost)
    (h1 : tc.basePrice = 30)
    (h2 : tc.option1Discount = 0.3)
    (h3 : tc.option2Discount = 0.2)
    (h4 : tc.freeTickets = 5) :
  (option1Cost tc 45 < option2Cost tc 45) ∧
  (∃ x : ℕ, x = 40 ∧ option1Cost tc x = option2Cost tc x) := by
  sorry


end NUMINAMATH_CALUDE_museum_ticket_problem_l156_15648


namespace NUMINAMATH_CALUDE_sin_period_l156_15656

theorem sin_period (x : ℝ) : 
  let f : ℝ → ℝ := fun x => Real.sin ((1/2) * x + 3)
  ∃ T : ℝ, T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S) ∧ T = 4 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sin_period_l156_15656


namespace NUMINAMATH_CALUDE_summer_work_hours_adjustment_l156_15618

theorem summer_work_hours_adjustment (
  original_hours_per_week : ℝ)
  (original_weeks : ℕ)
  (total_earnings : ℝ)
  (lost_weeks : ℕ)
  (h1 : original_hours_per_week = 20)
  (h2 : original_weeks = 12)
  (h3 : total_earnings = 3000)
  (h4 : lost_weeks = 2)
  (h5 : total_earnings = original_hours_per_week * original_weeks * (total_earnings / (original_hours_per_week * original_weeks)))
  : ∃ new_hours_per_week : ℝ,
    new_hours_per_week * (original_weeks - lost_weeks) * (total_earnings / (original_hours_per_week * original_weeks)) = total_earnings ∧
    new_hours_per_week = 24 :=
by sorry

end NUMINAMATH_CALUDE_summer_work_hours_adjustment_l156_15618


namespace NUMINAMATH_CALUDE_exponent_multiplication_l156_15657

theorem exponent_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l156_15657


namespace NUMINAMATH_CALUDE_equivalent_representations_l156_15649

theorem equivalent_representations (n : ℕ+) :
  (∃ (x y : ℕ+), n = 3 * x^2 + y^2) ↔ (∃ (u v : ℕ+), n = u^2 + u * v + v^2) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_representations_l156_15649


namespace NUMINAMATH_CALUDE_cauchy_inequality_2d_l156_15695

theorem cauchy_inequality_2d (a b c d : ℝ) : 
  (a * c + b * d)^2 ≤ (a^2 + b^2) * (c^2 + d^2) ∧ 
  ((a * c + b * d)^2 = (a^2 + b^2) * (c^2 + d^2) ↔ a * d = b * c) :=
sorry

end NUMINAMATH_CALUDE_cauchy_inequality_2d_l156_15695


namespace NUMINAMATH_CALUDE_range_of_m_for_always_nonnegative_quadratic_l156_15664

theorem range_of_m_for_always_nonnegative_quadratic :
  {m : ℝ | ∀ x : ℝ, x^2 + m*x + 2*m + 5 ≥ 0} = {m : ℝ | -2 ≤ m ∧ m ≤ 10} := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_for_always_nonnegative_quadratic_l156_15664


namespace NUMINAMATH_CALUDE_perfect_cube_from_sum_l156_15661

theorem perfect_cube_from_sum (a b c : ℤ) (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h_sum : ∃ (n : ℤ), a / b + b / c + c / a = n) : 
  ∃ (m : ℤ), a * b * c = m^3 := by
  sorry

end NUMINAMATH_CALUDE_perfect_cube_from_sum_l156_15661


namespace NUMINAMATH_CALUDE_expand_and_simplify_l156_15625

theorem expand_and_simplify (a : ℝ) : a * (a + 2) - 2 * a = a^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l156_15625


namespace NUMINAMATH_CALUDE_f_condition_equivalent_to_a_range_l156_15666

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x + (1/2) * a * x^2 + a * x

theorem f_condition_equivalent_to_a_range :
  ∀ a : ℝ, (∀ x : ℝ, 2 * Real.exp 1 * f a x + Real.exp 1 + 2 ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_f_condition_equivalent_to_a_range_l156_15666


namespace NUMINAMATH_CALUDE_cody_marbles_l156_15604

/-- The number of marbles Cody gave to his brother -/
def marbles_given : ℕ := 5

/-- The number of marbles Cody has now -/
def marbles_now : ℕ := 7

/-- The initial number of marbles Cody had -/
def initial_marbles : ℕ := marbles_now + marbles_given

theorem cody_marbles : initial_marbles = 12 := by
  sorry

end NUMINAMATH_CALUDE_cody_marbles_l156_15604


namespace NUMINAMATH_CALUDE_merchant_discount_percentage_l156_15633

/-- Proves that if a merchant marks up goods by 60% and makes a 20% profit after offering a discount, then the discount percentage is 25%. -/
theorem merchant_discount_percentage 
  (markup_percentage : ℝ) 
  (profit_percentage : ℝ) 
  (discount_percentage : ℝ) 
  (h1 : markup_percentage = 60) 
  (h2 : profit_percentage = 20) : 
  discount_percentage = 25 := by
  sorry

end NUMINAMATH_CALUDE_merchant_discount_percentage_l156_15633


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l156_15624

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Theorem statement
theorem geometric_sequence_middle_term 
  (a : ℕ → ℝ) 
  (h : is_geometric_sequence a) 
  (h1 : a 0 = 1) 
  (h4 : a 4 = 4) : 
  a 2 = 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l156_15624


namespace NUMINAMATH_CALUDE_project_hours_ratio_l156_15641

/-- Proves that given the conditions of the project hours, the ratio of Pat's time to Kate's time is 4:3 -/
theorem project_hours_ratio :
  ∀ (pat kate mark : ℕ),
  pat + kate + mark = 189 →
  ∃ (r : ℚ), pat = r * kate →
  pat = (1 : ℚ) / 3 * mark →
  mark = kate + 105 →
  r = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_project_hours_ratio_l156_15641


namespace NUMINAMATH_CALUDE_special_integer_count_l156_15677

/-- Count of positive integers less than 100,000 with at most two different digits -/
def count_special_integers : ℕ :=
  let single_digit_count := 45
  let two_digit_count_no_zero := 1872
  let two_digit_count_with_zero := 234
  single_digit_count + two_digit_count_no_zero + two_digit_count_with_zero

/-- The count of positive integers less than 100,000 with at most two different digits is 2151 -/
theorem special_integer_count : count_special_integers = 2151 := by
  sorry

end NUMINAMATH_CALUDE_special_integer_count_l156_15677


namespace NUMINAMATH_CALUDE_box_volume_l156_15652

/-- The volume of a box formed by cutting squares from corners of a square sheet -/
theorem box_volume (sheet_side : ℝ) (corner_cut : ℝ) : 
  sheet_side = 12 → corner_cut = 2 → 
  (sheet_side - 2 * corner_cut) * (sheet_side - 2 * corner_cut) * corner_cut = 128 :=
by
  sorry

end NUMINAMATH_CALUDE_box_volume_l156_15652


namespace NUMINAMATH_CALUDE_quiz_score_theorem_l156_15663

/-- Represents a quiz score configuration -/
structure QuizScore where
  correct : ℕ
  unanswered : ℕ
  incorrect : ℕ

/-- The quiz scoring system -/
def quizScoring (qs : QuizScore) : ℚ :=
  4 * qs.correct + 1.5 * qs.unanswered

/-- Predicate for valid quiz configurations -/
def isValidQuizScore (qs : QuizScore) : Prop :=
  qs.correct + qs.unanswered + qs.incorrect = 30

/-- Predicate for scores achievable in exactly three ways -/
def hasExactlyThreeConfigurations (score : ℚ) : Prop :=
  ∃ (c1 c2 c3 : QuizScore),
    c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧
    isValidQuizScore c1 ∧ isValidQuizScore c2 ∧ isValidQuizScore c3 ∧
    quizScoring c1 = score ∧ quizScoring c2 = score ∧ quizScoring c3 = score ∧
    ∀ c, isValidQuizScore c ∧ quizScoring c = score → c = c1 ∨ c = c2 ∨ c = c3

theorem quiz_score_theorem :
  ∃ score, 0 ≤ score ∧ score ≤ 120 ∧ hasExactlyThreeConfigurations score := by
  sorry

end NUMINAMATH_CALUDE_quiz_score_theorem_l156_15663


namespace NUMINAMATH_CALUDE_modular_exponentiation_16_cube_mod_7_l156_15654

theorem modular_exponentiation_16_cube_mod_7 :
  ∃ m : ℕ, 16^3 ≡ m [ZMOD 7] ∧ 0 ≤ m ∧ m < 7 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_modular_exponentiation_16_cube_mod_7_l156_15654


namespace NUMINAMATH_CALUDE_percentage_not_red_roses_is_92_percent_l156_15611

/-- Represents the number of flowers of each type in the garden -/
structure GardenFlowers where
  roses : ℕ
  tulips : ℕ
  daisies : ℕ
  lilies : ℕ
  sunflowers : ℕ

/-- Calculates the total number of flowers in the garden -/
def totalFlowers (g : GardenFlowers) : ℕ :=
  g.roses + g.tulips + g.daisies + g.lilies + g.sunflowers

/-- Calculates the number of red roses in the garden -/
def redRoses (g : GardenFlowers) : ℕ :=
  g.roses / 2

/-- Calculates the percentage of flowers that are not red roses -/
def percentageNotRedRoses (g : GardenFlowers) : ℚ :=
  (totalFlowers g - redRoses g : ℚ) / (totalFlowers g : ℚ) * 100

/-- Theorem stating that 92% of flowers in the given garden are not red roses -/
theorem percentage_not_red_roses_is_92_percent (g : GardenFlowers) 
  (h1 : g.roses = 25)
  (h2 : g.tulips = 40)
  (h3 : g.daisies = 60)
  (h4 : g.lilies = 15)
  (h5 : g.sunflowers = 10) :
  percentageNotRedRoses g = 92 := by
  sorry

end NUMINAMATH_CALUDE_percentage_not_red_roses_is_92_percent_l156_15611


namespace NUMINAMATH_CALUDE_ring_arrangement_count_l156_15673

def ring_arrangements (total_rings : ℕ) (rings_to_use : ℕ) (fingers : ℕ) : ℕ :=
  Nat.choose total_rings rings_to_use * 
  Nat.factorial rings_to_use * 
  Nat.choose (rings_to_use + fingers - 1) (fingers - 1)

theorem ring_arrangement_count :
  ring_arrangements 8 5 4 = 376320 :=
by sorry

end NUMINAMATH_CALUDE_ring_arrangement_count_l156_15673


namespace NUMINAMATH_CALUDE_truck_speed_l156_15603

/-- Proves that a truck traveling 600 meters in 40 seconds has a speed of 54 kilometers per hour -/
theorem truck_speed : ∀ (distance : ℝ) (time : ℝ) (speed_ms : ℝ) (speed_kmh : ℝ),
  distance = 600 →
  time = 40 →
  speed_ms = distance / time →
  speed_kmh = speed_ms * 3.6 →
  speed_kmh = 54 := by
  sorry

#check truck_speed

end NUMINAMATH_CALUDE_truck_speed_l156_15603


namespace NUMINAMATH_CALUDE_combined_female_average_score_l156_15623

theorem combined_female_average_score 
  (a b c d : ℕ) 
  (adam_avg : (71 * a + 76 * b) / (a + b) = 74)
  (baker_avg : (81 * c + 90 * d) / (c + d) = 84)
  (male_avg : (71 * a + 81 * c) / (a + c) = 79) :
  (76 * b + 90 * d) / (b + d) = 84 :=
sorry

end NUMINAMATH_CALUDE_combined_female_average_score_l156_15623


namespace NUMINAMATH_CALUDE_sqrt_problem_l156_15698

theorem sqrt_problem (h : Real.sqrt 100.4004 = 10.02) : Real.sqrt 1.004004 = 1.002 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_problem_l156_15698


namespace NUMINAMATH_CALUDE_system_solution_l156_15605

theorem system_solution : ∃! (x y z : ℝ), 
  x * y / (x + y) = 1 / 3 ∧
  y * z / (y + z) = 1 / 4 ∧
  z * x / (z + x) = 1 / 5 ∧
  x = 1 / 2 ∧ y = 1 ∧ z = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l156_15605


namespace NUMINAMATH_CALUDE_second_train_length_second_train_length_problem_l156_15638

/-- Calculates the length of the second train given the speeds of two trains moving in opposite directions, the length of the first train, and the time taken to cross each other. -/
theorem second_train_length 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (length1 : ℝ) 
  (crossing_time : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let total_distance := relative_speed * crossing_time / 3600
  total_distance - length1

/-- The length of the second train is 0.9 km given the specified conditions -/
theorem second_train_length_problem : 
  second_train_length 60 90 1.10 47.99999999999999 = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_second_train_length_second_train_length_problem_l156_15638


namespace NUMINAMATH_CALUDE_ratio_unchanged_l156_15609

/-- Represents the number of animals in the zoo -/
structure ZooPopulation where
  cheetahs : ℕ
  pandas : ℕ

/-- The zoo population 5 years ago -/
def initial_population : ZooPopulation := sorry

/-- The current zoo population -/
def current_population : ZooPopulation :=
  { cheetahs := initial_population.cheetahs + 2,
    pandas := initial_population.pandas + 6 }

/-- The ratio of cheetahs to pandas -/
def cheetah_panda_ratio (pop : ZooPopulation) : ℚ :=
  pop.cheetahs / pop.pandas

theorem ratio_unchanged :
  cheetah_panda_ratio initial_population = cheetah_panda_ratio current_population :=
by sorry

end NUMINAMATH_CALUDE_ratio_unchanged_l156_15609


namespace NUMINAMATH_CALUDE_inequality_proof_l156_15646

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_xyz : x * y * z ≥ 1) :
  (x^4 + y) * (y^4 + z) * (z^4 + x) ≥ (x + y^2) * (y + z^2) * (z + x^2) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l156_15646


namespace NUMINAMATH_CALUDE_parabola_directrix_l156_15669

/-- Given a parabola y² = 2px and a point M(1, m) on the parabola,
    with the distance from M to its focus being 5,
    prove that the equation of the directrix is x = -4 -/
theorem parabola_directrix (p : ℝ) (m : ℝ) : 
  m^2 = 2*p  -- M(1, m) is on the parabola y² = 2px
  → (1 - p/2)^2 + m^2 = 25  -- Distance from M to focus is 5
  → -p/2 = -4  -- Equation of directrix is x = -p/2
  := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l156_15669


namespace NUMINAMATH_CALUDE_product_125_sum_31_l156_15616

theorem product_125_sum_31 (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 125 →
  (a : ℕ) + b + c = 31 := by
sorry

end NUMINAMATH_CALUDE_product_125_sum_31_l156_15616


namespace NUMINAMATH_CALUDE_gcd_lcm_product_30_45_l156_15645

theorem gcd_lcm_product_30_45 : Nat.gcd 30 45 * Nat.lcm 30 45 = 1350 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_30_45_l156_15645


namespace NUMINAMATH_CALUDE_sqrt_four_fourth_power_sum_l156_15685

theorem sqrt_four_fourth_power_sum : Real.sqrt (4^4 + 4^4 + 4^4 + 4^4) = 32 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_fourth_power_sum_l156_15685


namespace NUMINAMATH_CALUDE_total_tickets_sold_l156_15636

theorem total_tickets_sold (adult_price child_price : ℕ) 
  (adult_tickets child_tickets total_receipts : ℕ) :
  adult_price = 12 →
  child_price = 4 →
  adult_tickets = 90 →
  child_tickets = 40 →
  total_receipts = 840 →
  adult_tickets * adult_price + child_tickets * child_price = total_receipts →
  adult_tickets + child_tickets = 130 := by
sorry

end NUMINAMATH_CALUDE_total_tickets_sold_l156_15636


namespace NUMINAMATH_CALUDE_chair_count_difference_l156_15693

/-- Represents the number of chairs of each color in a classroom. -/
structure ClassroomChairs where
  blue : Nat
  green : Nat
  white : Nat

/-- Theorem about the difference in chair counts in a classroom. -/
theorem chair_count_difference 
  (chairs : ClassroomChairs) 
  (h1 : chairs.blue = 10)
  (h2 : chairs.green = 3 * chairs.blue)
  (h3 : chairs.blue + chairs.green + chairs.white = 67) :
  chairs.blue + chairs.green - chairs.white = 13 := by
  sorry


end NUMINAMATH_CALUDE_chair_count_difference_l156_15693


namespace NUMINAMATH_CALUDE_four_by_four_min_cuts_five_by_five_min_cuts_l156_15687

/-- Represents a square grid of size n x n -/
structure Square (n : ℕ) where
  size : ℕ
  size_eq : size = n

/-- Minimum number of cuts required to divide a square into unit squares -/
def min_cuts (s : Square n) : ℕ :=
  sorry

/-- Pieces can be overlapped during cutting -/
axiom overlap_allowed : ∀ (n : ℕ) (s : Square n), True

theorem four_by_four_min_cuts :
  ∀ (s : Square 4), min_cuts s = 4 :=
sorry

theorem five_by_five_min_cuts :
  ∀ (s : Square 5), min_cuts s = 6 :=
sorry

end NUMINAMATH_CALUDE_four_by_four_min_cuts_five_by_five_min_cuts_l156_15687


namespace NUMINAMATH_CALUDE_bake_sale_donation_ratio_is_one_to_one_l156_15668

/-- Represents the financial details of Andrew's bake sale fundraiser. -/
structure BakeSale where
  total_earnings : ℕ
  ingredient_cost : ℕ
  personal_donation : ℕ
  total_homeless_donation : ℕ

/-- Calculates the ratio of homeless shelter donation to food bank donation. -/
def donation_ratio (sale : BakeSale) : ℚ :=
  let available_for_donation := sale.total_earnings - sale.ingredient_cost
  let homeless_donation := sale.total_homeless_donation - sale.personal_donation
  let food_bank_donation := available_for_donation - homeless_donation
  homeless_donation / food_bank_donation

/-- Theorem stating that the donation ratio is 1:1 for the given bake sale. -/
theorem bake_sale_donation_ratio_is_one_to_one 
  (sale : BakeSale) 
  (h1 : sale.total_earnings = 400)
  (h2 : sale.ingredient_cost = 100)
  (h3 : sale.personal_donation = 10)
  (h4 : sale.total_homeless_donation = 160) : 
  donation_ratio sale = 1 := by
  sorry

end NUMINAMATH_CALUDE_bake_sale_donation_ratio_is_one_to_one_l156_15668


namespace NUMINAMATH_CALUDE_no_function_pair_exists_l156_15678

theorem no_function_pair_exists : ¬∃ (f g : ℝ → ℝ), ∀ x : ℝ, f (g x) = x^2 ∧ g (f x) = x^3 := by
  sorry

end NUMINAMATH_CALUDE_no_function_pair_exists_l156_15678


namespace NUMINAMATH_CALUDE_nonnegative_solutions_count_l156_15667

theorem nonnegative_solutions_count : ∃! (x : ℝ), x ≥ 0 ∧ x^2 + 6*x = 18 := by
  sorry

end NUMINAMATH_CALUDE_nonnegative_solutions_count_l156_15667


namespace NUMINAMATH_CALUDE_no_primes_divisible_by_45_l156_15692

-- Definition of a prime number
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Theorem statement
theorem no_primes_divisible_by_45 : ¬∃ p : ℕ, isPrime p ∧ 45 ∣ p := by
  sorry

end NUMINAMATH_CALUDE_no_primes_divisible_by_45_l156_15692


namespace NUMINAMATH_CALUDE_max_value_quadratic_l156_15690

theorem max_value_quadratic (x : ℝ) : 
  (∃ (z : ℝ), z = x^2 - 14*x + 10) → 
  (∃ (max_z : ℝ), max_z = -39 ∧ ∀ (y : ℝ), y = x^2 - 14*x + 10 → y ≤ max_z) :=
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l156_15690


namespace NUMINAMATH_CALUDE_table_length_l156_15629

/-- Proves that a rectangular table with an area of 54 square meters and a width of 600 centimeters has a length of 900 centimeters. -/
theorem table_length (area : ℝ) (width : ℝ) (length : ℝ) : 
  area = 54 → 
  width = 6 →
  area = length * width →
  length * 100 = 900 := by
  sorry

#check table_length

end NUMINAMATH_CALUDE_table_length_l156_15629


namespace NUMINAMATH_CALUDE_paint_replacement_theorem_l156_15628

def paint_replacement_fractions (initial_red initial_blue initial_green : ℚ)
                                (replacement_red replacement_blue replacement_green : ℚ)
                                (final_red final_blue final_green : ℚ) : Prop :=
  let r := (initial_red - final_red) / (initial_red - replacement_red)
  let b := (initial_blue - final_blue) / (initial_blue - replacement_blue)
  let g := (initial_green - final_green) / (initial_green - replacement_green)
  r = 2/3 ∧ b = 3/5 ∧ g = 7/15

theorem paint_replacement_theorem :
  paint_replacement_fractions (60/100) (40/100) (25/100) (30/100) (15/100) (10/100) (40/100) (25/100) (18/100) :=
by
  sorry

end NUMINAMATH_CALUDE_paint_replacement_theorem_l156_15628


namespace NUMINAMATH_CALUDE_division_result_l156_15607

theorem division_result : (64 : ℝ) / 0.08 = 800 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l156_15607


namespace NUMINAMATH_CALUDE_largest_t_value_l156_15665

theorem largest_t_value : 
  let f (t : ℝ) := (15 * t^2 - 38 * t + 14) / (4 * t - 3) + 6 * t
  ∃ (t_max : ℝ), t_max = 1 ∧ 
    (∀ (t : ℝ), f t = 7 * t - 2 → t ≤ t_max) ∧
    (f t_max = 7 * t_max - 2) :=
by sorry

end NUMINAMATH_CALUDE_largest_t_value_l156_15665


namespace NUMINAMATH_CALUDE_relationship_2x_3sinx_l156_15680

theorem relationship_2x_3sinx :
  ∃ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧
  (∀ x : ℝ, 0 < x ∧ x < θ → 2 * x < 3 * Real.sin x) ∧
  (2 * θ = 3 * Real.sin θ) ∧
  (∀ x : ℝ, θ < x ∧ x < π / 2 → 2 * x > 3 * Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_relationship_2x_3sinx_l156_15680


namespace NUMINAMATH_CALUDE_max_profit_is_1200_l156_15632

/-- Represents the cost and profit calculation for a shopping mall's purchasing plan. -/
structure ShoppingMall where
  cost_A : ℝ  -- Cost price of good A
  cost_B : ℝ  -- Cost price of good B
  sell_A : ℝ  -- Selling price of good A
  sell_B : ℝ  -- Selling price of good B
  total_units : ℕ  -- Total units to purchase

/-- Calculates the profit for a given purchasing plan. -/
def profit (sm : ShoppingMall) (units_A : ℕ) : ℝ :=
  let units_B := sm.total_units - units_A
  (sm.sell_A * units_A + sm.sell_B * units_B) - (sm.cost_A * units_A + sm.cost_B * units_B)

/-- Theorem stating that the maximum profit is $1200 under the given conditions. -/
theorem max_profit_is_1200 (sm : ShoppingMall) 
  (h1 : sm.cost_A + 3 * sm.cost_B = 240)
  (h2 : 2 * sm.cost_A + sm.cost_B = 130)
  (h3 : sm.sell_A = 40)
  (h4 : sm.sell_B = 90)
  (h5 : sm.total_units = 100)
  : ∃ (units_A : ℕ), 
    units_A ≥ 4 * (sm.total_units - units_A) ∧ 
    ∀ (x : ℕ), x ≥ 4 * (sm.total_units - x) → profit sm units_A ≥ profit sm x :=
by sorry

end NUMINAMATH_CALUDE_max_profit_is_1200_l156_15632


namespace NUMINAMATH_CALUDE_garden_perimeter_l156_15688

/-- The perimeter of a rectangular garden with width 16 meters and the same area as a rectangular playground of length 16 meters and width 12 meters is 56 meters. -/
theorem garden_perimeter (garden_width playground_length playground_width : ℝ) :
  garden_width = 16 →
  playground_length = 16 →
  playground_width = 12 →
  garden_width * (playground_length * playground_width / garden_width) + 2 * garden_width = 56 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l156_15688


namespace NUMINAMATH_CALUDE_amount_after_two_years_l156_15642

/-- Calculate the amount after n years with a given initial value and annual increase rate -/
def amountAfterYears (initialValue : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initialValue * (1 + rate) ^ years

/-- Theorem: Given an initial amount of 6400 and an annual increase rate of 1/8,
    the amount after 2 years will be 8100 -/
theorem amount_after_two_years :
  let initialValue : ℝ := 6400
  let rate : ℝ := 1/8
  let years : ℕ := 2
  amountAfterYears initialValue rate years = 8100 := by
  sorry

end NUMINAMATH_CALUDE_amount_after_two_years_l156_15642


namespace NUMINAMATH_CALUDE_arithmetic_equality_l156_15608

theorem arithmetic_equality : 4 * (8 - 3) - 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l156_15608


namespace NUMINAMATH_CALUDE_min_value_theorem_l156_15600

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4*a + 3*b - 1 = 0) :
  ∃ (min : ℝ), min = 3 + 2*Real.sqrt 2 ∧ 
  ∀ (x : ℝ), x = 1/(2*a + b) + 1/(a + b) → x ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l156_15600


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l156_15612

/-- Given a line y = mx + b, prove that mb < -1 --/
theorem line_slope_intercept_product (m b : ℝ) : m * b < -1 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l156_15612


namespace NUMINAMATH_CALUDE_swimming_pool_volume_l156_15614

/-- The volume of a cylindrical swimming pool -/
theorem swimming_pool_volume (diameter : ℝ) (depth : ℝ) (volume : ℝ) :
  diameter = 16 →
  depth = 4 →
  volume = π * (diameter / 2)^2 * depth →
  volume = 256 * π := by
  sorry

end NUMINAMATH_CALUDE_swimming_pool_volume_l156_15614


namespace NUMINAMATH_CALUDE_two_digit_average_decimal_l156_15660

theorem two_digit_average_decimal (m n : ℕ) : 
  10 ≤ m ∧ m < 100 ∧ 10 ≤ n ∧ n < 100 →  -- m and n are 2-digit positive integers
  (m + n) / 2 = m + n / 100 →             -- their average equals the decimal representation
  max m n = 50 :=                         -- the larger of the two is 50
by sorry

end NUMINAMATH_CALUDE_two_digit_average_decimal_l156_15660


namespace NUMINAMATH_CALUDE_exam_mode_l156_15671

/-- Represents a score in the music theory exam -/
structure Score where
  value : ℕ
  deriving Repr

/-- Represents the frequency of each score -/
def ScoreFrequency := Score → ℕ

/-- The set of all scores in the exam -/
def ExamScores : Set Score := sorry

/-- The frequency distribution of scores in the exam -/
def examFrequency : ScoreFrequency := sorry

/-- Definition of mode: the score that appears most frequently -/
def isMode (s : Score) (freq : ScoreFrequency) (scores : Set Score) : Prop :=
  ∀ t ∈ scores, freq s ≥ freq t

/-- The mode of the exam scores is 88 -/
theorem exam_mode :
  ∃ s : Score, s.value = 88 ∧ isMode s examFrequency ExamScores := by sorry

end NUMINAMATH_CALUDE_exam_mode_l156_15671


namespace NUMINAMATH_CALUDE_max_ab_value_l156_15675

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (heq : a^2 + b^2 - 6*a = 0) :
  ∃ (max_ab : ℝ), max_ab = (27 * Real.sqrt 3) / 4 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x^2 + y^2 - 6*x = 0 → x*y ≤ max_ab :=
sorry

end NUMINAMATH_CALUDE_max_ab_value_l156_15675


namespace NUMINAMATH_CALUDE_tangent_difference_l156_15627

/-- Given two circles in a plane, this theorem proves that the difference between
    the squares of their external and internal tangent lengths is 30. -/
theorem tangent_difference (r₁ r₂ x y A₁₀ : ℝ) : 
  r₁ > 0 → r₂ > 0 → A₁₀ > 0 →
  r₁ * r₂ = 15 / 2 →
  x^2 + (r₁ + r₂)^2 = A₁₀^2 →
  y^2 + (r₁ - r₂)^2 = A₁₀^2 →
  y^2 - x^2 = 30 := by
sorry

end NUMINAMATH_CALUDE_tangent_difference_l156_15627


namespace NUMINAMATH_CALUDE_function_composition_multiplication_l156_15653

-- Define the composition operation
def compose (f g : ℝ → ℝ) : ℝ → ℝ := λ x => f (g x)

-- Define the multiplication operation
def multiply (f g : ℝ → ℝ) : ℝ → ℝ := λ x => f x * g x

-- State the theorem
theorem function_composition_multiplication (f g h : ℝ → ℝ) :
  compose (multiply f g) h = multiply (compose f h) (compose g h) := by
  sorry

end NUMINAMATH_CALUDE_function_composition_multiplication_l156_15653


namespace NUMINAMATH_CALUDE_least_positive_angle_theorem_l156_15670

theorem least_positive_angle_theorem : ∃ θ : ℝ,
  θ > 0 ∧
  θ ≤ 90 ∧
  Real.cos (10 * π / 180) = Real.sin (30 * π / 180) + Real.sin (θ * π / 180) ∧
  ∀ φ : ℝ, φ > 0 ∧ φ < θ →
    Real.cos (10 * π / 180) ≠ Real.sin (30 * π / 180) + Real.sin (φ * π / 180) ∧
  θ = 80 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_angle_theorem_l156_15670


namespace NUMINAMATH_CALUDE_first_platform_length_l156_15682

/-- The length of a train in meters. -/
def train_length : ℝ := 350

/-- The time taken to cross the first platform in seconds. -/
def time_first : ℝ := 15

/-- The length of the second platform in meters. -/
def length_second : ℝ := 250

/-- The time taken to cross the second platform in seconds. -/
def time_second : ℝ := 20

/-- The length of the first platform in meters. -/
def length_first : ℝ := 100

theorem first_platform_length :
  (train_length + length_first) / time_first = (train_length + length_second) / time_second :=
by sorry

end NUMINAMATH_CALUDE_first_platform_length_l156_15682


namespace NUMINAMATH_CALUDE_number_problem_l156_15683

theorem number_problem (x : ℚ) : (3 / 4) * x = x - 19 → x = 76 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l156_15683


namespace NUMINAMATH_CALUDE_max_at_neg_two_l156_15676

/-- The function f(x) that we're analyzing -/
def f (m : ℝ) (x : ℝ) : ℝ := x * (x - m)^2

/-- The derivative of f(x) -/
def f_deriv (m : ℝ) (x : ℝ) : ℝ := (x - m)^2 + 2*x*(x - m)

theorem max_at_neg_two (m : ℝ) :
  (∀ x : ℝ, f m x ≤ f m (-2)) → m = -2 :=
sorry

end NUMINAMATH_CALUDE_max_at_neg_two_l156_15676


namespace NUMINAMATH_CALUDE_divisor_problem_l156_15626

theorem divisor_problem : ∃ (x : ℕ), x > 0 ∧ 181 = 9 * x + 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l156_15626


namespace NUMINAMATH_CALUDE_max_table_height_l156_15613

/-- Given a triangle DEF with side lengths 26, 28, and 34, prove that the maximum height k
    of a table formed by making right angle folds parallel to each side is 96√55/54. -/
theorem max_table_height (DE EF FD : ℝ) (h_DE : DE = 26) (h_EF : EF = 28) (h_FD : FD = 34) :
  let s := (DE + EF + FD) / 2
  let A := Real.sqrt (s * (s - DE) * (s - EF) * (s - FD))
  let h_e := 2 * A / EF
  let h_f := 2 * A / FD
  let k := h_e * h_f / (h_e + h_f)
  k = 96 * Real.sqrt 55 / 54 :=
by sorry

end NUMINAMATH_CALUDE_max_table_height_l156_15613


namespace NUMINAMATH_CALUDE_sufficient_condition_increasing_f_increasing_on_interval_l156_15617

/-- A sufficient condition for f(x) = x^2 + 2ax + 1 to be increasing on (1, +∞) -/
theorem sufficient_condition_increasing (a : ℝ) (h : a = -1) :
  ∀ x y, 1 < x → x < y → x^2 + 2*a*x + 1 < y^2 + 2*a*y + 1 := by
  sorry

/-- Definition of the function f(x) = x^2 + 2ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 1

/-- The function f is increasing on (1, +∞) when a = -1 -/
theorem f_increasing_on_interval (a : ℝ) (h : a = -1) :
  StrictMonoOn (f a) (Set.Ioi 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_increasing_f_increasing_on_interval_l156_15617


namespace NUMINAMATH_CALUDE_sqrt_32_minus_cos_45_plus_one_minus_sqrt_2_squared_l156_15639

theorem sqrt_32_minus_cos_45_plus_one_minus_sqrt_2_squared :
  Real.sqrt 32 - Real.cos (π / 4) + (1 - Real.sqrt 2) ^ 2 = 3 + (3 / 2) * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_32_minus_cos_45_plus_one_minus_sqrt_2_squared_l156_15639


namespace NUMINAMATH_CALUDE_product_of_roots_l156_15610

theorem product_of_roots : Real.sqrt 4 ^ (1/3) * Real.sqrt 8 ^ (1/4) = 2 * Real.sqrt 32 ^ (1/12) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l156_15610


namespace NUMINAMATH_CALUDE_hyperbola_midpoint_existence_l156_15658

theorem hyperbola_midpoint_existence :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁^2 - y₁^2/9 = 1) ∧
    (x₂^2 - y₂^2/9 = 1) ∧
    ((x₁ + x₂)/2 = -1) ∧
    ((y₁ + y₂)/2 = -4) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_midpoint_existence_l156_15658


namespace NUMINAMATH_CALUDE_fraction_comparison_l156_15622

theorem fraction_comparison : 
  (9 : ℚ) / 21 = (3 : ℚ) / 7 ∧ 
  (12 : ℚ) / 28 = (3 : ℚ) / 7 ∧ 
  (30 : ℚ) / 70 = (3 : ℚ) / 7 ∧ 
  (13 : ℚ) / 28 ≠ (3 : ℚ) / 7 := by
sorry

end NUMINAMATH_CALUDE_fraction_comparison_l156_15622


namespace NUMINAMATH_CALUDE_max_area_inscribed_triangle_l156_15602

/-- The maximum area of a right-angled isosceles triangle inscribed in a 12x15 rectangle -/
theorem max_area_inscribed_triangle (a b : ℝ) (ha : a = 12) (hb : b = 15) :
  let max_area := Real.sqrt (min a b ^ 2 / 2)
  ∃ (x y : ℝ), x ≤ a ∧ y ≤ b ∧ x = y ∧ x * y / 2 = max_area ^ 2 ∧ max_area ^ 2 = 72 := by
  sorry


end NUMINAMATH_CALUDE_max_area_inscribed_triangle_l156_15602


namespace NUMINAMATH_CALUDE_solution_set_not_empty_or_specific_interval_l156_15672

theorem solution_set_not_empty_or_specific_interval (a : ℝ) :
  ∃ x : ℝ, a * (x - a) * (a * x + a) ≥ 0 ∧
  ¬(∀ x : ℝ, a * (x - a) * (a * x + a) < 0) ∧
  ¬(∀ x : ℝ, (a * (x - a) * (a * x + a) ≥ 0) ↔ (a ≤ x ∧ x ≤ -1)) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_not_empty_or_specific_interval_l156_15672


namespace NUMINAMATH_CALUDE_cement_bags_ratio_l156_15696

theorem cement_bags_ratio (bags1 : ℕ) (weight1 : ℚ) (cost1 : ℚ) (cost2 : ℚ) (weight_ratio : ℚ) :
  bags1 = 80 →
  weight1 = 50 →
  cost1 = 6000 →
  cost2 = 10800 →
  weight_ratio = 3 / 5 →
  (cost2 / (cost1 / bags1 * weight_ratio)) / bags1 = 3 / 1 := by
sorry

end NUMINAMATH_CALUDE_cement_bags_ratio_l156_15696


namespace NUMINAMATH_CALUDE_parallelogram_diagonal_square_l156_15681

structure Parallelogram :=
  (W X Y Z : ℝ × ℝ)
  (is_parallelogram : sorry)
  (area : ℝ)
  (area_eq : area = 24)

def projection (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

def distance (A B : ℝ × ℝ) : ℝ := sorry

theorem parallelogram_diagonal_square
  (WXYZ : Parallelogram)
  (P : ℝ × ℝ)
  (Q : ℝ × ℝ)
  (R : ℝ × ℝ)
  (S : ℝ × ℝ)
  (h_P : P = projection WXYZ.W WXYZ.X WXYZ.Z)
  (h_Q : Q = projection WXYZ.Y WXYZ.X WXYZ.Z)
  (h_R : R = projection WXYZ.X WXYZ.W WXYZ.Y)
  (h_S : S = projection WXYZ.Z WXYZ.W WXYZ.Y)
  (h_PQ : distance P Q = 9)
  (h_RS : distance R S = 10)
  : ∃ (m n p : ℕ), 
    (distance WXYZ.X WXYZ.Z)^2 = m + n * Real.sqrt p ∧
    p.Prime ∧
    m + n + p = 211 := by sorry

end NUMINAMATH_CALUDE_parallelogram_diagonal_square_l156_15681


namespace NUMINAMATH_CALUDE_cosine_product_fifteen_l156_15621

theorem cosine_product_fifteen : 
  (Real.cos (π/15)) * (Real.cos (2*π/15)) * (Real.cos (3*π/15)) * 
  (Real.cos (4*π/15)) * (Real.cos (5*π/15)) * (Real.cos (6*π/15)) * 
  (Real.cos (7*π/15)) = -1/128 := by
sorry

end NUMINAMATH_CALUDE_cosine_product_fifteen_l156_15621


namespace NUMINAMATH_CALUDE_negation_of_proposition_l156_15619

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l156_15619


namespace NUMINAMATH_CALUDE_park_ticket_cost_l156_15651

theorem park_ticket_cost (teacher_count student_count ticket_price total_budget : ℕ) :
  teacher_count = 3 →
  student_count = 9 →
  ticket_price = 22 →
  total_budget = 300 →
  (teacher_count + student_count) * ticket_price ≤ total_budget :=
by
  sorry

end NUMINAMATH_CALUDE_park_ticket_cost_l156_15651


namespace NUMINAMATH_CALUDE_point_A_in_third_quadrant_l156_15659

/-- A point in the Cartesian coordinate system is in the third quadrant if and only if
    both its x and y coordinates are negative. -/
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- The point A with coordinates (-1, -3) lies in the third quadrant. -/
theorem point_A_in_third_quadrant :
  third_quadrant (-1) (-3) := by
  sorry

end NUMINAMATH_CALUDE_point_A_in_third_quadrant_l156_15659


namespace NUMINAMATH_CALUDE_coin_value_calculation_l156_15679

def total_value (num_dimes num_quarters : ℕ) (dime_value quarter_value : ℚ) : ℚ :=
  (num_dimes : ℚ) * dime_value + (num_quarters : ℚ) * quarter_value

theorem coin_value_calculation :
  total_value 22 10 (10 / 100) (25 / 100) = 470 / 100 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_calculation_l156_15679


namespace NUMINAMATH_CALUDE_max_triangle_perimeter_l156_15640

theorem max_triangle_perimeter (x : ℕ) : 
  x > 0 ∧ x < 17 ∧ 8 + x > 9 ∧ 9 + x > 8 → 
  ∀ y : ℕ, y > 0 ∧ y < 17 ∧ 8 + y > 9 ∧ 9 + y > 8 → 
  8 + 9 + x ≥ 8 + 9 + y ∧ 
  8 + 9 + x ≤ 33 :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_perimeter_l156_15640
