import Mathlib

namespace NUMINAMATH_CALUDE_estimate_sum_approximately_equal_500_l1415_141565

def round_to_nearest_hundred (n : ℕ) : ℕ :=
  (n + 50) / 100 * 100

def approximately_equal (a b : ℕ) : Prop :=
  round_to_nearest_hundred a = round_to_nearest_hundred b

theorem estimate_sum_approximately_equal_500 :
  approximately_equal (208 + 298) 500 := by sorry

end NUMINAMATH_CALUDE_estimate_sum_approximately_equal_500_l1415_141565


namespace NUMINAMATH_CALUDE_pears_theorem_l1415_141547

def pears_problem (keith_picked mike_picked sarah_picked keith_gave mike_gave sarah_gave : ℕ) : Prop :=
  let keith_left := keith_picked - keith_gave
  let mike_left := mike_picked - mike_gave
  let sarah_left := sarah_picked - sarah_gave
  keith_left + mike_left + sarah_left = 15

theorem pears_theorem :
  pears_problem 47 12 22 46 5 15 := by sorry

end NUMINAMATH_CALUDE_pears_theorem_l1415_141547


namespace NUMINAMATH_CALUDE_max_value_of_g_l1415_141577

-- Define the function g(x)
def g (x : ℝ) : ℝ := 4 * x - x^4

-- State the theorem
theorem max_value_of_g :
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x, x ∈ Set.Icc 0 2 → g x ≤ g c) ∧
  g c = 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l1415_141577


namespace NUMINAMATH_CALUDE_least_possible_difference_l1415_141583

theorem least_possible_difference (x y z : ℤ) : 
  x < y ∧ y < z ∧ 
  y - x > 5 ∧ 
  Even x ∧ Odd y ∧ Odd z →
  ∀ w, w = z - x → w ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_difference_l1415_141583


namespace NUMINAMATH_CALUDE_common_root_equations_l1415_141504

theorem common_root_equations (p : ℤ) (x : ℚ) : 
  (3 * x^2 - 4 * x + p - 2 = 0 ∧ x^2 - 2 * p * x + 5 = 0) ↔ (p = 3 ∧ x = 1) :=
by sorry

#check common_root_equations

end NUMINAMATH_CALUDE_common_root_equations_l1415_141504


namespace NUMINAMATH_CALUDE_ellen_legos_l1415_141539

/-- The number of legos Ellen lost -/
def lost_legos : ℕ := 57

/-- The number of legos Ellen currently has -/
def current_legos : ℕ := 323

/-- The initial number of legos Ellen had -/
def initial_legos : ℕ := lost_legos + current_legos

theorem ellen_legos : initial_legos = 380 := by
  sorry

end NUMINAMATH_CALUDE_ellen_legos_l1415_141539


namespace NUMINAMATH_CALUDE_pam_has_ten_bags_l1415_141501

/-- Represents the number of apples in each of Gerald's bags -/
def geralds_bag_count : ℕ := 40

/-- Represents the ratio of apples in Pam's bags to Gerald's bags -/
def pam_to_gerald_ratio : ℕ := 3

/-- Represents the total number of apples Pam has -/
def pam_total_apples : ℕ := 1200

/-- Calculates the number of bags Pam has -/
def pam_bag_count : ℕ := pam_total_apples / (geralds_bag_count * pam_to_gerald_ratio)

theorem pam_has_ten_bags : pam_bag_count = 10 := by
  sorry

end NUMINAMATH_CALUDE_pam_has_ten_bags_l1415_141501


namespace NUMINAMATH_CALUDE_product_inequality_l1415_141537

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a + b + c + 2 = a * b * c) :
  (a + 1) * (b + 1) * (c + 1) ≥ 27 ∧
  ((a + 1) * (b + 1) * (c + 1) = 27 ↔ a = 2 ∧ b = 2 ∧ c = 2) := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l1415_141537


namespace NUMINAMATH_CALUDE_lemon_heads_in_package_l1415_141525

/-- The number of Lemon Heads Louis ate -/
def total_lemon_heads : ℕ := 54

/-- The number of whole boxes Louis ate -/
def whole_boxes : ℕ := 9

/-- The number of Lemon Heads in one package -/
def lemon_heads_per_package : ℕ := total_lemon_heads / whole_boxes

theorem lemon_heads_in_package : lemon_heads_per_package = 6 := by
  sorry

end NUMINAMATH_CALUDE_lemon_heads_in_package_l1415_141525


namespace NUMINAMATH_CALUDE_prism_128_cubes_ratio_l1415_141574

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Checks if the given dimensions form a valid prism of 128 cubes -/
def is_valid_prism (d : PrismDimensions) : Prop :=
  d.width * d.length * d.height = 128

/-- Checks if the given dimensions have the ratio 1:1:2 -/
def has_ratio_1_1_2 (d : PrismDimensions) : Prop :=
  d.width = d.length ∧ d.height = 2 * d.width

/-- Theorem stating that a valid prism of 128 cubes has dimensions with ratio 1:1:2 -/
theorem prism_128_cubes_ratio :
  ∀ d : PrismDimensions, is_valid_prism d → has_ratio_1_1_2 d :=
by sorry

end NUMINAMATH_CALUDE_prism_128_cubes_ratio_l1415_141574


namespace NUMINAMATH_CALUDE_total_discount_calculation_l1415_141585

theorem total_discount_calculation (original_price : ℝ) (initial_discount : ℝ) (additional_discount : ℝ) :
  initial_discount = 0.5 →
  additional_discount = 0.25 →
  let sale_price := original_price * (1 - initial_discount)
  let final_price := sale_price * (1 - additional_discount)
  let total_discount := (original_price - final_price) / original_price
  total_discount = 0.625 :=
by sorry

end NUMINAMATH_CALUDE_total_discount_calculation_l1415_141585


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_l1415_141562

theorem cubic_roots_sum_of_cubes (p q : ℝ) (r s : ℂ) : 
  (r^3 - p*r^2 + q*r - p = 0) → 
  (s^3 - p*s^2 + q*s - p = 0) → 
  r^3 + s^3 = p^3 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_l1415_141562


namespace NUMINAMATH_CALUDE_emilys_glue_sticks_l1415_141524

theorem emilys_glue_sticks (total : ℕ) (sisters : ℕ) (emilys : ℕ) : 
  total = 13 → sisters = 7 → emilys = total - sisters → emilys = 6 :=
by sorry

end NUMINAMATH_CALUDE_emilys_glue_sticks_l1415_141524


namespace NUMINAMATH_CALUDE_equation_solution_l1415_141582

theorem equation_solution (x : ℝ) :
  x ≠ -1 → x ≠ 1 → (x / (x + 1) = 2 / (x^2 - 1)) → x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1415_141582


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1415_141505

theorem complex_magnitude_problem (z : ℂ) (h : z * (1 + Complex.I) = 1 - Complex.I) :
  Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1415_141505


namespace NUMINAMATH_CALUDE_division_remainder_problem_l1415_141506

theorem division_remainder_problem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 172 →
  divisor = 17 →
  quotient = 10 →
  dividend = divisor * quotient + remainder →
  remainder = 2 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l1415_141506


namespace NUMINAMATH_CALUDE_circle_equation_k_value_l1415_141522

theorem circle_equation_k_value (k : ℝ) :
  (∀ x y : ℝ, x^2 + 12*x + y^2 + 8*y - k = 0 ↔ (x + 6)^2 + (y + 4)^2 = 25) →
  k = -27 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_k_value_l1415_141522


namespace NUMINAMATH_CALUDE_special_arithmetic_sequence_general_term_l1415_141541

/-- An arithmetic sequence with a1 = 4 and a1, a5, a13 forming a geometric sequence -/
structure SpecialArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d
  a1_eq_4 : a 1 = 4
  geometric_subsequence : ∃ r : ℝ, a 5 = a 1 * r ∧ a 13 = a 5 * r

/-- The general term formula for the special arithmetic sequence -/
def general_term (seq : SpecialArithmeticSequence) (n : ℕ) : ℝ :=
  n + 3

theorem special_arithmetic_sequence_general_term (seq : SpecialArithmeticSequence) :
  ∀ n : ℕ, seq.a n = general_term seq n ∨ seq.a n = 4 := by
  sorry

end NUMINAMATH_CALUDE_special_arithmetic_sequence_general_term_l1415_141541


namespace NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l1415_141566

def M : Set ℝ := {1, 2}
def N (a : ℝ) : Set ℝ := {a^2}

theorem a_eq_one_sufficient_not_necessary :
  (∀ a : ℝ, a = 1 → N a ⊆ M) ∧
  (∃ a : ℝ, a ≠ 1 ∧ N a ⊆ M) :=
by sorry

end NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l1415_141566


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1415_141543

theorem simplify_and_evaluate (a b : ℝ) (ha : a = Real.sqrt 3 - Real.sqrt 11) (hb : b = Real.sqrt 3 + Real.sqrt 11) :
  (a^2 - b^2) / (a^2 * b - a * b^2) / (1 + (a^2 + b^2) / (2 * a * b)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1415_141543


namespace NUMINAMATH_CALUDE_opposite_of_negative_nine_l1415_141509

theorem opposite_of_negative_nine :
  ∃ x : ℤ, x + (-9) = 0 ∧ x = 9 :=
sorry

end NUMINAMATH_CALUDE_opposite_of_negative_nine_l1415_141509


namespace NUMINAMATH_CALUDE_inequality_holds_iff_k_in_range_l1415_141548

theorem inequality_holds_iff_k_in_range (k : ℝ) : 
  (k > 0 ∧ 
   ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ + x₂ = k → 
   (1/x₁ - x₁) * (1/x₂ - x₂) ≥ (k/2 - 2/k)^2) 
  ↔ 
  (0 < k ∧ k ≤ 2 * Real.sqrt (Real.sqrt 5 - 2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_k_in_range_l1415_141548


namespace NUMINAMATH_CALUDE_f_value_at_one_l1415_141510

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function g : ℝ → ℝ is odd if g(-x) = -g(x) for all x -/
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem f_value_at_one
  (f g : ℝ → ℝ)
  (h_even : IsEven f)
  (h_odd : IsOdd g)
  (h_eq : ∀ x, f x - g x = x^2 - x + 1) :
  f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_one_l1415_141510


namespace NUMINAMATH_CALUDE_sqrt_225_equals_15_l1415_141561

theorem sqrt_225_equals_15 : Real.sqrt 225 = 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_225_equals_15_l1415_141561


namespace NUMINAMATH_CALUDE_gcd_of_45_135_225_l1415_141578

theorem gcd_of_45_135_225 : Nat.gcd 45 (Nat.gcd 135 225) = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_45_135_225_l1415_141578


namespace NUMINAMATH_CALUDE_bottle_caps_added_l1415_141567

theorem bottle_caps_added (initial_caps : ℕ) (final_caps : ℕ) (added_caps : ℕ) : 
  initial_caps = 7 → final_caps = 14 → added_caps = final_caps - initial_caps → added_caps = 7 :=
by sorry

end NUMINAMATH_CALUDE_bottle_caps_added_l1415_141567


namespace NUMINAMATH_CALUDE_annual_concert_ticket_sales_l1415_141550

theorem annual_concert_ticket_sales 
  (total_tickets : ℕ) 
  (student_price non_student_price : ℚ) 
  (total_revenue : ℚ) 
  (h1 : total_tickets = 150)
  (h2 : student_price = 5)
  (h3 : non_student_price = 8)
  (h4 : total_revenue = 930) :
  ∃ (student_tickets : ℕ), 
    student_tickets = 90 ∧ 
    ∃ (non_student_tickets : ℕ), 
      student_tickets + non_student_tickets = total_tickets ∧
      student_price * student_tickets + non_student_price * non_student_tickets = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_annual_concert_ticket_sales_l1415_141550


namespace NUMINAMATH_CALUDE_range_of_m_l1415_141579

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, Real.exp (|2*x + 1|) + m ≥ 0) ↔ m ≥ -1 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l1415_141579


namespace NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l1415_141503

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence with the given properties,
    the 10th term is 66. -/
theorem arithmetic_sequence_10th_term
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_5th_term : a 5 = 26)
  (h_8th_term : a 8 = 50) :
  a 10 = 66 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l1415_141503


namespace NUMINAMATH_CALUDE_password_selection_rule_probability_of_A_in_seventh_week_l1415_141595

/-- Represents the probability of password A being used in week k -/
def P (k : ℕ) : ℚ :=
  if k = 1 then 1
  else (3/4) * (-1/3)^(k-2) + 1/4

/-- The condition that the password for each week is chosen randomly from
    the three not used in the previous week -/
theorem password_selection_rule (k : ℕ) :
  k > 1 → P k = (1/3) * (1 - P (k-1)) :=
sorry

theorem probability_of_A_in_seventh_week :
  P 7 = 61/243 :=
sorry

end NUMINAMATH_CALUDE_password_selection_rule_probability_of_A_in_seventh_week_l1415_141595


namespace NUMINAMATH_CALUDE_diamond_four_three_l1415_141500

-- Define the diamond operation
def diamond (a b : ℝ) : ℝ := 4 * a + 3 * b - 2 * a * b

-- Theorem statement
theorem diamond_four_three : diamond 4 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_diamond_four_three_l1415_141500


namespace NUMINAMATH_CALUDE_alice_lost_second_game_l1415_141534

/-- Represents a participant in the arm-wrestling contest -/
inductive Participant
  | Alice
  | Belle
  | Cathy

/-- Represents the state of a participant in a game -/
inductive GameState
  | Playing
  | Resting

/-- Represents the result of a game for a participant -/
inductive GameResult
  | Win
  | Lose

/-- The total number of games played -/
def totalGames : Nat := 21

/-- The number of times each participant played -/
def timesPlayed (p : Participant) : Nat :=
  match p with
  | Participant.Alice => 10
  | Participant.Belle => 15
  | Participant.Cathy => 17

/-- The state of a participant in a specific game -/
def participantState (p : Participant) (gameNumber : Nat) : GameState := sorry

/-- The result of a game for a participant -/
def gameResult (p : Participant) (gameNumber : Nat) : Option GameResult := sorry

theorem alice_lost_second_game :
  gameResult Participant.Alice 2 = some GameResult.Lose := by sorry

end NUMINAMATH_CALUDE_alice_lost_second_game_l1415_141534


namespace NUMINAMATH_CALUDE_payback_time_l1415_141593

def initial_cost : ℝ := 25000
def monthly_revenue : ℝ := 4000
def monthly_expenses : ℝ := 1500

theorem payback_time :
  let monthly_profit := monthly_revenue - monthly_expenses
  (initial_cost / monthly_profit : ℝ) = 10 := by sorry

end NUMINAMATH_CALUDE_payback_time_l1415_141593


namespace NUMINAMATH_CALUDE_gift_items_solution_l1415_141594

theorem gift_items_solution :
  ∃ (x y z : ℕ) (x' y' z' : ℕ),
    x + y + z = 20 ∧
    60 * x + 50 * y + 10 * z = 720 ∧
    x' + y' + z' = 20 ∧
    60 * x' + 50 * y' + 10 * z' = 720 ∧
    ((x = 4 ∧ y = 8 ∧ z = 8) ∨ (x = 8 ∧ y = 3 ∧ z = 9)) ∧
    ((x' = 4 ∧ y' = 8 ∧ z' = 8) ∨ (x' = 8 ∧ y' = 3 ∧ z' = 9)) ∧
    ¬(x = x' ∧ y = y' ∧ z = z') :=
by
  sorry

#check gift_items_solution

end NUMINAMATH_CALUDE_gift_items_solution_l1415_141594


namespace NUMINAMATH_CALUDE_area_between_circles_l1415_141572

theorem area_between_circles (r : ℝ) (R : ℝ) : 
  r = 3 →                   -- radius of smaller circle
  R = 3 * r →               -- radius of larger circle is three times the smaller
  π * R^2 - π * r^2 = 72*π  -- area between circles is 72π
  := by sorry

end NUMINAMATH_CALUDE_area_between_circles_l1415_141572


namespace NUMINAMATH_CALUDE_coefficient_sum_equals_eight_l1415_141591

-- Define the function f
def f (x : ℝ) : ℝ := (2 * x - 1) ^ 4

-- Define the coefficients a₀, a₁, a₂, a₃, a₄
variables (a₀ a₁ a₂ a₃ a₄ : ℝ)

-- State the theorem
theorem coefficient_sum_equals_eight :
  (∀ x, f x = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4) →
  a₁ + 2 * a₂ + 3 * a₃ + 4 * a₄ = 8 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_sum_equals_eight_l1415_141591


namespace NUMINAMATH_CALUDE_y_order_l1415_141588

/-- Quadratic function f(x) = -x² + 4x - 5 -/
def f (x : ℝ) : ℝ := -x^2 + 4*x - 5

/-- Given three points on the graph of f -/
def A : ℝ × ℝ := (-4, f (-4))
def B : ℝ × ℝ := (-3, f (-3))
def C : ℝ × ℝ := (1, f 1)

/-- y-coordinates of the points -/
def y₁ : ℝ := A.2
def y₂ : ℝ := B.2
def y₃ : ℝ := C.2

theorem y_order : y₁ < y₂ ∧ y₂ < y₃ := by
  sorry

end NUMINAMATH_CALUDE_y_order_l1415_141588


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1415_141564

theorem complex_fraction_simplification :
  (5 - 7 * Complex.I) / (2 - 3 * Complex.I) = 31/13 + (1/13) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1415_141564


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_and_parallel_line_l1415_141514

structure Plane where

structure Line where

def perpendicular (l : Line) (p : Plane) : Prop := sorry

def parallel (l : Line) (p : Plane) : Prop := sorry

def perpendicular_lines (l1 l2 : Line) : Prop := sorry

theorem line_perpendicular_to_plane_and_parallel_line 
  (α : Plane) (m n : Line) 
  (h1 : perpendicular m α) 
  (h2 : parallel n α) : 
  perpendicular_lines m n := by sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_and_parallel_line_l1415_141514


namespace NUMINAMATH_CALUDE_emerson_rowing_distance_l1415_141598

/-- The total distance covered by Emerson on his rowing trip -/
def total_distance (initial_distance : ℕ) (second_segment : ℕ) (final_segment : ℕ) : ℕ :=
  initial_distance + second_segment + final_segment

/-- Theorem stating that Emerson's total distance is 39 miles -/
theorem emerson_rowing_distance :
  total_distance 6 15 18 = 39 := by
  sorry

end NUMINAMATH_CALUDE_emerson_rowing_distance_l1415_141598


namespace NUMINAMATH_CALUDE_blue_paint_calculation_l1415_141551

/-- Given a paint mixture with a ratio of blue to green paint and a total number of cans,
    calculate the number of cans of blue paint required. -/
def blue_paint_cans (blue_ratio green_ratio total_cans : ℕ) : ℕ :=
  (blue_ratio * total_cans) / (blue_ratio + green_ratio)

/-- Theorem stating that for a 4:3 ratio of blue to green paint and 42 total cans,
    24 cans of blue paint are required. -/
theorem blue_paint_calculation :
  blue_paint_cans 4 3 42 = 24 := by
  sorry

end NUMINAMATH_CALUDE_blue_paint_calculation_l1415_141551


namespace NUMINAMATH_CALUDE_math_reading_homework_difference_l1415_141563

theorem math_reading_homework_difference (reading_pages math_pages : ℕ) 
  (h1 : reading_pages = 12) 
  (h2 : math_pages = 23) : 
  math_pages - reading_pages = 11 := by
  sorry

end NUMINAMATH_CALUDE_math_reading_homework_difference_l1415_141563


namespace NUMINAMATH_CALUDE_range_of_m_l1415_141569

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 4*x - 2

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ∈ Set.Icc (-6) (-2)) ∧
  (∀ y ∈ Set.Icc (-6) (-2), ∃ x ∈ Set.Icc 0 m, f x = y) →
  m ∈ Set.Icc 2 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1415_141569


namespace NUMINAMATH_CALUDE_largest_whole_number_nine_times_less_than_150_l1415_141568

theorem largest_whole_number_nine_times_less_than_150 :
  ∃ (x : ℤ), x = 16 ∧ (∀ y : ℤ, 9 * y < 150 → y ≤ x) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_whole_number_nine_times_less_than_150_l1415_141568


namespace NUMINAMATH_CALUDE_sum_of_four_digit_odd_and_multiples_of_ten_l1415_141575

/-- The number of four-digit odd numbers -/
def A : ℕ := 4500

/-- The number of four-digit multiples of 10 -/
def B : ℕ := 900

/-- The sum of four-digit odd numbers and four-digit multiples of 10 is 5400 -/
theorem sum_of_four_digit_odd_and_multiples_of_ten : A + B = 5400 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_digit_odd_and_multiples_of_ten_l1415_141575


namespace NUMINAMATH_CALUDE_cos_180_degrees_l1415_141535

theorem cos_180_degrees : Real.cos (π) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_180_degrees_l1415_141535


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l1415_141573

-- Define the slopes of the two lines
def slope1 : ℚ := -1/5
def slope2 (b : ℚ) : ℚ := -b/4

-- Define the perpendicularity condition
def perpendicular (b : ℚ) : Prop := slope1 * slope2 b = -1

-- Theorem statement
theorem perpendicular_lines_b_value : 
  ∀ b : ℚ, perpendicular b → b = -20 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l1415_141573


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l1415_141538

/-- Given a line with slope -5 passing through (4, 2), prove that m + b = 17 in y = mx + b -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = -5 → 
  2 = m * 4 + b → 
  m + b = 17 := by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l1415_141538


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l1415_141560

/-- Given a rectangle with perimeter 72 meters and length-to-width ratio of 3:2,
    prove that its diagonal length is √673.92 meters. -/
theorem rectangle_diagonal (length width : ℝ) : 
  (2 * (length + width) = 72) →
  (length / width = 3 / 2) →
  Real.sqrt (length^2 + width^2) = Real.sqrt 673.92 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l1415_141560


namespace NUMINAMATH_CALUDE_goat_price_problem_l1415_141576

theorem goat_price_problem (total_cost num_cows num_goats cow_price : ℕ) 
  (h1 : total_cost = 1500)
  (h2 : num_cows = 2)
  (h3 : num_goats = 10)
  (h4 : cow_price = 400) :
  (total_cost - num_cows * cow_price) / num_goats = 70 := by
  sorry

end NUMINAMATH_CALUDE_goat_price_problem_l1415_141576


namespace NUMINAMATH_CALUDE_max_sum_consecutive_integers_with_product_constraint_l1415_141533

theorem max_sum_consecutive_integers_with_product_constraint : 
  ∀ n : ℕ, n * (n + 1) < 500 → n + (n + 1) ≤ 43 :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_consecutive_integers_with_product_constraint_l1415_141533


namespace NUMINAMATH_CALUDE_train_crossing_pole_time_l1415_141512

/-- Proves that a train with a given length and speed takes a specific time to cross a pole -/
theorem train_crossing_pole_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 250 →
  train_speed_kmh = 100 →
  crossing_time = 90 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) :=
by
  sorry

#check train_crossing_pole_time

end NUMINAMATH_CALUDE_train_crossing_pole_time_l1415_141512


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l1415_141502

theorem largest_prime_factor_of_expression :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (9^3 + 8^5 - 4^5) ∧
  ∀ (q : ℕ), Nat.Prime q → q ∣ (9^3 + 8^5 - 4^5) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l1415_141502


namespace NUMINAMATH_CALUDE_problem_statement_l1415_141508

theorem problem_statement (x₁ x₂ x₃ x₄ n : ℝ) 
  (h1 : x₁ ≠ x₂)
  (h2 : (x₁ + x₃) * (x₁ + x₄) = n - 10)
  (h3 : (x₂ + x₃) * (x₂ + x₄) = n - 10)
  (h4 : x₁ + x₂ + x₃ + x₄ = 0) :
  let p := (x₁ + x₃) * (x₂ + x₃) + (x₁ + x₄) * (x₂ + x₄)
  p = 2 * n - 20 := by
  sorry


end NUMINAMATH_CALUDE_problem_statement_l1415_141508


namespace NUMINAMATH_CALUDE_ninth_term_of_specific_arithmetic_sequence_l1415_141530

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) : ℕ → ℚ :=
  λ n => a₁ + (n - 1 : ℚ) * d

theorem ninth_term_of_specific_arithmetic_sequence :
  ∃ (d : ℚ), 
    let seq := arithmetic_sequence (3/4) d
    seq 1 = 3/4 ∧ seq 17 = 1/2 ∧ seq 9 = 5/8 :=
by
  sorry

end NUMINAMATH_CALUDE_ninth_term_of_specific_arithmetic_sequence_l1415_141530


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l1415_141590

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (3*x - 1)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 4^7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l1415_141590


namespace NUMINAMATH_CALUDE_probability_four_blue_marbles_l1415_141554

def total_marbles : ℕ := 20
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 12
def num_trials : ℕ := 8
def num_blue_picked : ℕ := 4

theorem probability_four_blue_marbles :
  (Nat.choose num_trials num_blue_picked) *
  (blue_marbles / total_marbles : ℚ) ^ num_blue_picked *
  (red_marbles / total_marbles : ℚ) ^ (num_trials - num_blue_picked) =
  90720 / 390625 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_blue_marbles_l1415_141554


namespace NUMINAMATH_CALUDE_cylinder_in_hemisphere_height_l1415_141544

theorem cylinder_in_hemisphere_height (r c h : ℝ) : 
  r > 0 → c > 0 → h > 0 →
  r = 7 → c = 3 →
  h^2 = r^2 - c^2 →
  h = Real.sqrt 40 := by
sorry

end NUMINAMATH_CALUDE_cylinder_in_hemisphere_height_l1415_141544


namespace NUMINAMATH_CALUDE_paint_room_time_l1415_141558

/-- The time (in hours) it takes Alice to paint the room alone -/
def alice_time : ℝ := 3

/-- The time (in hours) it takes Bob to paint the room alone -/
def bob_time : ℝ := 6

/-- The duration (in hours) of the break Alice and Bob take -/
def break_time : ℝ := 2

/-- The total time (in hours) it takes Alice and Bob to paint the room together, including the break -/
def total_time : ℝ := 4

theorem paint_room_time :
  (1 / alice_time + 1 / bob_time) * (total_time - break_time) = 1 :=
sorry

end NUMINAMATH_CALUDE_paint_room_time_l1415_141558


namespace NUMINAMATH_CALUDE_sin_sixty_degrees_l1415_141528

theorem sin_sixty_degrees : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sixty_degrees_l1415_141528


namespace NUMINAMATH_CALUDE_fuel_tank_capacity_l1415_141549

/-- The capacity of a fuel tank given specific conditions -/
theorem fuel_tank_capacity : ∃ (C : ℝ), 
  (0.12 * 82 + 0.16 * (C - 82) = 30) ∧ 
  (C = 208) := by
  sorry

end NUMINAMATH_CALUDE_fuel_tank_capacity_l1415_141549


namespace NUMINAMATH_CALUDE_crosswalk_stripe_distance_l1415_141557

theorem crosswalk_stripe_distance
  (curb_length : ℝ)
  (street_width : ℝ)
  (stripe_length : ℝ)
  (h_curb : curb_length = 25)
  (h_width : street_width = 60)
  (h_stripe : stripe_length = 50) :
  curb_length * street_width / stripe_length = 30 := by
sorry

end NUMINAMATH_CALUDE_crosswalk_stripe_distance_l1415_141557


namespace NUMINAMATH_CALUDE_range_of_a_l1415_141546

theorem range_of_a (a : ℝ) : 
  (∀ b : ℝ, ∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ |x^2 + a*x + b| ≥ 1) ↔ 
  (a ≥ 1 ∨ a ≤ -3) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1415_141546


namespace NUMINAMATH_CALUDE_min_value_expression_l1415_141517

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (8 * z) / (3 * x + 2 * y) + (8 * x) / (2 * y + 3 * z) + y / (x + z) ≥ 4.5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1415_141517


namespace NUMINAMATH_CALUDE_population_ratio_l1415_141536

theorem population_ratio (x y z : ℕ) (hxy : x = 6 * y) (hyz : y = 2 * z) : x / z = 12 := by
  sorry

end NUMINAMATH_CALUDE_population_ratio_l1415_141536


namespace NUMINAMATH_CALUDE_babylonian_square_58_l1415_141589

-- Define the pattern function
def babylonian_square (n : Nat) : Nat × Nat :=
  let square := n * n
  let quotient := square / 60
  let remainder := square % 60
  if remainder = 0 then (quotient - 1, 60) else (quotient, remainder)

-- Theorem statement
theorem babylonian_square_58 : babylonian_square 58 = (56, 4) := by
  sorry

end NUMINAMATH_CALUDE_babylonian_square_58_l1415_141589


namespace NUMINAMATH_CALUDE_inequality_implies_upper_bound_l1415_141552

theorem inequality_implies_upper_bound (a : ℝ) :
  (∀ x : ℝ, |x + 2| + |x - 3| > a) → a < 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_upper_bound_l1415_141552


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1415_141556

def polynomial (x : ℝ) : ℝ :=
  -3 * (x^8 - 2*x^5 + 4*x^3 - 6) + 5 * (2*x^4 + 3*x^2 - x) - 2 * (3*x^6 - 7)

theorem sum_of_coefficients : 
  (polynomial 1) = 37 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1415_141556


namespace NUMINAMATH_CALUDE_combination_equality_l1415_141513

theorem combination_equality (x : ℕ) : (Nat.choose 5 3 + Nat.choose 5 4 = Nat.choose x 4) ↔ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_l1415_141513


namespace NUMINAMATH_CALUDE_minimum_distance_between_curves_l1415_141559

noncomputable def min_distance : ℝ := Real.sqrt 2 / 2 * (1 - Real.log 2)

theorem minimum_distance_between_curves :
  ∃ (a b : ℝ),
    (1/2 : ℝ) * Real.exp a = (1/2 : ℝ) * Real.exp a ∧
    b = b ∧
    ∀ (x y : ℝ),
      (1/2 : ℝ) * Real.exp x = (1/2 : ℝ) * Real.exp x →
      y = y →
      Real.sqrt ((x - y)^2 + ((1/2 : ℝ) * Real.exp x - y)^2) ≥ min_distance :=
by sorry

end NUMINAMATH_CALUDE_minimum_distance_between_curves_l1415_141559


namespace NUMINAMATH_CALUDE_class_size_l1415_141599

/-- The number of students in a class, given information about their sports participation -/
theorem class_size (football : ℕ) (tennis : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : football = 26)
  (h2 : tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 11) :
  football + tennis - both + neither = 40 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l1415_141599


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1415_141584

theorem fraction_evaluation (a b c : ℝ) (ha : a = 2) (hb : b = 3) (hc : c = 1) :
  6 / (a + b + c) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1415_141584


namespace NUMINAMATH_CALUDE_digimon_pack_cost_is_445_l1415_141532

/-- The cost of a pack of Digimon cards -/
def digimon_pack_cost : ℝ := 4.45

/-- The number of Digimon card packs bought -/
def num_digimon_packs : ℕ := 4

/-- The cost of the baseball card deck -/
def baseball_deck_cost : ℝ := 6.06

/-- The total amount spent on cards -/
def total_spent : ℝ := 23.86

/-- Theorem stating that the cost of each Digimon card pack is $4.45 -/
theorem digimon_pack_cost_is_445 :
  digimon_pack_cost * num_digimon_packs + baseball_deck_cost = total_spent :=
by sorry

end NUMINAMATH_CALUDE_digimon_pack_cost_is_445_l1415_141532


namespace NUMINAMATH_CALUDE_polynomial_division_l1415_141596

-- Define the polynomials
def P (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 + 8 * x - 1
def D (x : ℝ) : ℝ := x - 3
def Q (x : ℝ) : ℝ := 3 * x^2 + 8
def R : ℝ := 23

-- State the theorem
theorem polynomial_division :
  ∀ x : ℝ, P x = D x * Q x + R := by sorry

end NUMINAMATH_CALUDE_polynomial_division_l1415_141596


namespace NUMINAMATH_CALUDE_nonagon_diagonals_octagon_diagonals_decagon_diagonals_l1415_141542

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a nonagon is 27 -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by sorry

/-- Theorem: The number of diagonals in an octagon is 20 -/
theorem octagon_diagonals : num_diagonals 8 = 20 := by sorry

/-- Theorem: The number of diagonals in a decagon is 35 -/
theorem decagon_diagonals : num_diagonals 10 = 35 := by sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_octagon_diagonals_decagon_diagonals_l1415_141542


namespace NUMINAMATH_CALUDE_quadratic_roots_bound_l1415_141515

theorem quadratic_roots_bound (a b c : ℝ) (x₁ x₂ : ℝ) (ha : a > 0) :
  let P : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (P x₁ = 0 ∧ P x₂ = 0) →
  (abs x₁ ≤ 1 ∧ abs x₂ ≤ 1) ↔ (a + b + c ≥ 0 ∧ a - b + c ≥ 0 ∧ a - c ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_bound_l1415_141515


namespace NUMINAMATH_CALUDE_parabola_properties_l1415_141555

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

-- Define the focus F
def focus : ℝ × ℝ := (4, 0)

-- Define a point M on the parabola
def point_on_parabola (M : ℝ × ℝ) : Prop :=
  parabola M.1 M.2

-- Define N as a point on the y-axis
def point_on_y_axis (N : ℝ × ℝ) : Prop :=
  N.1 = 0

-- Define M as the midpoint of FN
def is_midpoint (F M N : ℝ × ℝ) : Prop :=
  M.1 = (F.1 + N.1) / 2 ∧ M.2 = (F.2 + N.2) / 2

-- Main theorem
theorem parabola_properties (M N : ℝ × ℝ) 
  (h1 : point_on_parabola M)
  (h2 : point_on_y_axis N)
  (h3 : is_midpoint focus M N) :
  (∀ x y, y^2 = 16 * x → x = -4 → False) ∧  -- Directrix equation
  (Real.sqrt ((focus.1 - N.1)^2 + (focus.2 - N.2)^2) = 12) ∧  -- |FN| = 12
  (1/2 * focus.1 * N.2 = 16 * Real.sqrt 2) :=  -- Area of triangle ONF
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l1415_141555


namespace NUMINAMATH_CALUDE_boys_meeting_on_circular_track_l1415_141570

/-- The number of times two boys meet on a circular track -/
def number_of_meetings (speed1 speed2 : ℝ) : ℕ :=
  -- We'll define this function later
  sorry

/-- Theorem: Two boys moving in opposite directions on a circular track with speeds
    of 5 ft/s and 9 ft/s will meet 13 times before returning to the starting point -/
theorem boys_meeting_on_circular_track :
  number_of_meetings 5 9 = 13 := by
  sorry

end NUMINAMATH_CALUDE_boys_meeting_on_circular_track_l1415_141570


namespace NUMINAMATH_CALUDE_midnight_temperature_l1415_141553

def morning_temp : ℝ := 30
def afternoon_rise : ℝ := 1
def midnight_drop : ℝ := 7

theorem midnight_temperature : 
  morning_temp + afternoon_rise - midnight_drop = 24 := by
  sorry

end NUMINAMATH_CALUDE_midnight_temperature_l1415_141553


namespace NUMINAMATH_CALUDE_composite_triangle_perimeter_l1415_141518

/-- A triangle composed of four smaller equilateral triangles -/
structure CompositeTriangle where
  /-- The side length of the smaller equilateral triangles -/
  small_side : ℝ
  /-- The perimeter of each smaller equilateral triangle is 9 -/
  small_perimeter : small_side * 3 = 9

/-- The perimeter of the large equilateral triangle -/
def large_perimeter (t : CompositeTriangle) : ℝ :=
  3 * (2 * t.small_side)

/-- Theorem: The perimeter of the large equilateral triangle is 18 -/
theorem composite_triangle_perimeter (t : CompositeTriangle) :
  large_perimeter t = 18 := by
  sorry

end NUMINAMATH_CALUDE_composite_triangle_perimeter_l1415_141518


namespace NUMINAMATH_CALUDE_max_distance_from_circle_to_point_l1415_141540

theorem max_distance_from_circle_to_point (z : ℂ) :
  Complex.abs z = 2 → (⨆ z, Complex.abs (z - Complex.I)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_from_circle_to_point_l1415_141540


namespace NUMINAMATH_CALUDE_min_distance_to_line_l1415_141511

theorem min_distance_to_line (x y : ℝ) : 
  (x + 2)^2 + (y - 3)^2 = 1 → 
  ∃ (min : ℝ), min = 15 ∧ ∀ (a b : ℝ), (a + 2)^2 + (b - 3)^2 = 1 → |3*a + 4*b - 26| ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l1415_141511


namespace NUMINAMATH_CALUDE_divisor_power_difference_l1415_141597

theorem divisor_power_difference (k : ℕ) : 
  (15 ^ k : ℕ) ∣ 759325 → 3 ^ k - k ^ 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_divisor_power_difference_l1415_141597


namespace NUMINAMATH_CALUDE_range_of_a_l1415_141516

-- Define the conditions
def condition_p (a : ℝ) : Prop := ∃ m : ℝ, m ∈ Set.Icc (-1) 1 ∧ a^2 - 5*a + 5 ≥ m + 2

def condition_q (a : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + a*x₁ + 2 = 0 ∧ x₂^2 + a*x₂ + 2 = 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (condition_p a ∨ condition_q a) ∧ ¬(condition_p a ∧ condition_q a) →
  a ≤ 1 ∨ (2 * Real.sqrt 2 ≤ a ∧ a < 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1415_141516


namespace NUMINAMATH_CALUDE_watch_ahead_by_16_minutes_l1415_141507

/-- Represents the time gain of a watch in minutes per hour -/
def time_gain : ℕ := 4

/-- Represents the start time in minutes after midnight -/
def start_time : ℕ := 10 * 60

/-- Represents the event time in minutes after midnight -/
def event_time : ℕ := 14 * 60

/-- Calculates the actual time passed given the time shown on the watch -/
def actual_time (watch_time : ℕ) : ℕ :=
  (watch_time * 60) / (60 + time_gain)

/-- Theorem stating that the watch shows 16 minutes ahead of the actual time -/
theorem watch_ahead_by_16_minutes :
  actual_time (event_time - start_time) = event_time - start_time - 16 := by
  sorry


end NUMINAMATH_CALUDE_watch_ahead_by_16_minutes_l1415_141507


namespace NUMINAMATH_CALUDE_tricycle_count_l1415_141580

theorem tricycle_count (total_children : ℕ) (total_wheels : ℕ) 
  (h1 : total_children = 12) 
  (h2 : total_wheels = 32) : ∃ (bicycles tricycles : ℕ), 
  bicycles + tricycles = total_children ∧ 
  2 * bicycles + 3 * tricycles = total_wheels ∧ 
  tricycles = 8 := by
sorry

end NUMINAMATH_CALUDE_tricycle_count_l1415_141580


namespace NUMINAMATH_CALUDE_number_equation_solution_l1415_141529

theorem number_equation_solution :
  ∃ x : ℝ, (3 * x = 2 * x - 7) ∧ (x = -7) := by sorry

end NUMINAMATH_CALUDE_number_equation_solution_l1415_141529


namespace NUMINAMATH_CALUDE_jim_initial_tree_rows_l1415_141531

/-- Proves that Jim started with 2 rows of trees given the problem conditions -/
theorem jim_initial_tree_rows : ∀ (initial_rows : ℕ), 
  (∀ (row : ℕ), row > 0 → row ≤ initial_rows + 5 → 4 * row ≤ 56) ∧
  (2 * (4 * (initial_rows + 5)) = 56) →
  initial_rows = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_jim_initial_tree_rows_l1415_141531


namespace NUMINAMATH_CALUDE_complex_number_location_l1415_141592

theorem complex_number_location (z : ℂ) (h : z = Complex.I * (1 + Complex.I)) :
  Complex.re z < 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l1415_141592


namespace NUMINAMATH_CALUDE_plane_sphere_intersection_l1415_141545

/-- Given a plane passing through (d,e,f) and intersecting the coordinate axes at D, E, F,
    with (u,v,w) as the center of the sphere through D, E, F, and the origin,
    prove that d/u + e/v + f/w = 2 -/
theorem plane_sphere_intersection (d e f u v w : ℝ) : 
  (∃ (δ ε ϕ : ℝ), 
    δ ≠ 0 ∧ ε ≠ 0 ∧ ϕ ≠ 0 ∧
    u^2 + v^2 + w^2 = (u - δ)^2 + v^2 + w^2 ∧
    u^2 + v^2 + w^2 = u^2 + (v - ε)^2 + w^2 ∧
    u^2 + v^2 + w^2 = u^2 + v^2 + (w - ϕ)^2 ∧
    d / δ + e / ε + f / ϕ = 1) →
  d / u + e / v + f / w = 2 :=
by sorry

end NUMINAMATH_CALUDE_plane_sphere_intersection_l1415_141545


namespace NUMINAMATH_CALUDE_point_six_units_from_negative_three_l1415_141527

theorem point_six_units_from_negative_three (x : ℝ) : 
  (|x - (-3)| = 6) ↔ (x = 3 ∨ x = -9) := by sorry

end NUMINAMATH_CALUDE_point_six_units_from_negative_three_l1415_141527


namespace NUMINAMATH_CALUDE_polygon_angle_sum_l1415_141523

theorem polygon_angle_sum (n : ℕ) (x : ℝ) : 
  n ≥ 3 → 
  0 < x → 
  x < 180 → 
  (n - 2) * 180 + x = 1350 → 
  n = 9 ∧ x = 90 := by
sorry

end NUMINAMATH_CALUDE_polygon_angle_sum_l1415_141523


namespace NUMINAMATH_CALUDE_trig_expression_equals_three_halves_l1415_141587

theorem trig_expression_equals_three_halves :
  (Real.sin (30 * π / 180) - 1) ^ 0 - Real.sqrt 2 * Real.sin (45 * π / 180) +
  Real.tan (60 * π / 180) * Real.cos (30 * π / 180) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_three_halves_l1415_141587


namespace NUMINAMATH_CALUDE_circular_garden_radius_l1415_141519

theorem circular_garden_radius (r : ℝ) (h : r > 0) : 2 * π * r = (1/3) * π * r^2 → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_radius_l1415_141519


namespace NUMINAMATH_CALUDE_addition_subtraction_elimination_not_factorization_l1415_141581

/-- Represents a mathematical method --/
inductive Method
  | TakingOutCommonFactor
  | CrossMultiplication
  | Formula
  | AdditionSubtractionElimination

/-- Predicate to determine if a method is a factorization method --/
def IsFactorizationMethod (m : Method) : Prop :=
  m = Method.TakingOutCommonFactor ∨ 
  m = Method.CrossMultiplication ∨ 
  m = Method.Formula

theorem addition_subtraction_elimination_not_factorization :
  ¬(IsFactorizationMethod Method.AdditionSubtractionElimination) :=
by sorry

end NUMINAMATH_CALUDE_addition_subtraction_elimination_not_factorization_l1415_141581


namespace NUMINAMATH_CALUDE_average_correction_l1415_141520

def correct_average (num_students : ℕ) (initial_average : ℚ) (wrong_mark : ℚ) (correct_mark : ℚ) : ℚ :=
  (num_students * initial_average - (wrong_mark - correct_mark)) / num_students

theorem average_correction (num_students : ℕ) (initial_average : ℚ) (wrong_mark : ℚ) (correct_mark : ℚ)
  (h1 : num_students = 30)
  (h2 : initial_average = 60)
  (h3 : wrong_mark = 90)
  (h4 : correct_mark = 15) :
  correct_average num_students initial_average wrong_mark correct_mark = 57.5 := by
sorry

end NUMINAMATH_CALUDE_average_correction_l1415_141520


namespace NUMINAMATH_CALUDE_jesse_book_reading_l1415_141586

theorem jesse_book_reading (total_pages : ℕ) (pages_read : ℕ) (pages_left : ℕ) : 
  pages_left = 166 → 
  pages_read = total_pages / 3 → 
  pages_left = 2 * total_pages / 3 → 
  pages_read = 83 := by
sorry

end NUMINAMATH_CALUDE_jesse_book_reading_l1415_141586


namespace NUMINAMATH_CALUDE_hash_2_3_4_l1415_141526

-- Define the # operation
def hash (a b c : ℝ) : ℝ := b^2 - 4*a*c + b

-- Theorem statement
theorem hash_2_3_4 : hash 2 3 4 = -20 := by sorry

end NUMINAMATH_CALUDE_hash_2_3_4_l1415_141526


namespace NUMINAMATH_CALUDE_hockey_league_games_l1415_141521

/-- The number of teams in the hockey league -/
def num_teams : ℕ := 19

/-- The number of times each team faces every other team -/
def games_per_pair : ℕ := 10

/-- The total number of games played in the season -/
def total_games : ℕ := num_teams * (num_teams - 1) / 2 * games_per_pair

theorem hockey_league_games :
  total_games = 1710 :=
sorry

end NUMINAMATH_CALUDE_hockey_league_games_l1415_141521


namespace NUMINAMATH_CALUDE_victors_percentage_l1415_141571

/-- Calculate the percentage of marks obtained given the marks scored and maximum marks -/
def calculatePercentage (marksScored : ℕ) (maxMarks : ℕ) : ℚ :=
  (marksScored : ℚ) / (maxMarks : ℚ) * 100

/-- Theorem stating that Victor's percentage of marks is 95% -/
theorem victors_percentage :
  let marksScored : ℕ := 285
  let maxMarks : ℕ := 300
  calculatePercentage marksScored maxMarks = 95 := by
  sorry


end NUMINAMATH_CALUDE_victors_percentage_l1415_141571
