import Mathlib

namespace NUMINAMATH_CALUDE_constant_term_expansion_l4020_402001

/-- The constant term in the expansion of (x-1)(x^2- 1/x)^6 is -15 -/
theorem constant_term_expansion : ∃ (f : ℝ → ℝ), 
  (∀ x ≠ 0, f x = (x - 1) * (x^2 - 1/x)^6) ∧ 
  (∃ c : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |f x - c| < ε) ∧
  (∃ c : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |f x - c| < ε) → c = -15 :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l4020_402001


namespace NUMINAMATH_CALUDE_C_ℝP_subset_Q_l4020_402002

-- Define set P
def P : Set ℝ := {y | ∃ x : ℝ, y = -x^2 + 1}

-- Define set Q
def Q : Set ℝ := {y | ∃ x : ℝ, y = 2^x}

-- Define the complement of P in ℝ
def C_ℝP : Set ℝ := {y | y ∉ P}

-- Theorem statement
theorem C_ℝP_subset_Q : C_ℝP ⊆ Q := by
  sorry

end NUMINAMATH_CALUDE_C_ℝP_subset_Q_l4020_402002


namespace NUMINAMATH_CALUDE_remaining_amount_l4020_402054

def initial_amount : ℚ := 343
def fraction_given : ℚ := 1/7
def num_recipients : ℕ := 2

theorem remaining_amount :
  initial_amount - (fraction_given * initial_amount * num_recipients) = 245 :=
by sorry

end NUMINAMATH_CALUDE_remaining_amount_l4020_402054


namespace NUMINAMATH_CALUDE_expand_expression_l4020_402086

theorem expand_expression (x y : ℝ) : (16*x + 18 - 7*y) * 3*x = 48*x^2 + 54*x - 21*x*y := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l4020_402086


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l4020_402010

theorem simplify_sqrt_sum : 
  Real.sqrt (8 + 4 * Real.sqrt 3) + Real.sqrt (8 - 4 * Real.sqrt 3) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l4020_402010


namespace NUMINAMATH_CALUDE_enough_beverages_l4020_402067

/-- Robin's hydration plan and beverage inventory --/
structure HydrationPlan where
  water_per_day : ℕ
  juice_per_day : ℕ
  soda_per_day : ℕ
  plan_duration : ℕ
  water_inventory : ℕ
  juice_inventory : ℕ
  soda_inventory : ℕ

/-- Theorem: Robin has enough beverages for her hydration plan --/
theorem enough_beverages (plan : HydrationPlan)
  (h1 : plan.water_per_day = 9)
  (h2 : plan.juice_per_day = 5)
  (h3 : plan.soda_per_day = 3)
  (h4 : plan.plan_duration = 60)
  (h5 : plan.water_inventory = 617)
  (h6 : plan.juice_inventory = 350)
  (h7 : plan.soda_inventory = 215) :
  plan.water_inventory ≥ plan.water_per_day * plan.plan_duration ∧
  plan.juice_inventory ≥ plan.juice_per_day * plan.plan_duration ∧
  plan.soda_inventory ≥ plan.soda_per_day * plan.plan_duration :=
by
  sorry

#check enough_beverages

end NUMINAMATH_CALUDE_enough_beverages_l4020_402067


namespace NUMINAMATH_CALUDE_rope_section_length_l4020_402074

theorem rope_section_length 
  (total_length : ℝ) 
  (art_fraction : ℝ) 
  (friend_fraction : ℝ) 
  (num_sections : ℕ) :
  total_length = 50 →
  art_fraction = 1/5 →
  friend_fraction = 1/2 →
  num_sections = 10 →
  let remaining_after_art := total_length * (1 - art_fraction)
  let remaining_after_friend := remaining_after_art * (1 - friend_fraction)
  remaining_after_friend / num_sections = 2 := by
sorry

end NUMINAMATH_CALUDE_rope_section_length_l4020_402074


namespace NUMINAMATH_CALUDE_limit_at_one_l4020_402000

-- Define the function f
def f (x : ℝ) : ℝ := x

-- State the theorem
theorem limit_at_one (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ →
    |(f (1 + Δx) - f 1) / Δx - 1| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_limit_at_one_l4020_402000


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l4020_402045

theorem rectangle_perimeter (long_side : ℝ) (short_side_difference : ℝ) :
  long_side = 1 →
  short_side_difference = 2/8 →
  let short_side := long_side - short_side_difference
  2 * long_side + 2 * short_side = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l4020_402045


namespace NUMINAMATH_CALUDE_sqrt_t6_plus_t4_l4020_402018

theorem sqrt_t6_plus_t4 (t : ℝ) : Real.sqrt (t^6 + t^4) = t^2 * Real.sqrt (t^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_t6_plus_t4_l4020_402018


namespace NUMINAMATH_CALUDE_equation_solutions_l4020_402030

theorem equation_solutions (x : ℝ) : 
  1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 8 ↔ 
  x = 7 ∨ x = -2 := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l4020_402030


namespace NUMINAMATH_CALUDE_employee_pay_l4020_402007

theorem employee_pay (total : ℝ) (x y : ℝ) (h1 : total = 770) (h2 : x = 1.2 * y) (h3 : x + y = total) : y = 350 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_l4020_402007


namespace NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l4020_402037

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem tenth_term_of_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_a2 : a 2 = 2) 
  (h_a3 : a 3 = 4) : 
  a 10 = 18 := by
sorry

end NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l4020_402037


namespace NUMINAMATH_CALUDE_four_is_17th_term_terms_before_4_l4020_402077

/-- An arithmetic sequence with first term 100 and common difference -6 -/
def arithmeticSequence (n : ℕ) : ℤ := 100 - 6 * (n - 1)

/-- The position of 4 in the sequence -/
def positionOf4 : ℕ := 17

theorem four_is_17th_term :
  arithmeticSequence positionOf4 = 4 ∧ 
  ∀ k : ℕ, k < positionOf4 → arithmeticSequence k > 4 :=
sorry

theorem terms_before_4 : 
  positionOf4 - 1 = 16 :=
sorry

end NUMINAMATH_CALUDE_four_is_17th_term_terms_before_4_l4020_402077


namespace NUMINAMATH_CALUDE_simplify_square_roots_l4020_402076

theorem simplify_square_roots : 
  (Real.sqrt 392 / Real.sqrt 56) - (Real.sqrt 252 / Real.sqrt 63) = Real.sqrt 7 - 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l4020_402076


namespace NUMINAMATH_CALUDE_parallel_lines_imply_a_eq_neg_one_l4020_402021

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal -/
def parallel (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.b = l₂.a * l₁.b

theorem parallel_lines_imply_a_eq_neg_one (a : ℝ) :
  let l₁ : Line := ⟨a, 2, 6⟩
  let l₂ : Line := ⟨1, a - 1, a^2 - 1⟩
  parallel l₁ l₂ → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_imply_a_eq_neg_one_l4020_402021


namespace NUMINAMATH_CALUDE_minimum_employees_needed_l4020_402004

/-- Represents the set of employees monitoring water pollution -/
def W : Finset Nat := sorry

/-- Represents the set of employees monitoring air pollution -/
def A : Finset Nat := sorry

/-- Represents the set of employees monitoring land pollution -/
def L : Finset Nat := sorry

theorem minimum_employees_needed : 
  (Finset.card W = 95) → 
  (Finset.card A = 80) → 
  (Finset.card L = 50) → 
  (Finset.card (W ∩ A) = 30) → 
  (Finset.card (A ∩ L) = 20) → 
  (Finset.card (W ∩ L) = 15) → 
  (Finset.card (W ∩ A ∩ L) = 10) → 
  Finset.card (W ∪ A ∪ L) = 170 := by
  sorry

end NUMINAMATH_CALUDE_minimum_employees_needed_l4020_402004


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l4020_402056

/-- Given a group of 9 persons where the average weight increases by 1.5 kg
    after replacing one person with a new person weighing 78.5 kg,
    prove that the weight of the replaced person was 65 kg. -/
theorem weight_of_replaced_person
  (n : ℕ) -- number of persons in the group
  (avg_increase : ℝ) -- increase in average weight
  (new_weight : ℝ) -- weight of the new person
  (h1 : n = 9) -- there are 9 persons in the group
  (h2 : avg_increase = 1.5) -- average weight increases by 1.5 kg
  (h3 : new_weight = 78.5) -- new person weighs 78.5 kg
  : ℝ :=
by
  sorry

#check weight_of_replaced_person

end NUMINAMATH_CALUDE_weight_of_replaced_person_l4020_402056


namespace NUMINAMATH_CALUDE_three_conditions_theorem_l4020_402012

def condition1 (a b : ℕ) : Prop := (a^2 + 6*a + 8) % b = 0

def condition2 (a b : ℕ) : Prop := a^2 + a*b - 6*b^2 - 15*b - 9 = 0

def condition3 (a b : ℕ) : Prop := (a + 2*b + 2) % 4 = 0

def condition4 (a b : ℕ) : Prop := Nat.Prime (a + 6*b + 2)

def satisfiesThreeConditions (a b : ℕ) : Prop :=
  (condition1 a b ∧ condition2 a b ∧ condition3 a b) ∨
  (condition1 a b ∧ condition2 a b ∧ condition4 a b) ∨
  (condition1 a b ∧ condition3 a b ∧ condition4 a b) ∨
  (condition2 a b ∧ condition3 a b ∧ condition4 a b)

theorem three_conditions_theorem :
  ∀ a b : ℕ, satisfiesThreeConditions a b ↔ ((a = 5 ∧ b = 1) ∨ (a = 17 ∧ b = 7)) :=
sorry

end NUMINAMATH_CALUDE_three_conditions_theorem_l4020_402012


namespace NUMINAMATH_CALUDE_smallest_repeating_block_of_8_11_l4020_402079

theorem smallest_repeating_block_of_8_11 : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 → (8 * 10^k) % 11 = (8 * 10^(k + n)) % 11) ∧
  (∀ (m : ℕ), m > 0 → m < n → ∃ (k : ℕ), k > 0 ∧ (8 * 10^k) % 11 ≠ (8 * 10^(k + m)) % 11) ∧
  n = 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_repeating_block_of_8_11_l4020_402079


namespace NUMINAMATH_CALUDE_red_flower_area_is_54_total_area_red_yellow_equal_red_yellow_half_total_l4020_402050

/-- Represents a rectangular plot with flowers and grass -/
structure FlowerPlot where
  length : ℝ
  width : ℝ
  red_flower_area : ℝ
  yellow_flower_area : ℝ
  grass_area : ℝ

/-- The properties of the flower plot as described in the problem -/
def school_plot : FlowerPlot where
  length := 18
  width := 12
  red_flower_area := 54
  yellow_flower_area := 54
  grass_area := 108

/-- Theorem stating that the area of red flowers in the school plot is 54 square meters -/
theorem red_flower_area_is_54 (plot : FlowerPlot) (h1 : plot = school_plot) :
  plot.red_flower_area = 54 := by
  sorry

/-- Theorem stating that the total area of the plot is length * width -/
theorem total_area (plot : FlowerPlot) : 
  plot.length * plot.width = plot.red_flower_area + plot.yellow_flower_area + plot.grass_area := by
  sorry

/-- Theorem stating that red and yellow flower areas are equal -/
theorem red_yellow_equal (plot : FlowerPlot) :
  plot.red_flower_area = plot.yellow_flower_area := by
  sorry

/-- Theorem stating that red and yellow flowers together occupy half the total area -/
theorem red_yellow_half_total (plot : FlowerPlot) :
  plot.red_flower_area + plot.yellow_flower_area = (plot.length * plot.width) / 2 := by
  sorry

end NUMINAMATH_CALUDE_red_flower_area_is_54_total_area_red_yellow_equal_red_yellow_half_total_l4020_402050


namespace NUMINAMATH_CALUDE_significant_figures_and_precision_of_0_03020_l4020_402022

/-- Represents a decimal number with its string representation -/
structure DecimalNumber where
  representation : String
  deriving Repr

/-- Counts the number of significant figures in a decimal number -/
def countSignificantFigures (n : DecimalNumber) : Nat :=
  sorry

/-- Determines the precision of a decimal number -/
inductive Precision
  | Tenths
  | Hundredths
  | Thousandths
  | TenThousandths
  deriving Repr

def getPrecision (n : DecimalNumber) : Precision :=
  sorry

theorem significant_figures_and_precision_of_0_03020 :
  let n : DecimalNumber := { representation := "0.03020" }
  countSignificantFigures n = 4 ∧ getPrecision n = Precision.TenThousandths :=
sorry

end NUMINAMATH_CALUDE_significant_figures_and_precision_of_0_03020_l4020_402022


namespace NUMINAMATH_CALUDE_smallest_five_digit_multiple_of_18_l4020_402082

theorem smallest_five_digit_multiple_of_18 : ∃ (n : ℕ), 
  (n = 10008) ∧ 
  (∃ (k : ℕ), n = 18 * k) ∧ 
  (n ≥ 10000) ∧ 
  (∀ (m : ℕ), (∃ (j : ℕ), m = 18 * j) → m ≥ 10000 → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_multiple_of_18_l4020_402082


namespace NUMINAMATH_CALUDE_eighth_term_value_arithmetic_sequence_eighth_term_l4020_402048

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- The sum of the first six terms
  sum_first_six : ℚ
  -- The seventh term
  seventh_term : ℚ
  -- Property: The sum of the first six terms is 21
  sum_property : sum_first_six = 21
  -- Property: The seventh term is 8
  seventh_property : seventh_term = 8

/-- Theorem: The eighth term of the arithmetic sequence is 65/7 -/
theorem eighth_term_value (seq : ArithmeticSequence) : ℚ :=
  65 / 7

/-- The main theorem: Given the conditions, the eighth term is 65/7 -/
theorem arithmetic_sequence_eighth_term (seq : ArithmeticSequence) :
  eighth_term_value seq = 65 / 7 := by
  sorry


end NUMINAMATH_CALUDE_eighth_term_value_arithmetic_sequence_eighth_term_l4020_402048


namespace NUMINAMATH_CALUDE_triangle_properties_l4020_402032

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

theorem triangle_properties (t : Triangle) 
  (m : Vector2D) (n : Vector2D) (angle_mn : ℝ) (area : ℝ) :
  m.x = Real.cos (t.C / 2) ∧ 
  m.y = Real.sin (t.C / 2) ∧
  n.x = Real.cos (t.C / 2) ∧ 
  n.y = -Real.sin (t.C / 2) ∧
  angle_mn = π / 3 ∧
  t.c = 7 / 2 ∧
  area = 3 * Real.sqrt 3 / 2 →
  t.C = π / 3 ∧ t.a + t.b = 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l4020_402032


namespace NUMINAMATH_CALUDE_fourth_root_of_12960000_l4020_402053

theorem fourth_root_of_12960000 : Real.sqrt (Real.sqrt 12960000) = 60 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_12960000_l4020_402053


namespace NUMINAMATH_CALUDE_f_properties_l4020_402099

def f (x : ℝ) : ℝ := x * (x + 1) * (x - 1)

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x > 2, ∀ y > x, f y > f x) ∧
  (∃! a b c, a < b ∧ b < c ∧ f a = 0 ∧ f b = 0 ∧ f c = 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l4020_402099


namespace NUMINAMATH_CALUDE_work_completion_men_count_first_group_size_l4020_402020

theorem work_completion_men_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun first_group_days second_group_men second_group_days result =>
    first_group_days * result = second_group_men * second_group_days

theorem first_group_size :
  ∃ (m : ℕ), work_completion_men_count 80 20 40 m ∧ m = 10 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_men_count_first_group_size_l4020_402020


namespace NUMINAMATH_CALUDE_jenna_max_tanning_time_l4020_402084

/-- Represents Jenna's tanning schedule and calculates the maximum tanning time in a month. -/
def jennaTanningSchedule : ℕ :=
  let minutesPerDay : ℕ := 30
  let daysPerWeek : ℕ := 2
  let weeksFirstPeriod : ℕ := 2
  let minutesLastTwoWeeks : ℕ := 80
  
  let minutesFirstTwoWeeks := minutesPerDay * daysPerWeek * weeksFirstPeriod
  minutesFirstTwoWeeks + minutesLastTwoWeeks

/-- Proves that Jenna's maximum tanning time in a month is 200 minutes. -/
theorem jenna_max_tanning_time : jennaTanningSchedule = 200 := by
  sorry

end NUMINAMATH_CALUDE_jenna_max_tanning_time_l4020_402084


namespace NUMINAMATH_CALUDE_inequality_proof_l4020_402040

theorem inequality_proof (x y z : ℝ) : (x^2 - y^2)^2 + (z - x)^2 + (x - 1)^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4020_402040


namespace NUMINAMATH_CALUDE_unique_positive_number_l4020_402039

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x + 8 = 128 / x := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_number_l4020_402039


namespace NUMINAMATH_CALUDE_factorial_315_trailing_zeros_l4020_402089

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- The factorial of 315 ends with 77 zeros -/
theorem factorial_315_trailing_zeros :
  trailingZeros 315 = 77 := by
  sorry

end NUMINAMATH_CALUDE_factorial_315_trailing_zeros_l4020_402089


namespace NUMINAMATH_CALUDE_total_pies_l4020_402058

theorem total_pies (percent_with_forks : ℝ) (pies_without_forks : ℕ) : 
  percent_with_forks = 0.68 →
  pies_without_forks = 640 →
  ∃ (total_pies : ℕ), 
    (1 - percent_with_forks) * (total_pies : ℝ) = pies_without_forks ∧
    total_pies = 2000 :=
by sorry

end NUMINAMATH_CALUDE_total_pies_l4020_402058


namespace NUMINAMATH_CALUDE_a_10_equals_1023_l4020_402025

def sequence_a : ℕ → ℕ
  | 0 => 1
  | n + 1 => sequence_a n + 2^(n + 1)

theorem a_10_equals_1023 : sequence_a 9 = 1023 := by
  sorry

end NUMINAMATH_CALUDE_a_10_equals_1023_l4020_402025


namespace NUMINAMATH_CALUDE_range_of_a_l4020_402057

theorem range_of_a (a : ℝ) : 
  a < 9 * a^3 - 11 * a ∧ 9 * a^3 - 11 * a < |a| → 
  a ∈ Set.Ioo (-2 * Real.sqrt 3 / 3) (-Real.sqrt 10 / 3) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l4020_402057


namespace NUMINAMATH_CALUDE_group_collection_l4020_402081

/-- Calculates the total amount collected in rupees given the number of students in a group,
    where each student contributes as many paise as there are members. -/
def totalCollected (numStudents : ℕ) : ℚ :=
  (numStudents * numStudents : ℚ) / 100

/-- Theorem stating that for a group of 96 students, the total amount collected is 92.16 rupees. -/
theorem group_collection :
  totalCollected 96 = 92.16 := by
  sorry

end NUMINAMATH_CALUDE_group_collection_l4020_402081


namespace NUMINAMATH_CALUDE_f_one_eq_zero_f_neg_one_eq_zero_f_is_odd_l4020_402087

-- Define a non-zero function f from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the functional equation
axiom func_eq : ∀ x y : ℝ, f (x * y) = y * f x + x * f y

-- Define f as a non-zero function
axiom f_nonzero : ∃ x : ℝ, f x ≠ 0

-- Theorem 1: f(1) = 0
theorem f_one_eq_zero : f 1 = 0 := by sorry

-- Theorem 2: f(-1) = 0
theorem f_neg_one_eq_zero : f (-1) = 0 := by sorry

-- Theorem 3: f is an odd function
theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

end NUMINAMATH_CALUDE_f_one_eq_zero_f_neg_one_eq_zero_f_is_odd_l4020_402087


namespace NUMINAMATH_CALUDE_equation_solution_range_l4020_402033

theorem equation_solution_range (x k : ℝ) : 2 * x + 3 * k = 1 → x < 0 → k > 1/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_range_l4020_402033


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_cosines_l4020_402071

theorem sum_of_reciprocal_cosines : ∃ ε : ℂ,
  (ε^7 = 1) ∧
  (ε ≠ 1) ∧
  (2 * Complex.cos (2 * Real.pi / 7) = ε + 1 / ε) ∧
  (2 * Complex.cos (4 * Real.pi / 7) = ε^2 + 1 / ε^2) ∧
  (2 * Complex.cos (6 * Real.pi / 7) = ε^3 + 1 / ε^3) ∧
  (1 + ε + ε^2 + ε^3 + ε^4 + ε^5 + ε^6 = 0) →
  1 / (2 * Complex.cos (2 * Real.pi / 7)) +
  1 / (2 * Complex.cos (4 * Real.pi / 7)) +
  1 / (2 * Complex.cos (6 * Real.pi / 7)) = -2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_cosines_l4020_402071


namespace NUMINAMATH_CALUDE_total_investment_l4020_402065

/-- Proves that the total investment of Vishal, Trishul, and Raghu is 5780 Rs. -/
theorem total_investment (raghu_investment : ℝ) 
  (h1 : raghu_investment = 2000)
  (h2 : ∃ trishul_investment : ℝ, trishul_investment = raghu_investment * 0.9)
  (h3 : ∃ vishal_investment : ℝ, vishal_investment = raghu_investment * 0.9 * 1.1) :
  raghu_investment + raghu_investment * 0.9 + raghu_investment * 0.9 * 1.1 = 5780 :=
by sorry


end NUMINAMATH_CALUDE_total_investment_l4020_402065


namespace NUMINAMATH_CALUDE_trigonometric_identities_l4020_402075

theorem trigonometric_identities :
  (Real.sin (62 * π / 180) * Real.cos (32 * π / 180) - Real.cos (62 * π / 180) * Real.sin (32 * π / 180) = 1/2) ∧
  (Real.sin (75 * π / 180) * Real.cos (75 * π / 180) ≠ Real.sqrt 3 / 4) ∧
  ((1 + Real.tan (75 * π / 180)) / (1 - Real.tan (75 * π / 180)) ≠ Real.sqrt 3) ∧
  (Real.sin (50 * π / 180) * (Real.sqrt 3 * Real.sin (10 * π / 180) + Real.cos (10 * π / 180)) / Real.cos (10 * π / 180) = 1) := by
  sorry

#check trigonometric_identities

end NUMINAMATH_CALUDE_trigonometric_identities_l4020_402075


namespace NUMINAMATH_CALUDE_opposite_faces_in_cube_l4020_402073

structure Cube where
  faces : Fin 6 → Char
  top : Fin 6
  front : Fin 6
  right : Fin 6
  back : Fin 6
  left : Fin 6
  bottom : Fin 6
  unique_faces : ∀ i j, i ≠ j → faces i ≠ faces j

def is_opposite (c : Cube) (f1 f2 : Fin 6) : Prop :=
  f1 ≠ f2 ∧ f1 ≠ c.top ∧ f1 ≠ c.bottom ∧ 
  f2 ≠ c.top ∧ f2 ≠ c.bottom ∧
  (f1 = c.front ∧ f2 = c.back ∨
   f1 = c.back ∧ f2 = c.front ∨
   f1 = c.left ∧ f2 = c.right ∨
   f1 = c.right ∧ f2 = c.left)

theorem opposite_faces_in_cube (c : Cube) 
  (h1 : c.faces c.top = 'A')
  (h2 : c.faces c.front = 'B')
  (h3 : c.faces c.right = 'C')
  (h4 : c.faces c.back = 'D')
  (h5 : c.faces c.left = 'E') :
  is_opposite c c.front c.back :=
by sorry

end NUMINAMATH_CALUDE_opposite_faces_in_cube_l4020_402073


namespace NUMINAMATH_CALUDE_davids_trip_expenses_l4020_402006

theorem davids_trip_expenses (initial_amount spent_amount remaining_amount : ℕ) : 
  initial_amount = 1800 →
  remaining_amount = 500 →
  spent_amount = initial_amount - remaining_amount →
  spent_amount - remaining_amount = 800 := by
  sorry

end NUMINAMATH_CALUDE_davids_trip_expenses_l4020_402006


namespace NUMINAMATH_CALUDE_melody_reading_fraction_l4020_402046

theorem melody_reading_fraction (english : ℕ) (science : ℕ) (civics : ℕ) (chinese : ℕ) 
  (total_pages : ℕ) (h1 : english = 20) (h2 : science = 16) (h3 : civics = 8) (h4 : chinese = 12) 
  (h5 : total_pages = 14) :
  ∃ (f : ℚ), f * (english + science + civics + chinese : ℚ) = total_pages ∧ f = 1/4 := by
sorry

end NUMINAMATH_CALUDE_melody_reading_fraction_l4020_402046


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4020_402023

/-- A geometric sequence with specified properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  a 1 + a 2 = 1 →
  a 2 + a 3 = 2 →
  a 6 + a 7 = 32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4020_402023


namespace NUMINAMATH_CALUDE_girls_in_class_l4020_402036

theorem girls_in_class (total_students : ℕ) (girl_ratio boy_ratio : ℕ) (h1 : total_students = 20) (h2 : girl_ratio = 2) (h3 : boy_ratio = 3) : 
  (girl_ratio * total_students) / (girl_ratio + boy_ratio) = 8 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l4020_402036


namespace NUMINAMATH_CALUDE_unique_digit_solution_l4020_402063

theorem unique_digit_solution :
  ∃! (a b c d e f g h i j : ℕ),
    (a ∈ Finset.range 10) ∧
    (b ∈ Finset.range 10) ∧
    (c ∈ Finset.range 10) ∧
    (d ∈ Finset.range 10) ∧
    (e ∈ Finset.range 10) ∧
    (f ∈ Finset.range 10) ∧
    (g ∈ Finset.range 10) ∧
    (h ∈ Finset.range 10) ∧
    (i ∈ Finset.range 10) ∧
    (j ∈ Finset.range 10) ∧
    ({a, b, c, d, e, f, g, h, i, j} : Finset ℕ).card = 10 ∧
    20 * (a - 8) = 20 ∧
    b / 2 + 17 = 20 ∧
    c * d - 4 = 20 ∧
    (e + 8) / 12 = f ∧
    4 * g + h = 20 ∧
    20 * (i - j) = 100 :=
by sorry

end NUMINAMATH_CALUDE_unique_digit_solution_l4020_402063


namespace NUMINAMATH_CALUDE_g_inverse_domain_l4020_402034

/-- The function g(x) = 2(x+1)^2 - 7 -/
def g (x : ℝ) : ℝ := 2 * (x + 1)^2 - 7

/-- d is the lower bound of the restricted domain [d,∞) -/
def d : ℝ := -1

/-- Theorem: -1 is the smallest value of d such that g has an inverse function on [d,∞) -/
theorem g_inverse_domain (x : ℝ) : 
  (∀ y z, x ≥ d → y ≥ d → z ≥ d → g y = g z → y = z) ∧ 
  (∀ e, e < d → ∃ y z, y > e ∧ z > e ∧ y ≠ z ∧ g y = g z) :=
sorry

end NUMINAMATH_CALUDE_g_inverse_domain_l4020_402034


namespace NUMINAMATH_CALUDE_a_range_l4020_402038

-- Define the line equation
def line_equation (a x y : ℝ) : ℝ := a * x + 2 * y - 1

-- Define the points A and B
def point_A : ℝ × ℝ := (3, -1)
def point_B : ℝ × ℝ := (-1, 2)

-- Define the condition for points being on the same side of the line
def same_side (a : ℝ) : Prop :=
  (line_equation a point_A.1 point_A.2) * (line_equation a point_B.1 point_B.2) > 0

-- Theorem stating the range of a
theorem a_range : 
  ∀ a : ℝ, same_side a ↔ a ∈ Set.Ioo 1 3 :=
sorry

end NUMINAMATH_CALUDE_a_range_l4020_402038


namespace NUMINAMATH_CALUDE_fraction_addition_simplification_l4020_402015

theorem fraction_addition_simplification :
  3 / 462 + 13 / 42 = 73 / 231 := by sorry

end NUMINAMATH_CALUDE_fraction_addition_simplification_l4020_402015


namespace NUMINAMATH_CALUDE_women_picnic_attendance_l4020_402009

/-- Represents the percentage of employees in decimal form -/
def Percentage : Type := { p : ℝ // 0 ≤ p ∧ p ≤ 1 }

/-- The percentage of men in the company -/
def men_percentage : Percentage := ⟨0.55, by norm_num⟩

/-- The percentage of men who attended the picnic -/
def men_attendance : Percentage := ⟨0.20, by norm_num⟩

/-- The percentage of all employees who attended the picnic -/
def total_attendance : Percentage := ⟨0.29, by norm_num⟩

/-- The percentage of women who attended the picnic -/
def women_attendance : Percentage := ⟨0.40, by norm_num⟩

theorem women_picnic_attendance :
  let women_percentage := 1 - men_percentage.val
  let men_at_picnic := men_percentage.val * men_attendance.val
  let women_at_picnic := total_attendance.val - men_at_picnic
  women_at_picnic / women_percentage = women_attendance.val := by
  sorry

end NUMINAMATH_CALUDE_women_picnic_attendance_l4020_402009


namespace NUMINAMATH_CALUDE_rectangle_length_proof_l4020_402096

theorem rectangle_length_proof (l w : ℝ) : l = w + 3 ∧ l * w = 4 → l = 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_proof_l4020_402096


namespace NUMINAMATH_CALUDE_sine_function_symmetry_axis_l4020_402069

theorem sine_function_symmetry_axis (φ : ℝ) : 
  (∀ x, ∃ y, y = Real.sin (3 * x + φ)) →
  |φ| < π / 2 →
  (∀ x, Real.sin (3 * x + φ) = Real.sin (3 * (3 * π / 2 - x) + φ)) →
  φ = π / 4 := by
sorry

end NUMINAMATH_CALUDE_sine_function_symmetry_axis_l4020_402069


namespace NUMINAMATH_CALUDE_p_sufficient_but_not_necessary_for_r_l4020_402044

-- Define the propositions
variable (p q r : Prop)

-- Define what it means for a condition to be sufficient but not necessary
def sufficient_but_not_necessary (a b : Prop) : Prop :=
  (a → b) ∧ ¬(b → a)

-- Define what it means for a condition to be necessary but not sufficient
def necessary_but_not_sufficient (a b : Prop) : Prop :=
  (b → a) ∧ ¬(a → b)

-- State the theorem
theorem p_sufficient_but_not_necessary_for_r
  (h1 : sufficient_but_not_necessary p q)
  (h2 : necessary_but_not_sufficient r q) :
  sufficient_but_not_necessary p r :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_but_not_necessary_for_r_l4020_402044


namespace NUMINAMATH_CALUDE_train_speed_l4020_402066

/-- The speed of a train given its length and time to cross a stationary point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 275) (h2 : time = 7) :
  ∃ (speed : ℝ), abs (speed - 141.43) < 0.01 ∧ speed = (length / time) * 3.6 := by
  sorry


end NUMINAMATH_CALUDE_train_speed_l4020_402066


namespace NUMINAMATH_CALUDE_candy_pencils_l4020_402003

/-- Proves that Candy has 9 pencils given the conditions in the problem -/
theorem candy_pencils :
  ∀ (calen_original caleb candy : ℕ),
  calen_original = caleb + 5 →
  caleb = 2 * candy - 3 →
  calen_original - 10 = 10 →
  candy = 9 := by
sorry

end NUMINAMATH_CALUDE_candy_pencils_l4020_402003


namespace NUMINAMATH_CALUDE_win_sector_area_l4020_402060

/-- Theorem: Area of WIN sector in a circular spinner game --/
theorem win_sector_area (r : ℝ) (p_win : ℝ) (p_bonus_lose : ℝ) (h1 : r = 8)
  (h2 : p_win = 1 / 4) (h3 : p_bonus_lose = 1 / 8) :
  p_win * π * r^2 = 16 * π := by sorry

end NUMINAMATH_CALUDE_win_sector_area_l4020_402060


namespace NUMINAMATH_CALUDE_savings_percentage_is_twenty_percent_l4020_402098

def monthly_salary : ℝ := 6250
def savings_after_increase : ℝ := 250
def expense_increase_rate : ℝ := 0.2

theorem savings_percentage_is_twenty_percent :
  ∃ P : ℝ, 
    savings_after_increase = monthly_salary - (1 + expense_increase_rate) * (monthly_salary - (P / 100) * monthly_salary) ∧
    P = 20 := by
  sorry

end NUMINAMATH_CALUDE_savings_percentage_is_twenty_percent_l4020_402098


namespace NUMINAMATH_CALUDE_cream_fraction_after_mixing_l4020_402083

/-- Represents the contents of a cup -/
structure CupContents where
  coffee : ℚ
  cream : ℚ

/-- Represents the mixing process -/
def mix_and_transfer (cup1 cup2 : CupContents) : (CupContents × CupContents) :=
  sorry

theorem cream_fraction_after_mixing :
  let initial_cup1 : CupContents := { coffee := 4, cream := 0 }
  let initial_cup2 : CupContents := { coffee := 0, cream := 4 }
  let (final_cup1, _) := mix_and_transfer initial_cup1 initial_cup2
  (final_cup1.cream / (final_cup1.coffee + final_cup1.cream)) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_cream_fraction_after_mixing_l4020_402083


namespace NUMINAMATH_CALUDE_cosine_symmetry_center_l4020_402031

/-- Given a cosine function y = 2cos(2x) translated π/12 units to the right,
    prove that (5π/6, 0) is one of its symmetry centers. -/
theorem cosine_symmetry_center :
  let f : ℝ → ℝ := λ x ↦ 2 * Real.cos (2 * (x - π/12))
  ∃ (k : ℤ), (5*π/6 : ℝ) = k*π/2 + π/3 ∧ 
    (∀ x : ℝ, f (5*π/6 + x) = f (5*π/6 - x)) :=
by sorry

end NUMINAMATH_CALUDE_cosine_symmetry_center_l4020_402031


namespace NUMINAMATH_CALUDE_last_three_average_l4020_402051

theorem last_three_average (list : List ℝ) (h1 : list.length = 7) 
  (h2 : list.sum / 7 = 60) (h3 : (list.take 4).sum / 4 = 55) : 
  (list.drop 4).sum / 3 = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_last_three_average_l4020_402051


namespace NUMINAMATH_CALUDE_equatorial_circumference_scientific_notation_l4020_402024

/-- Scientific notation representation -/
structure ScientificNotation where
  a : ℝ
  n : ℤ
  h1 : 1 ≤ a
  h2 : a < 10

/-- Check if a ScientificNotation represents a given number -/
def represents (sn : ScientificNotation) (x : ℝ) : Prop :=
  sn.a * (10 : ℝ) ^ sn.n = x

/-- The equatorial circumference in meters -/
def equatorialCircumference : ℝ := 40000000

/-- Theorem stating that 4 × 10^7 is the correct scientific notation for the equatorial circumference -/
theorem equatorial_circumference_scientific_notation :
  ∃ sn : ScientificNotation, sn.a = 4 ∧ sn.n = 7 ∧ represents sn equatorialCircumference :=
sorry

end NUMINAMATH_CALUDE_equatorial_circumference_scientific_notation_l4020_402024


namespace NUMINAMATH_CALUDE_periodic_function_value_l4020_402042

def periodic_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2 * Real.pi) = f x

theorem periodic_function_value 
  (f : ℝ → ℝ) 
  (h1 : periodic_function f) 
  (h2 : f 0 = 0) : 
  f (4 * Real.pi) = 0 := by
sorry

end NUMINAMATH_CALUDE_periodic_function_value_l4020_402042


namespace NUMINAMATH_CALUDE_function_properties_l4020_402088

/-- Given function f with parameter a > 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1/a) * Real.log x + 1/x - x

theorem function_properties (a : ℝ) (h : a > 1) :
  (∀ x ∈ Set.Ioo (0 : ℝ) (1/a), (deriv (f a)) x < 0) ∧
  (∀ x ∈ Set.Ioo (1/a) 1, (deriv (f a)) x > 0) ∧
  (a ≥ 3 → ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    (deriv (f a)) x₁ = (deriv (f a)) x₂ ∧ x₁ + x₂ > 6/5) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l4020_402088


namespace NUMINAMATH_CALUDE_reflection_of_circle_center_l4020_402049

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem reflection_of_circle_center :
  let original_center : ℝ × ℝ := (8, -3)
  let reflected_center := reflect_about_neg_x original_center
  reflected_center = (3, -8) := by
sorry

end NUMINAMATH_CALUDE_reflection_of_circle_center_l4020_402049


namespace NUMINAMATH_CALUDE_total_time_theorem_l4020_402011

/-- The time Carlotta spends practicing for each minute of singing -/
def practice_time : ℕ := 3

/-- The time Carlotta spends throwing tantrums for each minute of singing -/
def tantrum_time : ℕ := 5

/-- The length of the final stage performance in minutes -/
def performance_length : ℕ := 6

/-- The total time spent per minute of singing -/
def total_time_per_minute : ℕ := 1 + practice_time + tantrum_time

/-- Theorem: The total combined amount of time Carlotta spends practicing, 
    throwing tantrums, and singing in the final stage performance is 54 minutes -/
theorem total_time_theorem : performance_length * total_time_per_minute = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_time_theorem_l4020_402011


namespace NUMINAMATH_CALUDE_prob_diff_color_is_29_50_l4020_402008

-- Define the contents of the boxes
def boxA : Finset (Fin 3) := {0, 0, 1, 1, 2}
def boxB : Finset (Fin 3) := {0, 0, 0, 0, 1, 1, 1, 2, 2}

-- Define the probability of drawing a ball of a different color
def prob_diff_color : ℚ :=
  let total_A := boxA.card
  let total_B := boxB.card + 1
  let prob_white := (boxA.filter (· = 0)).card / total_A *
                    (boxB.filter (· ≠ 0)).card / total_B
  let prob_red := (boxA.filter (· = 1)).card / total_A *
                  (boxB.filter (· ≠ 1)).card / total_B
  let prob_black := (boxA.filter (· = 2)).card / total_A *
                    (boxB.filter (· ≠ 2)).card / total_B
  prob_white + prob_red + prob_black

-- Theorem statement
theorem prob_diff_color_is_29_50 : prob_diff_color = 29 / 50 := by
  sorry

end NUMINAMATH_CALUDE_prob_diff_color_is_29_50_l4020_402008


namespace NUMINAMATH_CALUDE_maximize_fruit_yield_l4020_402070

/-- Maximizing fruit yield in an orchard --/
theorem maximize_fruit_yield (x : ℝ) :
  let initial_trees : ℝ := 100
  let initial_yield_per_tree : ℝ := 600
  let yield_decrease_per_tree : ℝ := 5
  let total_trees : ℝ := x + initial_trees
  let new_yield_per_tree : ℝ := initial_yield_per_tree - yield_decrease_per_tree * x
  let total_yield : ℝ := total_trees * new_yield_per_tree
  (∀ z : ℝ, total_yield ≥ (z + initial_trees) * (initial_yield_per_tree - yield_decrease_per_tree * z)) →
  x = 10 := by
sorry

end NUMINAMATH_CALUDE_maximize_fruit_yield_l4020_402070


namespace NUMINAMATH_CALUDE_maxim_birth_probability_l4020_402055

/-- The year Maxim starts first grade -/
def start_year : ℕ := 2014

/-- Maxim's age when starting first grade -/
def start_age : ℕ := 6

/-- The day of the year when Maxim starts first grade (1st September) -/
def start_day : ℕ := 244

/-- The number of days in a year (assuming non-leap year) -/
def days_in_year : ℕ := 365

/-- The year we're interested in for Maxim's birth -/
def birth_year_of_interest : ℕ := 2008

/-- The number of days from 1st January to 31st August in 2008 (leap year) -/
def days_in_2008_until_august : ℕ := 244

theorem maxim_birth_probability :
  let total_possible_days := days_in_year
  let favorable_days := days_in_2008_until_august
  (favorable_days : ℚ) / total_possible_days = 244 / 365 := by
  sorry

end NUMINAMATH_CALUDE_maxim_birth_probability_l4020_402055


namespace NUMINAMATH_CALUDE_perfect_square_condition_l4020_402090

/-- A polynomial of the form ax^2 + bx + c is a perfect square trinomial if and only if
    there exist real numbers p and q such that ax^2 + bx + c = (px + q)^2 -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

/-- The main theorem stating the condition for the given polynomial to be a perfect square trinomial -/
theorem perfect_square_condition (k : ℝ) :
  is_perfect_square_trinomial 4 (-(k-1)) 9 ↔ k = 13 ∨ k = -11 := by
  sorry


end NUMINAMATH_CALUDE_perfect_square_condition_l4020_402090


namespace NUMINAMATH_CALUDE_lemonade_lemons_l4020_402005

/-- Given that each glass of lemonade requires 2 lemons and Jane can make 9 glasses,
    prove that the total number of lemons is 18. -/
theorem lemonade_lemons :
  ∀ (lemons_per_glass : ℕ) (glasses : ℕ) (total_lemons : ℕ),
    lemons_per_glass = 2 →
    glasses = 9 →
    total_lemons = lemons_per_glass * glasses →
    total_lemons = 18 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_lemons_l4020_402005


namespace NUMINAMATH_CALUDE_student_number_problem_l4020_402085

theorem student_number_problem (x : ℤ) : x = 48 ↔ 5 * x - 138 = 102 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l4020_402085


namespace NUMINAMATH_CALUDE_perpendicular_line_inclination_angle_l4020_402091

/-- The inclination angle of a line perpendicular to x + √3y - 1 = 0 is π/3 -/
theorem perpendicular_line_inclination_angle : 
  let original_line : Real → Real → Prop := λ x y => x + Real.sqrt 3 * y - 1 = 0
  let perpendicular_slope : Real := Real.sqrt 3
  let inclination_angle : Real := Real.pi / 3
  ∀ x y, original_line x y → 
    ∃ m : Real, m * perpendicular_slope = -1 ∧ 
    Real.tan inclination_angle = perpendicular_slope :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_inclination_angle_l4020_402091


namespace NUMINAMATH_CALUDE_meals_without_restrictions_l4020_402052

theorem meals_without_restrictions (total_clients : ℕ) (vegan kosher gluten_free halal dairy_free nut_free : ℕ)
  (vegan_kosher vegan_gluten kosher_gluten halal_dairy gluten_nut : ℕ)
  (vegan_halal_gluten kosher_dairy_nut : ℕ)
  (h1 : total_clients = 80)
  (h2 : vegan = 15)
  (h3 : kosher = 18)
  (h4 : gluten_free = 12)
  (h5 : halal = 10)
  (h6 : dairy_free = 8)
  (h7 : nut_free = 4)
  (h8 : vegan_kosher = 5)
  (h9 : vegan_gluten = 6)
  (h10 : kosher_gluten = 3)
  (h11 : halal_dairy = 4)
  (h12 : gluten_nut = 2)
  (h13 : vegan_halal_gluten = 2)
  (h14 : kosher_dairy_nut = 1) :
  total_clients - (vegan + kosher + gluten_free + halal + dairy_free + nut_free - 
    (vegan_kosher + vegan_gluten + kosher_gluten + halal_dairy + gluten_nut) + 
    (vegan_halal_gluten + kosher_dairy_nut)) = 30 := by
  sorry


end NUMINAMATH_CALUDE_meals_without_restrictions_l4020_402052


namespace NUMINAMATH_CALUDE_car_trip_average_speed_l4020_402035

/-- Given a car's trip with two segments:
    1. 40 miles on local roads at 20 mph
    2. 180 miles on highway at 60 mph
    The average speed of the entire trip is 44 mph -/
theorem car_trip_average_speed :
  let local_distance : ℝ := 40
  let local_speed : ℝ := 20
  let highway_distance : ℝ := 180
  let highway_speed : ℝ := 60
  let total_distance : ℝ := local_distance + highway_distance
  let total_time : ℝ := local_distance / local_speed + highway_distance / highway_speed
  total_distance / total_time = 44 := by sorry

end NUMINAMATH_CALUDE_car_trip_average_speed_l4020_402035


namespace NUMINAMATH_CALUDE_max_value_x_minus_2y_l4020_402072

theorem max_value_x_minus_2y (x y : ℝ) : 
  x^2 - 4*x + y^2 = 0 → 
  ∃ (max : ℝ), (∀ (x' y' : ℝ), x'^2 - 4*x' + y'^2 = 0 → x' - 2*y' ≤ max) ∧ 
                (∃ (x₀ y₀ : ℝ), x₀^2 - 4*x₀ + y₀^2 = 0 ∧ x₀ - 2*y₀ = max) ∧
                max = 2*Real.sqrt 5 + 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_x_minus_2y_l4020_402072


namespace NUMINAMATH_CALUDE_ellipse_with_foci_on_y_axis_range_l4020_402093

/-- The equation of the curve -/
def equation (x y k : ℝ) : Prop := x^2 / (k - 5) + y^2 / (10 - k) = 1

/-- The condition for the equation to represent an ellipse -/
def is_ellipse (k : ℝ) : Prop := k - 5 > 0 ∧ 10 - k > 0

/-- The condition for the foci to be on the y-axis -/
def foci_on_y_axis (k : ℝ) : Prop := 10 - k > k - 5

/-- The theorem stating the range of k for which the equation represents an ellipse with foci on the y-axis -/
theorem ellipse_with_foci_on_y_axis_range (k : ℝ) :
  is_ellipse k ∧ foci_on_y_axis k ↔ k ∈ Set.Ioo 5 7.5 :=
sorry

end NUMINAMATH_CALUDE_ellipse_with_foci_on_y_axis_range_l4020_402093


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l4020_402019

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x| = 2 * x + 1 :=
by
  -- The unique solution is x = -1/3
  use -1/3
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l4020_402019


namespace NUMINAMATH_CALUDE_ellipse_tangent_min_length_l4020_402017

/-
  Define the ellipse C₁: x²/a² + y² = 1 (a > 1)
  where |F₁F₂|² is the arithmetic mean of |A₁A₂|² and |B₁B₂|²
-/
def C₁ (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 = 1 ∧ a > 1 ∧ 
  ∃ (b c : ℝ), 2 * (2*c)^2 = (2*a)^2 + (2*b)^2 ∧ b = 1

-- Define the curve C₂
def C₂ (t x y : ℝ) : Prop :=
  (x - t)^2 + y^2 = (t^2 + Real.sqrt 3 * t)^2 ∧ 0 < t ∧ t ≤ Real.sqrt 2 / 2

-- Define the tangent line l passing through the left vertex of C₁
def tangent_line (a t k : ℝ) : Prop :=
  ∃ (x y : ℝ), C₂ t x y ∧ y = k * (x + Real.sqrt 3)

-- Theorem statement
theorem ellipse_tangent_min_length :
  ∃ (a : ℝ), C₁ a (-Real.sqrt 3) 0 ∧
  (∀ x y, C₁ a x y ↔ x^2 / 3 + y^2 = 1) ∧
  (∀ t k, tangent_line a t k →
    ∃ (x y : ℝ), C₁ a x y ∧ y = k * (x + Real.sqrt 3) ∧
    ∀ (x' y' : ℝ), C₁ a x' y' ∧ y' = k * (x' + Real.sqrt 3) →
      (x - (-Real.sqrt 3))^2 + y^2 ≥ 3/2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_tangent_min_length_l4020_402017


namespace NUMINAMATH_CALUDE_square_side_length_l4020_402041

/-- Given a square ABCD with side length x, prove that x = 12 under the given conditions --/
theorem square_side_length (x : ℝ) 
  (h1 : x > 0) -- Ensure x is positive
  (h2 : x^2 - (1/2) * ((x-5) * (x-4)) - (7/2) * (x-7) - 2*(x-1) - 3.5 = 78) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l4020_402041


namespace NUMINAMATH_CALUDE_square_equality_base_is_ten_l4020_402092

/-- The base in which 34 squared equals 1296 -/
def base_b : ℕ := sorry

/-- The representation of 34 in base b -/
def thirty_four_b (b : ℕ) : ℕ := 3 * b + 4

/-- The representation of 1296 in base b -/
def twelve_ninety_six_b (b : ℕ) : ℕ := b^3 + 2*b^2 + 9*b + 6

/-- The theorem stating that the square of 34 in base b equals 1296 in base b -/
theorem square_equality (b : ℕ) : (thirty_four_b b)^2 = twelve_ninety_six_b b := by sorry

/-- The main theorem proving that the base b is 10 -/
theorem base_is_ten : base_b = 10 := by sorry

end NUMINAMATH_CALUDE_square_equality_base_is_ten_l4020_402092


namespace NUMINAMATH_CALUDE_complement_P_intersect_Q_P_proper_subset_Q_iff_a_in_range_l4020_402029

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a + 1}
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- Part 1
theorem complement_P_intersect_Q : 
  (Set.univ \ P 3) ∩ Q = {x : ℝ | -2 ≤ x ∧ x < 4} := by sorry

-- Part 2
theorem P_proper_subset_Q_iff_a_in_range : 
  ∀ a : ℝ, (P a ⊂ Q ∧ P a ≠ Q) ↔ 0 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_complement_P_intersect_Q_P_proper_subset_Q_iff_a_in_range_l4020_402029


namespace NUMINAMATH_CALUDE_quadratic_complex_solution_sum_l4020_402094

theorem quadratic_complex_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, 6 * x^2 + 1 = 5 * x - 16 ↔ x = a + b * I ∨ x = a - b * I) →
  a + b^2 = 443/144 := by
sorry

end NUMINAMATH_CALUDE_quadratic_complex_solution_sum_l4020_402094


namespace NUMINAMATH_CALUDE_lily_bought_20_ducks_l4020_402061

/-- The number of ducks Lily bought -/
def lily_ducks : ℕ := sorry

/-- The number of geese Lily bought -/
def lily_geese : ℕ := 10

/-- The number of ducks Rayden bought -/
def rayden_ducks : ℕ := 3 * lily_ducks

/-- The number of geese Rayden bought -/
def rayden_geese : ℕ := 4 * lily_geese

/-- The total number of birds Lily has -/
def lily_total : ℕ := lily_ducks + lily_geese

/-- The total number of birds Rayden has -/
def rayden_total : ℕ := rayden_ducks + rayden_geese

theorem lily_bought_20_ducks :
  lily_ducks = 20 ∧
  rayden_total = lily_total + 70 := by
  sorry

end NUMINAMATH_CALUDE_lily_bought_20_ducks_l4020_402061


namespace NUMINAMATH_CALUDE_series_sum_l4020_402064

theorem series_sum : 
  (3/4 : ℚ) + 5/8 + 9/16 + 17/32 + 33/64 + 65/128 - (7/2 : ℚ) = -1/128 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l4020_402064


namespace NUMINAMATH_CALUDE_probability_derek_julia_captains_l4020_402097

theorem probability_derek_julia_captains (total_players : Nat) (num_teams : Nat) (team_size : Nat) (captains_per_team : Nat) :
  total_players = 64 →
  num_teams = 8 →
  team_size = 8 →
  captains_per_team = 2 →
  num_teams * team_size = total_players →
  (probability_both_captains : ℚ) = 5 / 84 :=
by
  sorry

end NUMINAMATH_CALUDE_probability_derek_julia_captains_l4020_402097


namespace NUMINAMATH_CALUDE_lucky_iff_power_of_two_l4020_402095

/-- Represents the three colors of cubes -/
inductive Color
  | White
  | Blue
  | Red

/-- Represents an arrangement of N cubes in a circle -/
def Arrangement (N : ℕ) := Fin N → Color

/-- Determines if an arrangement is good (final cube color doesn't depend on starting position) -/
def is_good (N : ℕ) (arr : Arrangement N) : Prop := sorry

/-- Determines if N is lucky (all arrangements of N cubes are good) -/
def is_lucky (N : ℕ) : Prop :=
  ∀ arr : Arrangement N, is_good N arr

/-- Main theorem: N is lucky if and only if it's a power of 2 -/
theorem lucky_iff_power_of_two (N : ℕ) :
  is_lucky N ↔ ∃ k : ℕ, N = 2^k :=
sorry

end NUMINAMATH_CALUDE_lucky_iff_power_of_two_l4020_402095


namespace NUMINAMATH_CALUDE_expression_evaluation_l4020_402059

theorem expression_evaluation : (-1)^10 * 2 + (-2)^3 / 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4020_402059


namespace NUMINAMATH_CALUDE_tea_mixture_profit_l4020_402026

/-- Proves that the given tea mixture achieves the desired profit -/
theorem tea_mixture_profit (x y : ℝ) : 
  x + y = 100 →
  0.32 * x + 0.40 * y = 34.40 →
  x = 70 ∧ y = 30 ∧ 
  (0.43 * 100 / (0.32 * x + 0.40 * y) - 1) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_tea_mixture_profit_l4020_402026


namespace NUMINAMATH_CALUDE_books_borrowed_after_lunch_l4020_402047

theorem books_borrowed_after_lunch (initial_books : ℕ) (borrowed_by_lunch : ℕ) (added_after_lunch : ℕ) (remaining_by_evening : ℕ) : 
  initial_books = 100 →
  borrowed_by_lunch = 50 →
  added_after_lunch = 40 →
  remaining_by_evening = 60 →
  initial_books - borrowed_by_lunch + added_after_lunch - remaining_by_evening = 30 := by
sorry

end NUMINAMATH_CALUDE_books_borrowed_after_lunch_l4020_402047


namespace NUMINAMATH_CALUDE_alkaline_probability_is_two_fifths_l4020_402080

/-- The number of total solutions -/
def total_solutions : ℕ := 5

/-- The number of alkaline solutions -/
def alkaline_solutions : ℕ := 2

/-- The probability of selecting an alkaline solution -/
def alkaline_probability : ℚ := alkaline_solutions / total_solutions

theorem alkaline_probability_is_two_fifths :
  alkaline_probability = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_alkaline_probability_is_two_fifths_l4020_402080


namespace NUMINAMATH_CALUDE_square_inequality_l4020_402028

theorem square_inequality (a b : ℝ) : a > b ∧ b > 0 → a^2 > b^2 ∧ ¬(∀ x y : ℝ, x^2 > y^2 → x > y ∧ y > 0) := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_l4020_402028


namespace NUMINAMATH_CALUDE_prime_square_minus_one_divisibility_l4020_402068

theorem prime_square_minus_one_divisibility (p : ℕ) :
  Prime p → p ≥ 7 →
  (∃ q : ℕ, Prime q ∧ q ≥ 7 ∧ 40 ∣ (q^2 - 1)) ∧
  (∃ r : ℕ, Prime r ∧ r ≥ 7 ∧ ¬(40 ∣ (r^2 - 1))) := by
  sorry

end NUMINAMATH_CALUDE_prime_square_minus_one_divisibility_l4020_402068


namespace NUMINAMATH_CALUDE_derivative_log2_l4020_402014

-- Define the base-2 logarithm function
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem derivative_log2 (x : ℝ) (h : x > 0) : 
  deriv log2 x = 1 / (x * Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_derivative_log2_l4020_402014


namespace NUMINAMATH_CALUDE_student_count_l4020_402027

/-- Proves the number of students in a class given certain height data -/
theorem student_count (initial_avg : ℝ) (incorrect_height : ℝ) (correct_height : ℝ) (actual_avg : ℝ) :
  initial_avg = 175 →
  incorrect_height = 151 →
  correct_height = 111 →
  actual_avg = 173 →
  ∃ n : ℕ, n * actual_avg = n * initial_avg - (incorrect_height - correct_height) ∧ n = 20 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l4020_402027


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l4020_402013

theorem no_real_roots_quadratic (m : ℝ) : 
  (∀ x : ℝ, -2 * x^2 + 6 * x + m ≠ 0) → m < -4.5 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l4020_402013


namespace NUMINAMATH_CALUDE_binary_digits_difference_l4020_402043

theorem binary_digits_difference : ∃ n m : ℕ, 
  (2^n ≤ 300 ∧ 300 < 2^(n+1)) ∧ 
  (2^m ≤ 1400 ∧ 1400 < 2^(m+1)) ∧ 
  m - n = 2 := by
sorry

end NUMINAMATH_CALUDE_binary_digits_difference_l4020_402043


namespace NUMINAMATH_CALUDE_square_field_area_l4020_402062

/-- The area of a square field with side length 8 meters is 64 square meters. -/
theorem square_field_area : 
  ∀ (side_length area : ℝ), 
  side_length = 8 → 
  area = side_length ^ 2 → 
  area = 64 := by sorry

end NUMINAMATH_CALUDE_square_field_area_l4020_402062


namespace NUMINAMATH_CALUDE_expression_evaluation_l4020_402078

theorem expression_evaluation : 
  (2023^3 - 3 * 2023^2 * 2024 + 5 * 2023 * 2024^2 - 2024^3 + 5) / (2023 * 2024) = 4048 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4020_402078


namespace NUMINAMATH_CALUDE_group_value_l4020_402016

theorem group_value (a : ℝ) (h : 21 ≤ a ∧ a < 41) : (21 + 41) / 2 = 31 := by
  sorry

#check group_value

end NUMINAMATH_CALUDE_group_value_l4020_402016
