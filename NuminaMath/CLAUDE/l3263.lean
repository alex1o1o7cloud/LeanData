import Mathlib

namespace NUMINAMATH_CALUDE_juice_packs_fit_l3263_326321

/-- The number of juice packs that can fit in a box without gaps -/
def juice_packs_in_box (box_width box_length box_height juice_width juice_length juice_height : ℕ) : ℕ :=
  (box_width * box_length * box_height) / (juice_width * juice_length * juice_height)

/-- Theorem stating that 72 juice packs fit in the given box -/
theorem juice_packs_fit :
  juice_packs_in_box 24 15 28 4 5 7 = 72 := by
  sorry

#eval juice_packs_in_box 24 15 28 4 5 7

end NUMINAMATH_CALUDE_juice_packs_fit_l3263_326321


namespace NUMINAMATH_CALUDE_red_to_blue_ratio_l3263_326318

/-- Represents the number of marbles of each color in Cara's bag. -/
structure MarbleCounts where
  total : ℕ
  yellow : ℕ
  green : ℕ
  red : ℕ
  blue : ℕ

/-- Theorem stating the ratio of red to blue marbles given the conditions -/
theorem red_to_blue_ratio (m : MarbleCounts) : 
  m.total = 60 ∧ 
  m.yellow = 20 ∧ 
  m.green = m.yellow / 2 ∧ 
  m.total = m.yellow + m.green + m.red + m.blue ∧ 
  m.blue = m.total / 4 →
  m.red / m.blue = 11 / 4 := by
  sorry

end NUMINAMATH_CALUDE_red_to_blue_ratio_l3263_326318


namespace NUMINAMATH_CALUDE_not_p_or_q_false_implies_p_or_q_true_l3263_326348

theorem not_p_or_q_false_implies_p_or_q_true (p q : Prop) :
  ¬(¬(p ∨ q)) → (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_not_p_or_q_false_implies_p_or_q_true_l3263_326348


namespace NUMINAMATH_CALUDE_least_reducible_n_l3263_326303

def is_reducible (a b : Int) : Prop :=
  Int.gcd a b > 1

def fraction_numerator (n : Int) : Int :=
  2*n - 26

def fraction_denominator (n : Int) : Int :=
  10*n + 12

theorem least_reducible_n :
  (∀ k : Nat, k > 0 ∧ k < 49 → ¬(is_reducible (fraction_numerator k) (fraction_denominator k))) ∧
  (is_reducible (fraction_numerator 49) (fraction_denominator 49)) :=
sorry

end NUMINAMATH_CALUDE_least_reducible_n_l3263_326303


namespace NUMINAMATH_CALUDE_construct_one_degree_angle_l3263_326369

-- Define the given angle
def given_angle : ℕ := 19

-- Define the target angle
def target_angle : ℕ := 1

-- Theorem stating that it's possible to construct the target angle from the given angle
theorem construct_one_degree_angle :
  ∃ n : ℕ, (n * given_angle) % 360 = target_angle :=
sorry

end NUMINAMATH_CALUDE_construct_one_degree_angle_l3263_326369


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l3263_326350

/-- The volume of a rectangular box with face areas 36, 18, and 8 square inches is 72 cubic inches. -/
theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 36)
  (area2 : w * h = 18)
  (area3 : l * h = 8) :
  l * w * h = 72 := by sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l3263_326350


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3263_326360

theorem decimal_to_fraction :
  (3.375 : ℚ) = 27 / 8 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3263_326360


namespace NUMINAMATH_CALUDE_possible_student_totals_l3263_326304

/-- Represents the distribution of students in groups -/
structure StudentDistribution where
  total_groups : Nat
  groups_with_13 : Nat
  total_students : Nat

/-- Checks if the distribution is valid according to the problem conditions -/
def is_valid_distribution (d : StudentDistribution) : Prop :=
  d.total_groups = 6 ∧
  d.groups_with_13 = 4 ∧
  (d.total_students = 76 ∨ d.total_students = 80)

/-- Theorem stating the possible total numbers of students -/
theorem possible_student_totals :
  ∀ d : StudentDistribution,
    is_valid_distribution d →
    (d.total_students = 76 ∨ d.total_students = 80) :=
by
  sorry

#check possible_student_totals

end NUMINAMATH_CALUDE_possible_student_totals_l3263_326304


namespace NUMINAMATH_CALUDE_total_potatoes_brought_home_l3263_326305

/-- The number of people who received potatoes -/
def num_people : ℕ := 3

/-- The number of potatoes each person received -/
def potatoes_per_person : ℕ := 8

/-- Theorem: The total number of potatoes brought home is 24 -/
theorem total_potatoes_brought_home : 
  num_people * potatoes_per_person = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_potatoes_brought_home_l3263_326305


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_is_gcd_l3263_326300

/-- The value of a pig in dollars -/
def pig_value : ℕ := 300

/-- The value of a goat in dollars -/
def goat_value : ℕ := 210

/-- The smallest positive debt that can be resolved using pigs and goats -/
def smallest_resolvable_debt : ℕ := 30

/-- Theorem stating that the smallest_resolvable_debt is the smallest positive integer
    that can be expressed as a linear combination of pig_value and goat_value -/
theorem smallest_resolvable_debt_is_gcd :
  smallest_resolvable_debt = Nat.gcd pig_value goat_value ∧
  ∀ d : ℕ, d > 0 → (∃ a b : ℤ, d = a * pig_value + b * goat_value) →
    d ≥ smallest_resolvable_debt :=
by sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_is_gcd_l3263_326300


namespace NUMINAMATH_CALUDE_seating_arrangement_l3263_326353

/-- The number of students per row and total number of students in a seating arrangement problem. -/
theorem seating_arrangement (S R : ℕ) 
  (h1 : S = 5 * R + 6)  -- When 5 students sit in a row, 6 are left without seats
  (h2 : S = 12 * (R - 3))  -- When 12 students sit in a row, 3 rows are empty
  : R = 6 ∧ S = 36 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_l3263_326353


namespace NUMINAMATH_CALUDE_grinder_price_correct_l3263_326379

/-- The purchase price of the grinder -/
def grinder_price : ℝ := 15000

/-- The purchase price of the mobile phone -/
def mobile_price : ℝ := 8000

/-- The selling price of the grinder -/
def grinder_sell_price : ℝ := 0.98 * grinder_price

/-- The selling price of the mobile phone -/
def mobile_sell_price : ℝ := 1.1 * mobile_price

/-- The total profit -/
def total_profit : ℝ := 500

theorem grinder_price_correct :
  grinder_sell_price + mobile_sell_price = grinder_price + mobile_price + total_profit :=
by sorry

end NUMINAMATH_CALUDE_grinder_price_correct_l3263_326379


namespace NUMINAMATH_CALUDE_max_a_value_l3263_326332

theorem max_a_value (a : ℝ) : (∀ x : ℝ, x^2 + |2*x - 6| ≥ a) → a ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l3263_326332


namespace NUMINAMATH_CALUDE_heels_cost_equals_savings_plus_contribution_l3263_326381

/-- The cost of heels Miranda wants to buy -/
def heels_cost : ℕ := 260

/-- The number of months Miranda saved money -/
def months_saved : ℕ := 3

/-- The amount Miranda saved per month -/
def savings_per_month : ℕ := 70

/-- The amount Miranda's sister contributed -/
def sister_contribution : ℕ := 50

/-- Theorem stating that the cost of the heels is equal to Miranda's total savings plus her sister's contribution -/
theorem heels_cost_equals_savings_plus_contribution :
  heels_cost = months_saved * savings_per_month + sister_contribution :=
by sorry

end NUMINAMATH_CALUDE_heels_cost_equals_savings_plus_contribution_l3263_326381


namespace NUMINAMATH_CALUDE_closest_fraction_l3263_326333

def medals_won : ℚ := 28 / 150

def options : List ℚ := [1/4, 1/5, 1/6, 1/7, 1/8]

theorem closest_fraction : 
  ∃ (closest : ℚ), closest ∈ options ∧ 
  ∀ (x : ℚ), x ∈ options → |medals_won - closest| ≤ |medals_won - x| ∧
  closest = 1/5 := by sorry

end NUMINAMATH_CALUDE_closest_fraction_l3263_326333


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3263_326394

-- Define the inequality system
def inequality_system (a b x : ℝ) : Prop :=
  x - a > 2 ∧ x + 1 < b

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  -1 < x ∧ x < 1

-- Theorem statement
theorem inequality_system_solution (a b : ℝ) :
  (∀ x, inequality_system a b x ↔ solution_set x) →
  (a + b)^2023 = -1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3263_326394


namespace NUMINAMATH_CALUDE_max_tournament_size_l3263_326310

/-- Represents a tournament with 2^n students --/
structure Tournament (n : ℕ) where
  students : Fin (2^n)
  day1_pairs : List (Fin (2^n) × Fin (2^n))
  day2_pairs : List (Fin (2^n) × Fin (2^n))

/-- The sets of pairs that played on both days are the same --/
def same_pairs (t : Tournament n) : Prop :=
  t.day1_pairs.toFinset = t.day2_pairs.toFinset

/-- The maximum value of n for which the tournament conditions hold --/
def max_n : ℕ := 3

/-- Theorem stating that 3 is the maximum value of n for which the tournament conditions hold --/
theorem max_tournament_size :
  ∀ n : ℕ, n > max_n → ¬∃ t : Tournament n, same_pairs t :=
sorry

end NUMINAMATH_CALUDE_max_tournament_size_l3263_326310


namespace NUMINAMATH_CALUDE_digit_reversal_difference_l3263_326384

theorem digit_reversal_difference (a b : ℕ) (h1 : a < 10) (h2 : b < 10) :
  ∃ k : ℤ, (10 * a + b) - (10 * b + a) = 9 * k :=
by sorry

end NUMINAMATH_CALUDE_digit_reversal_difference_l3263_326384


namespace NUMINAMATH_CALUDE_cos_power_sum_l3263_326335

theorem cos_power_sum (α : ℝ) (x : ℝ) (n : ℕ) (h : x ≠ 0) :
  x + 1/x = 2 * Real.cos α → x^n + 1/x^n = 2 * Real.cos (n * α) := by
  sorry

end NUMINAMATH_CALUDE_cos_power_sum_l3263_326335


namespace NUMINAMATH_CALUDE_smallest_label_on_1993_l3263_326316

theorem smallest_label_on_1993 (n : ℕ) (h : n > 0) :
  (n * (n + 1) / 2) % 2000 = 1021 →
  ∀ m, 0 < m ∧ m < n → (m * (m + 1) / 2) % 2000 ≠ 1021 →
  n = 118 := by
sorry

end NUMINAMATH_CALUDE_smallest_label_on_1993_l3263_326316


namespace NUMINAMATH_CALUDE_expression_behavior_l3263_326352

/-- Given a > b > c, this theorem characterizes the behavior of the expression (a-x)(b-x)/(c-x) for different values of x. -/
theorem expression_behavior (a b c x : ℝ) (h : a > b ∧ b > c) :
  let f := fun (x : ℝ) => (a - x) * (b - x) / (c - x)
  (x < c ∨ (b < x ∧ x < a) → f x > 0) ∧
  ((c < x ∧ x < b) ∨ x > a → f x < 0) ∧
  (x = a ∨ x = b → f x = 0) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ y, 0 < |y - c| ∧ |y - c| < δ → |f y| > 1/ε) :=
by sorry

end NUMINAMATH_CALUDE_expression_behavior_l3263_326352


namespace NUMINAMATH_CALUDE_estimate_larger_than_actual_l3263_326337

theorem estimate_larger_than_actual (x y z : ℝ) 
  (h1 : x > y) (h2 : y > 0) (h3 : z > 0) : 
  (x + 2*z) - (y - 2*z) > x - y := by
  sorry

end NUMINAMATH_CALUDE_estimate_larger_than_actual_l3263_326337


namespace NUMINAMATH_CALUDE_function_convexity_concavity_l3263_326359

-- Function convexity/concavity theorem
theorem function_convexity_concavity :
  -- x² is convex everywhere
  (∀ (x₁ x₂ q₁ q₂ : ℝ), q₁ > 0 → q₂ > 0 → q₁ + q₂ = 1 →
    q₁ * x₁^2 + q₂ * x₂^2 - (q₁ * x₁ + q₂ * x₂)^2 ≥ 0) ∧
  -- √x is concave everywhere
  (∀ (x₁ x₂ q₁ q₂ : ℝ), x₁ > 0 → x₂ > 0 → q₁ > 0 → q₂ > 0 → q₁ + q₂ = 1 →
    q₁ * Real.sqrt x₁ + q₂ * Real.sqrt x₂ - Real.sqrt (q₁ * x₁ + q₂ * x₂) ≤ 0) ∧
  -- x³ is convex for x > 0 and concave for x < 0
  (∀ (x₁ x₂ q₁ q₂ : ℝ), x₁ > 0 → x₂ > 0 → q₁ > 0 → q₂ > 0 → q₁ + q₂ = 1 →
    q₁ * x₁^3 + q₂ * x₂^3 - (q₁ * x₁ + q₂ * x₂)^3 ≥ 0) ∧
  (∀ (x₁ x₂ q₁ q₂ : ℝ), x₁ < 0 → x₂ < 0 → q₁ > 0 → q₂ > 0 → q₁ + q₂ = 1 →
    q₁ * x₁^3 + q₂ * x₂^3 - (q₁ * x₁ + q₂ * x₂)^3 ≤ 0) ∧
  -- 1/x is convex for x > 0 and concave for x < 0
  (∀ (x₁ x₂ q₁ q₂ : ℝ), x₁ > 0 → x₂ > 0 → q₁ > 0 → q₂ > 0 → q₁ + q₂ = 1 →
    q₁ / x₁ + q₂ / x₂ - 1 / (q₁ * x₁ + q₂ * x₂) ≥ 0) ∧
  (∀ (x₁ x₂ q₁ q₂ : ℝ), x₁ < 0 → x₂ < 0 → q₁ > 0 → q₂ > 0 → q₁ + q₂ = 1 →
    q₁ / x₁ + q₂ / x₂ - 1 / (q₁ * x₁ + q₂ * x₂) ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_function_convexity_concavity_l3263_326359


namespace NUMINAMATH_CALUDE_seventeen_pairs_sold_l3263_326374

/-- Represents the sales data for an optometrist's contact lens business --/
structure ContactLensSales where
  soft_price : ℝ
  hard_price : ℝ
  soft_hard_difference : ℕ
  discount_rate : ℝ
  total_sales : ℝ

/-- Calculates the total number of contact lens pairs sold given the sales data --/
def total_pairs_sold (sales : ContactLensSales) : ℕ :=
  sorry

/-- Theorem stating that given the specific sales data, 17 pairs of lenses were sold --/
theorem seventeen_pairs_sold :
  let sales := ContactLensSales.mk 175 95 7 0.1 2469
  total_pairs_sold sales = 17 := by
  sorry

end NUMINAMATH_CALUDE_seventeen_pairs_sold_l3263_326374


namespace NUMINAMATH_CALUDE_multiplication_associativity_l3263_326327

theorem multiplication_associativity (x y z : ℝ) : (x * y) * z = x * (y * z) := by
  sorry

end NUMINAMATH_CALUDE_multiplication_associativity_l3263_326327


namespace NUMINAMATH_CALUDE_minor_premise_identification_l3263_326373

-- Define the types
def Shape : Type := String

-- Define the properties
def IsRectangle (s : Shape) : Prop := s = "rectangle"
def IsParallelogram (s : Shape) : Prop := s = "parallelogram"
def IsTriangle (s : Shape) : Prop := s = "triangle"

-- Define the syllogism statements
def MajorPremise : Prop := ∀ s : Shape, IsRectangle s → IsParallelogram s
def MinorPremise : Prop := ∃ s : Shape, IsTriangle s ∧ ¬IsParallelogram s
def Conclusion : Prop := ∃ s : Shape, IsTriangle s ∧ ¬IsRectangle s

-- Theorem to prove
theorem minor_premise_identification :
  MinorPremise = (∃ s : Shape, IsTriangle s ∧ ¬IsParallelogram s) :=
by sorry

end NUMINAMATH_CALUDE_minor_premise_identification_l3263_326373


namespace NUMINAMATH_CALUDE_class_average_after_exclusion_l3263_326364

/-- Proves that given a class of 10 students with an average mark of 80,
    if 5 students with an average mark of 70 are excluded,
    the average mark of the remaining students is 90. -/
theorem class_average_after_exclusion
  (total_students : ℕ)
  (total_average : ℚ)
  (excluded_students : ℕ)
  (excluded_average : ℚ)
  (h1 : total_students = 10)
  (h2 : total_average = 80)
  (h3 : excluded_students = 5)
  (h4 : excluded_average = 70) :
  let remaining_students := total_students - excluded_students
  let total_marks := total_students * total_average
  let excluded_marks := excluded_students * excluded_average
  let remaining_marks := total_marks - excluded_marks
  remaining_marks / remaining_students = 90 := by
  sorry


end NUMINAMATH_CALUDE_class_average_after_exclusion_l3263_326364


namespace NUMINAMATH_CALUDE_power_function_through_point_l3263_326398

/-- A power function passing through (2, 1/8) has exponent -3 -/
theorem power_function_through_point (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, x > 0 → f x = x^α) →  -- f is a power function for positive x
  f 2 = 1/8 →                 -- f passes through (2, 1/8)
  α = -3 := by
sorry


end NUMINAMATH_CALUDE_power_function_through_point_l3263_326398


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l3263_326334

theorem fraction_equals_zero (x : ℝ) : (x + 1) / (x - 2) = 0 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l3263_326334


namespace NUMINAMATH_CALUDE_pages_left_to_read_l3263_326323

/-- Given a book with specified total pages, pages read, daily reading rate, and reading duration,
    calculate the number of pages left to read after the reading period. -/
theorem pages_left_to_read 
  (total_pages : ℕ) 
  (pages_read : ℕ) 
  (pages_per_day : ℕ) 
  (days : ℕ) 
  (h1 : total_pages = 381) 
  (h2 : pages_read = 149) 
  (h3 : pages_per_day = 20) 
  (h4 : days = 7) :
  total_pages - pages_read - (pages_per_day * days) = 92 := by
  sorry


end NUMINAMATH_CALUDE_pages_left_to_read_l3263_326323


namespace NUMINAMATH_CALUDE_combinatorial_identities_l3263_326396

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of k-permutations of n -/
def permutation (n k : ℕ) : ℕ := sorry

theorem combinatorial_identities :
  (∀ n k : ℕ, k > 0 → k * binomial n k = n * binomial (n - 1) (k - 1)) ∧
  binomial 2014 2013 + permutation 5 3 = 2074 := by sorry

end NUMINAMATH_CALUDE_combinatorial_identities_l3263_326396


namespace NUMINAMATH_CALUDE_a_8_equals_16_l3263_326387

def sequence_property (a : ℕ+ → ℕ) : Prop :=
  ∀ p q : ℕ+, a (p + q) = a p * a q

theorem a_8_equals_16 (a : ℕ+ → ℕ) (h1 : sequence_property a) (h2 : a 2 = 2) :
  a 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_a_8_equals_16_l3263_326387


namespace NUMINAMATH_CALUDE_sample_size_proof_l3263_326371

theorem sample_size_proof (n : ℕ) : 
  (∃ (x : ℚ), 
    x > 0 ∧ 
    2*x + 3*x + 4*x + 6*x + 4*x + x = 1 ∧ 
    (2*x + 3*x + 4*x) * n = 27) → 
  n = 60 := by
sorry

end NUMINAMATH_CALUDE_sample_size_proof_l3263_326371


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l3263_326372

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {1, 2, 3, 4}

-- Define set B
def B : Set ℝ := {x : ℝ | 2 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem intersection_complement_theorem :
  A ∩ (U \ B) = {1, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l3263_326372


namespace NUMINAMATH_CALUDE_three_digit_reverse_divisible_by_11_l3263_326311

theorem three_digit_reverse_divisible_by_11 (a b c : Nat) (ha : a ≠ 0) (hb : b < 10) (hc : c < 10) :
  ∃ k : Nat, 100001 * a + 10010 * b + 1100 * c = 11 * k := by
  sorry

end NUMINAMATH_CALUDE_three_digit_reverse_divisible_by_11_l3263_326311


namespace NUMINAMATH_CALUDE_harkamal_payment_l3263_326343

/-- The amount Harkamal paid to the shopkeeper -/
def total_amount (grape_quantity grape_rate mango_quantity mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem: Given the conditions, Harkamal paid 1010 to the shopkeeper -/
theorem harkamal_payment : total_amount 8 70 9 50 = 1010 := by
  sorry

end NUMINAMATH_CALUDE_harkamal_payment_l3263_326343


namespace NUMINAMATH_CALUDE_pen_users_count_l3263_326330

/-- Represents the number of attendants at a meeting using different writing tools -/
structure MeetingAttendants where
  pencil_users : ℕ
  single_tool_users : ℕ
  both_tools_users : ℕ

/-- Calculates the number of attendants who used a pen -/
def pen_users (m : MeetingAttendants) : ℕ :=
  m.single_tool_users + m.both_tools_users - (m.pencil_users - m.both_tools_users)

/-- Theorem stating that the number of pen users is 15 given the conditions -/
theorem pen_users_count (m : MeetingAttendants) 
  (h1 : m.pencil_users = 25)
  (h2 : m.single_tool_users = 20)
  (h3 : m.both_tools_users = 10) : 
  pen_users m = 15 := by
  sorry

#eval pen_users { pencil_users := 25, single_tool_users := 20, both_tools_users := 10 }

end NUMINAMATH_CALUDE_pen_users_count_l3263_326330


namespace NUMINAMATH_CALUDE_fraction_sum_equals_four_l3263_326306

theorem fraction_sum_equals_four : 
  (2 : ℚ) / 15 + 4 / 15 + 6 / 15 + 8 / 15 + 10 / 15 + 30 / 15 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_four_l3263_326306


namespace NUMINAMATH_CALUDE_m_fourth_plus_n_fourth_l3263_326320

theorem m_fourth_plus_n_fourth (m n : ℝ) 
  (h1 : m - n = -5)
  (h2 : m^2 + n^2 = 13) : 
  m^4 + n^4 = 97 := by
  sorry

end NUMINAMATH_CALUDE_m_fourth_plus_n_fourth_l3263_326320


namespace NUMINAMATH_CALUDE_two_books_from_different_genres_l3263_326393

/-- Represents the number of books in each genre -/
def booksPerGenre : Nat := 4

/-- Represents the number of genres -/
def numGenres : Nat := 3

/-- Theorem: The number of ways to select two books from different genres 
    given three genres with four books each is 48 -/
theorem two_books_from_different_genres :
  (booksPerGenre * booksPerGenre * (numGenres * (numGenres - 1) / 2)) = 48 := by
  sorry

end NUMINAMATH_CALUDE_two_books_from_different_genres_l3263_326393


namespace NUMINAMATH_CALUDE_product_of_exponents_l3263_326302

theorem product_of_exponents (p r s : ℕ) : 
  3^p + 3^4 = 90 → 2^r + 44 = 76 → 5^3 + 6^s = 1421 → p * r * s = 40 := by
sorry

end NUMINAMATH_CALUDE_product_of_exponents_l3263_326302


namespace NUMINAMATH_CALUDE_expression_simplification_l3263_326358

theorem expression_simplification (x : ℝ) :
  (3*x^3 + 4*x^2 + 5)*(2*x - 1) - (2*x - 1)*(x^2 + 2*x - 8) + (x^2 - 2*x + 3)*(2*x - 1)*(x - 2) =
  8*x^4 - 2*x^3 - 5*x^2 + 32*x - 15 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3263_326358


namespace NUMINAMATH_CALUDE_zeros_product_less_than_one_l3263_326370

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x / x - Real.log x + x - a

theorem zeros_product_less_than_one (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 ∧ x₂ > 0 ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ x₁ ≠ x₂ → x₁ * x₂ < 1 := by
  sorry

end NUMINAMATH_CALUDE_zeros_product_less_than_one_l3263_326370


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3263_326357

-- Define the original expression
def original_expression := (4 : ℚ) / (3 * (7 : ℚ)^(1/4))

-- Define the rationalized expression
def rationalized_expression := (4 * (343 : ℚ)^(1/4)) / 21

-- State the theorem
theorem rationalize_denominator :
  original_expression = rationalized_expression ∧
  ¬ (∃ (p : ℕ), Prime p ∧ (343 : ℕ) % p^4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3263_326357


namespace NUMINAMATH_CALUDE_problem_statement_l3263_326390

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - a^2*Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x^2 - a^2*x - a*Real.log x

theorem problem_statement :
  (∀ a : ℝ, (∃ x_min : ℝ, x_min > 0 ∧ f a x_min = 0 ∧ ∀ x : ℝ, x > 0 → f a x ≥ 0) →
    a = 1 ∨ a = -2 * Real.exp (3/4)) ∧
  (∀ a : ℝ, (∀ x : ℝ, x > 0 → g a x ≥ 0) →
    0 ≤ a ∧ a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3263_326390


namespace NUMINAMATH_CALUDE_cost_price_example_l3263_326388

/-- Given a selling price and a profit percentage, calculate the cost price -/
def cost_price (selling_price : ℚ) (profit_percentage : ℚ) : ℚ :=
  selling_price / (1 + profit_percentage / 100)

/-- Theorem: Given a selling price of 500 and a profit of 25%, the cost price is 400 -/
theorem cost_price_example : cost_price 500 25 = 400 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_example_l3263_326388


namespace NUMINAMATH_CALUDE_triangle_side_length_l3263_326367

/-- Given a triangle ABC with angle A = 60°, side b = 8, and area = 12√3,
    prove that side a = 2√13 -/
theorem triangle_side_length (A B C : ℝ) (h_angle : A = 60 * π / 180)
    (h_side_b : B = 8) (h_area : (1/2) * B * C * Real.sin A = 12 * Real.sqrt 3) :
    A = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3263_326367


namespace NUMINAMATH_CALUDE_no_extreme_points_iff_m_in_range_l3263_326395

/-- A cubic function parameterized by m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + x + 2023

/-- The derivative of f with respect to x -/
def f_deriv (m : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*m*x + 1

/-- Predicate for f having no extreme points -/
def has_no_extreme_points (m : ℝ) : Prop :=
  ∀ x : ℝ, f_deriv m x ≠ 0 ∨ (f_deriv m x = 0 ∧ ∀ y : ℝ, f_deriv m y ≥ 0 ∨ ∀ y : ℝ, f_deriv m y ≤ 0)

theorem no_extreme_points_iff_m_in_range :
  ∀ m : ℝ, has_no_extreme_points m ↔ -Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_no_extreme_points_iff_m_in_range_l3263_326395


namespace NUMINAMATH_CALUDE_gcd_problem_l3263_326399

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, a = 2345 * k) :
  Int.gcd (a^2 + 10*a + 25) (a + 5) = a + 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l3263_326399


namespace NUMINAMATH_CALUDE_max_students_is_eight_l3263_326324

/-- Represents the relation of two students knowing each other -/
def knows (n : ℕ) : (Fin n → Fin n → Prop) := sorry

/-- The property that in any group of 3 students, at least 2 know each other -/
def three_two_know (n : ℕ) (knows : Fin n → Fin n → Prop) : Prop :=
  ∀ (a b c : Fin n), a ≠ b ∧ b ≠ c ∧ a ≠ c →
    knows a b ∨ knows b c ∨ knows a c

/-- The property that in any group of 4 students, at least 2 do not know each other -/
def four_two_dont_know (n : ℕ) (knows : Fin n → Fin n → Prop) : Prop :=
  ∀ (a b c d : Fin n), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d →
    ¬(knows a b) ∨ ¬(knows a c) ∨ ¬(knows a d) ∨
    ¬(knows b c) ∨ ¬(knows b d) ∨ ¬(knows c d)

/-- The maximum number of students satisfying the conditions is 8 -/
theorem max_students_is_eight :
  (∃ (n : ℕ), n = 8 ∧
    three_two_know n (knows n) ∧
    four_two_dont_know n (knows n)) ∧
  (∀ (m : ℕ), m > 8 →
    ¬(three_two_know m (knows m) ∧
      four_two_dont_know m (knows m))) :=
by sorry

end NUMINAMATH_CALUDE_max_students_is_eight_l3263_326324


namespace NUMINAMATH_CALUDE_heather_walk_distance_l3263_326315

/-- The total distance Heather walked at the county fair -/
def total_distance (d1 d2 d3 : ℝ) : ℝ := d1 + d2 + d3

/-- Theorem stating the total distance Heather walked -/
theorem heather_walk_distance :
  let d1 : ℝ := 0.33  -- Distance from car to entrance
  let d2 : ℝ := 0.33  -- Distance to carnival rides
  let d3 : ℝ := 0.08  -- Distance from carnival rides back to car
  total_distance d1 d2 d3 = 0.74 := by
  sorry

end NUMINAMATH_CALUDE_heather_walk_distance_l3263_326315


namespace NUMINAMATH_CALUDE_particle_motion_l3263_326389

/-- Height of the particle in meters after t seconds -/
def s (t : ℝ) : ℝ := 180 * t - 18 * t^2

/-- Time at which the particle reaches its highest point -/
def t_max : ℝ := 5

/-- The highest elevation reached by the particle -/
def h_max : ℝ := 450

theorem particle_motion :
  (∀ t : ℝ, s t ≤ h_max) ∧
  s t_max = h_max :=
sorry

end NUMINAMATH_CALUDE_particle_motion_l3263_326389


namespace NUMINAMATH_CALUDE_cake_cross_section_is_rectangle_l3263_326382

/-- A cylindrical cake -/
structure Cake where
  base_diameter : ℝ
  height : ℝ

/-- The cross-section of a cake when cut along its diameter -/
inductive CrossSection
  | Rectangle
  | Circle
  | Square
  | Undetermined

/-- The shape of the cross-section when a cylindrical cake is cut along its diameter -/
def cross_section_shape (c : Cake) : CrossSection :=
  CrossSection.Rectangle

/-- Theorem: The cross-section of a cylindrical cake with base diameter 3 cm and height 9 cm, 
    when cut along its diameter, is a rectangle -/
theorem cake_cross_section_is_rectangle :
  let c : Cake := { base_diameter := 3, height := 9 }
  cross_section_shape c = CrossSection.Rectangle := by
  sorry

end NUMINAMATH_CALUDE_cake_cross_section_is_rectangle_l3263_326382


namespace NUMINAMATH_CALUDE_class_average_l3263_326313

theorem class_average (total_students : ℕ) (top_scorers : ℕ) (top_score : ℕ) (zero_scorers : ℕ) (rest_average : ℕ) :
  total_students = 25 →
  top_scorers = 3 →
  top_score = 95 →
  zero_scorers = 5 →
  rest_average = 45 →
  (top_scorers * top_score + zero_scorers * 0 + (total_students - top_scorers - zero_scorers) * rest_average) / total_students = 42 := by
  sorry

end NUMINAMATH_CALUDE_class_average_l3263_326313


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3263_326339

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 2 + a 5 + a 8 + a 11 = 48 →
  a 6 + a 7 = 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3263_326339


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l3263_326361

theorem perfect_square_polynomial (g : ℕ) : 
  (∃ k : ℕ, g^4 + g^3 + g^2 + g + 1 = k^2) → g = 3 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l3263_326361


namespace NUMINAMATH_CALUDE_largest_non_representable_integer_l3263_326345

theorem largest_non_representable_integer
  (a b c : ℕ+) 
  (coprime_ab : Nat.Coprime a b)
  (coprime_bc : Nat.Coprime b c)
  (coprime_ca : Nat.Coprime c a) :
  ¬ ∃ (x y z : ℕ), 2 * a * b * c - a * b - b * c - c * a = x * b * c + y * c * a + z * a * b :=
sorry

end NUMINAMATH_CALUDE_largest_non_representable_integer_l3263_326345


namespace NUMINAMATH_CALUDE_one_minus_repeating_eight_eq_one_ninth_l3263_326319

/-- The value of 0.888... (repeating decimal) -/
def repeating_eight : ℚ := 8/9

/-- Proof that 1 - 0.888... = 1/9 -/
theorem one_minus_repeating_eight_eq_one_ninth :
  1 - repeating_eight = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_one_minus_repeating_eight_eq_one_ninth_l3263_326319


namespace NUMINAMATH_CALUDE_eldorado_license_plates_l3263_326378

/-- The number of vowels that can be used as the first letter of a license plate. -/
def numVowels : ℕ := 5

/-- The number of letters in the alphabet. -/
def numLetters : ℕ := 26

/-- The number of digits (0-9). -/
def numDigits : ℕ := 10

/-- The total number of valid license plates in Eldorado. -/
def totalLicensePlates : ℕ := numVowels * numLetters * numLetters * numDigits * numDigits

theorem eldorado_license_plates :
  totalLicensePlates = 338000 :=
by sorry

end NUMINAMATH_CALUDE_eldorado_license_plates_l3263_326378


namespace NUMINAMATH_CALUDE_policeman_can_catch_gangster_l3263_326354

/-- Represents a point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square --/
structure Square where
  sideLength : ℝ
  center : Point
  
/-- Represents the policeman --/
structure Policeman where
  position : Point
  speed : ℝ

/-- Represents the gangster --/
structure Gangster where
  position : Point
  speed : ℝ

/-- A function to check if a point is on the edge of a square --/
def isOnEdge (p : Point) (s : Square) : Prop :=
  (p.x = s.center.x - s.sideLength / 2 ∨ p.x = s.center.x + s.sideLength / 2) ∨
  (p.y = s.center.y - s.sideLength / 2 ∨ p.y = s.center.y + s.sideLength / 2)

/-- The main theorem --/
theorem policeman_can_catch_gangster 
  (s : Square) 
  (p : Policeman) 
  (g : Gangster) 
  (h1 : s.sideLength > 0)
  (h2 : p.position = s.center)
  (h3 : isOnEdge g.position s)
  (h4 : p.speed = g.speed / 2)
  (h5 : g.speed > 0) :
  ∃ (t : ℝ) (pFinal gFinal : Point), 
    t ≥ 0 ∧
    isOnEdge pFinal s ∧
    isOnEdge gFinal s ∧
    ∃ (edge : Set Point), 
      edge.Subset {p | isOnEdge p s} ∧
      pFinal ∈ edge ∧
      gFinal ∈ edge :=
by sorry

end NUMINAMATH_CALUDE_policeman_can_catch_gangster_l3263_326354


namespace NUMINAMATH_CALUDE_line_parallel_to_x_axis_line_through_1_2_parallel_to_x_axis_l3263_326375

/-- A line parallel to the x-axis passing through a point (x₀, y₀) has the equation y = y₀ -/
theorem line_parallel_to_x_axis (x₀ y₀ : ℝ) :
  let line := {(x, y) : ℝ × ℝ | y = y₀}
  (∀ (x : ℝ), (x, y₀) ∈ line) ∧ ((x₀, y₀) ∈ line) → 
  ∀ (x y : ℝ), (x, y) ∈ line ↔ y = y₀ :=
by sorry

/-- The equation of the line passing through (1,2) and parallel to the x-axis is y = 2 -/
theorem line_through_1_2_parallel_to_x_axis :
  let line := {(x, y) : ℝ × ℝ | y = 2}
  (∀ (x : ℝ), (x, 2) ∈ line) ∧ ((1, 2) ∈ line) → 
  ∀ (x y : ℝ), (x, y) ∈ line ↔ y = 2 :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_to_x_axis_line_through_1_2_parallel_to_x_axis_l3263_326375


namespace NUMINAMATH_CALUDE_root_of_f_l3263_326368

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- State that f_inv is indeed the inverse of f
axiom inverse_relation : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- Given condition: f⁻¹(0) = 2
axiom inverse_intersect_y : f_inv 0 = 2

-- Theorem to prove
theorem root_of_f (h : f_inv 0 = 2) : f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_of_f_l3263_326368


namespace NUMINAMATH_CALUDE_faye_pencils_l3263_326380

/-- The number of rows of pencils and crayons -/
def num_rows : ℕ := 30

/-- The number of pencils in each row -/
def pencils_per_row : ℕ := 24

/-- The total number of pencils -/
def total_pencils : ℕ := num_rows * pencils_per_row

theorem faye_pencils : total_pencils = 720 := by
  sorry

end NUMINAMATH_CALUDE_faye_pencils_l3263_326380


namespace NUMINAMATH_CALUDE_trigonometric_sum_l3263_326301

theorem trigonometric_sum (x : ℝ) : 
  (Real.cos x + Real.cos (x + 2 * Real.pi / 3) + Real.cos (x + 4 * Real.pi / 3) = 0) ∧
  (Real.sin x + Real.sin (x + 2 * Real.pi / 3) + Real.sin (x + 4 * Real.pi / 3) = 0) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_sum_l3263_326301


namespace NUMINAMATH_CALUDE_specific_cube_surface_area_l3263_326340

/-- Represents the heights of cuts in the cube -/
structure CutHeights where
  h1 : ℝ
  h2 : ℝ
  h3 : ℝ

/-- Calculates the total surface area of a stacked solid formed from a cube -/
def totalSurfaceArea (cubeSideLength : ℝ) (cuts : CutHeights) : ℝ :=
  sorry

/-- Theorem stating the total surface area of the specific cut and stacked cube -/
theorem specific_cube_surface_area :
  let cubeSideLength : ℝ := 2
  let cuts : CutHeights := { h1 := 1/4, h2 := 1/4 + 1/5, h3 := 1/4 + 1/5 + 1/8 }
  totalSurfaceArea cubeSideLength cuts = 12 := by
  sorry

end NUMINAMATH_CALUDE_specific_cube_surface_area_l3263_326340


namespace NUMINAMATH_CALUDE_tangent_line_curve1_tangent_lines_curve2_l3263_326344

-- Define the curves
def curve1 (x : ℝ) : ℝ := x^3 + x^2 + 1
def curve2 (x : ℝ) : ℝ := x^2

-- Define the points
def P1 : ℝ × ℝ := (-1, 1)
def P2 : ℝ × ℝ := (3, 5)

-- Theorem for the first curve
theorem tangent_line_curve1 :
  ∃ (k m : ℝ), k * P1.1 + m * P1.2 + 2 = 0 ∧
  ∀ x y, y = curve1 x → k * x + m * y + 2 = 0 → x = P1.1 ∧ y = P1.2 :=
sorry

-- Theorem for the second curve
theorem tangent_lines_curve2 :
  ∃ (k1 m1 k2 m2 : ℝ),
  (k1 * P2.1 + m1 * P2.2 + 1 = 0 ∧ k2 * P2.1 + m2 * P2.2 + 25 = 0) ∧
  (∀ x y, y = curve2 x → (k1 * x + m1 * y + 1 = 0 ∨ k2 * x + m2 * y + 25 = 0) → x = P2.1 ∧ y = P2.2) ∧
  (k1 = 2 ∧ m1 = -1) ∧ (k2 = 10 ∧ m2 = -1) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_curve1_tangent_lines_curve2_l3263_326344


namespace NUMINAMATH_CALUDE_arrange_six_books_two_pairs_l3263_326377

/-- The number of ways to arrange books with some identical copies -/
def arrange_books (total : ℕ) (identical_pairs : ℕ) (unique : ℕ) : ℕ :=
  Nat.factorial total / (Nat.factorial 2 ^ identical_pairs)

/-- Theorem: Arranging 6 books with 2 identical pairs and 2 unique books -/
theorem arrange_six_books_two_pairs : arrange_books 6 2 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_arrange_six_books_two_pairs_l3263_326377


namespace NUMINAMATH_CALUDE_negation_of_universal_absolute_value_l3263_326326

theorem negation_of_universal_absolute_value :
  (¬ ∀ x : ℝ, x = |x|) ↔ (∃ x : ℝ, x ≠ |x|) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_absolute_value_l3263_326326


namespace NUMINAMATH_CALUDE_inverse_g_solution_l3263_326317

noncomputable section

variables (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0)

def g (x : ℝ) : ℝ := 1 / (c * x + d)

theorem inverse_g_solution :
  let x := 1 / (-2 * c + d)
  g x = 1 / 2 := by sorry

end

end NUMINAMATH_CALUDE_inverse_g_solution_l3263_326317


namespace NUMINAMATH_CALUDE_expand_and_compare_coefficients_l3263_326351

theorem expand_and_compare_coefficients (m n : ℤ) : 
  (∀ x : ℤ, (x + 4) * (x - 2) = x^2 + m*x + n) → m = 2 ∧ n = -8 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_compare_coefficients_l3263_326351


namespace NUMINAMATH_CALUDE_tenby_position_l3263_326322

def letters : List Char := ['B', 'E', 'N', 'T', 'Y']

def word : String := "TENBY"

def alphabetical_position (w : String) (l : List Char) : ℕ :=
  sorry

theorem tenby_position :
  alphabetical_position word letters = 75 := by
  sorry

end NUMINAMATH_CALUDE_tenby_position_l3263_326322


namespace NUMINAMATH_CALUDE_lcm_of_prime_and_nonmultiple_lcm_1227_40_l3263_326349

theorem lcm_of_prime_and_nonmultiple (p n : ℕ) (h_prime : Nat.Prime p) (h_not_dvd : ¬p ∣ n) :
  Nat.lcm p n = p * n :=
by sorry

theorem lcm_1227_40 :
  Nat.lcm 1227 40 = 49080 :=
by sorry

end NUMINAMATH_CALUDE_lcm_of_prime_and_nonmultiple_lcm_1227_40_l3263_326349


namespace NUMINAMATH_CALUDE_worker_number_40th_segment_l3263_326347

/-- Calculates the individual number of a worker in systematic sampling -/
def systematicSamplingNumber (totalStaff : ℕ) (segments : ℕ) (startNumber : ℕ) (segmentIndex : ℕ) : ℕ :=
  startNumber + (segmentIndex - 1) * (totalStaff / segments)

/-- Proves that the individual number of the worker from the 40th segment is 394 -/
theorem worker_number_40th_segment :
  systematicSamplingNumber 620 62 4 40 = 394 := by
  sorry

#eval systematicSamplingNumber 620 62 4 40

end NUMINAMATH_CALUDE_worker_number_40th_segment_l3263_326347


namespace NUMINAMATH_CALUDE_solve_stamp_problem_l3263_326366

def stamp_problem (initial_stamps final_stamps mike_stamps : ℕ) : Prop :=
  let harry_stamps := final_stamps - initial_stamps - mike_stamps
  harry_stamps - 2 * mike_stamps = 10

theorem solve_stamp_problem :
  stamp_problem 3000 3061 17 := by
  sorry

end NUMINAMATH_CALUDE_solve_stamp_problem_l3263_326366


namespace NUMINAMATH_CALUDE_inequality_proof_l3263_326376

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 1) :
  x^12 - y^12 + 2 * x^6 * y^6 ≤ π / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3263_326376


namespace NUMINAMATH_CALUDE_point_on_angle_negative_pi_third_l3263_326386

/-- Given a point P(2,y) on the terminal side of angle -π/3, prove that y = -2√3 -/
theorem point_on_angle_negative_pi_third (y : ℝ) : 
  (∃ P : ℝ × ℝ, P.1 = 2 ∧ P.2 = y ∧ P.2 / P.1 = Real.tan (-π/3)) → 
  y = -2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_angle_negative_pi_third_l3263_326386


namespace NUMINAMATH_CALUDE_parallel_line_equation_l3263_326341

/-- A line parallel to y = -4x + 2023 that intersects the y-axis at (0, -5) has the equation y = -4x - 5 -/
theorem parallel_line_equation (k b : ℝ) : 
  (∀ x y, y = k * x + b ↔ y = -4 * x + 2023) →  -- parallel condition
  (b = -5) →                                   -- y-intercept condition
  (∀ x y, y = k * x + b ↔ y = -4 * x - 5) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l3263_326341


namespace NUMINAMATH_CALUDE_star_calculation_l3263_326342

/-- The ⋆ operation for real numbers -/
def star (x y : ℝ) : ℝ := (x + y) * (x - y)

/-- Theorem stating that 3 ⋆ (5 ⋆ 6) = -112 -/
theorem star_calculation : star 3 (star 5 6) = -112 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l3263_326342


namespace NUMINAMATH_CALUDE_monomial_product_l3263_326312

theorem monomial_product (a b : ℤ) (x y : ℝ) (h1 : 4 * a - b = 2) (h2 : a + b = 3) :
  (-2 * x^(4*a-b) * y^3) * ((1/2) * x^2 * y^(a+b)) = -x^4 * y^6 := by
  sorry

end NUMINAMATH_CALUDE_monomial_product_l3263_326312


namespace NUMINAMATH_CALUDE_library_books_before_grant_l3263_326307

/-- The number of books purchased with the grant -/
def books_purchased : ℕ := 2647

/-- The total number of books after the grant -/
def total_books : ℕ := 8582

/-- The number of books before the grant -/
def books_before : ℕ := total_books - books_purchased

theorem library_books_before_grant : books_before = 5935 := by
  sorry

end NUMINAMATH_CALUDE_library_books_before_grant_l3263_326307


namespace NUMINAMATH_CALUDE_lcm_of_ratio_numbers_l3263_326308

theorem lcm_of_ratio_numbers (a b : ℕ) (h1 : a = 48) (h2 : b * 8 = a * 9) : 
  Nat.lcm a b = 432 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_numbers_l3263_326308


namespace NUMINAMATH_CALUDE_regular_decagon_exterior_angle_regular_decagon_exterior_angle_is_36_l3263_326362

/-- The exterior angle of a regular decagon is 36 degrees. -/
theorem regular_decagon_exterior_angle : ℝ :=
  let n : ℕ := 10  -- number of sides in a decagon
  let interior_angle_sum : ℝ := 180 * (n - 2)
  let interior_angle : ℝ := interior_angle_sum / n
  let exterior_angle : ℝ := 180 - interior_angle
  exterior_angle

/-- Proof that the exterior angle of a regular decagon is 36 degrees. -/
theorem regular_decagon_exterior_angle_is_36 : 
  regular_decagon_exterior_angle = 36 := by
  sorry

end NUMINAMATH_CALUDE_regular_decagon_exterior_angle_regular_decagon_exterior_angle_is_36_l3263_326362


namespace NUMINAMATH_CALUDE_quadratic_completing_square_l3263_326336

/-- Given a quadratic equation 16x^2 - 32x - 512 = 0, when transformed
    to the form (x + p)^2 = q, the value of q is 33. -/
theorem quadratic_completing_square :
  ∃ (p : ℝ), ∀ (x : ℝ),
    16 * x^2 - 32 * x - 512 = 0 ↔ (x + p)^2 = 33 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_l3263_326336


namespace NUMINAMATH_CALUDE_ball_bearing_savings_ball_bearing_savings_correct_l3263_326329

/-- Calculates the savings when buying ball bearings during a sale with a bulk discount -/
theorem ball_bearing_savings
  (num_machines : ℕ)
  (bearings_per_machine : ℕ)
  (regular_price : ℚ)
  (sale_price : ℚ)
  (bulk_discount : ℚ)
  (h1 : num_machines = 10)
  (h2 : bearings_per_machine = 30)
  (h3 : regular_price = 1)
  (h4 : sale_price = 3/4)
  (h5 : bulk_discount = 1/5)
  : ℚ :=
  let total_bearings := num_machines * bearings_per_machine
  let regular_cost := total_bearings * regular_price
  let sale_cost := total_bearings * sale_price
  let discounted_cost := sale_cost * (1 - bulk_discount)
  let savings := regular_cost - discounted_cost
  120

theorem ball_bearing_savings_correct : ball_bearing_savings 10 30 1 (3/4) (1/5) rfl rfl rfl rfl rfl = 120 := by
  sorry

end NUMINAMATH_CALUDE_ball_bearing_savings_ball_bearing_savings_correct_l3263_326329


namespace NUMINAMATH_CALUDE_product_of_square_roots_l3263_326392

theorem product_of_square_roots (q : ℝ) (hq : q ≥ 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q) = 126 * q * Real.sqrt q :=
by sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l3263_326392


namespace NUMINAMATH_CALUDE_intersection_point_l3263_326383

/-- The slope of the first line -/
def m₁ : ℚ := 3

/-- The y-intercept of the first line -/
def b₁ : ℚ := -2

/-- The x-coordinate of the given point -/
def x₀ : ℚ := 2

/-- The y-coordinate of the given point -/
def y₀ : ℚ := 2

/-- The slope of the perpendicular line -/
def m₂ : ℚ := -1 / m₁

/-- The y-intercept of the perpendicular line -/
def b₂ : ℚ := y₀ - m₂ * x₀

/-- The x-coordinate of the intersection point -/
def x_intersect : ℚ := (b₂ - b₁) / (m₁ - m₂)

/-- The y-coordinate of the intersection point -/
def y_intersect : ℚ := m₁ * x_intersect + b₁

theorem intersection_point :
  (x_intersect = 7/5) ∧ (y_intersect = 11/5) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l3263_326383


namespace NUMINAMATH_CALUDE_rent_expenditure_l3263_326328

def monthly_salary : ℕ := 18000
def savings_percentage : ℚ := 1/10
def savings : ℕ := 1800
def milk_expense : ℕ := 1500
def groceries_expense : ℕ := 4500
def education_expense : ℕ := 2500
def petrol_expense : ℕ := 2000
def misc_expense : ℕ := 700

theorem rent_expenditure :
  let total_expenses := milk_expense + groceries_expense + education_expense + petrol_expense + misc_expense
  let rent := monthly_salary - (total_expenses + savings)
  (savings = (savings_percentage * monthly_salary).num) →
  rent = 6000 := by sorry

end NUMINAMATH_CALUDE_rent_expenditure_l3263_326328


namespace NUMINAMATH_CALUDE_correct_sums_count_l3263_326355

theorem correct_sums_count (total : ℕ) (correct : ℕ) (incorrect : ℕ) : 
  total = 75 → 
  incorrect = 2 * correct → 
  total = correct + incorrect →
  correct = 25 := by
sorry

end NUMINAMATH_CALUDE_correct_sums_count_l3263_326355


namespace NUMINAMATH_CALUDE_b_win_probability_l3263_326356

/-- Represents the outcome of a single die roll -/
def DieRoll := Fin 6

/-- Represents the state of the game after each roll -/
structure GameState where
  rolls : List DieRoll
  turn : Bool  -- true for A's turn, false for B's turn

/-- Checks if a number is a multiple of 2 -/
def isMultipleOf2 (n : ℕ) : Bool := n % 2 = 0

/-- Checks if a number is a multiple of 3 -/
def isMultipleOf3 (n : ℕ) : Bool := n % 3 = 0

/-- Sums the last n rolls in the game state -/
def sumLastNRolls (state : GameState) (n : ℕ) : ℕ :=
  (state.rolls.take n).map (fun x => x.val + 1) |>.sum

/-- Determines if the game has ended and who the winner is -/
def gameResult (state : GameState) : Option Bool :=
  if state.rolls.length < 2 then
    none
  else if state.rolls.length < 3 then
    if isMultipleOf3 (sumLastNRolls state 2) then some false else none
  else
    let lastThreeSum := sumLastNRolls state 3
    let lastTwoSum := sumLastNRolls state 2
    if isMultipleOf2 lastThreeSum && !isMultipleOf3 lastTwoSum then
      some true  -- A wins
    else if isMultipleOf3 lastTwoSum && !isMultipleOf2 lastThreeSum then
      some false  -- B wins
    else
      none  -- Game continues

/-- The probability that player B wins the game -/
def probabilityBWins : ℚ := 5/9

theorem b_win_probability :
  probabilityBWins = 5/9 := by sorry

end NUMINAMATH_CALUDE_b_win_probability_l3263_326356


namespace NUMINAMATH_CALUDE_min_value_inequality_l3263_326385

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((x^2 + 4*y^2) * (2*x^2 + 3*y^2)).sqrt) / (x*y) ≥ 2 * Real.sqrt (2 * Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_min_value_inequality_l3263_326385


namespace NUMINAMATH_CALUDE_chlorine_original_cost_l3263_326331

/-- The original cost of a liter of chlorine -/
def chlorine_cost : ℝ := sorry

/-- The sale price of chlorine as a percentage of its original price -/
def chlorine_sale_percent : ℝ := 0.80

/-- The original price of a box of soap -/
def soap_original_price : ℝ := 16

/-- The sale price of a box of soap -/
def soap_sale_price : ℝ := 12

/-- The number of liters of chlorine bought -/
def chlorine_quantity : ℕ := 3

/-- The number of boxes of soap bought -/
def soap_quantity : ℕ := 5

/-- The total savings when buying chlorine and soap at sale prices -/
def total_savings : ℝ := 26

theorem chlorine_original_cost :
  chlorine_cost = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_chlorine_original_cost_l3263_326331


namespace NUMINAMATH_CALUDE_g_behavior_at_infinity_l3263_326338

def g (x : ℝ) : ℝ := -3 * x^4 + 15 * x^2 - 10

theorem g_behavior_at_infinity :
  (∀ ε > 0, ∃ N > 0, ∀ x : ℝ, x > N → g x < -ε) ∧
  (∀ ε > 0, ∃ N > 0, ∀ x : ℝ, x < -N → g x < -ε) :=
sorry

end NUMINAMATH_CALUDE_g_behavior_at_infinity_l3263_326338


namespace NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l3263_326325

theorem quadratic_polynomial_satisfies_conditions :
  ∃ p : ℝ → ℝ,
    (∀ x, p x = x^2 + 1) ∧
    p (-3) = 10 ∧
    p 0 = 1 ∧
    p 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l3263_326325


namespace NUMINAMATH_CALUDE_remainder_of_sum_squares_plus_20_l3263_326346

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem remainder_of_sum_squares_plus_20 : 
  (sum_of_squares 15 + 20) % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_sum_squares_plus_20_l3263_326346


namespace NUMINAMATH_CALUDE_trig_ratio_problem_l3263_326391

theorem trig_ratio_problem (a : ℝ) (h : 2 * Real.sin a = 3 * Real.cos a) :
  (4 * Real.sin a + Real.cos a) / (5 * Real.sin a - 2 * Real.cos a) = 14 / 11 := by
  sorry

end NUMINAMATH_CALUDE_trig_ratio_problem_l3263_326391


namespace NUMINAMATH_CALUDE_inequality_proof_l3263_326363

theorem inequality_proof (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  (1/a) + (1/b) + (9/c) + (25/d) ≥ 100/(a + b + c + d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3263_326363


namespace NUMINAMATH_CALUDE_largest_non_sum_of_composites_l3263_326314

def isComposite (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < n ∧ n % k = 0

def isSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, isComposite a ∧ isComposite b ∧ n = a + b

theorem largest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → isSumOfTwoComposites n) ∧
  ¬isSumOfTwoComposites 11 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_composites_l3263_326314


namespace NUMINAMATH_CALUDE_lines_parallel_l3263_326365

/-- Two lines in the plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope ∧ l1.intercept ≠ l2.intercept

theorem lines_parallel : 
  let line1 : Line := ⟨-1, 0⟩
  let line2 : Line := ⟨-1, 6⟩
  parallel line1 line2 := by
  sorry

end NUMINAMATH_CALUDE_lines_parallel_l3263_326365


namespace NUMINAMATH_CALUDE_tan_570_degrees_l3263_326309

theorem tan_570_degrees : Real.tan (570 * π / 180) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_570_degrees_l3263_326309


namespace NUMINAMATH_CALUDE_arithmetic_computation_l3263_326397

theorem arithmetic_computation : 143 - 13 + 31 + 17 = 178 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l3263_326397
