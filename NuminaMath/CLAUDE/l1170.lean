import Mathlib

namespace NUMINAMATH_CALUDE_red_balls_count_l1170_117021

/-- Given a bag of balls with red and yellow colors, prove that the number of red balls is 6 -/
theorem red_balls_count (total_balls : ℕ) (prob_red : ℚ) : 
  total_balls = 15 → prob_red = 2/5 → (prob_red * total_balls : ℚ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l1170_117021


namespace NUMINAMATH_CALUDE_square_side_length_l1170_117044

theorem square_side_length (rectangle_width : ℝ) (rectangle_length : ℝ) (square_side : ℝ) : 
  rectangle_width = 4 →
  rectangle_length = 16 →
  square_side ^ 2 = rectangle_width * rectangle_length →
  square_side = 8 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l1170_117044


namespace NUMINAMATH_CALUDE_sarah_shirts_l1170_117041

/-- The total number of shirts Sarah owns after buying new shirts -/
theorem sarah_shirts (initial_shirts new_shirts : ℕ) 
  (h1 : initial_shirts = 9)
  (h2 : new_shirts = 8) : 
  initial_shirts + new_shirts = 17 := by
  sorry

end NUMINAMATH_CALUDE_sarah_shirts_l1170_117041


namespace NUMINAMATH_CALUDE_lcm_of_9_12_15_l1170_117014

theorem lcm_of_9_12_15 : Nat.lcm 9 (Nat.lcm 12 15) = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_9_12_15_l1170_117014


namespace NUMINAMATH_CALUDE_phone_bill_minutes_l1170_117022

def monthly_fee : ℚ := 2
def per_minute_rate : ℚ := 12 / 100
def total_bill : ℚ := 2336 / 100

theorem phone_bill_minutes : 
  ∃ (minutes : ℕ), 
    (monthly_fee + per_minute_rate * minutes) = total_bill ∧ 
    minutes = 178 := by
  sorry

end NUMINAMATH_CALUDE_phone_bill_minutes_l1170_117022


namespace NUMINAMATH_CALUDE_alloy_interchange_mass_l1170_117056

theorem alloy_interchange_mass (m₁ m₂ x : ℝ) : 
  m₁ = 6 →
  m₂ = 12 →
  0 < x →
  x < m₁ →
  x < m₂ →
  x / m₁ = (m₂ - x) / m₂ →
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_alloy_interchange_mass_l1170_117056


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1170_117001

theorem simplify_and_evaluate (x : ℝ) (h : x = 3) :
  5 * x^2 - 7 * x - (3 * x^2 - 2 * (-x^2 + 4 * x - 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1170_117001


namespace NUMINAMATH_CALUDE_bad_oranges_l1170_117027

theorem bad_oranges (total_oranges : ℕ) (num_students : ℕ) (reduction : ℕ) : 
  total_oranges = 108 →
  num_students = 12 →
  reduction = 3 →
  (total_oranges / num_students - reduction) * num_students = total_oranges - 36 :=
by sorry

end NUMINAMATH_CALUDE_bad_oranges_l1170_117027


namespace NUMINAMATH_CALUDE_school_population_theorem_l1170_117055

theorem school_population_theorem :
  ∀ (boys girls : ℕ),
  boys + girls = 300 →
  girls = (boys * 100) / 300 →
  boys = 225 := by
sorry

end NUMINAMATH_CALUDE_school_population_theorem_l1170_117055


namespace NUMINAMATH_CALUDE_exponent_problem_l1170_117083

theorem exponent_problem (a m n : ℕ) (h1 : a^m = 3) (h2 : a^n = 5) : a^(2*m + n) = 45 := by
  sorry

end NUMINAMATH_CALUDE_exponent_problem_l1170_117083


namespace NUMINAMATH_CALUDE_thirteen_people_evaluations_l1170_117087

/-- The number of evaluations for a group of people, where each pair is categorized into one of three categories. -/
def num_evaluations (n : ℕ) : ℕ := n.choose 2 * 3

/-- Theorem: For a group of 13 people, where each pair is categorized into one of three categories, the total number of evaluations is 234. -/
theorem thirteen_people_evaluations : num_evaluations 13 = 234 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_people_evaluations_l1170_117087


namespace NUMINAMATH_CALUDE_transformations_of_f_l1170_117058

def f (x : ℝ) : ℝ := 3 * x + 4

def shift_left_down (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = f (x + 1) - 2

def reflect_y_axis (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = f (-x)

def reflect_y_eq_1 (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = 2 - f x

def reflect_y_eq_neg_x (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = (x + 4) / 3

def reflect_point (g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, g x = f (2 * a - x) + 2 * (b - a)

theorem transformations_of_f :
  (∃ g : ℝ → ℝ, shift_left_down g ∧ (∀ x, g x = 3 * x + 5)) ∧
  (∃ g : ℝ → ℝ, reflect_y_axis g ∧ (∀ x, g x = -3 * x + 4)) ∧
  (∃ g : ℝ → ℝ, reflect_y_eq_1 g ∧ (∀ x, g x = -3 * x - 2)) ∧
  (∃ g : ℝ → ℝ, reflect_y_eq_neg_x g) ∧
  (∀ a b : ℝ, ∃ g : ℝ → ℝ, reflect_point g a b ∧ (∀ x, g x = 3 * x + 2 * b - 6 * a - 4)) :=
sorry

end NUMINAMATH_CALUDE_transformations_of_f_l1170_117058


namespace NUMINAMATH_CALUDE_tangent_line_at_point_2_neg6_l1170_117059

-- Define the function f(x) = x³ + x - 16
def f (x : ℝ) : ℝ := x^3 + x - 16

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_line_at_point_2_neg6 :
  let x₀ : ℝ := 2
  let y₀ : ℝ := -6
  let m : ℝ := f' x₀
  (∀ x y, y - y₀ = m * (x - x₀) ↔ 13 * x - y - 32 = 0) ∧
  f x₀ = y₀ ∧
  m = 13 := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_2_neg6_l1170_117059


namespace NUMINAMATH_CALUDE_answer_key_problem_l1170_117093

theorem answer_key_problem (total_ways : ℕ) (tf_questions : ℕ) (mc_questions : ℕ) : 
  total_ways = 384 → 
  tf_questions = 3 → 
  mc_questions = 3 → 
  (∃ (n : ℕ), total_ways = 6 * n^mc_questions) →
  (∃ (n : ℕ), n = 4 ∧ total_ways = 6 * n^mc_questions) := by
sorry

end NUMINAMATH_CALUDE_answer_key_problem_l1170_117093


namespace NUMINAMATH_CALUDE_cube_inequality_l1170_117051

theorem cube_inequality (a b : ℝ) : a > b → a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l1170_117051


namespace NUMINAMATH_CALUDE_negation_of_p_l1170_117010

open Real

def p : Prop := ∃ x : ℚ, 2^(x : ℝ) - log x < 2

theorem negation_of_p : ¬p ↔ ∀ x : ℚ, 2^(x : ℝ) - log x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_negation_of_p_l1170_117010


namespace NUMINAMATH_CALUDE_children_percentage_l1170_117072

def total_passengers : ℕ := 60
def adult_passengers : ℕ := 45

theorem children_percentage : 
  (total_passengers - adult_passengers) / total_passengers * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_children_percentage_l1170_117072


namespace NUMINAMATH_CALUDE_parabola_intercepts_sum_l1170_117096

theorem parabola_intercepts_sum (d e f : ℝ) : 
  (∀ x, 3 * x^2 - 9 * x + 5 = 3 * 0^2 - 9 * 0 + 5 → d = 3 * 0^2 - 9 * 0 + 5) →
  (3 * e^2 - 9 * e + 5 = 0) →
  (3 * f^2 - 9 * f + 5 = 0) →
  d + e + f = 8 := by
sorry

end NUMINAMATH_CALUDE_parabola_intercepts_sum_l1170_117096


namespace NUMINAMATH_CALUDE_magnitude_relationship_l1170_117008

theorem magnitude_relationship : 
  let a := Real.sin (5 * Real.pi / 7)
  let b := Real.cos (2 * Real.pi / 7)
  let c := Real.tan (2 * Real.pi / 7)
  b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_magnitude_relationship_l1170_117008


namespace NUMINAMATH_CALUDE_count_multiples_of_12_and_9_l1170_117005

def count_multiples (lower upper divisor : ℕ) : ℕ :=
  (upper / divisor) - ((lower - 1) / divisor)

theorem count_multiples_of_12_and_9 : 
  count_multiples 50 400 (Nat.lcm 12 9) = 10 := by
  sorry

end NUMINAMATH_CALUDE_count_multiples_of_12_and_9_l1170_117005


namespace NUMINAMATH_CALUDE_carrots_not_used_l1170_117054

theorem carrots_not_used (total : ℕ) (before_lunch_fraction : ℚ) (end_of_day_fraction : ℚ) : 
  total = 300 →
  before_lunch_fraction = 2 / 5 →
  end_of_day_fraction = 3 / 5 →
  (total - (before_lunch_fraction * total).num - (end_of_day_fraction * (total - (before_lunch_fraction * total).num)).num) = 72 := by
  sorry

end NUMINAMATH_CALUDE_carrots_not_used_l1170_117054


namespace NUMINAMATH_CALUDE_decimal_addition_l1170_117016

theorem decimal_addition : (0.9 : ℝ) + 0.09 = 0.99 := by
  sorry

end NUMINAMATH_CALUDE_decimal_addition_l1170_117016


namespace NUMINAMATH_CALUDE_marbles_lost_fraction_l1170_117029

theorem marbles_lost_fraction (initial_marbles : ℕ) (additional_marbles : ℕ) (new_marbles : ℕ) (final_marbles : ℕ)
  (h1 : initial_marbles = 12)
  (h2 : additional_marbles = 10)
  (h3 : new_marbles = 25)
  (h4 : final_marbles = 41) :
  (initial_marbles - (final_marbles - additional_marbles - new_marbles)) / initial_marbles = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_marbles_lost_fraction_l1170_117029


namespace NUMINAMATH_CALUDE_power_17_mod_28_l1170_117052

theorem power_17_mod_28 : 17^2023 % 28 = 17 := by
  sorry

end NUMINAMATH_CALUDE_power_17_mod_28_l1170_117052


namespace NUMINAMATH_CALUDE_rectangle_perimeter_relation_l1170_117012

/-- Given a figure divided into equal squares, this theorem proves the relationship
    between the perimeters of two rectangles formed by these squares. -/
theorem rectangle_perimeter_relation (square_side : ℝ) 
  (h1 : square_side > 0)
  (h2 : 3 * square_side * 2 + 2 * square_side = 112) : 
  4 * square_side * 2 + 2 * square_side = 140 := by
  sorry

#check rectangle_perimeter_relation

end NUMINAMATH_CALUDE_rectangle_perimeter_relation_l1170_117012


namespace NUMINAMATH_CALUDE_container_height_l1170_117060

/-- The height of a cylindrical container A, given specific conditions --/
theorem container_height (r_A r_B : ℝ) (h : ℝ → ℝ) :
  r_A = 2 →
  r_B = 3 →
  (∀ x, h x = (2/3 * x - 6)) →
  (π * r_A^2 * x = π * r_B^2 * h x) →
  x = 27 :=
by sorry

end NUMINAMATH_CALUDE_container_height_l1170_117060


namespace NUMINAMATH_CALUDE_unique_number_from_dialogue_l1170_117086

/-- Represents a two-digit natural number -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Calculates the number of divisors of a natural number -/
def numberOfDivisors (n : ℕ) : ℕ := sorry

/-- Checks if the number satisfies the dialogue conditions -/
def satisfiesDialogueConditions (n : ℕ) : Prop :=
  TwoDigitNumber n ∧
  (∀ m : ℕ, TwoDigitNumber m → sumOfDigits m = sumOfDigits n → m ≠ n) ∧
  (numberOfDivisors n ≠ 2 ∧ numberOfDivisors n ≠ 12) ∧
  (∀ m : ℕ, TwoDigitNumber m → 
    sumOfDigits m = sumOfDigits n → 
    numberOfDivisors m = numberOfDivisors n → 
    m = n)

theorem unique_number_from_dialogue :
  ∃! n : ℕ, satisfiesDialogueConditions n ∧ n = 30 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_from_dialogue_l1170_117086


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l1170_117091

theorem right_triangle_third_side (a b x : ℝ) : 
  a = 3 → b = 4 → (a^2 + b^2 = x^2 ∨ a^2 + x^2 = b^2) → x = 5 ∨ x = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l1170_117091


namespace NUMINAMATH_CALUDE_total_suitcases_l1170_117024

/-- The number of siblings in Lily's family -/
def num_siblings : Nat := 6

/-- The number of parents in Lily's family -/
def num_parents : Nat := 2

/-- The number of grandparents in Lily's family -/
def num_grandparents : Nat := 2

/-- The number of other relatives in Lily's family -/
def num_other_relatives : Nat := 3

/-- The number of suitcases each parent brings -/
def suitcases_per_parent : Nat := 3

/-- The number of suitcases each grandparent brings -/
def suitcases_per_grandparent : Nat := 2

/-- The total number of suitcases brought by other relatives -/
def suitcases_other_relatives : Nat := 8

/-- The sum of suitcases brought by siblings -/
def siblings_suitcases : Nat := (List.range num_siblings).sum.succ

/-- The total number of suitcases brought by Lily's family -/
theorem total_suitcases : 
  siblings_suitcases + 
  (num_parents * suitcases_per_parent) + 
  (num_grandparents * suitcases_per_grandparent) + 
  suitcases_other_relatives = 39 := by
  sorry

end NUMINAMATH_CALUDE_total_suitcases_l1170_117024


namespace NUMINAMATH_CALUDE_triangle_inequality_left_equality_condition_right_equality_condition_l1170_117082

/-- A triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq_ab : a + b > c
  triangle_ineq_bc : b + c > a
  triangle_ineq_ca : c + a > b

theorem triangle_inequality (t : Triangle) :
  3 * (t.a * t.b + t.b * t.c + t.c * t.a) ≤ (t.a + t.b + t.c)^2 ∧
  (t.a + t.b + t.c)^2 < 4 * (t.a * t.b + t.b * t.c + t.c * t.a) := by
  sorry

theorem left_equality_condition (t : Triangle) :
  3 * (t.a * t.b + t.b * t.c + t.c * t.a) = (t.a + t.b + t.c)^2 ↔ t.a = t.b ∧ t.b = t.c := by
  sorry

theorem right_equality_condition (t : Triangle) :
  (t.a + t.b + t.c)^2 = 4 * (t.a * t.b + t.b * t.c + t.c * t.a) ↔
  t.a + t.b = t.c ∨ t.b + t.c = t.a ∨ t.c + t.a = t.b := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_left_equality_condition_right_equality_condition_l1170_117082


namespace NUMINAMATH_CALUDE_papers_per_notepad_l1170_117079

/-- The number of folds applied to the paper -/
def num_folds : ℕ := 3

/-- The number of days a notepad lasts -/
def days_per_notepad : ℕ := 4

/-- The number of notes written per day -/
def notes_per_day : ℕ := 10

/-- The number of smaller pieces obtained from one letter-size paper after folding -/
def pieces_per_paper : ℕ := 2^num_folds

/-- The total number of notes in one notepad -/
def notes_per_notepad : ℕ := days_per_notepad * notes_per_day

/-- Theorem: The number of letter-size papers needed for one notepad is 5 -/
theorem papers_per_notepad : (notes_per_notepad + pieces_per_paper - 1) / pieces_per_paper = 5 := by
  sorry

end NUMINAMATH_CALUDE_papers_per_notepad_l1170_117079


namespace NUMINAMATH_CALUDE_average_equation_solution_l1170_117084

theorem average_equation_solution (a : ℝ) : 
  ((2 * a + 16) + (3 * a - 8)) / 2 = 84 → a = 32 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_solution_l1170_117084


namespace NUMINAMATH_CALUDE_p_amount_l1170_117080

theorem p_amount (p : ℚ) : p = (1/4) * p + 42 → p = 56 := by
  sorry

end NUMINAMATH_CALUDE_p_amount_l1170_117080


namespace NUMINAMATH_CALUDE_not_all_squares_congruent_l1170_117033

-- Define a square
structure Square where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define congruence for squares
def congruent (s1 s2 : Square) : Prop :=
  s1.side_length = s2.side_length

-- Theorem: It is false that all squares are congruent to each other
theorem not_all_squares_congruent : ¬ ∀ (s1 s2 : Square), congruent s1 s2 := by
  sorry

-- Other properties of squares (for completeness, not directly used in the proof)
def is_convex (s : Square) : Prop := true
def has_four_right_angles (s : Square) : Prop := true
def has_equal_diagonals (s : Square) : Prop := true
def similar (s1 s2 : Square) : Prop := true

end NUMINAMATH_CALUDE_not_all_squares_congruent_l1170_117033


namespace NUMINAMATH_CALUDE_sqrt_196_equals_14_l1170_117076

theorem sqrt_196_equals_14 : Real.sqrt 196 = 14 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_196_equals_14_l1170_117076


namespace NUMINAMATH_CALUDE_water_purification_equation_l1170_117009

/-- Represents the water purification scenario -/
structure WaterPurification where
  total_area : ℝ
  efficiency_increase : ℝ
  days_saved : ℝ
  daily_rate : ℝ

/-- Theorem stating the correct equation for the water purification scenario -/
theorem water_purification_equation (wp : WaterPurification) 
  (h1 : wp.total_area = 2400)
  (h2 : wp.efficiency_increase = 0.2)
  (h3 : wp.days_saved = 40)
  (h4 : wp.daily_rate > 0) :
  (wp.total_area * (1 + wp.efficiency_increase)) / wp.daily_rate - wp.total_area / wp.daily_rate = wp.days_saved :=
by sorry

end NUMINAMATH_CALUDE_water_purification_equation_l1170_117009


namespace NUMINAMATH_CALUDE_dart_board_probability_l1170_117065

/-- The probability of a dart landing in the center square of a regular octagon dart board -/
theorem dart_board_probability (s : ℝ) (h : s > 0) : 
  let octagon_area := 2 * (1 + Real.sqrt 2) * s^2
  let center_square_area := (s/2)^2
  center_square_area / octagon_area = 1 / (4 + 4 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_dart_board_probability_l1170_117065


namespace NUMINAMATH_CALUDE_average_weight_problem_l1170_117036

/-- Given the average weight of three people and two of them, prove the average weight of two of them. -/
theorem average_weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 43 →  -- The average weight of a, b, and c is 43 kg
  (a + b) / 2 = 40 →      -- The average weight of a and b is 40 kg
  b = 37 →                -- The weight of b is 37 kg
  (b + c) / 2 = 43        -- The average weight of b and c is 43 kg
  := by sorry

end NUMINAMATH_CALUDE_average_weight_problem_l1170_117036


namespace NUMINAMATH_CALUDE_cafeteria_pies_l1170_117042

/-- Given a cafeteria with initial apples, apples handed out, and apples per pie,
    calculate the number of pies that can be made. -/
def calculate_pies (initial_apples : ℕ) (apples_handed_out : ℕ) (apples_per_pie : ℕ) : ℕ :=
  (initial_apples - apples_handed_out) / apples_per_pie

/-- Theorem stating that with 96 initial apples, 42 apples handed out,
    and 6 apples per pie, the cafeteria can make 9 pies. -/
theorem cafeteria_pies :
  calculate_pies 96 42 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l1170_117042


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1170_117002

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_properties
  (a b c : ℝ)
  (h1 : f a b c (-1) = -1)
  (h2 : f a b c 0 = -7/4)
  (h3 : f a b c 1 = -2)
  (h4 : f a b c 2 = -7/4) :
  (f a b c 3 = -1) ∧
  (∀ x, f a b c x ≥ -2) ∧
  (f a b c 1 = -2) ∧
  (∀ x₁ x₂, -1 < x₁ → x₁ < 0 → 1 < x₂ → x₂ < 2 → f a b c x₁ > f a b c x₂) ∧
  (∀ x, 0 ≤ x → x ≤ 5 → -2 ≤ f a b c x ∧ f a b c x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1170_117002


namespace NUMINAMATH_CALUDE_raffle_ticket_sales_l1170_117047

theorem raffle_ticket_sales (total_avg : ℝ) (male_avg : ℝ) (female_avg : ℝ) :
  total_avg = 66 →
  male_avg = 58 →
  (1 : ℝ) * male_avg + 2 * female_avg = 3 * total_avg →
  female_avg = 70 := by
  sorry

end NUMINAMATH_CALUDE_raffle_ticket_sales_l1170_117047


namespace NUMINAMATH_CALUDE_x_zero_not_necessary_nor_sufficient_l1170_117081

theorem x_zero_not_necessary_nor_sufficient :
  ¬(∀ x : ℝ, x^2 - 2*x = 0 ↔ x = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_x_zero_not_necessary_nor_sufficient_l1170_117081


namespace NUMINAMATH_CALUDE_point_on_y_axis_l1170_117049

/-- A point lies on the y-axis if and only if its x-coordinate is 0 -/
def lies_on_y_axis (x y : ℝ) : Prop := x = 0

/-- The x-coordinate of point P -/
def x_coord (m : ℝ) : ℝ := 6 - 2*m

/-- The y-coordinate of point P -/
def y_coord (m : ℝ) : ℝ := 4 - m

/-- Theorem: If the point P(6-2m, 4-m) lies on the y-axis, then m = 3 -/
theorem point_on_y_axis (m : ℝ) : lies_on_y_axis (x_coord m) (y_coord m) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l1170_117049


namespace NUMINAMATH_CALUDE_oldest_child_age_l1170_117025

def children_ages (ages : Fin 5 → ℕ) : Prop :=
  -- The average age is 6
  (ages 0 + ages 1 + ages 2 + ages 3 + ages 4) / 5 = 6 ∧
  -- Ages are different
  ∀ i j, i ≠ j → ages i ≠ ages j ∧
  -- Difference between consecutive ages is 2
  ∀ i : Fin 4, ages i.succ = ages i + 2

theorem oldest_child_age (ages : Fin 5 → ℕ) (h : children_ages ages) :
  ages 0 = 10 := by
  sorry

end NUMINAMATH_CALUDE_oldest_child_age_l1170_117025


namespace NUMINAMATH_CALUDE_highway_distance_is_4km_l1170_117019

/-- Represents the travel scenario between two points A and B -/
structure TravelScenario where
  highway_speed : ℝ
  path_speed : ℝ
  time_difference : ℝ
  distance_difference : ℝ

/-- The distance from A to B along the highway given the travel scenario -/
def highway_distance (scenario : TravelScenario) : ℝ :=
  scenario.path_speed * scenario.time_difference

/-- Theorem stating that for the given scenario, the highway distance is 4 km -/
theorem highway_distance_is_4km (scenario : TravelScenario) 
  (h1 : scenario.highway_speed = 5)
  (h2 : scenario.path_speed = 4)
  (h3 : scenario.time_difference = 1)
  (h4 : scenario.distance_difference = 6) :
  highway_distance scenario = 4 := by
  sorry

#eval highway_distance { highway_speed := 5, path_speed := 4, time_difference := 1, distance_difference := 6 }

end NUMINAMATH_CALUDE_highway_distance_is_4km_l1170_117019


namespace NUMINAMATH_CALUDE_solution_k_value_l1170_117078

theorem solution_k_value (x y k : ℝ) 
  (hx : x = -1)
  (hy : y = 2)
  (heq : 2 * x + k * y = 6) :
  k = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_k_value_l1170_117078


namespace NUMINAMATH_CALUDE_tank_capacity_is_40_l1170_117068

/-- Represents the total capacity of a water tank in gallons. -/
def tank_capacity : ℝ := sorry

/-- The tank is initially 3/4 full of water. -/
axiom initial_fill : (3 / 4 : ℝ) * tank_capacity = tank_capacity - 5

/-- Adding 5 gallons of water makes the tank 7/8 full. -/
axiom after_adding : (7 / 8 : ℝ) * tank_capacity = tank_capacity

/-- The tank's total capacity is 40 gallons. -/
theorem tank_capacity_is_40 : tank_capacity = 40 := by sorry

end NUMINAMATH_CALUDE_tank_capacity_is_40_l1170_117068


namespace NUMINAMATH_CALUDE_zigzag_angle_theorem_l1170_117063

/-- A structure representing a zigzag line in a rectangle --/
structure ZigzagRectangle where
  ACB : ℝ
  FEG : ℝ
  DCE : ℝ
  DEC : ℝ

/-- The theorem stating that given specific angle measurements in a zigzag rectangle,
    the angle θ formed by the zigzag line is equal to 11 degrees --/
theorem zigzag_angle_theorem (z : ZigzagRectangle) 
  (h1 : z.ACB = 10)
  (h2 : z.FEG = 26)
  (h3 : z.DCE = 14)
  (h4 : z.DEC = 33) :
  ∃ θ : ℝ, θ = 11 := by
  sorry

end NUMINAMATH_CALUDE_zigzag_angle_theorem_l1170_117063


namespace NUMINAMATH_CALUDE_find_m_l1170_117040

def U : Set ℕ := {0, 1, 2, 3}

def A (m : ℝ) : Set ℕ := {x ∈ U | (x : ℝ)^2 + m * x = 0}

theorem find_m :
  ∃ m : ℝ, (U \ A m = {1, 2}) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l1170_117040


namespace NUMINAMATH_CALUDE_father_son_age_ratio_l1170_117026

def father_son_ages (son_age : ℕ) (age_difference : ℕ) : Prop :=
  ∃ (k : ℕ), (son_age + age_difference + 2) = k * (son_age + 2)

theorem father_son_age_ratio :
  let son_age : ℕ := 22
  let age_difference : ℕ := 24
  father_son_ages son_age age_difference →
  (son_age + age_difference + 2) / (son_age + 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_father_son_age_ratio_l1170_117026


namespace NUMINAMATH_CALUDE_value_of_expression_l1170_117048

-- Define the function g
def g (p q r s t : ℝ) (x : ℝ) : ℝ := p * x^4 + q * x^3 + r * x^2 + s * x + t

-- State the theorem
theorem value_of_expression (p q r s t : ℝ) 
  (h : g p q r s t (-1) = 4) : 
  12 * p - 6 * q + 3 * r - 2 * s + t = 13 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1170_117048


namespace NUMINAMATH_CALUDE_ashley_wedding_guests_l1170_117073

/-- Calculates the number of wedding guests based on champagne requirements. -/
def wedding_guests (glasses_per_guest : ℕ) (servings_per_bottle : ℕ) (bottles_needed : ℕ) : ℕ :=
  (servings_per_bottle / glasses_per_guest) * bottles_needed

/-- Theorem stating that Ashley has 120 wedding guests. -/
theorem ashley_wedding_guests :
  wedding_guests 2 6 40 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ashley_wedding_guests_l1170_117073


namespace NUMINAMATH_CALUDE_rectangular_prism_area_volume_relation_l1170_117004

theorem rectangular_prism_area_volume_relation 
  (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) :
  (x * y) * (y * z) * (z * x) = (x * y * z)^3 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_area_volume_relation_l1170_117004


namespace NUMINAMATH_CALUDE_calculate_english_marks_l1170_117071

/-- Proves that given a student's marks in 4 subjects and an average across 5 subjects,
    we can determine the marks in the fifth subject. -/
theorem calculate_english_marks (math physics chem bio : ℕ) (average : ℚ)
    (h_math : math = 65)
    (h_physics : physics = 82)
    (h_chem : chem = 67)
    (h_bio : bio = 85)
    (h_average : average = 79)
    : ∃ english : ℕ, english = 96 ∧ 
      (english + math + physics + chem + bio : ℚ) / 5 = average :=
by
  sorry

end NUMINAMATH_CALUDE_calculate_english_marks_l1170_117071


namespace NUMINAMATH_CALUDE_list_price_calculation_l1170_117039

theorem list_price_calculation (list_price : ℝ) : 
  (0.15 * (list_price - 15) = 0.25 * (list_price - 25)) → 
  list_price = 40 := by
  sorry

end NUMINAMATH_CALUDE_list_price_calculation_l1170_117039


namespace NUMINAMATH_CALUDE_committee_arrangement_l1170_117090

theorem committee_arrangement (n m : ℕ) (hn : n = 7) (hm : m = 3) :
  (Nat.choose (n + m) m) = 120 := by
  sorry

end NUMINAMATH_CALUDE_committee_arrangement_l1170_117090


namespace NUMINAMATH_CALUDE_grass_seed_bags_l1170_117017

theorem grass_seed_bags (lawn_length lawn_width coverage_per_bag extra_coverage : ℕ) 
  (h1 : lawn_length = 22)
  (h2 : lawn_width = 36)
  (h3 : coverage_per_bag = 250)
  (h4 : extra_coverage = 208) :
  (lawn_length * lawn_width + extra_coverage) / coverage_per_bag = 4 := by
  sorry

end NUMINAMATH_CALUDE_grass_seed_bags_l1170_117017


namespace NUMINAMATH_CALUDE_equipment_cost_proof_l1170_117034

/-- The number of players on the team -/
def num_players : ℕ := 16

/-- The cost of a jersey in dollars -/
def jersey_cost : ℚ := 25

/-- The cost of shorts in dollars -/
def shorts_cost : ℚ := 15.20

/-- The cost of socks in dollars -/
def socks_cost : ℚ := 6.80

/-- The total cost of equipment for all players -/
def total_cost : ℚ := num_players * (jersey_cost + shorts_cost + socks_cost)

theorem equipment_cost_proof : total_cost = 752 := by
  sorry

end NUMINAMATH_CALUDE_equipment_cost_proof_l1170_117034


namespace NUMINAMATH_CALUDE_hexagon_sixth_angle_l1170_117062

/-- A hexagon with given angle measures -/
structure Hexagon where
  Q : ℝ
  R : ℝ
  S : ℝ
  T : ℝ
  U : ℝ
  V : ℝ
  sum_angles : Q + R + S + T + U + V = 720
  Q_value : Q = 110
  R_value : R = 135
  S_value : S = 140
  T_value : T = 95
  U_value : U = 100

/-- The sixth angle of a hexagon with five known angles measures 140° -/
theorem hexagon_sixth_angle (h : Hexagon) : h.V = 140 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_sixth_angle_l1170_117062


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1170_117035

theorem polynomial_divisibility (x : ℝ) : 
  let f : ℝ → ℝ := λ x => -x^4 - x^3 - x + 1
  ∃ (u v : ℝ → ℝ), 
    f x = (x^2 + 1) * (u x) ∧ 
    f x + 1 = (x^3 + x^2 + 1) * (v x) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1170_117035


namespace NUMINAMATH_CALUDE_train_length_l1170_117020

/-- Given a bridge and a train, prove the length of the train -/
theorem train_length 
  (bridge_length : ℝ) 
  (train_cross_time : ℝ) 
  (man_cross_time : ℝ) 
  (train_speed : ℝ) 
  (h1 : bridge_length = 180) 
  (h2 : train_cross_time = 20) 
  (h3 : man_cross_time = 8) 
  (h4 : train_speed = 15) : 
  ∃ train_length : ℝ, train_length = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1170_117020


namespace NUMINAMATH_CALUDE_johns_extra_hours_l1170_117038

/-- Given John's work conditions, prove the number of extra hours he works for the bonus -/
theorem johns_extra_hours (regular_wage : ℝ) (regular_hours : ℝ) (bonus : ℝ) (bonus_hourly_rate : ℝ)
  (h1 : regular_wage = 80)
  (h2 : regular_hours = 8)
  (h3 : bonus = 20)
  (h4 : bonus_hourly_rate = 10) :
  (regular_wage + bonus) / bonus_hourly_rate - regular_hours = 2 := by
  sorry

end NUMINAMATH_CALUDE_johns_extra_hours_l1170_117038


namespace NUMINAMATH_CALUDE_urn_probability_theorem_l1170_117006

/-- The number of blue balls in the second urn -/
def M : ℝ := 7.4

/-- The probability of drawing two balls of the same color -/
def same_color_probability : ℝ := 0.65

/-- The number of green balls in the first urn -/
def green_balls_urn1 : ℕ := 3

/-- The number of blue balls in the first urn -/
def blue_balls_urn1 : ℕ := 7

/-- The number of green balls in the second urn -/
def green_balls_urn2 : ℕ := 20

theorem urn_probability_theorem :
  (green_balls_urn1 / (green_balls_urn1 + blue_balls_urn1 : ℝ)) * (green_balls_urn2 / (green_balls_urn2 + M : ℝ)) +
  (blue_balls_urn1 / (green_balls_urn1 + blue_balls_urn1 : ℝ)) * (M / (green_balls_urn2 + M : ℝ)) =
  same_color_probability :=
by sorry

end NUMINAMATH_CALUDE_urn_probability_theorem_l1170_117006


namespace NUMINAMATH_CALUDE_heptagon_diagonals_l1170_117050

/-- The number of distinct diagonals in a convex heptagon -/
def num_diagonals_heptagon : ℕ := 14

/-- The number of sides in a heptagon -/
def heptagon_sides : ℕ := 7

/-- Theorem: The number of distinct diagonals in a convex heptagon is 14 -/
theorem heptagon_diagonals :
  num_diagonals_heptagon = (heptagon_sides * (heptagon_sides - 3)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_heptagon_diagonals_l1170_117050


namespace NUMINAMATH_CALUDE_lionel_walked_four_miles_l1170_117092

-- Define the constants from the problem
def esther_yards : ℕ := 975
def niklaus_feet : ℕ := 1287
def total_feet : ℕ := 25332
def feet_per_yard : ℕ := 3
def feet_per_mile : ℕ := 5280

-- Define Lionel's distance in miles
def lionel_miles : ℚ := 4

-- Theorem statement
theorem lionel_walked_four_miles :
  (total_feet - (esther_yards * feet_per_yard + niklaus_feet)) / feet_per_mile = lionel_miles := by
  sorry

end NUMINAMATH_CALUDE_lionel_walked_four_miles_l1170_117092


namespace NUMINAMATH_CALUDE_consecutive_lucky_years_exist_l1170_117097

def is_lucky_year (n : ℕ) : Prop :=
  let a := n / 100
  let b := n % 100
  n % (a + b) = 0

theorem consecutive_lucky_years_exist : ∃ n : ℕ, is_lucky_year n ∧ is_lucky_year (n + 1) :=
sorry

end NUMINAMATH_CALUDE_consecutive_lucky_years_exist_l1170_117097


namespace NUMINAMATH_CALUDE_roots_sum_and_product_l1170_117061

def absolute_value_equation (x : ℝ) : Prop :=
  |x|^3 + |x|^2 - 4*|x| - 12 = 0

theorem roots_sum_and_product :
  ∃ (roots : Finset ℝ), 
    (∀ x ∈ roots, absolute_value_equation x) ∧
    (roots.sum id = 0) ∧
    (roots.prod id = -4) :=
sorry

end NUMINAMATH_CALUDE_roots_sum_and_product_l1170_117061


namespace NUMINAMATH_CALUDE_tree_planting_equation_l1170_117094

theorem tree_planting_equation (x : ℝ) (h : x > 0) : 
  (180 / x - 180 / (1.5 * x) = 2) ↔ 
  (∃ (planned_trees actual_trees : ℝ),
    planned_trees = 180 / x ∧
    actual_trees = 180 / (1.5 * x) ∧
    planned_trees - actual_trees = 2 ∧
    180 / x > 2) := by sorry

end NUMINAMATH_CALUDE_tree_planting_equation_l1170_117094


namespace NUMINAMATH_CALUDE_line_symmetry_x_axis_symmetric_line_2x_plus_1_l1170_117015

/-- Given a line y = mx + b, its symmetric line with respect to the x-axis is y = -mx - b -/
theorem line_symmetry_x_axis (m b : ℝ) :
  let original_line := fun (x : ℝ) => m * x + b
  let symmetric_line := fun (x : ℝ) => -m * x - b
  ∀ x y : ℝ, y = original_line x ↔ -y = symmetric_line x :=
by sorry

/-- The line symmetric to y = 2x + 1 with respect to the x-axis is y = -2x - 1 -/
theorem symmetric_line_2x_plus_1 :
  let original_line := fun (x : ℝ) => 2 * x + 1
  let symmetric_line := fun (x : ℝ) => -2 * x - 1
  ∀ x y : ℝ, y = original_line x ↔ -y = symmetric_line x :=
by sorry

end NUMINAMATH_CALUDE_line_symmetry_x_axis_symmetric_line_2x_plus_1_l1170_117015


namespace NUMINAMATH_CALUDE_cloth_sale_problem_l1170_117089

/-- Proves that the number of metres of cloth sold is 200 given the specified conditions -/
theorem cloth_sale_problem (total_selling_price : ℕ) (loss_per_metre : ℕ) (cost_price_per_metre : ℕ) : 
  total_selling_price = 12000 →
  loss_per_metre = 12 →
  cost_price_per_metre = 72 →
  (total_selling_price / (cost_price_per_metre - loss_per_metre) : ℕ) = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_problem_l1170_117089


namespace NUMINAMATH_CALUDE_tangent_line_to_exp_plus_x_l1170_117003

/-- A line y = mx + b is tangent to a curve y = f(x) at point (x₀, f(x₀)) if:
    1. The line passes through the point (x₀, f(x₀))
    2. The slope of the line equals the derivative of f at x₀ -/
def is_tangent_line (f : ℝ → ℝ) (f' : ℝ → ℝ) (m b x₀ : ℝ) : Prop :=
  f x₀ = m * x₀ + b ∧ f' x₀ = m

theorem tangent_line_to_exp_plus_x (b : ℝ) :
  (∃ x₀ : ℝ, is_tangent_line (λ x => Real.exp x + x) (λ x => Real.exp x + 1) 2 b x₀) →
  b = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_exp_plus_x_l1170_117003


namespace NUMINAMATH_CALUDE_unique_solution_rational_equation_l1170_117007

theorem unique_solution_rational_equation :
  ∃! x : ℚ, x ≠ 4 ∧ x ≠ 1 ∧
  (3 * x^2 - 15 * x + 12) / (2 * x^2 - 10 * x + 8) = x - 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_rational_equation_l1170_117007


namespace NUMINAMATH_CALUDE_karls_drive_distance_l1170_117085

/-- Represents the problem of calculating Karl's total drive distance --/
theorem karls_drive_distance :
  -- Conditions
  let miles_per_gallon : ℝ := 35
  let tank_capacity : ℝ := 14
  let initial_drive : ℝ := 350
  let gas_bought : ℝ := 8
  let final_tank_fraction : ℝ := 1/2

  -- Definitions derived from conditions
  let initial_gas_used : ℝ := initial_drive / miles_per_gallon
  let remaining_gas_after_initial_drive : ℝ := tank_capacity - initial_gas_used
  let gas_after_refuel : ℝ := remaining_gas_after_initial_drive + gas_bought
  let final_gas : ℝ := tank_capacity * final_tank_fraction
  let gas_used_second_leg : ℝ := gas_after_refuel - final_gas
  let second_leg_distance : ℝ := gas_used_second_leg * miles_per_gallon
  let total_distance : ℝ := initial_drive + second_leg_distance

  -- Theorem statement
  total_distance = 525 := by
  sorry

end NUMINAMATH_CALUDE_karls_drive_distance_l1170_117085


namespace NUMINAMATH_CALUDE_fraction_simplification_l1170_117045

theorem fraction_simplification (a b c : ℝ) (h : a + b + c ≠ 0) :
  (a^2 + 3*a*b + b^2 - c^2) / (a^2 + 3*a*c + c^2 - b^2) = (a + b - c) / (a - b + c) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1170_117045


namespace NUMINAMATH_CALUDE_exact_division_condition_l1170_117043

-- Define the polynomial x^4 + 1
def f (x : ℂ) : ℂ := x^4 + 1

-- Define the trinomial x^2 + px + q
def g (p q x : ℂ) : ℂ := x^2 + p*x + q

-- Define the condition for exact division
def is_exact_division (p q : ℂ) : Prop :=
  ∃ (h : ℂ → ℂ), ∀ x, f x = (g p q x) * (h x)

-- State the theorem
theorem exact_division_condition :
  ∀ p q : ℂ, is_exact_division p q ↔ 
    ((p = 0 ∧ q = Complex.I) ∨ 
     (p = 0 ∧ q = -Complex.I) ∨ 
     (p = Real.sqrt 2 ∧ q = 1) ∨ 
     (p = -Real.sqrt 2 ∧ q = 1)) :=
by sorry

end NUMINAMATH_CALUDE_exact_division_condition_l1170_117043


namespace NUMINAMATH_CALUDE_bicycle_ride_average_speed_l1170_117057

/-- Prove that given an initial ride of 8 miles at 20 mph, riding an additional 16 miles at 40 mph 
    will result in an average speed of 30 mph for the entire trip. -/
theorem bicycle_ride_average_speed 
  (initial_distance : ℝ) (initial_speed : ℝ) (second_speed : ℝ) (target_average_speed : ℝ)
  (additional_distance : ℝ) :
  initial_distance = 8 ∧ 
  initial_speed = 20 ∧ 
  second_speed = 40 ∧ 
  target_average_speed = 30 ∧
  additional_distance = 16 →
  (initial_distance + additional_distance) / 
    ((initial_distance / initial_speed) + (additional_distance / second_speed)) = 
  target_average_speed :=
by sorry

end NUMINAMATH_CALUDE_bicycle_ride_average_speed_l1170_117057


namespace NUMINAMATH_CALUDE_ticket_price_reduction_l1170_117066

theorem ticket_price_reduction (original_price : ℚ) 
  (h1 : original_price = 50)
  (h2 : ∃ (x : ℚ), x > 0 ∧ 
    (4/3 * x) * (original_price - 25/2) = (5/4) * (x * original_price)) :
  original_price - 25/2 = 46.875 := by
sorry

end NUMINAMATH_CALUDE_ticket_price_reduction_l1170_117066


namespace NUMINAMATH_CALUDE_f_range_l1170_117067

noncomputable def f (x : ℝ) : ℝ :=
  (1/2) * Real.sin (2*x) * Real.tan x + 2 * Real.sin x * Real.tan (x/2)

theorem f_range :
  Set.range f = Set.Icc 0 3 ∪ Set.Ioo 3 4 :=
sorry

end NUMINAMATH_CALUDE_f_range_l1170_117067


namespace NUMINAMATH_CALUDE_robins_hair_length_l1170_117031

/-- Given Robin's initial hair length and the amount cut, calculate the remaining length -/
theorem robins_hair_length (initial_length cut_length : ℕ) : 
  initial_length = 17 → cut_length = 4 → initial_length - cut_length = 13 := by
  sorry

end NUMINAMATH_CALUDE_robins_hair_length_l1170_117031


namespace NUMINAMATH_CALUDE_eccentricity_product_range_l1170_117075

/-- An ellipse and a hyperbola with common foci -/
structure ConicPair where
  F₁ : ℝ × ℝ  -- Left focus
  F₂ : ℝ × ℝ  -- Right focus
  P : ℝ × ℝ   -- Intersection point
  e₁ : ℝ      -- Eccentricity of ellipse
  e₂ : ℝ      -- Eccentricity of hyperbola

/-- The conditions given in the problem -/
def satisfies_conditions (pair : ConicPair) : Prop :=
  pair.F₁.1 < 0 ∧ pair.F₂.1 > 0 ∧  -- Foci on x-axis, centered at origin
  pair.P.1 > 0 ∧ pair.P.2 > 0 ∧    -- P in first quadrant
  ‖pair.P - pair.F₁‖ = ‖pair.P - pair.F₂‖ ∧  -- Isosceles triangle
  ‖pair.P - pair.F₁‖ = 10 ∧        -- |PF₁| = 10
  pair.e₁ > 0 ∧ pair.e₂ > 0        -- Positive eccentricities

theorem eccentricity_product_range (pair : ConicPair) 
  (h : satisfies_conditions pair) : 
  pair.e₁ * pair.e₂ > 1/3 ∧ 
  ∀ M, ∃ pair', satisfies_conditions pair' ∧ pair'.e₁ * pair'.e₂ > M :=
by sorry

end NUMINAMATH_CALUDE_eccentricity_product_range_l1170_117075


namespace NUMINAMATH_CALUDE_tangent_circles_diametric_intersection_l1170_117069

-- Define the types for our geometric objects
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given circles and points
variable (c c1 c2 : Circle)
variable (A B P Q : Point)

-- Define the property of internal tangency
def internallyTangent (c1 c2 : Circle) (P : Point) : Prop :=
  -- The distance between centers is the difference of radii
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius - c2.radius)^2
  -- P lies on both circles
  ∧ (P.x - c1.center.1)^2 + (P.y - c1.center.2)^2 = c1.radius^2
  ∧ (P.x - c2.center.1)^2 + (P.y - c2.center.2)^2 = c2.radius^2

-- Define the property of a point lying on a circle
def pointOnCircle (P : Point) (c : Circle) : Prop :=
  (P.x - c.center.1)^2 + (P.y - c.center.2)^2 = c.radius^2

-- Define the property of points being diametrically opposite on a circle
def diametricallyOpposite (P Q : Point) (c : Circle) : Prop :=
  (P.x - c.center.1) = -(Q.x - c.center.1) ∧ (P.y - c.center.2) = -(Q.y - c.center.2)

-- State the theorem
theorem tangent_circles_diametric_intersection :
  internallyTangent c c1 A
  → internallyTangent c c2 B
  → ∃ (M N : Point),
    pointOnCircle M c
    ∧ pointOnCircle N c
    ∧ diametricallyOpposite M N c
    ∧ (∃ (t : ℝ), M = ⟨A.x + t * (P.x - A.x), A.y + t * (P.y - A.y)⟩)
    ∧ (∃ (s : ℝ), N = ⟨B.x + s * (Q.x - B.x), B.y + s * (Q.y - B.y)⟩) :=
by sorry

end NUMINAMATH_CALUDE_tangent_circles_diametric_intersection_l1170_117069


namespace NUMINAMATH_CALUDE_smallest_AC_solution_exists_l1170_117032

-- Define the triangle and its properties
def Triangle (AC CD : ℕ) : Prop :=
  ∃ (AB BD : ℕ),
    AB = AC ∧  -- AB = AC
    BD * BD = 68 ∧  -- BD² = 68
    AC = (CD * CD + 68) / (2 * CD) ∧  -- Derived from the Pythagorean theorem
    CD < 10 ∧  -- CD is less than 10
    Nat.Prime CD  -- CD is prime

-- State the theorem
theorem smallest_AC :
  ∀ AC CD, Triangle AC CD → AC ≥ 18 :=
by sorry

-- State the existence of a solution
theorem solution_exists :
  ∃ AC CD, Triangle AC CD ∧ AC = 18 :=
by sorry

end NUMINAMATH_CALUDE_smallest_AC_solution_exists_l1170_117032


namespace NUMINAMATH_CALUDE_average_sale_calculation_l1170_117046

theorem average_sale_calculation (sale1 sale2 sale3 sale4 : ℕ) :
  sale1 = 2500 →
  sale2 = 4000 →
  sale3 = 3540 →
  sale4 = 1520 →
  (sale1 + sale2 + sale3 + sale4) / 4 = 2890 := by
sorry

end NUMINAMATH_CALUDE_average_sale_calculation_l1170_117046


namespace NUMINAMATH_CALUDE_unique_integer_function_l1170_117088

def IntegerFunction (f : ℤ → ℚ) : Prop :=
  ∀ (x y z : ℤ), 
    (∀ (c : ℚ), f x < c ∧ c < f y → ∃ (w : ℤ), f w = c) ∧
    (x + y + z = 0 → f x + f y + f z = f x * f y * f z)

theorem unique_integer_function : 
  ∃! (f : ℤ → ℚ), IntegerFunction f ∧ (∀ x : ℤ, f x = 0) :=
sorry

end NUMINAMATH_CALUDE_unique_integer_function_l1170_117088


namespace NUMINAMATH_CALUDE_sin_120_degrees_l1170_117064

theorem sin_120_degrees : Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_120_degrees_l1170_117064


namespace NUMINAMATH_CALUDE_mans_rowing_speed_in_still_water_l1170_117000

/-- Proves that a man's rowing speed in still water is 15 km/h given the conditions of downstream travel --/
theorem mans_rowing_speed_in_still_water :
  let current_speed : ℝ := 3 -- km/h
  let distance : ℝ := 60 / 1000 -- 60 meters converted to km
  let time : ℝ := 11.999040076793857 / 3600 -- seconds converted to hours
  let downstream_speed : ℝ := distance / time
  downstream_speed = current_speed + 15 := by sorry

end NUMINAMATH_CALUDE_mans_rowing_speed_in_still_water_l1170_117000


namespace NUMINAMATH_CALUDE_bike_ride_problem_l1170_117070

/-- Bike ride problem -/
theorem bike_ride_problem (total_distance : ℝ) (total_time : ℝ) (rest_time : ℝ) 
  (fast_speed : ℝ) (slow_speed : ℝ) 
  (h1 : total_distance = 142)
  (h2 : total_time = 8)
  (h3 : rest_time = 0.5)
  (h4 : fast_speed = 22)
  (h5 : slow_speed = 15) :
  ∃ energetic_time : ℝ, 
    energetic_time * fast_speed + (total_time - rest_time - energetic_time) * slow_speed = total_distance ∧ 
    energetic_time = 59 / 14 := by
  sorry


end NUMINAMATH_CALUDE_bike_ride_problem_l1170_117070


namespace NUMINAMATH_CALUDE_smallest_five_digit_multiple_l1170_117030

theorem smallest_five_digit_multiple : ∃ (n : ℕ), 
  (n ≥ 10000 ∧ n < 100000) ∧ 
  (15 ∣ n) ∧ (45 ∣ n) ∧ (54 ∣ n) ∧ 
  (∃ (k : ℕ), n = 2^k * (n / 2^k)) ∧
  (∀ (m : ℕ), m < n → 
    ¬((m ≥ 10000 ∧ m < 100000) ∧ 
      (15 ∣ m) ∧ (45 ∣ m) ∧ (54 ∣ m) ∧ 
      (∃ (j : ℕ), m = 2^j * (m / 2^j)))) ∧
  n = 69120 := by
sorry

end NUMINAMATH_CALUDE_smallest_five_digit_multiple_l1170_117030


namespace NUMINAMATH_CALUDE_figure_area_proof_l1170_117053

theorem figure_area_proof (square_side : ℝ) (gray_area white_area : ℝ) 
  (h1 : gray_area = square_side^2)
  (h2 : white_area = (3/2) * square_side^2)
  (h3 : gray_area = white_area + 0.6)
  (h4 : square_side^2 = 1.2) :
  5 * square_side^2 = 6 := by sorry

end NUMINAMATH_CALUDE_figure_area_proof_l1170_117053


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l1170_117011

/-- The value of k for which the line y = kx (k > 0) is tangent to the circle (x-√3)^2 + y^2 = 1 -/
theorem tangent_line_to_circle (k : ℝ) : 
  k > 0 ∧ 
  (∃ (x y : ℝ), y = k * x ∧ (x - Real.sqrt 3)^2 + y^2 = 1) ∧
  (∀ (x y : ℝ), y = k * x → (x - Real.sqrt 3)^2 + y^2 ≥ 1) →
  k = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l1170_117011


namespace NUMINAMATH_CALUDE_factorization_equality_l1170_117099

theorem factorization_equality (a b : ℝ) : 2 * a^2 * b - 8 * b = 2 * b * (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1170_117099


namespace NUMINAMATH_CALUDE_angle_AOB_is_right_angle_l1170_117074

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 3*x

-- Define a line passing through (3,0)
def line_through_3_0 (t : ℝ) (x y : ℝ) : Prop := x = t*y + 3

-- Define the intersection points
def intersection_points (t : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
  line_through_3_0 t x₁ y₁ ∧ line_through_3_0 t x₂ y₂

-- Theorem statement
theorem angle_AOB_is_right_angle (t : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  intersection_points t x₁ y₁ x₂ y₂ →
  x₁ * x₂ + y₁ * y₂ = 0 :=
sorry

end NUMINAMATH_CALUDE_angle_AOB_is_right_angle_l1170_117074


namespace NUMINAMATH_CALUDE_percentage_decrease_l1170_117013

theorem percentage_decrease (initial : ℝ) (increase_percent : ℝ) (final : ℝ) :
  initial = 1500 →
  increase_percent = 20 →
  final = 1080 →
  ∃ (decrease_percent : ℝ),
    final = (initial * (1 + increase_percent / 100)) * (1 - decrease_percent / 100) ∧
    decrease_percent = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_decrease_l1170_117013


namespace NUMINAMATH_CALUDE_water_evaporation_proof_l1170_117028

theorem water_evaporation_proof (initial_mass : ℝ) (initial_water_percentage : ℝ) 
  (final_water_percentage : ℝ) (evaporated_water : ℝ) : 
  initial_mass = 500 →
  initial_water_percentage = 0.85 →
  final_water_percentage = 0.75 →
  evaporated_water = 200 →
  (initial_mass * initial_water_percentage - evaporated_water) / (initial_mass - evaporated_water) = final_water_percentage :=
by
  sorry

end NUMINAMATH_CALUDE_water_evaporation_proof_l1170_117028


namespace NUMINAMATH_CALUDE_total_pay_calculation_l1170_117098

def first_job_pay : ℕ := 2125
def pay_difference : ℕ := 375

def second_job_pay : ℕ := first_job_pay - pay_difference

def total_pay : ℕ := first_job_pay + second_job_pay

theorem total_pay_calculation : total_pay = 3875 := by
  sorry

end NUMINAMATH_CALUDE_total_pay_calculation_l1170_117098


namespace NUMINAMATH_CALUDE_probability_one_of_each_l1170_117077

def num_shirts : ℕ := 6
def num_shorts : ℕ := 8
def num_socks : ℕ := 9
def num_hats : ℕ := 4
def total_items : ℕ := num_shirts + num_shorts + num_socks + num_hats
def items_to_select : ℕ := 4

theorem probability_one_of_each :
  (num_shirts.choose 1 * num_shorts.choose 1 * num_socks.choose 1 * num_hats.choose 1) / total_items.choose items_to_select = 96 / 975 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_of_each_l1170_117077


namespace NUMINAMATH_CALUDE_right_triangle_acute_angle_theorem_l1170_117037

theorem right_triangle_acute_angle_theorem :
  ∀ (a b : ℝ), 
  a > 0 ∧ b > 0 →  -- Ensuring positive angles
  a = 2 * b →      -- One acute angle is twice the other
  a + b = 90 →     -- Sum of acute angles in a right triangle is 90°
  a = 60 :=        -- The larger acute angle is 60°
by sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angle_theorem_l1170_117037


namespace NUMINAMATH_CALUDE_ellipse_focal_property_l1170_117018

-- Define the ellipse
def ellipse (x y b : ℝ) : Prop := x^2 / 4 + y^2 / b^2 = 1

-- Define the constraint on b
def b_constraint (b : ℝ) : Prop := 0 < b ∧ b < 2

-- Define the maximum value of |BF_2| + |AF_2|
def max_focal_sum (b : ℝ) : Prop := ∃ (A B F_2 : ℝ × ℝ), 
  ∀ (P Q : ℝ × ℝ), dist P F_2 + dist Q F_2 ≤ dist A F_2 + dist B F_2 ∧ 
  dist A F_2 + dist B F_2 = 5

-- Theorem statement
theorem ellipse_focal_property (b : ℝ) :
  b_constraint b →
  (∀ x y, ellipse x y b → max_focal_sum b) →
  b = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_focal_property_l1170_117018


namespace NUMINAMATH_CALUDE_base3_addition_theorem_l1170_117023

/-- Convert a base 3 number represented as a list of digits to its decimal equivalent -/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- Convert a decimal number to its base 3 representation -/
def decimalToBase3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else go (m / 3) ((m % 3) :: acc)
    go n []

theorem base3_addition_theorem :
  let a := [2]
  let b := [2, 2]
  let c := [2, 0, 2]
  let d := [2, 2, 0, 2]
  let result := [0, 1, 0, 1, 2]
  base3ToDecimal a + base3ToDecimal b + base3ToDecimal c + base3ToDecimal d =
  base3ToDecimal result := by
  sorry

end NUMINAMATH_CALUDE_base3_addition_theorem_l1170_117023


namespace NUMINAMATH_CALUDE_factors_of_180_l1170_117095

def number_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem factors_of_180 : number_of_factors 180 = 18 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_180_l1170_117095
