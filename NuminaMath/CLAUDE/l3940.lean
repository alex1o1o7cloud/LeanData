import Mathlib

namespace NUMINAMATH_CALUDE_pair_op_theorem_l3940_394056

/-- Definition of the custom operation ⊗ for pairs of real numbers -/
def pair_op (a b c d : ℝ) : ℝ × ℝ := (a * c - b * d, a * d + b * c)

/-- Theorem stating that if (1, 2) ⊗ (p, q) = (5, 0), then p + q equals some real number -/
theorem pair_op_theorem (p q : ℝ) :
  pair_op 1 2 p q = (5, 0) → ∃ r : ℝ, p + q = r := by
  sorry

end NUMINAMATH_CALUDE_pair_op_theorem_l3940_394056


namespace NUMINAMATH_CALUDE_ratio_of_sums_is_301_480_l3940_394015

/-- Calculate the sum of an arithmetic sequence -/
def sum_arithmetic (a1 : ℚ) (d : ℚ) (an : ℚ) : ℚ :=
  let n : ℚ := (an - a1) / d + 1
  n * (a1 + an) / 2

/-- The ratio of sums of two specific arithmetic sequences -/
def ratio_of_sums : ℚ :=
  (sum_arithmetic 2 3 41) / (sum_arithmetic 4 4 60)

theorem ratio_of_sums_is_301_480 : ratio_of_sums = 301 / 480 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sums_is_301_480_l3940_394015


namespace NUMINAMATH_CALUDE_ratio_difference_l3940_394072

theorem ratio_difference (a b c : ℕ) (h1 : a + b + c > 0) : 
  (a : ℚ) / (a + b + c) = 3 / 15 →
  (b : ℚ) / (a + b + c) = 5 / 15 →
  (c : ℚ) / (a + b + c) = 7 / 15 →
  c = 70 →
  c - a = 40 := by
sorry

end NUMINAMATH_CALUDE_ratio_difference_l3940_394072


namespace NUMINAMATH_CALUDE_lcm_bound_implies_lower_bound_l3940_394047

theorem lcm_bound_implies_lower_bound (a : Fin 2000 → ℕ+) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_order : ∀ i j, i < j → a i < a j)
  (h_upper_bound : ∀ i, a i < 4000)
  (h_lcm : ∀ i j, i ≠ j → Nat.lcm (a i) (a j) ≥ 4000) :
  a 0 ≥ 1334 := by
sorry

end NUMINAMATH_CALUDE_lcm_bound_implies_lower_bound_l3940_394047


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3940_394001

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 10 →
  a 4 = 24 →
  a 6 = 38 →
  a 3 + a 5 = 48 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3940_394001


namespace NUMINAMATH_CALUDE_A_3_2_l3940_394076

def A : Nat → Nat → Nat
  | 0, n => n + 1
  | m + 1, 0 => A m 1
  | m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_2 : A 3 2 = 29 := by sorry

end NUMINAMATH_CALUDE_A_3_2_l3940_394076


namespace NUMINAMATH_CALUDE_distance_between_points_l3940_394008

/-- The curve equation -/
def curve_equation (x y : ℝ) : Prop := y^2 + x^4 = 2*x^2*y + 1

/-- Theorem stating that for any real number e, if (e, a) and (e, b) are points on the curve y^2 + x^4 = 2x^2y + 1, then |a-b| = 2 -/
theorem distance_between_points (e a b : ℝ) 
  (ha : curve_equation e a) 
  (hb : curve_equation e b) : 
  |a - b| = 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3940_394008


namespace NUMINAMATH_CALUDE_modular_inverse_35_mod_36_l3940_394018

theorem modular_inverse_35_mod_36 : ∃ x : ℤ, (35 * x) % 36 = 1 ∧ x % 36 = 35 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_35_mod_36_l3940_394018


namespace NUMINAMATH_CALUDE_pirate_count_l3940_394070

/-- Represents the total number of pirates on the schooner -/
def total_pirates : ℕ := 30

/-- Represents the number of pirates who did not participate in the battle -/
def non_participants : ℕ := 10

/-- Represents the percentage of battle participants who lost an arm -/
def arm_loss_percentage : ℚ := 54 / 100

/-- Represents the percentage of battle participants who lost both an arm and a leg -/
def both_loss_percentage : ℚ := 34 / 100

/-- Represents the fraction of all pirates who lost a leg -/
def leg_loss_fraction : ℚ := 2 / 3

theorem pirate_count : 
  total_pirates = 30 ∧
  non_participants = 10 ∧
  arm_loss_percentage = 54 / 100 ∧
  both_loss_percentage = 34 / 100 ∧
  leg_loss_fraction = 2 / 3 →
  total_pirates = 30 :=
by sorry

end NUMINAMATH_CALUDE_pirate_count_l3940_394070


namespace NUMINAMATH_CALUDE_circle_radius_is_four_l3940_394097

theorem circle_radius_is_four (r : ℝ) (h : 2 * (2 * Real.pi * r) = Real.pi * r^2) : r = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_four_l3940_394097


namespace NUMINAMATH_CALUDE_equation_representation_l3940_394066

theorem equation_representation (x : ℝ) : 
  (2 * x + 4 = 8) → (∃ y : ℝ, y = 2 * x + 4 ∧ y = 8) := by
  sorry

end NUMINAMATH_CALUDE_equation_representation_l3940_394066


namespace NUMINAMATH_CALUDE_solve_for_a_find_b_find_c_find_d_l3940_394020

-- Part 1
def simultaneous_equations (a u : ℝ) : Prop :=
  3/a + 1/u = 7/2 ∧ 2/a - 3/u = 6

theorem solve_for_a : ∃ a u : ℝ, simultaneous_equations a u ∧ a = 3/2 :=
sorry

-- Part 2
def equation_with_solutions (p q b : ℝ) (a : ℝ) : Prop :=
  p * 0 + q * (3*a) + b * 1 = 1 ∧
  p * (9*a) + q * (-1) + b * 2 = 1 ∧
  p * 0 + q * (3*a) + b * 0 = 1

theorem find_b : ∃ p q b a : ℝ, equation_with_solutions p q b a ∧ b = 0 :=
sorry

-- Part 3
def line_through_points (m c b : ℝ) : Prop :=
  5 = m * (b + 4) + c ∧
  2 = m * (-2) + c

theorem find_c : ∃ m c b : ℝ, line_through_points m c b ∧ c = 3 :=
sorry

-- Part 4
def inequality_solution (c d : ℝ) : Prop :=
  ∀ x : ℝ, d ≤ x ∧ x ≤ 1 ↔ x^2 + 5*x - 2*c ≤ 0

theorem find_d : ∃ c d : ℝ, inequality_solution c d ∧ d = -6 :=
sorry

end NUMINAMATH_CALUDE_solve_for_a_find_b_find_c_find_d_l3940_394020


namespace NUMINAMATH_CALUDE_number_equation_l3940_394050

theorem number_equation (x : ℝ) : 2 * x + 5 = 17 ↔ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l3940_394050


namespace NUMINAMATH_CALUDE_complex_fraction_real_l3940_394063

/-- Given that i is the imaginary unit and (a+2i)/(1+i) is a real number, prove that a = 2 -/
theorem complex_fraction_real (a : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (↑a + 2 * Complex.I) / (1 + Complex.I) ∈ Set.range (Complex.ofReal : ℝ → ℂ) → 
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_real_l3940_394063


namespace NUMINAMATH_CALUDE_function_domain_l3940_394030

/-- The domain of the function y = ln(x+1) / sqrt(-x^2 - 3x + 4) -/
theorem function_domain (x : ℝ) : 
  (x + 1 > 0 ∧ -x^2 - 3*x + 4 > 0) ↔ -1 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_function_domain_l3940_394030


namespace NUMINAMATH_CALUDE_sin_theta_value_l3940_394004

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x - 8 * (Real.cos (x / 2))^2

theorem sin_theta_value (θ : ℝ) :
  (∀ x, f x ≤ f θ) → Real.sin θ = 3/5 :=
by sorry

end NUMINAMATH_CALUDE_sin_theta_value_l3940_394004


namespace NUMINAMATH_CALUDE_child_grandmother_weight_ratio_l3940_394064

def family_weights (total_weight daughter_weight daughter_child_weight : ℝ) : Prop :=
  total_weight = 120 ∧ daughter_child_weight = 60 ∧ daughter_weight = 48

theorem child_grandmother_weight_ratio 
  (total_weight daughter_weight daughter_child_weight : ℝ) 
  (h : family_weights total_weight daughter_weight daughter_child_weight) : 
  (daughter_child_weight - daughter_weight) / (total_weight - daughter_child_weight) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_child_grandmother_weight_ratio_l3940_394064


namespace NUMINAMATH_CALUDE_probability_is_correct_l3940_394039

def total_vehicles : ℕ := 20000
def shattered_windshields : ℕ := 600

def probability_shattered_windshield : ℚ :=
  shattered_windshields / total_vehicles

theorem probability_is_correct : 
  probability_shattered_windshield = 3 / 100 := by sorry

end NUMINAMATH_CALUDE_probability_is_correct_l3940_394039


namespace NUMINAMATH_CALUDE_tabitha_current_age_l3940_394025

def tabitha_hair_colors (age : ℕ) : ℕ :=
  age - 13

theorem tabitha_current_age : 
  ∃ (current_age : ℕ), 
    tabitha_hair_colors current_age = 5 ∧ 
    tabitha_hair_colors (current_age + 3) = 8 ∧ 
    current_age = 18 := by
  sorry

end NUMINAMATH_CALUDE_tabitha_current_age_l3940_394025


namespace NUMINAMATH_CALUDE_union_A_M_eq_real_union_B_complement_M_eq_B_l3940_394011

-- Define the sets A, B, and M
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 9}
def B (b : ℝ) : Set ℝ := {x | 8 - b < x ∧ x < b}
def M : Set ℝ := {x | x < -1 ∨ x > 5}

-- Statement for the first part of the problem
theorem union_A_M_eq_real (a : ℝ) :
  A a ∪ M = Set.univ ↔ -4 ≤ a ∧ a ≤ -1 :=
sorry

-- Statement for the second part of the problem
theorem union_B_complement_M_eq_B (b : ℝ) :
  B b ∪ (Set.univ \ M) = B b ↔ b > 9 :=
sorry

end NUMINAMATH_CALUDE_union_A_M_eq_real_union_B_complement_M_eq_B_l3940_394011


namespace NUMINAMATH_CALUDE_complex_modulus_theorem_l3940_394034

theorem complex_modulus_theorem (z : ℂ) (h : z * Complex.I = 1 - Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_theorem_l3940_394034


namespace NUMINAMATH_CALUDE_circle_plus_four_two_l3940_394093

/-- Definition of the ⊕ operation -/
def circle_plus (a b : ℝ) : ℝ := 5 * a + 6 * b

/-- Theorem stating that 4 ⊕ 2 = 32 -/
theorem circle_plus_four_two : circle_plus 4 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_circle_plus_four_two_l3940_394093


namespace NUMINAMATH_CALUDE_max_sundays_in_45_days_l3940_394085

/-- The number of days we're considering at the start of the year -/
def days_considered : ℕ := 45

/-- The day of the week, represented as a number from 0 to 6 -/
inductive DayOfWeek : Type
  | Sunday : DayOfWeek
  | Monday : DayOfWeek
  | Tuesday : DayOfWeek
  | Wednesday : DayOfWeek
  | Thursday : DayOfWeek
  | Friday : DayOfWeek
  | Saturday : DayOfWeek

/-- Function to count the number of Sundays in the first n days of a year starting on a given day -/
def count_sundays (start_day : DayOfWeek) (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of Sundays in the first 45 days of a year starting on Sunday is 7 -/
theorem max_sundays_in_45_days : 
  count_sundays DayOfWeek.Sunday days_considered = 7 :=
sorry

end NUMINAMATH_CALUDE_max_sundays_in_45_days_l3940_394085


namespace NUMINAMATH_CALUDE_water_pouring_problem_l3940_394060

def water_remaining (n : ℕ) : ℚ :=
  1 / (n + 1)

theorem water_pouring_problem :
  ∃ n : ℕ, water_remaining n = 1 / 10 ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_water_pouring_problem_l3940_394060


namespace NUMINAMATH_CALUDE_shooting_probabilities_l3940_394043

/-- The probability of shooter A hitting the target -/
def P_A : ℝ := 0.9

/-- The probability of shooter B hitting the target -/
def P_B : ℝ := 0.8

/-- The probability of both A and B hitting the target -/
def P_both : ℝ := P_A * P_B

/-- The probability of at least one of A and B hitting the target -/
def P_at_least_one : ℝ := 1 - (1 - P_A) * (1 - P_B)

theorem shooting_probabilities :
  P_both = 0.72 ∧ P_at_least_one = 0.98 := by
  sorry

end NUMINAMATH_CALUDE_shooting_probabilities_l3940_394043


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3940_394069

theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  exterior_angle = 24 → n * exterior_angle = 360 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3940_394069


namespace NUMINAMATH_CALUDE_share_difference_example_l3940_394073

/-- Given a total profit and proportions for distribution among three parties,
    calculate the difference between the shares of the second and third parties. -/
def shareDifference (totalProfit : ℕ) (propA propB propC : ℕ) : ℕ :=
  let totalParts := propA + propB + propC
  let partValue := totalProfit / totalParts
  let shareB := propB * partValue
  let shareC := propC * partValue
  shareC - shareB

/-- Prove that for a total profit of 20000 distributed in the proportion 2:3:5,
    the difference between C's and B's shares is 4000. -/
theorem share_difference_example : shareDifference 20000 2 3 5 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_share_difference_example_l3940_394073


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3940_394040

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {2, 4, 5, 7, 8}

theorem intersection_of_A_and_B : A ∩ B = {2, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3940_394040


namespace NUMINAMATH_CALUDE_book_distribution_ways_l3940_394080

/-- The number of ways to distribute books to students -/
def distribute_books (num_book_types : ℕ) (num_students : ℕ) (min_copies : ℕ) : ℕ :=
  num_book_types ^ num_students

/-- Theorem: There are 125 ways to distribute 3 books to 3 students from 5 types of books -/
theorem book_distribution_ways :
  distribute_books 5 3 3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_book_distribution_ways_l3940_394080


namespace NUMINAMATH_CALUDE_family_weight_ratio_l3940_394032

/-- Given the weights of three generations in a family, prove the ratio of the child's weight to the grandmother's weight -/
theorem family_weight_ratio :
  ∀ (grandmother daughter child : ℝ),
  grandmother + daughter + child = 110 →
  daughter + child = 60 →
  daughter = 50 →
  child / grandmother = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_family_weight_ratio_l3940_394032


namespace NUMINAMATH_CALUDE_angle_problem_l3940_394037

theorem angle_problem (θ : Real) 
  (h1 : π/2 < θ ∧ θ < π)  -- θ is in the second quadrant
  (h2 : Real.tan (θ + π/4) = 1/2) :
  Real.tan θ = -1/3 ∧ 
  Real.sin (π/2 - 2*θ) + Real.sin (π + 2*θ) = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_angle_problem_l3940_394037


namespace NUMINAMATH_CALUDE_price_fluctuation_l3940_394057

theorem price_fluctuation (p : ℝ) (original_price : ℝ) : 
  (original_price * (1 + p / 100) * (1 - p / 100) = 1) →
  (original_price = 10000 / (10000 - p^2)) :=
by sorry

end NUMINAMATH_CALUDE_price_fluctuation_l3940_394057


namespace NUMINAMATH_CALUDE_min_ABFG_value_l3940_394068

/-- Represents a seven-digit number ABCDEFG -/
structure SevenDigitNumber where
  digits : Fin 7 → Nat
  is_valid : ∀ i, digits i < 10

/-- Extracts a five-digit number from a seven-digit number -/
def extract_five_digits (n : SevenDigitNumber) (start : Fin 3) : Nat :=
  (n.digits start) * 10000 + (n.digits (start + 1)) * 1000 + 
  (n.digits (start + 2)) * 100 + (n.digits (start + 3)) * 10 + 
  (n.digits (start + 4))

/-- Extracts a four-digit number ABFG from a seven-digit number ABCDEFG -/
def extract_ABFG (n : SevenDigitNumber) : Nat :=
  (n.digits 0) * 1000 + (n.digits 1) * 100 + (n.digits 5) * 10 + (n.digits 6)

/-- The main theorem -/
theorem min_ABFG_value (n : SevenDigitNumber) 
  (h1 : extract_five_digits n 1 % 2013 = 0)
  (h2 : extract_five_digits n 3 % 1221 = 0) :
  3036 ≤ extract_ABFG n :=
sorry

end NUMINAMATH_CALUDE_min_ABFG_value_l3940_394068


namespace NUMINAMATH_CALUDE_modulus_of_specific_complex_number_l3940_394092

open Complex

theorem modulus_of_specific_complex_number :
  let z : ℂ := (2 - I) / (2 + I)
  ‖z‖ = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_specific_complex_number_l3940_394092


namespace NUMINAMATH_CALUDE_jane_usable_sheets_l3940_394054

/-- Represents the total number of sheets Jane has for each type and size --/
structure TotalSheets where
  brownA4 : ℕ
  yellowA4 : ℕ
  yellowA3 : ℕ

/-- Represents the number of damaged sheets (less than 70% intact) for each type and size --/
structure DamagedSheets where
  brownA4 : ℕ
  yellowA4 : ℕ
  yellowA3 : ℕ

/-- Calculates the number of usable sheets given the total and damaged sheets --/
def usableSheets (total : TotalSheets) (damaged : DamagedSheets) : ℕ :=
  (total.brownA4 - damaged.brownA4) + (total.yellowA4 - damaged.yellowA4) + (total.yellowA3 - damaged.yellowA3)

theorem jane_usable_sheets :
  let total := TotalSheets.mk 28 18 9
  let damaged := DamagedSheets.mk 3 5 2
  usableSheets total damaged = 45 := by
  sorry

end NUMINAMATH_CALUDE_jane_usable_sheets_l3940_394054


namespace NUMINAMATH_CALUDE_total_notes_count_l3940_394006

/-- Proves that given a total amount of Rs. 10350 in Rs. 50 and Rs. 500 notes, 
    with 57 notes of Rs. 50 denomination, the total number of notes is 72. -/
theorem total_notes_count (total_amount : ℕ) (fifty_note_count : ℕ) : 
  total_amount = 10350 →
  fifty_note_count = 57 →
  ∃ (five_hundred_note_count : ℕ),
    total_amount = fifty_note_count * 50 + five_hundred_note_count * 500 ∧
    fifty_note_count + five_hundred_note_count = 72 :=
by sorry

end NUMINAMATH_CALUDE_total_notes_count_l3940_394006


namespace NUMINAMATH_CALUDE_ratio_sum_theorem_l3940_394083

theorem ratio_sum_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a / b = 3 / 4) (h4 : b = 12) : a + b = 21 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_theorem_l3940_394083


namespace NUMINAMATH_CALUDE_john_gum_purchase_l3940_394090

/-- The number of packs of gum John bought -/
def num_gum_packs : ℕ := 2

/-- The number of candy bars John bought -/
def num_candy_bars : ℕ := 3

/-- The cost of one candy bar in dollars -/
def candy_bar_cost : ℚ := 3/2

/-- The total amount John paid in dollars -/
def total_paid : ℚ := 6

/-- The cost of one pack of gum in dollars -/
def gum_pack_cost : ℚ := candy_bar_cost / 2

theorem john_gum_purchase :
  num_gum_packs * gum_pack_cost + num_candy_bars * candy_bar_cost = total_paid :=
by sorry

end NUMINAMATH_CALUDE_john_gum_purchase_l3940_394090


namespace NUMINAMATH_CALUDE_sum_of_squares_of_quadratic_solutions_l3940_394035

theorem sum_of_squares_of_quadratic_solutions : 
  let a : ℝ := -2
  let b : ℝ := -4
  let c : ℝ := -42
  let α : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let β : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  α^2 + β^2 = 46 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_quadratic_solutions_l3940_394035


namespace NUMINAMATH_CALUDE_min_value_theorem_l3940_394002

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → 2 * x' + y' = 2 → 1 / x' - y' ≥ 2 * Real.sqrt 2 - 2) ∧
  (1 / (Real.sqrt 2 / 2) - (2 - Real.sqrt 2) = 2 * Real.sqrt 2 - 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3940_394002


namespace NUMINAMATH_CALUDE_one_third_recipe_ingredients_l3940_394088

def full_recipe_flour : ℚ := 27/4  -- 6 3/4 cups of flour
def full_recipe_sugar : ℚ := 5/2   -- 2 1/2 cups of sugar
def recipe_fraction : ℚ := 1/3     -- One-third of the recipe

theorem one_third_recipe_ingredients :
  recipe_fraction * full_recipe_flour = 9/4 ∧
  recipe_fraction * full_recipe_sugar = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_one_third_recipe_ingredients_l3940_394088


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l3940_394029

theorem ceiling_floor_sum : ⌈(7 : ℚ) / 3⌉ + ⌊-(7 : ℚ) / 3⌋ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l3940_394029


namespace NUMINAMATH_CALUDE_minimum_point_of_translated_abs_value_function_l3940_394082

-- Define the function f
def f (x : ℝ) : ℝ := 2 * abs (x - 3) - 4 + 4

-- Theorem statement
theorem minimum_point_of_translated_abs_value_function :
  ∃ (x : ℝ), ∀ (y : ℝ), f y ≥ f x ∧ f x = 0 ∧ x = 3 :=
sorry

end NUMINAMATH_CALUDE_minimum_point_of_translated_abs_value_function_l3940_394082


namespace NUMINAMATH_CALUDE_remaining_cube_volume_l3940_394022

/-- The remaining volume of a cube after removing a cylindrical section -/
theorem remaining_cube_volume (cube_side : ℝ) (cylinder_radius : ℝ) : 
  cube_side = 8 → cylinder_radius = 2 → 
  (cube_side ^ 3 : ℝ) - π * cylinder_radius ^ 2 * cube_side = 512 - 32 * π := by
  sorry

end NUMINAMATH_CALUDE_remaining_cube_volume_l3940_394022


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l3940_394067

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a-2, 2*a-2}

theorem subset_implies_a_equals_one (a : ℝ) :
  A a ⊆ B a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l3940_394067


namespace NUMINAMATH_CALUDE_floor_of_4_7_l3940_394075

theorem floor_of_4_7 : ⌊(4.7 : ℝ)⌋ = 4 := by sorry

end NUMINAMATH_CALUDE_floor_of_4_7_l3940_394075


namespace NUMINAMATH_CALUDE_difference_of_extreme_valid_numbers_l3940_394000

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ 
  (n.digits 10).count 2 = 3 ∧ 
  (n.digits 10).count 0 = 1

def largest_valid_number : ℕ := 2220
def smallest_valid_number : ℕ := 2022

theorem difference_of_extreme_valid_numbers :
  largest_valid_number - smallest_valid_number = 198 ∧
  is_valid_number largest_valid_number ∧
  is_valid_number smallest_valid_number ∧
  (∀ n : ℕ, is_valid_number n → n ≤ largest_valid_number) ∧
  (∀ n : ℕ, is_valid_number n → smallest_valid_number ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_difference_of_extreme_valid_numbers_l3940_394000


namespace NUMINAMATH_CALUDE_total_selling_price_cloth_l3940_394071

/-- Calculate the total selling price of cloth given the quantity, profit per meter, and cost price per meter. -/
theorem total_selling_price_cloth (quantity : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ) :
  quantity = 85 →
  profit_per_meter = 15 →
  cost_price_per_meter = 85 →
  quantity * (cost_price_per_meter + profit_per_meter) = 8500 := by
  sorry

end NUMINAMATH_CALUDE_total_selling_price_cloth_l3940_394071


namespace NUMINAMATH_CALUDE_q_is_false_l3940_394042

theorem q_is_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q :=
by sorry

end NUMINAMATH_CALUDE_q_is_false_l3940_394042


namespace NUMINAMATH_CALUDE_root_ratio_quadratic_equation_l3940_394095

theorem root_ratio_quadratic_equation (a b c : ℝ) (h1 : a ≠ 0) 
  (h2 : ∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ x/y = 2/3 ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0) :
  6*b^2 = 25*a*c := by
  sorry

end NUMINAMATH_CALUDE_root_ratio_quadratic_equation_l3940_394095


namespace NUMINAMATH_CALUDE_polygon_with_170_diagonals_has_20_sides_l3940_394078

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular polygon with 170 diagonals has 20 sides -/
theorem polygon_with_170_diagonals_has_20_sides :
  ∃ (n : ℕ), n > 2 ∧ num_diagonals n = 170 → n = 20 :=
by
  sorry

#check polygon_with_170_diagonals_has_20_sides

end NUMINAMATH_CALUDE_polygon_with_170_diagonals_has_20_sides_l3940_394078


namespace NUMINAMATH_CALUDE_birds_on_fence_l3940_394012

theorem birds_on_fence (initial_birds : ℕ) (total_birds : ℕ) (new_birds : ℕ) : 
  initial_birds = 2 → total_birds = 6 → new_birds = total_birds - initial_birds → new_birds = 4 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l3940_394012


namespace NUMINAMATH_CALUDE_faster_train_speed_faster_train_speed_result_l3940_394084

/-- Calculates the speed of the faster train given the conditions of the problem -/
theorem faster_train_speed (length_train1 length_train2 : ℝ)
                            (speed_slower : ℝ)
                            (crossing_time : ℝ) : ℝ :=
  let total_length := length_train1 + length_train2
  let total_length_km := total_length / 1000
  let crossing_time_hours := crossing_time / 3600
  let relative_speed := total_length_km / crossing_time_hours
  speed_slower + relative_speed

/-- The speed of the faster train is approximately 45.95 kmph -/
theorem faster_train_speed_result :
  ∃ ε > 0, |faster_train_speed 200 150 40 210 - 45.95| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_faster_train_speed_faster_train_speed_result_l3940_394084


namespace NUMINAMATH_CALUDE_carwash_problem_l3940_394033

theorem carwash_problem (car_price truck_price suv_price : ℕ) 
                        (total_raised : ℕ) 
                        (num_cars num_trucks : ℕ) : 
  car_price = 5 →
  truck_price = 6 →
  suv_price = 7 →
  total_raised = 100 →
  num_cars = 7 →
  num_trucks = 5 →
  ∃ (num_suvs : ℕ), 
    num_suvs * suv_price + num_cars * car_price + num_trucks * truck_price = total_raised ∧
    num_suvs = 5 := by
  sorry

end NUMINAMATH_CALUDE_carwash_problem_l3940_394033


namespace NUMINAMATH_CALUDE_cat_meow_ratio_l3940_394053

/-- Given three cats meowing, prove the ratio of meows per minute of the third cat to the second cat -/
theorem cat_meow_ratio :
  ∀ (cat1_rate cat2_rate cat3_rate : ℚ),
  cat1_rate = 3 →
  cat2_rate = 2 * cat1_rate →
  5 * (cat1_rate + cat2_rate + cat3_rate) = 55 →
  cat3_rate / cat2_rate = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_cat_meow_ratio_l3940_394053


namespace NUMINAMATH_CALUDE_de_morgan_laws_l3940_394051

universe u

theorem de_morgan_laws {α : Type u} (A B : Set α) : 
  ((A ∪ B)ᶜ = Aᶜ ∩ Bᶜ) ∧ ((A ∩ B)ᶜ = Aᶜ ∪ Bᶜ) := by
  sorry

end NUMINAMATH_CALUDE_de_morgan_laws_l3940_394051


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3940_394099

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, 4 * x^2 - 3 * x + 2 < 0) ↔ (∃ x : ℝ, 4 * x^2 - 3 * x + 2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3940_394099


namespace NUMINAMATH_CALUDE_keystone_arch_angle_l3940_394041

/-- A keystone arch composed of congruent isosceles trapezoids -/
structure KeystoneArch where
  /-- The number of trapezoids in the arch -/
  num_trapezoids : ℕ
  /-- The measure of the central angle between two adjacent trapezoids in degrees -/
  central_angle : ℝ
  /-- The measure of the smaller interior angle of each trapezoid in degrees -/
  smaller_angle : ℝ
  /-- The measure of the larger interior angle of each trapezoid in degrees -/
  larger_angle : ℝ
  /-- The sum of interior angles of a trapezoid is 360° -/
  angle_sum : smaller_angle + larger_angle = 180
  /-- The central angle is related to the number of trapezoids -/
  central_angle_def : central_angle = 360 / num_trapezoids
  /-- The smaller angle plus half the central angle is 90° -/
  smaller_angle_def : smaller_angle + central_angle / 2 = 90

/-- Theorem: In a keystone arch with 10 congruent isosceles trapezoids, 
    the larger interior angle of each trapezoid is 99° -/
theorem keystone_arch_angle (arch : KeystoneArch) 
    (h : arch.num_trapezoids = 10) : arch.larger_angle = 99 := by
  sorry

end NUMINAMATH_CALUDE_keystone_arch_angle_l3940_394041


namespace NUMINAMATH_CALUDE_safari_creatures_l3940_394021

/-- Proves that given 150 creatures with 624 legs total, where some are two-legged ostriches
    and others are six-legged aliens, the number of ostriches is 69. -/
theorem safari_creatures (total_creatures : ℕ) (total_legs : ℕ) 
    (h1 : total_creatures = 150)
    (h2 : total_legs = 624) : 
  ∃ (ostriches aliens : ℕ),
    ostriches + aliens = total_creatures ∧
    2 * ostriches + 6 * aliens = total_legs ∧
    ostriches = 69 := by
  sorry

end NUMINAMATH_CALUDE_safari_creatures_l3940_394021


namespace NUMINAMATH_CALUDE_product_of_roots_quadratic_l3940_394059

theorem product_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - x₁ - 6 = 0) → (x₂^2 - x₂ - 6 = 0) → x₁ * x₂ = -6 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_quadratic_l3940_394059


namespace NUMINAMATH_CALUDE_amy_music_files_l3940_394038

theorem amy_music_files :
  ∀ (initial_music_files : ℕ),
    initial_music_files + 21 - 23 = 2 →
    initial_music_files = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_amy_music_files_l3940_394038


namespace NUMINAMATH_CALUDE_power_sum_sequence_l3940_394024

theorem power_sum_sequence (a b : ℝ) : 
  a + b = 1 → 
  a^2 + b^2 = 3 → 
  a^3 + b^3 = 4 → 
  a^4 + b^4 = 7 → 
  a^5 + b^5 = 11 → 
  a^10 + b^10 = 123 := by
sorry

end NUMINAMATH_CALUDE_power_sum_sequence_l3940_394024


namespace NUMINAMATH_CALUDE_bake_sale_brownie_cost_l3940_394079

/-- Proves that the cost per brownie is $2 given the conditions of the bake sale --/
theorem bake_sale_brownie_cost (total_revenue : ℝ) (num_pans : ℕ) (pieces_per_pan : ℕ) :
  total_revenue = 32 →
  num_pans = 2 →
  pieces_per_pan = 8 →
  (total_revenue / (num_pans * pieces_per_pan : ℝ)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_bake_sale_brownie_cost_l3940_394079


namespace NUMINAMATH_CALUDE_two_primes_between_lower_limit_and_14_l3940_394046

theorem two_primes_between_lower_limit_and_14 : 
  ∃ (x : ℕ), x ≤ 7 ∧ 
  (∃ (p q : ℕ), Prime p ∧ Prime q ∧ x < p ∧ p < q ∧ q < 14) ∧
  (∀ (y : ℕ), y > 7 → ¬(∃ (p q : ℕ), Prime p ∧ Prime q ∧ y < p ∧ p < q ∧ q < 14)) :=
sorry

end NUMINAMATH_CALUDE_two_primes_between_lower_limit_and_14_l3940_394046


namespace NUMINAMATH_CALUDE_hyperbola_properties_l3940_394089

/-- A hyperbola with foci on the x-axis, real axis length 4√2, and eccentricity √5/2 -/
structure Hyperbola where
  /-- Real axis length -/
  real_axis_length : ℝ
  real_axis_length_eq : real_axis_length = 4 * Real.sqrt 2
  /-- Eccentricity -/
  e : ℝ
  e_eq : e = Real.sqrt 5 / 2

/-- Standard form of the hyperbola equation -/
def standard_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 8 - y^2 / 2 = 1

/-- Equation of the trajectory of point Q -/
def trajectory_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 8 - y^2 / 4 = 1 ∧ x ≠ 2 * Real.sqrt 2 ∧ x ≠ -2 * Real.sqrt 2

theorem hyperbola_properties (h : Hyperbola) :
  (∀ x y, standard_equation h x y ↔ 
    x^2 / (2 * Real.sqrt 2)^2 - y^2 / ((Real.sqrt 5 / 2) * 2 * Real.sqrt 2)^2 = 1) ∧
  (∀ x y, trajectory_equation h x y ↔
    x^2 / 8 - y^2 / 4 = 1 ∧ x ≠ 2 * Real.sqrt 2 ∧ x ≠ -2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l3940_394089


namespace NUMINAMATH_CALUDE_graceGardenTopBedRows_l3940_394049

/-- Represents the garden structure and seed distribution --/
structure Garden where
  totalSeeds : ℕ
  topBedSeedsPerRow : ℕ
  mediumBedRows : ℕ
  mediumBedSeedsPerRow : ℕ
  numMediumBeds : ℕ

/-- Calculates the number of rows in the top bed --/
def topBedRows (g : Garden) : ℕ :=
  (g.totalSeeds - g.numMediumBeds * g.mediumBedRows * g.mediumBedSeedsPerRow) / g.topBedSeedsPerRow

/-- Theorem stating that for Grace's garden, the top bed can hold 8 rows --/
theorem graceGardenTopBedRows :
  let g : Garden := {
    totalSeeds := 320,
    topBedSeedsPerRow := 25,
    mediumBedRows := 3,
    mediumBedSeedsPerRow := 20,
    numMediumBeds := 2
  }
  topBedRows g = 8 := by
  sorry

end NUMINAMATH_CALUDE_graceGardenTopBedRows_l3940_394049


namespace NUMINAMATH_CALUDE_intersection_with_complement_of_B_l3940_394003

variable (U A B : Finset ℕ)

theorem intersection_with_complement_of_B (hU : U = {1, 2, 3, 4, 5, 6, 7})
  (hA : A = {3, 4, 5}) (hB : B = {1, 3, 6}) :
  A ∩ (U \ B) = {4, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_of_B_l3940_394003


namespace NUMINAMATH_CALUDE_market_spending_l3940_394077

theorem market_spending (mildred_spent candice_spent joseph_spent_percentage joseph_spent mom_total : ℝ) :
  mildred_spent = 25 →
  candice_spent = 35 →
  joseph_spent_percentage = 0.8 →
  joseph_spent = 45 →
  mom_total = 150 →
  mom_total - (mildred_spent + candice_spent + joseph_spent) = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_market_spending_l3940_394077


namespace NUMINAMATH_CALUDE_cone_ratio_l3940_394044

/-- For a cone with a central angle of 120° in its unfolded lateral surface,
    the ratio of its base radius to its slant height is 1/3 -/
theorem cone_ratio (r : ℝ) (l : ℝ) (h : r > 0) (h' : l > 0) :
  2 * Real.pi * r = 2 * Real.pi / 3 * l → r / l = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_cone_ratio_l3940_394044


namespace NUMINAMATH_CALUDE_binary_multiplication_example_l3940_394009

/-- Represents a binary number as a list of bits (0 or 1) -/
def BinaryNum := List Nat

/-- Converts a decimal number to its binary representation -/
def to_binary (n : Nat) : BinaryNum :=
  sorry

/-- Converts a binary number to its decimal representation -/
def to_decimal (b : BinaryNum) : Nat :=
  sorry

/-- Multiplies two binary numbers -/
def binary_multiply (a b : BinaryNum) : BinaryNum :=
  sorry

theorem binary_multiplication_example :
  let a : BinaryNum := [1, 0, 1, 0, 1]  -- 10101₂
  let b : BinaryNum := [1, 0, 1]        -- 101₂
  let result : BinaryNum := [1, 1, 0, 1, 0, 0, 1]  -- 1101001₂
  binary_multiply a b = result :=
by sorry

end NUMINAMATH_CALUDE_binary_multiplication_example_l3940_394009


namespace NUMINAMATH_CALUDE_specific_management_structure_count_l3940_394086

/-- The number of ways to form a management structure --/
def management_structure_count (total_employees : ℕ) (ceo_count : ℕ) (vp_count : ℕ) (managers_per_vp : ℕ) : ℕ :=
  total_employees * 
  (Nat.choose (total_employees - 1) vp_count) * 
  (Nat.choose (total_employees - 1 - vp_count) managers_per_vp) * 
  (Nat.choose (total_employees - 1 - vp_count - managers_per_vp) managers_per_vp)

/-- Theorem stating the number of ways to form the specific management structure --/
theorem specific_management_structure_count :
  management_structure_count 13 1 2 3 = 349800 := by
  sorry

end NUMINAMATH_CALUDE_specific_management_structure_count_l3940_394086


namespace NUMINAMATH_CALUDE_dispersion_measure_properties_l3940_394094

-- Define a type for datasets
structure Dataset where
  data : List ℝ

-- Define a type for dispersion measures
structure DispersionMeasure where
  measure : Dataset → ℝ

-- Statement 1: Multiple values can be used to describe the degree of dispersion
def multipleValuesUsed (d : DispersionMeasure) : Prop :=
  ∃ (d1 d2 : DispersionMeasure), d1 ≠ d2

-- Statement 2: One should make full use of the obtained data
def fullDataUsed (d : DispersionMeasure) : Prop :=
  ∀ (dataset : Dataset), d.measure dataset = d.measure dataset

-- Statement 3: For different datasets, when the degree of dispersion is large, this value should be smaller (incorrect statement)
def incorrectDispersionRelation (d : DispersionMeasure) : Prop :=
  ∃ (dataset1 dataset2 : Dataset),
    d.measure dataset1 > d.measure dataset2 →
    d.measure dataset1 < d.measure dataset2

theorem dispersion_measure_properties :
  ∃ (d : DispersionMeasure),
    multipleValuesUsed d ∧
    fullDataUsed d ∧
    ¬incorrectDispersionRelation d :=
  sorry

end NUMINAMATH_CALUDE_dispersion_measure_properties_l3940_394094


namespace NUMINAMATH_CALUDE_meaningful_zero_power_l3940_394017

theorem meaningful_zero_power (m : ℝ) (h : m ≠ -1) : (m + 1) ^ (0 : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_zero_power_l3940_394017


namespace NUMINAMATH_CALUDE_focus_on_negative_y_axis_l3940_394065

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 + y = 0

-- Define the focus of a parabola
def focus (p q : ℝ) : Prop := ∃ (a : ℝ), p = 0 ∧ q = -1/(4*a)

-- Theorem statement
theorem focus_on_negative_y_axis :
  ∃ (p q : ℝ), focus p q ∧ ∀ (x y : ℝ), parabola x y → q < 0 :=
sorry

end NUMINAMATH_CALUDE_focus_on_negative_y_axis_l3940_394065


namespace NUMINAMATH_CALUDE_exists_k_configuration_l3940_394096

/-- A configuration of black cells on an infinite white checker plane. -/
structure BlackCellConfiguration where
  cells : Set (ℤ × ℤ)
  finite : Set.Finite cells

/-- A line on the infinite plane (vertical, horizontal, or diagonal). -/
inductive Line
  | Vertical (x : ℤ) : Line
  | Horizontal (y : ℤ) : Line
  | Diagonal (a b c : ℤ) : Line

/-- The number of black cells on a given line for a given configuration. -/
def blackCellsOnLine (config : BlackCellConfiguration) (line : Line) : ℕ :=
  sorry

/-- A configuration satisfies the k-condition if every line contains
    either k black cells or no black cells. -/
def satisfiesKCondition (config : BlackCellConfiguration) (k : ℕ+) : Prop :=
  ∀ line, blackCellsOnLine config line = k ∨ blackCellsOnLine config line = 0

/-- For any positive integer k, there exists a configuration of black cells
    satisfying the k-condition. -/
theorem exists_k_configuration (k : ℕ+) :
  ∃ (config : BlackCellConfiguration), satisfiesKCondition config k :=
sorry

end NUMINAMATH_CALUDE_exists_k_configuration_l3940_394096


namespace NUMINAMATH_CALUDE_hexagon_circle_area_ratio_l3940_394055

/-- Given a regular hexagon and a circle with equal perimeter/circumference,
    the ratio of the area of the hexagon to the area of the circle is π√3/6 -/
theorem hexagon_circle_area_ratio :
  ∀ (s r : ℝ),
  s > 0 → r > 0 →
  6 * s = 2 * Real.pi * r →
  (3 * Real.sqrt 3 / 2 * s^2) / (Real.pi * r^2) = Real.pi * Real.sqrt 3 / 6 := by
sorry

end NUMINAMATH_CALUDE_hexagon_circle_area_ratio_l3940_394055


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3940_394098

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  d : ℚ      -- Common difference
  seq_def : ∀ n, a (n + 1) = a n + d  -- Definition of arithmetic sequence

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

/-- Main theorem -/
theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence) 
  (h1 : S seq 5 = seq.a 8 + 5)
  (h2 : S seq 6 = seq.a 7 + seq.a 9 - 5) :
  seq.d = -55 / 19 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3940_394098


namespace NUMINAMATH_CALUDE_regular_hexagon_perimeter_l3940_394027

/-- The perimeter of a regular hexagon given its radius -/
theorem regular_hexagon_perimeter (radius : ℝ) : 
  radius = 3 → 6 * radius = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_hexagon_perimeter_l3940_394027


namespace NUMINAMATH_CALUDE_molecular_weight_Al2S3_proof_l3940_394052

/-- The molecular weight of Al2S3 in g/mol -/
def molecular_weight_Al2S3 : ℝ := 150

/-- The number of moles used in the given condition -/
def moles : ℝ := 3

/-- The total weight of the given number of moles in grams -/
def total_weight : ℝ := 450

/-- Theorem: The molecular weight of Al2S3 is 150 g/mol, given that 3 moles weigh 450 grams -/
theorem molecular_weight_Al2S3_proof : 
  molecular_weight_Al2S3 = total_weight / moles := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_Al2S3_proof_l3940_394052


namespace NUMINAMATH_CALUDE_gunther_free_time_l3940_394014

/-- Represents the time in minutes for each cleaning task -/
structure CleaningTasks where
  vacuum : ℕ
  dust : ℕ
  mop : ℕ
  brush_cat : ℕ

/-- Calculates the total cleaning time in hours -/
def total_cleaning_time (tasks : CleaningTasks) (num_cats : ℕ) : ℚ :=
  (tasks.vacuum + tasks.dust + tasks.mop + tasks.brush_cat * num_cats) / 60

/-- Theorem: If Gunther has no cats and 30 minutes of free time left after cleaning,
    his initial free time was 2.75 hours -/
theorem gunther_free_time (tasks : CleaningTasks) 
    (h1 : tasks.vacuum = 45)
    (h2 : tasks.dust = 60)
    (h3 : tasks.mop = 30)
    (h4 : tasks.brush_cat = 5)
    (h5 : total_cleaning_time tasks 0 + 0.5 = 2.75) : True :=
  sorry

end NUMINAMATH_CALUDE_gunther_free_time_l3940_394014


namespace NUMINAMATH_CALUDE_dog_tether_area_l3940_394028

/-- The area outside a regular hexagon that can be reached by a tethered point -/
theorem dog_tether_area (side_length : Real) (rope_length : Real) :
  side_length = 1 ∧ rope_length = 3 →
  let hexagon_area := 3 * Real.sqrt 3 / 2 * side_length^2
  let tether_area := 2 * Real.pi * rope_length^2 / 3 + Real.pi * (rope_length - side_length)^2 / 3
  tether_area - hexagon_area = 22 * Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_dog_tether_area_l3940_394028


namespace NUMINAMATH_CALUDE_other_number_proof_l3940_394058

theorem other_number_proof (x y : ℕ+) 
  (h1 : Nat.lcm x y = 2640)
  (h2 : Nat.gcd x y = 24)
  (h3 : x = 240) :
  y = 264 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l3940_394058


namespace NUMINAMATH_CALUDE_jellybean_box_scaling_l3940_394005

theorem jellybean_box_scaling (bert_jellybeans : ℕ) (scale_factor : ℕ) : 
  bert_jellybeans = 150 → scale_factor = 3 →
  (scale_factor ^ 3 : ℕ) * bert_jellybeans = 4050 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_box_scaling_l3940_394005


namespace NUMINAMATH_CALUDE_probability_of_passing_is_correct_l3940_394074

-- Define the number of shots in the test
def num_shots : ℕ := 3

-- Define the minimum number of successful shots required to pass
def min_successful_shots : ℕ := 2

-- Define the probability of making a single shot
def single_shot_probability : ℝ := 0.6

-- Define the function to calculate the probability of passing the test
def probability_of_passing : ℝ := sorry

-- Theorem stating that the probability of passing is 0.648
theorem probability_of_passing_is_correct : probability_of_passing = 0.648 := by sorry

end NUMINAMATH_CALUDE_probability_of_passing_is_correct_l3940_394074


namespace NUMINAMATH_CALUDE_surface_area_volume_incomparable_l3940_394062

-- Define the edge length of the cube
def edge_length : ℝ := 6

-- Define the surface area of the cube
def surface_area (l : ℝ) : ℝ := 6 * l^2

-- Define the volume of the cube
def volume (l : ℝ) : ℝ := l^3

-- Theorem stating that surface area and volume are incomparable
theorem surface_area_volume_incomparable :
  ¬(∃ (ord : ℝ → ℝ → Prop), 
    (∀ a b, ord a b ∨ ord b a) ∧ 
    (∀ a b c, ord a b → ord b c → ord a c) ∧
    (∀ a b, ord a b → ord b a → a = b) ∧
    (ord (surface_area edge_length) (volume edge_length) ∨ 
     ord (volume edge_length) (surface_area edge_length))) :=
by sorry


end NUMINAMATH_CALUDE_surface_area_volume_incomparable_l3940_394062


namespace NUMINAMATH_CALUDE_karting_routes_count_l3940_394023

/-- Represents the number of routes ending at point A after n minutes -/
def M_n_A : ℕ → ℕ
| 0 => 0
| 1 => 0
| 2 => 1
| (n+3) => M_n_A (n+1) + M_n_A n

/-- The race duration in minutes -/
def race_duration : ℕ := 10

/-- Theorem stating that the number of routes ending at A after 10 minutes
    is equal to the 10th number in the defined Fibonacci-like sequence -/
theorem karting_routes_count : M_n_A race_duration = 34 := by
  sorry

end NUMINAMATH_CALUDE_karting_routes_count_l3940_394023


namespace NUMINAMATH_CALUDE_product_less_than_square_l3940_394031

theorem product_less_than_square : 1234567 * 1234569 < 1234568^2 := by
  sorry

end NUMINAMATH_CALUDE_product_less_than_square_l3940_394031


namespace NUMINAMATH_CALUDE_cost_per_person_is_correct_l3940_394007

def item1_base_cost : ℝ := 40
def item1_tax_rate : ℝ := 0.05
def item1_discount_rate : ℝ := 0.10

def item2_base_cost : ℝ := 70
def item2_tax_rate : ℝ := 0.08
def item2_coupon : ℝ := 5

def item3_base_cost : ℝ := 100
def item3_tax_rate : ℝ := 0.06
def item3_discount_rate : ℝ := 0.10

def num_people : ℕ := 3

def calculate_item1_cost : ℝ := 
  let cost_after_tax := item1_base_cost * (1 + item1_tax_rate)
  cost_after_tax * (1 - item1_discount_rate)

def calculate_item2_cost : ℝ := 
  item2_base_cost * (1 + item2_tax_rate) - item2_coupon

def calculate_item3_cost : ℝ := 
  let cost_after_tax := item3_base_cost * (1 + item3_tax_rate)
  cost_after_tax * (1 - item3_discount_rate)

def total_cost : ℝ := 
  calculate_item1_cost + calculate_item2_cost + calculate_item3_cost

theorem cost_per_person_is_correct : 
  total_cost / num_people = 67.93 := by sorry

end NUMINAMATH_CALUDE_cost_per_person_is_correct_l3940_394007


namespace NUMINAMATH_CALUDE_laps_run_l3940_394087

/-- Proves the number of laps run in a track given total distance, lap length, and remaining laps --/
theorem laps_run (total_distance : ℕ) (lap_length : ℕ) (remaining_laps : ℕ) 
  (h1 : total_distance = 2400)
  (h2 : lap_length = 150)
  (h3 : remaining_laps = 4) :
  (total_distance / lap_length) - remaining_laps = 12 := by
  sorry

#check laps_run

end NUMINAMATH_CALUDE_laps_run_l3940_394087


namespace NUMINAMATH_CALUDE_shoe_problem_contradiction_l3940_394026

theorem shoe_problem_contradiction (becky bobby bonny : ℕ) : 
  (bonny = 2 * becky - 5) →
  (bobby = 3 * becky) →
  (bonny = bobby) →
  False :=
by sorry

end NUMINAMATH_CALUDE_shoe_problem_contradiction_l3940_394026


namespace NUMINAMATH_CALUDE_taxi_cost_formula_correct_l3940_394010

/-- Represents the total cost in dollars for a taxi ride -/
def taxiCost (T : ℕ) : ℤ :=
  10 + 5 * T - 10 * (if T > 5 then 1 else 0)

/-- Theorem stating the correctness of the taxi cost formula -/
theorem taxi_cost_formula_correct (T : ℕ) :
  taxiCost T = 10 + 5 * T - 10 * (if T > 5 then 1 else 0) := by
  sorry

#check taxi_cost_formula_correct

end NUMINAMATH_CALUDE_taxi_cost_formula_correct_l3940_394010


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l3940_394016

theorem triangle_side_calculation (A B C : ℝ) (a b c : ℝ) 
  (h1 : 0 < A ∧ A < π) 
  (h2 : 0 < B ∧ B < π)
  (h3 : 0 < C ∧ C < π)
  (h4 : A + B + C = π)
  (h5 : Real.cos A = 4/5)
  (h6 : Real.cos C = 5/13)
  (h7 : a = 13)
  (h8 : a / Real.sin A = b / Real.sin B)
  (h9 : b / Real.sin B = c / Real.sin C)
  : b = 21 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l3940_394016


namespace NUMINAMATH_CALUDE_max_distance_between_circles_l3940_394045

/-- Circle C₁ with equation x² + (y+3)² = 1 -/
def C₁ (x y : ℝ) : Prop := x^2 + (y+3)^2 = 1

/-- Circle C₂ with equation (x-4)² + y² = 4 -/
def C₂ (x y : ℝ) : Prop := (x-4)^2 + y^2 = 4

/-- The maximum distance between any point on C₁ and any point on C₂ is 8 -/
theorem max_distance_between_circles :
  ∃ (max_dist : ℝ),
    (∀ (x₁ y₁ x₂ y₂ : ℝ), C₁ x₁ y₁ → C₂ x₂ y₂ → 
      (x₁ - x₂)^2 + (y₁ - y₂)^2 ≤ max_dist^2) ∧
    (∃ (x₁ y₁ x₂ y₂ : ℝ), C₁ x₁ y₁ ∧ C₂ x₂ y₂ ∧ 
      (x₁ - x₂)^2 + (y₁ - y₂)^2 = max_dist^2) ∧
    max_dist = 8 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_between_circles_l3940_394045


namespace NUMINAMATH_CALUDE_dice_puzzle_l3940_394036

/-- Given five dice with 21 dots each and 43 visible dots, prove that 62 dots are not visible -/
theorem dice_puzzle (num_dice : ℕ) (dots_per_die : ℕ) (visible_dots : ℕ) : 
  num_dice = 5 → dots_per_die = 21 → visible_dots = 43 → 
  num_dice * dots_per_die - visible_dots = 62 := by
  sorry

end NUMINAMATH_CALUDE_dice_puzzle_l3940_394036


namespace NUMINAMATH_CALUDE_evaluate_expression_l3940_394091

theorem evaluate_expression : 3 + 3 * (3 ^ (3 ^ 3)) - 3 ^ 3 = 22876792454937 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3940_394091


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3940_394081

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def equation (z : ℂ) : Prop :=
  z * i = ((i + 1) / (i - 1)) ^ 2016

-- Theorem statement
theorem complex_equation_solution :
  ∃ (z : ℂ), equation z ∧ z = -i :=
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3940_394081


namespace NUMINAMATH_CALUDE_min_value_xy_minus_2x_l3940_394061

theorem min_value_xy_minus_2x (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : y * Real.log x + y * Real.log y = Real.exp x) : 
  ∃ (m : ℝ), m = 2 - 2 * Real.log 2 ∧ 
  ∀ (z : ℝ), z > 0 → y * z * Real.log (y * z) = z * Real.exp z → 
  x * y - 2 * x ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_xy_minus_2x_l3940_394061


namespace NUMINAMATH_CALUDE_sodium_chloride_formation_l3940_394019

-- Define the chemical reaction
structure Reaction where
  hcl : ℕ  -- moles of Hydrochloric acid
  nahco3 : ℕ  -- moles of Sodium bicarbonate
  nacl : ℕ  -- moles of Sodium chloride produced

-- Define the stoichiometric relationship
def stoichiometric_relationship (r : Reaction) : Prop :=
  r.nacl = min r.hcl r.nahco3

-- Theorem statement
theorem sodium_chloride_formation (r : Reaction) 
  (h1 : r.hcl = 2)  -- 2 moles of Hydrochloric acid
  (h2 : r.nahco3 = 2)  -- 2 moles of Sodium bicarbonate
  (h3 : stoichiometric_relationship r)  -- The reaction follows the stoichiometric relationship
  : r.nacl = 2 :=
by sorry

end NUMINAMATH_CALUDE_sodium_chloride_formation_l3940_394019


namespace NUMINAMATH_CALUDE_base_k_conversion_l3940_394048

/-- Given a base k, convert a list of digits to its decimal representation -/
def toDecimal (k : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + k * acc) 0

/-- The problem statement -/
theorem base_k_conversion :
  ∃ (k : ℕ), k > 0 ∧ toDecimal k [2, 3, 1] = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_base_k_conversion_l3940_394048


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3940_394013

/-- Calculates the speed of a train in km/hr given its length in meters and time to cross a pole in seconds. -/
def trainSpeed (length : Float) (time : Float) : Float :=
  (length / 1000) / (time / 3600)

/-- Theorem stating that a train with a length of 450.00000000000006 meters 
    crossing a pole in 27 seconds has a speed of 60 km/hr. -/
theorem train_speed_calculation :
  let length : Float := 450.00000000000006
  let time : Float := 27
  trainSpeed length time = 60 := by
  sorry

#eval trainSpeed 450.00000000000006 27

end NUMINAMATH_CALUDE_train_speed_calculation_l3940_394013
