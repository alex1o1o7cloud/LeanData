import Mathlib

namespace NUMINAMATH_CALUDE_pigeons_on_pole_l1737_173764

theorem pigeons_on_pole (initial_pigeons : ℕ) (pigeons_flew_away : ℕ) (pigeons_left : ℕ) : 
  initial_pigeons = 8 → pigeons_flew_away = 3 → pigeons_left = initial_pigeons - pigeons_flew_away → pigeons_left = 5 := by
  sorry

end NUMINAMATH_CALUDE_pigeons_on_pole_l1737_173764


namespace NUMINAMATH_CALUDE_election_winner_percentage_l1737_173746

theorem election_winner_percentage (winner_votes loser_votes : ℕ) 
  (h1 : winner_votes = 1344)
  (h2 : winner_votes - loser_votes = 288) :
  (winner_votes : ℚ) / ((winner_votes : ℚ) + (loser_votes : ℚ)) * 100 = 56 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l1737_173746


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1737_173743

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ+ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ+, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ+ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ+, a n > 0) →
  a 1 * a 3 + 2 * a 2 * a 5 + a 4 * a 6 = 36 →
  a 2 + a 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1737_173743


namespace NUMINAMATH_CALUDE_consecutive_good_numbers_l1737_173752

/-- A number is good if it can be expressed as 2^x + y^2 for nonnegative integers x and y. -/
def IsGood (n : ℕ) : Prop :=
  ∃ x y : ℕ, n = 2^x + y^2

/-- A set of 5 consecutive numbers is a good set if all numbers in the set are good. -/
def IsGoodSet (s : Fin 5 → ℕ) : Prop :=
  (∀ i : Fin 5, IsGood (s i)) ∧ (∀ i : Fin 4, s (Fin.succ i) = s i + 1)

/-- The theorem states that there are only six sets of 5 consecutive good numbers. -/
theorem consecutive_good_numbers :
  ∀ s : Fin 5 → ℕ, IsGoodSet s →
    (s = ![1, 2, 3, 4, 5]) ∨
    (s = ![2, 3, 4, 5, 6]) ∨
    (s = ![8, 9, 10, 11, 12]) ∨
    (s = ![9, 10, 11, 12, 13]) ∨
    (s = ![288, 289, 290, 291, 292]) ∨
    (s = ![289, 290, 291, 292, 293]) :=
by sorry


end NUMINAMATH_CALUDE_consecutive_good_numbers_l1737_173752


namespace NUMINAMATH_CALUDE_log_base_2_derivative_l1737_173769

theorem log_base_2_derivative (x : ℝ) (h : x > 0) : 
  deriv (fun x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_log_base_2_derivative_l1737_173769


namespace NUMINAMATH_CALUDE_number_of_factors_60_l1737_173737

/-- The number of positive factors of 60 is 12. -/
theorem number_of_factors_60 : Finset.card (Nat.divisors 60) = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_60_l1737_173737


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1737_173783

def A : Set ℝ := {y | ∃ x : ℝ, y = x + 1}
def B : Set ℝ := {y | ∃ x : ℝ, y = 2 * x}

theorem intersection_of_A_and_B :
  A ∩ B = {y : ℝ | y ≥ 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1737_173783


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1737_173700

theorem diophantine_equation_solution : ∃ (u v : ℤ), 364 * u + 154 * v = 14 ∧ u = 3 ∧ v = -7 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1737_173700


namespace NUMINAMATH_CALUDE_exists_irrational_less_than_four_l1737_173795

theorem exists_irrational_less_than_four : ∃ x : ℝ, Irrational x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_exists_irrational_less_than_four_l1737_173795


namespace NUMINAMATH_CALUDE_knicks_knacks_knocks_conversion_l1737_173734

/-- Given the conversion rates between knicks, knacks, and knocks, 
    this theorem proves that 36 knocks are equivalent to 40 knicks. -/
theorem knicks_knacks_knocks_conversion : 
  ∀ (knick knack knock : ℚ),
  (5 * knick = 3 * knack) →
  (4 * knack = 6 * knock) →
  (36 * knock = 40 * knick) := by
sorry

end NUMINAMATH_CALUDE_knicks_knacks_knocks_conversion_l1737_173734


namespace NUMINAMATH_CALUDE_line_circle_intersection_range_l1737_173772

/-- The equation of the circle C -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 8*x + 12 = 0

/-- The equation of the line -/
def line_equation (x y k : ℝ) : Prop :=
  y = k*x - 2

/-- The condition for the line to have at least one common point with the circle -/
def has_common_point (k : ℝ) : Prop :=
  ∃ x y : ℝ, circle_equation x y ∧ line_equation x y k

/-- The theorem stating the range of k for which the line has at least one common point with the circle -/
theorem line_circle_intersection_range :
  ∀ k : ℝ, has_common_point k ↔ -4/3 ≤ k ∧ k ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_range_l1737_173772


namespace NUMINAMATH_CALUDE_hair_length_after_growth_and_cut_l1737_173730

theorem hair_length_after_growth_and_cut (x : ℝ) : 
  let initial_length : ℝ := 14
  let growth : ℝ := x
  let cut_length : ℝ := 20
  let final_length : ℝ := initial_length + growth - cut_length
  final_length = x - 6 := by sorry

end NUMINAMATH_CALUDE_hair_length_after_growth_and_cut_l1737_173730


namespace NUMINAMATH_CALUDE_train_theorem_l1737_173717

def train_problem (initial : ℕ) 
  (stop1_off stop1_on : ℕ)
  (stop2_off stop2_on stop2_first_off : ℕ)
  (stop3_off stop3_on stop3_first_off : ℕ)
  (stop4_off stop4_on stop4_second_off : ℕ)
  (stop5_off stop5_on : ℕ) : Prop :=
  let after_stop1 := initial - stop1_off + stop1_on
  let after_stop2 := after_stop1 - stop2_off + stop2_on - stop2_first_off
  let after_stop3 := after_stop2 - stop3_off + stop3_on - stop3_first_off
  let after_stop4 := after_stop3 - stop4_off + stop4_on - stop4_second_off
  let final := after_stop4 - stop5_off + stop5_on
  final = 26

theorem train_theorem : train_problem 48 13 5 9 10 2 7 4 3 16 7 5 8 15 := by
  sorry

end NUMINAMATH_CALUDE_train_theorem_l1737_173717


namespace NUMINAMATH_CALUDE_absolute_value_equals_sqrt_of_square_l1737_173724

theorem absolute_value_equals_sqrt_of_square (x : ℝ) : |x| = Real.sqrt (x^2) := by sorry

end NUMINAMATH_CALUDE_absolute_value_equals_sqrt_of_square_l1737_173724


namespace NUMINAMATH_CALUDE_subset_implies_a_in_set_l1737_173782

def A : Set ℝ := {x | x^2 = 1}
def B (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem subset_implies_a_in_set (a : ℝ) : B a ⊆ A → a ∈ ({-1, 0, 1} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_in_set_l1737_173782


namespace NUMINAMATH_CALUDE_f_increasing_iff_three_solutions_iff_l1737_173701

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x * |x - a| + 2 * x

-- Theorem 1: f(x) is increasing on ℝ iff -2 ≤ a ≤ 2
theorem f_increasing_iff (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ -2 ≤ a ∧ a ≤ 2 :=
sorry

-- Theorem 2: f(x) = bf(a) has three distinct real solutions iff b ∈ (1, 9/8)
theorem three_solutions_iff (a : ℝ) (h : -2 ≤ a ∧ a ≤ 4) :
  (∃ b : ℝ, 1 < b ∧ b < 9/8 ∧ ∃ x y z : ℝ, x < y ∧ y < z ∧
    f a x = b * f a a ∧ f a y = b * f a a ∧ f a z = b * f a a) ↔
  (2 < a ∧ a ≤ 4) :=
sorry

end

end NUMINAMATH_CALUDE_f_increasing_iff_three_solutions_iff_l1737_173701


namespace NUMINAMATH_CALUDE_minimum_bills_for_exchange_l1737_173749

/-- Represents the count of bills for each denomination --/
structure BillCounts where
  hundred : ℕ
  fifty : ℕ
  twenty : ℕ
  ten : ℕ
  five : ℕ
  two : ℕ

/-- Calculates the total value of bills --/
def total_value (bills : BillCounts) : ℕ :=
  100 * bills.hundred + 50 * bills.fifty + 20 * bills.twenty +
  10 * bills.ten + 5 * bills.five + 2 * bills.two

/-- Checks if the bill counts satisfy the given constraints --/
def satisfies_constraints (bills : BillCounts) : Prop :=
  bills.hundred ≥ 3 ∧ bills.fifty ≥ 2 ∧ bills.twenty ≤ 4

/-- Initial bill counts --/
def initial_bills : BillCounts :=
  { hundred := 0, fifty := 12, twenty := 10, ten := 8, five := 15, two := 5 }

/-- Theorem stating the minimum number of bills needed for exchange --/
theorem minimum_bills_for_exchange :
  ∃ (exchange_bills : BillCounts),
    total_value exchange_bills = 3000 ∧
    satisfies_constraints exchange_bills ∧
    exchange_bills.hundred = 18 ∧
    exchange_bills.fifty = 3 ∧
    exchange_bills.twenty = 4 ∧
    exchange_bills.five = 1 ∧
    exchange_bills.ten = 0 ∧
    exchange_bills.two = 0 ∧
    (∀ (other_bills : BillCounts),
      total_value other_bills = 3000 →
      satisfies_constraints other_bills →
      total_value other_bills ≥ total_value exchange_bills) :=
sorry

end NUMINAMATH_CALUDE_minimum_bills_for_exchange_l1737_173749


namespace NUMINAMATH_CALUDE_g_lower_bound_l1737_173773

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a|

-- Define the function g
def g (a x : ℝ) : ℝ := f a x + f a (x + 2)

-- Theorem statement
theorem g_lower_bound (a : ℝ) (h : ∀ x, f a x ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2) :
  ∀ x, g a x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_g_lower_bound_l1737_173773


namespace NUMINAMATH_CALUDE_digit_410_of_7_29_l1737_173767

/-- The decimal expansion of 7/29 has a repeating cycle of 28 digits -/
def cycle_length : ℕ := 28

/-- The repeating cycle of digits in the decimal expansion of 7/29 -/
def repeating_cycle : List ℕ := [2, 4, 1, 3, 7, 9, 3, 1, 0, 3, 4, 4, 8, 2, 7, 5, 8, 6, 2, 0, 6, 8, 9, 6, 5, 5, 1, 7]

/-- The position we're interested in -/
def target_position : ℕ := 410

theorem digit_410_of_7_29 : 
  (repeating_cycle.get! ((target_position - 1) % cycle_length)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_digit_410_of_7_29_l1737_173767


namespace NUMINAMATH_CALUDE_line_symmetry_l1737_173720

/-- Given a line l₁: y = 2x + 1 and a point p: (1, 1), 
    the line l₂: y = 2x - 3 is symmetric to l₁ about p -/
theorem line_symmetry (x y : ℝ) : 
  (y = 2*x + 1) → 
  (∃ (x' y' : ℝ), y' = 2*x' - 3 ∧ 
    ((x + x') / 2 = 1 ∧ (y + y') / 2 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_line_symmetry_l1737_173720


namespace NUMINAMATH_CALUDE_store_credit_card_discount_l1737_173761

theorem store_credit_card_discount (original_price sale_discount_percent coupon_discount total_savings : ℝ) :
  original_price = 125 ∧
  sale_discount_percent = 20 ∧
  coupon_discount = 10 ∧
  total_savings = 44 →
  let sale_discount := original_price * (sale_discount_percent / 100)
  let price_after_sale := original_price - sale_discount
  let price_after_coupon := price_after_sale - coupon_discount
  let store_credit_discount := total_savings - sale_discount - coupon_discount
  (store_credit_discount / price_after_coupon) * 100 = 10 := by sorry

end NUMINAMATH_CALUDE_store_credit_card_discount_l1737_173761


namespace NUMINAMATH_CALUDE_remaining_slices_l1737_173796

def total_slices : ℕ := 2 * 8

def slices_after_friends : ℕ := total_slices - (total_slices / 4)

def slices_after_family : ℕ := slices_after_friends - (slices_after_friends / 3)

def slices_after_alex : ℕ := slices_after_family - 3

theorem remaining_slices : slices_after_alex = 5 := by
  sorry

end NUMINAMATH_CALUDE_remaining_slices_l1737_173796


namespace NUMINAMATH_CALUDE_solution_in_interval_l1737_173739

open Real

theorem solution_in_interval :
  ∃! x₀ : ℝ, 2 < x₀ ∧ x₀ < 3 ∧ Real.log x₀ + x₀ = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_in_interval_l1737_173739


namespace NUMINAMATH_CALUDE_sqrt_sum_reciprocals_l1737_173789

theorem sqrt_sum_reciprocals : Real.sqrt (1 / 4 + 1 / 25) = Real.sqrt 29 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_reciprocals_l1737_173789


namespace NUMINAMATH_CALUDE_roselyn_initial_books_l1737_173741

def books_problem (books_to_rebecca : ℕ) (books_remaining : ℕ) : Prop :=
  let books_to_mara := 3 * books_to_rebecca
  let total_given := books_to_rebecca + books_to_mara
  let initial_books := total_given + books_remaining
  initial_books = 220

theorem roselyn_initial_books :
  books_problem 40 60 := by
  sorry

end NUMINAMATH_CALUDE_roselyn_initial_books_l1737_173741


namespace NUMINAMATH_CALUDE_clothing_factory_payment_theorem_l1737_173785

/-- Represents the payment calculation for two discount plans in a clothing factory. -/
def ClothingFactoryPayment (x : ℕ) : Prop :=
  let suitPrice : ℕ := 400
  let tiePrice : ℕ := 80
  let numSuits : ℕ := 20
  let y₁ : ℕ := suitPrice * numSuits + (x - numSuits) * tiePrice
  let y₂ : ℕ := (suitPrice * numSuits + tiePrice * x) * 9 / 10
  (x > 20) →
  (y₁ = 80 * x + 6400) ∧
  (y₂ = 72 * x + 7200) ∧
  (x = 30 → y₁ < y₂)

theorem clothing_factory_payment_theorem :
  ∀ x : ℕ, ClothingFactoryPayment x :=
sorry

end NUMINAMATH_CALUDE_clothing_factory_payment_theorem_l1737_173785


namespace NUMINAMATH_CALUDE_grandfather_grandmother_age_difference_is_two_l1737_173777

/-- The age difference between Milena's grandfather and grandmother -/
def grandfather_grandmother_age_difference (milena_age : ℕ) (grandmother_age_factor : ℕ) (milena_grandfather_age_difference : ℕ) : ℕ :=
  (milena_age + milena_grandfather_age_difference) - (milena_age * grandmother_age_factor)

theorem grandfather_grandmother_age_difference_is_two :
  grandfather_grandmother_age_difference 7 9 58 = 2 := by
  sorry

end NUMINAMATH_CALUDE_grandfather_grandmother_age_difference_is_two_l1737_173777


namespace NUMINAMATH_CALUDE_probability_shirt_shorts_hat_l1737_173794

/-- The number of shirts in the drawer -/
def num_shirts : ℕ := 6

/-- The number of pairs of shorts in the drawer -/
def num_shorts : ℕ := 7

/-- The number of pairs of socks in the drawer -/
def num_socks : ℕ := 6

/-- The number of hats in the drawer -/
def num_hats : ℕ := 3

/-- The total number of articles of clothing in the drawer -/
def total_articles : ℕ := num_shirts + num_shorts + num_socks + num_hats

/-- The number of articles to be chosen -/
def num_chosen : ℕ := 3

theorem probability_shirt_shorts_hat : 
  (num_shirts.choose 1 * num_shorts.choose 1 * num_hats.choose 1 : ℚ) / 
  (total_articles.choose num_chosen) = 63 / 770 :=
sorry

end NUMINAMATH_CALUDE_probability_shirt_shorts_hat_l1737_173794


namespace NUMINAMATH_CALUDE_percentage_difference_l1737_173799

theorem percentage_difference : (0.80 * 45) - ((4 : ℚ) / 5 * 25) = 16 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1737_173799


namespace NUMINAMATH_CALUDE_min_a_value_l1737_173732

-- Define the conditions
def p (x : ℝ) : Prop := |x + 1| ≤ 2
def q (x a : ℝ) : Prop := x ≤ a

-- Define what it means for p to be sufficient but not necessary for q
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p x → q x a) ∧ ¬(∀ x, q x a → p x)

-- State the theorem
theorem min_a_value :
  ∀ a : ℝ, sufficient_not_necessary a → a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_min_a_value_l1737_173732


namespace NUMINAMATH_CALUDE_probability_at_least_one_B_l1737_173705

/-- The probability of selecting at least one question of type B when randomly choosing 2 questions out of 5, where 2 are of type A and 3 are of type B -/
theorem probability_at_least_one_B (total : Nat) (type_A : Nat) (type_B : Nat) (select : Nat) : 
  total = 5 → type_A = 2 → type_B = 3 → select = 2 →
  (Nat.choose total select - Nat.choose type_A select) / Nat.choose total select = 9 / 10 := by
sorry


end NUMINAMATH_CALUDE_probability_at_least_one_B_l1737_173705


namespace NUMINAMATH_CALUDE_multiplicative_inverse_123_mod_455_l1737_173792

theorem multiplicative_inverse_123_mod_455 : ∃ x : ℕ, x < 455 ∧ (123 * x) % 455 = 1 :=
by
  use 223
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_123_mod_455_l1737_173792


namespace NUMINAMATH_CALUDE_estimate_fish_population_l1737_173733

/-- Estimates the number of fish in a pond using the capture-recapture method. -/
theorem estimate_fish_population (initial_marked : ℕ) (second_sample : ℕ) (marked_in_second : ℕ) :
  initial_marked = 200 →
  second_sample = 100 →
  marked_in_second = 20 →
  (initial_marked * second_sample) / marked_in_second = 1000 :=
by
  sorry

#check estimate_fish_population

end NUMINAMATH_CALUDE_estimate_fish_population_l1737_173733


namespace NUMINAMATH_CALUDE_triangle_properties_l1737_173719

-- Define the triangle ABC
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 4)

-- Define the equations of median and altitude
def median_eq (x y : ℝ) : Prop := 2 * x + y - 7 = 0
def altitude_eq (x y : ℝ) : Prop := 2 * x - y - 2 = 0

-- Define vertex C
def C : ℝ × ℝ := (3, 1)

-- Define the area of the triangle
def triangle_area : ℝ := 3

-- Theorem statement
theorem triangle_properties :
  median_eq (C.1) (C.2) ∧ 
  altitude_eq (C.1) (C.2) →
  C = (3, 1) ∧ 
  triangle_area = 3 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1737_173719


namespace NUMINAMATH_CALUDE_lcm_gcf_relations_l1737_173702

theorem lcm_gcf_relations :
  (∃! n : ℕ, Nat.lcm n 16 = 52 ∧ Nat.gcd n 16 = 8) ∧
  (¬ ∃ n : ℕ, Nat.lcm n 20 = 84 ∧ Nat.gcd n 20 = 4) ∧
  (∃! n : ℕ, Nat.lcm n 24 = 120 ∧ Nat.gcd n 24 = 6) := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_relations_l1737_173702


namespace NUMINAMATH_CALUDE_cylinder_cone_sphere_volumes_l1737_173727

/-- Given a cylinder with volume 72π, prove the volumes of a cone and sphere with related dimensions. -/
theorem cylinder_cone_sphere_volumes (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  π * r^2 * h = 72 * π → 
  (1/3 : ℝ) * π * r^2 * h = 24 * π ∧ 
  (4/3 : ℝ) * π * (h/2)^3 = 12 * r * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_cone_sphere_volumes_l1737_173727


namespace NUMINAMATH_CALUDE_rose_count_prediction_l1737_173758

/-- Given a sequence of rose counts for four consecutive months, where the differences
    between consecutive counts form an arithmetic sequence with a common difference of 12,
    prove that the next term in the sequence will be 224. -/
theorem rose_count_prediction (a b c d : ℕ) (hab : b - a = 18) (hbc : c - b = 30) (hcd : d - c = 42) :
  d + (d - c + 12) = 224 :=
sorry

end NUMINAMATH_CALUDE_rose_count_prediction_l1737_173758


namespace NUMINAMATH_CALUDE_hexagram_arrangement_exists_and_unique_l1737_173713

def Hexagram := Fin 7 → Fin 7

def is_valid_arrangement (h : Hexagram) : Prop :=
  (∀ i : Fin 7, ∃! j : Fin 7, h j = i) ∧
  (h 0 + h 1 + h 3 = 12) ∧
  (h 0 + h 2 + h 4 = 12) ∧
  (h 1 + h 2 + h 5 = 12) ∧
  (h 3 + h 4 + h 5 = 12) ∧
  (h 0 + h 6 + h 5 = 12) ∧
  (h 1 + h 6 + h 4 = 12) ∧
  (h 2 + h 6 + h 3 = 12)

theorem hexagram_arrangement_exists_and_unique :
  ∃! h : Hexagram, is_valid_arrangement h :=
sorry

end NUMINAMATH_CALUDE_hexagram_arrangement_exists_and_unique_l1737_173713


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l1737_173766

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The problem statement -/
theorem base_conversion_subtraction :
  let base_9_number := [4, 2, 3]  -- 324 in base 9 (least significant digit first)
  let base_6_number := [1, 2, 2]  -- 221 in base 6 (least significant digit first)
  (to_base_10 base_9_number 9) - (to_base_10 base_6_number 6) = 180 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l1737_173766


namespace NUMINAMATH_CALUDE_isosceles_triangle_solution_l1737_173710

/-- Represents a system of linear equations in two variables -/
structure LinearSystem where
  eq1 : ℝ → ℝ → ℝ → ℝ
  eq2 : ℝ → ℝ → ℝ → ℝ

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  leg : ℝ
  base : ℝ

/-- The main theorem -/
theorem isosceles_triangle_solution (a : ℝ) : 
  let system : LinearSystem := {
    eq1 := fun x y a => 3 * x - y - (2 * a - 5)
    eq2 := fun x y a => x + 2 * y - (3 * a + 3)
  }
  let x := a - 1
  let y := a + 2
  (x > 0 ∧ y > 0) →
  (∃ t : IsoscelesTriangle, t.leg = x ∧ t.base = y ∧ 2 * t.leg + t.base = 12) →
  system.eq1 x y a = 0 ∧ system.eq2 x y a = 0 →
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_solution_l1737_173710


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l1737_173754

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p, p < k → Nat.Prime p → ¬(p ∣ n)

theorem smallest_composite_no_small_factors :
  (is_composite 529) ∧
  (has_no_prime_factors_less_than 529 20) ∧
  (∀ m : ℕ, m < 529 → ¬(is_composite m ∧ has_no_prime_factors_less_than m 20)) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l1737_173754


namespace NUMINAMATH_CALUDE_rectangle_area_double_triangle_area_double_circle_area_quadruple_fraction_unchanged_triple_negative_more_negative_all_statements_correct_l1737_173770

-- Statement A
theorem rectangle_area_double (b h : ℝ) (h_pos : 0 < h) :
  2 * (b * h) = b * (2 * h) := by sorry

-- Statement B
theorem triangle_area_double (b h : ℝ) (h_pos : 0 < h) :
  2 * ((1/2) * b * h) = (1/2) * (2 * b) * h := by sorry

-- Statement C
theorem circle_area_quadruple (r : ℝ) (r_pos : 0 < r) :
  (π * (2 * r)^2) = 4 * (π * r^2) := by sorry

-- Statement D
theorem fraction_unchanged (a b : ℝ) (b_nonzero : b ≠ 0) :
  (2 * a) / (2 * b) = a / b := by sorry

-- Statement E
theorem triple_negative_more_negative (x : ℝ) (x_neg : x < 0) :
  3 * x < x := by sorry

-- All statements are correct
theorem all_statements_correct :
  (∀ b h, 0 < h → 2 * (b * h) = b * (2 * h)) ∧
  (∀ b h, 0 < h → 2 * ((1/2) * b * h) = (1/2) * (2 * b) * h) ∧
  (∀ r, 0 < r → (π * (2 * r)^2) = 4 * (π * r^2)) ∧
  (∀ a b, b ≠ 0 → (2 * a) / (2 * b) = a / b) ∧
  (∀ x, x < 0 → 3 * x < x) := by sorry

end NUMINAMATH_CALUDE_rectangle_area_double_triangle_area_double_circle_area_quadruple_fraction_unchanged_triple_negative_more_negative_all_statements_correct_l1737_173770


namespace NUMINAMATH_CALUDE_parabola_intersection_l1737_173762

/-- Given a parabola y = ax^2 + x + c that intersects the x-axis at x = 1, prove that a + c = -1 -/
theorem parabola_intersection (a c : ℝ) : 
  (∀ x, a*x^2 + x + c = 0 → x = 1) → a + c = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_l1737_173762


namespace NUMINAMATH_CALUDE_max_integer_values_quadratic_l1737_173716

/-- Given a quadratic function f(x) = ax² + bx + c where a > 100,
    the maximum number of integer x values satisfying |f(x)| ≤ 50 is 2 -/
theorem max_integer_values_quadratic (a b c : ℝ) (ha : a > 100) :
  (∃ (n : ℕ), ∀ (S : Finset ℤ),
    (∀ x ∈ S, |a * x^2 + b * x + c| ≤ 50) →
    S.card ≤ n) ∧
  (∃ (S : Finset ℤ), (∀ x ∈ S, |a * x^2 + b * x + c| ≤ 50) ∧ S.card = 2) :=
sorry

end NUMINAMATH_CALUDE_max_integer_values_quadratic_l1737_173716


namespace NUMINAMATH_CALUDE_gcd_of_90_and_405_l1737_173786

theorem gcd_of_90_and_405 : Nat.gcd 90 405 = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_90_and_405_l1737_173786


namespace NUMINAMATH_CALUDE_gcd_204_85_l1737_173763

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_204_85_l1737_173763


namespace NUMINAMATH_CALUDE_ratio_problem_l1737_173750

theorem ratio_problem (x y z : ℚ) (h1 : x / y = 3) (h2 : z / y = 4) :
  (y + z) / (x + z) = 5 / 7 := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l1737_173750


namespace NUMINAMATH_CALUDE_task_completion_time_l1737_173711

/-- Given workers A, B, and C with work rates a, b, and c respectively,
    if A and B together complete a task in 8 hours,
    A and C together complete it in 6 hours,
    and B and C together complete it in 4.8 hours,
    then A, B, and C working together will complete the task in 4 hours. -/
theorem task_completion_time (a b c : ℝ) 
  (hab : 8 * (a + b) = 1)
  (hac : 6 * (a + c) = 1)
  (hbc : 4.8 * (b + c) = 1) :
  (a + b + c)⁻¹ = 4 := by sorry

end NUMINAMATH_CALUDE_task_completion_time_l1737_173711


namespace NUMINAMATH_CALUDE_min_squares_sum_l1737_173759

theorem min_squares_sum (n : ℕ) (h1 : n < 8) (h2 : ∃ a : ℕ, 3 * n + 1 = a ^ 2) :
  (∃ k : ℕ, (∃ (x y z : ℕ), n + 1 = x^2 + y^2 + z^2) ∧
            (∀ m : ℕ, m < k → ¬∃ (a b c : ℕ), n + 1 = a^2 + b^2 + c^2 ∧ 
              (∀ i : ℕ, i > m → c^2 = 0))) ∧
  (∀ k : ℕ, (∃ (x y z : ℕ), n + 1 = x^2 + y^2 + z^2) ∧
            (∀ m : ℕ, m < k → ¬∃ (a b c : ℕ), n + 1 = a^2 + b^2 + c^2 ∧ 
              (∀ i : ℕ, i > m → c^2 = 0)) → k ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_min_squares_sum_l1737_173759


namespace NUMINAMATH_CALUDE_frog_jump_probability_l1737_173729

-- Define the square
def Square := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 6}

-- Define a valid jump
def ValidJump (p q : ℝ × ℝ) : Prop :=
  (p.1 = q.1 ∧ |p.2 - q.2| = 1) ∨ (p.2 = q.2 ∧ |p.1 - q.1| = 1)

-- Define the boundary of the square
def Boundary (p : ℝ × ℝ) : Prop :=
  p.1 = 0 ∨ p.1 = 6 ∨ p.2 = 0 ∨ p.2 = 6

-- Define vertical sides
def VerticalSide (p : ℝ × ℝ) : Prop :=
  (p.1 = 0 ∨ p.1 = 6) ∧ 0 ≤ p.2 ∧ p.2 ≤ 6

-- Define the probability function
noncomputable def P (p : ℝ × ℝ) : ℝ :=
  sorry -- The actual implementation would go here

-- State the theorem
theorem frog_jump_probability :
  P (2, 3) = 3/5 :=
sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l1737_173729


namespace NUMINAMATH_CALUDE_arithmetic_sequence_divisibility_l1737_173704

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_divisibility
  (a : ℕ → ℕ)
  (h_arith : is_arithmetic_sequence a)
  (h_div : ∀ n : ℕ, 2005 ∣ a n * a (n + 31)) :
  ∀ n : ℕ, 2005 ∣ a n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_divisibility_l1737_173704


namespace NUMINAMATH_CALUDE_solution_difference_l1737_173779

theorem solution_difference : ∃ p q : ℝ, 
  (p - 4) * (p + 4) = 24 * p - 96 ∧ 
  (q - 4) * (q + 4) = 24 * q - 96 ∧ 
  p ≠ q ∧ 
  p > q ∧ 
  p - q = 16 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l1737_173779


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_a_equals_two_sufficient_a_equals_two_not_necessary_a_equals_two_sufficient_not_necessary_l1737_173731

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∃ x : ℝ, x^2 - 3*x + a = 0) ↔ a ≤ 9/4 :=
by sorry

theorem a_equals_two_sufficient (x : ℝ) :
  x^2 - 3*x + 2 = 0 → ∃ y : ℝ, y^2 - 3*y + 2 = 0 :=
by sorry

theorem a_equals_two_not_necessary :
  ∃ a : ℝ, a ≠ 2 ∧ (∃ x : ℝ, x^2 - 3*x + a = 0) :=
by sorry

theorem a_equals_two_sufficient_not_necessary :
  (∀ x : ℝ, x^2 - 3*x + 2 = 0 → ∃ y : ℝ, y^2 - 3*y + 2 = 0) ∧
  (∃ a : ℝ, a ≠ 2 ∧ (∃ x : ℝ, x^2 - 3*x + a = 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_a_equals_two_sufficient_a_equals_two_not_necessary_a_equals_two_sufficient_not_necessary_l1737_173731


namespace NUMINAMATH_CALUDE_dress_design_count_l1737_173709

/-- The number of fabric colors available -/
def num_colors : Nat := 5

/-- The number of patterns available -/
def num_patterns : Nat := 5

/-- The number of fabric materials available -/
def num_materials : Nat := 2

/-- A dress design consists of exactly one color, one pattern, and one material -/
structure DressDesign where
  color : Fin num_colors
  pattern : Fin num_patterns
  material : Fin num_materials

/-- The total number of possible dress designs -/
def total_designs : Nat := num_colors * num_patterns * num_materials

theorem dress_design_count : total_designs = 50 := by
  sorry

end NUMINAMATH_CALUDE_dress_design_count_l1737_173709


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1737_173725

theorem absolute_value_inequality (x : ℝ) :
  2 ≤ |x - 3| ∧ |x - 3| ≤ 8 ↔ (-5 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 11) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1737_173725


namespace NUMINAMATH_CALUDE_wenlock_years_ago_l1737_173718

/-- The year when the Wenlock Olympian Games were first held -/
def wenlock_first_year : ℕ := 1850

/-- The reference year (when the Olympic Games mascot 'Wenlock' was named) -/
def reference_year : ℕ := 2012

/-- The number of years between the first Wenlock Olympian Games and the reference year -/
def years_difference : ℕ := reference_year - wenlock_first_year

theorem wenlock_years_ago : years_difference = 162 := by
  sorry

end NUMINAMATH_CALUDE_wenlock_years_ago_l1737_173718


namespace NUMINAMATH_CALUDE_special_circle_equation_l1737_173765

/-- A circle passing through the origin and point (1, 1) with its center on the line 2x + 3y + 1 = 0 -/
def special_circle (x y : ℝ) : Prop :=
  (x - 4)^2 + (y + 3)^2 = 25

/-- The line on which the center of the circle lies -/
def center_line (x y : ℝ) : Prop :=
  2*x + 3*y + 1 = 0

theorem special_circle_equation :
  ∀ x y : ℝ,
  (special_circle x y ↔
    (x^2 + y^2 = 0 ∨ (x - 1)^2 + (y - 1)^2 = 0) ∧
    ∃ c_x c_y : ℝ, center_line c_x c_y ∧ (x - c_x)^2 + (y - c_y)^2 = (c_x^2 + c_y^2)) :=
by sorry

end NUMINAMATH_CALUDE_special_circle_equation_l1737_173765


namespace NUMINAMATH_CALUDE_football_team_right_handed_count_l1737_173790

theorem football_team_right_handed_count 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (left_handed_fraction : ℚ) :
  total_players = 120 →
  throwers = 67 →
  left_handed_fraction = 2 / 5 →
  ∃ (right_handed : ℕ), 
    right_handed = throwers + (total_players - throwers - Int.floor ((total_players - throwers : ℚ) * left_handed_fraction)) ∧
    right_handed = 99 :=
by sorry

end NUMINAMATH_CALUDE_football_team_right_handed_count_l1737_173790


namespace NUMINAMATH_CALUDE_smallest_addition_for_divisibility_by_11_l1737_173757

theorem smallest_addition_for_divisibility_by_11 (n : ℕ) (h : n = 8261955) :
  ∃ k : ℕ, k > 0 ∧ (n + k) % 11 = 0 ∧ ∀ m : ℕ, m > 0 → (n + m) % 11 = 0 → k ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_addition_for_divisibility_by_11_l1737_173757


namespace NUMINAMATH_CALUDE_tetrahedron_properties_l1737_173755

-- Define the vertices of the tetrahedron
def A₁ : ℝ × ℝ × ℝ := (4, -1, 3)
def A₂ : ℝ × ℝ × ℝ := (-2, 1, 0)
def A₃ : ℝ × ℝ × ℝ := (0, -5, 1)
def A₄ : ℝ × ℝ × ℝ := (3, 2, -6)

-- Function to calculate the volume of a tetrahedron
def tetrahedron_volume (a b c d : ℝ × ℝ × ℝ) : ℝ := sorry

-- Function to calculate the height from a point to a plane defined by three points
def height_to_plane (point plane1 plane2 plane3 : ℝ × ℝ × ℝ) : ℝ := sorry

theorem tetrahedron_properties :
  let volume := tetrahedron_volume A₁ A₂ A₃ A₄
  let height := height_to_plane A₄ A₁ A₂ A₃
  volume = 136 / 3 ∧ height = 17 / Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_properties_l1737_173755


namespace NUMINAMATH_CALUDE_saras_result_unique_l1737_173791

/-- Represents a student's exam results -/
structure ExamResult where
  correct : ℕ
  wrong : ℕ
  unanswered : ℕ

/-- Calculates the score based on the exam result -/
def calculateScore (result : ExamResult) : ℕ :=
  30 + 5 * result.correct - 2 * result.wrong - result.unanswered

/-- Theorem: Sara's exam result is uniquely determined -/
theorem saras_result_unique :
  ∃! result : ExamResult,
    result.correct + result.wrong + result.unanswered = 30 ∧
    calculateScore result = 90 ∧
    (∀ s : ℕ, 85 < s ∧ s < 90 → 
      ∃ r1 r2 : ExamResult, r1 ≠ r2 ∧ 
        calculateScore r1 = s ∧ 
        calculateScore r2 = s ∧
        r1.correct + r1.wrong + r1.unanswered = 30 ∧
        r2.correct + r2.wrong + r2.unanswered = 30) ∧
    result.correct = 12 := by
  sorry

#check saras_result_unique

end NUMINAMATH_CALUDE_saras_result_unique_l1737_173791


namespace NUMINAMATH_CALUDE_integer_solutions_quadratic_equation_l1737_173775

theorem integer_solutions_quadratic_equation :
  ∀ x y : ℤ, x^2 + 2*x*y + 3*y^2 - 2*x + y + 1 = 0 ↔ 
  (x = 1 ∧ y = 0) ∨ (x = 1 ∧ y = -1) ∨ (x = 3 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_quadratic_equation_l1737_173775


namespace NUMINAMATH_CALUDE_average_of_data_l1737_173715

def data : List ℝ := [4, 6, 5, 8, 7, 6]

theorem average_of_data :
  (data.sum / data.length : ℝ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_average_of_data_l1737_173715


namespace NUMINAMATH_CALUDE_remainder_98_102_div_12_l1737_173756

theorem remainder_98_102_div_12 : (98 * 102) % 12 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_98_102_div_12_l1737_173756


namespace NUMINAMATH_CALUDE_last_digit_of_product_l1737_173787

theorem last_digit_of_product : (3^65 * 6^59 * 7^71) % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_product_l1737_173787


namespace NUMINAMATH_CALUDE_intersection_segment_length_l1737_173797

-- Define the parabola and line
def parabola (x y : ℝ) : Prop := x^2 = -4*y
def line (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ line A.1 A.2 ∧
  parabola B.1 B.2 ∧ line B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem intersection_segment_length :
  ∀ A B : ℝ × ℝ, intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 := by sorry

end NUMINAMATH_CALUDE_intersection_segment_length_l1737_173797


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l1737_173748

def b (n : ℕ) : ℕ := (n^2).factorial + n

theorem max_gcd_consecutive_terms : 
  ∃ (k : ℕ), k ≥ 1 ∧ Nat.gcd (b k) (b (k+1)) = 2 ∧ 
  ∀ (n : ℕ), n ≥ 1 → Nat.gcd (b n) (b (n+1)) ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l1737_173748


namespace NUMINAMATH_CALUDE_simplify_calculations_l1737_173721

theorem simplify_calculations :
  ((999 : ℕ) * 999 + 1999 = 1000000) ∧
  ((9 : ℕ) * 72 * 125 = 81000) ∧
  ((416 : ℤ) - 327 + 184 - 273 = 0) := by
  sorry

end NUMINAMATH_CALUDE_simplify_calculations_l1737_173721


namespace NUMINAMATH_CALUDE_sqrt_product_equals_sqrt_of_product_l1737_173753

theorem sqrt_product_equals_sqrt_of_product :
  Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_sqrt_of_product_l1737_173753


namespace NUMINAMATH_CALUDE_june_rainfall_l1737_173747

def rainfall_march : ℝ := 3.79
def rainfall_april : ℝ := 4.5
def rainfall_may : ℝ := 3.95
def rainfall_july : ℝ := 4.67
def average_rainfall : ℝ := 4
def num_months : ℕ := 5

theorem june_rainfall :
  let total_rainfall := average_rainfall * num_months
  let known_rainfall := rainfall_march + rainfall_april + rainfall_may + rainfall_july
  let june_rainfall := total_rainfall - known_rainfall
  june_rainfall = 3.09 := by sorry

end NUMINAMATH_CALUDE_june_rainfall_l1737_173747


namespace NUMINAMATH_CALUDE_valid_outfit_choices_l1737_173712

/-- Represents the number of types of each clothing item -/
def num_types : ℕ := 8

/-- Represents the number of colors available -/
def num_colors : ℕ := 8

/-- Calculates the total number of outfit combinations -/
def total_combinations : ℕ := num_types^4

/-- Calculates the number of outfits where all items are the same color -/
def same_color_outfits : ℕ := num_colors

/-- Theorem: The number of valid outfit choices is 4088 -/
theorem valid_outfit_choices : 
  total_combinations - same_color_outfits = 4088 := by sorry

end NUMINAMATH_CALUDE_valid_outfit_choices_l1737_173712


namespace NUMINAMATH_CALUDE_dereks_initial_lunch_spending_l1737_173742

/-- Represents the problem of determining Derek's initial lunch spending --/
theorem dereks_initial_lunch_spending 
  (derek_initial : ℕ) 
  (derek_dad_lunch : ℕ) 
  (derek_extra_lunch : ℕ) 
  (dave_initial : ℕ) 
  (dave_mom_lunch : ℕ) 
  (dave_extra : ℕ) 
  (h1 : derek_initial = 40)
  (h2 : derek_dad_lunch = 11)
  (h3 : derek_extra_lunch = 5)
  (h4 : dave_initial = 50)
  (h5 : dave_mom_lunch = 7)
  (h6 : dave_extra = 33)
  : ∃ (derek_self_lunch : ℕ), 
    derek_self_lunch = 14 ∧ 
    dave_initial - dave_mom_lunch = 
    (derek_initial - (derek_self_lunch + derek_dad_lunch + derek_extra_lunch)) + dave_extra :=
by sorry

end NUMINAMATH_CALUDE_dereks_initial_lunch_spending_l1737_173742


namespace NUMINAMATH_CALUDE_min_distance_ellipse_to_N_l1737_173798

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x + 3)^2 + y^2) + Real.sqrt ((x - 3)^2 + y^2) = 10

/-- The fixed point N -/
def N : ℝ × ℝ := (-6, 0)

/-- The minimum distance from a point on the ellipse to N -/
def min_distance_to_N : ℝ := 1

/-- Theorem stating the minimum distance from any point on the ellipse to N is 1 -/
theorem min_distance_ellipse_to_N :
  ∀ x y : ℝ, ellipse_equation x y →
  ∃ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y ∧
  ∀ (q : ℝ × ℝ), ellipse_equation q.1 q.2 →
  dist p N ≤ dist q N ∧ dist p N = min_distance_to_N :=
sorry

end NUMINAMATH_CALUDE_min_distance_ellipse_to_N_l1737_173798


namespace NUMINAMATH_CALUDE_fraction_cube_three_fourths_cubed_l1737_173745

theorem fraction_cube (a b : ℚ) : (a / b) ^ 3 = a ^ 3 / b ^ 3 := by sorry

theorem three_fourths_cubed : (3 / 4 : ℚ) ^ 3 = 27 / 64 := by sorry

end NUMINAMATH_CALUDE_fraction_cube_three_fourths_cubed_l1737_173745


namespace NUMINAMATH_CALUDE_paper_tray_height_l1737_173738

theorem paper_tray_height (side_length : ℝ) (cut_distance : ℝ) (cut_angle : ℝ) :
  side_length = 120 →
  cut_distance = Real.sqrt 20 →
  cut_angle = π / 4 →
  ∃ (height : ℝ), height = (800 : ℝ) ^ (1/4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_paper_tray_height_l1737_173738


namespace NUMINAMATH_CALUDE_game_result_l1737_173784

def g (n : ℕ) : ℕ :=
  if n % 2 = 0 ∧ n % 5 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

def allie_rolls : List ℕ := [5, 4, 1, 2]
def betty_rolls : List ℕ := [10, 3, 3, 2]

theorem game_result : 
  (List.sum (List.map g allie_rolls)) * (List.sum (List.map g betty_rolls)) = 66 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l1737_173784


namespace NUMINAMATH_CALUDE_binomial_expansion_arithmetic_progression_l1737_173751

theorem binomial_expansion_arithmetic_progression (n : ℕ) : 
  (∃ (a d : ℚ), 
    (1 : ℚ) = a ∧ 
    (n : ℚ) / 2 = a + d ∧ 
    (n * (n - 1) : ℚ) / 8 = a + 2 * d) ↔ 
  n = 8 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_arithmetic_progression_l1737_173751


namespace NUMINAMATH_CALUDE_solve_for_a_l1737_173771

def A (a : ℝ) : Set ℝ := {a - 2, a^2 + 4*a, 10}

theorem solve_for_a : ∀ a : ℝ, -3 ∈ A a → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l1737_173771


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l1737_173706

/-- Given a line with slope 5 passing through the point (-2, 4), 
    prove that the sum of its slope and y-intercept is 19. -/
theorem line_slope_intercept_sum : 
  ∀ (m b : ℝ), 
    m = 5 →                  -- The slope is 5
    4 = m * (-2) + b →       -- The line passes through (-2, 4)
    m + b = 19 :=            -- The sum of slope and y-intercept is 19
by
  sorry


end NUMINAMATH_CALUDE_line_slope_intercept_sum_l1737_173706


namespace NUMINAMATH_CALUDE_jackie_cosmetics_purchase_l1737_173728

/-- The cost of a bottle of lotion -/
def lotion_cost : ℚ := 6

/-- The number of bottles of lotion purchased -/
def lotion_quantity : ℕ := 3

/-- The amount needed to reach the free shipping threshold -/
def additional_amount : ℚ := 12

/-- The free shipping threshold -/
def free_shipping_threshold : ℚ := 50

/-- The cost of a bottle of shampoo or conditioner -/
def shampoo_conditioner_cost : ℚ := 10

theorem jackie_cosmetics_purchase :
  2 * shampoo_conditioner_cost + lotion_cost * lotion_quantity + additional_amount = free_shipping_threshold := by
  sorry

end NUMINAMATH_CALUDE_jackie_cosmetics_purchase_l1737_173728


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1737_173781

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1737_173781


namespace NUMINAMATH_CALUDE_point_on_line_l1737_173726

/-- Given that the point (x, -3) lies on the straight line joining (2, 10) and (6, 2) in the xy-plane, prove that x = 8.5 -/
theorem point_on_line (x : ℝ) :
  (∃ t : ℝ, t ∈ (Set.Icc 0 1) ∧
    x = 2 * (1 - t) + 6 * t ∧
    -3 = 10 * (1 - t) + 2 * t) →
  x = 8.5 := by
sorry

end NUMINAMATH_CALUDE_point_on_line_l1737_173726


namespace NUMINAMATH_CALUDE_monotonicity_and_range_of_a_l1737_173780

noncomputable section

variable (a : ℝ)
variable (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := (x + a) / (a * Real.exp x)

theorem monotonicity_and_range_of_a :
  (a ≠ 0) →
  ((a > 0 → 
    (∀ x₁ x₂, x₁ < 1 - a ∧ x₂ < 1 - a → f a x₁ < f a x₂) ∧
    (∀ x₁ x₂, x₁ > 1 - a ∧ x₂ > 1 - a → f a x₁ > f a x₂)) ∧
   (a < 0 → 
    (∀ x₁ x₂, x₁ < 1 - a ∧ x₂ < 1 - a → f a x₁ > f a x₂) ∧
    (∀ x₁ x₂, x₁ > 1 - a ∧ x₂ > 1 - a → f a x₁ < f a x₂))) ∧
  ((∀ x > 0, (3 + 2 * Real.log x) / Real.exp x ≤ f a x + 2 * x) →
   (a ∈ Set.Iic (-1/2) ∪ Set.Ioi 0)) := by
  sorry

end NUMINAMATH_CALUDE_monotonicity_and_range_of_a_l1737_173780


namespace NUMINAMATH_CALUDE_expression_change_l1737_173793

theorem expression_change (x : ℝ) (b : ℝ) (h : b > 0) :
  (b*x)^2 - 5 - (x^2 - 5) = (b^2 - 1) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_change_l1737_173793


namespace NUMINAMATH_CALUDE_simplify_expression_l1737_173735

theorem simplify_expression : 18 * (14 / 15) * (1 / 12) - (1 / 5) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1737_173735


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1737_173714

theorem sum_of_squares_of_roots (x y : ℝ) : 
  (3 * x^2 - 7 * x + 5 = 0) → 
  (3 * y^2 - 7 * y + 5 = 0) → 
  (x^2 + y^2 = 19/9) := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1737_173714


namespace NUMINAMATH_CALUDE_system_of_inequalities_solution_l1737_173708

theorem system_of_inequalities_solution :
  ∀ x y : ℤ,
    (2 * x - y > 3 ∧ 3 - 2 * x + y > 0) ↔ ((x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = 1)) :=
by sorry

end NUMINAMATH_CALUDE_system_of_inequalities_solution_l1737_173708


namespace NUMINAMATH_CALUDE_base_number_proof_l1737_173788

theorem base_number_proof (x : ℝ) (h : Real.sqrt (x^12) = 64) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l1737_173788


namespace NUMINAMATH_CALUDE_odd_function_value_at_negative_one_l1737_173722

-- Define the function f
noncomputable def f (c : ℝ) : ℝ → ℝ := fun x =>
  if x ≥ 0 then 3^x - 2*x + c else -(3^(-x) - 2*(-x) + c)

-- State the theorem
theorem odd_function_value_at_negative_one (c : ℝ) :
  (∀ x, f c x = -(f c (-x))) → f c (-1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_value_at_negative_one_l1737_173722


namespace NUMINAMATH_CALUDE_two_numbers_difference_l1737_173707

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 8) 
  (square_diff : x^2 - y^2 = 48) : 
  |x - y| = 6 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l1737_173707


namespace NUMINAMATH_CALUDE_game_cost_calculation_l1737_173760

theorem game_cost_calculation (total_earned : ℕ) (blade_cost : ℕ) (num_games : ℕ) 
  (h1 : total_earned = 69)
  (h2 : blade_cost = 24)
  (h3 : num_games = 9)
  (h4 : total_earned ≥ blade_cost) :
  (total_earned - blade_cost) / num_games = 5 := by
  sorry

end NUMINAMATH_CALUDE_game_cost_calculation_l1737_173760


namespace NUMINAMATH_CALUDE_count_numbers_with_property_l1737_173774

/-- Represents a four-digit number -/
def FourDigitNumber := { n : ℕ // 1000 ≤ n ∧ n ≤ 9999 }

/-- Extracts the leftmost digit of a four-digit number -/
def leftmostDigit (n : FourDigitNumber) : ℕ := n.val / 1000

/-- Extracts the three-digit number obtained by removing the leftmost digit -/
def rightThreeDigits (n : FourDigitNumber) : ℕ := n.val % 1000

/-- Checks if a four-digit number satisfies the given property -/
def satisfiesProperty (n : FourDigitNumber) : Prop :=
  7 * (rightThreeDigits n) = n.val

theorem count_numbers_with_property :
  ∃ (S : Finset FourDigitNumber),
    (∀ n ∈ S, satisfiesProperty n) ∧
    (∀ n : FourDigitNumber, satisfiesProperty n → n ∈ S) ∧
    Finset.card S = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_property_l1737_173774


namespace NUMINAMATH_CALUDE_equidistant_points_characterization_l1737_173744

/-- A ray in a plane --/
structure Ray where
  start : ℝ × ℝ
  direction : ℝ × ℝ

/-- The set of points equidistant from two rays --/
def EquidistantPoints (ray1 ray2 : Ray) : Set (ℝ × ℝ) :=
  sorry

/-- Angle bisector of two lines --/
def AngleBisector (line1 line2 : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  sorry

/-- Perpendicular bisector of a segment --/
def PerpendicularBisector (a b : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry

/-- Parabola with focus and directrix --/
def Parabola (focus : ℝ × ℝ) (directrix : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  sorry

/-- The line containing a ray --/
def LineContainingRay (ray : Ray) : Set (ℝ × ℝ) :=
  sorry

theorem equidistant_points_characterization (ray1 ray2 : Ray) :
  EquidistantPoints ray1 ray2 =
    (AngleBisector (LineContainingRay ray1) (LineContainingRay ray2)) ∪
    (if ray1.start ≠ ray2.start then PerpendicularBisector ray1.start ray2.start else ∅) ∪
    (Parabola ray1.start (LineContainingRay ray2)) ∪
    (Parabola ray2.start (LineContainingRay ray1)) :=
  sorry

end NUMINAMATH_CALUDE_equidistant_points_characterization_l1737_173744


namespace NUMINAMATH_CALUDE_triangle_angle_B_l1737_173776

/-- In a triangle ABC, given side lengths and an angle, prove that angle B has two possible values. -/
theorem triangle_angle_B (a b : ℝ) (A B : ℝ) : 
  a = (5 * Real.sqrt 3) / 3 → 
  b = 5 → 
  A = π / 6 → 
  (B = π / 3 ∨ B = 2 * π / 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l1737_173776


namespace NUMINAMATH_CALUDE_jellybean_box_capacity_l1737_173703

theorem jellybean_box_capacity (bert_capacity : ℕ) (scale_factor : ℕ) : 
  bert_capacity = 150 → 
  scale_factor = 3 → 
  (scale_factor ^ 3 : ℕ) * bert_capacity = 4050 := by
sorry

end NUMINAMATH_CALUDE_jellybean_box_capacity_l1737_173703


namespace NUMINAMATH_CALUDE_housing_boom_calculation_l1737_173723

/-- The number of houses in Lawrence County before the housing boom. -/
def houses_before : ℕ := 1426

/-- The number of houses in Lawrence County after the housing boom. -/
def houses_after : ℕ := 2000

/-- The number of houses built during the housing boom. -/
def houses_built : ℕ := houses_after - houses_before

theorem housing_boom_calculation :
  houses_built = 574 :=
by sorry

end NUMINAMATH_CALUDE_housing_boom_calculation_l1737_173723


namespace NUMINAMATH_CALUDE_library_books_problem_l1737_173778

theorem library_books_problem (initial_books : ℕ) : 
  initial_books - 120 + 35 - 15 = 150 → initial_books = 250 := by
  sorry

end NUMINAMATH_CALUDE_library_books_problem_l1737_173778


namespace NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l1737_173768

theorem unique_solution_trigonometric_equation :
  ∃! x : Real, 0 < x ∧ x < 180 ∧
  Real.tan ((150 : Real) * degree - x * degree) = 
    (Real.sin ((150 : Real) * degree) - Real.sin (x * degree)) / 
    (Real.cos ((150 : Real) * degree) - Real.cos (x * degree)) ∧
  x = 105 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l1737_173768


namespace NUMINAMATH_CALUDE_sequence_property_l1737_173740

theorem sequence_property (a : ℕ → ℕ) 
  (h_bijective : Function.Bijective a) 
  (h_positive : ∀ n, a n > 0) : 
  ∃ ℓ m : ℕ, 1 < ℓ ∧ ℓ < m ∧ a 1 + a m = 2 * a ℓ := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l1737_173740


namespace NUMINAMATH_CALUDE_complex_expression_sum_l1737_173736

theorem complex_expression_sum (z : ℂ) : 
  z = Complex.exp (4 * Real.pi * I / 7) →
  z / (1 + z^2) + z^2 / (1 + z^4) + z^3 / (1 + z^6) = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_sum_l1737_173736
