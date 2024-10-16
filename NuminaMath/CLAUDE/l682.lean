import Mathlib

namespace NUMINAMATH_CALUDE_john_payment_john_payment_is_8400_l682_68268

/-- Calculates John's payment for lawyer fees --/
theorem john_payment (upfront_fee : ℕ) (hourly_rate : ℕ) (court_hours : ℕ) 
  (prep_time_multiplier : ℕ) (paperwork_fee : ℕ) (transport_costs : ℕ) : ℕ :=
  let total_hours := court_hours + prep_time_multiplier * court_hours
  let total_fee := upfront_fee + hourly_rate * total_hours + paperwork_fee + transport_costs
  total_fee / 2

/-- Proves that John's payment is $8400 given the specified conditions --/
theorem john_payment_is_8400 : 
  john_payment 1000 100 50 2 500 300 = 8400 := by
  sorry

end NUMINAMATH_CALUDE_john_payment_john_payment_is_8400_l682_68268


namespace NUMINAMATH_CALUDE_second_hole_depth_calculation_l682_68223

/-- Calculates the depth of a second hole given the conditions of two digging projects -/
def second_hole_depth (workers1 hours1 depth1 workers2 hours2 : ℕ) : ℚ :=
  let man_hours1 := workers1 * hours1
  let man_hours2 := workers2 * hours2
  (man_hours2 * depth1 : ℚ) / man_hours1

theorem second_hole_depth_calculation (workers1 hours1 depth1 extra_workers hours2 : ℕ) :
  second_hole_depth workers1 hours1 depth1 (workers1 + extra_workers) hours2 = 40 :=
by
  -- The proof goes here
  sorry

#eval second_hole_depth 45 8 30 80 6

end NUMINAMATH_CALUDE_second_hole_depth_calculation_l682_68223


namespace NUMINAMATH_CALUDE_special_sequence_properties_l682_68270

/-- A sequence satisfying certain conditions -/
structure SpecialSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  p : ℝ
  h1 : a 1 = 2
  h2 : ∀ n, a n ≠ 0
  h3 : ∀ n, a n * a (n + 1) = p * S n + 2
  h4 : ∀ n, S (n + 1) = S n + a (n + 1)

/-- The main theorem about the special sequence -/
theorem special_sequence_properties (seq : SpecialSequence) :
  (∀ n, seq.a (n + 2) - seq.a n = seq.p) ∧
  (∃ p : ℝ, p = 2 ∧ 
    (∃ d : ℝ, ∀ n, |seq.a (n + 1)| - |seq.a n| = d)) :=
sorry

end NUMINAMATH_CALUDE_special_sequence_properties_l682_68270


namespace NUMINAMATH_CALUDE_trevors_age_problem_l682_68215

theorem trevors_age_problem (trevor_age_decade_ago : ℕ) (brother_current_age : ℕ) : 
  trevor_age_decade_ago = 16 →
  brother_current_age = 32 →
  ∃ x : ℕ, x = 20 ∧ 2 * (trevor_age_decade_ago + 10 - x) = brother_current_age - x :=
by sorry

end NUMINAMATH_CALUDE_trevors_age_problem_l682_68215


namespace NUMINAMATH_CALUDE_smallest_w_l682_68216

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w : 
  let w := 2571912
  ∀ x : ℕ, x > 0 →
    (is_factor (2^5) (3692 * x) ∧
     is_factor (3^4) (3692 * x) ∧
     is_factor (7^3) (3692 * x) ∧
     is_factor (17^2) (3692 * x)) →
    x ≥ w :=
by sorry

end NUMINAMATH_CALUDE_smallest_w_l682_68216


namespace NUMINAMATH_CALUDE_smallest_block_size_l682_68283

/-- Represents a rectangular block of cubes -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the total number of cubes in the block -/
def Block.totalCubes (b : Block) : ℕ :=
  b.length * b.width * b.height

/-- Calculates the number of invisible cubes when three faces are shown -/
def Block.invisibleCubes (b : Block) : ℕ :=
  (b.length - 1) * (b.width - 1) * (b.height - 1)

/-- Theorem stating the smallest possible value of N -/
theorem smallest_block_size :
  ∃ (b : Block), b.invisibleCubes = 378 ∧
    b.totalCubes = 560 ∧
    (∀ (b' : Block), b'.invisibleCubes = 378 → b'.totalCubes ≥ 560) := by
  sorry

end NUMINAMATH_CALUDE_smallest_block_size_l682_68283


namespace NUMINAMATH_CALUDE_dimes_in_shorts_l682_68267

/-- Given a total amount of money and the number of dimes in a jacket, 
    calculate the number of dimes in the shorts. -/
theorem dimes_in_shorts 
  (total : ℚ) 
  (jacket_dimes : ℕ) 
  (dime_value : ℚ) 
  (h1 : total = 19/10) 
  (h2 : jacket_dimes = 15) 
  (h3 : dime_value = 1/10) : 
  ↑jacket_dimes * dime_value + 4 * dime_value = total :=
sorry

end NUMINAMATH_CALUDE_dimes_in_shorts_l682_68267


namespace NUMINAMATH_CALUDE_machine_doesnt_require_repair_l682_68250

/-- Represents a weighing machine --/
structure WeighingMachine where
  max_deviation : ℝ
  nominal_mass : ℝ
  all_deviations_bounded : Prop
  standard_deviation_bounded : Prop

/-- Determines if a weighing machine requires repair --/
def requires_repair (m : WeighingMachine) : Prop :=
  m.max_deviation > 0.1 * m.nominal_mass ∨
  ¬m.all_deviations_bounded ∨
  ¬m.standard_deviation_bounded

/-- Theorem stating that the machine does not require repair --/
theorem machine_doesnt_require_repair (m : WeighingMachine)
  (h1 : m.max_deviation = 37)
  (h2 : m.max_deviation ≤ 0.1 * m.nominal_mass)
  (h3 : m.all_deviations_bounded)
  (h4 : m.standard_deviation_bounded) :
  ¬(requires_repair m) :=
sorry

end NUMINAMATH_CALUDE_machine_doesnt_require_repair_l682_68250


namespace NUMINAMATH_CALUDE_grunters_win_probability_l682_68220

theorem grunters_win_probability (num_games : ℕ) (win_prob : ℚ) :
  num_games = 6 →
  win_prob = 3/5 →
  (win_prob ^ num_games : ℚ) = 729/15625 := by
  sorry

end NUMINAMATH_CALUDE_grunters_win_probability_l682_68220


namespace NUMINAMATH_CALUDE_complex_equation_solution_l682_68205

theorem complex_equation_solution (a : ℝ) (z : ℂ) : 
  z * Complex.I = (a + 1 : ℂ) + 4 * Complex.I → Complex.abs z = 5 → a = 2 ∨ a = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l682_68205


namespace NUMINAMATH_CALUDE_no_integer_satisfies_conditions_l682_68260

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating that no positive integer A satisfies the given conditions -/
theorem no_integer_satisfies_conditions : ¬ ∃ A : ℕ+, 
  (sumOfDigits A = 16) ∧ (sumOfDigits (2 * A) = 17) := by sorry

end NUMINAMATH_CALUDE_no_integer_satisfies_conditions_l682_68260


namespace NUMINAMATH_CALUDE_x1_value_l682_68296

theorem x1_value (x1 x2 x3 x4 : Real) 
  (h1 : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1/3) :
  x1 = 4/5 := by sorry

end NUMINAMATH_CALUDE_x1_value_l682_68296


namespace NUMINAMATH_CALUDE_money_distribution_l682_68272

/-- Given a distribution of money in the ratio 3 : 5 : 7 among three people,
    where the second person's share is 1500, 
    the difference between the first and third person's shares is 1200. -/
theorem money_distribution (total : ℕ) (f v r : ℕ) : 
  (f + v + r = total) →
  (3 * v = 5 * f) →
  (5 * r = 7 * v) →
  (v = 1500) →
  (r - f = 1200) :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_l682_68272


namespace NUMINAMATH_CALUDE_total_fish_count_l682_68286

/-- Represents the number of fish in Jonah's aquariums -/
def total_fish (x y : ℕ) : ℤ :=
  let first_aquarium := 14 + 2 - 2 * x + 3
  let second_aquarium := 18 + 4 - 4 * y + 5
  first_aquarium + second_aquarium

/-- The theorem stating the total number of fish in both aquariums -/
theorem total_fish_count (x y : ℕ) : total_fish x y = 46 - 2 * x - 4 * y := by
  sorry

end NUMINAMATH_CALUDE_total_fish_count_l682_68286


namespace NUMINAMATH_CALUDE_function_value_ordering_l682_68221

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function f is increasing on [0, +∞) if f(x) ≤ f(y) for all 0 ≤ x ≤ y -/
def IncreasingOnNonnegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

theorem function_value_ordering (f : ℝ → ℝ) 
    (heven : EvenFunction f) (hincr : IncreasingOnNonnegative f) :
    f 1 < f (-2) ∧ f (-2) < f (-3) := by
  sorry

end NUMINAMATH_CALUDE_function_value_ordering_l682_68221


namespace NUMINAMATH_CALUDE_pauls_crayons_l682_68264

theorem pauls_crayons (erasers : ℕ) (crayons_difference : ℕ) :
  erasers = 457 →
  crayons_difference = 66 →
  erasers + crayons_difference = 523 :=
by sorry

end NUMINAMATH_CALUDE_pauls_crayons_l682_68264


namespace NUMINAMATH_CALUDE_intersection_A_B_l682_68238

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (x - 2)}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Ioo 2 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l682_68238


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l682_68274

theorem negation_of_universal_statement (S : Set ℝ) :
  (¬ ∀ x ∈ S, |x - 1| ≠ 2) ↔ (∃ x ∈ S, |x - 1| = 2) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l682_68274


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l682_68289

/-- The motion equation of an object -/
def s (t : ℝ) : ℝ := t^3 + t^2 - 1

/-- The velocity function derived from the motion equation -/
def v (t : ℝ) : ℝ := 3 * t^2 + 2 * t

theorem instantaneous_velocity_at_3 : v 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l682_68289


namespace NUMINAMATH_CALUDE_sum_real_imag_parts_of_complex_l682_68214

theorem sum_real_imag_parts_of_complex (z : ℂ) : z = (1 + 2*I) / (1 - 2*I) → (z.re + z.im = 1/5) := by
  sorry

end NUMINAMATH_CALUDE_sum_real_imag_parts_of_complex_l682_68214


namespace NUMINAMATH_CALUDE_class_size_is_69_l682_68265

/-- Represents the number of students in a class with given enrollment data for French and German courses -/
def total_students (french : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ) : ℕ :=
  (french + german - both) + neither

/-- Theorem stating that the total number of students in the class is 69 -/
theorem class_size_is_69 :
  total_students 41 22 9 15 = 69 := by
  sorry

end NUMINAMATH_CALUDE_class_size_is_69_l682_68265


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l682_68256

theorem system_of_equations_solution :
  ∃! (a b : ℝ), 3*a + 2*b = -26 ∧ 2*a - b = -22 :=
by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l682_68256


namespace NUMINAMATH_CALUDE_opposite_numbers_sum_l682_68231

theorem opposite_numbers_sum (a b : ℝ) : a + b = 0 → 3*a + 3*b + 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_sum_l682_68231


namespace NUMINAMATH_CALUDE_complement_of_union_equals_singleton_l682_68254

def U : Finset Int := {-3, -2, -1, 0, 1, 2, 3, 4, 5, 6}
def A : Finset Int := {-1, 0, 1, 2, 3}
def B : Finset Int := {-2, 3, 4, 5, 6}

theorem complement_of_union_equals_singleton : U \ (A ∪ B) = {-3} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_singleton_l682_68254


namespace NUMINAMATH_CALUDE_sum_outside_layers_l682_68213

/-- Represents a 3D cube with specific properties -/
structure Cube3D where
  size : Nat
  total_units : Nat
  sum_per_line : ℝ
  special_value : ℝ

/-- Theorem stating the sum of numbers outside three layers in a specific cube -/
theorem sum_outside_layers (c : Cube3D) 
  (h_size : c.size = 20)
  (h_units : c.total_units = 8000)
  (h_sum : c.sum_per_line = 1)
  (h_special : c.special_value = 10) :
  let total_sum := c.size * c.size * c.sum_per_line
  let layer_sum := 3 * c.sum_per_line - 2 * c.sum_per_line + c.special_value
  total_sum - layer_sum = 392 := by
  sorry

end NUMINAMATH_CALUDE_sum_outside_layers_l682_68213


namespace NUMINAMATH_CALUDE_age_ratio_proof_l682_68247

/-- 
Given three people a, b, and c, where:
- a is two years older than b
- The total age of a, b, and c is 27
- b is 10 years old

This theorem proves that the ratio of b's age to c's age is 2:1
-/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →
  a + b + c = 27 →
  b = 10 →
  b = 2 * c :=
by
  sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l682_68247


namespace NUMINAMATH_CALUDE_continuous_fraction_equality_l682_68227

theorem continuous_fraction_equality : 1 + 2 / (3 + 6/7) = 41/27 := by
  sorry

end NUMINAMATH_CALUDE_continuous_fraction_equality_l682_68227


namespace NUMINAMATH_CALUDE_sum_of_fraction_parts_l682_68208

def repeating_decimal : ℚ := 2.5252525

theorem sum_of_fraction_parts : ∃ (n d : ℕ), 
  repeating_decimal = n / d ∧ 
  Nat.gcd n d = 1 ∧ 
  n + d = 349 := by sorry

end NUMINAMATH_CALUDE_sum_of_fraction_parts_l682_68208


namespace NUMINAMATH_CALUDE_people_who_left_gym_l682_68251

theorem people_who_left_gym (initial_people : ℕ) (people_came_in : ℕ) (current_people : ℕ)
  (h1 : initial_people = 16)
  (h2 : people_came_in = 5)
  (h3 : current_people = 19) :
  initial_people + people_came_in - current_people = 2 := by
sorry

end NUMINAMATH_CALUDE_people_who_left_gym_l682_68251


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l682_68218

-- Define the ★ operation
def star (a b : ℤ) : ℤ := a * b - 1

-- Theorem 1
theorem problem_1 : star (-1) 3 = -4 := by sorry

-- Theorem 2
theorem problem_2 : star (-2) (star (-3) (-4)) = -21 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l682_68218


namespace NUMINAMATH_CALUDE_not_in_S_iff_one_or_multiple_of_five_l682_68203

def S : Set Nat := sorry

axiom two_in_S : 2 ∈ S

axiom square_in_S_implies_n_in_S : ∀ n : Nat, n^2 ∈ S → n ∈ S

axiom n_in_S_implies_n_plus_5_squared_in_S : ∀ n : Nat, n ∈ S → (n + 5)^2 ∈ S

axiom S_is_smallest : ∀ T : Set Nat, 
  (2 ∈ T ∧ 
   (∀ n : Nat, n^2 ∈ T → n ∈ T) ∧ 
   (∀ n : Nat, n ∈ T → (n + 5)^2 ∈ T)) → 
  S ⊆ T

theorem not_in_S_iff_one_or_multiple_of_five (n : Nat) :
  n ∉ S ↔ n = 1 ∨ ∃ k : Nat, n = 5 * k :=
sorry

end NUMINAMATH_CALUDE_not_in_S_iff_one_or_multiple_of_five_l682_68203


namespace NUMINAMATH_CALUDE_square_product_inequality_l682_68277

theorem square_product_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > a*b ∧ a*b > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_product_inequality_l682_68277


namespace NUMINAMATH_CALUDE_mark_young_fish_count_l682_68269

/-- The number of tanks Mark has for pregnant fish -/
def num_tanks : ℕ := 3

/-- The number of pregnant fish in each tank -/
def fish_per_tank : ℕ := 4

/-- The number of young fish each pregnant fish gives birth to -/
def young_per_fish : ℕ := 20

/-- The total number of young fish Mark has at the end -/
def total_young_fish : ℕ := num_tanks * fish_per_tank * young_per_fish

theorem mark_young_fish_count : total_young_fish = 240 := by
  sorry

end NUMINAMATH_CALUDE_mark_young_fish_count_l682_68269


namespace NUMINAMATH_CALUDE_debt_average_payment_l682_68298

theorem debt_average_payment 
  (total_installments : ℕ) 
  (first_payment_count : ℕ) 
  (first_payment_amount : ℚ) 
  (additional_amount : ℚ) : 
  total_installments = 100 → 
  first_payment_count = 30 → 
  first_payment_amount = 620 → 
  additional_amount = 110 → 
  let remaining_payment_count := total_installments - first_payment_count
  let remaining_payment_amount := first_payment_amount + additional_amount
  let total_amount := 
    (first_payment_count * first_payment_amount) + 
    (remaining_payment_count * remaining_payment_amount)
  total_amount / total_installments = 697 := by
sorry

end NUMINAMATH_CALUDE_debt_average_payment_l682_68298


namespace NUMINAMATH_CALUDE_platform_length_l682_68245

/-- The length of a platform given train passing times -/
theorem platform_length (train_length : ℝ) (time_pass_man : ℝ) (time_cross_platform : ℝ) 
  (h1 : train_length = 178)
  (h2 : time_pass_man = 8)
  (h3 : time_cross_platform = 20) :
  let train_speed := train_length / time_pass_man
  let platform_length := train_speed * time_cross_platform - train_length
  platform_length = 267 := by
sorry

end NUMINAMATH_CALUDE_platform_length_l682_68245


namespace NUMINAMATH_CALUDE_calculate_expression_l682_68295

theorem calculate_expression : (2200 - 2090)^2 / (144 + 25) = 64 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l682_68295


namespace NUMINAMATH_CALUDE_max_value_of_expression_l682_68206

open Real

theorem max_value_of_expression (t : ℝ) :
  (∃ (max : ℝ), ∀ (t : ℝ), (3^t - 4*t^2)*t / 9^t ≤ max) ∧
  (∃ (t_max : ℝ), (3^t_max - 4*t_max^2)*t_max / 9^t_max = sqrt 3 / 9) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l682_68206


namespace NUMINAMATH_CALUDE_min_probability_cards_unique_min_probability_cards_l682_68280

/-- Represents the probability of a card being red-side up after flips -/
def probability_red_up (k : ℕ) : ℚ :=
  if k ≤ 25 then
    (676 - 52 * k + 2 * k^2) / 676
  else
    (676 - 52 * (51 - k) + 2 * (51 - k)^2) / 676

/-- The statement to prove -/
theorem min_probability_cards :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 50 →
    (probability_red_up 13 ≤ probability_red_up k ∧
     probability_red_up 38 ≤ probability_red_up k) :=
by sorry

/-- Uniqueness of the minimum probability cards -/
theorem unique_min_probability_cards :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 50 →
    (k ≠ 13 ∧ k ≠ 38 →
      probability_red_up 13 < probability_red_up k ∧
      probability_red_up 38 < probability_red_up k) :=
by sorry

end NUMINAMATH_CALUDE_min_probability_cards_unique_min_probability_cards_l682_68280


namespace NUMINAMATH_CALUDE_gcd_1043_2295_l682_68266

theorem gcd_1043_2295 : Nat.gcd 1043 2295 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1043_2295_l682_68266


namespace NUMINAMATH_CALUDE_lucas_50th_term_mod_5_l682_68275

def lucas : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | n + 2 => lucas n + lucas (n + 1)

theorem lucas_50th_term_mod_5 : lucas 49 % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_lucas_50th_term_mod_5_l682_68275


namespace NUMINAMATH_CALUDE_sandy_token_ratio_l682_68259

theorem sandy_token_ratio : 
  ∀ (total_tokens : ℕ) (num_siblings : ℕ) (extra_tokens : ℕ),
    total_tokens = 1000000 →
    num_siblings = 4 →
    extra_tokens = 375000 →
    ∃ (tokens_per_sibling : ℕ),
      tokens_per_sibling * num_siblings + (tokens_per_sibling + extra_tokens) = total_tokens ∧
      (tokens_per_sibling + extra_tokens) * 2 = total_tokens :=
by sorry

end NUMINAMATH_CALUDE_sandy_token_ratio_l682_68259


namespace NUMINAMATH_CALUDE_student_mistake_difference_l682_68242

theorem student_mistake_difference : 
  let number := 384
  let correct_fraction := 5 / 16
  let incorrect_fraction := 5 / 6
  let correct_answer := correct_fraction * number
  let incorrect_answer := incorrect_fraction * number
  incorrect_answer - correct_answer = 200 := by
sorry

end NUMINAMATH_CALUDE_student_mistake_difference_l682_68242


namespace NUMINAMATH_CALUDE_total_hours_worked_l682_68211

/-- Given a person working for 3 days and 2.5 hours each day, 
    the total hours worked is equal to 7.5 hours. -/
theorem total_hours_worked (days : ℕ) (hours_per_day : ℝ) : 
  days = 3 → hours_per_day = 2.5 → days * hours_per_day = 7.5 := by
sorry

end NUMINAMATH_CALUDE_total_hours_worked_l682_68211


namespace NUMINAMATH_CALUDE_bacteria_growth_l682_68258

/-- The number of cells after a given number of days, where the initial population
    doubles every two days. -/
def cell_population (initial_cells : ℕ) (days : ℕ) : ℕ :=
  initial_cells * 2^(days / 2)

/-- Theorem stating that given an initial population of 4 cells that double
    every two days, the number of cells after 10 days is 64. -/
theorem bacteria_growth : cell_population 4 10 = 64 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_l682_68258


namespace NUMINAMATH_CALUDE_millet_percentage_in_brand_A_l682_68239

/-- The percentage of millet in Brand A -/
def millet_in_A : ℝ := 0.4

/-- The percentage of sunflower in Brand A -/
def sunflower_in_A : ℝ := 0.6

/-- The percentage of millet in Brand B -/
def millet_in_B : ℝ := 0.65

/-- The percentage of Brand A in the mix -/
def brand_A_in_mix : ℝ := 0.6

/-- The percentage of Brand B in the mix -/
def brand_B_in_mix : ℝ := 0.4

/-- The percentage of millet in the mix -/
def millet_in_mix : ℝ := 0.5

theorem millet_percentage_in_brand_A :
  millet_in_A * brand_A_in_mix + millet_in_B * brand_B_in_mix = millet_in_mix ∧
  millet_in_A + sunflower_in_A = 1 :=
by sorry

end NUMINAMATH_CALUDE_millet_percentage_in_brand_A_l682_68239


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l682_68225

theorem sum_of_fractions_equals_one
  (a b c p q r : ℝ)
  (eq1 : 19 * p + b * q + c * r = 0)
  (eq2 : a * p + 29 * q + c * r = 0)
  (eq3 : a * p + b * q + 56 * r = 0)
  (ha : a ≠ 19)
  (hp : p ≠ 0) :
  a / (a - 19) + b / (b - 29) + c / (c - 56) = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l682_68225


namespace NUMINAMATH_CALUDE_man_lot_ownership_l682_68240

theorem man_lot_ownership (lot_value : ℝ) (sold_fraction : ℝ) (sold_value : ℝ) :
  lot_value = 9200 →
  sold_fraction = 1 / 10 →
  sold_value = 460 →
  (sold_value / sold_fraction) / lot_value = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_man_lot_ownership_l682_68240


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l682_68246

def P : Set (ℝ × ℝ) := {p | p.1 + p.2 = 3}
def Q : Set (ℝ × ℝ) := {q | q.1 - q.2 = 5}

theorem intersection_of_P_and_Q : P ∩ Q = {(4, -1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l682_68246


namespace NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l682_68210

/-- The y-coordinate of the point on the y-axis that is equidistant from A(3, 0) and B(-4, 5) -/
theorem equidistant_point_y_coordinate : 
  ∃ y : ℝ, (y = 16/5) ∧ 
  ((0 - 3)^2 + (y - 0)^2 = (0 - (-4))^2 + (y - 5)^2) := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l682_68210


namespace NUMINAMATH_CALUDE_grandparents_uncle_difference_l682_68217

/-- Represents the money Gwen received from each family member -/
structure MoneyReceived where
  dad : ℕ
  mom : ℕ
  uncle : ℕ
  aunt : ℕ
  cousin : ℕ
  grandparents : ℕ

/-- The amount of money Gwen received for her birthday -/
def gwens_birthday_money : MoneyReceived :=
  { dad := 5
  , mom := 10
  , uncle := 8
  , aunt := 3
  , cousin := 6
  , grandparents := 15
  }

/-- Theorem stating the difference between money received from grandparents and uncle -/
theorem grandparents_uncle_difference :
  gwens_birthday_money.grandparents - gwens_birthday_money.uncle = 7 := by
  sorry

end NUMINAMATH_CALUDE_grandparents_uncle_difference_l682_68217


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l682_68262

/-- The perimeter of an equilateral triangle with side length 5 cm is 15 cm. -/
theorem equilateral_triangle_perimeter :
  ∀ (side_length perimeter : ℝ),
  side_length = 5 →
  perimeter = 3 * side_length →
  perimeter = 15 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l682_68262


namespace NUMINAMATH_CALUDE_initial_stuffed_animals_l682_68299

def stuffed_animals (x : ℕ) : Prop :=
  let after_mom := x + 2
  let from_dad := 3 * after_mom
  x + 2 + from_dad = 48

theorem initial_stuffed_animals :
  ∃ (x : ℕ), stuffed_animals x ∧ x = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_stuffed_animals_l682_68299


namespace NUMINAMATH_CALUDE_calculation_proof_l682_68281

theorem calculation_proof :
  (6.42 - 2.8 + 3.58 = 7.2) ∧ (0.36 / (0.4 * (6.1 - 4.6)) = 0.6) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l682_68281


namespace NUMINAMATH_CALUDE_petya_friends_count_l682_68229

/-- The number of Petya's friends -/
def num_friends : ℕ := 19

/-- The number of stickers Petya has -/
def total_stickers : ℕ := num_friends * 5 + 8

/-- Theorem: Petya has 19 friends -/
theorem petya_friends_count : 
  (total_stickers = num_friends * 5 + 8) ∧ 
  (total_stickers = num_friends * 6 - 11) → 
  num_friends = 19 :=
by
  sorry


end NUMINAMATH_CALUDE_petya_friends_count_l682_68229


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l682_68297

-- Define propositions p and q
variable (p q : Prop)

-- Define the given conditions
variable (h1 : p → q)
variable (h2 : ¬(¬p → ¬q))

-- State the theorem
theorem p_sufficient_not_necessary :
  (∃ (r : Prop), r → q) ∧ ¬(q → p) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l682_68297


namespace NUMINAMATH_CALUDE_sphere_wedge_properties_l682_68257

/-- Represents a sphere cut into eight congruent wedges -/
structure SphereWedge where
  circumference : ℝ
  num_wedges : ℕ

/-- Calculates the volume of one wedge of the sphere -/
def wedge_volume (s : SphereWedge) : ℝ := sorry

/-- Calculates the surface area of one wedge of the sphere -/
def wedge_surface_area (s : SphereWedge) : ℝ := sorry

theorem sphere_wedge_properties (s : SphereWedge) 
  (h1 : s.circumference = 16 * Real.pi)
  (h2 : s.num_wedges = 8) : 
  wedge_volume s = (256 / 3) * Real.pi ∧ 
  wedge_surface_area s = 32 * Real.pi := by sorry

end NUMINAMATH_CALUDE_sphere_wedge_properties_l682_68257


namespace NUMINAMATH_CALUDE_inequality_solution_set_l682_68234

theorem inequality_solution_set (x : ℝ) : 
  -x^2 - 2*x + 3 > 0 ↔ -3 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l682_68234


namespace NUMINAMATH_CALUDE_bacteria_population_correct_l682_68243

def bacteria_population (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    2^(n/2 + 1)
  else
    2^((n+1)/2)

theorem bacteria_population_correct :
  ∀ n : ℕ,
  (bacteria_population n = 2^(n/2 + 1) ∧ n % 2 = 0) ∨
  (bacteria_population n = 2^((n+1)/2) ∧ n % 2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_bacteria_population_correct_l682_68243


namespace NUMINAMATH_CALUDE_janet_number_problem_l682_68228

theorem janet_number_problem (x : ℝ) : ((x - 3) * 3 + 3) / 3 = 10 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_janet_number_problem_l682_68228


namespace NUMINAMATH_CALUDE_problem_solution_l682_68273

theorem problem_solution (m n : ℝ) (hm : m^2 - 2*m = 1) (hn : n^2 - 2*n = 1) (hne : m ≠ n) :
  (m + n) - (m * n) = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l682_68273


namespace NUMINAMATH_CALUDE_smallest_digit_for_divisibility_by_nine_l682_68263

theorem smallest_digit_for_divisibility_by_nine :
  ∀ d : Nat, d ≤ 9 →
    (526000 + d * 1000 + 45) % 9 = 0 →
    d ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_digit_for_divisibility_by_nine_l682_68263


namespace NUMINAMATH_CALUDE_similar_triangles_perimeter_l682_68249

-- Define the two similar triangles
def Triangle1 : Type := Unit
def Triangle2 : Type := Unit

-- Define the height ratio
def height_ratio : ℚ := 2 / 3

-- Define the sum of perimeters
def total_perimeter : ℝ := 50

-- Define the perimeters of the two triangles
def perimeter1 : ℝ := 20
def perimeter2 : ℝ := 30

-- Theorem statement
theorem similar_triangles_perimeter :
  (perimeter1 / perimeter2 = height_ratio) ∧
  (perimeter1 + perimeter2 = total_perimeter) :=
sorry

end NUMINAMATH_CALUDE_similar_triangles_perimeter_l682_68249


namespace NUMINAMATH_CALUDE_unique_multiplier_l682_68261

def is_distinct (a b c d e f g h i j : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
  g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
  h ≠ i ∧ h ≠ j ∧
  i ≠ j

def is_valid_multiplier (a b c d : ℕ) : Prop :=
  a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d > 0 ∧ d < 10

def multiplier_to_num (a b c d : ℕ) : ℕ :=
  a * 1000 + b * 100 + c * 10 + d

def result_to_num (e f g h i j : ℕ) : ℕ :=
  e * 100000 + f * 10000 + g * 1000 + h * 100 + i * 10 + j

theorem unique_multiplier :
  ∃! (a b c d : ℕ),
    is_valid_multiplier a b c d ∧
    (∃ (e f g h i j : ℕ),
      is_distinct a b c d e f g h i j ∧
      multiplier_to_num a b 0 d * 1995 = result_to_num e f g h i j) ∧
    multiplier_to_num a b 0 d = 306 :=
sorry

end NUMINAMATH_CALUDE_unique_multiplier_l682_68261


namespace NUMINAMATH_CALUDE_square_with_removed_triangles_l682_68292

/-- Given a square with side length s, from which two pairs of identical isosceles right triangles
    are removed to form a rectangle, if the total area removed is 180 m², then the diagonal of the
    remaining rectangle is 18 m. -/
theorem square_with_removed_triangles (s : ℝ) (x y : ℝ) : 
  x ≥ 0 → y ≥ 0 → x + y = s → x^2 + y^2 = 180 → 
  Real.sqrt (2 * (x^2 + y^2)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_square_with_removed_triangles_l682_68292


namespace NUMINAMATH_CALUDE_sqrt3_expressions_l682_68237

theorem sqrt3_expressions (x y : ℝ) 
  (hx : x = Real.sqrt 3 + 1) 
  (hy : y = Real.sqrt 3 - 1) : 
  (x^2 + 2*x*y + y^2 = 12) ∧ (x^2 - y^2 = 4 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt3_expressions_l682_68237


namespace NUMINAMATH_CALUDE_smallest_valid_coloring_l682_68207

/-- A coloring function that assigns a color (represented by a natural number) to each integer in the range [2, 31] -/
def Coloring := Fin 30 → Nat

/-- Predicate to check if a coloring is valid according to the given conditions -/
def IsValidColoring (c : Coloring) : Prop :=
  ∀ m n, 2 ≤ m ∧ m ≤ 31 ∧ 2 ≤ n ∧ n ≤ 31 →
    m ≠ n → m % n = 0 → c (m - 2) ≠ c (n - 2)

/-- The existence of a valid coloring using k colors -/
def ExistsValidColoring (k : Nat) : Prop :=
  ∃ c : Coloring, IsValidColoring c ∧ ∀ i, c i < k

/-- The main theorem: The smallest number of colors needed is 4 -/
theorem smallest_valid_coloring : (∃ k, ExistsValidColoring k ∧ ∀ j, j < k → ¬ExistsValidColoring j) ∧
                                   ExistsValidColoring 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_coloring_l682_68207


namespace NUMINAMATH_CALUDE_unique_triple_lcm_l682_68255

theorem unique_triple_lcm : 
  ∃! (x y z : ℕ+), 
    Nat.lcm x y = 180 ∧ 
    Nat.lcm x z = 420 ∧ 
    Nat.lcm y z = 1260 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_lcm_l682_68255


namespace NUMINAMATH_CALUDE_opposite_of_2023_l682_68288

theorem opposite_of_2023 : 
  ∀ x : ℤ, x + 2023 = 0 ↔ x = -2023 := by
sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l682_68288


namespace NUMINAMATH_CALUDE_train_stop_time_l682_68294

/-- Proves that a train with given speeds stops for 20 minutes per hour -/
theorem train_stop_time (speed_without_stops speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 48)
  (h2 : speed_with_stops = 32) :
  (1 - speed_with_stops / speed_without_stops) * 60 = 20 := by
  sorry

#check train_stop_time

end NUMINAMATH_CALUDE_train_stop_time_l682_68294


namespace NUMINAMATH_CALUDE_herman_bird_feeding_l682_68287

/-- The number of days Herman feeds the birds -/
def feeding_days : ℕ := 90

/-- The amount of food Herman gives per feeding in cups -/
def food_per_feeding : ℚ := 1/2

/-- The number of feedings per day -/
def feedings_per_day : ℕ := 2

/-- Calculates the total amount of food needed for the feeding period -/
def total_food_needed (days : ℕ) (food_per_feeding : ℚ) (feedings_per_day : ℕ) : ℚ :=
  (days : ℚ) * food_per_feeding * (feedings_per_day : ℚ)

theorem herman_bird_feeding :
  total_food_needed feeding_days food_per_feeding feedings_per_day = 90 := by
  sorry

end NUMINAMATH_CALUDE_herman_bird_feeding_l682_68287


namespace NUMINAMATH_CALUDE_inequality_solution_range_l682_68212

/-- For the inequality x^2 + ax - 2 < 0 to have a solution in the interval [1, 4],
    the range of the real number a is (-∞, 1) -/
theorem inequality_solution_range (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 4, x^2 + a*x - 2 < 0) ↔ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l682_68212


namespace NUMINAMATH_CALUDE_sqrt_condition_implies_range_l682_68236

-- Define the condition for a meaningful square root
def meaningful_sqrt (x : ℝ) : Prop := 2 * x - 1 ≥ 0

-- Theorem statement
theorem sqrt_condition_implies_range (x : ℝ) : 
  meaningful_sqrt x → x ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_condition_implies_range_l682_68236


namespace NUMINAMATH_CALUDE_profit_without_discount_l682_68291

/-- Represents the profit percentage and discount percentage as rational numbers -/
def ProfitWithDiscount : ℚ := 44 / 100
def DiscountPercentage : ℚ := 4 / 100

/-- Theorem: If a shopkeeper earns a 44% profit after offering a 4% discount, 
    they would earn a 50% profit without the discount -/
theorem profit_without_discount 
  (cost_price : ℚ) 
  (selling_price : ℚ) 
  (marked_price : ℚ) 
  (h1 : selling_price = cost_price * (1 + ProfitWithDiscount))
  (h2 : selling_price = marked_price * (1 - DiscountPercentage))
  : (marked_price - cost_price) / cost_price = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_profit_without_discount_l682_68291


namespace NUMINAMATH_CALUDE_obtuse_angle_measure_l682_68233

/-- An obtuse angle divided by a perpendicular line into two angles with a ratio of 6:1 measures 105°. -/
theorem obtuse_angle_measure (θ : ℝ) (h1 : 90 < θ) (h2 : θ < 180) : 
  ∃ (α β : ℝ), α + β = θ ∧ α / β = 6 ∧ θ = 105 := by
  sorry

end NUMINAMATH_CALUDE_obtuse_angle_measure_l682_68233


namespace NUMINAMATH_CALUDE_distance_after_walk_l682_68252

/-- The distance from the starting point after walking 5 miles east, 
    turning 45 degrees north, and walking 7 miles. -/
theorem distance_after_walk (east_distance : ℝ) (angle : ℝ) (final_distance : ℝ) 
  (h1 : east_distance = 5)
  (h2 : angle = 45)
  (h3 : final_distance = 7) : 
  Real.sqrt (74 + 35 * Real.sqrt 2) = 
    Real.sqrt ((east_distance + final_distance * Real.sqrt 2 / 2) ^ 2 + 
               (final_distance * Real.sqrt 2 / 2) ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_distance_after_walk_l682_68252


namespace NUMINAMATH_CALUDE_davids_remaining_money_is_19_90_l682_68209

/-- Calculates David's remaining money after expenses and taxes -/
def davidsRemainingMoney (rate1 rate2 rate3 : ℝ) (hours : ℝ) (shoePrice : ℝ) 
  (shoeDiscount taxRate giftFraction : ℝ) : ℝ :=
  let totalEarnings := (rate1 + rate2 + rate3) * hours
  let taxAmount := totalEarnings * taxRate
  let discountedShoePrice := shoePrice * (1 - shoeDiscount)
  let remainingAfterShoes := totalEarnings - taxAmount - discountedShoePrice
  remainingAfterShoes * (1 - giftFraction)

/-- Theorem stating that David's remaining money is $19.90 -/
theorem davids_remaining_money_is_19_90 :
  davidsRemainingMoney 14 18 20 2 75 0.15 0.1 (1/3) = 19.90 := by
  sorry

end NUMINAMATH_CALUDE_davids_remaining_money_is_19_90_l682_68209


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l682_68248

theorem modulus_of_complex_number (z : ℂ) : z = 1 - 2*I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l682_68248


namespace NUMINAMATH_CALUDE_square_equation_solution_l682_68278

theorem square_equation_solution : ∃ x : ℝ, (72 - x)^2 = x^2 ∧ x = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_solution_l682_68278


namespace NUMINAMATH_CALUDE_sum_proper_divisors_243_l682_68285

theorem sum_proper_divisors_243 : 
  (Finset.filter (fun x => x ≠ 243 ∧ 243 % x = 0) (Finset.range 244)).sum id = 121 := by
  sorry

end NUMINAMATH_CALUDE_sum_proper_divisors_243_l682_68285


namespace NUMINAMATH_CALUDE_five_thursdays_in_august_l682_68253

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a date in a month -/
structure Date :=
  (day : Nat)
  (dayOfWeek : DayOfWeek)

/-- A month with its dates -/
structure Month :=
  (dates : List Date)
  (numDays : Nat)

def july : Month := sorry
def august : Month := sorry

/-- Counts the number of occurrences of a specific day in a month -/
def countDaysInMonth (m : Month) (d : DayOfWeek) : Nat := sorry

theorem five_thursdays_in_august 
  (h1 : july.numDays = 31)
  (h2 : august.numDays = 31)
  (h3 : countDaysInMonth july DayOfWeek.Tuesday = 5) :
  countDaysInMonth august DayOfWeek.Thursday = 5 := by sorry

end NUMINAMATH_CALUDE_five_thursdays_in_august_l682_68253


namespace NUMINAMATH_CALUDE_garden_length_l682_68241

/-- The length of a rectangular garden with perimeter 600 meters and breadth 200 meters is 100 meters. -/
theorem garden_length (perimeter : ℝ) (breadth : ℝ) (h1 : perimeter = 600) (h2 : breadth = 200) :
  2 * (breadth + (perimeter / 2 - breadth)) = perimeter ∧ perimeter / 2 - breadth = 100 := by
  sorry

end NUMINAMATH_CALUDE_garden_length_l682_68241


namespace NUMINAMATH_CALUDE_parallelogram_circle_theorem_l682_68230

/-- Represents a parallelogram KLMN with a circle tangent to NK and NM, passing through L, 
    and intersecting KL at C and ML at D. -/
structure ParallelogramWithCircle where
  -- The length of side KL
  kl : ℝ
  -- The ratio KC : LC
  kc_lc_ratio : ℝ × ℝ
  -- The ratio LD : MD
  ld_md_ratio : ℝ × ℝ

/-- Theorem stating that under the given conditions, KN = 10 -/
theorem parallelogram_circle_theorem (p : ParallelogramWithCircle) 
  (h1 : p.kl = 8)
  (h2 : p.kc_lc_ratio = (4, 5))
  (h3 : p.ld_md_ratio = (8, 1)) :
  ∃ (kn : ℝ), kn = 10 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_circle_theorem_l682_68230


namespace NUMINAMATH_CALUDE_sum_plus_even_count_l682_68224

def sum_of_range (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_in_range (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_plus_even_count : 
  let x := sum_of_range 50 60
  let y := count_even_in_range 50 60
  x + y = 611 := by sorry

end NUMINAMATH_CALUDE_sum_plus_even_count_l682_68224


namespace NUMINAMATH_CALUDE_marbles_collection_sum_l682_68271

def total_marbles (adam mary greg john sarah : ℕ) : ℕ :=
  adam + mary + greg + john + sarah

theorem marbles_collection_sum :
  ∀ (adam mary greg john sarah : ℕ),
    adam = 29 →
    mary = adam - 11 →
    greg = adam + 14 →
    john = 2 * mary →
    sarah = greg - 7 →
    total_marbles adam mary greg john sarah = 162 :=
by
  sorry

end NUMINAMATH_CALUDE_marbles_collection_sum_l682_68271


namespace NUMINAMATH_CALUDE_committee_selection_l682_68235

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem committee_selection :
  choose 18 3 = 816 := by sorry

end NUMINAMATH_CALUDE_committee_selection_l682_68235


namespace NUMINAMATH_CALUDE_final_quantity_of_B_l682_68293

/-- Represents the quantity of each product type -/
structure ProductQuantities where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total cost of the products -/
def totalCost (q : ProductQuantities) : ℕ :=
  2 * q.a + 3 * q.b + 5 * q.c

/-- Represents the problem constraints -/
structure ProblemConstraints where
  initial : ProductQuantities
  final : ProductQuantities
  initialCost : totalCost initial = 20
  finalCost : totalCost final = 20
  returnedTwoItems : initial.a + initial.b + initial.c = final.a + final.b + final.c + 2
  atLeastOne : final.a ≥ 1 ∧ final.b ≥ 1 ∧ final.c ≥ 1

theorem final_quantity_of_B (constraints : ProblemConstraints) : constraints.final.b = 1 := by
  sorry


end NUMINAMATH_CALUDE_final_quantity_of_B_l682_68293


namespace NUMINAMATH_CALUDE_inequality_preservation_l682_68201

theorem inequality_preservation (a b : ℝ) (h : a > b) : a - 5 > b - 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l682_68201


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_9_and_6_l682_68284

theorem smallest_common_multiple_of_9_and_6 :
  ∃ n : ℕ, n > 0 ∧ 9 ∣ n ∧ 6 ∣ n ∧ ∀ m : ℕ, m > 0 → 9 ∣ m → 6 ∣ m → n ≤ m :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_9_and_6_l682_68284


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l682_68200

/-- An arithmetic sequence with common difference -2 and S_3 = 21 reaches its maximum sum at n = 5 -/
theorem arithmetic_sequence_max_sum (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (∀ n, a (n + 1) - a n = -2) →  -- Common difference is -2
  S 3 = 21 →                     -- S_3 = 21
  (∀ n, S n = (n * (a 1 + a n)) / 2) →  -- Definition of S_n for arithmetic sequence
  (∃ m, ∀ k, S k ≤ S m) →       -- S_n has a maximum value
  (∀ k, S k ≤ S 5) :=            -- The maximum occurs at n = 5
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l682_68200


namespace NUMINAMATH_CALUDE_binomial_expansion_equality_l682_68204

theorem binomial_expansion_equality (x : ℝ) : 
  (x - 1)^4 - 4*x*(x - 1)^3 + 6*x^2*(x - 1)^2 - 4*x^3*(x - 1) * x^4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_equality_l682_68204


namespace NUMINAMATH_CALUDE_matrix_determinant_l682_68290

def matrix : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, -3, 3],
    ![0,  5, -1],
    ![4, -2, 1]]

theorem matrix_determinant :
  Matrix.det matrix = -45 := by sorry

end NUMINAMATH_CALUDE_matrix_determinant_l682_68290


namespace NUMINAMATH_CALUDE_voting_stabilizes_l682_68219

/-- Represents the state of votes in a circular arrangement -/
def VoteState := Vector Bool 25

/-- Represents the next state of votes based on the current state -/
def nextState (current : VoteState) : VoteState :=
  Vector.ofFn (fun i =>
    let prev := current.get ((i - 1 + 25) % 25)
    let next := current.get ((i + 1) % 25)
    let curr := current.get i
    if prev = next then curr else !curr)

/-- Theorem stating that the voting pattern will eventually stabilize -/
theorem voting_stabilizes : ∃ (n : ℕ), ∀ (initial : VoteState),
  ∃ (k : ℕ), k ≤ n ∧ nextState^[k] initial = nextState^[k+1] initial :=
sorry


end NUMINAMATH_CALUDE_voting_stabilizes_l682_68219


namespace NUMINAMATH_CALUDE_marathon_duration_in_minutes_l682_68202

-- Define the duration of the marathon
def marathon_hours : ℕ := 15
def marathon_minutes : ℕ := 35

-- Theorem to prove
theorem marathon_duration_in_minutes :
  marathon_hours * 60 + marathon_minutes = 935 := by
  sorry

end NUMINAMATH_CALUDE_marathon_duration_in_minutes_l682_68202


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l682_68282

theorem smaller_number_in_ratio (x y : ℝ) : 
  x > 0 → y > 0 → x / y = 2 / 5 → x + y = 21 → min x y = 6 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l682_68282


namespace NUMINAMATH_CALUDE_boxes_with_neither_l682_68244

theorem boxes_with_neither (total : ℕ) (markers : ℕ) (crayons : ℕ) (both : ℕ)
  (h1 : total = 15)
  (h2 : markers = 8)
  (h3 : crayons = 4)
  (h4 : both = 3) :
  total - (markers + crayons - both) = 6 := by
sorry

end NUMINAMATH_CALUDE_boxes_with_neither_l682_68244


namespace NUMINAMATH_CALUDE_equation_solution_l682_68222

theorem equation_solution :
  let f (x : ℝ) := 4 * (3 * x)^2 + (3 * x) + 5 - (3 * (9 * x^2 + 3 * x + 3))
  ∀ x : ℝ, f x = 0 ↔ x = (1 + Real.sqrt 5) / 3 ∨ x = (1 - Real.sqrt 5) / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l682_68222


namespace NUMINAMATH_CALUDE_min_value_of_expression_l682_68226

theorem min_value_of_expression :
  (∀ x : ℝ, (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2023 ≥ 2022) ∧
  (∃ x : ℝ, (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2023 = 2022) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l682_68226


namespace NUMINAMATH_CALUDE_sum_of_fractions_to_decimal_l682_68232

theorem sum_of_fractions_to_decimal : (5 : ℚ) / 16 + (1 : ℚ) / 4 = (5625 : ℚ) / 10000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_to_decimal_l682_68232


namespace NUMINAMATH_CALUDE_solve_equation_l682_68279

theorem solve_equation : ∃ x : ℝ, 3 * x = (36 - x) + 16 ∧ x = 13 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l682_68279


namespace NUMINAMATH_CALUDE_age_ratio_proof_l682_68276

/-- Prove that given the conditions, the ratio of B's age to C's age is 2:1 -/
theorem age_ratio_proof (A B C : ℕ) : 
  A = B + 2 →  -- A is two years older than B
  A + B + C = 37 →  -- The total of the ages of A, B, and C is 37
  B = 14 →  -- B is 14 years old
  B / C = 2  -- The ratio of B's age to C's age is 2:1
  :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l682_68276
