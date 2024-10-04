import Mathlib
import Mathlib.Algebra.ArithmeticSequence
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Order
import Mathlib.Algebra.LinearAlgebra.Basic
import Mathlib.Algebra.Order
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.Trigonometry.SinCos
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.StarsAndBars
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Zmod.Basic
import Mathlib.GraphTheory.Connectivity.Basic
import Mathlib.Logic.Basic
import Mathlib.MeasureTheory.Integral
import Mathlib.NumberTheory.Primes
import Mathlib.Probability.Basic
import Mathlib.Probability.RandomVariable
import Mathlib.ProbabilityTheory.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic

namespace sin_5x_over_sin_x_l159_159010

theorem sin_5x_over_sin_x (x : ℝ) (h : sin (3 * x) / sin x = 6 / 5) : sin (5 * x) / sin x = -0.76 := 
by
  sorry

end sin_5x_over_sin_x_l159_159010


namespace inverse_function_value_l159_159914

def f (x : ℝ) : ℝ := 2^x - 1

theorem inverse_function_value :
  f⁻¹(3) = 2 :=
by
  sorry

end inverse_function_value_l159_159914


namespace proof_of_problem_l159_159484

open Real

noncomputable def problem_statement : Prop :=
  ∀ (x y : ℝ), 
    (4^x / 2^(x + y) = 8) →
    (9^(x + y) / 3^(5 * y) = 243) →
    x * y = 4

theorem proof_of_problem : problem_statement :=
by
  -- Proof to be filled in.
  sorry

end proof_of_problem_l159_159484


namespace product_xyz_l159_159982

theorem product_xyz (x y z : ℝ) (h1 : x + 1 / y = 2) (h2 : y + 1 / z = 2) (h3 : x + 1 / z = 3) : x * y * z = 2 := 
by sorry

end product_xyz_l159_159982


namespace cleaning_time_together_l159_159324

-- Definitions based on the conditions provided
def tom_cleaning_time : ℝ := 6 -- hours to clean the entire house
def tom_half_cleaning_time : ℝ := tom_cleaning_time / 2 -- time Tom takes to clean half the house
def nick_cleaning_time := 3 * tom_half_cleaning_time -- time Nick takes to clean the entire house

-- Rates of cleaning (house per hour)
def tom_cleaning_rate : ℝ := 1 / tom_cleaning_time
def nick_cleaning_rate : ℝ := 1 / nick_cleaning_time

-- Combined cleaning rate
def combined_cleaning_rate : ℝ := tom_cleaning_rate + nick_cleaning_rate

-- Time for Nick and Tom to clean the house together
def combined_cleaning_time : ℝ := 1 / combined_cleaning_rate

-- Proof statement
theorem cleaning_time_together : combined_cleaning_time = 3.6 :=
by
  -- Proof omitted
  sorry

end cleaning_time_together_l159_159324


namespace fruits_eaten_total_l159_159702

variable (apples blueberries bonnies : ℕ)

noncomputable def total_fruits_eaten : ℕ :=
  let third_dog_bonnies := 60
  let second_dog_blueberries := 3 / 4 * third_dog_bonnies
  let first_dog_apples := 3 * second_dog_blueberries
  first_dog_apples + second_dog_blueberries + third_dog_bonnies

theorem fruits_eaten_total:
  let third_dog_bonnies := 60
  let second_dog_blueberries := 3 * third_dog_bonnies / 4
  let first_dog_apples := 3 * second_dog_blueberries
  first_dog_apples + second_dog_blueberries + third_dog_bonnies = 240 := by
  sorry

end fruits_eaten_total_l159_159702


namespace find_remainder_l159_159556

-- Main statement with necessary definitions and conditions
theorem find_remainder (x : ℤ) (h : (x + 11) % 31 = 18) :
  x % 62 = 7 :=
sorry

end find_remainder_l159_159556


namespace parallel_transitivity_l159_159333

theorem parallel_transitivity {a b c : Type} [has_parallel a b] [has_parallel b c] : has_parallel a c :=
begin
  sorry
end

end parallel_transitivity_l159_159333


namespace vasya_birthday_l159_159707

def DayOfWeek := 
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

open DayOfWeek

def day_after (d : DayOfWeek) : DayOfWeek :=
match d with
| Sunday => Monday
| Monday => Tuesday
| Tuesday => Wednesday
| Wednesday => Thursday
| Thursday => Friday
| Friday => Saturday
| Saturday => Sunday

def day_before (d : DayOfWeek) : DayOfWeek :=
match d with
| Sunday => Saturday
| Monday => Sunday
| Tuesday => Monday
| Wednesday => Tuesday
| Thursday => Wednesday
| Friday => Thursday
| Saturday => Friday

theorem vasya_birthday (day_said : DayOfWeek) 
    (h1 : day_after (day_after day_said) = Sunday) 
    (h2 : day_said = day_after VasyaBirthday) 
    : VasyaBirthday = Thursday :=
sorry

end vasya_birthday_l159_159707


namespace hyperbola_eccentricity_l159_159938

-- Define hyperbola with one asymptote y = x
def is_asymptote (h : Hyperbola) (l : Line) : Prop :=
  l = Line.mk 1 0 0 1 -- Represents y - x = 0 which simplifies to y = x

-- Define equilateral hyperbola
def is_equilateral (h : Hyperbola) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ h = Hyperbola.mk a b

-- Eccentricity of an equilateral hyperbola
def eccentricity (h : Hyperbola) : ℝ :=
  if is_equilateral h then sqrt 2 else 0

theorem hyperbola_eccentricity (h : Hyperbola) (l : Line) :
  is_asymptote h l → eccentricity h = sqrt 2 :=
by
  sorry

end hyperbola_eccentricity_l159_159938


namespace sofiya_wins_l159_159658

/-- Define the initial configuration and game rules -/
def initial_configuration : Type := { n : Nat // n = 2025 }

/--
  Define the game such that Sofiya starts and follows the strategy of always
  removing a neighbor from the arc with an even number of people.
-/
def winning_strategy (n : initial_configuration) : Prop :=
  n.1 % 2 = 1 ∧ 
  (∀ turn : Nat, turn % 2 = 0 → 
    (∃ arc : initial_configuration, arc.1 % 2 = 0 ∧ arc.1 < n.1) ∧
    (∀ marquis_turn : Nat, marquis_turn % 2 = 1 → 
      (∃ arc : initial_configuration, arc.1 % 2 = 1)))

/-- Sofiya has the winning strategy given the conditions of the game -/
theorem sofiya_wins : winning_strategy ⟨2025, rfl⟩ :=
sorry

end sofiya_wins_l159_159658


namespace find_n_l159_159275

open Nat

theorem find_n (d : ℕ → ℕ) (n : ℕ) (h1 : ∀ j, d (j + 1) > d j) (h2 : n = d 13 + d 14 + d 15) (h3 : (d 5 + 1)^3 = d 15 + 1) : 
  n = 1998 :=
by
  sorry

end find_n_l159_159275


namespace total_feet_l159_159373

theorem total_feet (heads hens : ℕ) (h1 : heads = 46) (h2 : hens = 22) : 
  ∃ feet : ℕ, feet = 140 := 
by 
  sorry

end total_feet_l159_159373


namespace vovochka_no_carry_correct_cases_vovochka_minimum_difference_l159_159125

-- Part (a)
theorem vovochka_no_carry_correct_cases :
  let digit_pairs_without_carry := ∑ k in finset.range 10, (k + 1)
  let three_digit_pairs_without_carry := 9 * digit_pairs_without_carry
  let total_valid_combinations := 81 * three_digit_pairs_without_carry * three_digit_pairs_without_carry
  total_valid_combinations = 244620 :=
by {
  -- Definitions used in conditions
  let digit_pairs_without_carry := ∑ k in finset.range 10, (k + 1)
  let three_digit_pairs_without_carry := 9 * digit_pairs_without_carry
  let total_valid_combinations := 81 * three_digit_pairs_without_carry * three_digit_pairs_without_carry

  -- Assert the correct answer
  have correct_total_cases : total_valid_combinations = 244620 := 
    -- solution provided proof here
    sorry,

  exact correct_total_cases
}

-- Part (b)
theorem vovochka_minimum_difference :
  let smallest_difference := 1800
  ∃ a b c x y z : ℕ, (a, b, c) < (x, y, z) ∧ a + x ≥ 1 ∧ b + y ≥ 1 ∧ c + z ≥ 1 ∧
  correct_sum a b c x y z - vovochka_sum a b c x y z = smallest_difference :=
by {
  -- Definitions used in conditions
  let correct_sum := λ (a b c x y z : ℕ), 100 * (a + x) + 10 * (b + y) + (c + z)
  let vovochka_sum := λ (a b c x y z : ℕ), 100 * (a + x) + 10 * (b + y) + (c + z) % 10

  -- Assert the correct answer
  have smallest_diff_exists : 
    ∃ a b c x y z : ℕ, (a, b, c) < (x, y, z) ∧ a + x ≥ 1 ∧ b + y ≥ 1 ∧ c + z ≥ 1 ∧
    correct_sum a b c x y z - vovochka_sum a b c x y z = 1800 := 
    -- solution provided proof here
    sorry,

  exact smallest_diff_exists
}

end vovochka_no_carry_correct_cases_vovochka_minimum_difference_l159_159125


namespace a_7_is_127_l159_159475

-- Define the sequence {a_n}
def a : ℕ → ℕ
| 0       => 0  -- Define a_0 which is not used but useful for indexing
| 1       => 1
| (n + 2) => 2 * (a (n + 1)) + 1

-- Prove that a_7 = 127
theorem a_7_is_127 : a 7 = 127 := 
sorry

end a_7_is_127_l159_159475


namespace solve_x_l159_159250

theorem solve_x (x : ℝ) (h : x^2 + 6 * x + 8 = -(x + 4) * (x + 6)) : 
  x = -4 := 
by
  sorry

end solve_x_l159_159250


namespace complex_magnitude_l159_159561

theorem complex_magnitude (z : ℂ) (h_abs_z_lt_1 : |z| < 1) 
    (h_condition : |conj(z) + 1 / z| = 5 / 2) : |z| = 1 / 2 :=
sorry

end complex_magnitude_l159_159561


namespace determine_a_l159_159915

-- Define the function f as given in the conditions
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 6

-- Formulate the proof statement
theorem determine_a (a : ℝ) (h : f a (-1) = 8) : a = -2 :=
by {
  sorry
}

end determine_a_l159_159915


namespace range_of_a_l159_159675

-- Define the function f
def f (x a : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

-- Define the condition for f to be decreasing on the interval (-∞, 4]
def is_decreasing_on (f : ℝ → ℝ) (a : ℝ) : Prop :=
 ∀ x ∈ (set.Iic 4 : set ℝ), (deriv (λ x, f x a)) x ≤ 0

-- Given statement to prove
theorem range_of_a (a : ℝ) (h : is_decreasing_on f a) : a ≤ -3 :=
 sorry

end range_of_a_l159_159675


namespace number_of_parakeets_per_cage_l159_159376

def num_cages : ℕ := 9
def parrots_per_cage : ℕ := 2
def total_birds : ℕ := 72

theorem number_of_parakeets_per_cage : (total_birds - (num_cages * parrots_per_cage)) / num_cages = 6 := by
  sorry

end number_of_parakeets_per_cage_l159_159376


namespace vowel_probability_is_seven_twenty_four_l159_159642

-- Define the set of students and their initials
def students_set := {i : Fin 26 | i ≠ 23 ∧ i ≠ 25 } -- Excluding XX (index 23) and ZZ (index 25)

-- Define the set of vowels including Y and W
def vowels_set := {0, 4, 8, 14, 20, 24, 22} -- Indices for A, E, I, O, U, Y, W

-- Define the set of student initials that are vowels
def vowel_initials := students_set ∩ vowels_set

-- Calculate the probability
def vowel_probability : ℚ :=
  (vowel_initials.card : ℚ) / (students_set.card : ℚ)

-- Statement of the proof problem
theorem vowel_probability_is_seven_twenty_four :
  vowel_probability = 7 / 24 :=
by
  sorry

end vowel_probability_is_seven_twenty_four_l159_159642


namespace side_length_of_S2_l159_159239

theorem side_length_of_S2 :
  ∀ (r s : ℕ), 
    (2 * r + s = 2000) → 
    (2 * r + 5 * s = 3030) → 
    s = 258 :=
by
  intros r s h1 h2
  sorry

end side_length_of_S2_l159_159239


namespace probability_two_red_two_blue_correct_l159_159344

noncomputable def num_ways_to_choose : ℕ → ℕ → ℕ :=
  λ n k, Nat.choose n k

noncomputable def probability_two_red_two_blue : ℚ :=
  let total_ways := num_ways_to_choose 20 4
  let ways_red := num_ways_to_choose 12 2
  let ways_blue := num_ways_to_choose 8 2
  (ways_red * ways_blue) / total_ways

theorem probability_two_red_two_blue_correct :
  probability_two_red_two_blue = 616 / 1615 :=
by
  sorry

end probability_two_red_two_blue_correct_l159_159344


namespace percent_equality_l159_159785

theorem percent_equality :
  (1 / 4 : ℝ) * 100 = (10 / 100 : ℝ) * 250 :=
by
  sorry

end percent_equality_l159_159785


namespace range_of_f_l159_159279

def f (x : ℝ) : ℝ := |1 - x| - |x - 3|

theorem range_of_f : set.range f = set.Icc (-2) 2 :=
by sorry

end range_of_f_l159_159279


namespace largest_prime_factor_of_3434_l159_159297

theorem largest_prime_factor_of_3434 : ∃ p, prime p ∧ p ∣ 3434 ∧ ∀ q, prime q ∧ q ∣ 3434 → q ≤ p :=
sorry

end largest_prime_factor_of_3434_l159_159297


namespace roots_polynomial_sum_l159_159194

theorem roots_polynomial_sum (x1 x2 x3 : ℝ) (m n : ℕ) (hmn : Nat.coprime m n)
  (hroot : ∀ (x : ℝ), (x = x1 ∨ x = x2 ∨ x = x3) -> x^3 + 3*x + 1 = 0)
  (hvalue : (m : ℝ)/n = (x1^2)/((5*x2 + 1)*(5*x3 + 1)) + (x2^2)/((5*x1 + 1)*(5*x3 + 1)) + (x3^2)/((5*x1 + 1)*(5*x2 + 1))) :
  m + n = 10 := 
sorry

end roots_polynomial_sum_l159_159194


namespace staircase_steps_l159_159530

theorem staircase_steps (x : ℕ) :
  x % 2 = 1 ∧
  x % 3 = 2 ∧
  x % 4 = 3 ∧
  x % 5 = 4 ∧
  x % 6 = 5 ∧
  x % 7 = 0 → 
  x ≡ 119 [MOD 420] :=
by
  sorry

end staircase_steps_l159_159530


namespace complex_modulus_power_l159_159888

theorem complex_modulus_power :
  complex.abs ((2 : ℂ) + (complex.I * real.sqrt 11))^4 = 225 :=
by
  sorry

end complex_modulus_power_l159_159888


namespace vovochka_no_carry_correct_cases_vovochka_minimum_difference_l159_159128

-- Part (a)
theorem vovochka_no_carry_correct_cases :
  let digit_pairs_without_carry := ∑ k in finset.range 10, (k + 1)
  let three_digit_pairs_without_carry := 9 * digit_pairs_without_carry
  let total_valid_combinations := 81 * three_digit_pairs_without_carry * three_digit_pairs_without_carry
  total_valid_combinations = 244620 :=
by {
  -- Definitions used in conditions
  let digit_pairs_without_carry := ∑ k in finset.range 10, (k + 1)
  let three_digit_pairs_without_carry := 9 * digit_pairs_without_carry
  let total_valid_combinations := 81 * three_digit_pairs_without_carry * three_digit_pairs_without_carry

  -- Assert the correct answer
  have correct_total_cases : total_valid_combinations = 244620 := 
    -- solution provided proof here
    sorry,

  exact correct_total_cases
}

-- Part (b)
theorem vovochka_minimum_difference :
  let smallest_difference := 1800
  ∃ a b c x y z : ℕ, (a, b, c) < (x, y, z) ∧ a + x ≥ 1 ∧ b + y ≥ 1 ∧ c + z ≥ 1 ∧
  correct_sum a b c x y z - vovochka_sum a b c x y z = smallest_difference :=
by {
  -- Definitions used in conditions
  let correct_sum := λ (a b c x y z : ℕ), 100 * (a + x) + 10 * (b + y) + (c + z)
  let vovochka_sum := λ (a b c x y z : ℕ), 100 * (a + x) + 10 * (b + y) + (c + z) % 10

  -- Assert the correct answer
  have smallest_diff_exists : 
    ∃ a b c x y z : ℕ, (a, b, c) < (x, y, z) ∧ a + x ≥ 1 ∧ b + y ≥ 1 ∧ c + z ≥ 1 ∧
    correct_sum a b c x y z - vovochka_sum a b c x y z = 1800 := 
    -- solution provided proof here
    sorry,

  exact smallest_diff_exists
}

end vovochka_no_carry_correct_cases_vovochka_minimum_difference_l159_159128


namespace total_feet_l159_159374

theorem total_feet (H C : ℕ) (h1 : H + C = 48) (h2 : H = 28) : 2 * H + 4 * C = 136 := 
by
  sorry

end total_feet_l159_159374


namespace num_gcd_values_l159_159775

-- Define the condition for the product of gcd and lcm
def is_valid_pair (a b : ℕ) : Prop :=
  gcd a b * Nat.lcm a b = 360

-- Define the main theorem statement
theorem num_gcd_values : 
  ∃ (n : ℕ), 
    (∀ a b, is_valid_pair a b → ∃ m (hm: m ≤ 360), gcd a b = m) ∧ 
    n = 12 := sorry

end num_gcd_values_l159_159775


namespace minimum_value_expression_l159_159014

noncomputable def minimum_expression (n : ℕ) (x : Fin n → ℝ) : ℝ :=
  let sin_sum := (Finset.univ.sum (λ i, Real.sin (x i))) in
  let cos_sum := (Finset.univ.sum (λ i, Real.cos (x i))) in
  (2 * sin_sum + cos_sum) * (sin_sum - 2 * cos_sum)

theorem minimum_value_expression (n : ℕ) (x : Fin n → ℝ) :
  minimum_expression n x ≥ -5 * n^2 / 2 := by
  sorry

end minimum_value_expression_l159_159014


namespace minimum_value_of_expression_l159_159068

theorem minimum_value_of_expression (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) :
  (⟦(a^2 + b^2) / (c * d)⟧^4 + ⟦(b^2 + c^2) / (a * d)⟧^4 + ⟦(c^2 + d^2) / (a * b)⟧^4 + ⟦(d^2 + a^2) / (b * c)⟧^4) = 64 :=
sorry

end minimum_value_of_expression_l159_159068


namespace vasya_birthday_is_thursday_l159_159731

def vasya_birthday_day_of_week (today_is_friday : Bool) (sunday_day_after_tomorrow : Bool) : String :=
  if today_is_friday && sunday_day_after_tomorrow then "Thursday" else "Unknown"

theorem vasya_birthday_is_thursday
  (today_is_friday : true)
  (sunday_day_after_tomorrow : true) : 
  vasya_birthday_day_of_week true true = "Thursday" := 
by
  -- assume today is Friday
  have h1 : today_is_friday = true := rfl
  -- assume Sunday is the day after tomorrow
  have h2 : sunday_day_after_tomorrow = true := rfl
  -- by our function definition, Vasya's birthday should be Thursday
  show vasya_birthday_day_of_week true true = "Thursday"
  sorry

end vasya_birthday_is_thursday_l159_159731


namespace vasya_birthday_l159_159709

def DayOfWeek := 
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

open DayOfWeek

def day_after (d : DayOfWeek) : DayOfWeek :=
match d with
| Sunday => Monday
| Monday => Tuesday
| Tuesday => Wednesday
| Wednesday => Thursday
| Thursday => Friday
| Friday => Saturday
| Saturday => Sunday

def day_before (d : DayOfWeek) : DayOfWeek :=
match d with
| Sunday => Saturday
| Monday => Sunday
| Tuesday => Monday
| Wednesday => Tuesday
| Thursday => Wednesday
| Friday => Thursday
| Saturday => Friday

theorem vasya_birthday (day_said : DayOfWeek) 
    (h1 : day_after (day_after day_said) = Sunday) 
    (h2 : day_said = day_after VasyaBirthday) 
    : VasyaBirthday = Thursday :=
sorry

end vasya_birthday_l159_159709


namespace no_integers_satisfy_equation_l159_159232

theorem no_integers_satisfy_equation :
  ∀ (a b c : ℤ), a^2 + b^2 - 8 * c ≠ 6 := by
  sorry

end no_integers_satisfy_equation_l159_159232


namespace Vasya_birthday_on_Thursday_l159_159720

-- Define the days of the week enumeration
inductive DayOfWeek
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open DayOfWeek

-- Define the conditions
def Sunday_is_the_day_after_tomorrow (today : DayOfWeek) : Prop :=
  match today with
  | Friday => true
  | _      => false

def Vasya_birthday_not_on_Sunday (birthday : DayOfWeek) : Prop :=
  birthday ≠ Sunday

def Today_is_next_day_after_birthday (today birthday : DayOfWeek) : Prop :=
  match (today, birthday) with
  | (Friday, Thursday) => true
  | _                  => false

-- The main theorem to prove Vasya's birthday was on Thursday
theorem Vasya_birthday_on_Thursday (birthday today : DayOfWeek) 
  (H1: Sunday_is_the_day_after_tomorrow today) 
  (H2: Vasya_birthday_not_on_Sunday birthday) 
  (H3: Today_is_next_day_after_birthday today birthday) : 
  birthday = Thursday :=
by
  sorry

end Vasya_birthday_on_Thursday_l159_159720


namespace find_x_from_percentage_l159_159337

theorem find_x_from_percentage : 
  ∃ x : ℚ, 0.65 * x = 0.20 * 487.50 := 
sorry

end find_x_from_percentage_l159_159337


namespace max_n_value_l159_159917

theorem max_n_value (a b c : ℝ) (n : ℕ) (h1 : a > b) (h2 : b > c) (h_ineq : 1/(a - b) + 1/(b - c) ≥ n/(a - c)) : n ≤ 4 := 
sorry

end max_n_value_l159_159917


namespace distance_from_point_to_focus_l159_159474

theorem distance_from_point_to_focus :
  ∀ y, (2, y) ∈ { p : ℝ × ℝ | p.2 ^ 2 = 8 * p.1 } →
  (let focus := (2, 0)
   in (let dist := λ (a b : ℝ × ℝ), Real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2)
       in dist (2, y) focus)) = 4 := 
by 
  intro y hy
  let focus := (2, 0)
  let dist := λ (a b : ℝ × ℝ), Real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2)
  have h_focus : focus = (2, 0) := by reflexivity
  sorry

end distance_from_point_to_focus_l159_159474


namespace line_equation_l159_159371

theorem line_equation
  (t : ℝ)
  (x : ℝ) (y : ℝ)
  (h1 : x = 3 * t + 6)
  (h2 : y = 5 * t - 10) :
  y = (5 / 3) * x - 20 :=
sorry

end line_equation_l159_159371


namespace boxes_in_carton_l159_159813

theorem boxes_in_carton (cost_per_pack : ℕ) (packs_per_box : ℕ) (cost_dozen_cartons : ℕ) 
  (h1 : cost_per_pack = 1) (h2 : packs_per_box = 10) (h3 : cost_dozen_cartons = 1440) :
  (cost_dozen_cartons / 12) / (cost_per_pack * packs_per_box) = 12 :=
by
  sorry

end boxes_in_carton_l159_159813


namespace range_f_l159_159026

variable {α : Type} [LinearOrder α] [Field α]

noncomputable def f (x : α) : α := (3 * x + 8) / (x - 4)

theorem range_f : set.range f = { y : α | y ≠ 3 } :=
by
  sorry  -- Proof is omitted, according to the instructions

end range_f_l159_159026


namespace repeat_six_as_common_fraction_l159_159884

noncomputable def repeat_six_as_fraction (x : Real) : Prop :=
  (x = Real.ofRat (2/3)) → (x = 0.666666...)

theorem repeat_six_as_common_fraction : ∃ (x : Real), repeat_six_as_fraction x :=
begin
  use 0.666666..., 
  unfold repeat_six_as_fraction,
  sorry
end

end repeat_six_as_common_fraction_l159_159884


namespace find_a_l159_159951

noncomputable def f (x a : ℝ) : ℝ := -x^2 + 2 * a * x + a - 1

theorem find_a (a : ℝ) (h1 : ∀ x ∈ Icc 0 1, f x a ≤ 1) (h2 : ∃ x ∈ Icc 0 1, f x a = 1) : a = 1 := 
sorry

end find_a_l159_159951


namespace friends_with_Ron_l159_159242

-- Ron is eating pizza with his friends 
def total_slices : Nat := 12
def slices_per_person : Nat := 4
def total_people := total_slices / slices_per_person
def ron_included := 1

theorem friends_with_Ron : total_people - ron_included = 2 := by
  sorry

end friends_with_Ron_l159_159242


namespace identify_false_coins_l159_159701

theorem identify_false_coins :
  ∃ (A₁ A₂ : ℕ), 
    -- The set {1, 3, 4, 5, 6}
    (A₁ = 0 ∨ A₁ = 1 ∨ A₁ = 2) ∧ 
    -- The set {2, 3, 4, 9, 10}
    (A₂ = 0 ∨ A₂ = 1 ∨ A₂ = 2) ∧
    -- Mapping from answers to false coin pairs
    ((A₁ = 0 ∧ A₂ = 0 → false_coins = {7, 8}) ∧
     (A₁ = 0 ∧ A₂ = 1 → false_coins = {8, 9}) ∧
     (A₁ = 0 ∧ A₂ = 2 → false_coins = {9, 10}) ∧
     (A₁ = 1 ∧ A₂ = 0 → false_coins = {6, 7}) ∧
     (A₁ = 1 ∧ A₂ = 1 → false_coins = {1, 2}) ∧
     (A₁ = 1 ∧ A₂ = 2 → false_coins = {2, 3}) ∧
     (A₁ = 2 ∧ A₂ = 0 → false_coins = {5, 6}) ∧
     (A₁ = 2 ∧ A₂ = 1 → false_coins = {4, 5}) ∧
     (A₁ = 2 ∧ A₂ = 2 → false_coins = {3, 4})) :=
sorry

end identify_false_coins_l159_159701


namespace club_positions_l159_159365

def num_ways_to_fill_positions (n : ℕ) : ℕ := n * (n - 1) * (n - 2) * (n - 3) * (n - 4) * (n - 5)

theorem club_positions : num_ways_to_fill_positions 12 = 665280 := by 
  sorry

end club_positions_l159_159365


namespace different_gcd_values_count_l159_159771

theorem different_gcd_values_count :
  let gcd_lcm_eq_prod (a b : ℕ) := Nat.gcd a b * Nat.lcm a b = a * b
  let prime_factors_360 := (2 ^ 3 * 3 ^ 2 * 5 ^ 1 : ℕ)
  (∃ a b : ℕ, gcd_lcm_eq_prod a b ∧ a * b = 360) →
  (∃ gcd_vals : Finset ℕ, gcd_vals = {1, 2, 3, 4, 6, 8, 12, 24} ∧ gcd_vals.card = 8) :=
begin
  sorry
end

end different_gcd_values_count_l159_159771


namespace sum_bi_abs_l159_159045

def R (x : ℝ) : ℝ := 1 - (1 / 2) * x + (1 / 4) * x^2 - (1 / 8) * x^3
def S (x : ℝ) : ℝ := R(x) * R(x^2) * R(x^4) * R(x^6) * R(x^8)
def b_i_sum_abs : ℝ := ∑ i in Finset.range 31, abs ((S 1) i)

theorem sum_bi_abs : b_i_sum_abs = (6875 / 32768) := by
  sorry

end sum_bi_abs_l159_159045


namespace star_vs_emilio_l159_159258

def star_numbers := list.range' 1 50

def replace_2_3_with_1 (n: ℕ): ℕ :=
  let s := n.digits 10
  s.foldr (λ d acc, acc * 10 + if d = 2 ∨ d = 3 then 1 else d) 0

def star_sum := star_numbers.sum

def emilio_sum := (star_numbers.map replace_2_3_with_1).sum

theorem star_vs_emilio:
  star_sum = emilio_sum + 310 :=
by
  -- Proof is omitted as it is not required in the exercise.
  sorry

end star_vs_emilio_l159_159258


namespace probability_two_red_two_blue_l159_159354

def total_marbles : ℕ := 20
def red_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def selected_marbles : ℕ := 4

theorem probability_two_red_two_blue :
  (12.choose 2 * 8.choose 2) / (20.choose 4) = 168 / 323 :=
  sorry

end probability_two_red_two_blue_l159_159354


namespace tangent_x_intercept_is_25_over_3_l159_159682

noncomputable def circle (x y : ℝ) : Prop := x^2 + y^2 = 25

def point_of_tangency : ℝ × ℝ := (3, 4)

noncomputable def tangent_line_x_intercept (x : ℝ) : Prop :=
  ∃ m : ℝ, m * (x - point_of_tangency.1) = point_of_tangency.2 ∧ m ≠ 0

theorem tangent_x_intercept_is_25_over_3 :
  tangent_line_x_intercept (25 / 3) :=
sorry

end tangent_x_intercept_is_25_over_3_l159_159682


namespace intersection_M_P_l159_159115

def M : set ℝ := {y | ∃ x : ℝ, y = x⁻²}
def P : set ℝ := {y | ∃ x : ℝ, y = x}
def answer : set ℝ := {y | y > 0}

theorem intersection_M_P : M ∩ P = answer := by
  sorry

end intersection_M_P_l159_159115


namespace Joan_shovels_in_50_minutes_l159_159601

noncomputable def Joan_shovel_time (J : ℝ) : Prop :=
  let combined_work_rate (J : ℝ) (M : ℝ) : ℝ := (1/J) + (1/M)
  let total_time_work_rate (T : ℝ) : ℝ := 1/T
  combined_work_rate J 20 = total_time_work_rate 14.29

theorem Joan_shovels_in_50_minutes : (J : ℝ) (Joan_shovel_time J) → J = 50 :=
by
  sorry

end Joan_shovels_in_50_minutes_l159_159601


namespace triple_f_of_two_l159_159217

def f (x : ℝ) : ℝ :=
  if x > 9 then real.sqrt x else x^2

theorem triple_f_of_two : f (f (f 2)) = 4 := 
by 
  sorry

end triple_f_of_two_l159_159217


namespace cos_minus_sin_l159_159913

noncomputable def f (α : ℝ) : ℝ := 
  (sin (real.pi - α) * cos (2 * real.pi - α) * tan (-real.pi + α)) / 
  (sin (-real.pi + α) * tan (-α + 3 * real.pi))

theorem cos_minus_sin (α : ℝ) (h1 : f α = (1 : ℝ) / 8)
  (h2 : (real.pi / 4) < α ∧ α < (real.pi / 2)) :
  (cos α - sin α) = (1 - (3 * sqrt 7)) / 8 :=
sorry

end cos_minus_sin_l159_159913


namespace problem1_problem2_l159_159805

-- Problem 1: Four-digit even numbers with no repeated digits from {1, 2, 3, 4, 5, 6, 7}
theorem problem1 :
  ∃ (n : ℕ), n = 360 ∧ 
  (∀ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  ({a, b, c, d} ⊆ {1, 2, 3, 4, 5, 6, 7}) ∧
  (d = 2 ∨ d = 4 ∨ d = 6) ∧
  (1000 * a + 100 * b + 10 * c + d = n))
  := sorry

-- Problem 2: Five-digit numbers divisible by 5 with no repeated digits from {0, 1, 2, 3, 4, 5}
theorem problem2 :
  ∃ (n : ℕ), n = 216 ∧ 
  (∀ (a b c d e : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
      b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
      c ≠ d ∧ c ≠ e ∧ 
      d ≠ e ∧ 
  ({a, b, c, d, e} ⊆ {0, 1, 2, 3, 4, 5}) ∧
  (e = 0 ∨ e = 5) ∧
  (a ≠ 0) ∧
  (10000 * a + 1000 * b + 100 * c + 10 * d + e = n))
  := sorry

end problem1_problem2_l159_159805


namespace vovochka_correct_sum_cases_vovochka_min_difference_l159_159155

-- Part (a)
theorem vovochka_correct_sum_cases : 
  (∑ (a : ℕ) in finset.range 10, (∑ (b : ℕ) in finset.range (10 - a), 1)) ^ 3 = 244620 :=
sorry

-- Part (b)
theorem vovochka_min_difference : 
  ∃ (a b c x y z : ℕ), (a * 100 + b * 10 + c + x * 100 + y * 10 + z) - (a + x) * 100 - (b + y) * 10 - (c + z) = 1800 :=
sorry

end vovochka_correct_sum_cases_vovochka_min_difference_l159_159155


namespace acute_triangle_l159_159650

theorem acute_triangle (a b c : ℝ) (n : ℕ) (h_n : 2 < n) (h_eq : a^n + b^n = c^n) : a^2 + b^2 > c^2 :=
sorry

end acute_triangle_l159_159650


namespace monotonic_intervals_a_neg_half_range_of_a_f_nonnegative_l159_159222

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (exp x - 1) + a * x^2

theorem monotonic_intervals_a_neg_half :
  let f_x := λ x => f (-1/2) x in
  (∀ x, -1 < x ∧ x < 0 → f_x x < 0) ∧
  (∀ x, x < -1 ∨ 0 < x → f_x x > 0) := sorry

theorem range_of_a_f_nonnegative :
  (∀ x ≥ 0, ∀ a, f a x ≥ 0) ↔ (-1 ≤ a) := sorry

end monotonic_intervals_a_neg_half_range_of_a_f_nonnegative_l159_159222


namespace solve_eq_fraction_l159_159422

theorem solve_eq_fraction (x : ℝ) (hx3 : x ≠ 3) (hx4 : x ≠ 4) :
  (3 / (x - 3) = 4 / (x - 4)) → x = 0 := 
by 
  intro h
  have hl := calc
    3 * (x - 4) = 4 * (x - 3) : by rw h
  3 * (x - 4) = 4 * (x - 3)
  then_dropout
  rw succ_pred_order_dec_eq_ne_eq
  solve | expr_struct_chain expr_struct_chain (sample_spl_crds [mathlib.tactic.est_fin_subseq x], Matching.LinearMatchingStraight"[]"), (calc.unidata_unenroll_checkₓ 12), (calc.logf_iff_pritem_pure_name_sample _timestep_pure_tpt_attr_ordinal 4 | eoi_ordel_lfun_iff_pure (min.v_pair.middle_synthesis_gap_expr_sample _del_calc.extraction_sample_fin_mdl)), (rewrite_aligned { 3, 4, fiper_ct_lib_prim ! omega_orient <-not_cases_eq })
    | simp_expr.app.div_side_r.exact [conf_pavoir] found_eq _
end
snope_contract_err_repr_sound_eq_dial_deflamtrm_9_eq_deg_id_pre_norm_prv.
qed sorry

#print solve_eq_fraction

end solve_eq_fraction_l159_159422


namespace original_manufacturing_cost_l159_159788

variable (SP OC : ℝ)
variable (ManuCost : ℝ) -- Declaring manufacturing cost

-- Current conditions
axiom profit_percentage_constant : ∀ SP, 0.5 * SP = SP - 50

-- Problem Statement
theorem original_manufacturing_cost : (∃ OC, 0.5 * SP - OC = 0.5 * SP) ∧ ManuCost = 50 → OC = 50 := by
  sorry

end original_manufacturing_cost_l159_159788


namespace angle_AMB_30_l159_159668

open EuclideanGeometry

variables {A B C A1 C1 M : Point}

theorem angle_AMB_30 (h1 : ∠ABC = 60) 
  (h2 : Altitude A A1 B C) 
  (h3 : Altitude C C1 A B)
  (h4 : ∠AMC = 60)
  (h5 : perpendicular (Line.mk B M) (Line.mk A1 C1))
  (h6 : M ≠ B) : ∠AMB = 30 :=
sorry

end angle_AMB_30_l159_159668


namespace cosine_of_angle_between_vectors_l159_159433

theorem cosine_of_angle_between_vectors (a1 b1 c1 a2 b2 c2 : ℝ) :
  let u := (a1, b1, c1)
  let v := (a2, b2, c2)
  let dot_product := a1 * a2 + b1 * b2 + c1 * c2
  let magnitude_u := Real.sqrt (a1^2 + b1^2 + c1^2)
  let magnitude_v := Real.sqrt (a2^2 + b2^2 + c2^2)
  dot_product / (magnitude_u * magnitude_v) = 
      (a1 * a2 + b1 * b2 + c1 * c2) / (Real.sqrt (a1^2 + b1^2 + c1^2) * Real.sqrt (a2^2 + b2^2 + c2^2)) :=
by
  let u := (a1, b1, c1)
  let v := (a2, b2, c2)
  let dot_product := a1 * a2 + b1 * b2 + c1 * c2
  let magnitude_u := Real.sqrt (a1^2 + b1^2 + c1^2)
  let magnitude_v := Real.sqrt (a2^2 + b2^2 + c2^2)
  sorry

end cosine_of_angle_between_vectors_l159_159433


namespace gcd_lcm_product_l159_159763

theorem gcd_lcm_product (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 360) :
  ∃ (s : Finset ℕ), s.card = 12 ∧ ∀ d ∈ s, d = Nat.gcd a b :=
sorry

end gcd_lcm_product_l159_159763


namespace average_weight_of_a_and_b_l159_159264

-- Define the parameters in the conditions
variables (A B C : ℝ)

-- Conditions given in the problem
theorem average_weight_of_a_and_b (h1 : (A + B + C) / 3 = 45) 
                                 (h2 : (B + C) / 2 = 43) 
                                 (h3 : B = 33) : (A + B) / 2 = 41 := 
sorry

end average_weight_of_a_and_b_l159_159264


namespace prove_m_range_l159_159118

theorem prove_m_range (m : ℝ) :
  (∀ x : ℝ, (2 * x + 5) / 3 - 1 ≤ 2 - x → 3 * (x - 1) + 5 > 5 * x + 2 * (m + x)) → m < -3 / 5 := by
  sorry

end prove_m_range_l159_159118


namespace first_half_day_wednesday_l159_159581

theorem first_half_day_wednesday (h1 : ¬(1 : ℕ) = (4 % 7) ∨ 1 % 7 != 0)
  (h2 : ∀ d : ℕ, d ≤ 31 → d % 7 = ((d + 3) % 7)) : 
  ∃ d : ℕ, d = 25 ∧ ∃ W : ℕ → Prop, W d := sorry

end first_half_day_wednesday_l159_159581


namespace sum_of_cubes_eq_neg_27_l159_159206

variable {a b c : ℝ}

-- Define the condition that k is the same for a, b, and c
def same_k (a b c k : ℝ) : Prop :=
  k = (a^3 + 9) / a ∧ k = (b^3 + 9) / b ∧ k = (c^3 + 9) / c

-- Theorem: Given the conditions, a^3 + b^3 + c^3 = -27
theorem sum_of_cubes_eq_neg_27 (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_same_k : ∃ k, same_k a b c k) :
  a^3 + b^3 + c^3 = -27 :=
sorry

end sum_of_cubes_eq_neg_27_l159_159206


namespace minimum_value_expression_l159_159066

theorem minimum_value_expression
  (a b c d : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  let A := (a^2 + b^2) / (c * d) ^ 4 + 
           (b^2 + c^2) / (a * d) ^ 4 + 
           (c^2 + d^2) / (a * b) ^ 4 + 
           (d^2 + a^2) / (b * c) ^ 4 in
  A ≥ 64 :=
by
  sorry

end minimum_value_expression_l159_159066


namespace amc_acronym_length_l159_159667

theorem amc_acronym_length :
  let grid_spacing : ℕ := 1
  let straight_segments : list ℕ := [/* lengths of each straight segment */]
  let slanted_segments : list ℝ := [/* lengths of each slanted segment */]
  in (sum straight_segments + sum slanted_segments = 13 + 4 * Real.sqrt 2) :=
by
  /- You would define the lengths of the segments explicitly here if necessary, following the conditions outlined. -/
  sorry

end amc_acronym_length_l159_159667


namespace base_n_representation_of_b_l159_159557

def a_in_base_n (n : ℕ) : ℕ := 2 * n + 1

def b_in_base_n (n : ℕ) : ℕ := n * n + n

theorem base_n_representation_of_b (n : ℕ) (h : n > 9) :
  (a_in_base_n n = 2 * n + 1) → (b_in_base_n n = n * (n + 1)) → 
  b_in_base_n n = n * n + n := by
  intros ha hb
  rw [b_in_base_n]
  exact hb

end base_n_representation_of_b_l159_159557


namespace avg_children_in_families_with_children_l159_159578

theorem avg_children_in_families_with_children :
  (n_families : ℕ) (avg_children_per_family : ℕ) (num_childless_families : ℕ) 
  (total_children : ℕ) (families_with_children : ℕ)
  (h1 : n_families = 12)
  (h2 : avg_children_per_family = 3)
  (h3 : num_childless_families = 3)
  (h4 : total_children = n_families * avg_children_per_family)
  (h5 : families_with_children = n_families - num_childless_families) :
  (total_children / families_with_children : ℝ) = 4.0 :=
by
  sorry

end avg_children_in_families_with_children_l159_159578


namespace vovochka_correct_sum_combinations_l159_159152

theorem vovochka_correct_sum_combinations : 
  let digit_pairs := finset.filter (λ p : ℕ × ℕ, (p.fst + p.snd) < 10) ((finset.range 10).product (finset.range 10))
  let no_carry_combinations := finset.card digit_pairs
  no_carry_combinations ^ 3 * 81 = 244620 := 
by
  sorry

end vovochka_correct_sum_combinations_l159_159152


namespace find_p_l159_159174

def distance_from_point_to_line (point : ℝ × ℝ) (line : ℝ → ℝ) : ℝ :=
| line 0

axiom distance_formula : ∀ (x1 x2 : ℝ), abs (x1 - x2) = distance_from_point_to_line (x1, 0) (λ y, x2)

theorem find_p (p : ℝ) (h : p > 0) (h_distance : distance_formula 3 (-p/2) = 4) : p = 2 := 
by
  sorry

end find_p_l159_159174


namespace number_of_correct_descriptions_l159_159866

open BigOperators

variable (ξ : Type) [ProbabilityTheory.ProbabilitySpace ξ]

-- (1) Condition: Eξ and Dξ are numerical values.
variable (Eξ Dξ : ℝ)
axiom expectation_and_variance_inherent : (∃ ξ_val : ξ, ∃ prob : ProbabilityTheory.probability ξ_val, 
                                  Eξ = ProbabilityTheory.expectation ξ_val ∧ 
                                  Dξ = ProbabilityTheory.variance ξ_val)

-- (2) Condition: If all possible values of ξ lie within [a, b], then a ≤ Eξ ≤ b.
variable (a b : ℝ)
axiom expectation_within_interval : (∀ ξ_val : ξ, ξ_val ∈ Set.Icc a b) → a ≤ Eξ ∧ Eξ ≤ b

-- (3) Condition: Expectation reflects average level, variance reflects fluctuation, etc.
axiom expectation_variance_properties : 
  (∃ ξ_val : ξ, ProbabilityTheory.expectation ξ_val = Eξ ∧ ProbabilityTheory.variance ξ_val = Dξ 
   ∧ (∀ x, x ∈ ξ_val → Eξ = ProbabilityTheory.expectation ξ_val))

-- (4) Condition: Expected value can be any real number, while variance is non-negative.
axiom expectation_and_variance_real : (∀ ξ_val : ξ, Eξ = ProbabilityTheory.expectation ξ_val 
                                    ∧ Dξ = ProbabilityTheory.variance ξ_val 
                                    ∧ Eξ ∈ Set.univ ℝ 
                                    ∧ 0 ≤ Dξ)

theorem number_of_correct_descriptions : 
  (∃ ξ_val : ξ, ∃ prob : ProbabilityTheory.probability ξ_val,
  Eξ = ProbabilityTheory.expectation ξ_val 
  ∧ Dξ = ProbabilityTheory.variance ξ_val 
  ∧ (∀ x, x ∈ ξ_val → (a ≤ Eξ ∧ Eξ ≤ b) 
  ∧ (∃ ξ_val : ξ, ProbabilityTheory.expectation ξ_val = Eξ ∧ ProbabilityTheory.variance ξ_val = Dξ 
     ∧ (∀ x, x ∈ ξ_val → Eξ = ProbabilityTheory.expectation ξ_val)) 
  ∧ Eξ ∈ Set.univ ℝ ∧ 0 ≤ Dξ))) := sorry

end number_of_correct_descriptions_l159_159866


namespace line_parabola_one_point_l159_159681

theorem line_parabola_one_point (k : ℝ) :
  (∃ x y : ℝ, y = k * x + 2 ∧ y^2 = 8 * x) 
  → (k = 0 ∨ k = 1) := 
by 
  sorry

end line_parabola_one_point_l159_159681


namespace triangle_third_side_lengths_l159_159524

theorem triangle_third_side_lengths : 
  ∃ (x : ℕ), (3 < x ∧ x < 11) ∧ (x ≠ 3) ∧ (x ≠ 11) ∧ 
    ((x = 4) ∨ (x = 5) ∨ (x = 6) ∨ (x = 7) ∨ (x = 8) ∨ (x = 9) ∨ (x = 10)) :=
by
  sorry

end triangle_third_side_lengths_l159_159524


namespace triangle_function_range_l159_159116

noncomputable def is_triangle_function (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
  f a + f b > f c ∧ f a + f c > f b ∧ f b + f c > f a

theorem triangle_function_range (m : ℝ) :
  (∀ x ∈ Set.Icc (1 / Real.exp 2) Real.exp, 
    is_triangle_function (λ x, x * Real.log x + m) (1 / Real.exp 2) (1 / Real.exp) Real.exp) ↔ 
  m ∈ Set.Ioi ((Real.exp 2 + 2) / Real.exp) :=
begin
  sorry
end

end triangle_function_range_l159_159116


namespace cube_root_neg_8_eq_neg_2_l159_159408

theorem cube_root_neg_8_eq_neg_2 : ∃ x : ℝ, x^3 = -8 ∧ x = -2 :=
by
  use -2
  split
  . show (-2 : ℝ)^3 = -8 by sorry
  . show -2 = -2 by rfl

end cube_root_neg_8_eq_neg_2_l159_159408


namespace right_triangle_side_condition_l159_159423

theorem right_triangle_side_condition (a d : ℝ) (h₁ : a > 0) (h₂ : d > 1) :
  (a * d^2)^2 = a^2 + (a * d)^2 ↔ d = Real.sqrt((1 + Real.sqrt 5) / 2) :=
by sorry

end right_triangle_side_condition_l159_159423


namespace gcd_lcm_product_l159_159764

theorem gcd_lcm_product (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 360) :
  ∃ (s : Finset ℕ), s.card = 12 ∧ ∀ d ∈ s, d = Nat.gcd a b :=
sorry

end gcd_lcm_product_l159_159764


namespace phi_is_sufficient_but_not_necessary_l159_159329

theorem phi_is_sufficient_but_not_necessary (φ : ℝ) :
  (∀ x, sin (2 * x + φ) = 0 → x = 0) → (φ = π → sin (2 * x + π) = -sin (2 * x)) ∧ (∃ φ', φ' ≠ π ∧ sin (2 * x + φ') = 0 ∧ ∀ x, x = 0) := sorry

end phi_is_sufficient_but_not_necessary_l159_159329


namespace vasya_birthday_day_l159_159738

/-- Define the days of the week as an inductive type --/
inductive Day
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday
deriving DecidableEq

open Day

/-- Function to get the day after a given day --/
def next_day : Day → Day
| Monday    => Tuesday
| Tuesday   => Wednesday
| Wednesday => Thursday
| Thursday  => Friday
| Friday    => Saturday
| Saturday  => Sunday
| Sunday    => Monday

/-- Function to get the day two days after a given day --/
def day_after_tomorrow (d : Day) : Day :=
  next_day (next_day d)

/-- Lean statement for the problem --/
theorem vasya_birthday_day (today : Day) (H1 : day_after_tomorrow today = Sunday) (H2 : next_day vasya_birthday = today) : vasya_birthday = Thursday := 
sorry

end vasya_birthday_day_l159_159738


namespace quadratic_function_exists_l159_159393

-- Define all functions according to the conditions
def f₁ (x : ℝ) : ℝ := (1 / x)
def f₂ (x : ℝ) : ℝ := (x + 1)
def f₃ (x : ℝ) : ℝ := (2 * x ^ 2 - 1)
def f₄ (x : ℝ) : ℝ := (2 / 3 * x)

-- The statement to be proved
theorem quadratic_function_exists :
  (∃ (i : Fin 4), 
    if i = 0 then 
      ∃ a b c : ℝ, a ≠ 0 ∧ f₁ = λ x, a * x^2 + b * x + c 
    else if i = 1 then 
      ∃ a b c : ℝ, a ≠ 0 ∧ f₂ = λ x, a * x^2 + b * x + c 
    else if i = 2 then 
      ∃ a b c : ℝ, a ≠ 0 ∧ f₃ = λ x, a * x^2 + b * x + c 
    else 
      ∃ a b c : ℝ, a ≠ 0 ∧ f₄ = λ x, a * x^2 + b * x + c 
  ). 
Proof :=
sorry

end quadratic_function_exists_l159_159393


namespace proof_of_m_l159_159073

noncomputable def m : ℕ := 10

theorem proof_of_m :
  ∃ (m : ℕ), 
    m > 0 ∧ 
    Nat.lcm 30 m = 90 ∧ 
    Nat.lcm m 50 = 200 ∧
    m = 10 := by
  use 10
  split
  · exact Nat.zero_lt_succ 9 -- Ensures m is positive
  split
  · exact Nat.lcm_eq_right 10 30 -- Ensures m lcm 30 is 90
  split
  · exact Nat.lcm_eq_right 10 50 -- Ensures m lcm 50 is 200
  · reflexivity -- Ensures m == 10
  sorry   -- Skip the detailed proof

end proof_of_m_l159_159073


namespace find_b2015_l159_159467

noncomputable def f_1 (x : ℝ) : ℝ := (x^2 + 2*x + 1) * exp x

def derive_f (f : ℝ → ℝ) : ℝ → ℝ := λ x, (deriv f x)

def f_n (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0     => f_1 x
  | n + 1 => derive_f (f_n n) x
  end

def functional_form (n : ℕ) : ℝ → ℝ :=
  λ x, let a_n := 0 in -- a_n is not provided explicitly in the problem
       let b_n := 2 * n in
       let c_n := sorry in -- c_n needs further computation based on pattern observed
       (a_n * x^2 + b_n * x + c_n) * exp x

theorem find_b2015 : functional_form 2015 = (λ x, 0 * x^2 + 4030 * x + sorry) := sorry

end find_b2015_l159_159467


namespace perpendicular_intersection_on_diagonal_l159_159901

theorem perpendicular_intersection_on_diagonal
  (A B C D M Q P R T : Point)
  (circumcircle_ABC : Circle)
  (H1 : is_circumcircle_of_rect ABCD circumcircle_ABC)
  (H2 : on_circle M circumcircle_ABC)
  (H3 : perpendicular MQ AD)
  (H4 : perpendicular MP BC)
  (H5 : perpendicular MR (extension_of AB))
  (H6 : perpendicular MT (extension_of CD)) :
  (is_perpendicular PR QT) ∧ (lies_on_diagonal ((intersection PR QT) AC)) :=
sorry

end perpendicular_intersection_on_diagonal_l159_159901


namespace incenter_is_equidistant_from_sides_l159_159686

theorem incenter_is_equidistant_from_sides
  (A B C : Type)
  [metric_space A] [metric_space B] [metric_space C]
  (triangle : Type) 
  (altitudes : triangle → (A × B × C))
  (angle_bisectors : triangle → (A × B × C))
  (medians : triangle → (A × B × C))
  (perpendicular_bisectors : triangle → (A × B × C))
  (is_incenter : (A × B × C) → Prop := λ p, ∀ t : triangle, angle_bisectors t = p)
  (equidistant_from_sides : (A × B × C) → Prop := λ p, ∀ t : triangle, p = angle_bisectors t → ∀ s1 s2 s3, dist p = dist s1 ∧ dist p = dist s2 ∧ dist p = dist s3)
  (incenter : A × B × C)
  (h1 : ∃ p, is_incenter p)
  (h2 : ∃ p, equidistant_from_sides p) :
  is_incenter incenter → equidistant_from_sides incenter :=
sorry

end incenter_is_equidistant_from_sides_l159_159686


namespace race_lead_distance_l159_159572

theorem race_lead_distance :
  ∀ (d12 d13 : ℝ) (s1 s2 s3 t : ℝ), 
  d12 = 2 →
  d13 = 4 →
  t > 0 →
  s1 = (d12 / t + s2) →
  s1 = (d13 / t + s3) →
  s2 * t - s3 * t = 2.5 :=
by
  sorry

end race_lead_distance_l159_159572


namespace opposite_signs_abs_larger_l159_159911

theorem opposite_signs_abs_larger (a b : ℝ) (h1 : a + b < 0) (h2 : a * b < 0) :
  (a < 0 ∧ b > 0 ∧ |a| > |b|) ∨ (a > 0 ∧ b < 0 ∧ |b| > |a|) :=
sorry

end opposite_signs_abs_larger_l159_159911


namespace find_z_coordinate_of_point_on_line_passing_through_l159_159821

theorem find_z_coordinate_of_point_on_line_passing_through
  (p1 p2 : ℝ × ℝ × ℝ)
  (x_value : ℝ)
  (z_value : ℝ)
  (h1 : p1 = (1, 3, 2))
  (h2 : p2 = (4, 2, -1))
  (h3 : x_value = 3)
  (param : ℝ)
  (h4 : x_value = (1 + 3 * param))
  (h5 : z_value = (2 - 3 * param)) :
  z_value = 0 := by
  sorry

end find_z_coordinate_of_point_on_line_passing_through_l159_159821


namespace num_gcd_values_l159_159772

-- Define the condition for the product of gcd and lcm
def is_valid_pair (a b : ℕ) : Prop :=
  gcd a b * Nat.lcm a b = 360

-- Define the main theorem statement
theorem num_gcd_values : 
  ∃ (n : ℕ), 
    (∀ a b, is_valid_pair a b → ∃ m (hm: m ≤ 360), gcd a b = m) ∧ 
    n = 12 := sorry

end num_gcd_values_l159_159772


namespace problem_l159_159555

theorem problem 
  (a : ℝ) 
  (h : (a + 1/(3*a))^2 = 3) : 
  27*a^3 + 1/(a^3) = ±54*Real.sqrt 3 := 
sorry

end problem_l159_159555


namespace num_gcd_values_l159_159774

-- Define the condition for the product of gcd and lcm
def is_valid_pair (a b : ℕ) : Prop :=
  gcd a b * Nat.lcm a b = 360

-- Define the main theorem statement
theorem num_gcd_values : 
  ∃ (n : ℕ), 
    (∀ a b, is_valid_pair a b → ∃ m (hm: m ≤ 360), gcd a b = m) ∧ 
    n = 12 := sorry

end num_gcd_values_l159_159774


namespace greatest_possible_value_l159_159905

theorem greatest_possible_value :
  ∃ (u v : ℝ) (a b : ℝ),
  (u + v = u^2 + v^2) ∧ 
  (u + v = u^4 + v^4) ∧ 
  (u + v = u^6 + v^6) ∧ 
  (u + v = u^8 + v^8) ∧ 
  (u + v = u^{10} + v^{10}) ∧ 
  (u + v = u^{12} + v^{12}) ∧ 
  (u + v = u^{14} + v^{14}) ∧ 
  (u + v = u^{16} + v^{16}) ∧ 
  (u + v = u^{18} + v^{18}) ∧ 
  (is_root (X^2 - C a * X + C b) u) ∧ 
  (is_root (X^2 - C a * X + C b) v) ∧ 
  ( ∀ x y : ℝ, (x ≠ 1 ∨ y ≠ 1) ∨  ( x^2 - x + 1 = 0) ∧ (y^2 - y + 1 = 0) ∨  ( x =y) ) → 
  greatest_possible_value u v = 2 :=
  sorry

end greatest_possible_value_l159_159905


namespace dot_product_OA_OB_OP_l159_159417

noncomputable section

open Classical

variables (P : ℝ × ℝ := (2, 1))
variables (f : ℝ → ℝ := λ x, (2 * x + 3) / (2 * x - 4))
variables (A : ℝ × ℝ := (-1, 11 / 2))
variables (B : ℝ × ℝ := (2, 1))

def vector (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.1, point.2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem dot_product_OA_OB_OP :
  let OA := vector A
  let OB := vector B
  let OP := vector P
  dot_product (OA.1 + OB.1, OA.2 + OB.2) OP = 17 / 2 :=
  by
    sorry

end dot_product_OA_OB_OP_l159_159417


namespace guppies_total_l159_159098

theorem guppies_total : 
  let HayleeGuppies := 3 * 12,
      JoseGuppies := HayleeGuppies / 2,
      CharlizGuppies := JoseGuppies / 3,
      NicolaiGuppies := 4 * CharlizGuppies,
      AliceGuppies := NicolaiGuppies + 5,
      BobGuppies := (JoseGuppies + CharlizGuppies) / 2,
      CameronGuppies := 2^(3) in
  HayleeGuppies + JoseGuppies + CharlizGuppies + NicolaiGuppies + AliceGuppies + BobGuppies + CameronGuppies = 133 :=
by
  -- Proof goes here
  sorry

end guppies_total_l159_159098


namespace translation_symmetric_l159_159678

theorem translation_symmetric (x : ℝ) : 
  let f := λ x, sin (2 * x + π / 3),
      g := λ x, sin (2 * (x - π / 12) + π / 3)
  in g (-π / 12) = 0 ↔ f (-π / 12) = 0 :=
by sorry

end translation_symmetric_l159_159678


namespace find_symmetric_curve_equation_l159_159013

def equation_of_curve_symmetric_to_line : Prop :=
  ∀ (x y : ℝ), (5 * x^2 + 12 * x * y - 22 * x - 12 * y - 19 = 0 ∧ x - y + 2 = 0) →
  12 * x * y + 5 * y^2 - 78 * y + 45 = 0

theorem find_symmetric_curve_equation : equation_of_curve_symmetric_to_line :=
sorry

end find_symmetric_curve_equation_l159_159013


namespace cone_volume_divide_by_pi_eval_l159_159367

noncomputable def cone_volume_divide_by_pi : ℝ :=
  let r := 40 / 3 in
  let h := 20 * Real.sqrt 5 / 3 in
  (1/3) * r^2 * h

theorem cone_volume_divide_by_pi_eval :
  cone_volume_divide_by_pi = 32000 * Real.sqrt 5 / 81 :=
by
  sorry

end cone_volume_divide_by_pi_eval_l159_159367


namespace Q_at_neg2_l159_159184

noncomputable def Q (z : ℂ) : ℂ :=
  ∏ k in finset.range 2019, z - (2 * complex.cos ((2 * k * real.pi) / 2019) + complex.sin ((2 * k * real.pi) / 2019) * complex.I)

theorem Q_at_neg2 : Q (-2) = (-1 - 3^2019) / 2^2018 := 
sorry

end Q_at_neg2_l159_159184


namespace minimum_e1_squared_plus_e2_squared_l159_159937

-- Definitions
variable {F1 F2 P : Point} 
variable (e1 e2 : ℝ)   -- eccentricities of ellipse and hyperbola respectively

-- Conditions
axiom commonFoci : ellipse F1 F2 -> hyperbola F1 F2 -> sharingFoci
axiom intersectionPoint  : ∃ P : Point, intersects_ellipse_hyperbola P
axiom angleFPF : angle F1 P F2 = 60

-- The Theorem
theorem minimum_e1_squared_plus_e2_squared : commonFoci → intersectionPoint → angleFPF → 
  (e1^2 + e2^2) = 1 + (Real.sqrt 3 / 2) :=
sorry

end minimum_e1_squared_plus_e2_squared_l159_159937


namespace maximum_surface_area_of_cuboid_l159_159088

noncomputable def max_surface_area_of_inscribed_cuboid (R : ℝ) :=
  let (a, b, c) := (R, R, R) -- assuming cube dimensions where a=b=c
  2 * a * b + 2 * a * c + 2 * b * c

theorem maximum_surface_area_of_cuboid (R : ℝ) (h : ∃ a b c : ℝ, a^2 + b^2 + c^2 = 4 * R^2) :
  max_surface_area_of_inscribed_cuboid R = 8 * R^2 :=
sorry

end maximum_surface_area_of_cuboid_l159_159088


namespace tanx_eq_2_sin2cos2_tanx_eq_2_cos_sin_ratio_l159_159909

theorem tanx_eq_2_sin2cos2 (x : ℝ) (h : Real.tan x = 2) : 
  (2 / 3) * (Real.sin x) ^ 2 + (1 / 4) * (Real.cos x) ^ 2 = 7 / 12 := 
by 
  sorry

theorem tanx_eq_2_cos_sin_ratio (x : ℝ) (h : Real.tan x = 2) : 
  (Real.cos x + Real.sin x) / (Real.cos x - Real.sin x) = -3 := 
by 
  sorry

end tanx_eq_2_sin2cos2_tanx_eq_2_cos_sin_ratio_l159_159909


namespace vector_projection_correct_l159_159438

-- Define the vectors
def a : ℝ × ℝ × ℝ := (3, -6, 2)
def b : ℝ × ℝ × ℝ := (1, -2, 3)

-- Define the dot product for ℝ × ℝ × ℝ
def dot_product (v w : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3

-- Define the projection function
def proj (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let scalar := (dot_product a b) / (dot_product b b)
  in (scalar * b.1, scalar * b.2, scalar * b.3)

-- The target projection
def target_projection : ℝ × ℝ × ℝ := (3/2, -3, 9/2)

-- The theorem to prove
theorem vector_projection_correct : proj a b = target_projection :=
  by
    -- This is where the proof would go
    sorry

end vector_projection_correct_l159_159438


namespace cube_skew_lines_no_30_degree_l159_159461

theorem cube_skew_lines_no_30_degree :
  ∀ (v1 v2 v3 v4 : ℝ × ℝ × ℝ),
    (v1 ∈ cube_vertices) → (v2 ∈ cube_vertices) → (v3 ∈ cube_vertices) → (v4 ∈ cube_vertices) →
    (is_skew v1 v2) → (is_skew v3 v4) →
    ¬ angle_between_skew_lines v1 v2 v3 v4 = 30 :=
by
  sorry

def cube_vertices : set (ℝ × ℝ × ℝ) :=
  {v | ∃ (x y z : ℝ), (x = 0 ∨ x = 1) ∧ (y = 0 ∨ y = 1) ∧ (z = 0 ∨ z = 1)}

def is_skew (p1 p2 : ℝ × ℝ × ℝ) : Prop :=
  -- Define the condition for points p1 and p2 to be skew (not parallel and not intersecting)
  sorry

def angle_between_skew_lines (p1 p2 p3 p4 : ℝ × ℝ × ℝ) : ℝ :=
  -- Calculate the angle between two skew lines formed by points (p1, p2) and (p3, p4)
  sorry

end cube_skew_lines_no_30_degree_l159_159461


namespace fractional_part_of_water_after_replacements_l159_159809

theorem fractional_part_of_water_after_replacements :
  let total_quarts := 25
  let removed_quarts := 5
  (1 - removed_quarts / (total_quarts : ℚ))^3 = 64 / 125 :=
by
  sorry

end fractional_part_of_water_after_replacements_l159_159809


namespace trash_cans_in_veterans_park_after_exchange_l159_159414

theorem trash_cans_in_veterans_park_after_exchange :
  ∀ (C L : ℕ),
    (24 : ℕ) = 24 →
    (C = 12 + 8) →
    (L = 2 * C) →
    let half_C := C / 2 in 
    let third_half_C := half_C / 3 in
    let two_third_half_C := 2 * half_C / 3 in
    let fourth_L := L / 4 in
    24 + third_half_C + fourth_L = 37 :=
by
  intros,
  sorry

end trash_cans_in_veterans_park_after_exchange_l159_159414


namespace daps_from_dips_l159_159107

section DapsDopsDips

variable (Daps Dops Dips : Type)
variable [has_scalar ℚ Daps] [has_scalar ℚ Dops] [has_scalar ℚ Dips]

-- Condition 1: 5 daps = 4 dops
def condition1 (d : Daps) (p : Dops) : Prop := (5 : ℚ) • d = (4 : ℚ) • p

-- Condition 2: 3 dops = 9 dips
def condition2 (p : Dops) (i : Dips) : Prop := (3 : ℚ) • p = (9 : ℚ) • i

-- Theorem: How many daps are equivalent to 54 dips?
theorem daps_from_dips (d : Daps) (p : Dops) (i : Dips)
  (h1 : condition1 d p) (h2 : condition2 p i) : (54 : ℚ) • i = (22.5 : ℚ) • d :=
by 
  sorry

end DapsDopsDips

end daps_from_dips_l159_159107


namespace projection_matrix_inverse_zero_l159_159611

theorem projection_matrix_inverse_zero (u : ℝ) (v : ℝ) (Q : Matrix (Fin 2) (Fin 2) ℝ) (h1 : u = 4) (h2 : v = 5)
  (h3 : Q = (Matrix.vecCons (Matrix.vecCons (4 / Real.sqrt 41) (5 / Real.sqrt 41) Matrix.vecEmpty)
                            (Matrix.vecCons (4 / Real.sqrt 41) (5 / Real.sqrt 41) Matrix.vecEmpty)).mul
              (Matrix.vecCons (Matrix.vecCons (4 / Real.sqrt 41) (5 / Real.sqrt 41) Matrix.vecEmpty)
                              (Matrix.vecCons (4 / Real.sqrt 41) (5 / Real.sqrt 41) Matrix.vecEmpty))) :
  Q.det = 0 → Q⁻¹ = 0 :=
by
  sorry

end projection_matrix_inverse_zero_l159_159611


namespace de_morgan_implication_l159_159208

variables (p q : Prop)

theorem de_morgan_implication (h : ¬(p ∧ q)) : ¬p ∨ ¬q :=
sorry

end de_morgan_implication_l159_159208


namespace quadratic_form_finite_and_invariant_l159_159193

def is_squarefree (n : ℤ) : Prop := ∀ k : ℤ, k * k ∣ n → k = 1 ∨ k = -1

def quadratic_form_pair_count (a b c n : ℤ) : ℤ :=
  Nat.card { p : ℤ × ℤ // a * p.1 * p.1 + b * p.1 * p.2 + c * p.2 * p.2 = n }

theorem quadratic_form_finite_and_invariant (a b c p : ℤ) (h_pos : a > 0)
  (h_eq : a * c - b * b = p) (h_squarefree : is_squarefree p) (h_pos_squarefree : p > 0) :
  ∀ (n : ℤ), ∃ M : ℤ, (quadratic_form_pair_count a b c n = M ∧
  ∀ k : ℕ, quadratic_form_pair_count a b c n = quadratic_form_pair_count a b c (p^k * n)) :=
by
  sorry

end quadratic_form_finite_and_invariant_l159_159193


namespace animals_on_stump_l159_159806

def possible_n_values (n : ℕ) : Prop :=
  n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9 ∨ n = 12 ∨ n = 15

theorem animals_on_stump (n : ℕ) (h1 : n ≥ 3) (h2 : n ≤ 20)
  (h3 : 11 ≥ (n + 1) / 3) (h4 : 9 ≥ n - (n + 1) / 3) : possible_n_values n :=
by {
  sorry
}

end animals_on_stump_l159_159806


namespace solve_sequence_sequence_b_sum_l159_159508

noncomputable def sequence_a : ℕ → ℝ 
| 1       := 1
| n + 1   := if n = 1 then 5 else sequence_a n + 2 * n

noncomputable def sequence_b (n : ℕ) : ℝ :=
(-1)^n * (sequence_a n + n)

noncomputable def sum_S (n : ℕ) : ℝ :=
finset.sum (finset.range (2 * n)) sequence_b

theorem solve_sequence {a : ℕ → ℝ} {λ : ℝ} (h1 : a 1 = 1) 
                                              (h3 : a 3 = 9)
                                              (hn : ∀ n ≥ 2, a n = a (n - 1) + λ (n - 1)) : 
  λ = 2 ∧ ∀ n, a n = n ^ 2 :=
sorry

theorem sequence_b_sum (n : ℕ) : sum_S n = 2 * n * (n + 1) :=
sorry

end solve_sequence_sequence_b_sum_l159_159508


namespace count_two_digit_numbers_with_digit_sum_perfect_square_under_25_l159_159973

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def digit_sum (n : ℕ) : ℕ := n / 10 + n % 10

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

theorem count_two_digit_numbers_with_digit_sum_perfect_square_under_25 :
  ∃ count : ℕ,
    (∀ n : ℕ, is_two_digit n → digit_sum n ∈ {1, 4, 9, 16} → n ∈ {10, 40, 31, 22, 13, 90, 81, 72, 63, 54, 45, 36, 27, 18, 97, 88, 79}) ∧
    count = 17 :=
by
  sorry

end count_two_digit_numbers_with_digit_sum_perfect_square_under_25_l159_159973


namespace range_of_f_l159_159037

def f (x : ℝ) : ℝ := (3 * x + 8) / (x - 4)

theorem range_of_f :
  set.range f = set.Iio 3 ∪ set.Ioi 3 :=
sorry

end range_of_f_l159_159037


namespace sunflower_percentage_in_brandA_l159_159816

theorem sunflower_percentage_in_brandA :
  (∀ (pA pB mA mB sB mX : ℝ),
    0 < pA ∧ pA < 1 ∧ -- Percentage of mix that is Brand A
    0 < pB ∧ pB < 1 ∧ -- Percentage of mix that is Brand B
    pA + pB = 1 ∧
    0 < mA ∧ mA < 1 ∧ -- Percentage of millet in Brand A
    0 < mB ∧ mB < 1 ∧ -- Percentage of millet in Brand B
    0 < sB ∧ sB < 1 ∧ -- Percentage of safflower in Brand B
    mX = pA * mA + pB * mB ∧ -- Final mix percentage of millet
    mX = 0.5 ∧ -- The mix is 50% millet
    mA = 0.4 ∧ -- Brand A is 40% millet
    mB = 0.65 ∧ -- Brand B is 65% millet
    sB = 0.35 -- Brand B is 35% safflower
    ) →
    ∃ (sA: ℝ), sA = 0.6 -- Percentage of sunflower in Brand A is 60%
    :=
begin
  -- Proof goes here
  sorry
end

end sunflower_percentage_in_brandA_l159_159816


namespace vovochka_add_correct_pairs_vovochka_min_diff_incorrect_l159_159133

-- Vovochka's addition method: adding two digits without carrying over
def vovochka_add (a b : ℕ) : ℕ := (a % 10 + b % 10) + ((a / 10 % 10 + b / 10 % 10) * 10) + ((a / 100 + b / 100) * 100)

-- Part (a): number of pairs producing correct result with Vovochka’s method
def correct_vovochka_pairs_count : ℕ := 244620

-- Part (b): smallest possible difference when Vovochka’s method is incorrect
def min_diff_vovochka_method : ℕ := 1800

-- Proving the number of correct cases equals 244620
theorem vovochka_add_correct_pairs :
  let count := ∑ (a b : ℕ) in finset.Icc 100 999, if vovochka_add a b = a + b then 1 else 0
  in count = correct_vovochka_pairs_count := sorry

-- Proving the smallest possible difference when Vovochka’s method is incorrect
theorem vovochka_min_diff_incorrect :
  let min_diff := min (∑ (a b : ℕ) in finset.Icc 100 999, if vovochka_add a b ≠ a + b then nat.abs (vovochka_add a b - (a + b)) else ⊤)
  in min_diff = min_diff_vovochka_method := sorry

end vovochka_add_correct_pairs_vovochka_min_diff_incorrect_l159_159133


namespace problem_statement_l159_159111

-- Definitions of x and y
def x : ℕ := 3
def y : ℕ := 4

-- The theorem we need to prove
theorem problem_statement : (x^5 + 3 * y^3) / 8 = 54.375 := 
by
  -- Proof goes here
  sorry

end problem_statement_l159_159111


namespace marble_probability_correct_l159_159348

noncomputable def marble_probability : ℚ :=
  let total_ways := (Nat.choose 20 4 : ℚ)
  let ways_two_red := (Nat.choose 12 2 : ℚ)
  let ways_two_blue := (Nat.choose 8 2 : ℚ)
  (ways_two_red * ways_two_blue) / total_ways

theorem marble_probability_correct : marble_probability = 56 / 147 :=
by
  -- Note: the proof is omitted as per instructions
  sorry

end marble_probability_correct_l159_159348


namespace area_ratio_R2_R1_l159_159190

noncomputable def R1 (x y : ℝ) : Prop := log (3 + x^2 + y^2) / log 10 ≤ 2 + log (2 * (x + y)) / log 10
noncomputable def R2 (x y : ℝ) : Prop := log (5 + x^2 + y^2) / log 10 ≤ 3 + log (3 * (x + y)) / log 10

theorem area_ratio_R2_R1 : (π * 4499995) / (π * 19997) = 225 :=
by
  sorry

end area_ratio_R2_R1_l159_159190


namespace temperature_decrease_denotation_l159_159989

theorem temperature_decrease_denotation :
  ∀ (increase : ℝ), (increase = 10) → ∃ (decrease : ℝ), decrease = -6 :=
by
  intro increase h_increase
  use -6
  sorry

end temperature_decrease_denotation_l159_159989


namespace simplify_expression_l159_159249

theorem simplify_expression (a b : ℝ) (h_a : a = sqrt 3 - sqrt 11) (h_b : b = sqrt 3 + sqrt 11) :
  (a^2 - b^2) / (a^2 * b - a * b^2) / (1 + (a^2 + b^2) / (2 * a * b)) = sqrt 3 / 6 :=
by
  -- This is a placeholder for the actual proof
  sorry

end simplify_expression_l159_159249


namespace probability_two_red_two_blue_l159_159351

theorem probability_two_red_two_blue (total_red total_blue : ℕ) (red_taken blue_taken selected : ℕ)
  (h_red_total : total_red = 12) (h_blue_total : total_blue = 8) (h_selected : selected = 4)
  (h_red_taken : red_taken = 2) (h_blue_taken : blue_taken = 2) :
  (Nat.choose total_red red_taken) * (Nat.choose total_blue blue_taken) /
    (Nat.choose (total_red + total_blue) selected : ℚ) = 1848 / 4845 := 
by 
  sorry

end probability_two_red_two_blue_l159_159351


namespace sum_of_integer_solutions_l159_159449

theorem sum_of_integer_solutions : 
  let f (x : ℤ) := x^4 - 13 * x^2 + 36 
  in (finset.sum (finset.filter (λ x, f x = 0) (finset.Icc (-3) 3)) id) = 0 :=
begin
  sorry
end

end sum_of_integer_solutions_l159_159449


namespace find_smaller_integer_l159_159670

theorem find_smaller_integer (m : ℕ) (n : ℕ) 
  (hm : m = 60) 
  (hn : 10 ≤ n ∧ n < 100) 
  (h_avg : (m + n) / 2 = 60 + n / 100) : n = 59 :=
by
  sorry

end find_smaller_integer_l159_159670


namespace opposite_of_neg_2_l159_159273

theorem opposite_of_neg_2 : ∃ y : ℝ, -2 + y = 0 ∧ y = 2 := by
  sorry

end opposite_of_neg_2_l159_159273


namespace radius_of_hole_l159_159573

-- Define the dimensions of the rectangular solid
def length1 : ℕ := 3
def length2 : ℕ := 8
def length3 : ℕ := 9

-- Define the radius of the hole
variable (r : ℕ)

-- Condition: The area of the 2 circles removed equals the lateral surface area of the cylinder
axiom area_condition : 2 * Real.pi * r^2 = 2 * Real.pi * r * length1

-- Prove that the radius of the cylindrical hole is 3
theorem radius_of_hole : r = 3 := by
  sorry

end radius_of_hole_l159_159573


namespace problem_correct_statements_eq_zero_l159_159311

theorem problem_correct_statements_eq_zero :
  let s1 := "Very small real numbers can form a set" → false
  let s2 := "The set {y | y = x^2 - 1} and the set {(x, y) | y = x^2 - 1} are the same set" → false
  let s3 := "The set consisting of the numbers 1, 3/2, 6/4, |-1/2|, 0.5 has 5 elements" → false
  let s4 := "The set {(x, y) | xy ≤ 0, x, y ∈ ℝ} refers to the set of points in the second and fourth quadrants" → false
  (s1 ∧ s2 ∧ s3 ∧ s4) → (false = true) :=
by
  intros s1 s2 s3 s4
  sorry

end problem_correct_statements_eq_zero_l159_159311


namespace prove_y_value_l159_159545

theorem prove_y_value (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 :=
by
  sorry

end prove_y_value_l159_159545


namespace tangency_points_form_cyclic_quadrilateral_l159_159064

variables {S1 S2 S3 S4 : Type} [circle S1] [circle S2] [circle S3] [circle S4]

-- Let's define some auxiliary types and conditions
variables (tangency : Π {S T : Type} [circle S] [circle T], Prop)

-- The tangency condition
axiom tangency_S1_S2 : tangency S1 S2
axiom tangency_S2_S3 : tangency S2 S3
axiom tangency_S3_S4 : tangency S3 S4
axiom tangency_S4_S1 : tangency S4 S1

-- Point of tangency exists for externally tangent circles
axiom A1_exists : ∃ A1 : point S1, ∃ A2 : point S2, tangency S1 S2
axiom A2_exists : ∃ A2 : point S2, ∃ A3 : point S3, tangency S2 S3
axiom A3_exists : ∃ A3 : point S3, ∃ A4 : point S4, tangency S3 S4
axiom A4_exists : ∃ A4 : point S4, ∃ A1 : point S1, tangency S4 S1

-- Prove the quadrilateral formed by these tangency points is cyclic
theorem tangency_points_form_cyclic_quadrilateral :
  ∃ A1 A2 A3 A4 : point,
    tangency S1 S2 → tangency S2 S3 → tangency S3 S4 → tangency S4 S1 →
    cyclic_quadrilateral A1 A2 A3 A4 :=
begin
  -- The detailed proof steps are omitted
  sorry
end

end tangency_points_form_cyclic_quadrilateral_l159_159064


namespace area_of_triangle_ABC_l159_159161

theorem area_of_triangle_ABC (a b c : ℝ) (C : ℝ) 
  (h1 : a = 1)
  (h2 : 2 * b - √3 * c = 2 * a * Real.cos C)
  (h3 : Real.sin C = √3 / 2) : 
  ∃ area : ℝ, area = √3 / 2 ∨ area = √3 / 4 := 
by
  sorry

end area_of_triangle_ABC_l159_159161


namespace area_of_triangle_l159_159925

variable {n : ℕ}
noncomputable def a (n : ℕ) : ℝ := 1 / (n * (n + 1))
noncomputable def S (n : ℕ) : ℝ := ∑ i in range n, a (i + 1)

theorem area_of_triangle (h : S 9 = 9 / 10) : 
  let line_eq : (x / (10 : ℝ)) + (y / (9 : ℝ)) = 1 in
  let x_intercept : ℝ := 10 in
  let y_intercept : ℝ := 9 in
  let area : ℝ := 1 / 2 * x_intercept * y_intercept in
  area = 45 := sorry

end area_of_triangle_l159_159925


namespace similarity_condition_l159_159266

-- Define polygons as list of Angles and list of Sides
structure Polygon where
  angles : List ℝ
  sides  : List ℝ

-- Similarity condition: two polygons P and Q are similar if all their corresponding angles are equal and corresponding sides are proportional
def areSimilar (P Q : Polygon) : Prop :=
  (∀ i, i < P.angles.length → P.angles.get i == Q.angles.get i) ∧
  (∃ k, ∀ i, i < P.sides.length → P.sides.get i = k * Q.sides.get i)

-- Problem statement to prove
theorem similarity_condition (P Q : Polygon) :
  (∀ i, i < P.angles.length → P.angles.get i == Q.angles.get i) ∧
  (∃ k, ∀ i, i < P.sides.length → P.sides.get i = k * Q.sides.get i) ↔ areSimilar P Q :=
by
  sorry

end similarity_condition_l159_159266


namespace solve_for_y_l159_159541

variables (x y : ℤ)

theorem solve_for_y (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 :=
by
  hint sorry

end solve_for_y_l159_159541


namespace probability_two_red_two_blue_l159_159356

def total_marbles : ℕ := 20
def red_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def selected_marbles : ℕ := 4

theorem probability_two_red_two_blue :
  (12.choose 2 * 8.choose 2) / (20.choose 4) = 168 / 323 :=
  sorry

end probability_two_red_two_blue_l159_159356


namespace sara_basketball_loss_l159_159247

theorem sara_basketball_loss (total_games : ℕ) (games_won : ℕ) (games_lost : ℕ) 
  (h1 : total_games = 16) 
  (h2 : games_won = 12) 
  (h3 : games_lost = total_games - games_won) : 
  games_lost = 4 :=
by
  sorry

end sara_basketball_loss_l159_159247


namespace P_not_on_circle_line_equations_l159_159077

-- Define the center of the circle and radius
def center : ℝ × ℝ := (-2, 1)
def radius : ℝ := 3

-- Define the point P
def P : ℝ × ℝ := (0, 2)

-- Define the equation to check if P lies on the circle
def circle_equation (x y : ℝ) : ℝ := (x + 2) ^ 2 + (y - 1) ^ 2

-- Prove P does not lie on the circle
theorem P_not_on_circle : ¬(circle_equation 0 2 = radius ^ 2) :=
by
  unfold circle_equation
  unfold radius
  simp
  norm_num
  contradiction
  sorry

-- Define the equations for the lines
def line1 (x : ℝ) : ℝ := 0
def line2 (x y : ℝ) : Prop := 3 * x + 4 * y = 8

-- Prove the equations of the lines
theorem line_equations (k : ℝ) :
  (∃(x y : ℝ), (x, y) = P ∧ line2 x y) ∨ (∃ x, P.fst = line1 x) :=
by
  intros
  sorry

end P_not_on_circle_line_equations_l159_159077


namespace thomas_total_training_hours_l159_159703

-- Define the conditions from the problem statement.
def training_hours_first_15_days : ℕ := 15 * 5
def training_hours_next_15_days : ℕ := (15 - 3) * (4 + 3)
def training_hours_next_12_days : ℕ := (12 - 2) * (4 + 3)

-- Prove that the total training hours equals 229.
theorem thomas_total_training_hours : 
  training_hours_first_15_days + training_hours_next_15_days + training_hours_next_12_days = 229 :=
by
  -- conditions as defined
  let t1 := 15 * 5
  let t2 := (15 - 3) * (4 + 3)
  let t3 := (12 - 2) * (4 + 3)
  show t1 + t2 + t3 = 229
  sorry

end thomas_total_training_hours_l159_159703


namespace range_of_f_l159_159034

def f (x : ℝ) : ℝ := (3 * x + 8) / (x - 4)

theorem range_of_f :
  set.range f = set.Iio 3 ∪ set.Ioi 3 :=
sorry

end range_of_f_l159_159034


namespace compute_expression_l159_159619

noncomputable def roots_of_poly := {p q r s : ℝ // (polynomial.C 1 * (polynomial.X - polynomial.C p) * (polynomial.X - polynomial.C q) * (polynomial.X - polynomial.C r) * (polynomial.X - polynomial.C s) = polynomial.X^4 - 24 * polynomial.X^3 + 50 * polynomial.X^2 - 26 * polynomial.X + 7)}

theorem compute_expression (p q r s : ℝ) (h : polynomial.C 1 * (polynomial.X - polynomial.C p) * (polynomial.X - polynomial.C q) * (polynomial.X - polynomial.C r) * (polynomial.X - polynomial.C s) = polynomial.X^4 - 24 * polynomial.X^3 + 50 * polynomial.X^2 - 26 * polynomial.X + 7) : 
  (p + q)^2 + (q + r)^2 + (r + s)^2 + (s + p)^2 + (p + r)^2 + (q + s)^2 = 1052 :=
sorry

end compute_expression_l159_159619


namespace inequality_bound_l159_159306

theorem inequality_bound (a x : ℝ) (h1 : |a| ≤ 1) (h2 : |x| ≤ 1) :
  |x^2 - ax - a^2| ≤ 5 / 4 :=
sorry

end inequality_bound_l159_159306


namespace minimum_PF_plus_PA_minimum_PB_plus_d_l159_159070

noncomputable def parabola_focus_distance (P : ℝ × ℝ) (F : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - F.1)^2 + P.2^2)

noncomputable def distance_to_directrix (P : ℝ × ℝ) : ℝ :=
  abs (P.1 + 1)

theorem minimum_PF_plus_PA
  (P : ℝ × ℝ)
  (F : ℝ × ℝ := (0, 0.5))
  (A : ℝ × ℝ := (5 / 4, 3 / 4))
  (hP : P.2^2 = P.1)
  : parabola_focus_distance(P, F) + sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) ≥ 3 / 2 :=
sorry

theorem minimum_PB_plus_d
  (P : ℝ × ℝ)
  (F : ℝ × ℝ := (0, 0.5))
  (B : ℝ × ℝ := (1 / 4, 2))
  (hP : P.2^2 = P.1)
  : parabola_focus_distance(P, F) + distance_to_directrix(P) ≥ 2 :=
sorry

end minimum_PF_plus_PA_minimum_PB_plus_d_l159_159070


namespace triangle_area_l159_159181

theorem triangle_area (a b : ℝ) (sinC sinA : ℝ) 
  (h1 : a = Real.sqrt 5) 
  (h2 : b = 3) 
  (h3 : sinC = 2 * sinA) : 
  ∃ (area : ℝ), area = 3 := 
by 
  sorry

end triangle_area_l159_159181


namespace prove_y_value_l159_159548

theorem prove_y_value (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 :=
by
  sorry

end prove_y_value_l159_159548


namespace quadratic_inequality_empty_set_l159_159991

theorem quadratic_inequality_empty_set (a : ℝ) :
  (∀ x : ℝ, ¬ (ax^2 - ax + 1 < 0)) ↔ (0 ≤ a ∧ a ≤ 4) :=
by
  sorry

end quadratic_inequality_empty_set_l159_159991


namespace compare_diff_functions_l159_159957

variable {R : Type*} [LinearOrderedField R]
variable {f g : R → R}
variable (h_fg : ∀ x, f' x > g' x)
variable {x1 x2 : R}

theorem compare_diff_functions (h : x1 < x2) : f x1 - f x2 < g x1 - g x2 :=
  sorry

end compare_diff_functions_l159_159957


namespace necessary_not_sufficient_l159_159599

-- Define the function y = x^2 - 2ax + 1
def quadratic_function (a x : ℝ) : ℝ := x^2 - 2 * a * x + 1

-- Define strict monotonicity on the interval [1, +∞)
def strictly_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f x < f y

-- Define the condition for the function to be strictly increasing on [1, +∞)
def condition_strict_increasing (a : ℝ) : Prop :=
  strictly_increasing_on (quadratic_function a) (Set.Ici 1)

-- The condition to prove
theorem necessary_not_sufficient (a : ℝ) :
  condition_strict_increasing a → (a ≤ 0) := sorry

end necessary_not_sufficient_l159_159599


namespace sum_min_max_values_l159_159210

theorem sum_min_max_values (p q r s : ℝ) 
  (h1 : p + q + r + s = 10) 
  (h2 : p^2 + q^2 + r^2 + s^2 = 20) :
  (let f := 3 * (p^3 + q^3 + r^3 + s^3) - (p^4 + q^4 + r^4 + s^4) in
    let m := 40 in
    let M := 52 in
    m + M = 92) :=
sorry

end sum_min_max_values_l159_159210


namespace max_weighted_sum_l159_159628

theorem max_weighted_sum :
  ∃ (a : ℕ → ℤ), (∑ i in range 24, a i) = 0 ∧ (∀ i, |a i| ≤ i) ∧ 
  (∑ i in range 24, (i + 1) * a (i + 1)) = 1432 := sorry

end max_weighted_sum_l159_159628


namespace complex_modulus_power_l159_159887

theorem complex_modulus_power :
  complex.abs ((2 : ℂ) + (complex.I * real.sqrt 11))^4 = 225 :=
by
  sorry

end complex_modulus_power_l159_159887


namespace partI_minimum_value_f_partII_value_sin_x_l159_159969

-- Definitions of vectors
def vector_m (x : ℝ) : ℝ × ℝ := (Real.cos (x / 2), -1)
def vector_n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (x / 2), Real.cos (x / 2) ^ 2)

-- Definition of function f
def f (x : ℝ) : ℝ := (vector_m x).1 * (vector_n x).1 + (vector_m x).2 * (vector_n x).2 + 1

-- Part (I): Minimum value of f(x) for x in [π/2, π]
theorem partI_minimum_value_f {x : ℝ} (hx : x ∈ Set.Icc (Real.pi / 2) Real.pi) :
  ∃ (x : ℝ), f x = 1 :=
sorry

-- Part (II): If f(x) = 11/10, find the value of sin x for x in [0, π/2]
theorem partII_value_sin_x {x : ℝ} (hx : x ∈ Set.Icc 0 (Real.pi / 2)) (hf : f x = 11 / 10) :
  Real.sin x = (3 * Real.sqrt 3 + 4) / 10 :=
sorry

end partI_minimum_value_f_partII_value_sin_x_l159_159969


namespace intersection_point_lines_AB_CD_l159_159171

def A := (8, -6, 5 : ℝ × ℝ × ℝ)
def B := (18, -16, 10 : ℝ × ℝ × ℝ)
def C := (-4, 6, -12 : ℝ × ℝ × ℝ)
def D := (4, -4, 8 : ℝ × ℝ × ℝ)

noncomputable def parametrizeLineAB (t : ℝ) : ℝ × ℝ × ℝ := 
  (8 + 10 * t, -6 - 10 * t, 5 + 5 * t)

noncomputable def parametrizeLineCD (s : ℝ) : ℝ × ℝ × ℝ := 
  (-4 + 8 * s, 6 - 10 * s, -12 + 20 * s)

theorem intersection_point_lines_AB_CD : 
  ∃ t s : ℝ, parametrizeLineAB t = parametrizeLineCD s ∧ parametrizeLineAB t = (20, -18, 11) := 
by
  sorry

end intersection_point_lines_AB_CD_l159_159171


namespace vasya_birthday_l159_159710

def DayOfWeek := 
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

open DayOfWeek

def day_after (d : DayOfWeek) : DayOfWeek :=
match d with
| Sunday => Monday
| Monday => Tuesday
| Tuesday => Wednesday
| Wednesday => Thursday
| Thursday => Friday
| Friday => Saturday
| Saturday => Sunday

def day_before (d : DayOfWeek) : DayOfWeek :=
match d with
| Sunday => Saturday
| Monday => Sunday
| Tuesday => Monday
| Wednesday => Tuesday
| Thursday => Wednesday
| Friday => Thursday
| Saturday => Friday

theorem vasya_birthday (day_said : DayOfWeek) 
    (h1 : day_after (day_after day_said) = Sunday) 
    (h2 : day_said = day_after VasyaBirthday) 
    : VasyaBirthday = Thursday :=
sorry

end vasya_birthday_l159_159710


namespace cyclic_pentagon_circumradius_l159_159869

theorem cyclic_pentagon_circumradius :
  ∀ (A B C D E : Point) (AB BC CD DE AE : ℝ),
    AB = 5 → BC = 5 → CD = 12 → DE = 12 → AE = 14 →
    cyclic A B C D E →
    circumradius A B C D E = 13 :=
by
  intros A B C D E AB BC CD DE AE
  intro h1 h2 h3 h4 h5 hcyclic
  sorry

end cyclic_pentagon_circumradius_l159_159869


namespace false_proposition_l159_159394

-- Definitions for the conditions
def quadrilateral (A B C D : Type) := sorry
def angle_equal (Q : Type) := sorry
def diagonals_bisect_at_right_angle (Q : Type) := sorry
def diagonals_equal (Q : Type) := sorry
def rhombus (Q : Type) := sorry
def square (Q : Type) := sorry
def parallelogram (Q : Type) := sorry
def rectangle (Q : Type) := sorry

-- The false proposition among the given conditions
theorem false_proposition :
  ∀ (Q : Type),
    (quadrilateral Q → angle_equal Q → rectangle Q) ∧
    (quadrilateral Q → diagonals_bisect_at_right_angle Q → rhombus Q) ∧
    (quadrilateral Q → diagonals_bisect_at_right_angle Q → diagonals_equal Q → ¬ square Q) ∧
    (parallelogram Q → diagonals_equal Q → rectangle Q) :=
begin
  intros,
  split,
  { intros, sorry }, -- Proof for condition 1
  split,
  { intros, sorry }, -- Proof for condition 2
  split,
  { intros, exact h3 }, -- Proof for condition 3 (false proposition)
  { intros, sorry }  -- Proof for condition 4
end

end false_proposition_l159_159394


namespace triangle_problem_l159_159566

noncomputable def length_of_side_c (a : ℝ) (cosB : ℝ) (C : ℝ) : ℝ :=
  a * (Real.sqrt 2 / 2) / (Real.sqrt (1 - cosB^2))

noncomputable def cos_A_minus_pi_over_6 (cosB : ℝ) (cosA : ℝ) (sinA : ℝ) : ℝ :=
  cosA * (Real.sqrt 3 / 2) + sinA * (1 / 2)

theorem triangle_problem (a : ℝ) (cosB : ℝ) (C : ℝ) 
  (ha : a = 6) (hcosB : cosB = 4/5) (hC : C = Real.pi / 4) : 
  (length_of_side_c a cosB C = 5 * Real.sqrt 2) ∧ 
  (cos_A_minus_pi_over_6 cosB (- (cosB * (Real.sqrt 2 / 2) - (Real.sqrt (1 - cosB^2) * (Real.sqrt 2 / 2)))) (Real.sqrt (1 - (- (cosB * (Real.sqrt 2 / 2) - (Real.sqrt (1 - cosB^2) * (Real.sqrt 2 / 2))))^2)) = (7 * Real.sqrt 2 - Real.sqrt 6) / 20) :=
by 
  sorry

end triangle_problem_l159_159566


namespace fibonacci_polynomial_property_l159_159191

-- Define the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n+2) => fibonacci n + fibonacci (n+1)

-- Define the polynomial P(x) of degree 990
noncomputable def P : ℕ → ℕ :=
  sorry  -- To be defined as a polynomial with specified properties

-- Statement of the problem (theorem)
theorem fibonacci_polynomial_property (P : ℕ → ℕ) (hP : ∀ k, 992 ≤ k → k ≤ 1982 → P k = fibonacci k) :
  P 1983 = fibonacci 1983 - 1 :=
sorry  -- Proof omitted

end fibonacci_polynomial_property_l159_159191


namespace john_money_left_l159_159188

def cost_of_drink (q : ℝ) : ℝ := q
def cost_of_small_pizza (q : ℝ) : ℝ := cost_of_drink q
def cost_of_large_pizza (q : ℝ) : ℝ := 4 * cost_of_drink q
def total_cost (q : ℝ) : ℝ := 2 * cost_of_drink q + 2 * cost_of_small_pizza q + cost_of_large_pizza q
def initial_money : ℝ := 50
def remaining_money (q : ℝ) : ℝ := initial_money - total_cost q

theorem john_money_left (q : ℝ) : remaining_money q = 50 - 8 * q :=
by
  sorry

end john_money_left_l159_159188


namespace math_problem_l159_159802

theorem math_problem (a b : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : a ≠ b) :
  (2 * (∑ i in Finset.range (a + 2), ite (i ≥ a ∧ i ≤ 2 * a) i 0)) / (a^2 + 3 * a + 2) +
  (6 * (real.sqrt a + real.sqrt b) / ((a - b)^0.6 * (a + 2))) /
  (real.div (real.sqrt a - real.sqrt b) ((a - b)^(-2 / 5))) = 3 :=
by 
  sorry

end math_problem_l159_159802


namespace daps_equivalent_to_dips_l159_159104

-- Define the units as types
def dap : Type := ℕ
def dop : Type := ℕ
def dip : Type := ℕ

-- Define the conditions
def daps_to_dops (d : ℕ) : ℝ := (5 * d) / 4
def dops_to_dips (d : ℕ) : ℝ := (3 * d) / 9

-- The theorem to prove that 54 dips are equivalent to 22.5 daps
theorem daps_equivalent_to_dips {d : ℕ} (h₁ : daps_to_dops d = d) (h₂ : dops_to_dips d = d) : 
  54 * ((5 : ℝ) / 12) = 22.5 :=
by
  sorry

end daps_equivalent_to_dips_l159_159104


namespace probability_two_red_two_blue_l159_159353

def total_marbles : ℕ := 20
def red_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def selected_marbles : ℕ := 4

theorem probability_two_red_two_blue :
  (12.choose 2 * 8.choose 2) / (20.choose 4) = 168 / 323 :=
  sorry

end probability_two_red_two_blue_l159_159353


namespace Albert_cabbage_count_l159_159835

-- Define the conditions
def rows := 12
def heads_per_row := 15

-- State the theorem
theorem Albert_cabbage_count : rows * heads_per_row = 180 := 
by sorry

end Albert_cabbage_count_l159_159835


namespace Melissa_initial_bananas_l159_159638

theorem Melissa_initial_bananas (bananas_shared bananas_left : ℕ) (h1 : bananas_shared = 4) (h2 : bananas_left = 84) : 
  bananas_shared + bananas_left = 88 :=
by
  rw [h1, h2]
  exact rfl

end Melissa_initial_bananas_l159_159638


namespace product_factors_1_to_12_l159_159882

theorem product_factors_1_to_12 :
  (∏ n in Finset.range 10, (1 - 1 / (n + 3))) = 1 / 6 :=
by
  sorry

end product_factors_1_to_12_l159_159882


namespace multiply_two_digit_by_nine_l159_159289

theorem multiply_two_digit_by_nine (A B : ℕ) (hA : A < 10) (hB : B < 10) :
  let n := 10 * A + B in
  ((n - (A + 1)) * 10 + (10 - B) = 90 * A + 9 * B) :=
by
  intros
  let n := 10 * A + B
  have h1: (n - (A + 1)) = 9 * A + (B - 1), sorry     -- simplify the subtraction
  have h2: (h1 * 10 + (10 - B)) = (90 * A + 9 * B), sorry -- combine results
  sorry

end multiply_two_digit_by_nine_l159_159289


namespace probability_two_red_two_blue_l159_159349

theorem probability_two_red_two_blue (total_red total_blue : ℕ) (red_taken blue_taken selected : ℕ)
  (h_red_total : total_red = 12) (h_blue_total : total_blue = 8) (h_selected : selected = 4)
  (h_red_taken : red_taken = 2) (h_blue_taken : blue_taken = 2) :
  (Nat.choose total_red red_taken) * (Nat.choose total_blue blue_taken) /
    (Nat.choose (total_red + total_blue) selected : ℚ) = 1848 / 4845 := 
by 
  sorry

end probability_two_red_two_blue_l159_159349


namespace smallest_m_plus_n_l159_159793

theorem smallest_m_plus_n (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_lt : m < n)
    (h_eq : 1978^m % 1000 = 1978^n % 1000) : m + n = 26 :=
sorry

end smallest_m_plus_n_l159_159793


namespace largest_y_on_ellipse_l159_159865

theorem largest_y_on_ellipse :
  ∀ x y : ℝ, (x-3)^2 / 49 + (y-4)^2 / 25 = 1 → y ≤ 9 :=
begin
  sorry
end

end largest_y_on_ellipse_l159_159865


namespace ellipse_standard_equation_range_op_oq_dot_product_l159_159947

theorem ellipse_standard_equation 
  (a b c : ℝ) (a_positive : a > 0) (b_positive : b > 0)
  (a_greater_b : a > b) (passes_through_point : ∃ y, (0, y) ∈ {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1 ∧ y = sqrt 2}) 
  (eccentricity : c / a = sqrt 6 / 3) (focal_relation : a^2 - b^2 = c^2) :
  ∀ x y : ℝ, x^2 / 6 + y^2 / 2 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1} :=
sorry

theorem range_op_oq_dot_product 
  (a b c k : ℝ) (a_positive : a > 0) (b_positive : b > 0)
  (a_greater_b : a > b) (passes_through_point : (0, sqrt 2) ∈ {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}) 
  (eccentricity : c / a = sqrt 6 / 3) (focal_relation : a^2 - b^2 = c^2) :
  ∀ l : ℝ, let P := (x, y) where (x, y) ∈ (line_passing_left_focus : ℝ → ℝ → Prop) in 
  let Q := (x', y') where (x', y') ∈ (line_passing_left_focus : ℝ → ℝ → Prop) in 
  ∃ (range : ℝ), (-6 ≤ range ∧ range < 10 / 3) ↔ (∃ (x1 x2 y1 y2 : ℝ), (x1 + x2 = -12 * k^2 / (1 + 3 * k^2) ∧ x1 * x2 = (12 * k^2 - 6) / (1 + 3 * k^2) ∧ range = x1 * x2 + y1 * y2)) :=
sorry

end ellipse_standard_equation_range_op_oq_dot_product_l159_159947


namespace different_gcd_values_count_l159_159770

theorem different_gcd_values_count :
  let gcd_lcm_eq_prod (a b : ℕ) := Nat.gcd a b * Nat.lcm a b = a * b
  let prime_factors_360 := (2 ^ 3 * 3 ^ 2 * 5 ^ 1 : ℕ)
  (∃ a b : ℕ, gcd_lcm_eq_prod a b ∧ a * b = 360) →
  (∃ gcd_vals : Finset ℕ, gcd_vals = {1, 2, 3, 4, 6, 8, 12, 24} ∧ gcd_vals.card = 8) :=
begin
  sorry
end

end different_gcd_values_count_l159_159770


namespace result_of_y_minus_3x_l159_159993

theorem result_of_y_minus_3x (x y : ℝ) (h1 : x + y = 8) (h2 : y - x = 7.5) : y - 3 * x = 7 :=
sorry

end result_of_y_minus_3x_l159_159993


namespace saved_money_is_30_l159_159857

def week_payout : ℕ := 5 * 3
def total_payout (weeks: ℕ) : ℕ := weeks * week_payout
def shoes_cost : ℕ := 120
def remaining_weeks : ℕ := 6
def remaining_earnings : ℕ := total_payout remaining_weeks
def saved_money : ℕ := shoes_cost - remaining_earnings

theorem saved_money_is_30 : saved_money = 30 := by
  -- Proof steps go here
  sorry

end saved_money_is_30_l159_159857


namespace total_time_eq_l159_159051

def gina_time : ℝ := 3
def tom_time : ℝ := 5
def together_time : ℝ := 2

theorem total_time_eq : 
  let gina_rate := 1 / gina_time in
  let tom_rate := 1 / tom_time in
  let combined_rate := gina_rate + tom_rate in
  let work_done_together := combined_rate * together_time in
  let remaining_work := 1 - work_done_together in
  let tom_remaining_time := remaining_work / tom_rate in
  let total_time := together_time + tom_remaining_time in
  total_time = 20 / 3 := 
by sorry

end total_time_eq_l159_159051


namespace finite_group_decomposition_l159_159044

theorem finite_group_decomposition (n : ℕ) (hn : n ≥ 3) :
  ∃ A B : set (zmod n), A ≠ ∅ ∧ B ≠ ∅ ∧
  A ≠ set.univ ∧ B ≠ set.univ ∧ 
  (A ∩ B).card = 1 ∧ 
  ∀ x : zmod n, ∃ a ∈ A, ∃ b ∈ B, a + b = x := 
sorry

end finite_group_decomposition_l159_159044


namespace distance_of_intersections_l159_159086

noncomputable def c1_parametric (t : ℝ) : ℝ × ℝ :=
  ( (real.sqrt 5) / 5 * t, (2 * real.sqrt 5) / 5 * t - 1 )

noncomputable def c2_polar (ρ θ : ℝ) : Prop :=
  ρ = 2 * real.cos θ - 4 * real.sin θ

noncomputable def c1 (x y : ℝ) : Prop :=
  y = 2 * x - 1

noncomputable def c2 (x y : ℝ) : Prop :=
  x^2 + y^2 = 2 * x - 4 * y

theorem distance_of_intersections :
  let x1 := (1 - real.sqrt 5) / 5
      y1 := 2 * x1 - 1
      x2 := (1 + real.sqrt 5) / 5
      y2 := 2 * x2 - 1
      d := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  in d = 2 * real.sqrt 2 :=
by sorry

end distance_of_intersections_l159_159086


namespace max_min_difference_adjacent_circular_l159_159502

theorem max_min_difference_adjacent_circular (xs : List ℕ) 
  (h : xs = [10, 6, 13, 4, 18]) :
  ∃ ys : List ℕ, (ys.perm xs ∧ (minimum_difference ys ys.head (ys.last ys)) = 9) := sorry

  def minimum_difference : List ℕ → ℕ → ℕ → ℕ := sorry

end max_min_difference_adjacent_circular_l159_159502


namespace different_gcd_values_count_l159_159769

theorem different_gcd_values_count :
  let gcd_lcm_eq_prod (a b : ℕ) := Nat.gcd a b * Nat.lcm a b = a * b
  let prime_factors_360 := (2 ^ 3 * 3 ^ 2 * 5 ^ 1 : ℕ)
  (∃ a b : ℕ, gcd_lcm_eq_prod a b ∧ a * b = 360) →
  (∃ gcd_vals : Finset ℕ, gcd_vals = {1, 2, 3, 4, 6, 8, 12, 24} ∧ gcd_vals.card = 8) :=
begin
  sorry
end

end different_gcd_values_count_l159_159769


namespace month_starts_on_friday_l159_159996

theorem month_starts_on_friday :
  (∃ days_in_month: ℕ, days_in_month ≤ 31 ∧
  (∃ weeks: ℕ, ∃ extra_days: ℕ, days_in_month = 7 * weeks + extra_days ∧ 
  weeks = 4 ∧ extra_days = 3 ∧ 
  (∃ fridays: ℕ, ∃ saturdays: ℕ, ∃ sundays: ℕ, 
  fridays = 5 ∧ saturdays = 5 ∧ sundays = 5)
)) → 
  (∃ start_day: string, start_day = "Friday") :=
by
  sorry

end month_starts_on_friday_l159_159996


namespace cross_section_area_l159_159057

-- Definitions for the conditions given in the problem

-- Cube with edge length 2
def edge_length (c : Cube) : ℝ := 2

-- Sphere tangent to each face of the cube
def tangent_sphere (c : Cube) (s : Sphere) : Prop :=
  ∀ face : Face, is_tangent s face

-- Plane slicing through the sphere
def slicing_plane (p : Plane) (c : Cube) : Prop :=
  passes_through_points p [c.A, c.C, c.B₁]

-- Proof that the area of the cross-section obtained by slicing the sphere with the plane ACB₁ is 2π/3
theorem cross_section_area (c : Cube) (s : Sphere) (p : Plane)
  (h1 : edge_length c = 2) 
  (h2 : tangent_sphere c s)
  (h3 : slicing_plane p c) :
  cross_section_area s p = (2 * π) / 3 :=
sorry

end cross_section_area_l159_159057


namespace sequence_perfect_square_l159_159693

-- Define the sequence a_n based on given recurrence relation
def a : ℕ → ℤ
| 0     := 0    -- Since we are defining from n ≥ 1, a_0 is not part of the sequence but let's give it a value.
| 1     := 1
| 2     := 12
| 3     := 20
| (n+4) := 2 * a (n+3) + 2 * a (n+2) - a (n+1)

-- Theorem stating that 1 + 4 * a_n * a_{n+1} is always a perfect square
theorem sequence_perfect_square (n : ℕ) : ∃ k : ℤ, 1 + 4 * (a n) * (a (n+1)) = k^2 :=
sorry

end sequence_perfect_square_l159_159693


namespace geometric_seq_and_general_formula_lambda_range_l159_159089

noncomputable def a_seq : ℕ → ℝ
| 0 => 1
| n + 1 => a_seq n / (a_seq n + 3)

theorem geometric_seq_and_general_formula :
  (∀ n : ℕ, n > 0 → (1 / a_seq n + 1 / 2) = (3 / 2) * (3 ^ (n - 1))) →
  (∀ n : ℕ, n > 0 → a_seq n = 2 / (3 ^ n - 1)) :=
sorry

noncomputable def b_seq (n : ℕ) : ℝ :=
  (3^n - 1) * n / 2^n * a_seq n

noncomputable def T_seq (n : ℕ) : ℝ :=
  ∑ k in finset.range (n + 1), b_seq k

theorem lambda_range :
  (∀ n : ℕ, n > 0 → (-1)^n * λ < T_seq n + (n / 2^(n-1))) →
  (-2 < λ ∧ λ < 3) :=
sorry

end geometric_seq_and_general_formula_lambda_range_l159_159089


namespace exists_infinite_sequence_l159_159656

/-- Sequence that satisfies the required properties. -/
def sequence (a : ℕ → ℕ) : Prop :=
  a 0 = 3 ∧ a 1 = 4 ∧ ∀ n ≥ 1, a (n + 1) = (a n) * (a n + 2) / 2

/-- Theorem statement for the existence of such a sequence. -/
theorem exists_infinite_sequence :
  ∃ (a : ℕ → ℕ), sequence a ∧ ∀ N, ∃ k : ℤ, (∑ i in Finset.range (N + 1), (a i)^2) = k^2 :=
begin
  sorry
end

end exists_infinite_sequence_l159_159656


namespace solve_arctan_eq_l159_159256

theorem solve_arctan_eq (y : ℝ) (h : arctan (1 / y) + arctan (1 / y^2) = π / 4) : y = 2 := 
by
  sorry

end solve_arctan_eq_l159_159256


namespace gcd_values_360_l159_159778

theorem gcd_values_360 : ∃ d : ℕ, d = 11 ∧ ∀ a b : ℕ, a * b = 360 → ∃ (g : ℕ), g = gcd a b ∧ finite {g | g = gcd a b ∧ a * b = 360} ∧ card {g | g = gcd a b ∧ a * b = 360} = 11 :=
sorry

end gcd_values_360_l159_159778


namespace handshake_count_l159_159846

-- Definitions of the conditions based on the problem statement
def total_people := 40
def group1 := 25
def group2 := 15
def sub_group2a := 5
def sub_group2b := 10

-- Theorem statement proving that the number of handshakes is 305 given the conditions
theorem handshake_count :
  let people := total_people,
      g1 := group1,
      g2 := group2,
      sg2a := sub_group2a,
      sg2b := sub_group2b,
      n1 := 39,     -- number of handshakes for 10 people who don't know anyone
      n2 := 25,     -- number of handshakes for 5 people who know 10 from the first group
      total := (sg2b * n1) + (sg2a * n2),
      handshake_total := total / 2
  in handshake_total = 305 := 
begin
  sorry
end

end handshake_count_l159_159846


namespace smallest_possible_value_of_AC_l159_159864

theorem smallest_possible_value_of_AC :
  ∃ (AC CD : ℤ), (∃ BD : ℤ, BD^2 = 85 ∧ BD ^ 2 = 85 ) ∧ AC = ((CD ^ 2 + 85) / (2 * CD)) ∧
            (AC = 11 ∧ ∀ (AC' CD' : ℤ), 
              (∃ BD' : ℤ, BD'^2 = 85 ∧ BD' ^ 2 = 85 ) ∧ AC' = ((CD' ^ 2 + 85) / (2 * CD')) → AC' ≥ 11)
  :=
begin
  sorry,
end

end smallest_possible_value_of_AC_l159_159864


namespace num_super_balanced_l159_159369

def is_super_balanced (n : ℕ) : Prop :=
  1000 <= n ∧ n <= 9999 ∧
  let d1 := n / 1000 in
  let d2 := (n / 100) % 10 in
  let d3 := (n / 10) % 10 in
  let d4 := n % 10 in
  let sum_left := d1 + d2 in
  let sum_right := d3 + d4 in
  1 <= sum_left ∧ sum_left <= 12 ∧ sum_left = sum_right

theorem num_super_balanced : ∃ n : ℕ, n = 494 ∧ ∀ m : ℕ, is_super_balanced m → m = n :=
by
  sorry

end num_super_balanced_l159_159369


namespace no_m_for_necessary_and_sufficient_condition_m_geq_3_for_necessary_condition_l159_159907

def P (x : ℝ) : Prop := x ^ 2 - 8 * x - 20 ≤ 0
def S (x : ℝ) (m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

theorem no_m_for_necessary_and_sufficient_condition :
  ¬ ∃ m : ℝ, ∀ x : ℝ, P x ↔ S x m :=
by sorry

theorem m_geq_3_for_necessary_condition :
  ∃ m : ℝ, (m ≥ 3) ∧ ∀ x : ℝ, S x m → P x :=
by sorry

end no_m_for_necessary_and_sufficient_condition_m_geq_3_for_necessary_condition_l159_159907


namespace B_investment_time_proof_l159_159819

def months_invested_by_B (profit_A profit_B total_investment_A total_investment_B total_profit : ℝ) : ℝ :=
  let total_time_A := 12
  let investment_time_A := total_investment_A * total_time_A
  let investment_time_B := total_investment_B * (12 - profit_A / (profit_B / (investment_time_A / (profit_B * (12 - (investment_time_B / total_investment_B))))))
  solve (total_profit = profit_A + profit_B) 
  solve (profit_A / profit_B = investment_time_A / investment_time_B)

theorem B_investment_time_proof
  (total_investment_A total_time_A total_investment_B total_profit profit_A : ℝ)
  (profit_B := total_profit - profit_A) 
  (investment_time_A := total_investment_A * total_time_A)
  (investment_time_B := total_investment_B * (12 - profit_A / (profit_B / (investment_time_A / (profit_B * (12 - (investment_time_B / total_investment_B))))))
  (total_time : ℝ := 12) :
  months_invested_by_B profit_A profit_B total_investment_A total_investment_B total_profit = 6 :=
by
  sorry

end B_investment_time_proof_l159_159819


namespace pyramid_volume_l159_159078

noncomputable def volume_of_pyramid (A1B1 : ℝ) (height : ℝ) : ℝ :=
  let base_area := A1B1 ^ 2
  in (1 / 3) * base_area * height

theorem pyramid_volume :
  let A1B1 := 4
  let height := 6
  volume_of_pyramid A1B1 height = 56 := 
by
  sorry

end pyramid_volume_l159_159078


namespace Richard_remaining_distance_l159_159652

theorem Richard_remaining_distance
  (total_distance : ℕ)
  (day1_distance : ℕ)
  (day2_distance : ℕ)
  (day3_distance : ℕ)
  (half_and_subtract : day2_distance = (day1_distance / 2) - 6)
  (total_distance_to_walk : total_distance = 70)
  (distance_day1 : day1_distance = 20)
  (distance_day3 : day3_distance = 10)
  : total_distance - (day1_distance + day2_distance + day3_distance) = 36 :=
  sorry

end Richard_remaining_distance_l159_159652


namespace acute_angle_formed_by_line_l159_159080

noncomputable def i_2017 : ℂ := complex.I ^ 2017
noncomputable def complex_number : ℂ := complex.sqrt 3 - i_2017
noncomputable def point_A : ℂ × ℂ := (complex.re complex_number, complex.im complex_number)
noncomputable def theta : ℝ := real.arctan (complex.im complex_number / complex.re complex_number)

theorem acute_angle_formed_by_line :
  θ = (5 * ℝ.pi) / 6 := 
sorry

end acute_angle_formed_by_line_l159_159080


namespace triangle_side_lengths_l159_159522

theorem triangle_side_lengths (x : ℤ) : 3 < x ∧ x < 11 → ∃ n : ℕ, n = 7 :=
by
  intro h
  use 7
  sorry

end triangle_side_lengths_l159_159522


namespace whole_milk_fat_percentage_l159_159358

def fat_in_some_milk : ℝ := 4
def percentage_less : ℝ := 0.5

theorem whole_milk_fat_percentage : ∃ (x : ℝ), fat_in_some_milk = percentage_less * x ∧ x = 8 :=
sorry

end whole_milk_fat_percentage_l159_159358


namespace problem_statement_l159_159936

noncomputable def point := ℝ × ℝ

def is_intersection_point (M : point) : Prop :=
  let x := M.1
  let y := M.2
  3 * x - y + 7 = 0 ∧ 2 * x + y + 3 = 0

def is_symmetric_point (M P : point) : Prop :=
  M.1 = P.1 ∧ M.2 = -P.2

def is_on_line (P : point) (l : ℝ → ℝ) : Prop :=
  P.2 = l P.1

def line_reflection (N P : point) (k : ℝ) : Prop :=
  let slope := k
  let reflected_slope := -slope
  P.2 = reflected_slope * (P.1 - N.1) + N.2

def distance_between_lines (l1 l2 : point → ℝ) (d: ℝ) : Prop :=
  let b1 := l1 (0, 1) - l1 (0, 0)
  let b2 := l2 (0, 1) - l2 (0, 0)
  abs(b1 - b2) / sqrt((l1 (0, 1) - l1 (0, 0))^2 + 1) = d

theorem problem_statement :
  let M : point := (-2, 1)
  let P : point := (-2, -1)
  let l3 := λ x : ℝ , (1 / 3) * x - 1 / 3
  let l_parallel1 := λ x : ℝ , (1 / 3) * x + 3
  let l_parallel2 := λ x : ℝ , (1 / 3) * x - 11 / 3
  is_intersection_point M ∧
  is_symmetric_point M P ∧
  is_on_line (1, 0) l3 ∧
  line_reflection (1, 0) P (-1 / 3) ∧
  distance_between_lines l3 l_parallel1 (sqrt 10) ∧
  distance_between_lines l3 l_parallel2 (sqrt 10) 
:= 
  sorry

end problem_statement_l159_159936


namespace max_value_of_expression_l159_159215

theorem max_value_of_expression (a b c : ℝ) (h1: 0 ≤ a) (h2: 0 ≤ b) (h3: 0 ≤ c) 
    (h_sum: a + b + c = 3) :
    (ab / (a + b) + ac / (a + c) + bc / (b + c) ≤ 3 / 2) :=
by
  sorry

end max_value_of_expression_l159_159215


namespace linear_local_mapping_form_l159_159605

theorem linear_local_mapping_form (T : C(ℝ) → C(ℝ)) 
  (h_lin : ∀ (c1 c2 : ℝ) (ψ1 ψ2 : C(ℝ)), T (c1 • ψ1 + c2 • ψ2) = c1 • (T ψ1) + c2 • (T ψ2)) 
  (h_loc : ∀ (ψ1 ψ2 : C(ℝ)) (I : set ℝ) (hI : ∀ x ∈ I, ψ1 x = ψ2 x), ∀ x ∈ I, (T ψ1) x = (T ψ2) x) :
  ∃ (f : ℝ → ℝ), continuous f ∧ ∀ (ψ : C(ℝ)) (x : ℝ), (T ψ) x = f x * ψ x :=
sorry

end linear_local_mapping_form_l159_159605


namespace problem_statement_l159_159512

variables {α β : Type*}
variables (m n : line) 
variables (α β : plane)

-- Definitions of propositions as conditions
def p : Prop := (α ∩ β = m) ∧ (m ⊥ n) → (n ⊥ α)
def q : Prop := (m ∥ α) ∧ (m ⊂ β) ∧ (α ∩ β = n) → (m ∥ n)

-- The final goal to prove proposition
theorem problem_statement (hαβ : α ≠ β) (hmn : m ≠ n) : ¬p ∧ q :=
sorry

end problem_statement_l159_159512


namespace trigonometric_sign_incorrect_l159_159424

theorem trigonometric_sign_incorrect:
  (165 < 180 ∧ 165 > 90 ∧ sin (165 * π / 180) > 0) ∧
  (280 < 360 ∧ 280 > 270 ∧ cos (280 * π / 180) > 0) ∧
  (170 < 180 ∧ 170 > 90 ∧ ¬ (tan (170 * π / 180) > 0)) ∧
  (310 < 360 ∧ 310 > 270 ∧ tan (310 * π / 180) < 0) →
  ¬ (tan (170 * π / 180) > 0) := 
by 
  sorry

end trigonometric_sign_incorrect_l159_159424


namespace vovochkas_no_carry_pairs_eq_correct_pairs_vovochkas_smallest_difference_l159_159142

theorem vovochkas_no_carry_pairs_eq_correct_pairs :
  let valid_digit_pairs := Nat.range 10.map (λ n, 10 - n)
  let sum_valid_digit_pairs := valid_digit_pairs.sum
  sum_valid_digit_pairs = 55 →
  let no_carry_combinations := 81 * 55 * 55
  no_carry_combinations = 244620 :=
by
  let valid_digit_pairs := List.map (λ n, 10 - n) (List.range 10)
  have h_valid_digit_sum : valid_digit_pairs.sum = 55 := by sorry
  let no_carry_combinations := 81 * 55 * 55
  have h_no_carry : no_carry_combinations = 244620 := by sorry
  exact h_no_carry

theorem vovochkas_smallest_difference :
  let incorrect_cases := [1800]
  incorrect_cases.minimum = 1800 :=
by
  let differences := [900, 90, 990]
  have h_min_diff : List.minimum differences = some 90 := by sorry
  let incorrect_cases := List.map (λ diff, 20 * diff) differences
  have h_min_incorrect : incorrect_cases.minimum = some 1800 := by sorry
  exact h_min_incorrect

end vovochkas_no_carry_pairs_eq_correct_pairs_vovochkas_smallest_difference_l159_159142


namespace find_pairs_l159_159113

theorem find_pairs (a b : ℕ) :
  (1111 * a) % (11 * b) = 11 * (a - b) →
  140 ≤ (1111 * a) / (11 * b) ∧ (1111 * a) / (11 * b) ≤ 160 →
  (a, b) = (3, 2) ∨ (a, b) = (6, 4) ∨ (a, b) = (7, 5) ∨ (a, b) = (9, 6) :=
by
  sorry

end find_pairs_l159_159113


namespace multiple_of_four_and_six_prime_sum_even_l159_159664

theorem multiple_of_four_and_six_prime_sum_even {a b : ℤ} 
  (h_a : ∃ m : ℤ, a = 4 * m) 
  (h_b1 : ∃ n : ℤ, b = 6 * n) 
  (h_b2 : Prime b) : 
  Even (a + b) := 
  by sorry

end multiple_of_four_and_six_prime_sum_even_l159_159664


namespace sams_seashells_l159_159244

theorem sams_seashells (mary_seashells : ℕ) (total_seashells : ℕ) (h_mary : mary_seashells = 47) (h_total : total_seashells = 65) : (total_seashells - mary_seashells) = 18 :=
by
  simp [h_mary, h_total]
  sorry

end sams_seashells_l159_159244


namespace sum_cubes_eq_neg_27_l159_159201

theorem sum_cubes_eq_neg_27 (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
 (h_eq : (a^3 + 9) / a = (b^3 + 9) / b ∧ (b^3 + 9) / b = (c^3 + 9) / c) :
 a^3 + b^3 + c^3 = -27 :=
by
  sorry

end sum_cubes_eq_neg_27_l159_159201


namespace projectile_reaches_75_feet_l159_159825

def projectile_height (t : ℝ) : ℝ := -16 * t^2 + 80 * t

theorem projectile_reaches_75_feet :
  ∃ t : ℝ, projectile_height t = 75 ∧ t = 1.25 :=
by
  -- Skipping the proof as instructed
  sorry

end projectile_reaches_75_feet_l159_159825


namespace line_perpendicular_to_plane_l159_159198

noncomputable theory

variables {Point Line Plane : Type}
variables (a b : Line) (α β : Plane)
variable (is_parallel : ∀ {l₁ l₂ : Line}, Prop)
variable (is_perpendicular : ∀ {l : Line} {p : Plane}, Prop)

-- Conditions
variable (a_parallel_b : is_parallel a b)
variable (b_perpendicular_alpha : is_perpendicular b α)

-- Problem statement
theorem line_perpendicular_to_plane :
  is_perpendicular a α :=
sorry

end line_perpendicular_to_plane_l159_159198


namespace evaluate_fff2_l159_159218

def f (x : ℝ) : ℝ :=
  if x > 9 then real.sqrt x else x^2

theorem evaluate_fff2 : f (f (f 2)) = 4 :=
by {
  -- Proof omitted
  sorry
}

end evaluate_fff2_l159_159218


namespace sum_of_cubes_eq_neg_27_l159_159202

theorem sum_of_cubes_eq_neg_27
  (a b c : ℝ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_condition : (a^3 + 9) / a = (b^3 + 9) / b ∧ (b^3 + 9) / b = (c^3 + 9) / c) :
  a^3 + b^3 + c^3 = -27 :=
sorry

end sum_of_cubes_eq_neg_27_l159_159202


namespace harmonic_mean_problem_l159_159076

-- Conditions
def S (n : ℕ) : ℕ := 2 * n^2 + n
def a (n : ℕ) : ℕ := if n = 1 then 3 else S(n) - S(n - 1)
def b (n : ℕ) : ℕ := (a(n) + 1) / 4

-- Theorem
theorem harmonic_mean_problem :
  (∑ i in Finset.range (2017), 1 / (b i * b (i + 1))) = 2016 / 2017 :=
by
  sorry

end harmonic_mean_problem_l159_159076


namespace equilateral_triangle_area_relationship_l159_159240

theorem equilateral_triangle_area_relationship :
  let W := (sqrt 3 / 4) * 6^2
  let X := (sqrt 3 / 4) * 8^2
  let Y := (sqrt 3 / 4) * 10^2
  W + X = Y :=
by
  sorry

end equilateral_triangle_area_relationship_l159_159240


namespace range_of_t_l159_159055

def f (x : ℝ) : ℝ := abs (x * exp x)
def g (x : ℝ) (t : ℝ) : ℝ := (f x)^2 - t * (f x)

theorem range_of_t (t : ℝ) :
  (∃ a b c d : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
  g a t = -2 ∧ g b t = -2 ∧ g c t = -2 ∧ g d t = -2) →
  t ∈ Ioi (1 / Real.exp 1 + 2 * Real.exp 1) :=
by sorry

end range_of_t_l159_159055


namespace vovochka_correct_sum_cases_vovochka_min_difference_l159_159157

-- Part (a)
theorem vovochka_correct_sum_cases : 
  (∑ (a : ℕ) in finset.range 10, (∑ (b : ℕ) in finset.range (10 - a), 1)) ^ 3 = 244620 :=
sorry

-- Part (b)
theorem vovochka_min_difference : 
  ∃ (a b c x y z : ℕ), (a * 100 + b * 10 + c + x * 100 + y * 10 + z) - (a + x) * 100 - (b + y) * 10 - (c + z) = 1800 :=
sorry

end vovochka_correct_sum_cases_vovochka_min_difference_l159_159157


namespace find_m_l159_159876

noncomputable def sum_of_log_divisors_eq_960 (m : ℕ) : Prop :=
  let sm := ∑ a in Finset.range (m+2), ∑ b in Finset.range (m+2), (a * Real.log10 2 + b * Real.log10 5)
  sm = 960

theorem find_m : ∃ m : ℕ, sum_of_log_divisors_eq_960 m :=
by
  exists 11
  unfold sum_of_log_divisors_eq_960
  sorry

end find_m_l159_159876


namespace smallest_sum_value_l159_159399

theorem smallest_sum_value :
  ∀ (x : Fin 100 → ℕ), 
  (∀ i, 1 ≤ x i ∧ x i ≤ 100) →
  (∀ i j, i ≠ j → x i ≠ x j) →
  ∃ S, S = ∑ i in Finset.range 99, abs (x (i+1) - x i) + abs (x 0 - x 99) ∧ S = 198 := 
by 
  sorry

end smallest_sum_value_l159_159399


namespace max_sum_terms_arithmetic_seq_l159_159170

theorem max_sum_terms_arithmetic_seq (a1 d : ℝ) (h1 : a1 > 0) 
  (h2 : 3 * (2 * a1 + 2 * d) = 11 * (2 * a1 + 10 * d)) :
  ∃ (n : ℕ),  (∀ k, 1 ≤ k ∧ k ≤ n → a1 + (k - 1) * d > 0) ∧  a1 + n * d ≤ 0 ∧ n = 7 :=
by
  sorry

end max_sum_terms_arithmetic_seq_l159_159170


namespace minimize_total_price_l159_159276

noncomputable def total_price (a : ℝ) (m x : ℝ) : ℝ :=
  a * ((m / 2 + x)^2 + (m / 2 - x)^2)

theorem minimize_total_price (a m : ℝ) : 
  ∃ y : ℝ, (∀ x, total_price a m x ≥ y) ∧ y = total_price a m 0 :=
by
  sorry

end minimize_total_price_l159_159276


namespace jeff_match_points_l159_159187

theorem jeff_match_points :
  (jeff_hours : ℕ) (points_per_minute : ℕ) (games_won : ℕ) 
  (H1 : jeff_hours = 2) 
  (H2 : points_per_minute = 5) 
  (H3 : games_won = 3) :
  (points_needed_per_game : ℕ) := 
  sorry

end jeff_match_points_l159_159187


namespace intersection_of_M_and_N_l159_159092

def M : Set ℝ := {x | 0 < x ∧ x < 3}
def N : Set ℝ := {x | x > 2 ∨ x < -2}
def expected_intersection : Set ℝ := {x | 2 < x ∧ x < 3}

theorem intersection_of_M_and_N : M ∩ N = expected_intersection := by
  sorry

end intersection_of_M_and_N_l159_159092


namespace vasya_birthday_was_thursday_l159_159712

variable (today : String)
variable (tomorrow : String)
variable (day_after_tomorrow : String)
variable (birthday : String)

-- Conditions given in the problem
axiom birthday_not_sunday : birthday ≠ "Sunday"
axiom sunday_day_after_tomorrow : day_after_tomorrow = "Sunday"

-- From the conditions, we need to prove that Vasya's birthday was on Thursday.
theorem vasya_birthday_was_thursday : birthday = "Thursday" :=
by
  -- Fill in the proof here
  sorry

end vasya_birthday_was_thursday_l159_159712


namespace inverse_of_projection_matrix_l159_159609

def projection_matrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let norm_sq := v.1^2 + v.2^2
  let u := (⟨v.1 / real.sqrt norm_sq, v.2 / real.sqrt norm_sq⟩ : ℝ × ℝ)
  Matrix.mul (Matrix.ofVec (λ _, u)) (Matrix.transpose (Matrix.ofVec (λ _, u)))

theorem inverse_of_projection_matrix
  (v : ℝ × ℝ)
  (v_eq : v = (⟨4, 5⟩ : ℝ × ℝ)) :
  ¬ ∃ inv : Matrix (Fin 2) (Fin 2) ℝ, inv * (projection_matrix v) = (1 : Matrix (Fin 2) (Fin 2) ℝ) :=
by
  have h : projection_matrix v = (⟨⟨1/41 * 16, 1/41 * 20⟩, ⟨1/41 * 20, 1/41 * 25⟩⟩ : Matrix (Fin 2) (Fin 2) ℝ),
    from sorry -- Here you can compute the matrix manually or use a library function.
  have det_zero: Matrix.det (projection_matrix v) = 0,
    from sorry -- Compute that the determinant is zero.
  rw Matrix.det at det_zero,
  intro h,
  cases h with inv h_inv,
  simp [Matrix.det_smul, Matrix.det_zero, det_zero] at h_inv,
  contradiction

end inverse_of_projection_matrix_l159_159609


namespace difference_is_three_times_f_l159_159103

def f (x : ℝ) : ℝ := 4 ^ x

theorem difference_is_three_times_f (x : ℝ) : f (x + 1) - f x = 3 * f x :=
by
  unfold f
  -- Here, you would perform the necessary steps and simplifications to finish the proof.
  sorry

end difference_is_three_times_f_l159_159103


namespace parallel_BE_DF_l159_159211

variables {A B C D E F X : Type}
variables [triangle ABC] 

theorem parallel_BE_DF 
  (h1 : ∃ A B C : Type, ∀ (2 * ∠CBA = 3 * ∠ACB))
  (h2 : ∃ D E : Type, D ∈ (segment A C) ∧ E ∈ (segment A C) ∧ D between A and E ∧ trisecting BD BE (∠CBA))
  (h3 : ∃ F : Type, F = intersection (line A B) (bisector ∠ACB))
  : parallel BE DF := 
sorry

end parallel_BE_DF_l159_159211


namespace vasya_birthday_l159_159727

/--
Vasya said the day after his birthday: "It's a pity that my birthday 
is not on a Sunday this year, because more guests would have come! 
But Sunday will be the day after tomorrow..."
On what day of the week was Vasya's birthday?
-/
theorem vasya_birthday (today : string)
  (h1 : today = "Friday")
  (h2 : ∀ day : string, day ≠ "Sunday" → Vasya's_birthday day) :
  Vasya's_birthday "Thursday" := by
  sorry

end vasya_birthday_l159_159727


namespace no_real_roots_l159_159019

noncomputable def sqrt_eq (x : ℝ) : Prop :=
  sqrt (x + 9) - sqrt (x - 5) + 2 = 0

theorem no_real_roots : ¬ ∃ x : ℝ, sqrt_eq x :=
by sorry

end no_real_roots_l159_159019


namespace solution_set_lg_eq_l159_159441

theorem solution_set_lg_eq (x : ℝ) :
  lg (4^x + 2) = lg (2^x) + lg 3 ↔ x = 0 ∨ x = 1 :=
by sorry

end solution_set_lg_eq_l159_159441


namespace integer_count_between_cubes_l159_159526

-- Definitions and conditions
def a : ℝ := 10.7
def b : ℝ := 10.8

-- Precomputed values
def a_cubed : ℝ := 1225.043
def b_cubed : ℝ := 1259.712

-- The theorem to prove
theorem integer_count_between_cubes (ha : a ^ 3 = a_cubed) (hb : b ^ 3 = b_cubed) :
  let start := Int.ceil a_cubed
  let end_ := Int.floor b_cubed
  end_ - start + 1 = 34 :=
by
  sorry

end integer_count_between_cubes_l159_159526


namespace probability_two_red_two_blue_correct_l159_159342

noncomputable def num_ways_to_choose : ℕ → ℕ → ℕ :=
  λ n k, Nat.choose n k

noncomputable def probability_two_red_two_blue : ℚ :=
  let total_ways := num_ways_to_choose 20 4
  let ways_red := num_ways_to_choose 12 2
  let ways_blue := num_ways_to_choose 8 2
  (ways_red * ways_blue) / total_ways

theorem probability_two_red_two_blue_correct :
  probability_two_red_two_blue = 616 / 1615 :=
by
  sorry

end probability_two_red_two_blue_correct_l159_159342


namespace remainder_8_pow_310_mod_9_l159_159749

theorem remainder_8_pow_310_mod_9 : (8 ^ 310) % 9 = 8 := 
by
  sorry

end remainder_8_pow_310_mod_9_l159_159749


namespace distribution_scheme_count_l159_159001

-- Definitions based on conditions
variable (village1 village2 village3 village4 : Type)
variables (quota1 quota2 quota3 quota4 : ℕ)

-- Conditions as given in the problem
def valid_distribution (v1 v2 v3 v4 : ℕ) : Prop :=
  v1 = 1 ∧ v2 = 2 ∧ v3 = 3 ∧ v4 = 4

-- The goal is to prove the number of permutations is equal to 24
theorem distribution_scheme_count :
  (∃ v1 v2 v3 v4 : ℕ, valid_distribution v1 v2 v3 v4) → 
  (4 * 3 * 2 * 1 = 24) :=
by 
  sorry

end distribution_scheme_count_l159_159001


namespace impossible_to_arrange_arcs_l159_159327

theorem impossible_to_arrange_arcs (R : ℝ) (S : sphere (0 : ℝ × ℝ × ℝ) R)
  (great_circle : Π θ : ℝ, set (ℝ × ℝ × ℝ))
  (is_great_circle : ∀ θ, is_circle (great_circle θ) ∧ ∃ plane : set (ℝ × ℝ × ℝ → ℝ), plane 0 = 0 ∧ plane.intersects (great_circle θ) S)
  (arc_length : ∀ θ, measure (great_circle θ) = 300 / 360 * 2 * real.pi * R)
  (no_intersections : ∀ θ₁ θ₂, θ₁ ≠ θ₂ → ∀ x ∈ great_circle θ₁, x ∉ great_circle θ₂) :
  false :=
begin
  sorry
end

end impossible_to_arrange_arcs_l159_159327


namespace width_of_field_l159_159322

-- Definitions based on conditions
def width (W : ℝ) : Prop := ∃ L, L = (7/5) * W ∧ 2 * L + 2 * W = 240 

-- Proof function defining the problem
theorem width_of_field : ∃ (W : ℝ), width W ∧ W = 50 :=
by 
  existsi 50
  unfold width
  existsi (7/5) * 50
  split
  {
    linarith
  }
  {
    linarith
  }
  sorry

end width_of_field_l159_159322


namespace sum_of_coefficients_l159_159868

theorem sum_of_coefficients :
  let expr := (x^2 - 3 * x * y + 2 * y^2) ^ 5 in
  ∑ (i : ℕ) in (finset.range 6), (coeff expr i) = 0
:=
begin
  sorry
end

end sum_of_coefficients_l159_159868


namespace range_of_lambda_l159_159565

noncomputable def triangle_problem (A B C a b c : ℝ) (λ : ℝ) : Prop :=
  A = 2 * Real.pi / 3 ∧ a = 1 ∧ 2 ≠ 0 ∧ (∃B ∈ Ioo 0 (Real.pi / 3), 
  ∃C ∈ Ioo 0 (Real.pi / 3), ∃b c ∈ Ioo 0 ∞, sin A / sin B = a / b ∧ sin A / sin C = a / c)

theorem range_of_lambda : ∀ A B C a b c λ,
  triangle_problem A B C a b c λ → 
  λ ∈ Ioo (1 / 2) 2 :=
by
  intros A B C a b c λ h
  sorry 

end range_of_lambda_l159_159565


namespace algorithm_output_l159_159260

-- Defining the process of the algorithm as a function
def algorithm : ℕ :=
  let rec loop (S i : ℕ) : ℕ :=
    if i > 12 then S
    else loop (S + i) (i + 2)
  in loop 0 1

-- Stating the theorem
theorem algorithm_output : algorithm = 36 :=
begin
  -- The proof would go here.
  sorry
end

end algorithm_output_l159_159260


namespace parallelogram_to_rectangle_l159_159419

-- Statement: Given a parallelogram, show that there exists a way to cut it into pieces
-- that can be rearranged to form a rectangle.
theorem parallelogram_to_rectangle :
  ∀ (P : Type) [parallelogram P], 
  ∃ (R : Type) [rectangle R], 
  (cut_and_rearrange P R) := sorry

end parallelogram_to_rectangle_l159_159419


namespace lucy_pick_probability_l159_159108

theorem lucy_pick_probability :
  let MATHEMATICS := "MATHEMATICS".to_list in
  let CALCULUS := "CALCULUS".to_list in
  let distinct_math := ['M', 'A', 'T', 'H', 'E', 'I', 'C', 'S'] in
  let common_letters := ['C', 'A', 'S'] in
  (distinct_math.count (λ x, x ∈ common_letters)) = 3 ∧
  (distinct_math.length = 8) →
  ((common_letters.length : ℚ) / (distinct_math.length : ℚ) = 3/8) :=
by simp; norm_num; sorry

end lucy_pick_probability_l159_159108


namespace factorial_multiple_square_l159_159890

theorem factorial_multiple_square (n : ℕ) (h : n > 0) : (Nat.factorial n) % (n^2) = 0 ↔ 
  n = 1 ∨ (n ≥ 6 ∧ ∃ k m : ℕ, (k > 1) ∧ (k < n) ∧ (m > 1) ∧ (m < n) ∧ k * m = n) :=
by 
  sorry

end factorial_multiple_square_l159_159890


namespace eliminate_y_l159_159661

theorem eliminate_y (x y : ℝ) (h1 : 2 * x + 3 * y = 1) (h2 : 3 * x - 6 * y = 7) :
  (4 * x + 6 * y) + (3 * x - 6 * y) = 9 :=
by
  sorry

end eliminate_y_l159_159661


namespace inverse_of_B_cubed_l159_159071

theorem inverse_of_B_cubed (B_inv : Matrix (Fin 2) (Fin 2) ℚ) 
  (hB : B_inv = Matrix.ofList [[3, 7], [-2, -5]]) :
  (Matrix.mul B_inv B_inv).mul B_inv = Matrix.ofList [[13, 0], [-42, -95]] :=
by {
  sorry
}

end inverse_of_B_cubed_l159_159071


namespace vasya_birthday_day_l159_159740

/-- Define the days of the week as an inductive type --/
inductive Day
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday
deriving DecidableEq

open Day

/-- Function to get the day after a given day --/
def next_day : Day → Day
| Monday    => Tuesday
| Tuesday   => Wednesday
| Wednesday => Thursday
| Thursday  => Friday
| Friday    => Saturday
| Saturday  => Sunday
| Sunday    => Monday

/-- Function to get the day two days after a given day --/
def day_after_tomorrow (d : Day) : Day :=
  next_day (next_day d)

/-- Lean statement for the problem --/
theorem vasya_birthday_day (today : Day) (H1 : day_after_tomorrow today = Sunday) (H2 : next_day vasya_birthday = today) : vasya_birthday = Thursday := 
sorry

end vasya_birthday_day_l159_159740


namespace triangle_side_lengths_l159_159523

theorem triangle_side_lengths (x : ℤ) : 3 < x ∧ x < 11 → ∃ n : ℕ, n = 7 :=
by
  intro h
  use 7
  sorry

end triangle_side_lengths_l159_159523


namespace truth_table_constructible_l159_159230

theorem truth_table_constructible
  (T : list (list bool)) -- This represents our truth table
  (logical_ops := ∧ ∨ ¬) -- Logical operators used
  (one_var_cases := { p, ¬ p, p ∧ ¬ p, p ∨ ¬ p }) -- Single variable cases
  (two_var_cases := { p ∧ q, p ∧ ¬ q, ¬ p ∧ q, ¬ p ∧ ¬ q, p ∧ ¬ p, p ∨ ¬ p }) -- Two variable cases
  (n_var_cases := { p_1 ∧ p_2 ∧ ... ∧ p_n, ¬ p_1 ∧ p_2 ∧ ... ∧ p_n, ..., p_1 ∧ ¬ p_1, p_1 ∨ ¬ p_1 }) -- \( n \) variable cases
  (n : ℕ) : ∀ T, ∃ P : formula, represents(T, P) :=
by
  sorry

end truth_table_constructible_l159_159230


namespace forestry_afforestation_fourth_year_l159_159817

theorem forestry_afforestation_fourth_year :
  let a : ℕ → ℝ := λ n, 10000 * (1.2 ^ n)
  a 3 = 17280 :=
by
  let a := λ n : ℕ, 10000 * (1.2 ^ (n : ℕ))
  have h : a 3 = 10000 * (1.2 ^ 3) := by trivial
  have h' : 10000 * (1.2 ^ 3) = 17280 := by norm_num
  exact eq.trans h h'

end forestry_afforestation_fourth_year_l159_159817


namespace prism_volume_max_l159_159166

noncomputable def maximum_prism_volume (b h : ℝ) : ℝ :=
  b^2 * h / 2

theorem prism_volume_max : 
  ∀ (b h : ℝ), 
  (2 * h * b + 2 * h * b + b^2 = 30) → 
  (asinh(1/b) = π/2) →
  (h = (15 - b^2) / (4 * b)) →
  maximum_prism_volume b h = 5 * sqrt 5 / 2 :=
by
  intros b h h_eq theta_eq h_def
  have h := (15 - b^2) / (4 * b)    -- Given h in terms of b
  rw [maximum_prism_volume, h]
  sorry

end prism_volume_max_l159_159166


namespace line_tangent_to_circle_l159_159489

noncomputable def circle_diameter : ℝ := 13
noncomputable def distance_from_center_to_line : ℝ := 6.5

theorem line_tangent_to_circle :
  ∀ (d r : ℝ), d = 13 → r = 6.5 → r = d/2 → distance_from_center_to_line = r → 
  (distance_from_center_to_line = r) := 
by
  intros d r hdiam hdist hradius hdistance
  sorry

end line_tangent_to_circle_l159_159489


namespace avg_b_c_is_45_l159_159263

-- Define the weights of a, b, and c
variables (a b c : ℝ)

-- Conditions given in the problem
def avg_a_b_c (a b c : ℝ) := (a + b + c) / 3 = 45
def avg_a_b (a b : ℝ) := (a + b) / 2 = 40
def weight_b (b : ℝ) := b = 35

-- Theorem statement
theorem avg_b_c_is_45 (a b c : ℝ) (h1 : avg_a_b_c a b c) (h2 : avg_a_b a b) (h3 : weight_b b) :
  (b + c) / 2 = 45 := by
  -- Proof omitted for brevity
  sorry

end avg_b_c_is_45_l159_159263


namespace sum_of_solutions_is_correct_l159_159336

noncomputable def sum_of_three_smallest_solutions : ℚ :=
  let solutions := 
    { x : ℚ // ∃ (n : ℤ), x - n = 2 / n ∧ 0 < x ∧ x < 5 ∧ (n = 2 ∨ n = 3 ∨ n = 4) } in
  let sorted_solutions := List.sort solutions.val in
  sorted_solutions.nth 0 + sorted_solutions.nth 1 + sorted_solutions.nth 2

theorem sum_of_solutions_is_correct : sum_of_three_smallest_solutions = 11 + 1 / 6 :=
sorry

end sum_of_solutions_is_correct_l159_159336


namespace pablo_fraction_give_to_mia_l159_159646

variable {chocolates : Type*} [linear_ordered_field chocolates]
variables (s p m : chocolates)
variables (x : chocolates) -- number of chocolates Pablo gives to Mia

theorem pablo_fraction_give_to_mia (h1: s = 1/2 * m) 
    (h2: m = 1/3 * p) 
    (h3: s + (1/4) * ((1/2) * m + x) = 2 * 1/2 * m) : 
  x = 1/2 * p := 
sorry

end pablo_fraction_give_to_mia_l159_159646


namespace beth_finishes_first_l159_159395

-- Variables and respective areas
variables (x y : ℝ) -- x is the area of Andy's lawn, y is the rate of Andy's mower

-- Derived areas of the lawns
def area_beth : ℝ := x / 3
def area_carlos : ℝ := x / 4

-- Derived mowing rates
def rate_beth : ℝ := y / 2
def rate_carlos : ℝ := y / 6

-- Time to mow each lawn
def time_andy : ℝ := x / y
def time_beth : ℝ := (x / 3) / (y / 2)
def time_carlos : ℝ := (x / 4) / (y / 6)

-- Theorem to prove Beth finishes first
theorem beth_finishes_first : time_beth x y < time_andy x y ∧ time_beth x y < time_carlos x y := by
  unfold time_beth time_andy time_carlos area_beth area_carlos rate_beth rate_carlos
  rw [div_div_div_cancel_right]
  rw [←div_mul_eq_div_mul (x / 3) 2 y, ←div_mul_eq_div_mul (x / 4) 2 y]
  sorry -- Proof is omitted as requested

end beth_finishes_first_l159_159395


namespace triangle_is_isosceles_l159_159121

theorem triangle_is_isosceles
  (A B C : ℝ)
  (h_sum_angles : A + B + C = π)
  (h_condition : (Real.sin A + Real.sin B) * (Real.cos A + Real.cos B) = 2 * Real.sin C) :
  A = B :=
sorry

end triangle_is_isosceles_l159_159121


namespace ones_digit_largest_power_of_2_divides_32_factorial_ones_digit_largest_power_of_3_divides_32_factorial_l159_159873

theorem ones_digit_largest_power_of_2_divides_32_factorial : 
  (2^31 % 10) = 8 := 
by
  sorry

theorem ones_digit_largest_power_of_3_divides_32_factorial : 
  (3^14 % 10) = 9 := 
by
  sorry

end ones_digit_largest_power_of_2_divides_32_factorial_ones_digit_largest_power_of_3_divides_32_factorial_l159_159873


namespace prob_XiaoMing_choose_B_prob_XiaoMing_XiaoGang_same_l159_159172

-- Define the set of projects
inductive Project
| A
| B
| C
| D

open Project

-- Probability of choosing a specific project from four equally likely projects
def prob_choose_project (p : Project) : ℚ :=
  if p = B then 1 / 4 else 1 / 4

theorem prob_XiaoMing_choose_B :
  prob_choose_project B = 1 / 4 :=
by
  simp [prob_choose_project]

-- Probability of Xiao Ming and Xiao Gang choosing the same project
def prob_same_choice (proj1 proj2 : Project) : ℚ :=
  if proj1 = proj2 then 1 / 4 else 1 / 16

theorem prob_XiaoMing_XiaoGang_same :
  (prob_same_choice A A + prob_same_choice B B + prob_same_choice C C + prob_same_choice D D) = 1 / 4 :=
by
  simp [prob_same_choice]
  norm_num
  sorry

end prob_XiaoMing_choose_B_prob_XiaoMing_XiaoGang_same_l159_159172


namespace vovochka_add_correct_pairs_vovochka_min_diff_incorrect_l159_159134

-- Vovochka's addition method: adding two digits without carrying over
def vovochka_add (a b : ℕ) : ℕ := (a % 10 + b % 10) + ((a / 10 % 10 + b / 10 % 10) * 10) + ((a / 100 + b / 100) * 100)

-- Part (a): number of pairs producing correct result with Vovochka’s method
def correct_vovochka_pairs_count : ℕ := 244620

-- Part (b): smallest possible difference when Vovochka’s method is incorrect
def min_diff_vovochka_method : ℕ := 1800

-- Proving the number of correct cases equals 244620
theorem vovochka_add_correct_pairs :
  let count := ∑ (a b : ℕ) in finset.Icc 100 999, if vovochka_add a b = a + b then 1 else 0
  in count = correct_vovochka_pairs_count := sorry

-- Proving the smallest possible difference when Vovochka’s method is incorrect
theorem vovochka_min_diff_incorrect :
  let min_diff := min (∑ (a b : ℕ) in finset.Icc 100 999, if vovochka_add a b ≠ a + b then nat.abs (vovochka_add a b - (a + b)) else ⊤)
  in min_diff = min_diff_vovochka_method := sorry

end vovochka_add_correct_pairs_vovochka_min_diff_incorrect_l159_159134


namespace subcommittee_count_l159_159685

theorem subcommittee_count :
  let total_members := 12
  let total_teachers := 5
  let subcommittee_size := 5
  let total_subcommittees := Nat.choose total_members subcommittee_size
  let non_teacher_subcommittees_with_0_teachers := Nat.choose (total_members - total_teachers) subcommittee_size
  let non_teacher_subcommittees_with_1_teacher :=
    Nat.choose total_teachers 1 * Nat.choose (total_members - total_teachers) (subcommittee_size - 1)
  (total_subcommittees
   - (non_teacher_subcommittees_with_0_teachers + non_teacher_subcommittees_with_1_teacher)) = 596 := 
by
  sorry

end subcommittee_count_l159_159685


namespace correct_quadratic_equation_l159_159378

theorem correct_quadratic_equation (α β : ℝ)
  (h₁ : (1 + β) / (2 + β) = -1 / α)
  (h₂ : (α * β^2 + 121) / (1 - α^2 * β) = 1) :
  (∀ x, x^2 + 12 * x + 10 = 0 ∨ x^2 - 10 * x - 12 = 0) :=
begin
  sorry
end

end correct_quadratic_equation_l159_159378


namespace problem_correct_l159_159008

noncomputable def problem : ℕ :=
  let val1 := 3 * 8^2 + 6 * 8^1 + 7 * 8^0
  let C := 12
  let D := 0
  let val2 := 4 * 13^2 + C * 13^1 + D * 13^0
  val1 + val2

theorem problem_correct : problem = 1079 :=
by 
  have h1 : 3 * 8^2 + 6 * 8^1 + 7 * 8^0 = 247 := by norm_num
  have h2 : 4 * 13^2 + 12 * 13^1 + 0 * 13^0 = 832 := by norm_num
  rw [←h1, ←h2]
  norm_num
  sorry

end problem_correct_l159_159008


namespace min_value_of_A_l159_159017

noncomputable def A (n : ℕ) (x : Fin n → ℝ) : ℝ :=
  let sin_sum := ∑ i, Real.sin (x i)
  let cos_sum := ∑ i, Real.cos (x i)
  (2 * sin_sum + cos_sum) * (sin_sum - 2 * cos_sum)

theorem min_value_of_A (n : ℕ) (x : Fin n → ℝ) : 
  ∃ x, A n x = - (5 * n^2) / 2 :=
sorry

end min_value_of_A_l159_159017


namespace total_families_l159_159997

theorem total_families (F_2dogs F_1dog F_2cats total_animals total_families : ℕ) 
  (h1: F_2dogs = 15)
  (h2: F_1dog = 20)
  (h3: total_animals = 80)
  (h4: 2 * F_2dogs + F_1dog + 2 * F_2cats = total_animals) :
  total_families = F_2dogs + F_1dog + F_2cats := 
by 
  sorry

end total_families_l159_159997


namespace min_top_block_sum_l159_159313

/-- Define the pyramid structure with weights corresponding to their contribution to the top block. -/
noncomputable def block_weights : List ℕ := 
  [1, 1, 1, 3, 3, 3, 3, 3, 3, 6]

/-- 
Given the conditions stated as follows:
 - The pyramid structure as described.
 - The rules for summing numbers from blocks below.

Prove that the smallest possible number that could be assigned to the top block is 114.
-/
theorem min_top_block_sum : 
  ∃ (assignment : Fin 10 → ℕ), 
  (∀ i : Fin 10, assignment i ≥ 1 ∧ assignment.sum ∧ assignment.prod') 
  ∧ (assignment.sum = 114) 
  := sorry

end min_top_block_sum_l159_159313


namespace largest_awesome_prime_l159_159519

def is_awesome_prime (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∀ q : ℕ, 1 ≤ q ∧ q < p → Nat.Prime (p + 2 * q)

theorem largest_awesome_prime : ∀ p : ℕ, is_awesome_prime p → p ≤ 3 :=
by
  intro p hp
  sorry

end largest_awesome_prime_l159_159519


namespace diplomats_neither_french_nor_russian_l159_159398

variable (total_diplomats : ℕ)
variable (speak_french : ℕ)
variable (not_speak_russian : ℕ)
variable (speak_both : ℕ)

theorem diplomats_neither_french_nor_russian {total_diplomats speak_french not_speak_russian speak_both : ℕ} 
  (h1 : total_diplomats = 100)
  (h2 : speak_french = 22)
  (h3 : not_speak_russian = 32)
  (h4 : speak_both = 10) :
  ((total_diplomats - (speak_french + (total_diplomats - not_speak_russian) - speak_both)) * 100) / total_diplomats = 20 := 
by
  sorry

end diplomats_neither_french_nor_russian_l159_159398


namespace sqrt_sum_eq_two_l159_159984

theorem sqrt_sum_eq_two (a b : ℝ) (h : abs (a - 1) + (b - 3)^2 = 0) : sqrt (a + b) = 2 :=
by
  sorry -- proof is not required as per instructions

end sqrt_sum_eq_two_l159_159984


namespace vector_collinear_of_magnitude_condition_l159_159967

variables (V : Type) [inner_product_space ℝ V]
variables (a b : V)
variables (non_zero_a : a ≠ 0)
variables (non_zero_b : b ≠ 0)

theorem vector_collinear_of_magnitude_condition (h : ∥a + b∥ = ∥a∥ - ∥b∥) : 
  ∃ (λ : ℝ), a = λ • b :=
sorry

end vector_collinear_of_magnitude_condition_l159_159967


namespace vector_parallel_l159_159514

theorem vector_parallel (x : ℝ) : ∃ x, (1, x) = k * (-2, 3) → x = -3 / 2 :=
by 
  sorry

end vector_parallel_l159_159514


namespace total_distance_from_point_A_total_fuel_consumed_l159_159363

/-- 
 A certain testing team is inspecting the power supply lines of cars.
 They define moving forward as positive and moving backward as negative.
 The distances (in kilometers) traveled from point A to the end of work are:
 +2, -3, +4, -2, -8, +17, -2, -3, +12, +7, -5.
 -/
def distances : List ℤ := [+2, -3, +4, -2, -8, +17, -2, -3, +12, +7, -5]

/-- How far is the end of work from point A? -/
theorem total_distance_from_point_A
  : distances.sum = 19 :=
sorry

/-- If the car consumes 0.4L of fuel per kilometer, how much fuel is consumed from point A to the end of work in total? -/
def fuel_consumption_rate : ℝ := 0.4

theorem total_fuel_consumed 
  (d_abs := distances.map Int.natAbs)
  : (d_abs.sum : ℝ) * fuel_consumption_rate = 26 :=
sorry

end total_distance_from_point_A_total_fuel_consumed_l159_159363


namespace distinct_real_roots_implies_positive_l159_159552

theorem distinct_real_roots_implies_positive (k : ℝ) (x1 x2 : ℝ) (h_distinct : x1 ≠ x2) 
  (h_root1 : x1^2 + 2*x1 - k = 0) 
  (h_root2 : x2^2 + 2*x2 - k = 0) : 
  x1^2 + x2^2 - 2 > 0 := 
sorry

end distinct_real_roots_implies_positive_l159_159552


namespace circle_radius_l159_159859

noncomputable def circle_problem (rD rE : ℝ) (m n : ℝ) :=
  rD = 2 * rE ∧
  rD = (Real.sqrt m) - n ∧
  m ≥ 0 ∧ n ≥ 0

theorem circle_radius (rE rD : ℝ) (m n : ℝ) (h : circle_problem rD rE m n) :
  m + n = 5.76 :=
by
  sorry

end circle_radius_l159_159859


namespace Vovochka_correct_pairs_count_l159_159144

def no_carry_pairs_count (digit_sum: ℕ → ℕ → Prop) : ℕ :=
  let count_pairs (lim: ℕ) : ℕ := (lim * (lim + 1)) / 2
  let digit_valid_pairs : ℕ := count_pairs 9
  (digit_valid_pairs * digit_valid_pairs) * 81

def digit_sum (x y: ℕ) : Prop := (x + y ≤ 9)

theorem Vovochka_correct_pairs_count :
  no_carry_pairs_count digit_sum = 244620 := by
  sorry

end Vovochka_correct_pairs_count_l159_159144


namespace Vasya_birthday_on_Thursday_l159_159721

-- Define the days of the week enumeration
inductive DayOfWeek
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open DayOfWeek

-- Define the conditions
def Sunday_is_the_day_after_tomorrow (today : DayOfWeek) : Prop :=
  match today with
  | Friday => true
  | _      => false

def Vasya_birthday_not_on_Sunday (birthday : DayOfWeek) : Prop :=
  birthday ≠ Sunday

def Today_is_next_day_after_birthday (today birthday : DayOfWeek) : Prop :=
  match (today, birthday) with
  | (Friday, Thursday) => true
  | _                  => false

-- The main theorem to prove Vasya's birthday was on Thursday
theorem Vasya_birthday_on_Thursday (birthday today : DayOfWeek) 
  (H1: Sunday_is_the_day_after_tomorrow today) 
  (H2: Vasya_birthday_not_on_Sunday birthday) 
  (H3: Today_is_next_day_after_birthday today birthday) : 
  birthday = Thursday :=
by
  sorry

end Vasya_birthday_on_Thursday_l159_159721


namespace multiply_polynomials_l159_159640

open Polynomial

variable {R : Type*} [CommRing R]

theorem multiply_polynomials (x : R) :
  (x^4 + 6*x^2 + 9) * (x^2 - 3) = x^4 + 6*x^2 :=
  sorry

end multiply_polynomials_l159_159640


namespace PR_plus_PS_eq_AF_l159_159332

-- Definitions based on problem conditions
variable (AB CD : ℝ)
variable (P : Point)
variable (AC BD AF PR PS PQ : Line)
variable (A B : Point)

-- Specify conditions
def is_isosceles_trapezoid (AB CD : ℝ) (A B C D : Point) : Prop :=
  (A ≠ B ∧ A ≠ D ∧ B ≠ D) ∧ (A ≠ C ∧ B ≠ C ∧ C ≠ D) ∧
  (A) ≠ (D) ∧
  (IsParallel AB CD) ∧ (AB ≠ CD)

def point_on_base_P (P : Point) (AB : Line) : Prop :=
  isOn(P, AB)

def is_perpendicular (l1 l2 : Line) : Prop :=
  l1 ⊥ l2

-- Objective: To prove PR + PS = AF

theorem PR_plus_PS_eq_AF (AB CD : ℝ) (A B C D P : Point)
    (AC BD AF PR PS PQ : Line)
    (h1 : is_isosceles_trapezoid AB CD A B C D)
    (h2 : point_on_base_P P AB)
    (h3 : is_perpendicular PR AC)
    (h4 : is_perpendicular PS BD)
    (h5 : is_perpendicular AF BD)
    (h6 : is_perpendicular PQ AF) :
  length PR + length PS = length AF :=
sorry

end PR_plus_PS_eq_AF_l159_159332


namespace cylinder_volume_l159_159697

theorem cylinder_volume (r h : ℝ) (hr : r = 5) (hh : h = 10) :
    π * r^2 * h = 250 * π := by
  -- We leave the actual proof as sorry for now
  sorry

end cylinder_volume_l159_159697


namespace find_starting_number_l159_159699

theorem find_starting_number :
  ∃ startnum : ℕ, startnum % 5 = 0 ∧ (∀ k : ℕ, 0 ≤ k ∧ k < 20 → startnum + 5 * k ≤ 100) ∧ startnum = 10 :=
sorry

end find_starting_number_l159_159699


namespace analogous_property_in_solid_geometry_l159_159173

theorem analogous_property_in_solid_geometry :
  (∀ (P : Type) [metric_space P] (triangle : set P), 
    (∀ (p : P), p ∈ triangle → (sum_of_distances_to_sides p triangle) = constant_value) → 
  ∃ (solid : Type) (surfaces_property : set solid), true) := 
begin
  -- sorry as a placeholder for the actual proof.
  sorry
end

end analogous_property_in_solid_geometry_l159_159173


namespace cross_product_self_sub_l159_159979

open Vector3

variable (u v : Vector3)

theorem cross_product_self_sub (h : u.cross v = ⟨3, -1, 5⟩) :
  (u - v).cross (u - v) = ⟨0, 0, 0⟩ :=
by
  sorry

end cross_product_self_sub_l159_159979


namespace find_n_l159_159485

theorem find_n
  (x : ℝ)
  (h1 : log 10 (sin x) + log 10 (cos x) = -2)
  (h2 : log 10 (sin x + cos x) = (1 / 2) * (log 10 n - 2)) :
  n = 102 := by
  sorry

end find_n_l159_159485


namespace complement_intersection_l159_159069

def A : set ℝ := { x | x ≤ 2 }
def B : set ℝ := { x | x^2 - 3 * x ≤ 0 }

theorem complement_intersection : (set.univ \ A) ∩ B = { x | 2 < x ∧ x ≤ 3 } := by
  sorry

end complement_intersection_l159_159069


namespace vasya_birthday_day_l159_159736

/-- Define the days of the week as an inductive type --/
inductive Day
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday
deriving DecidableEq

open Day

/-- Function to get the day after a given day --/
def next_day : Day → Day
| Monday    => Tuesday
| Tuesday   => Wednesday
| Wednesday => Thursday
| Thursday  => Friday
| Friday    => Saturday
| Saturday  => Sunday
| Sunday    => Monday

/-- Function to get the day two days after a given day --/
def day_after_tomorrow (d : Day) : Day :=
  next_day (next_day d)

/-- Lean statement for the problem --/
theorem vasya_birthday_day (today : Day) (H1 : day_after_tomorrow today = Sunday) (H2 : next_day vasya_birthday = today) : vasya_birthday = Thursday := 
sorry

end vasya_birthday_day_l159_159736


namespace repeating_decimal_fraction_sum_l159_159307

theorem repeating_decimal_fraction_sum {x : ℚ} (h : x = 0.527527527527...527) :
    let fraction_form := 527 / 999
    let sum := 527 + 999
    sum = 1526 := 
by
  have h1 : x = 527 / 999, from sorry,
  sorry

end repeating_decimal_fraction_sum_l159_159307


namespace possible_values_AC_l159_159704

theorem possible_values_AC (A B C : ℝ) (hAB : abs (A - B) = 3) (hBC : abs (B - C) = 5) :
  abs (A - C) = 2 ∨ abs (A - C) = 8 :=
begin
  sorry
end

end possible_values_AC_l159_159704


namespace polynomial_solution_l159_159889

theorem polynomial_solution (f : ℝ → ℝ) (hf : ∀ a b c : ℝ, ab + bc + ca = 0 → f(a - b) + f(b - c) + f(c - a) = 2 * f(a + b + c)) :
  ∃ p q : ℝ, ∀ x : ℝ, f x = p * x^4 + q * x^2 :=
by sorry

end polynomial_solution_l159_159889


namespace identify_false_condition_l159_159458

-- Define the quadratic function
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions provided in the problem
def condition_A (a b c : ℝ) : Prop := quadratic_function a b c (-1) = 0
def condition_B (a b c : ℝ) : Prop := 2 * a + b = 0
def condition_C (a b c : ℝ) : Prop := quadratic_function a b c 1 = 3
def condition_D (a b c : ℝ) : Prop := quadratic_function a b c 2 = 8

-- Main theorem stating which condition is false
theorem identify_false_condition (a b c : ℝ) (ha : a ≠ 0) : ¬ condition_A a b c ∨ ¬ condition_B a b c ∨ ¬ condition_C a b c ∨  ¬ condition_D a b c :=
by
sorry

end identify_false_condition_l159_159458


namespace isosceles_triangle_perimeter_l159_159992

noncomputable def triangle_perimeter (a b : Nat) : Nat := a + b * 2

theorem isosceles_triangle_perimeter (a b : Nat) (hiso : a = 4 ∧ b = 5 ∨ a = 5 ∧ b = 4) : 
  triangle_perimeter a b = 13 ∨ triangle_perimeter b a = 14 :=
by
  cases hiso
  { left
    rw [hiso.left, hiso.right]
    simp
  }
  { right
    rw [hiso.left, hiso.right]
    simp
  }

end isosceles_triangle_perimeter_l159_159992


namespace max_value_of_f_l159_159955

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 / (Real.sin x + 2)

theorem max_value_of_f : ∃ x : ℝ, f x = 1 ∧ (∀ y : ℝ, f y ≤ 1) :=
by
  -- Definitions from conditions
  let t (x : ℝ) := Real.sin x + 2
  have h₁ : ∀ x, 1 ≤ t x ∧ t x ≤ 3 := sorry
  have h₂ : ∀ x, (Real.sin x)^2 = (t x - 2)^2 := sorry

  -- Goal: prove maximum value of f(x) is 1
  have h₃ : ∀ t, t ∈ Icc 1 3 → (t - 2)^2 / t ≤ 1 := sorry
    
  use 0  -- example value to illustrate "exists"
  split
  { exact sorry } -- prove f(0) = 1
  { exact sorry } -- prove ∀ y, f(y) ≤ 1

end max_value_of_f_l159_159955


namespace probability_exactly_two_eights_l159_159009

theorem probability_exactly_two_eights :
  let n : ℕ := 15,
      k : ℕ := 2,
      prob_eight : ℚ := 1 / 8,
      prob_not_eight : ℚ := 7 / 8,
      combinations : ℚ := Nat.choose n k,
      probability : ℚ := combinations * prob_eight^k * prob_not_eight^(n - k)
  in (probability : ℝ) ≈ 0.084 :=
by
  sorry

end probability_exactly_two_eights_l159_159009


namespace max_n_Sn_pos_l159_159197

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a 1 + (↑n - 1) * (a 2 - a 1))

theorem max_n_Sn_pos (a : ℕ → ℝ) (S : ℕ → ℝ) (n_max : ℕ):
  is_arithmetic_sequence a →
  (∀ n : ℕ, S n = sum_of_first_n_terms a n) →
  a 1 > 0 →
  root_of_quadratic a 7 a 8 (λ x, x^2 + x - 2023) →
  ∀ n : ℕ, n <= n_max →
  S n > 0 :=
sorry

def root_of_quadratic (x y : ℝ) (f : ℝ → ℝ) : Prop :=
  f x = 0 ∧ f y = 0 ∧ x ≠ y ∧ x + y = -1 ∧ x * y = -2023

end max_n_Sn_pos_l159_159197


namespace slope_angle_range_l159_159221

noncomputable def curve (x : ℝ) := x^3 - real.sqrt 3 * x + 3 / 5

noncomputable def derivative (x : ℝ) := 3 * x^2 - real.sqrt 3

/-!
  Prove that the range of the slope angle α of the tangent line at any point P on the curve
  y = x^3 - √3*x + 3/5 falls into the interval [0, π/2) ∪ [2π/3, π).
-/
theorem slope_angle_range :
  ∀ (x : ℝ), (3 * x^2 - real.sqrt 3) ≥ - real.sqrt 3 → 
  ∃ (α : ℝ), 0 ≤ α ∧ α < real.pi / 2 ∨ (2 * real.pi / 3 ≤ α ∧ α < real.pi) := 
sorry

end slope_angle_range_l159_159221


namespace triangle_area_l159_159797

theorem triangle_area
  (perimeter : ℝ)
  (inradius : ℝ)
  (h₁ : perimeter = 42)
  (h₂ : inradius = 5.0) :
  let semiperimeter := perimeter / 2 in
  semiperimeter * inradius = 105 :=
by
  sorry

end triangle_area_l159_159797


namespace find_minimum_value_l159_159933

variable (a b c d : ℝ)

def condition1 : Prop := ln(b + 1) + a - 3 * b = 0
def condition2 : Prop := 2 * d - c + sqrt 5 = 0

theorem find_minimum_value (h1 : condition1 a b) (h2 : condition2 c d) : 
  (a - c)^2 + (b - d)^2 = 1 :=
by
  -- proof goes here
  sorry

end find_minimum_value_l159_159933


namespace sum_two_digit_divisors_153_l159_159617

theorem sum_two_digit_divisors_153 :
  let d_possible := {d : ℕ | d > 0 ∧ d < 100 ∧ 153 % d = 0} in
  ∑ d in d_possible, d = 68 :=
sorry

end sum_two_digit_divisors_153_l159_159617


namespace part_a_part_b_part_c_l159_159257

-- Part (a)
theorem part_a (X Y Z : ℝ × ℝ) 
  (hX : X = (2, 4))
  (hY : Y = (0, 0))
  (hZ : Z = (4, 0)) :
  let b := dist Y Z,
      h := X.2,
      s := b * h / (b + h) in
  b = 4 ∧ h = 4 ∧ s = 2 :=
by
  have : b = dist (0, 0) (4, 0) := rfl
  have : b = 4 := by simp [dist]
  have : h = 4 := rfl
  have : s = b * h / (b + h) := rfl
  have : s = (4 * 4) / 8 := rfl
  have : s = 2 := by norm_num
  exact ⟨this, this, this⟩

-- Part (b)
theorem part_b (h : ℝ) (s b : ℝ) 
  (h_eq : h = 3) 
  (s_eq : s = 2) :
  let expr := s * (b + h) = b * h in
  b = 6 :=
sorry

-- Part (c)
theorem part_c (s : ℝ) (area : ℝ) 
  (square_area : area = 2017) :
  let s := Real.sqrt area in
  let bh := 2 * area in
  let triangle_area := bh / 2 in
  triangle_area = 2017 ^ 2 / 2 :=
sorry

end part_a_part_b_part_c_l159_159257


namespace vladik_number_problem_l159_159742

theorem vladik_number_problem (N : ℕ) (seq : ℕ → ℕ) 
  (h1 : seq 0 = N) 
  (h2 : ∀ i : ℕ, prime (seq i / seq (i + 1)) ∧ seq (i + 1) = seq i / (seq i / seq (i + 1))) 
  (h3 : seq 21 = 1) 
  (h4 : (∑ i in Finset.range 22, seq i) = N / 2) : 
  N = 2 * 3 ^ 21 :=
by sorry

end vladik_number_problem_l159_159742


namespace first_problem_solution_set_second_problem_a_range_l159_159499

-- Define the function f(x) = |2x - a| + |x - 1|
def f (x : ℝ) (a : ℝ) : ℝ := abs (2 * x - a) + abs (x - 1)

-- First problem: When a = 3, the solution set of the inequality f(x) ≥ 2
theorem first_problem_solution_set (x : ℝ) : (f x 3 ≥ 2) ↔ (x ≤ 2 / 3 ∨ x ≥ 2) :=
by sorry

-- Second problem: If f(x) ≥ 5 - x for ∀ x ∈ ℝ, find the range of the real number a
theorem second_problem_a_range (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 5 - x) ↔ (6 ≤ a) :=
by sorry

end first_problem_solution_set_second_problem_a_range_l159_159499


namespace gcd_values_360_l159_159779

theorem gcd_values_360 : ∃ d : ℕ, d = 11 ∧ ∀ a b : ℕ, a * b = 360 → ∃ (g : ℕ), g = gcd a b ∧ finite {g | g = gcd a b ∧ a * b = 360} ∧ card {g | g = gcd a b ∧ a * b = 360} = 11 :=
sorry

end gcd_values_360_l159_159779


namespace probability_same_log_floor_l159_159238

-- Define real numbers x and y chosen independently and uniformly at random from (0, 1)
def x_random : Type := {x : ℝ // 0 < x ∧ x < 1}
def y_random : Type := {y : ℝ // 0 < y ∧ y < 1}

-- Express the condition that the floors of logarithms base 3 are equal
def same_log_floor (x y : ℝ) : Prop := 
  (floor (Real.log x / Real.log 3) = floor (Real.log y / Real.log 3))

-- Prove that the probability of the floors being equal is 1/8
theorem probability_same_log_floor : 
  let x := x_random
  let y := y_random
  (1 / 8 : ℝ) = 
  (probability (same_log_floor x y)) := 
sorry

end probability_same_log_floor_l159_159238


namespace option_B_incorrect_l159_159308

-- Definitions of p and q
def p : Prop := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → exp x ≥ 1
def q : Prop := ∃ x : ℝ, x^2 + x + 1 < 0

-- Statement to prove that p ∧ q is false
theorem option_B_incorrect : ¬ (p ∧ q) := 
by
  sorry

end option_B_incorrect_l159_159308


namespace max_possible_integer_results_l159_159183

theorem max_possible_integer_results : 
  ∀ (n : ℕ) (initial_list : List ℕ) (has_fractional_parts : ∀ k ∈ initial_list, ∃ f : ℚ, 0 < f ∧ f < 1), 
  initial_list.length = 100 → 
  (∀ i j ∈ initial_list, Integer.floor ((i + j : ℚ)) = i + j - (i + j) %% 1) →
  ∃ k: ℕ, k = 51 :=
by
  sorry

end max_possible_integer_results_l159_159183


namespace solve_for_x_l159_159254

theorem solve_for_x (x : ℝ) : x^2 + 6 * x + 8 = -(x + 4) * (x + 6) ↔ x = -4 :=
by {
  sorry
}

end solve_for_x_l159_159254


namespace complement_of_union_l159_159259

open Set

variable (U M N : Set ℕ)

theorem complement_of_union 
  (hU : U = {1, 2, 3, 4})
  (hM : M = {1, 2})
  (hN : N = {2, 3}) : 
  (U \ (M ∪ N)) = {4} := 
by 
  unfold U M N at hU hM hN
  rw [hU, hM, hN]
  simp
  sorry

end complement_of_union_l159_159259


namespace inequality_proof_l159_159233

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 1) :
    (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 :=
begin
  sorry
end

end inequality_proof_l159_159233


namespace count_integers_in_range_with_increasing_digits_l159_159527

-- Definition of conditions and proof goal
def has_two_digits_in_increasing_order (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  (d1 < d2) ∧ (d2 < d3)

theorem count_integers_in_range_with_increasing_digits :
  ∃ n : ℕ, n = 10 ∧
  (∀ k : ℕ, (200 ≤ k ∧ k ≤ 250) → has_two_digits_in_increasing_order(k) → k = 10) :=
by
  sorry

end count_integers_in_range_with_increasing_digits_l159_159527


namespace gcd_values_count_l159_159761

theorem gcd_values_count (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 360) : 
  (∃ S : Finset ℕ, S.card = 12 ∧ ∀ d ∈ S, d = Nat.gcd a b) :=
by
  sorry

end gcd_values_count_l159_159761


namespace Lean_equiv_problem_l159_159074

noncomputable def h (x : ℝ) (a : ℝ) := (x ^ 2 - 1) * x + a
noncomputable def k (x : ℝ) (a : ℝ) := (x - 1 / x) * Real.log x + a

theorem Lean_equiv_problem
  (h1 h2 h3 h4 : ℝ)
  (a : ℝ)
  (h1_lt_h2 : h1 < h2)
  (h3_lt_h4 : h3 < h4)
  (h1_zero : h h1 a = 0)
  (h2_zero : h h2 a = 0)
  (h3_zero : k h3 a = 0)
  (h4_zero : k h4 a = 0) :
  (a < -3 → h3 + h4 > 10 / 3) ∧ 
  (∃ a', h2 = h3 ∧ h2^3 = h4 ∧ h1 = -h2) :=
sorry

end Lean_equiv_problem_l159_159074


namespace range_of_f_l159_159035

def f (x : ℝ) : ℝ := (3 * x + 8) / (x - 4)

theorem range_of_f :
  set.range f = set.Iio 3 ∪ set.Ioi 3 :=
sorry

end range_of_f_l159_159035


namespace part1_part2_part3_part4_l159_159584

-- Definitions of points A, B, and C
def A (a : ℝ) := (-2, a + 1)
def B (a : ℝ) := (a - 1, 4)
def C (b : ℝ) := (b - 2, b)

-- Proof that the coordinates of point C when it is on the x-axis are (-2, 0)
theorem part1 (b : ℝ) (h : C(b).2 = 0) : C(b) = (-2, 0) := by
  sorry

-- Proof that the coordinates of point C when it is on the y-axis are (0, 2)
theorem part2 (b : ℝ) (h : C(b).1 = 0) : C(b) = (0, 2) := by
  sorry

-- Proof that the distance between points A and B when AB is parallel to the x-axis is 4
theorem part3 (a : ℝ) (h : (A a).2 = (B a).2) : (dist (A a) (B a) = 4) := by
  sorry

-- Proof that the coordinates of point C when CD is perpendicular to the x-axis and CD = 1 are (-1, 1) or (-3, -1)
theorem part4 (b : ℝ) (h : abs (b) = 1) : C(b) = (-1, 1) ∨ C(b) = (3, -1) := by
  sorry

end part1_part2_part3_part4_l159_159584


namespace find_positive_difference_l159_159050

theorem find_positive_difference 
  (p1 p2 : ℝ × ℝ) (q1 q2 : ℝ × ℝ) 
  (h_p1 : p1 = (0, 8)) (h_p2 : p2 = (4, 0))
  (h_q1 : q1 = (0, 5)) (h_q2 : q2 = (10, 0))
  (y : ℝ) (hy : y = 20) :
  let m_p := (p2.2 - p1.2) / (p2.1 - p1.1)
  let b_p := p1.2 - m_p * p1.1
  let x_p := (y - b_p) / m_p
  let m_q := (q2.2 - q1.2) / (q2.1 - q1.1)
  let b_q := q1.2 - m_q * q1.1
  let x_q := (y - b_q) / m_q
  abs (x_p - x_q) = 24 :=
by
  sorry

end find_positive_difference_l159_159050


namespace anna_needs_13_gallons_l159_159844

noncomputable def pillar_radius : ℝ := 8 / 2
noncomputable def pillar_height : ℝ := 20
noncomputable def paint_coverage_per_gallon : ℝ := 400
noncomputable def num_pillars : ℕ := 10

def lateral_surface_area_per_pillar (r h : ℝ) : ℝ := 2 * real.pi * r * h

def total_lateral_surface_area (area_per_pillar : ℝ) (num_pillars : ℕ) : ℝ :=
  num_pillars * area_per_pillar

def gallons_of_paint_needed (total_area : ℝ) (coverage : ℝ) : ℝ :=
  total_area / coverage

def final_gallons_of_paint (gallons_needed : ℝ) : ℕ :=
  nat.ceil gallons_needed

theorem anna_needs_13_gallons :
  final_gallons_of_paint
    (gallons_of_paint_needed
      (total_lateral_surface_area
        (lateral_surface_area_per_pillar pillar_radius pillar_height)
        num_pillars)
      paint_coverage_per_gallon) = 13 :=
begin
  -- The actual proof would go here.
  sorry
end

end anna_needs_13_gallons_l159_159844


namespace average_age_of_adults_is_28_l159_159261

-- Define all the necessary quantities and conditions

constant members_count : ℕ := 30
constant average_age_club : ℕ := 22
constant girls_count : ℕ := 10
constant boys_count : ℕ := 10
constant adults_count : ℕ := 10
constant average_age_girls : ℕ := 18
constant average_age_boys : ℕ := 20

-- Define the sum of ages based on the averages given

noncomputable def sum_ages_club : ℕ := members_count * average_age_club
noncomputable def sum_ages_girls : ℕ := girls_count * average_age_girls
noncomputable def sum_ages_boys : ℕ := boys_count * average_age_boys
noncomputable def sum_ages_adults : ℕ := sum_ages_club - sum_ages_girls - sum_ages_boys

-- Define the average age of adults using sum of adult ages and count of adults
noncomputable def average_age_adults : ℕ := sum_ages_adults / adults_count

-- State the theorem to be proven
theorem average_age_of_adults_is_28 : average_age_adults = 28 := 
  by sorry

end average_age_of_adults_is_28_l159_159261


namespace projection_of_a_on_b_eq_neg_3_div_2_l159_159916

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Conditions
def mag_a : ℝ := 3
def mag_b : ℝ := 4
def angle_ab : Real := 120 * Real.pi / 180 -- angle in radians

-- Definitions based on conditions
def vec_a_length : Real := ∥a∥ = mag_a
def vec_b_length : Real := ∥b∥ = mag_b
def cos_angle_ab : Real := Real.cos angle_ab

-- The final theorem to prove
theorem projection_of_a_on_b_eq_neg_3_div_2 (ha: vec_a_length a) (hb: vec_b_length b) (hab: cos_angle_ab = -1/2) :
  ∥a∥ * cos (120 * Real.pi / 180) = -3/2 :=
  sorry

end projection_of_a_on_b_eq_neg_3_div_2_l159_159916


namespace arithmetic_sequence_sum_l159_159586

variable {a_n : ℕ → ℝ}

def S (n : ℕ) := ∑ i in finset.range n, a_n i

theorem arithmetic_sequence_sum (h : S 10 = 120) : a_n 2 + a_n 9 = 24 :=
sorry

end arithmetic_sequence_sum_l159_159586


namespace num_gcd_values_l159_159776

-- Define the condition for the product of gcd and lcm
def is_valid_pair (a b : ℕ) : Prop :=
  gcd a b * Nat.lcm a b = 360

-- Define the main theorem statement
theorem num_gcd_values : 
  ∃ (n : ℕ), 
    (∀ a b, is_valid_pair a b → ∃ m (hm: m ≤ 360), gcd a b = m) ∧ 
    n = 12 := sorry

end num_gcd_values_l159_159776


namespace reciprocal_sum_l159_159820

theorem reciprocal_sum (x1 x2 x3 k : ℝ) (h : ∀ x, x^2 + k * x - k * x3 = 0 ∧ x ≠ 0 → x = x1 ∨ x = x2) :
  (1 / x1 + 1 / x2 = 1 / x3) := by
  sorry

end reciprocal_sum_l159_159820


namespace gcd_values_count_l159_159759

theorem gcd_values_count (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 360) : 
  (∃ S : Finset ℕ, S.card = 12 ∧ ∀ d ∈ S, d = Nat.gcd a b) :=
by
  sorry

end gcd_values_count_l159_159759


namespace vovochka_add_correct_pairs_vovochka_min_diff_incorrect_l159_159135

-- Vovochka's addition method: adding two digits without carrying over
def vovochka_add (a b : ℕ) : ℕ := (a % 10 + b % 10) + ((a / 10 % 10 + b / 10 % 10) * 10) + ((a / 100 + b / 100) * 100)

-- Part (a): number of pairs producing correct result with Vovochka’s method
def correct_vovochka_pairs_count : ℕ := 244620

-- Part (b): smallest possible difference when Vovochka’s method is incorrect
def min_diff_vovochka_method : ℕ := 1800

-- Proving the number of correct cases equals 244620
theorem vovochka_add_correct_pairs :
  let count := ∑ (a b : ℕ) in finset.Icc 100 999, if vovochka_add a b = a + b then 1 else 0
  in count = correct_vovochka_pairs_count := sorry

-- Proving the smallest possible difference when Vovochka’s method is incorrect
theorem vovochka_min_diff_incorrect :
  let min_diff := min (∑ (a b : ℕ) in finset.Icc 100 999, if vovochka_add a b ≠ a + b then nat.abs (vovochka_add a b - (a + b)) else ⊤)
  in min_diff = min_diff_vovochka_method := sorry

end vovochka_add_correct_pairs_vovochka_min_diff_incorrect_l159_159135


namespace logarithmic_division_l159_159856

theorem logarithmic_division :
  (log (1 / 4) - log 25) / 100 ^ (-1 / 2 : ℝ) = -20 :=
by
  -- Sorry is used to skip the proof
  sorry

end logarithmic_division_l159_159856


namespace number_of_guests_l159_159743

-- Define the conditions as hypotheses
def shrimp_per_guest := 5
def cost_per_pound := 17
def shrimp_per_pound := 20
def total_spent := 170

-- Define the target hypothesis
theorem number_of_guests :
  ∃ guests : ℕ,
    (guests * shrimp_per_guest) = ((total_spent / cost_per_pound) * shrimp_per_pound) :=
begin
  sorry -- proof to be filled in later
end

end number_of_guests_l159_159743


namespace vasya_birthday_was_thursday_l159_159715

variable (today : String)
variable (tomorrow : String)
variable (day_after_tomorrow : String)
variable (birthday : String)

-- Conditions given in the problem
axiom birthday_not_sunday : birthday ≠ "Sunday"
axiom sunday_day_after_tomorrow : day_after_tomorrow = "Sunday"

-- From the conditions, we need to prove that Vasya's birthday was on Thursday.
theorem vasya_birthday_was_thursday : birthday = "Thursday" :=
by
  -- Fill in the proof here
  sorry

end vasya_birthday_was_thursday_l159_159715


namespace tangent_line_circumcircle_l159_159486

variables {A B C O M N G I P D F : Type}

-- Given conditions
def circumcircle_of_triangle (O : Point) (A B C : Point) : Prop :=
  is_circumcircle O A B C

def midpoints (M : Point) (N : Point) (A B C : Point) : Prop :=
  midpoint M A B ∧ midpoint N A C

def centroid_incenter (G I : Point) (A B C : Point) : Prop :=
  is_centroid G A B C ∧ is_incenter I A B C

def side_condition (B C : LineSegment) (A B C : Point) : Prop :=
  2 * length B C = length A B + length A C

-- Theorem to prove
theorem tangent_line_circumcircle
  (O A B C M N G I : Point)
  (P F D : Point) : 
  circumcircle_of_triangle O A B C →
  side_condition B C A B C →
  midpoints M N A B C →
  centroid_incenter G I A B C →
  tangent_at_point (line_through G I) (circumcircle_of_triangle P A M N) :=
sorry

end tangent_line_circumcircle_l159_159486


namespace sequence_a1371_bound_l159_159179

def sequence (a : ℕ → ℝ) : Prop :=
  a 0 = 1 ∧ a 1 = 2 ∧ (∀ n ≥ 1, a (n + 1) = a n + (a (n - 1)) / (1 + (a (n - 1)^2)))

theorem sequence_a1371_bound (a : ℕ → ℝ) (h : sequence a) :
  52 < a 1371 ∧ a 1371 < 65 :=
sorry

end sequence_a1371_bound_l159_159179


namespace count_valid_two_digit_numbers_l159_159386

def is_valid_two_digit_number (n : ℕ) : Prop :=
  let a := n / 10 in
  let b := n % 10 in
  10 ≤ n ∧ n < 100 ∧ (n - (a + b)) % 10 = 7

theorem count_valid_two_digit_numbers :
  finset.card (finset.filter is_valid_two_digit_number (finset.Icc 10 99)) = 10 :=
sorry

end count_valid_two_digit_numbers_l159_159386


namespace part_a_smallest_number_of_lines_part_b_smallest_number_of_lines_l159_159335

theorem part_a_smallest_number_of_lines (n : ℕ) : 
  (n * (n - 1)) / 2 ≥ 2022 → ∃ k : ℕ, k = 65 :=
sorry

theorem part_b_smallest_number_of_lines (n : ℕ) : 
  (n * (n - 1)) / 2 ≥ 2023 → ∃ k : ℕ, k = 65 :=
sorry

end part_a_smallest_number_of_lines_part_b_smallest_number_of_lines_l159_159335


namespace total_students_is_48_l159_159878

-- Definitions according to the given conditions
def boys'_row := 24
def girls'_row := 24

-- Theorem based on the question and the correct answer
theorem total_students_is_48 :
  boys'_row + girls'_row = 48 :=
by
  sorry

end total_students_is_48_l159_159878


namespace area_of_defined_region_eq_14_point_4_l159_159747

def defined_region (x y : ℝ) : Prop :=
  |5 * x - 20| + |3 * y + 9| ≤ 6

def region_area : ℝ :=
  14.4

theorem area_of_defined_region_eq_14_point_4 :
  (∃ (x y : ℝ), defined_region x y) → region_area = 14.4 :=
by
  sorry

end area_of_defined_region_eq_14_point_4_l159_159747


namespace probability_product_divisible_by_7_l159_159292

theorem probability_product_divisible_by_7 :
  let S := Finset.range 51 \ Finset.singleton 0 in
  let total_combinations := S.card.choose 2 in
  let multiples_of_7 := S.filter (λ x, x % 7 = 0) in
  let non_multiples_of_7 := S.filter (λ x, x % 7 ≠ 0) in
  let non_multiple_combinations := non_multiples_of_7.card.choose 2 in
  let prob_non_divisible := non_multiple_combinations / total_combinations.to_rat in
  let prob_divisible := 1 - prob_non_divisible in
  prob_divisible = 46 / 175 :=
by
  sorry

end probability_product_divisible_by_7_l159_159292


namespace vasya_birthday_is_thursday_l159_159732

def vasya_birthday_day_of_week (today_is_friday : Bool) (sunday_day_after_tomorrow : Bool) : String :=
  if today_is_friday && sunday_day_after_tomorrow then "Thursday" else "Unknown"

theorem vasya_birthday_is_thursday
  (today_is_friday : true)
  (sunday_day_after_tomorrow : true) : 
  vasya_birthday_day_of_week true true = "Thursday" := 
by
  -- assume today is Friday
  have h1 : today_is_friday = true := rfl
  -- assume Sunday is the day after tomorrow
  have h2 : sunday_day_after_tomorrow = true := rfl
  -- by our function definition, Vasya's birthday should be Thursday
  show vasya_birthday_day_of_week true true = "Thursday"
  sorry

end vasya_birthday_is_thursday_l159_159732


namespace number_of_true_propositions_l159_159932

def p := True -- "3 is an odd number" (true)
def q := True -- "3 is the smallest prime number" (true)

theorem number_of_true_propositions : (if p ∧ q then 1 else 0) + (if p ∨ q then 1 else 0) + (if ¬ p then 1 else 0) + (if ¬ q then 1 else 0) = 2 :=
by
  -- The proof steps can be added here, but are omitted according to instructions.
  sorry

end number_of_true_propositions_l159_159932


namespace locus_of_midpoint_l159_159921

theorem locus_of_midpoint (α β X Y : ℝ) (h₁ : α < β)
  (h₂ : β - α = 2)
  (h₃ : X = (α + β) / 2)
  (h₄ : Y = (α^2 + β^2) / 2) : 
  Y = X^2 + 1 := by
  suffices : (α + β)^2 + 4 = 2 * (2 * Y), from sorry
  sorry

end locus_of_midpoint_l159_159921


namespace similar_triangles_homothety_rotation_l159_159651

theorem similar_triangles_homothety_rotation :
  ∀ (A1 B1 C1 : ℝ × ℝ × ℝ) (A2 B2 C2 : ℝ × ℝ × ℝ),
    similar_triangles_in_space A1 B1 C1 A2 B2 C2
    → ¬ congruent_triangles A1 B1 C1 A2 B2 C2
    → ∃ (S : ℝ × ℝ × ℝ) (k : ℝ) (θ : ℝ) (axis : ℝ × ℝ × ℝ),
        homothety_and_rotation_exists A2 B2 C2 A1 B1 C1 S k θ axis :=
by
  sorry

end similar_triangles_homothety_rotation_l159_159651


namespace probability_two_red_two_blue_correct_l159_159343

noncomputable def num_ways_to_choose : ℕ → ℕ → ℕ :=
  λ n k, Nat.choose n k

noncomputable def probability_two_red_two_blue : ℚ :=
  let total_ways := num_ways_to_choose 20 4
  let ways_red := num_ways_to_choose 12 2
  let ways_blue := num_ways_to_choose 8 2
  (ways_red * ways_blue) / total_ways

theorem probability_two_red_two_blue_correct :
  probability_two_red_two_blue = 616 / 1615 :=
by
  sorry

end probability_two_red_two_blue_correct_l159_159343


namespace production_difference_total_weekly_production_l159_159359

noncomputable def actual_production (planned : Int) (adjustment : Int) : Int :=
  planned + adjustment

noncomputable def total_production (planned : Int) (adjustments : List Int) : Int :=
  planned * adjustments.length + (adjustments.foldl (· + ·) 0)

-- Conditions
def planned_daily_production : Int := 100
def daily_adjustments : List Int := [-1, 3, -2, 4, 7, -5, -10]

-- Question 1: Difference in production between the highest and lowest production days
def highest_adjustment : Int := daily_adjustments.foldl Int.max Int.min
def lowest_adjustment : Int := daily_adjustments.foldl Int.min Int.max
theorem production_difference : 
  actual_production planned_daily_production highest_adjustment -
  actual_production planned_daily_production lowest_adjustment = 17 := 
sorry

-- Question 2: Total production in the week
theorem total_weekly_production : 
  total_production planned_daily_production daily_adjustments = 696 := 
sorry

end production_difference_total_weekly_production_l159_159359


namespace white_water_addition_l159_159585

theorem white_water_addition :
  ∃ (W H I T E A R : ℕ), 
  W ≠ H ∧ W ≠ I ∧ W ≠ T ∧ W ≠ E ∧ W ≠ A ∧ W ≠ R ∧
  H ≠ I ∧ H ≠ T ∧ H ≠ E ∧ H ≠ A ∧ H ≠ R ∧
  I ≠ T ∧ I ≠ E ∧ I ≠ A ∧ I ≠ R ∧
  T ≠ E ∧ T ≠ A ∧ T ≠ R ∧
  E ≠ A ∧ E ≠ R ∧
  A ≠ R ∧
  W = 8 ∧ I = 6 ∧ P = 1 ∧ C = 9 ∧ N = 0 ∧
  (10000 * W + 1000 * H + 100 * I + 10 * T + E) + 
  (10000 * W + 1000 * A + 100 * T + 10 * E + R) = 169069 :=
by 
  sorry

end white_water_addition_l159_159585


namespace part_a_part_b_l159_159818

def is_nice (f : ℕ → ℕ) : Prop :=
∀ a b, nat.iterate f a b = f (a + b - 1)

variable (g : ℕ → ℕ)
variable (A : ℕ)

axiom g_nice : is_nice g
axiom A_condition : g (A + 2018) = g A + 1

theorem part_a (n : ℕ) (h : n ≥ A + 2) : g (n + (2017 ^ 2017)) = g n :=
sorry

theorem part_b (h : g (A + 1) ≠ g (A + 1 + (2017 ^ 2017))) (n : ℕ) (h0 : n < A) : g n = n + 1 :=
sorry

end part_a_part_b_l159_159818


namespace min_value_of_A_l159_159016

noncomputable def A (n : ℕ) (x : Fin n → ℝ) : ℝ :=
  let sin_sum := ∑ i, Real.sin (x i)
  let cos_sum := ∑ i, Real.cos (x i)
  (2 * sin_sum + cos_sum) * (sin_sum - 2 * cos_sum)

theorem min_value_of_A (n : ℕ) (x : Fin n → ℝ) : 
  ∃ x, A n x = - (5 * n^2) / 2 :=
sorry

end min_value_of_A_l159_159016


namespace min_value_a2_plus_b_minus_4_sq_l159_159948

theorem min_value_a2_plus_b_minus_4_sq (a b : ℝ) (f : ℝ → ℝ := λ x, x^2 + a*x + b - 3) :
  (∃ x ∈ set.Icc (1:ℝ) (2:ℝ), f x = 0) → a^2 + (b - 4)^2 = 2 :=
sorry

end min_value_a2_plus_b_minus_4_sq_l159_159948


namespace total_bricks_calculation_l159_159291

def bricks_in_row : Nat := 30
def rows_in_wall : Nat := 50
def number_of_walls : Nat := 2
def total_bricks_for_both_walls : Nat := 3000

theorem total_bricks_calculation (h1 : bricks_in_row = 30) 
                                      (h2 : rows_in_wall = 50) 
                                      (h3 : number_of_walls = 2) : 
                                      bricks_in_row * rows_in_wall * number_of_walls = total_bricks_for_both_walls :=
by
  sorry

end total_bricks_calculation_l159_159291


namespace ellipse_standard_form_l159_159505

theorem ellipse_standard_form (α : ℝ) 
  (x y : ℝ) 
  (hx : x = 5 * Real.cos α) 
  (hy : y = 3 * Real.sin α) : 
  (x^2 / 25) + (y^2 / 9) = 1 := 
by 
  sorry

end ellipse_standard_form_l159_159505


namespace polynomial_degree_and_terms_l159_159591

-- Definition for polynomial and its corresponding properties
def polynomial := x^2 + 2*x + 18

-- The degree of the polynomial should be 2
def degree_of_polynomial : ℕ := 2

-- The number of distinct terms in the polynomial should be 3
def number_of_terms : ℕ := 3

-- Proof statement for degree and number of terms
theorem polynomial_degree_and_terms : 
  ∃ (d n : ℕ), d = degree_of_polynomial ∧ n = number_of_terms :=
by
  exact ⟨2, 3, rfl, rfl⟩

end polynomial_degree_and_terms_l159_159591


namespace alice_wins_game_l159_159836

theorem alice_wins_game : 
  ∀ (n : ℕ) (coins: ℕ) (alice_first_move : coins = 1331) (valid_turns : ∀ turn, turn = 1 ∨ turn = turn - 1 ∨ turn = turn + 1),
  wins alice :=
sorry

end alice_wins_game_l159_159836


namespace solve_for_x_l159_159253

theorem solve_for_x (x : ℝ) : x^2 + 6 * x + 8 = -(x + 4) * (x + 6) ↔ x = -4 :=
by {
  sorry
}

end solve_for_x_l159_159253


namespace determine_power_of_two_dividing_product_of_orderings_l159_159845

/-- Given 10 singers where each singer either has no preference or wishes to perform right after another singer,
   compute the product of all possible nonzero values of valid singer orderings n, and prove that the largest 
   nonnegative integer k such that 2^k divides that product is 38. -/
theorem determine_power_of_two_dividing_product_of_orderings :
  let n_values := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      p := (n_values.map (λ x : ℕ, if x = 0 then 1 else x.factorial)).foldr (λ x acc, x * acc) 1
      largest_pow_two_divides_p := (nat.factorial 10).trailing_zeroes
  in largest_pow_two_divides_p = 38 :=
by sorry

end determine_power_of_two_dividing_product_of_orderings_l159_159845


namespace sin_cos_expr_l159_159462

open Real

theorem sin_cos_expr :
  ∀ (x : ℝ), -π < x ∧ x < 0 ∧ sin x + cos x = 1 / 5 →
    (sin x - cos x = -7 / 5) ∧ 
    ((3 * sin (x / 2) ^ 2 - 2 * sin (x / 2) * cos (x / 2) + cos (x / 2) ^ 2) /
      (tan x + 1 / tan x) = -132 / 125) :=
by
  intros x hx
  have h1 : sin x + cos x = 1 / 5 := hx.2.2
  have h2 : -π < x ∧ x < 0 := (hx.1, hx.2.1)
  sorry

end sin_cos_expr_l159_159462


namespace integer_points_on_segment_l159_159745

open Int

def is_integer_point (x y : ℝ) : Prop := ∃ (a b : ℤ), x = a ∧ y = b

def f (n : ℕ) : ℕ := 
  if 3 ∣ n then 2
  else 0

theorem integer_points_on_segment (n : ℕ) (hn : 0 < n) :
  (f n) = if 3 ∣ n then 2 else 0 := 
  sorry

end integer_points_on_segment_l159_159745


namespace value_of_abs_h_l159_159418

theorem value_of_abs_h (h : ℝ) : 
  (∃ r s : ℝ, (r + s = -4 * h) ∧ (r * s = -5) ∧ (r^2 + s^2 = 13)) → 
  |h| = (Real.sqrt 3) / 4 :=
by
  sorry

end value_of_abs_h_l159_159418


namespace hyperbola_properties_l159_159490

theorem hyperbola_properties (M : ℝ × ℝ)
  (hM : M = (2, -2))
  (h_asymptote : ∀ x y, (x^2 - 2 * y^2 = 2) → (x^2 - 2 * y^2) / (x^2 - a^2) = (2 * y^2) / (2 * b^2)) :
  ∃ (λ : ℝ), (∀ x y, x^2 - 2 * y^2 = λ) ∧ 
             (λ = -4) ∧ 
             ((x^2 / 4 - y^2 / 2 = 1) ∧ 
             (let e := real.sqrt (3/2)
             in e = real.sqrt (3/2)) ∧
             (y = x * real.sqrt (1/2) ∨ y = -x * real.sqrt (1/2))) :=
begin
  sorry
end

end hyperbola_properties_l159_159490


namespace construct_triangle_l159_159058

-- Defining the problem in Lean 4
variables (A' A'': ℝ) (s1 s2: ℝ) (ρ ρa: ℝ) (b c: ℝ)

-- Conditions (Point A, Plane S, Distances)
def A : ℝ × ℝ := (A', A'')
def S : ℝ × ℝ := (s1, s2)

-- Mathematical equivalence of proof goal
theorem construct_triangle 
  (h1: ρ > 0) 
  (h2: ρa > 0) 
  (h3: b > c) 
  (h4: (s1 - b) * (s2 - c) = ρ * ρa)
  (h5: (s2 - c) - (s1 - b) = b - c): Prop :=
∃ (B C: ℝ × ℝ), 
  ∃ (a b c: ℝ),
  A' = (B.1 + C.1) / 2 ∧ A'' = (B.2 + C.2) / 2 ∧
  a = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
  b = (A'.1 - B.1)^2 + (A''.2 - B.2)^2 ∧
  c = (A'.1 - C.1)^2 + (A''.2 - C.2)^2 ∧
  ρ = area_ABC Δ / s ∧
  ρa = area_ext_A Δ / s ∧
  (s - b) * (s - c) = ρ * ρa ∧
  (s - c) - (s - b) = b - c
-- skipping the proof
sorry

end construct_triangle_l159_159058


namespace amount_saved_per_person_l159_159225

-- Definitions based on the conditions
def original_price := 60
def discounted_price := 48
def number_of_people := 3
def discount := original_price - discounted_price

-- Proving that each person paid 4 dollars less.
theorem amount_saved_per_person : discount / number_of_people = 4 :=
by
  sorry

end amount_saved_per_person_l159_159225


namespace part1_part2_part3_l159_159081

-- Define the natural logarithm function
noncomputable def f (x : ℝ) := Real.log x

-- Part 1
theorem part1 (F g : ℝ → ℝ) (t : ℝ) (x : ℝ) (hl : ∃ l, F x = g x ∧ F'(x) = g'(x))
  (hF : F = λ x, t * f x) (hg : g = λ x, x^2 - 1) : 
  t = 2 :=
sorry

-- Part 2
theorem part2 (x : ℝ) (hx : 0 < x) : 
  |f x - x| > f x / x + 1 / 2 :=
sorry

-- Part 3
theorem part3 (a : ℝ) : 
  (∀ m ∈ Icc (0 : ℝ) (3/2), ∀ x ∈ Icc 1 (Real.exp 2), m * f x ≥ a + x) → 
  a ≤ -Real.exp 2 :=
sorry

end part1_part2_part3_l159_159081


namespace projection_eq_l159_159021

-- Make the notation easier for 3D vector
notation "⟨" x ", " y ", " z "⟩" => ![x, y, z]

def plane := { v : ℝ × ℝ × ℝ // 2 * v.1 + 3 * v.2 - v.3 = 0 }
def vector_v := (2, -3, 4)
def normal_vector := ⟨2, 3, -1⟩
def projection_onto_plane := ⟨23/7, -15/14, 47/14⟩

theorem projection_eq :
  ∃ p : plane, p.1 = projection_onto_plane :=
by
  sorry

end projection_eq_l159_159021


namespace sum_of_consecutive_integers_l159_159562

theorem sum_of_consecutive_integers (n : ℕ) (h : n * (n + 1) * (n + 2) * (n + 3) = 358800) : 
  n + (n + 1) + (n + 2) + (n + 3) = 98 :=
sorry

end sum_of_consecutive_integers_l159_159562


namespace multiplicative_inverse_of_AB_l159_159632

def A : ℕ := 222222
def B : ℕ := 476190
def N : ℕ := 189
def modulus : ℕ := 1000000

theorem multiplicative_inverse_of_AB :
  (A * B * N) % modulus = 1 % modulus :=
by
  sorry

end multiplicative_inverse_of_AB_l159_159632


namespace num_unique_m_values_l159_159621

theorem num_unique_m_values : 
  ∃ (s : Finset Int), 
  (∀ (x1 x2 : Int), x1 * x2 = 36 → x1 + x2 ∈ s) ∧ 
  s.card = 10 := 
sorry

end num_unique_m_values_l159_159621


namespace perpendicular_to_AB_l159_159331

open EuclideanGeometry

variables {A B C K L M N K' L' M' N' : Point}
variables {triangle_ABC : Triangle}
variables (h_triangle_ABC : IsAcuteTriangle triangle_ABC)
variables (segment_AC segment_AB segment_BC : Line)
variables (E F : Point)

-- Definitions related to the rectangles KLMN and K'L'M'N'
variables (rectangle_KLMN rectangle_K'L'M'N' : Rectangle)
variables (h_N_AC : N ∈ segment_AC)
variables (h_K_AB : K ∈ segment_AB)
variables (h_L_AB : L ∈ segment_AB)
variables (h_translate : TranslateParallel rectangle_KLMN AC rectangle_K'L'M'N')
variables (h_M'_BC : M' ∈ segment_BC)

-- Definitions of intersections
variables (intersection_CL'_AB : Intersection (LineThrough C L') segment_AB)
variables (intersection_AM_CB : Intersection (LineThrough A M) segment_CB)
variables (line_connecting_intersections : Line := LineThrough intersection_CL'_AB intersection_AM_CB)

-- The theorem to be proven
theorem perpendicular_to_AB :
  Perpendicular line_connecting_intersections segment_AB :=
sorry

end perpendicular_to_AB_l159_159331


namespace vovochka_correct_sum_cases_vovochka_min_difference_l159_159158

-- Part (a)
theorem vovochka_correct_sum_cases : 
  (∑ (a : ℕ) in finset.range 10, (∑ (b : ℕ) in finset.range (10 - a), 1)) ^ 3 = 244620 :=
sorry

-- Part (b)
theorem vovochka_min_difference : 
  ∃ (a b c x y z : ℕ), (a * 100 + b * 10 + c + x * 100 + y * 10 + z) - (a + x) * 100 - (b + y) * 10 - (c + z) = 1800 :=
sorry

end vovochka_correct_sum_cases_vovochka_min_difference_l159_159158


namespace coefficient_x6_in_polynomial_expansion_l159_159872

theorem coefficient_x6_in_polynomial_expansion :
  let polynomial := (2 - x + 3 * x^2)^5 in
  coefficient_of_x6 polynomial = 270 := by
  let polynomial := (2 - x + 3 * x^2)^5
  have h1 : polynomial = ∑ i in range(6+1), coeff_xn_in_expansion 5 (2, -x, 3*x^2) i
  sorry

end coefficient_x6_in_polynomial_expansion_l159_159872


namespace harvesting_days_l159_159987

theorem harvesting_days :
  (∀ (harvesters : ℕ) (days : ℕ) (mu : ℕ), 2 * 3 * (75 : ℕ) = 450) →
  (7 * 4 * (75 : ℕ) = 2100) :=
by
  sorry

end harvesting_days_l159_159987


namespace banana_cantaloupe_cost_l159_159904

theorem banana_cantaloupe_cost {a b c d : ℕ} 
  (h1 : a + b + c + d = 20) 
  (h2 : d = 2 * a)
  (h3 : c = a - b) : b + c = 5 :=
sorry

end banana_cantaloupe_cost_l159_159904


namespace stuffed_animals_count_l159_159602

theorem stuffed_animals_count
  (total_prizes : ℕ)
  (frisbees : ℕ)
  (yoyos : ℕ)
  (h1 : total_prizes = 50)
  (h2 : frisbees = 18)
  (h3 : yoyos = 18) :
  (total_prizes - (frisbees + yoyos) = 14) :=
by
  sorry

end stuffed_animals_count_l159_159602


namespace vovochkas_no_carry_pairs_eq_correct_pairs_vovochkas_smallest_difference_l159_159137

theorem vovochkas_no_carry_pairs_eq_correct_pairs :
  let valid_digit_pairs := Nat.range 10.map (λ n, 10 - n)
  let sum_valid_digit_pairs := valid_digit_pairs.sum
  sum_valid_digit_pairs = 55 →
  let no_carry_combinations := 81 * 55 * 55
  no_carry_combinations = 244620 :=
by
  let valid_digit_pairs := List.map (λ n, 10 - n) (List.range 10)
  have h_valid_digit_sum : valid_digit_pairs.sum = 55 := by sorry
  let no_carry_combinations := 81 * 55 * 55
  have h_no_carry : no_carry_combinations = 244620 := by sorry
  exact h_no_carry

theorem vovochkas_smallest_difference :
  let incorrect_cases := [1800]
  incorrect_cases.minimum = 1800 :=
by
  let differences := [900, 90, 990]
  have h_min_diff : List.minimum differences = some 90 := by sorry
  let incorrect_cases := List.map (λ diff, 20 * diff) differences
  have h_min_incorrect : incorrect_cases.minimum = some 1800 := by sorry
  exact h_min_incorrect

end vovochkas_no_carry_pairs_eq_correct_pairs_vovochkas_smallest_difference_l159_159137


namespace right_triangle_perimeter_area_ratio_l159_159862

theorem right_triangle_perimeter_area_ratio 
  (a b : ℝ) (h : a > 0 ∧ b > 0) 
  (hyp : ∀ c, c = Real.sqrt (a^2 + b^2))
  : (a + b + Real.sqrt (a^2 + b^2)) / (0.5 * a * b) = 5 → (∃! x y : ℝ, x + y + Real.sqrt (x^2 + y^2) / (0.5 * x * y) = 5) :=
by
  sorry   -- Proof is omitted as per instructions.

end right_triangle_perimeter_area_ratio_l159_159862


namespace triangle_properties_l159_159927

variables {a k m : Real} 

def sin_epsilon := m / (2 * k)
def cos_epsilon := (1 / (2 * k)) * Real.sqrt (4 * k ^ 2 - m ^ 2)
def b := 2 * Real.sqrt (k ^ 2 + a * (a - Real.sqrt (4 * k ^ 2 - m ^ 2)))
def c := 2 * Real.sqrt (k ^ 2 + (a / 2) * ((a / 2) - Real.sqrt (4 * k ^ 2 - m ^ 2)))
def gamma := Real.arcsin (m / b)
def beta := Real.arcsin (m / c)

theorem triangle_properties (a k m : Real) (h1 : k ≠ 0) (h2 : m ≠ 0) (h3 : 4 * k ^ 2 - m ^ 2 ≥ 0) :
  b = 2 * Real.sqrt (k ^ 2 + a * (a - Real.sqrt (4 * k ^ 2 - m ^ 2))) ∧
  c = 2 * Real.sqrt (k ^ 2 + (a / 2) * ((a / 2) - Real.sqrt (4 * k ^ 2 - m ^ 2))) ∧ 
  Real.sin gamma = m / b ∧ 
  Real.sin beta = m / c :=
by
  sorry

end triangle_properties_l159_159927


namespace average_first_last_l159_159085

theorem average_first_last (l : List ℤ) (h_l : l = [-3, 2, 5, 8, 11])
  (h1 : ∃ xs ys, xs.length = 1 ∧ ys.length = 3 ∧ l = xs ++ [11] ++ ys)
  (h2 : ∃ xs ys, xs.length = 3 ∧ ys.length = 1 ∧ l = xs ++ [-3] ++ ys)
  (h3 : ∃ xs ys zs, xs.length = 1 ∧ ys.length = 1 ∧ zs.length = 1 ∧ 
    l = xs ++ [5] ++ ys ++ [5] ++ zs) :
  (l.headI + l.getLast (by sorry)) / 2 = 6.5 :=
begin
  sorry
end

end average_first_last_l159_159085


namespace different_gcd_values_count_l159_159767

theorem different_gcd_values_count :
  let gcd_lcm_eq_prod (a b : ℕ) := Nat.gcd a b * Nat.lcm a b = a * b
  let prime_factors_360 := (2 ^ 3 * 3 ^ 2 * 5 ^ 1 : ℕ)
  (∃ a b : ℕ, gcd_lcm_eq_prod a b ∧ a * b = 360) →
  (∃ gcd_vals : Finset ℕ, gcd_vals = {1, 2, 3, 4, 6, 8, 12, 24} ∧ gcd_vals.card = 8) :=
begin
  sorry
end

end different_gcd_values_count_l159_159767


namespace triangle_inequality_proof_l159_159598

noncomputable def triangle_inequality 
(P : Point) (A B C : Point) 
(d_a d_b d_c : ℝ) 
(R_a R_b R_c : ℝ)
(sin_A sin_B sin_C : ℝ) : 
  Prop :=
3 * (d_a^2 + d_b^2 + d_c^2) ≥ (R_a * sin_A)^2 + (R_b * sin_B)^2 + (R_c * sin_C)^2

theorem triangle_inequality_proof
(P : Point) (A B C : Point) 
(d_a d_b d_c : ℝ) 
(R_a R_b R_c : ℝ)
(sin_A sin_B sin_C : ℝ) 
(h1 : d_a = distance_from P to (line_through B C))
(h2 : d_b = distance_from P to (line_through C A))
(h3 : d_c = distance_from P to (line_through A B))
(h4 : R_a = distance P A)
(h5 : R_b = distance P B)
(h6 : R_c = distance P C)
(h7 : sin_A = sin (angle BAC))
(h8 : sin_B = sin (angle ABC))
(h9 : sin_C = sin (angle BCA))
:
  3 * (d_a^2 + d_b^2 + d_c^2) ≥ (R_a * sin_A)^2 + (R_b * sin_B)^2 + (R_c * sin_C)^2 :=
by sorry

end triangle_inequality_proof_l159_159598


namespace sam_seashells_l159_159245

def seashells_problem := 
  let mary_seashells := 47
  let total_seashells := 65
  (total_seashells - mary_seashells) = 18

theorem sam_seashells :
  seashells_problem :=
by
  sorry

end sam_seashells_l159_159245


namespace minimum_value_expression_l159_159065

theorem minimum_value_expression
  (a b c d : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  let A := (a^2 + b^2) / (c * d) ^ 4 + 
           (b^2 + c^2) / (a * d) ^ 4 + 
           (c^2 + d^2) / (a * b) ^ 4 + 
           (d^2 + a^2) / (b * c) ^ 4 in
  A ≥ 64 :=
by
  sorry

end minimum_value_expression_l159_159065


namespace minimum_force_to_submerge_l159_159299

-- Definitions of given constants
def cube_density : ℝ := 400 -- kg/m^3
def water_density : ℝ := 1000 -- kg/m^3
def cube_volume_cm3 : ℝ := 10 -- cm^3
def gravity : ℝ := 10 -- m/s^2

-- Conversion factor from cm^3 to m^3
def cm3_to_m3 (v : ℝ) : ℝ := v * 10^(-6)

-- Given the conditions, the proof goal is to show that the minimum force required to submerge the cube is 0.06 N
theorem minimum_force_to_submerge : 
  let cube_volume_m3 := cm3_to_m3 cube_volume_cm3 in
  let cube_mass := cube_density * cube_volume_m3 in
  let cube_weight := cube_mass * gravity in
  let buoyant_force := water_density * cube_volume_m3 * gravity in
  buoyant_force - cube_weight = 0.06 := by
  sorry

end minimum_force_to_submerge_l159_159299


namespace car_balanced_by_cubes_l159_159396

variable (M Ball Cube : ℝ)

-- Conditions from the problem
axiom condition1 : M = Ball + 2 * Cube
axiom condition2 : M + Cube = 2 * Ball

-- Theorem to prove
theorem car_balanced_by_cubes : M = 5 * Cube := sorry

end car_balanced_by_cubes_l159_159396


namespace production_days_l159_159790

theorem production_days (n : ℕ) (h₁ : (50 * n + 95) / (n + 1) = 55) : 
    n = 8 := 
    sorry

end production_days_l159_159790


namespace at_most_one_root_l159_159941

noncomputable def isMonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ {x₁ x₂ : ℝ}, x₁ < x₂ → f x₁ ≤ f x₂

theorem at_most_one_root (f : ℝ → ℝ) (mono_inc : isMonotonicallyIncreasing f) :
  ∃! x, f x = 0 :=
begin
  sorry
end

end at_most_one_root_l159_159941


namespace triangle_angle_ratios_implies_reciprocal_sum_l159_159123

-- Given a triangle with angle ratios 4:2:1
noncomputable def x := 180 / 7

def angle_A := 4 * x
def angle_B := 2 * x
def angle_C := x

variable (a b c : ℝ)

-- Given Ptolemy's Theorem and relationships, prove the following:
theorem triangle_angle_ratios_implies_reciprocal_sum
  (hA : angle_A + angle_B + angle_C = 180)
  (ha : cos angle_A = (b^2 + c^2 - a^2) / (2 * b * c))
  (hb : cos angle_B = (a^2 + c^2 - b^2) / (2 * a * c))
  (hc : cos angle_C = (a^2 + b^2 - c^2) / (2 * a * b)) :
  (1 / a) + (1 / b) = 1 / c := sorry

end triangle_angle_ratios_implies_reciprocal_sum_l159_159123


namespace marble_probability_correct_l159_159345

noncomputable def marble_probability : ℚ :=
  let total_ways := (Nat.choose 20 4 : ℚ)
  let ways_two_red := (Nat.choose 12 2 : ℚ)
  let ways_two_blue := (Nat.choose 8 2 : ℚ)
  (ways_two_red * ways_two_blue) / total_ways

theorem marble_probability_correct : marble_probability = 56 / 147 :=
by
  -- Note: the proof is omitted as per instructions
  sorry

end marble_probability_correct_l159_159345


namespace prime_factorization_675_l159_159870

theorem prime_factorization_675 :
  ∃ (n h : ℕ), n > 1 ∧ n = 3 ∧ h = 225 ∧ 675 = (3^3) * (5^2) :=
by
  sorry

end prime_factorization_675_l159_159870


namespace a_is_perpendicular_to_b_l159_159511

def vector_perpendicular : Prop :=
  let a := (1, -2)
  let b := (2, 1)
  dot_product a b = 0

theorem a_is_perpendicular_to_b : vector_perpendicular :=
by sorry

/- Definitions used for the theorem. -/
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

end a_is_perpendicular_to_b_l159_159511


namespace total_employees_l159_159189

-- Given Conditions
variables (M W : ℕ) 
variables (white_cost black_cost total_cost : ℕ)
variables (same_number_of_men_and_women : M = W)
variables (costs : white_cost = 20 ∧ black_cost = 18)
variables (total_spent : total_cost = 660)
variables (womens_discount : 5)

-- Conclusion to be Proved
theorem total_employees:
  (2 * M * (white_cost + black_cost) = total_cost) → 
  (M = W) →
  (white_cost = 20) →
  (black_cost = 18) →
  (womens_discount = 5) →
  (total_cost = 660) →
  2 * (M + M) = 20 :=
by
  sorry

end total_employees_l159_159189


namespace gcd_values_360_l159_159781

theorem gcd_values_360 : ∃ d : ℕ, d = 11 ∧ ∀ a b : ℕ, a * b = 360 → ∃ (g : ℕ), g = gcd a b ∧ finite {g | g = gcd a b ∧ a * b = 360} ∧ card {g | g = gcd a b ∧ a * b = 360} = 11 :=
sorry

end gcd_values_360_l159_159781


namespace part_a_part_b_l159_159062

variable {A B C M N P O : Type}
variable [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C]
variable [AffineSpace ℝ M] [AffineSpace ℝ N] [AffineSpace ℝ P] [AffineSpace ℝ O]

-- Conditions: Points M, N, and P divide the sides in ratio p:q
variable (p q : ℝ)
variable (AM MB BN NC CP PA : ℝ)
variable (h_ratio1 : AM / MB = p / q)
variable (h_ratio2 : BN / NC = p / q)
variable (h_ratio3 : CP / PA = p / q)

-- Medians intersection condition
variable (median_intersection1 : IsCentroid ABC O)
variable (median_intersection2 : IsCentroid MNP O)
variable (median_intersection3 : IsCentroid (triangle_of_lines AN BP CM) O)

-- Proofs
theorem part_a : IsCentroid MNP O := by
  sorry

theorem part_b : IsCentroid (triangle_of_lines AN BP CM) O := by
  sorry

end part_a_part_b_l159_159062


namespace evaluate_expression_evaluate_fraction_l159_159550

theorem evaluate_expression (x y : ℕ) (hx : x = 3) (hy : y = 4) : 
  3 * x^3 + 4 * y^3 = 337 :=
by
  sorry

theorem evaluate_fraction (x y : ℕ) (hx : x = 3) (hy : y = 4) 
  (h : 3 * x^3 + 4 * y^3 = 337) :
  (3 * x^3 + 4 * y^3) / 9 = 37 + 4/9 :=
by
  sorry

end evaluate_expression_evaluate_fraction_l159_159550


namespace sum_A_C_eq_eight_l159_159004

theorem sum_A_C_eq_eight
  (A B C D : ℕ)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_set : A ∈ {2, 3, 4, 5} ∧ B ∈ {2, 3, 4, 5} ∧ C ∈ {2, 3, 4, 5} ∧ D ∈ {2, 3, 4, 5})
  (h_fraction : (A : ℚ) / B - (C : ℚ) / D = 1) :
  A + C = 8 :=
sorry

end sum_A_C_eq_eight_l159_159004


namespace num_distinct_x_intercepts_l159_159521

theorem num_distinct_x_intercepts : 
  let f := λ x : ℝ, (x - 3) * (x^2 + 3 * x + 2) in
  ∃ (xs : set ℝ), (∀ x ∈ xs, f x = 0) ∧ xs = {3, -1, -2} ∧ xs.card = 3 :=
by
  let f : ℝ → ℝ := λ x, (x - 3) * (x^2 + 3 * x + 2)
  have hxs : ∀ x ∈ {3, -1, -2}.to_finset, f x = 0 := by sorry
  have h_card : {3, -1, -2}.to_finset.card = 3 := by sorry
  use {3, -1, -2}
  split
  · exact hxs
  split
  · rfl
  · exact h_card

end num_distinct_x_intercepts_l159_159521


namespace vasya_birthday_l159_159724

/--
Vasya said the day after his birthday: "It's a pity that my birthday 
is not on a Sunday this year, because more guests would have come! 
But Sunday will be the day after tomorrow..."
On what day of the week was Vasya's birthday?
-/
theorem vasya_birthday (today : string)
  (h1 : today = "Friday")
  (h2 : ∀ day : string, day ≠ "Sunday" → Vasya's_birthday day) :
  Vasya's_birthday "Thursday" := by
  sorry

end vasya_birthday_l159_159724


namespace range_of_a_l159_159500

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x < - (Real.sqrt 3) / 3 ∨ x > (Real.sqrt 3) / 3 →
    a * (3 * x^2 - 1) > 0) →
  a > 0 :=
by
  sorry

end range_of_a_l159_159500


namespace find_m_if_parallel_l159_159516

-- Given vectors
def a (m : ℝ) : ℝ × ℝ := (2, m)
def b (m : ℝ) : ℝ × ℝ := (m, 2)

-- Parallel condition and the result that m must be -2 or 2
theorem find_m_if_parallel (m : ℝ) (h : ∃ k : ℝ, a m = (k * (b m).fst, k * (b m).snd)) : 
  m = -2 ∨ m = 2 :=
sorry

end find_m_if_parallel_l159_159516


namespace tan_alpha_eq_one_third_l159_159980

variable (α : ℝ)

theorem tan_alpha_eq_one_third (h : Real.tan (α + Real.pi / 4) = 2) : Real.tan α = 1 / 3 :=
sorry

end tan_alpha_eq_one_third_l159_159980


namespace original_price_of_cycle_l159_159815

variable (SP g P : ℝ)

theorem original_price_of_cycle (h₁ : SP = 1620) (h₂ : g = 8) : P = 1500 :=
by
  let gain_factor := 1 + g / 100
  have h₃ : SP = P * gain_factor := by sorry
  have h_gain_factor : gain_factor = 1.08 := by sorry
  have h₄ : 1620 = P * 1.08 := by sorry
  exact (calc
    P = 1620 / 1.08 : by sorry
    ... = 1500 : by sorry)

end original_price_of_cycle_l159_159815


namespace polygon_sides_from_exterior_angle_l159_159558

theorem polygon_sides_from_exterior_angle (E : ℝ) (h : E = 40) : 
  ∃ n : ℕ, n = 360 / E ∧ n = 9 := by
  have h1 : E ≠ 0 := by 
    linarith
  use 360 / E
  split
  case left =>
    rfl
  case right =>
    rw h
    norm_num

end polygon_sides_from_exterior_angle_l159_159558


namespace count_two_digit_numbers_with_perfect_square_digit_sum_eq_17_l159_159974

-- Define two-digit numbers, their digit sum, and condition for perfect squares
def is_two_digit_number (n : ℕ) : Prop := n >= 10 ∧ n < 100

def digit_sum (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  tens + units

def is_perfect_square_less_than_eq_25 (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m ∧ m <= 25

-- Define the main theorem to prove
theorem count_two_digit_numbers_with_perfect_square_digit_sum_eq_17 :
  (finset.filter (λ n, is_two_digit_number n ∧ digit_sum n ∈ {1, 4, 9, 16}) 
    (finset.range 100)).card = 17 :=
sorry

end count_two_digit_numbers_with_perfect_square_digit_sum_eq_17_l159_159974


namespace unique_10_tuple_l159_159435

theorem unique_10_tuple :
  ∃! (x : Fin 10 → ℝ), 
    (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 +
    (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + (x 6 - x 7)^2 +
    (x 7 - x 8)^2 + (x 8 - x 9)^2 + x 9^2 = 1 / 11 :=
begin
  sorry
end

end unique_10_tuple_l159_159435


namespace gcd_lcm_sum_l159_159305

variable (a b : ℕ)

-- Definition for gcd
def gcdOf (a b : ℕ) : ℕ := Nat.gcd a b

-- Definition for lcm
def lcmOf (a b : ℕ) : ℕ := Nat.lcm a b

-- Statement of the problem
theorem gcd_lcm_sum (h1 : a = 8) (h2 : b = 12) : gcdOf a b + lcmOf a b = 28 := by
  sorry

end gcd_lcm_sum_l159_159305


namespace min_force_to_submerge_cube_l159_159301

theorem min_force_to_submerge_cube :
  ∀ (V : ℝ) (ρ_cube ρ_water : ℝ) (g : ℝ),
  V = 10 * 10^(-6) →  -- volume in m^3
  ρ_cube = 400 →      -- density of cube in kg/m^3
  ρ_water = 1000 →    -- density of water in kg/m^3
  g = 10 →            -- acceleration due to gravity in m/s^2
  (ρ_water * V * g - ρ_cube * V * g = 0.06) :=
begin
  intros V ρ_cube ρ_water g,
  sorry
end

end min_force_to_submerge_cube_l159_159301


namespace line_equation_from_conditions_l159_159470

noncomputable def circle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

noncomputable def line (m x y : ℝ) : Prop := m * x - y + 1 - m = 0

noncomputable def is_point_inside_circle (x y : ℝ) : Prop := x = 1 ∧ y = 1

noncomputable def AP_condition (A_x A_y B_x B_y : ℝ) : Prop :=
  2 * (1 - A_x, -m * A_x + m) = (B_x - 1, m * B_x - m)

theorem line_equation_from_conditions (m : ℝ) :
  (∃ A_x A_y B_x B_y : ℝ, circle A_x A_y ∧ circle B_x B_y ∧ line m A_x A_y ∧ line m B_x B_y ∧ is_point_inside_circle 1 1 ∧ AP_condition A_x A_y B_x B_y) →
  (line m 0 0 ∨ line m 2 (-2)) :=
sorry

end line_equation_from_conditions_l159_159470


namespace calc_f_log2_20_l159_159457
noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom domain_R : ∀ x : ℝ, x ∈ ℝ
axiom functional_eq : ∀ x : ℝ, f (4 - x) + f x = 0
axiom specific_eq : ∀ x : ℝ, -2 < x ∧ x < 0 → f x = 2 ^ x

theorem calc_f_log2_20 : f (Real.log 20 / Real.log 2) = -4 / 5 := sorry

end calc_f_log2_20_l159_159457


namespace partition_condition_iff_l159_159231

theorem partition_condition_iff {n : ℕ} (X : Finset ℕ)
  (hX : X = Finset.range (n + 1) \ {0}) :
  (∃ (P : Finset (Finset ℕ)), (∀ (A ∈ P), ∃ (a b : ℕ), a ∈ X ∧ b ∈ X ∧ A = {a, b}) ∧ P.card = n / 2) ↔
  ∃ k : ℕ, n = 3 * 4 * k ∨ n = 3 * (4 * k + 1) := 
sorry

end partition_condition_iff_l159_159231


namespace circle_through_foci_l159_159607

variables {a b : ℝ} (h0 : a > 0) (h1 : b > 0) (h2 : a > b)
variables {x y : ℝ}

def ellipse (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

theorem circle_through_foci 
  (P : ℝ × ℝ) (B1 B2 F1 F2 : ℝ × ℝ)
  (h3 : ellipse P.1 P.2 a b)
  (h4 : P ≠ (a, 0))
  (h5 : P ≠ (-a, 0))
  (tangent_at_P : {L : ℝ × ℝ → Prop // ∀ pt : ℝ × ℝ, L pt ↔ (pt.fst * P.fst / a^2 + pt.snd * P.snd / b^2 = 1)})
  (tangent_at_A1 : {L : ℝ × ℝ → Prop // ∀ pt : ℝ × ℝ, L pt ↔ pt.fst = a})
  (tangent_at_A2 : {L : ℝ × ℝ → Prop // ∀ pt : ℝ × ℝ, L pt ↔ pt.fst = -a})
  (hB1 : ∃ pt : ℝ × ℝ, tangent_at_P.val pt ∧ tangent_at_A1.val pt ∧ pt = B1)
  (hB2 : ∃ pt : ℝ × ℝ, tangent_at_P.val pt ∧ tangent_at_A2.val pt ∧ pt = B2)
  (hC : ∃ C : ℝ × ℝ, C = ((B1.1 + B2.1) / 2, (B1.2 + B2.2) / 2)) :
  isCircleThroughDiameter B1 B2 F1 F2 := 
  sorry

end circle_through_foci_l159_159607


namespace gcd_values_count_l159_159760

theorem gcd_values_count (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 360) : 
  (∃ S : Finset ℕ, S.card = 12 ∧ ∀ d ∈ S, d = Nat.gcd a b) :=
by
  sorry

end gcd_values_count_l159_159760


namespace multiples_of_7_but_not_21_below_300_l159_159528

theorem multiples_of_7_but_not_21_below_300 : 
  {n : ℕ | n < 300 ∧ n % 7 = 0 ∧ n % 21 ≠ 0}.card = 28 :=
by
  sorry

end multiples_of_7_but_not_21_below_300_l159_159528


namespace distinct_m_values_count_l159_159623

theorem distinct_m_values_count :
  ∃ (m_values : Finset ℤ), (∀ x1 x2 : ℤ, x1 * x2 = 36 → m_values ∈ { x1 + x2 }) ∧ m_values.card = 10 :=
by
  sorry

end distinct_m_values_count_l159_159623


namespace sum_of_valid_ns_l159_159442

-- We state the conditions as hypotheses
def generates_impossible_postage (denoms : List ℕ) (limit : ℕ) : Prop :=
  ∀ n, n > limit → ∃ (a b c : ℕ), (a * denoms.head + b * denoms.tail.head! + c * denoms.tail.tail.head! = n)

def greatest_unformable_postage (denoms : List ℕ) (n : ℕ) (limit : ℕ) : Prop :=
  generates_impossible_postage denoms limit ∧ ¬ ∃ (a b c : ℕ), (a * denoms.head + b * denoms.tail.head! + c * denoms.tail.tail.head! = n)

-- List of denominations: 5, n, and n+1
def denominations (n: ℕ) : List ℕ := [5, n, n + 1]

-- Main statement
theorem sum_of_valid_ns :
  (∑ n in [24, 47], if greatest_unformable_postage (denominations n) n 91 then n else 0) = 71 := 
by 
  sorry

end sum_of_valid_ns_l159_159442


namespace union_of_A_and_B_l159_159634

open Set

-- Definitions for the conditions
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Statement of the theorem
theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 4} :=
by
  sorry

end union_of_A_and_B_l159_159634


namespace quadratic_always_positive_l159_159431

theorem quadratic_always_positive (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 2) * x - k + 4 > 0) ↔ -2 * Real.sqrt 3 < k ∧ k < 2 * Real.sqrt 3 := by
  sorry

end quadratic_always_positive_l159_159431


namespace find_p_plus_q_l159_159608

-- Definition of the set S'
def S' : Set ℕ := { n | 1 ≤ n ∧ n ≤ 2^30 ∧ (nat.popcount n = 2) }

-- Statement about the problem
theorem find_p_plus_q : 
  let p := 1 in
  let q := 62 in
  p + q = 63 :=
by
  sorry

end find_p_plus_q_l159_159608


namespace function_property_l159_159314

theorem function_property (f : ℝ → ℝ) 
  (h₀ : ∀ x > 1, ∀ y > 1, ∀ u > 0, ∀ v > 0, f (x^u * y^v) ≤ f(x)^(1 / (4 * u)) * f(y)^(1 / (4 * v))) : 
  ∃ C > 0, ∀ x > 1, f(x) = C^(1 / real.log x) :=
sorry

end function_property_l159_159314


namespace conic_section_is_ellipse_l159_159875

theorem conic_section_is_ellipse :
  ∀ (x y : ℝ), sqrt (x^2 + (y - 2)^2) + sqrt ((x - 6)^2 + (y + 4)^2) = 12 →
  ∃ (a b : ℝ × ℝ), a = (0, 2) ∧ b = (6, -4) ∧ 
  (∀ p : ℝ × ℝ, sqrt ((p.1 - a.1)^2 + (p.2 - a.2)^2) + sqrt ((p.1 - b.1)^2 + (p.2 - b.2)^2) = 12) ∧
  ∀ d : ℝ, d = sqrt ((6 - 0)^2 + (-4 - 2)^2) → 12 > d :=
begin
  sorry
end

end conic_section_is_ellipse_l159_159875


namespace gcd_values_count_l159_159757

theorem gcd_values_count (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 360) : 
  (∃ S : Finset ℕ, S.card = 12 ∧ ∀ d ∈ S, d = Nat.gcd a b) :=
by
  sorry

end gcd_values_count_l159_159757


namespace cos_max_in_interval_l159_159533

theorem cos_max_in_interval :
  ∀ m : ℝ, (∀ x, x ∈ set.Icc (Real.pi / 4) (2 * Real.pi / 3) → Real.cos x ≤ m) ↔ m ≥ Real.sqrt 2 / 2 := 
sorry

end cos_max_in_interval_l159_159533


namespace inverse_of_projection_matrix_l159_159610

def projection_matrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let norm_sq := v.1^2 + v.2^2
  let u := (⟨v.1 / real.sqrt norm_sq, v.2 / real.sqrt norm_sq⟩ : ℝ × ℝ)
  Matrix.mul (Matrix.ofVec (λ _, u)) (Matrix.transpose (Matrix.ofVec (λ _, u)))

theorem inverse_of_projection_matrix
  (v : ℝ × ℝ)
  (v_eq : v = (⟨4, 5⟩ : ℝ × ℝ)) :
  ¬ ∃ inv : Matrix (Fin 2) (Fin 2) ℝ, inv * (projection_matrix v) = (1 : Matrix (Fin 2) (Fin 2) ℝ) :=
by
  have h : projection_matrix v = (⟨⟨1/41 * 16, 1/41 * 20⟩, ⟨1/41 * 20, 1/41 * 25⟩⟩ : Matrix (Fin 2) (Fin 2) ℝ),
    from sorry -- Here you can compute the matrix manually or use a library function.
  have det_zero: Matrix.det (projection_matrix v) = 0,
    from sorry -- Compute that the determinant is zero.
  rw Matrix.det at det_zero,
  intro h,
  cases h with inv h_inv,
  simp [Matrix.det_smul, Matrix.det_zero, det_zero] at h_inv,
  contradiction

end inverse_of_projection_matrix_l159_159610


namespace isosceles_triangle_of_trigonometric_identity_l159_159122

theorem isosceles_triangle_of_trigonometric_identity
    (A B C : ℝ)
    (h1 : 2 * cos B * sin A = sin C)
    (h2 : -real.pi < A - B)
    (h3 : A - B < real.pi) :
    A = B ∨ A = C ∨ B = C := sorry

end isosceles_triangle_of_trigonometric_identity_l159_159122


namespace negative_coefficient_expanded_polynomial_l159_159649

theorem negative_coefficient_expanded_polynomial :
  ∃ (p : ℕ → ℤ), (∀ n : ℕ, p n = (coeff (x^2 - x + 1)^2014 n)) ∧ (p 1 = -2014) ∧ (p 1 < 0) :=
by
  sorry

end negative_coefficient_expanded_polynomial_l159_159649


namespace avg_rate_of_change_sq_1_2_l159_159262

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the average rate of change function
def avg_rate_of_change (f : ℝ → ℝ) (a b : ℝ) : ℝ := (f b - f a) / (b - a)

-- State the theorem to be proved
theorem avg_rate_of_change_sq_1_2 : avg_rate_of_change f 1 2 = 3 :=
by
  sorry

end avg_rate_of_change_sq_1_2_l159_159262


namespace M_gt_N_l159_159053

noncomputable def M : ℝ := ∫ x in -1..1, |x|
noncomputable def N : ℝ := cos (real.pi / 6) - sin (real.pi / 6)

theorem M_gt_N : M > N := by
  sorry

end M_gt_N_l159_159053


namespace area_of_square_with_diagonal_40_l159_159796

theorem area_of_square_with_diagonal_40 {d : ℝ} (h : d = 40) : ∃ A : ℝ, A = 800 :=
by
  sorry

end area_of_square_with_diagonal_40_l159_159796


namespace most_cost_effective_plan_l159_159281

def fare (x : ℕ) : ℝ :=
  if x ≤ 3 then 5
  else if x ≤ 10 then 1.2 * x + 1.4
  else 1.8 * x - 4.6

theorem most_cost_effective_plan :
  let plan1 := fare 30,
      plan2 := 2 * fare 15,
      plan3 := 3 * fare 10
  in plan3 < plan2 ∧ plan2 < plan1 :=
by
  sorry

end most_cost_effective_plan_l159_159281


namespace solve_equation_l159_159038

theorem solve_equation (x : ℂ) : 
  (15 * x - x^2) / (x + 1) * (x + (15 - x) / (x + 1)) = 30 ↔ 
  x = 0 ∨ x = 15 ∨ x = (1 + complex.I * complex.sqrt 7) / 2 ∨ x = (1 - complex.I * complex.sqrt 7) / 2 :=
sorry

end solve_equation_l159_159038


namespace projection_matrix_inverse_zero_l159_159612

theorem projection_matrix_inverse_zero (u : ℝ) (v : ℝ) (Q : Matrix (Fin 2) (Fin 2) ℝ) (h1 : u = 4) (h2 : v = 5)
  (h3 : Q = (Matrix.vecCons (Matrix.vecCons (4 / Real.sqrt 41) (5 / Real.sqrt 41) Matrix.vecEmpty)
                            (Matrix.vecCons (4 / Real.sqrt 41) (5 / Real.sqrt 41) Matrix.vecEmpty)).mul
              (Matrix.vecCons (Matrix.vecCons (4 / Real.sqrt 41) (5 / Real.sqrt 41) Matrix.vecEmpty)
                              (Matrix.vecCons (4 / Real.sqrt 41) (5 / Real.sqrt 41) Matrix.vecEmpty))) :
  Q.det = 0 → Q⁻¹ = 0 :=
by
  sorry

end projection_matrix_inverse_zero_l159_159612


namespace Janette_jerky_count_l159_159186

theorem Janette_jerky_count (n b l d r t : ℕ) 
    (h1 : n = 5)
    (h2 : b = 1)
    (h3 : l = 1)
    (h4 : d = 2)
    (h5 : r = 10)
    (h6 : (r * 2 + (b + l + d) * n) = t) : t = 40 := 
by
  have h7 : (1 + 1 + 2) * 5 = 20 := by
    calc (1 + 1 + 2) * 1 = 4 * 5
  have h8 : r * 2 = 20 := by
    calc 10 * 2
  have h9 : (20 + h7) = 40 := by sorry
  exact h9 sorry

end Janette_jerky_count_l159_159186


namespace number_of_younger_siblings_l159_159641

-- Definitions based on the problem conditions
def Nicole_cards : ℕ := 400
def Cindy_cards : ℕ := 2 * Nicole_cards
def Combined_cards : ℕ := Nicole_cards + Cindy_cards
def Rex_cards : ℕ := Combined_cards / 2
def Rex_remaining_cards : ℕ := 150
def Total_shares : ℕ := Rex_cards / Rex_remaining_cards
def Rex_share : ℕ := 1

-- The theorem to prove how many younger siblings Rex has
theorem number_of_younger_siblings :
  Total_shares - Rex_share = 3 :=
  by
    sorry

end number_of_younger_siblings_l159_159641


namespace max_value_quadratic_function_l159_159553

noncomputable def quadratic_function (x : ℝ) : ℝ := -3 * x^2 + 6 * x + 2

theorem max_value_quadratic_function : ∃ x : ℝ, quadratic_function x = 5 :=
by
  use 1
  show quadratic_function 1 = 5
  calc
    quadratic_function 1 = -3 * 1^2 + 6 * 1 + 2 : rfl
                      ... = 5 : by norm_num

end max_value_quadratic_function_l159_159553


namespace exists_infinite_n_divides_2_sigma_n_sub_1_l159_159783

theorem exists_infinite_n_divides_2_sigma_n_sub_1 :
  ∃ (f : ℕ → ℕ), (∀ k, (f k ∣ 2^((∑ d in (finset.filter (λ x, x ∣ (f k)) (finset.range ((f k) + 1)))), - 1))) :=
sorry

end exists_infinite_n_divides_2_sigma_n_sub_1_l159_159783


namespace gcd_values_count_l159_159758

theorem gcd_values_count (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 360) : 
  (∃ S : Finset ℕ, S.card = 12 ∧ ∀ d ∈ S, d = Nat.gcd a b) :=
by
  sorry

end gcd_values_count_l159_159758


namespace variance_of_data_set_l159_159946

-- Define the data set
def data_set : List ℝ := [10, 6, 8, 5, 6]

-- Define the function to calculate the mean of a list of real numbers
noncomputable def mean (l : List ℝ) : ℝ :=
  (l.sum) / l.length

-- Define the function to calculate the variance of a list of real numbers
noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  in (l.map (λ x, (x - m) ^ 2)).sum / l.length

-- The main theorem to be proved
theorem variance_of_data_set : variance data_set = 16 / 5 := by
  sorry

end variance_of_data_set_l159_159946


namespace dressing_children_ways_l159_159048

/-- Four children are playing a game in a circle. There are four different colors of clothes available 
    (unlimited number of each color). Two adjacent children must wear clothes of different colors.
    The number of different ways to dress the children considering only the difference in colors is 48. 
 -/
theorem dressing_children_ways : 
  let colors := {1, 2, 3, 4}, children := {A, B, C, D : Fin 4}
  ∃ (dressings : Fin 4 → Fin 4), 
    (∀ (i j : Fin 4), i ≠ j → dressings i ≠ dressings j) ∧ 
    ∃ n ways, n = 48 :=
sorry

end dressing_children_ways_l159_159048


namespace coprime_gcd_power_sub_l159_159615

open Nat

theorem coprime_gcd_power_sub {a b m n : ℕ} (hab_coprime : coprime a b) :
  gcd (a^n - b^n) (a^m - b^m) = a^(gcd m n) - b^(gcd m n) :=
  sorry

end coprime_gcd_power_sub_l159_159615


namespace seq_formula_l159_159509

theorem seq_formula (a : ℕ → ℕ) (h₀ : a 1 = 2) (h₁ : ∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n - 1) :
  ∀ n : ℕ, 0 < n → a n = 2 ^ (n - 1) + 1 := 
by 
  sorry

end seq_formula_l159_159509


namespace equation_of_directrix_l159_159673

theorem equation_of_directrix (a : ℝ) (h : a = 2) : 
  ∀ x : ℝ, y = a * x^2 → y = -1 / (4 * a) := 
by
  intros x hp
  rw [h] at *
  have ha : a = 2 := h
  calc
    y = 2 * x ^ 2 := by sorry
       ... = a * x ^ 2 := by rw [ha]
       ... = -1 / (4 * 2) := by sorry
       ... = -1 / 8 := by norm_num

end equation_of_directrix_l159_159673


namespace parallel_line_slope_l159_159874

theorem parallel_line_slope (x y : ℝ) :
  (∃ b : ℝ, y = (1/2) * x - b) →
  (∀ m n : ℝ, 3 * m - 6 * n = 21 → (1 / 2)) :=
by
  sorry

end parallel_line_slope_l159_159874


namespace find_n_pi_l159_159639

def shed_side_length : ℝ := 10
def rope_attachment_position : ℝ := 2
def rope_length : ℝ := 14
def reachable_area (s : ℝ) (p : ℝ) (l : ℝ) : ℝ :=
  let full_circle_area := π * l^2
  let quarter_circle_area r := π * r^2 / 4
  let three_quarters_area := (3 / 4) * full_circle_area
  let small_quarters_area := 2 * quarter_circle_area (l - p)
  three_quarters_area + small_quarters_area

theorem find_n_pi :
  reachable_area shed_side_length rope_attachment_position rope_length = 155 * π :=
by sorry

end find_n_pi_l159_159639


namespace min_sum_fractions_l159_159195

namespace ProofProblem

variable {A B C D : ℕ}

/-- 
Prove that the minimum value of the sum of fractions 
formed by selecting four distinct digits from the set 
{0, 1, 2, 3, 4, 5, 6, 7, 8, 9} is 1/8, with the 
condition that denominators cannot be zero.
--/
theorem min_sum_fractions :
  A != B ∧ A != C ∧ A != D ∧ B != C ∧ B != D ∧ C != D ∧
  A ∈ {0,1,2,3,4,5,6,7,8,9} ∧ B ∈ {0,1,2,3,4,5,6,7,8,9} ∧
  C ∈ {0,1,2,3,4,5,6,7,8,9} ∧ D ∈ {0,1,2,3,4,5,6,7,8,9} ∧
  B ≠ 0 ∧ D ≠ 0 
  → ((A:B:ℚ) + (C:D:ℚ) = 1 / 8) := 
begin
  sorry
end

end ProofProblem

end min_sum_fractions_l159_159195


namespace vasya_birthday_l159_159729

/--
Vasya said the day after his birthday: "It's a pity that my birthday 
is not on a Sunday this year, because more guests would have come! 
But Sunday will be the day after tomorrow..."
On what day of the week was Vasya's birthday?
-/
theorem vasya_birthday (today : string)
  (h1 : today = "Friday")
  (h2 : ∀ day : string, day ≠ "Sunday" → Vasya's_birthday day) :
  Vasya's_birthday "Thursday" := by
  sorry

end vasya_birthday_l159_159729


namespace arithmetic_sequence_a7_l159_159587

theorem arithmetic_sequence_a7 (a : ℕ → ℤ) (h1 : a 1 = 3) (h3 : a 3 = 5) (h_arith : ∀ n : ℕ, a (n + 1) = a n + (a 2 - a 1)) : a 7 = 9 :=
by
  sorry

end arithmetic_sequence_a7_l159_159587


namespace a_seq_formula_T_seq_sum_l159_159476

-- Definition of the sequence \( \{a_n\} \)
def a_seq (n : ℕ) (p : ℤ) : ℤ := 2 * n + 5

-- Condition: Sum of the first n terms \( s_n = n^2 + pn \)
def s_seq (n : ℕ) (p : ℤ) : ℤ := n^2 + p * n

-- Condition: \( \{a_2, a_5, a_{10}\} \) form a geometric sequence
def is_geometric (a2 a5 a10 : ℤ) : Prop :=
  a2 * a10 = a5 * a5

-- Definition of the sequence \( \{b_n\} \)
def b_seq (n : ℕ) (p : ℤ) : ℚ := 1 + 5 / (a_seq n p * a_seq (n + 1) p)

-- Function to find the sum of first n terms of \( \{b_n\} \)
def T_seq (n : ℕ) (p : ℤ) : ℚ :=
  n + 5 * (1 / (7 : ℚ) - 1 / (2 * n + 7 : ℚ)) + n / (14 * n + 49 : ℚ)

theorem a_seq_formula (p : ℤ) : ∀ n, a_seq n p = 2 * n + 5 :=
by
  sorry

theorem T_seq_sum (p : ℤ) : ∀ n, T_seq n p = (14 * n^2 + 54 * n) / (14 * n + 49) :=
by
  sorry

end a_seq_formula_T_seq_sum_l159_159476


namespace rationalize_denominator_l159_159237

theorem rationalize_denominator :
  let A := 9
  let B := 7
  let C := -18
  let D := 0
  let S := 2
  let F := 111
  (A + B + C + D + S + F = 111) ∧ 
  (
    (1 / (Real.sqrt 5 + Real.sqrt 6 + 2 * Real.sqrt 2)) * 
    ((Real.sqrt 5 + Real.sqrt 6) - 2 * Real.sqrt 2) * 
    (3 - 2 * Real.sqrt 30) / 
    (3^2 - (2 * Real.sqrt 30)^2) = 
    (9 * Real.sqrt 5 + 7 * Real.sqrt 6 - 18 * Real.sqrt 2) / 111
  ) := by
  sorry

end rationalize_denominator_l159_159237


namespace dot_product_AB_BC_l159_159597

variable (AB AC : ℝ × ℝ)

def BC (AB AC : ℝ × ℝ) : ℝ × ℝ := (AC.1 - AB.1, AC.2 - AB.2)

def dot_product (u v : ℝ × ℝ) : ℝ := (u.1 * v.1) + (u.2 * v.2)

theorem dot_product_AB_BC :
  ∀ (AB AC : ℝ × ℝ), AB = (2, 3) → AC = (3, 4) →
  dot_product AB (BC AB AC) = 5 :=
by
  intros
  unfold BC
  unfold dot_product
  sorry

end dot_product_AB_BC_l159_159597


namespace three_digit_number_formed_by_last_three_digits_l159_159391

theorem three_digit_number_formed_by_last_three_digits : 
  let sequence := (list.nat.filter (λ n, (digit.nth n 0 = some 2)) (list.range (10^5))) in
    let digits_count := sequence.foldl (λ (sum, n), sum + (digit.length n)) 0 in
    digits_count >= 1500 ∧ (sequence.nth_digit 1498) = some 2
    → (sequence.nth_digit 1499) = some 9
    → (sequence.nth_digit 1500) = some 4
    → true :=
by
  sorry

end three_digit_number_formed_by_last_three_digits_l159_159391


namespace brick_in_box_probability_l159_159236

theorem brick_in_box_probability :
  let S := Finset.range 2014.succ
  let select_without_replacement (n : ℕ) (s : Finset ℕ) : Finset (Finset ℕ) := s.powerset.filter (λ t, t.card = n)
  let a := select_without_replacement 3 S
  let remaining_set := S \ a
  let b := select_without_replacement 3 remaining_set
  let a_sorted := a.sort (<)
  let b_sorted := b.sort (<)
  a_sorted[0] < b_sorted[0] ∧ a_sorted[1] < b_sorted[1] ∧ a_sorted[2] < b_sorted[2] ->
  (((a_sorted[0], a_sorted[1], a_sorted[2]), (b_sorted[0], b_sorted[1], b_sorted[2])) ∈ a.product b).card.toRat / ((Finset.card a) * (Finset.card b)).toRat = 1/4 := sorry

end brick_in_box_probability_l159_159236


namespace unattainable_value_l159_159041

theorem unattainable_value : ∀ x : ℝ, x ≠ -4/3 → (y = (2 - x) / (3 * x + 4) → y ≠ -1/3) :=
by
  intro x hx h
  rw [eq_comm] at h
  sorry

end unattainable_value_l159_159041


namespace solve_for_y_l159_159544

variables (x y : ℤ)

theorem solve_for_y (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 :=
by
  hint sorry

end solve_for_y_l159_159544


namespace x_pow_12_eq_one_l159_159540

theorem x_pow_12_eq_one (x : ℝ) (h : x + 1/x = 2) : x^12 = 1 :=
sorry

end x_pow_12_eq_one_l159_159540


namespace line_parallel_or_in_plane_l159_159633

theorem line_parallel_or_in_plane {d n : ℝ × ℝ × ℝ} (hd : d = (6, 2, 3)) (hn : n = (-1, 3, 0)) : 
  d.1 * n.1 + d.2 * n.2 + d.3 * n.3 = 0 → 
  (let l_in_plane_or_parallel := "line l is in plane α or parallel to it" in 
  true) :=
by {
  intros,
  sorry
}

end line_parallel_or_in_plane_l159_159633


namespace Vovochka_correct_pairs_count_l159_159146

def no_carry_pairs_count (digit_sum: ℕ → ℕ → Prop) : ℕ :=
  let count_pairs (lim: ℕ) : ℕ := (lim * (lim + 1)) / 2
  let digit_valid_pairs : ℕ := count_pairs 9
  (digit_valid_pairs * digit_valid_pairs) * 81

def digit_sum (x y: ℕ) : Prop := (x + y ≤ 9)

theorem Vovochka_correct_pairs_count :
  no_carry_pairs_count digit_sum = 244620 := by
  sorry

end Vovochka_correct_pairs_count_l159_159146


namespace total_participants_l159_159288

theorem total_participants
  (F M : ℕ) 
  (half_female_democrats : F / 2 = 125)
  (one_third_democrats : (F + M) / 3 = (125 + M / 4))
  : F + M = 1750 :=
by
  sorry

end total_participants_l159_159288


namespace volume_of_pyramid_l159_159939

-- Definitions from the problem conditions:
def base_edge_length := 3
def side_edge_length := 5
def base_area := 6 * (1 / 2 * base_edge_length * (base_edge_length * Real.sqrt 3 / 2))
def height := Real.sqrt (side_edge_length^2 - base_edge_length^2)

-- Proof problem statement: Proving the volume of the pyramid is 18 * Real.sqrt 3
theorem volume_of_pyramid : 
  (1 / 3) * base_area * height = 18 * Real.sqrt 3 := sorry

end volume_of_pyramid_l159_159939


namespace solution_correct_l159_159328

noncomputable def solve_system : set (ℝ × ℝ × ℝ) :=
{
  (0, 0, 0),
  ((3 + Real.sqrt 5) / 2, 1, 2 + Real.sqrt 3),
  ((3 + Real.sqrt 5) / 2, 1, 2 - Real.sqrt 3),
  ((3 - Real.sqrt 5) / 2, 1, 2 + Real.sqrt 3),
  ((3 - Real.sqrt 5) / 2, 1, 2 - Real.sqrt 3)
}

theorem solution_correct (x y z : ℝ) :
  (x + y = 5 * x * y / (1 + x * y)) ∧ 
  (y + z = 6 * y * z / (1 + y * z)) ∧ 
  (z + x = 7 * z * x / (1 + z * x)) →
  (x, y, z) ∈ solve_system :=
sorry

end solution_correct_l159_159328


namespace tan_sum_l159_159101

theorem tan_sum (x y : ℝ) 
  (h1 : sin x + sin y = 85 / 65) 
  (h2 : cos x + cos y = 84 / 65) : 
  tan x + tan y = 717 / 143 :=
by 
  sorry

end tan_sum_l159_159101


namespace least_positive_difference_l159_159654

def sequenceA : List ℕ := [3, 9, 27, 81]

def sequenceB : List ℕ := [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195]

theorem least_positive_difference :
  ∃ d, d > 0 ∧ (∀ a ∈ sequenceA, ∀ b ∈ sequenceB, abs (a - b) ≥ d) ∧
       (∀ d', d' > 0 → (∀ a ∈ sequenceA, ∀ b ∈ sequenceB, abs (a - b) ≥ d') → d ≤ d') ∧ d = 3 := 
by {
  sorry
}

end least_positive_difference_l159_159654


namespace total_loads_washed_l159_159831

theorem total_loads_washed (a b : ℕ) (h1 : a = 8) (h2 : b = 6) : a + b = 14 :=
by
  sorry

end total_loads_washed_l159_159831


namespace solve_x_l159_159252

theorem solve_x (x : ℝ) (h : x^2 + 6 * x + 8 = -(x + 4) * (x + 6)) : 
  x = -4 := 
by
  sorry

end solve_x_l159_159252


namespace decrease_percent_in_revenue_l159_159323

theorem decrease_percent_in_revenue (T C : ℝ) :
  let original_revenue := T * C in
  let new_revenue := (0.70 * T) * (1.10 * C) in
  let decrease_in_revenue := original_revenue - new_revenue in
  let decrease_percent := (decrease_in_revenue / original_revenue) * 100 in
  decrease_percent = 23 :=
by
  sorry

end decrease_percent_in_revenue_l159_159323


namespace domain_of_function_l159_159483

theorem domain_of_function (x : ℝ) (h : 0 < x ∧ x < π) :
  (sin x > 1/2 ∧ 1/2 - cos x ≥ 0) ↔ (π/3 ≤ x ∧ x < 5*π/6) :=
by
  sorry

end domain_of_function_l159_159483


namespace max_possible_min_diff_l159_159504

def given_numbers : List ℝ := [10, 6, 13, 4, 18]

def circular_min_diff (l : List ℝ) : ℝ :=
  let circular_pairs := List.zip l (List.dropLast l ++ [List.head l])
  List.minimum (circular_pairs.map (λ ⟨a, b⟩, |a - b|))

theorem max_possible_min_diff : circular_min_diff given_numbers = 9 := sorry

end max_possible_min_diff_l159_159504


namespace range_of_f_l159_159030

noncomputable def f (x : ℝ) : ℝ := (3 * x + 8) / (x - 4)

theorem range_of_f : set.range f = set.univ \ {3} :=
by
  sorry

end range_of_f_l159_159030


namespace vovochkas_no_carry_pairs_eq_correct_pairs_vovochkas_smallest_difference_l159_159138

theorem vovochkas_no_carry_pairs_eq_correct_pairs :
  let valid_digit_pairs := Nat.range 10.map (λ n, 10 - n)
  let sum_valid_digit_pairs := valid_digit_pairs.sum
  sum_valid_digit_pairs = 55 →
  let no_carry_combinations := 81 * 55 * 55
  no_carry_combinations = 244620 :=
by
  let valid_digit_pairs := List.map (λ n, 10 - n) (List.range 10)
  have h_valid_digit_sum : valid_digit_pairs.sum = 55 := by sorry
  let no_carry_combinations := 81 * 55 * 55
  have h_no_carry : no_carry_combinations = 244620 := by sorry
  exact h_no_carry

theorem vovochkas_smallest_difference :
  let incorrect_cases := [1800]
  incorrect_cases.minimum = 1800 :=
by
  let differences := [900, 90, 990]
  have h_min_diff : List.minimum differences = some 90 := by sorry
  let incorrect_cases := List.map (λ diff, 20 * diff) differences
  have h_min_incorrect : incorrect_cases.minimum = some 1800 := by sorry
  exact h_min_incorrect

end vovochkas_no_carry_pairs_eq_correct_pairs_vovochkas_smallest_difference_l159_159138


namespace limit_proven_l159_159798

theorem limit_proven (f : ℝ → ℝ) (a L : ℝ) (h_lim : ∀ε > 0, ∃ δ > 0, ∀ x, 0 < |x - a| ∧ |x - a| < δ → |f x - L| < ε) :
  f = (λ x => (2 * x^2 + 15 * x + 7) / (x + 7)) ∧ a = -7 ∧ L = -13 :=
begin
  -- Context setup
  assume f a L ε hε,
  -- δ(ε) definition and Cauchy limit argument
  use (ε / 2),
  split,
  exact (ε / 2),
  assume x hx,
  split,
  linarith,
  rw abs_of_pos hε,
  -- Simplifications following the solution in step 3 to reach |2x + 14| < ε
  -- Prove the final implication
  sorry -- Placeholder for the proof
end

end limit_proven_l159_159798


namespace problem1_problem2_l159_159968

variables {x : ℝ} {a b c A B C : ℝ}
def m : ℝ × ℝ := (2 * real.sqrt 3 * real.sin (x / 4), 2)
def n : ℝ × ℝ := (real.cos (x / 4), real.cos (x / 4) ^ 2)
def dot_prod (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def f (x : ℝ) : ℝ := dot_prod m n

-- Prove that for dot product of m and n being 2,
-- the value of cos(x + π / 3) is 1 / 2
theorem problem1 (h : dot_prod m n = 2) : real.cos (x + real.pi / 3) = 1 / 2 := sorry

-- Prove that the range of f(A) is (2, 3) given (2a - c)cos B = bcos C
theorem problem2 (h1 : dot_prod m n = f A)
  (h2 : (2 * a - c) * real.cos B = b * real.cos C)
  (h3 : A + B + C = real.pi) : 2 < f A ∧ f A < 3 := sorry

end problem1_problem2_l159_159968


namespace game_A_vs_game_B_l159_159811

-- Define the problem in Lean 4
theorem game_A_vs_game_B (p_head : ℝ) (p_tail : ℝ) (independent : Prop)
  (prob_A : ℝ) (prob_B : ℝ) (delta : ℝ) :
  p_head = 3/4 → p_tail = 1/4 → independent →
  prob_A = (binom 4 3) * (p_head ^ 3) * (p_tail ^ 1) + (p_head ^ 4) →
  prob_B = ((p_head ^ 2 + p_tail ^ 2) ^ 2) →
  delta = prob_A - prob_B →
  delta = 89/256 :=
by
  intros hph hpt hind hpa hpb hdelta
  rw [hph, hpt, hpa, hpb, hdelta]
  sorry

end game_A_vs_game_B_l159_159811


namespace eq_inf_solutions_l159_159900

theorem eq_inf_solutions (a b : ℝ) : 
    (∀ x : ℝ, 4 * (3 * x - a) = 3 * (4 * x + b)) ↔ b = -(4 / 3) * a := by
  sorry

end eq_inf_solutions_l159_159900


namespace unattainable_value_l159_159040

theorem unattainable_value : ∀ x : ℝ, x ≠ -4/3 → (y = (2 - x) / (3 * x + 4) → y ≠ -1/3) :=
by
  intro x hx h
  rw [eq_comm] at h
  sorry

end unattainable_value_l159_159040


namespace min_value_of_quadratic_l159_159751

theorem min_value_of_quadratic :
  ∃ y : ℝ, (∀ x : ℝ, y^2 - 6 * y + 5 ≥ (x - 3)^2 - 4) ∧ (y^2 - 6 * y + 5 = -4) :=
by sorry

end min_value_of_quadratic_l159_159751


namespace periodic_function_sum_l159_159539

noncomputable def f (x : ℕ) : ℝ := Real.sin (↑x * Real.pi / 3)

theorem periodic_function_sum :
  ∑ i in Finset.range 2016, f (i + 1) = 0 :=
by sorry

end periodic_function_sum_l159_159539


namespace find_general_term_find_sum_of_b_l159_159479

variables {n : ℕ} (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Given conditions
axiom a5 : a 5 = 10
axiom S7 : S 7 = 56

-- Definition of S (Sum of first n terms of an arithmetic sequence)
def S_def (a : ℕ → ℕ) (n : ℕ) : ℕ := n * (a 1 + a n) / 2

-- Definition of the arithmetic sequence
def a_arith_seq (n : ℕ) : ℕ := 2 * n

-- Assuming the axiom for the arithmetic sequence sum
axiom S_is_arith : S 7 = S_def a 7

theorem find_general_term : a = a_arith_seq := 
by sorry

-- Sequence b
def b (n : ℕ) : ℕ := 2 + 9 ^ n

-- Sum of first n terms of sequence b
def T (n : ℕ) : ℕ := (Finset.range n).sum b

-- Prove T_n formula
theorem find_sum_of_b : ∀ n, T n = 2 * n + 9 / 8 * (9 ^ n - 1) :=
by sorry

end find_general_term_find_sum_of_b_l159_159479


namespace cube_root_neg_eight_l159_159409

/-- The cube root of -8 is -2. -/
theorem cube_root_neg_eight : real.cbrt (-8) = -2 :=
by 
  -- The proof steps would go here, but we include 'sorry' to indicate we're not proving it now.
  sorry

end cube_root_neg_eight_l159_159409


namespace time_to_pass_telegraph_post_l159_159318

-- Definitions from the problem conditions
def train_length : ℝ := 60    -- length of the train in meters
def train_speed_kmph : ℝ := 72  -- speed of the train in kmph

-- Intermediate conversions
def kmph_to_mps (speed_kmph : ℝ) := (speed_kmph * 1000) / 3600

-- The correct answer we are going to prove
def time_to_pass_post : ℝ := 3  -- time in seconds

-- The main statement to prove
theorem time_to_pass_telegraph_post :
  let speed_mps := kmph_to_mps train_speed_kmph in
  let time := train_length / speed_mps in
  time = time_to_pass_post :=
by
  sorry

end time_to_pass_telegraph_post_l159_159318


namespace expectation_inverse_quadratic_form_exists_l159_159214

open MeasureTheory ProbabilityTheory

noncomputable def gaussian_vector (d: ℕ) : Type := sorry -- Gaussian vector in ℝ^d with unit covariance matrix

axiom symmetric_positive_definite (d: ℕ) (B: Matrix (Fin d) (Fin d) ℝ) : Prop := 
B.isSymmetric ∧ ∀ z : Fin d → ℝ, z ≠ 0 → (z ⬝ᵥ (B.mul_vec z)) > 0

-- Main statement
theorem expectation_inverse_quadratic_form_exists (d: ℕ) (ξ : gaussian_vector d) (B: Matrix (Fin d) (Fin d) ℝ)
  (hB : symmetric_positive_definite d B) :
  ∃ (E : ℝ), E = 𝔼[(ξᵀ ⬝ᵥ (B.mul_vec ξ))⁻¹] ↔ d > 2 :=
sorry

end expectation_inverse_quadratic_form_exists_l159_159214


namespace range_of_f_l159_159031

noncomputable def f (x : ℝ) : ℝ := (3 * x + 8) / (x - 4)

theorem range_of_f : set.range f = set.univ \ {3} :=
by
  sorry

end range_of_f_l159_159031


namespace sequence_formula_l159_159192

theorem sequence_formula (a : ℕ → ℤ) (h1 : a 1 = 1)
  (h2 : ∀ n: ℕ, a (n + 1) = 2 * a n + n * (1 + 2^n)) :
  ∀ n : ℕ, a n = 2^(n - 2) * (n^2 - n + 6) - n - 1 :=
by intro n; sorry

end sequence_formula_l159_159192


namespace vovochkas_no_carry_pairs_eq_correct_pairs_vovochkas_smallest_difference_l159_159140

theorem vovochkas_no_carry_pairs_eq_correct_pairs :
  let valid_digit_pairs := Nat.range 10.map (λ n, 10 - n)
  let sum_valid_digit_pairs := valid_digit_pairs.sum
  sum_valid_digit_pairs = 55 →
  let no_carry_combinations := 81 * 55 * 55
  no_carry_combinations = 244620 :=
by
  let valid_digit_pairs := List.map (λ n, 10 - n) (List.range 10)
  have h_valid_digit_sum : valid_digit_pairs.sum = 55 := by sorry
  let no_carry_combinations := 81 * 55 * 55
  have h_no_carry : no_carry_combinations = 244620 := by sorry
  exact h_no_carry

theorem vovochkas_smallest_difference :
  let incorrect_cases := [1800]
  incorrect_cases.minimum = 1800 :=
by
  let differences := [900, 90, 990]
  have h_min_diff : List.minimum differences = some 90 := by sorry
  let incorrect_cases := List.map (λ diff, 20 * diff) differences
  have h_min_incorrect : incorrect_cases.minimum = some 1800 := by sorry
  exact h_min_incorrect

end vovochkas_no_carry_pairs_eq_correct_pairs_vovochkas_smallest_difference_l159_159140


namespace sequence_formula_l159_159384

noncomputable def a : ℕ → ℕ
| 0       => 2
| (n + 1) => a n ^ 2 - n * a n + 1

theorem sequence_formula (n : ℕ) : a n = n + 2 :=
by
  induction n with
  | zero => sorry
  | succ n ih => sorry

end sequence_formula_l159_159384


namespace max_intersection_l159_159748

open Real

variable (p q : Polynomial ℝ)

def third_degree_with_leading_one (f : Polynomial ℝ) : Prop :=
  degree f = 3 ∧ (leadingCoeff f) = 1

theorem max_intersection (hp : third_degree_with_leading_one p) (hq : third_degree_with_leading_one q) :
    ∃ k : ℕ, k = 2 ∧
             ∀ x : ℝ,  (p - q).eval x = 0 → count_roots (p - q) k := 
sorry

end max_intersection_l159_159748


namespace different_gcd_values_count_l159_159768

theorem different_gcd_values_count :
  let gcd_lcm_eq_prod (a b : ℕ) := Nat.gcd a b * Nat.lcm a b = a * b
  let prime_factors_360 := (2 ^ 3 * 3 ^ 2 * 5 ^ 1 : ℕ)
  (∃ a b : ℕ, gcd_lcm_eq_prod a b ∧ a * b = 360) →
  (∃ gcd_vals : Finset ℕ, gcd_vals = {1, 2, 3, 4, 6, 8, 12, 24} ∧ gcd_vals.card = 8) :=
begin
  sorry
end

end different_gcd_values_count_l159_159768


namespace triangle_area_of_tangent_line_l159_159432

-- Define the curve and the conditions
def curve (x : ℝ) : ℝ := Real.exp x
def point_of_tangency : ℝ × ℝ := (0, 1)

-- The proof problem to be stated.
theorem triangle_area_of_tangent_line (x y : ℝ) 
  (h_curve : curve x = y)
  (h_tangent : point_of_tangency = (0, 1)) : 
  (1 / 2) = 
    let slope := (Real.deriv curve) 0 in 
    let y_intercept := 1 in 
    let x_intercept := -1 in 
    1 / 2 * (y_intercept * x_intercept).abs := 
sorry

end triangle_area_of_tangent_line_l159_159432


namespace number_of_partitions_with_integer_roots_l159_159100

def sum_elements_set (s : Set ℕ) : ℕ :=
  s.to_list.sum

theorem number_of_partitions_with_integer_roots :
  let original_set := {n | ∃ k ≤ 2005, n = 2^k}
  ∃ (A B : Set ℕ), A ∪ B = original_set ∧ A ∩ B = ∅ ∧
  (∃ x₁ x₂ : ℤ, (x₁ * x₂ = sum_elements_set B) ∧ (x₁ + x₂ = sum_elements_set A) ∧
    (x₁^2 - sum_elements_set A * x₁ + sum_elements_set B = 0)) ↔
  1003 := 
sorry

end number_of_partitions_with_integer_roots_l159_159100


namespace hash_of_hash_l159_159455

def hash_right (x : ℝ) : ℝ := x + 5
def hash_left (x : ℝ) : ℝ := x - 5

theorem hash_of_hash : hash_left (hash_right 18) = 18 :=
by
  simp [hash_right, hash_left]
  sorry

end hash_of_hash_l159_159455


namespace average_marks_math_chem_l159_159696

-- Definitions to capture the conditions
variables (M P C : ℕ)
variable (cond1 : M + P = 32)
variable (cond2 : C = P + 20)

-- The theorem to prove
theorem average_marks_math_chem (M P C : ℕ) 
  (cond1 : M + P = 32) 
  (cond2 : C = P + 20) : 
  (M + C) / 2 = 26 := 
sorry

end average_marks_math_chem_l159_159696


namespace range_of_a_l159_159463

theorem range_of_a (a : ℝ) (x : ℝ) : 
  (a - 4 < x ∧ x < a + 4) → (1 < x ∧ x < 3) ↔ (a ∈ set.Icc (-1) 5) :=
begin
  sorry -- Proof omitted
end

end range_of_a_l159_159463


namespace C_class_has_45_students_l159_159286

open Nat

-- Let A, B, and C be the number of students in classes A, B, and C respectively
def A_class_students : Nat := 44
def B_class_students (A_class_students : Nat) : Nat := A_class_students + 2
def C_class_students (B_class_students : Nat) : Nat := B_class_students - 1

theorem C_class_has_45_students (A_class_students : Nat) (B_class_students : Nat) (C_class_students : Nat) :
  A_class_students = 44 → 
  B_class_students = A_class_students + 2 → 
  C_class_students = B_class_students - 1 → 
  C_class_students = 45 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry  -- Proof skipped using sorry

end C_class_has_45_students_l159_159286


namespace differentiable_function_product_one_l159_159631

theorem differentiable_function_product_one (f : ℝ → ℝ) (h_cont : ∀ x ∈ set.Icc (0:ℝ) 1, ∃ l, tendsto f (𝓝[set.Icc 0 1] x) (𝓝 l))
  (h_diff : ∀ x ∈ set.Ioo (0:ℝ) 1, differentiable_at ℝ f x)
  (h_f0 : f 0 = 0) (h_f1 : f 1 = 1) :
  ∃ a b ∈ set.Ioo (0:ℝ) 1, a ≠ b ∧ (fderiv ℝ f a).to_fun 1 * (fderiv ℝ f b).to_fun 1 = 1 := 
sorry

end differentiable_function_product_one_l159_159631


namespace sum_odd_numbers_l159_159902

theorem sum_odd_numbers (n : ℕ) : (finset.range n.succ).sum (λ k, 2 * k + 1) = 2 ^ n := by
  sorry

end sum_odd_numbers_l159_159902


namespace find_pq_of_orthogonal_and_equal_magnitudes_l159_159614

noncomputable def vec_a (p : ℝ) : ℝ × ℝ × ℝ := (4, p, -2)
noncomputable def vec_b (q : ℝ) : ℝ × ℝ × ℝ := (3, 2, q)

theorem find_pq_of_orthogonal_and_equal_magnitudes
    (p q : ℝ)
    (h1 : 4 * 3 + p * 2 + (-2) * q = 0)
    (h2 : 4^2 + p^2 + (-2)^2 = 3^2 + 2^2 + q^2) :
    (p, q) = (-29/12, 43/12) :=
by {
  sorry
}

end find_pq_of_orthogonal_and_equal_magnitudes_l159_159614


namespace polynomial_solution_l159_159012

noncomputable def polynomial_equality_problem (P Q : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, P(x + Q(y)) = Q(x + P(y))

theorem polynomial_solution (P Q : ℝ → ℝ) (h : polynomial_equality_problem P Q) :
  (P = Q) ∨ 
  (∃ u v : ℝ, u ≠ v ∧ P = λ x, x + u ∧ Q = λ x, x + v) :=
sorry

end polynomial_solution_l159_159012


namespace seq_geometric_sum_of_seq_l159_159595

noncomputable def a : ℕ+ → ℤ
| ⟨1, _⟩ := 2
| ⟨n + 1, _⟩ := 4 * a ⟨n, by simp⟩ - 3 * n + 1

theorem seq_geometric (n : ℕ+) : ∃ q : ℤ, a n - n = q^(n - 1) :=
sorry

theorem sum_of_seq (n : ℕ) : S n = (n * (n + 1)) / 2 + (4^n - 1) / 3 :=
sorry

end seq_geometric_sum_of_seq_l159_159595


namespace polynomial_coefficient_sum_l159_159883

theorem polynomial_coefficient_sum :
  let f := (x^2 - x + 1)^6 in
  (f.coeff 12 + f.coeff 10 + f.coeff 8 + f.coeff 6 + f.coeff 4 + f.coeff 2 + f.coeff 0 = 365) :=
sorry

end polynomial_coefficient_sum_l159_159883


namespace find_number_of_clerks_l159_159321

-- Define the conditions 
def avg_salary_per_head_staff : ℝ := 90
def avg_salary_officers : ℝ := 600
def avg_salary_clerks : ℝ := 84
def number_of_officers : ℕ := 2

-- Define the variable C (number of clerks)
def number_of_clerks : ℕ := sorry   -- We will prove that this is 170

-- Define the total salary equations based on the conditions
def total_salary_officers := number_of_officers * avg_salary_officers
def total_salary_clerks := number_of_clerks * avg_salary_clerks
def total_number_of_staff := number_of_officers + number_of_clerks
def total_salary := total_salary_officers + total_salary_clerks

-- Define the average salary per head equation 
def avg_salary_eq : Prop := avg_salary_per_head_staff = total_salary / total_number_of_staff

theorem find_number_of_clerks (h : avg_salary_eq) : number_of_clerks = 170 :=
sorry

end find_number_of_clerks_l159_159321


namespace valid_k_sum_correct_l159_159923

def sum_of_valid_k : ℤ :=
  (List.range 17).sum * 1734 + (List.range 17).sum * 3332

theorem valid_k_sum_correct : sum_of_valid_k = 5066 := by
  sorry

end valid_k_sum_correct_l159_159923


namespace reporters_local_politics_percentage_l159_159362

theorem reporters_local_politics_percentage
  (T : ℕ) -- Total number of reporters
  (P : ℝ) -- Percentage of reporters covering politics
  (h1 : 30 / 100 * (P / 100) * T = (P / 100 - 0.7 * (P / 100)) * T)
  (h2 : 92.85714285714286 / 100 * T = (1 - P / 100) * T):
  (0.7 * (P / 100) * T) / T = 5 / 100 :=
by
  sorry

end reporters_local_politics_percentage_l159_159362


namespace number_of_sets_A_l159_159093

theorem number_of_sets_A (U : Set ℕ) (B : Set ℕ) (hU : U = {1, 2, 3}) (hB : B = {1, 2}) :
  {A : Set ℕ | A ∩ B = {1}}.size = 2 :=
sorry

end number_of_sets_A_l159_159093


namespace sequence_sum_eq_5_l159_159091

noncomputable def a_n (n : ℕ) : ℤ := (-1 : ℤ)^n * (n + 1)

theorem sequence_sum_eq_5 : (∑ i in Finset.range 10, a_n (i + 1)) = 5 := by
  sorry

end sequence_sum_eq_5_l159_159091


namespace max_food_per_guest_l159_159272

theorem max_food_per_guest (total_food : ℕ) (min_guests : ℕ)
    (H1 : total_food = 406) (H2 : min_guests = 163) :
    2 ≤ total_food / min_guests ∧ total_food / min_guests < 3 := by
  sorry

end max_food_per_guest_l159_159272


namespace potato_gun_distance_l159_159848

noncomputable def length_of_football_field_in_yards : ℕ := 200
noncomputable def conversion_factor_yards_to_feet : ℕ := 3
noncomputable def length_of_football_field_in_feet : ℕ := length_of_football_field_in_yards * conversion_factor_yards_to_feet

noncomputable def dog_running_speed : ℕ := 400
noncomputable def time_for_dog_to_fetch_potato : ℕ := 9
noncomputable def total_distance_dog_runs : ℕ := dog_running_speed * time_for_dog_to_fetch_potato

noncomputable def actual_distance_to_potato : ℕ := total_distance_dog_runs / 2

noncomputable def distance_in_football_fields : ℕ := actual_distance_to_potato / length_of_football_field_in_feet

theorem potato_gun_distance :
  distance_in_football_fields = 3 :=
by
  sorry

end potato_gun_distance_l159_159848


namespace angle_in_third_quadrant_l159_159537

theorem angle_in_third_quadrant (α : ℝ) 
  (h1 : sin α * tan α < 0)
  (h2 : cos α / tan α < 0) : 
  (π < α) ∧ (α < 3 * π / 2) := 
sorry

end angle_in_third_quadrant_l159_159537


namespace boys_without_calculators_proof_l159_159567

-- Defining the conditions
def total_boys : ℕ := 20
def total_students_with_calculators : ℕ := 25
def girls_with_calculators : ℕ := 15

-- Define the question as a proposition to prove
def boys_without_calculators : Prop :=
  let boys_with_calculators := total_students_with_calculators - girls_with_calculators in
  let boys_without_calculators := total_boys - boys_with_calculators in
  boys_without_calculators = 10

-- Statement without proof
theorem boys_without_calculators_proof : boys_without_calculators :=
by
  sorry

end boys_without_calculators_proof_l159_159567


namespace sin_frac_pi_four_add_alpha_cos_frac_pi_six_sub_two_alpha_l159_159468

theorem sin_frac_pi_four_add_alpha
  (α : ℝ)
  (h1 : α ∈ set.Ioo (π / 2) π)
  (h2 : Real.sin α = 3 / 5) :
  Real.sin (π / 4 + α) = - (Real.sqrt 2) / 10 :=
by
  sorry

theorem cos_frac_pi_six_sub_two_alpha
  (α : ℝ)
  (h1 : α ∈ set.Ioo (π / 2) π)
  (h2 : Real.sin α = 3 / 5) :
  Real.cos (π / 6 - 2 * α) = (7 * Real.sqrt 3 - 24) / 50 :=
by
  sorry

end sin_frac_pi_four_add_alpha_cos_frac_pi_six_sub_two_alpha_l159_159468


namespace range_of_a_l159_159117

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x + 2| - |x - 1| ≥ a^3 - 4a^2 - 3) ↔ a ≤ 4 :=
by
  sorry

end range_of_a_l159_159117


namespace triangle_AFG_is_isosceles_l159_159213

variables {A B C D E F G : Point} -- Define the points
variables {ω : Circle} -- Define the inscribed circle

-- Define the isosceles trapezoid and relevant parallelism
axiom isosceles_trapezoid (ABCD : Trapezoid A B C D) : is_isosceles_trapezoid ABCD ∧ parallel A B C D

-- Define the incircle of triangle BCD and its tangency point E
axiom incircle_BCD (BCD : Triangle B C D) (ω : incircle B C D) (E : touches ω C D) : on_circle E ω

-- Define the point F on the angle bisector of ∠DAC such that EF ⊥ CD
axiom point_F (DAC : Angle A D C) (bisector_DAC : bisects F (angle_bisector DAC)) : perpendicular F E C D

-- Define the circumscribed circle of triangle ACF and intersection points with CD
axiom circumscribed_circle_ACF (ACF : Triangle A C F) (G : intersection (circle_of_triangle A C F) C D) : 
  on_circle G (circle_of_triangle A C F)

-- Statement to prove: Triangle AFG is isosceles (AF = FG)
theorem triangle_AFG_is_isosceles (ABCD : Trapezoid A B C D) (ω : Circle)
  (incircle_BCD : incircle B C D) (E : touches ω C D) (F : Point) (G : Point)
  (isosceles_trapezoid ABCD) (on_circle E ω) (bisector_DAC : bisects F (angle_bisector (Angle A D C)))
  (perpendicular F E C D) (on_circle G (circle_of_triangle A C F)) :
  distance A F = distance F G :=
sorry

end triangle_AFG_is_isosceles_l159_159213


namespace max_C_is_3_l159_159481

theorem max_C_is_3 {n : ℕ} (h_n : n ≥ 2) (p : Fin n → ℝ) (x : Fin n → ℝ)
  (h_sum_p : ∑ i, p i = 1)
  (h_pos_p : ∀ i, p i > 0)
  (h_pos_x : ∀ i, x i > 0)
  (h_diff_x : ∃ i j, i ≠ j ∧ x i ≠ x j) :
  (∀ k, k = 2 ∨ k = 3 →
    let d_k := ∑ i, p i * (x i) ^ k - (∑ i, p i * x i) ^ k
    in ∀ C, (d_3 / d_2) ≥ C * Finset.min' Finset.univ (λ i, x i) → C ≤ 3) :=
sorry

end max_C_is_3_l159_159481


namespace tan_undefined_at_270_l159_159434

theorem tan_undefined_at_270
  (n : ℤ)
  (h : -180 < n ∧ n < 180)
  (h₀ : ∀ k : ℤ, ¬ deriving ⟨θ = 90 + 180 * k⟩)
  (h₁ : tan 270 = tan (90 + 180))
  : n = 90 ∨ n = -90 :=
sorry

end tan_undefined_at_270_l159_159434


namespace find_A_and_B_l159_159429

theorem find_A_and_B (A B : ℝ) :
  (∀ x : ℝ, (x ≠ 12 ∧ x ≠ -3) → 
    (6 * x + 3) / (x^2 - 9 * x - 36) = A / (x - 12) + B / (x + 3)) →
  (A, B) = (5, 1) :=
by {
  intros h,
  -- Rest of the proof would normally go here
  sorry
}

end find_A_and_B_l159_159429


namespace find_pq_of_orthogonal_and_equal_magnitudes_l159_159613

noncomputable def vec_a (p : ℝ) : ℝ × ℝ × ℝ := (4, p, -2)
noncomputable def vec_b (q : ℝ) : ℝ × ℝ × ℝ := (3, 2, q)

theorem find_pq_of_orthogonal_and_equal_magnitudes
    (p q : ℝ)
    (h1 : 4 * 3 + p * 2 + (-2) * q = 0)
    (h2 : 4^2 + p^2 + (-2)^2 = 3^2 + 2^2 + q^2) :
    (p, q) = (-29/12, 43/12) :=
by {
  sorry
}

end find_pq_of_orthogonal_and_equal_magnitudes_l159_159613


namespace prove_y_value_l159_159547

theorem prove_y_value (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 :=
by
  sorry

end prove_y_value_l159_159547


namespace determine_signs_l159_159227

theorem determine_signs (a b c : ℝ) (h1 : a != 0 ∧ b != 0 ∧ c == 0)
  (h2 : a > 0 ∨ (b + c) > 0) : a > 0 ∧ b < 0 ∧ c = 0 :=
by
  sorry

end determine_signs_l159_159227


namespace exponent_sum_l159_159295

theorem exponent_sum : (7^-2)^0 + (7^0)^-3 = 2 :=
by
  sorry

end exponent_sum_l159_159295


namespace six_digit_permutation_number_l159_159897

-- Define that a number is a permutation of another
def is_permutation_of (a b : ℕ) : Prop :=
  let a_digits := a.digits 10;
  let b_digits := b.digits 10;
  multiset.of_list a_digits = multiset.of_list b_digits

theorem six_digit_permutation_number :
  ∃ x : ℕ, 100000 ≤ x ∧ x ≤ 999999 ∧ 
           is_permutation_of (2 * x) x ∧
           is_permutation_of (3 * x) x ∧
           is_permutation_of (4 * x) x ∧
           is_permutation_of (5 * x) x ∧
           is_permutation_of (6 * x) x ∧
           x = 142857 :=
by
  sorry

end six_digit_permutation_number_l159_159897


namespace min_lines_to_separate_points_l159_159212

theorem min_lines_to_separate_points (n : ℕ) (hn : n ≥ 2) (points : Fin n → ℝ × ℝ)
  (h_collinear : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → ¬ collinear (points i) (points j) (points k)) :
  ∃ m, (∀ (i j : Fin n), i ≠ j → ∃ (l : ℝ × ℝ → Prop), (l (points i) ∧ ¬ l (points j)) ∨ (¬ l (points i) ∧ l (points j))) ∧ m = ⌈ (n : ℝ) / 2 ⌉ :=
sorry

-- Auxiliary definitions

def collinear (p q r : ℝ × ℝ) : Prop :=
∃ a b c : ℝ, a * p.1 + b * p.2 + c = 0 ∧ a * q.1 + b * q.2 + c = 0 ∧ a * r.1 + b * r.2 + c = 0

noncomputable def ceiling (x : ℝ) : ℕ := (Real.ceil x).to_nat

end min_lines_to_separate_points_l159_159212


namespace fraction_filled_l159_159787

variables (E P p : ℝ)

-- Condition 1: The empty vessel weighs 12% of its total weight when filled.
axiom cond1 : E = 0.12 * (E + P)

-- Condition 2: The weight of the partially filled vessel is one half that of a completely filled vessel.
axiom cond2 : E + p = 1 / 2 * (E + P)

theorem fraction_filled : p / P = 19 / 44 :=
by
  sorry

end fraction_filled_l159_159787


namespace sum_of_cubes_eq_neg_27_l159_159207

variable {a b c : ℝ}

-- Define the condition that k is the same for a, b, and c
def same_k (a b c k : ℝ) : Prop :=
  k = (a^3 + 9) / a ∧ k = (b^3 + 9) / b ∧ k = (c^3 + 9) / c

-- Theorem: Given the conditions, a^3 + b^3 + c^3 = -27
theorem sum_of_cubes_eq_neg_27 (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_same_k : ∃ k, same_k a b c k) :
  a^3 + b^3 + c^3 = -27 :=
sorry

end sum_of_cubes_eq_neg_27_l159_159207


namespace magnitude_of_sum_l159_159054

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x, 1)

def is_parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (u.1, u.2) = (k * v.1, k * v.2)

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem magnitude_of_sum (x : ℝ) (h : is_parallel vector_a (vector_a.1 - vector_b x.1, vector_a.2 - vector_b x.2)) :
  magnitude (vector_a.1 + vector_b x.1, vector_a.2 + vector_b x.2) = 3 * real.sqrt 5 / 2 :=
sorry

end magnitude_of_sum_l159_159054


namespace tank_full_capacity_is_72_l159_159755

theorem tank_full_capacity_is_72 (x : ℝ) 
  (h1 : 0.9 * x - 0.4 * x = 36) : 
  x = 72 := 
sorry

end tank_full_capacity_is_72_l159_159755


namespace nested_H_value_l159_159867

def H (x : ℝ) : ℝ := (x - 3)^2 / 2 - 2

variable (hH2 : H 2 = 1 / 2)
variable (hHhalf : H (1 / 2) = 5)
variable (hH5 : H 5 = 1 / 2)

theorem nested_H_value : H (H (H (H (H 2)))) = 1 / 2 := by
  rw [hH2, hHhalf, hH5, hHhalf, hH5]
  exact sorry

end nested_H_value_l159_159867


namespace average_of_B_and_C_l159_159167

theorem average_of_B_and_C (x : ℚ) (A B C : ℚ)
  (h1 : A = 4 * x) (h2 : B = 6 * x) (h3 : C = 9 * x) (h4 : A = 50) :
  (B + C) / 2 = 93.75 := 
sorry

end average_of_B_and_C_l159_159167


namespace count_valid_sets_l159_159963

open Set

-- Define sets A, B and the union
def A : Set ℕ := { n | n < 12 } -- This is just a placeholder for the actual set
def B : Set ℕ := { n | n < 13 } -- This is just a placeholder for the actual set

-- Conditions given in the problem
axiom h1 : A.card = 12
axiom h2 : B.card = 13
axiom h3 : (A ∪ B).card = 15

-- Define set C and conditions involving it
noncomputable def valid_set_count : ℕ :=
  {C : Set ℕ | C ⊆ A ∪ B ∧ (C ∩ A ∩ B).card = 1 ∧ ¬C ⊆ B}.card

-- The proof statement
theorem count_valid_sets : valid_set_count = 240 := 
by sorry

end count_valid_sets_l159_159963


namespace phone_repair_cost_l159_159413

theorem phone_repair_cost (P : ℝ) : 
  let phone_repairs_cost := 5 * P,
      laptop_repairs_cost := 2 * 15,
      computer_repairs_cost := 2 * 18,
      total_weekly_earnings := 121
  in phone_repairs_cost + laptop_repairs_cost + computer_repairs_cost = total_weekly_earnings → P = 11 :=
by
  intros h,
  let h_eq := calc
    5 * P + 2 * 15 + 2 * 18 = 121 : h
    5 * P + 30 + 36 = 121       : rfl
    5 * P + 66 = 121            : rfl
    5 * P = 121 - 66            : by ring
    5 * P = 55                  : by ring,
  exact eq_of_mul_eq_mul_of_ne_zero five_ne_zero h_eq

end phone_repair_cost_l159_159413


namespace domino_path_count_l159_159224

-- Definitions of coordinates for A and B
def A := (0, 4)
def B := (6, 0)

-- Conditions of the movements from A to B
def movements (right down : ℕ) : List (ℕ × ℕ) :=
  List.replicate right (1, 0) ++ List.replicate down (0, 1)

-- Function to count combinations
noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.descFactorial n k / (Nat.factorial k)

-- Main theorem statement
theorem domino_path_count : 
  ∃ (n m : ℕ), n = 6 ∧ m = 4 ∧ 
  (∀ (A B : ℕ × ℕ), A = (0, 4) ∧ B = (6, 0) → 
  binomial (n + m) m = 210) :=
by
  -- Introduce integers n=6, m=4 that represent grid dimensions.
  let n := 6
  let m := 4

  -- Sessions to start the path from A to B
  intros A B hA hB

  -- Calculate the binomial coefficient for the path count
  have : binomial (n + m) m = 210, from sorry

  -- Return the result ensuring conditions are met
  existsi n, existsi m
  split, refl, split, refl, assumption

end domino_path_count_l159_159224


namespace complement_of_A_in_U_l159_159965

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the set A
def A : Set ℕ := {3, 4, 5}

-- Statement to prove the complement of A with respect to U
theorem complement_of_A_in_U : U \ A = {1, 2, 6} := 
  by sorry

end complement_of_A_in_U_l159_159965


namespace sum_of_solutions_eq_zero_l159_159445

theorem sum_of_solutions_eq_zero : 
  let eqn := λ x : ℤ, x^4 - 13 * x^2 + 36 in
  ∑ x in { x : ℤ | eqn x = 0 }.toFinset = 0 :=
by sorry

end sum_of_solutions_eq_zero_l159_159445


namespace num_ways_choose_cards_card_selection_proof_l159_159998

theorem num_ways_choose_cards (p k n : ℕ) : ℕ :=
  let max_0_neg_n := max 0 (-n)
  let min_k_p_n := min k (p - n)
  min_k_p_n - max_0_neg_n + 1

-- Note: Replace with a theorem statement for proof verification
theorem card_selection_proof (p k n : ℤ) : 
  ∀ (r b : ℕ), 
    (0 ≤ r ∧ r ≤ p) ∧ (0 ≤ b ∧ b ≤ k) ∧ (r = b + n) → 
    (num_ways_choose_cards p k n) = (min (k:ℤ) (p:ℤ - n:ℤ) - max 0 (- n:ℤ) + 1 : ℕ) := 
by 
  sorry

end num_ways_choose_cards_card_selection_proof_l159_159998


namespace max_product_l159_159606

-- Given an integer n and a table of dimension n x n, the maximal value of 
-- the product of n numbers chosen such that no two numbers are from the same row or column 
-- is (n-1)^n * (n+1)!
theorem max_product (n : ℕ) (h : 0 < n) :
  ∃ P : Fin n → ℕ, (∀ i j : Fin n, i ≠ j → P i ≠ P j ∧ P i = n * i.val + (n - i.val - 1)) ∧
    (Finset.univ.prod (λ i : Fin n, P i) = (n - 1)^n * (n + 1)!) := 
sorry

end max_product_l159_159606


namespace math_problem_l159_159412

theorem math_problem :
  ((-1/3)^(-2) + abs (real.sqrt 3 - 1) - (real.pi - real.tan (real.pi / 3))^0 - 2 * real.cos (real.pi / 6)) = 7 :=
by
  -- We skip the proof as requested
  sorry

end math_problem_l159_159412


namespace shaded_region_area_l159_159404

noncomputable def line1 (x : ℝ) : ℝ := -(3 / 10) * x + 5
noncomputable def line2 (x : ℝ) : ℝ := -(5 / 7) * x + 47 / 7

noncomputable def intersection_x : ℝ := 17 / 5

noncomputable def area_under_curve (f g : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, (g x - f x)

theorem shaded_region_area : 
  area_under_curve line1 line2 0 intersection_x = 1.91 :=
sorry

end shaded_region_area_l159_159404


namespace marble_probability_correct_l159_159347

noncomputable def marble_probability : ℚ :=
  let total_ways := (Nat.choose 20 4 : ℚ)
  let ways_two_red := (Nat.choose 12 2 : ℚ)
  let ways_two_blue := (Nat.choose 8 2 : ℚ)
  (ways_two_red * ways_two_blue) / total_ways

theorem marble_probability_correct : marble_probability = 56 / 147 :=
by
  -- Note: the proof is omitted as per instructions
  sorry

end marble_probability_correct_l159_159347


namespace least_semiprime_trio_l159_159892

def is_semiprime (n : ℕ) : Prop :=
  (∃ p q : ℕ, prime p ∧ prime q ∧ n = p * q)

def condition (n : ℕ) : Prop :=
  is_semiprime n ∧ is_semiprime (n + 1) ∧ is_semiprime (n + 2)

theorem least_semiprime_trio : ∃ n : ℕ, condition n ∧ ∀ m : ℕ, m < n → ¬ condition m :=
begin
  use 33,
  split,
  {
    -- Prove that 33, 34, and 35 are semiprimes
    sorry
  },
  {
    -- Prove that there is no smaller number satisfying the condition
    sorry
  }
end

end least_semiprime_trio_l159_159892


namespace base12_divisible_by_11_l159_159361

theorem base12_divisible_by_11 (n : ℕ) (digits : ℕ → ℕ) (h_digits : ∀ i, digits i < 12) (H1 : n = ∑ i in finset.range k, (digits i) * 12^i) :
  (∑ i in finset.range k, digits i) % 11 = 0 → n % 11 = 0 :=
by
  sorry

end base12_divisible_by_11_l159_159361


namespace vasya_birthday_day_l159_159737

/-- Define the days of the week as an inductive type --/
inductive Day
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday
deriving DecidableEq

open Day

/-- Function to get the day after a given day --/
def next_day : Day → Day
| Monday    => Tuesday
| Tuesday   => Wednesday
| Wednesday => Thursday
| Thursday  => Friday
| Friday    => Saturday
| Saturday  => Sunday
| Sunday    => Monday

/-- Function to get the day two days after a given day --/
def day_after_tomorrow (d : Day) : Day :=
  next_day (next_day d)

/-- Lean statement for the problem --/
theorem vasya_birthday_day (today : Day) (H1 : day_after_tomorrow today = Sunday) (H2 : next_day vasya_birthday = today) : vasya_birthday = Thursday := 
sorry

end vasya_birthday_day_l159_159737


namespace race_distance_p_l159_159319

noncomputable def distance_run_p (p q D : ℝ) : ℝ :=
  if (p = 1.2 * q ∧ D + 300 / p = D / q) then D + 300 else 0

theorem race_distance_p
  (p q : ℝ)
  (h1 : p = 1.2 * q)
  (h2 : ∃ D, D / q = (D + 300) / p) :
  distance_run_p p q (some h2) = 1800 :=
sorry

end race_distance_p_l159_159319


namespace geo_sequence_proof_l159_159473

noncomputable def geometric_sequence_terms {a_1 q : ℝ}
    (h1 : a_1 + a_1 * q^2 = 10)
    (h2 : a_1 * q^3 + a_1 * q^5 = 5 / 4) 
    : ℝ × ℝ :=
  let a_4 := a_1 * q^3 in
  let s_5 := a_1 * (1 - q^5) / (1 - q) in
  (a_4, s_5)

theorem geo_sequence_proof : 
    ∃ (a_1 q : ℝ), 
      (a_1 + a_1 * q^2 = 10) ∧ 
      (a_1 * q^3 + a_1 * q^5 = 5 / 4) ∧ 
      geometric_sequence_terms h1 h2 = (1, 31 / 2) :=
by
  sorry

end geo_sequence_proof_l159_159473


namespace unicorn_tethered_l159_159387

def unicorn_problem :=
  ∃ (p q r : ℕ), (r.prime) ∧
    (25^2 = 10^2 + (5 * real.sqrt 6)^2) ∧
    (10 * real.sqrt 7)^2 = ((10 * real.sqrt 7 - 5 * real.sqrt 6) * (3 * x))^2 ∧ 
    x = (75 - real.sqrt 1050) / 3 ∧ 
    p = 75 ∧ q = 1050 ∧ r = 3 ∧ 
    p + q + r = 1128

theorem unicorn_tethered : unicorn_problem :=
by 
  sorry

end unicorn_tethered_l159_159387


namespace pencils_total_l159_159600

-- Defining the conditions
def packs_to_pencils (packs : ℕ) : ℕ := packs * 12

def jimin_packs : ℕ := 2
def jimin_individual_pencils : ℕ := 7

def yuna_packs : ℕ := 1
def yuna_individual_pencils : ℕ := 9

-- Translating to Lean 4 statement
theorem pencils_total : 
  packs_to_pencils jimin_packs + jimin_individual_pencils + packs_to_pencils yuna_packs + yuna_individual_pencils = 52 := 
by
  sorry

end pencils_total_l159_159600


namespace prob_snow_both_days_l159_159277

-- Definitions for the conditions
def prob_snow_monday : ℚ := 40 / 100
def prob_snow_tuesday : ℚ := 30 / 100

def independent_events (A B : Prop) : Prop := true  -- A placeholder definition of independence

-- The proof problem: 
theorem prob_snow_both_days : 
  independent_events (prob_snow_monday = 0.40) (prob_snow_tuesday = 0.30) →
  prob_snow_monday * prob_snow_tuesday = 0.12 := 
by 
  sorry

end prob_snow_both_days_l159_159277


namespace diplomats_spoke_latin_l159_159644

/-- Of the diplomats who attended a summit conference, given that:
 - There are 120 diplomats in total
 - 32 diplomats did not speak Russian
 - 20% of the diplomats spoke neither Latin nor Russian
 - 10% of the diplomats spoke both Latin and Russian
then the number of diplomats who spoke Latin is 20. --/
theorem diplomats_spoke_latin 
  (total_diplomats : ℕ)
  (non_speakers_Russian : ℕ)
  (percent_neither : ℕ)
  (percent_both : ℕ)
  (total_diplomats = 120)
  (non_speakers_Russian = 32)
  (percent_neither = 20)
  (percent_both = 10) : 
  ∃ (spoke_latin : ℕ), spoke_latin = 20 := 
sorry

end diplomats_spoke_latin_l159_159644


namespace sin_theta_fourth_quadrant_l159_159464

-- Given conditions
variables {θ : ℝ} (h1 : Real.cos θ = 1 / 3) (h2 : 3 * pi / 2 < θ ∧ θ < 2 * pi)

-- Proof statement
theorem sin_theta_fourth_quadrant : Real.sin θ = -2 * Real.sqrt 2 / 3 :=
sorry

end sin_theta_fourth_quadrant_l159_159464


namespace final_price_difference_l159_159858

noncomputable def OP : ℝ := 78.2 / 0.85
noncomputable def IP : ℝ := 78.2 + 0.25 * 78.2
noncomputable def DP : ℝ := 97.75 - 0.10 * 97.75
noncomputable def FP : ℝ := 87.975 + 0.0725 * 87.975

theorem final_price_difference : OP - FP = -2.3531875 := 
by sorry

end final_price_difference_l159_159858


namespace volume_divided_by_pi_l159_159814

-- Definitions and conditions from the problem
noncomputable def original_radius (R : ℝ) := 24
noncomputable def sector_angle (θ : ℝ) := 240

-- The main theorem statement
theorem volume_divided_by_pi : 
  let r : ℝ := 16
  let h : ℝ := 8 * Real.sqrt 5
  let V : ℝ := (1 / 3) * Math.PI * r^2 * h 
  V / Math.PI = 2048 * Real.sqrt 5 / 3 :=
by
  let r := 16
  let h := 8 * Real.sqrt 5
  let V := (1 / 3) * Math.PI * r^2 * h 
  have : V / Math.PI = 2048 * Real.sqrt 5 / 3 := sorry
  exact this

end volume_divided_by_pi_l159_159814


namespace find_phi_values_l159_159970

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2.1 * v.2.1 + u.2.2 * v.2.2

theorem find_phi_values (φ : ℝ) (hφ : 0 ≤ φ ∧ φ < 2 * Real.pi) :
  let a := (Real.sin (2 * φ), (1 : ℝ), Real.tan φ)
      b := (1, Real.cos φ, Real.cot φ) in
  dot_product a b = 1 ↔ φ = Real.pi / 2 ∨ φ = 3 * Real.pi / 2 ∨ φ = 7 * Real.pi / 6 ∨ φ = 11 * Real.pi / 6 :=
by
  sorry

end find_phi_values_l159_159970


namespace sum_of_cubes_eq_neg_27_l159_159205

variable {a b c : ℝ}

-- Define the condition that k is the same for a, b, and c
def same_k (a b c k : ℝ) : Prop :=
  k = (a^3 + 9) / a ∧ k = (b^3 + 9) / b ∧ k = (c^3 + 9) / c

-- Theorem: Given the conditions, a^3 + b^3 + c^3 = -27
theorem sum_of_cubes_eq_neg_27 (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_same_k : ∃ k, same_k a b c k) :
  a^3 + b^3 + c^3 = -27 :=
sorry

end sum_of_cubes_eq_neg_27_l159_159205


namespace integer_solution_sum_eq_zero_l159_159446

theorem integer_solution_sum_eq_zero : 
  (∑ x in (Finset.filter (λ x : ℤ, x^4 - 13 * x^2 + 36 = 0) (Finset.Icc -3 3)), x) = 0 := by
sorry

end integer_solution_sum_eq_zero_l159_159446


namespace angle_between_line_and_plane_l159_159945

noncomputable def vector_angle (m n : ℝ) : ℝ := 120

theorem angle_between_line_and_plane (m n : ℝ) : 
  (vector_angle m n = 120) → (90 - (vector_angle m n - 90) = 30) :=
by sorry

end angle_between_line_and_plane_l159_159945


namespace evaluate_g_at_neg_one_l159_159110

def g (x : ℝ) : ℝ := 5 * x^3 - 7 * x^2 - 3 * x + 9

theorem evaluate_g_at_neg_one : g (-1) = 7 :=
by 
  -- lean proof here
  sorry

end evaluate_g_at_neg_one_l159_159110


namespace lattice_points_on_line_segment_l159_159860

-- Define the structure of a point in 2D
structure Point2D where
  x : Int
  y : Int

-- Define the gcd function, ensuring it returns a non-negative result.
def gcd (a b : Int) : Int :=
  if b = 0 then abs a else gcd b (a % b)

-- Define the two endpoints of the line segment
def P1 : Point2D := { x := 5, y := 26 }
def P2 : Point2D := { x := 40, y := 146 }

-- Define the question as a formal theorem statement
theorem lattice_points_on_line_segment :
  let difference_x := P2.x - P1.x
  let difference_y := P2.y - P1.y
  gcd difference_x difference_y + 1 = 6 := 
by
  -- For the tuples, you can declare them as variables for use in the proof
  let difference_x := P2.x - P1.x
  let difference_y := P2.y - P1.y
  have gcd_def := gcd difference_x difference_y
  -- Skip the actual steps of the proof with the sorry keyword
  sorry

end lattice_points_on_line_segment_l159_159860


namespace number_of_ways_33520_l159_159582

theorem number_of_ways_33520 : 
  ∃ (n : ℕ), n = 48 ∧ 
  (∀ (digits : List ℕ), digits = [3, 3, 5, 2, 0] → 
  ∃ (arrangements : Finset (List ℕ)), 
  (∀ (num : List ℕ), num ∈ arrangements → num.head ≠ 0) ∧ 
  arrangements.card = n) :=
sorry

end number_of_ways_33520_l159_159582


namespace vovochkas_no_carry_pairs_eq_correct_pairs_vovochkas_smallest_difference_l159_159141

theorem vovochkas_no_carry_pairs_eq_correct_pairs :
  let valid_digit_pairs := Nat.range 10.map (λ n, 10 - n)
  let sum_valid_digit_pairs := valid_digit_pairs.sum
  sum_valid_digit_pairs = 55 →
  let no_carry_combinations := 81 * 55 * 55
  no_carry_combinations = 244620 :=
by
  let valid_digit_pairs := List.map (λ n, 10 - n) (List.range 10)
  have h_valid_digit_sum : valid_digit_pairs.sum = 55 := by sorry
  let no_carry_combinations := 81 * 55 * 55
  have h_no_carry : no_carry_combinations = 244620 := by sorry
  exact h_no_carry

theorem vovochkas_smallest_difference :
  let incorrect_cases := [1800]
  incorrect_cases.minimum = 1800 :=
by
  let differences := [900, 90, 990]
  have h_min_diff : List.minimum differences = some 90 := by sorry
  let incorrect_cases := List.map (λ diff, 20 * diff) differences
  have h_min_incorrect : incorrect_cases.minimum = some 1800 := by sorry
  exact h_min_incorrect

end vovochkas_no_carry_pairs_eq_correct_pairs_vovochkas_smallest_difference_l159_159141


namespace trig_inequality_l159_159648

theorem trig_inequality (x : ℝ) : 
  cos x ^ 4 + sin x ^ 2 - sin (2 * x) * cos x ≥ 0 := 
sorry

end trig_inequality_l159_159648


namespace Petya_can_verify_coins_l159_159229

theorem Petya_can_verify_coins :
  ∃ (c₁ c₂ c₃ c₅ : ℕ), 
  (c₁ = 1 ∧ c₂ = 2 ∧ c₃ = 3 ∧ c₅ = 5) ∧
  (∃ (w : ℕ), w = 9) ∧
  (∃ (cond : ℕ → Prop), 
    cond 1 ∧ cond 2 ∧ cond 3 ∧ cond 5) := sorry

end Petya_can_verify_coins_l159_159229


namespace KL_perp_AD_l159_159647

open EuclideanGeometry

variables {A B C D X K L : Point}
variables [non_trapezoid_quadrilateral A B C D]
variables [inside_point X A B C D]
variables [angle_condition : angle_sum_eq_180 X A D B C]
variables [K_def : angle_bisector_meets D_altitude A D X A B X K]
variables [L_def : angle_bisector_meets A_altitude A D X D C X L]
variables [perpendicular_bisector_1 : perp B K C X]
variables [perpendicular_bisector_2 : perp C L B X]
variables [circumcenter_lies_on_KL : circumcenter_ADX_on_line_KL A D X K L]

theorem KL_perp_AD : perp K L A D :=
sorry

end KL_perp_AD_l159_159647


namespace sam_seashells_l159_159246

def seashells_problem := 
  let mary_seashells := 47
  let total_seashells := 65
  (total_seashells - mary_seashells) = 18

theorem sam_seashells :
  seashells_problem :=
by
  sorry

end sam_seashells_l159_159246


namespace polynomial_degree_and_terms_l159_159593

theorem polynomial_degree_and_terms (p : Polynomial ℝ) (h : p = Polynomial.C 18 + Polynomial.X * Polynomial.C 2 + Polynomial.C 1 * Polynomial.X ^ 2) : 
  (p.degree = 2) ∧ (p.sum (λ n a, if a ≠ 0 then 1 else 0) = 3) :=
by
  sorry

end polynomial_degree_and_terms_l159_159593


namespace zero_distance_eq_three_l159_159920

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then log a (x + 1) + x - 2 else x + 4 - (1 / a)^(x + 1)

theorem zero_distance_eq_three (a : ℝ) (x₁ x₂ : ℝ) (ha : a > 2) (h₁ : f a x₁ = 0) (h₂ : f a x₂ = 0) :
  |x₁ - x₂| = 3 :=
sorry

end zero_distance_eq_three_l159_159920


namespace gcd_lcm_product_l159_159765

theorem gcd_lcm_product (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 360) :
  ∃ (s : Finset ℕ), s.card = 12 ∧ ∀ d ∈ s, d = Nat.gcd a b :=
sorry

end gcd_lcm_product_l159_159765


namespace prove_y_value_l159_159546

theorem prove_y_value (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 :=
by
  sorry

end prove_y_value_l159_159546


namespace sequence_nth_term_l159_159280

theorem sequence_nth_term (n : ℕ) : (∃ s : ℕ → ℕ, 
  (s 1 = 10^0) ∧ (s 2 = 10^1) ∧ (s 3 = 10^2) ∧ (s 4 = 10^3) ∧ ∀ k, s (k + 1) = 10^k ∧ s n = 10^(n-1)) :=
begin
  sorry
end

end sequence_nth_term_l159_159280


namespace friends_with_Ron_l159_159241

-- Ron is eating pizza with his friends 
def total_slices : Nat := 12
def slices_per_person : Nat := 4
def total_people := total_slices / slices_per_person
def ron_included := 1

theorem friends_with_Ron : total_people - ron_included = 2 := by
  sorry

end friends_with_Ron_l159_159241


namespace sum_of_integer_solutions_l159_159451

theorem sum_of_integer_solutions : 
  let f (x : ℤ) := x^4 - 13 * x^2 + 36 
  in (finset.sum (finset.filter (λ x, f x = 0) (finset.Icc (-3) 3)) id) = 0 :=
begin
  sorry
end

end sum_of_integer_solutions_l159_159451


namespace parity_implies_even_sum_l159_159554

theorem parity_implies_even_sum (n m : ℤ) (h : Even (n^2 + m^2 + n * m)) : ¬Odd (n + m) :=
sorry

end parity_implies_even_sum_l159_159554


namespace KA_bisects_BC_l159_159364

noncomputable section

open EuclideanGeometry

-- Definitions and conditions
variables {A B C M N K : Point}
variables (circle : Circle A B C)
variables (tangentB tangentC : Line)

-- Conditions
-- Ensure N is the intersection point of tangents at B and C
def tangent_at_B : tangentB = tangent_at_point circle B := sorry
def tangent_at_C : tangentC = tangent_at_point circle C := sorry
def intersection_point_N : intersection_point tangentB tangentC = N := sorry

-- Ensure M is on the circle and AM is parallel to BC
def M_on_circle : M ∈ circle := sorry
def AM_parallel_BC : Parallel (line_through A M) (line_through B C) := sorry

-- Ensure K is the intersection of MN and the circle
def K_intersection : ∃ K, K ∈ circle ∧ lies_on_line K (line_through M N) := sorry

-- Prove: K is the midpoint of BC 
theorem KA_bisects_BC (h : Prove_that_KA_bisects_BC) : midpoint K B C := sorry

end KA_bisects_BC_l159_159364


namespace quadratic_roots_square_cube_sum_l159_159690

theorem quadratic_roots_square_cube_sum
  (a b c : ℝ) (h : a ≠ 0) (x1 x2 : ℝ)
  (hx : ∀ (x : ℝ), a * x^2 + b * x + c = 0 ↔ x = x1 ∨ x = x2) :
  (x1^2 + x2^2 = (b^2 - 2 * a * c) / a^2) ∧
  (x1^3 + x2^3 = (3 * a * b * c - b^3) / a^3) :=
by
  sorry

end quadratic_roots_square_cube_sum_l159_159690


namespace circC_diameter_solution_l159_159415

noncomputable def circD_diameter : ℝ := 30
noncomputable def circD_radius : ℝ := circD_diameter / 2
noncomputable def circD_area : ℝ := π * (circD_radius ^ 2)

noncomputable def circC_diameter (d : ℝ) : ℝ := d
noncomputable def circC_radius (d : ℝ) : ℝ := d / 2
noncomputable def circC_area (d : ℝ) : ℝ := π * (circC_radius d) ^ 2

noncomputable def shaded_area (d : ℝ) : ℝ := circD_area - circC_area d

theorem circC_diameter_solution : 
  ∃ d : ℝ, shaded_area d / circC_area d = 7 ∧ d = 21.22 :=
by
  sorry

end circC_diameter_solution_l159_159415


namespace composite_shape_sum_l159_159830

def triangular_prism_faces := 5
def triangular_prism_edges := 9
def triangular_prism_vertices := 6

def pentagonal_prism_additional_faces := 7
def pentagonal_prism_additional_edges := 10
def pentagonal_prism_additional_vertices := 5

def pyramid_additional_faces := 5
def pyramid_additional_edges := 5
def pyramid_additional_vertices := 1

def resulting_shape_faces := triangular_prism_faces - 1 + pentagonal_prism_additional_faces + pyramid_additional_faces
def resulting_shape_edges := triangular_prism_edges + pentagonal_prism_additional_edges + pyramid_additional_edges
def resulting_shape_vertices := triangular_prism_vertices + pentagonal_prism_additional_vertices + pyramid_additional_vertices

def sum_faces_edges_vertices := resulting_shape_faces + resulting_shape_edges + resulting_shape_vertices

theorem composite_shape_sum : sum_faces_edges_vertices = 51 :=
by
  unfold sum_faces_edges_vertices resulting_shape_faces resulting_shape_edges resulting_shape_vertices
  unfold triangular_prism_faces triangular_prism_edges triangular_prism_vertices
  unfold pentagonal_prism_additional_faces pentagonal_prism_additional_edges pentagonal_prism_additional_vertices
  unfold pyramid_additional_faces pyramid_additional_edges pyramid_additional_vertices
  simp
  sorry

end composite_shape_sum_l159_159830


namespace unpainted_cubes_count_l159_159339

-- Definition of the cube and painting conditions
def cube_size : ℕ := 6
def painted_grid_size : ℕ := 4
def total_unit_cubes : ℕ := cube_size * cube_size * cube_size

-- Calculating the number of unpainted cubes
theorem unpainted_cubes_count :
  let total_unit_cubes := total_unit_cubes in
  let painted_per_face := painted_grid_size * painted_grid_size in
  let total_painted_initial := 6 * painted_per_face in
  let corners_per_face := 4 in
  let edges_per_face_excluding_corners := 12 in
  let unique_painted_per_face := corners_per_face + edges_per_face_excluding_corners in
  let total_unique_painted := 4 * 6 + 16 in
  let unpainted := total_unit_cubes - total_unique_painted
  in unpainted = 176 := by
  sorry

end unpainted_cubes_count_l159_159339


namespace sum_of_distinct_A_plus_B_l159_159375

theorem sum_of_distinct_A_plus_B :
  ∃ (A B : ℕ), (A < 10) ∧ (B < 10) ∧ ((100 * A + 10 * B + 4) % 8 = 0) ∧ 
                ([A + B | A < 10 ∧ B < 10 ∧ (100 * A + 10 * B + 4) % 8 = 0]
                .erase_duplicates.sum = 6) :=
by
  sorry

end sum_of_distinct_A_plus_B_l159_159375


namespace find_S6_l159_159282

variable {a : ℕ → ℝ} 
variable {S : ℕ → ℝ}

/-- sum_of_first_n_terms_of_geometric_sequence -/
def sum_of_first_n_terms_of_geometric_sequence (S : ℕ → ℝ) : Prop :=
  ∃ a1 r, ∀ n, S n = a1 * (1 - r^(n+1)) / (1 - r)

-- Given conditions
axiom geom_seq_positive_terms : ∀ n, a n > 0
axiom sum_S2 : S 2 = 3
axiom sum_S4 : S 4 = 15

theorem find_S6 : S 6 = 63 := by
  sorry

end find_S6_l159_159282


namespace sum_of_areas_of_rectangles_with_60_squares_l159_159700

theorem sum_of_areas_of_rectangles_with_60_squares :
  let n := 60 in 
  let areas := (finset.filter (λ p : ℕ × ℕ, p.1 * p.2 = n) (finset.Icc (1, 1) (n, n))).sum (λ p, p.1 * p.2) in
  areas = 360 :=
by {
  let n := 60,
  let pairs := finset.filter (λ p : ℕ × ℕ, p.1 * p.2 = n) (finset.Icc (1, 1) (n, n)),
  have h_distinct_pairs : pairs.card = 6, {
    sorry
  },
  have h_area_each : ∀ p ∈ pairs, p.1 * p.2 = n, {
    intros,
    simp at *,
    assumption
  },
  have h_sum_areas : ∑ p in pairs, n = n * pairs.card, {
    rw finset.sum_const,
    exact h_distinct_pairs
  },
  rw h_sum_areas,
  simp,
  exact eq.refl 360
}

end sum_of_areas_of_rectangles_with_60_squares_l159_159700


namespace train_boxcars_capacity_l159_159653

theorem train_boxcars_capacity :
  let red_boxcars := 3
  let blue_boxcars := 4
  let black_boxcars := 7
  let black_capacity := 4000
  let blue_capacity := black_capacity * 2
  let red_capacity := blue_capacity * 3
  (black_boxcars * black_capacity) + (blue_boxcars * blue_capacity) + (red_boxcars * red_capacity) = 132000 := by
  sorry

end train_boxcars_capacity_l159_159653


namespace average_of_remaining_two_numbers_l159_159320

theorem average_of_remaining_two_numbers (a b c d e f : ℝ) 
  (h1 : (a + b + c + d + e + f) / 6 = 8) 
  (h2 : (a + b + c + d) / 4 = 5) : 
  (e + f) / 2 = 14 := 
by  
  sorry

end average_of_remaining_two_numbers_l159_159320


namespace probability_two_red_two_blue_l159_159352

theorem probability_two_red_two_blue (total_red total_blue : ℕ) (red_taken blue_taken selected : ℕ)
  (h_red_total : total_red = 12) (h_blue_total : total_blue = 8) (h_selected : selected = 4)
  (h_red_taken : red_taken = 2) (h_blue_taken : blue_taken = 2) :
  (Nat.choose total_red red_taken) * (Nat.choose total_blue blue_taken) /
    (Nat.choose (total_red + total_blue) selected : ℚ) = 1848 / 4845 := 
by 
  sorry

end probability_two_red_two_blue_l159_159352


namespace roots_of_quadratic_l159_159271

theorem roots_of_quadratic (a b c : ℝ) (h_eq : a = 2 ∧ b = 3 ∧ c = -2) : 
  let Δ := b^2 - 4 * a * c in Δ > 0 → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 2 * x1^2 + 3 * x1 - 2 = 0 ∧ 2 * x2^2 + 3 * x2 - 2 = 0 := 
by
  intros
  sorry

end roots_of_quadratic_l159_159271


namespace arithmetic_sequence_properties_l159_159943

theorem arithmetic_sequence_properties (a : ℕ → ℕ) (c : ℕ → ℝ) (T : ℕ → ℝ) :
  (∀ n, a n = n) → (c n = 2 / (a (n + 1) * a (n + 2))) →
  (T n = ∑ i in finset.range n, c i) →
  (a 1 + a 2 + a 3 = 6) →
  (a 5 = 5) →
  (∀ n, T n = (n : ℝ) / (n + 2)) :=
by
  intros h1 h2 h3 h_cond1 h_cond2
  sorry

end arithmetic_sequence_properties_l159_159943


namespace necessary_sufficient_condition_for_real_roots_l159_159935

noncomputable theory

variables (n : ℕ) (A : ℂ) (x : ℝ)

-- Declare the assumptions:
-- n is a positive integer
axiom pos_n : n > 0

-- i is the imaginary unit
def i : ℂ := complex.I

-- A is a complex number
variable hA : A ≠ 0

-- Given equation
axiom eqn : (1 + i * x) / (1 - i * x) ^ n = A

-- The goal is to prove |A| = 1
theorem necessary_sufficient_condition_for_real_roots :
  |A| = 1 :=
sorry

end necessary_sufficient_condition_for_real_roots_l159_159935


namespace calc_result_l159_159854

theorem calc_result :
  12 / 4 - 3 - 16 + 4 * 6 = 8 := by
  sorry

end calc_result_l159_159854


namespace problem1_problem2_l159_159096

-- Define the lines l1 and l2
def line1 : LinearSpace ℝ := {x | 2 * x.1 - x.2 = 0}
def line2 : LinearSpace ℝ := {x | x.1 + x.2 + 2 = 0}

-- Problem 1
theorem problem1 : 
  ∃ l : LinearSpace ℝ, 
    (∀ p : ℝ × ℝ, p ∈ l → p = (1, 1)) ∧ 
    (∀ p q : ℝ × ℝ, p ∈ l → q ∈ line1 → orthogonal p q) → 
    ∀ x : ℝ × ℝ, x ∈ l ↔ 2 * x.2 + x.1 - 3 = 0 :=
sorry

-- Problem 2
theorem problem2 :
  ∃ (M : ℝ × ℝ → ℝ), 
    (∀ p : ℝ × ℝ, p ∈ M → p ∈ line1) ∧ 
    (∀ p : ℝ × ℝ, p ∈ M → p.1 = 0) ∧ 
    (forall c : ℝ × ℝ → ℝ, chord_intercepted_length c line2 = sqrt(2)) → 
    ∀ x : ℝ × ℝ, M x ↔ (x.1 + 5/7)^2 + (x.2 + 10/7)^2 = 25/49 ∨ 
                   (x.1 + 1)^2 + (x.2 + 2)^2 = 1 :=
sorry

end problem1_problem2_l159_159096


namespace minimum_value_PN_PM_l159_159960

section MathProofProblem

variable {m : ℝ}

def line_l (m : ℝ) (x y : ℝ) : Prop := (3 * m + 1) * x + (2 + 2 * m) * y - 8 = 0 
def line_l1 (x : ℝ) : Prop := x = -1
def line_l2 (y : ℝ) : Prop := y = -1

theorem minimum_value_PN_PM 
  (P : ℝ × ℝ)
  (hP : ∀ m, line_l m P.1 P.2)
  (M N : ℝ × ℝ)
  (hM : ∃ m, line_l m M.1 M.2 ∧ line_l1 M.1) 
  (hN : ∃ m, line_l m N.1 N.2 ∧ line_l2 N.2) :
  P = (-4, 6) ∧ ∃ k : ℝ, (|PM| : ℝ )^2 = 9 *(3 * (abs(k))^3)+ sqrt(3) *44*2/8 := by
  sorry

end MathProofProblem

end minimum_value_PN_PM_l159_159960


namespace rational_inequality_solution_l159_159660

variable (x : ℝ)

def inequality_conditions : Prop := (2 * x - 1) / (x + 1) > 1

def inequality_solution : Prop := x < -1 ∨ x > 2

theorem rational_inequality_solution : inequality_conditions x → inequality_solution x :=
by
  sorry

end rational_inequality_solution_l159_159660


namespace max_sum_of_squares_of_sides_of_inscribed_triangle_l159_159837

noncomputable section

variable {V : Type*} [inner_product_space ℝ V]

open Real


def sum_squares_of_sides_maximized (R : ℝ) (A B C : V) (O : V) 
  (hA : dist O A = R) (hB : dist O B = R) (hC : dist O C = R) : Prop :=
  let a := A - O
  let b := B - O
  let c := C - O
  (dist A B)^2 + (dist B C)^2 + (dist C A)^2 = 9 * R^2

theorem max_sum_of_squares_of_sides_of_inscribed_triangle (R : ℝ) (A B C : V) (O : V) 
  (hA : dist O A = R) (hB : dist O B = R) (hC : dist O C = R) : 
  sum_squares_of_sides_maximized R A B C O hA hB hC := sorry

end max_sum_of_squares_of_sides_of_inscribed_triangle_l159_159837


namespace sum_of_marked_intervals_leq_half_l159_159645

def line_segment := set.Icc (0 : ℝ) 1

def marked_intervals (I : set ℝ) :=
  I ⊆ line_segment ∧ ∀ x y ∈ I, x ≠ y → abs (x - y) ≠ 0.1

theorem sum_of_marked_intervals_leq_half :
  ∀ (intervals : set (set ℝ)),
    (∀ I ∈ intervals, marked_intervals I) →
    (intervals ⊆ line_segment) →
    sum (λ I, I.measure) intervals ≤ 0.5 := 
sorry

end sum_of_marked_intervals_leq_half_l159_159645


namespace letter_arrangements_4D_6E_5F_l159_159520

theorem letter_arrangements_4D_6E_5F :
  let countArrangements : ℕ := 
    (Finset.range 5).sum (λ j, Nat.choose 4 j * Nat.choose 6 (4 - j) * Nat.choose 5 j) in
  countArrangements = 
    (Finset.range 5).sum (λ j, Nat.choose 4 j * Nat.choose 6 (4 - j) * Nat.choose 5 j) :=
by
  sorry

end letter_arrangements_4D_6E_5F_l159_159520


namespace fraction_of_fliers_sent_out_l159_159312

-- Definitions based on the conditions
def total_fliers : ℕ := 2500
def fliers_next_day : ℕ := 1500

-- Defining the fraction sent in the morning as x
variable (x : ℚ)

-- The remaining fliers after morning
def remaining_fliers_morning := (1 - x) * total_fliers

-- The remaining fliers after afternoon
def remaining_fliers_afternoon := remaining_fliers_morning - (1/4) * remaining_fliers_morning

-- The theorem statement
theorem fraction_of_fliers_sent_out :
  remaining_fliers_afternoon = fliers_next_day → x = 1/5 :=
sorry

end fraction_of_fliers_sent_out_l159_159312


namespace max_possible_min_diff_l159_159503

def given_numbers : List ℝ := [10, 6, 13, 4, 18]

def circular_min_diff (l : List ℝ) : ℝ :=
  let circular_pairs := List.zip l (List.dropLast l ++ [List.head l])
  List.minimum (circular_pairs.map (λ ⟨a, b⟩, |a - b|))

theorem max_possible_min_diff : circular_min_diff given_numbers = 9 := sorry

end max_possible_min_diff_l159_159503


namespace vovochka_correct_sum_cases_vovochka_min_difference_l159_159160

-- Part (a)
theorem vovochka_correct_sum_cases : 
  (∑ (a : ℕ) in finset.range 10, (∑ (b : ℕ) in finset.range (10 - a), 1)) ^ 3 = 244620 :=
sorry

-- Part (b)
theorem vovochka_min_difference : 
  ∃ (a b c x y z : ℕ), (a * 100 + b * 10 + c + x * 100 + y * 10 + z) - (a + x) * 100 - (b + y) * 10 - (c + z) = 1800 :=
sorry

end vovochka_correct_sum_cases_vovochka_min_difference_l159_159160


namespace tangents_segment_equal_l159_159293

-- Define the conditions: existence of two circles with equal radii and common tangents
variables (O1 O2 : Point) (R : ℝ)
-- Assume circles have equal radii
axiom circles_with_equal_radii : dist O1 O2 = 2 * R

-- Define common tangents
variables (A B C D P Q : Point)
-- These tangents touch the circles at points A, B, C, D for external and P, Q for internal
axiom external_tangent_1 : tangent_at_circle A O1 R
axiom external_tangent_2 : tangent_at_circle B O1 R
axiom external_tangent_3 : tangent_at_circle C O2 R
axiom external_tangent_4 : tangent_at_circle D O2 R
axiom internal_tangent_1 : tangent_at_circle P O1 R
axiom internal_tangent_2 : tangent_at_circle Q O2 R

-- Definition of tangents being equal
def segment_length_equal (X Y Z W : Point) :=
dist X Y = dist Z W

-- Final statement proving the segments' length equality
theorem tangents_segment_equal :
  segment_length_equal P Q A B :=
sorry

end tangents_segment_equal_l159_159293


namespace vovochka_correct_sum_cases_vovochka_min_difference_l159_159156

-- Part (a)
theorem vovochka_correct_sum_cases : 
  (∑ (a : ℕ) in finset.range 10, (∑ (b : ℕ) in finset.range (10 - a), 1)) ^ 3 = 244620 :=
sorry

-- Part (b)
theorem vovochka_min_difference : 
  ∃ (a b c x y z : ℕ), (a * 100 + b * 10 + c + x * 100 + y * 10 + z) - (a + x) * 100 - (b + y) * 10 - (c + z) = 1800 :=
sorry

end vovochka_correct_sum_cases_vovochka_min_difference_l159_159156


namespace problem1_problem2_l159_159517

-- Define the vectors m and b based on x
def m (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
def b (x : ℝ) : ℝ × ℝ := (-Real.cos x, Real.sqrt 3 * Real.cos x)

-- Define the function f(x)
def f (x : ℝ) : ℝ := (m x).1 * ((1/2) * (m x).1) + (m x).2 * ((1/2) * (m x).2) - ((m x).1 * (b x).1 + (m x).2 * (b x).2)

-- Problem 1: Prove the range of x for f(x) >= 1/2
theorem problem1 (x : ℝ) (k : ℤ) : f x >= 1/2 ↔ x ∈ Set.Icc (k * Real.pi - Real.pi / 2) (k * Real.pi + Real.pi / 6) :=
sorry

-- Problem 2: Define variables and conditions for the triangle problem
variables (A B C a b c : ℝ)
variables (B_angle_cond f_B_cond : ℝ)

-- Given conditions
def B_angle : Prop := B = B_angle_cond
def f_condition : Prop := f (B / 2) = 1
def b_value : Prop := b = 1
def c_value : Prop := c = Real.sqrt 3

-- Problem 2: Prove a = 1 given the conditions
theorem problem2 (B_angle_cond : ℝ) (hB: B_angle) (hf: f_condition) (hb: b_value) (hc: c_value) : a = 1 :=
sorry

end problem1_problem2_l159_159517


namespace Vovochka_correct_pairs_count_l159_159145

def no_carry_pairs_count (digit_sum: ℕ → ℕ → Prop) : ℕ :=
  let count_pairs (lim: ℕ) : ℕ := (lim * (lim + 1)) / 2
  let digit_valid_pairs : ℕ := count_pairs 9
  (digit_valid_pairs * digit_valid_pairs) * 81

def digit_sum (x y: ℕ) : Prop := (x + y ≤ 9)

theorem Vovochka_correct_pairs_count :
  no_carry_pairs_count digit_sum = 244620 := by
  sorry

end Vovochka_correct_pairs_count_l159_159145


namespace incorrect_statements_l159_159570

def problem_statement_A (A : Prop) : Prop :=
  A = "A conjecture is a statement that has been proven to be true."

def problem_statement_D (D : Prop) : Prop :=
  D = "It is acceptable to include irrelevant information in a proof as long as the argument is logical."

theorem incorrect_statements (A D : Prop) (hA : problem_statement_A A) (hD : problem_statement_D D) : 
  ¬A ∧ ¬D :=
by
  sorry

end incorrect_statements_l159_159570


namespace reciprocal_of_fraction_diff_l159_159439

theorem reciprocal_of_fraction_diff : 
  (∃ (a b : ℚ), a = 1/4 ∧ b = 1/5 ∧ (1 / (a - b)) = 20) :=
sorry

end reciprocal_of_fraction_diff_l159_159439


namespace silver_medals_count_l159_159808

def total_medals := 67
def gold_medals := 19
def bronze_medals := 16
def silver_medals := total_medals - gold_medals - bronze_medals

theorem silver_medals_count : silver_medals = 32 := by
  -- Proof goes here
  sorry

end silver_medals_count_l159_159808


namespace sequence_general_formula_range_of_t_l159_159961

theorem sequence_general_formula (n : ℕ) (h: 1 ≤ n) : 
  ∀ (a : ℕ → ℕ),
  (a 1 = 2) → 
  (∀ (k : ℕ), k ≥ 2 → a k - a (k - 1) = 2 * k) →
  a n = n * (n + 1) :=
begin
  sorry
end

theorem range_of_t (t : ℝ) :
  (∀ b_n: ℕ → ℝ, 
  (∀ (n : ℕ), n > 0 → 
  b_n n = ∑ k in (Finset.range (2 * n + 1)).filter (λ x, n + 1 ≤ x), 1 / (k * (k + 1))) →
  t^2 - 2 * t + 1 / 6 > b_n n) ↔ 
  t < 0 ∨ t > 2 :=
begin
  sorry
end

end sequence_general_formula_range_of_t_l159_159961


namespace product_of_all_real_values_r_l159_159020

theorem product_of_all_real_values_r : (∏ r in {x | ∃ x_with_cond, (1/(3 * x_with_cond) = (r - x_with_cond)/8 ∧ ∀ xx, 3 * xx^2 - 3 * r * xx + 8 = 0 → x_with_cond = xx)}, r) = -32/3 := 
sorry

end product_of_all_real_values_r_l159_159020


namespace probability_of_event_l159_159357

open ProbabilityTheory

def a_n (n successes: ℕ) (n shots: ℕ) : ℚ := n successes / n shots

def successful_shot_probability : ℚ := 1 / 2

theorem probability_of_event :
  (a_6 = 1 / 2 ∧ ∀ n, (n: ℕ) < 6 → a_n ≤ 1 / 2) ⇔ (ℙ (independent_event (a_n) 6 full.set) = 5 / 64) :=
sorry

end probability_of_event_l159_159357


namespace line_with_largest_angle_of_inclination_l159_159782

theorem line_with_largest_angle_of_inclination :
  let L_A := ∀ x y, x + 2 * y + 3 = 0
  let L_B := ∀ x y, 2 * x - y + 1 = 0
  let L_C := ∀ x y, x + y + 1 = 0
  let L_D := ∀ x, x + 1 = 0
  (∃ x y, 2 * x - y + 1 = 0 ∧ (∀ a b, x + 2 * y + 3 = 0 → (∀ a b, x + y + 1 = 0 → (∀ x, x + 1 = 0 → true)))) :=
  sorry

end line_with_largest_angle_of_inclination_l159_159782


namespace curve_c_eq_rectangular_line_l_eq_rectangular_max_OA_OB_equals_2sqrt3_l159_159590

noncomputable def polar_to_rectangular_curve_c (θ : ℝ) : ℝ × ℝ :=
  let ρ := 2 * Real.cos θ
  (ρ * Real.cos θ, ρ * Real.sin θ)

noncomputable def polar_to_rectangular_line_l (θ : ℝ) : ℝ × ℝ :=
  let ρ := (3 / 2) / Real.cos (θ - Real.pi / 3)
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem curve_c_eq_rectangular :
  ∀ (x y : ℝ), ((∃ θ : ℝ, (x, y) = polar_to_rectangular_curve_c θ) ↔ (x - 1)^2 + y^2 = 1) :=
by
  sorry

theorem line_l_eq_rectangular :
  ∀ (x y : ℝ), ((∃ θ : ℝ, (x, y) = polar_to_rectangular_line_l θ) ↔ x + Real.sqrt 3 * y - 3 = 0) :=
by
  sorry

theorem max_OA_OB_equals_2sqrt3 :
  ∀ (θ : ℝ), let oa := 2 * Real.cos θ
                  ob := 2 * Real.cos (θ + Real.pi / 3)
              ∃ (θ : ℝ), oa + ob = 2 * Real.sqrt 3 :=
by
  sorry

end curve_c_eq_rectangular_line_l_eq_rectangular_max_OA_OB_equals_2sqrt3_l159_159590


namespace area_AKD_correct_l159_159596

structure Trapezoid :=
(base1 base2 side1 side2 : ℝ) (diagonalIntersection : ℝ)

// Definitions based on given conditions
def AB := 27
def DC := 18
def AD := 3
def BC := 6 * Real.sqrt 2

def area_of_triangle_AKD (t : Trapezoid) : ℝ :=
  if t.base1 = AB ∧ t.base2 = DC ∧ t.side1 = AD ∧ t.side2 = BC
  then (54 * Real.sqrt 2) / 5
  else 0

theorem area_AKD_correct (t : Trapezoid) :
  t.base1 = AB → t.base2 = DC → t.side1 = AD → t.side2 = BC → 
  area_of_triangle_AKD t = (54 * Real.sqrt 2) / 5 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end area_AKD_correct_l159_159596


namespace option_d_is_deductive_l159_159310

theorem option_d_is_deductive :
  (∀ (r : ℝ), S_r = Real.pi * r^2) → (S_1 = Real.pi) :=
by
  sorry

end option_d_is_deductive_l159_159310


namespace parabola_intersects_x_axis_expression_l159_159942

theorem parabola_intersects_x_axis_expression (m : ℝ) (h : m^2 - m - 1 = 0) : m^2 - m + 2017 = 2018 := 
by 
  sorry

end parabola_intersects_x_axis_expression_l159_159942


namespace nth_inequality_l159_159643

theorem nth_inequality (n : ℕ) : ∑ k in Finset.range (n + 1), Real.sqrt (k * (k + 1)) < (n + 1) ^ 2 / 2 := 
sorry

end nth_inequality_l159_159643


namespace number_of_n_for_integer_l159_159919

theorem number_of_n_for_integer (i : ℂ) (h_i : i^2 = -1) : 
  {n : ℤ | ((n : ℂ) + i)^4 ∈ ℤ}.to_finset.card = 3 := 
by 
  sorry

end number_of_n_for_integer_l159_159919


namespace coplanar_sufficient_conditions_l159_159047

-- Define conditions
def condition_1 (L1 L2 L3 : AffineLine ℝ 3) : Prop :=
  (L1 ≠ L2) ∧ (L2 ≠ L3) ∧ (L3 ≠ L1) ∧ ∃ (p q r : AffinePoint ℝ 3), p ∈ L1 ∧ p ∈ L2 ∧ q ∈ L2 ∧ q ∈ L3 ∧ r ∈ L3 ∧ r ∈ L1

def condition_2 (L1 L2 L3 : AffineLine ℝ 3) : Prop :=
  L1 ∥ L2 ∧ L2 ∥ L3

def condition_3 (L1 L2 L3 : AffineLine ℝ 3) : Prop :=
  ∃ (p : AffinePoint ℝ 3), p ∈ L1 ∧ p ∈ L2 ∧ p ∈ L3

def condition_4 (L1 L2 L3 : AffineLine ℝ 3) : Prop :=
  ∃ (L : AffineLine ℝ 3), (L ∥ L1 ∧ L ∥ L2 ∧ L3 ∩ (L1 ∪ L2).nonempty)

-- Define coplanar condition
def coplanar (L1 L2 L3 : AffineLine ℝ 3) : Prop :=
  ∃ (p q r : AffinePoint ℝ 3), ∃ (s t u : ℝ), p + s • q + t • r = 0

-- Formalize the proof as a statement
theorem coplanar_sufficient_conditions (L1 L2 L3 : AffineLine ℝ 3) :
  (condition_1 L1 L2 L3 ∨ condition_4 L1 L2 L3) → coplanar L1 L2 L3 :=
by
  sorry

end coplanar_sufficient_conditions_l159_159047


namespace probability_palindrome_div_13_correct_l159_159340

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def seven_digit_palindrome (n : ℕ) : Prop :=
  n >= 1000000 ∧ n < 10000000 ∧ is_palindrome n

def probability_palindrome_div_13 : ℚ :=
  let palindromes := (finset.filter seven_digit_palindrome (finset.Ico 1000000 10000000)).to_list
  let palindromes_div_13 := palindromes.filter (λ n, is_palindrome (n / 13))
  ((palindromes_div_13.length : ℚ) / palindromes.length).to_rat

theorem probability_palindrome_div_13_correct :
  probability_palindrome_div_13 = 5/143 := sorry

end probability_palindrome_div_13_correct_l159_159340


namespace reflex_angle_at_H_l159_159049

theorem reflex_angle_at_H 
  (C D F M H : Type*)
  (point_on_line : ∀ (P : Type*), P = C ∨ P = D ∨ P = F ∨ P = M)
  (angle_CDH : ∠CDH = 150)
  (angle_HFM : ∠HFM = 95) : 
  (reflex_angle_h : ∀ (H : Type*), 360 - 30 - 85 - 65 = 180) :=
sorry

end reflex_angle_at_H_l159_159049


namespace find_k_l159_159822

-- Define the line equation and slopes
def line_eq_slope : ℝ := 4
def line_eq_intercept : ℝ := 6
def line_eq_const : ℝ := 18

-- Define the points
def point1 : Prod ℝ ℝ := (3, -12)
def point2_y : ℝ := 24
def k : ℝ := 57

-- Define the slope of the original line
def line_slope : ℝ := 2 / 3

-- Define the slope formula for the two points
def points_slope (k : ℝ) : ℝ := (point2_y + 12) / (k - 3)

-- Prove that the points_slope equals the line_slope
theorem find_k : parts_slope k = line_slope → k = 57 := by
  sorry

end find_k_l159_159822


namespace distinct_m_values_count_l159_159624

theorem distinct_m_values_count :
  ∃ (m_values : Finset ℤ), (∀ x1 x2 : ℤ, x1 * x2 = 36 → m_values ∈ { x1 + x2 }) ∧ m_values.card = 10 :=
by
  sorry

end distinct_m_values_count_l159_159624


namespace intersect_on_median_l159_159325

open_locale real

structure Triangle :=
(A B C : Point)
(α β γ : Angle)
(right_angle : β = 90)
(incircle : Circle)
(touches : ∀ {P}, P ∈ incircle → is_tangent P [A, B, C])
(symmetry_A2 : reflection B C B1 = A2)
(symmetry_C2 : reflection A B B1 = C2)
(median : Line := midpoint B C)

theorem intersect_on_median (T : Triangle)
(A1 A2 C1 C2 : Point) (B1 : Point) :
T.touches C1 → T.touches A1 → 
T.touches B1 →
T.symmetry_A2 → T.symmetry_C2 →
Intersection (line_through A1 A2) (line_through C1 C2) ∈ T.median :=
sorry

end intersect_on_median_l159_159325


namespace trajectory_equation_maximum_area_triangle_OAB_l159_159175

theorem trajectory_equation (x y : ℝ) (M : ℝ × ℝ) (d : ℝ) (P : ℝ × ℝ) :
  M = (1, 0) → P = (x, y) → d = abs (x - 2) →
  ((real.sqrt ((x - 1)^2 + y^2)) / d = real.sqrt 2 / 2) →
  (x^2 / 2 + y^2 = 1) :=
sorry

theorem maximum_area_triangle_OAB (A B D O : ℝ × ℝ) (x1 y1 x2 y2 : ℝ) :
  A = (x1, y1) → B = (x2, y2) →
  ((x1^2)/2 + y1^2 = 1) → ((x2^2)/2 + y2^2 = 1) →
  (D = ((x1 + x2) / 2, (y1 + y2) / 2)) →

  let m := sqrt 6 / 2 in
  let l := (λ x, -x + m) in
  let h := (real.sqrt 2) * abs m in
  let AB := real.sqrt ((3*m^2)/2) in
  (1 / 2 * h * AB = sqrt 2 / 2) ∧
  (l = - (x) + sqrt 6 / 2 ∨ l = - (x) - sqrt 6 / 2) :=
sorry

end trajectory_equation_maximum_area_triangle_OAB_l159_159175


namespace ratio_unit_price_XY_l159_159402

-- Define the volume and price of Brand Y soda
variables (v p : ℝ) (hv : v ≠ 0) (hp : p ≠ 0)

-- Define the conditions for Brand X
def volume_X := 1.3 * v
def price_X := 0.8 * p

-- The unit prices
def unit_price_Y : ℝ := p / v
def unit_price_X : ℝ := price_X / volume_X

-- The theorem to prove the ratio of unit prices
theorem ratio_unit_price_XY : unit_price_X / unit_price_Y = 8 / 13 :=
by
  -- You can complete the proof here with the required steps.
  sorry

end ratio_unit_price_XY_l159_159402


namespace sum_cubes_eq_neg_27_l159_159200

theorem sum_cubes_eq_neg_27 (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
 (h_eq : (a^3 + 9) / a = (b^3 + 9) / b ∧ (b^3 + 9) / b = (c^3 + 9) / c) :
 a^3 + b^3 + c^3 = -27 :=
by
  sorry

end sum_cubes_eq_neg_27_l159_159200


namespace oranges_total_and_avg_cost_l159_159847

-- Conditions
def rate_4_for_15 := 15 / 4
def rate_7_for_25 := 25 / 7
def purchase_groups (n : ℕ) : ℕ := 7 * n
def total_cost (n : ℕ) : ℕ := 25 * n

-- Problem statement: Prove if buying 28 oranges, the total cost is 100 cents and the average cost per orange is 25/7 cents.
theorem oranges_total_and_avg_cost (n : ℕ) (h : purchase_groups n = 28) : 
  total_cost n = 100 ∧ (total_cost n) / 28 = (25 / 7 : ℚ) :=
by
  sorry

end oranges_total_and_avg_cost_l159_159847


namespace Vovochka_correct_pairs_count_l159_159143

def no_carry_pairs_count (digit_sum: ℕ → ℕ → Prop) : ℕ :=
  let count_pairs (lim: ℕ) : ℕ := (lim * (lim + 1)) / 2
  let digit_valid_pairs : ℕ := count_pairs 9
  (digit_valid_pairs * digit_valid_pairs) * 81

def digit_sum (x y: ℕ) : Prop := (x + y ≤ 9)

theorem Vovochka_correct_pairs_count :
  no_carry_pairs_count digit_sum = 244620 := by
  sorry

end Vovochka_correct_pairs_count_l159_159143


namespace odd_and_increasing_l159_159839

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f(x)

def increasing_on_domain (f : ℝ → ℝ) : Prop := 
  ∀ x y : ℝ, x < y → f(x) ≤ f(y)

def function_B (x : ℝ) : ℝ := x * abs(x)

theorem odd_and_increasing :
  is_odd_function function_B ∧ increasing_on_domain function_B :=
by
  split;
  sorry

end odd_and_increasing_l159_159839


namespace remainder_of_largest_divided_by_second_largest_l159_159287

theorem remainder_of_largest_divided_by_second_largest :
  let numbers := [10, 11, 12, 13, 14] in
  let largest := list.maximum numbers in
  let second_largest := list.maximum (numbers.erase largest) in
  largest % second_largest = 1 :=
by
  let numbers := [10, 11, 12, 13, 14]
  let largest := list.maximum numbers
  let second_largest := list.maximum (numbers.erase largest)
  sorry

end remainder_of_largest_divided_by_second_largest_l159_159287


namespace agatha_bike_budget_l159_159834

def total_initial : ℕ := 60
def cost_frame : ℕ := 15
def cost_front_wheel : ℕ := 25
def total_spent : ℕ := cost_frame + cost_front_wheel
def total_left : ℕ := total_initial - total_spent

theorem agatha_bike_budget : total_left = 20 := by
  sorry

end agatha_bike_budget_l159_159834


namespace vovochka_correct_sum_cases_vovochka_min_difference_l159_159159

-- Part (a)
theorem vovochka_correct_sum_cases : 
  (∑ (a : ℕ) in finset.range 10, (∑ (b : ℕ) in finset.range (10 - a), 1)) ^ 3 = 244620 :=
sorry

-- Part (b)
theorem vovochka_min_difference : 
  ∃ (a b c x y z : ℕ), (a * 100 + b * 10 + c + x * 100 + y * 10 + z) - (a + x) * 100 - (b + y) * 10 - (c + z) = 1800 :=
sorry

end vovochka_correct_sum_cases_vovochka_min_difference_l159_159159


namespace fibonacci_identity_l159_159666

noncomputable def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

theorem fibonacci_identity (n : ℕ) (h : 1 ≤ n) :
  fibonacci (2*n - 1)^2 + fibonacci (2*n + 1)^2 + 1 = 3 * fibonacci (2*n - 1) * fibonacci (2*n + 1) :=
sorry

end fibonacci_identity_l159_159666


namespace polynomial_degree_and_terms_l159_159592

-- Definition for polynomial and its corresponding properties
def polynomial := x^2 + 2*x + 18

-- The degree of the polynomial should be 2
def degree_of_polynomial : ℕ := 2

-- The number of distinct terms in the polynomial should be 3
def number_of_terms : ℕ := 3

-- Proof statement for degree and number of terms
theorem polynomial_degree_and_terms : 
  ∃ (d n : ℕ), d = degree_of_polynomial ∧ n = number_of_terms :=
by
  exact ⟨2, 3, rfl, rfl⟩

end polynomial_degree_and_terms_l159_159592


namespace monotonic_increase_interval_l159_159480

theorem monotonic_increase_interval 
  (even_func : ∀ x : ℝ, 2 * sin (ω * x + φ) = 2 * sin (ω * (-x) + φ))
  (omega_pos : 0 < ω)
  (phi_range : 0 < φ ∧ φ < π)
  (y_intersects : ∃ x1 x2 : ℝ, 2 * sin (ω * x1 + φ) = 2 ∧ 2 * sin (ω * x2 + φ) = 2 ∧ x1 ≠ x2)
  (min_abs_diff : ∀ x1 x2 : ℝ, 2 * sin (ω * x1 + φ) = 2 ∧ 2 * sin (ω * x2 + φ) = 2 → |x2 - x1| ≥ π) :
  ∃ a b : ℝ, a = -π / 2 ∧ b = -π / 4 ∧ ∀ x1 x2 : ℝ, a ≤ x1 ∧ x1 < x2 ∧ x2 ≤ b → 2 * sin (ω * x1 + φ) < 2 * sin (ω * x2 + φ) :=
sorry

end monotonic_increase_interval_l159_159480


namespace fraction_of_stickers_unclaimed_l159_159390

theorem fraction_of_stickers_unclaimed (x : ℕ) :
  let a := (4 / 10 : ℚ) * x
  let b := (3 / 10 : ℚ) * (x - a)
  let c := (2 / 10 : ℚ) * (x - a - b)
  let d := (1 / 10 : ℚ) * (x - a - b - c)
  (x - (a + b + c + d)) / x = 2844 / 10000 :=
by
  intros
  let a := (4 / 10 : ℚ) * x
  let b := (3 / 10 : ℚ) * (x - a)
  let c := (2 / 10 : ℚ) * (x - a - b)
  let d := (1 / 10 : ℚ) * (x - a - b - c)
  have h : (x - (a + b + c + d)) = (2844 / 10000) * x := sorry
  have h' : (x - (a + b + c + d)) / x = 2844 / 10000 := by
    rw [(x - (a + b + c + d))] at h
    exact h
  exact h'

end fraction_of_stickers_unclaimed_l159_159390


namespace measure_of_B_area_of_triangle_l159_159994

noncomputable def triangle_sides_conditions (a b c : ℝ) (A B C : ℝ) : Prop :=
(2 * a - c) * Real.cos B = b * Real.cos C

theorem measure_of_B (a b c A B C : ℝ) (h1 : B * Real.pi / 180 ∈ Icc 0 Real.pi) 
(h2 : (2 * a - c) * Real.cos B = b * Real.cos C) 
(h3 : a = 2 * Real.sin A) (h4 : b = 2 * Real.sin B) (h5 : c = 2 * Real.sin C) :
B = Real.pi / 3 :=
sorry

theorem area_of_triangle (a b c A B C : ℝ) (h1 : b = Real.sqrt 7) 
(h2 : a + c = 4) (h3 : B = Real.pi / 3) (h4 : a * c = 3) :
(1 / 2) * a * c * Real.sin B = 3 * Real.sqrt 3 / 4 :=
sorry

end measure_of_B_area_of_triangle_l159_159994


namespace vasya_birthday_is_thursday_l159_159735

def vasya_birthday_day_of_week (today_is_friday : Bool) (sunday_day_after_tomorrow : Bool) : String :=
  if today_is_friday && sunday_day_after_tomorrow then "Thursday" else "Unknown"

theorem vasya_birthday_is_thursday
  (today_is_friday : true)
  (sunday_day_after_tomorrow : true) : 
  vasya_birthday_day_of_week true true = "Thursday" := 
by
  -- assume today is Friday
  have h1 : today_is_friday = true := rfl
  -- assume Sunday is the day after tomorrow
  have h2 : sunday_day_after_tomorrow = true := rfl
  -- by our function definition, Vasya's birthday should be Thursday
  show vasya_birthday_day_of_week true true = "Thursday"
  sorry

end vasya_birthday_is_thursday_l159_159735


namespace intersect_line_curve_C1_find_intersection_polar_coords_l159_159087

noncomputable def line_parametric (t : ℝ) (α : ℝ) : (ℝ × ℝ) := (-1 + t * Real.cos α, 1 + t * Real.sin α)
noncomputable def curve_C1 (t : ℝ) : (ℝ × ℝ) := (2 + 2 * Real.cos t, 4 + 2 * Real.sin t)
noncomputable def curve_C2 (θ : ℝ) : ℝ := 4 * Real.cos θ

theorem intersect_line_curve_C1 (α : ℝ) (t₀ : ℝ) (t₁ : ℝ) :
  let (x1, y1) := line_parametric t₀ α,
      (x2, y2) := line_parametric t₁ α,
      (x3, y3) := curve_C1 t₀,
      (x4, y4) := curve_C1 t₁ in
  (y2 - y1) / (x2 - x1) = 2 → ∃ t, (2 + 2 * Real.cos t, 4 + 2 * Real.sin t) = line_parametric t α :=
sorry

theorem find_intersection_polar_coords (t : ℝ) :
  let (x, y) := curve_C1 t in
  4 * Real.cos (Real.atan2 y (x - 4)) = ρ → ρ = 2 * Real.sqrt 2 ∧ Real.atan2 y (x - 4) = π / 4 :=
sorry

end intersect_line_curve_C1_find_intersection_polar_coords_l159_159087


namespace evaluate_expression_at_a_eq_2_l159_159426

theorem evaluate_expression_at_a_eq_2 : 
  ∀ (a : ℝ), a = 2 → (7 * a ^ 2 - 15 * a + 5) * (3 * a - 4) = 6 := 
by
  intros a ha
  rw ha
  sorry

end evaluate_expression_at_a_eq_2_l159_159426


namespace euler_formula_first_quadrant_l159_159880

noncomputable def euler_formula (x : ℝ) : ℂ := real.exp (complex.I * x)

theorem euler_formula_first_quadrant :
  let z := euler_formula (π / 3)
  in 0 < z.re ∧ 0 < z.im :=
by
  let z : ℂ := euler_formula (π / 3)
  have : z = complex.cos (π / 3) + complex.sin (π / 3) * complex.I := sorry
  have : 0 < complex.cos (π / 3) := sorry
  have : 0 < complex.sin (π / 3) := sorry
  exact ⟨this, this⟩

end euler_formula_first_quadrant_l159_159880


namespace plot_length_is_80_l159_159382

noncomputable def length_of_plot (L W P d : ℕ) : Prop :=
  W = 50 ∧ d = 5 ∧ P = 56 → 
  2 * ((L / d) + 1) + 2 * ((W / d) + 1) = P → L = 80

theorem plot_length_is_80 : length_of_plot :=
begin
  sorry
end

end plot_length_is_80_l159_159382


namespace arithmetic_sequence_properties_l159_159695

theorem arithmetic_sequence_properties {S_n a_n b_n : ℕ → ℕ} :
  a_n 3 = 9 →
  S_n 6 = 60 →
  (∀ n : ℕ, b_n (n + 1) - b_n n = a_n n) →
  b_n 1 = 3 →
  (∀ n : ℕ, a_n n = 2 * n + 3) ∧
  (∀ n : ℕ, T_n n = (3 / 4) - (1 / (2 * (n + 1))) - (1 / (2 * (n + 2)))) :=
by
  sorry

end arithmetic_sequence_properties_l159_159695


namespace triangle_third_side_lengths_l159_159525

theorem triangle_third_side_lengths : 
  ∃ (x : ℕ), (3 < x ∧ x < 11) ∧ (x ≠ 3) ∧ (x ≠ 11) ∧ 
    ((x = 4) ∨ (x = 5) ∨ (x = 6) ∨ (x = 7) ∨ (x = 8) ∨ (x = 9) ∨ (x = 10)) :=
by
  sorry

end triangle_third_side_lengths_l159_159525


namespace wrench_force_inverse_variation_l159_159268

theorem wrench_force_inverse_variation :
  ∃ (k : ℝ), (∀ (L : ℝ) (F : ℝ), L * F = k) →
  (∃ F : ℝ, ∀ L : ℝ, L = 12 → F = 300 → k = 3600) →
  (∀ F : ℝ, ∃ L : ℝ, L = 18 → F * L = 3600 → F = 200) :=
begin
  --
  sorry
end

end wrench_force_inverse_variation_l159_159268


namespace sports_club_members_l159_159168

theorem sports_club_members :
  ∀ (N B T Neither : ℕ), N = 27 → B = 17 → T = 19 → Neither = 2 →
    (B + T - (N - Neither) = 11) :=
by
  intros N B T Neither hN hB hT hNeither
  rw [hN, hB, hT, hNeither]
  -- Proving the equality
  have h : 17 + 19 - (27 - 2) = 11 := by sorry
  exact h

end sports_club_members_l159_159168


namespace minimum_positive_sum_l159_159879

open Nat

theorem minimum_positive_sum 
    (a : Fin 200 → ℤ) 
    (h1 : ∀ i, a i = 1 ∨ a i = -1) :
    ∃ m > 0, m = ∑ i in Finset.range 200, ∑ j in Finset.Ico i.succ 200, a i * a j ∧ m = 28 :=
by
    sorry

end minimum_positive_sum_l159_159879


namespace days_to_finish_by_b_l159_159317

theorem days_to_finish_by_b (A B C : ℚ) 
  (h1 : A + B + C = 1 / 5) 
  (h2 : A = 1 / 9) 
  (h3 : A + C = 1 / 7) : 
  1 / B = 12.115 :=
by
  sorry

end days_to_finish_by_b_l159_159317


namespace percentage_boys_from_school_A_l159_159995

theorem percentage_boys_from_school_A (P : ℝ) (h1 : 0.3 * (P / 100) * 550 = 77) (h2 : ¬ 30 / 100 ∗ 550) : 
  P = 20 :=
by
  sorry

end percentage_boys_from_school_A_l159_159995


namespace Rachel_homework_difference_l159_159234

theorem Rachel_homework_difference (m r : ℕ) (hm : m = 8) (hr : r = 14) : r - m = 6 := 
by 
  sorry

end Rachel_homework_difference_l159_159234


namespace vasya_birthday_is_thursday_l159_159730

def vasya_birthday_day_of_week (today_is_friday : Bool) (sunday_day_after_tomorrow : Bool) : String :=
  if today_is_friday && sunday_day_after_tomorrow then "Thursday" else "Unknown"

theorem vasya_birthday_is_thursday
  (today_is_friday : true)
  (sunday_day_after_tomorrow : true) : 
  vasya_birthday_day_of_week true true = "Thursday" := 
by
  -- assume today is Friday
  have h1 : today_is_friday = true := rfl
  -- assume Sunday is the day after tomorrow
  have h2 : sunday_day_after_tomorrow = true := rfl
  -- by our function definition, Vasya's birthday should be Thursday
  show vasya_birthday_day_of_week true true = "Thursday"
  sorry

end vasya_birthday_is_thursday_l159_159730


namespace mass_of_BaCl2_produced_l159_159949

theorem mass_of_BaCl2_produced :
  (∀ (initial_moles_BaCl2 : ℝ) (initial_moles_NaOH : ℝ),
    initial_moles_BaCl2 = 8 → initial_moles_NaOH = 12 →
    (∃ (molar_mass_BaCl2 : ℝ) (mol_product : ℝ),
      molar_mass_BaCl2 = 208.23 ∧ mol_product = initial_moles_BaCl2 ∧
      mass_BaCl2 = mol_product * molar_mass_BaCl2 ∧
      mass_BaCl2 = 1665.84)) :=
by
  intros initial_moles_BaCl2 initial_moles_NaOH h1 h2
  use 208.23
  use initial_moles_BaCl2
  split
  case h1_1 =>
    rfl
  case h1_2 =>
    split
    case h1_2_1 =>
      exact h1
    case h1_2_2 =>
      split
      case h1_2_2_1 =>
        unfold mol_product
        exact h1
      case h1_2_2_2 =>
        unfold mass_BaCl2
        exact h1

end mass_of_BaCl2_produced_l159_159949


namespace Vasya_birthday_on_Thursday_l159_159723

-- Define the days of the week enumeration
inductive DayOfWeek
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open DayOfWeek

-- Define the conditions
def Sunday_is_the_day_after_tomorrow (today : DayOfWeek) : Prop :=
  match today with
  | Friday => true
  | _      => false

def Vasya_birthday_not_on_Sunday (birthday : DayOfWeek) : Prop :=
  birthday ≠ Sunday

def Today_is_next_day_after_birthday (today birthday : DayOfWeek) : Prop :=
  match (today, birthday) with
  | (Friday, Thursday) => true
  | _                  => false

-- The main theorem to prove Vasya's birthday was on Thursday
theorem Vasya_birthday_on_Thursday (birthday today : DayOfWeek) 
  (H1: Sunday_is_the_day_after_tomorrow today) 
  (H2: Vasya_birthday_not_on_Sunday birthday) 
  (H3: Today_is_next_day_after_birthday today birthday) : 
  birthday = Thursday :=
by
  sorry

end Vasya_birthday_on_Thursday_l159_159723


namespace sum_of_cubes_eq_neg_27_l159_159204

theorem sum_of_cubes_eq_neg_27
  (a b c : ℝ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_condition : (a^3 + 9) / a = (b^3 + 9) / b ∧ (b^3 + 9) / b = (c^3 + 9) / c) :
  a^3 + b^3 + c^3 = -27 :=
sorry

end sum_of_cubes_eq_neg_27_l159_159204


namespace math_competition_contestants_l159_159165

/-- 
In a math competition, 5 problems were assigned. 
There were no two contestants who solved exactly the same problems. 
For any problem that is disregarded, for each contestant, 
there is another contestant who solved the same set of the remaining 4 problems.
We need to prove that the number of contestants is 32.
-/
theorem math_competition_contestants : 
  ∃ n : ℕ, n = 32 ∧ 
    ∀ contestants : Finset (Fin 5 → Bool), 
      (∀ c1 c2 ∈ contestants, (c1 ≠ c2) → (c1 ≠ᵀ[Fin 5] c2)) ∧
      (∀ p ∈ Finset.univ, ∀ c1 ∈ contestants, ∃ c2 ∈ contestants, 
        (∀ q ∈ Finset.erase Finset.univ p, c1 q = c2 q) ∧ c1 q ≠ c2 q) :=
by
  use 32
  sorry

end math_competition_contestants_l159_159165


namespace value_of_Priyanka_l159_159109

-- Defining the context with the conditions
variables (X : ℕ) (Neha : ℕ) (Sonali Priyanka Sadaf Tanu : ℕ)
-- The conditions given in the problem
axiom h1 : Neha = X
axiom h2 : Sonali = 15
axiom h3 : Priyanka = 15
axiom h4 : Sadaf = Neha
axiom h5 : Tanu = Neha

-- Stating the theorem we need to prove
theorem value_of_Priyanka : Priyanka = 15 :=
by
  sorry

end value_of_Priyanka_l159_159109


namespace minimum_force_to_submerge_l159_159298

-- Definitions of given constants
def cube_density : ℝ := 400 -- kg/m^3
def water_density : ℝ := 1000 -- kg/m^3
def cube_volume_cm3 : ℝ := 10 -- cm^3
def gravity : ℝ := 10 -- m/s^2

-- Conversion factor from cm^3 to m^3
def cm3_to_m3 (v : ℝ) : ℝ := v * 10^(-6)

-- Given the conditions, the proof goal is to show that the minimum force required to submerge the cube is 0.06 N
theorem minimum_force_to_submerge : 
  let cube_volume_m3 := cm3_to_m3 cube_volume_cm3 in
  let cube_mass := cube_density * cube_volume_m3 in
  let cube_weight := cube_mass * gravity in
  let buoyant_force := water_density * cube_volume_m3 * gravity in
  buoyant_force - cube_weight = 0.06 := by
  sorry

end minimum_force_to_submerge_l159_159298


namespace triangle_area_ratio_l159_159124

noncomputable def area_ratio_triangle (X Y Z W : ℝ) (h : ℝ) (XW WZ : ℝ) : Prop :=
  XW = 4 ∧ WZ = 14 ∧ W = 4 + 14 ∧ 
  (1/2 * XW * h) / (1/2 * WZ * h) = 2 / 7

theorem triangle_area_ratio (X Y Z W : ℝ) (h : ℝ) (XW WZ : ℝ) : 
  XW = 4 → WZ = 14 → W = 4 + 14 → (area_ratio_triangle X Y Z W h XW WZ) :=
by
  intros
  rw area_ratio_triangle
  sorry

end triangle_area_ratio_l159_159124


namespace old_machine_rate_correct_l159_159823

noncomputable def old_machine_rate (R : ℕ) := R
def new_machine_rate : ℕ := 150
def total_combined_bolts : ℕ := 450
def operation_time_minutes : ℕ := 108
def operation_time_hours : ℝ := 108 / 60

theorem old_machine_rate_correct (R : ℕ) (h : R + new_machine_rate * operation_time_hours = total_combined_bolts) : old_machine_rate R = 100 :=
by
  sorry

end old_machine_rate_correct_l159_159823


namespace volume_of_pyramid_l159_159863

noncomputable def triangle_pyramid_volume (A B C : ℝ × ℝ) : ℝ :=
  let D := ((B.1 + C.1) / 2, (B.2 + C.2) / 2) in
  let E := ((A.1 + C.1) / 2, (A.2 + C.2) / 2) in
  let F := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) in
  let DE := (real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2)) in
  let EF := (real.sqrt ((E.1 - F.1)^2 + (E.2 - F.2)^2)) in
  let FD := (real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2)) in
  let s := (DE + EF + FD) / 2 in
  let A_DEF := real.sqrt (s * (s - DE) * (s - EF) * (s - FD)) in
  (1 / 3) * A_DEF * 10

theorem volume_of_pyramid :
  triangle_pyramid_volume (0, 0) (30, 0) (15, 20) ≈ 2436.11 :=
by sorry

end volume_of_pyramid_l159_159863


namespace bernardo_probability_correct_l159_159400

noncomputable def probability_bernardo_larger_than_silvia : ℚ := 
  265 / 440

theorem bernardo_probability_correct :
  let B := { x : set ℕ // x ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} ∧ x.card = 3 ∧ ∃ y ∈ x, y % 2 = 1 }
  let S := { y : set ℕ // y ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ y.card = 3 ∧ ∃ z ∈ y, z % 2 = 1 }
  probability_bernardo_larger_than_silvia = 265 / 440 :=
sorry

end bernardo_probability_correct_l159_159400


namespace min_force_to_submerge_cube_l159_159300

theorem min_force_to_submerge_cube :
  ∀ (V : ℝ) (ρ_cube ρ_water : ℝ) (g : ℝ),
  V = 10 * 10^(-6) →  -- volume in m^3
  ρ_cube = 400 →      -- density of cube in kg/m^3
  ρ_water = 1000 →    -- density of water in kg/m^3
  g = 10 →            -- acceleration due to gravity in m/s^2
  (ρ_water * V * g - ρ_cube * V * g = 0.06) :=
begin
  intros V ρ_cube ρ_water g,
  sorry
end

end min_force_to_submerge_cube_l159_159300


namespace max_discarded_grapes_l159_159810

theorem max_discarded_grapes (n : ℕ) : ∃ r, r < 8 ∧ n % 8 = r ∧ r = 7 :=
by
  sorry

end max_discarded_grapes_l159_159810


namespace sum_of_solutions_eq_zero_l159_159443

theorem sum_of_solutions_eq_zero : 
  let eqn := λ x : ℤ, x^4 - 13 * x^2 + 36 in
  ∑ x in { x : ℤ | eqn x = 0 }.toFinset = 0 :=
by sorry

end sum_of_solutions_eq_zero_l159_159443


namespace f_monotonically_decreasing_f_max_value_f_min_value_l159_159950
-- Definitions and conditions provided
def f (x : ℝ) := x / (x - 1)

-- Prove that for all x in [2, 5], f(x) is monotonically decreasing
theorem f_monotonically_decreasing : ∀ {x1 x2 : ℝ}, (2 ≤ x1) → (x1 ≤ 5) → (2 ≤ x2) → (x2 ≤ 5) → x1 < x2 → f x1 > f x2 :=
by sorry

-- Find the maximum and minimum values of f(x) in [2, 5]
theorem f_max_value : f 2 = 2 :=
by sorry

theorem f_min_value : f 5 = 5 / 4 :=
by sorry

end f_monotonically_decreasing_f_max_value_f_min_value_l159_159950


namespace coprime_four_digit_integers_diff_at_least_4000_l159_159039

theorem coprime_four_digit_integers_diff_at_least_4000 :
  ∃ (A B : ℕ), (A = 8001) ∧ (B = 4001) ∧ (A.gcd B = 1) ∧
  (∀ (m n : ℕ), 0 < m → 0 < n → |A^m - B^n| ≥ 4000) :=
by
  use [8001, 4001]
  split
  { refl }
  split
  { refl }
  split
  { norm_num }
  intros m n hm hn
  -- This is where the proof steps would go
  sorry

end coprime_four_digit_integers_diff_at_least_4000_l159_159039


namespace factorize_polynomial_l159_159886

theorem factorize_polynomial (x y : ℝ) : 
  (x^2 - y^2 - 2 * x - 4 * y - 3) = (x + y + 1) * (x - y - 3) :=
  sorry

end factorize_polynomial_l159_159886


namespace arithmetic_sequence_property_l159_159930

noncomputable def a_n : ℕ → ℝ := sorry

def S (n : ℕ) := ∑ i in (finset.range n), a_n i

theorem arithmetic_sequence_property :
  (∀ n, a_n n = a_n 0 + n * (a_n 1 - a_n 0)) →
  (a_n 3 + a_n 8 = 13) →
  (S 7 = 35) →
  (a_n 7 = 8) :=
by
  intro ha hn hs
  sorry

end arithmetic_sequence_property_l159_159930


namespace min_x2_plus_y2_plus_4z2_l159_159056

theorem min_x2_plus_y2_plus_4z2 (x y z : ℝ) (h : x + y + z = 1) : x^2 + y^2 + 4z^2 ≥ 4 / 9 :=
sorry

end min_x2_plus_y2_plus_4z2_l159_159056


namespace quadratic_identity_l159_159983

theorem quadratic_identity (x : ℝ) : 
  (3*x + 1)^2 + 2*(3*x + 1)*(x - 3) + (x - 3)^2 = 16*x^2 - 16*x + 4 :=
by
  sorry

end quadratic_identity_l159_159983


namespace trigonometric_identity_l159_159906

-- Define the condition given in the problem
variable (α : ℝ)
def condition := (tan α) / (tan α - 1) = -1

-- The statement we need to prove
theorem trigonometric_identity (h : condition α) : (sin α - 3 * cos α) / (sin α + cos α) = -5 / 3 := 
by
  sorry

end trigonometric_identity_l159_159906


namespace min_value_expression_l159_159466

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 9^a = 3^(1 - b)) :
  ∃ (min_val : ℝ), min_val = (1 / (81 * a) + 2 / (81 * b) + a * b) ∧ min_val = 2 / 9 :=
by
  sorry

end min_value_expression_l159_159466


namespace logan_additional_income_l159_159223

-- Conditions
def logan_income : ℕ := 65000
def rent_expenses : ℕ := 20000
def grocery_expenses : ℕ := 5000
def gas_expenses : ℕ := 8000
def target_amount : ℕ := 42000

-- Question: How much more money must he make each year?
theorem logan_additional_income (additional_income : ℕ) :
  additional_income = 10000 :=
by
  let total_expenses := rent_expenses + grocery_expenses + gas_expenses
  let remaining_money := logan_income - total_expenses
  let additional_income_needed := target_amount - remaining_money
  have h : additional_income_needed = 10000 := sorry
  exact h

end logan_additional_income_l159_159223


namespace measure_angle_C_l159_159705

theorem measure_angle_C (A B C : ℝ) (h1 : A = 60) (h2 : B = 60) (h3 : C = 60 - 10) (sum_angles : A + B + C = 180) : C = 53.33 :=
by
  sorry

end measure_angle_C_l159_159705


namespace gcd_values_360_l159_159777

theorem gcd_values_360 : ∃ d : ℕ, d = 11 ∧ ∀ a b : ℕ, a * b = 360 → ∃ (g : ℕ), g = gcd a b ∧ finite {g | g = gcd a b ∧ a * b = 360} ∧ card {g | g = gcd a b ∧ a * b = 360} = 11 :=
sorry

end gcd_values_360_l159_159777


namespace smallest_n_condition_l159_159625

theorem smallest_n_condition (n : ℕ) (x : ℕ → ℝ) (h₀ : ∀ i, 0 ≤ x i)
(h₁ : ∑ i in finset.range n, x i = 1) 
(h₂ : ∑ i in finset.range n, (x i) ^ 2 ≤ 1/50 ) :
  n ≥ 50 :=
sorry

end smallest_n_condition_l159_159625


namespace ratio_perimeters_l159_159580

-- Defining isosceles triangle ABC
variables {A B C P E F : Type}
variables [Point A] [Point B] [Point C] [Point P] [Point E] [Point F]

-- Assume conditions
axiom is_isosceles_triangle: isosceles_triangle A B C
axiom base_is_one_fourth_perimeter: ∃ L l : ℝ, base_length B C = L / 4 ∧ leg_length A B = l ∧ leg_length A C = l ∧ perimeter A B C = L
axiom P_on_base: Point_on_segment P B C
axiom E_F_parallel_to_legs: parallel (line P E) (leg_line A B) ∧ parallel (line P F) (leg_line A C)

-- Define perimeter of quadrilateral AEPF
noncomputable def perimeter_AEPF : ℝ :=
  let l := leg_length A B in
  2 * l + 2 * l

-- Define perimeter of triangle ABC
noncomputable def perimeter_ABC : ℝ :=
  let L := perimeter A B C in
  L

-- Define the ratio and prove it
theorem ratio_perimeters : perimeter_AEPF / perimeter_ABC = 3 / 4 :=
sorry

end ratio_perimeters_l159_159580


namespace vovochka_no_carry_correct_cases_vovochka_minimum_difference_l159_159129

-- Part (a)
theorem vovochka_no_carry_correct_cases :
  let digit_pairs_without_carry := ∑ k in finset.range 10, (k + 1)
  let three_digit_pairs_without_carry := 9 * digit_pairs_without_carry
  let total_valid_combinations := 81 * three_digit_pairs_without_carry * three_digit_pairs_without_carry
  total_valid_combinations = 244620 :=
by {
  -- Definitions used in conditions
  let digit_pairs_without_carry := ∑ k in finset.range 10, (k + 1)
  let three_digit_pairs_without_carry := 9 * digit_pairs_without_carry
  let total_valid_combinations := 81 * three_digit_pairs_without_carry * three_digit_pairs_without_carry

  -- Assert the correct answer
  have correct_total_cases : total_valid_combinations = 244620 := 
    -- solution provided proof here
    sorry,

  exact correct_total_cases
}

-- Part (b)
theorem vovochka_minimum_difference :
  let smallest_difference := 1800
  ∃ a b c x y z : ℕ, (a, b, c) < (x, y, z) ∧ a + x ≥ 1 ∧ b + y ≥ 1 ∧ c + z ≥ 1 ∧
  correct_sum a b c x y z - vovochka_sum a b c x y z = smallest_difference :=
by {
  -- Definitions used in conditions
  let correct_sum := λ (a b c x y z : ℕ), 100 * (a + x) + 10 * (b + y) + (c + z)
  let vovochka_sum := λ (a b c x y z : ℕ), 100 * (a + x) + 10 * (b + y) + (c + z) % 10

  -- Assert the correct answer
  have smallest_diff_exists : 
    ∃ a b c x y z : ℕ, (a, b, c) < (x, y, z) ∧ a + x ≥ 1 ∧ b + y ≥ 1 ∧ c + z ≥ 1 ∧
    correct_sum a b c x y z - vovochka_sum a b c x y z = 1800 := 
    -- solution provided proof here
    sorry,

  exact smallest_diff_exists
}

end vovochka_no_carry_correct_cases_vovochka_minimum_difference_l159_159129


namespace vovochka_correct_sum_combinations_l159_159151

theorem vovochka_correct_sum_combinations : 
  let digit_pairs := finset.filter (λ p : ℕ × ℕ, (p.fst + p.snd) < 10) ((finset.range 10).product (finset.range 10))
  let no_carry_combinations := finset.card digit_pairs
  no_carry_combinations ^ 3 * 81 = 244620 := 
by
  sorry

end vovochka_correct_sum_combinations_l159_159151


namespace max_stopping_time_l159_159163

-- Define the initial conditions and assumptions
def initialFare : ℝ := 10
def baseDistance : ℝ := 4
def additionalFarePerKm : ℝ := 2
def conversionStopTimeToKm : ℝ := 5
def distanceAirportToHotel : ℝ := 15

-- Define the fare function
def fare (ξ : ℝ) : ℝ :=
  if ξ ≤ baseDistance then initialFare
  else initialFare + additionalFarePerKm * (ξ - baseDistance)

-- Assume given values
def ξ_given : ℝ := 15
def η_given_fare : ℕ := 38

-- Maximum stopping time calculation implementation
def stoppingTime (ξ : ℝ) (ξ_given : ℝ) : ℝ :=
  conversionStopTimeToKm * (ξ - ξ_given)

-- Main theorem
theorem max_stopping_time : 
  ∃ ξ_max : ℝ, ξ_max = (η_given_fare - 12) / 2 ∧ 
  stoppingTime ξ_max ξ_given = 15 :=
by
  sorry

end max_stopping_time_l159_159163


namespace completing_square_solution_l159_159756

theorem completing_square_solution (x : ℝ) :
  x^2 - 4*x - 3 = 0 ↔ (x - 2)^2 = 7 :=
sorry

end completing_square_solution_l159_159756


namespace vasya_birthday_is_thursday_l159_159733

def vasya_birthday_day_of_week (today_is_friday : Bool) (sunday_day_after_tomorrow : Bool) : String :=
  if today_is_friday && sunday_day_after_tomorrow then "Thursday" else "Unknown"

theorem vasya_birthday_is_thursday
  (today_is_friday : true)
  (sunday_day_after_tomorrow : true) : 
  vasya_birthday_day_of_week true true = "Thursday" := 
by
  -- assume today is Friday
  have h1 : today_is_friday = true := rfl
  -- assume Sunday is the day after tomorrow
  have h2 : sunday_day_after_tomorrow = true := rfl
  -- by our function definition, Vasya's birthday should be Thursday
  show vasya_birthday_day_of_week true true = "Thursday"
  sorry

end vasya_birthday_is_thursday_l159_159733


namespace vovochka_no_carry_correct_cases_vovochka_minimum_difference_l159_159127

-- Part (a)
theorem vovochka_no_carry_correct_cases :
  let digit_pairs_without_carry := ∑ k in finset.range 10, (k + 1)
  let three_digit_pairs_without_carry := 9 * digit_pairs_without_carry
  let total_valid_combinations := 81 * three_digit_pairs_without_carry * three_digit_pairs_without_carry
  total_valid_combinations = 244620 :=
by {
  -- Definitions used in conditions
  let digit_pairs_without_carry := ∑ k in finset.range 10, (k + 1)
  let three_digit_pairs_without_carry := 9 * digit_pairs_without_carry
  let total_valid_combinations := 81 * three_digit_pairs_without_carry * three_digit_pairs_without_carry

  -- Assert the correct answer
  have correct_total_cases : total_valid_combinations = 244620 := 
    -- solution provided proof here
    sorry,

  exact correct_total_cases
}

-- Part (b)
theorem vovochka_minimum_difference :
  let smallest_difference := 1800
  ∃ a b c x y z : ℕ, (a, b, c) < (x, y, z) ∧ a + x ≥ 1 ∧ b + y ≥ 1 ∧ c + z ≥ 1 ∧
  correct_sum a b c x y z - vovochka_sum a b c x y z = smallest_difference :=
by {
  -- Definitions used in conditions
  let correct_sum := λ (a b c x y z : ℕ), 100 * (a + x) + 10 * (b + y) + (c + z)
  let vovochka_sum := λ (a b c x y z : ℕ), 100 * (a + x) + 10 * (b + y) + (c + z) % 10

  -- Assert the correct answer
  have smallest_diff_exists : 
    ∃ a b c x y z : ℕ, (a, b, c) < (x, y, z) ∧ a + x ≥ 1 ∧ b + y ≥ 1 ∧ c + z ≥ 1 ∧
    correct_sum a b c x y z - vovochka_sum a b c x y z = 1800 := 
    -- solution provided proof here
    sorry,

  exact smallest_diff_exists
}

end vovochka_no_carry_correct_cases_vovochka_minimum_difference_l159_159127


namespace evaluate_expression_l159_159425

theorem evaluate_expression (x y : ℕ) (h1 : x = 2) (h2 : y = 3) : 3 * x^y + 4 * y^x = 60 :=
by
  rw [h1, h2]
  sorry

end evaluate_expression_l159_159425


namespace part1_part2_l159_159471

-- Part 1: proving the range of a
theorem part1 (a : ℝ) : (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → x^2 + a * x + 3 ≥ a) ↔ (a ∈ Icc (-7 : ℝ) 2) :=
by
  sorry

-- Part 2: proving the range of x
theorem part2 (a : ℝ) : (a ∈ Icc (4 : ℝ) 6) → ∀ x : ℝ, x^2 + a * x + 3 ≥ 0 ↔ 
  x ∈ (Iic (-3 - real.sqrt 6) ∪ Ici (-3 + real.sqrt 6)) :=
by
  sorry

end part1_part2_l159_159471


namespace min_chocolates_for_most_l159_159877

theorem min_chocolates_for_most (a b c d : ℕ) (h : a < b ∧ b < c ∧ c < d)
  (h_sum : a + b + c + d = 50) : d ≥ 14 := sorry

end min_chocolates_for_most_l159_159877


namespace vasya_birthday_day_l159_159739

/-- Define the days of the week as an inductive type --/
inductive Day
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday
deriving DecidableEq

open Day

/-- Function to get the day after a given day --/
def next_day : Day → Day
| Monday    => Tuesday
| Tuesday   => Wednesday
| Wednesday => Thursday
| Thursday  => Friday
| Friday    => Saturday
| Saturday  => Sunday
| Sunday    => Monday

/-- Function to get the day two days after a given day --/
def day_after_tomorrow (d : Day) : Day :=
  next_day (next_day d)

/-- Lean statement for the problem --/
theorem vasya_birthday_day (today : Day) (H1 : day_after_tomorrow today = Sunday) (H2 : next_day vasya_birthday = today) : vasya_birthday = Thursday := 
sorry

end vasya_birthday_day_l159_159739


namespace graduation_photo_arrangements_l159_159655

theorem graduation_photo_arrangements : 
  ∃ (n : ℕ), n = 192 ∧ 
    ∀ students : Fin 7 → Prop, 
    (students 0 = 'A' ∧ students 3 = 'A' ∧ students 6 = 'A') /\
    ((students 1 = 'B' ∧ students 2 = 'C') ∨ (students 5 = 'B' ∧ students 4 = 'C')) /
    ((students 2 = 'B' ∧ students 1 = 'C') ∨ (students 4 = 'B' ∧ students 5 = 'C')) /
    (students 0 = 'B' ∧ students 1 = 'C' ∧ students 3 = 'A') /
    (students 5 = 'B' ∧ students 6 = 'C' ∧ students 3 = 'A'),
    sorry

end graduation_photo_arrangements_l159_159655


namespace perpendicular_lines_l159_159966

noncomputable def l1 (a : ℝ) : ℝ → ℝ → Prop := 
  λ x y, 3 * x + 2 * a * y - 1 = 0
  
noncomputable def l2 (a : ℝ) : ℝ → ℝ → Prop := 
  λ x y, a * x - y + 2 = 0
  
def perpendicular (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x1 y1 x2 y2 : ℝ, l1 x1 y1 → l2 x2 y2 → (x2 - x1) * (y2 - y1) = (-1) * ((x2 - x1) * (y2 - y1))

theorem perpendicular_lines (a : ℝ) (h : perpendicular (l1 a) (l2 a)) : a = 0 :=
  sorry

end perpendicular_lines_l159_159966


namespace vovochkas_no_carry_pairs_eq_correct_pairs_vovochkas_smallest_difference_l159_159139

theorem vovochkas_no_carry_pairs_eq_correct_pairs :
  let valid_digit_pairs := Nat.range 10.map (λ n, 10 - n)
  let sum_valid_digit_pairs := valid_digit_pairs.sum
  sum_valid_digit_pairs = 55 →
  let no_carry_combinations := 81 * 55 * 55
  no_carry_combinations = 244620 :=
by
  let valid_digit_pairs := List.map (λ n, 10 - n) (List.range 10)
  have h_valid_digit_sum : valid_digit_pairs.sum = 55 := by sorry
  let no_carry_combinations := 81 * 55 * 55
  have h_no_carry : no_carry_combinations = 244620 := by sorry
  exact h_no_carry

theorem vovochkas_smallest_difference :
  let incorrect_cases := [1800]
  incorrect_cases.minimum = 1800 :=
by
  let differences := [900, 90, 990]
  have h_min_diff : List.minimum differences = some 90 := by sorry
  let incorrect_cases := List.map (λ diff, 20 * diff) differences
  have h_min_incorrect : incorrect_cases.minimum = some 1800 := by sorry
  exact h_min_incorrect

end vovochkas_no_carry_pairs_eq_correct_pairs_vovochkas_smallest_difference_l159_159139


namespace area_of_figure_l159_159850

theorem area_of_figure : 
  (∫ t in 0..π/6, -96 * 3 * (sin t) ^ 4 * (cos t) ^ 2) * 2 = 6 * π - 9 * sqrt 3 := sorry

end area_of_figure_l159_159850


namespace gcd_lcm_product_l159_159762

theorem gcd_lcm_product (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 360) :
  ∃ (s : Finset ℕ), s.card = 12 ∧ ∀ d ∈ s, d = Nat.gcd a b :=
sorry

end gcd_lcm_product_l159_159762


namespace sams_seashells_l159_159243

theorem sams_seashells (mary_seashells : ℕ) (total_seashells : ℕ) (h_mary : mary_seashells = 47) (h_total : total_seashells = 65) : (total_seashells - mary_seashells) = 18 :=
by
  simp [h_mary, h_total]
  sorry

end sams_seashells_l159_159243


namespace smallest_integer_for_divisibility_l159_159750

def sum_two_digit_pairs (n : Nat) : Nat :=
  let digits := n.digits.reverse
  digits.chunk(2).map Nat.ofDigits.sum

def alternating_sum_three_digit_groups (n : Nat) : Int :=
  let digits := n.digits.reverse
  digits.chunk(3).map Nat.ofDigits
    |>.enum.map (fun (i, x) => if i % 2 = 0 then x else -x).sum

theorem smallest_integer_for_divisibility :
  ∃ (k : Nat), k = 22 ∧
    (let sum_pairs := sum_two_digit_pairs (987654 + k) in
     sum_pairs % 19 = 0) ∧
    (let alt_sum_groups := alternating_sum_three_digit_groups (987654 + k) in
     alt_sum_groups % 8 = 0) := 
sorry

end smallest_integer_for_divisibility_l159_159750


namespace load_capacity_l159_159005

theorem load_capacity : ∀ T H : ℕ, T = 3 → H = 9 → (L = 35 * T^3 / H^3) → L = 35 / 27 :=
begin
  intros T H hT hH hL,
  rw [hT, hH] at hL,
  simp at hL,
  exact hL,
end

end load_capacity_l159_159005


namespace total_tin_in_new_alloy_l159_159807

-- Definition of the conditions
def alloy_A_weight : ℝ := 60
def alloy_B_weight : ℝ := 100

-- Definition of the ratios
def lead_to_tin_ratio_A : ℝ × ℝ := (3, 2)
def tin_to_copper_ratio_B : ℝ × ℝ := (1, 4)

-- The sum of parts in the ratios
def total_parts_A : ℝ := lead_to_tin_ratio_A.1 + lead_to_tin_ratio_A.2
def total_parts_B : ℝ := tin_to_copper_ratio_B.1 + tin_to_copper_ratio_B.2

-- Tin weights
def tin_weight_A : ℝ := (lead_to_tin_ratio_A.2 / total_parts_A) * alloy_A_weight
def tin_weight_B : ℝ := (tin_to_copper_ratio_B.1 / total_parts_B) * alloy_B_weight

theorem total_tin_in_new_alloy : tin_weight_A + tin_weight_B = 44 := 
by sorry

end total_tin_in_new_alloy_l159_159807


namespace number_of_proper_subsets_of_M_l159_159535

def satisfies_log_condition (x : ℤ) : Prop := 
  real.log_inv (1/3) x ≥ -1

def M : set ℤ := { x ∈ ℤ | satisfies_log_condition x }

theorem number_of_proper_subsets_of_M : 
  M = {1, 2, 3} → (2 ^ M.card - 1) = 7 := 
by
  intros h
  simp [h]
  sorry

end number_of_proper_subsets_of_M_l159_159535


namespace number_of_complex_numbers_satisfying_conditions_l159_159018

   noncomputable def number_of_solutions (z : ℂ) : ℕ := 
     if (|z| = 1 ∧ |(z / conj z) + (conj z / z)| = 2) then 4 else 0

   theorem number_of_complex_numbers_satisfying_conditions : 
     ∃ z : ℂ, |z| = 1 ∧ |(z / conj z) + (conj z / z)| = 2 → number_of_solutions z = 4 :=
   by {
     sorry
   }
   
end number_of_complex_numbers_satisfying_conditions_l159_159018


namespace unattainable_y_l159_159042

theorem unattainable_y (x : ℚ) (h : x ≠ -4 / 3) : (∀ y : ℚ, (y = (2 - x) / (3 * x + 4)) → y ≠ -1 / 3) :=
by
  intro y
  intro h1
  have h2 : 3 * (-1 / 3) + 1 = 0 := by norm_num
  have h3 : 3 * x + 4 ≠ 0 := by
    intro h4
    have h5 : x = -4 / 3 := by linarith
    exact h h5
  rw h2 at h1
  exact h3 h1

end unattainable_y_l159_159042


namespace value_of_m_l159_159046

theorem value_of_m (m : ℤ) (h1 : abs m = 2) (h2 : m - 2 ≠ 0) : m = -2 :=
by
  sorry

end value_of_m_l159_159046


namespace range_of_ϕ_l159_159953

noncomputable def f (x : ℝ) (ϕ : ℝ) : ℝ := 2 * sin (2 * x + ϕ) + 1

theorem range_of_ϕ (ϕ : ℝ) :
  (∀ x : ℝ, -π / 12 < x ∧ x < π / 3 → f x ϕ > 1) →
  abs ϕ ≤ π / 2 →
  π / 6 ≤ ϕ ∧ ϕ ≤ π / 3 :=
sorry

end range_of_ϕ_l159_159953


namespace function_properties_l159_159940

theorem function_properties (f : ℝ → ℝ) (hf : ∀ x y, f(x) + f(y) = 2 * f((x + y) / 2) * f((x - y) / 2)) :
    (f(1) = 1/2 → f(2) = -1/2) ∧ 
    (f(1) = 0 → (Σ(n : ℕ), n ∈ (range (2023 - 11)/2 + 1))) = 0 → (Σ(n : ℕ) , n ∈ (range (2023/2 - 5/2 + 1) f n) = 0) :=
by
  sorry

end function_properties_l159_159940


namespace minute_hand_coincides_hour_hand_11_times_l159_159316

noncomputable def number_of_coincidences : ℕ := 11

theorem minute_hand_coincides_hour_hand_11_times :
  ∀ (t : ℝ), (0 < t ∧ t < 12) → ∃(n : ℕ), (1 ≤ n ∧ n ≤ 11) ∧ t = (n * 1 + n * (5 / 11)) :=
sorry

end minute_hand_coincides_hour_hand_11_times_l159_159316


namespace arithmetic_seq_sum_l159_159487

theorem arithmetic_seq_sum (a_n : ℕ → ℝ) (h_arith_seq : ∃ d, ∀ n, a_n (n + 1) = a_n n + d)
    (h_sum : a_n 5 + a_n 8 = 24) : a_n 6 + a_n 7 = 24 := by
  sorry

end arithmetic_seq_sum_l159_159487


namespace vasya_birthday_l159_159708

def DayOfWeek := 
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

open DayOfWeek

def day_after (d : DayOfWeek) : DayOfWeek :=
match d with
| Sunday => Monday
| Monday => Tuesday
| Tuesday => Wednesday
| Wednesday => Thursday
| Thursday => Friday
| Friday => Saturday
| Saturday => Sunday

def day_before (d : DayOfWeek) : DayOfWeek :=
match d with
| Sunday => Saturday
| Monday => Sunday
| Tuesday => Monday
| Wednesday => Tuesday
| Thursday => Wednesday
| Friday => Thursday
| Saturday => Friday

theorem vasya_birthday (day_said : DayOfWeek) 
    (h1 : day_after (day_after day_said) = Sunday) 
    (h2 : day_said = day_after VasyaBirthday) 
    : VasyaBirthday = Thursday :=
sorry

end vasya_birthday_l159_159708


namespace agatha_bike_budget_l159_159833

def total_initial : ℕ := 60
def cost_frame : ℕ := 15
def cost_front_wheel : ℕ := 25
def total_spent : ℕ := cost_frame + cost_front_wheel
def total_left : ℕ := total_initial - total_spent

theorem agatha_bike_budget : total_left = 20 := by
  sorry

end agatha_bike_budget_l159_159833


namespace sides_of_right_triangle_l159_159669

theorem sides_of_right_triangle (r : ℝ) (a b c : ℝ) 
  (h_area : (2 / (2 / r)) * 2 = 2 * r) 
  (h_right : a^2 + b^2 = c^2) :
  (a = r ∧ b = (4 / 3) * r ∧ c = (5 / 3) * r) ∨
  (b = r ∧ a = (4 / 3) * r ∧ c = (5 / 3) * r) :=
sorry

end sides_of_right_triangle_l159_159669


namespace unattainable_y_l159_159043

theorem unattainable_y (x : ℚ) (h : x ≠ -4 / 3) : (∀ y : ℚ, (y = (2 - x) / (3 * x + 4)) → y ≠ -1 / 3) :=
by
  intro y
  intro h1
  have h2 : 3 * (-1 / 3) + 1 = 0 := by norm_num
  have h3 : 3 * x + 4 ≠ 0 := by
    intro h4
    have h5 : x = -4 / 3 := by linarith
    exact h h5
  rw h2 at h1
  exact h3 h1

end unattainable_y_l159_159043


namespace num_unique_m_values_l159_159622

theorem num_unique_m_values : 
  ∃ (s : Finset Int), 
  (∀ (x1 x2 : Int), x1 * x2 = 36 → x1 + x2 ∈ s) ∧ 
  s.card = 10 := 
sorry

end num_unique_m_values_l159_159622


namespace courtyard_width_l159_159368

def width_of_courtyard (w : ℝ) : Prop :=
  28 * 100 * 100 * w = 13788 * 22 * 12

theorem courtyard_width :
  ∃ w : ℝ, width_of_courtyard w ∧ abs (w - 13.012) < 0.001 :=
by
  sorry

end courtyard_width_l159_159368


namespace constant_term_binomial_expansion_l159_159119

theorem constant_term_binomial_expansion :
  ∃ n r : ℕ, 2 ^ n = 64 ∧ 6 - 2 * r = 0 ∧ nat.choose 6 3 = 20 :=
begin
  use [6, 3],
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num }
end

end constant_term_binomial_expansion_l159_159119


namespace vovochka_no_carry_correct_cases_vovochka_minimum_difference_l159_159130

-- Part (a)
theorem vovochka_no_carry_correct_cases :
  let digit_pairs_without_carry := ∑ k in finset.range 10, (k + 1)
  let three_digit_pairs_without_carry := 9 * digit_pairs_without_carry
  let total_valid_combinations := 81 * three_digit_pairs_without_carry * three_digit_pairs_without_carry
  total_valid_combinations = 244620 :=
by {
  -- Definitions used in conditions
  let digit_pairs_without_carry := ∑ k in finset.range 10, (k + 1)
  let three_digit_pairs_without_carry := 9 * digit_pairs_without_carry
  let total_valid_combinations := 81 * three_digit_pairs_without_carry * three_digit_pairs_without_carry

  -- Assert the correct answer
  have correct_total_cases : total_valid_combinations = 244620 := 
    -- solution provided proof here
    sorry,

  exact correct_total_cases
}

-- Part (b)
theorem vovochka_minimum_difference :
  let smallest_difference := 1800
  ∃ a b c x y z : ℕ, (a, b, c) < (x, y, z) ∧ a + x ≥ 1 ∧ b + y ≥ 1 ∧ c + z ≥ 1 ∧
  correct_sum a b c x y z - vovochka_sum a b c x y z = smallest_difference :=
by {
  -- Definitions used in conditions
  let correct_sum := λ (a b c x y z : ℕ), 100 * (a + x) + 10 * (b + y) + (c + z)
  let vovochka_sum := λ (a b c x y z : ℕ), 100 * (a + x) + 10 * (b + y) + (c + z) % 10

  -- Assert the correct answer
  have smallest_diff_exists : 
    ∃ a b c x y z : ℕ, (a, b, c) < (x, y, z) ∧ a + x ≥ 1 ∧ b + y ≥ 1 ∧ c + z ≥ 1 ∧
    correct_sum a b c x y z - vovochka_sum a b c x y z = 1800 := 
    -- solution provided proof here
    sorry,

  exact smallest_diff_exists
}

end vovochka_no_carry_correct_cases_vovochka_minimum_difference_l159_159130


namespace smallest_angle_in_acute_triangle_l159_159182

theorem smallest_angle_in_acute_triangle (a b c d : ℝ) (triangle1 : a = 110 ∧ a > 90) 
(triangle2 : b + c + d = 180 ∧ b < 90 ∧ c < 90 ∧ d < 90 ∧ b ∈ {75,65,15} ∧ c ∈ {75,65,15} ∧ d ∈ {75,65,15}):
(a + b + c = 180 ∨ a + b + d = 180 ∨ a + c + d = 180) → (min b (min c d)) = 15 :=
by sorry

end smallest_angle_in_acute_triangle_l159_159182


namespace option_D_is_unit_vector_l159_159097

variables {V : Type*} [inner_product_space ℝ V]

-- Given conditions
variables (a b : V)
variables (h1 : ∥a∥ = 1) (h2 : ∥b∥ = 1)
variable (h3 : inner a b = 1/2)

-- Define the vector option D
def option_D : V := a - b

-- The statement to be proven
theorem option_D_is_unit_vector : ∥option_D a b∥ = 1 :=
by {
  -- Given conditions
  exact sorry,
}

end option_D_is_unit_vector_l159_159097


namespace smaller_number_is_7_l159_159976

theorem smaller_number_is_7 (x y : ℕ) (h1 : x * y = 56) (h2 : x + y = 15) (h3 : x ≤ y) (h4 : x ∣ 28) : x = 7 :=
  sorry

end smaller_number_is_7_l159_159976


namespace product_of_binomial_coefficients_less_than_zero_l159_159177

-- Define the definite integral as a condition
def definite_integral : ℝ := ∫ x in -Real.pi/2..0, Real.sin x

-- Define the binomial coefficients for the given expansion
def binomial_coefficients (a : ℝ) (n : ℕ) : list ℤ :=
  list.map (λ k, (if k % 2 = 0 then 1 else -1) * Real.to_int (Nat.choose n k)) (list.range (n + 1))

-- Calculate the number of ways to choose a positive and a negative coefficient
noncomputable def num_ways_pos_neg_pairs : ℕ := 5 * 6

-- Calculate the total number of ways to choose two coefficients from 11
noncomputable def total_ways_choose_two : ℕ := Nat.choose 11 2

-- Calculate the probability
noncomputable def probability_product_negative : ℚ := 
  num_ways_pos_neg_pairs / total_ways_choose_two

-- Prove the desired probability
theorem product_of_binomial_coefficients_less_than_zero :
  definite_integral = -1 → probability_product_negative = 6/11 :=
begin
  intro h,
  rw definite_integral at h,
  exact sorry,
end

end product_of_binomial_coefficients_less_than_zero_l159_159177


namespace largest_m_for_binomial_sum_l159_159852

theorem largest_m_for_binomial_sum :
  ∃ m : ℕ, (m ≤ 11) ∧ (nat.choose 10 4 + nat.choose 10 5 = nat.choose 11 m) ∧ (∀ k, (nat.choose 10 4 + nat.choose 10 5 = nat.choose 11 k) → k ≤ m) ∧ m = 6 :=
by
  sorry

end largest_m_for_binomial_sum_l159_159852


namespace find_sum_l159_159209

noncomputable def p (x : ℝ) : ℝ := x^3 - 6*x^2 + 11*x - 6 + 10*x

theorem find_sum :
  (p 0) + (p 5) = 68 := by
begin
  -- The proof would go here, but it's marked with 'sorry' as per instructions.
  sorry
end

end find_sum_l159_159209


namespace collinearity_condition_not_sufficient_nor_necessary_l159_159515

theorem collinearity_condition_not_sufficient_nor_necessary (x : ℝ) : 
  let a := (1, 2 - x)
  let b := (2 + x, 3)
  (|a| = sqrt 2) ↔ collinear a b :=
sorry

end collinearity_condition_not_sufficient_nor_necessary_l159_159515


namespace cloth_cost_theorem_l159_159385

def cloth_cost_calc (meters_sold : ℕ) (selling_price : ℕ) (loss_per_meter : ℕ) (total_loss : ℕ) (total_cost_price : ℕ) (cost_price_per_meter : ℕ) :=
  meters_sold = 450 ∧
  selling_price = 18000 ∧
  loss_per_meter = 5 ∧
  total_loss = loss_per_meter * meters_sold ∧
  total_cost_price = selling_price + total_loss ∧
  cost_price_per_meter = total_cost_price / meters_sold

theorem cloth_cost_theorem :
  ∃ (cost_price_per_meter : ℕ),
    cloth_cost_calc 450 18000 5 2250 20250 45
sorry

end cloth_cost_theorem_l159_159385


namespace vasya_birthday_day_l159_159741

/-- Define the days of the week as an inductive type --/
inductive Day
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday
deriving DecidableEq

open Day

/-- Function to get the day after a given day --/
def next_day : Day → Day
| Monday    => Tuesday
| Tuesday   => Wednesday
| Wednesday => Thursday
| Thursday  => Friday
| Friday    => Saturday
| Saturday  => Sunday
| Sunday    => Monday

/-- Function to get the day two days after a given day --/
def day_after_tomorrow (d : Day) : Day :=
  next_day (next_day d)

/-- Lean statement for the problem --/
theorem vasya_birthday_day (today : Day) (H1 : day_after_tomorrow today = Sunday) (H2 : next_day vasya_birthday = today) : vasya_birthday = Thursday := 
sorry

end vasya_birthday_day_l159_159741


namespace solve_for_y_l159_159542

variables (x y : ℤ)

theorem solve_for_y (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 :=
by
  hint sorry

end solve_for_y_l159_159542


namespace roots_form_parallelogram_l159_159421

noncomputable def polynomial (b : ℝ) : Polynomial ℂ :=
  Polynomial.Coe.monomial 4 1 - Polynomial.Coe.monomial 3 (8 : ℂ)
  + Polynomial.Coe.monomial 2 (13 * b : ℂ)
  - Polynomial.Coe.monomial 1 (3 * (2 * b^2 + 5 * b - 3) : ℂ)
  - Polynomial.Coe.monomial 0 1

def isParallelogram (z : ℂ → Prop) : Prop :=
  ∃ z1 z2 : ℂ, z z1 ∧ z z2 ∧ z (-z1) ∧ z (-z2)

theorem roots_form_parallelogram (b : ℝ) :
  (∃ p : Polynomial ℂ, p = polynomial b ∧
    isParallelogram (λ x, Polynomial.Roots p x)) →
  b = 3 / 2 :=
by
  sorry

end roots_form_parallelogram_l159_159421


namespace sum_of_possible_m_l159_159630

theorem sum_of_possible_m {a b c m : ℂ} 
  (h_distinct: a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_eq: (a + 1) / (2 - b) = m ∧ (b + 1) / (2 - c) = m ∧ (c + 1) / (2 - a) = m) :
  ∑ m', m' ∈ {m | (m^2 - m + 1) = 0} = 1 :=
by
  sorry

end sum_of_possible_m_l159_159630


namespace standard_deviation_sqrt2_l159_159477

theorem standard_deviation_sqrt2 (a : ℝ) (s : fin 5 → ℝ) (h : s = ![a, 1, 2, 3, 4]) (h_avg : ((a + 1 + 2 + 3 + 4) / 5) = 2) :
  (∑ i, (s i - 2) ^ 2 / 5).sqrt = Real.sqrt 2 :=
by
  sorry

end standard_deviation_sqrt2_l159_159477


namespace parallel_condition_perpendicular_condition_l159_159094

-- Definitions of given vectors
def vec_a : ℝ × ℝ := (3, 2)
def vec_b : ℝ × ℝ := (-1, 2)
def vec_c : ℝ × ℝ := (4, 1)

-- Functions for vector operations
def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def vec_scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- (1) Find the coordinates of (vec_a + λ * vec_c)
def coords_vec_a_add_lambda_vec_c (λ : ℝ) : ℝ × ℝ :=
  vec_add vec_a (vec_scalar_mul λ vec_c)

-- (2) Find the coordinates of (2 * vec_b - vec_a)
def coords_2_vec_b_sub_vec_a : ℝ × ℝ :=
  vec_add (vec_scalar_mul 2 vec_b) (vec_scalar_mul (-1) vec_a)

-- (3) If (vec_a + λ * vec_c) is parallel to (2 * vec_b - vec_a), find λ
theorem parallel_condition (λ : ℝ) : 
  (coords_vec_a_add_lambda_vec_c λ).1 / (coords_2_vec_b_sub_vec_a).1 = 
  (coords_vec_a_add_lambda_vec_c λ).2 / (coords_2_vec_b_sub_vec_a).2 → 
  λ = -16/13 :=
sorry

-- (4) If (vec_a + λ * vec_c) is perpendicular to (2 * vec_b - vec_a), find λ
theorem perpendicular_condition (λ : ℝ) :
  (coords_vec_a_add_lambda_vec_c λ).1 * (coords_2_vec_b_sub_vec_a).1 + 
  (coords_vec_a_add_lambda_vec_c λ).2 * (coords_2_vec_b_sub_vec_a).2 = 0 → 
  λ = 11/18 :=
sorry

end parallel_condition_perpendicular_condition_l159_159094


namespace solve_phi_l159_159330

noncomputable def phi_solution (x : ℝ) (lambda : ℝ) (tildeC : ℝ) : ℝ :=
if lambda ≠ -2 then 
  (2 * lambda / (2 + lambda)) * (Real.sin (Real.log x)) + 2 * x 
else 
  tildeC * (Real.sin (Real.log x))

theorem solve_phi (lambda : ℝ) (tildeC : ℝ) :
  (∀ x, phi_solution x lambda tildeC - lambda * (\int t in 0..1, Real.sin (Real.log x) * (phi_solution t lambda tildeC)) = 2 * x) :=
by
  sorry

end solve_phi_l159_159330


namespace ratio_p_r_l159_159618

     variables (p q r s : ℚ)

     -- Given conditions
     def ratio_p_q := p / q = 3 / 5
     def ratio_r_s := r / s = 5 / 4
     def ratio_s_q := s / q = 1 / 3

     -- Statement to be proved
     theorem ratio_p_r 
       (h1 : ratio_p_q p q)
       (h2 : ratio_r_s r s) 
       (h3 : ratio_s_q s q) : 
       p / r = 36 / 25 :=
     sorry
     
end ratio_p_r_l159_159618


namespace distinctMonicQuadraticPolynomials_count_l159_159436

noncomputable def countDistinctMonicQuadraticPolynomials : Nat := 5111

theorem distinctMonicQuadraticPolynomials_count :
  let polynomials := { (a, b) : ℕ × ℕ |
                        a + b ≤ 141 ∧ 3^a + 3^b ≤ 3^141 ∧ a >= b }
  ∃! (n : Nat), n = countDistinctMonicQuadraticPolynomials ∧
    (polynomials.card = n) := 
by
  sorry

end distinctMonicQuadraticPolynomials_count_l159_159436


namespace max_value_f_l159_159334

-- Definition of the function f(x)
def f (x ϕ : ℝ) : ℝ := sin (x + 2 * ϕ) - 2 * sin ϕ * cos (x + ϕ)

-- Theorem stating the maximum value of f(x) is 1
theorem max_value_f : ∀ (ϕ : ℝ), ∃ x : ℝ, f x ϕ = 1 := by
  sorry

end max_value_f_l159_159334


namespace vovochka_no_carry_correct_cases_vovochka_minimum_difference_l159_159126

-- Part (a)
theorem vovochka_no_carry_correct_cases :
  let digit_pairs_without_carry := ∑ k in finset.range 10, (k + 1)
  let three_digit_pairs_without_carry := 9 * digit_pairs_without_carry
  let total_valid_combinations := 81 * three_digit_pairs_without_carry * three_digit_pairs_without_carry
  total_valid_combinations = 244620 :=
by {
  -- Definitions used in conditions
  let digit_pairs_without_carry := ∑ k in finset.range 10, (k + 1)
  let three_digit_pairs_without_carry := 9 * digit_pairs_without_carry
  let total_valid_combinations := 81 * three_digit_pairs_without_carry * three_digit_pairs_without_carry

  -- Assert the correct answer
  have correct_total_cases : total_valid_combinations = 244620 := 
    -- solution provided proof here
    sorry,

  exact correct_total_cases
}

-- Part (b)
theorem vovochka_minimum_difference :
  let smallest_difference := 1800
  ∃ a b c x y z : ℕ, (a, b, c) < (x, y, z) ∧ a + x ≥ 1 ∧ b + y ≥ 1 ∧ c + z ≥ 1 ∧
  correct_sum a b c x y z - vovochka_sum a b c x y z = smallest_difference :=
by {
  -- Definitions used in conditions
  let correct_sum := λ (a b c x y z : ℕ), 100 * (a + x) + 10 * (b + y) + (c + z)
  let vovochka_sum := λ (a b c x y z : ℕ), 100 * (a + x) + 10 * (b + y) + (c + z) % 10

  -- Assert the correct answer
  have smallest_diff_exists : 
    ∃ a b c x y z : ℕ, (a, b, c) < (x, y, z) ∧ a + x ≥ 1 ∧ b + y ≥ 1 ∧ c + z ≥ 1 ∧
    correct_sum a b c x y z - vovochka_sum a b c x y z = 1800 := 
    -- solution provided proof here
    sorry,

  exact smallest_diff_exists
}

end vovochka_no_carry_correct_cases_vovochka_minimum_difference_l159_159126


namespace find_f2_l159_159269

theorem find_f2 (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f x + 3 * f (1 - x) = 2 * x ^ 2) :
  f 2 = -1 / 4 :=
by
  sorry

end find_f2_l159_159269


namespace problem_statement_l159_159804

variable (S : Set Point) (k n : ℕ)

-- Definition for "Any two points in S are not collinear"
def not_collinear (S : Set Point) : Prop :=
  ∀ P Q R ∈ S, P ≠ Q ∧ Q ≠ R ∧ P ≠ R → ¬Collinear P Q R

-- Definition for "For any point P in S, there are at least k points in S equidistant from P"
def equidistant_points (S : Set Point) (k : ℕ) : Prop :=
  ∀ P ∈ S, ∃ (ps : Set Point), ps ⊆ S ∧ P ∉ ps ∧ card ps ≥ k ∧ ∀ Q ∈ ps, dist P Q = d

-- Main theorem statement
theorem problem_statement (h1 : not_collinear S) (h2 : equidistant_points S k) (h3 : card S = n) :
  k ≤ 1 / 2 + √(2 * n) :=
sorry

end problem_statement_l159_159804


namespace prime_p_range_l159_159981

open Classical

variable {p : ℤ} (hp_prime : Prime p)

def is_integer_root (a b c : ℤ) := 
  ∃ x y : ℤ, x * y = c ∧ x + y = -b

theorem prime_p_range (hp_roots : is_integer_root 1 p (-500 * p)) : 1 < p ∧ p ≤ 10 :=
by
  sorry

end prime_p_range_l159_159981


namespace num_gcd_values_l159_159773

-- Define the condition for the product of gcd and lcm
def is_valid_pair (a b : ℕ) : Prop :=
  gcd a b * Nat.lcm a b = 360

-- Define the main theorem statement
theorem num_gcd_values : 
  ∃ (n : ℕ), 
    (∀ a b, is_valid_pair a b → ∃ m (hm: m ≤ 360), gcd a b = m) ∧ 
    n = 12 := sorry

end num_gcd_values_l159_159773


namespace vasya_birthday_was_thursday_l159_159716

variable (today : String)
variable (tomorrow : String)
variable (day_after_tomorrow : String)
variable (birthday : String)

-- Conditions given in the problem
axiom birthday_not_sunday : birthday ≠ "Sunday"
axiom sunday_day_after_tomorrow : day_after_tomorrow = "Sunday"

-- From the conditions, we need to prove that Vasya's birthday was on Thursday.
theorem vasya_birthday_was_thursday : birthday = "Thursday" :=
by
  -- Fill in the proof here
  sorry

end vasya_birthday_was_thursday_l159_159716


namespace Yuna_place_l159_159006

theorem Yuna_place (Eunji_place : ℕ) (distance : ℕ) (Yuna_place : ℕ) 
  (h1 : Eunji_place = 100) 
  (h2 : distance = 11) 
  (h3 : Yuna_place = Eunji_place + distance) : 
  Yuna_place = 111 := 
sorry

end Yuna_place_l159_159006


namespace find_n_l159_159924

-- Define the sequence using the given conditions
def seq : ℕ → ℕ
| 1     := 2
| (n+1) := seq n + 3

-- State the main theorem to be proved
theorem find_n (a_n_eq : seq n = 2009) : n = 670 :=
sorry

end find_n_l159_159924


namespace total_number_of_soldiers_is_98_l159_159665

-- Definitions for the problem conditions
def initial_back_lookers (n : ℕ) := n - 2
def initial_eye_lookers (n : ℕ) := initial_back_lookers n / 6
def final_eye_lookers (n : ℕ) := (initial_back_lookers n) / 7
def edge_changes_by_two := 2

-- Main statement encapsulating the problem and conditions
theorem total_number_of_soldiers_is_98 :
  ∃ (n : ℕ), 
  (n - 2) / 6 - ((n - 2) / 7) = edge_changes_by_two / 2 ∧
  n = 98 := 
begin
  -- This is where the proof would go.
  sorry
end

end total_number_of_soldiers_is_98_l159_159665


namespace expand_and_simplify_l159_159428

-- Define the polynomials
def poly1 := (x : ℝ) ^ 2 - 3 * x + 3
def poly2 := (x : ℝ) ^ 2 + 3 * x + 3
def result := (x : ℝ) ^ 4 - 3 * (x : ℝ) ^ 2 + 9

-- Statement to prove
theorem expand_and_simplify (x : ℝ) : (poly1 * poly2) = result :=
by
  sorry

end expand_and_simplify_l159_159428


namespace angle_MKD_right_angle_l159_159589

structure Parallelogram (A B C D : Type) :=
  (sides_parallel : ∀ {X Y Z W : Type}, X = A ∧ Y = B ∧ Z = C ∧ W = D → X = W ∧ Y = Z)

variable {A B C D H M K : Type}

-- Given conditions
variable [Parallelogram A B C D]
variable (BH_perpendicular_to_AD : ∀ {X Y : Type}, X = B ∧ Y = H → BH_perpendicular_to_AD) 
variable (point_M_equidistant : ∀ {X Y Z : Type}, X = M ∧ Y = C ∧ Z = D → point_M_equidistant) 
variable (K_midpoint : ∀ {X Y : Type}, X = K ∧ Y = AB → K_midpoint) 

-- The theorem to be proved
theorem angle_MKD_right_angle : ∀ {A B C D H M K : Type}
  [Parallelogram A B C D] 
  (BH_perpendicular_to_AD : ∀ {X Y : Type}, X = B ∧ Y = H → BH_perpendicular_to_AD) 
  (point_M_equidistant : ∀ {X Y Z : Type}, X = M ∧ Y = C ∧ Z = D → point_M_equidistant) 
  (K_midpoint : ∀ {X Y : Type}, X = K ∧ Y = AB → K_midpoint) ,
   ∠M K D = 90°
:= 
by 
  sorry

end angle_MKD_right_angle_l159_159589


namespace average_multiplied_is_correct_l159_159891

theorem average_multiplied_is_correct (x : ℝ) : 
  let terms := [0, 2*x, 4*x, 8*x, 16*x] in
  let multiplied_terms := terms.map (λ t, 3 * t) in
  let sum_of_multiplied := multiplied_terms.foldr (· + ·) 0 in
  let average := sum_of_multiplied / multiplied_terms.length in
  average = 18 * x :=
by
  sorry

end average_multiplied_is_correct_l159_159891


namespace quadratic_unique_solution_l159_159689

theorem quadratic_unique_solution (a c : ℕ) (h1 : a + c = 29) (h2 : a < c)
  (h3 : (20^2 - 4 * a * c) = 0) : (a, c) = (4, 25) :=
by
  sorry

end quadratic_unique_solution_l159_159689


namespace minimal_questions_to_identify_numbers_l159_159469

theorem minimal_questions_to_identify_numbers (a : Fin 2005 → ℕ) (h_distinct : Function.Injective a) :
  ∃ m, (∀ q : ℕ, (q < m → ¬ (∀ (questions : Fin q → Fin 2005 × Fin 2005 × Fin 2005), ∃ a' : Fin 2005 → ℕ, a' = a  ∧ Function.Injective a')))
  ∧ (m = 1003) :=
begin
  -- sorry
end

end minimal_questions_to_identify_numbers_l159_159469


namespace range_f_l159_159027

variable {α : Type} [LinearOrder α] [Field α]

noncomputable def f (x : α) : α := (3 * x + 8) / (x - 4)

theorem range_f : set.range f = { y : α | y ≠ 3 } :=
by
  sorry  -- Proof is omitted, according to the instructions

end range_f_l159_159027


namespace vasya_birthday_is_thursday_l159_159734

def vasya_birthday_day_of_week (today_is_friday : Bool) (sunday_day_after_tomorrow : Bool) : String :=
  if today_is_friday && sunday_day_after_tomorrow then "Thursday" else "Unknown"

theorem vasya_birthday_is_thursday
  (today_is_friday : true)
  (sunday_day_after_tomorrow : true) : 
  vasya_birthday_day_of_week true true = "Thursday" := 
by
  -- assume today is Friday
  have h1 : today_is_friday = true := rfl
  -- assume Sunday is the day after tomorrow
  have h2 : sunday_day_after_tomorrow = true := rfl
  -- by our function definition, Vasya's birthday should be Thursday
  show vasya_birthday_day_of_week true true = "Thursday"
  sorry

end vasya_birthday_is_thursday_l159_159734


namespace hyperbola_eccentricity_l159_159084

theorem hyperbola_eccentricity (b : ℝ) (h : ∀ (x : ℝ) (y : ℝ), x^2 - (y^2 / b^2) = 1)
  (distance_focus_asymptote : dist (sqrt (1 + b^2), 0) (line_y_eq_bx : ∀ x, y = b * x) = 2) :
  eccentricity b = sqrt 5 :=
sorry

end hyperbola_eccentricity_l159_159084


namespace find_a2_plus_a9_l159_159176

namespace ArithmeticSequence

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a(n + 1) = a(n) + d

def sum_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range(n), a i

theorem find_a2_plus_a9 (a : ℕ → ℝ) (h_seq : is_arithmetic_sequence a)
  (h_sum10 : sum_arithmetic_sequence a 10 = 120) :
  a 1 + a 8 = 24 :=
sorry

end ArithmeticSequence

end find_a2_plus_a9_l159_159176


namespace max_min_difference_adjacent_circular_l159_159501

theorem max_min_difference_adjacent_circular (xs : List ℕ) 
  (h : xs = [10, 6, 13, 4, 18]) :
  ∃ ys : List ℕ, (ys.perm xs ∧ (minimum_difference ys ys.head (ys.last ys)) = 9) := sorry

  def minimum_difference : List ℕ → ℕ → ℕ → ℕ := sorry

end max_min_difference_adjacent_circular_l159_159501


namespace f_f_of_neg_pi_div_3_l159_159497

def f (x : ℝ) : ℝ :=
if x >= 0 then Real.sqrt x else Real.cos x

theorem f_f_of_neg_pi_div_3 :
  f (f (- (Real.pi / 3))) = Real.sqrt 2 / 2 :=
by
  sorry

end f_f_of_neg_pi_div_3_l159_159497


namespace rectangle_division_l159_159379

theorem rectangle_division:
  ∀ (m n : ℕ), 
  m = 19 → n = 65 →
  (m * n + (m + n - Nat.gcd m n)) = 1318 := 
by
  intros m n hm hn
  rw [hm, hn]
  have h_: 19 + 65 - Nat.gcd 19 65 = 83 := by
    rw [Nat.gcd_eq_right (Nat.dvd_trans (Nat.gcd_dvd_left 19 65) (Nat.gcd_eq_right 65 0).symm)]
    norm_num
  calc
    19 * 65 + 83 = 1235 + 83 : by norm_num
    ... = 1318 : by norm_num

end rectangle_division_l159_159379


namespace maximize_area_l159_159827

/-- Define the perimeter constraint for the flower bed. --/
def flower_bed_perimeter : ℝ := 12

/-- Define the shapes under consideration. --/
inductive Shape
| equilateral_triangle
| square
| regular_hexagon
| circle

/-- Define the area function for each shape given the perimeter constraint. --/
def area (shape : Shape) : ℝ :=
  match shape with
  | Shape.equilateral_triangle => 4 * Real.sqrt 3
  | Shape.square => 9
  | Shape.regular_hexagon => 6 * Real.sqrt 3
  | Shape.circle => 36 / Real.pi

/-- Prove that the circle maximizes the area under the given perimeter constraint. --/
theorem maximize_area : ∀ shape, area shape ≤ area Shape.circle := by
  sorry

end maximize_area_l159_159827


namespace vovochka_add_correct_pairs_vovochka_min_diff_incorrect_l159_159136

-- Vovochka's addition method: adding two digits without carrying over
def vovochka_add (a b : ℕ) : ℕ := (a % 10 + b % 10) + ((a / 10 % 10 + b / 10 % 10) * 10) + ((a / 100 + b / 100) * 100)

-- Part (a): number of pairs producing correct result with Vovochka’s method
def correct_vovochka_pairs_count : ℕ := 244620

-- Part (b): smallest possible difference when Vovochka’s method is incorrect
def min_diff_vovochka_method : ℕ := 1800

-- Proving the number of correct cases equals 244620
theorem vovochka_add_correct_pairs :
  let count := ∑ (a b : ℕ) in finset.Icc 100 999, if vovochka_add a b = a + b then 1 else 0
  in count = correct_vovochka_pairs_count := sorry

-- Proving the smallest possible difference when Vovochka’s method is incorrect
theorem vovochka_min_diff_incorrect :
  let min_diff := min (∑ (a b : ℕ) in finset.Icc 100 999, if vovochka_add a b ≠ a + b then nat.abs (vovochka_add a b - (a + b)) else ⊤)
  in min_diff = min_diff_vovochka_method := sorry

end vovochka_add_correct_pairs_vovochka_min_diff_incorrect_l159_159136


namespace cos_C_in_right_triangle_l159_159120

theorem cos_C_in_right_triangle (x : ℝ) (h0 : 0 < x) (h1 : tan C = 1 / 2) :
  cos C = 2 * Real.sqrt 5 / 5 :=
by
  -- Use the given conditions h0 and h1
  sorry

end cos_C_in_right_triangle_l159_159120


namespace ratio_of_scores_l159_159162

theorem ratio_of_scores (Lizzie Nathalie Aimee teammates : ℕ) (combinedLN : ℕ)
    (team_total : ℕ) (m : ℕ) :
    Lizzie = 4 →
    Nathalie = Lizzie + 3 →
    combinedLN = Lizzie + Nathalie →
    Aimee = m * combinedLN →
    teammates = 17 →
    team_total = Lizzie + Nathalie + Aimee + teammates →
    team_total = 50 →
    (Aimee / combinedLN) = 2 :=
by 
    sorry

end ratio_of_scores_l159_159162


namespace distribute_apples_l159_159392

theorem distribute_apples :
  let a, b, c : ℕ in
  (a >= 3) ∧ (b >= 3) ∧ (c >= 3) ∧ (a + b + c = 26) →
  fintype.card { (a', b', c' : ℕ) // a' + b' + c' = 17 } = 171 :=
by
  sorry

end distribute_apples_l159_159392


namespace range_of_f_l159_159033

noncomputable def f (x : ℝ) : ℝ := (3 * x + 8) / (x - 4)

theorem range_of_f : set.range f = set.univ \ {3} :=
by
  sorry

end range_of_f_l159_159033


namespace vovochka_add_correct_pairs_vovochka_min_diff_incorrect_l159_159131

-- Vovochka's addition method: adding two digits without carrying over
def vovochka_add (a b : ℕ) : ℕ := (a % 10 + b % 10) + ((a / 10 % 10 + b / 10 % 10) * 10) + ((a / 100 + b / 100) * 100)

-- Part (a): number of pairs producing correct result with Vovochka’s method
def correct_vovochka_pairs_count : ℕ := 244620

-- Part (b): smallest possible difference when Vovochka’s method is incorrect
def min_diff_vovochka_method : ℕ := 1800

-- Proving the number of correct cases equals 244620
theorem vovochka_add_correct_pairs :
  let count := ∑ (a b : ℕ) in finset.Icc 100 999, if vovochka_add a b = a + b then 1 else 0
  in count = correct_vovochka_pairs_count := sorry

-- Proving the smallest possible difference when Vovochka’s method is incorrect
theorem vovochka_min_diff_incorrect :
  let min_diff := min (∑ (a b : ℕ) in finset.Icc 100 999, if vovochka_add a b ≠ a + b then nat.abs (vovochka_add a b - (a + b)) else ⊤)
  in min_diff = min_diff_vovochka_method := sorry

end vovochka_add_correct_pairs_vovochka_min_diff_incorrect_l159_159131


namespace sum_consecutive_triangular_sum_triangular_2020_l159_159843

-- Define triangular numbers
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

-- The theorem to be proved
theorem sum_consecutive_triangular (n : ℕ) : triangular n + triangular (n + 1) = (n + 1)^2 :=
by 
  sorry

-- Applying the theorem for the specific case of n = 2020
theorem sum_triangular_2020 : triangular 2020 + triangular 2021 = 2021^2 :=
by 
  exact sum_consecutive_triangular 2020

end sum_consecutive_triangular_sum_triangular_2020_l159_159843


namespace constant_term_sum_l159_159563

theorem constant_term_sum (n : ℕ) (h : 2^n = 64) : 
  ∃ r : ℕ, let T_r := Nat.choose n r * 3^(n - r) * (3 * r) in
  T_r = 135 := 
by
  sorry

end constant_term_sum_l159_159563


namespace marble_probability_correct_l159_159346

noncomputable def marble_probability : ℚ :=
  let total_ways := (Nat.choose 20 4 : ℚ)
  let ways_two_red := (Nat.choose 12 2 : ℚ)
  let ways_two_blue := (Nat.choose 8 2 : ℚ)
  (ways_two_red * ways_two_blue) / total_ways

theorem marble_probability_correct : marble_probability = 56 / 147 :=
by
  -- Note: the proof is omitted as per instructions
  sorry

end marble_probability_correct_l159_159346


namespace area_of_rectangular_field_l159_159380

-- Definitions from conditions
def L : ℕ := 20
def total_fencing : ℕ := 32

-- Additional variables inferred from the conditions
def W : ℕ := (total_fencing - L) / 2

-- The theorem statement
theorem area_of_rectangular_field : L * W = 120 :=
by
  -- Definitions and substitutions are included in the theorem proof
  sorry

end area_of_rectangular_field_l159_159380


namespace problem_a_lt_zero_b_lt_neg_one_l159_159465

theorem problem_a_lt_zero_b_lt_neg_one (a b : ℝ) (ha : a < 0) (hb : b < -1) : 
  ab > a ∧ a > ab^2 := 
by
  sorry

end problem_a_lt_zero_b_lt_neg_one_l159_159465


namespace solve_for_x_l159_159659

theorem solve_for_x (x : ℤ) (h : 15 * 2 = x - 3 + 5) : x = 28 :=
sorry

end solve_for_x_l159_159659


namespace part1_part2_l159_159954

-- Define the function f(x)
def f (x : ℝ) : ℝ := x + 1 / x

-- Part 1: Prove that f(x) is monotonically decreasing on (0,1) and monotonically increasing on (1,+∞).
theorem part1 (x : ℝ) : (0 < x ∧ x < 1 → ∀ x1 x2 : ℝ, (0 < x1 ∧ x1 < 1 ∧ 0 < x2 ∧ x2 < 1 ∧ x1 < x2) → f x2 < f x1) ∧ (1 < x → ∀ x1 x2 : ℝ, (1 < x1 ∧ 1 < x2 ∧ x1 < x2) → f x1 < f x2)
  := by
  sorry

-- Part 2: Given x ∈ [1/2, 2], prove the range of f(1 / f(x)) is [5/2, 29/10].
theorem part2 (x : ℝ) (hx : 1 / 2 ≤ x ∧ x ≤ 2) : (5 / 2 ≤ f (1 / f x) ∧ f (1 / f x) ≤ 29 / 10)
  := by
  sorry

end part1_part2_l159_159954


namespace power_function_through_point_l159_159506
-- Import the necessary library

-- Define the problem
theorem power_function_through_point :
  ∃ (a : ℝ → ℝ), ∀ x, a x = x ^ (1 / 2) ∧ a (1/2) = (sqrt 2) / 2 :=
begin
  -- Proof is omitted with sorry
  sorry
end

end power_function_through_point_l159_159506


namespace daps_equivalent_to_dips_l159_159105

-- Define the units as types
def dap : Type := ℕ
def dop : Type := ℕ
def dip : Type := ℕ

-- Define the conditions
def daps_to_dops (d : ℕ) : ℝ := (5 * d) / 4
def dops_to_dips (d : ℕ) : ℝ := (3 * d) / 9

-- The theorem to prove that 54 dips are equivalent to 22.5 daps
theorem daps_equivalent_to_dips {d : ℕ} (h₁ : daps_to_dops d = d) (h₂ : dops_to_dips d = d) : 
  54 * ((5 : ℝ) / 12) = 22.5 :=
by
  sorry

end daps_equivalent_to_dips_l159_159105


namespace integer_solutions_l159_159430

theorem integer_solutions :
  { (x, y) : ℤ × ℤ | x^2 = 1 + 4 * y^3 * (y + 2) } = {(1, 0), (1, -2), (-1, 0), (-1, -2)} :=
by
  sorry

end integer_solutions_l159_159430


namespace cos_pi_six_minus_alpha_l159_159934

noncomputable def alpha : ℝ := sorry -- since we need a specific alpha value to satisfy the condition

theorem cos_pi_six_minus_alpha (α : ℝ) (h : cos (α - π / 3) + cos α = 4 * sqrt 3 / 5) : 
  cos (π / 6 - α) = 4 / 5 :=
by 
  sorry

end cos_pi_six_minus_alpha_l159_159934


namespace angle_between_AK_and_BC_l159_159929

variable {A B C D E K : Point}
variable [PlaneGeometry]

-- Given conditions
variables (h_triangle : acute_triangle A B C)
variables (h_circle : ∃ ω : Circle, diameter ω B C ∧ intersects ω A B (finset.insert D ∅) ∧ intersects ω A C (finset.insert E ∅))
variables (h_tangent_D : tangent_to ω D K)
variables (h_tangent_E : tangent_to ω E K)

-- Goal
theorem angle_between_AK_and_BC (h_triangle : acute_triangle A B C)
    (h_circle : ∃ ω : Circle, diameter ω B C ∧ intersects ω A B (finset.insert D ∅) ∧ intersects ω A C (finset.insert E ∅))
    (h_tangent_D : tangent_to ω D K)
    (h_tangent_E : tangent_to ω E K) :
    angle (Line.mk A K) (Line.mk B C) = 90 :=
sorry

end angle_between_AK_and_BC_l159_159929


namespace men_in_first_group_l159_159662

theorem men_in_first_group
  (M : ℕ) -- number of men in the first group
  (h1 : M * 8 * 24 = 12 * 8 * 16) : M = 8 :=
sorry

end men_in_first_group_l159_159662


namespace dacid_average_marks_is_75_l159_159789

/-- Defining the marks obtained in each subject as constants -/
def english_marks : ℕ := 76
def mathematics_marks : ℕ := 65
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85

/-- Total marks calculation -/
def total_marks : ℕ :=
  english_marks + mathematics_marks + physics_marks + chemistry_marks + biology_marks

/-- Number of subjects -/
def number_of_subjects : ℕ := 5

/-- Average marks calculation -/
def average_marks : ℕ :=
  total_marks / number_of_subjects

/-- Theorem proving that Dacid's average marks is 75 -/
theorem dacid_average_marks_is_75 : average_marks = 75 :=
  sorry

end dacid_average_marks_is_75_l159_159789


namespace elliptic_eccentricity_range_l159_159494

section
variables {a b : Real} (P : Real × Real)
variable (e : Real)

-- Conditions
def ellipse (x y : Real) (a b : Real) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def circle (x y : Real) (b : Real) : Prop := x^2 + y^2 = b^2
def point_on_ellipse (P : Real × Real) (a b : Real) : Prop := ellipse P.1 P.2 a b

-- Eccentricity conditions
def eccentricity_range (e : Real) : Prop := e ∈ Set.Ioo (Real.sqrt 3 / 2) 1

-- Main theorem
theorem elliptic_eccentricity_range:
  a > b → b > 0 →
  (∃ P : Real × Real, point_on_ellipse P a b ∧ angle (f (PA P)) (f (PB P)) = 2 * Real.pi / 3) →
  eccentricity_range e :=
sorry
end

end elliptic_eccentricity_range_l159_159494


namespace prime_numbers_between_l159_159285

theorem prime_numbers_between 
  (x : ℕ) 
  (h1 : 2 <= x)
  (h2 : ∀ p q : ℕ, nat.prime p ∧ nat.prime q ∧ (2 ≤ p ∧ p < x) ∧ (2 ≤ q ∧ q < x) ∧ p ≠ q → ∃ r : ℕ, nat.prime r ∧ p < r ∧ r < x → r = 13 ∨ r = 17 ∨ r = 11) 
  (h3 : ∀ p : ℕ, nat.prime p ∧ (2 ≤ p ∧ p < 19) ∧ (p ≠ 11 ∧ p ≠ 13 ∧ p ≠ 17)) :
  x = 18 :=
by
  sorry

end prime_numbers_between_l159_159285


namespace range_of_f_l159_159032

noncomputable def f (x : ℝ) : ℝ := (3 * x + 8) / (x - 4)

theorem range_of_f : set.range f = set.univ \ {3} :=
by
  sorry

end range_of_f_l159_159032


namespace volume_ratio_l159_159278

theorem volume_ratio (r : ℝ) (h_sphere : r > 0) :
  let V_sphere := (4/3) * π * r^3,
      V_hemisphere := (1/2) * (4/3) * π * (2*r)^3,
      V_cylinder := π * (2*r)^2 * (2*r),
      V_combined := V_hemisphere + V_cylinder
  in V_sphere / V_combined = 1 / 10 :=
by
  sorry

end volume_ratio_l159_159278


namespace max_eggs_l159_159829

theorem max_eggs (x : ℕ) 
  (h1 : x < 200) 
  (h2 : x % 3 = 2) 
  (h3 : x % 4 = 3) 
  (h4 : x % 5 = 4) : 
  x = 179 := 
by
  sorry

end max_eggs_l159_159829


namespace geometric_seq_a_n_plus_3_general_term_a_n_sum_S_n_l159_159090

noncomputable def a : ℕ → ℤ
| 0     := -2
| (n+1) := 3 * a n + 6

theorem geometric_seq_a_n_plus_3 :
  ∀ n, (a n + 3 = 3 ^ n) :=
by sorry

theorem general_term_a_n :
  ∀ n, (a n = 3 ^ (n - 1) - 3) :=
by sorry

theorem sum_S_n :
  ∀ n, (∑ i in finset.range n, a (i + 1) = (1 / 2) * 3 ^ n - 3 * n - (1 / 2)) :=
by sorry

end geometric_seq_a_n_plus_3_general_term_a_n_sum_S_n_l159_159090


namespace Vasya_birthday_on_Thursday_l159_159718

-- Define the days of the week enumeration
inductive DayOfWeek
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open DayOfWeek

-- Define the conditions
def Sunday_is_the_day_after_tomorrow (today : DayOfWeek) : Prop :=
  match today with
  | Friday => true
  | _      => false

def Vasya_birthday_not_on_Sunday (birthday : DayOfWeek) : Prop :=
  birthday ≠ Sunday

def Today_is_next_day_after_birthday (today birthday : DayOfWeek) : Prop :=
  match (today, birthday) with
  | (Friday, Thursday) => true
  | _                  => false

-- The main theorem to prove Vasya's birthday was on Thursday
theorem Vasya_birthday_on_Thursday (birthday today : DayOfWeek) 
  (H1: Sunday_is_the_day_after_tomorrow today) 
  (H2: Vasya_birthday_not_on_Sunday birthday) 
  (H3: Today_is_next_day_after_birthday today birthday) : 
  birthday = Thursday :=
by
  sorry

end Vasya_birthday_on_Thursday_l159_159718


namespace planes_intersect_in_one_line_l159_159274

theorem planes_intersect_in_one_line 
  (S A B C A1 B1 C1 : Point)
  (plane_SA plane_SB plane_SC plane_SAA1 plane_SBB1 plane_SCC1 plane_BC plane_AC plane_AB : Plane)
  (trihedral_angle : ¬ RightAngle (angle plane_SA plane_SB) ∧ 
                     ¬ RightAngle (angle plane_SB plane_SC) ∧ 
                     ¬ RightAngle (angle plane_SC plane_SA))
  (perpendicular_planes :
    (plane_perpendicular plane_SA plane_BC) ∧
    (plane_perpendicular plane_SB plane_AC) ∧
    (plane_perpendicular plane_SC plane_AB) ∧
    (plane_perpendicular plane_SAA1 plane_BC) ∧
    (plane_perpendicular plane_SBB1 plane_AC) ∧
    (plane_perpendicular plane_SCC1 plane_AB)) :
    ∃ (l : Line), intersect plane_SAA1 l ∧ intersect plane_SBB1 l ∧ intersect plane_SCC1 l :=
sorry

end planes_intersect_in_one_line_l159_159274


namespace find_number_l159_159754

theorem find_number (x : ℝ) (h : x / 5 = 70 + x / 6) : x = 2100 :=
sorry

end find_number_l159_159754


namespace range_of_f_l159_159022

noncomputable def f (x : ℝ) : ℝ := (3 * x + 8) / (x - 4)

theorem range_of_f : Set.range f = {y : ℝ | y ≠ 3} :=
sorry

end range_of_f_l159_159022


namespace sin_tan_identity_l159_159795

theorem sin_tan_identity :
  sin (10 * (Real.pi / 180)) + (Real.sqrt 3 / 4) * tan (10 * (Real.pi / 180)) = 1 / 4 :=
by
  sorry

end sin_tan_identity_l159_159795


namespace vovochka_correct_sum_combinations_l159_159154

theorem vovochka_correct_sum_combinations : 
  let digit_pairs := finset.filter (λ p : ℕ × ℕ, (p.fst + p.snd) < 10) ((finset.range 10).product (finset.range 10))
  let no_carry_combinations := finset.card digit_pairs
  no_carry_combinations ^ 3 * 81 = 244620 := 
by
  sorry

end vovochka_correct_sum_combinations_l159_159154


namespace integer_solution_sum_eq_zero_l159_159447

theorem integer_solution_sum_eq_zero : 
  (∑ x in (Finset.filter (λ x : ℤ, x^4 - 13 * x^2 + 36 = 0) (Finset.Icc -3 3)), x) = 0 := by
sorry

end integer_solution_sum_eq_zero_l159_159447


namespace range_of_f_l159_159024

noncomputable def f (x : ℝ) : ℝ := (3 * x + 8) / (x - 4)

theorem range_of_f : Set.range f = {y : ℝ | y ≠ 3} :=
sorry

end range_of_f_l159_159024


namespace gcd_lcm_product_l159_159766

theorem gcd_lcm_product (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 360) :
  ∃ (s : Finset ℕ), s.card = 12 ∧ ∀ d ∈ s, d = Nat.gcd a b :=
sorry

end gcd_lcm_product_l159_159766


namespace sqrt_expression_eq_l159_159411

theorem sqrt_expression_eq : 
  (Real.sqrt 18 / Real.sqrt 6 - Real.sqrt 12 + Real.sqrt 48 * Real.sqrt (1/3)) = -Real.sqrt 3 + 4 := 
by
  sorry

end sqrt_expression_eq_l159_159411


namespace compute_value_l159_159416

theorem compute_value : (142 + 29 + 26 + 14) * 2 = 422 := 
by 
  sorry

end compute_value_l159_159416


namespace triangle_angle_C_l159_159799

theorem triangle_angle_C (A B C : Point) (O : Point)
  (hO : is_orthocenter O A B C)
  (h_circle : ∃ I1 I2 : Point, lies_on_circle I1 I2 A B O)
  (h_interior : lies_in_interior I1 A C ∧ lies_in_interior I2 B C) :
  60 < angle A B C ∧ angle A B C < 90 :=
by
  sorry

end triangle_angle_C_l159_159799


namespace sum_of_solutions_eq_zero_l159_159444

theorem sum_of_solutions_eq_zero : 
  let eqn := λ x : ℤ, x^4 - 13 * x^2 + 36 in
  ∑ x in { x : ℤ | eqn x = 0 }.toFinset = 0 :=
by sorry

end sum_of_solutions_eq_zero_l159_159444


namespace count_N_lt_500_with_solution_l159_159971

theorem count_N_lt_500_with_solution:
  let count := (finset.range 500).filter (λ N, ∃ x : ℝ, 0 < x ∧ x ^ (⌈x⌉.to_nat) = N) in
  count.card = 199 :=
by
  sorry

end count_N_lt_500_with_solution_l159_159971


namespace arithmetic_sequence_sum_l159_159855

theorem arithmetic_sequence_sum :
  let a1 := 1
  let d := 2
  let n := 10
  let an := 19
  let sum := 100
  let general_term := fun (n : ℕ) => a1 + (n - 1) * d
  (general_term n = an) → (n = 10) → (sum = (n * (a1 + an)) / 2) →
  sum = 100 :=
by
  sorry

end arithmetic_sequence_sum_l159_159855


namespace lily_guesses_at_least_two_wrong_l159_159636

noncomputable def lily_quiz_probability : ℚ :=
  let p_wrong := 3 / 4 in
  let probability := 
    (binomial 6 2) * (p_wrong ^ 2) * ((1 - p_wrong) ^ 4) +
    (binomial 6 3) * (p_wrong ^ 3) * ((1 - p_wrong) ^ 3) +
    (binomial 6 4) * (p_wrong ^ 4) * ((1 - p_wrong) ^ 2) +
    (binomial 6 5) * (p_wrong ^ 5) * ((1 - p_wrong) ^ 1) +
    (binomial 6 6) * (p_wrong ^ 6) * ((1 - p_wrong) ^ 0) in
  probability

theorem lily_guesses_at_least_two_wrong :
  lily_quiz_probability = 4077 / 4096 :=
by sorry

end lily_guesses_at_least_two_wrong_l159_159636


namespace sum_of_common_divisors_l159_159898

theorem sum_of_common_divisors (a b c d e : ℕ) 
  (ha : a = 60) (hb : b = 90) (hc : c = 150) (hd : d = 180) (he : e = 210) :
  ∑ n in { n | n ∣ a ∧ n ∣ b ∧ n ∣ c ∧ n ∣ d ∧ n ∣ e }, n = 72 :=
by 
  sorry

end sum_of_common_divisors_l159_159898


namespace combination_eq_l159_159534

theorem combination_eq (n : ℕ) (h : nat.choose n 3 = nat.choose (n-1) 3 + nat.choose (n-1) 4) : n = 7 :=
sorry

end combination_eq_l159_159534


namespace distance_from_center_of_base_to_lateral_face_l159_159559

theorem distance_from_center_of_base_to_lateral_face (a : ℝ) 
  (h1 : ∀ (faces1 faces2 lateral_angle : ℝ) (angle_eq : lateral_angle = 90), lateral_angle = 90) 
  (h2 : ∀ (base_triangle : ℕ → ℝ) (length : base_triangle 3 = a), base_triangle 3 = a)
  : ∃ (d : ℝ), d = a * Real.sqrt 2 / 2 :=
begin
  sorry
end

end distance_from_center_of_base_to_lateral_face_l159_159559


namespace complex_quadrant_proof_l159_159493

noncomputable def z : ℂ := (2 - complex.i) ^ 2

theorem complex_quadrant_proof :
  z.re > 0 ∧ z.im < 0 :=
by
  sorry

end complex_quadrant_proof_l159_159493


namespace evaluate_infinite_radical_l159_159881

theorem evaluate_infinite_radical :
  ∃ x : ℝ, x = sqrt (18 + 2 * x) ∧ x = 6 :=
by
  -- A proof goes here to show that x = 6 satisfies the equation x = sqrt (18 + 2x)
  sorry

end evaluate_infinite_radical_l159_159881


namespace vasya_birthday_l159_159728

/--
Vasya said the day after his birthday: "It's a pity that my birthday 
is not on a Sunday this year, because more guests would have come! 
But Sunday will be the day after tomorrow..."
On what day of the week was Vasya's birthday?
-/
theorem vasya_birthday (today : string)
  (h1 : today = "Friday")
  (h2 : ∀ day : string, day ≠ "Sunday" → Vasya's_birthday day) :
  Vasya's_birthday "Thursday" := by
  sorry

end vasya_birthday_l159_159728


namespace minimum_value_expression_l159_159015

noncomputable def minimum_expression (n : ℕ) (x : Fin n → ℝ) : ℝ :=
  let sin_sum := (Finset.univ.sum (λ i, Real.sin (x i))) in
  let cos_sum := (Finset.univ.sum (λ i, Real.cos (x i))) in
  (2 * sin_sum + cos_sum) * (sin_sum - 2 * cos_sum)

theorem minimum_value_expression (n : ℕ) (x : Fin n → ℝ) :
  minimum_expression n x ≥ -5 * n^2 / 2 := by
  sorry

end minimum_value_expression_l159_159015


namespace smallest_x_to_make_1152_x_perfect_cube_l159_159303

theorem smallest_x_to_make_1152_x_perfect_cube :
  ∃ x : ℕ, (1152 * x = 24^3 ∧ (∀ y, (1152 * y = 24^3) → y ≥ x)) :=
by
  let x := 12
  use x
  split
  {
    calc
      1152 * x = 1152 * 12 : by rfl
      ... = 2^7 * 3^2 * 2^2 * 3^1 : by norm_num
      ... = 2^9 * 3^3 : by ring_nf
      ... = (2^3)^3 * (3^1)^3 : by rw [pow_mul, pow_mul]
      ... = (8 * 3)^3 : by ring_nf
      ... = 24^3 : by norm_num,
  }
  {
    intro y
    intro hyp
    calc
      y = 12 : sorry -- proving y >= x when hyp holds involves more detailed steps
  }

end smallest_x_to_make_1152_x_perfect_cube_l159_159303


namespace quadratic_completion_l159_159688

theorem quadratic_completion 
  (h : ∀ x : ℝ, x^2 + 800 * x + 2400 = (x + 400)^2 - 157600) :
  let b := 400 in
  let c := -157600 in
  c / b = -394 :=
by
  intros
  exact h x
  use b
  use c
  exact sorry

end quadratic_completion_l159_159688


namespace vovochka_correct_sum_combinations_l159_159150

theorem vovochka_correct_sum_combinations : 
  let digit_pairs := finset.filter (λ p : ℕ × ℕ, (p.fst + p.snd) < 10) ((finset.range 10).product (finset.range 10))
  let no_carry_combinations := finset.card digit_pairs
  no_carry_combinations ^ 3 * 81 = 244620 := 
by
  sorry

end vovochka_correct_sum_combinations_l159_159150


namespace prove_bn_lt_3_l159_159616

noncomputable def b : ℕ → ℝ
| 0       := 1
| (n+1) := b n * (n+1)^(2^(-n))

theorem prove_bn_lt_3 : ∀ n : ℕ, n > 0 → b n < 3 :=
by
  sorry

end prove_bn_lt_3_l159_159616


namespace XY_length_constant_l159_159326

theorem XY_length_constant
  (O A B R X Y : Point)
  (circle : Circle O)
  (angle_AOB : angle A O B = 60)
  (R_on_arc : R ∈ circle.arc A B)
  (X_on_OA : X ∈ line_segment O A) 
  (Y_on_OB : Y ∈ line_segment O B)
  (angle_RXO : angle R X O = 65)
  (angle_RYO : angle R Y O = 115) :
  ∃ k, ∀ R, length (line_segment X Y) = k :=
by
  sorry

end XY_length_constant_l159_159326


namespace find_s_l159_159629

variable (n r s c d : ℝ)

noncomputable def poly1_roots (c d : ℝ) : Prop :=
  c * d = 3

noncomputable def poly2_roots (c d : ℝ) : Prop :=
  (c + 2 / d) * (d + 2 / c) = s

theorem find_s (h1 : poly1_roots c d) (h2 : poly2_roots c d) : s = 25 / 3 := by
  sorry

end find_s_l159_159629


namespace sum_elements_l159_159072

variable {a : ℤ} (hpos : a > 0)

def A : Set ℤ := {x | abs (x - a) < a + 1 / 2}
def B : Set ℤ := {x | abs x < 2 * a}

theorem sum_elements (hpos : a > 0) :
  (∑ x in (A ∪ B).toFinset, x) = 2 * a := by
  sorry

end sum_elements_l159_159072


namespace appliance_savings_l159_159283

theorem appliance_savings :
  let in_store_price := 99.99
  let tv_payment := 29.98
  let shipping_handling := 9.98
  let total_tv_price := 3 * tv_payment + shipping_handling
  let savings := in_store_price - total_tv_price
  savings * 100 = 7 := 
by {
  let in_store_price := 99.99
  let tv_payment := 29.98
  let shipping_handling := 9.98
  let total_tv_price := 3 * tv_payment + shipping_handling
  let savings := in_store_price - total_tv_price
  let converted_savings := savings * 100
  have hs : converted_savings = 7, sorry
  exact hs
}

end appliance_savings_l159_159283


namespace range_of_f_l159_159025

noncomputable def f (x : ℝ) : ℝ := (3 * x + 8) / (x - 4)

theorem range_of_f : Set.range f = {y : ℝ | y ≠ 3} :=
sorry

end range_of_f_l159_159025


namespace correct_statements_eq_3_l159_159180

def class (k : ℤ) : Set ℤ := { n | ∃ m : ℤ, n = 5 * m + k }

def statement_1 : Prop := 2013 ∈ class 3

def statement_2 : Prop := -2 ∈ class 2

def statement_3 : Prop := ( ⋃ k in {0, 1, 2, 3, 4}, class k ) = Set.univ

def statement_4 : Prop := ∀ a b : ℤ, (a - b) % 5 = 0 ↔ (a ∈ class 0 ∧ b ∈ class 0)

def correct_statements : ℕ :=
  [statement_1, statement_2, statement_3, statement_4].count true

theorem correct_statements_eq_3 : correct_statements = 3 := by
  sorry

end correct_statements_eq_3_l159_159180


namespace range_f_l159_159028

variable {α : Type} [LinearOrder α] [Field α]

noncomputable def f (x : α) : α := (3 * x + 8) / (x - 4)

theorem range_f : set.range f = { y : α | y ≠ 3 } :=
by
  sorry  -- Proof is omitted, according to the instructions

end range_f_l159_159028


namespace simplify_expression_l159_159248

theorem simplify_expression (y : ℝ) : 2 - (2 - (2 - (2 - (2 - y)))) = 4 - y :=
by
  sorry

end simplify_expression_l159_159248


namespace probability_two_red_two_blue_correct_l159_159341

noncomputable def num_ways_to_choose : ℕ → ℕ → ℕ :=
  λ n k, Nat.choose n k

noncomputable def probability_two_red_two_blue : ℚ :=
  let total_ways := num_ways_to_choose 20 4
  let ways_red := num_ways_to_choose 12 2
  let ways_blue := num_ways_to_choose 8 2
  (ways_red * ways_blue) / total_ways

theorem probability_two_red_two_blue_correct :
  probability_two_red_two_blue = 616 / 1615 :=
by
  sorry

end probability_two_red_two_blue_correct_l159_159341


namespace monotonic_intervals_a_leq_0_monotonic_intervals_a_gt_0_ln_sum_gt_2_l159_159958

noncomputable def f (x : ℝ) := 2 * Real.log x

def g (a x : ℝ) := (1 / 2) * a * x^2 + (2 * a - 1) * x

def h (a x : ℝ) := f x - g a x

theorem monotonic_intervals_a_leq_0 (a : ℝ) (h₀ : a ≤ 0) :
  ∀ x > 0, Monotone (λ x, h a x) :=
sorry

theorem monotonic_intervals_a_gt_0 (a : ℝ) (h₀ : a > 0) :
  ∀ x > 0, 
    (MonotoneOn (λ x, h a x) (set.Ioo 0 (1 / a))) ∧
    (AntitoneOn (λ x, h a x) (set.Ioi (1 / a))) :=
sorry
  
theorem ln_sum_gt_2 {a x₁ x₂ : ℝ} (hx : x₁ ≠ x₂)
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0)
  (h₁ : f x₁ - a * x₁ = 0)
  (h₂ : f x₂ - a * x₂ = 0) :
  Real.log x₁ + Real.log x₂ > 2 :=
sorry

end monotonic_intervals_a_leq_0_monotonic_intervals_a_gt_0_ln_sum_gt_2_l159_159958


namespace min_value_f_on_0_3_l159_159684

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem min_value_f_on_0_3 : ∃ (x ∈ set.Icc 0 3), ∀ y ∈ set.Icc 0 3, f x ≤ f y :=
by
  have min_val : ∃ x : ℝ, x ∈ set.Icc 0 3 ∧ f x = -15 := sorry,
  exact min_val

end min_value_f_on_0_3_l159_159684


namespace daps_from_dips_l159_159106

section DapsDopsDips

variable (Daps Dops Dips : Type)
variable [has_scalar ℚ Daps] [has_scalar ℚ Dops] [has_scalar ℚ Dips]

-- Condition 1: 5 daps = 4 dops
def condition1 (d : Daps) (p : Dops) : Prop := (5 : ℚ) • d = (4 : ℚ) • p

-- Condition 2: 3 dops = 9 dips
def condition2 (p : Dops) (i : Dips) : Prop := (3 : ℚ) • p = (9 : ℚ) • i

-- Theorem: How many daps are equivalent to 54 dips?
theorem daps_from_dips (d : Daps) (p : Dops) (i : Dips)
  (h1 : condition1 d p) (h2 : condition2 p i) : (54 : ℚ) • i = (22.5 : ℚ) • d :=
by 
  sorry

end DapsDopsDips

end daps_from_dips_l159_159106


namespace log5_x_l159_159549

theorem log5_x (x : ℝ) (h : x = (Real.log 2 / Real.log 4) ^ (Real.log 16 / Real.log 2) ^ 2) :
    Real.log x / Real.log 5 = -16 / (Real.log 2 / Real.log 5) := by
  sorry

end log5_x_l159_159549


namespace find_x_of_set_l159_159899

theorem find_x_of_set (x : ℕ) : (8 + 14 + 20 + 7 + x + 16) / 6 = 12 -> x = 7 :=
by 
  assume h: (8 + 14 + 20 + 7 + x + 16) / 6 = 12,
  sorry

end find_x_of_set_l159_159899


namespace car_speed_is_80_l159_159315

variable (v_car v_train t : ℝ)

-- Conditions
def train_speed (v_car : ℝ) : ℝ := 1.5 * v_car
def train_time (t : ℝ) : ℝ := t - (12.5 / 60)

-- Statement
theorem car_speed_is_80
  (h1 : 75 = v_car * t)
  (h2 : 75 = train_speed v_car * train_time t) :
  v_car = 80 :=
begin
  -- Proof is omitted
  sorry
end

end car_speed_is_80_l159_159315


namespace vasya_birthday_was_thursday_l159_159713

variable (today : String)
variable (tomorrow : String)
variable (day_after_tomorrow : String)
variable (birthday : String)

-- Conditions given in the problem
axiom birthday_not_sunday : birthday ≠ "Sunday"
axiom sunday_day_after_tomorrow : day_after_tomorrow = "Sunday"

-- From the conditions, we need to prove that Vasya's birthday was on Thursday.
theorem vasya_birthday_was_thursday : birthday = "Thursday" :=
by
  -- Fill in the proof here
  sorry

end vasya_birthday_was_thursday_l159_159713


namespace vasya_birthday_l159_159725

/--
Vasya said the day after his birthday: "It's a pity that my birthday 
is not on a Sunday this year, because more guests would have come! 
But Sunday will be the day after tomorrow..."
On what day of the week was Vasya's birthday?
-/
theorem vasya_birthday (today : string)
  (h1 : today = "Friday")
  (h2 : ∀ day : string, day ≠ "Sunday" → Vasya's_birthday day) :
  Vasya's_birthday "Thursday" := by
  sorry

end vasya_birthday_l159_159725


namespace num_two_digit_integers_with_five_factors_l159_159529

theorem num_two_digit_integers_with_five_factors :
  {n : ℕ | ∃ (p : ℕ), Prime p ∧ n = p^4 ∧ 10 ≤ n ∧ n ≤ 99}.to_finset.card = 2 :=
by sorry

end num_two_digit_integers_with_five_factors_l159_159529


namespace find_values_l159_159635

theorem find_values 
  (y1 y2 z1 z2 r1 x1 x2 : ℤ) 
  (h1 : y1 = 2) 
  (h2 : y2 = 3) 
  (h3 : z1 = 3) 
  (h4 : z2 = 5) 
  (h5 : r1 = 1) 
  (h6 : x1 = 4) 
  (h7 : x2 = 6) :
  let y := 3 * (y1 + y2) + 4
      z := 2 * z1 ^ 2 - z2
      r := 3 * r1 + 2
      x := 2 * x1 * y1 - x2 + 10
  in y = 19 ∧ z = 13 ∧ r = 5 ∧ x = 20 := by
  sorry

end find_values_l159_159635


namespace find_m_value_l159_159372

noncomputable def list_with_conditions (l : List ℤ) : Prop :=
  l.mode.getOrElse 0 = 32 ∧
  l.sum / l.length = 22 ∧
  l.minimum.getOrElse 0 = 10 ∧
  let m := l.sort.select (l.length / 2) in
  (m ∈ l) ∧
  let l1 := (l.map (λ x, if x = m then m + 10 else x)) in
  l1.sum / l1.length = 24 ∧
  l1.sort.select (l1.length / 2) = m + 10 ∧
  let l2 := (l.map (λ x, if x = m then m - 8 else x)) in
  l2.sort.select (l2.length / 2) = m - 4

theorem find_m_value (l : List ℤ) (h : list_with_conditions l) : 
  let m := l.sort.select (l.length / 2)
  in m = 20 :=
by
  sorry

end find_m_value_l159_159372


namespace find_m_of_lcm_conditions_l159_159488

theorem find_m_of_lcm_conditions (m : ℕ) (h_pos : 0 < m)
  (h1 : Int.lcm 18 m = 54)
  (h2 : Int.lcm m 45 = 180) : m = 36 :=
sorry

end find_m_of_lcm_conditions_l159_159488


namespace min_value_f_range_of_a_l159_159959

-- Conditions
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - (a + 1) * Real.log x - a / x
noncomputable def g (x : ℝ) : ℝ := 1/2 * x^2 + Real.exp x - x * Real.exp x

-- Proof problems
theorem min_value_f (x : ℝ) (a : ℝ) (hx : 1 ≤ x ∧ x ≤ Real.exp 1) :
  (a ≤ 1 → ∃ v, (∀ x ∈ Icc 1 (Real.exp 1), f x a ≥ v) ∧ v = 1 - a) ∧
  (1 < a ∧ a < Real.exp 1 → ∃ v, (∀ x ∈ Icc 1 (Real.exp 1), f x a ≥ v) ∧ v = a - (a + 1) * Real.log a - 1) ∧
  (a ≥ Real.exp 1 → ∃ v, (∀ x ∈ Icc 1 (Real.exp 1), f x a ≥ v) ∧ v = Real.exp 1 - (a + 1) - a / Real.exp 1) :=
sorry

theorem range_of_a (a : ℝ) :
  a < 1 → (∃ x1 ∈ Icc (Real.exp 1) (Real.exp 2), ∀ x2 ∈ Icc (-2) 0, f x1 a < g x2) → a ∈ Ioo ((Real.exp 2 - 2 * Real.exp 1) / (Real.exp 1 + 1)) 1 :=
sorry

end min_value_f_range_of_a_l159_159959


namespace gcd_values_360_l159_159780

theorem gcd_values_360 : ∃ d : ℕ, d = 11 ∧ ∀ a b : ℕ, a * b = 360 → ∃ (g : ℕ), g = gcd a b ∧ finite {g | g = gcd a b ∧ a * b = 360} ∧ card {g | g = gcd a b ∧ a * b = 360} = 11 :=
sorry

end gcd_values_360_l159_159780


namespace solve_for_x_l159_159255

theorem solve_for_x (x : ℝ) : x^2 + 6 * x + 8 = -(x + 4) * (x + 6) ↔ x = -4 :=
by {
  sorry
}

end solve_for_x_l159_159255


namespace exists_rat_not_int_add_pow_int_l159_159459

theorem exists_rat_not_int_add_pow_int (n : ℕ) : 
  (odd n ↔ ∃ a b : ℚ, (0 < a ∧ 0 < b ∧ ¬a ∈ ℤ ∧ ¬b ∈ ℤ) ∧ (a + b ∈ ℤ) ∧ (a^n + b^n ∈ ℤ)) :=
sorry

end exists_rat_not_int_add_pow_int_l159_159459


namespace ball_distribution_l159_159531

theorem ball_distribution :
  (∃ (f : ℕ → list ℕ), (∀ i < 4, list.sum (f i) = 6) 
  ∧ (((multiset.of (f 0)) :: (multiset.of (f 1)) :: (multiset.of (f 2)) :: (multiset.of (f 3)) :: []).to_multiset.choice = 187)) :=
sorry

end ball_distribution_l159_159531


namespace wang_ming_friends_not_set_l159_159309

-- Define the specifics of what constitutes a set
structure Set (α : Type) :=
  (elems : α → Prop)
  (distinct : ∀ x y, elems x → elems y → x = y)

-- Definition conditions for each group
def male_students_class2_grade1 : Set (String) :=
{ elems := λ x, x ∈ ["student1", "student2", "student3"], -- Hypothetical list of students
  distinct := sorry } -- Proof of distinctness

def parents_of_students : Set (String) :=
{ elems := λ x, x ∈ ["parent1", "parent2", "parent3"], -- Hypothetical list of parents
  distinct := sorry } -- Proof of distinctness

def all_relatives_li_ming : Set (String) :=
{ elems := λ x, x ∈ ["relative1", "relative2", "relative3"], -- Hypothetical list of relatives
  distinct := sorry } -- Proof of distinctness

-- Definition for Wang Ming's good friends (this should fail to be a set due to subjectivity)
def good_friends_wang_ming : String → Prop := λ x, x = "friend1" ∨ x = "friend2"

-- Problem statement: Prove Wang Ming's good friends cannot form a set
theorem wang_ming_friends_not_set : ¬ (∃ s : Set String, s.elems = good_friends_wang_ming) :=
begin
  sorry
end

end wang_ming_friends_not_set_l159_159309


namespace parabola_equation_correct_fixed_point_proof_l159_159518

theorem parabola_equation_correct (P : ℝ × ℝ) (F : ℝ × ℝ) (h : F = (2, 0)) :
  (dist P F = |P.1 + 4| - 2) -> P.2^2 = 8 * P.1 := sorry

theorem fixed_point_proof (L : ℝ → ℝ) (hL : ∃ k b, L = (λ x, k * x + b))
  (A B : ℝ × ℝ) (hA : A ∈ {(x, y) | y^2 = 8 * x}) (hB : B ∈ {(x, y) | y^2 = 8 * x})
  (hOA_OB : A.1 * B.1 = -A.2 * B.2) : ∃ k, ∃ b, L = (λ x, k * x + b) ∧ L 8 = 0 := sorry

end parabola_equation_correct_fixed_point_proof_l159_159518


namespace cube_root_neg_8_eq_neg_2_l159_159407

theorem cube_root_neg_8_eq_neg_2 : ∃ x : ℝ, x^3 = -8 ∧ x = -2 :=
by
  use -2
  split
  . show (-2 : ℝ)^3 = -8 by sorry
  . show -2 = -2 by rfl

end cube_root_neg_8_eq_neg_2_l159_159407


namespace Faye_age_l159_159000

theorem Faye_age (D E C F : ℕ) (h1 : D = E - 5) (h2 : E = C + 3) (h3 : F = C + 2) (hD : D = 18) : F = 22 :=
by
  sorry

end Faye_age_l159_159000


namespace Vasya_birthday_on_Thursday_l159_159722

-- Define the days of the week enumeration
inductive DayOfWeek
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open DayOfWeek

-- Define the conditions
def Sunday_is_the_day_after_tomorrow (today : DayOfWeek) : Prop :=
  match today with
  | Friday => true
  | _      => false

def Vasya_birthday_not_on_Sunday (birthday : DayOfWeek) : Prop :=
  birthday ≠ Sunday

def Today_is_next_day_after_birthday (today birthday : DayOfWeek) : Prop :=
  match (today, birthday) with
  | (Friday, Thursday) => true
  | _                  => false

-- The main theorem to prove Vasya's birthday was on Thursday
theorem Vasya_birthday_on_Thursday (birthday today : DayOfWeek) 
  (H1: Sunday_is_the_day_after_tomorrow today) 
  (H2: Vasya_birthday_not_on_Sunday birthday) 
  (H3: Today_is_next_day_after_birthday today birthday) : 
  birthday = Thursday :=
by
  sorry

end Vasya_birthday_on_Thursday_l159_159722


namespace Vasya_birthday_on_Thursday_l159_159719

-- Define the days of the week enumeration
inductive DayOfWeek
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open DayOfWeek

-- Define the conditions
def Sunday_is_the_day_after_tomorrow (today : DayOfWeek) : Prop :=
  match today with
  | Friday => true
  | _      => false

def Vasya_birthday_not_on_Sunday (birthday : DayOfWeek) : Prop :=
  birthday ≠ Sunday

def Today_is_next_day_after_birthday (today birthday : DayOfWeek) : Prop :=
  match (today, birthday) with
  | (Friday, Thursday) => true
  | _                  => false

-- The main theorem to prove Vasya's birthday was on Thursday
theorem Vasya_birthday_on_Thursday (birthday today : DayOfWeek) 
  (H1: Sunday_is_the_day_after_tomorrow today) 
  (H2: Vasya_birthday_not_on_Sunday birthday) 
  (H3: Today_is_next_day_after_birthday today birthday) : 
  birthday = Thursday :=
by
  sorry

end Vasya_birthday_on_Thursday_l159_159719


namespace salary_increase_mean_variance_l159_159366

variables {x : ℕ → ℝ} {n : ℕ} -- n = 10, x is the sequence of salaries

/-- Mean calculation -/
def mean (x : ℕ → ℝ) (n : ℕ) : ℝ := (∑ i in finset.range n, x i) / n

/-- Variance calculation -/
def variance (x : ℕ → ℝ) (n : ℕ) (mean_x : ℝ) : ℝ :=
(∑ i in finset.range n, (x i - mean_x) ^ 2) / n

theorem salary_increase_mean_variance (x : ℕ → ℝ) (n : ℕ) (mean_x : ℝ) (variance_x : ℝ) (h_mean : mean x n = mean_x) (h_variance : variance x n mean_x = variance_x) :
  mean (λ i, x i + 100) n = mean_x + 100 ∧
  variance (λ i, x i + 100) n (mean_x + 100) = variance_x :=
by sorry

end salary_increase_mean_variance_l159_159366


namespace complex_number_in_fourth_quadrant_l159_159265

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := 2 - complex.I
  (z.re > 0) ∧ (z.im < 0) :=
by
  let z : ℂ := 2 - complex.I
  have h1 : z.re = 2 := rfl
  have h2 : z.im = -1 := rfl
  exact And.intro (by rfl) (by rfl)

end complex_number_in_fourth_quadrant_l159_159265


namespace carbon_dioxide_moles_l159_159894

theorem carbon_dioxide_moles (HCl NaHCO3: ℕ) (h_HCl: HCl = 1) (h_NaHCO3: NaHCO3 = 1) : ∃ CO2, CO2 = 1 :=
by {
  use 1,
  exact rfl,
}

end carbon_dioxide_moles_l159_159894


namespace inscribed_circle_radius_l159_159405

def radius_of_inscribed_circle (DE DF EF : ℝ) (s r : ℝ) :=
  s = (DE + DF + EF) / 2 ∧
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF)) in
  K = s * r

theorem inscribed_circle_radius :
  ∀ (DE DF EF : ℝ), DE = 26 → DF = 15 → EF = 17 →
  ∃ r : ℝ, radius_of_inscribed_circle DE DF EF 29 r ∧ r = 81 * Real.sqrt 2 / 29 :=
by 
  intros DE DF EF hDE hDF hEF
  use (81 * Real.sqrt 2 / 29)
  split
  { rw [hDE, hDF, hEF]
    unfold radius_of_inscribed_circle
    split
    { linarith }
    { simp }
  },
  sorry

end inscribed_circle_radius_l159_159405


namespace minimal_polynomial_with_roots_l159_159437

theorem minimal_polynomial_with_roots (p : ℚ[X]) :
  (∀ x, (x = 2 + real.sqrt 3) → polynomial.aeval x p = 0) ∧
  (∀ x, (x = 2 + real.sqrt 5) → polynomial.aeval x p = 0) ∧
  polynomial.leading_coeff p = 1 →
  p = polynomial.X^4 - 8 * polynomial.X^3 + 14 * polynomial.X^2 + 8 * polynomial.X - 3 :=
by
  sorry

end minimal_polynomial_with_roots_l159_159437


namespace base_conversion_and_addition_l159_159885

theorem base_conversion_and_addition :
  let a₈ : ℕ := 3 * 8^2 + 5 * 8^1 + 6 * 8^0
  let c₁₄ : ℕ := 4 * 14^2 + 12 * 14^1 + 3 * 14^0
  a₈ + c₁₄ = 1193 :=
by
  sorry

end base_conversion_and_addition_l159_159885


namespace selection_count_l159_159169

theorem selection_count : 
  let rows := 6
  let columns := 7
  ∃ (people : Fin rows := 6 × Fin columns := 7), 
    (card { S : Finset (Fin rows := 6 × Fin columns := 7) | S.card = 3 ∧ 
      ∀ {p1 p2 : Fin rows := 6 × Fin columns := 7} , (p1 ∈ S ∧ p2 ∈ S ∧ p1 ≠ p2 → p1.fst ≠ p2.fst ∧ p1.snd ≠ p2.snd)}) = 4200 := 
by
  sorry

end selection_count_l159_159169


namespace tangent_line_eq_max_area_triangle_CPQ_l159_159482

-- Define the circle equation
def circle (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define the point A
def A : ℝ × ℝ := (1, 0)

-- Tangent case: The line passing through A and tangent to the circle
theorem tangent_line_eq (l : ℝ → ℝ) (h_line : l 1 = 0) :
  (∃ k : ℝ, ∀ x, l x = k * (x - 1) ∧
  (abs (3 * k - 4 - k) / real.sqrt (k^2 + 1) = 2 ∨ l 1 = 0) ∧
  ((l 1 = 0 → l = (λ x, 1) ∨ l = (λ x, 3 / 4 * (x - 1)))) :=
sorry

-- Intersection case: The line passing through A intersects with the circle and maximum area of triangle CPQ
theorem max_area_triangle_CPQ (l : ℝ → ℝ) (h_line : l 1 = 0) :
  (∃ k : ℝ, ∀ x, l x = k * (x - 1) ∧
  (abs (2 * k - 4) / real.sqrt (k^2 + 1) = real.sqrt 2) ∧
  l = (λ x, x - 1) ∨ l = (λ x, 7 * (x - 1))) :=
sorry

end tangent_line_eq_max_area_triangle_CPQ_l159_159482


namespace polynomial_evaluation_l159_159007

theorem polynomial_evaluation (x : ℝ) (h1 : x^2 - 4 * x - 12 = 0) (h2 : 0 < x) : x^3 - 4 * x^2 - 12 * x + 16 = 16 := 
by
  sorry

end polynomial_evaluation_l159_159007


namespace area_of_circle_l159_159851

-- Define the given condition
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 + 10 * x + 2 = 5 * y - 4 * x + 13

-- Define the proof problem
theorem area_of_circle :
  (∃ x y : ℝ, circle_eq x y) →
  ∃ r : ℝ, r = real.sqrt (265 / 4) ∧ π * r^2 = 265 * π / 4 :=
by
  intro h
  sorry

end area_of_circle_l159_159851


namespace infinite_zeros_in_S_l159_159962

-- Define the sequence a_n
def a (n : ℕ) : ℤ :=
  if n % 4 = 0 then -↑(n + 1) else
  if n % 4 = 1 then ↑n else
  if n % 4 = 2 then ↑n else
  -↑(n + 1)

-- Define the sequence S_k as partial sum of a_n
def S : ℕ → ℤ
| 0       => a 0
| (n + 1) => S n + a (n + 1)

-- Proposition: S_k contains infinitely many zeros
theorem infinite_zeros_in_S : ∀ n : ℕ, ∃ m > n, S m = 0 := sorry

end infinite_zeros_in_S_l159_159962


namespace university_expenses_deposit_l159_159784

theorem university_expenses_deposit
  (x : ℕ)
  (monthly_interest_rate : ℝ)
  (total_months : ℕ)
  (expected_expenses : ℕ)
  (a24 : ℝ)
  (a25 : ℝ)
  (a26 : ℝ) :
  let interest_factor := 1 + monthly_interest_rate
  let total_sum := x * (interest_factor^25 - 1) / (interest_factor - 1)
  monthly_interest_rate = 0.02 →
  total_months = 25 →
  expected_expenses = 60000 →
  a24 = 1.61 →
  a25 = 1.64 →
  a26 = 1.67 →
  total_sum = expected_expenses * 0.02 →
  x = 1875 := by
  sorry

end university_expenses_deposit_l159_159784


namespace range_of_f_l159_159036

def f (x : ℝ) : ℝ := (3 * x + 8) / (x - 4)

theorem range_of_f :
  set.range f = set.Iio 3 ∪ set.Ioi 3 :=
sorry

end range_of_f_l159_159036


namespace pats_and_mats_numbers_l159_159228

theorem pats_and_mats_numbers (x y : ℕ) (hxy : x ≠ y) (hx_gt_hy : x > y) 
    (h_sum : (x + y) + (x - y) + x * y + (x / y) = 98) : x = 12 ∧ y = 6 :=
by
  sorry

end pats_and_mats_numbers_l159_159228


namespace Rachel_trip_representation_l159_159235

theorem Rachel_trip_representation :
  ∃ (graph : ℕ → ℝ), graph = GraphE ∧
  (∀ t, 0 ≤ t ∧ t < t1 → graph t = f_city_traffic_north t) ∧
  (∀ t, t1 ≤ t ∧ t < t1 + 15 → graph t = distance_Annie) ∧
  (∀ t, t1 + 15 ≤ t ∧ t < t2 → graph t = f_highway_north t) ∧
  (∀ t, t2 ≤ t ∧ t < t2 + 120 → graph t = distance_shopping_center) ∧
  (∀ t, t2 + 120 ≤ t ∧ t < t3 → graph t = f_highway_south t) ∧
  (∀ t, t3 ≤ t ∧ t < t3 + 15 → graph t = distance_Annie) ∧
  (∀ t, t3 + 15 ≤ t ∧ t < t4 → graph t = f_city_traffic_south t) ∧
  (graph t4 = 0) :=
by
  sorry

end Rachel_trip_representation_l159_159235


namespace probability_of_both_qualified_bottles_expected_number_of_days_with_unqualified_milk_l159_159683

noncomputable def qualification_rate : ℝ := 0.8
def probability_both_qualified (rate : ℝ) : ℝ := rate * rate
def unqualified_rate (rate : ℝ) : ℝ := 1 - rate
def expected_days (n : ℕ) (p : ℝ) : ℝ := n * p

theorem probability_of_both_qualified_bottles : 
  probability_both_qualified qualification_rate = 0.64 :=
by sorry

theorem expected_number_of_days_with_unqualified_milk :
  expected_days 3 (unqualified_rate qualification_rate) = 1.08 :=
by sorry

end probability_of_both_qualified_bottles_expected_number_of_days_with_unqualified_milk_l159_159683


namespace find_angle_C_find_area_triang_l159_159910

-- Statement 1: Given certain trigonometric conditions within a triangle, prove the angle C.
theorem find_angle_C (A B C : ℝ) (a b c : ℝ) (h1 : c * sin A = √3 * a * cos C) (h2 : sin A ≠ 0) (h3 : C ∈ set.Ioo 0 real.pi) :
  C = real.pi / 3 :=
sorry

-- Statement 2: Given additional conditions and the previously proven angle, prove the area of the triangle.
theorem find_area_triang (A B : ℝ) (a b c : ℝ) (hC : C = real.pi / 3) (h4 : c = √21) (h5 : sin C + sin (B - A) = 5 * sin (2 * A)) :
  let S := (1 / 2) * a * b * sin C
  in S = 5 * √3 / 4 :=
sorry

end find_angle_C_find_area_triang_l159_159910


namespace coronavirus_transmission_l159_159680

theorem coronavirus_transmission (x : ℝ) 
  (H: (1 + x) ^ 2 = 225) : (1 + x) ^ 2 = 225 :=
  by
    sorry

end coronavirus_transmission_l159_159680


namespace parallel_lines_condition_l159_159801

theorem parallel_lines_condition (a : ℝ) :
  ( ∀ x y : ℝ, (a * x + 2 * y + 2 = 0 → ∃ C₁ : ℝ, x - 2 * y = C₁) 
  ∧ (x + (a - 1) * y + 1 = 0 → ∃ C₂ : ℝ, x - 2 * y = C₂) )
  ↔ a = -1 :=
sorry

end parallel_lines_condition_l159_159801


namespace probability_two_red_two_blue_l159_159350

theorem probability_two_red_two_blue (total_red total_blue : ℕ) (red_taken blue_taken selected : ℕ)
  (h_red_total : total_red = 12) (h_blue_total : total_blue = 8) (h_selected : selected = 4)
  (h_red_taken : red_taken = 2) (h_blue_taken : blue_taken = 2) :
  (Nat.choose total_red red_taken) * (Nat.choose total_blue blue_taken) /
    (Nat.choose (total_red + total_blue) selected : ℚ) = 1848 / 4845 := 
by 
  sorry

end probability_two_red_two_blue_l159_159350


namespace combined_hits_and_misses_total_l159_159576

/-
  Prove that given the conditions for each day regarding the number of misses and
  the ratio of misses to hits, the combined total of hits and misses for the 
  three days is 322.
-/

theorem combined_hits_and_misses_total :
  (∀ (H1 : ℕ) (H2 : ℕ) (H3 : ℕ), 
    (2 * H1 = 60) ∧ (3 * H2 = 84) ∧ (5 * H3 = 100) →
    60 + 84 + 100 + H1 + H2 + H3 = 322) :=
by
  sorry

end combined_hits_and_misses_total_l159_159576


namespace vasya_birthday_was_thursday_l159_159717

variable (today : String)
variable (tomorrow : String)
variable (day_after_tomorrow : String)
variable (birthday : String)

-- Conditions given in the problem
axiom birthday_not_sunday : birthday ≠ "Sunday"
axiom sunday_day_after_tomorrow : day_after_tomorrow = "Sunday"

-- From the conditions, we need to prove that Vasya's birthday was on Thursday.
theorem vasya_birthday_was_thursday : birthday = "Thursday" :=
by
  -- Fill in the proof here
  sorry

end vasya_birthday_was_thursday_l159_159717


namespace general_term_formula_sum_of_squares_lt_l159_159507

def seq (n : ℕ) : ℝ :=
  match n with
  | 0     => 0
  | 1     => 1
  | 2     => 1 / 4
  | n + 1 => (n-1) * (seq n) / (n - seq n)

theorem general_term_formula (n : ℕ) (hn : n > 0) :
  seq n = 1 / (3 * n - 2) :=
sorry

theorem sum_of_squares_lt (n : ℕ) (hn : n > 0) :
  ∑ k in finset.range (n + 1), (seq k)^2 < 7 / 6 :=
sorry

end general_term_formula_sum_of_squares_lt_l159_159507


namespace max_jogs_l159_159849

theorem max_jogs (jags jigs jogs jugs : ℕ) : 2 * jags + 3 * jigs + 8 * jogs + 5 * jugs = 72 → jags ≥ 1 → jigs ≥ 1 → jugs ≥ 1 → jogs ≤ 7 :=
by
  sorry

end max_jogs_l159_159849


namespace triangle_angle_measurements_l159_159564

theorem triangle_angle_measurements
    (A B C D E : Point)
    (h1 : Segment A B) 
    (h2 : Segment A C) 
    (h3 : Segment B C)
    (h4 : angle_bisector h2 h3 D)
    (h5 : angle_bisector h1 h2 E)
    (h6 : angle B D E = 24)
    (h7 : angle C E D = 18) :
    angle A B C = 12 ∧ angle A C B = 72 ∧ angle B A C = 96 :=
by
  sorry

end triangle_angle_measurements_l159_159564


namespace sum_cubes_eq_neg_27_l159_159199

theorem sum_cubes_eq_neg_27 (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
 (h_eq : (a^3 + 9) / a = (b^3 + 9) / b ∧ (b^3 + 9) / b = (c^3 + 9) / c) :
 a^3 + b^3 + c^3 = -27 :=
by
  sorry

end sum_cubes_eq_neg_27_l159_159199


namespace fraction_of_capital_subscribed_l159_159388

theorem fraction_of_capital_subscribed (T : ℝ) (x : ℝ) :
  let B_capital := (1 / 4) * T
  let C_capital := (1 / 5) * T
  let Total_profit := 2445
  let A_profit := 815
  A_profit / Total_profit = x → x = 1 / 3 :=
by
  sorry

end fraction_of_capital_subscribed_l159_159388


namespace hotdog_eating_ratio_l159_159003

theorem hotdog_eating_ratio (x : ℕ) 
  (h1 : 12 > 0) 
  (h2 : 18 > 0) 
  (h3 : 0.75 * x = 18) : 
  (x / 12 = 2) := 
by
  sorry

end hotdog_eating_ratio_l159_159003


namespace vasya_birthday_l159_159706

def DayOfWeek := 
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

open DayOfWeek

def day_after (d : DayOfWeek) : DayOfWeek :=
match d with
| Sunday => Monday
| Monday => Tuesday
| Tuesday => Wednesday
| Wednesday => Thursday
| Thursday => Friday
| Friday => Saturday
| Saturday => Sunday

def day_before (d : DayOfWeek) : DayOfWeek :=
match d with
| Sunday => Saturday
| Monday => Sunday
| Tuesday => Monday
| Wednesday => Tuesday
| Thursday => Wednesday
| Friday => Thursday
| Saturday => Friday

theorem vasya_birthday (day_said : DayOfWeek) 
    (h1 : day_after (day_after day_said) = Sunday) 
    (h2 : day_said = day_after VasyaBirthday) 
    : VasyaBirthday = Thursday :=
sorry

end vasya_birthday_l159_159706


namespace chord_length_not_equal_l159_159495

theorem chord_length_not_equal (k : ℝ) :
  let ellipse_eq := λ x y : ℝ, (x^2) / 8 + (y^2) / 4 = 1
  let line1 := λ x : ℝ, k * x + 1
  let lineD := λ x : ℝ, -k * x + 2
  ∀ x1 x2 y1 y2 : ℝ,
    (ellipse_eq x1 (line1 x1)) ∧ (ellipse_eq x2 (line1 x2)) ∧
    (ellipse_eq x1 (lineD x1)) ∧ (ellipse_eq x2 (lineD x2)) →
    (x1 ≠ x2 → y1 ≠ y2) :=
sorry

end chord_length_not_equal_l159_159495


namespace area_of_path_l159_159381

theorem area_of_path (length_field width_field path_width : ℝ) :
  let length_total := length_field + 2 * path_width,
      width_total := width_field + 2 * path_width,
      area_total := length_total * width_total,
      area_field := length_field * width_field in
  area_total - area_field = 675 :=
by
  sorry

example : area_of_path 75 55 2.5 :=
by
  sorry

end area_of_path_l159_159381


namespace camp_children_total_l159_159178

theorem camp_children_total (C : ℕ) :
  (0.10 * C = 0.05 * (C + 60) → C = 60) :=
by
  intro h
  have h1: 0.10 * C = 0.05 * (C + 60),
  { exact h },
  sorry

end camp_children_total_l159_159178


namespace prism_ordered_triples_count_l159_159383

theorem prism_ordered_triples_count :
  ∃ (a : ℕ) (b : ℕ) (c : ℕ), 
  a ≤ b ∧ b ≤ c ∧ b = 2023 ∧ (∃ k : ℚ, k < 1 ∧ a = k * b ∧ c = b / k) ∧ ∃ n = 7, ∃ (S : Finset (ℕ × ℕ × ℕ)) , 
  (∀ (x y z : ℕ), (x, y, z) ∈ S ↔ x ≤ y ∧ y ≤ z ∧ y = 2023 ∧ x = y * 2023 / z ∧ 2023^2 = z * x) ∧ 
  n = S.card :=
sorry

end prism_ordered_triples_count_l159_159383


namespace candy_diff_eq_58_l159_159401

variable (candyPart1 candyPart2 chocolate : ℕ)
variable (totalCandy totalChocolate : ℕ)

def totalCandy := candyPart1 + candyPart2
def totalChocolate := chocolate
def candyMoreThanChocolate := totalCandy - totalChocolate

-- given conditions
def bobbyAteCandy := candyPart1 = 38 ∧ candyPart2 = 36
def bobbyAteChocolate := chocolate = 16

-- The statement to prove
theorem candy_diff_eq_58 (h1 : bobbyAteCandy) (h2 : bobbyAteChocolate) : candyMoreThanChocolate = 58 :=
by
  sorry

end candy_diff_eq_58_l159_159401


namespace large_bottle_capacity_is_correct_l159_159698

-- Define that we have 12 liters and 400 milliliters of total oil
def total_oil_ml : ℕ := 12 * 1000 + 400

-- Assume that the capacity of the small bottle is s milliliters
variables (s : ℕ)

-- Define the capacity of the large bottle in terms of the small bottle
def large_bottle_ml : ℕ := s + (2 * 1000) + 600

-- Define the total capacity being the sum of the capacities of the small and large bottles
def total_capacity : Prop := s + large_bottle_ml s = total_oil_ml

-- Define the large bottle capacity in liters
def large_bottle_liters (s : ℕ) : ℝ := (large_bottle_ml s : ℝ) / 1000

-- The proof statement. We need to prove that if there are 12 liters and 400 milliliters of oil
-- and the large bottle can hold 2 liters and 600 milliliters more than the small bottle, then 
-- the capacity of the large bottle is 7.5 liters
theorem large_bottle_capacity_is_correct :
  (∃ s, total_capacity s) → large_bottle_liters (9800 / 2) = 7.5 :=
by
  sorry

end large_bottle_capacity_is_correct_l159_159698


namespace count_two_digit_numbers_with_perfect_square_digit_sum_eq_17_l159_159975

-- Define two-digit numbers, their digit sum, and condition for perfect squares
def is_two_digit_number (n : ℕ) : Prop := n >= 10 ∧ n < 100

def digit_sum (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  tens + units

def is_perfect_square_less_than_eq_25 (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m ∧ m <= 25

-- Define the main theorem to prove
theorem count_two_digit_numbers_with_perfect_square_digit_sum_eq_17 :
  (finset.filter (λ n, is_two_digit_number n ∧ digit_sum n ∈ {1, 4, 9, 16}) 
    (finset.range 100)).card = 17 :=
sorry

end count_two_digit_numbers_with_perfect_square_digit_sum_eq_17_l159_159975


namespace dodecagon_ratio_l159_159574

theorem dodecagon_ratio (A B C D E F G H I J K L M N : Point)
  (h1 : is_regular_dodecagon A B C D E F G H I J K L)
  (hM : M = midpoint C D)
  (hN : N = midpoint I J) :
  ratio_area ABCM GHIJMN = 2 / 3 :=
sorry

end dodecagon_ratio_l159_159574


namespace num_factors_x_l159_159988

theorem num_factors_x (x : ℕ) (h : 2011^(2011^2012) = x^x) : ∃ n : ℕ, n = 2012 ∧  ∀ d : ℕ, d ∣ x -> d ≤ n :=
sorry

end num_factors_x_l159_159988


namespace find_angle_BKC_equals_30_l159_159926

section geometric_proof

-- Assume ABCD is a square with sides of equal length
variables (A B C D K : Type) [square ABCD]

-- K is a point on the extension of diagonal AC such that BK = AC
variables [on_extension_of_diagonal A B C D K AC]
variables [eq_length BK AC]

-- Define the proof requirement: angle ∠BKC = 30°
def angle_BKC_equals_30 : Prop :=
  ∠BKC = 30°

-- State the theorem
theorem find_angle_BKC_equals_30 : angle_BKC_equals_30 :=
sorry

end geometric_proof

end find_angle_BKC_equals_30_l159_159926


namespace find_N_is_20_l159_159896

theorem find_N_is_20 : ∃ (N : ℤ), ∃ (u v : ℤ), (N + 5 = u ^ 2) ∧ (N - 11 = v ^ 2) ∧ (N = 20) :=
by
  sorry

end find_N_is_20_l159_159896


namespace median_is_6_or_9_l159_159061

-- Definitions and conditions
def dataSet (x : ℕ) := {a : ℕ | a = 2 ∨ a = 9 ∨ a = 6 ∨ a = 10 ∨ a = x}

-- Inequality conditions
def cond1 (x : ℕ) : Prop := 2021 * x - 4042 > 0
def cond2 (x : ℕ) : Prop := 14 - 2 * (x - 3) > 0

-- Prove the median is 6 or 9
theorem median_is_6_or_9 (x : ℕ) (h1 : cond1 x) (h2 : cond2 x) (hx : x ∈ dataSet x) : 
  (∀ A : Finset ℕ, A = {2, 6, 9, 10, x} → (A.median = 6 ∨ A.median = 9)) :=
sorry

end median_is_6_or_9_l159_159061


namespace units_digit_factorial_sum_l159_159452

theorem units_digit_factorial_sum : 
  (1! + 2! + 3! + 4! + ∑ n in (finset.range 2006).filter (λ x, x ≥ 5), (n + 5)!) % 10 = 3 :=
by
  sorry

end units_digit_factorial_sum_l159_159452


namespace four_digit_multiple_of_11_and_7_l159_159011

theorem four_digit_multiple_of_11_and_7 (a b c d : ℕ) (N : ℕ) (h : N = 3454) :
  (N = 1000 * a + 100 * b + 10 * c + d) ∧ 
  (a - b + c - d = 0 ∨ a - b + c - d = 11 ∨ a - b + c - d = -11) ∧
  (10 * a + c) % 7 = 0 ∧
  (a + b + c + d = d^2) :=
by
  have h3454 : N = 3454 := h
  sorry

end four_digit_multiple_of_11_and_7_l159_159011


namespace reflect_across_y_axis_l159_159267

theorem reflect_across_y_axis (x y : ℝ) (P : x = -1 ∧ y = 2) :
  let P' := (1, 2) in P' = (if x = -1 then (-x, y) else (x, y)) :=
sorry

end reflect_across_y_axis_l159_159267


namespace range_of_a_l159_159677

noncomputable def log_range {a : ℝ} (x : ℝ) : Prop :=
    | log a x | > 1

theorem range_of_a (a : ℝ) :
    (∀ x : ℝ, x ∈ set.Ici 2 → log_range a x) ↔ (1 / 2 < a ∧ a < 1) ∨ (1 < a ∧ a < 2) :=
by
  sorry

end range_of_a_l159_159677


namespace vasya_birthday_l159_159726

/--
Vasya said the day after his birthday: "It's a pity that my birthday 
is not on a Sunday this year, because more guests would have come! 
But Sunday will be the day after tomorrow..."
On what day of the week was Vasya's birthday?
-/
theorem vasya_birthday (today : string)
  (h1 : today = "Friday")
  (h2 : ∀ day : string, day ≠ "Sunday" → Vasya's_birthday day) :
  Vasya's_birthday "Thursday" := by
  sorry

end vasya_birthday_l159_159726


namespace solve_for_y_l159_159543

variables (x y : ℤ)

theorem solve_for_y (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 :=
by
  hint sorry

end solve_for_y_l159_159543


namespace base8_subtraction_correct_l159_159746

def base8_sub (a b : Nat) : Nat := sorry  -- function to perform base 8 subtraction

theorem base8_subtraction_correct :
  base8_sub 0o126 0o45 = 0o41 := sorry

end base8_subtraction_correct_l159_159746


namespace simplify_fraction_l159_159427

theorem simplify_fraction (x y : ℝ) (h : x ≠ y) : (x^6 - y^6) / (x^3 - y^3) = x^3 + y^3 :=
by
  sorry

end simplify_fraction_l159_159427


namespace full_recipes_needed_l159_159389

theorem full_recipes_needed :
  (let total_students := 108 in 
   let attendance_rate := 0.60 in
   let cookies_per_student := 3 in
   let cookies_per_recipe := 18 in
   let attending_students := Int.ceil (total_students * attendance_rate) in
   let total_cookies_needed := attending_students * cookies_per_student in
   Int.ceil (total_cookies_needed / cookies_per_recipe) = 11) := 
sorry

end full_recipes_needed_l159_159389


namespace find_m_value_l159_159082

theorem find_m_value (m : ℝ) (h : ∃ k : ℝ, ∀ x : ℝ, (m + 1) * x^(m^2 - 2) = k * x^(-1)) : m = 1 :=
sorry

end find_m_value_l159_159082


namespace total_fish_in_lake_l159_159812

-- Given conditions:
def initiallyTaggedFish : ℕ := 100
def capturedFish : ℕ := 100
def taggedFishInAugust : ℕ := 5
def taggedFishMortalityRate : ℝ := 0.3
def newcomerFishRate : ℝ := 0.2

-- Proof to show that the total number of fish at the beginning of April is 1120
theorem total_fish_in_lake (initiallyTaggedFish capturedFish taggedFishInAugust : ℕ) 
  (taggedFishMortalityRate newcomerFishRate : ℝ) : 
  (taggedFishInAugust : ℝ) / (capturedFish * (1 - newcomerFishRate)) = 
  ((initiallyTaggedFish * (1 - taggedFishMortalityRate)) : ℝ) / (1120 : ℝ) :=
by 
  sorry

end total_fish_in_lake_l159_159812


namespace binomial_coefficient_third_term_l159_159560

theorem binomial_coefficient_third_term (x a : ℝ) (h : 10 * a^3 * x = 80) : a = 2 :=
by
  sorry

end binomial_coefficient_third_term_l159_159560


namespace max_rectangle_area_l159_159826

theorem max_rectangle_area (l w : ℕ) (h_perimeter : 2 * l + 2 * w = 40) (h1 : l + w = 20) (hlw : l = 10 ∨ w = 10) : 
(l = 10 ∧ w = 10 ∧ l * w = 100) :=
by sorry

end max_rectangle_area_l159_159826


namespace find_line_m_l159_159491

/-- Define point P -/
def P : ℝ × ℝ := (-2, 5)

/-- Define line l given point P and slope -3/4 -/
def line_l_slope := -3/4
def line_l := { l : ℝ × ℝ → Prop | ∃ c : ℝ, ∀ (x y : ℝ), l (x, y) ↔ (x - (-2)) * (-3) = (y - 5) * 4 == 0 }

/-- Define the distance between point P and a line as a given distance formula -/
def distance (P : ℝ × ℝ) (l : ℝ × ℝ → Prop) : ℝ := sorry -- Distance formula implementation

/-- The proof problem -/
theorem find_line_m (c : ℝ) :
  (∀ (x y : ℝ), (3 * x + 4 * y + c = 0)) ∧
  (distance P (λ p, 3 * p.1 + 4 * p.2 + c = 0) = 3) →
  (c = 1 ∨ c = -29) :=
by
  sorry

end find_line_m_l159_159491


namespace problem_relationship_l159_159912

noncomputable def a := 0.5^(-1/3 : ℝ)
noncomputable def b := (3/5 : ℝ)^(-1/3 : ℝ)
noncomputable def c := Real.logBase 2.5 1.5

theorem problem_relationship :
  c < b ∧ b < a :=
by
  sorry

end problem_relationship_l159_159912


namespace vasya_birthday_l159_159711

def DayOfWeek := 
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

open DayOfWeek

def day_after (d : DayOfWeek) : DayOfWeek :=
match d with
| Sunday => Monday
| Monday => Tuesday
| Tuesday => Wednesday
| Wednesday => Thursday
| Thursday => Friday
| Friday => Saturday
| Saturday => Sunday

def day_before (d : DayOfWeek) : DayOfWeek :=
match d with
| Sunday => Saturday
| Monday => Sunday
| Tuesday => Monday
| Wednesday => Tuesday
| Thursday => Wednesday
| Friday => Thursday
| Saturday => Friday

theorem vasya_birthday (day_said : DayOfWeek) 
    (h1 : day_after (day_after day_said) = Sunday) 
    (h2 : day_said = day_after VasyaBirthday) 
    : VasyaBirthday = Thursday :=
sorry

end vasya_birthday_l159_159711


namespace sequence_geometric_and_sum_l159_159060

theorem sequence_geometric_and_sum (a : ℕ → ℝ) 
  (h1 : a 1 = 2 / 3) 
  (h2 : ∀ n, a (n + 1) = 2 * a n / (a n + 1)) :
  (∃ (b : ℕ → ℝ) (r : ℝ), 
    b 1 = 1 / a 1 - 1 ∧ 
    (∀ n, b (n + 1) = r * b n) ∧ 
    r = 1 / 2) 
  ∧ 
  (∀ n, 
    let S_n := (∑ k in Finset.range n, k / a k) in 
    S_n = (n^2 + n + 4) / 2 - (2 + n) / 2^n) :=
by
  -- proof goes here
  sorry

end sequence_geometric_and_sum_l159_159060


namespace find_x_of_equation_l159_159986

-- Defining the condition and setting up the proof goal
theorem find_x_of_equation
  (h : (1/2)^25 * (1/x)^12.5 = 1/(18^25)) :
  x = 0.1577 := 
sorry

end find_x_of_equation_l159_159986


namespace solve_equation_l159_159694

-- Let x be a real number.
variable {x : ℝ}

-- Define the equation.
def equation : Prop := (5 / (x + 1)) - (4 / x) = 0

-- State the theorem to prove.
theorem solve_equation : equation → x = 4 :=
sorry

end solve_equation_l159_159694


namespace time_to_travel_BA_l159_159691

noncomputable theory

-- Define the conditions
def total_distance : ℝ := 21
def uphill_speed : ℝ := 4
def downhill_speed : ℝ := 6
def time_to_travel_AB : ℝ := 4.25

-- Proof statement
theorem time_to_travel_BA :
  ∃ (du dd : ℝ), (du + dd = total_distance) ∧ 
                 (du / uphill_speed + dd / downhill_speed = time_to_travel_AB) ∧
                 ((dd / uphill_speed) + (du / downhill_speed) = 4.5) :=
sorry

end time_to_travel_BA_l159_159691


namespace arithmetic_seq_and_sum_properties_l159_159063

noncomputable def arithmetic_sequence (d : ℕ) : ℕ → ℕ 
| 0     := 1
| (n+1) := arithmetic_sequence d n + d

def is_geometric_sequence (a1 a2 a5 : ℕ) : Prop := a2^2 = a1 * a5

def general_formula_arith_seq : Prop :=
  ∀ (a1 : ℕ), a1 = 1 →
  ∃ (d : ℕ) (a : ℕ → ℕ), d = 2 ∧ a = λ n : ℕ, 2*n - 1

def sum_first_n_terms_b (a : ℕ → ℕ) : ℕ → ℝ 
| 0     := 0
| (n+1) := (1 : ℝ) / ((a n).toReal * (a (n+1)).toReal) + sum_first_n_terms_b n

def correct_sum_first_n_terms_b (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, sum_first_n_terms_b a n = (n.toReal / (2*n.toReal + 1))

theorem arithmetic_seq_and_sum_properties :
  (∃ a : ℕ → ℕ, general_formula_arith_seq → (∀ n : ℕ, a(n) = 2*n - 1)) →
  (∀ n : ℕ, correct_sum_first_n_terms_b (λ n : ℕ, 2*n - 1)) :=
sorry

end arithmetic_seq_and_sum_properties_l159_159063


namespace ratio_of_sums_l159_159492

noncomputable def seq_sum (seq : Nat → ℝ) (n : Nat) : ℝ :=
  (Finset.range n).sum seq

variable {a b : Nat → ℝ}

noncomputable def S (n : Nat) : ℝ := seq_sum a n
noncomputable def T (n : Nat) : ℝ := seq_sum b n

theorem ratio_of_sums (h : ∀ n : ℕ, S n / T n = (3 * n + 2) / (2 * n + 1)) :
  a 7 / b 5 = 41 / 19 :=
by
  sorry

end ratio_of_sums_l159_159492


namespace vasya_birthday_was_thursday_l159_159714

variable (today : String)
variable (tomorrow : String)
variable (day_after_tomorrow : String)
variable (birthday : String)

-- Conditions given in the problem
axiom birthday_not_sunday : birthday ≠ "Sunday"
axiom sunday_day_after_tomorrow : day_after_tomorrow = "Sunday"

-- From the conditions, we need to prove that Vasya's birthday was on Thursday.
theorem vasya_birthday_was_thursday : birthday = "Thursday" :=
by
  -- Fill in the proof here
  sorry

end vasya_birthday_was_thursday_l159_159714


namespace log_a_b_lt_a_b_lt_b_a_l159_159052

theorem log_a_b_lt_a_b_lt_b_a (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) : 
  log a b < a^b ∧ a^b < b^a :=
by
  sorry

end log_a_b_lt_a_b_lt_b_a_l159_159052


namespace g_1000_is_1820_l159_159270

-- Definitions and conditions from the problem
def g (n : ℕ) : ℕ := sorry -- exact definition is unknown, we will assume conditions

-- Conditions as given
axiom g_g (n : ℕ) : g (g n) = 3 * n
axiom g_3n_plus_1 (n : ℕ) : g (3 * n + 1) = 3 * n + 2

-- Statement to prove
theorem g_1000_is_1820 : g 1000 = 1820 :=
by
  sorry

end g_1000_is_1820_l159_159270


namespace value_of_h_otimes_h_otimes_h_l159_159871

variable (h x y : ℝ)

-- Define the new operation
def otimes (x y : ℝ) := x^3 - x * y + y^2

-- Prove that h ⊗ (h ⊗ h) = h^6 - h^4 + h^3
theorem value_of_h_otimes_h_otimes_h :
  otimes h (otimes h h) = h^6 - h^4 + h^3 := by
  sorry

end value_of_h_otimes_h_otimes_h_l159_159871


namespace Vovochka_correct_pairs_count_l159_159148

def no_carry_pairs_count (digit_sum: ℕ → ℕ → Prop) : ℕ :=
  let count_pairs (lim: ℕ) : ℕ := (lim * (lim + 1)) / 2
  let digit_valid_pairs : ℕ := count_pairs 9
  (digit_valid_pairs * digit_valid_pairs) * 81

def digit_sum (x y: ℕ) : Prop := (x + y ≤ 9)

theorem Vovochka_correct_pairs_count :
  no_carry_pairs_count digit_sum = 244620 := by
  sorry

end Vovochka_correct_pairs_count_l159_159148


namespace coeff_x9_exp_eq_num_ways_l159_159903

/-- 
Prove that the coefficient of x^9 in the expanded expression of 
(1+x)(1+x^2)(1+x^3)...(1+x^11) is equal to the number of ways to 
select weights from {1, 2, 3, ..., 11} grams such that their total weight 
is exactly 9 grams.
-/
theorem coeff_x9_exp_eq_num_ways :
  (coeff (expand (1 + x) * (1 + x^2) * (1 + x^3) * ... * (1 + x^11) x^9) = 
    cardinal {ws : set ℕ | ∀ w ∈ ws, w ∈ {1, 2, 3, ..., 11} ∧ (sum ws) = 9}) :=
sorry

end coeff_x9_exp_eq_num_ways_l159_159903


namespace range_of_f_l159_159023

noncomputable def f (x : ℝ) : ℝ := (3 * x + 8) / (x - 4)

theorem range_of_f : Set.range f = {y : ℝ | y ≠ 3} :=
sorry

end range_of_f_l159_159023


namespace product_single_three_digit_l159_159687

theorem product_single_three_digit (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 100 ≤ b ∧ b ≤ 999) :
  100 ≤ a * b ∧ a * b < 10000 := 
begin
  sorry
end

end product_single_three_digit_l159_159687


namespace cube_root_neg_eight_l159_159410

/-- The cube root of -8 is -2. -/
theorem cube_root_neg_eight : real.cbrt (-8) = -2 :=
by 
  -- The proof steps would go here, but we include 'sorry' to indicate we're not proving it now.
  sorry

end cube_root_neg_eight_l159_159410


namespace boy_scouts_with_permission_slips_l159_159370

theorem boy_scouts_with_permission_slips (S : ℕ) 
  (H1 : 0.60 * S = total_slips) 
  (H2 : 0.45 * S = boy_scouts) 
  (H3 : 0.6818 * (0.55 * S) = girl_scouts_with_slips) :
  B = 50 :=
by
  -- Define the total_slips, boy_scouts and girl_scouts_with_slips as given in hypotheses
  let total_slips := 0.60 * S
  let boy_scouts := 0.45 * S
  let girl_scouts := 0.55 * S
  let girl_scouts_with_slips := 0.6818 * girl_scouts
  -- Calculate boy scouts with slips
  let boy_scouts_with_slips := total_slips - girl_scouts_with_slips
  -- Express B in terms of percentage
  let B := boy_scouts_with_slips / boy_scouts * 100
  -- Assert the final result
  sorry

end boy_scouts_with_permission_slips_l159_159370


namespace PMNC_is_cyclic_l159_159603

-- Definition of collinear points and conditions
variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Collinear points
axiom collinear : ∀ (A B C D : Type), Type

-- Distances
axiom AB_eq_BC : distance A B = distance B C
axiom AC_eq_CD : distance A C = distance C D

-- Circle passing through points B and D
axiom circle_w : is_circumscribed B D

-- Line through A intersects circle at P and Q with Q in segment AP
variables (P Q : Type)
axiom line_intersects_circle : ∃ (P Q : Type), lies_on_line A P ∧ lies_on_line A Q ∧ lies_on_circle P w ∧ lies_on_circle Q w ∧ lies_on_segment A Q P

-- Midpoint M of PD
variables (M : Type)
axiom midpoint_M : ∃ M, is_midpoint M P D

-- Reflection R of Q across line l
variables (l : Type) [has_reflection l]
variables (R : Type)
axiom reflection_R : reflection l Q = R

-- Intersection of PR and MB at point N
variables (N : Type)
axiom intersect_N : ∃ N, lies_on_intersection (line_segment P R) (line_segment M B) N

-- Prove quadrilateral PMNC is cyclic
theorem PMNC_is_cyclic : is_cyclic_quadrilateral P M N C :=
by
  -- Insert proof here
  sorry

end PMNC_is_cyclic_l159_159603


namespace trigonometric_identity_simplification_l159_159657

theorem trigonometric_identity_simplification :
  ∀ (deg : Type) [Real.deg_eq_rad deg],
  let x := Real.deg_to_rad deg
  in (Real.tan (20 * x) + Real.tan (30 * x) + Real.tan (60 * x) + Real.tan (80 * x)) / Real.cos (10 * x) = 
     2 / (Real.sqrt 3 * Real.sin (70 * x) * (Real.sin (10 * x))^2) :=
by
  sorry

end trigonometric_identity_simplification_l159_159657


namespace solve_x_l159_159251

theorem solve_x (x : ℝ) (h : x^2 + 6 * x + 8 = -(x + 4) * (x + 6)) : 
  x = -4 := 
by
  sorry

end solve_x_l159_159251


namespace repeating_decimal_six_denominator_l159_159671

theorem repeating_decimal_six_denominator :
  let S : ℝ := 2 / 3 in
  S = 0.6666666... ∧ (∀ E: (ℤ → ℝ), S = E) → denominator (2/3) = 3 :=
sorry

end repeating_decimal_six_denominator_l159_159671


namespace best_fit_line_slope_l159_159637

variables {x1 x2 x3 y1 y2 y3 : ℝ}

theorem best_fit_line_slope (h1 : x1 < x2) (h2 : x2 < x3) (h3 : x3 - x2 = x2 - x1) : 
  let b := (y3 - y1) / (x3 - x1) in 
  b = (y3 - y1) / (x3 - x1) :=
sorry

end best_fit_line_slope_l159_159637


namespace probability_option_one_correct_probability_option_two_correct_compare_probabilities_events_independence_l159_159226

-- Definitions for the problem conditions
def box_A_balls := {red := 2, yellow := 4}
def fair_die_outcomes := {1, 2, 3, 4, 5, 6}

-- Probability calculations for the given conditions
def probability_gift_option_one : ℚ :=
  (2/6) * (4/5) + (4/6) * (2/5)

def probability_gift_option_two : ℚ :=
  (2/6) * (4/6) + (4/6) * (2/6)

def probability_gift_option_one_at_least_one : ℚ :=
  1 - ((4/6) * (3/5))

def probability_gift_option_two_at_least_one : ℚ :=
  1 - ((4/6) * (4/6))

def probability_events_independent (E F : event) : Prop :=
  P(E ∩ F) = P(E) * P(F)

-- Prove desired statements
theorem probability_option_one_correct : probability_gift_option_one = 8/15 := sorry

theorem probability_option_two_correct : probability_gift_option_two = 4/9 := sorry

theorem compare_probabilities : probability_gift_option_one_at_least_one > probability_gift_option_two_at_least_one := sorry

theorem events_independence : 
  probability_events_independent (event_first_roll_is_1) (event_sum_of_rolls_is_7) := sorry

end probability_option_one_correct_probability_option_two_correct_compare_probabilities_events_independence_l159_159226


namespace simplify_expression_l159_159786

variable (a b : ℝ)

theorem simplify_expression (h : 0 < a ∧ a < 2b) :
  1.15 * (sqrt (a^2 - 4*a*b + 4*b^2) / sqrt (a^2 + 4*a*b + 4*b^2))
  - (8*a*b / (a^2 - 4*b^2)) + (2*b / (a - 2*b)) = a / (2*b - a) := sorry

end simplify_expression_l159_159786


namespace sum_of_cubes_eq_neg_27_l159_159203

theorem sum_of_cubes_eq_neg_27
  (a b c : ℝ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_condition : (a^3 + 9) / a = (b^3 + 9) / b ∧ (b^3 + 9) / b = (c^3 + 9) / c) :
  a^3 + b^3 + c^3 = -27 :=
sorry

end sum_of_cubes_eq_neg_27_l159_159203


namespace coefficient_of_x_pow_8_in_expansion_l159_159296

theorem coefficient_of_x_pow_8_in_expansion :
  (∑ k in Finset.range (11 + 1), (Nat.choose 11 k) * x^(11 - k) * (-1)^k)  = -165 * x^8 :=
by
  sorry

end coefficient_of_x_pow_8_in_expansion_l159_159296


namespace Vovochka_correct_pairs_count_l159_159147

def no_carry_pairs_count (digit_sum: ℕ → ℕ → Prop) : ℕ :=
  let count_pairs (lim: ℕ) : ℕ := (lim * (lim + 1)) / 2
  let digit_valid_pairs : ℕ := count_pairs 9
  (digit_valid_pairs * digit_valid_pairs) * 81

def digit_sum (x y: ℕ) : Prop := (x + y ≤ 9)

theorem Vovochka_correct_pairs_count :
  no_carry_pairs_count digit_sum = 244620 := by
  sorry

end Vovochka_correct_pairs_count_l159_159147


namespace triangle_inradius_l159_159577

theorem triangle_inradius (A s r : ℝ) (h₁ : A = 3 * s) (h₂ : A = r * s) (h₃ : s ≠ 0) : r = 3 :=
by
  -- Proof omitted
  sorry

end triangle_inradius_l159_159577


namespace evaluate_fff2_l159_159219

def f (x : ℝ) : ℝ :=
  if x > 9 then real.sqrt x else x^2

theorem evaluate_fff2 : f (f (f 2)) = 4 :=
by {
  -- Proof omitted
  sorry
}

end evaluate_fff2_l159_159219


namespace length_of_EF_l159_159583

theorem length_of_EF 
  (AB BC : ℝ)
  (hAB : AB = 10)
  (hBC : BC = 12)
  (DE DF : ℝ)
  (DEF_area : DE * DF / 2 = (AB * BC) / 3)
  (hDEDF : DE = DF) :
  DE * sqrt (2 * (AB * BC) / 3) = 4 * sqrt 10 :=
by
  sorry

end length_of_EF_l159_159583


namespace vovochka_add_correct_pairs_vovochka_min_diff_incorrect_l159_159132

-- Vovochka's addition method: adding two digits without carrying over
def vovochka_add (a b : ℕ) : ℕ := (a % 10 + b % 10) + ((a / 10 % 10 + b / 10 % 10) * 10) + ((a / 100 + b / 100) * 100)

-- Part (a): number of pairs producing correct result with Vovochka’s method
def correct_vovochka_pairs_count : ℕ := 244620

-- Part (b): smallest possible difference when Vovochka’s method is incorrect
def min_diff_vovochka_method : ℕ := 1800

-- Proving the number of correct cases equals 244620
theorem vovochka_add_correct_pairs :
  let count := ∑ (a b : ℕ) in finset.Icc 100 999, if vovochka_add a b = a + b then 1 else 0
  in count = correct_vovochka_pairs_count := sorry

-- Proving the smallest possible difference when Vovochka’s method is incorrect
theorem vovochka_min_diff_incorrect :
  let min_diff := min (∑ (a b : ℕ) in finset.Icc 100 999, if vovochka_add a b ≠ a + b then nat.abs (vovochka_add a b - (a + b)) else ⊤)
  in min_diff = min_diff_vovochka_method := sorry

end vovochka_add_correct_pairs_vovochka_min_diff_incorrect_l159_159132


namespace correct_system_of_equations_l159_159928

-- Define the variables x and y as integers
variables (x y : ℤ)

-- Define the conditions
def condition1 := x = y + 1
def condition2 := 10 * x + y = 10 * y + x + 9

-- Define the system of equations according to the conditions
def system_of_equations := (condition1, condition2)

-- The statement asserting the correctness of the system based on the conditions
theorem correct_system_of_equations : 
  system_of_equations = (x = y + 1, 10 * x + y = 10 * y + x + 9) :=
by
  sorry

end correct_system_of_equations_l159_159928


namespace equalChargesAtFour_agencyADecisionWhenTen_l159_159828

-- Define the conditions as constants
def fullPrice : ℕ := 240
def agencyADiscount : ℕ := 50
def agencyBDiscount : ℕ := 60

-- Define the total charge function for both agencies
def totalChargeAgencyA (students: ℕ) : ℕ :=
  fullPrice * students * agencyADiscount / 100 + fullPrice

def totalChargeAgencyB (students: ℕ) : ℕ :=
  fullPrice * (students + 1) * agencyBDiscount / 100

-- Define the equivalence when the number of students is 4
theorem equalChargesAtFour : totalChargeAgencyA 4 = totalChargeAgencyB 4 := by sorry

-- Define the decision when there are 10 students
theorem agencyADecisionWhenTen : totalChargeAgencyA 10 < totalChargeAgencyB 10 := by sorry

end equalChargesAtFour_agencyADecisionWhenTen_l159_159828


namespace slope_of_line_is_neg_four_thirds_l159_159302

noncomputable def line_equation : ℝ × ℝ → Prop :=
  λ p, 3 * p.1 + 2 = -4 * p.2 - 9

theorem slope_of_line_is_neg_four_thirds :
  ∀ (m : ℝ), (∀ p : ℝ × ℝ, line_equation p → p.1 = m * p.2 + (-(11 / 3))) → m = -(4 / 3) :=
by
  sorry

end slope_of_line_is_neg_four_thirds_l159_159302


namespace median_weight_at_20_l159_159794

noncomputable def S_i (i : ℕ) (a : ℝ) : ℝ := a + (i - 1) * (1 / 5)

def turkey_weight (i t : ℕ) (a : ℝ) : ℝ :=
  S_i i a * t + 200 - i

def average_weight_condition (a : ℝ) : Prop :=
  (∑ i in range 100, turkey_weight i a a) / 100 = 150.5

theorem median_weight_at_20 (a : ℝ) (h_a_pos : a > 0) (h_avg : average_weight_condition a) :
  (turkey_weight 50 20 a + turkey_weight 51 20 a) / 2 = 200.5 :=
  sorry

end median_weight_at_20_l159_159794


namespace remainder_div_30_l159_159792

theorem remainder_div_30 : ∀ (a : ℕ), (a = 44 * 432 + 0) → (a % 30 = 18) :=
by
  intro a
  intro h
  rw [h]
  simp
  sorry

end remainder_div_30_l159_159792


namespace find_k_l159_159075

theorem find_k (k : ℝ) (h : (3 : ℝ)^2 - k * (3 : ℝ) - 6 = 0) : k = 1 :=
by
  sorry

end find_k_l159_159075


namespace value_of_m_l159_159510

theorem value_of_m (m : ℕ) : 
  (let A := {1, 2, 3} in
   let B := {1, m} in
   A ∩ B = B) → (m = 2 ∨ m = 3) :=
by
  intro h
  sorry

end value_of_m_l159_159510


namespace count_two_digit_numbers_with_digit_sum_perfect_square_under_25_l159_159972

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def digit_sum (n : ℕ) : ℕ := n / 10 + n % 10

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

theorem count_two_digit_numbers_with_digit_sum_perfect_square_under_25 :
  ∃ count : ℕ,
    (∀ n : ℕ, is_two_digit n → digit_sum n ∈ {1, 4, 9, 16} → n ∈ {10, 40, 31, 22, 13, 90, 81, 72, 63, 54, 45, 36, 27, 18, 97, 88, 79}) ∧
    count = 17 :=
by
  sorry

end count_two_digit_numbers_with_digit_sum_perfect_square_under_25_l159_159972


namespace proof_F_4_f_5_l159_159102

def f (a : ℤ) : ℤ := a - 2

def F (a b : ℤ) : ℤ := a * b + b^2

theorem proof_F_4_f_5 :
  F 4 (f 5) = 21 := by
  sorry

end proof_F_4_f_5_l159_159102


namespace value_of_n_in_arithmetic_sequence_l159_159588

theorem value_of_n_in_arithmetic_sequence :
  (∃ (a d : ℝ) (n : ℕ), a = 1/3 ∧ (a + d) + (a + 4 * d) = 4 ∧ a + (n - 1) * d = 33) →
  n = 50 := sorry

end value_of_n_in_arithmetic_sequence_l159_159588


namespace even_three_digit_numbers_count_l159_159294

theorem even_three_digit_numbers_count :
  let digits := {0, 1, 2, 3, 4}
  let even_pred : ℕ → Prop := λ n, n % 2 = 0
  let three_digit_even_numbers := { n | even_pred n ∧ 100 ≤ n ∧ n < 1000 ∧ 
    ∀ c ∈ (nat.digits 10 n), ∀ d ∈ (nat.digits 10 n), c ≠ d → c ∈ digits ∧ d ∈ digits }
  (finset.card three_digit_even_numbers = 30 → three_digit_even_numbers.card = 30) :=
sorry

end even_three_digit_numbers_count_l159_159294


namespace find_x_l159_159536

-- We define the given condition
def given_condition (x : ℝ) : Prop :=
  (1 / 8) * (2 ^ 36) = 4 ^ x

-- We state the theorem to prove x = 16.5 given the condition
theorem find_x (x : ℝ) (h : given_condition x) : x = 16.5 :=
by
  sorry

end find_x_l159_159536


namespace sqrt_inequality_l159_159842

theorem sqrt_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  sqrt (b^2 - a*c) < sqrt 3 * a :=
by
  have : (a - c) * (a - b) > 0
  sorry

end sqrt_inequality_l159_159842


namespace michael_reaches_one_in_seven_steps_l159_159663

def divide_by_two (n : ℕ) := n / 2

-- Define the function that Michael uses in each step, i.e., dividing by 2 and taking floor.
def michael_sequence : ℕ → ℕ 
| 0        := 128
| (n + 1)  := divide_by_two (michael_sequence n)

-- Define the statement that after 7 steps the sequence will be 1 starting from 128.
theorem michael_reaches_one_in_seven_steps : michael_sequence 7 = 1 :=
sorry

end michael_reaches_one_in_seven_steps_l159_159663


namespace brenda_age_l159_159832

-- Define ages of Addison, Brenda, Carlos, and Janet
variables (A B C J : ℕ)

-- Formalize the conditions from the problem
def condition1 := A = 4 * B
def condition2 := C = 2 * B
def condition3 := A = J

-- State the theorem we aim to prove
theorem brenda_age (A B C J : ℕ) (h1 : condition1 A B)
                                (h2 : condition2 C B)
                                (h3 : condition3 A J) :
  B = J / 4 :=
sorry

end brenda_age_l159_159832


namespace max_n_sum_gt_zero_l159_159944

theorem max_n_sum_gt_zero (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)
  (h_arith_seq : ∀ n, a (n+1) = a n + d)
  (h_sn_max : ∃ n, S n = list.sum (list.map a (list.range n + 1)) ∧ ∀ m, S m ≤ S n)
  (h_ratio_neg : a 11 / a 10 < -1)
  : ∃ (n : ℕ), n = 19 ∧ S n > 0 :=
by
  sorry

end max_n_sum_gt_zero_l159_159944


namespace problem_statement_l159_159803

noncomputable section

def f : ℝ → ℝ := sorry

def f' (x : ℝ) : ℝ := sorry

theorem problem_statement 
  (h1 : ∀ x : ℝ, x ≠ 0 → differentiable_at ℝ f x)
  (h2 : ∀ x : ℝ, x > 0 → x * f' x - f x < 0) : 
  let a := f (2 ^ 0.2) / (2 ^ 0.2)
  let b := f (0.2 ^ 2) / (0.2 ^ 2)
  let c := f (Real.log 2 5) / (Real.log 2 5)
  in c < a ∧ a < b := sorry

end problem_statement_l159_159803


namespace tan_domain_l159_159672

open Set

theorem tan_domain (k : ℤ) : 
  let dom := {x : ℝ | ∀ k : ℤ, x ≠ (5 * Real.pi / 12) + (k * Real.pi / 2)} in
  ∀ x, x ∈ dom ↔ ¬ ∃ k : ℤ, x = (5 * Real.pi / 12) + (k * Real.pi / 2) := 
by 
  sorry

end tan_domain_l159_159672


namespace polynomial_degree_and_terms_l159_159594

theorem polynomial_degree_and_terms (p : Polynomial ℝ) (h : p = Polynomial.C 18 + Polynomial.X * Polynomial.C 2 + Polynomial.C 1 * Polynomial.X ^ 2) : 
  (p.degree = 2) ∧ (p.sum (λ n a, if a ≠ 0 then 1 else 0) = 3) :=
by
  sorry

end polynomial_degree_and_terms_l159_159594


namespace infinite_sum_of_5_inverse_array_l159_159338

theorem infinite_sum_of_5_inverse_array 
  : ∃ (m n : ℕ), nat.coprime m n ∧ (∑' r:ℕ, ∑' c:ℕ, (1 / (5^r.succ) * 1 / (5^c.succ))) = (↑m / ↑n) ∧ (m + n) % 1001 = 41 := 
by 
  sorry

end infinite_sum_of_5_inverse_array_l159_159338


namespace correct_propositions_l159_159838

/- Definitions for each proposition -/

def prop1 : Prop := ∀ (X Y : Type), (X ≠ Y) → correlation X Y
def prop2 : Prop := ∃ (r : ℝ), correlation (circumference r) r
def prop3 : Prop := ∀ (P D : Type), non_deterministic_relationship D P 
def prop4 : Prop := ∃ (scatter_plot : Type), meaningless_regression scatter_plot
def prop5 : Prop := ∀ (X Y : Type), deterministic_relationship (regression_line X Y)

/- Proof Problem Statement -/

theorem correct_propositions :
  ¬ prop1 ∧ ¬ prop2 ∧ prop3 ∧ prop4 ∧ prop5 :=
by
  sorry

end correct_propositions_l159_159838


namespace yellow_yellow_pairs_l159_159397

variable (students_total : ℕ := 150)
variable (blue_students : ℕ := 65)
variable (yellow_students : ℕ := 85)
variable (total_pairs : ℕ := 75)
variable (blue_blue_pairs : ℕ := 30)

theorem yellow_yellow_pairs : 
  (yellow_students - (blue_students - blue_blue_pairs * 2)) / 2 = 40 :=
by 
  -- proof goes here
  sorry

end yellow_yellow_pairs_l159_159397


namespace line_intersects_circle_l159_159114

theorem line_intersects_circle (a b : ℝ) (h : a^2 + b^2 > 1) : 
  let d2 := (1 / (Real.sqrt (a^2 + b^2))) in
  d2 < 1 :=
by {
  let r := 1,
  let d1 := Real.sqrt (a^2 + b^2),
  sorry
}

end line_intersects_circle_l159_159114


namespace relation_between_abc_l159_159538

-- Specify the definitions given in the conditions
def a : ℝ := Real.logBase 2.1 0.6
def b : ℝ := 2.1 ^ 0.6
def c : ℝ := Real.logBase 0.5 0.6

-- State the theorem that captures the relationship (b > c > a)
theorem relation_between_abc (a_def : a = Real.logBase 2.1 0.6) 
                             (b_def : b = 2.1 ^ 0.6) 
                             (c_def : c = Real.logBase 0.5 0.6) :
                             b > c ∧ c > a :=
by {
  sorry  -- Proof goes here
}

end relation_between_abc_l159_159538


namespace scientific_notation_of_number_l159_159692

theorem scientific_notation_of_number :
  ∃ (a : ℝ) (n : ℤ), 0.00000002 = a * 10^n ∧ a = 2 ∧ n = -8 :=
by
  sorry

end scientific_notation_of_number_l159_159692


namespace constant_function_l159_159627

theorem constant_function {f : ℕ → ℕ} (h : ∀ x y : ℕ, x * f y + y * f x = (x + y) * f (x^2 + y^2)) : ∃ c : ℕ, ∀ x, f x = c := 
sorry

end constant_function_l159_159627


namespace value_of_sums_squared_l159_159977

noncomputable def polynomial_equation (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ) : Prop :=
  (X - 2)^8 = a + a_1 * (X - 1) + a_2 * (X - 1)^2 + a_3 * (X - 1)^3 + a_4 * (X - 1)^4 + 
    a_5 * (X - 1)^5 + a_6 * (X - 1)^6 + a_7 * (X - 1)^7 + a_8 * (X - 1)^8

theorem value_of_sums_squared (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ)
  (h : polynomial_equation a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8) :
  (a_2 + a_4 + a_6 + a_8)^2 - (a_1 + a_3 + a_5 + a_7)^2 = -255 :=
sorry

end value_of_sums_squared_l159_159977


namespace fixed_point_of_function_l159_159676

theorem fixed_point_of_function (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : 
  ∃ (x y : ℝ), (x = -1) ∧ (y = -1) ∧ (y = a ^ (x + 1) - 2) := 
by 
  use -1
  use -1
  split
  sorry

end fixed_point_of_function_l159_159676


namespace trig_identity_proof_l159_159752

noncomputable def value_expr : ℝ :=
  (2 * Real.cos (10 * Real.pi / 180) - Real.sin (20 * Real.pi / 180)) / Real.sin (70 * Real.pi / 180)

theorem trig_identity_proof : value_expr = Real.sqrt 3 :=
by
  sorry

end trig_identity_proof_l159_159752


namespace propositions_correct_l159_159220
open Classical

-- Definitions for the propositions
def prop1 (a b c : ℝ → ℝ → ℝ) : Prop := a ⟂ b ∧ b ⟂ c → a ⟂ c
def prop2 (a b c : ℝ → ℝ → ℝ) : Prop := skew_lines a b ∧ skew_lines b c → skew_lines a c
def prop3 (a b c : ℝ → ℝ → ℝ) : Prop := intersects a b ∧ intersects b c → intersects a c
def prop4 (a b c : ℝ → ℝ → ℝ) : Prop := coplanar a b ∧ coplanar b c → coplanar a c
def prop5 (a b c : ℝ → ℝ → ℝ) : Prop := parallel a b ∧ parallel b c → parallel a c

-- Main theorem stating that out of the five propositions, only one is true
theorem propositions_correct (a b c : ℝ → ℝ → ℝ) :
  (¬ prop1 a b c ∧ ¬ prop2 a b c ∧ ¬ prop3 a b c ∧ ¬ prop4 a b c ∧ prop5 a b c) ↔ true :=
begin
  sorry
end

end propositions_correct_l159_159220


namespace range_of_a_l159_159083

noncomputable def f (a x : ℝ) : ℝ := -x^3 + 3 * a^2 * x - 4 * a

theorem range_of_a (a : ℝ) (h : a > 0) : a > Real.sqrt 2 ↔
  ∃ x₁ x₂ x₃ : ℝ, f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 :=
begin
  sorry
end

end range_of_a_l159_159083


namespace find_chord_length_l159_159893

def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

def parametric_line (t : ℝ) : ℝ × ℝ :=
  let x := 2 - t / 2
  let y := -1 + t / 2
  (x, y)

def is_on_line (x y : ℝ) : Prop :=
  x + y - 1 = 0

def distance_from_center_to_line : ℝ :=
  (1 / Real.sqrt 2)

def chord_length (R d : ℝ) : ℝ :=
  2 * Real.sqrt (R^2 - d^2)

theorem find_chord_length : chord_length 2 distance_from_center_to_line = Real.sqrt 14 :=
  sorry

end find_chord_length_l159_159893


namespace prob_not_spade_on_first_draw_is_three_quarters_l159_159999

-- Define number of total cards and number of spades
def total_cards : ℕ := 52
def spades : ℕ := 13

-- Define the event of drawing a spade
def prob_spade : ℚ := spades / total_cards

-- Define the event of not drawing a spade on the first draw
def prob_not_spade_first_draw : ℚ := 1 - prob_spade

-- The theorem statement 
theorem prob_not_spade_on_first_draw_is_three_quarters :
  prob_not_spade_first_draw = 3 / 4 :=
by
  -- This is a placeholder for the proof
  sorry

end prob_not_spade_on_first_draw_is_three_quarters_l159_159999


namespace frog_starts_at_3_l159_159575

def frog_escape_probability : ℕ → ℚ
| 0     := 0
| 14    := 1
| n+1   := if h : n+1 < 14 then 
              (n+1) / 14 * frog_escape_probability n + (1 - (n+1)/14) * frog_escape_probability (n+2)
            else 0

theorem frog_starts_at_3 : frog_escape_probability 3 = 357 / 500 := 
sorry

end frog_starts_at_3_l159_159575


namespace probability_smaller_divides_larger_l159_159460

def number_set : Set ℕ := {1, 2, 3, 6}
def pairs := {(1, 2), (1, 3), (1, 6), (2, 3), (2, 6), (3, 6)}

def successful_pairs := {(1, 2), (1, 3), (1, 6), (2, 6)}

theorem probability_smaller_divides_larger :
  (Set.card successful_pairs).toRational / (Set.card pairs).toRational = 2 / 3 := by  
  sorry

end probability_smaller_divides_larger_l159_159460


namespace minimum_value_of_expression_l159_159067

theorem minimum_value_of_expression (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) :
  (⟦(a^2 + b^2) / (c * d)⟧^4 + ⟦(b^2 + c^2) / (a * d)⟧^4 + ⟦(c^2 + d^2) / (a * b)⟧^4 + ⟦(d^2 + a^2) / (b * c)⟧^4) = 64 :=
sorry

end minimum_value_of_expression_l159_159067


namespace taxi_fare_l159_159569

theorem taxi_fare (x : ℕ) (h : x > 3) : 
  (let base_fare := 8 in
  let additional_fare_per_km := 1.6 in
  let y := base_fare + additional_fare_per_km * (x - 3)
  in y = 1.6 * x + 3.2) :=
by {
  sorry -- Proof goes here
}

end taxi_fare_l159_159569


namespace sale_price_lower_by_2_5_percent_l159_159290

open Real

theorem sale_price_lower_by_2_5_percent (x : ℝ) : 
  let increased_price := 1.30 * x
  let sale_price := 0.75 * increased_price 
  sale_price = 0.975 * x :=
by
  let increased_price := 1.30 * x
  let sale_price := 0.75 * increased_price 
  show sale_price = 0.975 * x
  sorry

end sale_price_lower_by_2_5_percent_l159_159290


namespace sum_of_reciprocals_or_disjoint_sum_l159_159626

open Set
open Nat

theorem sum_of_reciprocals_or_disjoint_sum (S : Set ℕ)
  (h : ∀ n, n ∈ S → (n > 0)) : 
  (∃ (F G : Finset ℕ), F ≠ G ∧ (∀ x, x ∈ F → x ∈ S) ∧ (∀ x, x ∈ G → x ∈ S) ∧ (∑ x in F, (1 : ℚ) / x) = (∑ x in G, (1 : ℚ) / x)) 
  ∨ 
  (∃ (r : ℚ), 0 < r ∧ r < 1 ∧ (∀ F : Finset ℕ, (∀ x, x ∈ F → x ∈ S) → (∑ x in F, (1 : ℚ) / x) ≠ r)) :=
sorry

end sum_of_reciprocals_or_disjoint_sum_l159_159626


namespace correct_function_l159_159931

noncomputable def is_even (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = f x

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
∀ x y, 0 < x → x < y → f x < f y

noncomputable def exists_neg_value (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∃ x ∈ I, f x < 0

noncomputable def candidate_functions := {
  λ (x : ℝ), x^2 - 3,
  λ (x : ℝ), 2^x + 2^(-x),
  λ (x : ℝ), Real.log (abs x) / Real.log 2,
  λ (x : ℝ), x - 1/x
}

theorem correct_function (f : ℝ → ℝ) (I : set ℝ) :
  f ∈ candidate_functions →
  is_even f →
  is_monotonically_increasing f →
  exists_neg_value f I →
  f = λ (x : ℝ), x^2 - 3 ∨ f = λ (x : ℝ), Real.log (abs x) / Real.log 2 := by
  sorry

end correct_function_l159_159931


namespace distance_BF_l159_159922

-- Given the focus F of the parabola y^2 = 4x
def focus_of_parabola : (ℝ × ℝ) := (1, 0)

-- Points A and B lie on the parabola y^2 = 4x
def point_A (x y : ℝ) := y^2 = 4 * x
def point_B (x y : ℝ) := y^2 = 4 * x

-- The line through F intersects the parabola at points A and B, and |AF| = 2
def distance_AF : ℝ := 2

-- Prove that |BF| = 2
theorem distance_BF : ∀ (A B F : ℝ × ℝ), 
  A = (1, F.2) → 
  B = (1, -F.2) → 
  F = (1, 0) → 
  |A.1 - F.1| + |A.2 - F.2| = distance_AF → 
  |B.1 - F.1| + |B.2 - F.2| = 2 :=
by
  intros A B F hA hB hF d_AF
  sorry

end distance_BF_l159_159922


namespace find_y_l159_159985

theorem find_y (y : ℕ) (hy : (1/2)^25 * (1/81)^12.5 = 1/(y^25)) : y = 18 :=
sorry

end find_y_l159_159985


namespace form_polygon_number_of_sides_perimeter_relation_l159_159513

-- Given the definitions of the problem
variables (A B C D E F O : Point)
-- Assume these points form triangles
variable (triangleABC : Triangle A B C)
variable (triangleDEF : Triangle D E F)
-- Assume X is any point within triangleABC and Y is any point within triangleDEF
variable (X : Point)
variable (Y : Point)
-- Assume points form a parallelogram OXYZ
variable (O : Point)
variable (Z : Point)

-- Problem (a)
theorem form_polygon : 
  (∃ X ∈ triangleABC, ∃ Y ∈ triangleDEF, IsParallelogram O X Y Z) → 
  is_polygon (Set { Z | ∃ X ∈ triangleABC, ∃ Y ∈ triangleDEF }):
sorry

-- Problem (b)
theorem number_of_sides : 
  (∃ X ∈ triangleABC, ∃ Y ∈ triangleDEF, IsParallelogram O X Y Z) →
  (polygon_sides (Set { Z | ∃ X ∈ triangleABC, ∃ Y ∈ triangleDEF }) = 6):
sorry

-- Problem (c)
theorem perimeter_relation : 
  (∃ X ∈ triangleABC, ∃ Y ∈ triangleDEF, IsParallelogram O X Y Z) →
  (polygon_perimeter (Set { Z | ∃ X ∈ triangleABC, ∃ Y ∈ triangleDEF }) = 
   perimeter triangleABC + perimeter triangleDEF):
sorry

end form_polygon_number_of_sides_perimeter_relation_l159_159513


namespace polygon_not_divisible_into_100_triangles_l159_159824

theorem polygon_not_divisible_into_100_triangles
  (P : Type)
  [polygon P]
  (h1 : divisible_into_rectangles P 100)
  (h2 : ¬ divisible_into_rectangles P 99) :
  ¬ divisible_into_triangles P 100 :=
sorry

end polygon_not_divisible_into_100_triangles_l159_159824


namespace compare_a_b_l159_159620

variable (x y : ℝ)

def a : ℝ := (x + y) / (1 + x + y)
def b : ℝ := (x / (1 + x)) + (y / (1 + y))

theorem compare_a_b (hx : x > 0) (hy : y > 0) : a x y < b x y := 
by 
  -- Proof steps are intentionally omitted, use sorry as a placeholder
  sorry

end compare_a_b_l159_159620


namespace n_mod_9_l159_159406

theorem n_mod_9 :
  let n := 2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888
  in n % 9 = 6 := 
by
  sorry

end n_mod_9_l159_159406


namespace number_of_baskets_l159_159284

-- Define the conditions
def total_peaches : Nat := 10
def red_peaches_per_basket : Nat := 4
def green_peaches_per_basket : Nat := 6
def peaches_per_basket : Nat := red_peaches_per_basket + green_peaches_per_basket

-- The goal is to prove that the number of baskets is 1 given the conditions

theorem number_of_baskets (h1 : total_peaches = 10)
                           (h2 : peaches_per_basket = red_peaches_per_basket + green_peaches_per_basket)
                           (h3 : red_peaches_per_basket = 4)
                           (h4 : green_peaches_per_basket = 6) : 
                           total_peaches / peaches_per_basket = 1 := by
                            sorry

end number_of_baskets_l159_159284


namespace no_player_can_win_l159_159744

-- Define the sets of coordinates and the initial state of the board
def Board : Type := (ℤ × ℤ)

-- Define the set of coordinates the game is being played on
def valid_coordinates : Set Board := {p | p.1 ∈ {-3, -2, -1, 0, 1, 2, 3} ∧ p.2 ∈ {-3, -2, -1, 0, 1, 2, 3}}

-- Define rules for the first player's move
def first_player_moves (pos : Board) : Set Board :=
  if pos ∈ valid_coordinates then {pos, (pos.2, -pos.1), (-pos.1, -pos.2), (-pos.2, pos.1)}
  else ∅

-- Define rules for the second player's move
def second_player_moves (pos : Board) : Set Board :=
  if pos ∈ valid_coordinates then {pos, (-pos.1, pos.2), (-pos.1, -pos.2), (pos.1, -pos.2)}
  else ∅

-- Statement: Prove that no player can reach a situation where only one token remains at position (0,1)
theorem no_player_can_win : ¬(∃ player_moves : Board → Set Board,
  ∀ b ∈ valid_coordinates, let tokens_remaining := valid_coordinates \ (player_moves b) in
  tokens_remaining = {(0, 1)}):
sorry

end no_player_can_win_l159_159744


namespace part1_part2_l159_159496

-- Define the function f(x)
def f (x : ℝ) : ℝ := (sqrt 3 * real.cos x + real.sin x) * real.abs (real.sin x)

theorem part1 (k : ℤ) (x : ℝ) (h1 : 2 * k * real.pi ≤ x ∧ x ≤ real.pi + 2 * k * real.pi) :
  f x = real.sin (2 * x - real.pi / 6) + 1/2 :=
sorry

theorem part2 (x : ℝ) (a : ℝ) (h2 : -real.pi / 2 ≤ x ∧ x ≤ real.pi / 2)
    (h3 : f x + real.log a / real.log 4 < 0) :
  0 < a ∧ a < 1 / 8 :=
sorry

end part1_part2_l159_159496


namespace eq_op_l159_159420

-- Define the operation ⊕
def op (x y : ℝ) : ℝ := x^3 + 2 * x - y

-- State the theorem to be proven
theorem eq_op (k : ℝ) : op k (op k k) = k := sorry

end eq_op_l159_159420


namespace gcd_lcm_sum_l159_159304

variable (a b : ℕ)

-- Definition for gcd
def gcdOf (a b : ℕ) : ℕ := Nat.gcd a b

-- Definition for lcm
def lcmOf (a b : ℕ) : ℕ := Nat.lcm a b

-- Statement of the problem
theorem gcd_lcm_sum (h1 : a = 8) (h2 : b = 12) : gcdOf a b + lcmOf a b = 28 := by
  sorry

end gcd_lcm_sum_l159_159304


namespace factors_of_48_multiple_of_6_l159_159099

theorem factors_of_48_multiple_of_6 : 
  (nat.factors 48).count (λ x, ∃ a b : ℕ, x = (2 ^ a) * (3 ^ b) ∧ a ≥ 1 ∧ b ≥ 1) = 4 :=
sorry

end factors_of_48_multiple_of_6_l159_159099


namespace vovochka_correct_sum_combinations_l159_159149

theorem vovochka_correct_sum_combinations : 
  let digit_pairs := finset.filter (λ p : ℕ × ℕ, (p.fst + p.snd) < 10) ((finset.range 10).product (finset.range 10))
  let no_carry_combinations := finset.card digit_pairs
  no_carry_combinations ^ 3 * 81 = 244620 := 
by
  sorry

end vovochka_correct_sum_combinations_l159_159149


namespace omega_correct_and_decreasing_intervals_l159_159498

noncomputable def f (ω x : ℝ) : ℝ :=
  sin(ω * x) * (cos(ω * x) - sqrt 3 * sin(ω * x)) + sqrt 3 / 2

theorem omega_correct_and_decreasing_intervals :
  (∀ ω > 0, ∀ x, (f ω x = sin (2 * ω * x + π / 3)) ↔ (ω = 2)) ∧
  (∀ x k : ℤ, (f 2 x).monotone_decreasing_on (set.Icc ((k:ℝ) * π / 2 + π / 24) ((k:ℝ) * π / 2 + 7 * π / 24))) :=
by
  sorry

end omega_correct_and_decreasing_intervals_l159_159498


namespace smallest_constant_l159_159440

theorem smallest_constant (D : ℝ) :
  (∀ (x y : ℝ), x^2 + 2*y^2 + 5 ≥ D*(2*x + 3*y) + 4) → D ≤ Real.sqrt (8 / 17) :=
by
  intros
  sorry

end smallest_constant_l159_159440


namespace locus_of_M_is_two_lines_l159_159059

open Real

-- Define a point A
variable (A : Point)

-- Define a line l
variable (l : Line)

-- Define reflection of A over l
noncomputable def reflect (A : Point) (l : Line) : Point := 
    sorry -- Reflection logic (to be defined)

-- Define a point B on the line l
variable (B : Point)
axiom B_on_l : B ∈ l

-- Define a predicate for equilateral triangle
def is_equilateral (A B M : Point) : Prop :=
    sorry -- Definition of equilateral triangle

-- Define locus for M such that ∠BAM = 60°
def locus_points (A l B : Point) : Set Point :=
    { M | is_equilateral A B M }

-- The target statement
theorem locus_of_M_is_two_lines :
  ∀ (A l : Point) (B : Point), B ∈ l → 
  (∃ A' : Point, A' = reflect A l) →
  locus_points A l B = { M | (∃ A' : Point, A' = reflect A l) ∧ (∠l A' M = 60° ∨ ∠l A' M = -60°) } :=
begin
  sorry -- Proof goes here
end

end locus_of_M_is_two_lines_l159_159059


namespace price_of_turban_l159_159791

-- Define the conditions
def one_year_salary (T : ℝ) : ℝ := 90 + T
def nine_month_salary (T : ℝ) : ℝ := 65 + T
def expected_nine_month_salary (T : ℝ) : ℝ := (3/4) * (90 + T)

-- The main statement to prove
theorem price_of_turban (T : ℝ) (h : expected_nine_month_salary T = nine_month_salary T) : T = 10 :=
by
  sorry

end price_of_turban_l159_159791


namespace coefficient_of_1_over_x_squared_l159_159990

theorem coefficient_of_1_over_x_squared 
  (n : ℕ) 
  (h : (nat.choose n 2) = (nat.choose n 6)) : 
  (nat.choose 8 5) = 56 := by
  sorry

end coefficient_of_1_over_x_squared_l159_159990


namespace cone_volume_l159_159079

theorem cone_volume (θ : ℝ) (A : ℝ) (l r h V : ℝ) 
  (hθ : θ = 2 * π / 3) 
  (hA : A = 3 * π)
  (h_area_eq : 1 / 2 * θ * l^2 = A)
  (h_l : l = 3)
  (h_arc_length_eq : 2 * π * r = θ * l / 2)
  (hr : r = 1)
  (h_height_eq : h = sqrt (l^2 - r^2))
  (h_height_val : h = 2 * sqrt 2)
  (hV : V = 1 / 3 * π * r^2 * h)
  : V = 2 * sqrt 2 * π / 3 :=
sorry

end cone_volume_l159_159079


namespace sin_sq_minus_cos_sq_l159_159978

theorem sin_sq_minus_cos_sq (α : ℝ) 
  (h1 : cos ((π / 2) - α) = - 1 / 3) : (sin α) ^ 2 - (cos α) ^ 2 = - 7 / 9 :=
sorry

end sin_sq_minus_cos_sq_l159_159978


namespace find_f_two_thirds_l159_159472

noncomputable def f : ℝ → ℝ 
-- f is defined but no explicit definition is given here as the proof is omitted

theorem find_f_two_thirds (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, f(x + y) = f(x) + f(y) + 6 * x * y)
  (h2 : f(-1) * f(1) ≥ 9) :
  f (2 / 3) = 4 / 3 := 
sorry

end find_f_two_thirds_l159_159472


namespace eccentricity_of_ellipse_l159_159196

variables {a b c : ℝ} (h1 : a > b > 0) (h2 : b^2 = a^2 - c^2)
  (h3 : 2 * c = a) (h4 : a = 1 / 2 * c)

theorem eccentricity_of_ellipse :
  let e := c / a in
  e = 1 / 2 := by
sorry

end eccentricity_of_ellipse_l159_159196


namespace side_lengths_of_triangle_l159_159956

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (Real.cos x + m) / (Real.cos x + 2)

theorem side_lengths_of_triangle (m : ℝ) (a b c : ℝ) 
  (h1 : f m a > 0) 
  (h2 : f m b > 0) 
  (h3 : f m c > 0) 
  (h4 : f m a + f m b > f m c)
  (h5 : f m a + f m c > f m b)
  (h6 : f m b + f m c > f m a) :
  m ∈ Set.Ioo (7/5 : ℝ) 5 :=
sorry

end side_lengths_of_triangle_l159_159956


namespace smallest_angle_equilateral_triangle_l159_159604

-- Definitions corresponding to the conditions
structure EquilateralTriangle :=
(vertices : Fin 3 → ℝ × ℝ)
(equilateral : ∀ i j, dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1))

def point_on_line_segment (p1 p2 : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
((1 - t) * p1.1 + t * p2.1, (1 - t) * p1.2 + t * p2.2)

-- Given an equilateral triangle ABC with vertices A, B, C,
-- and points D on AB, E on AC, D1 on BC, and E1 on BC,
-- such that AB = DB + BD_1 and AC = CE + CE_1,
-- prove the smallest angle between DE_1 and ED_1 is 60 degrees.

theorem smallest_angle_equilateral_triangle
  (ABC : EquilateralTriangle)
  (A B C D E D₁ E₁ : ℝ × ℝ)
  (on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = point_on_line_segment A B t)
  (on_AC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = point_on_line_segment A C t)
  (on_BC : ∃ t₁ t₂ : ℝ, 0 ≤ t₁ ∧ t₁ ≤ 1 ∧ D₁ = point_on_line_segment B C t₁ ∧
                         0 ≤ t₂ ∧ t₂ ≤ 1 ∧ E₁ = point_on_line_segment B C t₂)
  (AB_property : dist A B = dist D B + dist B D₁)
  (AC_property : dist A C = dist E C + dist C E₁) :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 60 ∧ θ = 60 :=
sorry

end smallest_angle_equilateral_triangle_l159_159604


namespace area_of_triangle_l159_159674

theorem area_of_triangle (ABC : Triangle) (acute_triangle : Acute ABC)
  (p : ℝ) (R : ℝ) (perimeter_altitudes : 2 * p = perimeter (feet_of_altitudes ABC)) 
  (circumradius : circumradius ABC = R) : area ABC = R * p :=
sorry

end area_of_triangle_l159_159674


namespace triple_f_of_two_l159_159216

def f (x : ℝ) : ℝ :=
  if x > 9 then real.sqrt x else x^2

theorem triple_f_of_two : f (f (f 2)) = 4 := 
by 
  sorry

end triple_f_of_two_l159_159216


namespace ratio_Y_share_to_total_profit_l159_159377

/--
Given that the total profit is Rs. 1000, and the difference between their profit shares is Rs. 200,
prove that the ratio of Y's share to the total profit is 2:5.
-/
theorem ratio_Y_share_to_total_profit 
  (total_profit : ℝ) (diff_share : ℝ) (h1 : total_profit = 1000)
  (h2 : ∃ a b, a / (a + b) * total_profit - b / (a + b) * total_profit = diff_share) :
  Y_share / total_profit = 2 / 5 :=
begin
  sorry
end

end ratio_Y_share_to_total_profit_l159_159377


namespace at_least_two_even_degree_l159_159164

noncomputable theory

-- Define the scenario as an undirected graph with 44 vertices
def studentGraph : Type := simple_graph (fin 44)

-- State the theorem
theorem at_least_two_even_degree (G : studentGraph) : 
  ∃ (v1 v2 : fin 44), G.degree v1 % 2 = 0 ∧ G.degree v2 % 2 = 0 := 
sorry

end at_least_two_even_degree_l159_159164


namespace valid_paths_equals_55_l159_159895

noncomputable def num_valid_paths : ℕ :=
  -- Define the dimensions of the grid
  let cols := 7
  let rows := 4
  -- Define the blocked paths
  let blocked1 := [(3, 3), (3, 2)]
  let blocked2 := [(5, 4), (5, 3)]
  55

theorem valid_paths_equals_55 :
  num_valid_paths = 55 :=
begin
  sorry
end

end valid_paths_equals_55_l159_159895


namespace sum_of_integer_solutions_l159_159450

theorem sum_of_integer_solutions : 
  let f (x : ℤ) := x^4 - 13 * x^2 + 36 
  in (finset.sum (finset.filter (λ x, f x = 0) (finset.Icc (-3) 3)) id) = 0 :=
begin
  sorry
end

end sum_of_integer_solutions_l159_159450


namespace min_value_func3_l159_159840

noncomputable def func1 (x : ℝ) : ℝ := x + 4 / x
noncomputable def func2 (x : ℝ) : ℝ := Real.sin x + 4 / Real.sin x
noncomputable def func3 (x : ℝ) : ℝ := Real.exp x + 4 * Real.exp(-x)
noncomputable def func4 (x : ℝ) : ℝ := Real.log x / Real.log 3 + 4 / (Real.log x / Real.log 3)

theorem min_value_func3 : ∃ x : ℝ, func3 x = 4 :=
sorry

end min_value_func3_l159_159840


namespace magic_square_d_e_sum_l159_159568

theorem magic_square_d_e_sum 
  (S : ℕ)
  (a b c d e : ℕ)
  (h1 : S = 45 + d)
  (h2 : S = 51 + e) :
  d + e = 57 :=
by
  sorry

end magic_square_d_e_sum_l159_159568


namespace log2_75_in_terms_of_a_b_l159_159908

variable (a b : ℝ)

-- Given conditions
def log2_9_eq_a (h : ℝ) : Prop := log 9 / log 2 = h
def log2_5_eq_b (h : ℝ) : Prop := log 5 / log 2 = h

-- Statement to be proven
theorem log2_75_in_terms_of_a_b (h1 : log2_9_eq_a a) (h2 : log2_5_eq_b b) : (log 75 / log 2) = (1/2) * a + 2 * b := sorry

end log2_75_in_terms_of_a_b_l159_159908


namespace cesaro_sum_51_l159_159454

theorem cesaro_sum_51 (B : Fin 50 → ℝ) (h : (∑ i in Finset.range 50, (∑ j in Finset.range (i + 1), B j)) / 50 = 500) :
  (2 + ∑ i in Finset.range 51, (2 + ∑ j in Finset.range i, B j)) / 51 = 492 := by
  sorry

end cesaro_sum_51_l159_159454


namespace cos_4theta_value_l159_159112

noncomputable def cos_four_theta (theta : ℝ) : ℝ :=
  2 * (2 * (cos theta)^2 - 1)^2 - 1

theorem cos_4theta_value (theta : ℝ) (h : ∑' n : ℕ, (cos theta)^(2 * n) = 9 / 2) :
  cos_four_theta theta = -31 / 81 :=
sorry

end cos_4theta_value_l159_159112


namespace carol_total_points_l159_159800

-- Define the conditions for Carol's game points.
def first_round_points := 17
def second_round_points := 6
def last_round_points := -16

-- Prove that the total points at the end of the game are 7.
theorem carol_total_points : first_round_points + second_round_points + last_round_points = 7 := by
  sorry

end carol_total_points_l159_159800


namespace integer_solution_sum_eq_zero_l159_159448

theorem integer_solution_sum_eq_zero : 
  (∑ x in (Finset.filter (λ x : ℤ, x^4 - 13 * x^2 + 36 = 0) (Finset.Icc -3 3)), x) = 0 := by
sorry

end integer_solution_sum_eq_zero_l159_159448


namespace blue_pigment_percentage_l159_159360

theorem blue_pigment_percentage (M G : ℝ) 
  (total_weight : M + G = 10)
  (maroon_blue_percentage : 0.5 * M)
  (maroon_red_weight : 0.5 * M = 2.5)
  (green_blue_percentage : 0.3 * G) :
  (0.5 * M + 0.3 * G) / 10 * 100 = 40 :=
by
  -- Calculation of M and G follows
  -- Situation where specific solution steps are used to fill in the variables
  have M_value : M = 5 := by
    linarith [maroon_red_weight]
  have G_value : G = 5 := by
    rw [M_value] at total_weight
    linarith [total_weight]
  sorry

end blue_pigment_percentage_l159_159360


namespace conic_sections_problem_l159_159841

-- Definitions and conditions
def fixed_points (A B : Point) : Prop := A ≠ B
def non_zero_constant (K : ℝ) : Prop := K ≠ 0
def hyperbola_condition (P A B : Point) (K : ℝ) : Prop := |P-A| - |P-B| = K ∧ K ≠ |A-B|

def quadratic_eq_roots : (2 * x^2 - 5 * x + 2 = 0) → {r1 r2 : ℝ // r1 = 1/2 ∧ r2 = 2}

def hyperbola_focus := (x^2 / 25 - y^2 / 9 = 1) → foci = (√34, 0)
def ellipse_focus := (x^2 / 35 + y^2 = 1) → foci = (√34, 0)

def parabola_property (chord : Line) (focus : Point) (directrix : Line) : Prop :=
  ∀(A B P : Point), midpoint(P, A, B) ∧ chord_through_focus(chord, focus) ∧ tangent_to_directrix(directrix)

-- Statement
theorem conic_sections_problem (A B P : Point) (K : ℝ)
  (h1 : fixed_points A B)
  (h2 : non_zero_constant K)
  (h3 : hyperbola_condition P A B K)
  (h4 : quadratic_eq_roots)
  (h5 : hyperbola_focus)
  (h6 : ellipse_focus)
  (h7 : parabola_property chom focus directrix) : ∀ prop, prop = 2 ∨ prop = 3 ∨ prop = 4 := 
  sorry

end conic_sections_problem_l159_159841


namespace measure_angle_R_l159_159579

-- Given conditions
variables {P Q R : Type}
variable {x : ℝ} -- x represents the measure of angles P and Q

-- Setting up the given conditions
def isosceles_triangle (P Q R : Type) (x : ℝ) : Prop :=
  x + x + (x + 40) = 180

-- Statement we need to prove
theorem measure_angle_R (P Q R : Type) (x : ℝ) (h : isosceles_triangle P Q R x) : ∃ r : ℝ, r = 86.67 :=
by {
  sorry
}

end measure_angle_R_l159_159579


namespace necessary_not_sufficient_cond_l159_159551

theorem necessary_not_sufficient_cond (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y < 4 → xy < 4) ∧ ¬(xy < 4 → x + y < 4) :=
  by
    sorry

end necessary_not_sufficient_cond_l159_159551


namespace bags_of_long_balloons_l159_159185

theorem bags_of_long_balloons (total_round_bags : ℕ) (balloons_per_round_bag : ℕ) (burst_round_balloons : ℕ) 
(total_balloons_left : ℕ) (balloons_per_long_bag : ℕ) : 
    total_round_bags = 5 → 
    balloons_per_round_bag = 20 → 
    burst_round_balloons = 5 → 
    total_balloons_left = 215 → 
    balloons_per_long_bag = 30 → 
    ∃ long_bags : ℕ, long_bags = 4 :=
by 
  assume h1 h2 h3 h4 h5
  have total_round_balloons_before := total_round_bags * balloons_per_round_bag,
  have remaining_round_balloons := total_round_balloons_before - burst_round_balloons,
  have remaining_long_balloons := total_balloons_left - remaining_round_balloons,
  have long_bags := remaining_long_balloons / balloons_per_long_bag,
  exact ⟨long_bags, rfl⟩,
sorry

end bags_of_long_balloons_l159_159185


namespace fraction_product_l159_159853

theorem fraction_product :
  (7/4) * (8/14) * (20/12) * (15/25) * (21/14) * (12/18) * (28/14) * (30/50) = 3/5 :=
by
  -- Step 1: Simplify each fraction
  have h1 : (8/14) = (4/7), by norm_num
  have h2 : (20/12) = (5/3), by norm_num
  have h3 : (15/25) = (3/5), by norm_num
  have h4 : (21/14) = (3/2), by norm_num
  have h5 : (12/18) = (2/3), by norm_num
  have h6 : (28/14) = 2, by norm_num
  have h7 : (30/50) = (3/5), by norm_num
  
  -- Step 2: Apply the simplifications and compute the product
  calc
    (7/4) * (8/14) * (20/12) * (15/25) * (21/14) * (12/18) * (28/14) * (30/50)
    = (7/4) * (4/7) * (5/3) * (3/5) * (3/2) * (2/3) * 2 * (3/5) : by rw [h1, h2, h3, h4, h5, h6, h7]
    ... = 1 * 1 * 1 * (3/5) : by norm_num
    ... = 3/5 : by norm_num

end fraction_product_l159_159853


namespace distinct_values_of_b_l159_159456

theorem distinct_values_of_b : ∃ b_list : List ℝ, b_list.length = 8 ∧ ∀ b ∈ b_list, ∃ p q : ℤ, p + q = b ∧ p * q = 8 * b :=
by
  sorry

end distinct_values_of_b_l159_159456


namespace problem_l159_159918

variables {n : ℕ} (x : Fin n → ℝ)

theorem problem (hnn : ∑ i, x i = n) (hnn0 : (∀ i, 0 ≤ x i)) :
  ∑ i, x i / (1 + (x i)^2) ≤ ∑ i, 1 / (1 + x i) := 
sorry

end problem_l159_159918


namespace provider_assignment_ways_l159_159002

theorem provider_assignment_ways (total_providers : ℕ) (children : ℕ) (h1 : total_providers = 15) (h2 : children = 4) : 
  (Finset.range total_providers).card.factorial / (Finset.range (total_providers - children)).card.factorial = 32760 :=
by
  rw [h1, h2]
  norm_num
  sorry

end provider_assignment_ways_l159_159002


namespace proof_problem_l159_159952

-- Define the function f(x)
def f (x : ℝ) (θ : ℝ) : ℝ := -x^2 + 2 * x * Real.tan θ + 1

-- Define the interval for x
def x_interval (x : ℝ) : Prop := -Real.sqrt 3 ≤ x ∧ x ≤ 1

-- Define the interval for θ
def θ_interval (θ : ℝ) : Prop := -Real.pi / 2 < θ ∧ θ < Real.pi / 2

-- State the theorem
theorem proof_problem :
  (∀ x, x_interval x → f x (-Real.pi / 4) ≤ 2 ∧ f x (-Real.pi / 4) ≥ -2) ∧
  (∀ θ, θ_interval θ → (∀ x1 x2, x_interval x1 → x_interval x2 → 
    (Real.tan θ ≤ -Real.sqrt 3 ∨ Real.tan θ ≥ 1) → 
    (f x1 θ ≤ f x2 θ ∨ f x1 θ ≥ f x2 θ)) →
    θ ∈ (-Real.pi / 2, -Real.pi / 3] ∪ [Real.pi / 4, Real.pi / 2)) :=
begin
  sorry
end

end proof_problem_l159_159952


namespace f_of_g_of_pi_div_two_l159_159964

def f (x : ℝ) : ℝ := x^3 + 2
def g (x : ℝ) : ℝ := 3 * Real.sin x + 2

theorem f_of_g_of_pi_div_two : f (g (Real.pi / 2)) = 127 := by
  sorry

end f_of_g_of_pi_div_two_l159_159964


namespace probability_two_red_two_blue_l159_159355

def total_marbles : ℕ := 20
def red_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def selected_marbles : ℕ := 4

theorem probability_two_red_two_blue :
  (12.choose 2 * 8.choose 2) / (20.choose 4) = 168 / 323 :=
  sorry

end probability_two_red_two_blue_l159_159355


namespace find_value_of_a_l159_159861

-- Define the points P1 and P2
def P1 : ℝ × ℝ := (-2, 3)
def P2 : ℝ × ℝ := (3, 5)

-- Define the direction vector with unknown a and given 4
def directionVector (a : ℝ) : ℝ × ℝ := (a, 4)

-- Define the problem statement: Finding the value of a such that the given conditions are satisfied
theorem find_value_of_a (a : ℝ) : 
  (P2.1 - P1.1, P2.2 - P1.2) = directionVector a -> 
  a = 10 := 
by 
  -- Proof goes here
  intro h1
  unfold directionVector at h1
  cases h1
  rw [sub_eq_add_neg, sub_eq_add_neg] at h1
  sorry

end find_value_of_a_l159_159861


namespace thousand_points_per_side_l159_159571

-- Define the problem statement
def plane := ℝ × ℝ
def points (n : ℕ) := fin n → plane

theorem thousand_points_per_side (pts : points 2000) : ∃ l : ℝ → plane → Prop, (∀ p : plane, ∃ (L R : finset plane), L.card = 1000 ∧ R.card = 1000 ∧ (∀ q ∈ L, l 0 q) ∧ (∀ q ∈ R, ¬ l 0 q)) :=
sorry

end thousand_points_per_side_l159_159571


namespace triangle_inequality_l159_159478

open EuclideanGeometry

variable {A B C O P Q R : Point}
variable {a b c : ℝ}

def isInTriangle (O : Point) (A B C : Point) : Prop := 
  -- Here we define a predicate for being inside the triangle
  sorry

def intersectAtPoints (A O : Point) (B C : Point) : Point := 
  -- Here we define how intersection at a specific point is obtained
  sorry

axiom triangle_sides {ABC : Triangle} (hABC : a > b > c) : 
  -- We define that the sides of the triangle meet the given condition
  sorry

axiom intersection_points {A B C O : Point} :
  intersectAtPoints A O B = P ∧ intersectAtPoints B O C = Q ∧ intersectAtPoints C O A = R

axiom distances {O P Q R : Point} :
  dist O P + dist O Q + dist O R < a

theorem triangle_inequality (h_in_triangle : isInTriangle O A B C) :
  dist O P + dist O Q + dist O R < a :=
begin
  -- proof will be provided
  sorry
end

end triangle_inequality_l159_159478


namespace october_birth_percentage_l159_159679

theorem october_birth_percentage 
  (jan feb mar apr may jun jul aug sep oct nov dec total : ℕ) 
  (h_total : total = 100)
  (h_jan : jan = 2) (h_feb : feb = 4) (h_mar : mar = 8) (h_apr : apr = 5) 
  (h_may : may = 4) (h_jun : jun = 9) (h_jul : jul = 7) (h_aug : aug = 12) 
  (h_sep : sep = 8) (h_oct : oct = 6) (h_nov : nov = 5) (h_dec : dec = 4) : 
  (oct : ℕ) * 100 / total = 6 := 
by
  sorry

end october_birth_percentage_l159_159679


namespace trailing_zeroes_of_15_fact_in_base_25_l159_159532

-- Definition of factorial and counting trailing zeroes in a given base
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def trailing_zeroes_in_base (n : ℕ) (b : ℕ) : ℕ :=
  let rec count (n: ℕ) (b: ℕ) (p: ℕ) (acc: ℕ) : ℕ :=
    if n < b then acc else count (n / b) b p (acc + 1)
  count n b b 0

theorem trailing_zeroes_of_15_fact_in_base_25 :
  trailing_zeroes_in_base (factorial 15) 25 = 1 :=
by
  sorry

end trailing_zeroes_of_15_fact_in_base_25_l159_159532


namespace least_number_to_subtract_997_l159_159753

theorem least_number_to_subtract_997 (x : ℕ) (h : x = 997) 
  : ∃ y : ℕ, ∀ m (h₁ : m = (997 - y)), 
    m % 5 = 3 ∧ m % 9 = 3 ∧ m % 11 = 3 ∧ y = 4 :=
by
  -- Proof omitted
  sorry

end least_number_to_subtract_997_l159_159753


namespace minimum_value_of_expression_l159_159095

theorem minimum_value_of_expression (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0)
  (h1 : ∃ (x y : ℝ), x^2 + y^2 + 2*a*x + a^2 - 9 = 0)
  (h2 : ∃ (x y : ℝ), x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0)
  (h3 : sqrt(a^2 + 4*b^2) = 4) :
  (4 / a^2 + 1 / b^2) = 1 :=
by
  sorry

end minimum_value_of_expression_l159_159095


namespace range_f_l159_159029

variable {α : Type} [LinearOrder α] [Field α]

noncomputable def f (x : α) : α := (3 * x + 8) / (x - 4)

theorem range_f : set.range f = { y : α | y ≠ 3 } :=
by
  sorry  -- Proof is omitted, according to the instructions

end range_f_l159_159029


namespace mean_and_median_eq_l159_159453

theorem mean_and_median_eq {x : ℝ} : 
  (9 * x = 675 + 2 * x) ∧ (list.median [30, 60, 70, 85, x, x, 100, 110, 220] = x) → 
  x = 96.42857 :=
by sorry

end mean_and_median_eq_l159_159453


namespace vovochka_correct_sum_combinations_l159_159153

theorem vovochka_correct_sum_combinations : 
  let digit_pairs := finset.filter (λ p : ℕ × ℕ, (p.fst + p.snd) < 10) ((finset.range 10).product (finset.range 10))
  let no_carry_combinations := finset.card digit_pairs
  no_carry_combinations ^ 3 * 81 = 244620 := 
by
  sorry

end vovochka_correct_sum_combinations_l159_159153


namespace calc_root_difference_l159_159403

theorem calc_root_difference :
  ((81: ℝ)^(1/4) + (32: ℝ)^(1/5) - (49: ℝ)^(1/2)) = -2 :=
by
  have h1 : (81: ℝ)^(1/4) = 3 := by sorry
  have h2 : (32: ℝ)^(1/5) = 2 := by sorry
  have h3 : (49: ℝ)^(1/2) = 7 := by sorry
  rw [h1, h2, h3]
  norm_num

end calc_root_difference_l159_159403
