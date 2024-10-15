import Mathlib

namespace NUMINAMATH_CALUDE_max_value_inequality_max_value_achievable_l2137_213740

theorem max_value_inequality (x y : ℝ) :
  (x + 3 * y + 5) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 35 :=
by sorry

theorem max_value_achievable :
  ∃ (x y : ℝ), (x + 3 * y + 5) / Real.sqrt (x^2 + y^2 + 4) = Real.sqrt 35 :=
by sorry

end NUMINAMATH_CALUDE_max_value_inequality_max_value_achievable_l2137_213740


namespace NUMINAMATH_CALUDE_three_two_digit_multiples_l2137_213718

theorem three_two_digit_multiples :
  (∃! (s : Finset ℕ), 
    (∀ x ∈ s, x > 0 ∧ 
      (∃! (m : Finset ℕ), 
        (∀ y ∈ m, 10 ≤ y ∧ y < 100 ∧ ∃ k, y = k * x) ∧ 
        m.card = 3)) ∧ 
    s.card = 9) := by sorry

end NUMINAMATH_CALUDE_three_two_digit_multiples_l2137_213718


namespace NUMINAMATH_CALUDE_solution_interval_l2137_213790

theorem solution_interval (c : ℝ) : (c / 4 ≤ 3 + c ∧ 3 + c < -3 * (1 + c)) ↔ c ∈ Set.Ici (-4) ∩ Set.Iio (-3/2) := by
  sorry

end NUMINAMATH_CALUDE_solution_interval_l2137_213790


namespace NUMINAMATH_CALUDE_complex_sum_fourth_powers_l2137_213730

theorem complex_sum_fourth_powers : 
  let z₁ : ℂ := (-1 + Complex.I * Real.sqrt 7) / 2
  let z₂ : ℂ := (-1 - Complex.I * Real.sqrt 7) / 2
  z₁^4 + z₂^4 = 1 := by sorry

end NUMINAMATH_CALUDE_complex_sum_fourth_powers_l2137_213730


namespace NUMINAMATH_CALUDE_find_x_l2137_213719

theorem find_x : ∃ x : ℝ, (0.5 * x = 0.25 * 1500 - 30) ∧ (x = 690) := by
  sorry

end NUMINAMATH_CALUDE_find_x_l2137_213719


namespace NUMINAMATH_CALUDE_even_function_inequality_l2137_213752

theorem even_function_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h_even : ∀ x, f (-x) = f x)
  (h_increasing : ∀ x y, x < y → x < 0 → f x < f y) :
  f (-2) ≥ f (a^2 - 4*a + 6) := by
  sorry

end NUMINAMATH_CALUDE_even_function_inequality_l2137_213752


namespace NUMINAMATH_CALUDE_skyler_song_difference_l2137_213759

def composer_songs (total_songs hit_songs top_100_extra : ℕ) : Prop :=
  let top_100_songs := hit_songs + top_100_extra
  let unreleased_songs := total_songs - (hit_songs + top_100_songs)
  hit_songs - unreleased_songs = 5

theorem skyler_song_difference :
  composer_songs 80 25 10 :=
by
  sorry

end NUMINAMATH_CALUDE_skyler_song_difference_l2137_213759


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_root_l2137_213734

theorem quadratic_equation_unique_root (b c : ℝ) :
  (∀ x : ℝ, 3 * x^2 + b * x + c = 0 ↔ x = -4) →
  b = 24 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_root_l2137_213734


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2137_213707

theorem arithmetic_calculation : 2 + 3 * 4 - 5 + 6 - (1 + 2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2137_213707


namespace NUMINAMATH_CALUDE_max_islands_is_36_l2137_213729

/-- Represents an archipelago with islands and bridges -/
structure Archipelago where
  N : Nat
  bridges : Fin N → Fin N → Bool

/-- The number of islands is at least 7 -/
def atLeastSevenIslands (a : Archipelago) : Prop :=
  a.N ≥ 7

/-- Any two islands are connected by at most one bridge -/
def atMostOneBridge (a : Archipelago) : Prop :=
  ∀ i j : Fin a.N, i ≠ j → (a.bridges i j = true → a.bridges j i = false)

/-- No more than 5 bridges lead from each island -/
def atMostFiveBridges (a : Archipelago) : Prop :=
  ∀ i : Fin a.N, (Finset.filter (fun j => a.bridges i j) (Finset.univ : Finset (Fin a.N))).card ≤ 5

/-- Among any 7 islands, there are always two that are connected by a bridge -/
def twoConnectedInSeven (a : Archipelago) : Prop :=
  ∀ s : Finset (Fin a.N), s.card = 7 →
    ∃ i j : Fin a.N, i ∈ s ∧ j ∈ s ∧ i ≠ j ∧ a.bridges i j

/-- The maximum number of islands satisfying the conditions is 36 -/
theorem max_islands_is_36 (a : Archipelago) :
    atLeastSevenIslands a →
    atMostOneBridge a →
    atMostFiveBridges a →
    twoConnectedInSeven a →
    a.N ≤ 36 := by
  sorry

end NUMINAMATH_CALUDE_max_islands_is_36_l2137_213729


namespace NUMINAMATH_CALUDE_numbers_with_seven_from_1_to_800_l2137_213721

def contains_seven (n : ℕ) : Bool :=
  sorry

def count_numbers_with_seven (lower : ℕ) (upper : ℕ) : ℕ :=
  sorry

theorem numbers_with_seven_from_1_to_800 :
  count_numbers_with_seven 1 800 = 62 :=
sorry

end NUMINAMATH_CALUDE_numbers_with_seven_from_1_to_800_l2137_213721


namespace NUMINAMATH_CALUDE_square_of_two_minus_sqrt_three_l2137_213717

theorem square_of_two_minus_sqrt_three : (2 - Real.sqrt 3) ^ 2 = 7 - 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_of_two_minus_sqrt_three_l2137_213717


namespace NUMINAMATH_CALUDE_outfits_count_l2137_213736

/-- The number of shirts available -/
def num_shirts : ℕ := 8

/-- The number of ties available -/
def num_ties : ℕ := 5

/-- The number of pants available -/
def num_pants : ℕ := 4

/-- The number of jackets available -/
def num_jackets : ℕ := 2

/-- The number of tie options (including not wearing a tie) -/
def tie_options : ℕ := num_ties + 1

/-- The number of jacket options (including not wearing a jacket) -/
def jacket_options : ℕ := num_jackets + 1

/-- The total number of possible outfits -/
def total_outfits : ℕ := num_shirts * num_pants * tie_options * jacket_options

theorem outfits_count : total_outfits = 576 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l2137_213736


namespace NUMINAMATH_CALUDE_inequalities_proof_l2137_213778

theorem inequalities_proof (a b : ℝ) : 
  (a^2 + b^2 ≥ (a + b)^2 / 2) ∧ (a^2 + b^2 ≥ 2*(a - b - 1)) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l2137_213778


namespace NUMINAMATH_CALUDE_cristinas_leftover_croissants_l2137_213732

/-- Represents the types of croissants --/
inductive CroissantType
  | Chocolate
  | Plain

/-- Represents a guest's dietary restriction --/
inductive DietaryRestriction
  | Vegan
  | ChocolateAllergy
  | NoRestriction

/-- Represents the croissant distribution problem --/
structure CroissantDistribution where
  total_croissants : ℕ
  chocolate_croissants : ℕ
  plain_croissants : ℕ
  guests : List DietaryRestriction
  more_chocolate : chocolate_croissants > plain_croissants

/-- The specific instance of the problem --/
def cristinas_distribution : CroissantDistribution := {
  total_croissants := 17,
  chocolate_croissants := 12,
  plain_croissants := 5,
  guests := [DietaryRestriction.Vegan, DietaryRestriction.Vegan, DietaryRestriction.Vegan,
             DietaryRestriction.ChocolateAllergy, DietaryRestriction.ChocolateAllergy,
             DietaryRestriction.NoRestriction, DietaryRestriction.NoRestriction],
  more_chocolate := by sorry
}

/-- Function to calculate the number of leftover croissants --/
def leftover_croissants (d : CroissantDistribution) : ℕ := 
  d.total_croissants - d.guests.length

/-- Theorem stating that the number of leftover croissants in Cristina's distribution is 3 --/
theorem cristinas_leftover_croissants :
  leftover_croissants cristinas_distribution = 3 := by sorry

end NUMINAMATH_CALUDE_cristinas_leftover_croissants_l2137_213732


namespace NUMINAMATH_CALUDE_complex_root_property_l2137_213755

variable (a b c d e m n : ℝ)
variable (z : ℂ)

theorem complex_root_property :
  (z = m + n * Complex.I) →
  (a * z^4 + Complex.I * b * z^2 + c * z^2 + Complex.I * d * z + e = 0) →
  (a * (-m + n * Complex.I)^4 + Complex.I * b * (-m + n * Complex.I)^2 + 
   c * (-m + n * Complex.I)^2 + Complex.I * d * (-m + n * Complex.I) + e = 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_root_property_l2137_213755


namespace NUMINAMATH_CALUDE_touching_x_axis_with_max_value_l2137_213763

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x

-- Define the derivative of f
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

-- Theorem statement
theorem touching_x_axis_with_max_value (a b m : ℝ) :
  m ≠ 0 →
  f a b m = 0 →
  f' a b m = 0 →
  (∀ x, f a b x ≤ 1/2) →
  (∃ x, f a b x = 1/2) →
  m = 3/2 := by
sorry

end NUMINAMATH_CALUDE_touching_x_axis_with_max_value_l2137_213763


namespace NUMINAMATH_CALUDE_equation_solutions_l2137_213739

theorem equation_solutions :
  (∃ x : ℚ, 6 * x - 4 = 3 * x + 2 ∧ x = 2) ∧
  (∃ x : ℚ, x / 4 - 3 / 5 = (x + 1) / 2 ∧ x = -22 / 5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2137_213739


namespace NUMINAMATH_CALUDE_sum_of_roots_equation_l2137_213700

theorem sum_of_roots_equation (x : ℝ) : 
  ((-15 * x) / (x^2 - 1) = (3 * x) / (x + 1) - 9 / (x - 1)) →
  (∃ y : ℝ, (-15 * y) / (y^2 - 1) = (3 * y) / (y + 1) - 9 / (y - 1) ∧ y ≠ x) →
  x + y = -1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_equation_l2137_213700


namespace NUMINAMATH_CALUDE_smallest_multiple_17_6_more_than_multiple_73_l2137_213785

theorem smallest_multiple_17_6_more_than_multiple_73 : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), n = 17 * k) ∧ 
  (∃ (m : ℕ), n = 73 * m + 6) ∧
  (∀ (x : ℕ), x > 0 → (∃ (k : ℕ), x = 17 * k) → (∃ (m : ℕ), x = 73 * m + 6) → x ≥ n) ∧
  n = 663 := by
sorry

end NUMINAMATH_CALUDE_smallest_multiple_17_6_more_than_multiple_73_l2137_213785


namespace NUMINAMATH_CALUDE_mixture_quantity_is_three_l2137_213715

/-- Represents the cost and quantity of a tea and coffee mixture --/
structure TeaCoffeeMixture where
  june_cost : ℝ  -- Cost per pound of both tea and coffee in June
  july_tea_cost : ℝ  -- Cost per pound of tea in July
  july_coffee_cost : ℝ  -- Cost per pound of coffee in July
  mixture_cost : ℝ  -- Total cost of the mixture in July
  mixture_quantity : ℝ  -- Quantity of the mixture in pounds

/-- Theorem stating the quantity of mixture bought given the conditions --/
theorem mixture_quantity_is_three (m : TeaCoffeeMixture) : 
  m.june_cost > 0 ∧ 
  m.july_coffee_cost = 2 * m.june_cost ∧ 
  m.july_tea_cost = 0.3 * m.june_cost ∧ 
  m.july_tea_cost = 0.3 ∧ 
  m.mixture_cost = 3.45 ∧ 
  m.mixture_quantity = m.mixture_cost / ((m.july_tea_cost + m.july_coffee_cost) / 2) →
  m.mixture_quantity = 3 := by
  sorry

#check mixture_quantity_is_three

end NUMINAMATH_CALUDE_mixture_quantity_is_three_l2137_213715


namespace NUMINAMATH_CALUDE_negative_difference_l2137_213709

theorem negative_difference (m n : ℝ) : -(m - n) = -m + n := by
  sorry

end NUMINAMATH_CALUDE_negative_difference_l2137_213709


namespace NUMINAMATH_CALUDE_notebook_count_l2137_213753

theorem notebook_count : ∃ (n : ℕ), n > 0 ∧ n + (n + 50) = 110 ∧ n = 30 := by sorry

end NUMINAMATH_CALUDE_notebook_count_l2137_213753


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2137_213745

-- Define the inequality
def inequality (x : ℝ) : Prop := -x^2 - 5*x + 6 ≥ 0

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | -6 ≤ x ∧ x ≤ 1}

-- Theorem statement
theorem inequality_solution_set : 
  {x : ℝ | inequality x} = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2137_213745


namespace NUMINAMATH_CALUDE_paint_fraction_in_15_minutes_l2137_213716

/-- The fraction of a wall that can be painted by two people working together,
    given their individual rates and a specific time. -/
def fractionPainted (rate1 rate2 time : ℚ) : ℚ :=
  (rate1 + rate2) * time

theorem paint_fraction_in_15_minutes :
  let heidi_rate : ℚ := 1 / 60
  let zoe_rate : ℚ := 1 / 90
  let time : ℚ := 15
  fractionPainted heidi_rate zoe_rate time = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_paint_fraction_in_15_minutes_l2137_213716


namespace NUMINAMATH_CALUDE_sum_remainder_mod_11_l2137_213758

theorem sum_remainder_mod_11 : 
  (123456 + 123457 + 123458 + 123459 + 123460 + 123461) % 11 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_11_l2137_213758


namespace NUMINAMATH_CALUDE_sequence_properties_l2137_213766

def sequence_a (n : ℕ) : ℝ := sorry

def sum_S (n : ℕ) : ℝ := sorry

axiom a_def (n : ℕ) : n ≠ 0 → sequence_a n = 2 * sum_S n - 1

def sequence_b (n : ℕ) : ℝ := (2 * n + 1) * sequence_a n

def sum_T (n : ℕ) : ℝ := sorry

theorem sequence_properties :
  (∀ n : ℕ, n ≠ 0 → sequence_a n = (-1)^(n-1)) ∧
  (∀ n : ℕ, sum_T n = 1 - (n + 1) * (-1)^n) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l2137_213766


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l2137_213708

/-- The volume of a sphere inscribed in a right circular cone -/
theorem inscribed_sphere_volume (d : ℝ) (h : d = 24) :
  let r := d * (Real.sqrt 2 - 1) / 2
  (4 / 3) * Real.pi * r^3 = (4 / 3) * Real.pi * (24 * (Real.sqrt 2 - 1))^3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l2137_213708


namespace NUMINAMATH_CALUDE_count_pairs_eq_fib_l2137_213762

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Count of pairs (α,S) with specific properties -/
def count_pairs (n : ℕ) : ℕ :=
  sorry

theorem count_pairs_eq_fib (n : ℕ) :
  count_pairs n = n! * fib (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_eq_fib_l2137_213762


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l2137_213782

theorem coefficient_x_cubed_in_expansion : 
  let expression := (fun x : ℝ => (x^2 + 1)^2 * (x - 1)^6)
  ∃ (a b c d e f g : ℝ), 
    (∀ x, expression x = a*x^9 + b*x^8 + c*x^7 + d*x^6 + e*x^5 + f*x^4 + (-32)*x^3 + g) :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l2137_213782


namespace NUMINAMATH_CALUDE_sequence_properties_l2137_213712

/-- A sequence where the sum of the first n terms is S_n = 2n^2 + 3n -/
def S (n : ℕ) : ℕ := 2 * n^2 + 3 * n

/-- The nth term of the sequence -/
def a (n : ℕ) : ℕ := 4 * n + 1

theorem sequence_properties :
  (∀ n : ℕ, S (n + 1) - S n = a (n + 1)) ∧
  (∀ n : ℕ, a (n + 1) - a n = 4) ∧
  (a 10 = 41) := by sorry

end NUMINAMATH_CALUDE_sequence_properties_l2137_213712


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2137_213726

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 - 3*a*x + 9 < 0) ↔ (a < -2 ∨ a > 2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2137_213726


namespace NUMINAMATH_CALUDE_baseball_football_fans_l2137_213704

theorem baseball_football_fans (total : ℕ) (baseball_only : ℕ) (football_only : ℕ) (neither : ℕ) 
  (h1 : total = 16)
  (h2 : baseball_only = 2)
  (h3 : football_only = 3)
  (h4 : neither = 6) :
  total - baseball_only - football_only - neither = 5 := by
sorry

end NUMINAMATH_CALUDE_baseball_football_fans_l2137_213704


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l2137_213731

/-- Represents the sampling methods --/
inductive SamplingMethod
  | StratifiedSampling
  | LotteryMethod
  | SystematicSampling
  | RandomNumberTableMethod

/-- Represents a grade with classes and students --/
structure Grade where
  num_classes : Nat
  students_per_class : Nat
  selected_number : Nat

/-- Determines the sampling method based on the grade structure --/
def determineSamplingMethod (g : Grade) : SamplingMethod :=
  if g.num_classes > 0 ∧ 
     g.students_per_class > 0 ∧ 
     g.selected_number > 0 ∧ 
     g.selected_number ≤ g.students_per_class
  then SamplingMethod.SystematicSampling
  else SamplingMethod.StratifiedSampling  -- Default case, not actually used in this problem

theorem systematic_sampling_theorem (g : Grade) :
  g.num_classes = 12 ∧ 
  g.students_per_class = 50 ∧ 
  g.selected_number = 14 →
  determineSamplingMethod g = SamplingMethod.SystematicSampling :=
by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l2137_213731


namespace NUMINAMATH_CALUDE_solution_set_l2137_213711

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : f 2 = 7)
variable (h2 : ∀ x : ℝ, deriv f x < 3)

-- Define the theorem
theorem solution_set (x : ℝ) : 
  f x < 3 * x + 1 ↔ x > 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_l2137_213711


namespace NUMINAMATH_CALUDE_only_consecutive_primes_fifth_power_difference_prime_l2137_213779

theorem only_consecutive_primes_fifth_power_difference_prime :
  ∀ p q : ℕ,
    Prime p → Prime q → p > q →
    Prime (p^5 - q^5) →
    p = 3 ∧ q = 2 :=
by sorry

end NUMINAMATH_CALUDE_only_consecutive_primes_fifth_power_difference_prime_l2137_213779


namespace NUMINAMATH_CALUDE_board_length_proof_l2137_213750

/-- Given a board cut into two pieces, where one piece is twice the length of the other
    and the shorter piece is 23 inches long, the total length of the board is 69 inches. -/
theorem board_length_proof (shorter_piece longer_piece total_length : ℕ) :
  shorter_piece = 23 →
  longer_piece = 2 * shorter_piece →
  total_length = shorter_piece + longer_piece →
  total_length = 69 := by
  sorry

#check board_length_proof

end NUMINAMATH_CALUDE_board_length_proof_l2137_213750


namespace NUMINAMATH_CALUDE_rectangle_rotation_path_length_l2137_213776

/-- The length of the path traveled by point A of a rectangle ABCD when rotated as described -/
theorem rectangle_rotation_path_length (AB CD BC AD : ℝ) (h1 : AB = 4) (h2 : CD = 4) (h3 : BC = 8) (h4 : AD = 8) :
  let diagonal := Real.sqrt (AB ^ 2 + AD ^ 2)
  let first_rotation_arc := (π / 2) * diagonal
  let second_rotation_arc := (π / 2) * diagonal
  first_rotation_arc + second_rotation_arc = 4 * Real.sqrt 5 * π :=
by sorry

end NUMINAMATH_CALUDE_rectangle_rotation_path_length_l2137_213776


namespace NUMINAMATH_CALUDE_existence_of_r_l2137_213783

/-- Two infinite sequences of rational numbers -/
def s : ℕ → ℚ := sorry
def t : ℕ → ℚ := sorry

/-- Neither sequence is constant -/
axiom not_constant_s : ∃ i j, s i ≠ s j
axiom not_constant_t : ∃ i j, t i ≠ t j

/-- For any integers i and j, (sᵢ - sⱼ)(tᵢ - tⱼ) is an integer -/
axiom product_is_integer : ∀ i j : ℕ, ∃ k : ℤ, (s i - s j) * (t i - t j) = k

/-- The main theorem to be proved -/
theorem existence_of_r : ∃ r : ℚ, 
  (∀ i j : ℕ, ∃ m : ℤ, (s i - s j) * r = m) ∧ 
  (∀ i j : ℕ, ∃ n : ℤ, (t i - t j) / r = n) :=
sorry

end NUMINAMATH_CALUDE_existence_of_r_l2137_213783


namespace NUMINAMATH_CALUDE_adeline_work_hours_l2137_213793

/-- Adeline's work schedule and earnings problem -/
theorem adeline_work_hours
  (hourly_rate : ℕ)
  (days_per_week : ℕ)
  (total_earnings : ℕ)
  (total_weeks : ℕ)
  (h1 : hourly_rate = 12)
  (h2 : days_per_week = 5)
  (h3 : total_earnings = 3780)
  (h4 : total_weeks = 7) :
  total_earnings / (total_weeks * days_per_week * hourly_rate) = 9 :=
by sorry

end NUMINAMATH_CALUDE_adeline_work_hours_l2137_213793


namespace NUMINAMATH_CALUDE_deepak_age_l2137_213724

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    determine Deepak's present age -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →   -- Age ratio condition
  rahul_age + 6 = 38 →                     -- Rahul's future age condition
  deepak_age = 24 := by                    -- Deepak's present age to prove
sorry

end NUMINAMATH_CALUDE_deepak_age_l2137_213724


namespace NUMINAMATH_CALUDE_megan_markers_l2137_213756

theorem megan_markers (x : ℕ) : x + 109 = 326 → x = 217 := by
  sorry

end NUMINAMATH_CALUDE_megan_markers_l2137_213756


namespace NUMINAMATH_CALUDE_sqrt_fraction_simplification_l2137_213702

theorem sqrt_fraction_simplification :
  (Real.sqrt 6) / (Real.sqrt 10) = (Real.sqrt 15) / 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_simplification_l2137_213702


namespace NUMINAMATH_CALUDE_rent_increase_percentage_l2137_213713

theorem rent_increase_percentage (last_year_earnings : ℝ) : 
  let last_year_rent := 0.20 * last_year_earnings
  let this_year_earnings := 1.20 * last_year_earnings
  let this_year_rent := 0.30 * this_year_earnings
  (this_year_rent / last_year_rent) * 100 = 180 := by
sorry

end NUMINAMATH_CALUDE_rent_increase_percentage_l2137_213713


namespace NUMINAMATH_CALUDE_constant_term_expansion_l2137_213786

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The coefficient of the kth term in the expansion -/
def coeff (n k : ℕ) : ℕ := binomial n k * 2^k

theorem constant_term_expansion (n : ℕ) 
  (h : coeff n 4 / coeff n 2 = 56 / 3) : 
  coeff n 2 = 180 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l2137_213786


namespace NUMINAMATH_CALUDE_triangle_perpendicular_bisector_distance_l2137_213784

/-- Given a triangle ABC with sides a, b, c where b > c, if a line HK perpendicular to BC
    divides the triangle into two equal areas, then the distance CK (from C to the foot of
    the perpendicular) is equal to (1/2) * sqrt(a^2 + b^2 - c^2). -/
theorem triangle_perpendicular_bisector_distance
  (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_order : b > c) (h_equal_areas : ∃ (k : ℝ), k > 0 ∧ k < b ∧ k * (a * k / (2 * b)) = a * b / 4) :
  ∃ (k : ℝ), k = (1/2) * Real.sqrt (a^2 + b^2 - c^2) ∧
              k > 0 ∧ k < b ∧ k * (a * k / (2 * b)) = a * b / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perpendicular_bisector_distance_l2137_213784


namespace NUMINAMATH_CALUDE_four_circles_theorem_l2137_213792

/-- Represents a square piece of paper -/
structure Paper :=
  (side : ℝ)
  (is_square : side > 0)

/-- Represents the state of the paper after folding and corner removal -/
structure FoldedPaper :=
  (original : Paper)
  (num_folds : ℕ)
  (corner_removed : Bool)

/-- Calculates the number of layers after folding -/
def num_layers (fp : FoldedPaper) : ℕ :=
  2^(fp.num_folds)

/-- Represents the hole pattern after unfolding -/
structure HolePattern :=
  (num_circles : ℕ)

/-- Function to determine the hole pattern after unfolding -/
def unfold_pattern (fp : FoldedPaper) : HolePattern :=
  { num_circles := if fp.corner_removed then (num_layers fp) / 4 else 0 }

theorem four_circles_theorem (p : Paper) :
  let fp := FoldedPaper.mk p 4 true
  (unfold_pattern fp).num_circles = 4 := by sorry

end NUMINAMATH_CALUDE_four_circles_theorem_l2137_213792


namespace NUMINAMATH_CALUDE_makeup_exam_probability_l2137_213799

/-- Given a class with a total number of students and a number of students who need to take a makeup exam,
    calculate the probability of a student participating in the makeup exam. -/
theorem makeup_exam_probability (total_students : ℕ) (makeup_students : ℕ) 
    (h1 : total_students = 42) (h2 : makeup_students = 3) :
    (makeup_students : ℚ) / total_students = 1 / 14 := by
  sorry

#check makeup_exam_probability

end NUMINAMATH_CALUDE_makeup_exam_probability_l2137_213799


namespace NUMINAMATH_CALUDE_shelf_rearrangement_l2137_213722

theorem shelf_rearrangement (n : ℕ) (k : ℕ) (m : ℕ) : 
  n = 8 → k = 2 → m = 4 →
  (Nat.choose n k) * ((m + 1) * m + Nat.choose (m + 1) k) = 840 := by
  sorry

end NUMINAMATH_CALUDE_shelf_rearrangement_l2137_213722


namespace NUMINAMATH_CALUDE_salt_solution_volume_salt_solution_volume_proof_l2137_213768

/-- Given a solution with initial salt concentration of 10% that becomes 8% salt
    after adding 16 gallons of water, prove the initial volume is 64 gallons. -/
theorem salt_solution_volume : ℝ → Prop :=
  fun initial_volume =>
    let initial_salt_concentration : ℝ := 0.10
    let final_salt_concentration : ℝ := 0.08
    let added_water : ℝ := 16
    let final_volume : ℝ := initial_volume + added_water
    initial_salt_concentration * initial_volume =
      final_salt_concentration * final_volume →
    initial_volume = 64

/-- Proof of the salt_solution_volume theorem -/
theorem salt_solution_volume_proof : salt_solution_volume 64 := by
  sorry

end NUMINAMATH_CALUDE_salt_solution_volume_salt_solution_volume_proof_l2137_213768


namespace NUMINAMATH_CALUDE_vector_parallelism_l2137_213791

theorem vector_parallelism (m : ℚ) : 
  let a : Fin 2 → ℚ := ![(-1), 2]
  let b : Fin 2 → ℚ := ![m, 1]
  (∃ (k : ℚ), k ≠ 0 ∧ (a + 2 • b) = k • (2 • a - b)) → m = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallelism_l2137_213791


namespace NUMINAMATH_CALUDE_function_ranges_l2137_213703

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - 2
def g (a : ℝ) (x : ℝ) : ℝ := -x^2 + x + a

-- State the theorem
theorem function_ranges :
  ∀ a : ℝ,
  (f a (-1) = 0) →
  (∀ x₁ ∈ Set.Icc (1/4 : ℝ) 1, ∃ x₂ ∈ Set.Icc 1 2, g a x₁ > f a x₂ + 3) →
  (Set.range (f a) = Set.Ici (-9/4 : ℝ)) ∧
  (a ∈ Set.Ioi 1) :=
by sorry

end NUMINAMATH_CALUDE_function_ranges_l2137_213703


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2137_213789

theorem rectangle_perimeter (a b : ℝ) (h : a * b > 2 * a + 2 * b) : 2 * a + 2 * b > 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2137_213789


namespace NUMINAMATH_CALUDE_tan_alpha_value_l2137_213742

theorem tan_alpha_value (α : Real) (h : Real.tan (α - π/4) = 1/6) : Real.tan α = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l2137_213742


namespace NUMINAMATH_CALUDE_champion_sequences_l2137_213798

/-- The number of letters in CHAMPION -/
def num_letters : ℕ := 8

/-- The number of letters in each sequence -/
def sequence_length : ℕ := 5

/-- The number of letters available for the last position (excluding N) -/
def last_position_options : ℕ := 6

/-- The number of positions to fill after fixing the first and last -/
def middle_positions : ℕ := sequence_length - 2

/-- The number of letters available for the middle positions -/
def middle_options : ℕ := num_letters - 2

theorem champion_sequences :
  (middle_options.factorial / (middle_options - middle_positions).factorial) * last_position_options = 720 := by
  sorry

end NUMINAMATH_CALUDE_champion_sequences_l2137_213798


namespace NUMINAMATH_CALUDE_larger_number_proof_l2137_213796

theorem larger_number_proof (L S : ℕ) (h1 : L > S) (h2 : L - S = 1335) (h3 : L = 6 * S + 15) :
  L = 1599 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2137_213796


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_1000_l2137_213773

theorem consecutive_integers_sum_1000 :
  ∃ (m k : ℕ), m > 0 ∧ k ≥ 0 ∧ (k + 1) * (2 * m + k) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_1000_l2137_213773


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l2137_213754

theorem quadratic_expression_value (x : ℝ) : x = 2 → x^2 - 3*x + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l2137_213754


namespace NUMINAMATH_CALUDE_josh_marbles_l2137_213738

/-- Theorem: If Josh had 16 marbles and lost 7 marbles, he now has 9 marbles. -/
theorem josh_marbles (initial : ℕ) (lost : ℕ) (final : ℕ) 
  (h1 : initial = 16) 
  (h2 : lost = 7) 
  (h3 : final = initial - lost) : 
  final = 9 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_l2137_213738


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_area_l2137_213794

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - x*y + x - 4*y = 12

/-- Point A is on the y-axis and satisfies the ellipse equation -/
def point_A : ℝ × ℝ := sorry

/-- Point C is on the y-axis and satisfies the ellipse equation -/
def point_C : ℝ × ℝ := sorry

/-- Point B is on the x-axis and satisfies the ellipse equation -/
def point_B : ℝ × ℝ := sorry

/-- Point D is on the x-axis and satisfies the ellipse equation -/
def point_D : ℝ × ℝ := sorry

/-- The area of the inscribed quadrilateral ABCD -/
def area_ABCD : ℝ := sorry

theorem inscribed_quadrilateral_area :
  ellipse_equation point_A.1 point_A.2 ∧
  ellipse_equation point_B.1 point_B.2 ∧
  ellipse_equation point_C.1 point_C.2 ∧
  ellipse_equation point_D.1 point_D.2 ∧
  point_A.1 = 0 ∧ point_C.1 = 0 ∧
  point_B.2 = 0 ∧ point_D.2 = 0 →
  area_ABCD = 28 := by sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_area_l2137_213794


namespace NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l2137_213741

theorem power_product_equals_sum_of_exponents (a : ℝ) : a^4 * a = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l2137_213741


namespace NUMINAMATH_CALUDE_equation_solution_l2137_213748

theorem equation_solution : ∃ y : ℚ, (8 + 3.2 * y = 0.8 * y + 40) ∧ (y = 40 / 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2137_213748


namespace NUMINAMATH_CALUDE_rationality_of_x_not_necessarily_rational_l2137_213706

theorem rationality_of_x (x : ℝ) :
  (∃ (a b : ℚ), x^7 = a ∧ x^12 = b) →
  ∃ (q : ℚ), x = q :=
sorry

theorem not_necessarily_rational (x : ℝ) :
  (∃ (a b : ℚ), x^9 = a ∧ x^12 = b) →
  ¬(∀ x : ℝ, ∃ (q : ℚ), x = q) :=
sorry

end NUMINAMATH_CALUDE_rationality_of_x_not_necessarily_rational_l2137_213706


namespace NUMINAMATH_CALUDE_intersection_and_complement_when_a_is_3_range_of_a_when_M_subset_N_l2137_213723

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 3*x - 18 ≤ 0}
def N (a : ℝ) : Set ℝ := {x | 1 - a ≤ x ∧ x ≤ 2*a + 1}

-- Theorem for part 1
theorem intersection_and_complement_when_a_is_3 :
  (M ∩ N 3 = {x | -2 ≤ x ∧ x ≤ 6}) ∧
  (Set.univ \ N 3 = {x | x < -2 ∨ x > 7}) := by
  sorry

-- Theorem for part 2
theorem range_of_a_when_M_subset_N :
  (∀ a : ℝ, M ⊆ N a) ↔ (∀ a : ℝ, a ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_complement_when_a_is_3_range_of_a_when_M_subset_N_l2137_213723


namespace NUMINAMATH_CALUDE_planes_parallel_iff_skew_lines_parallel_l2137_213769

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes and lines
variable (parallel : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the "contained in" relation for lines and planes
variable (contained_in : Line → Plane → Prop)

-- Define the "skew" relation for lines
variable (skew : Line → Line → Prop)

-- Theorem statement
theorem planes_parallel_iff_skew_lines_parallel (α β : Plane) :
  parallel α β ↔
  ∃ (a b : Line),
    skew a b ∧
    contained_in a α ∧
    contained_in b β ∧
    parallel_line_plane a β ∧
    parallel_line_plane b α :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_iff_skew_lines_parallel_l2137_213769


namespace NUMINAMATH_CALUDE_triangle_EC_length_l2137_213725

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the points D and E
def D (t : Triangle) : ℝ × ℝ := sorry
def E (t : Triangle) : ℝ × ℝ := sorry

-- Define the length of a segment
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the angle between two segments
def angle (p q r : ℝ × ℝ) : ℝ := sorry

-- Define perpendicularity
def perpendicular (p q r s : ℝ × ℝ) : Prop := sorry

theorem triangle_EC_length (t : Triangle) : 
  angle t.A t.B t.C = π/4 →          -- ∠A = 45°
  length t.B t.C = 10 →              -- BC = 10
  perpendicular (D t) t.B t.A t.C → -- BD ⊥ AC
  perpendicular (E t) t.C t.A t.B → -- CE ⊥ AB
  angle (D t) t.B t.C = 2 * angle (E t) t.C t.B → -- m∠DBC = 2m∠ECB
  length (E t) t.C = 5 * Real.sqrt 6 := by
    sorry

#check triangle_EC_length

end NUMINAMATH_CALUDE_triangle_EC_length_l2137_213725


namespace NUMINAMATH_CALUDE_toothpick_grid_count_l2137_213772

/-- Calculates the number of toothpicks in a grid with missing toothpicks in regular intervals -/
def toothpick_count (length width : ℕ) (row_interval column_interval : ℕ) : ℕ :=
  let vertical_lines := length + 1
  let horizontal_lines := width + 1
  let vertical_missing := (vertical_lines / column_interval) * width
  let horizontal_missing := (horizontal_lines / row_interval) * length
  let vertical_count := vertical_lines * width - vertical_missing
  let horizontal_count := horizontal_lines * length - horizontal_missing
  vertical_count + horizontal_count

/-- The total number of toothpicks in the specified grid -/
theorem toothpick_grid_count :
  toothpick_count 45 25 5 4 = 2304 := by sorry

end NUMINAMATH_CALUDE_toothpick_grid_count_l2137_213772


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l2137_213797

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a,
    b := p.b - 2 * p.a * h,
    c := p.c + p.a * h^2 - p.b * h }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a,
    b := p.b,
    c := p.c + v }

theorem parabola_shift_theorem (x : ℝ) :
  let initial_parabola := Parabola.mk (-2) 0 0
  let shifted_left := shift_horizontal initial_parabola 1
  let final_parabola := shift_vertical shifted_left 3
  final_parabola.a * x^2 + final_parabola.b * x + final_parabola.c = -2 * (x + 1)^2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l2137_213797


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_exists_l2137_213735

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a quadrilateral
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the property of a quadrilateral being circumscribed around a circle
def isCircumscribed (q : Quadrilateral) (c : Circle) : Prop :=
  sorry

-- Define the property of a point being on a circle
def isOnCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  sorry

-- Define homothety between two quadrilaterals
def isHomothetic (q1 q2 : Quadrilateral) : Prop :=
  sorry

-- Theorem statement
theorem inscribed_quadrilateral_exists (ABCD : Quadrilateral) (c : Circle) :
  isCircumscribed ABCD c →
  isOnCircle ABCD.A c →
  isOnCircle ABCD.B c →
  isOnCircle ABCD.C c →
  ∃ (EFGH : Quadrilateral), isCircumscribed EFGH c ∧ isHomothetic ABCD EFGH :=
sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_exists_l2137_213735


namespace NUMINAMATH_CALUDE_logarithm_product_identity_l2137_213767

theorem logarithm_product_identity (x y : ℝ) (hx : x > 0) (hy : y > 0) (hy1 : y ≠ 1) :
  Real.log x ^ 2 / Real.log (y ^ 3) *
  Real.log (y ^ 3) / Real.log (x ^ 4) *
  Real.log (x ^ 4) / Real.log (y ^ 5) *
  Real.log (y ^ 5) / Real.log (x ^ 2) =
  Real.log x / Real.log y := by
  sorry

end NUMINAMATH_CALUDE_logarithm_product_identity_l2137_213767


namespace NUMINAMATH_CALUDE_kevin_distance_after_seven_leaps_l2137_213780

/-- The sum of the first n terms of a geometric series with first term a and common ratio r -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- Kevin's total distance hopped after n leaps -/
def kevinDistance (n : ℕ) : ℚ :=
  geometricSum (1/4) (3/4) n

/-- Theorem: Kevin's total distance after 7 leaps is 14197/16384 -/
theorem kevin_distance_after_seven_leaps :
  kevinDistance 7 = 14197 / 16384 := by
  sorry

end NUMINAMATH_CALUDE_kevin_distance_after_seven_leaps_l2137_213780


namespace NUMINAMATH_CALUDE_marble_selection_ways_l2137_213728

def total_marbles : ℕ := 15
def red_marbles : ℕ := 2
def green_marbles : ℕ := 2
def blue_marbles : ℕ := 2
def marbles_to_choose : ℕ := 5
def special_marbles_to_choose : ℕ := 2

theorem marble_selection_ways :
  (Nat.choose 3 2 * (Nat.choose red_marbles 1 * Nat.choose green_marbles 1 +
   Nat.choose red_marbles 1 * Nat.choose blue_marbles 1 +
   Nat.choose green_marbles 1 * Nat.choose blue_marbles 1) +
   Nat.choose 3 1 * Nat.choose red_marbles 2) *
  Nat.choose (total_marbles - (red_marbles + green_marbles + blue_marbles)) (marbles_to_choose - special_marbles_to_choose) = 3300 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_ways_l2137_213728


namespace NUMINAMATH_CALUDE_container_weight_l2137_213720

/-- Given the weights of different metal bars, calculate the total weight of a container --/
theorem container_weight (copper_weight tin_weight steel_weight : ℝ) 
  (h1 : steel_weight = 2 * tin_weight)
  (h2 : steel_weight = copper_weight + 20)
  (h3 : copper_weight = 90) : 
  20 * steel_weight + 20 * copper_weight + 20 * tin_weight = 5100 := by
  sorry

#check container_weight

end NUMINAMATH_CALUDE_container_weight_l2137_213720


namespace NUMINAMATH_CALUDE_max_chosen_squares_29x29_l2137_213727

/-- The maximum number of squares that can be chosen on an n×n chessboard 
    such that for every selected square, there exists at most one square 
    with both row and column numbers greater than or equal to the selected 
    square's row and column numbers. -/
def max_chosen_squares (n : ℕ) : ℕ :=
  if n % 2 = 0 then 3 * (n / 2) else 3 * (n / 2) + 1

/-- Theorem stating that the maximum number of chosen squares for a 29×29 chessboard is 43. -/
theorem max_chosen_squares_29x29 : max_chosen_squares 29 = 43 := by
  sorry

end NUMINAMATH_CALUDE_max_chosen_squares_29x29_l2137_213727


namespace NUMINAMATH_CALUDE_car_distance_l2137_213733

theorem car_distance (train_speed : ℝ) (car_speed_ratio : ℝ) (time_minutes : ℝ) : 
  train_speed = 120 →
  car_speed_ratio = 2/3 →
  time_minutes = 15 →
  (car_speed_ratio * train_speed) * (time_minutes / 60) = 20 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_l2137_213733


namespace NUMINAMATH_CALUDE_solve_coloring_books_problem_l2137_213744

def coloring_books_problem (initial_stock : ℝ) (coupons_per_book : ℝ) (total_coupons_used : ℕ) : Prop :=
  initial_stock = 40.0 ∧
  coupons_per_book = 4.0 ∧
  total_coupons_used = 80 →
  initial_stock - (total_coupons_used : ℝ) / coupons_per_book = 20

theorem solve_coloring_books_problem :
  ∃ (initial_stock coupons_per_book : ℝ) (total_coupons_used : ℕ),
    coloring_books_problem initial_stock coupons_per_book total_coupons_used :=
by
  sorry

end NUMINAMATH_CALUDE_solve_coloring_books_problem_l2137_213744


namespace NUMINAMATH_CALUDE_max_rented_trucks_l2137_213777

theorem max_rented_trucks (total_trucks : ℕ) (return_rate : ℚ) (min_saturday_trucks : ℕ) :
  total_trucks = 20 →
  return_rate = 1/2 →
  min_saturday_trucks = 10 →
  ∃ (max_rented : ℕ), max_rented ≤ total_trucks ∧
    max_rented * return_rate = total_trucks - min_saturday_trucks ∧
    ∀ (rented : ℕ), rented ≤ total_trucks ∧ 
      rented * return_rate = total_trucks - min_saturday_trucks →
      rented ≤ max_rented :=
by sorry

end NUMINAMATH_CALUDE_max_rented_trucks_l2137_213777


namespace NUMINAMATH_CALUDE_assignPositions_eq_95040_l2137_213747

/-- The number of ways to assign 5 distinct positions to 5 people chosen from a group of 12 people,
    where each person can only hold one position. -/
def assignPositions : ℕ := 12 * 11 * 10 * 9 * 8

/-- Theorem stating that the number of ways to assign the positions is 95,040. -/
theorem assignPositions_eq_95040 : assignPositions = 95040 := by
  sorry

end NUMINAMATH_CALUDE_assignPositions_eq_95040_l2137_213747


namespace NUMINAMATH_CALUDE_calculate_expression_l2137_213761

theorem calculate_expression : (Real.sqrt 3) ^ 0 + 2⁻¹ + Real.sqrt 2 * Real.cos (45 * π / 180) - |-(1/2)| = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2137_213761


namespace NUMINAMATH_CALUDE_custom_mul_theorem_l2137_213710

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := 3 * a - 2 * b^2

/-- Theorem stating that if a * 6 = -3 using the custom multiplication, then a = 23 -/
theorem custom_mul_theorem (a : ℝ) (h : custom_mul a 6 = -3) : a = 23 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_theorem_l2137_213710


namespace NUMINAMATH_CALUDE_janous_problem_l2137_213760

def is_valid_triple (x y z : ℕ+) : Prop :=
  x ∣ (y + 1) ∧ y ∣ (z + 1) ∧ z ∣ (x + 1)

def solution_set : Set (ℕ+ × ℕ+ × ℕ+) :=
  {(1, 1, 1), (1, 2, 1), (1, 1, 2), (1, 3, 2), (2, 1, 1), (2, 1, 3), (3, 1, 2), 
   (2, 2, 2), (2, 2, 3), (2, 3, 2), (3, 2, 2)}

theorem janous_problem :
  ∀ x y z : ℕ+, is_valid_triple x y z ↔ (x, y, z) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_janous_problem_l2137_213760


namespace NUMINAMATH_CALUDE_max_draws_for_all_pairs_l2137_213787

/-- Represents the number of items of a specific color -/
structure ColorCount where
  total : Nat
  deriving Repr

/-- Calculates the maximum number of draws needed to guarantee a pair for a single color -/
def maxDrawsForColor (count : ColorCount) : Nat :=
  count.total + 1

/-- The box containing hats and gloves -/
structure Box where
  red : ColorCount
  green : ColorCount
  orange : ColorCount
  blue : ColorCount
  yellow : ColorCount

/-- Calculates the total maximum draws needed for all colors -/
def totalMaxDraws (box : Box) : Nat :=
  maxDrawsForColor box.red +
  maxDrawsForColor box.green +
  maxDrawsForColor box.orange +
  maxDrawsForColor box.blue +
  maxDrawsForColor box.yellow

/-- The given box with the specified item counts -/
def givenBox : Box :=
  { red := { total := 41 },
    green := { total := 23 },
    orange := { total := 11 },
    blue := { total := 15 },
    yellow := { total := 10 } }

theorem max_draws_for_all_pairs (box : Box := givenBox) :
  totalMaxDraws box = 105 := by
  sorry

end NUMINAMATH_CALUDE_max_draws_for_all_pairs_l2137_213787


namespace NUMINAMATH_CALUDE_smallest_square_with_specific_digits_l2137_213770

theorem smallest_square_with_specific_digits : 
  let n : ℕ := 666667
  ∀ m : ℕ, m < n → 
    (m ^ 2 < 444445 * 10^6) ∨ 
    (m ^ 2 ≥ 444446 * 10^6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_with_specific_digits_l2137_213770


namespace NUMINAMATH_CALUDE_negation_equivalence_l2137_213771

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x ≥ 3 ∧ x^2 - 2*x + 3 < 0) ↔ (∀ x : ℝ, x ≥ 3 → x^2 - 2*x + 3 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2137_213771


namespace NUMINAMATH_CALUDE_division_problem_l2137_213795

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 23)
  (h2 : divisor = 4)
  (h3 : remainder = 3)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 5 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2137_213795


namespace NUMINAMATH_CALUDE_problem_statement_l2137_213705

theorem problem_statement (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) 
  (h2 : c^2 + d^2 = 1) : 
  (ab + cd ≤ 1) ∧ (-2 ≤ a + Real.sqrt 3 * b) ∧ (a + Real.sqrt 3 * b ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2137_213705


namespace NUMINAMATH_CALUDE_stratified_sample_male_teachers_l2137_213774

theorem stratified_sample_male_teachers 
  (male_teachers : ℕ) 
  (female_teachers : ℕ) 
  (sample_size : ℕ) 
  (h1 : male_teachers = 56) 
  (h2 : female_teachers = 42) 
  (h3 : sample_size = 14) : 
  ℕ :=
by
  -- The proof goes here
  sorry

#check stratified_sample_male_teachers

end NUMINAMATH_CALUDE_stratified_sample_male_teachers_l2137_213774


namespace NUMINAMATH_CALUDE_sophie_germain_identity_l2137_213788

theorem sophie_germain_identity (a b : ℝ) : 
  a^4 + 4*b^4 = (a^2 + 2*a*b + 2*b^2) * (a^2 - 2*a*b + 2*b^2) := by
  sorry

end NUMINAMATH_CALUDE_sophie_germain_identity_l2137_213788


namespace NUMINAMATH_CALUDE_cone_height_l2137_213749

/-- Given a cone with base radius 1 and central angle of the unfolded side view 2/3π,
    the height of the cone is 2√2. -/
theorem cone_height (r : ℝ) (θ : ℝ) (h : ℝ) : 
  r = 1 → θ = (2/3) * Real.pi → h = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_l2137_213749


namespace NUMINAMATH_CALUDE_distance_sasha_kolya_l2137_213737

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  position : ℝ

/-- The race setup -/
structure Race where
  sasha : Runner
  lyosha : Runner
  kolya : Runner
  length : ℝ

/-- Conditions of the race -/
def race_conditions (r : Race) : Prop :=
  r.length = 100 ∧
  r.sasha.speed > 0 ∧
  r.lyosha.speed > 0 ∧
  r.kolya.speed > 0 ∧
  r.sasha.position = r.length ∧
  r.lyosha.position = r.length - 10 ∧
  r.kolya.position = r.lyosha.position * (r.kolya.speed / r.lyosha.speed)

/-- The theorem to be proved -/
theorem distance_sasha_kolya (r : Race) (h : race_conditions r) :
  r.sasha.position - r.kolya.position = 19 := by
  sorry

end NUMINAMATH_CALUDE_distance_sasha_kolya_l2137_213737


namespace NUMINAMATH_CALUDE_num_triangles_on_square_l2137_213764

/-- The number of points on each side of the square (excluding corners) -/
def points_per_side : ℕ := 7

/-- The total number of points on all sides of the square -/
def total_points : ℕ := 4 * points_per_side

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The number of different triangles formed by selecting three distinct points
    from a set of points on the sides of a square (excluding corners) -/
theorem num_triangles_on_square : 
  choose total_points 3 - 4 * (choose points_per_side 3) = 3136 := by
  sorry

end NUMINAMATH_CALUDE_num_triangles_on_square_l2137_213764


namespace NUMINAMATH_CALUDE_calculation_proof_l2137_213757

theorem calculation_proof :
  (∃ x : ℝ, x ^ 2 = 2 ∧
    (Real.sqrt 6 * Real.sqrt (1/3) - Real.sqrt 16 * Real.sqrt 18 = -11 * x) ∧
    ((2 - Real.sqrt 5) * (2 + Real.sqrt 5) + (2 - Real.sqrt 2) ^ 2 = 5 - 4 * x)) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2137_213757


namespace NUMINAMATH_CALUDE_factorization_equality_l2137_213781

theorem factorization_equality (a : ℝ) : 2 * a^2 - 4 * a + 2 = 2 * (a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2137_213781


namespace NUMINAMATH_CALUDE_total_apples_is_33_l2137_213746

/-- The number of apples picked by each person -/
structure ApplePickers where
  mike : ℕ
  nancy : ℕ
  keith : ℕ
  jennifer : ℕ
  tom : ℕ
  stacy : ℕ

/-- The total number of apples picked -/
def total_apples (pickers : ApplePickers) : ℕ :=
  pickers.mike + pickers.nancy + pickers.keith + pickers.jennifer + pickers.tom + pickers.stacy

/-- Theorem stating that the total number of apples picked is 33 -/
theorem total_apples_is_33 (pickers : ApplePickers) 
    (h_mike : pickers.mike = 7)
    (h_nancy : pickers.nancy = 3)
    (h_keith : pickers.keith = 6)
    (h_jennifer : pickers.jennifer = 5)
    (h_tom : pickers.tom = 8)
    (h_stacy : pickers.stacy = 4) : 
  total_apples pickers = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_is_33_l2137_213746


namespace NUMINAMATH_CALUDE_fish_tank_balls_l2137_213743

/-- The number of goldfish in the tank -/
def num_goldfish : ℕ := 3

/-- The number of platyfish in the tank -/
def num_platyfish : ℕ := 10

/-- The number of red balls each goldfish plays with -/
def red_balls_per_goldfish : ℕ := 10

/-- The number of white balls each platyfish plays with -/
def white_balls_per_platyfish : ℕ := 5

/-- The total number of balls in the fish tank -/
def total_balls : ℕ := num_goldfish * red_balls_per_goldfish + num_platyfish * white_balls_per_platyfish

theorem fish_tank_balls : total_balls = 80 := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_balls_l2137_213743


namespace NUMINAMATH_CALUDE_central_square_illumination_l2137_213775

theorem central_square_illumination (n : ℕ) (h_odd : Odd n) :
  ∃ (min_lamps : ℕ),
    min_lamps = (n + 1)^2 / 2 ∧
    (∀ (lamps : ℕ),
      (∀ (i j : ℕ), i ≤ n ∧ j ≤ n →
        ∃ (k₁ k₂ : ℕ), k₁ ≠ k₂ ∧ k₁ ≤ lamps ∧ k₂ ≤ lamps ∧
          ((i = 0 ∨ i = n ∨ j = 0 ∨ j = n) →
            (k₁ ≤ 4 ∧ k₂ ≤ 4))) →
      lamps ≥ min_lamps) :=
by sorry

end NUMINAMATH_CALUDE_central_square_illumination_l2137_213775


namespace NUMINAMATH_CALUDE_tank_capacity_l2137_213765

/-- Proves that a tank with given leak and inlet rates has a capacity of 1728 litres -/
theorem tank_capacity (leak_empty_time : ℝ) (inlet_rate : ℝ) (combined_empty_time : ℝ) 
  (h1 : leak_empty_time = 8) 
  (h2 : inlet_rate = 6) 
  (h3 : combined_empty_time = 12) : ℝ :=
by
  -- Define the capacity of the tank
  let capacity : ℝ := 1728

  -- State that the capacity is equal to 1728 litres
  have capacity_eq : capacity = 1728 := by rfl

  -- The proof would go here
  sorry


end NUMINAMATH_CALUDE_tank_capacity_l2137_213765


namespace NUMINAMATH_CALUDE_restaurant_production_june_l2137_213751

theorem restaurant_production_june :
  let weekday_cheese_pizzas := 60 + 40
  let weekday_pepperoni_pizzas := 2 * weekday_cheese_pizzas
  let weekday_beef_hotdogs := 30
  let weekday_chicken_hotdogs := 30
  let weekend_cheese_pizzas := 50 + 30
  let weekend_pepperoni_pizzas := 2 * weekend_cheese_pizzas
  let weekend_beef_hotdogs := 20
  let weekend_chicken_hotdogs := 30
  let weekend_bbq_chicken_pizzas := 25
  let weekend_veggie_pizzas := 15
  let weekdays_in_june := 20
  let weekends_in_june := 10
  
  (weekday_cheese_pizzas * weekdays_in_june + weekend_cheese_pizzas * weekends_in_june = 2800) ∧
  (weekday_pepperoni_pizzas * weekdays_in_june + weekend_pepperoni_pizzas * weekends_in_june = 5600) ∧
  (weekday_beef_hotdogs * weekdays_in_june + weekend_beef_hotdogs * weekends_in_june = 800) ∧
  (weekday_chicken_hotdogs * weekdays_in_june + weekend_chicken_hotdogs * weekends_in_june = 900) ∧
  (weekend_bbq_chicken_pizzas * weekends_in_june = 250) ∧
  (weekend_veggie_pizzas * weekends_in_june = 150) := by
  sorry

end NUMINAMATH_CALUDE_restaurant_production_june_l2137_213751


namespace NUMINAMATH_CALUDE_smallest_a_value_smallest_a_exists_l2137_213701

/-- A three-digit even number -/
def ThreeDigitEven := {n : ℕ // 100 ≤ n ∧ n ≤ 998 ∧ Even n}

/-- The sum of five three-digit even numbers -/
def SumFiveNumbers := 4306

theorem smallest_a_value (A B C D E : ThreeDigitEven) 
  (h_order : A.val < B.val ∧ B.val < C.val ∧ C.val < D.val ∧ D.val < E.val)
  (h_sum : A.val + B.val + C.val + D.val + E.val = SumFiveNumbers) :
  A.val ≥ 326 := by
  sorry

theorem smallest_a_exists :
  ∃ (A B C D E : ThreeDigitEven),
    A.val < B.val ∧ B.val < C.val ∧ C.val < D.val ∧ D.val < E.val ∧
    A.val + B.val + C.val + D.val + E.val = SumFiveNumbers ∧
    A.val = 326 := by
  sorry

end NUMINAMATH_CALUDE_smallest_a_value_smallest_a_exists_l2137_213701


namespace NUMINAMATH_CALUDE_average_headcount_l2137_213714

-- Define the list of spring term headcounts
def spring_headcounts : List Nat := [11000, 10200, 10800, 11300]

-- Define the number of terms
def num_terms : Nat := 4

-- Theorem to prove the average headcount
theorem average_headcount :
  (spring_headcounts.sum / num_terms : ℚ) = 10825 := by
  sorry

end NUMINAMATH_CALUDE_average_headcount_l2137_213714
