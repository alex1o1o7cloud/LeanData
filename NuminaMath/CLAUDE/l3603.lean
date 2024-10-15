import Mathlib

namespace NUMINAMATH_CALUDE_unique_non_negative_one_result_l3603_360392

theorem unique_non_negative_one_result :
  (-1 * 1 = -1) ∧
  ((-1) / (-1) ≠ -1) ∧
  (-2015 / 2015 = -1) ∧
  ((-1)^9 * (-1)^2 = -1) :=
by sorry

end NUMINAMATH_CALUDE_unique_non_negative_one_result_l3603_360392


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l3603_360362

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem seventh_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_sum1 : a 1 + a 2 = 3) 
  (h_sum2 : a 2 + a 3 = 6) : 
  a 7 = 64 := by
sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l3603_360362


namespace NUMINAMATH_CALUDE_alberto_bjorn_bike_distance_l3603_360397

/-- The problem of comparing distances biked by Alberto and Bjorn -/
theorem alberto_bjorn_bike_distance :
  let alberto_rate : ℝ := 80 / 4  -- Alberto's constant rate in miles per hour
  let alberto_time : ℝ := 4  -- Alberto's total time in hours
  let bjorn_rate1 : ℝ := 20  -- Bjorn's first rate in miles per hour
  let bjorn_rate2 : ℝ := 25  -- Bjorn's second rate in miles per hour
  let bjorn_time1 : ℝ := 2  -- Bjorn's time at first rate in hours
  let bjorn_time2 : ℝ := 2  -- Bjorn's time at second rate in hours
  
  let alberto_distance : ℝ := alberto_rate * alberto_time
  let bjorn_distance : ℝ := bjorn_rate1 * bjorn_time1 + bjorn_rate2 * bjorn_time2
  
  alberto_distance - bjorn_distance = -10
  := by sorry

end NUMINAMATH_CALUDE_alberto_bjorn_bike_distance_l3603_360397


namespace NUMINAMATH_CALUDE_cake_mass_proof_l3603_360317

/-- The original mass of the cake in grams -/
def original_mass : ℝ := 750

/-- The mass of cake eaten by Carlson as a fraction -/
def carlson_fraction : ℝ := 0.4

/-- The mass of cake eaten by Little Man in grams -/
def little_man_mass : ℝ := 150

/-- The fraction of remaining cake eaten by Freken Bok -/
def freken_bok_fraction : ℝ := 0.3

/-- The additional mass of cake eaten by Freken Bok in grams -/
def freken_bok_additional : ℝ := 120

/-- The mass of cake crumbs eaten by Matilda in grams -/
def matilda_crumbs : ℝ := 90

theorem cake_mass_proof :
  let remaining_after_carlson := original_mass * (1 - carlson_fraction)
  let remaining_after_little_man := remaining_after_carlson - little_man_mass
  let remaining_after_freken_bok := remaining_after_little_man * (1 - freken_bok_fraction) - freken_bok_additional
  remaining_after_freken_bok = matilda_crumbs :=
by sorry

end NUMINAMATH_CALUDE_cake_mass_proof_l3603_360317


namespace NUMINAMATH_CALUDE_rubber_band_difference_l3603_360322

theorem rubber_band_difference (total : ℕ) (aira_initial : ℕ) (samantha_extra : ℕ) (equal_share : ℕ)
  (h1 : total = 18)
  (h2 : aira_initial = 4)
  (h3 : samantha_extra = 5)
  (h4 : equal_share = 6) :
  let samantha_initial := aira_initial + samantha_extra
  let joe_initial := total - samantha_initial - aira_initial
  joe_initial - aira_initial = 1 := by sorry

end NUMINAMATH_CALUDE_rubber_band_difference_l3603_360322


namespace NUMINAMATH_CALUDE_correct_seating_arrangements_l3603_360394

/-- The number of ways to arrange n distinct objects -/
def factorial (n : ℕ) : ℕ := (List.range n).foldr (· * ·) 1

/-- The number of people to be seated -/
def total_people : ℕ := 10

/-- The number of people with seating restrictions -/
def restricted_people : ℕ := 4

/-- The number of ways to arrange 10 people in a row, where 4 specific people cannot sit in 4 consecutive seats -/
def seating_arrangements : ℕ := 
  factorial total_people - factorial (total_people - restricted_people + 1) * factorial restricted_people

theorem correct_seating_arrangements : seating_arrangements = 3507840 := by
  sorry

end NUMINAMATH_CALUDE_correct_seating_arrangements_l3603_360394


namespace NUMINAMATH_CALUDE_triangle_problem_l3603_360372

theorem triangle_problem (A B C : Real) (a b c : Real) :
  (0 < A) ∧ (A < π) ∧
  (0 < B) ∧ (B < π) ∧
  (0 < C) ∧ (C < π) ∧
  (A + B + C = π) ∧
  (c * Real.cos B + b * Real.cos C = 2 * a * Real.cos A) ∧
  (a = 2) ∧
  (1/2 * b * c * Real.sin A = Real.sqrt 3) →
  (A = π/3) ∧ (a + b + c = 6) := by
sorry


end NUMINAMATH_CALUDE_triangle_problem_l3603_360372


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3603_360355

theorem arithmetic_sequence_length : ∀ (a₁ d : ℤ) (n : ℕ),
  a₁ = 165 ∧ d = -6 ∧ (a₁ + d * (n - 1 : ℤ) ≤ 24) ∧ (a₁ + d * ((n - 1) - 1 : ℤ) > 24) →
  n = 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3603_360355


namespace NUMINAMATH_CALUDE_square_field_area_l3603_360375

/-- Given a square field where a horse takes 7 hours to run around it at a speed of 20 km/h,
    the area of the field is 1225 km². -/
theorem square_field_area (s : ℝ) (h : s > 0) : 
  (4 * s = 20 * 7) → s^2 = 1225 := by sorry

end NUMINAMATH_CALUDE_square_field_area_l3603_360375


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l3603_360345

theorem quadratic_roots_relation (p A B : ℤ) : 
  (∃ α β : ℝ, α + 1 ≠ β + 1 ∧ 
    (∀ x : ℝ, x^2 + p*x + 19 = 0 ↔ x = α + 1 ∨ x = β + 1) ∧
    (∀ x : ℝ, x^2 - A*x + B = 0 ↔ x = α ∨ x = β)) →
  A + B = 18 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l3603_360345


namespace NUMINAMATH_CALUDE_fish_filets_count_l3603_360389

/-- The number of fish filets Ben and his family will have -/
def fish_filets : ℕ :=
  let ben_fish := 4
  let judy_fish := 1
  let billy_fish := 3
  let jim_fish := 2
  let susie_fish := 5
  let total_caught := ben_fish + judy_fish + billy_fish + jim_fish + susie_fish
  let thrown_back := 3
  let kept_fish := total_caught - thrown_back
  let filets_per_fish := 2
  kept_fish * filets_per_fish

theorem fish_filets_count : fish_filets = 24 := by
  sorry

end NUMINAMATH_CALUDE_fish_filets_count_l3603_360389


namespace NUMINAMATH_CALUDE_skating_minutes_on_eleventh_day_l3603_360378

def minutes_per_day_first_period : ℕ := 80
def days_first_period : ℕ := 6
def minutes_per_day_second_period : ℕ := 105
def days_second_period : ℕ := 4
def target_average : ℕ := 95
def total_days : ℕ := 11

theorem skating_minutes_on_eleventh_day :
  (minutes_per_day_first_period * days_first_period +
   minutes_per_day_second_period * days_second_period +
   145) / total_days = target_average :=
by sorry

end NUMINAMATH_CALUDE_skating_minutes_on_eleventh_day_l3603_360378


namespace NUMINAMATH_CALUDE_different_color_probability_l3603_360304

def total_chips : ℕ := 7 + 5
def red_chips : ℕ := 7
def green_chips : ℕ := 5

theorem different_color_probability :
  (red_chips * green_chips : ℚ) / (total_chips * (total_chips - 1) / 2) = 35 / 66 := by
  sorry

end NUMINAMATH_CALUDE_different_color_probability_l3603_360304


namespace NUMINAMATH_CALUDE_cube_root_simplification_l3603_360331

theorem cube_root_simplification :
  let x : ℝ := 5488000
  let y : ℝ := 2744
  let z : ℝ := 343
  (1000 = 10^3) →
  (y = 2^3 * z) →
  (z = 7^3) →
  x^(1/3) = 140 * 2^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l3603_360331


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3603_360303

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), x = 0.56 ∧ x = 56 / 99 :=
by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3603_360303


namespace NUMINAMATH_CALUDE_pascal_and_coin_toss_l3603_360385

/-- Pascal's Triangle row sum -/
def pascal_row_sum (n : ℕ) : ℕ := 2^n

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Probability of k successes in n independent trials -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial n k : ℚ) * p^k * (1 - p)^(n - k)

theorem pascal_and_coin_toss :
  pascal_row_sum 10 = 1024 ∧
  binomial_probability 10 5 (1/2) = 63/256 := by sorry

end NUMINAMATH_CALUDE_pascal_and_coin_toss_l3603_360385


namespace NUMINAMATH_CALUDE_fraction_integer_condition_l3603_360349

theorem fraction_integer_condition (p : ℕ+) :
  (↑p : ℚ) ∈ ({3, 5, 9, 35} : Set ℚ) ↔ ∃ (k : ℤ), k > 0 ∧ (3 * p + 25 : ℚ) / (2 * p - 5 : ℚ) = k := by
  sorry

end NUMINAMATH_CALUDE_fraction_integer_condition_l3603_360349


namespace NUMINAMATH_CALUDE_quadratic_function_satisfies_conditions_l3603_360341

/-- The quadratic equation x^2 - x + 1 = 0 has roots α and β -/
def has_roots (α β : ℂ) : Prop :=
  α^2 - α + 1 = 0 ∧ β^2 - β + 1 = 0

/-- The quadratic function f(x) = x^2 - 2x + 2 -/
def f (x : ℂ) : ℂ := x^2 - 2*x + 2

/-- Theorem stating that f(x) satisfies the required conditions -/
theorem quadratic_function_satisfies_conditions (α β : ℂ) 
  (h : has_roots α β) : f α = β ∧ f β = α ∧ f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_satisfies_conditions_l3603_360341


namespace NUMINAMATH_CALUDE_max_value_of_f_in_interval_l3603_360391

def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 5

theorem max_value_of_f_in_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-4 : ℝ) (4 : ℝ) ∧
  ∀ (x : ℝ), x ∈ Set.Icc (-4 : ℝ) (4 : ℝ) → f x ≤ f c ∧ f c = 10 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_in_interval_l3603_360391


namespace NUMINAMATH_CALUDE_equation_solutions_l3603_360361

theorem equation_solutions :
  (∃! x : ℝ, x^2 - 2*x = -1) ∧
  (∀ x : ℝ, (x + 3)^2 = 2*x*(x + 3) ↔ x = -3 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3603_360361


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l3603_360357

theorem simplify_fraction_product : 5 * (18 / 7) * (21 / -63) = -30 / 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l3603_360357


namespace NUMINAMATH_CALUDE_divisors_of_2_pow_n_minus_1_l3603_360390

theorem divisors_of_2_pow_n_minus_1 (n : ℕ) (d : ℕ) (h1 : Odd n) (h2 : d > 0) (h3 : d ∣ (2^n - 1)) :
  d % 8 = 1 ∨ d % 8 = 7 :=
sorry

end NUMINAMATH_CALUDE_divisors_of_2_pow_n_minus_1_l3603_360390


namespace NUMINAMATH_CALUDE_exists_special_set_l3603_360387

/-- A function that checks if a natural number is a perfect power -/
def isPerfectPower (n : ℕ) : Prop :=
  ∃ (b k : ℕ), k > 1 ∧ n = b^k

/-- The existence of a set of 1992 positive integers with the required property -/
theorem exists_special_set : ∃ (S : Finset ℕ), 
  (S.card = 1992) ∧ 
  (∀ (T : Finset ℕ), T ⊆ S → isPerfectPower (T.sum id)) :=
sorry

end NUMINAMATH_CALUDE_exists_special_set_l3603_360387


namespace NUMINAMATH_CALUDE_smith_family_seating_arrangement_l3603_360311

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smith_family_seating_arrangement :
  let total_arrangements := factorial 7
  let no_adjacent_boys := factorial 4 * factorial 3
  total_arrangements - no_adjacent_boys = 4896 :=
by sorry

end NUMINAMATH_CALUDE_smith_family_seating_arrangement_l3603_360311


namespace NUMINAMATH_CALUDE_max_value_sum_cubes_fourth_powers_l3603_360383

theorem max_value_sum_cubes_fourth_powers (a b c : ℕ+) 
  (h : a + b + c = 2) : 
  (∀ x y z : ℕ+, x + y + z = 2 → a + b^3 + c^4 ≥ x + y^3 + z^4) ∧ 
  (∃ x y z : ℕ+, x + y + z = 2 ∧ a + b^3 + c^4 = x + y^3 + z^4) :=
sorry

end NUMINAMATH_CALUDE_max_value_sum_cubes_fourth_powers_l3603_360383


namespace NUMINAMATH_CALUDE_adam_shelf_capacity_l3603_360371

/-- The number of action figures that can fit on each shelf. -/
def figures_per_shelf : ℕ := 9

/-- The number of shelves in Adam's room. -/
def number_of_shelves : ℕ := 3

/-- The total number of action figures that can fit on all shelves. -/
def total_figures : ℕ := figures_per_shelf * number_of_shelves

theorem adam_shelf_capacity :
  total_figures = 27 :=
by sorry

end NUMINAMATH_CALUDE_adam_shelf_capacity_l3603_360371


namespace NUMINAMATH_CALUDE_trig_identity_l3603_360363

theorem trig_identity (α : ℝ) : 
  (3 - 4 * Real.cos (2 * α) + Real.cos (4 * α)) / 
  (3 + 4 * Real.cos (2 * α) + Real.cos (4 * α)) = 
  (Real.tan α) ^ 4 / 3.396 := by sorry

end NUMINAMATH_CALUDE_trig_identity_l3603_360363


namespace NUMINAMATH_CALUDE_largest_product_sum_of_digits_l3603_360367

def is_prime (p : ℕ) : Prop := sorry

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem largest_product_sum_of_digits :
  ∃ (n d e : ℕ),
    is_prime d ∧ is_prime e ∧ is_prime (10 * e + d) ∧
    d ∈ ({5, 7} : Set ℕ) ∧ e ∈ ({3, 7} : Set ℕ) ∧
    n = d * e * (10 * e + d) ∧
    (∀ (m d' e' : ℕ),
      is_prime d' ∧ is_prime e' ∧ is_prime (10 * e' + d') ∧
      d' ∈ ({5, 7} : Set ℕ) ∧ e' ∈ ({3, 7} : Set ℕ) ∧
      m = d' * e' * (10 * e' + d') →
      m ≤ n) ∧
    sum_of_digits n = 21 :=
by sorry

end NUMINAMATH_CALUDE_largest_product_sum_of_digits_l3603_360367


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3603_360326

theorem complex_number_in_first_quadrant : 
  let z : ℂ := 1 / (1 - Complex.I)
  0 < z.re ∧ 0 < z.im :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3603_360326


namespace NUMINAMATH_CALUDE_no_solutions_for_diophantine_equation_l3603_360334

theorem no_solutions_for_diophantine_equation :
  ¬∃ (m : ℕ+) (p q : ℕ), 
    Nat.Prime p ∧ Nat.Prime q ∧ 2^(m : ℕ) * p^2 + 1 = q^7 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_diophantine_equation_l3603_360334


namespace NUMINAMATH_CALUDE_least_common_period_is_36_l3603_360365

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 6) + f (x - 6) = f x

/-- The period of a function -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

/-- The least common positive period for all functions satisfying the functional equation -/
def LeastCommonPeriod (p : ℝ) : Prop :=
  p > 0 ∧
  (∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f → IsPeriod f p) ∧
  ∀ q, q > 0 → (∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f → IsPeriod f q) → p ≤ q

theorem least_common_period_is_36 : LeastCommonPeriod 36 := by
  sorry

end NUMINAMATH_CALUDE_least_common_period_is_36_l3603_360365


namespace NUMINAMATH_CALUDE_smallest_odd_four_prime_factors_l3603_360312

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def has_exactly_four_prime_factors (n : ℕ) : Prop :=
  ∃ (p q r s : ℕ), is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime s ∧
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
    n = p * q * r * s

theorem smallest_odd_four_prime_factors :
  (1155 % 2 = 1) ∧
  has_exactly_four_prime_factors 1155 ∧
  ∀ n : ℕ, n < 1155 → (n % 2 = 1 → ¬has_exactly_four_prime_factors n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_odd_four_prime_factors_l3603_360312


namespace NUMINAMATH_CALUDE_exists_composite_invariant_under_triplet_replacement_l3603_360376

/-- A function that replaces a triplet of digits at a given position in a natural number --/
def replaceTriplet (n : ℕ) (pos : ℕ) (newTriplet : ℕ) : ℕ :=
  sorry

/-- Predicate to check if a number is composite --/
def isComposite (n : ℕ) : Prop :=
  ∃ a b, a > 1 ∧ b > 1 ∧ n = a * b

/-- The main theorem statement --/
theorem exists_composite_invariant_under_triplet_replacement :
  ∃ (N : ℕ), ∀ (pos : ℕ) (newTriplet : ℕ),
    isComposite (replaceTriplet N pos newTriplet) :=
  sorry

end NUMINAMATH_CALUDE_exists_composite_invariant_under_triplet_replacement_l3603_360376


namespace NUMINAMATH_CALUDE_prob_A_level_l3603_360319

/-- The probability of producing a B-level product -/
def prob_B : ℝ := 0.03

/-- The probability of producing a C-level product -/
def prob_C : ℝ := 0.01

/-- Theorem: The probability of selecting an A-level product is 0.96 -/
theorem prob_A_level (h1 : prob_B = 0.03) (h2 : prob_C = 0.01) :
  1 - (prob_B + prob_C) = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_level_l3603_360319


namespace NUMINAMATH_CALUDE_square_area_from_oblique_projection_l3603_360395

/-- Represents a square in 2D space -/
structure Square where
  side_length : ℝ
  area : ℝ := side_length ^ 2

/-- Represents a parallelogram in 2D space -/
structure Parallelogram where
  side1 : ℝ
  side2 : ℝ

/-- Represents an oblique projection transformation -/
def obliqueProjection (s : Square) : Parallelogram :=
  sorry

theorem square_area_from_oblique_projection 
  (s : Square) 
  (p : Parallelogram) 
  (h1 : p = obliqueProjection s) 
  (h2 : p.side1 = 4 ∨ p.side2 = 4) : 
  s.area = 16 ∨ s.area = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_oblique_projection_l3603_360395


namespace NUMINAMATH_CALUDE_linear_inequality_solution_set_l3603_360318

theorem linear_inequality_solution_set 
  (m n : ℝ) 
  (h1 : m = -1) 
  (h2 : n = -1) : 
  {x : ℝ | m * x - n ≤ 2} = {x : ℝ | x ≥ -1} := by
sorry

end NUMINAMATH_CALUDE_linear_inequality_solution_set_l3603_360318


namespace NUMINAMATH_CALUDE_arcsin_arccos_eq_arctan_pi_fourth_l3603_360353

theorem arcsin_arccos_eq_arctan_pi_fourth :
  ∃ x : ℝ, x = 0 ∧ Real.arcsin x + Real.arccos (1 - x) = Real.arctan x + π / 4 :=
by sorry

end NUMINAMATH_CALUDE_arcsin_arccos_eq_arctan_pi_fourth_l3603_360353


namespace NUMINAMATH_CALUDE_language_group_selection_l3603_360324

theorem language_group_selection (total : Nat) (english : Nat) (japanese : Nat)
  (h_total : total = 9)
  (h_english : english = 7)
  (h_japanese : japanese = 3)
  (h_at_least_one : english + japanese ≥ total) :
  (english * japanese) - (english + japanese - total) = 20 := by
  sorry

end NUMINAMATH_CALUDE_language_group_selection_l3603_360324


namespace NUMINAMATH_CALUDE_pyramid_volume_in_cube_l3603_360382

theorem pyramid_volume_in_cube (s : ℝ) (h : s > 0) :
  let cube_volume := s^3
  let pyramid_volume := (1/3) * (s^2/2) * s
  pyramid_volume = (1/6) * cube_volume := by
sorry

end NUMINAMATH_CALUDE_pyramid_volume_in_cube_l3603_360382


namespace NUMINAMATH_CALUDE_ratio_problem_l3603_360396

theorem ratio_problem (a b c d : ℚ) 
  (h1 : a / b = 5 / 4)
  (h2 : c / d = 4 / 1)
  (h3 : d / b = 2 / 5) :
  a / c = 25 / 32 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l3603_360396


namespace NUMINAMATH_CALUDE_strawberry_harvest_l3603_360388

/-- Calculates the total number of strawberries harvested in a rectangular garden --/
theorem strawberry_harvest (length width : ℕ) (plants_per_sqft : ℕ) (strawberries_per_plant : ℕ) :
  length = 10 → width = 12 → plants_per_sqft = 5 → strawberries_per_plant = 8 →
  length * width * plants_per_sqft * strawberries_per_plant = 4800 := by
  sorry

#check strawberry_harvest

end NUMINAMATH_CALUDE_strawberry_harvest_l3603_360388


namespace NUMINAMATH_CALUDE_pool_filling_time_l3603_360379

theorem pool_filling_time (R : ℝ) (h1 : R > 0) : 
  (R + 1.5 * R) * 5 = 1 → R * 12.5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_pool_filling_time_l3603_360379


namespace NUMINAMATH_CALUDE_largest_m_is_nine_l3603_360332

/-- A quadratic function f(x) = ax² + bx + c satisfying specific conditions -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  symmetry : ∀ x : ℝ, a * (x - 4)^2 + b * (x - 4) + c = a * (2 - x)^2 + b * (2 - x) + c
  lower_bound : ∀ x : ℝ, a * x^2 + b * x + c ≥ x
  upper_bound : ∀ x ∈ Set.Ioo 0 2, a * x^2 + b * x + c ≤ ((x + 1) / 2)^2
  min_value : ∃ x : ℝ, ∀ y : ℝ, a * x^2 + b * x + c ≤ a * y^2 + b * y + c ∧ a * x^2 + b * x + c = 0

/-- The theorem stating that the largest m > 1 satisfying the given conditions is 9 -/
theorem largest_m_is_nine (f : QuadraticFunction) :
  ∃ m : ℝ, m = 9 ∧ 
  (∃ t : ℝ, ∀ x ∈ Set.Icc 1 m, f.a * (x + t)^2 + f.b * (x + t) + f.c ≤ x) ∧
  ∀ m' > m, ¬∃ t : ℝ, ∀ x ∈ Set.Icc 1 m', f.a * (x + t)^2 + f.b * (x + t) + f.c ≤ x :=
sorry

end NUMINAMATH_CALUDE_largest_m_is_nine_l3603_360332


namespace NUMINAMATH_CALUDE_prob_a_wins_l3603_360347

/-- Given a chess game between players A and B, this theorem proves
    the probability of player A winning, given the probabilities of
    a draw and A not losing. -/
theorem prob_a_wins (prob_draw prob_a_not_lose : ℚ)
  (h_draw : prob_draw = 1/2)
  (h_not_lose : prob_a_not_lose = 5/6) :
  prob_a_not_lose - prob_draw = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_prob_a_wins_l3603_360347


namespace NUMINAMATH_CALUDE_compensation_problem_l3603_360364

/-- Represents the compensation amounts for cow, horse, and sheep respectively -/
structure Compensation where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The problem statement -/
theorem compensation_problem (comp : Compensation) : 
  -- Total compensation is 5 measures (50 liters)
  comp.a + comp.b + comp.c = 50 →
  -- Sheep ate half as much as horse
  comp.c = (1/2) * comp.b →
  -- Horse ate half as much as cow
  comp.b = (1/2) * comp.a →
  -- Compensation is proportional to what each animal ate
  (∃ (k : ℚ), k > 0 ∧ comp.a = k * 4 ∧ comp.b = k * 2 ∧ comp.c = k * 1) →
  -- Prove that a, b, c form a geometric sequence with ratio 1/2 and c = 50/7
  (comp.b = (1/2) * comp.a ∧ comp.c = (1/2) * comp.b) ∧ comp.c = 50/7 := by
  sorry

end NUMINAMATH_CALUDE_compensation_problem_l3603_360364


namespace NUMINAMATH_CALUDE_sqrt_12_bounds_l3603_360328

theorem sqrt_12_bounds : 3 < Real.sqrt 12 ∧ Real.sqrt 12 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_bounds_l3603_360328


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3603_360340

theorem inscribed_circle_radius (a b c : ℝ) (h1 : a = 6) (h2 : b = 12) (h3 : c = 18) :
  let r := (1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))⁻¹
  r = 36 / 17 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3603_360340


namespace NUMINAMATH_CALUDE_triangle_max_area_l3603_360301

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the maximum area of the triangle is 2 + √3 when b²-2√3bc*sin(A)+c²=4 and a=2 -/
theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  b^2 - 2 * Real.sqrt 3 * b * c * Real.sin A + c^2 = 4 →
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  A + B + C = Real.pi →
  (∃ (S : ℝ), S = (1/2) * b * c * Real.sin A ∧
    ∀ (S' : ℝ), S' = (1/2) * b * c * Real.sin A → S' ≤ 2 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l3603_360301


namespace NUMINAMATH_CALUDE_prob_three_red_standard_deck_l3603_360384

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (h_total : total_cards = 52)
  (h_red : red_cards = 26)

/-- The probability of drawing three red cards from a standard deck -/
def prob_three_red (d : Deck) : ℚ :=
  (d.red_cards * (d.red_cards - 1) * (d.red_cards - 2)) / 
  (d.total_cards * (d.total_cards - 1) * (d.total_cards - 2))

/-- Theorem stating the probability of drawing three red cards from a standard deck -/
theorem prob_three_red_standard_deck :
  ∃ (d : Deck), prob_three_red d = 200 / 1701 :=
sorry

end NUMINAMATH_CALUDE_prob_three_red_standard_deck_l3603_360384


namespace NUMINAMATH_CALUDE_marks_score_is_46_l3603_360314

def highest_score : ℕ := 98
def score_range : ℕ := 75

def least_score : ℕ := highest_score - score_range

def marks_score : ℕ := 2 * least_score

theorem marks_score_is_46 : marks_score = 46 := by
  sorry

end NUMINAMATH_CALUDE_marks_score_is_46_l3603_360314


namespace NUMINAMATH_CALUDE_perimeter_semicircular_arcs_on_square_l3603_360306

/-- The perimeter of a region bounded by semicircular arcs on a square's sides -/
theorem perimeter_semicircular_arcs_on_square (side_length : Real) :
  side_length = 4 / Real.pi →
  (4 : Real) * (Real.pi * side_length / 2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_semicircular_arcs_on_square_l3603_360306


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l3603_360338

theorem polynomial_division_quotient (z : ℝ) : 
  4 * z^5 - 3 * z^4 + 2 * z^3 - 5 * z^2 + 7 * z - 3 = 
  (z + 2) * (4 * z^4 - 11 * z^3 + 24 * z^2 - 53 * z + 113) + (-229) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l3603_360338


namespace NUMINAMATH_CALUDE_collinear_dots_probability_l3603_360356

/-- Represents a 5x5 grid of dots -/
def Grid := Fin 5 × Fin 5

/-- The total number of dots in the grid -/
def total_dots : Nat := 25

/-- The number of ways to choose 4 dots from the total dots -/
def total_choices : Nat := Nat.choose total_dots 4

/-- The number of sets of 4 collinear dots in the grid -/
def collinear_sets : Nat := 28

/-- The probability of choosing 4 collinear dots -/
def collinear_probability : Rat := collinear_sets / total_choices

theorem collinear_dots_probability :
  collinear_probability = 4 / 1807 := by sorry

end NUMINAMATH_CALUDE_collinear_dots_probability_l3603_360356


namespace NUMINAMATH_CALUDE_boys_from_maple_high_school_l3603_360380

theorem boys_from_maple_high_school (total_students : ℕ) (total_boys : ℕ) (total_girls : ℕ)
  (jonas_students : ℕ) (clay_students : ℕ) (maple_students : ℕ)
  (jonas_girls : ℕ) (clay_girls : ℕ) :
  total_students = 150 →
  total_boys = 85 →
  total_girls = 65 →
  jonas_students = 50 →
  clay_students = 70 →
  maple_students = 30 →
  jonas_girls = 25 →
  clay_girls = 30 →
  total_students = total_boys + total_girls →
  total_students = jonas_students + clay_students + maple_students →
  (maple_students - (total_girls - jonas_girls - clay_girls) : ℤ) = 20 := by
sorry

end NUMINAMATH_CALUDE_boys_from_maple_high_school_l3603_360380


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l3603_360342

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  side : ℕ  -- Length of equal sides
  base : ℕ  -- Length of the base
  isValid : 2 * side > base  -- Triangle inequality

/-- Check if two triangles have the same perimeter -/
def samePerimeter (t1 t2 : IsoscelesTriangle) : Prop :=
  2 * t1.side + t1.base = 2 * t2.side + t2.base

/-- Check if two triangles have the same area -/
def sameArea (t1 t2 : IsoscelesTriangle) : Prop :=
  t1.base * (t1.side ^ 2 - (t1.base / 2) ^ 2).sqrt = 
  t2.base * (t2.side ^ 2 - (t2.base / 2) ^ 2).sqrt

/-- Check if the base ratio of two triangles is 5:4 -/
def baseRatio54 (t1 t2 : IsoscelesTriangle) : Prop :=
  5 * t2.base = 4 * t1.base

/-- The main theorem -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    samePerimeter t1 t2 ∧
    sameArea t1 t2 ∧
    baseRatio54 t1 t2 ∧
    (∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      samePerimeter s1 s2 →
      sameArea s1 s2 →
      baseRatio54 s1 s2 →
      2 * t1.side + t1.base ≤ 2 * s1.side + s1.base) ∧
    2 * t1.side + t1.base = 138 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l3603_360342


namespace NUMINAMATH_CALUDE_expand_expression_l3603_360350

theorem expand_expression (x : ℝ) : 24 * (3 * x - 4) = 72 * x - 96 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3603_360350


namespace NUMINAMATH_CALUDE_decimal_multiplication_l3603_360323

theorem decimal_multiplication (a b c : ℚ) (h1 : a = 0.025) (h2 : b = 3.84) (h3 : c = 0.096) 
  (h4 : (25 : ℕ) * 384 = 9600) : a * b = c := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_l3603_360323


namespace NUMINAMATH_CALUDE_class_size_is_ten_l3603_360386

/-- The number of students who scored 92 -/
def high_scorers : ℕ := 5

/-- The number of students who scored 80 -/
def mid_scorers : ℕ := 4

/-- The score of the last student -/
def last_score : ℕ := 70

/-- The minimum required average score -/
def min_average : ℕ := 85

/-- The total number of students in the class -/
def total_students : ℕ := high_scorers + mid_scorers + 1

theorem class_size_is_ten :
  total_students = 10 ∧
  (high_scorers * 92 + mid_scorers * 80 + last_score) / total_students ≥ min_average := by
  sorry

end NUMINAMATH_CALUDE_class_size_is_ten_l3603_360386


namespace NUMINAMATH_CALUDE_smallest_n_for_terminating_decimal_l3603_360344

/-- A fraction a/b is a terminating decimal if b can be written as 2^m * 5^n for some non-negative integers m and n. -/
def IsTerminatingDecimal (a b : ℕ) : Prop :=
  ∃ (m n : ℕ), b = 2^m * 5^n

/-- The smallest positive integer n such that n/(n+107) is a terminating decimal is 143. -/
theorem smallest_n_for_terminating_decimal : 
  (∀ k : ℕ, 0 < k → k < 143 → ¬ IsTerminatingDecimal k (k + 107)) ∧ 
  IsTerminatingDecimal 143 (143 + 107) := by
  sorry

#check smallest_n_for_terminating_decimal

end NUMINAMATH_CALUDE_smallest_n_for_terminating_decimal_l3603_360344


namespace NUMINAMATH_CALUDE_stating_locus_of_vertex_c_l3603_360313

/-- Represents a triangle ABC with specific properties -/
structure SpecialTriangle where
  /-- Length of side AB -/
  ab : ℝ
  /-- Length of median from A to BC -/
  median_a : ℝ
  /-- Length of altitude from A to BC -/
  altitude_a : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  /-- Center of the circle -/
  center : ℝ × ℝ
  /-- Radius of the circle -/
  radius : ℝ

/-- 
Theorem stating that the locus of vertex C in the special triangle 
is a circle with specific properties
-/
theorem locus_of_vertex_c (t : SpecialTriangle) 
  (h1 : t.ab = 6)
  (h2 : t.median_a = 4)
  (h3 : t.altitude_a = 3) :
  ∃ (c : Circle), 
    c.radius = 3 ∧ 
    c.center.1 = 4 ∧ 
    c.center.2 = 3 ∧
    (∀ (x y : ℝ), (x, y) ∈ {p | (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2} ↔ 
      ∃ (triangle : SpecialTriangle), 
        triangle.ab = t.ab ∧ 
        triangle.median_a = t.median_a ∧ 
        triangle.altitude_a = t.altitude_a) :=
sorry

end NUMINAMATH_CALUDE_stating_locus_of_vertex_c_l3603_360313


namespace NUMINAMATH_CALUDE_hall_breadth_is_12_l3603_360360

def hall_length : ℝ := 15
def hall_volume : ℝ := 1200

theorem hall_breadth_is_12 (b h : ℝ) 
  (area_eq : 2 * (hall_length * b) = 2 * (hall_length * h + b * h))
  (volume_eq : hall_length * b * h = hall_volume) :
  b = 12 := by sorry

end NUMINAMATH_CALUDE_hall_breadth_is_12_l3603_360360


namespace NUMINAMATH_CALUDE_attendant_claimed_two_shirts_l3603_360352

-- Define the given conditions
def trousers : ℕ := 10
def total_bill : ℕ := 140
def shirt_cost : ℕ := 5
def trouser_cost : ℕ := 9
def missing_shirts : ℕ := 8

-- Define the function to calculate the number of shirts the attendant initially claimed
def attendant_claim : ℕ :=
  let trouser_total : ℕ := trousers * trouser_cost
  let shirt_total : ℕ := total_bill - trouser_total
  let actual_shirts : ℕ := shirt_total / shirt_cost
  actual_shirts - missing_shirts

-- Theorem statement
theorem attendant_claimed_two_shirts :
  attendant_claim = 2 := by sorry

end NUMINAMATH_CALUDE_attendant_claimed_two_shirts_l3603_360352


namespace NUMINAMATH_CALUDE_x_squared_plus_inverse_x_squared_l3603_360309

theorem x_squared_plus_inverse_x_squared (x : ℝ) (h : x - 1/x = 3) : x^2 + 1/x^2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_inverse_x_squared_l3603_360309


namespace NUMINAMATH_CALUDE_B_is_midpoint_of_AC_l3603_360358

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define points A, B, C, and O
variable (O A B C : V)

-- Define the collinearity of points A, B, and C
def collinear (A B C : V) : Prop :=
  ∃ t : ℝ, B - A = t • (C - A)

-- Define the vector equation
def vector_equation (m : ℝ) : Prop :=
  m • (A - O) - 2 • (B - O) + (C - O) = 0

-- Theorem statement
theorem B_is_midpoint_of_AC 
  (h_collinear : collinear A B C)
  (h_equation : ∃ m : ℝ, vector_equation O A B C m) :
  B - O = (1/2) • ((A - O) + (C - O)) :=
sorry

end NUMINAMATH_CALUDE_B_is_midpoint_of_AC_l3603_360358


namespace NUMINAMATH_CALUDE_consecutive_product_theorem_l3603_360393

theorem consecutive_product_theorem (n : ℕ) : 
  (∃ m : ℕ, 9*n^2 + 5*n - 26 = m * (m + 1)) → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_theorem_l3603_360393


namespace NUMINAMATH_CALUDE_bob_rope_art_fraction_l3603_360300

theorem bob_rope_art_fraction (total_length : ℝ) (remaining_length : ℝ) (num_sections : ℕ) (section_length : ℝ) : 
  total_length = 50 ∧ 
  remaining_length = 20 ∧ 
  num_sections = 10 ∧ 
  section_length = 2 ∧ 
  remaining_length = num_sections * section_length →
  (total_length - remaining_length * 2) / total_length = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_bob_rope_art_fraction_l3603_360300


namespace NUMINAMATH_CALUDE_circle_area_diameter_increase_l3603_360346

theorem circle_area_diameter_increase (A D A' D' : ℝ) :
  A' = 6 * A →
  A = (π / 4) * D^2 →
  A' = (π / 4) * D'^2 →
  D' = Real.sqrt 6 * D :=
by sorry

end NUMINAMATH_CALUDE_circle_area_diameter_increase_l3603_360346


namespace NUMINAMATH_CALUDE_sum_of_digits_for_special_triangle_l3603_360321

/-- Given a positive integer n, returns the sum of its digits -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The sum of the first n natural numbers -/
def triangle_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_of_digits_for_special_triangle : 
  ∃ (N : ℕ), (triangle_sum N = 2145) ∧ (sum_of_digits N = 11) :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_for_special_triangle_l3603_360321


namespace NUMINAMATH_CALUDE_inequality_proof_l3603_360366

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 1/a + 1/b = 1) :
  ∀ n : ℕ, (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3603_360366


namespace NUMINAMATH_CALUDE_squirrel_nuts_collected_l3603_360325

/-- Represents the number of nuts eaten on day k -/
def nutsEatenOnDay (k : ℕ) : ℕ := k

/-- Represents the fraction of remaining nuts eaten each day -/
def fractionEaten : ℚ := 1 / 100

/-- Represents the number of nuts remaining before eating on day k -/
def nutsRemaining (k : ℕ) (totalNuts : ℕ) : ℕ :=
  totalNuts - (k - 1) * (k - 1 + 1) / 2

/-- Represents the number of nuts eaten on day k including the fraction -/
def totalNutsEatenOnDay (k : ℕ) (totalNuts : ℕ) : ℚ :=
  nutsEatenOnDay k + fractionEaten * (nutsRemaining k totalNuts - nutsEatenOnDay k)

/-- The theorem stating the total number of nuts collected by the squirrel -/
theorem squirrel_nuts_collected :
  ∃ n : ℕ, n > 0 ∧
    (∀ k : ℕ, k < n → totalNutsEatenOnDay k 9801 < nutsRemaining k 9801) ∧
    nutsRemaining n 9801 = n :=
  sorry

end NUMINAMATH_CALUDE_squirrel_nuts_collected_l3603_360325


namespace NUMINAMATH_CALUDE_average_of_three_liquids_l3603_360320

/-- Given the average of water and milk is 94 liters and there are 100 liters of coffee,
    prove that the average of water, milk, and coffee is 96 liters. -/
theorem average_of_three_liquids (water_milk_avg : ℝ) (coffee : ℝ) :
  water_milk_avg = 94 →
  coffee = 100 →
  (2 * water_milk_avg + coffee) / 3 = 96 := by
sorry

end NUMINAMATH_CALUDE_average_of_three_liquids_l3603_360320


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3603_360305

-- Define the sets A and B
def A : Set ℝ := {x | 2 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 3 * x - 7 ≥ 8 - 2 * x}

-- State the theorem
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x : ℝ | x ≥ 5} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3603_360305


namespace NUMINAMATH_CALUDE_select_five_from_eight_l3603_360329

theorem select_five_from_eight : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_select_five_from_eight_l3603_360329


namespace NUMINAMATH_CALUDE_system_solution_l3603_360373

theorem system_solution :
  let solutions : List (ℤ × ℤ × ℤ) := [(0, 12, 0), (2, 7, 3), (4, 2, 6)]
  ∀ x y z : ℤ,
    (x + y + z = 12 ∧ 8*x + 5*y + 3*z = 60) ↔ (x, y, z) ∈ solutions :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3603_360373


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l3603_360354

-- Define the conversion rate from paise to rupees
def paise_to_rupees (paise : ℚ) : ℚ := paise / 100

-- Define variables a and b
def a : ℚ := sorry
def b : ℚ := sorry

-- State the theorem
theorem sum_of_a_and_b : 
  (0.5 / 100 * a = paise_to_rupees 65) → 
  (1.25 / 100 * b = paise_to_rupees 104) → 
  (a + b = 213.2) := by sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l3603_360354


namespace NUMINAMATH_CALUDE_max_rectangle_area_l3603_360368

/-- Given a rectangle with perimeter 160 feet and length twice its width,
    the maximum area that can be enclosed is 12800/9 square feet. -/
theorem max_rectangle_area (w : ℝ) (l : ℝ) (h1 : w > 0) (h2 : l > 0) 
    (h3 : 2 * w + 2 * l = 160) (h4 : l = 2 * w) : w * l ≤ 12800 / 9 := by
  sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l3603_360368


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3603_360398

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- General term of the sequence
  S : ℕ → ℝ  -- Sum of the first n terms
  b : ℕ → ℝ  -- Related sequence
  h1 : a 3 = 10
  h2 : S 6 = 72
  h3 : ∀ n, b n = (1/2) * a n - 30

/-- The minimum value of the sum of the first n terms of b_n -/
def T_min (seq : ArithmeticSequence) : ℝ :=
  Finset.sum (Finset.range 15) (λ i => seq.b (i + 1))

/-- Main theorem about the arithmetic sequence and its properties -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = 4 * n - 2) ∧ T_min seq = -225 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3603_360398


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3603_360335

def arithmeticSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n => a₁ + d * (n - 1)

theorem arithmetic_sequence_general_term :
  let a₁ : ℝ := -1
  let d : ℝ := 4
  ∀ n : ℕ, arithmeticSequence a₁ d n = 4 * n - 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3603_360335


namespace NUMINAMATH_CALUDE_cookie_jar_final_amount_l3603_360315

theorem cookie_jar_final_amount : 
  let initial_amount : ℚ := 21
  let doris_spent : ℚ := 6
  let martha_spent : ℚ := doris_spent / 2
  let john_added : ℚ := 10
  let john_spent_percentage : ℚ := 1 / 4
  let final_amount : ℚ := 
    (initial_amount - doris_spent - martha_spent + john_added) * 
    (1 - john_spent_percentage)
  final_amount = 33 / 2 := by sorry

end NUMINAMATH_CALUDE_cookie_jar_final_amount_l3603_360315


namespace NUMINAMATH_CALUDE_popcorn_soda_cost_l3603_360333

/-- Calculate the total cost of popcorn and soda purchases with discounts and tax --/
theorem popcorn_soda_cost : ∃ (total_cost : ℚ),
  (let popcorn_price : ℚ := 14.7 / 5
   let soda_price : ℚ := 2
   let popcorn_quantity : ℕ := 4
   let soda_quantity : ℕ := 3
   let popcorn_discount : ℚ := 0.1
   let soda_discount : ℚ := 0.05
   let popcorn_tax : ℚ := 0.06
   let soda_tax : ℚ := 0.07

   let popcorn_subtotal : ℚ := popcorn_price * popcorn_quantity
   let soda_subtotal : ℚ := soda_price * soda_quantity

   let popcorn_discounted : ℚ := popcorn_subtotal * (1 - popcorn_discount)
   let soda_discounted : ℚ := soda_subtotal * (1 - soda_discount)

   let popcorn_total : ℚ := popcorn_discounted * (1 + popcorn_tax)
   let soda_total : ℚ := soda_discounted * (1 + soda_tax)

   total_cost = popcorn_total + soda_total) ∧
  (total_cost ≥ 17.31 ∧ total_cost < 17.33) := by
  sorry

#eval (14.7 / 5 * 4 * 0.9 * 1.06 + 2 * 3 * 0.95 * 1.07 : ℚ)

end NUMINAMATH_CALUDE_popcorn_soda_cost_l3603_360333


namespace NUMINAMATH_CALUDE_smallest_four_digit_geometric_even_l3603_360307

def is_geometric_sequence (a b c d : ℕ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r

def digits_are_distinct (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 4 ∧ digits.toFinset.card = 4

theorem smallest_four_digit_geometric_even :
  ∀ n : ℕ,
    1000 ≤ n ∧ n < 10000 ∧
    digits_are_distinct n ∧
    is_geometric_sequence (n / 1000) ((n / 100) % 10) ((n / 10) % 10) (n % 10) ∧
    Even n →
    n ≥ 1248 :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_geometric_even_l3603_360307


namespace NUMINAMATH_CALUDE_lcm_gcd_12_15_l3603_360339

theorem lcm_gcd_12_15 :
  (Nat.lcm 12 15 * Nat.gcd 12 15 = 180) ∧
  (Nat.lcm 12 15 + Nat.gcd 12 15 = 63) := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_12_15_l3603_360339


namespace NUMINAMATH_CALUDE_interest_difference_implies_sum_l3603_360369

/-- Proves that if the difference between compound interest and simple interest
    on a sum at 5% per annum for 2 years is Rs. 60, then the sum is Rs. 24,000. -/
theorem interest_difference_implies_sum (P : ℝ) : 
  P * ((1 + 0.05)^2 - 1) - P * (0.05 * 2) = 60 → P = 24000 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_implies_sum_l3603_360369


namespace NUMINAMATH_CALUDE_product_terminal_zeros_l3603_360337

/-- The number of terminal zeros in a natural number -/
def terminalZeros (n : ℕ) : ℕ := sorry

/-- The product of 50, 480, and 7 -/
def product : ℕ := 50 * 480 * 7

theorem product_terminal_zeros : terminalZeros product = 3 := by sorry

end NUMINAMATH_CALUDE_product_terminal_zeros_l3603_360337


namespace NUMINAMATH_CALUDE_system_solution_l3603_360381

/-- Given a system of linear equations and an additional equation,
    prove that k must equal 2 for the equations to have a common solution. -/
theorem system_solution (k : ℝ) : 
  (∃ x y : ℝ, x + y = 5*k ∧ x - y = k ∧ 2*x + 3*y = 24) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3603_360381


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l3603_360351

/-- The cost of one dozen pens given the ratio of pen to pencil cost and the total cost of 3 pens and 5 pencils -/
theorem cost_of_dozen_pens (pen_cost pencil_cost : ℕ) : 
  pen_cost = 5 * pencil_cost →  -- Condition 1: pen cost is 5 times pencil cost
  3 * pen_cost + 5 * pencil_cost = 240 →  -- Condition 2: total cost of 3 pens and 5 pencils
  12 * pen_cost = 720 :=  -- Conclusion: cost of one dozen pens
by
  sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l3603_360351


namespace NUMINAMATH_CALUDE_four_black_faces_symmetry_l3603_360343

/-- Represents the symmetry types of a cube. -/
inductive CubeSymmetryType
  | A
  | B1
  | B2
  | C

/-- Represents a cube with some faces painted black. -/
structure PaintedCube where
  blackFaces : Finset (Fin 6)
  blackFaceCount : blackFaces.card = 4

/-- Returns the symmetry type of a painted cube. -/
def symmetryType (cube : PaintedCube) : CubeSymmetryType :=
  sorry

/-- Theorem stating that a cube with four black faces has a symmetry type equivalent to B1 or B2. -/
theorem four_black_faces_symmetry (cube : PaintedCube) :
  symmetryType cube = CubeSymmetryType.B1 ∨ symmetryType cube = CubeSymmetryType.B2 :=
sorry

end NUMINAMATH_CALUDE_four_black_faces_symmetry_l3603_360343


namespace NUMINAMATH_CALUDE_gcd_of_squares_sum_l3603_360374

theorem gcd_of_squares_sum : Nat.gcd (123^2 + 235^2 + 347^2) (122^2 + 234^2 + 348^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_sum_l3603_360374


namespace NUMINAMATH_CALUDE_line_transformation_theorem_l3603_360348

/-- Given a line with equation y = mx + b, returns a new line with half the slope and twice the y-intercept -/
def transform_line (m b : ℚ) : ℚ × ℚ := (m / 2, 2 * b)

theorem line_transformation_theorem :
  let original_line := ((2 : ℚ) / 3, 4)
  let transformed_line := transform_line original_line.1 original_line.2
  transformed_line = ((1 : ℚ) / 3, 8) := by sorry

end NUMINAMATH_CALUDE_line_transformation_theorem_l3603_360348


namespace NUMINAMATH_CALUDE_switcheroo_period_l3603_360336

/-- Represents a word of length 2^n -/
def Word (n : ℕ) := Fin (2^n) → Char

/-- Performs a single switcheroo operation on a word -/
def switcheroo (n : ℕ) (w : Word n) : Word n :=
  sorry

/-- Returns true if two words are equal -/
def word_eq (n : ℕ) (w1 w2 : Word n) : Prop :=
  ∀ i, w1 i = w2 i

/-- Applies the switcheroo operation m times -/
def apply_switcheroo (n m : ℕ) (w : Word n) : Word n :=
  sorry

theorem switcheroo_period (n : ℕ) :
  ∀ w : Word n, word_eq n (apply_switcheroo n (2^n) w) w ∧
  ∀ m : ℕ, m < 2^n → ¬(word_eq n (apply_switcheroo n m w) w) :=
by sorry

end NUMINAMATH_CALUDE_switcheroo_period_l3603_360336


namespace NUMINAMATH_CALUDE_triangle_angle_C_l3603_360377

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if A = π/6, a = 1, and b = √3, then C = π/2 -/
theorem triangle_angle_C (A B C a b c : Real) : 
  A = π/6 → a = 1 → b = Real.sqrt 3 → 
  a / Real.sin A = b / Real.sin B →
  A + B + C = π →
  C = π/2 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l3603_360377


namespace NUMINAMATH_CALUDE_dealership_sales_theorem_l3603_360308

/-- Represents the ratio of trucks to minivans sold -/
def truck_to_minivan_ratio : ℚ := 5 / 3

/-- Number of trucks expected to be sold -/
def expected_trucks : ℕ := 45

/-- Price of each truck in dollars -/
def truck_price : ℕ := 25000

/-- Price of each minivan in dollars -/
def minivan_price : ℕ := 20000

/-- Calculates the expected number of minivans to be sold -/
def expected_minivans : ℕ := (expected_trucks * 3) / 5

/-- Calculates the total revenue from truck and minivan sales -/
def total_revenue : ℕ := expected_trucks * truck_price + expected_minivans * minivan_price

theorem dealership_sales_theorem :
  expected_minivans = 27 ∧ total_revenue = 1665000 := by
  sorry

end NUMINAMATH_CALUDE_dealership_sales_theorem_l3603_360308


namespace NUMINAMATH_CALUDE_smallest_positive_integer_l3603_360399

theorem smallest_positive_integer (a : ℝ) : 
  ∃ (b : ℤ), (∀ (x : ℝ), (x + 2) * (x + 5) * (x + 8) * (x + 11) + b > 0) ∧ 
  (∀ (c : ℤ), c < b → ∃ (y : ℝ), (y + 2) * (y + 5) * (y + 8) * (y + 11) + c ≤ 0) ∧
  b = 82 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_l3603_360399


namespace NUMINAMATH_CALUDE_archer_fish_catch_l3603_360327

def fish_problem (first_round : ℕ) (second_round_increase : ℕ) (third_round_percentage : ℕ) : Prop :=
  let second_round := first_round + second_round_increase
  let third_round := second_round + (second_round * third_round_percentage) / 100
  let total_fish := first_round + second_round + third_round
  total_fish = 60

theorem archer_fish_catch :
  fish_problem 8 12 60 :=
sorry

end NUMINAMATH_CALUDE_archer_fish_catch_l3603_360327


namespace NUMINAMATH_CALUDE_u_limit_and_bound_l3603_360359

def u : ℕ → ℚ
  | 0 => 1/3
  | n + 1 => 3 * u n - 3 * (u n)^2

theorem u_limit_and_bound : 
  (∀ k : ℕ, u k = 1/3) ∧ |u 0 - 1/3| ≤ 1/(2^1000) := by sorry

end NUMINAMATH_CALUDE_u_limit_and_bound_l3603_360359


namespace NUMINAMATH_CALUDE_construction_cost_difference_equals_profit_l3603_360330

/-- Represents the construction and sale details of houses in an area --/
structure HouseData where
  other_sale_price : ℕ
  certain_sale_multiplier : ℚ
  profit : ℕ

/-- Calculates the difference in construction cost between a certain house and other houses --/
def construction_cost_difference (data : HouseData) : ℕ :=
  data.profit

theorem construction_cost_difference_equals_profit (data : HouseData)
  (h1 : data.other_sale_price = 320000)
  (h2 : data.certain_sale_multiplier = 3/2)
  (h3 : data.profit = 60000) :
  construction_cost_difference data = data.profit := by
  sorry

#eval construction_cost_difference { other_sale_price := 320000, certain_sale_multiplier := 3/2, profit := 60000 }

end NUMINAMATH_CALUDE_construction_cost_difference_equals_profit_l3603_360330


namespace NUMINAMATH_CALUDE_sum_of_powers_l3603_360310

theorem sum_of_powers (ω : ℂ) (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  ω^20 + ω^24 + ω^28 + ω^32 + ω^36 + ω^40 + ω^44 + ω^48 + ω^52 + ω^56 + ω^60 + ω^64 + ω^68 = (ω^2 - 1) / (ω^4 - 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l3603_360310


namespace NUMINAMATH_CALUDE_train_length_l3603_360316

theorem train_length (platform_time : ℝ) (pole_time : ℝ) (platform_length : ℝ)
  (h1 : platform_time = 39)
  (h2 : pole_time = 18)
  (h3 : platform_length = 350) :
  ∃ (train_length : ℝ) (train_speed : ℝ),
    train_length = train_speed * pole_time ∧
    train_length + platform_length = train_speed * platform_time ∧
    train_length = 300 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3603_360316


namespace NUMINAMATH_CALUDE_second_group_average_l3603_360370

theorem second_group_average (n₁ : ℕ) (n₂ : ℕ) (avg₁ : ℝ) (avg_total : ℝ) :
  n₁ = 30 →
  n₂ = 20 →
  avg₁ = 20 →
  avg_total = 24 →
  ∃ avg₂ : ℝ,
    (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂) = avg_total ∧
    avg₂ = 30 := by
  sorry

end NUMINAMATH_CALUDE_second_group_average_l3603_360370


namespace NUMINAMATH_CALUDE_geometric_sum_eight_terms_l3603_360302

theorem geometric_sum_eight_terms :
  let a₀ : ℚ := 2/3
  let r : ℚ := 1/3
  let n : ℕ := 8
  let S := a₀ * (1 - r^n) / (1 - r)
  S = 6560/6561 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_eight_terms_l3603_360302
