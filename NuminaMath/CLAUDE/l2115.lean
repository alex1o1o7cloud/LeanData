import Mathlib

namespace NUMINAMATH_CALUDE_daves_coins_l2115_211542

theorem daves_coins (n : ℕ) : n > 0 ∧ 
  n % 7 = 2 ∧ 
  n % 5 = 3 ∧ 
  n % 3 = 1 ∧ 
  (∀ m : ℕ, m > 0 → m % 7 = 2 → m % 5 = 3 → m % 3 = 1 → n ≤ m) → 
  n = 58 := by
sorry

end NUMINAMATH_CALUDE_daves_coins_l2115_211542


namespace NUMINAMATH_CALUDE_factorization_implies_c_value_l2115_211564

theorem factorization_implies_c_value (c : ℝ) :
  (∀ x : ℝ, x^2 + 3*x + c = (x + 1)*(x + 2)) → c = 2 := by
sorry

end NUMINAMATH_CALUDE_factorization_implies_c_value_l2115_211564


namespace NUMINAMATH_CALUDE_man_birth_year_l2115_211502

-- Define the birth year function
def birthYear (x : ℕ) : ℕ := x^2 - x - 2

-- State the theorem
theorem man_birth_year :
  ∃ x : ℕ, 
    (birthYear x > 1900) ∧ 
    (birthYear x < 1950) ∧ 
    (birthYear x = 1890) := by
  sorry

end NUMINAMATH_CALUDE_man_birth_year_l2115_211502


namespace NUMINAMATH_CALUDE_deck_total_cost_l2115_211520

def deck_length : ℝ := 30
def deck_width : ℝ := 40
def base_cost_per_sqft : ℝ := 3
def sealant_cost_per_sqft : ℝ := 1

theorem deck_total_cost :
  deck_length * deck_width * (base_cost_per_sqft + sealant_cost_per_sqft) = 4800 := by
  sorry

end NUMINAMATH_CALUDE_deck_total_cost_l2115_211520


namespace NUMINAMATH_CALUDE_consecutive_product_square_extension_l2115_211578

theorem consecutive_product_square_extension (n : ℕ) (h : n * (n + 1) > 12) : 
  ∃! k : ℕ, k < 100 ∧ ∃ m : ℕ, 100 * (n * (n + 1)) + k = m^2 :=
sorry

end NUMINAMATH_CALUDE_consecutive_product_square_extension_l2115_211578


namespace NUMINAMATH_CALUDE_bank_deposit_l2115_211547

theorem bank_deposit (n : ℕ) (x y : ℕ) (h1 : n = 100 * x + y) (h2 : 0 ≤ y ∧ y ≤ 99) 
  (h3 : (x : ℝ) + y = 0.02 * n) : n = 4950 := by
  sorry

end NUMINAMATH_CALUDE_bank_deposit_l2115_211547


namespace NUMINAMATH_CALUDE_square_difference_equality_l2115_211567

theorem square_difference_equality : (23 + 12)^2 - (23 - 12)^2 = 1104 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l2115_211567


namespace NUMINAMATH_CALUDE_tram_length_l2115_211549

/-- The length of a tram given its passing time and tunnel transit time -/
theorem tram_length (passing_time tunnel_time tunnel_length : ℝ) 
  (h1 : passing_time = 4)
  (h2 : tunnel_time = 12)
  (h3 : tunnel_length = 64)
  (h4 : passing_time > 0)
  (h5 : tunnel_time > 0)
  (h6 : tunnel_length > 0) :
  (tunnel_length * passing_time) / (tunnel_time - passing_time) = 32 := by
  sorry

end NUMINAMATH_CALUDE_tram_length_l2115_211549


namespace NUMINAMATH_CALUDE_min_value_of_f_l2115_211507

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x - 3| + Real.exp x

theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = 6 - Real.log 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2115_211507


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2115_211518

theorem sqrt_equation_solution :
  ∃! x : ℝ, Real.sqrt ((2 + Real.sqrt 3) ^ x) + Real.sqrt ((2 - Real.sqrt 3) ^ x) = 6 ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2115_211518


namespace NUMINAMATH_CALUDE_triangle_area_l2115_211534

theorem triangle_area (A B C : ℝ × ℝ) : 
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  (BC = 8 ∧ AB = 10 ∧ AC^2 + BC^2 = AB^2) →
  (1/2 * BC * AC = 24) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2115_211534


namespace NUMINAMATH_CALUDE_no_valid_arrangement_l2115_211538

/-- Represents a circular arrangement of numbers from 1 to 60 -/
def CircularArrangement := Fin 60 → ℕ

/-- Checks if the sum of two numbers with k numbers between them is divisible by n -/
def SatisfiesDivisibilityCondition (arr : CircularArrangement) (k n : ℕ) : Prop :=
  ∀ i : Fin 60, (arr i + arr ((i + k + 1) % 60)) % n = 0

/-- Checks if the arrangement satisfies all given conditions -/
def SatisfiesAllConditions (arr : CircularArrangement) : Prop :=
  (∀ i : Fin 60, arr i ∈ Finset.range 60) ∧ 
  (Finset.card (Finset.image arr Finset.univ) = 60) ∧
  SatisfiesDivisibilityCondition arr 1 2 ∧
  SatisfiesDivisibilityCondition arr 2 3 ∧
  SatisfiesDivisibilityCondition arr 6 7

theorem no_valid_arrangement : ¬ ∃ arr : CircularArrangement, SatisfiesAllConditions arr := by
  sorry

end NUMINAMATH_CALUDE_no_valid_arrangement_l2115_211538


namespace NUMINAMATH_CALUDE_maria_trip_fraction_l2115_211558

theorem maria_trip_fraction (total_distance : ℝ) (first_stop_fraction : ℝ) (final_leg : ℝ) :
  total_distance = 480 →
  first_stop_fraction = 1/2 →
  final_leg = 180 →
  (total_distance - first_stop_fraction * total_distance - final_leg) / (total_distance - first_stop_fraction * total_distance) = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_maria_trip_fraction_l2115_211558


namespace NUMINAMATH_CALUDE_poly_arrangement_l2115_211504

/-- The original polynomial -/
def original_poly (x y : ℝ) : ℝ := 3*x*y^3 - x^2*y^3 - 9*y + x^3

/-- The polynomial arranged in ascending order of x -/
def arranged_poly (x y : ℝ) : ℝ := -9*y + 3*x*y^3 - x^2*y^3 + x^3

/-- Theorem stating that the arranged polynomial is equivalent to the original polynomial -/
theorem poly_arrangement (x y : ℝ) : original_poly x y = arranged_poly x y := by
  sorry

end NUMINAMATH_CALUDE_poly_arrangement_l2115_211504


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_101_l2115_211569

theorem gcd_of_powers_of_101 : Nat.gcd (101^11 + 1) (101^11 + 101^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_101_l2115_211569


namespace NUMINAMATH_CALUDE_height_growth_l2115_211568

theorem height_growth (current_height : ℝ) (growth_rate : ℝ) (previous_height : ℝ) : 
  current_height = 126 ∧ 
  growth_rate = 0.05 ∧ 
  current_height = previous_height * (1 + growth_rate) → 
  previous_height = 120 := by
sorry

end NUMINAMATH_CALUDE_height_growth_l2115_211568


namespace NUMINAMATH_CALUDE_remainder_problem_l2115_211573

theorem remainder_problem (n a b c : ℕ) (hn : 0 < n) 
  (ha : n % 3 = a) (hb : n % 5 = b) (hc : n % 7 = c) 
  (heq : 4 * a + 3 * b + 2 * c = 30) : 
  n % 105 = 29 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2115_211573


namespace NUMINAMATH_CALUDE_triangle_area_is_86_div_7_l2115_211505

/-- The slope of the first line -/
def m1 : ℚ := 3/4

/-- The slope of the second line -/
def m2 : ℚ := -2

/-- The x-coordinate of the intersection point of the first two lines -/
def x0 : ℚ := 1

/-- The y-coordinate of the intersection point of the first two lines -/
def y0 : ℚ := 3

/-- The equation of the third line: x + y = 8 -/
def line3 (x y : ℚ) : Prop := x + y = 8

/-- The area of the triangle formed by the three lines -/
def triangle_area : ℚ := 86/7

/-- Theorem stating that the area of the triangle is 86/7 -/
theorem triangle_area_is_86_div_7 : triangle_area = 86/7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_86_div_7_l2115_211505


namespace NUMINAMATH_CALUDE_course_size_l2115_211580

theorem course_size (total : ℕ) 
  (h1 : total / 5 + total / 4 + total / 2 + 40 = total) : total = 800 := by
  sorry

end NUMINAMATH_CALUDE_course_size_l2115_211580


namespace NUMINAMATH_CALUDE_parallelogram_area_specific_parallelogram_area_l2115_211552

/-- The area of a parallelogram with given base, side length, and included angle --/
theorem parallelogram_area (base : ℝ) (side : ℝ) (angle : ℝ) : 
  base > 0 → side > 0 → 0 < angle ∧ angle < π →
  abs (base * side * Real.sin angle - 498.465) < 0.001 := by
  sorry

/-- Specific instance of the parallelogram area theorem --/
theorem specific_parallelogram_area : 
  abs (22 * 25 * Real.sin (65 * π / 180) - 498.465) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_specific_parallelogram_area_l2115_211552


namespace NUMINAMATH_CALUDE_minimum_m_value_l2115_211560

theorem minimum_m_value (m : ℕ) : 
  (∀ n : ℕ, n ≥ 2 → (n.factorial : ℝ) ^ (2 / (n * (n - 1))) < m) ↔ m ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_minimum_m_value_l2115_211560


namespace NUMINAMATH_CALUDE_sugar_amount_l2115_211514

/-- Represents the quantities of ingredients in a bakery storage room. -/
structure BakeryStorage where
  sugar : ℝ
  flour : ℝ
  bakingSoda : ℝ
  eggs : ℝ
  chocolateChips : ℝ

/-- Represents the ratios between ingredients in the bakery storage room. -/
def BakeryRatios (s : BakeryStorage) : Prop :=
  s.sugar / s.flour = 5 / 2 ∧
  s.flour / s.bakingSoda = 10 / 1 ∧
  s.eggs / s.sugar = 3 / 4 ∧
  s.chocolateChips / s.flour = 3 / 5

/-- Represents the new ratios after adding more baking soda and chocolate chips. -/
def NewRatios (s : BakeryStorage) : Prop :=
  s.flour / (s.bakingSoda + 60) = 8 / 1 ∧
  s.eggs / s.sugar = 5 / 6

/-- Theorem stating that given the conditions, the amount of sugar is 6000 pounds. -/
theorem sugar_amount (s : BakeryStorage) 
  (h1 : BakeryRatios s) (h2 : NewRatios s) : s.sugar = 6000 := by
  sorry


end NUMINAMATH_CALUDE_sugar_amount_l2115_211514


namespace NUMINAMATH_CALUDE_investment_problem_l2115_211517

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The problem statement -/
theorem investment_problem : 
  let principal : ℝ := 4000
  let rate : ℝ := 0.1
  let time : ℕ := 2
  compound_interest principal rate time = 4840.000000000001 := by
sorry

end NUMINAMATH_CALUDE_investment_problem_l2115_211517


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2115_211544

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (a = 2 → (a - 1) * (a - 2) = 0) ∧ 
  ¬(∀ a : ℝ, (a - 1) * (a - 2) = 0 → a = 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2115_211544


namespace NUMINAMATH_CALUDE_inverse_proportion_y_relation_l2115_211503

theorem inverse_proportion_y_relation (k : ℝ) (y₁ y₂ : ℝ) 
  (h1 : k < 0) 
  (h2 : y₁ = k / (-4)) 
  (h3 : y₂ = k / (-1)) : 
  y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_y_relation_l2115_211503


namespace NUMINAMATH_CALUDE_probability_equals_three_elevenths_l2115_211523

/-- A quadruple of non-negative integers satisfying 2p + q + r + s = 4 -/
def ValidQuadruple : Type := 
  { quad : Fin 4 → ℕ // 2 * quad 0 + quad 1 + quad 2 + quad 3 = 4 }

/-- The set of all valid quadruples -/
def AllQuadruples : Finset ValidQuadruple := sorry

/-- The set of quadruples satisfying p + q + r + s = 3 -/
def SatisfyingQuadruples : Finset ValidQuadruple :=
  AllQuadruples.filter (fun quad => quad.val 0 + quad.val 1 + quad.val 2 + quad.val 3 = 3)

theorem probability_equals_three_elevenths :
  Nat.card SatisfyingQuadruples / Nat.card AllQuadruples = 3 / 11 := by sorry

end NUMINAMATH_CALUDE_probability_equals_three_elevenths_l2115_211523


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l2115_211556

/-- Given two vectors a and b in R^2, prove that their difference has magnitude 1 -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) : 
  a = (Real.cos (75 * π / 180), Real.sin (75 * π / 180)) →
  b = (Real.cos (15 * π / 180), Real.sin (15 * π / 180)) →
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l2115_211556


namespace NUMINAMATH_CALUDE_inverse_of_49_mod_89_l2115_211535

theorem inverse_of_49_mod_89 (h : (7⁻¹ : ZMod 89) = 55) : (49⁻¹ : ZMod 89) = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_49_mod_89_l2115_211535


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2115_211592

def A : Set Nat := {1, 2, 4}
def B : Set Nat := {2, 4, 6}

theorem union_of_A_and_B : A ∪ B = {1, 2, 4, 6} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2115_211592


namespace NUMINAMATH_CALUDE_player_b_wins_in_five_l2115_211557

/-- The probability that Player B wins a best-of-five series in exactly 5 matches,
    given that Player A wins each match with probability 3/4 -/
theorem player_b_wins_in_five (p : ℚ) (h : p = 3/4) :
  let q := 1 - p
  let prob_tied_after_four := 6 * q^2 * p^2
  let prob_b_wins_fifth := q
  prob_tied_after_four * prob_b_wins_fifth = 27/512 :=
by sorry

end NUMINAMATH_CALUDE_player_b_wins_in_five_l2115_211557


namespace NUMINAMATH_CALUDE_valid_combinations_for_elixir_l2115_211585

/-- Represents the number of different magical roots. -/
def num_roots : ℕ := 4

/-- Represents the number of different mystical minerals. -/
def num_minerals : ℕ := 6

/-- Represents the number of minerals incompatible with one root. -/
def minerals_incompatible_with_one_root : ℕ := 2

/-- Represents the number of roots incompatible with one mineral. -/
def roots_incompatible_with_one_mineral : ℕ := 2

/-- Represents the total number of incompatible combinations. -/
def total_incompatible_combinations : ℕ :=
  minerals_incompatible_with_one_root + roots_incompatible_with_one_mineral

/-- Theorem stating the number of valid combinations for the wizard's elixir. -/
theorem valid_combinations_for_elixir :
  num_roots * num_minerals - total_incompatible_combinations = 20 := by
  sorry

end NUMINAMATH_CALUDE_valid_combinations_for_elixir_l2115_211585


namespace NUMINAMATH_CALUDE_beidou_chip_scientific_notation_correct_l2115_211500

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The value of the "Fourth Generation Beidou Chip" size in meters -/
def beidou_chip_size : ℝ := 0.000000022

/-- The scientific notation representation of the Beidou chip size -/
def beidou_chip_scientific : ScientificNotation :=
  { coefficient := 2.2
    exponent := -8
    is_valid := by sorry }

theorem beidou_chip_scientific_notation_correct :
  beidou_chip_size = beidou_chip_scientific.coefficient * (10 : ℝ) ^ beidou_chip_scientific.exponent :=
by sorry

end NUMINAMATH_CALUDE_beidou_chip_scientific_notation_correct_l2115_211500


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l2115_211516

theorem polynomial_division_quotient :
  let dividend : Polynomial ℚ := 10 * X^3 + 20 * X^2 - 9 * X + 3
  let divisor : Polynomial ℚ := 5 * X + 3
  let quotient : Polynomial ℚ := 2 * X^2 - X
  (dividend).div divisor = quotient := by sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l2115_211516


namespace NUMINAMATH_CALUDE_mark_fruit_consumption_l2115_211529

/-- Given the total number of fruit pieces, the number kept for next week,
    and the number brought to school on Friday, calculate the number of
    pieces eaten in the first four days. -/
def fruitEatenInFourDays (total : ℕ) (keptForNextWeek : ℕ) (broughtFriday : ℕ) : ℕ :=
  total - keptForNextWeek - broughtFriday

/-- Theorem stating that given 10 pieces of fruit, if 2 are kept for next week
    and 3 are brought to school on Friday, then 5 pieces were eaten in the first four days. -/
theorem mark_fruit_consumption :
  fruitEatenInFourDays 10 2 3 = 5 := by
  sorry

#eval fruitEatenInFourDays 10 2 3

end NUMINAMATH_CALUDE_mark_fruit_consumption_l2115_211529


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l2115_211530

/-- Given a circle D with equation x^2 + 4y - 16 = -y^2 + 12x + 16,
    prove that its center (c,d) and radius s satisfy c + d + s = 4 + 6√2 -/
theorem circle_center_radius_sum (x y c d s : ℝ) : 
  (∀ x y, x^2 + 4*y - 16 = -y^2 + 12*x + 16) → 
  ((x - c)^2 + (y - d)^2 = s^2) → 
  c + d + s = 4 + 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l2115_211530


namespace NUMINAMATH_CALUDE_recurrence_sequence_a8_l2115_211536

/-- A strictly increasing sequence of positive integers satisfying the given recurrence relation. -/
def RecurrenceSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n, n ≥ 1 → a (n + 2) = a n + a (n + 1))

theorem recurrence_sequence_a8 (a : ℕ → ℕ) (h : RecurrenceSequence a) (h7 : a 7 = 120) : 
  a 8 = 194 := by
  sorry

end NUMINAMATH_CALUDE_recurrence_sequence_a8_l2115_211536


namespace NUMINAMATH_CALUDE_minimum_advantageous_discount_l2115_211509

theorem minimum_advantageous_discount (n : ℕ) : n = 29 ↔ 
  (∀ m : ℕ, m < n → 
    ((1 - m / 100 : ℝ) ≥ (1 - 0.12)^2 ∨
     (1 - m / 100 : ℝ) ≥ (1 - 0.08)^2 * (1 - 0.09) ∨
     (1 - m / 100 : ℝ) ≥ (1 - 0.20) * (1 - 0.10))) ∧
  ((1 - n / 100 : ℝ) < (1 - 0.12)^2 ∧
   (1 - n / 100 : ℝ) < (1 - 0.08)^2 * (1 - 0.09) ∧
   (1 - n / 100 : ℝ) < (1 - 0.20) * (1 - 0.10)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_advantageous_discount_l2115_211509


namespace NUMINAMATH_CALUDE_problem_statement_l2115_211574

theorem problem_statement : 
  let p := (3 + 3 = 5)
  let q := (5 > 2)
  ¬(p ∧ q) ∧ ¬p := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2115_211574


namespace NUMINAMATH_CALUDE_chocolate_difference_is_fifteen_l2115_211539

/-- The number of chocolates Nick has -/
def nick_chocolates : ℕ := 10

/-- The number of chocolates Alix initially had -/
def alix_initial_chocolates : ℕ := 3 * nick_chocolates

/-- The number of chocolates taken from Alix -/
def chocolates_taken : ℕ := 5

/-- The number of chocolates Alix has after some were taken -/
def alix_remaining_chocolates : ℕ := alix_initial_chocolates - chocolates_taken

/-- The difference in chocolates between Alix and Nick -/
def chocolate_difference : ℕ := alix_remaining_chocolates - nick_chocolates

theorem chocolate_difference_is_fifteen : chocolate_difference = 15 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_difference_is_fifteen_l2115_211539


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2115_211596

-- Define the sets M and N
def M : Set ℝ := {x | x + 1 > 0}
def N : Set ℝ := {x | 2*x - 1 < 0}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = Set.Ioo (-1 : ℝ) (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2115_211596


namespace NUMINAMATH_CALUDE_max_difference_reversed_digits_l2115_211589

/-- Two-digit positive integer -/
def TwoDigitInt (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Reverses the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem max_difference_reversed_digits (q r : ℕ) :
  TwoDigitInt q ∧ TwoDigitInt r ∧
  r = reverseDigits q ∧
  (q > r → q - r < 20) ∧
  (r > q → r - q < 20) →
  (q > r → q - r ≤ 18) ∧
  (r > q → r - q ≤ 18) :=
sorry

end NUMINAMATH_CALUDE_max_difference_reversed_digits_l2115_211589


namespace NUMINAMATH_CALUDE_sixtysecond_term_is_seven_five_l2115_211531

/-- Represents an integer pair in the sequence -/
structure IntegerPair :=
  (first : ℕ)
  (second : ℕ)

/-- Generates the nth term of the sequence -/
def sequenceTerm (n : ℕ) : IntegerPair :=
  sorry

/-- The main theorem stating that the 62nd term is (7,5) -/
theorem sixtysecond_term_is_seven_five :
  sequenceTerm 62 = IntegerPair.mk 7 5 := by
  sorry

end NUMINAMATH_CALUDE_sixtysecond_term_is_seven_five_l2115_211531


namespace NUMINAMATH_CALUDE_projection_onto_yOz_plane_l2115_211595

-- Define the types for points and vectors in 3D space
def Point3D := ℝ × ℝ × ℝ
def Vector3D := ℝ × ℝ × ℝ

-- Define the projection onto the yOz plane
def projectOntoYOZ (p : Point3D) : Point3D :=
  (0, p.2.1, p.2.2)

-- Define the vector from origin to a point
def vectorFromOrigin (p : Point3D) : Vector3D := p

-- Theorem statement
theorem projection_onto_yOz_plane (A : Point3D) (h : A = (1, 6, 2)) :
  vectorFromOrigin (projectOntoYOZ A) = (0, 6, 2) := by
  sorry

end NUMINAMATH_CALUDE_projection_onto_yOz_plane_l2115_211595


namespace NUMINAMATH_CALUDE_college_students_count_l2115_211598

theorem college_students_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 175) : boys + girls = 455 := by
  sorry

end NUMINAMATH_CALUDE_college_students_count_l2115_211598


namespace NUMINAMATH_CALUDE_f_equals_neg_tan_f_at_eight_pi_thirds_l2115_211540

/-- The function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin (Real.pi + x) * Real.cos (Real.pi - x) * Real.sin (2 * Real.pi - x)) /
  (Real.sin (Real.pi / 2 + x) * Real.cos (x - Real.pi / 2) * Real.cos (-x))

/-- Theorem stating that f(x) = -tan(x) for all x -/
theorem f_equals_neg_tan (x : ℝ) : f x = -Real.tan x := by sorry

/-- Theorem stating that f(8π/3) = -√3 -/
theorem f_at_eight_pi_thirds : f (8 * Real.pi / 3) = -Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_f_equals_neg_tan_f_at_eight_pi_thirds_l2115_211540


namespace NUMINAMATH_CALUDE_inverse_difference_evaluation_l2115_211526

theorem inverse_difference_evaluation (x y : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hxy : 5*x - 3*y ≠ 0) : 
  (5*x - 3*y)⁻¹ * ((5*x)⁻¹ - (3*y)⁻¹) = -1 / (15*x*y) := by
  sorry

end NUMINAMATH_CALUDE_inverse_difference_evaluation_l2115_211526


namespace NUMINAMATH_CALUDE_symmetric_point_l2115_211572

/-- Given a point P(2,1) and a line x - y + 1 = 0, prove that the point Q(0,3) is symmetric to P with respect to the line. -/
theorem symmetric_point (P Q : ℝ × ℝ) (line : ℝ → ℝ → ℝ) : 
  P = (2, 1) → 
  Q = (0, 3) → 
  line x y = x - y + 1 →
  (Q.1 - P.1) * (Q.2 - P.2) = -1 ∧ 
  line ((P.1 + Q.1) / 2) ((P.2 + Q.2) / 2) = 0 :=
sorry


end NUMINAMATH_CALUDE_symmetric_point_l2115_211572


namespace NUMINAMATH_CALUDE_remainder_problem_l2115_211551

theorem remainder_problem (n : ℤ) (h : n % 9 = 4) : (4 * n - 11) % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2115_211551


namespace NUMINAMATH_CALUDE_darnels_scooping_rate_l2115_211515

/-- Proves Darrel's scooping rate given the problem conditions -/
theorem darnels_scooping_rate 
  (steven_rate : ℝ) 
  (total_time : ℝ) 
  (total_load : ℝ) 
  (h1 : steven_rate = 75)
  (h2 : total_time = 30)
  (h3 : total_load = 2550) :
  ∃ (darrel_rate : ℝ), 
    (steven_rate + darrel_rate) * total_time = total_load ∧ 
    darrel_rate = 10 := by
  sorry

end NUMINAMATH_CALUDE_darnels_scooping_rate_l2115_211515


namespace NUMINAMATH_CALUDE_changsha_gdp_scientific_notation_l2115_211528

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ coefficient
  h2 : coefficient < 10

/-- The GDP value of Changsha city in 2022 -/
def changsha_gdp : ℕ := 1400000000000

/-- Converts a natural number to its scientific notation representation -/
def to_scientific_notation (n : ℕ) : ScientificNotation :=
  sorry

theorem changsha_gdp_scientific_notation :
  to_scientific_notation changsha_gdp =
    ScientificNotation.mk 1.4 12 (by norm_num) (by norm_num) :=
  sorry

end NUMINAMATH_CALUDE_changsha_gdp_scientific_notation_l2115_211528


namespace NUMINAMATH_CALUDE_parabola_line_intersection_length_l2115_211554

/-- Parabola represented by parametric equations x = 4t² and y = 4t -/
structure Parabola where
  t : ℝ
  x : ℝ := 4 * t^2
  y : ℝ := 4 * t

/-- Line with slope 1 passing through a point -/
structure Line where
  slope : ℝ := 1
  point : ℝ × ℝ

/-- Represents the intersection points of the line and the parabola -/
structure Intersection where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- The focus of a parabola with equation y² = 4x -/
def focus : ℝ × ℝ := (1, 0)

theorem parabola_line_intersection_length 
  (p : Parabola) 
  (l : Line) 
  (i : Intersection) :
  l.point = focus → 
  (∃ t₁ t₂ : ℝ, 
    i.A = (4 * t₁^2, 4 * t₁) ∧ 
    i.B = (4 * t₂^2, 4 * t₂) ∧ 
    i.A.2 = l.slope * i.A.1 + (l.point.2 - l.slope * l.point.1) ∧
    i.B.2 = l.slope * i.B.1 + (l.point.2 - l.slope * l.point.1)) →
  Real.sqrt ((i.A.1 - i.B.1)^2 + (i.A.2 - i.B.2)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_length_l2115_211554


namespace NUMINAMATH_CALUDE_system_solutions_l2115_211583

/-- The system of equations -/
def system (x₁ x₂ x₃ x₄ x₅ y : ℝ) : Prop :=
  x₅ + x₂ = y * x₁ ∧
  x₁ + x₃ = y * x₂ ∧
  x₂ + x₄ = y * x₃ ∧
  x₃ + x₅ = y * x₄ ∧
  x₄ + x₁ = y * x₃

/-- The solutions to the system of equations -/
theorem system_solutions :
  ∀ x₁ x₂ x₃ x₄ x₅ y : ℝ,
  system x₁ x₂ x₃ x₄ x₅ y →
  (y = 2 ∧ x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄ ∧ x₄ = x₅) ∨
  ((y = (-1 + Real.sqrt 5) / 2 ∨ y = (-1 - Real.sqrt 5) / 2) ∧
   x₂ = y * x₁ ∧ x₃ = y * x₂ ∧ x₄ = y * x₃) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l2115_211583


namespace NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l2115_211548

/-- Acme T-Shirt Company's pricing function -/
def acme_cost (x : ℕ) : ℝ := 75 + 12 * x

/-- Gamma T-Shirt Company's pricing function -/
def gamma_cost (x : ℕ) : ℝ := 18 * x

/-- The minimum number of shirts for which Acme is cheaper than Gamma -/
def min_shirts_for_acme_cheaper : ℕ := 13

theorem acme_cheaper_at_min_shirts :
  acme_cost min_shirts_for_acme_cheaper < gamma_cost min_shirts_for_acme_cheaper ∧
  ∀ n : ℕ, n < min_shirts_for_acme_cheaper → 
    acme_cost n ≥ gamma_cost n :=
by sorry

end NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l2115_211548


namespace NUMINAMATH_CALUDE_multiply_sum_problem_l2115_211550

theorem multiply_sum_problem (x : ℝ) (h : x = 62.5) :
  ∃! y : ℝ, ((x + 5) * y / 5) - 5 = 22 := by
  sorry

end NUMINAMATH_CALUDE_multiply_sum_problem_l2115_211550


namespace NUMINAMATH_CALUDE_triangle_area_product_l2115_211591

theorem triangle_area_product (a b : ℝ) : 
  a > 0 → b > 0 → (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ a * x + b * y = 6) → 
  (1/2 * (6/a) * (6/b) = 6) → a * b = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_product_l2115_211591


namespace NUMINAMATH_CALUDE_congruence_solution_l2115_211543

theorem congruence_solution (n : ℕ) : n = 21 → 0 ≤ n ∧ n < 47 ∧ (13 * n) % 47 = 8 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l2115_211543


namespace NUMINAMATH_CALUDE_max_vertex_sum_l2115_211511

/-- Represents a parabola passing through specific points -/
structure Parabola where
  a : ℤ
  T : ℤ
  h : T ≠ 0

/-- Calculates the sum of vertex coordinates for a given parabola -/
def vertexSum (p : Parabola) : ℚ :=
  p.T - (36 : ℚ) * p.T^2 / (2 * p.T + 2)^2

/-- Theorem stating the maximum value of the vertex sum -/
theorem max_vertex_sum :
  ∀ p : Parabola, vertexSum p ≤ (-5 : ℚ) / 4 := by sorry

end NUMINAMATH_CALUDE_max_vertex_sum_l2115_211511


namespace NUMINAMATH_CALUDE_sqrt_three_difference_of_squares_l2115_211582

theorem sqrt_three_difference_of_squares : (Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_difference_of_squares_l2115_211582


namespace NUMINAMATH_CALUDE_walnut_trees_cut_down_count_l2115_211501

/-- The number of walnut trees cut down in the park --/
def walnut_trees_cut_down (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that 13 walnut trees were cut down --/
theorem walnut_trees_cut_down_count : 
  walnut_trees_cut_down 42 29 = 13 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_cut_down_count_l2115_211501


namespace NUMINAMATH_CALUDE_only_undergraduateGraduates2013_is_well_defined_set_l2115_211571

-- Define the universe of discourse
def Universe : Type := Set (Nat → Bool)

-- Define the options
def undergraduateGraduates2013 : Universe := sorry
def highWheatProductionCities2013 : Universe := sorry
def famousMathematicians : Universe := sorry
def numbersCloseToPI : Universe := sorry

-- Define a predicate for well-defined sets
def isWellDefinedSet (S : Universe) : Prop := sorry

-- Theorem statement
theorem only_undergraduateGraduates2013_is_well_defined_set :
  isWellDefinedSet undergraduateGraduates2013 ∧
  ¬isWellDefinedSet highWheatProductionCities2013 ∧
  ¬isWellDefinedSet famousMathematicians ∧
  ¬isWellDefinedSet numbersCloseToPI :=
sorry

end NUMINAMATH_CALUDE_only_undergraduateGraduates2013_is_well_defined_set_l2115_211571


namespace NUMINAMATH_CALUDE_square_of_real_not_always_positive_l2115_211590

theorem square_of_real_not_always_positive : ¬ (∀ a : ℝ, a^2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_square_of_real_not_always_positive_l2115_211590


namespace NUMINAMATH_CALUDE_last_two_digits_sum_factorials_l2115_211577

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_factorials : ℕ → ℕ
  | 0 => 0
  | n + 1 => factorial (5 * n + 3) + sum_factorials n

theorem last_two_digits_sum_factorials :
  last_two_digits (sum_factorials 20) = 26 := by sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_factorials_l2115_211577


namespace NUMINAMATH_CALUDE_optimal_sampling_methods_l2115_211581

structure Community where
  high_income : Nat
  middle_income : Nat
  low_income : Nat

structure Survey where
  sample_size : Nat
  population_size : Nat

inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

def survey1 : Survey := {
  sample_size := 100,
  population_size := 125 + 280 + 95
}

def survey2 : Survey := {
  sample_size := 3,
  population_size := 12
}

def community : Community := {
  high_income := 125,
  middle_income := 280,
  low_income := 95
}

def optimal_sampling_method (s : Survey) (c : Option Community) : SamplingMethod :=
  sorry

theorem optimal_sampling_methods :
  optimal_sampling_method survey1 (some community) = SamplingMethod.Stratified ∧
  optimal_sampling_method survey2 none = SamplingMethod.SimpleRandom :=
sorry

end NUMINAMATH_CALUDE_optimal_sampling_methods_l2115_211581


namespace NUMINAMATH_CALUDE_sufficient_condition_l2115_211594

theorem sufficient_condition (a : ℝ) : a ≥ 0 → a^2 + a ≥ 0 := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_l2115_211594


namespace NUMINAMATH_CALUDE_orange_juice_orders_l2115_211533

/-- Proves that the number of members who ordered orange juice is 12 --/
theorem orange_juice_orders (total_members : ℕ) 
  (h1 : total_members = 30)
  (h2 : ∃ lemon_orders : ℕ, lemon_orders = (2 : ℕ) * total_members / (5 : ℕ))
  (h3 : ∃ remaining : ℕ, remaining = total_members - (2 : ℕ) * total_members / (5 : ℕ))
  (h4 : ∃ mango_orders : ℕ, mango_orders = remaining / (3 : ℕ))
  (h5 : ∃ orange_orders : ℕ, orange_orders = total_members - ((2 : ℕ) * total_members / (5 : ℕ) + remaining / (3 : ℕ))) :
  orange_orders = 12 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_orders_l2115_211533


namespace NUMINAMATH_CALUDE_karabases_more_numerous_l2115_211587

/-- Represents the inhabitants of Perra-Terra -/
inductive Inhabitant
  | Karabas
  | Barabas

/-- The number of acquaintances each type of inhabitant has -/
def acquaintances (i : Inhabitant) : Nat × Nat :=
  match i with
  | Inhabitant.Karabas => (6, 9)  -- (Karabases, Barabases)
  | Inhabitant.Barabas => (10, 7) -- (Karabases, Barabases)

theorem karabases_more_numerous :
  ∃ (K B : Nat), K > B ∧
  K * (acquaintances Inhabitant.Karabas).2 = B * (acquaintances Inhabitant.Barabas).1 :=
by sorry

end NUMINAMATH_CALUDE_karabases_more_numerous_l2115_211587


namespace NUMINAMATH_CALUDE_calculation_proofs_l2115_211532

theorem calculation_proofs :
  (4.5 * 0.9 + 5.5 * 0.9 = 9) ∧
  (1.6 * (2.25 + 10.5 / 1.5) = 14.8) ∧
  (0.36 / ((6.1 - 4.6) * 0.8) = 0.3) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proofs_l2115_211532


namespace NUMINAMATH_CALUDE_rectangle_x_satisfies_conditions_rectangle_x_unique_solution_l2115_211593

/-- The value of x for a rectangle with specific properties -/
def rectangle_x : ℝ := 1.924

/-- The length of the rectangle -/
def length (x : ℝ) : ℝ := 5 * x

/-- The width of the rectangle -/
def width (x : ℝ) : ℝ := 2 * x + 3

/-- The area of the rectangle -/
def area (x : ℝ) : ℝ := length x * width x

/-- The perimeter of the rectangle -/
def perimeter (x : ℝ) : ℝ := 2 * (length x + width x)

/-- Theorem stating that rectangle_x satisfies the given conditions -/
theorem rectangle_x_satisfies_conditions :
  area rectangle_x = 2 * perimeter rectangle_x ∧
  length rectangle_x > 0 ∧
  width rectangle_x > 0 := by
  sorry

/-- Theorem stating that rectangle_x is the unique solution -/
theorem rectangle_x_unique_solution :
  ∀ y : ℝ, (area y = 2 * perimeter y ∧ length y > 0 ∧ width y > 0) → y = rectangle_x := by
  sorry

end NUMINAMATH_CALUDE_rectangle_x_satisfies_conditions_rectangle_x_unique_solution_l2115_211593


namespace NUMINAMATH_CALUDE_bread_roll_combinations_eq_21_l2115_211527

/-- The number of ways to distribute n identical items into k distinct groups -/
def starsAndBars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of combinations of bread rolls Tom could purchase -/
def breadRollCombinations : ℕ := starsAndBars 5 3

theorem bread_roll_combinations_eq_21 : breadRollCombinations = 21 := by
  sorry

end NUMINAMATH_CALUDE_bread_roll_combinations_eq_21_l2115_211527


namespace NUMINAMATH_CALUDE_tan_product_pi_ninths_l2115_211599

theorem tan_product_pi_ninths : 
  Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_pi_ninths_l2115_211599


namespace NUMINAMATH_CALUDE_division_subtraction_problem_l2115_211522

theorem division_subtraction_problem (x : ℝ) : 
  (848 / x) - 100 = 6 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_division_subtraction_problem_l2115_211522


namespace NUMINAMATH_CALUDE_total_lives_theorem_l2115_211513

def cat_lives : ℕ := 9

def dog_lives : ℕ := cat_lives - 3

def mouse_lives : ℕ := dog_lives + 7

def elephant_lives : ℕ := 2 * cat_lives - 5

def fish_lives : ℕ := min (dog_lives + mouse_lives) (elephant_lives / 2)

theorem total_lives_theorem :
  cat_lives + dog_lives + mouse_lives + elephant_lives + fish_lives = 47 := by
  sorry

end NUMINAMATH_CALUDE_total_lives_theorem_l2115_211513


namespace NUMINAMATH_CALUDE_square_triangle_area_equality_l2115_211546

theorem square_triangle_area_equality (square_perimeter : ℝ) (triangle_height : ℝ) (x : ℝ) :
  square_perimeter = 64 →
  triangle_height = 64 →
  (square_perimeter / 4) ^ 2 = (1 / 2) * triangle_height * x →
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_area_equality_l2115_211546


namespace NUMINAMATH_CALUDE_farm_count_solution_l2115_211586

/-- Represents the count of animals in a farm -/
structure FarmCount where
  hens : ℕ
  cows : ℕ

/-- Checks if the given farm count satisfies the conditions -/
def isValidFarmCount (f : FarmCount) : Prop :=
  f.hens + f.cows = 46 ∧ 2 * f.hens + 4 * f.cows = 140

/-- Theorem stating that the farm with 22 hens satisfies the conditions -/
theorem farm_count_solution :
  ∃ (f : FarmCount), isValidFarmCount f ∧ f.hens = 22 := by
  sorry

#check farm_count_solution

end NUMINAMATH_CALUDE_farm_count_solution_l2115_211586


namespace NUMINAMATH_CALUDE_race_time_l2115_211537

/-- The time A takes to complete a 1 kilometer race, given that A can give B a start of 50 meters or 10 seconds. -/
theorem race_time : ℝ := by
  -- Define the race distance
  let race_distance : ℝ := 1000

  -- Define the head start distance
  let head_start_distance : ℝ := 50

  -- Define the head start time
  let head_start_time : ℝ := 10

  -- Define A's time to complete the race
  let time_A : ℝ := 200

  -- Prove that A's time is 200 seconds
  have h1 : race_distance / time_A * (time_A - head_start_time) = race_distance - head_start_distance := by sorry
  
  -- The final statement that proves the theorem
  exact time_A


end NUMINAMATH_CALUDE_race_time_l2115_211537


namespace NUMINAMATH_CALUDE_trig_simplification_l2115_211508

theorem trig_simplification :
  (Real.sin (20 * π / 180) + Real.sin (40 * π / 180) + Real.sin (60 * π / 180) + Real.sin (80 * π / 180)) /
  (Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.sin (30 * π / 180)) =
  8 * Real.cos (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l2115_211508


namespace NUMINAMATH_CALUDE_product_of_roots_l2115_211588

theorem product_of_roots (t : ℝ) : 
  let equation := fun t : ℝ => 18 * t^2 + 45 * t - 500
  let product_of_roots := -500 / 18
  product_of_roots = -250 / 9 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l2115_211588


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l2115_211562

theorem quadratic_equation_unique_solution (a : ℝ) (h1 : a ≠ 0) 
  (h2 : ∃! x, a * x^2 + 12 * x + 9 = 0) :
  ∃ x, a * x^2 + 12 * x + 9 = 0 ∧ x = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l2115_211562


namespace NUMINAMATH_CALUDE_problem_solution_l2115_211525

def A : Set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a < 0}

theorem problem_solution :
  (∃ a : ℝ, 
    (A ∩ B a = {x : ℝ | 1/2 ≤ x ∧ x < 2} ∧
     A ∪ B a = {x : ℝ | -2 < x ∧ x ≤ 3})) ∧
  (∀ a : ℝ, (Aᶜ ∩ B a = B a) ↔ a ≥ -1/4) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2115_211525


namespace NUMINAMATH_CALUDE_sum_of_union_elements_l2115_211506

def A : Finset ℕ := {2, 0, 1, 9}

def B : Finset ℕ := Finset.image (· * 2) A

theorem sum_of_union_elements : Finset.sum (A ∪ B) id = 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_union_elements_l2115_211506


namespace NUMINAMATH_CALUDE_rectangle_with_hole_area_l2115_211575

theorem rectangle_with_hole_area (x : ℝ) :
  let large_length : ℝ := 2*x + 9
  let large_width : ℝ := x + 6
  let hole_side : ℝ := x - 1
  let large_area : ℝ := large_length * large_width
  let hole_area : ℝ := hole_side * hole_side
  large_area - hole_area = x^2 + 23*x + 53 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_with_hole_area_l2115_211575


namespace NUMINAMATH_CALUDE_ratio_c_over_a_l2115_211510

theorem ratio_c_over_a (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_arithmetic_seq : 2 * Real.log (a * c) = Real.log (a * b) + Real.log (b * c))
  (h_relation : 4 * (a + c) = 17 * b) :
  c / a = 16 ∨ c / a = 1 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_c_over_a_l2115_211510


namespace NUMINAMATH_CALUDE_max_volume_rectangular_solid_l2115_211570

/-- Given a rectangular solid where the sum of all edges is 18 meters,
    and the length is twice the width, the maximum volume is 3 cubic meters. -/
theorem max_volume_rectangular_solid :
  ∃ (w l h : ℝ),
    w > 0 ∧ l > 0 ∧ h > 0 ∧
    l = 2 * w ∧
    4 * w + 4 * l + 4 * h = 18 ∧
    ∀ (w' l' h' : ℝ),
      w' > 0 → l' > 0 → h' > 0 →
      l' = 2 * w' →
      4 * w' + 4 * l' + 4 * h' = 18 →
      w * l * h ≥ w' * l' * h' ∧
    w * l * h = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_volume_rectangular_solid_l2115_211570


namespace NUMINAMATH_CALUDE_jessica_money_l2115_211512

theorem jessica_money (rodney ian jessica : ℕ) 
  (h1 : rodney = ian + 35)
  (h2 : ian = jessica / 2)
  (h3 : jessica = rodney + 15) : 
  jessica = 100 := by
sorry

end NUMINAMATH_CALUDE_jessica_money_l2115_211512


namespace NUMINAMATH_CALUDE_not_always_input_start_output_end_l2115_211559

/-- Represents the types of boxes in a program flowchart -/
inductive FlowchartBox
  | Start
  | Input
  | Process
  | Output
  | End

/-- Represents a program flowchart as a list of boxes -/
def Flowchart := List FlowchartBox

/-- Checks if the input box immediately follows the start box -/
def inputFollowsStart (f : Flowchart) : Prop :=
  match f with
  | FlowchartBox.Start :: FlowchartBox.Input :: _ => True
  | _ => False

/-- Checks if the output box immediately precedes the end box -/
def outputPrecedesEnd (f : Flowchart) : Prop :=
  match f.reverse with
  | FlowchartBox.End :: FlowchartBox.Output :: _ => True
  | _ => False

/-- Theorem stating that it's not always true that input must follow start
    and output must precede end in a flowchart -/
theorem not_always_input_start_output_end :
  ∃ (f : Flowchart), ¬(inputFollowsStart f ∧ outputPrecedesEnd f) :=
sorry

end NUMINAMATH_CALUDE_not_always_input_start_output_end_l2115_211559


namespace NUMINAMATH_CALUDE_fruit_tree_count_l2115_211597

/-- Proves that given 18 streets, with every other tree being a fruit tree,
    and equal numbers of three types of fruit trees,
    the number of each type of fruit tree is 3. -/
theorem fruit_tree_count (total_streets : ℕ) (fruit_tree_types : ℕ) : 
  total_streets = 18 → 
  fruit_tree_types = 3 → 
  (total_streets / 2) / fruit_tree_types = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_tree_count_l2115_211597


namespace NUMINAMATH_CALUDE_modulo_31_problem_l2115_211555

theorem modulo_31_problem : ∃! n : ℤ, 0 ≤ n ∧ n < 31 ∧ 81256 ≡ n [ZMOD 31] ∧ n = 16 := by
  sorry

end NUMINAMATH_CALUDE_modulo_31_problem_l2115_211555


namespace NUMINAMATH_CALUDE_number_pair_uniqueness_l2115_211524

theorem number_pair_uniqueness (S P : ℝ) (h : S^2 ≥ 4*P) :
  let x₁ := (S + Real.sqrt (S^2 - 4*P)) / 2
  let x₂ := (S - Real.sqrt (S^2 - 4*P)) / 2
  let y₁ := S - x₁
  let y₂ := S - x₂
  ∀ x y : ℝ, (x + y = S ∧ x * y = P) ↔ ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
by
  sorry

end NUMINAMATH_CALUDE_number_pair_uniqueness_l2115_211524


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2115_211579

/-- The equation (3x+5)(x-3) = -55 + kx has exactly one real solution if and only if k = 18 or k = -26 -/
theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3*x + 5)*(x - 3) = -55 + k*x) ↔ (k = 18 ∨ k = -26) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2115_211579


namespace NUMINAMATH_CALUDE_problems_completed_is_120_l2115_211545

/-- The number of problems completed given the conditions in the problem -/
def problems_completed (p t : ℕ) : ℕ := p * t

/-- The conditions of the problem -/
def problem_conditions (p t : ℕ) : Prop :=
  p > 15 ∧ t > 0 ∧ p * t = (3 * p - 6) * (t - 3)

/-- The theorem stating that under the given conditions, 120 problems are completed -/
theorem problems_completed_is_120 :
  ∃ p t : ℕ, problem_conditions p t ∧ problems_completed p t = 120 :=
sorry

end NUMINAMATH_CALUDE_problems_completed_is_120_l2115_211545


namespace NUMINAMATH_CALUDE_equation_solution_l2115_211563

theorem equation_solution : 
  {x : ℝ | x + 36 / (x - 5) = -12} = {-8, 3} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2115_211563


namespace NUMINAMATH_CALUDE_pool_filling_time_l2115_211519

/-- Proves the time required to fill a pool given the pool capacity, bucket size, and time per trip -/
theorem pool_filling_time 
  (pool_capacity : ℕ) 
  (bucket_size : ℕ) 
  (seconds_per_trip : ℕ) 
  (h1 : pool_capacity = 84)
  (h2 : bucket_size = 2)
  (h3 : seconds_per_trip = 20) :
  (pool_capacity / bucket_size) * seconds_per_trip / 60 = 14 := by
  sorry

#check pool_filling_time

end NUMINAMATH_CALUDE_pool_filling_time_l2115_211519


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2115_211553

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (5 + Real.sqrt x) = 4 → x = 121 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2115_211553


namespace NUMINAMATH_CALUDE_stating_distinguishable_triangles_l2115_211576

/-- Represents the number of colors available -/
def num_colors : ℕ := 8

/-- Represents the number of small triangles in the large triangle -/
def num_triangles : ℕ := 4

/-- 
Calculates the number of ways to color a large equilateral triangle 
made of 4 smaller triangles using 8 colors, where no adjacent triangles 
can have the same color.
-/
def count_colorings : ℕ := 
  num_colors * (num_colors - 1) * (num_colors - 2) * (num_colors - 3)

/-- 
Theorem stating that the number of distinguishable large equilateral triangles 
is equal to 1680.
-/
theorem distinguishable_triangles : count_colorings = 1680 := by
  sorry

end NUMINAMATH_CALUDE_stating_distinguishable_triangles_l2115_211576


namespace NUMINAMATH_CALUDE_min_sum_a_b_l2115_211561

theorem min_sum_a_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1 / a + 2 / b = 2) :
  a + b ≥ (3 + 2 * Real.sqrt 2) / 2 := by
sorry

end NUMINAMATH_CALUDE_min_sum_a_b_l2115_211561


namespace NUMINAMATH_CALUDE_girls_in_class_l2115_211565

theorem girls_in_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) (h_total : total = 35) (h_ratio : ratio_girls = 3 ∧ ratio_boys = 4) :
  ∃ (girls : ℕ), girls * ratio_boys = (total - girls) * ratio_girls ∧ girls = 15 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l2115_211565


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l2115_211541

theorem inequality_system_solution_range (a : ℝ) : 
  (∀ x : ℝ, (2 * x - 1 < 3 ∧ x - a < 0) ↔ x < a) → 
  a ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l2115_211541


namespace NUMINAMATH_CALUDE_sum_of_numbers_l2115_211521

theorem sum_of_numbers (a b c : ℝ) (ha : a = 0.8) (hb : b = 1/2) (hc : c = 0.5)
  (ga : a > 0.1) (gb : b > 0.1) (gc : c > 0.1) : a + b + c = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l2115_211521


namespace NUMINAMATH_CALUDE_point_coordinates_product_l2115_211566

theorem point_coordinates_product (y₁ y₂ : ℝ) : 
  (((4 : ℝ) - 7)^2 + (y₁ - (-3))^2 = 13^2) →
  (((4 : ℝ) - 7)^2 + (y₂ - (-3))^2 = 13^2) →
  y₁ ≠ y₂ →
  y₁ * y₂ = -151 := by
sorry

end NUMINAMATH_CALUDE_point_coordinates_product_l2115_211566


namespace NUMINAMATH_CALUDE_bottles_taken_back_l2115_211584

/-- The number of bottles Debby takes back home is equal to the number of bottles she brought minus the number of bottles drunk. -/
theorem bottles_taken_back (bottles_brought bottles_drunk : ℕ) :
  bottles_brought ≥ bottles_drunk →
  bottles_brought - bottles_drunk = bottles_brought - bottles_drunk :=
by sorry

end NUMINAMATH_CALUDE_bottles_taken_back_l2115_211584
