import Mathlib

namespace NUMINAMATH_CALUDE_single_shot_exclusivity_two_shooters_not_exclusive_hit_or_miss_exclusivity_at_least_one_not_exclusive_l344_34482

-- Define the basic events
def hits_9_rings : Prop := sorry
def hits_8_rings : Prop := sorry
def A_hits_10_rings : Prop := sorry
def B_hits_8_rings : Prop := sorry
def A_hits_target : Prop := sorry
def B_hits_target : Prop := sorry

-- Define compound events
def both_hit_target : Prop := A_hits_target ∧ B_hits_target
def neither_hit_target : Prop := ¬A_hits_target ∧ ¬B_hits_target
def at_least_one_hits : Prop := A_hits_target ∨ B_hits_target
def A_misses_B_hits : Prop := ¬A_hits_target ∧ B_hits_target

-- Define mutual exclusivity
def mutually_exclusive (p q : Prop) : Prop := ¬(p ∧ q)

-- Theorem statements
theorem single_shot_exclusivity : 
  mutually_exclusive hits_9_rings hits_8_rings := by sorry

theorem two_shooters_not_exclusive : 
  ¬(mutually_exclusive A_hits_10_rings B_hits_8_rings) := by sorry

theorem hit_or_miss_exclusivity : 
  mutually_exclusive both_hit_target neither_hit_target := by sorry

theorem at_least_one_not_exclusive : 
  ¬(mutually_exclusive at_least_one_hits A_misses_B_hits) := by sorry

end NUMINAMATH_CALUDE_single_shot_exclusivity_two_shooters_not_exclusive_hit_or_miss_exclusivity_at_least_one_not_exclusive_l344_34482


namespace NUMINAMATH_CALUDE_donation_theorem_l344_34477

/-- The number of 8th grade students who donated books -/
def eighth_grade_donors : ℕ := 450

/-- The number of 7th grade students who donated books -/
def seventh_grade_donors : ℕ := eighth_grade_donors - 150

/-- The total number of books donated by both grades -/
def total_books : ℕ := 1800

/-- The average number of books donated per 8th grade student -/
def eighth_grade_avg : ℚ := total_books / eighth_grade_donors

/-- The average number of books donated per 7th grade student -/
def seventh_grade_avg : ℚ := 1.5 * eighth_grade_avg

theorem donation_theorem :
  eighth_grade_donors = 450 ∧
  seventh_grade_donors = eighth_grade_donors - 150 ∧
  total_books = 1800 ∧
  seventh_grade_avg = 1.5 * eighth_grade_avg ∧
  total_books = eighth_grade_donors * eighth_grade_avg + seventh_grade_donors * seventh_grade_avg :=
by sorry

end NUMINAMATH_CALUDE_donation_theorem_l344_34477


namespace NUMINAMATH_CALUDE_sqrt_product_quotient_l344_34442

theorem sqrt_product_quotient : Real.sqrt 3 * Real.sqrt 10 / Real.sqrt 6 = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_quotient_l344_34442


namespace NUMINAMATH_CALUDE_sum_of_first_12_mod_9_l344_34421

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_of_first_12_mod_9 : sum_of_first_n 12 % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_12_mod_9_l344_34421


namespace NUMINAMATH_CALUDE_words_with_consonant_count_l344_34498

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 6

/-- The number of vowels in the alphabet -/
def vowel_count : ℕ := 2

/-- The length of the words we're considering -/
def word_length : ℕ := 5

/-- The total number of possible words -/
def total_words : ℕ := alphabet_size ^ word_length

/-- The number of words containing only vowels -/
def all_vowel_words : ℕ := vowel_count ^ word_length

/-- The number of words with at least one consonant -/
def words_with_consonant : ℕ := total_words - all_vowel_words

theorem words_with_consonant_count : words_with_consonant = 7744 := by
  sorry

end NUMINAMATH_CALUDE_words_with_consonant_count_l344_34498


namespace NUMINAMATH_CALUDE_arc_length_of_sector_l344_34468

/-- The arc length of a sector with radius 8 cm and central angle 45° is 2π cm. -/
theorem arc_length_of_sector (r : ℝ) (θ_deg : ℝ) (l : ℝ) : 
  r = 8 → θ_deg = 45 → l = r * (θ_deg * π / 180) → l = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_arc_length_of_sector_l344_34468


namespace NUMINAMATH_CALUDE_expression_simplification_l344_34485

theorem expression_simplification (a b c : ℝ) (ha : a = 12) (hb : b = 14) (hc : c = 18) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) / 
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l344_34485


namespace NUMINAMATH_CALUDE_cycle_gain_percent_l344_34433

/-- Calculates the gain percent given the cost price and selling price -/
def gainPercent (costPrice sellingPrice : ℚ) : ℚ :=
  ((sellingPrice - costPrice) / costPrice) * 100

/-- Proves that the gain percent on a cycle bought for Rs. 930 and sold for Rs. 1210 is approximately 30.11% -/
theorem cycle_gain_percent :
  let costPrice : ℚ := 930
  let sellingPrice : ℚ := 1210
  abs (gainPercent costPrice sellingPrice - 30.11) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_cycle_gain_percent_l344_34433


namespace NUMINAMATH_CALUDE_special_polygon_properties_l344_34412

/-- A polygon where the sum of interior angles is twice the sum of exterior angles -/
structure SpecialPolygon where
  sides : ℕ
  sum_interior_angles : ℝ
  sum_exterior_angles : ℝ
  interior_exterior_relation : sum_interior_angles = 2 * sum_exterior_angles

theorem special_polygon_properties (p : SpecialPolygon) :
  p.sum_interior_angles = 720 ∧ p.sides = 6 := by
  sorry

end NUMINAMATH_CALUDE_special_polygon_properties_l344_34412


namespace NUMINAMATH_CALUDE_find_S_value_l344_34488

-- Define the relationship between R, S, and T
def relationship (R S T : ℝ) : Prop :=
  ∃ c : ℝ, c > 0 ∧ ∀ R S T, R = c * S^2 / T

-- Define the initial condition
def initial_condition (R S T : ℝ) : Prop :=
  R = 2 ∧ S = 1 ∧ T = 3

-- Theorem to prove
theorem find_S_value (R S T : ℝ) :
  relationship R S T →
  initial_condition R S T →
  R = 18 ∧ T = 2 →
  S = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_find_S_value_l344_34488


namespace NUMINAMATH_CALUDE_cubic_root_difference_l344_34470

theorem cubic_root_difference : ∃ (r₁ r₂ r₃ : ℝ),
  (∀ x : ℝ, x^3 - 7*x^2 + 11*x - 6 = 0 ↔ (x = r₁ ∨ x = r₂ ∨ x = r₃)) ∧
  max r₁ (max r₂ r₃) - min r₁ (min r₂ r₃) = 2 :=
sorry

end NUMINAMATH_CALUDE_cubic_root_difference_l344_34470


namespace NUMINAMATH_CALUDE_reading_time_difference_l344_34413

/-- The difference in reading time between two people reading the same book -/
theorem reading_time_difference (xanthia_rate molly_rate book_pages : ℕ) : 
  xanthia_rate = 150 → 
  molly_rate = 75 → 
  book_pages = 300 → 
  (book_pages / molly_rate - book_pages / xanthia_rate) * 60 = 120 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_difference_l344_34413


namespace NUMINAMATH_CALUDE_overlap_area_theorem_l344_34460

/-- Represents a square sheet of paper -/
structure Sheet :=
  (side : ℝ)
  (rotation : ℝ)

/-- Calculates the area of overlap between three rotated square sheets -/
def area_of_overlap (s1 s2 s3 : Sheet) : ℝ :=
  sorry

/-- The theorem stating the area of overlap for the given problem -/
theorem overlap_area_theorem :
  let s1 : Sheet := { side := 8, rotation := 0 }
  let s2 : Sheet := { side := 8, rotation := 45 }
  let s3 : Sheet := { side := 8, rotation := 90 }
  area_of_overlap s1 s2 s3 = 96 :=
sorry

end NUMINAMATH_CALUDE_overlap_area_theorem_l344_34460


namespace NUMINAMATH_CALUDE_largest_power_of_seven_divisor_l344_34448

theorem largest_power_of_seven_divisor : ∃ (n : ℕ), 
  (∀ (k : ℕ), 7^k ∣ (Nat.factorial 200 / (Nat.factorial 90 * Nat.factorial 30)) → k ≤ n) ∧
  (7^n ∣ (Nat.factorial 200 / (Nat.factorial 90 * Nat.factorial 30))) ∧
  n = 15 := by
  sorry

end NUMINAMATH_CALUDE_largest_power_of_seven_divisor_l344_34448


namespace NUMINAMATH_CALUDE_f_six_of_two_l344_34494

def f (x : ℝ) : ℝ := 3 * x - 1

def f_iter (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n + 1 => f (f_iter n x)

theorem f_six_of_two : f_iter 6 2 = 1094 := by sorry

end NUMINAMATH_CALUDE_f_six_of_two_l344_34494


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l344_34418

theorem arithmetic_calculation : 10 * 1.8 - 2 * 1.5 / 0.3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l344_34418


namespace NUMINAMATH_CALUDE_mixture_problem_l344_34403

/-- Represents the quantities of milk, water, and juice in a mixture --/
structure Mixture where
  milk : ℝ
  water : ℝ
  juice : ℝ

/-- Calculates the total quantity of a mixture --/
def totalQuantity (m : Mixture) : ℝ := m.milk + m.water + m.juice

/-- Checks if the given quantities form the specified ratio --/
def isRatio (m : Mixture) (r : Mixture) : Prop :=
  m.milk / r.milk = m.water / r.water ∧ m.milk / r.milk = m.juice / r.juice

/-- The main theorem to prove --/
theorem mixture_problem (initial : Mixture) (final : Mixture) : 
  isRatio initial ⟨5, 3, 4⟩ → 
  final.milk = initial.milk ∧ 
  final.water = initial.water + 12 ∧ 
  final.juice = initial.juice + 6 →
  isRatio final ⟨5, 9, 8⟩ →
  totalQuantity initial = 24 := by
  sorry

end NUMINAMATH_CALUDE_mixture_problem_l344_34403


namespace NUMINAMATH_CALUDE_circle_area_difference_l344_34465

theorem circle_area_difference : 
  let r1 : ℝ := 30
  let c2 : ℝ := 30
  let area1 := π * r1^2
  let r2 := c2 / (2 * π)
  let area2 := π * r2^2
  area1 - area2 = (225 * (4 * π^2 - 1)) / π := by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l344_34465


namespace NUMINAMATH_CALUDE_beau_current_age_l344_34496

/-- Represents a person with an age -/
structure Person where
  age : ℕ

/-- Represents Beau and his three sons -/
structure Family where
  beau : Person
  son1 : Person
  son2 : Person
  son3 : Person

/-- The age of Beau's sons today -/
def sonAgeToday : ℕ := 16

/-- The theorem stating Beau's current age -/
theorem beau_current_age (f : Family) : 
  (f.son1.age = sonAgeToday) ∧ 
  (f.son2.age = sonAgeToday) ∧ 
  (f.son3.age = sonAgeToday) ∧ 
  (f.beau.age = f.son1.age + f.son2.age + f.son3.age + 3) → 
  f.beau.age = 42 := by
  sorry


end NUMINAMATH_CALUDE_beau_current_age_l344_34496


namespace NUMINAMATH_CALUDE_arithmetic_mean_lower_bound_l344_34435

theorem arithmetic_mean_lower_bound (a₁ a₂ a₃ : ℝ) 
  (h_positive : a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0) 
  (h_sum : 2*a₁ + 3*a₂ + a₃ = 1) : 
  (1/(a₁ + a₂) + 1/(a₂ + a₃)) / 2 ≥ (3 + 2*Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_lower_bound_l344_34435


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l344_34466

def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem intersection_complement_theorem :
  N ∩ (Set.univ \ M) = {x | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l344_34466


namespace NUMINAMATH_CALUDE_longest_side_is_72_l344_34452

def rectangle_problem (length width : ℝ) : Prop :=
  length > 0 ∧ 
  width > 0 ∧ 
  2 * (length + width) = 240 ∧ 
  length * width = 12 * 240

theorem longest_side_is_72 : 
  ∃ (length width : ℝ), 
    rectangle_problem length width ∧ 
    (length ≥ width → length = 72) ∧
    (width > length → width = 72) :=
sorry

end NUMINAMATH_CALUDE_longest_side_is_72_l344_34452


namespace NUMINAMATH_CALUDE_sum_rounded_to_hundredth_l344_34491

-- Define the repeating decimals
def repeating_decimal_37 : ℚ := 37 + 37 / 99
def repeating_decimal_15 : ℚ := 15 + 15 / 99

-- Define the sum of the repeating decimals
def sum : ℚ := repeating_decimal_37 + repeating_decimal_15

-- Define a function to round to the nearest hundredth
def round_to_hundredth (x : ℚ) : ℚ := 
  ⌊x * 100 + 1/2⌋ / 100

-- Theorem statement
theorem sum_rounded_to_hundredth : 
  round_to_hundredth sum = 52 / 100 := by sorry

end NUMINAMATH_CALUDE_sum_rounded_to_hundredth_l344_34491


namespace NUMINAMATH_CALUDE_distance_to_nearest_town_l344_34416

theorem distance_to_nearest_town (d : ℝ) : 
  (¬ (d ≥ 8)) ∧ (¬ (d ≤ 7)) ∧ (¬ (d ≤ 6)) ∧ (¬ (d ≥ 5)) → 
  d ∈ Set.Ioo 7 8 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_nearest_town_l344_34416


namespace NUMINAMATH_CALUDE_division_and_addition_l344_34400

theorem division_and_addition : (-300) / (-75) + 10 = 14 := by
  sorry

end NUMINAMATH_CALUDE_division_and_addition_l344_34400


namespace NUMINAMATH_CALUDE_custom_op_theorem_l344_34406

/-- Custom operation ã — -/
def custom_op (a b : ℝ) : ℝ := 2 * a - 3 * b + a * b

theorem custom_op_theorem :
  ∃ X : ℝ, X + 2 * (custom_op 1 3) = 7 →
  3 * (custom_op 1 2) = 12 * 1 - 18 := by
sorry

end NUMINAMATH_CALUDE_custom_op_theorem_l344_34406


namespace NUMINAMATH_CALUDE_existence_of_triple_l344_34438

theorem existence_of_triple (n : ℕ) :
  let A := Finset.range (2^(n+1))
  ∀ S : Finset ℕ, S ⊆ A → S.card = 2*n + 1 →
    ∃ a b c : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
      a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
      (b * c : ℝ) < 2 * (a^2 : ℝ) ∧ 2 * (a^2 : ℝ) < 4 * (b * c : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_triple_l344_34438


namespace NUMINAMATH_CALUDE_line_equation_proof_l344_34415

/-- A line in the 2D plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.yIntercept

theorem line_equation_proof (l : Line) (p : Point) :
  l.slope = 2 ∧ p = ⟨0, 3⟩ ∧ pointOnLine l p →
  l = ⟨2, 3⟩ :=
sorry

end NUMINAMATH_CALUDE_line_equation_proof_l344_34415


namespace NUMINAMATH_CALUDE_greatest_divisor_with_digit_sum_l344_34449

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem greatest_divisor_with_digit_sum (a b : ℕ) (ha : a = 4665) (hb : b = 6905) :
  ∃ (n : ℕ), n = 40 ∧ 
  (b - a) % n = 0 ∧
  sum_of_digits n = 4 ∧
  ∀ (m : ℕ), m > n → ((b - a) % m = 0 → sum_of_digits m ≠ 4) :=
sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_digit_sum_l344_34449


namespace NUMINAMATH_CALUDE_contest_sequences_equal_combination_l344_34462

/-- Represents the number of players in each team -/
def team_size : ℕ := 7

/-- Represents the total number of players from both teams -/
def total_players : ℕ := 2 * team_size

/-- Represents the number of different possible sequences of matches in the contest -/
def match_sequences : ℕ := Nat.choose total_players team_size

theorem contest_sequences_equal_combination :
  match_sequences = 3432 := by
  sorry

end NUMINAMATH_CALUDE_contest_sequences_equal_combination_l344_34462


namespace NUMINAMATH_CALUDE_right_triangle_area_and_perimeter_l344_34493

theorem right_triangle_area_and_perimeter : 
  ∀ (triangle : Set ℝ) (leg1 leg2 hypotenuse : ℝ),
  -- Conditions
  leg1 = 30 →
  leg2 = 45 →
  hypotenuse^2 = leg1^2 + leg2^2 →
  -- Definitions
  let area := (1/2) * leg1 * leg2
  let perimeter := leg1 + leg2 + hypotenuse
  -- Theorem
  area = 675 ∧ perimeter = 129 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_and_perimeter_l344_34493


namespace NUMINAMATH_CALUDE_fraction_equality_l344_34432

theorem fraction_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x / y + y / x = 4) : 
  x * y / (x^2 - y^2) = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l344_34432


namespace NUMINAMATH_CALUDE_half_filled_cylindrical_tank_volume_l344_34404

/-- The volume of water in a half-filled cylindrical tank lying on its side -/
theorem half_filled_cylindrical_tank_volume
  (r : ℝ) -- radius of the tank
  (h : ℝ) -- height (length) of the tank
  (hr : r = 5) -- given radius is 5 feet
  (hh : h = 10) -- given height is 10 feet
  : (1 / 2 * π * r^2 * h) = 125 * π := by
  sorry

end NUMINAMATH_CALUDE_half_filled_cylindrical_tank_volume_l344_34404


namespace NUMINAMATH_CALUDE_parallel_vectors_trig_identity_l344_34402

/-- Given vectors a and b where a is parallel to b, prove that 2sin(α)cos(α) = -4/5 -/
theorem parallel_vectors_trig_identity (α : ℝ) :
  let a : ℝ × ℝ := (Real.cos α, -2)
  let b : ℝ × ℝ := (Real.sin α, 1)
  (∃ (k : ℝ), a = k • b) →
  2 * Real.sin α * Real.cos α = -4/5 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_trig_identity_l344_34402


namespace NUMINAMATH_CALUDE_three_to_six_minus_one_prime_factors_l344_34434

theorem three_to_six_minus_one_prime_factors :
  let n := 3^6 - 1
  ∃ (p q r : Nat), Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    n % p = 0 ∧ n % q = 0 ∧ n % r = 0 ∧
    (∀ (s : Nat), Nat.Prime s → n % s = 0 → s = p ∨ s = q ∨ s = r) ∧
    p + q + r = 22 :=
by sorry

end NUMINAMATH_CALUDE_three_to_six_minus_one_prime_factors_l344_34434


namespace NUMINAMATH_CALUDE_translation_result_l344_34492

-- Define the properties of a triangle
structure Triangle :=
  (shape : Type)
  (size : ℝ)
  (orientation : ℝ)

-- Define the translation operation
def translate (t : Triangle) : Triangle := t

-- Define the given shaded triangle
def shaded_triangle : Triangle := sorry

-- Define the options A, B, C, D, E
def option_A : Triangle := sorry
def option_B : Triangle := sorry
def option_C : Triangle := sorry
def option_D : Triangle := sorry
def option_E : Triangle := sorry

-- State the theorem
theorem translation_result :
  ∀ (t : Triangle),
    translate t = t →
    translate shaded_triangle = option_D :=
by sorry

end NUMINAMATH_CALUDE_translation_result_l344_34492


namespace NUMINAMATH_CALUDE_smallest_a_inequality_l344_34495

theorem smallest_a_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 1) :
  ∀ a : ℝ, a ≥ 2/9 →
    a * (x^2 + y^2 + z^2) + x * y * z ≥ a / 3 + 1 / 27 ∧
    ∀ b : ℝ, b < 2/9 →
      ∃ x' y' z' : ℝ, x' ≥ 0 ∧ y' ≥ 0 ∧ z' ≥ 0 ∧ x' + y' + z' = 1 ∧
        b * (x'^2 + y'^2 + z'^2) + x' * y' * z' < b / 3 + 1 / 27 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_inequality_l344_34495


namespace NUMINAMATH_CALUDE_plant_is_red_daisy_l344_34441

structure Plant where
  color : String
  type : String

structure Statement where
  person : String
  plant : Plant

def is_partially_correct (actual : Plant) (statement : Statement) : Prop :=
  (actual.color = statement.plant.color) ≠ (actual.type = statement.plant.type)

theorem plant_is_red_daisy (actual : Plant) 
  (anika_statement : Statement)
  (bill_statement : Statement)
  (cathy_statement : Statement)
  (h1 : anika_statement.person = "Anika" ∧ anika_statement.plant = ⟨"red", "rose"⟩)
  (h2 : bill_statement.person = "Bill" ∧ bill_statement.plant = ⟨"purple", "daisy"⟩)
  (h3 : cathy_statement.person = "Cathy" ∧ cathy_statement.plant = ⟨"red", "dahlia"⟩)
  (h4 : is_partially_correct actual anika_statement)
  (h5 : is_partially_correct actual bill_statement)
  (h6 : is_partially_correct actual cathy_statement)
  : actual = ⟨"red", "daisy"⟩ := by
  sorry

end NUMINAMATH_CALUDE_plant_is_red_daisy_l344_34441


namespace NUMINAMATH_CALUDE_no_solution_when_k_is_seven_l344_34469

theorem no_solution_when_k_is_seven (k : ℝ) (h : k = 7) :
  ¬ ∃ x : ℝ, x ≠ 3 ∧ x ≠ 5 ∧ (x^2 - 1) / (x - 3) = (x^2 - k) / (x - 5) :=
by
  sorry

end NUMINAMATH_CALUDE_no_solution_when_k_is_seven_l344_34469


namespace NUMINAMATH_CALUDE_mans_speed_with_current_is_15_l344_34497

/-- 
Given a current speed and a man's speed against the current,
calculate the man's speed with the current.
-/
def mans_speed_with_current (current_speed : ℝ) (speed_against_current : ℝ) : ℝ :=
  speed_against_current + 2 * current_speed

/-- 
Theorem: Given a current speed of 2.5 km/hr and a speed against the current of 10 km/hr,
the man's speed with the current is 15 km/hr.
-/
theorem mans_speed_with_current_is_15 :
  mans_speed_with_current 2.5 10 = 15 := by
  sorry

#eval mans_speed_with_current 2.5 10

end NUMINAMATH_CALUDE_mans_speed_with_current_is_15_l344_34497


namespace NUMINAMATH_CALUDE_water_difference_before_exchange_l344_34490

/-- The difference in water amounts before the exchange, given the conditions of the problem -/
theorem water_difference_before_exchange 
  (S H : ℝ) -- S and H represent the initial amounts of water for Seungmin and Hyoju
  (h1 : S > H) -- Seungmin has more water than Hyoju
  (h2 : S - 0.43 - (H + 0.43) = 0.88) -- Difference after exchange
  : S - H = 1.74 := by sorry

end NUMINAMATH_CALUDE_water_difference_before_exchange_l344_34490


namespace NUMINAMATH_CALUDE_triangle_properties_l344_34499

/-- Triangle with sides a, b, c opposite to angles A, B, C --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about properties of a specific triangle --/
theorem triangle_properties (t : Triangle) 
  (h_acute : t.A > 0 ∧ t.A < π/2 ∧ t.B > 0 ∧ t.B < π/2 ∧ t.C > 0 ∧ t.C < π/2)
  (h_cosine : t.a * Real.cos t.A + t.b * Real.cos t.B = t.c) :
  (t.a = t.b) ∧ 
  (∀ (circumcircle_area : ℝ), circumcircle_area = π → 
    7 < (3 * t.b^2 + t.b + 4 * t.c) / t.a ∧ 
    (3 * t.b^2 + t.b + 4 * t.c) / t.a < 7 * Real.sqrt 2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l344_34499


namespace NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_l344_34427

theorem quadratic_equation_distinct_roots (k : ℝ) :
  k = 1 → ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 2 * x₁^2 - k = 0 ∧ 2 * x₂^2 - k = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_l344_34427


namespace NUMINAMATH_CALUDE_reciprocal_opposite_equation_l344_34481

theorem reciprocal_opposite_equation (a b c d : ℝ) 
  (h1 : a * b = 1)  -- a and b are reciprocals
  (h2 : c + d = 0)  -- c and d are opposites
  : (a * b) ^ 4 - 3 * (c + d) ^ 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_opposite_equation_l344_34481


namespace NUMINAMATH_CALUDE_unique_integer_divisible_by_18_with_sqrt_between_28_and_28_2_l344_34443

theorem unique_integer_divisible_by_18_with_sqrt_between_28_and_28_2 :
  ∃! n : ℕ+, 
    (∃ k : ℕ, n = 18 * k) ∧ 
    (28 < (n : ℝ).sqrt ∧ (n : ℝ).sqrt < 28.2) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_integer_divisible_by_18_with_sqrt_between_28_and_28_2_l344_34443


namespace NUMINAMATH_CALUDE_n_fourth_plus_four_composite_l344_34479

theorem n_fourth_plus_four_composite (n : ℕ) (h : n > 1) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^4 + 4 = a * b :=
by sorry

end NUMINAMATH_CALUDE_n_fourth_plus_four_composite_l344_34479


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l344_34419

theorem complex_fraction_simplification :
  (7 + 16 * Complex.I) / (3 - 4 * Complex.I) = 6 - (38 / 7) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l344_34419


namespace NUMINAMATH_CALUDE_find_b_l344_34428

-- Define the ratio relationship
def ratio_relation (x y z : ℚ) : Prop :=
  ∃ (k : ℚ), x = 4 * k ∧ y = 3 * k ∧ z = 7 * k

-- Define the main theorem
theorem find_b (x y z b : ℚ) :
  ratio_relation x y z →
  y = 15 * b - 5 * z + 25 →
  z = 21 →
  b = 89 / 15 := by
  sorry


end NUMINAMATH_CALUDE_find_b_l344_34428


namespace NUMINAMATH_CALUDE_expected_score_is_80_l344_34446

/-- A math test with multiple-choice questions -/
structure MathTest where
  num_questions : ℕ
  points_per_correct : ℕ
  prob_correct : ℝ

/-- Expected score for a math test -/
def expected_score (test : MathTest) : ℝ :=
  test.num_questions * test.points_per_correct * test.prob_correct

/-- Theorem: The expected score for the given test is 80 points -/
theorem expected_score_is_80 (test : MathTest) 
    (h1 : test.num_questions = 25)
    (h2 : test.points_per_correct = 4)
    (h3 : test.prob_correct = 0.8) : 
  expected_score test = 80 := by
  sorry

end NUMINAMATH_CALUDE_expected_score_is_80_l344_34446


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l344_34444

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set Nat := {1, 2}

-- Define set B
def B : Set Nat := {0, 2, 5}

-- State the theorem
theorem complement_intersection_theorem :
  (U \ A) ∩ B = {0, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l344_34444


namespace NUMINAMATH_CALUDE_quadratic_real_root_l344_34423

theorem quadratic_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_root_l344_34423


namespace NUMINAMATH_CALUDE_equation_solution_l344_34483

theorem equation_solution : 
  ∀ x : ℝ, 
    (((x + 1)^2 + 1) / (x + 1) + ((x + 4)^2 + 4) / (x + 4) = 
     ((x + 2)^2 + 2) / (x + 2) + ((x + 3)^2 + 3) / (x + 3)) ↔ 
    (x = 0 ∨ x = -5/2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l344_34483


namespace NUMINAMATH_CALUDE_cat_food_percentage_l344_34420

/-- Proves that given 7 dogs and 4 cats, where all dogs receive equal amounts of food,
    all cats receive equal amounts of food, and the total food for all cats equals
    the food for one dog, the percentage of total food that one cat receives is 1/32. -/
theorem cat_food_percentage :
  ∀ (dog_food cat_food : ℚ),
  dog_food > 0 →
  cat_food > 0 →
  4 * cat_food = dog_food →
  (cat_food / (7 * dog_food + 4 * cat_food)) = 1 / 32 :=
by
  sorry

end NUMINAMATH_CALUDE_cat_food_percentage_l344_34420


namespace NUMINAMATH_CALUDE_broken_marbles_percentage_l344_34447

theorem broken_marbles_percentage (total_broken : ℕ) (set1_count : ℕ) (set2_count : ℕ) (set2_broken_percent : ℚ) :
  total_broken = 17 →
  set1_count = 50 →
  set2_count = 60 →
  set2_broken_percent = 20 / 100 →
  ∃ (set1_broken_percent : ℚ),
    set1_broken_percent = 10 / 100 ∧
    total_broken = set1_broken_percent * set1_count + set2_broken_percent * set2_count :=
by sorry

end NUMINAMATH_CALUDE_broken_marbles_percentage_l344_34447


namespace NUMINAMATH_CALUDE_quadratic_min_iff_m_gt_neg_one_l344_34445

/-- A quadratic function with coefficient (m + 1) has a minimum value if and only if m > -1 -/
theorem quadratic_min_iff_m_gt_neg_one (m : ℝ) :
  (∃ (min : ℝ), ∀ (x : ℝ), (m + 1) * x^2 ≥ min) ↔ m > -1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_min_iff_m_gt_neg_one_l344_34445


namespace NUMINAMATH_CALUDE_john_finish_time_l344_34464

/-- The time it takes for John to finish the job by himself -/
def john_time : ℝ := 1.5

/-- The time it takes for David to finish the job by himself -/
def david_time : ℝ := 2 * john_time

/-- The time it takes for John and David to finish the job together -/
def combined_time : ℝ := 1

theorem john_finish_time :
  (1 / john_time + 1 / david_time) * combined_time = 1 ∧ david_time = 2 * john_time → john_time = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_john_finish_time_l344_34464


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l344_34459

theorem imaginary_part_of_z (a : ℝ) (h1 : a > 0) (h2 : Complex.abs (Complex.mk 1 a) = Real.sqrt 5) :
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l344_34459


namespace NUMINAMATH_CALUDE_sum_of_m_values_l344_34471

/-- A triangle with vertices at (0,0), (2,2), and (8m,0) is divided into two equal areas by a line y = mx. -/
def Triangle (m : ℝ) := {A : ℝ × ℝ | A = (0, 0) ∨ A = (2, 2) ∨ A = (8*m, 0)}

/-- The line that divides the triangle into two equal areas -/
def DividingLine (m : ℝ) := {(x, y) : ℝ × ℝ | y = m * x}

/-- The condition that the line divides the triangle into two equal areas -/
def EqualAreasCondition (m : ℝ) : Prop := 
  ∃ (x : ℝ), (x, m*x) ∈ DividingLine m ∧ 
  (x = 4*m + 1) ∧ (m*x = 1)

/-- The theorem stating that the sum of all possible values of m is -1/4 -/
theorem sum_of_m_values (m₁ m₂ : ℝ) : 
  (EqualAreasCondition m₁ ∧ EqualAreasCondition m₂ ∧ m₁ ≠ m₂) → 
  m₁ + m₂ = -1/4 := by sorry

end NUMINAMATH_CALUDE_sum_of_m_values_l344_34471


namespace NUMINAMATH_CALUDE_hema_rahul_ratio_l344_34425

-- Define variables for ages
variable (Raj Ravi Hema Rahul : ℚ)

-- Define the conditions
axiom raj_older : Raj = Ravi + 3
axiom hema_younger : Hema = Ravi - 2
axiom raj_triple : Raj = 3 * Rahul
axiom raj_twenty : Raj = 20
axiom raj_hema_ratio : Raj = Hema + (1/3) * Hema

-- Theorem to prove
theorem hema_rahul_ratio : Hema / Rahul = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_hema_rahul_ratio_l344_34425


namespace NUMINAMATH_CALUDE_smallest_n_congruence_three_satisfies_congruence_three_is_smallest_l344_34405

theorem smallest_n_congruence (n : ℕ) : n > 0 ∧ 23 * n ≡ 789 [MOD 8] → n ≥ 3 :=
by sorry

theorem three_satisfies_congruence : 23 * 3 ≡ 789 [MOD 8] :=
by sorry

theorem three_is_smallest (m : ℕ) : m > 0 ∧ 23 * m ≡ 789 [MOD 8] → m ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_three_satisfies_congruence_three_is_smallest_l344_34405


namespace NUMINAMATH_CALUDE_fruit_distribution_l344_34401

/-- The number of ways to distribute n identical items among k distinct recipients --/
def distribute_identical (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute m distinct items among k distinct recipients --/
def distribute_distinct (m k : ℕ) : ℕ := k^m

theorem fruit_distribution :
  let apples : ℕ := 6
  let distinct_fruits : ℕ := 3  -- orange, plum, tangerine
  let people : ℕ := 3
  distribute_identical apples people * distribute_distinct distinct_fruits people = 756 := by
sorry

end NUMINAMATH_CALUDE_fruit_distribution_l344_34401


namespace NUMINAMATH_CALUDE_equation_solutions_l344_34440

theorem equation_solutions :
  ∀ a b : ℤ, 3 * a^2 * b^2 + b^2 = 517 + 30 * a^2 ↔ 
  ((a = 2 ∧ b = 7) ∨ (a = -2 ∧ b = 7) ∨ (a = 2 ∧ b = -7) ∨ (a = -2 ∧ b = -7)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l344_34440


namespace NUMINAMATH_CALUDE_ethanol_percentage_in_fuel_A_l344_34453

/-- Proves that the percentage of ethanol in fuel A is 12% -/
theorem ethanol_percentage_in_fuel_A :
  let tank_capacity : ℝ := 208
  let fuel_A_volume : ℝ := 82
  let fuel_B_ethanol_percentage : ℝ := 0.16
  let total_ethanol : ℝ := 30
  let fuel_B_volume : ℝ := tank_capacity - fuel_A_volume
  let fuel_A_ethanol_percentage : ℝ := (total_ethanol - fuel_B_ethanol_percentage * fuel_B_volume) / fuel_A_volume
  fuel_A_ethanol_percentage = 0.12 := by sorry

end NUMINAMATH_CALUDE_ethanol_percentage_in_fuel_A_l344_34453


namespace NUMINAMATH_CALUDE_min_value_condition_l344_34489

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then |x + a| + |x - 2|
  else x^2 - a*x + (1/2)*a + 1

theorem min_value_condition (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 2*a) ∧ (∃ x : ℝ, f a x = 2*a) ↔ a = -Real.sqrt 13 - 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_condition_l344_34489


namespace NUMINAMATH_CALUDE_ball_probabilities_l344_34424

/-- The number of red balls in the bag -/
def num_red : ℕ := 3

/-- The number of white balls in the bag -/
def num_white : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_red + num_white

/-- The number of balls drawn -/
def num_drawn : ℕ := 2

theorem ball_probabilities :
  (num_red * (num_red - 1) / (total_balls * (total_balls - 1)) = 3 / 10) ∧
  (1 - (num_white * (num_white - 1) / (total_balls * (total_balls - 1))) = 9 / 10) :=
sorry

end NUMINAMATH_CALUDE_ball_probabilities_l344_34424


namespace NUMINAMATH_CALUDE_min_vertices_for_perpendicular_diagonals_l344_34409

theorem min_vertices_for_perpendicular_diagonals : 
  (∀ k : ℕ, k < 28 → ¬(∃ m : ℕ, 2 * m = k ∧ m * (m - 1)^2 / 2 ≥ 1000)) ∧ 
  (∃ m : ℕ, 2 * m = 28 ∧ m * (m - 1)^2 / 2 ≥ 1000) := by
  sorry

end NUMINAMATH_CALUDE_min_vertices_for_perpendicular_diagonals_l344_34409


namespace NUMINAMATH_CALUDE_complement_implies_set_l344_34461

def U : Set ℕ := {1, 3, 5, 7}

theorem complement_implies_set (M : Set ℕ) : 
  U = {1, 3, 5, 7} → (U \ M = {5, 7}) → M = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_implies_set_l344_34461


namespace NUMINAMATH_CALUDE_raised_bed_width_l344_34458

theorem raised_bed_width (num_beds : ℕ) (length height : ℝ) (num_bags : ℕ) (soil_per_bag : ℝ) :
  num_beds = 2 →
  length = 8 →
  height = 1 →
  num_bags = 16 →
  soil_per_bag = 4 →
  (num_bags : ℝ) * soil_per_bag / num_beds / (length * height) = 4 :=
by sorry

end NUMINAMATH_CALUDE_raised_bed_width_l344_34458


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l344_34437

theorem min_reciprocal_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 2) : 
  1/x + 1/y + 1/z ≥ 9/2 := by
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l344_34437


namespace NUMINAMATH_CALUDE_min_gymnasts_is_30_l344_34407

/-- Represents the total number of handshakes in a gymnastics meet -/
def total_handshakes : ℕ := 465

/-- Calculates the number of handshakes given the number of gymnasts -/
def handshakes (n : ℕ) : ℕ := (n * (n - 1)) / 2 + n

/-- Proves that 30 is the minimum number of gymnasts that satisfies the conditions -/
theorem min_gymnasts_is_30 :
  ∀ n : ℕ, n > 0 → n % 2 = 0 → handshakes n = total_handshakes → n ≥ 30 :=
by sorry

end NUMINAMATH_CALUDE_min_gymnasts_is_30_l344_34407


namespace NUMINAMATH_CALUDE_stratified_sample_size_l344_34430

/-- Represents the ratio of quantities for three product models -/
structure ProductRatio :=
  (a : ℕ)
  (b : ℕ)
  (c : ℕ)

/-- Calculates the total sample size given the number of items from the smallest group -/
def calculateSampleSize (ratio : ProductRatio) (smallestGroupSample : ℕ) : ℕ :=
  smallestGroupSample * (ratio.a + ratio.b + ratio.c) / ratio.a

/-- Theorem: For a stratified sample with ratio 3:4:7, if the smallest group has 9 items, the total sample size is 42 -/
theorem stratified_sample_size (ratio : ProductRatio) (h1 : ratio.a = 3) (h2 : ratio.b = 4) (h3 : ratio.c = 7) :
  calculateSampleSize ratio 9 = 42 := by
  sorry

#eval calculateSampleSize ⟨3, 4, 7⟩ 9

end NUMINAMATH_CALUDE_stratified_sample_size_l344_34430


namespace NUMINAMATH_CALUDE_solve_equation_l344_34473

theorem solve_equation (y : ℝ) (h : 3 * y + 2 = 11) : 6 * y + 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l344_34473


namespace NUMINAMATH_CALUDE_a_7_equals_two_l344_34431

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- Arithmetic sequence property -/
def is_arithmetic (b : Sequence) : Prop :=
  ∀ n m : ℕ, b (n + 1) - b n = b (m + 1) - b m

theorem a_7_equals_two (a b : Sequence) 
  (h1 : ∀ n, a n ≠ 0)
  (h2 : a 4 - 2 * a 7 + a 8 = 0)
  (h3 : is_arithmetic b)
  (h4 : b 7 = a 7)
  (h5 : b 2 < b 8)
  (h6 : b 8 < b 11) :
  a 7 = 2 :=
sorry

end NUMINAMATH_CALUDE_a_7_equals_two_l344_34431


namespace NUMINAMATH_CALUDE_existence_of_odd_powers_representation_l344_34414

theorem existence_of_odd_powers_representation (m : ℤ) :
  ∃ (a b k : ℤ), 
    Odd a ∧ 
    Odd b ∧ 
    k ≥ 0 ∧ 
    2 * m = a^19 + b^99 + k * 2^1999 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_odd_powers_representation_l344_34414


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_parallel_lines_l344_34475

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines
variable (parallel_lines : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Theorem statement
theorem planes_parallel_if_perpendicular_to_parallel_lines
  (m n : Line) (α β : Plane) :
  parallel_lines m n →
  perpendicular_line_plane m α →
  perpendicular_line_plane n β →
  parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_parallel_lines_l344_34475


namespace NUMINAMATH_CALUDE_wipes_used_correct_l344_34480

/-- Calculates the number of wipes used before refilling -/
def wipes_used (initial : ℕ) (refill : ℕ) (final : ℕ) : ℕ :=
  initial + refill - final

theorem wipes_used_correct (initial refill final : ℕ) 
  (h_initial : initial = 70)
  (h_refill : refill = 10)
  (h_final : final = 60) :
  wipes_used initial refill final = 20 := by
  sorry

#eval wipes_used 70 10 60

end NUMINAMATH_CALUDE_wipes_used_correct_l344_34480


namespace NUMINAMATH_CALUDE_chairs_left_theorem_l344_34467

/-- The number of chairs left to move given the total number of chairs and the number of chairs moved by each person. -/
def chairs_left_to_move (total : ℕ) (moved_by_carey : ℕ) (moved_by_pat : ℕ) : ℕ :=
  total - (moved_by_carey + moved_by_pat)

/-- Theorem stating that given 74 total chairs, with 28 moved by Carey and 29 moved by Pat, there are 17 chairs left to move. -/
theorem chairs_left_theorem : chairs_left_to_move 74 28 29 = 17 := by
  sorry

end NUMINAMATH_CALUDE_chairs_left_theorem_l344_34467


namespace NUMINAMATH_CALUDE_clown_balloons_l344_34456

/-- The number of balloons in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of balloons the clown initially has -/
def initial_dozens : ℕ := 3

/-- The number of boys who buy a balloon -/
def boys : ℕ := 3

/-- The number of girls who buy a balloon -/
def girls : ℕ := 12

/-- The number of balloons the clown is left with after selling to boys and girls -/
def remaining_balloons : ℕ := initial_dozens * dozen - (boys + girls)

theorem clown_balloons : remaining_balloons = 21 := by
  sorry

end NUMINAMATH_CALUDE_clown_balloons_l344_34456


namespace NUMINAMATH_CALUDE_solve_melanie_dimes_l344_34408

def melanie_dimes_problem (initial_dimes dad_dimes mom_dimes current_total : ℕ) : Prop :=
  initial_dimes + dad_dimes + mom_dimes = current_total

theorem solve_melanie_dimes :
  ∃ initial_dimes : ℕ,
    melanie_dimes_problem initial_dimes 8 4 19 ∧
    initial_dimes = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_melanie_dimes_l344_34408


namespace NUMINAMATH_CALUDE_king_spade_then_spade_probability_l344_34436

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (spades : Nat)
  (king_of_spades : Nat)

/-- The probability of drawing a King of Spades followed by any Spade from a standard 52-card deck -/
def probability_king_spade_then_spade (d : Deck) : Rat :=
  (d.king_of_spades : Rat) / d.total_cards * (d.spades - 1) / (d.total_cards - 1)

/-- Theorem stating the probability of drawing a King of Spades followed by any Spade 
    from a standard 52-card deck is 1/221 -/
theorem king_spade_then_spade_probability :
  probability_king_spade_then_spade ⟨52, 13, 1⟩ = 1 / 221 := by
  sorry

end NUMINAMATH_CALUDE_king_spade_then_spade_probability_l344_34436


namespace NUMINAMATH_CALUDE_point_division_l344_34454

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define points A, B, and P
variable (A B P : V)

-- Define the condition that P is on the line segment AB
def on_line_segment (P A B : V) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B

-- Define the ratio condition
def ratio_condition (P A B : V) : Prop := ∃ (k : ℝ), k > 0 ∧ 2 • (P - A) = k • (B - P) ∧ 7 • (P - A) = k • (B - P)

-- Theorem statement
theorem point_division (h1 : on_line_segment P A B) (h2 : ratio_condition P A B) :
  P = (7/9 : ℝ) • A + (2/9 : ℝ) • B :=
sorry

end NUMINAMATH_CALUDE_point_division_l344_34454


namespace NUMINAMATH_CALUDE_sqrt_necessary_not_sufficient_l344_34455

-- Define the necessary condition
def necessary_condition (x y : ℝ) : Prop :=
  (∀ x y, (Real.log x > Real.log y) → (Real.sqrt x > Real.sqrt y))

-- Define the sufficient condition
def sufficient_condition (x y : ℝ) : Prop :=
  (∀ x y, (Real.sqrt x > Real.sqrt y) → (Real.log x > Real.log y))

-- Theorem stating that the condition is necessary but not sufficient
theorem sqrt_necessary_not_sufficient :
  (∃ x y, necessary_condition x y) ∧ (¬∃ x y, sufficient_condition x y) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_necessary_not_sufficient_l344_34455


namespace NUMINAMATH_CALUDE_ancient_chinese_rope_problem_l344_34472

theorem ancient_chinese_rope_problem (x y : ℝ) :
  (1/2 : ℝ) * x - y = 5 ∧ y - (1/3 : ℝ) * x = 2 → x = 42 ∧ y = 16 := by
  sorry

end NUMINAMATH_CALUDE_ancient_chinese_rope_problem_l344_34472


namespace NUMINAMATH_CALUDE_min_p_plus_q_l344_34426

theorem min_p_plus_q (p q : ℕ+) (h : 108 * p = q ^ 3) : 
  ∃ (p' q' : ℕ+), 108 * p' = q' ^ 3 ∧ p' + q' ≤ p + q ∧ p' + q' = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_p_plus_q_l344_34426


namespace NUMINAMATH_CALUDE_geometry_relations_l344_34429

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem geometry_relations 
  (l m : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : subset m β) :
  (parallel_planes α β → perpendicular_lines l m) ∧
  ¬(perpendicular_lines l m → parallel_planes α β) ∧
  ¬(perpendicular_planes α β → parallel_lines l m) ∧
  (parallel_lines l m → perpendicular_planes α β) :=
sorry

end NUMINAMATH_CALUDE_geometry_relations_l344_34429


namespace NUMINAMATH_CALUDE_equivalent_discount_l344_34410

theorem equivalent_discount (original_price : ℝ) 
  (first_discount second_discount : ℝ) 
  (h1 : first_discount = 0.3) 
  (h2 : second_discount = 0.2) :
  let price_after_first := original_price * (1 - first_discount)
  let final_price := price_after_first * (1 - second_discount)
  let equivalent_discount := 1 - (final_price / original_price)
  equivalent_discount = 0.44 := by sorry

end NUMINAMATH_CALUDE_equivalent_discount_l344_34410


namespace NUMINAMATH_CALUDE_sqrt_one_plus_a_squared_is_quadratic_radical_l344_34411

/-- A function is a quadratic radical if it's the square root of an expression 
    that yields a real number for all real values of its variable. -/
def is_quadratic_radical (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, g x ≥ 0) ∧ (∀ x, f x = Real.sqrt (g x))

/-- The function f(a) = √(1 + a²) is a quadratic radical. -/
theorem sqrt_one_plus_a_squared_is_quadratic_radical :
  is_quadratic_radical (fun a => Real.sqrt (1 + a^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_sqrt_one_plus_a_squared_is_quadratic_radical_l344_34411


namespace NUMINAMATH_CALUDE_max_regions_theorem_l344_34487

/-- Represents a circular disk with chords and secant lines. -/
structure DiskWithChords where
  n : ℕ
  chord_count : ℕ := 2 * n + 1
  secant_count : ℕ := 2

/-- Calculates the maximum number of non-overlapping regions in the disk. -/
def max_regions (disk : DiskWithChords) : ℕ :=
  8 * disk.n + 8

/-- Theorem stating the maximum number of non-overlapping regions. -/
theorem max_regions_theorem (disk : DiskWithChords) (h : disk.n > 0) :
  max_regions disk = 8 * disk.n + 8 := by
  sorry

end NUMINAMATH_CALUDE_max_regions_theorem_l344_34487


namespace NUMINAMATH_CALUDE_george_boxes_count_l344_34484

/-- The number of blocks each box can hold -/
def blocks_per_box : ℕ := 6

/-- The total number of blocks George has -/
def total_blocks : ℕ := 12

/-- The number of boxes George has -/
def number_of_boxes : ℕ := total_blocks / blocks_per_box

theorem george_boxes_count : number_of_boxes = 2 := by
  sorry

end NUMINAMATH_CALUDE_george_boxes_count_l344_34484


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l344_34478

theorem contrapositive_equivalence (a b : ℝ) :
  (((a ≠ 0 ∨ b ≠ 0) → a^2 + b^2 ≠ 0) ↔ 
   (a^2 + b^2 = 0 → a = 0 ∧ b = 0)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l344_34478


namespace NUMINAMATH_CALUDE_quadratic_root_sum_product_l344_34476

theorem quadratic_root_sum_product (p q : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x + y = 9 ∧ x * y = 14) →
  p + q = 69 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_product_l344_34476


namespace NUMINAMATH_CALUDE_inequality_proof_l344_34463

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 9*y + 3*z) * (x + 4*y + 2*z) * (2*x + 12*y + 9*z) ≥ 1029 * x * y * z := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l344_34463


namespace NUMINAMATH_CALUDE_smallest_marble_collection_l344_34422

theorem smallest_marble_collection (M : ℕ) : 
  M > 1 → 
  M % 5 = 2 → 
  M % 6 = 2 → 
  M % 7 = 2 → 
  (∀ n : ℕ, n > 1 ∧ n % 5 = 2 ∧ n % 6 = 2 ∧ n % 7 = 2 → n ≥ M) → 
  M = 212 :=
by sorry

end NUMINAMATH_CALUDE_smallest_marble_collection_l344_34422


namespace NUMINAMATH_CALUDE_kolya_tolya_ages_l344_34417

/-- Represents a person's age as a two-digit number -/
structure Age :=
  (tens : Nat)
  (ones : Nat)
  (is_valid : tens < 10 ∧ ones < 10)

/-- Calculates the numeric value of an Age -/
def Age.value (a : Age) : Nat :=
  10 * a.tens + a.ones

/-- Reverses the digits of an Age -/
def Age.reverse (a : Age) : Age :=
  ⟨a.ones, a.tens, a.is_valid.symm⟩

theorem kolya_tolya_ages :
  ∃ (kolya_age tolya_age : Age),
    -- Kolya is older than Tolya
    kolya_age.value > tolya_age.value ∧
    -- Both ages are less than 100
    kolya_age.value < 100 ∧ tolya_age.value < 100 ∧
    -- Reversing Kolya's age gives Tolya's age
    kolya_age.reverse = tolya_age ∧
    -- The difference of squares is a perfect square
    ∃ (k : Nat), (kolya_age.value ^ 2 - tolya_age.value ^ 2 = k ^ 2) ∧
    -- Kolya is 65 and Tolya is 56
    kolya_age.value = 65 ∧ tolya_age.value = 56 := by
  sorry

end NUMINAMATH_CALUDE_kolya_tolya_ages_l344_34417


namespace NUMINAMATH_CALUDE_decreasing_function_a_range_l344_34451

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then -4 * x + 2 * a else x^2 - a * x + 4

-- Define what it means for f to be decreasing on ℝ
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- Main theorem statement
theorem decreasing_function_a_range :
  ∃ a_min a_max : ℝ, a_min = 2 ∧ a_max = 3 ∧
  (∀ a : ℝ, is_decreasing (f a) ↔ a_min ≤ a ∧ a ≤ a_max) :=
sorry

end NUMINAMATH_CALUDE_decreasing_function_a_range_l344_34451


namespace NUMINAMATH_CALUDE_de_morgan_and_jenkins_birth_years_l344_34450

def birth_year_de_morgan (x : ℕ) : Prop :=
  x^2 - x = 1806

def birth_year_jenkins (a b m n : ℕ) : Prop :=
  (a^4 + b^4) - (a^2 + b^2) = 1860 ∧
  2 * m^2 - 2 * m = 1860 ∧
  3 * n^4 - 3 * n = 1860

theorem de_morgan_and_jenkins_birth_years :
  ∃ (x a b m n : ℕ),
    birth_year_de_morgan x ∧
    birth_year_jenkins a b m n :=
sorry

end NUMINAMATH_CALUDE_de_morgan_and_jenkins_birth_years_l344_34450


namespace NUMINAMATH_CALUDE_min_values_for_constrained_x_y_l344_34474

theorem min_values_for_constrained_x_y :
  ∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 1 →
  (∀ a b : ℝ, a > 0 → b > 0 → 2 * a + b = 1 → 2 / x + 1 / y ≤ 2 / a + 1 / b) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → 2 * a + b = 1 → 4 * x^2 + y^2 ≤ 4 * a^2 + b^2) ∧
  (2 / x + 1 / y = 9) ∧
  (4 * x^2 + y^2 = 1/2) := by
sorry

end NUMINAMATH_CALUDE_min_values_for_constrained_x_y_l344_34474


namespace NUMINAMATH_CALUDE_remainder_of_division_l344_34486

theorem remainder_of_division (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) 
    (h1 : dividend = 1235678)
    (h2 : divisor = 127)
    (h3 : remainder < divisor)
    (h4 : dividend = quotient * divisor + remainder) :
  remainder = 69 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_division_l344_34486


namespace NUMINAMATH_CALUDE_solution_set_implies_m_value_l344_34457

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 3 * x + a

-- Define the property of m and 1 being roots of the equation f a x = 0
def are_roots (a m : ℝ) : Prop := f a m = 0 ∧ f a 1 = 0

-- Define the property of (m, 1) being the solution set of the inequality
def is_solution_set (a m : ℝ) : Prop :=
  ∀ x, f a x < 0 ↔ m < x ∧ x < 1

-- State the theorem
theorem solution_set_implies_m_value (a m : ℝ) :
  is_solution_set a m → m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_m_value_l344_34457


namespace NUMINAMATH_CALUDE_girls_in_senior_year_l344_34439

theorem girls_in_senior_year 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (girls_boys_diff : ℕ) 
  (h1 : total_students = 1200)
  (h2 : sample_size = 100)
  (h3 : girls_boys_diff = 20) :
  let boys_in_sample := (sample_size + girls_boys_diff) / 2
  let girls_in_sample := sample_size - boys_in_sample
  let sampling_ratio := sample_size / total_students
  (girls_in_sample * (total_students / sample_size) : ℚ) = 480 := by
sorry

end NUMINAMATH_CALUDE_girls_in_senior_year_l344_34439
