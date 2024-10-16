import Mathlib

namespace NUMINAMATH_CALUDE_unique_complex_root_l3349_334966

/-- The equation has exactly one complex root if and only if k = 2i or k = -2i -/
theorem unique_complex_root (k : ℂ) : 
  (∃! z : ℂ, (z^2 / (z+1)) + (z^2 / (z+2)) = k*z^2) ↔ k = 2*I ∨ k = -2*I :=
by sorry

end NUMINAMATH_CALUDE_unique_complex_root_l3349_334966


namespace NUMINAMATH_CALUDE_smallest_top_block_l3349_334924

/-- Represents the pyramid structure -/
structure Pyramid :=
  (layer1 : Fin 15 → ℕ)
  (layer2 : Fin 10 → ℕ)
  (layer3 : Fin 6 → ℕ)
  (layer4 : ℕ)

/-- The rule for assigning numbers to upper blocks -/
def upper_block_rule (a b c : ℕ) : ℕ := 2 * (a + b + c)

/-- The pyramid satisfies the numbering rules -/
def valid_pyramid (p : Pyramid) : Prop :=
  (∀ i : Fin 15, p.layer1 i ∈ Finset.range 16) ∧
  (∀ i : Fin 10, ∃ a b c : Fin 15, p.layer2 i = upper_block_rule (p.layer1 a) (p.layer1 b) (p.layer1 c)) ∧
  (∀ i : Fin 6, ∃ a b c : Fin 10, p.layer3 i = upper_block_rule (p.layer2 a) (p.layer2 b) (p.layer2 c)) ∧
  (∃ a b c : Fin 6, p.layer4 = upper_block_rule (p.layer3 a) (p.layer3 b) (p.layer3 c))

/-- The theorem stating the smallest possible number for the top block -/
theorem smallest_top_block (p : Pyramid) (h : valid_pyramid p) : p.layer4 ≥ 48 := by
  sorry

end NUMINAMATH_CALUDE_smallest_top_block_l3349_334924


namespace NUMINAMATH_CALUDE_five_sixths_of_twelve_fifths_minus_half_l3349_334999

theorem five_sixths_of_twelve_fifths_minus_half :
  (5 / 6 : ℚ) * (12 / 5 : ℚ) - (1 / 2 : ℚ) = (3 / 2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_five_sixths_of_twelve_fifths_minus_half_l3349_334999


namespace NUMINAMATH_CALUDE_rhombus_tiling_2n_gon_rhombus_tiling_2002_gon_l3349_334981

/-- The number of rhombuses needed to tile a regular 2n-gon -/
def num_rhombuses (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating the number of rhombuses in a tiling of a regular 2n-gon -/
theorem rhombus_tiling_2n_gon (n : ℕ) (h : n > 1) :
  num_rhombuses n = n * (n - 1) / 2 :=
by sorry

/-- Corollary for the specific case of a 2002-gon -/
theorem rhombus_tiling_2002_gon :
  num_rhombuses 1001 = 500500 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_tiling_2n_gon_rhombus_tiling_2002_gon_l3349_334981


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3349_334908

theorem geometric_sequence_sum (a r : ℝ) : 
  a + a * r = 7 →
  a * (r^6 - 1) / (r - 1) = 91 →
  a + a * r + a * r^2 + a * r^3 = 28 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3349_334908


namespace NUMINAMATH_CALUDE_prime_power_theorem_l3349_334997

theorem prime_power_theorem (p q : ℕ) : 
  p > 1 → q > 1 → 
  Nat.Prime p → Nat.Prime q → 
  Nat.Prime (7 * p + q) → Nat.Prime (p * q + 11) → 
  p^q = 8 ∨ p^q = 9 := by
sorry

end NUMINAMATH_CALUDE_prime_power_theorem_l3349_334997


namespace NUMINAMATH_CALUDE_jacket_price_l3349_334922

theorem jacket_price (jacket_count : ℕ) (shorts_count : ℕ) (pants_count : ℕ) 
  (shorts_price : ℚ) (pants_price : ℚ) (total_spent : ℚ) :
  jacket_count = 3 → 
  shorts_count = 2 →
  pants_count = 4 →
  shorts_price = 6 →
  pants_price = 12 →
  total_spent = 90 →
  ∃ (jacket_price : ℚ), 
    jacket_price * jacket_count + shorts_price * shorts_count + pants_price * pants_count = total_spent ∧
    jacket_price = 10 :=
by sorry

end NUMINAMATH_CALUDE_jacket_price_l3349_334922


namespace NUMINAMATH_CALUDE_p_is_third_degree_trinomial_l3349_334982

-- Define the polynomial
def p (x y : ℝ) : ℝ := 2 * x^2 - 3 * x * y + 5 * x * y^2

-- Theorem statement
theorem p_is_third_degree_trinomial :
  (∃ (a b c : ℝ) (f g h : ℕ → ℕ → ℕ), 
    (∀ x y, p x y = a * x^(f 0 0) * y^(f 0 1) + b * x^(g 0 0) * y^(g 0 1) + c * x^(h 0 0) * y^(h 0 1)) ∧
    (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧
    (max (f 0 0 + f 0 1) (max (g 0 0 + g 0 1) (h 0 0 + h 0 1)) = 3)) :=
by sorry


end NUMINAMATH_CALUDE_p_is_third_degree_trinomial_l3349_334982


namespace NUMINAMATH_CALUDE_combination_equality_l3349_334969

theorem combination_equality (x : ℕ) : 
  (Nat.choose 8 x = Nat.choose 8 (2*x - 1)) → (x = 1 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_l3349_334969


namespace NUMINAMATH_CALUDE_afternoon_session_count_l3349_334965

/-- Represents the number of kids in each session for a sport -/
structure SportSessions :=
  (morning : ℕ)
  (afternoon : ℕ)
  (evening : ℕ)
  (undecided : ℕ)

/-- Calculates the total number of kids in afternoon sessions across all sports -/
def total_afternoon_kids (soccer : SportSessions) (basketball : SportSessions) (swimming : SportSessions) : ℕ :=
  soccer.afternoon + basketball.afternoon + swimming.afternoon

theorem afternoon_session_count :
  ∀ (total_kids : ℕ) 
    (soccer basketball swimming : SportSessions),
  total_kids = 2000 →
  soccer.morning + soccer.afternoon + soccer.evening + soccer.undecided = 400 →
  basketball.morning + basketball.afternoon + basketball.evening = 300 →
  swimming.morning + swimming.afternoon + swimming.evening = 300 →
  soccer.morning = 100 →
  soccer.afternoon = 280 →
  soccer.undecided = 20 →
  basketball.evening = 180 →
  basketball.morning = basketball.afternoon →
  swimming.morning = swimming.afternoon →
  swimming.afternoon = swimming.evening →
  ∃ (soccer_new basketball_new swimming_new : SportSessions),
    soccer_new.morning = soccer.morning + 30 →
    soccer_new.afternoon = soccer.afternoon - 30 →
    soccer_new.evening = soccer.evening →
    soccer_new.undecided = soccer.undecided →
    basketball_new = basketball →
    swimming_new.morning = swimming.morning + 15 →
    swimming_new.afternoon = swimming.afternoon - 15 →
    swimming_new.evening = swimming.evening →
    total_afternoon_kids soccer_new basketball_new swimming_new = 395 :=
by sorry

end NUMINAMATH_CALUDE_afternoon_session_count_l3349_334965


namespace NUMINAMATH_CALUDE_farmer_land_ownership_l3349_334921

/-- Proves that a farmer owns 5000 acres of land given the described land usage -/
theorem farmer_land_ownership : ∀ (total_land : ℝ),
  (0.9 * total_land * 0.1 + 0.9 * total_land * 0.8 + 450 = 0.9 * total_land) →
  total_land = 5000 := by
  sorry

end NUMINAMATH_CALUDE_farmer_land_ownership_l3349_334921


namespace NUMINAMATH_CALUDE_emilee_earnings_l3349_334985

/-- Given the earnings of three people with specific conditions, prove Emilee's earnings. -/
theorem emilee_earnings (total : ℕ) (terrence_earnings : ℕ) (jermaine_extra : ℕ) :
  total = 90 →
  terrence_earnings = 30 →
  jermaine_extra = 5 →
  ∃ emilee_earnings : ℕ,
    emilee_earnings = total - (terrence_earnings + (terrence_earnings + jermaine_extra)) ∧
    emilee_earnings = 25 :=
by sorry

end NUMINAMATH_CALUDE_emilee_earnings_l3349_334985


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3349_334951

theorem trigonometric_identity (α β γ : Real) :
  Real.sin α + Real.sin β + Real.sin γ - 
  Real.sin (α + β) * Real.cos γ - Real.cos (α + β) * Real.sin γ = 
  4 * Real.sin ((α + β) / 2) * Real.sin ((β + γ) / 2) * Real.sin ((γ + α) / 2) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3349_334951


namespace NUMINAMATH_CALUDE_same_solution_implies_c_value_l3349_334901

theorem same_solution_implies_c_value (x c : ℚ) : 
  (3 * x + 5 = 1) ∧ (c * x - 8 = -5) → c = -9/4 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_c_value_l3349_334901


namespace NUMINAMATH_CALUDE_horner_method_value_l3349_334988

def horner_polynomial (x : ℝ) : ℝ := (((-6 * x + 5) * x + 0) * x + 2) * x + 6

theorem horner_method_value :
  horner_polynomial 3 = -115 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_value_l3349_334988


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l3349_334909

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 10| = |x^2 + x - 2| :=
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l3349_334909


namespace NUMINAMATH_CALUDE_grooming_time_calculation_l3349_334915

/-- Proves that the time to clip each claw is 10 seconds given the grooming conditions --/
theorem grooming_time_calculation (total_time : ℕ) (num_claws : ℕ) (ear_cleaning_time : ℕ) (shampoo_time_minutes : ℕ) :
  total_time = 640 →
  num_claws = 16 →
  ear_cleaning_time = 90 →
  shampoo_time_minutes = 5 →
  ∃ (claw_clip_time : ℕ),
    claw_clip_time = 10 ∧
    total_time = num_claws * claw_clip_time + 2 * ear_cleaning_time + shampoo_time_minutes * 60 :=
by sorry

end NUMINAMATH_CALUDE_grooming_time_calculation_l3349_334915


namespace NUMINAMATH_CALUDE_sin_20_cos_10_plus_sin_10_sin_70_l3349_334938

theorem sin_20_cos_10_plus_sin_10_sin_70 :
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) +
  Real.sin (10 * π / 180) * Real.sin (70 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_20_cos_10_plus_sin_10_sin_70_l3349_334938


namespace NUMINAMATH_CALUDE_arithmetic_sum_l3349_334940

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 = 2 →
  a 2 + a 3 = 13 →
  a 4 + a 5 + a 6 = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_l3349_334940


namespace NUMINAMATH_CALUDE_flood_damage_conversion_l3349_334952

/-- Calculates the equivalent amount in USD given an amount in AUD and the exchange rate -/
def convert_aud_to_usd (amount_aud : ℝ) (exchange_rate : ℝ) : ℝ :=
  amount_aud * exchange_rate

/-- Theorem stating the conversion of flood damage from AUD to USD -/
theorem flood_damage_conversion :
  let damage_aud : ℝ := 45000000
  let exchange_rate : ℝ := 0.75
  convert_aud_to_usd damage_aud exchange_rate = 33750000 := by
  sorry

#check flood_damage_conversion

end NUMINAMATH_CALUDE_flood_damage_conversion_l3349_334952


namespace NUMINAMATH_CALUDE_decreasing_f_implies_a_bound_l3349_334986

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x + 1

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 1

-- Theorem statement
theorem decreasing_f_implies_a_bound (a : ℝ) :
  (∀ x ∈ Set.Ioo (-2/3) (-1/3), f_deriv a x < 0) →
  a ≥ 7/4 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_f_implies_a_bound_l3349_334986


namespace NUMINAMATH_CALUDE_sum_exterior_angles_dodecagon_l3349_334993

/-- The number of sides in a dodecagon -/
def dodecagon_sides : ℕ := 12

/-- The sum of exterior angles of any polygon in degrees -/
def sum_exterior_angles : ℝ := 360

/-- Theorem: The sum of the exterior angles of a regular dodecagon is 360° -/
theorem sum_exterior_angles_dodecagon :
  sum_exterior_angles = 360 := by sorry

end NUMINAMATH_CALUDE_sum_exterior_angles_dodecagon_l3349_334993


namespace NUMINAMATH_CALUDE_max_jogs_is_six_l3349_334920

/-- Represents the quantity of each item Bill can buy --/
structure Purchase where
  jags : Nat
  jigs : Nat
  jogs : Nat

/-- Calculates the total cost of a purchase --/
def totalCost (p : Purchase) : Nat :=
  p.jags * 1 + p.jigs * 2 + p.jogs * 7

/-- Checks if a purchase satisfies all conditions --/
def isValidPurchase (p : Purchase) : Prop :=
  p.jags ≥ 1 ∧ p.jigs ≥ 1 ∧ p.jogs ≥ 1 ∧ totalCost p = 50

/-- Theorem stating that the maximum number of jogs Bill can buy is 6 --/
theorem max_jogs_is_six :
  (∃ p : Purchase, isValidPurchase p ∧ p.jogs = 6) ∧
  (∀ p : Purchase, isValidPurchase p → p.jogs ≤ 6) :=
sorry

end NUMINAMATH_CALUDE_max_jogs_is_six_l3349_334920


namespace NUMINAMATH_CALUDE_exam_score_exam_score_specific_case_l3349_334928

theorem exam_score (total_questions : ℕ) (correct_answers : ℕ) 
  (marks_per_correct : ℕ) (marks_lost_per_wrong : ℕ) : ℕ :=
  let wrong_answers := total_questions - correct_answers
  let total_marks := correct_answers * marks_per_correct - wrong_answers * marks_lost_per_wrong
  total_marks

theorem exam_score_specific_case : exam_score 75 40 4 1 = 125 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_exam_score_specific_case_l3349_334928


namespace NUMINAMATH_CALUDE_max_score_15_cards_l3349_334956

/-- The score of a hand of cards -/
def score (R B Y : ℕ) : ℕ :=
  R + 2 * R * B + 3 * B * Y

/-- The theorem stating the maximum score achievable with 15 cards -/
theorem max_score_15_cards :
  ∃ R B Y : ℕ,
    R + B + Y = 15 ∧
    ∀ R' B' Y' : ℕ, R' + B' + Y' = 15 →
      score R' B' Y' ≤ score R B Y ∧
      score R B Y = 168 :=
sorry

end NUMINAMATH_CALUDE_max_score_15_cards_l3349_334956


namespace NUMINAMATH_CALUDE_staplers_remaining_l3349_334943

/-- The number of staplers left after stapling reports -/
def staplers_left (initial_staplers : ℕ) (dozens_stapled : ℕ) : ℕ :=
  initial_staplers - dozens_stapled * 12

/-- Theorem: Given 50 initial staplers and 3 dozen reports stapled, 14 staplers are left -/
theorem staplers_remaining : staplers_left 50 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_staplers_remaining_l3349_334943


namespace NUMINAMATH_CALUDE_monotonic_increasing_iff_b_range_l3349_334929

/-- The function y = (1/3)x³ + bx² + (b+2)x + 3 is monotonically increasing on ℝ 
    if and only if b < -1 or b > 2 -/
theorem monotonic_increasing_iff_b_range (b : ℝ) : 
  (∀ x : ℝ, StrictMono (fun x => (1/3) * x^3 + b * x^2 + (b + 2) * x + 3)) ↔ 
  (b < -1 ∨ b > 2) := by
sorry

end NUMINAMATH_CALUDE_monotonic_increasing_iff_b_range_l3349_334929


namespace NUMINAMATH_CALUDE_sine_cosine_identity_l3349_334944

theorem sine_cosine_identity : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (160 * π / 180) * Real.sin (170 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_identity_l3349_334944


namespace NUMINAMATH_CALUDE_equation_solution_l3349_334906

theorem equation_solution : ∃ x₁ x₂ : ℝ, x₁ = -5 ∧ x₂ = 3 ∧ 
  (∀ x : ℝ, (x + 3) * (x - 1) = 12 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3349_334906


namespace NUMINAMATH_CALUDE_halloween_candy_eaten_l3349_334978

/-- The number of candy pieces eaten on Halloween night -/
def candy_eaten (debby_initial : ℕ) (sister_initial : ℕ) (remaining : ℕ) : ℕ :=
  debby_initial + sister_initial - remaining

/-- Theorem stating the number of candy pieces eaten on Halloween night -/
theorem halloween_candy_eaten :
  candy_eaten 32 42 39 = 35 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_eaten_l3349_334978


namespace NUMINAMATH_CALUDE_min_value_of_ratio_l3349_334971

theorem min_value_of_ratio (x y : ℝ) (h1 : x + y - 3 ≤ 0) (h2 : x - y + 1 ≥ 0) (h3 : y ≥ 1) :
  ∃ (x₀ y₀ : ℝ), x₀ + y₀ - 3 ≤ 0 ∧ x₀ - y₀ + 1 ≥ 0 ∧ y₀ ≥ 1 ∧
    ∀ (x' y' : ℝ), x' + y' - 3 ≤ 0 → x' - y' + 1 ≥ 0 → y' ≥ 1 → y₀ / x₀ ≤ y' / x' :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_ratio_l3349_334971


namespace NUMINAMATH_CALUDE_n_has_21_digits_l3349_334936

/-- The smallest positive integer satisfying the given conditions -/
def n : ℕ := sorry

/-- n is divisible by 30 -/
axiom n_div_30 : 30 ∣ n

/-- n^2 is a perfect fourth power -/
axiom n_sq_fourth_power : ∃ k : ℕ, n^2 = k^4

/-- n^3 is a perfect fifth power -/
axiom n_cube_fifth_power : ∃ k : ℕ, n^3 = k^5

/-- n is the smallest positive integer satisfying the conditions -/
axiom n_smallest : ∀ m : ℕ, m > 0 → (30 ∣ m) → (∃ k : ℕ, m^2 = k^4) → (∃ k : ℕ, m^3 = k^5) → n ≤ m

/-- The number of digits in a natural number -/
def num_digits (x : ℕ) : ℕ := sorry

/-- The main theorem: n has 21 digits -/
theorem n_has_21_digits : num_digits n = 21 := by sorry

end NUMINAMATH_CALUDE_n_has_21_digits_l3349_334936


namespace NUMINAMATH_CALUDE_indeterminate_m_l3349_334970

/-- An odd function from ℝ to ℝ -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem indeterminate_m (f : ℝ → ℝ) (m : ℝ) 
  (hodd : OddFunction f) (hm : f m = 2) (hm2 : f (m^2 - 2) = -2) :
  ¬ (∀ n : ℝ, f n = 2 → n = m) :=
sorry

end NUMINAMATH_CALUDE_indeterminate_m_l3349_334970


namespace NUMINAMATH_CALUDE_floor_equation_natural_numbers_l3349_334912

theorem floor_equation_natural_numbers (a b : ℕ) :
  (a ≠ 0 ∧ b ≠ 0) →
  (Int.floor (a^2 / b : ℚ) + Int.floor (b^2 / a : ℚ) = 
   Int.floor ((a^2 + b^2) / (a * b) : ℚ) + a * b) ↔ 
  (b = a^2 + 1 ∨ a = b^2 + 1) :=
sorry

end NUMINAMATH_CALUDE_floor_equation_natural_numbers_l3349_334912


namespace NUMINAMATH_CALUDE_total_handshakes_l3349_334968

/-- Represents the number of married couples at the event -/
def num_couples : ℕ := 15

/-- Represents the total number of people at the event -/
def total_people : ℕ := 2 * num_couples

/-- Calculates the number of handshakes among men -/
def handshakes_among_men : ℕ := (num_couples - 1) * (num_couples - 2) / 2

/-- Calculates the number of handshakes between men and women (excluding spouses) -/
def handshakes_men_women : ℕ := num_couples * (num_couples - 1)

/-- Theorem stating the total number of handshakes at the event -/
theorem total_handshakes : 
  handshakes_among_men + handshakes_men_women = 301 := by sorry

end NUMINAMATH_CALUDE_total_handshakes_l3349_334968


namespace NUMINAMATH_CALUDE_probability_from_odds_probability_3_5_odds_l3349_334947

/-- Given odds of a:b in favor of an event, the probability of the event occurring is a/(a+b) -/
theorem probability_from_odds (a b : ℕ) (h : a > 0 ∧ b > 0) :
  let odds := a / b
  let probability := a / (a + b)
  probability = odds / (1 + odds) :=
by sorry

/-- The probability of an event with odds 3:5 in its favor is 3/8 -/
theorem probability_3_5_odds :
  let a := 3
  let b := 5
  let probability := a / (a + b)
  probability = 3 / 8 :=
by sorry

end NUMINAMATH_CALUDE_probability_from_odds_probability_3_5_odds_l3349_334947


namespace NUMINAMATH_CALUDE_circle_distance_extrema_l3349_334911

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

-- Define points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the distance function d
def d (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  (x + 1)^2 + y^2 + (x - 1)^2 + y^2

-- Theorem statement
theorem circle_distance_extrema :
  (∃ P : ℝ × ℝ, C P.1 P.2 ∧ ∀ Q : ℝ × ℝ, C Q.1 Q.2 → d P ≥ d Q) ∧
  (∃ P : ℝ × ℝ, C P.1 P.2 ∧ ∀ Q : ℝ × ℝ, C Q.1 Q.2 → d P ≤ d Q) ∧
  (∀ P : ℝ × ℝ, C P.1 P.2 → d P ≤ 74) ∧
  (∀ P : ℝ × ℝ, C P.1 P.2 → d P ≥ 34) :=
sorry

end NUMINAMATH_CALUDE_circle_distance_extrema_l3349_334911


namespace NUMINAMATH_CALUDE_ellipse_point_inside_circle_l3349_334932

theorem ellipse_point_inside_circle 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a > b) 
  (he : c / a = 1 / 2) 
  (hf : c > 0) 
  (x₁ x₂ : ℝ) 
  (hroots : x₁ * x₂ = -c / a ∧ x₁ + x₂ = -b / a) :
  x₁^2 + x₂^2 < 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_point_inside_circle_l3349_334932


namespace NUMINAMATH_CALUDE_no_winning_strategy_card_game_probability_l3349_334977

/-- Represents a deck of cards with red and black suits. -/
structure Deck :=
  (red : ℕ)
  (black : ℕ)

/-- Represents a strategy for playing the card game. -/
def Strategy := Deck → Bool

/-- The probability of winning given a deck and a strategy. -/
def winProbability (d : Deck) (s : Strategy) : ℚ :=
  d.red / (d.red + d.black)

/-- The theorem stating that no strategy can have a winning probability greater than 0.5. -/
theorem no_winning_strategy (d : Deck) (s : Strategy) :
  d.red = d.black → winProbability d s = 1/2 := by
  sorry

/-- The main theorem stating that for any strategy, 
    the probability of winning is always 0.5 for a standard deck. -/
theorem card_game_probability (s : Strategy) : 
  ∀ d : Deck, d.red = d.black → d.red + d.black = 52 → winProbability d s = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_no_winning_strategy_card_game_probability_l3349_334977


namespace NUMINAMATH_CALUDE_cosine_value_l3349_334914

theorem cosine_value (α : ℝ) (h : 2 * Real.cos (2 * α) + 9 * Real.sin α = 4) :
  Real.cos α = Real.sqrt 15 / 4 ∨ Real.cos α = -Real.sqrt 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_l3349_334914


namespace NUMINAMATH_CALUDE_alternate_arrangement_probability_l3349_334959

/-- The number of male fans -/
def num_male : ℕ := 3

/-- The number of female fans -/
def num_female : ℕ := 3

/-- The total number of fans -/
def total_fans : ℕ := num_male + num_female

/-- The number of ways to arrange fans alternately -/
def alternate_arrangements : ℕ := 2 * (Nat.factorial num_male) * (Nat.factorial num_female)

/-- The total number of possible arrangements -/
def total_arrangements : ℕ := Nat.factorial total_fans

/-- The probability of arranging fans alternately -/
def prob_alternate : ℚ := alternate_arrangements / total_arrangements

theorem alternate_arrangement_probability :
  prob_alternate = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_alternate_arrangement_probability_l3349_334959


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l3349_334933

/-- Isosceles triangle with given side length and area -/
structure IsoscelesTriangle where
  -- Side lengths
  ab : ℝ
  bc : ℝ
  -- Isosceles condition
  isIsosceles : ab = ac
  -- Given side length
  bcLength : bc = 16
  -- Area
  area : ℝ
  areaValue : area = 120

/-- The length of AB in the isosceles triangle -/
def sideLength (t : IsoscelesTriangle) : ℝ := t.ab

/-- Theorem: The length of AB in the given isosceles triangle is 17 -/
theorem isosceles_triangle_side_length (t : IsoscelesTriangle) :
  sideLength t = 17 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l3349_334933


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3349_334955

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x - 4 < 0}
def B : Set ℝ := {x | x^2 - 4*x + 3 > 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | (-1 < x ∧ x < 1) ∨ (3 < x ∧ x < 4)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3349_334955


namespace NUMINAMATH_CALUDE_set_operations_and_complements_l3349_334979

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | -2 ≤ x ∧ x < 2}

-- Define the theorem
theorem set_operations_and_complements :
  (A ∩ B = {x | -1 ≤ x ∧ x < 2}) ∧
  (A ∪ B = {x | -2 ≤ x ∧ x ≤ 3}) ∧
  ((Uᶜ ∪ (A ∩ B)) = {x | x < -1 ∨ 2 ≤ x}) ∧
  ((Uᶜ ∪ (A ∪ B)) = {x | x < -2 ∨ 3 < x}) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_and_complements_l3349_334979


namespace NUMINAMATH_CALUDE_beaver_carrots_l3349_334983

theorem beaver_carrots :
  ∀ (beaver_burrows rabbit_burrows : ℕ),
    beaver_burrows = rabbit_burrows + 5 →
    5 * beaver_burrows = 7 * rabbit_burrows →
    5 * beaver_burrows = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_beaver_carrots_l3349_334983


namespace NUMINAMATH_CALUDE_sum_of_real_solutions_l3349_334984

theorem sum_of_real_solutions (a : ℝ) (h : a > 1) :
  ∃ x : ℝ, x > 0 ∧ Real.sqrt (a - Real.sqrt (a + x)) = x ∧
  x = (Real.sqrt (4 * a - 3) - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_real_solutions_l3349_334984


namespace NUMINAMATH_CALUDE_roots_shift_l3349_334913

-- Define the original polynomial
def original_poly (x : ℝ) : ℝ := x^3 - 6*x + 5

-- Define the resulting polynomial
def resulting_poly (x : ℝ) : ℝ := x^3 + 9*x^2 + 21*x + 14

theorem roots_shift :
  ∀ (a b c : ℝ),
  (original_poly a = 0 ∧ original_poly b = 0 ∧ original_poly c = 0) →
  (∀ x : ℝ, resulting_poly x = 0 ↔ (x = a - 3 ∨ x = b - 3 ∨ x = c - 3)) :=
by sorry

end NUMINAMATH_CALUDE_roots_shift_l3349_334913


namespace NUMINAMATH_CALUDE_irrational_root_theorem_l3349_334987

theorem irrational_root_theorem (a : ℝ) :
  (¬ (∃ (q : ℚ), a = q)) →
  (∃ (s p : ℤ), a + (a^3 - 6*a) = s ∧ a*(a^3 - 6*a) = p) →
  (a = -1 - Real.sqrt 2 ∨
   a = -Real.sqrt 5 ∨
   a = 1 - Real.sqrt 2 ∨
   a = -1 + Real.sqrt 2 ∨
   a = Real.sqrt 5 ∨
   a = 1 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_irrational_root_theorem_l3349_334987


namespace NUMINAMATH_CALUDE_base_sum_22_l3349_334902

def F₁ (R : ℕ) : ℚ := (4*R + 5) / (R^2 - 1)
def F₂ (R : ℕ) : ℚ := (5*R + 4) / (R^2 - 1)

theorem base_sum_22 (R₁ R₂ : ℕ) : 
  (F₁ R₁ = 0.454545 ∧ F₂ R₁ = 0.545454) →
  (F₁ R₂ = 3 / 10 ∧ F₂ R₂ = 7 / 10) →
  R₁ + R₂ = 22 := by sorry

end NUMINAMATH_CALUDE_base_sum_22_l3349_334902


namespace NUMINAMATH_CALUDE_probability_neither_mix_l3349_334975

/-- Represents the set of all buyers -/
def TotalBuyers : ℕ := 100

/-- Represents the number of buyers who purchase cake mix -/
def CakeMixBuyers : ℕ := 50

/-- Represents the number of buyers who purchase muffin mix -/
def MuffinMixBuyers : ℕ := 40

/-- Represents the number of buyers who purchase both cake mix and muffin mix -/
def BothMixesBuyers : ℕ := 19

/-- Theorem stating the probability of selecting a buyer who purchases neither cake mix nor muffin mix -/
theorem probability_neither_mix (TotalBuyers CakeMixBuyers MuffinMixBuyers BothMixesBuyers : ℕ) 
  (h1 : TotalBuyers = 100)
  (h2 : CakeMixBuyers = 50)
  (h3 : MuffinMixBuyers = 40)
  (h4 : BothMixesBuyers = 19) :
  (TotalBuyers - (CakeMixBuyers + MuffinMixBuyers - BothMixesBuyers)) / TotalBuyers = 29 / 100 := by
  sorry

end NUMINAMATH_CALUDE_probability_neither_mix_l3349_334975


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3349_334957

theorem complex_number_quadrant : ∀ z : ℂ, 
  (3 - 2*I) * z = 4 + 3*I → 
  (0 < z.re ∧ 0 < z.im) := by
sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3349_334957


namespace NUMINAMATH_CALUDE_inverse_of_A_squared_l3349_334945

def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, -1; 2, 1]

theorem inverse_of_A_squared :
  A⁻¹ = !![3, -1; 2, 1] →
  (A^2)⁻¹ = !![7, -4; 8, -1] := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_squared_l3349_334945


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l3349_334939

/-- Given a circle D with equation x^2 + 20x + y^2 + 18y = -36,
    prove that its center coordinates (p, q) and radius s satisfy p + q + s = -19 + Real.sqrt 145 -/
theorem circle_center_radius_sum (x y : ℝ) :
  x^2 + 20*x + y^2 + 18*y = -36 →
  ∃ (p q s : ℝ), (∀ (x y : ℝ), (x - p)^2 + (y - q)^2 = s^2 ↔ x^2 + 20*x + y^2 + 18*y = -36) ∧
                 p + q + s = -19 + Real.sqrt 145 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l3349_334939


namespace NUMINAMATH_CALUDE_inequality_holds_C_is_maximum_l3349_334958

noncomputable def C : ℝ := (Real.sqrt (13 + 16 * Real.sqrt 2) - 1) / 2

theorem inequality_holds (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  x^3 + y^3 + z^3 + C * (x*y^2 + y*z^2 + z*x^2) ≥ (C + 1) * (x^2*y + y^2*z + z^2*x) :=
by sorry

theorem C_is_maximum : 
  ∀ D : ℝ, D > C → ∃ x y z : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
    x^3 + y^3 + z^3 + D * (x*y^2 + y*z^2 + z*x^2) < (D + 1) * (x^2*y + y^2*z + z^2*x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_C_is_maximum_l3349_334958


namespace NUMINAMATH_CALUDE_prob_shortest_diagonal_nonagon_l3349_334916

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n ≥ 3

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of shortest diagonals in a regular polygon with n sides -/
def num_shortest_diagonals (n : ℕ) : ℕ := n

/-- The probability of selecting a shortest diagonal in a regular polygon -/
def prob_shortest_diagonal (n : ℕ) : ℚ :=
  (num_shortest_diagonals n : ℚ) / (num_diagonals n : ℚ)

theorem prob_shortest_diagonal_nonagon :
  prob_shortest_diagonal 9 = 1/3 := by sorry

end NUMINAMATH_CALUDE_prob_shortest_diagonal_nonagon_l3349_334916


namespace NUMINAMATH_CALUDE_max_value_properties_l3349_334960

noncomputable def f (s : ℝ) (x : ℝ) : ℝ := (Real.log s) / (1 + x) - Real.log s

theorem max_value_properties (s : ℝ) (x₀ : ℝ) 
  (h_max : ∀ x, f s x ≤ f s x₀) :
  f s x₀ = x₀ ∧ f s x₀ < (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_max_value_properties_l3349_334960


namespace NUMINAMATH_CALUDE_smallest_prime_with_30_divisors_l3349_334941

/-- A function that counts the number of positive divisors of a natural number -/
def count_divisors (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is prime -/
def is_prime (n : ℕ) : Prop := sorry

/-- The expression p^3 + 4p^2 + 4p -/
def f (p : ℕ) : ℕ := p^3 + 4*p^2 + 4*p

theorem smallest_prime_with_30_divisors :
  ∀ p : ℕ, is_prime p → (∀ q < p, is_prime q → count_divisors (f q) ≠ 30) →
  count_divisors (f p) = 30 → p = 43 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_with_30_divisors_l3349_334941


namespace NUMINAMATH_CALUDE_plane_sphere_intersection_ratio_sum_l3349_334923

/-- The theorem states that for a plane intersecting the coordinate axes and a sphere passing through these intersection points and the origin, the sum of the ratios of a point on the plane to the sphere's center coordinates is 2. -/
theorem plane_sphere_intersection_ratio_sum (k : ℝ) (a b c p q r : ℝ) : 
  k ≠ 0 → -- k is a non-zero constant
  p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 → -- p, q, r are non-zero (as they are denominators)
  (∃ (α β γ : ℝ), α ≠ 0 ∧ β ≠ 0 ∧ γ ≠ 0 ∧ -- A, B, C exist and are distinct from O
    (k*a/α + k*b/β + k*c/γ = 1) ∧ -- plane equation
    (p^2 + q^2 + r^2 = (p - α)^2 + q^2 + r^2) ∧ -- sphere equation for A
    (p^2 + q^2 + r^2 = p^2 + (q - β)^2 + r^2) ∧ -- sphere equation for B
    (p^2 + q^2 + r^2 = p^2 + q^2 + (r - γ)^2)) → -- sphere equation for C
  k*a/p + k*b/q + k*c/r = 2 := by
sorry

end NUMINAMATH_CALUDE_plane_sphere_intersection_ratio_sum_l3349_334923


namespace NUMINAMATH_CALUDE_least_integer_absolute_value_l3349_334925

theorem least_integer_absolute_value (x : ℤ) : 
  (∀ y : ℤ, y < x → |3 * y + 4| > 18) ∧ |3 * x + 4| ≤ 18 → x = -7 :=
sorry

end NUMINAMATH_CALUDE_least_integer_absolute_value_l3349_334925


namespace NUMINAMATH_CALUDE_flu_transmission_rate_l3349_334935

/-- 
Given two rounds of flu transmission where a total of 121 people are infected,
prove that on average, one person infects 10 others in each round.
-/
theorem flu_transmission_rate : 
  ∃ x : ℕ, 
    (1 + x + x * (1 + x) = 121) ∧ 
    (x = 10) := by
  sorry

end NUMINAMATH_CALUDE_flu_transmission_rate_l3349_334935


namespace NUMINAMATH_CALUDE_quadratic_decreasing_interval_l3349_334917

/-- A quadratic function f(x) = x^2 + bx + c -/
def quadratic (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

/-- The derivative of a quadratic function -/
def quadratic_derivative (b : ℝ) (x : ℝ) : ℝ := 2*x + b

theorem quadratic_decreasing_interval (b c : ℝ) :
  (∀ x ≤ 1, quadratic_derivative b x ≤ 0) →
  (∃ x > 1, quadratic_derivative b x > 0) →
  b = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_interval_l3349_334917


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_l3349_334948

theorem greatest_integer_with_gcd_six (n : ℕ) : 
  (n < 50 ∧ Nat.gcd n 18 = 6 ∧ ∀ m : ℕ, m < 50 ∧ Nat.gcd m 18 = 6 → m ≤ n) → n = 42 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_l3349_334948


namespace NUMINAMATH_CALUDE_bridesmaids_dresses_completion_time_l3349_334972

def dress_hours : List Nat := [15, 18, 20, 22, 24, 26, 28]
def hours_per_week : Nat := 5

theorem bridesmaids_dresses_completion_time : 
  ∃ (total_hours : Nat),
    total_hours = dress_hours.sum ∧
    (total_hours / hours_per_week : ℚ) ≤ 31 ∧
    31 < (total_hours / hours_per_week : ℚ) + 1 :=
by sorry

end NUMINAMATH_CALUDE_bridesmaids_dresses_completion_time_l3349_334972


namespace NUMINAMATH_CALUDE_min_moves_to_black_l3349_334946

/-- Represents a chessboard with alternating colors -/
structure Chessboard :=
  (size : Nat)
  (alternating : Bool)

/-- Represents a move on the chessboard -/
structure Move :=
  (top_left : Nat × Nat)
  (bottom_right : Nat × Nat)

/-- Function to apply a move to a chessboard -/
def apply_move (board : Chessboard) (move : Move) : Chessboard := sorry

/-- Function to check if all squares are black -/
def all_black (board : Chessboard) : Bool := sorry

/-- Theorem stating the minimum number of moves required -/
theorem min_moves_to_black (board : Chessboard) :
  board.size = 98 ∧ board.alternating →
  (∃ (moves : List Move), all_black (moves.foldl apply_move board) ∧ moves.length = 98) ∧
  (∀ (moves : List Move), all_black (moves.foldl apply_move board) → moves.length ≥ 98) :=
sorry

end NUMINAMATH_CALUDE_min_moves_to_black_l3349_334946


namespace NUMINAMATH_CALUDE_expected_rounds_is_correct_l3349_334992

/-- Represents the number of rounds in the ball-drawing experiment -/
inductive Round : Type
  | one : Round
  | two : Round
  | three : Round

/-- The probability distribution of the number of rounds -/
def prob (r : Round) : ℚ :=
  match r with
  | Round.one => 1/4
  | Round.two => 1/12
  | Round.three => 2/3

/-- The expected number of rounds -/
def expected_rounds : ℚ := 29/12

/-- Theorem stating that the expected number of rounds is 29/12 -/
theorem expected_rounds_is_correct :
  (prob Round.one * 1 + prob Round.two * 2 + prob Round.three * 3 : ℚ) = expected_rounds := by
  sorry


end NUMINAMATH_CALUDE_expected_rounds_is_correct_l3349_334992


namespace NUMINAMATH_CALUDE_base_6_addition_l3349_334998

/-- Addition in base 6 -/
def add_base_6 (a b : ℕ) : ℕ :=
  (a + b) % 36

/-- Conversion from base 6 to decimal -/
def base_6_to_decimal (n : ℕ) : ℕ :=
  (n / 10) * 6 + (n % 10)

theorem base_6_addition :
  add_base_6 (base_6_to_decimal 4) (base_6_to_decimal 14) = base_6_to_decimal 22 := by
  sorry

#eval add_base_6 (base_6_to_decimal 4) (base_6_to_decimal 14)
#eval base_6_to_decimal 22

end NUMINAMATH_CALUDE_base_6_addition_l3349_334998


namespace NUMINAMATH_CALUDE_inequality_and_arithmetic_geometric_mean_l3349_334967

theorem inequality_and_arithmetic_geometric_mean :
  (∀ x y : ℝ, x > 0 → y > 0 → x^3 + y^3 ≥ x^2*y + x*y^2) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → (x^3 + y^3 = x^2*y + x*y^2 ↔ x = y)) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → (a + b + c) / 3 ≥ (a*b*c)^(1/3)) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → ((a + b + c) / 3 = (a*b*c)^(1/3) ↔ a = b ∧ b = c)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_arithmetic_geometric_mean_l3349_334967


namespace NUMINAMATH_CALUDE_younger_person_age_l3349_334919

/-- Given two people with an age difference of 20 years, where 15 years ago the elder was twice as old as the younger, prove that the younger person's present age is 35 years. -/
theorem younger_person_age (younger elder : ℕ) : 
  elder - younger = 20 →
  elder - 15 = 2 * (younger - 15) →
  younger = 35 := by
  sorry

end NUMINAMATH_CALUDE_younger_person_age_l3349_334919


namespace NUMINAMATH_CALUDE_ensemble_size_l3349_334949

/-- Represents the "Sunshine" ensemble --/
structure Ensemble where
  violin_players : ℕ
  bass_players : ℕ
  violin_avg_age : ℝ
  bass_avg_age : ℝ

/-- Represents the ensemble after Igor's switch --/
structure EnsembleAfterSwitch where
  violin_players : ℕ
  bass_players : ℕ
  violin_avg_age : ℝ
  bass_avg_age : ℝ

/-- Theorem stating the size of the ensemble --/
theorem ensemble_size (e : Ensemble) (e_after : EnsembleAfterSwitch) : 
  e.violin_players + e.bass_players = 23 :=
by
  have h1 : e.violin_avg_age = 22 := by sorry
  have h2 : e.bass_avg_age = 45 := by sorry
  have h3 : e_after.violin_players = e.violin_players + 1 := by sorry
  have h4 : e_after.bass_players = e.bass_players - 1 := by sorry
  have h5 : e_after.violin_avg_age = e.violin_avg_age + 1 := by sorry
  have h6 : e_after.bass_avg_age = e.bass_avg_age + 1 := by sorry
  sorry

#check ensemble_size

end NUMINAMATH_CALUDE_ensemble_size_l3349_334949


namespace NUMINAMATH_CALUDE_point_satisfies_constraint_local_maximum_at_point_main_theorem_l3349_334904

/-- The constraint function g(x₁, x₂) = x₁ - 2x₂ + 3 -/
def g (x₁ x₂ : ℝ) : ℝ := x₁ - 2*x₂ + 3

/-- The objective function f(x₁, x₂) = x₂² - x₁² -/
def f (x₁ x₂ : ℝ) : ℝ := x₂^2 - x₁^2

/-- The point (1, 2) satisfies the constraint -/
theorem point_satisfies_constraint : g 1 2 = 0 := by sorry

/-- The function f has a local maximum at (1, 2) under the constraint g(x₁, x₂) = 0 -/
theorem local_maximum_at_point :
  ∃ ε > 0, ∀ x₁ x₂ : ℝ, 
    g x₁ x₂ = 0 → 
    (x₁ - 1)^2 + (x₂ - 2)^2 < ε^2 → 
    f x₁ x₂ ≤ f 1 2 := by sorry

/-- The main theorem: f has a local maximum at (1, 2) under the constraint g(x₁, x₂) = 0 -/
theorem main_theorem : 
  ∃ (x₁ x₂ : ℝ), g x₁ x₂ = 0 ∧ 
  ∃ ε > 0, ∀ y₁ y₂ : ℝ, 
    g y₁ y₂ = 0 → 
    (y₁ - x₁)^2 + (y₂ - x₂)^2 < ε^2 → 
    f y₁ y₂ ≤ f x₁ x₂ :=
by
  use 1, 2
  constructor
  · exact point_satisfies_constraint
  · exact local_maximum_at_point

end NUMINAMATH_CALUDE_point_satisfies_constraint_local_maximum_at_point_main_theorem_l3349_334904


namespace NUMINAMATH_CALUDE_specific_hexagon_area_l3349_334918

/-- An irregular hexagon in 2D space -/
structure IrregularHexagon where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ
  v5 : ℝ × ℝ
  v6 : ℝ × ℝ

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Calculate the area of an irregular hexagon -/
def hexagonArea (h : IrregularHexagon) : ℝ := sorry

/-- The specific irregular hexagon from the problem -/
def specificHexagon : IrregularHexagon :=
  { v1 := (0, 0)
  , v2 := (2, 4)
  , v3 := (5, 4)
  , v4 := (7, 0)
  , v5 := (5, -4)
  , v6 := (2, -4) }

/-- Theorem: The area of the specific irregular hexagon is 32 square units -/
theorem specific_hexagon_area :
  hexagonArea specificHexagon = 32 := by sorry

end NUMINAMATH_CALUDE_specific_hexagon_area_l3349_334918


namespace NUMINAMATH_CALUDE_truck_travel_distance_truck_specific_distance_l3349_334954

/-- Represents the distance a truck can travel given an amount of gas -/
def distance_traveled (miles_per_gallon : ℝ) (gallons : ℝ) : ℝ :=
  miles_per_gallon * gallons

theorem truck_travel_distance 
  (initial_distance : ℝ) 
  (initial_gas : ℝ) 
  (new_gas : ℝ) : 
  initial_distance > 0 → 
  initial_gas > 0 → 
  new_gas > 0 → 
  distance_traveled (initial_distance / initial_gas) new_gas = 
    (initial_distance / initial_gas) * new_gas := by
  sorry

/-- Proves that a truck traveling 240 miles on 10 gallons of gas can travel 360 miles on 15 gallons of gas -/
theorem truck_specific_distance : 
  distance_traveled (240 / 10) 15 = 360 := by
  sorry

end NUMINAMATH_CALUDE_truck_travel_distance_truck_specific_distance_l3349_334954


namespace NUMINAMATH_CALUDE_hemisphere_base_area_l3349_334937

theorem hemisphere_base_area (r : ℝ) (h : r > 0) : 3 * Real.pi * r^2 = 9 → Real.pi * r^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_base_area_l3349_334937


namespace NUMINAMATH_CALUDE_ohara_triple_49_64_l3349_334964

/-- Definition of an O'Hara triple -/
def is_ohara_triple (a b y : ℕ) : Prop :=
  Real.sqrt a + Real.sqrt b = y

/-- Theorem: If (49, 64, y) is an O'Hara triple, then y = 15 -/
theorem ohara_triple_49_64 (y : ℕ) :
  is_ohara_triple 49 64 y → y = 15 := by
  sorry

end NUMINAMATH_CALUDE_ohara_triple_49_64_l3349_334964


namespace NUMINAMATH_CALUDE_sqrt_x_minus_3_real_range_l3349_334934

theorem sqrt_x_minus_3_real_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 3) → x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_3_real_range_l3349_334934


namespace NUMINAMATH_CALUDE_count_valid_domains_l3349_334927

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- Define the set of possible domain elements
def domain_elements : Set ℝ := {-Real.sqrt 2, -1, 1, Real.sqrt 2}

-- Define the range
def target_range : Set ℝ := {1, 2}

-- Define a valid domain
def is_valid_domain (S : Set ℝ) : Prop :=
  S ⊆ domain_elements ∧ f '' S = target_range

-- Theorem statement
theorem count_valid_domains :
  ∃ (valid_domains : Finset (Set ℝ)),
    (∀ S ∈ valid_domains, is_valid_domain S) ∧
    (∀ S, is_valid_domain S → S ∈ valid_domains) ∧
    valid_domains.card = 9 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_domains_l3349_334927


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l3349_334994

theorem unique_solution_to_equation :
  ∃! x : ℝ, x > 0 ∧ x^(Real.log x / Real.log 5) = x^2 / 25 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l3349_334994


namespace NUMINAMATH_CALUDE_square_of_sum_leq_sum_of_squares_l3349_334980

theorem square_of_sum_leq_sum_of_squares (a b : ℝ) :
  ((a + b) / 2) ^ 2 ≤ (a^2 + b^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_leq_sum_of_squares_l3349_334980


namespace NUMINAMATH_CALUDE_sin_increasing_on_interval_l3349_334953

-- Define the sine function (already defined in Mathlib)
-- def sin : ℝ → ℝ := Real.sin

-- Define the interval
def interval : Set ℝ := Set.Icc (-Real.pi / 2) (Real.pi / 2)

-- State the theorem
theorem sin_increasing_on_interval :
  StrictMonoOn Real.sin interval :=
sorry

end NUMINAMATH_CALUDE_sin_increasing_on_interval_l3349_334953


namespace NUMINAMATH_CALUDE_regular_octagon_diagonal_length_l3349_334930

/-- The length of a diagonal in a regular octagon -/
theorem regular_octagon_diagonal_length :
  ∀ (side_length : ℝ),
  side_length > 0 →
  ∃ (diagonal_length : ℝ),
  diagonal_length = side_length * Real.sqrt (2 + Real.sqrt 2) ∧
  diagonal_length^2 = 2 * side_length^2 + side_length^2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_diagonal_length_l3349_334930


namespace NUMINAMATH_CALUDE_exists_good_number_in_interval_l3349_334973

/-- A function that checks if a natural number is a "good number" (all digits ≤ 5) -/
def is_good_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d ≤ 5

/-- The main theorem: For any natural number x, there exists a "good number" y in [x, 9/5x) -/
theorem exists_good_number_in_interval (x : ℕ) : 
  ∃ y : ℕ, x ≤ y ∧ y < (9 * x) / 5 ∧ is_good_number y :=
sorry

end NUMINAMATH_CALUDE_exists_good_number_in_interval_l3349_334973


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3349_334950

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 0}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -2 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3349_334950


namespace NUMINAMATH_CALUDE_polynomial_equality_l3349_334900

theorem polynomial_equality (d e c : ℝ) : 
  (∀ x : ℝ, (6 * x^2 - 5 * x + 10/3) * (d * x^2 + e * x + c) = 
    18 * x^4 - 5 * x^3 + 15 * x^2 - (50/3) * x + 45/3) → 
  c = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3349_334900


namespace NUMINAMATH_CALUDE_board_covering_l3349_334989

-- Define a function to check if a board can be covered by dominoes
def can_cover_board (n m k : ℕ+) : Prop :=
  ∃ (arrangement : ℕ → ℕ → Bool), ∀ (i j : ℕ), i < n.val ∧ j < m.val →
    (arrangement i j = true ∧ arrangement (i + 1) j = true ∧ i + 1 < n.val) ∨
    (arrangement i j = true ∧ arrangement i (j + 1) = true ∧ j + 1 < m.val)

-- State the theorem
theorem board_covering (n m k : ℕ+) :
  can_cover_board n m k ↔ (k.val ∣ n.val ∨ k.val ∣ m.val) :=
sorry

end NUMINAMATH_CALUDE_board_covering_l3349_334989


namespace NUMINAMATH_CALUDE_democrats_to_participants_ratio_l3349_334962

/-- Proof of the ratio of democrats to total participants in a meeting --/
theorem democrats_to_participants_ratio :
  ∀ (total_participants : ℕ) 
    (female_democrats : ℕ) 
    (female_ratio : ℚ) 
    (male_ratio : ℚ),
  total_participants = 870 →
  female_democrats = 145 →
  female_ratio = 1/2 →
  male_ratio = 1/4 →
  (female_democrats * 2 + (total_participants - female_democrats * 2) * male_ratio) / total_participants = 1/3 :=
by
  sorry

#check democrats_to_participants_ratio

end NUMINAMATH_CALUDE_democrats_to_participants_ratio_l3349_334962


namespace NUMINAMATH_CALUDE_num_connecting_lines_correct_l3349_334907

/-- The number of straight lines connecting the intersection points of n intersecting lines -/
def num_connecting_lines (n : ℕ) : ℚ :=
  (n^2 * (n-1)^2 - 2*n * (n-1)) / 8

/-- Theorem stating that num_connecting_lines gives the correct number of lines -/
theorem num_connecting_lines_correct (n : ℕ) :
  num_connecting_lines n = (n^2 * (n-1)^2 - 2*n * (n-1)) / 8 :=
by sorry

end NUMINAMATH_CALUDE_num_connecting_lines_correct_l3349_334907


namespace NUMINAMATH_CALUDE_fish_ratio_l3349_334990

theorem fish_ratio (jerk_fish : ℕ) (total_fish : ℕ) : 
  jerk_fish = 144 → total_fish = 432 → 
  (total_fish - jerk_fish) / jerk_fish = 2 := by
sorry

end NUMINAMATH_CALUDE_fish_ratio_l3349_334990


namespace NUMINAMATH_CALUDE_x_minus_y_equals_half_l3349_334996

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {2, 0, x}
def B (x y : ℝ) : Set ℝ := {1/x, |x|, y/x}

-- State the theorem
theorem x_minus_y_equals_half (x y : ℝ) : A x = B x y → x - y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_half_l3349_334996


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l3349_334931

theorem inscribed_quadrilateral_fourth_side 
  (r : ℝ) 
  (a b c d : ℝ) 
  (θ : ℝ) 
  (h1 : r = 150 * Real.sqrt 2)
  (h2 : a = 150 ∧ b = 150 ∧ c = 150)
  (h3 : θ = 120 * π / 180) : 
  d = 375 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l3349_334931


namespace NUMINAMATH_CALUDE_square_sum_inequality_l3349_334903

theorem square_sum_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  a^2 + b^2 ≥ 1/2 := by sorry

end NUMINAMATH_CALUDE_square_sum_inequality_l3349_334903


namespace NUMINAMATH_CALUDE_final_bill_calculation_l3349_334942

def original_bill : ℝ := 400
def late_charge_rate : ℝ := 0.02

def final_amount : ℝ := original_bill * (1 + late_charge_rate)^3

theorem final_bill_calculation : 
  ∃ (ε : ℝ), abs (final_amount - 424.48) < ε ∧ ε > 0 :=
by sorry

end NUMINAMATH_CALUDE_final_bill_calculation_l3349_334942


namespace NUMINAMATH_CALUDE_simple_interest_rate_problem_l3349_334961

theorem simple_interest_rate_problem (P A T : ℝ) (h1 : P = 12500) (h2 : A = 18500) (h3 : T = 8) : 
  let SI := A - P
  let R := (SI * 100) / (P * T)
  R = 6 := by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_problem_l3349_334961


namespace NUMINAMATH_CALUDE_wall_bricks_count_l3349_334926

theorem wall_bricks_count (x : ℝ) 
  (h1 : x > 0)  -- Ensure positive number of bricks
  (h2 : (x / 8 + x / 12 - 15) > 0)  -- Ensure positive combined rate
  (h3 : 6 * (x / 8 + x / 12 - 15) = x)  -- Equation from working together for 6 hours
  : x = 360 := by
  sorry

end NUMINAMATH_CALUDE_wall_bricks_count_l3349_334926


namespace NUMINAMATH_CALUDE_second_chapter_pages_l3349_334963

theorem second_chapter_pages (total_chapters : ℕ) (total_pages : ℕ) (second_chapter_length : ℕ) :
  total_chapters = 2 →
  total_pages = 81 →
  second_chapter_length = 68 →
  second_chapter_length = 68 := by
  sorry

end NUMINAMATH_CALUDE_second_chapter_pages_l3349_334963


namespace NUMINAMATH_CALUDE_sequence_difference_l3349_334976

def sequence_property (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) : Prop :=
  (∀ n : ℕ+, a n ≠ 0) ∧ 
  (∀ n : ℕ+, a n * a (n + 1) = S n)

theorem sequence_difference (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) 
  (h : sequence_property a S) : a 3 - a 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_difference_l3349_334976


namespace NUMINAMATH_CALUDE_joan_seashells_l3349_334910

theorem joan_seashells (sam_shells : ℕ) (total_shells : ℕ) (joan_shells : ℕ) 
  (h1 : sam_shells = 35)
  (h2 : total_shells = 53)
  (h3 : total_shells = sam_shells + joan_shells) :
  joan_shells = 18 := by
  sorry

end NUMINAMATH_CALUDE_joan_seashells_l3349_334910


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3349_334974

theorem inequality_solution_set (a b : ℝ) (d : ℝ) (h1 : a > 0) 
  (h2 : ∀ x : ℝ, x^2 + a*x + b > 0 ↔ x ≠ d) :
  ∃ x₁ x₂ : ℝ, (∀ x : ℝ, x^2 + a*x - b < 0 ↔ x₁ < x ∧ x < x₂) ∧ x₁ * x₂ ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3349_334974


namespace NUMINAMATH_CALUDE_square_side_length_l3349_334991

theorem square_side_length 
  (total_width : ℕ) 
  (total_height : ℕ) 
  (r : ℕ) 
  (s : ℕ) :
  total_width = 3300 →
  total_height = 2000 →
  2 * r + s = total_height →
  2 * r + 3 * s = total_width →
  s = 650 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3349_334991


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l3349_334995

theorem quadratic_two_roots 
  (a b c α : ℝ) 
  (f : ℝ → ℝ) 
  (h_quad : ∀ x, f x = a * x^2 + b * x + c) 
  (h_exists : a * f α < 0) : 
  ∃ x₁ x₂, x₁ < α ∧ α < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l3349_334995


namespace NUMINAMATH_CALUDE_square_side_length_l3349_334905

/-- Given a circle with area 100 and a square whose perimeter equals the circle's area,
    the length of one side of the square is 25. -/
theorem square_side_length (circle_area : ℝ) (square_perimeter : ℝ) :
  circle_area = 100 →
  square_perimeter = circle_area →
  square_perimeter = 4 * 25 :=
by
  sorry

#check square_side_length

end NUMINAMATH_CALUDE_square_side_length_l3349_334905
