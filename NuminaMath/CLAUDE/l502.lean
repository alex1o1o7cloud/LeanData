import Mathlib

namespace NUMINAMATH_CALUDE_number_of_boys_l502_50249

theorem number_of_boys (total_amount : ℕ) (total_people : ℕ) (boy_amount : ℕ) (girl_amount : ℕ) 
  (h1 : total_amount = 460)
  (h2 : total_people = 41)
  (h3 : boy_amount = 12)
  (h4 : girl_amount = 8) :
  ∃ (boys : ℕ), boys = 33 ∧ 
  boys * boy_amount + (total_people - boys) * girl_amount = total_amount :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_boys_l502_50249


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l502_50229

theorem arithmetic_mean_problem (y : ℝ) : 
  (8 + 12 + 23 + 17 + y) / 5 = 15 → y = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l502_50229


namespace NUMINAMATH_CALUDE_A_C_mutually_exclusive_not_complementary_l502_50251

-- Define the event space
def Ω : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define events A, B, and C
def A : Set Nat := {1, 3}
def B : Set Nat := {3, 5}
def C : Set Nat := {2, 4, 6}

-- Define mutually exclusive events
def mutually_exclusive (X Y : Set Nat) : Prop := X ∩ Y = ∅

-- Define complementary events
def complementary (X Y : Set Nat) : Prop := X ∪ Y = Ω ∧ X ∩ Y = ∅

-- Theorem to prove
theorem A_C_mutually_exclusive_not_complementary :
  mutually_exclusive A C ∧ ¬complementary A C :=
sorry

end NUMINAMATH_CALUDE_A_C_mutually_exclusive_not_complementary_l502_50251


namespace NUMINAMATH_CALUDE_tan_138_less_than_tan_143_l502_50241

theorem tan_138_less_than_tan_143 :
  let angle1 : Real := 138 * π / 180
  let angle2 : Real := 143 * π / 180
  (π / 2 < angle1 ∧ angle1 < π) →
  (π / 2 < angle2 ∧ angle2 < π) →
  (∀ x y, π / 2 < x ∧ x < y ∧ y < π → Real.tan x > Real.tan y) →
  Real.tan angle1 < Real.tan angle2 :=
by
  sorry

end NUMINAMATH_CALUDE_tan_138_less_than_tan_143_l502_50241


namespace NUMINAMATH_CALUDE_painting_time_equation_l502_50293

/-- The time it takes Sarah to paint the room alone (in hours) -/
def sarah_time : ℝ := 4

/-- The time it takes Tom to paint the room alone (in hours) -/
def tom_time : ℝ := 6

/-- The duration of the break (in hours) -/
def break_time : ℝ := 2

/-- The total time it takes Sarah and Tom to paint the room together, including the break (in hours) -/
noncomputable def total_time : ℝ := sorry

/-- Theorem stating the equation that the total time satisfies -/
theorem painting_time_equation :
  (1 / sarah_time + 1 / tom_time) * (total_time - break_time) = 1 := by sorry

end NUMINAMATH_CALUDE_painting_time_equation_l502_50293


namespace NUMINAMATH_CALUDE_specific_flowerbed_area_l502_50210

/-- Represents a circular flowerbed with a straight path through its center -/
structure Flowerbed where
  diameter : ℝ
  pathWidth : ℝ

/-- Calculates the plantable area of a flowerbed -/
def plantableArea (f : Flowerbed) : ℝ := sorry

/-- Theorem stating the plantable area of a specific flowerbed configuration -/
theorem specific_flowerbed_area :
  let f : Flowerbed := { diameter := 20, pathWidth := 4 }
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |plantableArea f - 58.66 * Real.pi| < ε :=
sorry

end NUMINAMATH_CALUDE_specific_flowerbed_area_l502_50210


namespace NUMINAMATH_CALUDE_root_sum_reciprocals_l502_50277

theorem root_sum_reciprocals (p q r s : ℂ) : 
  (p^4 - 6*p^3 + 11*p^2 - 6*p + 3 = 0) →
  (q^4 - 6*q^3 + 11*q^2 - 6*q + 3 = 0) →
  (r^4 - 6*r^3 + 11*r^2 - 6*r + 3 = 0) →
  (s^4 - 6*s^3 + 11*s^2 - 6*s + 3 = 0) →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 11/3 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocals_l502_50277


namespace NUMINAMATH_CALUDE_game_probability_theorem_l502_50205

def game_probability (total_rounds : ℕ) 
                     (alex_prob : ℚ) 
                     (mel_chelsea_ratio : ℕ) 
                     (alex_wins mel_wins chelsea_wins : ℕ) : Prop :=
  let mel_prob := (1 - alex_prob) * (mel_chelsea_ratio / (mel_chelsea_ratio + 1 : ℚ))
  let chelsea_prob := (1 - alex_prob) * (1 / (mel_chelsea_ratio + 1 : ℚ))
  let specific_outcome_prob := alex_prob ^ alex_wins * mel_prob ^ mel_wins * chelsea_prob ^ chelsea_wins
  let arrangements := Nat.choose total_rounds alex_wins * Nat.choose (total_rounds - alex_wins) mel_wins
  (specific_outcome_prob * arrangements : ℚ) = 76545 / 823543

theorem game_probability_theorem : 
  game_probability 7 (3/7) 3 4 2 1 :=
sorry

end NUMINAMATH_CALUDE_game_probability_theorem_l502_50205


namespace NUMINAMATH_CALUDE_lenny_pens_boxes_l502_50284

theorem lenny_pens_boxes : ∀ (total_pens : ℕ) (pens_per_box : ℕ),
  pens_per_box = 5 →
  (total_pens : ℚ) * (3 / 5 : ℚ) * (3 / 4 : ℚ) = 45 →
  total_pens / pens_per_box = 20 :=
by
  sorry

#check lenny_pens_boxes

end NUMINAMATH_CALUDE_lenny_pens_boxes_l502_50284


namespace NUMINAMATH_CALUDE_boat_round_trip_time_l502_50245

/-- Calculates the total time for a round trip boat journey given the boat's speed in standing water, 
    the stream's speed, and the distance to the destination. -/
theorem boat_round_trip_time 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (distance : ℝ) 
  (h1 : boat_speed = 8) 
  (h2 : stream_speed = 6) 
  (h3 : distance = 210) : 
  distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed) = 120 := by
  sorry

#check boat_round_trip_time

end NUMINAMATH_CALUDE_boat_round_trip_time_l502_50245


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l502_50201

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁^2 - 4*x₁ - 5 = 0 ∧ 
  x₂^2 - 4*x₂ - 5 = 0 ∧ 
  x₁ = 5 ∧ 
  x₂ = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l502_50201


namespace NUMINAMATH_CALUDE_function_inequality_l502_50237

open Real

noncomputable def f (x : ℝ) : ℝ := x / cos x

theorem function_inequality (x₁ x₂ x₃ : ℝ) 
  (h₁ : |x₁| < π/2) (h₂ : |x₂| < π/2) (h₃ : |x₃| < π/2)
  (h₄ : f x₁ + f x₂ ≥ 0) (h₅ : f x₂ + f x₃ ≥ 0) (h₆ : f x₃ + f x₁ ≥ 0) :
  f (x₁ + x₂ + x₃) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l502_50237


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l502_50255

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    if the distance from one focus to an asymptote is √5/3 * c,
    where c is the semi-focal length, then the eccentricity is 3/2. -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) →
  (c * b / Real.sqrt (a^2 + b^2) = c * Real.sqrt 5 / 3) →
  c^2 = a^2 + b^2 →
  c / a = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l502_50255


namespace NUMINAMATH_CALUDE_book_arrangement_count_l502_50212

/-- The number of ways to arrange two types of indistinguishable objects in a row -/
def arrangement_count (n m : ℕ) : ℕ :=
  Nat.choose (n + m) n

/-- Theorem: Arranging 4 copies of one book and 5 copies of another book yields 126 possibilities -/
theorem book_arrangement_count :
  arrangement_count 4 5 = 126 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l502_50212


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l502_50207

theorem triangle_angle_calculation (a b c : ℝ) (A B C : ℝ) :
  a = 3 * Real.sqrt 2 →
  c = 3 →
  C = π / 6 →
  A = π / 4 ∨ A = 3 * π / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l502_50207


namespace NUMINAMATH_CALUDE_ratio_problem_l502_50272

theorem ratio_problem (a b x m : ℝ) : 
  a > 0 ∧ b > 0 ∧ 
  a / b = 4 / 5 ∧
  x = a * (1 + 0.25) ∧
  m = b * (1 - 0.80) →
  m / x = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l502_50272


namespace NUMINAMATH_CALUDE_min_bilingual_students_l502_50220

theorem min_bilingual_students (total : ℕ) (hindi : ℕ) (english : ℕ) 
  (h_total : total = 40)
  (h_hindi : hindi = 30)
  (h_english : english = 20) :
  ∃ (both : ℕ), both ≥ hindi + english - total ∧ 
    ∀ (x : ℕ), x ≥ hindi + english - total → x ≥ both :=
by sorry

end NUMINAMATH_CALUDE_min_bilingual_students_l502_50220


namespace NUMINAMATH_CALUDE_m_values_l502_50256

def A : Set ℝ := {1, 3}

def B (m : ℝ) : Set ℝ := {x | m * x - 3 = 0}

theorem m_values (m : ℝ) : A ∪ B m = A → m ∈ ({0, 1, 3} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_m_values_l502_50256


namespace NUMINAMATH_CALUDE_set_union_problem_l502_50215

theorem set_union_problem (a b : ℕ) : 
  let A : Set ℕ := {5, 2^a}
  let B : Set ℕ := {a, b}
  A ∩ B = {8} →
  A ∪ B = {3, 5, 8} := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l502_50215


namespace NUMINAMATH_CALUDE_stewart_farm_horse_food_l502_50268

/-- Proves that given the ratio of sheep to horses is 4:7, there are 32 sheep on the farm,
    and the farm needs a total of 12,880 ounces of horse food per day,
    each horse needs 230 ounces of horse food per day. -/
theorem stewart_farm_horse_food (sheep : ℕ) (horses : ℕ) (total_food : ℕ) :
  sheep = 32 →
  4 * horses = 7 * sheep →
  total_food = 12880 →
  total_food / horses = 230 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_horse_food_l502_50268


namespace NUMINAMATH_CALUDE_exist_common_members_l502_50299

/-- A structure representing a parliament with committees -/
structure Parliament :=
  (members : Finset ℕ)
  (committees : Finset (Finset ℕ))
  (h_member_count : members.card = 1600)
  (h_committee_count : committees.card = 16000)
  (h_committee_size : ∀ c ∈ committees, c.card = 80)
  (h_committees_subset : ∀ c ∈ committees, c ⊆ members)

/-- Theorem stating that there exist at least two committees with at least 4 common members -/
theorem exist_common_members (p : Parliament) :
  ∃ c1 c2 : Finset ℕ, c1 ∈ p.committees ∧ c2 ∈ p.committees ∧ c1 ≠ c2 ∧ (c1 ∩ c2).card ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_exist_common_members_l502_50299


namespace NUMINAMATH_CALUDE_specific_arrangement_probability_l502_50208

def num_red_lamps : ℕ := 4
def num_blue_lamps : ℕ := 4
def num_lamps_on : ℕ := 4

def probability_specific_arrangement : ℚ := 3/49

theorem specific_arrangement_probability :
  let total_lamps := num_red_lamps + num_blue_lamps
  let total_arrangements := (total_lamps.choose num_red_lamps) * (total_lamps.choose num_lamps_on)
  let favorable_outcomes := (total_lamps - 2).choose (num_red_lamps - 1) * (total_lamps - 2).choose (num_lamps_on - 1)
  (favorable_outcomes : ℚ) / total_arrangements = probability_specific_arrangement :=
sorry

end NUMINAMATH_CALUDE_specific_arrangement_probability_l502_50208


namespace NUMINAMATH_CALUDE_k_mod_8_l502_50274

/-- An integer m covers 1998 if 1, 9, 9, 8 appear in this order as digits of m. -/
def covers_1998 (m : ℕ) : Prop := sorry

/-- k(n) is the number of positive integers that cover 1998 and have exactly n digits, all different from 0. -/
def k (n : ℕ) : ℕ := sorry

/-- The main theorem: k(n) is congruent to 1 modulo 8 for all n ≥ 5. -/
theorem k_mod_8 (n : ℕ) (h : n ≥ 5) : k n ≡ 1 [MOD 8] := by sorry

end NUMINAMATH_CALUDE_k_mod_8_l502_50274


namespace NUMINAMATH_CALUDE_five_pages_thirty_lines_each_l502_50243

/-- Given a page capacity and number of pages, calculates the total lines of information. -/
def total_lines (lines_per_page : ℕ) (num_pages : ℕ) : ℕ :=
  lines_per_page * num_pages

/-- Theorem stating that 5 pages with 30 lines each result in 150 total lines. -/
theorem five_pages_thirty_lines_each :
  total_lines 30 5 = 150 := by
  sorry

end NUMINAMATH_CALUDE_five_pages_thirty_lines_each_l502_50243


namespace NUMINAMATH_CALUDE_two_zeros_neither_necessary_nor_sufficient_l502_50213

open Real

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ) (f' : ℝ → ℝ)

-- Define the property of f' having exactly two zeros in (0, 2)
def has_two_zeros (f' : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 ∧ f' x₁ = 0 ∧ f' x₂ = 0 ∧
  ∀ x, 0 < x ∧ x < 2 ∧ f' x = 0 → x = x₁ ∨ x = x₂

-- Define the property of f having exactly two extreme points in (0, 2)
def has_two_extreme_points (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 ∧ f' x₁ = 0 ∧ f' x₂ = 0 ∧
  (∀ x, 0 < x ∧ x < x₁ → f' x ≠ 0) ∧
  (∀ x, x₁ < x ∧ x < x₂ → f' x ≠ 0) ∧
  (∀ x, x₂ < x ∧ x < 2 → f' x ≠ 0)

-- Theorem stating that has_two_zeros is neither necessary nor sufficient for has_two_extreme_points
theorem two_zeros_neither_necessary_nor_sufficient :
  ¬(∀ f f', has_two_zeros f' → has_two_extreme_points f f') ∧
  ¬(∀ f f', has_two_extreme_points f f' → has_two_zeros f') :=
sorry

end NUMINAMATH_CALUDE_two_zeros_neither_necessary_nor_sufficient_l502_50213


namespace NUMINAMATH_CALUDE_number_puzzle_l502_50246

theorem number_puzzle : ∃ x : ℤ, (x + 2) - 3 = 7 ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l502_50246


namespace NUMINAMATH_CALUDE_dog_reachable_area_theorem_l502_50269

/-- The area outside a regular hexagonal doghouse that a dog can reach when tethered to a vertex -/
def dogReachableArea (side_length : Real) (rope_length : Real) : Real :=
  -- Definition to be filled
  sorry

/-- Theorem stating the area the dog can reach outside the doghouse -/
theorem dog_reachable_area_theorem :
  dogReachableArea 1 4 = (82 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_dog_reachable_area_theorem_l502_50269


namespace NUMINAMATH_CALUDE_milk_cans_problem_l502_50240

theorem milk_cans_problem (x y : ℕ) : 
  x = 2 * y ∧ 
  x - 30 = 3 * (y - 20) → 
  x = 60 ∧ y = 30 := by
sorry

end NUMINAMATH_CALUDE_milk_cans_problem_l502_50240


namespace NUMINAMATH_CALUDE_prove_abc_equation_l502_50282

theorem prove_abc_equation (a b c : ℝ) 
  (h1 : a^4 * b^3 * c^5 = 18) 
  (h2 : a^3 * b^5 * c^4 = 8) : 
  a^5 * b * c^6 = 81/2 := by
  sorry

end NUMINAMATH_CALUDE_prove_abc_equation_l502_50282


namespace NUMINAMATH_CALUDE_cos_pi_minus_2alpha_l502_50279

theorem cos_pi_minus_2alpha (α : ℝ) (h : Real.sin (3 * Real.pi / 2 + α) = 1 / 3) :
  Real.cos (Real.pi - 2 * α) = - 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_minus_2alpha_l502_50279


namespace NUMINAMATH_CALUDE_polka_dot_price_is_67_l502_50214

def checkered_price : ℝ := 75
def plain_price : ℝ := 45
def striped_price : ℝ := 63
def total_price : ℝ := 250

def checkered_per_yard : ℝ := 7.5
def plain_per_yard : ℝ := 6
def striped_per_yard : ℝ := 9
def polka_dot_per_yard : ℝ := 4.5

def discount_rate : ℝ := 0.1
def discount_threshold : ℝ := 10

def polka_dot_price : ℝ := total_price - (checkered_price + plain_price + striped_price)

theorem polka_dot_price_is_67 : polka_dot_price = 67 := by
  sorry

end NUMINAMATH_CALUDE_polka_dot_price_is_67_l502_50214


namespace NUMINAMATH_CALUDE_charlotte_theorem_l502_50298

/-- Represents the state of boxes after each step of adding marbles --/
def BoxState (n : ℕ) := ℕ → ℕ

/-- Initial state of boxes --/
def initial_state (n : ℕ) : BoxState n :=
  λ i => if i ≤ n ∧ i > 0 then i else 0

/-- Add a marble to each box --/
def add_to_all (state : BoxState n) : BoxState n :=
  λ i => state i + 1

/-- Add a marble to boxes divisible by k --/
def add_to_divisible (k : ℕ) (state : BoxState n) : BoxState n :=
  λ i => if state i % k = 0 then state i + 1 else state i

/-- Perform Charlotte's procedure --/
def charlotte_procedure (n : ℕ) : BoxState n :=
  let initial := initial_state n
  let after_first_step := add_to_all initial
  (List.range n).foldl (λ state k => add_to_divisible (k + 2) state) after_first_step

/-- Check if all boxes have exactly n+1 marbles --/
def all_boxes_have_n_plus_one (n : ℕ) (state : BoxState n) : Prop :=
  ∀ i, i > 0 → i ≤ n → state i = n + 1

/-- The main theorem --/
theorem charlotte_theorem (n : ℕ) :
  all_boxes_have_n_plus_one n (charlotte_procedure n) ↔ Nat.Prime (n + 1) :=
sorry

end NUMINAMATH_CALUDE_charlotte_theorem_l502_50298


namespace NUMINAMATH_CALUDE_provider_choice_count_l502_50200

/-- The total number of service providers --/
def total_providers : ℕ := 25

/-- The number of providers available to the youngest child --/
def restricted_providers : ℕ := 15

/-- The number of children --/
def num_children : ℕ := 4

/-- The number of ways to choose service providers for the children --/
def choose_providers : ℕ := total_providers * (total_providers - 1) * (total_providers - 2) * restricted_providers

theorem provider_choice_count :
  choose_providers = 207000 :=
sorry

end NUMINAMATH_CALUDE_provider_choice_count_l502_50200


namespace NUMINAMATH_CALUDE_train_crossing_time_l502_50276

/-- Theorem: Time taken for two trains to cross each other
    Given two trains moving in opposite directions with specified speeds and lengths,
    prove that the time taken for the slower train to cross the faster train is 24 seconds. -/
theorem train_crossing_time (speed1 speed2 length1 length2 : ℝ) 
    (h1 : speed1 = 315)
    (h2 : speed2 = 135)
    (h3 : length1 = 1.65)
    (h4 : length2 = 1.35) :
    (length1 + length2) / (speed1 + speed2) * 3600 = 24 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l502_50276


namespace NUMINAMATH_CALUDE_like_terms_exponent_product_l502_50296

theorem like_terms_exponent_product (a b : ℤ) : 
  (6 = -2 * a) → (b = 2) → a * b = -6 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_product_l502_50296


namespace NUMINAMATH_CALUDE_truth_table_results_l502_50265

variable (p q : Prop)

theorem truth_table_results :
  (∀ p, ¬(p ∧ ¬p)) ∧
  (∀ p, p ∨ ¬p) ∧
  (∀ p q, ¬(p ∧ q) ↔ (¬p ∨ ¬q)) ∧
  (∀ p q, (p ∨ q) ∨ ¬p) :=
by sorry

end NUMINAMATH_CALUDE_truth_table_results_l502_50265


namespace NUMINAMATH_CALUDE_theater_ticket_difference_l502_50261

theorem theater_ticket_difference :
  ∀ (orchestra_price balcony_price : ℕ) 
    (total_tickets total_cost : ℕ) 
    (orchestra_tickets balcony_tickets : ℕ),
  orchestra_price = 18 →
  balcony_price = 12 →
  total_tickets = 450 →
  total_cost = 6300 →
  orchestra_tickets + balcony_tickets = total_tickets →
  orchestra_price * orchestra_tickets + balcony_price * balcony_tickets = total_cost →
  balcony_tickets - orchestra_tickets = 150 := by
sorry

end NUMINAMATH_CALUDE_theater_ticket_difference_l502_50261


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_squared_l502_50257

theorem sum_of_fourth_powers_squared (A B C : ℤ) (h : A + B + C = 0) :
  2 * (A^4 + B^4 + C^4) = (A^2 + B^2 + C^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_squared_l502_50257


namespace NUMINAMATH_CALUDE_lcm_18_35_l502_50281

theorem lcm_18_35 : Nat.lcm 18 35 = 630 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_35_l502_50281


namespace NUMINAMATH_CALUDE_bottom_right_figure_impossible_l502_50278

/-- Represents a rhombus with a fixed white and gray pattern -/
structure Rhombus :=
  (pattern : ℕ → ℕ → Bool)

/-- Represents a rotation of a rhombus -/
def rotate (r : Rhombus) (angle : ℕ) : Rhombus :=
  sorry

/-- Represents a larger figure composed of rhombuses -/
structure LargeFigure :=
  (shape : List (Rhombus × ℕ × ℕ))

/-- The specific larger figure that cannot be assembled (bottom right) -/
def bottomRightFigure : LargeFigure :=
  sorry

/-- Predicate to check if a larger figure can be assembled using only rotations of the given rhombus -/
def canAssemble (r : Rhombus) (lf : LargeFigure) : Prop :=
  sorry

/-- Theorem stating that the bottom right figure cannot be assembled -/
theorem bottom_right_figure_impossible (r : Rhombus) :
  ¬ (canAssemble r bottomRightFigure) :=
sorry

end NUMINAMATH_CALUDE_bottom_right_figure_impossible_l502_50278


namespace NUMINAMATH_CALUDE_counterexample_exists_l502_50231

theorem counterexample_exists : ∃ n : ℕ, 
  (Even n) ∧ (¬ Prime n) ∧ (¬ Prime (n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l502_50231


namespace NUMINAMATH_CALUDE_total_students_at_competition_l502_50218

/-- The number of students from each school at a science fair competition --/
structure SchoolAttendance where
  quantum : ℕ
  schrodinger : ℕ
  einstein : ℕ
  newton : ℕ
  galileo : ℕ
  pascal : ℕ
  faraday : ℕ

/-- The conditions of the science fair competition --/
def scienceFairConditions (s : SchoolAttendance) : Prop :=
  s.quantum = 90 ∧
  s.schrodinger = (2 * s.quantum) / 3 ∧
  s.einstein = (4 * s.schrodinger) / 9 ∧
  s.newton = (5 * s.einstein) / 12 ∧
  s.galileo = (11 * s.newton) / 20 ∧
  s.pascal = (13 * s.galileo) / 50 ∧
  s.faraday = 4 * (s.quantum + s.schrodinger + s.einstein + s.newton + s.galileo + s.pascal)

/-- The theorem stating the total number of students at the competition --/
theorem total_students_at_competition (s : SchoolAttendance) 
  (h : scienceFairConditions s) : 
  s.quantum + s.schrodinger + s.einstein + s.newton + s.galileo + s.pascal + s.faraday = 980 := by
  sorry

end NUMINAMATH_CALUDE_total_students_at_competition_l502_50218


namespace NUMINAMATH_CALUDE_sum_in_base7_l502_50248

/-- Converts a base 7 number represented as a list of digits to its decimal equivalent -/
def toDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 7 * acc + d) 0

/-- Converts a decimal number to its base 7 representation as a list of digits -/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem sum_in_base7 :
  toBase7 (toDecimal [4, 2, 3, 1] + toDecimal [1, 3, 5, 2, 6]) = [6, 0, 0, 6, 0] := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base7_l502_50248


namespace NUMINAMATH_CALUDE_domain_of_f_l502_50253

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem domain_of_f (f : ℝ → ℝ) :
  (∀ x, f (2 * x - 3) ≠ 0 → -2 ≤ x ∧ x ≤ 2) →
  (∀ y, f y ≠ 0 → -7 ≤ y ∧ y ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_domain_of_f_l502_50253


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l502_50280

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 10
  let θ : ℝ := 3 * π / 4
  let φ : ℝ := π / 4
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x = -5 ∧ y = 5 ∧ z = 5 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l502_50280


namespace NUMINAMATH_CALUDE_ceiling_equality_abs_diff_l502_50289

-- Define the ceiling function
noncomputable def ceiling (x : ℝ) : ℤ :=
  Int.ceil x

-- State the theorem
theorem ceiling_equality_abs_diff (x y : ℝ) :
  (∀ x y, ceiling x = ceiling y → |x - y| < 1) ∧
  (∃ x y, |x - y| < 1 ∧ ceiling x ≠ ceiling y) :=
by sorry

end NUMINAMATH_CALUDE_ceiling_equality_abs_diff_l502_50289


namespace NUMINAMATH_CALUDE_polynomial_factorization_l502_50219

theorem polynomial_factorization (x : ℤ) : 
  (x^3 - x^2 + 2*x - 1) * (x^3 - x - 1) = x^6 - x^5 + x^4 - x^3 - x^2 - x + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l502_50219


namespace NUMINAMATH_CALUDE_factorization_equality_l502_50216

theorem factorization_equality (x y : ℝ) : 
  4 * (x - y + 1) + y * (y - 2 * x) = (y - 2) * (y - 2 - 2 * x) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l502_50216


namespace NUMINAMATH_CALUDE_sum_reciprocals_of_sum_and_product_l502_50204

theorem sum_reciprocals_of_sum_and_product (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (hsum : x + y = 10) (hprod : x * y = 20) : 
  1 / x + 1 / y = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_of_sum_and_product_l502_50204


namespace NUMINAMATH_CALUDE_point_on_number_line_l502_50209

/-- Given points P, Q, and R on a number line, where Q is halfway between P and R,
    P is at -6, and Q is at -1, prove that R is at 4. -/
theorem point_on_number_line (P Q R : ℝ) : 
  Q = (P + R) / 2 → P = -6 → Q = -1 → R = 4 := by
  sorry

end NUMINAMATH_CALUDE_point_on_number_line_l502_50209


namespace NUMINAMATH_CALUDE_andreas_living_room_area_l502_50264

theorem andreas_living_room_area :
  ∀ (room_area carpet_area : ℝ),
    carpet_area = 6 * 12 →
    room_area * 0.2 = carpet_area →
    room_area = 360 :=
by
  sorry

end NUMINAMATH_CALUDE_andreas_living_room_area_l502_50264


namespace NUMINAMATH_CALUDE_max_sum_cubes_max_sum_cubes_achieved_l502_50206

theorem max_sum_cubes (a b c d e : ℝ) (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) :
  a^3 + b^3 + c^3 + d^3 + e^3 ≤ 5 * Real.sqrt 5 :=
by sorry

theorem max_sum_cubes_achieved (h : ∃ a b c d e : ℝ, a^2 + b^2 + c^2 + d^2 + e^2 = 5 ∧ a^3 + b^3 + c^3 + d^3 + e^3 = 5 * Real.sqrt 5) :
  ∃ a b c d e : ℝ, a^2 + b^2 + c^2 + d^2 + e^2 = 5 ∧ a^3 + b^3 + c^3 + d^3 + e^3 = 5 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_cubes_max_sum_cubes_achieved_l502_50206


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_9883_l502_50226

theorem largest_prime_factor_of_9883 :
  ∃ (p : ℕ), p.Prime ∧ p ∣ 9883 ∧ ∀ (q : ℕ), q.Prime → q ∣ 9883 → q ≤ p :=
by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_9883_l502_50226


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l502_50285

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

def arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b

theorem geometric_sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a (n + 1) = 2 * a n) →
  arithmetic_sequence (a 2) (a 3 + 1) (a 4) →
  (∀ n : ℕ, b n = a n + n) →
  a 1 = 1 ∧
  (∀ n : ℕ, a n = 2^(n - 1)) ∧
  (b 1 + b 2 + b 3 + b 4 + b 5 = 46) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l502_50285


namespace NUMINAMATH_CALUDE_farmer_plots_allocation_l502_50275

theorem farmer_plots_allocation (x y : ℕ) (h : x ≠ y) : ∃ (a b : ℕ), a^2 + b^2 = 2 * (x^2 + y^2) := by
  sorry

end NUMINAMATH_CALUDE_farmer_plots_allocation_l502_50275


namespace NUMINAMATH_CALUDE_list_fraction_problem_l502_50292

theorem list_fraction_problem (l : List ℝ) (n : ℝ) (h1 : l.length = 21) 
  (h2 : n ∈ l) (h3 : n = 4 * ((l.sum - n) / 20)) : 
  n = (1 / 6) * l.sum :=
sorry

end NUMINAMATH_CALUDE_list_fraction_problem_l502_50292


namespace NUMINAMATH_CALUDE_ellipse_a_range_l502_50247

-- Define the ellipse equation
def ellipse_equation (x y a : ℝ) : Prop :=
  x^2 / a^2 + y^2 / (a + 6) = 1

-- Define the condition that the ellipse has foci on the x-axis
def foci_on_x_axis (a : ℝ) : Prop :=
  a^2 > a + 6 ∧ a + 6 > 0

-- Theorem stating the range of a
theorem ellipse_a_range :
  ∀ a : ℝ, (∃ x y : ℝ, ellipse_equation x y a ∧ foci_on_x_axis a) →
  (a > 3 ∨ (-6 < a ∧ a < -2)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_a_range_l502_50247


namespace NUMINAMATH_CALUDE_intersection_x_sum_zero_l502_50232

theorem intersection_x_sum_zero (x₁ x₂ : ℝ) : 
  x₁^2 + 9^2 = 169 → x₂^2 + 9^2 = 169 → x₁ + x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_x_sum_zero_l502_50232


namespace NUMINAMATH_CALUDE_natural_equation_example_natural_equation_condition_l502_50234

-- Definition of a natural equation
def is_natural_equation (a b c : ℤ) : Prop :=
  ∃ x₁ x₂ : ℤ, (a * x₁^2 + b * x₁ + c = 0) ∧ 
              (a * x₂^2 + b * x₂ + c = 0) ∧ 
              (abs (x₁ - x₂) = 1) ∧
              (a ≠ 0)

-- Theorem 1: x² + 3x + 2 = 0 is a natural equation
theorem natural_equation_example : is_natural_equation 1 3 2 := by
  sorry

-- Theorem 2: x² - (m+1)x + m = 0 is a natural equation iff m = 0 or m = 2
theorem natural_equation_condition (m : ℤ) : 
  is_natural_equation 1 (-(m+1)) m ↔ m = 0 ∨ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_natural_equation_example_natural_equation_condition_l502_50234


namespace NUMINAMATH_CALUDE_multiplier_is_three_l502_50262

theorem multiplier_is_three :
  ∃ (x : ℤ), 
    (3 * x = (62 - x) + 26) ∧ 
    (x = 22) → 
    3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_is_three_l502_50262


namespace NUMINAMATH_CALUDE_root_sum_square_theorem_l502_50227

theorem root_sum_square_theorem (m n : ℝ) : 
  (m^2 + 2*m - 2025 = 0) → 
  (n^2 + 2*n - 2025 = 0) → 
  (m ≠ n) →
  (m^2 + 3*m + n = 2023) := by
sorry

end NUMINAMATH_CALUDE_root_sum_square_theorem_l502_50227


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l502_50230

theorem polynomial_remainder_theorem (x : ℝ) : 
  (4 * x^3 - 10 * x^2 + 15 * x - 17) % (4 * x - 8) = 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l502_50230


namespace NUMINAMATH_CALUDE_ellipse_t_range_l502_50250

def is_ellipse (t : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / (10 - t) + y^2 / (t - 4) = 1

theorem ellipse_t_range :
  {t : ℝ | is_ellipse t} = {t | t ∈ (Set.Ioo 4 7) ∪ (Set.Ioo 7 10)} :=
by sorry

end NUMINAMATH_CALUDE_ellipse_t_range_l502_50250


namespace NUMINAMATH_CALUDE_root_implies_m_value_always_real_roots_l502_50228

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 - m*x + m - 1

-- Theorem 1: If x = 3 is a root, then m = 4
theorem root_implies_m_value (m : ℝ) : quadratic m 3 = 0 → m = 4 := by sorry

-- Theorem 2: The quadratic equation always has real roots
theorem always_real_roots (m : ℝ) : ∃ x : ℝ, quadratic m x = 0 := by sorry

end NUMINAMATH_CALUDE_root_implies_m_value_always_real_roots_l502_50228


namespace NUMINAMATH_CALUDE_select_three_from_fifteen_l502_50242

theorem select_three_from_fifteen (n k : ℕ) : n = 15 ∧ k = 3 → Nat.choose n k = 455 := by
  sorry

end NUMINAMATH_CALUDE_select_three_from_fifteen_l502_50242


namespace NUMINAMATH_CALUDE_first_orphanage_donation_l502_50258

/-- Given a total donation and donations to two orphanages, 
    calculate the donation to the first orphanage -/
def donation_to_first_orphanage (total : ℚ) (second : ℚ) (third : ℚ) : ℚ :=
  total - (second + third)

theorem first_orphanage_donation 
  (total : ℚ) (second : ℚ) (third : ℚ)
  (h_total : total = 650)
  (h_second : second = 225)
  (h_third : third = 250) :
  donation_to_first_orphanage total second third = 175 := by
  sorry

end NUMINAMATH_CALUDE_first_orphanage_donation_l502_50258


namespace NUMINAMATH_CALUDE_triangle_area_zero_l502_50287

def point_a : ℝ × ℝ × ℝ := (2, 3, 1)
def point_b : ℝ × ℝ × ℝ := (8, 6, 4)
def point_c : ℝ × ℝ × ℝ := (14, 9, 7)

def triangle_area (a b c : ℝ × ℝ × ℝ) : ℝ := sorry

theorem triangle_area_zero :
  triangle_area point_a point_b point_c = 0 := by sorry

end NUMINAMATH_CALUDE_triangle_area_zero_l502_50287


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l502_50225

theorem pure_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := Complex.mk (a^2 + 2*a - 3) (a^2 - 4*a + 3)
  (z.re = 0 ∧ z.im ≠ 0) → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l502_50225


namespace NUMINAMATH_CALUDE_nina_shirt_price_l502_50224

/-- Given Nina's shopping scenario, prove the price of each shirt. -/
theorem nina_shirt_price :
  -- Define the number and price of toys
  let num_toys : ℕ := 3
  let price_per_toy : ℕ := 10
  -- Define the number and price of card packs
  let num_card_packs : ℕ := 2
  let price_per_card_pack : ℕ := 5
  -- Define the number of shirts (equal to toys + card packs)
  let num_shirts : ℕ := num_toys + num_card_packs
  -- Define the total amount spent
  let total_spent : ℕ := 70
  -- Calculate the cost of toys and card packs
  let cost_toys_and_cards : ℕ := num_toys * price_per_toy + num_card_packs * price_per_card_pack
  -- Calculate the remaining amount spent on shirts
  let amount_spent_on_shirts : ℕ := total_spent - cost_toys_and_cards
  -- Calculate the price per shirt
  let price_per_shirt : ℕ := amount_spent_on_shirts / num_shirts
  -- Prove that the price per shirt is 6
  price_per_shirt = 6 := by sorry

end NUMINAMATH_CALUDE_nina_shirt_price_l502_50224


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l502_50297

theorem greatest_divisor_with_remainders : 
  let a := 690
  let b := 875
  let r₁ := 10
  let r₂ := 25
  Int.gcd (a - r₁) (b - r₂) = 170 :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l502_50297


namespace NUMINAMATH_CALUDE_existence_of_special_polygon_l502_50223

-- Define what it means for a polygon to have a center of symmetry
def has_center_of_symmetry (P : Set ℝ × ℝ) : Prop := sorry

-- Define what it means for a set to be a polygon
def is_polygon (P : Set ℝ × ℝ) : Prop := sorry

-- Define what it means for a polygon to be convex
def is_convex (P : Set ℝ × ℝ) : Prop := sorry

-- Define what it means for a polygon to be divided into two parts
def can_be_divided_into (P A B : Set ℝ × ℝ) : Prop := sorry

theorem existence_of_special_polygon : 
  ∃ (P A B : Set ℝ × ℝ), 
    is_polygon P ∧ 
    ¬(has_center_of_symmetry P) ∧
    is_polygon A ∧ 
    is_polygon B ∧
    is_convex A ∧ 
    is_convex B ∧
    can_be_divided_into P A B ∧
    has_center_of_symmetry A ∧
    has_center_of_symmetry B := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_polygon_l502_50223


namespace NUMINAMATH_CALUDE_conic_is_ellipse_l502_50267

-- Define the conic section equation
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y+2)^2) + Real.sqrt ((x-4)^2 + (y-3)^2) = 8

-- Define the focal points
def focal_point1 : ℝ × ℝ := (0, -2)
def focal_point2 : ℝ × ℝ := (4, 3)

-- Theorem stating that the equation describes an ellipse
theorem conic_is_ellipse :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (x y : ℝ), conic_equation x y ↔
    (x - (focal_point1.1 + focal_point2.1) / 2)^2 / a^2 +
    (y - (focal_point1.2 + focal_point2.2) / 2)^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_l502_50267


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l502_50283

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (seq.a 1 + seq.a n) * n / 2

/-- The main theorem -/
theorem arithmetic_sequence_sum_ratio
  (seq : ArithmeticSequence)
  (h : seq.a 2 / seq.a 4 = 7 / 6) :
  S seq 7 / S seq 3 = 2 / 1 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l502_50283


namespace NUMINAMATH_CALUDE_count_six_digit_integers_l502_50233

/-- The number of different positive six-digit integers that can be formed using the digits 1, 1, 3, 3, 3, 8 -/
def sixDigitIntegersCount : ℕ := 60

/-- The multiset of digits used to form the integers -/
def digits : Multiset ℕ := {1, 1, 3, 3, 3, 8}

theorem count_six_digit_integers : 
  (Multiset.card digits = 6) → 
  (Multiset.count 1 digits = 2) → 
  (Multiset.count 3 digits = 3) → 
  (Multiset.count 8 digits = 1) → 
  sixDigitIntegersCount = 60 := by sorry

end NUMINAMATH_CALUDE_count_six_digit_integers_l502_50233


namespace NUMINAMATH_CALUDE_paper_I_maximum_mark_l502_50239

/-- The maximum mark for Paper I -/
def maximum_mark : ℝ := 150

/-- The passing percentage for Paper I -/
def passing_percentage : ℝ := 0.40

/-- The marks secured by the candidate -/
def secured_marks : ℝ := 40

/-- The marks by which the candidate failed -/
def failing_margin : ℝ := 20

/-- Theorem stating that the maximum mark for Paper I is 150 -/
theorem paper_I_maximum_mark :
  (passing_percentage * maximum_mark = secured_marks + failing_margin) ∧
  (maximum_mark = 150) := by
  sorry

end NUMINAMATH_CALUDE_paper_I_maximum_mark_l502_50239


namespace NUMINAMATH_CALUDE_sum_of_15th_set_l502_50203

/-- Defines the first element of the nth set in the sequence -/
def first_element (n : ℕ) : ℕ := 
  1 + (n - 1) * n / 2

/-- Defines the last element of the nth set in the sequence -/
def last_element (n : ℕ) : ℕ := 
  first_element n + n - 1

/-- Defines the sum of elements in the nth set -/
def S (n : ℕ) : ℕ := 
  n * (first_element n + last_element n) / 2

theorem sum_of_15th_set : S 15 = 1695 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_15th_set_l502_50203


namespace NUMINAMATH_CALUDE_min_distance_to_line_l502_50236

/-- The minimum value of (x-2)^2 + (y-2)^2 given that x - y - 1 = 0 -/
theorem min_distance_to_line : 
  (∃ (m : ℝ), ∀ (x y : ℝ), x - y - 1 = 0 → (x - 2)^2 + (y - 2)^2 ≥ m) ∧ 
  (∃ (x y : ℝ), x - y - 1 = 0 ∧ (x - 2)^2 + (y - 2)^2 = 1/2) :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l502_50236


namespace NUMINAMATH_CALUDE_bags_difference_l502_50291

def bags_on_monday : ℕ := 7
def bags_on_next_day : ℕ := 12

theorem bags_difference : bags_on_next_day - bags_on_monday = 5 := by
  sorry

end NUMINAMATH_CALUDE_bags_difference_l502_50291


namespace NUMINAMATH_CALUDE_clothing_percentage_proof_l502_50263

theorem clothing_percentage_proof (food_percent : ℝ) (other_percent : ℝ) 
  (clothing_tax_rate : ℝ) (food_tax_rate : ℝ) (other_tax_rate : ℝ) 
  (total_tax_rate : ℝ) :
  food_percent = 20 →
  other_percent = 30 →
  clothing_tax_rate = 4 →
  food_tax_rate = 0 →
  other_tax_rate = 8 →
  total_tax_rate = 4.4 →
  (100 - food_percent - other_percent) * clothing_tax_rate / 100 + 
    food_percent * food_tax_rate / 100 + 
    other_percent * other_tax_rate / 100 = total_tax_rate →
  100 - food_percent - other_percent = 50 := by
sorry

end NUMINAMATH_CALUDE_clothing_percentage_proof_l502_50263


namespace NUMINAMATH_CALUDE_jeffrey_steps_l502_50244

-- Define Jeffrey's walking pattern
def forward_steps : ℕ := 3
def backward_steps : ℕ := 2

-- Define the distance between house and mailbox
def distance : ℕ := 66

-- Define the function to calculate total steps
def total_steps (fwd : ℕ) (bwd : ℕ) (dist : ℕ) : ℕ :=
  dist * (fwd + bwd)

-- Theorem statement
theorem jeffrey_steps :
  total_steps forward_steps backward_steps distance = 330 := by
  sorry

end NUMINAMATH_CALUDE_jeffrey_steps_l502_50244


namespace NUMINAMATH_CALUDE_three_tangent_lines_l502_50202

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

-- Define a line passing through (1, 0)
def line_through_P (m : ℝ) (x y : ℝ) : Prop := y = m * (x - 1)

-- Define the condition for a line to intersect the hyperbola at only one point
def single_intersection (m : ℝ) : Prop :=
  ∃! (x y : ℝ), hyperbola x y ∧ line_through_P m x y

-- Main theorem
theorem three_tangent_lines :
  ∃ (m₁ m₂ m₃ : ℝ), 
    single_intersection m₁ ∧ 
    single_intersection m₂ ∧ 
    single_intersection m₃ ∧
    m₁ ≠ m₂ ∧ m₁ ≠ m₃ ∧ m₂ ≠ m₃ ∧
    ∀ (m : ℝ), single_intersection m → m = m₁ ∨ m = m₂ ∨ m = m₃ :=
sorry

end NUMINAMATH_CALUDE_three_tangent_lines_l502_50202


namespace NUMINAMATH_CALUDE_sqrt_calculation_l502_50211

theorem sqrt_calculation : 
  Real.sqrt 48 / Real.sqrt 3 + Real.sqrt (1/2) * Real.sqrt 12 - Real.sqrt 24 = 4 - Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculation_l502_50211


namespace NUMINAMATH_CALUDE_problem_statement_l502_50273

theorem problem_statement :
  (∃ (a b : ℝ), abs (a + b) < 1 ∧ abs a + abs b ≥ 1) ∧
  (∀ x : ℝ, (x ≤ -3 ∨ x ≥ 1) ↔ |x + 1| - 2 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l502_50273


namespace NUMINAMATH_CALUDE_snowball_theorem_l502_50235

/-- Represents the action of throwing a snowball -/
def throws (n : Nat) (m : Nat) : Prop := sorry

/-- The number of children -/
def num_children : Nat := 43

theorem snowball_theorem :
  (∀ i : Nat, i > 0 ∧ i ≤ num_children → ∃! j : Nat, j > 0 ∧ j ≤ num_children ∧ throws i j) ∧
  (∀ j : Nat, j > 0 ∧ j ≤ num_children → ∃! i : Nat, i > 0 ∧ i ≤ num_children ∧ throws i j) ∧
  (∃ x : Nat, x > 0 ∧ x ≤ num_children ∧ throws 1 x ∧ throws x 2) ∧
  (∃ y : Nat, y > 0 ∧ y ≤ num_children ∧ throws 2 y ∧ throws y 3) ∧
  (∃ z : Nat, z > 0 ∧ z ≤ num_children ∧ throws num_children z ∧ throws z 1) →
  ∃ w : Nat, w = 24 ∧ throws w 3 := by sorry

end NUMINAMATH_CALUDE_snowball_theorem_l502_50235


namespace NUMINAMATH_CALUDE_tan_sum_product_equals_one_l502_50222

theorem tan_sum_product_equals_one :
  let tan15 : ℝ := 2 - Real.sqrt 3
  let tan30 : ℝ := Real.sqrt 3 / 3
  tan15 + tan30 + tan15 * tan30 = 1 := by
sorry

end NUMINAMATH_CALUDE_tan_sum_product_equals_one_l502_50222


namespace NUMINAMATH_CALUDE_kimberly_skittles_l502_50259

/-- The number of Skittles Kimberly initially had -/
def initial_skittles : ℕ := 5

/-- The number of Skittles Kimberly bought -/
def bought_skittles : ℕ := 7

/-- The total number of Skittles Kimberly has after buying more -/
def total_skittles : ℕ := 12

/-- Theorem stating that the initial number of Skittles plus the bought Skittles equals the total Skittles -/
theorem kimberly_skittles : initial_skittles + bought_skittles = total_skittles := by
  sorry

end NUMINAMATH_CALUDE_kimberly_skittles_l502_50259


namespace NUMINAMATH_CALUDE_rectangle_locus_l502_50217

/-- Given a rectangle with length l and width w, and a fixed number b,
    this theorem states that the locus of all points P(x, y) in the plane of the rectangle
    such that the sum of the squares of the distances from P to the four vertices
    of the rectangle equals b is a circle if and only if b > l^2 + w^2. -/
theorem rectangle_locus (l w b : ℝ) :
  (∃ (c : ℝ × ℝ) (r : ℝ),
    ∀ (x y : ℝ),
      (x - 0)^2 + (y - 0)^2 + (x - l)^2 + (y - 0)^2 +
      (x - l)^2 + (y - w)^2 + (x - 0)^2 + (y - w)^2 = b ↔
      (x - c.1)^2 + (y - c.2)^2 = r^2) ↔
  b > l^2 + w^2 := by sorry

end NUMINAMATH_CALUDE_rectangle_locus_l502_50217


namespace NUMINAMATH_CALUDE_age_difference_l502_50295

theorem age_difference (sachin_age rahul_age : ℕ) : 
  sachin_age = 63 →
  sachin_age * 9 = rahul_age * 7 →
  rahul_age - sachin_age = 18 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l502_50295


namespace NUMINAMATH_CALUDE_magnitude_of_complex_number_l502_50290

theorem magnitude_of_complex_number (i : ℂ) : i^2 = -1 → Complex.abs ((1 + i) - 2 / i) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_number_l502_50290


namespace NUMINAMATH_CALUDE_stating_min_nickels_needed_l502_50288

/-- Represents the cost of the book in cents -/
def book_cost : ℕ := 4750

/-- Represents the value of four $10 bills in cents -/
def ten_dollar_bills : ℕ := 4000

/-- Represents the value of five half-dollars in cents -/
def half_dollars : ℕ := 250

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- 
Theorem stating that the minimum number of nickels needed to reach 
or exceed the book cost, given the other money available, is 100.
-/
theorem min_nickels_needed : 
  ∀ n : ℕ, (n * nickel_value + ten_dollar_bills + half_dollars ≥ book_cost) → n ≥ 100 := by
  sorry

end NUMINAMATH_CALUDE_stating_min_nickels_needed_l502_50288


namespace NUMINAMATH_CALUDE_min_value_expression_l502_50260

theorem min_value_expression (x y : ℤ) (h : 4*x + 5*y = 7) :
  ∃ (m : ℤ), m = 1 ∧ ∀ (a b : ℤ), 4*a + 5*b = 7 → 5*|a| - 3*|b| ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l502_50260


namespace NUMINAMATH_CALUDE_A_equals_6x_squared_A_plus_2B_A_plus_2B_at_negative_one_l502_50286

-- Define polynomials A and B
def B (x : ℝ) : ℝ := 4 * x^2 - 5 * x - 7

axiom A_minus_2B (x : ℝ) : ∃ A : ℝ → ℝ, A x - 2 * (B x) = -2 * x^2 + 10 * x + 14

-- Theorem 1: A = 6x^2
theorem A_equals_6x_squared : ∃ A : ℝ → ℝ, ∀ x : ℝ, A x = 6 * x^2 := by sorry

-- Theorem 2: A + 2B = 14x^2 - 10x - 14
theorem A_plus_2B (x : ℝ) : ∃ A : ℝ → ℝ, A x + 2 * (B x) = 14 * x^2 - 10 * x - 14 := by sorry

-- Theorem 3: When x = -1, A + 2B = 10
theorem A_plus_2B_at_negative_one : ∃ A : ℝ → ℝ, A (-1) + 2 * (B (-1)) = 10 := by sorry

end NUMINAMATH_CALUDE_A_equals_6x_squared_A_plus_2B_A_plus_2B_at_negative_one_l502_50286


namespace NUMINAMATH_CALUDE_no_preimage_iff_k_less_than_neg_two_l502_50266

/-- The function f: ℝ → ℝ defined by f(x) = x² - 2x - 1 -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 1

/-- Theorem stating that there is no real solution to f(x) = k if and only if k < -2 -/
theorem no_preimage_iff_k_less_than_neg_two :
  ∀ k : ℝ, (¬∃ x : ℝ, f x = k) ↔ k < -2 := by sorry

end NUMINAMATH_CALUDE_no_preimage_iff_k_less_than_neg_two_l502_50266


namespace NUMINAMATH_CALUDE_valid_paths_count_l502_50254

/-- The number of paths on a grid from (0,0) to (m,n) -/
def gridPaths (m n : ℕ) : ℕ := Nat.choose (m + n) n

/-- The number of paths from A to C on the 6x3 grid -/
def pathsAtoC : ℕ := gridPaths 4 1

/-- The number of paths from D to B on the 6x3 grid -/
def pathsDtoB : ℕ := gridPaths 2 2

/-- The number of paths from A to E on the 6x3 grid -/
def pathsAtoE : ℕ := gridPaths 2 2

/-- The number of paths from F to B on the 6x3 grid -/
def pathsFtoB : ℕ := gridPaths 4 0

/-- The total number of paths on the 6x3 grid -/
def totalPaths : ℕ := gridPaths 6 3

/-- The number of invalid paths through the first forbidden segment -/
def invalidPaths1 : ℕ := pathsAtoC * pathsDtoB

/-- The number of invalid paths through the second forbidden segment -/
def invalidPaths2 : ℕ := pathsAtoE * pathsFtoB

theorem valid_paths_count :
  totalPaths - (invalidPaths1 + invalidPaths2) = 48 := by sorry

end NUMINAMATH_CALUDE_valid_paths_count_l502_50254


namespace NUMINAMATH_CALUDE_cylinder_lateral_area_l502_50270

/-- Given a cylinder with a square cross-section of area 4, its lateral area is 4π. -/
theorem cylinder_lateral_area (r h : ℝ) : 
  r * r = 4 → 2 * π * r * h = 4 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_area_l502_50270


namespace NUMINAMATH_CALUDE_circle_area_through_triangle_points_l502_50238

/-- Given a right triangle PQR with legs PQ = 6 and PR = 8, the area of the circle 
    passing through points Q, R, and the midpoint M of hypotenuse QR is 25π. -/
theorem circle_area_through_triangle_points (P Q R M : ℝ × ℝ) : 
  -- Triangle PQR is a right triangle
  (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0 →
  -- PQ = 6
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 36 →
  -- PR = 8
  (R.1 - P.1)^2 + (R.2 - P.2)^2 = 64 →
  -- M is the midpoint of QR
  M = ((Q.1 + R.1) / 2, (Q.2 + R.2) / 2) →
  -- The area of the circle passing through Q, R, and M
  π * ((Q.1 - M.1)^2 + (Q.2 - M.2)^2) = 25 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_through_triangle_points_l502_50238


namespace NUMINAMATH_CALUDE_new_tax_rate_is_30_percent_l502_50221

/-- Calculates the new tax rate given the initial rate, income, and tax savings -/
def calculate_new_tax_rate (initial_rate : ℚ) (income : ℚ) (savings : ℚ) : ℚ :=
  let initial_tax := initial_rate * income
  let new_tax := initial_tax - savings
  new_tax / income

theorem new_tax_rate_is_30_percent :
  let initial_rate : ℚ := 45 / 100
  let income : ℚ := 48000
  let savings : ℚ := 7200
  calculate_new_tax_rate initial_rate income savings = 30 / 100 := by
sorry

end NUMINAMATH_CALUDE_new_tax_rate_is_30_percent_l502_50221


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l502_50252

/-- Given a point P(2, -5), its symmetric point P' with respect to the x-axis has coordinates (2, 5) -/
theorem symmetric_point_wrt_x_axis :
  let P : ℝ × ℝ := (2, -5)
  let P' : ℝ × ℝ := (2, 5)
  (∀ (x y : ℝ), (x, y) = P → (x, -y) = P') :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l502_50252


namespace NUMINAMATH_CALUDE_min_value_of_expression_l502_50294

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_parallel : (1 : ℝ) / 2 = (x - 2) / (-6 * y)) :
  (3 / x + 1 / y) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l502_50294


namespace NUMINAMATH_CALUDE_total_squares_on_grid_l502_50271

/-- Represents a point on the 5x5 grid -/
structure GridPoint where
  x : Fin 5
  y : Fin 5

/-- Represents the set of 20 nails on the grid -/
def NailSet : Set GridPoint :=
  sorry

/-- Determines if four points form a square -/
def isSquare (p1 p2 p3 p4 : GridPoint) : Prop :=
  sorry

/-- Counts the number of squares that can be formed using the nails -/
def countSquares (nails : Set GridPoint) : Nat :=
  sorry

theorem total_squares_on_grid :
  countSquares NailSet = 21 :=
sorry

end NUMINAMATH_CALUDE_total_squares_on_grid_l502_50271
