import Mathlib

namespace NUMINAMATH_GPT_percentage_of_knives_l1951_195176

def initial_knives : Nat := 6
def initial_forks : Nat := 12
def initial_spoons : Nat := 3 * initial_knives
def traded_knives : Nat := 10
def traded_spoons : Nat := 6

theorem percentage_of_knives :
  100 * (initial_knives + traded_knives) / (initial_knives + initial_forks + initial_spoons - traded_spoons + traded_knives) = 40 := by
  sorry

end NUMINAMATH_GPT_percentage_of_knives_l1951_195176


namespace NUMINAMATH_GPT_friends_count_is_four_l1951_195107

def number_of_friends (Melanie Benny Sally Jessica : ℕ) (total_cards : ℕ) : ℕ :=
  4

theorem friends_count_is_four (Melanie Benny Sally Jessica : ℕ) (total_cards : ℕ) (h1 : total_cards = 12) :
  number_of_friends Melanie Benny Sally Jessica total_cards = 4 :=
by
  sorry

end NUMINAMATH_GPT_friends_count_is_four_l1951_195107


namespace NUMINAMATH_GPT_original_bill_amount_l1951_195139

/-- 
If 8 people decided to split the restaurant bill evenly and each paid $314.15 after rounding
up to the nearest cent, then the original bill amount was $2513.20.
-/
theorem original_bill_amount (n : ℕ) (individual_share : ℝ) (total_amount : ℝ) 
  (h1 : n = 8) (h2 : individual_share = 314.15) 
  (h3 : total_amount = n * individual_share) : 
  total_amount = 2513.20 :=
by
  sorry

end NUMINAMATH_GPT_original_bill_amount_l1951_195139


namespace NUMINAMATH_GPT_total_cost_is_correct_l1951_195116

-- Definitions based on conditions
def bedroomDoorCount : ℕ := 3
def outsideDoorCount : ℕ := 2
def outsideDoorCost : ℕ := 20
def bedroomDoorCost : ℕ := outsideDoorCost / 2

-- Total costs calculations
def totalBedroomCost : ℕ := bedroomDoorCount * bedroomDoorCost
def totalOutsideCost : ℕ := outsideDoorCount * outsideDoorCost
def totalCost : ℕ := totalBedroomCost + totalOutsideCost

-- Proof statement
theorem total_cost_is_correct : totalCost = 70 := 
by
  sorry

end NUMINAMATH_GPT_total_cost_is_correct_l1951_195116


namespace NUMINAMATH_GPT_ratio_grass_area_weeded_l1951_195167

/-- Lucille earns six cents for every weed she pulls. -/
def earnings_per_weed : ℕ := 6

/-- There are eleven weeds in the flower bed. -/
def weeds_flower_bed : ℕ := 11

/-- There are fourteen weeds in the vegetable patch. -/
def weeds_vegetable_patch : ℕ := 14

/-- There are thirty-two weeds in the grass around the fruit trees. -/
def weeds_grass_total : ℕ := 32

/-- Lucille bought a soda for 99 cents on her break. -/
def soda_cost : ℕ := 99

/-- Lucille has 147 cents left after the break. -/
def cents_left : ℕ := 147

/-- Statement to prove: The ratio of the grass area Lucille weeded to the total grass area around the fruit trees is 1:2. -/
theorem ratio_grass_area_weeded :
  (earnings_per_weed * (weeds_flower_bed + weeds_vegetable_patch) + earnings_per_weed * (weeds_flower_bed + (weeds_grass_total - (earnings_per_weed + soda_cost)) / earnings_per_weed) = soda_cost + cents_left)
→ ((earnings_per_weed  * (32 - (147 + 99) / earnings_per_weed)) / weeds_grass_total) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_grass_area_weeded_l1951_195167


namespace NUMINAMATH_GPT_hyungjun_initial_paint_count_l1951_195149

theorem hyungjun_initial_paint_count (X : ℝ) (h1 : X / 2 - (X / 6 + 5) = 5) : X = 30 :=
sorry

end NUMINAMATH_GPT_hyungjun_initial_paint_count_l1951_195149


namespace NUMINAMATH_GPT_trey_nail_usage_l1951_195197

theorem trey_nail_usage (total_decorations nails thumbtacks sticky_strips : ℕ) 
  (h1 : nails = 2 * total_decorations / 3)
  (h2 : sticky_strips = 15)
  (h3 : sticky_strips = 3 * (total_decorations - 2 * total_decorations / 3) / 5) :
  nails = 50 :=
by
  sorry

end NUMINAMATH_GPT_trey_nail_usage_l1951_195197


namespace NUMINAMATH_GPT_x_days_worked_l1951_195134

theorem x_days_worked (W : ℝ) :
  let x_work_rate := W / 20
  let y_work_rate := W / 24
  let y_days := 12
  let y_work_done := y_work_rate * y_days
  let total_work := W
  let work_done_by_x := (W - y_work_done) / x_work_rate
  work_done_by_x = 10 := 
by
  sorry

end NUMINAMATH_GPT_x_days_worked_l1951_195134


namespace NUMINAMATH_GPT_smallest_prime_12_less_than_square_l1951_195113

def is_perfect_square (n : ℕ) := ∃ k : ℕ, k * k = n

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_prime_12_less_than_square : ∃ n : ℕ, is_prime n ∧ (∃ k : ℕ, k * k = n + 12) ∧ n = 13 :=
by
  sorry

end NUMINAMATH_GPT_smallest_prime_12_less_than_square_l1951_195113


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1951_195117

theorem geometric_sequence_sum (n : ℕ) (S : ℕ → ℚ) (a : ℚ) :
  (∀ n, S n = (1 / 2) * 3^(n + 1) - a) →
  S 1 - (S 2 - S 1)^2 = (S 2 - S 1) * (S 3 - S 2) →
  a = 3 / 2 :=
by
  intros hSn hgeo
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1951_195117


namespace NUMINAMATH_GPT_who_is_werewolf_choose_companion_l1951_195192

-- Define inhabitants with their respective statements
inductive Inhabitant
| A | B | C

-- Assume each inhabitant can be either a knight (truth-teller) or a liar
def is_knight (i : Inhabitant) : Prop := sorry

-- Define statements made by each inhabitant
def A_statement : Prop := ∃ werewolf : Inhabitant, werewolf = Inhabitant.C
def B_statement : Prop := ¬(∃ werewolf : Inhabitant, werewolf = Inhabitant.B)
def C_statement : Prop := ∃ liar1 liar2 : Inhabitant, liar1 ≠ liar2 ∧ liar1 ≠ Inhabitant.C ∧ liar2 ≠ Inhabitant.C

-- Define who is the werewolf (liar)
def is_werewolf (i : Inhabitant) : Prop := ¬is_knight i

-- The given conditions from statements
axiom A_is_knight : is_knight Inhabitant.A ↔ A_statement
axiom B_is_knight : is_knight Inhabitant.B ↔ B_statement
axiom C_is_knight : is_knight Inhabitant.C ↔ C_statement

-- The conclusion: C is the werewolf and thus a liar.
theorem who_is_werewolf : is_werewolf Inhabitant.C :=
by sorry

-- Choosing a companion: 
-- If C is a werewolf, we prefer to pick A as a companion over B or C.
theorem choose_companion (worry_about_werewolf : Bool) : Inhabitant :=
if worry_about_werewolf then Inhabitant.A else sorry

end NUMINAMATH_GPT_who_is_werewolf_choose_companion_l1951_195192


namespace NUMINAMATH_GPT_total_cost_eq_16000_l1951_195133

theorem total_cost_eq_16000 (F M T : ℕ) (n : ℕ) (hF : F = 12000) (hM : M = 200) (hT : T = 16000) :
  T = F + M * n → n = 20 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_eq_16000_l1951_195133


namespace NUMINAMATH_GPT_minimum_additional_squares_needed_to_achieve_symmetry_l1951_195127

def initial_grid : List (ℕ × ℕ) := [(1, 4), (4, 1)] -- Initial shaded squares

def is_symmetric (grid : List (ℕ × ℕ)) : Prop :=
  ∀ (x y : ℕ × ℕ), x ∈ grid → y ∈ grid →
    ((x.1 = 2 * 2 - y.1 ∧ x.2 = y.2) ∨
     (x.1 = y.1 ∧ x.2 = 5 - y.2) ∨
     (x.1 = 2 * 2 - y.1 ∧ x.2 = 5 - y.2))

def additional_squares_needed : ℕ :=
  6 -- As derived in the solution steps, 6 additional squares are needed to achieve symmetry

theorem minimum_additional_squares_needed_to_achieve_symmetry :
  ∀ (initial_shades : List (ℕ × ℕ)),
    initial_shades = initial_grid →
    ∃ (additional : List (ℕ × ℕ)),
      initial_shades ++ additional = symmetric_grid ∧
      additional.length = additional_squares_needed :=
by 
-- skip the proof
sorry

end NUMINAMATH_GPT_minimum_additional_squares_needed_to_achieve_symmetry_l1951_195127


namespace NUMINAMATH_GPT_expand_product_l1951_195174

theorem expand_product (x : ℝ) : 2 * (x + 3) * (x + 6) = 2 * x^2 + 18 * x + 36 :=
by
  sorry

end NUMINAMATH_GPT_expand_product_l1951_195174


namespace NUMINAMATH_GPT_find_value_of_expression_l1951_195168

variable {a b c d x : ℝ}

-- Conditions
def opposites (a b : ℝ) : Prop := a + b = 0
def reciprocals (c d : ℝ) : Prop := c * d = 1
def abs_three (x : ℝ) : Prop := |x| = 3

-- Proof
theorem find_value_of_expression (h1 : opposites a b) (h2 : reciprocals c d) 
  (h3 : abs_three x) : ∃ res : ℝ, (res = 3 ∨ res = -3) ∧ res = 10 * a + 10 * b + c * d * x :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_expression_l1951_195168


namespace NUMINAMATH_GPT_cost_of_two_pans_is_20_l1951_195189

variable (cost_of_pan : ℕ)

-- Conditions
def pots_cost := 3 * 20
def total_cost := 100
def pans_eq_cost := total_cost - pots_cost
def cost_of_pan_per_pans := pans_eq_cost / 4

-- Proof statement
theorem cost_of_two_pans_is_20 
  (h1 : pots_cost = 60)
  (h2 : total_cost = 100)
  (h3 : pans_eq_cost = total_cost - pots_cost)
  (h4 : cost_of_pan_per_pans = pans_eq_cost / 4)
  : 2 * cost_of_pan_per_pans = 20 :=
by sorry

end NUMINAMATH_GPT_cost_of_two_pans_is_20_l1951_195189


namespace NUMINAMATH_GPT_can_form_triangle_8_6_4_l1951_195166

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem can_form_triangle_8_6_4 : can_form_triangle 8 6 4 :=
by
  unfold can_form_triangle
  simp
  exact ⟨by linarith, by linarith, by linarith⟩

end NUMINAMATH_GPT_can_form_triangle_8_6_4_l1951_195166


namespace NUMINAMATH_GPT_brit_age_after_vacation_l1951_195163

-- Define the given conditions and the final proof question

-- Rebecca's age is 25 years
def rebecca_age : ℕ := 25

-- Brittany is older than Rebecca by 3 years
def brit_age_before_vacation (rebecca_age : ℕ) : ℕ := rebecca_age + 3

-- Brittany goes on a 4-year vacation
def vacation_duration : ℕ := 4

-- Prove that Brittany’s age when she returns from her vacation is 32
theorem brit_age_after_vacation (rebecca_age vacation_duration : ℕ) : brit_age_before_vacation rebecca_age + vacation_duration = 32 :=
by
  sorry

end NUMINAMATH_GPT_brit_age_after_vacation_l1951_195163


namespace NUMINAMATH_GPT_black_cars_in_parking_lot_l1951_195144

theorem black_cars_in_parking_lot :
  let total_cars := 3000
  let blue_percent := 0.40
  let red_percent := 0.25
  let green_percent := 0.15
  let yellow_percent := 0.10
  let black_percent := 1 - (blue_percent + red_percent + green_percent + yellow_percent)
  let number_of_black_cars := total_cars * black_percent
  number_of_black_cars = 300 :=
by
  sorry

end NUMINAMATH_GPT_black_cars_in_parking_lot_l1951_195144


namespace NUMINAMATH_GPT_base8_subtraction_l1951_195131

theorem base8_subtraction : (52 - 27 : ℕ) = 23 := by sorry

end NUMINAMATH_GPT_base8_subtraction_l1951_195131


namespace NUMINAMATH_GPT_total_rides_correct_l1951_195145

-- Definitions based on the conditions:
def billy_rides : ℕ := 17
def john_rides : ℕ := 2 * billy_rides
def mother_rides : ℕ := john_rides + 10
def total_rides : ℕ := billy_rides + john_rides + mother_rides

-- The theorem to prove their total bike rides.
theorem total_rides_correct : total_rides = 95 := by
  sorry

end NUMINAMATH_GPT_total_rides_correct_l1951_195145


namespace NUMINAMATH_GPT_sum_of_roots_l1951_195181

open Real

theorem sum_of_roots (r s : ℝ) (P : ℝ → ℝ) (Q : ℝ × ℝ) (m : ℝ) :
  (∀ (x : ℝ), P x = x^2) → 
  Q = (20, 14) → 
  (∀ m : ℝ, (m^2 - 80 * m + 56 < 0) ↔ (r < m ∧ m < s)) →
  r + s = 80 :=
by {
  -- sketched proof goes here
  sorry
}

end NUMINAMATH_GPT_sum_of_roots_l1951_195181


namespace NUMINAMATH_GPT_incorrect_statement_C_l1951_195196

-- Lean 4 statement to verify correctness of problem translation
theorem incorrect_statement_C (n : ℕ) (w : ℕ → ℕ) :
  (w 1 = 55) ∧
  (w 2 = 110) ∧
  (w 3 = 160) ∧
  (w 4 = 200) ∧
  (w 5 = 254) ∧
  (w 6 = 300) ∧
  (w 7 = 350) →
  ¬(∀ n, w n = 55 * n) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_incorrect_statement_C_l1951_195196


namespace NUMINAMATH_GPT_values_of_k_real_equal_roots_l1951_195114

theorem values_of_k_real_equal_roots (k : ℝ) :
  (∀ x : ℝ, 3 * x^2 - (k + 2) * x + 12 = 0 → x * x = 0) ↔ (k = 10 ∨ k = -14) :=
by
  sorry

end NUMINAMATH_GPT_values_of_k_real_equal_roots_l1951_195114


namespace NUMINAMATH_GPT_center_of_circle_l1951_195159

theorem center_of_circle (x y : ℝ) : 
  (x^2 + y^2 = 6 * x - 10 * y + 9) → 
  (∃ c : ℝ × ℝ, c = (3, -5) ∧ c.1 + c.2 = -2) :=
by
  sorry

end NUMINAMATH_GPT_center_of_circle_l1951_195159


namespace NUMINAMATH_GPT_kaleb_lives_left_l1951_195111

theorem kaleb_lives_left (initial_lives : ℕ) (lives_lost : ℕ) (remaining_lives : ℕ) :
  initial_lives = 98 → lives_lost = 25 → remaining_lives = initial_lives - lives_lost → remaining_lives = 73 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_kaleb_lives_left_l1951_195111


namespace NUMINAMATH_GPT_boat_width_l1951_195110

-- Definitions: river width, number of boats, and space between/banks
def river_width : ℝ := 42
def num_boats : ℕ := 8
def space_between : ℝ := 2

-- Prove the width of each boat given the conditions
theorem boat_width : 
  ∃ w : ℝ, 
    8 * w + 7 * space_between + 2 * space_between = river_width ∧
    w = 3 :=
by
  sorry

end NUMINAMATH_GPT_boat_width_l1951_195110


namespace NUMINAMATH_GPT_phil_baseball_cards_left_l1951_195147

-- Step a): Define the conditions
def packs_week := 20
def weeks_year := 52
def lost_factor := 1 / 2

-- Step c): Establish the theorem statement
theorem phil_baseball_cards_left : 
  (packs_week * weeks_year * (1 - lost_factor) = 520) := 
  by
    -- proof steps will come here
    sorry

end NUMINAMATH_GPT_phil_baseball_cards_left_l1951_195147


namespace NUMINAMATH_GPT_license_plate_palindrome_probability_find_m_plus_n_l1951_195164

noncomputable section

open Nat

def is_palindrome {α : Type} (seq : List α) : Prop :=
  seq = seq.reverse

def number_of_three_digit_palindromes : ℕ :=
  10 * 10  -- explanation: 10 choices for the first and last digits, 10 for the middle digit

def total_three_digit_numbers : ℕ :=
  10^3  -- 1000

def prob_three_digit_palindrome : ℚ :=
  number_of_three_digit_palindromes / total_three_digit_numbers

def number_of_three_letter_palindromes : ℕ :=
  26 * 26  -- 26 choices for the first and last letters, 26 for the middle letter

def total_three_letter_combinations : ℕ :=
  26^3  -- 26^3

def prob_three_letter_palindrome : ℚ :=
  number_of_three_letter_palindromes / total_three_letter_combinations

def prob_either_palindrome : ℚ :=
  prob_three_digit_palindrome + prob_three_letter_palindrome - (prob_three_digit_palindrome * prob_three_letter_palindrome)

def m : ℕ := 7
def n : ℕ := 52

theorem license_plate_palindrome_probability :
  prob_either_palindrome = 7 / 52 := sorry

theorem find_m_plus_n :
  m + n = 59 := rfl

end NUMINAMATH_GPT_license_plate_palindrome_probability_find_m_plus_n_l1951_195164


namespace NUMINAMATH_GPT_area_diminished_by_64_percent_l1951_195169

/-- Given a rectangular field where both the length and width are diminished by 40%, 
    prove that the area is diminished by 64%. -/
theorem area_diminished_by_64_percent (L W : ℝ) :
  let L' := 0.6 * L
  let W' := 0.6 * W
  let A := L * W
  let A' := L' * W'
  (A - A') / A * 100 = 64 :=
by
  sorry

end NUMINAMATH_GPT_area_diminished_by_64_percent_l1951_195169


namespace NUMINAMATH_GPT_max_composite_rel_prime_set_l1951_195128

theorem max_composite_rel_prime_set : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, 10 ≤ n ∧ n ≤ 99 ∧ ¬Nat.Prime n) ∧ 
  (∀ (a b : ℕ), a ∈ S → b ∈ S → a ≠ b → Nat.gcd a b = 1) ∧ 
  S.card = 4 := by
sorry

end NUMINAMATH_GPT_max_composite_rel_prime_set_l1951_195128


namespace NUMINAMATH_GPT_average_age_group_l1951_195103

theorem average_age_group (n : ℕ) (T : ℕ) (h1 : T = n * 14) (h2 : T + 32 = (n + 1) * 15) : n = 17 :=
by
  sorry

end NUMINAMATH_GPT_average_age_group_l1951_195103


namespace NUMINAMATH_GPT_sum_a_16_to_20_l1951_195188

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
axiom S_def : ∀ n, S n = a 0 * (1 - (a 1 / a 0) ^ n) / (1 - (a 1 / a 0))
axiom S_5_eq_2 : S 5 = 2
axiom S_10_eq_6 : S 10 = 6

-- Theorem to prove
theorem sum_a_16_to_20 : a 16 + a 17 + a 18 + a 19 + a 20 = 16 :=
by
  sorry

end NUMINAMATH_GPT_sum_a_16_to_20_l1951_195188


namespace NUMINAMATH_GPT_relationship_correct_l1951_195154

noncomputable def log_base (a b : ℝ) : ℝ := Real.log b / Real.log a

theorem relationship_correct (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) :
  log_base a b < a^b ∧ a^b < b^a :=
by sorry

end NUMINAMATH_GPT_relationship_correct_l1951_195154


namespace NUMINAMATH_GPT_max_profit_300_l1951_195140

noncomputable def total_cost (x : ℝ) : ℝ := 20000 + 100 * x

noncomputable def total_revenue (x : ℝ) : ℝ :=
if x ≤ 400 then (400 * x - (1 / 2) * x^2)
else 80000

noncomputable def total_profit (x : ℝ) : ℝ :=
total_revenue x - total_cost x

theorem max_profit_300 :
    ∃ x : ℝ, (total_profit x = (total_revenue 300 - total_cost 300)) := sorry

end NUMINAMATH_GPT_max_profit_300_l1951_195140


namespace NUMINAMATH_GPT_largest_digit_A_l1951_195146

theorem largest_digit_A (A : ℕ) (h1 : (31 + A) % 3 = 0) (h2 : 96 % 4 = 0) : 
  A ≤ 7 ∧ (∀ a, a > 7 → ¬((31 + a) % 3 = 0 ∧ 96 % 4 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_largest_digit_A_l1951_195146


namespace NUMINAMATH_GPT_periodic_function_l1951_195104

open Real

theorem periodic_function (f : ℝ → ℝ) 
  (h_bound : ∀ x : ℝ, |f x| ≤ 1)
  (h_func_eq : ∀ x : ℝ, f (x + 13 / 42) + f x = f (x + 1 / 6) + f (x + 1 / 7)) : 
  ∀ x : ℝ, f (x + 1) = f x := 
  sorry

end NUMINAMATH_GPT_periodic_function_l1951_195104


namespace NUMINAMATH_GPT_length_of_segment_AB_l1951_195165

noncomputable def speed_relation_first (x v1 v2 : ℝ) : Prop :=
  300 / v1 = (x - 300) / v2

noncomputable def speed_relation_second (x v1 v2 : ℝ) : Prop :=
  (x + 100) / v1 = (x - 100) / v2

theorem length_of_segment_AB :
  (∃ (x v1 v2 : ℝ),
    x > 0 ∧
    v1 > 0 ∧
    v2 > 0 ∧
    speed_relation_first x v1 v2 ∧
    speed_relation_second x v1 v2) →
  ∃ x : ℝ, x = 500 :=
by
  sorry

end NUMINAMATH_GPT_length_of_segment_AB_l1951_195165


namespace NUMINAMATH_GPT_perimeter_of_similar_triangle_l1951_195135

theorem perimeter_of_similar_triangle (a b c d : ℕ) (h_iso : (a = 12) ∧ (b = 24) ∧ (c = 24)) (h_sim : d = 30) 
  : (d + 2 * b) = 150 := by
  sorry

end NUMINAMATH_GPT_perimeter_of_similar_triangle_l1951_195135


namespace NUMINAMATH_GPT_range_of_a_plus_b_at_least_one_nonnegative_l1951_195125

-- Conditions
variable (x : ℝ) (a := x^2 - 1) (b := 2 * x + 2)

-- Proof Problem 1: Prove that the range of a + b is [0, +∞)
theorem range_of_a_plus_b : (a + b) ≥ 0 :=
by sorry

-- Proof Problem 2: Prove by contradiction that at least one of a or b is greater than or equal to 0
theorem at_least_one_nonnegative : ¬(a < 0 ∧ b < 0) :=
by sorry

end NUMINAMATH_GPT_range_of_a_plus_b_at_least_one_nonnegative_l1951_195125


namespace NUMINAMATH_GPT_ellipse_standard_form_l1951_195101

theorem ellipse_standard_form (α : ℝ) 
  (x y : ℝ) 
  (hx : x = 5 * Real.cos α) 
  (hy : y = 3 * Real.sin α) : 
  (x^2 / 25) + (y^2 / 9) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_ellipse_standard_form_l1951_195101


namespace NUMINAMATH_GPT_actual_diameter_of_tissue_l1951_195130

theorem actual_diameter_of_tissue (magnification_factor : ℝ) (magnified_diameter : ℝ) (image_magnified : magnification_factor = 1000 ∧ magnified_diameter = 2) : (1 / magnification_factor) * magnified_diameter = 0.002 :=
by
  sorry

end NUMINAMATH_GPT_actual_diameter_of_tissue_l1951_195130


namespace NUMINAMATH_GPT_marie_finishes_fourth_task_at_11_40_am_l1951_195112

-- Define the given conditions
def start_time : ℕ := 7 * 60 -- start time in minutes from midnight (7:00 AM)
def second_task_end_time : ℕ := 9 * 60 + 20 -- end time of second task in minutes from midnight (9:20 AM)
def num_tasks : ℕ := 4 -- four tasks
def task_duration : ℕ := (second_task_end_time - start_time) / 2 -- duration of one task

-- Define the goal to prove: the end time of the fourth task
def fourth_task_finish_time : ℕ := second_task_end_time + 2 * task_duration

theorem marie_finishes_fourth_task_at_11_40_am : fourth_task_finish_time = 11 * 60 + 40 := by
  sorry

end NUMINAMATH_GPT_marie_finishes_fourth_task_at_11_40_am_l1951_195112


namespace NUMINAMATH_GPT_subtract_digits_value_l1951_195132

theorem subtract_digits_value (A B : ℕ) (h1 : A ≠ B) (h2 : 2 * 1000 + A * 100 + 3 * 10 + 2 - (B * 100 + B * 10 + B) = 1 * 1000 + B * 100 + B * 10 + B) :
  B - A = 3 :=
by
  sorry

end NUMINAMATH_GPT_subtract_digits_value_l1951_195132


namespace NUMINAMATH_GPT_prop_converse_inverse_contrapositive_correct_statements_l1951_195180

-- Defining the proposition and its types
def prop (x : ℕ) : Prop := x > 0 → x^2 ≥ 0
def converse (x : ℕ) : Prop := x^2 ≥ 0 → x > 0
def inverse (x : ℕ) : Prop := ¬ (x > 0) → x^2 < 0
def contrapositive (x : ℕ) : Prop := x^2 < 0 → ¬ (x > 0)

-- The proof problem
theorem prop_converse_inverse_contrapositive_correct_statements :
  (∃! (p : Prop), p = (∀ x : ℕ, converse x) ∨ p = (∀ x : ℕ, inverse x) ∨ p = (∀ x : ℕ, contrapositive x) ∧ p = True) :=
sorry

end NUMINAMATH_GPT_prop_converse_inverse_contrapositive_correct_statements_l1951_195180


namespace NUMINAMATH_GPT_batsman_average_increase_l1951_195161

theorem batsman_average_increase (A : ℝ) (X : ℝ) (runs_11th_inning : ℝ) (average_11th_inning : ℝ) 
  (h_runs_11th_inning : runs_11th_inning = 85) 
  (h_average_11th_inning : average_11th_inning = 35) 
  (h_eq : (10 * A + runs_11th_inning) / 11 = average_11th_inning) :
  X = 5 := 
by 
  sorry

end NUMINAMATH_GPT_batsman_average_increase_l1951_195161


namespace NUMINAMATH_GPT_largest_possible_rational_root_l1951_195115

noncomputable def rational_root_problem : Prop :=
  ∃ (a b c : ℕ), (a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ 100 ∧ b ≤ 100 ∧ c ≤ 100) ∧
  ∀ p q : ℤ, (q ≠ 0) → (a * p^2 + b * p + c * q = 0) → 
  (p / q) ≤ -1 / 99

theorem largest_possible_rational_root : rational_root_problem :=
sorry

end NUMINAMATH_GPT_largest_possible_rational_root_l1951_195115


namespace NUMINAMATH_GPT_eval_at_3_l1951_195138

theorem eval_at_3 : (3^3)^(3^3) = 27^27 :=
by sorry

end NUMINAMATH_GPT_eval_at_3_l1951_195138


namespace NUMINAMATH_GPT_list_price_of_article_l1951_195148

theorem list_price_of_article (P : ℝ) (h : 0.882 * P = 57.33) : P = 65 :=
by
  sorry

end NUMINAMATH_GPT_list_price_of_article_l1951_195148


namespace NUMINAMATH_GPT_sum_of_solutions_eq_zero_l1951_195194

theorem sum_of_solutions_eq_zero (x : ℝ) :
  (∃ x_1 x_2 : ℝ, (|x_1 - 20| + |x_2 + 20| = 2020) ∧ (x_1 + x_2 = 0)) :=
sorry

end NUMINAMATH_GPT_sum_of_solutions_eq_zero_l1951_195194


namespace NUMINAMATH_GPT_smallest_AAB_value_l1951_195187

theorem smallest_AAB_value {A B : ℕ} (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) (h_distinct : A ≠ B) (h_eq : 10 * A + B = (1 / 9) * (100 * A + 10 * A + B)) :
  100 * A + 10 * A + B = 225 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_smallest_AAB_value_l1951_195187


namespace NUMINAMATH_GPT_arithmetic_sequence_150th_term_l1951_195100

theorem arithmetic_sequence_150th_term :
  let a₁ := 3
  let d := 4
  a₁ + (150 - 1) * d = 599 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_150th_term_l1951_195100


namespace NUMINAMATH_GPT_ben_chairs_in_10_days_l1951_195172

noncomputable def chairs_built_per_day (hours_per_shift : ℕ) (hours_per_chair : ℕ) : ℕ :=
  hours_per_shift / hours_per_chair

theorem ben_chairs_in_10_days 
  (hours_per_shift : ℕ)
  (hours_per_chair : ℕ)
  (days: ℕ)
  (h_shift: hours_per_shift = 8)
  (h_chair: hours_per_chair = 5)
  (h_days: days = 10) : 
  chairs_built_per_day hours_per_shift hours_per_chair * days = 10 :=
by 
  -- We insert a placeholder 'sorry' to be replaced by an actual proof.
  sorry

end NUMINAMATH_GPT_ben_chairs_in_10_days_l1951_195172


namespace NUMINAMATH_GPT_find_third_number_l1951_195183

theorem find_third_number (x : ℝ) : 3 + 33 + x + 3.33 = 369.63 → x = 330.30 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_third_number_l1951_195183


namespace NUMINAMATH_GPT_difference_of_squares_l1951_195106

theorem difference_of_squares : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end NUMINAMATH_GPT_difference_of_squares_l1951_195106


namespace NUMINAMATH_GPT_student_count_l1951_195129

theorem student_count (ratio : ℝ) (teachers : ℕ) (students : ℕ)
  (h1 : ratio = 27.5)
  (h2 : teachers = 42)
  (h3 : ratio * (teachers : ℝ) = students) :
  students = 1155 :=
sorry

end NUMINAMATH_GPT_student_count_l1951_195129


namespace NUMINAMATH_GPT_height_of_each_step_l1951_195153

-- Define the number of steps in each staircase
def first_staircase_steps : ℕ := 20
def second_staircase_steps : ℕ := 2 * first_staircase_steps
def third_staircase_steps : ℕ := second_staircase_steps - 10

-- Define the total steps climbed
def total_steps_climbed : ℕ := first_staircase_steps + second_staircase_steps + third_staircase_steps

-- Define the total height climbed
def total_height_climbed : ℝ := 45

-- Prove the height of each step
theorem height_of_each_step : (total_height_climbed / total_steps_climbed) = 0.5 := by
  sorry

end NUMINAMATH_GPT_height_of_each_step_l1951_195153


namespace NUMINAMATH_GPT_abs_five_minus_two_e_l1951_195190

noncomputable def e : ℝ := 2.718

theorem abs_five_minus_two_e : |5 - 2 * e| = 0.436 := by
  sorry

end NUMINAMATH_GPT_abs_five_minus_two_e_l1951_195190


namespace NUMINAMATH_GPT_fraction_of_boxes_loaded_by_day_crew_l1951_195177

-- Definitions based on the conditions
variables (D W : ℕ)  -- Day crew per worker boxes (D) and number of workers (W)

-- Helper Definitions
def boxes_day_crew : ℕ := D * W  -- Total boxes by day crew
def boxes_night_crew : ℕ := (3 * D / 4) * (3 * W / 4)  -- Total boxes by night crew
def total_boxes : ℕ := boxes_day_crew D W + boxes_night_crew D W  -- Total boxes by both crews

-- The main theorem
theorem fraction_of_boxes_loaded_by_day_crew :
  (boxes_day_crew D W : ℚ) / (total_boxes D W : ℚ) = 16/25 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_boxes_loaded_by_day_crew_l1951_195177


namespace NUMINAMATH_GPT_binomial_prime_div_l1951_195102

theorem binomial_prime_div {p : ℕ} {m : ℕ} (hp : Nat.Prime p) (hm : 0 < m) : (Nat.choose (p ^ m) p - p ^ (m - 1)) % p ^ m = 0 := 
  sorry

end NUMINAMATH_GPT_binomial_prime_div_l1951_195102


namespace NUMINAMATH_GPT_factorial_mod_11_l1951_195195

def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_mod_11 : (factorial 13) % 11 = 0 := by
  sorry

end NUMINAMATH_GPT_factorial_mod_11_l1951_195195


namespace NUMINAMATH_GPT_cube_root_21952_is_28_l1951_195199

theorem cube_root_21952_is_28 :
  ∃ n : ℕ, n^3 = 21952 ∧ n = 28 :=
sorry

end NUMINAMATH_GPT_cube_root_21952_is_28_l1951_195199


namespace NUMINAMATH_GPT_greatest_pq_plus_r_l1951_195143

theorem greatest_pq_plus_r (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) 
  (h : p * q + q * r + r * p = 2016) : 
  pq + r ≤ 1008 :=
sorry

end NUMINAMATH_GPT_greatest_pq_plus_r_l1951_195143


namespace NUMINAMATH_GPT_greatest_k_dividing_n_l1951_195173

noncomputable def num_divisors (n : ℕ) : ℕ :=
  n.divisors.card

theorem greatest_k_dividing_n (n : ℕ) (h_pos : n > 0)
  (h_n_divisors : num_divisors n = 120)
  (h_5n_divisors : num_divisors (5 * n) = 144) :
  ∃ k : ℕ, 5^k ∣ n ∧ (∀ m : ℕ, 5^m ∣ n → m ≤ k) ∧ k = 4 :=
by sorry

end NUMINAMATH_GPT_greatest_k_dividing_n_l1951_195173


namespace NUMINAMATH_GPT_molecular_weight_of_6_moles_l1951_195193

-- Define the molecular weight of the compound
def molecular_weight : ℕ := 1404

-- Define the number of moles
def number_of_moles : ℕ := 6

-- The hypothesis would be the molecular weight condition
theorem molecular_weight_of_6_moles : number_of_moles * molecular_weight = 8424 :=
by sorry

end NUMINAMATH_GPT_molecular_weight_of_6_moles_l1951_195193


namespace NUMINAMATH_GPT_last_third_speed_l1951_195137

-- Definitions based on the conditions in the problem statement
def first_third_speed : ℝ := 80
def second_third_speed : ℝ := 30
def average_speed : ℝ := 45

-- Definition of the distance covered variable (non-zero to avoid division by zero)
variable (D : ℝ) (hD : D ≠ 0)

-- The unknown speed during the last third of the distance
noncomputable def V : ℝ := 
  D / ((D / 3 / first_third_speed) + (D / 3 / second_third_speed) + (D / 3 / average_speed))

-- The theorem to prove
theorem last_third_speed : V = 48 :=
by
  sorry

end NUMINAMATH_GPT_last_third_speed_l1951_195137


namespace NUMINAMATH_GPT_segment_length_in_meters_l1951_195123

-- Conditions
def inch_to_meters : ℝ := 500
def segment_length_in_inches : ℝ := 7.25

-- Theorem to prove
theorem segment_length_in_meters : segment_length_in_inches * inch_to_meters = 3625 := by
  sorry

end NUMINAMATH_GPT_segment_length_in_meters_l1951_195123


namespace NUMINAMATH_GPT_carl_typing_hours_per_day_l1951_195170

theorem carl_typing_hours_per_day (words_per_minute : ℕ) (total_words : ℕ) (days : ℕ) (hours_per_day : ℕ) :
  words_per_minute = 50 →
  total_words = 84000 →
  days = 7 →
  hours_per_day = (total_words / days) / (words_per_minute * 60) →
  hours_per_day = 4 :=
by
  intros h_word_rate h_total_words h_days h_hrs_formula
  rewrite [h_word_rate, h_total_words, h_days] at h_hrs_formula
  exact h_hrs_formula

end NUMINAMATH_GPT_carl_typing_hours_per_day_l1951_195170


namespace NUMINAMATH_GPT_min_value_a_plus_b_l1951_195118

theorem min_value_a_plus_b (a b : ℕ) (h₁ : 79 ∣ (a + 77 * b)) (h₂ : 77 ∣ (a + 79 * b)) : a + b = 193 :=
by
  sorry

end NUMINAMATH_GPT_min_value_a_plus_b_l1951_195118


namespace NUMINAMATH_GPT_find_ccb_l1951_195151

theorem find_ccb (a b c : ℕ) 
  (h1: a ≠ b) 
  (h2: a ≠ c) 
  (h3: b ≠ c) 
  (h4: b = 1) 
  (h5: (10 * a + b) ^ 2 = 100 * c + 10 * c + b) 
  (h6: 100 * c + 10 * c + b > 300) : 
  100 * c + 10 * c + b = 441 :=
sorry

end NUMINAMATH_GPT_find_ccb_l1951_195151


namespace NUMINAMATH_GPT_find_a_l1951_195108

-- Define the given context (condition)
def condition (a : ℝ) : Prop := 0.5 / 100 * a = 75 / 100 -- since 1 paise = 1/100 rupee

-- Define the statement to prove
theorem find_a (a : ℝ) (h : condition a) : a = 150 := 
sorry

end NUMINAMATH_GPT_find_a_l1951_195108


namespace NUMINAMATH_GPT_required_tiles_0_4m_l1951_195182

-- Defining given conditions
def num_tiles_0_3m : ℕ := 720
def side_length_0_3m : ℝ := 0.3
def side_length_0_4m : ℝ := 0.4

-- The problem statement translated to Lean 4
theorem required_tiles_0_4m : (side_length_0_4m ^ 2) * (405 : ℝ) = (side_length_0_3m ^ 2) * (num_tiles_0_3m : ℝ) := 
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_required_tiles_0_4m_l1951_195182


namespace NUMINAMATH_GPT_fair_bets_allocation_l1951_195120

theorem fair_bets_allocation (p_a : ℚ) (p_b : ℚ) (coins : ℚ) 
  (h_prob : p_a = 3 / 4 ∧ p_b = 1 / 4) (h_coins : coins = 96) : 
  (coins * p_a = 72) ∧ (coins * p_b = 24) :=
by 
  sorry

end NUMINAMATH_GPT_fair_bets_allocation_l1951_195120


namespace NUMINAMATH_GPT_quotient_is_six_l1951_195179

def larger_number (L : ℕ) : Prop := L = 1620
def difference (L S : ℕ) : Prop := L - S = 1365
def division_remainder (L S Q : ℕ) : Prop := L = S * Q + 15

theorem quotient_is_six (L S Q : ℕ) 
  (hL : larger_number L) 
  (hdiff : difference L S) 
  (hdiv : division_remainder L S Q) : Q = 6 :=
sorry

end NUMINAMATH_GPT_quotient_is_six_l1951_195179


namespace NUMINAMATH_GPT_ice_cream_sales_l1951_195142

theorem ice_cream_sales : 
  let tuesday_sales := 12000
  let wednesday_sales := 2 * tuesday_sales
  let total_sales := tuesday_sales + wednesday_sales
  total_sales = 36000 := 
by 
  sorry

end NUMINAMATH_GPT_ice_cream_sales_l1951_195142


namespace NUMINAMATH_GPT_sum_of_c_and_d_l1951_195185

theorem sum_of_c_and_d (c d : ℝ) 
  (h1 : ∀ x, x ≠ 2 ∧ x ≠ -1 → x^2 + c * x + d ≠ 0)
  (h_asymp_2 : 2^2 + c * 2 + d = 0)
  (h_asymp_neg1 : (-1)^2 + c * (-1) + d = 0) :
  c + d = -3 :=
by 
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_sum_of_c_and_d_l1951_195185


namespace NUMINAMATH_GPT_jerry_time_proof_l1951_195162

noncomputable def tom_walk_speed (step_length_tom : ℕ) (pace_tom : ℕ) : ℕ := 
  step_length_tom * pace_tom

noncomputable def tom_distance_to_office (walk_speed_tom : ℕ) (time_tom : ℕ) : ℕ :=
  walk_speed_tom * time_tom

noncomputable def jerry_walk_speed (step_length_jerry : ℕ) (pace_jerry : ℕ) : ℕ :=
  step_length_jerry * pace_jerry

noncomputable def jerry_time_to_office (distance_to_office : ℕ) (walk_speed_jerry : ℕ) : ℚ :=
  distance_to_office / walk_speed_jerry

theorem jerry_time_proof :
  let step_length_tom := 80
  let pace_tom := 85
  let time_tom := 20
  let step_length_jerry := 70
  let pace_jerry := 110
  let office_distance := tom_distance_to_office (tom_walk_speed step_length_tom pace_tom) time_tom
  let jerry_speed := jerry_walk_speed step_length_jerry pace_jerry
  jerry_time_to_office office_distance jerry_speed = 53/3 := 
by
  sorry

end NUMINAMATH_GPT_jerry_time_proof_l1951_195162


namespace NUMINAMATH_GPT_find_S6_l1951_195124

variable (a : ℕ → ℝ) (S_n : ℕ → ℝ)

-- The sequence {a_n} is given as a geometric sequence
-- Partial sums are given as S_2 = 1 and S_4 = 3

-- Conditions
axiom geom_sequence : ∀ n : ℕ, a (n + 1) / a n = a 1 / a 0
axiom S2 : S_n 2 = 1
axiom S4 : S_n 4 = 3

-- Theorem statement
theorem find_S6 : S_n 6 = 7 :=
sorry

end NUMINAMATH_GPT_find_S6_l1951_195124


namespace NUMINAMATH_GPT_possible_apple_counts_l1951_195141

theorem possible_apple_counts (n : ℕ) (h₁ : 70 ≤ n) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end NUMINAMATH_GPT_possible_apple_counts_l1951_195141


namespace NUMINAMATH_GPT_travel_distance_proof_l1951_195150

-- Definitions based on conditions
def total_distance : ℕ := 369
def amoli_speed : ℕ := 42
def amoli_time : ℕ := 3
def anayet_speed : ℕ := 61
def anayet_time : ℕ := 2

-- Calculate distances traveled
def amoli_distance : ℕ := amoli_speed * amoli_time
def anayet_distance : ℕ := anayet_speed * anayet_time

-- Calculate total distance covered
def total_distance_covered : ℕ := amoli_distance + anayet_distance

-- Define remaining distance to travel
def remaining_distance (total : ℕ) (covered : ℕ) : ℕ := total - covered

-- The theorem to prove
theorem travel_distance_proof : remaining_distance total_distance total_distance_covered = 121 := by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_travel_distance_proof_l1951_195150


namespace NUMINAMATH_GPT_total_points_seven_players_l1951_195171

theorem total_points_seven_players (S : ℕ) (x : ℕ) 
  (hAlex : Alex_scored = S / 4)
  (hBen : Ben_scored = 2 * S / 7)
  (hCharlie : Charlie_scored = 15)
  (hTotal : S / 4 + 2 * S / 7 + 15 + x = S)
  (hMultiple : S = 56) : 
  x = 11 := 
sorry

end NUMINAMATH_GPT_total_points_seven_players_l1951_195171


namespace NUMINAMATH_GPT_find_sum_invested_l1951_195121

noncomputable def sum_invested (interest_difference: ℝ) (rate1: ℝ) (rate2: ℝ) (time: ℝ): ℝ := 
  interest_difference * 100 / (time * (rate1 - rate2))

theorem find_sum_invested :
  let interest_difference := 600
  let rate1 := 18 / 100
  let rate2 := 12 / 100
  let time := 2
  sum_invested interest_difference rate1 rate2 time = 5000 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_invested_l1951_195121


namespace NUMINAMATH_GPT_minimum_additional_squares_to_symmetry_l1951_195191

-- Define the type for coordinates in the grid
structure Coord where
  x : Nat
  y : Nat

-- Define the conditions
def initial_shaded_squares : List Coord := [
  ⟨2, 4⟩, ⟨3, 2⟩, ⟨5, 1⟩, ⟨1, 4⟩
]

def grid_size : Coord := ⟨6, 5⟩

def vertical_line_of_symmetry : Nat := 3 -- between columns 3 and 4
def horizontal_line_of_symmetry : Nat := 2 -- between rows 2 and 3

-- Define reflection across lines of symmetry
def reflect_vertical (c : Coord) : Coord :=
  ⟨2 * vertical_line_of_symmetry - c.x, c.y⟩

def reflect_horizontal (c : Coord) : Coord :=
  ⟨c.x, 2 * horizontal_line_of_symmetry - c.y⟩

def reflect_both (c : Coord) : Coord :=
  reflect_vertical (reflect_horizontal c)

-- Define the theorem
theorem minimum_additional_squares_to_symmetry :
  ∃ (additional_squares : Nat), additional_squares = 5 := 
sorry

end NUMINAMATH_GPT_minimum_additional_squares_to_symmetry_l1951_195191


namespace NUMINAMATH_GPT_break_even_price_correct_l1951_195136

-- Conditions
def variable_cost_per_handle : ℝ := 0.60
def fixed_cost_per_week : ℝ := 7640
def handles_per_week : ℝ := 1910

-- Define the correct answer for the price per handle to break even
def break_even_price_per_handle : ℝ := 4.60

-- The statement to prove
theorem break_even_price_correct :
  fixed_cost_per_week + (variable_cost_per_handle * handles_per_week) / handles_per_week = break_even_price_per_handle :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_break_even_price_correct_l1951_195136


namespace NUMINAMATH_GPT_measure_angle_T_l1951_195109

theorem measure_angle_T (P Q R S T : ℝ) (h₀ : P = R) (h₁ : R = T) (h₂ : Q + S = 180)
  (h_sum : P + Q + R + T + S = 540) : T = 120 :=
by
  sorry

end NUMINAMATH_GPT_measure_angle_T_l1951_195109


namespace NUMINAMATH_GPT_min_book_corner_cost_l1951_195186

theorem min_book_corner_cost :
  ∃ x : ℕ, 0 ≤ x ∧ x ≤ 30 ∧
  80 * x + 30 * (30 - x) ≤ 1900 ∧
  50 * x + 60 * (30 - x) ≤ 1620 ∧
  860 * x + 570 * (30 - x) = 22320 := sorry

end NUMINAMATH_GPT_min_book_corner_cost_l1951_195186


namespace NUMINAMATH_GPT_isosceles_triangle_vertex_angle_l1951_195175

theorem isosceles_triangle_vertex_angle (α β γ : ℝ) (h_triangle : α + β + γ = 180)
  (h_isosceles : α = β ∨ β = α ∨ α = γ ∨ γ = α ∨ β = γ ∨ γ = β)
  (h_ratio : α / γ = 1 / 4 ∨ γ / α = 1 / 4) :
  (γ = 20 ∨ γ = 120) :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_vertex_angle_l1951_195175


namespace NUMINAMATH_GPT_total_polled_votes_proof_l1951_195160

-- Define the conditions
variables (V : ℕ) -- total number of valid votes
variables (invalid_votes : ℕ) -- number of invalid votes
variables (total_polled_votes : ℕ) -- total polled votes
variables (candidateA_votes candidateB_votes : ℕ) -- votes for candidate A and B respectively

-- Assume the known conditions
variable (h1 : candidateA_votes = 45 * V / 100) -- candidate A got 45% of valid votes
variable (h2 : candidateB_votes = 55 * V / 100) -- candidate B got 55% of valid votes
variable (h3 : candidateB_votes - candidateA_votes = 9000) -- candidate A was defeated by 9000 votes
variable (h4 : invalid_votes = 83) -- there are 83 invalid votes
variable (h5 : total_polled_votes = V + invalid_votes) -- total polled votes is sum of valid and invalid votes

-- Define the theorem to prove
theorem total_polled_votes_proof : total_polled_votes = 90083 :=
by 
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_total_polled_votes_proof_l1951_195160


namespace NUMINAMATH_GPT_circle_tangent_to_line_at_parabola_focus_l1951_195105

noncomputable def parabola_focus : (ℝ × ℝ) := (2, 0)

def line_eq (p : ℝ × ℝ) : Prop := p.2 = p.1

def circle_eq (center radius : ℝ) (p : ℝ × ℝ) : Prop := 
  (p.1 - center)^2 + p.2^2 = radius

theorem circle_tangent_to_line_at_parabola_focus : 
  ∀ p : ℝ × ℝ, (circle_eq 2 2 p ↔ (line_eq p ∧ p = parabola_focus)) := by
  sorry

end NUMINAMATH_GPT_circle_tangent_to_line_at_parabola_focus_l1951_195105


namespace NUMINAMATH_GPT_sum_of_coefficients_l1951_195157

theorem sum_of_coefficients : 
  ∃ (a b c d e f g h j k : ℤ), 
    (27 * x^6 - 512 * y^6 = (a * x + b * y) * (c * x^2 + d * x * y + e * y^2) * (f * x + g * y) * (h * x^2 + j * x * y + k * y^2)) → 
    (a + b + c + d + e + f + g + h + j + k = 92) :=
sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1951_195157


namespace NUMINAMATH_GPT_find_multiple_l1951_195155

theorem find_multiple (m : ℤ) : 38 + m * 43 = 124 → m = 2 := by
  intro h
  sorry

end NUMINAMATH_GPT_find_multiple_l1951_195155


namespace NUMINAMATH_GPT_cost_of_one_book_l1951_195152

theorem cost_of_one_book (x : ℝ) : 
  (9 * x = 11) ∧ (13 * x = 15) → x = 1.23 :=
by sorry

end NUMINAMATH_GPT_cost_of_one_book_l1951_195152


namespace NUMINAMATH_GPT_find_x_plus_y_l1951_195156

variable (x y : ℝ)

theorem find_x_plus_y (h1 : |x| + x + y = 8) (h2 : x + |y| - y = 10) : x + y = 14 / 5 := 
by
  sorry

end NUMINAMATH_GPT_find_x_plus_y_l1951_195156


namespace NUMINAMATH_GPT_geometric_sequence_a3_a5_l1951_195178

-- Define the geometric sequence condition using a function
def is_geometric_seq (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

-- Define the given conditions
variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (h1 : is_geometric_seq a)
variable (h2 : a 1 > 0)
variable (h3 : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25)

-- The main goal is to prove: a 3 + a 5 = 5
theorem geometric_sequence_a3_a5 : a 3 + a 5 = 5 :=
by
  simp [is_geometric_seq] at h1
  obtain ⟨q, ⟨hq_pos, hq⟩⟩ := h1
  sorry

end NUMINAMATH_GPT_geometric_sequence_a3_a5_l1951_195178


namespace NUMINAMATH_GPT_discount_percentage_l1951_195198

theorem discount_percentage (sale_price original_price : ℝ) (h1 : sale_price = 480) (h2 : original_price = 600) : 
  100 * (original_price - sale_price) / original_price = 20 := by 
  sorry

end NUMINAMATH_GPT_discount_percentage_l1951_195198


namespace NUMINAMATH_GPT_Dana_pencils_equals_combined_l1951_195119

-- Definitions based on given conditions
def pencils_Jayden : ℕ := 20
def pencils_Marcus (pencils_Jayden : ℕ) : ℕ := pencils_Jayden / 2
def pencils_Dana (pencils_Jayden : ℕ) : ℕ := pencils_Jayden + 15
def pencils_Ella (pencils_Marcus : ℕ) : ℕ := 3 * pencils_Marcus - 5
def combined_pencils (pencils_Marcus : ℕ) (pencils_Ella : ℕ) : ℕ := pencils_Marcus + pencils_Ella

-- Theorem to prove:
theorem Dana_pencils_equals_combined (pencils_Jayden : ℕ := 20) : 
  pencils_Dana pencils_Jayden = combined_pencils (pencils_Marcus pencils_Jayden) (pencils_Ella (pencils_Marcus pencils_Jayden)) := by
  sorry

end NUMINAMATH_GPT_Dana_pencils_equals_combined_l1951_195119


namespace NUMINAMATH_GPT_vehicle_A_must_pass_B_before_B_collides_with_C_l1951_195126

theorem vehicle_A_must_pass_B_before_B_collides_with_C
  (V_A : ℝ) -- speed of vehicle A in mph
  (V_B : ℝ := 40) -- speed of vehicle B in mph
  (V_C : ℝ := 65) -- speed of vehicle C in mph
  (distance_AB : ℝ := 100) -- distance between A and B in ft
  (distance_BC : ℝ := 250) -- initial distance between B and C in ft
  : (V_A > (100 * 65 - 150 * 40) / 250) :=
by {
  sorry
}

end NUMINAMATH_GPT_vehicle_A_must_pass_B_before_B_collides_with_C_l1951_195126


namespace NUMINAMATH_GPT_shortest_distance_to_circle_l1951_195122

def center : ℝ × ℝ := (8, 7)
def radius : ℝ := 5
def point : ℝ × ℝ := (1, -2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2))

theorem shortest_distance_to_circle :
  distance point center - radius = Real.sqrt 130 - 5 :=
by
  sorry

end NUMINAMATH_GPT_shortest_distance_to_circle_l1951_195122


namespace NUMINAMATH_GPT_new_percentage_of_girls_is_5_l1951_195158

theorem new_percentage_of_girls_is_5
  (initial_children : ℕ)
  (percentage_boys : ℕ)
  (added_boys : ℕ)
  (initial_total_boys : ℕ)
  (initial_total_girls : ℕ)
  (new_total_boys : ℕ)
  (new_total_children : ℕ)
  (new_percentage_girls : ℕ)
  (h1 : initial_children = 60)
  (h2 : percentage_boys = 90)
  (h3 : added_boys = 60)
  (h4 : initial_total_boys = (percentage_boys * initial_children / 100))
  (h5 : initial_total_girls = initial_children - initial_total_boys)
  (h6 : new_total_boys = initial_total_boys + added_boys)
  (h7 : new_total_children = initial_children + added_boys)
  (h8 : new_percentage_girls = (initial_total_girls * 100 / new_total_children)) :
  new_percentage_girls = 5 :=
by sorry

end NUMINAMATH_GPT_new_percentage_of_girls_is_5_l1951_195158


namespace NUMINAMATH_GPT_sum_of_corners_9x9_grid_l1951_195184

theorem sum_of_corners_9x9_grid : 
  let topLeft := 1
  let topRight := 9
  let bottomLeft := 73
  let bottomRight := 81
  topLeft + topRight + bottomLeft + bottomRight = 164 :=
by {
  let topLeft := 1
  let topRight := 9
  let bottomLeft := 73
  let bottomRight := 81
  show topLeft + topRight + bottomLeft + bottomRight = 164
  sorry
}

end NUMINAMATH_GPT_sum_of_corners_9x9_grid_l1951_195184
