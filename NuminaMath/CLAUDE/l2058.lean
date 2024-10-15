import Mathlib

namespace NUMINAMATH_CALUDE_lcm_18_35_l2058_205846

theorem lcm_18_35 : Nat.lcm 18 35 = 630 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_35_l2058_205846


namespace NUMINAMATH_CALUDE_max_candies_drawn_exists_ten_candies_drawn_l2058_205828

/-- Represents the number of candies of each color --/
structure CandyCount where
  yellow : ℕ
  red : ℕ
  blue : ℕ

/-- Represents the state of candies before and after drawing --/
structure CandyState where
  initial : CandyCount
  drawn : ℕ
  final : CandyCount

/-- Checks if the candy state satisfies all conditions --/
def satisfiesConditions (state : CandyState) : Prop :=
  state.initial.yellow * 3 = state.initial.red * 5 ∧
  state.final.yellow = 2 ∧
  state.final.red = 2 ∧
  state.final.blue ≥ 5 ∧
  state.drawn = state.initial.yellow + state.initial.red + state.initial.blue -
                (state.final.yellow + state.final.red + state.final.blue)

/-- Theorem stating that the maximum number of candies Petya can draw is 10 --/
theorem max_candies_drawn (state : CandyState) :
  satisfiesConditions state → state.drawn ≤ 10 :=
by
  sorry

/-- Theorem stating that it's possible to draw exactly 10 candies while satisfying all conditions --/
theorem exists_ten_candies_drawn :
  ∃ state : CandyState, satisfiesConditions state ∧ state.drawn = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_max_candies_drawn_exists_ten_candies_drawn_l2058_205828


namespace NUMINAMATH_CALUDE_notebook_cost_l2058_205840

/-- Given the following conditions:
  * Total spent on school supplies is $32
  * A backpack costs $15
  * A pack of pens costs $1
  * A pack of pencils costs $1
  * 5 multi-subject notebooks were bought
Prove that each notebook costs $3 -/
theorem notebook_cost (total_spent : ℚ) (backpack_cost : ℚ) (pen_cost : ℚ) (pencil_cost : ℚ) (notebook_count : ℕ) :
  total_spent = 32 →
  backpack_cost = 15 →
  pen_cost = 1 →
  pencil_cost = 1 →
  notebook_count = 5 →
  (total_spent - backpack_cost - pen_cost - pencil_cost) / notebook_count = 3 := by
  sorry

#check notebook_cost

end NUMINAMATH_CALUDE_notebook_cost_l2058_205840


namespace NUMINAMATH_CALUDE_mod_eight_thirteen_fourth_l2058_205839

theorem mod_eight_thirteen_fourth (m : ℕ) : 
  13^4 ≡ m [ZMOD 8] → 0 ≤ m → m < 8 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_mod_eight_thirteen_fourth_l2058_205839


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l2058_205851

theorem roots_sum_of_squares (p q r s : ℝ) : 
  (r^2 - p*r + q = 0) → (s^2 - p*s + q = 0) → r^2 + s^2 = p^2 - 2*q :=
by sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l2058_205851


namespace NUMINAMATH_CALUDE_third_number_problem_l2058_205854

theorem third_number_problem (x : ℝ) : 
  (14 + 32 + x) / 3 = (21 + 47 + 22) / 3 + 3 → x = 53 := by
  sorry

end NUMINAMATH_CALUDE_third_number_problem_l2058_205854


namespace NUMINAMATH_CALUDE_rings_per_game_l2058_205832

theorem rings_per_game (total_rings : ℕ) (num_games : ℕ) (rings_per_game : ℕ) 
  (h1 : total_rings = 48) 
  (h2 : num_games = 8) 
  (h3 : total_rings = num_games * rings_per_game) : 
  rings_per_game = 6 := by
  sorry

end NUMINAMATH_CALUDE_rings_per_game_l2058_205832


namespace NUMINAMATH_CALUDE_coefficient_x_squared_is_80_l2058_205893

/-- The coefficient of x^2 in the expansion of (2x + 1/x^2)^5 -/
def coefficient_x_squared : ℕ :=
  (Nat.choose 5 1) * (2^4)

/-- Theorem stating that the coefficient of x^2 in the expansion of (2x + 1/x^2)^5 is 80 -/
theorem coefficient_x_squared_is_80 : coefficient_x_squared = 80 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_is_80_l2058_205893


namespace NUMINAMATH_CALUDE_scallop_dinner_cost_l2058_205886

/-- Calculates the cost of scallops for a dinner party. -/
def scallop_cost (people : ℕ) (scallops_per_person : ℕ) (scallops_per_pound : ℕ) (cost_per_pound : ℚ) : ℚ :=
  (people * scallops_per_person : ℚ) / scallops_per_pound * cost_per_pound

/-- Proves that the cost of scallops for 8 people, given 2 scallops per person, 
    is $48.00, when 8 scallops weigh one pound and cost $24.00 per pound. -/
theorem scallop_dinner_cost : 
  scallop_cost 8 2 8 24 = 48 := by
  sorry

end NUMINAMATH_CALUDE_scallop_dinner_cost_l2058_205886


namespace NUMINAMATH_CALUDE_set_intersection_union_theorem_l2058_205876

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

-- State the theorem
theorem set_intersection_union_theorem (a b : ℝ) :
  A ∪ B a b = Set.univ ∧ A ∩ B a b = Set.Ioc 3 4 → a = -3 ∧ b = -4 := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_union_theorem_l2058_205876


namespace NUMINAMATH_CALUDE_binary_equals_21_l2058_205827

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- The binary representation of the number in question -/
def binary_number : List Bool := [true, false, true, false, true]

/-- Theorem stating that the given binary number equals 21 in decimal -/
theorem binary_equals_21 : binary_to_decimal binary_number = 21 := by
  sorry

end NUMINAMATH_CALUDE_binary_equals_21_l2058_205827


namespace NUMINAMATH_CALUDE_composition_value_l2058_205821

theorem composition_value (c d : ℝ) 
  (f : ℝ → ℝ) (g : ℝ → ℝ) 
  (hf : ∀ x, f x = 5*x + c)
  (hg : ∀ x, g x = c*x + 3)
  (h_comp : ∀ x, f (g x) = 15*x + d) : 
  d = 18 := by
sorry

end NUMINAMATH_CALUDE_composition_value_l2058_205821


namespace NUMINAMATH_CALUDE_three_year_deposit_optimal_l2058_205874

/-- Represents the deposit options available --/
inductive DepositOption
  | OneYearRepeated
  | OneYearThenTwoYear
  | TwoYearThenOneYear
  | ThreeYear

/-- Calculates the final amount for a given deposit option --/
def calculateFinalAmount (option : DepositOption) (initialDeposit : ℝ) : ℝ :=
  match option with
  | .OneYearRepeated => initialDeposit * (1 + 0.0414 * 0.8)^3
  | .OneYearThenTwoYear => initialDeposit * (1 + 0.0414 * 0.8) * (1 + 0.0468 * 0.8 * 2)
  | .TwoYearThenOneYear => initialDeposit * (1 + 0.0468 * 0.8 * 2) * (1 + 0.0414 * 0.8)
  | .ThreeYear => initialDeposit * (1 + 0.0540 * 3 * 0.8)

/-- Theorem stating that the three-year fixed deposit option yields the highest return --/
theorem three_year_deposit_optimal (initialDeposit : ℝ) (h : initialDeposit > 0) :
  ∀ option : DepositOption, calculateFinalAmount .ThreeYear initialDeposit ≥ calculateFinalAmount option initialDeposit :=
by sorry

end NUMINAMATH_CALUDE_three_year_deposit_optimal_l2058_205874


namespace NUMINAMATH_CALUDE_triangle_existence_l2058_205810

theorem triangle_existence (a b c d e : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) : 
  ∃ (x y z : ℝ), 
    x = Real.sqrt (b^2 + c^2 + d^2) ∧ 
    y = Real.sqrt (a^2 + b^2 + c^2 + e^2 + 2*a*c) ∧ 
    z = Real.sqrt (a^2 + d^2 + e^2 + 2*d*e) ∧ 
    x + y > z ∧ y + z > x ∧ z + x > y :=
sorry

end NUMINAMATH_CALUDE_triangle_existence_l2058_205810


namespace NUMINAMATH_CALUDE_max_sum_of_entries_l2058_205825

def numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_of_entries (top : List ℕ) (left : List ℕ) : ℕ :=
  (top.sum * left.sum)

def is_valid_partition (l1 l2 : List ℕ) : Prop :=
  l1.length = 4 ∧ l2.length = 4 ∧ (l1 ++ l2).toFinset = numbers.toFinset

theorem max_sum_of_entries :
  ∃ (top left : List ℕ), 
    is_valid_partition top left ∧ 
    sum_of_entries top left = 1440 ∧
    ∀ (t l : List ℕ), is_valid_partition t l → sum_of_entries t l ≤ 1440 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_entries_l2058_205825


namespace NUMINAMATH_CALUDE_is_14th_term_l2058_205889

/-- The sequence term for a given index -/
def sequenceTerm (n : ℕ) : ℚ := (n + 3 : ℚ) / (n + 1 : ℚ)

/-- Theorem stating that 17/15 is the 14th term of the sequence -/
theorem is_14th_term : sequenceTerm 14 = 17 / 15 := by
  sorry

end NUMINAMATH_CALUDE_is_14th_term_l2058_205889


namespace NUMINAMATH_CALUDE_gcd_process_max_rows_l2058_205892

/-- Represents the GCD process described in the problem -/
def gcd_process (initial_sequence : List Nat) : Nat :=
  sorry

/-- The maximum number of rows in the GCD process -/
def max_rows : Nat := 501

/-- Theorem stating that the maximum number of rows in the GCD process is 501 -/
theorem gcd_process_max_rows :
  ∀ (seq : List Nat),
    (∀ n ∈ seq, 500 ≤ n ∧ n ≤ 1499) →
    seq.length = 1000 →
    gcd_process seq ≤ max_rows :=
  sorry

end NUMINAMATH_CALUDE_gcd_process_max_rows_l2058_205892


namespace NUMINAMATH_CALUDE_bowen_purchase_ratio_l2058_205812

/-- Represents the purchase of pens and pencils -/
structure Purchase where
  pen_price : ℚ
  pencil_price : ℚ
  num_pens : ℕ
  total_spent : ℚ

/-- Calculates the ratio of pencils to pens for a given purchase -/
def pencil_to_pen_ratio (p : Purchase) : ℚ × ℚ :=
  let pencil_cost := p.total_spent - p.pen_price * p.num_pens
  let num_pencils := pencil_cost / p.pencil_price
  let gcd := Nat.gcd (Nat.floor num_pencils) p.num_pens
  ((num_pencils / gcd), (p.num_pens / gcd))

/-- Theorem stating that for the given purchase conditions, the ratio of pencils to pens is 7:5 -/
theorem bowen_purchase_ratio : 
  let p : Purchase := {
    pen_price := 15/100,
    pencil_price := 25/100,
    num_pens := 40,
    total_spent := 20
  }
  pencil_to_pen_ratio p = (7, 5) := by sorry

end NUMINAMATH_CALUDE_bowen_purchase_ratio_l2058_205812


namespace NUMINAMATH_CALUDE_sum_of_numbers_with_given_hcf_and_lcm_factors_l2058_205845

theorem sum_of_numbers_with_given_hcf_and_lcm_factors
  (a b : ℕ+)
  (h_hcf : Nat.gcd a b = 23)
  (h_lcm : Nat.lcm a b = 81328) :
  a + b = 667 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_with_given_hcf_and_lcm_factors_l2058_205845


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2058_205860

/-- The line ax+by-2=0 passes through the point (4,2) for all a and b that satisfy 2a+b=1 -/
theorem line_passes_through_fixed_point (a b : ℝ) (h : 2*a + b = 1) :
  a*4 + b*2 - 2 = 0 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2058_205860


namespace NUMINAMATH_CALUDE_rogers_new_crayons_l2058_205831

/-- Given that Roger has 4 used crayons, 8 broken crayons, and a total of 14 crayons,
    prove that the number of new crayons is 2. -/
theorem rogers_new_crayons (used : ℕ) (broken : ℕ) (total : ℕ) (new : ℕ) :
  used = 4 →
  broken = 8 →
  total = 14 →
  new + used + broken = total →
  new = 2 := by
  sorry

end NUMINAMATH_CALUDE_rogers_new_crayons_l2058_205831


namespace NUMINAMATH_CALUDE_cody_reading_time_l2058_205882

def read_series (total_books : ℕ) (first_week : ℕ) (second_week : ℕ) (subsequent_weeks : ℕ) : ℕ :=
  let books_first_two_weeks := first_week + second_week
  let remaining_books := total_books - books_first_two_weeks
  let additional_weeks := (remaining_books + subsequent_weeks - 1) / subsequent_weeks
  2 + additional_weeks

theorem cody_reading_time :
  read_series 54 6 3 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_cody_reading_time_l2058_205882


namespace NUMINAMATH_CALUDE_quadratic_function_range_l2058_205815

/-- Given a quadratic function f(x) = ax^2 - c, prove that if -4 ≤ f(1) ≤ -1 and -1 ≤ f(2) ≤ 5, then -1 ≤ f(3) ≤ 20. -/
theorem quadratic_function_range (a c : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^2 - c
  (-4 ≤ f 1 ∧ f 1 ≤ -1) → (-1 ≤ f 2 ∧ f 2 ≤ 5) → (-1 ≤ f 3 ∧ f 3 ≤ 20) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l2058_205815


namespace NUMINAMATH_CALUDE_carters_dog_height_l2058_205833

-- Define heights in inches
def betty_height : ℕ := 3 * 12  -- 3 feet converted to inches
def carter_height : ℕ := betty_height + 12
def dog_height : ℕ := carter_height / 2

-- Theorem statement
theorem carters_dog_height : dog_height = 24 := by
  sorry

end NUMINAMATH_CALUDE_carters_dog_height_l2058_205833


namespace NUMINAMATH_CALUDE_favorite_song_probability_l2058_205842

/-- Represents a digital music player with a collection of songs. -/
structure MusicPlayer where
  numSongs : Nat
  shortestSongDuration : Nat
  durationIncrement : Nat
  favoriteSongDuration : Nat
  playbackDuration : Nat

/-- Calculates the probability of not hearing the favorite song in full 
    within the given playback duration. -/
def probabilityNoFavoriteSong (player : MusicPlayer) : Rat :=
  sorry

/-- Theorem stating the probability of not hearing the favorite song in full
    for the specific music player configuration. -/
theorem favorite_song_probability (player : MusicPlayer) 
  (h1 : player.numSongs = 12)
  (h2 : player.shortestSongDuration = 40)
  (h3 : player.durationIncrement = 40)
  (h4 : player.favoriteSongDuration = 300)
  (h5 : player.playbackDuration = 360) :
  probabilityNoFavoriteSong player = 43 / 48 := by
  sorry

end NUMINAMATH_CALUDE_favorite_song_probability_l2058_205842


namespace NUMINAMATH_CALUDE_plain_pancakes_count_l2058_205894

theorem plain_pancakes_count (total : ℕ) (blueberry : ℕ) (banana : ℕ) 
  (h1 : total = 67) (h2 : blueberry = 20) (h3 : banana = 24) : 
  total - (blueberry + banana) = 23 := by
  sorry

end NUMINAMATH_CALUDE_plain_pancakes_count_l2058_205894


namespace NUMINAMATH_CALUDE_original_number_proof_l2058_205817

theorem original_number_proof : 
  ∃ (x : ℕ), x = 6 ∧ 
  (∀ (y : ℕ), y < x → ¬(25 ∣ (y + 19))) ∧ 
  (25 ∣ (x + 19)) := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2058_205817


namespace NUMINAMATH_CALUDE_expression_evaluation_equation_solutions_l2058_205896

-- Part 1
theorem expression_evaluation :
  |Real.sqrt 3 - 1| - 2 * Real.cos (60 * π / 180) + (Real.sqrt 3 - 2)^2 + Real.sqrt 12 = 5 - Real.sqrt 3 := by
  sorry

-- Part 2
theorem equation_solutions (x : ℝ) :
  2 * (x - 3)^2 = x^2 - 9 ↔ x = 3 ∨ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_equation_solutions_l2058_205896


namespace NUMINAMATH_CALUDE_keyboard_mouse_cost_ratio_l2058_205844

/-- Given a mouse cost and total expenditure, proves the ratio of keyboard to mouse cost -/
theorem keyboard_mouse_cost_ratio 
  (mouse_cost : ℝ) 
  (total_cost : ℝ) 
  (h1 : mouse_cost = 16) 
  (h2 : total_cost = 64) 
  (h3 : ∃ n : ℝ, total_cost = mouse_cost + n * mouse_cost) :
  ∃ n : ℝ, n = 3 ∧ total_cost = mouse_cost + n * mouse_cost :=
sorry

end NUMINAMATH_CALUDE_keyboard_mouse_cost_ratio_l2058_205844


namespace NUMINAMATH_CALUDE_simplify_polynomial_l2058_205859

theorem simplify_polynomial (w : ℝ) : 
  3*w + 4 - 6*w - 5 + 7*w + 8 - 9*w - 10 + 2*w^2 = 2*w^2 - 5*w - 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l2058_205859


namespace NUMINAMATH_CALUDE_arnold_protein_consumption_l2058_205899

/-- Calculates the total protein consumed given the protein content of different food items. -/
def total_protein_consumed (collagen_protein_per_2_scoops : ℕ) (protein_powder_per_scoop : ℕ) (steak_protein : ℕ) : ℕ :=
  let collagen_protein := collagen_protein_per_2_scoops / 2
  collagen_protein + protein_powder_per_scoop + steak_protein

/-- Proves that the total protein consumed is 86 grams given the specific food items. -/
theorem arnold_protein_consumption : 
  total_protein_consumed 18 21 56 = 86 := by
  sorry

end NUMINAMATH_CALUDE_arnold_protein_consumption_l2058_205899


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2058_205853

theorem inequality_equivalence (x : ℝ) : 
  (x / 4 ≤ 3 + x ∧ 3 + x < -3 * (1 + x)) ↔ -4 ≤ x ∧ x < -3/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2058_205853


namespace NUMINAMATH_CALUDE_sum_of_first_2015_digits_l2058_205879

/-- The repeating decimal 0.0142857 -/
def repeatingDecimal : ℚ := 1 / 7

/-- The length of the repeating part of the decimal -/
def repeatLength : ℕ := 6

/-- The sum of digits in one complete cycle of the repeating part -/
def cycleSum : ℕ := 27

/-- The number of complete cycles in the first 2015 digits -/
def completeCycles : ℕ := 2015 / repeatLength

/-- The number of remaining digits after complete cycles -/
def remainingDigits : ℕ := 2015 % repeatLength

/-- The sum of the remaining digits -/
def remainingSum : ℕ := 20

theorem sum_of_first_2015_digits : 
  (cycleSum * completeCycles + remainingSum : ℕ) = 9065 :=
sorry

end NUMINAMATH_CALUDE_sum_of_first_2015_digits_l2058_205879


namespace NUMINAMATH_CALUDE_star_card_probability_l2058_205888

theorem star_card_probability (total_cards : ℕ) (num_ranks : ℕ) (num_suits : ℕ) 
  (h1 : total_cards = 65)
  (h2 : num_ranks = 13)
  (h3 : num_suits = 5)
  (h4 : total_cards = num_ranks * num_suits) :
  (num_ranks : ℚ) / total_cards = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_star_card_probability_l2058_205888


namespace NUMINAMATH_CALUDE_f_properties_l2058_205863

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x < 0 then -x^2 - 4*x - 3
  else if x = 0 then 0
  else x^2 - 4*x + 3

-- State the theorem
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x > 0, f x = x^2 - 4*x + 3) →  -- given condition for x > 0
  (f (f (-1)) = 0) ∧  -- part 1
  (∀ x, f x = if x < 0 then -x^2 - 4*x - 3
              else if x = 0 then 0
              else x^2 - 4*x + 3) :=  -- part 2
by sorry

end NUMINAMATH_CALUDE_f_properties_l2058_205863


namespace NUMINAMATH_CALUDE_cone_height_from_sector_l2058_205871

/-- Given a sector paper with radius 13 cm and area 65π cm², prove that when formed into a cone, the height of the cone is 12 cm. -/
theorem cone_height_from_sector (r : ℝ) (h : ℝ) :
  r = 13 →
  r * r * π / 2 = 65 * π →
  h = 12 :=
by sorry

end NUMINAMATH_CALUDE_cone_height_from_sector_l2058_205871


namespace NUMINAMATH_CALUDE_triangle_centroid_inequality_l2058_205878

/-- Given a triangle ABC with side lengths a, b, and c, centroid G, and an arbitrary point P,
    prove that a⋅PA³ + b⋅PB³ + c⋅PC³ ≥ 3abc⋅PG -/
theorem triangle_centroid_inequality (A B C P : ℝ × ℝ) 
    (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  let G := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)
  let PA := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
  let PB := Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)
  let PC := Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2)
  let PG := Real.sqrt ((P.1 - G.1)^2 + (P.2 - G.2)^2)
  a * PA^3 + b * PB^3 + c * PC^3 ≥ 3 * a * b * c * PG := by
sorry


end NUMINAMATH_CALUDE_triangle_centroid_inequality_l2058_205878


namespace NUMINAMATH_CALUDE_jamies_shoes_cost_l2058_205850

/-- The cost of Jamie's shoes given the total cost and James' items -/
theorem jamies_shoes_cost (total_cost : ℕ) (coat_cost : ℕ) (jeans_cost : ℕ) : 
  total_cost = 110 →
  coat_cost = 40 →
  jeans_cost = 20 →
  total_cost = coat_cost + 2 * jeans_cost + (total_cost - (coat_cost + 2 * jeans_cost)) →
  (total_cost - (coat_cost + 2 * jeans_cost)) = 30 := by
sorry

end NUMINAMATH_CALUDE_jamies_shoes_cost_l2058_205850


namespace NUMINAMATH_CALUDE_march_production_l2058_205856

/-- Represents the monthly production function -/
def production_function (x : ℝ) : ℝ := x + 1

/-- March is represented by the number 3 -/
def march : ℝ := 3

/-- Theorem stating that the estimated production for March is 4 -/
theorem march_production :
  production_function march = 4 := by sorry

end NUMINAMATH_CALUDE_march_production_l2058_205856


namespace NUMINAMATH_CALUDE_yacht_weight_excess_excess_weight_l2058_205804

/-- Represents the weight of an animal in sheep equivalents -/
structure AnimalWeight where
  sheep : ℕ

/-- Represents the count of each animal type -/
structure AnimalCounts where
  cows : ℕ
  foxes : ℕ
  zebras : ℕ

/-- Defines the weight equivalents for each animal type -/
def animalWeights : AnimalWeight :=
  { sheep := 1 }

def cowWeight : AnimalWeight :=
  { sheep := 3 }

def foxWeight : AnimalWeight :=
  { sheep := 2 }

def zebraWeight : AnimalWeight :=
  { sheep := 5 }

/-- Calculates the total weight of all animals in sheep equivalents -/
def totalWeight (counts : AnimalCounts) : ℕ :=
  counts.cows * cowWeight.sheep +
  counts.foxes * foxWeight.sheep +
  counts.zebras * zebraWeight.sheep

/-- The theorem to be proved -/
theorem yacht_weight_excess (counts : AnimalCounts)
  (h1 : counts.cows = 20)
  (h2 : counts.foxes = 15)
  (h3 : counts.zebras = 3 * counts.foxes)
  : totalWeight counts = 315 := by
  sorry

/-- The main theorem stating the excess weight -/
theorem excess_weight (counts : AnimalCounts)
  (h1 : counts.cows = 20)
  (h2 : counts.foxes = 15)
  (h3 : counts.zebras = 3 * counts.foxes)
  : totalWeight counts - 300 = 15 := by
  sorry

end NUMINAMATH_CALUDE_yacht_weight_excess_excess_weight_l2058_205804


namespace NUMINAMATH_CALUDE_floor_plus_double_eq_15_4_l2058_205835

theorem floor_plus_double_eq_15_4 :
  ∃! r : ℝ, ⌊r⌋ + 2 * r = 15.4 ∧ r = 5.2 :=
by sorry

end NUMINAMATH_CALUDE_floor_plus_double_eq_15_4_l2058_205835


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l2058_205883

theorem difference_of_squares_example : (17 + 10)^2 - (17 - 10)^2 = 680 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l2058_205883


namespace NUMINAMATH_CALUDE_total_card_units_traded_l2058_205822

/-- Represents the types of trading cards -/
inductive CardType
| A
| B
| C

/-- Represents a trading round -/
structure TradingRound where
  padmaInitial : CardType → ℕ
  robertInitial : CardType → ℕ
  padmaTrades : CardType → ℕ
  robertTrades : CardType → ℕ
  ratios : CardType → CardType → ℚ

/-- Calculates the total card units traded in a round -/
def cardUnitsTradedInRound (round : TradingRound) : ℚ :=
  sorry

/-- The three trading rounds -/
def round1 : TradingRound := {
  padmaInitial := λ | CardType.A => 50 | CardType.B => 45 | CardType.C => 30,
  robertInitial := λ _ => 0,  -- Not specified in the problem
  padmaTrades := λ | CardType.A => 5 | CardType.B => 12 | CardType.C => 0,
  robertTrades := λ | CardType.C => 20 | _ => 0,
  ratios := λ | CardType.A, CardType.C => 2 | CardType.B, CardType.C => 3/2 | _, _ => 1
}

def round2 : TradingRound := {
  padmaInitial := λ _ => 0,  -- Not relevant for this round
  robertInitial := λ | CardType.A => 60 | CardType.B => 50 | CardType.C => 40,
  robertTrades := λ | CardType.A => 10 | CardType.B => 3 | CardType.C => 15,
  padmaTrades := λ | CardType.A => 8 | CardType.B => 18 | CardType.C => 0,
  ratios := λ | CardType.A, CardType.B => 3/2 | CardType.B, CardType.C => 2 | CardType.C, CardType.A => 1 | _, _ => 1
}

def round3 : TradingRound := {
  padmaInitial := λ _ => 0,  -- Not relevant for this round
  robertInitial := λ _ => 0,  -- Not relevant for this round
  padmaTrades := λ | CardType.B => 15 | CardType.C => 10 | CardType.A => 0,
  robertTrades := λ | CardType.A => 12 | _ => 0,
  ratios := λ | CardType.A, CardType.B => 5/4 | CardType.C, CardType.A => 6/5 | _, _ => 1
}

/-- The main theorem stating the total card units traded -/
theorem total_card_units_traded :
  cardUnitsTradedInRound round1 + cardUnitsTradedInRound round2 + cardUnitsTradedInRound round3 = 94.75 := by
  sorry

end NUMINAMATH_CALUDE_total_card_units_traded_l2058_205822


namespace NUMINAMATH_CALUDE_triangle_max_area_l2058_205800

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions and theorem
theorem triangle_max_area (t : Triangle) 
  (h1 : Real.sin t.A + Real.sqrt 2 * Real.sin t.B = 2 * Real.sin t.C)
  (h2 : t.b = 3) :
  ∃ (max_area : ℝ), max_area = (9 + 3 * Real.sqrt 3) / 4 ∧ 
    ∀ (area : ℝ), area ≤ max_area := by
  sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2058_205800


namespace NUMINAMATH_CALUDE_quadrilateral_area_relations_integer_areas_perfect_square_product_l2058_205885

/-- Given a convex quadrilateral ABCD with diagonals intersecting at point P,
    S_ABP, S_BCP, S_CDP, and S_ADP are the areas of triangles ABP, BCP, CDP, and ADP respectively. -/
def QuadrilateralAreas (S_ABP S_BCP S_CDP S_ADP : ℝ) : Prop :=
  S_ABP > 0 ∧ S_BCP > 0 ∧ S_CDP > 0 ∧ S_ADP > 0

theorem quadrilateral_area_relations
  (S_ABP S_BCP S_CDP S_ADP : ℝ)
  (h : QuadrilateralAreas S_ABP S_BCP S_CDP S_ADP) :
  S_ADP = (S_ABP * S_CDP) / S_BCP ∧
  S_ABP * S_BCP * S_CDP * S_ADP = (S_ADP * S_BCP)^2 := by
  sorry

/-- If the areas of the four triangles are integers, their product is a perfect square. -/
theorem integer_areas_perfect_square_product
  (S_ABP S_BCP S_CDP S_ADP : ℤ)
  (h : QuadrilateralAreas (S_ABP : ℝ) (S_BCP : ℝ) (S_CDP : ℝ) (S_ADP : ℝ)) :
  ∃ (n : ℤ), S_ABP * S_BCP * S_CDP * S_ADP = n^2 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_relations_integer_areas_perfect_square_product_l2058_205885


namespace NUMINAMATH_CALUDE_infinite_series_sum_l2058_205836

/-- Given positive real numbers c and d where c > d, the sum of the infinite series
    1/(cd) + 1/(c(3c-d)) + 1/((3c-d)(5c-2d)) + 1/((5c-2d)(7c-3d)) + ...
    is equal to 1/((c-d)d). -/
theorem infinite_series_sum (c d : ℝ) (hc : c > 0) (hd : d > 0) (h : c > d) :
  let series := fun n : ℕ => 1 / ((2 * n - 1) * c - (n - 1) * d) / ((2 * n + 1) * c - n * d)
  ∑' n, series n = 1 / ((c - d) * d) := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l2058_205836


namespace NUMINAMATH_CALUDE_max_value_sum_of_fractions_l2058_205848

theorem max_value_sum_of_fractions (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  (a * b) / (a + b) + (a * c) / (a + c) + (b * c) / (b + c) ≤ 9 / 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_fractions_l2058_205848


namespace NUMINAMATH_CALUDE_rectangular_solid_depth_l2058_205803

/-- The surface area of a rectangular solid given its dimensions -/
def surface_area (length width depth : ℝ) : ℝ :=
  2 * (length * width + length * depth + width * depth)

/-- Theorem: A rectangular solid with length 10, width 9, and surface area 408 has depth 6 -/
theorem rectangular_solid_depth :
  ∃ (depth : ℝ), surface_area 10 9 depth = 408 ∧ depth = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_depth_l2058_205803


namespace NUMINAMATH_CALUDE_square_distance_sum_l2058_205808

theorem square_distance_sum (s : Real) (h : s = 4) : 
  let midpoint_distance := 2 * s / 2
  let diagonal_distance := s * Real.sqrt 2
  let side_distance := s
  2 * midpoint_distance + 2 * Real.sqrt (midpoint_distance^2 + (s/2)^2) + diagonal_distance + side_distance = 10 + 4 * Real.sqrt 5 + 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_square_distance_sum_l2058_205808


namespace NUMINAMATH_CALUDE_points_collinear_l2058_205826

/-- Given vectors a and b in a vector space, and points A, B, C, D such that
    AB = a + 2b, BC = -5a + 6b, and CD = 7a - 2b, prove that A, B, and D are collinear. -/
theorem points_collinear 
  {V : Type*} [AddCommGroup V] [Module ℝ V]
  (a b : V) (A B C D : V) 
  (hAB : B - A = a + 2 • b)
  (hBC : C - B = -5 • a + 6 • b)
  (hCD : D - C = 7 • a - 2 • b) :
  ∃ (t : ℝ), D - A = t • (B - A) :=
sorry

end NUMINAMATH_CALUDE_points_collinear_l2058_205826


namespace NUMINAMATH_CALUDE_marble_probability_l2058_205895

/-- The probability of drawing a red, blue, or green marble from a bag -/
theorem marble_probability (red blue green yellow : ℕ) : 
  red = 4 → blue = 3 → green = 2 → yellow = 6 → 
  (red + blue + green : ℚ) / (red + blue + green + yellow) = 0.6 := by
sorry

end NUMINAMATH_CALUDE_marble_probability_l2058_205895


namespace NUMINAMATH_CALUDE_grace_walk_distance_l2058_205824

/-- The number of blocks Grace walked south -/
def blocks_south : ℕ := 4

/-- The number of blocks Grace walked west -/
def blocks_west : ℕ := 8

/-- The length of one block in miles -/
def block_length : ℚ := 1 / 4

/-- The total distance Grace walked in miles -/
def total_distance : ℚ := (blocks_south + blocks_west : ℚ) * block_length

theorem grace_walk_distance :
  total_distance = 3 := by sorry

end NUMINAMATH_CALUDE_grace_walk_distance_l2058_205824


namespace NUMINAMATH_CALUDE_f_is_locally_odd_l2058_205849

/-- Definition of a locally odd function -/
def LocallyOdd (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f (-x) = -f x

/-- The quadratic function we're examining -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x - 4 * a

/-- Theorem: The function f is locally odd for any real a -/
theorem f_is_locally_odd (a : ℝ) : LocallyOdd (f a) := by
  sorry


end NUMINAMATH_CALUDE_f_is_locally_odd_l2058_205849


namespace NUMINAMATH_CALUDE_point_movement_on_number_line_l2058_205823

theorem point_movement_on_number_line :
  ∀ (a b c : ℝ),
    b = a - 3 →
    c = b + 5 →
    c = 1 →
    a = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_movement_on_number_line_l2058_205823


namespace NUMINAMATH_CALUDE_f_2007_equals_negative_two_l2058_205809

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_2007_equals_negative_two (f : ℝ → ℝ) 
  (h1 : isEven f) 
  (h2 : ∀ x, f (2 + x) = f (2 - x)) 
  (h3 : f (-3) = -2) : 
  f 2007 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_2007_equals_negative_two_l2058_205809


namespace NUMINAMATH_CALUDE_parallel_vectors_fraction_l2058_205872

theorem parallel_vectors_fraction (x : ℝ) :
  let a : ℝ × ℝ := (Real.sin x, 3/2)
  let b : ℝ × ℝ := (Real.cos x, -1)
  (a.1 * b.2 = a.2 * b.1) →
  (2 * Real.sin x - Real.cos x) / (4 * Real.sin x + 3 * Real.cos x) = 4/3 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_fraction_l2058_205872


namespace NUMINAMATH_CALUDE_horner_third_step_value_l2058_205855

def f (x : ℝ) : ℝ := x^5 - 2*x^4 + 3*x^3 - 7*x^2 + 6*x - 3

def horner_step (n : ℕ) (x : ℝ) (coeffs : List ℝ) : ℝ :=
  match n, coeffs with
  | 0, _ => 0
  | n+1, a::rest => a + x * horner_step n x rest
  | _, _ => 0

theorem horner_third_step_value :
  let coeffs := [1, -2, 3, -7, 6, -3]
  let x := 2
  horner_step 3 x coeffs = -1 := by sorry

end NUMINAMATH_CALUDE_horner_third_step_value_l2058_205855


namespace NUMINAMATH_CALUDE_sams_mystery_books_l2058_205890

/-- The number of mystery books Sam bought at the school's book fair -/
def mystery_books : ℕ := sorry

/-- The number of adventure books Sam bought -/
def adventure_books : ℕ := 13

/-- The number of used books Sam bought -/
def used_books : ℕ := 15

/-- The number of new books Sam bought -/
def new_books : ℕ := 15

/-- The total number of books Sam bought -/
def total_books : ℕ := used_books + new_books

theorem sams_mystery_books : 
  mystery_books = total_books - adventure_books ∧ 
  mystery_books = 17 := by sorry

end NUMINAMATH_CALUDE_sams_mystery_books_l2058_205890


namespace NUMINAMATH_CALUDE_swan_population_after_ten_years_l2058_205881

/-- The number of swans after a given number of years, given an initial population and a doubling period. -/
def swan_population (initial_population : ℕ) (doubling_period : ℕ) (years : ℕ) : ℕ :=
  initial_population * (2 ^ (years / doubling_period))

/-- Theorem stating that given an initial population of 15 swans, if the population doubles every 2 years, then after 10 years, the population will be 480 swans. -/
theorem swan_population_after_ten_years :
  swan_population 15 2 10 = 480 := by
  sorry

#eval swan_population 15 2 10

end NUMINAMATH_CALUDE_swan_population_after_ten_years_l2058_205881


namespace NUMINAMATH_CALUDE_arctan_sum_in_triangle_l2058_205865

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (angleC : ℝ)
  (pos_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (pos_angleC : angleC > 0)
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b)

-- State the theorem
theorem arctan_sum_in_triangle (t : Triangle) : 
  Real.arctan (t.a / (t.b + t.c - t.a)) + Real.arctan (t.b / (t.a + t.c - t.b)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_in_triangle_l2058_205865


namespace NUMINAMATH_CALUDE_tyrone_nickels_l2058_205830

/-- Represents the contents of Tyrone's piggy bank -/
structure PiggyBank where
  one_dollar_bills : Nat
  five_dollar_bills : Nat
  quarters : Nat
  dimes : Nat
  nickels : Nat
  pennies : Nat

/-- Calculates the total value in dollars of the contents of the piggy bank -/
def total_value (pb : PiggyBank) : Rat :=
  pb.one_dollar_bills + 
  5 * pb.five_dollar_bills + 
  (1/4) * pb.quarters + 
  (1/10) * pb.dimes + 
  (1/20) * pb.nickels + 
  (1/100) * pb.pennies

/-- Tyrone's piggy bank contents -/
def tyrone_piggy_bank : PiggyBank :=
  { one_dollar_bills := 2
  , five_dollar_bills := 1
  , quarters := 13
  , dimes := 20
  , nickels := 8  -- This is what we want to prove
  , pennies := 35 }

theorem tyrone_nickels : 
  total_value tyrone_piggy_bank = 13 := by sorry

end NUMINAMATH_CALUDE_tyrone_nickels_l2058_205830


namespace NUMINAMATH_CALUDE_points_deducted_for_incorrect_l2058_205801

def test_questions : ℕ := 30
def correct_answer_points : ℕ := 20
def maria_final_score : ℕ := 325
def maria_correct_answers : ℕ := 19

theorem points_deducted_for_incorrect (deducted_points : ℕ) : 
  (maria_correct_answers * correct_answer_points) - 
  ((test_questions - maria_correct_answers) * deducted_points) = 
  maria_final_score → 
  deducted_points = 5 := by
sorry

end NUMINAMATH_CALUDE_points_deducted_for_incorrect_l2058_205801


namespace NUMINAMATH_CALUDE_four_digit_sum_problem_l2058_205870

theorem four_digit_sum_problem (a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 →
  6 * (a + b + c + d) * 1111 = 73326 →
  ({a, b, c, d} : Finset ℕ) = {1, 2, 3, 5} :=
by sorry

end NUMINAMATH_CALUDE_four_digit_sum_problem_l2058_205870


namespace NUMINAMATH_CALUDE_tan_alpha_2_implies_ratio_3_l2058_205877

theorem tan_alpha_2_implies_ratio_3 (α : Real) (h : Real.tan α = 2) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_2_implies_ratio_3_l2058_205877


namespace NUMINAMATH_CALUDE_box_height_is_55cm_l2058_205897

/-- The height of the box Bob needs to reach the light fixture -/
def box_height (ceiling_height light_fixture_distance bob_height bob_reach : ℝ) : ℝ :=
  ceiling_height - light_fixture_distance - (bob_height + bob_reach)

/-- Theorem stating the height of the box Bob needs -/
theorem box_height_is_55cm :
  let ceiling_height : ℝ := 300
  let light_fixture_distance : ℝ := 15
  let bob_height : ℝ := 180
  let bob_reach : ℝ := 50
  box_height ceiling_height light_fixture_distance bob_height bob_reach = 55 := by
  sorry

#eval box_height 300 15 180 50

end NUMINAMATH_CALUDE_box_height_is_55cm_l2058_205897


namespace NUMINAMATH_CALUDE_andy_socks_difference_l2058_205862

theorem andy_socks_difference (black_socks : ℕ) (white_socks : ℕ) : 
  black_socks = 6 →
  white_socks = 4 * black_socks →
  (white_socks / 2) - black_socks = 6 := by
  sorry

end NUMINAMATH_CALUDE_andy_socks_difference_l2058_205862


namespace NUMINAMATH_CALUDE_probability_at_least_one_history_or_geography_l2058_205866

def total_outcomes : ℕ := Nat.choose 5 2

def favorable_outcomes : ℕ := Nat.choose 2 1 * Nat.choose 3 1 + Nat.choose 2 2

theorem probability_at_least_one_history_or_geography :
  (favorable_outcomes : ℚ) / total_outcomes = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_history_or_geography_l2058_205866


namespace NUMINAMATH_CALUDE_max_distance_is_25km_l2058_205868

def car_position (t : ℝ) : ℝ := 40 * t

def motorcycle_position (t : ℝ) : ℝ := 16 * t^2 + 9

def distance (t : ℝ) : ℝ := |motorcycle_position t - car_position t|

theorem max_distance_is_25km :
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 2 ∧
  ∀ s : ℝ, 0 ≤ s ∧ s ≤ 2 → distance t ≥ distance s ∧
  distance t = 25 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_is_25km_l2058_205868


namespace NUMINAMATH_CALUDE_r_fraction_of_total_l2058_205813

theorem r_fraction_of_total (total : ℚ) (r_amount : ℚ) 
  (h1 : total = 4000)
  (h2 : r_amount = 1600) :
  r_amount / total = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_r_fraction_of_total_l2058_205813


namespace NUMINAMATH_CALUDE_initial_marbles_equals_sum_l2058_205811

/-- The number of marbles Connie initially had -/
def initial_marbles : ℕ := sorry

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℕ := 73

/-- The number of marbles Connie has left -/
def marbles_left : ℕ := 70

/-- Theorem stating that the initial number of marbles equals the sum of marbles given away and marbles left -/
theorem initial_marbles_equals_sum : initial_marbles = marbles_given + marbles_left := by sorry

end NUMINAMATH_CALUDE_initial_marbles_equals_sum_l2058_205811


namespace NUMINAMATH_CALUDE_multiply_and_distribute_l2058_205847

theorem multiply_and_distribute (a b : ℝ) : -a * b * (-b + 1) = a * b^2 - a * b := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_distribute_l2058_205847


namespace NUMINAMATH_CALUDE_sum_first_15_odd_from_5_l2058_205869

/-- The sum of the first n odd positive integers starting from a given odd number -/
def sumOddIntegers (start : ℕ) (n : ℕ) : ℕ :=
  let lastTerm := start + 2 * (n - 1)
  (start + lastTerm) * n / 2

/-- The proposition that the sum of the first 15 odd positive integers starting from 5 is 315 -/
theorem sum_first_15_odd_from_5 : sumOddIntegers 5 15 = 315 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_15_odd_from_5_l2058_205869


namespace NUMINAMATH_CALUDE_power_inequality_l2058_205816

theorem power_inequality (a x y : ℝ) (ha : 0 < a) (ha1 : a < 1) (h : a^x < a^y) : x^3 > y^3 := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l2058_205816


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_of_cubes_l2058_205857

/-- Represents an arithmetic sequence with n+1 terms, first term y, and common difference 4 -/
def arithmetic_sequence (y : ℤ) (n : ℕ) : List ℤ :=
  List.range (n + 1) |>.map (fun i => y + 4 * i)

/-- The sum of cubes of all terms in the sequence -/
def sum_of_cubes (seq : List ℤ) : ℤ :=
  seq.map (fun x => x^3) |>.sum

theorem arithmetic_sequence_sum_of_cubes (y : ℤ) (n : ℕ) :
  n > 6 →
  sum_of_cubes (arithmetic_sequence y n) = -5832 →
  n = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_of_cubes_l2058_205857


namespace NUMINAMATH_CALUDE_grains_in_gray_areas_l2058_205807

/-- Given two circles with equal total grains, prove that the sum of their non-overlapping parts is 61 grains -/
theorem grains_in_gray_areas (total_circle1 total_circle2 overlap : ℕ) 
  (h1 : total_circle1 = 110)
  (h2 : total_circle2 = 87)
  (h3 : overlap = 68)
  (h4 : total_circle1 = total_circle2) : 
  (total_circle1 - overlap) + (total_circle2 - overlap) = 61 := by
  sorry

#check grains_in_gray_areas

end NUMINAMATH_CALUDE_grains_in_gray_areas_l2058_205807


namespace NUMINAMATH_CALUDE_common_points_on_line_l2058_205898

-- Define the circles and line
def circle1 (a : ℝ) (x y : ℝ) : Prop := x^2 + (y - 1)^2 = a^2 ∧ a > 0
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4
def line (x y : ℝ) : Prop := y = 2*x

-- Define the theorem
theorem common_points_on_line (a : ℝ) : 
  (∀ x y : ℝ, circle1 a x y ∧ circle2 x y → line x y) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_common_points_on_line_l2058_205898


namespace NUMINAMATH_CALUDE_category_d_cost_after_discount_l2058_205843

/-- Represents the cost and discount information for a category of items --/
structure Category where
  percentage : Real
  discount_rate : Real

/-- Calculates the cost of items in a category after applying the discount --/
def cost_after_discount (total_cost : Real) (category : Category) : Real :=
  let cost_before_discount := total_cost * category.percentage
  cost_before_discount * (1 - category.discount_rate)

/-- Theorem stating that the cost of category D items after discount is 562.5 --/
theorem category_d_cost_after_discount (total_cost : Real) (category_d : Category) :
  total_cost = 2500 →
  category_d.percentage = 0.25 →
  category_d.discount_rate = 0.10 →
  cost_after_discount total_cost category_d = 562.5 := by
  sorry

#check category_d_cost_after_discount

end NUMINAMATH_CALUDE_category_d_cost_after_discount_l2058_205843


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2058_205864

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p ^ 2 + 9 * p - 21 = 0) → 
  (3 * q ^ 2 + 9 * q - 21 = 0) → 
  (3 * p - 4) * (6 * q - 8) = 14 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2058_205864


namespace NUMINAMATH_CALUDE_mailman_delivery_l2058_205829

/-- Represents the different types of mail delivered by the mailman -/
structure MailDelivery where
  junkMail : ℕ
  magazines : ℕ
  newspapers : ℕ
  bills : ℕ
  postcards : ℕ

/-- Calculates the total number of mail pieces delivered -/
def totalMail (delivery : MailDelivery) : ℕ :=
  delivery.junkMail + delivery.magazines + delivery.newspapers + delivery.bills + delivery.postcards

/-- Theorem stating that the total mail delivered is 20 pieces -/
theorem mailman_delivery :
  ∃ (delivery : MailDelivery),
    delivery.junkMail = 6 ∧
    delivery.magazines = 5 ∧
    delivery.newspapers = 3 ∧
    delivery.bills = 4 ∧
    delivery.postcards = 2 ∧
    totalMail delivery = 20 := by
  sorry

end NUMINAMATH_CALUDE_mailman_delivery_l2058_205829


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_17_l2058_205867

theorem smallest_four_digit_multiple_of_17 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 17 ∣ n → 1003 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_17_l2058_205867


namespace NUMINAMATH_CALUDE_neg_three_at_neg_two_l2058_205834

-- Define the "@" operation
def at_op (x y : ℤ) : ℤ := x * y - y

-- Theorem statement
theorem neg_three_at_neg_two : at_op (-3) (-2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_neg_three_at_neg_two_l2058_205834


namespace NUMINAMATH_CALUDE_units_digit_of_n_l2058_205802

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the problem statement
theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 21^6) (h2 : unitsDigit m = 7) :
  unitsDigit n = 3 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l2058_205802


namespace NUMINAMATH_CALUDE_fruit_basket_combinations_l2058_205858

/-- The number of ways to choose apples for a fruit basket -/
def apple_choices : ℕ := 3

/-- The number of ways to choose oranges for a fruit basket -/
def orange_choices : ℕ := 8

/-- The total number of fruit basket combinations -/
def total_combinations : ℕ := apple_choices * orange_choices

/-- Theorem stating the number of possible fruit baskets -/
theorem fruit_basket_combinations :
  total_combinations = 36 :=
sorry

end NUMINAMATH_CALUDE_fruit_basket_combinations_l2058_205858


namespace NUMINAMATH_CALUDE_village_population_is_100_l2058_205891

/-- Represents the number of people in a youth summer village with specific characteristics. -/
def village_population (total : ℕ) (not_working : ℕ) (with_families : ℕ) (shower_singers : ℕ) (working_no_family_singers : ℕ) : Prop :=
  not_working = 50 ∧
  with_families = 25 ∧
  shower_singers = 75 ∧
  working_no_family_singers = 50 ∧
  total = not_working + with_families + shower_singers - working_no_family_singers

theorem village_population_is_100 :
  ∃ (total : ℕ), village_population total 50 25 75 50 ∧ total = 100 := by
  sorry

end NUMINAMATH_CALUDE_village_population_is_100_l2058_205891


namespace NUMINAMATH_CALUDE_solve_equation_l2058_205814

theorem solve_equation (x : ℚ) : (3 * x + 5) / 7 = 13 → x = 86 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2058_205814


namespace NUMINAMATH_CALUDE_carls_flowerbed_area_l2058_205887

/-- Represents a rectangular flowerbed with fencing --/
structure Flowerbed where
  short_posts : ℕ  -- Number of posts on the shorter side (including corners)
  long_posts : ℕ   -- Number of posts on the longer side (including corners)
  post_spacing : ℕ -- Spacing between posts in yards

/-- Calculates the area of the flowerbed --/
def Flowerbed.area (fb : Flowerbed) : ℕ :=
  (fb.short_posts - 1) * (fb.long_posts - 1) * fb.post_spacing * fb.post_spacing

/-- Theorem stating the area of Carl's flowerbed --/
theorem carls_flowerbed_area :
  ∃ fb : Flowerbed,
    fb.short_posts + fb.long_posts = 13 ∧
    fb.long_posts = 3 * fb.short_posts - 2 ∧
    fb.post_spacing = 3 ∧
    fb.area = 144 := by
  sorry

end NUMINAMATH_CALUDE_carls_flowerbed_area_l2058_205887


namespace NUMINAMATH_CALUDE_largest_multiple_of_11_below_negative_85_l2058_205820

theorem largest_multiple_of_11_below_negative_85 :
  ∀ n : ℤ, n % 11 = 0 → n < -85 → n ≤ -88 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_11_below_negative_85_l2058_205820


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l2058_205837

theorem opposite_of_negative_fraction :
  -(-(1 / 2023)) = 1 / 2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l2058_205837


namespace NUMINAMATH_CALUDE_ratio_p_to_r_l2058_205884

theorem ratio_p_to_r (p q r s : ℚ) 
  (h1 : p / q = 5 / 4)
  (h2 : r / s = 4 / 3)
  (h3 : s / q = 1 / 5) :
  p / r = 75 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_p_to_r_l2058_205884


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_equals_negative_one_l2058_205875

/-- Given a function f(x) = ax^2 + x^2 that reaches an extreme value at x = -2,
    prove that a = -1 --/
theorem extreme_value_implies_a_equals_negative_one (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + x^2
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-2 - ε) (-2 + ε), f x ≤ f (-2) ∨ f x ≥ f (-2)) →
  a = -1 :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_equals_negative_one_l2058_205875


namespace NUMINAMATH_CALUDE_solution_replacement_fraction_l2058_205861

theorem solution_replacement_fraction (Q : ℝ) (h : Q > 0) :
  let initial_conc : ℝ := 0.70
  let replacement_conc : ℝ := 0.25
  let new_conc : ℝ := 0.35
  let x : ℝ := (new_conc * Q - initial_conc * Q) / (replacement_conc * Q - initial_conc * Q)
  x = 7 / 9 := by
sorry

end NUMINAMATH_CALUDE_solution_replacement_fraction_l2058_205861


namespace NUMINAMATH_CALUDE_modular_inverse_34_mod_35_l2058_205841

theorem modular_inverse_34_mod_35 : ∃ x : ℕ, x ≤ 34 ∧ (34 * x) % 35 = 1 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_34_mod_35_l2058_205841


namespace NUMINAMATH_CALUDE_ben_game_probability_l2058_205838

theorem ben_game_probability (p_lose p_tie : ℚ) 
  (h_lose : p_lose = 5 / 11)
  (h_tie : p_tie = 1 / 11)
  (h_total : p_lose + p_tie + (1 - p_lose - p_tie) = 1) :
  1 - p_lose - p_tie = 5 / 11 := by
sorry

end NUMINAMATH_CALUDE_ben_game_probability_l2058_205838


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l2058_205819

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem fifteenth_term_of_sequence (a₁ a₂ : ℚ) (h₁ : a₁ = 12) (h₂ : a₂ = 4) :
  geometric_sequence a₁ (a₂ / a₁) 15 = 12 / 4782969 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l2058_205819


namespace NUMINAMATH_CALUDE_joes_pocket_money_l2058_205880

theorem joes_pocket_money (initial_money : ℚ) : 
  (initial_money * (1 - (1/9 + 2/5)) = 220) → initial_money = 450 := by
  sorry

end NUMINAMATH_CALUDE_joes_pocket_money_l2058_205880


namespace NUMINAMATH_CALUDE_charging_time_is_112_5_l2058_205806

/-- Represents the charging time for each device type -/
structure ChargingTimes where
  smartphone : ℝ
  tablet : ℝ
  laptop : ℝ

/-- Represents the charging percentages for each device -/
structure ChargingPercentages where
  smartphone : ℝ
  tablet : ℝ
  laptop : ℝ

/-- Calculates the total charging time given the full charging times and charging percentages -/
def totalChargingTime (times : ChargingTimes) (percentages : ChargingPercentages) : ℝ :=
  times.tablet * percentages.tablet +
  times.smartphone * percentages.smartphone +
  times.laptop * percentages.laptop

/-- Theorem stating that the total charging time is 112.5 minutes -/
theorem charging_time_is_112_5 (times : ChargingTimes) (percentages : ChargingPercentages) :
  times.smartphone = 26 →
  times.tablet = 53 →
  times.laptop = 80 →
  percentages.smartphone = 0.75 →
  percentages.tablet = 1 →
  percentages.laptop = 0.5 →
  totalChargingTime times percentages = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_charging_time_is_112_5_l2058_205806


namespace NUMINAMATH_CALUDE_polynomial_roots_l2058_205873

theorem polynomial_roots (p : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    x^4 + 2*p*x^3 - x^2 + 2*p*x + 1 = 0 ∧ 
    y^4 + 2*p*y^3 - y^2 + 2*p*y + 1 = 0) ↔ 
  -3/4 ≤ p ∧ p ≤ -1/4 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l2058_205873


namespace NUMINAMATH_CALUDE_cinnamon_amount_l2058_205805

/-- The amount of nutmeg used in tablespoons -/
def nutmeg : ℝ := 0.5

/-- The difference in tablespoons between cinnamon and nutmeg -/
def difference : ℝ := 0.17

/-- The amount of cinnamon used in tablespoons -/
def cinnamon : ℝ := nutmeg + difference

theorem cinnamon_amount : cinnamon = 0.67 := by
  sorry

end NUMINAMATH_CALUDE_cinnamon_amount_l2058_205805


namespace NUMINAMATH_CALUDE_count_valid_three_digit_numbers_l2058_205852

/-- The count of three-digit numbers with specific exclusions -/
def valid_three_digit_numbers : ℕ :=
  let total_three_digit_numbers := 900
  let numbers_with_two_same_nonadjacent_digits := 81
  let numbers_with_increasing_digits := 28
  total_three_digit_numbers - (numbers_with_two_same_nonadjacent_digits + numbers_with_increasing_digits)

/-- Theorem stating the count of valid three-digit numbers -/
theorem count_valid_three_digit_numbers :
  valid_three_digit_numbers = 791 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_three_digit_numbers_l2058_205852


namespace NUMINAMATH_CALUDE_jimmy_passing_points_l2058_205818

def points_per_exam : ℕ := 20
def number_of_exams : ℕ := 3
def points_lost_for_behavior : ℕ := 5
def additional_points_can_lose : ℕ := 5

def points_to_pass : ℕ := 50

theorem jimmy_passing_points :
  points_to_pass = 
    points_per_exam * number_of_exams - 
    points_lost_for_behavior - 
    additional_points_can_lose :=
by
  sorry

end NUMINAMATH_CALUDE_jimmy_passing_points_l2058_205818
