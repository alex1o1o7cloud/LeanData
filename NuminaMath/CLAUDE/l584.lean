import Mathlib

namespace NUMINAMATH_CALUDE_ice_cream_cost_calculation_l584_58466

/-- Calculates the cost of each ice-cream cup given the order details and total amount paid --/
theorem ice_cream_cost_calculation
  (chapati_count : ℕ)
  (rice_count : ℕ)
  (vegetable_count : ℕ)
  (ice_cream_count : ℕ)
  (chapati_cost : ℕ)
  (rice_cost : ℕ)
  (vegetable_cost : ℕ)
  (total_paid : ℕ)
  (h1 : chapati_count = 16)
  (h2 : rice_count = 5)
  (h3 : vegetable_count = 7)
  (h4 : ice_cream_count = 6)
  (h5 : chapati_cost = 6)
  (h6 : rice_cost = 45)
  (h7 : vegetable_cost = 70)
  (h8 : total_paid = 961) :
  (total_paid - (chapati_count * chapati_cost + rice_count * rice_cost + vegetable_count * vegetable_cost)) / ice_cream_count = 25 := by
  sorry


end NUMINAMATH_CALUDE_ice_cream_cost_calculation_l584_58466


namespace NUMINAMATH_CALUDE_geni_phone_expense_l584_58452

/-- Represents a telephone plan with fixed fee, free minutes, and per-minute rate -/
structure TelephonePlan where
  fixedFee : ℝ
  freeMinutes : ℕ
  ratePerMinute : ℝ

/-- Calculates the bill for a given usage in minutes -/
def calculateBill (plan : TelephonePlan) (usageMinutes : ℕ) : ℝ :=
  plan.fixedFee + max 0 (usageMinutes - plan.freeMinutes) * plan.ratePerMinute

/-- Converts hours and minutes to total minutes -/
def toMinutes (hours : ℕ) (minutes : ℕ) : ℕ := hours * 60 + minutes

theorem geni_phone_expense :
  let plan : TelephonePlan := { fixedFee := 18, freeMinutes := 600, ratePerMinute := 0.03 }
  let januaryUsage : ℕ := toMinutes 15 17
  let februaryUsage : ℕ := toMinutes 9 55
  calculateBill plan januaryUsage + calculateBill plan februaryUsage = 45.51 := by
  sorry

end NUMINAMATH_CALUDE_geni_phone_expense_l584_58452


namespace NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l584_58425

open Real

theorem function_inequality_implies_parameter_bound 
  (f g : ℝ → ℝ) 
  (h : ∀ x > 0, f x = 2 * x * log x ∧ g x = -x^2 + a * x - 3) 
  (h2 : ∀ x > 0, f x > g x) : 
  a < 4 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l584_58425


namespace NUMINAMATH_CALUDE_divisibility_properties_l584_58463

theorem divisibility_properties (a : ℤ) : 
  (2 ∣ (a^2 - a)) ∧ (3 ∣ (a^3 - a)) := by sorry

end NUMINAMATH_CALUDE_divisibility_properties_l584_58463


namespace NUMINAMATH_CALUDE_kamals_math_marks_l584_58415

def english_marks : ℕ := 66
def physics_marks : ℕ := 77
def chemistry_marks : ℕ := 62
def biology_marks : ℕ := 75
def average_marks : ℚ := 69
def total_subjects : ℕ := 5

theorem kamals_math_marks :
  let total_marks := average_marks * total_subjects
  let known_marks_sum := english_marks + physics_marks + chemistry_marks + biology_marks
  let math_marks := total_marks - known_marks_sum
  math_marks = 65 := by sorry

end NUMINAMATH_CALUDE_kamals_math_marks_l584_58415


namespace NUMINAMATH_CALUDE_chef_manager_wage_difference_l584_58456

/-- Represents the hourly wages at Joe's Steakhouse -/
structure SteakhouseWages where
  manager : ℝ
  dishwasher : ℝ
  chef : ℝ

/-- The conditions for wages at Joe's Steakhouse -/
def wage_conditions (w : SteakhouseWages) : Prop :=
  w.manager = 8.50 ∧
  w.dishwasher = w.manager / 2 ∧
  w.chef = w.dishwasher * 1.22

theorem chef_manager_wage_difference (w : SteakhouseWages) 
  (h : wage_conditions w) : w.manager - w.chef = 3.315 := by
  sorry

#check chef_manager_wage_difference

end NUMINAMATH_CALUDE_chef_manager_wage_difference_l584_58456


namespace NUMINAMATH_CALUDE_total_lives_after_third_level_l584_58439

/-- Game rules for calculating lives --/
def game_lives : ℕ → ℕ :=
  let initial_lives := 2
  let first_level_gain := 6 / 2
  let second_level_gain := 11 - 3
  let third_level_multiplier := 2
  fun level =>
    if level = 0 then
      initial_lives
    else if level = 1 then
      initial_lives + first_level_gain
    else if level = 2 then
      initial_lives + first_level_gain + second_level_gain
    else
      initial_lives + first_level_gain + second_level_gain +
      (first_level_gain + second_level_gain) * third_level_multiplier

/-- Theorem stating the total number of lives after completing the third level --/
theorem total_lives_after_third_level :
  game_lives 3 = 35 := by sorry

end NUMINAMATH_CALUDE_total_lives_after_third_level_l584_58439


namespace NUMINAMATH_CALUDE_converse_and_inverse_false_l584_58453

-- Define the types
def Quadrilateral : Type := sorry
def Rhombus : Type := sorry
def Parallelogram : Type := sorry

-- Define the properties
def is_rhombus : Quadrilateral → Prop := sorry
def is_parallelogram : Quadrilateral → Prop := sorry

-- Given statement
axiom rhombus_is_parallelogram : ∀ q : Quadrilateral, is_rhombus q → is_parallelogram q

-- Theorem to prove
theorem converse_and_inverse_false : 
  (∃ q : Quadrilateral, is_parallelogram q ∧ ¬is_rhombus q) ∧ 
  (∃ q : Quadrilateral, ¬is_rhombus q ∧ is_parallelogram q) := by
  sorry

end NUMINAMATH_CALUDE_converse_and_inverse_false_l584_58453


namespace NUMINAMATH_CALUDE_leo_marbles_count_l584_58471

/-- The number of marbles in each pack -/
def marbles_per_pack : ℕ := 10

/-- The fraction of packs given to Manny -/
def manny_fraction : ℚ := 1/4

/-- The fraction of packs given to Neil -/
def neil_fraction : ℚ := 1/8

/-- The number of packs Leo kept for himself -/
def leo_packs : ℕ := 25

/-- The total number of packs Leo had initially -/
def total_packs : ℕ := 40

/-- The total number of marbles Leo had initially -/
def total_marbles : ℕ := total_packs * marbles_per_pack

theorem leo_marbles_count :
  manny_fraction * total_packs + neil_fraction * total_packs + leo_packs = total_packs ∧
  total_marbles = 400 :=
sorry

end NUMINAMATH_CALUDE_leo_marbles_count_l584_58471


namespace NUMINAMATH_CALUDE_range_of_a_l584_58428

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 1

-- State the theorem
theorem range_of_a :
  (∀ x ∈ Set.Ioo 0 2, f a x ≥ 0) ↔ a ∈ Set.Iic 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l584_58428


namespace NUMINAMATH_CALUDE_f_value_at_2_l584_58454

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem f_value_at_2 (a b : ℝ) (h : f a b (-2) = 10) : f a b 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_2_l584_58454


namespace NUMINAMATH_CALUDE_chicken_bucket_price_l584_58404

/-- Represents the price of a chicken bucket with sides -/
def bucket_price (people_per_bucket : ℕ) (total_people : ℕ) (total_cost : ℕ) : ℚ :=
  total_cost / (total_people / people_per_bucket)

/-- Proves that the price of each chicken bucket with sides is $12 -/
theorem chicken_bucket_price :
  bucket_price 6 36 72 = 12 := by
  sorry

end NUMINAMATH_CALUDE_chicken_bucket_price_l584_58404


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_neg_two_l584_58443

theorem fraction_zero_implies_x_neg_two (x : ℝ) :
  (abs x - 2) / (x^2 - 4*x + 4) = 0 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_neg_two_l584_58443


namespace NUMINAMATH_CALUDE_total_flour_used_l584_58491

-- Define the amount of wheat flour used
def wheat_flour : ℝ := 0.2

-- Define the amount of white flour used
def white_flour : ℝ := 0.1

-- Theorem stating the total amount of flour used
theorem total_flour_used : wheat_flour + white_flour = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_total_flour_used_l584_58491


namespace NUMINAMATH_CALUDE_hungarian_deck_probabilities_l584_58495

/-- Represents a Hungarian deck of cards -/
structure HungarianDeck :=
  (cards : Finset (Fin 32))
  (suits : Fin 4)
  (cardsPerSuit : Fin 8)

/-- Calculates the probability of drawing at least one ace given three cards of different suits -/
def probAceGivenDifferentSuits (deck : HungarianDeck) : ℚ :=
  169 / 512

/-- Calculates the probability of drawing at least one ace when drawing three cards -/
def probAceThreeCards (deck : HungarianDeck) : ℚ :=
  421 / 1240

/-- Calculates the probability of drawing three cards of different suits with at least one ace -/
def probDifferentSuitsWithAce (deck : HungarianDeck) : ℚ :=
  169 / 1240

/-- Main theorem stating the probabilities for the given scenarios -/
theorem hungarian_deck_probabilities (deck : HungarianDeck) :
  (probAceGivenDifferentSuits deck = 169 / 512) ∧
  (probAceThreeCards deck = 421 / 1240) ∧
  (probDifferentSuitsWithAce deck = 169 / 1240) :=
sorry

end NUMINAMATH_CALUDE_hungarian_deck_probabilities_l584_58495


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_half_l584_58474

theorem reciprocal_of_negative_half : ((-1/2)⁻¹ : ℚ) = -2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_half_l584_58474


namespace NUMINAMATH_CALUDE_tan_one_condition_l584_58451

theorem tan_one_condition (x : Real) : 
  (∃ k : Int, x = (k * Real.pi) / 4) ∧ 
  (∃ y : Real, (∃ m : Int, y = (m * Real.pi) / 4) ∧ Real.tan y = 1) ∧ 
  (∃ z : Real, (∃ n : Int, z = (n * Real.pi) / 4) ∧ Real.tan z ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_one_condition_l584_58451


namespace NUMINAMATH_CALUDE_cone_volume_from_half_sector_l584_58444

/-- The volume of a right circular cone formed by rolling up a half-sector of a circle --/
theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) :
  let sector_arc_length := r * π
  let cone_base_radius := sector_arc_length / (2 * π)
  let cone_slant_height := r
  let cone_height := Real.sqrt (cone_slant_height^2 - cone_base_radius^2)
  let cone_volume := (1/3) * π * cone_base_radius^2 * cone_height
  cone_volume = 9 * π * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cone_volume_from_half_sector_l584_58444


namespace NUMINAMATH_CALUDE_ceiling_times_self_equals_210_l584_58462

theorem ceiling_times_self_equals_210 :
  ∃! (x : ℝ), ⌈x⌉ * x = 210 ∧ x = 14 := by sorry

end NUMINAMATH_CALUDE_ceiling_times_self_equals_210_l584_58462


namespace NUMINAMATH_CALUDE_stock_price_calculation_l584_58449

theorem stock_price_calculation (closing_price : ℝ) (percent_increase : ℝ) (opening_price : ℝ) : 
  closing_price = 16 → 
  percent_increase = 6.666666666666665 → 
  closing_price = opening_price * (1 + percent_increase / 100) →
  opening_price = 15 := by
sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l584_58449


namespace NUMINAMATH_CALUDE_abs_x_eq_x_condition_l584_58401

theorem abs_x_eq_x_condition (x : ℝ) :
  (∀ x, |x| = x → x^2 ≥ -x) ∧
  (∃ x, x^2 ≥ -x ∧ |x| ≠ x) := by
  sorry

end NUMINAMATH_CALUDE_abs_x_eq_x_condition_l584_58401


namespace NUMINAMATH_CALUDE_y_coordinate_abs_value_l584_58422

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the problem conditions
def satisfiesConditions (p : Point2D) : Prop :=
  abs p.y = (1/2) * abs p.x ∧ abs p.x = 10

-- State the theorem
theorem y_coordinate_abs_value (p : Point2D) 
  (h : satisfiesConditions p) : abs p.y = 5 := by
  sorry

end NUMINAMATH_CALUDE_y_coordinate_abs_value_l584_58422


namespace NUMINAMATH_CALUDE_hoonjeong_marbles_l584_58498

theorem hoonjeong_marbles :
  ∀ (initial_marbles : ℝ),
    (initial_marbles * (1 - 0.2) * (1 - 0.35) = 130) →
    initial_marbles = 250 :=
by
  sorry

end NUMINAMATH_CALUDE_hoonjeong_marbles_l584_58498


namespace NUMINAMATH_CALUDE_cunningham_lambs_count_l584_58445

/-- Represents the total number of lambs owned by farmer Cunningham -/
def total_lambs : ℕ := 6048

/-- Represents the number of white lambs -/
def white_lambs : ℕ := 193

/-- Represents the number of black lambs -/
def black_lambs : ℕ := 5855

/-- Theorem stating that the total number of lambs is the sum of white and black lambs -/
theorem cunningham_lambs_count : total_lambs = white_lambs + black_lambs := by
  sorry

end NUMINAMATH_CALUDE_cunningham_lambs_count_l584_58445


namespace NUMINAMATH_CALUDE_sqrt_identity_l584_58421

theorem sqrt_identity (a b : ℝ) (h : a > Real.sqrt b) :
  Real.sqrt ((a + Real.sqrt (a^2 - b)) / 2) + Real.sqrt ((a - Real.sqrt (a^2 - b)) / 2) =
  Real.sqrt (a + Real.sqrt b) ∧
  Real.sqrt ((a + Real.sqrt (a^2 - b)) / 2) - Real.sqrt ((a - Real.sqrt (a^2 - b)) / 2) =
  Real.sqrt (a - Real.sqrt b) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_identity_l584_58421


namespace NUMINAMATH_CALUDE_valid_paths_count_l584_58493

/-- Represents a point in the 2D lattice --/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents a move in the lattice --/
inductive Move
  | Right : Move
  | Up : Move
  | Diagonal : Move
  | LongRight : Move

/-- Checks if a sequence of moves is valid (no right angle turns) --/
def isValidPath (path : List Move) : Bool :=
  sorry

/-- Checks if a path leads from (0,0) to (7,5) --/
def leadsTo7_5 (path : List Move) : Bool :=
  sorry

/-- Counts the number of valid paths from (0,0) to (7,5) --/
def countValidPaths : ℕ :=
  sorry

/-- The main theorem stating that the number of valid paths is N --/
theorem valid_paths_count :
  ∃ N : ℕ, countValidPaths = N :=
sorry

end NUMINAMATH_CALUDE_valid_paths_count_l584_58493


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l584_58472

/-- The line x = my + 2 is tangent to the circle x^2 + 2x + y^2 + 2y = 0 if and only if m = 1 or m = -7 -/
theorem line_tangent_to_circle (m : ℝ) : 
  (∀ x y : ℝ, x = m * y + 2 → x^2 + 2*x + y^2 + 2*y ≠ 0) ∨
  (∃ x y : ℝ, x = m * y + 2 ∧ x^2 + 2*x + y^2 + 2*y = 0 ∧
    ∀ x' y' : ℝ, x' = m * y' + 2 → x'^2 + 2*x' + y'^2 + 2*y' ≥ 0) ↔ 
  m = 1 ∨ m = -7 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l584_58472


namespace NUMINAMATH_CALUDE_sum_of_constants_l584_58484

theorem sum_of_constants (a b : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (a + b / x = 1 ↔ x = -1)) ∧
  (∀ x : ℝ, x ≠ 0 → (a + b / x = 7 ↔ x = -3)) →
  a + b = 19 := by
sorry

end NUMINAMATH_CALUDE_sum_of_constants_l584_58484


namespace NUMINAMATH_CALUDE_meaningful_square_root_range_l584_58448

theorem meaningful_square_root_range (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (2 - x)) ↔ x < 2 := by
sorry

end NUMINAMATH_CALUDE_meaningful_square_root_range_l584_58448


namespace NUMINAMATH_CALUDE_expensive_feed_cost_l584_58464

/-- Prove that the cost per pound of the more expensive dog feed is $0.53 --/
theorem expensive_feed_cost 
  (total_mix : ℝ) 
  (target_cost : ℝ) 
  (cheap_feed_cost : ℝ) 
  (cheap_feed_amount : ℝ) 
  (h1 : total_mix = 35)
  (h2 : target_cost = 0.36)
  (h3 : cheap_feed_cost = 0.18)
  (h4 : cheap_feed_amount = 17)
  : ∃ expensive_feed_cost : ℝ, 
    expensive_feed_cost = 0.53 ∧
    expensive_feed_cost * (total_mix - cheap_feed_amount) + 
    cheap_feed_cost * cheap_feed_amount = 
    target_cost * total_mix := by
  sorry

end NUMINAMATH_CALUDE_expensive_feed_cost_l584_58464


namespace NUMINAMATH_CALUDE_sum_of_multiples_l584_58440

def smallest_two_digit_multiple_of_7 : ℕ := sorry

def smallest_three_digit_multiple_of_5 : ℕ := sorry

theorem sum_of_multiples : 
  smallest_two_digit_multiple_of_7 + smallest_three_digit_multiple_of_5 = 114 := by sorry

end NUMINAMATH_CALUDE_sum_of_multiples_l584_58440


namespace NUMINAMATH_CALUDE_power_twenty_equals_r_s_l584_58499

theorem power_twenty_equals_r_s (a b : ℤ) (R S : ℝ) 
  (hR : R = 2^a) (hS : S = 5^b) : 
  R^(2*b) * S^a = 20^(a*b) := by
sorry

end NUMINAMATH_CALUDE_power_twenty_equals_r_s_l584_58499


namespace NUMINAMATH_CALUDE_jessicas_mothers_death_years_jessicas_mothers_death_years_proof_l584_58446

/-- Prove that the number of years passed since Jessica's mother's death is 10 -/
theorem jessicas_mothers_death_years : ℕ :=
  let jessica_current_age : ℕ := 40
  let mother_hypothetical_age : ℕ := 70
  let years_passed : ℕ → Prop := λ x =>
    -- Jessica was half her mother's age when her mother died
    2 * (jessica_current_age - x) = jessica_current_age - x + x ∧
    -- Jessica's mother would be 70 if she were alive now
    jessica_current_age - x + x = mother_hypothetical_age
  10

theorem jessicas_mothers_death_years_proof :
  jessicas_mothers_death_years = 10 := by sorry

end NUMINAMATH_CALUDE_jessicas_mothers_death_years_jessicas_mothers_death_years_proof_l584_58446


namespace NUMINAMATH_CALUDE_complex_power_difference_l584_58477

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) :
  (1 + i)^16 - (1 - i)^16 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l584_58477


namespace NUMINAMATH_CALUDE_dice_probability_l584_58476

/-- The number of dice being rolled -/
def n : ℕ := 7

/-- The number of faces on each die -/
def faces : ℕ := 6

/-- The number of faces showing a number greater than 4 -/
def favorable_faces : ℕ := 2

/-- The number of dice that should show a number greater than 4 -/
def k : ℕ := 3

/-- The probability of rolling a number greater than 4 on a single die -/
def p : ℚ := favorable_faces / faces

/-- The probability of not rolling a number greater than 4 on a single die -/
def q : ℚ := 1 - p

theorem dice_probability :
  (n.choose k * p^k * q^(n-k) : ℚ) = 560/2187 := by sorry

end NUMINAMATH_CALUDE_dice_probability_l584_58476


namespace NUMINAMATH_CALUDE_number_calculation_l584_58413

theorem number_calculation (x : ℚ) : (x - 2) / 13 = 4 → (x - 5) / 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l584_58413


namespace NUMINAMATH_CALUDE_minimum_satisfying_number_l584_58418

def is_multiple_of (a b : ℕ) : Prop := ∃ k, a = b * k

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def satisfies_conditions (A : ℕ) : Prop :=
  A > 0 ∧
  is_multiple_of A 3 ∧
  ¬is_multiple_of A 9 ∧
  is_multiple_of (A + digit_product A) 9

theorem minimum_satisfying_number :
  satisfies_conditions 138 ∧ ∀ A : ℕ, A < 138 → ¬satisfies_conditions A :=
sorry

end NUMINAMATH_CALUDE_minimum_satisfying_number_l584_58418


namespace NUMINAMATH_CALUDE_prob_same_club_is_one_third_l584_58402

/-- The number of clubs -/
def num_clubs : ℕ := 3

/-- The number of students -/
def num_students : ℕ := 2

/-- The probability of two students joining the same club given equal probability of joining any club -/
def prob_same_club : ℚ := 1 / 3

/-- Theorem stating that the probability of two students joining the same club is 1/3 -/
theorem prob_same_club_is_one_third :
  prob_same_club = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_same_club_is_one_third_l584_58402


namespace NUMINAMATH_CALUDE_smallest_common_factor_l584_58417

theorem smallest_common_factor (n : ℕ) : 
  (∀ m : ℕ, m < 43 → gcd (5 * m - 3) (11 * m + 4) = 1) ∧ 
  gcd (5 * 43 - 3) (11 * 43 + 4) > 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l584_58417


namespace NUMINAMATH_CALUDE_fair_compensation_is_two_l584_58455

/-- Represents the scenario of two merchants selling cows and buying sheep --/
structure MerchantScenario where
  num_cows : ℕ
  num_sheep : ℕ
  lamb_price : ℕ

/-- The conditions of the problem --/
def scenario_conditions (s : MerchantScenario) : Prop :=
  ∃ (q : ℕ),
    s.num_sheep = 2 * q + 1 ∧
    s.num_cows ^ 2 = 10 * s.num_sheep + s.lamb_price ∧
    s.lamb_price < 10 ∧
    s.lamb_price > 0

/-- The fair compensation amount --/
def fair_compensation (s : MerchantScenario) : ℕ :=
  (10 - s.lamb_price) / 2

/-- Theorem stating that the fair compensation is 2 yuan --/
theorem fair_compensation_is_two (s : MerchantScenario) 
  (h : scenario_conditions s) : fair_compensation s = 2 := by
  sorry


end NUMINAMATH_CALUDE_fair_compensation_is_two_l584_58455


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l584_58407

/-- Represents an ellipse with foci on the x-axis -/
def is_ellipse_x_axis (m : ℝ) : Prop :=
  m^2 - 1 > 3

/-- The condition m^2 > 5 is sufficient but not necessary for the equation to represent an ellipse with foci on the x-axis -/
theorem sufficient_not_necessary_condition :
  (∀ m : ℝ, m^2 > 5 → is_ellipse_x_axis m) ∧
  (∃ m : ℝ, m^2 ≤ 5 ∧ is_ellipse_x_axis m) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l584_58407


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l584_58487

theorem absolute_value_inequality (x : ℝ) : 
  (3 ≤ |x + 2| ∧ |x + 2| ≤ 7) ↔ ((-9 ≤ x ∧ x ≤ -5) ∨ (1 ≤ x ∧ x ≤ 5)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l584_58487


namespace NUMINAMATH_CALUDE_pens_per_student_is_five_l584_58411

-- Define the given constants
def num_students : ℕ := 30
def notebooks_per_student : ℕ := 3
def binders_per_student : ℕ := 1
def highlighters_per_student : ℕ := 2
def pen_cost : ℚ := 0.5
def notebook_cost : ℚ := 1.25
def binder_cost : ℚ := 4.25
def highlighter_cost : ℚ := 0.75
def teacher_discount : ℚ := 100
def total_spent : ℚ := 260

-- Define the theorem
theorem pens_per_student_is_five :
  let cost_per_student_excl_pens := notebooks_per_student * notebook_cost + 
                                    binders_per_student * binder_cost + 
                                    highlighters_per_student * highlighter_cost
  let total_cost_excl_pens := num_students * cost_per_student_excl_pens
  let total_spent_before_discount := total_spent + teacher_discount
  let total_spent_on_pens := total_spent_before_discount - total_cost_excl_pens
  let total_pens := total_spent_on_pens / pen_cost
  let pens_per_student := total_pens / num_students
  pens_per_student = 5 := by sorry

end NUMINAMATH_CALUDE_pens_per_student_is_five_l584_58411


namespace NUMINAMATH_CALUDE_equation_solution_l584_58432

theorem equation_solution (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) :
  (2 / (x - 1) - (x + 2) / (x^2 - x) = 0) ↔ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l584_58432


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l584_58488

theorem root_sum_reciprocal (p q r A B C : ℝ) : 
  (p ≠ q ∧ q ≠ r ∧ p ≠ r) →
  (∀ x, x^3 - 20*x^2 + 96*x - 91 = 0 ↔ (x = p ∨ x = q ∨ x = r)) →
  (∀ s, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 20*s^2 + 96*s - 91) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 225 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l584_58488


namespace NUMINAMATH_CALUDE_product_equals_square_l584_58490

theorem product_equals_square : 
  1000 * 2.998 * 2.998 * 100 = (29980 : ℝ)^2 := by sorry

end NUMINAMATH_CALUDE_product_equals_square_l584_58490


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l584_58478

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (3 * x) % 31 = 15 % 31 ∧ 
  ∀ (y : ℕ), y > 0 ∧ (3 * y) % 31 = 15 % 31 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l584_58478


namespace NUMINAMATH_CALUDE_ariane_victory_condition_l584_58436

/-- The game between Ariane and Bérénice -/
def game (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 30 ∧
  ∃ (S : Finset ℕ),
    S.card = n ∧
    (∀ x ∈ S, x ≥ 1 ∧ x ≤ 30) ∧
    (∀ d : ℕ, d ≥ 2 →
      (∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ d ∣ a ∧ d ∣ b) ∨
      (∀ a b : ℕ, a ∈ S → b ∈ S → a ≠ b → ¬(d ∣ a ∧ d ∣ b)))

/-- Ariane's winning condition -/
def ariane_wins (n : ℕ) : Prop :=
  game n ∧
  ∃ (S : Finset ℕ),
    S.card = n ∧
    (∀ x ∈ S, x ≥ 1 ∧ x ≤ 30) ∧
    ∀ d : ℕ, d ≥ 2 →
      ∀ a b : ℕ, a ∈ S → b ∈ S → a ≠ b → ¬(d ∣ a ∧ d ∣ b)

/-- The main theorem: Ariane can ensure victory if and only if 1 ≤ n ≤ 11 -/
theorem ariane_victory_condition :
  ∀ n : ℕ, ariane_wins n ↔ (1 ≤ n ∧ n ≤ 11) :=
sorry

end NUMINAMATH_CALUDE_ariane_victory_condition_l584_58436


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l584_58458

theorem pure_imaginary_complex_number (a : ℝ) : 
  (∃ z : ℂ, z = Complex.mk (a^2 + a - 2) (a^2 - 3*a + 2) ∧ z.re = 0 ∧ z.im ≠ 0) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l584_58458


namespace NUMINAMATH_CALUDE_emily_subtraction_l584_58465

theorem emily_subtraction (h : 51^2 = 50^2 + 101) : 50^2 - 49^2 = 99 := by
  sorry

end NUMINAMATH_CALUDE_emily_subtraction_l584_58465


namespace NUMINAMATH_CALUDE_equation_holds_for_seven_halves_l584_58419

theorem equation_holds_for_seven_halves : 
  let x : ℚ := 7/2
  let y : ℚ := (x^2 - 9) / (x - 3)
  y = 3*x - 4 := by sorry

end NUMINAMATH_CALUDE_equation_holds_for_seven_halves_l584_58419


namespace NUMINAMATH_CALUDE_find_number_l584_58403

theorem find_number : ∃! x : ℝ, 7 * x + 21.28 = 50.68 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l584_58403


namespace NUMINAMATH_CALUDE_blue_sock_pairs_l584_58427

theorem blue_sock_pairs (n : ℕ) (k : ℕ) : n = 4 ∧ k = 2 → Nat.choose n k = 6 := by
  sorry

end NUMINAMATH_CALUDE_blue_sock_pairs_l584_58427


namespace NUMINAMATH_CALUDE_exists_n_satisfying_conditions_l584_58447

/-- The number of distinct prime factors of n -/
def omega (n : ℕ) : ℕ := sorry

/-- The sum of the exponents in the prime factorization of n -/
def Omega (n : ℕ) : ℕ := sorry

/-- For any fixed positive integer k and positive reals α and β,
    there exists a positive integer n > 1 satisfying the given conditions -/
theorem exists_n_satisfying_conditions (k : ℕ) (α β : ℝ) 
    (hk : k > 0) (hα : α > 0) (hβ : β > 0) :
  ∃ n : ℕ, n > 1 ∧ 
    (omega (n + k) : ℝ) / (omega n) > α ∧
    (Omega (n + k) : ℝ) / (Omega n) < β := by
  sorry

end NUMINAMATH_CALUDE_exists_n_satisfying_conditions_l584_58447


namespace NUMINAMATH_CALUDE_dans_trip_l584_58485

/-- The distance from Dan's home to his workplace -/
def distance : ℝ := 160

/-- The time of the usual trip in minutes -/
def usual_time : ℝ := 240

/-- The time spent driving at normal speed on the particular day -/
def normal_speed_time : ℝ := 120

/-- The speed reduction factor due to heavy traffic -/
def speed_reduction : ℝ := 0.75

/-- The total trip time on the particular day -/
def total_time : ℝ := 330

theorem dans_trip :
  distance = distance * (normal_speed_time / usual_time + 
    (total_time - normal_speed_time) / (usual_time / speed_reduction)) := by
  sorry

end NUMINAMATH_CALUDE_dans_trip_l584_58485


namespace NUMINAMATH_CALUDE_complex_division_l584_58481

theorem complex_division (z : ℂ) : z = -2 + I → z / (1 + I) = -1/2 + 3/2 * I := by sorry

end NUMINAMATH_CALUDE_complex_division_l584_58481


namespace NUMINAMATH_CALUDE_shaniqua_earnings_l584_58412

/-- Calculates the total earnings for Shaniqua's hair services -/
def total_earnings (haircut_price : ℕ) (style_price : ℕ) (num_haircuts : ℕ) (num_styles : ℕ) : ℕ :=
  haircut_price * num_haircuts + style_price * num_styles

/-- Proves that Shaniqua's total earnings for 8 haircuts and 5 styles are $221 -/
theorem shaniqua_earnings : total_earnings 12 25 8 5 = 221 := by
  sorry

end NUMINAMATH_CALUDE_shaniqua_earnings_l584_58412


namespace NUMINAMATH_CALUDE_problem_solution_l584_58414

/-- A function satisfying the given property for all real numbers -/
def satisfies_property (g : ℝ → ℝ) : Prop :=
  ∀ a c : ℝ, c^3 * g a = a^3 * g c

theorem problem_solution (g : ℝ → ℝ) (h1 : satisfies_property g) (h2 : g 3 ≠ 0) :
  (g 6 - g 2) / g 3 = 208 / 27 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l584_58414


namespace NUMINAMATH_CALUDE_sum_of_squares_16_to_30_l584_58435

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_squares_16_to_30 :
  sum_of_squares 30 - sum_of_squares 15 = 8215 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_16_to_30_l584_58435


namespace NUMINAMATH_CALUDE_one_cow_one_bag_days_l584_58483

-- Define the given condition
def total_cows : ℕ := 26
def total_bags : ℕ := 26
def total_days : ℕ := 26

-- Define the function to calculate days for one cow to eat one bag
def days_for_one_cow_one_bag (cows bags days : ℕ) : ℕ :=
  if cows > 0 ∧ bags > 0 ∧ days > 0 then days else 0

-- Theorem statement
theorem one_cow_one_bag_days :
  days_for_one_cow_one_bag total_cows total_bags total_days = 26 := by
  sorry

end NUMINAMATH_CALUDE_one_cow_one_bag_days_l584_58483


namespace NUMINAMATH_CALUDE_second_container_capacity_l584_58405

/-- Represents a container with dimensions and sand capacity -/
structure Container where
  height : ℝ
  width : ℝ
  length : ℝ
  sandCapacity : ℝ

/-- Theorem stating the sand capacity of the second container -/
theorem second_container_capacity 
  (c1 : Container) 
  (c2 : Container) 
  (h1 : c1.height = 3)
  (h2 : c1.width = 4)
  (h3 : c1.length = 6)
  (h4 : c1.sandCapacity = 72)
  (h5 : c2.height = 3 * c1.height)
  (h6 : c2.width = 2 * c1.width)
  (h7 : c2.length = c1.length) :
  c2.sandCapacity = 432 := by
  sorry


end NUMINAMATH_CALUDE_second_container_capacity_l584_58405


namespace NUMINAMATH_CALUDE_min_weighings_three_l584_58486

/-- Represents the outcome of a weighing --/
inductive WeighingOutcome
  | Equal : WeighingOutcome
  | LeftHeavier : WeighingOutcome
  | RightHeavier : WeighingOutcome

/-- Represents a coin --/
inductive Coin
  | Real : Coin
  | Fake : Coin

/-- Represents a weighing strategy --/
def WeighingStrategy := List (List Coin × List Coin)

/-- The total number of coins --/
def totalCoins : Nat := 2023

/-- The number of fake coins --/
def fakeCoins : Nat := 2

/-- The number of real coins --/
def realCoins : Nat := totalCoins - fakeCoins

/-- A function that determines the outcome of a weighing --/
def weighOutcome (left right : List Coin) : WeighingOutcome := sorry

/-- A function that determines if a strategy is valid --/
def isValidStrategy (strategy : WeighingStrategy) : Prop := sorry

/-- A function that determines if a strategy solves the problem --/
def solvesProblem (strategy : WeighingStrategy) : Prop := sorry

/-- The main theorem stating that the minimum number of weighings is 3 --/
theorem min_weighings_three :
  ∃ (strategy : WeighingStrategy),
    strategy.length = 3 ∧
    isValidStrategy strategy ∧
    solvesProblem strategy ∧
    ∀ (other : WeighingStrategy),
      isValidStrategy other →
      solvesProblem other →
      other.length ≥ 3 := by sorry

end NUMINAMATH_CALUDE_min_weighings_three_l584_58486


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l584_58469

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecularWeight (carbon_atoms : ℕ) (hydrogen_atoms : ℕ) (oxygen_atoms : ℕ) 
                    (carbon_weight : ℝ) (hydrogen_weight : ℝ) (oxygen_weight : ℝ) : ℝ :=
  (carbon_atoms : ℝ) * carbon_weight + (hydrogen_atoms : ℝ) * hydrogen_weight + (oxygen_atoms : ℝ) * oxygen_weight

/-- The molecular weight of a compound with 3 Carbon, 6 Hydrogen, and 1 Oxygen is approximately 58.078 g/mol -/
theorem compound_molecular_weight :
  let carbon_atoms : ℕ := 3
  let hydrogen_atoms : ℕ := 6
  let oxygen_atoms : ℕ := 1
  let carbon_weight : ℝ := 12.01
  let hydrogen_weight : ℝ := 1.008
  let oxygen_weight : ℝ := 16.00
  ∃ ε > 0, |molecularWeight carbon_atoms hydrogen_atoms oxygen_atoms 
                            carbon_weight hydrogen_weight oxygen_weight - 58.078| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l584_58469


namespace NUMINAMATH_CALUDE_wilson_pays_twelve_l584_58475

/-- The total amount Wilson pays at the fast-food restaurant -/
def wilsonTotalPaid (hamburgerPrice : ℕ) (hamburgerCount : ℕ) (colaPrice : ℕ) (colaCount : ℕ) (discountAmount : ℕ) : ℕ :=
  hamburgerPrice * hamburgerCount + colaPrice * colaCount - discountAmount

/-- Theorem: Wilson pays $12 in total -/
theorem wilson_pays_twelve :
  wilsonTotalPaid 5 2 2 3 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_wilson_pays_twelve_l584_58475


namespace NUMINAMATH_CALUDE_james_pages_per_year_l584_58410

/-- Calculates the number of pages James writes in a year -/
def pages_per_year (pages_per_letter : ℕ) (friends : ℕ) (times_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  pages_per_letter * friends * times_per_week * weeks_per_year

/-- Proves that James writes 624 pages in a year -/
theorem james_pages_per_year :
  pages_per_year 3 2 2 52 = 624 := by
  sorry

end NUMINAMATH_CALUDE_james_pages_per_year_l584_58410


namespace NUMINAMATH_CALUDE_decimal_division_remainder_l584_58468

theorem decimal_division_remainder (n : ℕ) (N : ℕ) : 
  (N % (2^n) = (N % 10^n) % (2^n)) ∧ (N % (5^n) = (N % 10^n) % (5^n)) := by sorry

end NUMINAMATH_CALUDE_decimal_division_remainder_l584_58468


namespace NUMINAMATH_CALUDE_total_cantaloupes_l584_58457

def fred_cantaloupes : ℕ := 38
def tim_cantaloupes : ℕ := 44

theorem total_cantaloupes : fred_cantaloupes + tim_cantaloupes = 82 := by
  sorry

end NUMINAMATH_CALUDE_total_cantaloupes_l584_58457


namespace NUMINAMATH_CALUDE_adoption_cost_theorem_l584_58409

def cat_cost : ℕ := 50
def adult_dog_cost : ℕ := 100
def puppy_cost : ℕ := 150

def cats_adopted : ℕ := 2
def adult_dogs_adopted : ℕ := 3
def puppies_adopted : ℕ := 2

def total_cost : ℕ := 
  cat_cost * cats_adopted + 
  adult_dog_cost * adult_dogs_adopted + 
  puppy_cost * puppies_adopted

theorem adoption_cost_theorem : total_cost = 700 := by
  sorry

end NUMINAMATH_CALUDE_adoption_cost_theorem_l584_58409


namespace NUMINAMATH_CALUDE_rectangle_division_condition_l584_58442

/-- A rectangle can be divided into two unequal but similar rectangles if and only if its longer side is more than twice the length of its shorter side. -/
theorem rectangle_division_condition (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≥ b) :
  (∃ x : ℝ, 0 < x ∧ x < a ∧ x * b = (a - x) * x) ↔ a > 2 * b :=
sorry

end NUMINAMATH_CALUDE_rectangle_division_condition_l584_58442


namespace NUMINAMATH_CALUDE_min_value_theorem_l584_58426

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  1/(a-1) + 4/(b-1) ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l584_58426


namespace NUMINAMATH_CALUDE_initial_apples_l584_58434

theorem initial_apples (minseok_ate jaeyoon_ate apples_left : ℕ) 
  (h1 : minseok_ate = 3)
  (h2 : jaeyoon_ate = 3)
  (h3 : apples_left = 2) : 
  minseok_ate + jaeyoon_ate + apples_left = 8 :=
by sorry

end NUMINAMATH_CALUDE_initial_apples_l584_58434


namespace NUMINAMATH_CALUDE_cabbage_distribution_l584_58424

/-- Given a cabbage patch with 12 rows and 180 total heads of cabbage,
    prove that there are 15 heads of cabbage in each row. -/
theorem cabbage_distribution (rows : ℕ) (total_heads : ℕ) (heads_per_row : ℕ) : 
  rows = 12 → total_heads = 180 → heads_per_row * rows = total_heads → heads_per_row = 15 := by
  sorry

end NUMINAMATH_CALUDE_cabbage_distribution_l584_58424


namespace NUMINAMATH_CALUDE_initial_average_calculation_l584_58494

theorem initial_average_calculation (n : ℕ) (correct_avg : ℚ) (error : ℚ) :
  n = 10 →
  correct_avg = 16 →
  error = 10 →
  (n * correct_avg - error) / n = 15 :=
by sorry

end NUMINAMATH_CALUDE_initial_average_calculation_l584_58494


namespace NUMINAMATH_CALUDE_ratio_equals_average_rate_of_change_l584_58400

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the interval endpoints
variable (x x₁ : ℝ)

-- Theorem statement
theorem ratio_equals_average_rate_of_change :
  (f x₁ - f x) / (x₁ - x) = (f x₁ - f x) / (x₁ - x) := by
  sorry

end NUMINAMATH_CALUDE_ratio_equals_average_rate_of_change_l584_58400


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l584_58416

theorem simplify_trig_expression (α : Real) (h : 270 * π / 180 < α ∧ α < 360 * π / 180) :
  Real.sqrt ((1/2) + (1/2) * Real.sqrt ((1/2) + (1/2) * Real.cos (2 * α))) = -Real.cos (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l584_58416


namespace NUMINAMATH_CALUDE_movie_watchers_l584_58496

theorem movie_watchers (total_seats empty_seats : ℕ) 
  (h1 : total_seats = 750)
  (h2 : empty_seats = 218) : 
  total_seats - empty_seats = 532 := by
sorry

end NUMINAMATH_CALUDE_movie_watchers_l584_58496


namespace NUMINAMATH_CALUDE_valid_orders_count_l584_58479

/-- The number of students to select -/
def n : ℕ := 4

/-- The total number of students -/
def total : ℕ := 8

/-- The number of special students (A and B) -/
def special : ℕ := 2

/-- Calculates the number of valid speaking orders -/
def validOrders : ℕ := sorry

theorem valid_orders_count :
  validOrders = 1140 := by sorry

end NUMINAMATH_CALUDE_valid_orders_count_l584_58479


namespace NUMINAMATH_CALUDE_two_std_dev_below_mean_l584_58408

def normal_distribution (mean : ℝ) (std_dev : ℝ) := { μ : ℝ // μ = mean }

theorem two_std_dev_below_mean 
  (μ : ℝ) (σ : ℝ) (h_μ : μ = 14.5) (h_σ : σ = 1.7) :
  ∃ (x : ℝ), x = μ - 2 * σ ∧ x = 11.1 :=
sorry

end NUMINAMATH_CALUDE_two_std_dev_below_mean_l584_58408


namespace NUMINAMATH_CALUDE_brownie_division_l584_58430

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents a tray of brownies -/
structure BrownieTray where
  tray : Dimensions
  piece : Dimensions

/-- Calculates the number of brownie pieces that can be cut from the tray -/
def num_pieces (bt : BrownieTray) : ℕ :=
  (area bt.tray) / (area bt.piece)

/-- Theorem: A 24-inch by 20-inch tray can be divided into exactly 40 pieces of 3-inch by 4-inch brownies -/
theorem brownie_division :
  let bt : BrownieTray := {
    tray := { length := 24, width := 20 },
    piece := { length := 3, width := 4 }
  }
  num_pieces bt = 40 := by sorry

end NUMINAMATH_CALUDE_brownie_division_l584_58430


namespace NUMINAMATH_CALUDE_polynomial_identity_l584_58480

theorem polynomial_identity (a b c : ℝ) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = 2 * (a - b) * (b - c) * (c - a) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l584_58480


namespace NUMINAMATH_CALUDE_fraction_meaningful_l584_58438

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (x - 1) / (x - 2)) ↔ x ≠ 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l584_58438


namespace NUMINAMATH_CALUDE_expression_value_l584_58492

theorem expression_value (α : Real) (h : Real.tan α = -3/4) :
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / 
  (Real.cos (11*π/12 - α) * Real.sin (9*π/2 + α)) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l584_58492


namespace NUMINAMATH_CALUDE_area_change_possibilities_l584_58433

/-- Represents the change in area of a rectangle when one side is increased by 3 cm
    and the other is decreased by 3 cm. --/
def areaChange (a b : ℝ) : ℝ := 3 * (a - b - 3)

/-- Theorem stating that the area change can be positive, negative, or zero. --/
theorem area_change_possibilities (a b : ℝ) :
  ∃ (x y z : ℝ), x > 0 ∧ y < 0 ∧ z = 0 ∧
  (areaChange x b = z ∨ areaChange a x = z) ∧
  (areaChange y b > 0 ∨ areaChange a y > 0) ∧
  (areaChange z b < 0 ∨ areaChange a z < 0) := by
  sorry

end NUMINAMATH_CALUDE_area_change_possibilities_l584_58433


namespace NUMINAMATH_CALUDE_expansion_coefficient_implies_a_value_l584_58437

/-- The coefficient of x^n in the expansion of (x + 1/x)^m -/
def binomialCoeff (m n : ℕ) : ℚ := sorry

/-- The coefficient of x^n in the expansion of (x^2 - a)(x + 1/x)^m -/
def expandedCoeff (m n : ℕ) (a : ℚ) : ℚ := 
  binomialCoeff m (m - n + 2) - a * binomialCoeff m (m - n)

theorem expansion_coefficient_implies_a_value : 
  expandedCoeff 10 6 a = 30 → a = 2 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_implies_a_value_l584_58437


namespace NUMINAMATH_CALUDE_freddy_age_l584_58461

/-- Given the ages of several people and their relationships, prove Freddy's age. -/
theorem freddy_age (stephanie tim job oliver tina freddy : ℝ) 
  (h1 : freddy = stephanie - 2.5)
  (h2 : 3 * stephanie = job + tim)
  (h3 : tim = oliver / 2)
  (h4 : oliver / 3 = tina)
  (h5 : tina = freddy - 2)
  (h6 : job = 5)
  (h7 : oliver = job + 10) : 
  freddy = 7 := by
  sorry

end NUMINAMATH_CALUDE_freddy_age_l584_58461


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l584_58450

theorem rectangular_solid_diagonal (x y z : ℝ) : 
  (2 * (x * y + y * z + z * x) = 62) →
  (4 * (x + y + z) = 48) →
  (x^2 + y^2 + z^2 : ℝ) = 82 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l584_58450


namespace NUMINAMATH_CALUDE_cookie_production_l584_58489

def initial_cookies : ℕ := 24
def initial_flour : ℕ := 3
def efficiency_improvement : ℚ := 1/10
def new_flour : ℕ := 4

def improved_cookies : ℕ := 35

theorem cookie_production : 
  let initial_efficiency : ℚ := initial_cookies / initial_flour
  let improved_efficiency : ℚ := initial_efficiency * (1 + efficiency_improvement)
  let theoretical_cookies : ℚ := improved_efficiency * new_flour
  ⌊theoretical_cookies⌋ = improved_cookies := by sorry

end NUMINAMATH_CALUDE_cookie_production_l584_58489


namespace NUMINAMATH_CALUDE_lance_reading_plan_l584_58467

/-- Given a book with a certain number of pages, calculate the number of pages 
    to read on the third day to finish the book, given the pages read on the 
    first two days and constraints for the third day. -/
def pagesOnThirdDay (totalPages : ℕ) (day1Pages : ℕ) (day2Reduction : ℕ) : ℕ :=
  let day2Pages := day1Pages - day2Reduction
  let remainingPages := totalPages - (day1Pages + day2Pages)
  ((remainingPages + 9) / 10) * 10

theorem lance_reading_plan :
  pagesOnThirdDay 100 35 5 = 40 := by sorry

end NUMINAMATH_CALUDE_lance_reading_plan_l584_58467


namespace NUMINAMATH_CALUDE_chord_AB_equation_tangent_circle_equation_l584_58420

-- Define the parabola E
def E (x y : ℝ) : Prop := x^2 = 4*y

-- Define point M
def M : ℝ × ℝ := (1, 4)

-- Define the chord AB passing through M as its midpoint
def chord_AB (x y : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ),
    E x1 y1 ∧ E x2 y2 ∧
    (x, y) = ((x1 + x2)/2, (y1 + y2)/2) ∧
    (x, y) = M

-- Define the tangent line l
def tangent_line (x0 y0 b : ℝ) : Prop :=
  E x0 y0 ∧ y0 = x0 + b

-- Theorem for the equation of line AB
theorem chord_AB_equation :
  ∀ x y : ℝ, chord_AB x y → x - 2*y + 7 = 0 := sorry

-- Theorem for the equation of the circle
theorem tangent_circle_equation :
  ∀ x0 y0 b : ℝ,
    tangent_line x0 y0 b →
    ∀ x y : ℝ, (x - 2)^2 + (y - 1)^2 = 4 := sorry

end NUMINAMATH_CALUDE_chord_AB_equation_tangent_circle_equation_l584_58420


namespace NUMINAMATH_CALUDE_phillips_jars_l584_58423

-- Define the given quantities
def cucumbers : ℕ := 10
def initial_vinegar : ℕ := 100
def pickles_per_cucumber : ℕ := 6
def pickles_per_jar : ℕ := 12
def vinegar_per_jar : ℕ := 10
def remaining_vinegar : ℕ := 60

-- Define the function to calculate the number of jars
def number_of_jars : ℕ :=
  min
    (cucumbers * pickles_per_cucumber / pickles_per_jar)
    ((initial_vinegar - remaining_vinegar) / vinegar_per_jar)

-- Theorem statement
theorem phillips_jars :
  number_of_jars = 4 :=
sorry

end NUMINAMATH_CALUDE_phillips_jars_l584_58423


namespace NUMINAMATH_CALUDE_correct_second_sale_price_l584_58431

/-- The price of a single toothbrush in yuan -/
def toothbrush_price : ℝ := sorry

/-- The price of a single tube of toothpaste in yuan -/
def toothpaste_price : ℝ := sorry

/-- The total price of 26 toothbrushes and 14 tubes of toothpaste -/
def first_sale : ℝ := 26 * toothbrush_price + 14 * toothpaste_price

/-- The recorded total price of the first sale -/
def first_sale_record : ℝ := 264

theorem correct_second_sale_price :
  first_sale = first_sale_record →
  39 * toothbrush_price + 21 * toothpaste_price = 396 := by
  sorry

end NUMINAMATH_CALUDE_correct_second_sale_price_l584_58431


namespace NUMINAMATH_CALUDE_bowl_cost_l584_58459

theorem bowl_cost (sets : ℕ) (bowls_per_set : ℕ) (total_cost : ℕ) : 
  sets = 12 → bowls_per_set = 2 → total_cost = 240 → 
  (total_cost : ℚ) / (sets * bowls_per_set : ℚ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_bowl_cost_l584_58459


namespace NUMINAMATH_CALUDE_composite_fraction_theorem_l584_58406

def first_eight_composites : List Nat := [4, 6, 8, 9, 10, 12, 14, 15]
def next_eight_composites : List Nat := [16, 18, 20, 21, 22, 24, 25, 26]
def first_prime : Nat := 2
def second_prime : Nat := 3

theorem composite_fraction_theorem :
  let numerator := (List.prod first_eight_composites + first_prime)
  let denominator := (List.prod next_eight_composites + second_prime)
  (numerator : ℚ) / denominator = 
    (4 * 6 * 8 * 9 * 10 * 12 * 14 * 15 + 2 : ℚ) / 
    (16 * 18 * 20 * 21 * 22 * 24 * 25 * 26 + 3) := by
  sorry

end NUMINAMATH_CALUDE_composite_fraction_theorem_l584_58406


namespace NUMINAMATH_CALUDE_vector_operation_result_l584_58441

def vector_operation : ℝ × ℝ := sorry

theorem vector_operation_result :
  vector_operation = (-3, 32) := by sorry

end NUMINAMATH_CALUDE_vector_operation_result_l584_58441


namespace NUMINAMATH_CALUDE_pentagon_cannot_cover_floor_l584_58473

/-- The interior angle of a regular polygon with n sides --/
def interior_angle (n : ℕ) : ℚ := (n - 2 : ℚ) * 180 / n

/-- Check if an angle divides 360° evenly --/
def divides_360 (angle : ℚ) : Prop := ∃ k : ℕ, (k : ℚ) * angle = 360

theorem pentagon_cannot_cover_floor :
  divides_360 (interior_angle 6) ∧ 
  divides_360 (interior_angle 4) ∧ 
  divides_360 (interior_angle 3) ∧ 
  ¬divides_360 (interior_angle 5) := by sorry

end NUMINAMATH_CALUDE_pentagon_cannot_cover_floor_l584_58473


namespace NUMINAMATH_CALUDE_units_digit_of_13_power_2003_l584_58429

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define a function to compute the units digit of 3^n
def unitsDigitOf3Power (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0  -- This case should never occur

-- State the theorem
theorem units_digit_of_13_power_2003 :
  unitsDigit (13^2003) = unitsDigitOf3Power 2003 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_13_power_2003_l584_58429


namespace NUMINAMATH_CALUDE_male_25_plus_percentage_proof_l584_58470

/-- The percentage of male students in a graduating class -/
def male_percentage : ℝ := 0.4

/-- The percentage of female students who are 25 years old or older -/
def female_25_plus_percentage : ℝ := 0.4

/-- The probability of randomly selecting a student less than 25 years old -/
def under_25_probability : ℝ := 0.56

/-- The percentage of male students who are 25 years old or older -/
def male_25_plus_percentage : ℝ := 0.5

theorem male_25_plus_percentage_proof :
  male_25_plus_percentage = 0.5 :=
by sorry

end NUMINAMATH_CALUDE_male_25_plus_percentage_proof_l584_58470


namespace NUMINAMATH_CALUDE_extremum_implies_a_eq_one_f_less_than_c_squared_implies_c_range_l584_58482

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 - 9*x + 5

-- Theorem 1: If f has an extremum at x = 1, then a = 1
theorem extremum_implies_a_eq_one (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1 ∨ f a x ≥ f a 1) →
  a = 1 :=
sorry

-- Theorem 2: If f(x) < c² for all x in [-4, 4], then c is in (-∞, -9) ∪ (9, +∞)
theorem f_less_than_c_squared_implies_c_range :
  (∀ x ∈ Set.Icc (-4) 4, f 1 x < c^2) →
  c ∈ Set.Iio (-9) ∪ Set.Ioi 9 :=
sorry

end NUMINAMATH_CALUDE_extremum_implies_a_eq_one_f_less_than_c_squared_implies_c_range_l584_58482


namespace NUMINAMATH_CALUDE_shifted_quadratic_sum_l584_58497

/-- The sum of coefficients after shifting a quadratic function -/
theorem shifted_quadratic_sum (a b c : ℝ) : 
  (∀ x, 3 * (x + 2)^2 + 2 * (x + 2) + 4 = a * x^2 + b * x + c) → 
  a + b + c = 37 := by
sorry

end NUMINAMATH_CALUDE_shifted_quadratic_sum_l584_58497


namespace NUMINAMATH_CALUDE_oil_measurement_l584_58460

/-- The amount of oil currently in Scarlett's measuring cup -/
def current_oil : ℝ := 0.16666666666666674

/-- The amount of oil Scarlett adds to the measuring cup -/
def added_oil : ℝ := 0.6666666666666666

/-- The total amount of oil after Scarlett adds more -/
def total_oil : ℝ := 0.8333333333333334

/-- Theorem stating that the current amount of oil plus the added amount equals the total amount -/
theorem oil_measurement :
  current_oil + added_oil = total_oil :=
by sorry

end NUMINAMATH_CALUDE_oil_measurement_l584_58460
