import Mathlib

namespace NUMINAMATH_CALUDE_red_marble_fraction_l2028_202807

theorem red_marble_fraction (total : ℝ) (h : total > 0) :
  let initial_blue := (2/3 : ℝ) * total
  let initial_red := total - initial_blue
  let new_red := 3 * initial_red
  let new_total := initial_blue + new_red
  new_red / new_total = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_red_marble_fraction_l2028_202807


namespace NUMINAMATH_CALUDE_cashback_is_twelve_percent_l2028_202843

/-- Calculates the cashback percentage given the total cost, rebate, and final cost -/
def cashback_percentage (total_cost rebate final_cost : ℚ) : ℚ :=
  let cost_after_rebate := total_cost - rebate
  let cashback_amount := cost_after_rebate - final_cost
  (cashback_amount / cost_after_rebate) * 100

/-- Theorem stating that the cashback percentage is 12% given the problem conditions -/
theorem cashback_is_twelve_percent :
  cashback_percentage 150 25 110 = 12 := by
  sorry

end NUMINAMATH_CALUDE_cashback_is_twelve_percent_l2028_202843


namespace NUMINAMATH_CALUDE_only_log29_undetermined_l2028_202857

-- Define the given logarithms
def log7 : ℝ := 0.8451
def log10 : ℝ := 1

-- Define a function to represent whether a logarithm can be determined
def can_determine (x : ℝ) : Prop := 
  ∃ (f : ℝ → ℝ → ℝ), f log7 log10 = Real.log x

-- State the theorem
theorem only_log29_undetermined :
  ¬(can_determine 29) ∧ 
  can_determine (5/9) ∧ 
  can_determine 35 ∧ 
  can_determine 700 ∧ 
  can_determine 0.6 := by
  sorry


end NUMINAMATH_CALUDE_only_log29_undetermined_l2028_202857


namespace NUMINAMATH_CALUDE_tan_squared_gamma_equals_tan_alpha_tan_beta_l2028_202893

theorem tan_squared_gamma_equals_tan_alpha_tan_beta 
  (α β γ : Real) 
  (h : (Real.sin γ)^2 / (Real.sin α)^2 = 1 - Real.tan (α - β) / Real.tan α) : 
  (Real.tan γ)^2 = Real.tan α * Real.tan β := by
  sorry

end NUMINAMATH_CALUDE_tan_squared_gamma_equals_tan_alpha_tan_beta_l2028_202893


namespace NUMINAMATH_CALUDE_anitas_class_size_l2028_202863

/-- The number of students in Anita's class -/
def num_students : ℕ := 360 / 6

/-- Theorem: The number of students in Anita's class is 60 -/
theorem anitas_class_size :
  num_students = 60 :=
by sorry

end NUMINAMATH_CALUDE_anitas_class_size_l2028_202863


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_l2028_202803

theorem sum_of_roots_cubic (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 - 3*x^2 + 2*x
  (∃ a b c : ℝ, f x = (x - a) * (x - b) * (x - c)) → 
  (a + b + c = 3) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_l2028_202803


namespace NUMINAMATH_CALUDE_sin_double_pi_minus_theta_l2028_202829

theorem sin_double_pi_minus_theta (θ : ℝ) 
  (h1 : 3 * (Real.cos θ)^2 = Real.tan θ + 3) 
  (h2 : ∀ k : ℤ, θ ≠ k * Real.pi) : 
  Real.sin (2 * (Real.pi - θ)) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_pi_minus_theta_l2028_202829


namespace NUMINAMATH_CALUDE_cos_symmetry_l2028_202895

/-- The function f(x) = cos(2x + π/3) is symmetric about the line x = π/3 -/
theorem cos_symmetry (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.cos (2 * x + π / 3)
  ∀ y : ℝ, f (π / 3 + y) = f (π / 3 - y) := by
  sorry

end NUMINAMATH_CALUDE_cos_symmetry_l2028_202895


namespace NUMINAMATH_CALUDE_max_availability_equal_all_days_l2028_202819

-- Define the days of the week
inductive Day
  | Mon
  | Tues
  | Wed
  | Thurs
  | Fri

-- Define the team members
inductive Member
  | Alice
  | Bob
  | Cindy
  | David

-- Define the availability function
def availability (m : Member) (d : Day) : Bool :=
  match m, d with
  | Member.Alice, Day.Mon => false
  | Member.Alice, Day.Tues => false
  | Member.Alice, Day.Wed => true
  | Member.Alice, Day.Thurs => true
  | Member.Alice, Day.Fri => false
  | Member.Bob, Day.Mon => true
  | Member.Bob, Day.Tues => false
  | Member.Bob, Day.Wed => false
  | Member.Bob, Day.Thurs => true
  | Member.Bob, Day.Fri => true
  | Member.Cindy, Day.Mon => false
  | Member.Cindy, Day.Tues => true
  | Member.Cindy, Day.Wed => false
  | Member.Cindy, Day.Thurs => false
  | Member.Cindy, Day.Fri => true
  | Member.David, Day.Mon => true
  | Member.David, Day.Tues => true
  | Member.David, Day.Wed => true
  | Member.David, Day.Thurs => false
  | Member.David, Day.Fri => false

-- Count available members for a given day
def availableCount (d : Day) : Nat :=
  (List.filter (fun m => availability m d) [Member.Alice, Member.Bob, Member.Cindy, Member.David]).length

-- Theorem: The maximum number of available members is equal for all days
theorem max_availability_equal_all_days :
  (List.map availableCount [Day.Mon, Day.Tues, Day.Wed, Day.Thurs, Day.Fri]).all (· = 2) := by
  sorry


end NUMINAMATH_CALUDE_max_availability_equal_all_days_l2028_202819


namespace NUMINAMATH_CALUDE_two_valid_configurations_l2028_202825

/-- Represents a square piece of the figure -/
inductive Square
| Base
| A
| B
| C
| D
| E
| F
| G

/-- Represents the L-shaped figure -/
def LShape := List Square

/-- Represents a configuration of squares -/
def Configuration := List Square

/-- Checks if a configuration can form a topless cubical box -/
def is_valid_box (config : Configuration) : Prop :=
  sorry

/-- The set of all possible configurations -/
def all_configurations : Set Configuration :=
  sorry

/-- The number of valid configurations that form a topless cubical box -/
def num_valid_configurations : ℕ :=
  sorry

/-- Theorem stating that there are exactly two valid configurations -/
theorem two_valid_configurations :
  num_valid_configurations = 2 :=
sorry

end NUMINAMATH_CALUDE_two_valid_configurations_l2028_202825


namespace NUMINAMATH_CALUDE_no_valid_labeling_exists_l2028_202880

/-- Represents a labeling of a 45-gon with digits 0-9 -/
def Labeling := Fin 45 → Fin 10

/-- Checks if a labeling is valid according to the problem conditions -/
def is_valid_labeling (l : Labeling) : Prop :=
  ∀ i j : Fin 10, i ≠ j →
    ∃! k : Fin 45, (l k = i ∧ l (k + 1) = j) ∨ (l k = j ∧ l (k + 1) = i)

/-- The main theorem stating that no valid labeling exists -/
theorem no_valid_labeling_exists : ¬∃ l : Labeling, is_valid_labeling l := by
  sorry

end NUMINAMATH_CALUDE_no_valid_labeling_exists_l2028_202880


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2028_202892

theorem fraction_to_decimal : (22 : ℚ) / 8 = 2.75 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2028_202892


namespace NUMINAMATH_CALUDE_ladybug_leaf_count_l2028_202885

theorem ladybug_leaf_count (ladybugs_per_leaf : ℕ) (total_ladybugs : ℕ) (h1 : ladybugs_per_leaf = 139) (h2 : total_ladybugs = 11676) :
  total_ladybugs / ladybugs_per_leaf = 84 := by
  sorry

end NUMINAMATH_CALUDE_ladybug_leaf_count_l2028_202885


namespace NUMINAMATH_CALUDE_percentage_increase_l2028_202874

theorem percentage_increase (x : ℝ) (h : x = 123.2) : 
  (x - 88) / 88 * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l2028_202874


namespace NUMINAMATH_CALUDE_product_of_ten_proper_fractions_equals_one_tenth_l2028_202866

theorem product_of_ten_proper_fractions_equals_one_tenth :
  ∃ (a b c d e f g h i j : ℚ),
    (0 < a ∧ a < 1) ∧
    (0 < b ∧ b < 1) ∧
    (0 < c ∧ c < 1) ∧
    (0 < d ∧ d < 1) ∧
    (0 < e ∧ e < 1) ∧
    (0 < f ∧ f < 1) ∧
    (0 < g ∧ g < 1) ∧
    (0 < h ∧ h < 1) ∧
    (0 < i ∧ i < 1) ∧
    (0 < j ∧ j < 1) ∧
    a * b * c * d * e * f * g * h * i * j = 1/10 :=
by sorry

end NUMINAMATH_CALUDE_product_of_ten_proper_fractions_equals_one_tenth_l2028_202866


namespace NUMINAMATH_CALUDE_min_cards_to_form_square_l2028_202859

/-- Represents the width of the rectangular card in centimeters -/
def card_width : ℕ := 20

/-- Represents the length of the rectangular card in centimeters -/
def card_length : ℕ := 8

/-- Represents the area of a single card in square centimeters -/
def card_area : ℕ := card_width * card_length

/-- Represents the side length of the smallest square that can be formed -/
def square_side : ℕ := Nat.lcm card_width card_length

/-- Represents the area of the smallest square that can be formed -/
def square_area : ℕ := square_side * square_side

/-- The minimum number of cards needed to form the smallest square -/
def min_cards : ℕ := square_area / card_area

theorem min_cards_to_form_square : min_cards = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_cards_to_form_square_l2028_202859


namespace NUMINAMATH_CALUDE_angle_minus_510_in_third_quadrant_l2028_202811

-- Define the function to convert an angle to its equivalent within 0° to 360°
def convertAngle (angle : Int) : Int :=
  angle % 360

-- Define the function to determine the quadrant of an angle
def getQuadrant (angle : Int) : Nat :=
  let convertedAngle := convertAngle angle
  if 0 ≤ convertedAngle ∧ convertedAngle < 90 then 1
  else if 90 ≤ convertedAngle ∧ convertedAngle < 180 then 2
  else if 180 ≤ convertedAngle ∧ convertedAngle < 270 then 3
  else 4

-- Theorem statement
theorem angle_minus_510_in_third_quadrant :
  getQuadrant (-510) = 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_minus_510_in_third_quadrant_l2028_202811


namespace NUMINAMATH_CALUDE_blue_eyed_brunettes_l2028_202804

theorem blue_eyed_brunettes (total : ℕ) (blue_eyed_blondes : ℕ) (brunettes : ℕ) (brown_eyed : ℕ) :
  total = 60 →
  blue_eyed_blondes = 20 →
  brunettes = 36 →
  brown_eyed = 23 →
  ∃ (blue_eyed_brunettes : ℕ),
    blue_eyed_brunettes = 17 ∧
    blue_eyed_brunettes + blue_eyed_blondes = total - brown_eyed ∧
    blue_eyed_brunettes + (brunettes - blue_eyed_brunettes) = brown_eyed :=
by sorry

end NUMINAMATH_CALUDE_blue_eyed_brunettes_l2028_202804


namespace NUMINAMATH_CALUDE_largest_angle_is_90_l2028_202896

-- Define an isosceles triangle with angles α, β, and γ
structure IsoscelesTriangle where
  α : Real
  β : Real
  γ : Real
  isIsosceles : (α = β) ∨ (α = γ) ∨ (β = γ)
  sumIs180 : α + β + γ = 180
  nonNegative : α ≥ 0 ∧ β ≥ 0 ∧ γ ≥ 0

-- Define the condition that two angles are in the ratio 1:2
def hasRatio1to2 (t : IsoscelesTriangle) : Prop :=
  (t.α = 2 * t.β) ∨ (t.β = 2 * t.α) ∨ (t.α = 2 * t.γ) ∨ (t.γ = 2 * t.α) ∨ (t.β = 2 * t.γ) ∨ (t.γ = 2 * t.β)

-- Theorem statement
theorem largest_angle_is_90 (t : IsoscelesTriangle) (h : hasRatio1to2 t) :
  max t.α (max t.β t.γ) = 90 := by sorry

end NUMINAMATH_CALUDE_largest_angle_is_90_l2028_202896


namespace NUMINAMATH_CALUDE_smallest_n_with_common_factor_l2028_202827

theorem smallest_n_with_common_factor : 
  ∀ n : ℕ, n > 0 → n < 38 → ¬(∃ k : ℕ, k > 1 ∧ k ∣ (11*n - 3) ∧ k ∣ (8*n + 4)) ∧
  ∃ k : ℕ, k > 1 ∧ k ∣ (11*38 - 3) ∧ k ∣ (8*38 + 4) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_common_factor_l2028_202827


namespace NUMINAMATH_CALUDE_mango_rate_per_kg_mango_rate_proof_l2028_202847

/-- The rate of mangoes per kilogram given the purchase conditions --/
theorem mango_rate_per_kg : ℝ → Prop :=
  fun rate =>
    let grape_quantity : ℝ := 8
    let grape_rate : ℝ := 70
    let mango_quantity : ℝ := 9
    let total_paid : ℝ := 1145
    grape_quantity * grape_rate + mango_quantity * rate = total_paid →
    rate = 65

/-- Proof of the mango rate per kilogram --/
theorem mango_rate_proof : mango_rate_per_kg 65 := by
  sorry

end NUMINAMATH_CALUDE_mango_rate_per_kg_mango_rate_proof_l2028_202847


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_15_13_l2028_202868

theorem half_abs_diff_squares_15_13 : 
  (1/2 : ℝ) * |15^2 - 13^2| = 28 := by sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_15_13_l2028_202868


namespace NUMINAMATH_CALUDE_purchase_cost_l2028_202837

/-- The cost of purchasing bananas and oranges -/
theorem purchase_cost (banana_quantity : ℕ) (orange_quantity : ℕ) 
  (banana_price : ℚ) (orange_price : ℚ) : 
  banana_quantity = 5 → 
  orange_quantity = 10 → 
  banana_price = 2 → 
  orange_price = (3/2) → 
  banana_quantity * banana_price + orange_quantity * orange_price = 25 := by
  sorry

end NUMINAMATH_CALUDE_purchase_cost_l2028_202837


namespace NUMINAMATH_CALUDE_john_mary_difference_l2028_202891

/-- The number of chickens each person took -/
structure ChickenCount where
  ray : ℕ
  john : ℕ
  mary : ℕ

/-- The conditions of the chicken distribution problem -/
def chicken_problem (c : ChickenCount) : Prop :=
  c.ray = 10 ∧
  c.john = c.ray + 11 ∧
  c.mary = c.ray + 6

/-- The theorem stating the difference between John's and Mary's chicken count -/
theorem john_mary_difference (c : ChickenCount) 
  (h : chicken_problem c) : c.john - c.mary = 5 := by
  sorry

end NUMINAMATH_CALUDE_john_mary_difference_l2028_202891


namespace NUMINAMATH_CALUDE_problem_3_l2028_202815

theorem problem_3 (a : ℝ) (h : a = 1 / (Real.sqrt 2 - 1)) : 4 * a^2 - 8 * a + 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_3_l2028_202815


namespace NUMINAMATH_CALUDE_single_digit_sum_l2028_202898

theorem single_digit_sum (a b : ℕ) : 
  a ∈ Finset.range 10 ∧ a ≠ 0 ∧
  b ∈ Finset.range 10 ∧ b ≠ 0 ∧
  82 * 10 * a + 7 + 6 * b = 190 →
  a + 2 * b = 7 := by
sorry

end NUMINAMATH_CALUDE_single_digit_sum_l2028_202898


namespace NUMINAMATH_CALUDE_zeros_after_decimal_point_l2028_202890

-- Define the fraction
def fraction : ℚ := 3 / (25^25)

-- Define the number of zeros after the decimal point
def num_zeros : ℕ := 18

-- Theorem statement
theorem zeros_after_decimal_point :
  (fraction * (10^num_zeros)).floor = 0 ∧
  (fraction * (10^(num_zeros + 1))).floor ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_zeros_after_decimal_point_l2028_202890


namespace NUMINAMATH_CALUDE_ticket_price_difference_l2028_202834

/-- Represents the total amount paid for pre-booked tickets -/
def prebooked_total : ℕ := 10 * 140 + 10 * 170

/-- Represents the total amount paid for tickets bought at the gate -/
def gate_total : ℕ := 8 * 190 + 12 * 210 + 10 * 300

/-- Theorem stating the difference in total amount paid -/
theorem ticket_price_difference : gate_total - prebooked_total = 3940 := by
  sorry

end NUMINAMATH_CALUDE_ticket_price_difference_l2028_202834


namespace NUMINAMATH_CALUDE_percentage_relation_l2028_202877

theorem percentage_relation (x a b : ℝ) (ha : a = 0.07 * x) (hb : b = 0.14 * x) :
  a / b = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_percentage_relation_l2028_202877


namespace NUMINAMATH_CALUDE_safe_locks_theorem_l2028_202855

/-- The number of people in the commission -/
def n : ℕ := 9

/-- The minimum number of people required to access the safe -/
def k : ℕ := 6

/-- The number of keys per lock -/
def keys_per_lock : ℕ := n - k + 1

/-- The number of locks required -/
def num_locks : ℕ := Nat.choose n keys_per_lock

theorem safe_locks_theorem : 
  num_locks = Nat.choose n keys_per_lock :=
by sorry

end NUMINAMATH_CALUDE_safe_locks_theorem_l2028_202855


namespace NUMINAMATH_CALUDE_square_pieces_count_l2028_202822

/-- Represents a square sheet of paper -/
structure SquareSheet :=
  (side : ℝ)
  (area : ℝ := side * side)

/-- Represents the state of the paper after folding and cutting -/
structure FoldedCutSheet :=
  (original : SquareSheet)
  (num_folds : ℕ)
  (num_cuts : ℕ)

/-- Counts the number of square pieces after unfolding -/
def count_square_pieces (sheet : FoldedCutSheet) : ℕ :=
  sorry

/-- Theorem stating that folding a square sheet twice and cutting twice results in 5 square pieces -/
theorem square_pieces_count (s : SquareSheet) :
  let folded_cut := FoldedCutSheet.mk s 2 2
  count_square_pieces folded_cut = 5 :=
sorry

end NUMINAMATH_CALUDE_square_pieces_count_l2028_202822


namespace NUMINAMATH_CALUDE_platform_length_l2028_202805

/-- Given a train of length 300 meters that crosses a platform in 27 seconds
    and a signal pole in 18 seconds, prove that the platform length is 150 meters. -/
theorem platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) :
  train_length = 300 →
  platform_time = 27 →
  pole_time = 18 →
  ∃ (platform_length : ℝ),
    platform_length = 150 ∧
    train_length / pole_time = (train_length + platform_length) / platform_time :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l2028_202805


namespace NUMINAMATH_CALUDE_hotel_profit_equation_l2028_202845

/-- Represents a hotel's pricing and occupancy model -/
structure Hotel where
  totalRooms : ℕ
  basePrice : ℕ
  priceStep : ℕ
  costPerRoom : ℕ

/-- Calculates the number of occupied rooms based on the current price -/
def occupiedRooms (h : Hotel) (price : ℕ) : ℕ :=
  h.totalRooms - (price - h.basePrice) / h.priceStep

/-- Calculates the profit for a given price -/
def profit (h : Hotel) (price : ℕ) : ℕ :=
  (price - h.costPerRoom) * occupiedRooms h price

/-- Theorem stating that the given equation correctly represents the hotel's profit -/
theorem hotel_profit_equation (desiredProfit : ℕ) :
  let h : Hotel := {
    totalRooms := 50,
    basePrice := 180,
    priceStep := 10,
    costPerRoom := 20
  }
  ∀ x : ℕ, profit h x = desiredProfit ↔ (x - 20) * (50 - (x - 180) / 10) = desiredProfit :=
by sorry

end NUMINAMATH_CALUDE_hotel_profit_equation_l2028_202845


namespace NUMINAMATH_CALUDE_min_value_theorem_l2028_202809

theorem min_value_theorem (x : ℝ) (h : x > 0) : 
  x + 1 / (2 * x) ≥ Real.sqrt 2 ∧ 
  ∃ y > 0, y + 1 / (2 * y) = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2028_202809


namespace NUMINAMATH_CALUDE_dry_mixed_fruits_weight_l2028_202833

/-- Calculates the weight of dry mixed fruits after dehydration -/
def weight_dry_mixed_fruits (fresh_grapes fresh_apples : ℝ) 
  (fresh_grapes_water fresh_apples_water : ℝ) : ℝ :=
  (1 - fresh_grapes_water) * fresh_grapes + (1 - fresh_apples_water) * fresh_apples

/-- Theorem: The weight of dry mixed fruits is 188 kg -/
theorem dry_mixed_fruits_weight :
  weight_dry_mixed_fruits 400 300 0.65 0.84 = 188 := by
  sorry

#eval weight_dry_mixed_fruits 400 300 0.65 0.84

end NUMINAMATH_CALUDE_dry_mixed_fruits_weight_l2028_202833


namespace NUMINAMATH_CALUDE_chewbacca_gum_pack_size_l2028_202886

theorem chewbacca_gum_pack_size :
  ∀ x : ℕ,
  (20 : ℚ) / 30 = (20 - x) / 30 →
  (20 : ℚ) / 30 = 20 / (30 + 5 * x) →
  x ≠ 0 →
  x = 14 := by
sorry

end NUMINAMATH_CALUDE_chewbacca_gum_pack_size_l2028_202886


namespace NUMINAMATH_CALUDE_lottery_winning_probability_l2028_202852

/-- The number of options for the MagicBall -/
def magicBallOptions : ℕ := 25

/-- The number of options for each TrophyBall -/
def trophyBallOptions : ℕ := 48

/-- The number of TrophyBalls to be selected -/
def trophyBallsToSelect : ℕ := 5

/-- The probability of winning the lottery -/
def winningProbability : ℚ := 1 / 63180547200

theorem lottery_winning_probability :
  1 / (magicBallOptions * (trophyBallOptions.factorial / (trophyBallOptions - trophyBallsToSelect).factorial)) = winningProbability :=
sorry

end NUMINAMATH_CALUDE_lottery_winning_probability_l2028_202852


namespace NUMINAMATH_CALUDE_sheena_sewing_hours_per_week_l2028_202897

/-- Proves that Sheena sews 4 hours per week given the problem conditions -/
theorem sheena_sewing_hours_per_week 
  (time_per_dress : ℕ) 
  (num_dresses : ℕ) 
  (total_weeks : ℕ) 
  (h1 : time_per_dress = 12)
  (h2 : num_dresses = 5)
  (h3 : total_weeks = 15) :
  (time_per_dress * num_dresses) / total_weeks = 4 :=
by sorry

end NUMINAMATH_CALUDE_sheena_sewing_hours_per_week_l2028_202897


namespace NUMINAMATH_CALUDE_simplify_expression_l2028_202870

theorem simplify_expression : ((5 * 10^7) / (2 * 10^2)) + (4 * 10^5) = 650000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2028_202870


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l2028_202814

theorem repeating_decimal_sum (a b c d : ℕ) : 
  (a ≤ 9) → (b ≤ 9) → (c ≤ 9) → (d ≤ 9) →
  ((10 * a + c) / 99 + (1000 * a + 100 * b + 10 * c + d) / 9999 = 17 / 37) →
  (a = 2 ∧ b = 3 ∧ c = 1 ∧ d = 5) := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l2028_202814


namespace NUMINAMATH_CALUDE_running_speed_calculation_l2028_202867

/-- Proves that the running speed is 8 km/hr given the problem conditions --/
theorem running_speed_calculation (walking_speed : ℝ) (total_distance : ℝ) (total_time : ℝ) 
  (h1 : walking_speed = 4)
  (h2 : total_distance = 8)
  (h3 : total_time = 1.5)
  (h4 : total_distance / 2 / walking_speed + total_distance / 2 / running_speed = total_time) :
  running_speed = 8 := by
  sorry


end NUMINAMATH_CALUDE_running_speed_calculation_l2028_202867


namespace NUMINAMATH_CALUDE_min_value_sum_squared_over_one_plus_l2028_202806

theorem min_value_sum_squared_over_one_plus (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_one : x + y + z = 1) : 
  x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z) ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_squared_over_one_plus_l2028_202806


namespace NUMINAMATH_CALUDE_max_pons_is_nine_nine_pons_achievable_l2028_202816

/-- Represents the number of items Bill can buy -/
structure ItemCounts where
  pan : ℕ
  pin : ℕ
  pon : ℕ

/-- Calculates the total cost of the items -/
def totalCost (items : ItemCounts) : ℕ :=
  3 * items.pan + 5 * items.pin + 10 * items.pon

/-- Checks if the item counts satisfy the conditions -/
def isValid (items : ItemCounts) : Prop :=
  items.pan ≥ 1 ∧ items.pin ≥ 1 ∧ items.pon ≥ 1 ∧ totalCost items = 100

/-- The maximum number of pons that can be purchased -/
def maxPons : ℕ := 9

theorem max_pons_is_nine :
  ∀ items : ItemCounts, isValid items → items.pon ≤ maxPons :=
by sorry

theorem nine_pons_achievable :
  ∃ items : ItemCounts, isValid items ∧ items.pon = maxPons :=
by sorry

end NUMINAMATH_CALUDE_max_pons_is_nine_nine_pons_achievable_l2028_202816


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2028_202856

theorem absolute_value_inequality (x : ℝ) : |x - 2| > 2 - x ↔ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2028_202856


namespace NUMINAMATH_CALUDE_star_replacement_impossibility_l2028_202871

theorem star_replacement_impossibility : ∀ (f : Fin 9 → Bool),
  ∃ (result : ℤ), result ≠ 0 ∧
  result = (if f 0 then 1 else -1) +
           (if f 1 then 2 else -2) +
           (if f 2 then 3 else -3) +
           (if f 3 then 4 else -4) +
           (if f 4 then 5 else -5) +
           (if f 5 then 6 else -6) +
           (if f 6 then 7 else -7) +
           (if f 7 then 8 else -8) +
           (if f 8 then 9 else -9) +
           10 :=
by
  sorry

end NUMINAMATH_CALUDE_star_replacement_impossibility_l2028_202871


namespace NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l2028_202884

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents the original plywood and its cuts -/
structure Plywood where
  length : ℝ
  width : ℝ
  num_pieces : ℕ

/-- Checks if a rectangle is a valid cut of the plywood -/
def is_valid_cut (p : Plywood) (r : Rectangle) : Prop :=
  p.length * p.width = p.num_pieces * r.length * r.width

theorem plywood_cut_perimeter_difference (p : Plywood) :
  p.length = 6 ∧ p.width = 9 ∧ p.num_pieces = 6 →
  ∃ (max_r min_r : Rectangle),
    is_valid_cut p max_r ∧
    is_valid_cut p min_r ∧
    ∀ (r : Rectangle), is_valid_cut p r →
      perimeter r ≤ perimeter max_r ∧
      perimeter min_r ≤ perimeter r ∧
      perimeter max_r - perimeter min_r = 10 := by
  sorry

end NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l2028_202884


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l2028_202850

/-- In a Cartesian coordinate system, if a point P has coordinates (3, -5),
    then its coordinates with respect to the origin are also (3, -5). -/
theorem point_coordinates_wrt_origin :
  ∀ (P : ℝ × ℝ), P = (3, -5) → P = (3, -5) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l2028_202850


namespace NUMINAMATH_CALUDE_hyperbola_dot_product_theorem_l2028_202817

/-- The hyperbola in the Cartesian coordinate system -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

/-- The left focus of the hyperbola -/
def F₁ : ℝ × ℝ := (-2, 0)

/-- The right focus of the hyperbola -/
def F₂ : ℝ × ℝ := (2, 0)

/-- A point on the hyperbola -/
structure HyperbolaPoint where
  x : ℝ
  y : ℝ
  on_hyperbola : hyperbola x y

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The vector from one point to another -/
def vector (a b : ℝ × ℝ) : ℝ × ℝ := (b.1 - a.1, b.2 - a.2)

/-- The theorem to be proved -/
theorem hyperbola_dot_product_theorem 
  (P Q : HyperbolaPoint) 
  (h_line : ∃ (m b : ℝ), P.y = m * P.x + b ∧ Q.y = m * Q.x + b ∧ F₁.2 = m * F₁.1 + b) 
  (h_dot_product : dot_product (vector F₁ F₂) (vector F₁ (P.x, P.y)) = 16) :
  dot_product (vector F₂ (P.x, P.y)) (vector F₂ (Q.x, Q.y)) = 27 / 13 := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_dot_product_theorem_l2028_202817


namespace NUMINAMATH_CALUDE_integral_equation_solution_l2028_202869

theorem integral_equation_solution (k : ℝ) : 
  (∫ (x : ℝ), 2*x - 3*x^2) = 0 → k = 0 ∨ k = 1 :=
by sorry

end NUMINAMATH_CALUDE_integral_equation_solution_l2028_202869


namespace NUMINAMATH_CALUDE_students_with_both_pets_l2028_202840

theorem students_with_both_pets (total : ℕ) (dog : ℕ) (cat : ℕ) (no_pet : ℕ) 
  (h_total : total = 50)
  (h_dog : dog = 30)
  (h_cat : cat = 35)
  (h_no_pet : no_pet = 3)
  (h_at_least_one : ∀ s, s ∈ Finset.range total → 
    (s ∈ Finset.range dog ∨ s ∈ Finset.range cat ∨ s ∈ Finset.range no_pet)) :
  Finset.card (Finset.range dog ∩ Finset.range cat) = 18 := by
  sorry

end NUMINAMATH_CALUDE_students_with_both_pets_l2028_202840


namespace NUMINAMATH_CALUDE_lemonade_sum_l2028_202832

theorem lemonade_sum : 
  let first_intermission : Float := 0.25
  let second_intermission : Float := 0.4166666666666667
  let third_intermission : Float := 0.25
  first_intermission + second_intermission + third_intermission = 0.9166666666666667 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_sum_l2028_202832


namespace NUMINAMATH_CALUDE_exists_captivating_number_l2028_202808

/-- A function that checks if a list of digits forms a captivating number -/
def is_captivating (digits : List Nat) : Prop :=
  digits.length = 7 ∧
  digits.toFinset = Finset.range 7 ∧
  ∀ k : Nat, k ∈ Finset.range 7 → 
    (digits.take k).foldl (fun acc d => acc * 10 + d) 0 % (k + 1) = 0

/-- Theorem stating the existence of at least one captivating number -/
theorem exists_captivating_number : ∃ digits : List Nat, is_captivating digits :=
  sorry

end NUMINAMATH_CALUDE_exists_captivating_number_l2028_202808


namespace NUMINAMATH_CALUDE_forest_ecosystem_l2028_202851

/-- Given a forest ecosystem where:
    - Each bird eats 12 beetles per day
    - Each snake eats 3 birds per day
    - Each jaguar eats 5 snakes per day
    - The jaguars in the forest eat 1080 beetles each day
    This theorem proves that there are 6 jaguars in the forest. -/
theorem forest_ecosystem (beetles_per_bird : ℕ) (birds_per_snake : ℕ) (snakes_per_jaguar : ℕ) 
                         (total_beetles_eaten : ℕ) : ℕ :=
  sorry

end NUMINAMATH_CALUDE_forest_ecosystem_l2028_202851


namespace NUMINAMATH_CALUDE_farmhouse_blocks_l2028_202876

def total_blocks : ℕ := 344
def building_blocks : ℕ := 80
def fenced_area_blocks : ℕ := 57
def leftover_blocks : ℕ := 84

theorem farmhouse_blocks :
  total_blocks - building_blocks - fenced_area_blocks - leftover_blocks = 123 := by
  sorry

end NUMINAMATH_CALUDE_farmhouse_blocks_l2028_202876


namespace NUMINAMATH_CALUDE_prob_red_is_three_tenths_l2028_202889

-- Define the contents of the bags
def bag_A : Finset (Fin 10) := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def bag_B : Finset (Fin 10) := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the colors of the balls
inductive Color
| Red
| White
| Black

-- Define the color distribution in bag A
def color_A : Fin 10 → Color
| 0 | 1 | 2 => Color.Red
| 3 | 4 => Color.White
| _ => Color.Black

-- Define the color distribution in bag B
def color_B : Fin 10 → Color
| 0 | 1 | 2 => Color.Red
| 3 | 4 | 5 => Color.White
| _ => Color.Black

-- Define the probability of drawing a red ball from bag B after transfer
def prob_red_after_transfer : ℚ :=
  3 / 10

-- Theorem statement
theorem prob_red_is_three_tenths :
  prob_red_after_transfer = 3 / 10 := by
  sorry


end NUMINAMATH_CALUDE_prob_red_is_three_tenths_l2028_202889


namespace NUMINAMATH_CALUDE_chi_square_significant_distribution_prob_sum_to_one_l2028_202881

/-- Represents the total number of students surveyed -/
def total_students : ℕ := 2000

/-- Represents the percentage of students with myopia -/
def myopia_rate : ℚ := 2/5

/-- Represents the percentage of students spending more than 1 hour on phones daily -/
def long_phone_usage_rate : ℚ := 1/5

/-- Represents the myopia rate among students who spend more than 1 hour on phones daily -/
def myopia_rate_long_usage : ℚ := 1/2

/-- Represents the significance level for the Chi-square test -/
def significance_level : ℚ := 1/1000

/-- Represents the critical value for the Chi-square test at α = 0.001 -/
def critical_value : ℚ := 10828/1000

/-- Represents the number of myopic students randomly selected -/
def selected_myopic_students : ℕ := 8

/-- Represents the number of selected myopic students spending more than 1 hour on phones -/
def long_usage_in_selection : ℕ := 2

/-- Represents the number of students randomly selected from the 8 myopic students -/
def final_selection : ℕ := 3

/-- Calculates the Chi-square value for the given data -/
def chi_square_value : ℚ := 20833/1000

/-- Theorem stating that the Chi-square value is greater than the critical value -/
theorem chi_square_significant : chi_square_value > critical_value := by sorry

/-- Calculates the probability of X = 0 in the distribution table -/
def prob_X_0 : ℚ := 5/14

/-- Calculates the probability of X = 1 in the distribution table -/
def prob_X_1 : ℚ := 15/28

/-- Calculates the probability of X = 2 in the distribution table -/
def prob_X_2 : ℚ := 3/28

/-- Theorem stating that the probabilities in the distribution table sum up to 1 -/
theorem distribution_prob_sum_to_one : prob_X_0 + prob_X_1 + prob_X_2 = 1 := by sorry

end NUMINAMATH_CALUDE_chi_square_significant_distribution_prob_sum_to_one_l2028_202881


namespace NUMINAMATH_CALUDE_combined_molecular_weight_l2028_202862

-- Define atomic weights
def carbon_weight : ℝ := 12.01
def chlorine_weight : ℝ := 35.45
def sulfur_weight : ℝ := 32.07
def fluorine_weight : ℝ := 19.00

-- Define molecular compositions
def ccl4_carbon_count : ℕ := 1
def ccl4_chlorine_count : ℕ := 4
def sf6_sulfur_count : ℕ := 1
def sf6_fluorine_count : ℕ := 6

-- Define number of moles
def ccl4_moles : ℕ := 9
def sf6_moles : ℕ := 5

-- Theorem statement
theorem combined_molecular_weight :
  let ccl4_weight := carbon_weight * ccl4_carbon_count + chlorine_weight * ccl4_chlorine_count
  let sf6_weight := sulfur_weight * sf6_sulfur_count + fluorine_weight * sf6_fluorine_count
  ccl4_weight * ccl4_moles + sf6_weight * sf6_moles = 2114.64 := by
  sorry

end NUMINAMATH_CALUDE_combined_molecular_weight_l2028_202862


namespace NUMINAMATH_CALUDE_sqrt_6_simplest_l2028_202894

def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y z : ℝ, y * y = x → z * z = x → y = z ∨ y = -z

theorem sqrt_6_simplest :
  is_simplest_sqrt 6 ∧
  ¬ is_simplest_sqrt 8 ∧
  ¬ is_simplest_sqrt 12 ∧
  ¬ is_simplest_sqrt 0.3 :=
sorry

end NUMINAMATH_CALUDE_sqrt_6_simplest_l2028_202894


namespace NUMINAMATH_CALUDE_average_of_multiples_of_10_l2028_202820

def multiples_of_10 : List ℕ := [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

theorem average_of_multiples_of_10 : 
  (List.sum multiples_of_10) / (List.length multiples_of_10) = 55 := by
  sorry

end NUMINAMATH_CALUDE_average_of_multiples_of_10_l2028_202820


namespace NUMINAMATH_CALUDE_coefficient_x3y5_l2028_202828

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℚ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the expression
def expression (x y : ℚ) : ℚ := (2/3 * x - 4/5 * y)^8

-- State the theorem
theorem coefficient_x3y5 :
  (binomial 8 3) * (2/3)^3 * (-4/5)^5 = -458752/84375 := by sorry

end NUMINAMATH_CALUDE_coefficient_x3y5_l2028_202828


namespace NUMINAMATH_CALUDE_smallest_number_l2028_202813

theorem smallest_number : ∀ (a b c d : ℝ), 
  a = -2023 → b = 0 → c = 0.999 → d = 1 →
  a < b ∧ a < c ∧ a < d :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l2028_202813


namespace NUMINAMATH_CALUDE_least_possible_bc_length_l2028_202888

theorem least_possible_bc_length 
  (AB AC DC BD : ℝ) 
  (hAB : AB = 8) 
  (hAC : AC = 10) 
  (hDC : DC = 7) 
  (hBD : BD = 15) : 
  ∃ (BC : ℕ), BC = 9 ∧ 
    BC > AC - AB ∧ 
    BC > BD - DC ∧ 
    ∀ (n : ℕ), n < 9 → (n ≤ AC - AB ∨ n ≤ BD - DC) :=
by sorry

end NUMINAMATH_CALUDE_least_possible_bc_length_l2028_202888


namespace NUMINAMATH_CALUDE_rudys_running_time_l2028_202838

/-- Calculates the total running time for Rudy given two separate runs -/
def totalRunningTime (distance1 : ℝ) (rate1 : ℝ) (distance2 : ℝ) (rate2 : ℝ) : ℝ :=
  distance1 * rate1 + distance2 * rate2

/-- Proves that Rudy's total running time is 88 minutes -/
theorem rudys_running_time :
  totalRunningTime 5 10 4 9.5 = 88 := by
  sorry

end NUMINAMATH_CALUDE_rudys_running_time_l2028_202838


namespace NUMINAMATH_CALUDE_square_rectangle_area_relationship_l2028_202865

theorem square_rectangle_area_relationship : 
  ∃ (x₁ x₂ : ℝ), 
    (∀ x : ℝ, 3 * (x - 2)^2 = (x - 3) * (x + 4) → x = x₁ ∨ x = x₂) ∧
    x₁ + x₂ = 19/2 := by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_area_relationship_l2028_202865


namespace NUMINAMATH_CALUDE_sin_shift_l2028_202883

open Real

theorem sin_shift (x : ℝ) :
  sin (3 * (x - π / 12)) = sin (3 * x - π / 4) := by sorry

end NUMINAMATH_CALUDE_sin_shift_l2028_202883


namespace NUMINAMATH_CALUDE_walker_round_trip_l2028_202810

/-- Ms. Walker's round trip driving problem -/
theorem walker_round_trip (speed_to_work : ℝ) (speed_from_work : ℝ) (total_time : ℝ) 
  (h1 : speed_to_work = 60)
  (h2 : speed_from_work = 40)
  (h3 : total_time = 1) :
  ∃ (distance : ℝ), distance / speed_to_work + distance / speed_from_work = total_time ∧ distance = 24 := by
  sorry

end NUMINAMATH_CALUDE_walker_round_trip_l2028_202810


namespace NUMINAMATH_CALUDE_smaller_is_999_l2028_202846

/-- Two 3-digit positive integers whose average equals their decimal concatenation -/
structure SpecialIntegerPair where
  m : ℕ
  n : ℕ
  m_three_digit : 100 ≤ m ∧ m ≤ 999
  n_three_digit : 100 ≤ n ∧ n ≤ 999
  avg_eq_concat : (m + n) / 2 = m + n / 1000

/-- The smaller of the two integers in a SpecialIntegerPair is 999 -/
theorem smaller_is_999 (pair : SpecialIntegerPair) : min pair.m pair.n = 999 := by
  sorry

end NUMINAMATH_CALUDE_smaller_is_999_l2028_202846


namespace NUMINAMATH_CALUDE_triangle_formation_condition_l2028_202826

theorem triangle_formation_condition (a b : ℝ) : 
  (∃ (c : ℝ), c = 1 ∧ a + b + c = 2) →
  (a + b > c ∧ a + c > b ∧ b + c > a) ↔ (a + b = 1 ∧ a ≥ 0 ∧ b ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_condition_l2028_202826


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l2028_202853

/-- The minimum distance between two points on different curves -/
theorem min_distance_between_curves (m : ℝ) : 
  let A := {x : ℝ | ∃ y, y = m ∧ y = 2 * (x + 1)}
  let B := {x : ℝ | ∃ y, y = m ∧ y = x + Real.log x}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ A ∧ x₂ ∈ B ∧ 
    (∀ (a b : ℝ), a ∈ A → b ∈ B → |x₂ - x₁| ≤ |b - a|) ∧
    |x₂ - x₁| = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l2028_202853


namespace NUMINAMATH_CALUDE_sum_a_d_l2028_202801

theorem sum_a_d (a b c d : ℤ) 
  (eq1 : a + b = 14) 
  (eq2 : b + c = 9) 
  (eq3 : c + d = 3) : 
  a + d = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_a_d_l2028_202801


namespace NUMINAMATH_CALUDE_principal_calculation_l2028_202887

/-- Proves that the principal amount is 1500 given the specified conditions --/
theorem principal_calculation (rate : ℝ) (time : ℝ) (amount : ℝ) :
  rate = 0.05 →
  time = 2.4 →
  amount = 1680 →
  (1 + rate * time) * 1500 = amount :=
by sorry

end NUMINAMATH_CALUDE_principal_calculation_l2028_202887


namespace NUMINAMATH_CALUDE_equal_integers_from_gcd_l2028_202821

theorem equal_integers_from_gcd (a b : ℤ) 
  (h : ∀ (n : ℤ), n ≥ 1 → Nat.gcd (Int.natAbs (a + n)) (Int.natAbs (b + n)) > 1) : 
  a = b := by
  sorry

end NUMINAMATH_CALUDE_equal_integers_from_gcd_l2028_202821


namespace NUMINAMATH_CALUDE_prove_b_value_l2028_202800

theorem prove_b_value (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 315 * b) : b = 7 := by
  sorry

end NUMINAMATH_CALUDE_prove_b_value_l2028_202800


namespace NUMINAMATH_CALUDE_function_equality_implies_sum_l2028_202802

theorem function_equality_implies_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f (x - 2) = 4 * x^2 + 9 * x + 5) →
  (∀ x, f x = a * x^2 + b * x + c) →
  a + b + c = 68 := by
sorry

end NUMINAMATH_CALUDE_function_equality_implies_sum_l2028_202802


namespace NUMINAMATH_CALUDE_salary_calculation_l2028_202873

def monthly_salary : ℝ → Prop := λ s => 
  let original_savings := 0.2 * s
  let original_expenses := 0.8 * s
  let increased_expenses := 1.2 * original_expenses
  s - increased_expenses = 250

theorem salary_calculation : ∃ s : ℝ, monthly_salary s ∧ s = 6250 := by
  sorry

end NUMINAMATH_CALUDE_salary_calculation_l2028_202873


namespace NUMINAMATH_CALUDE_joans_total_seashells_l2028_202872

/-- The number of seashells Joan found on the beach -/
def initial_seashells : ℕ := 70

/-- The number of seashells Sam gave to Joan -/
def additional_seashells : ℕ := 27

/-- Theorem: Joan's total number of seashells is 97 -/
theorem joans_total_seashells : initial_seashells + additional_seashells = 97 := by
  sorry

end NUMINAMATH_CALUDE_joans_total_seashells_l2028_202872


namespace NUMINAMATH_CALUDE_log_equation_implies_y_equals_nine_l2028_202864

theorem log_equation_implies_y_equals_nine 
  (x y : ℝ) 
  (h : x > 0) 
  (h2x : 2*x > 0) 
  (hy : y > 0) : 
  (Real.log x / Real.log 3) * (Real.log (2*x) / Real.log x) * (Real.log y / Real.log (2*x)) = 2 → 
  y = 9 := by
sorry

end NUMINAMATH_CALUDE_log_equation_implies_y_equals_nine_l2028_202864


namespace NUMINAMATH_CALUDE_parallel_vectors_dot_product_l2028_202824

def a (m n : ℝ) : Fin 3 → ℝ := ![1, 3*m - 1, n - 2]
def b (m n : ℝ) : Fin 3 → ℝ := ![2, 3*m + 1, 3*n - 4]

def parallel (u v : Fin 3 → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ (i : Fin 3), u i = k * v i

def dot_product (u v : Fin 3 → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1) + (u 2) * (v 2)

theorem parallel_vectors_dot_product (m n : ℝ) :
  parallel (a m n) (b m n) → dot_product (a m n) (b m n) = 18 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_dot_product_l2028_202824


namespace NUMINAMATH_CALUDE_cos_150_degrees_l2028_202844

theorem cos_150_degrees :
  Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 :=
by
  -- Define the cosine subtraction identity
  have cos_subtraction_identity (a b : ℝ) :
    Real.cos (a - b) = Real.cos a * Real.cos b + Real.sin a * Real.sin b :=
    sorry

  -- Express 150° as 180° - 30°
  have h1 : 150 * π / 180 = π - (30 * π / 180) :=
    sorry

  -- Use the cosine subtraction identity
  have h2 : Real.cos (150 * π / 180) =
    Real.cos π * Real.cos (30 * π / 180) + Real.sin π * Real.sin (30 * π / 180) :=
    sorry

  -- Evaluate the expression
  sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l2028_202844


namespace NUMINAMATH_CALUDE_quadratic_root_product_l2028_202842

theorem quadratic_root_product (p q : ℝ) : 
  (∃ x : ℂ, x^2 + p*x + q = 0 ∧ x = 3 - 4*Complex.I) → p*q = -150 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_product_l2028_202842


namespace NUMINAMATH_CALUDE_parallelogram_smaller_angle_l2028_202818

theorem parallelogram_smaller_angle (smaller_angle larger_angle : ℝ) : 
  larger_angle = smaller_angle + 90 →
  smaller_angle + larger_angle = 180 →
  smaller_angle = 45 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_smaller_angle_l2028_202818


namespace NUMINAMATH_CALUDE_optimal_distribution_minimizes_cost_l2028_202848

noncomputable section

/-- Represents the distribution of potatoes among three farms -/
structure PotatoDistribution where
  farm1 : ℝ
  farm2 : ℝ
  farm3 : ℝ

/-- The cost function for potato distribution -/
def cost (d : PotatoDistribution) : ℝ :=
  4 * d.farm1 + 3 * d.farm2 + d.farm3

/-- Checks if a distribution satisfies all constraints -/
def isValid (d : PotatoDistribution) : Prop :=
  d.farm1 ≥ 0 ∧ d.farm2 ≥ 0 ∧ d.farm3 ≥ 0 ∧
  d.farm1 + d.farm2 + d.farm3 = 12 ∧
  d.farm1 + 4 * d.farm2 + 3 * d.farm3 ≤ 40 ∧
  d.farm1 ≤ 10 ∧ d.farm2 ≤ 8 ∧ d.farm3 ≤ 6

/-- The optimal distribution of potatoes -/
def optimalDistribution : PotatoDistribution :=
  { farm1 := 2/3, farm2 := 16/3, farm3 := 6 }

/-- Theorem stating that the optimal distribution minimizes the cost -/
theorem optimal_distribution_minimizes_cost :
  isValid optimalDistribution ∧
  ∀ d : PotatoDistribution, isValid d → cost optimalDistribution ≤ cost d :=
sorry

end

end NUMINAMATH_CALUDE_optimal_distribution_minimizes_cost_l2028_202848


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2028_202899

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^3 + 4 * x^2 + 9 * x - 5) - (2 * x^3 + 2 * x^2 + 6 * x - 18) = 
  x^3 + 2 * x^2 + 3 * x + 13 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2028_202899


namespace NUMINAMATH_CALUDE_min_cost_grass_seed_l2028_202830

/-- Represents a bag of grass seed -/
structure SeedBag where
  weight : Nat
  price : Rat

/-- Calculates the total weight of a list of seed bags -/
def totalWeight (bags : List SeedBag) : Nat :=
  bags.foldl (fun acc bag => acc + bag.weight) 0

/-- Calculates the total price of a list of seed bags -/
def totalPrice (bags : List SeedBag) : Rat :=
  bags.foldl (fun acc bag => acc + bag.price) 0

/-- Theorem: The minimum cost to buy between 65 and 80 pounds of grass seed is $98.75 -/
theorem min_cost_grass_seed (bag5 bag10 bag25 : SeedBag)
    (h1 : bag5.weight = 5 ∧ bag5.price = 1385 / 100)
    (h2 : bag10.weight = 10 ∧ bag10.price = 2040 / 100)
    (h3 : bag25.weight = 25 ∧ bag25.price = 3225 / 100) :
    ∃ (bags : List SeedBag),
      (totalWeight bags ≥ 65 ∧ totalWeight bags ≤ 80) ∧
      totalPrice bags = 9875 / 100 ∧
      ∀ (other_bags : List SeedBag),
        (totalWeight other_bags ≥ 65 ∧ totalWeight other_bags ≤ 80) →
        totalPrice other_bags ≥ 9875 / 100 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_grass_seed_l2028_202830


namespace NUMINAMATH_CALUDE_original_tomatoes_cost_l2028_202878

def original_order : ℝ := 25
def new_tomatoes : ℝ := 2.20
def old_lettuce : ℝ := 1.00
def new_lettuce : ℝ := 1.75
def old_celery : ℝ := 1.96
def new_celery : ℝ := 2.00
def delivery_tip : ℝ := 8.00
def new_total : ℝ := 35

theorem original_tomatoes_cost (x : ℝ) : 
  x = 3.41 ↔ 
  x + old_lettuce + old_celery + delivery_tip = new_total ∧
  new_tomatoes + new_lettuce + new_celery + delivery_tip = new_total :=
by sorry

end NUMINAMATH_CALUDE_original_tomatoes_cost_l2028_202878


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l2028_202879

def f (x : ℝ) := 3 * x^3 - 9 * x + 5

theorem f_max_min_on_interval :
  ∃ (max min : ℝ) (x_max x_min : ℝ),
    x_max ∈ Set.Icc (-3 : ℝ) 3 ∧
    x_min ∈ Set.Icc (-3 : ℝ) 3 ∧
    (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≥ f x_min) ∧
    f x_max = max ∧
    f x_min = min ∧
    max = 59 ∧
    min = -49 ∧
    x_max = 3 ∧
    x_min = -3 :=
by sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l2028_202879


namespace NUMINAMATH_CALUDE_cricket_players_l2028_202882

/-- The number of students who like to play basketball -/
def B : ℕ := 12

/-- The number of students who like to play both basketball and cricket -/
def B_and_C : ℕ := 3

/-- The number of students who like to play basketball or cricket or both -/
def B_or_C : ℕ := 17

/-- The number of students who like to play cricket -/
def C : ℕ := B_or_C - B + B_and_C

theorem cricket_players : C = 8 := by
  sorry

end NUMINAMATH_CALUDE_cricket_players_l2028_202882


namespace NUMINAMATH_CALUDE_mendez_family_mean_age_l2028_202861

/-- The Mendez family children problem -/
theorem mendez_family_mean_age :
  let ages : List ℝ := [5, 5, 10, 12, 15]
  let mean_age := (ages.sum) / (ages.length : ℝ)
  mean_age = 9.4 := by
sorry

end NUMINAMATH_CALUDE_mendez_family_mean_age_l2028_202861


namespace NUMINAMATH_CALUDE_right_triangle_probability_l2028_202841

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Finset (ℕ × ℕ)
  regular : vertices.card = 8
  -- Additional properties of a regular octagon could be added here

/-- A triangle formed by three vertices of a regular octagon -/
structure OctagonTriangle (octagon : RegularOctagon) where
  vertices : Finset (ℕ × ℕ)
  subset : vertices ⊆ octagon.vertices
  three_points : vertices.card = 3

/-- Predicate to determine if a triangle is right-angled -/
def is_right_triangle (triangle : OctagonTriangle octagon) : Prop :=
  sorry -- Definition of a right triangle in terms of the octagon's geometry

/-- The set of all possible triangles from an octagon -/
def all_triangles (octagon : RegularOctagon) : Finset (OctagonTriangle octagon) :=
  sorry

/-- The set of right triangles from an octagon -/
def right_triangles (octagon : RegularOctagon) : Finset (OctagonTriangle octagon) :=
  sorry

/-- The main theorem -/
theorem right_triangle_probability (octagon : RegularOctagon) :
  (right_triangles octagon).card / (all_triangles octagon).card = 2 / 7 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_probability_l2028_202841


namespace NUMINAMATH_CALUDE_isoscelesTrapezoidArea_l2028_202839

/-- An isosceles trapezoid inscribed around a circle -/
structure IsoscelesTrapezoid where
  /-- Length of the longer base -/
  longerBase : ℝ
  /-- One of the base angles in radians -/
  baseAngle : ℝ
  /-- Assumption that the longer base is 18 -/
  longerBaseIs18 : longerBase = 18
  /-- Assumption that the base angle is arccos(0.6) -/
  baseAngleIsArccos06 : baseAngle = Real.arccos 0.6

/-- The area of the isosceles trapezoid -/
def areaOfTrapezoid (t : IsoscelesTrapezoid) : ℝ := 101.25

/-- Theorem stating that the area of the isosceles trapezoid is 101.25 -/
theorem isoscelesTrapezoidArea (t : IsoscelesTrapezoid) : 
  areaOfTrapezoid t = 101.25 := by
  sorry

end NUMINAMATH_CALUDE_isoscelesTrapezoidArea_l2028_202839


namespace NUMINAMATH_CALUDE_line_slope_problem_l2028_202875

theorem line_slope_problem (a : ℝ) : 
  (3 * a - 7) / (a - 2) = 2 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_problem_l2028_202875


namespace NUMINAMATH_CALUDE_octal_734_eq_476_l2028_202835

-- Define the octal number as a list of digits
def octal_number : List Nat := [7, 3, 4]

-- Define the function to convert octal to decimal
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

-- Theorem statement
theorem octal_734_eq_476 :
  octal_to_decimal octal_number = 476 := by
  sorry

end NUMINAMATH_CALUDE_octal_734_eq_476_l2028_202835


namespace NUMINAMATH_CALUDE_prob_different_colors_l2028_202854

/-- Probability of drawing two different colored chips -/
theorem prob_different_colors (blue yellow red : ℕ) 
  (h_blue : blue = 6)
  (h_yellow : yellow = 4)
  (h_red : red = 2) :
  let total := blue + yellow + red
  (blue * yellow + blue * red + yellow * red) * 2 / (total * total) = 11 / 18 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_colors_l2028_202854


namespace NUMINAMATH_CALUDE_hyperbola_slope_product_l2028_202849

/-- Hyperbola theorem -/
theorem hyperbola_slope_product (a b x₀ y₀ : ℝ) (ha : a > 0) (hb : b > 0) 
  (hp : x₀^2 / a^2 - y₀^2 / b^2 = 1) (hx : x₀ ≠ a ∧ x₀ ≠ -a) : 
  (y₀ / (x₀ + a)) * (y₀ / (x₀ - a)) = b^2 / a^2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_slope_product_l2028_202849


namespace NUMINAMATH_CALUDE_largest_house_number_l2028_202823

/-- The sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Check if all digits in a natural number are distinct -/
def has_distinct_digits (n : ℕ) : Prop := sorry

/-- The largest 3-digit number with distinct digits whose sum equals the sum of digits in 5039821 -/
theorem largest_house_number : 
  ∃ (n : ℕ), 
    100 ≤ n ∧ n < 1000 ∧ 
    has_distinct_digits n ∧
    digit_sum n = digit_sum 5039821 ∧
    ∀ (m : ℕ), 100 ≤ m ∧ m < 1000 ∧ 
      has_distinct_digits m ∧ 
      digit_sum m = digit_sum 5039821 → 
      m ≤ n ∧
    n = 981 := by sorry

end NUMINAMATH_CALUDE_largest_house_number_l2028_202823


namespace NUMINAMATH_CALUDE_max_cookies_bound_l2028_202812

def num_jars : Nat := 2023

/-- Represents the state of cookie jars -/
def JarState := Fin num_jars → Nat

/-- Elmo's action of adding cookies to two distinct jars -/
def elmo_action (state : JarState) : JarState := sorry

/-- Cookie Monster's action of eating cookies from the jar with the most cookies -/
def monster_action (state : JarState) : JarState := sorry

/-- One complete cycle of Elmo's and Cookie Monster's actions -/
def cycle (state : JarState) : JarState := monster_action (elmo_action state)

/-- The maximum number of cookies in any jar -/
def max_cookies (state : JarState) : Nat :=
  Finset.sup (Finset.univ : Finset (Fin num_jars)) (fun i => state i)

theorem max_cookies_bound (initial_state : JarState) :
  ∀ n : Nat, max_cookies ((cycle^[n]) initial_state) ≤ 12 := by sorry

end NUMINAMATH_CALUDE_max_cookies_bound_l2028_202812


namespace NUMINAMATH_CALUDE_dividend_calculation_l2028_202858

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 36)
  (h2 : quotient = 19)
  (h3 : remainder = 2) : 
  divisor * quotient + remainder = 686 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2028_202858


namespace NUMINAMATH_CALUDE_point_inside_ellipse_l2028_202860

theorem point_inside_ellipse (a : ℝ) : 
  (a^2 / 4 + 1 / 2 < 1) → (-Real.sqrt 2 < a ∧ a < Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_point_inside_ellipse_l2028_202860


namespace NUMINAMATH_CALUDE_chapter_page_difference_l2028_202836

theorem chapter_page_difference (chapter1 chapter2 chapter3 : ℕ) 
  (h1 : chapter1 = 35)
  (h2 : chapter2 = 18)
  (h3 : chapter3 = 3) :
  chapter2 - chapter3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_chapter_page_difference_l2028_202836


namespace NUMINAMATH_CALUDE_sin_180_degrees_l2028_202831

/-- The sine of 180 degrees is 0. -/
theorem sin_180_degrees : Real.sin (π) = 0 := by sorry

end NUMINAMATH_CALUDE_sin_180_degrees_l2028_202831
