import Mathlib

namespace NUMINAMATH_CALUDE_negative_two_b_cubed_l2811_281137

theorem negative_two_b_cubed (b : ℝ) : (-2 * b)^3 = -8 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_b_cubed_l2811_281137


namespace NUMINAMATH_CALUDE_conic_intersection_lines_concurrent_l2811_281158

-- Define the type for a conic
def Conic := Type

-- Define the type for a point
def Point := Type

-- Define the type for a line
def Line := Type

-- Define a function to check if a point is on a conic
def point_on_conic (p : Point) (c : Conic) : Prop := sorry

-- Define a function to create a line from two points
def line_through_points (p q : Point) : Line := sorry

-- Define a function to check if three lines are concurrent
def are_concurrent (l₁ l₂ l₃ : Line) : Prop := sorry

-- Define the theorem
theorem conic_intersection_lines_concurrent 
  (𝓔₁ 𝓔₂ 𝓔₃ : Conic) 
  (A B : Point) 
  (h_common : point_on_conic A 𝓔₁ ∧ point_on_conic A 𝓔₂ ∧ point_on_conic A 𝓔₃ ∧
              point_on_conic B 𝓔₁ ∧ point_on_conic B 𝓔₂ ∧ point_on_conic B 𝓔₃)
  (C D E F G H : Point)
  (h_intersections : point_on_conic C 𝓔₁ ∧ point_on_conic C 𝓔₂ ∧
                     point_on_conic D 𝓔₁ ∧ point_on_conic D 𝓔₂ ∧
                     point_on_conic E 𝓔₁ ∧ point_on_conic E 𝓔₃ ∧
                     point_on_conic F 𝓔₁ ∧ point_on_conic F 𝓔₃ ∧
                     point_on_conic G 𝓔₂ ∧ point_on_conic G 𝓔₃ ∧
                     point_on_conic H 𝓔₂ ∧ point_on_conic H 𝓔₃)
  (ℓ₁₂ := line_through_points C D)
  (ℓ₁₃ := line_through_points E F)
  (ℓ₂₃ := line_through_points G H) :
  are_concurrent ℓ₁₂ ℓ₁₃ ℓ₂₃ := by
  sorry

end NUMINAMATH_CALUDE_conic_intersection_lines_concurrent_l2811_281158


namespace NUMINAMATH_CALUDE_no_positive_integers_satisfying_conditions_l2811_281101

theorem no_positive_integers_satisfying_conditions : 
  ¬ ∃ (a b c d : ℕ+) (p : ℕ), 
    (a.val * b.val = c.val * d.val) ∧ 
    (a.val + b.val + c.val + d.val = p) ∧ 
    Nat.Prime p :=
by sorry

end NUMINAMATH_CALUDE_no_positive_integers_satisfying_conditions_l2811_281101


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2811_281112

/-- An arithmetic sequence. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem stating that a_15 = 24 for the given arithmetic sequence. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arithmetic : ArithmeticSequence a)
    (h_sum : a 3 + a 13 = 20)
    (h_a2 : a 2 = -2) : 
  a 15 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2811_281112


namespace NUMINAMATH_CALUDE_factorial_inequality_l2811_281148

theorem factorial_inequality (n : ℕ) (h : n ≥ 1) : n.factorial ≤ ((n + 1) / 2 : ℝ) ^ n := by
  sorry

end NUMINAMATH_CALUDE_factorial_inequality_l2811_281148


namespace NUMINAMATH_CALUDE_average_speed_is_69_l2811_281144

def speeds : List ℝ := [90, 30, 60, 120, 45]
def total_time : ℝ := 5

theorem average_speed_is_69 :
  (speeds.sum / total_time) = 69 := by sorry

end NUMINAMATH_CALUDE_average_speed_is_69_l2811_281144


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_union_B_C_implies_a_bound_l2811_281193

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2*x - 4 ≥ x - 2}
def C (a : ℝ) : Set ℝ := {x | 2*x + a > 0}

-- Theorem for part (1)
theorem complement_intersection_A_B :
  (A ∩ B)ᶜ = {x | x < 2 ∨ x ≥ 3} := by sorry

-- Theorem for part (2)
theorem union_B_C_implies_a_bound (a : ℝ) :
  B ∪ C a = C a → a > -4 := by sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_union_B_C_implies_a_bound_l2811_281193


namespace NUMINAMATH_CALUDE_min_root_product_sum_l2811_281140

def f (x : ℝ) : ℝ := x^4 + 14*x^3 + 52*x^2 + 56*x + 16

theorem min_root_product_sum (z₁ z₂ z₃ z₄ : ℝ) 
  (hroots : (∀ x, f x = 0 ↔ x = z₁ ∨ x = z₂ ∨ x = z₃ ∨ x = z₄)) :
  (∀ (σ : Equiv.Perm (Fin 4)), 
    |z₁ * z₂ + z₃ * z₄| ≥ 8 ∧
    |z₁ * z₃ + z₂ * z₄| ≥ 8 ∧
    |z₁ * z₄ + z₂ * z₃| ≥ 8) ∧
  (∃ (σ : Equiv.Perm (Fin 4)), 
    |z₁ * z₂ + z₃ * z₄| = 8 ∨
    |z₁ * z₃ + z₂ * z₄| = 8 ∨
    |z₁ * z₄ + z₂ * z₃| = 8) :=
by sorry

end NUMINAMATH_CALUDE_min_root_product_sum_l2811_281140


namespace NUMINAMATH_CALUDE_right_triangle_min_leg_sum_l2811_281134

theorem right_triangle_min_leg_sum (a b : ℝ) (h_right : a > 0 ∧ b > 0) (h_area : (1/2) * a * b = 50) :
  a + b ≥ 20 ∧ (a + b = 20 ↔ a = 10 ∧ b = 10) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_min_leg_sum_l2811_281134


namespace NUMINAMATH_CALUDE_prob_three_primes_in_six_dice_l2811_281191

-- Define a 12-sided die
def twelve_sided_die : Finset ℕ := Finset.range 12

-- Define prime numbers on a 12-sided die
def primes_on_die : Finset ℕ := {2, 3, 5, 7, 11}

-- Define the probability of rolling a prime on a single die
def prob_prime : ℚ := (primes_on_die.card : ℚ) / (twelve_sided_die.card : ℚ)

-- Define the probability of rolling a non-prime on a single die
def prob_non_prime : ℚ := 1 - prob_prime

-- Define the number of dice
def num_dice : ℕ := 6

-- Define the number of dice showing prime
def num_prime_dice : ℕ := 3

-- Theorem statement
theorem prob_three_primes_in_six_dice : 
  (Nat.choose num_dice num_prime_dice : ℚ) * 
  (prob_prime ^ num_prime_dice) * 
  (prob_non_prime ^ (num_dice - num_prime_dice)) = 857500 / 2985984 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_primes_in_six_dice_l2811_281191


namespace NUMINAMATH_CALUDE_parkway_elementary_soccer_l2811_281102

theorem parkway_elementary_soccer (total_students : ℕ) (boys : ℕ) (soccer_players : ℕ) 
  (h1 : total_students = 450)
  (h2 : boys = 320)
  (h3 : soccer_players = 250)
  (h4 : (86 : ℚ) / 100 * soccer_players = ↑(boys_playing_soccer))
  (boys_playing_soccer : ℕ) :
  total_students - boys - (soccer_players - boys_playing_soccer) = 95 :=
by sorry

end NUMINAMATH_CALUDE_parkway_elementary_soccer_l2811_281102


namespace NUMINAMATH_CALUDE_sum_of_base5_digits_2010_l2811_281198

/-- Converts a natural number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- Sums the digits in a list of natural numbers -/
def sumDigits (l : List ℕ) : ℕ :=
  l.foldl (·+·) 0

/-- Theorem stating that the sum of digits in the base-5 representation of 2010 equals 6 -/
theorem sum_of_base5_digits_2010 :
  sumDigits (toBase5 2010) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_base5_digits_2010_l2811_281198


namespace NUMINAMATH_CALUDE_winter_break_probability_l2811_281126

/-- The probability of getting exactly k successes in n independent trials,
    where each trial has probability p of success. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- The number of days in the winter break -/
def num_days : ℕ := 5

/-- The probability of clear weather on each day -/
def prob_clear : ℝ := 0.4

/-- The desired number of clear days -/
def desired_clear_days : ℕ := 2

theorem winter_break_probability :
  binomial_probability num_days desired_clear_days prob_clear = 216 / 625 := by
  sorry

end NUMINAMATH_CALUDE_winter_break_probability_l2811_281126


namespace NUMINAMATH_CALUDE_division_simplification_l2811_281199

theorem division_simplification (a b : ℝ) (h : a ≠ 0) :
  (-4 * a^2 + 12 * a^3 * b) / (-4 * a^2) = 1 - 3 * a * b := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l2811_281199


namespace NUMINAMATH_CALUDE_petes_number_l2811_281130

theorem petes_number : ∃ x : ℝ, 5 * (3 * x - 5) = 200 ∧ x = 15 := by sorry

end NUMINAMATH_CALUDE_petes_number_l2811_281130


namespace NUMINAMATH_CALUDE_jasmine_carry_weight_l2811_281129

/-- The weight of a bag of chips in ounces -/
def chipBagWeight : ℕ := 20

/-- The weight of a tin of cookies in ounces -/
def cookieTinWeight : ℕ := 9

/-- The number of bags of chips Jasmine buys -/
def numChipBags : ℕ := 6

/-- The ratio of tins of cookies to bags of chips Jasmine buys -/
def cookieToChipRatio : ℕ := 4

/-- The number of ounces in a pound -/
def ouncesPerPound : ℕ := 16

/-- Theorem: Given the conditions, Jasmine has to carry 21 pounds -/
theorem jasmine_carry_weight :
  (numChipBags * chipBagWeight +
   numChipBags * cookieToChipRatio * cookieTinWeight) / ouncesPerPound = 21 := by
  sorry

end NUMINAMATH_CALUDE_jasmine_carry_weight_l2811_281129


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l2811_281196

theorem shopkeeper_profit (discount : ℝ) (profit_with_discount : ℝ) :
  discount = 0.04 →
  profit_with_discount = 0.26 →
  let cost_price := 100
  let selling_price := cost_price * (1 + profit_with_discount)
  let marked_price := selling_price / (1 - discount)
  let profit_without_discount := (marked_price - cost_price) / cost_price
  profit_without_discount = 0.3125 := by sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l2811_281196


namespace NUMINAMATH_CALUDE_darry_climbed_152_steps_l2811_281111

/-- The number of steps Darry climbed today -/
def total_steps : ℕ :=
  let full_ladder_steps : ℕ := 11
  let full_ladder_climbs : ℕ := 10
  let small_ladder_steps : ℕ := 6
  let small_ladder_climbs : ℕ := 7
  full_ladder_steps * full_ladder_climbs + small_ladder_steps * small_ladder_climbs

theorem darry_climbed_152_steps : total_steps = 152 := by
  sorry

end NUMINAMATH_CALUDE_darry_climbed_152_steps_l2811_281111


namespace NUMINAMATH_CALUDE_amy_video_files_l2811_281106

/-- Proves that Amy had 21 video files initially -/
theorem amy_video_files :
  ∀ (initial_music_files deleted_files remaining_files : ℕ),
    initial_music_files = 4 →
    deleted_files = 23 →
    remaining_files = 2 →
    initial_music_files + (deleted_files + remaining_files) - initial_music_files = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_amy_video_files_l2811_281106


namespace NUMINAMATH_CALUDE_correct_sum_after_card_swap_l2811_281138

theorem correct_sum_after_card_swap : 
  ∃ (a b : ℕ), 
    (a + b = 81380) ∧ 
    (a ≠ 37541 ∨ b ≠ 43839) ∧
    (∃ (x y : ℕ), (x = 37541 ∧ y = 43839) ∧ (x + y = 80280)) :=
by sorry

end NUMINAMATH_CALUDE_correct_sum_after_card_swap_l2811_281138


namespace NUMINAMATH_CALUDE_equation_solution_l2811_281121

theorem equation_solution (x : ℚ) : 1 / (x + 1/5) = 5/3 → x = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2811_281121


namespace NUMINAMATH_CALUDE_scooter_price_l2811_281136

theorem scooter_price (upfront_payment : ℝ) (upfront_percentage : ℝ) (total_price : ℝ) : 
  upfront_payment = 240 → 
  upfront_percentage = 20 → 
  upfront_payment = (upfront_percentage / 100) * total_price → 
  total_price = 1200 := by
sorry

end NUMINAMATH_CALUDE_scooter_price_l2811_281136


namespace NUMINAMATH_CALUDE_motel_rent_problem_l2811_281119

/-- Represents the total rent charged by a motel on a given night -/
def TotalRent (r40 r60 : ℕ) : ℝ := 40 * r40 + 60 * r60

/-- The problem statement -/
theorem motel_rent_problem (r40 r60 : ℕ) :
  (∃ (total : ℝ), total = TotalRent r40 r60 ∧
    0.8 * total = TotalRent (r40 + 10) (r60 - 10)) →
  TotalRent r40 r60 = 1000 := by
  sorry

#check motel_rent_problem

end NUMINAMATH_CALUDE_motel_rent_problem_l2811_281119


namespace NUMINAMATH_CALUDE_initial_boys_on_slide_l2811_281143

theorem initial_boys_on_slide (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  additional = 13 → total = 35 → initial + additional = total → initial = 22 := by
  sorry

end NUMINAMATH_CALUDE_initial_boys_on_slide_l2811_281143


namespace NUMINAMATH_CALUDE_chromium_percentage_proof_l2811_281108

/-- The percentage of chromium in the first alloy -/
def chromium_percentage_1 : ℝ := 10

/-- The percentage of chromium in the second alloy -/
def chromium_percentage_2 : ℝ := 8

/-- The weight of the first alloy in kg -/
def weight_1 : ℝ := 15

/-- The weight of the second alloy in kg -/
def weight_2 : ℝ := 35

/-- The percentage of chromium in the new alloy -/
def chromium_percentage_new : ℝ := 8.6

/-- The total weight of the new alloy in kg -/
def total_weight : ℝ := weight_1 + weight_2

theorem chromium_percentage_proof :
  (chromium_percentage_1 / 100) * weight_1 + (chromium_percentage_2 / 100) * weight_2 =
  (chromium_percentage_new / 100) * total_weight :=
by sorry

end NUMINAMATH_CALUDE_chromium_percentage_proof_l2811_281108


namespace NUMINAMATH_CALUDE_exp_of_5_in_30_factorial_l2811_281165

/-- The exponent of 5 in the prime factorization of n! -/
def exp_of_5_in_factorial (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The exponent of 5 in the prime factorization of 30! is 7 -/
theorem exp_of_5_in_30_factorial :
  exp_of_5_in_factorial 30 = 7 := by
  sorry

end NUMINAMATH_CALUDE_exp_of_5_in_30_factorial_l2811_281165


namespace NUMINAMATH_CALUDE_max_volume_rotating_cube_max_volume_is_eight_l2811_281149

/-- The maximum volume of a cube that can rotate freely inside a cube with edge length 2 -/
theorem max_volume_rotating_cube (outer_edge : ℝ) (h : outer_edge = 2) :
  ∃ (inner_edge : ℝ),
    inner_edge > 0 ∧
    inner_edge * Real.sqrt 3 ≤ outer_edge * Real.sqrt 3 ∧
    ∀ (x : ℝ), x > 0 → x * Real.sqrt 3 ≤ outer_edge * Real.sqrt 3 → x^3 ≤ inner_edge^3 :=
by
  sorry

/-- The maximum volume of the rotating cube is 8 -/
theorem max_volume_is_eight (outer_edge : ℝ) (h : outer_edge = 2) :
  ∃ (inner_edge : ℝ),
    inner_edge > 0 ∧
    inner_edge * Real.sqrt 3 ≤ outer_edge * Real.sqrt 3 ∧
    inner_edge^3 = 8 ∧
    ∀ (x : ℝ), x > 0 → x * Real.sqrt 3 ≤ outer_edge * Real.sqrt 3 → x^3 ≤ 8 :=
by
  sorry

end NUMINAMATH_CALUDE_max_volume_rotating_cube_max_volume_is_eight_l2811_281149


namespace NUMINAMATH_CALUDE_raspberry_green_grape_difference_l2811_281192

def fruit_salad (green_grapes raspberries red_grapes : ℕ) : Prop :=
  green_grapes + raspberries + red_grapes = 102 ∧
  red_grapes = 67 ∧
  red_grapes = 3 * green_grapes + 7 ∧
  raspberries < green_grapes

theorem raspberry_green_grape_difference 
  (green_grapes raspberries red_grapes : ℕ) :
  fruit_salad green_grapes raspberries red_grapes →
  green_grapes - raspberries = 5 := by
sorry

end NUMINAMATH_CALUDE_raspberry_green_grape_difference_l2811_281192


namespace NUMINAMATH_CALUDE_train_cars_estimate_l2811_281153

/-- The number of cars Trey counted -/
def cars_counted : ℕ := 8

/-- The time (in seconds) Trey spent counting -/
def counting_time : ℕ := 15

/-- The total time (in seconds) the train took to pass -/
def total_time : ℕ := 210

/-- The estimated number of cars in the train -/
def estimated_cars : ℕ := 112

/-- Theorem stating that the estimated number of cars is approximately correct -/
theorem train_cars_estimate :
  abs ((cars_counted : ℚ) / counting_time * total_time - estimated_cars) < 1 := by
  sorry


end NUMINAMATH_CALUDE_train_cars_estimate_l2811_281153


namespace NUMINAMATH_CALUDE_product_expansion_sum_l2811_281181

theorem product_expansion_sum (a b c d : ℝ) :
  (∀ x, (2*x^2 - 3*x + 5)*(5 - x) = a*x^3 + b*x^2 + c*x + d) →
  a + b + c + d = 16 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l2811_281181


namespace NUMINAMATH_CALUDE_roberto_chicken_investment_break_even_l2811_281173

/-- Represents Roberto's chicken investment scenario -/
structure ChickenInvestment where
  num_chickens : ℕ
  cost_per_chicken : ℕ
  weekly_feed_cost : ℕ
  eggs_per_chicken_per_week : ℕ
  previous_dozen_cost : ℕ

/-- Calculates the break-even point in weeks for the chicken investment -/
def break_even_point (ci : ChickenInvestment) : ℕ :=
  let initial_cost := ci.num_chickens * ci.cost_per_chicken
  let weekly_egg_production := ci.num_chickens * ci.eggs_per_chicken_per_week
  let weekly_savings := ci.previous_dozen_cost - ci.weekly_feed_cost
  initial_cost / weekly_savings + 1

/-- Theorem stating that Roberto's chicken investment breaks even after 81 weeks -/
theorem roberto_chicken_investment_break_even :
  let ci : ChickenInvestment := {
    num_chickens := 4,
    cost_per_chicken := 20,
    weekly_feed_cost := 1,
    eggs_per_chicken_per_week := 3,
    previous_dozen_cost := 2
  }
  break_even_point ci = 81 := by sorry

end NUMINAMATH_CALUDE_roberto_chicken_investment_break_even_l2811_281173


namespace NUMINAMATH_CALUDE_arcsin_one_eq_pi_div_two_l2811_281161

-- Define arcsin function
noncomputable def arcsin (x : ℝ) : ℝ :=
  Real.arcsin x

-- State the theorem
theorem arcsin_one_eq_pi_div_two :
  arcsin 1 = π / 2 :=
sorry

end NUMINAMATH_CALUDE_arcsin_one_eq_pi_div_two_l2811_281161


namespace NUMINAMATH_CALUDE_exactly_100_valid_rules_l2811_281175

/-- A type representing a set of 100 cards drawn from an infinite deck of real numbers. -/
def CardSet := Fin 100 → ℝ

/-- A rule for determining the winner between two sets of cards. -/
def WinningRule := CardSet → CardSet → Bool

/-- The condition that the winner only depends on the relative order of the 200 cards. -/
def RelativeOrderCondition (rule : WinningRule) : Prop :=
  ∀ (A B : CardSet) (f : ℝ → ℝ), StrictMono f →
    rule A B = rule (f ∘ A) (f ∘ B)

/-- The condition that if a_i > b_i for all i, then A beats B. -/
def DominanceCondition (rule : WinningRule) : Prop :=
  ∀ (A B : CardSet), (∀ i, A i > B i) → rule A B

/-- The transitivity condition: if A beats B and B beats C, then A beats C. -/
def TransitivityCondition (rule : WinningRule) : Prop :=
  ∀ (A B C : CardSet), rule A B → rule B C → rule A C

/-- A valid rule satisfies all three conditions. -/
def ValidRule (rule : WinningRule) : Prop :=
  RelativeOrderCondition rule ∧ DominanceCondition rule ∧ TransitivityCondition rule

/-- The main theorem: there are exactly 100 valid rules. -/
theorem exactly_100_valid_rules :
  ∃! (rules : Finset WinningRule), rules.card = 100 ∧ ∀ rule ∈ rules, ValidRule rule :=
sorry

end NUMINAMATH_CALUDE_exactly_100_valid_rules_l2811_281175


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_zero_l2811_281109

-- Define the functions f and g
def f (A B x : ℝ) : ℝ := A * x^2 + B * x + 1
def g (A B x : ℝ) : ℝ := B * x^2 + A * x + 1

-- State the theorem
theorem sum_of_coefficients_is_zero (A B : ℝ) :
  A ≠ B →
  (∀ x, f A B (g A B x) - g A B (f A B x) = x^4 + 5*x^3 + x^2 - 4*x) →
  A + B = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_zero_l2811_281109


namespace NUMINAMATH_CALUDE_new_student_weight_l2811_281171

theorem new_student_weight (initial_count : ℕ) (initial_avg : ℝ) (new_count : ℕ) (new_avg : ℝ) : 
  initial_count = 19 →
  initial_avg = 15 →
  new_count = initial_count + 1 →
  new_avg = 14.9 →
  (initial_count : ℝ) * initial_avg + (new_count * new_avg - initial_count * initial_avg) = 13 := by
  sorry

end NUMINAMATH_CALUDE_new_student_weight_l2811_281171


namespace NUMINAMATH_CALUDE_lawn_mowing_payment_l2811_281177

theorem lawn_mowing_payment (rate : ℚ) (lawns_mowed : ℚ) : 
  rate = 15 / 4 → lawns_mowed = 5 / 2 → rate * lawns_mowed = 75 / 8 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_payment_l2811_281177


namespace NUMINAMATH_CALUDE_vector_magnitude_l2811_281166

def a : ℝ × ℝ := (1, 1)
def b : ℝ → ℝ × ℝ := λ y ↦ (3, y)

theorem vector_magnitude (y : ℝ) : 
  (∃ k : ℝ, b y - a = k • a) → ‖b y - a‖ = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l2811_281166


namespace NUMINAMATH_CALUDE_triangle_inequality_l2811_281156

/-- Given a non-isosceles triangle with sides a, b, c and area S,
    prove the inequality relating the sides and the area. -/
theorem triangle_inequality (a b c S : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- sides are positive
  a ≠ b ∧ b ≠ c ∧ a ≠ c →  -- non-isosceles condition
  S > 0 →  -- area is positive
  S = Real.sqrt (((a + b + c) / 2) * (((a + b + c) / 2) - a) * 
    (((a + b + c) / 2) - b) * (((a + b + c) / 2) - c)) →  -- Heron's formula
  (a^3 / ((a-b)*(a-c))) + (b^3 / ((b-c)*(b-a))) + 
    (c^3 / ((c-a)*(c-b))) > 2 * 3^(3/4) * S^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2811_281156


namespace NUMINAMATH_CALUDE_balloon_height_calculation_l2811_281157

theorem balloon_height_calculation (initial_budget : ℚ) (sheet_cost : ℚ) (rope_cost : ℚ) (propane_cost : ℚ) (helium_price_per_oz : ℚ) (height_per_oz : ℚ) : 
  initial_budget = 200 →
  sheet_cost = 42 →
  rope_cost = 18 →
  propane_cost = 14 →
  helium_price_per_oz = 3/2 →
  height_per_oz = 113 →
  ((initial_budget - sheet_cost - rope_cost - propane_cost) / helium_price_per_oz) * height_per_oz = 9492 :=
by sorry

end NUMINAMATH_CALUDE_balloon_height_calculation_l2811_281157


namespace NUMINAMATH_CALUDE_unique_valid_swap_l2811_281182

/-- Represents a time between 6 and 7 o'clock -/
structure Time6To7 where
  hour : ℝ
  minute : ℝ
  h_range : 6 < hour ∧ hour < 7
  m_range : 0 ≤ minute ∧ minute < 60

/-- Checks if swapping hour and minute hands results in a valid time -/
def is_valid_swap (t : Time6To7) : Prop :=
  ∃ (t' : Time6To7), t.hour = t'.minute / 5 ∧ t.minute = t'.hour * 5

/-- The main theorem stating there's exactly one time where swapping hands is valid -/
theorem unique_valid_swap : ∃! (t : Time6To7), is_valid_swap t :=
sorry

end NUMINAMATH_CALUDE_unique_valid_swap_l2811_281182


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2811_281128

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_product (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 4 * a 5 * a 6 = 27 →
  a 1 * a 9 = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2811_281128


namespace NUMINAMATH_CALUDE_consecutive_cubes_divisibility_l2811_281160

theorem consecutive_cubes_divisibility (a : ℤ) : 
  ∃ (k₁ k₂ : ℤ), 3 * a * (a^2 + 2) = 3 * a * k₁ ∧ 3 * a * (a^2 + 2) = 9 * k₂ := by
  sorry

end NUMINAMATH_CALUDE_consecutive_cubes_divisibility_l2811_281160


namespace NUMINAMATH_CALUDE_coconut_oil_needed_l2811_281190

/-- Calculates the amount of coconut oil needed for baking brownies --/
theorem coconut_oil_needed
  (butter_per_cup : ℝ)
  (coconut_oil_per_cup : ℝ)
  (butter_available : ℝ)
  (total_baking_mix : ℝ)
  (h1 : butter_per_cup = 2)
  (h2 : coconut_oil_per_cup = 2)
  (h3 : butter_available = 4)
  (h4 : total_baking_mix = 6) :
  (total_baking_mix - butter_available / butter_per_cup) * coconut_oil_per_cup = 8 :=
by sorry

end NUMINAMATH_CALUDE_coconut_oil_needed_l2811_281190


namespace NUMINAMATH_CALUDE_olivias_papers_l2811_281185

/-- Given an initial number of papers and a number of papers used,
    calculate the remaining number of papers. -/
def remaining_papers (initial : ℕ) (used : ℕ) : ℕ :=
  initial - used

/-- Theorem stating that given 81 initial papers and 56 used papers,
    the remaining number is 25. -/
theorem olivias_papers :
  remaining_papers 81 56 = 25 := by
  sorry

end NUMINAMATH_CALUDE_olivias_papers_l2811_281185


namespace NUMINAMATH_CALUDE_no_real_roots_when_m_is_one_m_range_for_specified_root_intervals_l2811_281142

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*m*x + 2*m + 1

-- Theorem 1: When m = 1, the equation has no real roots
theorem no_real_roots_when_m_is_one :
  ∀ x : ℝ, f 1 x ≠ 0 := by sorry

-- Theorem 2: Range of m when roots are in specified intervals
theorem m_range_for_specified_root_intervals :
  (∃ x y : ℝ, x ∈ Set.Ioo (-1) 0 ∧ y ∈ Set.Ioo 1 2 ∧ f m x = 0 ∧ f m y = 0) ↔
  m ∈ Set.Ioo (-5/6) (-1/2) := by sorry

end NUMINAMATH_CALUDE_no_real_roots_when_m_is_one_m_range_for_specified_root_intervals_l2811_281142


namespace NUMINAMATH_CALUDE_womens_doubles_handshakes_l2811_281155

/-- The number of handshakes in a women's doubles tennis tournament --/
theorem womens_doubles_handshakes :
  let num_teams : ℕ := 4
  let team_size : ℕ := 2
  let total_players : ℕ := num_teams * team_size
  let handshakes_per_player : ℕ := total_players - team_size
  total_players * handshakes_per_player / 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_womens_doubles_handshakes_l2811_281155


namespace NUMINAMATH_CALUDE_max_ab_max_expression_min_sum_l2811_281195

-- Define the conditions
def is_valid_pair (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = 1

-- Theorem 1: Maximum value of ab
theorem max_ab (a b : ℝ) (h : is_valid_pair a b) :
  a * b ≤ 1/4 ∧ ∃ (a₀ b₀ : ℝ), is_valid_pair a₀ b₀ ∧ a₀ * b₀ = 1/4 :=
sorry

-- Theorem 2: Maximum value of 4a - 1/(4b)
theorem max_expression (a b : ℝ) (h : is_valid_pair a b) :
  4*a - 1/(4*b) ≤ 2 ∧ ∃ (a₀ b₀ : ℝ), is_valid_pair a₀ b₀ ∧ 4*a₀ - 1/(4*b₀) = 2 :=
sorry

-- Theorem 3: Minimum value of 1/a + 2/b
theorem min_sum (a b : ℝ) (h : is_valid_pair a b) :
  1/a + 2/b ≥ 3 + 2*Real.sqrt 2 ∧ ∃ (a₀ b₀ : ℝ), is_valid_pair a₀ b₀ ∧ 1/a₀ + 2/b₀ = 3 + 2*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_ab_max_expression_min_sum_l2811_281195


namespace NUMINAMATH_CALUDE_parabola_directrix_l2811_281103

/-- 
Given a parabola y² = 2px with intersection point (4, 0), 
prove that its directrix has the equation x = -4 
-/
theorem parabola_directrix (p : ℝ) : 
  (∀ x y : ℝ, y^2 = 2*p*x) →  -- Equation of the parabola
  (0^2 = 2*p*4) →            -- Intersection point (4, 0)
  (x = -4) →                 -- Equation of the directrix
  True := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2811_281103


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l2811_281123

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (m n : Line) (β : Plane) :
  parallel m n → 
  perpendicular_line_plane n β → 
  perpendicular_line_plane m β :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l2811_281123


namespace NUMINAMATH_CALUDE_coin_sequence_count_l2811_281120

/-- Represents a coin toss sequence -/
def CoinSequence := List Bool

/-- Counts the number of specific subsequences in a coin sequence -/
def countSubsequences (seq : CoinSequence) : Nat × Nat × Nat × Nat :=
  sorry

/-- Checks if a coin sequence has the required number of subsequences -/
def hasRequiredSubsequences (seq : CoinSequence) : Bool :=
  let (hh, ht, th, tt) := countSubsequences seq
  hh = 3 ∧ ht = 2 ∧ th = 5 ∧ tt = 6

/-- Generates all possible 17-toss coin sequences -/
def allSequences : List CoinSequence :=
  sorry

/-- Counts the number of sequences with required subsequences -/
def countValidSequences : Nat :=
  (allSequences.filter hasRequiredSubsequences).length

theorem coin_sequence_count : countValidSequences = 840 := by
  sorry

end NUMINAMATH_CALUDE_coin_sequence_count_l2811_281120


namespace NUMINAMATH_CALUDE_burn_all_bridges_probability_l2811_281186

/-- The number of islands in the lake -/
def num_islands : ℕ := 2013

/-- The probability of choosing a new bridge at each step -/
def prob_new_bridge : ℚ := 2/3

/-- The probability of burning all bridges -/
def prob_burn_all : ℚ := num_islands * prob_new_bridge ^ (num_islands - 1)

/-- Theorem stating the probability of burning all bridges -/
theorem burn_all_bridges_probability :
  prob_burn_all = num_islands * (2/3) ^ (num_islands - 1) := by sorry

end NUMINAMATH_CALUDE_burn_all_bridges_probability_l2811_281186


namespace NUMINAMATH_CALUDE_correct_average_l2811_281125

theorem correct_average (n : ℕ) (initial_avg incorrect_num correct_num : ℚ) : 
  n = 10 → 
  initial_avg = 16 → 
  incorrect_num = 26 → 
  correct_num = 46 → 
  (n * initial_avg - incorrect_num + correct_num) / n = 18 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l2811_281125


namespace NUMINAMATH_CALUDE_cost_of_480_chocolates_l2811_281172

/-- The cost of buying a given number of chocolates, given the box size and box cost -/
def chocolate_cost (total_chocolates : ℕ) (box_size : ℕ) (box_cost : ℕ) : ℕ :=
  (total_chocolates / box_size) * box_cost

/-- Theorem: The cost of 480 chocolates is $96, given that a box of 40 chocolates costs $8 -/
theorem cost_of_480_chocolates :
  chocolate_cost 480 40 8 = 96 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_480_chocolates_l2811_281172


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2811_281113

open Set

-- Define the sets A and B
def A : Set ℝ := {x | x < 5}
def B : Set ℝ := {x | x < 2}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (Bᶜ) = {x : ℝ | 2 ≤ x ∧ x < 5} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2811_281113


namespace NUMINAMATH_CALUDE_smallest_possible_a_l2811_281167

theorem smallest_possible_a (a b c : ℝ) : 
  (∃ (x y : ℝ), y = a * (x - 1/3)^2 - 1/4) →  -- parabola with vertex (1/3, -1/4)
  (∃ (x y : ℝ), y = a * x^2 + b * x + c) →    -- equation of parabola
  (a > 0) →                                   -- a is positive
  (∃ (n : ℤ), 2 * a + b + 3 * c = n) →        -- 2a + b + 3c is an integer
  (∀ (a' : ℝ), a' ≥ 9/16 ∨ ¬(
    (∃ (x y : ℝ), y = a' * (x - 1/3)^2 - 1/4) ∧
    (∃ (x y : ℝ), y = a' * x^2 + b * x + c) ∧
    (a' > 0) ∧
    (∃ (n : ℤ), 2 * a' + b + 3 * c = n)
  )) :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_a_l2811_281167


namespace NUMINAMATH_CALUDE_pie_eating_contest_l2811_281179

theorem pie_eating_contest (student1 student2 student3 : ℚ) 
  (h1 : student1 = 5/6)
  (h2 : student2 = 7/8)
  (h3 : student3 = 2/3) :
  max student1 (max student2 student3) - min student1 (min student2 student3) = 5/24 := by
sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l2811_281179


namespace NUMINAMATH_CALUDE_smallest_checkered_rectangle_l2811_281176

/-- A rectangle that can be divided into both 1 × 13 rectangles and three-cell corners -/
structure CheckeredRectangle where
  width : ℕ
  height : ℕ
  dividable_13 : width * height % 13 = 0
  dividable_3 : width ≥ 2 ∧ height ≥ 2

/-- The area of a CheckeredRectangle -/
def area (r : CheckeredRectangle) : ℕ := r.width * r.height

/-- The perimeter of a CheckeredRectangle -/
def perimeter (r : CheckeredRectangle) : ℕ := 2 * (r.width + r.height)

/-- The set of all valid CheckeredRectangles -/
def valid_rectangles : Set CheckeredRectangle :=
  {r : CheckeredRectangle | true}

theorem smallest_checkered_rectangle :
  ∃ (r : CheckeredRectangle),
    r ∈ valid_rectangles ∧
    area r = 78 ∧
    (∀ (s : CheckeredRectangle), s ∈ valid_rectangles → area s ≥ area r) ∧
    (∃ (p : List ℕ), p = [38, 58, 82] ∧ (perimeter r) ∈ p) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_checkered_rectangle_l2811_281176


namespace NUMINAMATH_CALUDE_quadratic_polynomials_inequalities_l2811_281132

/-- Given three quadratic polynomials with the specified properties, 
    exactly two out of three inequalities are satisfied. -/
theorem quadratic_polynomials_inequalities 
  (a b c d e f : ℝ) 
  (h1 : ∃ x : ℝ, (x^2 + a*x + b = 0 ∧ x^2 + c*x + d = 0) ∨ 
                 (x^2 + a*x + b = 0 ∧ x^2 + e*x + f = 0) ∨ 
                 (x^2 + c*x + d = 0 ∧ x^2 + e*x + f = 0))
  (h2 : ¬ ∃ x : ℝ, x^2 + a*x + b = 0 ∧ x^2 + c*x + d = 0 ∧ x^2 + e*x + f = 0) :
  (((a^2 + c^2 - e^2)/4 > b + d - f) ∧ 
   ((c^2 + e^2 - a^2)/4 > d + f - b) ∧ 
   ((e^2 + a^2 - c^2)/4 ≤ f + b - d)) ∨
  (((a^2 + c^2 - e^2)/4 > b + d - f) ∧ 
   ((c^2 + e^2 - a^2)/4 ≤ d + f - b) ∧ 
   ((e^2 + a^2 - c^2)/4 > f + b - d)) ∨
  (((a^2 + c^2 - e^2)/4 ≤ b + d - f) ∧ 
   ((c^2 + e^2 - a^2)/4 > d + f - b) ∧ 
   ((e^2 + a^2 - c^2)/4 > f + b - d)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomials_inequalities_l2811_281132


namespace NUMINAMATH_CALUDE_prop_logic_l2811_281145

theorem prop_logic (p q : Prop) (h1 : ¬p) (h2 : p ∨ q) : ¬p ∧ q := by
  sorry

end NUMINAMATH_CALUDE_prop_logic_l2811_281145


namespace NUMINAMATH_CALUDE_a_100_eq_344934_l2811_281147

/-- Sequence defined by a(n) = a(n-1) + n^2 for n ≥ 1, with a(0) = 2009 -/
def a : ℕ → ℕ
  | 0 => 2009
  | n + 1 => a n + (n + 1)^2

/-- The 100th term of the sequence a is 344934 -/
theorem a_100_eq_344934 : a 100 = 344934 := by
  sorry

end NUMINAMATH_CALUDE_a_100_eq_344934_l2811_281147


namespace NUMINAMATH_CALUDE_solve_equation_l2811_281194

theorem solve_equation (x : ℝ) : 
  5 * x^(1/3) - 3 * (x / x^(2/3)) = 9 + x^(1/3) ↔ x = 729 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2811_281194


namespace NUMINAMATH_CALUDE_f_derivative_l2811_281133

noncomputable def f (x : ℝ) : ℝ := -2 * Real.exp x * Real.sin x

theorem f_derivative (x : ℝ) : 
  deriv f x = -2 * Real.exp x * (Real.sin x + Real.cos x) := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_l2811_281133


namespace NUMINAMATH_CALUDE_not_perfect_square_p_squared_plus_q_power_l2811_281114

theorem not_perfect_square_p_squared_plus_q_power (p q : ℕ) (hp : Prime p) (hq : Prime q) (h_perfect_square : ∃ a : ℕ, p + q^2 = a^2) :
  ∀ n : ℕ, ¬∃ b : ℕ, p^2 + q^n = b^2 :=
by sorry

end NUMINAMATH_CALUDE_not_perfect_square_p_squared_plus_q_power_l2811_281114


namespace NUMINAMATH_CALUDE_parabola_vertex_and_focus_l2811_281159

/-- A parabola is defined by the equation x = (1/8) * y^2 -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = (1/8) * p.2^2}

/-- The vertex of a parabola is the point where it turns -/
def Vertex (P : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- The focus of a parabola is a fixed point used in its geometric definition -/
def Focus (P : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

theorem parabola_vertex_and_focus :
  Vertex Parabola = (0, 0) ∧ Focus Parabola = (1/2, 0) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_and_focus_l2811_281159


namespace NUMINAMATH_CALUDE_kyle_money_after_snowboarding_l2811_281189

theorem kyle_money_after_snowboarding (dave_money : ℕ) (kyle_initial_money : ℕ) 
  (h1 : dave_money = 46) 
  (h2 : kyle_initial_money = 3 * dave_money - 12) 
  (h3 : kyle_initial_money ≥ 12) : 
  kyle_initial_money - (kyle_initial_money / 3) = 84 := by
  sorry

end NUMINAMATH_CALUDE_kyle_money_after_snowboarding_l2811_281189


namespace NUMINAMATH_CALUDE_line_intersection_x_axis_l2811_281174

/-- Given a line y = ax + b passing through points (0, 2) and (-3, 0),
    prove that the solution to ax + b = 0 is x = -3. -/
theorem line_intersection_x_axis 
  (a b : ℝ) 
  (h1 : 2 = a * 0 + b) 
  (h2 : 0 = a * (-3) + b) : 
  ∀ x, a * x + b = 0 ↔ x = -3 :=
sorry

end NUMINAMATH_CALUDE_line_intersection_x_axis_l2811_281174


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_l2811_281170

theorem sum_of_squares_zero (a b c : ℝ) :
  (a - 6)^2 + (b - 3)^2 + (c - 2)^2 = 0 → a + b + c = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_l2811_281170


namespace NUMINAMATH_CALUDE_octopus_leg_solution_l2811_281117

-- Define the possible number of legs for an octopus
inductive LegCount : Type
  | six : LegCount
  | seven : LegCount
  | eight : LegCount

-- Define the colors of the octopuses
inductive OctopusColor : Type
  | blue : OctopusColor
  | green : OctopusColor
  | yellow : OctopusColor
  | red : OctopusColor

-- Define a function to determine if an octopus is truthful based on its leg count
def isTruthful (legs : LegCount) : Prop :=
  match legs with
  | LegCount.six => True
  | LegCount.seven => False
  | LegCount.eight => True

-- Define a function to convert LegCount to a natural number
def legCountToNat (legs : LegCount) : ℕ :=
  match legs with
  | LegCount.six => 6
  | LegCount.seven => 7
  | LegCount.eight => 8

-- Define the claims made by each octopus
def claim (color : OctopusColor) : ℕ :=
  match color with
  | OctopusColor.blue => 28
  | OctopusColor.green => 27
  | OctopusColor.yellow => 26
  | OctopusColor.red => 25

-- Define the theorem
theorem octopus_leg_solution :
  ∃ (legs : OctopusColor → LegCount),
    (legs OctopusColor.green = LegCount.six) ∧
    (legs OctopusColor.blue = LegCount.seven) ∧
    (legs OctopusColor.yellow = LegCount.seven) ∧
    (legs OctopusColor.red = LegCount.seven) ∧
    (∀ (c : OctopusColor), isTruthful (legs c) ↔ (claim c = legCountToNat (legs OctopusColor.blue) + legCountToNat (legs OctopusColor.green) + legCountToNat (legs OctopusColor.yellow) + legCountToNat (legs OctopusColor.red))) :=
  sorry

end NUMINAMATH_CALUDE_octopus_leg_solution_l2811_281117


namespace NUMINAMATH_CALUDE_same_day_of_week_l2811_281146

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Given a year and a day number, returns the day of the week -/
def dayOfWeek (year : Nat) (dayNumber : Nat) : DayOfWeek := sorry

theorem same_day_of_week (year : Nat) :
  dayOfWeek year 15 = DayOfWeek.Monday →
  dayOfWeek year 197 = DayOfWeek.Monday :=
by
  sorry

end NUMINAMATH_CALUDE_same_day_of_week_l2811_281146


namespace NUMINAMATH_CALUDE_moores_law_1985_to_1995_l2811_281139

/-- Moore's law doubling period in years -/
def moore_period : ℕ := 2

/-- Initial year for transistor count -/
def initial_year : ℕ := 1985

/-- Final year for transistor count -/
def final_year : ℕ := 1995

/-- Initial transistor count in 1985 -/
def initial_transistors : ℕ := 500000

/-- Calculate the number of transistors according to Moore's law -/
def transistor_count (start_year end_year start_count : ℕ) : ℕ :=
  start_count * 2 ^ ((end_year - start_year) / moore_period)

/-- Theorem stating that the transistor count in 1995 is 16,000,000 -/
theorem moores_law_1985_to_1995 :
  transistor_count initial_year final_year initial_transistors = 16000000 := by
  sorry

end NUMINAMATH_CALUDE_moores_law_1985_to_1995_l2811_281139


namespace NUMINAMATH_CALUDE_derivative_cos_ln_l2811_281141

open Real

theorem derivative_cos_ln (x : ℝ) (h : x > 0) :
  deriv (λ x => cos (log x)) x = -1/x * sin (log x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_cos_ln_l2811_281141


namespace NUMINAMATH_CALUDE_one_third_of_recipe_l2811_281163

theorem one_third_of_recipe (full_recipe : ℚ) (one_third_recipe : ℚ) : 
  full_recipe = 17 / 3 ∧ one_third_recipe = full_recipe / 3 → one_third_recipe = 17 / 9 := by
  sorry

#check one_third_of_recipe

end NUMINAMATH_CALUDE_one_third_of_recipe_l2811_281163


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_2016_n_193_divisible_by_2016_smallest_n_is_193_l2811_281118

theorem smallest_n_divisible_by_2016 :
  ∀ n : ℕ, n > 1 → (3 * n^3 + 2013) % 2016 = 0 → n ≥ 193 :=
by sorry

theorem n_193_divisible_by_2016 :
  (3 * 193^3 + 2013) % 2016 = 0 :=
by sorry

theorem smallest_n_is_193 :
  ∃! n : ℕ, n > 1 ∧ (3 * n^3 + 2013) % 2016 = 0 ∧
  ∀ m : ℕ, m > 1 → (3 * m^3 + 2013) % 2016 = 0 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_2016_n_193_divisible_by_2016_smallest_n_is_193_l2811_281118


namespace NUMINAMATH_CALUDE_field_width_l2811_281110

/-- A rectangular field with length 7/5 of its width and perimeter 336 meters has a width of 70 meters -/
theorem field_width (w : ℝ) (h1 : w > 0) : 
  2 * (7/5 * w + w) = 336 → w = 70 := by
  sorry

end NUMINAMATH_CALUDE_field_width_l2811_281110


namespace NUMINAMATH_CALUDE_chocolate_bar_expense_l2811_281150

def chocolate_bar_cost : ℚ := 3/2  -- $1.50 represented as a rational number
def smores_per_bar : ℕ := 3
def num_scouts : ℕ := 15
def smores_per_scout : ℕ := 2

theorem chocolate_bar_expense : 
  ↑num_scouts * ↑smores_per_scout / ↑smores_per_bar * chocolate_bar_cost = 15 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_expense_l2811_281150


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l2811_281107

/-- The general form equation of a line passing through (1, 1) with slope -3 -/
theorem line_equation_through_point_with_slope :
  ∃ (A B C : ℝ), A ≠ 0 ∨ B ≠ 0 ∧
  (∀ x y : ℝ, A * x + B * y + C = 0 ↔ y - 1 = -3 * (x - 1)) ∧
  A = 3 ∧ B = 1 ∧ C = -4 := by
sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l2811_281107


namespace NUMINAMATH_CALUDE_min_squares_partition_l2811_281131

/-- Represents a square with an integer side length -/
structure Square where
  side : ℕ

/-- Represents a partition of a square into smaller squares -/
structure Partition where
  squares : List Square

/-- Check if a partition is valid for an 11x11 square -/
def isValidPartition (p : Partition) : Prop :=
  (p.squares.map (λ s => s.side * s.side)).sum = 11 * 11 ∧
  p.squares.all (λ s => s.side > 0 ∧ s.side < 11)

/-- The theorem stating the minimum number of squares in a valid partition -/
theorem min_squares_partition :
  ∃ (p : Partition), isValidPartition p ∧ p.squares.length = 11 ∧
  ∀ (q : Partition), isValidPartition q → p.squares.length ≤ q.squares.length :=
sorry

end NUMINAMATH_CALUDE_min_squares_partition_l2811_281131


namespace NUMINAMATH_CALUDE_total_scholarship_amount_l2811_281183

-- Define the scholarship amounts
def wendy_scholarship : ℕ := 20000
def kelly_scholarship : ℕ := 2 * wendy_scholarship
def nina_scholarship : ℕ := kelly_scholarship - 8000

-- Theorem statement
theorem total_scholarship_amount :
  wendy_scholarship + kelly_scholarship + nina_scholarship = 92000 := by
  sorry

end NUMINAMATH_CALUDE_total_scholarship_amount_l2811_281183


namespace NUMINAMATH_CALUDE_profit_percentage_doubling_l2811_281180

theorem profit_percentage_doubling (cost_price : ℝ) (original_profit_percentage : ℝ) 
  (h1 : original_profit_percentage = 60) :
  let original_selling_price := cost_price * (1 + original_profit_percentage / 100)
  let new_selling_price := 2 * original_selling_price
  let new_profit := new_selling_price - cost_price
  let new_profit_percentage := (new_profit / cost_price) * 100
  new_profit_percentage = 220 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_doubling_l2811_281180


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2811_281162

/-- Given a geometric sequence {a_n} with common ratio q = 2 and S_n being the sum of the first n terms, 
    prove that S_4 / a_2 = -15/2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = 2 * a n) →  -- Common ratio q = 2
  (∀ n, S n = (a 1) * (1 - 2^n) / (1 - 2)) →  -- Sum formula for geometric sequence
  S 4 / a 2 = -15/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2811_281162


namespace NUMINAMATH_CALUDE_range_of_m_for_inequality_l2811_281164

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2|

-- State the theorem
theorem range_of_m_for_inequality (h : ∀ x ≤ 5, f x ≤ 3) :
  {m : ℝ | ∀ x, f x + (x + 5) ≥ m} = Set.Iic 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_for_inequality_l2811_281164


namespace NUMINAMATH_CALUDE_four_valid_start_days_l2811_281127

/-- Represents the days of the week -/
inductive Weekday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Counts the number of occurrences of a specific weekday in a 30-day month starting from a given day -/
def countWeekday (start : Weekday) (target : Weekday) : Nat :=
  sorry

/-- Checks if Tuesdays and Fridays are equal in number for a given starting day -/
def hasSameTuesdaysAndFridays (start : Weekday) : Bool :=
  countWeekday start Weekday.Tuesday = countWeekday start Weekday.Friday

/-- The set of all weekdays -/
def allWeekdays : List Weekday :=
  [Weekday.Monday, Weekday.Tuesday, Weekday.Wednesday, Weekday.Thursday, 
   Weekday.Friday, Weekday.Saturday, Weekday.Sunday]

/-- The main theorem stating that exactly 4 weekdays satisfy the condition -/
theorem four_valid_start_days :
  (allWeekdays.filter hasSameTuesdaysAndFridays).length = 4 :=
  sorry

end NUMINAMATH_CALUDE_four_valid_start_days_l2811_281127


namespace NUMINAMATH_CALUDE_alex_class_size_l2811_281124

/-- In a class, given a student who is both the 30th best and 30th worst, 
    the total number of students in the class is 59. -/
theorem alex_class_size (n : ℕ) 
  (h1 : ∃ (alex : ℕ), alex ≤ n ∧ alex = 30)  -- Alex is 30th best
  (h2 : ∃ (alex : ℕ), alex ≤ n ∧ alex = 30)  -- Alex is 30th worst
  : n = 59 := by
  sorry

end NUMINAMATH_CALUDE_alex_class_size_l2811_281124


namespace NUMINAMATH_CALUDE_cube_difference_l2811_281104

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 53) : a^3 - b^3 = 385 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_l2811_281104


namespace NUMINAMATH_CALUDE_assembly_line_arrangements_l2811_281151

def num_tasks : ℕ := 5

theorem assembly_line_arrangements :
  (Finset.range num_tasks).card.factorial = 120 := by
  sorry

end NUMINAMATH_CALUDE_assembly_line_arrangements_l2811_281151


namespace NUMINAMATH_CALUDE_plane_division_l2811_281178

/-- Given m parallel lines and n non-parallel lines on a plane,
    where no more than two lines pass through any single point,
    the number of regions into which these lines divide the plane
    is 1 + (n(n+1))/2 + m(n+1). -/
theorem plane_division (m n : ℕ) : ℕ := by
  sorry

#check plane_division

end NUMINAMATH_CALUDE_plane_division_l2811_281178


namespace NUMINAMATH_CALUDE_marker_distance_l2811_281116

theorem marker_distance (k : ℝ) (h1 : k > 0) 
  (h2 : ∀ n : ℕ+, Real.sqrt ((4:ℝ)^2 + (4*k)^2) = 31) : 
  Real.sqrt ((12:ℝ)^2 + (12*k)^2) = 93 := by sorry

end NUMINAMATH_CALUDE_marker_distance_l2811_281116


namespace NUMINAMATH_CALUDE_equal_ratios_fraction_l2811_281105

theorem equal_ratios_fraction (x y z : ℝ) (h : x/2 = y/3 ∧ y/3 = z/4) :
  (x + y) / (3*y - 2*z) = 5 := by
  sorry

end NUMINAMATH_CALUDE_equal_ratios_fraction_l2811_281105


namespace NUMINAMATH_CALUDE_exterior_angle_regular_octagon_exterior_angle_regular_octagon_is_45_l2811_281135

/-- The measure of an exterior angle in a regular octagon is 45 degrees. -/
theorem exterior_angle_regular_octagon : ℝ :=
  let n : ℕ := 8  -- number of sides in an octagon
  let interior_angle_sum : ℝ := (n - 2) * 180
  let interior_angle : ℝ := interior_angle_sum / n
  let exterior_angle : ℝ := 180 - interior_angle
  exterior_angle

/-- The measure of an exterior angle in a regular octagon is 45 degrees. -/
theorem exterior_angle_regular_octagon_is_45 : exterior_angle_regular_octagon = 45 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_regular_octagon_exterior_angle_regular_octagon_is_45_l2811_281135


namespace NUMINAMATH_CALUDE_spring_work_l2811_281169

/-- Work done to stretch a spring -/
theorem spring_work (force : Real) (compression : Real) (stretch : Real) : 
  force = 10 →
  compression = 0.1 →
  stretch = 0.06 →
  (1/2) * (force / compression) * stretch^2 = 0.18 := by
  sorry

end NUMINAMATH_CALUDE_spring_work_l2811_281169


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l2811_281184

/-- Represents the position function of a particle -/
def S (t : ℝ) : ℝ := 2 * t^3

/-- Represents the velocity function of a particle -/
def V (t : ℝ) : ℝ := 6 * t^2

theorem instantaneous_velocity_at_3 :
  V 3 = 54 :=
sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l2811_281184


namespace NUMINAMATH_CALUDE_expression_simplification_l2811_281115

theorem expression_simplification :
  ∀ q : ℚ, ((7*q+3)-3*q*2)*4+(5-2/4)*(8*q-12) = 40*q - 42 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2811_281115


namespace NUMINAMATH_CALUDE_zoo_trip_students_l2811_281100

theorem zoo_trip_students (buses : Nat) (students_per_bus : Nat) (car_students : Nat) :
  buses = 7 →
  students_per_bus = 53 →
  car_students = 4 →
  buses * students_per_bus + car_students = 375 := by
  sorry

end NUMINAMATH_CALUDE_zoo_trip_students_l2811_281100


namespace NUMINAMATH_CALUDE_unique_triple_l2811_281187

theorem unique_triple : ∃! (a b c : ℤ), 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  a + b = c ∧ 
  b * c = a ∧ 
  a = -4 ∧ b = 2 ∧ c = -2 := by sorry

end NUMINAMATH_CALUDE_unique_triple_l2811_281187


namespace NUMINAMATH_CALUDE_a_sum_cube_minus_product_l2811_281188

noncomputable def a (i : ℕ) (x : ℝ) : ℝ := ∑' n, (x ^ (3 * n + i)) / (Nat.factorial (3 * n + i))

theorem a_sum_cube_minus_product (x : ℝ) :
  (a 0 x) ^ 3 + (a 1 x) ^ 3 + (a 2 x) ^ 3 - 3 * (a 0 x) * (a 1 x) * (a 2 x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_a_sum_cube_minus_product_l2811_281188


namespace NUMINAMATH_CALUDE_average_of_specific_odds_l2811_281154

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_in_range (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 10

def is_less_than_6 (n : ℕ) : Prop := n < 6

def meets_conditions (n : ℕ) : Prop :=
  is_odd n ∧ is_in_range n ∧ is_less_than_6 n

def numbers_meeting_conditions : List ℕ :=
  [1, 3, 5]

theorem average_of_specific_odds :
  (numbers_meeting_conditions.sum : ℚ) / numbers_meeting_conditions.length = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_specific_odds_l2811_281154


namespace NUMINAMATH_CALUDE_jerrys_age_l2811_281197

theorem jerrys_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 2 * jerry_age - 4 →
  mickey_age = 22 →
  jerry_age = 13 := by
sorry

end NUMINAMATH_CALUDE_jerrys_age_l2811_281197


namespace NUMINAMATH_CALUDE_division_problem_l2811_281152

theorem division_problem (divisor quotient remainder : ℕ) (h1 : divisor = 21) (h2 : quotient = 8) (h3 : remainder = 3) :
  divisor * quotient + remainder = 171 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2811_281152


namespace NUMINAMATH_CALUDE_sum_of_squares_positive_and_negative_l2811_281122

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_squares_positive_and_negative :
  2 * (sum_of_squares 50) = 85850 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_positive_and_negative_l2811_281122


namespace NUMINAMATH_CALUDE_remainder_theorem_l2811_281168

theorem remainder_theorem (r : ℤ) : (r^11 - 3) % (r - 2) = 2045 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2811_281168
